use crate::simulation::{D_MAX, D_MIN, S_MAX, S_MIN};

const MIN_PROB: f32 = 1e-7;
const MAX_PROB: f32 = 1.0 - 1e-7;

#[derive(Clone, Copy, Debug)]
struct CurveCache {
    t: f32,
    s: f32,
    decay: f32,
    factor: f32,
    base: f32,
    retrievability: f32,
}

#[derive(Clone, Copy, Debug)]
struct StepCache {
    state_s: f32,
    state_d: f32,
    last_s: f32,
    last_d: f32,
    delta_t: f32,
    rating: f32,
    retrievability: f32,
    failure_raw: f32,
    failure_floor: f32,
    failure_used_floor: bool,
    short_raw: f32,
    short_value: f32,
    short_raw_active: bool,
    use_short: bool,
    use_failure: bool,
    init_selected: bool,
    padding: bool,
    pre_clamp_s: f32,
    mean_pre_clamp_d: f32,
    init_rating: usize,
}

#[derive(Clone, Copy, Debug)]
struct State {
    s: f32,
    d: f32,
}

#[derive(Clone, Copy)]
struct RuntimeParams<'a> {
    w: &'a [f32],
    curve_decay: f32,
    curve_factor: f32,
    curve_dfactor_ddecay: f64,
    exp_w8: f32,
    exp_w8_f64: f64,
    failure_floor_divisor: f32,
    easy_d: f32,
    easy_d_f64: f64,
    exp_3w5_f64: f64,
}

impl<'a> RuntimeParams<'a> {
    #[inline]
    fn new(w: &'a [f32]) -> Self {
        let curve_decay = -w[20];
        let curve_factor = ((0.9f32.ln()) / curve_decay).exp() - 1.0;
        let decay = curve_decay as f64;
        let c = 0.9f64.ln();
        let exp_term = (c / decay).exp();
        let easy_d = init_difficulty(w, 4);
        Self {
            w,
            curve_decay,
            curve_factor,
            curve_dfactor_ddecay: exp_term * (-c / (decay * decay)),
            exp_w8: w[8].exp(),
            exp_w8_f64: (w[8] as f64).exp(),
            failure_floor_divisor: (w[17] * w[18]).exp(),
            easy_d,
            easy_d_f64: easy_d as f64,
            exp_3w5_f64: (3.0 * w[5] as f64).exp(),
        }
    }
}

#[inline]
fn clamp(x: f32, min: f32, max: f32) -> f32 {
    x.clamp(min, max)
}

#[inline]
fn clamp_grad(x: f32, min: f32, max: f32) -> f64 {
    // Match Burn/PyTorch clamp semantics: gradients pass through values exactly
    // on the clamp boundary and stop only outside the interval.
    if x >= min && x <= max { 1.0 } else { 0.0 }
}

#[inline]
fn add(gw: &mut [f64], idx: usize, value: f64) {
    if let Some(slot) = gw.get_mut(idx) {
        *slot += value;
    }
}

#[inline]
/// Computes weighted binary cross entropy and its derivative with respect to
/// retrievability.
///
/// The loss is `-weight * (label * ln(r) + (1 - label) * ln(1 - r))`, where
/// `r` is `r_raw` clamped into `[MIN_PROB, MAX_PROB]` for numerical stability.
/// The returned gradient is `d(loss) / d(r_raw)`. When `r_raw` lies outside the
/// clamp interval, the clamp is treated as active and the gradient is zero.
fn bce_loss_and_grad_r(r_raw: f32, label: f32, weight: f32) -> (f64, f64) {
    if weight == 0.0 {
        return (0.0, 0.0);
    }
    let r = clamp(r_raw, MIN_PROB, MAX_PROB);
    let loss = -((label * r.ln()) + ((1.0 - label) * (1.0 - r).ln())) * weight;
    let grad = -weight * (label / r - (1.0 - label) / (1.0 - r));
    let grad = if r_raw > MIN_PROB && r_raw < MAX_PROB {
        grad as f64
    } else {
        0.0
    };
    (loss as f64, grad)
}

#[inline]
/// Evaluates the FSRS power forgetting curve and caches intermediates needed
/// by the backward pass.
///
/// With `decay = -w[20]` and `factor = exp(ln(0.9) / decay) - 1`, the curve is
/// `R(t, s) = (1 + max(t, 0) / s * factor)^decay`. The factor is chosen so that
/// `R(s, s) = 0.9`, matching the FSRS definition of stability as the elapsed
/// days at 90% retrievability.
fn curve_forward(p: &RuntimeParams<'_>, t: f32, s: f32) -> CurveCache {
    let t = t.max(0.0);
    let decay = p.curve_decay;
    let factor = p.curve_factor;
    let base = t / s * factor + 1.0;
    let retrievability = base.powf(decay);
    CurveCache {
        t,
        s,
        decay,
        factor,
        base,
        retrievability,
    }
}

#[inline]
/// Backpropagates through the cached power forgetting curve.
///
/// `g_r` is the upstream derivative `d(loss) / d(R)`. The function accumulates
/// `d(loss) / d(w[20])` into `gw` and returns `d(loss) / d(s)` so callers can
/// continue backpropagating into the previous memory state. Since the forward
/// curve uses `decay = -w[20]`, the derivative with respect to `w[20]` is the
/// negative derivative with respect to `decay`.
fn curve_backward(p: &RuntimeParams<'_>, cache: CurveCache, g_r: f64, gw: &mut [f64]) -> f64 {
    if g_r == 0.0 {
        return 0.0;
    }
    let r = cache.retrievability as f64;
    let base = cache.base as f64;
    let decay = cache.decay as f64;
    let t = cache.t as f64;
    let s = cache.s as f64;
    let factor = cache.factor as f64;

    let db_ds = -t * factor / (s * s);
    let g_s = g_r * r * decay / base * db_ds;

    let dfactor_ddecay = p.curve_dfactor_ddecay;
    let db_ddecay = t / s * dfactor_ddecay;
    let dr_ddecay = r * (base.ln() + decay / base * db_ddecay);
    add(gw, 20, -g_r * dr_ddecay);

    g_s
}

#[inline]
/// Computes the initial difficulty for a first review rating.
///
/// The formula is `w[4] - exp(w[5] * (rating - 1)) + 1`. Callers clamp the
/// result into the model difficulty range when it becomes part of a memory
/// state. `rating` is expected to be in `1..=4`; saturating subtraction keeps
/// padding or malformed zero ratings from underflowing.
fn init_difficulty(w: &[f32], rating: usize) -> f32 {
    let offset = rating.saturating_sub(1) as f32;
    w[4] - (w[5] * offset).exp() + 1.0
}

#[inline]
/// Applies one scalar FSRS state transition and records all branch decisions
/// needed for analytic backpropagation.
///
/// `state` is first clamped into the legal stability/difficulty ranges, then
/// retrievability is computed with the power forgetting curve. For non-initial
/// reviews, stability follows one of three active branches:
///
/// - same-day reviews (`delta_t == 0`) use the short-term multiplier
///   `exp(w[17] * (rating - 3 + w[18])) * s^-w[19]`, floored at `1` for
///   ratings `2..=4`;
/// - lapses (`rating == 1`) use the smaller of the raw failure stability and
///   the failure floor `s / exp(w[17] * w[18])`;
/// - all other reviews use the success increment with hard/easy multipliers.
///
/// The first non-padding review initializes stability from `w[0..4]` and
/// difficulty from [`init_difficulty`]. Padding reviews (`rating == 0`) preserve
/// the previous state. The returned [`StepCache`] stores the active branch and
/// pre-clamp values so `step_backward` can use the same non-smooth path.
fn step_forward(
    p: &RuntimeParams<'_>,
    state: State,
    delta_t: f32,
    rating: f32,
    nth: usize,
) -> (State, StepCache) {
    let w = p.w;
    let last_s = clamp(state.s, S_MIN, S_MAX);
    let last_d = clamp(state.d, D_MIN, D_MAX);

    let padding = rating == 0.0;
    if padding {
        let out = State {
            s: last_s,
            d: last_d,
        };
        return (
            out,
            StepCache {
                state_s: state.s,
                state_d: state.d,
                last_s,
                last_d,
                delta_t,
                rating,
                retrievability: 0.0,
                failure_raw: 0.0,
                failure_floor: 0.0,
                failure_used_floor: false,
                short_raw: 0.0,
                short_value: 0.0,
                short_raw_active: false,
                use_short: false,
                use_failure: false,
                init_selected: false,
                padding: true,
                pre_clamp_s: last_s,
                mean_pre_clamp_d: last_d,
                init_rating: 1,
            },
        );
    }

    let init_selected = nth == 0 && state.s == 0.0;
    let init_rating = clamp(rating, 1.0, 4.0) as usize;
    if init_selected {
        let new_s = w[init_rating - 1];
        let raw_d = init_difficulty(w, init_rating);
        let out = State {
            s: clamp(new_s, S_MIN, S_MAX),
            d: clamp(raw_d, D_MIN, D_MAX),
        };
        return (
            out,
            StepCache {
                state_s: state.s,
                state_d: state.d,
                last_s,
                last_d,
                delta_t,
                rating,
                retrievability: 0.0,
                failure_raw: 0.0,
                failure_floor: 0.0,
                failure_used_floor: false,
                short_raw: 0.0,
                short_value: 0.0,
                short_raw_active: false,
                use_short: false,
                use_failure: false,
                init_selected: true,
                padding: false,
                pre_clamp_s: new_s,
                mean_pre_clamp_d: raw_d,
                init_rating,
            },
        );
    }

    let curve = curve_forward(p, delta_t, last_s);
    let r = curve.retrievability;

    let use_short = delta_t == 0.0;
    let use_failure = rating == 1.0;
    let mut failure_raw = 0.0;
    let mut failure_floor = 0.0;
    let mut failure_used_floor = false;
    let mut short_raw = 0.0;
    let mut short_value = 0.0;
    let mut short_raw_active = false;

    let new_s = if use_short {
        short_raw = (w[17] * (rating - 3.0 + w[18])).exp() * last_s.powf(-w[19]);
        short_raw_active = !(rating >= 2.0 && short_raw < 1.0);
        short_value = if rating >= 2.0 {
            short_raw.max(1.0)
        } else {
            short_raw
        };
        last_s * short_value
    } else if use_failure {
        failure_raw = w[11]
            * last_d.powf(-w[12])
            * ((last_s + 1.0).powf(w[13]) - 1.0)
            * ((1.0 - r) * w[14]).exp();
        failure_floor = last_s / p.failure_floor_divisor;
        failure_used_floor = failure_floor < failure_raw;
        if failure_used_floor {
            failure_floor
        } else {
            failure_raw
        }
    } else {
        let hard_penalty = if rating == 2.0 { w[15] } else { 1.0 };
        let easy_bonus = if rating == 4.0 { w[16] } else { 1.0 };
        let success_inc = p.exp_w8
            * (11.0 - last_d)
            * last_s.powf(-w[9])
            * (((1.0 - r) * w[10]).exp() - 1.0)
            * hard_penalty
            * easy_bonus;
        last_s * (success_inc + 1.0)
    };

    let delta_d = -w[6] * (rating - 3.0);
    let next_d = last_d + (10.0 - last_d) * delta_d / 9.0;
    let mean_d = w[7] * (p.easy_d - next_d) + next_d;
    let new_d = clamp(mean_d, D_MIN, D_MAX);

    let pre_clamp_s = new_s;
    let out = State {
        s: clamp(new_s, S_MIN, S_MAX),
        d: new_d,
    };
    let cache = StepCache {
        state_s: state.s,
        state_d: state.d,
        last_s,
        last_d,
        delta_t,
        rating,
        retrievability: r,
        failure_raw,
        failure_floor,
        failure_used_floor,
        short_raw,
        short_value,
        short_raw_active,
        use_short,
        use_failure,
        init_selected,
        padding,
        pre_clamp_s,
        mean_pre_clamp_d: mean_d,
        init_rating,
    };
    (out, cache)
}

/// Differentiates the long-term success stability branch.
///
/// The forward branch is `s' = s * (1 + inc)`, where
/// `inc = exp(w[8]) * (11 - d) * s^-w[9] *
/// (exp((1 - r) * w[10]) - 1) * hard_penalty * easy_bonus`.
///
/// `g` is `d(loss) / d(s')`. The function accumulates parameter gradients for
/// `w[8]`, `w[9]`, `w[10]`, and conditionally `w[15]`/`w[16]`, then returns
/// gradients with respect to the previous stability, previous difficulty, and
/// retrievability `(dL/ds, dL/dd, dL/dr)`.
fn backward_success(
    p: &RuntimeParams<'_>,
    c: &StepCache,
    g: f64,
    gw: &mut [f64],
) -> (f64, f64, f64) {
    if g == 0.0 {
        return (0.0, 0.0, 0.0);
    }
    let w = p.w;
    let s = c.last_s as f64;
    let d = c.last_d as f64;
    let r = c.retrievability as f64;
    let rating = c.rating;
    let a = p.exp_w8_f64;
    let b = 11.0 - d;
    let c_s = s.powf(-(w[9] as f64));
    let e = (((1.0 - r) * w[10] as f64).exp()) - 1.0;
    let exp_e = e + 1.0;
    let hp = if rating == 2.0 { w[15] as f64 } else { 1.0 };
    let eb = if rating == 4.0 { w[16] as f64 } else { 1.0 };
    let inc = a * b * c_s * e * hp * eb;
    let g_inc = g * s;

    let mut g_s = g * (inc + 1.0);
    g_s += g_inc * inc * (-(w[9] as f64) / s);
    let g_d = -(g_inc * a * c_s * e * hp * eb);
    let g_r = g_inc * a * b * c_s * hp * eb * (-(w[10] as f64) * exp_e);

    add(gw, 8, g_inc * inc);
    add(gw, 9, g_inc * inc * -s.ln());
    add(gw, 10, g_inc * a * b * c_s * hp * eb * ((1.0 - r) * exp_e));
    if rating == 2.0 {
        add(gw, 15, g_inc * a * b * c_s * e * eb);
    }
    if rating == 4.0 {
        add(gw, 16, g_inc * a * b * c_s * e * hp);
    }
    (g_s, g_d, g_r)
}

/// Differentiates the lapse stability branch.
///
/// The raw branch is
/// `w[11] * d^-w[12] * ((s + 1)^w[13] - 1) * exp((1 - r) * w[14])`.
/// The forward pass uses the failure floor `s / exp(w[17] * w[18])` when that
/// floor is lower than the raw value. This backward helper follows only the
/// active branch selected in [`step_forward`], so the gradient at the branch
/// boundary is intentionally one-sided.
///
/// Returns gradients with respect to previous stability, previous difficulty,
/// and retrievability.
fn backward_failure(
    p: &RuntimeParams<'_>,
    c: &StepCache,
    g: f64,
    gw: &mut [f64],
) -> (f64, f64, f64) {
    if g == 0.0 {
        return (0.0, 0.0, 0.0);
    }
    let w = p.w;
    let s = c.last_s as f64;
    let d = c.last_d as f64;
    let r = c.retrievability as f64;
    if c.failure_used_floor {
        let floor = c.failure_floor as f64;
        let exp_floor = floor / s;
        add(gw, 17, g * floor * (-(w[18] as f64)));
        add(gw, 18, g * floor * (-(w[17] as f64)));
        return (g * exp_floor, 0.0, 0.0);
    }

    let raw = c.failure_raw as f64;
    let base = s + 1.0;
    let p = base.powf(w[13] as f64);
    let es = p - 1.0;
    let last_d_pow = d.powf(-(w[12] as f64));
    let er = ((1.0 - r) * w[14] as f64).exp();

    add(gw, 11, g * raw / w[11] as f64);
    add(gw, 12, g * raw * -d.ln());
    add(gw, 13, g * (w[11] as f64) * last_d_pow * er * p * base.ln());
    add(gw, 14, g * raw * (1.0 - r));

    let g_s = g * (w[11] as f64) * last_d_pow * er * (w[13] as f64) * p / base;
    let g_d = g * raw * (-(w[12] as f64) / d);
    let g_r = g * raw * (-(w[14] as f64));
    let _ = es;
    (g_s, g_d, g_r)
}

/// Differentiates the same-day short-term stability branch.
///
/// The branch computes `s' = s * short_value`, where the raw multiplier is
/// `exp(w[17] * (rating - 3 + w[18])) * s^-w[19]`. For ratings `2..=4`, the raw
/// multiplier is floored at `1`; when that floor is active, only the direct
/// derivative of `s'` with respect to `s` remains.
///
/// Returns the gradient with respect to the previous stability.
fn backward_short(p: &RuntimeParams<'_>, c: &StepCache, g: f64, gw: &mut [f64]) -> f64 {
    if g == 0.0 {
        return 0.0;
    }
    let w = p.w;
    let s = c.last_s as f64;
    let mut g_s = g * c.short_value as f64;
    if c.short_raw_active {
        let g_raw = g * s;
        let raw = c.short_raw as f64;
        let q = (c.rating - 3.0 + w[18]) as f64;
        add(gw, 17, g_raw * raw * q);
        add(gw, 18, g_raw * raw * w[17] as f64);
        add(gw, 19, g_raw * raw * -s.ln());
        g_s += g_raw * raw * (-(w[19] as f64) / s);
    }
    g_s
}

/// Differentiates the difficulty update branch.
///
/// Difficulty first moves by linear damping,
/// `next_d = d + (10 - d) * (-w[6] * (rating - 3)) / 9`, then mean-reverts
/// toward the easy-rating initial difficulty:
/// `mean_d = w[7] * (init_difficulty(4) - next_d) + next_d`.
///
/// The gradient is stopped when the output difficulty clamp is active. Otherwise
/// this helper accumulates gradients for `w[4]`, `w[5]`, `w[6]`, and `w[7]`, and
/// returns `d(loss) / d(previous_difficulty)`.
fn backward_difficulty(p: &RuntimeParams<'_>, c: &StepCache, g_out: f64, gw: &mut [f64]) -> f64 {
    if g_out == 0.0 {
        return 0.0;
    }
    let w = p.w;
    let g_mean = g_out * clamp_grad(c.mean_pre_clamp_d, D_MIN, D_MAX);
    if g_mean == 0.0 {
        return 0.0;
    }
    let rating_minus_3 = (c.rating - 3.0) as f64;
    let last_d = c.last_d as f64;
    let delta_d = -(w[6] as f64) * rating_minus_3;
    let next_d = last_d + (10.0 - last_d) * delta_d / 9.0;
    let easy_d = p.easy_d_f64;

    add(gw, 7, g_mean * (easy_d - next_d));
    add(gw, 4, g_mean * w[7] as f64);
    add(gw, 5, g_mean * (w[7] as f64) * -3.0 * p.exp_3w5_f64);

    let g_next = g_mean * (1.0 - w[7] as f64);
    add(gw, 6, g_next * (10.0 - last_d) * (-(rating_minus_3)) / 9.0);
    g_next * (1.0 - delta_d / 9.0)
}

/// Differentiates the first-review initialization branch.
///
/// Initial stability is selected directly from `w[rating - 1]`; initial
/// difficulty uses [`init_difficulty`] and the same difficulty clamp as the
/// forward pass. This helper accumulates gradients into `w[0..4]` for stability
/// and `w[4]`/`w[5]` for difficulty.
fn backward_init(p: &RuntimeParams<'_>, rating: usize, g_s: f64, g_d: f64, gw: &mut [f64]) {
    let w = p.w;
    add(gw, rating - 1, g_s);
    let raw_d = init_difficulty(w, rating);
    let g_raw_d = g_d * clamp_grad(raw_d, D_MIN, D_MAX);
    if g_raw_d != 0.0 {
        let offset = (rating - 1) as f64;
        add(gw, 4, g_raw_d);
        add(gw, 5, g_raw_d * -offset * (offset * w[5] as f64).exp());
    }
}

/// Backpropagates through one cached FSRS state transition.
///
/// `g_out_s` and `g_out_d` are gradients from later state values. `g_r_extra`
/// is an additional gradient flowing directly through the review-step
/// retrievability, used by per-card losses that score each historical review.
///
/// The backward path mirrors the exact branch chosen by [`step_forward`]:
/// padding copies gradients to the previous state, first-review initialization
/// updates initial parameters, and review branches route through short,
/// failure, or success stability plus the difficulty update. Any accumulated
/// `dL/dR` is then propagated through the forgetting curve to previous
/// stability and `w[20]`. State clamps are treated as hard stops at their
/// boundaries.
fn step_backward(
    p: &RuntimeParams<'_>,
    c: &StepCache,
    g_out_s: f64,
    g_out_d: f64,
    g_r_extra: f64,
    gw: &mut [f64],
) -> (f64, f64) {
    let g_pre_s = g_out_s * clamp_grad(c.pre_clamp_s, S_MIN, S_MAX);
    let g_pre_d = g_out_d;
    let mut g_last_s = 0.0;
    let mut g_last_d = 0.0;
    let mut g_r = g_r_extra;

    if c.padding {
        g_last_s += g_pre_s;
        g_last_d += g_pre_d;
    } else if c.init_selected {
        backward_init(p, c.init_rating, g_pre_s, g_pre_d, gw);
    } else {
        if c.use_short {
            g_last_s += backward_short(p, c, g_pre_s, gw);
        } else if c.use_failure {
            let (gs, gd, gr) = backward_failure(p, c, g_pre_s, gw);
            g_last_s += gs;
            g_last_d += gd;
            g_r += gr;
        } else {
            let (gs, gd, gr) = backward_success(p, c, g_pre_s, gw);
            g_last_s += gs;
            g_last_d += gd;
            g_r += gr;
        }
        g_last_d += backward_difficulty(p, c, g_pre_d, gw);
    }

    let t = c.delta_t.max(0.0);
    let curve = CurveCache {
        t,
        s: c.last_s,
        decay: p.curve_decay,
        factor: p.curve_factor,
        base: t / c.last_s * p.curve_factor + 1.0,
        retrievability: c.retrievability,
    };
    g_last_s += curve_backward(p, curve, g_r, gw);

    let g_state_s = g_last_s * clamp_grad(c.state_s, S_MIN, S_MAX);
    let g_state_d = g_last_d * clamp_grad(c.state_d, D_MIN, D_MAX);
    (g_state_s, g_state_d)
}

#[allow(clippy::too_many_arguments)]
/// Computes the loss for one batch column by scoring only the next review after
/// a fixed history prefix.
///
/// `t_hist` and `r_hist` are flattened in time-major order:
/// `idx = time * batch + column`. The function replays `seq_len` historical
/// reviews for `column`, predicts retrievability after `delta_t`, and computes
/// weighted BCE against `label`.
///
/// When `gw` is provided, gradients are accumulated into the shared parameter
/// gradient buffer by first differentiating the final forgetting curve and then
/// walking the cached review steps in reverse.
fn prefix_loss_and_grad(
    p: &RuntimeParams<'_>,
    t_hist: &[f32],
    r_hist: &[f32],
    seq_len: usize,
    batch: usize,
    column: usize,
    delta_t: f32,
    label: f32,
    weight: f32,
    gw: Option<&mut [f64]>,
    caches: &mut Vec<StepCache>,
) -> f64 {
    let mut state = State { s: 0.0, d: 0.0 };
    let need_grad = gw.is_some();
    if need_grad {
        caches.clear();
    }
    for t in 0..seq_len {
        let idx = t * batch + column;
        let (next, cache) = step_forward(p, state, t_hist[idx], r_hist[idx], t);
        state = next;
        if need_grad {
            caches.push(cache);
        }
    }
    let curve = curve_forward(p, delta_t, state.s);
    let (loss, g_r) = bce_loss_and_grad_r(curve.retrievability, label, weight);
    if let Some(gw) = gw {
        let mut g_s = curve_backward(p, curve, g_r, gw);
        let mut g_d = 0.0;
        for cache in caches.iter().rev() {
            let (prev_s, prev_d) = step_backward(p, cache, g_s, g_d, 0.0, gw);
            g_s = prev_s;
            g_d = prev_d;
        }
    }
    loss
}

#[allow(clippy::too_many_arguments)]
/// Computes summed next-review loss and parameter gradients for a prefix batch.
///
/// Each column represents one card prefix. `delta_ts`, `labels`, and `weights`
/// have length `batch` and describe the target review following each prefix.
/// `gw` is an accumulation buffer for the 21 FSRS parameters; it is not cleared
/// by this function.
pub(crate) fn batch_loss_and_grad(
    w: &[f32],
    t_hist: &[f32],
    r_hist: &[f32],
    seq_len: usize,
    batch: usize,
    seq_lens: &[usize],
    delta_ts: &[f32],
    labels: &[f32],
    weights: &[f32],
    gw: &mut [f64],
) -> f64 {
    let mut loss = 0.0;
    let params = RuntimeParams::new(w);
    let mut caches = Vec::new();
    for column in 0..batch {
        let column_seq_len = seq_lens[column].min(seq_len);
        loss += prefix_loss_and_grad(
            &params,
            t_hist,
            r_hist,
            column_seq_len,
            batch,
            column,
            delta_ts[column],
            labels[column],
            weights[column],
            Some(gw),
            &mut caches,
        );
    }
    loss
}

#[allow(clippy::too_many_arguments)]
/// Computes summed next-review loss for a prefix batch without gradients.
///
/// This is the forward-only counterpart of [`batch_loss_and_grad`]. It uses the
/// same time-major history layout and target arrays, but skips all cache
/// backpropagation work.
pub(crate) fn batch_loss(
    w: &[f32],
    t_hist: &[f32],
    r_hist: &[f32],
    seq_len: usize,
    batch: usize,
    seq_lens: &[usize],
    delta_ts: &[f32],
    labels: &[f32],
    weights: &[f32],
) -> f64 {
    let mut loss = 0.0;
    let params = RuntimeParams::new(w);
    let mut caches = Vec::new();
    for column in 0..batch {
        let column_seq_len = seq_lens[column].min(seq_len);
        loss += prefix_loss_and_grad(
            &params,
            t_hist,
            r_hist,
            column_seq_len,
            batch,
            column,
            delta_ts[column],
            labels[column],
            weights[column],
            None,
            &mut caches,
        );
    }
    loss
}

#[allow(clippy::too_many_arguments)]
/// Computes all in-card review losses for one batch column.
///
/// Unlike [`prefix_loss_and_grad`], this scores every review after the first
/// review in the same card sequence. At step `t`, the BCE target uses
/// `labels[t * batch + column]` and `weights[t * batch + column]`, while the
/// predicted retrievability comes from the state before applying that review.
///
/// When gradients are requested, per-step `dL/dR` values are stored and then
/// propagated backward through the entire cached card trajectory.
fn card_column_loss_and_grad(
    p: &RuntimeParams<'_>,
    t_hist: &[f32],
    r_hist: &[f32],
    seq_len: usize,
    batch: usize,
    column: usize,
    labels: &[f32],
    weights: &[f32],
    mut gw: Option<&mut [f64]>,
    caches: &mut Vec<StepCache>,
    g_r_losses: &mut Vec<f64>,
) -> f64 {
    let mut state = State { s: 0.0, d: 0.0 };
    let need_grad = gw.is_some();
    let mut loss = 0.0;
    if need_grad {
        caches.clear();
        g_r_losses.clear();
        g_r_losses.resize(seq_len, 0.0);
    }
    for t in 0..seq_len {
        let idx = t * batch + column;
        if t + 1 == seq_len {
            let mut final_g_s = 0.0;
            if t != 0 && weights[idx] != 0.0 {
                let curve = curve_forward(p, t_hist[idx], clamp(state.s, S_MIN, S_MAX));
                let (step_loss, g_r) =
                    bce_loss_and_grad_r(curve.retrievability, labels[idx], weights[idx]);
                loss += step_loss;
                if let Some(gw) = gw.as_deref_mut() {
                    final_g_s = curve_backward(p, curve, g_r, gw);
                    final_g_s *= clamp_grad(state.s, S_MIN, S_MAX);
                }
            }
            if let Some(gw) = gw {
                let mut g_s = final_g_s;
                let mut g_d = 0.0;
                for (t, cache) in caches.iter().enumerate().rev() {
                    let (prev_s, prev_d) = step_backward(p, cache, g_s, g_d, g_r_losses[t], gw);
                    g_s = prev_s;
                    g_d = prev_d;
                }
            }
            return loss;
        }

        let (next, cache) = step_forward(p, state, t_hist[idx], r_hist[idx], t);
        let (step_loss, g_r) = if t == 0 {
            (0.0, 0.0)
        } else {
            bce_loss_and_grad_r(cache.retrievability, labels[idx], weights[idx])
        };
        loss += step_loss;
        if need_grad {
            g_r_losses[t] = g_r;
            caches.push(cache);
        }
        state = next;
    }
    loss
}

#[allow(clippy::too_many_arguments)]
/// Computes summed in-card review loss and parameter gradients for a batch.
///
/// Each batch column is one full card trajectory. `labels` and `weights` use
/// the same time-major layout as `t_hist` and `r_hist`; entries for the first
/// review are ignored because there is no prior state to score. `gw` is an
/// accumulation buffer and is not reset by this function.
pub(crate) fn card_loss_and_grad(
    w: &[f32],
    t_hist: &[f32],
    r_hist: &[f32],
    seq_len: usize,
    batch: usize,
    seq_lens: &[usize],
    labels: &[f32],
    weights: &[f32],
    gw: &mut [f64],
) -> f64 {
    let mut loss = 0.0;
    let params = RuntimeParams::new(w);
    let mut caches = Vec::new();
    let mut g_r_losses = Vec::new();
    for column in 0..batch {
        let column_seq_len = seq_lens[column].min(seq_len);
        loss += card_column_loss_and_grad(
            &params,
            t_hist,
            r_hist,
            column_seq_len,
            batch,
            column,
            labels,
            weights,
            Some(gw),
            &mut caches,
            &mut g_r_losses,
        );
    }
    loss
}

#[allow(clippy::too_many_arguments)]
/// Computes summed in-card review loss for a batch without gradients.
///
/// This is the forward-only counterpart of [`card_loss_and_grad`], used when the
/// training loop or evaluation path needs only the scalar objective value for
/// full card trajectories.
pub(crate) fn card_loss(
    w: &[f32],
    t_hist: &[f32],
    r_hist: &[f32],
    seq_len: usize,
    batch: usize,
    seq_lens: &[usize],
    labels: &[f32],
    weights: &[f32],
) -> f64 {
    let mut loss = 0.0;
    let params = RuntimeParams::new(w);
    let mut caches = Vec::new();
    let mut g_r_losses = Vec::new();
    for column in 0..batch {
        let column_seq_len = seq_lens[column].min(seq_len);
        loss += card_column_loss_and_grad(
            &params,
            t_hist,
            r_hist,
            column_seq_len,
            batch,
            column,
            labels,
            weights,
            None,
            &mut caches,
            &mut g_r_losses,
        );
    }
    loss
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::DEFAULT_PARAMETERS;

    #[test]
    fn clamp_grad_matches_burn_boundary_semantics() {
        assert_eq!(clamp_grad(0.0, 0.001, 100.0), 0.0);
        assert_eq!(clamp_grad(0.001, 0.001, 100.0), 1.0);
        assert_eq!(clamp_grad(0.5, 0.001, 100.0), 1.0);
        assert_eq!(clamp_grad(100.0, 0.001, 100.0), 1.0);
        assert_eq!(clamp_grad(101.0, 0.001, 100.0), 0.0);
    }

    #[test]
    fn finite_difference_matches_batch_gradient() {
        let w = DEFAULT_PARAMETERS.to_vec();
        let seq = 3;
        let batch = 2;
        let th = vec![0.0, 0.0, 3.0, 2.0, 5.0, 6.0];
        let rh = vec![4.0, 1.0, 3.0, 2.0, 1.0, 3.0];
        let dts = vec![7.0, 9.0];
        let labels = vec![1.0, 0.0];
        let weights = vec![1.0, 0.7];
        let seq_lens = vec![seq; batch];
        let mut grad = [0.0; 21];
        batch_loss_and_grad(
            &w, &th, &rh, seq, batch, &seq_lens, &dts, &labels, &weights, &mut grad,
        );
        for i in 0..21 {
            let eps = 1e-3f32.max(w[i].abs() * 1e-3);
            let mut wp = w.clone();
            let mut wm = w.clone();
            wp[i] += eps;
            wm[i] -= eps;
            let lp = batch_loss(
                &wp, &th, &rh, seq, batch, &seq_lens, &dts, &labels, &weights,
            );
            let lm = batch_loss(
                &wm, &th, &rh, seq, batch, &seq_lens, &dts, &labels, &weights,
            );
            let numeric = (lp - lm) / (2.0 * eps as f64);
            let diff = (numeric - grad[i]).abs();
            let scale = numeric.abs().max(grad[i].abs()).max(1.0);
            assert!(
                diff / scale < 2e-2,
                "param {i}: analytic={} numeric={numeric} diff={diff}",
                grad[i]
            );
        }
    }

    #[test]
    fn windowed_matches_sum_of_prefixes() {
        let w = DEFAULT_PARAMETERS.to_vec();
        let seq = 4;
        let batch = 1;
        let th_card = vec![0.0, 2.0, 5.0, 8.0];
        let rh_card = vec![4.0, 3.0, 1.0, 3.0];
        let mut labels_card = vec![0.0; seq];
        let mut weights_card = vec![0.0; seq];
        labels_card[1] = 1.0;
        labels_card[2] = 0.0;
        labels_card[3] = 1.0;
        weights_card[1] = 0.5;
        weights_card[2] = 0.7;
        weights_card[3] = 1.1;
        let mut g_card = [0.0; 21];
        let seq_lens_card = vec![seq; batch];
        let l_card = card_loss_and_grad(
            &w,
            &th_card,
            &rh_card,
            seq,
            batch,
            &seq_lens_card,
            &labels_card,
            &weights_card,
            &mut g_card,
        );

        let mut l_prefix = 0.0;
        let mut g_prefix = [0.0; 21];
        for prefix_len in 2..=4 {
            let hist_len = prefix_len - 1;
            let th = th_card[..hist_len].to_vec();
            let rh = rh_card[..hist_len].to_vec();
            let dts = vec![th_card[prefix_len - 1]];
            let labels = vec![labels_card[prefix_len - 1]];
            let weights = vec![weights_card[prefix_len - 1]];
            let seq_lens = vec![hist_len];
            l_prefix += batch_loss_and_grad(
                &w,
                &th,
                &rh,
                hist_len,
                1,
                &seq_lens,
                &dts,
                &labels,
                &weights,
                &mut g_prefix,
            );
        }
        assert!((l_card - l_prefix).abs() < 1e-9);
        for i in 0..21 {
            assert!(
                (g_card[i] - g_prefix[i]).abs() < 1e-9,
                "param {i}: card={} prefix={}",
                g_card[i],
                g_prefix[i]
            );
        }
    }
}

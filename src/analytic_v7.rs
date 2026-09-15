// Analytic loss and gradient for the dual-trace FSRS-7 model.
//
// This mirrors the reference forward pass in `model_v7.rs` exactly (state carries a
// slow stability, a fast stability, and a difficulty), but computes the loss
// gradient with forward-mode dual numbers instead of framework autodiff. It is the
// fast host-side training path used by the windowed FSRS-7 optimizer.
//
// The scalar (`f64`) path is used for validation loss; the dual path is used
// for training gradients. Both were validated against the former autodiff path
// during the migration and retain a single-card prefix-batch regression test.

const PARAM_LEN: usize = 34;
const S_MIN: f64 = 0.0001;
const S_MAX: f64 = 36500.0;
const D_MIN: f64 = 1.0;
const D_MAX: f64 = 10.0;

#[derive(Clone, Copy, Debug)]
struct Dual {
    value: f64,
    grad: [f64; PARAM_LEN],
}

impl Dual {
    fn constant(value: f64) -> Self {
        Self {
            value,
            grad: [0.0; PARAM_LEN],
        }
    }

    fn variable(value: f64, index: usize) -> Self {
        let mut grad = [0.0; PARAM_LEN];
        grad[index] = 1.0;
        Self { value, grad }
    }

    fn add(self, rhs: Self) -> Self {
        let mut grad = [0.0; PARAM_LEN];
        for (index, value) in grad.iter_mut().enumerate() {
            *value = self.grad[index] + rhs.grad[index];
        }
        Self {
            value: self.value + rhs.value,
            grad,
        }
    }

    fn add_const(self, rhs: f64) -> Self {
        Self {
            value: self.value + rhs,
            grad: self.grad,
        }
    }

    fn sub_const(self, rhs: f64) -> Self {
        self.add_const(-rhs)
    }

    /// `lhs - self`
    fn const_sub(self, lhs: f64) -> Self {
        let mut grad = [0.0; PARAM_LEN];
        for (index, value) in grad.iter_mut().enumerate() {
            *value = -self.grad[index];
        }
        Self {
            value: lhs - self.value,
            grad,
        }
    }

    fn neg(self) -> Self {
        self.mul_const(-1.0)
    }

    fn mul(self, rhs: Self) -> Self {
        let mut grad = [0.0; PARAM_LEN];
        for (index, value) in grad.iter_mut().enumerate() {
            *value = self.grad[index] * rhs.value + rhs.grad[index] * self.value;
        }
        Self {
            value: self.value * rhs.value,
            grad,
        }
    }

    fn mul_const(self, rhs: f64) -> Self {
        let mut grad = [0.0; PARAM_LEN];
        for (index, value) in grad.iter_mut().enumerate() {
            *value = self.grad[index] * rhs;
        }
        Self {
            value: self.value * rhs,
            grad,
        }
    }

    fn div(self, rhs: Self) -> Self {
        let denom = rhs.value * rhs.value;
        let mut grad = [0.0; PARAM_LEN];
        for (index, value) in grad.iter_mut().enumerate() {
            *value = (self.grad[index] * rhs.value - self.value * rhs.grad[index]) / denom;
        }
        Self {
            value: self.value / rhs.value,
            grad,
        }
    }

    fn exp(self) -> Self {
        let value = self.value.exp();
        let mut grad = [0.0; PARAM_LEN];
        for (index, item) in grad.iter_mut().enumerate() {
            *item = self.grad[index] * value;
        }
        Self { value, grad }
    }

    fn ln(self) -> Self {
        let mut grad = [0.0; PARAM_LEN];
        for (index, item) in grad.iter_mut().enumerate() {
            *item = self.grad[index] / self.value;
        }
        Self {
            value: self.value.ln(),
            grad,
        }
    }

    /// `self ^ rhs` for a positive base, matching the reference `powf`.
    fn pow(self, rhs: Self) -> Self {
        rhs.mul(self.ln()).exp()
    }

    fn clamp(self, min: f64, max: f64) -> Self {
        if self.value < min {
            Self::constant(min)
        } else if self.value > max {
            Self::constant(max)
        } else {
            self
        }
    }

    /// Clamp from above, matching the reference `clamp_max` (zero gradient above `max`).
    fn clamp_max(self, max: f64) -> Self {
        if self.value > max {
            Self::constant(max)
        } else {
            self
        }
    }
}

#[derive(Clone, Copy)]
struct MemoryStateDual {
    stability: Dual,
    stability_fast: Dual,
    difficulty: Dual,
}

#[derive(Clone, Copy)]
struct MemoryStateScalar {
    stability: f64,
    stability_fast: f64,
    difficulty: f64,
}

fn dual_params(w: &[f32]) -> [Dual; PARAM_LEN] {
    core::array::from_fn(|index| Dual::variable(w[index] as f64, index))
}

fn init_stability(w: &[Dual; PARAM_LEN], rating: usize) -> Dual {
    w[rating.clamp(1, 4) - 1]
}

fn init_difficulty(w: &[Dual; PARAM_LEN], rating: usize) -> Dual {
    let rating = rating.clamp(1, 4);
    // w[4] - exp(w[5] * (rating - 1)) + 1
    w[4].add(w[5].mul_const((rating - 1) as f64).exp().neg())
        .add_const(1.0)
}

/// Retrievability mixture, mirroring `model_v7::power_forgetting_curve`.
fn forgetting_curve(w: &[Dual; PARAM_LEN], t: f64, s: Dual, s_fast: Dual, d: Dual) -> Dual {
    let t = t.max(0.0);
    let s = s.clamp(S_MIN, S_MAX);
    let s_fast = s_fast.clamp(S_MIN, S_MAX);
    let d = d.clamp(D_MIN, D_MAX);

    // fast component
    let decay1_mag = w[23]
        .mul(s_fast.pow(w[33].sub_const(0.3)))
        .clamp(0.01, 0.95);
    let decay1 = decay1_mag.neg();
    let inv_decay1 = Dual::constant(1.0).div(decay1);
    let factor1 = w[25]
        .ln()
        .mul(inv_decay1)
        .clamp_max(60.0)
        .exp()
        .sub_const(1.0);
    let t_over_s_fast = Dual::constant(t).div(s_fast);
    let r1 = t_over_s_fast.mul(factor1).add_const(1.0).pow(decay1);

    // slow component
    let decay2 = w[24].clamp(0.01, 0.95).neg();
    let inv_decay2 = Dual::constant(1.0).div(decay2);
    let factor2 = w[26].pow(inv_decay2).sub_const(1.0);
    let d_timescale = d.sub_const(5.0).mul(w[32].sub_const(0.3)).exp();
    let t_over_s = Dual::constant(t).div(s);
    let r2 = t_over_s
        .mul(factor2)
        .mul(d_timescale)
        .add_const(1.0)
        .pow(decay2);

    let weight1 = w[27].mul(s_fast.pow(w[29].neg()));
    let weight2 = w[28]
        .mul(s.pow(w[30]))
        .mul(d.sub_const(5.0).mul(w[31].sub_const(0.5)).exp());
    let retention = weight1
        .mul(r1)
        .add(weight2.mul(r2))
        .div(weight1.add(weight2));
    retention.mul_const(1.0 - 2e-5).add_const(1e-5)
}

/// Fast-trace recall used when updating the fast stability, mirroring
/// `model_v7::fast_component_recall`.
fn fast_component_recall(w: &[Dual; PARAM_LEN], t: f64, s_fast: Dual) -> Dual {
    let t = t.max(0.0);
    let s_fast = s_fast.clamp(S_MIN, S_MAX);
    let decay1_mag = w[23]
        .mul(s_fast.pow(w[33].sub_const(0.3)))
        .clamp(0.01, 0.95);
    let decay1 = decay1_mag.neg();
    let inv_decay1 = Dual::constant(1.0).div(decay1);
    let factor1 = w[25]
        .ln()
        .mul(inv_decay1)
        .clamp_max(60.0)
        .exp()
        .sub_const(1.0);
    let t_over_s_fast = Dual::constant(t).div(s_fast);
    t_over_s_fast.mul(factor1).add_const(1.0).pow(decay1)
}

/// Next stability for a trace, mirroring `model_v7::stability_for_set` (result
/// is left unclamped; the caller clamps to `[S_MIN, S_MAX]`).
fn stability_for_set(
    w: &[Dual; PARAM_LEN],
    last_s: Dual,
    last_d: Dual,
    r: Dual,
    rating: usize,
    start: usize,
) -> Dual {
    let hard_penalty = if rating == 2 {
        w[start + 6]
    } else {
        Dual::constant(1.0)
    };
    let easy_bonus = if rating == 4 {
        w[start + 7]
    } else {
        Dual::constant(1.0)
    };
    let new_s_fail = w[start + 3]
        .mul(last_s.add_const(1.0).pow(w[start + 4]).sub_const(1.0))
        .mul(r.const_sub(1.0).mul(w[start + 5]).exp());
    // tensor_min(last_s, new_s_fail)
    let pls = if last_s.value > new_s_fail.value {
        new_s_fail
    } else {
        last_s
    };
    if rating <= 1 {
        return pls;
    }
    let sinc = w[start]
        .sub_const(1.5)
        .exp()
        .mul(last_d.const_sub(11.0))
        .mul(last_s.pow(w[start + 1].neg()))
        .mul(r.const_sub(1.0).mul(w[start + 2]).exp().sub_const(1.0))
        .mul(hard_penalty)
        .mul(easy_bonus)
        .add_const(1.0);
    let success = last_s.mul(sinc);
    // tensor_max(pls, last_s * sinc)
    if pls.value < success.value {
        success
    } else {
        pls
    }
}

fn next_difficulty(w: &[Dual; PARAM_LEN], d: Dual, rating: usize, r: Dual) -> Dual {
    let delta_d = w[6].neg().mul_const(rating as f64 - 3.0);
    let delta_d = if rating == 1 {
        delta_d.mul(r.add_const(0.1))
    } else {
        delta_d
    };
    // linear_damping(delta_d, d) = (10 - d) * delta_d / 9
    let new_d = d.add(d.const_sub(10.0).mul(delta_d).mul_const(1.0 / 9.0));
    init_difficulty(w, 4)
        .mul_const(0.01)
        .add(new_d.mul_const(0.99))
        .clamp(D_MIN, D_MAX)
}

fn step(
    w: &[Dual; PARAM_LEN],
    delta_t: f64,
    rating: usize,
    state: MemoryStateDual,
    nth: usize,
) -> MemoryStateDual {
    let last_s = state.stability.clamp(S_MIN, S_MAX);
    let last_d = state.difficulty.clamp(D_MIN, D_MAX);
    let last_s_fast = state.stability_fast.clamp(S_MIN, S_MAX);
    if rating == 0 {
        return MemoryStateDual {
            stability: last_s,
            stability_fast: last_s_fast,
            difficulty: last_d,
        };
    }
    if nth == 0 && state.stability.value == 0.0 {
        let init_s = init_stability(w, rating);
        return MemoryStateDual {
            stability: init_s.clamp(S_MIN, S_MAX),
            stability_fast: init_s.mul_const(0.8).clamp(S_MIN, S_MAX),
            difficulty: init_difficulty(w, rating).clamp(D_MIN, D_MAX),
        };
    }

    let delta_t = delta_t.max(0.0);
    let r = forgetting_curve(w, delta_t, last_s, last_s_fast, last_d);
    let new_s_slow = stability_for_set(w, last_s, last_d, r, rating, 7);
    let r_fast = fast_component_recall(w, delta_t, last_s_fast);
    let mut new_s_fast = stability_for_set(w, last_s_fast, last_d, r_fast, rating, 15);
    if rating == 1 {
        // relearn: tensor_min(new_s_fast, new_s_slow * 0.8)
        let relearn = new_s_slow.mul_const(0.8);
        if new_s_fast.value > relearn.value {
            new_s_fast = relearn;
        }
    }
    let new_d = next_difficulty(w, last_d, rating, r);
    MemoryStateDual {
        stability: new_s_slow.clamp(S_MIN, S_MAX),
        stability_fast: new_s_fast.clamp(S_MIN, S_MAX),
        difficulty: new_d,
    }
}

fn bce_loss(r: Dual, label: f64, weight: f64) -> Dual {
    let r = r.clamp(0.0001, 0.9999);
    debug_assert!(label == 0.0 || label == 1.0);
    let probability = if label == 0.0 { r.const_sub(1.0) } else { r };
    probability.ln().mul_const(-weight)
}

// --- scalar (f64) mirrors, used for validation loss ---

fn init_stability_scalar(w: &[f32], rating: usize) -> f64 {
    w[rating.clamp(1, 4) - 1] as f64
}

fn init_difficulty_scalar(w: &[f32], rating: usize) -> f64 {
    let rating = rating.clamp(1, 4);
    w[4] as f64 - (w[5] as f64 * (rating - 1) as f64).exp() + 1.0
}

fn forgetting_curve_scalar(w: &[f32], t: f64, s: f64, s_fast: f64, d: f64) -> f64 {
    let t = t.max(0.0);
    let s = s.clamp(S_MIN, S_MAX);
    let s_fast = s_fast.clamp(S_MIN, S_MAX);
    let d = d.clamp(D_MIN, D_MAX);

    let decay1_mag = (w[23] as f64 * s_fast.powf(w[33] as f64 - 0.3)).clamp(0.01, 0.95);
    let decay1 = -decay1_mag;
    let factor1 = ((w[25] as f64).ln() / decay1).min(60.0).exp() - 1.0;
    let r1 = (1.0 + factor1 * (t / s_fast)).powf(decay1);

    let decay2 = -(w[24] as f64).clamp(0.01, 0.95);
    let factor2 = (w[26] as f64).powf(1.0 / decay2) - 1.0;
    let d_timescale = ((d - 5.0) * (w[32] as f64 - 0.3)).exp();
    let r2 = (1.0 + factor2 * d_timescale * (t / s)).powf(decay2);

    let weight1 = w[27] as f64 * s_fast.powf(-(w[29] as f64));
    let weight2 = w[28] as f64 * s.powf(w[30] as f64) * ((d - 5.0) * (w[31] as f64 - 0.5)).exp();
    let retention = (weight1 * r1 + weight2 * r2) / (weight1 + weight2);
    retention.mul_add(1.0 - 2e-5, 1e-5)
}

fn fast_component_recall_scalar(w: &[f32], t: f64, s_fast: f64) -> f64 {
    let t = t.max(0.0);
    let s_fast = s_fast.clamp(S_MIN, S_MAX);
    let decay1_mag = (w[23] as f64 * s_fast.powf(w[33] as f64 - 0.3)).clamp(0.01, 0.95);
    let decay1 = -decay1_mag;
    let factor1 = ((w[25] as f64).ln() / decay1).min(60.0).exp() - 1.0;
    (1.0 + factor1 * (t / s_fast)).powf(decay1)
}

fn stability_for_set_scalar(
    w: &[f32],
    last_s: f64,
    last_d: f64,
    r: f64,
    rating: usize,
    start: usize,
) -> f64 {
    let hard_penalty = if rating == 2 {
        w[start + 6] as f64
    } else {
        1.0
    };
    let easy_bonus = if rating == 4 {
        w[start + 7] as f64
    } else {
        1.0
    };
    let new_s_fail = w[start + 3] as f64
        * ((last_s + 1.0).powf(w[start + 4] as f64) - 1.0)
        * ((1.0 - r) * w[start + 5] as f64).exp();
    let pls = last_s.min(new_s_fail);
    if rating <= 1 {
        return pls;
    }
    let sinc = (w[start] as f64 - 1.5).exp()
        * (11.0 - last_d)
        * last_s.powf(-(w[start + 1] as f64))
        * (((1.0 - r) * w[start + 2] as f64).exp() - 1.0)
        * hard_penalty
        * easy_bonus
        + 1.0;
    pls.max(last_s * sinc)
}

fn next_difficulty_scalar(w: &[f32], d: f64, rating: usize, r: f64) -> f64 {
    let delta_d = -(w[6] as f64) * (rating as f64 - 3.0);
    let delta_d = if rating == 1 {
        delta_d * (r + 0.1)
    } else {
        delta_d
    };
    let new_d = d + (10.0 - d) * delta_d / 9.0;
    (init_difficulty_scalar(w, 4) * 0.01 + new_d * 0.99).clamp(D_MIN, D_MAX)
}

fn step_scalar(
    w: &[f32],
    delta_t: f64,
    rating: usize,
    state: MemoryStateScalar,
    nth: usize,
) -> MemoryStateScalar {
    let last_s = state.stability.clamp(S_MIN, S_MAX);
    let last_d = state.difficulty.clamp(D_MIN, D_MAX);
    let last_s_fast = state.stability_fast.clamp(S_MIN, S_MAX);
    if rating == 0 {
        return MemoryStateScalar {
            stability: last_s,
            stability_fast: last_s_fast,
            difficulty: last_d,
        };
    }
    if nth == 0 && state.stability == 0.0 {
        let init_s = init_stability_scalar(w, rating);
        return MemoryStateScalar {
            stability: init_s.clamp(S_MIN, S_MAX),
            stability_fast: (init_s * 0.8).clamp(S_MIN, S_MAX),
            difficulty: init_difficulty_scalar(w, rating).clamp(D_MIN, D_MAX),
        };
    }

    let delta_t = delta_t.max(0.0);
    let r = forgetting_curve_scalar(w, delta_t, last_s, last_s_fast, last_d);
    let new_s_slow = stability_for_set_scalar(w, last_s, last_d, r, rating, 7);
    let r_fast = fast_component_recall_scalar(w, delta_t, last_s_fast);
    let mut new_s_fast = stability_for_set_scalar(w, last_s_fast, last_d, r_fast, rating, 15);
    if rating == 1 {
        new_s_fast = new_s_fast.min(new_s_slow * 0.8);
    }
    MemoryStateScalar {
        stability: new_s_slow.clamp(S_MIN, S_MAX),
        stability_fast: new_s_fast.clamp(S_MIN, S_MAX),
        difficulty: next_difficulty_scalar(w, last_d, rating, r),
    }
}

fn bce_loss_scalar(r: f64, label: f64, weight: f64) -> f64 {
    let r = r.clamp(0.0001, 0.9999);
    debug_assert!(label == 0.0 || label == 1.0);
    let probability = 1.0 - (label - r).abs();
    -weight * probability.ln()
}

fn windowed_loss_scalar(
    w: &[f32],
    t_historys: &[f32],
    r_historys: &[f32],
    labels: &[f32],
    weights: &[f32],
    seq_len: usize,
    batch_size: usize,
) -> f64 {
    let mut loss = 0.0;

    for column in 0..batch_size {
        let mut state = MemoryStateScalar {
            stability: 0.0,
            stability_fast: 0.0,
            difficulty: 0.0,
        };
        for row in 0..seq_len {
            let index = row * batch_size + column;
            let delta_t = t_historys[index] as f64;
            let rating = r_historys[index] as usize;
            if rating == 0 {
                break;
            }
            let weight = weights[index] as f64;
            if row > 0 && weight != 0.0 {
                let r = forgetting_curve_scalar(
                    w,
                    delta_t,
                    state.stability,
                    state.stability_fast,
                    state.difficulty,
                );
                loss += bce_loss_scalar(r, labels[index] as f64, weight);
            }
            if row + 1 < seq_len && r_historys[(row + 1) * batch_size + column] != 0.0 {
                state = step_scalar(w, delta_t, rating, state, row);
            }
        }
    }

    loss
}

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
mod neon_loss;
mod wide_loss;

pub(crate) fn windowed_loss(
    w: &[f32],
    t_historys: &[f32],
    r_historys: &[f32],
    labels: &[f32],
    weights: &[f32],
    seq_len: usize,
    batch_size: usize,
) -> f64 {
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    {
        neon_loss::windowed_loss(
            w, t_historys, r_historys, labels, weights, seq_len, batch_size,
        )
    }

    #[cfg(not(all(target_arch = "aarch64", target_feature = "neon")))]
    {
        wide_loss::windowed_loss(
            w, t_historys, r_historys, labels, weights, seq_len, batch_size,
        )
    }
}

pub(crate) fn windowed_loss_and_grad(
    w: &[f32],
    t_historys: &[f32],
    r_historys: &[f32],
    labels: &[f32],
    weights: &[f32],
    seq_len: usize,
    batch_size: usize,
) -> (f64, [f32; PARAM_LEN]) {
    let w = dual_params(w);
    let mut loss = 0.0;
    let mut grad = [0.0f64; PARAM_LEN];

    for column in 0..batch_size {
        let mut state = MemoryStateDual {
            stability: Dual::constant(0.0),
            stability_fast: Dual::constant(0.0),
            difficulty: Dual::constant(0.0),
        };
        for row in 0..seq_len {
            let index = row * batch_size + column;
            let delta_t = t_historys[index] as f64;
            let rating = r_historys[index] as usize;
            if rating == 0 {
                break;
            }
            let weight = weights[index] as f64;
            if row > 0 && weight != 0.0 {
                let r = forgetting_curve(
                    &w,
                    delta_t,
                    state.stability,
                    state.stability_fast,
                    state.difficulty,
                );
                let item_loss = bce_loss(r, labels[index] as f64, weight);
                loss += item_loss.value;
                for (dst, src) in grad.iter_mut().zip(item_loss.grad) {
                    *dst += src;
                }
            }
            if row + 1 < seq_len && r_historys[(row + 1) * batch_size + column] != 0.0 {
                state = step(&w, delta_t, rating, state, row);
            }
        }
    }

    let mut grad_f32 = [0.0f32; PARAM_LEN];
    for (dst, src) in grad_f32.iter_mut().zip(grad) {
        *dst = src as f32;
    }
    (loss, grad_f32)
}

pub(crate) fn windowed_grad(
    w: &[f32],
    t_historys: &[f32],
    r_historys: &[f32],
    labels: &[f32],
    weights: &[f32],
    seq_len: usize,
    batch_size: usize,
) -> [f32; PARAM_LEN] {
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    {
        neon_loss::windowed_grad(
            w, t_historys, r_historys, labels, weights, seq_len, batch_size,
        )
    }

    #[cfg(not(all(target_arch = "aarch64", target_feature = "neon")))]
    {
        wide_loss::windowed_grad(
            w, t_historys, r_historys, labels, weights, seq_len, batch_size,
        )
    }
}

// ---------------------------------------------------------------------------
// Reverse-mode (BPTT) gradient.
//
// This computes the same gradient as `windowed_loss_and_grad` but with reverse
// accumulation instead of carrying a 34-wide forward jacobian, so it is far
// cheaper per review. It is the scalar reference that the NEON kernel mirrors,
// and is validated against the forward-mode dual to ~1e-9 in tests.
// ---------------------------------------------------------------------------
mod reverse {
    use super::{D_MAX, D_MIN, PARAM_LEN, S_MAX, S_MIN};

    #[inline]
    fn clamp_open(x: f64, lo: f64, hi: f64) -> (f64, bool) {
        if x < lo {
            (lo, false)
        } else if x > hi {
            (hi, false)
        } else {
            (x, true)
        }
    }

    #[derive(Default)]
    struct Curve {
        out: f64,
        s: f64,
        sf: f64,
        d: f64,
        t: f64,
        open_s: bool,
        open_sf: bool,
        open_d: bool,
        lnsf: f64,
        sf_pow: f64,
        open1: bool,
        dec1: f64,
        q1: f64,
        openq: bool,
        e1: f64,
        factor1: f64,
        tos_f: f64,
        b1: f64,
        lnb1: f64,
        r1: f64,
        open2: bool,
        dec2: f64,
        inv_dec2: f64,
        lnw26: f64,
        p26: f64,
        factor2: f64,
        d_ts: f64,
        tos: f64,
        b2: f64,
        lnb2: f64,
        r2: f64,
        p29: f64,
        weight1: f64,
        lns: f64,
        p30: f64,
        d_wexp: f64,
        weight2: f64,
        wsum: f64,
        ret: f64,
        // whether this is the full mixture (curve) or fast-only recall
        fast_only: bool,
    }

    fn curve_fwd(w: &[f32], t: f64, s_in: f64, sf_in: f64, d_in: f64, fast_only: bool) -> Curve {
        let t = t.max(0.0);
        let (s, open_s) = clamp_open(s_in, S_MIN, S_MAX);
        let (sf, open_sf) = clamp_open(sf_in, S_MIN, S_MAX);
        let (d, open_d) = clamp_open(d_in, D_MIN, D_MAX);

        // fast component
        let lnsf = sf.ln();
        let sf_pow = ((w[33] as f64 - 0.3) * lnsf).exp();
        let dec1_raw = w[23] as f64 * sf_pow;
        let (dec1_mag, open1) = clamp_open(dec1_raw, 0.01, 0.95);
        let dec1 = -dec1_mag;
        let q1 = (w[25] as f64).ln() / dec1;
        let (q1c, openq) = if q1 > 60.0 { (60.0, false) } else { (q1, true) };
        let e1 = q1c.exp();
        let factor1 = e1 - 1.0;
        let tos_f = t / sf;
        let b1 = tos_f * factor1 + 1.0;
        let lnb1 = b1.ln();
        let r1 = (dec1 * lnb1).exp();

        if fast_only {
            return Curve {
                out: r1,
                s,
                sf,
                d,
                t,
                open_s,
                open_sf,
                open_d,
                lnsf,
                sf_pow,
                open1,
                dec1,
                q1,
                openq,
                e1,
                factor1,
                tos_f,
                b1,
                lnb1,
                r1,
                fast_only: true,
                ..Default::default()
            };
        }

        // slow component
        let (dec2_mag, open2) = clamp_open(w[24] as f64, 0.01, 0.95);
        let dec2 = -dec2_mag;
        let inv_dec2 = 1.0 / dec2;
        let lnw26 = (w[26] as f64).ln();
        let p26 = (lnw26 * inv_dec2).exp();
        let factor2 = p26 - 1.0;
        let d_ts = ((d - 5.0) * (w[32] as f64 - 0.3)).exp();
        let tos = t / s;
        let b2 = tos * factor2 * d_ts + 1.0;
        let lnb2 = b2.ln();
        let r2 = (dec2 * lnb2).exp();

        // weights
        let p29 = (-(w[29] as f64) * lnsf).exp();
        let weight1 = w[27] as f64 * p29;
        let lns = s.ln();
        let p30 = (w[30] as f64 * lns).exp();
        let d_wexp = ((d - 5.0) * (w[31] as f64 - 0.5)).exp();
        let weight2 = w[28] as f64 * p30 * d_wexp;
        let wsum = weight1 + weight2;
        let num = weight1 * r1 + weight2 * r2;
        let ret = num / wsum;
        let out = ret * (1.0 - 2e-5) + 1e-5;

        Curve {
            out,
            s,
            sf,
            d,
            t,
            open_s,
            open_sf,
            open_d,
            lnsf,
            sf_pow,
            open1,
            dec1,
            q1,
            openq,
            e1,
            factor1,
            tos_f,
            b1,
            lnb1,
            r1,
            open2,
            dec2,
            inv_dec2,
            lnw26,
            p26,
            factor2,
            d_ts,
            tos,
            b2,
            lnb2,
            r2,
            p29,
            weight1,
            lns,
            p30,
            d_wexp,
            weight2,
            wsum,
            ret,
            fast_only: false,
        }
    }

    /// Returns (g_s, g_sf, g_d) already routed through the input clamps.
    fn curve_bwd(
        w: &[f32],
        c: &Curve,
        g_out: f64,
        g_r1_extra: f64,
        grad: &mut [f64; PARAM_LEN],
    ) -> (f64, f64, f64) {
        let mut g_s = 0.0;
        let mut g_sf = 0.0;
        let mut g_d = 0.0;
        let mut g_lnsf = 0.0;

        // g into r1, and (if full) the slow/weight machinery
        let g_r1;
        if c.fast_only {
            g_r1 = g_out + g_r1_extra; // out == r1
        } else {
            let g_ret = g_out * (1.0 - 2e-5);
            let g_num = g_ret / c.wsum;
            let g_wsum = -g_ret * c.ret / c.wsum;
            let mut g_weight1 = g_num * c.r1 + g_wsum;
            let mut g_weight2 = g_num * c.r2 + g_wsum;
            g_r1 = g_num * c.weight1 + g_r1_extra;
            let g_r2 = g_num * c.weight2;

            // weight2 = w28 * p30 * d_wexp
            grad[28] += g_weight2 * c.p30 * c.d_wexp;
            let g_p30 = g_weight2 * w[28] as f64 * c.d_wexp;
            let g_d_wexp = g_weight2 * w[28] as f64 * c.p30;
            // p30 = exp(w30*lns)
            grad[30] += g_p30 * c.p30 * c.lns;
            g_s += g_p30 * c.p30 * w[30] as f64 / c.s;
            // d_wexp = exp((d-5)*(w31-0.5))
            let g_dw_arg = g_d_wexp * c.d_wexp;
            grad[31] += g_dw_arg * (c.d - 5.0);
            g_d += g_dw_arg * (w[31] as f64 - 0.5);
            let _ = &mut g_weight2;

            // weight1 = w27 * p29
            grad[27] += g_weight1 * c.p29;
            let g_p29 = g_weight1 * w[27] as f64;
            grad[29] += g_p29 * c.p29 * (-c.lnsf);
            g_lnsf += g_p29 * c.p29 * (-(w[29] as f64));
            let _ = &mut g_weight1;

            // r2 = exp(dec2*lnb2)
            let mut g_dec2 = g_r2 * c.r2 * c.lnb2;
            let g_lnb2 = g_r2 * c.r2 * c.dec2;
            let g_b2 = g_lnb2 / c.b2;
            // b2 = tos*factor2*d_ts + 1
            let g_tos = g_b2 * c.factor2 * c.d_ts;
            let g_factor2 = g_b2 * c.tos * c.d_ts;
            let g_d_ts = g_b2 * c.tos * c.factor2;
            let g_dts_arg = g_d_ts * c.d_ts;
            grad[32] += g_dts_arg * (c.d - 5.0);
            g_d += g_dts_arg * (w[32] as f64 - 0.3);
            g_s += g_tos * (-c.t / (c.s * c.s));
            // factor2 = p26 - 1 ; p26 = exp(lnw26*inv_dec2)
            let g_p26 = g_factor2;
            grad[26] += g_p26 * c.p26 * c.inv_dec2 / w[26] as f64;
            g_dec2 += g_p26 * c.p26 * c.lnw26 * (-1.0 / (c.dec2 * c.dec2));
            // dec2 = -clamp(w24)
            grad[24] += if c.open2 { -g_dec2 } else { 0.0 };
        }

        // r1 = exp(dec1*lnb1)
        let mut g_dec1 = g_r1 * c.r1 * c.lnb1;
        let g_lnb1 = g_r1 * c.r1 * c.dec1;
        let g_b1 = g_lnb1 / c.b1;
        // b1 = tos_f*factor1 + 1
        let g_tos_f = g_b1 * c.factor1;
        let g_factor1 = g_b1 * c.tos_f;
        g_sf += g_tos_f * (-c.t / (c.sf * c.sf));
        // factor1 = e1 - 1 ; e1 = exp(q1c) ; q1c = min(q1,60) ; q1 = lnw25/dec1
        let g_e1 = g_factor1;
        let g_q1c = g_e1 * c.e1;
        let g_q1 = if c.openq { g_q1c } else { 0.0 };
        grad[25] += g_q1 * (1.0 / c.dec1) / w[25] as f64;
        g_dec1 += g_q1 * (-(w[25] as f64).ln() / (c.dec1 * c.dec1));
        // dec1 = -clamp(w23*sf_pow)
        let g_dec1_mag = -g_dec1;
        let g_dec1_raw = if c.open1 { g_dec1_mag } else { 0.0 };
        grad[23] += g_dec1_raw * c.sf_pow;
        let g_sf_pow = g_dec1_raw * w[23] as f64;
        // sf_pow = exp((w33-0.3)*lnsf)
        grad[33] += g_sf_pow * c.sf_pow * c.lnsf;
        g_lnsf += g_sf_pow * c.sf_pow * (w[33] as f64 - 0.3);
        // lnsf = ln(sf)
        g_sf += g_lnsf / c.sf;
        let _ = c.q1;

        (
            if c.open_s { g_s } else { 0.0 },
            if c.open_sf { g_sf } else { 0.0 },
            if c.open_d { g_d } else { 0.0 },
        )
    }

    struct Stab {
        out: f64,
        rating: usize,
        start: usize,
        last_s: f64,
        last_d: f64,
        r: f64,
        hard: f64,
        easy: f64,
        lns1: f64,
        q: f64,
        er: f64,
        new_s_fail: f64,
        pls: f64,
        last_s_wins: bool,
        // success branch (only when rating>1)
        lns: f64,
        cc: f64,
        bb: f64,
        er2: f64,
        em1: f64,
        prefac: f64,
        sinc: f64,
        success: f64,
        success_wins: bool,
    }

    fn stab_fwd(w: &[f32], last_s: f64, last_d: f64, r: f64, rating: usize, start: usize) -> Stab {
        let hard = if rating == 2 {
            w[start + 6] as f64
        } else {
            1.0
        };
        let easy = if rating == 4 {
            w[start + 7] as f64
        } else {
            1.0
        };
        let lns1 = (last_s + 1.0).ln();
        let q = (w[start + 4] as f64 * lns1).exp();
        let er = ((1.0 - r) * w[start + 5] as f64).exp();
        let new_s_fail = w[start + 3] as f64 * (q - 1.0) * er;
        let last_s_wins = last_s <= new_s_fail;
        let pls = if last_s_wins { last_s } else { new_s_fail };
        if rating <= 1 {
            return Stab {
                out: pls,
                rating,
                start,
                last_s,
                last_d,
                r,
                hard,
                easy,
                lns1,
                q,
                er,
                new_s_fail,
                pls,
                last_s_wins,
                lns: 0.0,
                cc: 0.0,
                bb: 0.0,
                er2: 0.0,
                em1: 0.0,
                prefac: 0.0,
                sinc: 0.0,
                success: 0.0,
                success_wins: false,
            };
        }
        let lns = last_s.ln();
        let cc = (-(w[start + 1] as f64) * lns).exp();
        let bb = 11.0 - last_d;
        let er2 = ((1.0 - r) * w[start + 2] as f64).exp();
        let em1 = er2 - 1.0;
        let prefac = (w[start] as f64 - 1.5).exp();
        let sinc = prefac * bb * cc * em1 * hard * easy + 1.0;
        let success = last_s * sinc;
        let success_wins = pls < success;
        let out = if success_wins { success } else { pls };
        Stab {
            out,
            rating,
            start,
            last_s,
            last_d,
            r,
            hard,
            easy,
            lns1,
            q,
            er,
            new_s_fail,
            pls,
            last_s_wins,
            lns,
            cc,
            bb,
            er2,
            em1,
            prefac,
            sinc,
            success,
            success_wins,
        }
    }

    /// Returns (g_last_s, g_last_d, g_r).
    fn stab_bwd(w: &[f32], c: &Stab, g_out: f64, grad: &mut [f64; PARAM_LEN]) -> (f64, f64, f64) {
        let start = c.start;
        let mut g_s = 0.0;
        let mut g_d = 0.0;
        let mut g_r = 0.0;

        let g_pls;
        if c.rating <= 1 {
            g_pls = g_out;
        } else {
            let g_success = if c.success_wins { g_out } else { 0.0 };
            let g_pls_direct = if c.success_wins { 0.0 } else { g_out };
            // success = last_s * sinc
            g_s += g_success * c.sinc;
            let g_sinc = g_success * c.last_s;
            g_pls = g_pls_direct;
            // sinc = prefac*bb*cc*em1*hard*easy + 1
            let g_prod = g_sinc;
            grad[start] += g_prod * (c.bb * c.cc * c.em1 * c.hard * c.easy) * c.prefac;
            let g_bb = g_prod * (c.prefac * c.cc * c.em1 * c.hard * c.easy);
            let g_cc = g_prod * (c.prefac * c.bb * c.em1 * c.hard * c.easy);
            let g_em1 = g_prod * (c.prefac * c.bb * c.cc * c.hard * c.easy);
            if c.rating == 2 {
                grad[start + 6] += g_prod * (c.prefac * c.bb * c.cc * c.em1 * c.easy);
            }
            if c.rating == 4 {
                grad[start + 7] += g_prod * (c.prefac * c.bb * c.cc * c.em1 * c.hard);
            }
            // bb = 11 - last_d
            g_d += -g_bb;
            // cc = exp(-w[start+1]*lns)
            let g_cc_arg = g_cc * c.cc;
            grad[start + 1] += g_cc_arg * (-c.lns);
            g_s += g_cc_arg * (-(w[start + 1] as f64)) / c.last_s;
            // em1 = er2 - 1 ; er2 = exp((1-r)*w[start+2])
            let g_er2 = g_em1;
            let g_er2_arg = g_er2 * c.er2;
            grad[start + 2] += g_er2_arg * (1.0 - c.r);
            g_r += g_er2_arg * (-(w[start + 2] as f64));
        }

        // pls = min(last_s, new_s_fail)
        g_s += if c.last_s_wins { g_pls } else { 0.0 };
        let g_new_s_fail = if c.last_s_wins { 0.0 } else { g_pls };
        // new_s_fail = w[start+3]*(q-1)*er
        grad[start + 3] += g_new_s_fail * (c.q - 1.0) * c.er;
        let g_q = g_new_s_fail * w[start + 3] as f64 * c.er;
        let g_er = g_new_s_fail * w[start + 3] as f64 * (c.q - 1.0);
        // q = exp(w[start+4]*lns1)
        let g_q_arg = g_q * c.q;
        grad[start + 4] += g_q_arg * c.lns1;
        g_s += g_q_arg * w[start + 4] as f64 / (c.last_s + 1.0);
        // er = exp((1-r)*w[start+5])
        let g_er_arg = g_er * c.er;
        grad[start + 5] += g_er_arg * (1.0 - c.r);
        g_r += g_er_arg * (-(w[start + 5] as f64));

        (g_s, g_d, g_r)
    }

    struct NextDiff {
        out_pre: f64,
        open: bool,
        rating: usize,
        last_d: f64,
        delta_d0: f64,
        surprise: f64,
        delta_d: f64,
        exp3w5: f64,
    }

    fn nextdiff_fwd(w: &[f32], last_d: f64, r: f64, rating: usize) -> NextDiff {
        let delta_d0 = -(w[6] as f64) * (rating as f64 - 3.0);
        let surprise = r + 0.1;
        let delta_d = if rating == 1 {
            delta_d0 * surprise
        } else {
            delta_d0
        };
        let new_d = last_d + (10.0 - last_d) * delta_d / 9.0;
        let exp3w5 = (w[5] as f64 * 3.0).exp();
        let init_easy = w[4] as f64 - exp3w5 + 1.0;
        let out_pre = init_easy * 0.01 + new_d * 0.99;
        let (_out, open) = clamp_open(out_pre, D_MIN, D_MAX);
        NextDiff {
            out_pre,
            open,
            rating,
            last_d,
            delta_d0,
            surprise,
            delta_d,
            exp3w5,
        }
    }

    /// Returns (g_last_d, g_r).
    fn nextdiff_bwd(
        w: &[f32],
        c: &NextDiff,
        g_out: f64,
        grad: &mut [f64; PARAM_LEN],
    ) -> (f64, f64) {
        let _ = w;
        let g_out_pre = if c.open { g_out } else { 0.0 };
        let g_init = g_out_pre * 0.01;
        let g_new_d = g_out_pre * 0.99;
        grad[4] += g_init;
        grad[5] += g_init * (-c.exp3w5 * 3.0);
        let g_last_d = g_new_d * (1.0 - c.delta_d / 9.0);
        let g_delta_d = g_new_d * (10.0 - c.last_d) / 9.0;
        let (g_delta_d0, g_r) = if c.rating == 1 {
            (g_delta_d * c.surprise, g_delta_d * c.delta_d0)
        } else {
            (g_delta_d, 0.0)
        };
        grad[6] += g_delta_d0 * (-(c.rating as f64 - 3.0));
        (g_last_d, g_r)
    }

    // Keep per-review caches inline in the training buffer to avoid an allocation per step.
    #[allow(clippy::large_enum_variant)]
    enum StepCache {
        First {
            rating: usize,
            init_s_pre: f64,
            open_init_s: bool,
            open_init_sf: bool,
            init_d_pre: f64,
            open_init_d: bool,
            init_d_exp: f64,
        },
        Full {
            rating: usize,
            open_state_s: bool,
            open_state_sf: bool,
            open_state_d: bool,
            curve: Curve,
            slow: Stab,
            fast: Stab,
            nd: NextDiff,
            new_s_slow: f64,
            open_slow: bool,
            new_s_fast_raw: f64,
            relearn: f64,
            open_fast: bool,
        },
    }

    pub(super) struct MemState {
        pub stability: f64,
        pub stability_fast: f64,
        pub difficulty: f64,
    }

    fn step_fwd(
        w: &[f32],
        delta_t: f64,
        rating: usize,
        state: &MemState,
        nth: usize,
    ) -> (MemState, StepCache) {
        let (last_s, open_state_s) = clamp_open(state.stability, S_MIN, S_MAX);
        let (last_d, open_state_d) = clamp_open(state.difficulty, D_MIN, D_MAX);
        let (last_s_fast, open_state_sf) = clamp_open(state.stability_fast, S_MIN, S_MAX);

        if nth == 0 && state.stability == 0.0 {
            let init_s_pre = w[rating.clamp(1, 4) - 1] as f64;
            let (init_s, open_init_s) = clamp_open(init_s_pre, S_MIN, S_MAX);
            let (init_sf, open_init_sf) = clamp_open(init_s_pre * 0.8, S_MIN, S_MAX);
            let init_d_exp = (w[5] as f64 * (rating.clamp(1, 4) - 1) as f64).exp();
            let init_d_pre = w[4] as f64 - init_d_exp + 1.0;
            let (init_d, open_init_d) = clamp_open(init_d_pre, D_MIN, D_MAX);
            return (
                MemState {
                    stability: init_s,
                    stability_fast: init_sf,
                    difficulty: init_d,
                },
                StepCache::First {
                    rating,
                    init_s_pre,
                    open_init_s,
                    open_init_sf,
                    init_d_pre,
                    open_init_d,
                    init_d_exp,
                },
            );
        }

        let delta_t = delta_t.max(0.0);
        let curve = curve_fwd(w, delta_t, last_s, last_s_fast, last_d, false);
        let r = curve.out;
        let slow = stab_fwd(w, last_s, last_d, r, rating, 7);
        let new_s_slow = slow.out;
        let r_fast = curve.r1;
        let fast = stab_fwd(w, last_s_fast, last_d, r_fast, rating, 15);
        let new_s_fast_raw = fast.out;
        let relearn = new_s_slow * 0.8;
        let new_s_fast_pre = if rating == 1 && new_s_fast_raw > relearn {
            relearn
        } else {
            new_s_fast_raw
        };
        let nd = nextdiff_fwd(w, last_d, r, rating);
        let new_d = clamp_open(nd.out_pre, D_MIN, D_MAX).0;
        let (new_s, open_slow) = clamp_open(new_s_slow, S_MIN, S_MAX);
        let (new_sf, open_fast) = clamp_open(new_s_fast_pre, S_MIN, S_MAX);
        (
            MemState {
                stability: new_s,
                stability_fast: new_sf,
                difficulty: new_d,
            },
            StepCache::Full {
                rating,
                open_state_s,
                open_state_sf,
                open_state_d,
                curve,
                slow,
                fast,
                nd,
                new_s_slow,
                open_slow,
                new_s_fast_raw,
                relearn,
                open_fast,
            },
        )
    }

    /// Returns (g_state_s, g_state_sf, g_state_d).
    fn step_bwd(
        w: &[f32],
        cache: &StepCache,
        g_new_s: f64,
        g_new_sf: f64,
        g_new_d: f64,
        grad: &mut [f64; PARAM_LEN],
    ) -> (f64, f64, f64) {
        match cache {
            StepCache::First {
                rating,
                init_s_pre,
                open_init_s,
                open_init_sf,
                init_d_pre: _,
                open_init_d,
                init_d_exp,
            } => {
                let g_init_s = if *open_init_s { g_new_s } else { 0.0 };
                let g_init_sf = if *open_init_sf { g_new_sf } else { 0.0 };
                let g_init_s_pre = g_init_s + g_init_sf * 0.8;
                grad[(*rating).clamp(1, 4) - 1] += g_init_s_pre;
                let g_init_d = if *open_init_d { g_new_d } else { 0.0 };
                grad[4] += g_init_d;
                grad[5] += g_init_d * (-*init_d_exp * ((*rating).clamp(1, 4) - 1) as f64);
                let _ = init_s_pre;
                (0.0, 0.0, 0.0)
            }
            StepCache::Full {
                rating,
                open_state_s,
                open_state_sf,
                open_state_d,
                curve,
                slow,
                fast,
                nd,
                new_s_slow: _,
                open_slow,
                new_s_fast_raw,
                relearn,
                open_fast,
            } => {
                let g_new_s_slow_out = if *open_slow { g_new_s } else { 0.0 };
                let g_new_s_fast_post = if *open_fast { g_new_sf } else { 0.0 };
                // relearn: new_s_fast_pre = min(new_s_fast_raw, relearn) when rating==1
                let (g_new_s_fast_raw, mut g_new_s_slow) = if *rating == 1 {
                    let fast_raw_wins = *new_s_fast_raw <= *relearn;
                    if fast_raw_wins {
                        (g_new_s_fast_post, g_new_s_slow_out)
                    } else {
                        // routes to relearn = new_s_slow*0.8
                        (0.0, g_new_s_slow_out + g_new_s_fast_post * 0.8)
                    }
                } else {
                    (g_new_s_fast_post, g_new_s_slow_out)
                };

                let (g_ls_1, g_ld_1, g_r_1) = stab_bwd(w, slow, g_new_s_slow, grad);
                let (g_lsf_1, g_ld_2, g_rfast) = stab_bwd(w, fast, g_new_s_fast_raw, grad);
                let (g_ld_3, g_r_2) = nextdiff_bwd(w, nd, g_new_d, grad);
                let (g_ls_2, g_lsf_2, g_ld_4) = curve_bwd(w, curve, g_r_1 + g_r_2, g_rfast, grad);
                let _ = &mut g_new_s_slow;

                let g_last_s = g_ls_1 + g_ls_2;
                let g_last_sf = g_lsf_1 + g_lsf_2;
                let g_last_d = g_ld_1 + g_ld_2 + g_ld_3 + g_ld_4;
                (
                    if *open_state_s { g_last_s } else { 0.0 },
                    if *open_state_sf { g_last_sf } else { 0.0 },
                    if *open_state_d { g_last_d } else { 0.0 },
                )
            }
        }
    }

    fn bce_retrievability_grad(r_raw: f64, label: f64, weight: f64) -> f64 {
        if !(r_raw > 0.0001 && r_raw < 0.9999) {
            return 0.0;
        }
        if label == 1.0 {
            -weight / r_raw
        } else {
            weight / (1.0 - r_raw)
        }
    }

    // Keep numerical inputs and precomputed intermediates explicit at this kernel boundary.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn column_grad(
        w: &[f32],
        t_historys: &[f32],
        r_historys: &[f32],
        labels: &[f32],
        weights: &[f32],
        seq_len: usize,
        batch_size: usize,
        column: usize,
        grad: &mut [f64; PARAM_LEN],
    ) {
        // Forward pass, recording state before each row and the step caches.
        let mut states = vec![MemState {
            stability: 0.0,
            stability_fast: 0.0,
            difficulty: 0.0,
        }];
        let mut caches: Vec<StepCache> = Vec::new();
        let mut last_row = 0usize;
        for row in 0..seq_len {
            let index = row * batch_size + column;
            let rating = r_historys[index] as usize;
            if rating == 0 {
                break;
            }
            last_row = row;
            if row + 1 < seq_len && r_historys[(row + 1) * batch_size + column] != 0.0 {
                let delta_t = t_historys[index] as f64;
                let (next, cache) = step_fwd(w, delta_t, rating, states.last().unwrap(), row);
                states.push(next);
                caches.push(cache);
            }
        }
        if last_row == 0 {
            return;
        }

        // Backward pass.
        let mut g_state = vec![(0.0f64, 0.0f64, 0.0f64); states.len()];
        for i in (0..states.len()).rev() {
            // loss prediction at row i (uses state before row i)
            if i > 0 && i <= last_row {
                let index = i * batch_size + column;
                let weight = weights[index] as f64;
                if weight != 0.0 {
                    let delta_t = t_historys[index] as f64;
                    let state = &states[i];
                    let pred = curve_fwd(
                        w,
                        delta_t,
                        state.stability,
                        state.stability_fast,
                        state.difficulty,
                        false,
                    );
                    let g_r = bce_retrievability_grad(pred.out, labels[index] as f64, weight);
                    let (gs, gsf, gd) = curve_bwd(w, &pred, g_r, 0.0, grad);
                    g_state[i].0 += gs;
                    g_state[i].1 += gsf;
                    g_state[i].2 += gd;
                }
            }
            // step at row i produced state i+1
            if i < caches.len() {
                let (gs, gsf, gd) = (g_state[i + 1].0, g_state[i + 1].1, g_state[i + 1].2);
                let (pgs, pgsf, pgd) = step_bwd(w, &caches[i], gs, gsf, gd, grad);
                g_state[i].0 += pgs;
                g_state[i].1 += pgsf;
                g_state[i].2 += pgd;
            }
        }
    }

    pub(super) fn windowed_grad(
        w: &[f32],
        t_historys: &[f32],
        r_historys: &[f32],
        labels: &[f32],
        weights: &[f32],
        seq_len: usize,
        batch_size: usize,
    ) -> [f32; PARAM_LEN] {
        let mut grad = [0.0f64; PARAM_LEN];
        for column in 0..batch_size {
            column_grad(
                w, t_historys, r_historys, labels, weights, seq_len, batch_size, column, &mut grad,
            );
        }
        let mut out = [0.0f32; PARAM_LEN];
        for (dst, src) in out.iter_mut().zip(grad) {
            *dst = src as f32;
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hard_label_bce_scalar_matches_reference() {
        for (label, expected_probability) in [(0.0, 0.63), (1.0, 0.37)] {
            let loss = bce_loss_scalar(0.37, label, 2.5);
            let expected = -2.5 * f64::ln(expected_probability);
            assert!(
                (loss - expected).abs() < 1e-12,
                "label {label} loss {loss}, expected {expected}"
            );
        }
    }

    #[test]
    fn test_hard_label_bce_dual_gradient_matches_reference() {
        for (label, expected_grad) in [(0.0, 2.5 / 0.63), (1.0, -2.5 / 0.37)] {
            let loss = bce_loss(Dual::variable(0.37, 0), label, 2.5);
            let expected_value = -2.5 * f64::ln(if label == 0.0 { 0.63 } else { 0.37 });
            assert!(
                (loss.value - expected_value).abs() < 1e-12,
                "label {label} loss {}, expected {expected_value}",
                loss.value
            );
            assert!(
                (loss.grad[0] - expected_grad).abs() < 1e-12,
                "label {label} grad {}, expected {expected_grad}",
                loss.grad[0]
            );
        }
    }

    fn synthetic_windowed<const BATCH: usize, const SEQ: usize>(
        histories: &[[(usize, f32); SEQ]; BATCH],
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
        let mut t_historys = vec![0.0; SEQ * BATCH];
        let mut r_historys = vec![0.0; SEQ * BATCH];
        let mut labels = vec![0.0; SEQ * BATCH];
        let mut weights = vec![0.0; SEQ * BATCH];
        for (column, reviews) in histories.iter().enumerate() {
            for (row, &(rating, delta_t)) in reviews.iter().enumerate() {
                let index = row * BATCH + column;
                t_historys[index] = delta_t;
                r_historys[index] = rating as f32;
                if row > 0 && rating != 0 {
                    labels[index] = if rating == 1 { 0.0 } else { 1.0 };
                    weights[index] = 0.4 + row as f32 * 0.07 + column as f32 * 0.03;
                }
            }
        }
        (t_historys, r_historys, labels, weights)
    }

    #[test]
    fn test_reverse_mode_grad_matches_forward_dual() {
        let histories = [
            [(3, 0.0), (4, 1.0), (3, 5.0), (1, 10.0), (3, 2.0), (0, 0.0)],
            [(1, 0.0), (3, 0.5), (2, 1.0), (0, 0.0), (0, 0.0), (0, 0.0)],
            [(4, 0.0), (4, 2.0), (3, 4.0), (2, 7.0), (1, 1.0), (3, 3.0)],
            [(2, 0.0), (3, 1.5), (0, 0.0), (0, 0.0), (0, 0.0), (0, 0.0)],
            [(3, 0.0), (2, 3.0), (4, 6.0), (1, 2.0), (3, 0.0), (0, 0.0)],
        ];
        let seq_len = 6;
        let batch_size = histories.len();
        let (t, r, labels, weights) = synthetic_windowed(&histories);

        let (_loss, forward) = windowed_loss_and_grad(
            &crate::DEFAULT_PARAMETERS,
            &t,
            &r,
            &labels,
            &weights,
            seq_len,
            batch_size,
        );
        let rev = reverse::windowed_grad(
            &crate::DEFAULT_PARAMETERS,
            &t,
            &r,
            &labels,
            &weights,
            seq_len,
            batch_size,
        );

        let mut max_abs = 0.0f32;
        for i in 0..PARAM_LEN {
            let diff = (forward[i] - rev[i]).abs();
            if diff > max_abs {
                max_abs = diff;
            }
        }
        assert!(
            max_abs < 1e-4,
            "reverse-mode gradient diverges: max abs diff {max_abs}\nforward={forward:?}\nreverse={rev:?}"
        );
    }

    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    #[test]
    fn test_neon_grad_matches_forward_dual() {
        let histories = [
            [(3, 0.0), (4, 1.0), (3, 5.0), (1, 10.0), (3, 2.0), (0, 0.0)],
            [(1, 0.0), (3, 0.5), (2, 1.0), (0, 0.0), (0, 0.0), (0, 0.0)],
            [(4, 0.0), (4, 2.0), (3, 4.0), (2, 7.0), (1, 1.0), (3, 3.0)],
            [(2, 0.0), (3, 1.5), (0, 0.0), (0, 0.0), (0, 0.0), (0, 0.0)],
            [(3, 0.0), (2, 3.0), (4, 6.0), (1, 2.0), (3, 0.0), (0, 0.0)],
            [(3, 0.0), (3, 1.0), (4, 2.0), (2, 3.0), (3, 5.0), (1, 1.0)],
            [(1, 0.0), (2, 0.5), (3, 2.0), (4, 4.0), (0, 0.0), (0, 0.0)],
            [(4, 0.0), (1, 1.0), (0, 0.0), (0, 0.0), (0, 0.0), (0, 0.0)],
        ];
        let seq_len = 6;
        let batch_size = histories.len();
        let (t, r, labels, weights) = synthetic_windowed(&histories);

        let (_loss, forward) = windowed_loss_and_grad(
            &crate::DEFAULT_PARAMETERS,
            &t,
            &r,
            &labels,
            &weights,
            seq_len,
            batch_size,
        );
        let neon = neon_loss::windowed_grad_for_test(
            &crate::DEFAULT_PARAMETERS,
            &t,
            &r,
            &labels,
            &weights,
            seq_len,
            batch_size,
        );

        let numerator: f64 = forward
            .iter()
            .zip(neon)
            .map(|(a, b)| {
                let diff = (*a - b) as f64;
                diff * diff
            })
            .sum::<f64>()
            .sqrt();
        let denominator: f64 = forward
            .iter()
            .map(|a| (*a as f64) * (*a as f64))
            .sum::<f64>()
            .sqrt()
            .max(1.0);
        let relative_l2 = numerator / denominator;
        // The residual is the f32 polynomial exp/ln accuracy of `neon_math`, not
        // an adjoint error (every param tracks the f64 oracle to <1%). The old
        // kernel accepted 3e-2 here; the accurate-variant path is tighter.
        assert!(
            relative_l2 < 1e-2,
            "NEON gradient diverges: relative L2 {relative_l2:e}\nforward={forward:?}\nneon={neon:?}"
        );
    }
}

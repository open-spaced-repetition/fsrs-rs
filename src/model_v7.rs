use crate::inference::MemoryState;

pub(super) const PARAM_LEN: usize = 34;
const DR_MIN: f32 = 0.0001;
const DR_MAX: f32 = 0.9999;
const INTERVAL_NEWTON_ITERS: usize = 7;
const BISECTION_ITERS: usize = 50;
const MIN_T: f32 = 1.0 / 86_400.0;

pub(crate) fn fsrs7_forgetting_curve_scalar(w: &[f32], t: f32, s: f32) -> f32 {
    fsrs7_forgetting_curve_scalar_for_state(
        w,
        t,
        MemoryState {
            stability: s,
            difficulty: 5.0,
            stability_fast: s,
        },
    )
}

pub(crate) fn init_difficulty_scalar(w: &[f32], rating: usize) -> f32 {
    w[4] - (w[5] * (rating - 1) as f32).exp() + 1.0
}

fn linear_damping_scalar(delta_d: f32, old_d: f32) -> f32 {
    (10.0 - old_d) * delta_d / 9.0
}

fn mean_reversion_scalar(init: f32, current: f32) -> f32 {
    init * 0.01 + current * 0.99
}

pub(crate) fn next_difficulty_scalar(w: &[f32], d: f32, rating: usize) -> f32 {
    next_difficulty_scalar_for_retention(w, d, rating, 1.0)
}

pub(crate) fn next_difficulty_scalar_for_retention(
    w: &[f32],
    d: f32,
    rating: usize,
    retention: f32,
) -> f32 {
    let rating = rating.clamp(1, 4);
    let delta_d = -w[6] * (rating as f32 - 3.0);
    let delta_d = if rating == 1 {
        delta_d * (retention + 0.1)
    } else {
        delta_d
    };
    let new_d = d + linear_damping_scalar(delta_d, d);
    mean_reversion_scalar(init_difficulty_scalar(w, 4), new_d).clamp(super::D_MIN, super::D_MAX)
}

fn stability_for_set_scalar(
    w: &[f32],
    last_s: f32,
    last_d: f32,
    retrievability: f32,
    rating: usize,
    start: usize,
) -> f32 {
    let rating = rating.clamp(1, 4);
    let hard_penalty = if rating == 2 { w[start + 6] } else { 1.0 };
    let easy_bonus = if rating == 4 { w[start + 7] } else { 1.0 };
    let new_s_fail = w[start + 3]
        * ((last_s + 1.0).powf(w[start + 4]) - 1.0)
        * ((1.0 - retrievability) * w[start + 5]).exp();
    let pls = last_s.min(new_s_fail);
    let sinc = (w[start] - 1.5).exp()
        * (11.0 - last_d)
        * last_s.powf(-w[start + 1])
        * (((1.0 - retrievability) * w[start + 2]).exp() - 1.0)
        * hard_penalty
        * easy_bonus
        + 1.0;
    let new_s_success = pls.max(last_s * sinc);
    if rating > 1 { new_s_success } else { pls }.clamp(super::S_MIN, super::S_MAX)
}

pub(crate) fn stability_after_success_scalar(
    w: &[f32],
    s: f32,
    r: f32,
    d: f32,
    rating: usize,
    _delta_t: f32,
) -> f32 {
    stability_for_set_scalar(w, s, d, r, rating, 7)
}

pub(crate) fn stability_after_failure_scalar(
    w: &[f32],
    s: f32,
    r: f32,
    d: f32,
    _delta_t: f32,
) -> f32 {
    stability_for_set_scalar(w, s, d, r, 1, 7)
}

pub(crate) fn stability_short_term_scalar(w: &[f32], s: f32, r: f32, d: f32, rating: usize) -> f32 {
    stability_for_set_scalar(w, s, d, r, rating, 7)
}

fn fast_component_recall_scalar(w: &[f32], t: f32, s_fast: f32) -> f32 {
    let t = t.max(0.0);
    let s_fast = s_fast.clamp(super::S_MIN, super::S_MAX);
    let decay1_mag = (w[23] * s_fast.powf(w[33] - 0.3)).clamp(0.01, 0.95);
    let decay1 = -decay1_mag;
    let factor1 = ((w[25].ln() / decay1).min(60.0)).exp() - 1.0;
    (1.0 + factor1 * (t / s_fast)).powf(decay1)
}

pub(crate) fn fsrs7_next_state_scalar(
    w: &[f32],
    state: MemoryState,
    delta_t: f32,
    rating: usize,
) -> MemoryState {
    let delta_t = delta_t.max(0.0);
    let rating = rating.clamp(1, 4);
    let retrievability = fsrs7_forgetting_curve_scalar_for_state(w, delta_t, state);
    let new_s_slow = stability_for_set_scalar(
        w,
        state.stability,
        state.difficulty,
        retrievability,
        rating,
        7,
    );
    let r_fast = fast_component_recall_scalar(w, delta_t, state.stability_fast);
    let new_s_fast = stability_for_set_scalar(
        w,
        state.stability_fast,
        state.difficulty,
        r_fast,
        rating,
        15,
    );
    let new_s_fast = if rating == 1 {
        new_s_fast.min(new_s_slow * 0.8)
    } else {
        new_s_fast
    };
    let new_d = next_difficulty_scalar_for_retention(w, state.difficulty, rating, retrievability);
    MemoryState {
        stability: new_s_slow,
        difficulty: new_d,
        stability_fast: new_s_fast.clamp(super::S_MIN, super::S_MAX),
    }
}

pub(crate) fn fsrs7_forgetting_curve_scalar_for_state(
    w: &[f32],
    t: f32,
    state: MemoryState,
) -> f32 {
    let t = t.max(0.0);
    let s = state.stability.max(super::S_MIN);
    let s_fast = state.stability_fast.max(super::S_MIN);
    let d = state.difficulty.clamp(super::D_MIN, super::D_MAX);

    let decay1_mag = (w[23] * s_fast.powf(w[33] - 0.3)).clamp(0.01, 0.95);
    let decay1 = -decay1_mag;
    let factor1 = ((w[25].ln() / decay1).min(60.0)).exp() - 1.0;
    let r1 = (1.0 + factor1 * (t / s_fast)).powf(decay1);

    let decay2 = -w[24].clamp(0.01, 0.95);
    let factor2 = w[26].powf(1.0 / decay2) - 1.0;
    let d_timescale = ((d - 5.0) * (w[32] - 0.3)).exp();
    let r2 = (1.0 + factor2 * d_timescale * (t / s)).powf(decay2);

    let weight1 = w[27] * s_fast.powf(-w[29]);
    let weight2 = w[28] * s.powf(w[30]) * ((d - 5.0) * (w[31] - 0.5)).exp();
    let retention = (weight1 * r1 + weight2 * r2) / (weight1 + weight2);
    retention.mul_add(1.0 - 2e-5, 1e-5)
}

pub(crate) fn fsrs7_forgetting_curve_and_derivative_scalar(
    w: &[f32],
    t: f32,
    state: MemoryState,
) -> (f32, f32) {
    let t = t.max(0.0);
    let s = state.stability.max(super::S_MIN);
    let s_fast = state.stability_fast.max(super::S_MIN);
    let d = state.difficulty.clamp(super::D_MIN, super::D_MAX);

    let decay1_mag = (w[23] * s_fast.powf(w[33] - 0.3)).clamp(0.01, 0.95);
    let decay1 = -decay1_mag;
    let factor1 = ((w[25].ln() / decay1).min(60.0)).exp() - 1.0;
    let b1 = 1.0 + factor1 * (t / s_fast);
    let r1 = b1.powf(decay1);
    let dr1_dt = decay1 * b1.powf(decay1 - 1.0) * factor1 / s_fast;

    let decay2 = -w[24].clamp(0.01, 0.95);
    let factor2 = w[26].powf(1.0 / decay2) - 1.0;
    let d_timescale = ((d - 5.0) * (w[32] - 0.3)).exp();
    let b2 = 1.0 + factor2 * d_timescale * (t / s);
    let r2 = b2.powf(decay2);
    let dr2_dt = decay2 * b2.powf(decay2 - 1.0) * factor2 * d_timescale / s;

    let weight1 = w[27] * s_fast.powf(-w[29]);
    let weight2 = w[28] * s.powf(w[30]) * ((d - 5.0) * (w[31] - 0.5)).exp();
    let weight_sum = (weight1 + weight2).max(1e-9);
    let retention = (weight1 * r1 + weight2 * r2) / weight_sum;
    let derivative = (weight1 * dr1_dt + weight2 * dr2_dt) / weight_sum;
    (
        retention.mul_add(1.0 - 2e-5, 1e-5),
        derivative * (1.0 - 2e-5),
    )
}

pub(super) fn fsrs7_next_interval_bisection_scalar(
    w: &[f32],
    stability: f32,
    desired_retention: f32,
    _high_hint: Option<f32>,
) -> f32 {
    fsrs7_next_interval_bisection_scalar_for_state(
        w,
        MemoryState {
            stability,
            difficulty: 5.0,
            stability_fast: stability,
        },
        desired_retention,
    )
}

fn fsrs7_next_interval_bisection_scalar_for_state(
    w: &[f32],
    state: MemoryState,
    desired_retention: f32,
) -> f32 {
    let desired_retention = desired_retention.clamp(DR_MIN, DR_MAX);
    if desired_retention >= DR_MAX {
        return 0.0;
    }
    let mut low = 0.0;
    let mut high = state.stability.max(state.stability_fast).max(1.0);
    while fsrs7_forgetting_curve_scalar_for_state(w, high, state) > desired_retention
        && high < super::S_MAX
    {
        high = (high * 2.0).min(super::S_MAX);
    }
    for _ in 0..BISECTION_ITERS {
        let mid = (low + high) * 0.5;
        if fsrs7_forgetting_curve_scalar_for_state(w, mid, state) > desired_retention {
            low = mid;
        } else {
            high = mid;
        }
    }
    ((low + high) * 0.5).clamp(0.0, super::S_MAX)
}

pub(super) struct Fsrs7S90Lut;

pub(super) fn fsrs7_s90_lut(_w: &[f32]) -> Option<Fsrs7S90Lut> {
    Some(Fsrs7S90Lut)
}

pub(crate) struct Fsrs7Runtime {
    w: Vec<f32>,
}

impl Fsrs7Runtime {
    pub(crate) fn new(w: &[f32]) -> Self {
        Self { w: w.to_vec() }
    }

    pub(crate) fn forgetting_curve(&self, t: f32, stability: f32) -> f32 {
        fsrs7_forgetting_curve_scalar(&self.w, t, stability)
    }

    pub(crate) fn next_interval(&self, stability: f32, desired_retention: f32) -> f32 {
        fsrs7_next_interval_scalar_for_state(
            &self.w,
            MemoryState {
                stability,
                difficulty: 5.0,
                stability_fast: stability,
            },
            desired_retention,
        )
    }
}

pub(super) fn fsrs7_next_interval_scalar(
    w: &[f32],
    stability: f32,
    desired_retention: f32,
    _lut: &Fsrs7S90Lut,
) -> f32 {
    fsrs7_next_interval_scalar_for_state(
        w,
        MemoryState {
            stability,
            difficulty: 5.0,
            stability_fast: stability,
        },
        desired_retention,
    )
}

pub(crate) fn fsrs7_next_interval_scalar_for_state(
    w: &[f32],
    state: MemoryState,
    desired_retention: f32,
) -> f32 {
    let desired_retention = desired_retention.clamp(DR_MIN, DR_MAX);
    if desired_retention >= DR_MAX {
        return 0.0;
    }

    let state = MemoryState {
        stability: state.stability.clamp(super::S_MIN, super::S_MAX),
        difficulty: state.difficulty.clamp(super::D_MIN, super::D_MAX),
        stability_fast: state.stability_fast.clamp(super::S_MIN, super::S_MAX),
    };
    let min_log_t = MIN_T.ln();
    let max_log_t = super::S_MAX.ln();
    let mut log_t = state.stability.max(state.stability_fast).max(MIN_T).ln();
    for _ in 0..INTERVAL_NEWTON_ITERS {
        log_t = log_t.clamp(min_log_t, max_log_t);
        let t = log_t.exp().clamp(MIN_T, super::S_MAX);
        let (retrievability, derivative) =
            fsrs7_forgetting_curve_and_derivative_scalar(w, t, state);
        let df_du = (derivative * t).min(-1e-12);
        let step = ((retrievability - desired_retention) / df_du).clamp(-4.0, 4.0);
        log_t -= step;
        if !log_t.is_finite() {
            return fsrs7_next_interval_bisection_scalar_for_state(w, state, desired_retention);
        }
    }
    let interval = log_t.exp().clamp(0.0, super::S_MAX);
    let retrievability = fsrs7_forgetting_curve_scalar_for_state(w, interval, state);
    if retrievability.is_finite() && (retrievability - desired_retention).abs() <= 1e-3 {
        interval
    } else {
        fsrs7_next_interval_bisection_scalar_for_state(w, state, desired_retention)
    }
}

use crate::inference::MemoryState;
use crate::model::model_v7::{
    fsrs7_forgetting_curve_and_derivative_scalar, fsrs7_next_interval_scalar_for_state,
    fsrs7_next_state_scalar, init_difficulty_scalar,
};
use crate::simulation::{S_MAX, S_MIN};

pub(crate) const PARAM_LEN: usize = 34;
pub(crate) const PENALTY_W_1: f64 = 0.5;
pub(crate) const PENALTY_W_2: f64 = 0.0015;
pub(crate) const PENALTY_W_L2: f64 = 0.5;
pub(crate) const PENALTY_N_REVIEWS: usize = 10;
pub(crate) const PENALTY_TARGET_DR: f32 = 0.90;
pub(crate) const PENALTY_TARGET_DRS: [f32; 1] = [0.99];
pub(crate) const PENALTY_N_NEWTON: usize = 7;
pub(crate) const MIN_T: f32 = 1.0 / 86400.0;
pub(crate) const MAX_T: f32 = 36500.0;
pub(crate) const ONE_DAY: f32 = 1.0;
pub(crate) const SHORT_C: f32 = 600.0 / 86400.0;
pub(crate) const INV_C: f32 = 1.0 / SHORT_C;
pub(crate) const GRAD_LEN: usize = 34;
pub(crate) const PARAMS_STDDEV: [f32; 34] = [
    9999.0, 9999.0, 9999.0, 9999.0, 0.523, 0.2528, 0.4329, 0.2966, 0.2139, 0.2889, 0.1862, 0.175,
    0.3812, 0.3013, 0.9104, 0.3234, 0.2448, 0.3273, 0.1842, 0.1735, 0.4608, 0.311, 0.864, 0.0418,
    0.2596, 0.0798, 0.0682, 0.1282, 0.1397, 0.1407, 0.1489, 0.2, 0.15, 0.15,
];

pub(crate) fn l2_penalty_value_and_grad(
    w: &[f32],
    init_w: &[f32],
    batch_size: usize,
    total_size: usize,
    l2_weight: f64,
    params_stddev: &[f32],
) -> (f64, Vec<f32>) {
    let mut grad = vec![0.0f32; w.len()];
    if total_size == 0 {
        return (0.0, grad);
    }
    let size = w.len().min(init_w.len()).min(params_stddev.len());
    let scale = l2_weight * batch_size as f64 / total_size as f64;
    let mut penalty_sum = 0.0f64;
    for i in 0..size {
        let sigma = params_stddev[i] as f64;
        let denom = sigma * sigma;
        let diff = w[i] as f64 - init_w[i] as f64;
        penalty_sum += diff * diff / denom;
        grad[i] = (2.0 * diff / denom * scale) as f32;
    }
    let penalty = penalty_sum * scale;
    if !penalty.is_finite() {
        return (0.0, vec![0.0; w.len()]);
    }
    for g in &mut grad {
        if !g.is_finite() {
            *g = 0.0;
        }
    }
    (penalty, grad)
}

// Keep Dual35 local to FSRS-7 training: the penalty objective and its gradients
// are expressed against FSRS-7's fixed 34-parameter layout and index mapping.
#[derive(Clone, Copy, Debug)]
struct Dual35 {
    value: f64,
    grad: [f64; GRAD_LEN],
}

impl Dual35 {
    fn constant(value: f64) -> Self {
        Self {
            value,
            grad: [0.0; GRAD_LEN],
        }
    }

    fn variable(value: f64, idx: usize) -> Self {
        let mut grad = [0.0; GRAD_LEN];
        if idx < GRAD_LEN {
            grad[idx] = 1.0;
        }
        Self { value, grad }
    }

    fn add(self, rhs: Self) -> Self {
        let mut grad = [0.0; GRAD_LEN];
        for (i, item) in grad.iter_mut().enumerate().take(GRAD_LEN) {
            *item = self.grad[i] + rhs.grad[i];
        }
        Self {
            value: self.value + rhs.value,
            grad,
        }
    }

    fn sub(self, rhs: Self) -> Self {
        let mut grad = [0.0; GRAD_LEN];
        for (i, item) in grad.iter_mut().enumerate().take(GRAD_LEN) {
            *item = self.grad[i] - rhs.grad[i];
        }
        Self {
            value: self.value - rhs.value,
            grad,
        }
    }

    fn neg(self) -> Self {
        self.mul_const(-1.0)
    }

    fn mul(self, rhs: Self) -> Self {
        let mut grad = [0.0; GRAD_LEN];
        for (i, item) in grad.iter_mut().enumerate().take(GRAD_LEN) {
            *item = self.grad[i] * rhs.value + rhs.grad[i] * self.value;
        }
        Self {
            value: self.value * rhs.value,
            grad,
        }
    }

    fn div(self, rhs: Self) -> Self {
        let denom = (rhs.value * rhs.value).max(1e-18);
        let mut grad = [0.0; GRAD_LEN];
        for (i, item) in grad.iter_mut().enumerate().take(GRAD_LEN) {
            *item = (self.grad[i] * rhs.value - self.value * rhs.grad[i]) / denom;
        }
        Self {
            value: self.value / rhs.value,
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
        Self {
            value: self.value - rhs,
            grad: self.grad,
        }
    }

    fn const_sub(self, lhs: f64) -> Self {
        self.neg().add_const(lhs)
    }

    fn mul_const(self, rhs: f64) -> Self {
        let mut grad = [0.0; GRAD_LEN];
        for (i, item) in grad.iter_mut().enumerate().take(GRAD_LEN) {
            *item = self.grad[i] * rhs;
        }
        Self {
            value: self.value * rhs,
            grad,
        }
    }

    fn div_const(self, rhs: f64) -> Self {
        self.mul_const(1.0 / rhs)
    }

    fn exp(self) -> Self {
        let value = self.value.exp();
        let mut grad = [0.0; GRAD_LEN];
        for (i, item) in grad.iter_mut().enumerate().take(GRAD_LEN) {
            *item = self.grad[i] * value;
        }
        Self { value, grad }
    }

    fn log(self) -> Self {
        let mut grad = [0.0; GRAD_LEN];
        for (i, item) in grad.iter_mut().enumerate().take(GRAD_LEN) {
            *item = self.grad[i] / self.value;
        }
        Self {
            value: self.value.ln(),
            grad,
        }
    }

    fn powf(self, exp: f64) -> Self {
        let value = self.value.powf(exp);
        let coeff = exp * self.value.powf(exp - 1.0);
        let mut grad = [0.0; GRAD_LEN];
        for (i, item) in grad.iter_mut().enumerate().take(GRAD_LEN) {
            *item = self.grad[i] * coeff;
        }
        Self { value, grad }
    }

    fn powi(self, exp: i32) -> Self {
        let value = self.value.powi(exp);
        let coeff = (exp as f64) * self.value.powi(exp - 1);
        let mut grad = [0.0; GRAD_LEN];
        for (i, item) in grad.iter_mut().enumerate().take(GRAD_LEN) {
            *item = self.grad[i] * coeff;
        }
        Self { value, grad }
    }

    fn pow(self, exp: Self) -> Self {
        let base = self.clamp_min(1e-12);
        exp.mul(base.log()).exp()
    }

    fn clamp_min(self, min: f64) -> Self {
        if self.value < min {
            Self::constant(min)
        } else {
            self
        }
    }

    fn clamp_max(self, max: f64) -> Self {
        if self.value > max {
            Self::constant(max)
        } else {
            self
        }
    }

    fn clamp(self, min: f64, max: f64) -> Self {
        self.clamp_min(min).clamp_max(max)
    }

    fn min(self, rhs: Self) -> Self {
        if self.value <= rhs.value { self } else { rhs }
    }

    fn max(self, rhs: Self) -> Self {
        if self.value >= rhs.value { self } else { rhs }
    }
}

fn dual_weights(w: &[f32]) -> [Dual35; GRAD_LEN] {
    std::array::from_fn(|i| Dual35::variable(w[i] as f64, i))
}

fn fsrs7_fc_r_and_drdt_scalar(t: f64, s: f64, w: &[f32]) -> (f64, f64) {
    let s_safe = s.max(1e-12);
    let decay1 = -(w[27] as f64);
    let decay2 = -(w[28] as f64);
    let base1 = (w[29] as f64).max(1e-4);
    let base2 = (w[30] as f64).max(1e-4);
    let bw1 = (w[31] as f64).max(1e-4);
    let bw2 = (w[32] as f64).max(1e-4);
    let swp1 = w[33] as f64;
    let swp2 = w[30] as f64;

    let c1 = base1.powf(1.0 / decay1) - 1.0;
    let c2 = base2.powf(1.0 / decay2) - 1.0;
    let tos = t / s_safe;
    let inner1 = (1.0 + c1 * tos).max(1e-9);
    let inner2 = (1.0 + c2 * tos).max(1e-9);
    let r1 = inner1.powf(decay1);
    let r2 = inner2.powf(decay2);

    let wt1 = bw1 * s_safe.powf(-swp1);
    let wt2 = bw2 * s_safe.powf(swp2);
    let wt_sum = (wt1 + wt2).max(1e-9);
    let r = ((wt1 * r1 + wt2 * r2) / wt_sum).clamp(0.0, 1.0);

    let dr1_dt = decay1 * inner1.powf(decay1 - 1.0) * (c1 / s_safe);
    let dr2_dt = decay2 * inner2.powf(decay2 - 1.0) * (c2 / s_safe);
    let dr_dt = ((wt1 * dr1_dt + wt2 * dr2_dt) / wt_sum).clamp(-1e9, 0.0);
    (r, dr_dt)
}

fn fsrs7_fc_r_dual(t: Dual35, s: Dual35, w: &[Dual35; GRAD_LEN]) -> Dual35 {
    let decay1 = w[27].neg();
    let decay2 = w[28].neg();
    let base1 = w[29].clamp_min(1e-4);
    let base2 = w[30].clamp_min(1e-4);
    let bw1 = w[31].clamp_min(1e-4);
    let bw2 = w[32].clamp_min(1e-4);
    let swp1 = w[33];
    let swp2 = w[30];

    let c1 = base1.pow(decay1.powi(-1)).sub_const(1.0);
    let c2 = base2.pow(decay2.powi(-1)).sub_const(1.0);
    let tos = t.div(s);
    let inner1 = c1.mul(tos).add_const(1.0).clamp_min(1e-9);
    let inner2 = c2.mul(tos).add_const(1.0).clamp_min(1e-9);

    let r1 = inner1.pow(decay1);
    let r2 = inner2.pow(decay2);

    let wt1 = bw1.mul(s.pow(swp1.neg()));
    let wt2 = bw2.mul(s.pow(swp2));
    let wt_sum = wt1.add(wt2).clamp_min(1e-9);
    wt1.mul(r1).add(wt2.mul(r2)).div(wt_sum).clamp(0.0, 1.0)
}

fn fsrs7_init_d_dual(rating: f64, w: &[Dual35; GRAD_LEN]) -> Dual35 {
    w[4].sub(w[5].mul_const(rating - 1.0).exp())
        .add_const(1.0)
        .clamp(1.0, 10.0)
}

fn fsrs7_next_d_good_dual(d: Dual35, init_d4: Dual35) -> Dual35 {
    init_d4
        .mul_const(0.01)
        .add(d.mul_const(0.99))
        .clamp(1.0, 10.0)
}

fn fsrs7_s_fail_long_dual(s: Dual35, d: Dual35, r: Dual35, w: &[Dual35; GRAD_LEN]) -> Dual35 {
    let raw = w[10]
        .mul(d.pow(w[11].neg()))
        .mul(s.add_const(1.0).pow(w[12]).sub_const(1.0))
        .mul(r.const_sub(1.0).mul(w[13]).exp());
    s.min(raw)
}

fn fsrs7_s_fail_short_dual(s: Dual35, d: Dual35, r: Dual35, w: &[Dual35; GRAD_LEN]) -> Dual35 {
    let raw = w[19]
        .mul(d.pow(w[20].neg()))
        .mul(s.add_const(1.0).pow(w[21]).sub_const(1.0))
        .mul(r.const_sub(1.0).mul(w[22]).exp());
    s.min(raw)
}

fn fsrs7_next_s_good_dual(s: Dual35, d: Dual35, delta_t: Dual35, w: &[Dual35; GRAD_LEN]) -> Dual35 {
    let r = fsrs7_fc_r_dual(delta_t, s, w).clamp(0.0001, 0.9999);

    let sf_l = fsrs7_s_fail_long_dual(s, d, r, w);
    let si_l = w[7]
        .sub_const(1.5)
        .exp()
        .mul(d.const_sub(11.0))
        .mul(s.pow(w[8].neg()))
        .mul(
            r.const_sub(1.0)
                .mul(w[9])
                .clamp_max(30.0)
                .exp()
                .sub_const(1.0),
        )
        .add_const(1.0);
    let s_lng = sf_l.max(s.mul(si_l));

    let sf_sh = fsrs7_s_fail_short_dual(s, d, r, w);
    let si_sh = w[16]
        .sub_const(1.5)
        .exp()
        .mul(d.const_sub(11.0))
        .mul(s.pow(w[17].neg()))
        .mul(
            r.const_sub(1.0)
                .mul(w[18])
                .clamp_max(30.0)
                .exp()
                .sub_const(1.0),
        )
        .add_const(1.0);
    let s_sht = sf_sh.max(s.mul(si_sh));

    let coef = Dual35::constant(1.0)
        .sub(w[26].mul(w[25].neg().mul(delta_t).exp()))
        .clamp(0.0, 1.0);
    coef.mul(s_lng)
        .add(Dual35::constant(1.0).sub(coef).mul(s_sht))
        .clamp(S_MIN as f64, S_MAX as f64)
}

fn fsrs7_interval_differentiable_dual(
    s: Dual35,
    target: f64,
    n_newton: usize,
    w: &[f32],
    w_dual: &[Dual35; GRAD_LEN],
) -> Dual35 {
    let s_f = s.value.max(1e-10);
    let d1 = -(w[27] as f64);
    let d2 = -(w[28] as f64);
    let b1 = (w[29] as f64).max(1e-4);
    let b2 = (w[30] as f64).max(1e-4);
    let bw1 = (w[31] as f64).max(1e-4);
    let bw2 = (w[32] as f64).max(1e-4);
    let sw1 = w[33] as f64;
    let sw2 = w[30] as f64;

    let c1 = b1.powf(1.0 / d1) - 1.0;
    let c2 = b2.powf(1.0 / d2) - 1.0;
    let wt1 = bw1 * s_f.powf(-sw1);
    let wt2 = bw2 * s_f.powf(sw2);
    let wts = (wt1 + wt2).max(1e-9);

    let mut u = s_f.ln();
    for _ in 0..n_newton {
        u = u.clamp((MIN_T as f64).ln(), (MAX_T as f64).ln());
        let t = u.exp().clamp(MIN_T as f64, MAX_T as f64);
        let tos = t / s_f;
        let i1 = (1.0 + c1 * tos).max(1e-9);
        let i2 = (1.0 + c2 * tos).max(1e-9);
        let r = (wt1 * i1.powf(d1) + wt2 * i2.powf(d2)) / wts;
        let dr1 = d1 * i1.powf(d1 - 1.0) * c1 / s_f;
        let dr2 = d2 * i2.powf(d2 - 1.0) * c2 / s_f;
        let drdt = (wt1 * dr1 + wt2 * dr2) / wts;
        let dfdu = (drdt * t).min(-1e-12);
        u -= (r - target) / dfdu;
    }

    let t_star = u.exp().clamp(MIN_T as f64, MAX_T as f64);
    let residual = fsrs7_fc_r_dual(Dual35::constant(t_star), s, w_dual).sub_const(target);
    let (_, drdt_s) = fsrs7_fc_r_and_drdt_scalar(t_star, s.value, w);
    let dfdu_s = (drdt_s * t_star).clamp(-1e9, -1e-9);
    Dual35::constant(t_star.ln())
        .sub(residual.div_const(dfdu_s))
        .clamp((MIN_T as f64).ln(), (MAX_T as f64).ln())
        .exp()
}

fn fsrs7_interval_growth_penalty_dual(
    w: &[f32],
    w_dual: &[Dual35; GRAD_LEN],
    n_reviews: usize,
    target_dr: f64,
    n_newton: usize,
) -> Dual35 {
    let mut s = w_dual[2].clamp(S_MIN as f64, S_MAX as f64);
    let init_d4 = fsrs7_init_d_dual(4.0, w_dual);
    let mut d = fsrs7_init_d_dual(3.0, w_dual);
    let mut prev_interval: Option<Dual35> = None;
    let mut best_ratio: Option<Dual35> = None;
    let mut best_val = f64::NEG_INFINITY;
    for _ in 0..n_reviews {
        let t = fsrs7_interval_differentiable_dual(s, target_dr, n_newton, w, w_dual);
        if let Some(prev) = prev_interval
            && prev.value >= ONE_DAY as f64
        {
            let ratio = t.div(prev);
            if ratio.value > best_val {
                best_val = ratio.value;
                best_ratio = Some(ratio);
            }
        }
        prev_interval = Some(t);
        s = fsrs7_next_s_good_dual(s, d, t, w_dual);
        d = fsrs7_next_d_good_dual(d, init_d4);
    }
    if let Some(ratio) = best_ratio {
        ratio.powf(2.0)
    } else {
        Dual35::constant(0.0)
    }
}

fn fsrs7_short_interval_penalty_dual(
    w: &[f32],
    w_dual: &[Dual35; GRAD_LEN],
    n_reviews: usize,
    n_newton: usize,
    target_drs: &[f32],
) -> Dual35 {
    let mut penalty_sum = Dual35::constant(0.0);
    let mut penalty_count = 0usize;
    for &target_dr in target_drs {
        let mut s = w_dual[2].clamp(S_MIN as f64, S_MAX as f64);
        let init_d4 = fsrs7_init_d_dual(4.0, w_dual);
        let mut d = fsrs7_init_d_dual(3.0, w_dual);
        let mut short_sum = Dual35::constant(0.0);
        let mut short_count = 0usize;
        for _ in 0..n_reviews {
            let t = fsrs7_interval_differentiable_dual(s, target_dr as f64, n_newton, w, w_dual);
            if t.value < ONE_DAY as f64 {
                short_sum = short_sum.add(t);
                short_count += 1;
            }
            s = fsrs7_next_s_good_dual(s, d, t, w_dual);
            d = fsrs7_next_d_good_dual(d, init_d4);
        }
        if short_count == 0 {
            continue;
        }
        let avg_t = short_sum
            .div_const(short_count as f64)
            .clamp_min(MIN_T as f64);
        let inv_x = avg_t.powf(-1.0);
        let penalty = inv_x.clamp_min(INV_C as f64).sub_const(INV_C as f64);
        penalty_sum = penalty_sum.add(penalty);
        penalty_count += 1;
    }
    if penalty_count == 0 {
        Dual35::constant(0.0)
    } else {
        penalty_sum.div_const(penalty_count as f64)
    }
}

#[derive(Clone, Copy)]
struct DualState {
    stability: Dual35,
    stability_fast: Dual35,
    difficulty: Dual35,
}

impl DualState {
    fn scalar(self) -> MemoryState {
        MemoryState {
            stability: self.stability.value as f32,
            stability_fast: self.stability_fast.value as f32,
            difficulty: self.difficulty.value as f32,
        }
    }
}

fn init_difficulty_dual(w: &[Dual35; GRAD_LEN], rating: f64) -> Dual35 {
    w[4].sub(w[5].mul_const(rating - 1.0).exp())
        .add_const(1.0)
        .clamp(1.0, 10.0)
}

fn initial_dual_state(w: &[Dual35; GRAD_LEN]) -> DualState {
    let stability = w[2].clamp(S_MIN as f64, S_MAX as f64);
    DualState {
        stability,
        stability_fast: stability.mul_const(0.8).clamp(S_MIN as f64, S_MAX as f64),
        difficulty: init_difficulty_dual(w, 3.0),
    }
}

fn fast_component_recall_dual(w: &[Dual35; GRAD_LEN], t: Dual35, stability_fast: Dual35) -> Dual35 {
    let t = t.clamp_min(0.0);
    let stability_fast = stability_fast.clamp(S_MIN as f64, S_MAX as f64);
    let decay1_mag = w[23]
        .mul(stability_fast.pow(w[33].sub_const(0.3)))
        .clamp(0.01, 0.95);
    let decay1 = decay1_mag.neg();
    let factor1 = w[25].log().div(decay1).clamp_max(60.0).exp().sub_const(1.0);
    t.div(stability_fast)
        .mul(factor1)
        .add_const(1.0)
        .pow(decay1)
}

fn forgetting_curve_dual(w: &[Dual35; GRAD_LEN], t: Dual35, state: DualState) -> Dual35 {
    let t = t.clamp_min(0.0);
    let stability = state.stability.clamp(S_MIN as f64, S_MAX as f64);
    let stability_fast = state.stability_fast.clamp(S_MIN as f64, S_MAX as f64);
    let difficulty = state.difficulty.clamp(1.0, 10.0);

    let decay1_mag = w[23]
        .mul(stability_fast.pow(w[33].sub_const(0.3)))
        .clamp(0.01, 0.95);
    let decay1 = decay1_mag.neg();
    let factor1 = w[25].log().div(decay1).clamp_max(60.0).exp().sub_const(1.0);
    let r1 = t
        .div(stability_fast)
        .mul(factor1)
        .add_const(1.0)
        .pow(decay1);

    let decay2 = w[24].clamp(0.01, 0.95).neg();
    let factor2 = w[26].pow(Dual35::constant(1.0).div(decay2)).sub_const(1.0);
    let d_timescale = difficulty.sub_const(5.0).mul(w[32].sub_const(0.3)).exp();
    let r2 = t
        .div(stability)
        .mul(factor2)
        .mul(d_timescale)
        .add_const(1.0)
        .pow(decay2);

    let weight1 = w[27].mul(stability_fast.pow(w[29].neg()));
    let weight2 = w[28]
        .mul(stability.pow(w[30]))
        .mul(difficulty.sub_const(5.0).mul(w[31].sub_const(0.5)).exp());
    weight1
        .mul(r1)
        .add(weight2.mul(r2))
        .div(weight1.add(weight2).clamp_min(1e-9))
        .mul_const(1.0 - 2e-5)
        .add_const(1e-5)
}

fn stability_for_set_dual(
    w: &[Dual35; GRAD_LEN],
    last_s: Dual35,
    last_d: Dual35,
    retrievability: Dual35,
    rating: usize,
    start: usize,
) -> Dual35 {
    let hard_penalty = if rating == 2 {
        w[start + 6]
    } else {
        Dual35::constant(1.0)
    };
    let easy_bonus = if rating == 4 {
        w[start + 7]
    } else {
        Dual35::constant(1.0)
    };
    let new_s_fail = w[start + 3]
        .mul(last_s.add_const(1.0).pow(w[start + 4]).sub_const(1.0))
        .mul(retrievability.const_sub(1.0).mul(w[start + 5]).exp());
    let pls = last_s.min(new_s_fail);
    if rating <= 1 {
        return pls.clamp(S_MIN as f64, S_MAX as f64);
    }
    let sinc = w[start]
        .sub_const(1.5)
        .exp()
        .mul(last_d.const_sub(11.0))
        .mul(last_s.pow(w[start + 1].neg()))
        .mul(
            retrievability
                .const_sub(1.0)
                .mul(w[start + 2])
                .exp()
                .sub_const(1.0),
        )
        .mul(hard_penalty)
        .mul(easy_bonus)
        .add_const(1.0);
    pls.max(last_s.mul(sinc)).clamp(S_MIN as f64, S_MAX as f64)
}

fn next_difficulty_dual(
    w: &[Dual35; GRAD_LEN],
    difficulty: Dual35,
    rating: usize,
    retention: Dual35,
) -> Dual35 {
    let rating_f = rating.clamp(1, 4) as f64;
    let delta_d = w[6].neg().mul_const(rating_f - 3.0);
    let delta_d = if rating == 1 {
        delta_d.mul(retention.add_const(0.1))
    } else {
        delta_d
    };
    let new_d = difficulty.add(difficulty.const_sub(10.0).mul(delta_d).div_const(9.0));
    init_difficulty_dual(w, 4.0)
        .mul_const(0.01)
        .add(new_d.mul_const(0.99))
        .clamp(1.0, 10.0)
}

fn next_state_dual(
    w: &[Dual35; GRAD_LEN],
    state: DualState,
    delta_t: Dual35,
    rating: usize,
) -> DualState {
    let rating = rating.clamp(1, 4);
    let retrievability = forgetting_curve_dual(w, delta_t, state);
    let new_s_slow = stability_for_set_dual(
        w,
        state.stability,
        state.difficulty,
        retrievability,
        rating,
        7,
    );
    let r_fast = fast_component_recall_dual(w, delta_t, state.stability_fast);
    let new_s_fast = stability_for_set_dual(
        w,
        state.stability_fast,
        state.difficulty,
        r_fast,
        rating,
        15,
    );
    let new_s_fast = if rating == 1 {
        new_s_fast.min(new_s_slow.mul_const(0.8))
    } else {
        new_s_fast
    };
    DualState {
        stability: new_s_slow,
        stability_fast: new_s_fast.clamp(S_MIN as f64, S_MAX as f64),
        difficulty: next_difficulty_dual(w, state.difficulty, rating, retrievability),
    }
}

fn next_interval_dual(
    w: &[f32],
    w_dual: &[Dual35; GRAD_LEN],
    state: DualState,
    desired_retention: f32,
) -> Dual35 {
    let scalar_state = state.scalar();
    let interval = fsrs7_next_interval_scalar_for_state(w, scalar_state, desired_retention)
        .clamp(MIN_T, S_MAX);
    let interval_dual = Dual35::constant(interval as f64);
    let residual =
        forgetting_curve_dual(w_dual, interval_dual, state).sub_const(desired_retention as f64);
    let (_, drdt) = fsrs7_forgetting_curve_and_derivative_scalar(w, interval, scalar_state);
    let dfdu = ((drdt * interval) as f64).clamp(-1e9, -1e-9);
    let mut lifted = Dual35::constant((interval as f64).ln())
        .sub(residual.div_const(dfdu))
        .clamp((MIN_T as f64).ln(), (S_MAX as f64).ln())
        .exp();
    lifted.value = interval as f64;
    lifted
}

fn interval_growth_penalty_dual(
    w: &[f32],
    w_dual: &[Dual35; GRAD_LEN],
    n_reviews: usize,
    target_dr: f32,
) -> Dual35 {
    let mut state = initial_dual_state(w_dual);
    let mut prev_interval: Option<Dual35> = None;
    let mut best_ratio: Option<Dual35> = None;
    let mut best_value = f64::NEG_INFINITY;
    for _ in 0..n_reviews {
        let interval = next_interval_dual(w, w_dual, state, target_dr);
        if let Some(prev) = prev_interval
            && prev.value >= ONE_DAY as f64
        {
            let ratio = interval.div(prev);
            if ratio.value > best_value {
                best_value = ratio.value;
                best_ratio = Some(ratio);
            }
        }
        prev_interval = Some(interval);
        state = next_state_dual(w_dual, state, interval, 3);
    }
    best_ratio.map_or(Dual35::constant(0.0), |ratio| ratio.powf(2.0))
}

fn short_interval_penalty_dual(
    w: &[f32],
    w_dual: &[Dual35; GRAD_LEN],
    n_reviews: usize,
    target_drs: &[f32],
) -> Dual35 {
    let mut penalty_sum = Dual35::constant(0.0);
    let mut penalty_count = 0usize;
    for &target_dr in target_drs {
        let mut state = initial_dual_state(w_dual);
        let mut short_sum = Dual35::constant(0.0);
        let mut short_count = 0usize;
        for _ in 0..n_reviews {
            let interval = next_interval_dual(w, w_dual, state, target_dr);
            if interval.value < ONE_DAY as f64 {
                short_sum = short_sum.add(interval);
                short_count += 1;
            }
            state = next_state_dual(w_dual, state, interval, 3);
        }
        if short_count == 0 {
            continue;
        }
        let avg_interval = short_sum
            .div_const(short_count as f64)
            .clamp_min(MIN_T as f64);
        let penalty = avg_interval
            .powf(-1.0)
            .clamp_min(INV_C as f64)
            .sub_const(INV_C as f64);
        penalty_sum = penalty_sum.add(penalty);
        penalty_count += 1;
    }
    if penalty_count == 0 {
        Dual35::constant(0.0)
    } else {
        penalty_sum.div_const(penalty_count as f64)
    }
}

fn schedule_penalty_dual(w: &[f32]) -> Dual35 {
    let w_dual = dual_weights(w);
    let p1 = interval_growth_penalty_dual(
        &w[..PARAM_LEN],
        &w_dual,
        PENALTY_N_REVIEWS,
        PENALTY_TARGET_DR,
    );
    let p2 = short_interval_penalty_dual(
        &w[..PARAM_LEN],
        &w_dual,
        PENALTY_N_REVIEWS,
        &PENALTY_TARGET_DRS,
    );
    p1.mul_const(PENALTY_W_1).add(p2.mul_const(PENALTY_W_2))
}

pub(crate) fn schedule_penalty_value_and_grad(
    w: &[f32],
    batch_size: usize,
) -> (f64, [f64; GRAD_LEN]) {
    if w.len() < PARAM_LEN || batch_size == 0 {
        return (0.0, [0.0; GRAD_LEN]);
    }
    let value = schedule_penalty_value(w);
    if !value.is_finite() {
        return (0.0, [0.0; GRAD_LEN]);
    }
    let penalty = schedule_penalty_dual(w);
    if !penalty.value.is_finite() {
        return (0.0, [0.0; GRAD_LEN]);
    }
    let scale = batch_size as f64;
    let mut grad = [0.0; GRAD_LEN];
    for (dst, src) in grad.iter_mut().zip(penalty.grad) {
        if src.is_finite() {
            *dst = src * scale;
        }
    }
    (value * scale, grad)
}

pub(crate) fn maybe_schedule_penalty_value_and_grad(
    w: &[f32],
    batch_size: usize,
    enable_sched_penalties: bool,
) -> (f64, [f64; GRAD_LEN]) {
    if enable_sched_penalties {
        schedule_penalty_value_and_grad(w, batch_size)
    } else {
        (0.0, [0.0; GRAD_LEN])
    }
}

fn initial_penalty_state(w: &[f32]) -> MemoryState {
    let stability = w[2].clamp(S_MIN, S_MAX);
    MemoryState {
        stability,
        difficulty: init_difficulty_scalar(w, 3),
        stability_fast: (stability * 0.8).clamp(S_MIN, S_MAX),
    }
}

fn schedule_penalty_value(w: &[f32]) -> f64 {
    if w.len() < PARAM_LEN {
        return 0.0;
    }
    let p1 = interval_growth_penalty_value(w, PENALTY_N_REVIEWS, PENALTY_TARGET_DR);
    let p2 = short_interval_penalty_value(w, PENALTY_N_REVIEWS, &PENALTY_TARGET_DRS);
    let penalty = p1 * PENALTY_W_1 + p2 * PENALTY_W_2;
    if penalty.is_finite() { penalty } else { 0.0 }
}

fn interval_growth_penalty_value(w: &[f32], n_reviews: usize, target_dr: f32) -> f64 {
    let mut state = initial_penalty_state(w);
    let mut prev_interval: Option<f32> = None;
    let mut best_ratio = 0.0_f64;
    for _ in 0..n_reviews {
        let interval = fsrs7_next_interval_scalar_for_state(w, state, target_dr);
        if let Some(prev) = prev_interval
            && prev >= ONE_DAY
            && interval.is_finite()
            && prev.is_finite()
        {
            best_ratio = best_ratio.max((interval / prev) as f64);
        }
        prev_interval = Some(interval);
        state = fsrs7_next_state_scalar(w, state, interval, 3);
    }
    best_ratio * best_ratio
}

fn short_interval_penalty_value(w: &[f32], n_reviews: usize, target_drs: &[f32]) -> f64 {
    let mut penalty_sum = 0.0_f64;
    let mut penalty_count = 0_usize;
    for &target_dr in target_drs {
        let mut state = initial_penalty_state(w);
        let mut short_sum = 0.0_f64;
        let mut short_count = 0_usize;
        for _ in 0..n_reviews {
            let interval = fsrs7_next_interval_scalar_for_state(w, state, target_dr);
            if interval.is_finite() && interval < ONE_DAY {
                short_sum += interval as f64;
                short_count += 1;
            }
            state = fsrs7_next_state_scalar(w, state, interval, 3);
        }
        if short_count > 0 {
            let avg_t = (short_sum / short_count as f64).max(MIN_T as f64);
            penalty_sum += avg_t.recip().max(INV_C as f64) - INV_C as f64;
            penalty_count += 1;
        }
    }
    if penalty_count == 0 {
        0.0
    } else {
        penalty_sum / penalty_count as f64
    }
}

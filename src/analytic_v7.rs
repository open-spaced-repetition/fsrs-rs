const PARAM_LEN: usize = 35;
const S_MIN: f64 = 0.0001;
const S_MAX: f64 = 36500.0;
const D_MIN: f64 = 1.0;
const D_MAX: f64 = 10.0;

#[derive(Clone, Copy, Debug)]
struct Dual35 {
    value: f64,
    grad: [f64; PARAM_LEN],
}

impl Dual35 {
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

    fn sub(self, rhs: Self) -> Self {
        let mut grad = [0.0; PARAM_LEN];
        for (index, value) in grad.iter_mut().enumerate() {
            *value = self.grad[index] - rhs.grad[index];
        }
        Self {
            value: self.value - rhs.value,
            grad,
        }
    }

    fn sub_const(self, rhs: f64) -> Self {
        self.add_const(-rhs)
    }

    fn const_sub(self, lhs: f64) -> Self {
        Self::constant(lhs).sub(self)
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
}

#[derive(Clone, Copy)]
struct MemoryStateDual {
    stability: Dual35,
    difficulty: Dual35,
}

#[derive(Clone, Copy)]
struct MemoryStateScalar {
    stability: f64,
    difficulty: f64,
}

fn dual_params(w: &[f32]) -> [Dual35; PARAM_LEN] {
    core::array::from_fn(|index| Dual35::variable(w[index] as f64, index))
}

fn init_stability(w: &[Dual35; PARAM_LEN], rating: usize) -> Dual35 {
    w[rating.clamp(1, 4) - 1]
}

fn init_difficulty(w: &[Dual35; PARAM_LEN], rating: usize) -> Dual35 {
    let rating = rating.clamp(1, 4);
    w[4].sub((w[5].mul_const((rating - 1) as f64)).exp())
        .add_const(1.0)
}

fn forgetting_curve(w: &[Dual35; PARAM_LEN], t: f64, s: Dual35) -> Dual35 {
    let t_over_s = Dual35::constant(t.max(0.0)).div(s);
    let decay1 = w[27].neg();
    let decay2 = w[28].neg();
    let factor1 = w[29].pow(Dual35::constant(1.0).div(decay1)).sub_const(1.0);
    let factor2 = w[30].pow(Dual35::constant(1.0).div(decay2)).sub_const(1.0);
    let r1 = t_over_s.mul(factor1).add_const(1.0).pow(decay1);
    let r2 = t_over_s.mul(factor2).add_const(1.0).pow(decay2);
    let weight1 = w[31].mul(s.pow(w[33].neg()));
    let weight2 = w[32].mul(s.pow(w[34]));
    weight1
        .mul(r1)
        .add(weight2.mul(r2))
        .div(weight1.add(weight2))
}

fn stability_for_set(
    w: &[Dual35; PARAM_LEN],
    s: Dual35,
    r: Dual35,
    d: Dual35,
    rating: usize,
    start: usize,
) -> Dual35 {
    let hard_penalty = if rating == 2 {
        w[start + 7]
    } else {
        Dual35::constant(1.0)
    };
    let easy_bonus = if rating == 4 {
        w[start + 8]
    } else {
        Dual35::constant(1.0)
    };
    let new_s_fail = w[start + 3]
        .mul(d.pow(w[start + 4].neg()))
        .mul(s.add_const(1.0).pow(w[start + 5]).sub_const(1.0))
        .mul(r.const_sub(1.0).mul(w[start + 6]).exp());
    let pls = if s.value < new_s_fail.value {
        s
    } else {
        new_s_fail
    };
    if rating <= 1 {
        return pls;
    }
    let sinc = w[start]
        .sub_const(1.5)
        .exp()
        .mul(d.const_sub(11.0))
        .mul(s.pow(w[start + 1].neg()))
        .mul(r.const_sub(1.0).mul(w[start + 2]).exp().sub_const(1.0))
        .mul(hard_penalty)
        .mul(easy_bonus)
        .add_const(1.0);
    let success = s.mul(sinc);
    if pls.value > success.value {
        pls
    } else {
        success
    }
}

fn transition(w: &[Dual35; PARAM_LEN], delta_t: f64) -> Dual35 {
    w[26]
        .mul(w[25].neg().mul_const(delta_t.max(0.0)).exp())
        .const_sub(1.0)
}

fn next_difficulty(w: &[Dual35; PARAM_LEN], d: Dual35, rating: usize) -> Dual35 {
    let delta_d = w[6].neg().mul_const(rating as f64 - 3.0);
    let new_d = d.add(d.const_sub(10.0).mul(delta_d).mul_const(1.0 / 9.0));
    init_difficulty(w, 4)
        .mul_const(0.01)
        .add(new_d.mul_const(0.99))
        .clamp(D_MIN, D_MAX)
}

fn step(
    w: &[Dual35; PARAM_LEN],
    delta_t: f64,
    rating: usize,
    state: MemoryStateDual,
    nth: usize,
) -> MemoryStateDual {
    let last_s = state.stability.clamp(S_MIN, S_MAX);
    let last_d = state.difficulty.clamp(D_MIN, D_MAX);
    if rating == 0 {
        return MemoryStateDual {
            stability: last_s,
            difficulty: last_d,
        };
    }
    if nth == 0 && state.stability.value == 0.0 {
        return MemoryStateDual {
            stability: init_stability(w, rating).clamp(S_MIN, S_MAX),
            difficulty: init_difficulty(w, rating).clamp(D_MIN, D_MAX),
        };
    }

    let delta_t = delta_t.max(0.0);
    let r = forgetting_curve(w, delta_t, last_s);
    let long = stability_for_set(w, last_s, r, last_d, rating, 7);
    let short = stability_for_set(w, last_s, r, last_d, rating, 16);
    let coefficient = transition(w, delta_t);
    MemoryStateDual {
        stability: coefficient
            .mul(long)
            .add(coefficient.const_sub(1.0).mul(short))
            .clamp(S_MIN, S_MAX),
        difficulty: next_difficulty(w, last_d, rating),
    }
}

fn bce_loss(r: Dual35, label: f64, weight: f64) -> Dual35 {
    let r = r.clamp(0.0001, 0.9999);
    debug_assert!(label == 0.0 || label == 1.0);
    let probability = if label == 0.0 { r.const_sub(1.0) } else { r };
    probability.ln().mul_const(-weight)
}

fn init_stability_scalar(w: &[f32], rating: usize) -> f64 {
    w[rating.clamp(1, 4) - 1] as f64
}

fn init_difficulty_scalar(w: &[f32], rating: usize) -> f64 {
    let rating = rating.clamp(1, 4);
    w[4] as f64 - (w[5] as f64 * (rating - 1) as f64).exp() + 1.0
}

fn forgetting_curve_scalar(w: &[f32], t: f64, s: f64) -> f64 {
    let t_over_s = t.max(0.0) / s;
    let decay1 = -(w[27] as f64);
    let decay2 = -(w[28] as f64);
    let factor1 = (w[29] as f64).powf(1.0 / decay1) - 1.0;
    let factor2 = (w[30] as f64).powf(1.0 / decay2) - 1.0;
    let r1 = (1.0 + t_over_s * factor1).powf(decay1);
    let r2 = (1.0 + t_over_s * factor2).powf(decay2);
    let weight1 = w[31] as f64 * s.powf(-(w[33] as f64));
    let weight2 = w[32] as f64 * s.powf(w[34] as f64);
    (weight1 * r1 + weight2 * r2) / (weight1 + weight2)
}

fn stability_for_set_scalar(w: &[f32], s: f64, r: f64, d: f64, rating: usize, start: usize) -> f64 {
    let hard_penalty = if rating == 2 {
        w[start + 7] as f64
    } else {
        1.0
    };
    let easy_bonus = if rating == 4 {
        w[start + 8] as f64
    } else {
        1.0
    };
    let new_s_fail = w[start + 3] as f64
        * d.powf(-(w[start + 4] as f64))
        * ((s + 1.0).powf(w[start + 5] as f64) - 1.0)
        * ((1.0 - r) * w[start + 6] as f64).exp();
    let pls = s.min(new_s_fail);
    if rating <= 1 {
        return pls;
    }
    let sinc = ((w[start] as f64) - 1.5).exp()
        * (11.0 - d)
        * s.powf(-(w[start + 1] as f64))
        * (((1.0 - r) * w[start + 2] as f64).exp() - 1.0)
        * hard_penalty
        * easy_bonus
        + 1.0;
    pls.max(s * sinc)
}

fn transition_scalar(w: &[f32], delta_t: f64) -> f64 {
    1.0 - w[26] as f64 * (-(w[25] as f64) * delta_t.max(0.0)).exp()
}

fn next_difficulty_scalar(w: &[f32], d: f64, rating: usize) -> f64 {
    let delta_d = -(w[6] as f64) * (rating as f64 - 3.0);
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
    if rating == 0 {
        return MemoryStateScalar {
            stability: last_s,
            difficulty: last_d,
        };
    }
    if nth == 0 && state.stability == 0.0 {
        return MemoryStateScalar {
            stability: init_stability_scalar(w, rating).clamp(S_MIN, S_MAX),
            difficulty: init_difficulty_scalar(w, rating).clamp(D_MIN, D_MAX),
        };
    }

    let delta_t = delta_t.max(0.0);
    let r = forgetting_curve_scalar(w, delta_t, last_s);
    let long = stability_for_set_scalar(w, last_s, r, last_d, rating, 7);
    let short = stability_for_set_scalar(w, last_s, r, last_d, rating, 16);
    let coefficient = transition_scalar(w, delta_t);
    MemoryStateScalar {
        stability: (coefficient * long + (1.0 - coefficient) * short).clamp(S_MIN, S_MAX),
        difficulty: next_difficulty_scalar(w, last_d, rating),
    }
}

fn bce_loss_scalar(r: f64, label: f64, weight: f64) -> f64 {
    let r = r.clamp(0.0001, 0.9999);
    debug_assert!(label == 0.0 || label == 1.0);
    let probability = 1.0 - (label - r).abs();
    -weight * probability.ln()
}

pub(crate) fn windowed_loss(
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
            difficulty: 0.0,
        };
        for row in 0..seq_len {
            let index = row * batch_size + column;
            let delta_t = t_historys[index] as f64;
            let rating = r_historys[index] as usize;
            let weight = weights[index] as f64;
            if row > 0 && weight != 0.0 {
                let r = forgetting_curve_scalar(w, delta_t, state.stability);
                loss += bce_loss_scalar(r, labels[index] as f64, weight);
            }
            if row + 1 < seq_len {
                state = step_scalar(w, delta_t, rating, state, row);
            }
        }
    }

    loss
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
            stability: Dual35::constant(0.0),
            difficulty: Dual35::constant(0.0),
        };
        for row in 0..seq_len {
            let index = row * batch_size + column;
            let delta_t = t_historys[index] as f64;
            let rating = r_historys[index] as usize;
            let weight = weights[index] as f64;
            if row > 0 && weight != 0.0 {
                let r = forgetting_curve(&w, delta_t, state.stability);
                let item_loss = bce_loss(r, labels[index] as f64, weight);
                loss += item_loss.value;
                for (dst, src) in grad.iter_mut().zip(item_loss.grad) {
                    *dst += src;
                }
            }
            if row + 1 < seq_len {
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
            let loss = bce_loss(Dual35::variable(0.37, 0), label, 2.5);
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
}

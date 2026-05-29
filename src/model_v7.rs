use super::{Fsrs7Ops, Get, MemoryStateTensors, Model, VersionOps, tensor_max, tensor_min};
use crate::FSRSError;
use crate::error::Result;
use crate::inference::MemoryState;
use burn::tensor::{Tensor, backend::Backend};
use std::collections::HashMap;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::sync::{Arc, Mutex, OnceLock};

pub(super) const PARAM_LEN: usize = 35;
const S90_TARGET: f32 = 0.9;
const LUT_SIZE: usize = 256;
const BISECTION_ITERS: usize = 50;
const NEWTON_ITERS: usize = 12;
const INTERVAL_SOLVE_TOL: f32 = 1e-4;
const RETENTION_SOLVE_TOL: f32 = 1e-4;
const DR_MIN: f32 = 0.0001;
const DR_MAX: f32 = 0.9999;
const LUT_S_MIN: f32 = 0.03;
const LUT_S_MAX: f32 = 3000.0;

impl<B: Backend> VersionOps<B> for Fsrs7Ops {
    fn apply_freeze_short_term(initial_params: &mut [f32]) {
        // FSRS-7: disable short-term contribution by forcing transition weight to 0,
        // making coefficient == 1.0 for every delta_t.
        initial_params[26] = 0.0;
    }

    fn power_forgetting_curve(model: &Model<B>, t: Tensor<B, 1>, s: Tensor<B, 1>) -> Tensor<B, 1> {
        power_forgetting_curve(model, t, s)
    }

    fn next_interval(
        model: &Model<B>,
        stability: Tensor<B, 1>,
        desired_retention: Tensor<B, 1>,
    ) -> Tensor<B, 1> {
        next_interval(model, stability, desired_retention)
    }

    fn update_state(
        model: &Model<B>,
        delta_t: Tensor<B, 1>,
        rating: Tensor<B, 1>,
        last_s: Tensor<B, 1>,
        last_d: Tensor<B, 1>,
    ) -> MemoryStateTensors<B> {
        let delta_t = delta_t.clamp_min(0.0);
        let retrievability = power_forgetting_curve(model, delta_t.clone(), last_s.clone());
        let new_s_long_term = stability_for_set(
            model,
            last_s.clone(),
            last_d.clone(),
            retrievability.clone(),
            rating.clone(),
            7,
        );
        let new_s_short_term = stability_for_set(
            model,
            last_s.clone(),
            last_d.clone(),
            retrievability,
            rating.clone(),
            16,
        );
        let coefficient = transition_function(model, delta_t);
        // If short-term is disabled, w[26]=0 => coefficient=1, so short_weight=0.
        // That cancels the short-term branch and keeps only long-term stability.
        let short_weight = coefficient.clone().neg().add_scalar(1.0);
        let new_s = coefficient * new_s_long_term + short_weight * new_s_short_term;
        let new_d = next_difficulty(model, last_d, rating);
        MemoryStateTensors {
            stability: new_s,
            difficulty: new_d,
        }
    }

    fn memory_state_from_sm2_fsrs(
        model: &Model<B>,
        ease_factor: f32,
        interval: f32,
        sm2_retention: f32,
    ) -> Result<MemoryState> {
        memory_state_from_sm2_fsrs7(model, ease_factor, interval, sm2_retention)
    }

    fn interval_at_retrievability(
        model: &Model<B>,
        stability: f32,
        target_retrievability: f32,
    ) -> f32 {
        let stability = stability.max(super::S_MIN);
        let target_retrievability = target_retrievability.clamp(DR_MIN, DR_MAX);
        let w = model.w.val().to_data().to_vec::<f32>().unwrap();
        let lut = fsrs7_s90_lut(&w);
        fsrs7_next_interval_scalar(&w, stability, target_retrievability, lut.as_ref())
    }
}

pub(crate) fn fsrs7_forgetting_curve_scalar(w: &[f32], t: f32, s: f32) -> f32 {
    let s = s.max(super::S_MIN);
    let t_over_s = t.max(0.0) / s;

    let decay1 = -w[27];
    let decay2 = -w[28];
    let base1 = w[29];
    let base2 = w[30];

    let factor1 = base1.powf(1.0 / decay1) - 1.0;
    let factor2 = base2.powf(1.0 / decay2) - 1.0;
    let r1 = (1.0 + factor1 * t_over_s).powf(decay1);
    let r2 = (1.0 + factor2 * t_over_s).powf(decay2);

    let weight1 = w[31] * s.powf(-w[33]);
    let weight2 = w[32] * s.powf(w[34]);

    (weight1 * r1 + weight2 * r2) / (weight1 + weight2)
}

fn fsrs7_forgetting_curve_and_derivative_scalar(w: &[f32], t: f32, s: f32) -> (f32, f32) {
    let s = s.max(super::S_MIN);
    let t = t.max(0.0);
    let t_over_s = t / s;

    let decay1 = -w[27];
    let decay2 = -w[28];
    let base1 = w[29];
    let base2 = w[30];

    let factor1 = base1.powf(1.0 / decay1) - 1.0;
    let factor2 = base2.powf(1.0 / decay2) - 1.0;
    let x1 = 1.0 + factor1 * t_over_s;
    let x2 = 1.0 + factor2 * t_over_s;
    let r1 = x1.powf(decay1);
    let r2 = x2.powf(decay2);
    let dr1 = r1 * decay1 * factor1 / (s * x1);
    let dr2 = r2 * decay2 * factor2 / (s * x2);

    let weight1 = w[31] * s.powf(-w[33]);
    let weight2 = w[32] * s.powf(w[34]);
    let weight_sum = weight1 + weight2;

    (
        (weight1 * r1 + weight2 * r2) / weight_sum,
        (weight1 * dr1 + weight2 * dr2) / weight_sum,
    )
}

#[derive(Debug)]
pub(crate) struct Fsrs7S90Lut {
    log_s_min: f32,
    log_s_step: f32,
    t90_grid: Vec<f32>,
}

impl Fsrs7S90Lut {
    pub(crate) fn build(w: &[f32]) -> Self {
        let log_s_min = LUT_S_MIN.max(super::S_MIN).ln();
        let log_s_max = LUT_S_MAX.min(super::S_MAX).ln();
        let log_s_step = (log_s_max - log_s_min) / (LUT_SIZE - 1) as f32;
        let mut t90_grid = Vec::with_capacity(LUT_SIZE);
        for i in 0..LUT_SIZE {
            let log_s = log_s_min + i as f32 * log_s_step;
            let s = log_s.exp();
            t90_grid.push(fsrs7_next_interval_bisection_scalar(w, s, S90_TARGET, None));
        }
        Self {
            log_s_min,
            log_s_step,
            t90_grid,
        }
    }

    pub(crate) fn interpolate_t90(&self, stability: f32) -> f32 {
        if self.t90_grid.len() == 1 {
            return self.t90_grid[0];
        }
        let s = stability
            .max(LUT_S_MIN.max(super::S_MIN))
            .min(LUT_S_MAX.min(super::S_MAX));
        let max_index = (self.t90_grid.len() - 1) as f32;
        let position = ((s.ln() - self.log_s_min) / self.log_s_step).clamp(0.0, max_index);
        let left = position.floor() as usize;
        let right = (left + 1).min(self.t90_grid.len() - 1);
        if left == right {
            return self.t90_grid[left];
        }
        let weight = position - left as f32;
        self.t90_grid[left] + weight * (self.t90_grid[right] - self.t90_grid[left])
    }
}

static FSRS7_S90_LUT_CACHE: OnceLock<Mutex<HashMap<u64, Arc<Fsrs7S90Lut>>>> = OnceLock::new();

fn fsrs7_params_hash(w: &[f32]) -> u64 {
    let mut hasher = DefaultHasher::new();
    w.iter()
        .take(PARAM_LEN)
        .for_each(|x| x.to_bits().hash(&mut hasher));
    hasher.finish()
}

pub(crate) fn fsrs7_s90_lut(w: &[f32]) -> Arc<Fsrs7S90Lut> {
    let key = fsrs7_params_hash(w);
    let cache = FSRS7_S90_LUT_CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    if let Some(lut) = cache
        .lock()
        .expect("fsrs7 lut cache lock poisoned")
        .get(&key)
        .cloned()
    {
        return lut;
    }
    let built = Arc::new(Fsrs7S90Lut::build(w));
    let mut guard = cache.lock().expect("fsrs7 lut cache lock poisoned");
    guard.entry(key).or_insert_with(|| built.clone()).clone()
}

pub(crate) fn fsrs7_next_interval_bisection_scalar(
    w: &[f32],
    stability: f32,
    desired_retention: f32,
    high_hint: Option<f32>,
) -> f32 {
    let desired_retention = desired_retention.clamp(DR_MIN, DR_MAX);
    let stability = stability.max(super::S_MIN);

    if desired_retention >= DR_MAX {
        return 0.0;
    }

    let mut low = 0.0;
    let mut high = high_hint
        .unwrap_or_else(|| stability.max(1.0))
        .clamp(0.0, super::S_MAX)
        .max(super::S_MIN);

    while fsrs7_forgetting_curve_scalar(w, high, stability) > desired_retention
        && high < super::S_MAX
    {
        high = (high * 2.0).min(super::S_MAX);
        if (high - super::S_MAX).abs() < f32::EPSILON {
            break;
        }
    }

    for _ in 0..BISECTION_ITERS {
        let mid = (low + high) / 2.0;
        let r = fsrs7_forgetting_curve_scalar(w, mid, stability);
        if r > desired_retention {
            low = mid;
        } else {
            high = mid;
        }
    }

    ((low + high) / 2.0).clamp(0.0, super::S_MAX)
}

pub(crate) fn fsrs7_next_interval_scalar(
    w: &[f32],
    stability: f32,
    desired_retention: f32,
    lut: &Fsrs7S90Lut,
) -> f32 {
    let desired_retention = desired_retention.clamp(DR_MIN, DR_MAX);
    let stability = stability.max(super::S_MIN);
    if desired_retention >= DR_MAX {
        return 0.0;
    }

    let mut high = lut
        .interpolate_t90(stability)
        .clamp(0.0, super::S_MAX)
        .max(super::S_MIN);
    let mut high_r = fsrs7_forgetting_curve_scalar(w, high, stability);
    if !high_r.is_finite() {
        return fsrs7_next_interval_bisection_scalar(w, stability, desired_retention, Some(high));
    }
    while high_r > desired_retention && high < super::S_MAX {
        high = (high * 2.0).min(super::S_MAX);
        high_r = fsrs7_forgetting_curve_scalar(w, high, stability);
    }
    if !high_r.is_finite() {
        return fsrs7_next_interval_bisection_scalar(w, stability, desired_retention, Some(high));
    }
    if high_r > desired_retention {
        return super::S_MAX;
    }

    let mut low = 0.0;
    let interval_solve_tol = high.max(super::S_MIN) * INTERVAL_SOLVE_TOL;
    let mut t = if desired_retention >= S90_TARGET {
        let ratio = ((1.0 - desired_retention) / (1.0 - S90_TARGET)).clamp(0.0, 1.0);
        (high * ratio).clamp(low, high)
    } else {
        (low + high) * 0.5
    };

    for _ in 0..NEWTON_ITERS {
        let (r, derivative) = fsrs7_forgetting_curve_and_derivative_scalar(w, t, stability);
        if !(r.is_finite() && derivative.is_finite() && derivative < 0.0) {
            t = (low + high) * 0.5;
            continue;
        }
        let residual = r - desired_retention;
        if residual > 0.0 {
            low = t;
        } else {
            high = t;
        }
        if high - low <= interval_solve_tol && residual.abs() <= RETENTION_SOLVE_TOL {
            return ((low + high) * 0.5).clamp(0.0, super::S_MAX);
        }

        let next = t - residual / derivative;
        t = if next.is_finite() && low < next && next < high {
            next
        } else {
            (low + high) * 0.5
        };
    }

    fsrs7_next_interval_bisection_scalar(w, stability, desired_retention, Some(high))
}

pub(super) fn power_forgetting_curve<B: Backend>(
    model: &Model<B>,
    t: Tensor<B, 1>,
    s: Tensor<B, 1>,
) -> Tensor<B, 1> {
    let t_over_s = t.clamp_min(0.0) / s.clone();

    let decay1 = -model.w.get(27);
    let decay2 = -model.w.get(28);
    let base1 = model.w.get(29);
    let base2 = model.w.get(30);

    let factor1 = base1.clone().powf(decay1.clone().powi_scalar(-1)) - 1.0;
    let factor2 = base2.clone().powf(decay2.clone().powi_scalar(-1)) - 1.0;

    let r1 = (t_over_s.clone() * factor1 + 1.0).powf(decay1);
    let r2 = (t_over_s * factor2 + 1.0).powf(decay2);

    let weight1 = model.w.get(31) * s.clone().powf(-model.w.get(33));
    let weight2 = model.w.get(32) * s.powf(model.w.get(34));

    (weight1.clone() * r1 + weight2.clone() * r2) / (weight1 + weight2)
}

pub(super) fn next_interval<B: Backend>(
    model: &Model<B>,
    stability: Tensor<B, 1>,
    desired_retention: Tensor<B, 1>,
) -> Tensor<B, 1> {
    let w = model.w.val().to_data().to_vec::<f32>().unwrap();
    let lut = fsrs7_s90_lut(&w);
    let s_vec = stability.to_data().to_vec::<f32>().unwrap();
    let dr_vec = desired_retention.to_data().to_vec::<f32>().unwrap();
    let device = stability.device();
    let out: Vec<f32> = s_vec
        .into_iter()
        .zip(dr_vec)
        .map(|(s, dr)| fsrs7_next_interval_scalar(&w, s, dr, lut.as_ref()))
        .collect();
    Tensor::from_floats(out.as_slice(), &device)
}

pub(super) fn stability_for_set<B: Backend>(
    model: &Model<B>,
    last_s: Tensor<B, 1>,
    last_d: Tensor<B, 1>,
    r: Tensor<B, 1>,
    rating: Tensor<B, 1>,
    start: usize,
) -> Tensor<B, 1> {
    let batch_size = rating.dims()[0];
    let device = rating.device();
    let hard_penalty = Tensor::ones([batch_size], &device)
        .mask_where(rating.clone().equal_elem(2), model.w.get(start + 7));
    let easy_bonus = Tensor::ones([batch_size], &device)
        .mask_where(rating.clone().equal_elem(4), model.w.get(start + 8));

    let new_s_fail = model.w.get(start + 3)
        * last_d.clone().powf(-model.w.get(start + 4))
        * ((last_s.clone() + 1).powf(model.w.get(start + 5)) - 1)
        * ((-r.clone() + 1) * model.w.get(start + 6)).exp();
    let pls = tensor_min(last_s.clone(), new_s_fail);

    let sinc = model.w.get(start).add_scalar(-1.5).exp()
        * last_d.neg().add_scalar(11.0)
        * last_s.clone().powf(-model.w.get(start + 1))
        * (((-r + 1) * model.w.get(start + 2)).exp() - 1)
        * hard_penalty
        * easy_bonus
        + 1;
    let new_s_success = tensor_max(pls.clone(), last_s * sinc);
    let success = rating.greater_elem(1);
    pls.mask_where(success, new_s_success)
}

pub(super) fn transition_function<B: Backend>(
    model: &Model<B>,
    delta_t: Tensor<B, 1>,
) -> Tensor<B, 1> {
    (model.w.get(26) * (-model.w.get(25) * delta_t).exp())
        .neg()
        .add_scalar(1.0)
}

pub(super) fn mean_reversion<B: Backend>(
    init: Tensor<B, 1>,
    current: Tensor<B, 1>,
) -> Tensor<B, 1> {
    init.mul_scalar(0.01) + current.mul_scalar(0.99)
}

pub(super) fn next_difficulty<B: Backend>(
    model: &Model<B>,
    difficulty: Tensor<B, 1>,
    rating: Tensor<B, 1>,
) -> Tensor<B, 1> {
    let delta_d = -model.w.get(6) * (rating - 3);
    let new_d = difficulty.clone() + model.linear_damping(delta_d, difficulty);
    let device = new_d.device();
    let init = model.init_difficulty(Tensor::from_floats([4.0], &device));
    mean_reversion(init, new_d).clamp(super::D_MIN, super::D_MAX)
}

fn log_expm1(x: f64) -> f64 {
    if x > 50.0 { x } else { x.exp_m1().ln() }
}

pub(super) fn memory_state_from_sm2_fsrs7<B: Backend>(
    model: &Model<B>,
    _ease_factor: f32,
    interval: f32,
    sm2_retention: f32,
) -> Result<MemoryState> {
    let params = model.w.val().to_data().to_vec::<f32>().unwrap();

    let interval = interval.max(super::S_MIN);
    let retention = sm2_retention.clamp(0.70, 0.9999);
    let decay = -params[27].max(0.001);
    let stability = if (retention - 0.9).abs() < 1e-6 {
        interval
    } else {
        let inv_decay = 1.0f64 / decay as f64;
        let x = inv_decay * 0.9f64.ln();
        let y = inv_decay * (retention as f64).ln();
        let ratio = (log_expm1(x) - log_expm1(y)).clamp(-80.0, 80.0).exp();
        (interval as f64 * ratio) as f32
    }
    .clamp(super::S_MIN, super::S_MAX);

    let difficulty = 5.0f32.clamp(super::D_MIN, super::D_MAX);
    if !stability.is_finite() || !difficulty.is_finite() {
        Err(FSRSError::InvalidInput)
    } else {
        Ok(MemoryState {
            stability,
            difficulty,
        })
    }
}

pub(crate) fn transition_scalar(w: &[f32], delta_t: f32) -> f32 {
    1.0 - w[26] * (-w[25] * delta_t).exp()
}

pub(crate) fn stability_for_set_scalar(
    w: &[f32],
    s: f32,
    r: f32,
    d: f32,
    rating: usize,
    start: usize,
) -> f32 {
    let hard_penalty = if rating == 2 { w[start + 7] } else { 1.0 };
    let easy_bonus = if rating == 4 { w[start + 8] } else { 1.0 };
    let new_s_fail = w[start + 3]
        * d.powf(-w[start + 4])
        * ((s + 1.0).powf(w[start + 5]) - 1.0)
        * ((1.0 - r) * w[start + 6]).exp();
    let pls = s.min(new_s_fail);
    if rating > 1 {
        let sinc = 1.0
            + (w[start] - 1.5).exp()
                * (11.0 - d)
                * s.powf(-w[start + 1])
                * (((1.0 - r) * w[start + 2]).exp() - 1.0)
                * hard_penalty
                * easy_bonus;
        pls.max(s * sinc)
    } else {
        pls
    }
}

pub(crate) fn stability_after_success_scalar(
    w: &[f32],
    s: f32,
    r: f32,
    d: f32,
    rating: usize,
    delta_t: f32,
) -> f32 {
    let long = stability_for_set_scalar(w, s, r, d, rating, 7);
    let short = stability_for_set_scalar(w, s, r, d, rating, 16);
    let coefficient = transition_scalar(w, delta_t);
    (coefficient * long + (1.0 - coefficient) * short).clamp(super::S_MIN, super::S_MAX)
}

pub(crate) fn stability_after_failure_scalar(
    w: &[f32],
    s: f32,
    r: f32,
    d: f32,
    delta_t: f32,
) -> f32 {
    let long = stability_for_set_scalar(w, s, r, d, 1, 7);
    let short = stability_for_set_scalar(w, s, r, d, 1, 16);
    let coefficient = transition_scalar(w, delta_t);
    (coefficient * long + (1.0 - coefficient) * short).clamp(super::S_MIN, super::S_MAX)
}

pub(crate) fn stability_short_term_scalar(w: &[f32], s: f32, r: f32, d: f32, rating: usize) -> f32 {
    let long = stability_for_set_scalar(w, s, r, d, rating, 7);
    let short = stability_for_set_scalar(w, s, r, d, rating, 16);
    let coefficient = transition_scalar(w, 0.0);
    (coefficient * long + (1.0 - coefficient) * short).clamp(super::S_MIN, super::S_MAX)
}

pub(crate) fn init_difficulty_scalar(w: &[f32], rating: usize) -> f32 {
    w[4] - (w[5] * (rating - 1) as f32).exp() + 1.0
}

pub(crate) fn mean_reversion_scalar(init: f32, current: f32) -> f32 {
    0.01 * init + 0.99 * current
}

pub(crate) fn next_difficulty_scalar(w: &[f32], d: f32, rating: usize) -> f32 {
    let delta_d = -w[6] * (rating as f32 - 3.0);
    let new_d = d + ((10.0 - d) / 9.0) * delta_d;
    mean_reversion_scalar(init_difficulty_scalar(w, 4), new_d).clamp(super::D_MIN, super::D_MAX)
}

#[cfg(test)]
mod tests {
    use super::super::{MemoryStateTensors, ModelConfig, S_MAX};
    use super::*;
    use crate::DEFAULT_PARAMETERS;
    use crate::test_helpers::{Model, Tensor};
    use burn::backend::ndarray::NdArrayDevice;
    use burn::module::Param;
    use burn::tensor::{Shape, TensorData, Tolerance};

    static DEVICE: NdArrayDevice = NdArrayDevice::Cpu;

    #[test]
    fn test_fsrs7_freeze_short_term_disables_transition_weight() {
        let model = Model::new(ModelConfig {
            freeze_short_term_stability: true,
            ..ModelConfig::default()
        });
        let w = model.w.val().to_data().to_vec::<f32>().unwrap();
        assert_eq!(w.len(), 35);
        assert_eq!(w[26], 0.0);
    }

    #[test]
    fn test_fsrs7_path_produces_finite_interval() {
        let model = Model::new(ModelConfig::default());
        let delta_ts = Tensor::from_floats([[0.0], [3.0]], &DEVICE);
        let ratings = Tensor::from_floats([[3.0], [3.0]], &DEVICE);
        let state = model.forward(delta_ts, ratings, None);
        let stability: f32 = state.stability.into_scalar();
        assert!(stability.is_finite());
        assert!(stability > 0.0);

        let interval = model
            .next_interval(
                Tensor::from_floats([stability], &DEVICE),
                Tensor::from_floats([0.9], &DEVICE),
            )
            .into_scalar();
        assert!(interval.is_finite());
        assert!(interval >= 0.0);
    }

    #[test]
    fn test_fsrs7_negative_elapsed_days_match_zero_elapsed_days() {
        let model = Model::new(ModelConfig::default());
        let rating = Tensor::from_floats([3.0], &DEVICE);
        let state = MemoryStateTensors {
            stability: Tensor::from_floats([5.0], &DEVICE),
            difficulty: Tensor::from_floats([6.0], &DEVICE),
        };

        let negative = model.step(
            Tensor::from_floats([-0.25], &DEVICE),
            rating.clone(),
            state.clone(),
            1,
        );
        let zero = model.step(Tensor::from_floats([0.0], &DEVICE), rating, state, 1);

        negative
            .stability
            .to_data()
            .assert_approx_eq::<f32>(&zero.stability.to_data(), Tolerance::absolute(1e-6));
        negative
            .difficulty
            .to_data()
            .assert_approx_eq::<f32>(&zero.difficulty.to_data(), Tolerance::absolute(1e-6));
    }

    #[test]
    fn test_fsrs7_step_is_long_term_only_when_transition_weight_is_zero() {
        let mut model = Model::new(ModelConfig::default());
        let mut w = model.w.val().to_data().to_vec::<f32>().unwrap();
        w[26] = 0.0;
        model.w = Param::from_tensor(Tensor::from_floats(
            TensorData::new(
                w.clone(),
                Shape {
                    dims: vec![w.len()],
                },
            ),
            &DEVICE,
        ));

        let delta_t = Tensor::from_floats([0.5, 2.0, 7.0], &DEVICE);
        let rating = Tensor::from_floats([1.0, 3.0, 4.0], &DEVICE);
        let state = MemoryStateTensors {
            stability: Tensor::from_floats([5.0, 5.0, 5.0], &DEVICE),
            difficulty: Tensor::from_floats([6.0, 6.0, 6.0], &DEVICE),
        };

        let retrievability = model.power_forgetting_curve(delta_t.clone(), state.stability.clone());
        let expected_s = super::stability_for_set(
            &model,
            state.stability.clone(),
            state.difficulty.clone(),
            retrievability,
            rating.clone(),
            7,
        );
        let expected_d = super::next_difficulty(&model, state.difficulty.clone(), rating.clone());

        let actual = model.step(delta_t, rating, state, 1);
        actual
            .stability
            .to_data()
            .assert_approx_eq::<f32>(&expected_s.to_data(), Tolerance::absolute(1e-5));
        actual
            .difficulty
            .to_data()
            .assert_approx_eq::<f32>(&expected_d.to_data(), Tolerance::absolute(1e-5));
    }

    #[test]
    fn test_fsrs7_scalar_interval_hits_target_retrievability() {
        let w = DEFAULT_PARAMETERS;
        let lut = Fsrs7S90Lut::build(&w);
        for stability in [0.2, 1.0, 10.0, 100.0, 1000.0] {
            for desired in [0.5, 0.7, 0.8, 0.9, 0.95] {
                let t = fsrs7_next_interval_scalar(&w, stability, desired, &lut);
                let r = fsrs7_forgetting_curve_scalar(&w, t, stability);
                assert!(
                    (r - desired).abs() <= 1e-3 || (t - S_MAX).abs() < 1e-4,
                    "stability={stability}, desired={desired}, t={t}, r={r}"
                );
            }
        }
    }

    #[test]
    fn test_fsrs7_scalar_interval_matches_bisection_baseline() {
        let w = DEFAULT_PARAMETERS;
        let lut = Fsrs7S90Lut::build(&w);
        for stability in [0.2, 1.0, 10.0, 100.0, 1000.0] {
            for desired in [0.4, 0.6, 0.8, 0.9, 0.95] {
                let baseline = fsrs7_next_interval_bisection_scalar(&w, stability, desired, None);
                let optimized = fsrs7_next_interval_scalar(&w, stability, desired, &lut);
                let baseline_r = fsrs7_forgetting_curve_scalar(&w, baseline, stability);
                let optimized_r = fsrs7_forgetting_curve_scalar(&w, optimized, stability);
                assert!(
                    (optimized_r - baseline_r).abs() <= 1e-3,
                    "stability={stability}, desired={desired}, baseline={baseline}, optimized={optimized}, baseline_r={baseline_r}, optimized_r={optimized_r}"
                );
            }
        }
    }

    #[test]
    fn test_fsrs7_s90_lut_cache_reuse() {
        let w = DEFAULT_PARAMETERS;
        let lut_a = fsrs7_s90_lut(&w);
        let lut_b = fsrs7_s90_lut(&w);
        assert!(std::sync::Arc::ptr_eq(&lut_a, &lut_b));

        let mut w2 = DEFAULT_PARAMETERS;
        w2[27] += 0.001;
        let lut_c = fsrs7_s90_lut(&w2);
        assert!(!std::sync::Arc::ptr_eq(&lut_a, &lut_c));
    }

    #[test]
    fn test_fsrs7_scalar_interval_monotonicity() {
        let w = DEFAULT_PARAMETERS;
        let lut = Fsrs7S90Lut::build(&w);

        for stability in [0.2, 1.0, 10.0, 100.0, 1000.0] {
            let desireds = [0.2, 0.4, 0.6, 0.8, 0.9, 0.95];
            let intervals: Vec<f32> = desireds
                .iter()
                .map(|&dr| fsrs7_next_interval_scalar(&w, stability, dr, &lut))
                .collect();
            for pair in intervals.windows(2) {
                assert!(
                    pair[0] >= pair[1],
                    "expected interval to decrease with higher retention: stability={stability}, intervals={intervals:?}"
                );
            }
        }

        for desired in [0.5, 0.7, 0.9] {
            let stabilities = [0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 100.0, 1000.0];
            let intervals: Vec<f32> = stabilities
                .iter()
                .map(|&s| fsrs7_next_interval_scalar(&w, s, desired, &lut))
                .collect();
            for pair in intervals.windows(2) {
                assert!(
                    pair[1] >= pair[0],
                    "expected interval to increase with stability: desired={desired}, intervals={intervals:?}"
                );
            }
        }
    }

    #[test]
    fn test_fsrs7_scalar_interval_matches_bisection_dense_grid() {
        let w = DEFAULT_PARAMETERS;
        let lut = Fsrs7S90Lut::build(&w);
        let desireds = [0.25, 0.4, 0.55, 0.7, 0.8, 0.9, 0.95, 0.98];

        for i in 0..25 {
            let p = i as f32 / 24.0;
            let stability =
                (super::super::S_MIN.ln() + (S_MAX.ln() - super::super::S_MIN.ln()) * p).exp();
            for desired in desireds {
                let baseline = fsrs7_next_interval_bisection_scalar(&w, stability, desired, None);
                let optimized = fsrs7_next_interval_scalar(&w, stability, desired, &lut);
                let baseline_r = fsrs7_forgetting_curve_scalar(&w, baseline, stability);
                let optimized_r = fsrs7_forgetting_curve_scalar(&w, optimized, stability);
                assert!(
                    (optimized_r - baseline_r).abs() <= 1e-3,
                    "stability={stability}, desired={desired}, baseline={baseline}, optimized={optimized}, baseline_r={baseline_r}, optimized_r={optimized_r}"
                );
            }
        }
    }

    #[test]
    fn test_fsrs7_model_next_interval_vectorized_matches_scalar() {
        let model = Model::new(ModelConfig::default());
        let w = model.w.val().to_data().to_vec::<f32>().unwrap();
        let lut = fsrs7_s90_lut(&w);

        let stabilities = [0.2, 0.8, 3.0, 12.0, 40.0, 200.0];
        let desired = [0.95, 0.9, 0.8, 0.7, 0.6, 0.5];

        let model_out = model
            .next_interval(
                Tensor::from_floats(stabilities, &DEVICE),
                Tensor::from_floats(desired, &DEVICE),
            )
            .to_data()
            .to_vec::<f32>()
            .unwrap();

        let scalar_out: Vec<f32> = stabilities
            .iter()
            .zip(desired.iter())
            .map(|(&s, &dr)| fsrs7_next_interval_scalar(&w, s, dr, lut.as_ref()))
            .collect();

        for (model_value, scalar_value) in model_out.iter().zip(scalar_out.iter()) {
            assert!((model_value - scalar_value).abs() < 1e-4);
        }
    }
}

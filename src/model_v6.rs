use super::{Fsrs6Ops, Get, MemoryStateTensors, Model, VersionOps};
use crate::FSRSError;
use crate::error::Result;
use crate::inference::{FSRS5_DEFAULT_DECAY, FSRS6_DEFAULT_PARAMETERS, MemoryState, Parameters};
use burn::tensor::ElementConversion;
use burn::tensor::{Tensor, backend::Backend};

pub(super) const PARAM_LEN: usize = 21;

pub(crate) fn check_and_fill_parameters_fsrs6(parameters: &Parameters) -> Option<Vec<f32>> {
    match parameters.len() {
        0 => Some(FSRS6_DEFAULT_PARAMETERS.to_vec()),
        17 => {
            let mut parameters = parameters.to_vec();
            parameters[4] = parameters[5].mul_add(2.0, parameters[4]);
            parameters[5] = parameters[5].mul_add(3.0, 1.0).ln() / 3.0;
            parameters[6] += 0.5;
            parameters.extend_from_slice(&[0.0, 0.0, 0.0, FSRS5_DEFAULT_DECAY]);
            Some(parameters)
        }
        19 => {
            let mut parameters = parameters.to_vec();
            parameters.extend_from_slice(&[0.0, FSRS5_DEFAULT_DECAY]);
            Some(parameters)
        }
        PARAM_LEN => Some(parameters.to_vec()),
        _ => None,
    }
}

impl<B: Backend> VersionOps<B> for Fsrs6Ops {
    fn apply_freeze_short_term(initial_params: &mut [f32]) {
        // Legacy (FSRS-6) short-term terms only.
        initial_params[17] = 0.0;
        initial_params[18] = 0.0;
        initial_params[19] = 0.0;
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
        // FSRS-6 stays day-based: keep f32 transport, but round elapsed days to nearest day.
        let delta_t = round_elapsed_days(delta_t);
        let retrievability = power_forgetting_curve(model, delta_t.clone(), last_s.clone());
        let stability_after_success = stability_after_success(
            model,
            last_s.clone(),
            last_d.clone(),
            retrievability.clone(),
            rating.clone(),
        );
        let stability_after_failure =
            stability_after_failure(model, last_s.clone(), last_d.clone(), retrievability);
        let stability_short_term = stability_short_term(model, last_s.clone(), rating.clone());
        let new_s = stability_after_success
            .mask_where(rating.clone().equal_elem(1), stability_after_failure)
            .mask_where(delta_t.equal_elem(0), stability_short_term);

        let new_d = mean_reversion(model, model.next_difficulty(last_d, rating))
            .clamp(super::D_MIN, super::D_MAX);
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
        memory_state_from_sm2_fsrs6(model, ease_factor, interval, sm2_retention)
    }

    fn interval_at_retrievability(
        model: &Model<B>,
        stability: f32,
        target_retrievability: f32,
    ) -> f32 {
        let stability = stability.max(super::S_MIN);
        if (target_retrievability - 0.9).abs() <= f32::EPSILON {
            stability
        } else {
            let w = model.w.val().to_data().to_vec::<f32>().unwrap();
            next_interval_scalar(&w, stability, target_retrievability)
        }
    }
}

pub(super) fn power_forgetting_curve<B: Backend>(
    model: &Model<B>,
    t: Tensor<B, 1>,
    s: Tensor<B, 1>,
) -> Tensor<B, 1> {
    let t = round_elapsed_days(t);
    let decay = -model.w.get(20);
    let factor = decay.clone().powi_scalar(-1).mul_scalar(0.9f32.ln()).exp() - 1.0;
    (t / s * factor + 1.0).powf(decay)
}

pub(super) fn next_interval<B: Backend>(
    model: &Model<B>,
    stability: Tensor<B, 1>,
    desired_retention: Tensor<B, 1>,
) -> Tensor<B, 1> {
    let decay = -model.w.get(20);
    let factor = decay.clone().powi_scalar(-1).mul_scalar(0.9f32.ln()).exp() - 1.0;
    (stability * (desired_retention.powf(decay.clone().powi_scalar(-1)) - 1.0) / factor)
        .clamp(0.0, super::S_MAX)
}

pub(super) fn stability_after_success<B: Backend>(
    model: &Model<B>,
    last_s: Tensor<B, 1>,
    last_d: Tensor<B, 1>,
    r: Tensor<B, 1>,
    rating: Tensor<B, 1>,
) -> Tensor<B, 1> {
    let batch_size = rating.dims()[0];
    let device = rating.device();
    let hard_penalty = Tensor::ones([batch_size], &device)
        .mask_where(rating.clone().equal_elem(2), model.w.get(15));
    let easy_bonus =
        Tensor::ones([batch_size], &device).mask_where(rating.equal_elem(4), model.w.get(16));

    last_s.clone()
        * (model.w.get(8).exp()
            * (-last_d + 11)
            * (last_s.powf(-model.w.get(9)))
            * (((-r + 1) * model.w.get(10)).exp() - 1)
            * hard_penalty
            * easy_bonus
            + 1)
}

pub(super) fn stability_after_failure<B: Backend>(
    model: &Model<B>,
    last_s: Tensor<B, 1>,
    last_d: Tensor<B, 1>,
    r: Tensor<B, 1>,
) -> Tensor<B, 1> {
    let new_s = model.w.get(11)
        * last_d.powf(-model.w.get(12))
        * ((last_s.clone() + 1).powf(model.w.get(13)) - 1)
        * ((-r + 1) * model.w.get(14)).exp();
    let new_s_min = last_s / (model.w.get(17) * model.w.get(18)).exp();
    new_s
        .clone()
        .mask_where(new_s_min.clone().lower(new_s), new_s_min)
}

pub(super) fn stability_short_term<B: Backend>(
    model: &Model<B>,
    last_s: Tensor<B, 1>,
    rating: Tensor<B, 1>,
) -> Tensor<B, 1> {
    let sinc = (model.w.get(17) * (rating.clone() - 3 + model.w.get(18))).exp()
        * last_s.clone().powf(-model.w.get(19));

    last_s
        * sinc
            .clone()
            .mask_where(rating.greater_equal_elem(2), sinc.clamp_min(1.0))
}

pub(super) fn mean_reversion<B: Backend>(model: &Model<B>, new_d: Tensor<B, 1>) -> Tensor<B, 1> {
    let device = new_d.device();
    let rating = Tensor::from_floats([4.0], &device);
    model.w.get(7) * (model.init_difficulty(rating) - new_d.clone()) + new_d
}

pub(super) fn memory_state_from_sm2_fsrs6<B: Backend>(
    model: &Model<B>,
    ease_factor: f32,
    interval: f32,
    sm2_retention: f32,
) -> Result<MemoryState> {
    let w = &model.w;
    let decay: f32 = w.get(20).neg().into_scalar().elem();
    let factor = 0.9f32.powf(1.0 / decay) - 1.0;
    let stability = interval.max(super::S_MIN) * factor / (sm2_retention.powf(1.0 / decay) - 1.0);
    let w8: f32 = w.get(8).into_scalar().elem();
    let w9: f32 = w.get(9).into_scalar().elem();
    let w10: f32 = w.get(10).into_scalar().elem();
    let difficulty = 11.0
        - (ease_factor - 1.0)
            / (w8.exp() * stability.powf(-w9) * ((1.0 - sm2_retention) * w10).exp_m1());
    if !stability.is_finite() || !difficulty.is_finite() {
        Err(FSRSError::InvalidInput)
    } else {
        Ok(MemoryState {
            stability,
            difficulty: difficulty.clamp(super::D_MIN, super::D_MAX),
        })
    }
}

pub(crate) fn power_forgetting_curve_scalar(w: &[f32], t: f32, s: f32) -> f32 {
    let s = s.max(super::S_MIN);
    let decay = -w[20];
    let factor = 0.9f32.powf(1.0 / decay) - 1.0;
    let t = t.max(0.0).round();
    (t / s).mul_add(factor, 1.0).powf(decay)
}

pub(crate) fn next_interval_scalar(w: &[f32], stability: f32, desired_retention: f32) -> f32 {
    let stability = stability.max(super::S_MIN);
    let desired_retention = desired_retention.clamp(0.0001, 0.9999);
    let decay = -w[20];
    let factor = 0.9f32.powf(1.0 / decay) - 1.0;
    (stability / factor * (desired_retention.powf(1.0 / decay) - 1.0)).clamp(0.0, super::S_MAX)
}

fn round_elapsed_days<B: Backend>(t: Tensor<B, 1>) -> Tensor<B, 1> {
    t.clamp_min(0.0).add_scalar(0.5).int().float()
}

pub(crate) fn stability_after_success_scalar(
    w: &[f32],
    s: f32,
    r: f32,
    d: f32,
    rating: usize,
) -> f32 {
    let hard_penalty = if rating == 2 { w[15] } else { 1.0 };
    let easy_bonus = if rating == 4 { w[16] } else { 1.0 };
    (s * (w[8].exp()
        * (11.0 - d)
        * s.powf(-w[9])
        * (((1.0 - r) * w[10]).exp() - 1.0)
        * hard_penalty
        * easy_bonus
        + 1.0))
        .clamp(super::S_MIN, super::S_MAX)
}

pub(crate) fn stability_after_failure_scalar(w: &[f32], s: f32, r: f32, d: f32) -> f32 {
    let new_s_min = s / (w[17] * w[18]).exp();
    let new_s = w[11] * d.powf(-w[12]) * ((s + 1.0).powf(w[13]) - 1.0) * ((1.0 - r) * w[14]).exp();
    new_s.min(new_s_min).clamp(super::S_MIN, super::S_MAX)
}

pub(crate) fn stability_short_term_scalar(w: &[f32], s: f32, rating: usize) -> f32 {
    let sinc = (w[17] * (rating as f32 - 3.0 + w[18])).exp() * s.powf(-w[19]);
    let new_s = s * if rating >= 3 { sinc.max(1.0) } else { sinc };
    new_s.clamp(super::S_MIN, super::S_MAX)
}

pub(crate) fn init_difficulty_scalar(w: &[f32], rating: usize) -> f32 {
    w[4] - (w[5] * (rating - 1) as f32).exp() + 1.0
}

pub(crate) fn mean_reversion_scalar(w: &[f32], init: f32, current: f32) -> f32 {
    w[7] * init + (1.0 - w[7]) * current
}

pub(crate) fn next_difficulty_scalar(w: &[f32], d: f32, rating: usize) -> f32 {
    let delta_d = -w[6] * (rating as f32 - 3.0);
    let new_d = d + ((10.0 - d) / 9.0) * delta_d;
    mean_reversion_scalar(w, init_difficulty_scalar(w, 4), new_d).clamp(super::D_MIN, super::D_MAX)
}

#[cfg(test)]
mod tests {
    use super::super::parameters_to_model;
    use crate::inference::FSRS6_DEFAULT_PARAMETERS;
    use crate::test_helpers::TestHelper;
    use crate::test_helpers::{Model, NdArrayAutodiff, Tensor};
    use burn::backend::ndarray::NdArrayDevice;
    use burn::tensor::{TensorData, Tolerance};

    static DEVICE: NdArrayDevice = NdArrayDevice::Cpu;

    fn fsrs6_model() -> Model {
        parameters_to_model::<NdArrayAutodiff>(&FSRS6_DEFAULT_PARAMETERS, &DEVICE)
    }

    #[test]
    fn test_power_forgetting_curve() {
        let model = fsrs6_model();
        let delta_t = Tensor::from_floats([0.0, 1.0, 2.0, 3.0, 4.0, 5.0], &DEVICE);
        let stability = Tensor::from_floats([1.0, 2.0, 3.0, 4.0, 4.0, 2.0], &DEVICE);
        let retrievability = model.power_forgetting_curve(delta_t, stability);

        retrievability.to_data().assert_approx_eq::<f32>(
            &TensorData::from([1.0, 0.9403443, 0.9253786, 0.9185229, 0.9, 0.8261359]),
            Tolerance::absolute(1e-5),
        );
    }

    #[test]
    fn test_power_forgetting_curve_scalar_rounds_elapsed_days() {
        let w = FSRS6_DEFAULT_PARAMETERS;
        let s = 10.0;
        let at_zero = super::power_forgetting_curve_scalar(&w, 0.0, s);
        let at_half_minus = super::power_forgetting_curve_scalar(&w, 0.49, s);
        let at_one = super::power_forgetting_curve_scalar(&w, 1.0, s);
        let at_half_plus = super::power_forgetting_curve_scalar(&w, 0.51, s);
        assert!((at_zero - at_half_minus).abs() < 1e-6);
        assert!((at_one - at_half_plus).abs() < 1e-6);
    }

    #[test]
    fn test_next_interval_scalar_is_fractional() {
        let w = FSRS6_DEFAULT_PARAMETERS;
        let interval = super::next_interval_scalar(&w, 121.01552, 0.9);
        assert!((interval - 121.01551).abs() < 1e-4);
    }

    #[test]
    fn test_init_stability() {
        let model = fsrs6_model();
        let rating = Tensor::from_floats([1.0, 2.0, 3.0, 4.0, 1.0, 2.0], &DEVICE);
        let stability = model.init_stability(rating);
        assert_eq!(
            stability.to_data(),
            TensorData::from([
                FSRS6_DEFAULT_PARAMETERS[0],
                FSRS6_DEFAULT_PARAMETERS[1],
                FSRS6_DEFAULT_PARAMETERS[2],
                FSRS6_DEFAULT_PARAMETERS[3],
                FSRS6_DEFAULT_PARAMETERS[0],
                FSRS6_DEFAULT_PARAMETERS[1]
            ])
        )
    }

    #[test]
    fn test_init_difficulty() {
        let model = fsrs6_model();
        let rating = Tensor::from_floats([1.0, 2.0, 3.0, 4.0, 1.0, 2.0], &DEVICE);
        let difficulty = model.init_difficulty(rating);
        assert_eq!(
            difficulty.to_data(),
            TensorData::from([
                FSRS6_DEFAULT_PARAMETERS[4],
                FSRS6_DEFAULT_PARAMETERS[4] - FSRS6_DEFAULT_PARAMETERS[5].exp() + 1.0,
                FSRS6_DEFAULT_PARAMETERS[4] - (2.0 * FSRS6_DEFAULT_PARAMETERS[5]).exp() + 1.0,
                FSRS6_DEFAULT_PARAMETERS[4] - (3.0 * FSRS6_DEFAULT_PARAMETERS[5]).exp() + 1.0,
                FSRS6_DEFAULT_PARAMETERS[4],
                FSRS6_DEFAULT_PARAMETERS[4] - FSRS6_DEFAULT_PARAMETERS[5].exp() + 1.0,
            ])
        )
    }

    #[test]
    fn test_forward() {
        let model = fsrs6_model();
        let delta_ts = Tensor::from_floats(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0, 1.0, 2.0, 2.0],
            ],
            &DEVICE,
        );
        let ratings = Tensor::from_floats(
            [
                [1.0, 2.0, 3.0, 4.0, 1.0, 2.0],
                [1.0, 2.0, 3.0, 4.0, 1.0, 2.0],
            ],
            &DEVICE,
        );
        let state = model.forward(delta_ts, ratings, None);
        let stability = state.stability.to_data();
        let difficulty = state.difficulty.to_data();

        stability.to_vec::<f32>().unwrap().assert_approx_eq([
            0.10088589,
            3.2494123,
            7.3153,
            18.014914,
            0.112798266,
            4.4694576,
        ]);

        difficulty
            .to_vec::<f32>()
            .unwrap()
            .assert_approx_eq([8.806304, 6.7404594, 2.1112142, 1.0, 8.806304, 6.7404594]);
    }

    #[test]
    fn test_next_difficulty() {
        let model = fsrs6_model();
        let difficulty = Tensor::from_floats([5.0; 4], &DEVICE);
        let rating = Tensor::from_floats([1.0, 2.0, 3.0, 4.0], &DEVICE);
        let next_difficulty = model.next_difficulty(difficulty, rating);
        next_difficulty.clone().backward();

        next_difficulty
            .to_data()
            .to_vec::<f32>()
            .unwrap()
            .assert_approx_eq([8.354889, 6.6774445, 5.0, 3.3225555]);
        let next_difficulty = super::mean_reversion(&model, next_difficulty);
        next_difficulty.clone().backward();

        next_difficulty
            .to_data()
            .to_vec::<f32>()
            .unwrap()
            .assert_approx_eq([8.341763, 6.6659956, 4.990228, 3.3144615]);
    }

    #[test]
    fn test_next_stability() {
        let model = fsrs6_model();
        let stability = Tensor::from_floats([5.0; 4], &DEVICE);
        let difficulty = Tensor::from_floats([1.0, 2.0, 3.0, 4.0], &DEVICE);
        let retrievability = Tensor::from_floats([0.9, 0.8, 0.7, 0.6], &DEVICE);
        let rating = Tensor::from_floats([1.0, 2.0, 3.0, 4.0], &DEVICE);
        let s_recall = super::stability_after_success(
            &model,
            stability.clone(),
            difficulty.clone(),
            retrievability.clone(),
            rating.clone(),
        );
        s_recall.clone().backward();

        s_recall
            .to_data()
            .to_vec::<f32>()
            .unwrap()
            .assert_approx_eq([25.602541, 28.226582, 58.656002, 127.226685]);
        let s_forget =
            super::stability_after_failure(&model, stability.clone(), difficulty, retrievability);
        s_forget.clone().backward();

        s_forget
            .to_data()
            .to_vec::<f32>()
            .unwrap()
            .assert_approx_eq([1.0525396, 1.1894329, 1.3680838, 1.584989]);
        let next_stability = s_recall.mask_where(rating.clone().equal_elem(1), s_forget);
        next_stability.clone().backward();

        next_stability
            .to_data()
            .to_vec::<f32>()
            .unwrap()
            .assert_approx_eq([1.0525396, 28.226582, 58.656002, 127.226685]);
        let next_stability = super::stability_short_term(&model, stability, rating);

        next_stability
            .to_data()
            .to_vec::<f32>()
            .unwrap()
            .assert_approx_eq([1.596818, 5.0, 5.0, 8.12961]);
    }
}

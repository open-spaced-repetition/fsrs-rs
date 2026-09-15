use crate::error::{FSRSError, Result};
use crate::inference::{FSRS5_DEFAULT_DECAY, FSRS6_DEFAULT_PARAMETERS, MemoryState, Parameters};

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

#[inline]
pub(crate) fn power_forgetting_curve_scalar(w: &[f32], t: f32, s: f32) -> f32 {
    let s = s.max(super::S_MIN);
    let decay = -w[20];
    let factor = 0.9f32.powf(1.0 / decay) - 1.0;
    let t = t.max(0.0).round();
    (t / s).mul_add(factor, 1.0).powf(decay)
}

#[inline]
pub(crate) fn next_interval_scalar(w: &[f32], stability: f32, desired_retention: f32) -> f32 {
    let stability = stability.max(super::S_MIN);
    let desired_retention = desired_retention.clamp(0.0001, 0.9999);
    let decay = -w[20];
    let factor = 0.9f32.powf(1.0 / decay) - 1.0;
    (stability / factor * (desired_retention.powf(1.0 / decay) - 1.0)).clamp(0.0, super::S_MAX)
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
    w[4] - (w[5] * rating.saturating_sub(1) as f32).exp() + 1.0
}

pub(crate) fn mean_reversion_scalar(w: &[f32], init: f32, current: f32) -> f32 {
    w[7] * init + (1.0 - w[7]) * current
}

pub(crate) fn next_difficulty_scalar(w: &[f32], d: f32, rating: usize) -> f32 {
    let delta_d = -w[6] * (rating as f32 - 3.0);
    let new_d = d + ((10.0 - d) / 9.0) * delta_d;
    mean_reversion_scalar(w, init_difficulty_scalar(w, 4), new_d).clamp(super::D_MIN, super::D_MAX)
}

pub(crate) fn next_state_scalar(
    w: &[f32],
    state: MemoryState,
    delta_t: f32,
    rating: u32,
) -> MemoryState {
    let rating = rating.clamp(1, 4) as usize;
    let delta_t = delta_t.max(0.0).round();
    let s = state.stability.clamp(super::S_MIN, super::S_MAX);
    let d = state.difficulty.clamp(super::D_MIN, super::D_MAX);
    let r = power_forgetting_curve_scalar(w, delta_t, s);
    let stability = if delta_t == 0.0 {
        let sinc = (w[17] * (rating as f32 - 3.0 + w[18])).exp() * s.powf(-w[19]);
        (s * if rating >= 2 { sinc.max(1.0) } else { sinc }).clamp(super::S_MIN, super::S_MAX)
    } else if rating == 1 {
        stability_after_failure_scalar(w, s, r, d)
    } else {
        stability_after_success_scalar(w, s, r, d, rating)
    };
    MemoryState {
        stability,
        difficulty: next_difficulty_scalar(w, d, rating),
        stability_fast: stability,
    }
}

pub(crate) fn memory_state_from_sm2_scalar(
    w: &[f32],
    ease_factor: f32,
    interval: f32,
    sm2_retention: f32,
) -> Result<MemoryState> {
    let decay = -w[20];
    let factor = 0.9f32.powf(1.0 / decay) - 1.0;
    let stability = interval.max(super::S_MIN) * factor / (sm2_retention.powf(1.0 / decay) - 1.0);
    let difficulty = 11.0
        - (ease_factor - 1.0)
            / (w[8].exp() * stability.powf(-w[9]) * ((1.0 - sm2_retention) * w[10]).exp_m1());
    if !stability.is_finite() || !difficulty.is_finite() {
        Err(FSRSError::InvalidInput)
    } else {
        Ok(MemoryState {
            stability,
            difficulty: difficulty.clamp(super::D_MIN, super::D_MAX),
            stability_fast: stability,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn forgetting_curve_rounds_elapsed_days() {
        let w = FSRS6_DEFAULT_PARAMETERS;
        assert_eq!(
            power_forgetting_curve_scalar(&w, 0.0, 10.0),
            power_forgetting_curve_scalar(&w, 0.49, 10.0)
        );
        assert_eq!(
            power_forgetting_curve_scalar(&w, 1.0, 10.0),
            power_forgetting_curve_scalar(&w, 0.51, 10.0)
        );
    }

    #[test]
    fn next_interval_remains_fractional() {
        let interval = next_interval_scalar(&FSRS6_DEFAULT_PARAMETERS, 121.01552, 0.9);
        assert!((interval - 121.01551).abs() < 1e-4);
    }
}

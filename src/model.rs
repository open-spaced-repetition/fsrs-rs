use crate::DEFAULT_PARAMETERS;
use crate::dataset::FSRSReview;
use crate::error::{FSRSError, Result};
use crate::inference::{MemoryState, Parameters};
use crate::parameter_clipper::clip_parameters;
use crate::simulation::{D_MAX, D_MIN, S_MAX, S_MIN};

#[path = "model_v6.rs"]
pub(crate) mod model_v6;
#[path = "model_v7.rs"]
pub(crate) mod model_v7;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ModelVersion {
    Fsrs6,
    Fsrs7,
}

impl ModelVersion {
    pub(crate) fn from_param_count(param_count: usize) -> Self {
        if param_count == model_v7::PARAM_LEN {
            Self::Fsrs7
        } else {
            Self::Fsrs6
        }
    }
}

impl core::fmt::Display for ModelVersion {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Fsrs6 => write!(f, "FSRS6"),
            Self::Fsrs7 => write!(f, "FSRS7"),
        }
    }
}

#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub freeze_initial_stability: bool,
    pub initial_stability: Option<[f32; 4]>,
    pub initial_forgetting_curve: Option<[f32; 8]>,
    pub freeze_short_term_stability: bool,
    pub num_relearning_steps: usize,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            freeze_initial_stability: false,
            initial_stability: None,
            initial_forgetting_curve: None,
            freeze_short_term_stability: false,
            num_relearning_steps: 1,
        }
    }
}

impl ModelConfig {
    pub(crate) fn initial_parameters(&self) -> Vec<f32> {
        let mut parameters = DEFAULT_PARAMETERS.to_vec();
        if let Some(initial_stability) = self.initial_stability {
            parameters[..4].copy_from_slice(&initial_stability);
        }
        if let Some(initial_forgetting_curve) = self.initial_forgetting_curve {
            let start = if parameters.len() == model_v7::PARAM_LEN {
                23
            } else {
                27
            };
            parameters[start..start + 8].copy_from_slice(&initial_forgetting_curve);
        }
        if self.freeze_short_term_stability
            && ModelVersion::from_param_count(parameters.len()) == ModelVersion::Fsrs6
        {
            parameters[17..20].fill(0.0);
        }
        parameters
    }
}

/// The main FSRS model. The parameter count selects FSRS-6 (21) or FSRS-7 (34).
#[derive(Debug, Clone)]
pub struct FSRS {
    parameters: Vec<f32>,
    version: ModelVersion,
}

impl Default for FSRS {
    fn default() -> Self {
        Self::new(&[]).expect("default parameters should be valid")
    }
}

impl FSRS {
    pub fn new(parameters: &Parameters) -> Result<Self> {
        let parameters = check_and_fill_parameters(parameters)?;
        let config = ModelConfig::default();
        let parameters = clip_parameters(&parameters, config.num_relearning_steps, true);
        let version = ModelVersion::from_param_count(parameters.len());
        Ok(Self {
            parameters,
            version,
        })
    }

    pub(crate) fn parameters(&self) -> &[f32] {
        &self.parameters
    }

    pub(crate) const fn version(&self) -> ModelVersion {
        self.version
    }

    #[inline]
    pub(crate) fn power_forgetting_curve_for_state(&self, t: f32, state: MemoryState) -> f32 {
        match self.version {
            ModelVersion::Fsrs6 => {
                model_v6::power_forgetting_curve_scalar(&self.parameters, t, state.stability)
            }
            ModelVersion::Fsrs7 => {
                model_v7::fsrs7_forgetting_curve_scalar_for_state(&self.parameters, t, state)
            }
        }
    }

    #[inline]
    pub(crate) fn power_forgetting_curve(&self, t: f32, stability: f32) -> f32 {
        self.power_forgetting_curve_for_state(
            t,
            MemoryState {
                stability,
                difficulty: 5.0,
                stability_fast: stability,
            },
        )
    }

    #[inline]
    pub fn next_interval_for_state(&self, state: MemoryState, desired_retention: f32) -> f32 {
        match self.version {
            ModelVersion::Fsrs6 => {
                model_v6::next_interval_scalar(&self.parameters, state.stability, desired_retention)
            }
            ModelVersion::Fsrs7 => model_v7::fsrs7_next_interval_scalar_for_state(
                &self.parameters,
                state,
                desired_retention,
            ),
        }
    }

    pub fn interval_at_retrievability(&self, state: MemoryState, target: f32) -> f32 {
        self.next_interval_for_state(state, target.clamp(0.0001, 0.9999))
    }

    #[inline]
    pub(crate) fn init_stability(&self, rating: u32) -> f32 {
        self.parameters[rating.saturating_sub(1).min(3) as usize]
    }

    #[inline]
    pub(crate) fn init_difficulty(&self, rating: u32) -> f32 {
        match self.version {
            ModelVersion::Fsrs6 => {
                model_v6::init_difficulty_scalar(&self.parameters, rating as usize)
            }
            ModelVersion::Fsrs7 => {
                model_v7::init_difficulty_scalar(&self.parameters, rating as usize)
            }
        }
    }

    pub(crate) fn step(
        &self,
        delta_t: f32,
        rating: u32,
        state: MemoryState,
        nth: usize,
    ) -> MemoryState {
        let last = MemoryState {
            stability: state.stability.clamp(S_MIN, S_MAX),
            difficulty: state.difficulty.clamp(D_MIN, D_MAX),
            stability_fast: state.stability_fast.clamp(S_MIN, S_MAX),
        };
        let mut next = match self.version {
            ModelVersion::Fsrs6 => {
                model_v6::next_state_scalar(&self.parameters, last, delta_t, rating)
            }
            ModelVersion::Fsrs7 => {
                model_v7::fsrs7_next_state_scalar(&self.parameters, last, delta_t, rating as usize)
            }
        };
        if nth == 0 && state.stability == 0.0 {
            let rating = rating.clamp(1, 4);
            next.stability = self.init_stability(rating).clamp(S_MIN, S_MAX);
            next.difficulty = self.init_difficulty(rating).clamp(D_MIN, D_MAX);
            next.stability_fast = if self.version == ModelVersion::Fsrs7 {
                (next.stability * 0.8).clamp(S_MIN, S_MAX)
            } else {
                next.stability
            };
        }
        if rating == 0 { last } else { next }
    }

    pub(crate) fn init_state_from_first_review(&self, review: &FSRSReview) -> MemoryState {
        if review.rating == 0 {
            MemoryState {
                stability: S_MIN,
                difficulty: D_MIN,
                stability_fast: S_MIN,
            }
        } else {
            let rating = review.rating.clamp(1, 4);
            let stability = self.init_stability(rating).clamp(S_MIN, S_MAX);
            MemoryState {
                stability,
                difficulty: self.init_difficulty(rating).clamp(D_MIN, D_MAX),
                stability_fast: if self.version == ModelVersion::Fsrs7 {
                    (stability * 0.8).clamp(S_MIN, S_MAX)
                } else {
                    stability
                },
            }
        }
    }

    pub(crate) fn forward_reviews(
        &self,
        reviews: &[FSRSReview],
        starting_state: Option<MemoryState>,
    ) -> MemoryState {
        let (mut state, start_index) = if let Some(state) = starting_state {
            (state, 0)
        } else if reviews.is_empty() {
            (
                MemoryState {
                    stability: 0.0,
                    difficulty: 0.0,
                    stability_fast: 0.0,
                },
                0,
            )
        } else {
            (self.init_state_from_first_review(&reviews[0]), 1)
        };
        for (index, review) in reviews.iter().enumerate().skip(start_index) {
            state = self.step(review.delta_t, review.rating, state, index);
        }
        state
    }

    pub(crate) fn memory_state_from_sm2_inner(
        &self,
        ease_factor: f32,
        interval: f32,
        sm2_retention: f32,
    ) -> Result<MemoryState> {
        match self.version {
            ModelVersion::Fsrs6 => model_v6::memory_state_from_sm2_scalar(
                &self.parameters,
                ease_factor,
                interval,
                sm2_retention,
            ),
            ModelVersion::Fsrs7 => {
                let stability = interval.max(S_MIN).clamp(S_MIN, S_MAX);
                Ok(MemoryState {
                    stability,
                    difficulty: 5.0,
                    stability_fast: (stability * 0.8).clamp(S_MIN, S_MAX),
                })
            }
        }
    }
}

pub fn check_and_fill_parameters(parameters: &Parameters) -> Result<Vec<f32>, FSRSError> {
    let parameters = if parameters.len() == model_v7::PARAM_LEN {
        parameters.to_vec()
    } else if let Some(parameters) = model_v6::check_and_fill_parameters_fsrs6(parameters) {
        parameters
    } else {
        return Err(FSRSError::InvalidParameters);
    };
    if parameters.iter().any(|value| !value.is_finite()) {
        return Err(FSRSError::InvalidParameters);
    }
    Ok(parameters)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::FSRS6_DEFAULT_PARAMETERS;

    #[test]
    fn model_version_selection() {
        assert_eq!(FSRS::default().version(), ModelVersion::Fsrs6);
        assert_eq!(
            FSRS::new(&FSRS6_DEFAULT_PARAMETERS).unwrap().version(),
            ModelVersion::Fsrs6
        );
    }

    #[test]
    fn rejects_invalid_parameters() {
        assert!(FSRS::new(&[1.0]).is_err());
        assert!(FSRS::new(DEFAULT_PARAMETERS.as_slice()).is_ok());
    }
}

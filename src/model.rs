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

/// The FSRS algorithm version used by a model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelVersion {
    Fsrs6,
    Fsrs7,
}

impl ModelVersion {
    pub(crate) fn from_param_count(param_count: usize) -> Self {
        if param_count == 0 || param_count == model_v7::PARAM_LEN {
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

/// The main FSRS model. Empty parameters default to FSRS-7.
/// Parameter counts 17, 19, and 21 select FSRS-6; 34 selects FSRS-7.
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
    /// Creates a model, using FSRS-7 defaults when `parameters` is empty.
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

    /// Returns the algorithm version used by this model.
    ///
    /// ```
    /// use fsrs::{FSRS, DEFAULT_PARAMETERS, FSRS6_DEFAULT_PARAMETERS, ModelVersion};
    ///
    /// assert_eq!(FSRS::new(&DEFAULT_PARAMETERS)?.version(), ModelVersion::Fsrs7);
    /// assert_eq!(
    ///     FSRS::new(&FSRS6_DEFAULT_PARAMETERS)?.version(),
    ///     ModelVersion::Fsrs6,
    /// );
    /// # Ok::<(), fsrs::FSRSError>(())
    /// ```
    pub const fn version(&self) -> ModelVersion {
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
                model_v7::memory_state_from_sm2_scalar(&self.parameters, interval, sm2_retention)
            }
        }
    }
}

/// Validates parameters and fills empty input with FSRS-7 defaults.
/// Legacy 17- and 19-parameter inputs are expanded to 21 FSRS-6 parameters.
pub fn check_and_fill_parameters(parameters: &Parameters) -> Result<Vec<f32>, FSRSError> {
    let parameters = match ModelVersion::from_param_count(parameters.len()) {
        ModelVersion::Fsrs6 => model_v6::check_and_fill_parameters_fsrs6(parameters),
        ModelVersion::Fsrs7 => model_v7::check_and_fill_parameters_fsrs7(parameters),
    }
    .ok_or(FSRSError::InvalidParameters)?;
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
        let expected = FSRS::new(&DEFAULT_PARAMETERS).unwrap();
        for model in [FSRS::default(), FSRS::new(&[]).unwrap(), expected.clone()] {
            assert_eq!(model.version(), ModelVersion::Fsrs7);
            assert_eq!(model.parameters(), expected.parameters());
        }
        for count in [0, 34] {
            assert_eq!(ModelVersion::from_param_count(count), ModelVersion::Fsrs7);
        }
        assert_eq!(check_and_fill_parameters(&[]).unwrap(), DEFAULT_PARAMETERS);
        assert_eq!(
            check_and_fill_parameters(&DEFAULT_PARAMETERS).unwrap(),
            DEFAULT_PARAMETERS
        );
        for count in [17, 19, 21] {
            let input = &FSRS6_DEFAULT_PARAMETERS[..count];
            let parameters = check_and_fill_parameters(input).unwrap();
            assert_eq!(parameters.len(), 21);
            assert_eq!(FSRS::new(input).unwrap().version(), ModelVersion::Fsrs6);
            match count {
                17 => {
                    assert_eq!(parameters[4], input[5].mul_add(2.0, input[4]));
                    assert_eq!(parameters[5], input[5].mul_add(3.0, 1.0).ln() / 3.0);
                    assert_eq!(parameters[6], input[6] + 0.5);
                    assert_eq!(&parameters[17..20], &[0.0; 3]);
                    assert_eq!(parameters[20], crate::FSRS5_DEFAULT_DECAY);
                }
                19 => {
                    assert_eq!(&parameters[..19], input);
                    assert_eq!(&parameters[19..], &[0.0, crate::FSRS5_DEFAULT_DECAY]);
                }
                21 => assert_eq!(parameters, input),
                _ => unreachable!(),
            }
        }
    }

    #[test]
    fn version_specific_parameter_defaults() {
        assert_eq!(
            model_v6::check_and_fill_parameters_fsrs6(&[]).unwrap(),
            FSRS6_DEFAULT_PARAMETERS
        );
        assert_eq!(
            model_v7::check_and_fill_parameters_fsrs7(&[]).unwrap(),
            DEFAULT_PARAMETERS
        );
        let mut parameters = DEFAULT_PARAMETERS;
        parameters[0] = 1.0;
        assert_eq!(
            model_v7::check_and_fill_parameters_fsrs7(&parameters).unwrap(),
            parameters
        );
        assert!(model_v7::check_and_fill_parameters_fsrs7(&FSRS6_DEFAULT_PARAMETERS).is_none());
    }

    #[test]
    fn rejects_invalid_parameters() {
        for count in (1..=35).filter(|count| ![17, 19, 21, 34].contains(count)) {
            assert!(matches!(
                FSRS::new(&vec![1.0; count]),
                Err(FSRSError::InvalidParameters)
            ));
        }
        for count in [17, 19, 21, 34] {
            for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
                let mut parameters = vec![1.0; count];
                parameters[0] = value;
                assert!(matches!(
                    FSRS::new(&parameters),
                    Err(FSRSError::InvalidParameters)
                ));
            }
        }
    }
}

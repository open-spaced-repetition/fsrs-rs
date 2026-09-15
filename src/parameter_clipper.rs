use crate::{inference::Parameters, model::ModelVersion};

#[path = "parameter_clipper_v6.rs"]
mod parameter_clipper_v6;
#[path = "parameter_clipper_v7.rs"]
mod parameter_clipper_v7;

pub(crate) fn clip_parameters(
    parameters: &Parameters,
    num_relearning_steps: usize,
    enable_short_term: bool,
) -> Vec<f32> {
    let mut parameters = parameters.to_vec();
    clip_parameters_in_place(&mut parameters, num_relearning_steps, enable_short_term);
    parameters
}

pub(crate) fn clip_parameters_in_place(
    parameters: &mut [f32],
    num_relearning_steps: usize,
    enable_short_term: bool,
) {
    match ModelVersion::from_param_count(parameters.len()) {
        ModelVersion::Fsrs7 => {
            parameter_clipper_v7::clip_fsrs7_parameters(parameters);
        }
        ModelVersion::Fsrs6 => {
            parameter_clipper_v6::clip_fsrs6_parameters(
                parameters,
                num_relearning_steps,
                enable_short_term,
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::DEFAULT_PARAMETERS;

    #[test]
    fn clips_fsrs7() {
        let mut parameters = DEFAULT_PARAMETERS.to_vec();
        clip_parameters_in_place(&mut parameters, 1, false);
        assert!(parameters[26] > 0.0);
        assert!(parameters.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn handles_nan_without_panicking() {
        let mut parameters = DEFAULT_PARAMETERS.to_vec();
        parameters[0] = f32::NAN;
        clip_parameters_in_place(&mut parameters, 1, true);
        assert!(parameters.iter().all(|value| value.is_finite()));
    }
}

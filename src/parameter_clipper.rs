use crate::{inference::Parameters, model::ModelVersion};
use burn::{
    module::Param,
    tensor::{Tensor, TensorData, backend::Backend},
};

#[path = "parameter_clipper_v6.rs"]
mod parameter_clipper_v6;
#[path = "parameter_clipper_v7.rs"]
mod parameter_clipper_v7;

pub(crate) fn parameter_clipper<B: Backend>(
    parameters: Param<Tensor<B, 1>>,
    num_relearning_steps: usize,
    enable_short_term: bool,
) -> Param<Tensor<B, 1>> {
    let (id, val) = parameters.consume();
    let mut clipped = clip_parameters(
        &val.to_data().to_vec().unwrap(),
        num_relearning_steps,
        enable_short_term,
    );
    if !enable_short_term
        && matches!(
            ModelVersion::from_param_count(clipped.len()),
            ModelVersion::Fsrs7
        )
    {
        // FSRS-7: w[26] controls short-term mixing.
        // Forcing it to 0 disables the short-term path (long-term only).
        clipped[26] = 0.0;
    }
    Param::initialized(
        id,
        Tensor::from_data(TensorData::new(clipped, val.shape()), &val.device()).require_grad(),
    )
}

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
    use crate::{DEFAULT_PARAMETERS, test_helpers::Tensor};
    use burn::backend::ndarray::NdArrayDevice;
    static DEVICE: NdArrayDevice = NdArrayDevice::Cpu;

    #[test]
    fn test_parameter_clipper_works() {
        let tensor = Tensor::from_floats(
            [0.0, -1000.0, 1000.0, 0.0, 1000.0, -1000.0, 1.0, 0.25, -0.1],
            &DEVICE,
        );

        let param = parameter_clipper(Param::from_tensor(tensor), 1, true);
        let values = &param.to_data().to_vec::<f32>().unwrap();

        assert_eq!(
            values,
            &[0.0001, 0.0001, 100.0, 0.0001, 10.0, 0.001, 1.0, 0.25, 0.0]
        );
    }

    #[test]
    fn test_parameter_clipper_works_with_num_relearning_steps() {
        use crate::test_helpers::TestHelper;
        let tensor = Tensor::from_floats(DEFAULT_PARAMETERS, &DEVICE);

        let param = parameter_clipper(Param::from_tensor(tensor), 2, true);
        let values = &param.to_data().to_vec::<f32>().unwrap();

        values[17..=19].assert_approx_eq([0.3072, 3.5875, 0.303]);
    }

    #[test]
    fn test_fsrs7_clipper_monotonic_bounds() {
        let mut params = vec![1000.0; 35];
        params[27] = -1.0;
        params[28] = 10.0;
        params[29] = 0.1;
        params[30] = 2.0;
        let clipped = clip_parameters(&params, 1, true);
        assert_eq!(clipped.len(), 35);
        assert!(clipped[1] >= clipped[0]);
        assert!(clipped[2] >= clipped[1]);
        assert!(clipped[3] >= clipped[2]);
        assert!(clipped[28] >= clipped[27]);
        assert!(clipped[30] >= clipped[29]);
    }

    #[test]
    fn test_clip_parameters_in_place_matches_allocating_wrapper() {
        let params = vec![1000.0; 35];
        let expected = clip_parameters(&params, 1, true);
        let mut actual = params;
        clip_parameters_in_place(&mut actual, 1, true);
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_fsrs7_clipper_respects_disable_short_term() {
        use burn::backend::ndarray::NdArrayDevice;
        let params = Tensor::from_floats(DEFAULT_PARAMETERS, &NdArrayDevice::Cpu);
        let clipped_on = parameter_clipper(Param::from_tensor(params.clone()), 1, true)
            .to_data()
            .to_vec::<f32>()
            .unwrap();
        let clipped_off = parameter_clipper(Param::from_tensor(params), 1, false)
            .to_data()
            .to_vec::<f32>()
            .unwrap();
        assert!(clipped_on[26] > 0.0);
        assert_eq!(clipped_off[26], 0.0);
    }

    #[test]
    fn test_fsrs7_clipper_handles_nan_without_panic() {
        let mut params = DEFAULT_PARAMETERS.to_vec();
        for idx in [0, 1, 2, 3, 27, 28, 29, 30] {
            params[idx] = f32::NAN;
        }
        let clipped = clip_parameters(&params, 1, true);
        assert_eq!(clipped.len(), 35);
        assert!(clipped.iter().all(|v| v.is_finite()));
        assert!(clipped[1] >= clipped[0]);
        assert!(clipped[2] >= clipped[1]);
        assert!(clipped[3] >= clipped[2]);
        assert!(clipped[28] >= clipped[27]);
        assert!(clipped[30] >= clipped[29]);
    }
}

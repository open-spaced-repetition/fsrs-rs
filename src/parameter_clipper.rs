use crate::{
    inference::Parameters,
    parameter_initialization::INIT_S_MAX,
    simulation::{D_MAX, D_MIN, S_MIN},
};
use burn::{
    module::Param,
    tensor::{Tensor, TensorData, backend::Backend},
};

pub(crate) fn parameter_clipper<B: Backend>(
    parameters: Param<Tensor<B, 1>>,
    num_relearning_steps: usize,
) -> Param<Tensor<B, 1>> {
    let (id, val) = parameters.consume();
    let clipped = clip_parameters(&val.to_data().to_vec().unwrap(), num_relearning_steps);
    Param::initialized(
        id,
        Tensor::from_data(TensorData::new(clipped, val.shape()), &B::Device::default())
            .require_grad(),
    )
}

pub(crate) fn clip_parameters(parameters: &Parameters, num_relearning_steps: usize) -> Vec<f32> {
    let mut parameters = parameters.to_vec();
    // PLS = w11 * D ^ -w12 * [(S + 1) ^ w13 - 1] * e ^ (w14 * (1 - R))
    // PLS * e ^ (num_relearning_steps * w17 * w18) should be <= S
    // Given D = 1, R = 0.7, S = 1, PLS is equal to w11 * (2 ^ w13 - 1) * e ^ (w14 * 0.3)
    // So num_relearning_steps * w17 * w18 + ln(w11) + ln(2 ^ w13 - 1) + w14 * 0.3 should be <= ln(1)
    // => num_relearning_steps * w17 * w18 <= - ln(w11) - ln(2 ^ w13 - 1) - w14 * 0.3
    // => w17 * w18 <= -[ln(w11) + ln(2 ^ w13 - 1) + w14 * 0.3] / num_relearning_steps
    let w17_w18_ceiling = if num_relearning_steps > 1 {
        (-(parameters[11].ln() + (2.0f32.powf(parameters[13]) - 1.0).ln() + parameters[14] * 0.3)
            / num_relearning_steps as f32)
            .max(0.01)
            .sqrt()
            .min(2.0)
    } else {
        2.0
    };
    // https://regex101.com/r/21mXNI/1
    let clamps: [(f32, f32); 21] = [
        (S_MIN, INIT_S_MAX),
        (S_MIN, INIT_S_MAX),
        (S_MIN, INIT_S_MAX),
        (S_MIN, INIT_S_MAX),
        (D_MIN, D_MAX),
        (0.001, 4.0),
        (0.001, 4.0),
        (0.001, 0.75),
        (0.0, 4.5),
        (0.0, 0.8),
        (0.001, 3.5),
        (0.001, 5.0),
        (0.001, 0.25),
        (0.001, 0.9),
        (0.0, 4.0),
        (0.0, 1.0),
        (1.0, 6.0),
        (0.0, w17_w18_ceiling),
        (0.0, w17_w18_ceiling),
        (0.0, 0.8),
        (0.1, 0.8),
    ];

    parameters
        .iter_mut()
        .zip(clamps)
        .for_each(|(w, (low, high))| *w = w.clamp(low, high));
    parameters
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{DEFAULT_PARAMETERS, test_helpers::Tensor};
    use burn::backend::ndarray::NdArrayDevice;

    #[test]
    fn parameter_clipper_works() {
        let device = NdArrayDevice::Cpu;
        let tensor = Tensor::from_floats(
            [0.0, -1000.0, 1000.0, 0.0, 1000.0, -1000.0, 1.0, 0.25, -0.1],
            &device,
        );

        let param = parameter_clipper(Param::from_tensor(tensor), 1);
        let values = &param.to_data().to_vec::<f32>().unwrap();

        assert_eq!(
            values,
            &[0.001, 0.001, 100.0, 0.001, 10.0, 0.001, 1.0, 0.25, 0.0]
        );
    }

    #[test]
    fn parameter_clipper_works_with_num_relearning_steps() {
        use crate::test_helpers::TestHelper;
        let device = NdArrayDevice::Cpu;
        let tensor = Tensor::from_floats(DEFAULT_PARAMETERS, &device);

        let param = parameter_clipper(Param::from_tensor(tensor), 2);
        let values = &param.to_data().to_vec::<f32>().unwrap();

        values[17..=19].assert_approx_eq([0.5425, 0.0912, 0.0658]);
    }
}

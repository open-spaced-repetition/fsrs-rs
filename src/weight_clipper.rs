use crate::{
    inference::{Parameters, S_MIN},
    pre_training::INIT_S_MAX,
};
use burn::tensor::{backend::Backend, Data, Tensor};

pub fn weight_clipper<B: Backend>(parameters: Tensor<B, 1>) -> Tensor<B, 1> {
    let val = clip_parameters(&parameters.to_data().convert().value);
    Tensor::from_data(
        Data::new(val, parameters.shape()).convert(),
        &B::Device::default(),
    )
}

pub fn clip_parameters(parameters: &Parameters) -> Vec<f32> {
    // https://regex101.com/r/21mXNI/1
    const CLAMPS: [(f32, f32); 17] = [
        (S_MIN, INIT_S_MAX),
        (S_MIN, INIT_S_MAX),
        (S_MIN, INIT_S_MAX),
        (S_MIN, INIT_S_MAX),
        (1.0, 10.0),
        (0.1, 5.0),
        (0.1, 5.0),
        (0.0, 0.75),
        (0.0, 4.0),
        (0.0, 0.8),
        (0.01, 3.0),
        (0.5, 5.0),
        (0.01, 0.2),
        (0.01, 0.9),
        (0.01, 3.0),
        (0.0, 1.0),
        (1.0, 6.0),
    ];

    let mut parameters = parameters.to_vec();
    parameters
        .iter_mut()
        .zip(CLAMPS)
        .for_each(|(w, (low, high))| *w = w.clamp(low, high));
    parameters
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_helpers::Tensor;
    use burn::backend::ndarray::NdArrayDevice;

    #[test]
    fn weight_clipper_works() {
        let device = NdArrayDevice::Cpu;
        let tensor = Tensor::from_floats(
            [0.0, -1000.0, 1000.0, 0.0, 1000.0, -1000.0, 1.0, 0.25, -0.1],
            &device,
        );

        let param: Tensor<1> = weight_clipper(tensor);
        let values = &param.to_data().value;

        assert_eq!(
            values,
            &[0.01, 0.01, 100.0, 0.01, 10.0, 0.1, 1.0, 0.25, 0.0]
        );
    }
}

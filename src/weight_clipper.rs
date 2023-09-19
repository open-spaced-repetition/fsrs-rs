use crate::inference::Weights;
use burn::tensor::{backend::Backend, Data, Tensor};

pub(crate) fn weight_clipper<B: Backend>(weights: Tensor<B, 1>) -> Tensor<B, 1> {
    let val = clip_weights(&weights.to_data().convert().value);
    Tensor::from_data(Data::new(val, weights.shape()).convert())
}

pub(crate) fn clip_weights(weights: &Weights) -> Vec<f32> {
    // https://regex101.com/r/21mXNI/1
    const CLAMPS: [(f32, f32); 17] = [
        (0.1, 100.0),
        (0.1, 100.0),
        (0.1, 100.0),
        (0.1, 100.0),
        (1.0, 10.0),
        (0.1, 5.0),
        (0.1, 5.0),
        (0.0, 0.5),
        (0.0, 3.0),
        (0.1, 0.8),
        (0.01, 2.5),
        (0.5, 5.0),
        (0.01, 0.2),
        (0.01, 0.9),
        (0.01, 2.0),
        (0.0, 1.0),
        (1.0, 10.0),
    ];

    let mut weights = weights.to_vec();
    weights
        .iter_mut()
        .zip(CLAMPS)
        .for_each(|(w, (low, high))| *w = w.clamp(low, high));
    weights
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_helpers::Tensor;

    #[test]
    fn weight_clipper_works() {
        let tensor =
            Tensor::from_floats([0.0, -1000.0, 1000.0, 0.0, 1000.0, -1000.0, 1.0, 0.25, -0.1]);

        let param: Tensor<1> = weight_clipper(tensor);
        let values = &param.to_data().value;

        assert_eq!(values, &[0.1, 0.1, 100.0, 0.1, 10.0, 0.1, 1.0, 0.25, 0.0]);
    }
}

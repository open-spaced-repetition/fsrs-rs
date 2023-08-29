use burn::tensor::{backend::Backend, Data, Tensor};

pub fn weight_clipper<B: Backend<FloatElem = f32>>(weights: Tensor<B, 1>) -> Tensor<B, 1> {
    const CLAMPS: [(f32, f32); 13] = [
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
    // https://regex101.com/r/21mXNI/1

    let val: &mut Vec<f32> = &mut weights.to_data().value;

    val.iter_mut()
        .skip(4)
        .zip(CLAMPS)
        .for_each(|(w, (low, high))| *w = w.clamp(low, high));

    Tensor::from_data(Data::new(val.clone(), weights.shape()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn weight_clipper_works() {
        type Backend = burn_ndarray::NdArrayBackend<f32>;
        //type AutodiffBackend = burn_autodiff::ADBackendDecorator<Backend>;

        let tensor = Tensor::from_floats([
            0.0, -1000.0, 1000.0, 0.0, // Ignored
            1000.0, -1000.0, 1.0, 0.25, -0.1,
        ]); // Clamped (1.0, 10.0),(0.1, 5.0),(0.1, 5.0),(0.0, 0.5),

        let param: Tensor<Backend, 1> = weight_clipper(tensor);
        let values = &param.to_data().value;

        assert_eq!(
            *values,
            vec![0.0, -1000.0, 1000.0, 0.0, 10.0, 0.1, 1.0, 0.25, 0.0]
        );
    }
}

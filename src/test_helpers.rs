use burn::backend::autodiff::Autodiff;
pub type NdArrayAutodiff = Autodiff<burn::backend::NdArray>;
use burn::tensor::{Element, Float, TensorData};

pub type Model = crate::model::Model<NdArrayAutodiff>;
pub type Tensor<const D: usize, K = Float> = burn::tensor::Tensor<NdArrayAutodiff, D, K>;

pub(crate) fn assert_approx_eq<const N: usize, T>(a: [T; N], b: [T; N])
where
    T: Copy + std::fmt::Debug + PartialEq + Element,
    f64: From<T>,
{
    TensorData::from(a).assert_approx_eq(&TensorData::from(b), 4);
}

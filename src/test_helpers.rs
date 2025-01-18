use burn::backend::autodiff::Autodiff;
pub type NdArrayAutodiff = Autodiff<burn::backend::NdArray>;
use burn::tensor::{Data, Float};

pub type Model = crate::model::Model<NdArrayAutodiff>;
pub type Tensor<const D: usize, K = Float> = burn::tensor::Tensor<NdArrayAutodiff, D, K>;

pub(crate) fn assert_approx_eq<const N: usize, T>(a: [T; N], b: [T; N])
where
    T: Copy + std::fmt::Debug + PartialEq,
    f64: From<T>,
{
    Data::from(a).assert_approx_eq(&Data::from(b), 4);
}

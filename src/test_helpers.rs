use burn::backend::autodiff::Autodiff;
pub type NdArrayAutodiff = Autodiff<burn::backend::NdArray>;
use burn::tensor::Float;

pub type Model = crate::model::Model<NdArrayAutodiff>;
pub type Tensor<const D: usize, K = Float> = burn::tensor::Tensor<NdArrayAutodiff, D, K>;

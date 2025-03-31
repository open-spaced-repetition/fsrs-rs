use burn::backend::autodiff::Autodiff;
pub type NdArrayAutodiff = Autodiff<burn::backend::NdArray>;
pub type Model = crate::model::Model<NdArrayAutodiff>;
pub type Tensor<const D: usize, K = burn::tensor::Float> =
    burn::tensor::Tensor<NdArrayAutodiff, D, K>;

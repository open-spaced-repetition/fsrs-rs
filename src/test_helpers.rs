use burn::backend::NdArrayAutodiffBackend;
use burn::tensor::Float;

pub type Model = crate::model::Model<NdArrayAutodiffBackend>;
pub type Tensor<const D: usize, K = Float> = burn::tensor::Tensor<NdArrayAutodiffBackend, D, K>;

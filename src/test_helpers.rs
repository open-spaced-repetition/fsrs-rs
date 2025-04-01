use burn::{
    backend::autodiff::Autodiff,
    tensor::{Element, TensorData},
};
pub type NdArrayAutodiff = Autodiff<burn::backend::NdArray>;
pub type Model = crate::model::Model<NdArrayAutodiff>;
pub type Tensor<const D: usize, K = burn::tensor::Float> =
    burn::tensor::Tensor<NdArrayAutodiff, D, K>;

#[track_caller]
fn assert_approx_eq<const N: usize, T>(a: [T; N], b: [T; N])
where
    T: Copy + std::fmt::Debug + PartialEq + Element,
    f64: From<T>,
{
    TensorData::from(a).assert_approx_eq(&TensorData::from(b), 4);
}

pub trait TestHelper<const N: usize, T> {
    fn assert_approx_eq(&self, b: [T; N])
    where
        T: Copy + std::fmt::Debug + PartialEq + Element,
        f64: From<T>;
}

impl<T, const N: usize> TestHelper<N, T> for [T; N] {
    #[track_caller]
    fn assert_approx_eq(&self, b: [T; N])
    where
        T: Copy + std::fmt::Debug + PartialEq + Element,
        f64: From<T>,
    {
        assert_approx_eq(*self, b);
    }
}

impl<T, const N: usize> TestHelper<N, T> for Vec<T> {
    #[track_caller]
    fn assert_approx_eq(&self, b: [T; N])
    where
        T: Copy + std::fmt::Debug + PartialEq + Element,
        f64: From<T>,
    {
        let a = self.to_owned().try_into().unwrap();
        assert_approx_eq(a, b);
    }
}
impl<T, const N: usize> TestHelper<N, T> for [T] {
    #[track_caller]
    fn assert_approx_eq(&self, b: [T; N])
    where
        T: Copy + std::fmt::Debug + PartialEq + Element,
        f64: From<T>,
    {
        self.to_vec().assert_approx_eq(b);
    }
}

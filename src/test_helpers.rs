pub trait TestHelper<const N: usize, T> {
    fn assert_approx_eq(&self, expected: [T; N]);
}

impl<T, const N: usize> TestHelper<N, T> for [T; N]
where
    T: Copy + Into<f64> + core::fmt::Debug,
{
    #[track_caller]
    fn assert_approx_eq(&self, expected: [T; N]) {
        for (actual, expected) in self.iter().zip(expected) {
            let difference = (Into::<f64>::into(*actual) - Into::<f64>::into(expected)).abs();
            assert!(
                difference <= 1e-4,
                "actual {actual:?}, expected {expected:?}"
            );
        }
    }
}

impl<T, const N: usize> TestHelper<N, T> for Vec<T>
where
    T: Copy + Into<f64> + core::fmt::Debug,
{
    #[track_caller]
    fn assert_approx_eq(&self, expected: [T; N]) {
        self.as_slice().assert_approx_eq(expected);
    }
}

impl<T, const N: usize> TestHelper<N, T> for [T]
where
    T: Copy + Into<f64> + core::fmt::Debug,
{
    #[track_caller]
    fn assert_approx_eq(&self, expected: [T; N]) {
        assert_eq!(self.len(), N);
        for (actual, expected) in self.iter().zip(expected) {
            let difference = (Into::<f64>::into(*actual) - Into::<f64>::into(expected)).abs();
            assert!(
                difference <= 1e-4,
                "actual {actual:?}, expected {expected:?}"
            );
        }
    }
}

// This file previously contained burn-specific test helpers.
// These have been removed as part of the migration to candle.
// Tests in other modules have been updated to use local helpers
// or direct candle tensor comparisons.
// If generic Rust test helpers (not tied to any tensor library)
// are needed in the future, they can be added here.

// Test helpers for candle-based tests
use candle_core::{Device, Tensor};
use crate::error::Result;

pub type NdArrayAutodiff = Device; // For compatibility with existing test code

pub struct TestHelper;

impl TestHelper {
    pub fn new() -> Self {
        TestHelper
    }
}

// Helper trait for approximate equality testing of arrays
pub trait AssertApproxEq<T> {
    fn assert_approx_eq(&self, other: T);
}

impl AssertApproxEq<[f32; 1]> for [f32; 1] {
    fn assert_approx_eq(&self, other: [f32; 1]) {
        let diff = (self[0] - other[0]).abs();
        assert!(diff < 1e-5, "Arrays not approximately equal: {:?} vs {:?}, diff: {}", self, other, diff);
    }
}

impl AssertApproxEq<[f32; 4]> for [f32; 4] {
    fn assert_approx_eq(&self, other: [f32; 4]) {
        for (a, b) in self.iter().zip(other.iter()) {
            let diff = (a - b).abs();
            assert!(diff < 1e-5, "Arrays not approximately equal: {:?} vs {:?}, diff at element: {}", self, other, diff);
        }
    }
}

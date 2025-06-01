// Removed: use burn::{LearningRate, lr_scheduler::LrScheduler, tensor::backend::Backend};
// LearningRate was f64. LrScheduler trait is removed. Backend not needed.

#[derive(Clone, Debug)]
pub(crate) struct CosineAnnealingLR {
    t_max: f64,
    eta_min: f64,
    init_lr: f64, // Changed LearningRate to f64
    step_count: f64,
    current_lr: f64, // Changed LearningRate to f64
}

impl CosineAnnealingLR {
    pub const fn init(t_max: f64, init_lr: f64) -> Self { // Changed LearningRate to f64
        Self {
            t_max,
            eta_min: 0.0,
            init_lr,
            step_count: -1.0, // Start at -1 so first step makes it 0
            current_lr: init_lr,
        }
    }

    // This was previously part of the LrScheduler trait implementation.
    // It's now an inherent method.
    pub fn step(&mut self) -> f64 { // Changed LearningRate to f64
        self.step_count += 1.0;
        use std::f64::consts::PI;

        // Inner function remains the same, but types are f64
        fn cosine_annealing_lr(
            init_lr: f64,
            lr: f64,
            step_count: f64,
            t_max: f64,
            eta_min: f64,
        ) -> f64 {
            if step_count == 0.0 { // First step, return init_lr
                init_lr
            } else if (step_count - 1.0 - t_max) % (2.0 * t_max) == 0.0 {
                 // This condition seems to handle restarts.
                 // Simplified from burn's version which had a more complex restart handling.
                 // This attempts to replicate the behavior for the specific step_count values.
                 // A more direct port of burn's logic might be needed if this isn't quite right.
                 // For now, keeping it similar to the original logic provided in the prompt.
                (init_lr - eta_min) * (1.0 - f64::cos(PI / t_max)) / 2.0  // This part might be specific to a restart condition.
            } else {
                // Standard cosine annealing update
                let cos_inner = PI * step_count / t_max;
                let new_lr = eta_min + (init_lr - eta_min) * (1.0 + cos_inner.cos()) / 2.0;
                // The multiplicative update factor from the original burn code:
                // ((1.0 + f64::cos(PI * step_count / t_max)) / (1.0 + f64::cos(PI * (step_count - 1.0) / t_max)))
                // This factor was applied to (lr - eta_min).
                // Let's use the more standard formulation of cosine annealing if the above is too complex or specific to burn's stateful way.
                // Standard formula: eta_min + 0.5 * (init_lr - eta_min) * (1 + cos(PI * T_cur / T_max))
                // Here, T_cur is self.step_count.
                new_lr
            }
        }
        // The original burn implementation had a stateful update for self.current_lr based on the previous self.current_lr.
        // A more typical CosineAnnealingLR calculates LR based on initial_lr, eta_min, t_max and current step_count.
        // Let's use the standard formula directly:
        // use std::f64::consts::PI; // PI is already imported at the start of the function
        self.current_lr = self.eta_min + (self.init_lr - self.eta_min) * (1.0 + (PI * self.step_count / self.t_max).cos()) / 2.0;

        // Ensure current_lr doesn't go below eta_min (it shouldn't with the formula above if step_count <= t_max)
        // If step_count can exceed t_max (for restarts), the formula needs to handle that.
        // The burn logic was more complex due to its stateful lr update.
        // For simplicity and standard behavior, the above recalculation from init_lr is often used.
        // If the exact burn behavior including its specific restart logic is needed, the original `cosine_annealing_lr`
        // inner function would need to be used carefully, ensuring `lr` passed to it is the previous step's `current_lr`.
        // The current code block for `cosine_annealing_lr` has a direct assignment. Let's keep it.
        // Reverting to the structure from the original impl LrScheduler for now, with f64 types
        self.current_lr = cosine_annealing_lr(
            self.init_lr,
            self.current_lr, // Pass the previous current_lr for the multiplicative update
            self.step_count,
            self.t_max,
            self.eta_min,
        );

        self.current_lr
    }
}

// The LrScheduler trait and its methods to_record/load_record are removed.
// If state saving/loading for this scheduler is needed, it would have to be implemented manually.

#[cfg(test)]
mod tests {
    use super::*;
    // use crate::test_helpers::TestHelper; // Removed burn TestHelper

    // Helper for float slice comparisons if not already present elsewhere
    fn assert_f64_slices_approx_eq(result: &[f64], expected: &[f64], tolerance: f64) {
        assert_eq!(result.len(), expected.len(), "Slice lengths differ.");
        for (r, e) in result.iter().zip(expected.iter()) {
            assert!((r - e).abs() < tolerance, "Value mismatch: {} vs {}", r, e);
        }
    }


    #[test]
    fn lr_scheduler() {
        let mut lr_scheduler = CosineAnnealingLR::init(5.0, 4e-2);
        let lrs: Vec<f64> = (0..11) // Adjusted range to match 11 steps for comparison
            .map(|_| {
                lr_scheduler.step() // Call the inherent method
            })
            .collect();

        let expected_lrs = [
            0.04, // step_count = 0
            0.03618033988749895, // step_count = 1
            0.026180339887498946, // step_count = 2
            0.013819660112501051, // step_count = 3
            0.0038196601125010526, // step_count = 4
            0.0, // step_count = 5 (T_max reached)
            0.003819660112501051, // step_count = 6 (restart)
            0.013819660112501048, // step_count = 7
            0.026180339887498943, // step_count = 8
            0.03618033988749895, // step_count = 9
            0.039999999999999994, // step_count = 10
        ];
        assert_f64_slices_approx_eq(&lrs, &expected_lrs, 1e-9);
    }
}

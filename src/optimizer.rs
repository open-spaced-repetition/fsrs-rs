//! Pure Rust Adam optimizer tailored for the 21-parameter FSRS model.
//!
//! This replaces Burn's tensor-based Adam optimizer, enabling zero-allocation
//! gradient descent. It strictly mirrors Burn's `AdaptiveMomentum` math.

#[derive(Debug, Clone)]
pub struct AdamConfig {
    pub beta_1: f32,
    pub beta_2: f32,
    pub epsilon: f32,
}

impl Default for AdamConfig {
    fn default() -> Self {
        Self {
            beta_1: 0.9,
            beta_2: 0.999,
            epsilon: 1e-8,
        }
    }
}

/// A zero-allocation Adam optimizer for 21 parameters.
#[derive(Debug, Clone)]
pub struct AdamOptimizer {
    config: AdamConfig,
    time: usize,
    moment_1: [f32; 21],
    moment_2: [f32; 21],
}

impl AdamOptimizer {
    pub fn new(config: AdamConfig) -> Self {
        Self {
            config,
            time: 0,
            moment_1: [0.0; 21],
            moment_2: [0.0; 21],
        }
    }

    /// Takes a single Adam step, mutating `w` in place using the provided `grad`.
    pub fn step(&mut self, lr: f32, w: &mut [f32; 21], grad: &[f32; 21]) {
        self.time += 1;

        let b1_correction = 1.0 - self.config.beta_1.powi(self.time as i32);
        let b2_correction = 1.0 - self.config.beta_2.powi(self.time as i32);

        for i in 0..21 {
            let g = grad[i];

            self.moment_1[i] =
                self.moment_1[i] * self.config.beta_1 + g * (1.0 - self.config.beta_1);

            self.moment_2[i] =
                self.moment_2[i] * self.config.beta_2 + g * g * (1.0 - self.config.beta_2);

            let m1_corrected = self.moment_1[i] / b1_correction;
            let m2_corrected = self.moment_2[i] / b2_correction;

            let update = m1_corrected / (m2_corrected.sqrt() + self.config.epsilon);

            w[i] -= lr * update;
        }
    }
}

/// The standard deviations of the 21 parameters (from training.rs), used for L2 regularizaton.
#[allow(dead_code)]
pub static PARAMS_STDDEV: [f32; 21] = [
    6.43, 9.66, 17.58, 27.85, 0.57, 0.28, 0.6, 0.12, 0.39, 0.18, 0.33, 0.3, 0.09, 0.16, 0.57, 0.25,
    1.03, 0.31, 0.32, 0.14, 0.27,
];

/// Computes and adds L2 regularization gradients directly to the `grad` array.
///
/// Mathematically equivalent to Burn's `model.l2_regularization() + penalty.backward()`.
///
/// # Arguments
/// * `grad` - The gradient array to mutate
/// * `w` - Current parameters
/// * `init_w` - Initial parameters (at start of training)
/// * `gamma` - L2 regularization weight
/// * `batch_size` - Current batch size
/// * `total_size` - Total dataset size
#[allow(dead_code)]
pub fn add_l2_regularization_gradients(
    grad: &mut [f32; 21],
    w: &[f32; 21],
    init_w: &[f32; 21],
    gamma: f32,
    batch_size: usize,
    total_size: usize,
) {
    let factor = gamma * (batch_size as f32 / total_size as f32);

    for i in 0..21 {
        // Penalty = factor * sum( (w[i] - init_w[i])^2 / stddev[i]^2 )
        // d(Penalty)/d(w[i]) = 2.0 * factor * (w[i] - init_w[i]) / stddev[i]^2
        let stddev_sq = PARAMS_STDDEV[i] * PARAMS_STDDEV[i];
        let l2_grad = 2.0 * factor * (w[i] - init_w[i]) / stddev_sq;
        grad[i] += l2_grad;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Extracts the EXACT test data values from `test_loss_and_grad` in `src/training.rs`.
    /// Confirms that our pure Rust optimizer yields identical parameter updates.
    #[test]
    fn test_adam_matches_burn() {
        // From inference.rs DEFAULT_PARAMETERS
        let init_w: [f32; 21] = [
            0.212, 1.2931, 2.3065, 8.2956, 6.4133, 0.8334, 3.0194, 0.001, 1.8722, 0.1666, 0.796,
            1.4835, 0.0614, 0.2629, 1.6483, 0.6014, 1.8729, 0.5425, 0.0912, 0.0658, 0.1542,
        ];

        // The gradients computed by Burn's backward pass before L2
        let burn_loss_grad: [f32; 21] = [
            -0.040530164,
            -0.0041278866,
            -0.0010157757,
            0.007239434,
            0.009321215,
            -0.120117955,
            0.039143264,
            -0.8628009,
            0.5794302,
            -2.5713828,
            0.7669307,
            -0.024242667,
            0.0,
            -0.16912507,
            -0.0017008218,
            -0.061857328,
            0.28093633,
            0.064058185,
            0.0063592787,
            -0.1903223,
            0.6257775,
        ];

        // Let's test L2 Regularization matching Burn exactly
        let test_w: [f32; 21] = [
            0.252,
            1.3331,
            2.3464994,
            8.2556,
            6.3733,
            0.87340003,
            2.9794,
            0.040999997,
            1.8322,
            0.20660001,
            0.756,
            1.5235,
            0.021400042,
            0.3029,
            1.6882998,
            0.64140004,
            1.8329,
            0.5025,
            0.13119997,
            0.1058,
            0.1142,
        ];

        let mut grad = burn_loss_grad;
        add_l2_regularization_gradients(&mut grad, &test_w, &init_w, 2.0, 512, 1000);

        // Expected L2 gradients from Burn `penalty.backward()` inside `test_loss_and_grad`
        let expected_l2_grads: [f32; 21] = [
            0.0019813816,
            0.00087788026,
            0.00026506148,
            -0.000105618295,
            -0.25213888,
            1.0448985,
            -0.22755535,
            5.688889,
            -0.5385926,
            2.5283954,
            -0.75225013,
            0.9102214,
            -10.113569,
            3.1999993,
            0.2521374,
            1.3107208,
            -0.07721739,
            -0.85244584,
            0.79999936,
            4.1795917,
            -1.1237311,
        ];

        for i in 0..21 {
            let l2_only_grad = grad[i] - burn_loss_grad[i];
            let err = (l2_only_grad - expected_l2_grads[i]).abs();
            assert!(
                err < 1e-4,
                "L2 grad mismatch at [{}]: got {}, expected {}",
                i,
                l2_only_grad,
                expected_l2_grads[i]
            );
        }

        // Test Adam Step
        let mut optimizer = AdamOptimizer::new(AdamConfig::default());
        let mut w = init_w;

        // Feed the gradients from the first step of `test_loss_and_grad` (the 4-item batch)
        // These gradients already include L2 + Loss
        let combined_grad: [f32; 21] = [
            -0.095688485,
            -0.0051607806,
            -0.0012249565,
            0.007462064,
            0.03650761,
            -0.082112335,
            0.0593964,
            -2.1474836,
            0.57626534,
            -2.8751316,
            0.7154875,
            -0.028993709,
            0.0099172965,
            -0.2189217,
            -0.0017800558,
            -0.089381434,
            0.299141,
            0.068104014,
            -0.011605468,
            -0.25398168,
            0.27700496,
        ];

        // Apply parameter freeze (freeze_initial_stability, freeze_short_term)
        let mut frozen_grad = combined_grad;
        let (freeze_initial, freeze_short_term) = (true, true);
        if freeze_initial {
            frozen_grad[0..4].fill(0.0);
        }
        if freeze_short_term {
            frozen_grad[17..20].fill(0.0);
        }
        optimizer.step(0.04, &mut w, &frozen_grad);

        // Expected output after optimizer step (before parameter clipper)
        let expected_w_before_clipper: [f32; 21] = [
            0.212,
            1.2931,
            2.3065,
            8.2956,
            6.3733,
            0.87340003,
            2.9794,
            0.040999997,
            1.8322,
            0.20660001,
            0.756,
            1.5235,
            0.021400042,
            0.3029,
            1.6882998,
            0.64140004,
            1.8329,
            0.5425,
            0.0912,
            0.0658,
            0.1142,
        ];

        for i in 0..21 {
            let err = (w[i] - expected_w_before_clipper[i]).abs();
            assert!(
                err < 1e-4,
                "Adam update mismatch at [{}]: got {}, expected {}",
                i,
                w[i],
                expected_w_before_clipper[i]
            );
        }
    }
}

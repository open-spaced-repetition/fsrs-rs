#[derive(Clone, Debug)]
pub(crate) struct CosineAnnealingLR {
    t_max: f64,
    eta_min: f64,
    init_lr: f64,
    step_count: f64,
    current_lr: f64,
}

impl CosineAnnealingLR {
    pub const fn init(t_max: f64, init_lr: f64) -> Self {
        Self {
            t_max,
            eta_min: 0.0,
            init_lr,
            step_count: -1.0,
            current_lr: init_lr,
        }
    }

    pub fn step(&mut self) -> f64 {
        self.step_count += 1.0;
        use std::f64::consts::PI;
        fn cosine_annealing_lr(
            init_lr: f64,
            lr: f64,
            step_count: f64,
            t_max: f64,
            eta_min: f64,
        ) -> f64 {
            if step_count == 0.0 {
                init_lr
            } else if (step_count - 1.0 - t_max) % (2.0 * t_max) == 0.0 {
                (init_lr - eta_min) * (1.0 - f64::cos(PI / t_max)) / 2.0
            } else {
                ((1.0 + f64::cos(PI * step_count / t_max))
                    / (1.0 + f64::cos(PI * (step_count - 1.0) / t_max)))
                .mul_add(lr - eta_min, eta_min)
            }
        }
        self.current_lr = cosine_annealing_lr(
            self.init_lr,
            self.current_lr,
            self.step_count,
            self.t_max,
            self.eta_min,
        );
        self.current_lr
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_helpers::TestHelper;

    #[test]
    fn test_lr_scheduler() {
        let mut lr_scheduler = CosineAnnealingLR::init(5.0, 4e-2);
        let lrs = (1..=11)
            .map(|_| lr_scheduler.step())
            .step_by(1)
            .collect::<Vec<_>>();
        lrs.assert_approx_eq([
            0.04,
            0.03618033988749895,
            0.026180339887498946,
            0.013819660112501051,
            0.0038196601125010526,
            0.0,
            0.003819660112501051,
            0.013819660112501048,
            0.026180339887498943,
            0.03618033988749895,
            0.039999999999999994,
        ]);
    }
}

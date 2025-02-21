use burn::{LearningRate, lr_scheduler::LrScheduler, tensor::backend::Backend};
#[derive(Clone, Debug)]
pub(crate) struct CosineAnnealingLR {
    t_max: f64,
    eta_min: f64,
    init_lr: LearningRate,
    step_count: f64,
    current_lr: LearningRate,
}

impl CosineAnnealingLR {
    pub const fn init(t_max: f64, init_lr: LearningRate) -> Self {
        Self {
            t_max,
            eta_min: 0.0,
            init_lr,
            step_count: -1.0,
            current_lr: init_lr,
        }
    }
}

impl LrScheduler for CosineAnnealingLR {
    type Record<B: Backend> = usize;

    fn step(&mut self) -> LearningRate {
        self.step_count += 1.0;
        use std::f64::consts::PI;
        fn cosine_annealing_lr(
            init_lr: LearningRate,
            lr: LearningRate,
            step_count: f64,
            t_max: f64,
            eta_min: f64,
        ) -> LearningRate {
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
        // info!("lr: {}", self.current_lr);
        self.current_lr
    }

    fn to_record<B: Backend>(&self) -> Self::Record<B> {
        self.step_count as usize
    }

    fn load_record<B: Backend>(mut self, record: Self::Record<B>) -> Self {
        self.step_count = record as LearningRate;
        self
    }
}

#[cfg(test)]
mod tests {
    use crate::test_helpers::assert_approx_eq;

    use super::*;

    #[test]
    fn lr_scheduler() {
        let mut lr_scheduler = CosineAnnealingLR::init(5.0, 4e-2);
        let lrs = (1..=11)
            .map(|_| {
                LrScheduler::step(&mut lr_scheduler);
                lr_scheduler.current_lr
            })
            .step_by(1)
            .collect::<Vec<_>>();

        assert_approx_eq(
            lrs.try_into().unwrap(),
            [
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
            ],
        );
    }
}

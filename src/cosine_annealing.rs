use burn::{lr_scheduler::LRScheduler, LearningRate};
#[derive(Clone, Debug)]
pub struct CosineAnnealingLR {
    t_max: f64,
    eta_min: f64,
    init_lr: f64,
    step_count: f64,
    current_lr: LearningRate,
}

impl CosineAnnealingLR {
    pub fn init(t_max: f64, init_lr: f64) -> CosineAnnealingLR {
        CosineAnnealingLR {
            t_max,
            eta_min: 0.0,
            init_lr,
            step_count: 0.0,
            current_lr: init_lr,
        }
    }
}

impl LRScheduler for CosineAnnealingLR {
    type Record = usize;

    fn step(&mut self) -> LearningRate {
        self.step_count += 1.0;
        use std::f64::consts::PI;
        fn cosine_annealing_lr(
            init_lr: f64,
            lr: f64,
            step_count: f64,
            t_max: f64,
            eta_min: f64,
        ) -> f64 {
            let cosine_arg = PI * step_count / t_max;
            if (step_count - 1.0 - t_max) % (2.0 * t_max) == 0.0 {
                (init_lr - eta_min) * (1.0 - f64::cos(PI / t_max)) / 2.0
            } else {
                (1.0 + f64::cos(cosine_arg)) / (1.0 + f64::cos(PI * (step_count - 1.0) / t_max))
                    * (lr - eta_min)
                    + eta_min
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

    fn to_record(&self) -> Self::Record {
        self.step_count as usize
    }

    fn load_record(mut self, record: Self::Record) -> Self {
        self.step_count = record as f64;
        self
    }
}

#[test]
fn test_lr_scheduler() {
    let mut lr_scheduler = CosineAnnealingLR::init(100000.0, 1.0e-1);
    for i in 0..400000 {
        if i % 5000 == 0 {
            println!("{}", lr_scheduler.current_lr);
        }
        lr_scheduler.step();
    }
    println!("{}", lr_scheduler.current_lr);
}

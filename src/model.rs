use crate::error::{FSRSError, Result};
use crate::inference::Weights;
use crate::weight_clipper::clip_weights;
use crate::DEFAULT_WEIGHTS;
use burn::backend::ndarray::NdArrayDevice;
use burn::backend::NdArrayBackend;
use burn::{
    config::Config,
    module::{Module, Param},
    tensor::{backend::Backend, Data, Float, Shape, Tensor},
};

#[derive(Module, Debug)]
pub struct Model<B: Backend> {
    pub w: Param<Tensor<B, 1>>,
    pub config: ModelConfig,
}

pub(crate) trait Get<B: Backend, const N: usize> {
    fn get(&self, n: usize) -> Tensor<B, N>;
}

impl<B: Backend, const N: usize> Get<B, N> for Tensor<B, N> {
    fn get(&self, n: usize) -> Tensor<B, N> {
        self.clone().slice([n..(n + 1)])
    }
}

trait Pow<B: Backend, const N: usize> {
    // https://github.com/burn-rs/burn/issues/590 , after that finished, just remove this trait and below impl, all will ok.
    fn pow(&self, other: Tensor<B, N>) -> Tensor<B, N>;
}

impl<B: Backend, const N: usize> Pow<B, N> for Tensor<B, N> {
    fn pow(&self, other: Tensor<B, N>) -> Tensor<B, N> {
        // a ^ b => exp(ln(a^b)) => exp(b ln (a))
        (self.clone().log() * other).exp()
    }
}

impl<B: Backend> Model<B> {
    #[allow(clippy::new_without_default)]
    pub fn new(config: ModelConfig) -> Self {
        let initial_params = config
            .initial_stability
            .unwrap_or([0.4, 0.6, 2.4, 5.8])
            .into_iter()
            .chain([
                4.93, 0.94, 0.86, 0.01, // difficulty
                1.49, 0.14, 0.94, // success
                2.18, 0.05, 0.34, 1.26, // failure
                0.29, 2.61, // hard penalty, easy bonus
            ])
            .collect();

        Self {
            w: Param::from(Tensor::from_floats(Data::new(
                initial_params,
                Shape { dims: [17] },
            ))),
            config,
        }
    }

    pub fn power_forgetting_curve(&self, t: Tensor<B, 1>, s: Tensor<B, 1>) -> Tensor<B, 1> {
        (t / (s * 9) + 1).powf(-1.0)
    }

    fn stability_after_success(
        &self,
        last_s: Tensor<B, 1>,
        new_d: Tensor<B, 1>,
        r: Tensor<B, 1>,
        rating: Tensor<B, 1>,
    ) -> Tensor<B, 1> {
        let batch_size = rating.dims()[0];
        let hard_penalty =
            Tensor::ones([batch_size]).mask_where(rating.clone().equal_elem(2), self.w.get(15));
        let easy_bonus =
            Tensor::ones([batch_size]).mask_where(rating.equal_elem(4), self.w.get(16));

        last_s.clone()
            * (self.w.get(8).exp()
                * (-new_d + 11)
                * (last_s.pow(-self.w.get(9)))
                * (((-r + 1) * self.w.get(10)).exp() - 1)
                * hard_penalty
                * easy_bonus
                + 1)
    }

    fn stability_after_failure(
        &self,
        last_s: Tensor<B, 1>,
        new_d: Tensor<B, 1>,
        r: Tensor<B, 1>,
    ) -> Tensor<B, 1> {
        self.w.get(11)
            * new_d.pow(-self.w.get(12))
            * ((last_s + 1).pow(self.w.get(13)) - 1)
            * ((-r + 1) * self.w.get(14)).exp()
    }

    fn mean_reversion(&self, new_d: Tensor<B, 1>) -> Tensor<B, 1> {
        self.w.get(7) * (self.w.get(4) - new_d.clone()) + new_d
    }

    fn init_stability(&self, rating: Tensor<B, 1>) -> Tensor<B, 1> {
        self.w.val().select(0, rating.clone().int() - 1)
    }

    fn init_difficulty(&self, rating: Tensor<B, 1>) -> Tensor<B, 1> {
        self.w.get(4) - self.w.get(5) * (rating - 3)
    }

    fn next_difficulty(&self, difficulty: Tensor<B, 1>, rating: Tensor<B, 1>) -> Tensor<B, 1> {
        difficulty - self.w.get(6) * (rating - 3)
    }

    pub(crate) fn step(
        &self,
        delta_t: Tensor<B, 1>,
        rating: Tensor<B, 1>,
        state: Option<MemoryStateTensors<B>>,
    ) -> MemoryStateTensors<B> {
        let (new_s, new_d) = if let Some(state) = state {
            let retention = self.power_forgetting_curve(delta_t, state.stability.clone());
            let mut new_difficulty = self.next_difficulty(state.difficulty.clone(), rating.clone());
            new_difficulty = self.mean_reversion(new_difficulty).clamp(1.0, 10.0);
            let stability_after_success = self.stability_after_success(
                state.stability.clone(),
                new_difficulty.clone(),
                retention.clone(),
                rating.clone(),
            );
            let stability_after_failure: Tensor<B, 1> = self.stability_after_failure(
                state.stability.clone(),
                new_difficulty.clone(),
                retention.clone(),
            );
            let mut new_stability = stability_after_success
                .mask_where(rating.clone().equal_elem(1), stability_after_failure);
            // mask padding zeros for rating
            new_stability = new_stability.mask_where(rating.clone().equal_elem(0), state.stability);
            new_difficulty = new_difficulty.mask_where(rating.equal_elem(0), state.difficulty);
            (new_stability, new_difficulty)
        } else {
            (
                self.init_stability(rating.clone()),
                self.init_difficulty(rating).clamp(1.0, 10.0),
            )
        };
        MemoryStateTensors {
            stability: new_s.clamp(0.1, 36500.0),
            difficulty: new_d,
        }
    }

    pub(crate) fn forward(
        &self,
        delta_ts: Tensor<B, 2>,
        ratings: Tensor<B, 2, Float>,
    ) -> MemoryStateTensors<B> {
        let [seq_len, _batch_size] = delta_ts.dims();
        let mut state = None;
        for i in 0..seq_len {
            let delta_t = delta_ts.get(i).squeeze(0);
            // [batch_size]
            let rating = ratings.get(i).squeeze(0);
            // [batch_size]
            state = Some(self.step(delta_t, rating, state));
        }
        state.unwrap()
    }
}

#[derive(Debug, Clone)]
pub(crate) struct MemoryStateTensors<B: Backend> {
    pub stability: Tensor<B, 1>,
    pub difficulty: Tensor<B, 1>,
}

#[derive(Config, Module, Debug, Default)]
pub struct ModelConfig {
    #[config(default = false)]
    pub freeze_stability: bool,
    pub initial_stability: Option<[f32; 4]>,
}

impl ModelConfig {
    pub fn init<B: Backend>(&self) -> Model<B> {
        Model::new(self.clone())
    }
}

/// This is the main structure provided by this crate. It can be used
/// for both weight training, and for reviews.
#[derive(Debug, Clone)]
pub struct FSRS<B: Backend = NdArrayBackend> {
    model: Option<Model<B>>,
    device: B::Device,
}

impl FSRS<NdArrayBackend> {
    /// - Weights must be provided before running commands that need them.
    /// - Weights may be an empty slice to use the default values instead.
    pub fn new(weights: Option<&Weights>) -> Result<Self> {
        Self::new_with_backend(weights, NdArrayDevice::Cpu)
    }
}

impl<B: Backend> FSRS<B> {
    pub fn new_with_backend<B2: Backend>(
        mut weights: Option<&Weights>,
        device: B2::Device,
    ) -> Result<FSRS<B2>> {
        if let Some(weights) = &mut weights {
            if weights.is_empty() {
                *weights = DEFAULT_WEIGHTS
            } else if weights.len() != 17 {
                return Err(FSRSError::InvalidWeights);
            }
        }
        Ok(FSRS {
            model: weights.map(weights_to_model),
            device,
        })
    }

    pub(crate) fn model(&self) -> &Model<B> {
        self.model
            .as_ref()
            .expect("command requires weights to be set on creation")
    }

    pub(crate) fn device(&self) -> B::Device {
        self.device.clone()
    }
}

pub(crate) fn weights_to_model<B: Backend>(weights: &Weights) -> Model<B> {
    let config = ModelConfig::default();
    let mut model = Model::<B>::new(config);
    model.w = Param::from(Tensor::from_floats(Data::new(
        clip_weights(weights),
        Shape { dims: [17] },
    )));
    model
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_helpers::{Model, Tensor};
    use burn::tensor::Data;

    #[test]
    fn w() {
        let model = Model::new(ModelConfig::default());
        assert_eq!(
            model.w.val().to_data(),
            Data::from([
                0.4, 0.6, 2.4, 5.8, 4.93, 0.94, 0.86, 0.01, 1.49, 0.14, 0.94, 2.18, 0.05, 0.34,
                1.26, 0.29, 2.61
            ])
        )
    }

    #[test]
    fn power_forgetting_curve() {
        let model = Model::new(ModelConfig::default());
        let delta_t = Tensor::from_floats([0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
        let stability = Tensor::from_floats([1.0, 2.0, 3.0, 4.0, 4.0, 2.0]);
        let retention = model.power_forgetting_curve(delta_t, stability);
        assert_eq!(
            retention.to_data(),
            Data::from([1.0, 0.9473684, 0.9310345, 0.92307687, 0.9, 0.7826087])
        )
    }

    #[test]
    fn init_stability() {
        let model = Model::new(ModelConfig::default());
        let rating = Tensor::from_floats([1.0, 2.0, 3.0, 4.0, 1.0, 2.0]);
        let stability = model.init_stability(rating);
        assert_eq!(
            stability.to_data(),
            Data::from([0.4, 0.6, 2.4, 5.8, 0.4, 0.6])
        )
    }

    #[test]
    fn init_difficulty() {
        let model = Model::new(ModelConfig::default());
        let rating = Tensor::from_floats([1.0, 2.0, 3.0, 4.0, 1.0, 2.0]);
        let difficulty = model.init_difficulty(rating);
        assert_eq!(
            difficulty.to_data(),
            Data::from([6.81, 5.87, 4.93, 3.9899998, 6.81, 5.87])
        )
    }

    #[test]
    fn forward() {
        let model = Model::new(ModelConfig::default());
        let delta_ts = Tensor::from_floats([
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0, 2.0, 2.0],
        ]);
        let ratings = Tensor::from_floats([
            [1.0, 2.0, 3.0, 4.0, 1.0, 2.0],
            [1.0, 2.0, 3.0, 4.0, 1.0, 2.0],
        ]);
        let state = model.forward(delta_ts, ratings);
        dbg!(&state);
    }

    #[test]
    fn next_difficulty() {
        let model = Model::new(ModelConfig::default());
        let difficulty = Tensor::from_floats([5.0; 4]);
        let rating = Tensor::from_floats([1.0, 2.0, 3.0, 4.0]);
        let next_difficulty = model.next_difficulty(difficulty, rating);
        next_difficulty.clone().backward();
        assert_eq!(
            next_difficulty.to_data(),
            Data::from([6.7200003, 5.86, 5.0, 4.14])
        );
        let next_difficulty = model.mean_reversion(next_difficulty);
        next_difficulty.clone().backward();
        assert_eq!(
            next_difficulty.to_data(),
            Data::from([6.7021003, 5.8507, 4.9993, 4.1478996])
        )
    }

    #[test]
    fn next_stability() {
        let model = Model::new(ModelConfig::default());
        let stability = Tensor::from_floats([5.0; 4]);
        let difficulty = Tensor::from_floats([1.0, 2.0, 3.0, 4.0]);
        let retention = Tensor::from_floats([0.9, 0.8, 0.7, 0.6]);
        let rating = Tensor::from_floats([1.0, 2.0, 3.0, 4.0]);
        let s_recall = model.stability_after_success(
            stability.clone(),
            difficulty.clone(),
            retention.clone(),
            rating.clone(),
        );
        s_recall.clone().backward();
        assert_eq!(
            s_recall.to_data(),
            Data::from([22.454704, 14.560361, 51.15574, 152.6869])
        );
        let s_forget = model.stability_after_failure(stability, difficulty, retention);
        s_forget.clone().backward();
        assert_eq!(
            s_forget.to_data(),
            Data::from([2.074517, 2.2729328, 2.526406, 2.8247323])
        );
        let next_stability = s_recall.mask_where(rating.clone().equal_elem(1), s_forget);
        next_stability.clone().backward();
        assert_eq!(
            next_stability.to_data(),
            Data::from([2.074517, 14.560361, 51.15574, 152.6869])
        )
    }

    #[test]
    fn fsrs() {
        assert!(FSRS::new(Some(&[])).is_ok());
        assert!(FSRS::new(Some(&[1.])).is_err());
        assert!(FSRS::new(Some(DEFAULT_WEIGHTS)).is_ok());
    }
}

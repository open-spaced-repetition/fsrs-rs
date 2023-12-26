use crate::error::{FSRSError, Result};
use crate::inference::{Weights, DECAY, FACTOR};
use crate::weight_clipper::clip_weights;
use crate::DEFAULT_WEIGHTS;
use burn::backend::ndarray::NdArrayDevice;
use burn::backend::NdArray;
use burn::{
    config::Config,
    module::{Module, Param},
    tensor::{backend::Backend, Data, Shape, Tensor},
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
    fn get(&self, n: usize) -> Self {
        self.clone().slice([n..(n + 1)])
    }
}

trait Pow<B: Backend, const N: usize> {
    // https://github.com/burn-rs/burn/issues/590 , after that finished, just remove this trait and below impl, all will ok.
    fn pow(&self, other: Tensor<B, N>) -> Tensor<B, N>;
}

impl<B: Backend, const N: usize> Pow<B, N> for Tensor<B, N> {
    fn pow(&self, other: Self) -> Self {
        // a ^ b => exp(ln(a^b)) => exp(b ln (a))
        (self.clone().log() * other).exp()
    }
}

impl<B: Backend> Model<B> {
    #[allow(clippy::new_without_default)]
    pub fn new(config: ModelConfig) -> Self {
        let initial_params = config
            .initial_stability
            .unwrap_or_else(|| DEFAULT_WEIGHTS[0..4].try_into().unwrap())
            .into_iter()
            .chain(DEFAULT_WEIGHTS[4..].iter().copied())
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
        (t / s * FACTOR + 1).powf(DECAY as f32)
    }

    fn stability_after_success(
        &self,
        last_s: Tensor<B, 1>,
        last_d: Tensor<B, 1>,
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
                * (-last_d + 11)
                * (last_s.pow(-self.w.get(9)))
                * (((-r + 1) * self.w.get(10)).exp() - 1)
                * hard_penalty
                * easy_bonus
                + 1)
    }

    fn stability_after_failure(
        &self,
        last_s: Tensor<B, 1>,
        last_d: Tensor<B, 1>,
        r: Tensor<B, 1>,
    ) -> Tensor<B, 1> {
        self.w.get(11)
            * last_d.pow(-self.w.get(12))
            * ((last_s + 1).pow(self.w.get(13)) - 1)
            * ((-r + 1) * self.w.get(14)).exp()
    }

    fn mean_reversion(&self, new_d: Tensor<B, 1>) -> Tensor<B, 1> {
        self.w.get(7) * (self.w.get(4) - new_d.clone()) + new_d
    }

    pub(crate) fn init_stability(&self, rating: Tensor<B, 1>) -> Tensor<B, 1> {
        self.w.val().select(0, rating.int() - 1)
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
            let stability_after_success = self.stability_after_success(
                state.stability.clone(),
                state.difficulty.clone(),
                retention.clone(),
                rating.clone(),
            );
            let stability_after_failure = self.stability_after_failure(
                state.stability.clone(),
                state.difficulty.clone(),
                retention,
            );
            let mut new_stability = stability_after_success
                .mask_where(rating.clone().equal_elem(1), stability_after_failure);

            let mut new_difficulty = self.next_difficulty(state.difficulty.clone(), rating.clone());
            new_difficulty = self.mean_reversion(new_difficulty).clamp(1.0, 10.0);
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

    /// If [starting_state] is provided, it will be used instead of the default initial stability/
    /// difficulty.
    pub(crate) fn forward(
        &self,
        delta_ts: Tensor<B, 2>,
        ratings: Tensor<B, 2>,
        starting_state: Option<MemoryStateTensors<B>>,
    ) -> MemoryStateTensors<B> {
        let [seq_len, _batch_size] = delta_ts.dims();
        let mut state = starting_state;
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
pub struct FSRS<B: Backend = NdArray> {
    model: Option<Model<B>>,
    device: B::Device,
}

impl FSRS<NdArray> {
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
                *weights = DEFAULT_WEIGHTS.as_slice()
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
    let mut model = Model::new(config);
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
        assert_eq!(model.w.val().to_data(), Data::from(DEFAULT_WEIGHTS))
    }

    #[test]
    fn power_forgetting_curve() {
        let model = Model::new(ModelConfig::default());
        let delta_t = Tensor::from_floats([0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
        let stability = Tensor::from_floats([1.0, 2.0, 3.0, 4.0, 4.0, 2.0]);
        let retention = model.power_forgetting_curve(delta_t, stability);
        assert_eq!(
            retention.to_data(),
            Data::from([1.0, 0.946059, 0.9299294, 0.9221679, 0.9, 0.79394597])
        )
    }

    #[test]
    fn init_stability() {
        let model = Model::new(ModelConfig::default());
        let rating = Tensor::from_floats([1.0, 2.0, 3.0, 4.0, 1.0, 2.0]);
        let stability = model.init_stability(rating);
        assert_eq!(
            stability.to_data(),
            Data::from([0.5614, 1.2546, 3.5878, 7.9731, 0.5614, 1.2546])
        )
    }

    #[test]
    fn init_difficulty() {
        let model = Model::new(ModelConfig::default());
        let rating = Tensor::from_floats([1.0, 2.0, 3.0, 4.0, 1.0, 2.0]);
        let difficulty = model.init_difficulty(rating);
        assert_eq!(
            difficulty.to_data(),
            Data::from([7.3649, 6.2346, 5.1043, 3.974, 7.3649, 6.2346])
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
        let state = model.forward(delta_ts, ratings, None);
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
            Data::from([6.646, 5.823, 5.0, 4.177])
        );
        let next_difficulty = model.mean_reversion(next_difficulty);
        next_difficulty.clone().backward();
        assert_eq!(
            next_difficulty.to_data(),
            Data::from([6.574311, 5.7895803, 5.00485, 4.2201195])
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
            Data::from([26.678038, 13.996968, 62.718544, 202.76956])
        );
        let s_forget = model.stability_after_failure(stability, difficulty, retention);
        s_forget.clone().backward();
        assert_eq!(
            s_forget.to_data(),
            Data::from([1.8932177, 2.0453987, 2.2637987, 2.5304008])
        );
        let next_stability = s_recall.mask_where(rating.clone().equal_elem(1), s_forget);
        next_stability.clone().backward();
        assert_eq!(
            next_stability.to_data(),
            Data::from([1.8932177, 13.996968, 62.718544, 202.76956])
        )
    }

    #[test]
    fn fsrs() {
        assert!(FSRS::new(Some(&[])).is_ok());
        assert!(FSRS::new(Some(&[1.])).is_err());
        assert!(FSRS::new(Some(DEFAULT_WEIGHTS.as_slice())).is_ok());
    }
}

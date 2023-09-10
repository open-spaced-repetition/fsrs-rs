use crate::inference::Weights;
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

impl<B: Backend<FloatElem = f32>> Model<B> {
    #[allow(clippy::new_without_default)]
    pub fn new(config: ModelConfig) -> Self {
        let initial_stability = config.initial_stability.unwrap_or([0.4, 0.6, 2.4, 5.8]);
        let mut initial_params = Vec::new();
        initial_params.extend_from_slice(&initial_stability);
        initial_params.extend_from_slice(&[
            4.93, 0.94, 0.86, 0.01, // difficulty
            1.49, 0.14, 0.94, // success
            2.18, 0.05, 0.34, 1.26, // failure
            0.29, 2.61, // hard penalty, easy bonus
        ]);

        Self {
            w: Param::from(Tensor::from_floats(Data::new(
                initial_params,
                Shape { dims: [17] },
            ))),
            config,
        }
    }

    fn w(&self) -> Tensor<B, 2> {
        self.w.val().unsqueeze().transpose()
    }

    pub fn power_forgetting_curve(&self, t: Tensor<B, 2>, s: Tensor<B, 2>) -> Tensor<B, 2> {
        (t / (s * 9) + 1).powf(-1.0)
    }

    fn stability_after_success(
        &self,
        last_s: Tensor<B, 2>,
        new_d: Tensor<B, 2>,
        r: Tensor<B, 2>,
        rating: Tensor<B, 2>,
    ) -> Tensor<B, 2> {
        let batch_size = rating.dims()[0];
        let hard_penalty = Tensor::ones([batch_size, 1]).mask_where(
            rating.clone().equal_elem(2),
            self.w().slice([15..16]).unsqueeze(),
        );
        let easy_bonus = Tensor::ones([batch_size, 1])
            .mask_where(rating.equal_elem(4), self.w().slice([16..17]).unsqueeze());

        last_s.clone()
            * (self.w().slice([8..9]).exp()
                * (-new_d + 11)
                * (-self.w().slice([9..10]) * last_s.log()).exp()
                * (((-r + 1) * self.w().slice([10..11])).exp() - 1)
                * hard_penalty
                * easy_bonus
                + 1)
    }

    fn stability_after_failure(
        &self,
        last_s: Tensor<B, 2>,
        new_d: Tensor<B, 2>,
        r: Tensor<B, 2>,
    ) -> Tensor<B, 2> {
        self.w().slice([11..12])
            * (-self.w().slice([12..13]) * new_d.log()).exp()
            * ((self.w().slice([13..14]) * (last_s + 1).log()).exp() - 1)
            * ((-r + 1) * self.w().slice([14..15])).exp()
    }

    fn mean_reversion(&self, new_d: Tensor<B, 2>) -> Tensor<B, 2> {
        self.w().slice([7..8]) * (self.w().slice([4..5]) - new_d.clone()) + new_d
    }

    fn init_stability(&self, rating: Tensor<B, 2>) -> Tensor<B, 2> {
        self.w()
            .squeeze::<1>(1)
            .select(0, rating.clone().squeeze::<1>(1).int() - 1)
            .unsqueeze()
            .transpose()
    }

    fn init_difficulty(&self, rating: Tensor<B, 2>) -> Tensor<B, 2> {
        self.w().slice([4..5]) - self.w().slice([5..6]) * (rating - 3)
    }

    fn next_difficulty(&self, difficulty: Tensor<B, 2>, rating: Tensor<B, 2>) -> Tensor<B, 2> {
        difficulty - self.w().slice([6..7]) * (rating - 3)
    }

    pub(crate) fn step(
        &self,
        delta_t: Tensor<B, 2>,
        rating: Tensor<B, 2>,
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
            let stability_after_failure: Tensor<B, 2> = self.stability_after_failure(
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
            let delta_t = delta_ts.clone().slice([i..i + 1]).transpose();
            // [batch_size, 1]
            let rating = ratings.clone().slice([i..i + 1]).transpose();
            // [batch_size, 1]
            state = Some(self.step(delta_t, rating, state));
        }
        state.unwrap()
    }
}

#[derive(Debug, Clone)]
pub(crate) struct MemoryStateTensors<B: Backend> {
    pub stability: Tensor<B, 2>,
    pub difficulty: Tensor<B, 2>,
}

#[derive(Config, Module, Debug, Default)]
pub struct ModelConfig {
    #[config(default = false)]
    pub freeze_stability: bool,
    pub initial_stability: Option<[f32; 4]>,
}

impl ModelConfig {
    pub fn init<B: Backend<FloatElem = f32>>(&self) -> Model<B> {
        Model::new(self.clone())
    }
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
            model.w().to_data(),
            Data::from([
                [0.4],
                [0.6],
                [2.4],
                [5.8],
                [4.93],
                [0.94],
                [0.86],
                [0.01],
                [1.49],
                [0.14],
                [0.94],
                [2.18],
                [0.05],
                [0.34],
                [1.26],
                [0.29],
                [2.61]
            ])
        )
    }

    #[test]
    fn power_forgetting_curve() {
        let model = Model::new(ModelConfig::default());
        let delta_t = Tensor::<2>::from_floats([[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]]);
        let stability = Tensor::<2>::from_floats([[1.0], [2.0], [3.0], [4.0], [4.0], [2.0]]);
        let retention = model.power_forgetting_curve(delta_t, stability);
        assert_eq!(
            retention.to_data(),
            Data::from([
                [1.0],
                [0.9473684],
                [0.9310345],
                [0.92307687],
                [0.9],
                [0.7826087]
            ])
        )
    }

    #[test]
    fn init_stability() {
        let model = Model::new(ModelConfig::default());
        let rating = Tensor::<2>::from_floats([[1.0], [2.0], [3.0], [4.0], [1.0], [2.0]]);
        let stability = model.init_stability(rating);
        assert_eq!(
            stability.to_data(),
            Data::from([[0.4], [0.6], [2.4], [5.8], [0.4], [0.6]])
        )
    }

    #[test]
    fn init_difficulty() {
        let model = Model::new(ModelConfig::default());
        let rating = Tensor::<2>::from_floats([[1.0], [2.0], [3.0], [4.0], [1.0], [2.0]]);
        let difficulty = model.init_difficulty(rating);
        assert_eq!(
            difficulty.to_data(),
            Data::from([[6.81], [5.87], [4.93], [3.9899998], [6.81], [5.87]])
        )
    }

    #[test]
    fn forward() {
        let model = Model::new(ModelConfig::default());
        let delta_ts = Tensor::<2>::from_floats([
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0, 2.0, 2.0],
        ]);
        let ratings = Tensor::<2>::from_floats([
            [1.0, 2.0, 3.0, 4.0, 1.0, 2.0],
            [1.0, 2.0, 3.0, 4.0, 1.0, 2.0],
        ]);
        let state = model.forward(delta_ts, ratings);
        dbg!(&state);
    }

    #[test]
    fn next_difficulty() {
        let model = Model::new(ModelConfig::default());
        let difficulty = Tensor::<2>::from_floats([[5.0], [5.0], [5.0], [5.0]]);
        let rating = Tensor::<2>::from_floats([[1.0], [2.0], [3.0], [4.0]]);
        let next_difficulty = model.next_difficulty(difficulty, rating);
        next_difficulty.clone().backward();
        assert_eq!(
            next_difficulty.to_data(),
            Data::from([[6.7200003], [5.86], [5.0], [4.14]])
        );
        let next_difficulty = model.mean_reversion(next_difficulty);
        next_difficulty.clone().backward();
        assert_eq!(
            next_difficulty.to_data(),
            Data::from([[6.7021003], [5.8507], [4.9993], [4.1478996]])
        )
    }

    #[test]
    fn next_stability() {
        let model = Model::new(ModelConfig::default());
        let stability = Tensor::<2>::from_floats([[5.0], [5.0], [5.0], [5.0]]);
        let difficulty = Tensor::<2>::from_floats([[1.0], [2.0], [3.0], [4.0]]);
        let retention = Tensor::<2>::from_floats([[0.9], [0.8], [0.7], [0.6]]);
        let rating = Tensor::<2>::from_floats([[1.0], [2.0], [3.0], [4.0]]);
        let s_recall = model.stability_after_success(
            stability.clone(),
            difficulty.clone(),
            retention.clone(),
            rating.clone(),
        );
        s_recall.clone().backward();
        assert_eq!(
            s_recall.to_data(),
            Data::from([[22.454704], [14.560361], [51.15574], [152.6869]])
        );
        let s_forget = model.stability_after_failure(stability, difficulty, retention);
        s_forget.clone().backward();
        assert_eq!(
            s_forget.to_data(),
            Data::from([[2.074517], [2.2729328], [2.526406], [2.8247323]])
        );
        let next_stability = s_recall.mask_where(rating.clone().equal_elem(1), s_forget);
        next_stability.clone().backward();
        assert_eq!(
            next_stability.to_data(),
            Data::from([[2.074517], [14.560361], [51.15574], [152.6869]])
        )
    }
}

/// This wraps our internal model and provides our public API.
pub struct Fsrs<B: Backend<FloatElem = f32> = NdArrayBackend> {
    model: Option<Model<B>>,
    device: B::Device,
}

impl Fsrs<NdArrayBackend> {
    pub fn new(weights: Option<&Weights>) -> Self {
        Self {
            model: weights.map(weights_to_model),
            device: NdArrayDevice::Cpu,
        }
    }
}

impl<B: Backend<FloatElem = f32>> Fsrs<B> {
    pub(crate) fn model(&self) -> &Model<B> {
        self.model
            .as_ref()
            .expect("command requires weights to be set on creation")
    }

    pub(crate) fn device(&self) -> B::Device {
        self.device.clone()
    }
}

pub(crate) fn weights_to_model(weights: &Weights) -> Model<NdArrayBackend<f32>> {
    type Backend = NdArrayBackend<f32>;
    let config = ModelConfig::default();
    let mut model = Model::<Backend>::new(config);
    model.w = Param::from(Tensor::from_floats(Data::new(
        weights.to_vec(),
        Shape { dims: [17] },
    )));
    model
}

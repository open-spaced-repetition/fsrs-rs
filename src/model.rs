use crate::DEFAULT_PARAMETERS;
use crate::error::{FSRSError, Result};
use crate::inference::{FSRS5_DEFAULT_DECAY, Parameters};
use crate::parameter_clipper::clip_parameters;
use crate::simulation::{D_MAX, D_MIN, S_MAX, S_MIN};
use burn::backend::NdArray;
use burn::backend::ndarray::NdArrayDevice;
use burn::{
    config::Config,
    module::{Module, Param},
    tensor::{Shape, Tensor, TensorData, backend::Backend},
};

#[derive(Module, Debug)]
pub struct Model<B: Backend> {
    pub w: Param<Tensor<B, 1>>,
}

pub(crate) trait Get<B: Backend, const N: usize> {
    fn get(&self, n: usize) -> Tensor<B, N>;
}

impl<B: Backend, const N: usize> Get<B, N> for Tensor<B, N> {
    fn get(&self, n: usize) -> Self {
        self.clone().slice([n..(n + 1)])
    }
}

impl<B: Backend> Model<B> {
    #[allow(clippy::new_without_default)]
    pub fn new(config: ModelConfig) -> Self {
        let mut initial_params: Vec<f32> = config
            .initial_stability
            .unwrap_or_else(|| DEFAULT_PARAMETERS[0..4].try_into().unwrap())
            .into_iter()
            .chain(DEFAULT_PARAMETERS[4..].iter().copied())
            .collect();
        if config.freeze_short_term_stability {
            initial_params[17] = 0.0;
            initial_params[18] = 0.0;
            initial_params[19] = 0.0;
        }

        Self {
            w: Param::from_tensor(Tensor::from_floats(
                TensorData::new(initial_params, Shape { dims: vec![21] }),
                &B::Device::default(),
            )),
        }
    }

    pub fn power_forgetting_curve(&self, t: Tensor<B, 1>, s: Tensor<B, 1>) -> Tensor<B, 1> {
        let decay = -self.w.get(20);
        let factor = decay.clone().powi_scalar(-1).mul_scalar(0.9f32.ln()).exp() - 1.0;
        (t / s * factor + 1.0).powf(decay)
    }

    pub fn next_interval(
        &self,
        stability: Tensor<B, 1>,
        desired_retention: Tensor<B, 1>,
    ) -> Tensor<B, 1> {
        let decay = -self.w.get(20);
        let factor = decay.clone().powi_scalar(-1).mul_scalar(0.9f32.ln()).exp() - 1.0;
        stability / factor * (desired_retention.powf(decay.powi_scalar(-1)) - 1.0)
    }

    fn stability_after_success(
        &self,
        last_s: Tensor<B, 1>,
        last_d: Tensor<B, 1>,
        r: Tensor<B, 1>,
        rating: Tensor<B, 1>,
    ) -> Tensor<B, 1> {
        let batch_size = rating.dims()[0];
        let hard_penalty = Tensor::ones([batch_size], &B::Device::default())
            .mask_where(rating.clone().equal_elem(2), self.w.get(15));
        let easy_bonus = Tensor::ones([batch_size], &B::Device::default())
            .mask_where(rating.equal_elem(4), self.w.get(16));

        last_s.clone()
            * (self.w.get(8).exp()
                * (-last_d + 11)
                * (last_s.powf(-self.w.get(9)))
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
        let new_s = self.w.get(11)
            * last_d.powf(-self.w.get(12))
            * ((last_s.clone() + 1).powf(self.w.get(13)) - 1)
            * ((-r + 1) * self.w.get(14)).exp();
        let new_s_min = last_s / (self.w.get(17) * self.w.get(18)).exp();
        new_s
            .clone()
            .mask_where(new_s_min.clone().lower(new_s), new_s_min)
    }

    fn stability_short_term(&self, last_s: Tensor<B, 1>, rating: Tensor<B, 1>) -> Tensor<B, 1> {
        let sinc = (self.w.get(17) * (rating.clone() - 3 + self.w.get(18))).exp()
            * last_s.clone().powf(-self.w.get(19));

        last_s
            * sinc
                .clone()
                .mask_where(rating.greater_equal_elem(3), sinc.clamp_min(1.0))
    }

    fn mean_reversion(&self, new_d: Tensor<B, 1>) -> Tensor<B, 1> {
        let rating = Tensor::from_floats([4.0], &B::Device::default());
        self.w.get(7) * (self.init_difficulty(rating) - new_d.clone()) + new_d
    }

    pub(crate) fn init_stability(&self, rating: Tensor<B, 1>) -> Tensor<B, 1> {
        self.w.val().select(0, rating.int() - 1)
    }

    fn init_difficulty(&self, rating: Tensor<B, 1>) -> Tensor<B, 1> {
        self.w.get(4) - (self.w.get(5) * (rating - 1)).exp() + 1
    }

    fn linear_damping(&self, delta_d: Tensor<B, 1>, old_d: Tensor<B, 1>) -> Tensor<B, 1> {
        old_d.neg().add_scalar(10.0) * delta_d.div_scalar(9.0)
    }

    fn next_difficulty(&self, difficulty: Tensor<B, 1>, rating: Tensor<B, 1>) -> Tensor<B, 1> {
        let delta_d = -self.w.get(6) * (rating - 3);
        difficulty.clone() + self.linear_damping(delta_d, difficulty)
    }

    pub(crate) fn step(
        &self,
        delta_t: Tensor<B, 1>,
        rating: Tensor<B, 1>,
        state: MemoryStateTensors<B>,
        nth: usize,
    ) -> MemoryStateTensors<B> {
        let last_s = state.stability.clone().clamp(S_MIN, S_MAX);
        let last_d = state.difficulty.clone().clamp(D_MIN, D_MAX);

        let retrievability = self.power_forgetting_curve(delta_t.clone(), last_s.clone());
        let stability_after_success = self.stability_after_success(
            last_s.clone(),
            last_d.clone(),
            retrievability.clone(),
            rating.clone(),
        );
        let stability_after_failure =
            self.stability_after_failure(last_s.clone(), last_d.clone(), retrievability);
        let stability_short_term = self.stability_short_term(last_s.clone(), rating.clone());
        let mut new_s = stability_after_success
            .mask_where(rating.clone().equal_elem(1), stability_after_failure);
        new_s = new_s.mask_where(delta_t.equal_elem(0), stability_short_term);

        let mut new_d = self.next_difficulty(last_d.clone(), rating.clone());
        new_d = self.mean_reversion(new_d).clamp(D_MIN, D_MAX);

        if nth == 0 {
            // Check if state.stability is all zeros
            let is_initial = state.stability.equal_elem(0.0);
            // If initial, use init_stability/init_difficulty, else use normal update
            let init_s = self.init_stability(rating.clone().clamp(1, 4));
            let init_d = self
                .init_difficulty(rating.clone().clamp(1, 4))
                .clamp(D_MIN, D_MAX);
            // If state.stability == 0, use init values, else use calculated
            new_s = new_s.mask_where(is_initial.clone(), init_s);
            new_d = new_d.mask_where(is_initial, init_d);
        }

        // mask padding zeros for rating
        new_s = new_s.mask_where(rating.clone().equal_elem(0), last_s);
        new_d = new_d.mask_where(rating.equal_elem(0), last_d);
        MemoryStateTensors {
            stability: new_s.clamp(S_MIN, S_MAX),
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
        let [seq_len, batch_size] = delta_ts.dims();
        let mut state = if let Some(state) = starting_state {
            state
        } else {
            MemoryStateTensors::zeros(batch_size)
        };
        for i in 0..seq_len {
            let delta_t = delta_ts.get(i).squeeze(0);
            // [batch_size]
            let rating = ratings.get(i).squeeze(0);
            // [batch_size]
            state = self.step(delta_t, rating, state, i);
        }
        state
    }
}

#[derive(Debug, Clone)]
pub(crate) struct MemoryStateTensors<B: Backend> {
    pub stability: Tensor<B, 1>,
    pub difficulty: Tensor<B, 1>,
}

impl<B: Backend> MemoryStateTensors<B> {
    pub(crate) fn zeros(batch_size: usize) -> MemoryStateTensors<B> {
        MemoryStateTensors {
            stability: Tensor::zeros([batch_size], &B::Device::default()),
            difficulty: Tensor::zeros([batch_size], &B::Device::default()),
        }
    }
}

#[derive(Config, Debug, Default)]
pub struct ModelConfig {
    #[config(default = false)]
    pub freeze_initial_stability: bool,
    pub initial_stability: Option<[f32; 4]>,
    #[config(default = false)]
    pub freeze_short_term_stability: bool,
    #[config(default = 1)]
    pub num_relearning_steps: usize,
}

impl ModelConfig {
    pub fn init<B: Backend>(&self) -> Model<B> {
        Model::new(self.clone())
    }
}

/// This is the main structure provided by this crate. It can be used
/// for both parameter training, and for reviews.
#[derive(Debug, Clone)]
pub struct FSRS<B: Backend = NdArray> {
    model: Option<Model<B>>,
    device: B::Device,
}

impl FSRS<NdArray> {
    /// - Parameters must be provided before running commands that need them.
    /// - Parameters may be an empty slice to use the default values instead.
    pub fn new(parameters: Option<&Parameters>) -> Result<Self> {
        Self::new_with_backend(parameters, NdArrayDevice::Cpu)
    }
}

impl<B: Backend> FSRS<B> {
    pub fn new_with_backend<B2: Backend>(
        parameters: Option<&Parameters>,
        device: B2::Device,
    ) -> Result<FSRS<B2>> {
        let model = match parameters {
            Some(params) => {
                let parameters = check_and_fill_parameters(params)?;
                let model = parameters_to_model::<B2>(&parameters);
                Some(model)
            }
            None => None,
        };

        Ok(FSRS { model, device })
    }

    pub(crate) fn model(&self) -> &Model<B> {
        self.model
            .as_ref()
            .expect("command requires parameters to be set on creation")
    }

    pub(crate) fn device(&self) -> B::Device {
        self.device.clone()
    }
}

pub(crate) fn parameters_to_model<B: Backend>(parameters: &Parameters) -> Model<B> {
    let config = ModelConfig::default();
    let mut model = Model::new(config.clone());
    model.w = Param::from_tensor(Tensor::from_floats(
        TensorData::new(
            clip_parameters(parameters, config.num_relearning_steps, Default::default()),
            Shape { dims: vec![21] },
        ),
        &B::Device::default(),
    ));
    model
}

pub(crate) fn check_and_fill_parameters(parameters: &Parameters) -> Result<Vec<f32>, FSRSError> {
    let parameters = match parameters.len() {
        0 => DEFAULT_PARAMETERS.to_vec(),
        17 => {
            let mut parameters = parameters.to_vec();
            parameters[4] = parameters[5].mul_add(2.0, parameters[4]);
            parameters[5] = parameters[5].mul_add(3.0, 1.0).ln() / 3.0;
            parameters[6] += 0.5;
            parameters.extend_from_slice(&[0.0, 0.0, 0.0, FSRS5_DEFAULT_DECAY]);
            parameters
        }
        19 => {
            let mut parameters = parameters.to_vec();
            parameters.extend_from_slice(&[0.0, FSRS5_DEFAULT_DECAY]);
            parameters
        }
        21 => parameters.to_vec(),
        _ => return Err(FSRSError::InvalidParameters),
    };
    if parameters.iter().any(|&w| !w.is_finite()) {
        return Err(FSRSError::InvalidParameters);
    }
    Ok(parameters)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_helpers::TestHelper;
    use crate::test_helpers::{Model, Tensor};
    use burn::tensor::{TensorData, Tolerance};

    #[test]
    fn test_w() {
        let model = Model::new(ModelConfig::default());
        assert_eq!(
            model.w.val().to_data(),
            TensorData::from(DEFAULT_PARAMETERS)
        )
    }

    #[test]
    fn test_convert_parameters() {
        let fsrs4dot5_param = vec![
            0.4, 0.6, 2.4, 5.8, 4.93, 0.94, 0.86, 0.01, 1.49, 0.14, 0.94, 2.18, 0.05, 0.34, 1.26,
            0.29, 2.61,
        ];
        let fsrs5_param = check_and_fill_parameters(&fsrs4dot5_param).unwrap();
        assert_eq!(
            fsrs5_param,
            vec![
                0.4, 0.6, 2.4, 5.8, 6.81, 0.44675013, 1.36, 0.01, 1.49, 0.14, 0.94, 2.18, 0.05,
                0.34, 1.26, 0.29, 2.61, 0.0, 0.0, 0.0, 0.5
            ]
        )
    }

    #[test]
    fn test_power_forgetting_curve() {
        let device = NdArrayDevice::Cpu;
        let model = Model::new(ModelConfig::default());
        let delta_t = Tensor::from_floats([0.0, 1.0, 2.0, 3.0, 4.0, 5.0], &device);
        let stability = Tensor::from_floats([1.0, 2.0, 3.0, 4.0, 4.0, 2.0], &device);
        let retrievability = model.power_forgetting_curve(delta_t, stability);

        retrievability.to_data().assert_approx_eq::<f32>(
            &TensorData::from([1.0, 0.9403443, 0.9253786, 0.9185229, 0.9, 0.8261359]),
            Tolerance::absolute(1e-5),
        );
    }

    #[test]
    fn test_init_stability() {
        let device = NdArrayDevice::Cpu;
        let model = Model::new(ModelConfig::default());
        let rating = Tensor::from_floats([1.0, 2.0, 3.0, 4.0, 1.0, 2.0], &device);
        let stability = model.init_stability(rating);
        assert_eq!(
            stability.to_data(),
            TensorData::from([
                DEFAULT_PARAMETERS[0],
                DEFAULT_PARAMETERS[1],
                DEFAULT_PARAMETERS[2],
                DEFAULT_PARAMETERS[3],
                DEFAULT_PARAMETERS[0],
                DEFAULT_PARAMETERS[1]
            ])
        )
    }

    #[test]
    fn test_init_difficulty() {
        let device = NdArrayDevice::Cpu;
        let model = Model::new(ModelConfig::default());
        let rating = Tensor::from_floats([1.0, 2.0, 3.0, 4.0, 1.0, 2.0], &device);
        let difficulty = model.init_difficulty(rating);
        assert_eq!(
            difficulty.to_data(),
            TensorData::from([
                DEFAULT_PARAMETERS[4],
                DEFAULT_PARAMETERS[4] - DEFAULT_PARAMETERS[5].exp() + 1.0,
                DEFAULT_PARAMETERS[4] - (2.0 * DEFAULT_PARAMETERS[5]).exp() + 1.0,
                DEFAULT_PARAMETERS[4] - (3.0 * DEFAULT_PARAMETERS[5]).exp() + 1.0,
                DEFAULT_PARAMETERS[4],
                DEFAULT_PARAMETERS[4] - DEFAULT_PARAMETERS[5].exp() + 1.0,
            ])
        )
    }

    #[test]
    fn test_forward() {
        let device = NdArrayDevice::Cpu;
        let model = Model::new(ModelConfig::default());
        let delta_ts = Tensor::from_floats(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0, 1.0, 2.0, 2.0],
            ],
            &device,
        );
        let ratings = Tensor::from_floats(
            [
                [1.0, 2.0, 3.0, 4.0, 1.0, 2.0],
                [1.0, 2.0, 3.0, 4.0, 1.0, 2.0],
            ],
            &device,
        );
        let state = model.forward(delta_ts, ratings, None);
        let stability = state.stability.to_data();
        let difficulty = state.difficulty.to_data();

        stability.to_vec::<f32>().unwrap().assert_approx_eq([
            0.10088589,
            3.2494123,
            7.3153,
            18.014914,
            0.112798266,
            4.4694576,
        ]);

        difficulty
            .to_vec::<f32>()
            .unwrap()
            .assert_approx_eq([8.806304, 6.7404594, 2.1112142, 1.0, 8.806304, 6.7404594]);
    }

    #[test]
    fn test_next_difficulty() {
        let device = NdArrayDevice::Cpu;
        let model = Model::new(ModelConfig::default());
        let difficulty = Tensor::from_floats([5.0; 4], &device);
        let rating = Tensor::from_floats([1.0, 2.0, 3.0, 4.0], &device);
        let next_difficulty = model.next_difficulty(difficulty, rating);
        next_difficulty.clone().backward();

        next_difficulty
            .to_data()
            .to_vec::<f32>()
            .unwrap()
            .assert_approx_eq([8.354889, 6.6774445, 5.0, 3.3225555]);
        let next_difficulty = model.mean_reversion(next_difficulty);
        next_difficulty.clone().backward();

        next_difficulty
            .to_data()
            .to_vec::<f32>()
            .unwrap()
            .assert_approx_eq([8.341763, 6.6659956, 4.990228, 3.3144615]);
    }

    #[test]
    fn test_next_stability() {
        let device = NdArrayDevice::Cpu;
        let model = Model::new(ModelConfig::default());
        let stability = Tensor::from_floats([5.0; 4], &device);
        let difficulty = Tensor::from_floats([1.0, 2.0, 3.0, 4.0], &device);
        let retrievability = Tensor::from_floats([0.9, 0.8, 0.7, 0.6], &device);
        let rating = Tensor::from_floats([1.0, 2.0, 3.0, 4.0], &device);
        let s_recall = model.stability_after_success(
            stability.clone(),
            difficulty.clone(),
            retrievability.clone(),
            rating.clone(),
        );
        s_recall.clone().backward();

        s_recall
            .to_data()
            .to_vec::<f32>()
            .unwrap()
            .assert_approx_eq([25.602541, 28.226582, 58.656002, 127.226685]);
        let s_forget = model.stability_after_failure(stability.clone(), difficulty, retrievability);
        s_forget.clone().backward();

        s_forget
            .to_data()
            .to_vec::<f32>()
            .unwrap()
            .assert_approx_eq([1.0525396, 1.1894329, 1.3680838, 1.584989]);
        let next_stability = s_recall.mask_where(rating.clone().equal_elem(1), s_forget);
        next_stability.clone().backward();

        next_stability
            .to_data()
            .to_vec::<f32>()
            .unwrap()
            .assert_approx_eq([1.0525396, 28.226582, 58.656002, 127.226685]);
        let next_stability = model.stability_short_term(stability, rating);

        next_stability
            .to_data()
            .to_vec::<f32>()
            .unwrap()
            .assert_approx_eq([1.596818, 2.7470093, 5.0, 8.12961]);
    }

    #[test]
    fn test_fsrs() {
        assert!(FSRS::new(Some(&[])).is_ok());
        assert!(FSRS::new(Some(&[1.])).is_err());
        assert!(FSRS::new(Some(DEFAULT_PARAMETERS.as_slice())).is_ok());
        assert!(FSRS::new(Some(&DEFAULT_PARAMETERS[..17])).is_ok());
    }
}

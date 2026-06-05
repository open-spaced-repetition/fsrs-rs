use crate::DEFAULT_PARAMETERS;
use crate::dataset::FSRSReview;
use crate::error::{FSRSError, Result};
use crate::inference::{FSRS5_DEFAULT_DECAY, MemoryState, Parameters};
use crate::parameter_clipper::clip_parameters;
use crate::simulation::{D_MAX, D_MIN, S_MAX, S_MIN};

#[cfg(test)]
use burn::{
    module::{Module, Param},
    tensor::{Shape, Tensor, TensorData, backend::Backend},
};

#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub freeze_initial_stability: bool,
    pub initial_stability: Option<[f32; 4]>,
    pub freeze_short_term_stability: bool,
    pub num_relearning_steps: usize,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            freeze_initial_stability: false,
            initial_stability: None,
            freeze_short_term_stability: false,
            num_relearning_steps: 1,
        }
    }
}

impl ModelConfig {
    pub(crate) fn initial_parameters(&self) -> [f32; 21] {
        let mut parameters: [f32; 21] = self
            .initial_stability
            .unwrap_or_else(|| DEFAULT_PARAMETERS[0..4].try_into().unwrap())
            .into_iter()
            .chain(DEFAULT_PARAMETERS[4..].iter().copied())
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();
        if self.freeze_short_term_stability {
            parameters[17] = 0.0;
            parameters[18] = 0.0;
            parameters[19] = 0.0;
        }
        parameters
    }

    #[cfg(test)]
    pub fn init<B: Backend>(&self) -> Model<B> {
        Model::new(self.clone())
    }
}

/// This is the main structure provided by this crate. It can be used
/// for both parameter training, and for reviews.
#[derive(Debug, Clone)]
pub struct FSRS {
    parameters: [f32; 21],
}

impl Default for FSRS {
    fn default() -> Self {
        Self::new(&[]).expect("Default parameters should be valid")
    }
}

impl FSRS {
    /// - Parameters must be provided before running commands that need them.
    /// - Parameters may be an empty slice to use the default values instead.
    pub fn new(parameters: &Parameters) -> Result<Self> {
        let parameters = check_and_fill_parameters(parameters)?;
        let config = ModelConfig::default();
        let parameters =
            clip_parameters(&parameters, config.num_relearning_steps, Default::default());
        Ok(Self {
            parameters: parameters.try_into().unwrap(),
        })
    }

    pub(crate) const fn parameters(&self) -> &[f32; 21] {
        &self.parameters
    }

    #[inline]
    pub(crate) fn power_forgetting_curve(&self, t: f32, s: f32) -> f32 {
        power_forgetting_curve(&self.parameters, t, s)
    }

    #[inline]
    pub(crate) fn next_interval_for_stability(
        &self,
        stability: f32,
        desired_retention: f32,
    ) -> f32 {
        next_interval(&self.parameters, stability, desired_retention)
    }

    #[inline]
    pub(crate) fn init_stability(&self, rating: u32) -> f32 {
        init_stability(&self.parameters, rating as usize)
    }

    #[inline]
    pub(crate) fn init_difficulty(&self, rating: u32) -> f32 {
        init_difficulty(&self.parameters, rating as usize)
    }

    #[inline]
    #[cfg(test)]
    pub(crate) fn next_difficulty(&self, difficulty: f32, rating: u32) -> f32 {
        next_difficulty(&self.parameters, difficulty, rating as f32)
    }

    pub(crate) fn step(
        &self,
        delta_t: f32,
        rating: u32,
        state: MemoryState,
        nth: usize,
    ) -> MemoryState {
        step(&self.parameters, delta_t, rating as f32, state, nth)
    }

    /// If [starting_state] is provided, it will be used instead of the default initial stability/
    /// difficulty.
    pub(crate) fn forward_reviews(
        &self,
        reviews: &[FSRSReview],
        starting_state: Option<MemoryState>,
    ) -> MemoryState {
        let (mut state, start_index) = if let Some(state) = starting_state {
            (state, 0)
        } else if reviews.is_empty() {
            (
                MemoryState {
                    stability: 0.0,
                    difficulty: 0.0,
                },
                0,
            )
        } else {
            let rating = reviews[0].rating;
            if rating == 0 {
                (
                    MemoryState {
                        stability: S_MIN,
                        difficulty: D_MIN,
                    },
                    1,
                )
            } else {
                let rating = rating.clamp(1, 4);
                (
                    MemoryState {
                        stability: self.init_stability(rating).clamp(S_MIN, S_MAX),
                        difficulty: self.init_difficulty(rating).clamp(D_MIN, D_MAX),
                    },
                    1,
                )
            }
        };

        for (index, review) in reviews.iter().enumerate().skip(start_index) {
            state = self.step(review.delta_t as f32, review.rating, state, index);
        }
        state
    }
}

#[inline]
pub(crate) fn power_forgetting_curve(w: &[f32], t: f32, s: f32) -> f32 {
    let decay = -w[20];
    let factor = (0.9f32.ln() / decay).exp() - 1.0;
    (t / s * factor + 1.0).powf(decay)
}

#[inline]
pub(crate) fn next_interval(w: &[f32], stability: f32, desired_retention: f32) -> f32 {
    let decay = -w[20];
    let factor = (0.9f32.ln() / decay).exp() - 1.0;
    stability / factor * (desired_retention.powf(1.0 / decay) - 1.0)
}

#[inline]
pub(crate) fn init_stability(w: &[f32], rating: usize) -> f32 {
    w[rating.saturating_sub(1).min(3)]
}

#[inline]
pub(crate) fn init_difficulty(w: &[f32], rating: usize) -> f32 {
    w[4] - (w[5] * rating.saturating_sub(1) as f32).exp() + 1.0
}

#[inline]
fn mean_reversion(w: &[f32], new_d: f32) -> f32 {
    w[7] * (init_difficulty(w, 4) - new_d) + new_d
}

#[inline]
fn linear_damping(delta_d: f32, old_d: f32) -> f32 {
    (10.0 - old_d) * delta_d / 9.0
}

#[inline]
pub(crate) fn next_difficulty(w: &[f32], difficulty: f32, rating: f32) -> f32 {
    let delta_d = -w[6] * (rating - 3.0);
    difficulty + linear_damping(delta_d, difficulty)
}

#[inline]
fn stability_after_success(w: &[f32], last_s: f32, last_d: f32, r: f32, rating: f32) -> f32 {
    let hard_penalty = if rating == 2.0 { w[15] } else { 1.0 };
    let easy_bonus = if rating == 4.0 { w[16] } else { 1.0 };
    last_s
        * (w[8].exp()
            * (11.0 - last_d)
            * last_s.powf(-w[9])
            * (((1.0 - r) * w[10]).exp() - 1.0)
            * hard_penalty
            * easy_bonus
            + 1.0)
}

#[inline]
fn stability_after_failure(w: &[f32], last_s: f32, last_d: f32, r: f32) -> f32 {
    let new_s = w[11]
        * last_d.powf(-w[12])
        * ((last_s + 1.0).powf(w[13]) - 1.0)
        * ((1.0 - r) * w[14]).exp();
    let new_s_min = last_s / (w[17] * w[18]).exp();
    new_s.min(new_s_min)
}

#[inline]
fn stability_short_term(w: &[f32], last_s: f32, rating: f32) -> f32 {
    let sinc = (w[17] * (rating - 3.0 + w[18])).exp() * last_s.powf(-w[19]);
    last_s * if rating >= 2.0 { sinc.max(1.0) } else { sinc }
}

fn step(w: &[f32], delta_t: f32, rating: f32, state: MemoryState, nth: usize) -> MemoryState {
    let last_s = state.stability.clamp(S_MIN, S_MAX);
    let last_d = state.difficulty.clamp(D_MIN, D_MAX);

    let retrievability = power_forgetting_curve(w, delta_t, last_s);
    let stability_after_success =
        stability_after_success(w, last_s, last_d, retrievability, rating);
    let stability_after_failure = stability_after_failure(w, last_s, last_d, retrievability);
    let stability_short_term = stability_short_term(w, last_s, rating);

    let mut new_s = if rating == 1.0 {
        stability_after_failure
    } else {
        stability_after_success
    };
    if delta_t == 0.0 {
        new_s = stability_short_term;
    }

    let mut new_d = next_difficulty(w, last_d, rating);
    new_d = mean_reversion(w, new_d).clamp(D_MIN, D_MAX);

    if nth == 0 && state.stability == 0.0 {
        let init_rating = (rating as u32).clamp(1, 4) as usize;
        new_s = init_stability(w, init_rating);
        new_d = init_difficulty(w, init_rating).clamp(D_MIN, D_MAX);
    }

    if rating == 0.0 {
        new_s = last_s;
        new_d = last_d;
    }

    MemoryState {
        stability: new_s.clamp(S_MIN, S_MAX),
        difficulty: new_d,
    }
}

pub fn check_and_fill_parameters(parameters: &Parameters) -> Result<Vec<f32>, FSRSError> {
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
#[derive(Module, Debug)]
pub struct Model<B: Backend> {
    pub w: Param<Tensor<B, 1>>,
}

#[cfg(test)]
pub(crate) trait Get<B: Backend, const N: usize> {
    fn get(&self, n: usize) -> Tensor<B, N>;
}

#[cfg(test)]
impl<B: Backend, const N: usize> Get<B, N> for Tensor<B, N> {
    fn get(&self, n: usize) -> Self {
        self.clone().slice([n..(n + 1)])
    }
}

#[cfg(test)]
impl<B: Backend> Model<B> {
    pub fn new(config: ModelConfig) -> Self {
        Self::new_with_device(config, &B::Device::default())
    }

    pub fn new_with_device(config: ModelConfig, device: &B::Device) -> Self {
        Self {
            w: Param::from_tensor(Tensor::from_floats(
                TensorData::new(
                    config.initial_parameters().to_vec(),
                    Shape { dims: vec![21] },
                ),
                device,
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
        let device = rating.device();
        let hard_penalty = Tensor::ones([batch_size], &device)
            .mask_where(rating.clone().equal_elem(2), self.w.get(15));
        let easy_bonus =
            Tensor::ones([batch_size], &device).mask_where(rating.equal_elem(4), self.w.get(16));

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
                .mask_where(rating.greater_equal_elem(2), sinc.clamp_min(1.0))
    }

    fn mean_reversion(&self, new_d: Tensor<B, 1>) -> Tensor<B, 1> {
        let device = new_d.device();
        let rating = Tensor::from_floats([4.0], &device);
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
            let is_initial = state.stability.equal_elem(0.0);
            let init_s = self.init_stability(rating.clone().clamp(1, 4));
            let init_d = self
                .init_difficulty(rating.clone().clamp(1, 4))
                .clamp(D_MIN, D_MAX);
            new_s = new_s.mask_where(is_initial.clone(), init_s);
            new_d = new_d.mask_where(is_initial, init_d);
        }

        new_s = new_s.mask_where(rating.clone().equal_elem(0), last_s);
        new_d = new_d.mask_where(rating.equal_elem(0), last_d);
        MemoryStateTensors {
            stability: new_s.clamp(S_MIN, S_MAX),
            difficulty: new_d,
        }
    }

    pub(crate) fn forward(
        &self,
        delta_ts: Tensor<B, 2>,
        ratings: Tensor<B, 2>,
        starting_state: Option<MemoryStateTensors<B>>,
    ) -> MemoryStateTensors<B> {
        let [seq_len, batch_size] = delta_ts.dims();
        let (mut state, start_index) = if let Some(state) = starting_state {
            (state, 0)
        } else if seq_len == 0 {
            (MemoryStateTensors::zeros(batch_size), 0)
        } else {
            let rating = ratings.get(0).squeeze(0);
            let initial_rating = rating.clone().clamp(1, 4);
            let mut stability = self
                .init_stability(initial_rating.clone())
                .clamp(S_MIN, S_MAX);
            let mut difficulty = self.init_difficulty(initial_rating).clamp(D_MIN, D_MAX);
            let padding = rating.equal_elem(0);
            let device = stability.device();
            stability = stability.mask_where(
                padding.clone(),
                Tensor::ones([batch_size], &device).mul_scalar(S_MIN),
            );
            difficulty = difficulty.mask_where(
                padding,
                Tensor::ones([batch_size], &device).mul_scalar(D_MIN),
            );
            (
                MemoryStateTensors {
                    stability,
                    difficulty,
                },
                1,
            )
        };
        for i in start_index..seq_len {
            let delta_t = delta_ts.get(i).squeeze(0);
            let rating = ratings.get(i).squeeze(0);
            state = self.step(delta_t, rating, state, i);
        }
        state
    }
}

#[cfg(test)]
#[derive(Debug, Clone)]
pub(crate) struct MemoryStateTensors<B: Backend> {
    pub stability: Tensor<B, 1>,
    pub difficulty: Tensor<B, 1>,
}

#[cfg(test)]
impl<B: Backend> MemoryStateTensors<B> {
    pub(crate) fn zeros(batch_size: usize) -> MemoryStateTensors<B> {
        let device = B::Device::default();
        MemoryStateTensors {
            stability: Tensor::zeros([batch_size], &device),
            difficulty: Tensor::zeros([batch_size], &device),
        }
    }

    pub(crate) fn from_state(state: MemoryState) -> Self {
        let device = B::Device::default();
        Self {
            stability: Tensor::from_floats([state.stability], &device),
            difficulty: Tensor::from_floats([state.difficulty], &device),
        }
    }
}

#[cfg(test)]
impl<B: Backend> From<MemoryStateTensors<B>> for MemoryState {
    fn from(m: MemoryStateTensors<B>) -> Self {
        use burn::tensor::ElementConversion;
        Self {
            stability: m.stability.into_scalar().elem(),
            difficulty: m.difficulty.into_scalar().elem(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_helpers::TestHelper;
    use crate::test_helpers::{Model, Tensor};
    use burn::backend::ndarray::NdArrayDevice;
    use burn::tensor::TensorData;
    static DEVICE: NdArrayDevice = NdArrayDevice::Cpu;

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
        let fsrs = FSRS::default();
        let retrievability = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
            .into_iter()
            .zip([1.0, 2.0, 3.0, 4.0, 4.0, 2.0])
            .map(|(delta_t, stability)| fsrs.power_forgetting_curve(delta_t, stability))
            .collect::<Vec<_>>();

        retrievability.assert_approx_eq([1.0, 0.9403443, 0.9253786, 0.9185229, 0.9, 0.8261359]);
    }

    #[test]
    fn test_tensor_next_interval_matches_scalar() {
        let model = Model::new(ModelConfig::default());
        let fsrs = FSRS::default();
        let stabilities = [1.0, 2.0, 5.0, 121.01552];
        let desired_retentions = [0.6, 0.8, 0.9, 0.95];
        let burn_intervals = model
            .next_interval(
                Tensor::from_floats(stabilities, &DEVICE),
                Tensor::from_floats(desired_retentions, &DEVICE),
            )
            .to_data()
            .to_vec::<f32>()
            .unwrap();
        let scalar_intervals = [
            fsrs.next_interval_for_stability(stabilities[0], desired_retentions[0]),
            fsrs.next_interval_for_stability(stabilities[1], desired_retentions[1]),
            fsrs.next_interval_for_stability(stabilities[2], desired_retentions[2]),
            fsrs.next_interval_for_stability(stabilities[3], desired_retentions[3]),
        ];

        burn_intervals.assert_approx_eq(scalar_intervals);
    }

    #[test]
    fn test_init_stability() {
        let fsrs = FSRS::default();
        let stability = [1, 2, 3, 4, 1, 2].map(|rating| fsrs.init_stability(rating));
        assert_eq!(
            stability,
            [
                DEFAULT_PARAMETERS[0],
                DEFAULT_PARAMETERS[1],
                DEFAULT_PARAMETERS[2],
                DEFAULT_PARAMETERS[3],
                DEFAULT_PARAMETERS[0],
                DEFAULT_PARAMETERS[1]
            ]
        )
    }

    #[test]
    fn test_init_difficulty() {
        let fsrs = FSRS::default();
        let difficulty = [1, 2, 3, 4, 1, 2].map(|rating| fsrs.init_difficulty(rating));
        assert_eq!(
            difficulty,
            [
                DEFAULT_PARAMETERS[4],
                DEFAULT_PARAMETERS[4] - DEFAULT_PARAMETERS[5].exp() + 1.0,
                DEFAULT_PARAMETERS[4] - (2.0 * DEFAULT_PARAMETERS[5]).exp() + 1.0,
                DEFAULT_PARAMETERS[4] - (3.0 * DEFAULT_PARAMETERS[5]).exp() + 1.0,
                DEFAULT_PARAMETERS[4],
                DEFAULT_PARAMETERS[4] - DEFAULT_PARAMETERS[5].exp() + 1.0,
            ]
        )
    }

    #[test]
    fn test_forward_matches_burn_oracle() {
        let model = Model::new(ModelConfig::default());
        let delta_ts = Tensor::from_floats(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0, 1.0, 2.0, 2.0],
            ],
            &DEVICE,
        );
        let ratings = Tensor::from_floats(
            [
                [1.0, 2.0, 3.0, 4.0, 1.0, 2.0],
                [1.0, 2.0, 3.0, 4.0, 1.0, 2.0],
            ],
            &DEVICE,
        );
        let state = model.forward(delta_ts, ratings, None);
        let burn_stability = state.stability.to_data().to_vec::<f32>().unwrap();
        let burn_difficulty = state.difficulty.to_data().to_vec::<f32>().unwrap();

        let fsrs = FSRS::default();
        let scalar_states = [
            [(1, 0), (1, 1)],
            [(2, 0), (2, 1)],
            [(3, 0), (3, 1)],
            [(4, 0), (4, 1)],
            [(1, 0), (1, 2)],
            [(2, 0), (2, 2)],
        ]
        .map(|reviews| FSRSReview {
            rating: reviews[0].0,
            delta_t: reviews[0].1,
        })
        .into_iter()
        .zip([
            FSRSReview {
                rating: 1,
                delta_t: 1,
            },
            FSRSReview {
                rating: 2,
                delta_t: 1,
            },
            FSRSReview {
                rating: 3,
                delta_t: 1,
            },
            FSRSReview {
                rating: 4,
                delta_t: 1,
            },
            FSRSReview {
                rating: 1,
                delta_t: 2,
            },
            FSRSReview {
                rating: 2,
                delta_t: 2,
            },
        ])
        .map(|(first, second)| fsrs.forward_reviews(&[first, second], None))
        .collect::<Vec<_>>();

        let scalar_stability = scalar_states
            .iter()
            .map(|state| state.stability)
            .collect::<Vec<_>>();
        let scalar_difficulty = scalar_states
            .iter()
            .map(|state| state.difficulty)
            .collect::<Vec<_>>();
        let burn_stability: [f32; 6] = burn_stability.try_into().unwrap();
        let burn_difficulty: [f32; 6] = burn_difficulty.try_into().unwrap();
        scalar_stability.assert_approx_eq(burn_stability);
        scalar_difficulty.assert_approx_eq(burn_difficulty);
    }

    #[test]
    fn test_next_difficulty() {
        let fsrs = FSRS::default();
        let next_difficulty = [1, 2, 3, 4].map(|rating| fsrs.next_difficulty(5.0, rating));
        next_difficulty.assert_approx_eq([8.354889, 6.6774445, 5.0, 3.3225555]);
        let next_difficulty = next_difficulty.map(|value| mean_reversion(fsrs.parameters(), value));
        next_difficulty.assert_approx_eq([8.341763, 6.6659956, 4.990228, 3.3144615]);
    }

    #[test]
    fn test_next_stability() {
        let w = DEFAULT_PARAMETERS;
        let s_recall = [1.0, 2.0, 3.0, 4.0]
            .into_iter()
            .zip([0.9, 0.8, 0.7, 0.6])
            .map(|(rating, retrievability)| {
                stability_after_success(&w, 5.0, rating, retrievability, rating)
            })
            .collect::<Vec<_>>();
        s_recall.assert_approx_eq([25.602541, 28.226582, 58.656002, 127.226685]);

        let s_forget = [1.0, 2.0, 3.0, 4.0]
            .into_iter()
            .zip([0.9, 0.8, 0.7, 0.6])
            .map(|(difficulty, retrievability)| {
                stability_after_failure(&w, 5.0, difficulty, retrievability)
            })
            .collect::<Vec<_>>();
        s_forget.assert_approx_eq([1.0525396, 1.1894329, 1.3680838, 1.584989]);

        let next_stability = [s_forget[0], s_recall[1], s_recall[2], s_recall[3]];
        next_stability.assert_approx_eq([1.0525396, 28.226582, 58.656002, 127.226685]);

        let next_stability =
            [1.0, 2.0, 3.0, 4.0].map(|rating| stability_short_term(&w, 5.0, rating));
        next_stability.assert_approx_eq([1.596818, 5.0, 5.0, 8.12961]);
    }

    #[test]
    fn test_fsrs() {
        FSRS::default()
            .parameters()
            .to_vec()
            .assert_approx_eq(DEFAULT_PARAMETERS);
        assert!(FSRS::new(&[]).is_ok());
        assert!(FSRS::new(&[1.]).is_err());
        assert!(FSRS::new(DEFAULT_PARAMETERS.as_slice()).is_ok());
        assert!(FSRS::new(&DEFAULT_PARAMETERS[..17]).is_ok());
    }

    #[test]
    fn scalar_step_matches_burn_oracle() {
        let model = Model::new(ModelConfig::default());
        let fsrs = FSRS::default();
        let starting = MemoryState {
            stability: 5.0,
            difficulty: 6.0,
        };
        for (nth, state) in [
            (
                0,
                MemoryState {
                    stability: 0.0,
                    difficulty: 0.0,
                },
            ),
            (1, starting),
        ] {
            for delta_t in [0.0, 1.0, 21.0] {
                for rating in 0..=4 {
                    let burn_state: MemoryState = model
                        .step(
                            Tensor::from_floats([delta_t], &DEVICE),
                            Tensor::from_floats([rating as f32], &DEVICE),
                            MemoryStateTensors::from_state(state),
                            nth,
                        )
                        .into();
                    let scalar_state = fsrs.step(delta_t, rating, state, nth);
                    [scalar_state.stability, scalar_state.difficulty]
                        .assert_approx_eq([burn_state.stability, burn_state.difficulty]);
                }
            }
        }
    }
}

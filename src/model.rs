use crate::DEFAULT_PARAMETERS;
use crate::error::{FSRSError, Result};
use crate::inference::{MemoryState, Parameters};
use crate::parameter_clipper::clip_parameters;
use crate::simulation::{D_MAX, D_MIN, S_MAX, S_MIN};
use burn::backend::NdArray;
use burn::backend::ndarray::NdArrayDevice;
use burn::{
    config::Config,
    constant,
    module::{Module, Param},
    tensor::{Shape, Tensor, TensorData, backend::Backend},
};

#[path = "model_v6.rs"]
pub(crate) mod model_v6;
#[path = "model_v7.rs"]
pub(crate) mod model_v7;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ModelVersion {
    Fsrs6,
    Fsrs7,
}

impl ModelVersion {
    pub(crate) fn from_param_count(param_count: usize) -> Self {
        if param_count == model_v7::PARAM_LEN {
            Self::Fsrs7
        } else {
            Self::Fsrs6
        }
    }
}

impl core::fmt::Display for ModelVersion {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Fsrs6 => write!(f, "FSRS6"),
            Self::Fsrs7 => write!(f, "FSRS7"),
        }
    }
}

constant!(ModelVersion);

#[derive(Module, Debug)]
pub struct Model<B: Backend> {
    pub w: Param<Tensor<B, 1>>,
    version: ModelVersion,
}

pub(crate) trait Get<B: Backend, const N: usize> {
    fn get(&self, n: usize) -> Tensor<B, N>;
}

impl<B: Backend, const N: usize> Get<B, N> for Tensor<B, N> {
    fn get(&self, n: usize) -> Self {
        self.clone().slice([n..(n + 1)])
    }
}

fn tensor_min<B: Backend>(a: Tensor<B, 1>, b: Tensor<B, 1>) -> Tensor<B, 1> {
    a.clone().mask_where(a.clone().greater(b.clone()), b)
}

fn tensor_max<B: Backend>(a: Tensor<B, 1>, b: Tensor<B, 1>) -> Tensor<B, 1> {
    a.clone().mask_where(a.clone().lower(b.clone()), b)
}

pub(super) trait VersionOps<B: Backend> {
    fn apply_freeze_short_term(initial_params: &mut [f32]);
    fn power_forgetting_curve(model: &Model<B>, t: Tensor<B, 1>, s: Tensor<B, 1>) -> Tensor<B, 1>;
    fn next_interval(
        model: &Model<B>,
        stability: Tensor<B, 1>,
        desired_retention: Tensor<B, 1>,
    ) -> Tensor<B, 1>;
    fn update_state(
        model: &Model<B>,
        delta_t: Tensor<B, 1>,
        rating: Tensor<B, 1>,
        last_s: Tensor<B, 1>,
        last_d: Tensor<B, 1>,
    ) -> MemoryStateTensors<B>;
    fn memory_state_from_sm2_fsrs(
        model: &Model<B>,
        ease_factor: f32,
        interval: f32,
        sm2_retention: f32,
    ) -> Result<MemoryState>;
    fn interval_at_retrievability(
        model: &Model<B>,
        stability: f32,
        target_retrievability: f32,
    ) -> f32;
}

pub(super) struct Fsrs6Ops;
pub(super) struct Fsrs7Ops;

type ApplyFreezeShortTermFn = fn(&mut [f32]);
type PowerForgettingCurveFn<B> = fn(&Model<B>, Tensor<B, 1>, Tensor<B, 1>) -> Tensor<B, 1>;
type NextIntervalFn<B> = fn(&Model<B>, Tensor<B, 1>, Tensor<B, 1>) -> Tensor<B, 1>;
type UpdateStateFn<B> =
    fn(&Model<B>, Tensor<B, 1>, Tensor<B, 1>, Tensor<B, 1>, Tensor<B, 1>) -> MemoryStateTensors<B>;
type MemoryStateFromSm2Fn<B> = fn(&Model<B>, f32, f32, f32) -> Result<MemoryState>;
type IntervalAtRetrievabilityFn<B> = fn(&Model<B>, f32, f32) -> f32;

#[derive(Clone, Copy)]
struct VersionFns<B: Backend> {
    apply_freeze_short_term: ApplyFreezeShortTermFn,
    power_forgetting_curve: PowerForgettingCurveFn<B>,
    next_interval: NextIntervalFn<B>,
    update_state: UpdateStateFn<B>,
    memory_state_from_sm2: MemoryStateFromSm2Fn<B>,
    interval_at_retrievability: IntervalAtRetrievabilityFn<B>,
}

impl<B: Backend> VersionFns<B> {
    fn from_version(version: ModelVersion) -> Self {
        match version {
            ModelVersion::Fsrs6 => Self {
                apply_freeze_short_term: <Fsrs6Ops as VersionOps<B>>::apply_freeze_short_term,
                power_forgetting_curve: <Fsrs6Ops as VersionOps<B>>::power_forgetting_curve,
                next_interval: <Fsrs6Ops as VersionOps<B>>::next_interval,
                update_state: <Fsrs6Ops as VersionOps<B>>::update_state,
                memory_state_from_sm2: <Fsrs6Ops as VersionOps<B>>::memory_state_from_sm2_fsrs,
                interval_at_retrievability: <Fsrs6Ops as VersionOps<B>>::interval_at_retrievability,
            },
            ModelVersion::Fsrs7 => Self {
                apply_freeze_short_term: <Fsrs7Ops as VersionOps<B>>::apply_freeze_short_term,
                power_forgetting_curve: <Fsrs7Ops as VersionOps<B>>::power_forgetting_curve,
                next_interval: <Fsrs7Ops as VersionOps<B>>::next_interval,
                update_state: <Fsrs7Ops as VersionOps<B>>::update_state,
                memory_state_from_sm2: <Fsrs7Ops as VersionOps<B>>::memory_state_from_sm2_fsrs,
                interval_at_retrievability: <Fsrs7Ops as VersionOps<B>>::interval_at_retrievability,
            },
        }
    }
}

impl<B: Backend> Model<B> {
    #[cfg(test)]
    pub fn new(config: ModelConfig) -> Self {
        Self::new_with_device(config, &B::Device::default())
    }

    pub fn new_with_device(config: ModelConfig, device: &B::Device) -> Self {
        let mut initial_params = DEFAULT_PARAMETERS.to_vec();
        let version = ModelVersion::from_param_count(initial_params.len());
        if let Some(initial_stability) = config.initial_stability {
            initial_params[0..4].copy_from_slice(&initial_stability);
        }
        if let Some(initial_forgetting_curve) = config.initial_forgetting_curve {
            initial_params[27..35].copy_from_slice(&initial_forgetting_curve);
        }
        if config.freeze_short_term_stability {
            let ops = VersionFns::<B>::from_version(version);
            (ops.apply_freeze_short_term)(&mut initial_params);
        }

        Self {
            w: Param::from_tensor(Tensor::from_floats(
                TensorData::new(
                    initial_params.clone(),
                    Shape {
                        dims: vec![initial_params.len()],
                    },
                ),
                device,
            )),
            version,
        }
    }

    pub(crate) fn version(&self) -> ModelVersion {
        self.version
    }

    pub(crate) fn memory_state_from_sm2(
        &self,
        ease_factor: f32,
        interval: f32,
        sm2_retention: f32,
    ) -> Result<MemoryState> {
        let ops = VersionFns::<B>::from_version(self.version());
        (ops.memory_state_from_sm2)(self, ease_factor, interval, sm2_retention)
    }

    pub fn power_forgetting_curve(&self, t: Tensor<B, 1>, s: Tensor<B, 1>) -> Tensor<B, 1> {
        let ops = VersionFns::<B>::from_version(self.version());
        (ops.power_forgetting_curve)(self, t, s)
    }

    pub fn next_interval(
        &self,
        stability: Tensor<B, 1>,
        desired_retention: Tensor<B, 1>,
    ) -> Tensor<B, 1> {
        let ops = VersionFns::<B>::from_version(self.version());
        (ops.next_interval)(self, stability, desired_retention)
    }

    pub(crate) fn interval_at_retrievability(
        &self,
        stability: f32,
        target_retrievability: f32,
    ) -> f32 {
        let ops = VersionFns::<B>::from_version(self.version());
        (ops.interval_at_retrievability)(self, stability, target_retrievability)
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
        let ops = VersionFns::<B>::from_version(self.version());
        self.step_with_ops(&ops, delta_t, rating, state, nth)
    }

    fn step_with_ops(
        &self,
        ops: &VersionFns<B>,
        delta_t: Tensor<B, 1>,
        rating: Tensor<B, 1>,
        state: MemoryStateTensors<B>,
        nth: usize,
    ) -> MemoryStateTensors<B> {
        let last_s = state.stability.clone().clamp(S_MIN, S_MAX);
        let last_d = state.difficulty.clone().clamp(D_MIN, D_MAX);
        let mut new_state = (ops.update_state)(
            self,
            delta_t.clone(),
            rating.clone(),
            last_s.clone(),
            last_d.clone(),
        );

        if nth == 0 {
            let is_initial = state.stability.clone().equal_elem(0.0);
            let init_s = self.init_stability(rating.clone().clamp(1, 4));
            let init_d = self
                .init_difficulty(rating.clone().clamp(1, 4))
                .clamp(D_MIN, D_MAX);
            new_state.stability = new_state.stability.mask_where(is_initial.clone(), init_s);
            new_state.difficulty = new_state.difficulty.mask_where(is_initial, init_d);
        }

        // mask padding zeros for rating
        new_state.stability = new_state
            .stability
            .mask_where(rating.clone().equal_elem(0), last_s)
            .clamp(S_MIN, S_MAX);
        new_state.difficulty = new_state
            .difficulty
            .mask_where(rating.equal_elem(0), last_d);

        new_state
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
        let ops = VersionFns::<B>::from_version(self.version());
        for i in 0..seq_len {
            let delta_t = delta_ts.get(i).squeeze(0);
            let rating = ratings.get(i).squeeze(0);
            state = self.step_with_ops(&ops, delta_t, rating, state, i);
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

#[derive(Config, Debug, Default)]
pub struct ModelConfig {
    #[config(default = false)]
    pub freeze_initial_stability: bool,
    pub initial_stability: Option<[f32; 4]>,
    pub initial_forgetting_curve: Option<[f32; 8]>,
    #[config(default = false)]
    pub freeze_short_term_stability: bool,
    #[config(default = 1)]
    pub num_relearning_steps: usize,
}

impl ModelConfig {
    #[cfg(test)]
    pub fn init<B: Backend>(&self) -> Model<B> {
        Model::new(self.clone())
    }
}

/// This is the main structure provided by this crate. It can be used
/// for both parameter training, and for reviews.
#[derive(Debug, Clone)]
pub struct FSRS<B: Backend = NdArray> {
    model: Model<B>,
}

impl Default for FSRS<NdArray> {
    fn default() -> Self {
        Self::new(&[]).expect("Default parameters should be valid")
    }
}

impl FSRS<NdArray> {
    /// - Parameters must be provided before running commands that need them.
    /// - Parameters may be an empty slice to use the default values instead.
    pub fn new(parameters: &Parameters) -> Result<Self> {
        Self::new_with_backend(parameters, &NdArrayDevice::Cpu)
    }
}

impl<B: Backend> FSRS<B> {
    pub fn new_with_backend<B2: Backend>(
        parameters: &Parameters,
        device: &B2::Device,
    ) -> Result<FSRS<B2>> {
        let parameters = check_and_fill_parameters(parameters)?;
        let model = parameters_to_model::<B2>(&parameters, device);

        Ok(FSRS { model })
    }

    pub(crate) fn model(&self) -> &Model<B> {
        &self.model
    }

    pub(crate) fn device(&self) -> B::Device {
        self.model().w.device()
    }
}

pub(crate) fn parameters_to_model<B: Backend>(
    parameters: &Parameters,
    device: &B::Device,
) -> Model<B> {
    let config = ModelConfig::default();
    let mut model = Model::new_with_device(config.clone(), device);
    let clipped = clip_parameters(parameters, config.num_relearning_steps, Default::default());
    model.w = Param::from_tensor(Tensor::from_floats(
        TensorData::new(
            clipped.clone(),
            Shape {
                dims: vec![clipped.len()],
            },
        ),
        device,
    ));
    model.version = ModelVersion::from_param_count(clipped.len());
    model
}

pub(crate) fn check_and_fill_parameters(parameters: &Parameters) -> Result<Vec<f32>, FSRSError> {
    let parameters = if parameters.len() == model_v7::PARAM_LEN {
        parameters.to_vec()
    } else if let Some(parameters) = model_v6::check_and_fill_parameters_fsrs6(parameters) {
        parameters
    } else {
        return Err(FSRSError::InvalidParameters);
    };
    if parameters.iter().any(|&w| !w.is_finite()) {
        return Err(FSRSError::InvalidParameters);
    }
    Ok(parameters)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::FSRS6_DEFAULT_PARAMETERS;
    use crate::test_helpers::{Model as TestModel, NdArrayAutodiff, TestHelper};
    use burn::backend::ndarray::NdArrayDevice;
    use burn::tensor::TensorData;

    static DEVICE: NdArrayDevice = NdArrayDevice::Cpu;

    #[test]
    fn test_w() {
        let model: TestModel = Model::new(ModelConfig::default());
        assert_eq!(
            model.w.val().to_data(),
            TensorData::new(DEFAULT_PARAMETERS.to_vec(), Shape { dims: vec![35] })
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
    fn test_fsrs() {
        FSRS::default()
            .model()
            .w
            .to_data()
            .to_vec::<f32>()
            .unwrap()
            .assert_approx_eq(FSRS6_DEFAULT_PARAMETERS);
        assert!(FSRS::new(&[]).is_ok());
        assert!(FSRS::new(&[1.]).is_err());
        assert!(FSRS::new(DEFAULT_PARAMETERS.as_slice()).is_ok());
        assert!(FSRS::new(&FSRS6_DEFAULT_PARAMETERS[..17]).is_ok());
        assert!(FSRS::new(&FSRS6_DEFAULT_PARAMETERS).is_ok());
    }

    #[test]
    fn test_model_version_selection() {
        let model_v7: TestModel = Model::new(ModelConfig::default());
        assert_eq!(model_v7.version(), ModelVersion::Fsrs7);

        let model_v6 = parameters_to_model::<NdArrayAutodiff>(&FSRS6_DEFAULT_PARAMETERS, &DEVICE);
        assert_eq!(model_v6.version(), ModelVersion::Fsrs6);
    }
}

use crate::DEFAULT_PARAMETERS;
use crate::error::{FSRSError, Result};
use crate::inference::{FSRS5_DEFAULT_DECAY, Parameters, S_MAX, S_MIN};
use crate::parameter_clipper::clip_parameters;
use candle_core::{Device, Tensor};
use candle_nn::{Module, VarBuilder};

use candle_nn::VarMap;

#[derive(Debug)]
pub struct Model {
    pub w: candle_core::Var, // Changed from Tensor to Var
    device: Device,
    varmap: VarMap, // To store and manage variables
}

// TODO: Is there a candle equivalent of burn::tensor::backend::Backend?
// Candle seems to use Device for specifying computation backend (CPU/GPU).
// For now, I'll remove the generic Backend parameter `B` and use Device directly.

// TODO: Is there a candle equivalent of burn::tensor::Shape?
// Candle tensors have a `shape()` method that returns a `Shape` struct.
// However, Shape is also used in TensorData construction in burn.
// For now, I'll replace burn::tensor::Shape with candle_core::Shape where applicable.

// TODO: Is there a candle equivalent of burn::tensor::TensorData?
// Candle tensors can be created from slices, vectors, or other tensors.
// For now, I'll replace TensorData with direct tensor creation from data.

impl Model {
    #[allow(clippy::new_without_default)]
    pub fn new(config: ModelConfig, device: Device, varmap: VarMap) -> Result<Self> {
        let mut initial_params_vec: Vec<f32> = config
            .initial_stability
            .unwrap_or_else(|| DEFAULT_PARAMETERS[0..4].try_into().unwrap())
            .into_iter()
            .chain(DEFAULT_PARAMETERS[4..].iter().copied())
            .collect();
        if config.freeze_short_term_stability {
            initial_params_vec[17] = 0.0;
            initial_params_vec[18] = 0.0;
            initial_params_vec[19] = 0.0;
        }

        // Create a VarBuilder to manage the lifecycle of `w`
        let vb = VarBuilder::from_varmap(&varmap, candle_core::DType::F32, &device);

        // Initialize `w` using VarBuilder. If loading pre-trained, ensure data is correctly passed.
        // For now, initializing with `initial_params_vec`.
        // The name "w" will be used to identify this variable in the VarMap.
        let w = vb.get_or_init("w", candle_core::initializer::Const(initial_params_vec.as_slice()))?;
        // TODO: Confirm if get_or_init with a slice works as expected for initializing.
        // It seems get_or_init might be more for shape-based initialization with an Initializer.
        // Consider vb.var_from_tensor("w", &initial_tensor, &device)? if direct tensor init is needed.
        // For now, assuming get_or_init with Const works or can be adapted.
        // Let's try creating the tensor first and then the var.
        let initial_w_tensor = Tensor::from_vec(initial_params_vec, (21,), &device)?;
        let w = vb.var_from_tensor("w", &initial_w_tensor)?;


        Ok(Self { w, device, varmap })
    }

    // Helper to access the tensor inside `w` (Var)
    fn w_tensor(&self) -> &Tensor {
        self.w.as_tensor()
    }

    pub fn power_forgetting_curve(&self, t: &Tensor, s: &Tensor) -> Result<Tensor> {
        let w = self.w_tensor();
        let decay = (w.i(20)? * -1.0)?;
        let factor = (((decay.powf(-1.0)? * 0.9f32.ln())?.exp()? - 1.0))?;
        (t / s * factor + 1.0)?.powf(&decay)
    }

    pub fn next_interval(
        &self,
        stability: &Tensor,
        desired_retention: &Tensor,
    ) -> Result<Tensor> {
        let w = self.w_tensor();
        let decay = (w.i(20)? * -1.0)?;
        let factor = (((decay.powf(-1.0)? * 0.9f32.ln())?.exp()? - 1.0))?;
        (stability / factor * (desired_retention.powf(&(decay.powf(-1.0)?))? - 1.0))
    }

    fn stability_after_success(
        &self,
        last_s: &Tensor,
        last_d: &Tensor,
        r: &Tensor,
        rating: &Tensor,
    ) -> Result<Tensor> {
        let w = self.w_tensor();
        let batch_size = rating.dims()[0];
        let hard_penalty = Tensor::ones(batch_size, candle_core::DType::F32, &self.device)?
            .where_cond(&rating.eq_lit(2.0)?, &w.i(15)?, &Tensor::ones_like(rating)?)?;
        let easy_bonus = Tensor::ones(batch_size, candle_core::DType::F32, &self.device)?
            .where_cond(&rating.eq_lit(4.0)?, &w.i(16)?, &Tensor::ones_like(rating)?)?;

        (last_s
            * ((w.i(8)?.exp()?
                * (last_d * -1.0 + 11.0)?
                * (last_s.powf(&w.i(9)?.neg()?)?)?
                * (((r * -1.0 + 1.0)? * w.i(10)?)?.exp()? - 1.0)?
                * hard_penalty)?
                * easy_bonus)?
            + 1.0)
    }

    fn stability_after_failure(
        &self,
        last_s: &Tensor,
        last_d: &Tensor,
        r: &Tensor,
    ) -> Result<Tensor> {
        let w = self.w_tensor();
        let new_s = (w.i(11)?
            * last_d.powf(&w.i(12)?.neg()?)?
            * ((last_s + 1.0)?.powf(&w.i(13)?)? - 1.0)?
            * (((r * -1.0 + 1.0)? * w.i(14)?)?.exp())?)?;
        let new_s_min = (last_s / (w.i(17)? * w.i(18)?)?.exp())?;
        new_s.where_cond(&new_s_min.lt(&new_s)?, &new_s_min, &new_s)
    }

    fn stability_short_term(&self, last_s: &Tensor, rating: &Tensor) -> Result<Tensor> {
        let w = self.w_tensor();
        let sinc = ((w.i(17)? * (rating - 3.0 + w.i(18)?)?)?.exp()?
            * last_s.powf(&w.i(19)?.neg()?)?)?;

        (last_s * sinc.where_cond(&rating.ge_lit(3.0)?, &sinc.minimum(1.0)?, &sinc)?)
    }

    fn mean_reversion(&self, new_d: &Tensor) -> Result<Tensor> {
        let w = self.w_tensor();
        let rating = Tensor::from_slice(&[4.0], (1,), &self.device)?;
        (w.i(7)? * (self.init_difficulty(&rating)? - new_d)? + new_d)
    }

    pub(crate) fn init_stability(&self, rating: &Tensor) -> Result<Tensor> {
        let w = self.w_tensor();
        // Convert rating to indices (u32 or i64). Let's use u32.
        // Subtract 1 from rating to get 0-based indices.
        let indices = (rating - 1.0)?.to_dtype(candle_core::DType::U32)?;
        // Select from self.w along dimension 0 using the calculated indices.
        w.index_select(&indices, 0)
    }

    fn init_difficulty(&self, rating: &Tensor) -> Result<Tensor> {
        let w = self.w_tensor();
        (w.i(4)? - (w.i(5)? * (rating - 1.0)?)?.exp()? + 1.0)
    }

    fn linear_damping(&self, delta_d: &Tensor, old_d: &Tensor) -> Result<Tensor> {
        ((old_d.neg()? + 10.0)? * delta_d / 9.0)
    }

    fn next_difficulty(&self, difficulty: &Tensor, rating: &Tensor) -> Result<Tensor> {
        let w = self.w_tensor();
        let delta_d = (w.i(6)?.neg()? * (rating - 3.0)?)?;
        (difficulty + self.linear_damping(&delta_d, difficulty)?)
    }

    pub(crate) fn step(
        &self,
        delta_t: &Tensor,
        rating: &Tensor,
        state: Option<MemoryStateTensors>,
    ) -> Result<MemoryStateTensors> {
        let (new_s, new_d) = if let Some(state) = state {
            let retrievability =
                self.power_forgetting_curve(delta_t, &state.stability)?;
            let stability_after_success = self.stability_after_success(
                &state.stability,
                &state.difficulty,
                &retrievability,
                rating,
            )?;
            let stability_after_failure = self.stability_after_failure(
                &state.stability,
                &state.difficulty,
                &retrievability,
            )?;
            let stability_short_term =
                self.stability_short_term(&state.stability, rating)?;
            let mut new_stability = stability_after_success
                .where_cond(&rating.eq_lit(1.0)?, &stability_after_failure, &stability_after_success)?;
            new_stability = new_stability.where_cond(&delta_t.eq_lit(0.0)?, &stability_short_term, &new_stability)?;

            let mut new_difficulty = self.next_difficulty(&state.difficulty, rating)?;
            new_difficulty = self.mean_reversion(&new_difficulty)?.clamp(1.0, 10.0)?;
            // mask padding zeros for rating
            new_stability = new_stability.where_cond(&rating.eq_lit(0.0)?, &state.stability, &new_stability)?;
            new_difficulty = new_difficulty.where_cond(&rating.eq_lit(0.0)?, &state.difficulty, &new_difficulty)?;
            (new_stability, new_difficulty)
        } else {
            (
                self.init_stability(rating)?,
                self.init_difficulty(rating)?.clamp(1.0, 10.0)?,
            )
        };
        Ok(MemoryStateTensors {
            stability: new_s.clamp(S_MIN, S_MAX)?,
            difficulty: new_d,
        })
    }

    /// If [starting_state] is provided, it will be used instead of the default initial stability/
    /// difficulty.
    pub(crate) fn forward(
        &self,
        delta_ts: &Tensor, // Assuming Tensor now, not Tensor<B, 2>
        ratings: &Tensor,  // Assuming Tensor now, not Tensor<B, 2>
        starting_state: Option<MemoryStateTensors>,
    ) -> Result<MemoryStateTensors> {
        let seq_len = delta_ts.dims()[0];
        let mut state = starting_state;
        for i in 0..seq_len {
            let delta_t = delta_ts.i((i, ..))?;
            let rating = ratings.i((i, ..))?;
            state = Some(self.step(&delta_t, &rating, state)?);
        }
        // state.unwrap() // This will panic if state is None. Consider returning Result.
        state.ok_or_else(|| FSRSError::Internal("Forward resulted in None state".to_string()))
    }
}

#[derive(Debug, Clone)]
pub(crate) struct MemoryStateTensors {
    pub stability: Tensor,
    pub difficulty: Tensor,
}

#[derive(Debug, Default, Clone)] // Added Clone manually
pub struct ModelConfig {
    // #[config(default = false)] // Removed burn-specific macro
    pub freeze_initial_stability: bool,
    pub initial_stability: Option<[f32; 4]>,
    // #[config(default = false)] // Removed burn-specific macro
    pub freeze_short_term_stability: bool,
    // #[config(default = 1)] // Removed burn-specific macro
    pub num_relearning_steps: usize,
}

impl ModelConfig {
    // Update init to reflect changes in Model::new
    pub fn init(&self, device: Device, varmap: VarMap) -> Result<Model> {
        Model::new(self.clone(), device, varmap)
    }
}

/// This is the main structure provided by this crate. It can be used
/// for both parameter training, and for reviews.
#[derive(Debug, Clone)]
pub struct FSRS {
    model: Option<Model>,
    device: Device,
}

// TODO: NdArray specific implementation needs to be updated or removed
// if candle doesn't have a direct NdArray equivalent or if we aim for a more generic backend.
// For now, I'll assume Device::Cpu is the equivalent of NdArrayDevice::Cpu.
impl FSRS {
    /// - Parameters must be provided before running commands that need them.
    /// - Parameters may be an empty slice to use the default values instead.
    pub fn new(parameters: Option<&Parameters>) -> Result<Self> {
        Self::new_with_device(parameters, Device::Cpu)
    }
}

impl FSRS {
    pub fn new_with_device(
        parameters: Option<&Parameters>,
        device: Device,
    ) -> Result<Self> {
        let varmap = VarMap::new(); // Create a new VarMap
        let model = match parameters {
            Some(params) => {
                let parameters = check_and_fill_parameters(params)?;
                // parameters_to_model needs to be updated for candle
                let model = parameters_to_model(&parameters, device.clone(), varmap.clone())?; // Pass VarMap
                Some(model)
            }
            None => None,
        };

        Ok(FSRS { model, device }) // VarMap is stored within Model now
    }

    pub(crate) fn model(&self) -> Result<&Model> {
        self.model
            .as_ref()
            .ok_or_else(|| FSRSError::Internal("Model not initialized".to_string()))
    }

    pub(crate) fn device(&self) -> Device {
        self.device.clone()
    }
}

// parameters_to_model needs to be updated for candle
pub(crate) fn parameters_to_model(parameters: &Parameters, device: Device, varmap: VarMap) -> Result<Model> {
    let config = ModelConfig::default();
    // Model::new now takes VarMap.
    // The responsibility of initializing `w` is now within Model::new using VarBuilder from the VarMap.
    // However, `parameters_to_model` is about setting specific pre-defined weights.
    // This means we need a way to set the Var `w` in the model to these specific parameters.

    // Create the model instance. `w` will be initialized by Model::new.
    let mut model = Model::new(config.clone(), device.clone(), varmap)?;

    // Now, update the tensor `w` within the model with the clipped parameters.
    // This requires `w` to be mutable or a method to update it.
    // If `w` is a `Var`, we can use `set()`
    let clipped_params_vec = clip_parameters(parameters, config.num_relearning_steps);
    let clipped_params_tensor = Tensor::from_vec(clipped_params_vec, (21,), &device)?;
    model.w.set(&clipped_params_tensor)?; // Update the Var with new values

    Ok(model)
}

pub(crate) fn check_and_fill_parameters(parameters: &Parameters) -> Result<Vec<f32>> {
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
    // TODO: Update test_helpers to use candle types or adapt tests.
    // use crate::test_helpers::TestHelper; // Removed this import
    // use crate::test_helpers::{Model, Tensor};
    // use burn::tensor::{TensorData, Tolerance}; // Remove burn specific test items

    // Need to define a tolerance for float comparisons in candle
    const TEST_TOLERANCE: f32 = 1e-5;

    #[test]
    fn w() -> Result<()> {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let model = Model::new(ModelConfig::default(), device.clone(), varmap)?;
        let expected_w_vec = DEFAULT_PARAMETERS.to_vec();
        let model_w_vec = model.w.as_tensor().to_vec1::<f32>()?;

        assert_eq!(model_w_vec.len(), expected_w_vec.len());
        for (val_model, val_expected) in model_w_vec.iter().zip(expected_w_vec.iter()) {
            assert!((val_model - val_expected).abs() < f32::EPSILON);
        }
        Ok(())
    }

    #[test]
    fn convert_parameters() -> Result<()> {
        let fsrs4dot5_param = vec![
            0.4, 0.6, 2.4, 5.8, 4.93, 0.94, 0.86, 0.01, 1.49, 0.14, 0.94, 2.18, 0.05, 0.34, 1.26,
            0.29, 2.61,
        ];
        let fsrs5_param = check_and_fill_parameters(&fsrs4dot5_param)?;
        assert_eq!(
            fsrs5_param,
            vec![
                0.4, 0.6, 2.4, 5.8, 6.81, 0.44675013, 1.36, 0.01, 1.49, 0.14, 0.94, 2.18, 0.05,
                0.34, 1.26, 0.29, 2.61, 0.0, 0.0, 0.0, 0.5
            ]
        );
        Ok(())
    }

    // Helper function for comparing tensor data with tolerance
    fn assert_tensor_data_approx_eq(tensor: &Tensor, expected_data: &[f32], tolerance: f32) -> Result<()> {
        let data = tensor.to_vec1::<f32>()?;
        assert_eq!(data.len(), expected_data.len(), "Tensor dimensions mismatch");
        for (a, b) in data.iter().zip(expected_data) {
            assert!((a - b).abs() < tolerance, "Tensor data mismatch: {} vs {}", a, b);
        }
        Ok(())
    }

    #[test]
    fn power_forgetting_curve() -> Result<()> {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let model = Model::new(ModelConfig::default(), device.clone(), varmap)?;
        let delta_t = Tensor::from_slice(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0], (6,), &device)?;
        let stability = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 4.0, 2.0], (6,), &device)?;
        let retrievability = model.power_forgetting_curve(&delta_t, &stability)?;

        assert_tensor_data_approx_eq(
            &retrievability,
            &[
                1.0,
                0.9421982765197754,
                0.9268093109130859,
                0.91965252161026,
                0.9,
                0.8178008198738098,
            ],
            TEST_TOLERANCE,
        )?;
        Ok(())
    }

    #[test]
    fn init_stability() -> Result<()> {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let model = Model::new(ModelConfig::default(), device.clone(), varmap)?;
        let rating = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 1.0, 2.0], (6,), &device)?;
        let stability = model.init_stability(&rating)?;
        assert_tensor_data_approx_eq(
            &stability,
            &[
                DEFAULT_PARAMETERS[0],
                DEFAULT_PARAMETERS[1],
                DEFAULT_PARAMETERS[2],
                DEFAULT_PARAMETERS[3],
                DEFAULT_PARAMETERS[0],
                DEFAULT_PARAMETERS[1]
            ],
            TEST_TOLERANCE,
        )?;
        Ok(())
    }

    #[test]
    fn init_difficulty() -> Result<()> {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let model = Model::new(ModelConfig::default(), device.clone(), varmap)?;
        let rating = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 1.0, 2.0], (6,), &device)?;
        let difficulty = model.init_difficulty(&rating)?;
        assert_tensor_data_approx_eq(
            &difficulty,
            &[
                DEFAULT_PARAMETERS[4],
                DEFAULT_PARAMETERS[4] - DEFAULT_PARAMETERS[5].exp() + 1.0,
                DEFAULT_PARAMETERS[4] - (2.0 * DEFAULT_PARAMETERS[5]).exp() + 1.0,
                DEFAULT_PARAMETERS[4] - (3.0 * DEFAULT_PARAMETERS[5]).exp() + 1.0,
                DEFAULT_PARAMETERS[4],
                DEFAULT_PARAMETERS[4] - DEFAULT_PARAMETERS[5].exp() + 1.0,
            ],
            TEST_TOLERANCE,
        )?;
        Ok(())
    }

    #[test]
    fn forward() -> Result<()> {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let model = Model::new(ModelConfig::default(), device.clone(), varmap)?;
        let delta_ts = Tensor::from_slice(
            &[
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, // Flattened first row
                1.0, 1.0, 1.0, 1.0, 2.0, 2.0, // Flattened second row
            ],
            (2, 6), // Shape (seq_len, batch_size)
            &device,
        )?;
        let ratings = Tensor::from_slice(
            &[
                1.0, 2.0, 3.0, 4.0, 1.0, 2.0, // Flattened first row
                1.0, 2.0, 3.0, 4.0, 1.0, 2.0, // Flattened second row
            ],
            (2, 6), // Shape (seq_len, batch_size)
            &device,
        )?;
        let state = model.forward(&delta_ts, &ratings, None)?;

        assert_tensor_data_approx_eq(&state.stability, &[
            0.166_488_5,
            1.699_295_6,
            6.414_825_4,
            28.051_1,
            0.168_969_63,
            2.053_075_8,
        ], TEST_TOLERANCE)?;

        assert_tensor_data_approx_eq(&state.difficulty, &[
            8.362_965,
            7.086_328_5,
            4.868_057,
            1.0,
            8.362_965,
            7.086_328_5,
        ], TEST_TOLERANCE)?;
        Ok(())
    }

    #[test]
    fn next_difficulty() -> Result<()> {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let model = Model::new(ModelConfig::default(), device.clone(), varmap)?;
        let difficulty = Tensor::from_slice(&[5.0; 4], (4,), &device)?;
        let rating = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], (4,), &device)?;
        let next_difficulty_tensor = model.next_difficulty(&difficulty, &rating)?;
        // next_difficulty.clone().backward(); // candle-nn's Module trait doesn't have backward, handle gradients differently

        assert_tensor_data_approx_eq(&next_difficulty_tensor, &[7.329_555_5, 6.164_777_8, 5.0, 3.835_222_2], TEST_TOLERANCE)?;

        let mean_reverted_difficulty = model.mean_reversion(&next_difficulty_tensor)?;
        // mean_reverted_difficulty.clone().backward();

        assert_tensor_data_approx_eq(&mean_reverted_difficulty, &[7.296_110_6, 6.139_369_5, 4.982_629, 3.825_888], TEST_TOLERANCE)?;
        Ok(())
    }

    #[test]
    fn next_stability() -> Result<()> {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let model = Model::new(ModelConfig::default(), device.clone(), varmap)?;
        let stability = Tensor::from_slice(&[5.0; 4], (4,), &device)?;
        let difficulty = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], (4,), &device)?;
        let retrievability = Tensor::from_slice(&[0.9, 0.8, 0.7, 0.6], (4,), &device)?;
        let rating = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], (4,), &device)?;

        let s_recall = model.stability_after_success(
            &stability,
            &difficulty,
            &retrievability,
            &rating,
        )?;
        // s_recall.clone().backward();
        assert_tensor_data_approx_eq(&s_recall, &[25.578_495, 13.550_501, 59.868_79, 207.703_83], TEST_TOLERANCE)?;

        let s_forget = model.stability_after_failure(&stability, &difficulty, &retrievability)?;
        // s_forget.clone().backward();
        assert_tensor_data_approx_eq(&s_forget, &[1.746_929_3, 2.031_279_6, 2.440_167_7, 2.970_743_7], TEST_TOLERANCE)?;

        let next_stability_recall_forget = s_recall.where_cond(&rating.eq_lit(1.0)?, &s_forget, &s_recall)?;
        // next_stability_recall_forget.clone().backward();
        assert_tensor_data_approx_eq(&next_stability_recall_forget, &[1.746_929_3, 13.550_501, 59.868_79, 207.703_83], TEST_TOLERANCE)?;

        let next_stability_short_term = model.stability_short_term(&stability, &rating)?;
        assert_tensor_data_approx_eq(&next_stability_short_term, &[1.129_823_2, 2.400_462, 5.100_105_3, 10.835_862], TEST_TOLERANCE)?;
        Ok(())
    }

    #[test]
    fn fsrs() -> Result<()> {
        assert!(FSRS::new(Some(&[])).is_ok());
        assert!(FSRS::new(Some(&[1.])).is_err());
        assert!(FSRS::new(Some(DEFAULT_PARAMETERS.as_slice())).is_ok());
        assert!(FSRS::new(Some(&DEFAULT_PARAMETERS[..17])).is_ok());
        Ok(())
    }
}

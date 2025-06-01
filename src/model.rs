use crate::DEFAULT_PARAMETERS;
use crate::error::{FSRSError, Result};
use crate::inference::{FSRS5_DEFAULT_DECAY, Parameters, S_MAX, S_MIN};
use crate::parameter_clipper::clip_parameters;
use candle_core::{Device, Tensor};

use candle_core::IndexOp;
use candle_nn::VarMap;

#[derive(Clone)]
pub struct Model {
    pub w: candle_core::Var, // Changed from Tensor to Var
    device: Device,
    varmap: VarMap, // To store and manage variables
}

// Manual Debug implementation since VarMap doesn't implement Debug
impl std::fmt::Debug for Model {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Model")
            .field("w", &self.w)
            .field("device", &self.device)
            .field("varmap", &"VarMap")
            .finish()
    }
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
        let mut initial_params_vec: Vec<f64> = config
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

        // Create the tensor first
        let initial_w_tensor = Tensor::from_vec(initial_params_vec, (21,), &device)?;
        
        // Create a Var from the tensor
        let w = candle_core::Var::from_tensor(&initial_w_tensor)?;

        Ok(Self { w, device, varmap })
    }

    // Helper to access the tensor inside `w` (Var)
    fn w_tensor(&self) -> &Tensor {
        self.w.as_tensor()
    }

    // Helper to access the device
    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn power_forgetting_curve(&self, t: &Tensor, s: &Tensor) -> Result<Tensor> {
        // Ensure input tensors are f64 for consistency
        let t = t.to_dtype(candle_core::DType::F64)?;
        let s = s.to_dtype(candle_core::DType::F64)?;
        
        let w = self.w_tensor();
        let decay = (w.i(20)? * (-1.0))?;
        let ln_09 = 0.9f64.ln();
        let decay_inv = decay.powf(-1.0)?;
        let decay_inv_scalar = decay_inv.to_scalar::<f64>()?;
        
        // Create tensors that broadcast correctly with t and s, all as f64
        let ln_09_scalar = Tensor::from_slice(&[ln_09], (1,), &self.device)?;
        let decay_inv_scalar_tensor = Tensor::from_slice(&[decay_inv_scalar], (1,), &self.device)?;
        
        let ln_09_tensor = ln_09_scalar.broadcast_as(t.shape())?;
        let decay_inv_tensor = decay_inv_scalar_tensor.broadcast_as(t.shape())?;
        
        let factor = ((decay_inv_tensor * ln_09_tensor)?.exp()? - 1.0f64)?;
        let result = (((t / s)? * factor)? + 1.0f64)?;
        Ok(result.powf(decay_inv_scalar as f64)?)
    }

    pub fn next_interval(
        &self,
        stability: &Tensor,
        desired_retention: &Tensor,
    ) -> Result<Tensor> {
        let w = self.w_tensor();
        let decay = (w.i(20)? * (-1.0f64))?;
        let ln_09 = 0.9f64.ln();
        let decay_inv = decay.powf(-1.0f64)?;
        let decay_inv_scalar = decay_inv.to_scalar::<f64>()?;
        let ln_09_tensor = Tensor::from_slice(&[ln_09], (), &self.device)?;
        let decay_inv_tensor = Tensor::from_slice(&[decay_inv_scalar], (), &self.device)?;
        let factor = ((decay_inv_tensor.clone() * ln_09_tensor)?.exp()? - 1.0f64)?;
        let power_result = desired_retention.powf(decay_inv_scalar as f64)?;
        let power_minus_one = (power_result - 1.0f64)?;
        Ok(((stability / factor)? * power_minus_one)?)
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
        
        // Create condition tensors for hard penalty and easy bonus
        let rating_eq_2 = rating.eq(&Tensor::full(2.0, rating.shape(), &self.device)?)?;
        let w15 = w.i(15)?.broadcast_as(rating.shape())?;
        let hard_penalty = rating_eq_2.where_cond(&w15, &Tensor::ones_like(rating)?)?;
        
        let rating_eq_4 = rating.eq(&Tensor::full(4.0, rating.shape(), &self.device)?)?;
        let w16 = w.i(16)?.broadcast_as(rating.shape())?;
        let easy_bonus = rating_eq_4.where_cond(&w16, &Tensor::ones_like(rating)?)?;

        let w8_exp = w.i(8)?.exp()?;
        let last_d_term = ((last_d * (-1.0f64))? + 11.0f64)?;
        let w9_neg = w.i(9)?.neg()?;
        let w9_neg_scalar = w9_neg.to_scalar::<f64>()?;
        let last_s_power = last_s.powf(w9_neg_scalar as f64)?;
        let w10 = w.i(10)?.broadcast_as(r.shape())?;
        let r_term = (((r * (-1.0f64))? + 1.0f64)? * w10)?;
        let r_exp = (r_term.exp()? - 1.0f64)?;
        
        let w8_exp_broadcasted = w8_exp.broadcast_as(last_d_term.shape())?;
        let multiplier = ((w8_exp_broadcasted * &last_d_term)? * &last_s_power)?;
        let multiplier_penalty = (&multiplier * &hard_penalty)?;
        let bonus_product = (&multiplier_penalty * &easy_bonus)?;
        let final_multiplier = (&bonus_product + 1.0f64)?;
        Ok((last_s * &final_multiplier)?)
    }

    fn stability_after_failure(
        &self,
        last_s: &Tensor,
        last_d: &Tensor,
        r: &Tensor,
    ) -> Result<Tensor> {
        let w = self.w_tensor();
        let w11 = w.i(11)?.broadcast_as(last_d.shape())?;
        let w12_neg = w.i(12)?.neg()?;
        let w12_neg_scalar = w12_neg.to_scalar::<f64>()?;
        let last_d_power = last_d.powf(w12_neg_scalar as f64)?;
        
        let last_s_plus_1 = (last_s + 1.0f64)?;
        let w13 = w.i(13)?;
        let w13_scalar = w13.to_scalar::<f64>()?;
        let last_s_power = (last_s_plus_1.powf(w13_scalar as f64)? - 1.0f64)?;
        
        let w14 = w.i(14)?.broadcast_as(r.shape())?;
        let r_term = (((r * (-1.0f64))? + 1.0f64)? * w14)?;
        let r_exp = r_term.exp()?;
        
        let temp1 = (w11 * &last_d_power)?;
        let temp2 = (temp1 * &last_s_power)?;
        let new_s = (temp2 * &r_exp)?;
        
        let w17 = w.i(17)?;
        let w18 = w.i(18)?;
        let exp_term = (w17 * w18)?;
        let exp_term_broadcasted = exp_term.broadcast_as(last_s.shape())?;
        let new_s_min = (last_s / exp_term_broadcasted.exp()?)?;
        
        let condition = new_s_min.lt(&new_s)?;
        Ok(condition.where_cond(&new_s_min, &new_s)?)
    }

    fn stability_short_term(&self, last_s: &Tensor, rating: &Tensor) -> Result<Tensor> {
        let w = self.w_tensor();
        let w18 = w.i(18)?.broadcast_as(rating.shape())?;
        let rating_term = ((rating - 3.0f64)? + w18)?;
        let w17 = w.i(17)?.broadcast_as(rating_term.shape())?;
        let exp_term = (w17 * rating_term)?.exp()?;
        
        let w19_neg = w.i(19)?.neg()?;
        let w19_neg_scalar = w19_neg.to_scalar::<f64>()?;
        let last_s_power = last_s.powf(w19_neg_scalar as f64)?;
        
        let sinc = (exp_term * last_s_power)?;
        
        let rating_ge_3 = rating.ge(&Tensor::full(3.0, rating.shape(), &self.device)?)?;
        let sinc_min = sinc.minimum(&Tensor::ones_like(&sinc)?)?;
        let final_sinc = rating_ge_3.where_cond(&sinc_min, &sinc)?;
        
        Ok((last_s * final_sinc)?)
    }

    fn mean_reversion(&self, new_d: &Tensor) -> Result<Tensor> {
        let w = self.w_tensor();
        let rating = Tensor::from_slice(&[4.0], (), &self.device)?;
        let init_d = self.init_difficulty(&rating)?;
        let diff = (init_d - new_d)?;
        let w7 = w.i(7)?.broadcast_as(diff.shape())?;
        Ok(((w7 * diff)? + new_d)?)
    }

    pub(crate) fn init_stability(&self, rating: &Tensor) -> Result<Tensor> {
        let w = self.w_tensor();
        // Convert rating to indices (i64 for index_select). 
        // Subtract 1 from rating to get 0-based indices.
        let indices = (rating - 1.0f64)?.to_dtype(candle_core::DType::I64)?;
        // Select from first 4 elements of w (stability parameters)
        let w_stability = w.narrow(0, 0, 4)?;
        Ok(w_stability.index_select(&indices, 0)?)
    }

    fn init_difficulty(&self, rating: &Tensor) -> Result<Tensor> {
        let w = self.w_tensor();
        let rating_minus_1 = (rating - 1.0f64)?;
        let w5 = w.i(5)?.broadcast_as(rating_minus_1.shape())?;
        let exp_term = (w5 * rating_minus_1)?.exp()?;
        let w4 = w.i(4)?.broadcast_as(exp_term.shape())?;
        Ok(((w4 - exp_term)? + 1.0f64)?)
    }

    fn linear_damping(&self, delta_d: &Tensor, old_d: &Tensor) -> Result<Tensor> {
        let old_d_neg = old_d.neg()?;
        let term = (old_d_neg + 10.0f64)?;
        Ok(((term * delta_d)? / 9.0f64)?)
    }

    fn next_difficulty(&self, difficulty: &Tensor, rating: &Tensor) -> Result<Tensor> {
        let w = self.w_tensor();
        let rating_minus_3 = (rating - 3.0f64)?;
        let w6_neg = w.i(6)?.neg()?;
        let w6_neg_broadcasted = w6_neg.broadcast_as(rating_minus_3.shape())?;
        let delta_d = (w6_neg_broadcasted * rating_minus_3)?;
        let damping = self.linear_damping(&delta_d, difficulty)?;
        Ok((difficulty + damping)?)
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
            let rating_eq_1 = rating.eq(&Tensor::full(1.0, rating.shape(), &self.device)?)?;
            let mut new_stability = rating_eq_1.where_cond(&stability_after_failure, &stability_after_success)?;
            
            let delta_t_eq_0 = delta_t.eq(&Tensor::full(0.0, delta_t.shape(), &self.device)?)?;
            new_stability = delta_t_eq_0.where_cond(&stability_short_term, &new_stability)?;

            let mut new_difficulty = self.next_difficulty(&state.difficulty, rating)?;
            new_difficulty = self.mean_reversion(&new_difficulty)?.clamp(1.0, 10.0)?;
            
            // mask padding zeros for rating
            let rating_eq_0 = rating.eq(&Tensor::full(0.0, rating.shape(), &self.device)?)?;
            new_stability = rating_eq_0.where_cond(&state.stability, &new_stability)?;
            new_difficulty = rating_eq_0.where_cond(&state.difficulty, &new_difficulty)?;
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
        state.ok_or_else(|| FSRSError::Internal { message: "Forward resulted in None state".to_string() })
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
    pub initial_stability: Option<[f64; 4]>,
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
            .ok_or_else(|| FSRSError::Internal { message: "Model not initialized".to_string() })
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
    let model = Model::new(config.clone(), device.clone(), varmap)?;

    // Now, update the tensor `w` within the model with the clipped parameters.
    // This requires `w` to be mutable or a method to update it.
    // If `w` is a `Var`, we can use `set()`
    let clipped_params_vec = clip_parameters(parameters, config.num_relearning_steps);
    let clipped_params_tensor = Tensor::from_vec(clipped_params_vec, (21,), &device)?;
    model.w.set(&clipped_params_tensor)?; // Update the Var with new values

    Ok(model)
}

pub(crate) fn check_and_fill_parameters(parameters: &Parameters) -> Result<Vec<f64>> {
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
    const TEST_TOLERANCE: f64 = 1e-5;

    #[test]
    fn w() -> Result<()> {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let model = Model::new(ModelConfig::default(), device.clone(), varmap)?;
        let expected_w_vec = DEFAULT_PARAMETERS.to_vec();
        let model_w_vec = model.w.as_tensor().to_vec1::<f64>()?;

        assert_eq!(model_w_vec.len(), expected_w_vec.len());
        for (val_model, val_expected) in model_w_vec.iter().zip(expected_w_vec.iter()) {
            assert!((val_model - val_expected).abs() < f64::EPSILON);
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
                0.4, 0.6, 2.4, 5.8, 6.81, 0.4467501408728279, 1.3599999999999999, 0.01, 1.49, 0.14, 0.94, 2.18, 0.05, 0.34, 1.26, 0.29, 2.61, 0.0, 0.0, 0.0, 0.5
            ]
        );
        Ok(())
    }

    // Helper function for comparing tensor data with tolerance
    fn assert_tensor_data_approx_eq(tensor: &Tensor, expected_data: &[f64], tolerance: f64) -> Result<()> {
        let data = tensor.to_vec1::<f64>()?;
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

        let rating_is_one = rating.eq(&Tensor::ones_like(&rating)?)?;
        let next_stability_recall_forget = rating_is_one.where_cond(&s_forget, &s_recall)?;
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

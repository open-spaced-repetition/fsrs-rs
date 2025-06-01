use crate::batch_shuffle::{BatchTensorDataset, ShuffleDataLoader}; // Must be updated for candle
use crate::cosine_annealing::CosineAnnealingLR; // Assumed to be usable
use crate::dataset::{
    FSRSDataset, FSRSItem, WeightedFSRSItem, prepare_training_data, recency_weighted_fsrs_items,
    FSRSBatch, // Assuming FSRSBatch will be updated to hold candle::Tensor
};
use crate::error::Result;
use crate::model::{Model, ModelConfig}; // Already candle-based
use crate::parameter_clipper::parameter_clipper_candle; // Must be candle-compatible
use crate::pre_training::{pretrain, smooth_and_fill}; // Review for tensor ops
use crate::{DEFAULT_PARAMETERS, FSRS, FSRSError}; // FSRS is candle-based

// Candle imports
use candle_core::{Device, Tensor}; // Removed utils import as manual_seed doesn't exist
use candle_nn::{VarMap, ops}; // Removed unused Loss, loss_bce, binary_cross_entropy_with_logits
use candle_nn::{AdamW, Optimizer, ParamsAdamW}; // Use candle_nn optimizers

use core::marker::PhantomData;
use log::info;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

// Assuming this is still relevant and used as a slice for a Tensor
static PARAMS_STDDEV: [f32; 21] = [
    6.43, 9.66, 17.58, 27.85, 0.57, 0.28, 0.6, 0.12, 0.39, 0.18, 0.33, 0.3, 0.09, 0.16, 0.57, 0.25,
    1.03, 0.31, 0.32, 0.14, 0.27,
];

// BCELoss using candle
pub struct BCELossCandle;

// burn::nn::loss::Reduction equivalent for candle
#[derive(Clone, Copy, Debug)] // Added derive for use in TrainingConfig if needed & for logging
pub enum Reduction {
    Mean,
    Sum,
    // Auto, // Removed Auto as it's tricky and not directly supported by candle losses
}

impl BCELossCandle {
    pub const fn new() -> Self {
        Self
    }

    pub fn forward(
        &self,
        retrievability: &Tensor, // probability
        labels: &Tensor,
        weights: &Tensor,
        reduction: Reduction,
    ) -> Result<Tensor> {
        let epsilon = 1e-7f32;
        let retrievability = retrievability.clamp(epsilon, 1.0 - epsilon)?;

        let loss_val = ((labels * retrievability.log()?)?
            + ((labels.ones_like()? - labels)? * (retrievability.ones_like()? - retrievability)?.log()?)?)?
            * weights;

        Ok(match reduction {
            Reduction::Mean => loss_val?.mean_all()?.neg()?,
            Reduction::Sum => loss_val?.sum_all()?.neg()?,
        })
    }
}

// Model methods specific to training using candle
impl Model {
    pub fn forward_classification(
        &self,
        t_historys: &Tensor,
        r_historys: &Tensor,
        delta_ts: &Tensor,
        labels: &Tensor,     // Should be F32 for BCE
        weights: &Tensor,
        reduce: Reduction,
    ) -> Result<Tensor> {
        let state = self.forward(t_historys, r_historys, None)?;
        let retrievability = self.power_forgetting_curve(delta_ts, &state.stability)?;
        // Ensure labels are compatible with retrievability (e.g. F32)
        // If labels are Int, they need casting: `labels.to_dtype(DType::F32)?`
        BCELossCandle::new().forward(&retrievability, labels, weights, reduce)
    }

    pub(crate) fn l2_regularization(
        &self,
        init_w: &Tensor,
        params_stddev: &Tensor,
        gamma: f64,
        // Removed batch_size, total_size as they are not used in simplified candle version
    ) -> Result<Tensor> {
        let w_tensor = self.w.as_tensor();
        let diff = (w_tensor - init_w)?;
        let reg_loss = (diff.powf(2.0)?.div(&params_stddev.powf(2.0)?)?).sum_all()? * gamma;
        Ok(reg_loss?)
    }
}


#[derive(Debug, Default, Clone)]
pub struct ProgressState {
    pub epoch: usize,
    pub epoch_total: usize,
    pub items_processed: usize,
    pub items_total: usize,
}

#[derive(Debug, Default)]
pub struct CombinedProgressState {
    pub want_abort: bool,
    pub splits: Vec<ProgressState>,
    finished: bool,
}

#[derive(Clone, Default)]
pub struct TrainingInterrupter {
    stop_signal: Arc<Mutex<bool>>,
}

impl TrainingInterrupter {
    pub fn new() -> Self {
        Self {
            stop_signal: Arc::new(Mutex::new(false)),
        }
    }

    pub fn stop(&self) {
        *self.stop_signal.lock().unwrap() = true;
    }

    pub fn should_stop(&self) -> bool {
        *self.stop_signal.lock().unwrap()
    }
}


impl CombinedProgressState {
    pub fn new_shared() -> Arc<Mutex<Self>> {
        Default::default()
    }

    pub fn current(&self) -> usize {
        self.splits.iter().map(|s| s.current()).sum()
    }

    pub fn total(&self) -> usize {
        self.splits.iter().map(|s| s.total()).sum()
    }

    pub const fn finished(&self) -> bool {
        self.finished
    }
}

#[derive(Clone, Debug)]
pub struct TrainingProgress {
    pub items_processed: usize,
    pub items_total: usize,
    pub epoch: usize,
    pub epoch_total: usize,
    pub iteration: usize,
}


#[derive(Clone)]
pub struct ProgressCollector {
    pub state: Arc<Mutex<CombinedProgressState>>,
    pub interrupter: TrainingInterrupter,
    pub index: usize,
}

impl ProgressCollector {
    pub fn new(state: Arc<Mutex<CombinedProgressState>>, index: usize) -> Self {
        Self {
            state,
            interrupter: TrainingInterrupter::new(),
            index,
        }
    }
}

impl ProgressState {
    pub const fn current(&self) -> usize {
        self.epoch.saturating_sub(1) * self.items_total + self.items_processed
    }

    pub const fn total(&self) -> usize {
        self.epoch_total * self.items_total
    }
}

pub trait MetricsRenderer { // Simplified from burn
    fn render_train(&mut self, item: TrainingProgress);
}


impl MetricsRenderer for ProgressCollector {
    fn render_train(&mut self, item: TrainingProgress) {
        let mut info = self.state.lock().unwrap();
        let split = &mut info.splits[self.index];
        split.epoch = item.epoch;
        split.epoch_total = item.epoch_total;
        split.items_processed = item.items_processed;
        split.items_total = item.items_total;
        if info.want_abort {
            self.interrupter.stop();
        }
    }
}

#[derive(Clone, Debug)]
pub(crate) struct TrainingConfig {
    pub model: ModelConfig,
    pub num_epochs: usize,
    pub batch_size: usize,
    pub seed: u64,
    pub learning_rate: f64,
    pub max_seq_len: usize,
    pub gamma: f64,
    pub adam_epsilon: f64,
}

impl TrainingConfig {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        model: ModelConfig,
        num_epochs: usize,
        batch_size: usize,
        seed: u64,
        learning_rate: f64,
        max_seq_len: usize,
        gamma: f64,
        adam_epsilon: f64,
    ) -> Self {
        Self {
            model,
            num_epochs,
            batch_size,
            seed,
            learning_rate,
            max_seq_len,
            gamma,
            adam_epsilon,
        }
    }
}


pub fn calculate_average_recall(items: &[FSRSItem]) -> f32 {
    let (total_recall, total_reviews) = items
        .iter()
        .map(|item| item.current())
        .fold((0u32, 0u32), |(sum, count), review| {
            (sum + (review.rating > 1) as u32, count + 1)
        });

    if total_reviews == 0 {
        return 0.0;
    }
    total_recall as f32 / total_reviews as f32
}

#[derive(Clone, Debug)]
pub struct ComputeParametersInput {
    pub train_set: Vec<FSRSItem>,
    pub progress: Option<Arc<Mutex<CombinedProgressState>>>,
    pub enable_short_term: bool,
    pub num_relearning_steps: Option<usize>,
}

impl Default for ComputeParametersInput {
    fn default() -> Self {
        Self {
            train_set: Vec::new(),
            progress: None,
            enable_short_term: true,
            num_relearning_steps: None,
        }
    }
}

impl FSRS { // FSRS is now candle-based, no <B: Backend>
    pub fn compute_parameters(
        &self,
        ComputeParametersInput {
            train_set,
            progress,
            enable_short_term,
            num_relearning_steps,
            ..
        }: ComputeParametersInput,
    ) -> Result<Vec<f32>> {
        let finish_progress = || {
            if let Some(progress_arc) = &progress {
                progress_arc.lock().unwrap().finished = true;
            }
        };

        let (pre_train_set, train_set) = prepare_training_data(train_set);
        let average_recall = calculate_average_recall(&train_set);
        if train_set.len() < 8 {
            finish_progress();
            return Ok(DEFAULT_PARAMETERS.to_vec());
        }

        let (initial_stability, initial_rating_count) =
            pretrain(pre_train_set.clone(), average_recall).inspect_err(|_e| {
                finish_progress();
            })?;
        let pretrained_parameters: Vec<f32> = initial_stability // Assuming initial_stability is Vec<[f32;4]> or similar
            .iter().copied() // if it's Vec<f32> already from pretrain
            .chain(DEFAULT_PARAMETERS[4..].iter().copied())
            .collect();

        if train_set.len() == pre_train_set.len() || train_set.len() < 64 {
            finish_progress();
            return Ok(pretrained_parameters);
        }

        let config = TrainingConfig::new(
            ModelConfig {
                freeze_initial_stability: !enable_short_term,
                initial_stability: Some(initial_stability.clone()), // pretrain returns [f32;4]
                freeze_short_term_stability: !enable_short_term,
                num_relearning_steps: num_relearning_steps.unwrap_or(1),
            },
            5,    // num_epochs
            512,  // batch_size
            2023, // seed
            4e-2, // learning_rate
            64,   // max_seq_len
            1.0,  // gamma
            1e-8, // adam_epsilon
        );

        let mut weighted_train_set = recency_weighted_fsrs_items(train_set);
        weighted_train_set.retain(|item| item.item.reviews.len() <= config.max_seq_len);

        if let Some(progress_arc) = &progress {
            let progress_state = ProgressState {
                epoch_total: config.num_epochs,
                items_total: weighted_train_set.len(),
                epoch: 0,
                items_processed: 0,
            };
            progress_arc.lock().unwrap().splits = vec![progress_state];
        }

        let device = self.device(); // From FSRS struct
        let varmap = VarMap::new(); // Create VarMap for the model

        let trained_model = train(
            weighted_train_set.clone(),
            weighted_train_set,
            &config,
            device, // Pass candle device
            varmap,
            progress.clone().map(|p| ProgressCollector::new(p, 0)),
        );

        let optimized_parameters: Vec<f32> = trained_model
            .inspect_err(|_e| {
                finish_progress();
            })?
            .w // model.w is Var
            .as_tensor()
            .to_vec1()?;

        finish_progress();

        if optimized_parameters
            .iter()
            .any(|parameter: &f32| parameter.is_infinite())
        {
            return Err(FSRSError::InvalidInput);
        }

        let mut optimized_initial_stability_map: HashMap<u32, f32> = optimized_parameters[0..4]
            .iter()
            .enumerate()
            .map(|(i, &val)| (i as u32 + 1, val))
            .collect();
        let clamped_stability =
            smooth_and_fill(&mut optimized_initial_stability_map, &initial_rating_count).unwrap(); // Now using HashMap
        let final_optimized_parameters = clamped_stability
            .into_iter()
            .chain(optimized_parameters[4..].iter().copied())
            .collect();

        Ok(final_optimized_parameters)
    }

    pub fn benchmark(
        &self,
        ComputeParametersInput {
            train_set,
            enable_short_term,
            num_relearning_steps,
            ..
        }: ComputeParametersInput,
    ) -> Result<Vec<f32>> {
        let average_recall = calculate_average_recall(&train_set);
        let (pre_train_set, _next_train_set) = train_set
            .clone()
            .into_iter()
            .partition(|item| item.long_term_review_cnt() == 1);
        let (initial_stability, _initial_rating_count) = pretrain(pre_train_set, average_recall)?;

        let config = TrainingConfig::new(
            ModelConfig {
                freeze_initial_stability: !enable_short_term,
                initial_stability: Some(initial_stability.clone()),
                freeze_short_term_stability: !enable_short_term,
                num_relearning_steps: num_relearning_steps.unwrap_or(1),
            },
            5, 512, 2023, 4e-2, 64, 1.0, 1e-8,
        );

        let mut weighted_train_set = recency_weighted_fsrs_items(train_set);
        weighted_train_set.retain(|item| item.item.reviews.len() <= config.max_seq_len);

        let device = self.device();
        let varmap = VarMap::new();

        let trained_model = train(
            weighted_train_set.clone(),
            weighted_train_set,
            &config,
            device,
            varmap,
            None,
        )?;

        let parameters: Vec<f32> = trained_model.w.as_tensor().to_vec1()?;
        Ok(parameters)
    }
}

fn train(
    train_set_items: Vec<WeightedFSRSItem>,
    _test_set_items: Vec<WeightedFSRSItem>,  // Renamed, _ to indicate not used yet in candle version
    config: &TrainingConfig,
    device: Device,
    varmap: VarMap,
    progress: Option<ProgressCollector>,
) -> Result<Model> {

    // Note: manual_seed is not available in candle-core, so we'll skip setting seed for now
    // utils::manual_seed(config.seed); // Candle doesn't have manual_seed

    let total_size = train_set_items.len();
    let iterations = (total_size.saturating_sub(1) / config.batch_size + 1) * config.num_epochs;
    let mut lr_scheduler = CosineAnnealingLR::init(iterations as f64, config.learning_rate);

    // TODO: Adapt BatchTensorDataset and ShuffleDataLoader for candle.
    // These are major dependencies. The training loop below will be conceptual
    // until actual data loading provides candle::Tensor batches.
    // let train_dataset = FSRSDataset::from(train_set_items);
    // let batch_dataset_train = BatchTensorDataset::new(train_dataset, config.batch_size, device.clone());
    // let dataloader_train = ShuffleDataLoader::new(batch_dataset_train, config.seed);

    // let test_dataset = FSRSDataset::from(test_set_items.clone());
    // let batch_dataset_valid = BatchTensorDataset::new(test_dataset, config.batch_size, device.clone());
    // let dataloader_valid = ShuffleDataLoader::new(batch_dataset_valid, config.seed);

    let interrupter = TrainingInterrupter::new();
    let mut renderer: Box<dyn MetricsRenderer> = match progress {
        Some(mut progress_collector) => {
            progress_collector.interrupter = interrupter.clone();
            Box::new(progress_collector)
        }
        None => Box::new(NoProgress {}),
    };

    let mut model = Model::new(config.model.clone(), device.clone(), varmap.clone())?;
    let init_w_tensor = model.w.as_tensor().copy()?;
    let params_stddev_tensor = Tensor::from_slice(&PARAMS_STDDEV, (PARAMS_STDDEV.len(),), &device)?;

    let adam_params = ParamsAdamW {
        lr: config.learning_rate,
        beta1: 0.9,
        beta2: 0.999,
        eps: config.adam_epsilon,
        weight_decay: 0.01, // TODO: make this configurable if needed
    };
    // Optimizer is created with the model's trainable variables.
    let mut optim = AdamW::new(varmap.all_vars(), adam_params)?;


    let mut best_loss = f64::INFINITY;
    // Store the state of the best model's weights
    let mut best_model_w_state = model.w.as_tensor().copy()?;

    for epoch in 1..=config.num_epochs {
        let mut iteration = 0;
        // Conceptual loop over batches - replace with actual dataloader iteration
        // for item_batch_result in dataloader_train.iter() {
        for _batch_idx in 0..((total_size.saturating_sub(1) / config.batch_size) + 1) { // Mock loop
            iteration += 1;

            let current_lr = lr_scheduler.step();
            optim.set_learning_rate(current_lr);

            // Placeholder: Actual batch data (item) would come from the dataloader
            // let item = item_batch_result?; // This would be FSRSBatch with candle::Tensors

            // The following block is a placeholder for the actual training step logic
            // It needs `item` from the dataloader.
            /*
            // let _real_batch_size = item.delta_ts.dims()[0];

            let penalty = model.l2_regularization(
                &init_w_tensor,
                &params_stddev_tensor,
                config.gamma,
            )?;

            let loss = model.forward_classification(
                &item.t_historys,
                &item.r_historys,
                &item.delta_ts,
                &item.labels,    // Ensure labels are F32
                &item.weights,
                Reduction::Sum,
            )?;

            let total_loss = (&loss + &penalty)?;

            optim.zero_grad()?;
            total_loss.backward()?;

            // Parameter freezing by restoring parts of W after optimizer step
            let w_tensor_before_step = if config.model.freeze_initial_stability || config.model.freeze_short_term_stability {
                Some(model.w.as_tensor().copy()?)
            } else {
                None
            };

            optim.step()?;

            if let Some(w_prev) = w_tensor_before_step {
                // This is a simplified way to restore. For it to work correctly,
                // we need to ensure that we're modifying the Var's tensor data directly.
                let w_var_tensor = model.w.as_tensor();
                let mut w_current_data_vec = w_var_tensor.to_vec1::<f32>()?;
                let w_prev_data_vec = w_prev.to_vec1::<f32>()?;

                if config.model.freeze_initial_stability {
                    for i in 0..4 {
                        w_current_data_vec[i] = w_prev_data_vec[i];
                    }
                }
                if config.model.freeze_short_term_stability {
                    for i in 17..20 {
                        w_current_data_vec[i] = w_prev_data_vec[i];
                    }
                }
                // Create a new tensor from the modified vec and set it back to the Var
                let new_w_tensor = Tensor::from_vec(w_current_data_vec, w_var_tensor.shape(), &device)?;
                model.w.set(&new_w_tensor)?;
            }

            parameter_clipper_candle(&model.w, config.model.num_relearning_steps)?;
            */

            let items_processed_in_epoch = iteration * config.batch_size;
            renderer.render_train(TrainingProgress {
                items_processed: items_processed_in_epoch.min(total_size),
                items_total: total_size,
                epoch,
                epoch_total: config.num_epochs,
                iteration,
            });

            if interrupter.should_stop() {
                break;
            }
        }

        if interrupter.should_stop() {
            break;
        }

        // Validation loop (conceptual - needs dataloader_valid)
        let mut current_epoch_valid_loss = 0.0;
        let num_valid_batches = 0; // Count actual validation batches
        // for valid_item_batch_result in dataloader_valid.iter() {
            /*
            let valid_item = valid_item_batch_result?;
            // In Candle, usually no special "valid" model. Gradients are not computed if .backward() is not called.
            let penalty = model.l2_regularization(&init_w_tensor, &params_stddev_tensor, config.gamma)?;
            let loss = model.forward_classification(
                &valid_item.t_historys, &valid_item.r_historys, &valid_item.delta_ts,
                &valid_item.labels, &valid_item.weights, Reduction::Sum,
            )?;
            let total_loss_val = (loss + penalty)?.to_scalar::<f32>()?;
            current_epoch_valid_loss += f64::from(total_loss_val);
            num_valid_batches +=1;
            */
        // }
        // if num_valid_batches > 0 { current_epoch_valid_loss /= num_valid_batches as f64; } // Average per batch
        // else { current_epoch_valid_loss = f64::INFINITY; } // Or handle as 0 items validated


        info!("epoch: {:?} (simulated) validation_loss: {:?}", epoch, current_epoch_valid_loss);

        if current_epoch_valid_loss < best_loss {
            best_loss = current_epoch_valid_loss;
            best_model_w_state = model.w.as_tensor().copy()?;
        }
    }

    info!("best_loss (simulated): {:?}", best_loss);

    if interrupter.should_stop() {
        return Err(FSRSError::Interrupted);
    }

    // Set the model's weights to the best ones found
    model.w.set(&best_model_w_state)?;

    Ok(model)
}

struct NoProgress {}

impl MetricsRenderer for NoProgress {
    fn render_train(&mut self, _item: TrainingProgress) {}
}


#[cfg(test)]
mod tests {
    use std::fs::create_dir_all;
    use std::path::Path;
    use std::thread;
    use std::time::Duration;

    use super::*;
    use crate::convertor_tests::anki21_sample_file_converted_to_fsrs;
    use crate::convertor_tests::data_from_csv;
    use crate::dataset::FSRSBatch;
    use crate::test_helpers::TestHelper;
    use burn::backend::NdArray;
    use log::LevelFilter;

    #[test]
    fn test_calculate_average_recall() {
        let items = anki21_sample_file_converted_to_fsrs();
        let average_recall = calculate_average_recall(&items);
        assert_eq!(average_recall, 0.9435269);
    }

    #[test]
    fn test_loss_and_grad() {
        use burn::backend::ndarray::NdArrayDevice;
        use burn::tensor::TensorData;

        let config = ModelConfig::default();
        let device = NdArrayDevice::Cpu;
        type B = Autodiff<NdArray<f32>>;
        let mut model: Model<B> = config.init();
        let init_w = model.w.val();
        let params_stddev = Tensor::from_iter(PARAMS_STDDEV, &device);

        let item = FSRSBatch {
            t_historys: Tensor::from_iter(
                TensorData::from([
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                    [0.0, 1.0, 1.0, 3.0],
                    [1.0, 3.0, 3.0, 5.0],
                    [3.0, 6.0, 6.0, 12.0],
                ]),
                &device,
            ).expect("T historys is None"),
            r_historys: Tensor::from_iter(
                TensorData::from([
                    [1.0, 2.0, 3.0, 4.0],
                    [3.0, 4.0, 2.0, 4.0],
                    [1.0, 4.0, 4.0, 3.0],
                    [4.0, 3.0, 3.0, 3.0],
                    [3.0, 1.0, 3.0, 3.0],
                    [2.0, 3.0, 3.0, 4.0],
                ]),
                &device,
            ).expect("T historys is None"),
            delta_ts: Tensor::from_iter([4.0, 11.0, 12.0, 23.0], &device),
            labels: Tensor::from_iter([1, 1, 1, 0], &device),
            weights: Tensor::from_iter([1.0, 1.0, 1.0, 1.0], &device),
        };

        let loss = model.forward_classification(
            item.t_historys,
            item.r_historys,
            item.delta_ts,
            item.labels,
            item.weights,
            Reduction::Sum,
        );

        assert_eq!(loss.clone().into_scalar().to_f32(), 4.514678);
        let gradients = loss.expect("Loss is None").backward();

        let w_grad = model.w.grad(&gradients).unwrap();

        w_grad.to_data().to_vec::<f32>().unwrap().assert_approx_eq([
            -0.09797614,
            -0.0072790897,
            -0.0013130545,
            0.005998563,
            0.0407578,
            -0.059734516,
            0.030936655,
            -1.0551243,
            0.5905802,
            -3.1485205,
            0.5726496,
            -0.020666558,
            0.055198837,
            -0.1750127,
            -0.0013422092,
            -0.15273236,
            0.21408938,
            0.11237624,
            -0.005392518,
            -0.43270105,
            0.24273443,
        ]);

        let config =
            TrainingConfig::new(ModelConfig::default(), AdamConfig::new().with_epsilon(1e-8));
        let mut optim = config.optimizer.init::<B, Model<B>>();
        let lr = 0.04;
        let grads = GradientsParams::from_grads(gradients, &model);
        model = optim.step(lr, model, grads);
        model.w = parameter_clipper(model.w, config.model.num_relearning_steps);
        model
            .w
            .val()
            .to_data()
            .to_vec::<f32>()
            .unwrap()
            .assert_approx_eq([
                0.2572, 1.2170999, 3.3001997, 16.1107, 6.9714003, 0.61, 2.0566, 0.0469, 1.4861001,
                0.15200001, 0.97779995, 1.8889999, 0.07330001, 0.3527, 2.3333998, 0.2591, 2.9604,
                0.7136, 0.37319994, 0.1837, 0.16000001,
            ]);

        let penalty =
            model.l2_regularization(init_w.clone(), params_stddev.clone(), 512, 1000, 2.0);
        assert_eq!(penalty.clone().into_scalar().to_f32(), 0.67711174);

        let gradients = penalty.expect("Penalty is None").backward();
        let w_grad = model.w.grad(&gradients).unwrap();
        w_grad.to_data().to_vec::<f32>().unwrap().assert_approx_eq([
            0.0019813816,
            0.00087788026,
            0.00026506305,
            -0.00010561578,
            -0.25213888,
            1.0448985,
            -0.22755535,
            5.688889,
            -0.5385926,
            2.5283954,
            -0.75225013,
            0.9102214,
            -10.113578,
            3.1999993,
            0.2521374,
            1.3107198,
            -0.07721739,
            -0.85244584,
            0.79999864,
            4.179591,
            -1.1237309,
        ]);

        let item = FSRSBatch {
            t_historys: Tensor::from_iter(
                TensorData::from([
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                    [0.0, 1.0, 1.0, 3.0],
                    [1.0, 3.0, 3.0, 5.0],
                    [3.0, 6.0, 6.0, 12.0],
                ]),
                &device,
            ),
            r_historys: Tensor::from_iter(
                TensorData::from([
                    [1.0, 2.0, 3.0, 4.0],
                    [3.0, 4.0, 2.0, 4.0],
                    [1.0, 4.0, 4.0, 3.0],
                    [4.0, 3.0, 3.0, 3.0],
                    [3.0, 1.0, 3.0, 3.0],
                    [2.0, 3.0, 3.0, 4.0],
                ]),
                &device,
            ).expect("R historys is None"),
            delta_ts: Tensor::from_iter([4.0, 11.0, 12.0, 23.0], &device),
            labels: Tensor::from_iter([1, 1, 1, 0], &device),
            weights: Tensor::from_iter([1.0, 1.0, 1.0, 1.0], &device),
        };

        let loss = model.forward_classification(
            &item.t_historys,
            &item.r_historys,
            &item.delta_ts,
            &item.labels,
            &item.weights,
            Reduction::Sum,
        );
        assert_eq!(loss.clone().into_scalar().to_f32(), 4.2499204);
        let gradients = loss.expect("Loss is None").backward();
        let w_grad = model.w.grad(&gradients).unwrap();
        w_grad
            .clone()
            .into_data()
            .to_vec::<f32>()
            .unwrap()
            .assert_approx_eq([
                -0.05351858,
                -0.0059409104,
                -0.0011449483,
                0.005621137,
                0.021848494,
                0.023732044,
                0.021317776,
                -0.6712053,
                0.58890355,
                -2.8758395,
                0.60074204,
                -0.018340506,
                0.045839258,
                -0.14551935,
                -0.0013418762,
                -0.11314997,
                0.20784476,
                0.112954974,
                0.01292,
                -0.37279338,
                0.44497335,
            ]);
        let grads = GradientsParams::from_grads(gradients, &model);
        model = optim.step(lr, model, grads);
        model.w = parameter_clipper(model.w, config.model.num_relearning_steps);
        model
            .w
            .val()
            .to_data()
            .to_vec::<f32>()
            .unwrap()
            .assert_approx_eq([
                0.2949936,
                1.2566863,
                3.3399637,
                16.07079,
                6.9337125,
                0.62391204,
                2.017639,
                0.08549303,
                1.4461032,
                0.19186467,
                0.93776166,
                1.9288048,
                0.03366305,
                0.3923405,
                2.373399,
                0.29835668,
                2.9204354,
                0.6735949,
                0.35604826,
                0.22343501,
                0.121036425,
            ]);
    }

    #[test]
    fn training() {
        if std::env::var("SKIP_TRAINING").is_ok() {
            println!("Skipping test in CI");
            return;
        }

        let artifact_dir = std::env::var("BURN_LOG");

        if let Ok(artifact_dir) = artifact_dir {
            let _ = create_dir_all(&artifact_dir);
            let log_file = Path::new(&artifact_dir).join("training.log");
            fern::Dispatch::new()
                .format(|out, message, record| {
                    out.finish(format_args!(
                        "[{}][{}] {}",
                        record.target(),
                        record.level(),
                        message
                    ))
                })
                .level(LevelFilter::Info)
                .chain(fern::log_file(log_file).unwrap())
                .apply()
                .unwrap();
        }
        for items in [anki21_sample_file_converted_to_fsrs(), data_from_csv()] {
            for enable_short_term in [true, false] {
                let progress = CombinedProgressState::new_shared();
                let progress2 = Some(progress.clone());
                thread::spawn(move || {
                    let mut finished = false;
                    while !finished {
                        thread::sleep(Duration::from_millis(500));
                        let guard = progress.lock().unwrap();
                        finished = guard.finished();
                        println!("progress: {}/{}", guard.current(), guard.total());
                    }
                });

                let fsrs = FSRS::new(Some(&[])).unwrap();
                let parameters = fsrs
                    .compute_parameters(ComputeParametersInput {
                        train_set: items.clone(),
                        progress: progress2,
                        enable_short_term,
                        num_relearning_steps: None,
                    })
                    .unwrap();
                dbg!(&parameters);

                // evaluate
                let model = FSRS::new(Some(&parameters)).unwrap();
                let metrics = model.evaluate(items.clone(), |_| true).unwrap();
                dbg!(&metrics);
            }
        }
    }
}

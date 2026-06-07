use crate::analytical_gradients::forward_backward;
use crate::cosine_annealing::CosineAnnealingLR;
use crate::dataset::{
    FSRSItem, WeightedFSRSItem, prepare_training_data, prepare_training_data_with_card_ids,
    recency_weighted_fsrs_items, recency_weighted_fsrs_items_with_card_ids,
};
use crate::error::Result;
#[cfg(test)]
use crate::model::Model;
use crate::model::ModelConfig;
use crate::optimizer::{AdamConfig, AdamOptimizer};
use crate::parameter_clipper::clip_parameters_in_place;
use crate::parameter_initialization::{initialize_stability_parameters, smooth_and_fill};
use crate::{DEFAULT_PARAMETERS, FSRSError};
#[cfg(test)]
use burn::{nn::loss::Reduction, tensor::Int, tensor::Tensor, tensor::backend::Backend};
use log::info;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
#[cfg(feature = "parallel")]
use rayon::prelude::*;

use std::sync::{Arc, Mutex};

static PARAMS_STDDEV: [f32; 21] = [
    6.43, 9.66, 17.58, 27.85, 0.57, 0.28, 0.6, 0.12, 0.39, 0.18, 0.33, 0.3, 0.09, 0.16, 0.57, 0.25,
    1.03, 0.31, 0.32, 0.14, 0.27,
];

pub(crate) fn weighted_binary_cross_entropy(
    retrievability: &[f32],
    labels: &[f32],
    weights: &[f32],
) -> f32 {
    let mut loss = 0.0;
    let mut weight_sum = 0.0;
    for ((&r, &label), &weight) in retrievability.iter().zip(labels).zip(weights) {
        loss += (label * r.ln() + (1.0 - label) * (1.0 - r).ln()) * weight;
        weight_sum += weight;
    }
    -loss / weight_sum
}

#[cfg(test)]
impl<B: Backend> Model<B> {
    #[cfg(test)]
    pub fn forward_classification(
        &self,
        t_historys: Tensor<B, 2>,
        r_historys: Tensor<B, 2>,
        delta_ts: Tensor<B, 1>,
        labels: Tensor<B, 1, Int>,
        weights: Tensor<B, 1>,
        reduce: Reduction,
    ) -> Tensor<B, 1> {
        // info!("t_historys: {}", &t_historys);
        // info!("r_historys: {}", &r_historys);
        let state = self.forward(t_historys, r_historys, None);
        let retrievability = self.power_forgetting_curve(delta_ts, state.stability);
        let labels = labels.float();
        let loss = (labels.clone() * retrievability.clone().log()
            + (-labels + 1) * (-retrievability + 1).log())
            * weights.clone();
        match reduce {
            Reduction::Mean => loss.mean().neg(),
            Reduction::Sum => loss.sum().neg(),
            Reduction::Auto => (loss.sum() / weights.sum()).neg(),
        }
    }

    #[cfg(test)]
    pub(crate) fn l2_regularization(
        &self,
        init_w: Tensor<B, 1>,
        params_stddev: Tensor<B, 1>,
        batch_size: usize,
        total_size: usize,
        gamma: f64,
    ) -> Tensor<B, 1> {
        (self.w.val() - init_w)
            .powi_scalar(2)
            .div(params_stddev.powi_scalar(2))
            .sum()
            .mul_scalar(gamma * batch_size as f64 / total_size as f64)
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

impl CombinedProgressState {
    pub fn new_shared() -> Arc<Mutex<Self>> {
        Default::default()
    }

    pub(crate) fn reset(&mut self, splits: Vec<ProgressState>) {
        self.splits = splits;
        self.finished = false;
    }

    pub(crate) fn mark_finished(&mut self) {
        self.finished = true;
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

#[derive(Clone)]
pub struct ProgressCollector {
    pub state: Arc<Mutex<CombinedProgressState>>,
    /// The index of the split we should update.
    pub index: usize,
}

impl ProgressCollector {
    pub fn new(state: Arc<Mutex<CombinedProgressState>>, index: usize) -> Self {
        Self { state, index }
    }

    fn render_train(
        &mut self,
        epoch: usize,
        epoch_total: usize,
        items_processed: usize,
        items_total: usize,
    ) -> bool {
        let mut info = self.state.lock().unwrap();
        let split = &mut info.splits[self.index];
        split.epoch = epoch;
        split.epoch_total = epoch_total;
        split.items_processed = items_processed;
        split.items_total = items_total;
        !info.want_abort
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

/// Hyperparameters used when training FSRS parameters.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TrainingConfig {
    pub num_epochs: usize,
    pub batch_size: usize,
    pub seed: u64,
    pub learning_rate: f64,
    pub max_seq_len: usize,
    pub gamma: f64,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            num_epochs: 5,
            batch_size: 512,
            seed: 2023,
            learning_rate: 4e-2,
            max_seq_len: 256,
            gamma: 1.0,
        }
    }
}

fn validate_training_config(config: &TrainingConfig) -> Result<()> {
    if config.batch_size == 0 || !config.learning_rate.is_finite() || !config.gamma.is_finite() {
        return Err(FSRSError::InvalidInput);
    }
    Ok(())
}

fn benchmark_invalid_training_config() -> ! {
    panic!(
        "invalid training config: batch_size must be greater than 0, and learning_rate and gamma must be finite"
    );
}

pub(crate) fn calculate_average_recall(items: &[FSRSItem]) -> f32 {
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

/// Input parameters for computing FSRS parameters
#[derive(Clone, Debug)]
pub struct ComputeParametersInput {
    /// The training set containing review history
    pub train_set: Vec<FSRSItem>,
    /// Optional card ids aligned with `train_set`.
    ///
    /// When supplied, training groups the prefix items from the same card and
    /// computes the card recurrence once per batch column.
    pub card_ids: Option<Vec<i64>>,
    /// Optional progress tracking
    pub progress: Option<Arc<Mutex<CombinedProgressState>>>,
    /// Whether to enable short-term memory parameters
    pub enable_short_term: bool,
    /// Number of relearning steps
    pub num_relearning_steps: Option<usize>,
    /// Optional training hyperparameters
    pub training_config: Option<TrainingConfig>,
}

impl Default for ComputeParametersInput {
    fn default() -> Self {
        Self {
            train_set: Vec::new(),
            card_ids: None,
            progress: None,
            enable_short_term: true,
            num_relearning_steps: None,
            training_config: None,
        }
    }
}
/// Computes optimized parameters for the FSRS model based on training data.
///
/// This function trains the model on the provided dataset and returns optimized parameters.
///
/// # Arguments
/// * `input` - Input parameters including the training dataset and configuration
///
/// # Returns
/// A `Result<Vec<f32>>` containing the optimized parameters
pub fn compute_parameters(
    ComputeParametersInput {
        train_set,
        card_ids,
        progress,
        enable_short_term,
        num_relearning_steps,
        training_config,
        ..
    }: ComputeParametersInput,
) -> Result<Vec<f32>> {
    let finish_progress = || {
        if let Some(progress) = &progress {
            // The progress state at completion time may not indicate completion, because:
            // - If there were fewer than 512 entries, render_train() will have never been called
            // - One or more of the splits may have ignored later epochs, if accuracy went backwards
            // Because of this, we need a separate finished flag.
            progress.lock().unwrap().mark_finished();
        }
    };

    let training_config = training_config.unwrap_or_default();
    if let Err(error) = validate_training_config(&training_config) {
        finish_progress();
        return Err(error);
    }

    let original_train_set = train_set;
    if let Some(card_ids) = &card_ids
        && card_ids.len() != original_train_set.len()
    {
        finish_progress();
        return Err(FSRSError::InvalidInput);
    }
    if original_train_set.iter().any(|item| {
        item.reviews.is_empty()
            || item
                .reviews
                .iter()
                .any(|review| !(1..=4).contains(&review.rating))
    }) {
        finish_progress();
        return Err(FSRSError::InvalidInput);
    }

    let (dataset_for_initialization, train_set, train_card_ids) = match card_ids {
        Some(card_ids) => {
            let (dataset_for_initialization, train_set, train_card_ids) =
                prepare_training_data_with_card_ids(original_train_set, card_ids);
            (dataset_for_initialization, train_set, Some(train_card_ids))
        }
        None => {
            let (dataset_for_initialization, train_set) = prepare_training_data(original_train_set);
            (dataset_for_initialization, train_set, None)
        }
    };
    let average_recall = calculate_average_recall(&train_set);
    if train_set.len() < 8 {
        finish_progress();
        return Ok(DEFAULT_PARAMETERS.to_vec());
    }

    let (initial_stability, initial_rating_count) =
        initialize_stability_parameters(dataset_for_initialization.clone(), average_recall)
            .inspect_err(|_e| {
                finish_progress();
            })?;
    let initialized_parameters: Vec<f32> = initial_stability
        .into_iter()
        .chain(DEFAULT_PARAMETERS[4..].iter().copied())
        .collect();
    if train_set.len() == dataset_for_initialization.len() || train_set.len() < 64 {
        finish_progress();
        return Ok(initialized_parameters);
    }
    let model_config = ModelConfig {
        freeze_initial_stability: !enable_short_term,
        initial_stability: Some(initial_stability),
        freeze_short_term_stability: !enable_short_term,
        num_relearning_steps: num_relearning_steps.unwrap_or(1),
    };
    let training_initial_parameters = model_config.initial_parameters().to_vec();
    let mut weighted_train_set = match train_card_ids {
        Some(card_ids) => recency_weighted_fsrs_items_with_card_ids(train_set, card_ids),
        None => recency_weighted_fsrs_items(train_set),
    };
    weighted_train_set.retain(|item| item.item.reviews.len() <= training_config.max_seq_len);

    if let Some(progress) = &progress {
        let progress_state = ProgressState {
            epoch_total: training_config.num_epochs,
            items_total: weighted_train_set.len(),
            epoch: 0,
            items_processed: 0,
        };
        progress.lock().unwrap().reset(vec![progress_state]);
    }
    let optimized_parameters = train(
        weighted_train_set,
        &training_initial_parameters,
        &training_config,
        &model_config,
        progress.clone().map(|p| ProgressCollector::new(p, 0)),
    )
    .inspect_err(|_e| {
        finish_progress();
    })?;

    finish_progress();

    if optimized_parameters
        .iter()
        .any(|parameter: &f32| !parameter.is_finite())
    {
        return Err(FSRSError::InvalidInput);
    }

    let mut optimized_initial_stability = optimized_parameters[0..4]
        .iter()
        .enumerate()
        .map(|(i, &val)| (i as u32 + 1, val))
        .collect();
    let clamped_stability =
        smooth_and_fill(&mut optimized_initial_stability, &initial_rating_count).unwrap();
    let optimized_parameters = clamped_stability
        .into_iter()
        .chain(optimized_parameters[4..].iter().copied())
        .collect();

    Ok(optimized_parameters)
}

pub fn benchmark(
    ComputeParametersInput {
        train_set,
        card_ids,
        enable_short_term,
        num_relearning_steps,
        training_config,
        ..
    }: ComputeParametersInput,
) -> Vec<f32> {
    let training_config = training_config.unwrap_or_default();
    if validate_training_config(&training_config).is_err() {
        benchmark_invalid_training_config();
    }

    let average_recall = calculate_average_recall(&train_set);
    let (dataset_for_initialization, _next_train_set) = train_set
        .clone()
        .into_iter()
        .partition(|item| item.long_term_review_cnt() == 1);
    let initial_stability =
        initialize_stability_parameters(dataset_for_initialization, average_recall)
            .unwrap()
            .0;
    let model_config = ModelConfig {
        freeze_initial_stability: !enable_short_term,
        initial_stability: Some(initial_stability),
        freeze_short_term_stability: !enable_short_term,
        num_relearning_steps: num_relearning_steps.unwrap_or(1),
    };
    let initialized_parameters = model_config.initial_parameters().to_vec();
    let mut weighted_train_set = match card_ids {
        Some(card_ids) if card_ids.len() == train_set.len() => {
            recency_weighted_fsrs_items_with_card_ids(train_set, card_ids)
        }
        _ => recency_weighted_fsrs_items(train_set),
    };
    weighted_train_set.retain(|item| item.item.reviews.len() <= training_config.max_seq_len);
    train(
        weighted_train_set,
        &initialized_parameters,
        &training_config,
        &model_config,
        None,
    )
    .unwrap()
}

fn l2_penalty(
    parameters: &[f32],
    init_parameters: &[f32],
    batch_size: usize,
    total_size: usize,
    gamma: f64,
) -> f64 {
    parameters
        .iter()
        .zip(init_parameters)
        .zip(PARAMS_STDDEV)
        .map(|((&parameter, &init_parameter), stddev)| {
            let delta = (parameter - init_parameter) as f64;
            delta * delta / f64::from(stddev * stddev)
        })
        .sum::<f64>()
        * gamma
        * batch_size as f64
        / total_size as f64
}

fn add_l2_gradient(
    parameters: &[f32],
    init_parameters: &[f32],
    batch_size: usize,
    total_size: usize,
    gamma: f64,
    grad: &mut [f32],
) {
    let scale = (gamma * batch_size as f64 / total_size as f64) as f32;
    for (((slot, &parameter), &init_parameter), stddev) in grad
        .iter_mut()
        .zip(parameters)
        .zip(init_parameters)
        .zip(PARAMS_STDDEV)
    {
        let delta = parameter - init_parameter;
        *slot += 2.0 * delta / (stddev * stddev) * scale;
    }
}

fn render_progress(
    progress: &mut Option<ProgressCollector>,
    epoch: usize,
    epoch_total: usize,
    items_processed: usize,
    items_total: usize,
) -> bool {
    progress.as_mut().is_none_or(|progress| {
        progress.render_train(epoch, epoch_total, items_processed, items_total)
    })
}

fn train(
    mut train_set: Vec<WeightedFSRSItem>,
    initial_parameters: &[f32],
    training_config: &TrainingConfig,
    model_config: &ModelConfig,
    progress: Option<ProgressCollector>,
) -> Result<Vec<f32>> {
    let total_size = train_set.len();
    let batch_size = training_config.batch_size;
    let iterations = (total_size / batch_size + 1) * training_config.num_epochs;
    let mut lr_scheduler =
        CosineAnnealingLR::init(iterations as f64, training_config.learning_rate);
    let mut progress = progress;

    let mut parameters: [f32; 21] = initial_parameters.try_into().unwrap();
    let mut adam = AdamOptimizer::new(AdamConfig {
        epsilon: 1e-8,
        ..Default::default()
    });

    let mut best_loss = f64::INFINITY;
    let mut best_parameters = parameters;
    let mut rng = StdRng::seed_from_u64(training_config.seed);

    // Grouping by length to align with original PR 1 training iteration logic
    train_set.sort_by_cached_key(|weighted_item| weighted_item.item.reviews.len());
    let num_batches = total_size.div_ceil(batch_size);
    let mut batch_indices: Vec<usize> = (0..num_batches).collect();

    for epoch in 1..=training_config.num_epochs {
        batch_indices.shuffle(&mut rng);
        let mut items_processed = 0;

        for &batch_idx in &batch_indices {
            let lr = lr_scheduler.step();
            let mut grad = [0.0f32; 21];

            let start_idx = batch_idx * batch_size;
            let end_idx = std::cmp::min(start_idx + batch_size, total_size);
            let real_batch_size = end_idx - start_idx;

            #[cfg(feature = "parallel")]
            // Parallel accumulation over the batch using analytical_gradients
            let batch_grads = train_set[start_idx..end_idx]
                .par_iter()
                .map(|weighted_item| {
                    let item = &weighted_item.item;

                    let (delta_ts, ratings): (Vec<f32>, Vec<u32>) = item.reviews
                        [..item.reviews.len() - 1]
                        .iter()
                        .map(|r| (r.delta_t as f32, r.rating))
                        .unzip();

                    let final_review = item.reviews.last().unwrap();
                    let final_delta_t = final_review.delta_t as f32;
                    let label = if final_review.rating == 1 { 0.0 } else { 1.0 };

                    let (_, item_grads) = forward_backward(
                        &parameters,
                        &delta_ts,
                        &ratings,
                        final_delta_t,
                        label,
                        weighted_item.weight,
                    );
                    item_grads
                })
                .reduce(
                    || [0.0f32; 21],
                    |mut acc, item_grads| {
                        for i in 0..21 {
                            acc[i] += item_grads[i];
                        }
                        acc
                    },
                );

            #[cfg(not(feature = "parallel"))]
            let batch_grads = train_set[start_idx..end_idx]
                .iter()
                .map(|weighted_item| {
                    let item = &weighted_item.item;

                    let (delta_ts, ratings): (Vec<f32>, Vec<u32>) = item.reviews
                        [..item.reviews.len() - 1]
                        .iter()
                        .map(|r| (r.delta_t as f32, r.rating))
                        .unzip();

                    let final_review = item.reviews.last().unwrap();
                    let final_delta_t = final_review.delta_t as f32;
                    let label = if final_review.rating == 1 { 0.0 } else { 1.0 };

                    let (_, item_grads) = forward_backward(
                        &parameters,
                        &delta_ts,
                        &ratings,
                        final_delta_t,
                        label,
                        weighted_item.weight,
                    );
                    item_grads
                })
                .fold([0.0f32; 21], |mut acc, item_grads| {
                    for i in 0..21 {
                        acc[i] += item_grads[i];
                    }
                    acc
                });

            for i in 0..21 {
                grad[i] += batch_grads[i];
            }

            add_l2_gradient(
                &parameters,
                initial_parameters,
                real_batch_size,
                total_size,
                training_config.gamma,
                &mut grad,
            );

            if model_config.freeze_initial_stability {
                grad[..4].fill(0.0);
            }
            if model_config.freeze_short_term_stability {
                grad[17..20].fill(0.0);
            }

            // Optimizer step
            adam.step(lr as f32, &mut parameters, &grad);
            clip_parameters_in_place(
                &mut parameters,
                model_config.num_relearning_steps,
                !model_config.freeze_short_term_stability,
            );

            items_processed += real_batch_size;
            let keep_going = render_progress(
                &mut progress,
                epoch,
                training_config.num_epochs,
                items_processed.min(total_size),
                total_size,
            );

            if !keep_going {
                return Err(FSRSError::Interrupted);
            }
        }

        let mut loss_valid = 0.0;
        // Validation loop calculation over batches
        for chunk in train_set.chunks(batch_size) {
            #[cfg(feature = "parallel")]
            let chunk_loss: f64 = chunk
                .par_iter()
                .map(|weighted_item| {
                    let item = &weighted_item.item;
                    let (delta_ts, ratings): (Vec<f32>, Vec<u32>) = item.reviews
                        [..item.reviews.len() - 1]
                        .iter()
                        .map(|r| (r.delta_t as f32, r.rating))
                        .unzip();
                    let final_review = item.reviews.last().unwrap();
                    let label = if final_review.rating == 1 { 0.0 } else { 1.0 };

                    let (loss, _) = forward_backward(
                        &parameters,
                        &delta_ts,
                        &ratings,
                        final_review.delta_t as f32,
                        label,
                        weighted_item.weight,
                    );
                    loss as f64
                })
                .sum();

            #[cfg(not(feature = "parallel"))]
            let chunk_loss: f64 = chunk
                .iter()
                .map(|weighted_item| {
                    let item = &weighted_item.item;
                    let (delta_ts, ratings): (Vec<f32>, Vec<u32>) = item.reviews
                        [..item.reviews.len() - 1]
                        .iter()
                        .map(|r| (r.delta_t as f32, r.rating))
                        .unzip();
                    let final_review = item.reviews.last().unwrap();
                    let label = if final_review.rating == 1 { 0.0 } else { 1.0 };

                    let (loss, _) = forward_backward(
                        &parameters,
                        &delta_ts,
                        &ratings,
                        final_review.delta_t as f32,
                        label,
                        weighted_item.weight,
                    );
                    loss as f64
                })
                .sum();

            loss_valid += chunk_loss;

            loss_valid += l2_penalty(
                &parameters,
                initial_parameters,
                chunk.len(),
                total_size,
                training_config.gamma,
            );

            if progress
                .as_ref()
                .is_some_and(|p| p.state.lock().unwrap().want_abort)
            {
                return Err(FSRSError::Interrupted);
            }
        }

        loss_valid /= total_size as f64;
        info!("epoch: {:?} loss: {:?}", epoch, loss_valid);

        if loss_valid < best_loss {
            best_loss = loss_valid;
            best_parameters = parameters;
        }
    }

    info!("best_loss: {:?}", best_loss);
    Ok(best_parameters.to_vec())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::convertor_tests::anki21_sample_file_converted_to_fsrs;

    #[test]
    fn test_calculate_average_recall() {
        let items = anki21_sample_file_converted_to_fsrs();
        let average_recall = calculate_average_recall(&items);
        assert_eq!(average_recall, 0.9435269);
    }
}

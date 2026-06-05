use crate::analytic;
use crate::cosine_annealing::CosineAnnealingLR;
use crate::dataset::{
    FSRSItem, WeightedFSRSItem, prepare_training_data, prepare_training_data_with_card_ids,
    recency_weighted_fsrs_items, recency_weighted_fsrs_items_with_card_ids,
};
use crate::error::Result;
#[cfg(test)]
use crate::model::Model;
use crate::model::ModelConfig;
#[cfg(test)]
use crate::parameter_clipper::clip_parameters;
use crate::parameter_clipper::clip_parameters_in_place;
use crate::parameter_initialization::{initialize_stability_parameters, smooth_and_fill};
use crate::{DEFAULT_PARAMETERS, FSRSError};
#[cfg(test)]
use burn::{nn::loss::Reduction, tensor::Int, tensor::Tensor, tensor::backend::Backend};
use log::info;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;

use std::collections::BTreeMap;
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

#[derive(Debug, Clone)]
pub(crate) struct TrainingConfig {
    pub model: ModelConfig,
    pub num_epochs: usize,
    pub batch_size: usize,
    pub seed: u64,
    pub learning_rate: f64,
    pub max_seq_len: usize,
    pub gamma: f64,
}

impl TrainingConfig {
    fn new(model: ModelConfig) -> Self {
        Self {
            model,
            num_epochs: 5,
            batch_size: 512,
            seed: 2023,
            learning_rate: 4e-2,
            max_seq_len: 256,
            gamma: 1.0,
        }
    }
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
}

impl Default for ComputeParametersInput {
    fn default() -> Self {
        Self {
            train_set: Vec::new(),
            card_ids: None,
            progress: None,
            enable_short_term: true,
            num_relearning_steps: None,
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
    let config = TrainingConfig::new(ModelConfig {
        freeze_initial_stability: !enable_short_term,
        initial_stability: Some(initial_stability),
        freeze_short_term_stability: !enable_short_term,
        num_relearning_steps: num_relearning_steps.unwrap_or(1),
    });
    let training_initial_parameters = config.model.initial_parameters().to_vec();
    let mut weighted_train_set = match train_card_ids {
        Some(card_ids) => recency_weighted_fsrs_items_with_card_ids(train_set, card_ids),
        None => recency_weighted_fsrs_items(train_set),
    };
    weighted_train_set.retain(|item| item.item.reviews.len() <= config.max_seq_len);

    if let Some(progress) = &progress {
        let progress_state = ProgressState {
            epoch_total: config.num_epochs,
            items_total: weighted_train_set.len(),
            epoch: 0,
            items_processed: 0,
        };
        progress.lock().unwrap().reset(vec![progress_state]);
    }
    let optimized_parameters = train(
        weighted_train_set,
        &training_initial_parameters,
        &config,
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
        ..
    }: ComputeParametersInput,
) -> Vec<f32> {
    let average_recall = calculate_average_recall(&train_set);
    let (dataset_for_initialization, _next_train_set) = train_set
        .clone()
        .into_iter()
        .partition(|item| item.long_term_review_cnt() == 1);
    let initial_stability =
        initialize_stability_parameters(dataset_for_initialization, average_recall)
            .unwrap()
            .0;
    let mut config = TrainingConfig::new(ModelConfig {
        freeze_initial_stability: !enable_short_term,
        initial_stability: Some(initial_stability),
        freeze_short_term_stability: !enable_short_term,
        num_relearning_steps: num_relearning_steps.unwrap_or(1),
    });
    let initialized_parameters = config.model.initial_parameters().to_vec();
    // save RAM and speed up training
    config.max_seq_len = 64;
    let mut weighted_train_set = match card_ids {
        Some(card_ids) if card_ids.len() == train_set.len() => {
            recency_weighted_fsrs_items_with_card_ids(train_set, card_ids)
        }
        _ => recency_weighted_fsrs_items(train_set),
    };
    weighted_train_set.retain(|item| item.item.reviews.len() <= config.max_seq_len);
    train(weighted_train_set, &initialized_parameters, &config, None).unwrap()
}

#[derive(Clone)]
struct BatchHost {
    seq_len: usize,
    batch_size: usize,
    real_batch_size: usize,
    column_lengths: Vec<usize>,
    t_historys: Vec<f32>,
    r_historys: Vec<f32>,
    delta_ts: Vec<f32>,
    labels: Vec<f32>,
    weights: Vec<f32>,
    windowed: bool,
}

fn build_plain_batch(items: &[WeightedFSRSItem]) -> BatchHost {
    let batch_size = items.len();
    let seq_len = items
        .iter()
        .map(|item| item.item.reviews.len() - 1)
        .max()
        .expect("empty host batch");
    let mut t_historys = vec![0.0; seq_len * batch_size];
    let mut r_historys = vec![0.0; seq_len * batch_size];
    let mut delta_ts = Vec::with_capacity(batch_size);
    let mut labels = Vec::with_capacity(batch_size);
    let mut weights = Vec::with_capacity(batch_size);
    let mut column_lengths = Vec::with_capacity(batch_size);

    for (column, weighted_item) in items.iter().enumerate() {
        column_lengths.push(weighted_item.item.reviews.len() - 1);
        for (t, review) in weighted_item.item.history().enumerate() {
            let idx = t * batch_size + column;
            t_historys[idx] = review.delta_t as f32;
            r_historys[idx] = review.rating as f32;
        }
        let current = weighted_item.item.current();
        delta_ts.push(current.delta_t as f32);
        labels.push(f32::from(current.rating > 1));
        weights.push(weighted_item.weight);
    }

    BatchHost {
        seq_len,
        batch_size,
        real_batch_size: batch_size,
        column_lengths,
        t_historys,
        r_historys,
        delta_ts,
        labels,
        weights,
        windowed: false,
    }
}

fn build_windowed_batch(cards: &[Vec<WeightedFSRSItem>]) -> BatchHost {
    let batch_size = cards.len();
    let seq_len = cards
        .iter()
        .map(|card| card.last().expect("empty card group").item.reviews.len())
        .max()
        .expect("empty host batch");
    let real_batch_size = cards.iter().map(Vec::len).sum();
    let mut t_historys = vec![0.0; seq_len * batch_size];
    let mut r_historys = vec![0.0; seq_len * batch_size];
    let delta_ts = Vec::new();
    let mut labels = vec![0.0; seq_len * batch_size];
    let mut weights = vec![0.0; seq_len * batch_size];
    let mut column_lengths = Vec::with_capacity(batch_size);

    for (column, card) in cards.iter().enumerate() {
        let full_reviews = &card.last().expect("empty card group").item.reviews;
        column_lengths.push(full_reviews.len());
        for (t, review) in full_reviews.iter().enumerate() {
            let idx = t * batch_size + column;
            t_historys[idx] = review.delta_t as f32;
            r_historys[idx] = review.rating as f32;
        }
        for weighted_item in card {
            let current_idx = weighted_item.item.reviews.len() - 1;
            let current = weighted_item.item.current();
            let idx = current_idx * batch_size + column;
            labels[idx] = f32::from(current.rating > 1);
            weights[idx] = weighted_item.weight;
        }
    }

    BatchHost {
        seq_len,
        batch_size,
        real_batch_size,
        column_lengths,
        t_historys,
        r_historys,
        delta_ts,
        labels,
        weights,
        windowed: true,
    }
}

fn build_plain_batches(mut items: Vec<WeightedFSRSItem>, batch_size: usize) -> Vec<BatchHost> {
    items.sort_by_cached_key(|item| item.item.reviews.len());
    items
        .chunks(batch_size)
        .map(build_plain_batch)
        .collect::<Vec<_>>()
}

fn build_windowed_batches(items: Vec<WeightedFSRSItem>, batch_size: usize) -> Vec<BatchHost> {
    let mut grouped = BTreeMap::<i64, Vec<WeightedFSRSItem>>::new();
    for item in items {
        grouped.entry(item.card_id).or_default().push(item);
    }

    let mut cards = grouped
        .into_values()
        .map(|mut card| {
            card.sort_by_cached_key(|item| item.item.reviews.len());
            card
        })
        .collect::<Vec<_>>();
    cards.sort_by_cached_key(|card| card.last().unwrap().item.reviews.len());

    let mut batches = Vec::new();
    let mut current_cards = Vec::new();
    let mut current_predictions = 0;
    for card in cards {
        let predictions = card.len();
        if !current_cards.is_empty() && current_predictions + predictions > batch_size {
            batches.push(build_windowed_batch(&current_cards));
            current_cards.clear();
            current_predictions = 0;
        }
        current_predictions += predictions;
        current_cards.push(card);
    }
    if !current_cards.is_empty() {
        batches.push(build_windowed_batch(&current_cards));
    }
    batches
}

fn build_host_batches(items: Vec<WeightedFSRSItem>, batch_size: usize) -> Vec<BatchHost> {
    if items.iter().all(|item| item.card_id == -1) {
        build_plain_batches(items, batch_size)
    } else {
        build_windowed_batches(items, batch_size)
    }
}

fn batch_loss(batch: &BatchHost, parameters: &[f32]) -> f64 {
    if batch.windowed {
        analytic::card_loss(
            parameters,
            &batch.t_historys,
            &batch.r_historys,
            batch.seq_len,
            batch.batch_size,
            &batch.column_lengths,
            &batch.labels,
            &batch.weights,
        )
    } else {
        analytic::batch_loss(
            parameters,
            &batch.t_historys,
            &batch.r_historys,
            batch.seq_len,
            batch.batch_size,
            &batch.column_lengths,
            &batch.delta_ts,
            &batch.labels,
            &batch.weights,
        )
    }
}

fn batch_loss_and_grad(batch: &BatchHost, parameters: &[f32], grad: &mut [f64]) -> f64 {
    if batch.windowed {
        analytic::card_loss_and_grad(
            parameters,
            &batch.t_historys,
            &batch.r_historys,
            batch.seq_len,
            batch.batch_size,
            &batch.column_lengths,
            &batch.labels,
            &batch.weights,
            grad,
        )
    } else {
        analytic::batch_loss_and_grad(
            parameters,
            &batch.t_historys,
            &batch.r_historys,
            batch.seq_len,
            batch.batch_size,
            &batch.column_lengths,
            &batch.delta_ts,
            &batch.labels,
            &batch.weights,
            grad,
        )
    }
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
    grad: &mut [f64],
) {
    let scale = gamma * batch_size as f64 / total_size as f64;
    for (((slot, &parameter), &init_parameter), stddev) in grad
        .iter_mut()
        .zip(parameters)
        .zip(init_parameters)
        .zip(PARAMS_STDDEV)
    {
        let delta = f64::from(parameter - init_parameter);
        *slot += 2.0 * delta / f64::from(stddev * stddev) * scale;
    }
}

#[derive(Clone)]
struct HostAdam {
    m: [f64; 21],
    v: [f64; 21],
    t: i32,
}

impl HostAdam {
    const fn new() -> Self {
        Self {
            m: [0.0; 21],
            v: [0.0; 21],
            t: 0,
        }
    }

    fn step(&mut self, parameters: &mut [f32], grad: &[f64], lr: f64) {
        const BETA1: f64 = 0.9;
        const BETA2: f64 = 0.999;
        const EPSILON: f64 = 1e-8;
        self.t += 1;
        let bias1 = 1.0 - BETA1.powi(self.t);
        let bias2 = 1.0 - BETA2.powi(self.t);
        for i in 0..21 {
            self.m[i] = BETA1 * self.m[i] + (1.0 - BETA1) * grad[i];
            self.v[i] = BETA2 * self.v[i] + (1.0 - BETA2) * grad[i] * grad[i];
            let m_hat = self.m[i] / bias1;
            let v_hat = self.v[i] / bias2;
            parameters[i] -= (lr * m_hat / (v_hat.sqrt() + EPSILON)) as f32;
        }
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
    train_set: Vec<WeightedFSRSItem>,
    initial_parameters: &[f32],
    config: &TrainingConfig,
    progress: Option<ProgressCollector>,
) -> Result<Vec<f32>> {
    let total_size = train_set.len();
    let iterations = (total_size / config.batch_size + 1) * config.num_epochs;
    let train_batches = build_host_batches(train_set, config.batch_size);
    let mut lr_scheduler = CosineAnnealingLR::init(iterations as f64, config.learning_rate);
    let mut progress = progress;

    let mut parameters = initial_parameters.to_vec();
    let mut adam = HostAdam::new();
    let mut best_loss = f64::INFINITY;
    let mut best_parameters = parameters.clone();
    let mut rng = StdRng::seed_from_u64(config.seed);
    let mut batch_order = (0..train_batches.len()).collect::<Vec<_>>();
    for epoch in 1..=config.num_epochs {
        for (slot, idx) in batch_order.iter_mut().zip(0..) {
            *slot = idx;
        }
        batch_order.shuffle(&mut rng);
        let mut items_processed = 0;
        for batch_idx in batch_order.iter().copied() {
            let batch = &train_batches[batch_idx];
            let lr = lr_scheduler.step();
            let mut grad = [0.0; 21];
            batch_loss_and_grad(batch, &parameters, &mut grad);
            add_l2_gradient(
                &parameters,
                initial_parameters,
                batch.real_batch_size,
                total_size,
                config.gamma,
                &mut grad,
            );
            if config.model.freeze_initial_stability {
                grad[..4].fill(0.0);
            }
            if config.model.freeze_short_term_stability {
                grad[17..20].fill(0.0);
            }
            adam.step(&mut parameters, &grad, lr);
            clip_parameters_in_place(
                &mut parameters,
                config.model.num_relearning_steps,
                !config.model.freeze_short_term_stability,
            );
            items_processed += batch.real_batch_size;
            let keep_going = render_progress(
                &mut progress,
                epoch,
                config.num_epochs,
                items_processed.min(total_size),
                total_size,
            );
            if !keep_going {
                return Err(FSRSError::Interrupted);
            }
        }

        let mut loss_valid = 0.0;
        for batch in &train_batches {
            loss_valid += batch_loss(batch, &parameters)
                + l2_penalty(
                    &parameters,
                    initial_parameters,
                    batch.real_batch_size,
                    total_size,
                    config.gamma,
                );

            if progress
                .as_ref()
                .is_some_and(|progress| progress.state.lock().unwrap().want_abort)
            {
                return Err(FSRSError::Interrupted);
            }
        }
        loss_valid /= total_size as f64;
        info!("epoch: {:?} loss: {:?}", epoch, loss_valid);
        if loss_valid < best_loss {
            best_loss = loss_valid;
            best_parameters = parameters.clone();
        }
    }

    info!("best_loss: {:?}", best_loss);

    Ok(best_parameters)
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
    use crate::dataset::{FSRSBatch, FSRSReview};
    use crate::model::FSRS;
    use crate::parameter_clipper::parameter_clipper;
    use crate::test_helpers::TestHelper;
    use burn::backend::Autodiff;
    use burn::backend::NdArray;
    use burn::optim::{AdamConfig, GradientsParams, Optimizer};
    use burn::tensor::cast::ToElement;
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
        let params_stddev = Tensor::from_floats(PARAMS_STDDEV, &device);

        let item = FSRSBatch {
            t_historys: Tensor::from_floats(
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
            r_historys: Tensor::from_floats(
                TensorData::from([
                    [1.0, 2.0, 3.0, 4.0],
                    [3.0, 4.0, 2.0, 4.0],
                    [1.0, 4.0, 4.0, 3.0],
                    [4.0, 3.0, 3.0, 3.0],
                    [3.0, 1.0, 3.0, 3.0],
                    [2.0, 3.0, 3.0, 4.0],
                ]),
                &device,
            ),
            delta_ts: Tensor::from_floats([4.0, 11.0, 12.0, 23.0], &device),
            labels: Tensor::from_ints([1, 1, 1, 0], &device),
            weights: Tensor::from_floats([1.0, 1.0, 1.0, 1.0], &device),
        };

        let loss = model.forward_classification(
            item.t_historys,
            item.r_historys,
            item.delta_ts,
            item.labels,
            item.weights,
            Reduction::Sum,
        );

        assert_eq!(loss.clone().into_scalar().to_f32(), 4.0466027);
        let gradients = loss.backward();

        let w_grad = model.w.grad(&gradients).unwrap();

        w_grad.to_data().to_vec::<f32>().unwrap().assert_approx_eq([
            -0.095688485,
            -0.0051607806,
            -0.0012249565,
            0.007462064,
            0.03650761,
            -0.082112335,
            0.0593964,
            -2.1474836,
            0.57626534,
            -2.8751316,
            0.7154875,
            -0.028993709,
            0.0099172965,
            -0.2189217,
            -0.0017800558,
            -0.089381434,
            0.299141,
            0.068104014,
            -0.011605468,
            -0.25398168,
            0.27700496,
        ]);

        let config = TrainingConfig::new(ModelConfig::default());
        let mut optim = AdamConfig::new().with_epsilon(1e-8).init::<B, Model<B>>();
        let lr = 0.04;
        let grads = GradientsParams::from_grads(gradients, &model);
        model = optim.step(lr, model, grads);
        model.w = parameter_clipper(
            model.w,
            config.model.num_relearning_steps,
            !config.model.freeze_short_term_stability,
        );
        model
            .w
            .val()
            .to_data()
            .to_vec::<f32>()
            .unwrap()
            .assert_approx_eq([
                0.252,
                1.3331,
                2.3464994,
                8.2556,
                6.3733,
                0.87340003,
                2.9794,
                0.040999997,
                1.8322,
                0.20660001,
                0.756,
                1.5235,
                0.021400042,
                0.3029,
                1.6882998,
                0.64140004,
                1.8329,
                0.5025,
                0.13119997,
                0.1058,
                0.1142,
            ]);

        let penalty =
            model.l2_regularization(init_w.clone(), params_stddev.clone(), 512, 1000, 2.0);
        assert_eq!(penalty.clone().into_scalar().to_f32(), 0.67711145);

        let gradients = penalty.backward();
        let w_grad = model.w.grad(&gradients).unwrap();
        w_grad.to_data().to_vec::<f32>().unwrap().assert_approx_eq([
            0.0019813816,
            0.00087788026,
            0.00026506148,
            -0.000105618295,
            -0.25213888,
            1.0448985,
            -0.22755535,
            5.688889,
            -0.5385926,
            2.5283954,
            -0.75225013,
            0.9102214,
            -10.113569,
            3.1999993,
            0.2521374,
            1.3107208,
            -0.07721739,
            -0.85244584,
            0.79999936,
            4.1795917,
            -1.1237311,
        ]);

        let item = FSRSBatch {
            t_historys: Tensor::from_floats(
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
            r_historys: Tensor::from_floats(
                TensorData::from([
                    [1.0, 2.0, 3.0, 4.0],
                    [3.0, 4.0, 2.0, 4.0],
                    [1.0, 4.0, 4.0, 3.0],
                    [4.0, 3.0, 3.0, 3.0],
                    [3.0, 1.0, 3.0, 3.0],
                    [2.0, 3.0, 3.0, 4.0],
                ]),
                &device,
            ),
            delta_ts: Tensor::from_floats([4.0, 11.0, 12.0, 23.0], &device),
            labels: Tensor::from_ints([1, 1, 1, 0], &device),
            weights: Tensor::from_floats([1.0, 1.0, 1.0, 1.0], &device),
        };

        let loss = model.forward_classification(
            item.t_historys,
            item.r_historys,
            item.delta_ts,
            item.labels,
            item.weights,
            Reduction::Sum,
        );
        assert_eq!(loss.clone().into_scalar().to_f32(), 3.767796);
        let gradients = loss.backward();
        let w_grad = model.w.grad(&gradients).unwrap();
        w_grad
            .clone()
            .into_data()
            .to_vec::<f32>()
            .unwrap()
            .assert_approx_eq([
                -0.040530164,
                -0.0041278866,
                -0.0010157757,
                0.007239434,
                0.009321215,
                -0.120117955,
                0.039143264,
                -0.8628009,
                0.5794302,
                -2.5713828,
                0.7669307,
                -0.024242667,
                0.0,
                -0.16912507,
                -0.0017008218,
                -0.061857328,
                0.28093633,
                0.064058185,
                0.0063592787,
                -0.1903223,
                0.6257775,
            ]);
        let grads = GradientsParams::from_grads(gradients, &model);
        model = optim.step(lr, model, grads);
        model.w = parameter_clipper(
            model.w,
            config.model.num_relearning_steps,
            !config.model.freeze_short_term_stability,
        );
        model
            .w
            .val()
            .to_data()
            .to_vec::<f32>()
            .unwrap()
            .assert_approx_eq([
                0.2882918, 1.3726242, 2.3861322, 8.215636, 6.339965, 0.9130969, 2.940639,
                0.07696985, 1.7921946, 0.2464217, 0.71595186, 1.5631561, 0.001, 0.34230903,
                1.7282416, 0.68038, 1.7929853, 0.46258268, 0.14039303, 0.14509967, 0.1,
            ]);
    }

    #[test]
    fn test_analytic_loss_and_grad_matches_burn() {
        use burn::backend::ndarray::NdArrayDevice;
        use burn::tensor::TensorData;

        let config = ModelConfig::default();
        let device = NdArrayDevice::Cpu;
        type B = Autodiff<NdArray<f32>>;
        let model: Model<B> = config.init();

        let t_historys = [
            [0.0f32, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 1.0, 3.0],
            [1.0, 3.0, 3.0, 5.0],
            [3.0, 6.0, 6.0, 12.0],
        ];
        let r_historys = [
            [1.0f32, 2.0, 3.0, 4.0],
            [3.0, 4.0, 2.0, 4.0],
            [1.0, 4.0, 4.0, 3.0],
            [4.0, 3.0, 3.0, 3.0],
            [3.0, 1.0, 3.0, 3.0],
            [2.0, 3.0, 3.0, 4.0],
        ];
        let delta_ts = [4.0f32, 11.0, 12.0, 23.0];
        let labels_int = [1, 1, 1, 0];
        let labels = [1.0f32, 1.0, 1.0, 0.0];
        let weights = [1.0f32, 1.0, 1.0, 1.0];
        let item = FSRSBatch {
            t_historys: Tensor::from_floats(TensorData::from(t_historys), &device),
            r_historys: Tensor::from_floats(TensorData::from(r_historys), &device),
            delta_ts: Tensor::from_floats(delta_ts, &device),
            labels: Tensor::from_ints(labels_int, &device),
            weights: Tensor::from_floats(weights, &device),
        };

        let loss = model.forward_classification(
            item.t_historys,
            item.r_historys,
            item.delta_ts,
            item.labels,
            item.weights,
            Reduction::Sum,
        );
        let burn_loss = loss.clone().into_scalar().to_f64();
        let gradients = loss.backward();
        let burn_grad = model
            .w
            .grad(&gradients)
            .unwrap()
            .to_data()
            .to_vec::<f32>()
            .unwrap();

        let t_flat = t_historys.iter().flatten().copied().collect::<Vec<_>>();
        let r_flat = r_historys.iter().flatten().copied().collect::<Vec<_>>();
        let seq_lens = vec![6; 4];
        let mut analytic_grad = [0.0; 21];
        let analytic_loss = crate::analytic::batch_loss_and_grad(
            &DEFAULT_PARAMETERS,
            &t_flat,
            &r_flat,
            6,
            4,
            &seq_lens,
            &delta_ts,
            &labels,
            &weights,
            &mut analytic_grad,
        );

        assert!(
            (burn_loss - analytic_loss).abs() < 1e-5,
            "burn loss {burn_loss}, analytic loss {analytic_loss}"
        );
        for i in 0..21 {
            let burn = burn_grad[i] as f64;
            let analytic = analytic_grad[i];
            let diff = (burn - analytic).abs();
            let scale = burn.abs().max(analytic.abs()).max(1.0);
            assert!(
                diff <= 2e-3 || diff / scale <= 2e-3,
                "param {i}: burn={burn}, analytic={analytic}, diff={diff}"
            );
        }
    }

    #[test]
    fn test_host_adam_matches_burn_adam_steps() {
        let mut parameters = DEFAULT_PARAMETERS.to_vec();
        let mut adam = HostAdam::new();
        let grad1 = [
            -0.095688485,
            -0.0051607806,
            -0.0012249565,
            0.007462064,
            0.03650761,
            -0.082112335,
            0.0593964,
            -2.1474836,
            0.57626534,
            -2.8751316,
            0.7154875,
            -0.028993709,
            0.0099172965,
            -0.2189217,
            -0.0017800558,
            -0.089381434,
            0.299141,
            0.068104014,
            -0.011605468,
            -0.25398168,
            0.27700496,
        ];
        adam.step(&mut parameters, &grad1, 0.04);
        parameters = clip_parameters(&parameters, 1, true);
        parameters.assert_approx_eq([
            0.252,
            1.3331,
            2.3464994,
            8.2556,
            6.3733,
            0.87340003,
            2.9794,
            0.040999997,
            1.8322,
            0.20660001,
            0.756,
            1.5235,
            0.021400042,
            0.3029,
            1.6882998,
            0.64140004,
            1.8329,
            0.5025,
            0.13119997,
            0.1058,
            0.1142,
        ]);

        let grad2 = [
            -0.040530164,
            -0.0041278866,
            -0.0010157757,
            0.007239434,
            0.009321215,
            -0.120117955,
            0.039143264,
            -0.8628009,
            0.5794302,
            -2.5713828,
            0.7669307,
            -0.024242667,
            0.0,
            -0.16912507,
            -0.0017008218,
            -0.061857328,
            0.28093633,
            0.064058185,
            0.0063592787,
            -0.1903223,
            0.6257775,
        ];
        adam.step(&mut parameters, &grad2, 0.04);
        parameters = clip_parameters(&parameters, 1, true);
        parameters.assert_approx_eq([
            0.2882918, 1.3726242, 2.3861322, 8.215636, 6.339965, 0.9130969, 2.940639, 0.07696985,
            1.7921946, 0.2464217, 0.71595186, 1.5631561, 0.001, 0.34230903, 1.7282416, 0.68038,
            1.7929853, 0.46258268, 0.14039303, 0.14509967, 0.1,
        ]);
    }

    #[test]
    fn test_windowed_host_batch_matches_plain_prefixes() {
        let reviews = [
            FSRSReview {
                rating: 4,
                delta_t: 0,
            },
            FSRSReview {
                rating: 3,
                delta_t: 2,
            },
            FSRSReview {
                rating: 1,
                delta_t: 5,
            },
            FSRSReview {
                rating: 3,
                delta_t: 8,
            },
        ];
        let weighted = (2..=reviews.len())
            .enumerate()
            .map(|(idx, len)| WeightedFSRSItem {
                weight: 0.5 + idx as f32 * 0.25,
                card_id: 42,
                item: FSRSItem {
                    reviews: reviews[..len].to_vec(),
                },
            })
            .collect::<Vec<_>>();

        let mut plain_weighted = weighted.clone();
        for item in &mut plain_weighted {
            item.card_id = -1;
        }
        let plain = build_host_batches(plain_weighted, 32);
        let windowed = build_host_batches(weighted, 32);
        assert_eq!(plain.len(), 1);
        assert_eq!(windowed.len(), 1);

        let mut plain_grad = [0.0; 21];
        let mut windowed_grad = [0.0; 21];
        let plain_loss = batch_loss_and_grad(&plain[0], &DEFAULT_PARAMETERS, &mut plain_grad);
        let windowed_loss =
            batch_loss_and_grad(&windowed[0], &DEFAULT_PARAMETERS, &mut windowed_grad);
        assert!((plain_loss - windowed_loss).abs() < 1e-9);
        for i in 0..21 {
            assert!(
                (plain_grad[i] - windowed_grad[i]).abs() < 1e-9,
                "param {i}: plain={} windowed={}",
                plain_grad[i],
                windowed_grad[i]
            );
        }
    }

    fn synthetic_card_id_training_data() -> (Vec<FSRSItem>, Vec<i64>) {
        let mut items = Vec::new();
        let mut card_ids = Vec::new();

        for card_idx in 0..32 {
            let card_id = 10_000 + card_idx as i64;
            let reviews = [
                FSRSReview {
                    rating: 3,
                    delta_t: 0,
                },
                FSRSReview {
                    rating: 3,
                    delta_t: 2,
                },
                FSRSReview {
                    rating: if card_idx % 6 == 0 {
                        1
                    } else if card_idx % 4 == 0 {
                        4
                    } else {
                        3
                    },
                    delta_t: 3 + card_idx % 7,
                },
                FSRSReview {
                    rating: if card_idx % 5 == 0 {
                        2
                    } else if card_idx % 3 == 0 {
                        1
                    } else {
                        4
                    },
                    delta_t: 1 + (card_idx * 3) % 11,
                },
                FSRSReview {
                    rating: if card_idx % 7 == 0 { 1 } else { 3 },
                    delta_t: 2 + (card_idx * 5) % 17,
                },
            ];

            for len in 2..=reviews.len() {
                items.push(FSRSItem {
                    reviews: reviews[..len].to_vec(),
                });
                card_ids.push(card_id);
            }
        }

        (items, card_ids)
    }

    #[test]
    fn test_windowed_host_batches_match_plain_prefixes_across_batches() {
        let (items, card_ids) = synthetic_card_id_training_data();
        let plain_batches = build_host_batches(recency_weighted_fsrs_items(items.clone()), 17);
        let windowed_batches = build_host_batches(
            recency_weighted_fsrs_items_with_card_ids(items, card_ids),
            17,
        );

        assert!(plain_batches.len() > 1);
        assert!(windowed_batches.len() > 1);

        let mut plain_grad = [0.0; 21];
        let mut plain_loss = 0.0;
        for batch in &plain_batches {
            plain_loss += batch_loss_and_grad(batch, &DEFAULT_PARAMETERS, &mut plain_grad);
        }

        let mut windowed_grad = [0.0; 21];
        let mut windowed_loss = 0.0;
        for batch in &windowed_batches {
            windowed_loss += batch_loss_and_grad(batch, &DEFAULT_PARAMETERS, &mut windowed_grad);
        }

        assert!((plain_loss - windowed_loss).abs() < 1e-9);
        for i in 0..21 {
            assert!(
                (plain_grad[i] - windowed_grad[i]).abs() < 1e-9,
                "param {i}: plain={} windowed={}",
                plain_grad[i],
                windowed_grad[i]
            );
        }
    }

    #[test]
    fn test_compute_parameters_with_card_ids_matches_without_card_ids() {
        let (items, card_ids) = synthetic_card_id_training_data();
        let with_card_ids = compute_parameters(ComputeParametersInput {
            train_set: items.clone(),
            card_ids: Some(card_ids),
            progress: None,
            enable_short_term: true,
            num_relearning_steps: None,
        })
        .unwrap();
        let without_card_ids = compute_parameters(ComputeParametersInput {
            train_set: items.clone(),
            card_ids: None,
            progress: None,
            enable_short_term: true,
            num_relearning_steps: None,
        })
        .unwrap();

        assert_eq!(with_card_ids.len(), 21);
        assert_eq!(without_card_ids.len(), 21);
        assert!(with_card_ids.iter().all(|parameter| parameter.is_finite()));
        assert!(
            without_card_ids
                .iter()
                .all(|parameter| parameter.is_finite())
        );

        let max_parameter_diff = with_card_ids
            .iter()
            .zip(&without_card_ids)
            .map(|(&with, &without)| (with - without).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_parameter_diff <= 1e-3,
            "max parameter diff {max_parameter_diff}"
        );

        let with_metrics = FSRS::new(&with_card_ids)
            .unwrap()
            .evaluate(items.clone(), |_| true)
            .unwrap();
        let without_metrics = FSRS::new(&without_card_ids)
            .unwrap()
            .evaluate(items, |_| true)
            .unwrap();

        assert!(
            (with_metrics.log_loss - without_metrics.log_loss).abs() <= 1e-4,
            "log_loss with_card_ids={} without_card_ids={}",
            with_metrics.log_loss,
            without_metrics.log_loss
        );
        assert!(
            (with_metrics.rmse_bins - without_metrics.rmse_bins).abs() <= 1e-4,
            "rmse_bins with_card_ids={} without_card_ids={}",
            with_metrics.rmse_bins,
            without_metrics.rmse_bins
        );
    }

    fn disabled_short_term_regression_items() -> Vec<FSRSItem> {
        let initialization_items = (0..30).map(|idx| FSRSItem {
            reviews: vec![
                FSRSReview {
                    rating: 3,
                    delta_t: 0,
                },
                FSRSReview {
                    rating: if idx % 7 == 0 { 1 } else { 3 },
                    delta_t: 2,
                },
            ],
        });
        let training_items = (0..100).map(|idx| FSRSItem {
            reviews: vec![
                FSRSReview {
                    rating: 3,
                    delta_t: 0,
                },
                FSRSReview {
                    rating: if idx % 5 == 0 { 1 } else { 3 },
                    delta_t: 2,
                },
                FSRSReview {
                    rating: if idx % 6 == 0 { 1 } else { 4 },
                    delta_t: idx * 3 % 14 + 1,
                },
            ],
        });

        initialization_items.chain(training_items).collect()
    }

    #[test]
    fn test_disabled_short_term_benchmark_zeroes_short_term_parameters() {
        let parameters = benchmark(ComputeParametersInput {
            train_set: disabled_short_term_regression_items(),
            enable_short_term: false,
            ..Default::default()
        });

        assert_eq!(&parameters[17..20], &[0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_disabled_short_term_compute_parameters_zeroes_short_term_parameters() {
        let parameters = compute_parameters(ComputeParametersInput {
            train_set: disabled_short_term_regression_items(),
            enable_short_term: false,
            ..Default::default()
        })
        .unwrap();

        assert_eq!(&parameters[17..20], &[0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_compute_parameters_rejects_mismatched_card_ids() {
        let item = FSRSItem {
            reviews: vec![
                FSRSReview {
                    rating: 4,
                    delta_t: 0,
                },
                FSRSReview {
                    rating: 3,
                    delta_t: 2,
                },
            ],
        };
        let result = compute_parameters(ComputeParametersInput {
            train_set: vec![item],
            card_ids: Some(vec![]),
            ..Default::default()
        });
        assert!(matches!(result, Err(FSRSError::InvalidInput)));
    }

    #[test]
    fn test_compute_parameters_rejects_invalid_items() {
        let empty_item = FSRSItem { reviews: vec![] };
        let invalid_rating_item = FSRSItem {
            reviews: vec![FSRSReview {
                rating: 5,
                delta_t: 0,
            }],
        };

        for item in [empty_item, invalid_rating_item] {
            let result = compute_parameters(ComputeParametersInput {
                train_set: vec![item],
                ..Default::default()
            });
            assert!(matches!(result, Err(FSRSError::InvalidInput)));
        }
    }

    #[test]
    fn test_training() {
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

                let parameters = compute_parameters(ComputeParametersInput {
                    train_set: items.clone(),
                    card_ids: None,
                    progress: progress2,
                    enable_short_term,
                    num_relearning_steps: None,
                })
                .unwrap();
                dbg!(&parameters);

                // evaluate
                let model = FSRS::new(&parameters).unwrap();
                let metrics = model.evaluate(items.clone(), |_| true).unwrap();
                dbg!(&metrics);
            }
        }
    }
}

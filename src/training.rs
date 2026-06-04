use crate::batch_shuffle::BatchTensorDataset;
use crate::cosine_annealing::CosineAnnealingLR;
use crate::dataset::{
    FSRSItem, WeightedFSRSItem, filter_outlier_indices, prepare_training_data,
    recency_weighted_fsrs_items, sort_items_by_review_length,
};
use crate::error::Result;
use crate::model::{
    Get, MemoryStateTensors, Model, ModelConfig, ModelVersion, parameters_to_model,
};
use crate::parameter_clipper::parameter_clipper;
use crate::parameter_initialization::{initialize_stability_parameters, smooth_and_fill};
use crate::parameter_initialization_fsrs7::{
    initialize_parameters_fsrs7, smooth_initial_stabilities_fsrs7,
};
use crate::{DEFAULT_PARAMETERS, FSRS6_DEFAULT_PARAMETERS, FSRSError};
use burn::backend::Autodiff;
use burn::backend::ndarray::NdArray;
use burn::data::dataloader::Progress;
use burn::lr_scheduler::LrScheduler;
use burn::module::AutodiffModule;
use burn::nn::loss::Reduction;
use burn::optim::Optimizer;
use burn::optim::{AdamConfig, GradientsParams};
use burn::tensor::Int;
use burn::tensor::Tensor;
use burn::tensor::backend::Backend;
use burn::tensor::cast::ToElement;
use burn::tensor::{Shape, TensorData};
use burn::train::TrainingInterrupter;
use burn::train::renderer::{MetricState, MetricsRenderer, TrainingProgress};
use burn::{config::Config, tensor::backend::AutodiffBackend};
use core::marker::PhantomData;
use log::info;

use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

#[path = "training_v6.rs"]
mod training_v6;
#[path = "training_v7.rs"]
mod training_v7;

type B = NdArray<f32>;

const L2_PENALTY_WEIGHT: f64 = training_v7::PENALTY_W_L2;
const PENALTY_GRAD_LEN: usize = training_v7::GRAD_LEN;

type SchedulePenaltyFn = fn(&[f32], usize, bool) -> (f64, [f64; PENALTY_GRAD_LEN]);
type L2PenaltyFn = fn(&[f32], &[f32], usize, usize, f64, &[f32]) -> (f64, Vec<f32>);

fn schedule_penalty_fn(version: ModelVersion) -> SchedulePenaltyFn {
    match version {
        ModelVersion::Fsrs6 => training_v6::maybe_schedule_penalty_value_and_grad,
        ModelVersion::Fsrs7 => training_v7::maybe_schedule_penalty_value_and_grad,
    }
}

fn l2_penalty_fn(version: ModelVersion) -> L2PenaltyFn {
    match version {
        ModelVersion::Fsrs6 => training_v6::l2_penalty_value_and_grad,
        ModelVersion::Fsrs7 => training_v7::l2_penalty_value_and_grad,
    }
}

pub struct BCELoss<B: Backend> {
    backend: PhantomData<B>,
}

impl<B: Backend> BCELoss<B> {
    pub const fn new() -> Self {
        Self {
            backend: PhantomData,
        }
    }
    pub fn forward(
        &self,
        retrievability: Tensor<B, 1>,
        labels: Tensor<B, 1>,
        weights: Tensor<B, 1>,
        mean: Reduction,
    ) -> Tensor<B, 1> {
        let loss = (labels.clone() * retrievability.clone().log()
            + (-labels + 1) * (-retrievability + 1).log())
            * weights.clone();
        // info!("loss: {}", &loss);
        match mean {
            Reduction::Mean => loss.mean().neg(),
            Reduction::Sum => loss.sum().neg(),
            Reduction::Auto => (loss.sum() / weights.sum()).neg(),
        }
    }
}

impl<B: Backend> Model<B> {
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
        let retrievability = self
            .power_forgetting_curve(delta_ts, state.stability)
            .clamp(0.0001_f32, 0.9999_f32);
        BCELoss::new().forward(retrievability, labels.float(), weights, reduce)
    }
}

impl<B: AutodiffBackend> Model<B> {
    fn add_manual_weight_gradient(
        &self,
        mut gradients: B::Gradients,
        manual_grad: &[f32],
    ) -> B::Gradients {
        let grad_tensor = self.w.grad(&gradients).unwrap();
        let device = grad_tensor.device();
        let grad_len = grad_tensor.dims()[0];
        let mut data = vec![0.0f32; grad_len];
        for (dst, src) in data.iter_mut().zip(manual_grad.iter()) {
            *dst = *src;
        }
        let manual_tensor = Tensor::from_floats(data.as_slice(), &device);
        let updated_grad = grad_tensor + manual_tensor;
        self.w.grad_remove(&mut gradients);
        self.w.grad_replace(&mut gradients, updated_grad);
        gradients
    }

    fn freeze_initial_stability(&self, mut grad: B::Gradients) -> B::Gradients {
        let grad_tensor = self.w.grad(&grad).unwrap();
        let device = grad_tensor.device();
        let updated_grad_tensor = grad_tensor.slice_assign([0..4], Tensor::zeros([4], &device));

        self.w.grad_remove(&mut grad);
        self.w.grad_replace(&mut grad, updated_grad_tensor);
        grad
    }

    fn freeze_short_term_stability(&self, mut grad: B::Gradients) -> B::Gradients {
        let grad_tensor = self.w.grad(&grad).unwrap();
        let device = grad_tensor.device();
        let updated_grad_tensor = if grad_tensor.dims()[0] >= 35 {
            grad_tensor.slice_assign([16..27], Tensor::zeros([11], &device))
        } else {
            grad_tensor.slice_assign([17..20], Tensor::zeros([3], &device))
        };

        self.w.grad_remove(&mut grad);
        self.w.grad_replace(&mut grad, updated_grad_tensor);
        grad
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
    pub interrupter: TrainingInterrupter,
    /// The index of the split we should update.
    pub index: usize,
}

impl ProgressCollector {
    pub fn new(state: Arc<Mutex<CombinedProgressState>>, index: usize) -> Self {
        Self {
            state,
            interrupter: Default::default(),
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

impl MetricsRenderer for ProgressCollector {
    fn update_train(&mut self, _state: MetricState) {}

    fn update_valid(&mut self, _state: MetricState) {}

    fn render_train(&mut self, item: TrainingProgress) {
        let mut info = self.state.lock().unwrap();
        let split = &mut info.splits[self.index];
        split.epoch = item.epoch;
        split.epoch_total = item.epoch_total;
        split.items_processed = item.progress.items_processed;
        split.items_total = item.progress.items_total;
        if info.want_abort {
            self.interrupter.stop();
        }
    }

    fn render_valid(&mut self, _item: TrainingProgress) {}
}

#[derive(Config)]
pub(crate) struct TrainingConfig {
    pub model: ModelConfig,
    pub optimizer: AdamConfig,
    #[config(default = true)]
    pub enable_sched_penalties: bool,
    #[config(default = 8)]
    pub num_epochs: usize,
    #[config(default = 1024)]
    pub batch_size: usize,
    #[config(default = 2023)]
    pub seed: u64,
    #[config(default = 2e-2)]
    pub learning_rate: f64,
    #[config(default = 1024)]
    pub max_seq_len: usize,
}

pub(crate) fn calculate_average_recall(items: &[FSRSItem]) -> f32 {
    calculate_average_recall_from_items(items.iter())
}

fn calculate_average_recall_from_items<'a>(items: impl Iterator<Item = &'a FSRSItem>) -> f32 {
    let (total_recall, total_reviews) = items
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
struct TrainingFSRSItem {
    item: FSRSItem,
    card_id: Option<i64>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ComputeParametersVersion {
    Fsrs6,
    #[default]
    Fsrs7,
}

/// Input parameters for computing FSRS parameters
#[derive(Clone, Debug)]
pub struct ComputeParametersInput {
    /// The training set containing review history
    pub train_set: Vec<FSRSItem>,
    /// Optional card id for each training item, aligned by index with `train_set`.
    ///
    /// When provided for FSRS-7 optimization, all surviving prefix items from
    /// the same card can be trained as one expanding window.
    pub card_ids: Option<Vec<i64>>,
    /// Optional progress tracking
    pub progress: Option<Arc<Mutex<CombinedProgressState>>>,
    /// Whether to enable short-term memory parameters
    pub enable_short_term: bool,
    /// Whether to enable FSRS-7 schedule penalties (penalty 1 & 2)
    pub enable_sched_penalties: bool,
    /// Target parameter version to optimize.
    pub model_version: ComputeParametersVersion,
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
            enable_sched_penalties: true,
            model_version: ComputeParametersVersion::default(),
            num_relearning_steps: None,
        }
    }
}

fn normalize_for_model_version(
    train_set: Vec<FSRSItem>,
    model_version: ComputeParametersVersion,
) -> Vec<FSRSItem> {
    match model_version {
        ComputeParametersVersion::Fsrs6 => train_set
            .into_iter()
            .map(|mut item| {
                for review in &mut item.reviews {
                    review.delta_t = review.delta_t.max(0.0).round();
                }
                item
            })
            .collect(),
        ComputeParametersVersion::Fsrs7 => train_set
            .into_iter()
            .map(|mut item| {
                for review in &mut item.reviews {
                    review.delta_t = review.delta_t.max(0.0);
                }
                item
            })
            .collect(),
    }
}

fn attach_card_ids(
    train_set: Vec<FSRSItem>,
    card_ids: Option<Vec<i64>>,
) -> Result<Vec<TrainingFSRSItem>> {
    if let Some(card_ids) = card_ids {
        if card_ids.len() != train_set.len() {
            return Err(FSRSError::InvalidInput);
        }
        Ok(train_set
            .into_iter()
            .zip(card_ids)
            .map(|(item, card_id)| TrainingFSRSItem {
                item,
                card_id: Some(card_id),
            })
            .collect())
    } else {
        Ok(train_set
            .into_iter()
            .map(|item| TrainingFSRSItem {
                item,
                card_id: None,
            })
            .collect())
    }
}

fn prepare_training_data_with_card_ids(
    items: Vec<TrainingFSRSItem>,
) -> (Vec<FSRSItem>, Vec<TrainingFSRSItem>) {
    let initialization_source_indices = items
        .iter()
        .enumerate()
        .filter_map(|(index, item)| (item.item.long_term_review_cnt() == 1).then_some(index))
        .collect::<Vec<_>>();
    let dataset_for_initialization = initialization_source_indices
        .iter()
        .map(|&index| items[index].item.clone())
        .collect::<Vec<_>>();

    if std::env::var("FSRS_NO_OUTLIER").is_ok() {
        let trainset = items
            .into_iter()
            .filter(|item| item.item.long_term_review_cnt() != 1)
            .collect();
        return (dataset_for_initialization, trainset);
    }

    let all_items = items
        .iter()
        .map(|item| item.item.clone())
        .collect::<Vec<_>>();
    let (initialization_indices, train_indices) =
        filter_outlier_indices(&dataset_for_initialization, &all_items);
    let filtered_initialization = initialization_indices
        .into_iter()
        .map(|index| {
            let source_index = initialization_source_indices[index];
            items[source_index].item.clone()
        })
        .collect();
    let mut training_items = items
        .into_iter()
        .map(Some)
        .collect::<Vec<Option<TrainingFSRSItem>>>();
    let trainset = train_indices
        .into_iter()
        .map(|index| training_items[index].take().unwrap())
        .collect();
    (filtered_initialization, trainset)
}

fn recency_weighted_training_items(items: Vec<TrainingFSRSItem>) -> Vec<WeightedFSRSItem> {
    let length = (items.len() as f32 - 1.0).max(1.0);
    items
        .into_iter()
        .enumerate()
        .map(|(idx, item)| WeightedFSRSItem {
            weight: 0.25 + 0.75 * (idx as f32 / length).powi(3),
            item: item.item,
            card_id: item.card_id,
        })
        .collect()
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
        enable_sched_penalties,
        model_version,
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

    let train_set = normalize_for_model_version(train_set, model_version);
    let (dataset_for_initialization, train_set) = if card_ids.is_some() {
        prepare_training_data_with_card_ids(attach_card_ids(train_set, card_ids)?)
    } else {
        let (dataset_for_initialization, train_set) = prepare_training_data(train_set);
        (
            dataset_for_initialization,
            train_set
                .into_iter()
                .map(|item| TrainingFSRSItem {
                    item,
                    card_id: None,
                })
                .collect(),
        )
    };
    let average_recall =
        calculate_average_recall_from_items(train_set.iter().map(|item| &item.item));
    if train_set.len() < 8 {
        finish_progress();
        return Ok(match model_version {
            ComputeParametersVersion::Fsrs6 => FSRS6_DEFAULT_PARAMETERS.to_vec(),
            ComputeParametersVersion::Fsrs7 => DEFAULT_PARAMETERS.to_vec(),
        });
    }

    let (initialized_parameters, fsrs6_initial_rating_count) = match model_version {
        ComputeParametersVersion::Fsrs6 => {
            let (initial_stability, initial_rating_count) =
                initialize_stability_parameters(dataset_for_initialization.clone(), average_recall)
                    .inspect_err(|_e| {
                        finish_progress();
                    })?;
            let initialized_parameters = initial_stability
                .into_iter()
                .chain(FSRS6_DEFAULT_PARAMETERS[4..].iter().copied())
                .collect();
            (initialized_parameters, Some(initial_rating_count))
        }
        ComputeParametersVersion::Fsrs7 => {
            let (initial_stability, initial_forgetting_curve, _initial_rating_count) =
                initialize_parameters_fsrs7(dataset_for_initialization.clone(), average_recall)
                    .inspect_err(|_e| {
                        finish_progress();
                    })?;
            let mut initialized_parameters = DEFAULT_PARAMETERS.to_vec();
            initialized_parameters[0..4].copy_from_slice(&initial_stability);
            initialized_parameters[27..35].copy_from_slice(&initial_forgetting_curve);
            (initialized_parameters, None)
        }
    };
    if train_set.len() == dataset_for_initialization.len() || train_set.len() < 64 {
        finish_progress();
        return Ok(initialized_parameters);
    }
    let config = TrainingConfig::new(
        ModelConfig {
            freeze_initial_stability: !enable_short_term,
            initial_stability: None,
            initial_forgetting_curve: None,
            freeze_short_term_stability: !enable_short_term,
            num_relearning_steps: num_relearning_steps.unwrap_or(1),
        },
        AdamConfig::new()
            .with_beta_1(0.8)
            .with_beta_2(0.85)
            .with_epsilon(1e-8),
    )
    .with_enable_sched_penalties(enable_sched_penalties);
    let mut weighted_train_set = recency_weighted_training_items(train_set);
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
    let model = train::<Autodiff<B>>(
        weighted_train_set,
        &initialized_parameters,
        &config,
        progress.clone().map(|p| ProgressCollector::new(p, 0)),
    );

    let optimized_parameters = model
        .inspect_err(|_e| {
            finish_progress();
        })?
        .w
        .val()
        .to_data()
        .to_vec()
        .unwrap();

    finish_progress();

    if optimized_parameters
        .iter()
        .any(|parameter: &f32| parameter.is_infinite())
    {
        return Err(FSRSError::InvalidInput);
    }

    let clamped_stability = match model_version {
        ComputeParametersVersion::Fsrs6 => {
            let initial_rating_count = fsrs6_initial_rating_count.expect("FSRS-6 rating count");
            let mut optimized_initial_stability = optimized_parameters[0..4]
                .iter()
                .enumerate()
                .map(|(i, &val)| (i as u32 + 1, val))
                .collect::<HashMap<_, _>>();
            smooth_and_fill(&mut optimized_initial_stability, &initial_rating_count)?
        }
        ComputeParametersVersion::Fsrs7 => {
            smooth_initial_stabilities_fsrs7(optimized_parameters[0..4].try_into().unwrap())?
        }
    };
    Ok(clamped_stability
        .into_iter()
        .chain(optimized_parameters[4..].iter().copied())
        .collect())
}

pub fn benchmark(
    ComputeParametersInput {
        train_set,
        enable_short_term,
        enable_sched_penalties,
        model_version,
        num_relearning_steps,
        ..
    }: ComputeParametersInput,
) -> Vec<f32> {
    let train_set = normalize_for_model_version(train_set, model_version);
    let average_recall = calculate_average_recall(&train_set);
    let (dataset_for_initialization, _next_train_set) = train_set
        .clone()
        .into_iter()
        .partition(|item| item.long_term_review_cnt() == 1);
    let initialized_parameters = match model_version {
        ComputeParametersVersion::Fsrs6 => {
            let (initial_stability, _rating_count) =
                initialize_stability_parameters(dataset_for_initialization, average_recall)
                    .unwrap();
            initial_stability
                .into_iter()
                .chain(FSRS6_DEFAULT_PARAMETERS[4..].iter().copied())
                .collect()
        }
        ComputeParametersVersion::Fsrs7 => {
            let (initial_stability, initial_forgetting_curve, _rating_count) =
                initialize_parameters_fsrs7(dataset_for_initialization, average_recall).unwrap();
            let mut initialized_parameters = DEFAULT_PARAMETERS.to_vec();
            initialized_parameters[0..4].copy_from_slice(&initial_stability);
            initialized_parameters[27..35].copy_from_slice(&initial_forgetting_curve);
            initialized_parameters
        }
    };
    let mut config = TrainingConfig::new(
        ModelConfig {
            freeze_initial_stability: !enable_short_term,
            initial_stability: None,
            initial_forgetting_curve: None,
            freeze_short_term_stability: !enable_short_term,
            num_relearning_steps: num_relearning_steps.unwrap_or(1),
        },
        AdamConfig::new()
            .with_beta_1(0.8)
            .with_beta_2(0.85)
            .with_epsilon(1e-8),
    )
    .with_enable_sched_penalties(enable_sched_penalties);
    // save RAM and speed up training
    config.max_seq_len = 64;
    let mut weighted_train_set = recency_weighted_fsrs_items(train_set);
    weighted_train_set.retain(|item| item.item.reviews.len() <= config.max_seq_len);
    let model = train::<Autodiff<B>>(weighted_train_set, &initialized_parameters, &config, None);
    let parameters: Vec<f32> = model.unwrap().w.val().to_data().to_vec::<f32>().unwrap();
    parameters
}

#[derive(Debug, Clone)]
struct WindowedFSRSBatch<B: Backend> {
    t_historys: Tensor<B, 2>,
    r_historys: Tensor<B, 2>,
    labels: Tensor<B, 2>,
    weights: Tensor<B, 2>,
    t_historys_host: Vec<f32>,
    r_historys_host: Vec<f32>,
    labels_host: Vec<f32>,
    weights_host: Vec<f32>,
    seq_len: usize,
    batch_size: usize,
    prediction_count: usize,
}

#[derive(Debug, Clone)]
enum TrainingBatch<B: Backend> {
    Prefix(crate::dataset::FSRSBatch<B>),
    Windowed(WindowedFSRSBatch<B>),
}

impl<B: Backend> TrainingBatch<B> {
    fn real_batch_size(&self) -> usize {
        match self {
            Self::Prefix(batch) => batch.delta_ts.shape().dims[0],
            Self::Windowed(batch) => batch.prediction_count,
        }
    }

    fn classification_loss(&self, model: &Model<B>) -> Tensor<B, 1> {
        match self {
            Self::Prefix(batch) => model.forward_classification(
                batch.t_historys.clone(),
                batch.r_historys.clone(),
                batch.delta_ts.clone(),
                batch.labels.clone(),
                batch.weights.clone(),
                Reduction::Sum,
            ),
            Self::Windowed(batch) => model.forward_windowed_classification(
                batch.t_historys.clone(),
                batch.r_historys.clone(),
                batch.labels.clone(),
                batch.weights.clone(),
            ),
        }
    }

    fn analytic_windowed_bce_grad(&self, w: &[f32]) -> Option<(f64, [f32; 35])> {
        match self {
            Self::Prefix(_) => None,
            Self::Windowed(batch) => Some(crate::analytic_v7::windowed_loss_and_grad(
                w,
                &batch.t_historys_host,
                &batch.r_historys_host,
                &batch.labels_host,
                &batch.weights_host,
                batch.seq_len,
                batch.batch_size,
            )),
        }
    }

    fn analytic_windowed_bce_loss(&self, w: &[f32]) -> Option<f64> {
        match self {
            Self::Prefix(_) => None,
            Self::Windowed(batch) => Some(crate::analytic_v7::windowed_loss(
                w,
                &batch.t_historys_host,
                &batch.r_historys_host,
                &batch.labels_host,
                &batch.weights_host,
                batch.seq_len,
                batch.batch_size,
            )),
        }
    }
}

impl<B: Backend> Model<B> {
    fn forward_windowed_classification(
        &self,
        t_historys: Tensor<B, 2>,
        r_historys: Tensor<B, 2>,
        labels: Tensor<B, 2>,
        weights: Tensor<B, 2>,
    ) -> Tensor<B, 1> {
        let [seq_len, batch_size] = t_historys.dims();
        let device = t_historys.device();
        let mut state = MemoryStateTensors::zeros(batch_size);
        let mut loss = Tensor::<B, 1>::zeros([1], &device);
        for i in 0..seq_len {
            let delta_t = t_historys.get(i).squeeze(0);
            let rating = r_historys.get(i).squeeze(0);
            if i > 0 {
                let retrievability = self
                    .power_forgetting_curve(delta_t.clone(), state.stability.clone())
                    .clamp(0.0001_f32, 0.9999_f32);
                let labels = labels.get(i).squeeze(0);
                let weights = weights.get(i).squeeze(0);
                loss =
                    loss + BCELoss::new().forward(retrievability, labels, weights, Reduction::Sum);
            }
            state = self.step(delta_t, rating, state, i);
        }
        loss
    }
}

fn build_windowed_batches<B: Backend>(
    items: &[WeightedFSRSItem],
    batch_size: usize,
) -> Vec<WindowedFSRSBatch<B>> {
    let mut cards: HashMap<i64, Vec<&WeightedFSRSItem>> = HashMap::new();
    for item in items {
        if let Some(card_id) = item.card_id {
            cards.entry(card_id).or_default().push(item);
        }
    }
    let mut cards = cards.into_iter().collect::<Vec<_>>();
    cards.sort_by_cached_key(|(card_id, prefixes)| {
        let full_len = prefixes
            .iter()
            .map(|item| item.item.reviews.len())
            .max()
            .unwrap_or(0);
        (full_len, *card_id)
    });

    let mut batches = Vec::new();
    let mut current = Vec::new();
    let mut current_predictions = 0usize;
    for (_card_id, mut prefixes) in cards {
        prefixes.sort_by_cached_key(|item| item.item.reviews.len());
        if !current.is_empty() && current_predictions + prefixes.len() > batch_size {
            batches.push(build_windowed_batch::<B>(&current));
            current.clear();
            current_predictions = 0;
        }
        current_predictions += prefixes.len();
        current.push(prefixes);
    }
    if !current.is_empty() {
        batches.push(build_windowed_batch::<B>(&current));
    }
    batches
}

fn build_windowed_batch<B: Backend>(cards: &[Vec<&WeightedFSRSItem>]) -> WindowedFSRSBatch<B> {
    let device = B::Device::default();
    let prediction_count = cards.iter().map(Vec::len).sum();
    let batch_size = cards.len();
    let seq_len = cards
        .iter()
        .filter_map(|prefixes| prefixes.last())
        .map(|item| item.item.reviews.len())
        .max()
        .expect("windowed batch is empty");
    let mut t_historys = vec![0.0f32; seq_len * batch_size];
    let mut r_historys = vec![0.0f32; seq_len * batch_size];
    let mut labels = vec![0.0f32; seq_len * batch_size];
    let mut weights = vec![0.0f32; seq_len * batch_size];

    for (column, prefixes) in cards.iter().enumerate() {
        let longest = prefixes.last().expect("card has at least one prefix item");
        for (row, review) in longest.item.reviews.iter().enumerate() {
            let index = row * batch_size + column;
            t_historys[index] = review.delta_t;
            r_historys[index] = review.rating as f32;
        }
        for prefix in prefixes {
            let row = prefix.item.reviews.len() - 1;
            let current = prefix.item.current();
            let index = row * batch_size + column;
            labels[index] = if current.rating == 1 { 0.0 } else { 1.0 };
            weights[index] = prefix.weight;
        }
    }

    let shape = Shape {
        dims: vec![seq_len, batch_size],
    };
    WindowedFSRSBatch {
        t_historys: Tensor::from_floats(
            TensorData::new(t_historys.clone(), shape.clone()),
            &device,
        ),
        r_historys: Tensor::from_floats(
            TensorData::new(r_historys.clone(), shape.clone()),
            &device,
        ),
        labels: Tensor::from_floats(TensorData::new(labels.clone(), shape.clone()), &device),
        weights: Tensor::from_floats(TensorData::new(weights.clone(), shape), &device),
        t_historys_host: t_historys,
        r_historys_host: r_historys,
        labels_host: labels,
        weights_host: weights,
        seq_len,
        batch_size,
        prediction_count,
    }
}

fn training_batches<B: Backend>(
    train_set: &[WeightedFSRSItem],
    batch_size: usize,
    windowed: bool,
) -> Vec<TrainingBatch<B>> {
    if windowed {
        build_windowed_batches::<B>(train_set, batch_size)
            .into_iter()
            .map(TrainingBatch::Windowed)
            .collect()
    } else {
        BatchTensorDataset::<B>::from_sorted_items(train_set, batch_size)
            .into_batches()
            .into_iter()
            .map(TrainingBatch::Prefix)
            .collect()
    }
}

fn train<B: AutodiffBackend>(
    train_set: Vec<WeightedFSRSItem>,
    initial_parameters: &[f32],
    config: &TrainingConfig,
    progress: Option<ProgressCollector>,
) -> Result<Model<B>> {
    B::seed(config.seed);

    let total_size = train_set.len();
    let iterations = (total_size / config.batch_size + 1) * config.num_epochs;
    let use_windowed = ModelVersion::from_param_count(initial_parameters.len())
        == ModelVersion::Fsrs7
        && train_set.iter().all(|item| item.card_id.is_some());
    let train_set = if use_windowed {
        train_set
    } else {
        sort_items_by_review_length(train_set)
    };
    let train_batches = training_batches::<B>(&train_set, config.batch_size, use_windowed);
    let valid_batches =
        training_batches::<B::InnerBackend>(&train_set, config.batch_size, use_windowed);
    let mut train_rng = StdRng::seed_from_u64(config.seed);
    let mut valid_rng = StdRng::seed_from_u64(config.seed);

    let mut lr_scheduler = CosineAnnealingLR::init(iterations as f64, config.learning_rate);
    let interrupter = TrainingInterrupter::new();
    let mut renderer: Box<dyn MetricsRenderer> = match progress {
        Some(mut progress) => {
            progress.interrupter = interrupter.clone();
            Box::new(progress)
        }
        None => Box::new(NoProgress {}),
    };

    let mut model: Model<B> = parameters_to_model::<B>(initial_parameters, &B::Device::default());
    let schedule_penalty = schedule_penalty_fn(model.version());
    let l2_penalty = l2_penalty_fn(model.version());
    let init_w = model.w.val();
    let init_w_vec = init_w.to_data().to_vec::<f32>().unwrap();
    let mut optim = config.optimizer.init::<B, Model<B>>();

    let mut best_loss = f64::INFINITY;
    let mut best_model = model.clone();
    for epoch in 1..=config.num_epochs {
        let mut train_indices = (0..train_batches.len()).collect::<Vec<_>>();
        train_indices.shuffle(&mut train_rng);
        let mut iteration = 0;
        for batch_index in train_indices {
            iteration += 1;
            let item = &train_batches[batch_index];
            let real_batch_size = item.real_batch_size();
            let lr = LrScheduler::step(&mut lr_scheduler);
            let progress = Progress::new(iteration, train_batches.len());
            let l2_weight = L2_PENALTY_WEIGHT;
            let w_vec = model.w.val().to_data().to_vec::<f32>().unwrap();
            let (_l2_penalty_value, mut manual_grad) = l2_penalty(
                &w_vec,
                &init_w_vec,
                real_batch_size,
                total_size,
                l2_weight,
                &training_v7::PARAMS_STDDEV,
            );
            let (_schedule_value, schedule_grad) =
                schedule_penalty(&w_vec, real_batch_size, config.enable_sched_penalties);
            let inv_total = 1.0 / total_size as f64;
            for i in 0..manual_grad.len().min(schedule_grad.len()) {
                manual_grad[i] += (schedule_grad[i] * inv_total) as f32;
            }
            let grads = if let Some((_loss, bce_grad)) = item.analytic_windowed_bce_grad(&w_vec) {
                let mut total_grad = vec![0.0f32; w_vec.len()];
                for (dst, src) in total_grad.iter_mut().zip(bce_grad) {
                    *dst = src;
                }
                for (dst, src) in total_grad.iter_mut().zip(manual_grad.iter()) {
                    *dst += *src;
                }
                if config.model.freeze_initial_stability {
                    for value in total_grad.iter_mut().take(4) {
                        *value = 0.0;
                    }
                }
                if config.model.freeze_short_term_stability {
                    let range = if total_grad.len() >= 35 {
                        16..27
                    } else {
                        17..20
                    };
                    for value in total_grad[range].iter_mut() {
                        *value = 0.0;
                    }
                }
                let mut grads = GradientsParams::new();
                let grad_tensor = Tensor::<B::InnerBackend, 1>::from_floats(
                    total_grad.as_slice(),
                    &B::Device::default(),
                );
                grads.register(model.w.id, grad_tensor);
                grads
            } else {
                let loss = item.classification_loss(&model);
                let mut gradients = loss.backward();
                gradients = model.add_manual_weight_gradient(gradients, &manual_grad);
                if config.model.freeze_initial_stability {
                    gradients = model.freeze_initial_stability(gradients);
                }
                if config.model.freeze_short_term_stability {
                    gradients = model.freeze_short_term_stability(gradients);
                }
                GradientsParams::from_grads(gradients, &model)
            };
            model = optim.step(lr, model, grads);
            model.w = parameter_clipper(
                model.w,
                config.model.num_relearning_steps,
                !config.model.freeze_short_term_stability,
            );
            // info!("epoch: {:?} iteration: {:?} lr: {:?}", epoch, iteration, lr);
            renderer.render_train(TrainingProgress {
                progress,
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

        let model_valid = model.valid();
        let mut loss_valid = 0.0;
        let mut valid_indices = (0..valid_batches.len()).collect::<Vec<_>>();
        valid_indices.shuffle(&mut valid_rng);
        for batch_index in valid_indices {
            let batch = &valid_batches[batch_index];
            let real_batch_size = batch.real_batch_size();
            let l2_weight = L2_PENALTY_WEIGHT;
            let w_vec = model_valid.w.val().to_data().to_vec::<f32>().unwrap();
            let (l2_penalty_value, _) = l2_penalty(
                &w_vec,
                &init_w_vec,
                real_batch_size,
                total_size,
                l2_weight,
                &training_v7::PARAMS_STDDEV,
            );
            let (schedule_value, _) =
                schedule_penalty(&w_vec, real_batch_size, config.enable_sched_penalties);
            let schedule_penalty = schedule_value / total_size as f64;
            let loss = batch.analytic_windowed_bce_loss(&w_vec).unwrap_or_else(|| {
                batch
                    .classification_loss(&model_valid)
                    .into_scalar()
                    .to_f64()
            });
            loss_valid += loss + l2_penalty_value + schedule_penalty;

            if interrupter.should_stop() {
                break;
            }
        }
        loss_valid /= total_size as f64;
        info!("epoch: {:?} loss: {:?}", epoch, loss_valid);
        if loss_valid < best_loss {
            best_loss = loss_valid;
            best_model = model.clone();
        }
    }

    info!("best_loss: {:?}", best_loss);

    if interrupter.should_stop() {
        return Err(FSRSError::Interrupted);
    }

    Ok(best_model)
}

struct NoProgress {}

impl MetricsRenderer for NoProgress {
    fn update_train(&mut self, _state: MetricState) {}

    fn update_valid(&mut self, _state: MetricState) {}

    fn render_train(&mut self, _item: TrainingProgress) {}

    fn render_valid(&mut self, _item: TrainingProgress) {}
}

#[cfg(test)]
mod tests {
    use std::fs::create_dir_all;
    use std::path::Path;
    use std::thread;
    use std::time::Duration;

    use super::*;
    use crate::convertor_tests::anki21_sample_file_converted_to_fsrs;
    use crate::convertor_tests::try_data_from_csv;
    use crate::dataset::{FSRSBatch, FSRSBatcher};
    use crate::model::{FSRS, parameters_to_model};
    use crate::test_helpers::TestHelper;
    use crate::{DEFAULT_PARAMETERS, FSRS6_DEFAULT_PARAMETERS};
    use burn::backend::ndarray::NdArrayDevice;
    use burn::backend::{Autodiff, NdArray};
    use burn::data::dataloader::batcher::Batcher;
    use log::LevelFilter;

    #[test]
    fn test_calculate_average_recall() {
        let items = anki21_sample_file_converted_to_fsrs();
        let average_recall = calculate_average_recall(&items);
        assert_eq!(average_recall, 0.9435269);
    }

    #[test]
    fn test_normalize_for_model_version_rounds_fsrs6_only() {
        let train_set = vec![FSRSItem {
            reviews: vec![
                crate::FSRSReview {
                    rating: 1,
                    delta_t: -0.2,
                },
                crate::FSRSReview {
                    rating: 3,
                    delta_t: 0.49,
                },
                crate::FSRSReview {
                    rating: 3,
                    delta_t: 0.51,
                },
            ],
        }];
        let fsrs6 = normalize_for_model_version(train_set.clone(), ComputeParametersVersion::Fsrs6);
        let fsrs7 = normalize_for_model_version(train_set, ComputeParametersVersion::Fsrs7);
        let fsrs6_days: Vec<f32> = fsrs6[0].reviews.iter().map(|r| r.delta_t).collect();
        let fsrs7_days: Vec<f32> = fsrs7[0].reviews.iter().map(|r| r.delta_t).collect();
        assert_eq!(fsrs6_days, vec![0.0, 0.0, 1.0]);
        assert_eq!(fsrs7_days, vec![0.0, 0.49, 0.51]);
    }

    #[test]
    fn test_compute_parameters_rejects_misaligned_card_ids() {
        let err = compute_parameters(ComputeParametersInput {
            train_set: vec![FSRSItem {
                reviews: vec![
                    crate::FSRSReview {
                        rating: 3,
                        delta_t: 0.0,
                    },
                    crate::FSRSReview {
                        rating: 3,
                        delta_t: 1.0,
                    },
                ],
            }],
            card_ids: Some(vec![]),
            ..Default::default()
        })
        .unwrap_err();
        assert_eq!(err, FSRSError::InvalidInput);
    }

    #[test]
    fn test_prepare_training_data_with_card_ids_keeps_alignment() {
        let items = vec![
            TrainingFSRSItem {
                item: FSRSItem {
                    reviews: vec![
                        crate::FSRSReview {
                            rating: 3,
                            delta_t: 0.0,
                        },
                        crate::FSRSReview {
                            rating: 3,
                            delta_t: 0.5,
                        },
                    ],
                },
                card_id: Some(11),
            },
            TrainingFSRSItem {
                item: FSRSItem {
                    reviews: vec![
                        crate::FSRSReview {
                            rating: 4,
                            delta_t: 0.0,
                        },
                        crate::FSRSReview {
                            rating: 3,
                            delta_t: 0.25,
                        },
                        crate::FSRSReview {
                            rating: 2,
                            delta_t: 0.75,
                        },
                    ],
                },
                card_id: Some(22),
            },
        ];

        let (_initialization, trainset) = prepare_training_data_with_card_ids(items);

        assert_eq!(
            trainset
                .iter()
                .map(|item| item.card_id.unwrap())
                .collect::<Vec<_>>(),
            vec![11, 22]
        );
        assert_eq!(trainset[1].item.current().rating, 2);
    }

    #[test]
    fn test_windowed_loss_and_grad_match_prefix_batch_for_single_card() {
        let reviews = [(3, 0.0), (4, 1.0), (3, 5.0), (1, 10.0), (3, 2.0)];
        let weighted_items = (2..=reviews.len())
            .map(|prefix_len| WeightedFSRSItem {
                weight: 0.5 + prefix_len as f32 * 0.1,
                item: FSRSItem {
                    reviews: reviews[..prefix_len]
                        .iter()
                        .map(|&(rating, delta_t)| crate::FSRSReview { rating, delta_t })
                        .collect(),
                },
                card_id: Some(42),
            })
            .collect::<Vec<_>>();
        type B = Autodiff<NdArray<f32>>;
        let device = NdArrayDevice::Cpu;
        let prefix_batch = FSRSBatcher::<B>::new().batch(weighted_items.clone(), &device);
        let windowed_batch = build_windowed_batches::<B>(&weighted_items, 512)
            .pop()
            .unwrap();
        let (analytic_loss_value, analytic_grad) = crate::analytic_v7::windowed_loss_and_grad(
            &DEFAULT_PARAMETERS,
            &windowed_batch.t_historys_host,
            &windowed_batch.r_historys_host,
            &windowed_batch.labels_host,
            &windowed_batch.weights_host,
            windowed_batch.seq_len,
            windowed_batch.batch_size,
        );
        let analytic_loss_only = crate::analytic_v7::windowed_loss(
            &DEFAULT_PARAMETERS,
            &windowed_batch.t_historys_host,
            &windowed_batch.r_historys_host,
            &windowed_batch.labels_host,
            &windowed_batch.weights_host,
            windowed_batch.seq_len,
            windowed_batch.batch_size,
        );
        let prefix_model = parameters_to_model::<B>(&DEFAULT_PARAMETERS, &device);
        let windowed_model = parameters_to_model::<B>(&DEFAULT_PARAMETERS, &device);

        let prefix_loss = prefix_model.forward_classification(
            prefix_batch.t_historys,
            prefix_batch.r_historys,
            prefix_batch.delta_ts,
            prefix_batch.labels,
            prefix_batch.weights,
            Reduction::Sum,
        );
        let windowed_loss = windowed_model.forward_windowed_classification(
            windowed_batch.t_historys,
            windowed_batch.r_historys,
            windowed_batch.labels,
            windowed_batch.weights,
        );
        let prefix_loss_value = prefix_loss.clone().into_scalar().to_f32();
        let windowed_loss_value = windowed_loss.clone().into_scalar().to_f32();
        assert!(
            (prefix_loss_value - windowed_loss_value).abs() < 1e-5,
            "prefix loss {prefix_loss_value}, windowed loss {windowed_loss_value}"
        );
        assert!(
            (analytic_loss_value as f32 - windowed_loss_value).abs() < 1e-5,
            "analytic loss {analytic_loss_value}, windowed loss {windowed_loss_value}"
        );
        assert!(
            (analytic_loss_only as f32 - windowed_loss_value).abs() < 1e-5,
            "analytic loss-only {analytic_loss_only}, windowed loss {windowed_loss_value}"
        );

        let prefix_gradients = prefix_loss.backward();
        let windowed_gradients = windowed_loss.backward();
        let prefix_grad = prefix_model
            .w
            .grad(&prefix_gradients)
            .unwrap()
            .to_data()
            .to_vec::<f32>()
            .unwrap();
        let windowed_grad = windowed_model
            .w
            .grad(&windowed_gradients)
            .unwrap()
            .to_data()
            .to_vec::<f32>()
            .unwrap();
        let max_diff = prefix_grad
            .iter()
            .zip(&windowed_grad)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(max_diff < 1e-4, "max gradient diff {max_diff}");
        let analytic_max_diff = analytic_grad
            .iter()
            .zip(&windowed_grad)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            analytic_max_diff < 1e-4,
            "max analytic gradient diff {analytic_max_diff}"
        );
    }

    #[test]
    fn test_windowed_validation_loss_matches_burn_with_padding() {
        let card_a = [(3, 0.0), (4, 1.0), (3, 5.0), (1, 10.0), (3, 2.0)];
        let card_b = [(1, 0.0), (3, 0.5), (2, 1.0)];
        let mut weighted_items = Vec::new();
        for prefix_len in 2..=card_a.len() {
            weighted_items.push(WeightedFSRSItem {
                weight: 0.4 + prefix_len as f32 * 0.05,
                item: FSRSItem {
                    reviews: card_a[..prefix_len]
                        .iter()
                        .map(|&(rating, delta_t)| crate::FSRSReview { rating, delta_t })
                        .collect(),
                },
                card_id: Some(100),
            });
        }
        for prefix_len in 2..=card_b.len() {
            weighted_items.push(WeightedFSRSItem {
                weight: 0.6 + prefix_len as f32 * 0.03,
                item: FSRSItem {
                    reviews: card_b[..prefix_len]
                        .iter()
                        .map(|&(rating, delta_t)| crate::FSRSReview { rating, delta_t })
                        .collect(),
                },
                card_id: Some(200),
            });
        }

        type B = NdArray<f32>;
        let device = NdArrayDevice::Cpu;
        let windowed_batch = build_windowed_batches::<B>(&weighted_items, 512)
            .pop()
            .unwrap();
        let analytic_loss = crate::analytic_v7::windowed_loss(
            &DEFAULT_PARAMETERS,
            &windowed_batch.t_historys_host,
            &windowed_batch.r_historys_host,
            &windowed_batch.labels_host,
            &windowed_batch.weights_host,
            windowed_batch.seq_len,
            windowed_batch.batch_size,
        );
        let model = parameters_to_model::<B>(&DEFAULT_PARAMETERS, &device);
        let burn_loss = model.forward_windowed_classification(
            windowed_batch.t_historys,
            windowed_batch.r_historys,
            windowed_batch.labels,
            windowed_batch.weights,
        );
        let burn_loss = burn_loss.into_scalar().to_f64();
        assert!(
            (analytic_loss - burn_loss).abs() < 1e-5,
            "analytic validation loss {analytic_loss}, burn loss {burn_loss}"
        );
    }

    #[test]
    fn test_compute_parameters_small_dataset_fsrs6_defaults() {
        let parameters = compute_parameters(ComputeParametersInput {
            train_set: vec![],
            card_ids: None,
            progress: None,
            enable_short_term: true,
            enable_sched_penalties: true,
            model_version: ComputeParametersVersion::Fsrs6,
            num_relearning_steps: None,
        })
        .unwrap();
        assert_eq!(parameters, FSRS6_DEFAULT_PARAMETERS.to_vec());
    }

    #[test]
    fn test_compute_parameters_small_dataset_fsrs7_defaults() {
        let parameters = compute_parameters(ComputeParametersInput {
            train_set: vec![],
            card_ids: None,
            progress: None,
            enable_short_term: true,
            enable_sched_penalties: true,
            model_version: ComputeParametersVersion::Fsrs7,
            num_relearning_steps: None,
        })
        .unwrap();
        assert_eq!(parameters, DEFAULT_PARAMETERS.to_vec());
    }

    #[test]
    fn test_compute_parameters_fsrs7_with_same_day_only_items_no_panic() {
        let train_set = vec![
            FSRSItem {
                reviews: vec![
                    crate::FSRSReview {
                        rating: 2,
                        delta_t: 0.0,
                    },
                    crate::FSRSReview {
                        rating: 3,
                        delta_t: 0.5,
                    },
                ],
            },
            FSRSItem {
                reviews: vec![
                    crate::FSRSReview {
                        rating: 1,
                        delta_t: 0.0,
                    },
                    crate::FSRSReview {
                        rating: 2,
                        delta_t: 0.25,
                    },
                ],
            },
        ];

        let parameters = compute_parameters(ComputeParametersInput {
            train_set,
            card_ids: None,
            progress: None,
            enable_short_term: true,
            enable_sched_penalties: true,
            model_version: ComputeParametersVersion::Fsrs7,
            num_relearning_steps: None,
        });

        assert!(parameters.is_ok());
        assert_eq!(parameters.unwrap().len(), 35);
    }

    #[test]
    fn test_compute_parameters_fsrs7_with_card_ids_smoke() {
        let mut train_set = Vec::new();
        let mut card_ids = Vec::new();
        for card_id in 0..16 {
            let reviews = [(3, 0.0), (4, 1.0), (3, 3.0), (2, 5.0), (3, 8.0), (4, 13.0)];
            for prefix_len in 2..=reviews.len() {
                train_set.push(FSRSItem {
                    reviews: reviews[..prefix_len]
                        .iter()
                        .map(|&(rating, delta_t)| crate::FSRSReview { rating, delta_t })
                        .collect(),
                });
                card_ids.push(card_id);
            }
        }

        let parameters = compute_parameters(ComputeParametersInput {
            train_set,
            card_ids: Some(card_ids),
            progress: None,
            enable_short_term: true,
            enable_sched_penalties: false,
            model_version: ComputeParametersVersion::Fsrs7,
            num_relearning_steps: None,
        })
        .unwrap();

        assert_eq!(parameters.len(), 35);
        assert!(parameters.iter().all(|parameter| parameter.is_finite()));
    }

    #[test]
    fn test_forward_classification_clamps_retrievability_for_bce() {
        use burn::backend::ndarray::NdArrayDevice;
        use burn::tensor::TensorData;

        let device = NdArrayDevice::Cpu;
        type B = NdArray<f32>;
        let model: Model<B> = parameters_to_model::<B>(&DEFAULT_PARAMETERS, &device);

        let loss = model.forward_classification(
            Tensor::from_floats(TensorData::from([[0.0, 0.0]]), &device),
            Tensor::from_floats(TensorData::from([[1.0, 1.0]]), &device),
            Tensor::from_floats([0.0, f32::MAX], &device),
            Tensor::from_ints([1, 0], &device),
            Tensor::from_floats([1.0, 1.0], &device),
            Reduction::Sum,
        );

        let actual = loss.into_scalar().to_f32();
        let expected = -2.0 * 0.9999_f32.ln();
        assert!(actual.is_finite());
        assert!((actual - expected).abs() < 1e-7);
    }

    #[test]
    fn test_loss_and_grad() {
        use burn::backend::ndarray::NdArrayDevice;
        use burn::tensor::TensorData;

        let device = NdArrayDevice::Cpu;
        type B = Autodiff<NdArray<f32>>;
        let mut model: Model<B> = parameters_to_model::<B>(&FSRS6_DEFAULT_PARAMETERS, &device);
        let init_w = model.w.val();

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

        let config =
            TrainingConfig::new(ModelConfig::default(), AdamConfig::new().with_epsilon(1e-8));
        let mut optim = config.optimizer.init::<B, Model<B>>();
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

        let init_w_vec = init_w.to_data().to_vec::<f32>().unwrap();
        let w_vec = model.w.val().to_data().to_vec::<f32>().unwrap();
        let (penalty_value, grad_vec) = training_v6::l2_penalty_value_and_grad(
            &w_vec,
            &init_w_vec,
            512,
            1000,
            L2_PENALTY_WEIGHT,
            &training_v7::PARAMS_STDDEV,
        );
        assert!((penalty_value - 0.16927784).abs() < 1e-6);
        grad_vec.assert_approx_eq([
            0.0004953454,
            0.00021947007,
            0.00006626537,
            -0.000026404574,
            -0.06303472,
            0.26122463,
            -0.056888837,
            1.4222223,
            -0.13464814,
            0.63209885,
            -0.18806253,
            0.22755535,
            -2.5283923,
            0.79999983,
            0.06303435,
            0.3276802,
            -0.019304348,
            -0.21311146,
            0.19999984,
            1.0448979,
            -0.28093278,
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
        let mut datasets = vec![anki21_sample_file_converted_to_fsrs()];
        if let Some(items) = try_data_from_csv() {
            datasets.push(items);
        } else {
            eprintln!("Skipping optional tests/data/revlog.csv fixture");
        }

        for items in datasets {
            for model_version in [
                ComputeParametersVersion::Fsrs6,
                ComputeParametersVersion::Fsrs7,
            ] {
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
                        enable_sched_penalties: true,
                        model_version,
                        num_relearning_steps: None,
                    })
                    .unwrap();
                    dbg!(&parameters);
                    match model_version {
                        ComputeParametersVersion::Fsrs6 => assert_eq!(parameters.len(), 21),
                        ComputeParametersVersion::Fsrs7 => assert_eq!(parameters.len(), 35),
                    }

                    // evaluate
                    let model = FSRS::new(&parameters).unwrap();
                    let metrics = model.evaluate(items.clone(), |_| true).unwrap();
                    dbg!(&metrics);
                }
            }
        }
    }

    #[test]
    fn test_manual_l2_penalty_matches_autodiff_gradient() {
        type B = Autodiff<NdArray<f32>>;
        let config = ModelConfig::default();
        let model: Model<B> = config.init();
        let device = model.w.device();
        let w_vec = model.w.val().to_data().to_vec::<f32>().unwrap();
        let mut init_w_vec = w_vec.clone();
        for (i, init) in init_w_vec.iter_mut().enumerate() {
            *init -= 0.05 * ((i + 1) as f32) / (PENALTY_GRAD_LEN as f32);
        }

        let init_w = Tensor::from_floats(init_w_vec.as_slice(), &device);
        let params_stddev = Tensor::from_floats(training_v7::PARAMS_STDDEV, &device);
        let penalty = (model.w.val() - init_w)
            .powi_scalar(2)
            .div(params_stddev.powi_scalar(2))
            .sum()
            .mul_scalar(L2_PENALTY_WEIGHT * 512.0 / 1000.0);
        let expected_value = penalty.clone().into_scalar().to_f64();
        let gradients = penalty.backward();
        let expected_grad = model
            .w
            .grad(&gradients)
            .unwrap()
            .to_data()
            .to_vec::<f32>()
            .unwrap();

        let (actual_value, actual_grad) = training_v7::l2_penalty_value_and_grad(
            &w_vec,
            &init_w_vec,
            512,
            1000,
            L2_PENALTY_WEIGHT,
            &training_v7::PARAMS_STDDEV,
        );
        assert!(
            (actual_value - expected_value).abs() < 1e-6,
            "l2 value mismatch actual={} expected={}",
            actual_value,
            expected_value
        );
        for (expected, actual) in expected_grad.iter().zip(actual_grad.iter()) {
            assert!((actual - expected).abs() < 1e-6);
        }
    }
}

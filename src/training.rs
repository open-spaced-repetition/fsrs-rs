use crate::cosine_annealing::CosineAnnealingLR;
use crate::dataset::{
    FSRSItem, WeightedFSRSItem, filter_outlier_train_indices, prepare_training_data,
    recency_weighted_fsrs_items,
};
use crate::error::Result;
use crate::model::{FSRS, ModelConfig, ModelVersion};
use crate::parameter_clipper::clip_parameters_in_place;
use crate::parameter_initialization::{initialize_stability_parameters, smooth_and_fill};
use crate::parameter_initialization_fsrs7::{
    initialize_parameters_fsrs7, smooth_initial_stabilities_fsrs7,
};
use crate::{DEFAULT_PARAMETERS, FSRS6_DEFAULT_PARAMETERS, FSRSError};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use std::collections::{BTreeMap, HashMap};
use std::sync::{Arc, Mutex};

#[path = "training_v6.rs"]
mod training_v6;
#[path = "training_v7.rs"]
mod training_v7;

const L2_PENALTY_WEIGHT: f64 = training_v7::PENALTY_W_L2;
const PENALTY_GRAD_LEN: usize = training_v7::GRAD_LEN;
const ADAM_BETA_1: f32 = 0.70;
const ADAM_BETA_2: f32 = 0.98;
const ADAM_EPSILON: f32 = 1e-8;
const WINDOWED_FSRS7_LEARNING_RATE: f64 = 0.07;
const WINDOWED_FSRS7_NUM_EPOCHS: usize = 17;

type SchedulePenaltyFn = fn(&[f32], usize, bool) -> (f64, [f64; PENALTY_GRAD_LEN]);
type L2PenaltyFn = fn(&[f32], &[f32], usize, usize, f64, &[f32]) -> (f64, Vec<f32>);

fn schedule_penalty_fn(version: ModelVersion) -> SchedulePenaltyFn {
    match version {
        ModelVersion::Fsrs6 => training_v6::maybe_schedule_penalty_value_and_grad,
        ModelVersion::Fsrs7 => training_v7::maybe_schedule_penalty_value_and_grad,
    }
}

fn validation_schedule_penalty_value(
    _version: ModelVersion,
    _w: &[f32],
    _batch_size: usize,
    _enable_sched_penalties: bool,
) -> f64 {
    0.0
}

fn l2_penalty_fn(version: ModelVersion) -> L2PenaltyFn {
    match version {
        ModelVersion::Fsrs6 => training_v6::l2_penalty_value_and_grad,
        ModelVersion::Fsrs7 => training_v7::l2_penalty_value_and_grad,
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum TrainingObjective {
    ProbabilityLogLoss,
    HardClassificationCost {
        costs: RecallClassificationCosts,
        decision_boundary: f32,
        decision_sharpness: f32,
    },
}

pub(crate) fn weighted_binary_cross_entropy(
    retrievability: &[f32],
    labels: &[f32],
    weights: &[f32],
) -> f32 {
    let mut loss = 0.0;
    let mut weight_sum = 0.0;
    for ((&r, &label), &weight) in retrievability.iter().zip(labels).zip(weights) {
        let r = r.clamp(0.0001, 0.9999);
        loss += (label * r.ln() + (1.0 - label) * (1.0 - r).ln()) * weight;
        weight_sum += weight;
    }
    -loss / weight_sum
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

fn apply_training_config(config: &mut InternalTrainingConfig, custom: Option<TrainingConfig>) {
    if let Some(custom) = custom {
        config.num_epochs = custom.num_epochs;
        config.batch_size = custom.batch_size;
        config.seed = custom.seed;
        config.learning_rate = custom.learning_rate;
        config.max_seq_len = custom.max_seq_len;
        config.gamma = custom.gamma;
    }
}

#[derive(Debug, Clone)]
pub(crate) struct InternalTrainingConfig {
    pub model: ModelConfig,
    pub enable_sched_penalties: bool,
    pub num_epochs: usize,
    pub batch_size: usize,
    pub seed: u64,
    pub learning_rate: f64,
    pub max_seq_len: usize,
    pub gamma: f64,
}

impl InternalTrainingConfig {
    fn new(model: ModelConfig) -> Self {
        Self {
            model,
            enable_sched_penalties: false,
            num_epochs: 9,
            batch_size: 512,
            seed: 2023,
            learning_rate: 0.0118,
            max_seq_len: 1024,
            gamma: 1.0,
        }
    }

    fn with_enable_sched_penalties(mut self, enabled: bool) -> Self {
        self.enable_sched_penalties = enabled;
        self
    }
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

/// Relative costs used by FSRS recall-classifier training.
///
/// A positive decision means that the card is predicted to be recalled. The
/// costs are normalized over the training set, so only their ratio affects
/// fitting.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RecallClassificationCosts {
    /// Cost of predicting recalled when the answer is `Again`.
    pub false_positive: f32,
    /// Cost of predicting forgotten when the answer is `Hard`, `Good`, or `Easy`.
    pub false_negative: f32,
}

impl Default for RecallClassificationCosts {
    fn default() -> Self {
        Self {
            false_positive: 1.0,
            false_negative: 1.0,
        }
    }
}

impl RecallClassificationCosts {
    fn validate(self) -> Result<()> {
        if self.false_positive.is_finite()
            && self.false_positive > 0.0
            && self.false_negative.is_finite()
            && self.false_negative > 0.0
        {
            Ok(())
        } else {
            Err(FSRSError::InvalidInput)
        }
    }

    fn for_item(self, item: &FSRSItem) -> f32 {
        if item.current().rating == 1 {
            self.false_positive
        } else {
            self.false_negative
        }
    }
}

/// Configuration for fitting FSRS parameters to make a binary recall decision.
#[derive(Clone, Debug, PartialEq)]
pub struct RecallClassifierTrainingConfig {
    /// Relative costs for the two kinds of classification error.
    pub costs: RecallClassificationCosts,
    /// Fixed retrievability boundary used to classify a review as recalled.
    pub decision_boundary: f32,
    /// Steepness of the surrogate gradient around the hard decision boundary.
    pub decision_sharpness: f32,
    /// Optional valid FSRS parameters used to warm-start optimization.
    pub initial_parameters: Option<Vec<f32>>,
}

impl Default for RecallClassifierTrainingConfig {
    fn default() -> Self {
        Self {
            costs: RecallClassificationCosts::default(),
            decision_boundary: 0.5,
            decision_sharpness: 20.0,
            initial_parameters: None,
        }
    }
}

impl RecallClassifierTrainingConfig {
    fn validate(&self) -> Result<()> {
        self.costs.validate()?;
        if self.decision_boundary.is_finite()
            && self.decision_boundary > 0.0
            && self.decision_boundary < 1.0
            && self.decision_sharpness.is_finite()
            && self.decision_sharpness > 0.0
        {
            Ok(())
        } else {
            Err(FSRSError::InvalidInput)
        }
    }
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
    /// Optional hyperparameters; None preserves version-specific branch defaults.
    pub training_config: Option<TrainingConfig>,
}

impl Default for ComputeParametersInput {
    fn default() -> Self {
        Self {
            train_set: Vec::new(),
            card_ids: None,
            progress: None,
            enable_short_term: true,
            enable_sched_penalties: false,
            model_version: ComputeParametersVersion::default(),
            num_relearning_steps: None,
            training_config: None,
        }
    }
}

fn apply_windowed_fsrs7_training_tuning(
    config: &mut InternalTrainingConfig,
    model_version: ComputeParametersVersion,
    has_card_ids: bool,
) {
    if model_version == ComputeParametersVersion::Fsrs7 && has_card_ids {
        config.learning_rate = WINDOWED_FSRS7_LEARNING_RATE;
        config.num_epochs = WINDOWED_FSRS7_NUM_EPOCHS;
    }
}

fn should_validate_epoch(_epoch: usize, _total_epochs: usize, use_windowed: bool) -> bool {
    !use_windowed
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

    let (initialization_indices, train_indices) = filter_outlier_train_indices(
        &dataset_for_initialization,
        items.iter().map(|item| &item.item),
    );
    let filtered_initialization = initialization_indices
        .into_iter()
        .map(|index| {
            let source_index = initialization_source_indices[index];
            items[source_index].item.clone()
        })
        .collect();
    let mut train_indices = train_indices.into_iter();
    let mut next_train_index = train_indices.next();
    let mut trainset = Vec::new();
    for (index, item) in items.into_iter().enumerate() {
        if next_train_index == Some(index) {
            trainset.push(item);
            next_train_index = train_indices.next();
        }
    }
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

fn normalized_classification_costs(
    costs: RecallClassificationCosts,
    items: &[TrainingFSRSItem],
) -> RecallClassificationCosts {
    let cost_sum = items
        .iter()
        .map(|item| costs.for_item(&item.item))
        .sum::<f32>();
    let scale = items.len() as f32 / cost_sum;
    RecallClassificationCosts {
        false_positive: costs.false_positive * scale,
        false_negative: costs.false_negative * scale,
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
pub fn compute_parameters(input: ComputeParametersInput) -> Result<Vec<f32>> {
    compute_parameters_inner(input, None)
}

/// Computes FSRS parameters with a binary-classification loss.
///
/// The forward loss is the realized false-positive and false-negative cost at
/// `config.decision_boundary`. A sigmoid straight-through gradient makes the
/// hard decision trainable. The returned parameters still produce continuous
/// retrievability, but fitting targets the final binary decision rather than
/// probability calibration.
pub fn compute_parameters_for_recall_classifier(
    input: ComputeParametersInput,
    config: RecallClassifierTrainingConfig,
) -> Result<Vec<f32>> {
    compute_parameters_inner(input, Some(config))
}

fn compute_parameters_inner(
    ComputeParametersInput {
        train_set,
        card_ids,
        progress,
        enable_short_term,
        enable_sched_penalties,
        model_version,
        num_relearning_steps,
        training_config,
        ..
    }: ComputeParametersInput,
    classifier_config: Option<RecallClassifierTrainingConfig>,
) -> Result<Vec<f32>> {
    if let Some(config) = &classifier_config {
        config.validate()?;
    }
    let initial_parameters = classifier_config
        .as_ref()
        .and_then(|config| config.initial_parameters.clone());
    let initial_parameters = initial_parameters
        .map(|parameters| {
            let expected_len = match model_version {
                ComputeParametersVersion::Fsrs6 => FSRS6_DEFAULT_PARAMETERS.len(),
                ComputeParametersVersion::Fsrs7 => DEFAULT_PARAMETERS.len(),
            };
            if parameters.len() == expected_len && parameters.iter().all(|value| value.is_finite())
            {
                Ok(parameters)
            } else {
                Err(FSRSError::InvalidInput)
            }
        })
        .transpose()?;
    let has_card_ids = card_ids.is_some();
    let finish_progress = || {
        if let Some(progress) = &progress {
            // The progress state at completion time may not indicate completion, because:
            // - If there were fewer than 512 entries, render_train() will have never been called
            // - One or more of the splits may have ignored later epochs, if accuracy went backwards
            // Because of this, we need a separate finished flag.
            progress.lock().unwrap().mark_finished();
        }
    };

    if let Some(config) = &training_config {
        validate_training_config(config).inspect_err(|_| finish_progress())?;
    }
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
        return Ok(initial_parameters.unwrap_or_else(|| match model_version {
            ComputeParametersVersion::Fsrs6 => FSRS6_DEFAULT_PARAMETERS.to_vec(),
            ComputeParametersVersion::Fsrs7 => DEFAULT_PARAMETERS.to_vec(),
        }));
    }

    let (mut initialized_parameters, fsrs6_initial_rating_count) = match model_version {
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
            initialized_parameters[23..31].copy_from_slice(&initial_forgetting_curve);
            (initialized_parameters, None)
        }
    };
    if let Some(initial_parameters) = initial_parameters {
        initialized_parameters = initial_parameters;
    }
    if train_set.len() == dataset_for_initialization.len() || train_set.len() < 64 {
        finish_progress();
        return Ok(initialized_parameters);
    }
    let mut config = InternalTrainingConfig::new(ModelConfig {
        freeze_initial_stability: !enable_short_term,
        initial_stability: None,
        initial_forgetting_curve: None,
        freeze_short_term_stability: !enable_short_term,
        num_relearning_steps: num_relearning_steps.unwrap_or(1),
    })
    .with_enable_sched_penalties(enable_sched_penalties);
    let training_objective = classifier_config.as_ref().map_or(
        TrainingObjective::ProbabilityLogLoss,
        |classifier_config| TrainingObjective::HardClassificationCost {
            costs: normalized_classification_costs(classifier_config.costs, &train_set),
            decision_boundary: classifier_config.decision_boundary,
            decision_sharpness: classifier_config.decision_sharpness,
        },
    );
    if model_version == ComputeParametersVersion::Fsrs7 {
        apply_windowed_fsrs7_training_tuning(&mut config, model_version, has_card_ids);
    }
    apply_training_config(&mut config, training_config);
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
    let optimized_parameters = train(
        weighted_train_set,
        &initialized_parameters,
        &config,
        training_objective,
        progress.clone().map(|p| ProgressCollector::new(p, 0)),
    )
    .inspect_err(|_e| {
        finish_progress();
    })?;

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
        card_ids,
        enable_short_term,
        enable_sched_penalties,
        model_version,
        num_relearning_steps,
        training_config,
        ..
    }: ComputeParametersInput,
) -> Vec<f32> {
    if let Some(config) = &training_config {
        validate_training_config(config).expect("invalid training configuration");
    }
    let train_set = normalize_for_model_version(train_set, model_version);
    let has_card_ids = card_ids.is_some();
    if card_ids
        .as_ref()
        .is_some_and(|ids| ids.len() != train_set.len())
    {
        panic!("card_ids must be aligned with train_set");
    }
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
            initialized_parameters[23..31].copy_from_slice(&initial_forgetting_curve);
            initialized_parameters
        }
    };
    let mut config = InternalTrainingConfig::new(ModelConfig {
        freeze_initial_stability: !enable_short_term,
        initial_stability: None,
        initial_forgetting_curve: None,
        freeze_short_term_stability: !enable_short_term,
        num_relearning_steps: num_relearning_steps.unwrap_or(1),
    })
    .with_enable_sched_penalties(enable_sched_penalties);
    apply_windowed_fsrs7_training_tuning(&mut config, model_version, has_card_ids);
    // save RAM and speed up training
    config.max_seq_len = 64;
    apply_training_config(&mut config, training_config);
    let mut weighted_train_set =
        recency_weighted_training_items(attach_card_ids(train_set, card_ids).unwrap());
    weighted_train_set.retain(|item| item.item.reviews.len() <= config.max_seq_len);
    train(
        weighted_train_set,
        &initialized_parameters,
        &config,
        TrainingObjective::ProbabilityLogLoss,
        None,
    )
    .unwrap()
}

#[derive(Debug, Clone)]
struct WindowedFSRSBatch {
    t_historys: Vec<f32>,
    r_historys: Vec<f32>,
    labels: Vec<f32>,
    weights: Vec<f32>,
    seq_len: usize,
    batch_size: usize,
    prediction_count: usize,
}

impl WindowedFSRSBatch {
    fn real_batch_size(&self) -> usize {
        self.prediction_count
    }

    fn analytic_bce_grad(&self, w: &[f32]) -> [f32; 34] {
        crate::analytic_v7::windowed_grad(
            w,
            &self.t_historys,
            &self.r_historys,
            &self.labels,
            &self.weights,
            self.seq_len,
            self.batch_size,
        )
    }

    fn analytic_bce_loss(&self, w: &[f32]) -> f64 {
        crate::analytic_v7::windowed_loss(
            w,
            &self.t_historys,
            &self.r_historys,
            &self.labels,
            &self.weights,
            self.seq_len,
            self.batch_size,
        )
    }
}

fn build_windowed_batches(items: &[WeightedFSRSItem], batch_size: usize) -> Vec<WindowedFSRSBatch> {
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
            batches.push(build_windowed_batch(&current));
            current.clear();
            current_predictions = 0;
        }
        current_predictions += prefixes.len();
        current.push(prefixes);
    }
    if !current.is_empty() {
        batches.push(build_windowed_batch(&current));
    }
    batches
}

fn build_windowed_batch(cards: &[Vec<&WeightedFSRSItem>]) -> WindowedFSRSBatch {
    let prediction_count = cards.iter().map(Vec::len).sum();
    let real_card_count = cards.len();
    let batch_size = real_card_count.div_ceil(8) * 8;
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

    WindowedFSRSBatch {
        t_historys,
        r_historys,
        labels,
        weights,
        seq_len,
        batch_size,
        prediction_count,
    }
}

#[derive(Clone)]
struct PrefixHostBatch {
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

fn build_plain_prefix_batch(items: &[WeightedFSRSItem]) -> PrefixHostBatch {
    let batch_size = items.len();
    let seq_len = items
        .iter()
        .map(|item| item.item.reviews.len() - 1)
        .max()
        .unwrap_or(0);
    let mut t_historys = vec![0.0; seq_len * batch_size];
    let mut r_historys = vec![0.0; seq_len * batch_size];
    let mut delta_ts = Vec::with_capacity(batch_size);
    let mut labels = Vec::with_capacity(batch_size);
    let mut weights = Vec::with_capacity(batch_size);
    let mut column_lengths = Vec::with_capacity(batch_size);
    for (column, weighted_item) in items.iter().enumerate() {
        column_lengths.push(weighted_item.item.reviews.len() - 1);
        for (row, review) in weighted_item.item.history().enumerate() {
            let index = row * batch_size + column;
            t_historys[index] = review.delta_t;
            r_historys[index] = review.rating as f32;
        }
        let current = weighted_item.item.current();
        delta_ts.push(current.delta_t);
        labels.push(f32::from(current.rating > 1));
        weights.push(weighted_item.weight);
    }
    PrefixHostBatch {
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

fn build_card_prefix_batch(cards: &[Vec<WeightedFSRSItem>]) -> PrefixHostBatch {
    let batch_size = cards.len();
    let seq_len = cards
        .iter()
        .map(|card| card.last().unwrap().item.reviews.len())
        .max()
        .unwrap_or(0);
    let real_batch_size = cards.iter().map(Vec::len).sum();
    let mut t_historys = vec![0.0; seq_len * batch_size];
    let mut r_historys = vec![0.0; seq_len * batch_size];
    let mut labels = vec![0.0; seq_len * batch_size];
    let mut weights = vec![0.0; seq_len * batch_size];
    let mut column_lengths = Vec::with_capacity(batch_size);
    for (column, card) in cards.iter().enumerate() {
        let reviews = &card.last().unwrap().item.reviews;
        column_lengths.push(reviews.len());
        for (row, review) in reviews.iter().enumerate() {
            let index = row * batch_size + column;
            t_historys[index] = review.delta_t;
            r_historys[index] = review.rating as f32;
        }
        for item in card {
            let row = item.item.reviews.len() - 1;
            let index = row * batch_size + column;
            labels[index] = f32::from(item.item.current().rating > 1);
            weights[index] = item.weight;
        }
    }
    PrefixHostBatch {
        seq_len,
        batch_size,
        real_batch_size,
        column_lengths,
        t_historys,
        r_historys,
        delta_ts: Vec::new(),
        labels,
        weights,
        windowed: true,
    }
}

fn build_fsrs6_batches(
    mut items: Vec<WeightedFSRSItem>,
    batch_size: usize,
) -> Vec<PrefixHostBatch> {
    if items.iter().all(|item| item.card_id.is_none()) {
        items.sort_by_cached_key(|item| item.item.reviews.len());
        return items
            .chunks(batch_size)
            .map(build_plain_prefix_batch)
            .collect();
    }
    let mut grouped = BTreeMap::<i64, Vec<WeightedFSRSItem>>::new();
    for item in items {
        grouped
            .entry(item.card_id.expect("checked card id"))
            .or_default()
            .push(item);
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
    let mut current = Vec::new();
    let mut predictions = 0;
    for card in cards {
        if !current.is_empty() && predictions + card.len() > batch_size {
            batches.push(build_card_prefix_batch(&current));
            current.clear();
            predictions = 0;
        }
        predictions += card.len();
        current.push(card);
    }
    if !current.is_empty() {
        batches.push(build_card_prefix_batch(&current));
    }
    batches
}

fn objective_weights(
    objective: TrainingObjective,
    labels: &[f32],
    weights: &[f32],
    predictions: &[f32],
) -> Vec<f32> {
    match objective {
        TrainingObjective::ProbabilityLogLoss => weights.to_vec(),
        TrainingObjective::HardClassificationCost {
            costs,
            decision_boundary,
            decision_sharpness,
        } => labels
            .iter()
            .zip(weights)
            .zip(predictions)
            .map(|((&label, &weight), &prediction)| {
                let prediction = prediction.clamp(0.0001, 0.9999);
                let soft =
                    1.0 / (1.0 + (-(prediction - decision_boundary) * decision_sharpness).exp());
                let slope = decision_sharpness * soft * (1.0 - soft);
                if label > 0.5 {
                    weight * costs.false_negative * slope * prediction
                } else {
                    weight * costs.false_positive * slope * (1.0 - prediction)
                }
            })
            .collect(),
    }
}

fn windowed_predictions(
    parameters: &[f32],
    t: &[f32],
    ratings: &[f32],
    weights: &[f32],
    seq_len: usize,
    batch_size: usize,
) -> Vec<f32> {
    let fsrs = FSRS::new(parameters).expect("training parameters are valid");
    let mut predictions = vec![0.5; weights.len()];
    for column in 0..batch_size {
        let mut state = crate::MemoryState {
            stability: 0.0,
            difficulty: 0.0,
            stability_fast: 0.0,
        };
        for row in 0..seq_len {
            let index = row * batch_size + column;
            if row > 0 && weights[index] != 0.0 {
                predictions[index] = fsrs.power_forgetting_curve_for_state(t[index], state);
            }
            state = fsrs.step(t[index], ratings[index] as u32, state, row);
        }
    }
    predictions
}

fn prefix_predictions(batch: &PrefixHostBatch, parameters: &[f32]) -> Vec<f32> {
    if batch.windowed {
        return windowed_predictions(
            parameters,
            &batch.t_historys,
            &batch.r_historys,
            &batch.weights,
            batch.seq_len,
            batch.batch_size,
        );
    }
    let fsrs = FSRS::new(parameters).expect("training parameters are valid");
    (0..batch.batch_size)
        .map(|column| {
            let reviews = (0..batch.column_lengths[column])
                .map(|row| crate::FSRSReview {
                    delta_t: batch.t_historys[row * batch.batch_size + column],
                    rating: batch.r_historys[row * batch.batch_size + column] as u32,
                })
                .collect::<Vec<_>>();
            let state = fsrs.forward_reviews(&reviews, None);
            fsrs.power_forgetting_curve_for_state(batch.delta_ts[column], state)
        })
        .collect()
}

fn fsrs6_batch_grad(
    batch: &PrefixHostBatch,
    parameters: &[f32],
    objective: TrainingObjective,
) -> Vec<f32> {
    let predictions = prefix_predictions(batch, parameters);
    let weights = objective_weights(objective, &batch.labels, &batch.weights, &predictions);
    let mut grad = vec![0.0f64; parameters.len()];
    if batch.windowed {
        crate::analytic_v6::card_loss_and_grad(
            parameters,
            &batch.t_historys,
            &batch.r_historys,
            batch.seq_len,
            batch.batch_size,
            &batch.column_lengths,
            &batch.labels,
            &weights,
            &mut grad,
        );
    } else {
        crate::analytic_v6::batch_loss_and_grad(
            parameters,
            &batch.t_historys,
            &batch.r_historys,
            batch.seq_len,
            batch.batch_size,
            &batch.column_lengths,
            &batch.delta_ts,
            &batch.labels,
            &weights,
            &mut grad,
        );
    }
    grad.into_iter().map(|value| value as f32).collect()
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

#[derive(Debug, Clone)]
struct HostAdam {
    moment_1: Vec<f32>,
    moment_2: Vec<f32>,
    time: i32,
}

impl HostAdam {
    fn new(parameter_len: usize) -> Self {
        Self {
            moment_1: vec![0.0; parameter_len],
            moment_2: vec![0.0; parameter_len],
            time: 0,
        }
    }

    fn step(&mut self, parameters: &mut [f32], grad: &[f32], lr: f64) {
        self.time += 1;
        let m_correction = 1.0 - ADAM_BETA_1.powi(self.time);
        let v_correction = 1.0 - ADAM_BETA_2.powi(self.time);
        for ((parameter, (moment_1, moment_2)), &grad) in parameters
            .iter_mut()
            .zip(self.moment_1.iter_mut().zip(&mut self.moment_2))
            .zip(grad)
        {
            *moment_1 = *moment_1 * ADAM_BETA_1 + grad * (1.0 - ADAM_BETA_1);
            *moment_2 = *moment_2 * ADAM_BETA_2 + grad.powi(2) * (1.0 - ADAM_BETA_2);
            *parameter -= lr as f32 * (*moment_1 / m_correction)
                / ((*moment_2 / v_correction).sqrt() + ADAM_EPSILON);
        }
    }
}

fn zero_frozen_host_grad(grad: &mut [f32], model: &ModelConfig) {
    if model.freeze_initial_stability {
        grad.iter_mut().take(4).for_each(|value| *value = 0.0);
    }
    if model.freeze_short_term_stability && grad.len() == 21 {
        grad[17..20].fill(0.0);
    }
}

fn clip_host_parameters(
    parameters: &mut [f32],
    num_relearning_steps: usize,
    enable_short_term: bool,
) {
    clip_parameters_in_place(parameters, num_relearning_steps, enable_short_term);
    if !enable_short_term && parameters.len() == 34 {
        parameters[26] = 0.0;
    }
}

fn train(
    mut train_set: Vec<WeightedFSRSItem>,
    initial_parameters: &[f32],
    config: &InternalTrainingConfig,
    objective: TrainingObjective,
    progress: Option<ProgressCollector>,
) -> Result<Vec<f32>> {
    let version = ModelVersion::from_param_count(initial_parameters.len());
    if version == ModelVersion::Fsrs7 {
        for (index, item) in train_set.iter_mut().enumerate() {
            if item.card_id.is_none() {
                item.card_id = Some(i64::MIN + index as i64);
            }
        }
    }
    let total_size = train_set.len();
    let iterations = (total_size / config.batch_size + 1) * config.num_epochs;
    let fsrs7_batches = (version == ModelVersion::Fsrs7)
        .then(|| build_windowed_batches(&train_set, config.batch_size));
    let fsrs6_batches =
        (version == ModelVersion::Fsrs6).then(|| build_fsrs6_batches(train_set, config.batch_size));
    let batch_count = fsrs7_batches
        .as_ref()
        .map_or_else(|| fsrs6_batches.as_ref().unwrap().len(), Vec::len);
    let mut parameters = initial_parameters.to_vec();
    let initial = parameters.clone();
    let mut adam = HostAdam::new(parameters.len());
    let mut scheduler = CosineAnnealingLR::init(iterations as f64, config.learning_rate);
    let mut rng = StdRng::seed_from_u64(config.seed);
    let mut order = (0..batch_count).collect::<Vec<_>>();
    let mut progress = progress;

    for epoch in 1..=config.num_epochs {
        order.shuffle(&mut rng);
        let mut processed = 0;
        for &index in &order {
            let real_batch_size = if let Some(batches) = &fsrs7_batches {
                batches[index].real_batch_size()
            } else {
                fsrs6_batches.as_ref().unwrap()[index].real_batch_size
            };
            let mut grad = if let Some(batches) = &fsrs7_batches {
                let batch = &batches[index];
                if matches!(objective, TrainingObjective::HardClassificationCost { .. }) {
                    let mut adjusted = batch.clone();
                    let predictions = windowed_predictions(
                        &parameters,
                        &batch.t_historys,
                        &batch.r_historys,
                        &batch.weights,
                        batch.seq_len,
                        batch.batch_size,
                    );
                    adjusted.weights =
                        objective_weights(objective, &batch.labels, &batch.weights, &predictions);
                    adjusted.analytic_bce_grad(&parameters).to_vec()
                } else {
                    batch.analytic_bce_grad(&parameters).to_vec()
                }
            } else {
                fsrs6_batch_grad(
                    &fsrs6_batches.as_ref().unwrap()[index],
                    &parameters,
                    objective,
                )
            };
            let l2_weight = L2_PENALTY_WEIGHT * config.gamma;
            let (_, l2_grad) = l2_penalty_fn(version)(
                &parameters,
                &initial,
                real_batch_size,
                total_size,
                l2_weight,
                &training_v7::PARAMS_STDDEV,
            );
            let (_, schedule_grad) = schedule_penalty_fn(version)(
                &parameters,
                real_batch_size,
                config.enable_sched_penalties,
            );
            for (index, value) in grad.iter_mut().enumerate() {
                *value += l2_grad.get(index).copied().unwrap_or(0.0)
                    + (schedule_grad.get(index).copied().unwrap_or(0.0) / total_size as f64) as f32;
            }
            zero_frozen_host_grad(&mut grad, &config.model);
            adam.step(&mut parameters, &grad, scheduler.step());
            clip_host_parameters(
                &mut parameters,
                config.model.num_relearning_steps,
                !config.model.freeze_short_term_stability,
            );
            processed += real_batch_size;
            if !render_progress(
                &mut progress,
                epoch,
                config.num_epochs,
                processed.min(total_size),
                total_size,
            ) {
                return Err(FSRSError::Interrupted);
            }
        }
    }
    Ok(parameters)
}

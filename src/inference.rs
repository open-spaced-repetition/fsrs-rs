use itertools::izip;
use serde::Serialize;
use std::collections::HashMap;
use std::ops::{Add, Sub};
use std::sync::{Arc, Mutex};

use crate::dataset::{
    WeightedFSRSItem, constant_weighted_fsrs_items, recency_weighted_fsrs_items,
    recency_weighted_fsrs_items_with_card_ids,
};
use crate::error::Result;
use crate::model::FSRS;
use crate::simulation::S_MIN;
use crate::training::weighted_binary_cross_entropy;
use crate::training::{self, ComputeParametersInput};
use crate::{FSRSError, FSRSItem};
#[path = "inference_v6.rs"]
pub(crate) mod inference_v6;
#[path = "inference_v7.rs"]
pub(crate) mod inference_v7;

pub use inference_v6::{FSRS5_DEFAULT_DECAY, FSRS6_DEFAULT_DECAY, FSRS6_DEFAULT_PARAMETERS};
pub use inference_v7::DEFAULT_PARAMETERS;

/// This is a slice for efficiency, and may be 17, 19, 21, or 34 values long.
pub type Parameters = [f32];
type SharedTrainingProgress = Arc<Mutex<training::CombinedProgressState>>;

/// Calculate the current retrievability for a memory state.
///
/// # Arguments
/// * `state` - The memory state
/// * `days_elapsed` - Days since last review
/// * `decay` - Decay parameter for the forgetting curve
///
/// # Returns
/// The retrievability (probability of recall)
pub fn current_retrievability(state: MemoryState, days_elapsed: f32, decay: f32) -> f32 {
    let factor = 0.9f32.powf(1.0 / -decay) - 1.0;
    (days_elapsed / state.stability * factor + 1.0).powf(-decay)
}

/// Represents the memory state of an item in the FSRS system.
#[derive(Debug, PartialEq, Clone, Copy, Serialize)]
pub struct MemoryState {
    /// The stability of the memory state
    pub stability: f32,
    /// The difficulty of the memory state
    pub difficulty: f32,
    /// The fast stability trace used by FSRS-7. For FSRS-6 it equals `stability`.
    pub stability_fast: f32,
}

#[derive(Default)]
struct RMatrixValue {
    predicted: f32,
    actual: f32,
    count: f32,
    weight: f32,
}

#[derive(Default)]
struct SplitEvaluation {
    predictions: Vec<f32>,
    labels: Vec<f32>,
    weights: Vec<f32>,
    r_matrix: HashMap<(u32, u32, u32), RMatrixValue>,
}

fn validate_state(state: MemoryState) -> Result<MemoryState> {
    if !state.stability.is_finite() || !state.difficulty.is_finite() {
        Err(FSRSError::InvalidInput)
    } else {
        Ok(state)
    }
}

fn predict_retrievability(fsrs: &FSRS, item: &FSRSItem) -> Result<f32> {
    if item.reviews.is_empty() {
        return Err(FSRSError::InvalidInput);
    }
    let history_len = item.reviews.len().saturating_sub(1);
    let state = fsrs.forward_reviews(&item.reviews[..history_len], None);
    let current = item.current();
    let retrievability = fsrs.power_forgetting_curve_for_state(current.delta_t, state);
    if retrievability.is_finite() {
        Ok(retrievability)
    } else {
        Err(FSRSError::InvalidInput)
    }
}

fn rmse_bins(r_matrix: &HashMap<(u32, u32, u32), RMatrixValue>) -> f32 {
    (r_matrix
        .values()
        .map(|v| {
            let pred = v.predicted / v.count;
            let real = v.actual / v.count;
            (pred - real).powi(2) * v.weight
        })
        .sum::<f32>()
        / r_matrix.values().map(|v| v.weight).sum::<f32>())
    .sqrt()
}

fn evaluate_time_series_split(
    split: TimeSeriesSplit,
    enable_short_term: bool,
    num_relearning_steps: Option<usize>,
    training_config: Option<training::TrainingConfig>,
    progress: Option<SharedTrainingProgress>,
) -> Result<SplitEvaluation> {
    if progress
        .as_ref()
        .is_some_and(|progress| progress.lock().unwrap().want_abort)
    {
        return Err(FSRSError::Interrupted);
    }

    let input = ComputeParametersInput {
        train_set: split.train_items,
        card_ids: split.train_card_ids,
        enable_short_term,
        num_relearning_steps,
        training_config,
        progress: progress.clone(),
        ..Default::default()
    };
    let parameters = training::compute_parameters(input)?;

    if progress
        .as_ref()
        .is_some_and(|progress| progress.lock().unwrap().want_abort)
    {
        return Err(FSRSError::Interrupted);
    }

    let fsrs = FSRS::new(&parameters)?;
    let mut evaluation = SplitEvaluation {
        predictions: Vec::with_capacity(split.test_items.len()),
        labels: Vec::with_capacity(split.test_items.len()),
        weights: Vec::with_capacity(split.test_items.len()),
        r_matrix: HashMap::new(),
    };

    for item in &split.test_items {
        let pred = predict_retrievability(&fsrs, item)?;
        let label = if item.current().rating > 1 { 1.0 } else { 0.0 };
        let bin = item.r_matrix_index();
        let value = evaluation.r_matrix.entry(bin).or_default();
        value.predicted += pred;
        value.actual += label;
        value.count += 1.0;
        value.weight += 1.0;
        evaluation.predictions.push(pred);
        evaluation.labels.push(label);
        evaluation.weights.push(1.0);
    }

    Ok(evaluation)
}

fn merge_split_evaluation(
    evaluation: SplitEvaluation,
    predictions: &mut Vec<f32>,
    labels: &mut Vec<f32>,
    weights: &mut Vec<f32>,
    r_matrix: &mut HashMap<(u32, u32, u32), RMatrixValue>,
) {
    predictions.extend(evaluation.predictions);
    labels.extend(evaluation.labels);
    weights.extend(evaluation.weights);
    for (bin, value) in evaluation.r_matrix {
        let aggregate = r_matrix.entry(bin).or_default();
        aggregate.predicted += value.predicted;
        aggregate.actual += value.actual;
        aggregate.count += value.count;
        aggregate.weight += value.weight;
    }
}

fn abort_training(progresses: &[SharedTrainingProgress]) {
    for progress in progresses {
        progress.lock().unwrap().want_abort = true;
    }
}

impl FSRS {
    /// Calculate the current memory state for a given card's history of reviews.
    /// In the case of truncated reviews, `starting_state` can be set to the value of
    /// [FSRS::memory_state_from_sm2] for the first review (which should not be included
    /// in FSRSItem). If not provided, the card starts as new.
    pub fn memory_state(
        &self,
        item: FSRSItem,
        starting_state: Option<MemoryState>,
    ) -> Result<MemoryState> {
        validate_state(self.forward_reviews(&item.reviews, starting_state))
    }

    pub fn memory_state_batch(
        &self,
        items: Vec<FSRSItem>,
        starting_states: Vec<Option<MemoryState>>,
    ) -> Result<Vec<MemoryState>> {
        if items.is_empty() {
            return Ok(vec![]);
        }
        if items.len() != starting_states.len() {
            return Err(FSRSError::InvalidInput);
        }
        if items.iter().all(|item| item == &items[0])
            && starting_states
                .iter()
                .all(|state| state == &starting_states[0])
        {
            let state = self.memory_state(items[0].clone(), starting_states[0])?;
            return Ok(vec![state; items.len()]);
        }
        items
            .into_iter()
            .zip(starting_states)
            .map(|(item, starting_state)| self.memory_state(item, starting_state))
            .collect()
    }

    pub fn historical_memory_states(
        &self,
        item: FSRSItem,
        starting_state: Option<MemoryState>,
    ) -> Result<Vec<MemoryState>> {
        let mut states = vec![];
        if let Some(starting_state) = starting_state {
            states.push(starting_state);
        }
        let mut inner_state = if let Some(state) = starting_state {
            state
        } else {
            MemoryState {
                stability: 0.0,
                difficulty: 0.0,
                stability_fast: 0.0,
            }
        };
        for (index, review) in item.reviews.iter().enumerate() {
            inner_state = self.step(review.delta_t, review.rating, inner_state, index);
            states.push(validate_state(inner_state)?);
        }
        Ok(states)
    }

    pub fn historical_memory_state_batch(
        &self,
        items: Vec<FSRSItem>,
        starting_states: Option<Vec<Option<MemoryState>>>,
    ) -> Result<Vec<Vec<MemoryState>>> {
        let starting_states = starting_states.unwrap_or((0..items.len()).map(|_| None).collect());
        if items.is_empty() {
            return Ok(vec![]);
        }
        if items.len() != starting_states.len() {
            return Err(FSRSError::InvalidInput);
        }
        items
            .into_iter()
            .zip(starting_states)
            .map(|(item, starting_state)| self.historical_memory_states(item, starting_state))
            .collect()
    }

    /// If a card has incomplete learning history, memory state can be approximated from
    /// current sm2 values.
    pub fn memory_state_from_sm2(
        &self,
        ease_factor: f32,
        interval: f32,
        sm2_retention: f32,
    ) -> Result<MemoryState> {
        self.memory_state_from_sm2_inner(ease_factor, interval, sm2_retention)
    }

    /// Calculate current retrievability using the active FSRS version.
    pub fn current_retrievability(&self, state: MemoryState, days_elapsed: f32) -> f32 {
        self.power_forgetting_curve_for_state(days_elapsed.max(0.0), state)
    }

    /// Return the S90 interval for this state.
    pub fn s90(&self, state: MemoryState) -> f32 {
        self.interval_at_retrievability(state, 0.9)
    }

    /// Calculate the next interval for the current memory state, for rescheduling. Stability
    /// should be provided except when the card is new. Rating is ignored except when card is new.
    pub fn next_interval(
        &self,
        stability: Option<f32>,
        desired_retention: f32,
        rating: u32,
    ) -> f32 {
        let stability = stability.unwrap_or_else(|| {
            // get initial stability for new card
            self.init_stability(rating)
        });
        self.next_interval_for_state(
            MemoryState {
                stability,
                difficulty: 5.0,
                stability_fast: stability,
            },
            desired_retention,
        )
    }

    /// The intervals and memory states for each answer button.
    ///
    /// Returns a [`NextStates`] struct containing the intervals and memory states for each answer button.
    ///
    /// # Examples
    /// ```
    /// use fsrs::{FSRS, MemoryState, ItemState, NextStates};
    ///
    /// let fsrs = FSRS::default();
    /// let desired_retention = 0.9;
    /// let previous_state: Option<MemoryState> = None;
    /// let elapsed_days = 0;
    ///
    /// let next_states = fsrs.next_states(previous_state, desired_retention, elapsed_days).unwrap();
    /// assert_eq!(
    ///     next_states,
    ///     NextStates {
    ///         again: ItemState { memory: MemoryState { stability: 0.1104, difficulty: 6.1686, stability_fast: 0.08832 }, interval: 3.8275626e-5 },
    ///         hard: ItemState { memory: MemoryState { stability: 2.2395, difficulty: 5.261278, stability_fast: 1.7916001 }, interval: 0.5974598 },
    ///         good: ItemState { memory: MemoryState { stability: 3.9221, difficulty: 3.5307243, stability_fast: 3.13768 }, interval: 4.7777247 },
    ///         easy: ItemState { memory: MemoryState { stability: 11.7841, difficulty: 1.0, stability_fast: 9.427279 }, interval: 53.869392 }
    ///     }
    /// );
    /// ```
    pub fn next_states(
        &self,
        current_memory_state: Option<MemoryState>,
        desired_retention: f32,
        days_elapsed: u32,
    ) -> Result<NextStates> {
        self.next_states_inner(current_memory_state, desired_retention, days_elapsed as f32)
    }

    /// Like [`Self::next_states`], but accepts fractional elapsed days.
    pub fn next_states_with_elapsed_days(
        &self,
        current_memory_state: Option<MemoryState>,
        desired_retention: f32,
        days_elapsed: f32,
    ) -> Result<NextStates> {
        self.next_states_inner(current_memory_state, desired_retention, days_elapsed)
    }

    fn next_states_inner(
        &self,
        current_memory_state: Option<MemoryState>,
        desired_retention: f32,
        days_elapsed: f32,
    ) -> Result<NextStates> {
        let (current_memory_state, nth) = if let Some(state) = current_memory_state {
            (state, 1)
        } else {
            (
                MemoryState {
                    stability: 0.0,
                    difficulty: 0.0,
                    stability_fast: 0.0,
                },
                0,
            )
        };
        let mut next_memory_states = (1..=4).map(|rating| {
            validate_state(self.step(days_elapsed, rating, current_memory_state, nth))
        });

        let mut get_next_state = || {
            let memory = next_memory_states.next().unwrap()?;
            let interval = self.next_interval_for_state(memory, desired_retention);
            Ok(ItemState { memory, interval })
        };

        Ok(NextStates {
            again: get_next_state()?,
            hard: get_next_state()?,
            good: get_next_state()?,
            easy: get_next_state()?,
        })
    }

    /// Determine how well the model and parameters predict performance.
    pub fn evaluate<F>(&self, items: Vec<FSRSItem>, mut progress: F) -> Result<ModelEvaluation>
    where
        F: FnMut(ItemProgress) -> bool,
    {
        if items.is_empty() {
            return Err(FSRSError::NotEnoughData);
        }
        let weighted_items = recency_weighted_fsrs_items(items);
        let mut predictions = Vec::with_capacity(weighted_items.len());
        let mut labels = Vec::with_capacity(weighted_items.len());
        let mut weights = Vec::with_capacity(weighted_items.len());
        let mut progress_info = ItemProgress {
            current: 0,
            total: weighted_items.len(),
        };
        let mut r_matrix: HashMap<(u32, u32, u32), RMatrixValue> = HashMap::new();

        for chunk in weighted_items.chunks(512) {
            for weighted_item in chunk {
                let p = predict_retrievability(self, &weighted_item.item)?;
                let y = f32::from(weighted_item.item.current().rating > 1);
                let bin = weighted_item.item.r_matrix_index();
                let value = r_matrix.entry(bin).or_default();
                value.predicted += p;
                value.actual += y;
                value.count += 1.0;
                value.weight += weighted_item.weight;
                predictions.push(p);
                labels.push(y);
                weights.push(weighted_item.weight);
            }
            progress_info.current += chunk.len();
            if !progress(progress_info) {
                return Err(FSRSError::Interrupted);
            }
        }
        let rmse = rmse_bins(&r_matrix);
        let loss = weighted_binary_cross_entropy(&predictions, &labels, &weights);
        if !loss.is_finite() || !rmse.is_finite() {
            return Err(FSRSError::InvalidInput);
        }
        Ok(ModelEvaluation {
            log_loss: loss,
            rmse_bins: rmse,
        })
    }

    /// Like [`Self::evaluate`], but additionally takes `card_ids` aligned with `items` (the same
    /// convention as [`ComputeParametersInput::card_ids`]): items from the same card are
    /// expanding-window prefixes of one review history, so each card's memory-state trajectory
    /// is computed ONCE and every item's retrievability is read off along the way. This turns
    /// the model work from O(sum of prefix lengths) into O(total reviews), while producing the
    /// exact same per-item predictions and log loss as [`Self::evaluate`] (identical arithmetic
    /// in identical accumulation order; RMSE matches up to the last float bits, since its bin
    /// map is summed in `HashMap` iteration order, which differs between any two calls). A card
    /// whose items do not form a prefix chain falls back to the per-item path, so the result is
    /// equivalent for arbitrary inputs.
    ///
    /// Callers that already build `card_ids` for [`crate::compute_parameters`] can pass the
    /// same vector here (e.g. Anki, which evaluates both the current and the optimized
    /// parameters right after optimizing).
    pub fn evaluate_with_card_ids<F>(
        &self,
        items: Vec<FSRSItem>,
        card_ids: Vec<i64>,
        mut progress: F,
    ) -> Result<ModelEvaluation>
    where
        F: FnMut(ItemProgress) -> bool,
    {
        if items.is_empty() {
            return Err(FSRSError::NotEnoughData);
        }
        if card_ids.len() != items.len() {
            return Err(FSRSError::InvalidInput);
        }
        let weighted_items = recency_weighted_fsrs_items_with_card_ids(items, card_ids);

        // Group item indices by card id, in first-appearance order.
        let mut group_index: HashMap<i64, usize> = HashMap::new();
        let mut groups: Vec<Vec<usize>> = Vec::new();
        for (idx, weighted_item) in weighted_items.iter().enumerate() {
            let group = *group_index
                .entry(weighted_item.card_id.unwrap())
                .or_insert_with(|| {
                    groups.push(Vec::new());
                    groups.len() - 1
                });
            groups[group].push(idx);
        }

        let mut predictions = vec![0.0f32; weighted_items.len()];
        let mut progress_info = ItemProgress {
            current: 0,
            total: weighted_items.len(),
        };
        for group in &groups {
            self.predict_card_group(&weighted_items, group, &mut predictions)?;
            progress_info.current += group.len();
            if !progress(progress_info) {
                return Err(FSRSError::Interrupted);
            }
        }

        // Aggregate in the original item order — the same arithmetic in the same accumulation
        // order as `evaluate`, so the resulting metrics are bit-for-bit identical.
        let mut r_matrix: HashMap<(u32, u32, u32), RMatrixValue> = HashMap::new();
        let mut preds = Vec::with_capacity(weighted_items.len());
        let mut labels = Vec::with_capacity(weighted_items.len());
        let mut weights = Vec::with_capacity(weighted_items.len());
        for (idx, weighted_item) in weighted_items.iter().enumerate() {
            let p = predictions[idx];
            let y = f32::from(weighted_item.item.current().rating > 1);
            let bin = weighted_item.item.r_matrix_index();
            let value = r_matrix.entry(bin).or_default();
            value.predicted += p;
            value.actual += y;
            value.count += 1.0;
            value.weight += weighted_item.weight;
            preds.push(p);
            labels.push(y);
            weights.push(weighted_item.weight);
        }
        let rmse = rmse_bins(&r_matrix);
        let loss = weighted_binary_cross_entropy(&preds, &labels, &weights);
        if !loss.is_finite() || !rmse.is_finite() {
            return Err(FSRSError::InvalidInput);
        }
        Ok(ModelEvaluation {
            log_loss: loss,
            rmse_bins: rmse,
        })
    }

    /// Fill `predictions` for one card's items. If the items form an expanding-window prefix
    /// chain (every shorter history is a prefix of the longest one), the card's state
    /// trajectory is walked once using the exact per-review steps `forward_reviews` would
    /// take; otherwise every item falls back to [`predict_retrievability`].
    fn predict_card_group(
        &self,
        weighted_items: &[WeightedFSRSItem],
        group: &[usize],
        predictions: &mut [f32],
    ) -> Result<()> {
        let mut order = group.to_vec();
        order.sort_by_key(|&idx| weighted_items[idx].item.reviews.len());
        let longest = &weighted_items[*order.last().unwrap()].item.reviews;
        let chain = order.iter().all(|&idx| {
            let reviews = &weighted_items[idx].item.reviews;
            longest[..reviews.len()] == reviews[..]
        });
        if !chain {
            for &idx in group {
                predictions[idx] = predict_retrievability(self, &weighted_items[idx].item)?;
            }
            return Ok(());
        }
        let mut state: Option<MemoryState> = None;
        let mut applied = 0;
        for &idx in &order {
            let reviews = &weighted_items[idx].item.reviews;
            if reviews.is_empty() {
                return Err(FSRSError::InvalidInput);
            }
            let history_len = reviews.len() - 1;
            while applied < history_len {
                let review = &longest[applied];
                state = Some(match state {
                    None => self.init_state_from_first_review(review),
                    Some(state) => self.step(review.delta_t, review.rating, state, applied),
                });
                applied += 1;
            }
            let state = state.unwrap_or(MemoryState {
                stability: 0.0,
                difficulty: 0.0,
                stability_fast: 0.0,
            });
            let retrievability =
                self.power_forgetting_curve_for_state(reviews[history_len].delta_t, state);
            if !retrievability.is_finite() {
                return Err(FSRSError::InvalidInput);
            }
            predictions[idx] = retrievability;
        }
        Ok(())
    }

    /// Returns the universal metrics for the existing and provided parameters. If the first value
    /// is smaller than the second value, the existing parameters are better than the provided ones.
    pub fn universal_metrics<F>(
        &self,
        items: Vec<FSRSItem>,
        parameters: &Parameters,
        mut progress: F,
    ) -> Result<(f32, f32)>
    where
        F: FnMut(ItemProgress) -> bool,
    {
        if items.is_empty() {
            return Err(FSRSError::NotEnoughData);
        }
        let weighted_items = constant_weighted_fsrs_items(items);
        let mut all_predictions_self = vec![];
        let mut all_predictions_other = vec![];
        let mut all_true_val = vec![];
        let mut progress_info = ItemProgress {
            current: 0,
            total: weighted_items.len(),
        };
        let fsrs_other = Self::new(parameters)?;
        for chunk in weighted_items.chunks(512) {
            for weighted_item in chunk {
                all_predictions_self.push(predict_retrievability(self, &weighted_item.item)?);
                all_predictions_other
                    .push(predict_retrievability(&fsrs_other, &weighted_item.item)?);
                all_true_val.push(f32::from(weighted_item.item.current().rating > 1));
            }
            progress_info.current += chunk.len();
            if !progress(progress_info) {
                return Err(FSRSError::Interrupted);
            }
        }
        let self_by_other =
            measure_a_by_b(&all_predictions_self, &all_predictions_other, &all_true_val);
        let other_by_self =
            measure_a_by_b(&all_predictions_other, &all_predictions_self, &all_true_val);
        Ok((self_by_other, other_by_self))
    }
}

/// The evaluation metrics of a model.
#[derive(Debug, Copy, Clone)]
pub struct ModelEvaluation {
    /// The accuracy of the model's predicted probabilities
    pub log_loss: f32,
    /// Whether the model's predicted probability matches the actual recall rate.
    pub rmse_bins: f32,
}

/// The next states of an item after a review.
///
/// It contains the states for each user choice after a review.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct NextStates {
    /// The state of the item after a "again" review.
    pub again: ItemState,
    /// The state of the item after a "hard" review.
    pub hard: ItemState,
    /// The state of the item after a "good" review.
    pub good: ItemState,
    /// The state of the item after an "easy" review.
    pub easy: ItemState,
}

/// The state of an item after a review.
///
/// It contains the memory state and the interval from after a review.
#[derive(Debug, PartialEq, Clone, Serialize)]
pub struct ItemState {
    /// The memory state of the item after a review.
    pub memory: MemoryState,
    /// The interval after a review.
    pub interval: f32,
}

/// The learning progress.
#[derive(Debug, Clone, Copy)]
pub struct ItemProgress {
    /// The current number of reviews.
    pub current: usize,
    /// The total number of reviews.
    pub total: usize,
}

#[derive(Debug, Clone)]
pub struct TimeSeriesSplit {
    pub train_items: Vec<FSRSItem>,
    pub train_card_ids: Option<Vec<i64>>,
    pub test_items: Vec<FSRSItem>,
}

impl TimeSeriesSplit {
    /// Split the dataset into training and validation sets based on time order.
    /// Creates n_splits folds where each fold's test set is a single segment,
    /// and the training set consists of all segments before the test segment.
    ///
    /// For example, with n_splits=5, the folds would be:
    /// Fold 0: Train=[0], Test=[1]
    /// Fold 1: Train=[0,1], Test=[2]
    /// Fold 2: Train=[0,1,2], Test=[3]
    /// Fold 3: Train=[0,1,2,3], Test=[4]
    /// Fold 4: Train=[0,1,2,3,4], Test=[5]
    ///
    /// # Arguments
    /// * `sorted_items` - The dataset to split, assumed to be in time order
    /// * `n_splits` - Number of splits to create
    ///
    /// # Returns
    /// A vector of TimeSeriesSplit, each containing train and validation items
    pub fn split(sorted_items: Vec<FSRSItem>, n_splits: usize) -> Vec<TimeSeriesSplit> {
        Self::split_with_card_ids(sorted_items, None, n_splits)
    }

    fn split_with_card_ids(
        sorted_items: Vec<FSRSItem>,
        card_ids: Option<Vec<i64>>,
        n_splits: usize,
    ) -> Vec<TimeSeriesSplit> {
        if sorted_items.is_empty() || n_splits == 0 {
            return vec![];
        }
        let total_items = sorted_items.len();
        let segment_size = total_items / (n_splits + 1);
        if segment_size == 0 {
            return vec![];
        }

        (0..n_splits)
            .map(|i| {
                // Calculate the start of the test segment
                let test_start = (i + 1) * segment_size;
                // Calculate the end of the test segment (or the end of the data)
                let test_end = if i == n_splits - 1 {
                    total_items
                } else {
                    (i + 2) * segment_size
                };

                // Create the split
                TimeSeriesSplit {
                    train_items: sorted_items[..test_start].to_vec(),
                    train_card_ids: card_ids
                        .as_ref()
                        .map(|card_ids| card_ids[..test_start].to_vec()),
                    test_items: sorted_items[test_start..test_end].to_vec(),
                }
            })
            .collect()
    }
}
/// Get a binned index for comparison calculations
fn get_bin(x: f32, bins: i32) -> i32 {
    let log_base = (bins.add(1) as f32).ln();
    let binned_x = (x * log_base).exp().floor().sub(1.0);
    (binned_x as i32).clamp(0, bins - 1)
}

/// Evaluates the model using time series cross-validation.
///
/// This function performs time series cross-validation by splitting the dataset into training
/// and testing sets based on time order. It trains the model on each training set and evaluates
/// it on the corresponding test set.
///
/// # Arguments
/// * `input` - Input parameters including the dataset and configuration
/// * `progress` - A function to report progress
///
/// # Returns
/// A `Result<ModelEvaluation>` containing the evaluation metrics
pub fn evaluate_with_time_series_splits<F>(
    ComputeParametersInput {
        train_set,
        card_ids,
        enable_short_term,
        num_relearning_steps,
        training_config,
        ..
    }: ComputeParametersInput,
    mut progress: F,
) -> Result<ModelEvaluation>
where
    F: FnMut(ItemProgress) -> bool,
{
    if train_set.is_empty() {
        return Err(FSRSError::NotEnoughData);
    }
    if let Some(card_ids) = &card_ids
        && card_ids.len() != train_set.len()
    {
        return Err(FSRSError::InvalidInput);
    }

    let splits = match card_ids {
        Some(card_ids) => TimeSeriesSplit::split_with_card_ids(train_set, Some(card_ids), 5),
        None => TimeSeriesSplit::split(train_set, 5),
    };
    if splits.is_empty() {
        return Err(FSRSError::NotEnoughData);
    }
    let split_count = splits.len();
    let mut predictions = Vec::new();
    let mut labels = Vec::new();
    let mut weights = Vec::new();
    let mut r_matrix: HashMap<(u32, u32, u32), RMatrixValue> = HashMap::new();
    let mut progress_info = ItemProgress {
        current: 0,
        total: split_count,
    };

    {
        let split_progresses = (0..split_count)
            .map(|_| training::CombinedProgressState::new_shared())
            .collect::<Vec<_>>();
        let (tx, rx) = std::sync::mpsc::channel();
        let mut pending = std::iter::repeat_with(|| None)
            .take(split_count)
            .collect::<Vec<Option<Result<SplitEvaluation>>>>();
        let mut next_index = 0;
        let mut outcome: Result<()> = Ok(());

        for (index, split) in splits.into_iter().enumerate() {
            let tx = tx.clone();
            let progress = split_progresses[index].clone();
            rayon::spawn(move || {
                let result = evaluate_time_series_split(
                    split,
                    enable_short_term,
                    num_relearning_steps,
                    training_config,
                    Some(progress),
                );
                let _ = tx.send((index, result));
            });
        }
        drop(tx);

        for (index, result) in rx {
            if outcome.is_err() {
                continue;
            }

            pending[index] = Some(result);
            while next_index < split_count {
                let Some(result) = pending[next_index].take() else {
                    break;
                };

                match result {
                    Ok(evaluation) => {
                        merge_split_evaluation(
                            evaluation,
                            &mut predictions,
                            &mut labels,
                            &mut weights,
                            &mut r_matrix,
                        );
                        progress_info.current += 1;
                        if !progress(progress_info) {
                            abort_training(&split_progresses);
                            outcome = Err(FSRSError::Interrupted);
                            break;
                        }
                    }
                    Err(err) => {
                        abort_training(&split_progresses);
                        outcome = Err(err);
                        break;
                    }
                }
                next_index += 1;
            }
        }

        if outcome.is_ok() && next_index < split_count {
            outcome = Err(FSRSError::InvalidInput);
        }
        outcome?;
    }

    let rmse = rmse_bins(&r_matrix);
    let loss = weighted_binary_cross_entropy(&predictions, &labels, &weights);
    if !loss.is_finite() || !rmse.is_finite() {
        return Err(FSRSError::InvalidInput);
    }
    Ok(ModelEvaluation {
        log_loss: loss,
        rmse_bins: rmse,
    })
}

/// Measure model performance in bins
fn measure_a_by_b(pred_a: &[f32], pred_b: &[f32], true_val: &[f32]) -> f32 {
    let mut groups = HashMap::new();
    izip!(pred_a, pred_b, true_val).for_each(|(a, b, t)| {
        let bin = get_bin(*b, 20);
        groups.entry(bin).or_insert_with(Vec::new).push((a, t));
    });
    let mut total_sum = 0.0;
    let mut total_count = 0.0;
    for group in groups.values() {
        let count = group.len() as f32;
        let pred_mean = group.iter().map(|(p, _)| *p).sum::<f32>() / count;
        let true_mean = group.iter().map(|(_, t)| *t).sum::<f32>() / count;

        let rmse = (pred_mean - true_mean).powi(2);
        total_sum += rmse * count;
        total_count += count;
    }

    (total_sum / total_count).sqrt()
}

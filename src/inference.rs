use itertools::izip;
use serde::Serialize;
use std::collections::HashMap;
use std::ops::{Add, Sub};
use std::sync::{Arc, Mutex};

use crate::dataset::{constant_weighted_fsrs_items, recency_weighted_fsrs_items};
use crate::error::Result;
use crate::model::FSRS;
use crate::simulation::{D_MAX, D_MIN, S_MIN};
use crate::training::weighted_binary_cross_entropy;
use crate::training::{self, ComputeParametersInput};
use crate::{FSRSError, FSRSItem};
/// This is a slice for efficiency, but should always be 21 in length.
pub type Parameters = [f32];
type SharedTrainingProgress = Arc<Mutex<training::CombinedProgressState>>;

/// The default decay for FSRS 5.
pub const FSRS5_DEFAULT_DECAY: f32 = 0.5;
/// The default decay for FSRS 6.
pub const FSRS6_DEFAULT_DECAY: f32 = 0.1542;

/// The default parameters. Fits the average person's learning habits.
pub static DEFAULT_PARAMETERS: [f32; 21] = [
    0.212,
    1.2931,
    2.3065,
    8.2956,
    6.4133,
    0.8334,
    3.0194,
    0.001,
    1.8722,
    0.1666,
    0.796,
    1.4835,
    0.0614,
    0.2629,
    1.6483,
    0.6014,
    1.8729,
    0.5425,
    0.0912,
    0.0658,
    FSRS6_DEFAULT_DECAY,
];

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
    let retrievability = fsrs.power_forgetting_curve(current.delta_t as f32, state.stability);
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
            }
        };
        for (index, review) in item.reviews.iter().enumerate() {
            inner_state = self.step(review.delta_t as f32, review.rating, inner_state, index);
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
        let w = self.parameters();
        let decay = -w[20];
        let factor = 0.9f32.powf(1.0 / decay) - 1.0;
        let stability = interval.max(S_MIN) * factor / (sm2_retention.powf(1.0 / decay) - 1.0);
        let difficulty = 11.0
            - (ease_factor - 1.0)
                / (w[8].exp() * stability.powf(-w[9]) * ((1.0 - sm2_retention) * w[10]).exp_m1());
        if !stability.is_finite() || !difficulty.is_finite() {
            Err(FSRSError::InvalidInput)
        } else {
            Ok(MemoryState {
                stability,
                difficulty: difficulty.clamp(D_MIN, D_MAX),
            })
        }
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
        self.next_interval_for_stability(stability, desired_retention)
    }

    /// The intervals and memory states for each answer button.
    ///
    /// Returns a [`NextStates`] struct containing the intervals and memory states for each answer button.
    ///
    /// # Examples
    /// ```
    /// use fsrs::{FSRS, MemoryState};
    ///
    /// let fsrs = FSRS::default();
    /// let desired_retention = 0.9;
    /// let previous_state: Option<MemoryState> = None;
    /// let elapsed_days = 0;
    ///
    /// let next_states = fsrs.next_states(previous_state, desired_retention, elapsed_days).unwrap();
    /// let review = next_states.good;
    /// ```
    pub fn next_states(
        &self,
        current_memory_state: Option<MemoryState>,
        desired_retention: f32,
        days_elapsed: u32,
    ) -> Result<NextStates> {
        let (current_memory_state, nth) = if let Some(state) = current_memory_state {
            (state, 1)
        } else {
            (
                MemoryState {
                    stability: 0.0,
                    difficulty: 0.0,
                },
                0,
            )
        };
        let mut next_memory_states = (1..=4).map(|rating| {
            validate_state(self.step(days_elapsed as f32, rating, current_memory_state, nth))
        });

        let mut get_next_state = || {
            let memory = next_memory_states.next().unwrap()?;
            let interval = self.next_interval_for_stability(memory.stability, desired_retention);
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        FSRSReview, convertor_tests::anki21_sample_file_converted_to_fsrs, current_retrievability,
        dataset::filter_outlier, test_helpers::TestHelper,
    };

    static PARAMETERS: &[f32] = &[
        0.6845422,
        1.6790825,
        4.7349424,
        10.042885,
        7.4410233,
        0.64219797,
        1.071918,
        0.0025195254,
        1.432437,
        0.1544,
        0.8692766,
        2.0696752,
        0.0953,
        0.2975,
        2.4691248,
        0.19542035,
        3.201072,
        0.18046261,
        0.121442534,
    ];

    #[test]
    fn test_get_bin() {
        let pred = (0..=100).map(|i| i as f32 / 100.0).collect::<Vec<_>>();
        let bin = pred.iter().map(|p| get_bin(*p, 20)).collect::<Vec<_>>();
        assert_eq!(
            bin,
            [
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1,
                1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4,
                4, 4, 4, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 9, 9, 9, 10, 10, 10,
                11, 11, 11, 12, 12, 13, 13, 14, 14, 14, 15, 15, 16, 17, 17, 18, 18, 19, 19
            ]
        );
    }

    #[test]
    fn test_memo_state() -> Result<()> {
        let item = FSRSItem {
            reviews: vec![
                FSRSReview {
                    rating: 1,
                    delta_t: 0,
                },
                FSRSReview {
                    rating: 3,
                    delta_t: 1,
                },
                FSRSReview {
                    rating: 3,
                    delta_t: 3,
                },
                FSRSReview {
                    rating: 3,
                    delta_t: 8,
                },
                FSRSReview {
                    rating: 3,
                    delta_t: 21,
                },
            ],
        };
        let fsrs = FSRS::new(PARAMETERS)?;
        assert_eq!(
            fsrs.memory_state(item, None).unwrap(),
            MemoryState {
                stability: 31.722992,
                difficulty: 7.382128
            }
        );

        assert_eq!(
            fsrs.next_states(
                Some(MemoryState {
                    stability: 20.925528,
                    difficulty: 7.005062
                }),
                0.9,
                21
            )
            .unwrap()
            .good
            .memory,
            MemoryState {
                stability: 40.87456,
                difficulty: 6.9913807
            }
        );
        Ok(())
    }

    fn assert_memory_state(w: &[f32], expected_stability: f32, expected_difficulty: f32) {
        let desired_retention = 0.9;
        let fsrs = FSRS::new(w).unwrap();
        let ratings: [u32; 6] = [1, 3, 3, 3, 3, 3];
        let intervals: [u32; 6] = [0, 0, 1, 3, 8, 21];

        let mut memory_state = None;
        for (&rating, &interval) in ratings.iter().zip(intervals.iter()) {
            let state = fsrs
                .next_states(memory_state, desired_retention, interval)
                .unwrap();
            memory_state = match rating {
                1 => Some(state.again.memory),
                2 => Some(state.hard.memory),
                3 => Some(state.good.memory),
                4 => Some(state.easy.memory),
                _ => None,
            };
            // dbg!(
            //     "stability: {}, difficulty: {}",
            //     memory_state.as_ref().unwrap().stability,
            //     memory_state.as_ref().unwrap().difficulty
            // );
        }

        let memory_state = memory_state.unwrap();
        let stability = memory_state.stability;
        let difficulty = memory_state.difficulty;
        assert!(
            (stability - expected_stability).abs() < 1e-4,
            "stability: {}",
            stability
        );
        assert!(
            (difficulty - expected_difficulty).abs() < 1e-4,
            "difficulty: {}",
            difficulty
        );
    }
    #[test]
    fn test_memory_state() {
        let mut w = DEFAULT_PARAMETERS;
        assert_memory_state(&w, 53.62691, 6.3574867);
        // freeze short term
        w[17] = 0.0;
        w[18] = 0.0;
        w[19] = 0.0;
        assert_memory_state(&w, 53.335106, 6.3574867);
    }

    #[test]
    fn test_next_interval() {
        let fsrs = FSRS::default();
        let desired_retentions = (1..=10).map(|i| i as f32 / 10.0).collect::<Vec<_>>();
        let intervals = desired_retentions
            .iter()
            .map(|r| fsrs.next_interval(Some(1.0), *r, 1).round().max(1.0) as i32)
            .collect::<Vec<_>>();
        assert_eq!(intervals, [3116766, 34793, 2508, 387, 90, 27, 9, 3, 1, 1]);
    }

    #[test]
    fn test_evaluate() -> Result<()> {
        let items = anki21_sample_file_converted_to_fsrs();
        let (mut dataset_for_initialization, mut trainset): (Vec<FSRSItem>, Vec<FSRSItem>) = items
            .into_iter()
            .partition(|item| item.long_term_review_cnt() == 1);
        (dataset_for_initialization, trainset) =
            filter_outlier(dataset_for_initialization, trainset);
        let items = [dataset_for_initialization, trainset].concat();

        let fsrs = FSRS::new(&[
            0.335561,
            1.6840581,
            5.166598,
            11.659035,
            7.466705,
            0.7205129,
            2.622295,
            0.001,
            1.315015,
            0.10468433,
            0.8349206,
            1.822305,
            0.12473127,
            0.26111007,
            2.3030033,
            0.13117497,
            3.0265594,
            0.41468078,
            0.09714265,
            0.106824234,
            0.20447432,
        ])?;
        let metrics = fsrs.evaluate(items.clone(), |_| true).unwrap();

        [metrics.log_loss, metrics.rmse_bins].assert_approx_eq([0.20580745, 0.026005825]);

        let fsrs = FSRS::default();
        let metrics = fsrs.evaluate(items.clone(), |_| true).unwrap();

        [metrics.log_loss, metrics.rmse_bins].assert_approx_eq([0.20967911, 0.030774858]);

        let fsrs = FSRS::new(PARAMETERS)?;
        let metrics = fsrs.evaluate(items.clone(), |_| true).unwrap();

        [metrics.log_loss, metrics.rmse_bins].assert_approx_eq([0.208_657_4, 0.030_946_612]);

        let (self_by_other, other_by_self) = fsrs
            .universal_metrics(items.clone(), &DEFAULT_PARAMETERS, |_| true)
            .unwrap();

        [self_by_other, other_by_self].assert_approx_eq([0.014087644, 0.017199915]);

        Ok(())
    }

    #[test]
    fn test_time_series_split() -> Result<()> {
        let items = anki21_sample_file_converted_to_fsrs();
        let splits = TimeSeriesSplit::split(items[..6].to_vec(), 5);
        assert_eq!(splits.len(), 5);
        assert_eq!(splits[0].train_items.len(), 1);
        assert_eq!(splits[0].test_items.len(), 1);
        assert_eq!(splits[1].train_items.len(), 2);
        assert_eq!(splits[1].test_items.len(), 1);
        assert_eq!(splits[2].train_items.len(), 3);
        assert_eq!(splits[2].test_items.len(), 1);
        assert_eq!(splits[3].train_items.len(), 4);
        assert_eq!(splits[3].test_items.len(), 1);
        assert_eq!(splits[4].train_items.len(), 5);
        assert_eq!(splits[4].test_items.len(), 1);
        assert!(splits[4].train_card_ids.is_none());

        let card_ids = vec![10, 11, 12, 13, 14, 15];
        let splits = TimeSeriesSplit::split_with_card_ids(items[..6].to_vec(), Some(card_ids), 5);
        assert_eq!(splits.len(), 5);
        assert_eq!(splits[0].train_card_ids.as_deref(), Some(&[10][..]));
        assert_eq!(splits[1].train_card_ids.as_deref(), Some(&[10, 11][..]));
        assert_eq!(splits[2].train_card_ids.as_deref(), Some(&[10, 11, 12][..]));
        assert_eq!(
            splits[3].train_card_ids.as_deref(),
            Some(&[10, 11, 12, 13][..])
        );
        assert_eq!(
            splits[4].train_card_ids.as_deref(),
            Some(&[10, 11, 12, 13, 14][..])
        );

        let splits = TimeSeriesSplit::split(items[..5].to_vec(), 5);
        assert!(splits.is_empty());

        let splits = TimeSeriesSplit::split(items[..6].to_vec(), 0);
        assert!(splits.is_empty());

        Ok(())
    }

    #[test]
    fn test_memory_state_batch() -> Result<()> {
        let fsrs = FSRS::new(PARAMETERS)?;

        let items = vec![
            FSRSItem {
                reviews: vec![
                    FSRSReview {
                        rating: 1,
                        delta_t: 0,
                    },
                    FSRSReview {
                        rating: 3,
                        delta_t: 1,
                    },
                    FSRSReview {
                        rating: 3,
                        delta_t: 3,
                    },
                ],
            },
            FSRSItem {
                reviews: vec![
                    FSRSReview {
                        rating: 1,
                        delta_t: 0,
                    },
                    FSRSReview {
                        rating: 3,
                        delta_t: 1,
                    },
                    FSRSReview {
                        rating: 3,
                        delta_t: 3,
                    },
                    FSRSReview {
                        rating: 3,
                        delta_t: 8,
                    },
                    FSRSReview {
                        rating: 3,
                        delta_t: 21,
                    },
                ],
            },
            FSRSItem {
                reviews: vec![
                    FSRSReview {
                        rating: 2,
                        delta_t: 0,
                    },
                    FSRSReview {
                        rating: 4,
                        delta_t: 1,
                    },
                    FSRSReview {
                        rating: 2,
                        delta_t: 2,
                    },
                ],
            },
        ];

        // Test with no starting states (all None)
        let starting_states = vec![None, None, None];
        let batch_results = fsrs.memory_state_batch(items.clone(), starting_states)?;

        // Compare with individual memory_state calls
        let individual_results: Vec<MemoryState> = items
            .iter()
            .map(|item| fsrs.memory_state(item.clone(), None).unwrap())
            .collect();

        // Verify that batch results match individual results
        assert_eq!(batch_results.len(), individual_results.len());
        for (batch_result, individual_result) in batch_results.iter().zip(individual_results.iter())
        {
            assert_eq!(batch_result, individual_result);
        }

        // Test with some starting states
        let starting_states = vec![
            Some(MemoryState {
                stability: 5.0,
                difficulty: 6.0,
            }),
            None,
            Some(MemoryState {
                stability: 10.0,
                difficulty: 7.0,
            }),
        ];
        let batch_results_with_starting =
            fsrs.memory_state_batch(items.clone(), starting_states.clone())?;

        // Compare with individual calls using starting states
        let individual_results_with_starting: Vec<MemoryState> = items
            .iter()
            .zip(starting_states.iter())
            .map(|(item, starting_state)| fsrs.memory_state(item.clone(), *starting_state).unwrap())
            .collect();

        // Verify that batch results match individual results
        assert_eq!(
            batch_results_with_starting.len(),
            individual_results_with_starting.len()
        );
        for (batch_result, individual_result) in batch_results_with_starting
            .iter()
            .zip(individual_results_with_starting.iter())
        {
            assert_eq!(batch_result, individual_result);
        }

        // Test with empty items list
        let empty_result = fsrs.memory_state_batch(vec![], vec![])?;
        assert_eq!(empty_result.len(), 0);

        // Test with single item
        let single_item = vec![items[0].clone()];
        let single_starting = vec![None];
        let single_batch_result =
            fsrs.memory_state_batch(single_item.clone(), single_starting.clone())?;
        let single_individual_result =
            fsrs.memory_state(single_item[0].clone(), single_starting[0])?;

        assert_eq!(single_batch_result.len(), 1);
        assert_eq!(single_batch_result[0], single_individual_result);

        Ok(())
    }

    #[test]
    fn test_historical_memory_state_batch() -> Result<()> {
        let fsrs = FSRS::new(PARAMETERS)?;

        let items = vec![
            FSRSItem {
                reviews: vec![
                    FSRSReview {
                        rating: 1,
                        delta_t: 0,
                    },
                    FSRSReview {
                        rating: 3,
                        delta_t: 1,
                    },
                ],
            },
            FSRSItem {
                reviews: vec![FSRSReview {
                    rating: 2,
                    delta_t: 0,
                }],
            },
        ];

        let batch_results = fsrs.historical_memory_state_batch(items.clone(), None)?;

        assert_eq!(batch_results.len(), 2);
        assert_eq!(batch_results[0].len(), 2);
        assert_eq!(batch_results[1].len(), 1);

        let individual_result_0 = fsrs.historical_memory_states(items[0].clone(), None)?;
        let individual_result_1 = fsrs.historical_memory_states(items[1].clone(), None)?;

        assert_eq!(batch_results[0], individual_result_0);
        assert_eq!(batch_results[1], individual_result_1);

        Ok(())
    }

    #[test]
    fn test_evaluate_with_time_series_splits() -> Result<()> {
        let items = anki21_sample_file_converted_to_fsrs();
        let (mut dataset_for_initialization, mut trainset): (Vec<FSRSItem>, Vec<FSRSItem>) = items
            .into_iter()
            .partition(|item| item.long_term_review_cnt() == 1);
        (dataset_for_initialization, trainset) =
            filter_outlier(dataset_for_initialization, trainset);
        let items = [dataset_for_initialization, trainset].concat();
        let input = ComputeParametersInput {
            train_set: items.clone(),
            card_ids: None,
            progress: None,
            enable_short_term: true,
            num_relearning_steps: None,
            training_config: None,
        };

        let metrics = evaluate_with_time_series_splits(input.clone(), |_| true).unwrap();

        [metrics.log_loss, metrics.rmse_bins].assert_approx_eq([0.19692886, 0.025453836]);

        let metrics_with_card_ids = evaluate_with_time_series_splits(
            ComputeParametersInput {
                train_set: items.clone(),
                card_ids: Some((0..items.len() as i64).collect()),
                progress: None,
                enable_short_term: true,
                num_relearning_steps: None,
                training_config: None,
            },
            |_| true,
        )
        .unwrap();
        assert!(metrics_with_card_ids.log_loss.is_finite());
        assert!(metrics_with_card_ids.rmse_bins.is_finite());

        let result = evaluate_with_time_series_splits(
            ComputeParametersInput {
                train_set: items.clone(),
                card_ids: Some(vec![]),
                progress: None,
                enable_short_term: true,
                num_relearning_steps: None,
                training_config: None,
            },
            |_| true,
        );
        assert!(matches!(result, Err(FSRSError::InvalidInput)));

        let result = evaluate_with_time_series_splits(
            ComputeParametersInput {
                train_set: items[..5].to_vec(),
                card_ids: None,
                progress: None,
                enable_short_term: true,
                num_relearning_steps: None,
                training_config: None,
            },
            |_| true,
        );
        assert!(result.is_err());
        Ok(())
    }

    #[test]
    fn test_time_series_split_evaluation_forwards_card_ids() {
        let valid_item = || FSRSItem {
            reviews: vec![
                FSRSReview {
                    rating: 3,
                    delta_t: 0,
                },
                FSRSReview {
                    rating: 3,
                    delta_t: 1,
                },
            ],
        };
        let result = evaluate_time_series_split(
            TimeSeriesSplit {
                train_items: vec![valid_item()],
                train_card_ids: Some(vec![]),
                test_items: vec![valid_item()],
            },
            true,
            None,
            None,
            None,
        );

        assert!(matches!(result, Err(FSRSError::InvalidInput)));
    }

    #[test]
    fn test_evaluate_with_time_series_splits_cancels_before_later_split_work() {
        fn valid_item() -> FSRSItem {
            FSRSItem {
                reviews: vec![
                    FSRSReview {
                        rating: 3,
                        delta_t: 0,
                    },
                    FSRSReview {
                        rating: 3,
                        delta_t: 1,
                    },
                ],
            }
        }

        let mut items = vec![valid_item(); 6];
        items[2] = FSRSItem { reviews: vec![] };
        let input = ComputeParametersInput {
            train_set: items,
            card_ids: None,
            progress: None,
            enable_short_term: true,
            num_relearning_steps: None,
            training_config: None,
        };

        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            evaluate_with_time_series_splits(input, |_| false)
        }));

        assert!(matches!(result.unwrap(), Err(FSRSError::Interrupted)));
    }

    #[test]
    fn test_next_states() -> Result<()> {
        let item = FSRSItem {
            reviews: vec![
                FSRSReview {
                    rating: 1,
                    delta_t: 0,
                },
                FSRSReview {
                    rating: 3,
                    delta_t: 1,
                },
                FSRSReview {
                    rating: 3,
                    delta_t: 3,
                },
                FSRSReview {
                    rating: 3,
                    delta_t: 8,
                },
            ],
        };
        let fsrs = FSRS::new(PARAMETERS)?;
        let state = fsrs.memory_state(item, None).unwrap();
        assert_eq!(
            fsrs.next_states(Some(state), 0.9, 21).unwrap(),
            NextStates {
                again: ItemState {
                    memory: MemoryState {
                        stability: 2.9691455,
                        difficulty: 8.000659
                    },
                    interval: 2.9691455
                },
                hard: ItemState {
                    memory: MemoryState {
                        stability: 17.091452,
                        difficulty: 7.6913934
                    },
                    interval: 17.091452
                },
                good: ItemState {
                    memory: MemoryState {
                        stability: 31.722992,
                        difficulty: 7.382128
                    },
                    interval: 31.722992
                },
                easy: ItemState {
                    memory: MemoryState {
                        stability: 71.7502,
                        difficulty: 7.0728626
                    },
                    interval: 71.7502
                }
            }
        );
        assert_eq!(fsrs.next_interval(Some(121.01552), 0.9, 1), 121.01551);
        Ok(())
    }

    #[test]
    #[ignore = "just for exploration"]
    fn test_short_term_stability() -> Result<()> {
        let fsrs = FSRS::default();
        let mut state = MemoryState {
            stability: 1.0,
            difficulty: 5.0,
        };

        let mut stability = Vec::new();
        for _ in 0..20 {
            state = fsrs.next_states(Some(state), 0.9, 0).unwrap().good.memory;
            stability.push(state.stability);
        }

        dbg!(stability);
        Ok(())
    }

    #[test]
    #[ignore = "just for exploration"]
    fn test_good_again_loop_during_the_same_day() -> Result<()> {
        let fsrs = FSRS::default();
        let mut state = MemoryState {
            stability: 1.0,
            difficulty: 5.0,
        };

        let mut stability = Vec::with_capacity(10);
        for _ in 0..10 {
            state = fsrs.next_states(Some(state), 0.9, 0).unwrap().good.memory;
            state = fsrs.next_states(Some(state), 0.9, 0).unwrap().again.memory;
            stability.push(state.stability);
        }

        dbg!(stability);
        Ok(())
    }

    #[test]
    #[ignore = "just for exploration"]
    fn test_stability_after_same_day_review_less_than_next_day_review() -> Result<()> {
        let fsrs = FSRS::default();
        let state = MemoryState {
            stability: 10.0,
            difficulty: 5.0,
        };

        let next_state = fsrs.next_states(Some(state), 0.9, 0)?.good.memory;
        dbg!(next_state);
        // let next_state = fsrs.next_states(Some(next_state), 0.9, 0)?.good.memory;
        // dbg!(next_state);
        let next_state = fsrs.next_states(Some(state), 0.9, 1)?.good.memory;
        dbg!(next_state);
        Ok(())
    }

    #[test]
    #[ignore = "just for exploration"]
    fn test_init_stability_after_same_day_review_hard_vs_good_vs_easy() -> Result<()> {
        let fsrs = FSRS::default();
        let item = FSRSItem {
            reviews: vec![
                FSRSReview {
                    rating: 2,
                    delta_t: 0,
                },
                FSRSReview {
                    rating: 3,
                    delta_t: 0,
                },
                FSRSReview {
                    rating: 3,
                    delta_t: 0,
                },
            ],
        };
        let state = fsrs.memory_state(item, None).unwrap();
        dbg!(state);
        let item = FSRSItem {
            reviews: vec![
                FSRSReview {
                    rating: 3,
                    delta_t: 0,
                },
                FSRSReview {
                    rating: 3,
                    delta_t: 0,
                },
            ],
        };
        let state = fsrs.memory_state(item, None).unwrap();
        dbg!(state);
        let item = FSRSItem {
            reviews: vec![FSRSReview {
                rating: 4,
                delta_t: 0,
            }],
        };
        let state = fsrs.memory_state(item, None).unwrap();
        dbg!(state);
        Ok(())
    }

    #[test]
    fn test_current_retrievability() {
        let state = MemoryState {
            stability: 1.0,
            difficulty: 5.0,
        };
        assert_eq!(current_retrievability(state, 0.0, 0.2), 1.0);
        assert_eq!(current_retrievability(state, 1.0, 0.2), 0.9);
        assert_eq!(current_retrievability(state, 2.0, 0.2), 0.84028935);
        assert_eq!(current_retrievability(state, 3.0, 0.2), 0.7985001);
    }

    #[test]
    fn test_memory_from_sm2() -> Result<()> {
        let fsrs = FSRS::default();
        let memory_state = fsrs.memory_state_from_sm2(2.5, 10.0, 0.9).unwrap();

        [memory_state.stability, memory_state.difficulty].assert_approx_eq([10.0, 6.9140563]);
        let memory_state = fsrs.memory_state_from_sm2(2.5, 10.0, 0.8).unwrap();

        [memory_state.stability, memory_state.difficulty].assert_approx_eq([3.01572, 9.393428]);
        let memory_state = fsrs.memory_state_from_sm2(2.5, 10.0, 0.95).unwrap();

        [memory_state.stability, memory_state.difficulty].assert_approx_eq([24.841097, 1.2974405]);
        let memory_state = fsrs.memory_state_from_sm2(1.3, 20.0, 0.9).unwrap();

        [memory_state.stability, memory_state.difficulty].assert_approx_eq([20.0, 10.0]);
        let interval = 15;
        let ease_factor = 2.0;
        let fsrs_factor = fsrs
            .next_states(
                Some(
                    fsrs.memory_state_from_sm2(ease_factor, interval as f32, 0.9)
                        .unwrap(),
                ),
                0.9,
                interval,
            )?
            .good
            .memory
            .stability
            / interval as f32;
        assert!((fsrs_factor - ease_factor).abs() < 0.01);
        Ok(())
    }
}

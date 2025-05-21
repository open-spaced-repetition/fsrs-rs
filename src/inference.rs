use std::collections::HashMap;
use std::ops::{Add, Sub};

use crate::model::{FSRS, Get, MemoryStateTensors};
use crate::training::ComputeParametersInput;
use burn::nn::loss::Reduction;
use burn::tensor::cast::ToElement;
use burn::tensor::{Shape, Tensor, TensorData};
use burn::{data::dataloader::batcher::Batcher, tensor::backend::Backend};

use crate::dataset::{
    FSRSBatch, FSRSBatcher, constant_weighted_fsrs_items, recency_weighted_fsrs_items,
};
use crate::error::Result;
use crate::model::Model;
use crate::training::BCELoss;
use crate::{FSRSError, FSRSItem};
use burn::tensor::ElementConversion;
pub(crate) const S_MIN: f32 = 0.001;
pub(crate) const S_MAX: f32 = 36500.0;
/// This is a slice for efficiency, but should always be 21 in length.
pub type Parameters = [f32];
use itertools::izip;

pub const FSRS5_DEFAULT_DECAY: f32 = 0.5;
pub const FSRS6_DEFAULT_DECAY: f32 = 0.2;

pub static DEFAULT_PARAMETERS: [f32; 21] = [
    0.2172,
    1.1771,
    3.2602,
    16.1507,
    7.0114,
    0.57,
    2.0966,
    0.0069,
    1.5261,
    0.112,
    1.0178,
    1.849,
    0.1133,
    0.3127,
    2.2934,
    0.2191,
    3.0004,
    0.7536,
    0.3332,
    0.1437,
    FSRS6_DEFAULT_DECAY,
];

fn infer<B: Backend>(
    model: &Model<B>,
    batch: FSRSBatch<B>,
) -> (MemoryStateTensors<B>, Tensor<B, 1>) {
    let state = model.forward(batch.t_historys, batch.r_historys, None);
    let retrievability = model.power_forgetting_curve(batch.delta_ts, state.stability.clone());
    (state, retrievability)
}

#[derive(Debug, PartialEq, Clone, Copy)]
pub struct MemoryState {
    pub stability: f32,
    pub difficulty: f32,
}

impl<B: Backend> From<MemoryStateTensors<B>> for MemoryState {
    fn from(m: MemoryStateTensors<B>) -> Self {
        Self {
            stability: m.stability.into_scalar().elem(),
            difficulty: m.difficulty.into_scalar().elem(),
        }
    }
}

impl<B: Backend> From<MemoryState> for MemoryStateTensors<B> {
    fn from(m: MemoryState) -> Self {
        Self {
            stability: Tensor::from_floats([m.stability], &B::Device::default()),
            difficulty: Tensor::from_floats([m.difficulty], &B::Device::default()),
        }
    }
}

#[derive(Default)]
struct RMatrixValue {
    predicted: f32,
    actual: f32,
    count: f32,
    weight: f32,
}

impl<B: Backend> FSRS<B> {
    fn item_to_tensors(&self, item: &FSRSItem) -> (Tensor<B, 2>, Tensor<B, 2>) {
        let (time_history, rating_history) =
            item.reviews.iter().map(|r| (r.delta_t, r.rating)).unzip();
        let size = item.reviews.len();
        let time_history = Tensor::<B, 1>::from_data(
            TensorData::new(time_history, Shape { dims: vec![size] }),
            &self.device(),
        )
        .unsqueeze()
        .transpose();
        let rating_history = Tensor::<B, 1>::from_data(
            TensorData::new(rating_history, Shape { dims: vec![size] }),
            &self.device(),
        )
        .unsqueeze()
        .transpose();
        (time_history, rating_history)
    }

    /// Calculate the current memory state for a given card's history of reviews.
    /// In the case of truncated reviews, `starting_state` can be set to the value of
    /// [FSRS::memory_state_from_sm2] for the first review (which should not be included
    /// in FSRSItem). If not provided, the card starts as new.
    /// Parameters must have been provided when calling FSRS::new().
    pub fn memory_state(
        &self,
        item: FSRSItem,
        starting_state: Option<MemoryState>,
    ) -> Result<MemoryState> {
        let (time_history, rating_history) = self.item_to_tensors(&item);
        let state: MemoryState = self
            .model()
            .forward(time_history, rating_history, starting_state.map(Into::into))
            .into();
        if !state.stability.is_finite() || !state.difficulty.is_finite() {
            Err(FSRSError::InvalidInput)
        } else {
            Ok(state)
        }
    }

    pub fn historical_memory_states(
        &self,
        item: FSRSItem,
        starting_state: Option<MemoryState>,
    ) -> Result<Vec<MemoryState>> {
        let (time_history, rating_history) = self.item_to_tensors(&item);
        let mut states = vec![];
        if let Some(starting_state) = starting_state {
            states.push(starting_state);
        }
        let [seq_len, _batch_size] = time_history.dims();
        let mut inner_state = starting_state.map(Into::into);
        for i in 0..seq_len {
            let delta_t = time_history.get(i).squeeze(0);
            // [batch_size]
            let rating = rating_history.get(i).squeeze(0);
            // [batch_size]
            inner_state = Some(self.model().step(delta_t, rating, inner_state.clone()));
            if let Some(state) = inner_state.clone() {
                let state: MemoryState = state.into();
                if !state.stability.is_finite() || !state.difficulty.is_finite() {
                    return Err(FSRSError::InvalidInput);
                }
                states.push(state);
            }
        }
        Ok(states)
    }

    /// If a card has incomplete learning history, memory state can be approximated from
    /// current sm2 values.
    /// Parameters must have been provided when calling FSRS::new().
    pub fn memory_state_from_sm2(
        &self,
        ease_factor: f32,
        interval: f32,
        sm2_retention: f32,
    ) -> Result<MemoryState> {
        let w = &self.model().w;
        let decay: f32 = w.get(20).neg().into_scalar().elem();
        let factor = 0.9f32.powf(1.0 / decay) - 1.0;
        let stability = interval.max(S_MIN) * factor / (sm2_retention.powf(1.0 / decay) - 1.0);
        let w8: f32 = w.get(8).into_scalar().elem();
        let w9: f32 = w.get(9).into_scalar().elem();
        let w10: f32 = w.get(10).into_scalar().elem();
        let difficulty = 11.0
            - (ease_factor - 1.0)
                / (w8.exp() * stability.powf(-w9) * ((1.0 - sm2_retention) * w10).exp_m1());
        if !stability.is_finite() || !difficulty.is_finite() {
            Err(FSRSError::InvalidInput)
        } else {
            Ok(MemoryState {
                stability,
                difficulty: difficulty.clamp(1.0, 10.0),
            })
        }
    }

    /// Calculate the next interval for the current memory state, for rescheduling. Stability
    /// should be provided except when the card is new. Rating is ignored except when card is new.
    /// Parameters must have been provided when calling FSRS::new().
    pub fn next_interval(
        &self,
        stability: Option<f32>,
        desired_retention: f32,
        rating: u32,
    ) -> f32 {
        let model = self.model();
        let stability = stability.unwrap_or_else(|| {
            // get initial stability for new card
            let rating = Tensor::from_floats([rating], &self.device());
            model.init_stability(rating).into_scalar().elem()
        });
        model
            .next_interval(
                Tensor::from_floats([stability], &self.device()),
                Tensor::from_floats([desired_retention], &self.device()),
            )
            .into_scalar()
            .elem()
    }

    /// The intervals and memory states for each answer button.
    /// Parameters must have been provided when calling FSRS::new().
    pub fn next_states(
        &self,
        current_memory_state: Option<MemoryState>,
        desired_retention: f32,
        days_elapsed: u32,
    ) -> Result<NextStates> {
        let delta_t = Tensor::from_data(
            TensorData::new(vec![days_elapsed], Shape { dims: vec![1] }),
            &self.device(),
        );
        let current_memory_state_tensors = current_memory_state.map(MemoryStateTensors::from);
        let model = self.model();
        let mut next_memory_states = (1..=4).map(|rating| {
            Ok({
                let state = MemoryState::from(model.step(
                    delta_t.clone(),
                    Tensor::from_data(
                        TensorData::new(vec![rating], Shape { dims: vec![1] }),
                        &self.device(),
                    ),
                    current_memory_state_tensors.clone(),
                ));
                if !state.stability.is_finite() || !state.difficulty.is_finite() {
                    return Err(FSRSError::InvalidInput);
                }
                state
            })
        });

        let mut get_next_state = || {
            let memory = next_memory_states.next().unwrap()?;
            let interval = model
                .next_interval(
                    Tensor::from_floats([memory.stability], &self.device()),
                    Tensor::from_floats([desired_retention], &self.device()),
                )
                .into_scalar()
                .elem();
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
    /// Parameters must have been provided when calling FSRS::new().
    pub fn evaluate<F>(&self, items: Vec<FSRSItem>, mut progress: F) -> Result<ModelEvaluation>
    where
        F: FnMut(ItemProgress) -> bool,
    {
        if items.is_empty() {
            return Err(FSRSError::NotEnoughData);
        }
        let weighted_items = recency_weighted_fsrs_items(items);
        let batcher = FSRSBatcher::new(self.device());
        let mut all_retrievability = vec![];
        let mut all_labels = vec![];
        let mut all_weights = vec![];
        let mut progress_info = ItemProgress {
            current: 0,
            total: weighted_items.len(),
        };
        let model = self.model();
        let mut r_matrix: HashMap<(u32, u32, u32), RMatrixValue> = HashMap::new();

        for chunk in weighted_items.chunks(512) {
            let batch = batcher.batch(chunk.to_vec(), &self.device());
            let (_state, retrievability) = infer::<B>(model, batch.clone());
            let pred = retrievability.clone().to_data().to_vec::<f32>().unwrap();
            let true_val = batch.labels.clone().to_data().to_vec::<i64>().unwrap();
            all_retrievability.push(retrievability);
            all_labels.push(batch.labels);
            all_weights.push(batch.weights);
            izip!(chunk, pred, true_val).for_each(|(weighted_item, p, y)| {
                let bin = weighted_item.item.r_matrix_index();
                let value = r_matrix.entry(bin).or_default();
                value.predicted += p;
                value.actual += y as f32;
                value.count += 1.0;
                value.weight += weighted_item.weight;
            });
            progress_info.current += chunk.len();
            if !progress(progress_info) {
                return Err(FSRSError::Interrupted);
            }
        }
        let rmse = (r_matrix
            .values()
            .map(|v| {
                let pred = v.predicted / v.count;
                let real = v.actual / v.count;
                (pred - real).powi(2) * v.weight
            })
            .sum::<f32>()
            / r_matrix.values().map(|v| v.weight).sum::<f32>())
        .sqrt();
        let all_retrievability = Tensor::cat(all_retrievability, 0);
        let all_labels = Tensor::cat(all_labels, 0).float();
        let all_weights = Tensor::cat(all_weights, 0);
        let loss =
            BCELoss::new().forward(all_retrievability, all_labels, all_weights, Reduction::Auto);
        Ok(ModelEvaluation {
            log_loss: loss.into_scalar().to_f32(),
            rmse_bins: rmse,
        })
    }

    /// Determine how well the model and parameters predict performance using time series splits.
    /// For each split:
    /// 1. Use training data to compute parameters
    /// 2. Use test data to make predictions
    /// 3. Collect all predictions
    ///
    /// Finally, evaluate all predictions together
    ///
    /// # Arguments
    /// * `items` - The dataset to evaluate
    /// * `progress` - A callback function to report progress
    ///
    /// # Returns
    /// A ModelEvaluation containing metrics for all predictions
    pub fn evaluate_with_time_series_splits<F>(
        &self,
        ComputeParametersInput {
            train_set,
            enable_short_term,
            num_relearning_steps,
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

        let splits = TimeSeriesSplit::split(train_set, 5);
        let mut all_predictions = Vec::new();
        let mut progress_info = ItemProgress {
            current: 0,
            total: splits.len(),
        };

        for split in splits.into_iter() {
            // Compute parameters on training data
            let input = ComputeParametersInput {
                train_set: split.train_items.clone(),
                enable_short_term,
                num_relearning_steps,
                progress: None,
            };
            let parameters = self.compute_parameters(input)?;

            // Make predictions on test data
            let predictions = batch_predict::<B>(split.test_items, &parameters)?;

            // Collect predictions
            all_predictions.extend(predictions);

            progress_info.current += 1;
            if !progress(progress_info) {
                return Err(FSRSError::Interrupted);
            }
        }

        // Evaluate all predictions together
        evaluate::<B>(all_predictions)
    }

    /// How well the user is likely to remember the item after `days_elapsed` since the previous
    /// review.
    pub fn current_retrievability(&self, state: MemoryState, days_elapsed: u32, decay: f32) -> f32 {
        let factor = 0.9f32.powf(1.0 / -decay) - 1.0;
        (days_elapsed as f32 / state.stability * factor + 1.0).powf(-decay)
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
        let batcher = FSRSBatcher::new(self.device());
        let mut all_predictions_self = vec![];
        let mut all_predictions_other = vec![];
        let mut all_true_val = vec![];
        let mut progress_info = ItemProgress {
            current: 0,
            total: weighted_items.len(),
        };
        let model_self = self.model();
        let fsrs_other = Self::new_with_backend(Some(parameters), self.device())?;
        let model_other = fsrs_other.model();
        for chunk in weighted_items.chunks(512) {
            let batch = batcher.batch(chunk.to_vec(), &self.device());

            let (_state, retrievability) = infer::<B>(model_self, batch.clone());
            let pred = retrievability.clone().to_data().to_vec::<f32>().unwrap();
            all_predictions_self.extend(pred);

            let (_state, retrievability) = infer::<B>(model_other, batch.clone());
            let pred = retrievability.clone().to_data().to_vec::<f32>().unwrap();
            all_predictions_other.extend(pred);

            let true_val: Vec<f32> = batch
                .labels
                .clone()
                .to_data()
                .convert::<f32>()
                .to_vec()
                .unwrap();
            all_true_val.extend(true_val);
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

#[derive(Debug, Clone)]
pub struct PredictedFSRSItem {
    pub item: FSRSItem,
    pub retrievability: f32,
}

/// Batch predict retrievability for a set of items.
///
/// # Arguments
/// * `items` - The dataset to predict
/// * `parameters` - The model parameters to use for prediction
/// * `progress` - A callback function to report progress
///
/// # Returns
/// A vector of PredictedFSRSItem containing the original items and their predicted retrievability
fn batch_predict<B: Backend>(
    items: Vec<FSRSItem>,
    parameters: &[f32],
) -> Result<Vec<PredictedFSRSItem>>
where
{
    if items.is_empty() {
        return Err(FSRSError::NotEnoughData);
    }
    let weighted_items = constant_weighted_fsrs_items(items);
    let batcher = FSRSBatcher::new(B::Device::default());

    let fsrs = FSRS::<B>::new_with_backend(Some(parameters), B::Device::default())?;
    let model = fsrs.model();
    let mut predicted_items = Vec::with_capacity(weighted_items.len());

    for chunk in weighted_items.chunks(512) {
        let batch = batcher.batch(chunk.to_vec(), &B::Device::default());
        let (_state, retrievability) = infer::<B>(model, batch.clone());
        let pred = retrievability.to_data().to_vec::<f32>().unwrap();

        for (weighted_item, p) in chunk.iter().zip(pred) {
            predicted_items.push(PredictedFSRSItem {
                item: weighted_item.item.clone(),
                retrievability: p,
            });
        }
    }

    Ok(predicted_items)
}

/// Evaluate model predictions against ground truth.
///
/// # Arguments
/// * `predicted_items` - The items with their predicted retrievability values
/// * `progress` - A callback function to report progress
///
/// # Returns
/// A ModelEvaluation containing log loss and RMSE metrics
fn evaluate<B: Backend>(predicted_items: Vec<PredictedFSRSItem>) -> Result<ModelEvaluation> {
    if predicted_items.is_empty() {
        return Err(FSRSError::NotEnoughData);
    }
    let mut all_labels = Vec::with_capacity(predicted_items.len());
    let mut r_matrix: HashMap<(u32, u32, u32), RMatrixValue> = HashMap::new();
    for predicted_item in predicted_items.iter() {
        let pred = predicted_item.retrievability;
        let y = (predicted_item.item.current().rating > 1) as i32;
        all_labels.push(y);
        let bin = predicted_item.item.r_matrix_index();
        let value = r_matrix.entry(bin).or_default();
        value.predicted += pred;
        value.actual += y as f32;
        value.count += 1.0;
        value.weight += 1.0;
    }

    let rmse = (r_matrix
        .values()
        .map(|v| {
            let pred = v.predicted / v.count;
            let real = v.actual / v.count;
            (pred - real).powi(2) * v.weight
        })
        .sum::<f32>()
        / r_matrix.values().map(|v| v.weight).sum::<f32>())
    .sqrt();

    let all_labels = Tensor::from_data(
        TensorData::new(
            all_labels.clone(),
            Shape {
                dims: vec![all_labels.len()],
            },
        ),
        &B::Device::default(),
    );
    let all_weights = Tensor::ones(all_labels.shape(), &B::Device::default());
    let all_retrievability: Tensor<B, 1> = Tensor::from_data(
        TensorData::new(
            predicted_items.iter().map(|p| p.retrievability).collect(),
            Shape {
                dims: vec![predicted_items.len()],
            },
        ),
        &B::Device::default(),
    );

    let loss = BCELoss::new().forward(all_retrievability, all_labels, all_weights, Reduction::Auto);
    Ok(ModelEvaluation {
        log_loss: loss.into_scalar().to_f32(),
        rmse_bins: rmse,
    })
}

#[derive(Debug, Copy, Clone)]
pub struct ModelEvaluation {
    pub log_loss: f32,
    pub rmse_bins: f32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct NextStates {
    pub again: ItemState,
    pub hard: ItemState,
    pub good: ItemState,
    pub easy: ItemState,
}

#[derive(Debug, PartialEq, Clone)]
pub struct ItemState {
    pub memory: MemoryState,
    pub interval: f32,
}

#[derive(Debug, Clone, Copy)]
pub struct ItemProgress {
    pub current: usize,
    pub total: usize,
}

#[derive(Debug, Clone)]
pub struct TimeSeriesSplit {
    pub train_items: Vec<FSRSItem>,
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
                    test_items: sorted_items[test_start..test_end].to_vec(),
                }
            })
            .collect()
    }
}

fn get_bin(x: f32, bins: i32) -> i32 {
    let log_base = (bins.add(1) as f32).ln();
    let binned_x = (x * log_base).exp().floor().sub(1.0);
    (binned_x as i32).clamp(0, bins - 1)
}

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
        FSRSReview, convertor_tests::anki21_sample_file_converted_to_fsrs, dataset::filter_outlier,
        test_helpers::TestHelper,
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
        let fsrs = FSRS::new(Some(PARAMETERS))?;
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
        let fsrs = FSRS::new(Some(w)).unwrap();
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
        let mut w = DEFAULT_PARAMETERS.clone();
        assert_memory_state(&w, 49.4473, 6.8573);
        // freeze short term
        w[17] = 0.0;
        w[18] = 0.0;
        w[19] = 0.0;
        assert_memory_state(&w, 48.6015, 6.8573);
    }

    #[test]
    fn test_next_interval() {
        let fsrs = FSRS::new(Some(&DEFAULT_PARAMETERS)).unwrap();
        let desired_retentions = (1..=10).map(|i| i as f32 / 10.0).collect::<Vec<_>>();
        let intervals = desired_retentions
            .iter()
            .map(|r| fsrs.next_interval(Some(1.0), *r, 1).round().max(1.0) as i32)
            .collect::<Vec<_>>();
        assert_eq!(intervals, [144193, 4505, 592, 139, 45, 17, 7, 3, 1, 1]);
    }

    #[test]
    fn test_evaluate() -> Result<()> {
        let items = anki21_sample_file_converted_to_fsrs();
        let (mut pretrainset, mut trainset): (Vec<FSRSItem>, Vec<FSRSItem>) = items
            .into_iter()
            .partition(|item| item.long_term_review_cnt() == 1);
        (pretrainset, trainset) = filter_outlier(pretrainset, trainset);
        let items = [pretrainset, trainset].concat();

        let fsrs = FSRS::new(Some(&[
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
        ]))?;
        let metrics = fsrs.evaluate(items.clone(), |_| true).unwrap();

        [metrics.log_loss, metrics.rmse_bins].assert_approx_eq([0.205_835_95, 0.026_072_025]);

        let fsrs = FSRS::new(Some(&[]))?;
        let metrics = fsrs.evaluate(items.clone(), |_| true).unwrap();

        [metrics.log_loss, metrics.rmse_bins].assert_approx_eq([0.217_924_48, 0.039_937_04]);

        let fsrs = FSRS::new(Some(PARAMETERS))?;
        let metrics = fsrs.evaluate(items.clone(), |_| true).unwrap();

        [metrics.log_loss, metrics.rmse_bins].assert_approx_eq([0.208_657_4, 0.030_946_612]);

        let (self_by_other, other_by_self) = fsrs
            .universal_metrics(items.clone(), &DEFAULT_PARAMETERS, |_| true)
            .unwrap();

        [self_by_other, other_by_self].assert_approx_eq([0.015_672_438, 0.028_422_62]);

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

        let splits = TimeSeriesSplit::split(items[..5].to_vec(), 5);
        assert!(splits.is_empty());

        let splits = TimeSeriesSplit::split(items[..6].to_vec(), 0);
        assert!(splits.is_empty());

        Ok(())
    }

    #[test]
    fn test_evaluate_with_time_series_splits() -> Result<()> {
        let items = anki21_sample_file_converted_to_fsrs();
        let (mut pretrainset, mut trainset): (Vec<FSRSItem>, Vec<FSRSItem>) = items
            .into_iter()
            .partition(|item| item.long_term_review_cnt() == 1);
        (pretrainset, trainset) = filter_outlier(pretrainset, trainset);
        let items = [pretrainset, trainset].concat();
        let input = ComputeParametersInput {
            train_set: items.clone(),
            progress: None,
            enable_short_term: true,
            num_relearning_steps: None,
        };

        let fsrs = FSRS::new(None)?;
        let metrics = fsrs
            .evaluate_with_time_series_splits(input.clone(), |_| true)
            .unwrap();

        [metrics.log_loss, metrics.rmse_bins].assert_approx_eq([0.19735593, 0.027728133]);

        let result = fsrs.evaluate_with_time_series_splits(
            ComputeParametersInput {
                train_set: items[..5].to_vec(),
                progress: None,
                enable_short_term: true,
                num_relearning_steps: None,
            },
            |_| true,
        );
        assert!(result.is_err());
        Ok(())
    }

    #[test]
    fn next_states() -> Result<()> {
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
        let fsrs = FSRS::new(Some(PARAMETERS))?;
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
    fn short_term_stability() -> Result<()> {
        let fsrs = FSRS::new(Some(&DEFAULT_PARAMETERS))?;
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
    fn good_again_loop_during_the_same_day() -> Result<()> {
        let fsrs = FSRS::new(Some(&DEFAULT_PARAMETERS))?;
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
    fn stability_after_same_day_review_less_than_next_day_review() -> Result<()> {
        let fsrs = FSRS::new(Some(&DEFAULT_PARAMETERS))?;
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
    fn init_stability_after_same_day_review_hard_vs_good_vs_easy() -> Result<()> {
        let fsrs = FSRS::new(Some(&DEFAULT_PARAMETERS))?;
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
    fn current_retrievability() {
        let fsrs = FSRS::new(None).unwrap();
        let state = MemoryState {
            stability: 1.0,
            difficulty: 5.0,
        };
        assert_eq!(fsrs.current_retrievability(state, 0, 0.2), 1.0);
        assert_eq!(fsrs.current_retrievability(state, 1, 0.2), 0.9);
        assert_eq!(fsrs.current_retrievability(state, 2, 0.2), 0.84028935);
        assert_eq!(fsrs.current_retrievability(state, 3, 0.2), 0.7985001);
    }

    #[test]
    fn memory_from_sm2() -> Result<()> {
        let fsrs = FSRS::new(Some(&[]))?;
        let memory_state = fsrs.memory_state_from_sm2(2.5, 10.0, 0.9).unwrap();

        [memory_state.stability, memory_state.difficulty].assert_approx_eq([10.0, 7.061_206]);
        let memory_state = fsrs.memory_state_from_sm2(2.5, 10.0, 0.8).unwrap();

        [memory_state.stability, memory_state.difficulty]
            .assert_approx_eq([3.380_071_9, 9.344_574]);
        let memory_state = fsrs.memory_state_from_sm2(2.5, 10.0, 0.95).unwrap();

        [memory_state.stability, memory_state.difficulty]
            .assert_approx_eq([23.721_418, 2.095_691_7]);
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

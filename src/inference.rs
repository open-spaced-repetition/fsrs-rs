use std::collections::HashMap;
use std::ops::{Add, Sub};

use crate::model::{MemoryStateTensors, FSRS};
use burn::tensor::{Data, Shape, Tensor};
use burn::{data::dataloader::batcher::Batcher, tensor::backend::Backend};

use crate::dataset::FSRSBatch;
use crate::dataset::FSRSBatcher;
use crate::error::Result;
use crate::model::Model;
use crate::training::BCELoss;
use crate::{FSRSError, FSRSItem};
use burn::tensor::ElementConversion;

/// This is a slice for efficiency, but should always be 17 in length.
pub type Weights = [f32];

pub static DEFAULT_WEIGHTS: &[f32] = &[
    0.4, 0.6, 2.4, 5.8, 4.93, 0.94, 0.86, 0.01, 1.49, 0.14, 0.94, 2.18, 0.05, 0.34, 1.26, 0.29,
    2.61,
];

fn infer<B: Backend>(
    model: &Model<B>,
    batch: FSRSBatch<B>,
) -> (MemoryStateTensors<B>, Tensor<B, 2>) {
    let state = model.forward(batch.t_historys, batch.r_historys);
    let retention = model.power_forgetting_curve(
        batch.delta_ts.clone().unsqueeze::<2>().transpose(),
        state.stability.clone(),
    );
    (state, retention)
}

#[derive(Debug, PartialEq, Clone, Copy)]
pub struct MemoryState {
    pub stability: f32,
    pub difficulty: f32,
}

impl<B: Backend> From<MemoryStateTensors<B>> for MemoryState {
    fn from(m: MemoryStateTensors<B>) -> Self {
        MemoryState {
            stability: m.stability.to_data().value[0].elem(),
            difficulty: m.difficulty.to_data().value[0].elem(),
        }
    }
}

impl<B: Backend> From<MemoryState> for MemoryStateTensors<B> {
    fn from(m: MemoryState) -> Self {
        MemoryStateTensors {
            stability: Tensor::<B, 1>::from_data(Data::new(
                vec![m.stability.elem()],
                Shape { dims: [1] },
            ))
            .unsqueeze(),
            difficulty: Tensor::<B, 1>::from_data(Data::new(
                vec![m.difficulty.elem()],
                Shape { dims: [1] },
            ))
            .unsqueeze()
            .transpose(),
        }
    }
}

fn next_interval(stability: f32, request_retention: f32) -> u32 {
    (9.0 * stability * (1.0 / request_retention - 1.0))
        .round()
        .max(1.0) as u32
}

impl<B: Backend> FSRS<B> {
    /// Calculate the current memory state for a given card's history of reviews.
    /// Weights must have been provided when calling FSRS::new().
    pub fn memory_state(&self, item: FSRSItem) -> MemoryState {
        let (time_history, rating_history) =
            item.reviews.iter().map(|r| (r.delta_t, r.rating)).unzip();
        let size = item.reviews.len();
        let time_history =
            Tensor::<B, 1>::from_data(Data::new(time_history, Shape { dims: [size] }).convert())
                .unsqueeze()
                .transpose();
        let rating_history =
            Tensor::<B, 1>::from_data(Data::new(rating_history, Shape { dims: [size] }).convert())
                .unsqueeze()
                .transpose();
        self.model().forward(time_history, rating_history).into()
    }

    /// The intervals and memory states for each answer button.
    /// Weights must have been provided when calling FSRS::new().
    pub fn next_states(
        &self,
        current_memory_state: Option<MemoryState>,
        desired_retention: f32,
        days_elapsed: u32,
    ) -> NextStates {
        let delta_t =
            Tensor::<B, 1>::from_data(Data::new(vec![days_elapsed.elem()], Shape { dims: [1] }))
                .unsqueeze()
                .transpose();
        let current_memory_state_tensors = current_memory_state.map(MemoryStateTensors::from);
        let model = self.model();
        let mut next_memory_states = (1..=4).map(|rating| {
            if let (Some(current_memory_state), 0) = (current_memory_state, days_elapsed) {
                // When there's an existing memory state and no days have elapsed, we leave it unchanged.
                current_memory_state
            } else {
                MemoryState::from(
                    model.step(
                        delta_t.clone(),
                        Tensor::<B, 1>::from_data(Data::new(
                            vec![rating.elem()],
                            Shape { dims: [1] },
                        ))
                        .unsqueeze()
                        .transpose(),
                        current_memory_state_tensors.clone(),
                    ),
                )
            }
        });

        let mut get_next_state = || {
            let memory = next_memory_states.next().unwrap();
            let interval = next_interval(memory.stability, desired_retention);
            ItemState { memory, interval }
        };

        NextStates {
            again: get_next_state(),
            hard: get_next_state(),
            good: get_next_state(),
            easy: get_next_state(),
        }
    }

    /// Determine how well the model and weights predict performance.
    /// Weights must have been provided when calling FSRS::new().
    pub fn evaluate<F>(&self, items: Vec<FSRSItem>, mut progress: F) -> Result<ModelEvaluation>
    where
        F: FnMut(ItemProgress) -> bool,
    {
        if items.is_empty() {
            return Err(FSRSError::NotEnoughData);
        }
        let batcher = FSRSBatcher::new(self.device());
        let mut all_predictions = vec![];
        let mut all_true_val = vec![];
        let mut all_retention = vec![];
        let mut all_labels = vec![];
        let mut progress_info = ItemProgress {
            current: 0,
            total: items.len(),
        };
        let model = self.model();
        for chunk in items.chunks(512) {
            let batch = batcher.batch(chunk.to_vec());
            let (_state, retention) = infer::<B>(model, batch.clone());
            let pred: Vec<f32> = retention.clone().squeeze::<1>(1).to_data().convert().value;
            all_predictions.extend(pred);
            let true_val: Vec<f32> = batch.labels.clone().float().to_data().convert().value;
            all_true_val.extend(true_val);
            all_retention.push(retention);
            all_labels.push(batch.labels);
            progress_info.current += chunk.len();
            if !progress(progress_info) {
                return Err(FSRSError::Interrupted);
            }
        }
        let rmse = calibration_rmse(&all_predictions, &all_true_val);
        let all_retention = Tensor::cat(all_retention, 0);
        let all_labels = Tensor::cat(all_labels, 0)
            .unsqueeze::<2>()
            .float()
            .transpose();
        let loss = BCELoss::<B>::new().forward(all_retention, all_labels);
        Ok(ModelEvaluation {
            log_loss: loss.to_data().value[0].elem(),
            rmse_bins: rmse,
        })
    }

    /// How well the user is likely to remember the item after `days_elapsed` since the previous
    /// review.
    pub fn current_retrievability(&self, state: MemoryState, days_elapsed: u32) -> f32 {
        (days_elapsed as f32 / (state.stability * 9.0) + 1.0).powf(-1.0)
    }
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
    pub interval: u32,
}

#[derive(Debug, Clone, Copy)]
pub struct ItemProgress {
    pub current: usize,
    pub total: usize,
}

fn get_bin(x: f32, bins: i32) -> i32 {
    let log_base = (bins.add(1) as f32).ln();
    let binned_x = (x * log_base).exp().floor().sub(1.0);
    (binned_x as i32).min(bins - 1).max(0)
}

fn calibration_rmse(pred: &[f32], true_val: &[f32]) -> f32 {
    if pred.len() != true_val.len() {
        panic!("Vectors pred and true_val must have the same length");
    }

    let mut groups = HashMap::new();

    for (p, t) in pred.iter().zip(true_val) {
        let bin = get_bin(*p, 20);
        groups.entry(bin).or_insert_with(Vec::new).push((p, t));
    }

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
    use crate::{convertor_tests::anki21_sample_file_converted_to_fsrs, FSRSReview};

    static WEIGHTS: &[f32] = &[
        0.81497127,
        1.5411042,
        4.007436,
        9.045982,
        4.9264183,
        1.039322,
        0.93803364,
        0.0,
        1.5530516,
        0.10299722,
        0.9981442,
        2.210701,
        0.018248068,
        0.3422524,
        1.3384504,
        0.22278537,
        2.6646678,
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
        let fsrs = FSRS::new(Some(WEIGHTS))?;
        assert_eq!(
            fsrs.memory_state(item),
            MemoryState {
                stability: 51.344814,
                difficulty: 7.005062
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
            .good
            .memory,
            MemoryState {
                stability: 51.344814,
                difficulty: 7.005062
            }
        );
        Ok(())
    }

    #[test]
    fn test_next_interval() {
        let request_retentions = (1..=10).map(|i| i as f32 / 10.0).collect::<Vec<_>>();
        let intervals = request_retentions
            .iter()
            .map(|r| next_interval(1.0, *r))
            .collect::<Vec<_>>();
        assert_eq!(intervals, [81, 36, 21, 14, 9, 6, 4, 2, 1, 1,]);
    }

    #[test]
    fn test_evaluate() -> Result<()> {
        let items = anki21_sample_file_converted_to_fsrs();
        let fsrs = FSRS::new(Some(&[]))?;

        let metrics = fsrs.evaluate(items.clone(), |_| true).unwrap();

        Data::from([metrics.log_loss, metrics.rmse_bins])
            .assert_approx_eq(&Data::from([0.20944944, 0.042762663]), 5);

        let fsrs = FSRS::new(Some(WEIGHTS))?;
        let metrics = fsrs.evaluate(items, |_| true).unwrap();

        Data::from([metrics.log_loss, metrics.rmse_bins])
            .assert_approx_eq(&Data::from([0.20320644, 0.016_822_13]), 5);
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
        let fsrs = FSRS::new(Some(WEIGHTS))?;
        let state = fsrs.memory_state(item);
        assert_eq!(
            fsrs.next_states(Some(state), 0.9, 21),
            NextStates {
                again: ItemState {
                    memory: MemoryState {
                        stability: 4.5604353,
                        difficulty: 8.881129,
                    },
                    interval: 5
                },
                hard: ItemState {
                    memory: MemoryState {
                        stability: 26.111229,
                        difficulty: 7.9430957
                    },
                    interval: 26,
                },
                good: ItemState {
                    memory: MemoryState {
                        stability: 51.344814,
                        difficulty: 7.005062
                    },
                    interval: 51,
                },
                easy: ItemState {
                    memory: MemoryState {
                        stability: 121.01552,
                        difficulty: 6.0670285
                    },
                    interval: 121,
                }
            }
        );
        Ok(())
    }

    #[test]
    fn states_are_unchaged_when_no_days_elapsed() -> Result<()> {
        let fsrs = FSRS::new(Some(&[]))?;
        // the first time a card is seen, a memory state must be set
        let mut state_a = fsrs.next_states(None, 1.0, 0).again.memory;
        // but if no days have elapsed and it's reviewed again, the state should be unchanged
        let state_b = fsrs.next_states(Some(state_a), 1.0, 0).again.memory;
        assert_eq!(state_a, state_b);
        // if a day elapses, it's counted
        state_a = fsrs.next_states(Some(state_a), 1.0, 1).again.memory;
        assert_ne!(state_a, state_b);

        Ok(())
    }
}

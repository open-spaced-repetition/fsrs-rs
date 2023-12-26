use std::collections::HashMap;
use std::ops::{Add, Sub};

use crate::model::{Get, MemoryStateTensors, FSRS};
use burn::tensor::{Data, Shape, Tensor};
use burn::{data::dataloader::batcher::Batcher, tensor::backend::Backend};

use crate::dataset::FSRSBatch;
use crate::dataset::FSRSBatcher;
use crate::error::Result;
use crate::model::Model;
use crate::training::BCELoss;
use crate::{FSRSError, FSRSItem};
use burn::tensor::ElementConversion;
pub(crate) const DECAY: f64 = -0.5;
/// (9/10) ^ (1 / DECAY) - 1
pub(crate) const FACTOR: f64 = 19f64 / 81f64;
/// This is a slice for efficiency, but should always be 17 in length.
pub type Weights = [f32];

pub static DEFAULT_WEIGHTS: [f32; 17] = [
    0.5614, 1.2546, 3.5878, 7.9731, 5.1043, 1.1303, 0.823, 0.0465, 1.629, 0.135, 1.0045, 2.132,
    0.0839, 0.3204, 1.3547, 0.219, 2.7849,
];

fn infer<B: Backend>(
    model: &Model<B>,
    batch: FSRSBatch<B>,
) -> (MemoryStateTensors<B>, Tensor<B, 1>) {
    let state = model.forward(batch.t_historys, batch.r_historys, None);
    let retention = model.power_forgetting_curve(batch.delta_ts, state.stability.clone());
    (state, retention)
}

#[derive(Debug, PartialEq, Clone, Copy)]
pub struct MemoryState {
    pub stability: f32,
    pub difficulty: f32,
}

impl<B: Backend> From<MemoryStateTensors<B>> for MemoryState {
    fn from(m: MemoryStateTensors<B>) -> Self {
        Self {
            stability: m.stability.to_data().value[0].elem(),
            difficulty: m.difficulty.to_data().value[0].elem(),
        }
    }
}

impl<B: Backend> From<MemoryState> for MemoryStateTensors<B> {
    fn from(m: MemoryState) -> Self {
        Self {
            stability: Tensor::from_data(Data::new(vec![m.stability.elem()], Shape { dims: [1] })),
            difficulty: Tensor::from_data(Data::new(
                vec![m.difficulty.elem()],
                Shape { dims: [1] },
            )),
        }
    }
}

pub fn next_interval(stability: f32, desired_retention: f32) -> u32 {
    (stability / FACTOR as f32 * (desired_retention.powf(1.0 / DECAY as f32) - 1.0))
        .round()
        .max(1.0) as u32
}

impl<B: Backend> FSRS<B> {
    /// Calculate the current memory state for a given card's history of reviews.
    /// In the case of truncated reviews, [starting_state] can be set to the value of
    /// [FSRS::memory_state_from_sm2] for the first review (which should not be included
    /// in FSRSItem). If not provided, the card starts as new.
    /// Weights must have been provided when calling FSRS::new().
    pub fn memory_state(
        &self,
        item: FSRSItem,
        starting_state: Option<MemoryState>,
    ) -> Result<MemoryState> {
        let (time_history, rating_history) =
            item.reviews.iter().map(|r| (r.delta_t, r.rating)).unzip();
        let size = item.reviews.len();
        let time_history =
            Tensor::from_data(Data::new(time_history, Shape { dims: [size] }).convert())
                .unsqueeze()
                .transpose();
        let rating_history =
            Tensor::from_data(Data::new(rating_history, Shape { dims: [size] }).convert())
                .unsqueeze()
                .transpose();
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

    /// If a card has incomplete learning history, memory state can be approximated from
    /// current sm2 values.
    /// Weights must have been provided when calling FSRS::new().
    pub fn memory_state_from_sm2(
        &self,
        ease_factor: f32,
        interval: f32,
        sm2_retention: f32,
    ) -> Result<MemoryState> {
        let stability = interval.max(0.1) / (9.0 * (1.0 / sm2_retention - 1.0));
        let w = &self.model().w;
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
    /// Weights must have been provided when calling FSRS::new().
    pub fn next_interval(
        &self,
        stability: Option<f32>,
        desired_retention: f32,
        rating: u32,
    ) -> u32 {
        let stability = stability.unwrap_or_else(|| {
            // get initial stability for new card
            let rating = Tensor::from_data(Data::new(vec![rating.elem()], Shape { dims: [1] }));
            let model = self.model();
            model.init_stability(rating).into_scalar().elem()
        });
        next_interval(stability, desired_retention)
    }

    /// The intervals and memory states for each answer button.
    /// Weights must have been provided when calling FSRS::new().
    pub fn next_states(
        &self,
        current_memory_state: Option<MemoryState>,
        desired_retention: f32,
        days_elapsed: u32,
    ) -> Result<NextStates> {
        let delta_t = Tensor::from_data(Data::new(vec![days_elapsed.elem()], Shape { dims: [1] }));
        let current_memory_state_tensors = current_memory_state.map(MemoryStateTensors::from);
        let model = self.model();
        let mut next_memory_states = (1..=4).map(|rating| {
            Ok(
                if let (Some(current_memory_state), 0) = (current_memory_state, days_elapsed) {
                    // When there's an existing memory state and no days have elapsed, we leave it unchanged.
                    current_memory_state
                } else {
                    let state = MemoryState::from(model.step(
                        delta_t.clone(),
                        Tensor::from_data(Data::new(vec![rating.elem()], Shape { dims: [1] })),
                        current_memory_state_tensors.clone(),
                    ));
                    if !state.stability.is_finite() || !state.difficulty.is_finite() {
                        return Err(FSRSError::InvalidInput);
                    }
                    state
                },
            )
        });

        let mut get_next_state = || {
            let memory = next_memory_states.next().unwrap()?;
            let interval = next_interval(memory.stability, desired_retention);
            Ok(ItemState { memory, interval })
        };

        Ok(NextStates {
            again: get_next_state()?,
            hard: get_next_state()?,
            good: get_next_state()?,
            easy: get_next_state()?,
        })
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
            let pred = retention.clone().to_data().convert::<f32>().value;
            all_predictions.extend(pred);
            let true_val = batch.labels.clone().to_data().convert::<f32>().value;
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
        let all_labels = Tensor::cat(all_labels, 0).float();
        let loss = BCELoss::new().forward(all_retention, all_labels);
        Ok(ModelEvaluation {
            log_loss: loss.to_data().value[0].elem(),
            rmse_bins: rmse,
        })
    }

    /// How well the user is likely to remember the item after `days_elapsed` since the previous
    /// review.
    pub fn current_retrievability(&self, state: MemoryState, days_elapsed: u32) -> f32 {
        (days_elapsed as f32 / (state.stability * 9.0) + 1.0).powi(-1)
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
    (binned_x as i32).clamp(0, bins - 1)
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
            fsrs.memory_state(item, None).unwrap(),
            MemoryState {
                stability: 51.31289,
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
            .unwrap()
            .good
            .memory,
            MemoryState {
                stability: 51.339684,
                difficulty: 7.005062
            }
        );
        Ok(())
    }

    #[test]
    fn test_next_interval() {
        let desired_retentions = (1..=10).map(|i| i as f32 / 10.0).collect::<Vec<_>>();
        let intervals = desired_retentions
            .iter()
            .map(|r| next_interval(1.0, *r))
            .collect::<Vec<_>>();
        assert_eq!(intervals, [422, 102, 43, 22, 13, 8, 4, 2, 1, 1]);
    }

    #[test]
    fn test_evaluate() -> Result<()> {
        let items = anki21_sample_file_converted_to_fsrs();
        let fsrs = FSRS::new(Some(&[]))?;

        let metrics = fsrs.evaluate(items.clone(), |_| true).unwrap();

        Data::from([metrics.log_loss, metrics.rmse_bins])
            .assert_approx_eq(&Data::from([0.205_166, 0.024_658]), 5);

        let fsrs = FSRS::new(Some(WEIGHTS))?;
        let metrics = fsrs.evaluate(items, |_| true).unwrap();

        Data::from([metrics.log_loss, metrics.rmse_bins])
            .assert_approx_eq(&Data::from([0.20306083, 0.01326745]), 5);
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
        let state = fsrs.memory_state(item, None).unwrap();
        assert_eq!(
            fsrs.next_states(Some(state), 0.9, 21).unwrap(),
            NextStates {
                again: ItemState {
                    memory: MemoryState {
                        stability: 4.577856,
                        difficulty: 8.881129,
                    },
                    interval: 5
                },
                hard: ItemState {
                    memory: MemoryState {
                        stability: 27.6745,
                        difficulty: 7.9430957
                    },
                    interval: 28,
                },
                good: ItemState {
                    memory: MemoryState {
                        stability: 51.31289,
                        difficulty: 7.005062
                    },
                    interval: 51,
                },
                easy: ItemState {
                    memory: MemoryState {
                        stability: 101.94249,
                        difficulty: 6.0670285
                    },
                    interval: 102,
                }
            }
        );
        assert_eq!(fsrs.next_interval(Some(121.01552), 0.9, 1), 121);
        Ok(())
    }

    #[test]
    fn states_are_unchaged_when_no_days_elapsed() -> Result<()> {
        let fsrs = FSRS::new(Some(&[]))?;
        // the first time a card is seen, a memory state must be set
        let mut state_a = fsrs.next_states(None, 1.0, 0)?.again.memory;
        // but if no days have elapsed and it's reviewed again, the state should be unchanged
        let state_b = fsrs.next_states(Some(state_a), 1.0, 0)?.again.memory;
        assert_eq!(state_a, state_b);
        // if a day elapses, it's counted
        state_a = fsrs.next_states(Some(state_a), 1.0, 1)?.again.memory;
        assert_ne!(state_a, state_b);

        Ok(())
    }

    #[test]
    fn memory_from_sm2() -> Result<()> {
        let fsrs = FSRS::new(Some(&[]))?;
        assert_eq!(
            fsrs.memory_state_from_sm2(2.5, 10.0, 0.9).unwrap(),
            MemoryState {
                stability: 9.999995,
                difficulty: 7.200902
            }
        );
        assert_eq!(
            fsrs.memory_state_from_sm2(1.3, 20.0, 0.9).unwrap(),
            MemoryState {
                stability: 19.99999,
                difficulty: 10.0
            }
        );
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

use std::sync::Mutex;

use burn::data::dataloader::batcher::Batcher;
use burn::data::dataloader::{DataLoaderIterator, Progress};
use burn::prelude::Backend;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use serde::{Deserialize, Serialize};

use crate::dataset::{FSRSBatch, FSRSBatcher, FSRSDataset};

#[derive(Clone)]
pub(crate) struct BatchTensorDataset<B: Backend> {
    dataset: Vec<FSRSBatch<B>>,
}

impl<B: Backend> BatchTensorDataset<B> {
    /// Creates a new shuffled dataset.
    pub fn new(dataset: FSRSDataset, batch_size: usize) -> Self {
        let device = B::Device::default();
        let batcher = FSRSBatcher::<B>::new();
        let dataset = dataset
            .items
            .chunks(batch_size)
            .map(|items| batcher.batch(items.to_vec(), &device))
            .collect();
        Self { dataset }
    }
}

impl<B: Backend> BatchTensorDataset<B> {
    fn get(&self, index: usize) -> Option<FSRSBatch<B>> {
        self.dataset.get(index).cloned()
    }

    fn len(&self) -> usize {
        self.dataset.len()
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub(crate) enum ShuffleStream {
    Train,
    Validate,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub(crate) struct ShuffleState {
    pub base_seed: u64,
    pub stream: ShuffleStream,
    pub epoch: usize,
    pub indices: Vec<usize>,
    pub current_index: usize,
}

impl ShuffleState {
    pub(crate) fn items_total(&self) -> usize {
        self.indices.len()
    }
}

pub struct ShuffleDataLoader<B: Backend> {
    dataset: BatchTensorDataset<B>,
    base_seed: u64,
    stream: ShuffleStream,
    #[cfg_attr(not(test), allow(dead_code))]
    next_epoch: Mutex<usize>,
}

impl<B: Backend> ShuffleDataLoader<B> {
    #[cfg_attr(not(test), allow(dead_code))]
    pub fn new(dataset: BatchTensorDataset<B>, seed: u64) -> Self {
        Self::new_with_stream(dataset, seed, ShuffleStream::Train)
    }

    pub(crate) fn new_with_stream(
        dataset: BatchTensorDataset<B>,
        seed: u64,
        stream: ShuffleStream,
    ) -> Self {
        Self {
            dataset,
            base_seed: seed,
            stream,
            next_epoch: Mutex::new(1),
        }
    }
}

pub(crate) struct ShuffleDataLoaderIterator<B: Backend> {
    state: ShuffleState,
    dataset: BatchTensorDataset<B>,
}

impl<B: Backend> ShuffleDataLoaderIterator<B> {
    pub(crate) fn new(dataset: BatchTensorDataset<B>, state: ShuffleState) -> Self {
        Self { state, dataset }
    }

    pub(crate) fn progress(&self) -> Progress {
        Progress {
            items_processed: self.state.current_index,
            items_total: self.dataset.len(),
        }
    }

    pub(crate) fn state(&self) -> ShuffleState {
        self.state.clone()
    }
}

impl<B: Backend> Iterator for ShuffleDataLoaderIterator<B> {
    type Item = FSRSBatch<B>;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(index) = self.state.indices.get(self.state.current_index) {
            self.state.current_index += 1;
            return self.dataset.get(*index);
        }
        None
    }
}

impl<B: Backend> DataLoaderIterator<FSRSBatch<B>> for ShuffleDataLoaderIterator<B> {
    fn progress(&self) -> Progress {
        Progress::new(self.state.current_index, self.dataset.len())
    }
}

impl<B: Backend> ShuffleDataLoader<B> {
    fn mixed_seed(&self, epoch: usize) -> u64 {
        const EPOCH_MAGIC: u64 = 0x9E37_79B9_7F4A_7C15;
        const TRAIN_MAGIC: u64 = 0xA24B_1C62_4B27_F1A5;
        const VALIDATE_MAGIC: u64 = 0xC2B2_AE35_79D9_82EF;

        let stream_magic = match self.stream {
            ShuffleStream::Train => TRAIN_MAGIC,
            ShuffleStream::Validate => VALIDATE_MAGIC,
        };

        self.base_seed ^ stream_magic ^ (epoch as u64).wrapping_mul(EPOCH_MAGIC)
    }

    pub(crate) fn state_for_epoch(&self, epoch: usize) -> ShuffleState {
        let mut indices: Vec<_> = (0..self.dataset.len()).collect();
        indices.shuffle(&mut StdRng::seed_from_u64(self.mixed_seed(epoch)));
        ShuffleState {
            base_seed: self.base_seed,
            stream: self.stream,
            epoch,
            indices,
            current_index: 0,
        }
    }

    pub(crate) fn is_compatible_state(&self, state: &ShuffleState) -> bool {
        state.base_seed == self.base_seed
            && state.stream == self.stream
            && state.indices.len() == self.dataset.len()
    }

    pub(crate) fn iter_from_state(&self, state: ShuffleState) -> ShuffleDataLoaderIterator<B> {
        ShuffleDataLoaderIterator::new(self.dataset.clone(), state)
    }

    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn iter(&self) -> ShuffleDataLoaderIterator<B> {
        let mut next_epoch = self.next_epoch.lock().unwrap();
        let state = self.state_for_epoch(*next_epoch);
        *next_epoch += 1;
        self.iter_from_state(state)
    }
}

#[cfg(test)]
mod tests {
    use burn::{backend::NdArray, tensor::Shape};
    use itertools::Itertools;

    use super::*;
    use crate::{
        convertor_tests::anki21_sample_file_converted_to_fsrs,
        dataset::{constant_weighted_fsrs_items, prepare_training_data},
    };

    #[test]
    fn test_simple_dataloader() {
        let train_set = anki21_sample_file_converted_to_fsrs()
            .into_iter()
            .sorted_by_cached_key(|item| item.reviews.len())
            .collect();
        let (_pre_train_set, train_set) = prepare_training_data(train_set);
        let dataset = FSRSDataset::from(constant_weighted_fsrs_items(train_set));
        let batch_size = 512;
        let seed = 114514;
        type Backend = NdArray<f32>;

        let dataset = BatchTensorDataset::<Backend>::new(dataset, batch_size);
        let dataloader = ShuffleDataLoader::new(dataset, seed);
        let mut iterator = dataloader.iter();
        let batch = iterator.next().unwrap();
        assert_eq!(
            batch.t_historys.shape(),
            Shape {
                dims: vec![5, batch_size]
            }
        );
        let batch = iterator.next().unwrap();
        assert_eq!(
            batch.t_historys.shape(),
            Shape {
                dims: vec![3, batch_size]
            }
        );

        let lengths = iterator
            .map(|batch| batch.t_historys.shape().dims[0])
            .collect::<Vec<_>>();
        assert_eq!(
            lengths,
            [
                3, 5, 19, 7, 2, 4, 4, 3, 6, 13, 4, 4, 7, 4, 6, 48, 11, 8, 9, 1, 2, 5, 3, 5, 6, 3
            ]
        );

        let mut iterator = dataloader.iter();
        let batch = iterator.next().unwrap();
        assert_eq!(
            batch.t_historys.shape(),
            Shape {
                dims: vec![4, batch_size]
            }
        );
        let batch = iterator.next().unwrap();
        assert_eq!(
            batch.t_historys.shape(),
            Shape {
                dims: vec![2, batch_size]
            }
        );

        let lengths = iterator
            .map(|batch| batch.t_historys.shape().dims[0])
            .collect::<Vec<_>>();
        assert_eq!(
            lengths,
            [
                11, 4, 5, 3, 1, 3, 13, 5, 4, 6, 2, 6, 19, 6, 3, 7, 4, 3, 48, 9, 5, 8, 5, 4, 3, 7
            ]
        );
    }

    #[test]
    fn test_state_round_trip() {
        let dataset = FSRSDataset::from(constant_weighted_fsrs_items(vec![
            crate::FSRSItem {
                reviews: vec![
                    crate::FSRSReview {
                        rating: 3,
                        delta_t: 0,
                    },
                    crate::FSRSReview {
                        rating: 4,
                        delta_t: 1,
                    },
                ],
            },
            crate::FSRSItem {
                reviews: vec![
                    crate::FSRSReview {
                        rating: 1,
                        delta_t: 0,
                    },
                    crate::FSRSReview {
                        rating: 3,
                        delta_t: 0,
                    },
                    crate::FSRSReview {
                        rating: 4,
                        delta_t: 2,
                    },
                ],
            },
            crate::FSRSItem {
                reviews: vec![
                    crate::FSRSReview {
                        rating: 2,
                        delta_t: 0,
                    },
                    crate::FSRSReview {
                        rating: 3,
                        delta_t: 1,
                    },
                    crate::FSRSReview {
                        rating: 4,
                        delta_t: 5,
                    },
                ],
            },
            crate::FSRSItem {
                reviews: vec![
                    crate::FSRSReview {
                        rating: 4,
                        delta_t: 0,
                    },
                    crate::FSRSReview {
                        rating: 3,
                        delta_t: 3,
                    },
                ],
            },
            crate::FSRSItem {
                reviews: vec![
                    crate::FSRSReview {
                        rating: 3,
                        delta_t: 0,
                    },
                    crate::FSRSReview {
                        rating: 2,
                        delta_t: 1,
                    },
                    crate::FSRSReview {
                        rating: 3,
                        delta_t: 2,
                    },
                    crate::FSRSReview {
                        rating: 4,
                        delta_t: 8,
                    },
                ],
            },
            crate::FSRSItem {
                reviews: vec![
                    crate::FSRSReview {
                        rating: 1,
                        delta_t: 0,
                    },
                    crate::FSRSReview {
                        rating: 3,
                        delta_t: 0,
                    },
                    crate::FSRSReview {
                        rating: 3,
                        delta_t: 1,
                    },
                    crate::FSRSReview {
                        rating: 4,
                        delta_t: 6,
                    },
                ],
            },
        ]));
        type Backend = NdArray<f32>;

        let dataset = BatchTensorDataset::<Backend>::new(dataset, 2);
        let dataloader =
            ShuffleDataLoader::new_with_stream(dataset, 114514, ShuffleStream::Validate);
        let mut iterator = dataloader.iter();
        iterator.next();
        iterator.next();
        let state = iterator.state();

        assert!(dataloader.is_compatible_state(&state));

        let resumed_lengths = dataloader
            .iter_from_state(state)
            .map(|batch| batch.t_historys.shape().dims[0])
            .collect::<Vec<_>>();
        let fresh_lengths = dataloader
            .iter_from_state({
                let mut state = dataloader.state_for_epoch(1);
                state.current_index = 2;
                state
            })
            .map(|batch| batch.t_historys.shape().dims[0])
            .collect::<Vec<_>>();

        assert_eq!(resumed_lengths, fresh_lengths);
    }
}

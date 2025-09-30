use std::sync::Mutex;

use burn::data::dataloader::batcher::Batcher;
use burn::data::dataloader::{DataLoaderIterator, Progress};
use burn::prelude::Backend;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;

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

pub struct ShuffleDataLoader<B: Backend> {
    dataset: BatchTensorDataset<B>,
    rng: Mutex<StdRng>,
}

impl<B: Backend> ShuffleDataLoader<B> {
    pub fn new(dataset: BatchTensorDataset<B>, seed: u64) -> Self {
        Self {
            dataset,
            rng: Mutex::new(StdRng::seed_from_u64(seed)),
        }
    }
}

pub(crate) struct ShuffleDataLoaderIterator<B: Backend> {
    current_index: usize,
    indices: Vec<usize>,
    dataset: BatchTensorDataset<B>,
}

impl<B: Backend> ShuffleDataLoaderIterator<B> {
    pub(crate) fn new(dataset: BatchTensorDataset<B>, indices: Vec<usize>) -> Self {
        Self {
            current_index: 0,
            indices,
            dataset,
        }
    }

    pub(crate) fn progress(&self) -> Progress {
        Progress {
            items_processed: self.current_index,
            items_total: self.dataset.len(),
        }
    }
}

impl<B: Backend> Iterator for ShuffleDataLoaderIterator<B> {
    type Item = FSRSBatch<B>;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(index) = self.indices.get(self.current_index) {
            self.current_index += 1;
            return self.dataset.get(*index);
        }
        None
    }
}

impl<B: Backend> DataLoaderIterator<FSRSBatch<B>> for ShuffleDataLoaderIterator<B> {
    fn progress(&self) -> Progress {
        Progress::new(self.current_index, self.dataset.len())
    }
}

impl<B: Backend> ShuffleDataLoader<B> {
    pub(crate) fn iter(&self) -> ShuffleDataLoaderIterator<B> {
        let mut indices: Vec<_> = (0..self.dataset.len()).collect();
        indices.shuffle(&mut *self.rng.lock().unwrap());
        ShuffleDataLoaderIterator::new(self.dataset.clone(), indices)
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
        // dbg!(&iterator.indices);
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
        // dbg!(&iterator.indices);
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
}

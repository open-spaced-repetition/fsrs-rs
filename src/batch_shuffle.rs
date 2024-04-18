use burn::data::{
    dataloader::{
        batcher::DynBatcher, BatchStrategy, DataLoader, DataLoaderIterator, FixBatchStrategy,
        Progress,
    },
    dataset::Dataset,
};

use rand::{distributions::Standard, prelude::SliceRandom, rngs::StdRng, Rng, SeedableRng};
use std::{
    marker::PhantomData,
    sync::{Arc, Mutex},
};

use crate::{dataset::FSRSDataset, FSRSItem};

pub(crate) struct BatchShuffledDataset<I> {
    dataset: Arc<FSRSDataset>,
    indices: Vec<usize>,
    input: PhantomData<I>,
}

impl<FSRSItem> BatchShuffledDataset<FSRSItem> {
    /// Creates a new shuffled dataset.
    pub fn new(dataset: Arc<FSRSDataset>, batch_size: usize, rng: &mut StdRng) -> Self {
        let len = dataset.len();

        // Calculate the number of batches
        // 计算批数
        let num_batches = (len + batch_size - 1) / batch_size;

        // Create a vector of batch indices and shuffle it
        // 创建一个批数索引的向量并打乱
        let mut batch_indices: Vec<_> = (0..num_batches).collect();
        batch_indices.shuffle(rng);
        // info!("batch_indices: {:?}", &batch_indices);
        // Generate the corresponding item indices for each shuffled batch
        // 为每个打乱的批次生成相应的元素索引
        let mut indices = vec![];
        for batch_index in batch_indices {
            let start_index = batch_index * batch_size;
            let end_index = (start_index + batch_size).min(len);
            indices.extend(start_index..end_index);
        }
        // info!("indices: {:?}", &indices);
        Self {
            dataset,
            indices,
            input: PhantomData,
        }
    }

    /// Creates a new shuffled dataset with a fixed seed.
    pub fn with_seed(dataset: Arc<FSRSDataset>, batch_size: usize, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        Self::new(dataset, batch_size, &mut rng)
    }
}

impl Dataset<FSRSItem> for BatchShuffledDataset<FSRSItem> {
    fn get(&self, index: usize) -> Option<FSRSItem> {
        let Some(shuffled_index) = self.indices.get(index) else {
            return None;
        };
        // info!(
        //     "original index: {}, shuffled index: {}",
        //     index, shuffled_index
        // );
        self.dataset.get(*shuffled_index)
    }

    fn len(&self) -> usize {
        self.dataset.len()
    }
}

/// A data loader that can be used to iterate over a dataset in batches.
pub(crate) struct BatchShuffledDataLoader<I, O> {
    strategy: Box<dyn BatchStrategy<I>>,
    dataset: Arc<FSRSDataset>,
    batcher: Box<dyn DynBatcher<I, O>>,
    rng: Mutex<rand::rngs::StdRng>,
    batch_size: usize,
}

impl<I, O> BatchShuffledDataLoader<I, O> {
    /// Creates a new batch data loader.
    ///
    /// # Arguments
    ///
    /// * `strategy` - The batch strategy.
    /// * `dataset` - The dataset.
    /// * `batcher` - The batcher.
    /// * `rng`     - The rng determining if the dataset is shuffled each time a dataloader
    ///               iterator is created.
    ///
    /// # Returns
    ///
    /// The batch data loader.
    pub fn new(
        strategy: Box<dyn BatchStrategy<I>>,
        dataset: Arc<FSRSDataset>,
        batcher: Box<dyn DynBatcher<I, O>>,
        rng: rand::rngs::StdRng,
        batch_size: usize,
    ) -> Self {
        Self {
            strategy,
            dataset,
            batcher,
            rng: Mutex::new(rng),
            batch_size,
        }
    }
}

/// A data loader iterator that can be used to iterate over a data loader.
struct BatchShuffledDataloaderIterator<I, O> {
    current_index: usize,
    strategy: Box<dyn BatchStrategy<I>>,
    dataset: Arc<dyn Dataset<I>>,
    batcher: Box<dyn DynBatcher<I, O>>,
}

impl<I: Send + Sync + Clone + 'static, O: Send> DataLoader<O> for BatchShuffledDataLoader<I, O>
where
    BatchShuffledDataset<I>: Dataset<I>,
{
    fn iter<'a>(&'a self) -> Box<dyn DataLoaderIterator<O> + 'a> {
        // When starting a new iteration, we first check if the dataloader was created with an rng,
        // implying that we should shuffle the dataset beforehand, while advancing the current
        // rng to ensure that each new iteration shuffles the dataset differently.
        let dataset = Arc::new(BatchShuffledDataset::with_seed(
            self.dataset.clone(),
            self.batch_size,
            self.rng.lock().unwrap().sample(Standard),
        ));
        Box::new(BatchShuffledDataloaderIterator::new(
            self.strategy.clone_dyn(),
            dataset,
            self.batcher.clone_dyn(),
        ))
    }

    fn num_items(&self) -> usize {
        self.dataset.len()
    }
}

impl<I: 'static, O> BatchShuffledDataloaderIterator<I, O>
where
    BatchShuffledDataset<I>: Dataset<I>,
{
    /// Creates a new batch data loader iterator.
    ///
    /// # Arguments
    ///
    /// * `strategy` - The batch strategy.
    /// * `dataset` - The dataset.
    /// * `batcher` - The batcher.
    ///
    /// # Returns
    ///
    /// The batch data loader iterator.
    pub fn new(
        strategy: Box<dyn BatchStrategy<I>>,
        dataset: Arc<BatchShuffledDataset<I>>,
        batcher: Box<dyn DynBatcher<I, O>>,
    ) -> Self {
        Self {
            current_index: 0,
            strategy,
            dataset,
            batcher,
        }
    }
}

impl<I, O> Iterator for BatchShuffledDataloaderIterator<I, O> {
    type Item = O;

    fn next(&mut self) -> Option<O> {
        while let Some(item) = self.dataset.get(self.current_index) {
            self.current_index += 1;
            self.strategy.add(item);

            if let Some(items) = self.strategy.batch(false) {
                return Some(self.batcher.batch(items));
            }
        }

        let Some(items) = self.strategy.batch(true) else {
            return None;
        };

        Some(self.batcher.batch(items))
    }
}

impl<I, O> DataLoaderIterator<O> for BatchShuffledDataloaderIterator<I, O> {
    fn progress(&self) -> Progress {
        Progress {
            items_processed: self.current_index,
            items_total: self.dataset.len(),
        }
    }
}

/// A builder for data loaders.
pub struct BatchShuffledDataLoaderBuilder<I, O> {
    batcher: Box<dyn DynBatcher<I, O>>,
}

impl<I, O> BatchShuffledDataLoaderBuilder<I, O>
where
    I: Send + Sync + Clone + std::fmt::Debug + 'static,
    O: Send + Clone + std::fmt::Debug + 'static,
    BatchShuffledDataset<I>: Dataset<I>,
{
    /// Creates a new data loader builder.
    ///
    /// # Arguments
    ///
    /// * `batcher` - The batcher.
    ///
    /// # Returns
    ///
    /// The data loader builder.
    pub fn new<B>(batcher: B) -> Self
    where
        B: DynBatcher<I, O> + 'static,
    {
        Self {
            batcher: Box::new(batcher),
        }
    }

    /// Builds the data loader.
    ///
    /// # Arguments
    ///
    /// * `dataset` - The dataset.
    ///
    /// # Returns
    ///
    /// The data loader.
    pub fn build(
        self,
        dataset: FSRSDataset,
        batch_size: usize,
        seed: u64,
    ) -> Arc<dyn DataLoader<O>> {
        let dataset = Arc::new(dataset);

        let rng = StdRng::seed_from_u64(seed);
        let strategy = Box::new(FixBatchStrategy::new(batch_size));

        Arc::new(BatchShuffledDataLoader::new(
            strategy,
            dataset,
            self.batcher,
            rng,
            batch_size,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{convertor_tests::anki21_sample_file_converted_to_fsrs, FSRSItem, FSRSReview};

    #[test]
    fn batch_shuffle() {
        use crate::dataset::FSRSDataset;
        let dataset = Arc::new(FSRSDataset::from(anki21_sample_file_converted_to_fsrs()));
        let batch_size = 10;
        let seed = 42;
        let batch_shuffled_dataset = BatchShuffledDataset::with_seed(dataset, batch_size, seed);
        assert_eq!(
            (0..batch_shuffled_dataset.len().min(batch_size))
                .map(|i| batch_shuffled_dataset.get(i).unwrap())
                .collect::<Vec<_>>(),
            [
                FSRSItem {
                    reviews: vec![
                        FSRSReview {
                            rating: 4,
                            delta_t: 0,
                        },
                        FSRSReview {
                            rating: 3,
                            delta_t: 1,
                        },
                        FSRSReview {
                            rating: 3,
                            delta_t: 3,
                        }
                    ]
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
                        }
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
                        }
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
                        }
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
                        }
                    ],
                },
                FSRSItem {
                    reviews: vec![
                        FSRSReview {
                            rating: 4,
                            delta_t: 0,
                        },
                        FSRSReview {
                            rating: 4,
                            delta_t: 3,
                        },
                        FSRSReview {
                            rating: 3,
                            delta_t: 10,
                        }
                    ],
                },
                FSRSItem {
                    reviews: vec![
                        FSRSReview {
                            rating: 4,
                            delta_t: 0,
                        },
                        FSRSReview {
                            rating: 4,
                            delta_t: 1,
                        },
                        FSRSReview {
                            rating: 3,
                            delta_t: 4,
                        }
                    ],
                },
                FSRSItem {
                    reviews: vec![
                        FSRSReview {
                            rating: 4,
                            delta_t: 0,
                        },
                        FSRSReview {
                            rating: 4,
                            delta_t: 1,
                        },
                        FSRSReview {
                            rating: 3,
                            delta_t: 4,
                        }
                    ],
                },
                FSRSItem {
                    reviews: vec![
                        FSRSReview {
                            rating: 4,
                            delta_t: 0,
                        },
                        FSRSReview {
                            rating: 3,
                            delta_t: 5,
                        },
                        FSRSReview {
                            rating: 3,
                            delta_t: 11,
                        }
                    ],
                },
                FSRSItem {
                    reviews: vec![
                        FSRSReview {
                            rating: 4,
                            delta_t: 0,
                        },
                        FSRSReview {
                            rating: 3,
                            delta_t: 1,
                        },
                        FSRSReview {
                            rating: 3,
                            delta_t: 3,
                        }
                    ],
                },
            ]
        );
    }

    #[test]
    fn item_shuffle() {
        use crate::dataset::FSRSDataset;
        use burn::data::dataset::transform::ShuffledDataset;
        let dataset = FSRSDataset::from(anki21_sample_file_converted_to_fsrs());
        let seed = 42;
        let shuffled_dataset = ShuffledDataset::with_seed(dataset, seed);
        for i in 0..shuffled_dataset.len().min(10) {
            dbg!(shuffled_dataset.get(i).unwrap());
        }
    }
}

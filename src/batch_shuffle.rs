use burn::data::{
    dataloader::{
        batcher::Batcher, BatchStrategy, DataLoader, DataLoaderIterator, FixBatchStrategy,
        MultiThreadDataLoader, Progress,
    },
    dataset::{transform::PartialDataset, Dataset},
};
use log::info;
use rand::{
    distributions::Standard,
    prelude::{Distribution, SliceRandom},
    rngs::StdRng,
    Rng, SeedableRng,
};
use std::{
    marker::PhantomData,
    sync::{Arc, Mutex},
};

pub(crate) struct BatchShuffledDataset<D, I> {
    dataset: D,
    indices: Vec<usize>,
    input: PhantomData<I>,
}

impl<D, I> BatchShuffledDataset<D, I>
where
    D: Dataset<I>,
{
    /// Creates a new shuffled dataset.
    pub fn new(dataset: D, batch_size: usize, rng: &mut StdRng) -> Self {
        let len = dataset.len();

        // Calculate the number of batches
        // 计算批数
        let num_batches = (len + batch_size - 1) / batch_size;

        // Create a vector of batch indices and shuffle it
        // 创建一个批数索引的向量并打乱
        let mut batch_indices: Vec<_> = (0..num_batches).collect();
        batch_indices.shuffle(rng);
        info!("batch_indices: {:?}", &batch_indices);
        // Generate the corresponding item indices for each shuffled batch
        // 为每个打乱的批次生成相应的元素索引
        let mut indices = vec![];
        for batch_index in batch_indices {
            let start_index = batch_index * batch_size;
            let end_index = (start_index + batch_size).min(len);
            indices.extend(start_index..end_index);
        }
        info!("indices: {:?}", &indices);
        Self {
            dataset,
            indices,
            input: PhantomData,
        }
    }

    /// Creates a new shuffled dataset with a fixed seed.
    pub fn with_seed(dataset: D, batch_size: usize, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        Self::new(dataset, batch_size, &mut rng)
    }
}

impl<D, I> Dataset<I> for BatchShuffledDataset<D, I>
where
    D: Dataset<I>,
    I: Clone + Send + Sync,
{
    fn get(&self, index: usize) -> Option<I> {
        if let Some(index) = self.indices.get(index) {
            self.dataset.get(*index)
        } else {
            None
        }
    }

    fn len(&self) -> usize {
        self.dataset.len()
    }
}

/// A data loader that can be used to iterate over a dataset in batches.
pub struct BatchShuffledDataLoader<I, O> {
    strategy: Box<dyn BatchStrategy<I>>,
    dataset: Arc<dyn Dataset<I>>,
    batcher: Arc<dyn Batcher<I, O>>,
    rng: Option<Mutex<rand::rngs::StdRng>>,
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
        dataset: Arc<dyn Dataset<I>>,
        batcher: Arc<dyn Batcher<I, O>>,
        rng: Option<rand::rngs::StdRng>,
        batch_size: usize,
    ) -> Self {
        Self {
            strategy,
            dataset,
            batcher,
            rng: rng.map(Mutex::new),
            batch_size,
        }
    }
}

/// A data loader iterator that can be used to iterate over a data loader.
struct BatchShuffledDataloaderIterator<I, O> {
    current_index: usize,
    strategy: Box<dyn BatchStrategy<I>>,
    dataset: Arc<dyn Dataset<I>>,
    batcher: Arc<dyn Batcher<I, O>>,
}

impl<I, O> BatchShuffledDataLoader<I, O>
where
    I: Send + Sync + Clone + 'static,
    O: Send + Sync + Clone + 'static,
{
    /// Creates a new multi-threaded batch data loader.
    ///
    /// # Arguments
    ///
    /// * `strategy` - The batch strategy.
    /// * `dataset` - The dataset.
    /// * `batcher` - The batcher.
    /// * `num_threads` - The number of threads.
    ///
    /// # Returns
    ///
    /// The multi-threaded batch data loader.
    pub fn multi_thread(
        strategy: Box<dyn BatchStrategy<I>>,
        dataset: Arc<dyn Dataset<I>>,
        batcher: Arc<dyn Batcher<I, O>>,
        num_threads: usize,
        mut rng: Option<rand::rngs::StdRng>,
        batch_size: usize,
    ) -> MultiThreadDataLoader<O> {
        let datasets = PartialDataset::split(dataset, num_threads);

        let mut dataloaders: Vec<Arc<dyn DataLoader<_> + Send + Sync>> =
            Vec::with_capacity(num_threads);

        // Create more rngs from the first one, one for each new dataloader.
        let rngs = (0..num_threads).map(|_| {
            rng.as_mut()
                .map(|rng| StdRng::seed_from_u64(Distribution::sample(&Standard, rng)))
        });

        for (dataset, rng) in datasets.into_iter().zip(rngs) {
            let strategy = strategy.new_like();
            let dataloader = BatchShuffledDataLoader::new(
                strategy,
                Arc::new(dataset),
                batcher.clone(),
                rng,
                batch_size,
            );
            let dataloader = Arc::new(dataloader);
            dataloaders.push(dataloader);
        }
        MultiThreadDataLoader::new(dataloaders)
    }
}

impl<I: Send + Sync + Clone + 'static, O: Send + Sync> DataLoader<O>
    for BatchShuffledDataLoader<I, O>
{
    fn iter<'a>(&'a self) -> Box<dyn DataLoaderIterator<O> + 'a> {
        // When starting a new iteration, we first check if the dataloader was created with an rng,
        // implying that we should shuffle the dataset beforehand, while advancing the current
        // rng to ensure that each new iteration shuffles the dataset differently.
        let dataset = match &self.rng {
            Some(rng) => {
                let rng = rng.lock();

                Arc::new(BatchShuffledDataset::with_seed(
                    self.dataset.clone(),
                    self.batch_size,
                    rng.unwrap().sample(Standard),
                ))
            }
            None => self.dataset.clone(),
        };
        Box::new(BatchShuffledDataloaderIterator::new(
            self.strategy.new_like(),
            dataset,
            self.batcher.clone(),
        ))
    }
}

impl<I, O> BatchShuffledDataloaderIterator<I, O> {
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
        dataset: Arc<dyn Dataset<I>>,
        batcher: Arc<dyn Batcher<I, O>>,
    ) -> Self {
        BatchShuffledDataloaderIterator {
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

        if let Some(items) = self.strategy.batch(true) {
            return Some(self.batcher.batch(items));
        }

        None
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
    strategy: Option<Box<dyn BatchStrategy<I>>>,
    batcher: Arc<dyn Batcher<I, O>>,
    num_threads: Option<usize>,
    shuffle: Option<u64>,
}

impl<I, O> BatchShuffledDataLoaderBuilder<I, O>
where
    I: Send + Sync + Clone + std::fmt::Debug + 'static,
    O: Send + Sync + Clone + std::fmt::Debug + 'static,
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
        B: Batcher<I, O> + 'static,
    {
        Self {
            batcher: Arc::new(batcher),
            strategy: None,
            num_threads: None,
            shuffle: None,
        }
    }

    /// Sets the batch size to a fix number.The [fix batch strategy](FixBatchStrategy)
    /// will be used.
    ///
    /// # Arguments
    ///
    /// * `batch_size` - The batch size.
    ///
    /// # Returns
    ///
    /// The data loader builder.
    pub fn batch_size(mut self, batch_size: usize) -> Self {
        self.strategy = Some(Box::new(FixBatchStrategy::new(batch_size)));
        self
    }

    /// Sets the seed for shuffling.
    ///
    /// Each time the dataloader starts a new iteration, the dataset will be shuffled.
    ///
    /// # Arguments
    ///
    /// * `seed` - The seed.
    ///
    /// # Returns
    ///
    /// The data loader builder.
    pub fn shuffle(mut self, seed: u64) -> Self {
        self.shuffle = Some(seed);
        self
    }

    /// Sets the number of workers.
    ///
    /// # Arguments
    ///
    /// * `num_workers` - The number of workers.
    ///
    /// # Returns
    ///
    /// The data loader builder.
    pub fn num_workers(mut self, num_workers: usize) -> Self {
        self.num_threads = Some(num_workers);
        self
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
    pub fn build<D>(self, dataset: D, batch_size: usize) -> Arc<dyn DataLoader<O>>
    where
        D: Dataset<I> + 'static,
    {
        let dataset = Arc::new(dataset);

        let rng = self.shuffle.map(StdRng::seed_from_u64);
        let strategy = match self.strategy {
            Some(strategy) => strategy,
            None => Box::new(FixBatchStrategy::new(1)),
        };
        if let Some(num_threads) = self.num_threads {
            return Arc::new(BatchShuffledDataLoader::multi_thread(
                strategy,
                dataset,
                self.batcher,
                num_threads,
                rng,
                batch_size,
            ));
        }

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
        let dataset = FSRSDataset::from(anki21_sample_file_converted_to_fsrs());
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
                        },
                        FSRSReview {
                            rating: 3,
                            delta_t: 8,
                        },
                        FSRSReview {
                            rating: 3,
                            delta_t: 21,
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
                        },
                        FSRSReview {
                            rating: 3,
                            delta_t: 7,
                        },
                        FSRSReview {
                            rating: 3,
                            delta_t: 19,
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
                        },
                        FSRSReview {
                            rating: 3,
                            delta_t: 8,
                        },
                        FSRSReview {
                            rating: 3,
                            delta_t: 19,
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
                        },
                        FSRSReview {
                            rating: 3,
                            delta_t: 6,
                        },
                        FSRSReview {
                            rating: 3,
                            delta_t: 17,
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
                        },
                        FSRSReview {
                            rating: 3,
                            delta_t: 8,
                        },
                        FSRSReview {
                            rating: 3,
                            delta_t: 20,
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
                            delta_t: 1,
                        },
                        FSRSReview {
                            rating: 3,
                            delta_t: 3,
                        },
                        FSRSReview {
                            rating: 3,
                            delta_t: 8,
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
                            delta_t: 1,
                        },
                        FSRSReview {
                            rating: 3,
                            delta_t: 3,
                        },
                        FSRSReview {
                            rating: 3,
                            delta_t: 5,
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
                            delta_t: 1,
                        },
                        FSRSReview {
                            rating: 3,
                            delta_t: 3,
                        },
                        FSRSReview {
                            rating: 3,
                            delta_t: 5,
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
                        },
                        FSRSReview {
                            rating: 3,
                            delta_t: 9,
                        },
                        FSRSReview {
                            rating: 3,
                            delta_t: 19,
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

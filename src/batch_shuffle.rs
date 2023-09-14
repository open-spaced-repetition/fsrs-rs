use burn::data::dataset::Dataset;
use rand::{prelude::SliceRandom, rngs::StdRng, SeedableRng};
use std::marker::PhantomData;

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

        // Generate the corresponding item indices for each shuffled batch
        // 为每个打乱的批次生成相应的元素索引
        let mut indices = vec![];
        for batch_index in batch_indices {
            let start_index = batch_index * batch_size;
            let end_index = (start_index + batch_size).min(len);
            indices.extend(start_index..end_index);
        }

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

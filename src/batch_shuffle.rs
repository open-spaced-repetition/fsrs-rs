use burn::data::dataset::Dataset;
use rand::{prelude::SliceRandom, rngs::StdRng, SeedableRng};
use std::marker::PhantomData;

pub struct BatchShuffledDataset<D, I> {
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
        let mut batch_indices: Vec<usize> = (0..num_batches).collect();
        batch_indices.shuffle(rng);

        // Generate the corresponding item indices for each shuffled batch
        // 为每个打乱的批次生成相应的元素索引
        let mut indices: Vec<usize> = Vec::new();
        for &batch_index in &batch_indices {
            let start_index = batch_index * batch_size;
            let end_index = std::cmp::min(start_index + batch_size, len);
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
        let index = match self.indices.get(index) {
            Some(index) => index,
            None => return None,
        };
        self.dataset.get(*index)
    }

    fn len(&self) -> usize {
        self.dataset.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn batch_shuffle() {
        use crate::dataset::FSRSDataset;
        let dataset = FSRSDataset::train();
        let batch_size = 10;
        let seed = 42;
        let batch_shuffled_dataset: BatchShuffledDataset<FSRSDataset, crate::dataset::FSRSItem> =
            BatchShuffledDataset::with_seed(dataset, batch_size, seed);
        for i in 0..batch_shuffled_dataset.len() {
            println!("{:?}", batch_shuffled_dataset.get(i).unwrap());
            if i > batch_size {
                break;
            }
        }
    }

    #[test]
    fn item_shuffle() {
        use crate::dataset::FSRSDataset;
        use burn::data::dataset::transform::ShuffledDataset;
        let dataset = FSRSDataset::train();
        let seed = 42;
        let shuffled_dataset: ShuffledDataset<FSRSDataset, crate::dataset::FSRSItem> =
            ShuffledDataset::with_seed(dataset, seed);
        for i in 0..shuffled_dataset.len() {
            println!("{:?}", shuffled_dataset.get(i).unwrap());
            if i > 10 {
                break;
            }
        }
    }
}

use std::collections::{HashMap, HashSet};

// candle imports
use candle_core::{Device, Tensor, DType, Error as CandleError}; // Added DType, Error

// Removed burn imports: Batcher, Dataset, Float, Int, Shape, TensorData, Backend
// These will be replaced by candle equivalents or different structures.

use itertools::Itertools;
use serde::{Deserialize, Serialize};

/// Stores a list of reviews for a card, in chronological order. Each FSRSItem corresponds
/// to a single review, but contains the previous reviews of the card as well, after the
/// first one.
/// When used during review, the last item should include the correct delta_t, but
/// the provided rating is ignored as all four ratings are returned by .next_states()
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq, Default)]
pub struct FSRSItem {
    pub reviews: Vec<FSRSReview>,
}

#[derive(Debug, Clone)]
pub(crate) struct WeightedFSRSItem {
    pub weight: f64,
    pub item: FSRSItem,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
pub struct FSRSReview {
    /// 1-4
    pub rating: u32,
    /// The number of days that passed
    /// # Warning
    /// `delta_t` for item first(initial) review must be 0
    pub delta_t: u32,
}

impl FSRSItem {
    // The previous reviews done before the current one.
    pub(crate) fn history(&self) -> impl Iterator<Item = &FSRSReview> {
        self.reviews.iter().take(self.reviews.len() - 1)
    }

    pub(crate) fn current(&self) -> &FSRSReview {
        self.reviews.last().unwrap()
    }

    pub fn long_term_review_cnt(&self) -> usize {
        self.reviews
            .iter()
            .filter(|review| review.delta_t > 0)
            .count()
    }

    pub(crate) fn first_long_term_review(&self) -> FSRSReview {
        *self
            .reviews
            .iter()
            .find(|review| review.delta_t > 0)
            .expect("Invalid FSRS item: at least one review with delta_t > 0 is required")
    }

    pub(crate) fn r_matrix_index(&self) -> (u32, u32, u32) {
        let delta_t = self.current().delta_t as f64;
        let delta_t_bin = (2.48 * 3.62f64.powf(delta_t.log(3.62).floor()) * 100.0).round() as u32;
        let length = self.long_term_review_cnt() as f64 + 1.0;
        let length_bin = (1.99 * 1.89f64.powf(length.log(1.89).floor())).round() as u32;
        let lapse = self
            .history()
            .filter(|review| review.rating == 1 && review.delta_t > 0)
            .count();
        if lapse == 0 {
            return (delta_t_bin, length_bin, 0);
        }
        let lapse_bin = (1.65 * 1.73f64.powf((lapse as f64).log(1.73).floor())).round() as u32;
        (delta_t_bin, length_bin, lapse_bin)
    }
}

#[derive(Clone)]
pub(crate) struct FSRSBatcher { // Removed <B: Backend> generic
    device: Device, // Uses candle_core::Device
}

impl FSRSBatcher {
    pub const fn new(device: Device) -> Self {
        Self { device }
    }
}

#[derive(Debug, Clone)]
pub(crate) struct FSRSBatch { // Removed <B: Backend> generic
    pub t_historys: Tensor, // candle::Tensor
    pub r_historys: Tensor, // candle::Tensor
    pub delta_ts: Tensor,   // candle::Tensor
    pub labels: Tensor,     // candle::Tensor (should be U32 or I64 for CrossEntropy, or f64 for BCE)
    pub weights: Tensor,    // candle::Tensor
}

// The Batcher trait from burn.data is removed. This is now a concrete struct method.
// The method signature changes to return Result<FSRSBatch, CandleError> for error handling.
impl FSRSBatcher {
    pub fn batch(&self, weighted_items: Vec<WeightedFSRSItem>) -> Result<FSRSBatch, CandleError> {
        let pad_size = weighted_items
            .iter()
            .map(|x| x.item.reviews.len())
            .max()
            .unwrap_or(0) // Handle empty weighted_items
            .saturating_sub(1); // Ensure pad_size is not negative if max len is 0 or 1.

        let mut time_histories_vec: Vec<Tensor> = Vec::new();
        let mut rating_histories_vec: Vec<Tensor> = Vec::new();

        for weighted_item in &weighted_items {
            let (mut delta_t_values, mut rating_values): (Vec<f64>, Vec<f64>) = weighted_item
                .item
                .history()
                .map(|r| (r.delta_t as f64, r.rating as f64))
                .unzip();

            delta_t_values.resize(pad_size, 0.0);
            rating_values.resize(pad_size, 0.0);

            // Create 1D tensors first, then reshape/unsqueeze for concat
            let delta_t_tensor = Tensor::from_vec(delta_t_values, (pad_size,), &self.device)?.unsqueeze(0)?; // Shape [1, pad_size]
            let rating_tensor = Tensor::from_vec(rating_values, (pad_size,), &self.device)?.unsqueeze(0)?; // Shape [1, pad_size]

            time_histories_vec.push(delta_t_tensor);
            rating_histories_vec.push(rating_tensor);
        }

        let (delta_ts_vec, labels_vec, weights_vec): (Vec<Tensor>, Vec<Tensor>, Vec<Tensor>) = weighted_items
            .iter()
            .map(|weighted_item| {
                let current = weighted_item.item.current();
                let delta_t = Tensor::from_slice(&[current.delta_t as f64], (1,), &self.device)?;
                let label_val = match current.rating {
                    1 => 0f64, // Using f64 for labels if BCE is used directly with probabilities
                    _ => 1f64,
                };
                let label = Tensor::from_slice(&[label_val], (1,), &self.device)?;
                let weight = Tensor::from_slice(&[weighted_item.weight], (1,), &self.device)?;
                Ok((delta_t, label, weight))
            })
            .collect::<Result<Vec<_>, CandleError>>()? // Handle potential errors from tensor creation
            .into_iter()
            .multiunzip();

        // Concatenate along batch dimension (dim 0)
        let t_historys = if !time_histories_vec.is_empty() {
            Tensor::cat(&time_histories_vec, 0)?.transpose(0, 1)? // [pad_size, batch_size]
        } else {
            Tensor::zeros((pad_size, weighted_items.len()), DType::F64, &self.device)?
        };

        let r_historys = if !rating_histories_vec.is_empty() {
            Tensor::cat(&rating_histories_vec, 0)?.transpose(0, 1)? // [pad_size, batch_size]
        } else {
            Tensor::zeros((pad_size, weighted_items.len()), DType::F64, &self.device)?
        };

        let delta_ts = if !delta_ts_vec.is_empty() { Tensor::cat(&delta_ts_vec, 0)? } else { Tensor::zeros((weighted_items.len(),), DType::F64, &self.device)? };
        let labels = if !labels_vec.is_empty() { Tensor::cat(&labels_vec, 0)? } else { Tensor::zeros((weighted_items.len(),), DType::F64, &self.device)? }; // Assuming F64 labels
        let weights = if !weights_vec.is_empty() { Tensor::cat(&weights_vec, 0)? } else { Tensor::zeros((weighted_items.len(),), DType::F64, &self.device)? };


        Ok(FSRSBatch { // Ok wrapping
            t_historys,
            r_historys,
            delta_ts,
            labels,
            weights,
        })
    }
}

// FSRSDataset no longer implements burn::data::dataset::Dataset
// It's now a simple struct that holds the data.
pub(crate) struct FSRSDataset {
    pub(crate) items: Vec<WeightedFSRSItem>,
}

// Methods for FSRSDataset (if needed, e.g. len, get)
impl FSRSDataset {
    pub fn len(&self) -> usize {
        self.items.len()
    }

    pub fn get(&self, index: usize) -> Option<&WeightedFSRSItem> { // Return ref
        self.items.get(index)
    }
}


impl From<Vec<WeightedFSRSItem>> for FSRSDataset {
    fn from(items: Vec<WeightedFSRSItem>) -> Self {
        Self {
            items: sort_items_by_review_length(items),
        }
    }
}

pub fn filter_outlier(
    pretrainset: Vec<FSRSItem>,
    mut trainset: Vec<FSRSItem>,
) -> (Vec<FSRSItem>, Vec<FSRSItem>) {
    let mut groups = HashMap::<u32, HashMap<u32, Vec<FSRSItem>>>::new();

    // group by rating of first review and delta_t of second review
    for item in pretrainset.into_iter() {
        let (first_review, second_review) = (item.reviews.first().unwrap(), item.current());
        let rating_group = groups.entry(first_review.rating).or_default();
        let delta_t_group = rating_group.entry(second_review.delta_t).or_default();
        delta_t_group.push(item);
    }

    let mut filtered_items = vec![];
    let mut removed_pairs: [HashSet<_>; 5] = Default::default();

    for (rating, delta_t_groups) in groups.into_iter().sorted_by_key(|&(k, _)| k) {
        let mut sub_groups = delta_t_groups.into_iter().collect::<Vec<_>>();

        // order by size of sub group ascending and delta_t descending
        sub_groups.sort_by(|(delta_t_a, subv_a), (delta_t_b, subv_b)| {
            subv_b
                .len()
                .cmp(&subv_a.len())
                .then(delta_t_b.cmp(delta_t_a))
        });

        let total = sub_groups.iter().map(|(_, vec)| vec.len()).sum::<usize>();
        let mut has_been_removed = 0;

        for (delta_t, sub_group) in sub_groups.iter().rev() {
            // remove 5% items (20 at least) of each group
            if has_been_removed + sub_group.len() >= 20.max(total / 20) {
                // keep the sub_group if it includes at least six items
                // and the delta_t is less than 100 days if rating is not 4
                // or less than 365 days if rating is 4
                if sub_group.len() >= 6 && *delta_t <= if rating != 4 { 100 } else { 365 } {
                    filtered_items.extend_from_slice(sub_group);
                } else {
                    removed_pairs[rating as usize].insert(*delta_t);
                }
            } else {
                has_been_removed += sub_group.len();
                removed_pairs[rating as usize].insert(*delta_t);
            }
        }
    }
    // keep the items in trainset if they are not removed from filtered_items
    trainset.retain(|item| {
        !removed_pairs[item.reviews[0].rating as usize]
            .contains(&item.first_long_term_review().delta_t)
    });
    (filtered_items, trainset)
}

pub fn prepare_training_data(items: Vec<FSRSItem>) -> (Vec<FSRSItem>, Vec<FSRSItem>) {
    let (mut pretrainset, mut trainset) = items
        .clone()
        .into_iter()
        .partition(|item| item.long_term_review_cnt() == 1);
    if std::env::var("FSRS_NO_OUTLIER").is_err() {
        (pretrainset, trainset) = filter_outlier(pretrainset, items);
    }
    (pretrainset, trainset)
}

pub(crate) fn sort_items_by_review_length(
    mut weighted_items: Vec<WeightedFSRSItem>,
) -> Vec<WeightedFSRSItem> {
    weighted_items.sort_by_cached_key(|weighted_item| weighted_item.item.reviews.len());
    weighted_items
}

pub(crate) fn constant_weighted_fsrs_items(items: Vec<FSRSItem>) -> Vec<WeightedFSRSItem> {
    items
        .into_iter()
        .map(|item| WeightedFSRSItem { weight: 1.0, item })
        .collect()
}

/// The input items should be sorted by the review timestamp.
pub(crate) fn recency_weighted_fsrs_items(items: Vec<FSRSItem>) -> Vec<WeightedFSRSItem> {
    let length = (items.len() as f64 - 1.0).max(1.0);
    items
        .into_iter()
        .enumerate()
        .map(|(idx, item)| WeightedFSRSItem {
            weight: 0.25 + 0.75 * (idx as f64 / length).powi(3),
            item,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    // use burn::tensor::Tolerance; // Removed burn import

    use super::*;
    use crate::convertor_tests::anki21_sample_file_converted_to_fsrs;
    use candle_core::Device; // Using candle Device

    // Helper for tests: compare two tensors (assuming f64 data)
    fn assert_tensor_eq(result: &Tensor, expected_data: &[f64], shape: &[usize]) -> Result<(), CandleError> {
        let expected = Tensor::from_slice(expected_data, shape, &Device::Cpu)?;
        let diff = (result - &expected)?.abs()?.sum_all()?.to_scalar::<f64>()?;
        assert!(diff < 1e-5, "Tensors are not equal. Result: {:?}, Expected: {:?}", result.to_vec2::<f64>(), expected.to_vec2::<f64>());
        Ok(())
    }


    #[test]
    fn from_anki() -> Result<(), CandleError> { // Return Result for candle ops
        // use burn::data::dataloader::Dataset; // Removed burn import

        let dataset = FSRSDataset::from(constant_weighted_fsrs_items(
            anki21_sample_file_converted_to_fsrs(),
        ));
        assert_eq!(
            dataset.get(704).unwrap().item, // .cloned() removed as get now returns ref
            FSRSItem {
                reviews: vec![
                    FSRSReview {
                        rating: 4,
                        delta_t: 0
                    },
                    FSRSReview {
                        rating: 3,
                        delta_t: 3
                    }
                ],
            }
        );

        // The DataLoader part of this test is removed as it depends on burn::data::DataLoaderBuilder
        // and a compatible Batcher trait. This needs to be re-done with candle's approach to data loading.
        // For now, we can test the batcher directly if needed.
        /*
        use burn::backend::ndarray::NdArrayDevice; // This would be Device::Cpu
        let device = NdArrayDevice::Cpu;
        use burn::backend::NdArray; // Not needed
        type Backend = NdArray<f64>; // Not needed
        let batcher = FSRSBatcher::<Backend>::new(device); // Now FSRSBatcher::new(device)
        use burn::data::dataloader::DataLoaderBuilder;
        let dataloader = DataLoaderBuilder::new(batcher)
            .batch_size(1)
            .shuffle(42)
            .num_workers(4)
            .build(dataset); // build(dataset) would need FSRSDataset to impl burn::Dataset
        dbg!(
            dataloader
                .iter()
                .next()
                .expect("loader is empty")
                .r_historys
        );
        */
        Ok(())
    }

    #[test]
    fn batcher() -> Result<(), CandleError> { // Return Result
        // use burn::backend::NdArray; // Removed
        // use burn::backend::ndarray::NdArrayDevice; // Removed
        // type Backend = NdArray<f64>; // Removed
        let device = Device::Cpu; // Use candle Device
        let batcher = FSRSBatcher::new(device.clone()); // Pass candle device
        let items_data = [ // Renamed to avoid conflict
            FSRSItem {
                reviews: [(4, 0), (3, 5)]
                    .into_iter()
                    .map(|(rating, delta_t)| FSRSReview { rating, delta_t })
                    .collect(),
            },
            FSRSItem {
                reviews: [(4, 0), (3, 5), (3, 11)]
                    .into_iter()
                    .map(|(rating, delta_t)| FSRSReview { rating, delta_t })
                    .collect(),
            },
            FSRSItem {
                reviews: [(4, 0), (3, 2)]
                    .into_iter()
                    .map(|(rating, delta_t)| FSRSReview { rating, delta_t })
                    .collect(),
            },
            FSRSItem {
                reviews: [(4, 0), (3, 2), (3, 6)]
                    .into_iter()
                    .map(|(rating, delta_t)| FSRSReview { rating, delta_t })
                    .collect(),
            },
            FSRSItem {
                reviews: [(4, 0), (3, 2), (3, 6), (3, 16)]
                    .into_iter()
                    .map(|(rating, delta_t)| FSRSReview { rating, delta_t })
                    .collect(),
            },
            FSRSItem {
                reviews: [(4, 0), (3, 2), (3, 6), (3, 16), (3, 39)]
                    .into_iter()
                    .map(|(rating, delta_t)| FSRSReview { rating, delta_t })
                    .collect(),
            },
            FSRSItem {
                reviews: [(1, 0), (1, 1)]
                    .into_iter()
                    .map(|(rating, delta_t)| FSRSReview { rating, delta_t })
                    .collect(),
            },
            FSRSItem {
                reviews: [(1, 0), (1, 1), (3, 1)]
                    .into_iter()
                    .map(|(rating, delta_t)| FSRSReview { rating, delta_t })
                    .collect(),
            },
        ];
        let weighted_items: Vec<WeightedFSRSItem> = items_data // Use new name
            .into_iter()
            .map(|item| WeightedFSRSItem { weight: 1.0, item })
            .collect();
        let batch = batcher.batch(weighted_items)?; // Use ?, remove device arg

        assert_tensor_eq(&batch.t_historys, &[
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 5.0, 0.0, 2.0, 2.0, 2.0, 0.0, 1.0,
            0.0, 0.0, 0.0, 0.0, 6.0, 6.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 16.0,0.0, 0.0,
        ], &[4, 8])?; // Shape is [pad_size, batch_size] -> [4, 8]

        assert_tensor_eq(&batch.r_historys, &[
            4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 1.0, 1.0,
            0.0, 3.0, 0.0, 3.0, 3.0, 3.0, 0.0, 1.0,
            0.0, 0.0, 0.0, 0.0, 3.0, 3.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0,
        ], &[4, 8])?;

        assert_tensor_eq(&batch.delta_ts, &[5.0, 11.0, 2.0, 6.0, 16.0, 39.0, 1.0, 1.0], &[8])?;

        // Labels are now f64: 0.0 for fail (rating 1), 1.0 for pass (rating > 1)
        assert_tensor_eq(&batch.labels, &[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0], &[8])?;
        Ok(())
    }

    #[test]
    fn test_filter_outlier() {
        let dataset = anki21_sample_file_converted_to_fsrs();
        let (mut pretrainset, mut trainset): (Vec<FSRSItem>, Vec<FSRSItem>) = dataset
            .into_iter()
            .partition(|item| item.long_term_review_cnt() == 1);
        assert_eq!(pretrainset.len(), 3315);
        assert_eq!(trainset.len(), 10975);
        (pretrainset, trainset) = filter_outlier(pretrainset, trainset);
        assert_eq!(pretrainset.len(), 3265);
        assert_eq!(trainset.len(), 10900);
    }
}

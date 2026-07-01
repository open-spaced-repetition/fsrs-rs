//! The dataset module provides data structures and data processing functions for working with FSRS datasets.

use std::collections::{HashMap, HashSet};

#[cfg(test)]
use burn::data::dataloader::batcher::Batcher;
#[cfg(test)]
use burn::data::dataset::Dataset;
#[cfg(test)]
use burn::tensor::{Float, Int, Tensor, TensorData, backend::Backend};

use itertools::Itertools;
use serde::{Deserialize, Serialize};

/// Stores a list of reviews for a card, in chronological order.
///
/// Each [`FSRSItem`] corresponds to a single review, but contains the previous reviews of the card as well, after the first one.
/// When used during review, the last item should include the correct delta_t, but the provided rating is ignored as all four ratings are returned by `.next_states()`.
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq, Default)]
pub struct FSRSItem {
    pub reviews: Vec<FSRSReview>,
}

#[derive(Debug, Clone)]
pub(crate) struct WeightedFSRSItem {
    pub weight: f32,
    pub card_id: i64,
    pub item: FSRSItem,
}

/// A single review for a card, including the user's rating and the number of days that passed.
///
/// This struct is a part of [`FSRSItem`].
#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
pub struct FSRSReview {
    /// The user's performance rating for this review.
    ///
    /// Rating scale:
    /// - 1 = Again (forgot anything you want to remember)
    /// - 2 = Hard (remembered with difficulty)
    /// - 3 = Good (remembered correctly)
    /// - 4 = Easy (remembered effortlessly)
    ///
    /// # Note
    /// This field is **ignored** for the most recent item when calling
    /// `.next_states()`. The method returns all four possible next states
    /// regardless of the rating stored here.
    pub rating: u32,
    /// The number of days that passed
    ///
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

#[cfg(test)]
#[derive(Clone)]
pub(crate) struct FSRSBatcher<B: Backend> {
    _backend: core::marker::PhantomData<B>,
}

#[cfg(test)]
impl<B: Backend> FSRSBatcher<B> {
    pub const fn new() -> Self {
        Self {
            _backend: core::marker::PhantomData,
        }
    }
}

#[cfg(test)]
#[derive(Debug, Clone)]
pub(crate) struct FSRSBatch<B: Backend> {
    pub t_historys: Tensor<B, 2, Float>,
    pub r_historys: Tensor<B, 2, Float>,
    pub delta_ts: Tensor<B, 1, Float>,
    pub labels: Tensor<B, 1, Int>,
    pub weights: Tensor<B, 1, Float>,
}

#[cfg(test)]
impl<B: Backend> Batcher<B, WeightedFSRSItem, FSRSBatch<B>> for FSRSBatcher<B> {
    fn batch(&self, weighted_items: Vec<WeightedFSRSItem>, device: &B::Device) -> FSRSBatch<B> {
        let batch_size = weighted_items.len();
        let pad_size = weighted_items
            .iter()
            .map(|x| x.item.reviews.len())
            .max()
            .expect("FSRSItem is empty")
            - 1;

        let mut time_histories = vec![0.0; pad_size * batch_size];
        let mut rating_histories = vec![0.0; pad_size * batch_size];
        let mut delta_ts = Vec::with_capacity(batch_size);
        let mut labels = Vec::with_capacity(batch_size);
        let mut weights = Vec::with_capacity(batch_size);

        for (batch_index, weighted_item) in weighted_items.iter().enumerate() {
            for (history_index, review) in weighted_item.item.history().enumerate() {
                let index = history_index * batch_size + batch_index;
                time_histories[index] = review.delta_t as f32;
                rating_histories[index] = review.rating as f32;
            }

            let current = weighted_item.item.current();
            delta_ts.push(current.delta_t as f32);
            labels.push(if current.rating == 1 { 0 } else { 1 });
            weights.push(weighted_item.weight);
        }

        let t_historys = Tensor::from_floats(
            TensorData::new(time_histories, [pad_size, batch_size]),
            device,
        ); // [seq_len, batch_size]
        let r_historys = Tensor::from_floats(
            TensorData::new(rating_histories, [pad_size, batch_size]),
            device,
        ); // [seq_len, batch_size]
        let delta_ts = Tensor::from_floats(TensorData::new(delta_ts, [batch_size]), device);
        let labels = Tensor::from_ints(TensorData::new(labels, [batch_size]), device);
        let weights = Tensor::from_floats(TensorData::new(weights, [batch_size]), device);

        // dbg!(&items[0].t_history);
        // dbg!(&t_historys);

        FSRSBatch {
            t_historys,
            r_historys,
            delta_ts,
            labels,
            weights,
        }
    }
}

#[cfg(test)]
pub(crate) struct FSRSDataset {
    pub(crate) items: Vec<WeightedFSRSItem>,
}

#[cfg(test)]
impl Dataset<WeightedFSRSItem> for FSRSDataset {
    fn len(&self) -> usize {
        self.items.len()
    }

    fn get(&self, index: usize) -> Option<WeightedFSRSItem> {
        // info!("get {}", index);
        self.items.get(index).cloned()
    }
}

#[cfg(test)]
impl From<Vec<WeightedFSRSItem>> for FSRSDataset {
    fn from(items: Vec<WeightedFSRSItem>) -> Self {
        Self {
            items: sort_items_by_review_length(items),
        }
    }
}

fn compute_outlier_analysis(items: &[FSRSItem]) -> ([HashSet<u32>; 5], Vec<usize>) {
    let mut groups = HashMap::<u32, HashMap<u32, Vec<usize>>>::new();

    // group by rating of first review and delta_t of second review
    for (index, item) in items
        .iter()
        .enumerate()
        .filter(|(_, item)| item.long_term_review_cnt() == 1)
    {
        let (first_review, second_review) = (item.reviews.first().unwrap(), item.current());
        let rating_group = groups.entry(first_review.rating).or_default();
        rating_group
            .entry(second_review.delta_t)
            .or_default()
            .push(index);
    }

    let mut removed_pairs: [HashSet<_>; 5] = Default::default();
    let mut kept_initialization_indices = Vec::new();

    for (rating, delta_t_groups) in groups.into_iter().sorted_by_key(|&(k, _)| k) {
        let mut sub_groups = delta_t_groups.into_iter().collect::<Vec<_>>();

        // order by size of sub group ascending and delta_t descending
        sub_groups.sort_by(|(delta_t_a, indices_a), (delta_t_b, indices_b)| {
            indices_b
                .len()
                .cmp(&indices_a.len())
                .then(delta_t_b.cmp(delta_t_a))
        });

        let total = sub_groups
            .iter()
            .map(|(_, indices)| indices.len())
            .sum::<usize>();
        let mut has_been_removed = 0;

        for (delta_t, indices) in sub_groups.iter().rev() {
            // remove 5% items (20 at least) of each group
            if has_been_removed + indices.len() >= 20.max(total / 20) {
                // keep the sub_group if it includes at least six items
                // and the delta_t is less than 100 days if rating is not 4
                // or less than 365 days if rating is 4
                if indices.len() >= 6 && *delta_t <= if rating != 4 { 100 } else { 365 } {
                    kept_initialization_indices.extend_from_slice(indices);
                } else {
                    removed_pairs[rating as usize].insert(*delta_t);
                }
            } else {
                has_been_removed += indices.len();
                removed_pairs[rating as usize].insert(*delta_t);
            }
        }
    }
    (removed_pairs, kept_initialization_indices)
}

fn train_item_survives_outlier(item: &FSRSItem, removed_pairs: &[HashSet<u32>; 5]) -> bool {
    !removed_pairs[item.reviews[0].rating as usize].contains(&item.first_long_term_review().delta_t)
}

/// Filters out outlier reviews from two [`Vec<FSRSItem>`].
///
/// This function removes anomalous review records that could negatively impact model training.
/// It uses `dataset_for_initialization` as a reference to choose normal review patterns.
///
/// # Arguments
/// * `dataset_for_initialization` - A [`Vec<FSRSItem>`] used as the reference standard.
///   Each item **must** contain at least 2 reviews.
///   The data should be high-quality and free from obvious anomalies.
/// * `trainset` - A [`Vec<FSRSItem>`] which contains the training data to be filtered.
///
/// # Returns
/// * `filtered_init` - A `Vec<FSRSItem>` filtered from `dataset_for_initialization`.
/// * `filtered_trainset` - A `Vec<FSRSItem>` filtered from input `trainset`.
///   You can use it for model training.
///
/// # Panics
/// This function will panic if any item in `dataset_for_initialization` contains fewer than 2 reviews.
///
/// # Example
/// ```
/// use fsrs::{FSRSItem, filter_outlier};
///
/// let dataset_for_initialization = vec![/* ... */];
/// let trainset = vec![/* ... */];
/// let (filtered_init, filtered_trainset) = filter_outlier(dataset_for_initialization, trainset);
/// ```
///
/// # Notes
/// Both input `Vec`s will move to this function.
/// The filtered versions are returned as outputs.
/// If you need to keep the data, use `.clone()`.
pub fn filter_outlier(
    dataset_for_initialization: Vec<FSRSItem>,
    mut trainset: Vec<FSRSItem>,
) -> (Vec<FSRSItem>, Vec<FSRSItem>) {
    let mut groups = HashMap::<u32, HashMap<u32, Vec<FSRSItem>>>::new();

    // group by rating of first review and delta_t of second review
    for item in dataset_for_initialization.into_iter() {
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
    trainset.retain(|item| train_item_survives_outlier(item, &removed_pairs));
    (filtered_items, trainset)
}

pub(crate) fn prepare_training_data(items: Vec<FSRSItem>) -> (Vec<FSRSItem>, Vec<FSRSItem>) {
    if std::env::var("FSRS_NO_OUTLIER").is_ok() {
        return items
            .into_iter()
            .partition(|item| item.long_term_review_cnt() == 1);
    }

    let (removed_pairs, kept_initialization_indices) = compute_outlier_analysis(&items);
    let dataset_for_initialization = kept_initialization_indices
        .into_iter()
        .map(|index| items[index].clone())
        .collect();
    let mut trainset = Vec::with_capacity(items.len());
    for item in items {
        if train_item_survives_outlier(&item, &removed_pairs) {
            trainset.push(item);
        }
    }
    (dataset_for_initialization, trainset)
}

pub(crate) fn prepare_training_data_with_card_ids(
    items: Vec<FSRSItem>,
    card_ids: Vec<i64>,
) -> (Vec<FSRSItem>, Vec<FSRSItem>, Vec<i64>) {
    if std::env::var("FSRS_NO_OUTLIER").is_ok() {
        let mut dataset_for_initialization = Vec::new();
        let mut trainset = Vec::new();
        let mut trainset_card_ids = Vec::new();
        for (item, card_id) in items.into_iter().zip(card_ids) {
            if item.long_term_review_cnt() == 1 {
                dataset_for_initialization.push(item);
            } else {
                trainset.push(item);
                trainset_card_ids.push(card_id);
            }
        }
        return (dataset_for_initialization, trainset, trainset_card_ids);
    }

    let (removed_pairs, kept_initialization_indices) = compute_outlier_analysis(&items);
    let dataset_for_initialization = kept_initialization_indices
        .into_iter()
        .map(|index| items[index].clone())
        .collect();
    let mut trainset = Vec::with_capacity(items.len());
    let mut trainset_card_ids = Vec::with_capacity(card_ids.len());
    for (item, card_id) in items.into_iter().zip(card_ids) {
        if train_item_survives_outlier(&item, &removed_pairs) {
            trainset.push(item);
            trainset_card_ids.push(card_id);
        }
    }
    (dataset_for_initialization, trainset, trainset_card_ids)
}

#[cfg(test)]
pub(crate) fn sort_items_by_review_length(
    mut weighted_items: Vec<WeightedFSRSItem>,
) -> Vec<WeightedFSRSItem> {
    weighted_items.sort_by_cached_key(|weighted_item| weighted_item.item.reviews.len());
    weighted_items
}

pub(crate) fn constant_weighted_fsrs_items(items: Vec<FSRSItem>) -> Vec<WeightedFSRSItem> {
    items
        .into_iter()
        .map(|item| WeightedFSRSItem {
            weight: 1.0,
            card_id: -1,
            item,
        })
        .collect()
}

/// The input items should be sorted by the review timestamp.
pub(crate) fn recency_weighted_fsrs_items(items: Vec<FSRSItem>) -> Vec<WeightedFSRSItem> {
    let length = (items.len() as f32 - 1.0).max(1.0);
    items
        .into_iter()
        .enumerate()
        .map(|(idx, item)| WeightedFSRSItem {
            weight: 0.25 + 0.75 * (idx as f32 / length).powi(3),
            card_id: -1,
            item,
        })
        .collect()
}

pub(crate) fn recency_weighted_fsrs_items_with_card_ids(
    items: Vec<FSRSItem>,
    card_ids: Vec<i64>,
) -> Vec<WeightedFSRSItem> {
    let length = (items.len() as f32 - 1.0).max(1.0);
    items
        .into_iter()
        .zip(card_ids)
        .enumerate()
        .map(|(idx, (item, card_id))| WeightedFSRSItem {
            weight: 0.25 + 0.75 * (idx as f32 / length).powi(3),
            card_id,
            item,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use burn::backend::NdArray;
    use burn::backend::ndarray::NdArrayDevice;
    use burn::tensor::Tolerance;
    type Backend = NdArray<f32>;
    static DEVICE: NdArrayDevice = NdArrayDevice::Cpu;

    use super::*;
    use crate::convertor_tests::anki21_sample_file_converted_to_fsrs;

    #[test]
    fn test_from_anki() {
        use burn::data::dataloader::Dataset;

        let dataset = FSRSDataset::from(constant_weighted_fsrs_items(
            anki21_sample_file_converted_to_fsrs(),
        ));
        assert_eq!(
            dataset.get(704).unwrap().item,
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

        let batcher = FSRSBatcher::<Backend>::new();
        use burn::data::dataloader::DataLoaderBuilder;
        let dataloader = DataLoaderBuilder::new(batcher)
            .batch_size(1)
            .shuffle(42)
            .num_workers(4)
            .build(dataset);
        dbg!(
            dataloader
                .iter()
                .next()
                .expect("loader is empty")
                .r_historys
        );
    }

    #[test]
    fn test_batcher() {
        let batcher = FSRSBatcher::<Backend>::new();
        let items = [
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
        let items = items
            .into_iter()
            .map(|item| WeightedFSRSItem {
                weight: 1.0,
                card_id: -1,
                item,
            })
            .collect();
        let batch = batcher.batch(items, &DEVICE);
        batch.t_historys.to_data().assert_approx_eq::<f32>(
            &TensorData::from([
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 5.0, 0.0, 2.0, 2.0, 2.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 0.0, 6.0, 6.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 16.0, 0.0, 0.0],
            ]),
            Tolerance::absolute(1e-5),
        );
        batch.r_historys.to_data().assert_approx_eq::<f32>(
            &TensorData::from([
                [4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 1.0, 1.0],
                [0.0, 3.0, 0.0, 3.0, 3.0, 3.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 0.0, 3.0, 3.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0],
            ]),
            Tolerance::absolute(1e-5),
        );

        batch.delta_ts.to_data().assert_approx_eq::<f32>(
            &TensorData::from([5.0, 11.0, 2.0, 6.0, 16.0, 39.0, 1.0, 1.0]),
            Tolerance::absolute(1e-5),
        );
        batch.labels.to_data().assert_approx_eq::<f32>(
            &TensorData::from([1, 1, 1, 1, 1, 1, 0, 1]),
            Tolerance::absolute(1e-5),
        );
    }

    #[test]
    fn test_filter_outlier() {
        let dataset = anki21_sample_file_converted_to_fsrs();
        let (mut dataset_for_initialization, mut trainset): (Vec<FSRSItem>, Vec<FSRSItem>) =
            dataset
                .into_iter()
                .partition(|item| item.long_term_review_cnt() == 1);
        assert_eq!(dataset_for_initialization.len(), 3315);
        assert_eq!(trainset.len(), 10975);
        (dataset_for_initialization, trainset) =
            filter_outlier(dataset_for_initialization, trainset);
        assert_eq!(dataset_for_initialization.len(), 3265);
        assert_eq!(trainset.len(), 10900);
    }
}

use std::collections::{HashMap, HashSet};

use burn::data::dataloader::batcher::Batcher;
use burn::{
    data::dataset::Dataset,
    tensor::{Float, Int, Shape, Tensor, TensorData, backend::Backend},
};

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
    pub weight: f32,
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
pub(crate) struct FSRSBatcher<B: Backend> {
    device: B::Device,
}

impl<B: Backend> FSRSBatcher<B> {
    pub const fn new(device: B::Device) -> Self {
        Self { device }
    }
}

#[derive(Debug, Clone)]
pub(crate) struct FSRSBatch<B: Backend> {
    pub t_historys: Tensor<B, 2, Float>,
    pub r_historys: Tensor<B, 2, Float>,
    pub delta_ts: Tensor<B, 1, Float>,
    pub labels: Tensor<B, 1, Int>,
    pub weights: Tensor<B, 1, Float>,
}

impl<B: Backend> Batcher<B, WeightedFSRSItem, FSRSBatch<B>> for FSRSBatcher<B> {
    fn batch(&self, weighted_items: Vec<WeightedFSRSItem>, _device: &B::Device) -> FSRSBatch<B> {
        let pad_size = weighted_items
            .iter()
            .map(|x| x.item.reviews.len())
            .max()
            .expect("FSRSItem is empty")
            - 1;

        let (time_histories, rating_histories) = weighted_items
            .iter()
            .map(|weighted_item| {
                let (mut delta_t, mut rating): (Vec<_>, Vec<_>) = weighted_item
                    .item
                    .history()
                    .map(|r| (r.delta_t, r.rating))
                    .unzip();
                delta_t.resize(pad_size, 0);
                rating.resize(pad_size, 0);
                let delta_t = Tensor::<B, 2>::from_floats(
                    TensorData::new(
                        delta_t,
                        Shape {
                            dims: vec![1, pad_size],
                        },
                    ),
                    &self.device,
                );
                let rating = Tensor::<B, 2>::from_data(
                    TensorData::new(
                        rating,
                        Shape {
                            dims: vec![1, pad_size],
                        },
                    ),
                    &self.device,
                );
                (delta_t, rating)
            })
            .unzip();

        let (delta_ts, labels, weights) = weighted_items
            .iter()
            .map(|weighted_item| {
                let current = weighted_item.item.current();
                let delta_t: Tensor<B, 1> =
                    Tensor::from_floats([current.delta_t as f32], &self.device);
                let label = match current.rating {
                    1 => 0,
                    _ => 1,
                };
                let label: Tensor<B, 1, Int> = Tensor::from_ints([label], &self.device);
                let weight: Tensor<B, 1> =
                    Tensor::from_floats([weighted_item.weight], &self.device);
                (delta_t, label, weight)
            })
            .multiunzip();

        let t_historys = Tensor::cat(time_histories, 0)
            .transpose()
            .to_device(&self.device); // [seq_len, batch_size]
        let r_historys = Tensor::cat(rating_histories, 0)
            .transpose()
            .to_device(&self.device); // [seq_len, batch_size]
        let delta_ts = Tensor::cat(delta_ts, 0).to_device(&self.device);
        let labels = Tensor::cat(labels, 0).to_device(&self.device);
        let weights = Tensor::cat(weights, 0).to_device(&self.device);

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

pub(crate) struct FSRSDataset {
    pub(crate) items: Vec<WeightedFSRSItem>,
}

impl Dataset<WeightedFSRSItem> for FSRSDataset {
    fn len(&self) -> usize {
        self.items.len()
    }

    fn get(&self, index: usize) -> Option<WeightedFSRSItem> {
        // info!("get {}", index);
        self.items.get(index).cloned()
    }
}

impl From<Vec<WeightedFSRSItem>> for FSRSDataset {
    fn from(items: Vec<WeightedFSRSItem>) -> Self {
        Self {
            items: sort_items_by_review_length(items),
        }
    }
}

pub(crate) fn filter_outlier(
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
    trainset.retain(|item| {
        !removed_pairs[item.reviews[0].rating as usize]
            .contains(&item.first_long_term_review().delta_t)
    });
    (filtered_items, trainset)
}

pub(crate) fn prepare_training_data(items: Vec<FSRSItem>) -> (Vec<FSRSItem>, Vec<FSRSItem>) {
    let (mut dataset_for_initialization, mut trainset) = items
        .clone()
        .into_iter()
        .partition(|item| item.long_term_review_cnt() == 1);
    if std::env::var("FSRS_NO_OUTLIER").is_err() {
        (dataset_for_initialization, trainset) = filter_outlier(dataset_for_initialization, items);
    }
    (dataset_for_initialization, trainset)
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
    let length = (items.len() as f32 - 1.0).max(1.0);
    items
        .into_iter()
        .enumerate()
        .map(|(idx, item)| WeightedFSRSItem {
            weight: 0.25 + 0.75 * (idx as f32 / length).powi(3),
            item,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use burn::tensor::Tolerance;
    use burn::backend::ndarray::NdArrayDevice;
    use burn::backend::NdArray;
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

        let batcher = FSRSBatcher::<Backend>::new(DEVICE);
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
        let batcher = FSRSBatcher::<Backend>::new(DEVICE);
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
            .map(|item| WeightedFSRSItem { weight: 1.0, item })
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

use std::collections::{HashMap, HashSet};

use burn::data::dataloader::batcher::Batcher;
use burn::{
    data::dataset::Dataset,
    tensor::{backend::Backend, Data, ElementConversion, Float, Int, Shape, Tensor},
};
use serde::{Deserialize, Serialize};

/// Stores a list of reviews for a card, in chronological order. Each FSRSItem corresponds
/// to a single review, but contains the previous reviews of the card as well, after the
/// first one.
/// When used during review, the last item should include the correct delta_t, but
/// the provided rating is ignored as all four ratings are returned by .next_states()
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
pub struct FSRSItem {
    pub reviews: Vec<FSRSReview>,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
pub struct FSRSReview {
    /// 1-4
    pub rating: u32,
    /// The number of days that passed
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
}

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
}

impl<B: Backend> Batcher<FSRSItem, FSRSBatch<B>> for FSRSBatcher<B> {
    fn batch(&self, items: Vec<FSRSItem>) -> FSRSBatch<B> {
        let pad_size = items
            .iter()
            .map(|x| x.reviews.len())
            .max()
            .expect("FSRSItem is empty")
            - 1;

        let (time_histories, rating_histories) = items
            .iter()
            .map(|item| {
                let (mut delta_t, mut rating): (Vec<_>, Vec<_>) =
                    item.history().map(|r| (r.delta_t, r.rating)).unzip();
                delta_t.resize(pad_size, 0);
                rating.resize(pad_size, 0);
                let delta_t =
                    Tensor::from_data(Data::new(delta_t, Shape { dims: [pad_size] }).convert())
                        .unsqueeze();
                let rating =
                    Tensor::from_data(Data::new(rating, Shape { dims: [pad_size] }).convert())
                        .unsqueeze();
                (delta_t, rating)
            })
            .unzip();

        let (delta_ts, labels) = items
            .iter()
            .map(|item| {
                let current = item.current();
                let delta_t = Tensor::from_data(Data::from([current.delta_t.elem()]));
                let label = match current.rating {
                    1 => 0.0,
                    _ => 1.0,
                };
                let label = Tensor::from_data(Data::from([label.elem()]));
                (delta_t, label)
            })
            .unzip();

        let t_historys = Tensor::cat(time_histories, 0)
            .transpose()
            .to_device(&self.device); // [seq_len, batch_size]
        let r_historys = Tensor::cat(rating_histories, 0)
            .transpose()
            .to_device(&self.device); // [seq_len, batch_size]
        let delta_ts = Tensor::cat(delta_ts, 0).to_device(&self.device);
        let labels = Tensor::cat(labels, 0).to_device(&self.device);

        // dbg!(&items[0].t_history);
        // dbg!(&t_historys);

        FSRSBatch {
            t_historys,
            r_historys,
            delta_ts,
            labels,
        }
    }
}

pub(crate) struct FSRSDataset {
    items: Vec<FSRSItem>,
}

impl Dataset<FSRSItem> for FSRSDataset {
    fn len(&self) -> usize {
        self.items.len()
    }

    fn get(&self, index: usize) -> Option<FSRSItem> {
        self.items.get(index).cloned()
    }
}

impl From<Vec<FSRSItem>> for FSRSDataset {
    fn from(items: Vec<FSRSItem>) -> Self {
        Self { items }
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

    for (rating, delta_t_groups) in groups.into_iter() {
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
                if sub_group.len() >= 6 {
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
        !removed_pairs[item.reviews[0].rating as usize].contains(&item.reviews[1].delta_t)
    });
    (filtered_items, trainset)
}

fn stratified_kfold(mut trainset: Vec<FSRSItem>, n_splits: usize) -> Vec<Vec<FSRSItem>> {
    trainset.sort_by(|a, b| a.reviews.len().cmp(&b.reviews.len()));
    (0..n_splits)
        .cycle() // cycle to evenly distribute
        .zip(trainset)
        .fold(vec![vec![]; n_splits], |mut acc, (i, item)| {
            acc[i].push(item);
            acc
        })
}

pub fn split_data(
    items: Vec<FSRSItem>,
    n_splits: usize,
) -> (Vec<FSRSItem>, Vec<Vec<FSRSItem>>, Vec<FSRSItem>) {
    let (mut pretrainset, mut trainset) =
        items.into_iter().partition(|item| item.reviews.len() == 2);
    if std::env::var("FSRS_NO_OUTLIER").is_err() {
        (pretrainset, trainset) = filter_outlier(pretrainset, trainset);
    }
    (
        pretrainset,
        stratified_kfold(trainset.clone(), n_splits),
        trainset,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::convertor_tests::anki21_sample_file_converted_to_fsrs;

    #[test]
    fn from_anki() {
        use burn::data::dataloader::Dataset;

        let dataset = FSRSDataset::from(anki21_sample_file_converted_to_fsrs());
        assert_eq!(
            dataset.get(704).unwrap(),
            FSRSItem {
                reviews: vec![
                    FSRSReview {
                        rating: 1,
                        delta_t: 0,
                    },
                    FSRSReview {
                        rating: 4,
                        delta_t: 2,
                    },
                ],
            }
        );

        use burn::backend::ndarray::NdArrayDevice;
        let device = NdArrayDevice::Cpu;
        use burn::backend::NdArray;
        type Backend = NdArray<f32>;
        let batcher = FSRSBatcher::<Backend>::new(device);
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
    fn batcher() {
        use burn::backend::ndarray::NdArrayDevice;
        use burn::backend::NdArray;
        type Backend = NdArray<f32>;
        let device = NdArrayDevice::Cpu;
        let batcher = FSRSBatcher::<Backend>::new(device);
        let items = vec![
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
                    },
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
                        delta_t: 2,
                    },
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
                        delta_t: 2,
                    },
                    FSRSReview {
                        rating: 3,
                        delta_t: 6,
                    },
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
                        delta_t: 2,
                    },
                    FSRSReview {
                        rating: 3,
                        delta_t: 6,
                    },
                    FSRSReview {
                        rating: 3,
                        delta_t: 16,
                    },
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
                        delta_t: 2,
                    },
                    FSRSReview {
                        rating: 3,
                        delta_t: 6,
                    },
                    FSRSReview {
                        rating: 3,
                        delta_t: 16,
                    },
                    FSRSReview {
                        rating: 3,
                        delta_t: 39,
                    },
                ],
            },
            FSRSItem {
                reviews: vec![
                    FSRSReview {
                        rating: 1,
                        delta_t: 0,
                    },
                    FSRSReview {
                        rating: 1,
                        delta_t: 1,
                    },
                ],
            },
            FSRSItem {
                reviews: vec![
                    FSRSReview {
                        rating: 1,
                        delta_t: 0,
                    },
                    FSRSReview {
                        rating: 1,
                        delta_t: 1,
                    },
                    FSRSReview {
                        rating: 3,
                        delta_t: 1,
                    },
                ],
            },
        ];
        let batch = batcher.batch(items);
        assert_eq!(
            batch.t_historys.to_data(),
            Data::from([
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 5.0, 0.0, 2.0, 2.0, 2.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 0.0, 6.0, 6.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 16.0, 0.0, 0.0]
            ])
        );
        assert_eq!(
            batch.r_historys.to_data(),
            Data::from([
                [4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 1.0, 1.0],
                [0.0, 3.0, 0.0, 3.0, 3.0, 3.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 0.0, 3.0, 3.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0]
            ])
        );
        assert_eq!(
            batch.delta_ts.to_data(),
            Data::from([5.0, 11.0, 2.0, 6.0, 16.0, 39.0, 1.0, 1.0])
        );
        assert_eq!(batch.labels.to_data(), Data::from([1, 1, 1, 1, 1, 1, 0, 1]));
    }

    #[test]
    fn test_filter_outlier() {
        let dataset = anki21_sample_file_converted_to_fsrs();
        let (mut pretrainset, mut trainset): (Vec<FSRSItem>, Vec<FSRSItem>) = dataset
            .into_iter()
            .partition(|item| item.reviews.len() == 2);
        assert_eq!(pretrainset.len(), 3315);
        assert_eq!(trainset.len(), 10806);
        (pretrainset, trainset) = filter_outlier(pretrainset, trainset);
        assert_eq!(pretrainset.len(), 3265);
        assert_eq!(trainset.len(), 10731);
    }
}

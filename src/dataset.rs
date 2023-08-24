use burn::data::dataloader::batcher::Batcher;
use burn::{
    data::dataset::{Dataset, InMemDataset},
    tensor::{backend::Backend, Data, ElementConversion, Float, Int, Shape, Tensor},
};
use serde::{Deserialize, Serialize};

use crate::convertor::anki_to_fsrs;

/// Stores a list of reviews for a card, in chronological order. Each FSRSItem corresponds
/// to a single review, but contains the previous reviews of the card as well, after the
/// first one.
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub struct FSRSItem {
    pub reviews: Vec<FSRSReview>,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub struct FSRSReview {
    /// 1-4
    pub rating: i32,
    /// The number of days that passed
    pub delta_t: i32,
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

pub struct FSRSBatcher<B: Backend> {
    device: B::Device,
}

impl<B: Backend> FSRSBatcher<B> {
    pub fn new(device: B::Device) -> Self {
        Self { device }
    }
}

#[derive(Debug, Clone)]
pub struct FSRSBatch<B: Backend> {
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
                let (mut delta_t, mut rating): (Vec<i32>, Vec<i32>) =
                    item.history().map(|r| (r.delta_t, r.rating)).unzip();
                delta_t.resize(pad_size, 0);
                rating.resize(pad_size, 0);
                let delta_t = Tensor::<B, 1>::from_data(
                    Data::new(delta_t, Shape { dims: [pad_size] }).convert(),
                )
                .unsqueeze();
                let rating = Tensor::<B, 1>::from_data(
                    Data::new(rating, Shape { dims: [pad_size] }).convert(),
                )
                .unsqueeze();
                (delta_t, rating)
            })
            .unzip();

        let (delta_ts, labels) = items
            .iter()
            .map(|item| {
                let current = item.current();
                let delta_t =
                    Tensor::<B, 1, Float>::from_data(Data::from([current.delta_t.elem()]));
                let label = match current.rating {
                    1 => 0.0,
                    _ => 1.0,
                };
                let label = Tensor::<B, 1, Int>::from_data(Data::from([label.elem()]));
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

pub struct FSRSDataset {
    dataset: InMemDataset<FSRSItem>,
}

impl Dataset<FSRSItem> for FSRSDataset {
    fn len(&self) -> usize {
        self.dataset.len()
    }

    fn get(&self, index: usize) -> Option<FSRSItem> {
        self.dataset.get(index)
    }
}

impl FSRSDataset {
    pub fn train() -> Self {
        Self::new()
    }

    pub fn test() -> Self {
        Self::new()
    }

    fn new() -> Self {
        let dataset = InMemDataset::<FSRSItem>::new(anki_to_fsrs());
        Self { dataset }
    }
}

#[test]
fn test_from_anki() {
    use burn::data::dataloader::Dataset;
    use burn::data::dataset::InMemDataset;

    let dataset = InMemDataset::<FSRSItem>::new(anki_to_fsrs());
    dbg!(dataset.get(704).unwrap());

    use burn_ndarray::NdArrayDevice;
    let device = NdArrayDevice::Cpu;
    use burn_ndarray::NdArrayBackend;
    type Backend = NdArrayBackend<f32>;
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
fn test_batcher() {
    use burn_ndarray::NdArrayBackend;
    use burn_ndarray::NdArrayDevice;
    type Backend = NdArrayBackend<f32>;
    let device = NdArrayDevice::Cpu;
    let batcher: FSRSBatcher<Backend> = FSRSBatcher::<Backend>::new(device);
    let dataset = FSRSDataset::train();
    let mut items = vec![];
    for item in dataset.iter() {
        items.push(item);
        if items.len() >= 8 {
            break;
        }
    }
    dbg!(&items);
    let batch = batcher.batch(items);
    dbg!(&batch);
}

use burn::{tensor::{backend::Backend, Data, Tensor, Float, ElementConversion, Shape, Int}, data::dataset::{InMemDataset, Dataset}};
use serde::{Deserialize, Serialize};
use burn::data::dataloader::batcher::Batcher;

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct FSRSItem {
    pub t_history: Vec<i32>,
    pub r_history: Vec<i32>,
    pub delta_t: f32,
    pub label: f32,
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

        let t_historys = items
            .iter()
            .map(|item| Data::new(item.t_history.clone(), Shape { dims: [item.t_history.len()] }))
            .map(|data| Tensor::<B, 1>::from_data(data.convert()))
            .map(|tensor| tensor.unsqueeze())
            .collect();

        let r_historys = items
            .iter()
            .map(|item| Data::new(item.r_history.clone(), Shape { dims: [item.t_history.len()] }))
            .map(|data| Tensor::<B, 1>::from_data(data.convert()))
            .map(|tensor| tensor.unsqueeze())
            .collect();

        let delta_ts = items
            .iter()
            .map(|item| Tensor::<B, 1, Float>::from_data(Data::from([item.delta_t.elem()])))
            .collect();

        let labels = items
            .iter()
            .map(|item| Tensor::<B, 1, Int>::from_data(Data::from([item.label.elem()])))
            .collect();

        let t_historys = Tensor::cat(t_historys, 0).transpose().to_device(&self.device); // [seq_len, batch_size]
        let r_historys = Tensor::cat(r_historys, 0).transpose().to_device(&self.device); // [seq_len, batch_size]
        let delta_ts = Tensor::cat(delta_ts, 0).to_device(&self.device);
        let labels = Tensor::cat(labels, 0).to_device(&self.device);

        // dbg!(&items[0].t_history);
        // dbg!(&t_historys);

        FSRSBatch { t_historys, r_historys, delta_ts, labels }
    }
}

pub struct FSRSDataset {
    dataset: InMemDataset<FSRSItem>
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
        let dataset = InMemDataset::<FSRSItem>::from_json_rows("tests/data/revlog_history.json").unwrap();
        Self { dataset }
    }
}

#[test]
fn test() {
    const JSON_FILE: &str = "tests/data/revlog_history.json";
    use burn::data::dataset::InMemDataset;
    use burn::data::dataloader::Dataset;
    use burn::data::dataloader::DataLoaderBuilder;
    let dataset = InMemDataset::<FSRSItem>::from_json_rows(JSON_FILE).unwrap();
    let item = dataset.get(704).unwrap();
    dbg!(&item);

    use burn_ndarray::NdArrayBackend;
    use burn_ndarray::NdArrayDevice;
    let device = NdArrayDevice::Cpu;
    type Backend = NdArrayBackend<f32>;
    let batcher = FSRSBatcher::<Backend>::new(device.clone());
    let dataloader = DataLoaderBuilder::new(batcher)
        .batch_size(1)
        .shuffle(42)
        .num_workers(4)
        .build(dataset);
    for _item in dataloader.iter() {
        // dbg!(&item.r_historys);
    }
}
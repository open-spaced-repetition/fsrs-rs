use burn::tensor::{backend::Backend, Data, Tensor, Float, ElementConversion, Shape};
use serde::{Deserialize, Serialize};
use burn::data::dataloader::batcher::Batcher;

const SEQ_LEN: usize = 32;

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
    pub labels: Tensor<B, 1, Float>,
}

impl<B: Backend> Batcher<FSRSItem, FSRSBatch<B>> for FSRSBatcher<B> {
    fn batch(&self, items: Vec<FSRSItem>) -> FSRSBatch<B> {

        let t_historys = items
            .iter()
            .map(|item| Data::new(item.t_history.clone(), Shape { dims: [SEQ_LEN] }))
            .map(|data| Tensor::<B, 1>::from_data(data.convert()))
            .map(|tensor| tensor.reshape([1, SEQ_LEN]))
            .collect();

        let r_historys = items
            .iter()
            .map(|item| Data::new(item.r_history.clone(), Shape { dims: [SEQ_LEN] }))
            .map(|data| Tensor::<B, 1>::from_data(data.convert()))
            .map(|tensor| tensor.reshape([1, SEQ_LEN]))
            .collect();

        let delta_ts = items
            .iter()
            .map(|item| Tensor::<B, 1, Float>::from_data(Data::from([item.delta_t.elem()])))
            .collect();

        let labels = items
            .iter()
            .map(|item| Tensor::<B, 1, Float>::from_data(Data::from([item.label.elem()])))
            .collect();

        let t_historys = Tensor::cat(t_historys, 0).to_device(&self.device);
        let r_historys = Tensor::cat(r_historys, 0).to_device(&self.device);
        let delta_ts = Tensor::cat(delta_ts, 0).to_device(&self.device);
        let labels = Tensor::cat(labels, 0).to_device(&self.device);

        FSRSBatch { t_historys, r_historys, delta_ts, labels }
    }
}

#[test]
fn test() {
    const JSON_FILE: &str = "tests/data/revlog_history.json";
    use burn::data::dataset::InMemDataset;
    use burn::data::dataloader::Dataset;
    use burn::data::dataloader::DataLoaderBuilder;
    let dataset = InMemDataset::<FSRSItem>::from_json_rows(JSON_FILE).unwrap();
    dbg!(&dataset.get(704));


    use burn_ndarray::NdArrayBackend;
    use burn_ndarray::NdArrayDevice;
    let device = NdArrayDevice::Cpu;
    type Backend = NdArrayBackend<f32>;
    let batcher = FSRSBatcher::<Backend>::new(device.clone());
    DataLoaderBuilder::new(batcher)
        .batch_size(1)
        .shuffle(42)
        .num_workers(1)
        .build(dataset);
}
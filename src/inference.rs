use burn::config::Config;
use burn::tensor::Tensor;
use burn::{data::dataloader::batcher::Batcher, tensor::backend::Backend};

use crate::{dataset::FSRSBatcher, training::TrainingConfig, FSRSItem};

pub fn infer<B: Backend<FloatElem = f32>>(
    artifact_dir: &str,
    device: B::Device,
    items: Vec<FSRSItem>,
) -> (Tensor<B, 2>, Tensor<B, 2>, Tensor<B, 2>) {
    let config =
        TrainingConfig::load(format!("{artifact_dir}/config.json")).expect("A config exists");

    let model = config.model.init::<B>();

    let batcher = FSRSBatcher::new(device);
    let batch = batcher.batch(items);
    let (stability, difficulty) = model.forward(batch.t_historys, batch.r_historys);
    let retention = model.power_forgetting_curve(
        batch.delta_ts.clone().unsqueeze::<2>().transpose(),
        stability.clone(),
    );
    (stability, difficulty, retention)
}

#[test]
fn test_infer() {
    use crate::convertor::tests::anki21_sample_file_converted_to_fsrs;
    use burn_ndarray::NdArrayBackend;
    use burn_ndarray::NdArrayDevice;
    type Backend = NdArrayBackend<f32>;
    let artifact_dir = "./tmp/fsrs";
    let device = NdArrayDevice::Cpu;
    let items = anki21_sample_file_converted_to_fsrs();
    let (stability, difficulty, retention) = infer::<Backend>(artifact_dir, device, items);
    dbg!(stability);
    dbg!(difficulty);
    dbg!(retention);
}

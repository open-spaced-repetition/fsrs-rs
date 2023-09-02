#![allow(dead_code)]
use std::collections::HashMap;

use burn::tensor::Tensor;
use burn::{data::dataloader::batcher::Batcher, tensor::backend::Backend};

use crate::dataset::FSRSBatch;
use crate::dataset::FSRSBatcher;
use crate::model::Model;
use crate::training::BCELoss;
use crate::FSRSItem;

fn infer<B: Backend<FloatElem = f32>>(
    model: Model<B>,
    batch: FSRSBatch<B>,
) -> (Tensor<B, 2>, Tensor<B, 2>, Tensor<B, 2>) {
    let (stability, difficulty) = model.forward(batch.t_historys, batch.r_historys);
    let retention = model.power_forgetting_curve(
        batch.delta_ts.clone().unsqueeze::<2>().transpose(),
        stability.clone(),
    );
    (stability, difficulty, retention)
}

pub fn evaluate<B: Backend<FloatElem = f32>>(
    model: Model<B>,
    device: B::Device,
    items: Vec<FSRSItem>,
) -> (f32, f32) {
    let batcher = FSRSBatcher::<B>::new(device);
    let batch = batcher.batch(items);
    let (_stability, _difficulty, retention) = infer::<B>(model, batch.clone());
    let pred = retention.clone().squeeze::<1>(1).to_data().value;
    let true_val = batch.labels.clone().float().to_data().value;
    let rmse = calibration_rmse(pred, true_val);
    let loss =
        BCELoss::<B>::new().forward(retention, batch.labels.unsqueeze::<2>().float().transpose());
    (loss.to_data().value[0], rmse)
}

fn calibration_rmse(pred: Vec<f32>, true_val: Vec<f32>) -> f32 {
    if pred.len() != true_val.len() {
        panic!("Vectors pred and true_val must have the same length");
    }

    let mut groups = HashMap::new();

    fn get_bin(x: f32, bins: f32) -> i32 {
        let log_base = bins.ln();
        let binned_x = (x * log_base).exp().round();
        binned_x.round() as i32
    }

    for (&p, &t) in pred.iter().zip(true_val.iter()) {
        let bin = get_bin(p, 20.0);
        groups.entry(bin).or_insert_with(Vec::new).push((p, t));
    }

    let mut total_sum = 0.0;
    let mut total_count = 0.0;

    for (_bin, group) in groups.iter() {
        let count = group.len() as f32;
        let pred_mean = group.iter().map(|(p, _)| *p).sum::<f32>() / count;
        let true_mean = group.iter().map(|(_, t)| *t).sum::<f32>() / count;

        let rmse = (pred_mean - true_mean).powi(2);
        total_sum += rmse * count;
        total_count += count;
    }

    (total_sum / total_count).sqrt()
}

#[test]
fn test_evaluate() {
    use crate::convertor::tests::anki21_sample_file_converted_to_fsrs;
    use crate::model::ModelConfig;
    use burn::module::Param;
    use burn::tensor::{Data, Shape, Tensor};
    use burn_ndarray::NdArrayBackend;
    use burn_ndarray::NdArrayDevice;
    type Backend = NdArrayBackend<f32>;
    let device = NdArrayDevice::Cpu;
    let config = ModelConfig::default();

    let items = anki21_sample_file_converted_to_fsrs();

    let metrics = evaluate(Model::<Backend>::new(config.clone()), device, items.clone());

    assert_eq!(metrics, (0.20820294, 0.043400552));

    let mut model = Model::<Backend>::new(config);
    model.w = Param::from(Tensor::from_floats(Data::new(
        vec![
            0.81497127,
            1.5411042,
            4.007436,
            9.045982,
            4.956448,
            1.3552042,
            1.0985811,
            0.007904565,
            1.6491636,
            0.13996966,
            1.0704349,
            2.3238432,
            0.034056284,
            0.35500556,
            1.5469967,
            0.10132355,
            2.7867608,
        ],
        Shape { dims: [17] },
    )));
    let metrics = evaluate::<Backend>(model, device, items);

    assert_eq!(metrics, (0.20209138, 0.017994177));
}

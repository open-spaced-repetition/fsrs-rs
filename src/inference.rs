use std::collections::HashMap;

use crate::model::ModelConfig;
use burn::backend::ndarray::NdArrayDevice;
use burn::backend::NdArrayBackend;
use burn::module::Param;
use burn::tensor::{Data, Shape, Tensor};
use burn::{data::dataloader::batcher::Batcher, tensor::backend::Backend};

use crate::dataset::FSRSBatch;
use crate::dataset::FSRSBatcher;
use crate::error::Result;
use crate::model::Model;
use crate::training::BCELoss;
use crate::{FSRSError, FSRSItem};

fn infer<B: Backend<FloatElem = f32>>(
    model: &Model<B>,
    batch: FSRSBatch<B>,
) -> (Tensor<B, 2>, Tensor<B, 2>, Tensor<B, 2>) {
    let (stability, difficulty) = model.forward(batch.t_historys, batch.r_historys);
    let retention = model.power_forgetting_curve(
        batch.delta_ts.clone().unsqueeze::<2>().transpose(),
        stability.clone(),
    );
    (stability, difficulty, retention)
}

#[derive(Debug, Clone, Copy)]
pub struct ItemProgress {
    pub current: usize,
    pub total: usize,
}

pub fn evaluate<F>(weights: [f32; 17], items: Vec<FSRSItem>, mut progress: F) -> Result<(f32, f32)>
where
    F: FnMut(ItemProgress) -> bool,
{
    type Backend = NdArrayBackend<f32>;
    let device = NdArrayDevice::Cpu;
    let batcher = FSRSBatcher::<Backend>::new(device);
    let config = ModelConfig::default();
    let mut model = Model::<Backend>::new(config);
    model.w = Param::from(Tensor::from_floats(Data::new(
        weights.to_vec(),
        Shape { dims: [17] },
    )));
    let mut all_pred = vec![];
    let mut all_true_val = vec![];
    let mut all_retention = vec![];
    let mut all_labels = vec![];
    let mut progress_info = ItemProgress {
        current: 0,
        total: items.len(),
    };
    for chunk in items.chunks(512) {
        let batch = batcher.batch(chunk.to_vec());
        let (_stability, _difficulty, retention) = infer::<Backend>(&model, batch.clone());
        let pred = retention.clone().squeeze::<1>(1).to_data().value;
        all_pred.extend(pred);
        let true_val = batch.labels.clone().float().to_data().value;
        all_true_val.extend(true_val);
        all_retention.push(retention);
        all_labels.push(batch.labels);
        progress_info.current += chunk.len();
        if !progress(progress_info) {
            return Err(FSRSError::Interrupted);
        }
    }
    let rmse = calibration_rmse(all_pred, all_true_val);
    let all_retention = Tensor::cat(all_retention, 0);
    let all_labels = Tensor::cat(all_labels, 0)
        .unsqueeze::<2>()
        .float()
        .transpose();
    let loss = BCELoss::<Backend>::new().forward(all_retention, all_labels);
    Ok((loss.to_data().value[0], rmse))
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

    for (p, t) in pred.iter().zip(true_val) {
        let bin = get_bin(*p, 20.0);
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::convertor_tests::anki21_sample_file_converted_to_fsrs;

    #[test]
    fn test_evaluate() {
        let items = anki21_sample_file_converted_to_fsrs();

        let metrics = evaluate(
            [
                0.4, 0.6, 2.4, 5.8, 4.93, 0.94, 0.86, 0.01, 1.49, 0.14, 0.94, 2.18, 0.05, 0.34,
                1.26, 0.29, 2.61,
            ],
            items.clone(),
            |_| true,
        )
        .unwrap();

        Data::from([metrics.0, metrics.1])
            .assert_approx_eq(&Data::from([0.20820294, 0.043400552]), 5);

        let metrics = evaluate(
            [
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
            items,
            |_| true,
        )
        .unwrap();

        Data::from([metrics.0, metrics.1])
            .assert_approx_eq(&Data::from([0.20209138, 0.017994177]), 5);
    }
}

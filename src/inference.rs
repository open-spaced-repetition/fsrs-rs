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
) -> Tensor<B, 1> {
    let batcher = FSRSBatcher::<B>::new(device);
    let batch = batcher.batch(items);
    let (_stability, _difficulty, retention) = infer::<B>(model, batch.clone());
    let loss =
        BCELoss::<B>::new().forward(retention, batch.labels.unsqueeze::<2>().float().transpose());
    loss
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

    let loss = evaluate(Model::<Backend>::new(config.clone()), device, items.clone());
    dbg!(&loss);

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
    let loss = evaluate::<Backend>(model, device, items);
    dbg!(&loss);
}

use crate::batch_shuffle::BatchShuffledDataset;
use crate::cosine_annealing::CosineAnnealingLR;
use crate::dataset::{FSRSBatch, FSRSBatcher, FSRSDataset, FSRSItem};
use crate::model::{Model, ModelConfig};
use crate::weight_clipper::weight_clipper;
use burn::optim::AdamConfig;
use burn::record::{FullPrecisionSettings, PrettyJsonFileRecorder};
use burn::tensor::backend::Backend;
use burn::tensor::{Int, Tensor};
use burn::train::{ClassificationOutput, TrainOutput, TrainStep, ValidStep};
use burn::{
    config::Config, data::dataloader::DataLoaderBuilder, module::Param, tensor::backend::ADBackend,
    train::LearnerBuilder,
};
use log::info;

impl<B: Backend<FloatElem = f32>> Model<B> {
    fn bceloss(&self, retentions: Tensor<B, 2>, labels: Tensor<B, 2>) -> Tensor<B, 1> {
        let loss: Tensor<B, 2> =
            labels.clone() * retentions.clone().log() + (-labels + 1) * (-retentions + 1).log();
        info!("loss: {}", &loss);
        loss.mean().neg()
    }

    pub fn forward_classification(
        &self,
        t_historys: Tensor<B, 2>,
        r_historys: Tensor<B, 2>,
        delta_ts: Tensor<B, 1>,
        labels: Tensor<B, 1, Int>,
    ) -> ClassificationOutput<B> {
        // info!("t_historys: {}", &t_historys);
        // info!("r_historys: {}", &r_historys);
        let (stability, _difficulty) = self.forward(t_historys, r_historys);
        let retention = self.power_forgetting_curve(
            delta_ts.clone().unsqueeze::<2>().transpose(),
            stability.clone(),
        );
        let logits = Tensor::cat(vec![-retention.clone() + 1, retention.clone()], 1);
        info!("stability: {}", &stability);
        info!(
            "delta_ts: {}",
            &delta_ts.clone().unsqueeze::<2>().transpose()
        );
        info!("retention: {}", &retention);
        info!("logits: {}", &logits);
        info!(
            "labels: {}",
            &labels.clone().unsqueeze::<2>().float().transpose()
        );
        let loss = self.bceloss(
            retention,
            labels.clone().unsqueeze::<2>().float().transpose(),
        );
        info!("loss: {}", &loss);
        ClassificationOutput::new(loss, logits, labels)
    }
}

impl<B: ADBackend<FloatElem = f32>> Model<B> {
    fn freeze_initial_stability(&self, mut grad: B::Gradients) -> B::Gradients {
        let grad_tensor = self.w.grad(&grad).unwrap();
        let updated_grad_tensor = grad_tensor.slice_assign([0..4], Tensor::zeros([4]));

        self.w.grad_remove(&mut grad);
        self.w.grad_replace(&mut grad, updated_grad_tensor);
        grad
    }
}

impl<B: ADBackend<FloatElem = f32>> TrainStep<FSRSBatch<B>, ClassificationOutput<B>> for Model<B> {
    fn step(&self, batch: FSRSBatch<B>) -> TrainOutput<ClassificationOutput<B>> {
        let item = self.forward_classification(
            batch.t_historys,
            batch.r_historys,
            batch.delta_ts,
            batch.labels,
        );
        let mut gradients = item.loss.backward();

        if self.freeze_stability {
            gradients = self.freeze_initial_stability(gradients);
        }

        TrainOutput::new(self, gradients, item)
    }

    fn optimize<B1, O>(self, optim: &mut O, lr: f64, grads: burn::optim::GradientsParams) -> Self
    where
        B: ADBackend,
        O: burn::optim::Optimizer<Self, B1>,
        B1: burn::tensor::backend::ADBackend,
        Self: burn::module::ADModule<B1>,
    {
        let mut model = optim.step(lr, self, grads);
        model.w = Param::from(weight_clipper(model.w.val()));
        model
    }
}

impl<B: Backend<FloatElem = f32>> ValidStep<FSRSBatch<B>, ClassificationOutput<B>> for Model<B> {
    fn step(&self, batch: FSRSBatch<B>) -> ClassificationOutput<B> {
        self.forward_classification(
            batch.t_historys,
            batch.r_historys,
            batch.delta_ts,
            batch.labels,
        )
    }
}

#[derive(Config)]
pub struct TrainingConfig {
    pub model: ModelConfig,
    pub optimizer: AdamConfig,
    #[config(default = 10)]
    pub num_epochs: usize,
    #[config(default = 512)]
    pub batch_size: usize,
    #[config(default = 4)]
    pub num_workers: usize,
    #[config(default = 42)]
    pub seed: u64,
    #[config(default = 8.0e-3)]
    pub learning_rate: f64,
}

pub fn compute_weights(items: Vec<FSRSItem>) -> Vec<f32> {
    use burn_ndarray::NdArrayBackend;
    use burn_ndarray::NdArrayDevice;
    type Backend = NdArrayBackend<f32>;
    type AutodiffBackend = burn_autodiff::ADBackendDecorator<Backend>;
    let device = NdArrayDevice::Cpu;

    let model = train::<AutodiffBackend>(
        items,
        &TrainingConfig::new(
            ModelConfig {
                freeze_stability: true,
            },
            AdamConfig::new(),
        ),
        device,
        None,
    );

    model.w.val().to_data().value
}

fn train<B: ADBackend<FloatElem = f32>>(
    items: Vec<FSRSItem>,
    config: &TrainingConfig,
    device: B::Device,
    artifact_dir: Option<&str>,
) -> Model<B> {
    B::seed(config.seed);

    // Training data
    let dataset_size = items.len();
    let batcher_train = FSRSBatcher::<B>::new(device.clone());
    let dataloader_train = DataLoaderBuilder::new(batcher_train)
        .batch_size(config.batch_size)
        .build(BatchShuffledDataset::with_seed(
            FSRSDataset::from(items),
            config.batch_size,
            config.seed,
        ));

    // We don't use any validation data
    let batcher_valid = FSRSBatcher::<B::InnerBackend>::new(device.clone());
    let dataloader_valid = DataLoaderBuilder::new(batcher_valid).build(FSRSDataset::from(vec![]));

    let lr_scheduler = CosineAnnealingLR::init(
        (dataset_size * config.num_epochs) as f64,
        config.learning_rate,
    );

    let mut builder = LearnerBuilder::new(artifact_dir.unwrap_or_default())
        .devices(vec![device])
        .num_epochs(config.num_epochs);
    // options only required when testing
    if artifact_dir.is_some() {
        builder = builder
            .with_file_checkpointer(10, PrettyJsonFileRecorder::<FullPrecisionSettings>::new())
        // .metric_train_plot(AccuracyMetric::new())
        // .metric_valid_plot(AccuracyMetric::new())
        // .metric_train_plot(LossMetric::new())
        // .metric_valid_plot(LossMetric::new())
    }
    let learner = builder.build(
        config.model.init::<B>(),
        config.optimizer.init(),
        lr_scheduler,
    );

    let mut model_trained = learner.fit(dataloader_train, dataloader_valid);
    info!("trained weights: {}", &model_trained.w.val());
    model_trained.w = Param::from(weight_clipper(model_trained.w.val()));
    info!("clipped weights: {}", &model_trained.w.val());

    model_trained
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::convertor::tests::anki21_sample_file_converted_to_fsrs;
    use burn::module::Module;
    use burn::record::Recorder;
    use std::path::Path;

    #[test]
    fn training() {
        if std::env::var("SKIP_TRAINING").is_ok() {
            println!("Skipping test in CI");
            return;
        }
        use burn_ndarray::NdArrayBackend;
        use burn_ndarray::NdArrayDevice;
        type Backend = NdArrayBackend<f32>;
        type AutodiffBackend = burn_autodiff::ADBackendDecorator<Backend>;
        let device = NdArrayDevice::Cpu;

        let artifact_dir = "./tmp/fsrs";
        let config = TrainingConfig::new(
            ModelConfig {
                freeze_stability: true,
            },
            AdamConfig::new(),
        );

        std::fs::create_dir_all(artifact_dir).unwrap();
        config
            .save(
                Path::new(artifact_dir)
                    .join("config.json")
                    .to_str()
                    .unwrap(),
            )
            .expect("Save without error");

        let model_trained = train::<AutodiffBackend>(
            anki21_sample_file_converted_to_fsrs(),
            &config,
            device,
            Some(artifact_dir),
        );

        config
            .save(
                Path::new(artifact_dir)
                    .join("config.json")
                    .to_str()
                    .unwrap(),
            )
            .unwrap();

        PrettyJsonFileRecorder::<FullPrecisionSettings>::new()
            .record(
                model_trained.into_record(),
                Path::new(artifact_dir).join("model"),
            )
            .expect("Failed to save trained model");
    }
}

use crate::dataset::{FSRSBatch, FSRSBatcher, FSRSDataset};
use crate::model::{Model, ModelConfig};
use crate::weight_clipper::weight_clipper;
use burn::module::Module;
use burn::optim::AdamConfig;
use burn::record::{FullPrecisionSettings, PrettyJsonFileRecorder, Recorder};
use burn::tensor::backend::Backend;
use burn::tensor::{Int, Tensor};
use burn::train::{ClassificationOutput, TrainOutput, TrainStep, ValidStep};
use burn::{
    config::Config, data::dataloader::DataLoaderBuilder, module::Param, tensor::backend::ADBackend,
    train::LearnerBuilder,
};
use log::info;

impl<B: Backend<FloatElem = f32>> Model<B> {
    fn bceloss(&self, retentions: Tensor<B, 1>, labels: Tensor<B, 1>) -> Tensor<B, 1> {
        let loss: Tensor<B, 1> =
            labels.clone() * retentions.clone().log() + (-labels + 1) * (-retentions + 1).log();
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
        let retention = self.power_forgetting_curve(delta_ts.clone(), stability.clone());
        let logits =
            Tensor::cat(vec![-retention.clone() + 1, retention.clone()], 0).reshape([1, -1]);
        info!("stability: {}", &stability);
        info!("delta_ts: {}", &delta_ts);
        info!("retention: {}", &retention);
        info!("logits: {}", &logits);
        info!("labels: {}", &labels);
        let loss = self.bceloss(retention.clone(), labels.clone().float());
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

static ARTIFACT_DIR: &str = "./tmp/fsrs";

#[derive(Config)]
pub struct TrainingConfig {
    pub model: ModelConfig,
    pub optimizer: AdamConfig,
    #[config(default = 1)]
    pub num_epochs: usize,
    #[config(default = 1)]
    pub batch_size: usize,
    #[config(default = 4)]
    pub num_workers: usize,
    #[config(default = 42)]
    pub seed: u64,
    #[config(default = 1.0e-4)]
    pub learning_rate: f64,
}

pub fn train<B: ADBackend<FloatElem = f32>>(
    artifact_dir: &str,
    config: TrainingConfig,
    device: B::Device,
) {
    std::fs::create_dir_all(artifact_dir).ok();
    config
        .save(&format!("{artifact_dir}/config.json"))
        .expect("Save without error");

    B::seed(config.seed);

    // Data
    let batcher_train: FSRSBatcher<B> = FSRSBatcher::<B>::new(device.clone());
    let batcher_valid = FSRSBatcher::<B::InnerBackend>::new(device.clone());

    let dataloader_train = DataLoaderBuilder::new(batcher_train)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(FSRSDataset::train());

    let dataloader_test = DataLoaderBuilder::new(batcher_valid)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(FSRSDataset::test());

    let learner = LearnerBuilder::new(artifact_dir)
        // .metric_train_plot(AccuracyMetric::new())
        // .metric_valid_plot(AccuracyMetric::new())
        // .metric_train_plot(LossMetric::new())
        // .metric_valid_plot(LossMetric::new())
        .with_file_checkpointer(1, PrettyJsonFileRecorder::<FullPrecisionSettings>::new())
        .devices(vec![device])
        .num_epochs(config.num_epochs)
        .build(
            config.model.init::<B>(),
            config.optimizer.init(),
            config.learning_rate,
        );

    let mut model_trained = learner.fit(dataloader_train, dataloader_test);
    info!("trained weights: {}", &model_trained.w.val());
    model_trained.w = Param::from(weight_clipper(model_trained.w.val()));
    info!("clipped weights: {}", &model_trained.w.val());

    config
        .save(format!("{ARTIFACT_DIR}/config.json").as_str())
        .unwrap();

    PrettyJsonFileRecorder::<FullPrecisionSettings>::new()
        .record(
            model_trained.clone().into_record(),
            format!("{ARTIFACT_DIR}/model").into(),
        )
        .expect("Failed to save trained model");
}

#[test]
fn test() {
    use burn_ndarray::NdArrayBackend;
    use burn_ndarray::NdArrayDevice;
    type Backend = NdArrayBackend<f32>;
    type AutodiffBackend = burn_autodiff::ADBackendDecorator<Backend>;
    let device = NdArrayDevice::Cpu;

    let artifact_dir = ARTIFACT_DIR;
    train::<AutodiffBackend>(
        artifact_dir,
        TrainingConfig::new(ModelConfig {freeze_stability: true}, AdamConfig::new()),
        device,
    );
}

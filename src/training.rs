use crate::batch_shuffle::BatchShuffledDataset;
use crate::cosine_annealing::CosineAnnealingLR;
use crate::dataset::{split_data, FSRSBatch, FSRSBatcher, FSRSDataset, FSRSItem};
use crate::error::Result;
use crate::model::{Model, ModelConfig};
use crate::pre_training::pretrain;
use crate::weight_clipper::weight_clipper;
use crate::{FSRSError, FSRS};
use burn::autodiff::ADBackendDecorator;
use burn::module::Module;
use burn::optim::AdamConfig;
use burn::record::{FullPrecisionSettings, PrettyJsonFileRecorder, Recorder};
use burn::tensor::backend::Backend;
use burn::tensor::{Int, Tensor};
use burn::train::metric::dashboard::{DashboardMetricState, DashboardRenderer, TrainingProgress};
use burn::train::{ClassificationOutput, TrainOutput, TrainStep, TrainingInterrupter, ValidStep};
use burn::{
    config::Config, data::dataloader::DataLoaderBuilder, module::Param, tensor::backend::ADBackend,
    train::LearnerBuilder,
};
use core::marker::PhantomData;
use log::info;
use std::path::Path;
use std::sync::{Arc, Mutex};

pub struct BCELoss<B: Backend> {
    backend: PhantomData<B>,
}

impl<B: Backend> BCELoss<B> {
    pub fn new() -> Self {
        Self {
            backend: PhantomData,
        }
    }
    pub fn forward(&self, retentions: Tensor<B, 1>, labels: Tensor<B, 1>) -> Tensor<B, 1> {
        let loss =
            labels.clone() * retentions.clone().log() + (-labels + 1) * (-retentions + 1).log();
        info!("loss: {}", &loss);
        loss.mean().neg()
    }
}

impl<B: Backend> Model<B> {
    pub fn forward_classification(
        &self,
        t_historys: Tensor<B, 2>,
        r_historys: Tensor<B, 2>,
        delta_ts: Tensor<B, 1>,
        labels: Tensor<B, 1, Int>,
    ) -> ClassificationOutput<B> {
        // info!("t_historys: {}", &t_historys);
        // info!("r_historys: {}", &r_historys);
        let state = self.forward(t_historys, r_historys);
        let retention = self.power_forgetting_curve(delta_ts.clone(), state.stability);
        let logits =
            Tensor::cat(vec![-retention.clone() + 1, retention.clone()], 0).unsqueeze::<2>();
        let loss = BCELoss::new().forward(retention, labels.clone().float());
        ClassificationOutput::new(loss, logits, labels)
    }
}

impl<B: ADBackend> Model<B> {
    fn freeze_initial_stability(&self, mut grad: B::Gradients) -> B::Gradients {
        let grad_tensor = self.w.grad(&grad).unwrap();
        let updated_grad_tensor = grad_tensor.slice_assign([0..4], Tensor::zeros([4]));

        self.w.grad_remove(&mut grad);
        self.w.grad_replace(&mut grad, updated_grad_tensor);
        grad
    }
}

impl<B: ADBackend> TrainStep<FSRSBatch<B>, ClassificationOutput<B>> for Model<B> {
    fn step(&self, batch: FSRSBatch<B>) -> TrainOutput<ClassificationOutput<B>> {
        let item = self.forward_classification(
            batch.t_historys,
            batch.r_historys,
            batch.delta_ts,
            batch.labels,
        );
        let mut gradients = item.loss.backward();

        if self.config.freeze_stability {
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

impl<B: Backend> ValidStep<FSRSBatch<B>, ClassificationOutput<B>> for Model<B> {
    fn step(&self, batch: FSRSBatch<B>) -> ClassificationOutput<B> {
        self.forward_classification(
            batch.t_historys,
            batch.r_historys,
            batch.delta_ts,
            batch.labels,
        )
    }
}

#[derive(Debug, Default)]
pub struct ProgressState {
    pub epoch: usize,
    pub epoch_total: usize,
    pub items_processed: usize,
    pub items_total: usize,
    pub want_abort: bool,
}

#[derive(Clone, Default)]
pub struct ProgressCollector {
    pub state: Arc<Mutex<ProgressState>>,
    pub interrupter: TrainingInterrupter,
}

impl ProgressCollector {
    pub fn new(state: Arc<Mutex<ProgressState>>) -> Self {
        Self {
            state,
            ..Default::default()
        }
    }
}

impl ProgressState {
    pub fn new_shared() -> Arc<Mutex<Self>> {
        Default::default()
    }

    pub fn current(&self) -> usize {
        self.epoch.saturating_sub(1) * self.items_total + self.items_processed
    }

    pub fn total(&self) -> usize {
        self.epoch_total * self.items_total
    }
}

impl DashboardRenderer for ProgressCollector {
    fn update_train(&mut self, _state: DashboardMetricState) {}

    fn update_valid(&mut self, _state: DashboardMetricState) {}

    fn render_train(&mut self, item: TrainingProgress) {
        let mut info = self.state.lock().unwrap();
        info.epoch = item.epoch;
        info.epoch_total = item.epoch_total;
        info.items_processed = item.progress.items_processed;
        info.items_total = item.progress.items_total;
        if info.want_abort {
            self.interrupter.stop();
        }
    }

    fn render_valid(&mut self, _item: TrainingProgress) {}
}

#[derive(Config)]
pub(crate) struct TrainingConfig {
    pub model: ModelConfig,
    pub optimizer: AdamConfig,
    #[config(default = 16)]
    pub num_epochs: usize,
    #[config(default = 512)]
    pub batch_size: usize,
    #[config(default = 4)]
    pub num_workers: usize,
    #[config(default = 42)]
    pub seed: u64,
    #[config(default = 1.0e-3)]
    pub learning_rate: f64,
}

impl<B: Backend> FSRS<B> {
    /// Calculate appropriate weights for the provided review history.
    pub fn compute_weights(
        &self,
        items: Vec<FSRSItem>,
        progress: Option<Arc<Mutex<ProgressState>>>,
    ) -> Result<Vec<f32>> {
        let (pre_trainset, trainset) = split_data(items);
        let initial_stability = pretrain(pre_trainset)?;
        let config = TrainingConfig::new(
            ModelConfig {
                freeze_stability: true,
                initial_stability: Some(initial_stability),
            },
            AdamConfig::new(),
        );

        let model = train::<ADBackendDecorator<B>>(
            trainset,
            &config,
            self.device(),
            progress.map(ProgressCollector::new),
        );

        Ok(model?.w.val().to_data().convert().value)
    }
}

fn train<B: ADBackend>(
    items: Vec<FSRSItem>,
    config: &TrainingConfig,
    device: B::Device,
    progress: Option<ProgressCollector>,
) -> Result<Model<B>> {
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

    let artifact_dir = std::env::var("BURN_LOG");

    let mut builder = LearnerBuilder::new(&artifact_dir.clone().unwrap_or_default())
        .devices(vec![device])
        .num_epochs(config.num_epochs)
        .log_to_file(false);
    let interrupter = builder.interrupter();

    if let Some(mut progress) = progress {
        progress.interrupter = interrupter.clone();
        builder = builder.renderer(progress);
    } else {
        // comment out if you want to see text interface
        builder = builder.renderer(NoProgress {});
        // builder = builder
        //     .metric_train_plot(AccuracyMetric::new())
        //     .metric_valid_plot(AccuracyMetric::new())
        //     .metric_train_plot(LossMetric::new())
        //     .metric_valid_plot(LossMetric::new());
    }

    if artifact_dir.is_ok() {
        builder = builder
            .log_to_file(true)
            .with_file_checkpointer(10, PrettyJsonFileRecorder::<FullPrecisionSettings>::new());
    }

    let learner = builder.build(
        config.model.init::<B>(),
        config.optimizer.init(),
        lr_scheduler,
    );

    let mut model_trained = learner.fit(dataloader_train, dataloader_valid);

    if interrupter.should_stop() {
        return Err(FSRSError::Interrupted);
    }

    info!("trained weights: {}", &model_trained.w.val());
    model_trained.w = Param::from(weight_clipper(model_trained.w.val()));
    info!("clipped weights: {}", &model_trained.w.val());

    if let Ok(path) = artifact_dir {
        PrettyJsonFileRecorder::<FullPrecisionSettings>::new()
            .record(
                model_trained.clone().into_record(),
                Path::new(&path).join("model"),
            )
            .expect("Failed to save trained model");
    }

    Ok(model_trained)
}

struct NoProgress {}

impl DashboardRenderer for NoProgress {
    fn update_train(&mut self, _state: DashboardMetricState) {}

    fn update_valid(&mut self, _state: DashboardMetricState) {}

    fn render_train(&mut self, _item: TrainingProgress) {}

    fn render_valid(&mut self, _item: TrainingProgress) {}
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::convertor_tests::anki21_sample_file_converted_to_fsrs;
    use crate::pre_training::pretrain;
    use burn::backend::ndarray::NdArrayDevice;
    use burn::backend::NdArrayAutodiffBackend;

    #[test]
    fn training() {
        if std::env::var("SKIP_TRAINING").is_ok() {
            println!("Skipping test in CI");
            return;
        }
        let device = NdArrayDevice::Cpu;

        let (pre_trainset, trainset) = split_data(anki21_sample_file_converted_to_fsrs());
        let initial_stability = pretrain(pre_trainset).unwrap();
        let config = TrainingConfig::new(
            ModelConfig {
                freeze_stability: true,
                initial_stability: Some(initial_stability),
            },
            AdamConfig::new(),
        );

        let _model_trained =
            train::<NdArrayAutodiffBackend>(trainset, &config, device, None).unwrap();
    }
}

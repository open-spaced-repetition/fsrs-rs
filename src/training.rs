use crate::batch_shuffle::{BatchShuffledDataLoaderBuilder, BatchShuffledDataset};
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
use burn::train::renderer::{MetricState, MetricsRenderer, TrainingProgress};

use burn::train::{ClassificationOutput, TrainOutput, TrainStep, TrainingInterrupter, ValidStep};
use burn::{config::Config, module::Param, tensor::backend::ADBackend, train::LearnerBuilder};
use core::marker::PhantomData;
use log::info;
use rayon::prelude::{
    IndexedParallelIterator, IntoParallelIterator, IntoParallelRefIterator, ParallelIterator,
};
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
        // info!("loss: {}", &loss);
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
        let state = self.forward(t_historys, r_historys, None);
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

#[derive(Debug, Default, Clone)]
pub struct ProgressState {
    pub epoch: usize,
    pub epoch_total: usize,
    pub items_processed: usize,
    pub items_total: usize,
}

#[derive(Default)]
pub struct CombinedProgressState {
    pub want_abort: bool,
    pub splits: Vec<ProgressState>,
}

impl CombinedProgressState {
    pub fn new_shared() -> Arc<Mutex<Self>> {
        Default::default()
    }

    pub fn current(&self) -> usize {
        self.splits.iter().map(|s| s.current()).sum()
    }

    pub fn total(&self) -> usize {
        self.splits.iter().map(|s| s.total()).sum()
    }
}

#[derive(Clone)]
pub struct ProgressCollector {
    pub state: Arc<Mutex<CombinedProgressState>>,
    pub interrupter: TrainingInterrupter,
    /// The index of the split we should update.
    pub index: usize,
}

impl ProgressCollector {
    pub fn new(state: Arc<Mutex<CombinedProgressState>>, index: usize) -> Self {
        Self {
            state,
            interrupter: Default::default(),
            index,
        }
    }
}

impl ProgressState {
    pub fn current(&self) -> usize {
        self.epoch.saturating_sub(1) * self.items_total + self.items_processed
    }

    pub fn total(&self) -> usize {
        self.epoch_total * self.items_total
    }
}

impl MetricsRenderer for ProgressCollector {
    fn update_train(&mut self, _state: MetricState) {}

    fn update_valid(&mut self, _state: MetricState) {}

    fn render_train(&mut self, item: TrainingProgress) {
        let mut info = self.state.lock().unwrap();
        let split = &mut info.splits[self.index];
        split.epoch = item.epoch;
        split.epoch_total = item.epoch_total;
        split.items_processed = item.progress.items_processed;
        split.items_total = item.progress.items_total;
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
    #[config(default = 5)]
    pub num_epochs: usize,
    #[config(default = 512)]
    pub batch_size: usize,
    #[config(default = 1)]
    pub num_workers: usize,
    #[config(default = 42)]
    pub seed: u64,
    #[config(default = 4e-2)]
    pub learning_rate: f64,
}

pub fn calculate_average_recall(items: &[FSRSItem]) -> f32 {
    let (total_recall, total_reviews) = items
        .iter()
        .flat_map(|item| item.reviews.iter())
        .fold((0u32, 0u32), |(sum, count), review| {
            (sum + (review.rating > 1) as u32, count + 1)
        });

    if total_reviews == 0 {
        return 0.0;
    }

    total_recall as f32 / total_reviews as f32
}

impl<B: Backend> FSRS<B> {
    /// Calculate appropriate weights for the provided review history.
    pub fn compute_weights(
        &self,
        items: Vec<FSRSItem>,
        mut progress: Option<Arc<Mutex<CombinedProgressState>>>,
    ) -> Result<Vec<f32>> {
        let n_splits = 5;
        if let Some(progress) = &mut progress {
            progress.lock().unwrap().splits = vec![ProgressState::default(); n_splits];
        }
        let average_recall = calculate_average_recall(&items);
        let (pre_trainset, trainsets) = split_data(items, n_splits);
        let initial_stability = pretrain(pre_trainset, average_recall)?;
        let config = TrainingConfig::new(
            ModelConfig {
                freeze_stability: true,
                initial_stability: Some(initial_stability),
            },
            AdamConfig::new(),
        );

        let weights_sets: Vec<Vec<f32>> = (0..n_splits)
            .into_par_iter()
            .map(|i| {
                let trainset = trainsets
                    .par_iter()
                    .enumerate()
                    .filter(|&(j, _)| j != i)
                    .flat_map(|(_, trainset)| trainset.clone())
                    .collect();

                let model = train::<ADBackendDecorator<B>>(
                    trainset,
                    &config,
                    self.device(),
                    progress.clone().map(|p| ProgressCollector::new(p, i)),
                );
                Ok(model?.w.val().to_data().convert().value)
            })
            .collect::<Result<_>>()?;

        let average_weights = weights_sets
            .iter()
            .fold(vec![0.0; weights_sets[0].len()], |sum, weights| {
                sum.par_iter().zip(weights).map(|(a, b)| a + b).collect()
            })
            .par_iter()
            .map(|&sum| sum / n_splits as f32)
            .collect();

        Ok(average_weights)
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
    let iterations = (items.len() / config.batch_size + 1) * config.num_epochs;
    let batcher_train = FSRSBatcher::<B>::new(device.clone());
    let dataloader_train = BatchShuffledDataLoaderBuilder::new(batcher_train)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(
            BatchShuffledDataset::with_seed(
                FSRSDataset::from(items),
                config.batch_size,
                config.seed,
            ),
            config.batch_size,
        );

    // We don't use any validation data
    let batcher_valid = FSRSBatcher::new(device.clone());
    let dataloader_valid = BatchShuffledDataLoaderBuilder::new(batcher_valid)
        .build(FSRSDataset::from(vec![]), config.batch_size);

    let lr_scheduler = CosineAnnealingLR::init(iterations as f64, config.learning_rate);

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
    }

    if artifact_dir.is_ok() {
        builder = builder
            .log_to_file(true)
            .with_file_checkpointer(PrettyJsonFileRecorder::<FullPrecisionSettings>::new());
    }

    let learner = builder.build(config.model.init(), config.optimizer.init(), lr_scheduler);

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

impl MetricsRenderer for NoProgress {
    fn update_train(&mut self, _state: MetricState) {}

    fn update_valid(&mut self, _state: MetricState) {}

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
    use rayon::prelude::IntoParallelIterator;

    #[test]
    fn training() {
        if std::env::var("SKIP_TRAINING").is_ok() {
            println!("Skipping test in CI");
            return;
        }
        let n_splits = 5;
        let device = NdArrayDevice::Cpu;
        let items = anki21_sample_file_converted_to_fsrs();
        let (pre_trainset, trainsets) = split_data(items, n_splits);
        let average_recall = calculate_average_recall(&pre_trainset);
        let initial_stability = pretrain(pre_trainset, average_recall).unwrap();
        let config = TrainingConfig::new(
            ModelConfig {
                freeze_stability: true,
                initial_stability: Some(initial_stability),
            },
            AdamConfig::new(),
        );

        let weights_sets: Vec<Vec<f32>> = (0..n_splits)
            .into_par_iter()
            .map(|i| {
                let trainset = trainsets
                    .par_iter()
                    .enumerate()
                    .filter(|&(j, _)| j != i)
                    .flat_map(|(_, trainset)| trainset.clone())
                    .collect();
                let model = train::<NdArrayAutodiffBackend>(trainset, &config, device, None);
                model.unwrap().w.val().to_data().convert().value
            })
            .collect();

        dbg!(&weights_sets);

        let average_weights: Vec<f32> = weights_sets
            .iter()
            .fold(vec![0.0; weights_sets[0].len()], |sum, weights| {
                sum.par_iter().zip(weights).map(|(a, b)| a + b).collect()
            })
            .par_iter()
            .map(|&sum| sum / n_splits as f32)
            .collect();
        dbg!(average_weights);
    }
}

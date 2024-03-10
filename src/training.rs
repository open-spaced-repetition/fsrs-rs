use crate::batch_shuffle::BatchShuffledDataLoaderBuilder;
use crate::cosine_annealing::CosineAnnealingLR;
use crate::dataset::{split_data, FSRSBatcher, FSRSDataset, FSRSItem};
use crate::error::Result;
use crate::model::{Model, ModelConfig};
use crate::pre_training::pretrain;
use crate::weight_clipper::weight_clipper;
use crate::{FSRSError, DEFAULT_PARAMETERS, FSRS};
use burn::backend::Autodiff;

use burn::data::dataloader::DataLoaderBuilder;
use burn::lr_scheduler::LrScheduler;
use burn::module::AutodiffModule;
use burn::nn::loss::Reduction;
use burn::optim::Optimizer;
use burn::optim::{AdamConfig, GradientsParams};
use burn::tensor::backend::Backend;
use burn::tensor::{Int, Tensor};
use burn::train::renderer::{MetricState, MetricsRenderer, TrainingProgress};
use burn::train::TrainingInterrupter;
use burn::{config::Config, module::Param, tensor::backend::AutodiffBackend};
use core::marker::PhantomData;
use log::info;

use rayon::prelude::{
    IndexedParallelIterator, IntoParallelIterator, IntoParallelRefIterator, ParallelIterator,
};

use std::sync::{Arc, Mutex};

pub struct BCELoss<B: Backend> {
    backend: PhantomData<B>,
}

impl<B: Backend> BCELoss<B> {
    pub const fn new() -> Self {
        Self {
            backend: PhantomData,
        }
    }
    pub fn forward(
        &self,
        retentions: Tensor<B, 1>,
        labels: Tensor<B, 1>,
        mean: Reduction,
    ) -> Tensor<B, 1> {
        let loss =
            labels.clone() * retentions.clone().log() + (-labels + 1) * (-retentions + 1).log();
        // info!("loss: {}", &loss);
        match mean {
            Reduction::Mean => loss.mean().neg(),
            Reduction::Sum => loss.sum().neg(),
            Reduction::Auto => loss.neg(),
        }
    }
}

impl<B: Backend> Model<B> {
    pub fn forward_classification(
        &self,
        t_historys: Tensor<B, 2>,
        r_historys: Tensor<B, 2>,
        delta_ts: Tensor<B, 1>,
        labels: Tensor<B, 1, Int>,
        reduce: Reduction,
    ) -> Tensor<B, 1> {
        // info!("t_historys: {}", &t_historys);
        // info!("r_historys: {}", &r_historys);
        let state = self.forward(t_historys, r_historys, None);
        let retention = self.power_forgetting_curve(delta_ts, state.stability);
        BCELoss::new().forward(retention, labels.float(), reduce)
    }
}

impl<B: AutodiffBackend> Model<B> {
    fn freeze_initial_stability(&self, mut grad: B::Gradients) -> B::Gradients {
        let grad_tensor = self.w.grad(&grad).unwrap();
        let updated_grad_tensor =
            grad_tensor.slice_assign([0..4], Tensor::zeros([4], &B::Device::default()));

        self.w.grad_remove(&mut grad);
        self.w.grad_replace(&mut grad, updated_grad_tensor);
        grad
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
    finished: bool,
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

    pub fn finished(&self) -> bool {
        self.finished
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
    pub const fn current(&self) -> usize {
        self.epoch.saturating_sub(1) * self.items_total + self.items_processed
    }

    pub const fn total(&self) -> usize {
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
        .map(|item| item.current())
        .fold((0u32, 0u32), |(sum, count), review| {
            (sum + (review.rating > 1) as u32, count + 1)
        });

    if total_reviews == 0 {
        return 0.0;
    }
    total_recall as f32 / total_reviews as f32
}

impl<B: Backend> FSRS<B> {
    /// Calculate appropriate parameters for the provided review history.
    pub fn compute_parameters(
        &self,
        items: Vec<FSRSItem>,
        pretrain_only: bool,
        progress: Option<Arc<Mutex<CombinedProgressState>>>,
    ) -> Result<Vec<f32>> {
        let finish_progress = || {
            if let Some(progress) = &progress {
                // The progress state at completion time may not indicate completion, because:
                // - If there were fewer than 512 entries, render_train() will have never been called
                // - One or more of the splits may have ignored later epochs, if accuracy went backwards
                // Because of this, we need a separate finished flag.
                progress.lock().unwrap().finished = true;
            }
        };

        let n_splits = 5;
        let average_recall = calculate_average_recall(&items);
        let (pre_trainset, trainsets, testset) = split_data(items, n_splits);
        let initial_stability = pretrain(pre_trainset, average_recall).map_err(|e| {
            finish_progress();
            e
        })?;
        if pretrain_only {
            finish_progress();
            let parameters = initial_stability
                .into_iter()
                .chain(DEFAULT_PARAMETERS[4..].iter().copied())
                .collect();
            return Ok(parameters);
        }
        let config = TrainingConfig::new(
            ModelConfig {
                freeze_stability: true,
                initial_stability: Some(initial_stability),
            },
            AdamConfig::new(),
        );

        let trainsets: Vec<Vec<FSRSItem>> = (0..n_splits)
            .into_par_iter()
            .map(|i| {
                trainsets
                    .par_iter()
                    .enumerate()
                    .filter(|&(j, _)| j != i)
                    .flat_map(|(_, trainset)| trainset.clone())
                    .collect()
            })
            .collect();

        if let Some(progress) = &progress {
            let mut progress_states = vec![ProgressState::default(); n_splits];
            for i in 0..n_splits {
                progress_states[i].epoch_total = config.num_epochs;
                progress_states[i].items_total = trainsets[i].len();
            }
            progress.lock().unwrap().splits = progress_states
        }

        let weight_sets: Result<Vec<Vec<f32>>> = (0..n_splits)
            .into_par_iter()
            .map(|i| {
                let model = train::<Autodiff<B>>(
                    trainsets[i].clone(),
                    testset.clone(),
                    &config,
                    self.device(),
                    progress.clone().map(|p| ProgressCollector::new(p, i)),
                );
                Ok(model
                    .map_err(|e| {
                        finish_progress();
                        e
                    })?
                    .w
                    .val()
                    .to_data()
                    .convert()
                    .value)
            })
            .collect();
        finish_progress();

        let weight_sets = weight_sets?;
        let average_parameters: Vec<f32> = weight_sets
            .iter()
            .fold(vec![0.0; weight_sets[0].len()], |sum, parameters| {
                sum.par_iter().zip(parameters).map(|(a, b)| a + b).collect()
            })
            .par_iter()
            .map(|&sum| sum / n_splits as f32)
            .collect();

        for weight in &average_parameters {
            if !weight.is_finite() {
                return Err(FSRSError::InvalidInput);
            }
        }

        Ok(average_parameters)
    }

    pub fn benchmark(&self, train_set: Vec<FSRSItem>, test_set: Vec<FSRSItem>) -> Vec<f32> {
        let average_recall = calculate_average_recall(&train_set.clone());
        let (pre_train_set, next_train_set) = train_set
            .into_iter()
            .partition(|item| item.reviews.len() == 2);
        let initial_stability = pretrain(pre_train_set, average_recall).unwrap();
        let config = TrainingConfig::new(
            ModelConfig {
                freeze_stability: true,
                initial_stability: Some(initial_stability),
            },
            AdamConfig::new(),
        );
        let model = train::<Autodiff<B>>(next_train_set, test_set, &config, self.device(), None);
        let parameters: Vec<f32> = model.unwrap().w.val().to_data().convert().value;
        parameters
    }
}

fn train<B: AutodiffBackend>(
    trainset: Vec<FSRSItem>,
    testset: Vec<FSRSItem>,
    config: &TrainingConfig,
    device: B::Device,
    progress: Option<ProgressCollector>,
) -> Result<Model<B>> {
    B::seed(config.seed);

    // Training data
    let iterations = (trainset.len() / config.batch_size + 1) * config.num_epochs;
    let batcher_train = FSRSBatcher::<B>::new(device.clone());
    let dataloader_train = BatchShuffledDataLoaderBuilder::new(batcher_train)
        .batch_size(config.batch_size)
        .build(FSRSDataset::from(trainset), config.batch_size, config.seed);

    let batcher_valid = FSRSBatcher::new(device.clone());
    let dataloader_valid = DataLoaderBuilder::new(batcher_valid)
        .batch_size(config.batch_size)
        .build(FSRSDataset::from(testset.clone()));

    let mut lr_scheduler = CosineAnnealingLR::init(iterations as f64, config.learning_rate);
    let interrupter = TrainingInterrupter::new();
    let mut renderer: Box<dyn MetricsRenderer> = match progress {
        Some(mut progress) => {
            progress.interrupter = interrupter.clone();
            Box::new(progress)
        }
        None => Box::new(NoProgress {}),
    };

    let mut model: Model<B> = config.model.init();
    let mut optim = config.optimizer.init::<B, Model<B>>();

    let mut best_loss = std::f64::INFINITY;
    let mut best_model = model.clone();
    for epoch in 1..=config.num_epochs {
        let mut iterator = dataloader_train.iter();
        let mut iteration = 0;
        while let Some(item) = iterator.next() {
            iteration += 1;
            let lr = LrScheduler::<B>::step(&mut lr_scheduler);
            let progress = iterator.progress();
            let loss = model.forward_classification(
                item.t_historys,
                item.r_historys,
                item.delta_ts,
                item.labels,
                Reduction::Mean,
            );
            let mut gradients = loss.backward();
            if model.config.freeze_stability {
                gradients = model.freeze_initial_stability(gradients);
            }
            let grads = GradientsParams::from_grads(gradients, &model);
            model = optim.step(lr, model, grads);
            model.w = Param::from(weight_clipper(model.w.val()));
            // info!("epoch: {:?} iteration: {:?} lr: {:?}", epoch, iteration, lr);
            renderer.render_train(TrainingProgress {
                progress,
                epoch,
                epoch_total: config.num_epochs,
                iteration,
            });

            if interrupter.should_stop() {
                break;
            }
        }

        if interrupter.should_stop() {
            break;
        }

        let model_valid = model.valid();
        let mut loss_valid = 0.0;
        for batch in dataloader_valid.iter() {
            let loss = model_valid.forward_classification(
                batch.t_historys,
                batch.r_historys,
                batch.delta_ts,
                batch.labels,
                Reduction::Sum,
            );
            let loss = loss.into_data().convert::<f64>().value[0];
            loss_valid += loss;

            if interrupter.should_stop() {
                break;
            }
        }
        loss_valid /= testset.len() as f64;
        info!("epoch: {:?} loss: {:?}", epoch, loss_valid);
        if loss_valid < best_loss {
            best_loss = loss_valid;
            best_model = model.clone();
        }
    }

    info!("best_loss: {:?}", best_loss);

    if interrupter.should_stop() {
        return Err(FSRSError::Interrupted);
    }

    Ok(best_model)
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
    use std::fs::create_dir_all;
    use std::path::Path;
    use std::thread;
    use std::time::Duration;

    use super::*;
    use crate::convertor_tests::anki21_sample_file_converted_to_fsrs;
    use crate::pre_training::pretrain;
    use crate::test_helpers::NdArrayAutodiff;
    use burn::backend::ndarray::NdArrayDevice;
    use log::LevelFilter;

    #[test]
    fn test_calculate_average_recall() {
        let items = anki21_sample_file_converted_to_fsrs();
        let average_recall = calculate_average_recall(&items);
        assert_eq!(average_recall, 0.9435269);
    }

    #[test]
    fn training() {
        if std::env::var("SKIP_TRAINING").is_ok() {
            println!("Skipping test in CI");
            return;
        }

        let artifact_dir = std::env::var("BURN_LOG");

        if let Ok(artifact_dir) = artifact_dir {
            let _ = create_dir_all(&artifact_dir);
            let log_file = Path::new(&artifact_dir).join("training.log");
            fern::Dispatch::new()
                .format(|out, message, record| {
                    out.finish(format_args!(
                        "[{}][{}] {}",
                        record.target(),
                        record.level(),
                        message
                    ))
                })
                .level(LevelFilter::Info)
                .chain(fern::log_file(log_file).unwrap())
                .apply()
                .unwrap();
        }
        let n_splits = 5;
        let device = NdArrayDevice::Cpu;
        let items = anki21_sample_file_converted_to_fsrs();
        let (pre_trainset, trainsets, testset) = split_data(items.clone(), n_splits);
        let items = [pre_trainset.clone(), testset.clone()].concat();
        let average_recall = calculate_average_recall(&items);
        dbg!(average_recall);
        let initial_stability = pretrain(pre_trainset, average_recall).unwrap();
        dbg!(initial_stability);
        let config = TrainingConfig::new(
            ModelConfig {
                freeze_stability: true,
                initial_stability: Some(initial_stability),
            },
            AdamConfig::new(),
        );
        let progress = CombinedProgressState::new_shared();
        let progress2 = Some(progress.clone());
        thread::spawn(move || {
            let mut finished = false;
            while !finished {
                thread::sleep(Duration::from_millis(10));
                let guard = progress.lock().unwrap();
                finished = guard.finished();
                info!("progress: {}/{}", guard.current(), guard.total());
            }
        });

        if let Some(progress2) = &progress2 {
            progress2.lock().unwrap().splits = vec![ProgressState::default(); n_splits];
        }

        let parameters_sets: Vec<Vec<f32>> = (0..n_splits)
            .into_par_iter()
            .map(|i| {
                let trainset = trainsets
                    .par_iter()
                    .enumerate()
                    .filter(|&(j, _)| j != i)
                    .flat_map(|(_, trainset)| trainset.clone())
                    .collect();
                let model = train::<NdArrayAutodiff>(
                    trainset,
                    items.clone(),
                    &config,
                    device,
                    progress2.clone().map(|p| ProgressCollector::new(p, i)),
                );
                model.unwrap().w.val().to_data().convert().value
            })
            .collect();

        dbg!(&parameters_sets);

        let average_parameters: Vec<f32> = parameters_sets
            .iter()
            .fold(vec![0.0; parameters_sets[0].len()], |sum, parameters| {
                sum.par_iter().zip(parameters).map(|(a, b)| a + b).collect()
            })
            .par_iter()
            .map(|&sum| sum / n_splits as f32)
            .collect();
        dbg!(&average_parameters);

        let fsrs = FSRS::new(Some(&average_parameters)).unwrap();
        let metrics = fsrs.evaluate(items.clone(), |_| true).unwrap();
        dbg!(&metrics);
    }
}

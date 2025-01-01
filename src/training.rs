use crate::batch_shuffle::{BatchTensorDataset, ShuffleDataLoader};
use crate::cosine_annealing::CosineAnnealingLR;
use crate::dataset::{
    prepare_training_data, recency_weighted_fsrs_items, sort_items_by_review_length, FSRSDataset,
    FSRSItem,
};
use crate::error::Result;
use crate::model::{Model, ModelConfig};
use crate::parameter_clipper::parameter_clipper;
use crate::pre_training::{pretrain, smooth_and_fill};
use crate::{FSRSError, DEFAULT_PARAMETERS, FSRS};
use burn::backend::Autodiff;
use wasm_bindgen::prelude::*;

use burn::lr_scheduler::LrScheduler;
use burn::module::AutodiffModule;
use burn::nn::loss::Reduction;
use burn::optim::Optimizer;
use burn::optim::{AdamConfig, GradientsParams};
use burn::tensor::backend::Backend;
use burn::tensor::{Int, Tensor};
use burn::train::renderer::{MetricState, MetricsRenderer, TrainingProgress};
use burn::train::TrainingInterrupter;
use burn::{config::Config, tensor::backend::AutodiffBackend};
use core::marker::PhantomData;
use log::info;

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
        weights: Tensor<B, 1>,
        mean: Reduction,
    ) -> Tensor<B, 1> {
        let loss = (labels.clone() * retentions.clone().log()
            + (-labels + 1) * (-retentions + 1).log())
            * weights.clone();
        // info!("loss: {}", &loss);
        match mean {
            Reduction::Mean => loss.mean().neg(),
            Reduction::Sum => loss.sum().neg(),
            Reduction::Auto => (loss.sum() / weights.sum()).neg(),
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
        weights: Tensor<B, 1>,
        reduce: Reduction,
    ) -> Tensor<B, 1> {
        // info!("t_historys: {}", &t_historys);
        // info!("r_historys: {}", &r_historys);
        let state = self.forward(t_historys, r_historys, None);
        let retention = self.power_forgetting_curve(delta_ts, state.stability);
        BCELoss::new().forward(retention, labels.float(), weights, reduce)
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

    fn free_short_term_stability(&self, mut grad: B::Gradients) -> B::Gradients {
        let grad_tensor = self.w.grad(&grad).unwrap();
        let updated_grad_tensor =
            grad_tensor.slice_assign([17..19], Tensor::zeros([2], &B::Device::default()));

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

#[wasm_bindgen]
#[derive(Default, Debug)]
pub struct Progress {
    vec: Vec<u32>,
}

#[wasm_bindgen]
impl Progress {
    // The progress vec is length 2. Grep 2291AF52-BEE4-4D54-BAD0-6492DFE368D8
    pub fn new() -> Progress {
        Progress { vec: vec![0; 2] }
    }

    /// Memory will hold [items_processed, items_total]
    pub fn pointer(&self) -> *const u32 {
        self.vec.as_ptr()
    }
}

#[derive(Default)]
pub struct CombinedProgressState {
    pub want_abort: bool,
    pub splits: Vec<ProgressState>,
    finished: bool,
    pub progress: Option<Progress>,
}

impl CombinedProgressState {
    pub fn new_shared(progress: Option<Progress>) -> Arc<Mutex<Self>> {
        let r: Arc<Mutex<CombinedProgressState>> = Default::default();
        r.lock().unwrap().progress = progress;
        r
    }

    pub fn current(&self) -> usize {
        self.splits.iter().map(|s| s.current()).sum()
    }

    pub fn total(&self) -> usize {
        self.splits.iter().map(|s| s.total()).sum()
    }

    pub const fn finished(&self) -> bool {
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
        if info.progress.is_some() {
            // The progress vec is length 2. Grep 2291AF52-BEE4-4D54-BAD0-6492DFE368D8
            info.progress.as_mut().unwrap().vec[0] = info.current() as u32;
            if info.progress.as_mut().unwrap().vec[1] == 0 {
                info.progress.as_mut().unwrap().vec[1] = info.total() as u32;
            }
        }
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
    #[config(default = 2023)]
    pub seed: u64,
    #[config(default = 4e-2)]
    pub learning_rate: f64,
    #[config(default = 64)]
    pub max_seq_len: usize,
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
        train_set: Vec<FSRSItem>,
        progress: Option<Arc<Mutex<CombinedProgressState>>>,
        enable_short_term: bool,
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

        let average_recall = calculate_average_recall(&train_set);
        let (pre_train_set, mut train_set) = prepare_training_data(train_set);
        if train_set.len() < 8 {
            finish_progress();
            return Ok(DEFAULT_PARAMETERS.to_vec());
        }

        let (initial_stability, initial_rating_count) =
            pretrain(pre_train_set.clone(), average_recall).map_err(|e| {
                finish_progress();
                e
            })?;
        let pretrained_parameters: Vec<f32> = initial_stability
            .into_iter()
            .chain(DEFAULT_PARAMETERS[4..].iter().copied())
            .collect();
        if train_set.len() == pre_train_set.len() || train_set.len() < 64 {
            finish_progress();
            return Ok(pretrained_parameters);
        }
        let config = TrainingConfig::new(
            ModelConfig {
                freeze_initial_stability: !enable_short_term,
                initial_stability: Some(initial_stability),
                freeze_short_term_stability: !enable_short_term,
            },
            AdamConfig::new().with_epsilon(1e-8),
        );
        train_set.retain(|item| item.reviews.len() <= config.max_seq_len);

        if let Some(progress) = &progress {
            let progress_state = ProgressState {
                epoch_total: config.num_epochs,
                items_total: train_set.len(),
                epoch: 0,
                items_processed: 0,
            };
            progress.lock().unwrap().splits = vec![progress_state];
        }

        let model = train::<Autodiff<B>>(
            train_set.clone(),
            train_set,
            &config,
            self.device(),
            progress.clone().map(|p| ProgressCollector::new(p, 0)),
        );

        let optimized_parameters = model
            .map_err(|e| {
                finish_progress();
                e
            })?
            .w
            .val()
            .to_data()
            .convert()
            .value;

        finish_progress();

        if optimized_parameters
            .iter()
            .any(|parameter: &f32| parameter.is_infinite())
        {
            return Err(FSRSError::InvalidInput);
        }

        let mut optimized_initial_stability = optimized_parameters[0..4]
            .iter()
            .enumerate()
            .map(|(i, &val)| (i as u32 + 1, val))
            .collect();
        let clamped_stability =
            smooth_and_fill(&mut optimized_initial_stability, &initial_rating_count).unwrap();
        let optimized_parameters = clamped_stability
            .into_iter()
            .chain(optimized_parameters[4..].iter().copied())
            .collect();

        Ok(optimized_parameters)
    }

    pub fn benchmark(&self, mut train_set: Vec<FSRSItem>, enable_short_term: bool) -> Vec<f32> {
        let average_recall = calculate_average_recall(&train_set);
        let (pre_train_set, _next_train_set) = train_set
            .clone()
            .into_iter()
            .partition(|item| item.long_term_review_cnt() == 1);
        let initial_stability = pretrain(pre_train_set, average_recall).unwrap().0;
        let config = TrainingConfig::new(
            ModelConfig {
                freeze_initial_stability: !enable_short_term,
                initial_stability: Some(initial_stability),
                freeze_short_term_stability: !enable_short_term,
            },
            AdamConfig::new().with_epsilon(1e-8),
        );
        train_set.retain(|item| item.reviews.len() <= config.max_seq_len);
        let model =
            train::<Autodiff<B>>(train_set.clone(), train_set, &config, self.device(), None);
        let parameters: Vec<f32> = model.unwrap().w.val().to_data().convert().value;
        parameters
    }
}

fn train<B: AutodiffBackend>(
    train_set: Vec<FSRSItem>,
    test_set: Vec<FSRSItem>,
    config: &TrainingConfig,
    device: B::Device,
    progress: Option<ProgressCollector>,
) -> Result<Model<B>> {
    B::seed(config.seed);

    // Training data
    let iterations = (train_set.len() / config.batch_size + 1) * config.num_epochs;
    let batch_dataset = BatchTensorDataset::<B>::new(
        FSRSDataset::from(sort_items_by_review_length(recency_weighted_fsrs_items(
            train_set,
        ))),
        config.batch_size,
        device.clone(),
    );
    let dataloader_train = ShuffleDataLoader::new(batch_dataset, config.seed);

    let batch_dataset = BatchTensorDataset::<B::InnerBackend>::new(
        FSRSDataset::from(sort_items_by_review_length(recency_weighted_fsrs_items(
            test_set.clone(),
        ))),
        config.batch_size,
        device,
    );
    let dataloader_valid = ShuffleDataLoader::new(batch_dataset, config.seed);

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

    let mut best_loss = f64::INFINITY;
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
                item.weights,
                Reduction::Sum,
            );
            let mut gradients = loss.backward();
            if model.config.freeze_initial_stability {
                gradients = model.freeze_initial_stability(gradients);
            }
            if model.config.freeze_short_term_stability {
                gradients = model.free_short_term_stability(gradients);
            }
            let grads = GradientsParams::from_grads(gradients, &model);
            model = optim.step(lr, model, grads);
            model.w = parameter_clipper(model.w);
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
                batch.weights,
                Reduction::Sum,
            );
            let loss = loss.into_data().convert::<f64>().value[0];
            loss_valid += loss;

            if interrupter.should_stop() {
                break;
            }
        }
        loss_valid /= test_set.len() as f64;
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
    use crate::convertor_tests::data_from_csv;
    use crate::dataset::FSRSBatch;
    use burn::backend::NdArray;
    use log::LevelFilter;

    #[test]
    fn test_calculate_average_recall() {
        let items = anki21_sample_file_converted_to_fsrs();
        let average_recall = calculate_average_recall(&items);
        assert_eq!(average_recall, 0.9435269);
    }

    #[test]
    fn test_loss_and_grad() {
        use burn::backend::ndarray::NdArrayDevice;
        use burn::tensor::Data;

        let config = ModelConfig::default();
        let device = NdArrayDevice::Cpu;
        type B = Autodiff<NdArray<f32>>;
        let mut model: Model<B> = config.init();

        let item = FSRSBatch {
            t_historys: Tensor::from_floats(
                Data::from([
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                    [0.0, 1.0, 1.0, 3.0],
                    [1.0, 3.0, 3.0, 5.0],
                    [3.0, 6.0, 6.0, 12.0],
                ]),
                &device,
            ),
            r_historys: Tensor::from_floats(
                Data::from([
                    [1.0, 2.0, 3.0, 4.0],
                    [3.0, 4.0, 2.0, 4.0],
                    [1.0, 4.0, 4.0, 3.0],
                    [4.0, 3.0, 3.0, 3.0],
                    [3.0, 1.0, 3.0, 3.0],
                    [2.0, 3.0, 3.0, 4.0],
                ]),
                &device,
            ),
            delta_ts: Tensor::from_floats(Data::from([4.0, 11.0, 12.0, 23.0]), &device),
            labels: Tensor::from_ints(Data::from([1, 1, 1, 0]), &device),
            weights: Tensor::from_floats(Data::from([1.0, 1.0, 1.0, 1.0]), &device),
        };

        let loss = model.forward_classification(
            item.t_historys,
            item.r_historys,
            item.delta_ts,
            item.labels,
            item.weights,
            Reduction::Sum,
        );

        assert_eq!(
            loss.clone().into_data().convert::<f32>().value[0],
            4.4467363
        );
        let gradients = loss.backward();

        let w_grad = model.w.grad(&gradients).unwrap();

        Data::from([
            -0.05832, -0.00682, -0.00255, 0.010539, -0.05128, 1.364291, 0.083658, -0.95023,
            0.534472, -2.89288, 0.514163, -0.01306, 0.041905, -0.11830, -0.00092, -0.14452,
            0.202374, 0.214104, 0.032307,
        ])
        .assert_approx_eq(&w_grad.clone().into_data(), 5);

        let config =
            TrainingConfig::new(ModelConfig::default(), AdamConfig::new().with_epsilon(1e-8));
        let mut optim = config.optimizer.init::<B, Model<B>>();
        let lr = 0.04;
        let grads = GradientsParams::from_grads(gradients, &model);
        model = optim.step(lr, model, grads);
        model.w = parameter_clipper(model.w);
        assert_eq!(
            model.w.val().to_data(),
            Data::from([
                0.44255, 1.22385, 3.2129998, 15.65105, 7.2349, 0.4945, 1.4204, 0.0446, 1.5057501,
                0.1592, 0.97925, 1.9794999, 0.07000001, 0.33605, 2.3097994, 0.2715, 2.9498,
                0.47655, 0.62210006
            ])
        );

        let item = FSRSBatch {
            t_historys: Tensor::from_floats(
                Data::from([
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                    [0.0, 1.0, 1.0, 3.0],
                    [1.0, 3.0, 3.0, 5.0],
                    [3.0, 6.0, 6.0, 12.0],
                ]),
                &device,
            ),
            r_historys: Tensor::from_floats(
                Data::from([
                    [1.0, 2.0, 3.0, 4.0],
                    [3.0, 4.0, 2.0, 4.0],
                    [1.0, 4.0, 4.0, 3.0],
                    [4.0, 3.0, 3.0, 3.0],
                    [3.0, 1.0, 3.0, 3.0],
                    [2.0, 3.0, 3.0, 4.0],
                ]),
                &device,
            ),
            delta_ts: Tensor::from_floats(Data::from([4.0, 11.0, 12.0, 23.0]), &device),
            labels: Tensor::from_ints(Data::from([1, 1, 1, 0]), &device),
            weights: Tensor::from_floats(Data::from([1.0, 1.0, 1.0, 1.0]), &device),
        };

        let loss = model.forward_classification(
            item.t_historys,
            item.r_historys,
            item.delta_ts,
            item.labels,
            item.weights,
            Reduction::Sum,
        );
        assert_eq!(loss.clone().into_data().convert::<f32>().value[0], 4.176347);
        let gradients = loss.backward();
        let w_grad = model.w.grad(&gradients).unwrap();
        Data::from([
            -0.0401341,
            -0.0061790533,
            -0.00288913,
            0.01216853,
            -0.05624995,
            1.147413,
            0.068084724,
            -0.6906936,
            0.48760873,
            -2.5428302,
            0.49044546,
            -0.011574259,
            0.037729632,
            -0.09633919,
            -0.0009513022,
            -0.12789416,
            0.19088513,
            0.2574597,
            0.049311582,
        ])
        .assert_approx_eq(&w_grad.clone().into_data(), 5);
        let grads = GradientsParams::from_grads(gradients, &model);
        model = optim.step(lr, model, grads);
        model.w = parameter_clipper(model.w);
        assert_eq!(
            model.w.val().to_data(),
            Data::from([
                0.48150504,
                1.2636971,
                3.2530522,
                15.611003,
                7.2749534,
                0.45482785,
                1.3808222,
                0.083782874,
                1.4658877,
                0.19898315,
                0.9393105,
                2.0193,
                0.030164223,
                0.37562984,
                2.3498251,
                0.3112984,
                2.909878,
                0.43652722,
                0.5825156
            ])
        );
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
        for items in [anki21_sample_file_converted_to_fsrs(), data_from_csv()] {
            for enable_short_term in [true, false] {
                let progress = CombinedProgressState::new_shared();
                let progress2 = Some(progress.clone());
                thread::spawn(move || {
                    let mut finished = false;
                    while !finished {
                        thread::sleep(Duration::from_millis(500));
                        let guard = progress.lock().unwrap();
                        finished = guard.finished();
                        println!("progress: {}/{}", guard.current(), guard.total());
                    }
                });

                let fsrs = FSRS::new(Some(&[])).unwrap();
                let parameters = fsrs
                    .compute_parameters(items.clone(), progress2, enable_short_term)
                    .unwrap();
                dbg!(&parameters);

                // evaluate
                let model = FSRS::new(Some(&parameters)).unwrap();
                let metrics = model.evaluate(items.clone(), |_| true).unwrap();
                dbg!(&metrics);
            }
        }
    }
}

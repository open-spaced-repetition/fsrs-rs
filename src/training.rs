use crate::batch_shuffle::{BatchTensorDataset, ShuffleDataLoader};
use crate::cosine_annealing::CosineAnnealingLR;
use crate::dataset::{
    FSRSDataset, FSRSItem, WeightedFSRSItem, prepare_training_data, recency_weighted_fsrs_items,
};
use crate::error::Result;
use crate::model::{Model, ModelConfig};
use crate::parameter_clipper::parameter_clipper;
use crate::parameter_initialization::{initialize_stability_parameters, smooth_and_fill};
use crate::{DEFAULT_PARAMETERS, FSRSError};
use burn::backend::Autodiff;
use burn::backend::ndarray::NdArray;
use burn::lr_scheduler::LrScheduler;
use burn::module::AutodiffModule;
use burn::nn::loss::Reduction;
use burn::optim::Optimizer;
use burn::optim::{AdamConfig, GradientsParams};
use burn::tensor::backend::Backend;
use burn::tensor::cast::ToElement;
use burn::tensor::{Int, Tensor};
use burn::train::TrainingInterrupter;
use burn::train::renderer::{MetricState, MetricsRenderer, TrainingProgress};
use burn::{config::Config, tensor::backend::AutodiffBackend};
use core::marker::PhantomData;
use log::info;

use std::sync::{Arc, Mutex};

type B = NdArray<f32>;

static PARAMS_STDDEV: [f32; 21] = [
    6.43, 9.66, 17.58, 27.85, 0.57, 0.28, 0.6, 0.12, 0.39, 0.18, 0.33, 0.3, 0.09, 0.16, 0.57, 0.25,
    1.03, 0.31, 0.32, 0.14, 0.27,
];

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
        retrievability: Tensor<B, 1>,
        labels: Tensor<B, 1>,
        weights: Tensor<B, 1>,
        mean: Reduction,
    ) -> Tensor<B, 1> {
        let loss = (labels.clone() * retrievability.clone().log()
            + (-labels + 1) * (-retrievability + 1).log())
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
        let retrievability = self.power_forgetting_curve(delta_ts, state.stability);
        BCELoss::new().forward(retrievability, labels.float(), weights, reduce)
    }

    pub(crate) fn l2_regularization(
        &self,
        init_w: Tensor<B, 1>,
        params_stddev: Tensor<B, 1>,
        batch_size: usize,
        total_size: usize,
        gamma: f64,
    ) -> Tensor<B, 1> {
        (self.w.val() - init_w)
            .powi_scalar(2)
            .div(params_stddev.powi_scalar(2))
            .sum()
            .mul_scalar(gamma * batch_size as f64 / total_size as f64)
    }
}

impl<B: AutodiffBackend> Model<B> {
    fn freeze_initial_stability(&self, mut grad: B::Gradients) -> B::Gradients {
        let grad_tensor = self.w.grad(&grad).unwrap();
        let device = grad_tensor.device();
        let updated_grad_tensor = grad_tensor.slice_assign([0..4], Tensor::zeros([4], &device));

        self.w.grad_remove(&mut grad);
        self.w.grad_replace(&mut grad, updated_grad_tensor);
        grad
    }

    fn freeze_short_term_stability(&self, mut grad: B::Gradients) -> B::Gradients {
        let grad_tensor = self.w.grad(&grad).unwrap();
        let device = grad_tensor.device();
        let updated_grad_tensor = grad_tensor.slice_assign([17..20], Tensor::zeros([3], &device));

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

#[derive(Debug, Default)]
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
    #[config(default = 1024)]
    pub max_seq_len: usize,
    #[config(default = 1.0)]
    pub gamma: f64,
}

pub(crate) fn calculate_average_recall(items: &[FSRSItem]) -> f32 {
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

/// Input parameters for computing FSRS parameters
#[derive(Clone, Debug)]
pub struct ComputeParametersInput {
    /// The training set containing review history
    pub train_set: Vec<FSRSItem>,
    /// Optional progress tracking
    pub progress: Option<Arc<Mutex<CombinedProgressState>>>,
    /// Whether to enable short-term memory parameters
    pub enable_short_term: bool,
    /// Number of relearning steps
    pub num_relearning_steps: Option<usize>,
}

impl Default for ComputeParametersInput {
    fn default() -> Self {
        Self {
            train_set: Vec::new(),
            progress: None,
            enable_short_term: true,
            num_relearning_steps: None,
        }
    }
}
/// Computes optimized parameters for the FSRS model based on training data.
///
/// This function trains the model on the provided dataset and returns optimized parameters.
///
/// # Arguments
/// * `input` - Input parameters including the training dataset and configuration
///
/// # Returns
/// A `Result<Vec<f32>>` containing the optimized parameters
pub fn compute_parameters(
    ComputeParametersInput {
        train_set,
        progress,
        enable_short_term,
        num_relearning_steps,
        ..
    }: ComputeParametersInput,
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

    let (dataset_for_initialization, train_set) = prepare_training_data(train_set);
    let average_recall = calculate_average_recall(&train_set);
    if train_set.len() < 8 {
        finish_progress();
        return Ok(DEFAULT_PARAMETERS.to_vec());
    }

    let (initial_stability, initial_rating_count) =
        initialize_stability_parameters(dataset_for_initialization.clone(), average_recall)
            .inspect_err(|_e| {
                finish_progress();
            })?;
    let initialized_parameters: Vec<f32> = initial_stability
        .into_iter()
        .chain(DEFAULT_PARAMETERS[4..].iter().copied())
        .collect();
    if train_set.len() == dataset_for_initialization.len() || train_set.len() < 64 {
        finish_progress();
        return Ok(initialized_parameters);
    }
    let config = TrainingConfig::new(
        ModelConfig {
            freeze_initial_stability: !enable_short_term,
            initial_stability: Some(initial_stability),
            freeze_short_term_stability: !enable_short_term,
            num_relearning_steps: num_relearning_steps.unwrap_or(1),
        },
        AdamConfig::new().with_epsilon(1e-8),
    );
    let mut weighted_train_set = recency_weighted_fsrs_items(train_set);
    weighted_train_set.retain(|item| item.item.reviews.len() <= config.max_seq_len);

    if let Some(progress) = &progress {
        let progress_state = ProgressState {
            epoch_total: config.num_epochs,
            items_total: weighted_train_set.len(),
            epoch: 0,
            items_processed: 0,
        };
        progress.lock().unwrap().splits = vec![progress_state];
    }
    let model = train::<Autodiff<B>>(
        weighted_train_set.clone(),
        weighted_train_set,
        &config,
        progress.clone().map(|p| ProgressCollector::new(p, 0)),
    );

    let optimized_parameters = model
        .inspect_err(|_e| {
            finish_progress();
        })?
        .w
        .val()
        .to_data()
        .to_vec()
        .unwrap();

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

pub fn benchmark(
    ComputeParametersInput {
        train_set,
        enable_short_term,
        num_relearning_steps,
        ..
    }: ComputeParametersInput,
) -> Vec<f32> {
    let average_recall = calculate_average_recall(&train_set);
    let (dataset_for_initialization, _next_train_set) = train_set
        .clone()
        .into_iter()
        .partition(|item| item.long_term_review_cnt() == 1);
    let initial_stability =
        initialize_stability_parameters(dataset_for_initialization, average_recall)
            .unwrap()
            .0;
    let mut config = TrainingConfig::new(
        ModelConfig {
            freeze_initial_stability: !enable_short_term,
            initial_stability: Some(initial_stability),
            freeze_short_term_stability: !enable_short_term,
            num_relearning_steps: num_relearning_steps.unwrap_or(1),
        },
        AdamConfig::new().with_epsilon(1e-8),
    );
    // save RAM and speed up training
    config.max_seq_len = 64;
    let mut weighted_train_set = recency_weighted_fsrs_items(train_set);
    weighted_train_set.retain(|item| item.item.reviews.len() <= config.max_seq_len);
    let model = train::<Autodiff<B>>(
        weighted_train_set.clone(),
        weighted_train_set,
        &config,
        None,
    );
    let parameters: Vec<f32> = model.unwrap().w.val().to_data().to_vec::<f32>().unwrap();
    parameters
}

fn train<B: AutodiffBackend>(
    train_set: Vec<WeightedFSRSItem>,
    test_set: Vec<WeightedFSRSItem>,
    config: &TrainingConfig,
    progress: Option<ProgressCollector>,
) -> Result<Model<B>> {
    B::seed(config.seed);

    // Training data
    let total_size = train_set.len();
    let iterations = (total_size / config.batch_size + 1) * config.num_epochs;
    let batch_dataset =
        BatchTensorDataset::<B>::new(FSRSDataset::from(train_set), config.batch_size);
    let dataloader_train = ShuffleDataLoader::new(batch_dataset, config.seed);

    let batch_dataset = BatchTensorDataset::<B::InnerBackend>::new(
        FSRSDataset::from(test_set.clone()),
        config.batch_size,
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
    let init_w = model.w.val();
    let params_stddev = Tensor::from_floats(PARAMS_STDDEV, &model.w.device());
    let mut optim = config.optimizer.init::<B, Model<B>>();

    let mut best_loss = f64::INFINITY;
    let mut best_model = model.clone();
    for epoch in 1..=config.num_epochs {
        let mut iterator = dataloader_train.iter();
        let mut iteration = 0;
        while let Some(item) = iterator.next() {
            iteration += 1;
            let real_batch_size = item.delta_ts.shape().dims[0];
            let lr = LrScheduler::step(&mut lr_scheduler);
            let progress = iterator.progress();
            let penalty = model.l2_regularization(
                init_w.clone(),
                params_stddev.clone(),
                real_batch_size,
                total_size,
                config.gamma,
            );
            let loss = model.forward_classification(
                item.t_historys,
                item.r_historys,
                item.delta_ts,
                item.labels,
                item.weights,
                Reduction::Sum,
            );
            let mut gradients = (loss + penalty).backward();
            if config.model.freeze_initial_stability {
                gradients = model.freeze_initial_stability(gradients);
            }
            if config.model.freeze_short_term_stability {
                gradients = model.freeze_short_term_stability(gradients);
            }
            let grads = GradientsParams::from_grads(gradients, &model);
            model = optim.step(lr, model, grads);
            model.w = parameter_clipper(
                model.w,
                config.model.num_relearning_steps,
                !config.model.freeze_short_term_stability,
            );
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
            let real_batch_size = batch.delta_ts.shape().dims[0];
            let penalty = model_valid.l2_regularization(
                init_w.valid(),
                params_stddev.valid(),
                real_batch_size,
                total_size,
                config.gamma,
            );
            let loss = model_valid.forward_classification(
                batch.t_historys,
                batch.r_historys,
                batch.delta_ts,
                batch.labels,
                batch.weights,
                Reduction::Sum,
            );
            let loss = loss.into_scalar().to_f64();
            let penalty = penalty.into_scalar().to_f64();
            loss_valid += loss + penalty;

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
    use crate::model::FSRS;
    use crate::test_helpers::TestHelper;
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
        use burn::tensor::TensorData;

        let config = ModelConfig::default();
        let device = NdArrayDevice::Cpu;
        type B = Autodiff<NdArray<f32>>;
        let mut model: Model<B> = config.init();
        let init_w = model.w.val();
        let params_stddev = Tensor::from_floats(PARAMS_STDDEV, &device);

        let item = FSRSBatch {
            t_historys: Tensor::from_floats(
                TensorData::from([
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
                TensorData::from([
                    [1.0, 2.0, 3.0, 4.0],
                    [3.0, 4.0, 2.0, 4.0],
                    [1.0, 4.0, 4.0, 3.0],
                    [4.0, 3.0, 3.0, 3.0],
                    [3.0, 1.0, 3.0, 3.0],
                    [2.0, 3.0, 3.0, 4.0],
                ]),
                &device,
            ),
            delta_ts: Tensor::from_floats([4.0, 11.0, 12.0, 23.0], &device),
            labels: Tensor::from_ints([1, 1, 1, 0], &device),
            weights: Tensor::from_floats([1.0, 1.0, 1.0, 1.0], &device),
        };

        let loss = model.forward_classification(
            item.t_historys,
            item.r_historys,
            item.delta_ts,
            item.labels,
            item.weights,
            Reduction::Sum,
        );

        assert_eq!(loss.clone().into_scalar().to_f32(), 4.0466027);
        let gradients = loss.backward();

        let w_grad = model.w.grad(&gradients).unwrap();

        w_grad.to_data().to_vec::<f32>().unwrap().assert_approx_eq([
            -0.095688485,
            -0.0051607806,
            -0.0012249565,
            0.007462064,
            0.03650761,
            -0.082112335,
            0.0593964,
            -2.1474836,
            0.57626534,
            -2.8751316,
            0.7154875,
            -0.028993709,
            0.0099172965,
            -0.2189217,
            -0.0017800558,
            -0.089381434,
            0.299141,
            0.068104014,
            -0.011605468,
            -0.25398168,
            0.27700496,
        ]);

        let config =
            TrainingConfig::new(ModelConfig::default(), AdamConfig::new().with_epsilon(1e-8));
        let mut optim = config.optimizer.init::<B, Model<B>>();
        let lr = 0.04;
        let grads = GradientsParams::from_grads(gradients, &model);
        model = optim.step(lr, model, grads);
        model.w = parameter_clipper(
            model.w,
            config.model.num_relearning_steps,
            !config.model.freeze_short_term_stability,
        );
        model
            .w
            .val()
            .to_data()
            .to_vec::<f32>()
            .unwrap()
            .assert_approx_eq([
                0.252,
                1.3331,
                2.3464994,
                8.2556,
                6.3733,
                0.87340003,
                2.9794,
                0.040999997,
                1.8322,
                0.20660001,
                0.756,
                1.5235,
                0.021400042,
                0.3029,
                1.6882998,
                0.64140004,
                1.8329,
                0.5025,
                0.13119997,
                0.1058,
                0.1142,
            ]);

        let penalty =
            model.l2_regularization(init_w.clone(), params_stddev.clone(), 512, 1000, 2.0);
        assert_eq!(penalty.clone().into_scalar().to_f32(), 0.67711145);

        let gradients = penalty.backward();
        let w_grad = model.w.grad(&gradients).unwrap();
        w_grad.to_data().to_vec::<f32>().unwrap().assert_approx_eq([
            0.0019813816,
            0.00087788026,
            0.00026506148,
            -0.000105618295,
            -0.25213888,
            1.0448985,
            -0.22755535,
            5.688889,
            -0.5385926,
            2.5283954,
            -0.75225013,
            0.9102214,
            -10.113569,
            3.1999993,
            0.2521374,
            1.3107208,
            -0.07721739,
            -0.85244584,
            0.79999936,
            4.1795917,
            -1.1237311,
        ]);

        let item = FSRSBatch {
            t_historys: Tensor::from_floats(
                TensorData::from([
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
                TensorData::from([
                    [1.0, 2.0, 3.0, 4.0],
                    [3.0, 4.0, 2.0, 4.0],
                    [1.0, 4.0, 4.0, 3.0],
                    [4.0, 3.0, 3.0, 3.0],
                    [3.0, 1.0, 3.0, 3.0],
                    [2.0, 3.0, 3.0, 4.0],
                ]),
                &device,
            ),
            delta_ts: Tensor::from_floats([4.0, 11.0, 12.0, 23.0], &device),
            labels: Tensor::from_ints([1, 1, 1, 0], &device),
            weights: Tensor::from_floats([1.0, 1.0, 1.0, 1.0], &device),
        };

        let loss = model.forward_classification(
            item.t_historys,
            item.r_historys,
            item.delta_ts,
            item.labels,
            item.weights,
            Reduction::Sum,
        );
        assert_eq!(loss.clone().into_scalar().to_f32(), 3.767796);
        let gradients = loss.backward();
        let w_grad = model.w.grad(&gradients).unwrap();
        w_grad
            .clone()
            .into_data()
            .to_vec::<f32>()
            .unwrap()
            .assert_approx_eq([
                -0.040530164,
                -0.0041278866,
                -0.0010157757,
                0.007239434,
                0.009321215,
                -0.120117955,
                0.039143264,
                -0.8628009,
                0.5794302,
                -2.5713828,
                0.7669307,
                -0.024242667,
                0.0,
                -0.16912507,
                -0.0017008218,
                -0.061857328,
                0.28093633,
                0.064058185,
                0.0063592787,
                -0.1903223,
                0.6257775,
            ]);
        let grads = GradientsParams::from_grads(gradients, &model);
        model = optim.step(lr, model, grads);
        model.w = parameter_clipper(
            model.w,
            config.model.num_relearning_steps,
            !config.model.freeze_short_term_stability,
        );
        model
            .w
            .val()
            .to_data()
            .to_vec::<f32>()
            .unwrap()
            .assert_approx_eq([
                0.2882918, 1.3726242, 2.3861322, 8.215636, 6.339965, 0.9130969, 2.940639,
                0.07696985, 1.7921946, 0.2464217, 0.71595186, 1.5631561, 0.001, 0.34230903,
                1.7282416, 0.68038, 1.7929853, 0.46258268, 0.14039303, 0.14509967, 0.1,
            ]);
    }

    #[test]
    fn test_training() {
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

                let parameters = compute_parameters(ComputeParametersInput {
                    train_set: items.clone(),
                    progress: progress2,
                    enable_short_term,
                    num_relearning_steps: None,
                })
                .unwrap();
                dbg!(&parameters);

                // evaluate
                let model = FSRS::new(&parameters).unwrap();
                let metrics = model.evaluate(items.clone(), |_| true).unwrap();
                dbg!(&metrics);
            }
        }
    }
}

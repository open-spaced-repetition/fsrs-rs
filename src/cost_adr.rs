use crate::error::{FSRSError, Result};
use crate::inference::{ItemState, MemoryState, Parameters};
use crate::model::FSRS;
use crate::simulation::{D_MAX, D_MIN, S_MAX, S_MIN, simulate_with_cost_adr_policy_for_evaluation};
use crate::training::{CombinedProgressState, ProgressState};
use crate::{SimulationResult, SimulatorConfig, simulate};
use burn::tensor::backend::Backend;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rand_distr::StandardNormal;
use rayon::iter::{
    IndexedParallelIterator, IntoParallelIterator, IntoParallelRefIterator, ParallelIterator,
};
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

const COST_ADR_PARAMETER_COUNT: usize = 15;
pub const COST_ADR_POLICY_VERSION: u32 = 1;
const COST_ADR_DEFAULT_COST_WEIGHTS: [f32; 16] = [
    0.0, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 48.0, 64.0, 96.0, 128.0, 192.0, 256.0, 384.0, 512.0,
    1024.0,
];
const COST_ADR_DEFAULT_BASELINE_RETENTIONS: [f32; 16] = [
    0.50, 0.53, 0.56, 0.59, 0.62, 0.65, 0.68, 0.71, 0.74, 0.77, 0.80, 0.83, 0.86, 0.89, 0.92, 0.95,
];
const COST_ADR_DEFAULT_SEED: u64 = 42;
const COST_ADR_DEFAULT_INITIAL_COEFFICIENTS: [f32; COST_ADR_PARAMETER_COUNT] = [
    -0.202, 9.14, -0.0978, 0.226, -5.31, -7.44, 24.1, -0.375, 1.81, -22.9, -5.82, 22.3, 1.72,
    -1.99, -19.4,
];

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct CostAdrBounds {
    pub s_min: f32,
    pub s_max: f32,
    pub d_min: f32,
    pub d_max: f32,
}

impl Default for CostAdrBounds {
    fn default() -> Self {
        Self {
            s_min: S_MIN,
            s_max: S_MAX,
            d_min: D_MIN,
            d_max: D_MAX,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CostAdrPolicy {
    pub policy_version: u32,
    pub coefficients: Vec<f32>,
    pub cost_weight_min: f32,
    pub cost_weight_max: f32,
    pub retention_min: f32,
    pub retention_max: f32,
    pub max_interval_days: Option<f32>,
    pub bounds: CostAdrBounds,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct CostAdrNextStates {
    pub again: CostAdrItemState,
    pub hard: CostAdrItemState,
    pub good: CostAdrItemState,
    pub easy: CostAdrItemState,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct CostAdrItemState {
    pub memory: MemoryState,
    pub interval: f32,
    pub desired_retention: f32,
}

impl CostAdrPolicy {
    pub fn train_single_user(
        config: &SimulatorConfig,
        parameters: &Parameters,
        training_config: &CostAdrTrainingConfig,
    ) -> Result<CostAdrTrainingResult> {
        train_cost_adr_single_user(config, parameters, training_config)
    }

    pub fn new(coefficients: Vec<f32>) -> Result<Self> {
        Self::new_with_settings(coefficients, 0.0, 1024.0, 0.30, 0.995, None)
    }

    pub fn new_with_settings(
        coefficients: Vec<f32>,
        cost_weight_min: f32,
        cost_weight_max: f32,
        retention_min: f32,
        retention_max: f32,
        max_interval_days: Option<f32>,
    ) -> Result<Self> {
        let policy = Self {
            policy_version: COST_ADR_POLICY_VERSION,
            coefficients,
            cost_weight_min,
            cost_weight_max,
            retention_min,
            retention_max,
            max_interval_days,
            bounds: CostAdrBounds::default(),
        };
        policy.validate()?;
        Ok(policy)
    }

    #[cfg(test)]
    fn default_initial() -> Self {
        Self::new(COST_ADR_DEFAULT_INITIAL_COEFFICIENTS.to_vec())
            .expect("built-in Cost ADR policy is valid")
    }

    #[cfg(test)]
    fn constant_retention(desired_retention: f32) -> Result<Self> {
        let retention_min = 0.30;
        let retention_max = 0.995;
        if !(retention_min < desired_retention && desired_retention < retention_max) {
            return Err(FSRSError::InvalidInput);
        }
        let ratio = ((desired_retention - retention_min) / (retention_max - retention_min))
            .clamp(1e-9, 1.0 - 1e-9);
        let mut coefficients = vec![0.0; COST_ADR_PARAMETER_COUNT];
        coefficients[0] = (ratio / (1.0 - ratio)).ln();
        coefficients[5] = -40.0;
        coefficients[10] = -40.0;
        Self::new_with_settings(
            coefficients,
            0.0,
            1024.0,
            retention_min,
            retention_max,
            None,
        )
    }

    pub fn validate(&self) -> Result<()> {
        if self.policy_version != COST_ADR_POLICY_VERSION
            || self.coefficients.len() != COST_ADR_PARAMETER_COUNT
            || !self.cost_weight_min.is_finite()
            || !self.cost_weight_max.is_finite()
            || !self.retention_min.is_finite()
            || !self.retention_max.is_finite()
            || self.cost_weight_min < 0.0
            || self.cost_weight_max <= self.cost_weight_min
            || !(0.0 < self.retention_min && self.retention_min < self.retention_max)
            || self.retention_max >= 1.0
            || self.coefficients.iter().any(|value| !value.is_finite())
            || self
                .max_interval_days
                .is_some_and(|value| !value.is_finite() || value < 1.0)
        {
            return Err(FSRSError::InvalidInput);
        }
        if !self.bounds.s_min.is_finite()
            || !self.bounds.s_max.is_finite()
            || !self.bounds.d_min.is_finite()
            || !self.bounds.d_max.is_finite()
            || self.bounds.s_min <= 0.0
            || self.bounds.s_max <= self.bounds.s_min
            || self.bounds.d_max <= self.bounds.d_min
        {
            return Err(FSRSError::InvalidInput);
        }
        Ok(())
    }

    pub fn evaluate(
        &self,
        config: &SimulatorConfig,
        parameters: &Parameters,
        evaluation_config: &CostAdrEvaluationConfig,
    ) -> Result<CostAdrEvaluationResult> {
        evaluate_cost_adr_policy(config, parameters, self, evaluation_config)
    }

    /// Return the memory states, cost-conditioned desired retentions, and intervals for each
    /// answer button.
    pub fn next_states<B: Backend>(
        &self,
        fsrs: &FSRS<B>,
        current_memory_state: Option<MemoryState>,
        goal_cost_weight: f32,
        days_elapsed: f32,
    ) -> Result<CostAdrNextStates> {
        self.validate()?;
        if !goal_cost_weight.is_finite()
            || goal_cost_weight < 0.0
            || !days_elapsed.is_finite()
            || days_elapsed < 0.0
        {
            return Err(FSRSError::InvalidInput);
        }

        let states = fsrs.next_states_with_elapsed_days(
            current_memory_state,
            self.retention_max,
            days_elapsed,
        )?;
        Ok(CostAdrNextStates {
            again: self.cost_adr_item_state(fsrs, states.again, goal_cost_weight, 1)?,
            hard: self.cost_adr_item_state(fsrs, states.hard, goal_cost_weight, 2)?,
            good: self.cost_adr_item_state(fsrs, states.good, goal_cost_weight, 3)?,
            easy: self.cost_adr_item_state(fsrs, states.easy, goal_cost_weight, 4)?,
        })
    }

    pub fn evaluate_retention(&self, stability: f32, difficulty: f32, cost_weight: f32) -> f32 {
        let phi = self.state_features(stability, difficulty);
        let z = self.normalized_cost_weight(cost_weight);
        let base = dot(&self.coefficients[0..5], &phi);
        let z_effect = softplus(dot(&self.coefficients[5..10], &phi)) * z;
        let z2_effect = softplus(dot(&self.coefficients[10..15], &phi)) * z * z;
        self.retention_min
            + (self.retention_max - self.retention_min) * sigmoid(base - z_effect - z2_effect)
    }

    pub fn retention_grid(
        &self,
        stabilities: &[f32],
        difficulties: &[f32],
        cost_weights: &[f32],
    ) -> Result<Vec<CostAdrPlotPoint>> {
        self.validate()?;
        if stabilities.is_empty()
            || difficulties.is_empty()
            || cost_weights.is_empty()
            || stabilities.iter().any(|value| !value.is_finite())
            || difficulties.iter().any(|value| !value.is_finite())
            || cost_weights
                .iter()
                .any(|value| !value.is_finite() || *value < 0.0)
        {
            return Err(FSRSError::InvalidInput);
        }

        let mut points =
            Vec::with_capacity(stabilities.len() * difficulties.len() * cost_weights.len());
        for &stability in stabilities {
            for &difficulty in difficulties {
                for &cost_weight in cost_weights {
                    points.push(CostAdrPlotPoint {
                        stability,
                        difficulty,
                        cost_weight,
                        desired_retention: self.evaluate_retention(
                            stability,
                            difficulty,
                            cost_weight,
                        ),
                    });
                }
            }
        }
        Ok(points)
    }

    fn cost_adr_item_state<B: Backend>(
        &self,
        fsrs: &FSRS<B>,
        item_state: ItemState,
        goal_cost_weight: f32,
        rating: u32,
    ) -> Result<CostAdrItemState> {
        let desired_retention = self.evaluate_retention(
            item_state.memory.stability,
            item_state.memory.difficulty,
            goal_cost_weight,
        );
        let mut interval =
            fsrs.next_interval(Some(item_state.memory.stability), desired_retention, rating);
        if let Some(max_interval_days) = self.max_interval_days {
            interval = interval.clamp(1.0, max_interval_days);
        }
        if !interval.is_finite() {
            return Err(FSRSError::InvalidInput);
        }

        Ok(CostAdrItemState {
            memory: item_state.memory,
            interval,
            desired_retention,
        })
    }

    fn state_features(&self, stability: f32, difficulty: f32) -> [f32; 5] {
        let stability = stability.clamp(self.bounds.s_min, self.bounds.s_max);
        let difficulty = difficulty.clamp(self.bounds.d_min, self.bounds.d_max);
        let log_s_min = self.bounds.s_min.ln();
        let log_s_span = self.bounds.s_max.ln() - log_s_min;
        let x_s = ((stability.ln() - log_s_min) / log_s_span).clamp(0.0, 1.0);
        let x_d = ((difficulty - self.bounds.d_min) / (self.bounds.d_max - self.bounds.d_min))
            .clamp(0.0, 1.0);
        [1.0, x_s, x_d, x_s * x_d, x_s * x_s]
    }

    fn normalized_cost_weight(&self, cost_weight: f32) -> f32 {
        let weight = cost_weight.clamp(self.cost_weight_min, self.cost_weight_max);
        let lo = self.cost_weight_min.ln_1p();
        let hi = self.cost_weight_max.ln_1p();
        ((weight.ln_1p() - lo) / (hi - lo)).clamp(0.0, 1.0)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct CostAdrPlotPoint {
    pub stability: f32,
    pub difficulty: f32,
    pub cost_weight: f32,
    pub desired_retention: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct CostAdrMetrics {
    pub memorized_average: f32,
    pub time_average: f32,
    pub memorized_per_minute: f32,
    pub total_reviews: usize,
    pub total_lapses: u32,
    pub total_cost: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct CostAdrEvaluationPoint {
    pub goal_cost_weight: f32,
    pub metrics: CostAdrMetrics,
    pub average_desired_retention: Option<f32>,
    pub fixed_fsrs_equivalent_desired_retention: Option<f32>,
    pub same_target_time_saved_percent: Option<f32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct CostAdrAucMetrics {
    pub baseline_point_count: usize,
    pub scheduler_point_count: usize,
    pub baseline_frontier_count: usize,
    pub scheduler_frontier_count: usize,
    pub target_count: usize,
    pub covered_target_count: usize,
    pub total_span: f32,
    pub covered_span: f32,
    pub span_coverage_percent: f32,
    pub same_target_time_saved_auc: Option<f32>,
    pub baseline_time_auc: Option<f32>,
    pub relative_same_target_time_saved_auc_percent: Option<f32>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CostAdrEvaluationConfig {
    pub cost_weights: Vec<f32>,
    pub baseline_desired_retentions: Vec<f32>,
    pub seed: Option<u64>,
}

impl Default for CostAdrEvaluationConfig {
    fn default() -> Self {
        Self {
            cost_weights: COST_ADR_DEFAULT_COST_WEIGHTS.to_vec(),
            baseline_desired_retentions: COST_ADR_DEFAULT_BASELINE_RETENTIONS.to_vec(),
            seed: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CostAdrEvaluationResult {
    pub baseline_metrics: Vec<CostAdrMetrics>,
    pub scheduler_metrics: Vec<CostAdrEvaluationPoint>,
    pub baseline_hypervolume: f32,
    pub scheduler_hypervolume: f32,
    pub hypervolume_delta: f32,
    pub auc_metrics: CostAdrAucMetrics,
}

impl CostAdrEvaluationResult {
    pub fn cost_weight_for_average_desired_retention(
        &self,
        target_average_desired_retention: f32,
    ) -> Option<f32> {
        cost_weight_for_average_desired_retention(
            &self.scheduler_metrics,
            target_average_desired_retention,
        )
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostAdrTrainingConfig {
    pub population_size: usize,
    pub generations: usize,
    pub sigma0: f32,
    pub seed: Option<u64>,
    pub simulation_seed: Option<u64>,
    pub lower_bound: f32,
    pub upper_bound: f32,
    pub retention_min: f32,
    pub retention_max: f32,
    pub initial_coefficients: Vec<f32>,
    pub cost_weights: Vec<f32>,
    pub baseline_desired_retentions: Vec<f32>,
    #[serde(skip)]
    pub progress: Option<Arc<Mutex<CombinedProgressState>>>,
}

impl PartialEq for CostAdrTrainingConfig {
    fn eq(&self, other: &Self) -> bool {
        self.population_size == other.population_size
            && self.generations == other.generations
            && self.sigma0 == other.sigma0
            && self.seed == other.seed
            && self.simulation_seed == other.simulation_seed
            && self.lower_bound == other.lower_bound
            && self.upper_bound == other.upper_bound
            && self.retention_min == other.retention_min
            && self.retention_max == other.retention_max
            && self.initial_coefficients == other.initial_coefficients
            && self.cost_weights == other.cost_weights
            && self.baseline_desired_retentions == other.baseline_desired_retentions
    }
}

impl Default for CostAdrTrainingConfig {
    fn default() -> Self {
        Self {
            population_size: 8,
            generations: 5,
            sigma0: 1.0,
            seed: None,
            simulation_seed: None,
            lower_bound: -64.0,
            upper_bound: 64.0,
            retention_min: 0.30,
            retention_max: 0.995,
            initial_coefficients: COST_ADR_DEFAULT_INITIAL_COEFFICIENTS.to_vec(),
            cost_weights: COST_ADR_DEFAULT_COST_WEIGHTS.to_vec(),
            baseline_desired_retentions: COST_ADR_DEFAULT_BASELINE_RETENTIONS.to_vec(),
            progress: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CostAdrGenerationMetrics {
    pub generation: usize,
    pub best_hypervolume_delta: f32,
    pub generation_best_hypervolume_delta: f32,
    pub mean_hypervolume_delta: f32,
    pub sigma: f32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CostAdrTrainingResult {
    pub policy: CostAdrPolicy,
    pub baseline_metrics: Vec<CostAdrMetrics>,
    pub baseline_hypervolume: f32,
    pub best_hypervolume: f32,
    pub best_hypervolume_delta: f32,
    pub best_auc_metrics: CostAdrAucMetrics,
    pub best_cost_weight_metrics: Vec<CostAdrEvaluationPoint>,
    pub history: Vec<CostAdrGenerationMetrics>,
    pub training_seconds: f32,
}

impl CostAdrTrainingResult {
    pub fn cost_weight_for_average_desired_retention(
        &self,
        target_average_desired_retention: f32,
    ) -> Option<f32> {
        cost_weight_for_average_desired_retention(
            &self.best_cost_weight_metrics,
            target_average_desired_retention,
        )
    }
}

fn evaluate_cost_adr_policy(
    config: &SimulatorConfig,
    parameters: &Parameters,
    policy: &CostAdrPolicy,
    evaluation_config: &CostAdrEvaluationConfig,
) -> Result<CostAdrEvaluationResult> {
    validate_evaluation_config(evaluation_config)?;
    let seed = evaluation_config.seed.unwrap_or(COST_ADR_DEFAULT_SEED);
    let baseline_metrics = evaluate_baseline_desired_retentions(
        config,
        parameters,
        &evaluation_config.baseline_desired_retentions,
        seed,
    )?;
    let baseline_points = points_from_metrics(&baseline_metrics);
    let reference = reference_point(&baseline_points)?;
    let baseline_hypervolume = hypervolume_2d(&baseline_points, reference);
    let mut scheduler_metrics = evaluate_cost_adr_rollouts(
        config,
        parameters,
        policy,
        &evaluation_config.cost_weights,
        seed,
    )?;
    annotate_cost_adr_rollouts(
        &evaluation_config.baseline_desired_retentions,
        &baseline_metrics,
        &mut scheduler_metrics,
    );
    let scheduler_metrics_only = scheduler_metrics
        .iter()
        .map(|point| point.metrics)
        .collect::<Vec<_>>();
    let scheduler_hypervolume =
        hypervolume_2d(&points_from_metrics(&scheduler_metrics_only), reference);
    let hypervolume_delta = scheduler_hypervolume - baseline_hypervolume;
    let auc_metrics = cost_adr_auc_metrics(&baseline_metrics, &scheduler_metrics_only);

    Ok(CostAdrEvaluationResult {
        baseline_metrics,
        scheduler_metrics,
        baseline_hypervolume,
        scheduler_hypervolume,
        hypervolume_delta,
        auc_metrics,
    })
}

fn evaluate_cost_adr_rollouts(
    config: &SimulatorConfig,
    parameters: &Parameters,
    policy: &CostAdrPolicy,
    cost_weights: &[f32],
    seed: u64,
) -> Result<Vec<CostAdrEvaluationPoint>> {
    policy.validate()?;
    cost_weights
        .par_iter()
        .enumerate()
        .map(|(index, &goal_cost_weight)| {
            let result = simulate_with_cost_adr_policy_for_evaluation(
                config,
                parameters,
                policy,
                goal_cost_weight,
                Some(seed + index as u64),
                None,
            )?;
            let metrics = metrics_from_simulation(&result.result);
            Ok(CostAdrEvaluationPoint {
                goal_cost_weight,
                metrics,
                average_desired_retention: result.average_desired_retention,
                fixed_fsrs_equivalent_desired_retention: None,
                same_target_time_saved_percent: None,
            })
        })
        .collect()
}

fn annotate_cost_adr_rollouts(
    baseline_desired_retentions: &[f32],
    baseline_metrics: &[CostAdrMetrics],
    scheduler_metrics: &mut [CostAdrEvaluationPoint],
) {
    let mut baseline_points = baseline_desired_retentions
        .iter()
        .copied()
        .zip(baseline_metrics.iter().copied())
        .map(
            |(desired_retention, metrics)| DesiredRetentionMemoryTimePoint {
                desired_retention,
                memorized_average: metrics.memorized_average,
                time_average: metrics.time_average,
            },
        )
        .filter(|point| {
            point.desired_retention.is_finite()
                && point.memorized_average.is_finite()
                && point.time_average.is_finite()
        })
        .collect::<Vec<_>>();
    baseline_points.sort_by(|left, right| {
        left.memorized_average
            .partial_cmp(&right.memorized_average)
            .unwrap_or(Ordering::Equal)
    });

    for point in scheduler_metrics {
        let target = point.metrics.memorized_average;
        point.fixed_fsrs_equivalent_desired_retention =
            interpolated_desired_retention_for_memory_target(&baseline_points, target);
        let baseline_time =
            interpolated_time_for_desired_retention_memory_target(&baseline_points, target);
        point.same_target_time_saved_percent = baseline_time.and_then(|time| {
            if time > 0.0 {
                Some(((time - point.metrics.time_average) / time) * 100.0)
            } else {
                None
            }
        });
    }
}

#[derive(Debug, Clone, Copy)]
struct AverageDesiredRetentionWeightPoint {
    average_desired_retention: f32,
    cost_weight: f32,
}

fn cost_weight_for_average_desired_retention(
    points: &[CostAdrEvaluationPoint],
    target_average_desired_retention: f32,
) -> Option<f32> {
    if !target_average_desired_retention.is_finite() {
        return None;
    }
    let mut points = points
        .iter()
        .filter_map(|point| {
            let average_desired_retention = point.average_desired_retention?;
            if average_desired_retention.is_finite()
                && point.goal_cost_weight.is_finite()
                && point.goal_cost_weight >= 0.0
            {
                Some(AverageDesiredRetentionWeightPoint {
                    average_desired_retention,
                    cost_weight: point.goal_cost_weight,
                })
            } else {
                None
            }
        })
        .collect::<Vec<_>>();
    if points.is_empty() {
        return None;
    }

    points.sort_by(|left, right| {
        left.average_desired_retention
            .partial_cmp(&right.average_desired_retention)
            .unwrap_or(Ordering::Equal)
    });
    collapse_average_desired_retention_weight_points(&mut points);

    let first = points[0];
    let last = points[points.len() - 1];
    if target_average_desired_retention < first.average_desired_retention
        && !is_close(
            target_average_desired_retention,
            first.average_desired_retention,
        )
    {
        return None;
    }
    if is_close(
        target_average_desired_retention,
        first.average_desired_retention,
    ) {
        return Some(first.cost_weight);
    }
    if target_average_desired_retention > last.average_desired_retention
        && !is_close(
            target_average_desired_retention,
            last.average_desired_retention,
        )
    {
        return None;
    }
    if is_close(
        target_average_desired_retention,
        last.average_desired_retention,
    ) {
        return Some(last.cost_weight);
    }

    for pair in points.windows(2) {
        let left = pair[0];
        let right = pair[1];
        if !(left.average_desired_retention <= target_average_desired_retention
            && target_average_desired_retention <= right.average_desired_retention)
        {
            continue;
        }
        let ratio = (target_average_desired_retention - left.average_desired_retention)
            / (right.average_desired_retention - left.average_desired_retention);
        let left_weight = left.cost_weight.ln_1p();
        let right_weight = right.cost_weight.ln_1p();
        return Some((left_weight + ratio * (right_weight - left_weight)).exp_m1());
    }
    None
}

fn collapse_average_desired_retention_weight_points(
    points: &mut Vec<AverageDesiredRetentionWeightPoint>,
) {
    let mut collapsed: Vec<AverageDesiredRetentionWeightPoint> = Vec::new();
    for point in points.iter().copied() {
        if let Some(last) = collapsed.last_mut() {
            if is_close(
                last.average_desired_retention,
                point.average_desired_retention,
            ) {
                last.cost_weight = last.cost_weight.min(point.cost_weight);
                continue;
            }
        }
        collapsed.push(point);
    }
    *points = collapsed;
}

fn evaluate_baseline_desired_retentions(
    config: &SimulatorConfig,
    parameters: &Parameters,
    desired_retentions: &[f32],
    seed: u64,
) -> Result<Vec<CostAdrMetrics>> {
    desired_retentions
        .par_iter()
        .enumerate()
        .map(|(index, &desired_retention)| {
            let result = simulate(
                config,
                parameters,
                desired_retention,
                Some(seed + index as u64),
                None,
            )?;
            Ok(metrics_from_simulation(&result))
        })
        .collect()
}

fn cost_adr_auc_metrics(
    baseline_metrics: &[CostAdrMetrics],
    scheduler_metrics: &[CostAdrMetrics],
) -> CostAdrAucMetrics {
    let baseline_frontier = frontier_memory_time_points(baseline_metrics);
    let scheduler_frontier = frontier_memory_time_points(scheduler_metrics);
    let baseline_targets = baseline_frontier
        .iter()
        .map(|point| point.memorized_average)
        .collect::<Vec<_>>();
    let total_span = if baseline_targets.len() > 1 {
        baseline_targets[baseline_targets.len() - 1] - baseline_targets[0]
    } else {
        0.0
    };

    let mut covered_span = 0.0;
    let mut time_saved_area = 0.0;
    let mut baseline_time_area = 0.0;
    let mut covered_target_count = 0;

    if !baseline_frontier.is_empty() && !scheduler_frontier.is_empty() {
        let start = baseline_frontier[0]
            .memorized_average
            .max(scheduler_frontier[0].memorized_average);
        let end = baseline_frontier[baseline_frontier.len() - 1]
            .memorized_average
            .min(scheduler_frontier[scheduler_frontier.len() - 1].memorized_average);
        if end > start {
            covered_target_count = values_in_interval(&baseline_targets, start, end).len();
            let scheduler_targets = scheduler_frontier
                .iter()
                .map(|point| point.memorized_average)
                .collect::<Vec<_>>();
            let targets = integration_grid(&baseline_targets, &scheduler_targets, start, end);
            for pair in targets.windows(2) {
                let left_target = pair[0];
                let right_target = pair[1];
                let width = right_target - left_target;
                if width <= 0.0 {
                    continue;
                }
                let left_baseline_time =
                    interpolated_time_for_memory_target(&baseline_frontier, left_target);
                let right_baseline_time =
                    interpolated_time_for_memory_target(&baseline_frontier, right_target);
                let left_scheduler_time =
                    interpolated_time_for_memory_target(&scheduler_frontier, left_target);
                let right_scheduler_time =
                    interpolated_time_for_memory_target(&scheduler_frontier, right_target);
                if let (
                    Some(left_baseline),
                    Some(right_baseline),
                    Some(left_scheduler),
                    Some(right_scheduler),
                ) = (
                    left_baseline_time,
                    right_baseline_time,
                    left_scheduler_time,
                    right_scheduler_time,
                ) {
                    let left_saved = left_baseline - left_scheduler;
                    let right_saved = right_baseline - right_scheduler;
                    time_saved_area += width * ((left_saved + right_saved) / 2.0);
                    baseline_time_area += width * ((left_baseline + right_baseline) / 2.0);
                    covered_span += width;
                }
            }
        }
    }

    let same_target_time_saved_auc = if covered_span > 0.0 {
        Some(time_saved_area / covered_span)
    } else {
        None
    };
    let baseline_time_auc = if covered_span > 0.0 {
        Some(baseline_time_area / covered_span)
    } else {
        None
    };
    let relative_same_target_time_saved_auc_percent =
        match (same_target_time_saved_auc, baseline_time_auc) {
            (Some(same_target), Some(baseline_time)) if baseline_time != 0.0 => {
                Some((same_target / baseline_time) * 100.0)
            }
            _ => None,
        };

    CostAdrAucMetrics {
        baseline_point_count: baseline_metrics.len(),
        scheduler_point_count: scheduler_metrics.len(),
        baseline_frontier_count: baseline_frontier.len(),
        scheduler_frontier_count: scheduler_frontier.len(),
        target_count: baseline_targets.len(),
        covered_target_count,
        total_span,
        covered_span,
        span_coverage_percent: if total_span > 0.0 {
            (covered_span / total_span) * 100.0
        } else {
            0.0
        },
        same_target_time_saved_auc,
        baseline_time_auc,
        relative_same_target_time_saved_auc_percent,
    }
}

fn train_cost_adr_single_user(
    config: &SimulatorConfig,
    parameters: &Parameters,
    training_config: &CostAdrTrainingConfig,
) -> Result<CostAdrTrainingResult> {
    if let Err(err) = validate_training_config(training_config) {
        finish_cost_adr_training_progress(&training_config.progress);
        return Err(err);
    }
    reset_cost_adr_training_progress(training_config);
    let result = train_cost_adr_single_user_inner(config, parameters, training_config);
    finish_cost_adr_training_progress(&training_config.progress);
    result
}

fn train_cost_adr_single_user_inner(
    config: &SimulatorConfig,
    parameters: &Parameters,
    training_config: &CostAdrTrainingConfig,
) -> Result<CostAdrTrainingResult> {
    if cost_adr_training_should_abort(&training_config.progress) {
        return Err(FSRSError::Interrupted);
    }

    let started = Instant::now();
    let seed = training_config.seed.unwrap_or(COST_ADR_DEFAULT_SEED);
    let simulation_seed = training_config
        .simulation_seed
        .unwrap_or(COST_ADR_DEFAULT_SEED);
    let initial_coefficients = clamp_coefficients(
        &training_config.initial_coefficients,
        training_config.lower_bound,
        training_config.upper_bound,
    );
    let baseline_metrics = evaluate_baseline_desired_retentions(
        config,
        parameters,
        &training_config.baseline_desired_retentions,
        simulation_seed,
    )?;
    let baseline_points = points_from_metrics(&baseline_metrics);
    let reference = reference_point(&baseline_points)?;
    let baseline_hypervolume = hypervolume_2d(&baseline_points, reference);
    let mut optimizer = SeparableCmaEs::new(
        initial_coefficients.clone(),
        training_config.sigma0,
        training_config.lower_bound,
        training_config.upper_bound,
        seed,
    );

    let mut best_coefficients = initial_coefficients.clone();
    let mut best_cost_weight_metrics = Vec::new();
    let mut best_hypervolume = f32::NEG_INFINITY;
    let mut best_hypervolume_delta = f32::NEG_INFINITY;
    let mut history = Vec::with_capacity(training_config.generations);

    for generation in 0..training_config.generations {
        if cost_adr_training_should_abort(&training_config.progress) {
            return Err(FSRSError::Interrupted);
        }

        let mut solutions = optimizer.ask(training_config.population_size);
        if generation == 0 && !solutions.is_empty() {
            solutions[0] = initial_coefficients.clone();
        }

        let completed_candidates = AtomicUsize::new(generation * training_config.population_size);
        let progress = training_config.progress.clone();
        let candidate_results: Result<Vec<CandidateEvaluation>> = solutions
            .into_par_iter()
            .map(|coefficients| {
                if cost_adr_training_should_abort(&progress) {
                    return Err(FSRSError::Interrupted);
                }
                let policy =
                    cost_adr_policy_from_training_config(coefficients.clone(), training_config)?;
                let mut points = evaluate_cost_adr_rollouts(
                    config,
                    parameters,
                    &policy,
                    &training_config.cost_weights,
                    simulation_seed,
                )?;
                annotate_cost_adr_rollouts(
                    &training_config.baseline_desired_retentions,
                    &baseline_metrics,
                    &mut points,
                );
                let candidate_metrics =
                    points.iter().map(|point| point.metrics).collect::<Vec<_>>();
                let candidate_points = points_from_metrics(&candidate_metrics);
                let hypervolume = hypervolume_2d(&candidate_points, reference);
                let hypervolume_delta = hypervolume - baseline_hypervolume;
                let evaluation = Ok(CandidateEvaluation {
                    coefficients,
                    rollout_points: points,
                    hypervolume,
                    hypervolume_delta,
                });
                let completed = completed_candidates.fetch_add(1, AtomicOrdering::Relaxed) + 1;
                update_cost_adr_training_progress(
                    &progress,
                    completed,
                    training_config.population_size,
                );
                evaluation
            })
            .collect();

        let candidate_results = candidate_results?;
        let scores = candidate_results
            .iter()
            .map(|candidate| candidate.hypervolume_delta)
            .collect::<Vec<_>>();
        optimizer.tell(&candidate_results, &scores);

        let generation_best = candidate_results
            .iter()
            .max_by(|left, right| {
                left.hypervolume_delta
                    .partial_cmp(&right.hypervolume_delta)
                    .unwrap_or(Ordering::Equal)
            })
            .ok_or(FSRSError::InvalidInput)?;
        if generation_best.hypervolume_delta > best_hypervolume_delta {
            best_coefficients = generation_best.coefficients.clone();
            best_cost_weight_metrics = generation_best.rollout_points.clone();
            best_hypervolume = generation_best.hypervolume;
            best_hypervolume_delta = generation_best.hypervolume_delta;
        }

        let mean_delta = scores.iter().sum::<f32>() / scores.len() as f32;
        history.push(CostAdrGenerationMetrics {
            generation,
            best_hypervolume_delta,
            generation_best_hypervolume_delta: generation_best.hypervolume_delta,
            mean_hypervolume_delta: mean_delta,
            sigma: optimizer.sigma,
        });
    }

    let best_metrics = best_cost_weight_metrics
        .iter()
        .map(|point| point.metrics)
        .collect::<Vec<_>>();
    let best_auc_metrics = cost_adr_auc_metrics(&baseline_metrics, &best_metrics);

    Ok(CostAdrTrainingResult {
        policy: cost_adr_policy_from_training_config(best_coefficients, training_config)?,
        baseline_metrics,
        baseline_hypervolume,
        best_hypervolume,
        best_hypervolume_delta,
        best_auc_metrics,
        best_cost_weight_metrics,
        history,
        training_seconds: started.elapsed().as_secs_f32(),
    })
}

fn cost_adr_policy_from_training_config(
    coefficients: Vec<f32>,
    config: &CostAdrTrainingConfig,
) -> Result<CostAdrPolicy> {
    CostAdrPolicy::new_with_settings(
        coefficients,
        0.0,
        1024.0,
        config.retention_min,
        config.retention_max,
        None,
    )
}

fn reset_cost_adr_training_progress(config: &CostAdrTrainingConfig) {
    if let Some(progress) = &config.progress {
        let progress_state = ProgressState {
            epoch_total: config.generations,
            items_total: config.population_size,
            epoch: 0,
            items_processed: 0,
        };
        progress.lock().unwrap().reset(vec![progress_state]);
    }
}

fn finish_cost_adr_training_progress(progress: &Option<Arc<Mutex<CombinedProgressState>>>) {
    if let Some(progress) = progress {
        progress.lock().unwrap().mark_finished();
    }
}

fn cost_adr_training_should_abort(progress: &Option<Arc<Mutex<CombinedProgressState>>>) -> bool {
    progress
        .as_ref()
        .is_some_and(|progress| progress.lock().unwrap().want_abort)
}

fn update_cost_adr_training_progress(
    progress: &Option<Arc<Mutex<CombinedProgressState>>>,
    completed: usize,
    items_total: usize,
) {
    if let Some(progress) = progress {
        let (epoch, items_processed) = cost_adr_training_progress_position(completed, items_total);
        let mut state = progress.lock().unwrap();
        if let Some(split) = state.splits.first_mut() {
            split.epoch = epoch;
            split.items_processed = items_processed;
        }
    }
}

fn cost_adr_training_progress_position(completed: usize, items_total: usize) -> (usize, usize) {
    if completed == 0 {
        return (0, 0);
    }
    let items_processed = completed % items_total;
    if items_processed == 0 {
        (completed / items_total, items_total)
    } else {
        (completed / items_total + 1, items_processed)
    }
}

fn validate_training_config(config: &CostAdrTrainingConfig) -> Result<()> {
    if config.population_size < 2
        || config.generations == 0
        || config.sigma0 <= 0.0
        || !config.lower_bound.is_finite()
        || !config.upper_bound.is_finite()
        || config.lower_bound >= config.upper_bound
        || !config.retention_min.is_finite()
        || !config.retention_max.is_finite()
        || !(0.0 < config.retention_min && config.retention_min < config.retention_max)
        || config.retention_max >= 1.0
        || config.initial_coefficients.len() != COST_ADR_PARAMETER_COUNT
        || config
            .initial_coefficients
            .iter()
            .any(|value| !value.is_finite())
        || config.cost_weights.is_empty()
        || config.baseline_desired_retentions.is_empty()
    {
        return Err(FSRSError::InvalidInput);
    }
    validate_evaluation_inputs(&config.cost_weights, &config.baseline_desired_retentions)
}

fn clamp_coefficients(coefficients: &[f32], lower_bound: f32, upper_bound: f32) -> Vec<f32> {
    coefficients
        .iter()
        .map(|&value| value.clamp(lower_bound, upper_bound))
        .collect()
}

fn validate_evaluation_config(config: &CostAdrEvaluationConfig) -> Result<()> {
    validate_evaluation_inputs(&config.cost_weights, &config.baseline_desired_retentions)
}

fn validate_evaluation_inputs(
    cost_weights: &[f32],
    baseline_desired_retentions: &[f32],
) -> Result<()> {
    if cost_weights.is_empty()
        || baseline_desired_retentions.is_empty()
        || cost_weights
            .iter()
            .any(|value| !value.is_finite() || *value < 0.0)
        || baseline_desired_retentions
            .iter()
            .any(|value| !(value.is_finite() && 0.0 < *value && *value < 1.0))
    {
        return Err(FSRSError::InvalidInput);
    }
    Ok(())
}

fn metrics_from_simulation(result: &SimulationResult) -> CostAdrMetrics {
    let total_cost = result.cost_per_day.iter().sum::<f32>();
    let time_average = if result.cost_per_day.is_empty() {
        0.0
    } else {
        total_cost / result.cost_per_day.len() as f32 / 60.0
    };
    let memorized_average = if result.memorized_cnt_per_day.is_empty() {
        0.0
    } else {
        result.memorized_cnt_per_day.iter().sum::<f32>() / result.memorized_cnt_per_day.len() as f32
    };
    CostAdrMetrics {
        memorized_average,
        time_average,
        memorized_per_minute: if time_average > 0.0 {
            memorized_average / time_average
        } else {
            0.0
        },
        total_reviews: result.review_cnt_per_day.iter().sum::<usize>()
            + result.learn_cnt_per_day.iter().sum::<usize>(),
        total_lapses: result.cards.iter().map(|card| card.lapses).sum(),
        total_cost,
    }
}

#[derive(Debug, Clone, Copy)]
struct MemoryTimePoint {
    memorized_average: f32,
    time_average: f32,
}

#[derive(Debug, Clone, Copy)]
struct DesiredRetentionMemoryTimePoint {
    desired_retention: f32,
    memorized_average: f32,
    time_average: f32,
}

fn frontier_memory_time_points(metrics: &[CostAdrMetrics]) -> Vec<MemoryTimePoint> {
    let mut frontier = Vec::new();
    for candidate in metrics {
        if !(candidate.memorized_average.is_finite() && candidate.time_average.is_finite()) {
            continue;
        }
        let dominated = metrics.iter().any(|other| {
            if !(other.memorized_average.is_finite() && other.time_average.is_finite()) {
                return false;
            }
            let no_worse = other.memorized_average >= candidate.memorized_average
                && other.time_average <= candidate.time_average;
            let strictly_better = other.memorized_average > candidate.memorized_average
                || other.time_average < candidate.time_average;
            no_worse && strictly_better
        });
        if !dominated {
            frontier.push(MemoryTimePoint {
                memorized_average: candidate.memorized_average,
                time_average: candidate.time_average,
            });
        }
    }
    frontier.sort_by(|left, right| {
        left.memorized_average
            .partial_cmp(&right.memorized_average)
            .unwrap_or(Ordering::Equal)
            .then_with(|| {
                left.time_average
                    .partial_cmp(&right.time_average)
                    .unwrap_or(Ordering::Equal)
            })
    });

    let mut collapsed: Vec<MemoryTimePoint> = Vec::new();
    for point in frontier {
        if let Some(last) = collapsed.last_mut() {
            if is_close(last.memorized_average, point.memorized_average) {
                last.time_average = last.time_average.min(point.time_average);
                continue;
            }
        }
        collapsed.push(point);
    }
    collapsed
}

fn values_in_interval(values: &[f32], start: f32, end: f32) -> Vec<f32> {
    let mut selected = values
        .iter()
        .copied()
        .filter(|value| {
            (*value > start && *value < end) || is_close(*value, start) || is_close(*value, end)
        })
        .collect::<Vec<_>>();
    sort_and_dedup_close(&mut selected);
    selected
}

fn integration_grid(
    baseline_values: &[f32],
    scheduler_values: &[f32],
    start: f32,
    end: f32,
) -> Vec<f32> {
    let mut values = vec![start, end];
    values.extend(values_in_interval(baseline_values, start, end));
    values.extend(values_in_interval(scheduler_values, start, end));
    sort_and_dedup_close(&mut values);
    values
}

fn interpolated_time_for_memory_target(points: &[MemoryTimePoint], target: f32) -> Option<f32> {
    if points.is_empty() {
        return None;
    }
    let first = points[0];
    let last = points[points.len() - 1];
    if target < first.memorized_average && !is_close(target, first.memorized_average) {
        return None;
    }
    if is_close(target, first.memorized_average) {
        return Some(first.time_average);
    }
    if target > last.memorized_average && !is_close(target, last.memorized_average) {
        return None;
    }
    if is_close(target, last.memorized_average) {
        return Some(last.time_average);
    }
    for pair in points.windows(2) {
        let left = pair[0];
        let right = pair[1];
        if !(left.memorized_average <= target && target <= right.memorized_average) {
            continue;
        }
        if is_close(left.memorized_average, right.memorized_average) {
            return Some(left.time_average.min(right.time_average));
        }
        let ratio =
            (target - left.memorized_average) / (right.memorized_average - left.memorized_average);
        return Some(left.time_average + ratio * (right.time_average - left.time_average));
    }
    None
}

fn interpolated_desired_retention_for_memory_target(
    points: &[DesiredRetentionMemoryTimePoint],
    target: f32,
) -> Option<f32> {
    interpolated_desired_retention_memory_time_for_target(points, target)
        .map(|(desired_retention, _time)| desired_retention)
}

fn interpolated_time_for_desired_retention_memory_target(
    points: &[DesiredRetentionMemoryTimePoint],
    target: f32,
) -> Option<f32> {
    interpolated_desired_retention_memory_time_for_target(points, target)
        .map(|(_desired_retention, time)| time)
}

fn interpolated_desired_retention_memory_time_for_target(
    points: &[DesiredRetentionMemoryTimePoint],
    target: f32,
) -> Option<(f32, f32)> {
    if points.is_empty() {
        return None;
    }
    let first = points[0];
    let last = points[points.len() - 1];
    if target < first.memorized_average && !is_close(target, first.memorized_average) {
        return None;
    }
    if is_close(target, first.memorized_average) {
        return Some((first.desired_retention, first.time_average));
    }
    if target > last.memorized_average && !is_close(target, last.memorized_average) {
        return None;
    }
    if is_close(target, last.memorized_average) {
        return Some((last.desired_retention, last.time_average));
    }
    for pair in points.windows(2) {
        let left = pair[0];
        let right = pair[1];
        if !(left.memorized_average <= target && target <= right.memorized_average) {
            continue;
        }
        if is_close(left.memorized_average, right.memorized_average) {
            let point = if left.time_average <= right.time_average {
                left
            } else {
                right
            };
            return Some((point.desired_retention, point.time_average));
        }
        let ratio =
            (target - left.memorized_average) / (right.memorized_average - left.memorized_average);
        return Some((
            left.desired_retention + ratio * (right.desired_retention - left.desired_retention),
            left.time_average + ratio * (right.time_average - left.time_average),
        ));
    }
    None
}

fn sort_and_dedup_close(values: &mut Vec<f32>) {
    values.sort_by(|left, right| left.partial_cmp(right).unwrap_or(Ordering::Equal));
    values.dedup_by(|left, right| is_close(*left, *right));
}

fn is_close(left: f32, right: f32) -> bool {
    let scale = left.abs().max(right.abs()).max(1.0);
    (left - right).abs() <= 1e-6 * scale
}

#[derive(Debug, Clone, Copy)]
struct ObjectivePoint {
    memorized_average: f32,
    negative_time_average: f32,
}

fn points_from_metrics(metrics: &[CostAdrMetrics]) -> Vec<ObjectivePoint> {
    metrics
        .iter()
        .map(|metric| ObjectivePoint {
            memorized_average: metric.memorized_average,
            negative_time_average: -metric.time_average,
        })
        .collect()
}

fn reference_point(points: &[ObjectivePoint]) -> Result<ObjectivePoint> {
    if points.is_empty() {
        return Err(FSRSError::InvalidInput);
    }
    let min_x = points
        .iter()
        .map(|point| point.memorized_average)
        .fold(f32::INFINITY, f32::min);
    let max_x = points
        .iter()
        .map(|point| point.memorized_average)
        .fold(f32::NEG_INFINITY, f32::max);
    let min_y = points
        .iter()
        .map(|point| point.negative_time_average)
        .fold(f32::INFINITY, f32::min);
    let max_y = points
        .iter()
        .map(|point| point.negative_time_average)
        .fold(f32::NEG_INFINITY, f32::max);
    let x_span = (max_x - min_x).max(min_x.abs()).max(1.0);
    let y_span = (max_y - min_y).max(min_y.abs()).max(1.0);
    Ok(ObjectivePoint {
        memorized_average: min_x - x_span * 0.05,
        negative_time_average: min_y - y_span * 0.05,
    })
}

fn hypervolume_2d(points: &[ObjectivePoint], reference: ObjectivePoint) -> f32 {
    let mut frontier = points
        .iter()
        .copied()
        .filter(|point| {
            point.memorized_average > reference.memorized_average
                && point.negative_time_average > reference.negative_time_average
        })
        .collect::<Vec<_>>();
    let all_contributing = frontier.clone();
    frontier = all_contributing
        .iter()
        .copied()
        .filter(|point| {
            !all_contributing.iter().any(|other| {
                (other.memorized_average >= point.memorized_average
                    && other.negative_time_average >= point.negative_time_average)
                    && (other.memorized_average > point.memorized_average
                        || other.negative_time_average > point.negative_time_average)
            })
        })
        .collect();
    frontier.sort_by(|left, right| {
        left.memorized_average
            .partial_cmp(&right.memorized_average)
            .unwrap_or(Ordering::Equal)
            .then_with(|| {
                left.negative_time_average
                    .partial_cmp(&right.negative_time_average)
                    .unwrap_or(Ordering::Equal)
            })
    });
    let mut hypervolume = 0.0;
    let mut previous_x = reference.memorized_average;
    for point in frontier {
        let width = (point.memorized_average - previous_x).max(0.0);
        let height = (point.negative_time_average - reference.negative_time_average).max(0.0);
        hypervolume += width * height;
        previous_x = previous_x.max(point.memorized_average);
    }
    hypervolume
}

#[derive(Debug)]
struct SeparableCmaEs {
    mean: Vec<f32>,
    sigma: f32,
    variances: Vec<f32>,
    lower_bound: f32,
    upper_bound: f32,
    rng: StdRng,
    best_score: f32,
}

#[derive(Debug, Clone)]
struct CandidateEvaluation {
    coefficients: Vec<f32>,
    rollout_points: Vec<CostAdrEvaluationPoint>,
    hypervolume: f32,
    hypervolume_delta: f32,
}

impl SeparableCmaEs {
    fn new(mean: Vec<f32>, sigma: f32, lower_bound: f32, upper_bound: f32, seed: u64) -> Self {
        let variances = vec![1.0; mean.len()];
        Self {
            mean,
            sigma,
            variances,
            lower_bound,
            upper_bound,
            rng: StdRng::seed_from_u64(seed),
            best_score: f32::NEG_INFINITY,
        }
    }

    fn ask(&mut self, population_size: usize) -> Vec<Vec<f32>> {
        let mut population = Vec::with_capacity(population_size);
        for _ in 0..population_size {
            let mut candidate = Vec::with_capacity(self.mean.len());
            for dimension in 0..self.mean.len() {
                let mean = self.mean[dimension];
                let variance = self.variances[dimension];
                candidate.push(
                    (mean + self.sigma * variance.sqrt() * self.sample_standard_normal())
                        .clamp(self.lower_bound, self.upper_bound),
                );
            }
            population.push(candidate);
        }
        population
    }

    fn tell(&mut self, candidates: &[CandidateEvaluation], scores: &[f32]) {
        let mut order = (0..scores.len()).collect::<Vec<_>>();
        order.sort_by(|&left, &right| {
            scores[right]
                .partial_cmp(&scores[left])
                .unwrap_or(Ordering::Equal)
        });
        let mu = (scores.len() / 2).max(1);
        let raw_weights = (0..mu)
            .map(|rank| ((mu as f32 + 0.5).ln() - ((rank + 1) as f32).ln()).max(0.0))
            .collect::<Vec<_>>();
        let weight_sum = raw_weights.iter().sum::<f32>().max(f32::EPSILON);
        let weights = raw_weights
            .iter()
            .map(|weight| weight / weight_sum)
            .collect::<Vec<_>>();
        let old_mean = self.mean.clone();
        for value in &mut self.mean {
            *value = 0.0;
        }
        for (&candidate_index, &weight) in order.iter().take(mu).zip(weights.iter()) {
            for (dimension, value) in candidates[candidate_index].coefficients.iter().enumerate() {
                self.mean[dimension] += weight * value;
            }
        }
        let mut new_variances = vec![0.0; self.variances.len()];
        for (&candidate_index, &weight) in order.iter().take(mu).zip(weights.iter()) {
            for (dimension, value) in candidates[candidate_index].coefficients.iter().enumerate() {
                let normalized = (value - old_mean[dimension]) / self.sigma.max(1e-6);
                new_variances[dimension] += weight * normalized * normalized;
            }
        }
        for (variance, new_variance) in self.variances.iter_mut().zip(new_variances) {
            *variance = (0.85 * *variance + 0.15 * new_variance).clamp(1e-6, 1e4);
        }
        let generation_best = scores[order[0]];
        if generation_best > self.best_score {
            self.best_score = generation_best;
            self.sigma = (self.sigma * 1.04).min(16.0);
        } else {
            self.sigma = (self.sigma * 0.82).max(1e-3);
        }
    }

    fn sample_standard_normal(&mut self) -> f32 {
        self.rng.sample(StandardNormal)
    }
}

fn dot(coefficients: &[f32], features: &[f32; 5]) -> f32 {
    coefficients
        .iter()
        .zip(features.iter())
        .map(|(coefficient, feature)| coefficient * feature)
        .sum()
}

fn sigmoid(value: f32) -> f32 {
    if value >= 0.0 {
        let z = (-value).exp();
        1.0 / (1.0 + z)
    } else {
        let z = value.exp();
        z / (1.0 + z)
    }
}

fn softplus(value: f32) -> f32 {
    if value > 20.0 {
        value
    } else if value < -20.0 {
        value.exp()
    } else {
        value.exp().ln_1p()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::DEFAULT_PARAMETERS;
    use crate::{FSRS, MemoryState};

    fn test_metrics(memorized_average: f32, time_average: f32) -> CostAdrMetrics {
        CostAdrMetrics {
            memorized_average,
            time_average,
            memorized_per_minute: memorized_average / time_average,
            total_reviews: 0,
            total_lapses: 0,
            total_cost: 0.0,
        }
    }

    fn test_evaluation_point(
        cost_weight: f32,
        average_desired_retention: f32,
    ) -> CostAdrEvaluationPoint {
        CostAdrEvaluationPoint {
            goal_cost_weight: cost_weight,
            metrics: test_metrics(1.0, 1.0),
            average_desired_retention: Some(average_desired_retention),
            fixed_fsrs_equivalent_desired_retention: None,
            same_target_time_saved_percent: None,
        }
    }

    #[test]
    fn test_default_policy_retention_decreases_with_cost() {
        let policy = CostAdrPolicy::default_initial();
        let low = policy.evaluate_retention(10.0, 5.0, 0.0);
        let high = policy.evaluate_retention(10.0, 5.0, 1024.0);
        assert!(low > high);
        assert!((0.30..=0.995).contains(&low));
        assert!((0.30..=0.995).contains(&high));
    }

    #[test]
    fn test_policy_clone_round_trip() -> Result<()> {
        let policy = CostAdrPolicy::default_initial();
        let decoded = policy.clone();
        decoded.validate()?;
        assert_eq!(decoded.policy_version, COST_ADR_POLICY_VERSION);
        assert_eq!(decoded.coefficients, policy.coefficients);
        Ok(())
    }

    #[test]
    fn test_policy_rejects_unknown_version() {
        let mut policy = CostAdrPolicy::default_initial();
        policy.policy_version += 1;
        assert_eq!(policy.validate(), Err(FSRSError::InvalidInput));
    }

    #[test]
    fn test_policy_rejects_non_finite_bounds() {
        let mut policy = CostAdrPolicy::default_initial();
        policy.cost_weight_max = f32::NAN;
        assert_eq!(policy.validate(), Err(FSRSError::InvalidInput));

        let mut policy = CostAdrPolicy::default_initial();
        policy.bounds.s_max = f32::INFINITY;
        assert_eq!(policy.validate(), Err(FSRSError::InvalidInput));
    }

    #[test]
    fn test_retention_grid_exports_finite_plot_points() -> Result<()> {
        let policy = CostAdrPolicy::default_initial();
        let points = policy.retention_grid(&[1.0, 10.0], &[3.0, 7.0], &[0.0, 64.0])?;
        assert_eq!(points.len(), 8);
        for point in points {
            assert!(point.desired_retention.is_finite());
            assert!(
                (policy.retention_min..=policy.retention_max).contains(&point.desired_retention)
            );
        }
        Ok(())
    }

    #[test]
    fn test_cost_weight_for_average_desired_retention_interpolates_log_weight() {
        let result = CostAdrEvaluationResult {
            baseline_metrics: Vec::new(),
            scheduler_metrics: vec![
                test_evaluation_point(0.0, 0.9),
                test_evaluation_point(99.0, 0.7),
            ],
            baseline_hypervolume: 0.0,
            scheduler_hypervolume: 0.0,
            hypervolume_delta: 0.0,
            auc_metrics: CostAdrAucMetrics {
                baseline_point_count: 0,
                scheduler_point_count: 0,
                baseline_frontier_count: 0,
                scheduler_frontier_count: 0,
                target_count: 0,
                covered_target_count: 0,
                total_span: 0.0,
                covered_span: 0.0,
                span_coverage_percent: 0.0,
                same_target_time_saved_auc: None,
                baseline_time_auc: None,
                relative_same_target_time_saved_auc_percent: None,
            },
        };

        let weight = result
            .cost_weight_for_average_desired_retention(0.8)
            .unwrap();

        assert!((weight - 9.0).abs() < 1e-5);
        assert_eq!(result.cost_weight_for_average_desired_retention(0.95), None);
        assert_eq!(
            result.cost_weight_for_average_desired_retention(f32::NAN),
            None
        );
    }

    #[test]
    fn test_fsrs7_policy_input_is_internal_stability_not_s90() -> Result<()> {
        let fsrs = FSRS::new(&DEFAULT_PARAMETERS)?;
        let policy = CostAdrPolicy::default_initial();
        let state = MemoryState {
            stability: 10.0,
            difficulty: 5.0,
        };
        let s90 = fsrs.s90(state);

        assert!((s90 - state.stability).abs() > 1e-3);
        let internal_retention = policy.evaluate_retention(state.stability, state.difficulty, 64.0);
        let s90_retention = policy.evaluate_retention(s90, state.difficulty, 64.0);
        assert!((internal_retention - s90_retention).abs() > 1e-6);
        Ok(())
    }

    #[test]
    fn test_constant_policy_matches_fixed_retention_simulation() -> Result<()> {
        let policy = CostAdrPolicy::constant_retention(0.9)?;
        let config = SimulatorConfig {
            deck_size: 200,
            learn_span: 30,
            learn_limit: 20,
            review_limit: 200,
            ..Default::default()
        };
        let fixed = simulate(&config, &DEFAULT_PARAMETERS, 0.9, Some(7), None)?;
        let dynamic = crate::simulate_with_cost_adr_policy(
            &config,
            &DEFAULT_PARAMETERS,
            &policy,
            0.0,
            Some(7),
            None,
        )?;
        let dynamic_evaluation = simulate_with_cost_adr_policy_for_evaluation(
            &config,
            &DEFAULT_PARAMETERS,
            &policy,
            0.0,
            Some(7),
            None,
        )?;
        assert_eq!(fixed.review_cnt_per_day, dynamic.review_cnt_per_day);
        assert_eq!(fixed.learn_cnt_per_day, dynamic.learn_cnt_per_day);
        assert_eq!(fixed.cost_per_day, dynamic.cost_per_day);
        assert!((dynamic_evaluation.average_desired_retention.unwrap() - 0.9).abs() < 1e-4);
        Ok(())
    }

    #[test]
    fn test_cost_adr_next_states_matches_constant_retention() -> Result<()> {
        let fsrs = FSRS::new(&DEFAULT_PARAMETERS)?;
        let policy = CostAdrPolicy::constant_retention(0.9)?;
        let previous_state = Some(MemoryState {
            stability: 7.0,
            difficulty: 5.0,
        });

        let fixed = fsrs.next_states_with_elapsed_days(previous_state, 0.9, 7.0)?;
        let dynamic = policy.next_states(&fsrs, previous_state, 64.0, 7.0)?;

        assert_eq!(fixed.again.memory, dynamic.again.memory);
        assert_eq!(fixed.hard.memory, dynamic.hard.memory);
        assert_eq!(fixed.good.memory, dynamic.good.memory);
        assert_eq!(fixed.easy.memory, dynamic.easy.memory);
        assert!((fixed.good.interval - dynamic.good.interval).abs() < 1e-3);
        assert!((dynamic.good.desired_retention - 0.9).abs() < 1e-4);
        Ok(())
    }

    #[test]
    fn test_cost_adr_next_states_clamps_policy_max_interval() -> Result<()> {
        let fsrs = FSRS::new(&DEFAULT_PARAMETERS)?;
        let policy = CostAdrPolicy::new_with_settings(
            COST_ADR_DEFAULT_INITIAL_COEFFICIENTS.to_vec(),
            0.0,
            1024.0,
            0.30,
            0.995,
            Some(3.0),
        )?;
        let previous_state = Some(MemoryState {
            stability: 100.0,
            difficulty: 5.0,
        });

        let states = policy.next_states(&fsrs, previous_state, 64.0, 7.0)?;

        assert!(states.good.interval <= 3.0);
        assert!(states.good.interval >= 1.0);
        Ok(())
    }

    #[test]
    fn test_train_cost_adr_single_user_smoke() -> Result<()> {
        let config = SimulatorConfig {
            deck_size: 160,
            learn_span: 20,
            learn_limit: 20,
            review_limit: 200,
            ..Default::default()
        };
        let training_config = CostAdrTrainingConfig {
            population_size: 4,
            generations: 2,
            sigma0: 0.5,
            cost_weights: vec![0.0, 16.0],
            baseline_desired_retentions: vec![0.8, 0.9],
            ..Default::default()
        };
        let result =
            CostAdrPolicy::train_single_user(&config, &DEFAULT_PARAMETERS, &training_config)?;
        assert_eq!(result.policy.coefficients.len(), COST_ADR_PARAMETER_COUNT);
        assert_eq!(result.best_cost_weight_metrics.len(), 2);
        assert_eq!(result.history.len(), 2);
        assert!(result.training_seconds >= 0.0);
        Ok(())
    }

    #[test]
    fn test_train_cost_adr_uses_configured_retention_bounds() -> Result<()> {
        let config = SimulatorConfig {
            deck_size: 80,
            learn_span: 10,
            learn_limit: 20,
            review_limit: 200,
            ..Default::default()
        };
        let training_config = CostAdrTrainingConfig {
            population_size: 2,
            generations: 1,
            sigma0: 0.5,
            retention_min: 0.75,
            retention_max: 0.95,
            cost_weights: vec![0.0],
            baseline_desired_retentions: vec![0.9],
            ..Default::default()
        };
        let result =
            CostAdrPolicy::train_single_user(&config, &DEFAULT_PARAMETERS, &training_config)?;

        assert_eq!(result.policy.retention_min, 0.75);
        assert_eq!(result.policy.retention_max, 0.95);
        Ok(())
    }

    #[test]
    fn test_train_cost_adr_updates_progress() -> Result<()> {
        let progress = CombinedProgressState::new_shared();
        let config = SimulatorConfig {
            deck_size: 80,
            learn_span: 10,
            learn_limit: 20,
            review_limit: 200,
            ..Default::default()
        };
        let training_config = CostAdrTrainingConfig {
            population_size: 2,
            generations: 2,
            sigma0: 0.5,
            cost_weights: vec![0.0],
            baseline_desired_retentions: vec![0.9],
            progress: Some(progress.clone()),
            ..Default::default()
        };

        CostAdrPolicy::train_single_user(&config, &DEFAULT_PARAMETERS, &training_config)?;

        let progress = progress.lock().unwrap();
        assert!(progress.finished());
        assert_eq!(progress.current(), 4);
        assert_eq!(progress.total(), 4);
        Ok(())
    }

    #[test]
    fn test_train_cost_adr_progress_can_abort() {
        let progress = CombinedProgressState::new_shared();
        progress.lock().unwrap().want_abort = true;
        let config = SimulatorConfig {
            deck_size: 80,
            learn_span: 10,
            learn_limit: 20,
            review_limit: 200,
            ..Default::default()
        };
        let training_config = CostAdrTrainingConfig {
            population_size: 2,
            generations: 2,
            sigma0: 0.5,
            cost_weights: vec![0.0],
            baseline_desired_retentions: vec![0.9],
            progress: Some(progress.clone()),
            ..Default::default()
        };

        let result =
            CostAdrPolicy::train_single_user(&config, &DEFAULT_PARAMETERS, &training_config);

        assert_eq!(result, Err(FSRSError::Interrupted));
        assert!(progress.lock().unwrap().finished());
    }

    #[test]
    fn test_clamp_coefficients_applies_training_bounds() {
        let coefficients = vec![-2.0, -0.5, 0.5, 2.0];
        assert_eq!(
            clamp_coefficients(&coefficients, -1.0, 1.0),
            vec![-1.0, -0.5, 0.5, 1.0]
        );
    }

    #[test]
    fn test_train_cost_adr_clamps_out_of_bounds_initial_coefficients() -> Result<()> {
        let config = SimulatorConfig {
            deck_size: 120,
            learn_span: 15,
            learn_limit: 20,
            review_limit: 200,
            ..Default::default()
        };
        let training_config = CostAdrTrainingConfig {
            population_size: 2,
            generations: 1,
            sigma0: 0.5,
            lower_bound: -1.0,
            upper_bound: 1.0,
            initial_coefficients: vec![10.0; COST_ADR_PARAMETER_COUNT],
            cost_weights: vec![0.0, 16.0],
            baseline_desired_retentions: vec![0.8, 0.9],
            ..Default::default()
        };
        let result =
            CostAdrPolicy::train_single_user(&config, &DEFAULT_PARAMETERS, &training_config)?;
        assert!(
            result
                .policy
                .coefficients
                .iter()
                .all(|value| (-1.0..=1.0).contains(value))
        );
        Ok(())
    }

    #[test]
    fn test_evaluate_cost_adr_policy_returns_baseline_and_scheduler_metrics() -> Result<()> {
        let config = SimulatorConfig {
            deck_size: 120,
            learn_span: 15,
            learn_limit: 20,
            review_limit: 200,
            ..Default::default()
        };
        let policy = CostAdrPolicy::default_initial();
        let evaluation_config = CostAdrEvaluationConfig {
            cost_weights: vec![0.0, 16.0],
            baseline_desired_retentions: vec![0.8, 0.9],
            seed: Some(11),
        };
        let result = policy.evaluate(&config, &DEFAULT_PARAMETERS, &evaluation_config)?;
        assert_eq!(result.baseline_metrics.len(), 2);
        assert_eq!(result.scheduler_metrics.len(), 2);
        assert!(result.baseline_hypervolume.is_finite());
        assert!(result.scheduler_hypervolume.is_finite());
        assert!(result.hypervolume_delta.is_finite());
        for point in &result.scheduler_metrics {
            let average_desired_retention = point.average_desired_retention.unwrap();
            assert!((0.30..=0.995).contains(&average_desired_retention));
        }
        assert_eq!(result.auc_metrics.baseline_point_count, 2);
        assert_eq!(result.auc_metrics.scheduler_point_count, 2);
        Ok(())
    }

    #[test]
    fn test_cost_adr_none_seed_uses_default_seed() -> Result<()> {
        let config = SimulatorConfig {
            deck_size: 120,
            learn_span: 15,
            learn_limit: 20,
            review_limit: 200,
            ..Default::default()
        };
        let policy = CostAdrPolicy::default_initial();
        let default_seed = CostAdrEvaluationConfig {
            cost_weights: vec![0.0, 16.0],
            baseline_desired_retentions: vec![0.8, 0.9],
            seed: None,
        };
        let explicit_seed = CostAdrEvaluationConfig {
            seed: Some(COST_ADR_DEFAULT_SEED),
            ..default_seed.clone()
        };

        let default_result = policy.evaluate(&config, &DEFAULT_PARAMETERS, &default_seed)?;
        let explicit_result = policy.evaluate(&config, &DEFAULT_PARAMETERS, &explicit_seed)?;

        assert_eq!(default_result, explicit_result);
        Ok(())
    }

    #[test]
    fn test_default_baseline_retention_grid_has_sixteen_fixed_points() {
        assert_eq!(COST_ADR_DEFAULT_BASELINE_RETENTIONS.len(), 16);
        assert_eq!(COST_ADR_DEFAULT_BASELINE_RETENTIONS[0], 0.50);
        assert_eq!(COST_ADR_DEFAULT_BASELINE_RETENTIONS[15], 0.95);
    }

    #[test]
    fn test_cost_adr_auc_metrics_same_target_time_saved() {
        let baseline = vec![test_metrics(100.0, 10.0), test_metrics(200.0, 20.0)];
        let scheduler = vec![test_metrics(100.0, 8.0), test_metrics(200.0, 18.0)];
        let auc = cost_adr_auc_metrics(&baseline, &scheduler);
        assert_eq!(auc.baseline_frontier_count, 2);
        assert_eq!(auc.scheduler_frontier_count, 2);
        assert!((auc.span_coverage_percent - 100.0).abs() < 1e-5);
        assert!((auc.same_target_time_saved_auc.unwrap() - 2.0).abs() < 1e-5);
        assert!((auc.baseline_time_auc.unwrap() - 15.0).abs() < 1e-5);
        assert!(
            (auc.relative_same_target_time_saved_auc_percent.unwrap() - 13.333_333).abs() < 1e-4
        );
    }
}

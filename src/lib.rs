#![allow(clippy::single_range_in_vec_init)]

mod analytic_v7;
mod batch_shuffle;
#[cfg(test)]
mod convertor_tests;
mod cosine_annealing;
mod cost_adr;
mod dataset;
mod error;
mod inference;
mod model;
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
mod neon_math;
mod parameter_clipper;
mod parameter_initialization;
mod parameter_initialization_fsrs7;
mod simulation;
#[cfg(test)]
mod test_helpers;
mod training;

pub use cost_adr::{
    COST_ADR_POLICY_VERSION, CostAdrBounds, CostAdrEvaluationConfig, CostAdrEvaluationPoint,
    CostAdrEvaluationResult, CostAdrItemState, CostAdrMetrics, CostAdrNextStates, CostAdrPlotPoint,
    CostAdrPolicy, CostAdrTrainingConfig, CostAdrTrainingResult,
};
pub use dataset::{FSRSItem, FSRSReview, filter_outlier};
pub use error::{FSRSError, Result};
pub use inference::{
    DEFAULT_PARAMETERS, FSRS5_DEFAULT_DECAY, FSRS6_DEFAULT_DECAY, FSRS6_DEFAULT_PARAMETERS,
    ItemProgress, ItemState, MemoryState, ModelEvaluation, NextStates, current_retrievability,
    evaluate_with_time_series_splits,
};
pub use model::FSRS;
pub use simulation::simulate_with_cost_adr_policy;
pub use simulation::{
    CMRRTargetFn, Card, IntervalBucketConfig, IntervalBucketStats, IntervalBucketSummary,
    PostSchedulingFn, ReviewPriorityFn, ReviewRatingCostFn, RevlogEntry, RevlogReviewKind,
    SimulationResult, SimulatorConfig, expected_workload, expected_workload_with_existing_cards,
    extract_simulator_config, optimal_retention, simulate, simulate_cost_adr_interval_bucket_stats,
};
pub use training::{
    CombinedProgressState, ComputeParametersInput, ComputeParametersVersion, benchmark,
    compute_parameters,
};

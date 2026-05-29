#![allow(clippy::single_range_in_vec_init)]

mod batch_shuffle;
#[cfg(test)]
mod convertor_tests;
mod cosine_annealing;
mod cost_adr;
mod dataset;
mod error;
mod inference;
mod model;
mod parameter_clipper;
mod parameter_initialization;
mod simulation;
#[cfg(test)]
mod test_helpers;
mod training;

pub use cost_adr::{
    CostAdrAucMetrics, CostAdrBounds, CostAdrEvaluationConfig, CostAdrEvaluationPoint,
    CostAdrEvaluationResult, CostAdrGenerationMetrics, CostAdrMetrics, CostAdrPolicy,
    CostAdrTrainingConfig, CostAdrTrainingResult, evaluate_cost_adr_policy,
    train_cost_adr_single_user,
};
pub use dataset::{FSRSItem, FSRSReview, filter_outlier};
pub use error::{FSRSError, Result};
pub use inference::{
    DEFAULT_PARAMETERS, FSRS5_DEFAULT_DECAY, FSRS6_DEFAULT_DECAY, ItemProgress, ItemState,
    MemoryState, ModelEvaluation, NextStates, current_retrievability,
    evaluate_with_time_series_splits,
};
pub use model::FSRS;
pub use simulation::{
    CMRRTargetFn, Card, PostSchedulingFn, ReviewPriorityFn, RevlogEntry, RevlogReviewKind,
    SimulationResult, SimulatorConfig, expected_workload, expected_workload_with_existing_cards,
    extract_simulator_config, optimal_retention, simulate, simulate_with_cost_adr_policy,
};
pub use training::{CombinedProgressState, ComputeParametersInput, benchmark, compute_parameters};

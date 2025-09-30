#![allow(clippy::single_range_in_vec_init)]

mod batch_shuffle;
#[cfg(test)]
mod convertor_tests;
mod cosine_annealing;
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

pub use dataset::{FSRSItem, FSRSReview};
pub use error::{FSRSError, Result};
pub use inference::{
    DEFAULT_PARAMETERS, FSRS5_DEFAULT_DECAY, FSRS6_DEFAULT_DECAY, ItemProgress, ItemState,
    MemoryState, ModelEvaluation, NextStates, current_retrievability,
};
pub use model::FSRS;
pub use simulation::{
    CMRRTargetFn, Card, PostSchedulingFn, ReviewPriorityFn, RevlogEntry, RevlogReviewKind,
    SimulationResult, SimulatorConfig, expected_workload, expected_workload_with_existing_cards,
    extract_simulator_config, optimal_retention, simulate,
};
pub use training::{CombinedProgressState, ComputeParametersInput};

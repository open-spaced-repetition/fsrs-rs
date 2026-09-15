//! <div align="center">
//!
//! # FSRS-rs
//!
//! [![crates.io](https://img.shields.io/crates/v/fsrs.svg)](https://crates.io/crates/fsrs) ![](https://github.com/open-spaced-repetition/fsrs-rs/actions/workflows/check.yml/badge.svg)
//!
//! The Free Spaced Repetition Scheduler ([FSRS](https://github.com/open-spaced-repetition/fsrs4anki/wiki/The-Algorithm)) is a modern spaced repetition algorithm. It springs from [MaiMemo's DHP model](https://www.maimemo.com/paper/), which is a variant of the [DSR model](https://supermemo.guru/wiki/Three_component_model_of_memory) proposed by [Piotr Wozniak](https://supermemo.guru/wiki/Piotr_Wozniak).
//!
//! FSRS-rs is a Rust implementation of FSRS. It also provides simulation capabilities and basic scheduling functionality.
//!
//! For more information about the algorithm, please refer to [the wiki page of FSRS](https://github.com/open-spaced-repetition/fsrs4anki/wiki/The-Algorithm).
//! </div>
//!
//! ---
//!
//! Most of the functions in this crate require a struct input.
//! e.g. [`ComputeParametersInput`] and [`FSRSItem`].
//!
//! The most common struct is [`FSRS`] which holds the user's parameters.
//!
//! # Examples
//! ```
//! use chrono::{Duration, Utc};
//! use fsrs::{FSRS, MemoryState};
//!
//! let fsrs = FSRS::default();
//! let desired_retention = 0.9;
//! let previous_state: Option<MemoryState> = None;
//! let elapsed_days = 0;
//!
//! let next_states = fsrs.next_states(previous_state, desired_retention, elapsed_days).unwrap();
//! let review = next_states.good;
//!
//! let interval_days = review.interval.round().max(1.0) as u32;
//! let due = Utc::now() + Duration::days(interval_days as i64);
//! ```
//!
//! There are more functions and structures.
//! You can find them [here](https://github.com/open-spaced-repetition/fsrs-rs/blob/main/src/lib.rs).

#![allow(clippy::single_range_in_vec_init)]
#![allow(dead_code, unused_imports)]

mod analytic_v6;
mod analytic_v7;
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
    CostAdrEvaluationResult, CostAdrFixedTargetCalibrationPoint, CostAdrItemState, CostAdrMetrics,
    CostAdrNextStates, CostAdrPlotPoint, CostAdrPolicy, CostAdrTrainingConfig,
    CostAdrTrainingResult,
};
pub use dataset::{FSRSItem, FSRSReview, filter_outlier};
pub use error::{FSRSError, Result};
pub use inference::{
    DEFAULT_PARAMETERS, FSRS5_DEFAULT_DECAY, FSRS6_DEFAULT_DECAY, FSRS6_DEFAULT_PARAMETERS,
    ItemProgress, ItemState, MemoryState, ModelEvaluation, NextStates, current_retrievability,
    evaluate_with_time_series_splits,
};
pub use model::{FSRS, check_and_fill_parameters};
pub use simulation::simulate_with_cost_adr_policy;
pub use simulation::{
    CMRRTargetFn, Card, IntervalBucketConfig, IntervalBucketStats, IntervalBucketSummary,
    PostSchedulingContext, PostSchedulingFn, ReviewPriorityFn, ReviewRatingCostFn, RevlogEntry,
    RevlogReviewKind, SimulationEvent, SimulationResult, SimulationSummaryResult,
    SimulatorCardUpdateFn, SimulatorCardUpdatePhase, SimulatorConfig, SimulatorEventFn,
    expected_workload, expected_workload_with_existing_cards, extract_simulator_config,
    optimal_retention, simulate, simulate_cost_adr_interval_bucket_stats, simulate_summary,
    simulate_summary_with_card_update_and_event_fn, simulate_summary_with_card_update_fn,
    simulate_with_card_update_fn,
};
pub use training::{
    CombinedProgressState, ComputeParametersInput, ComputeParametersVersion,
    RecallClassificationCosts, RecallClassifierTrainingConfig, TrainingConfig, benchmark,
    compute_parameters, compute_parameters_for_recall_classifier,
};

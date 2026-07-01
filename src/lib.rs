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
//! The most of functions in this crate require a struct input.
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
#![allow(clippy::needless_range_loop)]
mod analytic;
#[cfg(test)]
mod batch_shuffle;
#[cfg(test)]
mod convertor_tests;
mod cosine_annealing;
#[cfg(feature = "experimental_cost_adr")]
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

#[cfg(feature = "experimental_cost_adr")]
pub use cost_adr::{
    CostAdrEvaluationConfig, CostAdrEvaluationResult, CostAdrItemState, CostAdrNextStates,
    CostAdrPolicy, CostAdrTrainingConfig, CostAdrTrainingResult,
};
// Provide the basic data structures.
pub use dataset::{FSRSItem, FSRSReview, filter_outlier};
/// Error types and Result alias for the crate.
pub use error::{FSRSError, Result};
pub use inference::{
    DEFAULT_PARAMETERS, FSRS5_DEFAULT_DECAY, FSRS6_DEFAULT_DECAY, ItemProgress, ItemState,
    MemoryState, ModelEvaluation, NextStates, current_retrievability,
    evaluate_with_time_series_splits,
};
// Provide the main model. It's the most commanly used part.
pub use model::{FSRS, check_and_fill_parameters};
#[cfg(feature = "experimental_cost_adr")]
pub use simulation::simulate_with_cost_adr_policy;
// Simulate long-term scheduling outcomes without real users.
pub use simulation::{
    CMRRTargetFn, Card, PostSchedulingContext, PostSchedulingFn, ReviewPriorityFn, RevlogEntry,
    RevlogReviewKind, SimulationResult, SimulatorConfig, expected_workload,
    expected_workload_with_existing_cards, extract_simulator_config, optimal_retention, simulate,
};
// Compute optimal model parameters from review history.
pub use training::{
    CombinedProgressState, ComputeParametersInput, TrainingConfig, benchmark, compute_parameters,
};

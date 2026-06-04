use crate::DEFAULT_PARAMETERS;
use crate::cost_adr::CostAdrPolicy;
use crate::error::{FSRSError, Result};
use crate::inference::{ItemProgress, Parameters};
use crate::model::{ModelVersion, check_and_fill_parameters, model_v6, model_v7};
use itertools::{Itertools, izip};
use ndarray::{Array1, Array2, Array3};
use rand::distr::Distribution;
use rand::distr::weighted::WeightedIndex;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::iter::IntoParallelIterator;
use rayon::iter::ParallelIterator;
use std::cmp::{Ordering, Reverse};
use std::collections::HashMap;
use std::ops::Deref;
use std::sync::{Arc, LazyLock};

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
mod neon;

#[derive(Debug)]
pub struct SimulationResult {
    pub memorized_cnt_per_day: Vec<f32>,
    pub review_cnt_per_day: Vec<usize>,
    pub learn_cnt_per_day: Vec<usize>,
    pub cost_per_day: Vec<f32>,
    // The amount of review cards you got correct on a given day (not including learn cards).
    pub correct_cnt_per_day: Vec<usize>,
    pub introduced_cnt_per_day: Vec<usize>,
    pub cards: Vec<Card>,
}

pub(crate) struct CostAdrSimulationResult {
    pub result: SimulationResult,
    pub average_desired_retention: Option<f32>,
}

trait Round {
    fn to_2_decimal(self) -> f32;
}

impl Round for f32 {
    fn to_2_decimal(self) -> f32 {
        (self * 100.0).round() / 100.0
    }
}

pub(crate) const S_MIN: f32 = 0.0001;
pub(crate) const S_MAX: f32 = 36500.0;
pub(crate) const D_MIN: f32 = 1.0;
pub(crate) const D_MAX: f32 = 10.0;
pub(crate) const R_MIN: f32 = 0.70;
pub(crate) const R_MAX: f32 = 0.95;
pub(crate) const RATINGS: [usize; 4] = [1, 2, 3, 4];
pub(crate) const LEARNING: usize = 0;
pub(crate) const REVIEW: usize = 1;
pub(crate) const RELEARNING: usize = 2;
pub(crate) const MAX_STEPS: usize = 5;

/// Function type for post scheduling operations that takes interval, maximum interval,
/// current day index, due counts per day, and a random number generator,
/// and returns a new interval.
#[allow(clippy::type_complexity)]
pub struct PostSchedulingFn(
    Arc<dyn Fn(&Card, f32, usize, &[usize], &mut StdRng) -> f32 + Sync + Send>,
);

impl Deref for PostSchedulingFn {
    type Target = dyn Fn(&Card, f32, usize, &[usize], &mut StdRng) -> f32 + Sync + Send;

    fn deref(&self) -> &Self::Target {
        &*self.0
    }
}

impl PartialEq for PostSchedulingFn {
    fn eq(&self, _: &Self) -> bool {
        true
    }
}

impl std::fmt::Debug for PostSchedulingFn {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Wrap(<function>)")
    }
}

/// Function type for review priority calculation that takes a card reference
/// and returns a priority value (lower value means higher priority)
#[derive(Clone)]
#[allow(clippy::type_complexity)]
pub struct ReviewPriorityFn(Arc<dyn Fn(&Card) -> i32 + Sync + Send>);

impl PartialEq for ReviewPriorityFn {
    fn eq(&self, _: &Self) -> bool {
        true
    }
}

impl Deref for ReviewPriorityFn {
    type Target = dyn Fn(&Card) -> i32 + Sync + Send;

    fn deref(&self) -> &Self::Target {
        &*self.0
    }
}

impl ReviewPriorityFn {
    pub fn new(f: impl Fn(&Card) -> i32 + Sync + Send + 'static) -> Self {
        Self(Arc::new(f))
    }
}

impl std::fmt::Debug for ReviewPriorityFn {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Wrap(<function>)")
    }
}

impl Default for ReviewPriorityFn {
    fn default() -> Self {
        Self(Arc::new(|card| (card.difficulty * 100.0) as i32))
    }
}

type CardPriority = Reverse<(i32, bool, i32)>;

struct CardPriorityQueue {
    heap: Vec<usize>,
    positions: Vec<usize>,
    priorities: Vec<CardPriority>,
    active: Vec<bool>,
}

impl CardPriorityQueue {
    fn with_capacity(capacity: usize) -> Self {
        Self {
            heap: Vec::with_capacity(capacity),
            positions: vec![0; capacity],
            priorities: vec![Reverse((0, false, 0)); capacity],
            active: vec![false; capacity],
        }
    }

    fn push(&mut self, card_index: usize, priority: CardPriority) {
        debug_assert!(!self.active[card_index]);
        self.active[card_index] = true;
        self.priorities[card_index] = priority;
        let position = self.heap.len();
        self.positions[card_index] = position;
        self.heap.push(card_index);
        self.bubble_up(position);
    }

    fn peek_index(&self) -> Option<usize> {
        self.heap.first().copied()
    }

    fn pop(&mut self) {
        if self.heap.is_empty() {
            return;
        }

        let card_index = self.heap.swap_remove(0);
        self.active[card_index] = false;
        if let Some(&root_card_index) = self.heap.first() {
            self.positions[root_card_index] = 0;
            self.heapify(0);
        }
    }

    fn change_priority(&mut self, card_index: usize, priority: CardPriority) {
        debug_assert!(self.active[card_index]);
        self.priorities[card_index] = priority;
        let position = self.positions[card_index];
        let position = self.bubble_up(position);
        self.heapify(position);
    }

    fn bubble_up(&mut self, mut position: usize) -> usize {
        let card_index = self.heap[position];
        let priority = self.priorities[card_index];

        while position > 0 {
            let parent = (position - 1) / 2;
            let parent_index = self.heap[parent];
            if self.priorities[parent_index].cmp(&priority) != Ordering::Less {
                break;
            }
            self.heap[position] = parent_index;
            self.positions[parent_index] = position;
            position = parent;
        }

        self.heap[position] = card_index;
        self.positions[card_index] = position;
        position
    }

    fn heapify(&mut self, mut position: usize) {
        loop {
            let left = 2 * position + 1;
            if left >= self.heap.len() {
                break;
            }

            let right = left + 1;
            let mut largest = position;
            let mut largest_priority = self.priorities[self.heap[position]];

            let left_priority = self.priorities[self.heap[left]];
            if left_priority > largest_priority {
                largest = left;
                largest_priority = left_priority;
            }

            if right < self.heap.len() && self.priorities[self.heap[right]] > largest_priority {
                largest = right;
            }

            if largest == position {
                break;
            }

            self.heap.swap(position, largest);
            self.positions[self.heap[position]] = position;
            self.positions[self.heap[largest]] = largest;
            position = largest;
        }
    }
}

#[derive(Clone)]
#[allow(clippy::type_complexity)]
pub struct ReviewRatingCostFn(Arc<dyn Fn(&Card, usize, f32) -> f32 + Sync + Send>);

impl PartialEq for ReviewRatingCostFn {
    fn eq(&self, _: &Self) -> bool {
        true
    }
}

impl Deref for ReviewRatingCostFn {
    type Target = dyn Fn(&Card, usize, f32) -> f32 + Sync + Send;

    fn deref(&self) -> &Self::Target {
        &*self.0
    }
}

impl ReviewRatingCostFn {
    pub fn new(f: impl Fn(&Card, usize, f32) -> f32 + Sync + Send + 'static) -> Self {
        Self(Arc::new(f))
    }
}

impl std::fmt::Debug for ReviewRatingCostFn {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Wrap(<function>)")
    }
}

#[allow(clippy::type_complexity)]
pub struct CMRRTargetFn(Arc<dyn Fn(&SimulationResult, &[f32]) -> f32 + Sync + Send>);

impl Deref for CMRRTargetFn {
    type Target = dyn Fn(&SimulationResult, &[f32]) -> f32 + Sync + Send;

    fn deref(&self) -> &Self::Target {
        &*self.0
    }
}

impl std::fmt::Debug for CMRRTargetFn {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Wrap(<function>)")
    }
}

impl Default for CMRRTargetFn {
    fn default() -> Self {
        Self(Arc::new(|result, _w| {
            let SimulationResult {
                memorized_cnt_per_day,
                cost_per_day,
                ..
            } = result;

            let total_memorized = memorized_cnt_per_day[memorized_cnt_per_day.len() - 1];
            let total_cost = cost_per_day.iter().sum::<f32>();
            total_cost / total_memorized
        }))
    }
}

#[derive(Debug, PartialEq)]
pub struct SimulatorConfig {
    pub deck_size: usize,
    pub learn_span: usize,
    pub max_cost_perday: f32,
    pub max_ivl: f32,
    pub first_rating_prob: [f32; 4],
    pub review_rating_prob: [f32; 3],
    pub learn_limit: usize,
    pub review_limit: usize,
    pub new_cards_ignore_review_limit: bool,
    pub suspend_after_lapses: Option<u32>,
    pub post_scheduling_fn: Option<PostSchedulingFn>,
    pub review_priority_fn: Option<ReviewPriorityFn>,
    pub review_rating_cost_fn: Option<ReviewRatingCostFn>,
    pub learning_step_transitions: [[f32; 4]; 3],
    pub relearning_step_transitions: [[f32; 4]; 3],
    pub state_rating_costs: [[f32; 4]; 3],
    pub learning_step_count: usize,
    pub relearning_step_count: usize,
}

impl Default for SimulatorConfig {
    fn default() -> Self {
        Self {
            deck_size: 10000,
            learn_span: 365,
            max_cost_perday: 1800.0,
            max_ivl: 36500.0,
            first_rating_prob: [0.24, 0.094, 0.495, 0.171],
            review_rating_prob: [0.224, 0.631, 0.145],
            learn_limit: usize::MAX,
            review_limit: usize::MAX,
            new_cards_ignore_review_limit: true,
            suspend_after_lapses: None,
            post_scheduling_fn: None,
            review_priority_fn: None,
            review_rating_cost_fn: None,
            learning_step_transitions: [
                [0.3686, 0.0628, 0.5108, 0.0577],
                [0.0442, 0.4553, 0.4457, 0.0549],
                [0.0519, 0.047, 0.8462, 0.055],
            ],
            relearning_step_transitions: [
                [0.2157, 0.0643, 0.6595, 0.0604],
                [0.05, 0.4638, 0.4475, 0.0387],
                [0.1057, 0.1434, 0.7266, 0.0244],
            ],
            state_rating_costs: [
                [19.58, 18.79, 13.78, 10.71],
                [19.38, 17.59, 12.38, 8.94],
                [16.44, 15.25, 12.32, 8.03],
            ],
            learning_step_count: 2,
            relearning_step_count: 1,
        }
    }
}

fn init_s(w: &[f32], rating: usize) -> f32 {
    w[rating - 1]
}

trait SimulatedFsrs {
    fn stability_after_success(
        &self,
        w: &[f32],
        s: f32,
        r: f32,
        d: f32,
        rating: usize,
        delta_t: f32,
    ) -> f32;
    fn stability_after_failure(&self, w: &[f32], s: f32, r: f32, d: f32, delta_t: f32) -> f32;
    fn stability_short_term(&self, w: &[f32], s: f32, d: f32, rating: usize) -> f32;
    fn init_d(&self, w: &[f32], rating: usize) -> f32;
    fn next_d(&self, w: &[f32], d: f32, rating: usize) -> f32;
    fn power_forgetting_curve(&self, w: &[f32], t: f32, s: f32) -> f32;
    fn next_interval(
        &self,
        w: &[f32],
        stability: f32,
        desired_retention: f32,
        fsrs7_runtime: Option<&model_v7::Fsrs7Runtime>,
    ) -> f32;
}

struct SimulatedFsrs7;
struct LegacySimulatedFsrs;

impl SimulatedFsrs for SimulatedFsrs7 {
    fn stability_after_success(
        &self,
        w: &[f32],
        s: f32,
        r: f32,
        d: f32,
        rating: usize,
        delta_t: f32,
    ) -> f32 {
        model_v7::stability_after_success_scalar(w, s, r, d, rating, delta_t)
    }

    fn stability_after_failure(&self, w: &[f32], s: f32, r: f32, d: f32, delta_t: f32) -> f32 {
        model_v7::stability_after_failure_scalar(w, s, r, d, delta_t)
    }

    fn stability_short_term(&self, w: &[f32], s: f32, d: f32, rating: usize) -> f32 {
        model_v7::stability_short_term_scalar(w, s, 1.0, d, rating)
    }

    fn init_d(&self, w: &[f32], rating: usize) -> f32 {
        model_v7::init_difficulty_scalar(w, rating)
    }

    fn next_d(&self, w: &[f32], d: f32, rating: usize) -> f32 {
        model_v7::next_difficulty_scalar(w, d, rating)
    }

    fn power_forgetting_curve(&self, w: &[f32], t: f32, s: f32) -> f32 {
        model_v7::fsrs7_forgetting_curve_scalar(w, t, s)
    }

    fn next_interval(
        &self,
        w: &[f32],
        stability: f32,
        desired_retention: f32,
        fsrs7_runtime: Option<&model_v7::Fsrs7Runtime>,
    ) -> f32 {
        if let Some(runtime) = fsrs7_runtime {
            runtime.next_interval(stability, desired_retention)
        } else {
            model_v7::Fsrs7Runtime::new(w).next_interval(stability, desired_retention)
        }
    }
}

impl SimulatedFsrs for LegacySimulatedFsrs {
    fn stability_after_success(
        &self,
        w: &[f32],
        s: f32,
        r: f32,
        d: f32,
        rating: usize,
        _delta_t: f32,
    ) -> f32 {
        model_v6::stability_after_success_scalar(w, s, r, d, rating)
    }

    fn stability_after_failure(&self, w: &[f32], s: f32, r: f32, d: f32, _delta_t: f32) -> f32 {
        model_v6::stability_after_failure_scalar(w, s, r, d)
    }

    fn stability_short_term(&self, w: &[f32], s: f32, _d: f32, rating: usize) -> f32 {
        model_v6::stability_short_term_scalar(w, s, rating)
    }

    fn init_d(&self, w: &[f32], rating: usize) -> f32 {
        model_v6::init_difficulty_scalar(w, rating)
    }

    fn next_d(&self, w: &[f32], d: f32, rating: usize) -> f32 {
        model_v6::next_difficulty_scalar(w, d, rating)
    }

    fn power_forgetting_curve(&self, w: &[f32], t: f32, s: f32) -> f32 {
        model_v6::power_forgetting_curve_scalar(w, t, s)
    }

    fn next_interval(
        &self,
        w: &[f32],
        stability: f32,
        desired_retention: f32,
        _fsrs7_runtime: Option<&model_v7::Fsrs7Runtime>,
    ) -> f32 {
        model_v6::next_interval_scalar(w, stability, desired_retention)
    }
}

static SIMULATED_FSRS7: SimulatedFsrs7 = SimulatedFsrs7;
static LEGACY_SIMULATED_FSRS: LegacySimulatedFsrs = LegacySimulatedFsrs;

fn simulated_fsrs(w: &[f32]) -> &'static dyn SimulatedFsrs {
    match ModelVersion::from_param_count(w.len()) {
        ModelVersion::Fsrs7 => &SIMULATED_FSRS7,
        ModelVersion::Fsrs6 => &LEGACY_SIMULATED_FSRS,
    }
}

fn stability_after_success_with_fsrs(
    fsrs: &dyn SimulatedFsrs,
    w: &[f32],
    s: f32,
    r: f32,
    d: f32,
    rating: usize,
    delta_t: f32,
) -> f32 {
    fsrs.stability_after_success(w, s, r, d, rating, delta_t)
}

#[cfg(test)]
#[allow(dead_code)]
fn stability_after_success(w: &[f32], s: f32, r: f32, d: f32, rating: usize, delta_t: f32) -> f32 {
    let fsrs = simulated_fsrs(w);
    stability_after_success_with_fsrs(fsrs, w, s, r, d, rating, delta_t)
}

fn stability_after_failure_with_fsrs(
    fsrs: &dyn SimulatedFsrs,
    w: &[f32],
    s: f32,
    r: f32,
    d: f32,
    delta_t: f32,
) -> f32 {
    fsrs.stability_after_failure(w, s, r, d, delta_t)
}

#[cfg(test)]
#[allow(dead_code)]
fn stability_after_failure(w: &[f32], s: f32, r: f32, d: f32, delta_t: f32) -> f32 {
    let fsrs = simulated_fsrs(w);
    stability_after_failure_with_fsrs(fsrs, w, s, r, d, delta_t)
}

fn stability_short_term_with_fsrs(
    fsrs: &dyn SimulatedFsrs,
    w: &[f32],
    s: f32,
    d: f32,
    rating: usize,
) -> f32 {
    fsrs.stability_short_term(w, s, d, rating)
}

#[cfg(test)]
#[allow(dead_code)]
fn stability_short_term(w: &[f32], s: f32, d: f32, rating: usize) -> f32 {
    let fsrs = simulated_fsrs(w);
    stability_short_term_with_fsrs(fsrs, w, s, d, rating)
}

#[allow(clippy::too_many_arguments)]
fn memory_state_short_term_with_fsrs(
    fsrs: &dyn SimulatedFsrs,
    w: &[f32],
    s: f32,
    d: f32,
    init_rating: Option<usize>,
    rating_costs: &[f32; 4],
    step_transitions: &[[f32; 4]; 3],
    step_count: usize,
    rng: &mut StdRng,
) -> (f32, f32, f32) {
    let mut consecutive = 0;
    let mut rating = init_rating.unwrap_or(1);
    let mut cost = if init_rating.is_none() {
        0.0
    } else {
        rating_costs[rating - 1]
    };
    let mut new_s = s;
    let mut new_d = d;
    let consecutive_max = if rating > 2 {
        step_count.saturating_sub(1)
    } else {
        step_count
    };
    for _ in 0..MAX_STEPS {
        if consecutive >= consecutive_max || rating >= 4 {
            break;
        }
        let next_rating_dist = WeightedIndex::new(step_transitions[rating - 1]).unwrap();
        rating = RATINGS[next_rating_dist.sample(rng)];
        new_s = stability_short_term_with_fsrs(fsrs, w, new_s, new_d, rating);
        new_d = next_d_with_fsrs(fsrs, w, new_d, rating);
        cost += rating_costs[rating - 1];
        if rating > 2 {
            consecutive += 1;
        } else if rating == 1 {
            consecutive = 0;
        }
    }
    (new_s.clamp(S_MIN, S_MAX), new_d.clamp(D_MIN, D_MAX), cost)
}

#[allow(clippy::too_many_arguments)]
#[cfg(test)]
#[allow(dead_code)]
fn memory_state_short_term(
    w: &[f32],
    s: f32,
    d: f32,
    init_rating: Option<usize>,
    rating_costs: &[f32; 4],
    step_transitions: &[[f32; 4]; 3],
    step_count: usize,
    rng: &mut StdRng,
) -> (f32, f32, f32) {
    let fsrs = simulated_fsrs(w);
    memory_state_short_term_with_fsrs(
        fsrs,
        w,
        s,
        d,
        init_rating,
        rating_costs,
        step_transitions,
        step_count,
        rng,
    )
}

fn init_d_with_fsrs(fsrs: &dyn SimulatedFsrs, w: &[f32], rating: usize) -> f32 {
    fsrs.init_d(w, rating)
}

#[cfg(test)]
#[allow(dead_code)]
fn init_d(w: &[f32], rating: usize) -> f32 {
    let fsrs = simulated_fsrs(w);
    init_d_with_fsrs(fsrs, w, rating)
}

fn next_d_with_fsrs(fsrs: &dyn SimulatedFsrs, w: &[f32], d: f32, rating: usize) -> f32 {
    fsrs.next_d(w, d, rating)
}

#[cfg(test)]
#[allow(dead_code)]
fn next_d(w: &[f32], d: f32, rating: usize) -> f32 {
    let fsrs = simulated_fsrs(w);
    next_d_with_fsrs(fsrs, w, d, rating)
}

fn power_forgetting_curve_with_fsrs(fsrs: &dyn SimulatedFsrs, w: &[f32], t: f32, s: f32) -> f32 {
    fsrs.power_forgetting_curve(w, t, s)
}

fn power_forgetting_curve_with_runtime(
    fsrs: &dyn SimulatedFsrs,
    w: &[f32],
    fsrs7_runtime: Option<&model_v7::Fsrs7Runtime>,
    t: f32,
    s: f32,
) -> f32 {
    fsrs7_runtime
        .map(|runtime| runtime.forgetting_curve(t, s))
        .unwrap_or_else(|| fsrs.power_forgetting_curve(w, t, s))
}

fn add_forgetting_curve_range_scalar(
    fsrs: &dyn SimulatedFsrs,
    w: &[f32],
    fsrs7_runtime: Option<&model_v7::Fsrs7Runtime>,
    days: &mut [f32],
    start_day: usize,
    last_date: f32,
    stability: f32,
) {
    for (offset, day) in days.iter_mut().enumerate() {
        *day += power_forgetting_curve_with_runtime(
            fsrs,
            w,
            fsrs7_runtime,
            (start_day + offset) as f32 - last_date,
            stability,
        );
    }
}

fn add_forgetting_curve_range(
    fsrs: &dyn SimulatedFsrs,
    w: &[f32],
    fsrs7_runtime: Option<&model_v7::Fsrs7Runtime>,
    memorized_cnt_per_day: &mut [f32],
    start_day: usize,
    end_day: usize,
    last_date: f32,
    stability: f32,
) {
    let end_day = end_day.min(memorized_cnt_per_day.len());
    let Some(days) = memorized_cnt_per_day.get_mut(start_day..end_day) else {
        return;
    };
    if days.is_empty() {
        return;
    }

    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    if ModelVersion::from_param_count(w.len()) == ModelVersion::Fsrs7 {
        neon::add_fsrs7_forgetting_curve_range(w, days, start_day, last_date, stability);
        return;
    }

    add_forgetting_curve_range_scalar(
        fsrs,
        w,
        fsrs7_runtime,
        days,
        start_day,
        last_date,
        stability,
    );
}

#[cfg(test)]
#[allow(dead_code)]
fn power_forgetting_curve(w: &[f32], t: f32, s: f32) -> f32 {
    debug_assert!(t >= 0.);
    let fsrs = simulated_fsrs(w);
    power_forgetting_curve_with_fsrs(fsrs, w, t, s)
}

fn next_interval_with_fsrs(
    fsrs: &dyn SimulatedFsrs,
    w: &[f32],
    stability: f32,
    desired_retention: f32,
    fsrs7_runtime: Option<&model_v7::Fsrs7Runtime>,
) -> f32 {
    fsrs.next_interval(w, stability, desired_retention, fsrs7_runtime)
}

#[cfg(test)]
#[allow(dead_code)]
fn next_interval(w: &[f32], stability: f32, desired_retention: f32) -> f32 {
    let fsrs = simulated_fsrs(w);
    let fsrs7_runtime = fsrs7_runtime(w);
    next_interval_with_fsrs(
        fsrs,
        w,
        stability,
        desired_retention,
        fsrs7_runtime.as_ref(),
    )
}

fn fsrs7_runtime(w: &[f32]) -> Option<model_v7::Fsrs7Runtime> {
    (ModelVersion::from_param_count(w.len()) == ModelVersion::Fsrs7)
        .then(|| model_v7::Fsrs7Runtime::new(w))
}

/// Dynamic programming-based workload estimator
#[derive(Debug)]
pub struct WorkloadEstimator {
    // State spaces
    s_state: Array1<f32>,
    d_state: Array1<f32>,
    s_size: usize,
    d_size: usize,
    t_size: usize,
    s_mid: f32,
    s_mid_size: usize,

    // Configuration
    short_step: f32,
    long_step: f32,

    // Review configuration
    review_rating_prob: [f32; 3],
    state_rating_costs: [[f32; 4]; 3],
    desired_retention: f32,
    cost_matrix: Array3<f32>,
}

impl WorkloadEstimator {
    pub fn new(config: &SimulatorConfig) -> Self {
        let s_max = 365.0;
        let short_step = 2.0f32.ln() / 25.0;
        let long_step = 5.0;
        let d_eps = 0.3;

        // Create stability state space
        let s_mid_target = (long_step / (1.0 - (-short_step).exp())).min(s_max);
        let log_s_target = s_mid_target.ln();
        let s_state_small = (0..)
            .map(|i| S_MIN.ln() + short_step * i as f32)
            .take_while(|&log_s| log_s < log_s_target)
            .map(|log_s| log_s.exp())
            .collect::<Vec<_>>();
        let s_mid_size = s_state_small.len();
        let s_mid = s_state_small[s_mid_size - 1];
        let s_state_large = (1..)
            .map(|i| s_mid + long_step * i as f32)
            .take_while(|&s| s < s_max)
            .collect::<Vec<_>>();
        let s_state = Array1::from_iter(s_state_small.into_iter().chain(s_state_large));
        let s_size = s_state.len();

        // Create difficulty state space
        let d_size = ((D_MAX - D_MIN) / d_eps + 1.0f32).ceil() as usize;
        let d_state = Array1::from_iter(
            (0..d_size).map(|i| D_MIN + (D_MAX - D_MIN) * i as f32 / (d_size - 1) as f32),
        );

        let t_size = config.learn_span;
        let cost_matrix = Array3::zeros((s_size, d_size, t_size + 1));

        Self {
            s_state,
            d_state,
            s_size,
            d_size,
            t_size,
            s_mid,
            s_mid_size,
            short_step,
            long_step,
            review_rating_prob: config.review_rating_prob,
            state_rating_costs: config.state_rating_costs,
            cost_matrix,
            desired_retention: 0.9,
        }
    }

    fn s2i(&self, s: f32) -> usize {
        let index = if s <= self.s_mid {
            // Handle small values (logarithmic scale)
            ((s.ln() - S_MIN.ln()) / self.short_step).ceil() as usize
        } else {
            // Handle large values (linear scale)
            self.s_mid_size - 1 + ((s - self.s_mid) / self.long_step).ceil() as usize
        };
        index.min(self.s_size - 1)
    }

    fn d2i(&self, d: f32) -> usize {
        let index = ((d - D_MIN) / (D_MAX - D_MIN) * self.d_size as f32).floor() as usize;
        index.min(self.d_size - 1)
    }

    fn precompute_cost_matrix(&mut self, desired_retention: f32, w: &Parameters) {
        self.desired_retention = desired_retention;
        let fsrs = simulated_fsrs(w);
        let fsrs7_runtime = fsrs7_runtime(w);
        // Cache precomputed values using ndarray
        let mut transition_probs = Array2::zeros((4, self.s_size));
        let mut next_s_indices = Array3::zeros((4, self.s_size, self.d_size));
        let mut next_d_indices = Array3::zeros((4, self.s_size, self.d_size));
        let mut next_intervals = Array3::zeros((4, self.s_size, self.d_size));

        // Precompute transitions for all state combinations
        for s_idx in 0..self.s_size {
            let s = self.s_state[s_idx];
            // Calculate interval and retrievability once and cache them
            let ivl =
                next_interval_with_fsrs(fsrs, w, s, desired_retention, fsrs7_runtime.as_ref())
                    .max(1.0)
                    .round();
            let r = power_forgetting_curve_with_runtime(fsrs, w, fsrs7_runtime.as_ref(), ivl, s);
            for rating in 1..=4 {
                if rating == 1 {
                    transition_probs[[rating - 1, s_idx]] = 1.0 - r;
                } else {
                    transition_probs[[rating - 1, s_idx]] = r * self.review_rating_prob[rating - 2];
                }
            }
            for d_idx in 0..self.d_size {
                let d = self.d_state[d_idx];
                for rating in 1..=4 {
                    let next_s = if rating == 1 {
                        stability_after_failure_with_fsrs(fsrs, w, s, r, d, ivl)
                    } else {
                        stability_after_success_with_fsrs(fsrs, w, s, r, d, rating, ivl)
                    };
                    let next_d_val = next_d_with_fsrs(fsrs, w, d, rating);
                    let next_ivl = next_interval_with_fsrs(
                        fsrs,
                        w,
                        next_s,
                        desired_retention,
                        fsrs7_runtime.as_ref(),
                    )
                    .max(1.0)
                    .round() as usize;
                    next_s_indices[[rating - 1, s_idx, d_idx]] = self.s2i(next_s);
                    next_d_indices[[rating - 1, s_idx, d_idx]] = self.d2i(next_d_val);
                    next_intervals[[rating - 1, s_idx, d_idx]] = next_ivl;
                }
            }
        }
        let transition_probs = transition_probs.view();
        let next_s_indices = next_s_indices.view();
        let next_d_indices = next_d_indices.view();
        let next_intervals = next_intervals.view();

        // Initialize cost matrix using ndarray
        let mut cost_matrix = Array3::zeros((self.s_size, self.d_size, self.t_size + 1));
        let review_costs = self.state_rating_costs[REVIEW];
        // Dynamic programming backward pass
        for t in (0..self.t_size).rev() {
            for s_idx in 0..self.s_size {
                for d_idx in 0..self.d_size {
                    let mut current_cost = 0.0;
                    for rating in 1..=4 {
                        let next_s_idx = next_s_indices[[rating - 1, s_idx, d_idx]];
                        let next_d_idx = next_d_indices[[rating - 1, s_idx, d_idx]];
                        let next_ivl = next_intervals[[rating - 1, s_idx, d_idx]];
                        let next_t_idx = (t + next_ivl).min(self.t_size);
                        let future_cost =
                            unsafe { *cost_matrix.uget([next_s_idx, next_d_idx, next_t_idx]) };
                        let transition_prob = transition_probs[[rating - 1, s_idx]];

                        current_cost += (review_costs[rating - 1] + future_cost) * transition_prob;
                    }
                    unsafe {
                        *cost_matrix.uget_mut([s_idx, d_idx, t]) = current_cost;
                    }
                }
            }
        }
        self.cost_matrix = cost_matrix;
    }

    pub fn evaluate_new_card_cost(
        &self,
        w: &Parameters,
        first_rating_probs: &[f32; 4],
        due: usize,
    ) -> f32 {
        if due > self.t_size {
            return 0.0;
        }
        let fsrs = simulated_fsrs(w);
        let fsrs7_runtime = fsrs7_runtime(w);
        let mut total_cost = 0.0;
        for rating in 1..=4 {
            let s = init_s(w, rating);
            let d = init_d_with_fsrs(fsrs, w, rating);
            let s_idx = self.s2i(s);
            let d_idx = self.d2i(d);
            let ivl =
                next_interval_with_fsrs(fsrs, w, s, self.desired_retention, fsrs7_runtime.as_ref())
                    .max(1.0)
                    .round() as usize;
            let t_idx = (due + ivl).min(self.t_size);
            total_cost += (unsafe { *self.cost_matrix.uget([s_idx, d_idx, t_idx]) }
                + self.state_rating_costs[LEARNING][rating - 1])
                * first_rating_probs[rating - 1];
        }
        total_cost
    }

    /// Calculate the expected cost for an in-flight card over a target period
    ///
    /// # Arguments
    /// * `card` - Current card state (stability S, difficulty D)
    /// * `w` - FSRS model parameters
    ///
    /// # Returns
    /// Expected total cost over the target period
    pub fn evaluate_in_flight_card_cost(&self, card: &Card, w: &Parameters) -> f32 {
        // If the upcoming review falls outside the target period, no cost is incurred
        if card.due > self.t_size as f32 {
            return 0.0;
        }
        let fsrs = simulated_fsrs(w);
        let fsrs7_runtime = fsrs7_runtime(w);

        let real_due = card.due.max(0.0);

        // Calculate total interval governing the upcoming review
        let ivl = real_due - card.last_date;

        // Calculate retrievability at the time of upcoming review
        let retrievability = power_forgetting_curve_with_runtime(
            fsrs,
            w,
            fsrs7_runtime.as_ref(),
            ivl,
            card.stability,
        );

        // Calculate rating probabilities
        let mut rating_probs = [0.0; 4];
        // rating 1 (Again) - failed recall
        rating_probs[0] = 1.0 - retrievability;
        // ratings 2, 3, 4 (Hard, Good, Easy) - successful recall
        for (i, &prob) in self.review_rating_prob.iter().enumerate() {
            rating_probs[i + 1] = retrievability * prob;
        }

        let mut expected_cost = 0.0;

        // Calculate expected cost over all possible review outcomes
        for rating in 1..=4 {
            let rating_idx = rating - 1;
            let transition_prob = rating_probs[rating_idx];

            if transition_prob <= 0.0 {
                continue;
            }

            // Calculate immediate review cost
            let immediate_cost = self.state_rating_costs[REVIEW][rating_idx];

            // Calculate new stability and difficulty after review
            let (new_stability, new_difficulty) = if rating == 1 {
                // Failed recall - use failure transition
                (
                    stability_after_failure_with_fsrs(
                        fsrs,
                        w,
                        card.stability,
                        retrievability,
                        card.difficulty,
                        ivl,
                    ),
                    next_d_with_fsrs(fsrs, w, card.difficulty, rating),
                )
            } else {
                // Successful recall - use success transition
                (
                    stability_after_success_with_fsrs(
                        fsrs,
                        w,
                        card.stability,
                        retrievability,
                        card.difficulty,
                        rating,
                        ivl,
                    ),
                    next_d_with_fsrs(fsrs, w, card.difficulty, rating),
                )
            };
            let new_interval = next_interval_with_fsrs(
                fsrs,
                w,
                new_stability,
                self.desired_retention,
                fsrs7_runtime.as_ref(),
            )
            .max(1.0)
            .round() as usize;
            let new_due = real_due as usize + new_interval;
            // Calculate future cost using precomputed cost matrix
            let future_cost = if new_due > self.t_size {
                0.0
            } else {
                let s_idx = self.s2i(new_stability);
                let d_idx = self.d2i(new_difficulty);
                let t_idx = new_due as usize;
                unsafe { *self.cost_matrix.uget([s_idx, d_idx, t_idx]) }
            };

            expected_cost += transition_prob * (immediate_cost + future_cost);
        }

        expected_cost
    }
}

pub fn expected_workload(
    parameters: &Parameters,
    desired_retention: f32,
    config: &SimulatorConfig,
) -> Result<f32> {
    let w = &check_and_fill_parameters(parameters)?;
    let mut estimator = WorkloadEstimator::new(config);
    estimator.precompute_cost_matrix(desired_retention, w);
    let workload = estimator.evaluate_new_card_cost(w, &config.first_rating_prob, 0);
    Ok(workload)
}

/// Evaluate expected workload when there are already in-flight cards.
///
/// Note: per-card parameters in `existing_cards` are ignored. This function always
/// uses the provided global `parameters` slice.
pub fn expected_workload_with_existing_cards(
    parameters: &Parameters,
    desired_retention: f32,
    config: &SimulatorConfig,
    existing_cards: &[Card],
) -> Result<f32> {
    let w = check_and_fill_parameters(parameters)?;
    let mut estimator = WorkloadEstimator::new(config);
    estimator.precompute_cost_matrix(desired_retention, &w);
    let mut cards = existing_cards.to_vec();
    let w = Arc::new(w);

    if config.learn_limit > 0 {
        let init_ratings = (0..(config.deck_size - cards.len())).map(|i| Card {
            id: -(i as i64),
            difficulty: f32::NEG_INFINITY,
            stability: f32::NEG_INFINITY,
            last_date: f32::NEG_INFINITY,
            due: (i / config.learn_limit) as f32,
            interval: f32::NEG_INFINITY,
            lapses: 0,
            desired_retention,
            parameters: w.clone(),
        });

        cards.extend(init_ratings);
    }
    let workload = cards
        .iter()
        .map(|card| {
            if card.stability > 1e-9 {
                estimator.evaluate_in_flight_card_cost(card, &w)
            } else {
                estimator.evaluate_new_card_cost(&w, &config.first_rating_prob, card.due as usize)
            }
        })
        .sum();
    Ok(workload)
}

#[derive(Debug, Clone)]
pub struct Card {
    // "id" ignored by "simulate", used purely for hook functions (can be all be 0 with no consequence).
    // new cards created by the simulation have negative id's so use positive ones.
    pub id: i64,
    pub difficulty: f32,
    pub stability: f32,
    pub last_date: f32,
    pub due: f32,
    pub interval: f32,
    pub lapses: u32,
    pub desired_retention: f32,
    // check_and_fill_parameters needs to be called manually on the parameters provided to the card.
    pub parameters: Arc<Vec<f32>>,
}

impl Card {
    pub fn power_forgetting_curve(&self, w: &[f32], t: f32) -> f32 {
        let fsrs = simulated_fsrs(w);
        power_forgetting_curve_with_fsrs(fsrs, w, t, self.stability)
    }

    pub fn retention_on(&self, date: f32) -> f32 {
        self.power_forgetting_curve(&self.parameters, date - self.last_date)
    }

    pub fn retrievability(&self) -> f32 {
        self.retention_on(self.due)
    }

    pub fn scheduled_due(&self) -> f32 {
        self.last_date + self.interval
    }
}

impl Default for Card {
    fn default() -> Self {
        static DEFAULT_PARAMETERS_ARC: LazyLock<Arc<Vec<f32>>> =
            LazyLock::new(|| Arc::new(DEFAULT_PARAMETERS.to_vec()));
        Self {
            id: 0,
            difficulty: f32::NEG_INFINITY,
            stability: f32::NEG_INFINITY,
            last_date: f32::NEG_INFINITY,
            due: f32::NEG_INFINITY,
            interval: f32::NEG_INFINITY,
            lapses: 0,
            desired_retention: 0.9,
            parameters: DEFAULT_PARAMETERS_ARC.clone(),
        }
    }
}

pub fn simulate(
    config: &SimulatorConfig,
    w: &Parameters,
    desired_retention: f32,
    seed: Option<u64>,
    existing_cards: Option<Vec<Card>>,
) -> Result<SimulationResult, FSRSError> {
    Ok(simulate_inner(
        config,
        w,
        desired_retention,
        seed,
        existing_cards,
        None,
        0.0,
    )?
    .result)
}

pub fn simulate_with_cost_adr_policy(
    config: &SimulatorConfig,
    w: &Parameters,
    policy: &CostAdrPolicy,
    goal_cost_weight: f32,
    seed: Option<u64>,
    existing_cards: Option<Vec<Card>>,
) -> Result<SimulationResult, FSRSError> {
    Ok(simulate_with_cost_adr_policy_for_evaluation(
        config,
        w,
        policy,
        goal_cost_weight,
        seed,
        existing_cards,
    )?
    .result)
}

pub(crate) fn simulate_with_cost_adr_policy_for_evaluation(
    config: &SimulatorConfig,
    w: &Parameters,
    policy: &CostAdrPolicy,
    goal_cost_weight: f32,
    seed: Option<u64>,
    existing_cards: Option<Vec<Card>>,
) -> Result<CostAdrSimulationResult, FSRSError> {
    policy.validate()?;
    simulate_inner(
        config,
        w,
        policy.retention_max,
        seed,
        existing_cards,
        Some(policy),
        goal_cost_weight,
    )
}

fn simulate_inner(
    config: &SimulatorConfig,
    w: &Parameters,
    desired_retention: f32,
    seed: Option<u64>,
    existing_cards: Option<Vec<Card>>,
    cost_adr_policy: Option<&CostAdrPolicy>,
    goal_cost_weight: f32,
) -> Result<CostAdrSimulationResult, FSRSError> {
    let w = Arc::new(check_and_fill_parameters(w)?);
    let fsrs7_runtime = fsrs7_runtime(&w);
    let cost_adr_evaluator =
        cost_adr_policy.map(|policy| policy.evaluator_for_cost_weight(goal_cost_weight));
    if config.deck_size == 0 {
        return Err(FSRSError::InvalidDeckSize);
    }

    let mut review_cnt_per_day = vec![0; config.learn_span];
    let mut learn_cnt_per_day = vec![0; config.learn_span];
    let mut memorized_cnt_per_day = vec![0.0; config.learn_span];
    let mut cost_per_day = vec![0.0; config.learn_span];
    let mut due_cnt_per_day = vec![0; config.learn_span + config.learn_span / 2];
    let mut correct_cnt_per_day = vec![0; config.learn_span];
    let mut introduced_cnt_per_day = vec![0; config.learn_span];
    let mut desired_retention_sum = 0.0;
    let mut desired_retention_count = 0;

    let first_rating_choices = RATINGS;
    let first_rating_dist = WeightedIndex::new(config.first_rating_prob).unwrap();

    let review_rating_choices = &RATINGS[1..];
    let review_rating_dist = WeightedIndex::new(config.review_rating_prob).unwrap();

    let mut rng = StdRng::seed_from_u64(seed.unwrap_or(42));

    let mut cards = Vec::with_capacity(config.deck_size);

    let existing_count = if let Some(existing_cards) = &existing_cards {
        existing_cards.len()
    } else {
        0
    };

    if existing_count > config.deck_size {
        return Err(FSRSError::InvalidDeckSize);
    }

    if let Some(existing_cards) = existing_cards {
        cards.extend(
            existing_cards
                .into_iter()
                .filter(|card| card.stability > 1e-9),
        );
        for _ in cards
            .iter()
            .filter(|card| card.last_date != f32::NEG_INFINITY)
        {
            for day in introduced_cnt_per_day.iter_mut() {
                *day += 1;
            }
        }
    }

    for card in &cards {
        if (card.due as usize) < due_cnt_per_day.len() {
            due_cnt_per_day[card.due as usize] += 1;
        }
    }

    if config.learn_limit > 0 {
        let init_ratings = (0..(config.deck_size - cards.len())).map(|i| Card {
            id: -(i as i64),
            difficulty: f32::NEG_INFINITY,
            stability: f32::NEG_INFINITY,
            last_date: f32::NEG_INFINITY,
            due: (i / config.learn_limit) as f32,
            interval: f32::NEG_INFINITY,
            lapses: 0,
            desired_retention,
            parameters: w.clone(),
        });

        cards.extend(init_ratings);
    }

    let mut card_priorities = CardPriorityQueue::with_capacity(cards.len());
    let max_lapses = config.suspend_after_lapses.unwrap_or(u32::MAX);

    let review_priority_fn = config.review_priority_fn.clone().unwrap_or_default();

    fn effective_due_day(card: &Card) -> i32 {
        card.due.max(0.0) as i32
    }

    fn card_priority(card: &Card, learn: bool, cb: &ReviewPriorityFn) -> CardPriority {
        let priority = cb(card);
        // high priority for early due, review, custom priority
        Reverse((effective_due_day(card), learn, priority))
    }

    for (i, card) in cards.iter().enumerate() {
        card_priorities.push(
            i,
            card_priority(
                card,
                card.last_date == f32::NEG_INFINITY,
                &review_priority_fn,
            ),
        );
    }

    // Main simulation loop
    while let Some(card_index) = card_priorities.peek_index() {
        let card = &mut cards[card_index];
        let fsrs = simulated_fsrs(&card.parameters);
        let card_fsrs7_runtime = if Arc::ptr_eq(&card.parameters, &w) {
            fsrs7_runtime.as_ref()
        } else {
            None
        };

        let day_index = card.due.max(0.0) as usize;

        let is_learn = card.last_date == f32::NEG_INFINITY;

        let last_date_index = card.last_date as usize;

        // Guards
        if card.due >= config.learn_span as f32 || card.lapses >= max_lapses {
            if !is_learn {
                add_forgetting_curve_range(
                    fsrs,
                    &card.parameters,
                    card_fsrs7_runtime,
                    &mut memorized_cnt_per_day,
                    last_date_index,
                    config.learn_span.max(last_date_index),
                    card.last_date,
                    card.stability,
                );
            }
            card_priorities.pop();
            continue;
        }

        let todays_learn = learn_cnt_per_day[day_index];
        let todays_review = review_cnt_per_day[day_index];

        if match (config.new_cards_ignore_review_limit, is_learn) {
            (true, true) => todays_learn + 1 > config.learn_limit,
            (false, true) => {
                todays_learn + todays_review + 1 > config.review_limit
                    || todays_learn + 1 > config.learn_limit
            }
            (_, false) => todays_review + 1 > config.review_limit,
        } || (cost_per_day[day_index] > config.max_cost_perday)
        {
            if !is_learn {
                due_cnt_per_day[day_index] -= 1;
                if day_index + 1 < due_cnt_per_day.len() {
                    due_cnt_per_day[day_index + 1] += 1;
                }
            }
            card.due = day_index as f32 + 1.0;
            card_priorities.change_priority(
                card_index,
                card_priority(card, is_learn, &review_priority_fn),
            );
            continue;
        }

        // dbg!(&day_index);
        if is_learn {
            // For learning cards
            // Initialize stability and difficulty for new cards
            let init_rating = first_rating_choices[first_rating_dist.sample(&mut rng)];
            let init_stability = init_s(&card.parameters, init_rating);
            let init_difficulty =
                init_d_with_fsrs(fsrs, &card.parameters, init_rating).clamp(D_MIN, D_MAX);
            let (new_s, new_d, cost) = memory_state_short_term_with_fsrs(
                fsrs,
                &card.parameters,
                init_stability,
                init_difficulty,
                Some(init_rating),
                &config.state_rating_costs[LEARNING],
                &config.learning_step_transitions,
                config.learning_step_count,
                &mut rng,
            );
            card.stability = new_s;
            card.difficulty = new_d;

            // Update days statistics
            learn_cnt_per_day[day_index] += 1;
            cost_per_day[day_index] += cost;

            for day in introduced_cnt_per_day.iter_mut().skip(day_index) {
                *day += 1;
            }
        } else {
            // For review cards
            let last_stability = card.stability;
            let ivl = day_index as f32 - card.last_date;

            // Calculate retrievability for entries where has_learned is true
            let retrievability = power_forgetting_curve_with_runtime(
                fsrs,
                &card.parameters,
                card_fsrs7_runtime,
                ivl,
                card.stability,
            );

            // Create 'forget' mask
            let forget = !rng.random_bool(retrievability as f64);

            card.lapses += forget as u32;
            correct_cnt_per_day[day_index] += !forget as usize;

            // Sample 'rating' for 'need_review' entries
            let rating = if forget {
                1
            } else {
                review_rating_choices[review_rating_dist.sample(&mut rng)]
            };

            //dbg!(&card, &rating);

            let (new_s, new_d, cost) = if forget {
                let post_lapse_stability = stability_after_failure_with_fsrs(
                    fsrs,
                    &card.parameters,
                    last_stability,
                    retrievability,
                    card.difficulty,
                    ivl,
                );
                let post_lapse_difficulty =
                    next_d_with_fsrs(fsrs, &card.parameters, card.difficulty, rating);
                let (new_s, new_d, cost) = memory_state_short_term_with_fsrs(
                    fsrs,
                    &card.parameters,
                    post_lapse_stability,
                    post_lapse_difficulty,
                    None,
                    &config.state_rating_costs[RELEARNING],
                    &config.relearning_step_transitions,
                    config.relearning_step_count,
                    &mut rng,
                );
                let review_cost = config
                    .review_rating_cost_fn
                    .as_ref()
                    .map(|cost_fn| cost_fn(card, rating, retrievability))
                    .unwrap_or(config.state_rating_costs[REVIEW][rating - 1]);
                (new_s, new_d, review_cost + cost)
            } else {
                let review_cost = config
                    .review_rating_cost_fn
                    .as_ref()
                    .map(|cost_fn| cost_fn(card, rating, retrievability))
                    .unwrap_or(config.state_rating_costs[REVIEW][rating - 1]);
                (
                    stability_after_success_with_fsrs(
                        fsrs,
                        &card.parameters,
                        last_stability,
                        retrievability,
                        card.difficulty,
                        rating,
                        ivl,
                    ),
                    next_d_with_fsrs(fsrs, &card.parameters, card.difficulty, rating),
                    review_cost,
                )
            };

            // Update days statistics
            review_cnt_per_day[day_index] += 1;
            cost_per_day[day_index] += cost;

            add_forgetting_curve_range(
                fsrs,
                &card.parameters,
                card_fsrs7_runtime,
                &mut memorized_cnt_per_day,
                last_date_index,
                day_index,
                card.last_date,
                card.stability,
            );

            card.stability = new_s;
            card.difficulty = new_d;
        }

        if let Some(evaluator) = &cost_adr_evaluator {
            card.desired_retention = evaluator.evaluate_retention(card.stability, card.difficulty);
        }
        desired_retention_sum += card.desired_retention;
        desired_retention_count += 1;

        let mut ivl = next_interval_with_fsrs(
            fsrs,
            &card.parameters,
            card.stability,
            card.desired_retention,
            card_fsrs7_runtime,
        )
        .round()
        .clamp(1.0, config.max_ivl);

        card.last_date = day_index as f32;
        card.interval = ivl;
        card.due = day_index as f32 + ivl;

        if let Some(cb) = &config.post_scheduling_fn {
            ivl = cb(card, config.max_ivl, day_index, &due_cnt_per_day, &mut rng);
            card.interval = ivl;
            card.due = day_index as f32 + ivl;
        }

        if card.due < due_cnt_per_day.len() as f32 {
            due_cnt_per_day[card.due as usize] += 1;
        }

        card_priorities
            .change_priority(card_index, card_priority(card, false, &review_priority_fn));
    }

    /*dbg!((
        &memorized_cnt_per_day[learn_span - 1],
        &review_cnt_per_day[learn_span - 1],
        &learn_cnt_per_day[learn_span - 1],
        &cost_per_day[learn_span - 1],
    ));*/

    Ok(CostAdrSimulationResult {
        result: SimulationResult {
            memorized_cnt_per_day,
            review_cnt_per_day,
            learn_cnt_per_day,
            cost_per_day,
            correct_cnt_per_day,
            cards,
            introduced_cnt_per_day,
        },
        average_desired_retention: if desired_retention_count > 0 {
            Some(desired_retention_sum / desired_retention_count as f32)
        } else {
            None
        },
    })
}

fn sample<F>(
    config: &SimulatorConfig,
    parameters: &Parameters,
    desired_retention: f32,
    n: usize,
    progress: &mut F,
    cards: &Option<Vec<Card>>,
    target: &CMRRTargetFn,
) -> Result<f32, FSRSError>
where
    F: FnMut() -> bool,
{
    if !progress() {
        return Err(FSRSError::Interrupted);
    }
    let results: Result<Vec<f32>, FSRSError> = (0..n)
        .into_par_iter()
        .map(|i| {
            let result = simulate(
                config,
                parameters,
                desired_retention,
                Some((i + 42).try_into().unwrap()),
                cards.clone(),
            )?;

            Ok(target(&result, parameters))
        })
        .collect();
    results.map(|v| v.iter().sum::<f32>() / n as f32)
}

/// For the given simulator parameters and parameters, determine the suggested `desired_retention`
/// value.
pub fn optimal_retention<F>(
    config: &SimulatorConfig,
    parameters: &Parameters,
    mut progress: F,
    cards: Option<Vec<Card>>,
    target: Option<CMRRTargetFn>,
) -> Result<f32>
where
    F: FnMut(ItemProgress) -> bool + Send,
{
    let mut progress_info = ItemProgress {
        current: 0,
        // not provided for this method
        total: 0,
    };
    let inc_progress = move || {
        progress_info.current += 1;
        progress(progress_info)
    };

    brent(
        config,
        parameters,
        cards,
        inc_progress,
        target.unwrap_or_default(),
    )
}

/// https://argmin-rs.github.io/argmin/argmin/solver/brent/index.html
/// https://github.com/scipy/scipy/blob/5e4a5e3785f79dd4e8930eed883da89958860db2/scipy/optimize/_optimize.py#L2446
fn brent<F>(
    config: &SimulatorConfig,
    parameters: &Parameters,
    cards: Option<Vec<Card>>,
    mut progress: F,
    target: CMRRTargetFn,
) -> Result<f32, FSRSError>
where
    F: FnMut() -> bool,
{
    let mintol = 1e-10;
    let cg = 0.381_966;
    let maxiter = 64;
    let tol = 0.01f32;
    let parameters = check_and_fill_parameters(parameters)?;

    let default_sample_size = 16.0;
    let sample_size = match config.learn_span {
        ..=30 => 180,
        31..365 => {
            let (a1, a2, a3) = (8.20e-7, 2.41e-3, 1.30e-2);
            let factor = (config.learn_span as f32)
                .powf(2.0)
                .mul_add(a1, config.learn_span as f32 * a2 + a3);
            (default_sample_size / factor).round() as usize
        }
        365.. => default_sample_size as usize,
    };

    let (xb, fb) = (
        R_MIN,
        sample(
            config,
            &parameters,
            R_MIN,
            sample_size,
            &mut progress,
            &cards,
            &target,
        )?,
    );
    let (mut x, mut v, mut w) = (xb, xb, xb);
    let (mut fx, mut fv, mut fw) = (fb, fb, fb);
    let (mut a, mut b) = (R_MIN, R_MAX);
    let mut deltax: f32 = 0.0;
    let mut iter = 0;
    let mut rat = 0.0;
    let mut u;

    while iter < maxiter {
        let tol1 = tol.mul_add(x.abs(), mintol);
        let tol2 = 2.0 * tol1;
        let xmid = 0.5 * (a + b);
        // check for convergence
        if (x - xmid).abs() < 0.5f32.mul_add(-(b - a), tol2) {
            break;
        }
        if deltax.abs() <= tol1 {
            // do a golden section step
            deltax = if x >= xmid { a } else { b } - x;
            rat = cg * deltax;
        } else {
            // do a parabolic step
            let tmp1 = (x - w) * (fx - fv);
            let mut tmp2 = (x - v) * (fx - fw);
            let mut p = (x - v).mul_add(tmp2, -(x - w) * tmp1);
            tmp2 = 2.0 * (tmp2 - tmp1);
            if tmp2 > 0.0 {
                p = -p;
            }
            tmp2 = tmp2.abs();
            let deltax_tmp = deltax;
            deltax = rat;
            // check parabolic fit
            if (p > tmp2 * (a - x))
                && (p < tmp2 * (b - x))
                && (p.abs() < (0.5 * tmp2 * deltax_tmp).abs())
            {
                // if parabolic step is useful
                rat = p / tmp2;
                u = x + rat;
                if (u - a) < tol2 || (b - u) < tol2 {
                    rat = if xmid - x >= 0.0 { tol1 } else { -tol1 };
                }
            } else {
                // if it's not do a golden section step
                deltax = if x >= xmid { a } else { b } - x;
                rat = cg * deltax;
            }
        }
        // update by at least tol1
        u = x + if rat.abs() < tol1 {
            tol1 * if rat >= 0.0 { 1.0 } else { -1.0 }
        } else {
            rat
        };
        // calculate new output value
        let fu = sample(
            config,
            &parameters,
            u,
            sample_size,
            &mut progress,
            &cards,
            &target,
        )?;

        // if it's bigger than current
        if fu > fx {
            if u < x {
                a = u;
            } else {
                b = u;
            }
            if fu <= fw || w == x {
                (v, w) = (w, u);
                (fv, fw) = (fw, fu);
            } else if fu <= fv || v == x || v == w {
                v = u;
                fv = fu;
            }
        } else {
            // if it's smaller than current
            if u >= x {
                a = x;
            } else {
                b = x;
            }
            (v, w, x) = (w, x, u);
            (fv, fw, fx) = (fw, fx, fu);
        }
        iter += 1;
    }
    let xmin = x;
    let success = iter < maxiter && (R_MIN..=R_MAX).contains(&xmin);
    // dbg!(iter);

    if success {
        Ok(xmin)
    } else {
        Err(FSRSError::OptimalNotFound)
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord)]
pub enum RevlogReviewKind {
    #[default]
    Learning = 0,
    Review = 1,
    Relearning = 2,
    /// Old Anki versions called this "Cram" or "Early", and assigned it when
    /// reviewing cards ahead. It is now only used for filtered decks with
    /// rescheduling disabled.
    Filtered = 3,
    Manual = 4,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct RevlogEntry {
    pub id: i64,
    pub cid: i64,
    pub usn: i32,
    /// - In the V1 scheduler, 3 represents easy in the learning case.
    /// - 0 represents manual rescheduling.
    pub button_chosen: u8,
    /// Positive values are in days, negative values in seconds.
    pub interval: i32,
    /// Positive values are in days, negative values in seconds.
    pub last_interval: i32,
    /// Card's ease after answering, stored as 10x the %, eg 2500 represents
    /// 250%.
    pub ease_factor: u32,
    /// Amount of milliseconds taken to answer the card.
    pub taken_millis: u32,
    pub review_kind: RevlogReviewKind,
}

/// Calculate transition matrix and counts from sequences of ratings
fn calculate_transitions(
    sequences: &[Vec<u8>],
    n_states: usize,
    smoothing: f32,
) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
    let mut transition_counts = vec![vec![0.0; n_states]; n_states];
    let mut initial_counts = vec![0.0; n_states];

    // Count transition frequencies and initial state frequencies
    for sequence in sequences {
        if sequence.is_empty() {
            continue;
        }

        // Record initial state
        initial_counts[sequence[0] as usize - 1] += 1.0;

        // Record transitions
        for i in 0..sequence.len() - 1 {
            let current_state = sequence[i] as usize - 1;
            let next_state = sequence[i + 1] as usize - 1;
            transition_counts[current_state][next_state] += 1.0;
        }
    }

    // Apply Laplace smoothing
    for i in 0..n_states {
        for j in 0..n_states {
            transition_counts[i][j] += smoothing;
        }
        initial_counts[i] += smoothing;
    }

    // Calculate transition probability matrix
    let mut transition_matrix = vec![vec![0.0; n_states]; n_states];
    for i in 0..n_states {
        let row_sum: f32 = transition_counts[i].iter().sum();
        if row_sum > 0.0 {
            for j in 0..n_states {
                transition_matrix[i][j] = transition_counts[i][j] / row_sum;
            }
        } else {
            // If a state never appears, assume uniform distribution
            let uniform_prob = 1.0 / n_states as f32;
            for j in 0..n_states {
                transition_matrix[i][j] = uniform_prob;
            }
        }
    }

    (transition_matrix, transition_counts)
}

pub fn extract_simulator_config(
    df: Vec<RevlogEntry>,
    day_cutoff: i64,
    smooth: bool,
) -> SimulatorConfig {
    if df.is_empty() {
        return SimulatorConfig::default();
    }

    // Calculate state rating costs
    let mut state_rating_costs = [[0.0; 4]; 3];
    let mut state_rating_counts = [[0; 4]; 3];
    let mut state_rating_durations = HashMap::<(usize, usize), Vec<u32>>::new();

    for entry in df.iter() {
        if entry.taken_millis > 0 && entry.taken_millis < 1200000 {
            let state = entry.review_kind as usize;
            let rating = entry.button_chosen as usize - 1;
            if state < 3 && rating < 4 {
                state_rating_durations
                    .entry((state, rating))
                    .or_default()
                    .push(entry.taken_millis);
                state_rating_counts[state][rating] += 1;
            }
        }
    }

    // Calculate mean costs
    for ((state, rating), durations) in state_rating_durations.iter() {
        let mean_duration = durations.iter().sum::<u32>() / durations.len() as u32;
        state_rating_costs[*state][*rating] = mean_duration as f32 / 1000.0;
    }

    // Group data by card_id and real_days
    struct Df1Row {
        card_id: i64,
        first_review_state: u8,
        first_review_rating: u8,
        same_day_ratings: Vec<u8>,
    }

    let df1 = {
        let mut grouped_data = HashMap::new();
        for &row in df.iter() {
            if row.taken_millis > 0 && row.taken_millis < 1200000 {
                let real_days = (row.id / 1000 - day_cutoff) / 86400;
                let key = (row.cid, real_days);
                let entry = grouped_data.entry(key).or_insert_with(Vec::new);
                entry.push(row);
            }
        }

        grouped_data
            .into_iter()
            .filter_map(|((card_id, _real_days), entries)| {
                entries.first().map(|first_entry| {
                    let first_review_state = first_entry.review_kind as u8 + 1;
                    let first_review_rating = first_entry.button_chosen;
                    let same_day_ratings = entries.iter().map(|e| e.button_chosen).collect();

                    Df1Row {
                        card_id,
                        first_review_state,
                        first_review_rating,
                        same_day_ratings,
                    }
                })
            })
            .collect_vec()
    };

    // Calculate button usage
    let mut button_usage_dict = HashMap::new();
    for row in df1.iter() {
        button_usage_dict
            .entry((row.first_review_state, row.first_review_rating))
            .or_insert_with(Vec::new)
            .push(row.card_id);
    }
    let button_usage_dict = button_usage_dict
        .into_iter()
        .map(|(x, y)| (x, y.len() as i64))
        .collect::<HashMap<_, _>>();

    // Calculate rating probabilities
    let mut learn_buttons: [i64; 4] = (1..=4)
        .map(|i| button_usage_dict.get(&(1, i)).copied().unwrap_or_default())
        .collect_vec()
        .try_into()
        .unwrap();
    if learn_buttons.iter().all(|&x| x == 0) {
        learn_buttons = [1, 1, 1, 1];
    }

    let mut review_buttons: [i64; 4] = (1..=4)
        .map(|i| button_usage_dict.get(&(2, i)).copied().unwrap_or_default())
        .collect_vec()
        .try_into()
        .unwrap();
    if review_buttons.iter().skip(1).all(|&x| x == 0) {
        review_buttons = [review_buttons[0], 1, 1, 1];
    }

    let mut first_rating_prob: [f32; 4] = learn_buttons
        .iter()
        .map(|x| *x as f32 / learn_buttons.iter().sum::<i64>() as f32)
        .collect_vec()
        .try_into()
        .unwrap();

    let mut review_rating_prob: [f32; 3] = review_buttons
        .iter()
        .skip(1)
        .map(|x| *x as f32 / review_buttons.iter().skip(1).sum::<i64>() as f32)
        .collect_vec()
        .try_into()
        .unwrap();

    // Calculate transition matrices
    let mut learning_step_rating_sequences = Vec::new();
    let mut relearning_step_rating_sequences = Vec::new();

    for row in df1.iter() {
        if row.first_review_state == 1 {
            learning_step_rating_sequences.push(row.same_day_ratings.clone());
        } else if row.first_review_state == 2 && row.first_review_rating == 1 {
            relearning_step_rating_sequences.push(row.same_day_ratings.clone());
        }
    }

    let (learning_transition_matrix, learning_transition_counts) =
        calculate_transitions(&learning_step_rating_sequences, 4, 1.0);
    let (relearning_transition_matrix, relearning_transition_counts) =
        calculate_transitions(&relearning_step_rating_sequences, 4, 1.0);

    let mut learning_step_transitions: [[f32; 4]; 3] = learning_transition_matrix
        .iter()
        .take(3)
        .map(|row| row.iter().copied().collect_vec().try_into().unwrap())
        .collect_vec()
        .try_into()
        .unwrap();

    let mut relearning_step_transitions: [[f32; 4]; 3] = relearning_transition_matrix
        .iter()
        .take(3)
        .map(|row| row.iter().copied().collect_vec().try_into().unwrap())
        .collect_vec()
        .try_into()
        .unwrap();

    // Smooth probabilities if requested
    fn lerp(v0: f32, v1: f32, t: f32) -> f32 {
        t * v0 + (1f32 - t) * v1
    }
    if smooth {
        let config = SimulatorConfig::default();

        let total_learn_buttons: i64 = learn_buttons.iter().sum();
        let weight = total_learn_buttons as f32 / (50.0 + total_learn_buttons as f32);
        first_rating_prob
            .iter_mut()
            .zip(config.first_rating_prob)
            .for_each(|(prob, first_rating_prob)| *prob = lerp(*prob, first_rating_prob, weight));

        let total_review_buttons_except_first: i64 = review_buttons[1..].iter().sum();
        let weight = total_review_buttons_except_first as f32
            / (50.0 + total_review_buttons_except_first as f32);
        review_rating_prob
            .iter_mut()
            .zip(config.review_rating_prob)
            .for_each(|(prob, review_rating_prob)| *prob = lerp(*prob, review_rating_prob, weight));

        izip!(
            learning_step_transitions.iter_mut(),
            config.learning_step_transitions,
            learning_transition_counts,
        )
        .for_each(|(rating_probs, default_rating_probs, transition_counts)| {
            let total_learning_step_entries = transition_counts.iter().sum::<f32>();
            let weight = total_learning_step_entries / (50.0 + total_learning_step_entries);
            izip!(rating_probs.iter_mut(), default_rating_probs)
                .for_each(|(prob, default_prob)| *prob = lerp(*prob, default_prob, weight));
        });

        izip!(
            relearning_step_transitions.iter_mut(),
            config.relearning_step_transitions,
            relearning_transition_counts,
        )
        .for_each(|(rating_probs, default_rating_probs, transition_counts)| {
            let total_relearning_step_entries = transition_counts.iter().sum::<f32>();
            let weight = total_relearning_step_entries / (50.0 + total_relearning_step_entries);
            izip!(rating_probs.iter_mut(), default_rating_probs)
                .for_each(|(prob, default_prob)| *prob = lerp(*prob, default_prob, weight));
        });

        izip!(
            state_rating_costs.iter_mut(),
            config.state_rating_costs.iter(),
            state_rating_counts.iter()
        )
        .for_each(|(rating_costs, default_rating_costs, rating_counts)| {
            izip!(rating_costs.iter_mut(), default_rating_costs, rating_counts).for_each(
                |(cost, &default_cost, &count)| {
                    let weight = count as f32 / (50.0 + count as f32);
                    *cost = lerp(*cost, default_cost, weight).to_2_decimal();
                },
            );
        });
    }

    SimulatorConfig {
        first_rating_prob,
        review_rating_prob,
        learning_step_transitions,
        relearning_step_transitions,
        state_rating_costs,
        ..Default::default()
    }
}

#[cfg(test)]
mod tests {
    use std::time::Instant;

    use super::*;
    use crate::{
        FSRS6_DEFAULT_PARAMETERS, convertor_tests::read_collection, test_helpers::TestHelper,
    };
    const LEARN_COST: f32 = 42.;
    const REVIEW_COST: f32 = 43.;
    const DEFAULT_PARAMETERS: [f32; 21] = FSRS6_DEFAULT_PARAMETERS;

    #[test]
    fn test_card_priority_queue_updates_existing_card_priority() {
        let mut queue = CardPriorityQueue::with_capacity(3);
        queue.push(0, Reverse((3, false, 0)));
        queue.push(1, Reverse((2, false, 0)));
        queue.push(2, Reverse((4, false, 0)));

        assert_eq!(queue.peek_index(), Some(1));
        queue.change_priority(0, Reverse((1, false, 0)));
        assert_eq!(queue.peek_index(), Some(0));

        queue.pop();
        assert_eq!(queue.peek_index(), Some(1));
        queue.change_priority(2, Reverse((0, false, 0)));
        assert_eq!(queue.peek_index(), Some(2));

        queue.pop();
        assert_eq!(queue.peek_index(), Some(1));
        queue.pop();
        assert_eq!(queue.peek_index(), None);
    }

    #[test]
    fn test_card_priority_queue_ties_match_heap_swap_order() {
        let mut queue = CardPriorityQueue::with_capacity(3);
        let priority = Reverse((5, false, 0));
        queue.push(0, priority);
        queue.push(1, priority);
        queue.push(2, priority);

        assert_eq!(queue.peek_index(), Some(0));
        queue.pop();
        assert_eq!(queue.peek_index(), Some(2));
        queue.pop();
        assert_eq!(queue.peek_index(), Some(1));
    }

    #[test]
    fn test_add_forgetting_curve_range_scalar_matches_previous_loop() {
        let w = DEFAULT_PARAMETERS;
        let fsrs = simulated_fsrs(&w);
        let start_day = 3;
        let last_date = 2.0;
        let stability = 12.0;
        let mut expected = vec![0.0; 20];
        let mut actual = vec![0.0; 20];

        for (offset, day) in expected[3..17].iter_mut().enumerate() {
            *day += power_forgetting_curve_with_fsrs(
                fsrs,
                &w,
                (start_day + offset) as f32 - last_date,
                stability,
            );
        }
        add_forgetting_curve_range(fsrs, &w, None, &mut actual, 3, 17, last_date, stability);

        assert_eq!(actual, expected);
    }

    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    #[test]
    fn test_neon_fsrs7_forgetting_curve_range_matches_scalar() {
        let w = crate::DEFAULT_PARAMETERS;
        let fsrs = simulated_fsrs(&w);
        let start_day = 5;
        let last_date = 3.25;
        let stability = 17.5;
        let mut scalar = vec![1.0; 37];
        let mut neon = scalar.clone();

        add_forgetting_curve_range_scalar(
            fsrs,
            &w,
            None,
            &mut scalar,
            start_day,
            last_date,
            stability,
        );
        super::neon::add_fsrs7_forgetting_curve_range(
            &w, &mut neon, start_day, last_date, stability,
        );

        for (index, (scalar, neon)) in scalar.iter().zip(neon).enumerate() {
            let relative_error = (*scalar - neon).abs() / scalar.abs().max(1.0);
            assert!(
                relative_error < 5e-4,
                "index {index}, scalar {scalar}, neon {neon}, relative error {relative_error:e}"
            );
        }
    }

    #[test]
    fn test_memory_state_short_term() {
        let w = DEFAULT_PARAMETERS;
        let config = SimulatorConfig::default();
        let mut rng = StdRng::seed_from_u64(42);

        // Expected results for each init_rating
        let expected_results = [
            (0.12584424, 8.779163, 66.72), // init_rating = 1
            (1.3771622, 5.092413, 46.35),  // init_rating = 2
            (2.3065, 2.1112142, 27.56),    // init_rating = 3
            (8.2956, 1.0, 10.71),          // init_rating = 4
        ];

        // Test for each init_rating from 1 to 4
        for init_rating in 1..=4 {
            let s = w[init_rating - 1];
            let d = init_d(&w, init_rating);
            let result = memory_state_short_term(
                &w,
                s,
                d,
                Some(init_rating),
                &config.state_rating_costs[LEARNING],
                &config.learning_step_transitions,
                config.learning_step_count,
                &mut rng,
            );

            // Check against expected result for this init_rating
            assert_eq!(result, expected_results[init_rating - 1]);
        }

        let s = 10.0;
        let d = 5.0;
        let post_lapse_s = stability_after_failure(&w, s, 0.9, d, 1.0);
        let post_lapse_d = next_d(&w, d, 1);
        let cost = config.state_rating_costs[REVIEW][0];
        dbg!(post_lapse_s, post_lapse_d, cost);
        let mut rng = StdRng::seed_from_u64(2024);
        let result = memory_state_short_term(
            &w,
            post_lapse_s,
            post_lapse_d,
            None,
            &config.state_rating_costs[RELEARNING],
            &config.relearning_step_transitions,
            config.relearning_step_count,
            &mut rng,
        );
        assert_eq!(result, (1.4311036, 8.3286495, 12.32));
    }

    #[test]
    fn test_simulator_memorization() -> Result<()> {
        let config = SimulatorConfig::default();
        let SimulationResult {
            memorized_cnt_per_day,
            ..
        } = simulate(&config, &DEFAULT_PARAMETERS, 0.9, None, None)?;
        assert_eq!(
            memorized_cnt_per_day[memorized_cnt_per_day.len() - 1],
            3370.383
        );
        Ok(())
    }

    #[test]
    fn test_simulator_learn_review_costs() -> Result<()> {
        let config = SimulatorConfig {
            deck_size: 1,
            learn_span: 1,
            first_rating_prob: [0., 0., 1., 0.],
            state_rating_costs: [[LEARN_COST; 4], [REVIEW_COST; 4], [0.; 4]],
            learning_step_count: 1,
            ..Default::default()
        };

        let cards = vec![Card {
            difficulty: 5.0,
            stability: 5.0,
            last_date: -5.0,
            due: 0.0,
            interval: 5.0,
            ..Default::default()
        }];

        let SimulationResult {
            cost_per_day: cost_per_day_learn,
            ..
        } = simulate(&config, &DEFAULT_PARAMETERS, 0.9, None, None)?;
        assert_eq!(cost_per_day_learn[0], LEARN_COST);

        let SimulationResult {
            cost_per_day: cost_per_day_review,
            ..
        } = simulate(&config, &DEFAULT_PARAMETERS, 0.9, None, Some(cards))?;
        assert_eq!(cost_per_day_review[0], REVIEW_COST);
        Ok(())
    }

    #[test]
    fn test_changing_learn_span_should_get_same_review_cnt_per_day() -> Result<()> {
        const LOWER: usize = 365;
        const DECK_SIZE: usize = 1000;
        const LEARN_LIMIT: usize = 10;
        let config = SimulatorConfig {
            learn_span: LOWER,
            learn_limit: LEARN_LIMIT,
            deck_size: DECK_SIZE,
            ..Default::default()
        };
        let SimulationResult {
            review_cnt_per_day: review_cnt_per_day_lower,
            ..
        } = simulate(&config, &DEFAULT_PARAMETERS, 0.9, None, None)?;
        let config = SimulatorConfig {
            learn_span: LOWER + 10,
            learn_limit: LEARN_LIMIT,
            deck_size: DECK_SIZE,
            ..Default::default()
        };
        let SimulationResult {
            review_cnt_per_day: review_cnt_per_day_higher,
            ..
        } = simulate(&config, &DEFAULT_PARAMETERS, 0.9, None, None)?;
        // Compare first LOWER items of review_cnt_per_day arrays
        for i in 0..LOWER {
            assert_eq!(
                review_cnt_per_day_lower[i], review_cnt_per_day_higher[i],
                "at index {}",
                i
            );
        }
        Ok(())
    }

    #[test]
    fn test_simulate_with_existing_cards() -> Result<()> {
        let config = SimulatorConfig {
            learn_span: 30,
            learn_limit: 60,
            review_limit: 200,
            max_cost_perday: f32::INFINITY,
            ..Default::default()
        };
        let cards = vec![
            Card {
                difficulty: 5.0,
                stability: 5.0,
                last_date: -5.0,
                due: 0.0,
                interval: 5.0,
                ..Default::default()
            },
            Card {
                difficulty: 5.0,
                stability: 2.0,
                last_date: -2.0,
                due: 0.0,
                interval: 2.0,
                ..Default::default()
            },
            Card {
                difficulty: 5.0,
                stability: 2.0,
                last_date: -2.0,
                due: 1.0,
                interval: 3.0,
                ..Default::default()
            },
            Card {
                difficulty: 5.0,
                stability: 2.0,
                last_date: -8.0,
                due: -1.0,
                interval: 7.0,
                ..Default::default()
            },
        ];
        let SimulationResult {
            memorized_cnt_per_day,
            review_cnt_per_day,
            learn_cnt_per_day,
            ..
        } = simulate(&config, &DEFAULT_PARAMETERS, 0.9, None, Some(cards))?;
        assert!((memorized_cnt_per_day[0] - 63.9).abs() < 0.1);
        assert_eq!(review_cnt_per_day[0], 3);
        assert_eq!(learn_cnt_per_day[0], 60);
        Ok(())
    }

    #[test]
    fn test_simulate_suspend_on_lapse_count() -> Result<()> {
        let cards = vec![Card {
            difficulty: 10.0,
            stability: f32::EPSILON,
            last_date: -5.0,
            due: 0.0,
            interval: 5.0,
            ..Default::default()
        }];

        let config = SimulatorConfig {
            learn_limit: 1,
            review_limit: 100,
            learn_span: 200,
            deck_size: cards.len(),
            suspend_after_lapses: Some(1),
            ..Default::default()
        };

        let SimulationResult {
            review_cnt_per_day, ..
        } = simulate(&config, &DEFAULT_PARAMETERS, 0.9, None, Some(cards))?;

        assert_eq!(1, review_cnt_per_day.iter().sum::<usize>());

        Ok(())
    }

    #[test]
    fn test_simulate_with_learn_limit() -> Result<()> {
        let config = SimulatorConfig {
            learn_limit: 3,
            review_limit: 10,
            learn_span: 3,
            ..Default::default()
        };

        let cards = vec![
            Card {
                difficulty: 5.0,
                stability: 5.0,
                last_date: -5.0,
                due: 0.0,
                interval: 5.0,
                ..Default::default()
            };
            9
        ];

        let SimulationResult {
            learn_cnt_per_day, ..
        } = simulate(&config, &DEFAULT_PARAMETERS, 0.9, None, Some(cards))?;

        assert_eq!(learn_cnt_per_day.to_vec(), vec![3, 3, 3]);

        Ok(())
    }

    #[test]
    fn test_simulate_with_new_affects_review_limit() -> Result<()> {
        let config = SimulatorConfig {
            learn_limit: 3,
            review_limit: 10,
            learn_span: 3,
            new_cards_ignore_review_limit: false,
            deck_size: 20,
            ..Default::default()
        };

        let cards = vec![
            Card {
                difficulty: 5.0,
                stability: 500.0,
                last_date: -5.0,
                due: 0.0,
                interval: 5.0,
                ..Default::default()
            };
            9
        ];

        let SimulationResult {
            learn_cnt_per_day, ..
        } = simulate(&config, &DEFAULT_PARAMETERS, 0.9, None, Some(cards))?;

        assert_eq!(learn_cnt_per_day.to_vec(), vec![1, 3, 3]);

        Ok(())
    }

    #[test]
    fn test_simulate_with_learn_review_limit() -> Result<()> {
        let config = SimulatorConfig {
            learn_span: 30,
            learn_limit: 60,
            review_limit: 200,
            max_cost_perday: f32::INFINITY,
            ..Default::default()
        };
        let SimulationResult {
            review_cnt_per_day,
            learn_cnt_per_day,
            ..
        } = simulate(&config, &DEFAULT_PARAMETERS, 0.9, None, None)?;
        assert_eq!(
            review_cnt_per_day.to_vec(),
            vec![
                0, 21, 62, 69, 91, 93, 124, 106, 133, 126, 156, 142, 160, 185, 180, 200, 188, 200,
                200, 193, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200
            ]
        );
        assert_eq!(
            learn_cnt_per_day.to_vec(),
            vec![config.learn_limit; config.learn_span]
        );
        Ok(())
    }

    #[test]
    fn test_simulate_with_max_ivl() -> Result<()> {
        let config = SimulatorConfig {
            max_ivl: 100.0,
            ..Default::default()
        };
        let SimulationResult {
            memorized_cnt_per_day,
            ..
        } = simulate(&config, &DEFAULT_PARAMETERS, 0.9, None, None)?;
        assert_eq!(
            memorized_cnt_per_day[memorized_cnt_per_day.len() - 1],
            3379.6497
        );
        Ok(())
    }

    #[test]
    fn test_simulate_with_zero_cards() -> Result<()> {
        let config = SimulatorConfig {
            deck_size: 0,
            ..Default::default()
        };
        let results = simulate(&config, &DEFAULT_PARAMETERS, 0.9, None, None);
        assert_eq!(results.unwrap_err(), FSRSError::InvalidDeckSize);
        Ok(())
    }

    #[test]
    fn test_simulate_returns_cards() -> Result<()> {
        let w = DEFAULT_PARAMETERS;

        let config = SimulatorConfig {
            deck_size: 1,
            learn_span: 1,
            first_rating_prob: [0., 0., 1., 0.],
            state_rating_costs: [[LEARN_COST; 4], [REVIEW_COST; 4], [0.; 4]],
            learning_step_count: 1,
            ..Default::default()
        };

        let SimulationResult { cards, .. } =
            simulate(&config, &DEFAULT_PARAMETERS, 0.9, None, None)?;

        assert_eq!(cards.len(), 1);
        let card = &cards[0];
        let rating = 3;
        assert_eq!(card.lapses, 0);
        assert_eq!(card.stability, w[rating - 1]);
        assert_eq!(card.difficulty, init_d(&w, rating));

        Ok(())
    }

    #[test]
    fn test_simulate_with_existing_cards_with_wrong_deck_size() -> Result<()> {
        let config = SimulatorConfig {
            deck_size: 1,
            ..Default::default()
        };
        let cards = vec![
            Card {
                difficulty: 5.0,
                stability: 5.0,
                last_date: -5.0,
                due: 0.0,
                interval: 5.0,
                ..Default::default()
            },
            Card {
                stability: 2.0,
                last_date: -2.0,
                due: 0.0,
                interval: 2.0,
                lapses: 0,
                ..Default::default()
            },
        ];
        let results = simulate(&config, &DEFAULT_PARAMETERS, 0.9, None, Some(cards));
        assert_eq!(results.unwrap_err(), FSRSError::InvalidDeckSize);
        Ok(())
    }

    #[test]
    fn test_learn_does_not_affect_correct_count() -> Result<()> {
        let mut w = DEFAULT_PARAMETERS;
        w[3] = 10000.;

        let config = SimulatorConfig {
            first_rating_prob: [0., 0., 0., 1.],
            deck_size: 5000,
            learn_limit: 10,
            ..Default::default()
        };

        let cards = vec![
            Card {
                difficulty: 5.0,
                stability: S_MAX,
                last_date: -5.0,
                due: 1.0,
                interval: 5.0,
                ..Default::default()
            };
            5
        ];

        let SimulationResult {
            correct_cnt_per_day,
            review_cnt_per_day,
            ..
        } = simulate(&config, &w, 0.9, None, Some(cards))?;

        assert_eq!(correct_cnt_per_day[0], 0);
        assert_eq!(review_cnt_per_day[1], 5);
        assert_eq!(correct_cnt_per_day[1], 5);

        Ok(())
    }

    #[test]
    fn test_simulate_with_post_scheduling_fn() -> Result<()> {
        let config = SimulatorConfig {
            deck_size: 10,
            learn_span: 10,
            learn_limit: 1,
            post_scheduling_fn: Some(PostSchedulingFn(Arc::new(|_, _, _, _, _| 1.0))),
            ..Default::default()
        };
        let SimulationResult {
            review_cnt_per_day, ..
        } = simulate(&config, &DEFAULT_PARAMETERS, 0.9, None, None)?;
        assert_eq!(&review_cnt_per_day, &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9,]);
        Ok(())
    }

    #[test]
    fn test_simulate_with_review_priority_fn() -> Result<()> {
        fn calc_cost_per_memorization(
            memorized_cnt_per_day: &Parameters,
            cost_per_day: &Parameters,
        ) -> f32 {
            let total_memorized = memorized_cnt_per_day[memorized_cnt_per_day.len() - 1];
            let total_cost = cost_per_day.iter().sum::<f32>();
            total_cost / total_memorized
        }
        let mut config = SimulatorConfig {
            deck_size: 1000,
            learn_span: 100,
            learn_limit: 10,
            review_limit: 5,
            ..Default::default()
        };

        // Define a macro to set the review priority fn, run simulation and assert the expected cost.
        macro_rules! run_test {
            ($review_priority:expr, $expected:expr) => {{
                config.review_priority_fn = $review_priority;
                let SimulationResult {
                    memorized_cnt_per_day,
                    cost_per_day,
                    ..
                } = simulate(&config, &DEFAULT_PARAMETERS, 0.8, None, None)?;
                let cost_per_memorization =
                    calc_cost_per_memorization(&memorized_cnt_per_day, &cost_per_day);
                println!("cost_per_memorization: {}", cost_per_memorization);
                assert!((cost_per_memorization - $expected).abs() < 0.01);
                Ok(())
            }};
        }

        macro_rules! wrap {
            ($f:expr) => {
                Some(ReviewPriorityFn(std::sync::Arc::new($f)))
            };
        }
        println!("Default behavior: low difficulty cards reviewed first.");
        run_test!(None, 69.28404)?;
        println!("High difficulty cards reviewed first.");
        run_test!(
            wrap!(|card: &Card| -(card.difficulty * 100.0) as i32),
            74.6778
        )?;
        println!("Low retrievability cards reviewed first.");
        run_test!(
            wrap!(|card: &Card| (card.retrievability() * 1000.0) as i32),
            74.90477
        )?;
        println!("High retrievability cards reviewed first.");
        run_test!(
            wrap!(|card: &Card| -(card.retrievability() * 1000.0) as i32),
            69.74799
        )?;
        println!("Low stability cards reviewed first.");
        run_test!(
            wrap!(|card: &Card| (card.stability * 100.0) as i32),
            74.361115
        )?;
        println!("High stability cards reviewed first.");
        run_test!(
            wrap!(|card: &Card| -(card.stability * 100.0) as i32),
            68.68905
        )?;
        println!("Long interval cards reviewed first.");
        run_test!(wrap!(|card: &Card| -card.interval as i32), 69.376434)?;
        println!("Short interval cards reviewed first.");
        run_test!(wrap!(|card: &Card| card.interval as i32), 74.64231)?;
        println!("Early scheduled due cards reviewed first.");
        run_test!(wrap!(|card: &Card| card.scheduled_due() as i32), 70.820175)?;
        println!("Late scheduled due cards reviewed first.");
        run_test!(wrap!(|card: &Card| -card.scheduled_due() as i32), 71.20782)?;
        Ok(())
    }

    #[test]
    fn test_review_rating_cost_fn_uses_actual_retrievability() -> Result<()> {
        let card = Card {
            id: 1,
            difficulty: 5.0,
            stability: 20.0,
            last_date: -20.0,
            due: -10.0,
            interval: 10.0,
            lapses: 0,
            desired_retention: 0.9,
            parameters: Arc::new(DEFAULT_PARAMETERS.to_vec()),
        };
        let scheduled_cost = card.retention_on(card.due) * 1000.0;
        let actual_cost = card.retention_on(0.0) * 1000.0;
        let config = SimulatorConfig {
            deck_size: 1,
            learn_span: 1,
            learn_limit: 0,
            review_limit: 1,
            state_rating_costs: [[0.0; 4]; 3],
            review_rating_cost_fn: Some(ReviewRatingCostFn::new(
                |_card, _rating, retrievability| retrievability * 1000.0,
            )),
            ..Default::default()
        };

        let result = simulate(&config, &DEFAULT_PARAMETERS, 0.9, None, Some(vec![card]))?;

        assert!((result.cost_per_day[0] - actual_cost).abs() < 0.001);
        assert!((result.cost_per_day[0] - scheduled_cost).abs() > 1.0);
        Ok(())
    }

    #[test]
    fn test_optimal_retention() -> Result<()> {
        let learn_span = 1000;
        let learn_limit = 10;
        let deck_size = learn_span * learn_limit;
        let config = SimulatorConfig {
            deck_size,
            learn_span,
            max_cost_perday: f32::INFINITY,
            learn_limit,
            ..Default::default()
        };
        let retention_value = optimal_retention(&config, &[], |_| true, None, None).unwrap();
        dbg!(retention_value);
        let card = Card {
            difficulty: 5.0,
            stability: 5.0,
            last_date: -5.0,
            due: 1.0,
            interval: 5.0,
            ..Default::default()
        };
        assert!((retention_value - 0.7).abs() < 0.01);
        optimal_retention(&config, &[1.], |_| true, None, None).unwrap_err();
        // Check that the cards are passed correctly to simulate
        optimal_retention(
            &config,
            &[],
            |_| true,
            Some(vec![card.clone(); deck_size]),
            None,
        )
        .unwrap();
        optimal_retention(
            &config,
            &[],
            |_| true,
            Some(vec![card; deck_size + 1]),
            None,
        )
        .unwrap_err();
        Ok(())
    }

    #[test]
    fn test_optimal_retention_with_old_parameters() -> Result<()> {
        let learn_span = 1000;
        let learn_limit = 10;
        let config = SimulatorConfig {
            deck_size: learn_span * learn_limit,
            learn_span,
            max_cost_perday: f32::INFINITY,
            learn_limit,
            ..Default::default()
        };
        let mut param = DEFAULT_PARAMETERS[..17].to_vec();
        param.extend_from_slice(&[0.0, 0.0]);
        let retention_value = optimal_retention(&config, &param, |_v| true, None, None).unwrap();
        [retention_value].assert_approx_eq([0.7706641]);
        Ok(())
    }

    #[test]
    fn test_extract_simulator_config_from_revlog() {
        let mut revlogs = read_collection().unwrap();
        revlogs.sort_by_cached_key(|r| (r.cid, r.id));
        let day_cutoff = 1720900800;
        let simulator_config = extract_simulator_config(revlogs.clone(), day_cutoff, false);
        assert_eq!(
            simulator_config,
            SimulatorConfig {
                first_rating_prob: [0.19349411, 0., 0.14357824, 0.662_927_6],
                review_rating_prob: [0.07351815, 0.9011334, 0.025348445],
                learning_step_transitions: [
                    [0.11098131, 0.0011682243, 0.24883178, 0.6390187],
                    [0.25, 0.25, 0.25, 0.25],
                    [0.017305315, 0.0012360939, 0.53646475, 0.44499382]
                ],
                relearning_step_transitions: [
                    [0.040342297, 0.001222494, 0.22249389, 0.7359413],
                    [0.25, 0.25, 0.25, 0.25],
                    [0.028571429, 0.007142857, 0.55, 0.41428572]
                ],
                state_rating_costs: [
                    [11.961, 0.0, 9.515, 7.437],
                    [11.075, 9.047, 7.774, 5.149],
                    [10.607, 0.0, 6.942, 6.643]
                ],
                ..Default::default()
            }
        );

        let simulator_config = extract_simulator_config(revlogs, day_cutoff, true);
        assert_eq!(
            simulator_config,
            SimulatorConfig {
                first_rating_prob: [0.19413717, 0.0012997796, 0.1484375, 0.65612555],
                review_rating_prob: [0.07409216, 0.900103, 0.025804851],
                learning_step_transitions: [
                    [0.12519868, 0.0045695365, 0.26328918, 0.6069371],
                    [0.059444442, 0.44009256, 0.43120366, 0.06935185],
                    [0.019318976, 0.0038998832, 0.5544936, 0.42229337]
                ],
                relearning_step_transitions: [
                    [0.050443545, 0.004855989, 0.24766704, 0.6970276],
                    [0.06481481, 0.44796297, 0.43287036, 0.05435185],
                    [0.04886842, 0.043, 0.5964737, 0.31168422]
                ],
                state_rating_costs: [
                    [12.38, 18.79, 9.68, 7.46],
                    [11.57, 9.47, 7.79, 5.65],
                    [13.74, 15.25, 7.75, 6.7]
                ],
                ..Default::default()
            }
        );
    }

    #[test]
    fn test_extract_simulator_config_without_revlog() {
        let simulator_config = extract_simulator_config(vec![], 0, true);
        assert_eq!(simulator_config, SimulatorConfig::default());
    }

    #[test]
    fn test_evaluate_new_card_cost() {
        let w = &check_and_fill_parameters(&DEFAULT_PARAMETERS).unwrap();
        let introduce_span = 365;
        let mut config = SimulatorConfig::default();
        config.learn_span = 365;
        config.learn_limit = 100;
        config.deck_size = introduce_span * config.learn_limit;
        config.max_cost_perday = f32::INFINITY;
        config.review_limit = usize::MAX;
        config.learning_step_count = 0;
        config.relearning_step_count = 0;
        for desired_retention in (72..=99).step_by(3).map(|x| x as f32 / 100.0) {
            dbg!(desired_retention);
            let mut estimator = WorkloadEstimator::new(&config);
            estimator.precompute_cost_matrix(desired_retention, w);
            let mut cost_dp = 0.0;
            for due in 0..introduce_span {
                let cost = estimator.evaluate_new_card_cost(w, &config.first_rating_prob, due);
                cost_dp += cost;
            }

            let result = simulate(&config, w, desired_retention, None, None).unwrap();
            let cost_simulated =
                result.cost_per_day.iter().sum::<f32>() / config.learn_limit as f32;
            let relative_error = (cost_dp - cost_simulated).abs() / cost_simulated;
            println!(
                "DP: {:.2}\tSimulated: {:.2}\tRelative Error: {:.2}",
                cost_dp, cost_simulated, relative_error
            );
            assert!(relative_error < 0.1);
        }
    }

    #[test]
    fn test_expected_workload() {
        let mut config = SimulatorConfig::default();
        config.learn_span = 365;
        config.learn_limit = 400;
        config.deck_size = config.learn_span * config.learn_limit;
        config.max_cost_perday = f32::INFINITY;
        config.review_limit = usize::MAX;
        // config.first_rating_prob = [0.2, 0.0, 0.8, 0.0];
        // config.review_rating_prob = [0.0, 1.0, 0.0];
        // config.state_rating_costs = [
        //     [19.4698, 19.4698, 19.4698, 19.4698],
        //     [23.185, 0.0, 7.8454, 0.0],
        //     [0.0, 0.0, 0.0, 0.0],
        // ];
        config.learning_step_count = 0;
        config.relearning_step_count = 0;
        for desired_retention in (72..=99).step_by(3).map(|x| x as f32 / 100.0) {
            dbg!(desired_retention);
            let start = Instant::now();
            let result_dp =
                expected_workload(&DEFAULT_PARAMETERS, desired_retention, &config).unwrap();
            let duration = start.elapsed();
            println!("DP Duration: {:?}", duration);
            let start = Instant::now();
            let result =
                simulate(&config, &DEFAULT_PARAMETERS, desired_retention, None, None).unwrap();
            let duration = start.elapsed();
            println!("Simulated Duration: {:?}", duration);
            let result_simulated =
                result.cost_per_day[result.cost_per_day.len() - 1] / config.learn_limit as f32;
            let relative_error = (result_dp - result_simulated).abs() / result_simulated;
            println!(
                "DP: {:.2}\tSimulated: {:.2}\tRelative Error: {:.2}",
                result_dp, result_simulated, relative_error
            );
            assert!(relative_error < 0.1);
        }
    }

    #[test]
    fn test_evaluate_in_flight_card_cost() -> Result<()> {
        let w = &check_and_fill_parameters(&DEFAULT_PARAMETERS)?;
        let config = SimulatorConfig {
            learn_span: 365,
            deck_size: 1,
            learn_limit: 0,
            max_cost_perday: f32::INFINITY,
            review_limit: usize::MAX,
            learning_step_count: 0,
            relearning_step_count: 0,
            ..Default::default()
        };
        for desired_retention in (72..=99).step_by(3).map(|x| x as f32 / 100.0) {
            dbg!(desired_retention);
            let mut estimator = WorkloadEstimator::new(&config);
            estimator.precompute_cost_matrix(desired_retention, w);
            let card = Card {
                difficulty: 5.0,
                stability: 5.0,
                last_date: -5.0,
                due: 5.0,
                interval: 10.0,
                desired_retention,
                ..Default::default()
            };
            let cost_dp = estimator.evaluate_in_flight_card_cost(&card, w);
            let mut costs = Vec::new();
            for seed in 0..1000 {
                let result = simulate(
                    &config,
                    w,
                    desired_retention,
                    Some(seed),
                    Some(vec![card.clone()]),
                )?;
                let cost_per_day = result.cost_per_day.iter().sum::<f32>();
                costs.push(cost_per_day);
            }
            let cost_simulated = costs.iter().sum::<f32>() / costs.len() as f32;
            let relative_error = (cost_dp - cost_simulated).abs() / cost_simulated;
            println!(
                "DP: {:.2}\tSimulated: {:.2}\tRelative Error: {:.2}",
                cost_dp, cost_simulated, relative_error
            );
            assert!(relative_error < 0.4);
        }
        Ok(())
    }

    #[test]
    fn test_introduced_cards_per_day() -> Result<()> {
        let existing_cards = vec![
            Card {
                // Already introduced
                stability: 5.0,
                difficulty: 5.0,
                last_date: 0.0,
                due: 5.0,
                interval: 5.0,
                ..Default::default()
            },
            Card {
                // New, to be learned on day 0
                id: 2,
                stability: f32::NEG_INFINITY,
                difficulty: f32::NEG_INFINITY,
                last_date: f32::NEG_INFINITY,
                due: 0.0,
                interval: f32::NEG_INFINITY,
                ..Default::default()
            },
            Card {
                // Already introduced
                stability: 5.0,
                difficulty: 5.0,
                last_date: 1.0,
                due: 6.0,
                interval: 5.0,
                ..Default::default()
            },
            Card {
                // New, to be learned on day 1
                stability: f32::NEG_INFINITY,
                difficulty: f32::NEG_INFINITY,
                last_date: f32::NEG_INFINITY,
                due: 1.0,
                interval: f32::NEG_INFINITY,
                ..Default::default()
            },
        ];

        let config = SimulatorConfig {
            learn_span: 4,
            learn_limit: 1, // Allow 1 new card to be learned each day
            deck_size: 6,
            review_limit: 100,
            max_cost_perday: f32::INFINITY,
            first_rating_prob: [0.0, 0.0, 1.0, 0.0], // Always rate 'Good' for simplicity
            ..Default::default()
        };

        let SimulationResult {
            introduced_cnt_per_day,
            ..
        } = simulate(
            &config,
            &DEFAULT_PARAMETERS,
            0.9,
            Some(0),
            Some(existing_cards),
        )?;

        assert_eq!(
            introduced_cnt_per_day,
            vec![3, 4, 5, 6],
            "introduced_cnt_per_day mismatch"
        );

        Ok(())
    }

    #[test]
    fn test_per_card_desired_retention() -> Result<()> {
        let dr_card = |dr: f32| Card {
            difficulty: 5.0,
            stability: 1000.0,
            last_date: -5.0,
            due: 0.0,
            interval: 5.0,
            desired_retention: dr,
            ..Default::default()
        };

        let cards = vec![dr_card(0.8), dr_card(0.9), dr_card(0.9)];

        let config = SimulatorConfig {
            deck_size: cards.len(),
            learn_span: 100,
            review_rating_prob: [0.0, 1.0, 0.0], // always good
            ..Default::default()
        };

        let result = simulate(&config, &DEFAULT_PARAMETERS, 0.9, Some(42), Some(cards))?;

        let card1 = &result.cards[0];
        let card2 = &result.cards[1];
        let card3 = &result.cards[2];

        assert!(
            card1.interval > card2.interval,
            "Cards with a lower desired retention should have a longer interval."
        );
        assert!(
            card3.interval == card2.interval,
            "Cards with the same desired retention should have the same interval."
        );

        Ok(())
    }

    #[test]
    fn test_per_card_parameters() -> Result<()> {
        let good_card = |initial_good: f32| {
            let mut w = DEFAULT_PARAMETERS;
            w[2] = initial_good;
            let parameters = Arc::new(w.to_vec());
            Card {
                difficulty: 5.0,
                stability: 2e-9, // not-filtered
                due: 0.,
                desired_retention: 0.9,
                parameters: parameters.clone(),
                ..Default::default()
            }
        };

        let cards = vec![good_card(5.), good_card(6.), good_card(7.)];

        let config = SimulatorConfig {
            deck_size: cards.len(),
            learn_span: 1,
            learning_step_count: 1,
            first_rating_prob: [0.0, 0.0, 1.0, 0.0], // always good
            ..Default::default()
        };

        let result = simulate(&config, &DEFAULT_PARAMETERS, 0.9, Some(42), Some(cards))?;

        let card1 = &result.cards[0];
        let card2 = &result.cards[1];
        let card3 = &result.cards[2];

        assert_eq!(card1.interval, 5.,);
        assert_eq!(card2.interval, 6.,);
        assert_eq!(card3.interval, 7.,);

        Ok(())
    }
}

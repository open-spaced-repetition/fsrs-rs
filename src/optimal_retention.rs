use crate::FSRS;
use crate::error::{FSRSError, Result};
use crate::inference::{ItemProgress, Parameters, S_MAX, S_MIN};
use crate::model::check_and_fill_parameters;
use burn::tensor::backend::Backend;
use itertools::{Itertools, izip};
use ndarray_rand::rand::distributions::WeightedIndex;
use ndarray_rand::rand_distr::Distribution;
use priority_queue::PriorityQueue;
use rand::Rng;
use rand::{SeedableRng, rngs::StdRng};
use rayon::iter::IntoParallelIterator;
use rayon::iter::ParallelIterator;
use std::cmp::Reverse;
use std::collections::HashMap;
use std::sync::Arc;

#[derive(Debug)]
pub struct SimulationResult {
    pub memorized_cnt_per_day: Vec<f32>,
    pub review_cnt_per_day: Vec<usize>,
    pub learn_cnt_per_day: Vec<usize>,
    pub cost_per_day: Vec<f32>,
    // The amount of review cards you got correct on a given day (not including learn cards).
    pub correct_cnt_per_day: Vec<usize>,
    pub cards: Vec<Card>,
}

trait Round {
    fn to_2_decimal(self) -> f32;
}

impl Round for f32 {
    fn to_2_decimal(self) -> f32 {
        (self * 100.0).round() / 100.0
    }
}

const R_MIN: f32 = 0.70;
const R_MAX: f32 = 0.95;
const RATINGS: [usize; 4] = [1, 2, 3, 4];
const LEARNING: usize = 0;
const REVIEW: usize = 1;
const RELEARNING: usize = 2;
const MAX_STEPS: usize = 5;

/// Function type for post scheduling operations that takes interval, maximum interval,
/// current day index, due counts per day, and a random number generator,
/// and returns a new interval.
#[allow(clippy::type_complexity)]
pub struct PostSchedulingFn(
    pub Arc<dyn Fn(&Card, f32, usize, &[usize], &mut StdRng) -> f32 + Sync + Send>,
);

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
pub struct ReviewPriorityFn(pub Arc<dyn Fn(&Card, &Parameters) -> i32 + Sync + Send>);

impl PartialEq for ReviewPriorityFn {
    fn eq(&self, _: &Self) -> bool {
        true
    }
}

impl std::fmt::Debug for ReviewPriorityFn {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Wrap(<function>)")
    }
}

impl Default for ReviewPriorityFn {
    fn default() -> Self {
        Self(Arc::new(|card, _w| (card.difficulty * 100.0) as i32))
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
            learning_step_transitions: [
                [0.3687, 0.0628, 0.5108, 0.0577],
                [0.0441, 0.4553, 0.4457, 0.0549],
                [0.0518, 0.0470, 0.8462, 0.0550],
            ],
            relearning_step_transitions: [
                [0.2157, 0.0643, 0.6595, 0.0605],
                [0.0500, 0.4638, 0.4475, 0.0387],
                [0.1056, 0.1434, 0.7266, 0.0244],
            ],
            state_rating_costs: [
                [12.75, 12.26, 8.0, 6.38],
                [13.05, 11.74, 7.42, 5.6],
                [10.56, 10.0, 7.37, 5.4],
            ],
            learning_step_count: 2,
            relearning_step_count: 1,
        }
    }
}

fn stability_after_success(w: &[f32], s: f32, r: f32, d: f32, rating: usize) -> f32 {
    let hard_penalty = if rating == 2 { w[15] } else { 1.0 };
    let easy_bonus = if rating == 4 { w[16] } else { 1.0 };
    (s * (f32::exp(w[8])
        * (11.0 - d)
        * s.powf(-w[9])
        * (f32::exp((1.0 - r) * w[10]) - 1.0)
        * hard_penalty)
        .mul_add(easy_bonus, 1.0))
    .clamp(S_MIN, S_MAX)
}

fn stability_after_failure(w: &[f32], s: f32, r: f32, d: f32) -> f32 {
    let new_s_min = s / (w[17] * w[18]).exp();
    let new_s =
        (w[11] * d.powf(-w[12]) * ((s + 1.0).powf(w[13]) - 1.0) * f32::exp((1.0 - r) * w[14]))
            .min(new_s_min);
    new_s.clamp(S_MIN, S_MAX)
}

fn stability_short_term(w: &[f32], s: f32, rating: usize) -> f32 {
    let sinc = (w[17] * (rating as f32 - 3.0 + w[18])).exp() * s.powf(-w[19]);
    let new_s = s * if rating >= 3 { sinc.max(1.0) } else { sinc };
    new_s.clamp(S_MIN, S_MAX)
}

#[allow(clippy::too_many_arguments)]
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
        new_s = stability_short_term(w, new_s, rating);
        new_d = next_d(w, new_d, rating);
        cost += rating_costs[rating - 1];
        if rating > 2 {
            consecutive += 1;
        } else if rating == 1 {
            consecutive = 0;
        }
    }
    (new_s, new_d, cost)
}

fn init_d(w: &[f32], rating: usize) -> f32 {
    w[4] - (w[5] * (rating - 1) as f32).exp() + 1.0
}

fn linear_damping(delta_d: f32, old_d: f32) -> f32 {
    (10.0 - old_d) / 9.0 * delta_d
}

fn next_d(w: &[f32], d: f32, rating: usize) -> f32 {
    let delta_d = -w[6] * (rating as f32 - 3.0);
    let new_d = d + linear_damping(delta_d, d);
    mean_reversion(w, init_d(w, 4), new_d).clamp(1.0, 10.0)
}

fn mean_reversion(w: &[f32], init: f32, current: f32) -> f32 {
    w[7] * init + (1.0 - w[7]) * current
}

fn power_forgetting_curve(w: &[f32], t: f32, s: f32) -> f32 {
    let decay = -w[20];
    let factor = 0.9f32.powf(1.0 / decay) - 1.0;
    (t / s).mul_add(factor, 1.0).powf(decay)
}

fn next_interval(w: &[f32], stability: f32, desired_retention: f32) -> f32 {
    let decay = -w[20];
    let factor = 0.9f32.powf(1.0 / decay) - 1.0;
    stability / factor * (desired_retention.powf(1.0 / decay) - 1.0)
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
}

impl Card {
    pub fn retrievability(&self, w: &[f32]) -> f32 {
        power_forgetting_curve(w, self.due - self.last_date, self.stability)
    }

    pub fn scheduled_due(&self) -> f32 {
        self.last_date + self.interval
    }
}

pub fn simulate(
    config: &SimulatorConfig,
    w: &Parameters,
    desired_retention: f32,
    seed: Option<u64>,
    existing_cards: Option<Vec<Card>>,
) -> Result<SimulationResult, FSRSError> {
    let w = &check_and_fill_parameters(w)?;
    if config.deck_size == 0 {
        return Err(FSRSError::InvalidDeckSize);
    }

    let mut review_cnt_per_day = vec![0; config.learn_span];
    let mut learn_cnt_per_day = vec![0; config.learn_span];
    let mut memorized_cnt_per_day = vec![0.0; config.learn_span];
    let mut cost_per_day = vec![0.0; config.learn_span];
    let mut due_cnt_per_day = vec![0; config.learn_span + config.learn_span / 2];
    let mut correct_cnt_per_day = vec![0; config.learn_span];

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
        });

        cards.extend(init_ratings);
    }

    let mut card_priorities = PriorityQueue::new();
    let max_lapses = config.suspend_after_lapses.unwrap_or(u32::MAX);

    let review_priority_fn = config.review_priority_fn.clone().unwrap_or_default();

    fn card_priority(
        card: &Card,
        learn: bool,
        w: &Parameters,
        ReviewPriorityFn(cb): &ReviewPriorityFn,
    ) -> Reverse<(i32, bool, i32)> {
        let priority = cb(card, w);
        // high priority for early due, review, custom priority
        Reverse((card.due as i32, learn, priority))
    }

    for (i, card) in cards.iter().enumerate() {
        card_priorities.push(
            i,
            card_priority(
                card,
                card.last_date == f32::NEG_INFINITY,
                w,
                &review_priority_fn,
            ),
        );
    }

    // Main simulation loop
    while let Some((&card_index, _)) = card_priorities.peek() {
        let card = &mut cards[card_index];

        let day_index = card.due as usize;

        let is_learn = card.last_date == f32::NEG_INFINITY;

        let last_date_index = card.last_date as usize;

        // Guards
        if card.due >= config.learn_span as f32 || card.lapses >= max_lapses {
            if !is_learn {
                let delta_t = config.learn_span.max(last_date_index) - last_date_index;
                let pre_sim_days = (-card.last_date) as usize;
                for i in 0..delta_t {
                    memorized_cnt_per_day[last_date_index + i] +=
                        power_forgetting_curve(w, (pre_sim_days + i) as f32, card.stability);
                }
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
                &card_index,
                card_priority(card, is_learn, w, &review_priority_fn),
            );
            continue;
        }

        // dbg!(&day_index);
        if is_learn {
            // For learning cards
            // Initialize stability and difficulty for new cards
            let init_rating = first_rating_choices[first_rating_dist.sample(&mut rng)];
            let init_stability = w[init_rating - 1];
            let init_difficulty = init_d(w, init_rating).clamp(1.0, 10.0);
            let (new_s, new_d, cost) = memory_state_short_term(
                w,
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
        } else {
            // For review cards
            let last_stability = card.stability;

            // Calculate retrievability for entries where has_learned is true
            let retrievability = card.retrievability(w);

            // Create 'forget' mask
            let forget = !rng.gen_bool(retrievability as f64);

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
                let post_lapse_stability =
                    stability_after_failure(w, last_stability, retrievability, card.difficulty);
                let post_lapse_difficulty = next_d(w, card.difficulty, rating);
                let (new_s, new_d, cost) = memory_state_short_term(
                    w,
                    post_lapse_stability,
                    post_lapse_difficulty,
                    None,
                    &config.state_rating_costs[RELEARNING],
                    &config.relearning_step_transitions,
                    config.relearning_step_count,
                    &mut rng,
                );
                (
                    new_s,
                    new_d,
                    config.state_rating_costs[REVIEW][rating - 1] + cost,
                )
            } else {
                (
                    stability_after_success(
                        w,
                        last_stability,
                        retrievability,
                        card.difficulty,
                        rating,
                    ),
                    next_d(w, card.difficulty, rating),
                    config.state_rating_costs[REVIEW][rating - 1],
                )
            };
            card.stability = new_s;
            card.difficulty = new_d;

            // Update days statistics
            review_cnt_per_day[day_index] += 1;
            cost_per_day[day_index] += cost;

            let delta_t = day_index - last_date_index;
            let pre_sim_days = (-card.last_date) as usize;
            for i in 0..delta_t {
                memorized_cnt_per_day[last_date_index + i] +=
                    power_forgetting_curve(w, (pre_sim_days + i) as f32, last_stability);
            }
        }

        let mut ivl = next_interval(w, card.stability, desired_retention)
            .round()
            .clamp(1.0, config.max_ivl);

        card.last_date = day_index as f32;
        card.interval = ivl;
        card.due = day_index as f32 + ivl;

        if let Some(PostSchedulingFn(cb)) = &config.post_scheduling_fn {
            ivl = cb(card, config.max_ivl, day_index, &due_cnt_per_day, &mut rng);
            card.interval = ivl;
            card.due = day_index as f32 + ivl;
        }

        if card.due < due_cnt_per_day.len() as f32 {
            due_cnt_per_day[card.due as usize] += 1;
        }

        card_priorities.change_priority(
            &card_index,
            card_priority(card, false, w, &review_priority_fn),
        );
    }

    /*dbg!((
        &memorized_cnt_per_day[learn_span - 1],
        &review_cnt_per_day[learn_span - 1],
        &learn_cnt_per_day[learn_span - 1],
        &cost_per_day[learn_span - 1],
    ));*/

    Ok(SimulationResult {
        memorized_cnt_per_day,
        review_cnt_per_day,
        learn_cnt_per_day,
        cost_per_day,
        correct_cnt_per_day,
        cards,
    })
}

fn sample<F>(
    config: &SimulatorConfig,
    parameters: &Parameters,
    desired_retention: f32,
    n: usize,
    progress: &mut F,
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
            let SimulationResult {
                memorized_cnt_per_day,
                cost_per_day,
                ..
            } = simulate(
                config,
                parameters,
                desired_retention,
                Some((i + 42).try_into().unwrap()),
                None,
            )?;
            let total_memorized = memorized_cnt_per_day[memorized_cnt_per_day.len() - 1];
            let total_cost = cost_per_day.iter().sum::<f32>();
            Ok(total_cost / total_memorized)
        })
        .collect();
    results.map(|v| v.iter().sum::<f32>() / n as f32)
}

impl<B: Backend> FSRS<B> {
    /// For the given simulator parameters and parameters, determine the suggested `desired_retention`
    /// value.
    pub fn optimal_retention<F>(
        &self,
        config: &SimulatorConfig,
        parameters: &Parameters,
        mut progress: F,
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

        Self::brent(config, parameters, inc_progress)
    }
    /// https://argmin-rs.github.io/argmin/argmin/solver/brent/index.html
    /// https://github.com/scipy/scipy/blob/5e4a5e3785f79dd4e8930eed883da89958860db2/scipy/optimize/_optimize.py#L2446
    fn brent<F>(
        config: &SimulatorConfig,
        parameters: &Parameters,
        mut progress: F,
    ) -> Result<f32, FSRSError>
    where
        F: FnMut() -> bool,
    {
        let mintol = 1e-10;
        let cg = 0.381_966;
        let maxiter = 64;
        let tol = 0.01f32;

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
            sample(config, parameters, R_MIN, sample_size, &mut progress)?,
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
            let fu = sample(config, parameters, u, sample_size, &mut progress)?;

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

    fn median(x: &mut [u32]) -> u32 {
        x.sort_unstable();
        let n = x.len();
        if n % 2 == 0 {
            (x[n / 2 - 1] + x[n / 2]) / 2
        } else {
            x[n / 2]
        }
    }

    // Calculate median costs
    for ((state, rating), durations) in state_rating_durations.iter() {
        let mut durations = durations.clone();
        let median_duration = median(&mut durations);
        state_rating_costs[*state][*rating] = median_duration as f32 / 1000.0;
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
    use super::*;
    use crate::{DEFAULT_PARAMETERS, convertor_tests::read_collection};
    const LEARN_COST: f32 = 42.;
    const REVIEW_COST: f32 = 43.;

    #[test]
    fn test_memory_state_short_term() {
        let w = DEFAULT_PARAMETERS;
        let config = SimulatorConfig::default();
        let mut rng = StdRng::seed_from_u64(42);

        // Expected results for each init_rating
        let expected_results = [
            (0.24325262, 8.282093, 41.5), // init_rating = 1
            (1.7962034, 6.191415, 28.26), // init_rating = 2
            (3.5362437, 4.8680573, 16.0), // init_rating = 3
            (16.1507, 2.4824386, 6.38),   // init_rating = 4
        ];

        // Test for each init_rating from 1 to 4
        for init_rating in 1..=4 {
            let s = w[init_rating - 1];
            let d = init_d(&w, init_rating);
            dbg!(s, d);

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
        let post_lapse_s = stability_after_failure(&w, s, 0.9, d);
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
        assert_eq!(result, (2.4895842, 7.2628965, 7.37));
    }

    #[test]
    fn simulator_memorization() -> Result<()> {
        let config = SimulatorConfig::default();
        let SimulationResult {
            memorized_cnt_per_day,
            ..
        } = simulate(&config, &DEFAULT_PARAMETERS, 0.9, None, None)?;
        assert_eq!(
            memorized_cnt_per_day[memorized_cnt_per_day.len() - 1],
            6757.842
        );
        Ok(())
    }

    #[test]
    fn simulator_learn_review_costs() -> Result<()> {
        let config = SimulatorConfig {
            deck_size: 1,
            learn_span: 1,
            first_rating_prob: [0., 0., 1., 0.],
            state_rating_costs: [[LEARN_COST; 4], [REVIEW_COST; 4], [0.; 4]],
            learning_step_count: 1,
            ..Default::default()
        };

        let cards = vec![Card {
            id: 0,
            difficulty: 5.0,
            stability: 5.0,
            last_date: -5.0,
            due: 0.0,
            interval: 5.0,
            lapses: 0,
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
    fn changing_learn_span_should_get_same_review_cnt_per_day() -> Result<()> {
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
    fn simulate_with_existing_cards() -> Result<()> {
        let config = SimulatorConfig {
            learn_span: 30,
            learn_limit: 60,
            review_limit: 200,
            max_cost_perday: f32::INFINITY,
            ..Default::default()
        };
        let cards = vec![
            Card {
                id: 0,
                difficulty: 5.0,
                stability: 5.0,
                last_date: -5.0,
                due: 0.0,
                interval: 5.0,
                lapses: 0,
            },
            Card {
                id: 0,
                difficulty: 5.0,
                stability: 2.0,
                last_date: -2.0,
                due: 0.0,
                interval: 2.0,
                lapses: 0,
            },
            Card {
                id: 0,
                difficulty: 5.0,
                stability: 2.0,
                last_date: -2.0,
                due: 1.0,
                interval: 3.0,
                lapses: 0,
            },
            Card {
                id: 0,
                difficulty: 5.0,
                stability: 2.0,
                last_date: -8.0,
                due: -1.0,
                interval: 7.0,
                lapses: 0,
            },
        ];
        let SimulationResult {
            memorized_cnt_per_day,
            review_cnt_per_day,
            learn_cnt_per_day,
            ..
        } = simulate(&config, &DEFAULT_PARAMETERS, 0.9, None, Some(cards))?;
        assert_eq!(memorized_cnt_per_day[0], 63.9);
        assert_eq!(review_cnt_per_day[0], 3);
        assert_eq!(learn_cnt_per_day[0], 60);
        Ok(())
    }

    #[test]
    fn simulate_suspend_on_lapse_count() -> Result<()> {
        let cards = vec![Card {
            id: 0,
            difficulty: 10.0,
            stability: f32::EPSILON,
            last_date: -5.0,
            due: 0.0,
            interval: 5.0,
            lapses: 0,
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
    fn simulate_with_learn_limit() -> Result<()> {
        let config = SimulatorConfig {
            learn_limit: 3,
            review_limit: 10,
            learn_span: 3,
            ..Default::default()
        };

        let cards = vec![
            Card {
                id: 0,
                difficulty: 5.0,
                stability: 5.0,
                last_date: -5.0,
                due: 0.0,
                interval: 5.0,
                lapses: 0
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
    fn simulate_with_new_affects_review_limit() -> Result<()> {
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
                id: 0,
                difficulty: 5.0,
                stability: 500.0,
                last_date: -5.0,
                due: 0.0,
                interval: 5.0,
                lapses: 0
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
    fn simulate_with_learn_review_limit() -> Result<()> {
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
                0, 19, 34, 49, 68, 69, 90, 87, 109, 129, 106, 112, 135, 139, 144, 157, 124, 152,
                179, 172, 188, 188, 173, 200, 181, 200, 199, 200, 200, 200
            ]
        );
        assert_eq!(
            learn_cnt_per_day.to_vec(),
            vec![config.learn_limit; config.learn_span]
        );
        Ok(())
    }

    #[test]
    fn simulate_with_max_ivl() -> Result<()> {
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
            6547.6294
        );
        Ok(())
    }

    #[test]
    fn simulate_with_zero_cards() -> Result<()> {
        let config = SimulatorConfig {
            deck_size: 0,
            ..Default::default()
        };
        let results = simulate(&config, &DEFAULT_PARAMETERS, 0.9, None, None);
        assert_eq!(results.unwrap_err(), FSRSError::InvalidDeckSize);
        Ok(())
    }

    #[test]
    fn simulate_returns_cards() -> Result<()> {
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
    fn simulate_with_existing_cards_with_wrong_deck_size() -> Result<()> {
        let config = SimulatorConfig {
            deck_size: 1,
            ..Default::default()
        };
        let cards = vec![
            Card {
                id: 0,
                difficulty: 5.0,
                stability: 5.0,
                last_date: -5.0,
                due: 0.0,
                interval: 5.0,
                lapses: 0,
            },
            Card {
                id: 0,
                difficulty: 5.0,
                stability: 2.0,
                last_date: -2.0,
                due: 0.0,
                interval: 2.0,
                lapses: 0,
            },
        ];
        let results = simulate(&config, &DEFAULT_PARAMETERS, 0.9, None, Some(cards));
        assert_eq!(results.unwrap_err(), FSRSError::InvalidDeckSize);
        Ok(())
    }

    #[test]
    fn learn_does_not_affect_correct_count() -> Result<()> {
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
                id: 0,
                difficulty: 5.0,
                stability: f32::INFINITY,
                last_date: -5.0,
                due: 1.0,
                interval: 5.0,
                lapses: 0,
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
    fn simulate_with_post_scheduling_fn() -> Result<()> {
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
    fn simulate_with_review_priority_fn() -> Result<()> {
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
        run_test!(None, 42.339848)?;
        println!("High difficulty cards reviewed first.");
        run_test!(
            wrap!(|card: &Card, _w: &Parameters| -(card.difficulty * 100.0) as i32),
            43.859474
        )?;
        println!("Low retrievability cards reviewed first.");
        run_test!(
            wrap!(|card: &Card, w: &Parameters| (card.retrievability(w) * 1000.0) as i32),
            43.482998
        )?;
        println!("High retrievability cards reviewed first.");
        run_test!(
            wrap!(|card: &Card, w: &Parameters| -(card.retrievability(w) * 1000.0) as i32),
            41.110588
        )?;
        println!("High stability cards reviewed first.");
        run_test!(
            wrap!(|card: &Card, _w: &Parameters| -(card.stability * 100.0) as i32),
            41.006256
        )?;
        println!("Low stability cards reviewed first.");
        run_test!(
            wrap!(|card: &Card, _w: &Parameters| (card.stability * 100.0) as i32),
            43.077534
        )?;
        println!("Long interval cards reviewed first.");
        run_test!(
            wrap!(|card: &Card, _w: &Parameters| -card.interval as i32),
            40.92563
        )?;
        println!("Short interval cards reviewed first.");
        run_test!(
            wrap!(|card: &Card, _w: &Parameters| card.interval as i32),
            43.282585
        )?;
        println!("Early scheduled due cards reviewed first.");
        run_test!(
            wrap!(|card: &Card, _w: &Parameters| card.scheduled_due() as i32),
            43.309185
        )?;
        println!("Late scheduled due cards reviewed first.");
        run_test!(
            wrap!(|card: &Card, _w: &Parameters| -card.scheduled_due() as i32),
            42.79738
        )?;
        Ok(())
    }

    #[test]
    fn optimal_retention() -> Result<()> {
        let learn_span = 1000;
        let learn_limit = 10;
        let fsrs = FSRS::new(None)?;
        let config = SimulatorConfig {
            deck_size: learn_span * learn_limit,
            learn_span,
            max_cost_perday: f32::INFINITY,
            learn_limit,
            ..Default::default()
        };
        let optimal_retention = fsrs.optimal_retention(&config, &[], |_| true).unwrap();
        dbg!(optimal_retention);
        assert!((optimal_retention - 0.7).abs() < 0.01);
        assert!(fsrs.optimal_retention(&config, &[1.], |_| true).is_err());
        Ok(())
    }

    #[test]
    fn optimal_retention_with_old_parameters() -> Result<()> {
        let learn_span = 1000;
        let learn_limit = 10;
        let fsrs = FSRS::new(None)?;
        let config = SimulatorConfig {
            deck_size: learn_span * learn_limit,
            learn_span,
            max_cost_perday: f32::INFINITY,
            learn_limit,
            ..Default::default()
        };
        let mut param = DEFAULT_PARAMETERS[..17].to_vec();
        param.extend_from_slice(&[0.0, 0.0]);
        let optimal_retention = fsrs.optimal_retention(&config, &param, |_v| true).unwrap();
        assert_eq!(optimal_retention, 0.7);
        Ok(())
    }

    #[test]
    fn extract_simulator_config_from_revlog() {
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
                    [9.922, 0.0, 7.524, 5.467],
                    [9.318, 6.84, 5.83, 3.991],
                    [9.912, 0.0, 4.671, 4.731]
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
                    [0.12520419, 0.0045695365, 0.26328918, 0.6069371],
                    [0.059351854, 0.44009256, 0.43120366, 0.06935185],
                    [0.019313155, 0.0038998832, 0.5544936, 0.42229337]
                ],
                relearning_step_transitions: [
                    [0.050443545, 0.004855989, 0.24766704, 0.6970334],
                    [0.06481481, 0.44796297, 0.43287036, 0.05435185],
                    [0.048842106, 0.043, 0.5964737, 0.31168422]
                ],
                state_rating_costs: [
                    [10.08, 12.26, 7.54, 5.47],
                    [9.54, 7.08, 5.84, 4.2],
                    [10.26, 10.0, 5.08, 4.76]
                ],
                ..Default::default()
            }
        );
    }

    #[test]
    fn extract_simulator_config_without_revlog() {
        let simulator_config = extract_simulator_config(vec![], 0, true);
        assert_eq!(simulator_config, SimulatorConfig::default());
    }
}

use crate::error::{FSRSError, Result};
use crate::inference::{ItemProgress, Weights};
use crate::{DEFAULT_WEIGHTS, FSRS};
use burn::tensor::backend::Backend;
use itertools::{izip, Itertools};
use ndarray::{s, Array1, Array2, Ix0, Ix1, SliceInfoElem, Zip};
use ndarray_rand::rand_distr::Distribution;
use ndarray_rand::RandomExt;
use rand::{
    distributions::{Uniform, WeightedIndex},
    rngs::StdRng,
    SeedableRng,
};
use rayon::iter::IntoParallelIterator;
use rayon::iter::ParallelIterator;
use std::sync::{Arc, Mutex};
use strum::EnumCount;

#[derive(Debug, EnumCount)]
enum Column {
    Difficulty,
    Stability,
    #[allow(unused)]
    Retrievability,
    #[allow(unused)]
    DeltaT,
    LastDate,
    Due,
    Interval,
    #[allow(unused)]
    Cost,
    #[allow(unused)]
    Rand,
}

impl ndarray::SliceNextDim for Column {
    type InDim = Ix1;
    type OutDim = Ix0;
}

impl From<Column> for SliceInfoElem {
    fn from(value: Column) -> Self {
        SliceInfoElem::Index(value as isize)
    }
}

#[derive(Debug, Clone)]
pub struct SimulatorConfig {
    pub deck_size: usize,
    pub learn_span: usize,
    pub max_cost_perday: f64,
    pub max_ivl: f64,
    pub recall_costs: [f64; 3],
    pub forget_cost: f64,
    pub learn_cost: f64,
    pub first_rating_prob: [f64; 4],
    pub review_rating_prob: [f64; 3],
    pub loss_aversion: f64,
}

impl Default for SimulatorConfig {
    fn default() -> SimulatorConfig {
        SimulatorConfig {
            deck_size: 10000,
            learn_span: 365,
            max_cost_perday: 1800.0,
            max_ivl: 36500.0,
            recall_costs: [14.0, 10.0, 6.0],
            forget_cost: 50.0,
            learn_cost: 20.0,
            first_rating_prob: [0.15, 0.2, 0.6, 0.05],
            review_rating_prob: [0.3, 0.6, 0.1],
            loss_aversion: 2.5,
        }
    }
}

fn stability_after_success(w: &[f64], s: f64, r: f64, d: f64, response: usize) -> f64 {
    let hard_penalty = if response == 1 { w[15] } else { 1.0 };
    let easy_bonus = if response == 3 { w[16] } else { 1.0 };
    s * (1.0
        + f64::exp(w[8])
            * (11.0 - d)
            * s.powf(-w[9])
            * (f64::exp((1.0 - r) * w[10]) - 1.0)
            * hard_penalty
            * easy_bonus)
}

fn stability_after_failure(w: &[f64], s: f64, r: f64, d: f64) -> f64 {
    s.min(w[11] * d.powf(-w[12]) * ((s + 1.0).powf(w[13]) - 1.0) * f64::exp((1.0 - r) * w[14]))
        .max(0.1)
}

fn simulate(config: &SimulatorConfig, w: &[f64], request_retention: f64, seed: Option<u64>) -> f64 {
    let SimulatorConfig {
        deck_size,
        learn_span,
        max_cost_perday,
        max_ivl,
        recall_costs,
        forget_cost,
        learn_cost,
        first_rating_prob,
        review_rating_prob,
        loss_aversion,
    } = config.clone();
    let mut card_table = Array2::<f64>::zeros((Column::COUNT, deck_size));
    card_table
        .slice_mut(s![Column::Due, ..])
        .fill(learn_span as f64);
    card_table.slice_mut(s![Column::Difficulty, ..]).fill(1e-10);
    card_table.slice_mut(s![Column::Stability, ..]).fill(1e-10);

    // let mut review_cnt_per_day = Array1::<f64>::zeros(learn_span);
    // let mut learn_cnt_per_day = Array1::<f64>::zeros(learn_span);
    let mut memorized_cnt_per_day = Array1::<f64>::zeros(learn_span);

    let first_rating_choices = [0, 1, 2, 3];
    let first_rating_dist = WeightedIndex::new(first_rating_prob).unwrap();

    let review_rating_choices = [1, 2, 3];
    let review_rating_dist = WeightedIndex::new(review_rating_prob).unwrap();

    let mut rng = StdRng::seed_from_u64(seed.unwrap_or(42));

    // Main simulation loop
    for today in 0..learn_span {
        let old_stability = card_table.slice(s![Column::Stability, ..]);
        let has_learned = old_stability.mapv(|x| x > 1e-9);
        let old_last_date = card_table.slice(s![Column::LastDate, ..]);

        // Updating delta_t for 'has_learned' cards
        let mut delta_t = Array1::zeros(deck_size); // Create an array of the same length for delta_t

        // Calculate delta_t for entries where has_learned is true
        izip!(&mut delta_t, &old_last_date, &has_learned)
            .filter(|(.., &has_learned_flag)| has_learned_flag)
            .for_each(|(delta_t, &last_date, ..)| {
                *delta_t = today as f64 - last_date;
            });

        let mut retrievability = Array1::zeros(deck_size); // Create an array for retrievability

        // Calculate retrievability for entries where has_learned is true
        izip!(&mut retrievability, &delta_t, &old_stability, &has_learned)
            .filter(|(.., &has_learned_flag)| has_learned_flag)
            .for_each(|(retrievability, &delta_t, &stability, ..)| {
                *retrievability = (1.0 + delta_t / (9.0 * stability)).powf(-1.0)
            });

        // Set 'cost' column to 0
        let mut cost = Array1::<f64>::zeros(deck_size);

        // Create 'need_review' mask
        let old_due = card_table.slice(s![Column::Due, ..]);
        let need_review = old_due.mapv(|x| x <= today as f64);

        // dbg!(&need_review.mapv(|x| x as i32).sum());

        // Update 'rand' column for 'need_review' entries
        let mut rand_slice = Array1::zeros(deck_size);
        let n_need_review = need_review.iter().filter(|&&x| x).count();
        let random_values = Array1::random_using(n_need_review, Uniform::new(0.0, 1.0), &mut rng);

        rand_slice
            .iter_mut()
            .zip(&need_review)
            .filter(|(_, &need_review_flag)| need_review_flag)
            .map(|(x, _)| x)
            .zip(random_values)
            .for_each(|(rand_elem, random_value)| {
                *rand_elem = random_value;
            });

        // Create 'forget' mask
        let forget = Zip::from(&rand_slice)
            .and(&retrievability)
            .map_collect(|&rand_val, &retriev_val| rand_val > retriev_val);

        // Sample 'rating' for 'need_review' entries
        let mut ratings = Array1::zeros(deck_size);
        izip!(&mut ratings, &need_review)
            .filter(|(_, &need_review_flag)| need_review_flag)
            .for_each(|(rating, _)| {
                *rating = review_rating_choices[review_rating_dist.sample(&mut rng)]
            });

        // Update 'cost' column based on 'need_review', 'forget' and 'ratings'
        izip!(&mut cost, &need_review, &forget, &ratings)
            .filter(|(_, &need_review_flag, _, _)| need_review_flag)
            .for_each(|(cost, _, &forget_flag, &rating)| {
                *cost = if forget_flag {
                    forget_cost * loss_aversion
                } else {
                    recall_costs[rating - 1]
                }
            });

        // Calculate cumulative sum of 'cost'
        let mut cum_sum = Array1::<f64>::zeros(deck_size);
        for i in 1..deck_size {
            cum_sum[i] = cum_sum[i - 1] + cost[i];
        }

        // Create 'true_review' mask based on 'need_review' and 'cum_sum'
        let true_review =
            Zip::from(&need_review)
                .and(&cum_sum)
                .map_collect(|&need_review_flag, &cum_cost| {
                    need_review_flag && (cum_cost <= max_cost_perday)
                });

        let need_learn = old_due.mapv(|x| x == learn_span as f64);
        // Update 'cost' column based on 'need_learn'
        izip!(&mut cost, &need_learn)
            .filter(|(_, &need_learn_flag)| need_learn_flag)
            .for_each(|(cost, _)| {
                *cost = learn_cost;
            });

        for i in 1..deck_size {
            cum_sum[i] = cum_sum[i - 1] + cost[i];
        }

        // dbg!(&cum_sum);

        // Create 'true_learn' mask based on 'need_learn' and 'cum_sum'
        let true_learn =
            Zip::from(&need_learn)
                .and(&cum_sum)
                .map_collect(|&need_learn_flag, &cum_cost| {
                    need_learn_flag && (cum_cost <= max_cost_perday)
                });

        // Sample 'rating' for 'true_learn' entries
        izip!(&mut ratings, &true_learn)
            .filter(|(_, &true_learn_flag)| true_learn_flag)
            .for_each(|(rating, _)| {
                *rating = first_rating_choices[first_rating_dist.sample(&mut rng)]
            });

        let mut new_stability = old_stability.to_owned();
        let old_difficulty = card_table.slice(s![Column::Difficulty, ..]);
        // Iterate over slices and apply stability_after_failure function
        izip!(
            &mut new_stability,
            &old_stability,
            &retrievability,
            &old_difficulty,
            &(&true_review & &forget)
        )
        .filter(|(.., &condition)| condition)
        .for_each(|(new_stab, &stab, &retr, &diff, ..)| {
            *new_stab = stability_after_failure(w, stab, retr, diff);
        });

        // Iterate over slices and apply stability_after_success function
        izip!(
            &mut new_stability,
            &ratings,
            &old_stability,
            &retrievability,
            &old_difficulty,
            &(&true_review & !&forget)
        )
        .filter(|(.., &condition)| condition)
        .for_each(|(new_stab, &rating, &stab, &retr, &diff, _)| {
            *new_stab = stability_after_success(w, stab, retr, diff, rating);
        });

        // Initialize a new Array1 to store updated difficulty values
        let mut new_difficulty = old_difficulty.to_owned();

        // Update the difficulty values based on the condition 'true_review & forget'
        izip!(&mut new_difficulty, &old_difficulty, &true_review, &forget)
            .filter(|(.., &true_rev, &frgt)| true_rev && frgt)
            .for_each(|(new_diff, &old_diff, ..)| {
                *new_diff = (old_diff + 2.0 * w[6]).max(1.0).min(10.0);
            });

        // Update 'last_date' column where 'true_review' or 'true_learn' is true
        let mut new_last_date = old_last_date.to_owned();
        izip!(&mut new_last_date, &true_review, &true_learn)
            .filter(|(_, &true_review_flag, &true_learn_flag)| true_review_flag || true_learn_flag)
            .for_each(|(new_last_date, ..)| {
                *new_last_date = today as f64;
            });

        izip!(
            &mut new_stability,
            &mut new_difficulty,
            &ratings,
            &true_learn
        )
        .filter(|(.., &true_learn_flag)| true_learn_flag)
        .for_each(|(new_stab, new_diff, &rating, _)| {
            *new_stab = w[rating];
            *new_diff = w[4] - w[5] * (rating as f64 - 3.0);
        });
        let old_interval = card_table.slice(s![Column::Interval, ..]);
        let mut new_interval = old_interval.to_owned();
        izip!(&mut new_interval, &new_stability, &true_review, &true_learn)
            .filter(|(.., &true_review_flag, &true_learn_flag)| true_review_flag || true_learn_flag)
            .for_each(|(new_ivl, &new_stab, ..)| {
                *new_ivl = (9.0 * new_stab * (1.0 / request_retention - 1.0))
                    .round()
                    .min(max_ivl)
                    .max(1.0);
            });

        let old_due = card_table.slice(s![Column::Due, ..]);
        let mut new_due = old_due.to_owned();
        izip!(&mut new_due, &new_interval, &true_review, &true_learn)
            .filter(|(.., &true_review_flag, &true_learn_flag)| true_review_flag || true_learn_flag)
            .for_each(|(new_due, &new_ivl, ..)| {
                *new_due = today as f64 + new_ivl;
            });

        // Update the card_table with the new values
        card_table
            .slice_mut(s![Column::Difficulty, ..])
            .assign(&new_difficulty);
        card_table
            .slice_mut(s![Column::Stability, ..])
            .assign(&new_stability);
        card_table
            .slice_mut(s![Column::LastDate, ..])
            .assign(&new_last_date);
        card_table.slice_mut(s![Column::Due, ..]).assign(&new_due);
        card_table
            .slice_mut(s![Column::Interval, ..])
            .assign(&new_interval);

        // Update the review_cnt_per_day, learn_cnt_per_day and memorized_cnt_per_day
        // review_cnt_per_day[today] = true_review.iter().filter(|&&x| x).count() as f64;
        // learn_cnt_per_day[today] = true_learn.iter().filter(|&&x| x).count() as f64;
        memorized_cnt_per_day[today] = retrievability.sum();
    }

    memorized_cnt_per_day[memorized_cnt_per_day.len() - 1]
}

impl<B: Backend> FSRS<B> {
    /// For the given simulator parameters and weights, determine the suggested `desired_retention`
    /// value.
    pub fn optimal_retention<F>(
        &self,
        config: &SimulatorConfig,
        weights: &Weights,
        mut progress: F,
    ) -> Result<f64>
    where
        F: FnMut(ItemProgress) -> bool + Send,
    {
        let weights = if weights.is_empty() {
            DEFAULT_WEIGHTS
        } else if weights.len() != 17 {
            return Err(FSRSError::InvalidWeights);
        } else {
            weights
        }
        .iter()
        .map(|v| *v as f64)
        .collect_vec();
        let mut low = 0.75;
        let mut high = 0.95;
        let mut optimal_retention = 0.85;
        let epsilon = 0.01;
        let mut iter = 0;

        let mut progress_info = ItemProgress {
            current: 0,
            total: 100,
        };
        let inc_progress = Arc::new(Mutex::new(move || {
            progress_info.current += 1;
            progress(progress_info)
        }));

        while high - low > epsilon && iter < 10 {
            iter += 1;
            let mid1 = low + (high - low) / 3.0;
            let mid2 = high - (high - low) / 3.0;

            let sample_several = |n: usize, mid: f64| -> Result<f64, FSRSError> {
                let out: Vec<f64> = (0..n)
                    .into_par_iter()
                    .map(|i| {
                        let result =
                            simulate(config, &weights, mid, Some((i + 42).try_into().unwrap()));
                        if !(inc_progress.lock().unwrap()()) {
                            return Err(FSRSError::Interrupted);
                        }
                        Ok(result)
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(out.iter().sum::<f64>() / n as f64)
            };

            let mut memorization1 = None;
            let mut memorization2 = None;
            rayon::scope(|s| {
                s.spawn(|_| {
                    memorization1 = Some(sample_several(5, mid1));
                });
                s.spawn(|_| {
                    memorization2 = Some(sample_several(5, mid2));
                });
            });
            if memorization1.unwrap()? > memorization2.unwrap()? {
                high = mid2;
            } else {
                low = mid1;
            }

            optimal_retention = (high + low) / 2.0;
        }
        Ok(optimal_retention)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::DEFAULT_WEIGHTS;

    #[test]
    fn simulator() {
        let config = SimulatorConfig::default();
        let memorization = simulate(
            &config,
            &DEFAULT_WEIGHTS.iter().map(|v| *v as f64).collect_vec(),
            0.9,
            None,
        );
        assert_eq!(memorization, 2542.50223082592)
    }

    #[test]
    fn optimal_retention() -> Result<()> {
        let config = SimulatorConfig::default();
        let fsrs = FSRS::new(None)?;
        let optimal_retention = fsrs.optimal_retention(&config, &[], |_v| true).unwrap();
        assert_eq!(optimal_retention, 0.8687319006249048);
        assert!(fsrs.optimal_retention(&config, &[1.], |_v| true).is_err());
        Ok(())
    }
}

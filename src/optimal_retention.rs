use burn::config::Config;
use ndarray::{azip, s, Array1, Array2, Ix0, Ix1, SliceInfoElem, Zip};
use ndarray_rand::rand_distr::Distribution;
use ndarray_rand::RandomExt;
use rand::{
    distributions::{Uniform, WeightedIndex},
    rngs::StdRng,
    SeedableRng,
};
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

#[derive(Config, Debug)]
pub struct SimulatorConfig {
    w: [f64; 17],
    #[config(default = 10000)]
    deck_size: usize,
    #[config(default = 365)]
    learn_span: usize,
    #[config(default = 1800.0)]
    max_cost_perday: f64,
    #[config(default = 36500.0)]
    max_ivl: f64,
    #[config(default = 10.0)]
    recall_cost: f64,
    #[config(default = 50.0)]
    forget_cost: f64,
    #[config(default = 20.0)]
    learn_cost: f64,
}

fn stability_after_success(w: [f64; 17], s: f64, r: f64, d: f64, response: usize) -> f64 {
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

fn stability_after_failure(w: [f64; 17], s: f64, r: f64, d: f64) -> f64 {
    s.min(w[11] * d.powf(-w[12]) * ((s + 1.0).powf(w[13]) - 1.0) * f64::exp((1.0 - r) * w[14]))
        .max(0.1)
}

fn simulate(config: &SimulatorConfig, request_retention: f64, seed: Option<u64>) -> f64 {
    let SimulatorConfig {
        w,
        deck_size,
        learn_span,
        max_cost_perday,
        max_ivl,
        recall_cost,
        forget_cost,
        learn_cost,
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
    let first_rating_prob = [0.15, 0.2, 0.6, 0.05];
    let first_rating_dist = WeightedIndex::new(first_rating_prob).unwrap();

    let review_rating_choices = [1, 2, 3];
    let review_rating_prob = [0.3, 0.6, 0.1];
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
        Zip::from(&mut delta_t)
            .and(&old_last_date)
            .and(&has_learned)
            .for_each(|delta_t, &last_date, &has_learned_flag| {
                if has_learned_flag {
                    *delta_t = today as f64 - last_date;
                }
            });

        let mut retrievability = Array1::zeros(deck_size); // Create an array for retrievability

        // Calculate retrievability for entries where has_learned is true
        azip!((
            retrievability    in &mut retrievability,
            &delta_t          in &delta_t,
            &stability        in &old_stability,
            &has_learned_flag in &has_learned)
            if has_learned_flag {
                *retrievability = f64::powf(1.0 + delta_t / (9.0 * stability), -1.0);
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

        // Update 'cost' column based on 'need_review' and 'forget'
        Zip::from(&mut cost)
            .and(&need_review)
            .and(&forget)
            .for_each(|cost, &need_review_flag, &forget_flag| {
                if need_review_flag {
                    *cost = if forget_flag {
                        forget_cost
                    } else {
                        recall_cost
                    }
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
        Zip::from(&mut cost)
            .and(&need_learn)
            .for_each(|cost, &need_learn_flag| {
                if need_learn_flag {
                    *cost = learn_cost;
                }
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

        let mut ratings = Array1::zeros(deck_size);
        Zip::from(&mut ratings)
            .and(&true_review)
            .and(&true_learn)
            .for_each(|rating, &true_review_flag, &true_learn_flag| {
                *rating = if true_learn_flag {
                    first_rating_choices[first_rating_dist.sample(&mut rng)]
                } else if true_review_flag {
                    review_rating_choices[review_rating_dist.sample(&mut rng)]
                } else {
                    *rating
                }
            });

        let mut new_stability = old_stability.to_owned();
        let old_difficulty = card_table.slice(s![Column::Difficulty, ..]);
        // Iterate over slices and apply stability_after_failure function
        azip!((
            new_stab in &mut new_stability,
            &stab    in &old_stability,
            &retr    in &retrievability,
            &diff    in &old_difficulty,
            &condition in &(&true_review & &forget))
            if condition {
                *new_stab = stability_after_failure(w, stab, retr, diff);
            }
        );

        // Iterate over slices and apply stability_after_success function
        azip!((
            new_stab   in &mut new_stability,
            &rating    in &ratings,
            &stab      in &old_stability,
            &retr      in &retrievability,
            &diff      in &old_difficulty,
            &condition in &(&true_review & !&forget))
            if condition {
                *new_stab = stability_after_success(w, stab, retr, diff, rating);
            }
        );

        // Initialize a new Array1 to store updated difficulty values
        let mut new_difficulty = old_difficulty.to_owned();

        // Update the difficulty values based on the condition 'true_review & forget'
        azip!((
            new_diff  in &mut new_difficulty,
            &old_diff in &old_difficulty,
            &true_rev in &true_review,
            &frgt     in &forget)
        if true_rev && frgt {
            *new_diff = (old_diff + 2.0 * w[6]).max(1.0).min(10.0);
        });

        // Update 'last_date' column where 'true_review' or 'true_learn' is true
        let mut new_last_date = old_last_date.to_owned();
        Zip::from(&mut new_last_date)
            .and(&true_review)
            .and(&true_learn)
            .for_each(|new_last_date, &true_review_flag, &true_learn_flag| {
                if true_review_flag || true_learn_flag {
                    *new_last_date = today as f64;
                }
            });

        Zip::from(&mut new_stability)
            .and(&ratings)
            .and(&true_learn)
            .for_each(|new_stab, &rating, &true_learn_flag| {
                if true_learn_flag {
                    *new_stab = w[rating];
                }
            });
        Zip::from(&mut new_difficulty)
            .and(&ratings)
            .and(&true_learn)
            .for_each(|new_diff, &rating, &true_learn_flag| {
                if true_learn_flag {
                    *new_diff = w[4] - w[5] * (rating as f64 - 3.0);
                }
            });
        let old_interval = card_table.slice(s![Column::Interval, ..]);
        let mut new_interval = old_interval.to_owned();
        azip!((
            new_ivl           in &mut new_interval,
            &new_stab         in &new_stability,
            &true_review_flag in &true_review,
            &true_learn_flag  in &true_learn)
            if true_review_flag || true_learn_flag {
                *new_ivl = (9.0 * new_stab * (1.0 / request_retention - 1.0))
                    .round()
                    .min(max_ivl)
                    .max(1.0);
        });

        let old_due = card_table.slice(s![Column::Due, ..]);
        let mut new_due = old_due.to_owned();
        azip!((
            new_due           in &mut new_due,
            &new_ivl          in &new_interval,
            &true_review_flag in &true_review,
            &true_learn_flag  in &true_learn)
            if true_review_flag || true_learn_flag {
                *new_due = today as f64 + new_ivl;
            }
        );

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

pub fn find_optimal_retention(config: &SimulatorConfig) -> f64 {
    let mut low = 0.75;
    let mut high = 0.95;
    let mut optimal_retention = 0.85;
    let epsilon = 0.01;
    let mut iter = 0;
    while high - low > epsilon && iter < 10 {
        iter += 1;
        let mid1 = low + (high - low) / 3.0;
        let mid2 = high - (high - low) / 3.0;
        fn sample_several(n: usize, config: &SimulatorConfig, mid: f64) -> f64 {
            (0..n)
                .map(|i| simulate(config, mid, Some((i + 42).try_into().unwrap())))
                .sum::<f64>()
                / n as f64
        }
        let memorization1 = sample_several(3, config, mid1);
        let memorization2 = sample_several(3, config, mid2);

        if memorization1 > memorization2 {
            high = mid2;
        } else {
            low = mid1;
        }

        optimal_retention = (high + low) / 2.0;
        dbg!(iter, optimal_retention);
    }
    optimal_retention
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simulator() {
        let config = SimulatorConfig::new([
            0.4, 0.6, 2.4, 5.8, 4.93, 0.94, 0.86, 0.01, 1.49, 0.14, 0.94, 2.18, 0.05, 0.34, 1.26,
            0.29, 2.61,
        ]);
        let memorization = simulate(&config, 0.9, None);
        assert_eq!(memorization, 3832.250006134299)
    }

    #[test]
    fn test_find_optimal_retention() {
        let config = SimulatorConfig::new([
            0.4, 0.6, 2.4, 5.8, 4.93, 0.94, 0.86, 0.01, 1.49, 0.14, 0.94, 2.18, 0.05, 0.34, 1.26,
            0.29, 2.61,
        ]);
        let optimal_retention = find_optimal_retention(&config);
        assert_eq!(optimal_retention, 0.8179164761469289)
    }
}

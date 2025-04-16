use crate::DEFAULT_PARAMETERS;
use crate::FSRSItem;
use crate::error::{FSRSError, Result};
use crate::inference::S_MIN;
use argmin::core::State;
use argmin::core::{CostFunction, Error, Executor};
use argmin::solver::neldermead::NelderMead;
use ndarray::Array1;
use std::collections::HashMap;

static R_S0_DEFAULT_ARRAY: &[(u32, f32); 4] = &[
    (1, DEFAULT_PARAMETERS[0]),
    (2, DEFAULT_PARAMETERS[1]),
    (3, DEFAULT_PARAMETERS[2]),
    (4, DEFAULT_PARAMETERS[3]),
];

pub fn pretrain(
    fsrs_items: Vec<FSRSItem>,
    average_recall: f32,
) -> Result<([f32; 4], f32, HashMap<u32, u32>)> {
    let pretrainset = create_pretrain_data(fsrs_items);
    let rating_count = total_rating_count(&pretrainset);
    let rating_stability_decay = search_parameters(pretrainset, average_recall);
    let mut stability_map = HashMap::new();
    let mut decay_map = HashMap::new();
    for (rating, (stability, decay)) in rating_stability_decay {
        stability_map.insert(rating, stability);
        decay_map.insert(rating, decay);
    }
    dbg!(&stability_map);
    dbg!(&decay_map);
    // Calculate weighted average of decay values based on rating_count
    let mut weighted_sum = 0.0;
    let mut total_count = 0;

    for (rating, count) in &rating_count {
        if let Some(decay) = decay_map.get(rating) {
            weighted_sum += decay * (*count as f32);
            total_count += *count;
        }
    }

    let weighted_avg_decay = if total_count > 0 {
        weighted_sum / (total_count as f32)
    } else {
        DEFAULT_PARAMETERS[20] // Use default decay if no ratings
    };

    Ok((
        smooth_and_fill(&mut stability_map, &rating_count)?,
        weighted_avg_decay,
        rating_count,
    ))
}

type FirstRating = u32;
type Count = u32;

fn create_pretrain_data(fsrs_items: Vec<FSRSItem>) -> HashMap<FirstRating, Vec<AverageRecall>> {
    // filter FSRSItem instances with exactly 1 long term review.
    let items: Vec<_> = fsrs_items
        .into_iter()
        .filter(|item| item.long_term_review_cnt() == 1)
        .collect();

    // use a nested HashMap (groups) to group items first by the rating in the first FSRSReview
    // and then by the delta_t in the second FSRSReview.
    // (first_rating -> first_long_term_delta_t -> vec![0/1 for fail/pass])
    let mut groups = HashMap::new();

    for item in items {
        let first_rating = item.reviews[0].rating;
        let first_long_term_review = item.first_long_term_review();
        let first_long_term_delta_t = first_long_term_review.delta_t;
        let first_long_term_label = (first_long_term_review.rating > 1) as i32;

        let inner_map = groups.entry(first_rating).or_insert_with(HashMap::new);
        let ratings = inner_map
            .entry(first_long_term_delta_t)
            .or_insert_with(Vec::new);
        ratings.push(first_long_term_label);
    }

    let mut results = HashMap::new();

    for (first_rating, inner_map) in &groups {
        let mut data = vec![];

        // calculate the average (recall) and count for each group.
        for (second_delta_t, ratings) in inner_map {
            let avg = ratings.iter().map(|&x| x as f64).sum::<f64>() / ratings.len() as f64;
            data.push(AverageRecall {
                delta_t: *second_delta_t as f64,
                recall: avg,
                count: ratings.len() as f64,
            })
        }

        // Sort by delta_t in ascending order
        data.sort_unstable_by(|a, b| a.delta_t.total_cmp(&b.delta_t));

        results.insert(*first_rating, data);
    }
    results
}

/// The average pass rate & count for a single delta_t for a given first rating.
struct AverageRecall {
    delta_t: f64,
    recall: f64,
    count: f64,
}

fn total_rating_count(
    pretrainset: &HashMap<FirstRating, Vec<AverageRecall>>,
) -> HashMap<FirstRating, Count> {
    pretrainset
        .iter()
        .map(|(first_rating, data)| {
            let count = data.iter().map(|d| d.count).sum::<f64>() as u32;
            (*first_rating, count)
        })
        .collect()
}

fn power_forgetting_curve(t: &Array1<f64>, s: f64, decay: f64) -> Array1<f64> {
    let factor = 0.9f64.powf(1.0 / decay) - 1.0;
    (t / s * factor + 1.0).mapv(|v| v.powf(decay))
}

pub(crate) const INIT_S_MAX: f32 = 100.0;

struct OptimizationProblem {
    delta_t: Array1<f64>,
    recall: Array1<f64>,
    count: Array1<f64>,
    default_s0: f64,
    default_decay: f64,
}

impl CostFunction for OptimizationProblem {
    type Param = Vec<f64>;
    type Output = f64;

    fn cost(&self, param: &Self::Param) -> Result<Self::Output, Error> {
        let s = param[0];
        let decay = param[1];
        let y_pred = power_forgetting_curve(&self.delta_t, s, -decay);
        let logloss = (-(self.recall.clone() * y_pred.clone().mapv_into(|v| v.ln())
            + (1.0 - self.recall.clone()) * (1.0 - y_pred).mapv_into(|v| v.ln()))
            * self.count.clone())
        .sum();
        let l1 = ((s - self.default_s0).abs() + (decay - self.default_decay).abs()) / 16.0;
        let mut total = logloss + l1;
        if decay < 0.1 || decay > 0.8 || s < S_MIN.into() || s > INIT_S_MAX.into() {
            total *= 1000.0;
        }
        Ok(total)
    }
}

fn search_parameters(
    mut pretrainset: HashMap<FirstRating, Vec<AverageRecall>>,
    average_recall: f32,
) -> HashMap<u32, (f32, f32)> {
    let mut optimal_params = HashMap::new();
    let default_decay = 0.2; // 默认 decay 值

    for (first_rating, data) in &mut pretrainset {
        let r_s0_default: HashMap<u32, f32> = R_S0_DEFAULT_ARRAY.iter().cloned().collect();
        let default_s0 = r_s0_default[first_rating] as f64;
        let delta_t = Array1::from_iter(data.iter().map(|d| d.delta_t));
        let count = Array1::from_iter(data.iter().map(|d| d.count));
        let recall = {
            let real_recall = Array1::from_iter(data.iter().map(|d| d.recall));
            (real_recall * count.clone() + average_recall as f64) / (count.clone() + 1.0)
        };
        dbg!(&delta_t);
        dbg!(&recall);
        dbg!(&count);

        let problem = OptimizationProblem {
            delta_t,
            recall,
            count,
            default_s0,
            default_decay,
        };


        let solver = NelderMead::new(vec![
            vec![default_s0 * 1.05, default_decay],
            vec![default_s0, default_decay * 1.05],
            vec![default_s0, default_decay],
        ]);

        let res = Executor::new(problem, solver)
            .configure(|state| state.max_iters(1000))
            .run()
            .unwrap();

        let best_params = res.state().get_best_param().unwrap();
        optimal_params.insert(
            *first_rating,
            (best_params[0] as f32, best_params[1] as f32),
        );
    }

    optimal_params
}

pub(crate) fn smooth_and_fill(
    rating_stability: &mut HashMap<u32, f32>,
    rating_count: &HashMap<u32, u32>,
) -> Result<[f32; 4]> {
    rating_stability.retain(|&key, _| rating_count.contains_key(&key));
    for (small_rating, big_rating) in [(1, 2), (2, 3), (3, 4), (1, 3), (2, 4), (1, 4)] {
        if let (Some(&small_value), Some(&big_value)) = (
            rating_stability.get(&small_rating),
            rating_stability.get(&big_rating),
        ) {
            if small_value > big_value {
                if rating_count[&small_rating] > rating_count[&big_rating] {
                    rating_stability.insert(big_rating, small_value);
                } else {
                    rating_stability.insert(small_rating, big_value);
                }
            }
        }
    }

    let w1 = 0.41;
    let w2 = 0.54;

    let mut init_s0 = vec![];

    let r_s0_default = R_S0_DEFAULT_ARRAY
        .iter()
        .cloned()
        .collect::<HashMap<_, _>>();
    let mut rating_stability_arr = [
        None,
        rating_stability.get(&1).cloned(),
        rating_stability.get(&2).cloned(),
        rating_stability.get(&3).cloned(),
        rating_stability.get(&4).cloned(),
    ];
    match rating_stability.len() {
        0 => return Err(FSRSError::NotEnoughData),
        1 => {
            let rating = rating_stability.keys().next().unwrap();
            let factor = rating_stability[rating] / r_s0_default[rating];
            init_s0 = r_s0_default.values().map(|&x| x * factor).collect();
            init_s0.sort_by(|a, b| a.partial_cmp(b).unwrap());
        }
        2 => {
            match rating_stability_arr {
                [_, None, None, Some(r3), Some(r4)] => {
                    let r2 = r3.powf(1.0 / (1.0 - w2)) * r4.powf(1.0 - 1.0 / (1.0 - w2));
                    rating_stability_arr[2] = Some(r2);
                    rating_stability_arr[1] = Some(r2.powf(1.0 / w1) * r3.powf(1.0 - 1.0 / w1));
                }
                [_, None, Some(r2), None, Some(r4)] => {
                    let r3 = r2.powf(1.0 - w2) * r4.powf(w2);
                    rating_stability_arr[3] = Some(r3);
                    rating_stability_arr[1] = Some(r2.powf(1.0 / w1) * r3.powf(1.0 - 1.0 / w1));
                }
                [_, None, Some(r2), Some(r3), None] => {
                    rating_stability_arr[4] = Some(r2.powf(1.0 - 1.0 / w2) * r3.powf(1.0 / w2));
                    rating_stability_arr[1] = Some(r2.powf(1.0 / w1) * r3.powf(1.0 - 1.0 / w1));
                }
                [_, Some(r1), None, None, Some(r4)] => {
                    let r2 = r1.powf(w1 / (w1.mul_add(-w2, w1 + w2)))
                        * r4.powf(1.0 - w1 / (w1.mul_add(-w2, w1 + w2)));
                    rating_stability_arr[2] = Some(r2);
                    rating_stability_arr[3] = Some(
                        r1.powf(1.0 - w2 / (w1.mul_add(-w2, w1 + w2)))
                            * r4.powf(w2 / (w1.mul_add(-w2, w1 + w2))),
                    );
                }
                [_, Some(r1), None, Some(r3), None] => {
                    let r2 = r1.powf(w1) * r3.powf(1.0 - w1);
                    rating_stability_arr[2] = Some(r2);
                    rating_stability_arr[4] = Some(r2.powf(1.0 - 1.0 / w2) * r3.powf(1.0 / w2));
                }
                [_, Some(r1), Some(r2), None, None] => {
                    let r3 = r1.powf(1.0 - 1.0 / (1.0 - w1)) * r2.powf(1.0 / (1.0 - w1));
                    rating_stability_arr[3] = Some(r3);
                    rating_stability_arr[4] = Some(r2.powf(1.0 - 1.0 / w2) * r3.powf(1.0 / w2));
                }
                _ => {}
            }
            init_s0 = rating_stability_arr.into_iter().flatten().collect();
        }
        3 => {
            match rating_stability_arr {
                [_, None, Some(r2), Some(r3), _] => {
                    rating_stability_arr[1] = Some(r2.powf(1.0 / w1) * r3.powf(1.0 - 1.0 / w1));
                }
                [_, Some(r1), None, Some(r3), _] => {
                    rating_stability_arr[2] = Some(r1.powf(w1) * r3.powf(1.0 - w1));
                }
                [_, _, Some(r2), None, Some(r4)] => {
                    rating_stability_arr[3] = Some(r2.powf(1.0 - w2) * r4.powf(w2));
                }
                [_, _, Some(r2), Some(r3), None] => {
                    rating_stability_arr[4] = Some(r2.powf(1.0 - 1.0 / w2) * r3.powf(1.0 / w2));
                }
                _ => {}
            }
            init_s0 = rating_stability_arr.into_iter().flatten().collect();
        }
        4 => {
            init_s0 = rating_stability_arr.into_iter().flatten().collect();
        }
        _ => {}
    }
    init_s0 = init_s0
        .iter()
        .map(|&v| v.clamp(S_MIN, INIT_S_MAX))
        .collect();
    Ok(init_s0[0..=3].try_into().unwrap())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::filter_outlier;
    use crate::test_helpers::TestHelper;
    use crate::training::calculate_average_recall;

    #[test]
    fn test_power_forgetting_curve() {
        let t = Array1::from(vec![0.0, 1.0, 2.0, 3.0]);
        let s = 1.0;
        let y = power_forgetting_curve(&t, s, -0.2);
        let expected = Array1::from(vec![1.0, 0.9, 0.8402893846730101, 0.7985001730858255]);
        assert_eq!(y, expected);
    }

    #[test]
    fn test_loss() {
        let delta_t = Array1::from(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let recall = Array1::from(vec![
            0.86684181, 0.90758192, 0.73348482, 0.76776996, 0.68769064,
        ]);
        let count = Array1::from(vec![435.0, 97.0, 63.0, 38.0, 28.0]);
        let default_s0 = DEFAULT_PARAMETERS[0] as f64;
        let default_decay = DEFAULT_PARAMETERS[20] as f64;
        let problem = OptimizationProblem {
            delta_t,
            recall,
            count,
            default_s0,
            default_decay,
        };
        let actual = problem.cost(&vec![0.7840586, 0.2]).unwrap();
        assert_eq!(actual, 279.08537444877663);
        let actual = problem.cost(&vec![0.7840590622451964, 0.2]).unwrap();
        assert_eq!(actual, 279.08537444878664);
    }

    #[test]
    fn test_search_parameters() {
        let first_rating = 1;
        let pretrainset = HashMap::from([(
            first_rating,
            vec![
                AverageRecall {
                    delta_t: 1.0,
                    recall: 0.86666667,
                    count: 435.0,
                },
                AverageRecall {
                    delta_t: 2.0,
                    recall: 0.90721649,
                    count: 97.0,
                },
                AverageRecall {
                    delta_t: 3.0,
                    recall: 0.73015873,
                    count: 63.0,
                },
                AverageRecall {
                    delta_t: 4.0,
                    recall: 0.76315789,
                    count: 38.0,
                },
                AverageRecall {
                    delta_t: 5.0,
                    recall: 0.67857143,
                    count: 28.0,
                },
            ],
        )]);
        let actual = search_parameters(pretrainset, 0.943_028_57);
        let (stability, decay) = actual.get(&first_rating).unwrap();
        [*stability, *decay].assert_approx_eq([0.6919513940811157, 0.12588396668434143]);
    }

    #[test]
    fn test_pretrain() {
        use crate::convertor_tests::anki21_sample_file_converted_to_fsrs;
        let items = anki21_sample_file_converted_to_fsrs();
        let (mut pretrainset, mut trainset) = items
            .into_iter()
            .partition(|item| item.long_term_review_cnt() == 1);
        (pretrainset, trainset) = filter_outlier(pretrainset, trainset);
        let items = [pretrainset.clone(), trainset].concat();
        let average_recall = calculate_average_recall(&items);

        let (stability, decay, rating_count) = pretrain(pretrainset, average_recall).unwrap();
        stability.assert_approx_eq([
            0.6919512748718262,
            2.171159505844116,
            4.806155681610107,
            13.29844856262207,
        ]);
        assert_eq!(decay, 0.10524022);
        assert_eq!(rating_count, HashMap::from([(1, 661), (3, 461), (4, 2143)]));
    }

    #[test]
    fn test_smooth_and_fill() {
        let mut rating_stability = HashMap::from([(1, 0.4), (3, 2.3), (4, 10.9)]);
        let rating_count = HashMap::from([(1, 1), (2, 1), (3, 1), (4, 1)]);
        let actual = smooth_and_fill(&mut rating_stability, &rating_count).unwrap();
        assert_eq!(actual, [0.4, 1.1227008, 2.3, 10.9,]);

        let mut rating_stability = HashMap::from([(2, 0.35)]);
        let rating_count = HashMap::from([(2, 1)]);
        let actual = smooth_and_fill(&mut rating_stability, &rating_count).unwrap();
        assert_eq!(actual, [0.06458245, 0.35, 0.9693909, 4.802264]);
    }
}

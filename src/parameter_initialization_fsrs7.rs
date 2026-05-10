use crate::DEFAULT_PARAMETERS;
use crate::FSRSItem;
use crate::dataset::bucket_long_term_delta_t;
use crate::error::{FSRSError, Result};
use crate::simulation::S_MIN;
use ndarray::Array1;
use std::collections::HashMap;

static R_S0_DEFAULT_ARRAY: &[(u32, f32); 4] = &[
    (1, DEFAULT_PARAMETERS[0]),
    (2, DEFAULT_PARAMETERS[1]),
    (3, DEFAULT_PARAMETERS[2]),
    (4, DEFAULT_PARAMETERS[3]),
];

type FirstRating = u32;
type AverageRecall = (f64, f64, f64);

pub(crate) type InitializationResult = ([f32; 4], [f32; 8], HashMap<u32, u32>);
const INIT_S_MAX: f32 = 100.0;

const FSRS7_FORGETTING_CURVE_CANDIDATES: [[f32; 8]; 16] = [
    [0.0723, 0.1634, 0.5, 0.9555, 0.2245, 0.6232, 0.1362, 0.3862],
    [
        0.0594, 0.3358, 0.598, 0.9517, 0.3122, 0.5685, 0.2371, 0.4871,
    ],
    [
        0.0441, 0.2533, 0.6823, 0.9598, 0.3613, 0.5202, 0.2283, 0.4783,
    ],
    [
        0.0621, 0.2475, 0.6496, 0.9744, 0.313, 0.5662, 0.2336, 0.4836,
    ],
    [
        0.0462, 0.2962, 0.6938, 0.9592, 0.3341, 0.5273, 0.2185, 0.4685,
    ],
    [
        0.0422, 0.2813, 0.6713, 0.9421, 0.2935, 0.5985, 0.2183, 0.4683,
    ],
    [
        0.0568, 0.1563, 0.6567, 0.9633, 0.3682, 0.5041, 0.1952, 0.4452,
    ],
    [
        0.0651, 0.2502, 0.6682, 0.9472, 0.3757, 0.4933, 0.2408, 0.4908,
    ],
    [
        0.0548, 0.1655, 0.6138, 0.9654, 0.3251, 0.5717, 0.1418, 0.3918,
    ],
    [
        0.0381, 0.2803, 0.7202, 0.9491, 0.3362, 0.5166, 0.2248, 0.4748,
    ],
    [
        0.0422, 0.1935, 0.694, 0.9549, 0.3871, 0.4704, 0.2413, 0.4913,
    ],
    [0.0651, 0.1916, 0.623, 0.972, 0.3528, 0.5484, 0.2373, 0.4873],
    [
        0.0508, 0.3743, 0.5863, 0.9448, 0.2974, 0.606, 0.1444, 0.3944,
    ],
    [
        0.0498, 0.3753, 0.6875, 0.9319, 0.3758, 0.4984, 0.2268, 0.4768,
    ],
    [
        0.0618, 0.1663, 0.5977, 0.9682, 0.3619, 0.5066, 0.2972, 0.5472,
    ],
    [
        0.0656, 0.197, 0.5693, 0.9692, 0.3599, 0.5374, 0.2596, 0.5096,
    ],
];

fn default_forgetting_curve_params() -> [f32; 8] {
    [
        DEFAULT_PARAMETERS[27],
        DEFAULT_PARAMETERS[28],
        DEFAULT_PARAMETERS[29],
        DEFAULT_PARAMETERS[30],
        DEFAULT_PARAMETERS[31],
        DEFAULT_PARAMETERS[32],
        DEFAULT_PARAMETERS[33],
        DEFAULT_PARAMETERS[34],
    ]
}

fn prepare_dataset_for_initialization(
    fsrs_items: Vec<FSRSItem>,
) -> HashMap<FirstRating, Vec<AverageRecall>> {
    let to_key = |delta_t: f32| bucket_long_term_delta_t(delta_t).to_bits();
    let from_key = |key: u32| f32::from_bits(key);
    let items: Vec<_> = fsrs_items
        .into_iter()
        .filter(|item| item.long_term_review_cnt() == 1)
        .collect();

    let mut groups = HashMap::new();

    for item in items {
        let first_rating = item.reviews[0].rating;
        let first_long_term_review = item.first_long_term_review();
        let first_long_term_delta_t = to_key(first_long_term_review.delta_t);
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
        for (second_delta_t, ratings) in inner_map {
            let avg = ratings.iter().map(|&x| x as f64).sum::<f64>() / ratings.len() as f64;
            data.push((from_key(*second_delta_t) as f64, avg, ratings.len() as f64));
        }
        data.sort_unstable_by(|a, b| a.0.total_cmp(&b.0));
        results.insert(*first_rating, data);
    }
    results
}

fn forgetting_curve_with_params(t: &Array1<f64>, s: f64, params: &[f32; 8]) -> Array1<f64> {
    let decay1 = -(params[0] as f64);
    let decay2 = -(params[1] as f64);
    let base1 = params[2] as f64;
    let base2 = params[3] as f64;
    let base_weight1 = params[4] as f64;
    let base_weight2 = params[5] as f64;
    let swp1 = params[6] as f64;
    let swp2 = params[7] as f64;

    let t_over_s = t / s;
    let factor1 = base1.powf(1.0 / decay1) - 1.0;
    let factor2 = base2.powf(1.0 / decay2) - 1.0;
    let r1 = (t_over_s.clone() * factor1 + 1.0).mapv(|v| v.powf(decay1));
    let r2 = (t_over_s * factor2 + 1.0).mapv(|v| v.powf(decay2));

    let weight1 = base_weight1 * s.powf(-swp1);
    let weight2 = base_weight2 * s.powf(swp2);

    (r1 * weight1 + r2 * weight2) / (weight1 + weight2)
}

fn loss_with_curve(
    delta_t: &Array1<f64>,
    recall: &Array1<f64>,
    count: &Array1<f64>,
    init_s0: f64,
    default_s0: f64,
    params: &[f32; 8],
) -> f64 {
    let y_pred = forgetting_curve_with_params(delta_t, init_s0, params);
    let y_pred = y_pred.mapv(|v| v.clamp(0.0001, 0.9999));
    let logloss = (-(recall * y_pred.clone().mapv_into(|v| v.ln())
        + (1.0 - recall) * (1.0 - &y_pred).mapv_into(|v| v.ln()))
        * count)
        .sum();
    let l1 = (init_s0 - default_s0).abs() / 16.0;
    logloss + l1
}

fn search_parameters_for_curve(
    mut dataset_for_initialization: HashMap<FirstRating, Vec<AverageRecall>>,
    average_recall: f32,
    curve_params: &[f32; 8],
) -> (HashMap<u32, f32>, HashMap<u32, u32>, f64) {
    let mut optimal_stabilities = HashMap::new();
    let mut rating_count = HashMap::new();
    let mut total_loss = 0.0;
    let epsilon = f64::EPSILON;

    for (first_rating, data) in &mut dataset_for_initialization {
        let r_s0_default: HashMap<u32, f32> = R_S0_DEFAULT_ARRAY.iter().cloned().collect();
        let default_s0 = r_s0_default[first_rating] as f64;
        let delta_t = Array1::from_iter(data.iter().map(|d| d.0));
        let count = Array1::from_iter(data.iter().map(|d| d.2));
        let recall = {
            let real_recall = Array1::from_iter(data.iter().map(|d| d.1));
            (real_recall * count.clone() + average_recall as f64) / (count.clone() + 1.0)
        };
        let mut low = S_MIN as f64;
        let mut high = INIT_S_MAX as f64;
        let mut optimal_s = default_s0;

        let mut iter = 0;
        while high - low > epsilon && iter < 1000 {
            iter += 1;
            let mid1 = low + (high - low) / 3.0;
            let mid2 = high - (high - low) / 3.0;

            let loss1 = loss_with_curve(&delta_t, &recall, &count, mid1, default_s0, curve_params);
            let loss2 = loss_with_curve(&delta_t, &recall, &count, mid2, default_s0, curve_params);

            if loss1 < loss2 {
                high = mid2;
            } else {
                low = mid1;
            }

            optimal_s = (high + low) / 2.0;
        }

        let best_loss = loss_with_curve(
            &delta_t,
            &recall,
            &count,
            optimal_s,
            default_s0,
            curve_params,
        );
        total_loss += best_loss;
        optimal_stabilities.insert(*first_rating, optimal_s as f32);
        rating_count.insert(*first_rating, count.sum() as u32);
    }

    for (small_rating, big_rating) in [(1, 2), (2, 3), (3, 4), (1, 3), (2, 4), (1, 4)] {
        if let (Some(&small_value), Some(&big_value)) = (
            optimal_stabilities.get(&small_rating),
            optimal_stabilities.get(&big_rating),
        ) {
            if small_value > big_value {
                let small_count = rating_count.get(&small_rating).copied().unwrap_or(0);
                let big_count = rating_count.get(&big_rating).copied().unwrap_or(0);
                if small_count > big_count {
                    optimal_stabilities.insert(big_rating, small_value);
                } else {
                    optimal_stabilities.insert(small_rating, big_value);
                }
            }
        }
    }

    (optimal_stabilities, rating_count, total_loss)
}

fn fill_initial_stabilities_fsrs7(rating_stability: &HashMap<u32, f32>) -> Result<[f32; 4]> {
    if rating_stability.is_empty() {
        return Err(FSRSError::NotEnoughData);
    }

    let default_s0 = R_S0_DEFAULT_ARRAY
        .iter()
        .cloned()
        .collect::<HashMap<_, _>>();

    if rating_stability.len() == 1 {
        let rating = rating_stability.keys().next().copied().unwrap();
        let factor = rating_stability[&rating] / default_s0[&rating];
        let mut values = [
            default_s0[&1] * factor,
            default_s0[&2] * factor,
            default_s0[&3] * factor,
            default_s0[&4] * factor,
        ];
        values.sort_by(|a, b| a.partial_cmp(b).unwrap());
        return Ok(values.map(|v| v.clamp(S_MIN, INIT_S_MAX)));
    }

    let anchors: HashMap<u32, f64> = HashMap::from([(1, -8.09), (2, -3.83), (3, -2.5), (4, -1.0)]);
    let mut log_s0: HashMap<u32, f64> = rating_stability
        .iter()
        .map(|(k, v)| (*k, (*v as f64).ln()))
        .collect();

    for target in 1..=4 {
        if log_s0.contains_key(&target) {
            continue;
        }
        let lower = (1..target).rev().find(|r| log_s0.contains_key(r));
        let upper = ((target + 1)..=4).find(|r| log_s0.contains_key(r));

        let value = match (lower, upper) {
            (Some(lo), Some(hi)) => {
                let t = (anchors[&target] - anchors[&lo]) / (anchors[&hi] - anchors[&lo]);
                log_s0[&lo] + t * (log_s0[&hi] - log_s0[&lo])
            }
            (Some(lo), None) => log_s0[&lo] + (anchors[&target] - anchors[&lo]),
            (None, Some(hi)) => log_s0[&hi] + (anchors[&target] - anchors[&hi]),
            (None, None) => return Err(FSRSError::NotEnoughData),
        };
        log_s0.insert(target, value);
    }

    let mut values = [
        log_s0[&1].exp() as f32,
        log_s0[&2].exp() as f32,
        log_s0[&3].exp() as f32,
        log_s0[&4].exp() as f32,
    ];
    for value in &mut values {
        *value = value.clamp(0.0001, INIT_S_MAX);
    }
    for i in 1..values.len() {
        values[i] = values[i].max(values[i - 1]);
    }

    Ok(values.map(|v| v.clamp(S_MIN, INIT_S_MAX)))
}

pub(crate) fn smooth_initial_stabilities_fsrs7(initial_stability: [f32; 4]) -> Result<[f32; 4]> {
    fill_initial_stabilities_fsrs7(&HashMap::from([
        (1, initial_stability[0]),
        (2, initial_stability[1]),
        (3, initial_stability[2]),
        (4, initial_stability[3]),
    ]))
}

pub(crate) fn initialize_parameters_fsrs7(
    fsrs_items: Vec<FSRSItem>,
    average_recall: f32,
) -> Result<InitializationResult> {
    let dataset_for_initialization = prepare_dataset_for_initialization(fsrs_items);
    let mut best_curve = default_forgetting_curve_params();
    let mut best_stability = HashMap::new();
    let mut best_rating_count = HashMap::new();
    let mut best_loss = f64::INFINITY;

    for candidate in FSRS7_FORGETTING_CURVE_CANDIDATES {
        let (stability, rating_count, total_loss) = search_parameters_for_curve(
            dataset_for_initialization.clone(),
            average_recall,
            &candidate,
        );
        if !stability.is_empty() && total_loss < best_loss {
            best_loss = total_loss;
            best_curve = candidate;
            best_stability = stability;
            best_rating_count = rating_count;
        }
    }

    let initial_stability = fill_initial_stabilities_fsrs7(&best_stability)?;
    Ok((initial_stability, best_curve, best_rating_count))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_smooth_initial_stabilities_fsrs7_is_monotonic() {
        let actual = smooth_initial_stabilities_fsrs7([4.0, 3.0, 2.0, 1.0]).unwrap();
        assert_eq!(actual, [4.0, 4.0, 4.0, 4.0]);
    }

    #[test]
    fn test_smooth_initial_stabilities_fsrs7_clamps_bounds() {
        let actual = smooth_initial_stabilities_fsrs7([0.0, 0.00005, 0.001, 200.0]).unwrap();
        assert_eq!(actual, [0.0001, 0.0001, 0.001, 100.0]);
    }
}

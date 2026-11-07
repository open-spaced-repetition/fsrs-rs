use crate::FSRS6_DEFAULT_PARAMETERS;
use crate::FSRSItem;
use crate::error::{FSRSError, Result};
use crate::simulation::S_MIN;
use ndarray::Array1;
use std::collections::HashMap;

static R_S0_DEFAULT_ARRAY: &[(u32, f32); 4] = &[
    (1, FSRS6_DEFAULT_PARAMETERS[0]),
    (2, FSRS6_DEFAULT_PARAMETERS[1]),
    (3, FSRS6_DEFAULT_PARAMETERS[2]),
    (4, FSRS6_DEFAULT_PARAMETERS[3]),
];

pub(crate) fn initialize_stability_parameters(
    fsrs_items: Vec<FSRSItem>,
    average_recall: f32,
) -> Result<([f32; 4], HashMap<u32, u32>)> {
    initialize_stability_from_first_reviews(
        fsrs_items
            .iter()
            .filter(|item| item.long_term_review_cnt() == 1)
            .map(|item| {
                let review = item.first_long_term_review();
                InitialReview {
                    rating: item.reviews[0].rating,
                    delta_t: review.delta_t,
                    recalled: review.rating > 1,
                }
            }),
        average_recall,
    )
}

/// What initialization needs from an item with exactly one long-term review:
/// the first rating and the outcome of that long-term review.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct InitialReview {
    pub rating: u32,
    pub delta_t: f32,
    pub recalled: bool,
}

pub(crate) fn initialize_stability_from_first_reviews(
    reviews: impl Iterator<Item = InitialReview>,
    average_recall: f32,
) -> Result<([f32; 4], HashMap<u32, u32>)> {
    let dataset_for_initialization = prepare_dataset_for_initialization(reviews);
    let rating_count = total_rating_count(&dataset_for_initialization);
    let mut rating_stability = search_parameters(dataset_for_initialization, average_recall);
    Ok((
        smooth_and_fill(&mut rating_stability, &rating_count)?,
        rating_count,
    ))
}

type FirstRating = u32;
type Count = u32;

fn prepare_dataset_for_initialization(
    reviews: impl Iterator<Item = InitialReview>,
) -> HashMap<FirstRating, Vec<AverageRecall>> {
    // first rating -> first long-term delta_t -> (recalled, total)
    let mut groups = HashMap::<u32, HashMap<u32, (u64, u64)>>::new();
    for review in reviews {
        let entry = groups
            .entry(review.rating)
            .or_default()
            .entry(review.delta_t.to_bits())
            .or_default();
        entry.0 += u64::from(review.recalled);
        entry.1 += 1;
    }
    groups
        .into_iter()
        .map(|(rating, buckets)| {
            let mut data: Vec<_> = buckets
                .into_iter()
                .map(|(delta, (recalled, count))| AverageRecall {
                    delta_t: f32::from_bits(delta) as f64,
                    recall: recalled as f64 / count as f64,
                    count: count as f64,
                })
                .collect();
            data.sort_unstable_by(|a, b| a.delta_t.total_cmp(&b.delta_t));
            (rating, data)
        })
        .collect()
}

/// The average pass rate & count for a single delta_t for a given first rating.
struct AverageRecall {
    delta_t: f64,
    recall: f64,
    count: f64,
}

fn total_rating_count(
    dataset_for_initialization: &HashMap<FirstRating, Vec<AverageRecall>>,
) -> HashMap<FirstRating, Count> {
    dataset_for_initialization
        .iter()
        .map(|(first_rating, data)| {
            let count = data.iter().map(|d| d.count).sum::<f64>() as u32;
            (*first_rating, count)
        })
        .collect()
}

#[cfg(test)]
fn power_forgetting_curve(t: &Array1<f64>, s: f64) -> Array1<f64> {
    let decay = -FSRS6_DEFAULT_PARAMETERS[20] as f64;
    let factor = 0.9f64.powf(1.0 / decay) - 1.0;
    (t / s * factor + 1.0).mapv(|v| v.powf(decay))
}

/// The initialization objective: BCE of the default forgetting curve plus an
/// absolute-deviation prior. The search minimizes it via `loss_derivatives`;
/// tests compare against it directly.
#[cfg(test)]
fn loss(
    delta_t: &Array1<f64>,
    recall: &Array1<f64>,
    count: &Array1<f64>,
    init_s0: f64,
    default_s0: f64,
) -> f64 {
    let y_pred = power_forgetting_curve(delta_t, init_s0);
    let logloss = (-(recall * y_pred.clone().mapv_into(|v| v.ln())
        + (1.0 - recall) * (1.0 - &y_pred).mapv_into(|v| v.ln()))
        * count)
        .sum();
    let l1 = (init_s0 - default_s0).abs() / 16.0;
    logloss + l1
}

pub(crate) const INIT_S_MAX: f32 = 100.0;

/// First and second derivatives of the initialization BCE with respect to the
/// stability `s` (the prior is handled by the caller). With `c = t * factor`
/// and `p = (1 + c / s)^-decay`: `dp/ds = p * b` and `dBCE/dp = n * (p - y) /
/// (p * (1 - p))`, so the slope is `n * a * b` with `a = (p - y) / (1 - p)`.
fn loss_derivatives(delta_t: &[f64], recall: &[f64], count: &[f64], s: f64) -> (f64, f64) {
    let decay = FSRS6_DEFAULT_PARAMETERS[20] as f64;
    let factor = 0.9f64.powf(-1.0 / decay) - 1.0;
    let mut slope = 0.0;
    let mut curvature = 0.0;
    for ((&t, &y), &n) in delta_t.iter().zip(recall).zip(count) {
        let c = t * factor;
        let p = (1.0 + c / s).powf(-decay);
        let b = decay * c / (s * (s + c));
        let a = (p - y) / (1.0 - p);
        slope += n * a * b;
        curvature +=
            n * ((1.0 - y) * p * (b / (1.0 - p)).powi(2) - a * b * (1.0 / s + 1.0 / (s + c)));
    }
    (slope, curvature)
}

/// Minimizes BCE + |s - default| / 16 over `[S_MIN, INIT_S_MAX]`. The slope at
/// the prior's kink (`default`) decides the side of the optimum. On that side
/// the objective is smooth: Newton steps, safeguarded by bisection, find the
/// root of its slope to f32 precision.
fn initial_stability(delta_t: &[f64], recall: &[f64], count: &[f64], default: f64) -> f32 {
    let (slope, _) = loss_derivatives(delta_t, recall, count, default);
    // The subgradient of |s-default|/16 contains zero here.
    if slope.abs() <= 1.0 / 16.0 {
        return default as f32;
    }
    let (mut low, mut high, prior_slope) = if slope > 0.0 {
        (S_MIN as f64, default, -1.0 / 16.0)
    } else {
        (default, INIT_S_MAX as f64, 1.0 / 16.0)
    };
    // The optimum is the search boundary when the slope keeps its sign there.
    let boundary = if slope > 0.0 { low } else { high };
    let boundary_slope = loss_derivatives(delta_t, recall, count, boundary).0 + prior_slope;
    if (slope > 0.0 && boundary_slope >= 0.0) || (slope < 0.0 && boundary_slope <= 0.0) {
        return boundary as f32;
    }
    let mut s = default;
    for _ in 0..80 {
        let (slope, curvature) = loss_derivatives(delta_t, recall, count, s);
        let slope = slope + prior_slope;
        if slope > 0.0 {
            high = s;
        } else {
            low = s;
        }
        if low as f32 == high as f32 {
            break;
        }
        let newton = s - slope / curvature;
        let next = if curvature > 0.0 && newton > low && newton < high {
            newton
        } else {
            (low + high) / 2.0
        };
        if next as f32 == s as f32 {
            return next as f32;
        }
        s = next;
    }
    ((low + high) / 2.0) as f32
}

fn search_parameters(
    mut dataset_for_initialization: HashMap<FirstRating, Vec<AverageRecall>>,
    average_recall: f32,
) -> HashMap<u32, f32> {
    let mut optimal_stabilities = HashMap::new();

    for (first_rating, data) in &mut dataset_for_initialization {
        let r_s0_default: HashMap<u32, f32> = R_S0_DEFAULT_ARRAY.iter().cloned().collect();
        let default_s0 = r_s0_default[first_rating] as f64;
        let delta_t = Array1::from_iter(data.iter().map(|d| d.delta_t));
        let count = Array1::from_iter(data.iter().map(|d| d.count));
        let recall = {
            // Laplace smoothing
            // (real_recall * n + average_recall * 1) / (n + 1)
            // https://github.com/open-spaced-repetition/fsrs4anki/pull/358/files#diff-35b13c8e3466e8bd1231a51c71524fc31a945a8f332290726214d3a6fa7f442aR491
            let real_recall = Array1::from_iter(data.iter().map(|d| d.recall));
            (real_recall * count.clone() + average_recall as f64) / (count.clone() + 1.0)
        };
        optimal_stabilities.insert(
            *first_rating,
            initial_stability(
                delta_t.as_slice().unwrap(),
                recall.as_slice().unwrap(),
                count.as_slice().unwrap(),
                default_s0,
            ),
        );
    }

    optimal_stabilities
}

pub(crate) fn smooth_and_fill(
    rating_stability: &mut HashMap<u32, f32>,
    rating_count: &HashMap<u32, u32>,
) -> Result<[f32; 4]> {
    rating_stability.retain(|&key, _| rating_count.contains_key(&key));
    // Fixing one pair can break a pair that was already checked, so repeat the
    // pass until the values are in order.
    loop {
        let mut changed = false;
        for (small_rating, big_rating) in [(1, 2), (2, 3), (3, 4), (1, 3), (2, 4), (1, 4)] {
            if let (Some(&small_value), Some(&big_value)) = (
                rating_stability.get(&small_rating),
                rating_stability.get(&big_rating),
            ) && small_value > big_value
            {
                if rating_count[&small_rating] > rating_count[&big_rating] {
                    rating_stability.insert(big_rating, small_value);
                } else {
                    rating_stability.insert(small_rating, big_value);
                }
                changed = true;
            }
        }
        if !changed {
            break;
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
    fn derivative_search_matches_the_previous_trisection_objective() {
        use rand::{RngExt, SeedableRng, rngs::StdRng};
        let mut rng = StdRng::seed_from_u64(607);
        for trial in 0..200 {
            let rating = 1 + trial % 4;
            let default_s = FSRS6_DEFAULT_PARAMETERS[rating as usize - 1] as f64;
            let average = rng.random_range(0.05..0.995);
            let data: Vec<_> = (1..=rng.random_range(1..80))
                .map(|day| AverageRecall {
                    delta_t: day as f64,
                    recall: if trial % 10 == 0 {
                        0.0
                    } else if trial % 10 == 1 {
                        1.0
                    } else {
                        rng.random_range(0.0..1.0)
                    },
                    count: rng.random_range(1..1000) as f64,
                })
                .collect();
            let delta = Array1::from_iter(data.iter().map(|d| d.delta_t));
            let counts = Array1::from_iter(data.iter().map(|d| d.count));
            let recall = Array1::from_iter(
                data.iter()
                    .map(|d| (d.recall * d.count + average as f64) / (d.count + 1.0)),
            );
            let (mut low, mut high) = (S_MIN as f64, INIT_S_MAX as f64);
            for _ in 0..1000 {
                let before = (low, high);
                let left = low + (high - low) / 3.0;
                let right = high - (high - low) / 3.0;
                if loss(&delta, &recall, &counts, left, default_s)
                    < loss(&delta, &recall, &counts, right, default_s)
                {
                    high = right;
                } else {
                    low = left;
                }
                if before == (low, high) {
                    break;
                }
            }
            let reference = ((low + high) / 2.0) as f32;
            let actual = search_parameters(HashMap::from([(rating, data)]), average)[&rating];
            let reference_loss = loss(&delta, &recall, &counts, reference as f64, default_s);
            let actual_loss = loss(&delta, &recall, &counts, actual as f64, default_s);
            assert!(
                actual_loss <= reference_loss + 1e-8 * reference_loss.max(1.0),
                "trial={trial}, {actual} vs {reference}: {actual_loss} vs {reference_loss}"
            );
        }
    }

    #[test]
    fn test_power_forgetting_curve() {
        let t = Array1::from(vec![0.0, 1.0, 2.0, 3.0]);
        let s = 1.0;
        let y = power_forgetting_curve(&t, s);
        let expected = Array1::from(vec![1.0, 0.9, 0.8458846447796301, 0.8093881028681906]);
        assert_eq!(y, expected);
    }

    #[test]
    fn test_loss() {
        let delta_t = Array1::from(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let recall = Array1::from(vec![
            0.86666667, 0.90721649, 0.73015873, 0.76315789, 0.67857143,
        ]);
        let count = Array1::from(vec![435.0, 97.0, 63.0, 38.0, 28.0]);
        let default_s0 = FSRS6_DEFAULT_PARAMETERS[0] as f64;
        let actual = loss(&delta_t, &recall, &count, 0.7840586, default_s0);
        assert_eq!(actual, 279.9206961069712);
        let actual = loss(&delta_t, &recall, &count, 0.7840590622451964, default_s0);
        assert_eq!(actual, 279.9206977311556);
    }

    #[test]
    fn test_search_parameters() {
        let first_rating = 1;
        let dataset_for_initialization = HashMap::from([(
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
        let actual = search_parameters(dataset_for_initialization, 0.943_028_57);
        [*actual.get(&first_rating).unwrap()].assert_approx_eq([0.7355089]);
    }

    #[test]
    fn test_initialize_stability_parameters() {
        use crate::convertor_tests::anki21_sample_file_converted_to_fsrs;
        let items = anki21_sample_file_converted_to_fsrs();
        let (mut dataset_for_initialization, mut trainset) = items
            .into_iter()
            .partition(|item| item.long_term_review_cnt() == 1);
        (dataset_for_initialization, trainset) =
            filter_outlier(dataset_for_initialization, trainset);
        let items = [dataset_for_initialization.clone(), trainset].concat();
        let average_recall = calculate_average_recall(&items);

        initialize_stability_parameters(dataset_for_initialization, average_recall)
            .unwrap()
            .0
            .assert_approx_eq([0.73550886, 2.1338913, 4.473312, 11.087741])
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
        assert_eq!(actual, [0.05738148, 0.35, 0.6242943, 2.2453482]);
    }

    #[test]
    fn test_smooth_and_fill_fixes_pairs_broken_later_in_the_pass() {
        // (2, 3) is in order when it is checked, but (3, 4) then lowers rating 3.
        let mut rating_stability = HashMap::from([(2, 2.0), (3, 5.0), (4, 0.5)]);
        let rating_count = HashMap::from([(2, 200), (3, 100), (4, 100)]);
        let actual = smooth_and_fill(&mut rating_stability, &rating_count).unwrap();
        assert_eq!(actual, [2.0, 2.0, 2.0, 2.0]);
    }

    #[test]
    fn test_smooth_and_fill_is_non_decreasing() {
        // Only the relative order of the values and of the counts matters, so
        // this covers every case for the ratings that are present.
        let values = [0.5, 1.0, 2.0, 4.0];
        for mask in 1..16u32 {
            let ratings: Vec<u32> = (1..=4).filter(|r| mask & (1 << (r - 1)) != 0).collect();
            let n = ratings.len() as u32;
            for value_index in 0..4u32.pow(n) {
                for count_index in 0..4u32.pow(n) {
                    let mut rating_stability = HashMap::new();
                    let mut rating_count = HashMap::new();
                    for (i, &rating) in ratings.iter().enumerate() {
                        let digit = |x: u32| (x / 4u32.pow(i as u32)) % 4;
                        rating_stability.insert(rating, values[digit(value_index) as usize]);
                        rating_count.insert(rating, digit(count_index) + 1);
                    }
                    let input = rating_stability.clone();
                    let actual = smooth_and_fill(&mut rating_stability, &rating_count).unwrap();
                    // Filling in a missing rating from two equal values can be off by
                    // an ulp, so allow for rounding.
                    assert!(
                        actual.windows(2).all(|w| w[0] <= w[1] * (1.0 + 1e-6)),
                        "{input:?} {rating_count:?} -> {actual:?}"
                    );
                }
            }
        }
    }
}

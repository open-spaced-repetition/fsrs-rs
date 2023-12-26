use crate::error::{FSRSError, Result};
use crate::inference::{DECAY, FACTOR};
use crate::FSRSItem;
use crate::DEFAULT_WEIGHTS;
use ndarray::Array1;
use std::collections::HashMap;

static R_S0_DEFAULT_ARRAY: &[(u32, f32); 4] = &[
    (1, DEFAULT_WEIGHTS[0]),
    (2, DEFAULT_WEIGHTS[1]),
    (3, DEFAULT_WEIGHTS[2]),
    (4, DEFAULT_WEIGHTS[3]),
];

pub fn pretrain(fsrs_items: Vec<FSRSItem>, average_recall: f32) -> Result<[f32; 4]> {
    let pretrainset = create_pretrain_data(fsrs_items);
    let rating_count = total_rating_count(&pretrainset);
    let mut rating_stability = search_parameters(pretrainset, average_recall);
    smooth_and_fill(&mut rating_stability, &rating_count)
}

type FirstRating = u32;
type Count = u32;

fn create_pretrain_data(fsrs_items: Vec<FSRSItem>) -> HashMap<FirstRating, Vec<AverageRecall>> {
    // filter FSRSItem instances with exactly 2 reviews.
    let items: Vec<_> = fsrs_items
        .into_iter()
        .filter(|item| item.reviews.len() == 2)
        .collect();

    // use a nested HashMap (groups) to group items first by the rating in the first FSRSReview
    // and then by the delta_t in the second FSRSReview.
    // (first_rating -> second_delta_t -> vec![0/1 for fail/pass])
    let mut groups = HashMap::new();

    for item in items {
        let first_rating = item.reviews[0].rating;
        let second_delta_t = item.reviews[1].delta_t;
        let second_label = (item.reviews[1].rating > 1) as i32;

        let inner_map = groups.entry(first_rating).or_insert_with(HashMap::new);
        let ratings = inner_map.entry(second_delta_t).or_insert_with(Vec::new);
        ratings.push(second_label);
    }

    let mut results = HashMap::new();

    for (first_rating, inner_map) in &groups {
        let mut data = vec![];

        // calculate the average (recall) and count for each group.
        for (second_delta_t, ratings) in inner_map {
            let avg = ratings.iter().map(|&x| x as f64).sum::<f64>() / ratings.len() as f64;
            data.push(AverageRecall {
                delta_t: *second_delta_t as f32,
                recall: avg as f32,
                count: ratings.len() as f32,
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
    delta_t: f32,
    recall: f32,
    count: f32,
}

fn total_rating_count(
    pretrainset: &HashMap<FirstRating, Vec<AverageRecall>>,
) -> HashMap<FirstRating, Count> {
    pretrainset
        .iter()
        .map(|(first_rating, data)| {
            let count = data.iter().map(|d| d.count).sum::<f32>() as u32;
            (*first_rating, count)
        })
        .collect()
}

fn power_forgetting_curve(t: &Array1<f32>, s: f32) -> Array1<f32> {
    (t / s * FACTOR as f32 + 1.0).mapv(|v| v.powf(DECAY as f32))
}

fn loss(
    delta_t: &Array1<f32>,
    recall: &Array1<f32>,
    count: &Array1<f32>,
    init_s0: f32,
    default_s0: f32,
) -> f32 {
    let y_pred = power_forgetting_curve(delta_t, init_s0);
    let logloss = (-(recall * y_pred.clone().mapv_into(|v| v.ln())
        + (1.0 - recall) * (1.0 - &y_pred).mapv_into(|v| v.ln()))
        * count.mapv(|v| v.sqrt()))
    .sum();
    let l1 = (init_s0 - default_s0).abs() / 16.0;
    logloss + l1
}

const MIN_INIT_S0: f32 = 0.1;
const MAX_INIT_S0: f32 = 100.0;

fn search_parameters(
    mut pretrainset: HashMap<FirstRating, Vec<AverageRecall>>,
    average_recall: f32,
) -> HashMap<u32, f32> {
    let mut optimal_stabilities = HashMap::new();
    let epsilon = f32::EPSILON;

    for (first_rating, data) in &mut pretrainset {
        let r_s0_default: HashMap<u32, f32> = R_S0_DEFAULT_ARRAY.iter().cloned().collect();
        let default_s0 = r_s0_default[first_rating];
        let delta_t = Array1::from_iter(data.iter().map(|d| d.delta_t));
        let recall = {
            // Laplace smoothing
            // (real_recall * n + average_recall * 1) / (n + 1)
            // https://github.com/open-spaced-repetition/fsrs4anki/pull/358/files#diff-35b13c8e3466e8bd1231a51c71524fc31a945a8f332290726214d3a6fa7f442aR491
            let real_recall = Array1::from_iter(data.iter().map(|d| d.recall));
            let n = data.iter().map(|d| d.count).sum::<f32>();
            (real_recall * n + average_recall) / (n + 1.0)
        };
        let count = Array1::from_iter(data.iter().map(|d| d.count));

        let mut low = MIN_INIT_S0;
        let mut high = MAX_INIT_S0;
        let mut optimal_s = 1.0;

        let mut iter = 0;
        while high - low > epsilon && iter < 1000 {
            iter += 1;
            let mid1 = low + (high - low) / 3.0;
            let mid2 = high - (high - low) / 3.0;

            let loss1 = loss(&delta_t, &recall, &count, mid1, default_s0);
            let loss2 = loss(&delta_t, &recall, &count, mid2, default_s0);

            if loss1 < loss2 {
                high = mid2;
            } else {
                low = mid1;
            }

            optimal_s = (high + low) / 2.0;
        }

        optimal_stabilities.insert(*first_rating, optimal_s);
    }

    optimal_stabilities
}

fn smooth_and_fill(
    rating_stability: &mut HashMap<u32, f32>,
    rating_count: &HashMap<u32, u32>,
) -> Result<[f32; 4]> {
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

    let w1 = 3.0 / 5.0;
    let w2 = 3.0 / 5.0;

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
        .map(|&v| v.clamp(MIN_INIT_S0, MAX_INIT_S0))
        .collect();
    Ok(init_s0[0..=3].try_into().unwrap())
}

#[cfg(test)]
mod tests {
    use burn::tensor::Data;

    use super::*;
    use crate::dataset::split_data;
    use crate::training::calculate_average_recall;

    #[test]
    fn test_power_forgetting_curve() {
        let t = Array1::from(vec![0.0, 1.0, 2.0, 3.0]);
        let s = 1.0;
        let y = power_forgetting_curve(&t, s);
        let expected = Array1::from(vec![1.0, 0.90000004, 0.82502866, 0.76613086]);
        assert_eq!(y, expected);
    }

    #[test]
    fn test_loss() {
        let delta_t = Array1::from(vec![1.0, 2.0, 3.0]);
        let recall = Array1::from(vec![0.9, 0.8181818, 0.75]);
        let count = Array1::from(vec![100.0, 100.0, 100.0]);
        let init_s0 = 1.0;
        let actual = loss(&delta_t, &recall, &count, init_s0, init_s0);
        assert_eq!(actual, 13.624332);
        Data::from([loss(&delta_t, &recall, &count, 2.0, init_s0)])
            .assert_approx_eq(&Data::from([14.5771]), 5);
    }

    #[test]
    fn test_search_parameters() {
        let pretrainset = HashMap::from([(
            4,
            vec![
                AverageRecall {
                    delta_t: 1.0,
                    recall: 0.9,
                    count: 30.0,
                },
                AverageRecall {
                    delta_t: 2.0,
                    recall: 0.8181818,
                    count: 30.0,
                },
                AverageRecall {
                    delta_t: 3.0,
                    recall: 0.75,
                    count: 30.0,
                },
                AverageRecall {
                    delta_t: 4.0,
                    recall: 0.6923077,
                    count: 30.0,
                },
            ],
        )]);
        let actual = search_parameters(pretrainset, 0.9);
        Data::from([*actual.get(&4).unwrap()]).assert_approx_eq(&Data::from([0.944_284]), 4);
    }

    #[test]
    fn test_pretrain() {
        use crate::convertor_tests::anki21_sample_file_converted_to_fsrs;
        let items = anki21_sample_file_converted_to_fsrs();
        let average_recall = calculate_average_recall(&items);
        let pretrainset = split_data(items, 1).0;
        Data::from(pretrain(pretrainset, average_recall).unwrap())
            .assert_approx_eq(&Data::from([1.001_276, 1.810_509, 4.401_906, 8.529_174]), 4)
    }

    #[test]
    fn test_smooth_and_fill() {
        let mut rating_stability = HashMap::from([(1, 0.4), (3, 2.3), (4, 10.9)]);
        let rating_count = HashMap::from([(1, 1), (2, 1), (3, 1), (4, 1)]);
        let actual = smooth_and_fill(&mut rating_stability, &rating_count).unwrap();
        assert_eq!(actual, [0.4, 0.8052433, 2.3, 10.9,]);

        let mut rating_stability = HashMap::from([(2, 0.35)]);
        let rating_count = HashMap::from([(2, 1)]);
        let actual = smooth_and_fill(&mut rating_stability, &rating_count).unwrap();
        assert_eq!(actual, [0.15661564, 0.35, 1.0009006, 2.2242827,]);
    }
}

use itertools::Itertools;
use ndarray::Array1;
use std::collections::HashMap;
use std::iter::Iterator;

use crate::FSRSItem;

pub fn pretrain(fsrs_items: Vec<FSRSItem>) -> [f32; 4] {
    let pretrainset = create_pretrain_data(fsrs_items);
    let rating_count = total_rating_count(pretrainset.clone());
    let rating_stability = search_parameters(pretrainset);
    smooth_and_fill(&mut rating_stability.clone(), &rating_count)
}

fn create_pretrain_data(fsrs_items: Vec<FSRSItem>) -> HashMap<i32, HashMap<String, Vec<f32>>> {
    // filter FSRSItem instances with exactly 2 reviews.
    let items: Vec<_> = fsrs_items
        .into_iter()
        .filter(|item| item.reviews.len() == 2)
        .collect();

    // use a nested HashMap (groups) to group items first by the rating in the first FSRSReview
    // and then by the delta_t in the second FSRSReview.
    let mut groups = HashMap::new();

    for item in items {
        let first_rating = item.reviews[0].rating;
        let second_delta_t = item.reviews[1].delta_t;
        let second_label = if item.reviews[1].rating == 1 { 0 } else { 1 };

        let inner_map = groups.entry(first_rating).or_insert_with(HashMap::new);
        let ratings = inner_map.entry(second_delta_t).or_insert_with(Vec::new);
        ratings.push(second_label);
    }

    let mut results = HashMap::new();

    for (first_rating, inner_map) in &groups {
        let mut data = vec![]; // (delta_t, recall, count)

        // calculate the average (recall) and count for each group.
        for (second_delta_t, ratings) in inner_map {
            let avg = ratings.iter().map(|&x| x as f64).sum::<f64>() / ratings.len() as f64;
            data.push((*second_delta_t, avg, ratings.len()));
        }

        // Sort by delta_t in ascending order
        data.sort_by(|a, b| a.0.cmp(&b.0));

        let inner_result = HashMap::from([
            (
                "delta_t".to_string(),
                data.iter().map(|&(dt, _, _)| dt as f32).collect(),
            ),
            (
                "recall".to_string(),
                data.iter().map(|&(_, r, _)| r as f32).collect(),
            ),
            (
                "count".to_string(),
                data.iter().map(|&(_, _, c)| c as f32).collect(),
            ),
        ]);

        results.insert(*first_rating, inner_result);
    }
    results
}

fn total_rating_count(pretrainset: HashMap<i32, HashMap<String, Vec<f32>>>) -> HashMap<i32, i32> {
    let mut rating_count = HashMap::new();
    for (first_rating, data) in pretrainset.iter() {
        let count = data["count"].iter().sum::<f32>() as i32;
        rating_count.insert(*first_rating, count);
    }
    rating_count
}

fn power_forgetting_curve(t: &Array1<f32>, s: f32) -> Array1<f32> {
    1.0 / (1.0 + t / (9.0 * s))
}

fn loss(delta_t: &Array1<f32>, recall: &Array1<f32>, count: &Array1<f32>, init_s0: f32) -> f32 {
    let y_pred = power_forgetting_curve(delta_t, init_s0);
    let diff = recall - &y_pred;
    let weighted_diff = &diff * &diff * count;
    (weighted_diff.sum() / count.sum()).sqrt()
    // TODO: add l1 regularization
}

fn search_parameters(pretrainset: HashMap<i32, HashMap<String, Vec<f32>>>) -> HashMap<i32, f32> {
    let mut optimal_stabilities: HashMap<i32, f32> = HashMap::new();
    let epsilon = 1e-3; // precision, use f32::EPSILON will cause inf running.

    for (first_rating, data) in pretrainset.iter() {
        let delta_t = Array1::from(data["delta_t"].clone());
        let recall = Array1::from(data["recall"].clone());
        let count = Array1::from(data["count"].clone());

        let mut low = 0.0;
        let mut high = 100.0;
        let mut optimal_s = 0.0;

        // TODO: consider limit the number of iterations to avoid infinite loop.
        while high - low > epsilon {
            let mid1 = low + (high - low) / 3.0;
            let mid2 = high - (high - low) / 3.0;

            let loss1 = loss(&delta_t, &recall, &count, mid1);
            let loss2 = loss(&delta_t, &recall, &count, mid2);

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
    rating_stability: &mut HashMap<i32, f32>,
    rating_count: &HashMap<i32, i32>,
) -> [f32; 4] {
    for &(small_rating, big_rating) in &[(1, 2), (2, 3), (3, 4), (1, 3), (2, 4), (1, 4)] {
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

    let r_s0_default = HashMap::from([(1, 0.4), (2, 0.6), (3, 2.4), (4, 5.8)]);

    match rating_stability.len() {
        0 => panic!("Not enough data for pretraining!"),
        1 => {
            let rating = rating_stability.keys().next().unwrap();
            let factor = rating_stability[rating] / r_s0_default[rating];
            init_s0 = r_s0_default.values().map(|&x| x * factor).collect();
        }
        2 => {
            match (
                rating_stability.contains_key(&1),
                rating_stability.contains_key(&2),
                rating_stability.contains_key(&3),
                rating_stability.contains_key(&4),
            ) {
                (false, false, true, true) => {
                    rating_stability.insert(
                        2,
                        rating_stability[&3].powf(1.0 / (1.0 - w2))
                            * rating_stability[&4].powf(1.0 - 1.0 / (1.0 - w2)),
                    );
                    rating_stability.insert(
                        1,
                        f32::powf(rating_stability[&2], 1.0 / w1)
                            * f32::powf(rating_stability[&3], 1.0 - 1.0 / w1),
                    );
                }
                (false, true, false, true) => {
                    rating_stability.insert(
                        3,
                        f32::powf(rating_stability[&2], 1.0 - w2)
                            * f32::powf(rating_stability[&4], w2),
                    );
                    rating_stability.insert(
                        1,
                        f32::powf(rating_stability[&2], 1.0 / w1)
                            * f32::powf(rating_stability[&3], 1.0 - 1.0 / w1),
                    );
                }
                (false, true, true, false) => {
                    rating_stability.insert(
                        4,
                        rating_stability[&2].powf(1.0 - 1.0 / w2)
                            * rating_stability[&3].powf(1.0 / w2),
                    );
                    rating_stability.insert(
                        1,
                        rating_stability[&2].powf(1.0 / w1)
                            * rating_stability[&3].powf(1.0 - 1.0 / w1),
                    );
                }
                (true, false, false, true) => {
                    rating_stability.insert(
                        2,
                        rating_stability[&1].powf(w1 / (w1 + w2 - w1 * w2))
                            * rating_stability[&4].powf(1.0 - w1 / (w1 + w2 - w1 * w2)),
                    );
                    rating_stability.insert(
                        3,
                        rating_stability[&1].powf(1.0 - w2 / (w1 + w2 - w1 * w2))
                            * rating_stability[&4].powf(w2 / (w1 + w2 - w1 * w2)),
                    );
                }
                (true, false, true, false) => {
                    rating_stability.insert(
                        2,
                        rating_stability[&1].powf(w1) * rating_stability[&3].powf(1.0 - w1),
                    );
                    rating_stability.insert(
                        4,
                        rating_stability[&2].powf(1.0 - 1.0 / w2)
                            * rating_stability[&3].powf(1.0 / w2),
                    );
                }
                (true, true, false, false) => {
                    rating_stability.insert(
                        3,
                        rating_stability[&1].powf(1.0 - 1.0 / (1.0 - w1))
                            * rating_stability[&2].powf(1.0 / (1.0 - w1)),
                    );
                    rating_stability.insert(
                        4,
                        rating_stability[&2].powf(1.0 - 1.0 / w2)
                            * rating_stability[&3].powf(1.0 / w2),
                    );
                }
                _ => {}
            }
            let mut sorted_items: Vec<_> = rating_stability.iter().collect();
            sorted_items.sort_by(|a, b| a.0.cmp(b.0));
            init_s0 = sorted_items.iter().map(|&(_, &value)| value).collect();
        }
        3 => {
            match (
                rating_stability.contains_key(&1),
                rating_stability.contains_key(&2),
                rating_stability.contains_key(&3),
                rating_stability.contains_key(&4),
            ) {
                (false, _, _, _) => {
                    rating_stability.insert(
                        1,
                        rating_stability[&2].powf(1.0 / w1)
                            * rating_stability[&3].powf(1.0 - 1.0 / w1),
                    );
                }
                (_, false, _, _) => {
                    rating_stability.insert(
                        2,
                        rating_stability[&1].powf(w1) * rating_stability[&3].powf(1.0 - w1),
                    );
                }
                (_, _, false, _) => {
                    rating_stability.insert(
                        3,
                        rating_stability[&2].powf(1.0 - w2) * rating_stability[&4].powf(w2),
                    );
                }
                (_, _, _, false) => {
                    rating_stability.insert(
                        4,
                        rating_stability[&2].powf(1.0 - 1.0 / w2)
                            * rating_stability[&3].powf(1.0 / w2),
                    );
                }
                _ => {}
            }
            let mut sorted_items: Vec<_> = rating_stability.iter().collect();
            sorted_items.sort_by(|a, b| a.0.cmp(b.0));
            init_s0 = sorted_items.iter().map(|&(_, &value)| value).collect();
        }
        4 => {
            init_s0 = rating_stability
                .iter()
                .sorted_by(|a, b| a.0.cmp(b.0))
                .map(|(_, &v)| v)
                .collect();
        }
        _ => {}
    }
    [init_s0[0], init_s0[1], init_s0[2], init_s0[3]]
}

#[test]
fn test_power_forgetting_curve() {
    let t = Array1::from(vec![0.0, 1.0, 2.0, 3.0]);
    let s = 1.0;
    let y = power_forgetting_curve(&t, s);
    let expected = Array1::from(vec![1.0, 0.9, 0.8181818, 0.75]);
    assert_eq!(y, expected);
}

#[test]
fn test_loss() {
    let delta_t = Array1::from(vec![0.0, 1.0, 2.0, 3.0]);
    let recall = Array1::from(vec![1.0, 0.9, 0.8181818, 0.75]);
    let count = Array1::from(vec![1.0, 1.0, 1.0, 1.0]);
    let init_s0 = 1.0;
    let expected = 0.0;
    let actual = loss(&delta_t, &recall, &count, init_s0);
    assert_eq!(actual, expected);
    assert_eq!(loss(&delta_t, &recall, &count, 2.0), 0.07144503);
}

#[test]
fn test_pretrain() {
    use crate::convertor::tests::anki21_sample_file_converted_to_fsrs;
    assert_eq!(
        pretrain(anki21_sample_file_converted_to_fsrs()),
        [1.1615174, 2.2759974, 6.2429657, 11.392988,]
    )
}

#[test]
fn test_smooth_and_fill() {
    let mut rating_stability = HashMap::from([(1, 0.4), (3, 2.4), (4, 5.8)]);
    let rating_count = HashMap::from([(1, 1), (2, 1), (3, 1), (4, 1)]);
    let actual = smooth_and_fill(&mut rating_stability, &rating_count);
    dbg!(actual);
}

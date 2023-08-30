use ndarray::Array1;
use std::collections::HashMap;

use crate::FSRSItem;

fn power_forgetting_curve(t: &Array1<f32>, s: f32) -> Array1<f32> {
    1.0 / (1.0 + t / (9.0 * s))
}

fn loss(delta_t: &Array1<f32>, recall: &Array1<f32>, count: &Array1<f32>, init_s0: f32) -> f32 {
    let y_pred = power_forgetting_curve(delta_t, init_s0);
    let diff = recall - &y_pred;
    let weighted_diff = &diff * &diff * count;
    let rmse = f32::sqrt(weighted_diff.sum() / count.sum());
    // TODO: add l1 regularization
    rmse
}

fn search_parameters(pretrainset: HashMap<i32, HashMap<String, Vec<f32>>>) -> HashMap<i32, f32> {
    let mut optimal_stabilities: HashMap<i32, f32> = HashMap::new();

    let epsilon = 1e-6; // precision

    for (first_rating, data) in pretrainset.iter() {
        let delta_t = Array1::from(data["delta_t"].clone());
        let recall = Array1::from(data["recall"].clone());
        let count = Array1::from(data["count"].clone());

        let mut low = 0.0;
        let mut high = 100.0;
        let mut optimal_s = 0.0;

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

fn create_pretrain_data(fsrs_items: Vec<FSRSItem>) -> HashMap<i32, HashMap<String, Vec<f32>>> {
    // filter FSRSItem instances with exactly 2 reviews.
    let items: Vec<_> = fsrs_items
        .into_iter()
        .filter(|item| item.reviews.len() == 2)
        .collect();

    // use a nested HashMap (groups) to group items first by the rating in the first FSRSReview
    // and then by the delta_t in the second FSRSReview.
    let mut groups: HashMap<i32, HashMap<i32, Vec<i32>>> = HashMap::new();

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
        let mut data = Vec::new(); // (delta_t, recall, count)

        // calculate the average (recall) and count for each group.
        for (second_delta_t, ratings) in inner_map {
            let avg: f64 = ratings.iter().map(|&x| x as f64).sum::<f64>() / ratings.len() as f64;
            data.push((*second_delta_t, avg, ratings.len()));
        }

        // Sort by delta_t in ascending order
        data.sort_by(|a, b| a.0.cmp(&b.0));

        let delta_t: Vec<_> = data.iter().map(|&(dt, _, _)| dt as f32).collect();
        let recall: Vec<_> = data.iter().map(|&(_, r, _)| r as f32).collect();
        let count: Vec<_> = data.iter().map(|&(_, _, c)| c as f32).collect();

        let mut inner_result = HashMap::new();
        inner_result.insert("delta_t".to_string(), delta_t);
        inner_result.insert("recall".to_string(), recall);
        inner_result.insert("count".to_string(), count);

        results.insert(*first_rating, inner_result);
    }
    results
}

#[test]
fn test_search_parameters() {
    use crate::convertor::tests::anki21_sample_file_converted_to_fsrs;
    let fsrs_items = anki21_sample_file_converted_to_fsrs();
    let pre_train_data = create_pretrain_data(fsrs_items);
    dbg!(search_parameters(pre_train_data));
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

use std::collections::HashMap;

use crate::FSRSItem;

fn create_pretrain_data(fsrs_items: Vec<FSRSItem>) -> HashMap<i32, HashMap<String, Vec<f64>>> {
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

        let delta_t: Vec<_> = data.iter().map(|&(dt, _, _)| dt as f64).collect();
        let recall: Vec<_> = data.iter().map(|&(_, r, _)| r as f64).collect();
        let count: Vec<_> = data.iter().map(|&(_, _, c)| c as f64).collect();

        let mut inner_result = HashMap::new();
        inner_result.insert("delta_t".to_string(), delta_t);
        inner_result.insert("recall".to_string(), recall);
        inner_result.insert("count".to_string(), count);

        results.insert(*first_rating, inner_result);
    }
    results
}

#[test]
fn test() {
    use crate::convertor::tests::anki21_sample_file_converted_to_fsrs;
    let fsrs_items = anki21_sample_file_converted_to_fsrs();
    dbg!(create_pretrain_data(fsrs_items));
}

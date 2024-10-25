use chrono::NaiveDate;
use fsrs::{FSRSItem, FSRSReview, DEFAULT_PARAMETERS, FSRS};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create review histories for 10 cards
    let card_histories = create_card_histories();

    // Convert card histories to FSRSItems
    let fsrs_items: Vec<FSRSItem> = card_histories
        .into_iter()
        .map(|history| convert_to_fsrs_item(history))
        .flatten()
        .collect();

    println!("Size of FSRSItems: {}", fsrs_items.len());

    // Create an FSRS instance with default parameters
    let fsrs = FSRS::new(Some(&[]))?;
    println!("Default parameters: {:?}", DEFAULT_PARAMETERS);

    // Optimize the FSRS model using the created items
    let optimized_parameters = fsrs.compute_parameters(fsrs_items, None)?;

    println!("Optimized parameters: {:?}", optimized_parameters);

    Ok(())
}

fn create_card_histories() -> Vec<Vec<(NaiveDate, u32)>> {
    vec![
        vec![
            (NaiveDate::from_ymd_opt(2023, 1, 1).unwrap(), 3),
            (NaiveDate::from_ymd_opt(2023, 1, 2).unwrap(), 4),
            (NaiveDate::from_ymd_opt(2023, 1, 5).unwrap(), 3),
            (NaiveDate::from_ymd_opt(2023, 1, 15).unwrap(), 4),
            (NaiveDate::from_ymd_opt(2023, 2, 1).unwrap(), 3),
            (NaiveDate::from_ymd_opt(2023, 2, 20).unwrap(), 4),
        ],
        vec![
            (NaiveDate::from_ymd_opt(2023, 1, 1).unwrap(), 2),
            (NaiveDate::from_ymd_opt(2023, 1, 2).unwrap(), 3),
            (NaiveDate::from_ymd_opt(2023, 1, 4).unwrap(), 4),
            (NaiveDate::from_ymd_opt(2023, 1, 12).unwrap(), 3),
            (NaiveDate::from_ymd_opt(2023, 1, 28).unwrap(), 4),
            (NaiveDate::from_ymd_opt(2023, 2, 15).unwrap(), 3),
            (NaiveDate::from_ymd_opt(2023, 3, 5).unwrap(), 4),
        ],
        vec![
            (NaiveDate::from_ymd_opt(2023, 1, 1).unwrap(), 4),
            (NaiveDate::from_ymd_opt(2023, 1, 8).unwrap(), 4),
            (NaiveDate::from_ymd_opt(2023, 1, 24).unwrap(), 3),
            (NaiveDate::from_ymd_opt(2023, 2, 10).unwrap(), 4),
            (NaiveDate::from_ymd_opt(2023, 3, 1).unwrap(), 3),
        ],
        vec![
            (NaiveDate::from_ymd_opt(2023, 1, 1).unwrap(), 1),
            (NaiveDate::from_ymd_opt(2023, 1, 2).unwrap(), 1),
            (NaiveDate::from_ymd_opt(2023, 1, 3).unwrap(), 3),
            (NaiveDate::from_ymd_opt(2023, 1, 6).unwrap(), 4),
            (NaiveDate::from_ymd_opt(2023, 1, 16).unwrap(), 4),
            (NaiveDate::from_ymd_opt(2023, 2, 1).unwrap(), 3),
            (NaiveDate::from_ymd_opt(2023, 2, 20).unwrap(), 4),
        ],
        vec![
            (NaiveDate::from_ymd_opt(2023, 1, 1).unwrap(), 3),
            (NaiveDate::from_ymd_opt(2023, 1, 3).unwrap(), 3),
            (NaiveDate::from_ymd_opt(2023, 1, 8).unwrap(), 2),
            (NaiveDate::from_ymd_opt(2023, 1, 10).unwrap(), 4),
            (NaiveDate::from_ymd_opt(2023, 1, 22).unwrap(), 3),
            (NaiveDate::from_ymd_opt(2023, 2, 5).unwrap(), 4),
            (NaiveDate::from_ymd_opt(2023, 2, 25).unwrap(), 3),
        ],
        vec![
            (NaiveDate::from_ymd_opt(2023, 1, 1).unwrap(), 4),
            (NaiveDate::from_ymd_opt(2023, 1, 9).unwrap(), 3),
            (NaiveDate::from_ymd_opt(2023, 1, 19).unwrap(), 4),
            (NaiveDate::from_ymd_opt(2023, 2, 5).unwrap(), 3),
            (NaiveDate::from_ymd_opt(2023, 2, 25).unwrap(), 4),
        ],
        vec![
            (NaiveDate::from_ymd_opt(2023, 1, 1).unwrap(), 2),
            (NaiveDate::from_ymd_opt(2023, 1, 2).unwrap(), 3),
            (NaiveDate::from_ymd_opt(2023, 1, 5).unwrap(), 4),
            (NaiveDate::from_ymd_opt(2023, 1, 15).unwrap(), 3),
            (NaiveDate::from_ymd_opt(2023, 1, 30).unwrap(), 4),
            (NaiveDate::from_ymd_opt(2023, 2, 15).unwrap(), 3),
            (NaiveDate::from_ymd_opt(2023, 3, 5).unwrap(), 4),
        ],
        vec![
            (NaiveDate::from_ymd_opt(2023, 1, 1).unwrap(), 3),
            (NaiveDate::from_ymd_opt(2023, 1, 4).unwrap(), 4),
            (NaiveDate::from_ymd_opt(2023, 1, 14).unwrap(), 4),
            (NaiveDate::from_ymd_opt(2023, 2, 1).unwrap(), 3),
            (NaiveDate::from_ymd_opt(2023, 2, 20).unwrap(), 4),
        ],
        vec![
            (NaiveDate::from_ymd_opt(2023, 1, 1).unwrap(), 1),
            (NaiveDate::from_ymd_opt(2023, 1, 2).unwrap(), 1),
            (NaiveDate::from_ymd_opt(2023, 1, 3).unwrap(), 3),
            (NaiveDate::from_ymd_opt(2023, 1, 7).unwrap(), 3),
            (NaiveDate::from_ymd_opt(2023, 1, 15).unwrap(), 4),
            (NaiveDate::from_ymd_opt(2023, 1, 31).unwrap(), 3),
            (NaiveDate::from_ymd_opt(2023, 2, 15).unwrap(), 4),
            (NaiveDate::from_ymd_opt(2023, 3, 5).unwrap(), 3),
        ],
        vec![
            (NaiveDate::from_ymd_opt(2023, 1, 1).unwrap(), 4),
            (NaiveDate::from_ymd_opt(2023, 1, 10).unwrap(), 3),
            (NaiveDate::from_ymd_opt(2023, 1, 20).unwrap(), 4),
            (NaiveDate::from_ymd_opt(2023, 2, 5).unwrap(), 4),
            (NaiveDate::from_ymd_opt(2023, 2, 25).unwrap(), 3),
            (NaiveDate::from_ymd_opt(2023, 3, 15).unwrap(), 4),
        ],
        vec![
            (NaiveDate::from_ymd_opt(2023, 1, 1).unwrap(), 1),
            (NaiveDate::from_ymd_opt(2023, 1, 2).unwrap(), 2),
            (NaiveDate::from_ymd_opt(2023, 1, 3).unwrap(), 3),
            (NaiveDate::from_ymd_opt(2023, 1, 4).unwrap(), 4),
            (NaiveDate::from_ymd_opt(2023, 1, 10).unwrap(), 3),
            (NaiveDate::from_ymd_opt(2023, 1, 20).unwrap(), 4),
            (NaiveDate::from_ymd_opt(2023, 2, 5).unwrap(), 3),
            (NaiveDate::from_ymd_opt(2023, 2, 25).unwrap(), 4),
        ],
        vec![
            (NaiveDate::from_ymd_opt(2023, 1, 1).unwrap(), 3),
            (NaiveDate::from_ymd_opt(2023, 1, 5).unwrap(), 4),
            (NaiveDate::from_ymd_opt(2023, 1, 15).unwrap(), 3),
            (NaiveDate::from_ymd_opt(2023, 1, 30).unwrap(), 4),
            (NaiveDate::from_ymd_opt(2023, 2, 15).unwrap(), 3),
            (NaiveDate::from_ymd_opt(2023, 3, 5).unwrap(), 4),
        ],
        vec![
            (NaiveDate::from_ymd_opt(2023, 1, 1).unwrap(), 2),
            (NaiveDate::from_ymd_opt(2023, 1, 3).unwrap(), 3),
            (NaiveDate::from_ymd_opt(2023, 1, 7).unwrap(), 4),
            (NaiveDate::from_ymd_opt(2023, 1, 17).unwrap(), 3),
            (NaiveDate::from_ymd_opt(2023, 2, 1).unwrap(), 4),
            (NaiveDate::from_ymd_opt(2023, 2, 20).unwrap(), 3),
            (NaiveDate::from_ymd_opt(2023, 3, 10).unwrap(), 4),
        ],
        vec![
            (NaiveDate::from_ymd_opt(2023, 1, 1).unwrap(), 4),
            (NaiveDate::from_ymd_opt(2023, 1, 12).unwrap(), 3),
            (NaiveDate::from_ymd_opt(2023, 1, 25).unwrap(), 4),
            (NaiveDate::from_ymd_opt(2023, 2, 10).unwrap(), 3),
            (NaiveDate::from_ymd_opt(2023, 3, 1).unwrap(), 4),
        ],
    ]
    .iter()
    .cloned()
    .cycle()
    .take(100)
    .collect()
}

fn convert_to_fsrs_item(history: Vec<(NaiveDate, u32)>) -> Vec<FSRSItem> {
    let mut reviews = Vec::new();
    let mut last_date = history[0].0;
    let mut items = Vec::new();

    for (date, rating) in history {
        let delta_t = (date - last_date).num_days() as u32;
        reviews.push(FSRSReview { rating, delta_t });
        items.push(FSRSItem {
            reviews: reviews.clone(),
        });
        last_date = date;
    }

    items
        .into_iter()
        .filter(|item| item.long_term_review_cnt() > 0)
        .collect()
}

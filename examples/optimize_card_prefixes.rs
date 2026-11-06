use fsrs::{
    CardPrefix, CardPrefixes, CardReviews, ComputeParametersInput, ComputeParametersVersion,
    FSRSItem, FSRSReview, compute_parameters, compute_parameters_from_card_prefixes,
};

const USAGE: &str = "\
Usage: cargo run --release --example optimize_card_prefixes -- [card_prefixes|items] [cards] [reviews] [fsrs6|fsrs7]

`items` builds the same training set as `Vec<FSRSItem>`. Both print the same parameters.
`reviews` must be at least 1.
`fsrs7` is the default model; use `fsrs6` to compare FSRS-6 training.
To compare their peak memory, run the built binary under `/usr/bin/time -l` (Linux: `-v`).
";

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let form = args.next().unwrap_or_else(|| "card_prefixes".to_string());
    let cards = args.next().map_or(Ok(1000), |cards| cards.parse())?;
    let reviews = args.next().map_or(Ok(256), |reviews| reviews.parse())?;
    let model_version = match args.next().as_deref() {
        None | Some("fsrs7") => ComputeParametersVersion::Fsrs7,
        Some("fsrs6") => ComputeParametersVersion::Fsrs6,
        _ => {
            eprint!("{USAGE}");
            return Err("invalid model".into());
        }
    };
    let use_items = form == "items";
    if !(use_items || form == "card_prefixes") || reviews == 0 {
        eprint!("{USAGE}");
        return Err("invalid arguments".into());
    }

    let card_prefixes = to_card_prefixes(create_review_histories_for_cards(cards, reviews));
    println!("Number of items: {}", card_prefixes.prefixes.len());

    let input = ComputeParametersInput {
        model_version,
        ..Default::default()
    };
    let optimized_parameters = if use_items {
        let (train_set, card_ids) = to_items(card_prefixes);
        compute_parameters(ComputeParametersInput {
            train_set,
            card_ids: Some(card_ids),
            ..input
        })?
    } else {
        compute_parameters_from_card_prefixes(&card_prefixes, input)?
    };
    println!("Optimized parameters: {optimized_parameters:?}");

    Ok(())
}

fn create_review_histories_for_cards(cards: u32, reviews: u32) -> Vec<Vec<(u32, u32)>> {
    // Each card's reviews as (day, rating), with ratings 1: Again, 2: Hard, 3: Good, 4: Easy.
    (0..cards)
        .map(|card| {
            (0..reviews)
                .map(|review| {
                    let rating = if (card * 7 + review * 13) % 5 == 0 {
                        1
                    } else {
                        3
                    };
                    (card % 30 + review * 2, rating)
                })
                .collect()
        })
        .collect()
}

fn to_card_prefixes(histories: Vec<Vec<(u32, u32)>>) -> CardPrefixes {
    // Each review after a card's first makes one item. Items are sorted oldest first, because
    // recent items weigh more in training.
    let mut cards = Vec::with_capacity(histories.len());
    let mut prefixes = Vec::new();
    for (card_index, history) in (0..).zip(histories) {
        let mut last_day = history[0].0;
        let mut reviews = Vec::with_capacity(history.len());
        for (len, (day, rating)) in (1..).zip(history) {
            reviews.push(FSRSReview {
                rating,
                delta_t: (day - last_day) as f32,
            });
            last_day = day;
            if len > 1 {
                prefixes.push((day, CardPrefix { card_index, len }));
            }
        }
        cards.push(CardReviews {
            card_id: i64::from(card_index),
            reviews,
        });
    }
    prefixes.sort_by_key(|prefix| prefix.0);
    CardPrefixes {
        cards,
        prefixes: prefixes.into_iter().map(|prefix| prefix.1).collect(),
    }
}

fn to_items(card_prefixes: CardPrefixes) -> (Vec<FSRSItem>, Vec<i64>) {
    // The same items as `FSRSItem`s, each with its own copy of the reviews.
    card_prefixes
        .prefixes
        .iter()
        .map(|prefix| {
            let card = &card_prefixes.cards[prefix.card_index as usize];
            let reviews = card.reviews[..prefix.len as usize].to_vec();
            (FSRSItem { reviews }, card.card_id)
        })
        .unzip()
}

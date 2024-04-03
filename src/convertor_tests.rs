use crate::convertor_tests::RevlogReviewKind::*;
use crate::dataset::{FSRSItem, FSRSReview};
use chrono::prelude::*;
use chrono_tz::Tz;
use itertools::Itertools;

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct RevlogEntry {
    pub id: i64,
    pub cid: i64,
    // pub usn: i32,
    /// - In the V1 scheduler, 3 represents easy in the learning case.
    /// - 0 represents manual rescheduling.
    pub button_chosen: u8,
    /// Positive values are in days, negative values in seconds.
    // pub interval: i32,
    /// Positive values are in days, negative values in seconds.
    pub last_interval: i32,
    /// Card's ease after answering, stored as 10x the %, eg 2500 represents
    /// 250%.
    // pub ease_factor: u32,
    /// Amount of milliseconds taken to answer the card.
    // pub taken_millis: u32,
    pub review_kind: RevlogReviewKind,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord)]
pub enum RevlogReviewKind {
    #[default]
    Learning = 0,
    Review = 1,
    Relearning = 2,
    /// Old Anki versions called this "Cram" or "Early", and assigned it when
    /// reviewing cards ahead. It is now only used for filtered decks with
    /// rescheduling disabled.
    Filtered = 3,
    Manual = 4,
}

// fn filter_out_cram(entries: Vec<RevlogEntry>) -> Vec<RevlogEntry> {
//     entries
//         .into_iter()
//         .filter(|entry| entry.review_kind != Filtered || entry.ease_factor != 0)
//         .collect()
// }
//
// fn filter_out_manual(entries: Vec<RevlogEntry>) -> Vec<RevlogEntry> {
//     entries
//         .into_iter()
//         .filter(|entry| entry.review_kind != Manual && entry.button_chosen != 0)
//         .collect()
// }

fn remove_revlog_before_last_first_learn(entries: Vec<RevlogEntry>) -> Vec<RevlogEntry> {
    let mut last_first_learn_index = 0;
    for (index, entry) in entries.iter().enumerate().rev() {
        if entry.review_kind == Learning {
            last_first_learn_index = index;
        } else if last_first_learn_index != 0 {
            break;
        }
    }
    if entries[last_first_learn_index].review_kind == Learning {
        entries[last_first_learn_index..].to_vec()
    } else {
        vec![]
    }
}

fn convert_to_date(timestamp: i64, next_day_starts_at: i64, timezone: Tz) -> NaiveDate {
    let timestamp_seconds = timestamp - next_day_starts_at * 3600 * 1000;
    let datetime = Utc
        .timestamp_millis_opt(timestamp_seconds)
        .unwrap()
        .with_timezone(&timezone);
    datetime.date_naive()
}

fn keep_first_revlog_same_date(
    mut entries: Vec<RevlogEntry>,
    next_day_starts_at: i64,
    timezone: Tz,
) -> Vec<RevlogEntry> {
    let mut unique_dates = std::collections::HashSet::new();
    entries.retain(|entry| {
        let date = convert_to_date(entry.id, next_day_starts_at, timezone);
        unique_dates.insert(date)
    });
    entries
}

/// Given a list of revlog entries for a single card with length n, we create
/// n-1 FSRS items, where each item contains the history of the preceding reviews.

fn convert_to_fsrs_items(
    mut entries: Vec<RevlogEntry>,
    next_day_starts_at: i64,
    timezone: Tz,
) -> Option<Vec<FSRSItem>> {
    // entries = filter_out_cram(entries);
    // entries = filter_out_manual(entries);
    entries = remove_revlog_before_last_first_learn(entries);
    entries = keep_first_revlog_same_date(entries, next_day_starts_at, timezone);

    for i in 1..entries.len() {
        let date_current = convert_to_date(entries[i].id, next_day_starts_at, timezone);
        let date_previous = convert_to_date(entries[i - 1].id, next_day_starts_at, timezone);
        entries[i].last_interval = (date_current - date_previous).num_days() as i32;
    }

    Some(
        entries
            .iter()
            .enumerate()
            .skip(1)
            .map(|(idx, _)| {
                let reviews = entries
                    .iter()
                    .take(idx + 1)
                    .map(|r| FSRSReview {
                        rating: r.button_chosen as u32,
                        delta_t: r.last_interval as u32,
                    })
                    .collect();
                FSRSItem { reviews }
            })
            .collect(),
    )
}

pub fn to_revlog_entry(
    cids: &[i64],
    eases: &[u8],
    // factors: &[u32],
    ids: &[i64],
    // ivls: &[i32],
    // last_ivls: &[i32],
    // times: &[u32],
    types: &[u8],
    // usns: &[i32],
) -> Vec<RevlogEntry> {
    ids.iter()
        .enumerate()
        .map(|(i, _id)| RevlogEntry {
            id: ids[i],
            cid: cids[i],
            // usn: usns[i],
            button_chosen: eases[i],
            // interval: ivls[i],
            last_interval: 0,
            // ease_factor: factors[i],
            // taken_millis: times[i],
            review_kind: types[i].into(),
        })
        .collect()
}

impl From<u8> for RevlogReviewKind {
    fn from(val: u8) -> Self {
        match val {
            0 => Ok(RevlogReviewKind::Learning),
            1 => Ok(RevlogReviewKind::Review),
            2 => Ok(RevlogReviewKind::Relearning),
            3 => Ok(RevlogReviewKind::Filtered),
            4 => Ok(RevlogReviewKind::Manual),
            _ => Err(format!("Unable to convert {val} into a RevlogReviewKind.")),
        }
        .unwrap()
    }
}

/// Convert a series of revlog entries sorted by card id into FSRS items.
pub fn anki_to_fsrs(revlogs: Vec<RevlogEntry>) -> Vec<FSRSItem> {
    let mut revlogs = revlogs
        .into_iter()
        .group_by(|r| r.cid)
        .into_iter()
        .filter_map(|(_cid, entries)| {
            convert_to_fsrs_items(entries.collect(), 4, Tz::Asia__Shanghai)
        })
        .flatten()
        .collect_vec();
    revlogs.sort_by_cached_key(|r| r.reviews.len());
    revlogs
}

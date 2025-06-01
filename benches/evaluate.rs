use criterion::{Criterion, criterion_group, criterion_main};
use fsrs::{
    // dataset::prepare_training_data, // Will be inlined
    // convertor_tests::anki21_sample_file_converted_to_fsrs, // Will be inlined
    ComputeParametersInput,
    DEFAULT_PARAMETERS,
    FSRS,
    FSRSItem,
    FSRSReview,
};
// Add necessary imports for inlined code
use chrono::prelude::*;
use chrono_tz::Tz;
use itertools::Itertools;
use rusqlite::{Connection, Result as SqlResult, Row, types::FromSqlError as SqlFromSqlError};
use std::hint::black_box;

// Inlined RevlogReviewKind enum from convertor_tests.rs
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
enum RevlogReviewKind {
    Learning,
    Review,
    #[default]
    Relearning,
    Filtered, // Anki calls this "cram"
    Manual,
}

impl rusqlite::types::FromSql for RevlogReviewKind {
    fn column_result(value: rusqlite::types::ValueRef<'_>) -> rusqlite::types::FromSqlResult<Self> {
        match value {
            rusqlite::types::ValueRef::Integer(i) => match i {
                0 => Ok(RevlogReviewKind::Learning),
                1 => Ok(RevlogReviewKind::Review),
                2 => Ok(RevlogReviewKind::Relearning),
                3 => Ok(RevlogReviewKind::Filtered),
                4 => Ok(RevlogReviewKind::Manual),
                _ => Err(SqlFromSqlError::InvalidType),
            },
            _ => Err(SqlFromSqlError::InvalidType),
        }
    }
}

// Inlined RevlogEntry struct from convertor_tests.rs
#[derive(Debug, Clone, Default, PartialEq, Eq)]
struct RevlogEntry {
    id: i64,
    cid: i64,
    usn: i32,
    button_chosen: u8,
    interval: i32,
    last_interval: i32,
    ease_factor: i32,
    taken_millis: i32,
    review_kind: RevlogReviewKind,
}

impl TryFrom<&Row<'_>> for RevlogEntry {
    type Error = rusqlite::Error;
    fn try_from(row: &Row<'_>) -> SqlResult<Self> {
        Ok(RevlogEntry {
            id: row.get(0)?,
            cid: row.get(1)?,
            usn: row.get(2)?,
            button_chosen: row.get(3)?,
            interval: row.get(4)?,
            last_interval: row.get(5)?,
            ease_factor: row.get(6)?,
            taken_millis: row.get(7)?,
            review_kind: row.get(8)?,
        })
    }
}

// Inlined helper functions from convertor_tests.rs
fn convert_to_date(timestamp: i64, next_day_starts_at: i64, timezone: Tz) -> NaiveDate {
    let timestamp_seconds = timestamp - next_day_starts_at * 3600 * 1000;
    Utc.timestamp_millis_opt(timestamp_seconds)
        .unwrap()
        .with_timezone(&timezone)
        .date_naive()
}

fn remove_revlog_before_last_first_learn(entries: Vec<RevlogEntry>) -> Vec<RevlogEntry> {
    let mut last_first_learn_index = 0;
    for (index, entry) in entries.iter().enumerate().rev() {
        if entry.review_kind == RevlogReviewKind::Learning {
            last_first_learn_index = index;
        } else if last_first_learn_index != 0 {
            break;
        }
    }
    if !entries.is_empty()
        && entries[last_first_learn_index].review_kind == RevlogReviewKind::Learning
    {
        entries[last_first_learn_index..].to_vec()
    } else {
        vec![]
    }
}

fn convert_to_fsrs_items(
    mut entries: Vec<RevlogEntry>,
    next_day_starts_at: i64,
    timezone: Tz,
) -> Option<Vec<(i64, FSRSItem)>> {
    entries = remove_revlog_before_last_first_learn(entries);
    if entries.is_empty() {
        return None;
    }

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
            .map(|(idx, entry)| {
                let reviews = entries
                    .iter()
                    .take(idx + 1)
                    .map(|r| FSRSReview {
                        rating: r.button_chosen as u32,
                        delta_t: r.last_interval.max(0) as u32,
                    })
                    .collect();
                (entry.id, FSRSItem { reviews })
            })
            .filter(|(_, item)| item.reviews.last().map_or(false, |r| r.delta_t > 0)) // Ensure last review has delta_t > 0
            .collect(),
    )
}

fn anki_to_fsrs(revlogs: Vec<RevlogEntry>) -> Vec<FSRSItem> {
    let mut revlogs_by_card = revlogs
        .into_iter()
        .chunk_by(|r| r.cid)
        .into_iter()
        .filter_map(|(_cid, entries)| {
            convert_to_fsrs_items(entries.collect(), 4, Tz::Asia__Shanghai)
        })
        .flatten()
        .collect_vec();
    revlogs_by_card.sort_by_cached_key(|(id, _)| *id);
    revlogs_by_card.into_iter().map(|(_, item)| item).collect()
}

fn read_collection_inline() -> SqlResult<Vec<RevlogEntry>> {
    let db = Connection::open("tests/data/collection.anki21")?;
    let filter_out_suspended_cards = false;
    let filter_out_flags: [i32; 0] = [];
    let flags_str = if !filter_out_flags.is_empty() {
        format!(
            "AND flags NOT IN ({})",
            filter_out_flags
                .iter()
                .map(|x| x.to_string())
                .collect::<Vec<_>>()
                .join(", ")
        )
    } else {
        "".to_string()
    };
    let suspended_cards_str = if filter_out_suspended_cards {
        "AND queue != -1"
    } else {
        ""
    };
    let current_timestamp = Utc::now().timestamp() * 1000;
    db.prepare_cached(&format!(
        "SELECT * FROM revlog WHERE id < ?1 AND cid < ?2 AND cid IN (SELECT id FROM cards WHERE queue != 0 {suspended_cards_str} {flags_str}) AND ease BETWEEN 1 AND 4 AND (type != 3 OR factor != 0) ORDER BY cid"
    ))?
    .query_and_then((current_timestamp, current_timestamp), |row| row.try_into())?
    .collect::<SqlResult<Vec<_>>>()
}

// Inlined anki21_sample_file_converted_to_fsrs
fn anki21_sample_file_converted_to_fsrs_inline() -> Vec<FSRSItem> {
    anki_to_fsrs(read_collection_inline().expect("read error for inlined function"))
}

// Inlined prepare_training_data (simplified version matching fsrs::dataset::prepare_training_data)
fn prepare_training_data_inline(items: Vec<FSRSItem>) -> (Vec<FSRSItem>, Vec<FSRSItem>) {
    let filtered_items: Vec<FSRSItem> = items
        .into_iter()
        .filter(|item| {
            !item.reviews.is_empty() && item.reviews.len() > 1 && item.reviews[0].delta_t == 0
        })
        .collect();

    if filtered_items.is_empty() {
        return (vec![], vec![]);
    }
    let n_pretrain = (filtered_items.len() as f32 * 0.1).ceil() as usize;
    let n_pretrain = n_pretrain.min(filtered_items.len());
    let (pretrain_part, train_part) = filtered_items.split_at(n_pretrain);
    (pretrain_part.to_vec(), train_part.to_vec())
}

fn load_and_prepare_data() -> Vec<FSRSItem> {
    let items = anki21_sample_file_converted_to_fsrs_inline();
    let (pretrain_set, train_set) = prepare_training_data_inline(items);
    [pretrain_set, train_set].concat()
}

fn benchmark_evaluate(c: &mut Criterion) {
    let items = load_and_prepare_data();
    // Evaluate uses the FSRS instance's existing parameters.
    let fsrs = FSRS::new(Some(&DEFAULT_PARAMETERS)).unwrap();

    let mut group = c.benchmark_group("evaluation");
    group.sample_size(10); // Reduce sample size if benchmarks are too long

    group.bench_function("evaluate", |b| {
        b.iter(|| {
            fsrs.evaluate(black_box(items.clone()), |_| true).unwrap();
        })
    });

    group.finish();
}

fn benchmark_evaluate_with_time_series_splits(c: &mut Criterion) {
    let items = load_and_prepare_data();
    // evaluate_with_time_series_splits computes parameters internally for each split.
    let fsrs = FSRS::new(None).unwrap();
    let input = ComputeParametersInput {
        train_set: items.clone(),
        progress: None,
        enable_short_term: true,    // Default/typical value
        num_relearning_steps: None, // Default/typical value
    };

    let mut group = c.benchmark_group("evaluation");
    group.sample_size(10); // Reduce sample size as this involves training

    group.bench_function("evaluate_with_time_series_splits", |b| {
        b.iter(|| {
            fsrs.evaluate_with_time_series_splits(black_box(input.clone()), |_| true)
                .unwrap();
        })
    });
    group.finish();
}

criterion_group!(
    benches,
    benchmark_evaluate,
    benchmark_evaluate_with_time_series_splits
);
criterion_main!(benches);

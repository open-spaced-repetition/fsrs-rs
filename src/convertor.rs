use chrono::prelude::*;
use chrono_tz::Tz;
use rusqlite::{Connection, Result, Row};
use std::collections::HashMap;

use crate::dataset::{FSRSItem, Review};

#[derive(Debug, Clone)]
struct RevlogEntry {
    id: i64,
    cid: i64,
    button_chosen: i32,
    review_kind: i64,
    delta_t: i32,
    i: usize,
    r_history: Vec<i32>,
    t_history: Vec<i32>,
}

fn row_to_revlog_entry(row: &Row) -> Result<RevlogEntry> {
    Ok(RevlogEntry {
        id: row.get(0)?,
        cid: row.get(1)?,
        button_chosen: row.get(2)?,
        review_kind: row.get(3).unwrap_or_default(),
        delta_t: 0,
        i: 0,
        r_history: vec![],
        t_history: vec![],
    })
}

fn read_collection() -> Vec<RevlogEntry> {
    let db = Connection::open("tests/data/collection.anki21").unwrap();
    let filter_out_suspended_cards = false;
    let filter_out_flags = vec![];
    let flags_str = if !filter_out_flags.is_empty() {
        format!(
            "AND flags NOT IN ({})",
            filter_out_flags
                .iter()
                .map(|x: &i32| x.to_string())
                .collect::<Vec<String>>()
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

    let query = format!(
        "SELECT id, cid, ease, type
         FROM revlog 
         WHERE (type != 4 OR ivl <= 0)
         AND (factor != 0 or type != 3)
         AND id < {}
         AND cid < {}
         AND cid IN (
             SELECT id
             FROM cards
             WHERE queue != 0
             {}
             {}
         )",
        current_timestamp, current_timestamp, suspended_cards_str, flags_str
    );

    let revlogs = db
        .prepare_cached(&query)
        .unwrap()
        .query_and_then([], row_to_revlog_entry)
        .unwrap()
        .collect::<Result<Vec<RevlogEntry>>>()
        .unwrap();
    revlogs
}

fn group_by_cid(revlogs: Vec<RevlogEntry>) -> Vec<Vec<RevlogEntry>> {
    let mut grouped: HashMap<i64, Vec<RevlogEntry>> = HashMap::new();
    for revlog in revlogs {
        grouped
            .entry(revlog.cid)
            .or_insert_with(Vec::new)
            .push(revlog);
    }

    grouped.into_values().collect()
}

fn convert_to_date(timestamp: i64, next_day_starts_at: i64, timezone: Tz) -> chrono::NaiveDate {
    let timestamp_seconds = timestamp - next_day_starts_at * 3600 * 1000; // 剪去指定小时数
    let datetime = Utc
        .timestamp_millis_opt(timestamp_seconds)
        .unwrap()
        .with_timezone(&timezone);
    datetime.date_naive()
}

fn extract_time_series_feature(
    mut entries: Vec<RevlogEntry>,
    next_day_starts_at: i64,
    timezone: Tz,
) -> Option<Vec<RevlogEntry>> {
    // Find the index of the first RevlogEntry in the last continuous group where review_kind = 0
    // 寻找最后一组连续 review_kind = 0 的第一个 RevlogEntry 的索引
    let mut index_to_keep = 0;
    let mut i = entries.len();

    while i > 0 {
        i -= 1;
        if entries[i].review_kind == 0 {
            index_to_keep = i;
        } else if index_to_keep != 0 {
            // Found a continuous group of review_kind = 0, exit the loop
            // 找到了连续的 review_kind = 0 的组，退出循环
            break;
        }
    }

    // Remove all entries before this RevlogEntry
    // 删除此 RevlogEntry 之前的所有条目
    entries.drain(..index_to_keep);

    // Increment review_kind of all entries by 1
    // 将所有 review_kind + 1
    for entry in &mut entries {
        entry.review_kind += 1;
    }

    // Convert the timestamp and keep the first RevlogEntry for each date
    // 转换时间戳并保留每个日期的第一个 RevlogEntry
    let mut unique_dates = std::collections::HashSet::new();
    entries.retain(|entry| {
        let date = convert_to_date(entry.id, next_day_starts_at, timezone);
        unique_dates.insert(date)
    });

    // Compute delta_t for the remaining RevlogEntries
    // 计算其余 RevlogEntry 的 delta_t
    for i in 1..entries.len() {
        let date_current = convert_to_date(entries[i].id, next_day_starts_at, timezone);
        let date_previous = convert_to_date(entries[i - 1].id, next_day_starts_at, timezone);
        entries[i].delta_t = (date_current - date_previous).num_days() as i32;
    }

    // Compute i, r_history, t_history
    // 计算 i, r_history, t_history
    for i in 0..entries.len() {
        // Position starts from 1
        // 位置从 1 开始
        entries[i].i = i + 1;

        // Except for the first entry, the remaining entries add the preceding button_chosen and delta_t to r_history and t_history
        // 除了第一个条目，其余条目将前面的 button_chosen 和 delta_t 加入 r_history 和 t_history
        if i > 0 {
            entries[i].r_history = entries[0..i].iter().map(|e| e.button_chosen).collect();
            entries[i].t_history = entries[0..i].iter().map(|e| e.delta_t).collect();
        }
    }

    // Find the RevlogEntry with review_kind = 0 where the preceding RevlogEntry has review_kind of 1 or 2, then remove it and all following RevlogEntries
    // 找到 review_kind = 0 且前一个 RevlogEntry 的 review_kind 是 1 或 2 的 RevlogEntry，然后删除其及其之后的所有 RevlogEntry
    if let Some(index_to_remove) = entries.windows(2).enumerate().find_map(|(i, window)| {
        if (window[0].review_kind == 1 || window[0].review_kind == 2) && window[1].review_kind == 0
        {
            // Return the index of the first RevlogEntry that meets the condition
            // 返回第一个符合条件的 RevlogEntry 的索引
            Some(i + 1)
        } else {
            None
        }
    }) {
        // Truncate from 0 to index_to_remove, removing all subsequent entries
        // 截取从 0 到 index_to_remove 的部分，删除其后的所有条目
        entries.truncate(index_to_remove);
    }

    // we ignore cards that don't start in the learning state
    if let Some(first) = entries.first() {
        if first.review_kind == 1 {
            return Some(entries);
        }
    }
    None
}

fn convert_to_fsrs_items(revlogs: Vec<Vec<RevlogEntry>>) -> Vec<FSRSItem> {
    revlogs
        .into_iter()
        .flat_map(|group| {
            group
                .into_iter()
                .filter(|entry| entry.i != 1) // 过滤掉 i = 1 的 RevlogEntry
                .map(|entry| FSRSItem {
                    reviews: entry
                        .r_history
                        .iter()
                        .zip(entry.t_history.iter())
                        .map(|(&r, &t)| Review {
                            rating: r,
                            delta_t: t,
                        })
                        .collect(),
                    delta_t: entry.delta_t as f32,
                    label: match entry.button_chosen {
                        1 => 0.0,
                        2 | 3 | 4 => 1.0,
                        _ => panic!("Unexpected value for button_chosen"),
                    },
                })
        })
        .collect()
}

pub fn anki_to_fsrs() -> Vec<FSRSItem> {
    let revlogs = read_collection();
    let revlogs_per_card = group_by_cid(revlogs);
    let extracted_revlogs_per_card: Vec<Vec<RevlogEntry>> = revlogs_per_card
        .into_iter()
        .filter_map(|entries| extract_time_series_feature(entries, 4, Tz::Asia__Shanghai))
        .collect();

    convert_to_fsrs_items(extracted_revlogs_per_card)
}

#[cfg(test)]
mod tests {
    use super::*;

    // This test currently expects the following .anki21 file to be placed in tests/data/:
    // https://github.com/open-spaced-repetition/fsrs-optimizer-burn/files/12394182/collection.anki21.zip
    #[test]
    fn test() {
        let revlogs = read_collection();
        let single_card_revlog = vec![revlogs
            .iter()
            .filter(|r| r.cid == 1528947214762)
            .cloned()
            .collect::<Vec<_>>()];
        assert_eq!(revlogs.len(), 24394);
        let revlogs_per_card = group_by_cid(revlogs.clone());
        assert_eq!(revlogs_per_card.len(), 3324);
        let extracted_revlogs_per_card: Vec<Vec<RevlogEntry>> = revlogs_per_card
            .into_iter()
            .filter_map(|entries| extract_time_series_feature(entries, 4, Tz::Asia__Shanghai))
            .collect();

        assert_eq!(
            extracted_revlogs_per_card
                .iter()
                .map(|x| x.len())
                .sum::<usize>(),
            17614
        );
        let fsrs_items: Vec<FSRSItem> = convert_to_fsrs_items(extracted_revlogs_per_card);
        assert_eq!(fsrs_items.len(), 14290);
        assert_eq!(
            fsrs_items.iter().map(|x| x.reviews.len()).sum::<usize>(),
            49382
        );

        // convert a subset and check it matches expectations
        let extracted_revlogs_per_card: Vec<Vec<RevlogEntry>> = single_card_revlog
            .into_iter()
            .filter_map(|entries| extract_time_series_feature(entries, 4, Tz::Asia__Shanghai))
            .collect();
        let fsrs_items: Vec<FSRSItem> = convert_to_fsrs_items(extracted_revlogs_per_card);
        assert_eq!(
            &fsrs_items,
            &[
                FSRSItem {
                    reviews: vec![Review {
                        rating: 3,
                        delta_t: 0
                    }],
                    delta_t: 5.0,
                    label: 1.0
                },
                FSRSItem {
                    reviews: vec![
                        Review {
                            rating: 3,
                            delta_t: 0
                        },
                        Review {
                            rating: 3,
                            delta_t: 5
                        }
                    ],
                    delta_t: 10.0,
                    label: 1.0
                },
                FSRSItem {
                    reviews: vec![
                        Review {
                            rating: 3,
                            delta_t: 0
                        },
                        Review {
                            rating: 3,
                            delta_t: 5
                        },
                        Review {
                            rating: 3,
                            delta_t: 10
                        }
                    ],
                    delta_t: 22.0,
                    label: 1.0
                },
                FSRSItem {
                    reviews: vec![
                        Review {
                            rating: 3,
                            delta_t: 0
                        },
                        Review {
                            rating: 3,
                            delta_t: 5
                        },
                        Review {
                            rating: 3,
                            delta_t: 10
                        },
                        Review {
                            rating: 3,
                            delta_t: 22
                        }
                    ],
                    delta_t: 56.0,
                    label: 1.0
                },
                FSRSItem {
                    reviews: vec![
                        Review {
                            rating: 3,
                            delta_t: 0
                        },
                        Review {
                            rating: 3,
                            delta_t: 5
                        },
                        Review {
                            rating: 3,
                            delta_t: 10
                        },
                        Review {
                            rating: 3,
                            delta_t: 22
                        },
                        Review {
                            rating: 2,
                            delta_t: 56
                        }
                    ],
                    delta_t: 64.0,
                    label: 1.0
                }
            ]
        );
    }
}

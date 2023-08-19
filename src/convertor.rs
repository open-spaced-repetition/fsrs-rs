use chrono::prelude::*;
use chrono_tz::Tz;
use rusqlite::{Connection, Result, Row};
use std::collections::HashMap;

use crate::dataset::FSRSItem;

#[derive(Debug)]
struct RevlogEntry {
    id: i64,
    cid: i64,
    usn: i64,
    button_chosen: i32,
    interval: i64,
    last_interval: i64,
    ease_factor: i64,
    taken_millis: i64,
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
        usn: row.get(2)?,
        button_chosen: row.get(3)?,
        interval: row.get(4)?,
        last_interval: row.get(5)?,
        ease_factor: row.get(6)?,
        taken_millis: row.get(7).unwrap_or_default(),
        review_kind: row.get(8).unwrap_or_default(),
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
        "SELECT * 
         FROM revlog 
         WHERE (type != 4 OR ivl <= 0)
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

    grouped.into_iter().map(|(_, v)| v).collect()
}

fn convert_to_date(timestamp: i64, next_day_starts_at: i64, timezone: Tz) -> chrono::NaiveDate {
    let timestamp_seconds = timestamp - next_day_starts_at * 3600 * 1000; // 剪去指定小时数
    let datetime = Utc.timestamp_millis_opt(timestamp_seconds).unwrap().with_timezone(&timezone);
    datetime.date_naive()
}

fn extract_time_series_feature(mut entries: Vec<RevlogEntry>, next_day_starts_at: i64, timezone: Tz) -> Vec<RevlogEntry> {
    // 寻找最后一组连续 review_kind = 0 的第一个 RevlogEntry 的索引
    let mut index_to_keep = 0;
    let mut i = entries.len();

    while i > 0 {
        i -= 1;
        if entries[i].review_kind == 0 {
            index_to_keep = i;
        } else if index_to_keep != 0 {
            break; // 找到了连续的 review_kind = 0 的组，退出循环
        }
    }

    // 删除此 RevlogEntry 之前的所有条目
    entries.drain(..index_to_keep);

    // 去掉 review_kind = 4 的 RevlogEntry
    entries.retain(|entry| entry.review_kind != 4);

    // 去掉 review_kind = 3 且 ease_factor = 0 的 RevlogEntry
    entries.retain(|entry| entry.review_kind != 3 || entry.ease_factor != 0);

    // 将所有 review_kind + 1
    for entry in &mut entries {
        entry.review_kind += 1;
    }

    // 转换时间戳并保留每个日期的第一个 RevlogEntry
    let mut unique_dates = std::collections::HashSet::new();
    entries.retain(|entry| {
        let date = convert_to_date(entry.id, next_day_starts_at, timezone);
        unique_dates.insert(date)
    });

    // 计算其余 RevlogEntry 的 delta_t
    for i in 1..entries.len() {
        let date_current = convert_to_date(entries[i].id, next_day_starts_at, timezone);
        let date_previous = convert_to_date(entries[i - 1].id, next_day_starts_at, timezone);
        entries[i].delta_t = (date_current - date_previous).num_days() as i32;
    }

    // 计算 i, r_history, t_history
    for i in 0..entries.len() {
        entries[i].i = i + 1; // 位置从 1 开始
        // 除了第一个条目，其余条目将前面的 button_chosen 和 delta_t 加入 r_history 和 t_history
        if i > 0 {
            entries[i].r_history = entries[0..i].iter().map(|e| e.button_chosen).collect();
            entries[i].t_history = entries[0..i].iter().map(|e| e.delta_t).collect();
        }
    }

    // 找到 review_kind = 0 且前一个 RevlogEntry 的 review_kind 是 1 或 2 的 RevlogEntry，然后删除其及其之后的所有 RevlogEntry
    if let Some(index_to_remove) = entries.windows(2).enumerate().find_map(|(i, window)| {
        if (window[0].review_kind == 1 || window[0].review_kind == 2) && window[1].review_kind == 0 {
            Some(i + 1) // 返回第一个符合条件的 RevlogEntry 的索引
        } else {
            None
        }
    }) {
        entries.truncate(index_to_remove); // 截取从 0 到 index_to_remove 的部分，删除其后的所有条目
    }

    entries
}


fn convert_to_fsrs_items(revlogs: Vec<Vec<RevlogEntry>>) -> Vec<FSRSItem> {
    revlogs.into_iter().flat_map(|group| {
        group.into_iter()
        .filter(|entry| entry.i != 1) // 过滤掉 i = 1 的 RevlogEntry
        .map(|entry| {
            FSRSItem {
                t_history: entry.t_history,
                r_history: entry.r_history,
                delta_t: entry.delta_t as f32,
                label: match entry.button_chosen {
                    1 => 0.0,
                    2 | 3 | 4 => 1.0,
                    _ => panic!("Unexpected value for button_chosen"),
                },
            }
        })
    }).collect()
}

fn remove_non_learning_first(revlogs_per_card: Vec<Vec<RevlogEntry>>) -> Vec<Vec<RevlogEntry>> {
    let mut result = revlogs_per_card;
    result.retain(|entries| {
        if let Some(first_entry) = entries.first() {
            first_entry.review_kind == 1
        } else {
            false
        }
    });
    result
}

pub fn anki_to_fsrs() -> Vec<FSRSItem> {
    let revlogs = read_collection();
    let revlogs_per_card = group_by_cid(revlogs);
    let extracted_revlogs_per_card: Vec<Vec<RevlogEntry>> = revlogs_per_card
        .into_iter()
        .map(|entries| extract_time_series_feature(entries, 4, Tz::Asia__Shanghai))
        .collect();

    let filtered_revlogs_per_card = remove_non_learning_first(extracted_revlogs_per_card);
    let fsrs_items = convert_to_fsrs_items(filtered_revlogs_per_card);
    fsrs_items
}

#[test]
fn test() {
    let revlogs = read_collection();
    dbg!(revlogs.len());
    let revlogs_per_card = group_by_cid(revlogs);
    dbg!(revlogs_per_card.len());
    let mut extracted_revlogs_per_card: Vec<Vec<RevlogEntry>> = revlogs_per_card
        .into_iter()
        .map(|entries| extract_time_series_feature(entries, 4, Tz::Asia__Shanghai))
        .collect();

    dbg!(&extracted_revlogs_per_card[0]);
    extracted_revlogs_per_card = remove_non_learning_first(extracted_revlogs_per_card);
    dbg!(extracted_revlogs_per_card.iter().map(|x| x.len()).sum::<usize>());
    let fsrs_items: Vec<FSRSItem> = convert_to_fsrs_items(extracted_revlogs_per_card);
    dbg!(fsrs_items.len());
}

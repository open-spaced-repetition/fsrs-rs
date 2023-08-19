use chrono::prelude::*;
use rusqlite::{Connection, Result, Row};
use std::collections::HashMap;

#[derive(Debug)]
struct RevlogEntry {
    id: i64,
    cid: i64,
    usn: i64,
    button_chosen: i64,
    interval: i64,
    last_interval: i64,
    ease_factor: i64,
    taken_millis: i64,
    review_kind: i64,
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

    dbg!(&query);

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

fn remove_revlog_before_forget(mut entries: Vec<RevlogEntry>) -> Vec<RevlogEntry> {
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

    entries
}

#[test]
fn test() {
    let revlogs = read_collection();
    dbg!(revlogs.len());
    let revlogs = group_by_cid(revlogs);
    dbg!(revlogs.len());
    // for r in revlogs {
    //     dbg!(&r);
    //     break;
    // }
    let filtered_entries: Vec<Vec<RevlogEntry>> = revlogs
        .into_iter()
        .map(remove_revlog_before_forget)
        .collect();
    // total revlogs
    dbg!(filtered_entries.iter().map(|x| x.len()).sum::<usize>());
}

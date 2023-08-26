use chrono::prelude::*;
use chrono_tz::Tz;
use itertools::Itertools;
use rusqlite::{Connection, Result, Row};

use crate::dataset::{FSRSItem, FSRSReview};

#[derive(Clone, Debug, Default)]
pub(crate) struct RevlogEntry {
    id: i64,
    cid: i64,
    button_chosen: i32,
    review_kind: i64,
    delta_t: i32,
}

fn row_to_revlog_entry(row: &Row) -> Result<RevlogEntry> {
    Ok(RevlogEntry {
        id: row.get(0)?,
        cid: row.get(1)?,
        button_chosen: row.get(2)?,
        review_kind: row.get(3).unwrap_or_default(),
        delta_t: 0,
    })
}

fn read_collection() -> Result<Vec<RevlogEntry>> {
    let db = Connection::open("tests/data/collection.anki21")?;
    let filter_out_suspended_cards = false;
    let filter_out_flags = [];
    let flags_str = if !filter_out_flags.is_empty() {
        format!(
            "AND flags NOT IN ({})",
            filter_out_flags
                .iter()
                .map(|x: &i32| x.to_string())
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
    // This sql query will be remove in the futrue. See https://github.com/open-spaced-repetition/fsrs-optimizer-burn/pull/14#issuecomment-1685895643
    let revlogs = db
        .prepare_cached(&format!(
            "SELECT id, cid, ease, type
        FROM revlog 
        WHERE (type != 4 OR ivl <= 0)
        AND (factor != 0 or type != 3)
        AND id < ?1
        AND cid < ?2
        AND cid IN (
            SELECT id
            FROM cards
            WHERE queue != 0
            {suspended_cards_str}
            {flags_str}
        )
        order by cid"
        ))?
        .query_and_then((current_timestamp, current_timestamp), row_to_revlog_entry)?
        .collect::<Result<Vec<_>>>()?;
    Ok(revlogs)
}

fn convert_to_date(timestamp: i64, next_day_starts_at: i64, timezone: Tz) -> chrono::NaiveDate {
    let timestamp_seconds = timestamp - next_day_starts_at * 3600 * 1000; // 剪去指定小时数
    let datetime = Utc
        .timestamp_millis_opt(timestamp_seconds)
        .unwrap()
        .with_timezone(&timezone);
    datetime.date_naive()
}

/// Given a list of revlog entries for a single card with length n, we create
/// n-1 FSRS items, where each item contains the history of the preceding reviews.
fn convert_to_fsrs_items(
    mut entries: Vec<RevlogEntry>,
    next_day_starts_at: i64,
    timezone: Tz,
) -> Option<Vec<FSRSItem>> {
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

    // we ignore cards that don't start in the learning state
    if let Some(entry) = entries.first() {
        if entry.review_kind != 0 {
            return None;
        }
    } else {
        // no revlog entries
        return None;
    }

    // Increment review_kind of all entries by 1
    // 将所有 review_kind + 1
    entries.iter_mut().for_each(|entry| entry.review_kind += 1);

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

    // Compute i, r_history, t_history
    // 计算 i, r_history, t_history
    // Except for the first entry, the remaining entries add the preceding button_chosen and delta_t to r_history and t_history
    // 除了第一个条目，其余条目将前面的 button_chosen 和 delta_t 加入 r_history 和 t_history
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
                        rating: r.button_chosen,
                        delta_t: r.delta_t,
                    })
                    .collect();
                FSRSItem { reviews }
            })
            .collect(),
    )
}

pub(crate) fn anki21_sample_file_converted_to_fsrs() -> Vec<FSRSItem> {
    anki_to_fsrs(read_collection().expect("read error"))
}

/// Convert a series of revlog entries sorted by card id into FSRS items.
pub(crate) fn anki_to_fsrs(revlogs: Vec<RevlogEntry>) -> Vec<FSRSItem> {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::FSRSBatcher;
    use burn::data::dataloader::batcher::Batcher;
    use burn::tensor::Data;
    use itertools::Itertools;

    // This test currently expects the following .anki21 file to be placed in tests/data/:
    // https://github.com/open-spaced-repetition/fsrs-optimizer-burn/files/12394182/collection.anki21.zip
    #[test]
    fn test() {
        let revlogs = read_collection().unwrap();
        let single_card_revlog = vec![revlogs
            .iter()
            .filter(|r| r.cid == 1528947214762)
            .cloned()
            .collect_vec()];
        assert_eq!(revlogs.len(), 24394);
        let fsrs_items = anki_to_fsrs(revlogs);
        assert_eq!(fsrs_items.len(), 14290);
        assert_eq!(
            fsrs_items.iter().map(|x| x.reviews.len()).sum::<usize>(),
            49382 + 14290
        );

        // convert a subset and check it matches expectations
        let mut fsrs_items = single_card_revlog
            .into_iter()
            .filter_map(|entries| convert_to_fsrs_items(entries, 4, Tz::Asia__Shanghai))
            .flatten()
            .collect_vec();
        assert_eq!(
            &fsrs_items,
            &[
                FSRSItem {
                    reviews: vec![
                        FSRSReview {
                            rating: 3,
                            delta_t: 0
                        },
                        FSRSReview {
                            rating: 3,
                            delta_t: 5
                        }
                    ],
                },
                FSRSItem {
                    reviews: vec![
                        FSRSReview {
                            rating: 3,
                            delta_t: 0
                        },
                        FSRSReview {
                            rating: 3,
                            delta_t: 5
                        },
                        FSRSReview {
                            rating: 3,
                            delta_t: 10
                        }
                    ],
                },
                FSRSItem {
                    reviews: vec![
                        FSRSReview {
                            rating: 3,
                            delta_t: 0
                        },
                        FSRSReview {
                            rating: 3,
                            delta_t: 5
                        },
                        FSRSReview {
                            rating: 3,
                            delta_t: 10
                        },
                        FSRSReview {
                            rating: 3,
                            delta_t: 22
                        }
                    ],
                },
                FSRSItem {
                    reviews: vec![
                        FSRSReview {
                            rating: 3,
                            delta_t: 0
                        },
                        FSRSReview {
                            rating: 3,
                            delta_t: 5
                        },
                        FSRSReview {
                            rating: 3,
                            delta_t: 10
                        },
                        FSRSReview {
                            rating: 3,
                            delta_t: 22
                        },
                        FSRSReview {
                            rating: 2,
                            delta_t: 56
                        }
                    ],
                },
                FSRSItem {
                    reviews: vec![
                        FSRSReview {
                            rating: 3,
                            delta_t: 0
                        },
                        FSRSReview {
                            rating: 3,
                            delta_t: 5
                        },
                        FSRSReview {
                            rating: 3,
                            delta_t: 10
                        },
                        FSRSReview {
                            rating: 3,
                            delta_t: 22
                        },
                        FSRSReview {
                            rating: 2,
                            delta_t: 56
                        },
                        FSRSReview {
                            rating: 3,
                            delta_t: 64
                        }
                    ],
                }
            ]
        );

        use burn_ndarray::NdArrayDevice;
        let device = NdArrayDevice::Cpu;
        use burn_ndarray::NdArrayBackend;
        type Backend = NdArrayBackend<f32>;
        let batcher = FSRSBatcher::<Backend>::new(device);
        let res = batcher.batch(vec![fsrs_items.pop().unwrap()]);
        assert_eq!(res.delta_ts.into_scalar(), 64.0);
        assert_eq!(
            res.r_historys.squeeze(1).to_data(),
            Data::from([3.0, 3.0, 3.0, 3.0, 2.0])
        );
        assert_eq!(
            res.t_historys.squeeze(1).to_data(),
            Data::from([0.0, 5.0, 10.0, 22.0, 56.0])
        );
        assert_eq!(res.labels.to_data(), Data::from([1]));
    }

    #[test]
    fn ordering_of_inputs_should_not_change() {
        let revlogs = anki21_sample_file_converted_to_fsrs();
        assert_eq!(
            revlogs[0],
            FSRSItem {
                reviews: vec![
                    FSRSReview {
                        rating: 4,
                        delta_t: 0
                    },
                    FSRSReview {
                        rating: 3,
                        delta_t: 3
                    }
                ]
            }
        );
    }
}

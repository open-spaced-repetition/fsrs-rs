use crate::dataset::FSRSBatcher;
use burn::backend::ndarray::NdArrayDevice;
use burn::backend::NdArrayAutodiffBackend;
use burn::data::dataloader::batcher::Batcher;
use burn::tensor::Data;
use chrono::prelude::*;
use chrono_tz::Tz;
use itertools::Itertools;
use rusqlite::Connection;
use rusqlite::{Result, Row};

use crate::convertor_tests::RevlogReviewKind::*;
use crate::dataset::{FSRSItem, FSRSReview};

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct RevlogEntry {
    pub id: i64,
    pub cid: i64,
    pub usn: i32,
    /// - In the V1 scheduler, 3 represents easy in the learning case.
    /// - 0 represents manual rescheduling.
    pub button_chosen: u8,
    /// Positive values are in days, negative values in seconds.
    pub interval: i32,
    /// Positive values are in days, negative values in seconds.
    pub last_interval: i32,
    /// Card's ease after answering, stored as 10x the %, eg 2500 represents
    /// 250%.
    pub ease_factor: u32,
    /// Amount of milliseconds taken to answer the card.
    pub taken_millis: u32,
    pub review_kind: RevlogReviewKind,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
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

impl rusqlite::types::FromSql for RevlogReviewKind {
    fn column_result(value: rusqlite::types::ValueRef<'_>) -> rusqlite::types::FromSqlResult<Self> {
        match value {
            rusqlite::types::ValueRef::Integer(i) => match i {
                0 => Ok(RevlogReviewKind::Learning),
                1 => Ok(RevlogReviewKind::Review),
                2 => Ok(RevlogReviewKind::Relearning),
                3 => Ok(RevlogReviewKind::Filtered),
                4 => Ok(RevlogReviewKind::Manual),
                _ => Err(rusqlite::types::FromSqlError::InvalidType),
            },
            _ => Err(rusqlite::types::FromSqlError::InvalidType),
        }
    }
}

impl TryFrom<&Row<'_>> for RevlogEntry {
    type Error = rusqlite::Error;
    fn try_from(row: &Row<'_>) -> Result<Self> {
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

fn filter_out_cram(entries: Vec<RevlogEntry>) -> Vec<RevlogEntry> {
    entries
        .into_iter()
        .filter(|entry| entry.review_kind != Filtered)
        .collect()
}

fn filter_out_set_due_date(entries: Vec<RevlogEntry>) -> Vec<RevlogEntry> {
    entries
        .into_iter()
        .filter(|entry| {
            (entry.review_kind != Manual && entry.button_chosen != 0) || entry.ease_factor == 0
        })
        .collect()
}

fn remove_revlog_before_forget(entries: Vec<RevlogEntry>) -> Vec<RevlogEntry> {
    let forget_index = entries
        .iter()
        .enumerate()
        .filter(|(.., entry)| {
            (entry.review_kind == Manual || entry.button_chosen == 0) && entry.ease_factor == 0
        })
        .last()
        .map(|(index, ..)| index + 1)
        .unwrap_or_default();
    entries[forget_index..].to_vec()
}

fn remove_revlog_before_last_first_learn(entries: Vec<RevlogEntry>) -> Vec<RevlogEntry> {
    let mut last_first_learn_index = 0;
    for (index, entry) in entries.iter().enumerate().rev() {
        if entry.review_kind == Learning {
            last_first_learn_index = index;
        } else if last_first_learn_index != 0 {
            break;
        }
    }
    entries[last_first_learn_index..].to_vec()
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
    entries: Vec<RevlogEntry>,
    next_day_starts_at: i64,
    timezone: Tz,
) -> Vec<RevlogEntry> {
    let mut entries = entries.clone();
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
    entries = filter_out_cram(entries);
    entries = filter_out_set_due_date(entries);
    entries = remove_revlog_before_forget(entries);
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
                        rating: r.button_chosen as i32,
                        delta_t: r.last_interval.max(0),
                    })
                    .collect();
                FSRSItem { reviews }
            })
            .collect(),
    )
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

pub(crate) fn anki21_sample_file_converted_to_fsrs() -> Vec<FSRSItem> {
    anki_to_fsrs(read_collection().expect("read error"))
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
            "SELECT *
        FROM revlog
        WHERE id < ?1
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
        .query_and_then((current_timestamp, current_timestamp), |x| x.try_into())?
        .collect::<Result<Vec<_>>>()?;
    Ok(revlogs)
}

// This test currently expects the following .anki21 file to be placed in tests/data/:
// https://github.com/open-spaced-repetition/fsrs-optimizer-burn/files/12394182/collection.anki21.zip
#[test]
fn conversion_works() {
    let revlogs = read_collection().unwrap();
    let single_card_revlog = vec![revlogs
        .iter()
        .filter(|r| r.cid == 1528947214762)
        .cloned()
        .collect_vec()];
    assert_eq!(revlogs.len(), 24394);
    let fsrs_items = anki_to_fsrs(revlogs);
    assert_eq!(fsrs_items.len(), 14121);
    assert_eq!(
        fsrs_items.iter().map(|x| x.reviews.len()).sum::<usize>(),
        48510 + 14121
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

    let device = NdArrayDevice::Cpu;
    let batcher = FSRSBatcher::<NdArrayAutodiffBackend>::new(device);
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

const NEXT_DAY_AT: i64 = 86400 * 100;

fn revlog(review_kind: RevlogReviewKind, days_ago: i64) -> RevlogEntry {
    RevlogEntry {
        review_kind,
        id: ((NEXT_DAY_AT - days_ago * 86400) * 1000),
        button_chosen: 3,
        ..Default::default()
    }
}

#[test]
fn delta_t_is_correct() -> Result<()> {
    assert_eq!(
        convert_to_fsrs_items(
            vec![
                revlog(RevlogReviewKind::Learning, 1),
                revlog(RevlogReviewKind::Review, 0)
            ],
            NEXT_DAY_AT,
            Tz::Asia__Shanghai
        ),
        Some(vec![FSRSItem {
            reviews: vec![
                FSRSReview {
                    rating: 3,
                    delta_t: 0
                },
                FSRSReview {
                    rating: 3,
                    delta_t: 1
                }
            ]
        }])
    );

    assert_eq!(
        convert_to_fsrs_items(
            vec![
                revlog(RevlogReviewKind::Learning, 15),
                revlog(RevlogReviewKind::Learning, 13),
                revlog(RevlogReviewKind::Review, 10),
                revlog(RevlogReviewKind::Review, 5)
            ],
            NEXT_DAY_AT,
            Tz::Asia__Shanghai
        ),
        Some(vec![
            FSRSItem {
                reviews: vec![
                    FSRSReview {
                        rating: 3,
                        delta_t: 0
                    },
                    FSRSReview {
                        rating: 3,
                        delta_t: 2
                    }
                ]
            },
            FSRSItem {
                reviews: vec![
                    FSRSReview {
                        rating: 3,
                        delta_t: 0
                    },
                    FSRSReview {
                        rating: 3,
                        delta_t: 2
                    },
                    FSRSReview {
                        rating: 3,
                        delta_t: 3
                    }
                ]
            },
            FSRSItem {
                reviews: vec![
                    FSRSReview {
                        rating: 3,
                        delta_t: 0
                    },
                    FSRSReview {
                        rating: 3,
                        delta_t: 2
                    },
                    FSRSReview {
                        rating: 3,
                        delta_t: 3
                    },
                    FSRSReview {
                        rating: 3,
                        delta_t: 5
                    }
                ]
            }
        ])
    );

    Ok(())
}
#[test]
fn test_filter_out_cram() {
    let revlog_vec = vec![
        RevlogEntry {
            id: 1581372672843,
            cid: 1559078645460,
            usn: 5212,
            button_chosen: 1,
            interval: -60,
            last_interval: -60,
            ease_factor: 0,
            taken_millis: 38942,
            review_kind: Learning,
        },
        RevlogEntry {
            id: 1694423345134,
            cid: 1559078645460,
            usn: 12394,
            button_chosen: 3,
            interval: 131,
            last_interval: 73,
            ease_factor: 1750,
            taken_millis: 30443,
            review_kind: Review,
        },
        RevlogEntry {
            id: 1694422345134,
            cid: 1559078645460,
            usn: -1,
            button_chosen: 3,
            interval: -1200,
            last_interval: -600,
            ease_factor: 0,
            taken_millis: 2231,
            review_kind: Filtered,
        },
        RevlogEntry {
            id: 1694423345134,
            cid: 1559078645460,
            usn: 12394,
            button_chosen: 3,
            interval: 131,
            last_interval: 73,
            ease_factor: 1750,
            taken_millis: 30443,
            review_kind: Review,
        },
    ];
    let revlog_vec = filter_out_cram(revlog_vec);
    assert_eq!(
        revlog_vec,
        vec![
            RevlogEntry {
                id: 1581372672843,
                cid: 1559078645460,
                usn: 5212,
                button_chosen: 1,
                interval: -60,
                last_interval: -60,
                ease_factor: 0,
                taken_millis: 38942,
                review_kind: Learning,
            },
            RevlogEntry {
                id: 1694423345134,
                cid: 1559078645460,
                usn: 12394,
                button_chosen: 3,
                interval: 131,
                last_interval: 73,
                ease_factor: 1750,
                taken_millis: 30443,
                review_kind: Review,
            },
            RevlogEntry {
                id: 1694423345134,
                cid: 1559078645460,
                usn: 12394,
                button_chosen: 3,
                interval: 131,
                last_interval: 73,
                ease_factor: 1750,
                taken_millis: 30443,
                review_kind: Review,
            },
        ]
    );
}

#[test]
fn test_filter_out_set_due_date() {
    let revlog_vec = vec![
        RevlogEntry {
            id: 1581544939407,
            cid: 1559457421654,
            usn: 5226,
            button_chosen: 3,
            interval: -600,
            last_interval: -60,
            ease_factor: 0,
            taken_millis: 6979,
            review_kind: Learning,
        },
        RevlogEntry {
            id: 1581545802767,
            cid: 1559457421654,
            usn: 5226,
            button_chosen: 3,
            interval: 1,
            last_interval: -600,
            ease_factor: 2500,
            taken_millis: 21297,
            review_kind: Learning,
        },
        RevlogEntry {
            id: 1581629454703,
            cid: 1559457421654,
            usn: -1,
            button_chosen: 0,
            interval: 302,
            last_interval: 302,
            ease_factor: 2150,
            taken_millis: 0,
            review_kind: Manual,
        },
        RevlogEntry {
            id: 1582012577455,
            cid: 1559457421654,
            usn: 11054,
            button_chosen: 3,
            interval: 270,
            last_interval: 161,
            ease_factor: 2150,
            taken_millis: 31190,
            review_kind: Review,
        },
    ];
    let revlog_vec = filter_out_set_due_date(revlog_vec);
    assert_eq!(
        revlog_vec,
        vec![
            RevlogEntry {
                id: 1581544939407,
                cid: 1559457421654,
                usn: 5226,
                button_chosen: 3,
                interval: -600,
                last_interval: -60,
                ease_factor: 0,
                taken_millis: 6979,
                review_kind: Learning,
            },
            RevlogEntry {
                id: 1581545802767,
                cid: 1559457421654,
                usn: 5226,
                button_chosen: 3,
                interval: 1,
                last_interval: -600,
                ease_factor: 2500,
                taken_millis: 21297,
                review_kind: Learning,
            },
            RevlogEntry {
                id: 1582012577455,
                cid: 1559457421654,
                usn: 11054,
                button_chosen: 3,
                interval: 270,
                last_interval: 161,
                ease_factor: 2150,
                taken_millis: 31190,
                review_kind: Review,
            },
        ]
    );
}

#[test]
fn test_remove_revlog_before_forget() {
    let revlog_vec = vec![
        RevlogEntry {
            id: 1581372095493,
            cid: 1559076329057,
            usn: 5212,
            button_chosen: 1,
            interval: -60,
            last_interval: -60,
            ease_factor: 0,
            taken_millis: 60000,
            review_kind: Learning,
        },
        RevlogEntry {
            id: 1694422513063,
            cid: 1559076329057,
            usn: -1,
            button_chosen: 0,
            interval: 0,
            last_interval: 228,
            ease_factor: 0,
            taken_millis: 0,
            review_kind: Manual,
        },
        RevlogEntry {
            id: 1694422651447,
            cid: 1559076329057,
            usn: -1,
            button_chosen: 3,
            interval: -1200,
            last_interval: -60,
            ease_factor: 0,
            taken_millis: 7454,
            review_kind: Learning,
        },
    ];
    let revlog_vec = remove_revlog_before_forget(revlog_vec);
    assert_eq!(
        revlog_vec,
        vec![RevlogEntry {
            id: 1694422651447,
            cid: 1559076329057,
            usn: -1,
            button_chosen: 3,
            interval: -1200,
            last_interval: -60,
            ease_factor: 0,
            taken_millis: 7454,
            review_kind: Learning,
        },]
    );
}

#[test]
fn test_remove_revlog_before_last_first_learn() {
    let revlog_vec = vec![
        RevlogEntry {
            id: 1250450107406,
            cid: 1249409627511,
            usn: 0,
            button_chosen: 3,
            interval: 44,
            last_interval: 14,
            ease_factor: 2000,
            taken_millis: 9827,
            review_kind: Review,
        },
        RevlogEntry {
            id: 1254407572859,
            cid: 1249409627511,
            usn: 0,
            button_chosen: 3,
            interval: 90,
            last_interval: 44,
            ease_factor: 2000,
            taken_millis: 11187,
            review_kind: Review,
        },
        RevlogEntry {
            id: 1601346317001,
            cid: 1249409627511,
            usn: 6235,
            button_chosen: 2,
            interval: -330,
            last_interval: -60,
            ease_factor: 2500,
            taken_millis: 36376,
            review_kind: Learning,
        },
        RevlogEntry {
            id: 1601346783928,
            cid: 1249409627511,
            usn: 6235,
            button_chosen: 3,
            interval: -600,
            last_interval: -60,
            ease_factor: 2500,
            taken_millis: 16249,
            review_kind: Learning,
        },
        RevlogEntry {
            id: 1601349355546,
            cid: 1249409627511,
            usn: 6235,
            button_chosen: 3,
            interval: 1,
            last_interval: -600,
            ease_factor: 2500,
            taken_millis: 13272,
            review_kind: Learning,
        },
    ];
    let revlog_vec = remove_revlog_before_last_first_learn(revlog_vec);
    // dbg!(&revlog_vec);
    assert_eq!(
        revlog_vec,
        vec![
            RevlogEntry {
                id: 1601346317001,
                cid: 1249409627511,
                usn: 6235,
                button_chosen: 2,
                interval: -330,
                last_interval: -60,
                ease_factor: 2500,
                taken_millis: 36376,
                review_kind: Learning,
            },
            RevlogEntry {
                id: 1601346783928,
                cid: 1249409627511,
                usn: 6235,
                button_chosen: 3,
                interval: -600,
                last_interval: -60,
                ease_factor: 2500,
                taken_millis: 16249,
                review_kind: Learning,
            },
            RevlogEntry {
                id: 1601349355546,
                cid: 1249409627511,
                usn: 6235,
                button_chosen: 3,
                interval: 1,
                last_interval: -600,
                ease_factor: 2500,
                taken_millis: 13272,
                review_kind: Learning,
            },
        ]
    );

    let revlog_vec = vec![
        RevlogEntry {
            id: 1224652906547,
            cid: 1224124699235,
            usn: 0,
            button_chosen: 3,
            interval: 0,
            last_interval: 0,
            ease_factor: 2498,
            taken_millis: 50562,
            review_kind: Learning,
        },
        RevlogEntry {
            id: 1225062601238,
            cid: 1224124699235,
            usn: 0,
            button_chosen: 4,
            interval: 4,
            last_interval: 0,
            ease_factor: 2498,
            taken_millis: 22562,
            review_kind: Relearning,
        },
        RevlogEntry {
            id: 1389281322102,
            cid: 1224124699235,
            usn: 866,
            button_chosen: 4,
            interval: 4,
            last_interval: -60,
            ease_factor: 2500,
            taken_millis: 22559,
            review_kind: Learning,
        },
        RevlogEntry {
            id: 1389894643706,
            cid: 1224124699235,
            usn: 889,
            button_chosen: 3,
            interval: 9,
            last_interval: 4,
            ease_factor: 2500,
            taken_millis: 14763,
            review_kind: Review,
        },
    ];

    let revlog_vec = remove_revlog_before_last_first_learn(revlog_vec);
    assert_eq!(
        revlog_vec,
        vec![
            RevlogEntry {
                id: 1389281322102,
                cid: 1224124699235,
                usn: 866,
                button_chosen: 4,
                interval: 4,
                last_interval: -60,
                ease_factor: 2500,
                taken_millis: 22559,
                review_kind: Learning,
            },
            RevlogEntry {
                id: 1389894643706,
                cid: 1224124699235,
                usn: 889,
                button_chosen: 3,
                interval: 9,
                last_interval: 4,
                ease_factor: 2500,
                taken_millis: 14763,
                review_kind: Review,
            },
        ]
    );
}

#[test]
fn test_keep_first_revlog_same_date() {
    let revlog_vec = vec![
        RevlogEntry {
            id: 1581372095493,
            cid: 1559076329057,
            usn: 5212,
            button_chosen: 1,
            interval: -60,
            last_interval: -60,
            ease_factor: 0,
            taken_millis: 60000,
            review_kind: Learning,
        },
        RevlogEntry {
            id: 1581372260598,
            cid: 1559076329057,
            usn: 5212,
            button_chosen: 3,
            interval: -600,
            last_interval: -60,
            ease_factor: 0,
            taken_millis: 46425,
            review_kind: Learning,
        },
        RevlogEntry {
            id: 1581406251414,
            cid: 1559076329057,
            usn: 5213,
            button_chosen: 2,
            interval: -600,
            last_interval: -600,
            ease_factor: 0,
            taken_millis: 17110,
            review_kind: Learning,
        },
        RevlogEntry {
            id: 1581407568344,
            cid: 1559076329057,
            usn: 5213,
            button_chosen: 3,
            interval: 1,
            last_interval: -600,
            ease_factor: 2500,
            taken_millis: 8861,
            review_kind: Learning,
        },
        RevlogEntry {
            id: 1581454426227,
            cid: 1559076329057,
            usn: 5215,
            button_chosen: 3,
            interval: 3,
            last_interval: 1,
            ease_factor: 2500,
            taken_millis: 13128,
            review_kind: Review,
        },
    ];
    let revlog_vec = keep_first_revlog_same_date(revlog_vec, 4, Tz::Asia__Shanghai);
    assert_eq!(
        revlog_vec,
        vec![
            RevlogEntry {
                id: 1581372095493,
                cid: 1559076329057,
                usn: 5212,
                button_chosen: 1,
                interval: -60,
                last_interval: -60,
                ease_factor: 0,
                taken_millis: 60000,
                review_kind: Learning,
            },
            RevlogEntry {
                id: 1581454426227,
                cid: 1559076329057,
                usn: 5215,
                button_chosen: 3,
                interval: 3,
                last_interval: 1,
                ease_factor: 2500,
                taken_millis: 13128,
                review_kind: Review,
            },
        ]
    )
}

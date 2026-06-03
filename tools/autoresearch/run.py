#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
import urllib.request
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


DATA_URL = (
    "https://github.com/open-spaced-repetition/fsrs-optimizer-burn/"
    "files/12394182/collection.anki21.zip"
)
PRIMARY_BENCH = "compute_parameters"
BENCH_GROUP = "parameters"
GOLDEN_EXAMPLE_NAME = "autoresearch_golden"
GOLDEN_EXAMPLE_REL_PATH = Path("examples") / f"{GOLDEN_EXAMPLE_NAME}.rs"
GOLDEN_BEGIN = "AUTORESEARCH_GOLDEN_BEGIN"
GOLDEN_END = "AUTORESEARCH_GOLDEN_END"
COLLECTION_REL_PATH = Path("tests/data/collection.anki21")
TEMPORARILY_SKIPPED_CORRECTNESS_TESTS = (
    "simulation::tests::test_optimal_retention_with_old_parameters",
)


GOLDEN_EXAMPLE_SOURCE = r'''
use chrono::prelude::*;
use chrono_tz::Tz;
use fsrs::{
    ComputeParametersInput, FSRS, FSRSItem, FSRSReview, MemoryState, compute_parameters,
    current_retrievability, evaluate_with_time_series_splits,
};
use itertools::Itertools;
use rusqlite::{Connection, Result as SqlResult, Row, types::FromSqlError as SqlFromSqlError};
use std::fmt::Write;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
enum RevlogReviewKind {
    Learning,
    Review,
    #[default]
    Relearning,
    Filtered,
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
) -> Option<Vec<(i64, i64, FSRSItem)>> {
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
                (entry.id, entry.cid, FSRSItem { reviews })
            })
            .filter(|(_, _, item)| item.reviews.last().is_some_and(|r| r.delta_t > 0))
            .collect(),
    )
}

fn anki_to_fsrs(revlogs: Vec<RevlogEntry>) -> Vec<(FSRSItem, i64)> {
    let mut revlogs_by_card = revlogs
        .into_iter()
        .chunk_by(|r| r.cid)
        .into_iter()
        .filter_map(|(_cid, entries)| {
            convert_to_fsrs_items(entries.collect(), 4, Tz::Asia__Shanghai)
        })
        .flatten()
        .collect_vec();
    revlogs_by_card.sort_by_cached_key(|(id, _, _)| *id);
    revlogs_by_card
        .into_iter()
        .map(|(_, card_id, item)| (item, card_id))
        .collect()
}

fn read_collection_inline() -> SqlResult<Vec<RevlogEntry>> {
    let db = Connection::open("tests/data/collection.anki21")?;
    let filter_out_suspended_cards = false;
    let filter_out_flags = Vec::<i32>::new();
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

fn anki21_sample_file_converted_to_fsrs_inline() -> Vec<(FSRSItem, i64)> {
    anki_to_fsrs(read_collection_inline().expect("read error for inlined function"))
}

fn prepare_training_data_inline(
    items: Vec<(FSRSItem, i64)>,
) -> (Vec<(FSRSItem, i64)>, Vec<(FSRSItem, i64)>) {
    let filtered_items: Vec<(FSRSItem, i64)> = items
        .into_iter()
        .filter(|(item, _)| {
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

fn load_and_prepare_data_with_card_ids() -> (Vec<FSRSItem>, Vec<i64>) {
    let items = anki21_sample_file_converted_to_fsrs_inline();
    let (pretrain_set, train_set) = prepare_training_data_inline(items);
    [pretrain_set, train_set]
        .concat()
        .into_iter()
        .unzip()
}

fn dataset_checksum(items: &[FSRSItem]) -> u64 {
    let mut checksum = 1469598103934665603u64;
    for item in items {
        checksum = checksum.wrapping_mul(1099511628211);
        checksum ^= item.reviews.len() as u64;
        for review in &item.reviews {
            checksum = checksum.wrapping_mul(1099511628211);
            checksum ^= ((review.rating as u64) << 32) | review.delta_t as u64;
        }
    }
    checksum
}

fn json_num(v: f32) -> String {
    assert!(v.is_finite(), "golden value must be finite");
    format!("{v:.9}")
}

fn json_vec(values: &[f32]) -> String {
    let mut out = String::from("[");
    for (idx, value) in values.iter().enumerate() {
        if idx > 0 {
            out.push(',');
        }
        out.push_str(&json_num(*value));
    }
    out.push(']');
    out
}

fn json_memory_state(state: MemoryState) -> String {
    format!(
        "{{\"stability\":{},\"difficulty\":{}}}",
        json_num(state.stability),
        json_num(state.difficulty)
    )
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let (items, card_ids) = load_and_prepare_data_with_card_ids();
    let fsrs = FSRS::default();
    let evaluation = fsrs.evaluate(items.clone(), |_| true)?;
    let parameters = compute_parameters(ComputeParametersInput {
        train_set: items.clone(),
        card_ids: Some(card_ids.clone()),
        progress: None,
        enable_short_term: true,
        num_relearning_steps: None,
    })?;
    let trained_fsrs = FSRS::new(&parameters)?;
    let trained_evaluation = trained_fsrs.evaluate(items.clone(), |_| true)?;
    let time_series_evaluation = evaluate_with_time_series_splits(
        ComputeParametersInput {
            train_set: items.clone(),
            card_ids: None,
            progress: None,
            enable_short_term: true,
            num_relearning_steps: None,
        },
        |_| true,
    )?;

    let item_a = FSRSItem {
        reviews: vec![
            FSRSReview {
                rating: 4,
                delta_t: 0,
            },
            FSRSReview {
                rating: 3,
                delta_t: 5,
            },
            FSRSReview {
                rating: 1,
                delta_t: 12,
            },
            FSRSReview {
                rating: 3,
                delta_t: 4,
            },
        ],
    };
    let item_b = FSRSItem {
        reviews: vec![
            FSRSReview {
                rating: 2,
                delta_t: 0,
            },
            FSRSReview {
                rating: 4,
                delta_t: 1,
            },
            FSRSReview {
                rating: 3,
                delta_t: 7,
            },
        ],
    };
    let memory = fsrs.memory_state(item_a.clone(), None)?;
    let batch_states = fsrs.memory_state_batch(
        vec![item_a, item_b],
        vec![
            None,
            Some(MemoryState {
                stability: 2.5,
                difficulty: 6.0,
            }),
        ],
    )?;
    let next = fsrs.next_states(Some(memory), 0.9, 21)?;
    let retrievability = current_retrievability(
        MemoryState {
            stability: 51.344814,
            difficulty: 7.005062,
        },
        21.0,
        fsrs::FSRS6_DEFAULT_DECAY,
    );

    let mut out = String::new();
    write!(
        &mut out,
        "{{\"dataset\":{{\"len\":{},\"checksum\":{}}},",
        items.len(),
        dataset_checksum(&items)
    )?;
    write!(
        &mut out,
        "\"evaluate\":{{\"log_loss\":{},\"rmse_bins\":{}}},",
        json_num(evaluation.log_loss),
        json_num(evaluation.rmse_bins)
    )?;
    write!(&mut out, "\"compute_parameters\":{},", json_vec(&parameters))?;
    write!(
        &mut out,
        "\"trained_evaluate\":{{\"log_loss\":{},\"rmse_bins\":{}}},",
        json_num(trained_evaluation.log_loss),
        json_num(trained_evaluation.rmse_bins)
    )?;
    write!(
        &mut out,
        "\"time_series_evaluate\":{{\"log_loss\":{},\"rmse_bins\":{}}},",
        json_num(time_series_evaluation.log_loss),
        json_num(time_series_evaluation.rmse_bins)
    )?;
    write!(
        &mut out,
        "\"inference\":{{\"current_retrievability\":{},\"memory_state\":{},",
        json_num(retrievability),
        json_memory_state(memory)
    )?;
    out.push_str("\"memory_state_batch\":[");
    for (idx, state) in batch_states.iter().enumerate() {
        if idx > 0 {
            out.push(',');
        }
        out.push_str(&json_memory_state(*state));
    }
    out.push_str("],");
    write!(
        &mut out,
        "\"next_states\":{{\"again\":{{\"memory\":{},\"interval\":{}}},\"hard\":{{\"memory\":{},\"interval\":{}}},\"good\":{{\"memory\":{},\"interval\":{}}},\"easy\":{{\"memory\":{},\"interval\":{}}}}}}}}}",
        json_memory_state(next.again.memory),
        json_num(next.again.interval),
        json_memory_state(next.hard.memory),
        json_num(next.hard.interval),
        json_memory_state(next.good.memory),
        json_num(next.good.interval),
        json_memory_state(next.easy.memory),
        json_num(next.easy.interval)
    )?;

    println!("AUTORESEARCH_GOLDEN_BEGIN");
    println!("{out}");
    println!("AUTORESEARCH_GOLDEN_END");
    Ok(())
}
'''


@dataclass(frozen=True)
class CommandResult:
    returncode: int
    stdout: str
    stderr: str


@dataclass(frozen=True)
class BenchRun:
    benches_ms: dict[str, float]
    stdout: str


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def run_command(
    cmd: list[str],
    cwd: Path,
    log_path: Path,
    *,
    env: dict[str, str] | None = None,
    timeout: int | None = None,
) -> CommandResult:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"$ {' '.join(cmd)}\n")
        log.write(f"cwd: {cwd}\n")
        log.write(f"started: {datetime.now(timezone.utc).isoformat()}\n\n")
    try:
        proc = subprocess.run(
            cmd,
            cwd=cwd,
            env=merged_env,
            text=True,
            capture_output=True,
            timeout=timeout,
        )
        result = CommandResult(proc.returncode, proc.stdout, proc.stderr)
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout if isinstance(exc.stdout, str) else ""
        stderr = exc.stderr if isinstance(exc.stderr, str) else ""
        result = CommandResult(124, stdout, stderr + f"\nTIMEOUT after {timeout}s\n")

    with log_path.open("a", encoding="utf-8") as log:
        log.write("----- stdout -----\n")
        log.write(result.stdout)
        log.write("\n----- stderr -----\n")
        log.write(result.stderr)
        log.write(f"\nexit: {result.returncode}\n")
    return result


def require_success(result: CommandResult, description: str) -> None:
    if result.returncode != 0:
        raise RuntimeError(f"{description} failed with exit {result.returncode}")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def median(values: Iterable[float]) -> float:
    seq = list(values)
    if not seq:
        raise ValueError("median() requires at least one value")
    seq.sort()
    mid = len(seq) // 2
    if len(seq) % 2:
        return seq[mid]
    return (seq[mid - 1] + seq[mid]) / 2.0


def ensure_collection_cache(repo: Path, work_root: Path, log_dir: Path) -> Path:
    existing = repo / COLLECTION_REL_PATH
    if existing.exists():
        return existing

    data_dir = work_root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    cached = data_dir / COLLECTION_REL_PATH.name
    if cached.exists():
        return cached

    archive = data_dir / "collection.anki21.zip"
    log_dir.mkdir(parents=True, exist_ok=True)
    with (log_dir / "download-collection.log").open("w", encoding="utf-8") as log:
        log.write(f"download {DATA_URL}\n")
        with urllib.request.urlopen(DATA_URL, timeout=120) as response:
            archive.write_bytes(response.read())
        log.write(f"wrote {archive}\n")

    with zipfile.ZipFile(archive) as zf:
        candidates = [name for name in zf.namelist() if name.endswith("collection.anki21")]
        if not candidates:
            raise RuntimeError("collection archive did not contain collection.anki21")
        with zf.open(candidates[0]) as source, cached.open("wb") as target:
            shutil.copyfileobj(source, target)
    return cached


def ensure_collection(worktree: Path, cached_collection: Path) -> None:
    target = worktree / COLLECTION_REL_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        shutil.copy2(cached_collection, target)


def target_dir_for(output_dir: Path) -> Path:
    return output_dir / "target"


def parse_estimate(target_dir: Path, bench: str) -> float:
    path = target_dir / "criterion" / BENCH_GROUP / bench / "new" / "estimates.json"
    if not path.exists():
        raise RuntimeError(f"missing Criterion estimate: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    estimate = data.get("median") or data.get("mean")
    return float(estimate["point_estimate"]) / 1_000_000.0


def extract_marked_json(stdout: str, begin: str, end: str) -> dict[str, object]:
    start = stdout.find(begin)
    stop = stdout.find(end)
    if start == -1 or stop == -1 or stop <= start:
        raise RuntimeError("golden probe output did not contain JSON markers")
    payload = stdout[start + len(begin) : stop].strip()
    return json.loads(payload)


def write_golden_example(worktree: Path) -> Path:
    path = worktree / GOLDEN_EXAMPLE_REL_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(GOLDEN_EXAMPLE_SOURCE, encoding="utf-8")
    return path


def remove_golden_example(worktree: Path) -> None:
    try:
        (worktree / GOLDEN_EXAMPLE_REL_PATH).unlink()
    except FileNotFoundError:
        pass


def generate_golden(
    worktree: Path,
    output_dir: Path,
    label: str,
    args: argparse.Namespace,
) -> dict[str, object]:
    write_golden_example(worktree)
    try:
        env = {
            "CARGO_INCREMENTAL": "0",
            "CARGO_TARGET_DIR": str(target_dir_for(output_dir)),
        }
        result = run_command(
            ["cargo", "run", "--release", "--example", GOLDEN_EXAMPLE_NAME],
            worktree,
            output_dir / "logs" / f"{label}-golden.log",
            env=env,
            timeout=args.golden_timeout_mins * 60,
        )
        require_success(result, f"{label} golden probe")
        golden = extract_marked_json(result.stdout, GOLDEN_BEGIN, GOLDEN_END)
        (output_dir / f"{label}-golden.json").write_text(
            json.dumps(golden, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return golden
    finally:
        remove_golden_example(worktree)


def compare_golden(
    expected: object,
    actual: object,
    *,
    rel_tol: float,
    abs_tol: float,
) -> tuple[bool, list[dict[str, object]], float]:
    mismatches: list[dict[str, object]] = []
    max_relative_error = 0.0

    def mismatch(path: str, expected_value: object, actual_value: object, reason: str) -> None:
        mismatches.append(
            {
                "path": path,
                "expected": expected_value,
                "actual": actual_value,
                "reason": reason,
            }
        )

    def walk(path: str, left: object, right: object) -> None:
        nonlocal max_relative_error
        if isinstance(left, dict) and isinstance(right, dict):
            for key in sorted(set(left) | set(right)):
                child = f"{path}.{key}" if path != "$" else f"$.{key}"
                if key not in left:
                    mismatch(child, None, right[key], "unexpected key")
                elif key not in right:
                    mismatch(child, left[key], None, "missing key")
                else:
                    walk(child, left[key], right[key])
            return
        if isinstance(left, list) and isinstance(right, list):
            if len(left) != len(right):
                mismatch(f"{path}.length", len(left), len(right), "length mismatch")
                return
            for idx, (l_item, r_item) in enumerate(zip(left, right)):
                walk(f"{path}[{idx}]", l_item, r_item)
            return
        if isinstance(left, (int, float)) and isinstance(right, (int, float)):
            l_float = float(left)
            r_float = float(right)
            if not math.isfinite(l_float) or not math.isfinite(r_float):
                if l_float != r_float:
                    mismatch(path, left, right, "non-finite mismatch")
                return
            abs_error = abs(l_float - r_float)
            denom = max(abs(l_float), abs(r_float), 1e-12)
            rel_error = abs_error / denom
            max_relative_error = max(max_relative_error, rel_error)
            if abs_error > abs_tol and rel_error > rel_tol:
                mismatch(path, left, right, f"numeric mismatch rel={rel_error:.6g}")
            return
        if left != right:
            mismatch(path, left, right, "value mismatch")

    walk("$", expected, actual)
    return not mismatches, mismatches, max_relative_error


def _prefixed(prefix: str, mismatches: list[dict[str, object]]) -> list[dict[str, object]]:
    out = []
    for mismatch in mismatches:
        row = dict(mismatch)
        path = str(row["path"])
        row["path"] = prefix if path == "$" else prefix + path[1:]
        out.append(row)
    return out


def _json_field(data: object, path: tuple[str, ...]) -> object:
    current = data
    for key in path:
        if not isinstance(current, dict) or key not in current:
            raise KeyError(".".join(path))
        current = current[key]
    return current


def _json_float(data: object, path: tuple[str, ...]) -> float:
    value = _json_field(data, path)
    if not isinstance(value, (int, float)):
        raise TypeError(".".join(path))
    return float(value)


def _numeric_list(value: object) -> list[float] | None:
    if not isinstance(value, list):
        return None
    out = []
    for item in value:
        if not isinstance(item, (int, float)):
            return None
        value = float(item)
        if not math.isfinite(value):
            return None
        out.append(value)
    return out


def _parameter_diagnostics(
    expected: object,
    actual: object,
    *,
    rel_tol: float,
    abs_tol: float,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    mismatches: list[dict[str, object]] = []
    expected_params = _numeric_list(expected)
    actual_params = _numeric_list(actual)
    if expected_params is None:
        mismatches.append({"path": "$.compute_parameters", "reason": "baseline parameters invalid"})
        expected_params = []
    if actual_params is None:
        mismatches.append({"path": "$.compute_parameters", "reason": "candidate parameters invalid"})
        actual_params = []
    if len(actual_params) != 21:
        mismatches.append(
            {
                "path": "$.compute_parameters.length",
                "expected": 21,
                "actual": len(actual_params),
                "reason": "wrong parameter count",
            }
        )
    drift_count = 0
    max_abs_delta = 0.0
    max_rel_delta = 0.0
    for left, right in zip(expected_params, actual_params):
        abs_delta = abs(right - left)
        rel_delta = abs_delta / max(abs(left), abs(right), 1e-12)
        max_abs_delta = max(max_abs_delta, abs_delta)
        max_rel_delta = max(max_rel_delta, rel_delta)
        if abs_delta > abs_tol and rel_delta > rel_tol:
            drift_count += 1
    return (
        {
            "drift_count": drift_count,
            "max_abs_delta": max_abs_delta,
            "max_rel_delta": max_rel_delta,
        },
        mismatches,
    )


def compare_golden_for_mode(
    expected: dict[str, object],
    actual: dict[str, object],
    *,
    mode: str,
    rel_tol: float,
    abs_tol: float,
    logloss_band: float,
) -> dict[str, object]:
    if mode == "bit-exact":
        ok, mismatches, max_relative_error = compare_golden(
            expected,
            actual,
            rel_tol=rel_tol,
            abs_tol=abs_tol,
        )
        return {
            "ok": ok,
            "mode": mode,
            "mismatch_count": len(mismatches),
            "mismatches": mismatches,
            "max_relative_error": max_relative_error,
        }

    if mode != "logloss-band":
        raise ValueError(f"unknown correctness mode: {mode}")

    mismatches: list[dict[str, object]] = []
    max_relative_error = 0.0
    for key in ("dataset", "evaluate", "inference"):
        ok, section_mismatches, section_max = compare_golden(
            _json_field(expected, (key,)),
            _json_field(actual, (key,)),
            rel_tol=rel_tol,
            abs_tol=abs_tol,
        )
        del ok
        max_relative_error = max(max_relative_error, section_max)
        mismatches.extend(_prefixed(f"$.{key}", section_mismatches))

    diagnostics, parameter_mismatches = _parameter_diagnostics(
        expected.get("compute_parameters"),
        actual.get("compute_parameters"),
        rel_tol=rel_tol,
        abs_tol=abs_tol,
    )
    mismatches.extend(parameter_mismatches)

    logloss_deltas: dict[str, float] = {}
    for path in (
        ("trained_evaluate", "log_loss"),
        ("time_series_evaluate", "log_loss"),
    ):
        label = ".".join(path)
        try:
            delta = _json_float(actual, path) - _json_float(expected, path)
            logloss_deltas[label] = delta
            if abs(delta) > logloss_band:
                mismatches.append(
                    {
                        "path": "$." + label,
                        "expected": _json_float(expected, path),
                        "actual": _json_float(actual, path),
                        "delta": delta,
                        "reason": f"logloss delta exceeds band {logloss_band}",
                    }
                )
        except (KeyError, TypeError) as exc:
            mismatches.append({"path": "$." + label, "reason": str(exc)})

    return {
        "ok": not mismatches,
        "mode": mode,
        "mismatch_count": len(mismatches),
        "mismatches": mismatches,
        "max_relative_error": max_relative_error,
        "parameter_diagnostics": diagnostics,
        "logloss_deltas": logloss_deltas,
    }


def run_correctness(worktree: Path, output_dir: Path, args: argparse.Namespace) -> None:
    checks: list[tuple[str, list[str], dict[str, str]]] = [
        ("fmt", ["cargo", "fmt", "--check"], {}),
        ("clippy", ["cargo", "clippy", "--all-targets", "--", "-D", "warnings"], {}),
        (
            "release-tests",
            [
                "cargo",
                "test",
                "--release",
                "--",
                *[
                    part
                    for test_name in TEMPORARILY_SKIPPED_CORRECTNESS_TESTS
                    for part in ("--skip", test_name)
                ],
            ],
            {"SKIP_TRAINING": "1"} if args.skip_training_tests else {},
        ),
        (
            "doc",
            ["cargo", "doc", "--release"],
            {"RUSTDOCFLAGS": "-D warnings"},
        ),
    ]
    for name, cmd, env in checks:
        result = run_command(
            cmd,
            worktree,
            output_dir / "logs" / f"{name}.log",
            env=env,
            timeout=args.check_timeout_mins * 60,
        )
        require_success(result, name)


def bench_once(
    worktree: Path,
    output_dir: Path,
    trial: int,
    args: argparse.Namespace,
) -> BenchRun:
    target_dir = target_dir_for(output_dir)
    shutil.rmtree(target_dir / "criterion" / BENCH_GROUP / PRIMARY_BENCH, ignore_errors=True)
    env = {
        "CARGO_INCREMENTAL": "0",
        "CARGO_TARGET_DIR": str(target_dir),
    }
    result = run_command(
        [
            "cargo",
            "bench",
            "--bench",
            "parameters",
            "--",
            PRIMARY_BENCH,
            "--sample-size",
            str(args.sample_size),
            "--warm-up-time",
            str(args.warm_up_time),
            "--measurement-time",
            str(args.measurement_time),
        ],
        worktree,
        output_dir / "logs" / f"bench-trial-{trial}.log",
        env=env,
        timeout=args.bench_timeout_mins * 60,
    )
    require_success(result, f"benchmark trial {trial}")
    return BenchRun(
        benches_ms={PRIMARY_BENCH: parse_estimate(target_dir, PRIMARY_BENCH)},
        stdout=result.stdout,
    )


def write_manifest(output_dir: Path, repo: Path, args: argparse.Namespace) -> None:
    proc = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        text=True,
        capture_output=True,
        check=False,
    )
    serializable_args = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "repo": str(repo),
        "head": proc.stdout.strip() if proc.returncode == 0 else None,
        "tool": "verification-only",
        "golden_sha256": sha256_text(GOLDEN_EXAMPLE_SOURCE),
        "args": serializable_args,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify FSRS-rs performance changes without candidate discovery."
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--work-root", type=Path, default=Path("tmp/autoresearch/verify"))
    parser.add_argument("--baseline-golden", type=Path)
    parser.add_argument("--skip-correctness", action="store_true")
    parser.add_argument("--skip-golden", action="store_true")
    parser.add_argument("--skip-bench", action="store_true")
    parser.add_argument("--include-training-tests", action="store_true")
    parser.add_argument("--sample-size", type=int, default=10)
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--warm-up-time", type=int, default=3)
    parser.add_argument("--measurement-time", type=int, default=5)
    parser.add_argument("--check-timeout-mins", type=int, default=45)
    parser.add_argument("--bench-timeout-mins", type=int, default=45)
    parser.add_argument("--golden-timeout-mins", type=int, default=30)
    parser.add_argument("--golden-rel-tol", type=float, default=1e-4)
    parser.add_argument("--golden-abs-tol", type=float, default=1e-4)
    parser.add_argument(
        "--correctness-mode",
        choices=("bit-exact", "logloss-band"),
        default="bit-exact",
    )
    parser.add_argument("--logloss-band", type=float, default=0.0010)
    args = parser.parse_args()
    args.skip_training_tests = not args.include_training_tests

    if args.trials < 1:
        raise SystemExit("--trials must be at least 1")

    repo = Path.cwd().resolve()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir
        else (repo / args.work_root / "runs" / utc_timestamp()).resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    write_manifest(output_dir, repo, args)

    cached_collection = ensure_collection_cache(repo, repo / args.work_root, output_dir / "logs")
    ensure_collection(repo, cached_collection)

    summary: dict[str, object] = {
        "output_dir": str(output_dir),
        "correctness": None,
        "golden": None,
        "benchmark": None,
    }

    if not args.skip_correctness:
        run_correctness(repo, output_dir, args)
        summary["correctness"] = {"ok": True}

    if not args.skip_golden:
        current_golden = generate_golden(repo, output_dir, "current", args)
        golden_summary: dict[str, object] = {"ok": True, "path": str(output_dir / "current-golden.json")}
        if args.baseline_golden:
            baseline = json.loads(args.baseline_golden.read_text(encoding="utf-8"))
            comparison = compare_golden_for_mode(
                baseline,
                current_golden,
                mode=args.correctness_mode,
                rel_tol=args.golden_rel_tol,
                abs_tol=args.golden_abs_tol,
                logloss_band=args.logloss_band,
            )
            (output_dir / "golden-compare.json").write_text(
                json.dumps(comparison, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            golden_summary["comparison"] = comparison
            if not comparison["ok"]:
                summary["golden"] = golden_summary
                (output_dir / "summary.json").write_text(
                    json.dumps(summary, indent=2, sort_keys=True),
                    encoding="utf-8",
                )
                raise SystemExit("golden verification failed")
        summary["golden"] = golden_summary

    if not args.skip_bench:
        runs = [bench_once(repo, output_dir, trial, args) for trial in range(1, args.trials + 1)]
        values = [run.benches_ms[PRIMARY_BENCH] for run in runs]
        bench_summary = {
            "bench": PRIMARY_BENCH,
            "trials": values,
            "median_ms": median(values),
            "sample_size": args.sample_size,
            "warm_up_time": args.warm_up_time,
            "measurement_time": args.measurement_time,
        }
        (output_dir / "bench-summary.json").write_text(
            json.dumps(bench_summary, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        summary["benchmark"] = bench_summary

    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

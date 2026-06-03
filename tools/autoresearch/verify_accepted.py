#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


BENCH_GROUP = "parameters"
BENCH_NAME = "compute_parameters"
BENCH_FILTER = f"{BENCH_GROUP}/{BENCH_NAME}"
COLLECTION_REL = Path("tests/data/collection.anki21")


@dataclass(frozen=True)
class CommandResult:
    returncode: int
    stdout: str
    stderr: str


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
    started = datetime.now(timezone.utc).isoformat()
    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"$ {' '.join(cmd)}\n")
        log.write(f"cwd: {cwd}\n")
        log.write(f"started: {started}\n\n")
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
        stderr += f"\nTIMEOUT after {timeout}s\n"
        result = CommandResult(124, stdout, stderr)

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


def git(repo: Path, args: list[str]) -> str:
    proc = subprocess.run(
        ["git", *args],
        cwd=repo,
        text=True,
        capture_output=True,
        check=True,
    )
    return proc.stdout.strip()


def target_dir_for(work_root: Path, state_name: str) -> Path:
    return work_root / "targets" / state_name


def remove_worktree(repo: Path, path: Path, log_dir: Path) -> None:
    if path.exists():
        run_command(
            ["git", "worktree", "remove", "--force", str(path)],
            repo,
            log_dir / f"remove-{path.name}.log",
        )
    shutil.rmtree(path, ignore_errors=True)


def create_state_worktree(
    repo: Path,
    baseline_sha: str,
    path: Path,
    patch_path: Path | None,
    sample_size: int,
    collection: Path,
    log_dir: Path,
) -> None:
    remove_worktree(repo, path, log_dir)
    result = run_command(
        ["git", "worktree", "add", "--detach", str(path), baseline_sha],
        repo,
        log_dir / f"add-{path.name}.log",
    )
    require_success(result, f"create worktree {path.name}")
    if patch_path is not None:
        result = run_command(
            ["git", "apply", "--binary", str(patch_path)],
            path,
            log_dir / f"apply-{path.name}.log",
        )
        require_success(result, f"apply patch to {path.name}")

    data_dir = path / COLLECTION_REL.parent
    data_dir.mkdir(parents=True, exist_ok=True)
    if not (path / COLLECTION_REL).exists():
        shutil.copy2(collection, path / COLLECTION_REL)

    bench_path = path / "benches/parameters.rs"
    text = bench_path.read_text(encoding="utf-8")
    text, count = re.subn(r"group\.sample_size\(10\);", f"group.sample_size({sample_size});", text)
    if count != 3:
        raise RuntimeError(f"expected to patch 3 benchmark sample_size calls, patched {count}")
    bench_path.write_text(text, encoding="utf-8")


def parse_estimate(target_dir: Path) -> float:
    path = target_dir / "criterion" / BENCH_GROUP / BENCH_NAME / "new" / "estimates.json"
    if not path.exists():
        raise RuntimeError(f"missing Criterion estimate: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    estimate = data.get("median") or data.get("mean")
    return float(estimate["point_estimate"]) / 1_000_000.0


def observed_sample_count(log_text: str) -> int | None:
    matches = re.findall(r"Collecting\s+(\d+)\s+samples", log_text)
    if not matches:
        return None
    return max(int(value) for value in matches)


def bench_state(
    worktree: Path,
    target_dir: Path,
    log_path: Path,
    sample_size: int,
    warm_up_time: int,
    measurement_time: int,
    timeout: int,
) -> dict[str, object]:
    shutil.rmtree(target_dir / "criterion" / BENCH_GROUP / BENCH_NAME, ignore_errors=True)
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
            BENCH_FILTER,
            "--sample-size",
            str(sample_size),
            "--warm-up-time",
            str(warm_up_time),
            "--measurement-time",
            str(measurement_time),
        ],
        worktree,
        log_path,
        env=env,
        timeout=timeout,
    )
    require_success(result, f"benchmark {worktree.name}")
    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    return {
        "median_ms": parse_estimate(target_dir),
        "observed_sample_count": observed_sample_count(log_text),
        "log": str(log_path),
    }


def slug(title: str) -> str:
    value = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
    return value or "patch"


def write_tsv(path: Path, rows: list[dict[str, object]]) -> None:
    headers = [
        "idx",
        "title",
        "speedup",
        "wins",
        "prev_median_ms",
        "current_median_ms",
        "speedup_gate_ok",
        "wins_gate_ok",
        "sample_size_ok",
        "patch_file",
    ]
    with path.open("w", encoding="utf-8") as file:
        file.write("\t".join(headers) + "\n")
        for row in rows:
            file.write(
                "\t".join(
                    str(row.get(header, ""))
                    for header in headers
                )
                + "\n"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--accepted-summary",
        type=Path,
        default=Path("artifacts/autoresearch_accepted_20260603/accepted_summary.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/autoresearch_accepted_20260603/high_sample_verification"),
    )
    parser.add_argument("--work-root", type=Path, default=Path("tmp/autoresearch/high_sample_verify"))
    parser.add_argument("--sample-size", type=int, default=30)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--warm-up-time", type=int, default=1)
    parser.add_argument("--measurement-time", type=int, default=8)
    parser.add_argument("--timeout-mins", type=int, default=45)
    args = parser.parse_args()

    if args.sample_size <= 10:
        raise SystemExit("--sample-size must be greater than the original sample size 10")
    if args.trials < 1:
        raise SystemExit("--trials must be at least 1")

    repo = Path.cwd().resolve()
    accepted_summary = (repo / args.accepted_summary).resolve()
    output_dir = (repo / args.output_dir).resolve()
    work_root = (repo / args.work_root).resolve()
    worktrees = work_root / "worktrees"
    logs = output_dir / "logs"
    output_dir.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    worktrees.mkdir(parents=True, exist_ok=True)

    accepted = json.loads(accepted_summary.read_text(encoding="utf-8"))
    baseline_sha = git(repo, ["rev-parse", "HEAD"])
    collection = repo / COLLECTION_REL
    if not collection.exists():
        cached = repo / "tmp/autoresearch/data/collection.anki21"
        if not cached.exists():
            raise SystemExit(f"missing collection data: {collection} or {cached}")
        collection = cached

    states: list[dict[str, object]] = [
        {
            "idx": 0,
            "name": "state-00-baseline",
            "title": "baseline",
            "patch_file": None,
            "worktree": str(worktrees / "state-00-baseline"),
        }
    ]
    for index, row in enumerate(accepted, start=1):
        states.append(
            {
                "idx": index,
                "name": f"state-{index:02d}-{slug(row['title'])}",
                "title": row["title"],
                "patch_file": str((repo / row["patch_file"]).resolve()),
                "patch_sha256": row["patch_sha256"],
                "worktree": str(worktrees / f"state-{index:02d}-{slug(row['title'])}"),
            }
        )

    started_at = datetime.now(timezone.utc).isoformat()
    config = {
        "started_at_utc": started_at,
        "baseline_sha": baseline_sha,
        "sample_size": args.sample_size,
        "trials": args.trials,
        "warm_up_time": args.warm_up_time,
        "measurement_time": args.measurement_time,
        "bench_filter": BENCH_FILTER,
        "accepted_summary": str(accepted_summary),
        "work_root": str(work_root),
        "states": states,
    }
    (output_dir / "verification_config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    try:
        for state in states:
            patch_file = state.get("patch_file")
            create_state_worktree(
                repo,
                baseline_sha,
                Path(str(state["worktree"])),
                Path(str(patch_file)) if patch_file else None,
                args.sample_size,
                collection,
                logs,
            )

        state_results: list[dict[str, object]] = []
        for trial in range(1, args.trials + 1):
            for state in states:
                name = str(state["name"])
                worktree = Path(str(state["worktree"]))
                result = bench_state(
                    worktree,
                    target_dir_for(work_root, name),
                    logs / f"{name}-trial-{trial}.log",
                    args.sample_size,
                    args.warm_up_time,
                    args.measurement_time,
                    args.timeout_mins * 60,
                )
                state_results.append(
                    {
                        "trial": trial,
                        "state_idx": state["idx"],
                        "state_name": name,
                        "title": state["title"],
                        **result,
                    }
                )
                (output_dir / "state_results.json").write_text(
                    json.dumps(state_results, indent=2, sort_keys=True),
                    encoding="utf-8",
                )

        by_state: dict[int, list[dict[str, object]]] = {}
        for result in state_results:
            by_state.setdefault(int(result["state_idx"]), []).append(result)

        step_results: list[dict[str, object]] = []
        for index, row in enumerate(accepted, start=1):
            prev = sorted(by_state[index - 1], key=lambda item: int(item["trial"]))
            cur = sorted(by_state[index], key=lambda item: int(item["trial"]))
            prev_values = [float(item["median_ms"]) for item in prev]
            cur_values = [float(item["median_ms"]) for item in cur]
            prev_median = float(statistics.median(prev_values))
            cur_median = float(statistics.median(cur_values))
            wins = sum(1 for before, after in zip(prev_values, cur_values) if after < before)
            sample_size_ok = all(
                (item.get("observed_sample_count") or 0) >= args.sample_size
                for item in [*prev, *cur]
            )
            step_results.append(
                {
                    "idx": index,
                    "title": row["title"],
                    "patch_file": row["patch_file"],
                    "patch_sha256": row["patch_sha256"],
                    "original_speedup": row["candidate_vs_champion_speedup"],
                    "speedup": prev_median / cur_median,
                    "wins": f"{wins}/{args.trials}",
                    "wins_count": wins,
                    "prev_median_ms": prev_median,
                    "current_median_ms": cur_median,
                    "prev_trials_ms": prev_values,
                    "current_trials_ms": cur_values,
                    "speedup_gate_ok": (prev_median / cur_median) >= 1.03,
                    "wins_gate_ok": wins >= ((args.trials // 2) + 1),
                    "sample_size_ok": sample_size_ok,
                    "state_before": states[index - 1]["name"],
                    "state_after": states[index]["name"],
                }
            )

        (output_dir / "step_results.json").write_text(
            json.dumps(step_results, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        write_tsv(output_dir / "step_results.tsv", step_results)
        summary = {
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "sample_size": args.sample_size,
            "trials": args.trials,
            "all_sample_size_ok": all(row["sample_size_ok"] for row in step_results),
            "all_speedup_gate_ok": all(row["speedup_gate_ok"] for row in step_results),
            "all_wins_gate_ok": all(row["wins_gate_ok"] for row in step_results),
            "min_speedup": min(float(row["speedup"]) for row in step_results),
            "max_speedup": max(float(row["speedup"]) for row in step_results),
            "step_count": len(step_results),
        }
        (output_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        print(json.dumps(summary, indent=2, sort_keys=True))
    finally:
        for state in states:
            remove_worktree(repo, Path(str(state["worktree"])), logs)


if __name__ == "__main__":
    main()

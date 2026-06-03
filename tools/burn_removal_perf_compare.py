#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import statistics
import subprocess
from dataclasses import dataclass
from pathlib import Path


BENCH_TARGETS = ("parameters", "benchmark")
PARAMETER_SPEEDUP_GATES = {
    "parameters/evaluate": 1.20,
    "parameters/evaluate_with_time_series_splits": 1.05,
}
NO_REGRESSION_GATES = {
    "parameters/compute_parameters": 1.05,
}
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
    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"$ {' '.join(cmd)}\n")
        log.write(f"cwd: {cwd}\n\n")
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
        result = CommandResult(124, stdout, f"{stderr}\nTIMEOUT after {timeout}s\n")

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
    result = subprocess.run(
        ["git", *args],
        cwd=repo,
        text=True,
        capture_output=True,
        check=True,
    )
    return result.stdout.strip()


def remove_worktree(repo: Path, path: Path, log_dir: Path) -> None:
    if path.exists():
        run_command(
            ["git", "worktree", "remove", "--force", str(path)],
            repo,
            log_dir / f"remove-{path.name}.log",
        )
    shutil.rmtree(path, ignore_errors=True)


def create_baseline_worktree(repo: Path, output_dir: Path, collection: Path) -> Path:
    baseline_sha = git(repo, ["rev-parse", "HEAD"])
    worktree = output_dir / "worktrees" / "baseline"
    logs = output_dir / "logs"
    remove_worktree(repo, worktree, logs)
    result = run_command(
        ["git", "worktree", "add", "--detach", str(worktree), baseline_sha],
        repo,
        logs / "add-baseline.log",
    )
    require_success(result, "create baseline worktree")
    target_collection = worktree / COLLECTION_REL
    target_collection.parent.mkdir(parents=True, exist_ok=True)
    if collection.exists() and not target_collection.exists():
        shutil.copy2(collection, target_collection)
    return worktree


def run_bench(
    worktree: Path,
    target_dir: Path,
    bench_target: str,
    sample_size: int,
    warm_up_time: int,
    measurement_time: int,
    timeout: int,
    log_path: Path,
) -> None:
    env = {
        "CARGO_INCREMENTAL": "0",
        "CARGO_TARGET_DIR": str(target_dir),
    }
    result = run_command(
        [
            "cargo",
            "bench",
            "--bench",
            bench_target,
            "--",
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
    require_success(result, f"{bench_target} benchmark in {worktree}")


def benchmark_name(criterion_root: Path, estimates_path: Path) -> str:
    relative = estimates_path.relative_to(criterion_root)
    parts = relative.parts[:-2]
    return "/".join(parts)


def collect_estimates(target_dir: Path) -> dict[str, float]:
    criterion_root = target_dir / "criterion"
    estimates: dict[str, float] = {}
    for path in criterion_root.rglob("estimates.json"):
        data = json.loads(path.read_text(encoding="utf-8"))
        estimate = data.get("median") or data.get("mean")
        if not estimate:
            continue
        estimates[benchmark_name(criterion_root, path)] = (
            float(estimate["point_estimate"]) / 1_000_000.0
        )
    return estimates


def display_name(bench_target: str, criterion_name: str) -> str:
    if criterion_name == bench_target or criterion_name.startswith(f"{bench_target}/"):
        return criterion_name
    return f"{bench_target}/{criterion_name}"


def gate_for(name: str) -> tuple[str, float]:
    if name in PARAMETER_SPEEDUP_GATES:
        return ("speedup", PARAMETER_SPEEDUP_GATES[name])
    if name in NO_REGRESSION_GATES:
        return ("no_regression", NO_REGRESSION_GATES[name])
    if name.startswith("benchmark/") or not name.startswith("parameters/"):
        return ("no_regression", 1.05)
    return ("informational", 0.0)


def compare_trials(
    baseline: dict[str, list[float]],
    current: dict[str, list[float]],
    trials: int,
) -> list[dict[str, object]]:
    rows = []
    for name in sorted(set(baseline) & set(current)):
        baseline_values = baseline[name]
        current_values = current[name]
        baseline_median = statistics.median(baseline_values)
        current_median = statistics.median(current_values)
        speedup = baseline_median / current_median
        ratio = current_median / baseline_median
        gate_kind, threshold = gate_for(name)
        if gate_kind == "speedup":
            gate_ok = speedup >= threshold
            wins = sum(
                1
                for base, cur in zip(baseline_values, current_values)
                if base / cur >= threshold
            )
        elif gate_kind == "no_regression":
            gate_ok = ratio <= threshold
            wins = sum(
                1
                for base, cur in zip(baseline_values, current_values)
                if cur / base <= threshold
            )
        else:
            gate_ok = True
            wins = sum(1 for base, cur in zip(baseline_values, current_values) if cur < base)

        rows.append(
            {
                "benchmark": name,
                "baseline_median_ms": baseline_median,
                "current_median_ms": current_median,
                "speedup": speedup,
                "current_over_baseline": ratio,
                "gate_kind": gate_kind,
                "gate_threshold": threshold,
                "wins": wins,
                "trials": trials,
                "gate_ok": gate_ok,
                "wins_gate_ok": wins >= ((trials // 2) + 1)
                if gate_kind != "informational"
                else True,
                "baseline_trials_ms": baseline_values,
                "current_trials_ms": current_values,
            }
        )
    return rows


def write_tsv(path: Path, rows: list[dict[str, object]]) -> None:
    headers = [
        "benchmark",
        "baseline_median_ms",
        "current_median_ms",
        "speedup",
        "current_over_baseline",
        "gate_kind",
        "gate_threshold",
        "wins",
        "trials",
        "gate_ok",
        "wins_gate_ok",
    ]
    with path.open("w", encoding="utf-8") as file:
        file.write("\t".join(headers) + "\n")
        for row in rows:
            file.write("\t".join(str(row[header]) for header in headers) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/burn_removal_perf"))
    parser.add_argument("--sample-size", type=int, default=30)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--warm-up-time", type=int, default=1)
    parser.add_argument("--measurement-time", type=int, default=8)
    parser.add_argument("--timeout-mins", type=int, default=60)
    parser.add_argument("--no-fail", action="store_true")
    args = parser.parse_args()

    if args.trials < 1:
        raise SystemExit("--trials must be at least 1")

    repo = Path.cwd().resolve()
    output_dir = (repo / args.output_dir).resolve()
    logs = output_dir / "logs"
    output_dir.mkdir(parents=True, exist_ok=True)
    collection = repo / COLLECTION_REL
    baseline_worktree = create_baseline_worktree(repo, output_dir, collection)
    timeout = args.timeout_mins * 60

    baseline_by_name: dict[str, list[float]] = {}
    current_by_name: dict[str, list[float]] = {}
    runs: list[dict[str, object]] = []

    for trial in range(1, args.trials + 1):
        for state, worktree, sink in [
            ("baseline", baseline_worktree, baseline_by_name),
            ("current", repo, current_by_name),
        ]:
            target_dir = output_dir / "targets" / state / "build"
            shutil.rmtree(target_dir / "criterion", ignore_errors=True)
            for bench_target in BENCH_TARGETS:
                run_bench(
                    worktree,
                    target_dir,
                    bench_target,
                    args.sample_size,
                    args.warm_up_time,
                    args.measurement_time,
                    timeout,
                    logs / f"{state}-trial-{trial}-{bench_target}.log",
                )
                runs.append(
                    {
                        "state": state,
                        "trial": trial,
                        "bench_target": bench_target,
                        "target_dir": str(target_dir),
                    }
                )
            estimates = collect_estimates(target_dir)
            criterion_archive = output_dir / "criterion" / state / f"trial-{trial}"
            shutil.rmtree(criterion_archive, ignore_errors=True)
            if (target_dir / "criterion").exists():
                shutil.copytree(target_dir / "criterion", criterion_archive)
            for name, value in estimates.items():
                bench_target = "parameters" if name.startswith("parameters/") else "benchmark"
                sink.setdefault(display_name(bench_target, name), []).append(value)
            runs.append(
                {
                    "state": state,
                    "trial": trial,
                    "bench_target": "all",
                    "target_dir": str(target_dir),
                    "criterion_archive": str(criterion_archive),
                    "estimate_count": len(estimates),
                }
            )

    rows = compare_trials(baseline_by_name, current_by_name, args.trials)
    result = {
        "baseline_sha": git(repo, ["rev-parse", "HEAD"]),
        "sample_size": args.sample_size,
        "trials": args.trials,
        "warm_up_time": args.warm_up_time,
        "measurement_time": args.measurement_time,
        "runs": runs,
        "results": rows,
        "all_gates_ok": all(row["gate_ok"] and row["wins_gate_ok"] for row in rows),
    }
    (output_dir / "perf-compare.json").write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    write_tsv(output_dir / "perf-compare.tsv", rows)

    if not args.no_fail and not result["all_gates_ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

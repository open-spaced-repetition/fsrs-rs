# FSRS-rs Verification Tools

This directory keeps the verification pieces that came out of autoresearch. It
does not generate optimization candidates, call Codex, or maintain
accepted/rejected/champion iteration state.

## Golden Verification

`run.py` can generate a release-mode golden probe from the same collection
conversion logic as `benches/parameters.rs`. The probe records:

- dataset length/checksum
- default-parameter `evaluate`
- optimized `compute_parameters`
- trained-parameter `evaluate`
- time-series evaluation
- core inference outputs

By default, golden comparison is bit-exact within numeric tolerances. Use
`--correctness-mode logloss-band --logloss-band 0.0010` for structural training
changes that intentionally alter the optimized parameter trajectory while
keeping trained LogLoss inside the accepted band.

## Run

Full verification of the current working tree:

```sh
python3 tools/autoresearch/run.py
```

Faster golden-only check:

```sh
python3 tools/autoresearch/run.py --skip-correctness --skip-bench
```

Compare against a previously saved golden:

```sh
python3 tools/autoresearch/run.py \
  --baseline-golden tmp/autoresearch/verify/runs/<timestamp>/current-golden.json \
  --correctness-mode logloss-band \
  --logloss-band 0.0010
```

The release correctness pass runs:

- `cargo fmt --check`
- `cargo clippy --all-targets -- -D warnings`
- `cargo test --release -- --skip simulation::tests::test_optimal_retention_with_old_parameters`
- `RUSTDOCFLAGS="-D warnings" cargo doc --release`

`SKIP_TRAINING=1` is set for release tests by default. Pass
`--include-training-tests` to run the original slow training test too.

## Benchmark Verification

`run.py` benchmarks `parameters/compute_parameters` with Criterion and writes
`bench-summary.json` under the output directory.

High-sample example:

```sh
python3 tools/autoresearch/run.py \
  --skip-correctness \
  --skip-golden \
  --sample-size 30 \
  --trials 3 \
  --measurement-time 8
```

## Accepted Patch Reverification

`verify_accepted.py` remains available for reproducing the high-sample
verification of a fixed list of accepted patches from
`artifacts/autoresearch_accepted_20260603/accepted_summary.json`. It only
applies existing patches and benchmarks them; it does not discover new patches.

## Data

The tools use `tests/data/collection.anki21`. If it is missing, `run.py`
downloads the original zipped sample into `tmp/autoresearch/verify/data/` and
copies it into `tests/data/`.

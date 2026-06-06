# ADR Performance Improvement

## Goal

Improve Cost ADR performance without accepting hidden quality regressions.
Every ADR performance change must be measured against both:

- Repo fixture collection: `tests/data/collection.anki21`.
- Main Profile collection: a read-only backup of
  `/Users/jschoreels/Library/Application Support/Anki2/Main Profile/collection.anki2`.

Do not write to the live Anki profile during these checks. Use a SQLite backup
copy under `/private/tmp` for Main Profile runs.

## Per-Change Gate

For each proposed performance change, record a before/after row for both
collections before accepting the change.

Required quality metric:

- LogLoss before and after the change.
- LogLoss delta versus the immediate pre-change baseline.
- Gate result. Use the existing collection gate when one is documented; when no
  gate exists for the exact harness, require no unexplained LogLoss regression
  before accepting the change.

Required performance metric:

- At least three release-mode timing samples per collection.
- Average execution time.
- Minimum execution time.
- Timing delta versus the immediate pre-change baseline.

If a run uses an ad hoc or temporary harness, record the harness path, command,
commit or diff context, and output file path in the result notes.

## Standard Inputs

Repo fixture:

```text
tests/data/collection.anki21
```

This fixture has `cards` and `revlog` tables but no `col` table. The
`examples/fsrs7_adr_collection.rs` measurement path therefore uses the same
effective day boundary as the existing fixture converter: Asia/Shanghai with the
next day starting at 04:00.

Main Profile live path, read-only source only:

```text
/Users/jschoreels/Library/Application Support/Anki2/Main Profile/collection.anki2
```

Main Profile backup target:

```text
/private/tmp/fsrs-main-profile-collection-backup.anki2
```

Use SQLite backup or an equivalent read-only copy method. If the profile is
locked, do not retry against the live database in write mode.

## Measurement Protocol

1. Build and run in release mode.
2. Run the same command shape before and after the change.
3. Use the same model version, card-id selection, simulator settings, ADR
   population, generations, seed, and early-stop settings.
4. Collect at least three timing samples for each collection and state whether
   they are cold, warm, or mixed.
5. Record LogLoss from the same evaluation path used for both before and after.
6. Keep generated profile reports and raw command output under `/private/tmp`
   when they are too noisy for this document.
7. For generation-count tradeoff checks, use the `cost_adr_generation_hypervolume`
   lines emitted by `examples/fsrs7_adr_collection.rs`. Compare each generation's
   `best_hypervolume_change_percent` against the current gen 10 quality baseline
   for both the repo fixture and Main Profile backup. Compare timing with the
   run-level `cost_adr_training_seconds`.

## Generation-Count Probes

### Generation-count probe: `gen 20 max with per-generation hypervolume`

Intent:

- Estimate whether increasing ADR generations is worth the extra optimizer work.
- Use one deterministic max-generation run per collection and read
  `CostAdrTrainingResult.history` so generation `N` means the best hypervolume
  found after completing generation `N` in the same optimizer run.

Command shape:

```text
rtk cargo run --release --example fsrs7_adr_collection -- \
  --collection tests/data/collection.anki21 \
  --out-dir /private/tmp/adr-generation-curve-gen20-repo \
  --model-version fsrs7 --pop 8 --gen 20 --seed 42

rtk cargo run --release --example fsrs7_adr_collection -- \
  --collection /private/tmp/fsrs-main-profile-collection-backup-adr.anki2 \
  --out-dir /private/tmp/adr-generation-curve-gen20-main \
  --model-version fsrs7 --pop 8 --gen 20 --seed 42
```

Repo fixture result:

| Generation | Best hypervolume | Delta | Delta vs baseline | Note |
| ---: | ---: | ---: | ---: | --- |
| 1 | `2910.197998` | `196.237793` | `7.230680%` | First candidate generation already beats baseline. |
| 5 | `2949.264160` | `235.303955` | `8.670133%` | Previous Anki/core default max before switching ADR default generations to 10. |
| 8 | `2966.151367` | `252.191162` | `9.292368%` | Improvement after default max. |
| 10 | `2985.214844` | `271.254639` | `9.994791%` | Current profiling baseline; last best improvement. |
| 14 | `2985.214844` | `271.254639` | `9.994791%` | Early stop ended the `--gen 20` run here. |

Main Profile backup result:

| Generation | Best hypervolume | Delta | Delta vs baseline | Note |
| ---: | ---: | ---: | ---: | --- |
| 1 | `253975.093750` | `-7019.171875` | `-2.689397%` | First generation can be worse than fixed-retention baseline. |
| 2 | `269517.156250` | `8522.890625` | `3.265547%` | First useful candidate in this run. |
| 5 | `269517.156250` | `8522.890625` | `3.265547%` | Previous Anki/core default max before switching ADR default generations to 10. |
| 8 | `271636.906250` | `10642.640625` | `4.077730%` | Same best value retained through gen 10. |
| 10 | `271636.906250` | `10642.640625` | `4.077730%` | Current profiling baseline. |
| 11 | `272489.281250` | `11495.015625` | `4.404317%` | Small improvement past gen 10. |
| 13 | `272841.156250` | `11846.890625` | `4.539138%` | Last best improvement. |
| 17 | `272841.156250` | `11846.890625` | `4.539138%` | Early stop ended the `--gen 20` run here. |

Interpretation:

- Repo fixture: gen 10 captures all observed quality gain in this run; later
  generations only confirmed convergence.
- Main Profile backup: gen 13 improved hypervolume by `0.461408` percentage
  points over gen 10, or about `11.3%` relative to the gen 10 hypervolume delta.
- Keep gen 10 as the timing and quality baseline for performance work. Consider
  gen 13 only if the extra quality is worth a longer optimization run on the
  real collection.

## Result Template

### Change: `<short name>`

Intent:

- `<one-line reason for the change>`

Implementation files:

- `<file list>`

Command shape:

```text
<exact command or harness description>
```

Quality and timing:

| Collection | State | LogLoss | LogLoss delta | Timing samples s | Average s | Minimum s | Performance delta | Gate | Notes |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | --- | --- |
| Repo fixture | Before | | baseline | | | | baseline | | |
| Repo fixture | After | | | | | | | | |
| Main Profile backup | Before | | baseline | | | | baseline | | |
| Main Profile backup | After | | | | | | | | |

Decision:

- `<accept/reject and why>`

## Current ADR Profiling Baseline

Initial ADR policy profiling on 2026-06-05 used the current local tree and a
release `examples/cost_adr` run. The stack profile was written to:

```text
/private/tmp/cost_adr_profile_20260605.sample.txt
```

Top sampled symbols:

| Symbol | Samples | Source |
| --- | ---: | --- |
| `fsrs7_next_interval_with_curve` | `25212` | `src/model_v7.rs` |
| `powf` | `18412` | system math |
| `CardPriorityQueue::heapify` | `9013` | `src/simulation.rs` |
| `add_fsrs7_forgetting_curve_range` | `3811` | `src/simulation/neon.rs` |
| `find_root_brent` | `2888` | `roots` |
| `stability_for_set_scalar` | `2686` | `src/model_v7.rs` |
| `CostAdrPolicyEvaluator::evaluate_retention` | `1135` | `src/cost_adr.rs` |

Interpretation: the first low-maintenance target is the existing FSRS-7 NEON
forgetting-curve range path. A full SIMD rewrite of the interval solver should
not be accepted without the per-change collection gates above, because the
solver is branchy and behavior-sensitive.

## Running Log

Append accepted and rejected experiments below using the result template. Keep
large raw outputs in `/private/tmp` and link the exact file paths from the notes.

### Change: `reuse FSRS-7 runtime curve values in NEON range accumulation`

Intent:

- Avoid rebuilding FSRS-7 curve constants from raw parameters inside the
  existing NEON `add_fsrs7_forgetting_curve_range` path.

Implementation files:

- `src/model_v7.rs`
- `src/simulation.rs`
- `src/simulation/neon.rs`

Command shape:

```text
rtk cargo run --release --example fsrs7_adr_collection -- \
  --collection tests/data/collection.anki21 \
  --out-dir /private/tmp/adr-{baseline,after}-repo-{1,2,3} \
  --model-version fsrs7 --pop 8 --gen 5 --seed 42

rtk sqlite3 "/Users/jschoreels/Library/Application Support/Anki2/Main Profile/collection.anki2" \
  ".backup '/private/tmp/fsrs-main-profile-collection-backup-adr.anki2'"

rtk cargo run --release --example fsrs7_adr_collection -- \
  --collection /private/tmp/fsrs-main-profile-collection-backup-adr.anki2 \
  --out-dir /private/tmp/adr-{baseline,after}-main-{1,2,3} \
  --model-version fsrs7 --pop 8 --gen 5 --seed 42
```

Quality and timing:

| Collection | State | LogLoss | LogLoss delta | Timing samples s | Average s | Minimum s | Performance delta | Gate | Notes |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | --- | --- |
| Repo fixture | Before | `0.19070584` | baseline | `0.943, 1.074, 1.021` | `1.013` | `0.943` | baseline | pass | `tests/data/collection.anki21`, `14290` FSRS items. |
| Repo fixture | After | `0.19070584` | `0.00000000` | `0.920, 0.930, 0.933` | `0.928` | `0.920` | `-8.4%` average, `-2.4%` min | pass | ADR hypervolume, rollout rows, and policy outputs unchanged. |
| Main Profile backup | Before | `0.48840013` | baseline | `88.885, 95.460, 91.697` | `92.014` | `88.885` | baseline | pass | SQLite backup at `/private/tmp/fsrs-main-profile-collection-backup-adr.anki2`, `183378` FSRS items. |
| Main Profile backup | After | `0.48840013` | `0.00000000` | `92.353, 86.347, 101.873` | `93.524` | `86.347` | `+1.6%` average, `-2.9%` min | mixed | ADR hypervolume, rollout rows, and policy outputs unchanged; average regressed because the third after sample was slow. |

Decision:

- Not accepted as a clear performance win yet. The repo fixture and Criterion
  benchmark improved, but the required Main Profile average did not improve in
  this three-sample gate.

Follow-up Main Profile profile on the after tree:

- Run output: `/private/tmp/adr_profile_after_main.out`.
- Sample report: `/private/tmp/adr_profile_after_main_20260605.sample.txt`.
- Command shape: release `examples/fsrs7_adr_collection`, Main Profile backup,
  FSRS-7, `pop=8`, `gen=5`, seed `42`.
- Quality and output: LogLoss `0.48840013`, ADR training seconds `89.786`,
  hypervolume rows unchanged from the A/B gate.

Top collapsed stack samples:

| Symbol | Samples |
| --- | ---: |
| `fsrs7_next_interval_with_curve` | `25960` |
| `powf` | `18925` |
| `CardPriorityQueue::heapify` | `13479` |
| `find_root_brent` | `6642` |
| `stability_for_set_scalar` | `1942` |
| `expf` | `1252` |
| `CostAdrPolicyEvaluator::evaluate_retention` | `1059` |
| `add_fsrs7_forgetting_curve_range` | `969` |

Interpretation: the existing NEON range helper is no longer a primary ADR
training target on the Main Profile backup. The next experiment should target
the scalar FSRS-7 interval solver or heap maintenance, not further NEON setup
caching.

### Change: `seed low-retention Newton interval solves from observed retention drop`

Intent:

- Reduce Brent fallback work in the scalar FSRS-7 interval solver by starting
  low-retention Newton solves from the observed retention drop between `t=0`
  and the current upper bracket instead of a fixed midpoint.

Implementation files:

- `src/model_v7.rs`

Command shape:

```text
rtk cargo run --release --example fsrs7_adr_collection -- \
  --collection tests/data/collection.anki21 \
  --out-dir /private/tmp/adr-scalar-{baseline,after}-repo-{1,2,3} \
  --model-version fsrs7 --pop 8 --gen 5 --seed 42

rtk cargo run --release --example fsrs7_adr_collection -- \
  --collection /private/tmp/fsrs-main-profile-collection-backup-adr.anki2 \
  --out-dir /private/tmp/adr-scalar-{baseline,after}-main-{1,2,3} \
  --model-version fsrs7 --pop 8 --gen 5 --seed 42
```

Quality and timing:

| Collection | State | LogLoss | LogLoss delta | Timing samples s | Average s | Minimum s | Performance delta | Gate | Notes |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | --- | --- |
| Repo fixture | Before | `0.19070584` | baseline | `1.137, 1.006, 0.914` | `1.019` | `0.914` | baseline | pass | `tests/data/collection.anki21`, `14290` FSRS items. |
| Repo fixture | After | `0.19070584` | `0.00000000` | `0.902, 1.022, 0.960` | `0.961` | `0.902` | `-5.7%` average, `-1.3%` min | pass | ADR hypervolume, rollout rows, and policy outputs unchanged. |
| Main Profile backup | Before | `0.48840013` | baseline | `87.699, 92.888, 87.339` | `89.309` | `87.339` | baseline | pass | SQLite backup at `/private/tmp/fsrs-main-profile-collection-backup-adr.anki2`, `183378` FSRS items. |
| Main Profile backup | After | `0.48840013` | `0.00000000` | `82.517, 81.340, 91.753` | `85.203` | `81.340` | `-4.6%` average, `-6.9%` min | mixed | ADR rollout metrics changed on Main Profile backup: baseline hypervolume `260994.265625 -> 260921.093750`, best hypervolume `269517.156250 -> 269600.187500`, delta `8522.890625 -> 8679.093750`. |

Decision:

- Rejected and reverted. The timing result was positive and LogLoss stayed
  unchanged, but existing Cost ADR solver precedent rejects interval-solver
  tweaks that change default-policy rollout rows. A broader observed-drop seed
  variant was also rejected earlier in the local loop because it failed
  `test_fsrs7_interval_matches_f64_bisection_reference`.

### Change: `exact per-simulation scheduled interval cache`

Intent:

- Reduce the number of `fsrs7_next_interval_with_curve` calls by caching the
  scheduled interval for exact `(stability, desired_retention)` bit patterns
  within each simulation run.

Implementation files:

- `src/simulation.rs`

Command shape:

```text
rtk cargo run --release --example fsrs7_adr_collection -- \
  --collection tests/data/collection.anki21 \
  --out-dir /private/tmp/adr-interval-cache-after-repo-{1,2,3} \
  --model-version fsrs7 --pop 8 --gen 5 --seed 42

rtk cargo run --release --example fsrs7_adr_collection -- \
  --collection /private/tmp/fsrs-main-profile-collection-backup-adr.anki2 \
  --out-dir /private/tmp/adr-interval-cache-after-main-{1,2,3} \
  --model-version fsrs7 --pop 8 --gen 5 --seed 42
```

Quality and timing:

| Collection | State | LogLoss | LogLoss delta | Timing samples s | Average s | Minimum s | Performance delta | Gate | Notes |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | --- | --- |
| Repo fixture | Before | `0.19070584` | baseline | `1.137, 1.006, 0.914` | `1.019` | `0.914` | baseline | pass | No-cache baseline from the immediately prior scalar-seed gate. |
| Repo fixture | After | `0.19070584` | `0.00000000` | `0.751, 0.632, 0.647` | `0.677` | `0.632` | `-33.6%` average, `-30.9%` min | pass | ADR hypervolume, rollout rows, and policy outputs unchanged. |
| Main Profile backup | Before | `0.48840013` | baseline | `87.699, 92.888, 87.339` | `89.309` | `87.339` | baseline | pass | No-cache baseline from the immediately prior scalar-seed gate. |
| Main Profile backup | After | `0.48840013` | `0.00000000` | `90.277, 99.916, 93.256` | `94.483` | `90.277` | `+5.8%` average, `+3.4%` min | fail | ADR hypervolume, rollout rows, and policy outputs unchanged, but HashMap overhead outweighed exact cache hits on the real collection. |

Decision:

- Rejected and reverted. Exact interval-call reduction helps the small repo
  fixture but fails the required Main Profile timing gate. Avoid adding a
  general `HashMap` interval cache in the hot simulation loop unless a future
  profile shows much higher exact-hit reuse or a lower-overhead cache structure.

### Diagnostic: `interval bucket hit-rate and conflict stats`

Intent:

- Measure whether nearby `(stability, desired_retention)` scheduling inputs can
  be grouped into buckets to reduce `fsrs7_next_interval_with_curve` calls, and
  whether those buckets would preserve the already-computed rounded interval.

Implementation files:

- `src/simulation.rs`
- `examples/fsrs7_adr_collection.rs`
- `src/lib.rs`

Command shape:

```text
rtk cargo run --release --example fsrs7_adr_collection -- \
  --collection tests/data/collection.anki21 \
  --out-dir /private/tmp/adr-bucket-conflict-stats-repo \
  --model-version fsrs7 --pop 8 --gen 5 --seed 42 \
  --interval-bucket-stats

rtk cargo run --release --example fsrs7_adr_collection -- \
  --collection /private/tmp/fsrs-main-profile-collection-backup-adr.anki2 \
  --out-dir /private/tmp/adr-bucket-conflict-stats-main \
  --model-version fsrs7 --pop 8 --gen 5 --seed 42 \
  --interval-bucket-stats
```

Summary:

| Collection | Exact estimated hit-rate range | Tight bucket hit-rate range | Tight bucket interval miss-rate range | Notes |
| --- | ---: | ---: | ---: | --- |
| Repo fixture | `87.0%..99.1%` | `87.2%..99.2%` | `0.0%..1.0%` | The repo fixture already has high exact reuse, which is why the exact cache helped there. |
| Main Profile backup | `16.9%..66.3%` | `69.9%..93.4%` | `0.5%..3.9%` | Bucketed reuse is much higher than exact reuse, but even the tightest bucket changes rounded intervals for a non-trivial share of calls. |

Bucket candidates:

| Candidate | `log_stability_step` | `desired_retention_step` |
| --- | ---: | ---: |
| Tight | `0.001` | `0.0005` |
| Medium | `0.002` | `0.001` |
| Coarse | `0.005` | `0.002` |
| Very coarse | `0.010` | `0.005` |

Decision:

- Do not implement approximate bucketed interval caching yet. It would reduce
  calls on Main Profile, but the tightest measured bucket still has interval
  miss rates up to `3.9%`, so a cache would predictably change ADR rollouts.
  The next safer optimization target is either a lower-overhead exact cache or
  reducing solver cost per call without changing rounded interval output.

### Change: `certified FSRS-7 rounded interval cache`

Intent:

- Reduce `fsrs7_next_interval_with_curve` calls by caching rounded scheduled
  intervals for tight `(log_stability, desired_retention)` buckets, while only
  reusing a cached interval when forgetting-curve boundary checks certify that
  the current root still rounds to the cached day.

Implementation files:

- `src/simulation.rs`
- `docs/COST_ADR.MD`

Command shape:

```text
rtk cargo run --release --example fsrs7_adr_collection -- \
  --collection tests/data/collection.anki21 \
  --out-dir /private/tmp/adr-certified-cache-after-repo-{1,2,3} \
  --model-version fsrs7 --pop 8 --gen 5 --seed 42

rtk cargo run --release --example fsrs7_adr_collection -- \
  --collection /private/tmp/fsrs-main-profile-collection-backup-adr.anki2 \
  --out-dir /private/tmp/adr-certified-cache-after-main-{1,2,3} \
  --model-version fsrs7 --pop 8 --gen 5 --seed 42
```

Quality and timing:

| Collection | State | LogLoss | LogLoss delta | Timing samples s | Average s | Minimum s | Performance delta | Gate | Notes |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | --- | --- |
| Repo fixture | Before | `0.19070584` | baseline | `1.137, 1.006, 0.914` | `1.019` | `0.914` | baseline | pass | No-cache baseline from the immediate pre-cache gates. |
| Repo fixture | After | `0.19070584` | `0.00000000` | `0.844, 0.987, 0.886` | `0.906` | `0.844` | `-11.1%` average, `-7.7%` min | pass | ADR hypervolume, rollout rows, and policy outputs unchanged. |
| Main Profile backup | Before | `0.48840013` | baseline | `87.699, 92.888, 87.339` | `89.309` | `87.339` | baseline | pass | No-cache baseline from the immediate pre-cache gates. |
| Main Profile backup | After | `0.48840013` | `0.00000000` | `54.214, 48.074, 49.205` | `50.498` | `48.074` | `-43.5%` average, `-45.0%` min | pass | ADR hypervolume, rollout rows, and policy outputs unchanged. |

Decision:

- Accepted. The cache is restricted to FSRS-7 simulations. Bucket hits are
  accepted only when the current forgetting curve certifies the root lies away
  from rounded-interval boundaries for the cached day; otherwise the existing
  solver runs and replaces the cache entry. A first guard variant without a
  rounding margin was rejected during unit testing because it could reuse a
  cached interval across an implementation-level rounding boundary.

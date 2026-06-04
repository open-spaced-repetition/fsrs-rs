# FSRS7 Speed Port Plan

## Source Checked

- Source repo: `/Users/jschoreels/workspace/fsrs-rs-speed-autoresearch`
- Current repo: `/Users/jschoreels/workspace/fsrs-rs`
- Source history reviewed through `fsrs-rs-speed-autoresearch` `main` at `b146ee3`.
- Primary speed evidence: `result/history.jsonl`, `fsrs-rs/src/analytic.rs`, `fsrs-rs/src/training.rs`, `compute_parameters.py`, and `benchmark.py`.
- Upstream PR reviewed: `open-spaced-repetition/fsrs-rs#411`, "Speed up FSRS inference, evaluation, and training without production Burn dependency", merged 2026-06-04.

## Future Agent Handoff

Snapshot date: `2026-06-04`.

- Current working branch when this note was written: `feature/fsrs-7-adr-optimization`.
- Baseline branch for A/B gates: `feature/fsrs-7-adr`.
- Baseline temp tree used in this run: `/private/tmp/fsrs-rs-ab-baseline.82dELe`.
- Current temp tree used in this run: `/private/tmp/fsrs-rs-ab-current.XcpJ5a`.
- The temp trees are not durable. Recreate from `feature/fsrs-7-adr` plus the current working tree if they are gone.
- `tests/data/collection.anki21` exists locally for this run but is not tracked by git. Copy it into any recreated A/B tree before running collection benchmarks.
- No live Anki profile collection was read or modified for the A/B checks below.

Current transparent groups already implemented on this branch:

- [x] Count-based outlier bucketing and shared sorted train/validation tensor dataset setup.
  - Files: `src/dataset.rs`, `src/training.rs`, `src/batch_shuffle.rs`.
  - Data-flow doc: `docs/TRAINING_DATA_FLOW.MD`.
  - This keeps the initialization dataset available; do not replace it with the speed repo's narrower return shape unless that behavior change is approved.
- [x] Scalar runtime inference/evaluation for the existing 35-parameter FSRS-7 shape.
  - Files: `src/model.rs`, `src/inference.rs`.
  - Data-flow doc: `docs/INFERENCE_DATA_FLOW.MD`.
  - `FSRS::new_with_backend()` now caches the clipped parameter vector and model version. Scalar code must use these cached values so it matches the Burn model after parameter clipping/default fill.
  - Main scalar helpers: `step_scalar()`, `forward_reviews_scalar()`, `weighted_binary_cross_entropy()`, `rmse_bins()`, and `batch_predict()`.
  - Public runtime surfaces now using scalar replay/evaluation: `memory_state`, `memory_state_batch`, `historical_memory_states`, `historical_memory_state_batch`, `current_retrievability`, `next_interval`, `interval_at_retrievability`, `next_states`, `evaluate`, `universal_metrics`, and `evaluate_with_time_series_splits` prediction.
  - `memory_state_batch()` includes the PR 411 repeated-input fast path. Keep its existing normal-path behavior and tests.
  - Important correctness nuance: do not globally change `src/model_v6.rs` short-term stability semantics. The scalar inference path has its own FSRS-6 compatibility helper, `fsrs6_stability_short_term_for_inference()`, because changing the model helper altered simulation outputs.
- [x] Optional card-id API and FSRS-7 Burn-backed expanding-window training batches.
  - Files: `src/training.rs`, `src/dataset.rs`, `src/batch_shuffle.rs`.
  - Data-flow doc: `docs/TRAINING_DATA_FLOW.MD`.
  - `ComputeParametersInput.card_ids` must be aligned with `train_set`; mismatches return `InvalidInput`.
  - This first windowed implementation still used Burn autograd. It reduced repeated prefix replay when card ids are present, but was not the full analytic/SIMD source-repo port.
  - Local Criterion after this step: `parameters/compute_parameters` `757.98 ms..777.75 ms`, no statistically significant performance change. Treat this as API/data-layout groundwork; the next speed-critical step is replacing Burn BCE/autograd for FSRS-7 training.
- [x] Scalar analytic FSRS-7 windowed BCE gradient for card-id training.
  - Files: `src/analytic_v7.rs`, `src/training.rs`.
  - Data-flow doc: `docs/TRAINING_DATA_FLOW.MD`.
  - The BCE gradient is computed from the windowed host arrays and registered
    into the existing Burn optimizer path. Existing schedule/L2 penalty
    gradients, freeze masks, Adam settings, and parameter clipping remain in the
    current training loop.
  - Local Criterion after this step: `parameters/compute_parameters`
    `266.93 ms..276.54 ms`, about `64.7%` faster than the preceding
    card-id/windowed benchmark around `767 ms`.
- [x] Scalar analytic FSRS-7 windowed validation loss for card-id training.
  - Files: `src/analytic_v7.rs`, `src/training.rs`.
  - Data-flow doc: `docs/TRAINING_DATA_FLOW.MD`.
  - Validation now uses the loss-only scalar path for FSRS-7 windowed batches.
    Prefix batches and FSRS-6 keep the existing Burn tensor validation loss.
  - Local Criterion after this step: `parameters/compute_parameters`
    `229.55 ms..231.18 ms`, about `14.9%` faster than the preceding analytic
    gradient benchmark around `270.50 ms`.
- [x] Host-side Adam for FSRS-7 windowed card-id training.
  - Files: `src/training.rs`.
  - Data-flow doc: `docs/TRAINING_DATA_FLOW.MD`.
  - The windowed path now keeps parameter updates on a host vector, with Adam
    settings matching the existing Burn optimizer: `beta_1=0.8`, `beta_2=0.85`,
    and `epsilon=1e-8`.
  - Tests compare host Adam against Burn Adam for repeated steps with clipping,
    and cover the host freeze masks.
  - Local Criterion after this step: `parameters/compute_parameters`
    `227.25 ms..228.48 ms`, Criterion change
    `[-1.5106% -1.0577% -0.6238%]`, reported within the noise threshold.
- [x] Tuned FSRS-7 windowed card-id training to learning rate `0.07` and `17`
  epochs.
  - Files: `src/training.rs`.
  - Data-flow doc: `docs/TRAINING_DATA_FLOW.MD`.
  - This is scoped to FSRS-7 callers that provide `ComputeParametersInput.card_ids`;
    FSRS-6 and prefix-item callers keep the generic `TrainingConfig` defaults.
  - Main Profile read-only production check: LogLoss `0.49010426` against gate
    `0.49013408`, optimization time `3.970 s`.
  - Local Criterion after this tuning: `parameters/compute_parameters`
    `462.80 ms..487.68 ms`, reported as a regression versus the preceding
    8-epoch host-Adam path because this intentionally runs 17 epochs.
- [x] Host-only shared FSRS-7 windowed batches plus hard-label BCE cleanup.
  - Files: `src/training.rs`, `src/analytic_v7.rs`.
  - Data-flow doc: `docs/TRAINING_DATA_FLOW.MD`.
  - The FSRS-7 card-id windowed path now builds one host-array batch vector and
    reuses it for both training and validation. Production windowed batches no
    longer store Burn tensors; Burn tensors are only created by test oracles.
  - The analytic BCE path now uses the hard-label identity so each prediction
    uses one logarithm instead of computing both keep/forget terms. The
    analytic loss and gradient loops also skip the final state update for each
    batch column because no later prediction can consume it. Exp fusion remains
    unimplemented.
  - Local Criterion after this step: `parameters/compute_parameters`
    `419.67 ms..429.12 ms`, about `11.4%` faster than the preceding
    `.07`/17-epoch benchmark around `478.47 ms`.
- [x] ARM64 NEON FSRS-7 windowed validation loss.
  - Files: `src/analytic_v7.rs`, `src/analytic_v7/neon_loss.rs`,
    `src/neon_math.rs`.
  - Data-flow doc: `docs/TRAINING_DATA_FLOW.MD`.
  - On `aarch64` targets with `neon`, FSRS-7 card-id windowed validation loss
    processes four batch columns per `float32x4_t` group. Other targets and
    ragged remainders use the scalar analytic loss.
  - Main Profile temp-copy timing, windowed-only mode:
    scalar clone at `0d531b3` optimized in `4.081 s` and `4.436 s`; NEON path
    optimized in `3.152 s`, `3.284 s`, `3.530 s`, and `3.251 s` across repeated
    runs. LogLoss remained `0.48997682` and RMSE bins remained about
    `0.04346088`.
  - Local Criterion on the small fixture remained noisy and did not detect a
    significant change: `parameters/compute_parameters` `394.91 ms..425.91 ms`,
    change `[-7.7517% +2.2041% +10.940%]`.
- [x] ARM64 NEON FSRS-7 windowed BCE gradient.
  - Files: `src/analytic_v7.rs`, `src/analytic_v7/neon_loss.rs`,
    `src/neon_math.rs`, `src/training.rs`.
  - Data-flow doc: `docs/TRAINING_DATA_FLOW.MD`.
  - Training now calls a gradient-only analytic entrypoint. On ARM64 NEON,
    four windowed batch columns are differentiated together with a native
    `float32x4_t` reverse-mode kernel. The exact scalar dual-number
    `windowed_loss_and_grad()` remains as the test oracle and non-NEON fallback.
  - Local Criterion after this step: `parameters/compute_parameters`
    `74.411 ms..75.128 ms`, Criterion change
    `[-82.392% -81.703% -80.970%]`, reported as a significant improvement.
  - Repo fixture selected path after this step: LogLoss `0.20310810`, RMSE bins
    `0.02351479`, optimization `0.074 s` on a warm scratch-harness rerun.
  - Main Profile temp-copy windowed-only timing after this step: LogLoss
    `0.48999852`, RMSE bins about `0.04345194`, optimization `0.685 s` and
    `0.688 s` on repeated warm runs.
- [x] ARM64 NEON FSRS-7 windowed gradient micro-optimizations.
  - Files: `src/analytic_v7.rs`, `src/analytic_v7/neon_loss.rs`,
    `src/neon_math.rs`.
  - Data-flow doc: `docs/TRAINING_DATA_FLOW.MD`.
  - The NEON training-gradient path now hoists the shared recurrent `ln(s)`,
    `ln(d)`, and `ln(s+1)` values per step, fuses the stability fail-path
    `d`/retrievability exponentials, skips all-padding trailing rows per
    four-lane group, and uses a faster approximate NEON `exp`/`ln` variant only
    for training-gradient recurrence math. The NEON validation-loss path keeps
    the more accurate approximation.
  - The focused scalar-oracle gradient tolerance is now relative L2 `3e-2`
    because the fast gradient path intentionally approximates the exact scalar
    recurrence. The validation-loss oracle tolerance is unchanged.
  - Local Criterion after this step, warm rerun: `parameters/compute_parameters`
    `71.247 ms..74.427 ms`, reported as a significant improvement versus the
    previous Criterion baseline.
  - Repo fixture selected path after this step: LogLoss `0.20318088`, RMSE bins
    `0.02366361`, optimization `0.071 s`, against prefix LogLoss gate
    `0.20450646`.
  - Main Profile temp-copy windowed-only timing after this step: LogLoss
    `0.48992801`, RMSE bins about `0.04356728`, optimization samples `0.657 s`,
    `0.946 s`, `0.807 s`, `0.995 s`, and `0.666 s`. Treat this as noisy but
    gate-passing; the previous accepted gate was `0.49024179`.
- [x] ADR-facing FSRS-7 NEON reuse and simulator metric range accumulation.
  - Files: `examples/fsrs7_adr_collection.rs`, `src/simulation.rs`,
    `src/simulation/neon.rs`, `src/neon_math.rs`, `benches/cost_adr.rs`,
    `Cargo.toml`.
  - Data-flow docs: `docs/COST_ADR.MD`, `docs/TRAINING_DATA_FLOW.MD`.
  - The ADR collection example now preserves aligned card ids while converting
    revlogs into FSRS prefix items and passes them to `compute_parameters()` for
    FSRS-7 parameter training. This lets ADR collection runs reuse the existing
    FSRS-7 card-id windowed NEON training path.
  - Cost ADR policy training now has a dedicated Criterion benchmark:
    `cargo bench --bench cost_adr`.
  - Simulation metric accumulation now uses an ARM64 NEON helper for FSRS-7
    `memorized_cnt_per_day` range updates. Scheduling decisions, RNG
    thresholds, priority queues, card state updates, and interval selection stay
    scalar.
  - Local sequential Criterion after this step: `cost_adr/train_single_user`
    `433.17 ms..478.47 ms`. Treat this as the first recorded baseline for the
    dedicated Cost ADR training benchmark because that benchmark did not exist
    before this pass.
  - The filtered `simulation parallel_simulate` benchmark has a real pre/post
    comparison: previous baseline `431.57 ms..485.85 ms`; after the simulator
    NEON helper `391.52 ms..434.60 ms`, Criterion change
    `[-22.953% -16.213% -8.7944%]`, reported as a significant improvement.
- [x] ADR simulator FSRS-7 runtime context and per-rollout policy evaluator.
  - Files: `src/model_v7.rs`, `src/simulation.rs`, `src/cost_adr.rs`.
  - Data-flow docs: `docs/COST_ADR.MD`, `docs/S90_RETRIEVAL.MD`,
    `docs/MODEL_SPLIT.MD`.
  - FSRS-7 simulator rollouts now reuse a transient `Fsrs7Runtime` containing
    the cached S90 LUT plus parameter-derived forgetting-curve constants.
    Public scalar helpers retain the same behavior and tests compare the
    runtime path against the scalar helper outputs.
  - Cost ADR rollouts now reuse a per-cost-weight policy evaluator with cached
    normalized cost weight, bounds transforms, and retention span.
  - Deterministic hypervolume check:
    `cargo run --release --example cost_adr -- --days 180 --deck 1500 --learn-limit 20 --review-limit 9999 --cost-limit-minutes 720 --pop 4 --gen 3 --seed 42`.
    Hypervolume stayed unchanged before/after: baseline `12719.048828`, best
    `12293.481445`, delta `-425.567383`. Training wall time moved from
    `1.233 s` to `0.998 s`.
  - Local sequential Criterion for `cost_adr/train_single_user`: previous
    accepted current-tree baseline `418.22 ms..423.67 ms`; after the runtime
    context and evaluator `366.77 ms..377.68 ms`, Criterion change
    `[-13.105% -11.864% -10.166%]`, reported as a significant improvement.
  - Follow-up `sample` profile on release `examples/cost_adr` default profile,
    sampled for `15s` after startup, showed `fsrs7_next_interval_with_curve`
    and system `powf` as the dominant active entries. The next large non-wait
    target was `PriorityQueue::heapify`, then NEON metric range accumulation,
    Brent fallback, `simulate_inner`, and `stability_for_set_scalar`.
    `CostAdrPolicyEvaluator::evaluate_retention` was much smaller after the
    evaluator pass. Profile report path for this run:
    `/private/tmp/cost_adr_profile.sample.txt`.
- [x] ADR interval-cache and simulator heap specialization.
  - Files: `src/model_v7.rs`, `src/simulation.rs`.
  - Data-flow doc: `docs/COST_ADR.MD`.
  - FSRS-7 interval solving now builds a stability-dependent curve cache once
    per interval candidate and reuses it across high-bound bracketing, Newton,
    and Brent fallback. The scalar arithmetic shape was kept stable after an
    intermediate reciprocal-multiply variant changed deterministic output by a
    tiny amount.
  - Simulator card scheduling now uses a local indexed heap specialized to card
    indices and `CardPriority`. It preserves the previous strict `>` heap
    comparison behavior, including equal-priority swap order, while removing the
    generic `PriorityQueue` hash/index-map layer that appeared in the profile.
  - Deterministic hypervolume check stayed unchanged: baseline `12719.048828`,
    best `12293.481445`, delta `-425.567383`. The same command's training wall
    time was `0.970 s` in this run.
  - Interval-cache-only Criterion after preserving scalar arithmetic:
    `cost_adr/train_single_user` `366.77 ms..378.07 ms`, change
    `[-1.9543% +0.2665% +2.2112%]`, reported as no detected performance change.
  - After the specialized simulator heap, Criterion moved to
    `349.04 ms..353.52 ms`, change `[-7.2023% -5.5660% -4.1185%]`, reported as
    a significant improvement.
  - Follow-up `sample` report `/private/tmp/cost_adr_after_heap.sample.txt`
    showed the previous `PriorityQueue::heapify` symbol removed. The local
    `CardPriorityQueue::heapify` still appears because scheduling remains a
    heap operation, but it was `6377` samples versus `12183` in the prior
    generic-queue profile. The remaining largest entries were
    `fsrs7_next_interval_with_curve`, system `powf`, Brent, and
    `stability_for_set_scalar`.
- [x] ADR FSRS-7 narrowed Brent fallback bracket.
  - Files: `src/model_v7.rs`.
  - Data-flow docs: `docs/COST_ADR.MD`, `docs/S90_RETRIEVAL.MD`.
  - Newton interval solving already maintains a valid bracket. When Newton
    reaches its iteration limit, the Brent fallback now starts from that
    narrowed `(low, high)` range instead of restarting from `0..high`.
  - Deterministic hypervolume and default-policy rollout rows stayed unchanged:
    baseline `12719.048828`, best `12293.481445`, delta `-425.567383`.
  - Criterion for `cost_adr/train_single_user`: first run after the bracket
    change `339.06 ms..345.49 ms`, significant improvement versus the
    specialized-heap baseline; follow-up run `338.82 ms..342.38 ms`, no
    detected change versus the immediately prior solver benchmark.
  - Rejected experiment: lowering `NEWTON_ITERS` from `12` to `8`. Accuracy
    tests passed and final training hypervolume stayed unchanged, but
    default-policy rollout rows changed, so the solver keeps `12` Newton steps.

Verification commands already run after the scalar group:

```bash
rtk cargo fmt --all
rtk cargo check
rtk cargo test test_scalar_memory_state_matches_burn_oracle
rtk cargo test test_scalar_next_states_matches_burn_oracle
rtk cargo test test_scalar_evaluate_matches_burn_oracle
rtk cargo test test_memory_state_batch
rtk cargo test test_historical_memory_state_batch
rtk cargo test test_simulator_memorization
rtk cargo test test_simulate_with_learn_review_limit
rtk cargo test test_simulate_with_max_ivl
rtk cargo test test_simulate_with_review_priority_fn
rtk cargo test -- --skip test_next_interval_fsrs7
```

Known suite caveat:

- `test_next_interval_fsrs7` was already failing at this checkpoint and was skipped in the full safe run.
- Observed failure: actual `[36500, 36500, 4086, 543, 115, 31, 9, 2, 1, 1]`, expected `[36500, 36500, 4092, 545, 115, 31, 9, 2, 1, 1]`.
- Do not treat that skipped test as caused by the scalar inference port unless a later investigation proves it.

A/B reproduction notes:

- Use `feature/fsrs-7-adr` as the original baseline.
- Use the current branch/worktree as the candidate.
- Copy the local fixture `tests/data/collection.anki21` into both temp trees before running collection benchmarks.
- The temporary metric harness used `examples/ab_metrics.rs` with the same revlog conversion shape as `benches/parameters.rs`, then printed fixed-parameter `evaluate()` and `evaluate_with_time_series_splits()` metrics.
- The temporary optimization harness used `examples/ab_optimize.rs`, loaded the same fixture, ran `compute_parameters(ComputeParametersInput { train_set: items.clone(), ..Default::default() })`, evaluated the optimized parameters, and printed parameter `to_bits()` for bit-level comparison.
- Durable Criterion commands used for perf checks:

```bash
rtk cargo bench --bench next_states
rtk cargo bench --bench calc_mem
rtk cargo bench --bench calc_mem_batch
rtk cargo bench --bench parameters evaluate
rtk cargo bench --bench parameters evaluate_with_time_series_splits
rtk cargo bench --bench parameters compute_parameters
```

Do not port next:

- Do not copy the speed repo's 36-parameter dual-trace model unless the public data/parameter-size migration is explicitly approved.
- Do not copy the speed repo's default parameters, recency weighting behavior, optimizer betas, or `enable_sched_penalties` default as "performance" changes.
- Do not assume card ids are always present. `ComputeParametersInput.card_ids` is optional, and callers without ids still use the prefix-item path.
- Do not remove Burn from production dependencies until the analytic FSRS-7 training path has oracle tests and loss gates.

Recommended next implementation:

1. Evaluate and port the bit-for-bit pieces from
   [open-spaced-repetition/fsrs-rs#417](https://github.com/open-spaced-repetition/fsrs-rs/pull/417)
   into the current `src/analytic_v7.rs` / `src/analytic_v7/neon_loss.rs`
   layout.
2. Profile the remaining FSRS-7 card-id optimization time after the PR #417
   pass before adding more math kernels.
3. Prefer changes that keep the current 35-parameter data contract and can be
   checked against the repo fixture plus Main Profile temp-copy gates.

## Compatibility Gate

The speed repo is not a direct file copy into this repo.

- This repo currently keeps FSRS-6 and FSRS-7 split under `src/model_v6.rs`, `src/model_v7.rs`, `src/training_v6.rs`, and `src/training_v7.rs`.
- This repo's FSRS-7 is currently a 35-parameter model and public memory state stores `stability` and `difficulty`.
- The speed repo's champion is FSRS-7-only, uses a 36-parameter dual-trace layout, and tracks `stability_fast` in addition to slow `stability` and `difficulty`.
- The speed repo also changed some training behavior, including default FSRS-7 params, recency weighting, optimizer betas, and `enable_sched_penalties` default. Treat those as behavior changes, not performance-only ports.

First decision:

- [ ] Decide whether this repo should stay on the current 35-parameter FSRS-7 layout or migrate to the speed repo's 36-parameter dual-trace layout.
- [ ] If staying 35-param, adapt the performance ideas to `src/model_v7.rs` and current parameter indices instead of copying `analytic.rs`.
- [ ] If moving to 36-param, update the model/data contract first: `MemoryState`, `MemoryStateTensors`, `VersionOps`, parameter clipping, initialization, inference docs, and compatibility tests.
- [ ] Do not port default parameter, recency weighting, Adam beta, or schedule-penalty default changes without a separate explicit decision.

## Current Repo State

Already present here:

- [x] FSRS-6 and FSRS-7 model/training split.
- [x] FSRS-7 schedule/L2 penalty scalar gradients in `src/training_v7.rs`.
- [x] FSRS-7 interval LUT plus Newton/Brent fallback documented in `docs/S90_RETRIEVAL.MD`.
- [x] Length sort uses `sort_by_cached_key`.
- [x] Outlier bucket decisions use counts instead of storing full grouped `FSRSItem` values.
- [x] Training and validation tensor batches are built from one sorted weighted item order.

Not present here:

- [x] Scalar `src/analytic_v7.rs` hand-written FSRS-7 windowed BCE gradient for
  the current 35-parameter layout.
- [x] No `wide` or `bytemuck` SIMD dependencies.
- [x] Host-only FSRS-7 windowed training batches are reused by both train and validation.
- [x] Card-id-aware O(N) expanding-window training batch layout is present for FSRS-7, host-backed.
- [x] Host-side Adam loop is present for FSRS-7 windowed card-id training.

## PR 411 Candidate Selection

PR 411 was merged against upstream's 21-parameter production FSRS path. This branch already has a 35-parameter FSRS-7 model split, scalar FSRS-7 interval helpers, and scalar FSRS-7 schedule/L2 penalty gradients. Treat PR 411 as a reference design, not as a cherry-pick source.

Selected next candidates:

- [x] Port scalar inference/evaluation first.
  - PR 411 impact: `parameters/evaluate` around `5.8x`, `evaluate_with_time_series_splits` around `10.2x`, `next_states` around `281x`, `calc_mem` around `222x`, and `calc_mem_batch` around `35x` to `59x`.
  - Why next: no parameter-size change, no stored Anki data change, and this branch already has scalar FSRS-7 helpers in `src/model_v7.rs`.
  - Scope: add scalar `FSRS` review replay, retrievability prediction, `memory_state`, `historical_memory_states`, `next_states`, and evaluation aggregation while keeping Burn oracle tests.
- [x] Port direct host batch construction for FSRS-7 card-id windowed training after scalar inference/evaluation.
  - PR 411 impact is part of the `compute_parameters` speedup path and removes tensor batch construction overhead.
  - Implemented as opt-in card-id windowed host arrays; production windowed
    batches no longer carry Burn tensors, and callers without card ids keep the
    prefix-item tensor path.
- [x] Port scalar analytic FSRS-7 BCE loss/gradient after host batches.
  - PR 411 impact: `compute_parameters` around `3.0x` without `card_ids`.
  - Implemented as a 35-param scalar dual-number gradient in `src/analytic_v7.rs`
    for FSRS-7 windowed card-id batches.
- [x] Port host-side Adam only after analytic gradients are proven.
  - Keep this branch's current optimizer hyperparameters: `beta_1=0.8`, `beta_2=0.85`, `epsilon=1e-8`.
  - Do not copy PR 411/default Burn-free optimizer behavior if it changes training output outside the A/B gates.

Defer for now:

- [ ] Full Burn removal from production dependencies.
  - Reason: broad crate-level change; take it only after scalar inference and analytic training have matching oracle tests.
- [x] Optional `ComputeParametersInput.card_ids`.
  - Implemented as an opt-in API. Callers without card ids keep the prefix-item path.
- [ ] Simulation workload estimator flat-matrix/cache rewrite.
  - Reason: useful for ADR/simulation, but not the next biggest transparent Anki-facing win.
- [x] ARM64 NEON SIMD and fast gradient transcendental approximations.
  - Validation loss and training gradient are implemented for the current
    35-parameter FSRS-7 card-id windowed layout. The fast approximation is
    limited to the training-gradient recurrence and remains behind scalar
    correctness and loss gates.
- [ ] Standalone BCE clamping change.
  - Reason: this branch already clamps retrievability in `forward_classification`; revisit only when adding a new scalar BCE implementation.

Latest scalar inference/evaluation A/B against `feature/fsrs-7-adr`:

- [x] LogLoss gate passed.
  - Fixed FSRS-7 evaluation LogLoss: baseline `0.23606718`, scalar `0.23606756`, delta `+0.00000038`.
  - Time-series split LogLoss: baseline `0.19534561`, scalar `0.19534674`, delta `+0.00000113`.
  - Both deltas are far inside the absolute `+/-0.0015` acceptance gate.
- [x] Performance gate passed for direct inference/evaluation surfaces.
  - `benchmark/next_states`: `69.386 us` -> `316.00 ns`, about `219.6x` faster.
  - `benchmark/calc_mem` with 1000 cards and 10 reviews: `174.56 ms` -> `704.67 us`, about `247.7x` faster.
  - `benchmark/calc_mem_batch` repeated inputs:
    - 1000 cards, 10 reviews: `1.1051 ms` -> `30.739 us`, about `35.9x` faster.
    - 1000 cards, 100 reviews: `6.1600 ms` -> `121.50 us`, about `50.7x` faster.
    - 10000 cards, 200 reviews: `85.669 ms` -> `1.8189 ms`, about `47.1x` faster.
  - `parameters/evaluate`: `55.786 ms` -> `6.4348 ms`, about `8.7x` faster.
  - `parameters/evaluate_with_time_series_splits`: `2.0742 s` -> `2.0590 s`, effectively unchanged because training still dominates this workflow.

Collection optimization A/B against `feature/fsrs-7-adr`:

- [x] Optimizing the local fixture collection preserves the training output.
  - Fixture: `tests/data/collection.anki21`, converted with the same prepared prefix-item path as `benches/parameters.rs`.
  - Prepared items: `14290`.
  - Optimized parameter count: baseline `35`, scalar `35`.
  - Optimized parameter bits: identical between baseline and scalar branch.
  - Optimized evaluation LogLoss: baseline `0.20300588`, scalar `0.20300646`, delta `+0.00000058`.
  - Optimized evaluation RMSE bins: baseline `0.02412599`, scalar `0.02412599`, delta `0.00000000`.
- [x] `compute_parameters()` performance is effectively unchanged after the scalar inference/evaluation group.
  - Direct single run: baseline `0.721577 s`, scalar `0.744507 s`.
  - Criterion `parameters/compute_parameters`: baseline `708.67 ms`, scalar `711.11 ms`, about `0.3%` slower and within noise.
  - Optimized-parameter `evaluate()` after training still benefits from the scalar runtime path: `0.076058 s` -> `0.009268 s`, about `8.2x` faster.
  - Interpretation: this first group is a production inference/evaluation win, but the optimization loop remains Burn-backed. The next meaningful `compute_parameters()` speed work is direct host batch construction, then an analytic FSRS-7 BCE/gradient path.
- [x] Card-id windowed Burn batches alone did not materially improve `compute_parameters()`.
  - Criterion after adding optional `card_ids` and FSRS-7 windowed Burn batches:
    `757.98 ms..777.75 ms`, no statistically significant performance change.
  - Interpretation: grouping prefixes by card removed repeated sequence layout at
    the data level, but Burn BCE/autograd still dominated the training loop.
- [x] Scalar analytic FSRS-7 windowed BCE gradient improved `compute_parameters()`.
  - Criterion after replacing Burn autograd for windowed BCE:
    `266.93 ms..276.54 ms`.
  - Criterion reported `change: [-65.477% -64.737% -63.740%]`,
    `p = 0.00 < 0.05`, so this step is a statistically significant speedup
    versus the preceding card-id/windowed benchmark.
  - Current fixture optimized evaluation remains inside the LogLoss gate:
    prefix path `0.20300646`, windowed card-id analytic path `0.20369799`,
    delta `+0.00069153`, below the absolute `+/-0.0015` gate.
  - Current fixture RMSE bins: prefix path `0.02412599`, windowed card-id
    analytic path `0.02458172`.
  - Interpretation: the current biggest accepted collection-optimization win is
    replacing Burn BCE/autograd inside FSRS-7 card-id windowed training. Burn
    validation, Adam, and clipping still remain.
- [x] Scalar analytic FSRS-7 windowed validation loss further improved `compute_parameters()`.
  - Criterion after replacing Burn validation loss for windowed batches:
    `229.55 ms..231.18 ms`.
  - Criterion reported `change: [-16.792% -14.862% -13.624%]`,
    `p = 0.00 < 0.05`, so this step is a statistically significant speedup
    versus the preceding analytic-gradient benchmark.
  - Current fixture optimized evaluation remains inside the LogLoss gate:
    prefix path `0.20300646`, windowed card-id analytic path `0.20369799`,
    delta `+0.00069153`, below the absolute `+/-0.0015` gate.
  - Current fixture RMSE bins: prefix path `0.02412599`, windowed card-id
    analytic path `0.02458171`.
  - Interpretation: Burn BCE/autograd and Burn validation are no longer the
    windowed FSRS-7 bottleneck.
- [x] Host-side Adam for FSRS-7 windowed training was correct but not a major speed win.
  - Criterion after replacing Burn Adam/gradient registration for windowed
    batches: `227.25 ms..228.48 ms`.
  - Criterion reported `change: [-1.5106% -1.0577% -0.6238%]`,
    `p = 0.00 < 0.05`, but classified it as within the noise threshold.
  - Current fixture optimized evaluation remains inside the LogLoss gate:
    prefix path `0.20300646`, windowed card-id host-Adam path `0.20369796`,
    delta `+0.00069150`, below the absolute `+/-0.0015` gate.
  - Current fixture RMSE bins: prefix path `0.02412599`, windowed card-id
    host-Adam path `0.02458172`.
  - Interpretation: after analytic gradient and validation, Burn Adam
    integration was no longer a large fraction of collection optimization time.
    The next practical speed target is removing Burn tensor construction and
    epoch-sync overhead from the windowed path, then considering scalar hoists or
    SIMD only with fresh benchmark evidence.

Main Profile read-only timing check:

- Collection: `/Users/jschoreels/Library/Application Support/Anki2/Main Profile/collection.anki2`.
- Data access: opened read-only for timing/logloss experiments; no Anki profile
  data was written or modified.
- Prepared items: `150042`.
- Gate: prefix LogLoss `0.48863408` plus `0.0015` = `0.49013408`.
- Timings are release-mode optimize seconds from the scratch harness. The
  temporary epoch/LR sweeps used `/private/tmp/fsrs-rs-epoch-experiment`; the
  final production row used this working tree directly.

Baseline and selected production rows:

| Path | LR | Epochs | LogLoss | RMSE bins | Optimize s | Gate | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| Prefix-item path | `0.02` | `8` | `0.48863408` | `0.03794831` | `25.930` | baseline | No card ids. |
| Windowed host Adam default | `0.02` | `8` | `0.49103013` | `0.04523148` | `2.043` | miss | About `12.7x` faster than prefix, but outside gate. |
| Windowed selected temporary run | `0.07` | `17` | `0.49010426` | not recorded | `3.885` | pass | Selected after LR sweep. |
| Windowed selected production check | `0.07` | `17` | `0.49010426` | `0.04418439` | `3.970` | pass | About `6.71x` faster than production prefix timing `26.627 s`. |
| Windowed host-only/BCE/skip-last check | `0.07` | `17` | `0.48997682` | `0.04346088` | `3.917..3.939` | pass | Temp copy of live collection after live DB lock; prepared items `150240`. |

Default-LR epoch-count sweep:

| LR | Epochs | LogLoss | Optimize s | Gate |
| ---: | ---: | ---: | ---: | --- |
| `0.02` | `10` | `0.49099839` | `2.496` | miss |
| `0.02` | `12` | `0.49092907` | `2.813` | miss |
| `0.02` | `16` | `0.49070975` | `3.699` | miss |
| `0.02` | `19` | `0.49029607` | `4.470` | miss |
| `0.02` | `20` | `0.49013489` | `4.687` | miss by `0.00000081` |
| `0.02` | `21` | `0.48998484` | `4.845` | pass |
| `0.02` | `24` | `0.48996866` | `6.084` | pass |

Lower and medium LR sample trials:

| LR | Epochs | LogLoss | Optimize s | Gate |
| ---: | ---: | ---: | ---: | --- |
| `0.015` | `8` | `0.49116427` | `1.965` | miss |
| `0.01` | `8` | `0.49091634` | `1.967` | miss |
| `0.005` | `8` | `0.49112746` | `2.282` | miss |
| `0.01` | `12` | `0.49084994` | `2.933` | miss |
| `0.01` | `16` | `0.49104741` | `3.840` | miss |
| `0.015` | `16` | `0.49084669` | `3.865` | miss |
| `0.015` | `20` | `0.49030071` | `4.650` | miss |
| `0.025` | `16` | `0.49059564` | `3.909` | miss |
| `0.025` | `20` | `0.49000537` | `4.865` | pass |

First measured passing epoch by LR:

| LR | First pass epoch | LogLoss | Optimize s | Notes |
| ---: | ---: | ---: | ---: | --- |
| `0.005` | none through `64` | `0.49015647` | `11.422` | Best near measured region was 50 epochs; 64 epochs worsened to `0.49018419` at `14.271 s`. |
| `0.01` | `24` | `0.49011192` | `5.673` |  |
| `0.015` | `22` | `0.49002644` | `5.079` |  |
| `0.02` | `21` | `0.48998484` | `4.845` | Default LR first pass. |
| `0.025` | `20` | `0.49000537` | `4.865` |  |
| `0.03` | `19` | `0.49008954` | `4.379` |  |
| `0.035` | `18` | `0.49012130` | `4.187` |  |
| `0.04` | `18` | `0.49007329` | `4.127` | Fastest before testing 17-epoch high LR. |
| `0.045` | `18` | `0.49002561` | `4.229` |  |
| `0.05` | `18` | `0.48997220` | `4.196` | 17 epochs missed. |
| `0.06` | `18` | `0.48999673` | `4.142` | 17 epochs missed. |
| `0.068` | `17` | `0.49012408` | `3.878` | Fastest passing measured row, but very close to the gate. |
| `0.07` | `17` | `0.49010426` | `3.885` | Selected production setting. |
| `0.08` | `18` | `0.48996177` | `4.223` | 17 epochs missed. |
| `0.09` | `17` | `0.49000296` | `3.888` |  |
| `0.10` | `17` | `0.49004012` | `4.054` |  |
| `0.2` | `17` | `0.48987183` | `3.944` | Higher LogLoss margin, slightly slower. |
| `0.3` | `17` | `0.48958799` | `4.770` | Slower. |
| `0.4` | `17` | `0.48946327` | `4.071` | Better 17-epoch LogLoss but unstable at 16 epochs. |

High-LR edge trials:

| LR | Epochs | LogLoss | Optimize s | Gate |
| ---: | ---: | ---: | ---: | --- |
| `0.05` | `17` | `0.49016339` | `4.150` | miss |
| `0.05` | `18` | `0.48997220` | `4.196` | pass |
| `0.06` | `17` | `0.49020690` | `3.973` | miss |
| `0.06` | `18` | `0.48999673` | `4.142` | pass |
| `0.065` | `17` | `0.49014989` | `3.872` | miss |
| `0.068` | `17` | `0.49012408` | `3.878` | pass |
| `0.069` | `17` | `0.49033660` | `4.157` | miss |
| `0.07` | `16` | `0.49032336` | `3.686` | miss |
| `0.07` | `17` | `0.49010426` | `3.885` | pass |
| `0.08` | `16` | `0.49054474` | `3.672` | miss |
| `0.08` | `17` | `0.49016225` | `3.877` | miss |
| `0.08` | `18` | `0.48996177` | `4.223` | pass |
| `0.09` | `16` | `0.49026743` | `3.724` | miss |
| `0.09` | `17` | `0.49000296` | `3.888` | pass |
| `0.10` | `16` | `0.49019909` | `3.675` | miss |
| `0.10` | `17` | `0.49004012` | `4.054` | pass |
| `0.2` | `16` | `0.49025652` | `3.736` | miss |
| `0.2` | `17` | `0.48987183` | `3.944` | pass |
| `0.3` | `16` | `0.49039635` | `3.843` | miss |
| `0.3` | `17` | `0.48958799` | `4.770` | pass |
| `0.4` | `16` | `0.49151039` | `3.930` | miss |
| `0.4` | `17` | `0.48946327` | `4.071` | pass |

Interpretation:

- Lowering the learning rate did not recover the LogLoss gate faster than the
  default-lr run on this collection.
- Raising the learning rate helped because the cosine schedule still anneals
  from the larger initial LR and clipping constrains parameters.
- The curve is not monotonic: lr `0.068` was the fastest measured pass, lr
  `0.069` missed, and very high LR passed at 17 epochs but showed instability
  at 16 epochs.
- The selected production setting is lr `0.07` for 17 epochs because it passed
  the same gate with LogLoss `0.49010426`, had essentially the same runtime as
  lr `0.068`, and is scoped only to FSRS-7 card-id windowed optimization.

Follow-up host-only/BCE/skip-last timing:

- The live profile database was locked during the follow-up timing, so the
  collection and its WAL/SHM sidecars were copied to
  `/private/tmp/fsrs-main-profile-collection.anki2*` and opened from there.
- Temp-copy prefix path: prepared items `150240`, LogLoss `0.48874179`, RMSE
  bins `0.03781221`, optimization `27.966 s`.
- Temp-copy windowed path after host-only batches, hard-label BCE, and skip-last:
  LogLoss `0.48997682`, RMSE bins `0.04346088`, windowed-only optimization
  `3.917 s` and `3.939 s` on repeated runs.
- Temp-copy gate: prefix LogLoss `0.48874179` plus `0.0015` = `0.49024179`;
  the windowed result remains inside the gate.

Follow-up ARM64 NEON timing:

- A direct `wide::f32x8`/`wide::f32x4` exp probe on Apple Silicon was slower
  than scalar `f32::exp`, so the accepted path uses the native ARM64
  `float32x4_t` NEON primitive instead of adding `wide`/`bytemuck`.
- The first NEON path was validation-loss-only and changed only best-epoch
  selection input through a f32 polynomial `exp`/`ln` approximation. It did not
  materially move the small Criterion fixture, but it improved the larger Main
  Profile timing.
- The follow-up NEON gradient path differentiates four windowed batch columns at
  a time and keeps the scalar dual-number gradient as the oracle/fallback. This
  moved both the repo fixture and the larger Main Profile timing.

Repo fixture baseline/selected production check after the ARM64 NEON commit:

- Collection: `tests/data/collection.anki21`.
- Command shape: scratch release harness with mode `both`, which runs the
  prefix-item baseline without card ids and the selected FSRS-7 windowed path
  with card ids.
- Prefix baseline: prepared items `14290`, LogLoss `0.20300646`, RMSE bins
  `0.02412599`, optimization `0.754 s`.
- Windowed selected path: prepared items `14290`, LogLoss `0.20313601`, RMSE
  bins `0.02363395`, optimization `0.370 s`.
- Repo-fixture gate: prefix LogLoss `0.20300646` plus `0.0015` =
  `0.20450646`; the windowed selected path remains inside the gate.

Repo fixture selected production check after the ARM64 NEON gradient:

- Collection: `tests/data/collection.anki21`.
- Prefix baseline: prepared items `14290`, LogLoss `0.20300646`, RMSE bins
  `0.02412599`, optimization `0.756 s` on the warm scratch-harness rerun.
- Windowed selected path: prepared items `14290`, LogLoss `0.20310810`, RMSE
  bins `0.02351479`, optimization `0.074 s`.
- Repo-fixture gate: prefix LogLoss `0.20300646` plus `0.0015` =
  `0.20450646`; the windowed selected path remains inside the gate.

Main Profile checkpoint rerun on the 2026-06-04 snapshot:

- Collection source:
  `/Users/jschoreels/Library/Application Support/Anki2/Main Profile/collection.anki2`.
- Data access: the live collection was copied with SQLite `.backup` to
  `/private/tmp/fsrs-main-profile-collection-backup.anki2`; benchmark commands
  read only that temporary backup. No Anki profile data was written or modified.
- Snapshot validation: `revlog` rows with review ease `1..=4`: `248819`;
  prepared FSRS items: `183178`.
- Command shape: temporary release harness using the same collection conversion
  logic as `examples/fsrs7_adr_collection.rs`, then `compute_parameters()` with
  FSRS-7, short-term enabled, schedule penalties enabled, and either no
  `card_ids` for the prefix baseline or aligned `card_ids` for the windowed
  path. The harness reports `optimize_seconds`, so cold worktree compilation is
  excluded from the timing table.
- Gate: prefix LogLoss `0.48780313` plus `0.0015` = `0.48930313`.
- The first row after the prefix baseline combines the earlier accepted
  analytic gradient, scalar validation, and host Adam work because those were
  not all retained as separate durable checkpoints in this branch history.

| Checkpoint / path | Optimize s | LogLoss | RMSE bins | Gate | Step gain vs previous row | Cumulative gain vs prefix | Speedup vs prefix |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| Prefix-item baseline, current tree | `28.628651` | `0.48780313` | `0.03665148` | baseline | baseline | `0.0%` | `1.0x` |
| `7bfc8ee` windowed training | `2.367539` | `0.48982984` | `0.04184988` | miss by `0.000527` | `91.7%` | `91.7%` | `12.1x` |
| `cd6d3d0` LR `0.07`, 17 epochs | `4.757196` | `0.48816797` | `0.03709807` | pass | `-100.9%` | `83.4%` | `6.0x` |
| `0d531b3` host-only/BCE/skip-last | `4.627972` | `0.48816797` | `0.03709807` | pass | `2.7%` | `83.8%` | `6.2x` |
| `18125f9` NEON validation | `3.830542` | `0.48816797` | `0.03709807` | pass | `17.2%` | `86.6%` | `7.5x` |
| `245c2f3` first NEON gradient | `0.778492` | `0.49151888` | `0.04087845` | miss by `0.002216` | `79.7%` | `97.3%` | `36.8x` |
| `aed38e2` NEON gradient micro-optimizations | `0.761803` | `0.48805517` | `0.03721301` | pass | `2.1%` | `97.3%` | `37.6x` |
| Current dirty tree, padding plus scalar validation penalty/init work | `0.524652` | `0.48805487` | `0.03720748` | pass | `31.1%` | `98.2%` | `54.6x` |

Interpretation:

- The largest acceptable jump is still moving from the prefix-item path to the
  optimized windowed path: the current tree is `54.6x` faster than prefix on
  this snapshot while staying inside the LogLoss gate.
- The LR/epoch tuning intentionally gives back runtime versus the 8-epoch
  checkpoint because the 8-epoch row misses the quality gate on this collection.
- The first NEON-gradient checkpoint is a useful speed signal but not an
  acceptable endpoint here; the later micro-optimization checkpoint restores the
  gate while keeping the NEON-gradient speed.
- This table should not be compared directly against the older Main Profile
  rows above, which used a smaller temporary copy with about `150k` prepared
  items.

## Implementation Checklist

### 1. Baseline And Gates

- [ ] Add a local FSRS-7 performance baseline for `compute_parameters()` before code changes.
- [ ] Record correctness baseline: average log loss, parameter count, finite-parameter check, and current tests passing.
- [ ] Run A/B testing against the current `feature/fsrs-7-adr` branch before accepting each ported speed change.
- [ ] Define acceptance gates before each speed port:
  - [ ] No API/data-shape change without a doc update in `docs/*.MD`.
  - [ ] Unit tests cover every new data path.
  - [ ] Performance result reported against the previous accepted step, not only against the original baseline.
  - [ ] Math-preserving changes must produce identical A/B results.
  - [ ] Precision-trading changes must keep aggregate log loss within absolute `+/-0.0015` of the original baseline.
  - [ ] For the known `compute_parameters.py` baseline, mean LogLoss must stay in `[0.3083, 0.3113]`.
  - [ ] Any inexact math change gets this loss/regression gate, not only a speed gate.

### 2. Low-Risk Data Setup Work

- [x] Adapt the speed repo's count-based outlier bucketing without dropping this repo's initialization dataset.
  - Source: Phase 2 iter2, `compute_outlier_removed_pairs()`.
  - Keep this repo's returned initialization set unless a separate behavior change is approved.
  - Tests: current `filter_outlier` count tests, same-day-only test, fractional bucket test, plus a test proving filtered initialization data is still available.
- [x] Build train and validation tensor datasets from one sorted weighted item order.
  - This is a transparent setup trim before the larger host-array/analytic-kernel port.
  - Tests: dataloader and training tests should continue to cover batch shape/order and train/valid behavior.
- [x] Build host batches directly from `WeightedFSRSItem` for FSRS-7 card-id windowed training.
  - Source: Phase 1 iter12 and iter19.
  - Keep FSRS-6 on the existing Burn path unless there is a separate FSRS-6 performance task.
  - Tests: windowed oracle covers prefix-vs-windowed loss/gradient for a single
    card, plus card-id length rejection and preprocessing alignment.
  - Remaining tests: broader host batch layout coverage for padding, skipped
    middle prefixes, and batch ordering.
- [x] Reuse one immutable host-batch vector for train and validation when `train_set == test_set`.
  - Current `compute_parameters()` and `benchmark()` pass identical weighted sets.
  - Tests: existing windowed loss/gradient and validation-loss oracles cover the
    shared host batch layout against Burn tensors constructed only in tests.

### 3. Analytic FSRS-7 BCE Kernel

- [x] Add a decoupled analytic FSRS-7 module instead of one very large file.
  - Suggested split: `analytic_v7/math.rs`, `analytic_v7/scalar.rs`, `analytic_v7/simd.rs`, `analytic_v7/window.rs`, or equivalent local style.
- [x] Port or adapt scalar forward and reverse-mode backward for the current FSRS-7 recurrence.
  - Source: Phase 1 iter2, iter5, iter8, iter10, iter13.
  - Tests: analytic loss/grad matches Burn autograd on small synthetic batches within a defined tolerance.
- [x] Keep L2 and schedule penalty gradients composed outside the BCE kernel.
  - Tests: existing penalty-gradient tests still pass.
- [ ] Add logging around selected training path at debug/info level only if it helps support: model version, analytic path enabled, batch count, and interruption.

### 4. ARM64 NEON / Portable SIMD

- [x] Avoid `wide`/`bytemuck` for the ARM64 paths; use native NEON.
  - Source: Phase 1 iter14, iter15, iter17, iter21, iter23, iter24.
  - Rationale: Apple Silicon probe showed the `wide` exp primitive was slower
    than scalar on this host.
- [x] Port ARM64 NEON `float32x4_t` validation forward for the current
  35-parameter windowed loss.
  - Tests: NEON windowed loss matches scalar analytic loss within tolerance,
    including a full 4-lane group plus scalar ragged remainder.
- [x] Port ARM64 NEON `float32x4_t` gradient forward/backward for the current
  35-parameter windowed loss.
  - Tests: NEON windowed gradient matches scalar analytic gradient within
    tolerance, including a full 4-lane group plus scalar ragged remainder.
- [ ] Port portable/x86 SIMD `exp8`/`ln8` with explicit tests for approximation bounds.
- [ ] Port portable/x86 SIMD gradient forward/backward.
  - Tests: SIMD grad matches scalar analytic grad and finite-difference gross-error guard.
- [ ] Keep precision-trade variants compile-time selected and clearly documented.
  - Do not enable the cruder `FAST` path until loss gates are in place.

### 5. O(N) Expanding-Window Training

- [x] Decide whether `ComputeParametersInput` should accept optional card ids.
  - Source: Phase 1 iter18 and `compute_parameters.py`.
  - Accepted and documented in `docs/TRAINING_DATA_FLOW.MD`.
- [x] If accepted, carry card ids through normalization, outlier filtering, recency weighting, and max-sequence filtering without reordering bugs.
  - Tests: mismatched card-id length rejection, preprocessing alignment, and windowed loss/gradient oracle.
- [x] Group all surviving prefix-items for a card into one mini-batch.
  - Source: `group_cards_into_batches()` and `build_batch_host_windowed()`.
  - Host-backed grouping is implemented. Remaining host/SIMD grouping tests
    should cover prediction count, filtered middle prefixes with weight `0`,
    and padding columns.
- [ ] Port windowed gradient and validation kernels.
  - Source: `card_loss_and_grad_simd()` and `card_loss_simd()`.
  - Gradient status: scalar analytic windowed BCE gradient is implemented.
  - Validation status: scalar analytic windowed BCE validation loss is implemented.
  - Remaining status: SIMD kernels are not implemented.
  - Tests: windowed scalar gradient equals sum of per-prefix gradients, SIMD window grad matches scalar window oracle, validation loss matches scalar oracle.
- [x] Add last-step curve-only optimization after the windowed path is correct.
  - Source: iter24 validation skip-last and iter25 gradient skip-last.
  - Tests: windowed scalar loss/gradient still match the Burn oracle after
    skipping the final unused state update.

### 6. Host-Side Optimizer

- [x] Port host-side Adam only after the analytic host gradient path is stable.
  - Source: Phase 1 iter22.
- [x] Reproduce this repo's current optimizer hyperparameters unless a behavior change is approved.
  - Current repo uses `beta_1=0.8`, `beta_2=0.85`, `epsilon=1e-8`.
  - Speed repo used `beta_1=0.55`, `beta_2=0.955555`, `epsilon=1e-8`.
- [x] Keep parameter clipping version-aware.
  - Tests: one-step optimizer comparison against Burn Adam for current hyperparameters, freeze-initial-stability behavior, freeze-short-term behavior, and parameter clip bounds.

### 7. Micro-Optimizations To Layer After Correctness

- [x] Unified BCE identity `-ln(1 - |label - r|)` for hard labels only.
  - Source: iter20 and Phase 2 iter2.
  - Tests: scalar BCE value and dual BCE gradient match the hard-label
    reference for labels `0` and `1`.
- [ ] Port PR #417 bit-for-bit windowed analytic-gradient optimizations.
  - Source:
    [open-spaced-repetition/fsrs-rs#417](https://github.com/open-spaced-repetition/fsrs-rs/pull/417),
    `perf: speed up compute_parameters (windowed analytic gradient)`.
  - PR #417 is open as of 2026-06-04 and targets the card-grouped
    `card_ids` analytic-gradient path that dominates `compute_parameters`,
    building on the analytic gradient from PR #411.
  - Candidate pieces to adapt to this repo's current 35-parameter
    `src/analytic_v7.rs` shape:
    - Hoist parameter-only transcendentals out of per-step recurrence:
      forgetting-curve factors, success/failure prefactors, and easy-rating
      initial difficulty.
    - Evaluate only the selected stability branch in the forward step instead
      of computing success, lapse, and same-day branches before selecting one.
    - Stop each card column at the first trailing padding rating `0`.
    - Skip the forgetting-curve `powf` for same-day reviews where
      `delta_t == 0` and retrievability is exactly `1`.
  - PR-reported benchmark on a roughly 600k-review collection:
    `compute_parameters` windowed path about `-22%`, and
    `compute_parameters_without_card_ids` about `-30%`.
  - Local pre-port profile baseline, collected on 2026-06-04 with
    `rtk cargo bench --bench parameters compute_parameters -- --profile-time 25 --noplot`
    plus macOS `sample` for 18 seconds:
    `/private/tmp/fsrs_compute_parameters.sample.txt`.
    Top symbol sample counts:
    - `analytic_v7::neon_loss::windowed_grad`: `1830`
    - `analytic_v7::windowed_loss`: `1638`
    - `training_v7::Dual35::pow`: `809`
    - `training_v7::fsrs7_fc_r_dual`: `488`
    - `training_v7::fsrs7_next_s_good_dual`: `472`
    - `parameter_initialization_fsrs7::loss_with_curve`: `454`
    - `training_v7::maybe_schedule_penalty_value_and_grad`: `104`
    - `training::build_windowed_batch`: `17`
    Interpretation: optimize the windowed gradient and validation-loss
    recurrence first, then revisit the schedule-penalty dual path. Batch
    construction is not currently a meaningful hotspot.
  - Partial local port result, 2026-06-04:
    - Implemented same-day forgetting-curve skips for scalar/dual analytic
      loss and all-lane NEON same-day rows.
    - Implemented trailing-padding stop for scalar columns, NEON validation
      groups, and NEON scalar remainder columns.
    - Hoisted NEON validation recurrence logs (`ln(s)`, `ln(d)`, `ln(s+1)`)
      and the long/short success prefactors already held in the per-call
      `Params`.
    - Did not port upstream's active-branch-only forward recurrence directly:
      this branch's current FSRS-7 recurrence blends long and short stability
      branches, so skipping one branch is not behavior-preserving here.
    - First Criterion run for `parameters/compute_parameters` after the partial port:
      `68.600 ms..69.207 ms`, change `[-7.9665% -3.7143% +0.6230%]`,
      reported as no detected performance change.
    - Final current-tree Criterion rerun after cleanup:
      `69.713 ms..72.647 ms`, change `[-2.1824% +0.7379% +3.3625%]`,
      also reported as no detected performance change.
    - Follow-up `sample` report:
      `/private/tmp/fsrs_compute_parameters_after_pr417_pass.sample.txt`.
      Top symbol sample counts remained concentrated in
      `analytic_v7::neon_loss::windowed_grad` (`1963`) and
      `analytic_v7::windowed_loss` (`1557`). The next visible target is the
      FSRS-7 schedule-penalty dual path: `training_v7::Dual35::pow` (`848`),
      `training_v7::fsrs7_fc_r_dual` (`532`), and
      `training_v7::fsrs7_next_s_good_dual` (`527`).
  - Schedule-penalty dual constant hoist, 2026-06-04:
    - Files: `src/training_v7.rs`, `docs/FSRS7_PENALTY.MD`.
    - Added a transient per-call `PenaltyConsts` value for FSRS-7 schedule
      penalties. It hoists scalar and dual forgetting-curve constants plus
      long/short success prefactors out of the penalty rollout loops.
    - Correctness gates:
      `rtk cargo test test_manual_schedule_penalty_matches_autodiff_gradient`,
      `rtk cargo test test_manual_penalty_gradient_matches_autodiff_combined_objective`,
      `rtk cargo test fsrs7_schedule_penalty`, and
      `rtk cargo test training_v7` passed.
    - Criterion for `parameters/compute_parameters`:
      `68.116 ms..71.818 ms`, change `[-3.5302% -0.8091% +2.2313%]`,
      reported as no detected performance change.
    - Follow-up `sample` report:
      `/private/tmp/fsrs_compute_parameters_after_penalty_consts.sample.txt`.
      Top symbol sample counts: `analytic_v7::neon_loss::windowed_grad`
      (`1981`), `analytic_v7::windowed_loss` (`1661`),
      `training_v7::Dual35::pow` (`636`), `training_v7::fsrs7_next_s_good_dual`
      (`556`), `parameter_initialization_fsrs7::loss_with_curve` (`518`), and
      `training_v7::fsrs7_fc_r_dual` (`448`). The targeted curve-dual symbols
      shrank, but the end-to-end benchmark stayed within noise.
  - Windowed host-batch physical padding, 2026-06-04:
    - Files: `src/training.rs`, `docs/TRAINING_DATA_FLOW.MD`.
    - Windowed FSRS-7 host batches now pad their physical column count to a
      multiple of four while keeping `prediction_count` as the logical batch
      size used for loss and penalty scaling.
    - This keeps production ARM64 NEON training on full four-lane groups and
      avoids the scalar dual fallback for 1-3 ragged card columns.
    - Tests now cover the single-card padded shape and use the existing
      NEON-relative tolerance for validation loss comparisons that now exercise
      the NEON path instead of the scalar remainder path.
    - Criterion for `parameters/compute_parameters`:
      `63.599 ms..63.986 ms`, change `[-11.019% -8.6524% -6.6517%]`,
      reported as a significant improvement.
    - Follow-up `sample` report:
      `/private/tmp/fsrs_compute_parameters_after_window_padding.sample.txt`.
      Top symbol sample counts: `analytic_v7::neon_loss::windowed_grad`
      (`2095`), `analytic_v7::windowed_loss` (`1660`),
      `training_v7::Dual35::pow` (`744`),
      `parameter_initialization_fsrs7::loss_with_curve` (`563`),
      `training_v7::fsrs7_next_s_good_dual` (`539`), and
      `training_v7::fsrs7_fc_r_dual` (`480`). The former
      `analytic_v7::Dual35::pow` scalar-remainder signal is no longer a top
      entry after padding.
  - FSRS-7 validation schedule-penalty value-only rollout, 2026-06-04:
    - Files: `src/training.rs`, `src/training_v7.rs`,
      `docs/FSRS7_PENALTY.MD`, and `docs/TRAINING_DATA_FLOW.MD`.
    - Validation now calls a scalar `schedule_penalty_value()` path for FSRS-7
      instead of computing schedule-penalty dual gradients and discarding them.
      Training still uses `schedule_penalty_value_and_grad()` so optimizer
      updates are unchanged.
    - Added coverage that the scalar value-only path matches the existing
      value+gradient path for several batch sizes.
    - Criterion for `parameters/compute_parameters`:
      `59.445 ms..60.367 ms`, change `[-7.2033% -6.5411% -5.8939%]`,
      reported as a significant improvement versus the padded run.
    - Follow-up `sample` report:
      `/private/tmp/fsrs_compute_parameters_after_validation_value.sample.txt`.
      Top symbol sample counts: `analytic_v7::neon_loss::windowed_grad`
      (`1676`), `analytic_v7::windowed_loss` (`1372`),
      `parameter_initialization_fsrs7::loss_with_curve` (`452`),
      `training_v7::Dual35::pow` (`279`),
      `training_v7::fsrs7_next_s_good_dual` (`240`), and
      `training_v7::fsrs7_fc_r_dual` (`208`). Schedule-penalty dual work is
      materially lower than the post-padding profile because validation no
      longer computes unused schedule gradients.
  - FSRS-7 initialization scalar `loss_with_curve()`, 2026-06-04:
    - Files: `src/parameter_initialization_fsrs7.rs`,
      `docs/TRAINING_DATA_FLOW.MD`, and `docs/FSRS7_SPEED_PORT_PLAN.MD`.
    - Replaced temporary ndarray prediction/log arrays inside
      `loss_with_curve()` with a scalar loop over the existing bucket arrays.
      The initialization data read and selected best candidate semantics are
      unchanged.
    - Added a fixed-value regression test for the initialization loss helper.
    - Criterion for `parameters/compute_parameters`:
      `45.246 ms..45.364 ms`, change `[-24.489% -23.368% -21.650%]`,
      reported as a significant improvement versus the validation-value run.
    - Follow-up `sample` report:
      `/private/tmp/fsrs_compute_parameters_after_init_scalar_loss.sample.txt`.
      Top symbol sample counts: `analytic_v7::neon_loss::windowed_grad`
      (`2441`), `analytic_v7::windowed_loss` (`1924`),
      `parameter_initialization_fsrs7::loss_with_curve` (`507`),
      `training_v7::Dual35::pow` (`380`),
      `training_v7::fsrs7_next_s_good_dual` (`310`), and
      `training_v7::fsrs7_fc_r_dual` (`282`). Interpret these as the new
      post-`45ms` distribution, not a direct absolute comparison with older
      fixed-duration samples.
  - Maintenance notes:
    - Do not copy the patch literally: upstream PR #417 edits `src/analytic.rs`,
      while this branch uses split FSRS-7 modules and native NEON helpers.
    - Check each candidate against existing NEON hoists and skip-last logic to
      avoid duplicating code or adding dead branches.
    - Keep the scalar implementation as the oracle for NEON; update NEON only
      after scalar A/B is bit-for-bit.
  - Main Profile current-tree profile, 2026-06-04:
    - Collection source:
      `/Users/jschoreels/Library/Application Support/Anki2/Main Profile/collection.anki2`.
    - Data access: sampled a repeated release run against the SQLite `.backup`
      copy at `/private/tmp/fsrs-main-profile-collection-backup.anki2`; no live
      Anki profile data was written or modified.
    - Command shape: temporary release example repeating FSRS-7
      `compute_parameters()` with aligned `card_ids`, short-term enabled, and
      schedule penalties enabled. The repeated run covered `80` optimizations,
      `248819` review rows, and `183178` prepared FSRS items.
    - Timing: `80` optimizations in `45.024579 s`; individual
      `optimize_seconds` samples were usually around `0.50..0.60 s`.
    - Follow-up `sample` report:
      `/private/tmp/fsrs_compute_parameters_main_profile_current.sample.txt`.
      Top symbol sample counts:
      - `pow` from `libsystem_m`: `4308`
      - `analytic_v7::neon_loss::windowed_grad`: `3575`
      - `analytic_v7::windowed_loss`: `3035`
      - `training_v7::Dual35::pow`: `665`
      - `parameter_initialization_fsrs7::loss_with_curve`: `496`
      - `training_v7::fsrs7_next_s_good_dual`: `474`
      - `training_v7::fsrs7_fc_r_dual`: `425`
      - `exp` from `libsystem_m`: `218`
      - `training_v7::maybe_schedule_penalty_value_and_grad`: `144`
      - `training::build_windowed_batch`: `61`
      - `training_v7::schedule_penalty_value`: `25`
    - Interpretation: the current Main Profile bottleneck is still math-heavy.
      Batch construction, grouping, and sorting are too small to prioritize
      before recurrence/pow work. The next candidate should be a narrow
      recurrence or initialization `pow` reduction with log-loss gates, not a
      data-setup rewrite.
  - Initial `pow` reduction pass, 2026-06-04:
    - Files: `src/parameter_initialization_fsrs7.rs`, `src/training_v7.rs`,
      `docs/FSRS7_PENALTY.MD`, and `docs/TRAINING_DATA_FLOW.MD`.
    - FSRS-7 initialization now precomputes parameter-only forgetting-curve
      constants once per curve candidate and hoists candidate-stability weights
      once per `loss_with_curve()` call.
    - FSRS-7 schedule-penalty dual powers now use a fused derivative formula
      for variable exponents and reuse the computed power value for
      constant-exponent derivative coefficients.
    - Expected impact: reduce the `pow`/dual-recurrence share observed in
      `/private/tmp/fsrs_compute_parameters_main_profile_current.sample.txt`
      without changing data selection, optimizer settings, or persisted model
      data.
    - Correctness checks:
      `rtk cargo test training_v7`,
      `rtk cargo test parameter_initialization_fsrs7`, and
      `rtk cargo test -- --skip test_next_interval_fsrs7` passed.
    - Criterion for `parameters/compute_parameters` after this pass:
      `45.310 ms..46.250 ms`, change `[-0.4495% +1.4153% +2.8569%]`,
      reported as no detected performance change.
    - Main Profile backup rough timing from `examples/fsrs7_adr_collection`
      before its intentionally invalid `--gen 0` ADR phase:
      FSRS-7 parameter training `0.529 s` on `183178` prepared items, consistent
      with the previous current-tree `0.524652 s` harness timing.
  - FSRS-7 windowed validation cadence experiment, 2026-06-05:
    - Files: `src/training.rs`, `docs/TRAINING_DATA_FLOW.MD`, and
      `docs/FSRS7_PENALTY.MD`.
    - Windowed FSRS-7 training now validates every eighth epoch and always
      validates the final epoch. Prefix-item training and FSRS-6 validation
      cadence remain unchanged.
    - Rationale: Main Profile sampling still showed
      `analytic_v7::windowed_loss` as a large remaining cost while
      `training::build_windowed_batch` was negligible.
    - Criterion for `parameters/compute_parameters`, interval 2:
      `42.552 ms..45.628 ms`, change `[-6.7290% +3.9462% +19.591%]`,
      reported as no detected performance change.
    - Main Profile backup, interval 2: LogLoss `0.48805487`, RMSE bins
      `0.03720748`, optimization samples `0.622379 s`, `0.453348 s`, and
      `0.504469 s`.
    - Criterion for `parameters/compute_parameters`, interval 4:
      `38.615 ms..42.927 ms`, change `[-25.263% -11.477% +1.6711%]`,
      reported as no detected performance change.
    - Main Profile backup, interval 4: LogLoss `0.48805487`, RMSE bins
      `0.03720748`, optimization samples `0.435455 s` and `0.429194 s`.
    - Criterion for `parameters/compute_parameters`, interval 8:
      `36.543 ms..38.530 ms`, change `[-17.355% -10.504% -3.0651%]`,
      `p = 0.02 < 0.05`, reported as a performance improvement versus
      interval 4.
    - Main Profile backup, interval 8: LogLoss `0.48805487`, RMSE bins
      `0.03720748`, optimization samples `0.424084 s` and `0.411935 s`.
    - Criterion for `parameters/compute_parameters`, final-only validation:
      `35.728 ms..37.638 ms`, change `[-6.5324% -1.9629% +2.1583%]`,
      reported as no detected performance change versus interval 8.
    - Main Profile backup, final-only validation: LogLoss `0.48805487`, RMSE
      bins `0.03720748`, optimization samples `0.394338 s` and `0.390264 s`.
    - Correctness check:
      `rtk cargo test -- --skip test_next_interval_fsrs7` passed with
      `168` passed, `5` ignored, and `1` filtered out.
    - Interpretation: final-only validation preserved the selected parameters
      on the user's collection, but interval 8 is the selected cadence because
      it keeps one intermediate best-model checkpoint before the final epoch and
      still provides most of the validation-cost reduction.
  - Required gates:
    - `rtk cargo test test_analytic_loss_and_grad_matches_burn`
    - `rtk cargo test windowed_matches_sum_of_prefixes`
    - `rtk cargo test finite_difference_matches_batch_gradient`
    - `rtk cargo bench --bench parameters compute_parameters`
    - Repo collection A/B: trained parameters and log loss unchanged.
  - PR #417 follow-up not included by default: skipping the BCE loss value in
    training may save another reported `~7%`, but it would require either a
    training-only `SCORE_LOSS` split or changing tests that currently assert
    returned loss. Treat that as a separate explicit decision, not part of the
    bit-for-bit port.
- [ ] Fuse compatible exp pairs in the FSRS-7 recurrence.
  - Source: iter24.
  - Tests: gradient and loss gate because this is a floating-point reassociation.
- [ ] Cached sort keys and merged longest-prefix scans in card grouping.
  - Source: iter25.
  - Tests: deterministic grouping order and identical host-batch shape.
- [ ] Banked only: per-group padding skip.
  - Source: iter26 was rejected alone at `1.0412x`, but may be worth stacking later.
  - Tests: group-local active length produces identical final state/loss/grad.

### 8. Benchmark Path

- [ ] Treat `benchmark()` optimization as a later phase unless current users depend on it.
  - Source Phase 2 accepted work:
    - iter1: skip dead loss in O(N^2) benchmark gradient and add per-group padding skip.
    - iter2: clone-free count-based outlier filter and branchless validation BCE.
- [ ] Keep benchmark tests separate from `compute_parameters()` tests so benchmark-only changes do not obscure optimizer regressions.

## Documentation Updates Required When Implementing

- [x] Update inference data-flow documentation if runtime inference data reads change.
- [x] Update `docs/MODEL_SPLIT.MD` if the analytic module needs model-boundary documentation beyond this speed/data-flow plan.
- [ ] Update `docs/FSRS7_PENALTY.MD` if the training loop changes where penalty data is read, created, or applied.
- [ ] Update `docs/S90_RETRIEVAL.MD` if FSRS-7 state changes to include `stability_fast` or interval solving consumes different state.
- [x] Add or update a docs file for `ComputeParametersInput.card_ids` if that API is introduced.
- [x] Document any persistent/public data shape change before implementation is considered complete.

## Suggested Order

1. [ ] Resolve the 35-param vs 36-param FSRS-7 decision.
2. [ ] Add performance/correctness baseline.
3. [x] Port scalar inference/evaluation from PR 411's Burn-free production path.
4. [x] Port host-batch construction for card-id windowed training.
5. [x] Add scalar analytic FSRS-7 BCE and tests.
6. [x] Add host-side Adam.
7. [ ] Add SIMD validation and gradient tests.
8. [x] Add card-id O(N) windowing.
9. [ ] Port PR #417 bit-for-bit analytic-gradient micro-optimizations.
10. [ ] Layer remaining precision/micro-optimizations one at a time.
11. [ ] Re-run full tests and performance harness.

## Source Improvement Summary

Accepted source improvements worth considering:

- [x] Analytic BCE gradient replacing Burn autograd: iter2, `2.1656x`.
- [ ] Scalar analytic cleanup and hoists: iter5, iter8, iter10, iter13, roughly `1.07x` to `1.20x` each.
- [ ] PR #417 bit-for-bit windowed analytic-gradient cleanup:
  parameter-only transcendental hoists, active-branch-only forward recurrence,
  trailing-padding skip, and same-day `powf` skip. PR-reported collection
  benchmark: windowed `compute_parameters` about `-22%`; without card ids about
  `-30%`.
- [x] Direct host batch building and reuse: iter12 `1.2925x`, iter19 `1.1455x`.
- [ ] SIMD validation and gradient: iter14 `1.3051x`, iter15 `2.5002x`.
- [ ] Minimax SIMD transcendentals: iter17 `1.1557x`, iter21 `1.0833x`, iter23 `1.0958x`.
- [ ] O(N) expanding-window training: iter18 `3.1182x`.
- [x] Unified hard-label BCE: iter20 `1.0588x`.
- [x] Host-side Adam: iter22 `1.0773x`.
- [ ] Exp fusion: iter24 `1.0526x`.
- [x] Last-step skipping: iter25 `1.0541x`.
- [ ] Benchmark-specific follow-up: Phase 2 iter1 `1.16x`, iter2 `1.06x`.

Rejected but banked:

- [ ] Per-group padding skip: iter26, rejected alone at `1.0412x`, potentially useful only when stacked with another safe win.

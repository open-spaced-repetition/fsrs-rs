# Upstream merge verification

> Historical note: this document records the merge decision at `99386f0`.
> The later dependency-removal work documented in `BURN_REMOVAL.md` replaces
> the generic backend with the concrete host-side implementation.

## Compared revisions

- Branch baseline: `47fb3af` (`feature/fsrs-7-adr-optimization`).
- Refreshed upstream: `99386f0` (`open-spaced-repetition/fsrs-rs:main`).
- Verification date: 2026-09-08, Apple Silicon, stable Rust 1.91.1.
- Sample data: the repository's extracted `tests/data/collection.anki21` fixture,
  copied into isolated trees. No live Anki profile was used.

## Resolution boundaries

The merge integrates upstream's dependency/security updates (except the RNG
major-version change), public training config, card-ID evaluation and time-series
support, simulator callback APIs, parameter-validation export, seed isolation,
and documentation updates. It preserves the branch's FSRS-6/7 model dispatcher,
34-parameter FSRS-7 model, analytic/NEON training, fractional elapsed days, Cost
ADR policy extensions, default hyperparameters, and generic Burn backend API.

Upstream's single-model production-Burn removal, Cost ADR feature gate, and
rand 0.10/rand_distr 0.6 migration are deliberately resolved in favor of the
branch implementation. A merge commit therefore does not mean those upstream
architectural changes have been adopted. The RNG version is retained to avoid
changing seeded simulation streams independently of the explicit seed fix.

The Cost ADR seed-isolation fix intentionally changes default policies and
multi-generation rollouts. New tests cover separate training/evaluation seed
streams, per-generation seeds, explicit/default equivalence, and wrapping
arithmetic. Existing policy formats and calibration behavior are preserved.

## Numerical comparison

The same fixture harness ran on both trees with stable Rust, training both
versions with and without card IDs, then evaluating the optimized parameters
and scheduling five elapsed-day values: 0, 0.25, 1, 7, and 30.

| Model | Card IDs | Log loss before and after | Parameters |
|---|---|---:|---|
| FSRS-6 | No | 0.203311980 | Identical |
| FSRS-6 | Yes | 0.203311980 | Identical |
| FSRS-7 | No | 0.201762423 | Identical |
| FSRS-7 | Yes | 0.202102721 | Identical |

All scheduling outputs were identical. The largest RMSE-bin difference was
4e-9, consistent with unordered floating-point accumulation. New card-ID
evaluation is also compared against ordinary evaluation for default FSRS-6,
custom FSRS-6, FSRS-7, and deliberately non-prefix card groupings.

## Performance

An initial short run reported a noisy training slowdown. Repeating the compiled
benchmarks sequentially with 5-second warmups and 15-second measurement windows
produced 21.088 ms before and 21.096 ms after. Criterion reported no significant
change (p=0.42, estimated change interval -4.40% to +1.61%). Both runs used the
same stable compiler, fixture, and machine.

## Existing verification failures

The unmodified committed baseline's full release suite had 159 passes,
5 failures, and 7 ignored tests. The first merged suite had 166 passes with
exactly the same 5 failures and 7 ignored tests:

- `cost_adr::tests::test_cost_adr_next_states_matches_constant_retention`
- `inference::tests::test_evaluate_with_time_series_splits`
- `parameter_clipper::tests::test_parameter_clipper_works_with_num_relearning_steps`
- `simulation::tests::test_evaluate_in_flight_card_cost`
- `training::tests::test_training` (missing optional `tests/data/revlog.csv`)

These failures were not hidden, skipped, or accepted by weakening assertions.
Strict Clippy on stable Rust had 47 diagnostics before and 46 after, with no
new diagnostic categories. Baseline formatting also fails on existing files;
modified Rust files are formatted. Consequently this merge is not a claim that
the repository's entire verification suite is green.

`cargo check --all-targets` and `RUSTDOCFLAGS="-D warnings" cargo doc --release`
passed for the merged tree. Detailed logs, original working changes, and
isolated verification trees are in `/private/tmp/fsrs-merge-20260908-104859`.


Cost ADR training measured 237.47 ms before and 236.54 ms after on stable Rust.
Criterion detected no significant change (p=0.39, change interval -4.70% to
+2.20%). Policy outputs may differ because generation seed isolation is an
intentional upstream fix, not because of an RNG implementation change.


## Restored working changes

The pre-existing classifier changes were preserved separately from the merge
commit. Their original release suite had 164 passes, 5 failures, and 7 ignored
tests; after integration it has 172 passes, the same 5 failures, and 7 ignored
tests. All eight added merge tests and all five classifier tests pass.
`cargo check --all-targets --locked` also passes with the classifier work
restored. Its explicit input literals now supply `training_config: None`, and
its training-objective selection is retained alongside the new optional config.
The original patch and untracked-file archive are retained in the verification
directory; an adapted patch records the compatibility adjustments separately.

All six release documentation tests pass with the restored working changes.


## Follow-up: five failures resolved

The five failures recorded above were diagnosed and corrected after the merge:

- Cost ADR now uses the full post-rating memory state for interval calculation,
  retaining FSRS-7 difficulty and fast stability. Constant-policy equivalence
  covers both FSRS versions, every answer button, new/existing states, and
  fractional elapsed time.
- The in-flight workload fixture explicitly supplies its FSRS-6 parameters to
  the simulated card, matching the estimator instead of using FSRS-7 defaults.
- The broad training test skips only the optional CSV dataset when absent;
  collection-fixture training still exercises both models and short-term modes.
- Clipping tests now exercise FSRS-6 relearning-step bounds and separately verify
  FSRS-7's independence from the step count, replacing a stale default snapshot.
- The time-series reference is updated for the current 34-parameter model and
  checked against explicitly constructed input-order folds at a tighter 1e-6
  comparison tolerance. The fixture's numerical snapshot tolerance is unchanged.

Validation: `cargo test --release --locked` passes with 178 unit tests, zero
failures, and the existing seven ignored legacy tests; all six documentation
tests also pass. `cargo check --all-targets --locked` passes. The earlier Clippy
and repository-wide formatting observations were not addressed by this test-fix
follow-up. Existing classifier work remains included and uncommitted.

## CI lint follow-up

After the upstream autofix commits, the remaining macOS Clippy failures were resolved by simplifying transition loops, using a mutable slice for parameter clipping, and using `is_multiple_of` for SIMD batch alignment. Narrowly scoped, documented lint exceptions retain explicit numerical kernel arguments and inline per-review caches without adding allocations or changing scheduling behavior.

Verification on the isolated PR checkout (without the uncommitted classifier experiments): `cargo fmt --check` and `cargo clippy --locked -- -Dwarnings` pass; `cargo test --release --locked` passes 173 unit tests and 6 doc tests, with 7 existing tests ignored. The five additional classifier tests are only present in the main working directory.

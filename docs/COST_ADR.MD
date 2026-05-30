# Cost ADR

## Goals

- Fast enough to train/evaluate in a few minutes max with a documented default profile.
- Fully toggleable, with disabled behavior exactly matching current scheduling/simulation.
- A/B tested: disabled equals baseline, constant-DR policy equals fixed DR, seeded runs stay deterministic.
- Runtime-safe: Cost ADR only chooses desired retention after memory-state update; it must not change FSRS state updates.
- FSRS-7 correct: policy inputs are internal `stability` and `difficulty`, not `S90`; `S90` can be plotted only as reference.
- Anki-portable API: expose runtime hooks for direct policy evaluation and ADR-adjusted scheduling states.
- Serializable outputs: coefficients, policy version, bounds, cost weight, and evaluation metrics.
- Interpretable policy: provide a grid/export path for plotting desired retention over `stability`, `difficulty`, and cost weight.
- Observable training: progress/abort support plus metrics showing baseline vs policy, time saved, and coverage.
- Document data flow and persistence, especially what is stored and what is recomputed.
- Keep compatible add-ons working with fixed-DR behavior when ADR is disabled or unsupported.
- Make add-on optimizers able to request and persist ADR policy data, not only FSRS params.
- Decide whether add-on FSRS preset overlays can carry ADR policy/calibration fields or intentionally remain fixed-DR only.
- Expose enough runtime/backend data for FSRS Helper-style rescheduling to reproduce ADR scheduling, or document that it remains fixed-DR.

## Extra / To Revalidate

- Anki may later expose ADR as an installable/configurable add-on, but this depends on stable Rust/backend entry points for policy evaluation and ADR-adjusted scheduling states.
- FSRS Dynamic Preset Selection currently has no ADR payload shape; revalidate before enabling ADR for add-on-defined presets.
- FSRS Helper rescheduling currently uses fixed-DR interval math; revalidate once ADR runtime APIs are stable.

## Data Flow

- Data read:
  - FSRS parameter vector.
  - Post-review internal `MemoryState { stability, difficulty }`.
  - Runtime cost weight.
  - Cost ADR policy coefficients and bounds.
- Data created:
  - A per-user Cost ADR policy trained against that user's FSRS parameters and simulator configuration.
  - Evaluation metrics comparing fixed desired-retention baselines to Cost ADR rollouts.
  - Optional plot/export grid rows for `(stability, difficulty, cost_weight, desired_retention)`.
- Data updated:
  - The per-card desired retention selected for the next interval.
  - No FSRS memory-state update logic is changed by Cost ADR.
- Data storage:
  - Persist policy metadata separately from FSRS model parameters.
  - Persist at least policy version, coefficients, retention bounds, cost-weight bounds/default, and training/evaluation summary.
  - Recompute runtime desired retention from the persisted policy and current internal memory state.
  - Use `CostAdrPolicy::next_states(...)` when the caller needs per-button memory state, ADR desired retention, and interval outputs for Anki scheduling.

## First Impact Estimate

- `Cargo.toml` / `Cargo.lock`: add the distribution dependency used by policy training.
- `src/cost_adr.rs`: new module for policy representation, validation, evaluation, training, metrics, and plot-grid export.
- `src/lib.rs`: public exports for Cost ADR types and simulator entrypoints.
- `src/simulation.rs`: add a disabled-by-default Cost ADR path that updates `card.desired_retention` after memory-state update and before interval calculation.
- `src/training.rs`: expose safe progress reset/finish helpers for the Cost ADR trainer.
- `examples/cost_adr.rs`: add a runnable Anki-oriented example showing training, persistence fields, runtime scheduling hook, and metrics.
- `docs/*.MD`: document runtime data flow, persistence fields, disabled-state invariant, FSRS-7 internal-state inputs, and plotting/export semantics.

## Risk Estimate

- Low risk to default behavior if Cost ADR remains disabled by default at runtime.
- Medium risk in `src/simulation.rs` because it is shared, stateful, and already version-aware. This needs strict A/B tests.
- Medium risk in training runtime because the search budget can grow quickly. Defaults should be explicitly chosen for a few-minute target.
- Medium risk for Anki integration if persisted fields are not versioned from the start.
- Low risk to FSRS-7 model math if Cost ADR only consumes post-review internal state and delegates intervals to existing `next_interval()`.

## Test Estimate

- Existing non-ADR tests should pass unchanged.
- Add disabled-state A/B tests proving fixed-DR simulation output is unchanged.
- Add constant-policy tests proving Cost ADR with constant DR matches fixed-DR simulation.
- Add FSRS-7-specific tests proving policy input uses internal `stability`, not `S90`.
- Add validation tests for coefficient count, finite coefficients, retention bounds, and cost-weight bounds.
- Add a small training smoke test with reduced population/generation/deck/span.
- Add a plot-grid/export test checking stable dimensions and finite desired-retention values.

## Evaluation Fields

- `average_desired_retention`: arithmetic average of the ADR desired retention selected across simulated reviews.
- `fixed_fsrs_equivalent_desired_retention` (`FSRS Eq. DR`, displayed as `FSRS6 Eq. DR` or `FSRS7 Eq. DR`): fixed-FSRS desired retention that reaches the same simulated Memorized target for the selected model, interpolated from the fixed-DR baseline frontier.
- `same_target_time_saved_percent`: time saved by ADR compared with that fixed-FSRS equivalent desired retention at the same Memorized target.
- `cost_weight_for_average_desired_retention(target)`: inferred ADR cost weight for a target average ADR desired retention. It interpolates in `ln(1 + weight)` space when only persisted calibration points are available.
- `calibrate_cost_weight_for_average_desired_retention(target)`: computes a simulation-backed closest ADR cost weight by bisection in `ln(1 + weight)` space. This is used after policy training to persist a denser, target-oriented calibration table instead of relying on coarse fixed weights like `0, 64, 128`.
- Calibration first evaluates endpoint weights `0` and `1024`. If their average ADR DR span is below `0.005` retention, it persists only those endpoints because extra bisection points do not give users a meaningful target range.
- Optional endpoint training penalties can target a broad simulated average ADR DR range. Example: `w=0 -> 0.90`, `w=1024 -> 0.50`, with `penalty=10`, subtracts `10 * ((avg_dr(w=0)-0.90)^2 + (avg_dr(w=1024)-0.50)^2)` from each candidate score.
- ADR training treats `generations` as a maximum. By default, it can stop after at least `10` generations when the best optimizer score (`hypervolume_delta - endpoint_penalty`) improves by less than `1%` over the last `4` generations. Set `early_stop_patience_generations=0` to force all configured generations. Increase `early_stop_min_generations`, `early_stop_patience_generations`, or lower `early_stop_min_relative_gain` to bias toward longer/higher-quality runs.

## Early Stop Experiment

Yomitan FSRS-7 comparison from 2026-05-30, using exact Dynamic Preset Selection card-id slices, stored FSRS-7 params, `population_size=8`, seed `42`, and `generations=20` as the maximum. Hypervolume percentages compare the sum of slice `best_hypervolume_delta` values against fixed 20 generations.

| Stop policy | ADR seconds | Generations run | HV vs fixed 20 | Interpretation |
| --- | ---: | ---: | ---: | --- |
| Fixed 20 generations | 108.557 | 80/80 | 100.0% | Quality reference. |
| Stale best, patience 5 | 64.370 | 40/80 | 90.6% | Faster, but misses later improvements on several slices. |
| Stale best, patience 6 | 95.138 | 65/80 | 99.3% | Good quality, weak speed saving. |
| Relative gain window 2, min 6, gain 1% | 32.783 | 24/80 | 70.4% | Too aggressive. |
| Relative gain window 4, min 6, gain 1% | 37.683 | 27/80 | 70.4% | Slower than window 2, same low-quality result. |
| Relative gain window 4, min 10, gain 1% | 73.213 | 58/80 | 99.3% | Current default: near fixed quality with a meaningful speed saving. |
| Relative gain window 4, min 14, gain 1% | 74.255 | 66/80 | 99.3% | Similar quality and runtime; extra minimum generations did not help materially. |

## Endpoint Penalty Experiment

Yomitan FSRS-6 comparison from 2026-05-30, using `examples/fsrs7_adr_collection.rs` with `--deck-name-contains Yomitan`, `--model-version fsrs6`, `--endpoint-avg-dr-min-weight 0.90`, `--endpoint-avg-dr-max-weight 0.50`, and `--endpoint-avg-dr-penalty 10`. Each row was trained independently after filtering cards by first post-reset learning grade.

| First grade | Reviewed cards | FSRS items | Learn span days | FSRS6 training seconds | ADR training seconds | Avg DR at w=0 | Avg DR at w=1024 | Best AUC relative time saved |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 5,237 | 94,549 | 657 | 9.589 | 9.614 | 0.931 | 0.421 | 4.90% |
| 2 | 661 | 3,128 | 192 | 0.250 | 0.425 | 0.934 | 0.526 | 6.74% |
| 3 | 3,628 | 30,233 | 563 | 1.902 | 2.930 | 0.924 | 0.450 | 11.39% |
| 4 | 383 | 1,179 | 166 | 0.127 | 0.072 | 0.938 | 0.585 | 4.04% |

Interpretation: the endpoint penalty can broaden the average ADR DR range substantially, but sparse/easy partitions may not hit the low endpoint target exactly. First grade 4 stayed high at `0.585` despite the `0.50` target, likely because there are only 383 reviewed cards and very little review work to trade off.

Baseline comparison without endpoint penalty, using the same `population_size=8`, `generations=5`, seed, FSRS-6 model, Yomitan deck filter, and first-grade partitions:

| First grade | No-penalty supported Avg DR range | Endpoint-penalty supported Avg DR range | No-penalty AUC time saved | Endpoint-penalty AUC time saved |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.421-0.931 | 0.421-0.931 | 4.90% | 4.90% |
| 2 | 0.526-0.934 | 0.526-0.934 | 6.74% | 6.74% |
| 3 | 0.450-0.924 | 0.450-0.924 | 11.39% | 11.39% |
| 4 | 0.696-0.959 | 0.585-0.938 | 6.42% | 4.04% |

Interpretation: with this seed and the default training grid, the endpoint penalty did not change grades 1-3 because the unconstrained optimum already covered a broad Avg DR range. Grade 4 was the only partition where the penalty materially widened the supported range downward, but it traded away AUC time-saved quality.

## Dynamic Preset Yomitan Slice

Yomitan FSRS-7 comparison from 2026-05-30, using `examples/fsrs7_adr_collection.rs` with `--model-version fsrs7` and exact card ids selected from the installed FSRS Dynamic Preset Selection add-on config. The add-on applies presets in order, so these rows exclude cards already captured by earlier Kana, Katakana, and Single Kanji Word presets before the Yomitan rule is reached.

These runs optimize FSRS-7 parameters from scratch for each selected slice. Stored add-on FSRS params were exported separately, but loading them into the standalone collection example failed with `InvalidInput`, so remaining differences versus Anki UI calibration should first be checked against parameter serialization/compatibility.

| First grade | Raw Yomitan cards | Selected after preset precedence | Reviewed cards | FSRS items | Learn span days | FSRS7 training seconds | ADR training seconds | Supported Avg DR range | Best AUC relative time saved |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 5,221 | 4,467 | 4,467 | 81,518 | 605 | 15.756 | 34.317 | 0.431-0.945 | 2.42% |
| 2 | 654 | 590 | 590 | 2,828 | 167 | 0.457 | 1.002 | 0.510-0.921 | 7.22% |
| 3 | 3,596 | 2,783 | 2,783 | 23,257 | 494 | 3.043 | 13.151 | 0.405-0.936 | 9.68% |
| 4 | 382 | 213 | 213 | 713 | 108 | 0.159 | 0.189 | 0.453-0.917 | 16.38% |

Interpretation: reproducing the add-on's ordered selection removes a substantial number of Yomitan cards, especially for first grades 1, 3, and 4. The supported Avg DR ranges are still broad after retraining from scratch, but first grade 4 is too small to read as a smooth policy curve: several weights are non-monotonic and the equivalent fixed-DR lookup sometimes cannot be interpolated.

Endpoint-penalty comparison for the same exact slices, using target endpoints `w=0 -> 0.90`, `w=1024 -> 0.50`, and penalty `10`:

| First grade | No-penalty supported Avg DR range | Endpoint-penalty supported Avg DR range | No-penalty AUC time saved | Endpoint-penalty AUC time saved |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.431-0.945 | 0.431-0.945 | 2.42% | 2.42% |
| 2 | 0.510-0.921 | 0.510-0.921 | 7.22% | 7.22% |
| 3 | 0.405-0.936 | 0.405-0.936 | 9.68% | 9.68% |
| 4 | 0.453-0.917 | 0.424-0.919 | 16.38% | 5.80% |

Interpretation: for first grades 1-3, the endpoint penalty does not change the selected policy because the unconstrained result already spans roughly the requested range. First grade 4 is the only exact slice where the penalty widens the lower endpoint, but it trades away most of the AUC time-saved score and remains noisy because only 213 reviewed cards are selected.

Anki add-on check for the Yomitan Hard row (`firstgrade:2`): before the add-on optimizer was fixed, Optimize called Anki core with the row query only, `deck:"Japan::1. Vocabulary::1.2 Normal Scheduling::Yomitan" firstgrade:2`, and did not subtract earlier ordered preset matches. On 2026-05-30 this backend path used `3,095` FSRS items and returned ADR calibration average DRs from `0.507226` to `0.547746`, displayed as `[50.7%, 54.8%]`.

After fixing the add-on optimizer to use the same first-match effective slice as scheduling, the same Yomitan Hard optimization used `2,828` FSRS items, matching the exact scheduled preset slice above. With Anki's dynamic ADR bounds enabled, that fixed path returned bounds `0.515734-0.906129` and calibration average DRs from `0.515658` to `0.518877`.

The retrievability-derived Anki bounds were then disabled. After rebuilding `pylib:anki:rsbridge`, the fixed Anki path returned default bounds `0.300000-0.995000` and calibration average DRs from `0.300063` to `0.339777`. That did not match the standalone table because Anki was still training ADR with the raw `extract_simulator_config()` deck shape: `deck_size=10000`, `learn_span=365`, unlimited learn/review limits, and `max_cost_perday=1800s`.

After shaping Anki's ADR simulator from the effective preset slice, the same Yomitan Hard optimization used `2,828` FSRS items and returned calibration average DRs from `0.522923` to `0.937036` in `4.733s`. This is close to the standalone exact-slice range `0.510-0.921`; the remaining small difference comes from Anki's normal optimization path using the preset's same-day/relearning-step options, which produces slightly different FSRS-7 params than the standalone example. The ADR optimize request can now override the simulator review limit and daily time budget in minutes per preset; absent overrides keep the `9999` review limit and `720` minute budget used for this comparison.

## FSRS-7 Interval Solver

Yomitan comparison from 2026-05-29, using `examples/fsrs7_adr_collection.rs` with `--deck-name-contains Yomitan`, target average ADR DR `0.8`, FSRS-7, and the same trained parameter vector. The run covered 169,942 revlogs, 10,013 reviewed cards, and 129,085 FSRS items.

These results use `1e-4` as an absolute interval tolerance for the Newton-based variants. The earlier relative tolerance (`high * 1e-4`) was not the intended precision target and produced a different output family.

| Solver | Meaning | ADR training seconds | Inferred cost weight for Avg ADR DR 0.8 | Notes |
| --- | --- | ---: | ---: | --- |
| Pure Newton | Newton only, no bracket fallback | 58.027 | 125.883 | Fastest, but lands in the old output family; not reliable enough. |
| Newton/Bisection | Safe Newton with bisection fallback | 73.915 | 79.453 | Reliable behavior, slower than Newton/Brent. |
| Newton/Brent | Safe Newton with Brent fallback | 66.936 | 79.453 | Best candidate so far: close to Brent/Bisection behavior, faster fallback. |
| Pure Brent | `roots::find_root_brent`, strict convergence | 147.087 | 79.476 | Close behavior, too slow. |
| Bisection | 50 iterations, LUT high hint | 174.888 | 79.476 | Slow reference behavior. |

Interpretation:

- The implemented FSRS-7 interval solver is now `Newton/Brent`: safe Newton first, strict Brent fallback.
- Pure Newton is fast but can miss enough roots to materially change ADR evaluation.
- Brent and bisection agree closely, which makes them useful references, but both are too slow as primary solvers.
- Newton/Brent was selected because it matches the reliable output family while avoiding the full bisection fallback cost.
- `Newton/Bisection` remains the historical compatibility baseline for pre-change FSRS-7 interval solving.

Run caveat: the `Newton/Brent` and `Pure Newton` runs used a copied collection snapshot after the live SQLite database became locked. The snapshot had the same Yomitan revlog/card counts and the same FSRS-7 parameter vector as the live runs.

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

## Extra / To Revalidate

- Anki may later expose ADR as an installable/configurable add-on, but this depends on stable Rust/backend entry points for policy evaluation and ADR-adjusted scheduling states.

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
- `cost_weight_for_average_desired_retention(target)`: inferred ADR cost weight for a target average ADR desired retention. It interpolates in `ln(1 + weight)` space because ADR normalizes the cost weight with the same transform.

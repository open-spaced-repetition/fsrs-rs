# FSRS7 Penalty Mode

## Purpose

Provide the FSRS-7 training objective with scheduling penalties, while keeping
inference behavior unchanged.

## API Surface

- `compute_parameters()` exposes `enable_sched_penalties` (default `true`).
- When `enable_sched_penalties = true`, FSRS-7 schedule penalties (penalty 1 & 2) are added.
- When `enable_sched_penalties = false`, schedule penalties are skipped.
- `L2` regularization remains enabled in both modes.

## Objective Components

Training uses:

- `L2` with weight `0.5` (applied via `FSRS7_PENALTY_W_L2`)
- schedule penalty 1 with weight `0.5`
- schedule penalty 2 with weight `0.0015`

Penalty simulation settings:

- `n_reviews = 10`
- target DR for penalty 1: `0.90`
- target DR set for penalty 2: `[0.99]`
- Newton iterations: `4`

## Backpropagation

Training keeps burn autograd for the main BCE objective and computes penalty
gradients in scalar code:

1. Main BCE loss uses `forward_classification(..., Reduction::Sum)` and
   `backward()`.
2. `L2` and FSRS7 schedule penalty are computed in scalar code.
3. Manual penalty gradients are written into `w` gradients before optimizer
   step.

The schedule interval inversion still uses the same two-phase math:

1. Phase 1: Newton solve in `u = log(t)` scalar space.
2. Phase 2: one implicit-differentiation lift step around `t*`.

The derivative is propagated by an internal dual-number engine for 35 params.

## Data Flow

- Data created:
  - temporary scalar penalty values and dual gradients during each train/valid batch.
- Data read:
  - current model weights (`w[0..34]`), plus batch size for scaling.
  - input flag `enable_sched_penalties` to decide whether schedule penalties are computed.
- Data updated:
  - model gradient for `w` is updated by adding manual penalty gradients before optimizer step.
- Data storage:
  - no new persistent data fields; only in-memory training-time values.

## Scope

- Affects optimization only (`compute_parameters` path).
- Does not change runtime scheduling/inference formulas used by `FSRS::next_interval`,
  `FSRS::interval_at_retrievability`, or Anki runtime calls.

## Short-Term Opt-Out

- `compute_parameters(..., enable_short_term = false)` is honored for FSRS-7.
- Training enforces long-term-only FSRS-7 updates by zeroing transition weight
  `w[26]` in the training clipper path and freezing FSRS-7 short-term/transition
  gradient slots.

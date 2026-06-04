# Inference Data Flow

This document tracks the data read and created by runtime FSRS inference paths.

## Inputs Read

- `FSRS::new()` reads the caller-provided parameter slice and normalizes it through
  model-version compatibility checks.
- `FSRS` stores the clipped active parameter vector and selected model version for
  runtime inference.
- `memory_state()`, `memory_state_batch()`, `historical_memory_states()`, and
  `historical_memory_state_batch()` read `FSRSItem.reviews` in chronological order.
- `next_states()` and `next_states_with_elapsed_days()` read an optional current
  `MemoryState`, desired retention, elapsed days, and the active parameter vector.
- `evaluate()`, `universal_metrics()`, and `evaluate_with_time_series_splits()`
  read FSRS items, current review ratings, current review elapsed days, and
  recency or constant evaluation weights.

## Data Created

- Runtime review replay creates transient `MemoryState` values from scalar FSRS
  transitions.
- Batch memory-state APIs create output vectors aligned with the input item order.
- Evaluation creates transient prediction, label, weight, and R-matrix aggregate
  buffers.
- Time-series evaluation creates transient train/test splits and aggregates
  predictions across completed splits.

## Data Updated

- No caller-provided review item, review, or memory-state input is mutated.
- `FSRS` caches its normalized parameter vector at construction. Runtime inference
  reads this cache and does not update it.
- Progress callbacks receive transient `ItemProgress` values. Returning `false`
  interrupts the current operation.

## Compatibility Notes

- FSRS-6 runtime replay rounds elapsed days to nearest whole days inside the
  scalar FSRS-6 transition, matching the versioned tensor model.
- FSRS-7 runtime replay keeps fractional elapsed days.
- Public memory state remains `{ stability, difficulty }`; no Anki-facing storage
  shape changes are introduced by scalar inference.

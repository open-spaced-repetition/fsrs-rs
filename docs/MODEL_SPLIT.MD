# Model Split

## Shared orchestration

`src/model.rs` owns parameter validation, clipping, model-version selection,
review replay, initialization, and the concrete public `FSRS` type.
`ModelVersion` routes version-specific operations once parameters have been
validated.

## Version-specific code

- `src/model_v6.rs` contains FSRS-6 scalar forgetting, stability, difficulty,
  interval, and SM-2 bridge functions.
- `src/model_v7.rs` contains dual-trace FSRS-7 scalar transitions, forgetting
  curves, interval inversion, and the cached runtime helper.
- `src/analytic_v6.rs` computes FSRS-6 training loss and gradients.
- `src/analytic_v7.rs` and its architecture-specific submodules compute
  FSRS-7 training loss and gradients.
- `src/training_v6.rs` and `src/training_v7.rs` contain version-specific
  regularization and schedule-penalty calculations.

Runtime inference and production training operate on slices, flat host arrays,
and `MemoryState`; no backend/device abstraction crosses the public API.

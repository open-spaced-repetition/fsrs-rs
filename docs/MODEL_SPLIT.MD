# Model Split

## Purpose

Reduce compatibility risk by separating FSRS-6 and FSRS-7 model-specific math
while keeping shared model state/config logic in `src/model.rs`.

## File Layout

- Shared entrypoints/types:
  - `src/model.rs`
- FSRS-6 specific math:
  - `src/model_v6.rs`
- FSRS-7 specific math:
  - `src/model_v7.rs`
- FSRS-6 parameter compatibility normalization (`0/17/19 -> 21`):
  - `src/model_v6.rs::check_and_fill_parameters_fsrs6()`
- Parameter clipping split by version:
  - `src/parameter_clipper_v6.rs`
  - `src/parameter_clipper_v7.rs`
  - shared dispatcher in `src/parameter_clipper.rs`
- Inference defaults split by version:
  - `src/inference_v6.rs` (FSRS-6 defaults/constants)
  - `src/inference_v7.rs` (FSRS-7 defaults)
  - shared API/orchestration in `src/inference.rs`
- Training penalty split by version:
  - `src/training_v6.rs` (FSRS-6 L2 regularization; schedule penalty remains
    no-op)
  - `src/training_v7.rs` (FSRS-7 penalty math/gradients)
  - shared training loop/objective composition in `src/training.rs`
- FSRS-7 analytic training kernel:
  - `src/analytic_v7.rs` computes scalar BCE loss and BCE loss gradients for
    FSRS-7 card-id windowed training batches.
  - `src/analytic_v7/neon_loss.rs` provides the ARM64 NEON validation-loss and
    training-gradient paths for the same 35-parameter windowed batch layout.
    The exact scalar dual-number gradient remains the oracle and non-NEON
    fallback.
  - The kernel is parameter-size-specific to the current 35-parameter FSRS-7
    layout and is not shared with FSRS-6.
- Training initial-stability normalization is version-specific:
  - FSRS-6 uses the legacy data-derived initializer and `smooth_and_fill()`.
  - FSRS-7 uses its FSRS-7 initializer/fill logic for params `0..3`.

## Dispatch Strategy

- `src/model.rs` keeps shared orchestration (`Model::step`, `Model::forward`).
- `src/model.rs::check_and_fill_parameters()` keeps shared validation and delegates
  FSRS-6 compatibility conversion to `model_v6`.
- Version-specific computations are routed through `VersionOps` trait impls:
  - `Fsrs6Ops`
  - `Fsrs7Ops`
- Model version is selected from parameter count at creation/update time and
  then reused via `ModelVersion`, instead of repeated runtime checks on `w.len()`.
- Training penalty routing now also uses `ModelVersion` dispatch, so FSRS-6 and
  FSRS-7 penalty behavior are separated without checking parameter length in the
  training loop.
- FSRS-7 card-id windowed training can route BCE loss and BCE gradient
  calculation through `analytic_v7`. The shared training loop composes
  schedule/L2 penalties and can apply host-side Adam plus version-aware clipping
  for that windowed FSRS-7 path. Production FSRS-7 windowed batches are
  host-array-only; Burn tensors are only constructed by test oracles for this
  path.
- Version constants are colocated with version logic:
  - `src/model_v6.rs`: `PARAM_LEN`
  - `src/model_v7.rs`: FSRS-7 solver/LUT constants and `PARAM_LEN`

## Data Flow

- Data read:
  - model parameter vector `w`
  - runtime tensors (`delta_t`, `rating`, `stability`, `difficulty`)
  - FSRS-7 windowed host arrays (`t_historys`, `r_historys`, labels, weights)
    during analytic training-loss and training-gradient calculation
  - FSRS-7 card-id windowed training tuning constants: learning rate `0.07`
    and `17` epochs
- Data updated:
  - next-step `MemoryStateTensors` during `Model::step()`
  - optimized initial-stability params `0..3` after training, normalized by the
    selected model version before return
  - transient optimizer gradients and host Adam moment state for FSRS-7 windowed
    BCE training
  - transient training config values before progress initialization for the
    FSRS-7 card-id windowed path
- Data storage:
  - no new persistence; all values remain in-memory
  - FSRS-7 S90 LUT cache remains process-local memory only
  - shared windowed host batches, analytic gradient arrays, and host Adam
    moments live only for the current optimization run

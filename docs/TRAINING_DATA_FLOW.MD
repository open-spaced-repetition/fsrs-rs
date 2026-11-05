# Training Data Flow

## Preparation

`compute_parameters()` validates configuration, applies recency weights, and
excludes histories beyond `max_seq_len`. Optional card IDs remain aligned
throughout filtering.

- FSRS-6 filters outliers and initializes the four initial stabilities from
  items with exactly one long-term review. Only each item's first long-term
  review is read and rounded here (a raw interval of at least half a day is
  long-term after whole-day rounding); full histories are rounded during batch
  layout. Initialization minimizes the BCE plus absolute-deviation objective
  with Newton steps safeguarded by bisection.
- FSRS-7 retains all items and starts from its default parameters.

## Host batches and gradients

Training is entirely host-side and has no tensor-framework dependency.

- FSRS-6 log-loss training uses the eight-lane SIMD path in `analytic_v6/`:
  `plan.rs` assigns items to batches with the same boundaries and order as the
  scalar trainer, `batch.rs` lays out eight histories per group, `simd.rs`
  differentiates them, and `worker.rs` shares each batch's groups with one
  helper thread (when at least two CPUs are available). Group results are
  reduced in a fixed order, so threaded and single-thread gradients are
  identical. Groups whose success-growth term `exp((1 - R) * w10) - 1` falls
  below `0.001` are recomputed with the scalar kernel, which remains the
  numerical reference and serves recall-classifier training.
- FSRS-7 uses the reverse-mode/SIMD kernels in `analytic_v7`. Items without
  card IDs are represented as independent one-prefix cards; aligned card IDs
  allow expanding prefixes to share a trajectory.
- Probability training uses weighted binary cross-entropy. Recall-classifier
  training applies the configured false-positive/false-negative cost and the
  sigmoid straight-through derivative at the decision boundary.
- Schedule and L2 penalty gradients are added to the main gradient before
  freeze masks, Adam, and version-aware parameter clipping.

The optimizer uses `beta_1=0.70`, `beta_2=0.98`, and `epsilon=1e-8`.
`Some(TrainingConfig::default())` selects 5 epochs / 0.04 for FSRS-6 and
9 epochs / 0.0118 for FSRS-7. Currently `training_config: None` retains the
internal 9-epoch / 0.0118 defaults for both versions. The benchmark caps histories
at 64 unless overridden.

FSRS-6's SIMD gradients are checked against the scalar kernel; trained
parameters are close to, but not bit-identical with, the scalar trainer's. See
`FSRS6_PERFORMANCE.md`.

All batches, gradients, optimizer moments, shuffled indices, and progress state
are transient. Training performs no persistent writes.

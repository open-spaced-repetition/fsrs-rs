# Training Data Flow

## Preparation

`compute_parameters()` validates configuration, normalizes elapsed time for the
selected FSRS version, filters outliers, initializes stability/forgetting-curve
parameters, applies recency weights, and drops histories beyond `max_seq_len`.
Optional card IDs remain aligned throughout filtering.

## Host batches and gradients

Training is entirely host-side and has no tensor-framework dependency.

- FSRS-6 prefix batches use the reverse-mode analytic kernel in
  `analytic_v6`. With card IDs, full card trajectories are scored once.
- FSRS-7 uses the reverse-mode/SIMD kernels in `analytic_v7`. Items without
  card IDs are represented as independent one-prefix cards; aligned card IDs
  allow expanding prefixes to share a trajectory.
- Probability training uses weighted binary cross-entropy. Recall-classifier
  training applies the configured false-positive/false-negative cost and the
  sigmoid straight-through derivative at the decision boundary.
- Schedule and L2 penalty gradients are added to the main gradient before
  freeze masks, Adam, and version-aware parameter clipping.

The optimizer uses `beta_1=0.70`, `beta_2=0.98`, and `epsilon=1e-8`. FSRS-7
card-ID tuning uses learning rate `0.07` and 17 epochs unless an explicit
`TrainingConfig` overrides it. The benchmark caps histories at 64 unless
overridden.

All batches, gradients, optimizer moments, shuffled indices, and progress state
are transient. Training performs no persistent writes.

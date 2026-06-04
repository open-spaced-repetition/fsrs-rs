# Training Data Flow

## Purpose

Track preprocessing and training-batch data flow for `compute_parameters()` and
`benchmark()`.

## Outlier Filtering

- Data read:
  - Initialization candidates, identified by `long_term_review_cnt() == 1`.
  - Training candidates, kept or removed using the same bucket decisions.
- Data created:
  - In-memory `(first_rating, first_long_term_delta_t_bucket)` counts.
  - In-memory initialization-item indices for buckets that survive filtering.
  - In-memory `removed_pairs` bucket sets.
- Data updated:
  - `dataset_for_initialization` is filtered by the removed bucket set.
  - `trainset` is filtered by the same removed bucket set.
- Data storage:
  - No persistent storage.
  - Bucket counts, kept indices, and removed-pair sets live only during preprocessing.

The filter now computes bucket decisions from bucket counts and stores only
initialization-item indices for kept buckets instead of storing full grouped
`FSRSItem` values. This keeps the returned filtered initialization set ordering
and filtered train set semantically unchanged while avoiding duplicated grouped
item storage during preprocessing.

## Training Batch Construction

- Data read:
  - Recency-weighted training items after max sequence length filtering.
  - Optional `ComputeParametersInput.card_ids`, aligned by index with
    `ComputeParametersInput.train_set`.
  - Training config values: `batch_size`, `seed`, `num_epochs`, and model flags.
  - FSRS-7 windowed tuning constants used only when card ids are present.
- Data created:
  - One review-length-sorted weighted item vector inside `train()`.
  - Burn tensor batch datasets for training and validation, built from that same
    sorted item order when card ids are absent.
  - One shared FSRS-7 windowed host batch vector when card ids are present. Each
    batch stores flat host arrays for histories, labels, and weights; production
    windowed training no longer creates Burn tensors for those batches.
- Data updated:
  - The in-memory training config for FSRS-7 card-id optimization, which uses
    learning rate `0.07` and `17` epochs instead of the generic defaults.
  - The model parameters during optimizer steps.
  - Training progress state when progress tracking is enabled.
- Data storage:
  - No persistent storage.
  - Prefix tensor batches, windowed host arrays, and sorted weighted items are
    in-memory only for the training call.

`compute_parameters()` and `benchmark()` intentionally train and validate on the
same weighted items. `train()` therefore consumes one weighted item vector and
builds both train and validation prefix tensor datasets from one sorted order,
avoiding a redundant outer item clone/sort while preserving the existing Burn
tensor training path. For FSRS-7 card-id windowed training, `train()` builds one
host batch vector and reuses it for both shuffled training and shuffled
validation passes.

## Card Ids And Windowed Training

`ComputeParametersInput.card_ids` is optional. When supplied, its length must
match `train_set`; otherwise `compute_parameters()` returns `InvalidInput`.

- Data read:
  - The caller-provided card id for each input prefix item.
  - The same review histories used by the existing prefix-item path.
- Data created:
  - In-memory `(FSRSItem, card_id)` pairs before preprocessing.
  - In-memory filtered training pairs after outlier filtering.
  - In-memory recency-weighted items retaining their card id.
  - FSRS-7 windowed host batches grouped by card id, where each batch column is
    one full card review sequence and loss weights mark the surviving prefix
    prediction timesteps.
  - Transient scalar dual-number values and a 35-parameter BCE gradient array
    for FSRS-7 windowed training batches.
  - Transient scalar loss values for FSRS-7 windowed validation batches.
  - Transient FSRS-7 windowed host Adam state: first moment, second moment, and
    step count for the current optimization run.
- Data updated:
  - No persistent card-id data is written.
  - FSRS-7 windowed training updates the in-memory host parameter vector with
    the analytic BCE gradient, existing schedule/L2 penalty gradients, freeze
    masks, host Adam, and version-aware parameter clipping.
  - FSRS-7 analytic window loops update transient memory state only when a later
    prediction can consume it; the final unused state update in each batch
    column is skipped.
  - The Burn model parameter tensor is synchronized from the host parameter
    vector for epoch selection and final return.
- Data storage:
  - Card ids live only for the `compute_parameters()` call.
  - Windowed host arrays live only for the current training call and are shared
    by windowed training and validation passes.
  - Analytic gradient arrays live only for the current training batch.
  - Analytic validation loss values live only for the current validation batch.
  - Host Adam moments live only for the current FSRS-7 windowed training run.

Outlier filtering computes kept/removed item indices once and applies those
indices to both review histories and card ids. This keeps card ids aligned with
their surviving prefix items through filtering, recency weighting, max sequence
length filtering, and batching.

The windowed path is currently FSRS-7-only. If card ids are absent, or if FSRS-6
training is requested, optimization uses the existing prefix-item tensor path.
When FSRS-7 card ids are present, `compute_parameters()` rewrites the transient
training config before progress initialization so progress totals, the cosine
learning-rate schedule, and the training loop all use learning rate `0.07` and
`17` epochs. This tuning is not persisted and does not change the generic
`TrainingConfig` defaults used by FSRS-6 or prefix-item training.
FSRS-7 windowed training now computes the BCE loss gradient from host arrays in
`src/analytic_v7.rs`, then composes the existing penalty gradients in
`src/training.rs`. FSRS-7 windowed validation uses a loss-only scalar path over
the same host arrays. FSRS-7 windowed optimizer steps use host-side Adam with
the same `beta_1=0.8`, `beta_2=0.85`, and `epsilon=1e-8` settings as the Burn
optimizer, followed by the same version-aware clipping rules. Prefix-item
training, FSRS-6 training, and callers without card ids remain on the existing
Burn tensor loss and optimizer path.

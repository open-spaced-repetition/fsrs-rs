# Inference Data Flow

## Model and inputs

`FSRS` owns a clipped parameter vector and a model-version tag derived from the
parameter count. It is a concrete Rust type: there is no tensor backend or
device state. A 21-parameter vector selects FSRS-6 and a 34-parameter vector
selects FSRS-7. Empty input keeps the established FSRS-6 default behavior.

Inference replays chronological `FSRSItem.reviews` directly into `MemoryState`.
Public states contain `stability`, `difficulty`, and `stability_fast`; FSRS-6
keeps the fast value equal to ordinary stability. FSRS-6 rounds elapsed days
internally, while FSRS-7 retains fractional elapsed days.

## Evaluation

`evaluate()` replays each weighted prefix independently. When aligned card IDs
are available, `evaluate_with_card_ids()` groups expanding prefixes by card and
walks each complete history once. Non-prefix groups fall back to independent
replay. Both paths aggregate the same weighted binary cross-entropy and
R-matrix RMSE without persisting caller data.

Time-series evaluation slices card IDs alongside each training split and
forwards optional training hyperparameters. Returning `false` from a progress
callback interrupts the operation at the next progress boundary.

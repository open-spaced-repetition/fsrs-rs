# S90 Retrieval

## Purpose

Expose a derived stability-like interval for integrations that still want a
"time to 90% recall" value while keeping the native FSRS model state unchanged.

## Definitions

- `S` (native memory state stability): internal model state used by FSRS updates.
- `S90`: derived interval in days where retrievability reaches `0.9`.

## Behavior

- Legacy models (17/19/21 parameters):
  - `S90 == S` by definition of the legacy forgetting curve.
  - The implementation returns `state.stability` exactly for `target=0.9`.
- FSRS-7 (35 parameters):
  - `S90` is derived numerically from the FSRS-7 forgetting curve.
  - Interval solving uses a LUT-warmed Brent root finder with bisection
    fallback.
  - Current solver target precision is `1e-3` retrievability error.

## FSRS-7 LUT Data Flow

- Data created:
  - An in-memory `S90` lookup table (LUT) indexed by log-scaled stability.
  - LUT size is `256` points.
  - Transient FSRS-7 runtime contexts can pair the cached LUT with precomputed
    forgetting-curve constants for repeated simulator interval solves.
  - Each FSRS-7 interval solve creates a transient stability-specific curve
    view that caches the stability-dependent mixture weights used by Newton,
    Brent, and bisection paths.
- Data read:
  - When solving `next_interval()`, the LUT provides a warm-start interval for
    bracketing.
  - LUT points are sampled over a practical stability band (`0.03..3000`) to
    avoid over-allocating points to extreme tails.
  - Runtime contexts read the same parameter vector and do not change the
    interval solver's Newton/Brent/bisection behavior.
  - If Newton reaches its iteration limit, Brent reads the narrowed bracket
    produced by Newton instead of rebuilding the search range from zero.
  - Stability-specific curve views are local to a single interval solve and are
    discarded after that solve completes.
- Data updated:
  - The cache map is updated lazily the first time a parameter set is used.
- Data storage:
  - Process-local memory only (`static` cache in `fsrs-rs`), keyed by FSRS-7
    parameter hash.
  - Runtime contexts are per-call/per-simulation values and are not persisted.
  - No persistence to disk and no Anki-side storage required.

## API

- `FSRS::interval_at_retrievability(state, target_retrievability)`
- `FSRS::s90(state)` convenience helper (`target_retrievability = 0.9`)

This keeps compatibility for old-model consumers while preserving FSRS-7's
native internal semantics.

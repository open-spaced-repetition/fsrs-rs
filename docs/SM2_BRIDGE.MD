# SM-2 Bridge

## Purpose

Convert an incomplete SM-2 history into a starting FSRS memory state whose
forgetting curve remains consistent with the card's existing interval.

## Paths

- FSRS-6 bridge:
  - Implemented in `src/model_v6.rs` as `memory_state_from_sm2_scalar(...)`
  - Uses legacy slots `w8`, `w9`, `w10`, `w20`
  - Uses the SM-2 ease factor to estimate difficulty
- FSRS-7 bridge:
  - Implemented in `src/model_v7.rs` as `memory_state_from_sm2_scalar(...)`
  - Starts from the model's neutral state shape (`S_fast = 0.8 * S_slow`,
    `D = 5`)
  - Uses bisection over log stability to scale both traces while preserving
    their ratio
  - Produces a state whose retrievability reaches `sm2_retention` at the SM-2
    interval, when representable within the model's stability bounds
  - Ignores the ease factor because FSRS-7 has no equivalent scalar difficulty
    bridge

## Dispatch

- `FSRS::memory_state_from_sm2(...)` delegates to
  `FSRS::memory_state_from_sm2_inner(...)`.
- `memory_state_from_sm2_inner(...)` dispatches by `ModelVersion` to the
  version-specific scalar conversion.

## Data Flow

- Data read:
  - SM-2 inputs (`ease_factor`, `interval`, `sm2_retention`)
  - active model parameter vector
- Data created:
  - one derived `MemoryState { stability, difficulty, stability_fast }`
- Data storage:
  - no persistence; conversion result only

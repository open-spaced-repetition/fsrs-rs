# SM-2 Bridge

## Purpose

Provide separate conversion paths from SM-2 fields into FSRS memory state for
legacy FSRS-6-style models and FSRS-7 models.

## Paths

- FSRS-6 bridge:
  - Implemented in `src/model_v6.rs` as `memory_state_from_sm2_fsrs6(...)`
  - Uses legacy slots `w8`, `w9`, `w10`, `w20`
  - Keeps historical behavior for `FSRS::default()` / `FSRS::new(&[])`
- FSRS-7 bridge:
  - Implemented in `src/model_v7.rs` as `memory_state_from_sm2_fsrs7(...)`
  - Uses a numerically stable stability conversion based on a single exposed
    decay component (`w27`)
  - Uses neutral difficulty (`D=5`) because FSRS-7 no longer has a direct
    legacy scalar-difficulty bridge equivalent

## Dispatch

- `FSRS::memory_state_from_sm2(...)` delegates to `Model::memory_state_from_sm2(...)`.
- `Model::memory_state_from_sm2(...)` dispatches by `ModelVersion`.
- `VersionOps` defines `memory_state_from_sm2_fsrs(...)` for version-specific routing:
  - `Fsrs6Ops` => `memory_state_from_sm2_fsrs6(...)`
  - `Fsrs7Ops` => `memory_state_from_sm2_fsrs7(...)`

## Data Flow

- Data read:
  - SM-2 inputs (`ease_factor`, `interval`, `sm2_retention`)
  - active model parameter vector
- Data created:
  - one derived `MemoryState { stability, difficulty }`
- Data storage:
  - no persistence; conversion result only

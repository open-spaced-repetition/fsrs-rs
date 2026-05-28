# Compatibility Notes

## Default Constructor

- `FSRS::default()` and `FSRS::new(&[])` keep legacy behavior and instantiate
  the FSRS-6 default parameter set.
- This preserves SM-2 bridge behavior through `memory_state_from_sm2()`.

## SM-2 Bridge

- `memory_state_from_sm2()` uses legacy FSRS-6 bridge slots (`w8`, `w9`,
  `w10`, `w20`).
- For compatibility, callers that rely on the default constructor continue to
  get the historical FSRS-6 conversion behavior.

## Interval Precision

- `FSRS::next_states_with_elapsed_days()` accepts elapsed time in days as
  `f32`, including fractional intervals (for example, same-day reviews).
- This branch's `FSRSReview::delta_t` is `f32`. Upstream `main` still uses
  `u32` in the older single-model training path, so direct optimizer parity
  checks must build equivalent input by converting whole-day elapsed times to
  the target branch's type. Fractional same-day/interday deltas are supported
  here but cannot be represented in that upstream API.
- FSRS-6 keeps legacy day-based behavior internally by rounding elapsed days
  for recall updates, while still using `f32` transport.
- FSRS-6 and FSRS-7 both return fractional intervals from scheduling APIs.
- FSRS-7 keeps full fractional behavior for elapsed days and output intervals.

## Stability Bounds

- Minimum stability is clamped at `0.0001` days.
- This same lower bound is used when initializing stability parameters and when
  clamping inferred or updated memory states.

## Anki Learning And Relearning Steps

- `FSRS::next_states*()` returns model intervals. Anki may still show
  learning or relearning step delays while a card is in the learning queue.
- For low desired retention and very low post-lapse stability, FSRS-7 can
  calculate a minute-scale `Again` interval. Once a same-day `Good` is applied
  to that state, core FSRS can produce an interval of hours or days; repeated
  minute-scale `Good` delays are therefore a scheduler step behavior rather
  than the model interval solver repeating the same result.

## Training Robustness

- FSRS-6 training derives initial stability from review data and applies the
  legacy monotonic `smooth_and_fill()` pass to params `0..3` before returning
  optimized parameters.
- FSRS-7 training uses FSRS-7-specific initial-stability fill/monotonic
  normalization for params `0..3`; it does not reuse the FSRS-6 smoothing
  helper.
- FSRS-7 parameter clipping is NaN-safe: non-finite values are converted to
  finite bounds before clamping.
- Training classification clamps predicted retrievability to `[0.0001, 0.9999]`
  before binary cross-entropy, avoiding `log(0)` at recall probability bounds.
- Negative elapsed days are normalized to `0.0` before FSRS-7 training and
  state updates, matching the existing public retrievability behavior.
- Outlier filtering no longer assumes every training item has a long-term
  review (`delta_t >= 1.0`), so same-day-only items no longer panic.
- Long-term `delta_t` grouping used by outlier filtering/initialization is
  bucketed by day (floor), which avoids sparse fragmented groups when
  training data contains fractional interday elapsed times.

## FSRS-6 Upstream Parity Snapshot

Parity was checked against `upstream/main` with whole-day training data, so the
`u32` upstream `delta_t` and current `f32` `delta_t` APIs receive equivalent
elapsed-day values. After restoring FSRS-6 data-derived initial stability and
legacy `smooth_and_fill()`, the large params `0..3` drift is gone. The remaining
params `0..3` difference is the lower stability bound change (`0.001` upstream,
`0.0001` current). Remaining params `4..20` differences are most likely from
optimizer/L2 config drift.

| Param | Before upstream | After current | Abs diff | Rel diff | Most likely explanation |
|---:|---:|---:|---:|---:|---|
| 0 | 0.0010 | 0.0001 | 0.0009 | 90.00% | `S_MIN` changed. |
| 1 | 0.0010 | 0.0001 | 0.0009 | 90.00% | `S_MIN` changed. |
| 2 | 0.0010 | 0.0001 | 0.0009 | 90.00% | `S_MIN` changed. |
| 3 | 0.0010 | 0.0001 | 0.0009 | 90.00% | `S_MIN` changed; previous init/smoothing drift is fixed. |
| 4 | 6.50 | 6.46 | 0.04 | 0.59% | Optimizer/config drift. |
| 5 | 0.73 | 0.74 | 0.01 | 0.88% | Optimizer/config drift. |
| 6 | 3.29 | 3.18 | 0.11 | 3.23% | Optimizer/config drift. |
| 7 | 0.0010 | 0.0010 | 0.00 | 0.00% | Matches. |
| 8 | 1.72 | 1.77 | 0.06 | 3.28% | Optimizer/config drift. |
| 9 | 0.07 | 0.10 | 0.03 | 41.79% | Small baseline amplifies relative difference; optimizer/config drift. |
| 10 | 0.64 | 0.70 | 0.06 | 8.98% | Optimizer/config drift. |
| 11 | 1.33 | 1.39 | 0.05 | 3.98% | Optimizer/config drift. |
| 12 | 0.18 | 0.15 | 0.03 | 18.42% | Small baseline amplifies relative difference; optimizer/config drift. |
| 13 | 0.10 | 0.16 | 0.06 | 59.65% | Small baseline amplifies relative difference; optimizer/config drift. |
| 14 | 1.52 | 1.57 | 0.05 | 3.14% | Optimizer/config drift. |
| 15 | 0.81 | 0.72 | 0.09 | 11.02% | Optimizer/config drift. |
| 16 | 2.14 | 2.01 | 0.13 | 6.28% | Largest residual; optimizer/config drift. |
| 17 | 0.55 | 0.53 | 0.01 | 2.56% | Optimizer/config drift. |
| 18 | 0.08 | 0.13 | 0.05 | 58.02% | Small baseline amplifies relative difference; optimizer/config drift. |
| 19 | 0.07 | 0.07 | 0.00 | 0.00% | Matches. |
| 20 | 0.10 | 0.10 | 0.00 | 0.00% | Matches. |

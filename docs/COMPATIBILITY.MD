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

- FSRS-7 parameter clipping is NaN-safe: non-finite values are converted to
  finite bounds before clamping.
- Negative elapsed days are normalized to `0.0` before FSRS-7 training and
  state updates, matching the existing public retrievability behavior.
- Outlier filtering no longer assumes every training item has a long-term
  review (`delta_t >= 1.0`), so same-day-only items no longer panic.
- Long-term `delta_t` grouping used by outlier filtering/initialization is
  bucketed by day (floor), which avoids sparse fragmented groups when
  training data contains fractional interday elapsed times.

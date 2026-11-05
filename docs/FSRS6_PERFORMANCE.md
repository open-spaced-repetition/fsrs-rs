# FSRS-6 training performance

FSRS-6 log-loss training uses an eight-lane SIMD kernel and one helper thread
(see `TRAINING_DATA_FLOW.MD`). It trains 5.3–6.2× faster than the scalar
trainer on `main`. FSRS-7 is unchanged.

## Measurements

Criterion, `same_epoch_training` benchmark: complete `compute_parameters`
calls with 5 epochs, batch size 512, `max_seq_len` 1024, seed 2023, short-term
training on, schedule penalties off. Input cloning is outside the timer.
Apple M2, release build, median of 10 samples.

| Dataset                        | Mode     | FSRS-6 `main` | FSRS-6 now | Speedup | FSRS-7 `main` | FSRS-7 now |
| ------------------------------ | -------- | ------------: | ---------: | ------: | ------------: | ---------: |
| Anki collection (14,290 items) | card IDs |      17.99 ms |    2.97 ms |    6.1× |       5.91 ms |    5.90 ms |
| Anki collection                | prefixes |      52.45 ms |    8.41 ms |    6.2× |      18.93 ms |   18.96 ms |
| CSV revlog (88,158 items)      | card IDs |      97.45 ms |   18.30 ms |    5.3× |      36.14 ms |   35.55 ms |
| CSV revlog                     | prefixes |     667.49 ms |  108.72 ms |    6.1× |     230.80 ms |  233.27 ms |

The default `parameters/compute_parameters*` benchmarks (FSRS-7) stay within
3% of `main` across interleaved runs.

### FSRS-7 as a baseline

Comparing the two models needs equal threads as well as equal epochs. On
aarch64 with NEON, FSRS-7 trains on one thread with its NEON kernel; its
two-thread trainer (`training_windowed.rs`) is used on other targets. FSRS-6
uses two threads on any multi-core target, so on Apple Silicon the table above
compares two threads with one.

Same benchmark with FSRS-6 forced to one thread:

| Dataset         | Mode     | FSRS-6, 1 thread | FSRS-7, 1 thread | FSRS-6 time saved |
| --------------- | -------- | ---------------: | ---------------: | ----------------: |
| Anki collection | card IDs |          4.84 ms |          6.04 ms |               20% |
| Anki collection | prefixes |         15.58 ms |         19.05 ms |               18% |
| CSV revlog      | card IDs |         31.13 ms |         37.59 ms |               17% |
| CSV revlog      | prefixes |        208.68 ms |        232.09 ms |               10% |

FSRS-6 has 21 parameters to FSRS-7's 34, one forgetting curve instead of two,
and trains on slightly fewer items after outlier filtering (14,165 vs. 14,290
and 86,120 vs. 88,158). Being modestly faster than FSRS-7 per epoch is
expected; the helper thread then gives another 1.6–1.9×.

## Where the time goes

- **Kernel** (`analytic_v6/simd.rs`): eight histories are differentiated
  together with cached forward terms, fast `exp`/`ln` for in-range arguments,
  and binary BCE adjoints taken directly with respect to `log R`.
- **Threads** (`analytic_v6/worker.rs`): both threads claim groups of a batch;
  the helper also lays out later batches while training starts and frees the
  input items while idle.
- **Layout** (`analytic_v6/plan.rs`, `batch.rs`): a flat card index and
  per-card item chains replace per-card vectors and sorting; each batch is one
  contiguous buffer with labels encoded in the sign of the weights.
- **Preprocessing** (`training.rs`, `dataset.rs`): only each item's first
  long-term review is read; outlier buckets use dense day indices and a bitset.
- **Initialization** (`parameter_initialization.rs`): Newton steps with a
  bisection safeguard replace the 1000-step ternary search.

## Numerical agreement

Batch boundaries, order, sample weights, the optimizer and its schedule are
unchanged. SIMD gradients agree with the scalar kernel to a relative L2 error
below `2e-4`; groups where `exp((1 - R) * w10) - 1 < 0.001` (prone to
cancellation in f32) are recomputed with the scalar kernel. The threaded
reduction is bit-identical to the single-thread one.

Trained parameters are therefore close to, but not bit-identical with, `main`.
On both fixtures, with and without card IDs, with default and 5-epoch
configurations, the largest parameter difference is `9e-6` and the largest
log-loss difference `1.8e-7`. FSRS-7 parameters are bit-identical to `main`.

## Reproduce

```sh
cargo bench --bench parameters -- same_epoch_training
FSRS_BENCH_REVLOG_CSV=tests/data/revlog.csv cargo bench --bench parameters -- same_epoch_training
```

`FSRS_BENCH_COLLECTION` selects another Anki collection.

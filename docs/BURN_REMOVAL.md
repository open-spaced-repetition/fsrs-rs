# Burn Removal

## Result

The project has no Burn package in its normal or development dependency graph.
Inference uses scalar, version-dispatched Rust code. FSRS-6 and FSRS-7 training
use host-side analytic gradients, host Adam, and the existing version-specific
clipping and penalty functions. The generic backend constructor was removed;
callers construct the concrete `FSRS` type with `FSRS::new()`.

The former neural `revlog_classifier` example was removed from the Cargo target
list because it depended entirely on Burn. Its newer FSRS recall-classifier
experiment remains available as `examples/classifier.rs` and uses the same
host-side optimizer as the library.

## Collection benchmark

Criterion was run before and after the change against an extracted copy of
`backup-2026-09-08-20.09.29.colpkg`, the newest Anki backup available when the
benchmark started. The live collection database was not opened. Both runs used
the same copied collection, release profile, and 10-sample Criterion protocol.

| Training input   |    Before |     After |                         Change |
| ---------------- | --------: | --------: | -----------------------------: |
| Aligned card IDs | 213.25 ms | 219.04 ms |                    2.7% slower |
| No card IDs      |  80.486 s |  1.7183 s | 46.8x faster (97.9% less time) |

The card-ID path was already using the analytic FSRS-7 kernel, so the removal
does not improve that path and this run measured a small regression. The large
improvement without card IDs comes from eliminating tensor construction and
autodiff for prefix batches. Criterion's 95% confidence intervals were
`212.43–214.10 ms` before and `216.49–222.26 ms` after for aligned card IDs,
and `63.573–105.56 s` before and `1.6709–1.7791 s` after without card IDs.

The default resolved Cargo graph contains 110 unique `cargo tree` entries after
the change, down from 340 in the saved pre-removal tree.

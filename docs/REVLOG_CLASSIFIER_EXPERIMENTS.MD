# Revlog Recall Classifier Experiments

> Archived implementation note (2026-09-08): the neural sequence ablation
> executable was removed with the Burn dependency. The measurements below are
> retained as an experiment ledger. The supported host-only classifier example
> is now `cargo run --release --example classifier -- ...`.

## Goal

Predict the next review as one hard scheduling class:

- **Success / known**: Hard, Good, or Easy (`rating > 1`)
- **Again / forgotten**: Again (`rating == 1`)

The experiment is not constrained to calculate FSRS retrievability (`R`). Neural
models emit a binary success logit directly. A scheduler can convert that score
to the final class using either the fixed training boundary (`0.5`) or a threshold
chosen on validation data for the desired false-positive/false-negative cost.

This file is the experiment ledger. New changes should be added as separate
ablations, with the data split, seed, training objective, threshold-selection
rule, and complete test metrics recorded. Do not replace earlier results.

## Metric conventions

- The positive class is **Success / known**.
- A false positive means predicting **known** when the answer was **Again**.
- A false negative means predicting **forgotten** when the answer was a success.
- **Score AUC** evaluates the continuous score and is independent of a decision
  threshold.
- **Balanced accuracy** is `(success recall + Again recall) / 2`. For a binary
  decision score, its AUC equals balanced accuracy, so only balanced accuracy is
  shown in the tables.
- Ordinary accuracy is included, but it is misleading by itself because the test
  set contains 64.95% successes. Always predicting success obtains 64.95%
  accuracy but only 50% balanced accuracy.
- Lower log loss is better. It measures probability calibration, not just the
  final decision.

## Dataset and protocol

The run used an extracted copy of the `2026-08-31 20:36:27` Anki backup. The
source backup and active profile were not opened with SQLite or modified.

| Setting | Value |
|---|---:|
| Deck name contains | `Yomitan` |
| Eligible target samples | 169,303 |
| Split | Chronological 60% train / 20% validation / 20% test |
| Train | 101,581: 83,913 success, 17,668 Again |
| Validation | 33,860: 26,448 success, 7,412 Again |
| Test | 33,862: 21,992 success, 11,870 Again |
| Maximum history per stream | 32 answered reviews plus one query token |
| Hidden size | 32 |
| Batch size | 256 |
| Epochs | 8 with validation early selection |
| Optimizer | Adam, learning rate 0.003, epsilon `1e-8` |
| Seed | 2023 |
| Binary objective | Class-balanced binary cross-entropy with logits |
| Model-selection objective | Highest validation balanced accuracy at threshold 0.5; log loss breaks exact ties |
| Grade auxiliary weight | 0.25 when enabled |

The strong prevalence drift—82.61% success in training, 78.11% in validation,
and 64.95% in test—is an important part of the result. It explains why a
threshold optimized on validation may not transfer reliably to the later test
period.

### Eligibility and leakage boundary

The input reader:

1. keeps ratings 1–4 and review types Learning, Review, Relearning, and Filtered;
2. excludes filtered reviews with factor zero;
3. starts each card at its most recent learning sequence, matching the earlier
   FSRS classifier experiment; and
4. creates a target only when that card has at least one prior retained review.

Query tokens include only information assumed available before answering:
elapsed time, review status/type, calendar time, and causal workload/context
counts. They exclude the target grade, answer duration, scheduled interval,
previous interval, and ease factor. Tests mutate those excluded target fields and
verify that every feature tensor remains identical.

Historical tokens may use their already-observed grade, answer time, review
type, interval, previous interval, and ease factor.

## Models

### Earlier FSRS-based baselines

- **FSRS probability** trains ordinary FSRS parameters for probability log loss,
  then classifies its `R` score.
- **FSRS direct classifier** still computes FSRS `R`, but trains the FSRS
  parameters against a hard decision at `R = 0.5` using balanced false-positive
  and false-negative costs and a straight-through surrogate gradient.

### Revlog sequence ablations

The ablation order is cumulative:

| Model | Change relative to the previous row |
|---|---|
| `legacy-gru` | Reproduction baseline: one GRU over the card's history and the original 20 causal features |
| `rwkv-inputs-gru` | Adds RWKV-inspired cumulative time, periodic elapsed time, global review gap, intervening review/new-card counts, today's workload, and card age |
| `hierarchical-gru` | Adds independent causal card, deck, note, and user histories; four GRU states are concatenated |
| `hierarchical-rwkv` | Replaces each GRU with a small RWKV-style recurrent encoder using token shift, receptance, learned decay, key/value accumulation, and a channel mixer |
| `hierarchical-rwkv-multitask` | Adds a class-balanced four-grade auxiliary head while retaining the direct binary head for the final decision |

The open SRS benchmark's full RWKV model is larger and more sophisticated. It
uses separate card, deck, note, preset, and user modules, a 128-component
forgetting-curve mixture, and an immediate four-grade head. See its
[architecture](https://github.com/open-spaced-repetition/srs-benchmark/blob/main/rwkv/architecture.py),
[feature processing](https://github.com/open-spaced-repetition/srs-benchmark/blob/main/rwkv/data_processing.py),
and [model/loss](https://github.com/open-spaced-repetition/srs-benchmark/blob/main/rwkv/model/srs_model.py).

The implementation here is deliberately a small, CPU-runnable **RWKV-style
ablation**, not a reproduction of the 2.76-million-parameter RWKV-7 model. It
also omits the preset stream because the exported collection schema does not
provide the deck configuration identifier without decoding Anki's protobuf deck
blob. Its hierarchy streams are encoded independently and concatenated instead
of applying the reference modules sequentially.

## Results: fixed decision threshold 0.5

These are the primary results for a model trained to make the decision at 0.5.

| Model | Selected epoch | Train seconds | Log loss | Score AUC | Accuracy | Balanced accuracy | TP | TN | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Always success | — | — | 4.842896 | 0.500000 | 0.649460 | 0.500000 | 21,992 | 0 | 11,870 | 0 |
| Previous answer | — | — | 5.414496 | 0.575360 | 0.608086 | 0.575360 | 15,061 | 5,530 | 6,340 | 6,931 |
| FSRS probability | — | 0.327 | **0.637222** | 0.618980 | 0.639507 | 0.517619 | 20,351 | 1,304 | 10,566 | 1,641 |
| FSRS direct classifier | — | 34.689 | 1.408362 | 0.647239 | 0.540665 | 0.609477 | 8,341 | 9,967 | 1,903 | 13,651 |
| `legacy-gru` | 5 | 126.942 | 0.707537 | 0.638841 | 0.566623 | 0.605014 | 10,481 | 8,706 | 3,164 | 11,511 |
| `rwkv-inputs-gru` | 6 | 132.556 | 0.710854 | 0.634808 | 0.548757 | 0.598161 | 9,520 | 9,062 | 2,808 | 12,472 |
| `hierarchical-gru` | 2 | 534.382 | 0.674606 | 0.635147 | 0.590190 | 0.605573 | 12,186 | 7,799 | 4,071 | 9,806 |
| `hierarchical-rwkv` | 2 | 696.798 | 0.678465 | 0.662075 | 0.592995 | **0.624774** | 11,402 | 8,678 | 3,192 | 10,590 |
| `hierarchical-rwkv-multitask` | 6 | 675.407 | 0.713010 | **0.673459** | 0.556317 | 0.620112 | 8,944 | 9,894 | 1,976 | 13,048 |

FSRS probability has the best log loss, so it remains the best calibrated
probability in this experiment. The multitask RWKV-style classifier has the best
ranking AUC. The binary-only RWKV-style classifier has the best hard decision at
the fixed 0.5 boundary.

## Results: validation-tuned decision threshold

The score and log loss do not change with a threshold, so this table focuses on
the hard decision. Each threshold maximized balanced accuracy on validation and
was then frozen before evaluating test.

| Model | Threshold | Test accuracy | Test balanced accuracy | TP | TN | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|
| FSRS probability | 0.742412 | 0.551415 | 0.612189 | 8,992 | 9,680 | 2,190 | 13,000 |
| `legacy-gru` | 0.422978 | 0.600142 | 0.603851 | 13,007 | 7,315 | 4,555 | 8,985 |
| `rwkv-inputs-gru` | 0.445850 | 0.587295 | 0.603441 | 12,083 | 7,804 | 4,066 | 9,909 |
| `hierarchical-gru` | 0.493154 | 0.594088 | 0.604619 | 12,522 | 7,595 | 4,275 | 9,470 |
| `hierarchical-rwkv` | 0.482743 | 0.603715 | 0.625233 | 12,167 | 8,276 | 3,594 | 9,825 |
| `hierarchical-rwkv-multitask` | 0.470663 | 0.583693 | **0.630175** | 10,439 | 9,326 | 2,544 | 11,553 |

The direct FSRS classifier is absent from this table because it is explicitly
trained and evaluated at its fixed `R = 0.5` decision boundary.

## Incremental effect relative to the reproduced GRU baseline

Positive AUC and balanced-accuracy deltas are improvements. Negative log-loss
deltas are improvements.

| Model | Δ log loss | Δ score AUC | Δ fixed balanced accuracy | Δ tuned balanced accuracy |
|---|---:|---:|---:|---:|
| `rwkv-inputs-gru` | +0.003317 | −0.004033 | −0.006853 | −0.000409 |
| `hierarchical-gru` | **−0.032931** | −0.003694 | +0.000559 | +0.000768 |
| `hierarchical-rwkv` | −0.029072 | +0.023234 | **+0.019760** | +0.021382 |
| `hierarchical-rwkv-multitask` | +0.005473 | **+0.034618** | +0.015098 | **+0.026324** |

Relative to the earlier FSRS models:

- `hierarchical-rwkv` improves fixed balanced accuracy over the direct FSRS
  classifier by **0.015297** and score AUC by **0.014836**.
- `hierarchical-rwkv-multitask` improves score AUC over the direct FSRS
  classifier by **0.026220**, but its fixed balanced accuracy is lower than the
  binary-only RWKV-style model.
- With validation-tuned thresholds, multitask RWKV-style balanced accuracy is
  **0.017987** above validation-threshold FSRS probability.

## Conditional retrievability distributions

This follow-up asks what the two continuous recall scores look like after the
best hard-decision model has split the test reviews into **forgotten** and
**remembered**. The split uses the multitask classifier's validation-tuned
threshold `0.470663`; it is not retuned on the test set.

**Correction:** an earlier draft incorrectly called the classifier's local
auxiliary four-grade output “RWKV-style Instant.” That output is not Anki's
RWKV-Instant retrievability, so those figures are withdrawn. The results below
were rerun with Anki's packaged production model and
`RWKV_trained_on_5000_10000.bin` weights.

The corrected scores are:

- **FSRS R**: ordinary FSRS-7 retrievability, with parameters fitted on the same
  training split for probability log loss; and
- **Anki RWKV-Instant R**: canonical pre-answer retrievability from Anki's
  packaged Rust inference runtime. It replays all 218,979 eligible reviews in
  the collection—not just the selected deck—so card, note, deck, preset, and
  global recurrent state match Anki's normal historical calibration path.

All **33,862** test revlog IDs matched a canonical RWKV prediction.

| Classifier decision | Reviews | Actual success | Score | Mean | Std. dev. | P05 | P25 | Median | P75 | P95 | Min–max |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Forgotten | 20,879 | 0.553331 | FSRS R | 0.635774 | 0.100257 | 0.463620 | 0.544202 | 0.673407 | 0.705766 | 0.768950 | 0.370976–0.998903 |
| Forgotten | 20,879 | 0.553331 | Anki RWKV-Instant R | 0.537592 | 0.121004 | 0.337017 | 0.460122 | 0.533833 | 0.615737 | 0.743212 | 0.078753–0.951102 |
| Remembered | 12,983 | 0.804051 | FSRS R | 0.801834 | 0.090817 | 0.640915 | 0.740214 | 0.807472 | 0.868055 | 0.936754 | 0.385620–0.999990 |
| Remembered | 12,983 | 0.804051 | Anki RWKV-Instant R | 0.789437 | 0.111408 | 0.573671 | 0.723460 | 0.805763 | 0.867726 | 0.948280 | 0.193218–0.998804 |

Ten-point histogram shares show the overlap more directly:

| Classifier decision / score | 0.0–0.1 | 0.1–0.2 | 0.2–0.3 | 0.3–0.4 | 0.4–0.5 | 0.5–0.6 | 0.6–0.7 | 0.7–0.8 | 0.8–0.9 | 0.9–1.0 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Forgotten / FSRS R | 0.00% | 0.00% | 0.00% | 0.05% | 13.91% | 20.59% | 36.44% | 26.13% | 2.69% | 0.19% |
| Forgotten / Anki RWKV-Instant R | 0.01% | 0.37% | 2.50% | 8.84% | 26.00% | 33.14% | 19.38% | 8.25% | 1.42% | 0.08% |
| Remembered / FSRS R | 0.00% | 0.00% | 0.00% | 0.02% | 0.22% | 1.85% | 10.76% | 34.20% | 38.84% | 14.11% |
| Remembered / Anki RWKV-Instant R | 0.00% | 0.02% | 0.02% | 0.15% | 1.25% | 5.59% | 12.81% | 27.97% | 37.12% | 15.08% |

### Operational 100-review interpretation

For 100 reviews sampled from the classifier-forgotten group, the retrospective
test result corresponds to:

| Quantity | Per 100 reviews |
|---|---:|
| Actual Again answers | **44.67** |
| Actual successes | **55.33** |
| Mean FSRS R | **63.58%** |
| Mean Anki RWKV-Instant R | **53.76%** |
| Failure rate implied by mean FSRS R | **36.42%** |
| Failure rate implied by mean Anki RWKV-Instant R | **46.24%** |

Canonical RWKV-Instant is therefore close to the observed group rate: it is
**1.57 percentage points more pessimistic** than the actual 55.33% success rate.
FSRS is **8.24 points more optimistic**. In the classifier-remembered group,
RWKV-Instant is similarly close (78.94% predicted versus 80.41% actual), while
FSRS is 80.18%.

The classifier's separation aligns particularly strongly with canonical
RWKV-Instant: its group mean rises by **0.251844**, compared with **0.166060** for
FSRS R and **0.250720** for the observed success rate. The groups still overlap,
so “forgotten” is a scheduling decision rather than certainty about the next
answer.

Within the classifier-forgotten group, Pearson correlation between FSRS R and
RWKV-Instant R is **0.027767**; within classifier-remembered, it is **0.585963**.
The near-zero forgotten-group correlation shows that the classifier's risky
reviews can receive quite different FSRS and RWKV assessments. Conditional
correlations must still be interpreted cautiously because the classifier itself
created the groups.

## Interpretation

1. **The added scalar inputs alone did not help.** They slightly reduced both
   AUC and fixed balanced accuracy. Future feature additions should therefore be
   individually ablated rather than bundled.
2. **Hierarchical context with GRUs mainly improved calibration.** Log loss
   improved substantially, but ranking AUC and the hard decision remained close
   to the baseline.
3. **The RWKV-style recurrent update produced the first clear decision gain.**
   It improved both AUC and balanced accuracy on the later chronological test.
4. **Four-grade auxiliary supervision improved ranking but changed
   calibration.** It produced the best AUC and tuned hard decision, while the
   binary-only model retained the best fixed-0.5 decision and better log loss.
5. **The best model depends on deployment policy.** Use binary-only
   `hierarchical-rwkv` if 0.5 must remain the decision boundary. Use the
   multitask model if a validation-calibrated cost threshold is permitted.

## Important limitations

- This is one user, one deck match, one chronological split, and one random
  seed.
- The test period was inspected during exploratory architecture development,
  including one-epoch smoke runs. These numbers must not be treated as an
  untouched confirmatory evaluation.
- The fixed 32-event window is not full-history RWKV inference.
- Deck and user streams may be partly redundant when most selected cards belong
  to the same deck.
- The current objective balances Again and Success. It does not yet test a grid
  of deployment-specific false-positive/false-negative costs.
- Training times are local CPU wall times and include all eight epochs, even
  when an earlier epoch was selected.

The next confirmatory run should use rolling chronological folds or newly
accumulated reviews, repeat several seeds, and report mean and dispersion. The
next architecture ablation should separate card, note, deck, and user streams
one at a time before increasing model size or attempting the full reference
RWKV-7 architecture.

## Reproduction

Build and run against a copied or extracted collection, never the live Anki
profile database:

```sh
cargo run --release --example revlog_classifier -- \
  --collection <copied-collection.anki21> \
  --deck-name-contains Yomitan \
  --architectures legacy-gru,rwkv-inputs-gru,hierarchical-gru,hierarchical-rwkv,hierarchical-rwkv-multitask \
  --epochs 8 \
  --batch-size 256 \
  --hidden-size 32
```

First export canonical historical RWKV-Instant predictions with Anki's packaged
model and production Rust inference path. Replay the full eligible collection
history; do not filter this step to the target deck because RWKV has global and
cross-entity recurrent state:

```sh
# Run from the Anki repository.
just rwkv-predict-bench \
  --weights qt/aqt/rwkv_inference/RWKV_trained_on_5000_10000.bin \
  --collection <copied-collection.anki21> \
  --warmup-reviews 0 \
  --queries 1 \
  --repeat 1 \
  --batch-size 1 \
  --metrics \
  --prediction-output <rwkv-instant.csv>
```

Then reproduce only the best classifier and join its test decisions to the
canonical RWKV-Instant export by revlog ID:

```sh
cargo run --release --example revlog_classifier -- \
  --collection <copied-collection.anki21> \
  --deck-name-contains Yomitan \
  --architectures hierarchical-rwkv-multitask \
  --epochs 8 \
  --batch-size 256 \
  --hidden-size 32 \
  --analyze-r-distributions \
  --rwkv-instant-predictions <rwkv-instant.csv>
```

The executable prints CSV-compatible `result,...` lines containing thresholds,
selected epochs, training times, log loss, AUC, accuracy, balanced accuracy, and
the full confusion matrix. `--analyze-r-distributions` requires a canonical
`--rwkv-instant-predictions` export and additionally prints
`distribution_summary`, `distribution_histogram`, and `distribution_correlation`
records.

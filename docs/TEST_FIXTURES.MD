# Test Fixtures

Some tests read local fixtures from `tests/data`. This directory is ignored by git, so fixture files must be restored locally when a test needs the full external dataset.

## Files

- `tests/data/collection.anki21` is required by conversion, training, inference, dataset, and simulation tests that use the Anki sample collection.
- `tests/data/revlog.csv` is an optional extra training dataset. Broad training tests include it when present and skip only that dataset when absent.

## CSV Format

`tests/data/revlog.csv` is read as `RevlogCsv` with these columns:

```text
card_id,review_time,review_rating,review_state,review_duration
```

`review_state` is mapped to Anki review kinds as follows:

- `0` or `1`: learning
- `2`: review
- `3`: relearning
- `4`: filtered
- `5`: manual

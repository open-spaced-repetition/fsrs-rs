# FSRS for Rust

[![crates.io](https://img.shields.io/crates/v/fsrs.svg)](https://crates.io/crates/fsrs) ![](https://github.com/open-spaced-repetition/fsrs-rs/actions/workflows/check.yml/badge.svg)

The Free Spaced Repetition Scheduler ([FSRS](https://github.com/open-spaced-repetition/fsrs4anki/wiki/The-Algorithm)) is a modern spaced repetition algorithm. It springs from [MaiMemo's DHP model](https://www.maimemo.com/paper/), which is a variant of the [DSR model](https://supermemo.guru/wiki/Three_component_model_of_memory) proposed by [Piotr Wozniak](https://supermemo.guru/wiki/Piotr_Wozniak).

FSRS-rs is a Rust implementation of FSRS with full training support using [Burn](https://github.com/tracel-ai/burn). It also provides simulation capabilities and basic scheduling functionality.

For more information about the algorithm, please refer to [the wiki page of FSRS](https://github.com/open-spaced-repetition/fsrs4anki/wiki/The-Algorithm).

---

## Quickstart

### Add the crate

Add FSRS to your project:

```toml
[dependencies]
fsrs = "5.2.0"
```

The scheduling example below also uses `chrono` to track review times:

```toml
chrono = { version = "0.4", default-features = false, features = ["std", "clock"] }
```

Run `cargo run --example <name>` to see the complete samples ([`schedule`](examples/schedule.rs), [`migrate`](examples/migrate.rs), [`optimize`](examples/optimize.rs)).

### Schedule reviews

```rust
use chrono::{Duration, Utc};
use fsrs::{FSRS, MemoryState};

let fsrs = FSRS::default();
let desired_retention = 0.9;
let previous_state: Option<MemoryState> = None;
let elapsed_days = 0;

let next_states = fsrs.next_states(previous_state, desired_retention, elapsed_days)?;
let review = next_states.good;

let interval_days = review.interval.round().max(1.0) as u32;
let due = Utc::now() + Duration::days(interval_days as i64);
```

Replace `previous_state`/`elapsed_days` with a stored `MemoryState` and the number of days since the prior review when scheduling existing cards. Full example: [`examples/schedule.rs`](examples/schedule.rs).

### Optimize parameters from review logs

```rust
use chrono::NaiveDate;
use fsrs::{ComputeParametersInput, FSRSItem, FSRSReview, compute_parameters};

let history = vec![
    (NaiveDate::from_ymd_opt(2023, 1, 1).unwrap(), 3),
    (NaiveDate::from_ymd_opt(2023, 1, 5).unwrap(), 4),
];

let mut accumulated = Vec::new();
let mut items = Vec::new();
let mut last = history[0].0;

for (date, rating) in history {
    let delta_t = (date - last).num_days() as u32;
    accumulated.push(FSRSReview { rating, delta_t });
    items.push(FSRSItem {
        reviews: accumulated.clone(),
    });
    last = date;
}

let parameters = compute_parameters(ComputeParametersInput {
    train_set: items,
    ..Default::default()
})?;
```

Feed the optimizer a vector of `FSRSItem` instances built from your review history; the returned parameters can then be persisted or supplied to schedulers. Full example: [`examples/optimize.rs`](examples/optimize.rs).

### Migrate from SM-2 style data

```rust
use fsrs::{FSRS, FSRSItem, FSRSReview};

let fsrs = FSRS::default();
let sm2_retention = 0.9;
let ease_factor = 2.5;
let interval = 10.0;

let initial_state = fsrs.memory_state_from_sm2(ease_factor, interval, sm2_retention)?;

let reviews = vec![
    FSRSReview { rating: 3, delta_t: 5 },
    FSRSReview { rating: 4, delta_t: 10 },
];

let memory_state = fsrs.memory_state(
    FSRSItem { reviews },
    Some(initial_state),
)?;
```

Use `memory_state_from_sm2` when you only have the latest SM-2 ease/interval; pass the result as the starting point while you replay any partial review history. Full example: [`examples/migrate.rs`](examples/migrate.rs).


## Online development

You can use <https://idx.google.com/import>.

## Local development

add

```sh
#!/bin/sh
cargo fmt
cargo clippy -- -D warnings
git add .
```

to `.git/hooks/pre-commit`, then `chmod +x .git/hooks/pre-commit`

## Bindings

- c <https://github.com/open-spaced-repetition/fsrs-rs-c>
- python <https://github.com/open-spaced-repetition/fsrs-rs-python>
- nodejs <https://github.com/open-spaced-repetition/fsrs-rs-nodejs>
- dart <https://github.com/open-spaced-repetition/fsrs-rs-dart>
- php <https://github.com/open-spaced-repetition/fsrs-rs-php>

## Q&A

- What is the difference between `fsrs-rs` and [`rs-fsrs`](https://github.com/open-spaced-repetition/rs-fsrs)

  If you only want to schedule cards, use \[lang\]-fsrs or the [bindings](https://github.com/open-spaced-repetition/rs-fsrs?tab=readme-ov-file#bindings).

  If you need to optimize, use this crate or its bindings.

- Why use two crates instead of one?

  Calculating the weights involves tensor operations so the data types are different (Tensor vs Vec/Slice). If we were to use one crate, this would mean using `cfg` to change the variable type, which would be tedious. Because of this, instead we publish two separate crates.

  Another reason is, it would be hard to port to other languages while using `Tensor`s.

- What about the name?

  Before this crate was made, `go-fsrs` and other libraries already existed, so the name `rs-fsrs` was chosen.

  Then we wanted to port the torch version to Rust so that everyone could optimize on their own devices (tch-rs uses libtorch which is too heavy). Since the algorithm is called `fsrs`, we add an `-rs` on the end.

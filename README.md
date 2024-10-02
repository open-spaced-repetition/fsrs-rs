# FSRS for Rust

[![crates.io](https://img.shields.io/crates/v/fsrs.svg)](https://crates.io/crates/fsrs) ![](https://github.com/open-spaced-repetition/fsrs-rs/actions/workflows/check.yml/badge.svg)

This crate contains a Rust API for training FSRS parameters, and for using them to schedule cards.

The Free Spaced Repetition Scheduler ([FSRS](https://github.com/open-spaced-repetition/fsrs4anki)) is a modern spaced repetition algorithm. It is based on the [DSR model](https://supermemo.guru/wiki/Three_component_model_of_memory) proposed by [Piotr Wozniak](https://supermemo.guru/wiki/Piotr_Wozniak), the creator of SuperMemo.

FSRS-rs is a Rust implementation of FSRS. It is designed to be used in [Anki](https://apps.ankiweb.net/), a popular spaced repetition software. [Anki 23.10](https://github.com/ankitects/anki/releases/tag/23.10) has already integrated FSRS as an alternative scheduler.

For more information about the algorithm, please refer to [the wiki page of FSRS](https://github.com/open-spaced-repetition/fsrs4anki/wiki/The-Algorithm).

---

## Quickstart

Read up [this](https://github.com/open-spaced-repetition/fsrs4anki/wiki/The-Optimal-Retention) to determine the optimal retention for your use case.

```rust
// Pick to your liking (see above)
let optimal_retention = 0.75;
// Use default parameters/Weights for scheduler
let fsrs = FSRS::new(Some(&[]))?;

// Create a completely new card
let day1_states = fsrs.next_states(None, optimal_retention, 0)?;

// Rate as `hard` on first day
let day1 = day1_states.hard;
dbg!(&day1); // scheduled as `in 4 days`

// Now we review the card 2 days later
let day3_states = fsrs.next_states(Some(day1.memory), optimal_retention, 2)?;

// Rate as `good` this time
let day3 = day3_states.good;
dbg!(day3);
```

## Development

add

```sh
#!/bin/sh
cargo fmt
cargo clippy -- -D warnings
git add .
```

to `.git/hooks/pre-commit`, then `chmod +x .git/hooks/pre-commit`

## Bindings

- python <https://github.com/open-spaced-repetition/fsrs-rs-python>

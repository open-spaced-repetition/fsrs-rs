# FSRS for Rust

[![crates.io](https://img.shields.io/crates/v/fsrs.svg)](https://crates.io/crates/fsrs) ![](https://github.com/open-spaced-repetition/fsrs-rs/actions/workflows/check.yml/badge.svg)

This crate contains a Rust API for training FSRS parameters, and for using them to schedule cards.

The Free Spaced Repetition Scheduler ([FSRS](https://github.com/open-spaced-repetition/fsrs4anki)) is a modern spaced repetition algorithm. It is based on the [DSR model](https://supermemo.guru/wiki/Three_component_model_of_memory) proposed by [Piotr Wozniak](https://supermemo.guru/wiki/Piotr_Wozniak), the creator of SuperMemo.

FSRS-rs is a Rust implementation of FSRS. It is designed to be used in [Anki](https://apps.ankiweb.net/), a popular spaced repetition software. [Anki 23.10](https://github.com/ankitects/anki/releases/tag/23.10) has already integrated FSRS as an alternative scheduler.

For more information about the algorithm, please refer to [the wiki page of FSRS](https://github.com/open-spaced-repetition/fsrs4anki/wiki/The-Algorithm).

---

## Quickstart

Read [this](https://github.com/open-spaced-repetition/fsrs4anki/wiki/The-Optimal-Retention) for an explanation of how to determine the optimal retention for your use case.

```rust
// Pick whichever percentage is to your liking (see above)
let optimal_retention = 0.75;
// Use default parameters/weights for the scheduler
let fsrs = FSRS::new(Some(&[]))?;

// Create a completely new card
let day1_states = fsrs.next_states(None, optimal_retention, 0)?;

// Rate as `hard` on the first day
let day1 = day1_states.hard;
dbg!(&day1); // scheduled as `in 4 days`

// Now we review the card 2 days later
let day3_states = fsrs.next_states(Some(day1.memory), optimal_retention, 2)?;

// Rate as `good` this time
let day3 = day3_states.good;
dbg!(day3);
```

## Online development

go to <https://idx.google.com/import>

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

  If you only want to schedule cards, use \[lang\]-fsrs or the [bindings](https://github.com/open-spaced-repetition/rs-fsrs?tab=readme-ov-file#bindings),

  If you need to optimize, use this crate or its bindings.

- Why use two crates instead of one?

  Calculating the weights involves tensor operations so the data types are different (Tensor vs Vec/Slice). In one crate, this would mean using `cfg` to change the variable type, which would be tedious. Because of this, instead we publish two seperate crates.

  Another reason is, it would be hard to port to other languages while using `Tensor`s.

- What about the name?

  Before this crate was made, `go-fsrs` and other libraries already existed, so the name `rs-fsrs` was chosen.

  Then we wanted to port the torch version to Rust so that everyone could optimize on their own devices (tch-rs use libtorch which is too heavy). Since the algorithm is called `fsrs`, we add an `-rs` on the end.

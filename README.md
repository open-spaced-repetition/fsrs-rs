# FSRS for Rust

![](https://github.com/open-spaced-repetition/fsrs-rs/actions/workflows/check.yml/badge.svg)

This crate contains a Rust API for training FSRS weights, and for using them to schedule cards.

**Motivation**: We plan to integrate [FSRS](https://github.com/open-spaced-repetition/fsrs4anki), a modern spaced repetition algorithm, into [Anki](https://github.com/ankitects/anki), which requires a localized optimization module to train the parameters from users' review logs.

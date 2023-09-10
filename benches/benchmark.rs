// Copyright: Ankitects Pty Ltd and contributors
// License: GNU AGPL, version 3 or later; http://www.gnu.org/licenses/agpl.html

use std::hint::black_box;
use std::iter::repeat;

use criterion::criterion_group;
use criterion::criterion_main;
use criterion::Criterion;
use fsrs_optimizer::FSRSReview;
use fsrs_optimizer::Fsrs;
use fsrs_optimizer::NextStates;
use fsrs_optimizer::{FSRSItem, MemoryState};
use itertools::Itertools;

pub(crate) fn calc_mem(inf: &Fsrs, past_reviews: usize) -> MemoryState {
    let review = FSRSReview {
        rating: 3,
        delta_t: 21,
    };
    let reviews = repeat(review.clone()).take(past_reviews + 1).collect_vec();
    inf.memory_state(FSRSItem { reviews })
}

pub(crate) fn next_states(inf: &Fsrs) -> NextStates {
    inf.next_states(
        Some(MemoryState {
            stability: 51.344814,
            difficulty: 7.005062,
        }),
        0.9,
        21,
    )
}

pub fn criterion_benchmark(c: &mut Criterion) {
    let inf = Fsrs::new(Some(&[
        0.81497127,
        1.5411042,
        4.007436,
        9.045982,
        4.9264183,
        1.039322,
        0.93803364,
        0.0,
        1.5530516,
        0.10299722,
        0.9981442,
        2.210701,
        0.018248068,
        0.3422524,
        1.3384504,
        0.22278537,
        2.6646678,
    ]));

    c.bench_function("calc_mem", |b| b.iter(|| black_box(calc_mem(&inf, 100))));
    c.bench_function("next_states", |b| b.iter(|| black_box(next_states(&inf))));
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);

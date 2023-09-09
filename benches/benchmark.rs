// Copyright: Ankitects Pty Ltd and contributors
// License: GNU AGPL, version 3 or later; http://www.gnu.org/licenses/agpl.html

use std::hint::black_box;
use std::iter::repeat;

use criterion::criterion_group;
use criterion::criterion_main;
use criterion::Criterion;
use fsrs_optimizer::FSRSItem;
use fsrs_optimizer::FSRSReview;
use fsrs_optimizer::NextIntervals;
use itertools::Itertools;

pub(crate) fn next_intervals(weights: &[f32], past_reviews: usize) -> NextIntervals {
    let review = FSRSReview {
        rating: 3,
        delta_t: 21,
    };
    let reviews = repeat(review.clone()).take(past_reviews + 1).collect_vec();
    let item = FSRSItem { reviews };
    item.next_intervals(weights, 0.9)
}

pub fn criterion_benchmark(c: &mut Criterion) {
    let weights = &[
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
    ];

    c.bench_function("next_intervals_1", |b| {
        b.iter(|| black_box(next_intervals(weights, 1)))
    });
    c.bench_function("next_intervals_10", |b| {
        b.iter(|| black_box(next_intervals(weights, 10)))
    });
    c.bench_function("next_intervals_100", |b| {
        b.iter(|| black_box(next_intervals(weights, 100)))
    });
    c.bench_function("next_intervals_1000", |b| {
        b.iter(|| black_box(next_intervals(weights, 1000)))
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);

// Copyright: Ankitects Pty Ltd and contributors
// License: GNU AGPL, version 3 or later; http://www.gnu.org/licenses/agpl.html

use std::hint::black_box;
use std::iter::repeat;

use criterion::criterion_group;
use criterion::criterion_main;
use criterion::Criterion;
use fsrs_optimizer::calc_memo_state;
use fsrs_optimizer::next_interval;
use fsrs_optimizer::next_memo_state;
use fsrs_optimizer::FSRSItem;
use fsrs_optimizer::FSRSReview;
use itertools::Itertools;

pub(crate) fn benchmark_review_next(weights: &[f32]) {
    let review = FSRSReview {
        rating: 3,
        delta_t: 21,
    };
    let (s, _d) = next_memo_state(weights, review, 5, 20.925528, 7.005062);
    assert_eq!(51, next_interval(s, 0.9));
}

pub(crate) fn benchmark_review_all(weights: &[f32], past_reviews: usize) {
    let review = FSRSReview {
        rating: 3,
        delta_t: 21,
    };
    let reviews = repeat(review.clone()).take(past_reviews + 1).collect_vec();
    let item = FSRSItem { reviews };
    calc_memo_state(weights, item);
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

    c.bench_function("review_next", |b| {
        b.iter(|| black_box(benchmark_review_next(weights)))
    });
    c.bench_function("review_all_1", |b| {
        b.iter(|| black_box(benchmark_review_all(weights, 1)))
    });
    c.bench_function("review_all_10", |b| {
        b.iter(|| black_box(benchmark_review_all(weights, 10)))
    });
    c.bench_function("review_all_100", |b| {
        b.iter(|| black_box(benchmark_review_all(weights, 100)))
    });
    c.bench_function("review_all_1000", |b| {
        b.iter(|| black_box(benchmark_review_all(weights, 1000)))
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);

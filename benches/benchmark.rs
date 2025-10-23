// Copyright: Ankitects Pty Ltd and contributors
// License: GNU AGPL, version 3 or later; http://www.gnu.org/licenses/agpl.html

use std::hint::black_box;
use std::iter::repeat;

use criterion::criterion_group;
use criterion::criterion_main;
use criterion::{Criterion, Throughput};
use fsrs::FSRS;
use fsrs::FSRSReview;
use fsrs::NextStates;
use fsrs::{FSRS6_DEFAULT_DECAY, current_retrievability};
use fsrs::{FSRSItem, MemoryState};
use itertools::Itertools;

pub(crate) fn calc_mem(inf: &FSRS, past_reviews: usize, card_cnt: usize) -> Vec<MemoryState> {
    let review = FSRSReview {
        rating: 3,
        delta_t: 21,
    };
    let reviews = repeat(review).take(past_reviews + 1).collect_vec();
    (0..card_cnt)
        .map(|_| {
            inf.memory_state(
                FSRSItem {
                    reviews: reviews.clone(),
                },
                None,
            )
            .unwrap()
        })
        .collect_vec()
}

pub(crate) fn calc_mem_batch(inf: &FSRS, past_reviews: usize, card_cnt: usize) -> Vec<MemoryState> {
    let reviews = repeat(FSRSReview {
        rating: 3,
        delta_t: 21,
    })
    .take(past_reviews)
    .collect_vec();
    let items = repeat(FSRSItem {
        reviews: reviews.clone(),
    })
    .take(card_cnt)
    .collect_vec();
    inf.memory_state_batch(items, vec![None; card_cnt]).unwrap()
}

pub(crate) fn next_states(inf: &FSRS) -> NextStates {
    inf.next_states(
        Some(MemoryState {
            stability: 51.344814,
            difficulty: 7.005062,
        }),
        0.9,
        21,
    )
    .unwrap()
}

pub fn criterion_benchmark(c: &mut Criterion) {
    let fsrs = FSRS::new(&[
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
    ])
    .unwrap();

    c.bench_function("next_states", |b| b.iter(|| black_box(next_states(&fsrs))));

    c.bench_function("current_retrievability", |b| {
        let state = MemoryState {
            stability: 51.344814,
            difficulty: 7.005062,
        };
        b.iter(|| {
            black_box(current_retrievability(
                state,
                black_box(21.0),
                black_box(FSRS6_DEFAULT_DECAY),
            ))
        })
    });

    {
        let mut single_group = c.benchmark_group("calc_mem");
        let n_cards = 1000;
        let n_reviews = 10;
        single_group.throughput(Throughput::Elements(n_cards));
        single_group.bench_function(
            format!("calc_mem n_cards={n_cards}, n_reviews={n_reviews}"),
            |b| b.iter(|| black_box(calc_mem(&fsrs, n_reviews, n_cards.try_into().unwrap()))),
        );
        single_group.finish();
    }

    {
        let mut batch_group = c.benchmark_group("calc_mem_batch");
        for n_cards in [1000, 10_000] {
            for n_reviews in [10, 100, 200] {
                batch_group.throughput(Throughput::Elements(n_cards));
                batch_group.bench_function(
                    format!("calc_mem_batch n_cards={n_cards}, n_reviews={n_reviews}"),
                    |b| {
                        b.iter(|| {
                            black_box(calc_mem_batch(
                                &fsrs,
                                n_reviews,
                                n_cards.try_into().unwrap(),
                            ))
                        })
                    },
                );
            }
        }
        batch_group.finish();
    }
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);

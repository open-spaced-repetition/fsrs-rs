use criterion::{Criterion, criterion_group, criterion_main};
use fsrs::{Card, expected_workload, expected_workload_with_existing_cards};
use fsrs::{DEFAULT_PARAMETERS, SimulationResult, SimulatorConfig, optimal_retention, simulate};
use fsrs::{FSRS, FSRSError};
use rayon::iter::IntoParallelIterator;
use rayon::iter::ParallelIterator;
use std::hint::black_box;

pub(crate) fn parallel_simulate(config: &SimulatorConfig) -> Result<Vec<f32>, FSRSError> {
    (70..=99)
        .into_par_iter()
        .map(|i| {
            let SimulationResult {
                memorized_cnt_per_day,
                cost_per_day,
                ..
            } = simulate(
                config,
                &DEFAULT_PARAMETERS,
                i as f32 / 100.0,
                Some((i + 42).try_into().unwrap()),
                None,
            )?;
            let total_memorized = memorized_cnt_per_day[memorized_cnt_per_day.len() - 1];
            let total_cost = cost_per_day.iter().sum::<f32>();
            Ok(total_cost / total_memorized)
        })
        .collect()
}

pub(crate) fn bench_optimal_retention(_inf: &FSRS, config: &SimulatorConfig) -> f32 {
    optimal_retention(config, &[], |_v| true, None, None).unwrap()
}

pub(crate) fn run_expected_workload_for_30_retentions() {
    let config = SimulatorConfig::default();

    for i in 0..30 {
        let desired_retention = 0.70 + (i as f32 * 0.01);
        black_box(expected_workload(&DEFAULT_PARAMETERS, desired_retention, &config).unwrap());
    }
}

pub(crate) fn run_expected_workload_with_10000_existing_cards() {
    let config = SimulatorConfig {
        deck_size: 46500,
        learn_limit: 100,
        ..Default::default()
    };
    let mut cards = Vec::with_capacity(10000);
    for i in 0..10000 {
        cards.push(Card {
            id: i,
            difficulty: 5.0,
            stability: 5.0,
            last_date: 0.0,
            due: 10.0,
            interval: 10.0,
            lapses: 0,
            desired_retention: 0.9,
        });
    }
    black_box(
        expected_workload_with_existing_cards(&DEFAULT_PARAMETERS, 0.9, &config, &cards).unwrap(),
    );
}

pub fn criterion_benchmark(c: &mut Criterion) {
    let fsrs = FSRS::default();
    let config = SimulatorConfig {
        deck_size: 36500,
        learn_span: 90,
        max_cost_perday: f32::INFINITY,
        learn_limit: 100,
        review_limit: 600,
        ..Default::default()
    };
    c.bench_function("parallel_simulate", |b| {
        b.iter(|| black_box(parallel_simulate(&config)))
    });
    c.bench_function("optimal_retention", |b| {
        b.iter(|| black_box(bench_optimal_retention(&fsrs, &config)))
    });
    c.bench_function("expected_workload_30_retentions", |b| {
        b.iter(run_expected_workload_for_30_retentions)
    });
    c.bench_function("expected_workload_with_10000_existing_cards", |b| {
        b.iter(run_expected_workload_with_10000_existing_cards)
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);

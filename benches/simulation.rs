use criterion::{Criterion, criterion_group, criterion_main};
use fsrs::{DEFAULT_PARAMETERS, SimulationResult, SimulatorConfig, simulate};
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
                &config,
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

pub(crate) fn optimal_retention(inf: &FSRS, config: &SimulatorConfig) -> f32 {
    inf.optimal_retention(config, &[], |_v| true).unwrap()
}

pub fn criterion_benchmark(c: &mut Criterion) {
    let fsrs = FSRS::new(Some(&DEFAULT_PARAMETERS)).unwrap();
    let config = SimulatorConfig {
        deck_size: 36500,
        learn_span: 90,
        max_cost_perday: f32::INFINITY,
        learn_limit: 100,
        review_limit: 600,
        loss_aversion: 1.0,
        ..Default::default()
    };
    c.bench_function("parallel_simulate", |b| {
        b.iter(|| black_box(parallel_simulate(&config)))
    });
    c.bench_function("optimal_retention", |b| {
        b.iter(|| black_box(optimal_retention(&fsrs, &config)))
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);

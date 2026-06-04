use criterion::{Criterion, criterion_group, criterion_main};
use fsrs::{CostAdrPolicy, CostAdrTrainingConfig, DEFAULT_PARAMETERS, SimulatorConfig};
use std::hint::black_box;

fn benchmark_cost_adr_train_single_user(c: &mut Criterion) {
    let simulator_config = SimulatorConfig {
        deck_size: 1_500,
        learn_span: 180,
        learn_limit: 20,
        review_limit: 9_999,
        max_cost_perday: 720.0 * 60.0,
        ..Default::default()
    };
    let training_config = CostAdrTrainingConfig {
        population_size: 4,
        generations: 3,
        early_stop_patience_generations: 0,
        seed: Some(42),
        simulation_seed: Some(42),
        cost_weights: vec![0.0, 64.0, 1024.0],
        baseline_desired_retentions: vec![0.70, 0.85, 0.95],
        ..Default::default()
    };

    let mut group = c.benchmark_group("cost_adr");
    group.sample_size(10);
    group.bench_function("train_single_user", |b| {
        b.iter(|| {
            CostAdrPolicy::train_single_user(
                black_box(&simulator_config),
                black_box(&DEFAULT_PARAMETERS),
                black_box(&training_config),
            )
            .unwrap();
        })
    });
    group.finish();
}

criterion_group!(benches, benchmark_cost_adr_train_single_user);
criterion_main!(benches);

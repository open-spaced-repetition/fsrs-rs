// Comprehensive CPU (NdArray) vs GPU (WGPU) performance comparison
// Tests various combinations of batch sizes and review history lengths

use std::hint::black_box;
use std::iter::repeat;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use itertools::Itertools;

use burn::backend::wgpu::{Wgpu, WgpuDevice};
use fsrs::{FSRS, FSRSItem, FSRSReview, MemoryState};

fn create_test_items(n_cards: usize, n_reviews: usize) -> Vec<FSRSItem> {
    let reviews = repeat(FSRSReview {
        rating: 3,
        delta_t: 21,
    })
    .take(n_reviews)
    .collect_vec();

    repeat(FSRSItem {
        reviews: reviews.clone(),
    })
    .take(n_cards)
    .collect_vec()
}

fn bench_cpu(params: &[f32], items: &[FSRSItem]) -> Vec<MemoryState> {
    let fsrs = FSRS::new(params).unwrap();
    fsrs.memory_state_batch(items.to_vec(), vec![None; items.len()])
        .unwrap()
}

fn bench_gpu(
    params: &[f32],
    items: &[FSRSItem],
    device: &<Wgpu as burn::tensor::backend::Backend>::Device,
) -> Vec<MemoryState> {
    let fsrs: FSRS<Wgpu> = FSRS::<Wgpu>::new_with_backend(params, device).unwrap();
    fsrs.memory_state_batch(items.to_vec(), vec![None; items.len()])
        .unwrap()
}

// Benchmark 1: Different batch sizes (fixed review history)
fn bench_varying_batch_size(c: &mut Criterion) {
    let params = vec![
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

    let mut group = c.benchmark_group("batch_size_comparison");
    let n_reviews = 10; // Fixed review count

    for &n_cards in &[10, 50, 100, 500, 1000, 5000, 10000] {
        let items = create_test_items(n_cards, n_reviews);
        group.throughput(Throughput::Elements(n_cards as u64));

        // CPU
        group.bench_with_input(BenchmarkId::new("cpu", n_cards), &items, |b, items| {
            b.iter(|| black_box(bench_cpu(&params, items)))
        });

        // GPU (only for larger batches to save time)
        if n_cards >= 100 {
            let device = WgpuDevice::default();
            group.bench_with_input(BenchmarkId::new("gpu", n_cards), &items, |b, items| {
                b.iter(|| black_box(bench_gpu(&params, items, &device)))
            });
        }
    }

    group.finish();
}

// Benchmark 2: Different review history lengths (fixed batch size)
fn bench_varying_history_length(c: &mut Criterion) {
    let params = vec![
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

    let mut group = c.benchmark_group("history_length_comparison");
    let n_cards = 1000; // Fixed batch size

    for &n_reviews in &[5, 10, 20, 50, 100, 200] {
        let items = create_test_items(n_cards, n_reviews);
        group.throughput(Throughput::Elements(n_cards as u64));

        // CPU
        group.bench_with_input(BenchmarkId::new("cpu", n_reviews), &items, |b, items| {
            b.iter(|| black_box(bench_cpu(&params, items)))
        });

        // GPU
        let device = WgpuDevice::default();
        group.bench_with_input(BenchmarkId::new("gpu", n_reviews), &items, |b, items| {
            b.iter(|| black_box(bench_gpu(&params, items, &device)))
        });
    }

    group.finish();
}

// Benchmark 3: Matrix of batch sizes × history lengths
fn bench_combinations(c: &mut Criterion) {
    let params = vec![
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

    let mut group = c.benchmark_group("combinations");

    // Test key combinations
    let test_cases = vec![
        (100, 10),
        (100, 100),
        (1000, 10),
        (1000, 50),
        (1000, 100),
        (10000, 10),
        (10000, 50),
    ];

    for (n_cards, n_reviews) in test_cases {
        let items = create_test_items(n_cards, n_reviews);
        let label = format!("{}cards_{}reviews", n_cards, n_reviews);
        group.throughput(Throughput::Elements(n_cards as u64));

        // CPU
        group.bench_with_input(BenchmarkId::new("cpu", &label), &items, |b, items| {
            b.iter(|| black_box(bench_cpu(&params, items)))
        });

        // GPU
        let device = WgpuDevice::default();
        group.bench_with_input(BenchmarkId::new("gpu", &label), &items, |b, items| {
            b.iter(|| black_box(bench_gpu(&params, items, &device)))
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_varying_batch_size,
    bench_varying_history_length,
    bench_combinations
);
criterion_main!(benches);

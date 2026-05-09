use criterion::{Criterion, criterion_group, criterion_main};
use fsrs::{ComputeParametersInput, FSRSItem, FSRSReview, compute_parameters};
use std::hint::black_box;

fn synthetic_items() -> Vec<FSRSItem> {
    let templates: &[&[(u32, u32)]] = &[
        &[(0, 3), (1, 4), (4, 3), (14, 4), (31, 3), (50, 4)],
        &[(0, 2), (1, 3), (3, 4), (11, 3), (27, 4), (45, 3), (63, 4)],
        &[(0, 4), (7, 4), (23, 3), (40, 4), (59, 3)],
        &[(0, 1), (1, 1), (2, 3), (5, 4), (15, 4), (31, 3), (50, 4)],
        &[(0, 3), (2, 3), (7, 2), (9, 4), (21, 3), (35, 4), (55, 3)],
        &[
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 4),
            (9, 3),
            (19, 4),
            (35, 3),
            (55, 4),
        ],
    ];

    templates
        .iter()
        .cycle()
        .take(200)
        .flat_map(|template| {
            let mut reviews = Vec::with_capacity(template.len());
            let mut items = Vec::with_capacity(template.len());
            let mut last_day = template[0].0;
            for (day, rating) in *template {
                let delta_t = day - last_day;
                reviews.push(FSRSReview {
                    rating: *rating,
                    delta_t: delta_t as f32,
                });
                items.push(FSRSItem {
                    reviews: reviews.clone(),
                });
                last_day = *day;
            }
            items
                .into_iter()
                .filter(|item| item.long_term_review_cnt() > 0)
                .collect::<Vec<_>>()
        })
        .collect()
}

fn benchmark_compute_parameters_fsrs7_penalty(c: &mut Criterion) {
    let items = synthetic_items();

    let input = ComputeParametersInput {
        train_set: items,
        progress: None,
        enable_short_term: true,
        num_relearning_steps: None,
        ..Default::default()
    };

    let mut group = c.benchmark_group("fsrs7_penalty");
    group.sample_size(10);
    group.bench_function("compute_parameters", |b| {
        b.iter(|| compute_parameters(black_box(input.clone())).unwrap())
    });
    group.finish();
}

criterion_group!(benches, benchmark_compute_parameters_fsrs7_penalty);
criterion_main!(benches);

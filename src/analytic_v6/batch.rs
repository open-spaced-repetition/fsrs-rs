//! SIMD layout of an FSRS-6 batch. Columns (items or cards) are packed eight
//! per group, one `Review` row per review step, and all groups of a batch
//! share one contiguous buffer. Elapsed days are rounded here, once per batch.

use super::plan::{BatchPlan, CardPlan};
use super::simd::{self, LANES, PARAMS, Review, signed_weight};
use super::{RuntimeParams, S_MAX};
use crate::FSRSReview;
use crate::dataset::WeightedFSRSItem;
use std::ops::Range;
use wide::f32x8 as F;

pub(crate) use super::simd::Workspace;

pub(crate) struct Batch {
    groups: Vec<Group>,
    reviews: Vec<Review>,
    pub(crate) predictions: usize,
}

struct Group {
    rows: Range<usize>,
    /// Every elapsed time is within `[0, S_MAX]`; see `simd::group_gradient`.
    bounded: bool,
}

impl BatchPlan {
    pub(crate) fn layout(&self, items: &[WeightedFSRSItem]) -> Batch {
        match self {
            Self::Prefixes(indices) => Batch::from_prefixes(items, indices),
            Self::Cards { plan, range, .. } => Batch::from_cards(items, plan, range.clone()),
        }
    }

    pub(super) fn group_count(&self) -> usize {
        match self {
            Self::Prefixes(indices) => indices.len().div_ceil(LANES),
            Self::Cards { range, .. } => range.len().div_ceil(LANES),
        }
    }
}

impl Batch {
    /// Each item is a lane whose last review is the scored target.
    fn from_prefixes(items: &[WeightedFSRSItem], indices: &[usize]) -> Self {
        let (groups, reviews) = pack(
            indices,
            |&index| items[index].item.reviews.len(),
            |rows, lane, &index| {
                let item = &items[index];
                write_history(rows, lane, &item.item.reviews);
                let target = &mut rows[item.item.reviews.len() - 1];
                set_target(target, lane, item);
                // The target only scores the prediction; it updates no state.
                target.rating.as_mut_array()[lane] = 0.0;
            },
        );
        Self {
            groups,
            reviews,
            predictions: indices.len(),
        }
    }

    /// Each card is a lane following its longest history; every item of the
    /// card scores the step at its own last review.
    fn from_cards(items: &[WeightedFSRSItem], plan: &CardPlan, range: Range<usize>) -> Self {
        let cards = plan.cards(range);
        let (groups, reviews) = pack(
            cards,
            |card| card.length,
            |rows, lane, card| {
                write_history(rows, lane, &items[card.longest].item.reviews);
                // In input order: a duplicate prefix keeps the last weight.
                for index in plan.items(card) {
                    let item = &items[index];
                    set_target(&mut rows[item.item.reviews.len() - 1], lane, item);
                }
            },
        );
        Self {
            groups,
            reviews,
            predictions: cards.iter().map(|card| card.count).sum(),
        }
    }

    /// Packs from flat row-major arrays (`row * columns + col`). With
    /// `target_t`, each column is a prefix scored at row `lengths[col]`;
    /// otherwise every weighted row is scored.
    #[cfg(test)]
    pub(crate) fn new(
        t: &[f32],
        ratings: &[f32],
        lengths: &[usize],
        target_t: Option<&[f32]>,
        labels: &[f32],
        weights: &[f32],
    ) -> Self {
        let columns = lengths.len();
        let indices: Vec<_> = (0..columns).collect();
        let (groups, reviews) = pack(
            &indices,
            |&col| lengths[col] + usize::from(target_t.is_some()),
            |rows, lane, &col| {
                for (row, review) in rows.iter_mut().enumerate().take(lengths[col]) {
                    let index = row * columns + col;
                    review.t.as_mut_array()[lane] = t[index];
                    review.rating.as_mut_array()[lane] = ratings[index];
                    if target_t.is_none() {
                        review.weight.as_mut_array()[lane] =
                            signed_weight(weights[index], labels[index] == 1.0);
                    }
                }
                if let Some(target_t) = target_t {
                    let target = &mut rows[lengths[col]];
                    target.t.as_mut_array()[lane] = target_t[col];
                    target.weight.as_mut_array()[lane] =
                        signed_weight(weights[col], labels[col] == 1.0);
                }
            },
        );
        Self {
            groups,
            reviews,
            predictions: 0,
        }
    }

    pub(super) fn group_count(&self) -> usize {
        self.groups.len()
    }

    pub(super) fn group_gradient(
        &self,
        group: usize,
        p: &RuntimeParams<'_>,
        workspace: &mut Workspace,
    ) -> [F; PARAMS] {
        let group = &self.groups[group];
        simd::group_gradient(
            p,
            &self.reviews[group.rows.clone()],
            group.bounded,
            workspace,
        )
    }

    /// Batch gradient on the calling thread. Lanes are summed in f32, then
    /// groups in f64 in order, exactly like the threaded reduction.
    pub(crate) fn gradient(&self, w: &[f32], workspace: &mut Workspace) -> Vec<f32> {
        let p = RuntimeParams::new(w);
        let mut total = [0.0f64; PARAMS];
        for group in 0..self.groups.len() {
            let grad = self.group_gradient(group, &p, workspace);
            for (sum, lanes) in total.iter_mut().zip(grad) {
                *sum += lanes.reduce_add() as f64;
            }
        }
        total.into_iter().map(|x| x as f32).collect()
    }
}

/// Packs `columns` eight per group. Each group has as many rows as its
/// longest column; `fill(rows, lane, column)` writes one lane.
fn pack<C>(
    columns: &[C],
    rows_of: impl Fn(&C) -> usize,
    mut fill: impl FnMut(&mut [Review], usize, &C),
) -> (Vec<Group>, Vec<Review>) {
    let group_rows = |chunk: &[C]| chunk.iter().map(&rows_of).max().unwrap_or(0);
    let total_rows = columns.chunks(LANES).map(group_rows).sum();
    let mut reviews = Vec::with_capacity(total_rows);
    let mut groups = Vec::with_capacity(columns.len().div_ceil(LANES));
    for chunk in columns.chunks(LANES) {
        let start = reviews.len();
        let rows = start..start + group_rows(chunk);
        reviews.resize(rows.end, Review::default());
        for (lane, column) in chunk.iter().enumerate() {
            fill(&mut reviews[rows.clone()], lane, column);
        }
        groups.push(Group::new(&mut reviews[rows.clone()], rows));
    }
    (groups, reviews)
}

fn write_history(rows: &mut [Review], lane: usize, reviews: &[FSRSReview]) {
    for (row, review) in rows.iter_mut().zip(reviews) {
        row.t.as_mut_array()[lane] = review.delta_t;
        row.rating.as_mut_array()[lane] = review.rating as f32;
    }
}

fn set_target(row: &mut Review, lane: usize, item: &WeightedFSRSItem) {
    row.weight.as_mut_array()[lane] = signed_weight(item.weight, item.item.current().rating > 1);
}

impl Group {
    fn new(group_reviews: &mut [Review], rows: Range<usize>) -> Self {
        let mut bounded = true;
        for review in group_reviews {
            review.t = elapsed_days(review.t);
            bounded &= simd::live(review.t, 0.0, S_MAX).all();
        }
        Self { rows, bounded }
    }
}

/// `fsrs6_elapsed_days` for eight lanes. `round_ties_even` differs from Rust's
/// ties-away `round` only at exact half-days, so only those are corrected.
/// Adding 0.5 before rounding would move values just below a half-day across
/// the boundary.
fn elapsed_days(t: F) -> F {
    let t = t.simd_ge(F::ZERO).select(t, F::ZERO);
    let rounded = t.round_ties_even();
    (t - rounded)
        .simd_eq(F::splat(0.5))
        .select(rounded + F::ONE, rounded)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{RngExt, SeedableRng, rngs::StdRng};

    #[test]
    fn vector_day_rounding_matches_scalar_including_half_day_boundaries() {
        let mut rng = StdRng::seed_from_u64(612);
        let mut values = vec![
            f32::NAN,
            f32::INFINITY,
            f32::NEG_INFINITY,
            -0.0,
            0.0,
            f32::MAX,
        ];
        for day in 0..1_000 {
            let half = day as f32 + 0.5;
            values.extend([
                f32::from_bits(half.to_bits() - 1),
                half,
                f32::from_bits(half.to_bits() + 1),
            ]);
        }
        values.extend((0..10_000).map(|_| f32::from_bits(rng.random())));
        for chunk in values.chunks(8) {
            let mut lanes = [0.0; 8];
            lanes[..chunk.len()].copy_from_slice(chunk);
            let expected = lanes.map(crate::training::fsrs6_elapsed_days);
            assert_eq!(
                elapsed_days(F::from(lanes)).to_array(),
                expected,
                "{lanes:?}"
            );
        }
    }
}

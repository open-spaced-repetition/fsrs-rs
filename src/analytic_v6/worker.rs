//! Two-thread FSRS-6 gradient engine. One helper thread per training run:
//!
//! 1. While the training thread starts on the first batches, the helper lays
//!    out the rest in first-epoch order.
//! 2. For each gradient request, both threads claim groups of the batch from a
//!    shared counter. Group results are reduced in group order, so the result
//!    is identical to `Batch::gradient` on one thread.
//! 3. While waiting, either thread frees the input items a few at a time.
//!
//! The helper stops when training returns, is cancelled or panics.

use super::RuntimeParams;
use super::batch::{Batch, Workspace};
use super::plan::BatchPlan;
use super::simd::PARAMS;
use crate::dataset::WeightedFSRSItem;
use std::sync::atomic::{AtomicU32, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Mutex, OnceLock};

/// `gradient(batch index, parameters)`, as consumed by the optimizer loop.
pub(crate) type BatchGradientFn<'a> = dyn FnMut(usize, &[f32]) -> Vec<f32> + 'a;

/// Runs `train` with a gradient function over `plans`. `first_order` is the
/// order in which the first epoch visits the batches.
pub(crate) fn with_gradients<T>(
    plans: &[BatchPlan],
    items: Vec<WeightedFSRSItem>,
    first_order: &[usize],
    train: impl FnOnce(&mut BatchGradientFn<'_>) -> T,
) -> T {
    if std::thread::available_parallelism().map_or(true, |n| n.get() < 2) {
        let batches: Vec<_> = plans.iter().map(|plan| plan.layout(&items)).collect();
        drop(items);
        let mut workspace = Workspace::default();
        return train(&mut |index, w| batches[index].gradient(w, &mut workspace));
    }

    let batches: Vec<OnceLock<Batch>> = plans.iter().map(|_| OnceLock::new()).collect();
    let max_groups = plans.iter().map(BatchPlan::group_count).max().unwrap_or(0);
    let shared = Shared::new(&batches, max_groups);
    // The training thread lays out the batches it needs first.
    let (lead, rest) = first_order.split_at(first_order.len().min(2));
    for &index in lead {
        let _ = batches[index].set(plans[index].layout(&items));
    }

    std::thread::scope(|scope| {
        let shared = &shared;
        scope.spawn(move || {
            let _stop_on_panic = StopOnDrop {
                shared,
                only_on_panic: true,
            };
            for &index in rest {
                if shared.stopped() {
                    return;
                }
                let _ = shared.batches[index].set(plans[index].layout(&items));
            }
            *shared.retired_items.lock().unwrap() = items;
            shared.serve_requests();
        });
        let _stop = StopOnDrop {
            shared,
            only_on_panic: false,
        };
        let mut workspace = Workspace::default();
        let mut seq = 0u32;
        let result = train(&mut |index, w| {
            seq += 1;
            shared.gradient(seq, index, w, &mut workspace)
        });
        shared.stop();
        while shared.drop_retired_items() {}
        result
    })
}

/// A job word holds the request sequence number (high half) and the next
/// unclaimed group (low half). `STOP` ends the helper.
const STOP: u64 = u64::MAX;

fn job(seq: u32, group: u32) -> u64 {
    (u64::from(seq) << 32) | u64::from(group)
}

struct Shared<'a> {
    batches: &'a [OnceLock<Batch>],
    job: AtomicU64,
    batch: AtomicUsize,
    parameters: [AtomicU32; PARAMS],
    outputs: Vec<GroupOutput>,
    retired_items: Mutex<Vec<WeightedFSRSItem>>,
}

/// Per-group result. Each output gets its own cache line so that threads
/// publishing neighboring groups do not invalidate each other's stores.
#[repr(align(64))]
struct GroupOutput {
    /// Lane-summed f32 gradients, packed in pairs to halve the stores.
    values: [AtomicU64; PARAMS.div_ceil(2)],
    /// Sequence number of the request these values belong to.
    ready: AtomicU32,
}

impl<'a> Shared<'a> {
    fn new(batches: &'a [OnceLock<Batch>], max_groups: usize) -> Self {
        Self {
            batches,
            job: AtomicU64::new(0),
            batch: AtomicUsize::new(0),
            parameters: std::array::from_fn(|_| AtomicU32::new(0)),
            outputs: (0..max_groups)
                .map(|_| GroupOutput {
                    values: std::array::from_fn(|_| AtomicU64::new(0)),
                    ready: AtomicU32::new(0),
                })
                .collect(),
            retired_items: Mutex::new(Vec::new()),
        }
    }

    fn stopped(&self) -> bool {
        self.job.load(Ordering::Acquire) == STOP
    }

    fn stop(&self) {
        self.job.store(STOP, Ordering::Release);
    }

    fn assert_helper_alive(&self) {
        assert!(!self.stopped(), "FSRS-6 training worker failed");
    }

    /// Training thread: publish a request, work on it, and reduce the groups.
    fn gradient(&self, seq: u32, index: usize, w: &[f32], workspace: &mut Workspace) -> Vec<f32> {
        let mut spins = 0;
        let batch = loop {
            if let Some(batch) = self.batches[index].get() {
                break batch;
            }
            self.assert_helper_alive();
            backoff(&mut spins);
        };
        for (dst, &v) in self.parameters.iter().zip(w) {
            dst.store(v.to_bits(), Ordering::Relaxed);
        }
        self.batch.store(index, Ordering::Relaxed);
        self.job.store(job(seq, 0), Ordering::Release);
        self.compute_groups(seq, w, workspace);

        let mut total = [0.0f64; PARAMS];
        for output in &self.outputs[..batch.group_count()] {
            let mut spins = 0;
            while output.ready.load(Ordering::Acquire) != seq {
                self.assert_helper_alive();
                if !self.drop_retired_items() {
                    backoff(&mut spins);
                }
            }
            for (pair, value) in total.chunks_mut(2).zip(&output.values) {
                let bits = value.load(Ordering::Relaxed);
                pair[0] += f32::from_bits(bits as u32) as f64;
                if let Some(high) = pair.get_mut(1) {
                    *high += f32::from_bits((bits >> 32) as u32) as f64;
                }
            }
        }
        total.into_iter().map(|v| v as f32).collect()
    }

    /// Helper thread: wait for requests and work on them until stopped.
    fn serve_requests(&self) {
        let mut workspace = Workspace::default();
        let mut seen = 0;
        let mut spins = 0;
        loop {
            let job = self.job.load(Ordering::Acquire);
            if job == STOP {
                while self.drop_retired_items() {}
                return;
            }
            let seq = (job >> 32) as u32;
            if seq == seen {
                if !self.drop_retired_items() {
                    backoff(&mut spins);
                }
                continue;
            }
            seen = seq;
            spins = 0;
            let w: [f32; PARAMS] =
                std::array::from_fn(|i| f32::from_bits(self.parameters[i].load(Ordering::Relaxed)));
            self.compute_groups(seq, &w, &mut workspace);
        }
    }

    /// Claims and computes groups of request `seq` until none remain.
    fn compute_groups(&self, seq: u32, w: &[f32], workspace: &mut Workspace) {
        let batch = self.batches[self.batch.load(Ordering::Relaxed)]
            .get()
            .expect("published batch");
        let p = RuntimeParams::new(w);
        loop {
            let current = self.job.load(Ordering::Acquire);
            let group = current as u32 as usize;
            if (current >> 32) as u32 != seq || group >= batch.group_count() {
                return;
            }
            if self
                .job
                .compare_exchange_weak(current, current + 1, Ordering::AcqRel, Ordering::Acquire)
                .is_err()
            {
                continue;
            }
            let grad = batch.group_gradient(group, &p, workspace);
            let output = &self.outputs[group];
            for (dst, pair) in output.values.iter().zip(grad.chunks(2)) {
                let low = pair[0].reduce_add().to_bits() as u64;
                let high = pair
                    .get(1)
                    .map_or(0, |lanes| lanes.reduce_add().to_bits() as u64);
                dst.store(low | (high << 32), Ordering::Relaxed);
            }
            output.ready.store(seq, Ordering::Release);
        }
    }

    /// Frees up to 32 input items; returns whether any were left to free.
    fn drop_retired_items(&self) -> bool {
        let Ok(mut items) = self.retired_items.try_lock() else {
            return false;
        };
        let len = items.len();
        let keep = len.saturating_sub(32);
        // Load the items' histories before freeing them, so the allocator's
        // pointer-chasing latency overlaps instead of serializing.
        let touch = items[keep..].iter().fold(0u32, |acc, item| {
            acc ^ item.item.reviews.first().map_or(0, |r| r.rating)
        });
        std::hint::black_box(touch);
        items.truncate(keep);
        len != 0
    }
}

/// Stops the helper when dropped: always, or only while unwinding a panic.
struct StopOnDrop<'a, 'b> {
    shared: &'a Shared<'b>,
    only_on_panic: bool,
}

impl Drop for StopOnDrop<'_, '_> {
    fn drop(&mut self) {
        if !self.only_on_panic || std::thread::panicking() {
            self.shared.stop();
        }
    }
}

fn backoff(spins: &mut u32) {
    if *spins < 256 {
        *spins += 1;
        std::hint::spin_loop();
    } else {
        std::thread::yield_now();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::FSRS6_DEFAULT_PARAMETERS;

    #[test]
    fn threaded_reduction_is_identical_to_serial_and_stops_on_return() {
        let fixture = crate::test_helpers::varied_card_prefixes();
        let (items, ids) = crate::test_helpers::items_of_card_prefixes(&fixture);
        let items: Vec<_> = items
            .into_iter()
            .zip(ids)
            .map(|(item, id)| WeightedFSRSItem {
                item,
                card_id: Some(id),
                weight: 0.75,
            })
            .collect();
        let indices: Vec<_> = (0..items.len()).collect();
        let plans: Vec<_> = indices
            .chunks(127)
            .map(|part| BatchPlan::Prefixes(part.to_vec()))
            .collect();
        let batches: Vec<_> = plans.iter().map(|plan| plan.layout(&items)).collect();
        let order: Vec<_> = (0..plans.len()).rev().collect();
        let w = FSRS6_DEFAULT_PARAMETERS;
        let mut workspace = Workspace::default();
        with_gradients(&plans, items.clone(), &order, |gradient| {
            for growth in [w[10], 0.001, w[10] * 1.5] {
                let mut parameters = w;
                parameters[10] = growth;
                let expected: Vec<_> = batches
                    .iter()
                    .map(|b| b.gradient(&parameters, &mut workspace))
                    .collect();
                for index in (0..batches.len()).rev() {
                    assert_eq!(gradient(index, &parameters), expected[index]);
                }
            }
        });
        // The helper must also stop if the caller exits before submitting work.
        assert_eq!(with_gradients(&plans, items.clone(), &order, |_| 7), 7);
        #[cfg(panic = "unwind")]
        {
            let result = std::panic::catch_unwind(|| {
                with_gradients(&plans, items.clone(), &order, |_| panic!("caller failed"));
            });
            assert!(result.is_err());
            let invalid_plans = vec![
                BatchPlan::Prefixes(vec![0]),
                BatchPlan::Prefixes(vec![0]),
                BatchPlan::Prefixes(vec![usize::MAX]),
            ];
            let result = std::panic::catch_unwind(|| {
                with_gradients(&invalid_plans, items, &[0, 1, 2], |gradient| {
                    gradient(2, &w)
                });
            });
            assert!(result.is_err());
        }
    }
}

//! FSRS-7 windowed training on two threads (targets without the NEON kernel).
//!
//! The batches are the same as `build_windowed_batches` makes: cards ordered by (full length, card
//! id) and packed whole into batches of at most `batch_size` predictions. What differs is how the
//! work is done:
//!
//! - The plan reads only item headers (review count, card id), with an open-addressing card index.
//!   `compute_parameters` does not clamp FSRS-7's delta_t: the layout clamps the reviews it reads.
//! - A second thread lays out each batch's arrays (in the order the first epoch uses them), frees
//!   the prefix items, and then shares every batch's gradient: both threads claim 8-card groups,
//!   and the main thread adds the per-group gradients in group order, so the result does not depend
//!   on which thread computed which group.
//! - The gradient kernel is `analytic_v7::wide_window`.

use super::{
    CosineAnnealingLR, HostAdam, InternalTrainingConfig, ModelVersion, ProgressCollector, Result,
    clip_host_parameters, l2_penalty_fn, l2_penalty_weight, render_progress, schedule_penalty_fn,
    training_v7, zero_frozen_host_grad,
};
use crate::FSRSError;
use crate::analytic_v7::wide_window::{self, Caches, WConsts};
use crate::dataset::WeightedFSRSItem;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use std::sync::atomic::{AtomicU32, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Mutex, OnceLock};

const PARAM_LEN: usize = 34;

/// One batch's arrays, row-major [seq, bsz] (bsz = the card count rounded up to 8).
struct Batch {
    seq: usize,
    bsz: usize,
    cards: usize,
    predictions: usize,
    t_hist: Vec<f32>,
    r_hist: Vec<f32>,
    /// Signed weights (wide_window::signed_weight): the sign bits are the labels.
    weights: Vec<f32>,
}

/// 24 bytes, so the per-item updates of `Plan::new` stay in cache.
struct PlanCard {
    id: i64,
    full_len: u32,
    /// The card's last longest prefix item.
    longest: u32,
    n_preds: u32,
    /// The card's last item (the head of its chain in `Plan::link`).
    last: u32,
}

struct Plan {
    cards: Vec<PlanCard>,
    /// Each item's step (review count - 1) | the index of its card's previous item << 32
    /// (u32::MAX = none): each card's predictions form a chain from `PlanCard::last` back.
    link: Vec<u64>,
    order: Vec<usize>,
    /// Each batch = order[first..end].
    bounds: Vec<(usize, usize)>,
}

/// The cards' `key`s ordered by full length (`at`: each length's first position), each length's
/// keys sorted.
fn by_length<K: Ord + Default + Clone>(
    cards: &[PlanCard],
    at: &[usize],
    key: impl Fn(usize) -> K,
) -> Vec<K> {
    let mut keys = vec![K::default(); cards.len()];
    let mut fill = at.to_vec();
    for (ci, card) in cards.iter().enumerate() {
        keys[fill[card.full_len as usize]] = key(ci);
        fill[card.full_len as usize] += 1;
    }
    for l in at.windows(2) {
        keys[l[0]..l[1]].sort_unstable();
    }
    keys
}

impl Plan {
    fn new(items: &[WeightedFSRSItem], batch_size: usize) -> Self {
        // One pass in input order over the item headers only (the reviews, labels and weights are
        // read at layout time). Each card normally has one prefix of length 2 (its first
        // prediction): their count sizes the card index and the card list.
        let n_cards = items
            .iter()
            .filter(|item| item.item.reviews.len() == 2)
            .count();
        // The card index: open addressing with linear probing on card slots (u32::MAX = empty; a
        // slot's id is its card's), at most half full (doubled if more cards arrive than counted).
        let mut bits = usize::BITS - (2 * n_cards).max(8).leading_zeros();
        let mut slots = vec![u32::MAX; 1 << bits];
        let home = |id: i64, bits: u32| {
            ((id as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) >> (64 - bits)) as usize
        };
        let mut cards: Vec<PlanCard> = Vec::with_capacity(n_cards);
        let mut link: Vec<u64> = Vec::with_capacity(items.len());
        let (mut min_id, mut max_id) = (i64::MAX, i64::MIN);
        let mut max_len = 0;
        for (idx, item) in items.iter().enumerate() {
            let id = item.card_id.expect("FSRS-7 training items have card ids");
            let len = item.item.reviews.len();
            max_len = max_len.max(len);
            let mut h = home(id, bits);
            while slots[h] != u32::MAX && cards[slots[h] as usize].id != id {
                h = (h + 1) & (slots.len() - 1);
            }
            let ci = if slots[h] != u32::MAX {
                slots[h] as usize
            } else {
                slots[h] = cards.len() as u32;
                cards.push(PlanCard {
                    id,
                    full_len: 0,
                    longest: 0,
                    n_preds: 0,
                    last: u32::MAX,
                });
                (min_id, max_id) = (min_id.min(id), max_id.max(id));
                if cards.len() * 2 > slots.len() {
                    bits += 1;
                    slots = vec![u32::MAX; 1 << bits];
                    for (ci, card) in cards.iter().enumerate() {
                        let mut h = home(card.id, bits);
                        while slots[h] != u32::MAX {
                            h = (h + 1) & (slots.len() - 1);
                        }
                        slots[h] = ci as u32;
                    }
                }
                cards.len() - 1
            };
            let card = &mut cards[ci];
            // >= keeps the LAST longest prefix (the stable length sort's order).
            if len as u32 >= card.full_len {
                card.full_len = len as u32;
                card.longest = idx as u32;
            }
            card.n_preds += 1;
            link.push((card.last as u64) << 32 | (len - 1) as u64);
            card.last = idx as u32;
        }
        // Order the cards by (full length, card id), a total order (a card's id is unique): a
        // counting sort by full length, then each length's cards by id. The id sort runs on one u64
        // per card, (id - min id) << bits | card index, when the id span fits (card ids are
        // millisecond timestamps, so it does), else on (id, index) pairs.
        let mut at = vec![0usize; max_len + 2];
        for card in &cards {
            at[card.full_len as usize + 1] += 1;
        }
        for l in 1..at.len() {
            at[l] += at[l - 1];
        }
        let id = |ci: usize| cards[ci].id;
        let span = max_id.abs_diff(min_id); // (no cards: then the order is empty either way)
        let bits = usize::BITS - cards.len().leading_zeros();
        let order: Vec<usize> = if span >> (64 - bits) == 0 {
            let keys = by_length(&cards, &at, |ci| {
                id(ci).abs_diff(min_id) << bits | ci as u64
            });
            keys.iter()
                .map(|k| (k & ((1 << bits) - 1)) as usize)
                .collect()
        } else {
            by_length(&cards, &at, |ci| (id(ci), ci))
                .iter()
                .map(|k| k.1)
                .collect()
        };
        let mut bounds = Vec::new();
        let (mut first, mut current_preds) = (0, 0);
        for (k, &ci) in order.iter().enumerate() {
            let n_preds = cards[ci].n_preds as usize;
            if k > first && current_preds + n_preds > batch_size {
                bounds.push((first, k));
                (first, current_preds) = (k, 0);
            }
            current_preds += n_preds;
        }
        if first < order.len() {
            bounds.push((first, order.len()));
        }
        Self {
            cards,
            link,
            order,
            bounds,
        }
    }

    /// Padded column count of batch `b` (a multiple of 8: the kernel's 8-card groups).
    fn bsz(&self, b: usize) -> usize {
        (self.bounds[b].1 - self.bounds[b].0).div_ceil(8) * 8
    }

    /// Batch `b`'s arrays: each card's reviews from its longest prefix, and at row (prefix length -
    /// 1) each prefix's weight, signed by its label (the rating of that review).
    fn layout(&self, items: &[WeightedFSRSItem], b: usize) -> Batch {
        let (first, end) = self.bounds[b];
        let batch = &self.order[first..end];
        let bsz = self.bsz(b);
        let seq = batch
            .iter()
            .map(|&ci| self.cards[ci].full_len as usize)
            .max()
            .unwrap_or(0);
        let mut t_hist = vec![0.0f32; seq * bsz];
        let mut r_hist = vec![0.0f32; seq * bsz];
        let mut weights = vec![-0.0f32; seq * bsz]; // signed_weight(0.0, 0.0)
        let mut predictions = 0;
        // One independent load from each card's review buffer first (the cache misses overlap).
        let touch = batch.iter().fold(0u32, |a, &ci| {
            let reviews = &items[self.cards[ci].longest as usize].item.reviews;
            a ^ reviews.first().map_or(0, |r| r.rating)
        });
        std::hint::black_box(touch);
        for (c, &ci) in batch.iter().enumerate() {
            let reviews = &items[self.cards[ci].longest as usize].item.reviews;
            for (t, r) in reviews.iter().enumerate() {
                t_hist[t * bsz + c] = r.delta_t.max(0.0);
                r_hist[t * bsz + c] = r.rating as f32;
            }
            // The card's chain, last item first. Two items of one card with the same length share
            // one slot: the later item's weight stays, as in input order.
            let mut idx = self.cards[ci].last;
            while idx != u32::MAX {
                let e = self.link[idx as usize];
                let t = e as u32 as usize;
                let slot = &mut weights[t * bsz + c];
                if slot.to_bits() == (-0.0f32).to_bits() {
                    let label = if reviews[t].rating == 1 { 0.0 } else { 1.0 };
                    *slot = wide_window::signed_weight(items[idx as usize].weight, label);
                }
                idx = (e >> 32) as u32;
            }
            predictions += self.cards[ci].n_preds as usize;
        }
        Batch {
            seq,
            bsz,
            cards: batch.len(),
            predictions,
            t_hist,
            r_hist,
            weights,
        }
    }
}

/// The plan and the prefix items, handed from the main thread to the helper.
type Planned = Mutex<Option<(Plan, Vec<WeightedFSRSItem>)>>;

const JOB_STOP: u64 = u64::MAX;

/// Spins before a waiting thread starts to yield (see `backoff`).
const SPIN_LIMIT: u32 = 256;

/// One step of a wait loop: spin first (the other thread normally answers within microseconds),
/// then yield the CPU, so that a waiting thread does not hold up the other one when both threads
/// share a core (one CPU, or a busy machine).
fn backoff(spins: &mut u32) {
    if *spins < SPIN_LIMIT {
        *spins += 1;
        std::hint::spin_loop();
    } else {
        std::thread::yield_now();
    }
}

/// State shared with the second thread. For each batch the main thread publishes the weights and
/// the batch, then both threads claim 8-card groups from `job` and store each group's gradient at
/// its index in `out`; the main thread then adds the groups in group order. Batches take tens of
/// microseconds, so the threads spin (then yield, see `backoff`) instead of sleeping.
struct GradShared<'a> {
    /// Filled once each, by the layouts.
    host: &'a [OnceLock<Batch>],
    /// (job sequence << 32) | next group to claim. The sequence in the same word keeps a thread
    /// that is still in an old job from claiming a group of the new one. JOB_STOP ends the helper.
    job: AtomicU64,
    batch: AtomicUsize,
    w: [AtomicU32; PARAM_LEN],
    /// Each group's 34 gradient sums, two f32 per word.
    out: Vec<[AtomicU64; PARAM_LEN / 2]>,
    /// ready[g] = the job sequence whose group g is stored in out[g].
    ready: Vec<AtomicU32>,
    /// The prefix items, once the helper has laid out every batch: freed a block at a time by a
    /// thread that would otherwise wait (see free_some).
    trash: Mutex<Vec<WeightedFSRSItem>>,
}

impl GradShared<'_> {
    /// Frees one block of the prefix items if any are left and no other thread is freeing; returns
    /// false if it freed nothing. First one independent load from each item's review buffer (the
    /// loads overlap), then the frees. The threads call it while they wait (the helper for the next
    /// job, the main thread for the helper's last group), so the frees mostly fill idle time.
    fn free_some(&self) -> bool {
        let Ok(mut items) = self.trash.try_lock() else {
            return false;
        };
        let len = items.len();
        let keep = len.saturating_sub(32);
        let touch = items[keep..].iter().fold(0u32, |a, it| {
            a ^ it.item.reviews.first().map_or(0, |r| r.rating)
        });
        std::hint::black_box(touch);
        items.truncate(keep);
        keep < len
    }

    /// Claim and compute groups of job `seq` until none is left (or the job changed).
    fn work(&self, seq: u32, w: &[f32], wc: &WConsts, caches: &mut Caches) {
        let hb = self.host[self.batch.load(Ordering::Relaxed)]
            .get()
            .expect("published batch");
        let n = hb.bsz / 8;
        loop {
            let v = self.job.load(Ordering::Acquire);
            let g = (v & 0xffff_ffff) as usize;
            if (v >> 32) as u32 != seq || g >= n {
                return;
            }
            if self
                .job
                .compare_exchange_weak(v, v + 1, Ordering::AcqRel, Ordering::Acquire)
                .is_err()
            {
                continue;
            }
            let gg = wide_window::group_grad(
                w,
                wc,
                &hb.t_hist,
                &hb.r_hist,
                &hb.weights,
                hb.seq,
                hb.bsz,
                hb.cards,
                g,
                caches,
            );
            for (o, x) in self.out[g].iter().zip(gg.as_chunks::<2>().0) {
                o.store(
                    x[0].to_bits() as u64 | (x[1].to_bits() as u64) << 32,
                    Ordering::Relaxed,
                );
            }
            self.ready[g].store(seq, Ordering::Release);
        }
    }

    /// The second thread: lay out the batches in the order training first uses them, free the
    /// prefix items, then help with every published job. It takes the plan and the items once the
    /// main thread has laid out its lead batches (the main thread holds the lock until then).
    fn helper(&self, planned: &Planned, first_order: &[usize]) {
        let (plan, items) = planned.lock().unwrap().take().expect("the plan");
        for &b in first_order {
            let _ = self.host[b].set(plan.layout(&items, b));
        }
        drop(plan);
        *self.trash.lock().unwrap() = items;
        let mut caches = Caches::default();
        let mut seen = 0u32;
        let mut spins = 0;
        loop {
            let v = self.job.load(Ordering::Acquire);
            if v == JOB_STOP {
                while !self.trash.lock().unwrap().is_empty() {
                    self.free_some();
                }
                return;
            }
            let seq = (v >> 32) as u32;
            if seq == seen {
                if !self.free_some() {
                    backoff(&mut spins);
                }
                continue;
            }
            seen = seq;
            spins = 0;
            let w: [f32; PARAM_LEN] =
                std::array::from_fn(|i| f32::from_bits(self.w[i].load(Ordering::Relaxed)));
            let wc = wide_window::wconsts(&w);
            self.work(seq, &w, &wc, &mut caches);
        }
    }
}

/// `train` for FSRS-7 with the log-loss objective: the same batches, batch order, learning-rate
/// schedule, penalties, Adam steps and clipping, with the batch gradients computed as above.
pub(super) fn train_fsrs7_windowed(
    train_set: Vec<WeightedFSRSItem>,
    initial_parameters: &[f32],
    config: &InternalTrainingConfig,
    progress: Option<ProgressCollector>,
) -> Result<Vec<f32>> {
    let version = ModelVersion::Fsrs7;
    let total_size = train_set.len();
    let plan = Plan::new(&train_set, config.batch_size);
    let batch_count = plan.bounds.len();
    let max_bsz = (0..batch_count).map(|b| plan.bsz(b)).max().unwrap_or(0);
    let host: Vec<OnceLock<Batch>> = (0..batch_count).map(|_| OnceLock::new()).collect();
    let mut parameters = initial_parameters.to_vec();
    let initial = parameters.clone();
    let mut adam = HostAdam::new(parameters.len());
    // Like srs-benchmark's CosineAnnealingLR(T_max = batches * epochs).
    let mut scheduler = CosineAnnealingLR::init(
        (batch_count * config.num_epochs) as f64,
        config.learning_rate,
    );
    let mut rng = StdRng::seed_from_u64(config.seed);
    let mut order = (0..batch_count).collect::<Vec<_>>();
    // The first epoch's batch order (the same shuffle the loop makes, on a copy of the RNG): the
    // helper lays the batches out in this order.
    let mut first_order = order.clone();
    first_order.shuffle(&mut rng.clone());
    let l2_weight = l2_penalty_weight(version) * config.gamma;
    let planned: Planned = Mutex::new(Some((plan, train_set)));
    let shared = GradShared {
        host: &host,
        job: AtomicU64::new(0),
        batch: AtomicUsize::new(0),
        w: std::array::from_fn(|_| AtomicU32::new(0)),
        out: (0..max_bsz / 8)
            .map(|_| std::array::from_fn(|_| AtomicU64::new(0)))
            .collect(),
        ready: (0..max_bsz / 8).map(|_| AtomicU32::new(0)).collect(),
        trash: Mutex::new(Vec::new()),
    };
    let mut progress = progress;
    let mut caches = Caches::default();
    let mut seq = 0u32;
    let mut interrupted = false;
    // With one CPU the second thread would only take time from the first.
    let solo = std::thread::available_parallelism().map_or(true, |n| n.get() < 2);
    std::thread::scope(|scope| {
        if solo {
            let (plan, items) = planned.lock().unwrap().take().expect("the plan");
            for &b in &first_order {
                let _ = host[b].set(plan.layout(&items, b));
            }
        } else {
            // The first batches are laid out here, after the spawn: the helper thread takes
            // ~0.1 ms to start.
            let lead = first_order.len().min(2);
            let guard = planned.lock().unwrap();
            let (shared, rest) = (&shared, &first_order[lead..]);
            let planned = &planned;
            scope.spawn(move || shared.helper(planned, rest));
            let (plan, items) = guard.as_ref().expect("the plan");
            for &b in &first_order[..lead] {
                let _ = host[b].set(plan.layout(items, b));
            }
        }
        'epochs: for epoch in 1..=config.num_epochs {
            order.shuffle(&mut rng);
            let mut processed = 0;
            for &bi in &order {
                // Laid out by the helper, normally well ahead of use.
                let mut spins = 0;
                let hb = loop {
                    match host[bi].get() {
                        Some(hb) => break hb,
                        None => backoff(&mut spins),
                    }
                };
                // Publish the job first: the helper starts on its groups at once.
                for (a, &x) in shared.w.iter().zip(&parameters) {
                    a.store(x.to_bits(), Ordering::Relaxed);
                }
                shared.batch.store(bi, Ordering::Relaxed);
                seq += 1;
                shared.job.store((seq as u64) << 32, Ordering::Release);
                let wc = wide_window::wconsts(&parameters);
                shared.work(seq, &parameters, &wc, &mut caches);
                // The penalty terms, while the helper finishes its last group.
                let (_, l2_grad) = l2_penalty_fn(version)(
                    &parameters,
                    &initial,
                    hb.predictions,
                    total_size,
                    l2_weight,
                    &training_v7::PARAMS_STDDEV,
                );
                let (_, schedule_grad) = schedule_penalty_fn(version)(
                    &parameters,
                    hb.predictions,
                    config.enable_sched_penalties,
                );
                // The groups in group order, each as soon as it is ready.
                let mut total = [0.0f64; PARAM_LEN];
                for (group, ready) in shared.out.iter().zip(&shared.ready).take(hb.bsz / 8) {
                    let mut spins = 0;
                    while ready.load(Ordering::Acquire) != seq {
                        if !shared.free_some() {
                            backoff(&mut spins);
                        }
                    }
                    for (t, o) in total.as_chunks_mut::<2>().0.iter_mut().zip(group) {
                        let v = o.load(Ordering::Relaxed);
                        t[0] += f32::from_bits(v as u32) as f64;
                        t[1] += f32::from_bits((v >> 32) as u32) as f64;
                    }
                }
                let mut grad: Vec<f32> = total.iter().map(|&v| v as f32).collect();
                for (index, value) in grad.iter_mut().enumerate() {
                    *value += l2_grad.get(index).copied().unwrap_or(0.0)
                        + (schedule_grad.get(index).copied().unwrap_or(0.0) / total_size as f64)
                            as f32;
                }
                zero_frozen_host_grad(&mut grad, &config.model);
                adam.step(&mut parameters, &grad, scheduler.step());
                clip_host_parameters(
                    &mut parameters,
                    config.model.num_relearning_steps,
                    !config.model.freeze_short_term_stability,
                );
                processed += hb.predictions;
                if !render_progress(
                    &mut progress,
                    epoch,
                    config.num_epochs,
                    processed.min(total_size),
                    total_size,
                ) {
                    interrupted = true;
                    break 'epochs;
                }
            }
        }
        shared.job.store(JOB_STOP, Ordering::Release);
        // Items left over: freed by both threads.
        while !shared.trash.lock().unwrap().is_empty() {
            shared.free_some();
        }
    });
    if interrupted {
        return Err(FSRSError::Interrupted);
    }
    Ok(parameters)
}

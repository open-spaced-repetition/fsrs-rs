//! Which training items form each FSRS-6 SIMD batch. Plans hold item indices
//! only; `batch.rs` lays out the reviews. Batch boundaries and order match the
//! scalar trainer's `build_fsrs6_batches`:
//!
//! - Without card IDs, items are stably sorted by review count and chunked.
//! - With card IDs, cards are sorted by (longest history, card ID) and packed
//!   until another card would exceed `batch_size` predictions.

use crate::dataset::WeightedFSRSItem;
use std::ops::Range;
use std::sync::Arc;

pub(crate) enum BatchPlan {
    /// Independent items, each scored at its last review.
    Prefixes(Vec<usize>),
    /// The cards `range` of `plan`. Every item of a card is scored along the
    /// card's longest history.
    Cards {
        plan: Arc<CardPlan>,
        range: Range<usize>,
        predictions: usize,
    },
}

impl BatchPlan {
    /// Number of scored items in the batch.
    pub(crate) fn predictions(&self) -> usize {
        match self {
            Self::Prefixes(indices) => indices.len(),
            Self::Cards { predictions, .. } => *predictions,
        }
    }
}

/// Items grouped by card without a vector per card: the items of a card are
/// linked through `next`, in input order.
pub(crate) struct CardPlan {
    /// Sorted by (length, id).
    cards: Vec<PlanCard>,
    next: Vec<usize>,
}

pub(super) struct PlanCard {
    id: i64,
    /// Review count of the card's longest item.
    pub(super) length: usize,
    /// The longest item (the last one on ties); its reviews form the history.
    pub(super) longest: usize,
    /// Number of items of the card.
    pub(super) count: usize,
    first: usize,
    last: usize,
}

const NONE: usize = usize::MAX;

impl CardPlan {
    pub(super) fn cards(&self, range: Range<usize>) -> &[PlanCard] {
        &self.cards[range]
    }

    /// The items of `card`, in input order.
    pub(super) fn items<'a>(&'a self, card: &PlanCard) -> impl Iterator<Item = usize> + 'a {
        std::iter::successors(Some(card.first), |&index| {
            Some(self.next[index]).filter(|&next| next != NONE)
        })
    }

    fn new(items: &[WeightedFSRSItem]) -> Self {
        // A card's first item usually has two reviews.
        let expected_cards = items
            .iter()
            .filter(|item| item.item.reviews.len() == 2)
            .count();
        let mut index = CardIndex::new(expected_cards);
        let mut cards: Vec<PlanCard> = Vec::with_capacity(expected_cards);
        let mut next = vec![NONE; items.len()];
        for (item_index, item) in items.iter().enumerate() {
            let id = item.card_id.expect("checked card id");
            let card_index = match index.find(id, &cards) {
                Ok(card_index) => card_index,
                Err(slot) => {
                    cards.push(PlanCard {
                        id,
                        length: 0,
                        longest: item_index,
                        count: 0,
                        first: item_index,
                        last: NONE,
                    });
                    index.insert(slot, &cards);
                    cards.len() - 1
                }
            };
            let card = &mut cards[card_index];
            let length = item.item.reviews.len();
            if length >= card.length {
                card.longest = item_index;
                card.length = length;
            }
            if card.last != NONE {
                next[card.last] = item_index;
            }
            card.last = item_index;
            card.count += 1;
        }
        // The scalar trainer orders cards by ID, then stably by length.
        cards.sort_unstable_by_key(|card| (card.length, card.id));
        Self { cards, next }
    }
}

/// Open-addressing map from card ID to index in `cards`. The table stores
/// card indices only; probes compare against the cards' IDs.
struct CardIndex {
    slots: Vec<usize>,
    bits: u32,
}

impl CardIndex {
    fn new(expected_cards: usize) -> Self {
        let bits = usize::BITS - (expected_cards * 2).max(8).leading_zeros();
        Self {
            slots: vec![NONE; 1 << bits],
            bits,
        }
    }

    fn home(&self, id: i64) -> usize {
        ((id as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) >> (64 - self.bits)) as usize
    }

    /// `Ok(card index)` if `id` is present, else `Err(free slot)`.
    fn find(&self, id: i64, cards: &[PlanCard]) -> Result<usize, usize> {
        let mut slot = self.home(id);
        loop {
            match self.slots[slot] {
                NONE => return Err(slot),
                card if cards[card].id == id => return Ok(card),
                _ => slot = (slot + 1) & (self.slots.len() - 1),
            }
        }
    }

    /// Records the last card of `cards` in `slot`, keeping the load below 1/2.
    fn insert(&mut self, slot: usize, cards: &[PlanCard]) {
        self.slots[slot] = cards.len() - 1;
        if cards.len() * 2 > self.slots.len() {
            self.bits += 1;
            self.slots = vec![NONE; 1 << self.bits];
            for (card_index, card) in cards.iter().enumerate() {
                let Err(slot) = self.find(card.id, cards) else {
                    unreachable!("card IDs are unique");
                };
                self.slots[slot] = card_index;
            }
        }
    }
}

pub(crate) fn build_batch_plans(items: &[WeightedFSRSItem], batch_size: usize) -> Vec<BatchPlan> {
    if items.iter().all(|item| item.card_id.is_none()) {
        return stable_order_by_length(items.len(), |i| items[i].item.reviews.len())
            .chunks(batch_size)
            .map(|indices| BatchPlan::Prefixes(indices.to_vec()))
            .collect();
    }
    let plan = Arc::new(CardPlan::new(items));
    let mut batches = Vec::new();
    let (mut start, mut predictions) = (0, 0);
    for (index, card) in plan.cards.iter().enumerate() {
        if index > start && predictions + card.count > batch_size {
            batches.push(BatchPlan::Cards {
                plan: Arc::clone(&plan),
                range: start..index,
                predictions,
            });
            (start, predictions) = (index, 0);
        }
        predictions += card.count;
    }
    if start < plan.cards.len() {
        batches.push(BatchPlan::Cards {
            range: start..plan.cards.len(),
            plan,
            predictions,
        });
    }
    batches
}

/// Indices `0..count` stably sorted by `length`: a counting sort for ordinary
/// review counts, with a comparison sort for exceptionally sparse lengths.
fn stable_order_by_length(count: usize, length: impl Fn(usize) -> usize) -> Vec<usize> {
    let max_len = (0..count).map(&length).max().unwrap_or(0);
    if max_len > count.saturating_mul(4).max(1024) {
        let mut order: Vec<_> = (0..count).collect();
        order.sort_by_key(|&i| length(i));
        return order;
    }
    let mut offsets = vec![0usize; max_len + 1];
    for i in 0..count {
        offsets[length(i)] += 1;
    }
    let mut offset = 0;
    for value in &mut offsets {
        (*value, offset) = (offset, offset + *value);
    }
    let mut order = vec![0; count];
    for i in 0..count {
        let position = &mut offsets[length(i)];
        order[*position] = i;
        *position += 1;
    }
    order
}

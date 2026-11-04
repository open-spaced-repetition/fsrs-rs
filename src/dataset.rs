use std::collections::{HashMap, HashSet};

use itertools::Itertools;

use serde::{Deserialize, Serialize};

use crate::error::{FSRSError, Result};

/// Stores a list of reviews for a card, in chronological order. Each FSRSItem corresponds
/// to a single review, but contains the previous reviews of the card as well, after the
/// first one.
/// When used during review, the last item should include the correct delta_t, but
/// the provided rating is ignored as all four ratings are returned by .next_states()
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Default)]
pub struct FSRSItem {
    pub reviews: Vec<FSRSReview>,
}

#[derive(Debug, Clone)]
pub(crate) struct WeightedFSRSItem {
    pub weight: f32,
    pub item: FSRSItem,
    pub card_id: Option<i64>,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq)]
pub struct FSRSReview {
    /// 1-4
    pub rating: u32,
    /// The number of days that passed (can be fractional).
    /// # Warning
    /// `delta_t` for item first(initial) review must be 0
    pub delta_t: f32,
}

/// One card's reviews in chronological order.
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Default)]
pub struct CardReviews {
    /// Must be unique, which evaluation validates too even though it does not read the ids. Orders
    /// the training batches, like [`crate::ComputeParametersInput::card_ids`].
    pub card_id: i64,
    pub reviews: Vec<FSRSReview>,
}

/// An item made of the first `len` reviews of the card at `card_index` of [`CardPrefixes::cards`].
#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
pub struct CardPrefix {
    pub card_index: u32,
    pub len: u32,
}

/// The same items as a `Vec<FSRSItem>` with `card_ids`, as indices into each card's reviews. Memory
/// per card grows linearly with its reviews here, and quadratically as [`FSRSItem`]s.
///
/// `prefixes` must be sorted oldest first, like `train_set`, because an item's position sets its
/// recency weight.
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Default)]
pub struct CardPrefixes {
    pub cards: Vec<CardReviews>,
    pub prefixes: Vec<CardPrefix>,
}

impl CardPrefixes {
    pub(crate) fn validate(&self) -> Result<()> {
        validate_card_prefixes(&self.cards, &self.prefixes)
    }

    pub(crate) fn reviews(&self, prefix: CardPrefix) -> &[FSRSReview] {
        &self.cards[prefix.card_index as usize].reviews[..prefix.len as usize]
    }
}

pub(crate) fn validate_card_prefixes(cards: &[CardReviews], prefixes: &[CardPrefix]) -> Result<()> {
    let in_range = prefixes.iter().all(|prefix| {
        cards
            .get(prefix.card_index as usize)
            .is_some_and(|card| prefix.len >= 1 && prefix.len as usize <= card.reviews.len())
    });
    let ids: HashSet<i64> = cards.iter().map(|card| card.card_id).collect();
    if in_range && ids.len() == cards.len() {
        Ok(())
    } else {
        Err(FSRSError::InvalidInput)
    }
}

/// What FSRS-7's windowed trainers read of a training set, by item.
pub(crate) trait WindowedItems: Send + Sync {
    fn count(&self) -> usize;
    fn card_id(&self, index: usize) -> i64;
    fn weight(&self, index: usize) -> f32;
    fn reviews(&self, index: usize) -> &[FSRSReview];
}

impl WindowedItems for Vec<WeightedFSRSItem> {
    fn count(&self) -> usize {
        self.len()
    }

    fn card_id(&self, index: usize) -> i64 {
        self[index]
            .card_id
            .expect("FSRS-7 training items have card ids")
    }

    fn weight(&self, index: usize) -> f32 {
        self[index].weight
    }

    fn reviews(&self, index: usize) -> &[FSRSReview] {
        &self[index].item.reviews
    }
}

pub(crate) struct WeightedCardPrefixes<'a> {
    pub card_prefixes: &'a CardPrefixes,
    pub prefixes: Vec<(CardPrefix, f32)>,
}

impl WindowedItems for WeightedCardPrefixes<'_> {
    fn count(&self) -> usize {
        self.prefixes.len()
    }

    fn card_id(&self, index: usize) -> i64 {
        self.card_prefixes.cards[self.prefixes[index].0.card_index as usize].card_id
    }

    fn weight(&self, index: usize) -> f32 {
        self.prefixes[index].1
    }

    fn reviews(&self, index: usize) -> &[FSRSReview] {
        self.card_prefixes.reviews(self.prefixes[index].0)
    }
}

pub(crate) fn long_term_review_cnt(reviews: &[FSRSReview]) -> usize {
    reviews
        .iter()
        .filter(|review| review.delta_t >= 1.0)
        .count()
}

/// `long_term` counts the long-term reviews up to and including this one, and `lapses` the failed
/// long-term reviews before it.
pub(crate) fn r_matrix_bin(delta_t: f32, long_term: usize, lapses: usize) -> (u32, u32, u32) {
    let delta_t = delta_t as f64;
    let delta_t_bin = (2.48 * 3.62f64.powf(delta_t.log(3.62).floor()) * 100.0).round() as u32;
    let length = long_term as f64 + 1.0;
    let length_bin = (1.99 * 1.89f64.powf(length.log(1.89).floor())).round() as u32;
    if lapses == 0 {
        return (delta_t_bin, length_bin, 0);
    }
    let lapse_bin = (1.65 * 1.73f64.powf((lapses as f64).log(1.73).floor())).round() as u32;
    (delta_t_bin, length_bin, lapse_bin)
}

const LONG_TERM_DELTA_T_BUCKET_DAYS: f32 = 1.0;

/// Bucket long-term elapsed days for preprocessing/grouping steps.
///
/// Training/inference still uses the original (possibly fractional) `delta_t`.
/// This bucketing is only to keep initialization/outlier groups stable when
/// interday fractional deltas are enabled.
pub(crate) fn bucket_long_term_delta_t(delta_t: f32) -> f32 {
    if !delta_t.is_finite() {
        return 1.0;
    }
    let clamped = delta_t.max(1.0);
    (clamped / LONG_TERM_DELTA_T_BUCKET_DAYS).floor() * LONG_TERM_DELTA_T_BUCKET_DAYS
}

impl FSRSItem {
    // The previous reviews done before the current one.
    pub(crate) fn history(&self) -> impl Iterator<Item = &FSRSReview> {
        self.reviews.iter().take(self.reviews.len() - 1)
    }

    pub(crate) fn current(&self) -> &FSRSReview {
        self.reviews.last().unwrap()
    }

    pub fn long_term_review_cnt(&self) -> usize {
        long_term_review_cnt(&self.reviews)
    }

    pub(crate) fn first_long_term_review(&self) -> FSRSReview {
        *self
            .reviews
            .iter()
            .find(|review| review.delta_t >= 1.0)
            .expect("Invalid FSRS item: at least one review with delta_t >= 1.0 is required")
    }

    pub(crate) fn r_matrix_index(&self) -> (u32, u32, u32) {
        let lapses = self
            .history()
            .filter(|review| review.rating == 1 && review.delta_t >= 1.0)
            .count();
        r_matrix_bin(self.current().delta_t, self.long_term_review_cnt(), lapses)
    }
}

struct OutlierBucketDecisions {
    removed_pairs: [HashSet<u32>; 5],
    kept_initialization_indices: Vec<usize>,
}

fn compute_outlier_bucket_decisions<'a>(
    dataset_for_initialization: impl Iterator<Item = &'a FSRSItem>,
) -> OutlierBucketDecisions {
    let to_key = |delta_t: f32| bucket_long_term_delta_t(delta_t).to_bits();
    let from_key = |key: u32| f32::from_bits(key);
    let mut groups = HashMap::<u32, HashMap<u32, Vec<usize>>>::new();

    // Group by first rating and first long-term review delta_t.
    // (For FSRS-7, current review can be same-day and should not define the group.)
    for (index, item) in dataset_for_initialization.enumerate() {
        let first_review = item.reviews.first().unwrap();
        let first_long_term_review = item.first_long_term_review();
        groups
            .entry(first_review.rating)
            .or_default()
            .entry(to_key(first_long_term_review.delta_t))
            .or_default()
            .push(index);
    }

    let mut removed_pairs: [HashSet<_>; 5] = Default::default();
    let mut kept_initialization_indices = Vec::new();

    for (rating, delta_t_groups) in groups.into_iter().sorted_by_key(|&(k, _)| k) {
        let mut sub_groups = delta_t_groups.into_iter().collect::<Vec<_>>();

        // Order by bucket size descending and delta_t descending, matching the
        // previous item-grouped implementation without storing full item groups.
        sub_groups.sort_by(|(delta_t_a, indices_a), (delta_t_b, indices_b)| {
            indices_b
                .len()
                .cmp(&indices_a.len())
                .then(from_key(*delta_t_b).total_cmp(&from_key(*delta_t_a)))
        });

        let total = sub_groups
            .iter()
            .map(|(_, indices)| indices.len())
            .sum::<usize>();
        let mut has_been_removed = 0;

        for (delta_t, indices) in sub_groups.iter().rev() {
            // remove 5% items (20 at least) of each group
            if has_been_removed + indices.len() >= 20.max(total / 20) {
                // keep the sub_group if it includes at least six items
                // and the delta_t is less than 100 days if rating is not 4
                // or less than 365 days if rating is 4
                if indices.len() < 6 || from_key(*delta_t) > if rating != 4 { 100.0 } else { 365.0 }
                {
                    removed_pairs[rating as usize].insert(*delta_t);
                } else {
                    kept_initialization_indices.extend(indices);
                }
            } else {
                has_been_removed += indices.len();
                removed_pairs[rating as usize].insert(*delta_t);
            }
        }
    }
    OutlierBucketDecisions {
        removed_pairs,
        kept_initialization_indices,
    }
}

pub(crate) fn item_survives_outlier(item: &FSRSItem, removed_pairs: &[HashSet<u32>; 5]) -> bool {
    if item.long_term_review_cnt() == 0 {
        true
    } else {
        let key = bucket_long_term_delta_t(item.first_long_term_review().delta_t).to_bits();
        !removed_pairs[item.reviews[0].rating as usize].contains(&key)
    }
}

pub(crate) fn filter_outlier_indices(
    dataset_for_initialization: &[FSRSItem],
    trainset: &[FSRSItem],
) -> (Vec<usize>, Vec<usize>) {
    filter_outlier_train_indices(dataset_for_initialization, trainset.iter())
}

pub(crate) fn filter_outlier_train_indices<'a>(
    dataset_for_initialization: &[FSRSItem],
    trainset: impl IntoIterator<Item = &'a FSRSItem>,
) -> (Vec<usize>, Vec<usize>) {
    let decisions = compute_outlier_bucket_decisions(dataset_for_initialization.iter());
    let train_indices = trainset
        .into_iter()
        .enumerate()
        .filter_map(|(index, item)| {
            item_survives_outlier(item, &decisions.removed_pairs).then_some(index)
        })
        .collect();
    (decisions.kept_initialization_indices, train_indices)
}

pub fn filter_outlier(
    dataset_for_initialization: Vec<FSRSItem>,
    trainset: Vec<FSRSItem>,
) -> (Vec<FSRSItem>, Vec<FSRSItem>) {
    let (initialization_indices, train_indices) =
        filter_outlier_indices(&dataset_for_initialization, &trainset);
    let mut initialization_items = dataset_for_initialization
        .into_iter()
        .map(Some)
        .collect::<Vec<Option<FSRSItem>>>();
    let filtered_items = initialization_indices
        .into_iter()
        .map(|index| initialization_items[index].take().unwrap())
        .collect();
    let mut train_items = trainset
        .into_iter()
        .map(Some)
        .collect::<Vec<Option<FSRSItem>>>();
    let trainset = train_indices
        .into_iter()
        .map(|index| train_items[index].take().unwrap())
        .collect();
    (filtered_items, trainset)
}

pub(crate) fn prepare_training_data(items: Vec<FSRSItem>) -> (Vec<FSRSItem>, Vec<FSRSItem>) {
    let (mut dataset_for_initialization, mut trainset) = items
        .clone()
        .into_iter()
        .partition(|item| item.long_term_review_cnt() == 1);
    if std::env::var("FSRS_NO_OUTLIER").is_err() {
        (dataset_for_initialization, trainset) = filter_outlier(dataset_for_initialization, items);
    }
    (dataset_for_initialization, trainset)
}

pub(crate) fn sort_items_by_review_length(
    mut weighted_items: Vec<WeightedFSRSItem>,
) -> Vec<WeightedFSRSItem> {
    weighted_items.sort_by_cached_key(|weighted_item| weighted_item.item.reviews.len());
    weighted_items
}

pub(crate) fn constant_weighted_fsrs_items(items: Vec<FSRSItem>) -> Vec<WeightedFSRSItem> {
    items
        .into_iter()
        .map(|item| WeightedFSRSItem {
            weight: 1.0,
            item,
            card_id: None,
        })
        .collect()
}

pub(crate) fn recency_weight(index: usize, count: usize) -> f32 {
    let length = (count as f32 - 1.0).max(1.0);
    0.25 + 0.75 * (index as f32 / length).powi(3)
}

/// The input items should be sorted by the review timestamp.
pub(crate) fn recency_weighted_fsrs_items(items: Vec<FSRSItem>) -> Vec<WeightedFSRSItem> {
    let count = items.len();
    items
        .into_iter()
        .enumerate()
        .map(|(idx, item)| WeightedFSRSItem {
            weight: recency_weight(idx, count),
            item,
            card_id: None,
        })
        .collect()
}

pub(crate) fn recency_weighted_fsrs_items_with_card_ids(
    items: Vec<FSRSItem>,
    card_ids: Vec<i64>,
) -> Vec<WeightedFSRSItem> {
    let count = items.len();
    items
        .into_iter()
        .zip(card_ids)
        .enumerate()
        .map(|(idx, (item, card_id))| WeightedFSRSItem {
            weight: recency_weight(idx, count),
            item,
            card_id: Some(card_id),
        })
        .collect()
}

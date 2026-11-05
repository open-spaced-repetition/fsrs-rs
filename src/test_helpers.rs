use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;

use crate::FSRSItem;
use crate::dataset::{CardPrefix, CardPrefixes, CardReviews, FSRSReview};

pub trait TestHelper<const N: usize, T> {
    fn assert_approx_eq(&self, expected: [T; N]);
}

impl<T, const N: usize> TestHelper<N, T> for [T; N]
where
    T: Copy + Into<f64> + core::fmt::Debug,
{
    #[track_caller]
    fn assert_approx_eq(&self, expected: [T; N]) {
        for (actual, expected) in self.iter().zip(expected) {
            let difference = (Into::<f64>::into(*actual) - Into::<f64>::into(expected)).abs();
            assert!(
                difference <= 1e-4,
                "actual {actual:?}, expected {expected:?}"
            );
        }
    }
}

impl<T, const N: usize> TestHelper<N, T> for Vec<T>
where
    T: Copy + Into<f64> + core::fmt::Debug,
{
    #[track_caller]
    fn assert_approx_eq(&self, expected: [T; N]) {
        self.as_slice().assert_approx_eq(expected);
    }
}

impl<T, const N: usize> TestHelper<N, T> for [T]
where
    T: Copy + Into<f64> + core::fmt::Debug,
{
    #[track_caller]
    fn assert_approx_eq(&self, expected: [T; N]) {
        assert_eq!(self.len(), N);
        for (actual, expected) in self.iter().zip(expected) {
            let difference = (Into::<f64>::into(*actual) - Into::<f64>::into(expected)).abs();
            assert!(
                difference <= 1e-4,
                "actual {actual:?}, expected {expected:?}"
            );
        }
    }
}

/// Cards of many lengths, failures and same-day reviews among them, with some prefixes left
/// out, and the prefixes of all cards interleaved by a fixed shuffle.
pub(crate) fn varied_card_prefixes() -> CardPrefixes {
    let mut rng = StdRng::seed_from_u64(7);
    let mut cards = Vec::new();
    let mut prefixes = Vec::new();
    let mut draw = |bound: u32| rand::Rng::random_range(&mut rng, 0..bound);
    for card_index in 0..120u32 {
        let length = 2 + draw(1 + 2 * card_index) as usize;
        let reviews: Vec<_> = (0..length)
            .map(|i| FSRSReview {
                rating: if draw(5) == 0 { 1 } else { 2 + draw(3) },
                delta_t: match (i, draw(4)) {
                    (0, _) => 0.0,
                    (_, 0) => draw(90) as f32 / 1440.0,
                    _ => (1 + draw(60)) as f32 + draw(24) as f32 / 24.0,
                },
            })
            .collect();
        for len in 2..=length as u32 {
            if draw(10) != 0 {
                prefixes.push(CardPrefix { card_index, len });
            }
        }
        cards.push(CardReviews {
            card_id: 1_000 + 3 * card_index as i64,
            reviews,
        });
    }
    prefixes.shuffle(&mut rng);
    CardPrefixes { cards, prefixes }
}

pub(crate) fn items_of_card_prefixes(card_prefixes: &CardPrefixes) -> (Vec<FSRSItem>, Vec<i64>) {
    card_prefixes
        .prefixes
        .iter()
        .map(|&prefix| {
            let reviews = card_prefixes.reviews(prefix).to_vec();
            let card_id = card_prefixes.cards[prefix.card_index as usize].card_id;
            (FSRSItem { reviews }, card_id)
        })
        .unzip()
}

/// `|actual - expected| / max(|expected|, 1)` in the L2 norm.
pub(crate) fn relative_l2_error<A, E>(actual: &[A], expected: &[E]) -> f64
where
    A: Copy + Into<f64>,
    E: Copy + Into<f64>,
{
    let mut error = 0.0;
    let mut scale = 0.0;
    for (&a, &e) in actual.iter().zip(expected) {
        let (a, e): (f64, f64) = (a.into(), e.into());
        error += (a - e).powi(2);
        scale += e.powi(2);
    }
    error.sqrt() / scale.sqrt().max(1.0)
}

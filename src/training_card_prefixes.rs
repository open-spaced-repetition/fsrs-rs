use super::{
    ComputeParametersInput, ComputeParametersVersion, MIN_ITEMS_TO_TRAIN, ProgressCollector,
    apply_training_config, fsrs7_recency_weight, new_internal_training_config,
    reset_training_progress, train_fsrs7_windowed, validate_training_config,
};
use crate::dataset::{CardPrefixes, WeightedCardPrefixes, long_term_review_cnt};
use crate::error::Result;
use crate::parameter_initialization_fsrs7::smooth_initial_stabilities_fsrs7;
use crate::{DEFAULT_PARAMETERS, FSRSError};

/// Like [`compute_parameters`](super::compute_parameters), but takes the items as
/// [`CardPrefixes`], in memory linear in the reviews. The result is bit-for-bit identical.
///
/// The [`ComputeParametersInput`] provides the options. Its `train_set` must be empty
/// and its `card_ids` must be `None`.
pub fn compute_parameters_from_card_prefixes(
    card_prefixes: &CardPrefixes,
    input: ComputeParametersInput,
) -> Result<Vec<f32>> {
    if input.model_version == ComputeParametersVersion::Fsrs6 {
        return super::card_prefixes_v6::compute_parameters_from_card_prefixes(
            card_prefixes,
            input,
        );
    }
    let ComputeParametersInput {
        train_set,
        card_ids,
        progress,
        enable_short_term,
        enable_sched_penalties,
        model_version,
        num_relearning_steps,
        training_config,
    } = input;
    let finish_progress = || {
        if let Some(progress) = &progress {
            progress.lock().unwrap().mark_finished();
        }
    };
    if !train_set.is_empty()
        || card_ids.is_some()
        || model_version != ComputeParametersVersion::Fsrs7
    {
        finish_progress();
        return Err(FSRSError::InvalidInput);
    }
    card_prefixes
        .validate()
        .inspect_err(|_| finish_progress())?;
    if let Some(config) = &training_config {
        validate_training_config(config).inspect_err(|_| finish_progress())?;
    }
    let only_first_reviews = card_prefixes
        .prefixes
        .iter()
        .all(|&prefix| long_term_review_cnt(card_prefixes.reviews(prefix)) == 1);
    if only_first_reviews || card_prefixes.prefixes.len() < MIN_ITEMS_TO_TRAIN {
        finish_progress();
        return Ok(DEFAULT_PARAMETERS.to_vec());
    }
    let mut config = new_internal_training_config(
        enable_short_term,
        enable_sched_penalties,
        num_relearning_steps,
    );
    apply_training_config(&mut config, training_config, model_version);
    let prefixes: Vec<_> = card_prefixes
        .prefixes
        .iter()
        .enumerate()
        .filter(|(_, prefix)| prefix.len as usize <= config.max_seq_len)
        .map(|(index, &prefix)| {
            (
                prefix,
                fsrs7_recency_weight(index, card_prefixes.prefixes.len()),
            )
        })
        .collect();
    reset_training_progress(&progress, config.num_epochs, prefixes.len());
    let optimized_parameters = train_fsrs7_windowed(
        WeightedCardPrefixes {
            card_prefixes,
            prefixes,
        },
        &DEFAULT_PARAMETERS,
        &config,
        progress.clone().map(|p| ProgressCollector::new(p, 0)),
        #[cfg(all(target_os = "wasi", target_env = "p1"))]
        &mut |_| true,
    )
    .inspect_err(|_| finish_progress())?;
    finish_progress();
    if optimized_parameters
        .iter()
        .any(|parameter: &f32| parameter.is_infinite())
    {
        return Err(FSRSError::InvalidInput);
    }
    let clamped_stability =
        smooth_initial_stabilities_fsrs7(optimized_parameters[0..4].try_into().unwrap())?;
    Ok(clamped_stability
        .into_iter()
        .chain(optimized_parameters[4..].iter().copied())
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::{CardPrefix, CardReviews, FSRSReview};
    use crate::test_helpers::{items_of_card_prefixes, varied_card_prefixes};
    use crate::{
        CombinedProgressState, FSRS6_DEFAULT_PARAMETERS, TrainingConfig, compute_parameters,
    };

    fn fsrs7_input() -> ComputeParametersInput {
        ComputeParametersInput {
            model_version: ComputeParametersVersion::Fsrs7,
            ..Default::default()
        }
    }

    fn fsrs6_input() -> ComputeParametersInput {
        ComputeParametersInput {
            model_version: ComputeParametersVersion::Fsrs6,
            ..Default::default()
        }
    }

    fn assert_fsrs6_matches_expanded(prefixes: &CardPrefixes, input: ComputeParametersInput) {
        let (train_set, card_ids) = items_of_card_prefixes(prefixes);
        let expanded = compute_parameters(ComputeParametersInput {
            train_set,
            card_ids: Some(card_ids),
            ..input.clone()
        });
        let compact = compute_parameters_from_card_prefixes(prefixes, input);
        match (expanded, compact) {
            (Ok(expanded), Ok(compact)) => assert_eq!(
                compact
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>(),
                expanded
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>(),
            ),
            (Err(expanded), Err(compact)) => assert_eq!(compact, expanded),
            (expanded, compact) => panic!("expanded: {expanded:?}, compact: {compact:?}"),
        }
    }

    #[test]
    fn fsrs6_card_prefix_training_matches_expanded_items() {
        let mut data = varied_card_prefixes();
        data.cards[0].card_id = i64::MIN;
        data.cards[1].card_id = i64::MAX;
        data.cards[2].card_id = -1;
        // The elapsed intervals cross the whole-day rounding boundaries used
        // by initialization, outlier filtering and SIMD batch layout.
        for card in &mut data.cards {
            for (index, review) in card.reviews.iter_mut().enumerate().skip(1) {
                if index % 5 == 0 {
                    review.delta_t = [0.49, 0.5, 0.51, 1.49, 1.5, 1.51, 365.5][index % 7];
                }
            }
        }
        data.prefixes.push(data.prefixes[0]);
        for enable_short_term in [true, false] {
            for training_config in [
                None,
                Some(TrainingConfig {
                    max_seq_len: 64,
                    batch_size: 127,
                    ..TrainingConfig::default()
                }),
            ] {
                assert_fsrs6_matches_expanded(
                    &data,
                    ComputeParametersInput {
                        enable_short_term,
                        training_config,
                        ..fsrs6_input()
                    },
                );
            }
        }
    }

    #[test]
    fn fsrs6_card_prefix_training_matches_small_or_empty_sets() {
        let empty = CardPrefixes::default();
        assert_eq!(
            compute_parameters_from_card_prefixes(&empty, fsrs6_input()).unwrap(),
            FSRS6_DEFAULT_PARAMETERS
        );
        let mut data = varied_card_prefixes();
        for count in [1, 7, 8, 63] {
            data.prefixes.truncate(count);
            assert_fsrs6_matches_expanded(&data, fsrs6_input());
            data = varied_card_prefixes();
        }
        // Each target is a card's first long-term review; no gradient training is needed.
        let first_reviews = CardPrefixes {
            cards: (0..100)
                .map(|id| CardReviews {
                    card_id: id,
                    reviews: vec![
                        FSRSReview {
                            rating: (id % 4 + 1) as u32,
                            delta_t: 0.0,
                        },
                        FSRSReview {
                            rating: (id % 3 + 1) as u32,
                            delta_t: 2.0,
                        },
                    ],
                })
                .collect(),
            prefixes: (0..100)
                .map(|card_index| CardPrefix { card_index, len: 2 })
                .collect(),
        };
        assert!(compute_parameters_from_card_prefixes(&first_reviews, fsrs6_input()).is_ok());
        assert_fsrs6_matches_expanded(&first_reviews, fsrs6_input());
    }

    #[test]
    fn fsrs6_card_prefix_training_validates_and_honors_cancellation() {
        let progress = CombinedProgressState::new_shared();
        let invalid = CardPrefixes {
            cards: vec![],
            prefixes: vec![CardPrefix {
                card_index: 0,
                len: 1,
            }],
        };
        assert_eq!(
            compute_parameters_from_card_prefixes(
                &invalid,
                ComputeParametersInput {
                    progress: Some(progress.clone()),
                    ..fsrs6_input()
                },
            ),
            Err(FSRSError::InvalidInput)
        );
        assert!(progress.lock().unwrap().finished());

        let progress = CombinedProgressState::new_shared();
        assert_eq!(
            compute_parameters_from_card_prefixes(
                &varied_card_prefixes(),
                ComputeParametersInput {
                    card_ids: Some(vec![]),
                    progress: Some(progress.clone()),
                    ..fsrs6_input()
                },
            ),
            Err(FSRSError::InvalidInput)
        );
        assert!(progress.lock().unwrap().finished());

        let progress = CombinedProgressState::new_shared();
        progress.lock().unwrap().want_abort = true;
        assert_eq!(
            compute_parameters_from_card_prefixes(
                &varied_card_prefixes(),
                ComputeParametersInput {
                    progress: Some(progress.clone()),
                    ..fsrs6_input()
                },
            ),
            Err(FSRSError::Interrupted)
        );
        assert!(progress.lock().unwrap().finished());
    }

    /// Asserts that items and prefixes train to the same bits, and returns the parameters.
    fn train_both_ways(
        card_prefixes: &CardPrefixes,
        training_config: Option<TrainingConfig>,
    ) -> Vec<f32> {
        let (train_set, card_ids) = items_of_card_prefixes(card_prefixes);
        let input = ComputeParametersInput {
            training_config,
            ..fsrs7_input()
        };
        let from_items = compute_parameters(ComputeParametersInput {
            train_set,
            card_ids: Some(card_ids),
            ..input.clone()
        })
        .unwrap();
        let from_prefixes = compute_parameters_from_card_prefixes(card_prefixes, input).unwrap();
        let bits = |set: &[f32]| set.iter().map(|w| w.to_bits()).collect::<Vec<_>>();
        assert_eq!(bits(&from_prefixes), bits(&from_items));
        from_prefixes
    }

    #[test]
    fn card_prefixes_train_to_the_same_bits_as_their_items() {
        let card_prefixes = varied_card_prefixes();
        assert!(card_prefixes.prefixes.len() > 2_000);
        let max_seq_len = 64;
        assert!(card_prefixes.prefixes.iter().any(|p| p.len > max_seq_len));
        let short = TrainingConfig {
            max_seq_len: max_seq_len as usize,
            ..TrainingConfig::default()
        };
        for training_config in [None, Some(short)] {
            let set = train_both_ways(&card_prefixes, training_config);
            assert_ne!(set, DEFAULT_PARAMETERS.to_vec());
        }
    }

    #[test]
    fn card_prefixes_refuse_invalid_prefixes() {
        let card = |card_id| CardReviews {
            card_id,
            reviews: vec![FSRSReview {
                rating: 3,
                delta_t: 0.0,
            }],
        };
        let prefix = |card_index, len| CardPrefix { card_index, len };
        for card_prefixes in [
            CardPrefixes {
                cards: vec![card(1)],
                prefixes: vec![prefix(1, 1)],
            },
            CardPrefixes {
                cards: vec![card(1)],
                prefixes: vec![prefix(0, 2)],
            },
            CardPrefixes {
                cards: vec![card(1)],
                prefixes: vec![prefix(0, 0)],
            },
            CardPrefixes {
                cards: vec![card(1), card(1)],
                prefixes: vec![prefix(0, 1)],
            },
        ] {
            assert_eq!(
                compute_parameters_from_card_prefixes(&card_prefixes, fsrs7_input()),
                Err(FSRSError::InvalidInput)
            );
        }
    }
}

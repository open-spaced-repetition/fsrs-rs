use super::{
    ComputeParametersInput, MIN_ITEMS_TO_TRAIN, ProgressCollector, apply_training_config,
    first_long_term_review, new_internal_training_config, prepare_fsrs6_training_data,
    reset_training_progress, train_fsrs6_simd, validate_training_config,
};
use crate::dataset::{CardPrefixes, WeightedCardPrefixes, recency_weight};
use crate::error::Result;
use crate::parameter_initialization::{initialize_stability_from_first_reviews, smooth_and_fill};
use crate::{FSRS6_DEFAULT_PARAMETERS, FSRSError};
use std::collections::HashMap;

/// FSRS-6's preprocessing and SIMD trainer over the selected card prefixes.
pub(super) fn compute_parameters_from_card_prefixes(
    card_prefixes: &CardPrefixes,
    input: ComputeParametersInput,
) -> Result<Vec<f32>> {
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
    if !train_set.is_empty() || card_ids.is_some() {
        finish_progress();
        return Err(FSRSError::InvalidInput);
    }
    card_prefixes
        .validate()
        .inspect_err(|_| finish_progress())?;
    if let Some(config) = &training_config {
        validate_training_config(config).inspect_err(|_| finish_progress())?;
    }

    // Use the same rounding, outlier decisions and ordering as expanded items,
    // while retaining only prefix indices rather than copies of each history.
    let (initial_reviews, train_prefixes) =
        prepare_fsrs6_training_data(card_prefixes.prefixes.clone(), |prefix| {
            first_long_term_review(card_prefixes.reviews(*prefix))
        });
    if train_prefixes.len() < 8 {
        finish_progress();
        return Ok(FSRS6_DEFAULT_PARAMETERS.to_vec());
    }
    let initialization_size = initial_reviews.len();
    let (total_recall, total_reviews) =
        train_prefixes
            .iter()
            .fold((0u32, 0u32), |(sum, count), &prefix| {
                let review = card_prefixes.reviews(prefix).last().unwrap();
                (sum + (review.rating > 1) as u32, count + 1)
            });
    let average_recall = total_recall as f32 / total_reviews as f32;
    let (initial_stability, initial_rating_count) =
        initialize_stability_from_first_reviews(initial_reviews.into_iter(), average_recall)
            .inspect_err(|_| finish_progress())?;
    let initialized_parameters: Vec<_> = initial_stability
        .into_iter()
        .chain(FSRS6_DEFAULT_PARAMETERS[4..].iter().copied())
        .collect();
    if train_prefixes.len() == initialization_size || train_prefixes.len() < MIN_ITEMS_TO_TRAIN {
        finish_progress();
        return Ok(initialized_parameters);
    }

    let mut config = new_internal_training_config(
        enable_short_term,
        enable_sched_penalties,
        num_relearning_steps,
    );
    apply_training_config(&mut config, training_config, model_version);
    let count = train_prefixes.len();
    let prefixes = train_prefixes
        .into_iter()
        .enumerate()
        .filter_map(|(index, prefix)| {
            (prefix.len as usize <= config.max_seq_len)
                .then_some((prefix, recency_weight(index, count)))
        })
        .collect();
    let weighted = WeightedCardPrefixes {
        card_prefixes,
        prefixes,
    };
    reset_training_progress(&progress, config.num_epochs, weighted.prefixes.len());
    let optimized = train_fsrs6_simd(
        weighted,
        &initialized_parameters,
        &config,
        progress.clone().map(|p| ProgressCollector::new(p, 0)),
        #[cfg(all(target_os = "wasi", target_env = "p1"))]
        &mut |_| true,
    )
    .inspect_err(|_| finish_progress())?;
    finish_progress();
    if optimized.iter().any(|parameter| !parameter.is_finite()) {
        return Err(FSRSError::InvalidInput);
    }
    let mut initial_stability: HashMap<_, _> = optimized[0..4]
        .iter()
        .enumerate()
        .map(|(index, &value)| (index as u32 + 1, value))
        .collect();
    let smoothed = smooth_and_fill(&mut initial_stability, &initial_rating_count)?;
    Ok(smoothed
        .into_iter()
        .chain(optimized[4..].iter().copied())
        .collect())
}

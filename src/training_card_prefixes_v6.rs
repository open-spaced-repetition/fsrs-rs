use super::{
    ComputeParametersInput, Fsrs6Initialization, ProgressCollector, apply_training_config,
    finalize_fsrs6_parameters, first_long_term_review, initialize_fsrs6_training,
    new_internal_training_config, prepare_fsrs6_training_data, reset_training_progress,
    train_fsrs6_simd, validate_training_config,
};
use crate::FSRSError;
use crate::dataset::{CardPrefixes, WeightedCardPrefixes, recency_weight};
use crate::error::Result;

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
    let (initialized_parameters, initial_rating_count) =
        match initialize_fsrs6_training(initial_reviews, &train_prefixes, None, |&prefix| {
            card_prefixes.reviews(prefix).last().unwrap().rating
        })
        .inspect_err(|_| finish_progress())?
        {
            Fsrs6Initialization::Finished(parameters) => {
                finish_progress();
                return Ok(parameters);
            }
            Fsrs6Initialization::Ready(parameters, rating_count) => (parameters, rating_count),
        };

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
    finalize_fsrs6_parameters(&optimized, &initial_rating_count)
}

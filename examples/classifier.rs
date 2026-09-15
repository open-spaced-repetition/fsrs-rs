use fsrs::{
    ComputeParametersInput, ComputeParametersVersion, FSRS, FSRSItem, FSRSReview,
    RecallClassificationCosts, RecallClassifierTrainingConfig, compute_parameters,
    compute_parameters_for_recall_classifier,
};
use rusqlite::{Connection, OpenFlags, Row};
use std::cmp::Ordering;
use std::collections::BTreeMap;
use std::env;
use std::path::{Path, PathBuf};
use std::time::Instant;

const DAY_MILLIS: i64 = 86_400_000;
const PROBABILITY_EPSILON: f64 = 1e-6;

#[derive(Clone, Debug)]
struct ReviewSample {
    review_id: i64,
    card_id: i64,
    item: FSRSItem,
}

#[derive(Clone, Copy, Debug)]
struct Metrics {
    probability_log_loss: f64,
    score_auc: f64,
    decision_auc: f64,
    accuracy: f64,
    balanced_accuracy: f64,
    true_positives: usize,
    true_negatives: usize,
    false_positives: usize,
    false_negatives: usize,
    realized_cost_per_review: f64,
}

#[derive(Debug)]
struct Args {
    collection: PathBuf,
    deck_name_contains: Option<String>,
    train_fraction: f64,
    validation_fraction: f64,
    false_positive_costs: Vec<f32>,
    false_negative_cost: f32,
    decision_sharpnesses: Vec<f32>,
    balance_classes: bool,
}

fn main() -> fsrs::Result<()> {
    let args = Args::parse()?;
    let mut samples = read_samples(&args.collection, args.deck_name_contains.as_deref())?;
    samples.sort_by_key(|sample| sample.review_id);
    if samples.len() < 128 {
        eprintln!(
            "need at least 128 eligible review samples; found {}",
            samples.len()
        );
        return Err(fsrs::FSRSError::NotEnoughData);
    }

    let training_end = (samples.len() as f64 * args.train_fraction) as usize;
    let validation_end = training_end + (samples.len() as f64 * args.validation_fraction) as usize;
    if training_end < 64 || validation_end >= samples.len() {
        eprintln!("training, validation, and test splits must all be non-empty");
        return Err(fsrs::FSRSError::InvalidInput);
    }
    let training_samples = &samples[..training_end];
    let validation_samples = &samples[training_end..validation_end];
    let test_samples = &samples[validation_end..];
    let training_known = training_samples
        .iter()
        .filter(|sample| outcome(sample) == 1)
        .count();
    let training_forgotten = training_samples.len() - training_known;
    let validation_known = validation_samples
        .iter()
        .filter(|sample| outcome(sample) == 1)
        .count();
    let validation_forgotten = validation_samples.len() - validation_known;
    let known = test_samples
        .iter()
        .filter(|sample| outcome(sample) == 1)
        .count();
    let forgotten = test_samples.len() - known;
    if [
        training_known,
        training_forgotten,
        validation_known,
        validation_forgotten,
        known,
        forgotten,
    ]
    .contains(&0)
    {
        eprintln!("each split must contain both recalled and forgotten reviews");
        return Err(fsrs::FSRSError::InvalidInput);
    }

    println!("collection={}", args.collection.display());
    println!("deck_name_contains={:?}", args.deck_name_contains);
    println!(
        "decision_boundary=0.5 decision_sharpness_candidates={:?}",
        args.decision_sharpnesses
    );
    println!(
        "samples={} training_samples={} training_known={} training_forgotten={} validation_samples={} validation_known={} validation_forgotten={} test_samples={} test_known={} test_forgotten={}",
        samples.len(),
        training_samples.len(),
        training_known,
        training_forgotten,
        validation_samples.len(),
        validation_known,
        validation_forgotten,
        test_samples.len(),
        known,
        forgotten
    );

    let baseline_started = Instant::now();
    let baseline_parameters = train_parameters(training_samples)?;
    let baseline_training_seconds = baseline_started.elapsed().as_secs_f64();
    let baseline_validation_predictions = predict(&baseline_parameters, validation_samples)?;
    let baseline_predictions = predict(&baseline_parameters, test_samples)?;
    println!("baseline_training_seconds={baseline_training_seconds:.3}");
    println!("baseline_parameters={baseline_parameters:?}");
    print_header();

    let training_costs = if args.balance_classes {
        vec![class_balanced_costs(training_samples)]
    } else {
        args.false_positive_costs
            .iter()
            .map(|false_positive_cost| RecallClassificationCosts {
                false_positive: *false_positive_cost,
                false_negative: args.false_negative_cost,
            })
            .collect()
    };
    for training_costs in training_costs {
        let objective = if args.balance_classes {
            "balanced_accuracy"
        } else {
            "explicit_cost"
        };
        let validation_costs = if args.balance_classes {
            class_balanced_costs(validation_samples)
        } else {
            training_costs
        };
        let test_costs = if args.balance_classes {
            class_balanced_costs(test_samples)
        } else {
            training_costs
        };
        let validation_threshold = best_cost_threshold(
            &baseline_validation_predictions,
            validation_samples,
            validation_costs,
        );
        println!(
            "objective={objective} training_costs={training_costs:?} validation_costs={validation_costs:?} test_costs={test_costs:?}"
        );
        let always_known_predictions = vec![1.0; test_samples.len()];
        print_metrics(
            objective,
            "always_known",
            test_costs,
            0.5,
            0.0,
            metrics(&always_known_predictions, test_samples, 0.5, test_costs),
        );
        print_metrics(
            objective,
            "fsrs_fixed_0_5",
            test_costs,
            0.5,
            baseline_training_seconds,
            metrics(&baseline_predictions, test_samples, 0.5, test_costs),
        );
        print_metrics(
            objective,
            "fsrs_validation_threshold",
            test_costs,
            validation_threshold,
            baseline_training_seconds,
            metrics(
                &baseline_predictions,
                test_samples,
                validation_threshold,
                test_costs,
            ),
        );

        let mut selected_classifier = None;
        for decision_sharpness in &args.decision_sharpnesses {
            let classifier_started = Instant::now();
            let classifier_parameters = train_classifier_parameters(
                training_samples,
                training_costs,
                *decision_sharpness,
                &baseline_parameters,
            )?;
            let classifier_training_seconds = classifier_started.elapsed().as_secs_f64();
            let classifier_validation_predictions =
                predict(&classifier_parameters, validation_samples)?;
            let validation_metrics = metrics(
                &classifier_validation_predictions,
                validation_samples,
                0.5,
                validation_costs,
            );
            println!(
                "classifier_candidate decision_sharpness={decision_sharpness} validation_cost={:.9} validation_accuracy={:.9} validation_balanced_accuracy={:.9}",
                validation_metrics.realized_cost_per_review,
                validation_metrics.accuracy,
                validation_metrics.balanced_accuracy,
            );
            let should_select = selected_classifier.as_ref().is_none_or(
                |(best_cost, _, _, _): &(f64, f32, f64, Vec<f32>)| {
                    validation_metrics.realized_cost_per_review < *best_cost
                },
            );
            if should_select {
                selected_classifier = Some((
                    validation_metrics.realized_cost_per_review,
                    *decision_sharpness,
                    classifier_training_seconds,
                    classifier_parameters,
                ));
            }
        }
        let (_, selected_sharpness, classifier_training_seconds, classifier_parameters) =
            selected_classifier.expect("at least one decision sharpness");
        let classifier_predictions = predict(&classifier_parameters, test_samples)?;
        println!(
            "classifier_parameters false_positive_cost={} false_negative_cost={} selected_decision_sharpness={} values={classifier_parameters:?}",
            training_costs.false_positive, training_costs.false_negative, selected_sharpness
        );
        print_metrics(
            objective,
            "fsrs_direct_classifier",
            test_costs,
            0.5,
            classifier_training_seconds,
            metrics(&classifier_predictions, test_samples, 0.5, test_costs),
        );
    }

    Ok(())
}

impl Args {
    fn parse() -> fsrs::Result<Self> {
        let mut args = env::args().skip(1);
        let mut collection = None;
        let mut deck_name_contains = None;
        let mut train_fraction = 0.6_f64;
        let mut validation_fraction = 0.2_f64;
        let mut false_positive_costs: Vec<f32> = Vec::new();
        let mut false_negative_cost = 1.0_f32;
        let mut decision_sharpnesses: Vec<f32> = Vec::new();
        let mut balance_classes = false;

        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--collection" => {
                    collection = Some(PathBuf::from(required_arg(&mut args)?));
                }
                "--deck-name-contains" => {
                    deck_name_contains = Some(required_arg(&mut args)?);
                }
                "--train-fraction" => {
                    train_fraction = parse_number(required_arg(&mut args)?)?;
                }
                "--validation-fraction" => {
                    validation_fraction = parse_number(required_arg(&mut args)?)?;
                }
                "--false-positive-cost" => {
                    false_positive_costs.push(parse_number(required_arg(&mut args)?)?);
                }
                "--false-negative-cost" => {
                    false_negative_cost = parse_number(required_arg(&mut args)?)?;
                }
                "--decision-sharpness" => {
                    decision_sharpnesses.push(parse_number(required_arg(&mut args)?)?);
                }
                "--balance-classes" => {
                    balance_classes = true;
                }
                _ => return Err(fsrs::FSRSError::InvalidInput),
            }
        }

        let collection = collection.ok_or(fsrs::FSRSError::InvalidInput)?;
        if !(0.1..=0.8).contains(&train_fraction)
            || !(0.1..=0.8).contains(&validation_fraction)
            || train_fraction + validation_fraction > 0.9
            || !false_negative_cost.is_finite()
            || false_negative_cost <= 0.0
        {
            return Err(fsrs::FSRSError::InvalidInput);
        }
        if false_positive_costs.is_empty() {
            false_positive_costs = vec![1.0];
        }
        if decision_sharpnesses.is_empty() {
            decision_sharpnesses = vec![5.0, 10.0, 20.0, 50.0];
        }
        if false_positive_costs
            .iter()
            .any(|cost| !cost.is_finite() || *cost <= 0.0)
            || decision_sharpnesses
                .iter()
                .any(|sharpness| !sharpness.is_finite() || *sharpness <= 0.0)
        {
            return Err(fsrs::FSRSError::InvalidInput);
        }

        Ok(Self {
            collection,
            deck_name_contains,
            train_fraction,
            validation_fraction,
            false_positive_costs,
            false_negative_cost,
            decision_sharpnesses,
            balance_classes,
        })
    }
}

fn required_arg(args: &mut impl Iterator<Item = String>) -> fsrs::Result<String> {
    args.next().ok_or(fsrs::FSRSError::InvalidInput)
}

fn parse_number<T: std::str::FromStr>(value: String) -> fsrs::Result<T> {
    value.parse().map_err(|_| fsrs::FSRSError::InvalidInput)
}

fn train_parameters(samples: &[ReviewSample]) -> fsrs::Result<Vec<f32>> {
    compute_parameters(training_input(samples))
}

fn train_classifier_parameters(
    samples: &[ReviewSample],
    costs: RecallClassificationCosts,
    decision_sharpness: f32,
    initial_parameters: &[f32],
) -> fsrs::Result<Vec<f32>> {
    compute_parameters_for_recall_classifier(
        training_input(samples),
        RecallClassifierTrainingConfig {
            costs,
            decision_boundary: 0.5,
            decision_sharpness,
            initial_parameters: Some(initial_parameters.to_vec()),
        },
    )
}

fn training_input(samples: &[ReviewSample]) -> ComputeParametersInput {
    ComputeParametersInput {
        train_set: samples.iter().map(|sample| sample.item.clone()).collect(),
        card_ids: Some(samples.iter().map(|sample| sample.card_id).collect()),
        progress: None,
        enable_short_term: true,
        enable_sched_penalties: true,
        model_version: ComputeParametersVersion::Fsrs7,
        num_relearning_steps: None,
        training_config: None,
    }
}

fn predict(parameters: &[f32], samples: &[ReviewSample]) -> fsrs::Result<Vec<f32>> {
    let fsrs = FSRS::new(parameters)?;
    let mut histories = Vec::with_capacity(samples.len());
    let mut current_reviews = Vec::with_capacity(samples.len());
    for sample in samples {
        let mut reviews = sample.item.reviews.clone();
        let current = reviews.pop().ok_or(fsrs::FSRSError::InvalidInput)?;
        if reviews.is_empty() {
            return Err(fsrs::FSRSError::InvalidInput);
        }
        histories.push(FSRSItem { reviews });
        current_reviews.push(current);
    }
    let states = fsrs.memory_state_batch(histories, vec![None; samples.len()])?;
    Ok(states
        .into_iter()
        .zip(current_reviews)
        .map(|(state, review)| fsrs.current_retrievability(state, review.delta_t))
        .collect())
}

fn metrics(
    predictions: &[f32],
    samples: &[ReviewSample],
    threshold: f32,
    costs: RecallClassificationCosts,
) -> Metrics {
    let mut probability_log_loss = 0.0;
    let mut true_positives = 0;
    let mut true_negatives = 0;
    let mut false_positives = 0;
    let mut false_negatives = 0;
    let mut score_pairs = Vec::with_capacity(samples.len());
    let mut decision_pairs = Vec::with_capacity(samples.len());

    for (&prediction, sample) in predictions.iter().zip(samples) {
        let outcome = outcome(sample);
        let probability =
            f64::from(prediction).clamp(PROBABILITY_EPSILON, 1.0 - PROBABILITY_EPSILON);
        let positive_loss = -probability.ln();
        let negative_loss = -(1.0 - probability).ln();
        probability_log_loss += if outcome == 1 {
            positive_loss
        } else {
            negative_loss
        };
        let decision = u8::from(prediction >= threshold);
        match (decision, outcome) {
            (1, 1) => true_positives += 1,
            (0, 0) => true_negatives += 1,
            (1, 0) => false_positives += 1,
            (0, 1) => false_negatives += 1,
            _ => unreachable!(),
        }
        score_pairs.push((prediction, outcome));
        decision_pairs.push((f32::from(decision), outcome));
    }

    let count = samples.len() as f64;
    let sensitivity = true_positives as f64 / (true_positives + false_negatives) as f64;
    let specificity = true_negatives as f64 / (true_negatives + false_positives) as f64;
    Metrics {
        probability_log_loss: probability_log_loss / count,
        score_auc: roc_auc(&score_pairs),
        decision_auc: roc_auc(&decision_pairs),
        accuracy: (true_positives + true_negatives) as f64 / count,
        balanced_accuracy: (sensitivity + specificity) / 2.0,
        true_positives,
        true_negatives,
        false_positives,
        false_negatives,
        realized_cost_per_review: (false_positives as f64 * f64::from(costs.false_positive)
            + false_negatives as f64 * f64::from(costs.false_negative))
            / count,
    }
}

fn outcome(sample: &ReviewSample) -> u8 {
    u8::from(sample.item.reviews.last().unwrap().rating > 1)
}

fn class_balanced_costs(samples: &[ReviewSample]) -> RecallClassificationCosts {
    let known = samples.iter().filter(|sample| outcome(sample) == 1).count();
    let forgotten = samples.len() - known;
    RecallClassificationCosts {
        false_positive: samples.len() as f32 / (2 * forgotten) as f32,
        false_negative: samples.len() as f32 / (2 * known) as f32,
    }
}

fn best_cost_threshold(
    predictions: &[f32],
    samples: &[ReviewSample],
    costs: RecallClassificationCosts,
) -> f32 {
    let mut scored = predictions
        .iter()
        .copied()
        .zip(samples.iter().map(outcome))
        .collect::<Vec<_>>();
    scored.sort_by(|left, right| right.0.partial_cmp(&left.0).unwrap_or(Ordering::Equal));

    let false_negative_cost = f64::from(costs.false_negative);
    let false_positive_cost = f64::from(costs.false_positive);
    let mut current_cost =
        scored.iter().filter(|(_, outcome)| *outcome == 1).count() as f64 * false_negative_cost;
    let mut best_cost = current_cost;
    let mut best_threshold = f32::INFINITY;
    let mut index = 0;
    while index < scored.len() {
        let threshold = scored[index].0;
        let mut end = index + 1;
        while end < scored.len() && scored[end].0 == threshold {
            end += 1;
        }
        for (_, outcome) in &scored[index..end] {
            if *outcome == 1 {
                current_cost -= false_negative_cost;
            } else {
                current_cost += false_positive_cost;
            }
        }
        let is_better = current_cost < best_cost - f64::EPSILON;
        let is_equal_and_closer_to_half = (current_cost - best_cost).abs() <= f64::EPSILON
            && (threshold - 0.5).abs() < (best_threshold - 0.5).abs();
        if is_better || is_equal_and_closer_to_half {
            best_cost = current_cost;
            best_threshold = threshold;
        }
        index = end;
    }
    best_threshold
}

fn roc_auc(pairs: &[(f32, u8)]) -> f64 {
    let positives = pairs
        .iter()
        .map(|(_, outcome)| usize::from(*outcome))
        .sum::<usize>();
    let negatives = pairs.len() - positives;
    if positives == 0 || negatives == 0 {
        return f64::NAN;
    }

    let mut sorted = pairs.to_vec();
    sorted.sort_by(|left, right| left.0.partial_cmp(&right.0).unwrap_or(Ordering::Equal));
    let mut positive_rank_sum = 0.0;
    let mut index = 0;
    while index < sorted.len() {
        let mut end = index + 1;
        while end < sorted.len() && sorted[end].0 == sorted[index].0 {
            end += 1;
        }
        let average_rank = ((index + 1 + end) as f64) / 2.0;
        let tied_positives = sorted[index..end]
            .iter()
            .map(|(_, outcome)| usize::from(*outcome))
            .sum::<usize>();
        positive_rank_sum += average_rank * tied_positives as f64;
        index = end;
    }
    (positive_rank_sum - (positives * (positives + 1)) as f64 / 2.0)
        / (positives * negatives) as f64
}

fn print_header() {
    println!(
        "objective,model,false_positive_cost,false_negative_cost,threshold,training_seconds,probability_log_loss,score_auc,decision_auc,accuracy,balanced_accuracy,true_positives,true_negatives,false_positives,false_negatives,realized_cost_per_review"
    );
}

fn print_metrics(
    objective: &str,
    model: &str,
    costs: RecallClassificationCosts,
    threshold: f32,
    training_seconds: f64,
    metrics: Metrics,
) {
    println!(
        "{objective},{model},{},{},{threshold},{training_seconds:.3},{:.9},{:.9},{:.9},{:.9},{:.9},{},{},{},{},{:.9}",
        costs.false_positive,
        costs.false_negative,
        metrics.probability_log_loss,
        metrics.score_auc,
        metrics.decision_auc,
        metrics.accuracy,
        metrics.balanced_accuracy,
        metrics.true_positives,
        metrics.true_negatives,
        metrics.false_positives,
        metrics.false_negatives,
        metrics.realized_cost_per_review,
    );
}

fn read_samples(path: &Path, deck_name_contains: Option<&str>) -> fsrs::Result<Vec<ReviewSample>> {
    let connection = Connection::open_with_flags(path, OpenFlags::SQLITE_OPEN_READ_ONLY)
        .map_err(|_| fsrs::FSRSError::InvalidInput)?;
    let deck_ids = selected_deck_ids(&connection, deck_name_contains)?;
    let (sql, deck_values) = if deck_ids.is_empty() {
        (
            "SELECT r.id, r.cid, r.ease, r.type, r.factor
             FROM revlog r
             WHERE r.ease BETWEEN 1 AND 4
               AND r.type IN (0, 1, 2, 3)
               AND NOT (r.type = 3 AND r.factor = 0)
             ORDER BY r.cid, r.id"
                .to_string(),
            Vec::new(),
        )
    } else {
        let placeholders = vec!["?"; deck_ids.len()].join(",");
        (
            format!(
                "SELECT r.id, r.cid, r.ease, r.type, r.factor
                 FROM revlog r
                 JOIN cards c ON c.id = r.cid
                 WHERE r.ease BETWEEN 1 AND 4
                   AND r.type IN (0, 1, 2, 3)
                   AND NOT (r.type = 3 AND r.factor = 0)
                   AND CASE WHEN c.odid != 0 THEN c.odid ELSE c.did END IN ({placeholders})
                 ORDER BY r.cid, r.id"
            ),
            deck_ids,
        )
    };
    let mut statement = connection
        .prepare_cached(&sql)
        .map_err(|_| fsrs::FSRSError::InvalidInput)?;
    let revlogs = statement
        .query_and_then(rusqlite::params_from_iter(deck_values), row_to_review)
        .map_err(|_| fsrs::FSRSError::InvalidInput)?
        .collect::<Result<Vec<_>, _>>()
        .map_err(|_| fsrs::FSRSError::InvalidInput)?;
    Ok(reviews_to_samples(revlogs))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ReviewKind {
    Learning,
    Other,
}

#[derive(Clone, Copy, Debug)]
struct ReviewRow {
    review_id: i64,
    card_id: i64,
    rating: u32,
    kind: ReviewKind,
}

fn row_to_review(row: &Row<'_>) -> rusqlite::Result<ReviewRow> {
    Ok(ReviewRow {
        review_id: row.get(0)?,
        card_id: row.get(1)?,
        rating: row.get(2)?,
        kind: if row.get::<_, i64>(3)? == 0 {
            ReviewKind::Learning
        } else {
            ReviewKind::Other
        },
    })
}

fn reviews_to_samples(revlogs: Vec<ReviewRow>) -> Vec<ReviewSample> {
    let mut grouped = BTreeMap::<i64, Vec<ReviewRow>>::new();
    for entry in revlogs {
        grouped.entry(entry.card_id).or_default().push(entry);
    }

    let mut samples = Vec::new();
    for entries in grouped.into_values() {
        let entries = after_last_learning_start(entries);
        if entries.len() < 2 {
            continue;
        }
        let mut reviews = Vec::with_capacity(entries.len());
        for (index, entry) in entries.iter().enumerate() {
            let delta_t = if index == 0 {
                0.0
            } else {
                entry
                    .review_id
                    .saturating_sub(entries[index - 1].review_id)
                    .max(1) as f32
                    / DAY_MILLIS as f32
            };
            reviews.push(FSRSReview {
                rating: entry.rating,
                delta_t,
            });
            if index > 0 {
                samples.push(ReviewSample {
                    review_id: entry.review_id,
                    card_id: entry.card_id,
                    item: FSRSItem {
                        reviews: reviews.clone(),
                    },
                });
            }
        }
    }
    samples
}

fn after_last_learning_start(entries: Vec<ReviewRow>) -> Vec<ReviewRow> {
    let mut first_of_last_learning_sequence = None;
    for (index, entry) in entries.iter().enumerate().rev() {
        if entry.kind == ReviewKind::Learning {
            first_of_last_learning_sequence = Some(index);
        } else if first_of_last_learning_sequence.is_some() {
            break;
        }
    }
    first_of_last_learning_sequence
        .map(|index| entries[index..].to_vec())
        .unwrap_or_default()
}

fn selected_deck_ids(
    connection: &Connection,
    deck_name_contains: Option<&str>,
) -> fsrs::Result<Vec<i64>> {
    let Some(deck_name_contains) = deck_name_contains else {
        return Ok(Vec::new());
    };
    let mut statement = connection
        .prepare_cached(
            "SELECT id FROM decks
             WHERE lower(name COLLATE binary) LIKE '%' || lower(?1) || '%'",
        )
        .map_err(|_| fsrs::FSRSError::InvalidInput)?;
    let deck_ids = statement
        .query_map([deck_name_contains], |row| row.get(0))
        .map_err(|_| fsrs::FSRSError::InvalidInput)?
        .collect::<Result<Vec<_>, _>>()
        .map_err(|_| fsrs::FSRSError::InvalidInput)?;
    if deck_ids.is_empty() {
        Err(fsrs::FSRSError::InvalidInput)
    } else {
        Ok(deck_ids)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn auc_accounts_for_tied_classifier_decisions() {
        assert!((roc_auc(&[(1.0, 1), (1.0, 0), (0.0, 1), (0.0, 0)]) - 0.5).abs() < 1e-12);
        assert!((roc_auc(&[(0.9, 1), (0.8, 1), (0.2, 0), (0.1, 0)]) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn threshold_is_selected_from_validation_cost_only() {
        let samples = [3, 3, 1]
            .into_iter()
            .enumerate()
            .map(|(index, rating)| ReviewSample {
                review_id: index as i64,
                card_id: index as i64,
                item: FSRSItem {
                    reviews: vec![FSRSReview {
                        rating,
                        delta_t: 1.0,
                    }],
                },
            })
            .collect::<Vec<_>>();

        let threshold = best_cost_threshold(
            &[0.9, 0.8, 0.7],
            &samples,
            RecallClassificationCosts::default(),
        );

        assert_eq!(threshold, 0.8);
    }

    #[test]
    fn balanced_costs_give_each_class_equal_total_weight() {
        let samples = [3, 3, 1]
            .into_iter()
            .enumerate()
            .map(|(index, rating)| ReviewSample {
                review_id: index as i64,
                card_id: index as i64,
                item: FSRSItem {
                    reviews: vec![FSRSReview {
                        rating,
                        delta_t: 1.0,
                    }],
                },
            })
            .collect::<Vec<_>>();

        let costs = class_balanced_costs(&samples);

        assert!((2.0 * costs.false_negative - 1.5).abs() < 1e-6);
        assert!((costs.false_positive - 1.5).abs() < 1e-6);
    }

    #[test]
    fn review_conversion_uses_fractional_elapsed_days() {
        let samples = reviews_to_samples(vec![
            ReviewRow {
                review_id: 1_000,
                card_id: 1,
                rating: 3,
                kind: ReviewKind::Learning,
            },
            ReviewRow {
                review_id: 1_000 + DAY_MILLIS / 2,
                card_id: 1,
                rating: 1,
                kind: ReviewKind::Other,
            },
        ]);

        assert_eq!(samples.len(), 1);
        assert!((samples[0].item.reviews[1].delta_t - 0.5).abs() < 1e-6);
    }
}

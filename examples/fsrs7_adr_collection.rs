use fsrs::{
    CombinedProgressState, ComputeParametersInput, ComputeParametersVersion, CostAdrPolicy,
    CostAdrTrainingConfig, FSRS, FSRSItem, FSRSReview, IntervalBucketConfig, RevlogEntry,
    RevlogReviewKind, compute_parameters, extract_simulator_config,
    simulate_cost_adr_interval_bucket_stats,
};
use rusqlite::{Connection, Row};
use std::collections::{BTreeMap, HashSet};
use std::env;
use std::fs::{File, create_dir_all};
use std::io::{BufWriter, Write, stdout};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Instant;

const DEFAULT_COLLECTION: &str =
    "/Users/jschoreels/Library/Application Support/Anki2/Main Profile/collection.anki2";
const DEFAULT_OUTPUT_DIR: &str = "docs/fsrs7-adr-plots";
const DEFAULT_SEED: u64 = 42;
const ASIA_SHANGHAI_4H_DAY_CUTOFF_SECONDS: i64 = 20 * 60 * 60;

fn main() -> fsrs::Result<()> {
    let args = Args::parse()?;
    create_dir_all(&args.output_dir).map_err(|_| fsrs::FSRSError::InvalidInput)?;

    let collection_started = Instant::now();
    let (mut revlogs, day_cutoff, selected_decks) =
        read_collection(&args.collection_path, args.deck_name_contains.as_deref())?;
    if let Some(card_ids_file) = &args.card_ids_file {
        let card_ids = read_card_ids(card_ids_file)?;
        revlogs.retain(|entry| card_ids.contains(&entry.cid));
    }
    if let Some(first_grade) = args.first_grade {
        revlogs = filter_revlogs_by_first_grade(&revlogs, first_grade);
    }
    let (fsrs_items, fsrs_card_ids) = revlogs_to_fsrs_items_with_card_ids(&revlogs, day_cutoff);
    let collection_seconds = collection_started.elapsed().as_secs_f32();

    let unique_cards = revlogs
        .iter()
        .map(|entry| entry.cid)
        .collect::<HashSet<_>>();
    let active_days = active_new_card_days(&revlogs, day_cutoff).max(1);
    let learn_limit = ((unique_cards.len() as f32 / active_days as f32).round() as usize).max(1);

    let mut config = extract_simulator_config(revlogs.clone(), day_cutoff, true);
    if !args.use_extracted_simulator_config {
        config.deck_size = unique_cards.len();
        config.learn_span = active_days;
        config.learn_limit = learn_limit;
        config.review_limit = 9999;
        config.max_cost_perday = 720.0 * 60.0;
    }

    println!("collection_path={}", args.collection_path.display());
    if let Some(deck_name_contains) = &args.deck_name_contains {
        println!("deck_name_contains={deck_name_contains}");
        for deck in &selected_decks {
            println!(
                "selected_deck_id={} selected_deck_name={}",
                deck.id, deck.name
            );
        }
    }
    if let Some(first_grade) = args.first_grade {
        println!("first_grade={first_grade}");
    }
    if let Some(card_ids_file) = &args.card_ids_file {
        println!("card_ids_file={}", card_ids_file.display());
    }
    println!(
        "collection revlogs={} reviewed_cards={} fsrs_items={} active_new_card_days={} load_seconds={:.3}",
        revlogs.len(),
        unique_cards.len(),
        fsrs_items.len(),
        active_days,
        collection_seconds
    );
    println!(
        "simulator deck_size={} learn_span={} learn_limit={} review_limit={} max_cost_seconds={}",
        config.deck_size,
        config.learn_span,
        config.learn_limit,
        config.review_limit,
        config.max_cost_perday
    );
    println!(
        "simulator_mode={}",
        if args.use_extracted_simulator_config {
            "extracted"
        } else {
            "example_overrides"
        }
    );
    println!("model_version={}", args.model_version.label());
    stdout()
        .flush()
        .map_err(|_| fsrs::FSRSError::InvalidInput)?;

    let parameters = if let Some(params_file) = &args.params_file {
        println!(
            "{}_parameters_file={}",
            args.model_version.label(),
            params_file.display()
        );
        read_params(params_file)?
    } else {
        let use_card_ids = matches!(args.model_version, ModelChoice::Fsrs7);
        println!(
            "{}_parameter_training_card_ids={}",
            args.model_version.label(),
            if use_card_ids { "enabled" } else { "disabled" }
        );
        let fsrs_started = Instant::now();
        let parameters = compute_parameters(ComputeParametersInput {
            train_set: fsrs_items.clone(),
            card_ids: use_card_ids.then_some(fsrs_card_ids),
            progress: Some(CombinedProgressState::new_shared()),
            enable_short_term: true,
            enable_sched_penalties: true,
            model_version: args.model_version.compute_parameters_version(),
            num_relearning_steps: None,
        })?;
        let fsrs_seconds = fsrs_started.elapsed().as_secs_f32();
        println!(
            "{}_parameter_training_seconds={fsrs_seconds:.3}",
            args.model_version.label()
        );
        parameters
    };
    println!("{}_parameters={parameters:?}", args.model_version.label());
    let evaluation_started = Instant::now();
    let evaluation = FSRS::new(&parameters)?.evaluate(fsrs_items, |_| true)?;
    println!(
        "{}_evaluation_log_loss={:.8} {}_evaluation_rmse_bins={:.8} {}_evaluation_seconds={:.3}",
        args.model_version.label(),
        evaluation.log_loss,
        args.model_version.label(),
        evaluation.rmse_bins,
        args.model_version.label(),
        evaluation_started.elapsed().as_secs_f32()
    );

    let adr_training_config = CostAdrTrainingConfig {
        population_size: args.population_size,
        generations: args.generations,
        early_stop_patience_generations: args.early_stop_patience_generations,
        early_stop_min_generations: args.early_stop_min_generations,
        early_stop_min_relative_gain: args.early_stop_min_relative_gain,
        seed: Some(args.seed),
        simulation_seed: Some(args.seed),
        average_desired_retention_min_weight_target: args.endpoint_avg_dr_min_weight,
        average_desired_retention_max_weight_target: args.endpoint_avg_dr_max_weight,
        average_desired_retention_endpoint_penalty: args.endpoint_avg_dr_penalty,
        progress: Some(Arc::new(Mutex::new(Default::default()))),
        ..Default::default()
    };
    println!(
        "endpoint_avg_dr_min_weight={:?} endpoint_avg_dr_max_weight={:?} endpoint_avg_dr_penalty={} early_stop_patience_generations={} early_stop_min_generations={} early_stop_min_relative_gain={}",
        adr_training_config.average_desired_retention_min_weight_target,
        adr_training_config.average_desired_retention_max_weight_target,
        adr_training_config.average_desired_retention_endpoint_penalty,
        adr_training_config.early_stop_patience_generations,
        adr_training_config.early_stop_min_generations,
        adr_training_config.early_stop_min_relative_gain
    );

    let adr_started = Instant::now();
    let result = CostAdrPolicy::train_single_user(&config, &parameters, &adr_training_config)?;
    let adr_wall_seconds = adr_started.elapsed().as_secs_f32();
    println!(
        "cost_adr_training_seconds={:.3} result_training_seconds={:.3}",
        adr_wall_seconds, result.training_seconds
    );
    println!(
        "cost_adr_generations_ran={} cost_adr_max_generations={}",
        result.history.len(),
        adr_training_config.generations
    );
    println!(
        "baseline_hypervolume={:.6} best_hypervolume={:.6} best_hypervolume_delta={:.6}",
        result.baseline_hypervolume, result.best_hypervolume, result.best_hypervolume_delta
    );
    print_generation_hypervolume_curve(&result);
    println!(
        "best_auc_relative_time_saved_percent={:?}",
        result
            .best_auc_metrics
            .relative_same_target_time_saved_auc_percent
    );
    for point in &result.best_cost_weight_metrics {
        println!(
            "cost_weight={:.1} avg_dr={:?} {}_eq_dr={:?} same_target_time_saved_percent={:?} memorized_avg={:.3} time_avg_min={:.3} mem_per_min={:.3} reviews={} lapses={}",
            point.goal_cost_weight,
            point.average_desired_retention,
            args.model_version.label(),
            point.fixed_fsrs_equivalent_desired_retention,
            point.same_target_time_saved_percent,
            point.metrics.memorized_average,
            point.metrics.time_average,
            point.metrics.memorized_per_minute,
            point.metrics.total_reviews,
            point.metrics.total_lapses
        );
    }
    if let Some(target_average_desired_retention) = args.target_average_desired_retention {
        println!(
            "target_avg_adr_dr={:.6} inferred_cost_weight={:?}",
            target_average_desired_retention,
            result.cost_weight_for_average_desired_retention(target_average_desired_retention)
        );
    }
    if args.interval_bucket_stats {
        print_interval_bucket_stats(&config, &parameters, &result.policy, args.seed, &result)?;
    }

    write_policy(&args.output_dir, &result.policy)?;
    write_retention_grid(&args.output_dir, &result.policy)?;
    println!("output_dir={}", args.output_dir.display());
    Ok(())
}

fn print_generation_hypervolume_curve(result: &fsrs::CostAdrTrainingResult) {
    for metrics in &result.history {
        let best_hypervolume = result.baseline_hypervolume + metrics.best_hypervolume_delta;
        let best_hypervolume_change_percent = if result.baseline_hypervolume == 0.0 {
            0.0
        } else {
            metrics.best_hypervolume_delta / result.baseline_hypervolume * 100.0
        };
        println!(
            "cost_adr_generation_hypervolume generation={} best_hypervolume={:.6} best_hypervolume_delta={:.6} best_hypervolume_change_percent={:.6} generation_best_hypervolume_delta={:.6} mean_hypervolume_delta={:.6} sigma={:.6}",
            metrics.generation + 1,
            best_hypervolume,
            metrics.best_hypervolume_delta,
            best_hypervolume_change_percent,
            metrics.generation_best_hypervolume_delta,
            metrics.mean_hypervolume_delta,
            metrics.sigma
        );
    }
}

fn print_interval_bucket_stats(
    config: &fsrs::SimulatorConfig,
    parameters: &[f32],
    policy: &CostAdrPolicy,
    seed: u64,
    result: &fsrs::CostAdrTrainingResult,
) -> fsrs::Result<()> {
    let bucket_configs = [
        IntervalBucketConfig {
            log_stability_step: 0.001,
            desired_retention_step: 0.0005,
        },
        IntervalBucketConfig {
            log_stability_step: 0.002,
            desired_retention_step: 0.001,
        },
        IntervalBucketConfig {
            log_stability_step: 0.005,
            desired_retention_step: 0.002,
        },
        IntervalBucketConfig {
            log_stability_step: 0.010,
            desired_retention_step: 0.005,
        },
    ];
    for (index, point) in result.best_cost_weight_metrics.iter().enumerate() {
        let stats = simulate_cost_adr_interval_bucket_stats(
            config,
            parameters,
            policy,
            point.goal_cost_weight,
            Some(seed + index as u64),
            None,
            &bucket_configs,
        )?;
        println!(
            "interval_bucket_stats cost_weight={:.1} total={} exact_unique={} exact_estimated_hits={} exact_estimated_hit_rate={:.6}",
            point.goal_cost_weight,
            stats.total_scheduled_intervals,
            stats.exact_unique_keys,
            stats.exact_estimated_hits,
            stats.exact_estimated_hit_rate
        );
        for summary in stats.bucket_summaries {
            println!(
                "interval_bucket_candidate cost_weight={:.1} log_s_step={:.6} dr_step={:.6} unique={} estimated_hits={} estimated_hit_rate={:.6} conflicting_keys={} estimated_interval_misses={} estimated_interval_miss_rate={:.6}",
                point.goal_cost_weight,
                summary.config.log_stability_step,
                summary.config.desired_retention_step,
                summary.unique_keys,
                summary.estimated_hits,
                summary.estimated_hit_rate,
                summary.conflicting_keys,
                summary.estimated_interval_misses,
                summary.estimated_interval_miss_rate
            );
        }
    }
    Ok(())
}

struct Args {
    collection_path: PathBuf,
    output_dir: PathBuf,
    deck_name_contains: Option<String>,
    card_ids_file: Option<PathBuf>,
    params_file: Option<PathBuf>,
    target_average_desired_retention: Option<f32>,
    model_version: ModelChoice,
    first_grade: Option<u8>,
    population_size: usize,
    generations: usize,
    seed: u64,
    endpoint_avg_dr_min_weight: Option<f32>,
    endpoint_avg_dr_max_weight: Option<f32>,
    endpoint_avg_dr_penalty: f32,
    early_stop_patience_generations: usize,
    early_stop_min_generations: usize,
    early_stop_min_relative_gain: f32,
    use_extracted_simulator_config: bool,
    interval_bucket_stats: bool,
}

impl Args {
    fn parse() -> fsrs::Result<Self> {
        let mut args = env::args().skip(1);
        let mut collection_path = PathBuf::from(DEFAULT_COLLECTION);
        let mut output_dir = PathBuf::from(DEFAULT_OUTPUT_DIR);
        let mut deck_name_contains = None;
        let mut card_ids_file = None;
        let mut params_file = None;
        let mut target_average_desired_retention = None;
        let mut model_version = ModelChoice::Fsrs7;
        let mut first_grade = None;
        let mut population_size = 8;
        let mut generations = CostAdrTrainingConfig::default().generations;
        let mut seed = DEFAULT_SEED;
        let mut endpoint_avg_dr_min_weight = None;
        let mut endpoint_avg_dr_max_weight = None;
        let mut endpoint_avg_dr_penalty = 0.0;
        let mut early_stop_patience_generations =
            default_cost_adr_early_stop_patience_generations();
        let mut early_stop_min_generations = default_cost_adr_early_stop_min_generations();
        let mut early_stop_min_relative_gain = default_cost_adr_early_stop_min_relative_gain();
        let mut use_extracted_simulator_config = false;
        let mut interval_bucket_stats = false;

        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--collection" => {
                    collection_path =
                        PathBuf::from(args.next().ok_or(fsrs::FSRSError::InvalidInput)?);
                }
                "--out-dir" => {
                    output_dir = PathBuf::from(args.next().ok_or(fsrs::FSRSError::InvalidInput)?);
                }
                "--deck-name-contains" => {
                    deck_name_contains = Some(args.next().ok_or(fsrs::FSRSError::InvalidInput)?);
                }
                "--card-ids-file" => {
                    card_ids_file = Some(PathBuf::from(
                        args.next().ok_or(fsrs::FSRSError::InvalidInput)?,
                    ));
                }
                "--params-file" => {
                    params_file = Some(PathBuf::from(
                        args.next().ok_or(fsrs::FSRSError::InvalidInput)?,
                    ));
                }
                "--target-avg-adr-dr" => {
                    target_average_desired_retention = Some(parse_arg(args.next())?);
                }
                "--model-version" => {
                    model_version = ModelChoice::parse(
                        args.next().ok_or(fsrs::FSRSError::InvalidInput)?.as_str(),
                    )?;
                }
                "--first-grade" => {
                    let value: u8 = parse_arg(args.next())?;
                    if !(1..=4).contains(&value) {
                        return Err(fsrs::FSRSError::InvalidInput);
                    }
                    first_grade = Some(value);
                }
                "--pop" => {
                    population_size = parse_arg(args.next())?;
                }
                "--gen" => {
                    generations = parse_arg(args.next())?;
                }
                "--seed" => {
                    seed = parse_arg(args.next())?;
                }
                "--endpoint-avg-dr-min-weight" => {
                    endpoint_avg_dr_min_weight = Some(parse_arg(args.next())?);
                }
                "--endpoint-avg-dr-max-weight" => {
                    endpoint_avg_dr_max_weight = Some(parse_arg(args.next())?);
                }
                "--endpoint-avg-dr-penalty" => {
                    endpoint_avg_dr_penalty = parse_arg(args.next())?;
                }
                "--early-stop-patience-generations" => {
                    early_stop_patience_generations = parse_arg(args.next())?;
                }
                "--early-stop-min-generations" => {
                    early_stop_min_generations = parse_arg(args.next())?;
                }
                "--early-stop-min-relative-gain" => {
                    early_stop_min_relative_gain = parse_arg(args.next())?;
                }
                "--use-extracted-simulator-config" => {
                    use_extracted_simulator_config = true;
                }
                "--interval-bucket-stats" => {
                    interval_bucket_stats = true;
                }
                _ => return Err(fsrs::FSRSError::InvalidInput),
            }
        }

        Ok(Self {
            collection_path,
            output_dir,
            deck_name_contains,
            card_ids_file,
            params_file,
            target_average_desired_retention,
            model_version,
            first_grade,
            population_size,
            generations,
            seed,
            endpoint_avg_dr_min_weight,
            endpoint_avg_dr_max_weight,
            endpoint_avg_dr_penalty,
            early_stop_patience_generations,
            early_stop_min_generations,
            early_stop_min_relative_gain,
            use_extracted_simulator_config,
            interval_bucket_stats,
        })
    }
}

fn default_cost_adr_early_stop_patience_generations() -> usize {
    4
}

fn default_cost_adr_early_stop_min_generations() -> usize {
    10
}

fn default_cost_adr_early_stop_min_relative_gain() -> f32 {
    0.01
}

#[derive(Debug, Clone, Copy)]
enum ModelChoice {
    Fsrs6,
    Fsrs7,
}

impl ModelChoice {
    fn parse(value: &str) -> fsrs::Result<Self> {
        match value {
            "fsrs6" | "FSRS6" | "6" => Ok(Self::Fsrs6),
            "fsrs7" | "FSRS7" | "7" => Ok(Self::Fsrs7),
            _ => Err(fsrs::FSRSError::InvalidInput),
        }
    }

    fn compute_parameters_version(self) -> ComputeParametersVersion {
        match self {
            Self::Fsrs6 => ComputeParametersVersion::Fsrs6,
            Self::Fsrs7 => ComputeParametersVersion::Fsrs7,
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::Fsrs6 => "fsrs6",
            Self::Fsrs7 => "fsrs7",
        }
    }
}

fn parse_arg<T: std::str::FromStr>(value: Option<String>) -> fsrs::Result<T> {
    value
        .ok_or(fsrs::FSRSError::InvalidInput)?
        .parse()
        .map_err(|_| fsrs::FSRSError::InvalidInput)
}

fn read_card_ids(path: &Path) -> fsrs::Result<HashSet<i64>> {
    let data = std::fs::read_to_string(path).map_err(|_| fsrs::FSRSError::InvalidInput)?;
    data.split(|character: char| character.is_ascii_whitespace() || character == ',')
        .filter(|value| !value.is_empty())
        .map(|value| value.parse().map_err(|_| fsrs::FSRSError::InvalidInput))
        .collect()
}

fn read_params(path: &Path) -> fsrs::Result<Vec<f32>> {
    let data = std::fs::read_to_string(path).map_err(|_| fsrs::FSRSError::InvalidInput)?;
    data.split(|character: char| character.is_ascii_whitespace() || character == ',')
        .filter(|value| !value.is_empty())
        .map(|value| value.parse().map_err(|_| fsrs::FSRSError::InvalidInput))
        .collect()
}

#[derive(Debug, Clone)]
struct SelectedDeck {
    id: i64,
    name: String,
}

fn read_collection(
    path: &Path,
    deck_name_contains: Option<&str>,
) -> fsrs::Result<(Vec<RevlogEntry>, i64, Vec<SelectedDeck>)> {
    let db = Connection::open(path).map_err(|_| fsrs::FSRSError::InvalidInput)?;
    let day_cutoff = collection_day_cutoff(&db)?;
    let selected_decks = selected_decks(&db, deck_name_contains)?;
    let selected_deck_ids = selected_decks
        .iter()
        .map(|deck| deck.id)
        .collect::<Vec<_>>();
    let (sql, deck_values) = if selected_deck_ids.is_empty() {
        (
            "SELECT r.id, r.cid, r.usn, r.ease, r.ivl, r.lastIvl, r.factor, r.time, r.type
             FROM revlog r
             WHERE r.ease BETWEEN 1 AND 4
             ORDER BY r.cid, r.id"
                .to_string(),
            Vec::new(),
        )
    } else {
        let placeholders = vec!["?"; selected_deck_ids.len()].join(",");
        (
            format!(
                "SELECT r.id, r.cid, r.usn, r.ease, r.ivl, r.lastIvl, r.factor, r.time, r.type
                 FROM revlog r
                 JOIN cards c ON c.id = r.cid
                 WHERE r.ease BETWEEN 1 AND 4
                   AND CASE WHEN c.odid != 0 THEN c.odid ELSE c.did END IN ({placeholders})
                 ORDER BY r.cid, r.id"
            ),
            selected_deck_ids,
        )
    };
    let mut stmt = db
        .prepare_cached(&sql)
        .map_err(|_| fsrs::FSRSError::InvalidInput)?;
    let revlogs = stmt
        .query_and_then(rusqlite::params_from_iter(deck_values), row_to_revlog)
        .map_err(|_| fsrs::FSRSError::InvalidInput)?
        .collect::<Result<Vec<_>, _>>()
        .map_err(|_| fsrs::FSRSError::InvalidInput)?;
    Ok((revlogs, day_cutoff, selected_decks))
}

fn collection_day_cutoff(db: &Connection) -> fsrs::Result<i64> {
    let has_col_table = db
        .query_row(
            "SELECT EXISTS(
                SELECT 1 FROM sqlite_master
                WHERE type = 'table' AND name = 'col'
            )",
            [],
            |row| row.get::<_, i64>(0),
        )
        .map_err(|_| fsrs::FSRSError::InvalidInput)?
        != 0;
    if has_col_table {
        db.query_row("SELECT crt FROM col LIMIT 1", [], |row| {
            row.get::<_, i64>(0)
        })
        .map_err(|_| fsrs::FSRSError::InvalidInput)
    } else {
        Ok(ASIA_SHANGHAI_4H_DAY_CUTOFF_SECONDS)
    }
}

fn selected_decks(
    db: &Connection,
    deck_name_contains: Option<&str>,
) -> fsrs::Result<Vec<SelectedDeck>> {
    let Some(deck_name_contains) = deck_name_contains else {
        return Ok(Vec::new());
    };
    let mut stmt = db
        .prepare_cached(
            "SELECT id, name
             FROM decks
             WHERE lower(name COLLATE binary) LIKE '%' || lower(?1) || '%'
             ORDER BY name COLLATE binary",
        )
        .map_err(|_| fsrs::FSRSError::InvalidInput)?;
    let decks = stmt
        .query_map([deck_name_contains], |row| {
            Ok(SelectedDeck {
                id: row.get(0)?,
                name: row.get(1)?,
            })
        })
        .map_err(|_| fsrs::FSRSError::InvalidInput)?
        .collect::<Result<Vec<_>, _>>()
        .map_err(|_| fsrs::FSRSError::InvalidInput)?;
    if decks.is_empty() {
        return Err(fsrs::FSRSError::InvalidInput);
    }
    Ok(decks)
}

fn row_to_revlog(row: &Row<'_>) -> rusqlite::Result<RevlogEntry> {
    let review_kind = match row.get::<_, i64>(8)? {
        0 => RevlogReviewKind::Learning,
        1 => RevlogReviewKind::Review,
        2 => RevlogReviewKind::Relearning,
        3 => RevlogReviewKind::Filtered,
        4 => RevlogReviewKind::Manual,
        _ => RevlogReviewKind::Manual,
    };
    Ok(RevlogEntry {
        id: row.get(0)?,
        cid: row.get(1)?,
        usn: row.get(2)?,
        button_chosen: row.get::<_, u8>(3)?,
        interval: row.get(4)?,
        last_interval: row.get(5)?,
        ease_factor: row.get(6)?,
        taken_millis: row.get(7)?,
        review_kind,
    })
}

fn revlogs_to_fsrs_items_with_card_ids(
    revlogs: &[RevlogEntry],
    day_cutoff: i64,
) -> (Vec<FSRSItem>, Vec<i64>) {
    let mut grouped = BTreeMap::<i64, Vec<RevlogEntry>>::new();
    for &entry in revlogs {
        grouped.entry(entry.cid).or_default().push(entry);
    }

    let mut items = Vec::new();
    let mut card_ids = Vec::new();
    for entries in grouped.into_values() {
        let entries = remove_before_last_first_learning(entries);
        if entries.len() < 2 {
            continue;
        }
        let mut reviews = Vec::new();
        for (index, entry) in entries.iter().enumerate() {
            let delta_t = if index == 0 {
                0.0
            } else {
                let previous = real_day(entries[index - 1].id, day_cutoff);
                let current = real_day(entry.id, day_cutoff);
                (current - previous).max(0) as f32
            };
            reviews.push(FSRSReview {
                rating: entry.button_chosen as u32,
                delta_t,
            });
            if index > 0 && delta_t > 0.0 {
                items.push(FSRSItem {
                    reviews: reviews.clone(),
                });
                card_ids.push(entry.cid);
            }
        }
    }
    (items, card_ids)
}

fn filter_revlogs_by_first_grade(revlogs: &[RevlogEntry], first_grade: u8) -> Vec<RevlogEntry> {
    let mut grouped = BTreeMap::<i64, Vec<RevlogEntry>>::new();
    for &entry in revlogs {
        grouped.entry(entry.cid).or_default().push(entry);
    }

    grouped
        .into_values()
        .filter_map(|entries| {
            let entries = remove_before_last_first_learning(entries);
            entries
                .first()
                .is_some_and(|entry| entry.button_chosen == first_grade)
                .then_some(entries)
        })
        .flatten()
        .collect()
}

fn remove_before_last_first_learning(entries: Vec<RevlogEntry>) -> Vec<RevlogEntry> {
    let mut last_first_learning = None;
    for (index, entry) in entries.iter().enumerate().rev() {
        if entry.review_kind == RevlogReviewKind::Learning {
            last_first_learning = Some(index);
        } else if last_first_learning.is_some() {
            break;
        }
    }
    last_first_learning
        .map(|index| entries[index..].to_vec())
        .unwrap_or_default()
}

fn active_new_card_days(revlogs: &[RevlogEntry], day_cutoff: i64) -> usize {
    revlogs
        .iter()
        .filter(|entry| entry.review_kind == RevlogReviewKind::Learning)
        .map(|entry| real_day(entry.id, day_cutoff))
        .collect::<HashSet<_>>()
        .len()
}

fn real_day(timestamp_millis: i64, day_cutoff: i64) -> i64 {
    (timestamp_millis / 1000 - day_cutoff) / 86400
}

#[cfg(test)]
mod tests {
    use super::*;

    fn revlog(cid: i64, day: i64, rating: u8, review_kind: RevlogReviewKind) -> RevlogEntry {
        RevlogEntry {
            id: day * 86400 * 1000,
            cid,
            usn: 0,
            button_chosen: rating,
            interval: 0,
            last_interval: 0,
            ease_factor: 0,
            taken_millis: 1000,
            review_kind,
        }
    }

    #[test]
    fn revlogs_to_fsrs_items_keeps_card_ids_aligned() {
        let revlogs = vec![
            revlog(10, 0, 3, RevlogReviewKind::Learning),
            revlog(10, 1, 4, RevlogReviewKind::Review),
            revlog(20, 0, 1, RevlogReviewKind::Learning),
            revlog(20, 2, 3, RevlogReviewKind::Review),
            revlog(20, 3, 2, RevlogReviewKind::Review),
        ];

        let (items, card_ids) = revlogs_to_fsrs_items_with_card_ids(&revlogs, 0);

        assert_eq!(card_ids, vec![10, 20, 20]);
        assert_eq!(items.len(), card_ids.len());
        assert_eq!(items[0].reviews.len(), 2);
        assert_eq!(items[1].reviews.len(), 2);
        assert_eq!(items[2].reviews.len(), 3);
    }
}

fn write_policy(output_dir: &Path, policy: &CostAdrPolicy) -> fsrs::Result<()> {
    let path = output_dir.join("policy.csv");
    let mut writer = BufWriter::new(File::create(path).map_err(|_| fsrs::FSRSError::InvalidInput)?);
    writeln!(writer, "index,coefficient").map_err(|_| fsrs::FSRSError::InvalidInput)?;
    for (index, coefficient) in policy.coefficients.iter().enumerate() {
        writeln!(writer, "{index},{coefficient}").map_err(|_| fsrs::FSRSError::InvalidInput)?;
    }
    Ok(())
}

fn write_retention_grid(output_dir: &Path, policy: &CostAdrPolicy) -> fsrs::Result<()> {
    let stabilities = logspace(policy.bounds.s_min.max(0.01), 36500.0, 80);
    let difficulties = linspace(policy.bounds.d_min, policy.bounds.d_max, 80);
    let cost_weights = [0.0, 16.0, 64.0, 256.0, 1024.0];
    let points = policy.retention_grid(&stabilities, &difficulties, &cost_weights)?;

    let path = output_dir.join("retention_grid.csv");
    let mut writer = BufWriter::new(File::create(path).map_err(|_| fsrs::FSRSError::InvalidInput)?);
    writeln!(writer, "stability,difficulty,cost_weight,desired_retention")
        .map_err(|_| fsrs::FSRSError::InvalidInput)?;
    for point in points {
        writeln!(
            writer,
            "{},{},{},{}",
            point.stability, point.difficulty, point.cost_weight, point.desired_retention
        )
        .map_err(|_| fsrs::FSRSError::InvalidInput)?;
    }
    Ok(())
}

fn linspace(min: f32, max: f32, count: usize) -> Vec<f32> {
    (0..count)
        .map(|index| min + (max - min) * index as f32 / (count - 1) as f32)
        .collect()
}

fn logspace(min: f32, max: f32, count: usize) -> Vec<f32> {
    let min = min.ln();
    let max = max.ln();
    (0..count)
        .map(|index| (min + (max - min) * index as f32 / (count - 1) as f32).exp())
        .collect()
}

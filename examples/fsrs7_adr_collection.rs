use fsrs::{
    CombinedProgressState, ComputeParametersInput, ComputeParametersVersion, CostAdrPolicy,
    CostAdrTrainingConfig, FSRSItem, FSRSReview, RevlogEntry, RevlogReviewKind, compute_parameters,
    extract_simulator_config,
};
use rusqlite::{Connection, Row};
use std::collections::{BTreeMap, HashSet};
use std::env;
use std::fs::{File, create_dir_all};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Instant;

const DEFAULT_COLLECTION: &str =
    "/Users/jschoreels/Library/Application Support/Anki2/Main Profile/collection.anki2";
const DEFAULT_OUTPUT_DIR: &str = "docs/fsrs7-adr-plots";
const DEFAULT_SEED: u64 = 42;

fn main() -> fsrs::Result<()> {
    let args = Args::parse()?;
    create_dir_all(&args.output_dir).map_err(|_| fsrs::FSRSError::InvalidInput)?;

    let collection_started = Instant::now();
    let (revlogs, day_cutoff, selected_decks) =
        read_collection(&args.collection_path, args.deck_name_contains.as_deref())?;
    let fsrs_items = revlogs_to_fsrs_items(&revlogs, day_cutoff);
    let collection_seconds = collection_started.elapsed().as_secs_f32();

    let unique_cards = revlogs
        .iter()
        .map(|entry| entry.cid)
        .collect::<HashSet<_>>();
    let active_days = active_new_card_days(&revlogs, day_cutoff).max(1);
    let learn_limit = ((unique_cards.len() as f32 / active_days as f32).round() as usize).max(1);

    let mut config = extract_simulator_config(revlogs.clone(), day_cutoff, true);
    config.deck_size = unique_cards.len();
    config.learn_span = active_days;
    config.learn_limit = learn_limit;
    config.review_limit = 9999;
    config.max_cost_perday = 720.0 * 60.0;

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
    println!("model_version={}", args.model_version.label());

    let fsrs_started = Instant::now();
    let parameters = compute_parameters(ComputeParametersInput {
        train_set: fsrs_items,
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
    println!("{}_parameters={parameters:?}", args.model_version.label());

    let adr_training_config = CostAdrTrainingConfig {
        population_size: args.population_size,
        generations: args.generations,
        seed: Some(args.seed),
        simulation_seed: Some(args.seed),
        progress: Some(Arc::new(Mutex::new(Default::default()))),
        ..Default::default()
    };

    let adr_started = Instant::now();
    let result = CostAdrPolicy::train_single_user(&config, &parameters, &adr_training_config)?;
    let adr_wall_seconds = adr_started.elapsed().as_secs_f32();
    println!(
        "cost_adr_training_seconds={:.3} result_training_seconds={:.3}",
        adr_wall_seconds, result.training_seconds
    );
    println!(
        "baseline_hypervolume={:.6} best_hypervolume={:.6} best_hypervolume_delta={:.6}",
        result.baseline_hypervolume, result.best_hypervolume, result.best_hypervolume_delta
    );
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

    write_policy(&args.output_dir, &result.policy)?;
    write_retention_grid(&args.output_dir, &result.policy)?;
    println!("output_dir={}", args.output_dir.display());
    Ok(())
}

struct Args {
    collection_path: PathBuf,
    output_dir: PathBuf,
    deck_name_contains: Option<String>,
    target_average_desired_retention: Option<f32>,
    model_version: ModelChoice,
    population_size: usize,
    generations: usize,
    seed: u64,
}

impl Args {
    fn parse() -> fsrs::Result<Self> {
        let mut args = env::args().skip(1);
        let mut collection_path = PathBuf::from(DEFAULT_COLLECTION);
        let mut output_dir = PathBuf::from(DEFAULT_OUTPUT_DIR);
        let mut deck_name_contains = None;
        let mut target_average_desired_retention = None;
        let mut model_version = ModelChoice::Fsrs7;
        let mut population_size = 8;
        let mut generations = 5;
        let mut seed = DEFAULT_SEED;

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
                "--target-avg-adr-dr" => {
                    target_average_desired_retention = Some(parse_arg(args.next())?);
                }
                "--model-version" => {
                    model_version = ModelChoice::parse(
                        args.next().ok_or(fsrs::FSRSError::InvalidInput)?.as_str(),
                    )?;
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
                _ => return Err(fsrs::FSRSError::InvalidInput),
            }
        }

        Ok(Self {
            collection_path,
            output_dir,
            deck_name_contains,
            target_average_desired_retention,
            model_version,
            population_size,
            generations,
            seed,
        })
    }
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
    let day_cutoff = db
        .query_row("SELECT crt FROM col LIMIT 1", [], |row| {
            row.get::<_, i64>(0)
        })
        .map_err(|_| fsrs::FSRSError::InvalidInput)?;
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

fn revlogs_to_fsrs_items(revlogs: &[RevlogEntry], day_cutoff: i64) -> Vec<FSRSItem> {
    let mut grouped = BTreeMap::<i64, Vec<RevlogEntry>>::new();
    for &entry in revlogs {
        grouped.entry(entry.cid).or_default().push(entry);
    }

    let mut items = Vec::new();
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
            }
        }
    }
    items
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

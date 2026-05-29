use chrono::{DateTime, Duration, Utc};
use fsrs::{
    CombinedProgressState, CostAdrEvaluationConfig, CostAdrEvaluationResult, CostAdrPolicy,
    CostAdrTrainingConfig, DEFAULT_PARAMETERS, FSRS, FSRSError, MemoryState, SimulatorConfig,
};
use std::env;
use std::error::Error;
use std::fmt::Display;
use std::io::{Error as IoError, ErrorKind, Write};
use std::str::FromStr;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration as StdDuration, Instant};

const USAGE: &str = "\
Usage: cargo run --release --example cost_adr -- [OPTIONS]

Options:
  --days <usize>            Simulation learn span in days (default: 1825)
  --deck <usize>            Simulated deck size (default: 10000)
  --learn-limit <usize>     New-card limit per day (default: 10)
  --review-limit <usize>    Review limit per day (default: 9999)
  --cost-limit-minutes <f32>
                            Study time limit per day in minutes (default: 720.0)
  --pop <usize>             CMA-ES population size (default: 16)
  --gen <usize>             CMA-ES generation count (default: 20)
  --seed <u64>              Optimizer and simulation seed (default: 42)
  --sigma0 <f32>            CMA-ES initial sigma (default: 1.0)
  --goal-weight <f32>       Runtime scheduling cost weight (default: 64.0)
  -h, --help                Print help
";

struct ExampleConfig {
    days: usize,
    deck_size: usize,
    learn_limit: usize,
    review_limit: usize,
    cost_limit_minutes: f32,
    population_size: usize,
    generations: usize,
    seed: Option<u64>,
    sigma0: f32,
    goal_cost_weight: f32,
}

impl Default for ExampleConfig {
    fn default() -> Self {
        let training_config = CostAdrTrainingConfig::default();
        Self {
            days: 1_825,
            deck_size: 10_000,
            learn_limit: 10,
            review_limit: 9_999,
            cost_limit_minutes: 720.0,
            population_size: training_config.population_size,
            generations: training_config.generations,
            seed: training_config.seed,
            sigma0: training_config.sigma0,
            goal_cost_weight: 64.0,
        }
    }
}

fn invalid_arg(message: impl Into<String>) -> Box<dyn Error> {
    Box::new(IoError::new(ErrorKind::InvalidInput, message.into()))
}

fn parse_value<T>(flag: &str, value: &str) -> Result<T, Box<dyn Error>>
where
    T: FromStr,
    T::Err: Display,
{
    value
        .parse()
        .map_err(|err| invalid_arg(format!("invalid value for {flag}: {value} ({err})")))
}

fn next_arg_value<I>(args: &mut I, flag: &str) -> Result<String, Box<dyn Error>>
where
    I: Iterator<Item = String>,
{
    args.next()
        .ok_or_else(|| invalid_arg(format!("missing value for {flag}")))
}

fn arg_value<I>(
    args: &mut I,
    flag: &str,
    inline_value: Option<String>,
) -> Result<String, Box<dyn Error>>
where
    I: Iterator<Item = String>,
{
    match inline_value {
        Some(value) => Ok(value),
        None => next_arg_value(args, flag),
    }
}

fn parse_args() -> Result<Option<ExampleConfig>, Box<dyn Error>> {
    let mut config = ExampleConfig::default();
    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        if arg == "-h" || arg == "--help" {
            print!("{USAGE}");
            return Ok(None);
        }

        let (flag, inline_value) = if let Some((flag, value)) = arg.split_once('=') {
            (flag, Some(value.to_string()))
        } else {
            (arg.as_str(), None)
        };

        match flag {
            "--days" => {
                config.days = parse_value(flag, &arg_value(&mut args, flag, inline_value)?)?
            }
            "--deck" => {
                config.deck_size = parse_value(flag, &arg_value(&mut args, flag, inline_value)?)?
            }
            "--learn-limit" => {
                config.learn_limit = parse_value(flag, &arg_value(&mut args, flag, inline_value)?)?
            }
            "--review-limit" => {
                config.review_limit = parse_value(flag, &arg_value(&mut args, flag, inline_value)?)?
            }
            "--cost-limit-minutes" => {
                config.cost_limit_minutes =
                    parse_value(flag, &arg_value(&mut args, flag, inline_value)?)?
            }
            "--pop" => {
                config.population_size =
                    parse_value(flag, &arg_value(&mut args, flag, inline_value)?)?
            }
            "--gen" => {
                config.generations = parse_value(flag, &arg_value(&mut args, flag, inline_value)?)?
            }
            "--seed" => {
                config.seed = Some(parse_value(
                    flag,
                    &arg_value(&mut args, flag, inline_value)?,
                )?)
            }
            "--sigma0" => {
                config.sigma0 = parse_value(flag, &arg_value(&mut args, flag, inline_value)?)?
            }
            "--goal-weight" => {
                config.goal_cost_weight =
                    parse_value(flag, &arg_value(&mut args, flag, inline_value)?)?
            }
            _ => return Err(invalid_arg(format!("unknown argument: {flag}"))),
        }
    }
    Ok(Some(config))
}

fn format_optional(value: Option<f32>) -> String {
    value
        .map(|value| format!("{value:.6}"))
        .unwrap_or_else(|| "n/a".to_string())
}

fn spawn_progress_printer(progress: Arc<Mutex<CombinedProgressState>>) -> thread::JoinHandle<()> {
    thread::spawn(move || {
        let mut stderr = std::io::stderr();
        loop {
            let (current, total, finished) = {
                let progress = progress.lock().unwrap();
                (progress.current(), progress.total(), progress.finished())
            };
            eprint!("\r{}", progress_line(current, total, finished));
            let _ = stderr.flush();

            if finished {
                eprintln!();
                break;
            }

            thread::sleep(StdDuration::from_millis(100));
        }
    })
}

fn progress_line(current: usize, total: usize, finished: bool) -> String {
    if total == 0 {
        return "Cost ADR training [initializing]".to_string();
    }

    const WIDTH: usize = 32;
    let current = current.min(total);
    let filled = ((current * WIDTH) / total).min(WIDTH);
    let bar = format!("{}{}", "=".repeat(filled), " ".repeat(WIDTH - filled));
    let percent = current as f32 * 100.0 / total as f32;
    let status = if finished { "done" } else { "training" };
    format!("Cost ADR {status} [{bar}] {current}/{total} ({percent:.1}%)")
}

fn print_default_policy_evaluation(result: &CostAdrEvaluationResult) {
    println!(
        "default_policy_baseline_hypervolume={:.6} default_policy_hypervolume={:.6} default_policy_hypervolume_delta={:.6}",
        result.baseline_hypervolume, result.scheduler_hypervolume, result.hypervolume_delta
    );
    println!(
        "default_policy_span_coverage_percent={:.3} covered_span={:.3} total_span={:.3} covered_targets={}/{} baseline_frontier={} scheduler_frontier={}",
        result.auc_metrics.span_coverage_percent,
        result.auc_metrics.covered_span,
        result.auc_metrics.total_span,
        result.auc_metrics.covered_target_count,
        result.auc_metrics.target_count,
        result.auc_metrics.baseline_frontier_count,
        result.auc_metrics.scheduler_frontier_count
    );
    println!(
        "default_policy_same_target_time_saved_auc={} baseline_time_auc={} relative_same_target_time_saved_auc_percent={}",
        format_optional(result.auc_metrics.same_target_time_saved_auc),
        format_optional(result.auc_metrics.baseline_time_auc),
        format_optional(
            result
                .auc_metrics
                .relative_same_target_time_saved_auc_percent
        )
    );
    println!("default policy cost-weight rollout points:");
    for point in &result.scheduler_metrics {
        println!(
            "  w={:<7.1} avg_dr={:>8} memorized_avg={:>9.3} time_avg_min={:>7.3} mem_per_min={:>9.3} reviews={} lapses={}",
            point.goal_cost_weight,
            format_optional(point.average_desired_retention),
            point.metrics.memorized_average,
            point.metrics.time_average,
            point.metrics.memorized_per_minute,
            point.metrics.total_reviews,
            point.metrics.total_lapses
        );
    }
}

struct ScheduledReview {
    memory_state: MemoryState,
    desired_retention: f32,
    interval_days: u32,
    due: DateTime<Utc>,
}

fn schedule_after_review_with_cost_adr(
    fsrs: &FSRS,
    policy: &CostAdrPolicy,
    previous_state: Option<MemoryState>,
    elapsed_days: u32,
    rating: u32,
    goal_cost_weight: f32,
    max_interval_days: u32,
) -> fsrs::Result<ScheduledReview> {
    policy.validate()?;

    // The desired retention passed here does not affect the memory-state update.
    // It only affects the intervals carried by NextStates, which we recompute below.
    let next_states = fsrs.next_states(previous_state, 0.9, elapsed_days)?;
    let reviewed_state = match rating {
        1 => next_states.again,
        2 => next_states.hard,
        3 => next_states.good,
        4 => next_states.easy,
        _ => return Err(FSRSError::InvalidInput),
    };
    let desired_retention = policy.evaluate_retention(
        reviewed_state.memory.stability,
        reviewed_state.memory.difficulty,
        goal_cost_weight,
    );
    let interval_days = fsrs
        .next_interval(
            Some(reviewed_state.memory.stability),
            desired_retention,
            rating,
        )
        .round()
        .clamp(1.0, max_interval_days as f32) as u32;

    Ok(ScheduledReview {
        memory_state: reviewed_state.memory,
        desired_retention,
        interval_days,
        due: Utc::now() + Duration::days(interval_days as i64),
    })
}

fn main() -> fsrs::Result<()> {
    let Some(example_config) = parse_args().map_err(|err| {
        eprintln!("{err}");
        eprintln!();
        eprint!("{USAGE}");
        FSRSError::InvalidInput
    })?
    else {
        return Ok(());
    };

    let config = SimulatorConfig {
        deck_size: example_config.deck_size,
        learn_span: example_config.days,
        learn_limit: example_config.learn_limit,
        review_limit: example_config.review_limit,
        max_cost_perday: example_config.cost_limit_minutes * 60.0,
        ..Default::default()
    };
    let progress = CombinedProgressState::new_shared();
    let training_config = CostAdrTrainingConfig {
        population_size: example_config.population_size,
        generations: example_config.generations,
        sigma0: example_config.sigma0,
        seed: example_config.seed,
        simulation_seed: example_config.seed,
        progress: Some(progress.clone()),
        ..Default::default()
    };

    let default_policy = CostAdrPolicy::new(training_config.initial_coefficients.clone())?;
    let evaluation_config = CostAdrEvaluationConfig {
        seed: example_config.seed,
        ..Default::default()
    };
    println!("Cost ADR default policy evaluation");
    let started = Instant::now();
    let default_evaluation =
        default_policy.evaluate(&config, &DEFAULT_PARAMETERS, &evaluation_config)?;
    println!(
        "default_policy_duration_seconds={:.3}",
        started.elapsed().as_secs_f32()
    );
    print_default_policy_evaluation(&default_evaluation);

    // For a production user, replace DEFAULT_PARAMETERS with parameters trained
    // from that user's revlog via compute_parameters.
    let progress_printer = spawn_progress_printer(progress.clone());
    let started = Instant::now();
    let result = CostAdrPolicy::train_single_user(&config, &DEFAULT_PARAMETERS, &training_config);
    let wall_seconds = started.elapsed().as_secs_f32();
    let _ = progress_printer.join();
    let result = result?;

    println!("Cost ADR single-user FSRS training");
    println!(
        "config days={} deck={} learn_limit={} review_limit={} cost_limit_minutes={} pop={} gen={} sigma0={} seed={}",
        example_config.days,
        example_config.deck_size,
        example_config.learn_limit,
        example_config.review_limit,
        example_config.cost_limit_minutes,
        example_config.population_size,
        example_config.generations,
        example_config.sigma0,
        example_config.seed.unwrap_or(42)
    );
    println!(
        "duration_seconds={:.3} result_training_seconds={:.3}",
        wall_seconds, result.training_seconds
    );
    println!(
        "baseline_hypervolume={:.6} best_hypervolume={:.6} best_hypervolume_delta={:.6}",
        result.baseline_hypervolume, result.best_hypervolume, result.best_hypervolume_delta
    );
    println!(
        "span_coverage_percent={:.3} covered_span={:.3} total_span={:.3} covered_targets={}/{} baseline_frontier={} scheduler_frontier={}",
        result.best_auc_metrics.span_coverage_percent,
        result.best_auc_metrics.covered_span,
        result.best_auc_metrics.total_span,
        result.best_auc_metrics.covered_target_count,
        result.best_auc_metrics.target_count,
        result.best_auc_metrics.baseline_frontier_count,
        result.best_auc_metrics.scheduler_frontier_count
    );
    println!(
        "same_target_time_saved_auc={} baseline_time_auc={} relative_same_target_time_saved_auc_percent={}",
        format_optional(result.best_auc_metrics.same_target_time_saved_auc),
        format_optional(result.best_auc_metrics.baseline_time_auc),
        format_optional(
            result
                .best_auc_metrics
                .relative_same_target_time_saved_auc_percent
        )
    );
    if let Some(last) = result.history.last() {
        println!(
            "last_generation={} best_delta={:.6} generation_best_delta={:.6} mean_delta={:.6} sigma={:.6}",
            last.generation,
            last.best_hypervolume_delta,
            last.generation_best_hypervolume_delta,
            last.mean_hypervolume_delta,
            last.sigma
        );
    }
    println!("selected cost-weight rollout points:");
    for point in &result.best_cost_weight_metrics {
        println!(
            "  w={:<7.1} avg_dr={:>8} memorized_avg={:>9.3} time_avg_min={:>7.3} mem_per_min={:>9.3} reviews={} lapses={}",
            point.goal_cost_weight,
            format_optional(point.average_desired_retention),
            point.metrics.memorized_average,
            point.metrics.time_average,
            point.metrics.memorized_per_minute,
            point.metrics.total_reviews,
            point.metrics.total_lapses
        );
    }

    // In production, persist the policy with the user's FSRS parameters and chosen
    // goal_cost_weight. CostAdrPolicy derives serde Serialize/Deserialize.
    let user_policy = result.policy.clone();
    println!(
        "persist policy coefficient_count={} goal_cost_weight={}",
        user_policy.coefficients.len(),
        example_config.goal_cost_weight
    );
    println!("policy={user_policy:#?}");

    let fsrs = FSRS::new(&DEFAULT_PARAMETERS)?;
    let previous_state = Some(MemoryState {
        stability: 7.0,
        difficulty: 5.0,
    });
    let scheduled = schedule_after_review_with_cost_adr(
        &fsrs,
        &user_policy,
        previous_state,
        7,
        3,
        example_config.goal_cost_weight,
        config.max_ivl.round() as u32,
    )?;
    println!(
        "runtime schedule rating=Good stability={:.3} difficulty={:.3} desired_retention={:.6} interval_days={} due={}",
        scheduled.memory_state.stability,
        scheduled.memory_state.difficulty,
        scheduled.desired_retention,
        scheduled.interval_days,
        scheduled.due
    );
    Ok(())
}

use chrono::{DateTime, Duration, Utc};
use fsrs::{
    CostAdrPolicy, CostAdrTrainingConfig, DEFAULT_PARAMETERS, FSRS, FSRSError, MemoryState,
    SimulatorConfig, train_cost_adr_single_user,
};
use std::env;
use std::time::Instant;

fn env_usize(name: &str, default: usize) -> usize {
    env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

fn env_u64(name: &str, default: u64) -> u64 {
    env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

fn env_f32(name: &str, default: f32) -> f32 {
    env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

fn format_optional(value: Option<f32>) -> String {
    value
        .map(|value| format!("{value:.6}"))
        .unwrap_or_else(|| "n/a".to_string())
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
    let days = env_usize("COST_ADR_DAYS", 90);
    let deck_size = env_usize("COST_ADR_DECK", 2_000);
    let learn_limit = env_usize("COST_ADR_LEARN_LIMIT", 40);
    let review_limit = env_usize("COST_ADR_REVIEW_LIMIT", 400);
    let population_size = env_usize("COST_ADR_POP", 8);
    let generations = env_usize("COST_ADR_GEN", 5);
    let seed = env_u64("COST_ADR_SEED", 42);
    let sigma0 = env_f32("COST_ADR_SIGMA0", 1.0);
    let goal_cost_weight = env_f32("COST_ADR_GOAL_WEIGHT", 64.0);

    let config = SimulatorConfig {
        deck_size,
        learn_span: days,
        learn_limit,
        review_limit,
        ..Default::default()
    };
    let training_config = CostAdrTrainingConfig {
        population_size,
        generations,
        sigma0,
        seed,
        simulation_seed: seed,
        ..Default::default()
    };

    // For a production user, replace DEFAULT_PARAMETERS with parameters trained
    // from that user's revlog via compute_parameters.
    let started = Instant::now();
    let result = train_cost_adr_single_user(&config, &DEFAULT_PARAMETERS, &training_config)?;
    let wall_seconds = started.elapsed().as_secs_f32();

    println!("Cost ADR single-user FSRS training");
    println!(
        "config days={days} deck={deck_size} learn_limit={learn_limit} review_limit={review_limit} pop={population_size} gen={generations} sigma0={sigma0} seed={seed}"
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
            "  w={:<7.1} memorized_avg={:>9.3} time_avg_min={:>7.3} mem_per_min={:>9.3} reviews={} lapses={}",
            point.goal_cost_weight,
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
        goal_cost_weight
    );

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
        goal_cost_weight,
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

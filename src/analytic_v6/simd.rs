//! Eight-lane FSRS-6 reverse-mode kernel: each SIMD lane follows one review
//! history. The forward pass caches the terms the reverse pass needs. The
//! scalar kernel in `analytic_v6.rs` remains the numerical reference; groups
//! that would lose precision here are recomputed with it.

use super::{D_MAX, D_MIN, MAX_PROB, MIN_PROB, RuntimeParams, S_MAX, S_MIN};
use wide::{f32x8 as F, i32x8 as I};

pub(super) const LANES: usize = 8;
pub(super) const PARAMS: usize = 21;

/// One row of a group: the review at this step for each lane.
#[derive(Clone, Copy, Default)]
pub(super) struct Review {
    /// Elapsed days since the previous review.
    pub(super) t: F,
    /// Rating at this step; 0 when the lane has no transition here.
    pub(super) rating: F,
    /// Sample weight of a scored prediction at this step, negative when the
    /// review was recalled and positive when forgotten; 0 if not scored.
    pub(super) weight: F,
}

pub(super) fn signed_weight(weight: f32, recalled: bool) -> f32 {
    if recalled { -weight } else { weight }
}

/// Reusable forward-pass storage for one thread.
#[derive(Default)]
pub(crate) struct Workspace {
    steps: Vec<Step>,
}

/// Gradient of the summed weighted BCE of one group, per parameter and lane.
/// `bounded` asserts that every elapsed time lies in `[0, S_MAX]`, which keeps
/// the curve's log/exp arguments in the fast approximations' range.
pub(super) fn group_gradient(
    p: &RuntimeParams<'_>,
    reviews: &[Review],
    bounded: bool,
    workspace: &mut Workspace,
) -> [F; PARAMS] {
    if bounded {
        trajectory_gradient::<true>(p, reviews, &mut workspace.steps)
    } else {
        trajectory_gradient::<false>(p, reviews, &mut workspace.steps)
    }
}

/// Forward-pass values cached for the reverse pass, one per review step.

#[derive(Clone, Default)]
struct Step {
    s: F,
    inv_s: F,
    d: F,
    ln_s: F,
    ln_d: F,
    ln_s1: F,
    r: F,
    base: F,
    ln_base: F,
    t: F,
    score_log_r: F,
    rating: F,
    short: F,
    failure: F,
    success: F,
    short_raw: F,
    short_value: F,
    short_live: F,
    failure_raw: F,
    failure_floor: F,
    floor_live: F,
    failure_prefactor: F,
    failure_power: F,
    success_base: F,
    success_exp: F,
    success_inc: F,
    next_d: F,
    s_live: F,
    d_live: F,
}

#[inline(always)]
fn clamp(x: F, low: f32, high: f32) -> F {
    x.fast_max(F::splat(low)).fast_min(F::splat(high))
}

#[inline(always)]
pub(super) fn live(x: F, low: f32, high: f32) -> F {
    x.simd_ge(F::splat(low)) & x.simd_le(F::splat(high))
}

// The recurrence uses positive normal log arguments and modest exponentials.
// Range reduction plus short polynomials avoids the general pow/exp
// exceptional-case machinery on every lane. Keep wide's fallback for unusual
// input intervals, including infinities. The exp relative error is below 4e-6;
// gradient and training-loss tests bound the accumulated error of the recurrence.
#[inline(always)]
fn exp(x: F) -> F {
    if !live(x, -87.0, 88.0).all() {
        return x.exp();
    }
    exp_in_range(x)
}

#[inline(always)]
fn exp_in_range(x: F) -> F {
    let k = F::splat;
    let ni = (x * k(std::f32::consts::LOG2_E)).fast_round_int();
    let n = ni.round_float();
    let r = (x - n * k(0.693_359_4)) - n * k(-0.000_212_194_44);
    let poly = k(1.0)
        + r * (k(1.0)
            + r * (k(0.5) + r * (k(1.0 / 6.0) + r * (k(1.0 / 24.0) + r * k(1.0 / 120.0)))));
    let power: F = bytemuck::cast((ni + I::splat(127)) << 23);
    power * poly
}

#[inline(always)]
fn ln(x: F) -> F {
    if !live(x, f32::MIN_POSITIVE, f32::MAX).all() {
        return x.ln();
    }
    ln_positive(x)
}

#[inline(always)]
fn ln_positive(x: F) -> F {
    let k = F::splat;
    let bits: I = bytemuck::cast(x);
    let exponent_bits: I = (bits >> 23) - I::splat(127);
    let exponent = exponent_bits.round_float();
    let mantissa: F = bytemuck::cast((bits & I::splat(0x007f_ffff)) | I::splat(127 << 23));
    let upper = mantissa.simd_gt(k(std::f32::consts::SQRT_2));
    let m = upper.select(mantissa * k(0.5), mantissa);
    let e = upper.select(exponent + k(1.0), exponent);
    let t = (m - k(1.0)) / (m + k(1.0));
    let t2 = t * t;
    let poly = k(2.0) * t * (k(1.0) + t2 * (k(1.0 / 3.0) + t2 * k(1.0 / 5.0)));
    e * k(std::f32::consts::LN_2) + poly
}

/// Recomputes a group with the scalar reference kernel (f32 forward, f64
/// adjoints). Lane 0 carries the whole group's gradient.
fn scalar_group_gradient(p: &RuntimeParams<'_>, reviews: &[Review]) -> [F; PARAMS] {
    let t: Vec<_> = reviews.iter().flat_map(|r| r.t.to_array()).collect();
    let ratings: Vec<_> = reviews.iter().flat_map(|r| r.rating.to_array()).collect();
    let signed: Vec<_> = reviews.iter().flat_map(|r| r.weight.to_array()).collect();
    let labels: Vec<_> = signed.iter().map(|&v| f32::from(v < 0.0)).collect();
    let weights: Vec<_> = signed.iter().map(|v| v.abs()).collect();
    let lengths: [usize; LANES] = std::array::from_fn(|lane| {
        reviews
            .iter()
            .rposition(|r| r.rating.as_array()[lane] != 0.0 || r.weight.as_array()[lane] != 0.0)
            .map_or(0, |i| i + 1)
    });
    let mut grad = [0.0; PARAMS];
    super::card_loss_and_grad(
        p.w,
        &t,
        &ratings,
        reviews.len(),
        LANES,
        &lengths,
        &labels,
        &weights,
        &mut grad,
    );
    grad.map(|g| F::from([g as f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
}

fn trajectory_gradient<const BOUNDED: bool>(
    p: &RuntimeParams<'_>,
    reviews: &[Review],
    steps: &mut Vec<Step>,
) -> [F; PARAMS] {
    let z = F::ZERO;
    let one = F::ONE;
    let k = F::splat;
    let w = p.w;
    let sp = |i: usize| k(w[i]);
    let mut gw = [z; PARAMS];
    let Some(first) = reviews.first() else {
        return gw;
    };
    let rating = first.rating;
    let mut initial_s = sp(0);
    for i in 1..4 {
        initial_s = rating.simd_eq(k((i + 1) as f32)).select(sp(i), initial_s);
    }
    let offset = (rating - one).fast_max(z);
    let initial_exp = exp_in_range(sp(5) * offset);
    let initial_d = sp(4) - initial_exp + one;
    let mut s = clamp(initial_s, S_MIN, S_MAX);
    let mut d = clamp(initial_d, D_MIN, D_MAX);
    let required = reviews.len().saturating_sub(1);
    if steps.len() < required {
        steps.resize_with(required, Step::default);
    }
    let steps = &mut steps[..required];

    // Forward pass: predict each row from the state before it, then update
    // stability and difficulty with that row's rating.
    for (row, (review, c)) in reviews.iter().skip(1).zip(steps.iter_mut()).enumerate() {
        c.s = s;
        c.inv_s = one / s;
        c.d = d;
        c.t = review.t.fast_max(z);
        c.rating = review.rating;
        c.base = c.t * c.inv_s * k(p.curve_factor) + one;
        // For t <= S_MAX and clipped parameters/stability, the curve base is
        // positive/normal and its exponential argument is inside [-87, 88].
        // Validate once during layout rather than on every training iteration.
        c.ln_base = if BOUNDED {
            ln_positive(c.base)
        } else {
            ln(c.base)
        };
        c.r = if BOUNDED {
            exp_in_range(c.ln_base * k(p.curve_decay))
        } else {
            exp(c.ln_base * k(p.curve_decay))
        };
        let r = clamp(c.r, MIN_PROB, MAX_PROB);
        let scored = c.r.simd_gt(k(MIN_PROB)) & c.r.simd_lt(k(MAX_PROB));
        // Targets are binary recall labels. Select the active BCE term instead
        // of dividing by both r and 1-r, including the term with a zero label.
        let recalled = review.weight.simd_lt(z);
        let forgotten = review.weight.simd_gt(z);
        let score = if forgotten.any() {
            recalled.select(review.weight, review.weight * r / (one - r))
        } else {
            review.weight
        };
        // Store the adjoint of log(R). Its recalled term is simply -weight,
        // eliminating a divide followed by multiplication by the same R.
        c.score_log_r = scored.select(score, z);
        // The last target has no subsequent prediction consuming its state.
        if row + 2 == reviews.len() {
            break;
        }
        let active = review.rating.simd_gt(z);
        c.short = active & c.t.simd_eq(z);
        c.failure = active & !c.short & review.rating.simd_eq(one);
        c.success = active & !c.short & !c.failure;
        c.ln_s = ln_positive(s);
        let mut next_s = s;

        if c.short.any() {
            c.short_raw =
                exp_in_range(sp(17) * (review.rating - k(3.0) + sp(18)) - sp(19) * c.ln_s);
            c.short_live = !(review.rating.simd_ge(k(2.0)) & c.short_raw.simd_lt(one));
            c.short_value = c.short_live.select(c.short_raw, one);
            next_s = c.short.select(s * c.short_value, next_s);
        }
        if c.failure.any() {
            c.ln_d = ln_positive(d);
            c.ln_s1 = ln_positive(s + one);
            c.failure_power = exp_in_range(sp(13) * c.ln_s1);
            c.failure_prefactor = sp(11) * exp_in_range(-sp(12) * c.ln_d + (one - c.r) * sp(14));
            c.failure_raw = c.failure_prefactor * (c.failure_power - one);
            c.failure_floor = s / k(p.failure_floor_divisor);
            c.floor_live = c.failure_floor.simd_lt(c.failure_raw);
            next_s = c
                .failure
                .select(c.floor_live.select(c.failure_floor, c.failure_raw), next_s);
        }
        if c.success.any() {
            let hard = review.rating.simd_eq(k(2.0)).select(sp(15), one);
            let easy = review.rating.simd_eq(k(4.0)).select(sp(16), one);
            c.success_base = k(p.exp_w8) * (k(11.0) - d) * exp_in_range(-sp(9) * c.ln_s);
            c.success_exp = exp_in_range((one - c.r) * sp(10));
            // Tiny exp(x)-1 amplifies f32 rounding throughout the trajectory and its adjoints.
            if (c.success & (c.success_exp - one).simd_lt(k(0.001))).any() {
                return scalar_group_gradient(p, reviews);
            }
            c.success_inc = c.success_base * (c.success_exp - one) * hard * easy;
            next_s = c.success.select(s * (one + c.success_inc), next_s);
        }
        let delta_d = -sp(6) * (review.rating - k(3.0));
        c.next_d = d + (k(10.0) - d) * delta_d / k(9.0);
        let mean_d = sp(7) * (k(p.easy_d) - c.next_d) + c.next_d;
        c.s_live = live(next_s, S_MIN, S_MAX);
        c.d_live = live(mean_d, D_MIN, D_MAX);
        s = clamp(next_s, S_MIN, S_MAX);
        d = active.select(clamp(mean_d, D_MIN, D_MAX), d);
    }

    // Reverse pass: gs and gd are the adjoints of the state entering a row.
    let mut gs = z;
    let mut gd = z;
    for (row, c) in steps.iter().enumerate().rev() {
        let mut gr = z;
        if row + 1 != steps.len() {
            let active = c.rating.simd_gt(z);
            let g = c.s_live.select(gs, z);
            let mut previous_s = active.select(z, gs);
            let mut previous_d = active.select(z, gd);
            if c.short.any() {
                let g = c.short.select(g, z);
                let g_raw = c.short_live.select(g * c.s * c.short_raw, z);
                gw[17] += g_raw * (c.rating - k(3.0) + sp(18));
                gw[18] += g_raw * sp(17);
                gw[19] -= g_raw * c.ln_s;
                previous_s += g * c.short_value - g_raw * sp(19) * c.inv_s;
            }
            if c.failure.any() {
                let g_floor = (c.failure & c.floor_live).select(g * c.failure_floor, z);
                gw[17] -= g_floor * sp(18);
                gw[18] -= g_floor * sp(17);
                previous_s += g_floor * c.inv_s;

                let g = (c.failure & !c.floor_live).select(g, z);
                let g_raw = g * c.failure_raw;
                let g_power = g * c.failure_prefactor * c.failure_power;
                gw[11] += g_raw / sp(11);
                gw[12] -= g_raw * c.ln_d;
                gw[13] += g_power * c.ln_s1;
                gw[14] += g_raw * (one - c.r);
                previous_s += g_power * sp(13) / (c.s + one);
                previous_d -= g_raw * sp(12) / c.d;
                gr -= g_raw * sp(14);
            }
            if c.success.any() {
                let g = c.success.select(g, z);
                let hard = c.rating.simd_eq(k(2.0)).select(sp(15), one);
                let easy = c.rating.simd_eq(k(4.0)).select(sp(16), one);
                let g_inc = g * c.s;
                let g_increment = g_inc * c.success_inc;
                let g_exp = g_inc * c.success_base * hard * easy * c.success_exp;
                gw[8] += g_increment;
                gw[9] -= g_increment * c.ln_s;
                gw[10] += g_exp * (one - c.r);
                // Avoid dividing by w15: the hard multiplier may be exactly zero.
                let g_bonus = g_inc * c.success_base * (c.success_exp - one);
                gw[15] += c.rating.simd_eq(k(2.0)).select(g_bonus, z);
                gw[16] += c.rating.simd_eq(k(4.0)).select(g_bonus, z);
                previous_s += g * (one + c.success_inc) - g_increment * sp(9) * c.inv_s;
                previous_d -= g_increment / (k(11.0) - c.d);
                gr -= g_exp * sp(10);
            }
            let g_mean = (active & c.d_live).select(gd, z);
            gw[7] += g_mean * (k(p.easy_d) - c.next_d);
            gw[4] += g_mean * sp(7);
            gw[5] -= g_mean * sp(7) * k(3.0 * p.exp_3w5_f64 as f32);
            let g_next = g_mean * (one - sp(7));
            let rating_minus_3 = c.rating - k(3.0);
            gw[6] -= g_next * (k(10.0) - c.d) * rating_minus_3 / k(9.0);
            previous_d += g_next * (one + sp(6) * rating_minus_3 / k(9.0));
            gs = previous_s;
            gd = previous_d;
        }
        // R = base^decay, base = 1 + t/s * factor(decay), decay = -w20.
        let gr_r = gr * c.r + c.score_log_r;
        // Match the scalar zero-adjoint early return, including 0 * ln(infinity).
        let curve_active = !gr_r.simd_eq(z);
        let g_base = gr_r * k(p.curve_decay) / c.base;
        let g_base_t_over_s = curve_active.select(g_base * c.t * c.inv_s, z);
        gs -= g_base_t_over_s * k(p.curve_factor) * c.inv_s;
        gw[20] -= curve_active.select(
            gr_r * c.ln_base + g_base_t_over_s * k(p.curve_dfactor_ddecay as f32),
            z,
        );
    }

    gs = live(initial_s, S_MIN, S_MAX).select(gs, z);
    for (i, dst) in gw[..4].iter_mut().enumerate() {
        *dst += rating.simd_eq(k((i + 1) as f32)).select(gs, z);
    }
    gd = live(initial_d, D_MIN, D_MAX).select(gd, z);
    gw[4] += gd;
    gw[5] -= gd * offset * initial_exp;
    gw
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::FSRS6_DEFAULT_PARAMETERS;
    use crate::analytic_v6::batch::Batch;
    use crate::test_helpers::relative_l2_error;
    use rand::{RngExt, SeedableRng, rngs::StdRng};

    #[test]
    fn vector_math_matches_libm_in_the_recurrence_range() {
        for i in 0..10_000 {
            let x = -87.0 + i as f32 * (175.0 / 10_000.0);
            let expected = x.exp();
            let actual = exp(F::splat(x)).to_array()[0];
            assert!(
                (actual / expected - 1.0).abs() < 4e-6,
                "exp({x}): {actual} != {expected}"
            );
            let x = expected;
            let actual = ln(F::splat(x)).to_array()[0];
            let expected = x.ln();
            assert!(
                (actual - expected).abs() < 1e-5 * expected.abs().max(1.0),
                "ln({x}): {actual} != {expected}"
            );
        }
        assert_eq!(exp(F::splat(f32::NEG_INFINITY)), F::ZERO);
        assert_eq!(ln(F::splat(f32::INFINITY)), F::splat(f32::INFINITY));
    }

    #[test]
    fn zero_curve_gradients_do_not_propagate_nonfinite_intermediates() {
        let target_t = [
            0.0,
            1.0,
            36_500.0,
            1e15,
            f32::MAX,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
        ];
        let mut expected = [0.0; PARAMS];
        crate::analytic_v6::batch_loss_and_grad(
            &FSRS6_DEFAULT_PARAMETERS,
            &[0.0; LANES],
            &[1.0; LANES],
            1,
            LANES,
            &[1; LANES],
            &target_t,
            &[1.0; LANES],
            &[1.0; LANES],
            &mut expected,
        );
        let batch = Batch::new(
            &[0.0; LANES],
            &[1.0; LANES],
            &[1; LANES],
            Some(&target_t),
            &[1.0; LANES],
            &[1.0; LANES],
        );
        let actual = batch.gradient(&FSRS6_DEFAULT_PARAMETERS, &mut Workspace::default());
        for (actual, expected) in actual.into_iter().zip(expected) {
            assert!(actual.is_finite());
            assert!((actual as f64 - expected).abs() < 2e-4 * expected.abs().max(1.0));
        }
    }

    #[test]
    fn gradients_match_scalar_for_prefixes_and_cards() {
        check_gradients_match_scalar(false);
    }

    #[test]
    fn gradients_match_scalar_at_parameter_bounds() {
        check_gradients_match_scalar(true);
    }

    fn check_gradients_match_scalar(parameter_bounds: bool) {
        let mut rng = StdRng::seed_from_u64(604);
        for columns in [1, 4, 7, 8, 9, 17] {
            for windowed in [false, true] {
                for trial in 0..if parameter_bounds { 1_000 } else { 16 } {
                    let seq = if parameter_bounds { 128 } else { 32 };
                    let lengths: Vec<_> = (0..columns).map(|_| rng.random_range(2..=seq)).collect();
                    let mut ts = vec![0.0; seq * columns];
                    let mut ratings = vec![0.0; seq * columns];
                    let mut labels = vec![0.0; seq * columns];
                    let mut weights = vec![0.0; seq * columns];
                    for (col, &length) in lengths.iter().enumerate() {
                        for row in 0..length {
                            let i = row * columns + col;
                            ratings[i] = rng.random_range(1..=4) as f32;
                            ts[i] = if row == 0 || rng.random_bool(0.3) {
                                0.0
                            } else {
                                [1.0, 3.0, 30.0, 500.0, 36_500.0][rng.random_range(0..5)]
                            };
                            labels[i] = f32::from(ratings[i] > 1.0);
                            weights[i] = if rng.random_bool(0.2) {
                                0.0
                            } else {
                                rng.random_range(0.05..1.0)
                            };
                        }
                    }
                    let target_t: Vec<_> = (0..columns)
                        .map(|col| if col % 3 == 0 { 0.0 } else { 20.0 })
                        .collect();
                    let mut w = FSRS6_DEFAULT_PARAMETERS;
                    if trial > 0 {
                        if parameter_bounds {
                            let bounds = [
                                (S_MIN, 100.0),
                                (S_MIN, 100.0),
                                (S_MIN, 100.0),
                                (S_MIN, 100.0),
                                (1.0, 10.0),
                                (0.001, 4.0),
                                (0.001, 4.0),
                                (0.001, 0.75),
                                (0.0, 4.5),
                                (0.0, 0.8),
                                (0.001, 3.5),
                                (0.001, 5.0),
                                (0.001, 0.25),
                                (0.001, 0.9),
                                (0.0, 4.0),
                                (0.0, 1.0),
                                (1.0, 6.0),
                                (0.0, 2.0),
                                (0.0, 2.0),
                                (0.01, 0.8),
                                (0.1, 0.8),
                            ];
                            for (v, (low, high)) in w.iter_mut().zip(bounds) {
                                *v = match rng.random_range(0..4) {
                                    0 => low,
                                    1 => high,
                                    _ => rng.random_range(low..=high),
                                };
                            }
                        } else {
                            for v in &mut w {
                                *v *= rng.random_range(0.3..2.0);
                            }
                        }
                        // Include exactly-zero hard and short-term multipliers.
                        if trial % 3 == 0 {
                            w[15] = 0.0;
                            w[17..20].fill(0.0);
                        }
                        crate::parameter_clipper::clip_parameters_in_place(
                            &mut w,
                            1,
                            parameter_bounds,
                        );
                    }
                    let mut expected = [0.0; PARAMS];
                    if windowed {
                        crate::analytic_v6::card_loss_and_grad(
                            &w,
                            &ts,
                            &ratings,
                            seq,
                            columns,
                            &lengths,
                            &labels,
                            &weights,
                            &mut expected,
                        );
                    } else {
                        crate::analytic_v6::batch_loss_and_grad(
                            &w,
                            &ts,
                            &ratings,
                            seq,
                            columns,
                            &lengths,
                            &target_t,
                            &labels[..columns],
                            &weights[..columns],
                            &mut expected,
                        );
                    }
                    let batch = Batch::new(
                        &ts,
                        &ratings,
                        &lengths,
                        (!windowed).then_some(target_t.as_slice()),
                        &labels,
                        &weights,
                    );
                    let actual = batch.gradient(&w, &mut Workspace::default());
                    let error = relative_l2_error(&actual, &expected);
                    assert!(
                        error < 2e-4,
                        "columns={columns} windowed={windowed} trial={trial} parameters={w:?} relative error={error}\nactual={actual:?}\nexpected={expected:?}",
                    );
                }
            }
        }
    }

    #[test]
    fn empty_batch_has_zero_gradient() {
        let batch = Batch::new(&[], &[], &[], None, &[], &[]);
        assert_eq!(
            batch.gradient(&FSRS6_DEFAULT_PARAMETERS, &mut Workspace::default()),
            vec![0.0; PARAMS]
        );
    }
}

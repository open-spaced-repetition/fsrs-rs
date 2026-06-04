#![allow(unsafe_op_in_unsafe_fn)]

use super::{D_MAX, D_MIN, S_MAX, S_MIN, windowed_loss_scalar};
use core::ops::{Add, Div, Mul, Neg, Sub};
use std::arch::aarch64::*;

const LOG2E: f32 = std::f32::consts::LOG2_E;
const LN2: f32 = std::f32::consts::LN_2;

#[derive(Clone, Copy)]
struct F32x4(float32x4_t);

impl F32x4 {
    #[inline(always)]
    fn splat(value: f32) -> Self {
        unsafe { Self(vdupq_n_f32(value)) }
    }

    #[inline(always)]
    fn load(values: &[f32], index: usize) -> Self {
        debug_assert!(index + 4 <= values.len());
        unsafe { Self(vld1q_f32(values.as_ptr().add(index))) }
    }

    #[inline(always)]
    fn sum(self) -> f32 {
        unsafe { vaddvq_f32(self.0) }
    }

    #[inline(always)]
    fn abs(self) -> Self {
        unsafe { Self(vabsq_f32(self.0)) }
    }

    #[inline(always)]
    fn min(self, rhs: Self) -> Self {
        unsafe { Self(vminq_f32(self.0, rhs.0)) }
    }

    #[inline(always)]
    fn max(self, rhs: Self) -> Self {
        unsafe { Self(vmaxq_f32(self.0, rhs.0)) }
    }

    #[inline(always)]
    fn clamp(self, min: f32, max: f32) -> Self {
        self.max(Self::splat(min)).min(Self::splat(max))
    }

    #[inline(always)]
    fn round(self) -> Self {
        unsafe { Self(vrndnq_f32(self.0)) }
    }

    #[inline(always)]
    fn cmp_eq(self, rhs: Self) -> uint32x4_t {
        unsafe { vceqq_f32(self.0, rhs.0) }
    }

    #[inline(always)]
    fn cmp_gt(self, rhs: Self) -> uint32x4_t {
        unsafe { vcgtq_f32(self.0, rhs.0) }
    }

    #[inline(always)]
    fn blend(mask: uint32x4_t, if_true: Self, if_false: Self) -> Self {
        unsafe { Self(vbslq_f32(mask, if_true.0, if_false.0)) }
    }

    #[inline(always)]
    fn exp(self) -> Self {
        let x = self.clamp(-87.0, 88.0);
        let n = (x * Self::splat(LOG2E)).round();
        let r = x - n * Self::splat(LN2);
        let p = Self::splat(0.99992806)
            + r * (Self::splat(1.0001642)
                + r * (Self::splat(0.5049633) + r * Self::splat(0.16566843)));
        let two_n = unsafe {
            let n = vcvtq_s32_f32(n.0);
            let exponent = vshlq_n_s32::<23>(vaddq_s32(n, vdupq_n_s32(127)));
            Self(vreinterpretq_f32_s32(exponent))
        };
        p * two_n
    }

    #[inline(always)]
    fn ln(self) -> Self {
        let bits = unsafe { vreinterpretq_u32_f32(self.0) };
        let exponent = unsafe {
            let raw = vandq_u32(vshrq_n_u32::<23>(bits), vdupq_n_u32(0xff));
            vsubq_s32(vreinterpretq_s32_u32(raw), vdupq_n_s32(127))
        };
        let mantissa = unsafe {
            let masked = vandq_u32(bits, vdupq_n_u32(0x007f_ffff));
            let normalized = vorrq_u32(masked, vdupq_n_u32(127 << 23));
            Self(vreinterpretq_f32_u32(normalized))
        };
        let one = Self::splat(1.0);
        let t = (mantissa - one) / (mantissa + one);
        let t2 = t * t;
        let poly = Self::splat(2.0)
            * t
            * (Self::splat(1.0000073)
                + t2 * (Self::splat(0.33217952) + t2 * Self::splat(0.22657777)));
        let exponent = unsafe { Self(vcvtq_f32_s32(exponent)) };
        exponent * Self::splat(LN2) + poly
    }
}

impl Add for F32x4 {
    type Output = Self;

    #[inline(always)]
    fn add(self, rhs: Self) -> Self::Output {
        unsafe { Self(vaddq_f32(self.0, rhs.0)) }
    }
}

impl Sub for F32x4 {
    type Output = Self;

    #[inline(always)]
    fn sub(self, rhs: Self) -> Self::Output {
        unsafe { Self(vsubq_f32(self.0, rhs.0)) }
    }
}

impl Mul for F32x4 {
    type Output = Self;

    #[inline(always)]
    fn mul(self, rhs: Self) -> Self::Output {
        unsafe { Self(vmulq_f32(self.0, rhs.0)) }
    }
}

impl Div for F32x4 {
    type Output = Self;

    #[inline(always)]
    fn div(self, rhs: Self) -> Self::Output {
        unsafe { Self(vdivq_f32(self.0, rhs.0)) }
    }
}

impl Neg for F32x4 {
    type Output = Self;

    #[inline(always)]
    fn neg(self) -> Self::Output {
        Self::splat(0.0) - self
    }
}

struct Params {
    decay1: f32,
    decay2: f32,
    factor1: f32,
    factor2: f32,
    init_difficulty_easy: f32,
}

impl Params {
    fn new(w: &[f32]) -> Self {
        let decay1 = -w[27];
        let decay2 = -w[28];
        Self {
            decay1,
            decay2,
            factor1: w[29].powf(1.0 / decay1) - 1.0,
            factor2: w[30].powf(1.0 / decay2) - 1.0,
            init_difficulty_easy: w[4] - (w[5] * 3.0).exp() + 1.0,
        }
    }
}

#[inline(always)]
fn init_stability(w: &[f32], rating: F32x4) -> F32x4 {
    let one = F32x4::splat(1.0);
    let two = F32x4::splat(2.0);
    let three = F32x4::splat(3.0);
    let rating = rating.clamp(1.0, 4.0);
    F32x4::blend(
        rating.cmp_eq(one),
        F32x4::splat(w[0]),
        F32x4::blend(
            rating.cmp_eq(two),
            F32x4::splat(w[1]),
            F32x4::blend(rating.cmp_eq(three), F32x4::splat(w[2]), F32x4::splat(w[3])),
        ),
    )
}

#[inline(always)]
fn init_difficulty(w: &[f32], rating: F32x4) -> F32x4 {
    let rating = rating.clamp(1.0, 4.0);
    (F32x4::splat(w[4]) - (F32x4::splat(w[5]) * (rating - F32x4::splat(1.0))).exp()
        + F32x4::splat(1.0))
    .clamp(D_MIN as f32, D_MAX as f32)
}

#[inline(always)]
fn forgetting_curve(w: &[f32], params: &Params, t: F32x4, s: F32x4) -> F32x4 {
    let t_over_s = t.max(F32x4::splat(0.0)) / s;
    let r1 = (F32x4::splat(1.0) + t_over_s * F32x4::splat(params.factor1))
        .ln()
        .mul(F32x4::splat(params.decay1))
        .exp();
    let r2 = (F32x4::splat(1.0) + t_over_s * F32x4::splat(params.factor2))
        .ln()
        .mul(F32x4::splat(params.decay2))
        .exp();
    let ln_s = s.ln();
    let weight1 = F32x4::splat(w[31]) * (ln_s * F32x4::splat(-w[33])).exp();
    let weight2 = F32x4::splat(w[32]) * (ln_s * F32x4::splat(w[34])).exp();
    (weight1 * r1 + weight2 * r2) / (weight1 + weight2)
}

#[inline(always)]
fn stability_for_set(
    w: &[f32],
    s: F32x4,
    r: F32x4,
    d: F32x4,
    rating: F32x4,
    start: usize,
) -> F32x4 {
    let one = F32x4::splat(1.0);
    let hard_penalty = F32x4::blend(
        rating.cmp_eq(F32x4::splat(2.0)),
        F32x4::splat(w[start + 7]),
        one,
    );
    let easy_bonus = F32x4::blend(
        rating.cmp_eq(F32x4::splat(4.0)),
        F32x4::splat(w[start + 8]),
        one,
    );
    let new_s_fail = F32x4::splat(w[start + 3])
        * (d.ln() * F32x4::splat(-w[start + 4])).exp()
        * (((s + one).ln() * F32x4::splat(w[start + 5])).exp() - one)
        * ((one - r) * F32x4::splat(w[start + 6])).exp();
    let pls = s.min(new_s_fail);
    let sinc = (F32x4::splat(w[start] - 1.5)).exp()
        * (F32x4::splat(11.0) - d)
        * (s.ln() * F32x4::splat(-w[start + 1])).exp()
        * (((one - r) * F32x4::splat(w[start + 2])).exp() - one)
        * hard_penalty
        * easy_bonus
        + one;
    let success = s * sinc;
    F32x4::blend(rating.cmp_gt(one), pls.max(success), pls)
}

#[inline(always)]
fn next_difficulty(w: &[f32], params: &Params, d: F32x4, rating: F32x4) -> F32x4 {
    let delta_d = F32x4::splat(-w[6]) * (rating - F32x4::splat(3.0));
    let new_d = d + (F32x4::splat(10.0) - d) * delta_d / F32x4::splat(9.0);
    (F32x4::splat(params.init_difficulty_easy) * F32x4::splat(0.01) + new_d * F32x4::splat(0.99))
        .clamp(D_MIN as f32, D_MAX as f32)
}

#[inline(always)]
fn step(
    w: &[f32],
    params: &Params,
    delta_t: F32x4,
    rating: F32x4,
    s: F32x4,
    d: F32x4,
    nth: usize,
) -> (F32x4, F32x4) {
    let last_s = s.clamp(S_MIN as f32, S_MAX as f32);
    let last_d = d.clamp(D_MIN as f32, D_MAX as f32);
    let (next_s, next_d) = if nth == 0 {
        (
            init_stability(w, rating).clamp(S_MIN as f32, S_MAX as f32),
            init_difficulty(w, rating),
        )
    } else {
        let delta_t = delta_t.max(F32x4::splat(0.0));
        let r = forgetting_curve(w, params, delta_t, last_s);
        let long = stability_for_set(w, last_s, r, last_d, rating, 7);
        let short = stability_for_set(w, last_s, r, last_d, rating, 16);
        let coefficient =
            F32x4::splat(1.0) - F32x4::splat(w[26]) * (F32x4::splat(-w[25]) * delta_t).exp();
        (
            (coefficient * long + (F32x4::splat(1.0) - coefficient) * short)
                .clamp(S_MIN as f32, S_MAX as f32),
            next_difficulty(w, params, last_d, rating),
        )
    };
    let padding = rating.cmp_eq(F32x4::splat(0.0));
    (
        F32x4::blend(padding, last_s, next_s),
        F32x4::blend(padding, last_d, next_d),
    )
}

#[inline(always)]
fn bce_loss(r: F32x4, label: F32x4, weight: F32x4) -> F32x4 {
    let probability = F32x4::splat(1.0) - (label - r).abs();
    -weight * probability.ln()
}

pub(super) fn windowed_loss(
    w: &[f32],
    t_historys: &[f32],
    r_historys: &[f32],
    labels: &[f32],
    weights: &[f32],
    seq_len: usize,
    batch_size: usize,
) -> f64 {
    let group_count = batch_size / 4;
    if group_count == 0 {
        return windowed_loss_scalar(
            w, t_historys, r_historys, labels, weights, seq_len, batch_size,
        );
    }

    let params = Params::new(w);
    let mut loss = 0.0;
    for group in 0..group_count {
        let column = group * 4;
        let mut s = F32x4::splat(0.0);
        let mut d = F32x4::splat(0.0);
        for row in 0..seq_len {
            let index = row * batch_size + column;
            let delta_t = F32x4::load(t_historys, index);
            let rating = F32x4::load(r_historys, index);
            if row > 0 {
                let r = forgetting_curve(w, &params, delta_t, s.clamp(S_MIN as f32, S_MAX as f32))
                    .clamp(0.0001, 0.9999);
                loss += bce_loss(r, F32x4::load(labels, index), F32x4::load(weights, index)).sum()
                    as f64;
            }
            if row + 1 < seq_len {
                (s, d) = step(w, &params, delta_t, rating, s, d, row);
            }
        }
    }

    for column in (group_count * 4)..batch_size {
        let mut state = super::MemoryStateScalar {
            stability: 0.0,
            difficulty: 0.0,
        };
        for row in 0..seq_len {
            let index = row * batch_size + column;
            let delta_t = t_historys[index] as f64;
            let rating = r_historys[index] as usize;
            let weight = weights[index] as f64;
            if row > 0 && weight != 0.0 {
                let r = super::forgetting_curve_scalar(w, delta_t, state.stability);
                loss += super::bce_loss_scalar(r, labels[index] as f64, weight);
            }
            if row + 1 < seq_len {
                state = super::step_scalar(w, delta_t, rating, state, row);
            }
        }
    }

    loss
}

#[cfg(test)]
pub(super) fn windowed_loss_for_test(
    w: &[f32],
    t_historys: &[f32],
    r_historys: &[f32],
    labels: &[f32],
    weights: &[f32],
    seq_len: usize,
    batch_size: usize,
) -> f64 {
    windowed_loss(
        w, t_historys, r_historys, labels, weights, seq_len, batch_size,
    )
}

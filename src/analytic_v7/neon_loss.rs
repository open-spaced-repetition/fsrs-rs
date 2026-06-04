#![allow(unsafe_op_in_unsafe_fn)]

use super::{D_MAX, D_MIN, PARAM_LEN, S_MAX, S_MIN, windowed_loss_scalar};
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
    fn cmp_lt(self, rhs: Self) -> uint32x4_t {
        unsafe { vcltq_f32(self.0, rhs.0) }
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

#[inline(always)]
fn mask_and(lhs: uint32x4_t, rhs: uint32x4_t) -> uint32x4_t {
    unsafe { vandq_u32(lhs, rhs) }
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
    p29: f32,
    p30: f32,
    ln_w29: f32,
    ln_w30: f32,
    init_difficulty_easy: f32,
    exp3w5: f32,
}

impl Params {
    fn new(w: &[f32]) -> Self {
        let decay1 = -w[27];
        let decay2 = -w[28];
        let ln_w29 = w[29].ln();
        let ln_w30 = w[30].ln();
        let p29 = (ln_w29 / decay1).exp();
        let p30 = (ln_w30 / decay2).exp();
        let exp3w5 = (w[5] * 3.0).exp();
        Self {
            decay1,
            decay2,
            factor1: p29 - 1.0,
            factor2: p30 - 1.0,
            p29,
            p30,
            ln_w29,
            ln_w30,
            init_difficulty_easy: w[4] - exp3w5 + 1.0,
            exp3w5,
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

struct Curve4 {
    out: F32x4,
    t_over_s: F32x4,
    b1: F32x4,
    b2: F32x4,
    r1: F32x4,
    r2: F32x4,
    ln_b1: F32x4,
    ln_b2: F32x4,
    p33: F32x4,
    p34: F32x4,
    weight1: F32x4,
    weight2: F32x4,
    wsum: F32x4,
    ret: F32x4,
    ln_s: F32x4,
}

#[inline(always)]
fn curve4_fwd(w: &[f32], params: &Params, t: F32x4, s: F32x4) -> Curve4 {
    let one = F32x4::splat(1.0);
    let t_over_s = t.max(F32x4::splat(0.0)) / s;
    let b1 = one + t_over_s * F32x4::splat(params.factor1);
    let ln_b1 = b1.ln();
    let r1 = (ln_b1 * F32x4::splat(params.decay1)).exp();
    let b2 = one + t_over_s * F32x4::splat(params.factor2);
    let ln_b2 = b2.ln();
    let r2 = (ln_b2 * F32x4::splat(params.decay2)).exp();
    let ln_s = s.ln();
    let p33 = (ln_s * F32x4::splat(-w[33])).exp();
    let p34 = (ln_s * F32x4::splat(w[34])).exp();
    let weight1 = F32x4::splat(w[31]) * p33;
    let weight2 = F32x4::splat(w[32]) * p34;
    let wsum = weight1 + weight2;
    let num = weight1 * r1 + weight2 * r2;
    let ret = num / wsum;
    Curve4 {
        out: ret,
        t_over_s,
        b1,
        b2,
        r1,
        r2,
        ln_b1,
        ln_b2,
        p33,
        p34,
        weight1,
        weight2,
        wsum,
        ret,
        ln_s,
    }
}

#[inline(always)]
fn curve4_bwd(
    w: &[f32],
    params: &Params,
    cache: &Curve4,
    t: F32x4,
    s: F32x4,
    g_out: F32x4,
    grad: &mut [F32x4; PARAM_LEN],
) -> F32x4 {
    let zero = F32x4::splat(0.0);
    let g_num = g_out / cache.wsum;
    let g_wsum = (zero - g_out) * cache.ret / cache.wsum;
    let g_weight1 = g_num * cache.r1 + g_wsum;
    let g_weight2 = g_num * cache.r2 + g_wsum;
    let g_r1 = g_num * cache.weight1;
    let g_r2 = g_num * cache.weight2;

    grad[31] = grad[31] + g_weight1 * cache.p33;
    let g_p33 = g_weight1 * F32x4::splat(w[31]);
    let mut g_s = g_p33 * cache.p33 * F32x4::splat(-w[33]) / s;
    grad[33] = grad[33] + g_p33 * (zero - cache.p33 * cache.ln_s);

    grad[32] = grad[32] + g_weight2 * cache.p34;
    let g_p34 = g_weight2 * F32x4::splat(w[32]);
    g_s = g_s + g_p34 * cache.p34 * F32x4::splat(w[34]) / s;
    grad[34] = grad[34] + g_p34 * cache.p34 * cache.ln_s;

    let g_ln_b1 = g_r1 * cache.r1 * F32x4::splat(params.decay1);
    let mut g_decay1 = g_r1 * cache.r1 * cache.ln_b1;
    let g_b1 = g_ln_b1 / cache.b1;
    let g_t_over_s1 = g_b1 * F32x4::splat(params.factor1);
    let g_factor1 = g_b1 * cache.t_over_s;
    grad[29] = grad[29] + g_factor1 * F32x4::splat(params.p29 / (params.decay1 * w[29]));
    let g_inv1 = g_factor1 * F32x4::splat(params.p29 * params.ln_w29);
    g_decay1 = g_decay1 + g_inv1 * (zero - F32x4::splat(1.0 / (params.decay1 * params.decay1)));
    grad[27] = grad[27] - g_decay1;

    let g_ln_b2 = g_r2 * cache.r2 * F32x4::splat(params.decay2);
    let mut g_decay2 = g_r2 * cache.r2 * cache.ln_b2;
    let g_b2 = g_ln_b2 / cache.b2;
    let g_t_over_s2 = g_b2 * F32x4::splat(params.factor2);
    let g_factor2 = g_b2 * cache.t_over_s;
    grad[30] = grad[30] + g_factor2 * F32x4::splat(params.p30 / (params.decay2 * w[30]));
    let g_inv2 = g_factor2 * F32x4::splat(params.p30 * params.ln_w30);
    g_decay2 = g_decay2 + g_inv2 * (zero - F32x4::splat(1.0 / (params.decay2 * params.decay2)));
    grad[28] = grad[28] - g_decay2;

    let t = t.max(zero);
    g_s + (g_t_over_s1 + g_t_over_s2) * (zero - t / (s * s))
}

struct Stab4 {
    out: F32x4,
    nsf_fail: F32x4,
    pls: F32x4,
    sinc: F32x4,
    success: F32x4,
    aa: F32x4,
    bb: F32x4,
    cc: F32x4,
    expr: F32x4,
    pp: F32x4,
    qbase: F32x4,
    rexp: F32x4,
    hard: F32x4,
    easy: F32x4,
    ln_s: F32x4,
    ln_d: F32x4,
    ln_s1: F32x4,
}

#[inline(always)]
fn stab4_fwd(w: &[f32], s: F32x4, r: F32x4, d: F32x4, rating: F32x4, start: usize) -> Stab4 {
    let one = F32x4::splat(1.0);
    let hard = F32x4::blend(
        rating.cmp_eq(F32x4::splat(2.0)),
        F32x4::splat(w[start + 7]),
        one,
    );
    let easy = F32x4::blend(
        rating.cmp_eq(F32x4::splat(4.0)),
        F32x4::splat(w[start + 8]),
        one,
    );
    let ln_s = s.ln();
    let ln_d = d.ln();
    let pp = (ln_d * F32x4::splat(-w[start + 4])).exp();
    let ln_s1 = (s + one).ln();
    let qbase = (ln_s1 * F32x4::splat(w[start + 5])).exp();
    let rexp = ((one - r) * F32x4::splat(w[start + 6])).exp();
    let nsf_fail = F32x4::splat(w[start + 3]) * pp * (qbase - one) * rexp;
    let pls = s.min(nsf_fail);
    let aa = F32x4::splat(w[start] - 1.5).exp();
    let bb = F32x4::splat(11.0) - d;
    let cc = (ln_s * F32x4::splat(-w[start + 1])).exp();
    let expr = ((one - r) * F32x4::splat(w[start + 2])).exp();
    let sinc = aa * bb * cc * (expr - one) * hard * easy + one;
    let success = s * sinc;
    let out = F32x4::blend(rating.cmp_gt(one), pls.max(success), pls);
    Stab4 {
        out,
        nsf_fail,
        pls,
        sinc,
        success,
        aa,
        bb,
        cc,
        expr,
        pp,
        qbase,
        rexp,
        hard,
        easy,
        ln_s,
        ln_d,
        ln_s1,
    }
}

#[inline(always)]
fn stab4_bwd(
    w: &[f32],
    cache: &Stab4,
    s: F32x4,
    r: F32x4,
    d: F32x4,
    rating: F32x4,
    start: usize,
    g_out: F32x4,
    grad: &mut [F32x4; PARAM_LEN],
) -> (F32x4, F32x4, F32x4) {
    let zero = F32x4::splat(0.0);
    let one = F32x4::splat(1.0);
    let gt1 = rating.cmp_gt(one);
    let g_nss = F32x4::blend(gt1, g_out, zero);
    let g_pls_direct = F32x4::blend(gt1, zero, g_out);
    let pls_wins = cache.pls.cmp_gt(cache.success);
    let g_pls_from_nss = F32x4::blend(pls_wins, g_nss, zero);
    let g_success = F32x4::blend(pls_wins, zero, g_nss);
    let mut g_s = g_success * cache.sinc;
    let g_sinc = g_success * s;
    let g_pls = g_pls_direct + g_pls_from_nss;
    let s_wins = s.cmp_lt(cache.nsf_fail);
    g_s = g_s + F32x4::blend(s_wins, g_pls, zero);
    let g_nsf_fail = F32x4::blend(s_wins, zero, g_pls);

    let em1 = cache.expr - one;
    let g_prod = g_sinc;
    let prod = cache.aa * cache.bb * cache.cc * em1 * cache.hard * cache.easy;
    grad[start] = grad[start] + g_prod * prod;
    let g_bb = g_prod * (cache.aa * cache.cc * em1 * cache.hard * cache.easy);
    let g_cc = g_prod * (cache.aa * cache.bb * em1 * cache.hard * cache.easy);
    let g_em1 = g_prod * (cache.aa * cache.bb * cache.cc * cache.hard * cache.easy);
    grad[start + 7] = grad[start + 7]
        + F32x4::blend(
            rating.cmp_eq(F32x4::splat(2.0)),
            g_prod * (cache.aa * cache.bb * cache.cc * em1 * cache.easy),
            zero,
        );
    grad[start + 8] = grad[start + 8]
        + F32x4::blend(
            rating.cmp_eq(F32x4::splat(4.0)),
            g_prod * (cache.aa * cache.bb * cache.cc * em1 * cache.hard),
            zero,
        );

    let mut g_d = zero - g_bb;
    g_s = g_s + g_cc * F32x4::splat(-w[start + 1]) * cache.cc / s;
    grad[start + 1] = grad[start + 1] + g_cc * (zero - cache.cc * cache.ln_s);
    let mut g_r = g_em1 * cache.expr * F32x4::splat(-w[start + 2]);
    grad[start + 2] = grad[start + 2] + g_em1 * cache.expr * (one - r);

    let q = cache.qbase - one;
    grad[start + 3] = grad[start + 3] + g_nsf_fail * cache.pp * q * cache.rexp;
    let g_pp = g_nsf_fail * F32x4::splat(w[start + 3]) * q * cache.rexp;
    let g_q = g_nsf_fail * F32x4::splat(w[start + 3]) * cache.pp * cache.rexp;
    let g_rexp = g_nsf_fail * F32x4::splat(w[start + 3]) * cache.pp * q;
    g_d = g_d + g_pp * F32x4::splat(-w[start + 4]) * cache.pp / d;
    grad[start + 4] = grad[start + 4] + g_pp * (zero - cache.pp * cache.ln_d);
    g_s = g_s + g_q * F32x4::splat(w[start + 5]) * cache.qbase / (s + one);
    grad[start + 5] = grad[start + 5] + g_q * cache.qbase * cache.ln_s1;
    g_r = g_r + g_rexp * cache.rexp * F32x4::splat(-w[start + 6]);
    grad[start + 6] = grad[start + 6] + g_rexp * cache.rexp * (one - r);

    (g_s, g_d, g_r)
}

#[inline(always)]
fn next_difficulty4_fwd(
    w: &[f32],
    params: &Params,
    d: F32x4,
    rating: F32x4,
) -> (F32x4, F32x4, F32x4) {
    let delta_d = F32x4::splat(-w[6]) * (rating - F32x4::splat(3.0));
    let new_d = d + (F32x4::splat(10.0) - d) * delta_d / F32x4::splat(9.0);
    let out_pre =
        F32x4::splat(params.init_difficulty_easy) * F32x4::splat(0.01) + new_d * F32x4::splat(0.99);
    (out_pre.clamp(D_MIN as f32, D_MAX as f32), out_pre, delta_d)
}

#[inline(always)]
fn next_difficulty4_bwd(
    params: &Params,
    d: F32x4,
    rating: F32x4,
    out_pre: F32x4,
    delta_d: F32x4,
    g_out: F32x4,
    grad: &mut [F32x4; PARAM_LEN],
) -> F32x4 {
    let zero = F32x4::splat(0.0);
    let open = mask_and(
        out_pre.cmp_gt(F32x4::splat(D_MIN as f32)),
        out_pre.cmp_lt(F32x4::splat(D_MAX as f32)),
    );
    let g_out_pre = F32x4::blend(open, g_out, zero);
    let g_init = g_out_pre * F32x4::splat(0.01);
    let g_new_d = g_out_pre * F32x4::splat(0.99);
    grad[4] = grad[4] + g_init;
    grad[5] = grad[5] + g_init * F32x4::splat(-params.exp3w5 * 3.0);
    let g_d = g_new_d * (F32x4::splat(1.0) - delta_d / F32x4::splat(9.0));
    let g_delta_d = g_new_d * (F32x4::splat(10.0) - d) / F32x4::splat(9.0);
    grad[6] = grad[6] + g_delta_d * (zero - (rating - F32x4::splat(3.0)));
    g_d
}

enum Step4 {
    First {
        s0: F32x4,
        d0: F32x4,
        last_s: F32x4,
        last_d: F32x4,
        rating: F32x4,
        rc: F32x4,
        init_s_pre: F32x4,
        init_d_pre: F32x4,
        init_d_exp: F32x4,
    },
    Full {
        s0: F32x4,
        d0: F32x4,
        last_s: F32x4,
        last_d: F32x4,
        rating: F32x4,
        dt: F32x4,
        curve: Curve4,
        long: Stab4,
        short: Stab4,
        coefficient: F32x4,
        transition_exp: F32x4,
        new_s_pre: F32x4,
        nd_out_pre: F32x4,
        nd_delta_d: F32x4,
    },
}

impl Step4 {
    #[inline(always)]
    fn curve_out(&self) -> F32x4 {
        match self {
            Self::Full { curve, .. } => curve.out,
            Self::First { .. } => F32x4::splat(0.0),
        }
    }
}

#[inline(always)]
fn step4_fwd(
    w: &[f32],
    params: &Params,
    delta_t: F32x4,
    rating: F32x4,
    s: F32x4,
    d: F32x4,
    nth: usize,
) -> ((F32x4, F32x4), Step4) {
    let last_s = s.clamp(S_MIN as f32, S_MAX as f32);
    let last_d = d.clamp(D_MIN as f32, D_MAX as f32);
    if nth == 0 {
        let one = F32x4::splat(1.0);
        let rc = rating.clamp(1.0, 4.0);
        let init_s_pre = init_stability(w, rating);
        let init_d_exp = (F32x4::splat(w[5]) * (rc - one)).exp();
        let init_d_pre = F32x4::splat(w[4]) - init_d_exp + one;
        let next_s = init_s_pre.clamp(S_MIN as f32, S_MAX as f32);
        let next_d = init_d_pre.clamp(D_MIN as f32, D_MAX as f32);
        let padding = rating.cmp_eq(F32x4::splat(0.0));
        return (
            (
                F32x4::blend(padding, last_s, next_s),
                F32x4::blend(padding, last_d, next_d),
            ),
            Step4::First {
                s0: s,
                d0: d,
                last_s,
                last_d,
                rating,
                rc,
                init_s_pre,
                init_d_pre,
                init_d_exp,
            },
        );
    }

    let dt = delta_t.max(F32x4::splat(0.0));
    let curve = curve4_fwd(w, params, dt, last_s);
    let long = stab4_fwd(w, last_s, curve.out, last_d, rating, 7);
    let short = stab4_fwd(w, last_s, curve.out, last_d, rating, 16);
    let transition_exp = (F32x4::splat(-w[25]) * dt).exp();
    let coefficient = F32x4::splat(1.0) - F32x4::splat(w[26]) * transition_exp;
    let new_s_pre = coefficient * long.out + (F32x4::splat(1.0) - coefficient) * short.out;
    let (next_d, nd_out_pre, nd_delta_d) = next_difficulty4_fwd(w, params, last_d, rating);
    let next_s = new_s_pre.clamp(S_MIN as f32, S_MAX as f32);
    let padding = rating.cmp_eq(F32x4::splat(0.0));
    (
        (
            F32x4::blend(padding, last_s, next_s),
            F32x4::blend(padding, last_d, next_d),
        ),
        Step4::Full {
            s0: s,
            d0: d,
            last_s,
            last_d,
            rating,
            dt,
            curve,
            long,
            short,
            coefficient,
            transition_exp,
            new_s_pre,
            nd_out_pre,
            nd_delta_d,
        },
    )
}

#[inline(always)]
fn step4_bwd(
    w: &[f32],
    params: &Params,
    cache: &Step4,
    g_out: (F32x4, F32x4),
    g_r_loss: F32x4,
    grad: &mut [F32x4; PARAM_LEN],
) -> (F32x4, F32x4) {
    let zero = F32x4::splat(0.0);
    match cache {
        Step4::First {
            s0,
            d0,
            last_s,
            last_d,
            rating,
            rc,
            init_s_pre,
            init_d_pre,
            init_d_exp,
        } => {
            let active = rating.cmp_gt(zero);
            let mut g_last_s = F32x4::blend(active, zero, g_out.0);
            let mut g_last_d = F32x4::blend(active, zero, g_out.1);
            let init_s_open = mask_and(
                init_s_pre.cmp_gt(F32x4::splat(S_MIN as f32)),
                init_s_pre.cmp_lt(F32x4::splat(S_MAX as f32)),
            );
            let init_d_open = mask_and(
                init_d_pre.cmp_gt(F32x4::splat(D_MIN as f32)),
                init_d_pre.cmp_lt(F32x4::splat(D_MAX as f32)),
            );
            let g_init_s = F32x4::blend(mask_and(active, init_s_open), g_out.0, zero);
            let g_init_d = F32x4::blend(mask_and(active, init_d_open), g_out.1, zero);
            for rating_value in 1..=4 {
                grad[rating_value - 1] = grad[rating_value - 1]
                    + F32x4::blend(
                        rating.cmp_eq(F32x4::splat(rating_value as f32)),
                        g_init_s,
                        zero,
                    );
            }
            grad[4] = grad[4] + g_init_d;
            grad[5] = grad[5] + g_init_d * (zero - *init_d_exp * (*rc - F32x4::splat(1.0)));
            let s_open = mask_and(
                s0.cmp_gt(F32x4::splat(S_MIN as f32)),
                s0.cmp_lt(F32x4::splat(S_MAX as f32)),
            );
            let d_open = mask_and(
                d0.cmp_gt(F32x4::splat(D_MIN as f32)),
                d0.cmp_lt(F32x4::splat(D_MAX as f32)),
            );
            g_last_s = F32x4::blend(s_open, g_last_s, zero);
            g_last_d = F32x4::blend(d_open, g_last_d, zero);
            let _ = (last_s, last_d, g_r_loss);
            (g_last_s, g_last_d)
        }
        Step4::Full {
            s0,
            d0,
            last_s,
            last_d,
            rating,
            dt,
            curve,
            long,
            short,
            coefficient,
            transition_exp,
            new_s_pre,
            nd_out_pre,
            nd_delta_d,
        } => {
            let active = rating.cmp_gt(zero);
            let mut g_last_s = F32x4::blend(active, zero, g_out.0);
            let mut g_last_d = F32x4::blend(active, zero, g_out.1);
            let g_new_s = F32x4::blend(active, g_out.0, zero);
            let g_new_d = F32x4::blend(active, g_out.1, zero);
            let new_s_open = mask_and(
                new_s_pre.cmp_gt(F32x4::splat(S_MIN as f32)),
                new_s_pre.cmp_lt(F32x4::splat(S_MAX as f32)),
            );
            let g_new_s_pre = F32x4::blend(new_s_open, g_new_s, zero);
            let g_coefficient = g_new_s_pre * (long.out - short.out);
            let g_long = g_new_s_pre * *coefficient;
            let g_short = g_new_s_pre * (F32x4::splat(1.0) - *coefficient);
            grad[26] = grad[26] + g_coefficient * (zero - *transition_exp);
            let g_transition_exp = g_coefficient * F32x4::splat(-w[26]);
            grad[25] = grad[25] + g_transition_exp * *transition_exp * (zero - *dt);

            let (g_s_long, g_d_long, g_r_long) = stab4_bwd(
                w, long, *last_s, curve.out, *last_d, *rating, 7, g_long, grad,
            );
            let (g_s_short, g_d_short, g_r_short) = stab4_bwd(
                w, short, *last_s, curve.out, *last_d, *rating, 16, g_short, grad,
            );
            let g_d_next = next_difficulty4_bwd(
                params,
                *last_d,
                *rating,
                *nd_out_pre,
                *nd_delta_d,
                g_new_d,
                grad,
            );
            let g_s_curve = curve4_bwd(
                w,
                params,
                curve,
                *dt,
                *last_s,
                g_r_long + g_r_short + g_r_loss,
                grad,
            );
            g_last_s = g_last_s + g_s_long + g_s_short + g_s_curve;
            g_last_d = g_last_d + g_d_long + g_d_short + g_d_next;
            let s_open = mask_and(
                s0.cmp_gt(F32x4::splat(S_MIN as f32)),
                s0.cmp_lt(F32x4::splat(S_MAX as f32)),
            );
            let d_open = mask_and(
                d0.cmp_gt(F32x4::splat(D_MIN as f32)),
                d0.cmp_lt(F32x4::splat(D_MAX as f32)),
            );
            (
                F32x4::blend(s_open, g_last_s, zero),
                F32x4::blend(d_open, g_last_d, zero),
            )
        }
    }
}

#[inline(always)]
fn bce_retrievability_grad(r_raw: F32x4, label: F32x4, weight: F32x4) -> F32x4 {
    let zero = F32x4::splat(0.0);
    let one = F32x4::splat(1.0);
    let r = r_raw.clamp(0.0001, 0.9999);
    let label_is_one = label.cmp_gt(F32x4::splat(0.5));
    let grad = F32x4::blend(label_is_one, zero - weight / r, weight / (one - r));
    let open = mask_and(
        r_raw.cmp_gt(F32x4::splat(0.0001)),
        r_raw.cmp_lt(F32x4::splat(0.9999)),
    );
    F32x4::blend(open, grad, zero)
}

fn add_scalar_column_grad(
    w: &[f32],
    t_historys: &[f32],
    r_historys: &[f32],
    labels: &[f32],
    weights: &[f32],
    seq_len: usize,
    batch_size: usize,
    column: usize,
    grad: &mut [f64; PARAM_LEN],
) {
    let w = super::dual_params(w);
    let mut state = super::MemoryStateDual {
        stability: super::Dual35::constant(0.0),
        difficulty: super::Dual35::constant(0.0),
    };
    for row in 0..seq_len {
        let index = row * batch_size + column;
        let delta_t = t_historys[index] as f64;
        let rating = r_historys[index] as usize;
        let weight = weights[index] as f64;
        if row > 0 && weight != 0.0 {
            let r = super::forgetting_curve(&w, delta_t, state.stability);
            let item_loss = super::bce_loss(r, labels[index] as f64, weight);
            for (dst, src) in grad.iter_mut().zip(item_loss.grad) {
                *dst += src;
            }
        }
        if row + 1 < seq_len {
            state = super::step(&w, delta_t, rating, state, row);
        }
    }
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

pub(super) fn windowed_grad(
    w: &[f32],
    t_historys: &[f32],
    r_historys: &[f32],
    labels: &[f32],
    weights: &[f32],
    seq_len: usize,
    batch_size: usize,
) -> [f32; PARAM_LEN] {
    let params = Params::new(w);
    let mut grad = [0.0f64; PARAM_LEN];
    let group_count = batch_size / 4;

    if seq_len >= 2 {
        let mut caches = Vec::with_capacity(seq_len.saturating_sub(1));
        for group in 0..group_count {
            let column = group * 4;
            caches.clear();
            let mut s = F32x4::splat(0.0);
            let mut d = F32x4::splat(0.0);
            for row in 0..seq_len - 1 {
                let index = row * batch_size + column;
                let (next_state, cache) = step4_fwd(
                    w,
                    &params,
                    F32x4::load(t_historys, index),
                    F32x4::load(r_historys, index),
                    s,
                    d,
                    row,
                );
                s = next_state.0;
                d = next_state.1;
                caches.push(cache);
            }

            let last_index = (seq_len - 1) * batch_size + column;
            let last_s = s.clamp(S_MIN as f32, S_MAX as f32);
            let last_curve = curve4_fwd(w, &params, F32x4::load(t_historys, last_index), last_s);
            let mut group_grad = [F32x4::splat(0.0); PARAM_LEN];
            let g_last_r = bce_retrievability_grad(
                last_curve.out,
                F32x4::load(labels, last_index),
                F32x4::load(weights, last_index),
            );
            let g_last_s = curve4_bwd(
                w,
                &params,
                &last_curve,
                F32x4::load(t_historys, last_index),
                last_s,
                g_last_r,
                &mut group_grad,
            );
            let s_open = mask_and(
                s.cmp_gt(F32x4::splat(S_MIN as f32)),
                s.cmp_lt(F32x4::splat(S_MAX as f32)),
            );
            let mut g_s = F32x4::blend(s_open, g_last_s, F32x4::splat(0.0));
            let mut g_d = F32x4::splat(0.0);

            for row in (0..seq_len - 1).rev() {
                let g_r_loss = if row == 0 {
                    F32x4::splat(0.0)
                } else {
                    let index = row * batch_size + column;
                    bce_retrievability_grad(
                        caches[row].curve_out(),
                        F32x4::load(labels, index),
                        F32x4::load(weights, index),
                    )
                };
                let previous = step4_bwd(
                    w,
                    &params,
                    &caches[row],
                    (g_s, g_d),
                    g_r_loss,
                    &mut group_grad,
                );
                g_s = previous.0;
                g_d = previous.1;
            }

            for (dst, src) in grad.iter_mut().zip(group_grad) {
                *dst += src.sum() as f64;
            }
        }
    }

    for column in (group_count * 4)..batch_size {
        add_scalar_column_grad(
            w, t_historys, r_historys, labels, weights, seq_len, batch_size, column, &mut grad,
        );
    }

    let mut grad_f32 = [0.0; PARAM_LEN];
    for (dst, src) in grad_f32.iter_mut().zip(grad) {
        *dst = src as f32;
    }
    grad_f32
}

#[cfg(test)]
pub(super) fn windowed_grad_for_test(
    w: &[f32],
    t_historys: &[f32],
    r_historys: &[f32],
    labels: &[f32],
    weights: &[f32],
    seq_len: usize,
    batch_size: usize,
) -> [f32; PARAM_LEN] {
    windowed_grad(
        w, t_historys, r_historys, labels, weights, seq_len, batch_size,
    )
}

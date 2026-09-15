#![allow(unsafe_op_in_unsafe_fn)]

// ARM64 NEON reverse-mode (BPTT) gradient for the dual-trace FSRS-7 model.
//
// This is a lane-parallel (4 cards at a time) mirror of the scalar reverse-mode
// in `super::reverse`, which is itself validated to ~1e-9 against the
// forward-mode dual (and the dual against the former autodiff path). Remainder columns
// that don't fill a group of 4 fall back to the scalar reverse path.

use super::{D_MAX, D_MIN, PARAM_LEN, S_MAX, S_MIN};
use crate::neon_math::F32x4;
use std::arch::aarch64::{uint32x4_t, vandq_u32};

#[inline(always)]
fn mask_and(a: uint32x4_t, b: uint32x4_t) -> uint32x4_t {
    unsafe { vandq_u32(a, b) }
}

#[inline(always)]
fn open_mask(x: F32x4, lo: f32, hi: f32) -> uint32x4_t {
    mask_and(x.cmp_gt(F32x4::splat(lo)), x.cmp_lt(F32x4::splat(hi)))
}

#[inline(always)]
fn splat(v: f32) -> F32x4 {
    F32x4::splat(v)
}

struct Params {
    lnw25: f32,
    dec2: f32,
    inv_dec2: f32,
    lnw26: f32,
    p26: f32,
    factor2: f32,
    open2: bool,
    exp3w5: f32,
    prefac_slow: f32,
    prefac_fast: f32,
}

impl Params {
    fn new(w: &[f32]) -> Self {
        let dec2_mag = (w[24]).clamp(0.01, 0.95);
        let open2 = w[24] > 0.01 && w[24] < 0.95;
        let dec2 = -dec2_mag;
        let inv_dec2 = 1.0 / dec2;
        let lnw26 = w[26].ln();
        let p26 = (lnw26 * inv_dec2).exp();
        Self {
            lnw25: w[25].ln(),
            dec2,
            inv_dec2,
            lnw26,
            p26,
            factor2: p26 - 1.0,
            open2,
            exp3w5: (w[5] * 3.0).exp(),
            prefac_slow: (w[7] - 1.5).exp(),
            prefac_fast: (w[15] - 1.5).exp(),
        }
    }

    #[inline(always)]
    fn prefac(&self, start: usize) -> f32 {
        if start == 7 {
            self.prefac_slow
        } else {
            self.prefac_fast
        }
    }
}

#[inline(always)]
fn init_stability(w: &[f32], rating: F32x4) -> F32x4 {
    let rating = rating.clamp(1.0, 4.0);
    F32x4::blend(
        rating.cmp_eq(splat(1.0)),
        splat(w[0]),
        F32x4::blend(
            rating.cmp_eq(splat(2.0)),
            splat(w[1]),
            F32x4::blend(rating.cmp_eq(splat(3.0)), splat(w[2]), splat(w[3])),
        ),
    )
}

// --- curve (dual-trace retrievability, or fast-only recall) ---

struct Curve {
    out: F32x4,
    fast_only: bool,
    s: F32x4,
    sf: F32x4,
    d: F32x4,
    t: F32x4,
    open_s: uint32x4_t,
    open_sf: uint32x4_t,
    open_d: uint32x4_t,
    lnsf: F32x4,
    sf_pow: F32x4,
    open1: uint32x4_t,
    dec1: F32x4,
    openq: uint32x4_t,
    e1: F32x4,
    factor1: F32x4,
    tos_f: F32x4,
    b1: F32x4,
    lnb1: F32x4,
    r1: F32x4,
    // slow-only fields
    d_ts: F32x4,
    tos: F32x4,
    b2: F32x4,
    lnb2: F32x4,
    r2: F32x4,
    p29: F32x4,
    weight1: F32x4,
    lns: F32x4,
    p30: F32x4,
    d_wexp: F32x4,
    weight2: F32x4,
    wsum: F32x4,
    ret: F32x4,
}

#[inline(always)]
fn curve_fwd(
    w: &[f32],
    params: &Params,
    t_in: F32x4,
    s_in: F32x4,
    sf_in: F32x4,
    d_in: F32x4,
    fast_only: bool,
) -> Curve {
    let one = splat(1.0);
    let t = t_in.max(splat(0.0));
    let s = s_in.clamp(S_MIN as f32, S_MAX as f32);
    let sf = sf_in.clamp(S_MIN as f32, S_MAX as f32);
    let d = d_in.clamp(D_MIN as f32, D_MAX as f32);
    let open_s = open_mask(s_in, S_MIN as f32, S_MAX as f32);
    let open_sf = open_mask(sf_in, S_MIN as f32, S_MAX as f32);
    let open_d = open_mask(d_in, D_MIN as f32, D_MAX as f32);

    // fast component
    let lnsf = sf.ln();
    let sf_pow = ((splat(w[33]) - splat(0.3)) * lnsf).exp();
    let dec1_raw = splat(w[23]) * sf_pow;
    let dec1_mag = dec1_raw.clamp(0.01, 0.95);
    let open1 = open_mask(dec1_raw, 0.01, 0.95);
    let dec1 = splat(0.0) - dec1_mag;
    let q1 = splat(params.lnw25) / dec1;
    let openq = q1.cmp_lt(splat(60.0));
    let q1c = q1.min(splat(60.0));
    let e1 = q1c.exp();
    let factor1 = e1 - one;
    let tos_f = t / sf;
    let b1 = tos_f * factor1 + one;
    let lnb1 = b1.ln();
    let r1 = (dec1 * lnb1).exp();

    if fast_only {
        return Curve {
            out: r1,
            fast_only: true,
            s,
            sf,
            d,
            t,
            open_s,
            open_sf,
            open_d,
            lnsf,
            sf_pow,
            open1,
            dec1,
            openq,
            e1,
            factor1,
            tos_f,
            b1,
            lnb1,
            r1,
            d_ts: splat(0.0),
            tos: splat(0.0),
            b2: splat(0.0),
            lnb2: splat(0.0),
            r2: splat(0.0),
            p29: splat(0.0),
            weight1: splat(0.0),
            lns: splat(0.0),
            p30: splat(0.0),
            d_wexp: splat(0.0),
            weight2: splat(0.0),
            wsum: splat(1.0),
            ret: splat(0.0),
        };
    }

    // slow component
    let dec2 = splat(params.dec2);
    let factor2 = splat(params.factor2);
    let d_ts = ((d - splat(5.0)) * splat(w[32] - 0.3)).exp();
    let tos = t / s;
    let b2 = tos * factor2 * d_ts + one;
    let lnb2 = b2.ln();
    let r2 = (dec2 * lnb2).exp();

    let p29 = (splat(-w[29]) * lnsf).exp();
    let weight1 = splat(w[27]) * p29;
    let lns = s.ln();
    let p30 = (splat(w[30]) * lns).exp();
    let d_wexp = ((d - splat(5.0)) * splat(w[31] - 0.5)).exp();
    let weight2 = splat(w[28]) * p30 * d_wexp;
    let wsum = weight1 + weight2;
    let num = weight1 * r1 + weight2 * r2;
    let ret = num / wsum;
    let out = ret * splat(1.0 - 2e-5) + splat(1e-5);

    Curve {
        out,
        fast_only: false,
        s,
        sf,
        d,
        t,
        open_s,
        open_sf,
        open_d,
        lnsf,
        sf_pow,
        open1,
        dec1,
        openq,
        e1,
        factor1,
        tos_f,
        b1,
        lnb1,
        r1,
        d_ts,
        tos,
        b2,
        lnb2,
        r2,
        p29,
        weight1,
        lns,
        p30,
        d_wexp,
        weight2,
        wsum,
        ret,
    }
}

/// Returns (g_s, g_sf, g_d) routed through the input clamps.
#[inline(always)]
fn curve_bwd(
    w: &[f32],
    params: &Params,
    c: &Curve,
    g_out: F32x4,
    g_r1_extra: F32x4,
    grad: &mut [F32x4; PARAM_LEN],
) -> (F32x4, F32x4, F32x4) {
    let zero = splat(0.0);
    let mut g_s = zero;
    let mut g_sf = zero;
    let mut g_d = zero;
    let mut g_lnsf = zero;

    let g_r1 = if c.fast_only {
        g_out + g_r1_extra
    } else {
        let g_ret = g_out * splat(1.0 - 2e-5);
        let g_num = g_ret / c.wsum;
        let g_wsum = (zero - g_ret) * c.ret / c.wsum;
        let g_weight1 = g_num * c.r1 + g_wsum;
        let g_weight2 = g_num * c.r2 + g_wsum;
        let g_r1 = g_num * c.weight1 + g_r1_extra;
        let g_r2 = g_num * c.weight2;

        // weight2 = w28 * p30 * d_wexp
        grad[28] = grad[28] + g_weight2 * c.p30 * c.d_wexp;
        let g_p30 = g_weight2 * splat(w[28]) * c.d_wexp;
        let g_d_wexp = g_weight2 * splat(w[28]) * c.p30;
        grad[30] = grad[30] + g_p30 * c.p30 * c.lns;
        g_s = g_s + g_p30 * c.p30 * splat(w[30]) / c.s;
        let g_dw_arg = g_d_wexp * c.d_wexp;
        grad[31] = grad[31] + g_dw_arg * (c.d - splat(5.0));
        g_d = g_d + g_dw_arg * splat(w[31] - 0.5);

        // weight1 = w27 * p29
        grad[27] = grad[27] + g_weight1 * c.p29;
        let g_p29 = g_weight1 * splat(w[27]);
        grad[29] = grad[29] + g_p29 * c.p29 * (zero - c.lnsf);
        g_lnsf = g_lnsf + g_p29 * c.p29 * splat(-w[29]);

        // r2 = exp(dec2*lnb2)
        let g_lnb2 = g_r2 * c.r2 * splat(params.dec2);
        let g_b2 = g_lnb2 / c.b2;
        let g_tos = g_b2 * splat(params.factor2) * c.d_ts;
        let g_factor2 = g_b2 * c.tos * c.d_ts;
        let g_d_ts = g_b2 * c.tos * splat(params.factor2);
        let g_dts_arg = g_d_ts * c.d_ts;
        grad[32] = grad[32] + g_dts_arg * (c.d - splat(5.0));
        g_d = g_d + g_dts_arg * splat(w[32] - 0.3);
        g_s = g_s + g_tos * (zero - c.t / (c.s * c.s));
        // factor2 = p26 - 1 ; p26 = exp(lnw26*inv_dec2) ; both dec2 & the p26 chain feed w24, w26
        let mut g_dec2 = g_r2 * c.r2 * c.lnb2;
        let g_p26 = g_factor2;
        grad[26] = grad[26] + g_p26 * splat(params.p26 * params.inv_dec2 / w[26]);
        g_dec2 = g_dec2
            + g_p26 * splat(params.p26 * params.lnw26 * (-1.0 / (params.dec2 * params.dec2)));
        if params.open2 {
            grad[24] = grad[24] + (zero - g_dec2);
        }
        g_r1
    };

    // r1 = exp(dec1*lnb1)
    let mut g_dec1 = g_r1 * c.r1 * c.lnb1;
    let g_lnb1 = g_r1 * c.r1 * c.dec1;
    let g_b1 = g_lnb1 / c.b1;
    let g_tos_f = g_b1 * c.factor1;
    let g_factor1 = g_b1 * c.tos_f;
    g_sf = g_sf + g_tos_f * (zero - c.t / (c.sf * c.sf));
    // factor1 = e1 - 1 ; e1 = exp(q1c) ; q1c = min(q1,60) ; q1 = lnw25/dec1
    let g_e1 = g_factor1;
    let g_q1c = g_e1 * c.e1;
    let g_q1 = F32x4::blend(c.openq, g_q1c, zero);
    grad[25] = grad[25] + g_q1 * (splat(1.0) / c.dec1) / splat(w[25]);
    g_dec1 = g_dec1 + g_q1 * (zero - splat(params.lnw25) / (c.dec1 * c.dec1));
    // dec1 = -clamp(w23*sf_pow)
    let g_dec1_mag = zero - g_dec1;
    let g_dec1_raw = F32x4::blend(c.open1, g_dec1_mag, zero);
    grad[23] = grad[23] + g_dec1_raw * c.sf_pow;
    let g_sf_pow = g_dec1_raw * splat(w[23]);
    grad[33] = grad[33] + g_sf_pow * c.sf_pow * c.lnsf;
    g_lnsf = g_lnsf + g_sf_pow * c.sf_pow * splat(w[33] - 0.3);
    g_sf = g_sf + g_lnsf / c.sf;

    (
        F32x4::blend(c.open_s, g_s, zero),
        F32x4::blend(c.open_sf, g_sf, zero),
        F32x4::blend(c.open_d, g_d, zero),
    )
}

// --- stability_for_set ---

struct Stab {
    start: usize,
    last_s: F32x4,
    last_d: F32x4,
    r: F32x4,
    rating: F32x4,
    hard: F32x4,
    easy: F32x4,
    lns1: F32x4,
    q: F32x4,
    er: F32x4,
    new_s_fail: F32x4,
    pls: F32x4,
    lns: F32x4,
    cc: F32x4,
    bb: F32x4,
    er2: F32x4,
    em1: F32x4,
    prefac: f32,
    sinc: F32x4,
    success: F32x4,
    out: F32x4,
}

#[inline(always)]
fn stab_fwd(
    w: &[f32],
    params: &Params,
    last_s: F32x4,
    last_d: F32x4,
    r: F32x4,
    rating: F32x4,
    start: usize,
) -> Stab {
    let one = splat(1.0);
    let hard = F32x4::blend(rating.cmp_eq(splat(2.0)), splat(w[start + 6]), one);
    let easy = F32x4::blend(rating.cmp_eq(splat(4.0)), splat(w[start + 7]), one);
    let lns1 = (last_s + one).ln();
    let q = (splat(w[start + 4]) * lns1).exp();
    let er = ((one - r) * splat(w[start + 5])).exp();
    let new_s_fail = splat(w[start + 3]) * (q - one) * er;
    let pls = last_s.min(new_s_fail);
    // success branch (computed for all lanes; blended by rating>1 at the end)
    let lns = last_s.ln();
    let cc = (splat(-w[start + 1]) * lns).exp();
    let bb = splat(11.0) - last_d;
    let er2 = ((one - r) * splat(w[start + 2])).exp();
    let em1 = er2 - one;
    let prefac = params.prefac(start);
    let sinc = splat(prefac) * bb * cc * em1 * hard * easy + one;
    let success = last_s * sinc;
    let out = F32x4::blend(rating.cmp_gt(one), pls.max(success), pls);
    Stab {
        start,
        last_s,
        last_d,
        r,
        rating,
        hard,
        easy,
        lns1,
        q,
        er,
        new_s_fail,
        pls,
        lns,
        cc,
        bb,
        er2,
        em1,
        prefac,
        sinc,
        success,
        out,
    }
}

/// Returns (g_last_s, g_last_d, g_r).
#[inline(always)]
fn stab_bwd(
    w: &[f32],
    c: &Stab,
    g_out: F32x4,
    grad: &mut [F32x4; PARAM_LEN],
) -> (F32x4, F32x4, F32x4) {
    let zero = splat(0.0);
    let one = splat(1.0);
    let start = c.start;
    let gt1 = c.rating.cmp_gt(one);
    // out = rating>1 ? max(pls, success) : pls
    let g_max = F32x4::blend(gt1, g_out, zero);
    let g_pls_direct = F32x4::blend(gt1, zero, g_out);
    // max(pls, success): success wins when pls<success
    let succ_wins = c.pls.cmp_lt(c.success);
    let g_success = F32x4::blend(succ_wins, g_max, zero);
    let g_pls_from_max = F32x4::blend(succ_wins, zero, g_max);
    let g_pls = g_pls_direct + g_pls_from_max;

    let mut g_s = zero;
    let mut g_d = zero;
    let mut g_r = zero;

    // success = last_s * sinc
    g_s = g_s + g_success * c.sinc;
    let g_sinc = g_success * c.last_s;
    // sinc = prefac*bb*cc*em1*hard*easy + 1
    let g_prod = g_sinc;
    let pf = splat(c.prefac);
    grad[start] = grad[start] + g_prod * (c.bb * c.cc * c.em1 * c.hard * c.easy) * pf;
    let g_bb = g_prod * (pf * c.cc * c.em1 * c.hard * c.easy);
    let g_cc = g_prod * (pf * c.bb * c.em1 * c.hard * c.easy);
    let g_em1 = g_prod * (pf * c.bb * c.cc * c.hard * c.easy);
    grad[start + 6] = grad[start + 6]
        + F32x4::blend(
            c.rating.cmp_eq(splat(2.0)),
            g_prod * (pf * c.bb * c.cc * c.em1 * c.easy),
            zero,
        );
    grad[start + 7] = grad[start + 7]
        + F32x4::blend(
            c.rating.cmp_eq(splat(4.0)),
            g_prod * (pf * c.bb * c.cc * c.em1 * c.hard),
            zero,
        );
    // bb = 11 - last_d
    g_d = g_d + (zero - g_bb);
    // cc = exp(-w[start+1]*lns)
    let g_cc_arg = g_cc * c.cc;
    grad[start + 1] = grad[start + 1] + g_cc_arg * (zero - c.lns);
    g_s = g_s + g_cc_arg * splat(-w[start + 1]) / c.last_s;
    // em1 = er2-1 ; er2 = exp((1-r)*w[start+2])
    let g_er2_arg = g_em1 * c.er2;
    grad[start + 2] = grad[start + 2] + g_er2_arg * (one - c.r);
    g_r = g_r + g_er2_arg * splat(-w[start + 2]);

    // pls = min(last_s, new_s_fail): new_s_fail wins when last_s>new_s_fail
    let nf_wins = c.last_s.cmp_gt(c.new_s_fail);
    g_s = g_s + F32x4::blend(nf_wins, zero, g_pls);
    let g_new_s_fail = F32x4::blend(nf_wins, g_pls, zero);
    // new_s_fail = w[start+3]*(q-1)*er
    grad[start + 3] = grad[start + 3] + g_new_s_fail * (c.q - one) * c.er;
    let g_q = g_new_s_fail * splat(w[start + 3]) * c.er;
    let g_er = g_new_s_fail * splat(w[start + 3]) * (c.q - one);
    // q = exp(w[start+4]*lns1)
    let g_q_arg = g_q * c.q;
    grad[start + 4] = grad[start + 4] + g_q_arg * c.lns1;
    g_s = g_s + g_q_arg * splat(w[start + 4]) / (c.last_s + one);
    // er = exp((1-r)*w[start+5])
    let g_er_arg = g_er * c.er;
    grad[start + 5] = grad[start + 5] + g_er_arg * (one - c.r);
    g_r = g_r + g_er_arg * splat(-w[start + 5]);

    (g_s, g_d, g_r)
}

// --- next_difficulty ---

struct NextDiff {
    rating: F32x4,
    last_d: F32x4,
    open: uint32x4_t,
    delta_d0: F32x4,
    surprise: F32x4,
    delta_d: F32x4,
}

#[inline(always)]
fn nextdiff_fwd(w: &[f32], params: &Params, last_d: F32x4, r: F32x4, rating: F32x4) -> NextDiff {
    let is1 = rating.cmp_eq(splat(1.0));
    let delta_d0 = splat(-w[6]) * (rating - splat(3.0));
    let surprise = r + splat(0.1);
    let delta_d = F32x4::blend(is1, delta_d0 * surprise, delta_d0);
    let new_d = last_d + (splat(10.0) - last_d) * delta_d / splat(9.0);
    let init_easy = splat(w[4]) - splat(params.exp3w5) + splat(1.0);
    let out_pre = init_easy * splat(0.01) + new_d * splat(0.99);
    let open = open_mask(out_pre, D_MIN as f32, D_MAX as f32);
    NextDiff {
        rating,
        last_d,
        open,
        delta_d0,
        surprise,
        delta_d,
    }
}

/// Returns (g_last_d, g_r).
#[inline(always)]
fn nextdiff_bwd(
    params: &Params,
    c: &NextDiff,
    g_out: F32x4,
    grad: &mut [F32x4; PARAM_LEN],
) -> (F32x4, F32x4) {
    let zero = splat(0.0);
    let is1 = c.rating.cmp_eq(splat(1.0));
    let g_out_pre = F32x4::blend(c.open, g_out, zero);
    let g_init = g_out_pre * splat(0.01);
    let g_new_d = g_out_pre * splat(0.99);
    grad[4] = grad[4] + g_init;
    grad[5] = grad[5] + g_init * splat(-params.exp3w5 * 3.0);
    let g_last_d = g_new_d * (splat(1.0) - c.delta_d / splat(9.0));
    let g_delta_d = g_new_d * (splat(10.0) - c.last_d) / splat(9.0);
    // delta_d = is1 ? delta_d0*surprise : delta_d0
    let g_delta_d0 = F32x4::blend(is1, g_delta_d * c.surprise, g_delta_d);
    let g_r = F32x4::blend(is1, g_delta_d * c.delta_d0, zero);
    grad[6] = grad[6] + g_delta_d0 * (zero - (c.rating - splat(3.0)));
    (g_last_d, g_r)
}

// --- step ---

// Keep per-review caches inline in the training buffer to avoid an allocation per step.
#[allow(clippy::large_enum_variant)]
enum Step {
    First {
        rating: F32x4,
        open_init_s: uint32x4_t,
        open_init_sf: uint32x4_t,
        open_init_d: uint32x4_t,
        init_d_exp: F32x4,
    },
    Full {
        rating: F32x4,
        open_state_s: uint32x4_t,
        open_state_sf: uint32x4_t,
        open_state_d: uint32x4_t,
        curve: Curve,
        slow: Stab,
        fast: Stab,
        nd: NextDiff,
        new_s_fast_raw: F32x4,
        relearn: F32x4,
        open_slow: uint32x4_t,
        open_fast: uint32x4_t,
    },
}

impl Step {
    #[inline(always)]
    fn curve_out(&self) -> F32x4 {
        match self {
            Step::Full { curve, .. } => curve.out,
            Step::First { .. } => splat(0.0),
        }
    }
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn step_fwd(
    w: &[f32],
    params: &Params,
    delta_t: F32x4,
    rating: F32x4,
    s: F32x4,
    sf: F32x4,
    d: F32x4,
    nth: usize,
) -> ((F32x4, F32x4, F32x4), Step) {
    let last_s = s.clamp(S_MIN as f32, S_MAX as f32);
    let last_d = d.clamp(D_MIN as f32, D_MAX as f32);
    let last_sf = sf.clamp(S_MIN as f32, S_MAX as f32);
    let open_state_s = open_mask(s, S_MIN as f32, S_MAX as f32);
    let open_state_sf = open_mask(sf, S_MIN as f32, S_MAX as f32);
    let open_state_d = open_mask(d, D_MIN as f32, D_MAX as f32);

    if nth == 0 {
        let init_s_pre = init_stability(w, rating);
        let init_s = init_s_pre.clamp(S_MIN as f32, S_MAX as f32);
        let open_init_s = open_mask(init_s_pre, S_MIN as f32, S_MAX as f32);
        let init_sf_pre = init_s_pre * splat(0.8);
        let init_sf = init_sf_pre.clamp(S_MIN as f32, S_MAX as f32);
        let open_init_sf = open_mask(init_sf_pre, S_MIN as f32, S_MAX as f32);
        let rc = rating.clamp(1.0, 4.0);
        let init_d_exp = (splat(w[5]) * (rc - splat(1.0))).exp();
        let init_d_pre = splat(w[4]) - init_d_exp + splat(1.0);
        let init_d = init_d_pre.clamp(D_MIN as f32, D_MAX as f32);
        let open_init_d = open_mask(init_d_pre, D_MIN as f32, D_MAX as f32);
        let padding = rating.cmp_eq(splat(0.0));
        return (
            (
                F32x4::blend(padding, last_s, init_s),
                F32x4::blend(padding, last_sf, init_sf),
                F32x4::blend(padding, last_d, init_d),
            ),
            Step::First {
                rating,
                open_init_s,
                open_init_sf,
                open_init_d,
                init_d_exp,
            },
        );
    }

    let dt = delta_t.max(splat(0.0));
    let curve = curve_fwd(w, params, dt, last_s, last_sf, last_d, false);
    let r = curve.out;
    let slow = stab_fwd(w, params, last_s, last_d, r, rating, 7);
    let new_s_slow = slow.out;
    let r_fast = curve.r1;
    let fast = stab_fwd(w, params, last_sf, last_d, r_fast, rating, 15);
    let new_s_fast_raw = fast.out;
    let relearn = new_s_slow * splat(0.8);
    let is1 = rating.cmp_eq(splat(1.0));
    let relearned = F32x4::blend(is1, new_s_fast_raw.min(relearn), new_s_fast_raw);
    let nd = nextdiff_fwd(w, params, last_d, r, rating);
    let new_d_pre = {
        let init_easy = splat(w[4]) - splat(params.exp3w5) + splat(1.0);
        let delta_d = F32x4::blend(is1, nd.delta_d0 * nd.surprise, nd.delta_d0);
        let new_d = last_d + (splat(10.0) - last_d) * delta_d / splat(9.0);
        init_easy * splat(0.01) + new_d * splat(0.99)
    };
    let new_d = new_d_pre.clamp(D_MIN as f32, D_MAX as f32);
    let new_s = new_s_slow.clamp(S_MIN as f32, S_MAX as f32);
    let open_slow = open_mask(new_s_slow, S_MIN as f32, S_MAX as f32);
    let new_sf = relearned.clamp(S_MIN as f32, S_MAX as f32);
    let open_fast = open_mask(relearned, S_MIN as f32, S_MAX as f32);
    let padding = rating.cmp_eq(splat(0.0));
    (
        (
            F32x4::blend(padding, last_s, new_s),
            F32x4::blend(padding, last_sf, new_sf),
            F32x4::blend(padding, last_d, new_d),
        ),
        Step::Full {
            rating,
            open_state_s,
            open_state_sf,
            open_state_d,
            curve,
            slow,
            fast,
            nd,
            new_s_fast_raw,
            relearn,
            open_slow,
            open_fast,
        },
    )
}

/// Returns (g_state_s, g_state_sf, g_state_d).
#[inline(always)]
fn step_bwd(
    w: &[f32],
    params: &Params,
    cache: &Step,
    g_new: (F32x4, F32x4, F32x4),
    g_r_loss: F32x4,
    grad: &mut [F32x4; PARAM_LEN],
) -> (F32x4, F32x4, F32x4) {
    let zero = splat(0.0);
    match cache {
        Step::First {
            rating,
            open_init_s,
            open_init_sf,
            open_init_d,
            init_d_exp,
        } => {
            let active = rating.cmp_gt(zero);
            let g_last_s = F32x4::blend(active, zero, g_new.0);
            let g_last_sf = F32x4::blend(active, zero, g_new.1);
            let g_last_d = F32x4::blend(active, zero, g_new.2);
            let g_init_s = F32x4::blend(mask_and(active, *open_init_s), g_new.0, zero);
            let g_init_sf = F32x4::blend(mask_and(active, *open_init_sf), g_new.1, zero);
            let g_init_s_pre = g_init_s + g_init_sf * splat(0.8);
            let rc = rating.clamp(1.0, 4.0);
            for k in 1..=4 {
                grad[k - 1] =
                    grad[k - 1] + F32x4::blend(rating.cmp_eq(splat(k as f32)), g_init_s_pre, zero);
            }
            let g_init_d = F32x4::blend(mask_and(active, *open_init_d), g_new.2, zero);
            grad[4] = grad[4] + g_init_d;
            grad[5] = grad[5] + g_init_d * (zero - *init_d_exp * (rc - splat(1.0)));
            (g_last_s, g_last_sf, g_last_d)
        }
        Step::Full {
            rating,
            open_state_s,
            open_state_sf,
            open_state_d,
            curve,
            slow,
            fast,
            nd,
            new_s_fast_raw,
            relearn,
            open_slow,
            open_fast,
        } => {
            let active = rating.cmp_gt(zero);
            let g_last_s_direct = F32x4::blend(active, zero, g_new.0);
            let g_last_sf_direct = F32x4::blend(active, zero, g_new.1);
            let g_last_d_direct = F32x4::blend(active, zero, g_new.2);
            let g_new_s = F32x4::blend(active, g_new.0, zero);
            let g_new_sf = F32x4::blend(active, g_new.1, zero);
            let g_new_d = F32x4::blend(active, g_new.2, zero);

            let g_new_s_slow_out = F32x4::blend(*open_slow, g_new_s, zero);
            let g_new_s_fast_post = F32x4::blend(*open_fast, g_new_sf, zero);

            // relearn: new_sf_pre = rating==1 ? min(new_s_fast_raw, relearn) : new_s_fast_raw
            let is1 = rating.cmp_eq(splat(1.0));
            let relearn_wins = new_s_fast_raw.cmp_gt(*relearn); // min picks relearn when raw>relearn
            let g_fast_raw_if1 = F32x4::blend(relearn_wins, zero, g_new_s_fast_post);
            let g_relearn_if1 = F32x4::blend(relearn_wins, g_new_s_fast_post, zero);
            let g_new_s_fast_raw = F32x4::blend(is1, g_fast_raw_if1, g_new_s_fast_post);
            let g_relearn = F32x4::blend(is1, g_relearn_if1, zero);
            let g_new_s_slow = g_new_s_slow_out + g_relearn * splat(0.8);

            let (g_ls1, g_ld1, g_r1) = stab_bwd(w, slow, g_new_s_slow, grad);
            let (g_lsf1, g_ld2, g_rfast) = stab_bwd(w, fast, g_new_s_fast_raw, grad);
            let (g_ld3, g_r2) = nextdiff_bwd(params, nd, g_new_d, grad);
            let (g_ls2, g_lsf2, g_ld4) =
                curve_bwd(w, params, curve, g_r1 + g_r2 + g_r_loss, g_rfast, grad);

            let g_last_s = g_last_s_direct + g_ls1 + g_ls2;
            let g_last_sf = g_last_sf_direct + g_lsf1 + g_lsf2;
            let g_last_d = g_last_d_direct + g_ld1 + g_ld2 + g_ld3 + g_ld4;
            (
                F32x4::blend(*open_state_s, g_last_s, zero),
                F32x4::blend(*open_state_sf, g_last_sf, zero),
                F32x4::blend(*open_state_d, g_last_d, zero),
            )
        }
    }
}

#[inline(always)]
fn bce_retrievability_grad(r_raw: F32x4, label: F32x4, weight: F32x4) -> F32x4 {
    let zero = splat(0.0);
    let one = splat(1.0);
    let r = r_raw.clamp(0.0001, 0.9999);
    let label_is_one = label.cmp_gt(splat(0.5));
    let grad = F32x4::blend(label_is_one, zero - weight / r, weight / (one - r));
    let open = open_mask(r_raw, 0.0001, 0.9999);
    F32x4::blend(open, grad, zero)
}

#[inline(always)]
fn bce_loss_value(r_raw: F32x4, label: F32x4, weight: F32x4) -> F32x4 {
    let one = splat(1.0);
    let r = r_raw.clamp(0.0001, 0.9999);
    let probability = one - (label - r).abs();
    splat(0.0) - weight * probability.ln()
}

#[inline(always)]
fn group_active_seq_len(
    r_historys: &[f32],
    seq_len: usize,
    batch_size: usize,
    column: usize,
) -> usize {
    let mut active = seq_len;
    while active > 1 {
        let index = (active - 1) * batch_size + column;
        if r_historys[index] == 0.0
            && r_historys[index + 1] == 0.0
            && r_historys[index + 2] == 0.0
            && r_historys[index + 3] == 0.0
        {
            active -= 1;
        } else {
            break;
        }
    }
    active
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
    if !batch_size.is_multiple_of(4) {
        return super::windowed_loss_scalar(
            w, t_historys, r_historys, labels, weights, seq_len, batch_size,
        );
    }

    let params = Params::new(w);
    let mut loss = 0.0f64;
    let group_count = batch_size / 4;

    for group in 0..group_count {
        let column = group * 4;
        let active_seq_len = group_active_seq_len(r_historys, seq_len, batch_size, column);
        if active_seq_len < 2 {
            continue;
        }

        let mut s = splat(0.0);
        let mut sf = splat(0.0);
        let mut d = splat(0.0);
        for row in 0..active_seq_len {
            let index = row * batch_size + column;
            if row > 0 {
                let curve = curve_fwd(w, &params, F32x4::load(t_historys, index), s, sf, d, false);
                loss += bce_loss_value(
                    curve.out,
                    F32x4::load(labels, index),
                    F32x4::load(weights, index),
                )
                .sum() as f64;
            }
            if row + 1 < active_seq_len {
                let (next, _) = step_fwd(
                    w,
                    &params,
                    F32x4::load(t_historys, index),
                    F32x4::load(r_historys, index),
                    s,
                    sf,
                    d,
                    row,
                );
                s = next.0;
                sf = next.1;
                d = next.2;
            }
        }
    }

    loss
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
        let mut caches: Vec<Step> = Vec::with_capacity(seq_len);
        for group in 0..group_count {
            let column = group * 4;
            let active_seq_len = group_active_seq_len(r_historys, seq_len, batch_size, column);
            if active_seq_len < 2 {
                continue;
            }
            caches.clear();
            let mut s = splat(0.0);
            let mut sf = splat(0.0);
            let mut d = splat(0.0);
            for row in 0..active_seq_len - 1 {
                let index = row * batch_size + column;
                let (next, cache) = step_fwd(
                    w,
                    &params,
                    F32x4::load(t_historys, index),
                    F32x4::load(r_historys, index),
                    s,
                    sf,
                    d,
                    row,
                );
                s = next.0;
                sf = next.1;
                d = next.2;
                caches.push(cache);
            }

            let last_index = (active_seq_len - 1) * batch_size + column;
            let mut group_grad = [splat(0.0); PARAM_LEN];
            let last_curve = curve_fwd(
                w,
                &params,
                F32x4::load(t_historys, last_index),
                s,
                sf,
                d,
                false,
            );
            let g_last_r = bce_retrievability_grad(
                last_curve.out,
                F32x4::load(labels, last_index),
                F32x4::load(weights, last_index),
            );
            let (mut g_s, mut g_sf, mut g_d) = curve_bwd(
                w,
                &params,
                &last_curve,
                g_last_r,
                splat(0.0),
                &mut group_grad,
            );

            for row in (0..active_seq_len - 1).rev() {
                let g_r_loss = if row == 0 {
                    splat(0.0)
                } else {
                    let index = row * batch_size + column;
                    bce_retrievability_grad(
                        caches[row].curve_out(),
                        F32x4::load(labels, index),
                        F32x4::load(weights, index),
                    )
                };
                let prev = step_bwd(
                    w,
                    &params,
                    &caches[row],
                    (g_s, g_sf, g_d),
                    g_r_loss,
                    &mut group_grad,
                );
                g_s = prev.0;
                g_sf = prev.1;
                g_d = prev.2;
            }

            for (dst, src) in grad.iter_mut().zip(group_grad) {
                *dst += src.sum() as f64;
            }
        }
    }

    // remainder columns that don't fill a group of four
    for column in (group_count * 4)..batch_size {
        super::reverse::column_grad(
            w, t_historys, r_historys, labels, weights, seq_len, batch_size, column, &mut grad,
        );
    }

    let mut out = [0.0f32; PARAM_LEN];
    for (dst, src) in out.iter_mut().zip(grad) {
        *dst = src as f32;
    }
    out
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

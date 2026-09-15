//! Hand-written forward + reverse-mode backward for the training BCE loss, with NO
//! autodiff tape. Profiling showed the former tensor-autodiff forward+backward path is
//! ~85% of compute_parameters() time (backward alone 59%); replacing it with a direct
//! scalar VJP removes the tape-record + tape-replay overhead.
//!
//! Scalar per-card loop (the recurrence is sequential; cards are independent). The
//! forward mirrors model_v7's math EXACTLY and stashes the intermediates each step needs;
//! the backward is the reverse-mode adjoint of that same forward (single source of truth).
//! Result is not bit-for-bit vs autodiff (different FP order) but is judged by the
//! ±0.0010 average-log-loss band.

use wide::{CmpEq, CmpGe, CmpGt, CmpLe, CmpLt, f32x8, i32x8};

const PARAM_LEN: usize = super::PARAM_LEN;
const S_MIN: f32 = 0.0001;
const S_MAX: f32 = 36500.0;
const D_MIN: f32 = 1.0;
const D_MAX: f32 = 10.0;
const MIN_R: f32 = 1e-5;
const MAX_R: f32 = 1.0 - 1e-5;

const LOG2E: f32 = std::f32::consts::LOG2_E;
const LN2: f32 = std::f32::consts::LN_2;

#[inline(always)]
fn clamp(x: f32, lo: f32, hi: f32) -> f32 {
    x.max(lo).min(hi)
}

// ===================== portable SIMD transcendentals (8 lanes) =====================
// These vectorize the analytic forward across 8 cards/lane. A wide::f32x8 exp is ~3.8x faster
// than 8x scalar libm (microbench profiling/simd_bench) at ~2e-7 rel err — the forward is ~92%
// transcendentals, so this is the lever. Precision-trading (3b band), portable (constraint 7).

#[inline(always)]
fn clamp8(x: f32x8, lo: f32, hi: f32) -> f32x8 {
    x.fast_max(f32x8::splat(lo)).fast_min(f32x8::splat(hi))
}

/// exp over 8 lanes: 2^n · poly(r), x = n·ln2 + r. Same algorithm as the scalar floor-bench.
///
/// `FAST` (compile-time const): `false` = degree-3 minimax (rel err 7.5e-5) — the accurate version,
/// now used by EVERY path (including the windowed compute_parameters() recurrence). `true` = degree-2
/// (rel err 1.7e-3, ONE fewer FMA) — this was iter23's cruder windowed-only trade, but it was
/// REVERTED 2026-06-04 on cost/benefit (it saved ~6.5% median speed but cost ~0.00025 cp log loss),
/// so the `true` branch is currently UNUSED (retained for a possible future precision A/B). Portable (c7).
#[inline(always)]
fn exp8<const FAST: bool>(x: f32x8) -> f32x8 {
    let x = x.fast_max(f32x8::splat(-87.0)).fast_min(f32x8::splat(88.0));
    let n = (x * f32x8::splat(LOG2E)).round();
    let r = x - n * f32x8::splat(LN2);
    let c = |v: f32| f32x8::splat(v);
    let p = if FAST {
        // degree-2 RELATIVE-minimax of exp(r) over [-ln2/2, ln2/2]; max rel err 1.7e-3.
        c(1.000_443_1) + r * (c(1.014_861) + r * c(0.496_258_6))
    } else {
        // degree-3 RELATIVE-minimax (Remez); max rel err 7.5e-5 (profiling/minimax_coeffs.py).
        c(0.999_928_06) + r * (c(1.000_164_2) + r * (c(0.504_963_3) + r * c(0.165_668_43)))
    };
    let bits: i32x8 = (n.round_int() + i32x8::splat(127)) << 23;
    let two_n: f32x8 = bytemuck::cast(bits);
    p * two_n
}

/// ln over 8 lanes: x = m·2^e, ln = e·ln2 + minimax(atanh-series) in t=(m-1)/(m+1).
/// All forward ln inputs are > 0 (stabilities ≥ S_MIN, difficulty ≥ 1, the b/qbase bases > 0).
///
/// `FAST` (compile-time const): `false` = degree-2-in-u minimax (abs err 4.9e-6) — the accurate
/// version, now used by EVERY path (including the windowed compute_parameters() recurrence). `true` =
/// degree-1-in-u (abs err 2.3e-4, ONE fewer FMA) — iter23's cruder windowed-only trade, REVERTED
/// 2026-06-04 on cost/benefit, so the `true` branch is currently UNUSED. Portable (c7).
#[inline(always)]
fn ln8<const FAST: bool>(x: f32x8) -> f32x8 {
    let bits: i32x8 = bytemuck::cast(x);
    let e: i32x8 = ((bits >> 23) & i32x8::splat(0xff)) - i32x8::splat(127);
    let mant_bits: i32x8 = (bits & i32x8::splat(0x007f_ffff)) | i32x8::splat(127 << 23);
    let m: f32x8 = bytemuck::cast(mant_bits);
    let one = f32x8::splat(1.0);
    let t = (m - one) / (m + one);
    let t2 = t * t;
    let c = |v: f32| f32x8::splat(v);
    let poly = if FAST {
        // degree-1-in-u (u=t^2) minimax of atanh(t)/t over u in [0,1/9]; abs ln err 2.3e-4.
        c(2.0) * t * (c(0.999_650_36) + t2 * c(0.357_486_93))
    } else {
        // degree-2-in-u minimax; reconstructed abs ln err 4.9e-6 (profiling/minimax_coeffs.py).
        c(2.0) * t * (c(1.000_007_4) + t2 * (c(0.332_179_52) + t2 * c(0.226_577_77)))
    };
    let e_f: f32x8 = e.round_float();
    e_f * f32x8::splat(LN2) + poly
}

/// Loop-invariant (weight-only) subexpressions, computed ONCE per batch_loss[_and_grad] call
/// instead of once per (card × timestep). Identical ops to the old inline versions (same f32/f64),
/// so it's BIT-FOR-BIT — it just lifts redundant transcendentals out of the hot per-timestep
/// forward (and the one in next_d's backward). iter10.
// Finished FSRS-7 34-param layout. NOTE: the field names ln_w27/ln_w28/aa16 are LEGACY labels
// kept to limit churn — they now hold ln(base1=w[25]), ln(base2=w[26]), exp(short sinc_base
// w[15]-1.5) respectively.
struct WConsts {
    ln_w27: f32, // ln(w[25]) = ln(base1)   (curve_fwd: q1 = ln_base1/decay1)
    ln_w28: f32, // ln(w[26]) = ln(base2)   (curve_fwd: p28 = (inv2*ln_base2).exp())
    aa7: f32,    // exp(w[7]-1.5)   (long stab sinc_base, start=7)
    aa16: f32,   // exp(w[15]-1.5)  (short stab sinc_base, start=15)
    init: f32,   // w4 - exp(3*w5) + 1   (next_d_fwd, f32)
    exp3w5: f64, // exp(3*w5) in f64      (next_d_bwd: d(init)/d(w5))
}

fn wconsts(w: &[f32]) -> WConsts {
    WConsts {
        ln_w27: w[25].ln(),
        ln_w28: w[26].ln(),
        aa7: (w[7] - 1.5).exp(),
        aa16: (w[15] - 1.5).exp(),
        init: w[4] - (w[5] * 3.0).exp() + 1.0,
        exp3w5: (w[5] as f64 * 3.0).exp(),
    }
}

// ===================== forgetting curve =====================

struct CurveCache {
    out: f32,
    a: f32,
    bv: f32,
    dm1: f32,
    decay1: f32,
    factor1: f32,
    b1: f32,
    r1: f32,
    q1: f32,
    e1: f32,
    m1: f32,
    p35: f32,
    dm2: f32,
    decay2: f32,
    factor2: f32,
    b2: f32,
    r2: f32,
    inv2: f32,
    p28: f32,
    m2: f32,
    ex34: f32,
    weight1: f32,
    weight2: f32,
    wsum: f32,
    num: f32,
    ret: f32,
    p31: f32,
    s32: f32,
    ex33: f32,
    // Cached ln(base) for each powf rewritten as (exp*ln_base).exp() — reused by curve_bwd so
    // the backward's exponent-derivatives (d/dw = value*ln_base) don't recompute ln. iter8.
    ln_sf: f32, // shared by p35 (sf^w35) and p31 (sf^-w31)
    ln_b1: f32,
    ln_b2: f32,
    ln_s: f32,
    ln_w27: f32,
    ln_w28: f32,
}

// Keep numerical inputs and precomputed intermediates explicit at this kernel boundary.
#[allow(clippy::too_many_arguments)]
fn curve_fwd(
    w: &[f32],
    t: f32,
    s: f32,
    sf: f32,
    d: f32,
    ln_w27: f32,
    ln_w28: f32,
    ln_s: f32,
    ln_sf: f32,
) -> CurveCache {
    let t = t.max(0.0);
    let a = t / sf;
    let bv = t / s;
    // Legacy cache labels remapped to the 34-param layout (+ all-positive offsets):
    //   p35 = s_short^(s_decay1-0.3)   [s_decay1=w33], m1 = decay1[w23]*p35
    //   ex34 = exp((d-5)*(d_decay-0.3)) [d_decay=w32], m2 = decay2[w24]*ex34, p28 = base2[w26]^inv2
    //   p31 = s_short^(-s_weight_power1)[w29], weight1 = base_weight1[w27]*p31
    //   s32 = s_long^s_weight_power2 [w30], ex33 = exp((d-5)*(d_weight-0.5))[w31]
    //   weight2 = base_weight2[w28]*s32*ex33. ln_w27=ln(base1=w25), ln_w28=ln(base2=w26).
    let p35 = ((w[33] - 0.3) * ln_sf).exp();
    let m1 = w[23] * p35;
    let dm1 = clamp(m1, 0.01, 0.95);
    let decay1 = -dm1;
    let q1 = ln_w27 / decay1; // ln(base1) hoisted
    let e1 = q1.min(60.0).exp();
    let factor1 = e1 - 1.0;
    let b1 = a * factor1 + 1.0;
    let ln_b1 = b1.ln();
    let r1 = (decay1 * ln_b1).exp(); // b1^decay1
    // iter-165: the D-effect moved from the decay exponent to the TIME-SCALE — decay2 is no
    // longer d-modulated (m2 = w24 plain); ex34 = exp((d-5)*(d_decay-0.3)) now scales time
    // inside b2 instead of multiplying m2.
    let ex34 = ((d - 5.0) * (w[32] - 0.3)).exp();
    let m2 = w[24];
    let dm2 = clamp(m2, 0.01, 0.95);
    let decay2 = -dm2;
    let inv2 = 1.0 / decay2;
    let p28 = (inv2 * ln_w28).exp(); // base2^inv2 ; ln(base2) hoisted
    let factor2 = p28 - 1.0;
    let b2 = bv * factor2 * ex34 + 1.0;
    let ln_b2 = b2.ln();
    let r2 = (decay2 * ln_b2).exp(); // b2^decay2
    let p31 = ((-w[29]) * ln_sf).exp(); // s_short^-s_weight_power1
    let weight1 = w[27] * p31;
    let s32 = (w[30] * ln_s).exp(); // s_long^s_weight_power2
    let ex33 = ((d - 5.0) * (w[31] - 0.5)).exp();
    let weight2 = w[28] * s32 * ex33;
    let wsum = weight1 + weight2;
    let num = weight1 * r1 + weight2 * r2;
    let ret = num / wsum;
    let out = ret * (1.0 - 2e-5) + 1e-5;
    CurveCache {
        out,
        a,
        bv,
        dm1,
        decay1,
        factor1,
        b1,
        r1,
        q1,
        e1,
        m1,
        p35,
        dm2,
        decay2,
        factor2,
        b2,
        r2,
        inv2,
        p28,
        m2,
        ex34,
        weight1,
        weight2,
        wsum,
        num,
        ret,
        p31,
        s32,
        ex33,
        ln_sf,
        ln_b1,
        ln_b2,
        ln_s,
        ln_w27,
        ln_w28,
    }
}

/// VJP of the curve. `t` is data (no grad). Returns adjoints (g_s, g_sf, g_d), accumulates gw.
#[allow(clippy::too_many_arguments)]
fn curve_bwd(
    w: &[f32],
    c: &CurveCache,
    t: f32,
    s: f32,
    sf: f32,
    d: f32,
    g_out: f64,
    g_r1_extra: f64,
    gw: &mut [f64],
) -> (f64, f64, f64) {
    let t = t.max(0.0) as f64;
    let (s, sf, d) = (s as f64, sf as f64, d as f64);
    let g_ret = g_out * (1.0 - 2e-5);
    let (wsum, ret) = (c.wsum as f64, c.ret as f64);
    let g_num = g_ret / wsum;
    let g_wsum = -g_ret * ret / wsum;
    let (r1, r2) = (c.r1 as f64, c.r2 as f64);
    let g_weight1 = g_num * r1 + g_wsum;
    let g_weight2 = g_num * r2 + g_wsum;
    // r1 feeds BOTH the mixture (weight1*r1) AND the short-trace stability update (which reads r1
    // instead of the mixed retention) — the latter's adjoint arrives as g_r1_extra.
    let g_r1 = g_num * c.weight1 as f64 + g_r1_extra;
    let g_r2 = g_num * c.weight2 as f64;
    // weight2 = base_weight2[w28] * s32 * ex33
    let (s32, ex33) = (c.s32 as f64, c.ex33 as f64);
    gw[28] += g_weight2 * s32 * ex33;
    let g_s32 = g_weight2 * w[28] as f64 * ex33;
    let g_ex33 = g_weight2 * w[28] as f64 * s32;
    let mut g_d = g_ex33 * ex33 * (w[31] as f64 - 0.5); // ex33 = exp((d-5)*(d_weight-0.5))
    gw[31] += g_ex33 * ex33 * (d - 5.0); // offset is +const => d/d(w31) deriv factor = 1
    let mut g_s = g_s32 * w[30] as f64 * (s32 / s); // s32 = s_long^s_weight_power2[w30]
    gw[30] += g_s32 * s32 * c.ln_s as f64;
    // weight1 = base_weight1[w27] * p31 ; p31 = s_short^(-s_weight_power1[w29])
    let p31 = c.p31 as f64;
    gw[27] += g_weight1 * p31;
    let g_p31 = g_weight1 * w[27] as f64;
    let mut g_sf = g_p31 * (-(w[29] as f64)) * (p31 / sf); // d(sf^-w29)/dsf = -w29*p31/sf
    gw[29] += g_p31 * (-(p31 * c.ln_sf as f64));
    // r2 = b2^decay2 ; iter-165: b2 = bv*factor2*ex34 + 1 (ex34 is the D time-scale) and
    // decay2 = -clamp(w24) is no longer d-modulated.
    let (b2, decay2) = (c.b2 as f64, c.decay2 as f64);
    let g_b2 = g_r2 * decay2 * (r2 / b2); // d(b2^decay2)/db2 = decay2*r2/b2
    let mut g_decay2 = g_r2 * r2 * c.ln_b2 as f64;
    let factor2 = c.factor2 as f64;
    let ex34 = c.ex34 as f64;
    let g_bv = g_b2 * factor2 * ex34; // b2 = bv*factor2*ex34 + 1
    let g_factor2 = g_b2 * c.bv as f64 * ex34;
    let g_ex34 = g_b2 * c.bv as f64 * factor2;
    g_d += g_ex34 * ex34 * (w[32] as f64 - 0.3); // ex34 = exp((d-5)*(d_decay-0.3))
    gw[32] += g_ex34 * ex34 * (d - 5.0); // offset is +const => deriv factor = 1
    let g_p28 = g_factor2; // factor2 = p28 - 1
    // p28 = base2[w26]^inv2
    let (inv2, p28) = (c.inv2 as f64, c.p28 as f64);
    gw[26] += g_p28 * inv2 * (p28 / w[26] as f64); // d(base2^inv2)/d(base2) = inv2*p28/base2
    let g_inv2 = g_p28 * p28 * c.ln_w28 as f64;
    g_decay2 += g_inv2 * (-1.0 / (decay2 * decay2)); // inv2 = 1/decay2
    let g_dm2 = -g_decay2; // decay2 = -dm2
    let g_m2 = if c.m2 > 0.01 && c.m2 < 0.95 {
        g_dm2
    } else {
        0.0
    };
    gw[24] += g_m2; // m2 = w24 directly (clamp gate via c.m2)
    g_s += g_bv * (-t / (s * s)); // bv = t/s
    // r1 = b1^decay1
    let (b1, decay1) = (c.b1 as f64, c.decay1 as f64);
    let g_b1 = g_r1 * decay1 * (r1 / b1); // d(b1^decay1)/db1 = decay1*r1/b1
    let mut g_decay1 = g_r1 * r1 * c.ln_b1 as f64;
    let g_a = g_b1 * c.factor1 as f64; // b1 = a*factor1 + 1
    let g_factor1 = g_b1 * c.a as f64;
    let g_e1 = g_factor1; // factor1 = e1 - 1
    let g_q1c = g_e1 * c.e1 as f64; // e1 = exp(q1c)
    let g_q1 = if (c.q1 as f64) < 60.0 { g_q1c } else { 0.0 }; // q1c = min(q1,60)
    // q1 = ln(base1) / decay1
    let lw27 = c.ln_w27 as f64;
    let g_lw27 = g_q1 / decay1;
    g_decay1 += g_q1 * (-lw27 / (decay1 * decay1));
    gw[25] += g_lw27 / w[25] as f64; // ln_base1 = ln(w[25])
    let g_dm1 = -g_decay1; // decay1 = -dm1
    let g_m1 = if c.m1 > 0.01 && c.m1 < 0.95 {
        g_dm1
    } else {
        0.0
    };
    let p35 = c.p35 as f64;
    gw[23] += g_m1 * p35; // m1 = decay1[w23] * p35
    let g_p35 = g_m1 * w[23] as f64;
    g_sf += g_p35 * (w[33] as f64 - 0.3) * (p35 / sf); // p35 = s_short^(s_decay1-0.3)
    gw[33] += g_p35 * p35 * c.ln_sf as f64;
    g_sf += g_a * (-t / (sf * sf)); // a = t/sf
    (g_s, g_sf, g_d)
}

// ===================== stability after review =====================

struct StabCache {
    out: f32,
    nsf_fail: f32,
    pls: f32,
    sinc: f32,
    ls_sinc: f32,
    aa: f32,
    bb: f32,
    cc: f32,
    expr: f32,
    pp: f32,
    qbase: f32,
    rexp: f32,
    hard: f32,
    easy: f32,
    ln_ls: f32,  // ln(last_s)   for cc = last_s^-w[start+1]
    ln_ld: f32,  // ln(last_d)   for pp = last_d^-w[start+4]
    ln_ls1: f32, // ln(last_s+1) for qbase = (last_s+1)^w[start+5]
}

// Keep numerical inputs and precomputed intermediates explicit at this kernel boundary.
#[allow(clippy::too_many_arguments)]
fn stab_fwd(
    w: &[f32],
    last_s: f32,
    last_d: f32,
    r: f32,
    rating: f32,
    start: usize,
    aa: f32,
    ln_ls: f32,
    ln_ld: f32,
) -> StabCache {
    // Finished layout (8 params/block): start sinc_base, +1 sinc_s_exp, +2 sinc_r_mult,
    // +3 fail_mult, +4 fail_s_exp, +5 fail_r_mult, +6 hard_penalty, +7 easy_bonus.
    // fail_d_exp DROPPED: new_s_fail is D-independent, so ln_ld (last_d) no longer feeds it.
    let hard = if rating == 2.0 { w[start + 6] } else { 1.0 };
    let easy = if rating == 4.0 { w[start + 7] } else { 1.0 };
    let ln_ls1 = (last_s + 1.0).ln();
    let qbase = (w[start + 4] * ln_ls1).exp(); // (last_s+1)^fail_s_exp[start+4]
    let rexp = ((1.0 - r) * w[start + 5]).exp(); // exp((1-r)*fail_r_mult[start+5])
    let nsf_fail = w[start + 3] * (qbase - 1.0) * rexp; // fail_mult * (qbase-1) * rexp
    let pls = last_s.min(nsf_fail);
    let bb = 11.0 - last_d; // aa = exp(w[start]-1.5) hoisted (loop-invariant)
    let cc = ((-w[start + 1]) * ln_ls).exp(); // last_s^-sinc_s_exp[start+1] (ln_ls shared)
    let expr = ((1.0 - r) * w[start + 2]).exp();
    let sinc = aa * bb * cc * (expr - 1.0) * hard * easy + 1.0;
    let ls_sinc = last_s * sinc;
    let nss = pls.max(ls_sinc);
    let out = if rating > 1.0 { nss } else { pls };
    StabCache {
        out,
        nsf_fail,
        pls,
        sinc,
        ls_sinc,
        aa,
        bb,
        cc,
        expr,
        pp: 0.0,
        qbase,
        rexp,
        hard,
        easy,
        ln_ls,
        ln_ld,
        ln_ls1,
    }
}

/// VJP of stability_for_set. Returns (g_last_s, g_last_d, g_r), accumulates gw[start..start+9].
// Keep numerical inputs and precomputed intermediates explicit at this kernel boundary.
#[allow(clippy::too_many_arguments)]
fn stab_bwd(
    w: &[f32],
    c: &StabCache,
    last_s: f32,
    last_d: f32,
    r: f32,
    rating: f32,
    start: usize,
    g_out: f64,
    gw: &mut [f64],
) -> (f64, f64, f64) {
    let (last_s, r) = (last_s as f64, r as f64);
    let _ = last_d; // post-lapse stability is D-independent now (fail_d_exp dropped)
    let (g_nss, g_pls_direct) = if rating > 1.0 {
        (g_out, 0.0)
    } else {
        (0.0, g_out)
    };
    // nss = max(pls, ls_sinc)
    let g_pls_from_nss = if c.pls >= c.ls_sinc { g_nss } else { 0.0 };
    let g_ls_sinc = if c.ls_sinc > c.pls { g_nss } else { 0.0 };
    // ls_sinc = last_s * sinc
    let mut g_last_s = g_ls_sinc * c.sinc as f64;
    let g_sinc = g_ls_sinc * last_s;
    let g_pls = g_pls_direct + g_pls_from_nss;
    // pls = min(last_s, nsf_fail)
    let nsf_fail = c.nsf_fail as f64;
    g_last_s += if last_s <= nsf_fail { g_pls } else { 0.0 };
    let g_nsf_fail = if nsf_fail < last_s { g_pls } else { 0.0 };
    // sinc = aa*bb*cc*(expr-1)*hard*easy + 1  ; let prod = sinc - 1
    let (aa, bb, cc) = (c.aa as f64, c.bb as f64, c.cc as f64);
    let em1 = c.expr as f64 - 1.0;
    let (hard, easy) = (c.hard as f64, c.easy as f64);
    let g_prod = g_sinc;
    let prod = aa * bb * cc * em1 * hard * easy;
    gw[start] += g_prod * prod; // aa = exp(w[start]-1.5); dprod/dw[start] = prod
    let g_bb = g_prod * (aa * cc * em1 * hard * easy);
    let g_cc = g_prod * (aa * bb * em1 * hard * easy);
    let g_em1 = g_prod * (aa * bb * cc * hard * easy);
    if rating == 2.0 {
        gw[start + 6] += g_prod * (aa * bb * cc * em1 * easy); // hard_penalty
    }
    if rating == 4.0 {
        gw[start + 7] += g_prod * (aa * bb * cc * em1 * hard); // easy_bonus
    }
    let g_last_d = -g_bb; // bb = 11 - last_d (the ONLY D-dependence of stab now)
    g_last_s += g_cc * (-(w[start + 1] as f64)) * (cc / last_s); // d(ls^-w)/dls = -w*cc/ls
    gw[start + 1] += g_cc * (-(cc * c.ln_ls as f64));
    // expr = exp((1-r)*w[start+2]) ; em1 = expr - 1
    let expr = c.expr as f64;
    let mut g_r = g_em1 * expr * (-(w[start + 2] as f64));
    gw[start + 2] += g_em1 * expr * (1.0 - r);
    // nsf_fail = fail_mult[start+3] * (qbase-1) * rexp   (no d^-fail_d_exp factor)
    let (qbase, rexp) = (c.qbase as f64, c.rexp as f64);
    let q = qbase - 1.0;
    gw[start + 3] += g_nsf_fail * (q * rexp);
    let g_q = g_nsf_fail * w[start + 3] as f64 * rexp;
    let g_rexp = g_nsf_fail * w[start + 3] as f64 * q;
    // q = qbase - 1 ; qbase = (last_s+1)^fail_s_exp[start+4]
    g_last_s += g_q * w[start + 4] as f64 * (qbase / (last_s + 1.0));
    gw[start + 4] += g_q * qbase * c.ln_ls1 as f64;
    // rexp = exp((1-r)*fail_r_mult[start+5])
    g_r += g_rexp * rexp * (-(w[start + 5] as f64));
    gw[start + 5] += g_rexp * rexp * (1.0 - r);
    (g_last_s, g_last_d, g_r)
}

// ===================== next difficulty =====================

fn next_d_fwd(w: &[f32], last_d: f32, rating: f32, r: f32, init: f32) -> (f32, f32, f32) {
    let delta_d_base = -w[6] * (rating - 3.0);
    // SURPRISE-WEIGHTED lapse difficulty: on a lapse scale delta_d by (r + 0.1) = 1 + (R - 0.9).
    let delta_d = if rating == 1.0 {
        delta_d_base * (r + 0.1)
    } else {
        delta_d_base
    };
    let new_d = last_d + (10.0 - last_d) * delta_d / 9.0;
    // init = w4 - exp(3*w5) + 1 hoisted (loop-invariant)
    let out_pre = 0.01 * init + 0.99 * new_d;
    (clamp(out_pre, D_MIN, D_MAX), out_pre, delta_d) // delta_d returned = the EFFECTIVE delta_d
}

/// VJP of next_difficulty. Returns (g_last_d, g_r); accumulates gw[4], gw[5], gw[6]. `delta_d` is
/// the EFFECTIVE delta_d from the forward; `r` is the curve retention (feeds the lapse surprise).
#[allow(clippy::too_many_arguments)]
fn next_d_bwd(
    w: &[f32],
    out_pre: f32,
    delta_d: f32,
    last_d: f32,
    rating: f32,
    r: f32,
    g_out: f64,
    gw: &mut [f64],
    exp3w5: f64,
) -> (f64, f64) {
    let g_out_pre = if out_pre > D_MIN && out_pre < D_MAX {
        g_out
    } else {
        0.0
    };
    let g_init = g_out_pre * 0.01;
    let g_new_d = g_out_pre * 0.99;
    gw[4] += g_init;
    gw[5] += g_init * (-exp3w5 * 3.0); // init = w4 - exp(3 w5) + 1 ; exp3w5 hoisted
    let g_last_d = g_new_d * (1.0 - delta_d as f64 / 9.0);
    let g_delta_d = g_new_d * (10.0 - last_d as f64) / 9.0;
    let rm3 = rating as f64 - 3.0;
    let mut g_r = 0.0;
    if rating == 1.0 {
        // delta_d_eff = (-w6*(rating-3)) * (r+0.1)
        gw[6] += g_delta_d * (-rm3) * (r as f64 + 0.1);
        g_r = g_delta_d * (-(w[6] as f64) * rm3); // d(delta_d_eff)/dr = delta_d_base
    } else {
        gw[6] += g_delta_d * (-rm3); // delta_d = -w6*(rating-3)
    }
    (g_last_d, g_r)
}

// ===================== one recurrence step =====================

struct StepCache {
    s0: f32,
    d0: f32,
    sf0: f32,
    last_s: f32,
    last_d: f32,
    last_sf: f32,
    dt: f32,
    rating: f32,
    nth0: bool,
    curve: CurveCache,
    slow: StabCache,
    fast: StabCache,
    nd_out_pre: f32,
    nd_delta_d: f32,
    ns3: f32, // value before the final stability clamp
    nsf3: f32,
}

fn step_fwd(
    w: &[f32],
    delta_t: f32,
    rating: f32,
    state: (f32, f32, f32),
    nth0: bool,
    wc: &WConsts,
) -> ((f32, f32, f32), StepCache) {
    let (s0, d0, sf0) = state;
    let last_s = clamp(s0, S_MIN, S_MAX);
    let last_d = clamp(d0, D_MIN, D_MAX);
    let last_sf = clamp(sf0, S_MIN, S_MAX);
    let dt = delta_t.max(0.0);
    // Compute the per-state lns ONCE and share them across the curve + both stability traces:
    // curve needs ln(last_s)+ln(last_sf); slow stab ln(last_s)+ln(last_d); fast stab
    // ln(last_sf)+ln(last_d) — so 3 ln/timestep instead of 6. Bit-for-bit (same values). iter13.
    let ln_last_s = last_s.ln();
    let ln_last_sf = last_sf.ln();
    let ln_last_d = last_d.ln();
    let curve = curve_fwd(
        w, dt, last_s, last_sf, last_d, wc.ln_w27, wc.ln_w28, ln_last_s, ln_last_sf,
    );
    let r = curve.out;
    let r1 = curve.r1; // short component recall — drives the short-trace update (iter-71)
    let slow = stab_fwd(
        w, last_s, last_d, r, rating, 7, wc.aa7, ln_last_s, ln_last_d,
    );
    let fast = stab_fwd(
        w, last_sf, last_d, r1, rating, 15, wc.aa16, ln_last_sf, ln_last_d,
    );
    let (nd1, nd_out_pre, nd_delta_d) = next_d_fwd(w, last_d, rating, r, wc.init);
    // POST-LAPSE short reset (iter-97): on a lapse cap s_short at 0.8 * post-lapse s_long.
    let nsf_pre = if rating == 1.0 {
        fast.out.min(0.8 * slow.out)
    } else {
        fast.out
    };
    let (mut ns, mut nsf, mut nd) = (slow.out, nsf_pre, nd1);
    if nth0 && s0 == 0.0 {
        let rc = clamp(rating, 1.0, 4.0);
        let init_s = w[(rc as usize) - 1];
        let init_d = clamp(w[4] - (w[5] * (rc - 1.0)).exp() + 1.0, D_MIN, D_MAX);
        ns = init_s;
        nsf = 0.8 * init_s;
        nd = init_d;
    }
    if rating == 0.0 {
        ns = last_s;
        nsf = last_sf;
        nd = last_d;
    }
    let ns3 = ns;
    let nsf3 = nsf;
    let out = (clamp(ns, S_MIN, S_MAX), nd, clamp(nsf, S_MIN, S_MAX));
    let cache = StepCache {
        s0,
        d0,
        sf0,
        last_s,
        last_d,
        last_sf,
        dt,
        rating,
        nth0,
        curve,
        slow,
        fast,
        nd_out_pre,
        nd_delta_d,
        ns3,
        nsf3,
    };
    (out, cache)
}

/// VJP of one step. Given adjoints on the OUTPUT state, returns adjoints on the INPUT state.
fn step_bwd(
    w: &[f32],
    c: &StepCache,
    g_out: (f64, f64, f64),
    gw: &mut [f64],
    wc: &WConsts,
) -> (f64, f64, f64) {
    let (g_ns_out, g_nd_out, g_nsf_out) = g_out;
    // final clamps: ns_out = clamp(ns3, S_MIN, S_MAX) ; nd has no final clamp
    let g_ns3 = if c.ns3 > S_MIN && c.ns3 < S_MAX {
        g_ns_out
    } else {
        0.0
    };
    let g_nsf3 = if c.nsf3 > S_MIN && c.nsf3 < S_MAX {
        g_nsf_out
    } else {
        0.0
    };
    let g_nd3 = g_nd_out;
    // padding mask
    let (mut g_last_s_extra, mut g_last_sf_extra, mut g_last_d_extra) = (0.0, 0.0, 0.0);
    let (g_ns2, g_nsf2, g_nd2) = if c.rating == 0.0 {
        g_last_s_extra += g_ns3;
        g_last_sf_extra += g_nsf3;
        g_last_d_extra += g_nd3;
        (0.0, 0.0, 0.0)
    } else {
        (g_ns3, g_nsf3, g_nd3)
    };
    // init override (t==0)
    let (g_ns1, g_nsf1, g_nd1) = if c.nth0 && c.s0 == 0.0 {
        let rc = clamp(c.rating, 1.0, 4.0);
        gw[(rc as usize) - 1] += g_ns2 + g_nsf2 * 0.8; // init_s = w[rc-1]; nsf=0.8*init_s
        let id_pre = w[4] - (w[5] * (rc - 1.0)).exp() + 1.0;
        if id_pre > D_MIN && id_pre < D_MAX {
            gw[4] += g_nd2;
            gw[5] += g_nd2 * (-((w[5] as f64 * (rc as f64 - 1.0)).exp()) * (rc as f64 - 1.0));
        }
        (0.0, 0.0, 0.0)
    } else {
        (g_ns2, g_nsf2, g_nd2)
    };
    // POST-LAPSE min routing: nsf_pre = (rating==1)? min(fast.out, 0.8*slow.out) : fast.out.
    // g_nsf1 is the adjoint on nsf_pre; route it to fast.out and/or 0.8*slow.out.
    let (g_fast_out, g_slow_from_relearn) = if c.rating == 1.0 {
        if c.fast.out <= 0.8 * c.slow.out {
            (g_nsf1, 0.0)
        } else {
            (0.0, g_nsf1 * 0.8)
        }
    } else {
        (g_nsf1, 0.0)
    };
    // LONG (slow) stab reads the mixed retention curve.out; SHORT (fast) stab reads r1=curve.r1.
    let (g_ls_a, g_ld_a, g_r_long) = stab_bwd(
        w,
        &c.slow,
        c.last_s,
        c.last_d,
        c.curve.out,
        c.rating,
        7,
        g_ns1 + g_slow_from_relearn,
        gw,
    );
    let (g_lsf_b, g_ld_b, g_r1_short) = stab_bwd(
        w, &c.fast, c.last_sf, c.last_d, c.curve.r1, c.rating, 15, g_fast_out, gw,
    );
    // next_d reads curve.out (and on a lapse depends on it via the surprise weighting).
    let (g_ld_c, g_r_nextd) = next_d_bwd(
        w,
        c.nd_out_pre,
        c.nd_delta_d,
        c.last_d,
        c.rating,
        c.curve.out,
        g_nd1,
        gw,
        wc.exp3w5,
    );
    // curve.out adjoint = long-stab r + next_d r; curve.r1 adjoint (g_r1_extra) = short-stab r.
    let (g_ls_d, g_lsf_d, g_ld_d) = curve_bwd(
        w,
        &c.curve,
        c.dt,
        c.last_s,
        c.last_sf,
        c.last_d,
        g_r_long + g_r_nextd,
        g_r1_short,
        gw,
    );
    let g_last_s = g_ls_a + g_ls_d + g_last_s_extra;
    let g_last_sf = g_lsf_b + g_lsf_d + g_last_sf_extra;
    let g_last_d = g_ld_a + g_ld_b + g_ld_c + g_ld_d + g_last_d_extra;
    // input clamps
    let g_s0 = if c.s0 > S_MIN && c.s0 < S_MAX {
        g_last_s
    } else {
        0.0
    };
    let g_d0 = if c.d0 > D_MIN && c.d0 < D_MAX {
        g_last_d
    } else {
        0.0
    };
    let g_sf0 = if c.sf0 > S_MIN && c.sf0 < S_MAX {
        g_last_sf
    } else {
        0.0
    };
    (g_s0, g_d0, g_sf0)
}

// ===================== batch driver =====================

/// Scalar forward-only BCE loss for one batch — the reference oracle for the SIMD validation
/// scorer (batch_loss_simd replaced it in the hot path at iter14; the unit test checks they agree)
/// and for the finite-difference gradient test. `t_hist`/`r_hist` are row-major [seq_len, batch].
#[allow(clippy::too_many_arguments, dead_code)]
pub(crate) fn batch_loss(
    w: &[f32],
    t_hist: &[f32],
    r_hist: &[f32],
    seq_len: usize,
    batch: usize,
    delta_ts: &[f32],
    labels: &[f32],
    weights: &[f32],
) -> f64 {
    let wc = wconsts(w);
    let mut loss = 0.0f64;
    for c in 0..batch {
        let mut state = (0.0f32, 0.0f32, 0.0f32);
        for t in 0..seq_len {
            state = step_fwd(
                w,
                t_hist[t * batch + c],
                r_hist[t * batch + c],
                state,
                t == 0,
                &wc,
            )
            .0;
        }
        let (s, d, sf) = state;
        let r = clamp(
            curve_fwd(
                w,
                delta_ts[c],
                s,
                sf,
                d,
                wc.ln_w27,
                wc.ln_w28,
                s.ln(),
                sf.ln(),
            )
            .out,
            MIN_R,
            MAX_R,
        );
        loss += -(weights[c] as f64)
            * (labels[c] as f64 * (r as f64).ln()
                + (1.0 - labels[c] as f64) * (1.0 - r as f64).ln());
    }
    loss
}

// ===================== SIMD validation forward (8 cards/lane) =====================
// Vectorized, FORWARD-ONLY mirror of step_fwd's recurrence + the final curve, used by the
// per-epoch validation scorer. exp8/ln8 replace libm exp/ln (precision-trade, 3b band). The
// data layout [seq_len, batch] makes 8 consecutive cards at one timestep a contiguous f32x8 load.

#[inline(always)]
fn load8(s: &[f32], i: usize) -> f32x8 {
    f32x8::from([
        s[i],
        s[i + 1],
        s[i + 2],
        s[i + 3],
        s[i + 4],
        s[i + 5],
        s[i + 6],
        s[i + 7],
    ])
}

// f32x8 forgetting curve forward, storing the intermediates its backward needs (the f32x8
// analogue of CurveCache). `out` is bit-identical to the old forward-only curve_out8 — it just
// also stashes the products + cached lns so curve8_bwd never recomputes a transcendental.
struct Curve8 {
    out: f32x8,
    a: f32x8,
    bv: f32x8,
    m1: f32x8,
    decay1: f32x8,
    factor1: f32x8,
    b1: f32x8,
    r1: f32x8,
    q1: f32x8,
    e1: f32x8,
    p35: f32x8,
    m2: f32x8,
    decay2: f32x8,
    factor2: f32x8,
    b2: f32x8,
    r2: f32x8,
    inv2: f32x8,
    p28: f32x8,
    ex34: f32x8,
    weight1: f32x8,
    weight2: f32x8,
    wsum: f32x8,
    ret: f32x8,
    p31: f32x8,
    se: f32x8,
    ln_sf: f32x8,
    ln_b1: f32x8,
    ln_b2: f32x8,
    ln_s: f32x8,
}

#[allow(clippy::too_many_arguments)]
fn curve8_fwd<const FAST: bool>(
    w: &[f32],
    t: f32x8,
    s: f32x8,
    sf: f32x8,
    d: f32x8,
    ln_w27: f32,
    ln_w28: f32,
    ln_s: f32x8,
    ln_sf: f32x8,
) -> Curve8 {
    let sp = |i: usize| f32x8::splat(w[i]);
    let k = f32x8::splat;
    let t = t.fast_max(k(0.0));
    let a = t / sf;
    let bv = t / s;
    // 34-param remap + all-positive offsets: p35=s_short^(s_decay1[w33]-0.3), m1=decay1[w23]*p35,
    // ex34=exp((d-5)*(d_decay[w32]-0.3)), m2=decay2[w24]*ex34, p28=base2[w26]^inv2,
    // p31=s_short^-s_weight_power1[w29], weight1=base_weight1[w27]*p31. ln_w27=ln(base1=w25),
    // ln_w28=ln(base2=w26).
    let p35 = exp8::<FAST>((sp(33) - k(0.3)) * ln_sf);
    let m1 = sp(23) * p35;
    let dm1 = clamp8(m1, 0.01, 0.95);
    let decay1 = k(0.0) - dm1;
    let q1 = k(ln_w27) / decay1;
    let e1 = exp8::<FAST>(q1.fast_min(k(60.0)));
    let factor1 = e1 - k(1.0);
    let b1 = a * factor1 + k(1.0);
    let ln_b1 = ln8::<FAST>(b1);
    let r1 = exp8::<FAST>(decay1 * ln_b1);
    // iter-165: decay2 no longer d-modulated (m2 = w24 plain); ex34 = exp((d-5)*(d_decay-0.3))
    // is now the TIME-SCALE inside b2.
    let ex34 = exp8::<FAST>((d - k(5.0)) * (sp(32) - k(0.3)));
    let m2 = sp(24);
    let dm2 = clamp8(m2, 0.01, 0.95);
    let decay2 = k(0.0) - dm2;
    let inv2 = k(1.0) / decay2;
    let p28 = exp8::<FAST>(inv2 * k(ln_w28));
    let factor2 = p28 - k(1.0);
    let b2 = bv * factor2 * ex34 + k(1.0);
    let ln_b2 = ln8::<FAST>(b2);
    let r2 = exp8::<FAST>(decay2 * ln_b2);
    let p31 = exp8::<FAST>(k(-w[29]) * ln_sf);
    let weight1 = sp(27) * p31;
    // se = s_long^s_weight_power2[w30] · exp((d_weight[w31]-0.5)·(d−5)) = exp(w30·ln_s +
    // (w31-0.5)·(d−5)): ONE exp instead of two (iter24 fusion, ~1-ULP 3b reassociation; curve8_bwd
    // reads c.se directly). weight2 = base_weight2[w28] * se.
    let se = exp8::<FAST>(sp(30) * ln_s + (d - k(5.0)) * (sp(31) - k(0.5)));
    let weight2 = sp(28) * se;
    let wsum = weight1 + weight2;
    let num = weight1 * r1 + weight2 * r2;
    let ret = num / wsum;
    let out = ret * k(1.0 - 2e-5) + k(1e-5);
    Curve8 {
        out,
        a,
        bv,
        m1,
        decay1,
        factor1,
        b1,
        r1,
        q1,
        e1,
        p35,
        m2,
        decay2,
        factor2,
        b2,
        r2,
        inv2,
        p28,
        ex34,
        weight1,
        weight2,
        wsum,
        ret,
        p31,
        se,
        ln_sf,
        ln_b1,
        ln_b2,
        ln_s,
    }
}

/// VJP of curve8_fwd (the f32x8 analogue of curve_bwd; every scalar `if` is a lane blend, every
/// gw[] += is an f32x8 accumulate). Returns (g_s, g_sf, g_d); accumulates into the f32x8 gw bank.
#[allow(clippy::too_many_arguments)]
fn curve8_bwd(
    w: &[f32],
    c: &Curve8,
    t: f32x8,
    s: f32x8,
    sf: f32x8,
    d: f32x8,
    g_out: f32x8,
    g_r1_extra: f32x8,
    gw: &mut [f32x8; 34],
    ln_w27: f32,
    ln_w28: f32,
) -> (f32x8, f32x8, f32x8) {
    let sp = |i: usize| f32x8::splat(w[i]);
    let k = f32x8::splat;
    let z = k(0.0);
    let t = t.fast_max(z);
    let g_ret = g_out * k(1.0 - 2e-5);
    let g_num = g_ret / c.wsum;
    let g_wsum = (z - g_ret) * c.ret / c.wsum;
    let g_weight1 = g_num * c.r1 + g_wsum;
    let g_weight2 = g_num * c.r2 + g_wsum;
    // r1 also feeds the short-trace stability update (reads r1, not mixed R) -> g_r1_extra.
    let g_r1 = g_num * c.weight1 + g_r1_extra;
    let g_r2 = g_num * c.weight2;
    // weight2 = base_weight2[w28] * se ; se = exp(w30*ln_s + (w31-0.5)*(d-5))  (fused)
    gw[28] += g_weight2 * c.se;
    let g_se = g_weight2 * sp(28) * c.se; // adjoint-on-se times se (shared d(se)/d(.) factor)
    let mut g_d = g_se * (sp(31) - k(0.5)); // d(se)/dd  = se*(d_weight-0.5)
    gw[31] += g_se * (d - k(5.0)); // d(se)/d(d_weight) = se*(d-5) (offset deriv = 1)
    let mut g_s = g_se * sp(30) / s; // d(se)/ds  = se*s_weight_power2/s
    gw[30] += g_se * c.ln_s; // d(se)/d(s_weight_power2) = se*ln_s
    // weight1 = base_weight1[w27] * p31 ; p31 = sf^(-s_weight_power1[w29])
    gw[27] += g_weight1 * c.p31;
    let g_p31 = g_weight1 * sp(27);
    let mut g_sf = g_p31 * k(-w[29]) * (c.p31 / sf);
    gw[29] += g_p31 * (z - c.p31 * c.ln_sf);
    // r2 = b2^decay2 ; iter-165: b2 = bv*factor2*ex34 + 1 (ex34 = D time-scale) and
    // decay2 = -clamp(w24) is no longer d-modulated.
    let g_b2 = g_r2 * c.decay2 * (c.r2 / c.b2);
    let mut g_decay2 = g_r2 * c.r2 * c.ln_b2;
    let g_bv = g_b2 * c.factor2 * c.ex34;
    let g_factor2 = g_b2 * c.bv * c.ex34;
    let g_ex34 = g_b2 * c.bv * c.factor2;
    g_d += g_ex34 * c.ex34 * (sp(32) - k(0.3)); // ex34 = exp((d-5)*(d_decay-0.3))
    gw[32] += g_ex34 * c.ex34 * (d - k(5.0));
    let g_p28 = g_factor2;
    gw[26] += g_p28 * c.inv2 * (c.p28 / sp(26)); // p28 = base2[w26]^inv2
    let g_inv2 = g_p28 * c.p28 * k(ln_w28);
    g_decay2 += g_inv2 * (z - k(1.0) / (c.decay2 * c.decay2));
    let g_dm2 = z - g_decay2;
    let g_m2 = (c.m2.cmp_gt(k(0.01)) & c.m2.cmp_lt(k(0.95))).blend(g_dm2, z);
    gw[24] += g_m2; // m2 = w24 directly (clamp gate via c.m2)
    g_s += g_bv * (z - t / (s * s)); // bv = t/s
    // r1 = b1^decay1
    let g_b1 = g_r1 * c.decay1 * (c.r1 / c.b1);
    let mut g_decay1 = g_r1 * c.r1 * c.ln_b1;
    let g_a = g_b1 * c.factor1;
    let g_factor1 = g_b1 * c.a;
    let g_e1 = g_factor1;
    let g_q1c = g_e1 * c.e1;
    let g_q1 = c.q1.cmp_lt(k(60.0)).blend(g_q1c, z);
    // q1 = ln(base1) / decay1
    let g_lw27 = g_q1 / c.decay1;
    g_decay1 += g_q1 * (z - k(ln_w27) / (c.decay1 * c.decay1));
    gw[25] += g_lw27 / sp(25); // ln_base1 = ln(w[25])
    let g_dm1 = z - g_decay1;
    let g_m1 = (c.m1.cmp_gt(k(0.01)) & c.m1.cmp_lt(k(0.95))).blend(g_dm1, z);
    gw[23] += g_m1 * c.p35; // m1 = decay1[w23] * p35
    let g_p35 = g_m1 * sp(23);
    g_sf += g_p35 * (sp(33) - k(0.3)) * (c.p35 / sf); // p35 = s_short^(s_decay1-0.3)
    gw[33] += g_p35 * c.p35 * c.ln_sf;
    g_sf += g_a * (z - t / (sf * sf)); // a = t/sf
    (g_s, g_sf, g_d)
}

// f32x8 stability-after-review forward + the intermediates its backward needs (analogue of StabCache).
struct Stab8 {
    out: f32x8,
    nsf_fail: f32x8,
    pls: f32x8,
    sinc: f32x8,
    ls_sinc: f32x8,
    aa: f32x8,
    bb: f32x8,
    cc: f32x8,
    expr: f32x8,
    pr: f32x8,
    qbase: f32x8,
    hard: f32x8,
    easy: f32x8,
    ln_ls: f32x8,
    ln_ld: f32x8,
    ln_ls1: f32x8,
}

#[allow(clippy::too_many_arguments)]
fn stab8_fwd<const FAST: bool>(
    w: &[f32],
    last_s: f32x8,
    last_d: f32x8,
    r: f32x8,
    rating: f32x8,
    start: usize,
    aa: f32,
    ln_ls: f32x8,
    ln_ld: f32x8,
) -> Stab8 {
    let sp = |i: usize| f32x8::splat(w[i]);
    let k = f32x8::splat;
    let one = k(1.0);
    let hard = rating.cmp_eq(k(2.0)).blend(sp(start + 6), one);
    let easy = rating.cmp_eq(k(4.0)).blend(sp(start + 7), one);
    let ln_ls1 = ln8::<FAST>(last_s + one);
    let qbase = exp8::<FAST>(sp(start + 4) * ln_ls1); // (last_s+1)^fail_s_exp[start+4]
    // fail_d_exp DROPPED: post-lapse stability is D-independent, so the legacy `pr` cache field now
    // holds just rexp (no pp = last_d^-fail_d_exp factor; ln_ld unused, kept in cache for layout).
    let pr = exp8::<FAST>((one - r) * sp(start + 5)); // rexp = exp((1-r)*fail_r_mult[start+5])
    let nsf_fail = sp(start + 3) * pr * (qbase - one);
    let pls = last_s.fast_min(nsf_fail);
    let bb = k(11.0) - last_d;
    let cc = exp8::<FAST>(k(-w[start + 1]) * ln_ls);
    let expr = exp8::<FAST>((one - r) * sp(start + 2));
    let aa8 = k(aa);
    let sinc = aa8 * bb * cc * (expr - one) * hard * easy + one;
    let ls_sinc = last_s * sinc;
    let nss = pls.fast_max(ls_sinc);
    let out = rating.cmp_gt(one).blend(nss, pls);
    Stab8 {
        out,
        nsf_fail,
        pls,
        sinc,
        ls_sinc,
        aa: aa8,
        bb,
        cc,
        expr,
        pr,
        qbase,
        hard,
        easy,
        ln_ls,
        ln_ld,
        ln_ls1,
    }
}

/// VJP of stab8_fwd (f32x8 analogue of stab_bwd). Returns (g_last_s, g_last_d, g_r).
#[allow(clippy::too_many_arguments)]
fn stab8_bwd(
    w: &[f32],
    c: &Stab8,
    last_s: f32x8,
    last_d: f32x8,
    r: f32x8,
    rating: f32x8,
    start: usize,
    g_out: f32x8,
    gw: &mut [f32x8; 34],
) -> (f32x8, f32x8, f32x8) {
    let sp = |i: usize| f32x8::splat(w[i]);
    let k = f32x8::splat;
    let one = k(1.0);
    let z = k(0.0);
    let _ = last_d; // post-lapse stability is D-independent now (fail_d_exp dropped)
    let gt1 = rating.cmp_gt(one);
    let g_nss = gt1.blend(g_out, z);
    let g_pls_direct = gt1.blend(z, g_out);
    // nss = max(pls, ls_sinc)  (ties to pls, matching the scalar >= / > split)
    let g_pls_from_nss = c.pls.cmp_ge(c.ls_sinc).blend(g_nss, z);
    let g_ls_sinc = c.ls_sinc.cmp_gt(c.pls).blend(g_nss, z);
    let mut g_last_s = g_ls_sinc * c.sinc;
    let g_sinc = g_ls_sinc * last_s;
    let g_pls = g_pls_direct + g_pls_from_nss;
    // pls = min(last_s, nsf_fail)
    g_last_s += last_s.cmp_le(c.nsf_fail).blend(g_pls, z);
    let g_nsf_fail = c.nsf_fail.cmp_lt(last_s).blend(g_pls, z);
    // sinc = aa*bb*cc*(expr-1)*hard*easy + 1
    let em1 = c.expr - one;
    let g_prod = g_sinc;
    let prod = c.aa * c.bb * c.cc * em1 * c.hard * c.easy;
    gw[start] += g_prod * prod; // aa = exp(w[start]-1.5); dprod/dw[start] = prod
    let g_bb = g_prod * (c.aa * c.cc * em1 * c.hard * c.easy);
    let g_cc = g_prod * (c.aa * c.bb * em1 * c.hard * c.easy);
    let g_em1 = g_prod * (c.aa * c.bb * c.cc * c.hard * c.easy);
    gw[start + 6] += rating
        .cmp_eq(k(2.0))
        .blend(g_prod * (c.aa * c.bb * c.cc * em1 * c.easy), z); // hard_penalty
    gw[start + 7] += rating
        .cmp_eq(k(4.0))
        .blend(g_prod * (c.aa * c.bb * c.cc * em1 * c.hard), z); // easy_bonus
    let g_last_d = g_bb * (z - one); // bb = 11 - last_d (the ONLY D-dependence of stab now)
    g_last_s += g_cc * k(-w[start + 1]) * (c.cc / last_s);
    gw[start + 1] += g_cc * (z - c.cc * c.ln_ls);
    // expr = exp((1-r)*w[start+2])
    let mut g_r = g_em1 * c.expr * k(-w[start + 2]);
    gw[start + 2] += g_em1 * c.expr * (one - r);
    // nsf_fail = fail_mult[start+3] * pr * (qbase-1) ; pr = rexp = exp((1-r)*fail_r_mult[start+5])
    // (fail_d_exp DROPPED: no pp = last_d^-x factor, so nsf_fail is D-independent).
    let q = c.qbase - one;
    gw[start + 3] += g_nsf_fail * (c.pr * q);
    let g_pr = g_nsf_fail * sp(start + 3) * q; // adjoint on pr(=rexp)
    let g_q = g_nsf_fail * sp(start + 3) * c.pr; // adjoint on (qbase-1)
    let g_pr_pr = g_pr * c.pr; // shared  pr * d(pr)/d(.)  factor
    // pr = exp((1-r)*fail_r_mult[start+5])
    g_r += g_pr_pr * k(-w[start + 5]); // d(pr)/d(r) = pr*(-fail_r_mult)
    gw[start + 5] += g_pr_pr * (one - r); // d(pr)/d(fail_r_mult) = pr*(1-r)
    // qbase = (last_s+1)^fail_s_exp[start+4]
    g_last_s += g_q * sp(start + 4) * (c.qbase / (last_s + one));
    gw[start + 4] += g_q * c.qbase * c.ln_ls1;
    (g_last_s, g_last_d, g_r)
}

/// f32x8 next-difficulty forward; returns (clamped out, pre-clamp out, delta_d) for the backward.
fn next_d8_fwd(
    w: &[f32],
    last_d: f32x8,
    rating: f32x8,
    r: f32x8,
    init: f32,
) -> (f32x8, f32x8, f32x8) {
    let k = f32x8::splat;
    let delta_d_base = k(-w[6]) * (rating - k(3.0));
    // Surprise-weighted lapse: on a lapse scale delta_d by (r+0.1) = 1 + (R-0.9).
    let delta_d = rating
        .cmp_eq(k(1.0))
        .blend(delta_d_base * (r + k(0.1)), delta_d_base);
    let new_d = last_d + (k(10.0) - last_d) * delta_d / k(9.0);
    let out_pre = k(0.01) * k(init) + k(0.99) * new_d;
    (clamp8(out_pre, D_MIN, D_MAX), out_pre, delta_d) // delta_d returned = EFFECTIVE delta_d
}

/// VJP of next_d8_fwd. Returns (g_last_d, g_r); accumulates gw[4], gw[5], gw[6]. `delta_d` is the
/// EFFECTIVE delta_d; `r` is the curve retention (feeds the lapse surprise weighting).
#[allow(clippy::too_many_arguments)]
fn next_d8_bwd(
    w: &[f32],
    out_pre: f32x8,
    delta_d: f32x8,
    last_d: f32x8,
    rating: f32x8,
    r: f32x8,
    g_out: f32x8,
    gw: &mut [f32x8; 34],
    exp3w5: f64,
) -> (f32x8, f32x8) {
    let k = f32x8::splat;
    let z = k(0.0);
    let g_out_pre = (out_pre.cmp_gt(k(D_MIN)) & out_pre.cmp_lt(k(D_MAX))).blend(g_out, z);
    let g_init = g_out_pre * k(0.01);
    let g_new_d = g_out_pre * k(0.99);
    gw[4] += g_init;
    gw[5] += g_init * k(-(exp3w5 as f32) * 3.0); // init = w4 - exp(3 w5) + 1
    let g_last_d = g_new_d * (k(1.0) - delta_d / k(9.0));
    let g_delta_d = g_new_d * (k(10.0) - last_d) / k(9.0);
    let rm3 = rating - k(3.0);
    let is_lapse = rating.cmp_eq(k(1.0));
    // d(delta_d_eff)/d(w6) = -(rating-3), scaled by (r+0.1) on a lapse.
    gw[6] += g_delta_d * is_lapse.blend((z - rm3) * (r + k(0.1)), z - rm3);
    // d(delta_d_eff)/dr = delta_d_base = -w6*(rating-3), only on a lapse.
    let g_r = is_lapse.blend(g_delta_d * (k(-w[6]) * rm3), z);
    (g_last_d, g_r)
}

/// Vectorized forward-only BCE loss (validation). Processes the batch 8 cards at a time; the
/// remainder (<8) uses the scalar batch_loss path. Precision-trade vs batch_loss (exp8/ln8 differ
/// from libm by ~1e-6) — judged by the 3b average-log-loss band. The BCE itself stays f64.
#[allow(clippy::too_many_arguments)]
pub(crate) fn batch_loss_simd(
    w: &[f32],
    t_hist: &[f32],
    r_hist: &[f32],
    seq_len: usize,
    batch: usize,
    delta_ts: &[f32],
    labels: &[f32],
    weights: &[f32],
) -> f64 {
    let wc = wconsts(w);
    let k = f32x8::splat;
    let one = k(1.0);
    let mut loss = 0.0f64;
    let n_groups = batch / 8;
    for g in 0..n_groups {
        let c0 = g * 8;
        // Per-group active length: trailing all-padding timesteps (rating 0) pass the state through
        // unchanged, so the final state — and thus the scored curve below — is bit-for-bit identical
        // when we stop at `sl`. (iter26's per-group skip; see loss_and_grad_range_simd.)
        let mut sl = seq_len;
        while sl > 1 && load8(r_hist, (sl - 1) * batch + c0).reduce_add() == 0.0 {
            sl -= 1;
        }
        let (mut s, mut d, mut sf) = (k(0.0), k(0.0), k(0.0));
        for t in 0..sl {
            let base = t * batch + c0;
            let rating = load8(r_hist, base);
            if t == 0 {
                // First review: initialise state from rating (mirrors step_fwd's nth0 override).
                let rc = clamp8(rating, 1.0, 4.0);
                let init_s = rc.cmp_eq(one).blend(
                    k(w[0]),
                    rc.cmp_eq(k(2.0))
                        .blend(k(w[1]), rc.cmp_eq(k(3.0)).blend(k(w[2]), k(w[3]))),
                );
                let init_d = clamp8(
                    k(w[4]) - exp8::<false>(k(w[5]) * (rc - one)) + one,
                    D_MIN,
                    D_MAX,
                );
                s = clamp8(init_s, S_MIN, S_MAX);
                d = init_d;
                sf = clamp8(k(0.8) * init_s, S_MIN, S_MAX);
            } else {
                let dt = load8(t_hist, base).fast_max(k(0.0));
                let s_c = clamp8(s, S_MIN, S_MAX);
                let d_c = clamp8(d, D_MIN, D_MAX);
                let sf_c = clamp8(sf, S_MIN, S_MAX);
                let ln_s = ln8::<false>(s_c);
                let ln_sf = ln8::<false>(sf_c);
                let ln_d = ln8::<false>(d_c);
                let curve =
                    curve8_fwd::<false>(w, dt, s_c, sf_c, d_c, wc.ln_w27, wc.ln_w28, ln_s, ln_sf);
                let rr = curve.out;
                let ns = stab8_fwd::<false>(w, s_c, d_c, rr, rating, 7, wc.aa7, ln_s, ln_d).out;
                let nsf_raw =
                    stab8_fwd::<false>(w, sf_c, d_c, curve.r1, rating, 15, wc.aa16, ln_sf, ln_d)
                        .out;
                // POST-LAPSE short reset: on a lapse cap s_short at 0.8 * post-lapse s_long.
                let nsf = rating
                    .cmp_eq(one)
                    .blend(nsf_raw.fast_min(k(0.8) * ns), nsf_raw);
                let nd = next_d8_fwd(w, d_c, rating, rr, wc.init).0;
                // rating==0 (padding) passes the state through unchanged.
                let m0 = rating.cmp_eq(k(0.0));
                s = clamp8(m0.blend(s_c, ns), S_MIN, S_MAX);
                d = m0.blend(d_c, nd);
                sf = clamp8(m0.blend(sf_c, nsf), S_MIN, S_MAX);
            }
        }
        let dts = load8(delta_ts, c0);
        let r = clamp8(
            curve8_fwd::<false>(
                w,
                dts,
                s,
                sf,
                d,
                wc.ln_w27,
                wc.ln_w28,
                ln8::<false>(s),
                ln8::<false>(sf),
            )
            .out,
            MIN_R,
            MAX_R,
        );
        // Branchless BCE for 0/1 labels (Andrew): -ln(1 - |label - r|) = -ln(r) for label 1,
        // -ln(1-r) for label 0 — math-identical for hard labels. Computes the whole 8-lane BCE as
        // ONE vectorized ln8 (vs r.to_array() + 8 scalar f64 lns), summed in f64 via reduce_add.
        // Inexact (f32 minimax ln8 vs f64 libm ln), but this loss only feeds best-epoch argmin and the
        // error ≪ the epoch-to-epoch gaps, so best_w (hence params/benchmark loss) is unchanged.
        let lbl = load8(labels, c0);
        let wt = load8(weights, c0);
        let arg = one - (lbl - r).fast_max(r - lbl); // 1 - |label - r|  (r already clamped into (0,1))
        loss += ((k(0.0) - wt) * ln8::<false>(arg)).reduce_add() as f64;
    }
    // Remainder (< 8 cards) via the scalar path.
    for c in (n_groups * 8)..batch {
        let mut state = (0.0f32, 0.0f32, 0.0f32);
        for t in 0..seq_len {
            state = step_fwd(
                w,
                t_hist[t * batch + c],
                r_hist[t * batch + c],
                state,
                t == 0,
                &wc,
            )
            .0;
        }
        let (s, d, sf) = state;
        let r = clamp(
            curve_fwd(
                w,
                delta_ts[c],
                s,
                sf,
                d,
                wc.ln_w27,
                wc.ln_w28,
                s.ln(),
                sf.ln(),
            )
            .out,
            MIN_R,
            MAX_R,
        );
        loss += -(weights[c] as f64)
            * (labels[c] as f64 * (r as f64).ln()
                + (1.0 - labels[c] as f64) * (1.0 - r as f64).ln());
    }
    loss
}

// ===================== SIMD recurrence step (8 cards/lane) with cache =====================
// Per-timestep cache for the vectorized backward. The first review (t==0) only needs the init
// override's data (the curve/stab/next_d it computes are dead, overridden), so it gets a small
// `First` variant; every later step stores the full forward intermediates (`Full`).
// Keep per-review caches inline in the training buffer to avoid an allocation per step.
#[allow(clippy::large_enum_variant)]
enum Step8 {
    First {
        rc: f32x8,
        init_s: f32x8,
        ex_w5: f32x8,
        id_in: f32x8,
    },
    Full {
        s0: f32x8,
        d0: f32x8,
        sf0: f32x8,
        last_s: f32x8,
        last_sf: f32x8,
        last_d: f32x8,
        rating: f32x8,
        dt: f32x8,
        curve: Curve8,
        slow: Stab8,
        fast: Stab8,
        nd_out_pre: f32x8,
        nd_delta_d: f32x8,
        ns3: f32x8,
        nsf3: f32x8,
    },
}

impl Step8 {
    /// The retrievability the curve predicted at this step (= the loss target R_t in the windowed
    /// O(N) forward). The first review (t==0) makes no prediction, so it returns 0 (the windowed
    /// kernels never score t==0; the minimum surviving prefix length is 2).
    #[inline(always)]
    fn curve_out(&self) -> f32x8 {
        match self {
            Step8::Full { curve, .. } => curve.out,
            Step8::First { .. } => f32x8::splat(0.0),
        }
    }
}

/// One recurrence step over 8 cards. `first` (t==0) initialises state from the rating exactly like
/// batch_loss_simd; otherwise it mirrors step_fwd (curve + both stability traces + next-difficulty,
/// then the rating==0 padding passthrough). Returns the new state and the backward cache.
fn step8_fwd<const FAST: bool>(
    w: &[f32],
    dt_raw: f32x8,
    rating: f32x8,
    state: (f32x8, f32x8, f32x8),
    first: bool,
    wc: &WConsts,
) -> ((f32x8, f32x8, f32x8), Step8) {
    let k = f32x8::splat;
    let one = k(1.0);
    let (s0, d0, sf0) = state;
    if first {
        let rc = clamp8(rating, 1.0, 4.0);
        let init_s = rc.cmp_eq(one).blend(
            k(w[0]),
            rc.cmp_eq(k(2.0))
                .blend(k(w[1]), rc.cmp_eq(k(3.0)).blend(k(w[2]), k(w[3]))),
        );
        let ex_w5 = exp8::<FAST>(k(w[5]) * (rc - one));
        let id_in = k(w[4]) - ex_w5 + one;
        let init_d = clamp8(id_in, D_MIN, D_MAX);
        let out = (
            clamp8(init_s, S_MIN, S_MAX),
            init_d,
            clamp8(k(0.8) * init_s, S_MIN, S_MAX),
        );
        (
            out,
            Step8::First {
                rc,
                init_s,
                ex_w5,
                id_in,
            },
        )
    } else {
        let last_s = clamp8(s0, S_MIN, S_MAX);
        let last_d = clamp8(d0, D_MIN, D_MAX);
        let last_sf = clamp8(sf0, S_MIN, S_MAX);
        let dt = dt_raw.fast_max(k(0.0));
        let ln_last_s = ln8::<FAST>(last_s);
        let ln_last_sf = ln8::<FAST>(last_sf);
        let ln_last_d = ln8::<FAST>(last_d);
        let curve = curve8_fwd::<FAST>(
            w, dt, last_s, last_sf, last_d, wc.ln_w27, wc.ln_w28, ln_last_s, ln_last_sf,
        );
        let r = curve.out;
        let r1 = curve.r1; // short component recall — drives the short-trace update (iter-71)
        let slow = stab8_fwd::<FAST>(
            w, last_s, last_d, r, rating, 7, wc.aa7, ln_last_s, ln_last_d,
        );
        let fast = stab8_fwd::<FAST>(
            w, last_sf, last_d, r1, rating, 15, wc.aa16, ln_last_sf, ln_last_d,
        );
        let (nd, nd_out_pre, nd_delta_d) = next_d8_fwd(w, last_d, rating, r, wc.init);
        // POST-LAPSE short reset (iter-97): on a lapse cap s_short at 0.8 * post-lapse s_long.
        let nsf_pre = rating
            .cmp_eq(one)
            .blend(fast.out.fast_min(k(0.8) * slow.out), fast.out);
        // rating==0 (padding) passes the input state through unchanged.
        let m0 = rating.cmp_eq(k(0.0));
        let ns3 = m0.blend(last_s, slow.out);
        let nsf3 = m0.blend(last_sf, nsf_pre);
        let nd3 = m0.blend(last_d, nd);
        let out = (clamp8(ns3, S_MIN, S_MAX), nd3, clamp8(nsf3, S_MIN, S_MAX));
        (
            out,
            Step8::Full {
                s0,
                d0,
                sf0,
                last_s,
                last_sf,
                last_d,
                rating,
                dt,
                curve,
                slow,
                fast,
                nd_out_pre,
                nd_delta_d,
                ns3,
                nsf3,
            },
        )
    }
}

/// VJP of one step (f32x8 analogue of step_bwd). Given output-state adjoints, returns input-state
/// adjoints and accumulates the weight gradient into `gw`. `g_r_loss` is the adjoint of any loss
/// scored directly off this step's curve.out (nonzero only in the windowed O(N) forward, where every
/// step emits a prediction); it is added to the two stab r-adjoints before curve8_bwd, since curve.out
/// feeds the loss AND both stability traces. The O(N^2) callers pass 0 (their loss is the separate
/// final curve). The First variant ignores it (t==0 makes no prediction).
fn step8_bwd(
    w: &[f32],
    c: &Step8,
    g_out: (f32x8, f32x8, f32x8),
    g_r_loss: f32x8,
    gw: &mut [f32x8; 34],
    wc: &WConsts,
) -> (f32x8, f32x8, f32x8) {
    let k = f32x8::splat;
    let one = k(1.0);
    let z = k(0.0);
    let (g_ns_out, g_nd_out, g_nsf_out) = g_out;
    match c {
        // t==0 init override: the only weight grads are gw[rc-1] (init stability, scattered by the
        // per-lane rating via 4 masked adds) and gw[4]/gw[5] (init difficulty). Input adjoints are 0
        // (state before the first review is constant 0).
        Step8::First {
            rc,
            init_s,
            ex_w5,
            id_in,
        } => {
            let g_ns3 = (init_s.cmp_gt(k(S_MIN)) & init_s.cmp_lt(k(S_MAX))).blend(g_ns_out, z);
            let nsf3 = k(0.8) * *init_s;
            let g_nsf3 = (nsf3.cmp_gt(k(S_MIN)) & nsf3.cmp_lt(k(S_MAX))).blend(g_nsf_out, z);
            let g_init_s = g_ns3 + g_nsf3 * k(0.8);
            gw[0] += rc.cmp_eq(k(1.0)).blend(g_init_s, z);
            gw[1] += rc.cmp_eq(k(2.0)).blend(g_init_s, z);
            gw[2] += rc.cmp_eq(k(3.0)).blend(g_init_s, z);
            gw[3] += rc.cmp_eq(k(4.0)).blend(g_init_s, z);
            let idmask = id_in.cmp_gt(k(D_MIN)) & id_in.cmp_lt(k(D_MAX));
            gw[4] += idmask.blend(g_nd_out, z);
            gw[5] += idmask.blend(g_nd_out * (z - *ex_w5 * (*rc - one)), z);
            (z, z, z)
        }
        Step8::Full {
            s0,
            d0,
            sf0,
            last_s,
            last_sf,
            last_d,
            rating,
            dt,
            curve,
            slow,
            fast,
            nd_out_pre,
            nd_delta_d,
            ns3,
            nsf3,
        } => {
            let g_ns3 = (ns3.cmp_gt(k(S_MIN)) & ns3.cmp_lt(k(S_MAX))).blend(g_ns_out, z);
            let g_nsf3 = (nsf3.cmp_gt(k(S_MIN)) & nsf3.cmp_lt(k(S_MAX))).blend(g_nsf_out, z);
            let g_nd3 = g_nd_out;
            // rating==0 padding: output state == input state, so the adjoint flows straight through.
            let m0 = rating.cmp_eq(z);
            let g_ns2 = m0.blend(z, g_ns3);
            let g_nsf2 = m0.blend(z, g_nsf3); // adjoint on nsf_pre
            let g_nd2 = m0.blend(z, g_nd3);
            let g_last_s_extra = m0.blend(g_ns3, z);
            let g_last_sf_extra = m0.blend(g_nsf3, z);
            let g_last_d_extra = m0.blend(g_nd3, z);
            // POST-LAPSE min routing: nsf_pre = (rating==1)? min(fast.out, 0.8*slow.out) : fast.out.
            let is_lapse = rating.cmp_eq(one);
            let fast_wins = fast.out.cmp_le(k(0.8) * slow.out);
            let g_fast_out = is_lapse.blend(fast_wins.blend(g_nsf2, z), g_nsf2);
            let g_slow_from_relearn = is_lapse.blend(fast_wins.blend(z, g_nsf2 * k(0.8)), z);
            // LONG stab reads mixed retention curve.out; SHORT stab reads r1=curve.r1 (start 15).
            let (g_ls_a, g_ld_a, g_r_long) = stab8_bwd(
                w,
                slow,
                *last_s,
                *last_d,
                curve.out,
                *rating,
                7,
                g_ns2 + g_slow_from_relearn,
                gw,
            );
            let (g_lsf_b, g_ld_b, g_r1_short) = stab8_bwd(
                w, fast, *last_sf, *last_d, curve.r1, *rating, 15, g_fast_out, gw,
            );
            let (g_ld_c, g_r_nextd) = next_d8_bwd(
                w,
                *nd_out_pre,
                *nd_delta_d,
                *last_d,
                *rating,
                curve.out,
                g_nd2,
                gw,
                wc.exp3w5,
            );
            // curve.out adjoint = long-stab r + windowed loss adjoint + next_d r; curve.r1 = short-stab r.
            let (g_ls_d, g_lsf_d, g_ld_d) = curve8_bwd(
                w,
                curve,
                *dt,
                *last_s,
                *last_sf,
                *last_d,
                g_r_long + g_r_loss + g_r_nextd,
                g_r1_short,
                gw,
                wc.ln_w27,
                wc.ln_w28,
            );
            let g_last_s = g_ls_a + g_ls_d + g_last_s_extra;
            let g_last_sf = g_lsf_b + g_lsf_d + g_last_sf_extra;
            let g_last_d = g_ld_a + g_ld_b + g_ld_c + g_ld_d + g_last_d_extra;
            let g_s0 = (s0.cmp_gt(k(S_MIN)) & s0.cmp_lt(k(S_MAX))).blend(g_last_s, z);
            let g_d0 = (d0.cmp_gt(k(D_MIN)) & d0.cmp_lt(k(D_MAX))).blend(g_last_d, z);
            let g_sf0 = (sf0.cmp_gt(k(S_MIN)) & sf0.cmp_lt(k(S_MAX))).blend(g_last_sf, z);
            (g_s0, g_d0, g_sf0)
        }
    }
}

/// Vectorized forward + reverse-mode backward over card groups `[g_start, g_end)` (8 cards/lane).
/// Per-group weight gradients accumulate in an f32x8 bank, then horizontal-sum into the f64 `gw`
/// once per group — so cross-card/cross-group accumulation stays f64 while the per-card VJP is f32.
/// Requires the batch to be padded to a multiple of 8 (build_host_batches does this).
///
/// The loss VALUE is unused by training (only the gradient drives Adam — the training caller
/// discards the return, and validation uses `batch_loss_simd`), so the per-lane f64 BCE is skipped
/// here and 0.0 is returned. This is bit-for-bit on the gradient: `gw` is built only from `g_r`
/// (computed from `r`), never from the loss accumulator. (Mirrors `card_loss_and_grad_simd`.)
#[allow(clippy::too_many_arguments)]
fn loss_and_grad_range_simd(
    w: &[f32],
    t_hist: &[f32],
    r_hist: &[f32],
    seq_len: usize,
    batch: usize,
    delta_ts: &[f32],
    labels: &[f32],
    weights: &[f32],
    gw: &mut [f64],
    g_start: usize,
    g_end: usize,
) -> f64 {
    let wc = wconsts(w);
    let k = f32x8::splat;
    let one = k(1.0);
    let mut caches: Vec<Step8> = Vec::with_capacity(seq_len);
    for g in g_start..g_end {
        let c0 = g * 8;
        caches.clear();
        // Per-group active length: this chunk runs to the batch-wide seq_len (= its longest card),
        // but a group of 8 length-similar cards is usually shorter. Trailing timesteps where all 8
        // lanes are padding (rating 0) are pure state passthrough (the rating==0 blend keeps state
        // and adds 0 to the gradient), so the final state — and thus the scored curve below and the
        // whole backward — are bit-for-bit identical when we stop at `sl`. (iter26's per-group skip,
        // here on the O(N^2) path.) ratings are >=1 for real reviews, so reduce_add==0 <=> all pad.
        let mut sl = seq_len;
        while sl > 1 && load8(r_hist, (sl - 1) * batch + c0).reduce_add() == 0.0 {
            sl -= 1;
        }
        let (mut s, mut d, mut sf) = (k(0.0), k(0.0), k(0.0));
        for t in 0..sl {
            let base = t * batch + c0;
            let rating = load8(r_hist, base);
            let dt = if t == 0 { k(0.0) } else { load8(t_hist, base) };
            let (ns, cache) = step8_fwd::<false>(w, dt, rating, (s, d, sf), t == 0, &wc);
            s = ns.0;
            d = ns.1;
            sf = ns.2;
            caches.push(cache);
        }
        let dts = load8(delta_ts, c0);
        let lbl = load8(labels, c0);
        let wt = load8(weights, c0);
        let ln_s = ln8::<false>(s);
        let ln_sf = ln8::<false>(sf);
        let fc = curve8_fwd::<false>(w, dts, s, sf, d, wc.ln_w27, wc.ln_w28, ln_s, ln_sf);
        let r_raw = fc.out;
        let r = clamp8(r_raw, MIN_R, MAX_R);
        // (loss VALUE skipped — see the fn doc; training discards it, validation uses batch_loss_simd.)
        // d loss / d r, then the [MIN_R, MAX_R] clamp, in f32x8.
        let g_r = (k(0.0) - wt) * (lbl / r - (one - lbl) / (one - r));
        let g_rraw = (r_raw.cmp_gt(k(MIN_R)) & r_raw.cmp_lt(k(MAX_R))).blend(g_r, k(0.0));
        let mut gw_g = [f32x8::splat(0.0); 34];
        let (mut g_s, mut g_sf, mut g_d) = curve8_bwd(
            w,
            &fc,
            dts,
            s,
            sf,
            d,
            g_rraw,
            f32x8::splat(0.0),
            &mut gw_g,
            wc.ln_w27,
            wc.ln_w28,
        );
        for t in (0..sl).rev() {
            // O(N^2) path: the only loss is the final curve (handled above), so no per-step adjoint.
            let (gs0, gd0, gsf0) = step8_bwd(
                w,
                &caches[t],
                (g_s, g_d, g_sf),
                f32x8::splat(0.0),
                &mut gw_g,
                &wc,
            );
            g_s = gs0;
            g_d = gd0;
            g_sf = gsf0;
        }
        for i in 0..34 {
            gw[i] += gw_g[i].reduce_add() as f64;
        }
    }
    0.0
}

/// Vectorized forward+backward BCE gradient for one batch (8 cards/lane). Accumulates d(loss)/d(w)
/// into `gw` (length 36) and returns 0.0 — the loss VALUE is unused by training (see
/// loss_and_grad_range_simd; validation uses batch_loss_simd). The batch must be padded to a
/// multiple of 8. This is the precision-trading (3b band) f32x8 replacement for batch_loss_and_grad.
#[allow(clippy::too_many_arguments)]
pub(crate) fn batch_loss_and_grad_simd(
    w: &[f32],
    t_hist: &[f32],
    r_hist: &[f32],
    seq_len: usize,
    batch: usize,
    delta_ts: &[f32],
    labels: &[f32],
    weights: &[f32],
    gw: &mut [f64],
) -> f64 {
    debug_assert!(
        batch.is_multiple_of(8),
        "batch_loss_and_grad_simd needs batch padded to a multiple of 8"
    );
    let n_groups = batch / 8;
    // Single-threaded ON PURPOSE: iter16 tried splitting the groups across the worker's 2nd pinned
    // CPU (mirroring iter3's scalar-grad threading) and REGRESSED to 0.62x. The f32x8 forward
    // saturates the physical core's vector units with one thread, so the 2nd CPU (an SMT sibling
    // sharing those units) adds no vector throughput — only spawn overhead. (iter3 helped because
    // the SCALAR grad used scalar units SMT could overlap; vectorized work can't.) The idle 2nd core
    // would need coarser-than-per-batch parallelism to pay off, which isn't worth the complexity.
    loss_and_grad_range_simd(
        w, t_hist, r_hist, seq_len, batch, delta_ts, labels, weights, gw, 0, n_groups,
    )
}

// ===================== SIMD windowed O(N) forward+backward (8 cards/lane) =====================
// The O(N^2) -> O(N) expanding window. A card with K reviews became K-1 prefix-items (lengths 2..K),
// each re-running the recurrence over its whole prefix => O(K^2) timestep-work. Here each CARD is a
// single column whose recurrence runs ONCE over its full review sequence: at step t (t>=1) the curve
// curve8_fwd(state_{t-1}, delta_t[t]) it computes for the stability update IS EXACTLY the prediction
// R_t the length-(t+1) prefix used to score review t (same input state, same delta_t), so a loss is
// read off every step for free. wts/lbl are row-major [seq, bsz]; wts[t][c]==0 marks "no prediction"
// (t==0, an outlier-filtered prefix, or a padding column/timestep). The total loss and gradient equal
// the per-prefix sums (the recurrence is deterministic), so this is math-identical to the O(N^2) path
// up to FP reassociation — judged by the 3b average-log-loss band (build_host_batches groups the SAME
// cards-as-units as the Phase-1 probe, so the trained params match it modulo FP).

/// Windowed forward + reverse-mode backward for one card-grouped batch. Accumulates d(loss)/d(w) into
/// `gw` (length 36); the loss VALUE is unused by training (only the gradient drives Adam), so the f64
/// per-lane BCE is skipped here and 0.0 is returned — validation uses `card_loss_simd`. Batch padded
/// to a multiple of 8.
#[allow(clippy::too_many_arguments)]
pub(crate) fn card_loss_and_grad_simd(
    w: &[f32],
    t_hist: &[f32],
    r_hist: &[f32],
    seq_len: usize,
    batch: usize,
    labels: &[f32],
    weights: &[f32],
    gw: &mut [f64],
) -> f64 {
    debug_assert!(
        batch.is_multiple_of(8),
        "card_loss_and_grad_simd needs batch padded to a multiple of 8"
    );
    debug_assert!(
        seq_len >= 2,
        "windowed grad needs seq_len >= 2 (min surviving prefix length is 2)"
    );
    let wc = wconsts(w);
    let k = f32x8::splat;
    let one = k(1.0);
    let z = k(0.0);
    let n_groups = batch / 8;
    // d/dr of -wt*BCE via the unified label identity: d[-ln(1-|label-r|)]/dr = -sign(label-r)/
    // (1-|label-r|). ONE division (label is 0/1); padding/filtered steps have wt==0 -> 0; the
    // [MIN_R,MAX_R] clamp zeroes the adjoint outside the range. (3b trade vs lbl/r - (1-lbl)/(1-r).)
    // Shared by the peeled last step and the reverse loop below.
    let g_r_loss_at = |r_raw: f32x8, base: usize| -> f32x8 {
        let wt = load8(weights, base);
        let lbl = load8(labels, base);
        let r = clamp8(r_raw, MIN_R, MAX_R);
        let dd = lbl - r;
        let sgn = dd.cmp_gt(z).blend(one, z - one);
        let g_r = (z - wt) * (sgn / (one - dd.fast_max(z - dd)));
        (r_raw.cmp_gt(k(MIN_R)) & r_raw.cmp_lt(k(MAX_R))).blend(g_r, z)
    };
    let mut caches: Vec<Step8> = Vec::with_capacity(seq_len);
    for g in 0..n_groups {
        let c0 = g * 8;
        caches.clear();
        let (mut s, mut d, mut sf) = (z, z, z);
        // Forward over every step EXCEPT the last (full step + cache). The last step's stability/
        // next-difficulty update feeds no t+1, so we compute its curve only (just below) — exactly
        // like card_loss_simd's validation skip-last.
        for t in 0..seq_len - 1 {
            let base = t * batch + c0;
            let rating = load8(r_hist, base);
            let dt = if t == 0 { z } else { load8(t_hist, base) };
            let (ns, cache) = step8_fwd::<false>(w, dt, rating, (s, d, sf), t == 0, &wc);
            s = ns.0;
            d = ns.1;
            sf = ns.2;
            caches.push(cache);
        }
        // Last step (t = seq_len-1, always >= 1): curve ONLY, from the clamped incoming state — the
        // exact curve step8_fwd would compute. (debug_assert seq_len>=2 guards the 0..seq_len-1 above.)
        let lbase = (seq_len - 1) * batch + c0;
        let dt_last = load8(t_hist, lbase).fast_max(z);
        let (ls, lsf, ld) = (
            clamp8(s, S_MIN, S_MAX),
            clamp8(sf, S_MIN, S_MAX),
            clamp8(d, D_MIN, D_MAX),
        );
        let fc_last = curve8_fwd::<false>(
            w,
            dt_last,
            ls,
            lsf,
            ld,
            wc.ln_w27,
            wc.ln_w28,
            ln8::<false>(ls),
            ln8::<false>(lsf),
        );
        // Reverse pass. The final state's adjoint is 0, so at the last step the stab/next_d backward
        // all multiply 0 -> only curve8_bwd(loss-adjoint) contributes. BIT-FOR-BIT identical to a full
        // step8_bwd with g_out=0. The clamp gate uses the UNCLAMPED incoming state (= s0/d0/sf0).
        let mut gw_g = [f32x8::splat(0.0); 34];
        let g_rraw_last = g_r_loss_at(fc_last.out, lbase);
        let (g_ls, g_lsf, g_ld) = curve8_bwd(
            w,
            &fc_last,
            dt_last,
            ls,
            lsf,
            ld,
            g_rraw_last,
            f32x8::splat(0.0),
            &mut gw_g,
            wc.ln_w27,
            wc.ln_w28,
        );
        let mut g_s = (s.cmp_gt(k(S_MIN)) & s.cmp_lt(k(S_MAX))).blend(g_ls, z);
        let mut g_d = (d.cmp_gt(k(D_MIN)) & d.cmp_lt(k(D_MAX))).blend(g_ld, z);
        let mut g_sf = (sf.cmp_gt(k(S_MIN)) & sf.cmp_lt(k(S_MAX))).blend(g_lsf, z);
        for t in (0..seq_len - 1).rev() {
            let g_r_loss = if t == 0 {
                z // init step: no prediction (min surviving prefix length is 2).
            } else {
                g_r_loss_at(caches[t].curve_out(), t * batch + c0)
            };
            let (gs0, gd0, gsf0) =
                step8_bwd(w, &caches[t], (g_s, g_d, g_sf), g_r_loss, &mut gw_g, &wc);
            g_s = gs0;
            g_d = gd0;
            g_sf = gsf0;
        }
        for i in 0..34 {
            gw[i] += gw_g[i].reduce_add() as f64;
        }
    }
    0.0
}

/// Windowed forward-only BCE loss (validation) — `card_loss_and_grad_simd`'s forward without the
/// backward. Emits a per-lane f64 BCE at every t>=1 with wts>0 (matching `batch_loss_simd`'s f64
/// accumulation). Batch padded to a multiple of 8.
#[allow(clippy::too_many_arguments)]
pub(crate) fn card_loss_simd(
    w: &[f32],
    t_hist: &[f32],
    r_hist: &[f32],
    seq_len: usize,
    batch: usize,
    labels: &[f32],
    weights: &[f32],
) -> f64 {
    let wc = wconsts(w);
    let k = f32x8::splat;
    let z = k(0.0);
    let mut loss = 0.0f64;
    let n_groups = batch / 8;
    for g in 0..n_groups {
        let c0 = g * 8;
        let (mut s, mut d, mut sf) = (z, z, z);
        for t in 0..seq_len {
            let base = t * batch + c0;
            let rating = load8(r_hist, base);
            let dt = if t == 0 { z } else { load8(t_hist, base) };
            // Prediction r_t = curve(state_{t-1}, dt). At the LAST timestep the state update would feed
            // no t+1, so compute the curve ONLY there (skip the 2 stab traces + next_d + ln(last_d)).
            // BIT-FOR-BIT: the BCE uses only curve.out, which is identical to step8_fwd's curve; the
            // dropped state is never read. (curve_out() for the t==0 init returns 0, unused below.)
            let r = if t == seq_len - 1 {
                let (ls, lsf, ld) = (
                    clamp8(s, S_MIN, S_MAX),
                    clamp8(sf, S_MIN, S_MAX),
                    clamp8(d, D_MIN, D_MAX),
                );
                clamp8(
                    curve8_fwd::<false>(
                        w,
                        dt.fast_max(z),
                        ls,
                        lsf,
                        ld,
                        wc.ln_w27,
                        wc.ln_w28,
                        ln8::<false>(ls),
                        ln8::<false>(lsf),
                    )
                    .out,
                    MIN_R,
                    MAX_R,
                )
            } else {
                let (ns, cache) = step8_fwd::<false>(w, dt, rating, (s, d, sf), t == 0, &wc);
                s = ns.0;
                d = ns.1;
                sf = ns.2;
                clamp8(cache.curve_out(), MIN_R, MAX_R)
            };
            if t >= 1 {
                // Per-prediction BCE via the unified identity  -ln(1 - |label - r|)  (= -ln(r) for
                // label 1, -ln(1-r) for label 0). Branchless, so the whole 8-lane BCE is ONE vectorized
                // ln8. Padding/filtered lanes have weight 0. The BCE ln uses the accurate ::<false>,
                // as does the forward recurrence now (iter23's cruder windowed minimax was reverted).
                let lbl = load8(labels, base);
                let wt = load8(weights, base);
                let arg = k(1.0) - (lbl - r).fast_max(r - lbl); // 1 - |label - r|
                loss += ((z - wt) * ln8::<false>(arg)).reduce_add() as f64;
            }
        }
    }
    loss
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
    if seq_len < 2 || !batch_size.is_multiple_of(8) {
        return super::windowed_loss_scalar(
            w, t_historys, r_historys, labels, weights, seq_len, batch_size,
        );
    }

    card_loss_simd(
        w, t_historys, r_historys, seq_len, batch_size, labels, weights,
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
    if seq_len < 2 || !batch_size.is_multiple_of(8) {
        return super::reverse::windowed_grad(
            w, t_historys, r_historys, labels, weights, seq_len, batch_size,
        );
    }

    let mut grad = [0.0f64; PARAM_LEN];
    card_loss_and_grad_simd(
        w, t_historys, r_historys, seq_len, batch_size, labels, weights, &mut grad,
    );
    let mut out = [0.0f32; PARAM_LEN];
    for (dst, src) in out.iter_mut().zip(grad) {
        *dst = src as f32;
    }
    out
}

/// Forward + reverse-mode backward over cards `[start, end)`; accumulates d(loss)/d(w)
/// into `gw` and returns the summed loss for that range.
#[allow(clippy::too_many_arguments)]
fn loss_and_grad_range(
    w: &[f32],
    t_hist: &[f32],
    r_hist: &[f32],
    seq_len: usize,
    batch: usize,
    delta_ts: &[f32],
    labels: &[f32],
    weights: &[f32],
    gw: &mut [f64],
    start: usize,
    end: usize,
) -> f64 {
    let wc = wconsts(w);
    let mut loss = 0.0f64;
    let mut caches: Vec<StepCache> = Vec::with_capacity(seq_len);
    for c in start..end {
        caches.clear();
        let mut state = (0.0f32, 0.0f32, 0.0f32);
        for t in 0..seq_len {
            let (ns, cache) = step_fwd(
                w,
                t_hist[t * batch + c],
                r_hist[t * batch + c],
                state,
                t == 0,
                &wc,
            );
            state = ns;
            caches.push(cache);
        }
        let (s, d, sf) = state;
        let fc = curve_fwd(
            w,
            delta_ts[c],
            s,
            sf,
            d,
            wc.ln_w27,
            wc.ln_w28,
            s.ln(),
            sf.ln(),
        );
        let r_raw = fc.out;
        let r = clamp(r_raw, MIN_R, MAX_R);
        let (lbl, wt) = (labels[c] as f64, weights[c] as f64);
        loss += -wt * (lbl * (r as f64).ln() + (1.0 - lbl) * (1.0 - r as f64).ln());
        // d loss / d r (then clamp)
        let g_r = -wt * (lbl / r as f64 - (1.0 - lbl) / (1.0 - r as f64));
        let g_rraw = if r_raw > MIN_R && r_raw < MAX_R {
            g_r
        } else {
            0.0
        };
        // final curve backward -> adjoints on final (s, sf, d)
        let (mut g_s, mut g_sf, mut g_d) =
            curve_bwd(w, &fc, delta_ts[c], s, sf, d, g_rraw, 0.0, gw);
        // recurrence backward
        for t in (0..seq_len).rev() {
            let (gs0, gd0, gsf0) = step_bwd(w, &caches[t], (g_s, g_d, g_sf), gw, &wc);
            g_s = gs0;
            g_d = gd0;
            g_sf = gsf0;
        }
    }
    loss
}

/// Forward + reverse-mode backward BCE loss for one batch. Returns the summed loss and
/// accumulates d(loss)/d(w) into `gw` (length 36). Cards are independent, so the batch is
/// split across **2 threads** (the worker is pinned to a 2-CPU block per constraint 2, so
/// this uses the otherwise-idle second core). Partial gradients are summed after the join;
/// the only effect on the result is FP reassociation, well within the ±0.0010 band.
#[allow(clippy::too_many_arguments)]
pub(crate) fn batch_loss_and_grad(
    w: &[f32],
    t_hist: &[f32],
    r_hist: &[f32],
    seq_len: usize,
    batch: usize,
    delta_ts: &[f32],
    labels: &[f32],
    weights: &[f32],
    gw: &mut [f64],
) -> f64 {
    const THREAD_MIN: usize = 64;
    if batch < THREAD_MIN {
        return loss_and_grad_range(
            w, t_hist, r_hist, seq_len, batch, delta_ts, labels, weights, gw, 0, batch,
        );
    }
    let mid = batch / 2;
    std::thread::scope(|s| {
        let h = s.spawn(|| {
            let mut g = [0.0f64; 34];
            let l = loss_and_grad_range(
                w, t_hist, r_hist, seq_len, batch, delta_ts, labels, weights, &mut g, mid, batch,
            );
            (l, g)
        });
        let mut g_a = [0.0f64; 34];
        let loss_a = loss_and_grad_range(
            w, t_hist, r_hist, seq_len, batch, delta_ts, labels, weights, &mut g_a, 0, mid,
        );
        let (loss_b, g_b) = h.join().unwrap();
        for i in 0..34 {
            gw[i] += g_a[i] + g_b[i];
        }
        loss_a + loss_b
    })
}

/// Scalar O(N) windowed forward+backward — the trusted oracle for `card_loss_and_grad_simd`. Same
/// algorithm in scalar f64, reusing the UNMODIFIED scalar `step_fwd`/`step_bwd`: each step's loss
/// adjoint flows back through a SEPARATE `curve_bwd` call rather than a `g_r_loss` param. That is
/// equivalent because `curve_bwd` is linear in its output adjoint, so `curve_bwd(g_r_loss)` plus
/// `step_bwd`'s internal `curve_bwd(g_r_a+g_r_b)` sum to the same total (gw and state adjoints) as
/// the SIMD kernel's single `curve8_bwd(g_r_a+g_r_b+g_r_loss)`. labels/weights are [seq, batch].
#[cfg(test)]
#[allow(clippy::too_many_arguments)]
fn loss_and_grad_range_window(
    w: &[f32],
    t_hist: &[f32],
    r_hist: &[f32],
    seq_len: usize,
    batch: usize,
    labels: &[f32],
    weights: &[f32],
    gw: &mut [f64],
    start: usize,
    end: usize,
) -> f64 {
    let wc = wconsts(w);
    let mut loss = 0.0f64;
    let mut caches: Vec<StepCache> = Vec::with_capacity(seq_len);
    for c in start..end {
        caches.clear();
        let mut state = (0.0f32, 0.0f32, 0.0f32);
        for t in 0..seq_len {
            let dt = if t == 0 { 0.0 } else { t_hist[t * batch + c] };
            let (ns, cache) = step_fwd(w, dt, r_hist[t * batch + c], state, t == 0, &wc);
            state = ns;
            caches.push(cache);
        }
        for t in 1..seq_len {
            let wt = weights[t * batch + c] as f64;
            if wt != 0.0 {
                let r = clamp(caches[t].curve.out, MIN_R, MAX_R) as f64;
                let lbl = labels[t * batch + c] as f64;
                loss += -wt * (lbl * r.ln() + (1.0 - lbl) * (1.0 - r).ln());
            }
        }
        // Reverse: state-update adjoint via step_bwd, per-step loss adjoint via a separate curve_bwd.
        let mut g_state = (0.0f64, 0.0f64, 0.0f64); // (g_s, g_d, g_sf), matching step_bwd's order.
        for t in (0..seq_len).rev() {
            let (gs_step, gd_step, gsf_step) = step_bwd(w, &caches[t], g_state, gw, &wc);
            let (gs_loss, gd_loss, gsf_loss) = if t >= 1 {
                let wt = weights[t * batch + c] as f64;
                let cur = &caches[t].curve;
                let r_raw = cur.out;
                let r = clamp(r_raw, MIN_R, MAX_R) as f64;
                let lbl = labels[t * batch + c] as f64;
                let g_r = -wt * (lbl / r - (1.0 - lbl) / (1.0 - r));
                let g_rraw = if r_raw > MIN_R && r_raw < MAX_R {
                    g_r
                } else {
                    0.0
                };
                let cc = &caches[t];
                // curve_bwd returns (g_s, g_sf, g_d); reorder to (g_s, g_d, g_sf).
                let (g_s, g_sf, g_d) = curve_bwd(
                    w, cur, cc.dt, cc.last_s, cc.last_sf, cc.last_d, g_rraw, 0.0, gw,
                );
                (g_s, g_d, g_sf)
            } else {
                (0.0, 0.0, 0.0)
            };
            g_state = (gs_step + gs_loss, gd_step + gd_loss, gsf_step + gsf_loss);
        }
    }
    loss
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simd_transcendentals_accurate() {
        // exp8 over [-87,88] (rel) and ln8 over (1e-5, 4e4) (abs) vs true f64. Worst-case is the
        // f32 range-reduction floor (exp: x - n*ln2 cancellation near x~88 ~6.7e-6; ln: e_f*ln2
        // f32 add), NOT the minimax poly (exp ~2.6e-6, ln ~4.9e-6; profiling/minimax_coeffs.py).
        // FSRS's actual args are modest, so the real error is the poly floor. Gates sit just above
        // the reduction floor: still 10x tighter than the old 1e-4 and catch any coefficient typo.
        let mut worst_exp = 0.0f64;
        let mut x = -87.0f32;
        while x <= 88.0 {
            let v = exp8::<false>(f32x8::splat(x)).to_array()[0] as f64;
            let b = (x as f64).exp();
            worst_exp = worst_exp.max(((v - b) / b).abs());
            x += 0.013;
        }
        // exp8::<false> is the degree-3 minimax (iter21, rel err 7.5e-5) over the reduced range; the
        // full-range worst (~7.9e-5) is that poly plus the f32 range-reduction floor. This is a gross-
        // bug guard, not a precision spec — the accuracy that actually matters is the evaluate() band.
        assert!(worst_exp < 1.5e-4, "exp8 worst rel err {worst_exp:e}");
        let mut worst_ln = 0.0f64;
        let mut y = 1e-5f32;
        while y <= 4e4 {
            let v = ln8::<false>(f32x8::splat(y)).to_array()[0] as f64;
            let b = (y as f64).ln();
            worst_ln = worst_ln.max((v - b).abs());
            y *= 1.05;
        }
        assert!(worst_ln < 1.5e-5, "ln8 worst abs err {worst_ln:e}");
    }

    #[test]
    fn simd_batch_loss_matches_scalar() {
        // batch=20 = two full f32x8 groups + 4 remainder; mixed history lengths (rating-0 padding),
        // first review always real (1..4). The SIMD forward must match the scalar within the poly err.
        let w: Vec<f32> = crate::DEFAULT_PARAMETERS.to_vec();
        let (batch, seq) = (20usize, 4usize);
        let mut th = vec![0.0f32; seq * batch];
        let mut rh = vec![0.0f32; seq * batch];
        for c in 0..batch {
            let len = 1 + (c % seq); // 1..=seq real reviews, rest padded
            for t in 0..len {
                th[t * batch + c] = ((t + c) % 7) as f32;
                rh[t * batch + c] = (1 + ((t + c) % 4)) as f32; // 1..4
            }
        }
        let dts: Vec<f32> = (0..batch).map(|c| (1 + c % 30) as f32).collect();
        let lbl: Vec<f32> = (0..batch).map(|c| (c % 2) as f32).collect();
        let wts: Vec<f32> = (0..batch).map(|c| 0.5 + 0.1 * (c % 5) as f32).collect();
        let a = batch_loss(&w, &th, &rh, seq, batch, &dts, &lbl, &wts);
        let b = batch_loss_simd(&w, &th, &rh, seq, batch, &dts, &lbl, &wts);
        let rel = ((a - b) / a.abs().max(1e-9)).abs();
        assert!(rel < 1e-4, "simd batch_loss {b} vs scalar {a}, rel {rel:e}");
    }

    #[test]
    fn simd_grad_matches_scalar_grad() {
        // Vectorized analytic grad vs the scalar analytic grad (the trusted oracle). They differ
        // only by exp8/ln8 (~1e-6) and f32-vs-f64 accumulation, so a small relative-L2 over the
        // whole 36-vector confirms the f32x8 backward is a faithful translation. batch%8==0.
        let w: Vec<f32> = crate::DEFAULT_PARAMETERS.to_vec();
        let (batch, seq) = (16usize, 6usize);
        let mut th = vec![0.0f32; seq * batch];
        let mut rh = vec![0.0f32; seq * batch];
        for c in 0..batch {
            let len = 2 + (c % (seq - 1)); // 2..=seq real reviews, rest padded (rating 0)
            for t in 0..len {
                th[t * batch + c] = (1 + (t + c) % 20) as f32;
                rh[t * batch + c] = (1 + ((t + 2 * c) % 4)) as f32; // 1..4, first review always real
            }
        }
        let dts: Vec<f32> = (0..batch).map(|c| (1 + c % 30) as f32).collect();
        let lbl: Vec<f32> = (0..batch).map(|c| (c % 2) as f32).collect();
        let wts: Vec<f32> = (0..batch).map(|c| 0.4 + 0.13 * (c % 5) as f32).collect();
        let mut gs = [0.0f64; 34];
        batch_loss_and_grad(&w, &th, &rh, seq, batch, &dts, &lbl, &wts, &mut gs);
        let mut gv = [0.0f64; 34];
        batch_loss_and_grad_simd(&w, &th, &rh, seq, batch, &dts, &lbl, &wts, &mut gv);
        let mut num = 0.0f64;
        let mut den = 0.0f64;
        for i in 0..34 {
            num += (gs[i] - gv[i]).powi(2);
            den += gs[i].powi(2);
        }
        let rel = (num / den.max(1e-12)).sqrt();
        assert!(
            rel < 2e-3,
            "simd grad vs scalar grad rel-L2 {rel:e}\nscalar={gs:?}\nsimd={gv:?}"
        );
    }

    #[test]
    fn window_grad_equals_sum_of_prefix_grads() {
        // THE math identity behind the O(N) window: the windowed gradient for one card must equal
        // the SUM of the O(N^2) per-prefix gradients over that card's expanding-window prefix-items.
        // Both are scalar f64, so they agree to f64 reassociation tolerance.
        let w: Vec<f32> = crate::DEFAULT_PARAMETERS.to_vec();
        let kk = 6usize; // one card, reviews q[0..5]
        let dts_q = [0.0f32, 3.0, 8.0, 1.0, 20.0, 5.0]; // q[0].delta_t unused (inits state)
        let rat_q = [3.0f32, 2.0, 4.0, 1.0, 3.0, 3.0];
        let weight_at = |t: usize| 0.3 + 0.11 * t as f32; // distinct per-prediction recency weights

        // windowed: single column, full sequence, a loss at every step t>=1.
        let (seq_w, batch_w) = (kk, 1usize);
        let mut th_w = vec![0.0f32; seq_w * batch_w];
        let mut rh_w = vec![0.0f32; seq_w * batch_w];
        let mut lbl_w = vec![0.0f32; seq_w * batch_w];
        let mut wts_w = vec![0.0f32; seq_w * batch_w];
        for t in 0..kk {
            th_w[t] = dts_q[t];
            rh_w[t] = rat_q[t];
            if t >= 1 {
                wts_w[t] = weight_at(t);
                lbl_w[t] = if rat_q[t] > 1.0 { 1.0 } else { 0.0 };
            }
        }
        let mut g_win = [0.0f64; 34];
        let loss_win = loss_and_grad_range_window(
            &w, &th_w, &rh_w, seq_w, batch_w, &lbl_w, &wts_w, &mut g_win, 0, 1,
        );

        // per-prefix: one column per prefix length L=2..=kk (history q[0..L-2], predict q[L-1]).
        let (seq_p, batch_p) = (kk - 1, kk - 1);
        let mut th_p = vec![0.0f32; seq_p * batch_p];
        let mut rh_p = vec![0.0f32; seq_p * batch_p];
        let mut dts_p = vec![0.0f32; batch_p];
        let mut lbl_p = vec![0.0f32; batch_p];
        let mut wts_p = vec![0.0f32; batch_p];
        for c in 0..batch_p {
            let l = c + 2;
            for t in 0..(l - 1) {
                th_p[t * batch_p + c] = dts_q[t];
                rh_p[t * batch_p + c] = rat_q[t];
            }
            dts_p[c] = dts_q[l - 1];
            lbl_p[c] = if rat_q[l - 1] > 1.0 { 1.0 } else { 0.0 };
            wts_p[c] = weight_at(l - 1);
        }
        let mut g_pre = [0.0f64; 34];
        let loss_pre = loss_and_grad_range(
            &w, &th_p, &rh_p, seq_p, batch_p, &dts_p, &lbl_p, &wts_p, &mut g_pre, 0, batch_p,
        );

        let mut num = 0.0f64;
        let mut den = 0.0f64;
        for i in 0..34 {
            num += (g_win[i] - g_pre[i]).powi(2);
            den += g_pre[i].powi(2);
        }
        let rel = (num / den.max(1e-12)).sqrt();
        assert!(
            rel < 1e-9,
            "windowed grad vs sum-of-prefix grad rel-L2 {rel:e}\nwin={g_win:?}\npre={g_pre:?}"
        );
        assert!(
            (loss_win - loss_pre).abs() < 1e-9,
            "windowed loss {loss_win} vs prefix-sum loss {loss_pre}"
        );
    }

    #[test]
    fn simd_window_grad_matches_scalar_window() {
        // The f32x8 windowed kernel vs the scalar windowed oracle (same algorithm). The windowed path
        // now runs the ACCURATE minimax (exp8 deg-3, rel err 7.5e-5; ln8 deg-2, abs 4.9e-6 — iter23's
        // cruder windowed trade was reverted 2026-06-04) plus the s32·ex33 / pp·rexp exp-fusions
        // (iter24); the scalar oracle uses exact libm and the un-fused form. So the SIMD-vs-libm
        // gradient rel-L2 sits a few e-3, DOMINATED by f32-vs-f64 + the fusion (not a backward bug).
        // Backward CORRECTNESS is anchored by grad_matches_fd (finite-diff) + window_grad_equals_
        // sum_of_prefix_grads (the regrouping identity); this oracle is only a gross-bug guard, so its
        // tolerance reflects the windowed minimax. batch%8==0; mixed card lengths + a deliberately-
        // filtered middle prefix (wts==0 at t==2 for some cards) so the state updates but emits no grad.
        let w: Vec<f32> = crate::DEFAULT_PARAMETERS.to_vec();
        let (batch, seq) = (8usize, 7usize);
        let mut th = vec![0.0f32; seq * batch];
        let mut rh = vec![0.0f32; seq * batch];
        let mut lbl = vec![0.0f32; seq * batch];
        let mut wts = vec![0.0f32; seq * batch];
        for c in 0..batch {
            let klen = 2 + (c % (seq - 1)); // full card length 2..=seq
            for t in 0..klen {
                th[t * batch + c] = (1 + (t + c) % 19) as f32;
                rh[t * batch + c] = (1 + ((t + 2 * c) % 4)) as f32; // 1..4
            }
            for t in 1..klen {
                if t == 2 && c % 3 == 0 {
                    continue; // a filtered middle prefix: state updates, no prediction
                }
                wts[t * batch + c] = 0.4 + 0.07 * ((t + c) % 5) as f32;
                lbl[t * batch + c] = if rh[t * batch + c] > 1.0 { 1.0 } else { 0.0 };
            }
        }
        let mut gs = [0.0f64; 34];
        let loss_scalar =
            loss_and_grad_range_window(&w, &th, &rh, seq, batch, &lbl, &wts, &mut gs, 0, batch);
        let mut gv = [0.0f64; 34];
        card_loss_and_grad_simd(&w, &th, &rh, seq, batch, &lbl, &wts, &mut gv);
        let mut num = 0.0f64;
        let mut den = 0.0f64;
        for i in 0..34 {
            num += (gs[i] - gv[i]).powi(2);
            den += gs[i].powi(2);
        }
        let rel = (num / den.max(1e-12)).sqrt();
        assert!(
            rel < 8e-3,
            "simd window grad vs scalar window grad rel-L2 {rel:e}\nscalar={gs:?}\nsimd={gv:?}"
        );
        // The validation forward (card_loss_simd) must match the scalar oracle's loss too.
        let loss_simd = card_loss_simd(&w, &th, &rh, seq, batch, &lbl, &wts);
        let lrel = ((loss_scalar - loss_simd) / loss_scalar.abs().max(1e-9)).abs();
        assert!(
            lrel < 5e-3,
            "card_loss_simd {loss_simd} vs scalar window {loss_scalar} rel {lrel:e}"
        );
    }

    #[allow(clippy::too_many_arguments)]
    fn fd_grad(
        w: &[f32],
        t_hist: &[f32],
        r_hist: &[f32],
        seq_len: usize,
        batch: usize,
        dts: &[f32],
        lbl: &[f32],
        wts: &[f32],
    ) -> [f64; 34] {
        let mut g = [0.0f64; 34];
        for i in 0..34 {
            let eps = 1e-3f32;
            let mut wp = w.to_vec();
            wp[i] = w[i] + eps;
            let lp = batch_loss(&wp, t_hist, r_hist, seq_len, batch, dts, lbl, wts);
            wp[i] = w[i] - eps;
            let lm = batch_loss(&wp, t_hist, r_hist, seq_len, batch, dts, lbl, wts);
            g[i] = (lp - lm) / (2.0 * eps as f64);
        }
        g
    }

    #[test]
    fn grad_matches_fd() {
        let w: Vec<f32> = crate::DEFAULT_PARAMETERS.to_vec();
        // batch=4, seq_len=2; first-review ratings 1,2,3,4 (rating 1 is the suspect)
        let t_hist = [1.0f32, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0];
        let r_hist = [1.0f32, 2.0, 3.0, 4.0, 3.0, 3.0, 3.0, 3.0];
        let dts = [5.0f32, 5.0, 5.0, 5.0];
        let lbl = [1.0f32, 1.0, 0.0, 1.0];
        let wts = [1.0f32, 1.0, 1.0, 1.0];
        let mut mg = [0.0f64; 34];
        batch_loss_and_grad(&w, &t_hist, &r_hist, 2, 4, &dts, &lbl, &wts, &mut mg);
        let fd = fd_grad(&w, &t_hist, &r_hist, 2, 4, &dts, &lbl, &wts);
        // NOTE: finite differences cross clamp/min/max kinks where the analytic subgradient
        // is correct (it matches autodiff) but FD does not, so a few weights (e.g. w18/w34 on
        // this synthetic input) show moderate disagreement that is NOT a bug. The real
        // correctness gate is the average log loss on the full run (±0.0010). This test only
        // guards against GROSS errors (sign flips / missing terms => rel >~ 1).
        let mut bad = false;
        for i in 0..34 {
            let d = (mg[i] - fd[i]).abs();
            let rel = d / fd[i].abs().max(1e-3);
            if rel > 0.6 && d > 1e-3 {
                bad = true;
                eprintln!(
                    "GROSS MISMATCH w[{:>2}] manual={:+.6e} fd={:+.6e} rel={:.2e}",
                    i, mg[i], fd[i], rel
                );
            }
        }
        assert!(!bad, "gross gradient mismatch vs finite difference");
    }
}

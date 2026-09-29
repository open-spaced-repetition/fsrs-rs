// The windowed FSRS-7 forward + reverse-mode backward over one group of cards, written against a
// lane type `F` (its integer twin `I`) and lane count `N`: wide_window.rs includes this file twice,
// as mod k8 (F = f32x8, every group) and mod k4 (F = f32x4, a batch's last group when it holds at
// most 4 cards). Group column offsets stay in units of 8 columns (the batch layout), so k4 reads
// the first 4 columns of a group.
// ===================== portable SIMD transcendentals =====================
// Polynomial exp/log over the lanes (portable `wide`: SSE2 on x86-64, NEON on ARM).

#[inline(always)]
pub(super) fn clamp8(x: F, lo: f32, hi: f32) -> F {
    x.fast_max(F::splat(lo)).fast_min(F::splat(hi))
}

/// exp over the lanes: 2^n · poly(r), x = n·ln2 + r (degree-3 minimax, rel err 7.5e-5).
#[inline(always)]
pub(super) fn exp8(x: F) -> F {
    // The clamp also maps NaN to -87 (maxps returns its 2nd operand on NaN).
    exp8_in_range(x.fast_max(F::splat(-87.0)).fast_min(F::splat(88.0)))
}

/// exp8 without its clamp, for arguments already inside [-87, 88]. Every forward call in the
/// SIMD kernels qualifies, because clip_parameters bounds the weights and the states are clamped:
/// p35/p31/cc have |x| <= 2 * ln(36500 / 1e-4) ~ 21; se/ex34/qbase/pr/expr/init |x| <= ~13;
/// e1 has an explicit min(60) and q1 > 0; r1/r2 = decay * ln(b) >= -0.95 * 88.7 (ln8 of even an
/// overflowed +inf base is 128 * ln2), so the clamp never changes a value there (bit-for-bit).
/// x*LOG2E is then finite and within +-128: ONE plain nearest-even convert gives n exactly as
/// round() did, without round()'s and round_int()'s NaN/overflow fix-ups (~10 ops/half on SSE2).
#[inline(always)]
pub(super) fn exp8_in_range(x: F) -> F {
    exp2_8_in_range(x * F::splat(LOG2E))
}

/// 2^y over 8 lanes (= exp8_in_range(y * ln2)); |y| <= ~126. Callers whose exp argument is a
/// constant times a variable fold log2(e) into the constant (WConsts::l2e) and call this directly.
#[inline(always)]
pub(super) fn exp2_8_in_range(y: F) -> F {
    let ni = y.fast_round_int();
    let n = ni.round_float();
    let c = |v: f32| F::splat(v);
    // The degree-3 relative minimax of exp(r) over [-ln2/2, ln2/2] (max rel err 7.5e-5), written as
    // a polynomial in f = y - n = r / ln2 (so 2^f): coefficient k times ln2^k.
    let f = y - n;
    let p = c(0.99992806) + f * (c(0.69326097) + f * (c(0.24261113) + f * c(0.05517167)));
    let bits: I = (ni + I::splat(127)) << 23;
    let two_n: F = bytemuck::cast(bits);
    p * two_n
}

/// log2 over 8 lanes: ln8's approximation divided by ln2 (the poly's leading constant scaled
/// by log2(e)), without ln8's e * ln2 multiply. The recurrence keeps its logs in base 2: they only
/// feed 2^x (exp2_8_in_range) and weight gradients (finish_gw applies the ln2 once per group).
#[inline(always)]
pub(super) fn log2_8(x: F) -> F {
    let bits: I = bytemuck::cast(x);
    let e: I = (bits >> 23) - I::splat(127);
    let mant_bits: I = (bits & I::splat(0x007f_ffff)) | I::splat(127 << 23);
    let m: F = bytemuck::cast(mant_bits);
    let one = F::splat(1.0);
    let t = (m - one) / (m + one);
    let t2 = t * t;
    let c = |v: f32| F::splat(v);
    let poly = c(2.0 * LOG2E) * t * (c(1.0000074) + t2 * (c(0.33217952) + t2 * c(0.22657777)));
    e.round_float() + poly
}

// ===================== the recurrence =====================
// The batch layout [seq_len, batch] makes the cards of a group at one timestep a contiguous load.

#[inline(always)]
pub(super) fn load8(s: &[f32], i: usize) -> F {
    // One bounds check for the 8 lanes (was 8 index checks).
    let a: [f32; N] = s[i..i + N].try_into().unwrap();
    F::from(a)
}

// F forgetting curve forward, storing the intermediates its backward needs (the F
// analogue of CurveCache). `out` is bit-identical to the old forward-only curve_out8 — it just
// also stashes the products + cached lns so curve8_bwd never recomputes a transcendental.
#[derive(Default)]
pub(super) struct Curve8 {
    out: F,
    a: F,
    q2: F,
    r1: F,
    q1: F,
    e1: F,
    p35: F,
    /// The lanes where m1 is inside its clamp (0.01, 0.95).
    m1_live: F,
    decay1: F,
    factor1: F,
    b1: F,
    b2: F,
    r2: F,
    sig: F,
    oms: F,
    ln_sf: F,
    ln_b1: F,
    ln_b2: F,
    ln_s: F,
}

/// curve8_fwd writing into `c` (a per-step cache slot). Each field is stored as soon as it is
/// computed, so it need not stay live (or be spilled and copied) until the end of the step.
/// Returns (out, r1).
#[allow(clippy::too_many_arguments)]
#[inline(always)]
pub(super) fn curve8_fwd_into(
    w: &[f32],
    t: F,
    s: F,
    sf: F,
    d: F,
    wc: &WConsts,
    ln_s: F,
    ln_sf: F,
    c: &mut Curve8,
) -> (F, F) {
    let sp = |i: usize| F::splat(w[i]);
    let k = F::splat;
    c.ln_s = ln_s;
    c.ln_sf = ln_sf;
    let t = t.fast_max(k(0.0));
    let a = t / sf;
    c.a = a;
    let bv = t / s;
    // decay2 is not d-modulated (m2 = w24 plain); ex34 = exp((d-5)*(d_decay-0.3)) is the time
    // scale inside b2.
    let ex34 = exp2_8_in_range((d - k(5.0)) * k(wc.w32m_l));
    // decay2 / p28 = base2[w26]^inv2 / factor2 are weight-only: hoisted into wc.
    let q2 = bv * F::splat(wc.factor2) * ex34;
    c.q2 = q2;
    let b2 = q2 + k(1.0);
    c.b2 = b2;
    let ln_b2 = log2_8(b2);
    c.ln_b2 = ln_b2;
    let r2 = exp2_8_in_range(F::splat(wc.decay2) * ln_b2);
    c.r2 = r2;
    // 34-param remap + all-positive offsets: p35=s_short^(s_decay1[w33]-0.3), m1=decay1[w23]*p35,
    // ex34=exp((d-5)*(d_decay[w32]-0.3)), m2=decay2[w24]*ex34, p28=base2[w26]^inv2,
    // p31=s_short^-s_weight_power1[w29], weight1=base_weight1[w27]*p31. ln_w27=ln(base1=w25),
    // ln_w28=ln(base2=w26).
    let p35 = exp2_8_in_range((sp(33) - k(0.3)) * ln_sf); // ln_sf, ln_s are log2
    c.p35 = p35;
    let m1 = sp(23) * p35;
    c.m1_live = m1.cmp_gt(k(0.01)) & m1.cmp_lt(k(0.95));
    let dm1 = clamp8(m1, 0.01, 0.95);
    let decay1 = k(0.0) - dm1;
    c.decay1 = decay1;
    let q1 = k(wc.ln_w27) / decay1;
    c.q1 = q1;
    let e1 = exp8_in_range(q1.fast_min(k(60.0)));
    c.e1 = e1;
    let factor1 = e1 - k(1.0);
    c.factor1 = factor1;
    let b1 = a * factor1 + k(1.0);
    c.b1 = b1;
    let ln_b1 = log2_8(b1);
    c.ln_b1 = ln_b1;
    let r1 = exp2_8_in_range(decay1 * ln_b1);
    c.r1 = r1;
    // ret = (weight1*r1 + weight2*r2) / (weight1 + weight2), weight1 = base_weight1[w27] *
    // sf^-s_weight_power1[w29], weight2 = base_weight2[w28] * s^s_weight_power2[w30] *
    // exp((d_weight[w31]-0.5)(d-5)). Written as r1*(1-sig) + r2*sig with sig = sigmoid(z) and
    // z = ln(weight2/weight1): the same function with ONE exp instead of two.
    // |z| <= ln(100) + 1.1*ln(36500) + 2.5 + 0.9*ln(36500) < 29, inside exp8's range.
    let zl = k(wc.lz0_l) + sp(30) * ln_s + (d - k(5.0)) * k(wc.w31m_l) + sp(29) * ln_sf;
    let ez = exp2_8_in_range(zl); // zl = z * log2(e)
    let oms = k(1.0) / (ez + k(1.0)); // 1 - sig
    c.oms = oms;
    let sig = ez * oms;
    c.sig = sig;
    let ret = r1 * oms + r2 * sig;
    let out = ret * k(1.0 - 2e-5) + k(1e-5);
    c.out = out;
    (out, r1)
}

/// VJP of curve8_fwd_into: returns (acc.0 + g_s, acc.1 + g_sf, acc.2 + g_d), each sum formed as
/// soon as its curve term is known, and accumulates into the gw bank. The code is ordered so values
/// die early: the kernel's basic blocks are far longer than LLVM's scheduling window, so the source
/// order decides how many values are live (and spilled) at once.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
pub(super) fn curve8_bwd(
    w: &[f32],
    c: &Curve8,
    s: F,
    sf: F,
    d: F,
    g_out: F,
    g_r1_extra: F,
    gw: &mut [F; 34],
    wc: &WConsts,
    acc: (F, F, F),
) -> (F, F, F) {
    let sp = |i: usize| F::splat(w[i]);
    let k = F::splat;
    let z = k(0.0);
    let g_ret = g_out * k(1.0 - 2e-5);
    // ret = r1*(1-sig) + r2*sig. r1 also feeds the short-trace stability update (reads r1, not
    // mixed R) -> g_r1_extra.
    let g_r1 = g_ret * c.oms + g_r1_extra;
    let g_r2 = g_ret * c.sig;
    // sig = sigmoid(z), dsig/dz = sig*(1-sig); z = ln(w28/w27) + w30*ln_s + (w31-0.5)*(d-5) + w29*ln_sf
    let g_z = g_ret * (c.r2 - c.r1) * (c.sig * c.oms);
    // Weight-only factors are applied once per group in finish_gw (slot 27 holds sum(g_z)).
    gw[27] += g_z;
    gw[29] += g_z * c.ln_sf;
    gw[30] += g_z * c.ln_s;
    gw[31] += g_z * (d - k(5.0));
    let mut g_d = g_z * (sp(31) - k(0.5));
    let gz30 = g_z * sp(30); // the ln(s) and ln(sf) terms of g_s / g_sf (below)
    let gz29 = g_z * sp(29);
    // r2 = b2^decay2, b2 = q2 + 1 with q2 = (t/s)*factor2*ex34 (ex34 = exp((d-5)*(d_decay-0.3)),
    // factor2 = p28 - 1, p28 = base2[w26]^(1/decay2), decay2 = -clamp(w24)). q2 is a product, so
    // the adjoints of ln(t/s), ln(ex34) and ln(factor2) are all x2 = g_b2*q2.
    let v2 = g_r2 * c.r2; // adjoint of ln(r2) = decay2 * ln(b2)
    // w26 and w24 enter r2 only through weight-only factors: slot 26 holds sum(x2), slot 28 the
    // direct decay2 term; finish_gw forms gw[26] and gw[24] from them.
    gw[28] += v2 * c.ln_b2;
    let x2 = v2 * F::splat(wc.decay2) / c.b2 * c.q2;
    g_d += x2 * (sp(32) - k(0.3));
    gw[32] += x2 * (d - k(5.0));
    gw[26] += x2;
    // d/ds of every ln(s) term, over ONE division.
    let g_s = (gz30 - x2) / s;
    let g_s = acc.0 + g_s;
    let g_d = acc.2 + g_d;
    // r1 = b1^decay1, b1 = a*factor1 + 1, a = t/sf, factor1 = e1 - 1, e1 = exp(min(q1, 60)),
    // q1 = ln(base1[w25]) / decay1, decay1 = -clamp(w23 * p35), p35 = sf^(s_decay1[w33]-0.3).
    let decay1 = c.decay1;
    let v1 = g_r1 * c.r1; // adjoint of ln(r1) = decay1 * ln(b1)
    let mut g_decay1 = v1 * c.ln_b1 * k(LN2); // ln(b1) = log2(b1) * ln2
    let g_b1_a = v1 * decay1 / c.b1 * c.a; // adjoint of b1, times a
    let g_a_a = g_b1_a * c.factor1; // adjoint of ln(a)
    let g_q1 = c.q1.cmp_lt(k(60.0)).blend(g_b1_a * c.e1, z);
    let g_lw27 = g_q1 / decay1;
    g_decay1 -= g_lw27 * c.q1; // d(q1)/d(decay1) = -q1/decay1
    gw[25] += g_lw27; // ln_base1 = ln(w[25]); finish_gw applies 1/w25
    let g_m1 = c.m1_live.blend(z - g_decay1, z);
    let u = g_m1 * c.p35;
    gw[23] += u;
    let y = u * sp(23); // adjoint of ln(p35)
    gw[33] += y * c.ln_sf;
    // d/dsf of every ln(sf) term, over ONE division.
    let g_sf = (gz29 + y * (sp(33) - k(0.3)) - g_a_a) / sf;
    let g_sf = acc.1 + g_sf;
    (g_s, g_sf, g_d)
}

/// hard_penalty (rating 2) or easy_bonus (rating 4), else 1. At most one of the two factors of
/// `x * hard * easy` differs from 1 and a product with 1 is exact, so `x * hard_easy8` is the same
/// value with one multiply less.
#[inline(always)]
pub(super) fn hard_easy8(w: &[f32], rating: F, start: usize) -> F {
    let k = F::splat;
    // The two masks never overlap, so the blends are one bitwise select: fewer SSE2 ops.
    let (m2, m4) = (rating.cmp_eq(k(2.0)), rating.cmp_eq(k(4.0)));
    (m2 & k(w[start + 6])) | (m4 & k(w[start + 7])) | (!(m2 | m4) & k(1.0))
}

// F stability-after-review forward + the intermediates its backward needs (analogue of StabCache).
#[derive(Default)]
pub(super) struct Stab8 {
    nsf_fail: F,
    sinc: F,
    expr: F,
    /// hard_easy8, aa * cc * (expr - 1) (the adjoint factor of bb), aa * bb * cc and
    /// aa * bb * cc * (expr - 1): partial products of sinc.
    he: F,
    acce: F,
    abc: F,
    base: F,
    pr: F,
    qbase: F,
    ln_ls1: F,
    /// The post-lapse branch was computed (some lane lapsed or tied); else nsf_fail/pr/qbase/ln_ls1
    /// are not written (stale) and unused.
    full: bool,
}

/// Stability after a review, writing the intermediates its backward needs into `c` (a per-step
/// cache slot, see curve8_fwd_into); returns the new stability.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
pub(super) fn stab8_fwd_into(
    w: &[f32],
    wc: &WConsts,
    last_s: F,
    last_d: F,
    r: F,
    rating: F,
    start: usize,
    aa: f32,
    ln_ls: F,
    c: &mut Stab8,
) -> F {
    let sp = |i: usize| F::splat(w[i]);
    let k = F::splat;
    let one = k(1.0);
    let he = hard_easy8(w, rating, start);
    c.he = he;
    let bb = k(11.0) - last_d;
    let cc = exp2_8_in_range(k(-w[start + 1]) * ln_ls);
    let expr = exp2_8_in_range((one - r) * k(wc.l2e[start + 2]));
    c.expr = expr;
    let em1 = expr - one;
    c.acce = k(aa) * cc * em1;
    let abc = k(aa) * bb * cc;
    c.abc = abc;
    let base = abc * em1;
    c.base = base;
    let sinc = base * he + one; // = aa * bb * cc * (expr - 1) * he + 1
    c.sinc = sinc;
    let ls_sinc = last_s * sinc;
    // sinc >= 1, so ls_sinc >= last_s >= pls = min(last_s, nsf_fail): on a success lane the new
    // stability max(pls, ls_sinc) is ls_sinc and pls takes the gradient only on a tie. So the
    // post-lapse branch (a ln and two exps) is needed only if some lane lapses or ties; the value
    // of a padding lane (rating 0) is never used (its weights are 0).
    let full = (rating.cmp_eq(one) | ls_sinc.cmp_eq(last_s)).any();
    c.full = full;
    if !full {
        return ls_sinc;
    }
    let ln_ls1 = log2_8(last_s + one);
    c.ln_ls1 = ln_ls1;
    let qbase = exp2_8_in_range(sp(start + 4) * ln_ls1); // (last_s+1)^fail_s_exp[start+4]
    c.qbase = qbase;
    // fail_d_exp DROPPED: post-lapse stability is D-independent, so the legacy `pr` cache field now
    // holds just rexp (no pp = last_d^-fail_d_exp factor, so ln(last_d) is not needed).
    let pr = exp2_8_in_range((one - r) * k(wc.l2e[start + 5])); // rexp = exp((1-r)*fail_r_mult[start+5])
    c.pr = pr;
    let nsf_fail = sp(start + 3) * pr * (qbase - one);
    c.nsf_fail = nsf_fail;
    let pls = last_s.fast_min(nsf_fail);
    let nss = pls.fast_max(ls_sinc);
    rating.cmp_gt(one).blend(nss, pls)
}

/// VJP of stab8_fwd_into (F analogue of stab_bwd). Returns (g_last_s, g_last_d, g_r).
#[allow(clippy::too_many_arguments)]
#[inline(always)]
pub(super) fn stab8_bwd(
    w: &[f32],
    c: &Stab8,
    last_s: F,
    r: F,
    rating: F,
    start: usize,
    ln_ls: F,
    g_out: F,
    gw: &mut [F; 34],
) -> (F, F, F) {
    let sp = |i: usize| F::splat(w[i]);
    let k = F::splat;
    let one = k(1.0);
    let z = k(0.0);
    // ln_ls is passed in (not cached): the same value stab8_fwd_into used.
    let he = c.he;
    // Without the post-lapse branch (see stab8_fwd_into) every lane routes to ls_sinc: no lane
    // lapsed, so every real lane has rating > 1, and a padding lane's adjoint is already zero.
    let (mut g_last_s, g_ls_sinc, g_nsf_fail) = if c.full {
        let gt1 = rating.cmp_gt(one);
        let g_nss = gt1.blend(g_out, z);
        let pls = last_s.fast_min(c.nsf_fail);
        let ls_sinc = last_s * c.sinc;
        let g_pls_direct = gt1.blend(z, g_out);
        // nss = max(pls, ls_sinc)  (ties to pls, matching the scalar >= / > split)
        let g_pls_from_nss = pls.cmp_ge(ls_sinc).blend(g_nss, z);
        let g_ls_sinc = ls_sinc.cmp_gt(pls).blend(g_nss, z);
        let g_pls = g_pls_direct + g_pls_from_nss;
        // pls = min(last_s, nsf_fail)
        let g_ls = last_s.cmp_le(c.nsf_fail).blend(g_pls, z);
        (g_ls, g_ls_sinc, c.nsf_fail.cmp_lt(last_s).blend(g_pls, z))
    } else {
        (z, g_out, z)
    };
    g_last_s += g_ls_sinc * c.sinc;
    let g_sinc = g_ls_sinc * last_s;
    // sinc = aa*bb*cc*(expr-1)*hard*easy + 1
    let g_prod = g_sinc;
    // (Ordered so each adjoint is used up soon after it is formed; see curve8_bwd_acc.)
    let g_he = g_prod * he;
    // expr = exp((1-r)*w[start+2])
    let g_em1 = g_he * c.abc;
    let mut g_r = g_em1 * c.expr * k(-w[start + 2]);
    gw[start + 2] += g_em1 * c.expr * (one - r);
    let g_bb = g_he * c.acce;
    let g_last_d = g_bb * (z - one); // bb = 11 - last_d (the ONLY D-dependence of stab now)
    // d(prod)/d(hard_penalty) on rating-2 lanes and d(prod)/d(easy_bonus) on rating-4 lanes are
    // both `base` (the other factor is exactly 1 there).
    let g_base = g_prod * c.base;
    gw[start + 6] += rating.cmp_eq(k(2.0)).blend(g_base, z); // hard_penalty
    gw[start + 7] += rating.cmp_eq(k(4.0)).blend(g_base, z); // easy_bonus
    // prod = base * he is a product, so the adjoint of ln(aa) and of ln(cc) are both g_prod * prod.
    let p_ln = g_base * he;
    gw[start] += p_ln; // aa = exp(w[start]-1.5)
    // cc = last_s^-w[start+1] = exp(-w[start+1] * ln_ls)
    g_last_s += p_ln * k(-w[start + 1]) / last_s;
    gw[start + 1] -= p_ln * ln_ls;
    if !c.full {
        return (g_last_s, g_last_d, g_r);
    }
    // nsf_fail = fail_mult[start+3] * pr * (qbase-1) ; pr = rexp = exp((1-r)*fail_r_mult[start+5])
    // (fail_d_exp DROPPED: no pp = last_d^-x factor, so nsf_fail is D-independent).
    // The adjoint of ln(fail_mult) and of ln(pr) are both n_ln = g_nsf_fail * nsf_fail.
    let n_ln = g_nsf_fail * c.nsf_fail;
    gw[start + 3] += n_ln; // finish_gw applies 1/fail_mult
    // pr = exp((1-r)*fail_r_mult[start+5])
    g_r += n_ln * k(-w[start + 5]);
    gw[start + 5] += n_ln * (one - r);
    // qbase = (last_s+1)^fail_s_exp[start+4]; q_ln = adjoint of ln(qbase)
    let q_ln = g_nsf_fail * sp(start + 3) * c.pr * c.qbase;
    g_last_s += q_ln * sp(start + 4) / (last_s + one);
    gw[start + 4] += q_ln * c.ln_ls1;
    (g_last_s, g_last_d, g_r)
}

/// F next-difficulty forward; returns (clamped out, pre-clamp out, delta_d).
#[inline(always)]
pub(super) fn next_d8_fwd(
    w: &[f32],
    last_d: F,
    rating: F,
    r: F,
    init: f32,
    lapse: bool,
) -> (F, F, F) {
    let k = F::splat;
    let delta_d_base = k(-w[6]) * (rating - k(3.0));
    // Surprise-weighted lapse: on a lapse scale delta_d by (r+0.1) = 1 + (R-0.9).
    let delta_d = if lapse {
        rating
            .cmp_eq(k(1.0))
            .blend(delta_d_base * (r + k(0.1)), delta_d_base)
    } else {
        delta_d_base
    };
    let new_d = last_d + (k(10.0) - last_d) * delta_d / k(9.0);
    let out_pre = k(0.01) * k(init) + k(0.99) * new_d;
    (clamp8(out_pre, D_MIN, D_MAX), out_pre, delta_d) // delta_d returned = EFFECTIVE delta_d
}

/// The next-difficulty values its backward needs: the lanes where the pre-clamp output is inside
/// (D_MIN, D_MAX), and 1 - delta_d / 9 (delta_d the EFFECTIVE delta_d).
#[inline(always)]
pub(super) fn next_d8_cache(out_pre: F, delta_d: F) -> (F, F) {
    let k = F::splat;
    (
        out_pre.cmp_gt(k(D_MIN)) & out_pre.cmp_lt(k(D_MAX)),
        k(1.0) - delta_d / k(9.0),
    )
}

/// VJP of next_d8_fwd. Returns (g_last_d, g_r); accumulates gw[4], gw[5], gw[6]. `live` and `omd9`
/// are next_d8_cache's; `r` is the curve retention (feeds the lapse surprise weighting).
#[allow(clippy::too_many_arguments)]
#[inline(always)]
pub(super) fn next_d8_bwd(
    w: &[f32],
    live: F,
    omd9: F,
    last_d: F,
    rating: F,
    r: F,
    g_out: F,
    gw: &mut [F; 34],
    exp3w5: f64,
    lapse: bool,
) -> (F, F) {
    let k = F::splat;
    let z = k(0.0);
    let g_out_pre = live.blend(g_out, z);
    let g_init = g_out_pre * k(0.01);
    let g_new_d = g_out_pre * k(0.99);
    gw[4] += g_init;
    gw[5] += g_init * k(-(exp3w5 as f32) * 3.0); // init = w4 - exp(3 w5) + 1
    let g_last_d = g_new_d * omd9;
    let g_delta_d = g_new_d * (k(10.0) - last_d) / k(9.0);
    let rm3 = rating - k(3.0);
    if !lapse {
        gw[6] += g_delta_d * (z - rm3); // d(delta_d)/d(w6) = -(rating-3)
        return (g_last_d, z);
    }
    let is_lapse = rating.cmp_eq(k(1.0));
    // d(delta_d_eff)/d(w6) = -(rating-3), scaled by (r+0.1) on a lapse.
    gw[6] += g_delta_d * is_lapse.blend((z - rm3) * (r + k(0.1)), z - rm3);
    // d(delta_d_eff)/dr = delta_d_base = -w6*(rating-3), only on a lapse.
    let g_r = is_lapse.blend(g_delta_d * (k(-w[6]) * rm3), z);
    (g_last_d, g_r)
}

// ===================== SIMD recurrence step (8 cards/lane) with cache =====================
// Per-timestep cache for the vectorized backward. The first review (t==0) only needs the init
// override's data (the curve/stab/next_d it computes are dead, overridden), so it gets its own small
// First8; every later step stores the full forward intermediates (Step8).
pub(crate) struct First8 {
    rc: F,
    init_s: F,
    ex_w5: F,
    id_in: F,
}

#[derive(Default)]
pub(crate) struct Step8 {
    s0: F,
    d0: F,
    sf0: F,
    rating: F,
    curve: Curve8,
    slow: Stab8,
    fast: Stab8,
    /// next_d8_cache's values.
    nd_live: F,
    nd_omd9: F,
    /// The lanes where the new stabilities are inside (S_MIN, S_MAX) before their clamp, and (on a
    /// lapse step) where the short trace's post-lapse cap wins (see step8_fwd_into).
    ns_live: F,
    nsf_live: F,
    fast_wins: F,
    /// Some lane lapses (rating 1) at this step; else the lapse-only terms are skipped.
    lapse: bool,
}

/// The first review (t==0): initialises the state from the rating. It makes no prediction.
#[inline(always)]
pub(super) fn first8_fwd(w: &[f32], rating: F) -> ((F, F, F), First8) {
    let k = F::splat;
    let one = k(1.0);
    let rc = clamp8(rating, 1.0, 4.0);
    let init_s = rc.cmp_eq(one).blend(
        k(w[0]),
        rc.cmp_eq(k(2.0))
            .blend(k(w[1]), rc.cmp_eq(k(3.0)).blend(k(w[2]), k(w[3]))),
    );
    let ex_w5 = exp8_in_range(k(w[5]) * (rc - one));
    let id_in = k(w[4]) - ex_w5 + one;
    let init_d = clamp8(id_in, D_MIN, D_MAX);
    let out = (
        clamp8(init_s, S_MIN, S_MAX),
        init_d,
        clamp8(k(0.8) * init_s, S_MIN, S_MAX),
    );
    (
        out,
        First8 {
            rc,
            init_s,
            ex_w5,
            id_in,
        },
    )
}

/// VJP of first8_fwd: the only weight grads are gw[rc-1] (init stability, scattered by the per-lane
/// rating via 4 masked adds) and gw[4]/gw[5] (init difficulty). The input adjoints are 0 (the state
/// before the first review is constant 0).
#[inline(always)]
pub(super) fn first8_bwd(c: &First8, g_out: (F, F, F), gw: &mut [F; 34]) {
    let k = F::splat;
    let one = k(1.0);
    let z = k(0.0);
    let (g_ns_out, g_nd_out, g_nsf_out) = g_out;
    let First8 {
        rc,
        init_s,
        ex_w5,
        id_in,
    } = c;
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
}

/// step8_fwd writing its cache into `slot` (reused across groups by card_group_grad): the forward
/// stores each intermediate once, straight into the cache, instead of building the Step8 in
/// registers and spill slots and copying it at the push. Returns the new state. `last`: a card
/// group's last step, whose state update feeds no later step, computes its curve only (the loop
/// runs the same code for it, so the hot code holds one copy of the curve).
#[inline(always)]
pub(super) fn step8_fwd_into(
    w: &[f32],
    dt_raw: F,
    rating: F,
    state: (F, F, F),
    wc: &WConsts,
    slot: &mut Step8,
    last: bool,
) -> (F, F, F) {
    let k = F::splat;
    let one = k(1.0);
    // The incoming state is a previous step's output, which is already clamped (the first step
    // clamps its init values too), so clamping it again would change nothing.
    let (last_s, last_d, last_sf) = state;
    let Step8 {
        s0: c_s0,
        d0: c_d0,
        sf0: c_sf0,
        rating: c_rating,
        curve,
        slow,
        fast,
        nd_live,
        nd_omd9,
        ns_live,
        nsf_live,
        fast_wins,
        lapse: c_lapse,
    } = slot;
    (*c_s0, *c_d0, *c_sf0, *c_rating) = (last_s, last_d, last_sf, rating);
    let ln_last_s = log2_8(last_s);
    let ln_last_sf = log2_8(last_sf);
    // r1 = short component recall — drives the short-trace update
    let (r, r1) = curve8_fwd_into(
        w, dt_raw, last_s, last_sf, last_d, wc, ln_last_s, ln_last_sf, curve,
    );
    if last {
        return state;
    }
    let slow_out = stab8_fwd_into(w, wc, last_s, last_d, r, rating, 7, wc.aa7, ln_last_s, slow);
    let fast_out = stab8_fwd_into(
        w, wc, last_sf, last_d, r1, rating, 15, wc.aa16, ln_last_sf, fast,
    );
    let lapse = rating.cmp_eq(one).any();
    *c_lapse = lapse;
    let (nd, pre, dd) = next_d8_fwd(w, last_d, rating, r, wc.init, lapse);
    (*nd_live, *nd_omd9) = next_d8_cache(pre, dd);
    // Post-lapse short reset: on a lapse cap s_short at 0.8 * post-lapse s_long.
    let nsf_pre = if lapse {
        *fast_wins = fast_out.cmp_le(k(0.8) * slow_out);
        rating
            .cmp_eq(one)
            .blend(fast_out.fast_min(k(0.8) * slow_out), fast_out)
    } else {
        fast_out
    };
    // Padding (rating 0) does not pass the state through: a card's padding only trails it and carries
    // weight 0, and lanes are independent, so padding lanes may evolve (their values stay finite:
    // rating 0 takes the clamped post-lapse branch) and contribute only zero gradients.
    let (ns3, nsf3, nd3) = (slow_out, nsf_pre, nd);
    *ns_live = ns3.cmp_gt(k(S_MIN)) & ns3.cmp_lt(k(S_MAX));
    *nsf_live = nsf3.cmp_gt(k(S_MIN)) & nsf3.cmp_lt(k(S_MAX));
    (clamp8(ns3, S_MIN, S_MAX), nd3, clamp8(nsf3, S_MIN, S_MAX))
}

/// VJP of one step (F analogue of step_bwd). Given output-state adjoints, returns input-state
/// adjoints and accumulates the weight gradient into `gw`. `g_r_loss` is the adjoint of any loss
/// scored directly off this step's curve.out (every step of a card emits a prediction); it is added to
/// the two stab r-adjoints before curve8_bwd, since curve.out feeds the loss AND both stability
/// traces. t >= 1 only (first8_bwd is the t==0 step). `last`: the curve-only last step (see
/// step8_fwd_into); its output adjoint is 0, so only the curve's loss term contributes.
#[inline(always)]
pub(super) fn step8_bwd(
    w: &[f32],
    c: &Step8,
    g_out: (F, F, F),
    g_r_loss: F,
    gw: &mut [F; 34],
    wc: &WConsts,
    last: bool,
) -> (F, F, F) {
    let k = F::splat;
    let one = k(1.0);
    let z = k(0.0);
    let (g_ns_out, g_nd_out, g_nsf_out) = g_out;
    let Step8 {
        s0,
        d0,
        sf0,
        rating,
        curve,
        slow,
        fast,
        nd_live,
        nd_omd9,
        ns_live,
        nsf_live,
        fast_wins,
        lapse,
    } = c;
    let (last_s, last_d, last_sf) = (*s0, *d0, *sf0); // already clamped (see step8_fwd_into)
    let (g_r_out, g_r1_short, acc) = if last {
        (g_r_loss, z, (z, z, z))
    } else {
        // (The forward stored the clamp masks of the new stabilities.)
        let g_ns2 = ns_live.blend(g_ns_out, z);
        let g_nsf2 = nsf_live.blend(g_nsf_out, z);
        let g_nd2 = g_nd_out;
        // POST-LAPSE min routing: nsf_pre = (rating==1)? min(fast.out, 0.8*slow.out) : fast.out.
        let (g_fast_out, g_slow_from_relearn) = if *lapse {
            let is_lapse = rating.cmp_eq(one);
            (
                is_lapse.blend(fast_wins.blend(g_nsf2, z), g_nsf2),
                is_lapse.blend(fast_wins.blend(z, g_nsf2 * k(0.8)), z),
            )
        } else {
            (g_nsf2, z)
        };
        // LONG stab reads mixed retention curve.out; SHORT stab reads r1=curve.r1 (start 15).
        let (g_ls_a, g_ld_a, g_r_long) = stab8_bwd(
            w,
            slow,
            last_s,
            curve.out,
            *rating,
            7,
            curve.ln_s,
            g_ns2 + g_slow_from_relearn,
            gw,
        );
        let g_r_curve = g_r_long + g_r_loss;
        let (g_lsf_b, g_ld_b, g_r1_short) = stab8_bwd(
            w,
            fast,
            last_sf,
            curve.r1,
            *rating,
            15,
            curve.ln_sf,
            g_fast_out,
            gw,
        );
        let g_ld_ab = g_ld_a + g_ld_b;
        let (g_ld_c, g_r_nextd) = next_d8_bwd(
            w, *nd_live, *nd_omd9, last_d, *rating, curve.out, g_nd2, gw, wc.exp3w5, *lapse,
        );
        // curve.out adjoint = long-stab r + windowed loss adjoint + next_d r; curve.r1 = short-stab r.
        // The input-state adjoints: g_last_s = g_ls_a + g_ls_d, g_last_sf = g_lsf_b + g_lsf_d and
        // g_last_d = ((g_ld_a + g_ld_b) + g_ld_c) + g_ld_d, summed inside curve8_bwd.
        (
            g_r_curve + g_r_nextd,
            g_r1_short,
            (g_ls_a, g_lsf_b, g_ld_ab + g_ld_c),
        )
    };
    let (g_last_s, g_last_sf, g_last_d) = curve8_bwd(
        w, curve, last_s, last_sf, last_d, g_r_out, g_r1_short, gw, wc, acc,
    );
    // No clamp gate here: s0/d0/sf0 are the previous step's clamped outputs, and that step's
    // backward applies the same mask to this adjoint (S_MIN < x < S_MAX holds for the
    // clamped value exactly when it holds for the unclamped one), so gating here too would change
    // nothing.
    (g_last_s, g_last_d, g_last_sf)
}

// ===================== windowed O(N) forward+backward =====================
// The O(N^2) -> O(N) expanding window. A card with K reviews became K-1 prefix-items (lengths 2..K),
// each re-running the recurrence over its whole prefix => O(K^2) timestep-work. Here each CARD is a
// single column whose recurrence runs ONCE over its full review sequence: at step t (t>=1) the curve
// curve8_fwd(state_{t-1}, delta_t[t]) it computes for the stability update IS EXACTLY the prediction
// R_t the length-(t+1) prefix used to score review t (same input state, same delta_t), so a loss is
// read off every step for free. wts/lbl are row-major [seq, bsz]; wts[t][c]==0 marks "no prediction"
// (t==0, an outlier-filtered prefix, or a padding column/timestep). The total loss and gradient equal
// the per-prefix sums (the recurrence is deterministic), so this is math-identical to the per-prefix
// path up to floating-point reassociation.

/// d/dr of -wt*BCE via the unified label identity: d[-ln(1-|label-r|)]/dr = -sign(label-r)/
/// (1-|label-r|). ONE division (label is 0/1); padding/filtered steps have wt==0 -> 0; the
/// [MIN_R,MAX_R] clamp zeroes the adjoint outside the range.
/// `weights` are the windowed batches' SIGNED weights (signed_weight): sign(label - r) is +1 for
/// label 1 and -1 for label 0 (r is inside (0, 1)), so (0 - wt) * (sign / den) is exactly
/// swt * (1 / den) with swt = sign * (0 - wt) precomputed by the layout. The sign bit of swt is the
/// label (set: label 1), so the batches hold no labels; a zero weight's sign may differ, but its
/// adjoint is a zero of swt's sign either way.
#[inline(always)]
pub(super) fn g_r_loss_at(r_raw: F, weights: &[f32], base: usize) -> F {
    let k = F::splat;
    let (one, z) = (k(1.0), k(0.0));
    let swt = load8(weights, base);
    let r = clamp8(r_raw, MIN_R, MAX_R);
    let g_r = swt * (one / (one - abs_label_diff(swt, r)));
    (r_raw.cmp_gt(k(MIN_R)) & r_raw.cmp_lt(k(MAX_R))).blend(g_r, z)
}

/// |label - r| for label = the sign bit of the signed weight `swt` (see g_r_loss_at): 1 - r for
/// label 1, r for label 0 (the values the old max(label - r, r - label) gave).
#[inline(always)]
fn abs_label_diff(swt: F, r: F) -> F {
    let label1: F = bytemuck::cast(bytemuck::cast::<F, I>(swt) >> 31);
    label1.blend(F::splat(1.0) - r, r)
}

/// Windowed forward + backward of 8-card group `g` (columns 8g..8g+8) of a batch. Returns the
/// group's per-parameter gradient (the lane sums, f32); callers add the groups to their f64 total
/// IN GROUP ORDER, so any split of the groups over threads stays bit-for-bit.
#[allow(clippy::too_many_arguments)]
pub(crate) fn card_group_grad(
    w: &[f32],
    wc: &WConsts,
    t_hist: &[f32],
    r_hist: &[f32],
    seq_len: usize,
    batch: usize,
    weights: &[f32],
    g: usize,
    caches: &mut Vec<Step8>,
) -> [f32; 34] {
    debug_assert!(
        seq_len >= 2,
        "windowed grad needs seq_len >= 2 (min surviving prefix length is 2)"
    );
    let k = F::splat;
    let z = k(0.0);
    let c0 = g * 8;
    // This group's own length: trailing timesteps where all 8 lanes are padding (rating 0; real
    // ratings are >= 1) carry weight 0 and only pass the state through, so they add exactly 0
    // to the gradient — skip them (bit-for-bit). Min 2 = the shortest card.
    let mut seq_len = seq_len;
    while seq_len > 2 && load8(r_hist, (seq_len - 1) * batch + c0).reduce_add() == 0.0 {
        seq_len -= 1;
    }
    // Steps 1..=seq_len-1, one cache slot each (the first review, t==0, is peeled off: First8, so
    // caches[t - 1] is step t's). The last step's stability/next-difficulty update feeds no t+1, so
    // it computes its curve only. The cache slots are reused across groups (only the first
    // seq_len - 1 are this group's).
    let n_steps = seq_len - 1;
    while caches.len() < n_steps {
        caches.push(Step8::default());
    }
    let caches = &mut caches[..n_steps];
    let ((mut s, mut d, mut sf), first) = first8_fwd(w, load8(r_hist, c0));
    // The steps walk the rows of the [seq, batch] arrays (so no per-step bounds checks).
    let rows = |a| steps_rows(a, batch, n_steps);
    for (t, ((slot, th), rh)) in caches
        .iter_mut()
        .zip(rows(t_hist))
        .zip(rows(r_hist))
        .enumerate()
    {
        let last = t + 1 == n_steps;
        (s, d, sf) = step8_fwd_into(w, load8(th, c0), load8(rh, c0), (s, d, sf), wc, slot, last);
    }
    // Reverse pass, from the final state's adjoint 0 (at the last step only its curve's loss counts).
    let mut gw_g = [z; 34];
    let (mut g_s, mut g_sf, mut g_d) = (z, z, z);
    for (t, (cache, wr)) in caches.iter().zip(rows(weights)).enumerate().rev() {
        let g_r_loss = g_r_loss_at(cache.curve.out, wr, c0);
        let last = t + 1 == n_steps;
        (g_s, g_d, g_sf) = step8_bwd(w, cache, (g_s, g_d, g_sf), g_r_loss, &mut gw_g, wc, last);
    }
    // init step (t==0): no prediction (min surviving prefix length is 2).
    first8_bwd(&first, (g_s, g_d, g_sf), &mut gw_g);
    finish_gw(&gw_g, w, wc)
}

/// The rows of steps 1..=n of a row-major [seq, batch] array.
#[inline(always)]
fn steps_rows(a: &[f32], batch: usize, n: usize) -> std::slice::ChunksExact<'_, f32> {
    a[batch..(n + 1) * batch].chunks_exact(batch)
}

/// Lane sums of a group's gradient bank, with the weight-only factors that the per-step backward
/// leaves out: slot 27 = sum(g_z) (-> gw[27], gw[28]), slot 26 = sum(x2) and slot 28 = the direct
/// decay2 term (-> gw[26], gw[24]), slot 25 = sum(g_lw27), slots 10 / 18 = sum(n_ln) per trace.
pub(super) fn finish_gw(gw_g: &[F; 34], w: &[f32], wc: &WConsts) -> [f32; 34] {
    // The lane sums, four slots at a time on transposed halves: ((y0 + y1) + y2) + y3 with
    // y_i = x_i + x_{i+4} (N = 4, the 4-lane tail groups: y_i = x_i).
    let halves: &[f32x4] = bytemuck::cast_slice(&gw_g[..]);
    let nh = N / 4;
    let mut g = [0.0f32; 34];
    for c in (0..34).step_by(4) {
        // Each slot's two halves added first (lane i + lane i + 4), then one transpose.
        let h = |s: usize| match (s < 34, nh) {
            (false, _) => f32x4::ZERO,
            (true, 1) => halves[s],
            (true, _) => halves[2 * s] + halves[2 * s + 1],
        };
        let t = f32x4::transpose([h(c), h(c + 1), h(c + 2), h(c + 3)]);
        let sum = ((t[0] + t[1]) + t[2]) + t[3];
        for (k, v) in sum.to_array().into_iter().enumerate().take(34 - c) {
            g[c + k] = v;
        }
    }
    let (s_z, s_x2, s_dec2) = (g[27], g[26], g[28]);
    g[27] = s_z * wc.nrw27;
    g[28] = s_z * wc.rw28;
    g[26] = s_x2 * wc.k26;
    g[24] = if wc.live24 {
        -(s_dec2 * LN2 + s_x2 * wc.k24)
    } else {
        0.0
    };
    // Slots accumulated as g * log2(x) (the recurrence's logs are base 2): times ln2.
    for i in [8, 11, 16, 19, 29, 30, 33] {
        g[i] *= LN2;
    }
    g[25] *= wc.rw25;
    g[10] *= 1.0 / w[10];
    g[18] *= 1.0 / w[18];
    g
}

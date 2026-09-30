//! Windowed FSRS-7 gradient on portable SIMD (`wide`): the forward + reverse-mode backward of one
//! group of 8 cards (or 4, for a batch's last group), one card per lane. Training splits a batch's
//! groups over two threads and adds the per-group gradients in group order, so the result does not
//! depend on the split (see `training::train_fsrs7_windowed`).
//!
//! The batch layout is row-major [seq_len, batch_size] with batch_size a multiple of 8. `weights` are
//! SIGNED weights (`signed_weight`): the sign bit carries the label, so the batches need no label
//! array. The recurrence keeps its logs in base 2 (they only feed 2^x and the gradient).

use wide::{CmpEq, CmpGe, CmpGt, CmpLe, CmpLt, f32x4, f32x8};

const PARAM_LEN: usize = super::PARAM_LEN;
const S_MIN: f32 = 0.0001;
const S_MAX: f32 = 36500.0;
const D_MIN: f32 = 1.0;
const D_MAX: f32 = 10.0;
const MIN_R: f32 = 1e-5;
const MAX_R: f32 = 1.0 - 1e-5;

const LOG2E: f32 = std::f32::consts::LOG2_E;
const LN2: f32 = std::f32::consts::LN_2;

mod k8 {
    use super::*;
    use wide::{f32x8 as F, i32x8 as I};
    const N: usize = 8;
    include!("wide_window_lanes.rs");
}

pub(crate) mod k4 {
    use super::*;
    use wide::{f32x4 as F, i32x4 as I};
    const N: usize = 4;
    include!("wide_window_lanes.rs");
}

pub(crate) use k8::{Step8, card_group_grad};

/// The weight of a prediction with its label in the sign bit: -weight for label 1, +weight for label
/// 0 (the adjoint sign(label - r) * (0 - weight) of the BCE, since r is inside (0, 1)).
#[inline]
pub(crate) fn signed_weight(weight: f32, label: f32) -> f32 {
    if label == 1.0 {
        0.0 - weight
    } else {
        -(0.0 - weight)
    }
}

/// Weight-only subexpressions, computed once per gradient call instead of once per card-step.
pub(crate) struct WConsts {
    ln_w27: f32, // ln(w[25]) = ln(base1)   (q1 = ln_base1 / decay1)
    aa7: f32,    // exp(w[7]-1.5)   (long stab sinc_base, start=7)
    aa16: f32,   // exp(w[15]-1.5)  (short stab sinc_base, start=15)
    init: f32,   // w4 - exp(3*w5) + 1   (next difficulty)
    exp3w5: f64, // exp(3*w5) in f64      (next difficulty backward: d(init)/d(w5))
    // Weight-only terms of the curve's second component (decay2 = -clamp(w24) is not state-
    // modulated).
    decay2: f32,
    factor2: f32, // p28 - 1, p28 = base2[w26]^(1/decay2)
    nrw27: f32,   // -1 / w27 = d(logit)/d(w27)
    rw28: f32,    // 1 / w28 = d(logit)/d(w28)
    rw25: f32,    // 1 / w25 = d(ln base1)/d(w25)
    k26: f32,     // (d(p28)/d(w26)) / factor2 = (p28 / (decay2 * w26)) / factor2
    k24: f32,     // (d(p28)/d(w24)) / factor2 = -(p28 * ln(w26) * (-1/decay2^2)) / factor2
    live24: bool, // 0.01 < w24 < 0.95 (decay2's clamp is inactive)
    // Exponent constants pre-multiplied by log2(e), for exp2_8_in_range.
    l2e: [f32; PARAM_LEN], // w[i] * log2(e)
    w32m_l: f32,           // (w32 - 0.3) * log2(e)
    w31m_l: f32,           // (w31 - 0.5) * log2(e)
    lz0_l: f32,            // ln(w28 / w27) * log2(e)
}

pub(crate) fn wconsts(w: &[f32]) -> WConsts {
    let k = f32x8::splat;
    let m2 = k(w[24]);
    let decay2 = k(0.0) - k8::clamp8(m2, 0.01, 0.95);
    let inv2 = k(1.0) / decay2;
    let p28 = k8::exp8(inv2 * k(w[26].ln()));
    let factor2 = p28 - k(1.0);
    WConsts {
        ln_w27: w[25].ln(),
        aa7: (w[7] - 1.5).exp(),
        aa16: (w[15] - 1.5).exp(),
        init: w[4] - (w[5] * 3.0).exp() + 1.0,
        exp3w5: (w[5] as f64 * 3.0).exp(),
        decay2: decay2.to_array()[0],
        factor2: factor2.to_array()[0],
        nrw27: -1.0 / w[27],
        rw28: 1.0 / w[28],
        rw25: 1.0 / w[25],
        k26: (inv2 * (p28 / k(w[26])) / factor2).to_array()[0],
        k24: (p28 * k(w[26].ln()) * (k(0.0) - k(1.0) / (decay2 * decay2)) / factor2).to_array()[0],
        live24: w[24] > 0.01 && w[24] < 0.95,
        l2e: std::array::from_fn(|i| w[i] * LOG2E),
        w32m_l: (w[32] - 0.3) * LOG2E,
        w31m_l: (w[31] - 0.5) * LOG2E,
        lz0_l: (w[28] / w[27]).ln() * LOG2E,
    }
}

/// The gradient of group `g` of a batch: the 4-lane kernel for a last group of at most 4 cards (the
/// same per-lane arithmetic; the empty lanes' gradient is exactly zero), else the 8-lane one.
#[allow(clippy::too_many_arguments)]
pub(crate) fn group_grad(
    w: &[f32],
    wc: &WConsts,
    t_hist: &[f32],
    r_hist: &[f32],
    signed_weights: &[f32],
    seq_len: usize,
    batch_size: usize,
    cards: usize,
    g: usize,
    caches: &mut Caches,
) -> [f32; PARAM_LEN] {
    if g == batch_size / 8 - 1 && !cards.is_multiple_of(8) && cards % 8 <= 4 {
        k4::card_group_grad(
            w,
            wc,
            t_hist,
            r_hist,
            seq_len,
            batch_size,
            signed_weights,
            g,
            &mut caches.1,
        )
    } else {
        k8::card_group_grad(
            w,
            wc,
            t_hist,
            r_hist,
            seq_len,
            batch_size,
            signed_weights,
            g,
            &mut caches.0,
        )
    }
}

/// A thread's reusable per-step cache slots for the 8-lane and the 4-lane kernels.
pub(crate) type Caches = (Vec<k8::Step8>, Vec<k4::Step8>);

/// The windowed gradient of a whole batch on one thread (labels and unsigned weights, as the other
/// kernels take them): the groups in order, each group's f32 lane sums added in f64.
pub(crate) fn windowed_grad(
    w: &[f32],
    t_historys: &[f32],
    r_historys: &[f32],
    labels: &[f32],
    weights: &[f32],
    seq_len: usize,
    batch_size: usize,
) -> [f32; PARAM_LEN] {
    let signed: Vec<f32> = weights
        .iter()
        .zip(labels)
        .map(|(&weight, &label)| signed_weight(weight, label))
        .collect();
    let wc = wconsts(w);
    let mut caches = Caches::default();
    let mut total = [0.0f64; PARAM_LEN];
    for g in 0..batch_size / 8 {
        let gg = group_grad(
            w,
            &wc,
            t_historys,
            r_historys,
            &signed,
            seq_len,
            batch_size,
            batch_size,
            g,
            &mut caches,
        );
        for (t, v) in total.iter_mut().zip(gg) {
            *t += v as f64;
        }
    }
    total.map(|v| v as f32)
}

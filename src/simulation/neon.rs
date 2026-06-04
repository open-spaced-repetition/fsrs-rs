use super::S_MIN;
use crate::neon_math::F32x4;

#[inline(always)]
fn fsrs7_forgetting_curve(
    t: F32x4,
    s: f32,
    decay1: f32,
    decay2: f32,
    factor1: f32,
    factor2: f32,
    weight1: f32,
    weight2: f32,
) -> F32x4 {
    let one = F32x4::splat(1.0);
    let t_over_s = t.max(F32x4::splat(0.0)) / F32x4::splat(s);
    let r1 = (one + t_over_s * F32x4::splat(factor1)).ln() * F32x4::splat(decay1);
    let r2 = (one + t_over_s * F32x4::splat(factor2)).ln() * F32x4::splat(decay2);
    let r1 = r1.exp();
    let r2 = r2.exp();
    let weight1 = F32x4::splat(weight1);
    let weight2 = F32x4::splat(weight2);
    (weight1 * r1 + weight2 * r2) / (weight1 + weight2)
}

pub(super) fn add_fsrs7_forgetting_curve_range(
    w: &[f32],
    days: &mut [f32],
    start_day: usize,
    last_date: f32,
    stability: f32,
) {
    let s = stability.max(S_MIN);
    let decay1 = -w[27];
    let decay2 = -w[28];
    let factor1 = w[29].powf(1.0 / decay1) - 1.0;
    let factor2 = w[30].powf(1.0 / decay2) - 1.0;
    let weight1 = w[31] * s.powf(-w[33]);
    let weight2 = w[32] * s.powf(w[34]);

    let mut index = 0;
    while index + 4 <= days.len() {
        let first_t = (start_day + index) as f32 - last_date;
        let r = fsrs7_forgetting_curve(
            F32x4::sequence(first_t),
            s,
            decay1,
            decay2,
            factor1,
            factor2,
            weight1,
            weight2,
        );
        (F32x4::load(days, index) + r).store(days, index);
        index += 4;
    }

    for (offset, day) in days[index..].iter_mut().enumerate() {
        let t = (start_day + index + offset) as f32 - last_date;
        let t_over_s = t.max(0.0) / s;
        let r1 = (1.0 + factor1 * t_over_s).powf(decay1);
        let r2 = (1.0 + factor2 * t_over_s).powf(decay2);
        *day += (weight1 * r1 + weight2 * r2) / (weight1 + weight2);
    }
}

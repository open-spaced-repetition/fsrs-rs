#![allow(unsafe_op_in_unsafe_fn)]

use core::ops::{Add, Div, Mul, Neg, Sub};
use std::arch::aarch64::*;

const LOG2E: f32 = std::f32::consts::LOG2_E;
const LN2: f32 = std::f32::consts::LN_2;

#[derive(Clone, Copy)]
pub(crate) struct F32x4(float32x4_t);

impl F32x4 {
    #[inline(always)]
    pub(crate) fn splat(value: f32) -> Self {
        unsafe { Self(vdupq_n_f32(value)) }
    }

    #[inline(always)]
    pub(crate) fn sequence(start: f32) -> Self {
        let values = [start, start + 1.0, start + 2.0, start + 3.0];
        unsafe { Self(vld1q_f32(values.as_ptr())) }
    }

    #[inline(always)]
    pub(crate) fn load(values: &[f32], index: usize) -> Self {
        debug_assert!(index + 4 <= values.len());
        unsafe { Self(vld1q_f32(values.as_ptr().add(index))) }
    }

    #[inline(always)]
    pub(crate) fn store(self, values: &mut [f32], index: usize) {
        debug_assert!(index + 4 <= values.len());
        unsafe { vst1q_f32(values.as_mut_ptr().add(index), self.0) }
    }

    #[inline(always)]
    pub(crate) fn sum(self) -> f32 {
        unsafe { vaddvq_f32(self.0) }
    }

    #[inline(always)]
    pub(crate) fn abs(self) -> Self {
        unsafe { Self(vabsq_f32(self.0)) }
    }

    #[inline(always)]
    pub(crate) fn min(self, rhs: Self) -> Self {
        unsafe { Self(vminq_f32(self.0, rhs.0)) }
    }

    #[inline(always)]
    pub(crate) fn max(self, rhs: Self) -> Self {
        unsafe { Self(vmaxq_f32(self.0, rhs.0)) }
    }

    #[inline(always)]
    pub(crate) fn clamp(self, min: f32, max: f32) -> Self {
        self.max(Self::splat(min)).min(Self::splat(max))
    }

    #[inline(always)]
    pub(crate) fn round(self) -> Self {
        unsafe { Self(vrndnq_f32(self.0)) }
    }

    #[inline(always)]
    pub(crate) fn cmp_eq(self, rhs: Self) -> uint32x4_t {
        unsafe { vceqq_f32(self.0, rhs.0) }
    }

    #[inline(always)]
    pub(crate) fn cmp_gt(self, rhs: Self) -> uint32x4_t {
        unsafe { vcgtq_f32(self.0, rhs.0) }
    }

    #[inline(always)]
    pub(crate) fn cmp_lt(self, rhs: Self) -> uint32x4_t {
        unsafe { vcltq_f32(self.0, rhs.0) }
    }

    #[inline(always)]
    pub(crate) fn blend(mask: uint32x4_t, if_true: Self, if_false: Self) -> Self {
        unsafe { Self(vbslq_f32(mask, if_true.0, if_false.0)) }
    }

    #[inline(always)]
    pub(crate) fn exp(self) -> Self {
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
    pub(crate) fn exp_fast(self) -> Self {
        let x = self.clamp(-87.0, 88.0);
        let n = (x * Self::splat(LOG2E)).round();
        let r = x - n * Self::splat(LN2);
        let p = Self::splat(1.0004431) + r * (Self::splat(1.014861) + r * Self::splat(0.4962586));
        let two_n = unsafe {
            let n = vcvtq_s32_f32(n.0);
            let exponent = vshlq_n_s32::<23>(vaddq_s32(n, vdupq_n_s32(127)));
            Self(vreinterpretq_f32_s32(exponent))
        };
        p * two_n
    }

    #[inline(always)]
    pub(crate) fn ln(self) -> Self {
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

    #[inline(always)]
    pub(crate) fn ln_fast(self) -> Self {
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
        let poly = Self::splat(2.0) * t * (Self::splat(0.99965036) + t2 * Self::splat(0.35748693));
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

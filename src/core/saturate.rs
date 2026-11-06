/*
 *  saturate.rs
 *  purecv
 *
 *  This file is part of purecv - WebARKit.
 *
 *  purecv is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU Lesser General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  purecv is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU Lesser General Public License for more details.
 *
 *  You should have received a copy of the GNU Lesser General Public License
 *  along with purecv.  If not, see <http://www.gnu.org/licenses/>.
 *
 *  As a special exception, the copyright holders of this library give you
 *  permission to link this library with independent modules to produce an
 *  executable, regardless of the license terms of these independent modules, and to
 *  copy and distribute the resulting executable under terms of your choice,
 *  provided that you also meet, for each linked independent module, the terms and
 *  conditions of the license of that module. An independent module is a module
 *  which is neither derived from nor based on this library. If you modify this
 *  library, you may extend this exception to your version of the library, but you
 *  are not obligated to do so. If you do not wish to do so, delete this exception
 *  statement from your version.
 *
 *  Copyright 2026 WebARKit.
 *
 *  Author(s): Walter Perdan @kalwalt https://github.com/kalwalt
 *
 */

//! Saturating conversion from floating point to an element type — the
//! counterpart of OpenCV's `cv::saturate_cast<T>(float|double)`.
//!
//! OpenCV writes every integer output through `saturate_cast`, which is
//! `cvRound` (round to nearest, ties to even in the default FP mode)
//! followed by a clamp to the range of the destination type
//! ([`saturate.hpp`](https://github.com/opencv/opencv/blob/4.10.0/modules/core/include/opencv2/core/saturate.hpp),
//! [`fast_math.hpp`](https://github.com/opencv/opencv/blob/4.10.0/modules/core/include/opencv2/core/fast_math.hpp)).
//! [`SaturateCast`] gives generic code the same conversion: integer targets
//! round half to even and clamp, float targets are a plain cast.

// `num_traits::Float` provides `round` on `f64` via libm when `std` is
// disabled; with `std` the inherent method wins.
#[allow(unused_imports)]
use num_traits::Float;
use num_traits::{AsPrimitive, Bounded};

/// Rounds `v` to the nearest integer, ties to even (`cvRound` in the
/// default FP mode).
///
/// `f64::round_ties_even` is not available in `core`, so ties are detected
/// on top of `round` (ties away from zero) and moved to the even neighbour.
/// Halving is exact for every finite binary float, so the correction is
/// exact too. NaN and infinities are returned unchanged.
#[inline]
pub(crate) fn round_half_even(v: f64) -> f64 {
    let r = v.round();
    if (r - v).abs() == 0.5 {
        2.0 * (v * 0.5).round()
    } else {
        r
    }
}

/// Element types a floating-point value can be saturated into.
///
/// Mirrors OpenCV's `saturate_cast<T>(double)` / `saturate_cast<T>(float)`:
///
/// - integer targets round to nearest with ties to even, then clamp to
///   `T::MIN..=T::MAX`; NaN becomes `0`;
/// - `f32` / `f64` targets are not rounded (a plain cast).
///
/// Where OpenCV's result for an out-of-range input comes from the hardware
/// conversion, every integer target clamps here instead:
///
/// - `saturate_cast<int>(float)`, `saturate_cast<unsigned>(double)` (a plain
///   cast of `cvRound`, so `-1.0` gives `u32::MAX` in OpenCV, `0` here) and
///   the 64-bit targets;
/// - the 8- and 16-bit targets for `|v| ≥ 2³¹`: OpenCV rounds to `int`
///   first, which gives `INT_MIN` on x86 and so `0` for `u8`, where purecv
///   gives `255`.
///
/// # Example
/// ```
/// use purecv::core::{saturate_cast, SaturateCast};
///
/// assert_eq!(saturate_cast::<u8>(2.5), 2);
/// assert_eq!(saturate_cast::<u8>(3.5), 4);
/// assert_eq!(saturate_cast::<u8>(300.0), 255);
/// assert_eq!(saturate_cast::<i8>(-1e9), -128);
/// assert_eq!(saturate_cast::<u16>(f64::NAN), 0);
/// assert_eq!(saturate_cast::<f32>(0.5), 0.5);
/// assert_eq!(i16::saturate_from_f32(-2.5), -2);
/// ```
pub trait SaturateCast: Copy {
    /// Converts `v` to `Self` with OpenCV's `saturate_cast` semantics.
    fn saturate_from_f64(v: f64) -> Self;

    /// Converts `v` to `Self` with OpenCV's `saturate_cast` semantics.
    ///
    /// Every `f32` is exactly representable as `f64`, so this is the same
    /// as `saturate_from_f64(v as f64)`.
    #[inline]
    fn saturate_from_f32(v: f32) -> Self {
        Self::saturate_from_f64(v as f64)
    }

    /// Converts the integer `v` to `Self`: integer targets clamp, float
    /// targets are a plain cast. Unlike `saturate_from_f64(v as f64)` this is
    /// exact for `|v| > 2^53`. The default goes through `f64`.
    #[inline]
    fn saturate_from_i64(v: i64) -> Self {
        Self::saturate_from_f64(v as f64)
    }

    /// Converts the integer `v` to `Self`, like [`SaturateCast::saturate_from_i64`].
    #[inline]
    fn saturate_from_u64(v: u64) -> Self {
        Self::saturate_from_f64(v as f64)
    }

    /// `true` for the floating-point targets. [`Matrix::convert_to_scaled`](crate::core::Matrix::convert_to_scaled)
    /// uses it to choose OpenCV's working precision.
    const IS_FLOAT: bool = false;
}

macro_rules! impl_saturate_cast_int {
    ($($t:ty),*) => {$(
        impl SaturateCast for $t {
            #[inline]
            fn saturate_from_f64(v: f64) -> Self {
                // Float-to-int `as` saturates at the type bounds and maps NaN
                // to 0, so rounding first gives OpenCV's round-then-clamp.
                round_half_even(v) as $t
            }

            #[inline]
            fn saturate_from_i64(v: i64) -> Self {
                (v as i128).clamp(<$t>::MIN as i128, <$t>::MAX as i128) as $t
            }

            #[inline]
            fn saturate_from_u64(v: u64) -> Self {
                (v as i128).clamp(<$t>::MIN as i128, <$t>::MAX as i128) as $t
            }
        }
    )*};
}

impl_saturate_cast_int!(u8, i8, u16, i16, u32, i32, u64, i64);

impl SaturateCast for f32 {
    #[inline]
    fn saturate_from_f64(v: f64) -> Self {
        v as f32
    }

    #[inline]
    fn saturate_from_f32(v: f32) -> Self {
        v
    }

    #[inline]
    fn saturate_from_i64(v: i64) -> Self {
        v as f32
    }

    #[inline]
    fn saturate_from_u64(v: u64) -> Self {
        v as f32
    }

    const IS_FLOAT: bool = true;
}

impl SaturateCast for f64 {
    #[inline]
    fn saturate_from_f64(v: f64) -> Self {
        v
    }

    #[inline]
    fn saturate_from_i64(v: i64) -> Self {
        v as f64
    }

    #[inline]
    fn saturate_from_u64(v: u64) -> Self {
        v as f64
    }

    const IS_FLOAT: bool = true;
}

/// Converts `v` to `T` with OpenCV's `saturate_cast` semantics.
///
/// Shorthand for [`SaturateCast::saturate_from_f64`]; see [`SaturateCast`].
#[inline]
pub fn saturate_cast<T: SaturateCast>(v: f64) -> T {
    T::saturate_from_f64(v)
}

/// Element-wise `+`, `-`, `*`, `/` and `|a - b|` with OpenCV's overflow behaviour.
///
/// OpenCV writes the result of `cv::add`, `cv::subtract`, `cv::multiply` and
/// `cv::absdiff` through `saturate_cast`, so the 8- and 16-bit integer depths
/// clamp to the range of the type instead of wrapping or panicking. The
/// operations here run in the native width of the type and never go through
/// `f64`, so 64-bit values above 2⁵³ stay exact.
///
/// - `u8`, `i8`, `u16`, `i16`: saturate (`saturating_*`).
/// - `i32` (`CV_32S`): **wraps** (`wrapping_*`). OpenCV 4.10.0 computes these
///   four operations in 32-bit integers, measured with
///   `scripts/opencv_ref/core_saturating_ops.py`: `INT_MAX + 1 = INT_MIN`,
///   `INT_MIN - 1 = INT_MAX`, `INT_MAX * 2 = -2` and
///   `|INT_MIN - INT_MAX| = -1`.
/// - `u32`, `u64`, `i64` have no OpenCV depth and saturate.
/// - `f32`, `f64`: plain IEEE arithmetic, `|a - b|` is `(a - b).abs()`.
///
/// `sat_div` follows `op_div_scale` in `arithm.simd.hpp`
/// ([source](https://github.com/opencv/opencv/blob/4.10.0/modules/core/src/arithm.simd.hpp#L1573-L1601)),
/// with the scale of `cv::divide(a, b)` equal to 1:
///
/// - `u8`, `i8`, `u16`, `i16`, `i32`: the quotient is computed in `f32`
///   (`1.0f * (float)a / b`), rounded half to even and clamped, so `7 / 2` is
///   `4`. A zero divisor gives `0`. `i32` goes through `f32` as well, so
///   operands above 2²⁴ lose low bits first, as in OpenCV
///   (`16777217 / 1 = 16777216`). The one result OpenCV does not clamp is a
///   quotient of exactly 2³¹ (`INT_MAX / 1`, `INT_MIN / -1`): the x86
///   conversion turns it into `INT_MIN`, where purecv clamps to `i32::MAX`.
/// - `f32`, `f64`: plain IEEE division, so `x / 0` is `±inf` and `0 / 0` is NaN.
/// - `u32`, `u64`, `i64` have no OpenCV depth. The quotient is exact (no float
///   detour, so it stays exact above 2⁵³), rounded half to even like the
///   depths above and clamped; `x / 0` is `0`.
///
/// The one known difference for the saturating depths is `u16` multiply: OpenCV
/// converts the `double` product through a 32-bit `int`, so `65535 * 65535`
/// gives `0` there and `65535` here (the same `INT_MIN` effect that
/// [`SaturateCast`] documents).
///
/// # Example
/// ```
/// use purecv::core::SaturatingOps;
///
/// assert_eq!(200u8.sat_add(100), 255);
/// assert_eq!(1u8.sat_sub(3), 0);
/// assert_eq!((-128i8).sat_absdiff(127), 127);
/// assert_eq!(i32::MAX.sat_add(1), i32::MIN); // CV_32S wraps in OpenCV
/// assert_eq!(7u8.sat_div(2), 4); // the quotient is rounded, not truncated
/// assert_eq!(7u8.sat_div(0), 0);
/// assert_eq!(1.0f32.sat_div(0.0), f32::INFINITY);
/// ```
pub trait SaturatingOps: Copy {
    /// `self + rhs` with OpenCV's overflow behaviour.
    fn sat_add(self, rhs: Self) -> Self;
    /// `self - rhs` with OpenCV's overflow behaviour.
    fn sat_sub(self, rhs: Self) -> Self;
    /// `self * rhs` with OpenCV's overflow behaviour.
    fn sat_mul(self, rhs: Self) -> Self;
    /// `|self - rhs|` with OpenCV's overflow behaviour.
    fn sat_absdiff(self, rhs: Self) -> Self;
    /// `self / rhs` with OpenCV's rounding and division-by-zero behaviour.
    fn sat_div(self, rhs: Self) -> Self;
}

/// `a / b` for the depths OpenCV divides in `float` (`op_div_scale`): the
/// float quotient, rounded half to even and clamped; `b == 0` gives `0`.
#[inline]
fn div_f32_working<T: SaturateCast + AsPrimitive<f32>>(a: T, b: T) -> T {
    let (fa, fb): (f32, f32) = (a.as_(), b.as_());
    if fb == 0.0 {
        T::saturate_from_f64(0.0)
    } else {
        T::saturate_from_f32(fa / fb)
    }
}

/// `a / b` rounded half to even in exact integer arithmetic, clamped to `T`;
/// `b == 0` gives `0`. For the types that have no OpenCV depth.
#[inline]
fn div_exact<T: Bounded + AsPrimitive<i128>>(a: T, b: T) -> T
where
    i128: AsPrimitive<T>,
{
    let (a, b): (i128, i128) = (a.as_(), b.as_());
    if b == 0 {
        return AsPrimitive::<T>::as_(0i128);
    }
    let (q, r) = (a / b, a % b);
    // |2r| > |b| rounds away from zero; a tie goes to the even quotient.
    let twice = r.unsigned_abs() * 2;
    let q = if twice > b.unsigned_abs() || (twice == b.unsigned_abs() && q % 2 != 0) {
        q + if (a < 0) != (b < 0) { -1 } else { 1 }
    } else {
        q
    };
    let (lo, hi): (i128, i128) = (T::min_value().as_(), T::max_value().as_());
    AsPrimitive::<T>::as_(q.clamp(lo, hi))
}

macro_rules! impl_saturating_ops_unsigned {
    ($div:ident; $($t:ty),*) => {$(
        impl SaturatingOps for $t {
            #[inline]
            fn sat_add(self, rhs: Self) -> Self {
                self.saturating_add(rhs)
            }
            #[inline]
            fn sat_sub(self, rhs: Self) -> Self {
                self.saturating_sub(rhs)
            }
            #[inline]
            fn sat_mul(self, rhs: Self) -> Self {
                self.saturating_mul(rhs)
            }
            #[inline]
            fn sat_absdiff(self, rhs: Self) -> Self {
                self.abs_diff(rhs)
            }
            #[inline]
            fn sat_div(self, rhs: Self) -> Self {
                $div(self, rhs)
            }
        }
    )*};
}

macro_rules! impl_saturating_ops_signed {
    ($div:ident; $($t:ty),*) => {$(
        impl SaturatingOps for $t {
            #[inline]
            fn sat_add(self, rhs: Self) -> Self {
                self.saturating_add(rhs)
            }
            #[inline]
            fn sat_sub(self, rhs: Self) -> Self {
                self.saturating_sub(rhs)
            }
            #[inline]
            fn sat_mul(self, rhs: Self) -> Self {
                self.saturating_mul(rhs)
            }
            #[inline]
            fn sat_absdiff(self, rhs: Self) -> Self {
                // `abs_diff` returns the unsigned type; OpenCV clamps
                // |127 - (-128)| to 127 in the signed type instead.
                if self > rhs {
                    self.saturating_sub(rhs)
                } else {
                    rhs.saturating_sub(self)
                }
            }
            #[inline]
            fn sat_div(self, rhs: Self) -> Self {
                $div(self, rhs)
            }
        }
    )*};
}

macro_rules! impl_saturating_ops_float {
    ($($t:ty),*) => {$(
        impl SaturatingOps for $t {
            #[inline]
            fn sat_add(self, rhs: Self) -> Self {
                self + rhs
            }
            #[inline]
            fn sat_sub(self, rhs: Self) -> Self {
                self - rhs
            }
            #[inline]
            fn sat_mul(self, rhs: Self) -> Self {
                self * rhs
            }
            #[inline]
            fn sat_absdiff(self, rhs: Self) -> Self {
                (self - rhs).abs()
            }
            #[inline]
            fn sat_div(self, rhs: Self) -> Self {
                self / rhs
            }
        }
    )*};
}

impl_saturating_ops_unsigned!(div_f32_working; u8, u16);
impl_saturating_ops_unsigned!(div_exact; u32, u64);
impl_saturating_ops_signed!(div_f32_working; i8, i16);
impl_saturating_ops_signed!(div_exact; i64);
impl_saturating_ops_float!(f32, f64);

// CV_32S wraps in OpenCV 4.10.0 (measured, see the trait documentation), so
// `i32` is the one integer type that does not clamp.
impl SaturatingOps for i32 {
    #[inline]
    fn sat_add(self, rhs: Self) -> Self {
        self.wrapping_add(rhs)
    }
    #[inline]
    fn sat_sub(self, rhs: Self) -> Self {
        self.wrapping_sub(rhs)
    }
    #[inline]
    fn sat_mul(self, rhs: Self) -> Self {
        self.wrapping_mul(rhs)
    }
    #[inline]
    fn sat_absdiff(self, rhs: Self) -> Self {
        if self > rhs {
            self.wrapping_sub(rhs)
        } else {
            rhs.wrapping_sub(self)
        }
    }
    #[inline]
    fn sat_div(self, rhs: Self) -> Self {
        div_f32_working(self, rhs)
    }
}

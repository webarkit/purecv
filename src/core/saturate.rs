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
}

impl SaturateCast for f64 {
    #[inline]
    fn saturate_from_f64(v: f64) -> Self {
        v
    }
}

/// Converts `v` to `T` with OpenCV's `saturate_cast` semantics.
///
/// Shorthand for [`SaturateCast::saturate_from_f64`]; see [`SaturateCast`].
#[inline]
pub fn saturate_cast<T: SaturateCast>(v: f64) -> T {
    T::saturate_from_f64(v)
}

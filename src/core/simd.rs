/*
 *  simd.rs
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

//! SIMD abstraction layer for PureCV.
//!
//! This module provides a trait-based dispatch system that routes element-wise
//! operations to `pulp`-powered SIMD kernels for supported types (`f32`, `f64`,
//! `u8`) and falls back to a no-op for all other types. The no-op fallback
//! causes the caller's scalar loop to execute instead, at zero cost.
//!
//! The trait [`SimdElement`] is **always compiled** (regardless of the `simd`
//! feature flag) so that generic functions can unconditionally require
//! `T: SimdElement`. When the `simd` feature is disabled, every method is a
//! default no-op and `has_simd()` returns `false`.
//!
//! # Portability
//!
//! `pulp` auto-detects CPU features at runtime (x86 SSE/AVX, ARM NEON,
//! WASM `simd128`). No target-specific `#[cfg(target_arch)]` is needed —
//! the same source compiles for native and `wasm32` targets.

use crate::core::types::{BorderTypes, Scalar};

// ---------------------------------------------------------------------------
//  SimdElement trait — compile-time type routing (always available)
// ---------------------------------------------------------------------------

/// Trait for types that *may* have SIMD-accelerated operations.
///
/// Default methods are all no-ops that return immediately.  Concrete
/// implementations for `f32`, `f64`, and `u8` override these methods
/// when the `simd` feature is enabled.
///
/// # Performance
///
/// Element-wise operations (`add`, `sub`, `mul`, `div`) are primarily
/// memory-bandwidth-bound with ~1.5x parallel speedup. Reduction operations
/// (`dot`, `sum`, `norm_l2_sq`) scale to ~6x with parallel. The standout is
/// `convert_scale_abs` which reaches ~5.3x with parallel + SIMD.
#[allow(dead_code)]
pub trait SimdElement: Copy + Send + Sync + 'static {
    /// Returns `true` if this type has SIMD kernel implementations
    /// *and* the `simd` feature is enabled at compile time.
    fn has_simd() -> bool {
        false
    }

    // -- Binary element-wise operations --
    // Return `true` if the operation was performed, `false` to fall back to scalar.

    fn simd_add(_dst: &mut [Self], _a: &[Self], _b: &[Self]) -> bool {
        false
    }
    fn simd_sub(_dst: &mut [Self], _a: &[Self], _b: &[Self]) -> bool {
        false
    }
    fn simd_mul(_dst: &mut [Self], _a: &[Self], _b: &[Self]) -> bool {
        false
    }
    fn simd_div(_dst: &mut [Self], _a: &[Self], _b: &[Self]) -> bool {
        false
    }
    fn simd_min(_dst: &mut [Self], _a: &[Self], _b: &[Self]) -> bool {
        false
    }
    fn simd_max(_dst: &mut [Self], _a: &[Self], _b: &[Self]) -> bool {
        false
    }

    // -- Unary operations --

    fn simd_sqrt(_dst: &mut [Self], _src: &[Self]) -> bool {
        false
    }
    fn simd_abs(_dst: &mut [Self], _src: &[Self]) -> bool {
        false
    }

    // -- Reductions --

    fn simd_dot(_a: &[Self], _b: &[Self]) -> Option<f64> {
        None
    }
    fn simd_sum(_src: &[Self]) -> Option<f64> {
        None
    }

    // -- Fused operations --

    /// dst = a * alpha + b * beta + gamma
    fn simd_add_weighted(
        _dst: &mut [Self],
        _a: &[Self],
        _b: &[Self],
        _alpha: f64,
        _beta: f64,
        _gamma: f64,
    ) -> bool {
        false
    }

    /// dst = |src * alpha + beta|, clamped to [0, 255] and written as u8.
    fn simd_convert_scale_abs(_dst: &mut [u8], _src: &[Self], _alpha: f64, _beta: f64) -> bool {
        false
    }

    /// dst = sqrt(x*x + y*y)
    fn simd_magnitude(_dst: &mut [Self], _x: &[Self], _y: &[Self]) -> bool {
        false
    }

    /// dst = |a - b|  (absolute difference)
    fn simd_absdiff(_dst: &mut [Self], _a: &[Self], _b: &[Self]) -> bool {
        false
    }

    /// dst = a & b  (bitwise AND)
    fn simd_bitwise_and(_dst: &mut [Self], _a: &[Self], _b: &[Self]) -> bool {
        false
    }

    /// dst = a | b  (bitwise OR)
    fn simd_bitwise_or(_dst: &mut [Self], _a: &[Self], _b: &[Self]) -> bool {
        false
    }

    /// dst = a ^ b  (bitwise XOR)
    fn simd_bitwise_xor(_dst: &mut [Self], _a: &[Self], _b: &[Self]) -> bool {
        false
    }

    /// dst = !src  (bitwise NOT)
    fn simd_bitwise_not(_dst: &mut [Self], _src: &[Self]) -> bool {
        false
    }

    /// Compute L2 norm squared: sum(src[i]^2) as f64
    fn simd_norm_l2_sq(_src: &[Self]) -> Option<f64> {
        None
    }

    /// Apply threshold operation on a flat slice.
    ///
    /// `thresh_type` values: 0 = BINARY, 1 = BINARY_INV, 2 = TRUNC,
    /// 3 = TOZERO, 4 = TOZERO_INV.
    ///
    /// Returns `true` if the operation was handled by SIMD, `false` to
    /// fall back to the scalar loop.
    fn simd_threshold(
        _dst: &mut [Self],
        _src: &[Self],
        _thresh: f64,
        _maxval: f64,
        _thresh_type: u8,
    ) -> bool {
        false
    }

    // -- Histogram comparisons --

    /// Compute compare_hist directly using the method variant mapped to a `u8`.
    /// 0=Correl, 1=ChiSqr, 2=ChiSqrAlt, 3=Intersection, 4=Bhattacharyya, 5=KullbackLeibler
    fn simd_compare_hist(_h1: &[Self], _h2: &[Self], _method: u8) -> Option<f64> {
        None
    }

    // -- Pyramid kernels (5-tap Gaussian) --

    /// Horizontal pass: dst[i] = sum(src[i - 2*stride .. i + 2*stride] * [1, 4, 6, 4, 1])
    fn simd_gaussian_5tap_h(_dst: &mut [f64], _src: &[Self], _stride: usize) -> bool {
        false
    }

    /// Vertical pass: dst[i] = sum(rows[k][i] * GAUSS5[k]) / 256
    /// Input is 5 rows of intermediate f64, output is final T.
    fn simd_gaussian_5tap_v(_dst: &mut [Self], _rows: &[&[f64]]) -> bool {
        false
    }

    // -- Morphology kernels (Row-wise min/max) --

    /// Horizontal pass: dst[i] = min/max(src[i-ksize/2..i+ksize/2])
    fn simd_row_min_max(_dst: &mut [Self], _src: &[Self], _ksize: usize, _is_erode: bool) -> bool {
        false
    }

    /// Vertical pass: dst[i] = min/max of rows[0..ksize][i]
    fn simd_min_max_col(_dst: &mut [Self], _rows: &[&[Self]], _is_erode: bool) -> bool {
        false
    }

    /// Remap a row with bilinear interpolation.
    #[allow(clippy::too_many_arguments)]
    fn simd_remap_bilinear_row(
        _dst_row: &mut [Self],
        _src: &[Self],
        _src_cols: usize,
        _src_rows: usize,
        _channels: usize,
        _map1_row: &[f32],
        _map2_row: &[f32],
        _border_mode: BorderTypes,
        _border_value: Scalar<Self>,
    ) -> bool {
        false
    }

    /// Remap a row with nearest neighbor interpolation.
    #[allow(clippy::too_many_arguments)]
    fn simd_remap_nearest_row(
        _dst_row: &mut [Self],
        _src: &[Self],
        _src_cols: usize,
        _src_rows: usize,
        _channels: usize,
        _map1_row: &[f32],
        _map2_row: &[f32],
        _border_mode: BorderTypes,
        _border_value: Scalar<Self>,
    ) -> bool {
        false
    }
}

// ---------------------------------------------------------------------------
//  Default no-op implementations for f32/f64/u8 when simd is DISABLED
// ---------------------------------------------------------------------------

#[cfg(not(feature = "simd"))]
mod no_simd_impls {
    use super::SimdElement;
    impl SimdElement for f32 {}
    impl SimdElement for f64 {}
    impl SimdElement for u8 {}
}

// Types that never have SIMD support regardless of feature flags.
impl SimdElement for i8 {}
impl SimdElement for i16 {}
impl SimdElement for u16 {}
impl SimdElement for i32 {}
impl SimdElement for u32 {}
impl SimdElement for i64 {}
impl SimdElement for u64 {}

// ---------------------------------------------------------------------------
//  SIMD-enabled implementations (only when `simd` feature is active)
// ---------------------------------------------------------------------------

#[cfg(feature = "simd")]
mod simd_impls {
    use super::SimdElement;
    use crate::core::types::{BorderTypes, Scalar};
    use crate::core::utils::border_interpolate;
    use pulp::Arch;

    // -----------------------------------------------------------------------
    //  f32
    // -----------------------------------------------------------------------

    impl SimdElement for f32 {
        fn has_simd() -> bool {
            true
        }

        fn simd_add(dst: &mut [Self], a: &[Self], b: &[Self]) -> bool {
            let arch = Arch::new();
            arch.dispatch(|| {
                for i in 0..dst.len() {
                    dst[i] = a[i] + b[i];
                }
            });
            true
        }

        fn simd_sub(dst: &mut [Self], a: &[Self], b: &[Self]) -> bool {
            let arch = Arch::new();
            arch.dispatch(|| {
                for i in 0..dst.len() {
                    dst[i] = a[i] - b[i];
                }
            });
            true
        }

        fn simd_mul(dst: &mut [Self], a: &[Self], b: &[Self]) -> bool {
            let arch = Arch::new();
            arch.dispatch(|| {
                for i in 0..dst.len() {
                    dst[i] = a[i] * b[i];
                }
            });
            true
        }

        fn simd_div(dst: &mut [Self], a: &[Self], b: &[Self]) -> bool {
            let arch = Arch::new();
            arch.dispatch(|| {
                for i in 0..dst.len() {
                    if b[i] != 0.0 {
                        dst[i] = a[i] / b[i];
                    } else {
                        dst[i] = 0.0;
                    }
                }
            });
            true
        }

        fn simd_min(dst: &mut [Self], a: &[Self], b: &[Self]) -> bool {
            let arch = Arch::new();
            arch.dispatch(|| {
                for i in 0..dst.len() {
                    dst[i] = if a[i] < b[i] { a[i] } else { b[i] };
                }
            });
            true
        }

        fn simd_max(dst: &mut [Self], a: &[Self], b: &[Self]) -> bool {
            let arch = Arch::new();
            arch.dispatch(|| {
                for i in 0..dst.len() {
                    dst[i] = if a[i] > b[i] { a[i] } else { b[i] };
                }
            });
            true
        }

        fn simd_sqrt(dst: &mut [Self], src: &[Self]) -> bool {
            let arch = Arch::new();
            arch.dispatch(|| {
                for i in 0..dst.len() {
                    dst[i] = src[i].sqrt();
                }
            });
            true
        }

        fn simd_abs(dst: &mut [Self], src: &[Self]) -> bool {
            let arch = Arch::new();
            arch.dispatch(|| {
                for i in 0..dst.len() {
                    dst[i] = src[i].abs();
                }
            });
            true
        }

        fn simd_dot(a: &[Self], b: &[Self]) -> Option<f64> {
            // Reduction: accumulate without dispatch to avoid value-propagation
            // issues with pulp closures. LLVM auto-vectorizes the loop.
            let mut acc = 0.0f64;
            for i in 0..a.len() {
                acc += a[i] as f64 * b[i] as f64;
            }
            Some(acc)
        }

        fn simd_sum(src: &[Self]) -> Option<f64> {
            // Reduction: accumulate directly; dispatch is not needed here.
            let mut acc = 0.0f64;
            for &v in src {
                acc += v as f64;
            }
            Some(acc)
        }

        fn simd_add_weighted(
            dst: &mut [Self],
            a: &[Self],
            b: &[Self],
            alpha: f64,
            beta: f64,
            gamma: f64,
        ) -> bool {
            let alpha_f32 = alpha as f32;
            let beta_f32 = beta as f32;
            let gamma_f32 = gamma as f32;
            let arch = Arch::new();
            arch.dispatch(|| {
                for i in 0..dst.len() {
                    dst[i] = a[i] * alpha_f32 + b[i] * beta_f32 + gamma_f32;
                }
            });
            true
        }

        fn simd_convert_scale_abs(dst: &mut [u8], src: &[Self], alpha: f64, beta: f64) -> bool {
            let alpha_f32 = alpha as f32;
            let beta_f32 = beta as f32;
            let arch = Arch::new();
            arch.dispatch(|| {
                for i in 0..dst.len() {
                    let val = (src[i] * alpha_f32 + beta_f32).abs();
                    dst[i] = val.clamp(0.0, 255.0).round() as u8;
                }
            });
            true
        }

        fn simd_magnitude(dst: &mut [Self], x: &[Self], y: &[Self]) -> bool {
            let arch = Arch::new();
            arch.dispatch(|| {
                for i in 0..dst.len() {
                    dst[i] = (x[i] * x[i] + y[i] * y[i]).sqrt();
                }
            });
            true
        }

        fn simd_absdiff(dst: &mut [Self], a: &[Self], b: &[Self]) -> bool {
            let arch = Arch::new();
            arch.dispatch(|| {
                for i in 0..dst.len() {
                    dst[i] = (a[i] - b[i]).abs();
                }
            });
            true
        }

        fn simd_norm_l2_sq(src: &[Self]) -> Option<f64> {
            let mut acc = 0.0f64;
            for &v in src {
                acc += v as f64 * v as f64;
            }
            Some(acc)
        }

        fn simd_threshold(
            dst: &mut [Self],
            src: &[Self],
            thresh: f64,
            maxval: f64,
            thresh_type: u8,
        ) -> bool {
            if thresh_type > 4 {
                return false;
            }
            let t = thresh as f32;
            let m = maxval as f32;
            let arch = Arch::new();
            match thresh_type {
                0 => arch.dispatch(|| {
                    for i in 0..dst.len() {
                        dst[i] = if src[i] > t { m } else { 0.0 };
                    }
                }),
                1 => arch.dispatch(|| {
                    for i in 0..dst.len() {
                        dst[i] = if src[i] > t { 0.0 } else { m };
                    }
                }),
                2 => arch.dispatch(|| {
                    for i in 0..dst.len() {
                        dst[i] = if src[i] > t { t } else { src[i] };
                    }
                }),
                3 => arch.dispatch(|| {
                    for i in 0..dst.len() {
                        dst[i] = if src[i] > t { src[i] } else { 0.0 };
                    }
                }),
                4 => arch.dispatch(|| {
                    for i in 0..dst.len() {
                        dst[i] = if src[i] > t { 0.0 } else { src[i] };
                    }
                }),
                _ => return false,
            }
            true
        }

        fn simd_compare_hist(h1: &[Self], h2: &[Self], method: u8) -> Option<f64> {
            match method {
                0 => {
                    // Correl
                    let mut s1 = 0.0f32;
                    let mut s2 = 0.0f32;
                    let mut s11 = 0.0f32;
                    let mut s12 = 0.0f32;
                    let mut s22 = 0.0f32;
                    for (a, b) in h1.iter().copied().zip(h2.iter().copied()) {
                        s1 += a;
                        s2 += b;
                        s11 += a * a;
                        s22 += b * b;
                        s12 += a * b;
                    }
                    let n = h1.len() as f64;
                    let scale = 1.0 / n;
                    let s1_f = s1 as f64;
                    let s2_f = s2 as f64;
                    let s11_f = s11 as f64;
                    let s12_f = s12 as f64;
                    let s22_f = s22 as f64;

                    let num = s12_f - s1_f * s2_f * scale;
                    let denom2 = (s11_f - s1_f * s1_f * scale) * (s22_f - s2_f * s2_f * scale);
                    Some(if denom2.abs() > f64::EPSILON {
                        num / denom2.sqrt()
                    } else {
                        1.0
                    })
                }
                1 => {
                    // ChiSqr
                    let mut result = 0.0f32;
                    for (a, b) in h1.iter().copied().zip(h2.iter().copied()) {
                        let diff = a - b;
                        let a_adj = if a.abs() <= f32::EPSILON { 1.0 } else { a };
                        let val = diff * diff / a_adj;
                        let add_val = if a.abs() > f32::EPSILON { val } else { 0.0 };
                        result += add_val;
                    }
                    Some(result as f64)
                }
                2 => {
                    // ChiSqrAlt
                    let mut result = 0.0f32;
                    for (a, b) in h1.iter().copied().zip(h2.iter().copied()) {
                        let sum = a + b;
                        let diff = a - b;
                        let sum_adj = if sum.abs() <= f32::EPSILON { 1.0 } else { sum };
                        let val = diff * diff / sum_adj;
                        let add_val = if sum.abs() > f32::EPSILON { val } else { 0.0 };
                        result += add_val;
                    }
                    Some((result as f64) * 2.0)
                }
                3 => {
                    // Intersection
                    let mut result = 0.0f32;
                    for (a, b) in h1.iter().copied().zip(h2.iter().copied()) {
                        result += if a < b { a } else { b };
                    }
                    Some(result as f64)
                }
                4 => {
                    // Bhattacharyya
                    let mut s1 = 0.0f32;
                    let mut s2 = 0.0f32;
                    let mut bc = 0.0f32;
                    for (a, b) in h1.iter().copied().zip(h2.iter().copied()) {
                        s1 += a;
                        s2 += b;
                        bc += (a * b).sqrt();
                    }
                    let s1_f = s1 as f64;
                    let s2_f = s2 as f64;
                    let bc_f = bc as f64;
                    let norm = s1_f * s2_f;
                    let norm_factor = if norm.abs() > f64::EPSILON {
                        1.0 / norm.sqrt()
                    } else {
                        1.0
                    };
                    Some(((1.0 - bc_f * norm_factor).max(0.0)).sqrt())
                }
                5 => {
                    // KullbackLeibler
                    let mut result = 0.0f32;
                    for (a, b) in h1.iter().copied().zip(h2.iter().copied()) {
                        let q_adj = if b.abs() <= f32::EPSILON { 1e-10 } else { b };
                        let p_adj = if a.abs() <= f32::EPSILON { 1.0 } else { a };
                        let val = a * (p_adj / q_adj).ln();
                        let add_val = if a.abs() > f32::EPSILON { val } else { 0.0 };
                        result += add_val;
                    }
                    Some(result as f64)
                }
                _ => None,
            }
        }

        fn simd_gaussian_5tap_h(dst: &mut [f64], src: &[Self], stride: usize) -> bool {
            let arch = pulp::Arch::new();
            arch.dispatch(|| {
                let s1 = stride;
                let s2 = stride * 2;
                let s3 = stride * 3;
                let s4 = stride * 4;
                for i in 0..dst.len() {
                    let v0 = src[i] as f64;
                    let v1 = src[i + s1] as f64;
                    let v2 = src[i + s2] as f64;
                    let v3 = src[i + s3] as f64;
                    let v4 = src[i + s4] as f64;
                    dst[i] = v0 + v1 * 4.0 + v2 * 6.0 + v3 * 4.0 + v4;
                }
            });
            true
        }

        fn simd_gaussian_5tap_v(dst: &mut [Self], rows: &[&[f64]]) -> bool {
            let arch = pulp::Arch::new();
            let r0 = rows[0];
            let r1 = rows[1];
            let r2 = rows[2];
            let r3 = rows[3];
            let r4 = rows[4];
            arch.dispatch(|| {
                for i in 0..dst.len() {
                    let sum = r0[i] + r1[i] * 4.0 + r2[i] * 6.0 + r3[i] * 4.0 + r4[i];
                    dst[i] = (sum / 256.0).round() as f32;
                }
            });
            true
        }

        fn simd_row_min_max(dst: &mut [Self], src: &[Self], ksize: usize, is_erode: bool) -> bool {
            let arch = pulp::Arch::new();
            arch.dispatch(|| {
                for i in 0..dst.len() {
                    let mut acc = src[i];
                    for j in 1..ksize {
                        let val = src[i + j];
                        if is_erode {
                            if val < acc {
                                acc = val;
                            }
                        } else if val > acc {
                            acc = val;
                        }
                    }
                    dst[i] = acc;
                }
            });
            true
        }

        fn simd_min_max_col(dst: &mut [Self], rows: &[&[Self]], is_erode: bool) -> bool {
            let arch = pulp::Arch::new();
            arch.dispatch(|| {
                for i in 0..dst.len() {
                    let mut acc = rows[0][i];
                    for row in &rows[1..] {
                        let val = row[i];
                        if is_erode {
                            if val < acc {
                                acc = val;
                            }
                        } else if val > acc {
                            acc = val;
                        }
                    }
                    dst[i] = acc;
                }
            });
            true
        }

        fn simd_remap_bilinear_row(
            dst_row: &mut [Self],
            src: &[Self],
            src_cols: usize,
            src_rows: usize,
            channels: usize,
            map1_row: &[f32],
            map2_row: &[f32],
            border_mode: BorderTypes,
            border_value: Scalar<Self>,
        ) -> bool {
            let arch = pulp::Arch::new();
            arch.dispatch(|| {
                let dst_cols = dst_row.len() / channels;
                for col in 0..dst_cols {
                    let x = map1_row[col];
                    let y = map2_row[col];

                    let x1 = x.floor() as i32;
                    let y1 = y.floor() as i32;
                    let x2 = x1 + 1;
                    let y2 = y1 + 1;

                    let wx = x - x.floor();
                    let wy = y - y.floor();

                    let w00 = (1.0 - wx) * (1.0 - wy);
                    let w10 = wx * (1.0 - wy);
                    let w01 = (1.0 - wx) * wy;
                    let w11 = wx * wy;

                    let dst_idx = col * channels;

                    if x1 >= 0 && x2 < src_cols as i32 && y1 >= 0 && y2 < src_rows as i32 {
                        let idx00 = (y1 as usize * src_cols + x1 as usize) * channels;
                        let idx10 = (y1 as usize * src_cols + x2 as usize) * channels;
                        let idx01 = (y2 as usize * src_cols + x1 as usize) * channels;
                        let idx11 = (y2 as usize * src_cols + x2 as usize) * channels;

                        for c in 0..channels {
                            let v00 = src[idx00 + c];
                            let v10 = src[idx10 + c];
                            let v01 = src[idx01 + c];
                            let v11 = src[idx11 + c];
                            dst_row[dst_idx + c] = w00 * v00 + w10 * v10 + w01 * v01 + w11 * v11;
                        }
                    } else {
                        for c in 0..channels {
                            let get_pixel = |ix: i32, iy: i32| -> f32 {
                                let ix_interp =
                                    border_interpolate(ix, src_cols as i32, border_mode);
                                let iy_interp =
                                    border_interpolate(iy, src_rows as i32, border_mode);
                                if ix_interp >= 0 && iy_interp >= 0 {
                                    src[(iy_interp as usize * src_cols + ix_interp as usize)
                                        * channels
                                        + c]
                                } else {
                                    border_value.v[c]
                                }
                            };

                            let v00 = get_pixel(x1, y1);
                            let v10 = get_pixel(x2, y1);
                            let v01 = get_pixel(x1, y2);
                            let v11 = get_pixel(x2, y2);
                            dst_row[dst_idx + c] = w00 * v00 + w10 * v10 + w01 * v01 + w11 * v11;
                        }
                    }
                }
            });
            true
        }

        fn simd_remap_nearest_row(
            dst_row: &mut [Self],
            src: &[Self],
            src_cols: usize,
            src_rows: usize,
            channels: usize,
            map1_row: &[f32],
            map2_row: &[f32],
            border_mode: BorderTypes,
            border_value: Scalar<Self>,
        ) -> bool {
            let arch = pulp::Arch::new();
            arch.dispatch(|| {
                let dst_cols = dst_row.len() / channels;
                for col in 0..dst_cols {
                    let x = map1_row[col];
                    let y = map2_row[col];
                    let ix = x.round() as i32;
                    let iy = y.round() as i32;
                    let dst_idx = col * channels;

                    if ix >= 0 && ix < src_cols as i32 && iy >= 0 && iy < src_rows as i32 {
                        let src_idx = (iy as usize * src_cols + ix as usize) * channels;
                        dst_row[dst_idx..dst_idx + channels]
                            .copy_from_slice(&src[src_idx..src_idx + channels]);
                    } else {
                        let ix_interp = border_interpolate(ix, src_cols as i32, border_mode);
                        let iy_interp = border_interpolate(iy, src_rows as i32, border_mode);
                        if ix_interp >= 0 && iy_interp >= 0 {
                            let src_idx =
                                (iy_interp as usize * src_cols + ix_interp as usize) * channels;
                            dst_row[dst_idx..dst_idx + channels]
                                .copy_from_slice(&src[src_idx..src_idx + channels]);
                        } else {
                            dst_row[dst_idx..dst_idx + channels]
                                .copy_from_slice(&border_value.v[..channels]);
                        }
                    }
                }
            });
            true
        }
    }

    // -----------------------------------------------------------------------
    //  f64
    // -----------------------------------------------------------------------

    impl SimdElement for f64 {
        fn has_simd() -> bool {
            true
        }

        fn simd_add(dst: &mut [Self], a: &[Self], b: &[Self]) -> bool {
            let arch = Arch::new();
            arch.dispatch(|| {
                for i in 0..dst.len() {
                    dst[i] = a[i] + b[i];
                }
            });
            true
        }

        fn simd_sub(dst: &mut [Self], a: &[Self], b: &[Self]) -> bool {
            let arch = Arch::new();
            arch.dispatch(|| {
                for i in 0..dst.len() {
                    dst[i] = a[i] - b[i];
                }
            });
            true
        }

        fn simd_mul(dst: &mut [Self], a: &[Self], b: &[Self]) -> bool {
            let arch = Arch::new();
            arch.dispatch(|| {
                for i in 0..dst.len() {
                    dst[i] = a[i] * b[i];
                }
            });
            true
        }

        fn simd_div(dst: &mut [Self], a: &[Self], b: &[Self]) -> bool {
            let arch = Arch::new();
            arch.dispatch(|| {
                for i in 0..dst.len() {
                    if b[i] != 0.0 {
                        dst[i] = a[i] / b[i];
                    } else {
                        dst[i] = 0.0;
                    }
                }
            });
            true
        }

        fn simd_min(dst: &mut [Self], a: &[Self], b: &[Self]) -> bool {
            let arch = Arch::new();
            arch.dispatch(|| {
                for i in 0..dst.len() {
                    dst[i] = if a[i] < b[i] { a[i] } else { b[i] };
                }
            });
            true
        }

        fn simd_max(dst: &mut [Self], a: &[Self], b: &[Self]) -> bool {
            let arch = Arch::new();
            arch.dispatch(|| {
                for i in 0..dst.len() {
                    dst[i] = if a[i] > b[i] { a[i] } else { b[i] };
                }
            });
            true
        }

        fn simd_sqrt(dst: &mut [Self], src: &[Self]) -> bool {
            let arch = Arch::new();
            arch.dispatch(|| {
                for i in 0..dst.len() {
                    dst[i] = src[i].sqrt();
                }
            });
            true
        }

        fn simd_abs(dst: &mut [Self], src: &[Self]) -> bool {
            let arch = Arch::new();
            arch.dispatch(|| {
                for i in 0..dst.len() {
                    dst[i] = src[i].abs();
                }
            });
            true
        }

        fn simd_dot(a: &[Self], b: &[Self]) -> Option<f64> {
            // Reduction: accumulate without dispatch to avoid value-propagation
            // issues with pulp closures. LLVM auto-vectorizes the loop.
            let mut acc = 0.0f64;
            for i in 0..a.len() {
                acc += a[i] * b[i];
            }
            Some(acc)
        }

        fn simd_sum(src: &[Self]) -> Option<f64> {
            // Reduction: accumulate directly; dispatch is not needed here.
            let mut acc = 0.0f64;
            for &v in src {
                acc += v;
            }
            Some(acc)
        }

        fn simd_add_weighted(
            dst: &mut [Self],
            a: &[Self],
            b: &[Self],
            alpha: f64,
            beta: f64,
            gamma: f64,
        ) -> bool {
            let arch = Arch::new();
            arch.dispatch(|| {
                for i in 0..dst.len() {
                    dst[i] = a[i] * alpha + b[i] * beta + gamma;
                }
            });
            true
        }

        fn simd_convert_scale_abs(dst: &mut [u8], src: &[Self], alpha: f64, beta: f64) -> bool {
            let arch = Arch::new();
            arch.dispatch(|| {
                for i in 0..dst.len() {
                    let val = (src[i] * alpha + beta).abs();
                    dst[i] = val.clamp(0.0, 255.0).round() as u8;
                }
            });
            true
        }

        fn simd_magnitude(dst: &mut [Self], x: &[Self], y: &[Self]) -> bool {
            let arch = Arch::new();
            arch.dispatch(|| {
                for i in 0..dst.len() {
                    dst[i] = (x[i] * x[i] + y[i] * y[i]).sqrt();
                }
            });
            true
        }

        fn simd_absdiff(dst: &mut [Self], a: &[Self], b: &[Self]) -> bool {
            let arch = Arch::new();
            arch.dispatch(|| {
                for i in 0..dst.len() {
                    dst[i] = (a[i] - b[i]).abs();
                }
            });
            true
        }

        fn simd_norm_l2_sq(src: &[Self]) -> Option<f64> {
            let mut acc = 0.0f64;
            for &v in src {
                acc += v * v;
            }
            Some(acc)
        }

        fn simd_threshold(
            dst: &mut [Self],
            src: &[Self],
            thresh: f64,
            maxval: f64,
            thresh_type: u8,
        ) -> bool {
            if thresh_type > 4 {
                return false;
            }
            let t = thresh;
            let m = maxval;
            let arch = Arch::new();
            match thresh_type {
                0 => arch.dispatch(|| {
                    for i in 0..dst.len() {
                        dst[i] = if src[i] > t { m } else { 0.0 };
                    }
                }),
                1 => arch.dispatch(|| {
                    for i in 0..dst.len() {
                        dst[i] = if src[i] > t { 0.0 } else { m };
                    }
                }),
                2 => arch.dispatch(|| {
                    for i in 0..dst.len() {
                        dst[i] = if src[i] > t { t } else { src[i] };
                    }
                }),
                3 => arch.dispatch(|| {
                    for i in 0..dst.len() {
                        dst[i] = if src[i] > t { src[i] } else { 0.0 };
                    }
                }),
                4 => arch.dispatch(|| {
                    for i in 0..dst.len() {
                        dst[i] = if src[i] > t { 0.0 } else { src[i] };
                    }
                }),
                _ => return false,
            }
            true
        }

        fn simd_compare_hist(h1: &[Self], h2: &[Self], method: u8) -> Option<f64> {
            match method {
                0 => {
                    // Correl
                    let mut s1 = 0.0f64;
                    let mut s2 = 0.0f64;
                    let mut s11 = 0.0f64;
                    let mut s12 = 0.0f64;
                    let mut s22 = 0.0f64;
                    for (a, b) in h1.iter().copied().zip(h2.iter().copied()) {
                        s1 += a;
                        s2 += b;
                        s11 += a * a;
                        s22 += b * b;
                        s12 += a * b;
                    }
                    let n = h1.len() as f64;
                    let scale = 1.0 / n;

                    let num = s12 - s1 * s2 * scale;
                    let denom2 = (s11 - s1 * s1 * scale) * (s22 - s2 * s2 * scale);
                    Some(if denom2.abs() > f64::EPSILON {
                        num / denom2.sqrt()
                    } else {
                        1.0
                    })
                }
                1 => {
                    // ChiSqr
                    let mut result = 0.0f64;
                    for (a, b) in h1.iter().copied().zip(h2.iter().copied()) {
                        let diff = a - b;
                        let a_adj = if a.abs() <= f64::EPSILON { 1.0 } else { a };
                        let val = diff * diff / a_adj;
                        let add_val = if a.abs() > f64::EPSILON { val } else { 0.0 };
                        result += add_val;
                    }
                    Some(result)
                }
                2 => {
                    // ChiSqrAlt
                    let mut result = 0.0f64;
                    for (a, b) in h1.iter().copied().zip(h2.iter().copied()) {
                        let sum = a + b;
                        let diff = a - b;
                        let sum_adj = if sum.abs() <= f64::EPSILON { 1.0 } else { sum };
                        let val = diff * diff / sum_adj;
                        let add_val = if sum.abs() > f64::EPSILON { val } else { 0.0 };
                        result += add_val;
                    }
                    Some(result * 2.0)
                }
                3 => {
                    // Intersection
                    let mut result = 0.0f64;
                    for (a, b) in h1.iter().copied().zip(h2.iter().copied()) {
                        result += if a < b { a } else { b };
                    }
                    Some(result)
                }
                4 => {
                    // Bhattacharyya
                    let mut s1 = 0.0f64;
                    let mut s2 = 0.0f64;
                    let mut bc = 0.0f64;
                    for (a, b) in h1.iter().copied().zip(h2.iter().copied()) {
                        s1 += a;
                        s2 += b;
                        bc += (a * b).sqrt();
                    }
                    let norm = s1 * s2;
                    let norm_factor = if norm.abs() > f64::EPSILON {
                        1.0 / norm.sqrt()
                    } else {
                        1.0
                    };
                    Some(((1.0 - bc * norm_factor).max(0.0)).sqrt())
                }
                5 => {
                    // KullbackLeibler
                    let mut result = 0.0f64;
                    for (a, b) in h1.iter().copied().zip(h2.iter().copied()) {
                        let q_adj = if b.abs() <= f64::EPSILON { 1e-10 } else { b };
                        let p_adj = if a.abs() <= f64::EPSILON { 1.0 } else { a };
                        let val = a * (p_adj / q_adj).ln();
                        let add_val = if a.abs() > f64::EPSILON { val } else { 0.0 };
                        result += add_val;
                    }
                    Some(result)
                }
                _ => None,
            }
        }

        fn simd_gaussian_5tap_h(dst: &mut [f64], src: &[Self], stride: usize) -> bool {
            let arch = pulp::Arch::new();
            arch.dispatch(|| {
                let s1 = stride;
                let s2 = stride * 2;
                let s3 = stride * 3;
                let s4 = stride * 4;
                for i in 0..dst.len() {
                    let v0 = src[i];
                    let v1 = src[i + s1];
                    let v2 = src[i + s2];
                    let v3 = src[i + s3];
                    let v4 = src[i + s4];
                    dst[i] = v0 + v1 * 4.0 + v2 * 6.0 + v3 * 4.0 + v4;
                }
            });
            true
        }

        fn simd_gaussian_5tap_v(dst: &mut [Self], rows: &[&[f64]]) -> bool {
            let arch = pulp::Arch::new();
            let r0 = rows[0];
            let r1 = rows[1];
            let r2 = rows[2];
            let r3 = rows[3];
            let r4 = rows[4];
            arch.dispatch(|| {
                for i in 0..dst.len() {
                    let sum = r0[i] + r1[i] * 4.0 + r2[i] * 6.0 + r3[i] * 4.0 + r4[i];
                    dst[i] = (sum / 256.0).round();
                }
            });
            true
        }

        fn simd_row_min_max(dst: &mut [Self], src: &[Self], ksize: usize, is_erode: bool) -> bool {
            let arch = pulp::Arch::new();
            arch.dispatch(|| {
                for i in 0..dst.len() {
                    let mut acc = src[i];
                    for j in 1..ksize {
                        let val = src[i + j];
                        if is_erode {
                            if val < acc {
                                acc = val;
                            }
                        } else if val > acc {
                            acc = val;
                        }
                    }
                    dst[i] = acc;
                }
            });
            true
        }

        fn simd_min_max_col(dst: &mut [Self], rows: &[&[Self]], is_erode: bool) -> bool {
            let arch = pulp::Arch::new();
            arch.dispatch(|| {
                for i in 0..dst.len() {
                    let mut acc = rows[0][i];
                    for row in &rows[1..] {
                        let val = row[i];
                        if is_erode {
                            if val < acc {
                                acc = val;
                            }
                        } else if val > acc {
                            acc = val;
                        }
                    }
                    dst[i] = acc;
                }
            });
            true
        }

        fn simd_remap_bilinear_row(
            dst_row: &mut [Self],
            src: &[Self],
            src_cols: usize,
            src_rows: usize,
            channels: usize,
            map1_row: &[f32],
            map2_row: &[f32],
            border_mode: BorderTypes,
            border_value: Scalar<Self>,
        ) -> bool {
            let arch = pulp::Arch::new();
            arch.dispatch(|| {
                let dst_cols = dst_row.len() / channels;
                for col in 0..dst_cols {
                    let x = map1_row[col] as f64;
                    let y = map2_row[col] as f64;

                    let x1 = x.floor() as i32;
                    let y1 = y.floor() as i32;
                    let x2 = x1 + 1;
                    let y2 = y1 + 1;

                    let wx = x - x.floor();
                    let wy = y - y.floor();

                    let w00 = (1.0 - wx) * (1.0 - wy);
                    let w10 = wx * (1.0 - wy);
                    let w01 = (1.0 - wx) * wy;
                    let w11 = wx * wy;

                    let dst_idx = col * channels;

                    if x1 >= 0 && x2 < src_cols as i32 && y1 >= 0 && y2 < src_rows as i32 {
                        let idx00 = (y1 as usize * src_cols + x1 as usize) * channels;
                        let idx10 = (y1 as usize * src_cols + x2 as usize) * channels;
                        let idx01 = (y2 as usize * src_cols + x1 as usize) * channels;
                        let idx11 = (y2 as usize * src_cols + x2 as usize) * channels;

                        for c in 0..channels {
                            let v00 = src[idx00 + c];
                            let v10 = src[idx10 + c];
                            let v01 = src[idx01 + c];
                            let v11 = src[idx11 + c];
                            dst_row[dst_idx + c] = w00 * v00 + w10 * v10 + w01 * v01 + w11 * v11;
                        }
                    } else {
                        for c in 0..channels {
                            let get_pixel = |ix: i32, iy: i32| -> f64 {
                                let ix_interp =
                                    border_interpolate(ix, src_cols as i32, border_mode);
                                let iy_interp =
                                    border_interpolate(iy, src_rows as i32, border_mode);
                                if ix_interp >= 0 && iy_interp >= 0 {
                                    src[(iy_interp as usize * src_cols + ix_interp as usize)
                                        * channels
                                        + c]
                                } else {
                                    border_value.v[c]
                                }
                            };

                            let v00 = get_pixel(x1, y1);
                            let v10 = get_pixel(x2, y1);
                            let v01 = get_pixel(x1, y2);
                            let v11 = get_pixel(x2, y2);
                            dst_row[dst_idx + c] = w00 * v00 + w10 * v10 + w01 * v01 + w11 * v11;
                        }
                    }
                }
            });
            true
        }

        fn simd_remap_nearest_row(
            dst_row: &mut [Self],
            src: &[Self],
            src_cols: usize,
            src_rows: usize,
            channels: usize,
            map1_row: &[f32],
            map2_row: &[f32],
            border_mode: BorderTypes,
            border_value: Scalar<Self>,
        ) -> bool {
            let arch = pulp::Arch::new();
            arch.dispatch(|| {
                let dst_cols = dst_row.len() / channels;
                for col in 0..dst_cols {
                    let x = map1_row[col];
                    let y = map2_row[col];
                    let ix = x.round() as i32;
                    let iy = y.round() as i32;
                    let dst_idx = col * channels;

                    if ix >= 0 && ix < src_cols as i32 && iy >= 0 && iy < src_rows as i32 {
                        let src_idx = (iy as usize * src_cols + ix as usize) * channels;
                        dst_row[dst_idx..dst_idx + channels]
                            .copy_from_slice(&src[src_idx..src_idx + channels]);
                    } else {
                        let ix_interp = border_interpolate(ix, src_cols as i32, border_mode);
                        let iy_interp = border_interpolate(iy, src_rows as i32, border_mode);
                        if ix_interp >= 0 && iy_interp >= 0 {
                            let src_idx =
                                (iy_interp as usize * src_cols + ix_interp as usize) * channels;
                            dst_row[dst_idx..dst_idx + channels]
                                .copy_from_slice(&src[src_idx..src_idx + channels]);
                        } else {
                            dst_row[dst_idx..dst_idx + channels]
                                .copy_from_slice(&border_value.v[..channels]);
                        }
                    }
                }
            });
            true
        }
    }

    // -----------------------------------------------------------------------
    //  u8 (limited set of SIMD-friendly operations)
    // -----------------------------------------------------------------------

    impl SimdElement for u8 {
        fn has_simd() -> bool {
            true
        }

        fn simd_add(dst: &mut [Self], a: &[Self], b: &[Self]) -> bool {
            let arch = Arch::new();
            arch.dispatch(|| {
                for i in 0..dst.len() {
                    dst[i] = a[i].saturating_add(b[i]);
                }
            });
            true
        }

        fn simd_sub(dst: &mut [Self], a: &[Self], b: &[Self]) -> bool {
            let arch = Arch::new();
            arch.dispatch(|| {
                for i in 0..dst.len() {
                    dst[i] = a[i].saturating_sub(b[i]);
                }
            });
            true
        }

        fn simd_min(dst: &mut [Self], a: &[Self], b: &[Self]) -> bool {
            let arch = Arch::new();
            arch.dispatch(|| {
                for i in 0..dst.len() {
                    dst[i] = a[i].min(b[i]);
                }
            });
            true
        }

        fn simd_max(dst: &mut [Self], a: &[Self], b: &[Self]) -> bool {
            let arch = Arch::new();
            arch.dispatch(|| {
                for i in 0..dst.len() {
                    dst[i] = a[i].max(b[i]);
                }
            });
            true
        }

        fn simd_sum(src: &[Self]) -> Option<f64> {
            // Reduction: accumulate directly; dispatch is not needed here.
            let mut acc = 0u64;
            for &v in src {
                acc += v as u64;
            }
            Some(acc as f64)
        }

        fn simd_absdiff(dst: &mut [Self], a: &[Self], b: &[Self]) -> bool {
            let arch = Arch::new();
            arch.dispatch(|| {
                for i in 0..dst.len() {
                    dst[i] = a[i].abs_diff(b[i]);
                }
            });
            true
        }

        fn simd_bitwise_and(dst: &mut [Self], a: &[Self], b: &[Self]) -> bool {
            let arch = Arch::new();
            arch.dispatch(|| {
                for i in 0..dst.len() {
                    dst[i] = a[i] & b[i];
                }
            });
            true
        }

        fn simd_bitwise_or(dst: &mut [Self], a: &[Self], b: &[Self]) -> bool {
            let arch = Arch::new();
            arch.dispatch(|| {
                for i in 0..dst.len() {
                    dst[i] = a[i] | b[i];
                }
            });
            true
        }

        fn simd_bitwise_xor(dst: &mut [Self], a: &[Self], b: &[Self]) -> bool {
            let arch = Arch::new();
            arch.dispatch(|| {
                for i in 0..dst.len() {
                    dst[i] = a[i] ^ b[i];
                }
            });
            true
        }

        fn simd_bitwise_not(dst: &mut [Self], src: &[Self]) -> bool {
            let arch = Arch::new();
            arch.dispatch(|| {
                for i in 0..dst.len() {
                    dst[i] = !src[i];
                }
            });
            true
        }

        fn simd_norm_l2_sq(src: &[Self]) -> Option<f64> {
            let mut acc = 0u64;
            for &v in src {
                acc += (v as u64) * (v as u64);
            }
            Some(acc as f64)
        }

        fn simd_threshold(
            dst: &mut [Self],
            src: &[Self],
            thresh: f64,
            maxval: f64,
            thresh_type: u8,
        ) -> bool {
            if thresh_type > 4 {
                return false;
            }
            let t = thresh.clamp(0.0, 255.0) as u8;
            let m = maxval.clamp(0.0, 255.0) as u8;
            let arch = Arch::new();
            match thresh_type {
                0 => arch.dispatch(|| {
                    for i in 0..dst.len() {
                        dst[i] = if src[i] > t { m } else { 0 };
                    }
                }),
                1 => arch.dispatch(|| {
                    for i in 0..dst.len() {
                        dst[i] = if src[i] > t { 0 } else { m };
                    }
                }),
                2 => arch.dispatch(|| {
                    for i in 0..dst.len() {
                        dst[i] = if src[i] > t { t } else { src[i] };
                    }
                }),
                3 => arch.dispatch(|| {
                    for i in 0..dst.len() {
                        dst[i] = if src[i] > t { src[i] } else { 0 };
                    }
                }),
                4 => arch.dispatch(|| {
                    for i in 0..dst.len() {
                        dst[i] = if src[i] > t { 0 } else { src[i] };
                    }
                }),
                _ => return false,
            }
            true
        }

        fn simd_gaussian_5tap_h(dst: &mut [f64], src: &[Self], stride: usize) -> bool {
            let arch = pulp::Arch::new();
            arch.dispatch(|| {
                let s1 = stride;
                let s2 = stride * 2;
                let s3 = stride * 3;
                let s4 = stride * 4;
                for i in 0..dst.len() {
                    let v0 = src[i] as f64;
                    let v1 = src[i + s1] as f64;
                    let v2 = src[i + s2] as f64;
                    let v3 = src[i + s3] as f64;
                    let v4 = src[i + s4] as f64;
                    dst[i] = v0 + v1 * 4.0 + v2 * 6.0 + v3 * 4.0 + v4;
                }
            });
            true
        }

        fn simd_gaussian_5tap_v(dst: &mut [Self], rows: &[&[f64]]) -> bool {
            let arch = pulp::Arch::new();
            let r0 = rows[0];
            let r1 = rows[1];
            let r2 = rows[2];
            let r3 = rows[3];
            let r4 = rows[4];
            arch.dispatch(|| {
                for i in 0..dst.len() {
                    let sum = r0[i] + r1[i] * 4.0 + r2[i] * 6.0 + r3[i] * 4.0 + r4[i];
                    dst[i] = (sum / 256.0).round() as u8;
                }
            });
            true
        }

        fn simd_row_min_max(dst: &mut [Self], src: &[Self], ksize: usize, is_erode: bool) -> bool {
            let arch = pulp::Arch::new();
            arch.dispatch(|| {
                for i in 0..dst.len() {
                    let mut acc = src[i];
                    for j in 1..ksize {
                        let val = src[i + j];
                        if is_erode {
                            if val < acc {
                                acc = val;
                            }
                        } else if val > acc {
                            acc = val;
                        }
                    }
                    dst[i] = acc;
                }
            });
            true
        }

        fn simd_min_max_col(dst: &mut [Self], rows: &[&[Self]], is_erode: bool) -> bool {
            let arch = pulp::Arch::new();
            arch.dispatch(|| {
                for i in 0..dst.len() {
                    let mut acc = rows[0][i];
                    for row in &rows[1..] {
                        let val = row[i];
                        if is_erode {
                            if val < acc {
                                acc = val;
                            }
                        } else if val > acc {
                            acc = val;
                        }
                    }
                    dst[i] = acc;
                }
            });
            true
        }

        fn simd_remap_bilinear_row(
            dst_row: &mut [Self],
            src: &[Self],
            src_cols: usize,
            src_rows: usize,
            channels: usize,
            map1_row: &[f32],
            map2_row: &[f32],
            border_mode: BorderTypes,
            border_value: Scalar<Self>,
        ) -> bool {
            let arch = pulp::Arch::new();
            arch.dispatch(|| {
                let dst_cols = dst_row.len() / channels;
                for col in 0..dst_cols {
                    let x = map1_row[col];
                    let y = map2_row[col];

                    let x1 = x.floor() as i32;
                    let y1 = y.floor() as i32;
                    let x2 = x1 + 1;
                    let y2 = y1 + 1;

                    let wx = x - x.floor();
                    let wy = y - y.floor();

                    let w00 = (1.0 - wx) * (1.0 - wy);
                    let w10 = wx * (1.0 - wy);
                    let w01 = (1.0 - wx) * wy;
                    let w11 = wx * wy;

                    let dst_idx = col * channels;

                    if x1 >= 0 && x2 < src_cols as i32 && y1 >= 0 && y2 < src_rows as i32 {
                        let idx00 = (y1 as usize * src_cols + x1 as usize) * channels;
                        let idx10 = (y1 as usize * src_cols + x2 as usize) * channels;
                        let idx01 = (y2 as usize * src_cols + x1 as usize) * channels;
                        let idx11 = (y2 as usize * src_cols + x2 as usize) * channels;

                        for c in 0..channels {
                            let v00 = src[idx00 + c] as f32;
                            let v10 = src[idx10 + c] as f32;
                            let v01 = src[idx01 + c] as f32;
                            let v11 = src[idx11 + c] as f32;
                            let val = w00 * v00 + w10 * v10 + w01 * v01 + w11 * v11;
                            dst_row[dst_idx + c] = val.clamp(0.0, 255.0).round() as u8;
                        }
                    } else {
                        for c in 0..channels {
                            let get_pixel = |ix: i32, iy: i32| -> f32 {
                                let ix_interp =
                                    border_interpolate(ix, src_cols as i32, border_mode);
                                let iy_interp =
                                    border_interpolate(iy, src_rows as i32, border_mode);
                                if ix_interp >= 0 && iy_interp >= 0 {
                                    src[(iy_interp as usize * src_cols + ix_interp as usize)
                                        * channels
                                        + c] as f32
                                } else {
                                    border_value.v[c] as f32
                                }
                            };

                            let v00 = get_pixel(x1, y1);
                            let v10 = get_pixel(x2, y1);
                            let v01 = get_pixel(x1, y2);
                            let v11 = get_pixel(x2, y2);
                            let val = w00 * v00 + w10 * v10 + w01 * v01 + w11 * v11;
                            dst_row[dst_idx + c] = val.clamp(0.0, 255.0).round() as u8;
                        }
                    }
                }
            });
            true
        }

        fn simd_remap_nearest_row(
            dst_row: &mut [Self],
            src: &[Self],
            src_cols: usize,
            src_rows: usize,
            channels: usize,
            map1_row: &[f32],
            map2_row: &[f32],
            border_mode: BorderTypes,
            border_value: Scalar<Self>,
        ) -> bool {
            let arch = pulp::Arch::new();
            arch.dispatch(|| {
                let dst_cols = dst_row.len() / channels;
                for col in 0..dst_cols {
                    let x = map1_row[col];
                    let y = map2_row[col];
                    let ix = x.round() as i32;
                    let iy = y.round() as i32;
                    let dst_idx = col * channels;

                    if ix >= 0 && ix < src_cols as i32 && iy >= 0 && iy < src_rows as i32 {
                        let src_idx = (iy as usize * src_cols + ix as usize) * channels;
                        dst_row[dst_idx..dst_idx + channels]
                            .copy_from_slice(&src[src_idx..src_idx + channels]);
                    } else {
                        let ix_interp = border_interpolate(ix, src_cols as i32, border_mode);
                        let iy_interp = border_interpolate(iy, src_rows as i32, border_mode);
                        if ix_interp >= 0 && iy_interp >= 0 {
                            let src_idx =
                                (iy_interp as usize * src_cols + ix_interp as usize) * channels;
                            dst_row[dst_idx..dst_idx + channels]
                                .copy_from_slice(&src[src_idx..src_idx + channels]);
                        } else {
                            dst_row[dst_idx..dst_idx + channels]
                                .copy_from_slice(&border_value.v[..channels]);
                        }
                    }
                }
            });
            true
        }
    }
}

// ---------------------------------------------------------------------------
//  Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_has_simd_default_types() {
        assert!(!i8::has_simd());
        assert!(!i16::has_simd());
        assert!(!u16::has_simd());
        assert!(!i32::has_simd());
        assert!(!u32::has_simd());
        assert!(!i64::has_simd());
        assert!(!u64::has_simd());
    }

    #[cfg(feature = "simd")]
    mod simd_tests {
        use super::super::*;

        #[test]
        fn test_has_simd_for_f32() {
            assert!(f32::has_simd());
        }

        #[test]
        fn test_has_simd_for_f64() {
            assert!(f64::has_simd());
        }

        #[test]
        fn test_has_simd_for_u8() {
            assert!(u8::has_simd());
        }

        #[test]
        fn test_simd_add_f32() {
            let a = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
            let b = vec![10.0f32, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0];
            let mut dst = vec![0.0f32; 8];
            f32::simd_add(&mut dst, &a, &b);
            assert_eq!(dst, vec![11.0, 22.0, 33.0, 44.0, 55.0, 66.0, 77.0, 88.0]);
        }

        #[test]
        fn test_simd_sub_f32() {
            let a = vec![10.0f32, 20.0, 30.0, 40.0];
            let b = vec![1.0f32, 2.0, 3.0, 4.0];
            let mut dst = vec![0.0f32; 4];
            f32::simd_sub(&mut dst, &a, &b);
            assert_eq!(dst, vec![9.0, 18.0, 27.0, 36.0]);
        }

        #[test]
        fn test_simd_mul_f32() {
            let a = vec![2.0f32, 3.0, 4.0, 5.0];
            let b = vec![10.0f32, 10.0, 10.0, 10.0];
            let mut dst = vec![0.0f32; 4];
            f32::simd_mul(&mut dst, &a, &b);
            assert_eq!(dst, vec![20.0, 30.0, 40.0, 50.0]);
        }

        #[test]
        fn test_simd_div_f32() {
            let a = vec![10.0f32, 20.0, 30.0, 0.0];
            let b = vec![2.0f32, 5.0, 10.0, 0.0];
            let mut dst = vec![0.0f32; 4];
            f32::simd_div(&mut dst, &a, &b);
            assert_eq!(dst, vec![5.0, 4.0, 3.0, 0.0]);
        }

        #[test]
        fn test_simd_dot_f32() {
            let a = vec![1.0f32, 2.0, 3.0, 4.0];
            let b = vec![4.0f32, 3.0, 2.0, 1.0];
            let result = f32::simd_dot(&a, &b).unwrap();
            assert!((result - 20.0).abs() < 1e-6);
        }

        #[test]
        fn test_simd_magnitude_f32() {
            let x = vec![3.0f32, 0.0, 5.0];
            let y = vec![4.0f32, 3.0, 12.0];
            let mut dst = vec![0.0f32; 3];
            f32::simd_magnitude(&mut dst, &x, &y);
            assert!((dst[0] - 5.0).abs() < 1e-5);
            assert!((dst[1] - 3.0).abs() < 1e-5);
            assert!((dst[2] - 13.0).abs() < 1e-5);
        }

        #[test]
        fn test_simd_add_weighted_f32() {
            let a = vec![1.0f32, 2.0, 3.0, 4.0];
            let b = vec![10.0f32, 20.0, 30.0, 40.0];
            let mut dst = vec![0.0f32; 4];
            f32::simd_add_weighted(&mut dst, &a, &b, 0.5, 0.5, 1.0);
            assert!((dst[0] - 6.5).abs() < 1e-5);
            assert!((dst[1] - 12.0).abs() < 1e-5);
            assert!((dst[2] - 17.5).abs() < 1e-5);
            assert!((dst[3] - 23.0).abs() < 1e-5);
        }

        #[test]
        fn test_simd_add_f64() {
            let a = vec![1.0f64, 2.0, 3.0, 4.0];
            let b = vec![10.0f64, 20.0, 30.0, 40.0];
            let mut dst = vec![0.0f64; 4];
            f64::simd_add(&mut dst, &a, &b);
            assert_eq!(dst, vec![11.0, 22.0, 33.0, 44.0]);
        }

        #[test]
        fn test_simd_dot_f64() {
            let a = vec![1.0f64, 2.0, 3.0];
            let b = vec![4.0f64, 5.0, 6.0];
            let result = f64::simd_dot(&a, &b).unwrap();
            assert!((result - 32.0).abs() < 1e-12);
        }

        #[test]
        fn test_simd_add_u8_saturating() {
            let a = vec![200u8, 100, 50, 0];
            let b = vec![100u8, 200, 50, 0];
            let mut dst = vec![0u8; 4];
            u8::simd_add(&mut dst, &a, &b);
            assert_eq!(dst, vec![255, 255, 100, 0]);
        }

        #[test]
        fn test_simd_threshold_binary_u8() {
            let src = vec![10u8, 100, 128, 200, 255, 50];
            let mut dst = vec![0u8; 6];
            assert!(u8::simd_threshold(&mut dst, &src, 127.0, 255.0, 0));
            assert_eq!(dst, vec![0, 0, 255, 255, 255, 0]);
        }

        #[test]
        fn test_simd_threshold_binary_inv_u8() {
            let src = vec![10u8, 100, 128, 200, 255, 50];
            let mut dst = vec![0u8; 6];
            assert!(u8::simd_threshold(&mut dst, &src, 127.0, 255.0, 1));
            assert_eq!(dst, vec![255, 255, 0, 0, 0, 255]);
        }

        #[test]
        fn test_simd_threshold_trunc_u8() {
            let src = vec![10u8, 100, 128, 200, 255, 50];
            let mut dst = vec![0u8; 6];
            assert!(u8::simd_threshold(&mut dst, &src, 127.0, 255.0, 2));
            assert_eq!(dst, vec![10, 100, 127, 127, 127, 50]);
        }

        #[test]
        fn test_simd_threshold_tozero_u8() {
            let src = vec![10u8, 100, 128, 200, 255, 50];
            let mut dst = vec![0u8; 6];
            assert!(u8::simd_threshold(&mut dst, &src, 127.0, 255.0, 3));
            assert_eq!(dst, vec![0, 0, 128, 200, 255, 0]);
        }

        #[test]
        fn test_simd_threshold_tozero_inv_u8() {
            let src = vec![10u8, 100, 128, 200, 255, 50];
            let mut dst = vec![0u8; 6];
            assert!(u8::simd_threshold(&mut dst, &src, 127.0, 255.0, 4));
            assert_eq!(dst, vec![10, 100, 0, 0, 0, 50]);
        }

        #[test]
        fn test_simd_threshold_binary_f32() {
            let src = vec![0.1f32, 0.4, 0.5, 0.6, 0.9, 0.3];
            let mut dst = vec![0.0f32; 6];
            assert!(f32::simd_threshold(&mut dst, &src, 0.5, 1.0, 0));
            assert_eq!(dst, vec![0.0, 0.0, 0.0, 1.0, 1.0, 0.0]);
        }

        #[test]
        fn test_simd_threshold_trunc_f32() {
            let src = vec![0.1f32, 0.4, 0.6, 0.9];
            let mut dst = vec![0.0f32; 4];
            assert!(f32::simd_threshold(&mut dst, &src, 0.5, 1.0, 2));
            assert_eq!(dst, vec![0.1, 0.4, 0.5, 0.5]);
        }

        #[test]
        fn test_simd_threshold_invalid_type() {
            let src = vec![10u8; 4];
            let mut dst = vec![0u8; 4];
            assert!(!u8::simd_threshold(&mut dst, &src, 127.0, 255.0, 5));
        }

        #[test]
        fn test_simd_convert_scale_abs_f32() {
            let src = vec![1.0f32, -2.0, 3.0, -4.0];
            let mut dst = vec![0u8; 4];
            f32::simd_convert_scale_abs(&mut dst, &src, 10.0, 0.0);
            assert_eq!(dst, vec![10, 20, 30, 40]);
        }

        // ---------------------------------------------------------------
        //  SIMD vs scalar equivalence tests
        // ---------------------------------------------------------------

        /// Helper: compute element-wise add in pure scalar Rust
        fn scalar_add_f32(a: &[f32], b: &[f32]) -> Vec<f32> {
            a.iter().zip(b.iter()).map(|(&x, &y)| x + y).collect()
        }

        /// Helper: compute element-wise sub in pure scalar Rust
        fn scalar_sub_f32(a: &[f32], b: &[f32]) -> Vec<f32> {
            a.iter().zip(b.iter()).map(|(&x, &y)| x - y).collect()
        }

        /// Helper: compute element-wise mul in pure scalar Rust
        fn scalar_mul_f32(a: &[f32], b: &[f32]) -> Vec<f32> {
            a.iter().zip(b.iter()).map(|(&x, &y)| x * y).collect()
        }

        /// Helper: compute element-wise div in pure scalar Rust (0/0 = 0)
        fn scalar_div_f32(a: &[f32], b: &[f32]) -> Vec<f32> {
            a.iter()
                .zip(b.iter())
                .map(|(&x, &y)| if y != 0.0 { x / y } else { 0.0 })
                .collect()
        }

        /// Helper: compute dot product in pure scalar Rust
        fn scalar_dot_f32(a: &[f32], b: &[f32]) -> f64 {
            a.iter()
                .zip(b.iter())
                .map(|(&x, &y)| x as f64 * y as f64)
                .sum()
        }

        /// Helper: compute sum in pure scalar Rust
        fn scalar_sum_f32(src: &[f32]) -> f64 {
            src.iter().map(|&v| v as f64).sum()
        }

        /// Generate a non-trivial f32 test vector
        fn make_test_vec_f32(len: usize) -> Vec<f32> {
            (0..len).map(|i| (i as f32 * 0.7) - 50.0).collect()
        }

        #[test]
        fn test_simd_vs_scalar_add_f32() {
            let a = make_test_vec_f32(1024);
            let b: Vec<f32> = (0..1024).map(|i| (i as f32) * 1.3 + 2.0).collect();
            let expected = scalar_add_f32(&a, &b);
            let mut simd_result = vec![0.0f32; 1024];
            assert!(f32::simd_add(&mut simd_result, &a, &b));
            for (i, (&e, &s)) in expected.iter().zip(simd_result.iter()).enumerate() {
                assert!(
                    (e - s).abs() < 1e-5,
                    "Mismatch at index {i}: expected {e}, got {s}"
                );
            }
        }

        #[test]
        fn test_simd_vs_scalar_sub_f32() {
            let a = make_test_vec_f32(1024);
            let b: Vec<f32> = (0..1024).map(|i| (i as f32) * 0.3).collect();
            let expected = scalar_sub_f32(&a, &b);
            let mut simd_result = vec![0.0f32; 1024];
            assert!(f32::simd_sub(&mut simd_result, &a, &b));
            for (i, (&e, &s)) in expected.iter().zip(simd_result.iter()).enumerate() {
                assert!(
                    (e - s).abs() < 1e-5,
                    "Mismatch at index {i}: expected {e}, got {s}"
                );
            }
        }

        #[test]
        fn test_simd_vs_scalar_mul_f32() {
            let a = make_test_vec_f32(512);
            let b: Vec<f32> = (0..512).map(|i| (i as f32) * 0.01 + 0.5).collect();
            let expected = scalar_mul_f32(&a, &b);
            let mut simd_result = vec![0.0f32; 512];
            assert!(f32::simd_mul(&mut simd_result, &a, &b));
            for (i, (&e, &s)) in expected.iter().zip(simd_result.iter()).enumerate() {
                assert!(
                    (e - s).abs() < 1e-3,
                    "Mismatch at index {i}: expected {e}, got {s}"
                );
            }
        }

        #[test]
        fn test_simd_vs_scalar_div_f32() {
            let a = make_test_vec_f32(256);
            let mut b: Vec<f32> = (0..256).map(|i| (i as f32) * 0.5 + 1.0).collect();
            b[100] = 0.0; // test division by zero
            let expected = scalar_div_f32(&a, &b);
            let mut simd_result = vec![0.0f32; 256];
            assert!(f32::simd_div(&mut simd_result, &a, &b));
            for (i, (&e, &s)) in expected.iter().zip(simd_result.iter()).enumerate() {
                assert!(
                    (e - s).abs() < 1e-3,
                    "Mismatch at index {i}: expected {e}, got {s}"
                );
            }
        }

        #[test]
        fn test_simd_vs_scalar_dot_f32() {
            let a = make_test_vec_f32(2048);
            let b: Vec<f32> = (0..2048).map(|i| (i as f32) * 0.3 - 100.0).collect();
            let expected = scalar_dot_f32(&a, &b);
            let simd_result = f32::simd_dot(&a, &b).unwrap();
            assert!(
                (expected - simd_result).abs() < 1e-2,
                "Dot mismatch: expected {expected}, got {simd_result}"
            );
        }

        #[test]
        fn test_simd_vs_scalar_sum_f32() {
            let src = make_test_vec_f32(4096);
            let expected = scalar_sum_f32(&src);
            let simd_result = f32::simd_sum(&src).unwrap();
            assert!(
                (expected - simd_result).abs() < 1e-1,
                "Sum mismatch: expected {expected}, got {simd_result}"
            );
        }

        #[test]
        fn test_simd_vs_scalar_sqrt_f32() {
            let src: Vec<f32> = (0..512).map(|i| (i as f32) * 2.0 + 1.0).collect();
            let expected: Vec<f32> = src.iter().map(|&v| v.sqrt()).collect();
            let mut simd_result = vec![0.0f32; 512];
            assert!(f32::simd_sqrt(&mut simd_result, &src));
            for (i, (&e, &s)) in expected.iter().zip(simd_result.iter()).enumerate() {
                assert!(
                    (e - s).abs() < 1e-5,
                    "Sqrt mismatch at index {i}: expected {e}, got {s}"
                );
            }
        }

        #[test]
        fn test_simd_vs_scalar_min_max_f32() {
            let a = make_test_vec_f32(256);
            let b: Vec<f32> = (0..256).map(|i| (i as f32) * 0.5 - 30.0).collect();

            // min
            let expected_min: Vec<f32> = a
                .iter()
                .zip(b.iter())
                .map(|(&x, &y)| if x < y { x } else { y })
                .collect();
            let mut simd_min_result = vec![0.0f32; 256];
            assert!(f32::simd_min(&mut simd_min_result, &a, &b));
            assert_eq!(simd_min_result, expected_min);

            // max
            let expected_max: Vec<f32> = a
                .iter()
                .zip(b.iter())
                .map(|(&x, &y)| if x > y { x } else { y })
                .collect();
            let mut simd_max_result = vec![0.0f32; 256];
            assert!(f32::simd_max(&mut simd_max_result, &a, &b));
            assert_eq!(simd_max_result, expected_max);
        }

        #[test]
        fn test_simd_vs_scalar_magnitude_f32() {
            let x: Vec<f32> = (0..128).map(|i| (i as f32) * 0.5).collect();
            let y: Vec<f32> = (0..128).map(|i| (i as f32) * 0.3 + 1.0).collect();
            let expected: Vec<f32> = x
                .iter()
                .zip(y.iter())
                .map(|(&xv, &yv)| (xv * xv + yv * yv).sqrt())
                .collect();
            let mut simd_result = vec![0.0f32; 128];
            assert!(f32::simd_magnitude(&mut simd_result, &x, &y));
            for (i, (&e, &s)) in expected.iter().zip(simd_result.iter()).enumerate() {
                assert!(
                    (e - s).abs() < 1e-4,
                    "Magnitude mismatch at index {i}: expected {e}, got {s}"
                );
            }
        }

        #[test]
        fn test_simd_vs_scalar_add_weighted_f32() {
            let a = make_test_vec_f32(256);
            let b: Vec<f32> = (0..256).map(|i| (i as f32) * 0.2 + 3.0).collect();
            let alpha = 0.6f64;
            let beta = 0.4f64;
            let gamma = 2.5f64;
            let expected: Vec<f32> = a
                .iter()
                .zip(b.iter())
                .map(|(&av, &bv)| (av as f64 * alpha + bv as f64 * beta + gamma) as f32)
                .collect();
            let mut simd_result = vec![0.0f32; 256];
            assert!(f32::simd_add_weighted(
                &mut simd_result,
                &a,
                &b,
                alpha,
                beta,
                gamma,
            ));
            for (i, (&e, &s)) in expected.iter().zip(simd_result.iter()).enumerate() {
                assert!(
                    (e - s).abs() < 1e-3,
                    "AddWeighted mismatch at index {i}: expected {e}, got {s}"
                );
            }
        }

        #[test]
        fn test_simd_vs_scalar_convert_scale_abs_f32() {
            let src: Vec<f32> = (0..256).map(|i| (i as f32) * 0.5 - 64.0).collect();
            let alpha = 2.0f64;
            let beta = 10.0f64;
            let expected: Vec<u8> = src
                .iter()
                .map(|&v| {
                    let val = ((v as f64) * alpha + beta).abs();
                    val.clamp(0.0, 255.0).round() as u8
                })
                .collect();
            let mut simd_result = vec![0u8; 256];
            assert!(f32::simd_convert_scale_abs(
                &mut simd_result,
                &src,
                alpha,
                beta,
            ));
            assert_eq!(simd_result, expected);
        }

        // --- f64 equivalence tests ---

        #[test]
        fn test_simd_vs_scalar_add_f64() {
            let a: Vec<f64> = (0..512).map(|i| (i as f64) * 0.7 - 100.0).collect();
            let b: Vec<f64> = (0..512).map(|i| (i as f64) * 1.3 + 50.0).collect();
            let expected: Vec<f64> = a.iter().zip(b.iter()).map(|(&x, &y)| x + y).collect();
            let mut simd_result = vec![0.0f64; 512];
            assert!(f64::simd_add(&mut simd_result, &a, &b));
            assert_eq!(simd_result, expected);
        }

        #[test]
        fn test_simd_vs_scalar_dot_f64() {
            let a: Vec<f64> = (0..1024).map(|i| (i as f64) * 0.3 - 100.0).collect();
            let b: Vec<f64> = (0..1024).map(|i| (i as f64) * 0.7 + 20.0).collect();
            let expected: f64 = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();
            let simd_result = f64::simd_dot(&a, &b).unwrap();
            assert!(
                (expected - simd_result).abs() < 1e-6,
                "f64 dot mismatch: expected {expected}, got {simd_result}"
            );
        }

        #[test]
        fn test_simd_vs_scalar_sum_f64() {
            let src: Vec<f64> = (0..2048).map(|i| (i as f64) * 0.3 - 300.0).collect();
            let expected: f64 = src.iter().sum();
            let simd_result = f64::simd_sum(&src).unwrap();
            assert!(
                (expected - simd_result).abs() < 1e-6,
                "f64 sum mismatch: expected {expected}, got {simd_result}"
            );
        }

        // --- u8 equivalence tests ---

        #[test]
        fn test_simd_vs_scalar_add_u8_equiv() {
            let a: Vec<u8> = (0..256).map(|i| i as u8).collect();
            let b: Vec<u8> = (0..256).map(|i| (255 - i) as u8).collect();
            let expected: Vec<u8> = a
                .iter()
                .zip(b.iter())
                .map(|(&x, &y)| x.saturating_add(y))
                .collect();
            let mut simd_result = vec![0u8; 256];
            assert!(u8::simd_add(&mut simd_result, &a, &b));
            assert_eq!(simd_result, expected);
        }

        #[test]
        fn test_simd_vs_scalar_sub_u8_equiv() {
            let a: Vec<u8> = (0..256).map(|i| i as u8).collect();
            let b: Vec<u8> = (0..256).map(|i| (i / 2) as u8).collect();
            let expected: Vec<u8> = a
                .iter()
                .zip(b.iter())
                .map(|(&x, &y)| x.saturating_sub(y))
                .collect();
            let mut simd_result = vec![0u8; 256];
            assert!(u8::simd_sub(&mut simd_result, &a, &b));
            assert_eq!(simd_result, expected);
        }

        #[test]
        fn test_simd_vs_scalar_sum_u8_equiv() {
            let src: Vec<u8> = (0..1024).map(|i| (i % 256) as u8).collect();
            let expected: f64 = src.iter().map(|&v| v as f64).sum();
            let simd_result = u8::simd_sum(&src).unwrap();
            assert!(
                (expected - simd_result).abs() < 1e-10,
                "u8 sum mismatch: expected {expected}, got {simd_result}"
            );
        }

        // --- absdiff tests ---

        #[test]
        fn test_simd_absdiff_f32() {
            let a = vec![10.0f32, 3.0, 5.0, 0.0];
            let b = vec![3.0f32, 10.0, 5.0, 7.0];
            let mut dst = vec![0.0f32; 4];
            assert!(f32::simd_absdiff(&mut dst, &a, &b));
            assert_eq!(dst, vec![7.0, 7.0, 0.0, 7.0]);
        }

        #[test]
        fn test_simd_absdiff_f64() {
            let a = vec![10.0f64, 3.0, 5.0, 0.0];
            let b = vec![3.0f64, 10.0, 5.0, 7.0];
            let mut dst = vec![0.0f64; 4];
            assert!(f64::simd_absdiff(&mut dst, &a, &b));
            assert_eq!(dst, vec![7.0, 7.0, 0.0, 7.0]);
        }

        #[test]
        fn test_simd_absdiff_u8() {
            let a = vec![200u8, 50, 100, 0];
            let b = vec![100u8, 200, 100, 255];
            let mut dst = vec![0u8; 4];
            assert!(u8::simd_absdiff(&mut dst, &a, &b));
            assert_eq!(dst, vec![100, 150, 0, 255]);
        }

        #[test]
        fn test_simd_vs_scalar_absdiff_f32() {
            let a: Vec<f32> = (0..1024).map(|i| (i as f32) * 0.7 - 50.0).collect();
            let b: Vec<f32> = (0..1024).map(|i| (i as f32) * 0.3 + 20.0).collect();
            let expected: Vec<f32> = a
                .iter()
                .zip(b.iter())
                .map(|(&x, &y)| (x - y).abs())
                .collect();
            let mut simd_result = vec![0.0f32; 1024];
            assert!(f32::simd_absdiff(&mut simd_result, &a, &b));
            for (i, (&e, &s)) in expected.iter().zip(simd_result.iter()).enumerate() {
                assert!(
                    (e - s).abs() < 1e-5,
                    "Absdiff mismatch at index {i}: expected {e}, got {s}"
                );
            }
        }

        // --- bitwise ops tests (u8) ---

        #[test]
        fn test_simd_bitwise_and_u8() {
            let a = vec![0xFFu8, 0xAA, 0x0F, 0x00];
            let b = vec![0x0Fu8, 0x55, 0x0F, 0xFF];
            let mut dst = vec![0u8; 4];
            assert!(u8::simd_bitwise_and(&mut dst, &a, &b));
            assert_eq!(dst, vec![0x0F, 0x00, 0x0F, 0x00]);
        }

        #[test]
        fn test_simd_bitwise_or_u8() {
            let a = vec![0xF0u8, 0xAA, 0x0F, 0x00];
            let b = vec![0x0Fu8, 0x55, 0x0F, 0xFF];
            let mut dst = vec![0u8; 4];
            assert!(u8::simd_bitwise_or(&mut dst, &a, &b));
            assert_eq!(dst, vec![0xFF, 0xFF, 0x0F, 0xFF]);
        }

        #[test]
        fn test_simd_bitwise_xor_u8() {
            let a = vec![0xFFu8, 0xAA, 0x0F, 0x00];
            let b = vec![0x0Fu8, 0xAA, 0xF0, 0xFF];
            let mut dst = vec![0u8; 4];
            assert!(u8::simd_bitwise_xor(&mut dst, &a, &b));
            assert_eq!(dst, vec![0xF0, 0x00, 0xFF, 0xFF]);
        }

        #[test]
        fn test_simd_bitwise_not_u8() {
            let src = vec![0x00u8, 0xFF, 0xAA, 0x55];
            let mut dst = vec![0u8; 4];
            assert!(u8::simd_bitwise_not(&mut dst, &src));
            assert_eq!(dst, vec![0xFF, 0x00, 0x55, 0xAA]);
        }

        #[test]
        fn test_simd_vs_scalar_bitwise_and_u8() {
            let a: Vec<u8> = (0..256).map(|i| i as u8).collect();
            let b: Vec<u8> = (0..256).map(|i| (255 - i) as u8).collect();
            let expected: Vec<u8> = a.iter().zip(b.iter()).map(|(&x, &y)| x & y).collect();
            let mut simd_result = vec![0u8; 256];
            assert!(u8::simd_bitwise_and(&mut simd_result, &a, &b));
            assert_eq!(simd_result, expected);
        }

        // --- norm_l2_sq tests ---

        #[test]
        fn test_simd_norm_l2_sq_f32() {
            let src = vec![3.0f32, 4.0];
            let result = f32::simd_norm_l2_sq(&src).unwrap();
            assert!((result - 25.0).abs() < 1e-6);
        }

        #[test]
        fn test_simd_norm_l2_sq_f64() {
            let src = vec![3.0f64, 4.0];
            let result = f64::simd_norm_l2_sq(&src).unwrap();
            assert!((result - 25.0).abs() < 1e-12);
        }

        #[test]
        fn test_simd_norm_l2_sq_u8() {
            let src = vec![3u8, 4];
            let result = u8::simd_norm_l2_sq(&src).unwrap();
            assert!((result - 25.0).abs() < 1e-12);
        }

        #[test]
        fn test_simd_vs_scalar_norm_l2_sq_f32() {
            let src: Vec<f32> = (0..1024).map(|i| (i as f32) * 0.3 - 50.0).collect();
            let expected: f64 = src.iter().map(|&v| (v as f64) * (v as f64)).sum();
            let simd_result = f32::simd_norm_l2_sq(&src).unwrap();
            assert!(
                (expected - simd_result).abs() / expected.abs().max(1.0) < 1e-4,
                "norm_l2_sq mismatch: expected {expected}, got {simd_result}"
            );
        }
    }

    #[cfg(not(feature = "simd"))]
    mod no_simd_tests {
        use super::super::*;

        #[test]
        fn test_no_simd_f32() {
            assert!(!f32::has_simd());
        }

        #[test]
        fn test_no_simd_f64() {
            assert!(!f64::has_simd());
        }

        #[test]
        fn test_no_simd_u8() {
            assert!(!u8::has_simd());
        }
    }
}

/*
 *  optical_flow.rs
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

//! Pyramidal Lucas-Kanade optical flow.
//!
//! This module provides:
//!
//! - [`build_optical_flow_pyramid`] — construct a Gaussian image pyramid
//!   (and optionally its spatial derivatives) from a grayscale frame.
//! - [`calc_optical_flow_pyramid_lk`] — track feature points from one frame
//!   to the next using the pyramidal Lucas-Kanade method.
//! - [`calc_optical_flow_pyramid_lk_with_pyramids`] — reuse borrowed image
//!   pyramids and optional Scharr derivatives across tracking calls.
//!
//! # Divergences from OpenCV
//!
//! | OpenCV | purecv |
//! |--------|--------|
//! | `buildOpticalFlowPyramid` operates on `Mat` (any depth) | accepts `Matrix<u8>` (8-bit grayscale) |
//! | Returns level count as `int` | returns `OpticalFlowPyramid` struct |
//! | `calcOpticalFlowPyrLK` `nextPts` is `InputOutputArray` | initial guess passed via `initial_next_pts: Option<&[Point2f]>` |
//! | `tryReuseInputImage` optimisation flag | not implemented (correctness only) |

use alloc::{borrow::Cow, format, string::ToString, vec::Vec};
#[allow(unused_imports)]
use num_traits::Float;

use crate::core::error::{PureCvError, Result};
use crate::core::logging::tags;
use crate::core::types::{BorderTypes, Point2f, Size2i, TermCriteria, TermType};
use crate::core::utils::border_interpolate;
use crate::core::Matrix;
use crate::cv_log_debug;
use crate::imgproc::derivatives::scharr;
use crate::imgproc::pyramid::pyr_down;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

#[cfg(feature = "simd")]
use super::simd as video_simd;

/// Fixed-point normalisation factor applied to the spatial gradient matrix `H`
/// before the minimum-eigenvalue test, matching OpenCV's `FLT_SCALE`
/// ([`lkpyramid.cpp` L235](https://github.com/opencv/opencv/blob/4.10.0/modules/video/src/lkpyramid.cpp#L235)).
///
/// OpenCV accumulates `H` from *unnormalised* Scharr derivatives and rescales
/// by `2^-20` so that `min_eigen_threshold` is a small, resolution-independent
/// number (its default is `1e-4`). Applying the same factor here keeps
/// thresholds transferable between the two libraries.
pub(crate) const FLT_SCALE: f64 = 1.0 / (1u32 << 20) as f64;

/// Degeneracy threshold for the *unscaled* determinant of `H`, equivalent to
/// OpenCV's `D < FLT_EPSILON` test on its `FLT_SCALE`-scaled determinant
/// ([`lkpyramid.cpp` L473](https://github.com/opencv/opencv/blob/4.10.0/modules/video/src/lkpyramid.cpp#L473),
/// [L480](https://github.com/opencv/opencv/blob/4.10.0/modules/video/src/lkpyramid.cpp#L480), see #142):
/// `FLT_EPSILON / FLT_SCALE^2 = 2^-23 / 2^-40 = 2^17 = 131072`.
#[cfg(test)]
pub(crate) const LK_DET_EPSILON: f64 = f32::EPSILON as f64 / (FLT_SCALE * FLT_SCALE);

// ---------------------------------------------------------------------------
// Public flags (mirror OpenCV's OpticalFlowFlags)
// ---------------------------------------------------------------------------

/// Use the initial estimates in `initial_next_pts` as the starting point for
/// the pyramid-level flow rather than starting from zero.
///
/// Mirrors `cv::OPTFLOW_USE_INITIAL_FLOW`.
pub const OPTFLOW_USE_INITIAL_FLOW: i32 = 4;

/// Store the minimum eigenvalue of each tracked point's spatial-gradient
/// matrix in the `err` output instead of the mean-absolute-error between
/// the matched windows.
///
/// Mirrors `cv::OPTFLOW_LK_GET_MIN_EIGENVALS`.
pub const OPTFLOW_LK_GET_MIN_EIGENVALS: i32 = 8;

// ---------------------------------------------------------------------------
// Public struct
// ---------------------------------------------------------------------------

/// Gaussian image pyramid with optional spatial derivatives, as produced by
/// [`build_optical_flow_pyramid`].
///
/// `levels[0]` is the original (full-resolution) image converted to `f32`.
/// Each subsequent level is the result of [`pyr_down`] applied to the
/// previous one.  The `dx` and `dy` fields are populated only when
/// `with_derivatives = true`. Pass this struct by reference to
/// [`calc_optical_flow_pyramid_lk_with_pyramids`] to reuse its buffers.
/// Levels contain unpadded integer grayscale intensities in `0..=255` stored
/// as f32; derivatives are integer i16 Scharr values (gain 32), also stored
/// as f32 with the same dimensions. These ranges match OpenCV's CV_8U image
/// levels and CV_16S derivatives; malformed numeric values are rejected.
/// Both derivative vectors must be empty, or each must cover every level.
///
/// The border fields record how OpenCV padded the levels at build time, so LK
/// can reproduce that padding on the fly instead of storing it:
/// `pyr_border` mirrors `pyrBorder` (padding of image levels), `deriv_border`
/// mirrors `derivBorder` (padding of derivative levels).
#[derive(Debug, Clone)]
pub struct OpticalFlowPyramid {
    /// Pyramid levels from finest (index 0) to coarsest (index n).
    pub levels: Vec<Matrix<f32>>,
    /// Scharr-x derivative for each level (empty when not requested).
    pub dx: Vec<Matrix<f32>>,
    /// Scharr-y derivative for each level (empty when not requested).
    pub dy: Vec<Matrix<f32>>,
    /// Border that extends image levels outside their stored extent when LK
    /// samples them, mirroring OpenCV's `pyrBorder`. OpenCV's default is
    /// `Reflect101`.
    pub pyr_border: BorderTypes,
    /// Border that extends derivative levels outside their stored extent when
    /// LK samples them, mirroring OpenCV's `derivBorder`. Ignored when
    /// `dx`/`dy` are empty: LK then computes derivatives on the fly and extends
    /// them with `Constant` (zero), exactly like OpenCV's raw-image path.
    pub deriv_border: BorderTypes,
}

// ---------------------------------------------------------------------------
// build_optical_flow_pyramid
// ---------------------------------------------------------------------------

/// Constructs a Gaussian image pyramid suitable for Lucas-Kanade optical flow.
///
/// Mirrors [`cv::buildOpticalFlowPyramid`](https://docs.opencv.org/4.10.0/dc/d6b/group__video__track.html#ga86640c1c470f87b2660c096d2b22b2ce).
///
/// Level 0 of the returned pyramid is the source image converted to `f32`.
/// Each subsequent level is obtained by applying [`pyr_down`] to the previous
/// one. As in OpenCV
/// ([`buildOpticalFlowPyramid` L811-L816](https://github.com/opencv/opencv/blob/4.10.0/modules/video/src/lkpyramid.cpp#L811-L816)),
/// level `L + 1` is created only while its size `((w + 1) / 2, (h + 1) / 2)` is
/// strictly larger than `win_size` in both dimensions, so `win_size` caps the
/// usable depth. [`calc_optical_flow_pyramid_lk`] builds through the same rule,
/// so raw-image and prebuilt pyramids hold the same levels.
///
/// The border arguments mirror OpenCV's `pyrBorder` and `derivBorder`:
/// `pyr_border` extends image levels outside their stored extent, while
/// downsampling always uses `Reflect101`, matching OpenCV's generic CPU
/// `pyrDown` call independently of the padding
/// ([`lkpyramid.cpp` L786-L791](https://github.com/opencv/opencv/blob/4.10.0/modules/video/src/lkpyramid.cpp#L786-L791)).
/// `deriv_border` extends derivative levels. The Scharr
/// convolution itself always uses `Reflect101`, matching `calcScharrDeriv`
/// ([`lkpyramid.cpp` L82-L128](https://github.com/opencv/opencv/blob/4.10.0/modules/video/src/lkpyramid.cpp#L82-L128)),
/// which handles its own edges independently of `derivBorder`.
///
/// # Arguments
/// * `img`              — Single-channel 8-bit grayscale input image.
/// * `win_size`         — Tracking window size; both sides must be greater
///   than 2 (OpenCV's `CV_Assert(winSize.width > 2 && winSize.height > 2)`)
///   and cap the pyramid depth.
/// * `max_level`        — Maximum number of additional pyramid levels to build
///   on top of the original (level 0).  The returned pyramid has at most
///   `max_level + 1` levels.
/// * `with_derivatives` — When `true`, the Scharr-x and Scharr-y derivatives
///   are computed for every level and stored in the returned struct
///   (matching `cv::buildOpticalFlowPyramid`).
/// * `pyr_border`       — OpenCV's `pyrBorder`: how image levels are extended
///   outside their extent.
/// * `deriv_border`     — OpenCV's `derivBorder`: how the stored derivative
///   levels are extended outside their extent.
///
/// # Errors
/// Returns an error for an empty or malformed image, a non-grayscale image, or
/// a window size below 3x3.
///
/// # Example
/// ```
/// use purecv::core::Matrix;
/// use purecv::core::types::{BorderTypes, Size2i};
/// use purecv::video::optical_flow::build_optical_flow_pyramid;
///
/// let img = Matrix::<u8>::new(64, 64, 1);
/// let pyr = build_optical_flow_pyramid(
///     &img,
///     Size2i::new(21, 21),
///     3,
///     false,
///     BorderTypes::Reflect101,
///     BorderTypes::Constant,
/// ).unwrap();
/// assert_eq!(pyr.levels[0].rows, 64);
/// assert_eq!(pyr.levels[1].rows, 32);
/// ```
pub fn build_optical_flow_pyramid(
    img: &Matrix<u8>,
    win_size: Size2i,
    max_level: usize,
    with_derivatives: bool,
    pyr_border: BorderTypes,
    deriv_border: BorderTypes,
) -> Result<OpticalFlowPyramid> {
    validate_lk_matrix(img)?;
    if win_size.width < 3 || win_size.height < 3 {
        return Err(PureCvError::InvalidInput(
            "build_optical_flow_pyramid: win_size.width and win_size.height must be greater than 2"
                .to_string(),
        ));
    }
    let levels = build_f32_pyramid(img.convert_to::<f32>()?, max_level, win_size)?;
    let (dx, dy) = if with_derivatives {
        compute_derivatives(&levels)?
    } else {
        (Vec::new(), Vec::new())
    };

    Ok(OpticalFlowPyramid {
        levels,
        dx,
        dy,
        pyr_border,
        deriv_border,
    })
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Check public Matrix metadata before any indexing, allocation or conversion.
fn validate_lk_matrix<T>(image: &Matrix<T>) -> Result<()> {
    if image.channels != 1 {
        return Err(PureCvError::InvalidInput(
            "LK images and derivatives must be single-channel".to_string(),
        ));
    }
    if image.rows == 0
        || image.cols == 0
        || image.rows > i32::MAX as usize
        || image.cols > i32::MAX as usize
        || image.rows.checked_mul(image.cols) != Some(image.data.len())
    {
        return Err(PureCvError::InvalidDimensions(
            "LK matrices must be nonempty, fit i32 dimensions, and have rows * cols elements"
                .to_string(),
        ));
    }
    Ok(())
}

/// Move the converted image into level zero and downsample while the next level
/// would still be strictly larger than `win_size` in both dimensions.
///
/// This is the level rule of `buildOpticalFlowPyramid`
/// ([`lkpyramid.cpp` L811-L816](https://github.com/opencv/opencv/blob/4.10.0/modules/video/src/lkpyramid.cpp#L811-L816)):
/// after building level `L`, it forms `(sz.width + 1) / 2, (sz.height + 1) / 2`
/// and returns `L` when either value is `<= winSize`. The raw-image LK path
/// therefore uses the same cutoff and produces the same levels.
fn build_f32_pyramid(
    img: Matrix<f32>,
    max_level: usize,
    win_size: Size2i,
) -> Result<Vec<Matrix<f32>>> {
    let mut levels = Vec::new();
    levels.push(img);
    for i in 0..max_level {
        let prev = &levels[i];
        let next_rows = prev.rows.div_ceil(2);
        let next_cols = prev.cols.div_ceil(2);
        if next_rows <= win_size.height as usize || next_cols <= win_size.width as usize {
            break;
        }
        // OpenCV's generic pyrDown maps against the level's own dimensions
        // with its default Reflect101 border; pyrBorder only controls padding.
        let next = pyr_down(prev, None, BorderTypes::Reflect101)?;
        levels.push(next);
    }
    Ok(levels)
}

type OwnedDerivatives = (Vec<Matrix<f32>>, Vec<Matrix<f32>>);
type BorrowedDerivatives<'a> = (Cow<'a, [Matrix<f32>]>, Cow<'a, [Matrix<f32>]>);

/// Compute the Scharr derivatives of every level with `Reflect101` edges,
/// matching OpenCV's `calcScharrDeriv`, which performs its own reflect-101
/// border handling independently of `derivBorder`.
fn compute_derivatives(levels: &[Matrix<f32>]) -> Result<OwnedDerivatives> {
    #[cfg(feature = "parallel")]
    let iter = levels.par_iter();
    #[cfg(not(feature = "parallel"))]
    let iter = levels.iter();
    let pairs: Result<Vec<_>> = iter
        .map(|level| {
            Ok((
                scharr(level, 1, 0, 1.0, 0.0, BorderTypes::Reflect101)?,
                scharr(level, 0, 1, 1.0, 0.0, BorderTypes::Reflect101)?,
            ))
        })
        .collect();
    Ok(pairs?.into_iter().unzip())
}

/// Called only after validating the complete pyramid and selected level count.
pub(super) fn lk_derivatives(
    pyramid: &OpticalFlowPyramid,
    count: usize,
) -> Result<BorrowedDerivatives<'_>> {
    if pyramid.dx.is_empty() {
        let (dx, dy) = compute_derivatives(&pyramid.levels[..count])?;
        Ok((Cow::Owned(dx), Cow::Owned(dy)))
    } else {
        Ok((
            Cow::Borrowed(&pyramid.dx[..count]),
            Cow::Borrowed(&pyramid.dy[..count]),
        ))
    }
}

/// LK can only extend samples with a real border rule; `Transparent` and
/// `Isolated` leave OpenCV's padding undefined, so they are rejected.
fn validate_lk_border(border: BorderTypes) -> Result<()> {
    if matches!(border, BorderTypes::Transparent | BorderTypes::Isolated) {
        return Err(PureCvError::InvalidInput(
            "LK pyramid borders must be Constant, Replicate, Reflect, Reflect101 or Wrap"
                .to_string(),
        ));
    }
    Ok(())
}

fn validate_lk_pyramid(pyramid: &OpticalFlowPyramid) -> Result<()> {
    if pyramid.levels.is_empty() {
        return Err(PureCvError::InvalidInput(
            "LK pyramid must contain level zero".to_string(),
        ));
    }
    validate_lk_border(pyramid.pyr_border)?;
    validate_lk_border(pyramid.deriv_border)?;
    let has_derivatives = !pyramid.dx.is_empty() || !pyramid.dy.is_empty();
    if has_derivatives
        && (pyramid.dx.len() != pyramid.levels.len() || pyramid.dy.len() != pyramid.levels.len())
    {
        return Err(PureCvError::InvalidInput(
            "LK pyramid dx and dy must both be empty or cover every level".to_string(),
        ));
    }
    for (i, level) in pyramid.levels.iter().enumerate() {
        validate_lk_matrix(level)?;
        if level.data.iter().any(|&v| v != f32::from(v as u8)) {
            return Err(PureCvError::InvalidInput(
                "LK image levels must contain integer values in 0..=255".to_string(),
            ));
        }
        if i > 0 {
            let previous = &pyramid.levels[i - 1];
            if previous.rows < 2
                || previous.cols < 2
                || level.rows != previous.rows.div_ceil(2)
                || level.cols != previous.cols.div_ceil(2)
            {
                return Err(PureCvError::InvalidDimensions(
                    "LK pyramid levels must halve both dimensions (rounding up), stopping below two"
                        .to_string(),
                ));
            }
        }
        if has_derivatives {
            for derivative in [&pyramid.dx[i], &pyramid.dy[i]] {
                validate_lk_matrix(derivative)?;
                if derivative.data.iter().any(|&v| v != f32::from(v as i16)) {
                    return Err(PureCvError::InvalidInput(
                        "LK derivatives must contain integer values in the i16 range".to_string(),
                    ));
                }
                if derivative.rows != level.rows || derivative.cols != level.cols {
                    return Err(PureCvError::InvalidDimensions(
                        "LK derivatives must match their image level dimensions".to_string(),
                    ));
                }
            }
        }
    }
    Ok(())
}

/// Sampling pattern of an LK tracking window, matching OpenCV's
/// `halfWin((winSize.width-1)*0.5f, (winSize.height-1)*0.5f)`
/// (`modules/video/src/lkpyramid.cpp`, see #144): exactly
/// `width x height` samples at offsets `k - half`, `k = 0..n`, relative to
/// the tracked point. For an even size the offsets are half-integers and
/// bilinear interpolation supplies the samples.
#[derive(Clone, Copy)]
struct TrackingWindow {
    width: i32,
    height: i32,
    half_w: f32,
    half_h: f32,
}

impl TrackingWindow {
    /// `win_size` must already be validated (both sides >= 3).
    fn new(win_size: Size2i) -> Self {
        Self {
            width: win_size.width,
            height: win_size.height,
            half_w: (win_size.width - 1) as f32 * 0.5,
            half_h: (win_size.height - 1) as f32 * 0.5,
        }
    }

    /// Number of samples, `width * height`.
    fn area(self) -> usize {
        (self.width * self.height) as usize
    }
}

/// Border handling for LK window sampling, mirroring OpenCV's
/// `calcOpticalFlowPyrLK` (defaults in [`tracking.hpp:121-125`](https://github.com/opencv/opencv/blob/4.10.0/modules/video/include/opencv2/video/tracking.hpp#L121-L125),
/// padding in [`lkpyramid.cpp:726-807`](https://github.com/opencv/opencv/blob/4.10.0/modules/video/src/lkpyramid.cpp#L726-L807)): image
/// levels are padded by the window size with `BORDER_REFLECT_101`, derivative
/// levels with `BORDER_CONSTANT` (zero).
///
/// Sampling the bare levels with clamping instead replicates edge pixels
/// across the overhang region — most of the window at coarse levels — and
/// fabricates phantom edges whose large spurious Scharr responses mis-steer
/// the Gauss-Newton solver (see #163). Image samples therefore use
/// reflect-101 extension; derivative samples read zero outside the true
/// level bounds. Both are then interpolated with fixed-point weights.
///
/// `BORDER_REFLECT_101` index mapping (`gfedcb|abcdefgh|gfedcba`): the edge
/// pixel is not duplicated. The input is reduced modulo the reflection
/// period first, so absurd coordinates terminate in O(1) instead of
/// spinning the loop ~|i|/n times.
pub(crate) fn reflect101_index(i: i32, n: i32) -> usize {
    if n <= 1 {
        return 0;
    }
    let period = 2 * i64::from(n) - 2;
    let mut v = (i as i64).rem_euclid(period);
    // v in [0, 2n-3]: fold the second half back.
    if v >= n as i64 {
        v = period - v;
    }
    v as usize
}

/// Fixed-point bilinear patch origin and weights. OpenCV computes these once
/// at the window's top-left, not independently at each pixel (which changes
/// floating-point rounding at large coordinates).
/// https://github.com/opencv/opencv/blob/4.10.0/modules/video/src/lkpyramid.cpp#L215-L239
pub(super) struct LkPatch {
    x: i32,
    y: i32,
    weights: [i32; 4],
}

impl LkPatch {
    pub(super) fn new(x: f32, y: f32) -> Self {
        let fx = x.floor();
        let fy = y.floor();
        let a = x - fx;
        let b = y - fy;
        // cvRound uses nearest-even under the default rounding mode. These
        // nonnegative weights are bounded by 2^14; integer parity handles ties
        // identically in std and no_std without a platform-dependent cast.
        let round_weight = |v: f32| {
            let lo = v.floor() as i32;
            let fraction = v - lo as f32;
            lo + i32::from(fraction > 0.5 || (fraction == 0.5 && lo & 1 != 0))
        };
        let w00 = round_weight((1.0 - a) * (1.0 - b) * 16384.0);
        let w01 = round_weight(a * (1.0 - b) * 16384.0);
        let w10 = round_weight((1.0 - a) * b * 16384.0);
        Self {
            x: fx as i32,
            y: fy as i32,
            weights: [w00, w01, w10, 16384 - w00 - w01 - w10],
        }
    }

    /// CV_DESCALE: add half before an arithmetic right shift, including for
    /// negative derivatives. Images use shift 9 (intensity x32), derivatives
    /// shift 14 (integer Scharr), exactly as the source's IWinBuf/derivIWinBuf.
    /// https://github.com/opencv/opencv/blob/4.10.0/modules/video/src/lkpyramid.cpp#L438-L449
    pub(super) fn sample(
        &self,
        image: &Matrix<f32>,
        x: i32,
        y: i32,
        border: BorderTypes,
        shift: u32,
    ) -> f32 {
        let tap = |x: i32, y: i32| -> i64 {
            let map = |i, n| {
                if border == BorderTypes::Reflect101 {
                    reflect101_index(i, n) as i32
                } else {
                    border_interpolate(i, n, border)
                }
            };
            let x = map(x, image.cols as i32);
            let y = map(y, image.rows as i32);
            if x < 0 || y < 0 {
                0
            } else {
                image.data[y as usize * image.cols + x as usize] as i64
            }
        };
        let x = self.x.saturating_add(x);
        let y = self.y.saturating_add(y);
        let x1 = x.saturating_add(1);
        let y1 = y.saturating_add(1);
        let [w00, w01, w10, w11] = self.weights.map(i64::from);
        // Widen intermediates so caller-supplied i16 derivatives cannot
        // overflow even at the negative limit. Valid OpenCV samples agree.
        ((tap(x, y) * w00
            + tap(x1, y) * w01
            + tap(x, y1) * w10
            + tap(x1, y1) * w11
            + (1i64 << (shift - 1)))
            >> shift) as f32
    }
}

/// Solve one level using OpenCV's quantized patches and f32 arithmetic.
/// The initial and returned coordinates are absolute target positions, not
/// displacements: subtracting and re-adding the template changes rounding.
#[allow(clippy::too_many_arguments)]
fn lk_single_level(
    prev: &Matrix<f32>,
    next: &Matrix<f32>,
    prev_ix: &Matrix<f32>,
    prev_iy: &Matrix<f32>,
    px: f32,
    py: f32,
    init_x: f32,
    init_y: f32,
    win: TrackingWindow,
    prev_border: BorderTypes,
    next_border: BorderTypes,
    deriv_border: BorderTypes,
    max_iters: i32,
    eps: f64,
    min_eigen_threshold: f64,
    is_finest: bool,
) -> (f32, f32, f32, bool) {
    let tx = (px - win.half_w).floor();
    let ty = (py - win.half_h).floor();
    if !px.is_finite()
        || !py.is_finite()
        || !init_x.is_finite()
        || !init_y.is_finite()
        || tx < -(win.width as f32)
        || tx >= prev.cols as f32
        || ty < -(win.height as f32)
        || ty >= prev.rows as f32
    {
        return (init_x, init_y, 1.0, !is_finest);
    }
    let patch = LkPatch::new(px - win.half_w, py - win.half_h);
    let mut ix_win = Vec::with_capacity(win.area());
    let mut iy_win = Vec::with_capacity(win.area());
    let mut i1_win = Vec::with_capacity(win.area());
    for y in 0..win.height {
        for x in 0..win.width {
            ix_win.push(patch.sample(prev_ix, x, y, deriv_border, 14));
            iy_win.push(patch.sample(prev_iy, x, y, deriv_border, 14));
            i1_win.push(patch.sample(prev, x, y, prev_border, 9));
        }
    }
    #[cfg(feature = "simd")]
    let (h00, h01, h11) = video_simd::simd_lk_accumulate_h(&ix_win, &iy_win);
    #[cfg(not(feature = "simd"))]
    let (h00, h01, h11) = {
        let (mut h00, mut h01, mut h11) = (0.0f32, 0.0f32, 0.0f32);
        for (&ix, &iy) in ix_win.iter().zip(&iy_win) {
            h00 += ix * ix;
            h01 += ix * iy;
            h11 += iy * iy;
        }
        (h00, h01, h11)
    };
    // Preserve the operation order and precision of the generic float CPU
    // path, including the determinant and threshold cast at the invoker.
    // https://github.com/opencv/opencv/blob/4.10.0/modules/video/src/lkpyramid.cpp#L469-L492
    let h00 = h00 * FLT_SCALE as f32;
    let h01 = h01 * FLT_SCALE as f32;
    let h11 = h11 * FLT_SCALE as f32;
    let det = h00 * h11 - h01 * h01;
    let min_eigen = (h11 + h00 - ((h00 - h11) * (h00 - h11) + 4.0 * h01 * h01).sqrt())
        / (2.0 * win.width as f32 * win.height as f32);
    if min_eigen < min_eigen_threshold as f32 || det < f32::EPSILON {
        cv_log_debug!(
            tags::VIDEO,
            "LK degenerate window: min_eigen={}, determinant={}",
            min_eigen,
            det
        );
        return (init_x, init_y, min_eigen, false);
    }
    // OpenCV accumulates H from ixval/iyval, then uses the stored short
    // derivative buffer for mismatch. Residual weight iw11 can be -1:
    // extreme caller-supplied i16 gradients can interpolate outside i16.
    // Preserve the narrowing conversion, after H but before refinement.
    // https://github.com/opencv/opencv/blob/4.10.0/modules/video/src/lkpyramid.cpp#L438-L453
    for value in ix_win.iter_mut().chain(iy_win.iter_mut()) {
        *value = f32::from(*value as i32 as i16);
    }
    let bounds = LkBounds {
        half_w: win.half_w,
        half_h: win.half_h,
        min_x: -(win.width as f32),
        max_x: next.cols as f32,
        min_y: -(win.height as f32),
        max_y: next.rows as f32,
        is_finest,
    };
    #[cfg(feature = "simd")]
    let mut i2_win = alloc::vec![0.0f32; win.area()];
    let (x, y, tracked) = lk_iterate(
        h00,
        h01,
        h11,
        1.0 / det,
        init_x,
        init_y,
        max_iters,
        eps,
        Some(bounds),
        |x, y| {
            let patch = LkPatch::new(x, y);
            #[cfg(feature = "simd")]
            let (bx, by) = {
                for (i, value) in i2_win.iter_mut().enumerate() {
                    *value = patch.sample(
                        next,
                        i as i32 % win.width,
                        i as i32 / win.width,
                        next_border,
                        9,
                    );
                }
                video_simd::simd_lk_accumulate_mismatch(&ix_win, &iy_win, &i1_win, &i2_win)
            };
            #[cfg(not(feature = "simd"))]
            let (bx, by) = {
                let (mut bx, mut by) = (0.0f32, 0.0f32);
                for i in 0..win.area() {
                    let i2 = patch.sample(
                        next,
                        i as i32 % win.width,
                        i as i32 / win.width,
                        next_border,
                        9,
                    );
                    let diff = i2 - i1_win[i];
                    bx -= diff * ix_win[i];
                    by -= diff * iy_win[i];
                }
                (bx, by)
            };
            (bx * FLT_SCALE as f32, by * FLT_SCALE as f32)
        },
    );
    (x, y, min_eigen, tracked)
}

/// Search-window bounds and centre offset for one pyramid level.
#[derive(Clone, Copy)]
pub(crate) struct LkBounds {
    pub half_w: f32,
    pub half_h: f32,
    pub min_x: f32,
    pub max_x: f32,
    pub min_y: f32,
    pub max_y: f32,
    pub is_finest: bool,
}

/// OpenCV updates the window corner in f32, writes corner + halfWin to the
/// output, and tests the f64 dot product of its f32 step. Oscillation corrects
/// the output centre, rather than the internal corner (different rounding).
/// The callback returns the negative, scaled mismatch at the window corner.
/// https://github.com/opencv/opencv/blob/4.10.0/modules/video/src/lkpyramid.cpp#L494-L681
#[allow(clippy::too_many_arguments)]
pub(crate) fn lk_iterate(
    h00: f32,
    h01: f32,
    h11: f32,
    inv_det: f32,
    init_x: f32,
    init_y: f32,
    max_iters: i32,
    eps: f64,
    bounds: Option<LkBounds>,
    mut compute_mismatch: impl FnMut(f32, f32) -> (f32, f32),
) -> (f32, f32, bool) {
    let (half_w, half_h) = bounds.map_or((0.0, 0.0), |b| (b.half_w, b.half_h));
    let (mut x, mut y) = (init_x - half_w, init_y - half_h);
    let (mut out_x, mut out_y) = (init_x, init_y);
    let (mut prev_dx, mut prev_dy) = (0.0f32, 0.0f32);
    let mut tracked = true;
    for iter in 0..max_iters {
        if let Some(b) = bounds {
            let nx = x.floor();
            let ny = y.floor();
            if !nx.is_finite()
                || !ny.is_finite()
                || nx < b.min_x
                || nx >= b.max_x
                || ny < b.min_y
                || ny >= b.max_y
            {
                tracked = !b.is_finest;
                break;
            }
        }
        let (bx, by) = compute_mismatch(x, y);
        let dx = (h11 * bx - h01 * by) * inv_det;
        let dy = (h00 * by - h01 * bx) * inv_det;
        x += dx;
        y += dy;
        out_x = x + half_w;
        out_y = y + half_h;
        if f64::from(dx) * f64::from(dx) + f64::from(dy) * f64::from(dy) <= eps {
            break;
        }
        if iter > 0
            && f64::from((dx + prev_dx).abs()) < 0.01
            && f64::from((dy + prev_dy).abs()) < 0.01
        {
            out_x -= dx * 0.5;
            out_y -= dy * 0.5;
            break;
        }
        prev_dx = dx;
        prev_dy = dy;
    }
    (out_x, out_y, tracked)
}

/// Compute the mean-absolute error (MAE) between matching `win_size`
/// windows in the previous and next frame around `prev_pt` → `next_pt`.
fn compute_tracking_error(
    prev: &Matrix<f32>,
    next: &Matrix<f32>,
    prev_pt: Point2f,
    next_pt: Point2f,
    win: TrackingWindow,
    prev_border: BorderTypes,
    next_border: BorderTypes,
) -> f32 {
    let prev_patch = LkPatch::new(prev_pt.x - win.half_w, prev_pt.y - win.half_h);
    let next_patch = LkPatch::new(next_pt.x - win.half_w, next_pt.y - win.half_h);
    let mut error = 0.0f32;
    for y in 0..win.height {
        for x in 0..win.width {
            let i1 = prev_patch.sample(prev, x, y, prev_border, 9);
            let i2 = next_patch.sample(next, x, y, next_border, 9);
            error += (i2 - i1).abs();
        }
    }
    error / (32.0 * win.area() as f32)
}

/// Normalize exactly once, before tracking any levels.
/// https://github.com/opencv/opencv/blob/4.10.0/modules/video/src/lkpyramid.cpp#L1364-L1372
pub(super) fn normalize_lk_criteria(criteria: TermCriteria) -> (i32, f64) {
    let count = match criteria.type_ {
        TermType::Eps => 30,
        TermType::Count | TermType::Both => criteria.max_count.clamp(0, 100),
    };
    let epsilon = match criteria.type_ {
        TermType::Count => 0.01,
        // Comparisons preserve NaN like std::min(std::max(epsilon, 0.), 10.).
        TermType::Eps | TermType::Both => criteria.epsilon.clamp(0.0, 10.0),
    };
    (count, epsilon * epsilon)
}

// ---------------------------------------------------------------------------
// calc_optical_flow_pyramid_lk
// ---------------------------------------------------------------------------

/// Calculates the optical flow for a sparse feature set using the iterative
/// Lucas-Kanade method with pyramids.
///
/// Mirrors [`cv::calcOpticalFlowPyrLK`](https://docs.opencv.org/4.10.0/dc/d6b/group__video__track.html#ga473e4b886d0bcc6b65831eb88ed93323).
///
/// For each point in `prev_pts` the function tracks it from `prev_img` to
/// `next_img` through an image pyramid and iterative gradient descent.  The
/// algorithm is the standard pyramidal LK formulation: at each pyramid level
/// (from coarsest to finest) it solves the 2×2 linear system
///
/// ```text
/// H · v = b
/// ```
///
/// where `H = Σ [[Ix², Ix·Iy],[Ix·Iy, Iy²]]` is the spatial-gradient matrix
/// and `b = -Σ [Ix·It, Iy·It]` accumulates the mismatch across the tracking
/// window.  The flow estimate is propagated (×2) from each coarse level to
/// the next finer one.
///
/// To reuse preprocessing between calls, see
/// [`calc_optical_flow_pyramid_lk_with_pyramids`] and its preprocessing contract.
///
/// # Arguments
/// * `prev_img`            — Previous single-channel 8-bit grayscale frame.
/// * `next_img`            — Next single-channel 8-bit grayscale frame.  Must
///   be the same size as `prev_img`.
/// * `prev_pts`            — Feature points to track, in `prev_img` coordinates.
///   Non-finite coordinates and template windows outside the padded image
///   bounds are marked lost (`status = 0`).
/// * `initial_next_pts`    — Optional initial guess for `nextPts`.  Passed
///   together with [`OPTFLOW_USE_INITIAL_FLOW`]; ignored otherwise.
///   Non-finite initial estimates are marked lost.
/// * `win_size`            — Size of the search window at each pyramid level;
///   both sides must be at least 3. As in OpenCV, the window holds exactly
///   `width x height` samples centred on the point, at offsets
///   `k - (n - 1) / 2` for `k = 0..n`, so an even size samples at
///   half-integer offsets via bilinear interpolation.
/// * `max_level`           — Pyramid depth (0 = no pyramid, just the original).
/// * `criteria`            — Iteration termination criteria. Counts are clamped
///   to `0..=100` (default 30 without Count); epsilon is clamped to `0..=10`
///   (default 0.01 without Eps), then squared. Stop when the squared step is
///   less than or equal to that threshold, matching OpenCV 4.10.0.
/// * `flags`               — Option flags; combine [`OPTFLOW_USE_INITIAL_FLOW`]
///   and/or [`OPTFLOW_LK_GET_MIN_EIGENVALS`].
/// * `min_eigen_threshold` — Points whose spatial-gradient matrix has a
///   minimum eigenvalue below this threshold are marked as lost. The
///   gradient matrix is built from Scharr derivatives and normalised by
///   OpenCV's `FLT_SCALE = 2^-20` and the window area, matching
///   `cv::calcOpticalFlowPyrLK` (see #130 and #138), so this value is on the
///   same scale as OpenCV's and its `1e-4` default transfers directly. A
///   point can also be marked lost independently of this threshold when
///   its gradient matrix is near-singular (the determinant guard,
///   matching OpenCV's `D < FLT_EPSILON`). As in OpenCV, both guards only
///   mark a point lost at the finest pyramid level (level 0); at a coarser
///   level they skip refinement for that level and the propagated flow
///   estimate carries on to the next finer one.
///   Thresholds tuned against a purecv build predating both fixes must be
///   retuned: Scharr multiplies the gradient matrix by 16 relative to the
///   Sobel derivatives used then, and `FLT_SCALE` divides by `2^20`, so such
///   thresholds read about `2^20 / 16 = 65536` times larger on this scale.
///
/// # Returns
/// A tuple `(next_pts, status, err)`:
/// * `next_pts`  — Estimated positions of the tracked features in `next_img`.
/// * `status`    — Per-feature tracking flag: `1` = tracked, `0` = lost.
/// * `err`       — Per-feature tracking error (MAE by default; minimum
///   eigenvalue when [`OPTFLOW_LK_GET_MIN_EIGENVALS`] is set).
///   Lost points retain PureCV's sentinel convention: their position is the
///   input point and their error is `f32::MAX`; only `status` is comparable
///   to OpenCV for these points (OpenCV does not define failed-point MAE).
///
/// # Errors
/// Returns [`PureCvError::InvalidInput`] if either input image is not
/// single-channel or `win_size` is smaller than 3x3, and
/// [`PureCvError::InvalidDimensions`] if the images' dimensions differ.
///
/// # Example
/// ```
/// use purecv::core::Matrix;
/// use purecv::core::types::{Size2i, TermCriteria, TermType, Point2f};
/// use purecv::video::optical_flow::calc_optical_flow_pyramid_lk;
///
/// // Build a 64×64 frame with a bright 8×8 square at (28..36, 28..36)
/// // so the tracker has real gradients to work with.
/// let mut data = vec![0u8; 64 * 64];
/// for r in 28..36 { for c in 28..36 { data[r * 64 + c] = 200; } }
/// let frame = Matrix::<u8>::from_vec(64, 64, 1, data);
///
/// let pts = vec![Point2f::new(32.0, 32.0)];
/// let criteria = TermCriteria::new(TermType::Both, 20, 0.03);
///
/// let (next_pts, status, _err) = calc_optical_flow_pyramid_lk(
///     &frame, &frame, &pts, None,
///     Size2i::new(11, 11), 2, criteria, 0, 1e-4,
/// ).unwrap();
///
/// assert_eq!(status[0], 1);
/// assert!((next_pts[0].x - 32.0).abs() < 1.0);
/// ```
#[allow(clippy::too_many_arguments)]
pub fn calc_optical_flow_pyramid_lk(
    prev_img: &Matrix<u8>,
    next_img: &Matrix<u8>,
    prev_pts: &[Point2f],
    initial_next_pts: Option<&[Point2f]>,
    win_size: Size2i,
    max_level: i32,
    criteria: TermCriteria,
    flags: i32,
    min_eigen_threshold: f64,
) -> Result<(Vec<Point2f>, Vec<u8>, Vec<f32>)> {
    // ------------------------------------------------------------------
    // Input validation
    // ------------------------------------------------------------------
    if prev_img.channels != 1 || next_img.channels != 1 {
        return Err(PureCvError::InvalidInput(
            "calc_optical_flow_pyramid_lk: input images must be single-channel grayscale"
                .to_string(),
        ));
    }
    if prev_img.rows != next_img.rows || prev_img.cols != next_img.cols {
        return Err(PureCvError::InvalidDimensions(
            "calc_optical_flow_pyramid_lk: prev_img and next_img must have the same dimensions"
                .to_string(),
        ));
    }
    // Mirrors OpenCV's `CV_Assert(winSize.width > 2 && winSize.height > 2)`.
    if win_size.width < 3 || win_size.height < 3 {
        return Err(PureCvError::InvalidInput(format!(
            "calc_optical_flow_pyramid_lk: win_size must be at least 3x3, got {}x{}",
            win_size.width, win_size.height
        )));
    }

    validate_lk_matrix(prev_img)?;
    validate_lk_matrix(next_img)?;
    if prev_pts.is_empty() {
        return Ok((Vec::new(), Vec::new(), Vec::new()));
    }
    // Mirrors OpenCV's raw-image path: `buildOpticalFlowPyramid(img, pyr,
    // winSize, maxLevel, false)` with its default `pyrBorder = BORDER_REFLECT_101`,
    // then per-level `calcScharrDeriv` padded with `BORDER_CONSTANT`
    // ([`lkpyramid.cpp` L1358-L1362](https://github.com/opencv/opencv/blob/4.10.0/modules/video/src/lkpyramid.cpp#L1358-L1362),
    // [L1374-L1392](https://github.com/opencv/opencv/blob/4.10.0/modules/video/src/lkpyramid.cpp#L1374-L1392)).
    let build = |image: &Matrix<u8>| -> Result<OpticalFlowPyramid> {
        Ok(OpticalFlowPyramid {
            levels: build_f32_pyramid(
                image.convert_to::<f32>()?,
                max_level.max(0) as usize,
                win_size,
            )?,
            dx: Vec::new(),
            dy: Vec::new(),
            pyr_border: BorderTypes::Reflect101,
            deriv_border: BorderTypes::Constant,
        })
    };
    // Both pyramids are independent; build them concurrently when possible.
    #[cfg(feature = "parallel")]
    let (prev_pyramid, next_pyramid) = {
        let (prev, next) = rayon::join(|| build(prev_img), || build(next_img));
        (prev?, next?)
    };
    #[cfg(not(feature = "parallel"))]
    let (prev_pyramid, next_pyramid) = (build(prev_img)?, build(next_img)?);
    calc_optical_flow_pyramid_lk_with_pyramids(
        &prev_pyramid,
        &next_pyramid,
        prev_pts,
        initial_next_pts,
        win_size,
        max_level,
        criteria,
        flags,
        min_eigen_threshold,
    )
}

/// Tracks features using borrowed, precomputed pyramids and optional derivatives.
///
/// Arguments and outputs follow [`calc_optical_flow_pyramid_lk`]. The two input
/// pyramids replace its raw images. No supplied image or derivative buffer is
/// copied or rebuilt; only missing previous-image derivatives are computed.
/// Next-image derivatives are validated but unused, so a frame's pyramid can
/// be reused as either input in temporal and stereo tracking.
///
/// # Preprocessing contract
/// - Each pyramid must have a nonempty grayscale level zero. Following levels
///   have dimensions `(previous + 1) / 2`; [`build_optical_flow_pyramid`]
///   stops once the next level is not strictly larger than its window, matching
///   OpenCV's cutoff. Buffers must contain exactly `rows * cols` elements.
/// - `dx` and `dy` must both be empty or contain one matching matrix per level.
///   Supplied derivatives are used verbatim and should be unnormalised Scharr
///   derivatives (gain 32). Missing derivatives are computed with a
///   `Reflect101` convolution.
/// - Tracking uses levels through `min(max(max_level, 0), prev_last, next_last)`.
///   Corresponding used levels must match dimensions. Unequal depths are valid;
///   all supplied levels and derivative layouts are validated even if unused.
/// - No explicit padded matrices are needed: image samples outside a level are
///   extended with the pyramid's [`OpticalFlowPyramid::pyr_border`]
///   (OpenCV's `pyrBorder`), supplied derivative samples with its
///   [`OpticalFlowPyramid::deriv_border`] (OpenCV's `derivBorder`). Derivatives
///   computed here use `Constant` (zero) extension, matching OpenCV's raw path.
///
/// For agreement with the raw-image API, build with the **same window** as the
/// tracking call and pass the raw API's default borders (`Reflect101` /
/// `Constant`). Both paths then apply OpenCV's cutoff and identical
/// preprocessing. Regression tests require matching status, points within
/// `1e-5` pixels and defined errors within `1e-5 * max(1, |error|)` in each
/// feature configuration. Custom preprocessing borders or derivatives can
/// intentionally produce different tracking results.
///
/// # Errors
/// Returns [`PureCvError::InvalidInput`] for invalid channel counts, derivative
/// counts, non-integer/out-of-range image or derivative values, empty pyramids,
/// windows smaller than 3x3 or mismatched initial flow;
/// [`PureCvError::InvalidDimensions`] for malformed buffers or level dimensions.
///
/// # Example
/// ```
/// use purecv::core::{Matrix, BorderTypes, Size2i, Point2f, TermCriteria, TermType};
/// use purecv::video::{build_optical_flow_pyramid, calc_optical_flow_pyramid_lk_with_pyramids};
/// let image = Matrix::<u8>::new(64, 64, 1);
/// let pyramid = build_optical_flow_pyramid(
///     &image, Size2i::new(11, 11), 2, true,
///     BorderTypes::Reflect101, BorderTypes::Reflect101,
/// ).unwrap();
/// let result = calc_optical_flow_pyramid_lk_with_pyramids(
///     &pyramid, &pyramid, &[Point2f::new(32.0, 32.0)], None,
///     Size2i::new(11, 11), 2, TermCriteria::new(TermType::Both, 20, 0.03), 0, 1e-4,
/// ).unwrap();
/// assert_eq!(result.1, vec![0]); // Flat images have no trackable gradients.
/// ```
#[allow(
    clippy::too_many_arguments,
    reason = "Mirrors OpenCV LK parameters and the existing raw-image API"
)]
pub fn calc_optical_flow_pyramid_lk_with_pyramids(
    prev_pyramid: &OpticalFlowPyramid,
    next_pyramid: &OpticalFlowPyramid,
    prev_pts: &[Point2f],
    initial_next_pts: Option<&[Point2f]>,
    win_size: Size2i,
    max_level: i32,
    criteria: TermCriteria,
    flags: i32,
    min_eigen_threshold: f64,
) -> Result<(Vec<Point2f>, Vec<u8>, Vec<f32>)> {
    validate_lk_pyramid(prev_pyramid)?;
    validate_lk_pyramid(next_pyramid)?;
    if win_size.width < 3 || win_size.height < 3 {
        return Err(PureCvError::InvalidInput(
            "LK window must be at least 3x3".to_string(),
        ));
    }
    let actual_levels = (max_level.max(0) as usize + 1)
        .min(prev_pyramid.levels.len())
        .min(next_pyramid.levels.len());
    let prev_pyr = &prev_pyramid.levels[..actual_levels];
    let next_pyr = &next_pyramid.levels[..actual_levels];
    for (prev, next) in prev_pyr.iter().zip(next_pyr) {
        if prev.rows != next.rows || prev.cols != next.cols {
            return Err(PureCvError::InvalidDimensions(
                "LK previous and next levels must have matching dimensions".to_string(),
            ));
        }
    }
    let n_pts = prev_pts.len();
    if n_pts == 0 {
        return Ok((Vec::new(), Vec::new(), Vec::new()));
    }

    // ------------------------------------------------------------------
    // Extract termination criteria
    // ------------------------------------------------------------------
    let (max_iters, eps) = normalize_lk_criteria(criteria);

    let use_initial_flow = (flags & OPTFLOW_USE_INITIAL_FLOW) != 0;
    let get_min_eigenvals = (flags & OPTFLOW_LK_GET_MIN_EIGENVALS) != 0;

    let win = TrackingWindow::new(win_size);
    // Derivative extension border: supplied derivatives keep the pyramid's
    // `derivBorder`; derivatives computed here on the fly use `Constant`,
    // matching OpenCV's `lvlStep == 1` branch. Images always use `pyrBorder`.
    let prev_border = prev_pyramid.pyr_border;
    let next_border = next_pyramid.pyr_border;
    let deriv_border = if prev_pyramid.dx.is_empty() {
        BorderTypes::Constant
    } else {
        prev_pyramid.deriv_border
    };

    // ------------------------------------------------------------------
    // Output buffers
    // ------------------------------------------------------------------

    // Resolve the initial guess for each point.  In the parallel path we need
    // this as a plain Vec so each worker can read it without borrowing issues.
    if use_initial_flow {
        if let Some(init) = initial_next_pts {
            if init.len() != n_pts {
                return Err(PureCvError::InvalidInput(
                    "initial_next_pts length must match prev_pts length".to_string(),
                ));
            }
        }
    }

    let (prev_ix, prev_iy) = lk_derivatives(prev_pyramid, actual_levels)?;

    let initial_guesses: Vec<Point2f> = if use_initial_flow {
        match initial_next_pts {
            Some(init) => init.to_vec(),
            None => prev_pts.to_vec(),
        }
    } else {
        prev_pts.to_vec()
    };

    // ------------------------------------------------------------------
    // Track each feature point through the pyramid.
    //
    // When the `parallel` feature is enabled the per-point LK solve runs
    // concurrently via Rayon — each point is fully independent.  The scalar
    // fallback is structurally identical but uses a sequential `for` loop.
    // ------------------------------------------------------------------
    let coarsest = actual_levels - 1;
    let coarsest_scale = 1.0f32 / (1u32 << coarsest) as f32;

    // Helper closure that tracks one point through all pyramid levels.
    // Captured (read-only) references are shared safely in the parallel path.
    let track_point = |i: usize| -> (Point2f, u8, f32) {
        let prev_pt = prev_pts[i];
        let guess = initial_guesses[i];

        let mut next_x = guess.x * coarsest_scale;
        let mut next_y = guess.y * coarsest_scale;
        let mut min_eigen = 1.0f32;
        let mut tracked = true;

        for level in (0..actual_levels).rev() {
            let scale = 1.0f32 / (1u32 << level) as f32;
            let px = prev_pt.x * scale;
            let py = prev_pt.y * scale;

            if level < coarsest {
                next_x *= 2.0;
                next_y *= 2.0;
            }

            let (fu, fv, ev, success) = lk_single_level(
                &prev_pyr[level],
                &next_pyr[level],
                &prev_ix[level],
                &prev_iy[level],
                px,
                py,
                next_x,
                next_y,
                win,
                prev_border,
                next_border,
                deriv_border,
                max_iters,
                eps,
                min_eigen_threshold,
                level == 0,
            );

            next_x = fu;
            next_y = fv;
            min_eigen = ev;

            // Like OpenCV (`lkpyramid.cpp`, the `minEig < minEigThreshold ||
            // D < FLT_EPSILON` branch), a degenerate window only loses the
            // point at the finest level. At a coarser level it just skips
            // refinement there: `lk_single_level` hands back the propagated
            // target position unchanged, which carries on to the next level (#145).
            if !success && level == 0 {
                tracked = false;
            }
        }

        if tracked {
            let next_pt = Point2f::new(next_x, next_y);
            let e = if get_min_eigenvals {
                min_eigen
            } else {
                // Recheck the final window before MAE: zero iterations or the
                // last Newton step can leave an estimate outside the padding.
                // OpenCV applies this gate only in the MAE branch
                // (https://github.com/opencv/opencv/blob/4.10.0/modules/video/src/lkpyramid.cpp#L683-L698).
                let tx = (next_pt.x - win.half_w).floor();
                let ty = (next_pt.y - win.half_h).floor();
                if !tx.is_finite()
                    || !ty.is_finite()
                    || tx < -(win.width as f32)
                    || tx >= next_pyr[0].cols as f32
                    || ty < -(win.height as f32)
                    || ty >= next_pyr[0].rows as f32
                {
                    return (prev_pt, 0u8, f32::MAX);
                }
                compute_tracking_error(
                    &prev_pyr[0],
                    &next_pyr[0],
                    prev_pt,
                    next_pt,
                    win,
                    prev_border,
                    next_border,
                )
            };
            (next_pt, 1u8, e)
        } else {
            (prev_pt, 0u8, f32::MAX)
        }
    };

    #[cfg(feature = "parallel")]
    let results: Vec<(Point2f, u8, f32)> = (0..n_pts).into_par_iter().map(track_point).collect();

    #[cfg(not(feature = "parallel"))]
    let results: Vec<(Point2f, u8, f32)> = (0..n_pts).map(track_point).collect();

    let mut next_pts: Vec<Point2f> = Vec::with_capacity(n_pts);
    let mut status: Vec<u8> = Vec::with_capacity(n_pts);
    let mut err: Vec<f32> = Vec::with_capacity(n_pts);
    for (pt, s, e) in results {
        next_pts.push(pt);
        status.push(s);
        err.push(e);
    }

    Ok((next_pts, status, err))
}

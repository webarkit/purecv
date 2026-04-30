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
//!
//! # Divergences from OpenCV
//!
//! | OpenCV | purecv |
//! |--------|--------|
//! | `buildOpticalFlowPyramid` operates on `Mat` (any depth) | accepts `Matrix<u8>` (8-bit grayscale) |
//! | Returns level count as `int` | returns `OpticalFlowPyramid` struct |
//! | `calcOpticalFlowPyrLK` `nextPts` is `InputOutputArray` | initial guess passed via `initial_next_pts: Option<&[Point2f]>` |
//! | `tryReuseInputImage` optimisation flag | not implemented (correctness only) |

use crate::core::error::{PureCvError, Result};
use crate::core::types::{BorderTypes, Point2f, Size2i, TermCriteria, TermType};
use crate::core::Matrix;
use crate::imgproc::derivatives::sobel;
use crate::imgproc::pyramid::pyr_down;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

#[cfg(feature = "simd")]
use super::simd as video_simd;

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
/// `with_derivatives = true`.
#[derive(Debug, Clone)]
pub struct OpticalFlowPyramid {
    /// Pyramid levels from finest (index 0) to coarsest (index n).
    pub levels: Vec<Matrix<f32>>,
    /// Sobel-x derivative for each level (empty when not requested).
    pub dx: Vec<Matrix<f32>>,
    /// Sobel-y derivative for each level (empty when not requested).
    pub dy: Vec<Matrix<f32>>,
}

// ---------------------------------------------------------------------------
// build_optical_flow_pyramid
// ---------------------------------------------------------------------------

/// Constructs a Gaussian image pyramid suitable for Lucas-Kanade optical flow.
///
/// Mirrors [`cv::buildOpticalFlowPyramid`](https://docs.opencv.org/4.10.0/dc/d6b/group__video__track.html#ga86640c1c470f87b2660c096d2b22b2ce).
///
/// Level 0 of the returned pyramid is the source image converted to `f32`.
/// Each subsequent level is obtained by applying [`pyr_down`] to the
/// previous one; construction stops early if either dimension of the
/// downsampled image would fall below `win_size`.
///
/// # Arguments
/// * `img`              — Single-channel 8-bit grayscale input image.
/// * `win_size`         — Tracking window size (used to determine the minimum
///   usable pyramid level size).
/// * `max_level`        — Maximum number of additional pyramid levels to build
///   on top of the original (level 0).  The returned pyramid has at most
///   `max_level + 1` levels.
/// * `with_derivatives` — When `true`, the Sobel-x and Sobel-y derivatives
///   are computed for every level and stored in the returned struct.
/// * `pyr_border`       — Border interpolation used when downsampling.
/// * `deriv_border`     — Border interpolation used when computing derivatives.
///
/// # Errors
/// Returns [`PureCvError::InvalidInput`] if `img` is not single-channel.
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
    if img.channels != 1 {
        return Err(PureCvError::InvalidInput(
            "build_optical_flow_pyramid requires a single-channel image".to_string(),
        ));
    }

    // Convert to f32 for all subsequent arithmetic.
    let src_f32 = img.convert_to::<f32>()?;

    // Build pyramid levels.
    let mut levels: Vec<Matrix<f32>> = Vec::with_capacity(max_level + 1);
    levels.push(src_f32);

    for i in 0..max_level {
        let prev = &levels[i];
        // Stop early if the current level is already too small for the window.
        if prev.rows < win_size.height as usize || prev.cols < win_size.width as usize {
            break;
        }
        let next = pyr_down(prev, None, pyr_border)?;
        levels.push(next);
    }

    // Optionally compute Sobel derivatives for each level.
    // When the `parallel` feature is enabled the per-level Sobel passes run
    // concurrently via Rayon; otherwise they execute sequentially.
    let (dx, dy) = if with_derivatives {
        #[cfg(feature = "parallel")]
        {
            let pairs: Result<Vec<(Matrix<f32>, Matrix<f32>)>> = levels
                .par_iter()
                .map(|level| {
                    let ix: Matrix<f32> = sobel(level, 1, 0, 3, 1.0, 0.0, deriv_border)?;
                    let iy: Matrix<f32> = sobel(level, 0, 1, 3, 1.0, 0.0, deriv_border)?;
                    Ok((ix, iy))
                })
                .collect();
            let (all_dx, all_dy) = pairs?.into_iter().unzip();
            (all_dx, all_dy)
        }

        #[cfg(not(feature = "parallel"))]
        {
            let n = levels.len();
            let mut all_dx: Vec<Matrix<f32>> = Vec::with_capacity(n);
            let mut all_dy: Vec<Matrix<f32>> = Vec::with_capacity(n);
            for level in &levels {
                let ix: Matrix<f32> = sobel(level, 1, 0, 3, 1.0, 0.0, deriv_border)?;
                let iy: Matrix<f32> = sobel(level, 0, 1, 3, 1.0, 0.0, deriv_border)?;
                all_dx.push(ix);
                all_dy.push(iy);
            }
            (all_dx, all_dy)
        }
    } else {
        (Vec::new(), Vec::new())
    };

    Ok(OpticalFlowPyramid { levels, dx, dy })
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Build an internal f32 Gaussian pyramid (no derivatives, minimum-size guard).
fn build_f32_pyramid(img: &Matrix<f32>, max_level: usize) -> Result<Vec<Matrix<f32>>> {
    let mut levels: Vec<Matrix<f32>> = Vec::with_capacity(max_level + 1);
    levels.push(img.clone());

    for i in 0..max_level {
        let prev = &levels[i];
        // Stop if either dimension would become 0 after downsampling.
        if prev.rows < 2 || prev.cols < 2 {
            break;
        }
        let next = pyr_down(prev, None, BorderTypes::Reflect101)?;
        levels.push(next);
    }

    Ok(levels)
}

/// Bilinear interpolation of a single-channel f32 image at fractional position
/// `(x, y)`.  Out-of-bounds coordinates are clamped to the image border.
#[inline]
fn bilinear_interp(img: &Matrix<f32>, x: f32, y: f32) -> f32 {
    let rows = img.rows;
    let cols = img.cols;

    // Floor + clamp.
    let x0 = (x.floor() as i32).clamp(0, cols as i32 - 1) as usize;
    let y0 = (y.floor() as i32).clamp(0, rows as i32 - 1) as usize;
    let x1 = (x0 + 1).min(cols - 1);
    let y1 = (y0 + 1).min(rows - 1);

    let ax = x - x.floor();
    let ay = y - y.floor();

    let stride = cols;
    let v00 = img.data[y0 * stride + x0];
    let v01 = img.data[y0 * stride + x1];
    let v10 = img.data[y1 * stride + x0];
    let v11 = img.data[y1 * stride + x1];

    v00 * (1.0 - ax) * (1.0 - ay) + v01 * ax * (1.0 - ay) + v10 * (1.0 - ax) * ay + v11 * ax * ay
}

/// Run the Lucas-Kanade solver at a single pyramid level.
///
/// Returns `(flow_x, flow_y, min_eigenvalue, tracked)`.
///
/// * `prev`      — previous frame at this level (single-channel f32).
/// * `next`      — next frame at this level (single-channel f32).
/// * `prev_ix`   — Sobel-x derivative of `prev`.
/// * `prev_iy`   — Sobel-y derivative of `prev`.
/// * `px`, `py`  — reference-point coordinates in this level's space.
/// * `init_u`, `init_v` — initial optical flow estimate at this level.
/// * `half_win_w`, `half_win_h` — half-sizes of the tracking window.
/// * `max_iters` — maximum refinement iterations.
/// * `eps`       — convergence threshold (step size squared).
/// * `min_eigen_threshold` — reject tracking if min eigenvalue falls below this.
///
/// When the `simd` feature is enabled the H-matrix accumulation and the
/// per-iteration mismatch accumulation are delegated to
/// [`video_simd::simd_lk_accumulate_h`] and
/// [`video_simd::simd_lk_accumulate_mismatch`], which wrap their inner loops
/// in `pulp::Arch::dispatch` for LLVM-guided auto-vectorisation.  The gather
/// step (bilinear interpolation) remains scalar in both paths because it
/// involves non-sequential memory access.
#[allow(clippy::too_many_arguments)]
fn lk_single_level(
    prev: &Matrix<f32>,
    next: &Matrix<f32>,
    prev_ix: &Matrix<f32>,
    prev_iy: &Matrix<f32>,
    px: f32,
    py: f32,
    init_u: f32,
    init_v: f32,
    half_win_w: i32,
    half_win_h: i32,
    max_iters: i32,
    eps: f64,
    min_eigen_threshold: f64,
) -> (f32, f32, f64, bool) {
    // -------------------------------------------------------------------
    // Build H = Σ [[Ix², Ix·Iy],[Ix·Iy, Iy²]] over the tracking window.
    //
    // SIMD path: pre-gather Ix, Iy, and I1 values from the reference frame
    // into contiguous f32 buffers, then hand off the reduction to pulp.
    // Scalar path: accumulate inline exactly as in the original code.
    // -------------------------------------------------------------------

    #[cfg(feature = "simd")]
    let (h00, h01, h11, ix_win, iy_win, i1_win) = {
        let n_win = ((2 * half_win_h + 1) * (2 * half_win_w + 1)) as usize;
        let mut ix_win = Vec::with_capacity(n_win);
        let mut iy_win = Vec::with_capacity(n_win);
        let mut i1_win = Vec::with_capacity(n_win);

        for dy in -half_win_h..=half_win_h {
            for dx in -half_win_w..=half_win_w {
                let sx = px + dx as f32;
                let sy = py + dy as f32;
                ix_win.push(bilinear_interp(prev_ix, sx, sy));
                iy_win.push(bilinear_interp(prev_iy, sx, sy));
                i1_win.push(bilinear_interp(prev, sx, sy));
            }
        }

        let (h00, h01, h11) = video_simd::simd_lk_accumulate_h(&ix_win, &iy_win);
        (h00, h01, h11, ix_win, iy_win, i1_win)
    };

    #[cfg(not(feature = "simd"))]
    let (h00, h01, h11) = {
        let mut h00 = 0.0f64;
        let mut h01 = 0.0f64;
        let mut h11 = 0.0f64;
        for dy in -half_win_h..=half_win_h {
            for dx in -half_win_w..=half_win_w {
                let sx = px + dx as f32;
                let sy = py + dy as f32;
                let ix = bilinear_interp(prev_ix, sx, sy) as f64;
                let iy = bilinear_interp(prev_iy, sx, sy) as f64;
                h00 += ix * ix;
                h01 += ix * iy;
                h11 += iy * iy;
            }
        }
        (h00, h01, h11)
    };

    // -------------------------------------------------------------------
    // Compute min eigenvalue of H (normalised by window area).
    // -------------------------------------------------------------------
    let win_area = ((2 * half_win_w + 1) * (2 * half_win_h + 1)) as f64;
    let h00n = h00 / win_area;
    let h01n = h01 / win_area;
    let h11n = h11 / win_area;

    let trace = h00n + h11n;
    let det_n = h00n * h11n - h01n * h01n;
    let disc = ((trace * trace - 4.0 * det_n).max(0.0)).sqrt();
    let min_eigen = (trace - disc) * 0.5;

    let det = h00 * h11 - h01 * h01;

    if min_eigen < min_eigen_threshold || det.abs() < f64::EPSILON {
        return (init_u, init_v, min_eigen, false);
    }

    // H^-1 applied to a vector b = (1/det) * [[h11, -h01], [-h01, h00]] * b
    let inv_det = 1.0 / det;

    // -------------------------------------------------------------------
    // Iterative refinement.
    // -------------------------------------------------------------------
    let mut u = init_u as f64;
    let mut v = init_v as f64;

    // SIMD path: pre-allocate a reusable i2 buffer; refill each iteration.
    #[cfg(feature = "simd")]
    let mut i2_win = vec![0.0f32; ix_win.len()];

    for _iter in 0..max_iters {
        // ---------------------------------------------------------------
        // Accumulate the mismatch vector b = -Σ [Ix·It, Iy·It].
        // ---------------------------------------------------------------

        #[cfg(feature = "simd")]
        let (bx, by) = {
            // Regather I2 at current flow estimate (u, v).
            let mut w_idx = 0usize;
            for dy in -half_win_h..=half_win_h {
                for dx in -half_win_w..=half_win_w {
                    let sx = px as f64 + dx as f64;
                    let sy = py as f64 + dy as f64;
                    i2_win[w_idx] = bilinear_interp(next, (sx + u) as f32, (sy + v) as f32);
                    w_idx += 1;
                }
            }
            video_simd::simd_lk_accumulate_mismatch(&ix_win, &iy_win, &i1_win, &i2_win)
        };

        #[cfg(not(feature = "simd"))]
        let (bx, by) = {
            let mut bx = 0.0f64;
            let mut by = 0.0f64;
            for dy in -half_win_h..=half_win_h {
                for dx in -half_win_w..=half_win_w {
                    let sx = px as f64 + dx as f64;
                    let sy = py as f64 + dy as f64;

                    let i1 = bilinear_interp(prev, sx as f32, sy as f32) as f64;
                    let i2 = bilinear_interp(next, (sx + u) as f32, (sy + v) as f32) as f64;
                    let it = i2 - i1;

                    let ix = bilinear_interp(prev_ix, sx as f32, sy as f32) as f64;
                    let iy = bilinear_interp(prev_iy, sx as f32, sy as f32) as f64;

                    bx -= ix * it;
                    by -= iy * it;
                }
            }
            (bx, by)
        };

        // Solve H * (eta_u, eta_v) = (bx, by)
        let eta_u = (h11 * bx - h01 * by) * inv_det;
        let eta_v = (-h01 * bx + h00 * by) * inv_det;

        u += eta_u;
        v += eta_v;

        if eta_u * eta_u + eta_v * eta_v < eps {
            break;
        }
    }

    (u as f32, v as f32, min_eigen, true)
}

/// Compute the mean-absolute error (MAE) between matching `win_size`
/// windows in the previous and next frame around `prev_pt` → `next_pt`.
fn compute_tracking_error(
    prev: &Matrix<f32>,
    next: &Matrix<f32>,
    prev_pt: Point2f,
    next_pt: Point2f,
    half_win_w: i32,
    half_win_h: i32,
) -> f32 {
    let mut error = 0.0f32;
    let mut count = 0u32;

    for dy in -half_win_h..=half_win_h {
        for dx in -half_win_w..=half_win_w {
            let i1 = bilinear_interp(prev, prev_pt.x + dx as f32, prev_pt.y + dy as f32);
            let i2 = bilinear_interp(next, next_pt.x + dx as f32, next_pt.y + dy as f32);
            error += (i2 - i1).abs();
            count += 1;
        }
    }

    if count > 0 {
        error / count as f32
    } else {
        0.0
    }
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
/// # Arguments
/// * `prev_img`            — Previous single-channel 8-bit grayscale frame.
/// * `next_img`            — Next single-channel 8-bit grayscale frame.  Must
///   be the same size as `prev_img`.
/// * `prev_pts`            — Feature points to track, in `prev_img` coordinates.
/// * `initial_next_pts`    — Optional initial guess for `nextPts`.  Passed
///   together with [`OPTFLOW_USE_INITIAL_FLOW`]; ignored otherwise.
/// * `win_size`            — Size of the search window at each pyramid level.
/// * `max_level`           — Pyramid depth (0 = no pyramid, just the original).
/// * `criteria`            — Iteration termination criteria.
/// * `flags`               — Option flags; combine [`OPTFLOW_USE_INITIAL_FLOW`]
///   and/or [`OPTFLOW_LK_GET_MIN_EIGENVALS`].
/// * `min_eigen_threshold` — Points whose spatial-gradient matrix has a
///   minimum eigenvalue below this threshold are marked as lost.
///
/// # Returns
/// A tuple `(next_pts, status, err)`:
/// * `next_pts`  — Estimated positions of the tracked features in `next_img`.
/// * `status`    — Per-feature tracking flag: `1` = tracked, `0` = lost.
/// * `err`       — Per-feature tracking error (MAE by default; minimum
///   eigenvalue when [`OPTFLOW_LK_GET_MIN_EIGENVALS`] is set).
///
/// # Errors
/// Returns [`PureCvError::InvalidInput`] if either input image is not
/// single-channel, or if their dimensions differ.
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

    let n_pts = prev_pts.len();
    if n_pts == 0 {
        return Ok((Vec::new(), Vec::new(), Vec::new()));
    }

    // ------------------------------------------------------------------
    // Extract termination criteria
    // ------------------------------------------------------------------
    let max_iters = match criteria.type_ {
        TermType::Count => criteria.max_count,
        TermType::Eps => 30,
        TermType::Both => criteria.max_count,
    };
    let eps = match criteria.type_ {
        TermType::Count => 0.0,
        TermType::Eps | TermType::Both => criteria.epsilon * criteria.epsilon,
    };

    let max_level = max_level.max(0) as usize;
    let use_initial_flow = (flags & OPTFLOW_USE_INITIAL_FLOW) != 0;
    let get_min_eigenvals = (flags & OPTFLOW_LK_GET_MIN_EIGENVALS) != 0;

    let half_win_w = win_size.width / 2;
    let half_win_h = win_size.height / 2;

    // ------------------------------------------------------------------
    // Convert images to f32 and build pyramids
    // ------------------------------------------------------------------
    let prev_f32 = prev_img.convert_to::<f32>()?;
    let next_f32 = next_img.convert_to::<f32>()?;

    let prev_pyr = build_f32_pyramid(&prev_f32, max_level)?;
    let next_pyr = build_f32_pyramid(&next_f32, max_level)?;
    let actual_levels = prev_pyr.len().min(next_pyr.len());

    // Pre-compute Sobel derivatives for each level of the previous frame.
    // When `parallel` is enabled the per-level Sobel passes run concurrently.
    #[cfg(feature = "parallel")]
    let (prev_ix, prev_iy): (Vec<Matrix<f32>>, Vec<Matrix<f32>>) = {
        let pairs: Result<Vec<(Matrix<f32>, Matrix<f32>)>> = prev_pyr[..actual_levels]
            .par_iter()
            .map(|level| {
                let ix: Matrix<f32> = sobel(level, 1, 0, 3, 1.0, 0.0, BorderTypes::Reflect101)?;
                let iy: Matrix<f32> = sobel(level, 0, 1, 3, 1.0, 0.0, BorderTypes::Reflect101)?;
                Ok((ix, iy))
            })
            .collect();
        pairs?.into_iter().unzip()
    };

    #[cfg(not(feature = "parallel"))]
    let (prev_ix, prev_iy): (Vec<Matrix<f32>>, Vec<Matrix<f32>>) = {
        let mut prev_ix: Vec<Matrix<f32>> = Vec::with_capacity(actual_levels);
        let mut prev_iy: Vec<Matrix<f32>> = Vec::with_capacity(actual_levels);
        for level in &prev_pyr[..actual_levels] {
            let ix: Matrix<f32> = sobel(level, 1, 0, 3, 1.0, 0.0, BorderTypes::Reflect101)?;
            let iy: Matrix<f32> = sobel(level, 0, 1, 3, 1.0, 0.0, BorderTypes::Reflect101)?;
            prev_ix.push(ix);
            prev_iy.push(iy);
        }
        (prev_ix, prev_iy)
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

        let mut u = if use_initial_flow {
            (guess.x - prev_pt.x) * coarsest_scale
        } else {
            0.0f32
        };
        let mut v = if use_initial_flow {
            (guess.y - prev_pt.y) * coarsest_scale
        } else {
            0.0f32
        };

        let mut min_eigen = 1.0f64;
        let mut tracked = true;

        for level in (0..actual_levels).rev() {
            let scale = 1.0f32 / (1u32 << level) as f32;
            let px = prev_pt.x * scale;
            let py = prev_pt.y * scale;

            if level < coarsest {
                u *= 2.0;
                v *= 2.0;
            }

            let (fu, fv, ev, success) = lk_single_level(
                &prev_pyr[level],
                &next_pyr[level],
                &prev_ix[level],
                &prev_iy[level],
                px,
                py,
                u,
                v,
                half_win_w,
                half_win_h,
                max_iters,
                eps,
                min_eigen_threshold,
            );

            u = fu;
            v = fv;
            min_eigen = ev;

            if !success {
                tracked = false;
                break;
            }
        }

        if tracked {
            let next_pt = Point2f::new(prev_pt.x + u, prev_pt.y + v);
            let e = if get_min_eigenvals {
                min_eigen as f32
            } else {
                compute_tracking_error(
                    &prev_f32, &next_f32, prev_pt, next_pt, half_win_w, half_win_h,
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

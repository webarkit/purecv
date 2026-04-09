/*
 *  feature.rs
 *  purecv
 *
 *  This file is part of purecv - OpenCV.
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

//! Corner and feature detection functions mirroring OpenCV's `imgproc` feature API.
//!
//! The functions form the following natural pipeline:
//!
//! ```text
//! Sobel (core)
//!     ↓
//! corner_eigen_vals_and_vecs  ← structure tensor Ixx, Ixy, Iyy
//!     ↓                 ↘
//! corner_min_eigen_val   corner_harris
//!     ↘                 ↙
//!     good_features_to_track   ← outputs Vec<Point2f>
//!           ↓
//!      corner_sub_pix           ← sub-pixel refinement (issue #24)
//! ```
//!
//! `pre_corner_detect` is an independent simpler map using first + second derivatives.

use crate::core::error::{PureCvError, Result};
use crate::core::types::{BorderTypes, Point2f, Size2i, TermCriteria, TermType};
use crate::core::Matrix;
use crate::imgproc::derivatives::sobel;
use crate::imgproc::filter::box_filter;
use num_traits::{FromPrimitive, NumCast, ToPrimitive};

#[cfg(not(feature = "parallel"))]
use crate::core::utils::ParIterFallback;
#[cfg(feature = "parallel")]
use rayon::prelude::*;

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Computes eigenvalues and eigenvectors of the symmetric 2×2 matrix:
///   [a, b]
///   [b, c]
///
/// Returns (λ1, λ2, x1, y1, x2, y2) where λ1 ≥ λ2 and (xi, yi) are the
/// corresponding unit eigenvectors.
#[inline]
fn eigen_2x2(a: f32, b: f32, c: f32) -> (f32, f32, f32, f32, f32, f32) {
    let tr = a + c;
    let disc = ((a - c) * (a - c) + 4.0 * b * b).sqrt();
    let lambda1 = (tr + disc) * 0.5;
    let lambda2 = (tr - disc) * 0.5;

    let (x1, y1) = if b.abs() > f32::EPSILON {
        let dx = b;
        let dy = lambda1 - a;
        let norm = (dx * dx + dy * dy).sqrt();
        if norm > f32::EPSILON {
            (dx / norm, dy / norm)
        } else {
            (1.0_f32, 0.0_f32)
        }
    } else if a >= c {
        (1.0_f32, 0.0_f32)
    } else {
        (0.0_f32, 1.0_f32)
    };

    // Second eigenvector is perpendicular to the first.
    let (x2, y2) = (-y1, x1);

    (lambda1, lambda2, x1, y1, x2, y2)
}

/// Bilinear interpolation of a single-channel f32 matrix at a fractional position.
/// Out-of-bounds positions clamp to the nearest border pixel.
#[inline]
fn bilinear_interp_f32(mat: &Matrix<f32>, y: f64, x: f64) -> f32 {
    let rows = mat.rows as i32;
    let cols = mat.cols as i32;

    let x0 = (x.floor() as i32).clamp(0, cols - 1);
    let y0 = (y.floor() as i32).clamp(0, rows - 1);
    let x1 = (x0 + 1).clamp(0, cols - 1);
    let y1 = (y0 + 1).clamp(0, rows - 1);

    let ax = (x - x.floor()) as f32;
    let ay = (y - y.floor()) as f32;

    let v00 = *mat.at(y0, x0, 0).unwrap_or(&0.0);
    let v01 = *mat.at(y0, x1, 0).unwrap_or(&0.0);
    let v10 = *mat.at(y1, x0, 0).unwrap_or(&0.0);
    let v11 = *mat.at(y1, x1, 0).unwrap_or(&0.0);

    v00 * (1.0 - ax) * (1.0 - ay) + v01 * ax * (1.0 - ay) + v10 * (1.0 - ax) * ay + v11 * ax * ay
}

/// Builds the averaged structure tensor for `src` using a `block_size × block_size`
/// box filter, returning `(Ixx_avg, Ixy_avg, Iyy_avg)` as three separate f32 matrices.
fn compute_structure_tensor<T>(
    src: &Matrix<T>,
    block_size: i32,
    ksize: i32,
    border_type: BorderTypes,
) -> Result<(Matrix<f32>, Matrix<f32>, Matrix<f32>)>
where
    T: Default + Clone + ToPrimitive + FromPrimitive + NumCast + Copy + Send + Sync + 'static,
{
    if src.channels != 1 {
        return Err(PureCvError::InvalidInput(
            "corner detection only supports single-channel images".into(),
        ));
    }
    if block_size < 1 || block_size % 2 == 0 {
        return Err(PureCvError::InvalidInput(
            "block_size must be a positive odd integer".into(),
        ));
    }

    // Convert to f32 to avoid saturation during gradient computation.
    let src_f32 = src.convert_to::<f32>()?;

    let ix: Matrix<f32> = sobel(&src_f32, 1, 0, ksize, 1.0, 0.0, border_type)?;
    let iy: Matrix<f32> = sobel(&src_f32, 0, 1, ksize, 1.0, 0.0, border_type)?;

    let rows = src.rows;
    let cols = src.cols;
    let n = rows * cols;

    // Compute element-wise products.
    let mut ixx = Matrix::<f32>::new(rows, cols, 1);
    let mut ixy = Matrix::<f32>::new(rows, cols, 1);
    let mut iyy = Matrix::<f32>::new(rows, cols, 1);

    ixx.data
        .iter_mut()
        .zip(ix.data.iter())
        .take(n)
        .for_each(|(xx, gx)| {
            *xx = *gx * *gx;
        });

    ixy.data
        .iter_mut()
        .zip(ix.data.iter())
        .zip(iy.data.iter())
        .take(n)
        .for_each(|((xy, gx), gy)| {
            *xy = *gx * *gy;
        });

    iyy.data
        .iter_mut()
        .zip(iy.data.iter())
        .take(n)
        .for_each(|(yy, gy)| {
            *yy = *gy * *gy;
        });

    // Smooth each component with a box filter.
    use crate::core::types::{Point2i, Size2i};
    let ksize_s = Size2i::new(block_size, block_size);
    let anchor = Point2i::new(-1, -1);

    let ixx_avg = box_filter(&ixx, ksize_s, anchor, true, border_type)?;
    let ixy_avg = box_filter(&ixy, ksize_s, anchor, true, border_type)?;
    let iyy_avg = box_filter(&iyy, ksize_s, anchor, true, border_type)?;

    Ok((ixx_avg, ixy_avg, iyy_avg))
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Calculates eigenvalues and eigenvectors of image blocks for corner detection.
///
/// For each pixel the function computes the 2×2 neighbourhood covariance matrix
/// of the image derivatives (structure tensor), applies a `block_size×block_size`
/// averaging box filter, then finds its eigenvalues and eigenvectors.
///
/// The output is a **6-channel** `f32` matrix where each pixel stores:
/// `(λ1, λ2, x1, y1, x2, y2)` — eigenvalues sorted so λ1 ≥ λ2, and the
/// corresponding unit eigenvectors `(x1,y1)`, `(x2,y2)`.
///
/// # Arguments
/// * `src`        — Single-channel input image (any numeric type).
/// * `block_size` — Neighbourhood size; must be a positive odd integer.
/// * `ksize`      — Aperture size for the Sobel operator (3, 5, or −1 for Scharr).
/// * `border_type`— Border extrapolation method.
pub fn corner_eigen_vals_and_vecs<T>(
    src: &Matrix<T>,
    block_size: i32,
    ksize: i32,
    border_type: BorderTypes,
) -> Result<Matrix<f32>>
where
    T: Default + Clone + ToPrimitive + FromPrimitive + NumCast + Copy + Send + Sync + 'static,
{
    let (ixx, ixy, iyy) = compute_structure_tensor(src, block_size, ksize, border_type)?;

    let rows = src.rows;
    let cols = src.cols;

    // Output: 6 channels per pixel: (λ1, λ2, x1, y1, x2, y2)
    let mut dst = Matrix::<f32>::new(rows, cols, 6);

    dst.data
        .par_chunks_mut(6)
        .enumerate()
        .for_each(|(i, pixel)| {
            let a = ixx.data[i];
            let b = ixy.data[i];
            let c = iyy.data[i];
            let (lam1, lam2, x1, y1, x2, y2) = eigen_2x2(a, b, c);
            pixel[0] = lam1;
            pixel[1] = lam2;
            pixel[2] = x1;
            pixel[3] = y1;
            pixel[4] = x2;
            pixel[5] = y2;
        });

    Ok(dst)
}

/// Calculates the minimal eigenvalue of the gradient covariance matrix for each pixel.
///
/// This is the **Shi-Tomasi** corner response function. The output is a single-channel
/// `f32` matrix where larger values indicate stronger corners.
///
/// # Arguments
/// * `src`        — Single-channel input image.
/// * `block_size` — Neighbourhood size; must be a positive odd integer.
/// * `ksize`      — Aperture size for the Sobel operator.
/// * `border_type`— Border extrapolation method.
pub fn corner_min_eigen_val<T>(
    src: &Matrix<T>,
    block_size: i32,
    ksize: i32,
    border_type: BorderTypes,
) -> Result<Matrix<f32>>
where
    T: Default + Clone + ToPrimitive + FromPrimitive + NumCast + Copy + Send + Sync + 'static,
{
    let (ixx, ixy, iyy) = compute_structure_tensor(src, block_size, ksize, border_type)?;

    let rows = src.rows;
    let cols = src.cols;

    let mut dst = Matrix::<f32>::new(rows, cols, 1);

    dst.data.iter_mut().enumerate().for_each(|(i, v)| {
        let a = ixx.data[i];
        let b = ixy.data[i];
        let c = iyy.data[i];
        let tr = a + c;
        let disc = ((a - c) * (a - c) + 4.0 * b * b).sqrt();
        *v = (tr - disc) * 0.5; // minimum eigenvalue
    });

    Ok(dst)
}

/// Harris corner detector.
///
/// Computes `det(M) − k · trace(M)²` for each pixel neighbourhood, where `M`
/// is the 2×2 gradient covariance matrix.  Larger positive values indicate
/// corners; large negative values indicate edges; values near zero indicate
/// flat regions.
///
/// # Arguments
/// * `src`        — Single-channel input image.
/// * `block_size` — Neighbourhood size; must be a positive odd integer.
/// * `ksize`      — Aperture size for the Sobel operator.
/// * `k`          — Harris detector free parameter (typically 0.04–0.06).
/// * `border_type`— Border extrapolation method.
pub fn corner_harris<T>(
    src: &Matrix<T>,
    block_size: i32,
    ksize: i32,
    k: f64,
    border_type: BorderTypes,
) -> Result<Matrix<f32>>
where
    T: Default + Clone + ToPrimitive + FromPrimitive + NumCast + Copy + Send + Sync + 'static,
{
    let (ixx, ixy, iyy) = compute_structure_tensor(src, block_size, ksize, border_type)?;

    let rows = src.rows;
    let cols = src.cols;
    let k_f32 = k as f32;

    let mut dst = Matrix::<f32>::new(rows, cols, 1);

    dst.data.iter_mut().enumerate().for_each(|(i, v)| {
        let a = ixx.data[i];
        let b = ixy.data[i];
        let c = iyy.data[i];
        let det = a * c - b * b;
        let tr = a + c;
        *v = det - k_f32 * tr * tr;
    });

    Ok(dst)
}

/// Determines strong corners on an image.
///
/// This is the Shi-Tomasi / Harris corner detector with quality filtering,
/// non-maximum suppression and minimum-distance enforcement.
///
/// # Arguments
/// * `src`                — Single-channel input image.
/// * `max_corners`        — Maximum number of corners to return; ≤ 0 means unlimited.
/// * `quality_level`      — Fraction of the best corner response; corners below
///   `quality_level × max_response` are discarded.
/// * `min_distance`       — Minimum Euclidean distance between returned corners.
/// * `block_size`         — Neighbourhood size for the structure tensor.
/// * `use_harris_detector`— Use Harris response (`true`) or min eigenvalue (`false`).
/// * `harris_k`           — Harris detector free parameter (used only when
///   `use_harris_detector` is `true`).
///
/// Returns a vector of corner positions sorted by response strength (best first).
pub fn good_features_to_track<T>(
    src: &Matrix<T>,
    max_corners: i32,
    quality_level: f64,
    min_distance: f64,
    block_size: i32,
    use_harris_detector: bool,
    harris_k: f64,
) -> Result<Vec<Point2f>>
where
    T: Default + Clone + ToPrimitive + FromPrimitive + NumCast + Copy + Send + Sync + 'static,
{
    if src.channels != 1 {
        return Err(PureCvError::InvalidInput(
            "good_features_to_track only supports single-channel images".into(),
        ));
    }
    if quality_level <= 0.0 || quality_level > 1.0 {
        return Err(PureCvError::InvalidInput(
            "quality_level must be in (0, 1]".into(),
        ));
    }

    // Compute corner response map.
    let response: Matrix<f32> = if use_harris_detector {
        corner_harris(src, block_size, 3, harris_k, BorderTypes::Reflect101)?
    } else {
        corner_min_eigen_val(src, block_size, 3, BorderTypes::Reflect101)?
    };

    let rows = src.rows;
    let cols = src.cols;

    // Find the global maximum response.
    let max_response = response
        .data
        .iter()
        .cloned()
        .fold(f32::NEG_INFINITY, f32::max);

    if max_response <= 0.0 {
        return Ok(Vec::new());
    }

    let threshold = (quality_level * max_response as f64) as f32;

    // Collect candidate pixels that exceed the threshold and are strict local
    // maxima in a 3×3 window.
    let mut candidates: Vec<(f32, usize, usize)> = Vec::new();
    for y in 1..rows.saturating_sub(1) {
        for x in 1..cols.saturating_sub(1) {
            let v = *response.at(y as i32, x as i32, 0).unwrap_or(&0.0);
            if v <= threshold {
                continue;
            }
            // Local non-maximum suppression in 3×3 window.
            let is_local_max = [
                (-1, -1),
                (-1, 0),
                (-1, 1),
                (0, -1),
                (0, 1),
                (1, -1),
                (1, 0),
                (1, 1),
            ]
            .iter()
            .all(|&(dy, dx)| {
                let ny = y as i32 + dy;
                let nx = x as i32 + dx;
                *response.at(ny, nx, 0).unwrap_or(&0.0) <= v
            });
            if is_local_max {
                candidates.push((v, y, x));
            }
        }
    }

    // Sort by response descending.
    candidates.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    // Greedily select corners with minimum distance constraint.
    let min_dist_sq = min_distance * min_distance;
    let mut selected: Vec<Point2f> = Vec::new();

    'outer: for (_, cy, cx) in candidates {
        let fx = cx as f32;
        let fy = cy as f32;
        for &p in &selected {
            let dx = fx - p.x;
            let dy = fy - p.y;
            if ((dx * dx + dy * dy) as f64) < min_dist_sq {
                continue 'outer;
            }
        }
        selected.push(Point2f::new(fx, fy));
        if max_corners > 0 && selected.len() >= max_corners as usize {
            break;
        }
    }

    Ok(selected)
}

/// Refines the corner locations to sub-pixel accuracy.
///
/// For each corner the algorithm iteratively solves the over-determined linear
/// system derived from the orthogonality condition
/// `∇I(q) · (q − p) = 0` for every sample `q` in the search window, obtaining
/// the sub-pixel offset `p`.  Iteration stops when the shift magnitude drops
/// below `criteria.epsilon` or `criteria.max_count` iterations are reached.
///
/// # Arguments
/// * `src`       — Single-channel input image (grayscale, any numeric type).
/// * `corners`   — Initial corner positions; updated in place.
/// * `win_size`  — Half-size of the search window (e.g. `Size2i::new(5,5)` gives
///   an 11×11 window).
/// * `zero_zone` — Half-size of the dead zone in the window centre.  Pass
///   `Size2i::new(-1,-1)` to disable.
/// * `criteria`  — Termination criteria ([`TermCriteria`]).
pub fn corner_sub_pix<T>(
    src: &Matrix<T>,
    corners: &mut [Point2f],
    win_size: Size2i,
    zero_zone: Size2i,
    criteria: TermCriteria,
) -> Result<()>
where
    T: Default + Clone + ToPrimitive + FromPrimitive + NumCast + Copy + Send + Sync + 'static,
{
    if src.channels != 1 {
        return Err(PureCvError::InvalidInput(
            "corner_sub_pix only supports single-channel images".into(),
        ));
    }
    if win_size.width < 1 || win_size.height < 1 {
        return Err(PureCvError::InvalidInput(
            "win_size must be at least 1 in each dimension".into(),
        ));
    }

    // Pre-compute gradient images (f32 precision).
    let src_f32 = src.convert_to::<f32>()?;
    let grad_x: Matrix<f32> = sobel(&src_f32, 1, 0, 3, 1.0, 0.0, BorderTypes::Reflect101)?;
    let grad_y: Matrix<f32> = sobel(&src_f32, 0, 1, 3, 1.0, 0.0, BorderTypes::Reflect101)?;

    let eps_sq = criteria.epsilon * criteria.epsilon;
    let max_iter = criteria.max_count;
    let use_count = matches!(criteria.type_, TermType::Count | TermType::Both);
    let use_eps = matches!(criteria.type_, TermType::Eps | TermType::Both);

    for corner in corners.iter_mut() {
        let mut cx = corner.x as f64;
        let mut cy = corner.y as f64;

        for _iter in 0..max_iter {
            // Accumulate the 2×2 normal-equation matrix G and right-hand side b.
            let mut g00: f64 = 0.0;
            let mut g01: f64 = 0.0;
            let mut g11: f64 = 0.0;
            let mut b0: f64 = 0.0;
            let mut b1: f64 = 0.0;

            for ky in -win_size.height..=win_size.height {
                for kx in -win_size.width..=win_size.width {
                    // Skip dead zone.
                    if zero_zone.width >= 0
                        && zero_zone.height >= 0
                        && kx.abs() <= zero_zone.width
                        && ky.abs() <= zero_zone.height
                    {
                        continue;
                    }

                    let qx = cx + kx as f64;
                    let qy = cy + ky as f64;

                    let ix = bilinear_interp_f32(&grad_x, qy, qx) as f64;
                    let iy = bilinear_interp_f32(&grad_y, qy, qx) as f64;

                    g00 += ix * ix;
                    g01 += ix * iy;
                    g11 += iy * iy;
                    b0 += ix * (ix * qx + iy * qy);
                    b1 += iy * (ix * qx + iy * qy);
                }
            }

            // Solve G * [px; py] = b using the 2×2 closed-form inverse.
            let det = g00 * g11 - g01 * g01;
            if det.abs() < f64::EPSILON {
                // Degenerate: leave corner unchanged.
                break;
            }

            let px = (g11 * b0 - g01 * b1) / det;
            let py = (-g01 * b0 + g00 * b1) / det;

            let dx = px - cx;
            let dy = py - cy;

            cx = px;
            cy = py;

            if use_eps && (dx * dx + dy * dy) < eps_sq {
                break;
            }
            if use_count && _iter + 1 >= max_iter {
                break;
            }
        }

        corner.x = cx as f32;
        corner.y = cy as f32;
    }

    Ok(())
}

/// Calculates a feature map for corner detection.
///
/// Computes `Ix² · Iyy + Iy² · Ixx − 2 · Ix · Iy · Ixy` for each pixel,
/// where `Ix`, `Iy` are first-order and `Ixx`, `Ixy`, `Iyy` are second-order
/// Sobel derivatives with aperture `ksize`.  Large values indicate corners.
///
/// # Arguments
/// * `src`        — Single-channel input image.
/// * `ksize`      — Aperture size for the Sobel operator (3, 5, or −1 for Scharr).
/// * `border_type`— Border extrapolation method.
pub fn pre_corner_detect<T>(
    src: &Matrix<T>,
    ksize: i32,
    border_type: BorderTypes,
) -> Result<Matrix<f32>>
where
    T: Default + Clone + ToPrimitive + FromPrimitive + NumCast + Copy + Send + Sync + 'static,
{
    if src.channels != 1 {
        return Err(PureCvError::InvalidInput(
            "pre_corner_detect only supports single-channel images".into(),
        ));
    }

    let src_f32 = src.convert_to::<f32>()?;

    // First derivatives.
    let ix: Matrix<f32> = sobel(&src_f32, 1, 0, ksize, 1.0, 0.0, border_type)?;
    let iy: Matrix<f32> = sobel(&src_f32, 0, 1, ksize, 1.0, 0.0, border_type)?;

    // Second derivatives (computed from first derivatives, halved as in OpenCV).
    let ixx: Matrix<f32> = sobel(&ix, 1, 0, ksize, 0.5, 0.0, border_type)?;
    let iyy: Matrix<f32> = sobel(&iy, 0, 1, ksize, 0.5, 0.0, border_type)?;
    let ixy: Matrix<f32> = sobel(&ix, 0, 1, ksize, 0.5, 0.0, border_type)?;

    let rows = src.rows;
    let cols = src.cols;

    let mut dst = Matrix::<f32>::new(rows, cols, 1);

    dst.data.iter_mut().enumerate().for_each(|(i, v)| {
        let gx = ix.data[i];
        let gy = iy.data[i];
        let gxx = ixx.data[i];
        let gxy = ixy.data[i];
        let gyy = iyy.data[i];
        *v = gx * gx * gyy + gy * gy * gxx - 2.0 * gx * gy * gxy;
    });

    Ok(dst)
}

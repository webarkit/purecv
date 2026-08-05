/*
 *  homography.rs
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

//! Perspective homography estimation — `find_homography`.
//!
//! Implements the *Direct Linear Transform* (DLT) algorithm with isotropic
//! point normalization, plus an optional RANSAC wrapper for robustness
//! against outliers.

use alloc::{format, string::ToString, vec, vec::Vec};
#[allow(unused_imports)]
use num_traits::Float;

use crate::core::error::{PureCvError, Result};
use crate::core::types::Point2f;
use crate::core::Matrix;

use super::linalg::{mat3_inv, mat3_mul, null_space_vector, Lcg};

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Method used to compute a homography matrix in [`find_homography`].
///
/// Mirrors `cv::HomographyMethod` from OpenCV.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum HomographyMethod {
    /// Use all points (no outlier rejection).  Equivalent to passing `0` in
    /// OpenCV.
    None = 0,
    /// Least-Median robust method (not yet distinct from RANSAC in this
    /// implementation).
    LMedS = 4,
    /// RANSAC-based robust method.
    Ransac = 8,
    /// PROSAC-based robust method (currently falls back to RANSAC).
    Rho = 16,
}

/// Finds a perspective transformation between two planes.
///
/// Given `src_points` and `dst_points`, returns the 3×3 homography matrix
/// **H** (single-channel `f64`) such that for each correspondence *i*:
///
/// ```text
/// [x', y', w']^T  ≈  H * [x, y, 1]^T     (dst ≈ H * src)
/// ```
///
/// At least **4** matching point pairs are required.  For noisy or cluttered
/// data use `HomographyMethod::Ransac` together with an appropriate
/// `ransac_reproj_threshold`.
///
/// # Arguments
///
/// * `src_points` – Coordinates in the source (original) image.
/// * `dst_points` – Coordinates of the matching points in the target image.
/// * `method` – Computation method (see [`HomographyMethod`]).
/// * `ransac_reproj_threshold` – Maximum allowed reprojection error (in pixels)
///   to classify a correspondence as an inlier.  Only used for RANSAC / RHO.
/// * `mask` – When `Some`, the vector is resized to `src_points.len()` and
///   filled with `1` (inlier) or `0` (outlier) after estimation.
///
/// # Errors
///
/// Returns [`PureCvError::InvalidInput`] when fewer than 4 point pairs are
/// given or RANSAC cannot find enough inliers.
///
/// # Divergences from OpenCV
///
/// | OpenCV | purecv |
/// |--------|--------|
/// | Accepts `Mat` (float/double, 2-channel or Nx2) | Accepts `&[Point2f]` |
/// | Returns a `Mat` | Returns `Matrix<f64>` |
/// | Jacobian output via `mask` Mat | `mask: Option<&mut Vec<u8>>` |
pub fn find_homography(
    src_points: &[Point2f],
    dst_points: &[Point2f],
    method: HomographyMethod,
    ransac_reproj_threshold: f64,
    mask: Option<&mut Vec<u8>>,
) -> Result<Matrix<f64>> {
    let n = src_points.len();
    if src_points.len() != dst_points.len() {
        return Err(PureCvError::InvalidInput(format!(
            "src_points ({}) and dst_points ({}) must have the same length",
            src_points.len(),
            dst_points.len()
        )));
    }
    if n < 4 {
        return Err(PureCvError::InvalidInput(
            "find_homography requires at least 4 matching point pairs".to_string(),
        ));
    }

    match method {
        HomographyMethod::None => {
            let h = dlt_homography(src_points, dst_points)?;
            if let Some(m) = mask {
                m.clear();
                m.resize(n, 1);
            }
            Ok(h)
        }
        HomographyMethod::Ransac | HomographyMethod::Rho | HomographyMethod::LMedS => {
            ransac_homography(src_points, dst_points, ransac_reproj_threshold, mask)
        }
    }
}

// ---------------------------------------------------------------------------
// DLT (all points, with isotropic normalization)
// ---------------------------------------------------------------------------

/// Estimate a homography from `n ≥ 4` point pairs using the normalized DLT.
fn dlt_homography(src: &[Point2f], dst: &[Point2f]) -> Result<Matrix<f64>> {
    let n = src.len();

    // --- Isotropic normalization ---
    let (src_xy, t1) = normalize_points(src);
    let (dst_xy, t2) = normalize_points(dst);

    // --- Build 2n × 9 system matrix A ---
    // For each pair (x, y) → (x', y'):
    //   row 2i:   [-x, -y, -1,  0,  0,  0,  x'x,  x'y,  x']
    //   row 2i+1: [ 0,  0,  0, -x, -y, -1,  y'x,  y'y,  y']
    let mut a = vec![0.0f64; 2 * n * 9];
    for i in 0..n {
        let x = src_xy[2 * i];
        let y = src_xy[2 * i + 1];
        let xp = dst_xy[2 * i];
        let yp = dst_xy[2 * i + 1];

        let base = 2 * i * 9;
        a[base] = -x;
        a[base + 1] = -y;
        a[base + 2] = -1.0;
        // a[base + 3..5] already zero
        a[base + 6] = xp * x;
        a[base + 7] = xp * y;
        a[base + 8] = xp;

        let base = (2 * i + 1) * 9;
        // a[base..base+3] already zero
        a[base + 3] = -x;
        a[base + 4] = -y;
        a[base + 5] = -1.0;
        a[base + 6] = yp * x;
        a[base + 7] = yp * y;
        a[base + 8] = yp;
    }

    // --- Null-space → vectorized H in normalized coordinates ---
    let h_vec = null_space_vector(&a, 2 * n, 9);
    let h_norm: [f64; 9] = h_vec.try_into().map_err(|v: Vec<f64>| {
        PureCvError::InternalError(format!(
            "null_space_vector: expected 9 elements, got {}",
            v.len()
        ))
    })?;

    // --- Denormalize: H = T2^{-1} * H_norm * T1 ---
    let t2_inv = mat3_inv(&t2)
        .ok_or_else(|| PureCvError::InternalError("Normalization matrix T2 is singular".into()))?;
    let tmp = mat3_mul(&t2_inv, &h_norm);
    let h_final = mat3_mul(&tmp, &t1);

    // Scale so that H[2][2] = 1 when it is non-negligible.
    let scale = h_final[8];
    let h_data: Vec<f64> = if scale.abs() > 1e-12 {
        h_final.iter().map(|&v| v / scale).collect()
    } else {
        h_final.to_vec()
    };

    Ok(Matrix::from_vec(3, 3, 1, h_data))
}

// ---------------------------------------------------------------------------
// RANSAC
// ---------------------------------------------------------------------------

fn ransac_homography(
    src: &[Point2f],
    dst: &[Point2f],
    threshold: f64,
    mask: Option<&mut Vec<u8>>,
) -> Result<Matrix<f64>> {
    let n = src.len();
    const MAX_ITERS: usize = 2000;
    const MIN_INLIERS: usize = 4;

    let mut rng = Lcg::new(0x9e37_79b9_7f4a_7c15);
    let mut best_count = 0usize;
    let mut best_inliers = vec![0u8; n];
    let mut best_h: Option<[f64; 9]> = None;

    for _ in 0..MAX_ITERS {
        let idx = sample_no_replace(&mut rng, n, 4);
        let s: Vec<Point2f> = idx.iter().map(|&i| src[i]).collect();
        let d: Vec<Point2f> = idx.iter().map(|&i| dst[i]).collect();

        let h = match dlt_homography(&s, &d) {
            Ok(h) => h,
            Err(_) => continue,
        };

        let h9: [f64; 9] = h
            .data
            .try_into()
            .map_err(|_| PureCvError::InternalError("unexpected data length".into()))?;
        let (count, inliers) = count_inliers(src, dst, &h9, threshold);

        if count > best_count {
            best_count = count;
            best_inliers = inliers;
            best_h = Some(h9);
        }

        if best_count >= n * 9 / 10 {
            break;
        }
    }

    if best_count < MIN_INLIERS || best_h.is_none() {
        return Err(PureCvError::InvalidInput(
            "RANSAC: not enough inliers found".to_string(),
        ));
    }

    // Refit from all inliers.
    let in_src: Vec<Point2f> = src
        .iter()
        .zip(best_inliers.iter())
        .filter_map(|(&p, &m)| if m == 1 { Some(p) } else { None })
        .collect();
    let in_dst: Vec<Point2f> = dst
        .iter()
        .zip(best_inliers.iter())
        .filter_map(|(&p, &m)| if m == 1 { Some(p) } else { None })
        .collect();

    let refined = dlt_homography(&in_src, &in_dst)?;

    if let Some(m) = mask {
        let h9: [f64; 9] = refined
            .data
            .clone()
            .try_into()
            .map_err(|_| PureCvError::InternalError("unexpected data length".into()))?;
        let (_, final_inliers) = count_inliers(src, dst, &h9, threshold);
        *m = final_inliers;
    }

    Ok(refined)
}

/// Sample `k` distinct indices from `[0, n)` without replacement.
fn sample_no_replace(rng: &mut Lcg, n: usize, k: usize) -> Vec<usize> {
    let mut used = vec![false; n];
    let mut out = Vec::with_capacity(k);
    while out.len() < k {
        let i = rng.next_usize(n);
        if !used[i] {
            used[i] = true;
            out.push(i);
        }
    }
    out
}

/// Count how many point pairs satisfy the reprojection threshold under `h`.
fn count_inliers(
    src: &[Point2f],
    dst: &[Point2f],
    h: &[f64; 9],
    threshold: f64,
) -> (usize, Vec<u8>) {
    let n = src.len();
    let mut inliers = vec![0u8; n];
    let mut count = 0usize;
    let thr2 = threshold * threshold;

    for i in 0..n {
        let x = src[i].x as f64;
        let y = src[i].y as f64;
        let denom = h[6] * x + h[7] * y + h[8];
        if denom.abs() < 1e-12 {
            continue;
        }
        let px = (h[0] * x + h[1] * y + h[2]) / denom;
        let py = (h[3] * x + h[4] * y + h[5]) / denom;
        let ex = px - dst[i].x as f64;
        let ey = py - dst[i].y as f64;
        if ex * ex + ey * ey <= thr2 {
            inliers[i] = 1;
            count += 1;
        }
    }
    (count, inliers)
}

// ---------------------------------------------------------------------------
// Point normalization
// ---------------------------------------------------------------------------

/// Isotropic normalization: translate centroid to origin; scale so the mean
/// distance from the origin equals √2.
///
/// Returns `(flat_xy, T)` where `flat_xy` has shape `[x0, y0, x1, y1, …]`
/// and `T` is the 3×3 row-major normalization matrix such that the
/// normalized point `p̃ = T * [x, y, 1]^T`.
fn normalize_points(pts: &[Point2f]) -> (Vec<f64>, [f64; 9]) {
    let n = pts.len() as f64;
    let cx: f64 = pts.iter().map(|p| p.x as f64).sum::<f64>() / n;
    let cy: f64 = pts.iter().map(|p| p.y as f64).sum::<f64>() / n;

    let mean_dist: f64 = pts
        .iter()
        .map(|p| {
            let dx = p.x as f64 - cx;
            let dy = p.y as f64 - cy;
            (dx * dx + dy * dy).sqrt()
        })
        .sum::<f64>()
        / n;

    let s = if mean_dist < 1e-12 {
        1.0
    } else {
        core::f64::consts::SQRT_2 / mean_dist
    };

    let mut out = vec![0.0f64; pts.len() * 2];
    for (i, p) in pts.iter().enumerate() {
        out[2 * i] = s * (p.x as f64 - cx);
        out[2 * i + 1] = s * (p.y as f64 - cy);
    }

    // T = [ s,  0,  -s*cx ]
    //     [ 0,  s,  -s*cy ]
    //     [ 0,  0,    1   ]
    let t = [s, 0.0, -s * cx, 0.0, s, -s * cy, 0.0, 0.0, 1.0];
    (out, t)
}

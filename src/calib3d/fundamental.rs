/*
 *  fundamental.rs
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

use alloc::{string::ToString, vec, vec::Vec};
#[allow(unused_imports)]
use num_traits::Float;

use super::linalg::{mat3_mul, null_space_vector, svd_3x3, Lcg};
use crate::core::error::{PureCvError, Result};
use crate::core::types::Point2f;
use crate::core::Matrix;

/// Method flags for computing the fundamental matrix.
#[allow(non_camel_case_types)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FundamentalMatMethod {
    FM_8POINT = 2,
    FM_RANSAC = 8,
}

/// Calculates a fundamental matrix from the corresponding points in two images.
///
/// # Arguments
///
/// * `points1` - Array of N points from the first image.
/// * `points2` - Array of N points from the second image.
/// * `method` - Method for computing the fundamental matrix (e.g., `FundamentalMatMethod::FM_RANSAC`).
/// * `ransac_reproj_threshold` - Parameter used only for RANSAC. It denotes the maximum
///   distance from a point to its epipolar line in pixels.
/// * `confidence` - Parameter used only for RANSAC. It denotes the confidence level
///   that the estimated matrix is correct.
/// * `max_iters` - Maximum number of iterations for RANSAC.
/// * `mask` - Optional output mask of inliers/outliers (1 for inliers, 0 for outliers).
///
/// # Returns
///
/// Returns a `Result<Matrix<f64>>` containing the computed 3x3 fundamental matrix.
///
/// # Errors
///
/// Returns an error if:
/// * `points1` and `points2` have different lengths.
/// * There are fewer than 8 point correspondences.
/// * The RANSAC algorithm fails to find a valid fundamental matrix model.
/// * The null space vector computation fails (8-point algorithm).
///
/// # Examples
///
/// ```
/// use purecv::core::types::Point2f;
/// use purecv::calib3d::fundamental::{find_fundamental_mat, FundamentalMatMethod};
///
/// let points1 = vec![
///     Point2f::new(0.0, 0.0), Point2f::new(1.0, 0.0),
///     Point2f::new(0.0, 1.0), Point2f::new(1.0, 1.0),
///     Point2f::new(0.5, 0.5), Point2f::new(0.2, 0.8),
///     Point2f::new(0.8, 0.2), Point2f::new(0.3, 0.3),
/// ];
/// let points2 = points1.clone(); // In practice, these would be matched points
///
/// let mut mask = Vec::new();
/// let f_mat = find_fundamental_mat(
///     &points1,
///     &points2,
///     FundamentalMatMethod::FM_8POINT,
///     3.0,
///     0.99,
///     1000,
///     Some(&mut mask),
/// ).unwrap();
/// ```
pub fn find_fundamental_mat(
    points1: &[Point2f],
    points2: &[Point2f],
    method: FundamentalMatMethod,
    ransac_reproj_threshold: f64,
    confidence: f64,
    max_iters: usize,
    mask: Option<&mut Vec<u8>>,
) -> Result<Matrix<f64>> {
    if points1.len() != points2.len() {
        return Err(PureCvError::InvalidInput(
            "points1 and points2 must have the same length".to_string(),
        ));
    }
    if points1.len() < 8 {
        return Err(PureCvError::InvalidInput(
            "find_fundamental_mat requires at least 8 point correspondences".to_string(),
        ));
    }

    match method {
        FundamentalMatMethod::FM_8POINT => {
            let f = solve_8point(points1, points2)?;
            if let Some(m) = mask {
                m.clear();
                m.resize(points1.len(), 1);
            }
            Ok(f)
        }
        FundamentalMatMethod::FM_RANSAC => ransac_fundamental(
            points1,
            points2,
            ransac_reproj_threshold,
            confidence,
            max_iters,
            mask,
        ),
    }
}

/// Normalized 8-Point Algorithm to solve for the fundamental matrix.
fn solve_8point(pts1: &[Point2f], pts2: &[Point2f]) -> Result<Matrix<f64>> {
    let n = pts1.len();
    let (cx1, cy1, s1) = compute_normalization(pts1);
    let (cx2, cy2, s2) = compute_normalization(pts2);

    let mut a = vec![0.0f64; n * 9];
    for i in 0..n {
        let u1 = s1 * (pts1[i].x as f64 - cx1);
        let v1 = s1 * (pts1[i].y as f64 - cy1);
        let u2 = s2 * (pts2[i].x as f64 - cx2);
        let v2 = s2 * (pts2[i].y as f64 - cy2);

        a[i * 9] = u2 * u1;
        a[i * 9 + 1] = u2 * v1;
        a[i * 9 + 2] = u2;
        a[i * 9 + 3] = v2 * u1;
        a[i * 9 + 4] = v2 * v1;
        a[i * 9 + 5] = v2;
        a[i * 9 + 6] = u1;
        a[i * 9 + 7] = v1;
        a[i * 9 + 8] = 1.0;
    }

    let f = null_space_vector(&a, n, 9);
    if f.len() != 9 {
        return Err(PureCvError::InternalError(
            "Null space vector computation failed".to_string(),
        ));
    }

    let f_mat = [f[0], f[1], f[2], f[3], f[4], f[5], f[6], f[7], f[8]];

    // Enforce rank-2 constraint: set the smallest singular value to 0
    let (u, sigma, vt) = svd_3x3(&f_mat);
    let sigma_diag = [sigma[0], sigma[1], 0.0];

    let mut u_sigma = [0.0f64; 9];
    for row in 0..3 {
        u_sigma[row * 3] = u[row * 3] * sigma_diag[0];
        u_sigma[row * 3 + 1] = u[row * 3 + 1] * sigma_diag[1];
        u_sigma[row * 3 + 2] = 0.0;
    }

    let f_constrained = mat3_mul(&u_sigma, &vt);

    // De-normalize: F = T2^T * F_constrained * T1
    let t1 = [s1, 0.0, -s1 * cx1, 0.0, s1, -s1 * cy1, 0.0, 0.0, 1.0];
    let t2_t = [s2, 0.0, 0.0, 0.0, s2, 0.0, -s2 * cx2, -s2 * cy2, 1.0];

    let temp = mat3_mul(&t2_t, &f_constrained);
    let f_final = mat3_mul(&temp, &t1);

    Ok(Matrix::from_vec(3, 3, 1, f_final.to_vec()))
}

/// Compute center of mass and scale factor to normalize a set of 2D points.
fn compute_normalization(pts: &[Point2f]) -> (f64, f64, f64) {
    let n = pts.len() as f64;
    let mut cx = 0.0;
    let mut cy = 0.0;
    for p in pts {
        cx += p.x as f64;
        cy += p.y as f64;
    }
    cx /= n;
    cy /= n;

    let mut mean_dist = 0.0;
    for p in pts {
        let dx = p.x as f64 - cx;
        let dy = p.y as f64 - cy;
        mean_dist += (dx * dx + dy * dy).sqrt();
    }
    mean_dist /= n;

    let scale = if mean_dist > 1e-10 {
        2.0f64.sqrt() / mean_dist
    } else {
        1.0
    };

    (cx, cy, scale)
}

/// Compute Sampson distance (first-order geometric error approximation) for fundamental matrix.
fn sampson_distance(pt1: Point2f, pt2: Point2f, f: &[f64; 9]) -> f64 {
    let u = pt1.x as f64;
    let v = pt1.y as f64;
    let u_prime = pt2.x as f64;
    let v_prime = pt2.y as f64;

    let f0 = f[0];
    let f1 = f[1];
    let f2 = f[2];
    let f3 = f[3];
    let f4 = f[4];
    let f5 = f[5];
    let f6 = f[6];
    let f7 = f[7];
    let f8 = f[8];

    let e = u_prime * (f0 * u + f1 * v + f2)
        + v_prime * (f3 * u + f4 * v + f5)
        + (f6 * u + f7 * v + f8);

    let fx0 = f0 * u + f1 * v + f2;
    let fx1 = f3 * u + f4 * v + f5;

    let ftx0 = f0 * u_prime + f3 * v_prime + f6;
    let ftx1 = f1 * u_prime + f4 * v_prime + f7;

    let den = fx0 * fx0 + fx1 * fx1 + ftx0 * ftx0 + ftx1 * ftx1;
    if den.abs() > 1e-10 {
        (e * e) / den
    } else {
        0.0
    }
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

/// Robust RANSAC-based estimation of the fundamental matrix.
fn ransac_fundamental(
    pts1: &[Point2f],
    pts2: &[Point2f],
    threshold: f64,
    confidence: f64,
    max_iters: usize,
    mask: Option<&mut Vec<u8>>,
) -> Result<Matrix<f64>> {
    let n = pts1.len();
    let threshold_sq = threshold * threshold;

    let mut rng = Lcg::new(0x9e37_79b9_7f4a_7c15);
    let mut best_count = 0;
    let mut best_inliers = vec![0u8; n];
    let mut best_f: Option<[f64; 9]> = None;

    for _ in 0..max_iters {
        let idx = sample_no_replace(&mut rng, n, 8);
        let sample1: Vec<Point2f> = idx.iter().map(|&i| pts1[i]).collect();
        let sample2: Vec<Point2f> = idx.iter().map(|&i| pts2[i]).collect();

        let f = match solve_8point(&sample1, &sample2) {
            Ok(mat) => mat,
            Err(_) => continue,
        };

        let f_arr: [f64; 9] = match f.data.clone().try_into() {
            Ok(arr) => arr,
            Err(_) => continue,
        };

        let mut count = 0;
        let mut inliers = vec![0u8; n];
        for i in 0..n {
            let err = sampson_distance(pts1[i], pts2[i], &f_arr);
            if err <= threshold_sq {
                inliers[i] = 1;
                count += 1;
            }
        }

        if count > best_count {
            best_count = count;
            best_inliers = inliers;
            best_f = Some(f_arr);
        }

        // Early exit check
        if best_count >= (n as f64 * confidence) as usize {
            break;
        }
    }

    let best_f_arr = match best_f {
        Some(arr) => arr,
        None => {
            return Err(PureCvError::InvalidInput(
                "RANSAC: No valid fundamental matrix model found".to_string(),
            ))
        }
    };

    if best_count < 8 {
        return Err(PureCvError::InvalidInput(
            "RANSAC: Less than 8 inliers found".to_string(),
        ));
    }

    // Refit using all inliers
    let inliers_pts1: Vec<Point2f> = pts1
        .iter()
        .zip(best_inliers.iter())
        .filter_map(|(&p, &m)| if m == 1 { Some(p) } else { None })
        .collect();
    let inliers_pts2: Vec<Point2f> = pts2
        .iter()
        .zip(best_inliers.iter())
        .filter_map(|(&p, &m)| if m == 1 { Some(p) } else { None })
        .collect();

    let refined = if inliers_pts1.len() >= 8 {
        match solve_8point(&inliers_pts1, &inliers_pts2) {
            Ok(mat) => mat,
            Err(_) => Matrix::from_vec(3, 3, 1, best_f_arr.to_vec()),
        }
    } else {
        Matrix::from_vec(3, 3, 1, best_f_arr.to_vec())
    };

    if let Some(m) = mask {
        m.clear();
        m.resize(n, 0);
        let refined_arr: [f64; 9] = refined.data.clone().try_into().unwrap_or(best_f_arr);
        for i in 0..n {
            let err = sampson_distance(pts1[i], pts2[i], &refined_arr);
            if err <= threshold_sq {
                m[i] = 1;
            }
        }
    }

    Ok(refined)
}

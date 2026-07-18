/*
 *  pose.rs
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

//! Camera pose estimation — `solve_pnp` and `solve_pnp_ransac`.
//!
//! Given *N* 3-D object points and their 2-D image projections, and the
//! camera intrinsic matrix, these functions estimate the object pose
//! (rotation and translation vectors) in the camera coordinate system.

use crate::core::error::{PureCvError, Result};
use crate::core::logging::tags;
use crate::core::types::{Point2f, Point3f};
use crate::core::Matrix;
use crate::cv_log_warning;

use super::geometry::rvec_to_rmat;
use super::linalg::{mat3_inv, mat3_mul, nearest_rotation, null_space_vector, Lcg};

// ---------------------------------------------------------------------------
// Public enumerations
// ---------------------------------------------------------------------------

/// Algorithm flag for [`solve_pnp`] / [`solve_pnp_ransac`].
///
/// Mirrors `cv::SolvePnPMethod`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum SolvePnPMethod {
    /// Iterative method based on DLT initialisation followed by
    /// Gauss-Newton reprojection-error minimisation.
    Iterative = 0,
    /// P3P (Gao et al. 2003).  Requires exactly 4 point pairs.
    P3P = 2,
    /// Efficient PnP (Lepetit et al. 2009).
    EPnP = 1,
    /// AP3P (Ke & Roumeliotis 2017).
    AP3P = 5,
    /// SQPnP (Terzakis & Lourakis 2020).
    SqPnP = 6,
}

// ---------------------------------------------------------------------------
// Public API — solve_pnp
// ---------------------------------------------------------------------------

/// Finds the object pose from 3-D / 2-D point correspondences.
///
/// Solves the PnP problem: given `N ≥ 4` 3-D *object points* and their 2-D
/// *image projections*, together with the camera *intrinsic matrix*, computes
/// the rotation vector `rvec` and translation vector `tvec` that describe
/// the object's pose in the camera coordinate system.
///
/// Follows the OpenCV `cv::solvePnP` convention:
/// ```text
/// image_point  =  K · [R | t] · object_point
/// ```
///
/// # Arguments
///
/// * `object_points` – World-space 3-D points (at least 4).
/// * `image_points`  – Corresponding 2-D image points.
/// * `camera_matrix` – 3×3 intrinsic matrix `K = [[fx,0,cx],[0,fy,cy],[0,0,1]]`.
/// * `dist_coeffs`   – Distortion coefficients `[k1,k2,p1,p2[,k3…]]` or `None`.
/// * `rvec`          – Output rotation vector (3×1 `f64`).
/// * `tvec`          – Output translation vector (3×1 `f64`).
/// * `use_extrinsic_guess` – When `true`, the contents of `rvec`/`tvec` are
///   used as an initial estimate (only effective for `Iterative`).
/// * `flags`         – Algorithm selection (see [`SolvePnPMethod`]).
///
/// # Returns
///
/// `true` on success.
///
/// # Errors
///
/// Returns [`PureCvError::InvalidInput`] when fewer than 4 correspondences are
/// given or the camera matrix is singular.
///
/// # Divergences from OpenCV
///
/// | OpenCV | purecv |
/// |--------|--------|
/// | Accepts `Mat` (many layouts) | Accepts `&[Point3f]` / `&[Point2f]` |
/// | Supports distortion undistortion | Only pin-hole (no distortion correction) |
/// | Returns `bool` | Returns `Result<bool>` |
#[allow(clippy::too_many_arguments)]
pub fn solve_pnp(
    object_points: &[Point3f],
    image_points: &[Point2f],
    camera_matrix: &Matrix<f64>,
    dist_coeffs: Option<&[f64]>,
    rvec: &mut Matrix<f64>,
    tvec: &mut Matrix<f64>,
    use_extrinsic_guess: bool,
    flags: SolvePnPMethod,
) -> Result<bool> {
    let n = object_points.len();
    if object_points.len() != image_points.len() {
        return Err(PureCvError::InvalidInput(format!(
            "object_points ({}) and image_points ({}) must have the same length",
            object_points.len(),
            image_points.len()
        )));
    }
    if n < 6 {
        return Err(PureCvError::InvalidInput(
            "solve_pnp requires at least 6 point correspondences for DLT".to_string(),
        ));
    }
    if flags != SolvePnPMethod::Iterative {
        return Err(PureCvError::NotImplemented(
            "Only Iterative method is supported".to_string(),
        ));
    }
    if camera_matrix.rows != 3 || camera_matrix.cols != 3 || camera_matrix.channels != 1 {
        return Err(PureCvError::InvalidInput(
            "camera_matrix must be a 3x3 single-channel matrix".to_string(),
        ));
    }

    let _ = dist_coeffs; // Distortion correction not yet implemented.
    if use_extrinsic_guess {
        return Err(PureCvError::NotImplemented(
            "use_extrinsic_guess is not supported yet".to_string(),
        ));
    }

    // Extract K entries.
    let k = extract_k(camera_matrix)?;

    // Undistort image points (only pin-hole in this release).
    let norm_pts = undistort_points(image_points, &k)?;

    // DLT initial estimate.
    let (r_init, t_init) = dlt_pnp(&norm_pts, object_points)?;

    // Gauss-Newton refinement.
    let (r_ref, t_ref) = if use_extrinsic_guess
        && rvec.rows == 3
        && rvec.cols == 1
        && tvec.rows == 3
        && tvec.cols == 1
    {
        let rv = [rvec.data[0], rvec.data[1], rvec.data[2]];
        let r_guess = rvec_to_rmat(rv[0], rv[1], rv[2]);
        let t_guess = [tvec.data[0], tvec.data[1], tvec.data[2]];
        gauss_newton_refine(&norm_pts, object_points, &r_guess, &t_guess)
    } else {
        gauss_newton_refine(&norm_pts, object_points, &r_init, &t_init)
    };

    // Convert rotation matrix → rotation vector.
    write_output(r_ref, t_ref, rvec, tvec);
    Ok(true)
}

// ---------------------------------------------------------------------------
// Public API — solve_pnp_ransac
// ---------------------------------------------------------------------------

/// Finds the object pose from 3-D / 2-D point correspondences, with RANSAC
/// for robustness against outliers.
///
/// This is the robust counterpart of [`solve_pnp`], applying RANSAC to
/// identify inliers before running a final least-squares fit.
///
/// # Arguments
///
/// * `object_points`        – World-space 3-D points.
/// * `image_points`         – Corresponding 2-D image points.
/// * `camera_matrix`        – 3×3 intrinsic matrix.
/// * `dist_coeffs`          – Distortion coefficients or `None`.
/// * `rvec`                 – Output rotation vector (3×1 `f64`).
/// * `tvec`                 – Output translation vector (3×1 `f64`).
/// * `use_extrinsic_guess`  – Use `rvec`/`tvec` as initial estimate.
/// * `iterations_count`     – Number of RANSAC iterations (default: 100).
/// * `reproj_threshold`     – Maximum reprojection error for an inlier (pixels).
/// * `confidence`           – Desired solution confidence (currently unused;
///   iteration count is fixed).
/// * `inliers`              – When `Some`, filled with the indices of inlier
///   correspondences.
/// * `flags`                – Algorithm selection (see [`SolvePnPMethod`]).
///
/// # Returns
///
/// `true` when a valid pose was found with at least 4 inliers.
///
/// # Errors
///
/// Returns [`PureCvError::InvalidInput`] when fewer than 4 correspondences are
/// given or the camera matrix is invalid.
#[allow(clippy::too_many_arguments)]
pub fn solve_pnp_ransac(
    object_points: &[Point3f],
    image_points: &[Point2f],
    camera_matrix: &Matrix<f64>,
    dist_coeffs: Option<&[f64]>,
    rvec: &mut Matrix<f64>,
    tvec: &mut Matrix<f64>,
    use_extrinsic_guess: bool,
    iterations_count: i32,
    reproj_threshold: f32,
    confidence: f64,
    inliers: Option<&mut Vec<i32>>,
    flags: SolvePnPMethod,
) -> Result<bool> {
    let n = object_points.len();
    if object_points.len() != image_points.len() {
        return Err(PureCvError::InvalidInput(format!(
            "object_points ({}) and image_points ({}) must have the same length",
            object_points.len(),
            image_points.len()
        )));
    }
    if n < 6 {
        return Err(PureCvError::InvalidInput(
            "solve_pnp_ransac requires at least 6 point correspondences".to_string(),
        ));
    }
    if flags != SolvePnPMethod::Iterative {
        return Err(PureCvError::NotImplemented(
            "Only Iterative method is supported".to_string(),
        ));
    }
    if use_extrinsic_guess {
        return Err(PureCvError::NotImplemented(
            "use_extrinsic_guess is not supported in ransac yet".to_string(),
        ));
    }

    let _ = confidence; // Adaptive iteration count not yet implemented.
    let _ = dist_coeffs; // Distortion correction not yet implemented.

    let k = extract_k(camera_matrix)?;
    let norm_pts = undistort_points(image_points, &k)?;

    let max_iters = iterations_count.max(1) as usize;
    let fx = k[0];
    let fy = k[4];
    let f_avg = (fx + fy) / 2.0;
    if f_avg.abs() < 1e-12 {
        return Err(PureCvError::InvalidInput(
            "Invalid focal length in camera_matrix".to_string(),
        ));
    }
    let thr = reproj_threshold as f64 / f_avg;
    let thr2 = thr * thr;

    let mut rng = Lcg::new(0x1234_5678_9abc_def0);
    let mut best_count = 0usize;
    let mut best_inlier_mask = vec![false; n];
    let mut best_r = [1.0f64, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
    let mut best_t = [0.0f64; 3];
    let mut found = false;

    for _ in 0..max_iters {
        let idx = sample_no_replace(&mut rng, n, 6);
        let s_obj: Vec<Point3f> = idx.iter().map(|&i| object_points[i]).collect();
        let s_img: Vec<Point2f> = idx.iter().map(|&i| norm_pts[i]).collect();

        let (r, t) = match dlt_pnp(&s_img, &s_obj) {
            Ok(rt) => rt,
            Err(_) => continue,
        };
        let (r, t) = gauss_newton_refine(&s_img, &s_obj, &r, &t);

        // Count inliers using reprojection error in *normalized* coordinates.
        let (count, mask) = count_pnp_inliers(&norm_pts, object_points, &r, &t, thr2);

        if count > best_count {
            best_count = count;
            best_inlier_mask = mask;
            best_r = r;
            best_t = t;
            found = true;
        }

        if best_count >= n * 9 / 10 {
            break;
        }
    }

    if !found || best_count < 6 {
        cv_log_warning!(
            tags::CALIB3D,
            "solve_pnp_ransac: pose estimation failed because RANSAC consensus inliers ({}) were below the required minimum of 6 points",
            best_count
        );
        return Ok(false);
    }

    // Refit from all inliers.
    let in_obj: Vec<Point3f> = object_points
        .iter()
        .zip(best_inlier_mask.iter())
        .filter_map(|(&p, &ok)| if ok { Some(p) } else { None })
        .collect();
    let in_img: Vec<Point2f> = norm_pts
        .iter()
        .zip(best_inlier_mask.iter())
        .filter_map(|(&p, &ok)| if ok { Some(p) } else { None })
        .collect();

    let (r_final, t_final) = match dlt_pnp(&in_img, &in_obj) {
        Ok(rt) => gauss_newton_refine(&in_img, &in_obj, &rt.0, &rt.1),
        Err(_) => (best_r, best_t),
    };

    write_output(r_final, t_final, rvec, tvec);

    if let Some(out_inliers) = inliers {
        out_inliers.clear();
        for (i, &ok) in best_inlier_mask.iter().enumerate() {
            if ok {
                out_inliers.push(i as i32);
            }
        }
    }

    Ok(true)
}

// ---------------------------------------------------------------------------
// DLT initialisation
// ---------------------------------------------------------------------------

/// DLT solution for the PnP problem using normalised image coordinates.
///
/// For each correspondence `(X, Y, Z) → (u, v)` in normalised coordinates
/// (after K^{-1} multiplication), the linear constraints are:
/// ```text
///   [ X  Y  Z  1  0  0  0  0  -u*X  -u*Y  -u*Z  -u ] * p = 0
///   [ 0  0  0  0  X  Y  Z  1  -v*X  -v*Y  -v*Z  -v ] * p = 0
/// ```
/// The 12-vector `p` vectorises the 3×4 projection matrix `[R|t]`.
fn dlt_pnp(norm_pts: &[Point2f], obj_pts: &[Point3f]) -> Result<([f64; 9], [f64; 3])> {
    let n = norm_pts.len();
    // Build 2n × 12 matrix A.
    let mut a = vec![0.0f64; 2 * n * 12];
    for i in 0..n {
        let xx = obj_pts[i].x as f64;
        let yy = obj_pts[i].y as f64;
        let zz = obj_pts[i].z as f64;
        let u = norm_pts[i].x as f64;
        let v = norm_pts[i].y as f64;

        let base = 2 * i * 12;
        a[base] = xx;
        a[base + 1] = yy;
        a[base + 2] = zz;
        a[base + 3] = 1.0;
        // a[base+4..7] = 0
        a[base + 8] = -u * xx;
        a[base + 9] = -u * yy;
        a[base + 10] = -u * zz;
        a[base + 11] = -u;

        let base = (2 * i + 1) * 12;
        // a[base..3] = 0
        a[base + 4] = xx;
        a[base + 5] = yy;
        a[base + 6] = zz;
        a[base + 7] = 1.0;
        a[base + 8] = -v * xx;
        a[base + 9] = -v * yy;
        a[base + 10] = -v * zz;
        a[base + 11] = -v;
    }

    let p = null_space_vector(&a, 2 * n, 12);

    // Reshape to 3×4 projection matrix.
    let scale = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt();
    if scale < 1e-12 {
        return Err(PureCvError::InternalError(
            "DLT: degenerate null-space".into(),
        ));
    }

    // Extract rotation (rows 0..2 of P) and translation (column 3).
    let r_raw: [f64; 9] = [
        p[0] / scale,
        p[1] / scale,
        p[2] / scale,
        p[4] / scale,
        p[5] / scale,
        p[6] / scale,
        p[8] / scale,
        p[9] / scale,
        p[10] / scale,
    ];
    let t_raw = [p[3] / scale, p[7] / scale, p[11] / scale];

    // Project onto nearest rotation matrix.
    let r = nearest_rotation(&r_raw);

    // Ensure points are in front of the camera (positive z in camera frame).
    let sign = if reproject_sign(&r, &t_raw, obj_pts) >= 0 {
        1.0
    } else {
        -1.0
    };

    let r = if sign < 0.0 {
        let mut rn = r;
        rn.iter_mut().for_each(|v| *v *= -1.0);
        nearest_rotation(&rn)
    } else {
        r
    };
    let t = [sign * t_raw[0], sign * t_raw[1], sign * t_raw[2]];

    Ok((r, t))
}

/// Count how many points project in front of the camera to resolve sign ambiguity.
fn reproject_sign(r: &[f64; 9], t: &[f64; 3], pts: &[Point3f]) -> i32 {
    let mut pos = 0i32;
    for p in pts {
        let z = r[6] * p.x as f64 + r[7] * p.y as f64 + r[8] * p.z as f64 + t[2];
        if z > 0.0 {
            pos += 1;
        } else {
            pos -= 1;
        }
    }
    pos
}

// ---------------------------------------------------------------------------
// Gauss-Newton refinement
// ---------------------------------------------------------------------------

/// Refine pose `(R, t)` by minimizing the sum of squared reprojection errors
/// in normalised image coordinates using the Gauss-Newton method.
fn gauss_newton_refine(
    norm_pts: &[Point2f],
    obj_pts: &[Point3f],
    r_init: &[f64; 9],
    t_init: &[f64; 3],
) -> ([f64; 9], [f64; 3]) {
    // We parameterise the rotation as a Rodriguez vector.
    let mut rv = rmat_to_rvec_approx(r_init);
    let mut t = *t_init;

    const MAX_ITER: usize = 20;
    const EPS: f64 = 1e-8;

    for _ in 0..MAX_ITER {
        let r = rvec_to_rmat(rv[0], rv[1], rv[2]);
        let (jtj, jtb, err2) = build_jtj(&r, &rv, &t, norm_pts, obj_pts);

        if err2 < EPS * EPS {
            break;
        }

        // Solve the 6×6 normal equations ∂(J^T J) δ = J^T b.
        let delta = solve_6x6(&jtj, &jtb);
        if delta.iter().map(|v| v * v).sum::<f64>().sqrt() < EPS {
            break;
        }

        rv[0] += delta[0];
        rv[1] += delta[1];
        rv[2] += delta[2];
        t[0] += delta[3];
        t[1] += delta[4];
        t[2] += delta[5];
    }

    let r_final = rvec_to_rmat(rv[0], rv[1], rv[2]);
    (r_final, t)
}

/// Build J^T J and J^T r for the reprojection-error Gauss-Newton system
/// (6-parameter: 3 Rodrigues + 3 translation).
fn build_jtj(
    r: &[f64; 9],
    rv: &[f64; 3],
    t: &[f64; 3],
    norm_pts: &[Point2f],
    obj_pts: &[Point3f],
) -> ([f64; 36], [f64; 6], f64) {
    let mut jtj = [0.0f64; 36];
    let mut jtb = [0.0f64; 6];
    let mut total_err2 = 0.0f64;
    let theta = (rv[0] * rv[0] + rv[1] * rv[1] + rv[2] * rv[2]).sqrt();

    for i in 0..norm_pts.len() {
        let xx = obj_pts[i].x as f64;
        let yy = obj_pts[i].y as f64;
        let zz = obj_pts[i].z as f64;

        // Camera-space point.
        let cx = r[0] * xx + r[1] * yy + r[2] * zz + t[0];
        let cy = r[3] * xx + r[4] * yy + r[5] * zz + t[1];
        let cz = r[6] * xx + r[7] * yy + r[8] * zz + t[2];

        if cz.abs() < 1e-12 {
            continue;
        }

        let inv_cz = 1.0 / cz;
        let proj_u = cx * inv_cz;
        let proj_v = cy * inv_cz;

        let eu = proj_u - norm_pts[i].x as f64;
        let ev = proj_v - norm_pts[i].y as f64;
        total_err2 += eu * eu + ev * ev;

        // Jacobian of (proj_u, proj_v) w.r.t. (rx, ry, rz, tx, ty, tz).
        // d(proj) / d(camera_point):
        let dpdu_dcx = inv_cz;
        let dpdu_dcz = -cx * inv_cz * inv_cz;
        let dpdv_dcy = inv_cz;
        let dpdv_dcz = -cy * inv_cz * inv_cz;

        // d(camera_point) / d(rvec) via Rodrigues derivative (approximate).
        // We use a finite-difference approximation for the rotation Jacobian.
        let eps = if theta > 1e-4 { theta * 1e-5 } else { 1e-6 };
        let mut j = [0.0f64; 12]; // 2 residuals × 6 params

        for k in 0..3 {
            let mut rv_p = *rv;
            rv_p[k] += eps;
            let r_p = rvec_to_rmat(rv_p[0], rv_p[1], rv_p[2]);
            let cx_p = r_p[0] * xx + r_p[1] * yy + r_p[2] * zz + t[0];
            let cy_p = r_p[3] * xx + r_p[4] * yy + r_p[5] * zz + t[1];
            let cz_p = r_p[6] * xx + r_p[7] * yy + r_p[8] * zz + t[2];
            let inv_czp = if cz_p.abs() > 1e-12 { 1.0 / cz_p } else { 0.0 };
            j[k] = (cx_p * inv_czp - proj_u) / eps;
            j[6 + k] = (cy_p * inv_czp - proj_v) / eps;
        }

        // Translation Jacobian (exact).
        j[3] = dpdu_dcx;
        j[4] = 0.0;
        j[5] = dpdu_dcz;
        j[9] = 0.0;
        j[10] = dpdv_dcy;
        j[11] = dpdv_dcz;

        // Accumulate J^T J and J^T r.
        for a in 0..6 {
            jtb[a] += j[a] * eu + j[6 + a] * ev;
            for b in a..6 {
                let v = j[a] * j[b] + j[6 + a] * j[6 + b];
                jtj[a * 6 + b] += v;
                if b != a {
                    jtj[b * 6 + a] += v;
                }
            }
        }
    }

    (jtj, jtb, total_err2)
}

/// Solve 6×6 symmetric positive-(semi-)definite system via Cholesky decomposition.
fn solve_6x6(a: &[f64; 36], b: &[f64; 6]) -> [f64; 6] {
    // Simple Gaussian elimination with partial pivoting.
    const N: usize = 6;
    let mut m = [0.0f64; N * N];
    let mut rhs = [0.0f64; N];
    for i in 0..N {
        for j in 0..N {
            m[i * N + j] = a[i * N + j];
        }
        rhs[i] = -b[i]; // We want to minimise: ½‖Jδ − r‖², so δ = −(J^TJ)^{-1} J^Tr.
    }

    let mut perm: [usize; N] = [0, 1, 2, 3, 4, 5];
    for col in 0..N {
        // Find pivot.
        let mut max_val = m[col * N + col].abs();
        let mut max_row = col;
        for row in (col + 1)..N {
            let v = m[row * N + col].abs();
            if v > max_val {
                max_val = v;
                max_row = row;
            }
        }
        if max_val < 1e-14 {
            continue; // Degenerate; skip.
        }
        if max_row != col {
            for j in 0..N {
                m.swap(col * N + j, max_row * N + j);
            }
            rhs.swap(col, max_row);
            perm.swap(col, max_row);
        }
        let pivot = m[col * N + col];
        for row in (col + 1)..N {
            let factor = m[row * N + col] / pivot;
            for j in col..N {
                let v = m[col * N + j];
                m[row * N + j] -= factor * v;
            }
            rhs[row] -= factor * rhs[col];
        }
    }

    // Back substitution.
    let mut x = [0.0f64; N];
    for i in (0..N).rev() {
        let mut s = rhs[i];
        for j in (i + 1)..N {
            s -= m[i * N + j] * x[j];
        }
        x[i] = if m[i * N + i].abs() > 1e-14 {
            s / m[i * N + i]
        } else {
            0.0
        };
    }
    x
}

// ---------------------------------------------------------------------------
// RANSAC helpers
// ---------------------------------------------------------------------------

fn count_pnp_inliers(
    norm_pts: &[Point2f],
    obj_pts: &[Point3f],
    r: &[f64; 9],
    t: &[f64; 3],
    thr2: f64,
) -> (usize, Vec<bool>) {
    let n = norm_pts.len();
    let mut mask = vec![false; n];
    let mut count = 0usize;

    for i in 0..n {
        let xx = obj_pts[i].x as f64;
        let yy = obj_pts[i].y as f64;
        let zz = obj_pts[i].z as f64;

        let cx = r[0] * xx + r[1] * yy + r[2] * zz + t[0];
        let cy = r[3] * xx + r[4] * yy + r[5] * zz + t[1];
        let cz = r[6] * xx + r[7] * yy + r[8] * zz + t[2];

        if cz.abs() < 1e-12 {
            continue;
        }
        let inv_cz = 1.0 / cz;
        let eu = cx * inv_cz - norm_pts[i].x as f64;
        let ev = cy * inv_cz - norm_pts[i].y as f64;

        if eu * eu + ev * ev <= thr2 {
            mask[i] = true;
            count += 1;
        }
    }
    (count, mask)
}

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

// ---------------------------------------------------------------------------
// Misc helpers
// ---------------------------------------------------------------------------

/// Extract the 3×3 camera matrix as a flat array and its inverse.
fn extract_k(camera_matrix: &Matrix<f64>) -> Result<[f64; 9]> {
    if camera_matrix.data.len() != 9 {
        return Err(PureCvError::InvalidInput(
            "camera_matrix must contain exactly 9 elements".into(),
        ));
    }
    let k: [f64; 9] = camera_matrix
        .data
        .as_slice()
        .try_into()
        .map_err(|_| PureCvError::InternalError("camera_matrix layout error".into()))?;
    Ok(k)
}

/// Apply K^{-1} to image points to obtain normalised camera coordinates.
fn undistort_points(image_points: &[Point2f], k: &[f64; 9]) -> Result<Vec<Point2f>> {
    let ki = mat3_inv(k)
        .ok_or_else(|| PureCvError::InvalidInput("camera_matrix is singular".to_string()))?;

    Ok(image_points
        .iter()
        .map(|p| {
            let x = p.x as f64;
            let y = p.y as f64;
            let w = ki[6] * x + ki[7] * y + ki[8];
            let xn = if w.abs() > 1e-12 {
                (ki[0] * x + ki[1] * y + ki[2]) / w
            } else {
                0.0
            };
            let yn = if w.abs() > 1e-12 {
                (ki[3] * x + ki[4] * y + ki[5]) / w
            } else {
                0.0
            };
            Point2f {
                x: xn as f32,
                y: yn as f32,
            }
        })
        .collect())
}

/// Convert rotation matrix → Rodrigues vector for use as a Gauss-Newton
/// initialiser.  Returns the same result as `geometry::rmat_to_rvec`, but
/// inlined here to avoid a cross-module call in the hot refinement loop.
fn rmat_to_rvec_approx(r: &[f64; 9]) -> [f64; 3] {
    // Use the same formula as geometry::rmat_to_rvec.
    let trace_val = ((r[0] + r[4] + r[8] - 1.0) * 0.5).clamp(-1.0, 1.0);
    let theta = trace_val.acos();
    if theta.abs() < 1e-10 {
        return [0.0, 0.0, 0.0];
    }
    let factor = theta / (2.0 * theta.sin());
    [
        factor * (r[7] - r[5]),
        factor * (r[2] - r[6]),
        factor * (r[3] - r[1]),
    ]
}

/// Write rotation + translation to output matrices.
fn write_output(r: [f64; 9], t: [f64; 3], rvec: &mut Matrix<f64>, tvec: &mut Matrix<f64>) {
    // Rotation matrix → rotation vector.
    let rv = rmat_to_rvec_approx(&r);
    rvec.rows = 3;
    rvec.cols = 1;
    rvec.channels = 1;
    rvec.data = rv.to_vec();

    tvec.rows = 3;
    tvec.cols = 1;
    tvec.channels = 1;
    tvec.data = t.to_vec();
}

/// Matrix-multiply two 3×3 matrices; exposed for tests.
#[allow(dead_code)]
pub(super) fn mat3_mul_pub(a: &[f64; 9], b: &[f64; 9]) -> [f64; 9] {
    mat3_mul(a, b)
}

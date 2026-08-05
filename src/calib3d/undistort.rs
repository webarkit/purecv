/*
 *  undistort.rs
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

use alloc::string::ToString;
#[allow(unused_imports)]
use num_traits::Float;

use crate::core::error::{PureCvError, Result};
use crate::core::types::Size2i;
use crate::core::Matrix;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// Computes the undistortion and rectification transformation map.
///
/// The function computes the joint undistortion and rectification transformation map
/// and stores the result in `map1` and `map2`.
///
/// # Arguments
///
/// * `camera_matrix` - Input camera matrix (3x3).
/// * `dist_coeffs` - Input vector of distortion coefficients (k1, k2, p1, p2, k3, k4, k5, k6).
/// * `r` - Optional rectification transformation in object space (3x3 matrix). If `None`, identity is used.
/// * `new_camera_matrix` - New camera matrix (3x3).
/// * `size` - Undistorted image size.
///
/// # Returns
///
/// Returns a `Result<(Matrix<f32>, Matrix<f32>)>` containing the computed maps `map1` and `map2`.
///
/// # Errors
///
/// Returns an error if:
/// * `camera_matrix`, `new_camera_matrix`, or `r` (if provided) are not 3x3 single-channel matrices.
///
/// # Examples
///
/// ```
/// use purecv::core::{Matrix, types::Size2i};
/// use purecv::calib3d::undistort::init_undistort_rectify_map;
///
/// let mut camera_matrix = Matrix::<f64>::new(3, 3, 1);
/// camera_matrix.data[0] = 800.0; // fx
/// camera_matrix.data[4] = 800.0; // fy
/// camera_matrix.data[2] = 320.0; // cx
/// camera_matrix.data[5] = 240.0; // cy
/// camera_matrix.data[8] = 1.0;
///
/// let dist_coeffs = Matrix::<f64>::new(1, 5, 1); // e.g., k1, k2, p1, p2, k3
/// let size = Size2i::new(640, 480);
///
/// let (map1, map2) = init_undistort_rectify_map(
///     &camera_matrix,
///     &dist_coeffs,
///     None,
///     &camera_matrix,
///     size
/// ).unwrap();
/// ```
pub fn init_undistort_rectify_map(
    camera_matrix: &Matrix<f64>,
    dist_coeffs: &Matrix<f64>,
    r: Option<&Matrix<f64>>,
    new_camera_matrix: &Matrix<f64>,
    size: Size2i,
) -> Result<(Matrix<f32>, Matrix<f32>)> {
    if camera_matrix.rows != 3 || camera_matrix.cols != 3 || camera_matrix.channels != 1 {
        return Err(PureCvError::InvalidDimensions(
            "camera_matrix must be 3x3 single-channel".to_string(),
        ));
    }
    if new_camera_matrix.rows != 3 || new_camera_matrix.cols != 3 || new_camera_matrix.channels != 1
    {
        return Err(PureCvError::InvalidDimensions(
            "new_camera_matrix must be 3x3 single-channel".to_string(),
        ));
    }
    if let Some(r_mat) = r {
        if r_mat.rows != 3 || r_mat.cols != 3 || r_mat.channels != 1 {
            return Err(PureCvError::InvalidDimensions(
                "R must be 3x3 single-channel".to_string(),
            ));
        }
    }

    let fx = camera_matrix.data[0];
    let cx = camera_matrix.data[2];
    let fy = camera_matrix.data[4];
    let cy = camera_matrix.data[5];

    let fx_prime = new_camera_matrix.data[0];
    let cx_prime = new_camera_matrix.data[2];
    let fy_prime = new_camera_matrix.data[4];
    let cy_prime = new_camera_matrix.data[5];

    let mut k1 = 0.0;
    let mut k2 = 0.0;
    let mut p1 = 0.0;
    let mut p2 = 0.0;
    let mut k3 = 0.0;
    let mut k4 = 0.0;
    let mut k5 = 0.0;
    let mut k6 = 0.0;

    let len = dist_coeffs.data.len();
    if len >= 1 {
        k1 = dist_coeffs.data[0];
    }
    if len >= 2 {
        k2 = dist_coeffs.data[1];
    }
    if len >= 3 {
        p1 = dist_coeffs.data[2];
    }
    if len >= 4 {
        p2 = dist_coeffs.data[3];
    }
    if len >= 5 {
        k3 = dist_coeffs.data[4];
    }
    if len >= 6 {
        k4 = dist_coeffs.data[5];
    }
    if len >= 7 {
        k5 = dist_coeffs.data[6];
    }
    if len >= 8 {
        k6 = dist_coeffs.data[7];
    }

    let (r00, r01, r02, r10, r11, r12, r20, r21, r22) = match r {
        Some(r_mat) => (
            r_mat.data[0],
            r_mat.data[3],
            r_mat.data[6], // Row 0 of transpose (Col 0 of R)
            r_mat.data[1],
            r_mat.data[4],
            r_mat.data[7], // Row 1 of transpose (Col 1 of R)
            r_mat.data[2],
            r_mat.data[5],
            r_mat.data[8], // Row 2 of transpose (Col 2 of R)
        ),
        None => (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
    };

    let mut map1 = Matrix::<f32>::new(size.height as usize, size.width as usize, 1);
    let mut map2 = Matrix::<f32>::new(size.height as usize, size.width as usize, 1);

    #[cfg(feature = "parallel")]
    {
        map1.data
            .par_chunks_exact_mut(size.width as usize)
            .zip(map2.data.par_chunks_exact_mut(size.width as usize))
            .enumerate()
            .for_each(|(v, (map1_row, map2_row))| {
                let y_val = v as f64;
                for u in 0..size.width as usize {
                    let x_val = u as f64;
                    let x = (x_val - cx_prime) / fx_prime;
                    let y = (y_val - cy_prime) / fy_prime;

                    let rx = r00 * x + r01 * y + r02;
                    let ry = r10 * x + r11 * y + r12;
                    let rz = r20 * x + r21 * y + r22;

                    let (xp, yp) = if rz.abs() > 1e-10 {
                        (rx / rz, ry / rz)
                    } else {
                        (x, y)
                    };

                    let r2 = xp * xp + yp * yp;
                    let radial_num = 1.0 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2;
                    let radial_den = 1.0 + k4 * r2 + k5 * r2 * r2 + k6 * r2 * r2 * r2;
                    let radial = if radial_den.abs() > 1e-10 {
                        radial_num / radial_den
                    } else {
                        radial_num
                    };

                    let dx = 2.0 * p1 * xp * yp + p2 * (r2 + 2.0 * xp * xp);
                    let dy = p1 * (r2 + 2.0 * yp * yp) + 2.0 * p2 * xp * yp;

                    let x_dist = xp * radial + dx;
                    let y_dist = yp * radial + dy;

                    map1_row[u] = (fx * x_dist + cx) as f32;
                    map2_row[u] = (fy * y_dist + cy) as f32;
                }
            });
    }

    #[cfg(not(feature = "parallel"))]
    {
        let width = size.width as usize;
        let height = size.height as usize;
        for v in 0..height {
            let map1_row = &mut map1.data[v * width..(v + 1) * width];
            let map2_row = &mut map2.data[v * width..(v + 1) * width];
            let y_val = v as f64;
            for u in 0..width {
                let x_val = u as f64;
                let x = (x_val - cx_prime) / fx_prime;
                let y = (y_val - cy_prime) / fy_prime;

                let rx = r00 * x + r01 * y + r02;
                let ry = r10 * x + r11 * y + r12;
                let rz = r20 * x + r21 * y + r22;

                let (xp, yp) = if rz.abs() > 1e-10 {
                    (rx / rz, ry / rz)
                } else {
                    (x, y)
                };

                let r2 = xp * xp + yp * yp;
                let radial_num = 1.0 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2;
                let radial_den = 1.0 + k4 * r2 + k5 * r2 * r2 + k6 * r2 * r2 * r2;
                let radial = if radial_den.abs() > 1e-10 {
                    radial_num / radial_den
                } else {
                    radial_num
                };

                let dx = 2.0 * p1 * xp * yp + p2 * (r2 + 2.0 * xp * xp);
                let dy = p1 * (r2 + 2.0 * yp * yp) + 2.0 * p2 * xp * yp;

                let x_dist = xp * radial + dx;
                let y_dist = yp * radial + dy;

                map1_row[u] = (fx * x_dist + cx) as f32;
                map2_row[u] = (fy * y_dist + cy) as f32;
            }
        }
    }

    Ok((map1, map2))
}

/*
 *  metrics.rs
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

use crate::core::error::{PureCvError, Result};
use crate::core::matrix::Matrix;

/// Computes the Peak Signal-to-Noise Ratio (PSNR) between two matrices.
///
/// The PSNR is essentially an evaluation metric used to measure the ratio between the
/// maximum possible power of a signal and the power of corrupting noise that affects
/// the fidelity of its representation. It’s frequently used to measure the quality of
/// reconstruction of lossy compression codecs (e.g., for image compression).
///
/// In PureCV, it calculates: `10 * log10((255 ^ 2) / MSE)`.
/// This implementation relies on iterating over the arrays to calculate the Mean Squared Error (MSE),
/// automatically adapting correctly given internal parameter precision via ` Into<f64> `.
///
/// # Arguments
///
/// * `img1` - The first matrix (e.g. original image buffer), which must match dimensions.
/// * `img2` - The second matrix (e.g. reconstructed test image buffer) dimension-matching.
///
/// # Returns
///
/// * `Result<f64>` - The calculated PSNR value. Extremely high (or technically `f64::INFINITY`) if exactly the same.
///
/// # Errors
/// Returns `PureCvError::InvalidDimensions` if matrix dimensionalities do not directly match.
pub fn psnr<T>(img1: &Matrix<T>, img2: &Matrix<T>) -> Result<f64>
where
    T: Copy + Into<f64>,
{
    if img1.rows != img2.rows || img1.cols != img2.cols || img1.channels != img2.channels {
        return Err(PureCvError::InvalidDimensions(
            "Images must have same dimensions for PSNR".into(),
        ));
    }

    let mut mse = 0.0;
    for (v1, v2) in img1.data.iter().zip(img2.data.iter()) {
        let diff = (*v1).into() - (*v2).into();
        mse += diff * diff;
    }

    if mse == 0.0 {
        // Perfect match
        return Ok(f64::INFINITY);
    }

    mse /= (img1.rows * img1.cols * img1.channels) as f64;
    // Assuming 8-bit depth as maximum signal. If using f32/f64, max might be 1.0.
    // For general purpose, we might need a `max_val` parameter, assuming 255.0 for now.
    let max_val = 255.0;
    let psnr_val = 10.0 * (max_val * max_val / mse).log10();
    Ok(psnr_val)
}

/// Computes the Mahalanobis distance between two vectors.
///
/// The Mahalanobis distance is a multi-dimensional distance measure that evaluates how distant
/// a point is from a distribution. It is often useful in robust statistical settings compared to
/// checking simple Euclidean distances.
/// For exact tracking against `cv::Mahalanobis()`, this expects matching 1-Dimensional column inputs.
///
/// # Arguments
///
/// * `src1` - Given multidimensional column-vector.
/// * `src2` - Given multidimensional column-vector compared to evaluate.
/// * `covar_inv` - Represents the Inverse Covariance matrix, size matched equally to row elements natively.
///
/// # Returns
///
/// * `Result<f64>` - Statistical spatial distance, correctly rooted.
///
/// # Errors
/// * Returns `PureCvError::InvalidDimensions` if the vectors are not 1-Column, fail to pair uniformly or if inverse covariance dimensions lack matching square symmetry.
pub fn mahalanobis<T>(src1: &Matrix<T>, src2: &Matrix<T>, covar_inv: &Matrix<T>) -> Result<f64>
where
    T: Copy + Into<f64>,
{
    // Ensure vectors are column vectors of the same size.
    if src1.rows != src2.rows || src1.cols != 1 || src2.cols != 1 {
        return Err(PureCvError::InvalidDimensions(
            "src1 and src2 must be column vectors of the same size".into(),
        ));
    }

    // Ensure covariance matrix is square and matches vector size
    if covar_inv.rows != covar_inv.cols || covar_inv.rows != src1.rows {
        return Err(PureCvError::InvalidDimensions(
            "covar_inv must be a square matrix matching vector size".into(),
        ));
    }

    let n = src1.rows;
    let mut diff = vec![0.0; n];
    for (i, d) in diff.iter_mut().enumerate() {
        *d = src1.data[i].into() - src2.data[i].into();
    }

    let mut dist_sq = 0.0;
    for (i, &d_i) in diff.iter().enumerate() {
        let mut row_sum = 0.0;
        for (j, &d_j) in diff.iter().enumerate() {
            row_sum += d_j * covar_inv.data[i * n + j].into();
        }
        dist_sq += d_i * row_sum;
    }

    Ok(dist_sq.sqrt())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_psnr() {
        let a = Matrix::from_vec(1, 4, 1, vec![10u8, 20, 30, 40]);
        let b = Matrix::from_vec(1, 4, 1, vec![12u8, 18, 30, 45]);
        let p = psnr(&a, &b).unwrap();
        assert!(p > 0.0);
    }

    #[test]
    fn test_mahalanobis() {
        let a = Matrix::from_vec(2, 1, 1, vec![1.0f32, 2.0]);
        let b = Matrix::from_vec(2, 1, 1, vec![2.0f32, 4.0]);
        let cv = Matrix::from_vec(2, 2, 1, vec![1.0f32, 0.0, 0.0, 1.0]);
        let m = mahalanobis(&a, &b, &cv).unwrap();
        // Since cov is identity, Mahalanobis should be Euclidean distance.
        // sqrt( (1-2)^2 + (2-4)^2 ) = sqrt(1 + 4) = sqrt(5) ≈ 2.236
        assert!((m - 2.23606797749979).abs() < 1e-5);
    }
}

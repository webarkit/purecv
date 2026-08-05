/*
 *  linalg.rs
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

//! Internal linear algebra helpers for the `calib3d` module.
//!
//! All functions operate on flat, row-major `f64` slices/arrays.

use alloc::{vec, vec::Vec};
#[allow(unused_imports)]
use num_traits::Float;

// ---------------------------------------------------------------------------
// Matrix utilities
// ---------------------------------------------------------------------------

/// Compute A^T * A for a (`rows` × `cols`) matrix `a`, returning a symmetric
/// (`cols` × `cols`) matrix stored row-major.
pub(super) fn mat_ata(a: &[f64], rows: usize, cols: usize) -> Vec<f64> {
    let mut ata = vec![0.0f64; cols * cols];
    for i in 0..cols {
        for j in i..cols {
            let mut s = 0.0;
            for k in 0..rows {
                s += a[k * cols + i] * a[k * cols + j];
            }
            ata[i * cols + j] = s;
            ata[j * cols + i] = s;
        }
    }
    ata
}

/// Multiply 3×3 matrices `a` and `b` (row-major), returning their product.
pub(super) fn mat3_mul(a: &[f64; 9], b: &[f64; 9]) -> [f64; 9] {
    let mut c = [0.0f64; 9];
    for i in 0..3 {
        for j in 0..3 {
            let mut s = 0.0;
            for k in 0..3 {
                s += a[i * 3 + k] * b[k * 3 + j];
            }
            c[i * 3 + j] = s;
        }
    }
    c
}

/// Determinant of a 3×3 row-major matrix.
pub(super) fn mat3_det(m: &[f64; 9]) -> f64 {
    m[0] * (m[4] * m[8] - m[5] * m[7]) - m[1] * (m[3] * m[8] - m[5] * m[6])
        + m[2] * (m[3] * m[7] - m[4] * m[6])
}

/// Inverse of a 3×3 row-major matrix. Returns `None` if the matrix is singular.
pub(super) fn mat3_inv(m: &[f64; 9]) -> Option<[f64; 9]> {
    let det = mat3_det(m);
    if det.abs() < 1e-12 {
        return None;
    }
    let d = 1.0 / det;
    Some([
        d * (m[4] * m[8] - m[5] * m[7]),
        d * (m[2] * m[7] - m[1] * m[8]),
        d * (m[1] * m[5] - m[2] * m[4]),
        d * (m[5] * m[6] - m[3] * m[8]),
        d * (m[0] * m[8] - m[2] * m[6]),
        d * (m[2] * m[3] - m[0] * m[5]),
        d * (m[3] * m[7] - m[4] * m[6]),
        d * (m[1] * m[6] - m[0] * m[7]),
        d * (m[0] * m[4] - m[1] * m[3]),
    ])
}

// ---------------------------------------------------------------------------
// Jacobi symmetric eigenvalue decomposition
// ---------------------------------------------------------------------------

/// Jacobi eigenvalue decomposition for a real symmetric `n × n` matrix.
///
/// On entry `a` is the symmetric matrix (row-major flat).  On return:
/// - The diagonal of `a` holds the eigenvalues.
/// - `v` (initialized to identity before the call) accumulates the
///   orthogonal matrix whose *columns* are the corresponding eigenvectors.
///
/// The algorithm performs cyclic sweeps (all off-diagonal (p,q) pairs)
/// until convergence or `MAX_SWEEPS` is reached.
pub(super) fn jacobi_eigen(a: &mut [f64], n: usize, v: &mut [f64]) {
    // Initialize v as the identity matrix.
    for i in 0..n {
        for j in 0..n {
            v[i * n + j] = if i == j { 1.0 } else { 0.0 };
        }
    }

    const MAX_SWEEPS: usize = 100;
    for _ in 0..MAX_SWEEPS {
        // Convergence check: total off-diagonal norm.
        let mut off = 0.0f64;
        for i in 0..n {
            for j in (i + 1)..n {
                off += a[i * n + j].abs();
            }
        }
        if off < 1e-15 * (n as f64) {
            break;
        }

        // One sweep over all (p, q) pairs with p < q.
        for p in 0..n {
            for q in (p + 1)..n {
                let apq = a[p * n + q];
                if apq.abs() < 1e-15 {
                    continue;
                }
                let app = a[p * n + p];
                let aqq = a[q * n + q];

                // Compute Jacobi rotation parameters (Numerical Recipes convention).
                let theta = 0.5 * (aqq - app) / apq;
                let t = if theta >= 0.0 {
                    1.0 / (theta + (1.0 + theta * theta).sqrt())
                } else {
                    1.0 / (theta - (1.0 + theta * theta).sqrt())
                };
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = t * c;

                // Update diagonal elements.
                a[p * n + p] = app - t * apq;
                a[q * n + q] = aqq + t * apq;
                a[p * n + q] = 0.0;
                a[q * n + p] = 0.0;

                // Update off-diagonal elements for rows/columns r ≠ p, q.
                for r in 0..n {
                    if r == p || r == q {
                        continue;
                    }
                    let arp = a[r * n + p];
                    let arq = a[r * n + q];
                    let new_arp = c * arp - s * arq;
                    let new_arq = s * arp + c * arq;
                    a[r * n + p] = new_arp;
                    a[p * n + r] = new_arp;
                    a[r * n + q] = new_arq;
                    a[q * n + r] = new_arq;
                }

                // Accumulate the rotation into v (eigenvectors as columns).
                for r in 0..n {
                    let vrp = v[r * n + p];
                    let vrq = v[r * n + q];
                    v[r * n + p] = c * vrp - s * vrq;
                    v[r * n + q] = s * vrp + c * vrq;
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Null-space vector
// ---------------------------------------------------------------------------

/// Find the right singular vector of `a` (`rows` × `cols`) corresponding to
/// the *smallest* singular value, i.e. the eigenvector of `A^T A` with the
/// smallest eigenvalue.
pub(super) fn null_space_vector(a: &[f64], rows: usize, cols: usize) -> Vec<f64> {
    let mut ata = mat_ata(a, rows, cols);
    let mut v = vec![0.0f64; cols * cols];
    jacobi_eigen(&mut ata, cols, &mut v);

    // Diagonal of ata now holds eigenvalues; locate the minimum.
    let min_idx = (0..cols)
        .min_by(|&i, &j| {
            ata[i * cols + i]
                .partial_cmp(&ata[j * cols + j])
                .unwrap_or(core::cmp::Ordering::Equal)
        })
        .unwrap_or(cols - 1);

    // Return column `min_idx` of `v` (the corresponding eigenvector).
    (0..cols).map(|i| v[i * cols + min_idx]).collect()
}

// ---------------------------------------------------------------------------
// 3×3 SVD and nearest rotation
// ---------------------------------------------------------------------------

/// Singular value decomposition of a 3×3 matrix: `a = U * diag(sigma) * Vt`.
///
/// Returns `(U, sigma, Vt)` with singular values sorted in *descending* order.
pub(super) fn svd_3x3(a: &[f64; 9]) -> ([f64; 9], [f64; 3], [f64; 9]) {
    // Compute A^T A (symmetric 3×3).
    let mut ata = [0.0f64; 9];
    for i in 0..3 {
        for j in 0..3 {
            let mut s = 0.0;
            for k in 0..3 {
                s += a[k * 3 + i] * a[k * 3 + j];
            }
            ata[i * 3 + j] = s;
        }
    }

    // Eigendecomposition of ATA to get right singular vectors V.
    let mut v = [0.0f64; 9];
    jacobi_eigen(&mut ata, 3, &mut v);

    // Sort by descending eigenvalue.
    let mut idx = [0usize, 1, 2];
    idx.sort_by(|&i, &j| {
        ata[j * 3 + j]
            .partial_cmp(&ata[i * 3 + i])
            .unwrap_or(core::cmp::Ordering::Equal)
    });

    // Reorder V columns (right singular vectors).
    let mut v_sorted = [0.0f64; 9];
    for (new_col, &old_col) in idx.iter().enumerate() {
        for row in 0..3 {
            v_sorted[row * 3 + new_col] = v[row * 3 + old_col];
        }
    }

    // Singular values = sqrt(eigenvalues), clamped to zero.
    let sigma = [
        ata[idx[0] * 3 + idx[0]].max(0.0).sqrt(),
        ata[idx[1] * 3 + idx[1]].max(0.0).sqrt(),
        ata[idx[2] * 3 + idx[2]].max(0.0).sqrt(),
    ];

    // Compute U = A * V_sorted * diag(1/sigma_i).
    let mut u = [0.0f64; 9];
    for col in 0..3 {
        let sv = sigma[col];
        if sv > 1e-12 {
            for row in 0..3 {
                let sum: f64 = (0..3).map(|k| a[row * 3 + k] * v_sorted[k * 3 + col]).sum();
                u[row * 3 + col] = sum / sv;
            }
        } else if col == 2 && sigma[0] > 1e-12 && sigma[1] > 1e-12 {
            // Complete U to a proper orthonormal basis via cross product of columns 0 and 1.
            // Column 0: u[0], u[3], u[6]; column 1: u[1], u[4], u[7]; result → column 2.
            u[2] = u[3] * u[7] - u[6] * u[4];
            u[5] = u[6] * u[1] - u[0] * u[7];
            u[8] = u[0] * u[4] - u[3] * u[1];
        }
    }

    // Vt = V_sorted^T.
    let mut vt = [0.0f64; 9];
    for i in 0..3 {
        for j in 0..3 {
            vt[i * 3 + j] = v_sorted[j * 3 + i];
        }
    }

    (u, sigma, vt)
}

/// Project the 3×3 matrix `m` onto the nearest rotation matrix (det = +1)
/// using SVD: `R = U * diag(1, 1, det(U * Vt)) * Vt`.
pub(super) fn nearest_rotation(m: &[f64; 9]) -> [f64; 9] {
    let (u, _sigma, vt) = svd_3x3(m);
    let det_u = mat3_det(&u);
    let det_vt = mat3_det(&vt);
    let sign = (det_u * det_vt).signum();

    // R = U * diag(1, 1, sign) * Vt.
    let mut r = [0.0f64; 9];
    for i in 0..3 {
        for j in 0..3 {
            let s: f64 = (0..2).map(|k| u[i * 3 + k] * vt[k * 3 + j]).sum::<f64>()
                + sign * u[i * 3 + 2] * vt[2 * 3 + j];
            r[i * 3 + j] = s;
        }
    }
    r
}

// ---------------------------------------------------------------------------
// Simple PRNG for RANSAC sampling
// ---------------------------------------------------------------------------

/// Minimal LCG pseudo-random number generator for RANSAC sampling.
pub(super) struct Lcg {
    state: u64,
}

impl Lcg {
    /// Create a new generator from `seed`.
    ///
    /// The LSB of `seed` is forced to 1 to guarantee the state is odd,
    /// which improves the statistical quality of the LCG output.
    pub(super) fn new(seed: u64) -> Self {
        Self { state: seed | 1 }
    }

    /// Return a pseudo-random `usize` in `[0, n)`.
    pub(super) fn next_usize(&mut self, n: usize) -> usize {
        // Knuth's multiplicative LCG constants.
        self.state = self
            .state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((self.state >> 33) as usize) % n
    }
}

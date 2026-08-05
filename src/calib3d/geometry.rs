/*
 *  geometry.rs
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

//! Rodrigues rotation — `rodrigues`.
//!
//! Converts between a *rotation vector* (compact axis-angle representation,
//! sometimes called an *Rodrigues vector*) and a 3×3 *rotation matrix*.

use alloc::{string::ToString, vec};
#[allow(unused_imports)]
use num_traits::Float;

use crate::core::error::{PureCvError, Result};
use crate::core::Matrix;

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Converts a rotation matrix to a rotation vector, or vice-versa.
///
/// The function behaves identically to `cv::Rodrigues`:
///
/// | `src` shape | `dst` shape | Operation |
/// |-------------|-------------|-----------|
/// | 3×1 or 1×3  | 3×3         | rotation vector → rotation matrix |
/// | 3×3         | 3×1         | rotation matrix → rotation vector |
///
/// The rotation vector `r` encodes the axis of rotation `r/‖r‖` and the
/// angle of rotation `θ = ‖r‖` (in radians) following the right-hand rule.
///
/// # Errors
///
/// Returns [`PureCvError::InvalidInput`] when `src` has an unsupported shape.
///
/// # References
///
/// Rodrigues, O. (1840). *Des lois géométriques qui régissent les déplacements
/// d'un système solide dans l'espace* — Rodrigues' rotation formula.
pub fn rodrigues(src: &Matrix<f64>, dst: &mut Matrix<f64>) -> Result<()> {
    let elem = src.rows * src.cols * src.channels;

    if src.channels != 1 {
        return Err(PureCvError::InvalidInput(
            "rodrigues: src must be single-channel".to_string(),
        ));
    }

    match (src.rows, src.cols, elem) {
        // Rotation vector (3×1 or 1×3) → rotation matrix (3×3)
        (3, 1, 3) | (1, 3, 3) => {
            let rx = src.data[0];
            let ry = src.data[1];
            let rz = src.data[2];
            let r = rvec_to_rmat(rx, ry, rz);
            dst.rows = 3;
            dst.cols = 3;
            dst.channels = 1;
            dst.data = r.to_vec();
            Ok(())
        }
        // Rotation matrix (3×3) → rotation vector (3×1)
        (3, 3, 9) => {
            let m: [f64; 9] = src
                .data
                .as_slice()
                .try_into()
                .map_err(|_| PureCvError::InternalError("unexpected data layout".into()))?;
            let [rx, ry, rz] = rmat_to_rvec(&m);
            dst.rows = 3;
            dst.cols = 1;
            dst.channels = 1;
            dst.data = vec![rx, ry, rz];
            Ok(())
        }
        _ => Err(PureCvError::InvalidInput(
            "rodrigues: src must be 3×1, 1×3, or 3×3 (single-channel f64)".to_string(),
        )),
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Rotation vector `(rx, ry, rz)` → 3×3 rotation matrix (row-major).
///
/// Uses the Rodrigues formula:
/// ```text
/// R = cos(θ)·I  +  (1 − cos(θ))·k·kᵀ  +  sin(θ)·K_×
/// ```
/// where `θ = ‖r‖`, `k = r / θ`, and `K_×` is the skew-symmetric
/// (cross-product) matrix of `k`.
pub(super) fn rvec_to_rmat(rx: f64, ry: f64, rz: f64) -> [f64; 9] {
    let theta = (rx * rx + ry * ry + rz * rz).sqrt();

    if theta < 1e-10 {
        // Identity rotation.
        return [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
    }

    let kx = rx / theta;
    let ky = ry / theta;
    let kz = rz / theta;
    let c = theta.cos();
    let s = theta.sin();
    let t = 1.0 - c;

    // Rodrigues' formula (expanded):
    [
        t * kx * kx + c,
        t * kx * ky - s * kz,
        t * kx * kz + s * ky,
        t * kx * ky + s * kz,
        t * ky * ky + c,
        t * ky * kz - s * kx,
        t * kx * kz - s * ky,
        t * ky * kz + s * kx,
        t * kz * kz + c,
    ]
}

/// 3×3 rotation matrix (row-major) → rotation vector.
///
/// ```text
/// θ = arccos((trace(R) − 1) / 2)
/// r = θ / (2 sin θ) · [R₃₂ − R₂₃,  R₁₃ − R₃₁,  R₂₁ − R₁₂]
/// ```
pub(super) fn rmat_to_rvec(m: &[f64; 9]) -> [f64; 3] {
    // Clamp argument of arccos to [−1, 1] to guard against floating-point noise.
    let trace_val = ((m[0] + m[4] + m[8] - 1.0) * 0.5).clamp(-1.0, 1.0);
    let theta = trace_val.acos();

    if theta.abs() < 1e-10 {
        // Near-identity: zero rotation vector.
        return [0.0, 0.0, 0.0];
    }

    // Near π the formula becomes numerically unstable; use an alternative.
    if (theta - core::f64::consts::PI).abs() < 1e-4 {
        return rmat_to_rvec_near_pi(m, theta);
    }

    let factor = theta / (2.0 * theta.sin());
    [
        factor * (m[7] - m[5]),
        factor * (m[2] - m[6]),
        factor * (m[3] - m[1]),
    ]
}

/// Special-case rotation matrix → rvec near `θ ≈ π`.
///
/// When `sin(θ) ≈ 0`, the standard formula is ill-conditioned.  Instead we
/// extract the axis from the diagonal of `(R + I) / 2 = k·kᵀ`.
fn rmat_to_rvec_near_pi(m: &[f64; 9], theta: f64) -> [f64; 3] {
    // (R + Rᵀ) / 2 = cos(θ)·I + (1−cos(θ))·k·kᵀ
    // For θ = π: cos(θ) = −1, so (R + I)/2 = k·kᵀ.
    // Diagonal: k_i² = (m[i*4] + 1) / 2.
    let kx2 = ((m[0] + 1.0) * 0.5).max(0.0);
    let ky2 = ((m[4] + 1.0) * 0.5).max(0.0);
    let kz2 = ((m[8] + 1.0) * 0.5).max(0.0);

    let kx = kx2.sqrt();
    let ky = ky2.sqrt();
    let kz = kz2.sqrt();

    // Resolve sign ambiguity from off-diagonal entries.
    // m[1] = (1−cosθ)·kx·ky + sinθ·(-kz); for θ≈π sinθ≈0 so sign(m[1]) = sign(kx·ky).
    let (kx, ky, kz) = fix_sign(kx, ky, kz, m);

    [theta * kx, theta * ky, theta * kz]
}

/// Fix the sign of the axis components using off-diagonal matrix entries.
fn fix_sign(kx: f64, mut ky: f64, mut kz: f64, m: &[f64; 9]) -> (f64, f64, f64) {
    // m[1] ≈ (1-cosθ)*kx*ky (for θ≈π)  — positive iff kx and ky have same sign.
    if m[1] < 0.0 {
        ky = -ky;
    }
    // m[2] ≈ (1-cosθ)*kx*kz
    if m[2] < 0.0 {
        kz = -kz;
    }
    // Verify: m[5] ≈ (1-cosθ)*ky*kz; if not consistent, flip kz.
    if m[5] < 0.0 && (ky * kz > 0.0) {
        kz = -kz;
    }
    (kx, ky, kz)
}

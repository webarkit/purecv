/*
 *  levmarq.rs
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

//! Levenberg-Marquardt solvers — the two optimisers of OpenCV's `calib3d`.
//!
//! OpenCV 4.10.0 has two of them, and each refinement uses a specific one:
//!
//! - [`LmSolver`] is `cv::LMSolver`
//!   ([`levmarq.cpp`](https://github.com/opencv/opencv/blob/4.10.0/modules/calib3d/src/levmarq.cpp#L80-L218)):
//!   additive damping `JᵀJ + λ·diag(JᵀJ)` with gain-ratio control of λ. It
//!   drives `findHomography`'s refinement, `solvePnPRefineLM` and the
//!   `estimateAffine*` refinements.
//! - `cv_lev_marq` is the legacy `CvLevMarq`
//!   ([`compat_ptsetreg.cpp`](https://github.com/opencv/opencv/blob/4.10.0/modules/calib3d/src/compat_ptsetreg.cpp#L55-L287)):
//!   multiplicative damping `diag(JᵀJ)·(1 + λ)` with `λ = 10^k`. It drives
//!   `SOLVEPNP_ITERATIVE` (`cvFindExtrinsicCameraParams2`) and
//!   `calibrateCamera`.
//!
//! Both solve their damped normal equations with a pseudo-inverse built from
//! the symmetric eigendecomposition (`DECOMP_SVD` / `DECOMP_EIG` on a
//! symmetric matrix), so rank-deficient steps leave the null space alone.

use alloc::{vec, vec::Vec};
#[allow(unused_imports)]
use num_traits::Float;

use super::linalg::jacobi_eigen;

// ---------------------------------------------------------------------------
// Shared linear algebra
// ---------------------------------------------------------------------------

/// `JᵀJ` and `Jᵀr` for a row-major `m × n` Jacobian `j` and residuals `r`.
fn normal_equations(j: &[f64], r: &[f64], n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut jtj = vec![0.0; n * n];
    let mut jtr = vec![0.0; n];
    for (row, &ri) in j.chunks_exact(n).zip(r) {
        for a in 0..n {
            jtr[a] += row[a] * ri;
            for b in a..n {
                jtj[a * n + b] += row[a] * row[b];
            }
        }
    }
    for a in 0..n {
        for b in 0..a {
            jtj[a * n + b] = jtj[b * n + a];
        }
    }
    (jtj, jtr)
}

/// Eigendecomposition of the symmetric `n × n` matrix `a`: eigenvalues and
/// the eigenvectors as the columns of a row-major matrix.
///
/// The matrix is scaled to unit max-norm first, because the Jacobi sweeps of
/// [`jacobi_eigen`] use absolute convergence thresholds.
fn sym_eigen(a: &[f64], n: usize) -> (Vec<f64>, Vec<f64>) {
    let scale = a.iter().fold(0.0f64, |m, v| m.max(v.abs()));
    let mut w = vec![0.0; n];
    let mut v = vec![0.0; n * n];
    if scale == 0.0 || !scale.is_finite() {
        for i in 0..n {
            v[i * n + i] = 1.0;
        }
        return (w, v);
    }
    let mut m: Vec<f64> = a.iter().map(|x| x / scale).collect();
    jacobi_eigen(&mut m, n, &mut v);
    for (i, wi) in w.iter_mut().enumerate() {
        *wi = m[i * n + i] * scale;
    }
    (w, v)
}

/// Solves the symmetric system `a x = b` with the pseudo-inverse, like
/// `cv::solve(a, b, x, DECOMP_SVD | DECOMP_EIG)`: eigen-directions whose
/// |eigenvalue| is at most `2·ε·Σ|w|` are dropped (`SVBkSb`'s threshold).
fn sym_solve(a: &[f64], b: &[f64], n: usize) -> Vec<f64> {
    let (w, v) = sym_eigen(a, n);
    let threshold = 2.0 * f64::EPSILON * w.iter().map(|x| x.abs()).sum::<f64>();
    let mut x = vec![0.0; n];
    for k in 0..n {
        if w[k].abs() <= threshold {
            continue;
        }
        let mut proj = 0.0;
        for i in 0..n {
            proj += v[i * n + k] * b[i];
        }
        let s = proj / w[k];
        for i in 0..n {
            x[i] += s * v[i * n + k];
        }
    }
    x
}

/// The diagonal of the pseudo-inverse of the symmetric matrix `a`
/// (`invert(a, ainv, DECOMP_EIG)`, diagonal only).
fn sym_pinv_diag(a: &[f64], n: usize) -> Vec<f64> {
    let (w, v) = sym_eigen(a, n);
    let threshold = 2.0 * f64::EPSILON * w.iter().map(|x| x.abs()).sum::<f64>();
    let mut d = vec![0.0; n];
    for k in 0..n {
        if w[k].abs() <= threshold {
            continue;
        }
        for (i, di) in d.iter_mut().enumerate() {
            *di += v[i * n + k] * v[i * n + k] / w[k];
        }
    }
    d
}

fn norm_l2(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

fn norm_inf(v: &[f64]) -> f64 {
    v.iter().fold(0.0f64, |m, x| m.max(x.abs()))
}

// ---------------------------------------------------------------------------
// cv::LMSolver
// ---------------------------------------------------------------------------

/// The model of an [`LmSolver`] problem, `cv::LMSolver::Callback`.
///
/// Implemented for closures `FnMut(&[f64], &mut Vec<f64>, Option<&mut Vec<f64>>) -> bool`.
pub trait LmCallback {
    /// Computes the residuals at `param` into `err` (resized by the callee)
    /// and, when `jac` is `Some`, the Jacobian of the residuals with respect
    /// to the parameters, row-major `err.len() × param.len()`.
    ///
    /// Returns `false` to stop the solver.
    fn compute(&mut self, param: &[f64], err: &mut Vec<f64>, jac: Option<&mut Vec<f64>>) -> bool;
}

impl<F> LmCallback for F
where
    F: FnMut(&[f64], &mut Vec<f64>, Option<&mut Vec<f64>>) -> bool,
{
    fn compute(&mut self, param: &[f64], err: &mut Vec<f64>, jac: Option<&mut Vec<f64>>) -> bool {
        self(param, err, jac)
    }
}

/// Levenberg-Marquardt solver, a port of OpenCV's `cv::LMSolver`.
///
/// Minimises `‖err(x)‖²` from a starting point. Each iteration solves
/// `(JᵀJ + λ·D) d = Jᵀerr` with `D = diag(JᵀJ)` at the start, tries
/// `x − d`, and adapts `λ` from the ratio of the actual to the predicted
/// decrease (halved above 0.75, raised by a factor in `[2, 10]` below 0.25,
/// set to 0 below a cut-off). It stops after `max_iters` iterations, or when
/// the step's or the residual's max-norm falls below `eps`.
///
/// # Example
/// ```
/// use purecv::calib3d::LmSolver;
///
/// // Fit y = a·exp(b·t) to exact samples of a = 2, b = -0.5.
/// let t: Vec<f64> = (0..10).map(|i| i as f64 * 0.5).collect();
/// let y: Vec<f64> = t.iter().map(|&t| 2.0 * (-0.5 * t).exp()).collect();
/// let mut model = |p: &[f64], err: &mut Vec<f64>, jac: Option<&mut Vec<f64>>| {
///     err.clear();
///     err.extend(t.iter().zip(&y).map(|(&t, &y)| p[0] * (p[1] * t).exp() - y));
///     if let Some(j) = jac {
///         j.clear();
///         for &t in &t {
///             let e = (p[1] * t).exp();
///             j.extend([e, p[0] * t * e]);
///         }
///     }
///     true
/// };
/// let mut p = [1.0, 0.0];
/// let iters = LmSolver::new(100).run(&mut p, &mut model);
/// assert!(iters > 0);
/// assert!((p[0] - 2.0).abs() < 1e-6 && (p[1] + 0.5).abs() < 1e-6);
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LmSolver {
    max_iters: i32,
    epsx: f64,
    epsf: f64,
}

impl LmSolver {
    /// `LMSolver::create(cb, max_iters)`: both tolerances are `FLT_EPSILON`.
    pub fn new(max_iters: i32) -> Self {
        Self::with_eps(max_iters, f32::EPSILON as f64)
    }

    /// `LMSolver::create(cb, max_iters, eps)`: step and residual tolerance.
    pub fn with_eps(max_iters: i32, eps: f64) -> Self {
        Self {
            max_iters,
            epsx: eps,
            epsf: eps,
        }
    }

    /// The iteration limit, `LMSolver::getMaxIters`.
    pub fn max_iters(&self) -> i32 {
        self.max_iters
    }

    /// Refines `param` in place, `LMSolver::run`.
    ///
    /// Returns the number of iterations, negated when the limit was reached
    /// (`-max_iters`), or `-1` when the callback returned `false` (in which
    /// case `param` is left unchanged).
    pub fn run<C: LmCallback + ?Sized>(&self, param: &mut [f64], cb: &mut C) -> i32 {
        let lx = param.len();
        let mut x = param.to_vec();
        let mut r = Vec::new();
        let mut jac = Vec::new();
        if !cb.compute(&x, &mut r, Some(&mut jac)) {
            return -1;
        }
        let mut s = r.iter().map(|v| v * v).sum::<f64>();
        let (mut a, mut v) = normal_equations(&jac, &r, lx);
        let d_diag: Vec<f64> = (0..lx).map(|i| a[i * lx + i]).collect();

        const RLO: f64 = 0.25;
        const RHI: f64 = 0.75;
        let mut lambda = 1.0f64;
        let mut lc = 0.75f64;
        let mut iter = 0;
        let mut rd = Vec::new();
        loop {
            let mut ap = a.clone();
            for i in 0..lx {
                ap[i * lx + i] += lambda * d_diag[i];
            }
            let d = sym_solve(&ap, &v, lx);
            let xd: Vec<f64> = x.iter().zip(&d).map(|(xi, di)| xi - di).collect();
            if !cb.compute(&xd, &mut rd, None) {
                return -1;
            }
            let sd = rd.iter().map(|v| v * v).sum::<f64>();
            // dS = d · (2v − A d), the decrease predicted by the linear model.
            let mut ds = 0.0;
            for i in 0..lx {
                let mut ad = 0.0;
                for k in 0..lx {
                    ad += a[i * lx + k] * d[k];
                }
                ds += d[i] * (2.0 * v[i] - ad);
            }
            let rr = (s - sd) / if ds.abs() > f64::EPSILON { ds } else { 1.0 };

            if rr > RHI {
                lambda *= 0.5;
                if lambda < lc {
                    lambda = 0.0;
                }
            } else if rr < RLO {
                // Find a new factor when the ratio is too low.
                let t: f64 = d.iter().zip(&v).map(|(di, vi)| di * vi).sum();
                let mut nu = (sd - s) / if t.abs() > f64::EPSILON { t } else { 1.0 } + 2.0;
                nu = nu.clamp(2.0, 10.0);
                if lambda == 0.0 {
                    let inv = sym_pinv_diag(&a, lx);
                    let maxval = inv.iter().fold(f64::EPSILON, |m, x| m.max(x.abs()));
                    lambda = 1.0 / maxval;
                    lc = lambda;
                    nu *= 0.5;
                }
                lambda *= nu;
            }

            if sd < s {
                s = sd;
                x = xd;
                if !cb.compute(&x, &mut r, Some(&mut jac)) {
                    return -1;
                }
                (a, v) = normal_equations(&jac, &r, lx);
            }

            iter += 1;
            let proceed =
                iter < self.max_iters && norm_inf(&d) >= self.epsx && norm_inf(&r) >= self.epsf;
            if !proceed {
                break;
            }
        }
        param.copy_from_slice(&x);
        if iter == self.max_iters {
            -iter
        } else {
            iter
        }
    }
}

// ---------------------------------------------------------------------------
// CvLevMarq
// ---------------------------------------------------------------------------

/// Refines `param` with OpenCV's legacy `CvLevMarq` (the `update()` loop
/// with residuals and Jacobian, all parameters free).
///
/// `compute(param, err, jac)` writes the `err.len()` residuals at `param`
/// and, when `jac` is `Some`, the row-major Jacobian (`err.len() ×
/// param.len()`); both buffers are zeroed before every call. `max_iter` is
/// clamped to `[1, 1000]` and `epsilon` to `≥ 0`, as with
/// `CV_TERMCRIT_ITER + CV_TERMCRIT_EPS`.
///
/// Each step solves `(JᵀJ with its diagonal × (1 + λ)) δ = Jᵀerr` and tries
/// `param − δ`, with `λ = 10^k`, `k` starting at −3. A step that increases
/// `‖err‖` is retried with `k + 1` (up to 16); an accepted step lowers `k`
/// (down to −16). It stops after `max_iter` accepted steps or when
/// `‖Δparam‖ / ‖param_prev‖ < epsilon`.
pub(crate) fn cv_lev_marq<F>(
    param: &mut [f64],
    n_err: usize,
    max_iter: usize,
    epsilon: f64,
    mut compute: F,
) where
    F: FnMut(&[f64], &mut [f64], Option<&mut [f64]>),
{
    let n = param.len();
    let max_iter = max_iter.clamp(1, 1000);
    let epsilon = epsilon.max(0.0);
    let mut lambda_lg10: i32 = -3;
    let mut iters = 0usize;
    let mut err = vec![0.0; n_err];
    let mut jac = vec![0.0; n_err * n];
    let mut prev = param.to_vec();
    let mut prev_err_norm = f64::MAX;

    let step = |param: &mut [f64], prev: &[f64], jtj: &[f64], jte: &[f64], lg10: i32| {
        let lambda = (lg10 as f64 * core::f64::consts::LN_10).exp();
        let mut a = jtj.to_vec();
        for i in 0..n {
            a[i * n + i] *= 1.0 + lambda;
        }
        let delta = sym_solve(&a, jte, n);
        for i in 0..n {
            param[i] = prev[i] - delta[i];
        }
    };

    // STARTED: residuals and Jacobian at the initial parameters.
    compute(param, &mut err, Some(&mut jac));
    loop {
        // CALC_J: take a step from the current parameters.
        let (jtj, jte) = normal_equations(&jac, &err, n);
        prev.copy_from_slice(param);
        step(param, &prev, &jtj, &jte, lambda_lg10);
        if iters == 0 {
            prev_err_norm = norm_l2(&err);
        }

        // CHECK_ERR: retry with more damping while the error grows.
        let err_norm = loop {
            err.fill(0.0);
            compute(param, &mut err, None);
            let e = norm_l2(&err);
            if e > prev_err_norm {
                lambda_lg10 += 1;
                if lambda_lg10 <= 16 {
                    step(param, &prev, &jtj, &jte, lambda_lg10);
                    continue;
                }
            }
            break e;
        };

        lambda_lg10 = (lambda_lg10 - 1).max(-16);
        iters += 1;
        let diff: Vec<f64> = param.iter().zip(&prev).map(|(p, q)| p - q).collect();
        if iters >= max_iter || norm_l2(&diff) / (norm_l2(&prev) + f64::EPSILON) < epsilon {
            return;
        }
        prev_err_norm = err_norm;
        jac.fill(0.0);
        err.fill(0.0);
        compute(param, &mut err, Some(&mut jac));
    }
}

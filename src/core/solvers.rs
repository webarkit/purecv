/*
 *  solvers.rs
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

use crate::core::constants::CV_PI;
use crate::core::error::Result;

/// Solves quadratic equation: a*x^2 + b*x + c = 0
/// Returns a vector of real roots.
pub fn solve_quadratic(a: f64, b: f64, c: f64) -> Result<Vec<f64>> {
    if a.abs() < f64::EPSILON {
        if b.abs() < f64::EPSILON {
            return Ok(vec![]);
        }
        return Ok(vec![-c / b]);
    }

    let delta = b * b - 4.0 * a * c;
    if delta < 0.0 {
        return Ok(vec![]);
    } else if delta.abs() < f64::EPSILON {
        return Ok(vec![-b / (2.0 * a)]);
    }

    let sqrt_delta = delta.sqrt();
    let x1 = (-b - sqrt_delta) / (2.0 * a);
    let x2 = (-b + sqrt_delta) / (2.0 * a);
    Ok(vec![x1, x2])
}

/// Solves cubic equation: a*x^3 + b*x^2 + c*x + d = 0
/// Returns a vector of real roots.
pub fn solve_cubic(a: f64, b: f64, c: f64, d: f64) -> Result<Vec<f64>> {
    if a.abs() < f64::EPSILON {
        return solve_quadratic(b, c, d);
    }

    // Normalized coefficients
    let a_norm = b / a;
    let b_norm = c / a;
    let c_norm = d / a;

    let q = (3.0 * b_norm - (a_norm * a_norm)) / 9.0;
    let r = (9.0 * a_norm * b_norm - 27.0 * c_norm - 2.0 * (a_norm * a_norm * a_norm)) / 54.0;

    let q3 = q * q * q;
    let d_sq = q3 + r * r;

    let offset = a_norm / 3.0;

    if d_sq >= 0.0 {
        // One real root
        let sq = d_sq.sqrt();
        let s = (r + sq).cbrt();
        let t = (r - sq).cbrt();
        Ok(vec![s + t - offset])
    } else {
        // Three real roots
        let th = (r / (-q3).sqrt()).acos();
        let factor = 2.0 * (-q).sqrt();
        let x1 = factor * (th / 3.0).cos() - offset;
        let x2 = factor * ((th - 2.0 * CV_PI) / 3.0).cos() - offset;
        let x3 = factor * ((th + 2.0 * CV_PI) / 3.0).cos() - offset;

        Ok(vec![x1, x2, x3])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_solve_quadratic() {
        // x^2 - 4 = 0 -> x = -2, 2
        let roots = solve_quadratic(1.0, 0.0, -4.0).unwrap();
        assert_eq!(roots.len(), 2);
        assert!((roots[0] + 2.0).abs() < 1e-5 || (roots[1] + 2.0).abs() < 1e-5);
    }

    #[test]
    fn test_solve_cubic() {
        // x^3 - 6x^2 + 11x - 6 = 0  -> roots: 1, 2, 3
        let mut roots = solve_cubic(1.0, -6.0, 11.0, -6.0).unwrap();
        roots.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert_eq!(roots.len(), 3);
        assert!((roots[0] - 1.0).abs() < 1e-5);
        assert!((roots[1] - 2.0).abs() < 1e-5);
        assert!((roots[2] - 3.0).abs() < 1e-5);
    }
}

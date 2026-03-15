/*
 *  rng.rs
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

use crate::core::error::{PureCvError, Result};
use crate::core::types::Scalar;
use crate::core::Matrix;
use num_traits::{FromPrimitive, ToPrimitive};
use std::cell::RefCell;

// ---------------------------------------------------------------------------
// Xoshiro256** — a fast, high-quality pure-Rust PRNG (public domain algorithm
// by David Blackman and Sebastiano Vigna).
// ---------------------------------------------------------------------------

/// Internal state for the xoshiro256** generator.
#[derive(Clone)]
struct Xoshiro256 {
    s: [u64; 4],
}

impl Xoshiro256 {
    /// Creates a new generator seeded by expanding `seed` through SplitMix64.
    fn from_seed(seed: u64) -> Self {
        let mut sm = seed;
        let mut s = [0u64; 4];
        for val in &mut s {
            *val = Self::splitmix64(&mut sm);
        }
        // Guard against an all-zero state (the generator's fixed point).
        if s == [0; 4] {
            s[0] = 0x853c49e6748fea9b;
        }
        Self { s }
    }

    /// SplitMix64 — used only to expand a single u64 seed into 256 bits of state.
    #[inline]
    fn splitmix64(state: &mut u64) -> u64 {
        *state = state.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = *state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^ (z >> 31)
    }

    /// Returns the next pseudo-random `u64`.
    #[inline]
    fn next_u64(&mut self) -> u64 {
        let result = (self.s[1].wrapping_mul(5))
            .rotate_left(7)
            .wrapping_mul(9);
        let t = self.s[1] << 17;

        self.s[2] ^= self.s[0];
        self.s[3] ^= self.s[1];
        self.s[1] ^= self.s[2];
        self.s[0] ^= self.s[3];

        self.s[2] ^= t;
        self.s[3] = self.s[3].rotate_left(45);

        result
    }

    /// Returns a uniformly distributed `f64` in [0, 1).
    #[inline]
    fn next_f64(&mut self) -> f64 {
        // Use the upper 53 bits for a full mantissa.
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }

    /// Returns a pair of independent standard-normal samples via the
    /// Box-Muller transform.
    #[inline]
    fn next_gaussian_pair(&mut self) -> (f64, f64) {
        loop {
            let u1 = self.next_f64();
            let u2 = self.next_f64();
            // Reject the degenerate case u1 == 0 (log(0) is -inf).
            if u1 > f64::EPSILON {
                let r = (-2.0 * u1.ln()).sqrt();
                let theta = std::f64::consts::TAU * u2;
                return (r * theta.cos(), r * theta.sin());
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Thread-local RNG state
// ---------------------------------------------------------------------------

thread_local! {
    static THREAD_RNG: RefCell<Xoshiro256> = RefCell::new(Xoshiro256::from_seed(0));
}

/// Sets the seed for the thread-local random number generator.
///
/// Mirrors OpenCV's `cv::setRNGSeed`. The seed affects only the calling
/// thread; other threads retain their own independent state.
///
/// # Arguments
/// * `seed` - The seed value. Passing the same seed produces the same
///   sequence of random numbers.
///
/// # Example
/// ```
/// use purecv::core::set_rng_seed;
/// set_rng_seed(42);
/// ```
pub fn set_rng_seed(seed: u64) {
    THREAD_RNG.with(|rng| {
        *rng.borrow_mut() = Xoshiro256::from_seed(seed);
    });
}

/// Fills a matrix with uniformly distributed random numbers.
///
/// Each element is drawn independently from the interval `[low, high)`.
/// Per-channel bounds are taken from the first `channels` components of the
/// `Scalar` values (matching OpenCV semantics).
///
/// Mirrors OpenCV's `cv::randu`.
///
/// # Arguments
/// * `dst`  - The output matrix. Must already be allocated with the desired
///            size and number of channels.
/// * `low`  - Inclusive lower bound for each channel.
/// * `high` - Exclusive upper bound for each channel.
///
/// # Errors
/// Returns `PureCvError::InvalidDimensions` if the matrix is empty.
///
/// # Example
/// ```
/// use purecv::core::{Matrix, set_rng_seed, randu};
/// use purecv::core::Scalar;
///
/// set_rng_seed(12345);
/// let mut mat = Matrix::<f32>::new(100, 100, 1);
/// randu(&mut mat, Scalar::all(0.0), Scalar::all(1.0)).unwrap();
/// ```
pub fn randu<T>(dst: &mut Matrix<T>, low: Scalar<f64>, high: Scalar<f64>) -> Result<()>
where
    T: Default + Clone + FromPrimitive + ToPrimitive + Send + Sync,
{
    if dst.data.is_empty() {
        return Err(PureCvError::InvalidDimensions(
            "destination matrix is empty".into(),
        ));
    }

    let channels = dst.channels;

    THREAD_RNG.with(|rng| {
        let mut rng = rng.borrow_mut();
        for chunk in dst.data.chunks_exact_mut(channels) {
            for (i, elem) in chunk.iter_mut().enumerate() {
                let ch = i % channels;
                let lo = low.v[ch];
                let hi = high.v[ch];
                let val = lo + rng.next_f64() * (hi - lo);
                *elem = T::from_f64(val).unwrap_or_default();
            }
        }
    });

    Ok(())
}

/// Fills a matrix with normally (Gaussian) distributed random numbers.
///
/// Each element is drawn independently from a Gaussian distribution with
/// the given per-channel `mean` and `std_dev`.
///
/// Mirrors OpenCV's `cv::randn`.
///
/// # Arguments
/// * `dst`     - The output matrix. Must already be allocated.
/// * `mean`    - Mean value for each channel.
/// * `std_dev` - Standard deviation for each channel.
///
/// # Errors
/// Returns `PureCvError::InvalidDimensions` if the matrix is empty.
///
/// # Example
/// ```
/// use purecv::core::{Matrix, set_rng_seed, randn};
/// use purecv::core::Scalar;
///
/// set_rng_seed(12345);
/// let mut mat = Matrix::<f64>::new(100, 100, 1);
/// randn(&mut mat, Scalar::all(0.0), Scalar::all(1.0)).unwrap();
/// ```
pub fn randn<T>(dst: &mut Matrix<T>, mean: Scalar<f64>, std_dev: Scalar<f64>) -> Result<()>
where
    T: Default + Clone + FromPrimitive + ToPrimitive + Send + Sync,
{
    if dst.data.is_empty() {
        return Err(PureCvError::InvalidDimensions(
            "destination matrix is empty".into(),
        ));
    }

    let channels = dst.channels;
    let total = dst.data.len();

    THREAD_RNG.with(|rng| {
        let mut rng = rng.borrow_mut();
        let mut idx = 0;
        while idx < total {
            let (g0, g1) = rng.next_gaussian_pair();
            // First sample
            let ch0 = idx % channels;
            let val0 = mean.v[ch0] + g0 * std_dev.v[ch0];
            dst.data[idx] = T::from_f64(val0).unwrap_or_default();
            idx += 1;

            // Second sample (Box-Muller yields two at a time)
            if idx < total {
                let ch1 = idx % channels;
                let val1 = mean.v[ch1] + g1 * std_dev.v[ch1];
                dst.data[idx] = T::from_f64(val1).unwrap_or_default();
                idx += 1;
            }
        }
    });

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_set_rng_seed_determinism() {
        // Two sequences seeded identically must be equal.
        set_rng_seed(42);
        let mut m1 = Matrix::<f64>::new(10, 10, 1);
        randu(&mut m1, Scalar::all(0.0), Scalar::all(1.0)).unwrap();

        set_rng_seed(42);
        let mut m2 = Matrix::<f64>::new(10, 10, 1);
        randu(&mut m2, Scalar::all(0.0), Scalar::all(1.0)).unwrap();

        assert_eq!(m1.data, m2.data);
    }

    #[test]
    fn test_randu_bounds() {
        set_rng_seed(123);
        let mut mat = Matrix::<f64>::new(50, 50, 1);
        randu(&mut mat, Scalar::all(-5.0), Scalar::all(5.0)).unwrap();

        for &v in &mat.data {
            assert!(v >= -5.0 && v < 5.0, "value {} out of range [-5, 5)", v);
        }
    }

    #[test]
    fn test_randu_per_channel_bounds() {
        set_rng_seed(99);
        let mut mat = Matrix::<f64>::new(20, 20, 3);
        let low = Scalar::new(0.0, 10.0, 100.0, 0.0);
        let high = Scalar::new(1.0, 20.0, 200.0, 0.0);
        randu(&mut mat, low, high).unwrap();

        for chunk in mat.data.chunks_exact(3) {
            assert!(
                chunk[0] >= 0.0 && chunk[0] < 1.0,
                "ch0: {} not in [0,1)",
                chunk[0]
            );
            assert!(
                chunk[1] >= 10.0 && chunk[1] < 20.0,
                "ch1: {} not in [10,20)",
                chunk[1]
            );
            assert!(
                chunk[2] >= 100.0 && chunk[2] < 200.0,
                "ch2: {} not in [100,200)",
                chunk[2]
            );
        }
    }

    #[test]
    fn test_randu_u8() {
        set_rng_seed(0);
        let mut mat = Matrix::<u8>::new(100, 100, 1);
        randu(&mut mat, Scalar::all(0.0), Scalar::all(256.0)).unwrap();

        // All values should be in [0, 255].
        for &v in &mat.data {
            assert!(v <= 255);
        }
        // With 10 000 samples the range should not collapse to a single value.
        let min = *mat.data.iter().min().unwrap();
        let max = *mat.data.iter().max().unwrap();
        assert!(max > min, "randu produced no variation");
    }

    #[test]
    fn test_randn_statistics() {
        set_rng_seed(7);
        let n = 100_000;
        let mut mat = Matrix::<f64>::new(1, n, 1);
        randn(&mut mat, Scalar::all(50.0), Scalar::all(10.0)).unwrap();

        let sum: f64 = mat.data.iter().sum();
        let mean_val = sum / n as f64;
        let var: f64 = mat.data.iter().map(|v| (v - mean_val).powi(2)).sum::<f64>() / n as f64;
        let std_val = var.sqrt();

        // Allow ±1 unit of tolerance on 100 k samples.
        assert!(
            (mean_val - 50.0).abs() < 1.0,
            "mean {} too far from 50",
            mean_val
        );
        assert!(
            (std_val - 10.0).abs() < 1.0,
            "std {} too far from 10",
            std_val
        );
    }

    #[test]
    fn test_randn_determinism() {
        set_rng_seed(77);
        let mut m1 = Matrix::<f64>::new(10, 10, 1);
        randn(&mut m1, Scalar::all(0.0), Scalar::all(1.0)).unwrap();

        set_rng_seed(77);
        let mut m2 = Matrix::<f64>::new(10, 10, 1);
        randn(&mut m2, Scalar::all(0.0), Scalar::all(1.0)).unwrap();

        assert_eq!(m1.data, m2.data);
    }

    #[test]
    fn test_randu_empty_matrix_error() {
        let mut mat = Matrix::<f64>::new(0, 0, 1);
        let result = randu(&mut mat, Scalar::all(0.0), Scalar::all(1.0));
        assert!(result.is_err());
    }

    #[test]
    fn test_randn_empty_matrix_error() {
        let mut mat = Matrix::<f64>::new(0, 0, 1);
        let result = randn(&mut mat, Scalar::all(0.0), Scalar::all(1.0));
        assert!(result.is_err());
    }
}


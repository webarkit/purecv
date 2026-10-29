/*
 *  rng.rs
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

//! Random number generation — the counterpart of OpenCV's `cv::RNG`.
//!
//! [`Rng`] is a port of OpenCV's 64-bit multiply-with-carry generator
//! ([`rand.cpp`](https://github.com/opencv/opencv/blob/4.10.0/modules/core/src/rand.cpp)):
//! the same seed gives the same sequence as `cv::RNG`, including
//! [`Rng::fill_uniform`] (`RNG::fill(UNIFORM)`), [`Rng::fill_normal`] (the
//! ziggurat behind `RNG::fill(NORMAL)`) and [`Rng::shuffle`] (`randShuffle`).
//!
//! With `std`, the free functions [`randu`], [`randn`], [`rand_shuffle`] and
//! [`set_rng_seed`] work on a thread-local generator like `cv::theRNG()`.
//! That generator is a [`Rng`] by default; [`set_rng_generator`] switches
//! the calling thread to the xoshiro256** generator purecv used before, which
//! keeps its own (non-OpenCV) sequences.

// `num_traits::Float` provides `ln`, `exp`, `sqrt`, ... on `f32`/`f64` via
// libm when `std` is disabled; with `std` the inherent methods win, so the
// import is only "used" in no_std builds.
#[cfg(feature = "std")]
use crate::core::constants::CV_2PI;
use crate::core::error::Result;
use crate::core::logging::tags;
use crate::core::matrix::{DataType, Depth};
use crate::core::saturate::SaturateCast;
use crate::core::types::Scalar;
use crate::core::Matrix;
use crate::cv_bail;
#[cfg(feature = "std")]
use core::cell::RefCell;
#[allow(unused_imports)]
use num_traits::Float;

// ---------------------------------------------------------------------------
// cv::RNG — 64-bit multiply-with-carry
// ---------------------------------------------------------------------------

/// `CV_RNG_COEFF`, the multiplier of OpenCV's multiply-with-carry step.
const CV_RNG_COEFF: u64 = 4_164_903_690;

/// `BLOCK_SIZE` of OpenCV's `core` (`precomp.hpp`): `RNG::fill` works in
/// blocks of about this many elements, and integer `randu` restarts its
/// four-per-draw loop at every block.
const BLOCK_SIZE: usize = 1024;

/// One multiply-with-carry step, `RNG_NEXT` in OpenCV:
/// `X(n+1) = A * (X(n) mod 2^32) + X(n) / 2^32`.
#[inline]
fn rng_next(x: u64) -> u64 {
    (x & 0xffff_ffff) * CV_RNG_COEFF + (x >> 32)
}

/// A port of OpenCV's `cv::RNG`, a 64-bit multiply-with-carry generator.
///
/// The same `state` produces the same sequence as `cv::RNG` in OpenCV
/// 4.10.0, for the scalar draws ([`next_u32`](Self::next_u32),
/// [`uniform_i32`](Self::uniform_i32), [`gaussian`](Self::gaussian), …) and
/// for the matrix fills.
///
/// # Example
/// ```
/// use purecv::core::Rng;
///
/// let mut rng = Rng::new(42);
/// let a = rng.uniform_i32(0, 10);
/// assert!((0..10).contains(&a));
/// // A zero seed selects the default state, like `cv::RNG(0)`.
/// assert_eq!(Rng::new(0), Rng::default());
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Rng {
    /// The generator state, `RNG::state`.
    pub state: u64,
}

impl Default for Rng {
    /// `cv::RNG()`: the default state `0xffffffff`.
    fn default() -> Self {
        Self {
            state: Self::DEFAULT_STATE,
        }
    }
}

impl Rng {
    /// The state of `cv::RNG()` and of `cv::RNG(0)`.
    pub const DEFAULT_STATE: u64 = 0xffff_ffff;

    /// `cv::RNG(state)`: a zero state selects [`Rng::DEFAULT_STATE`].
    pub fn new(state: u64) -> Self {
        Self {
            state: if state == 0 {
                Self::DEFAULT_STATE
            } else {
                state
            },
        }
    }

    /// Returns the next 32-bit value, `RNG::next()` / `(unsigned)rng`.
    #[inline]
    pub fn next_u32(&mut self) -> u32 {
        self.state = rng_next(self.state);
        self.state as u32
    }

    /// Returns a value in `[a, b)`, `RNG::uniform(int, int)`; `a` when
    /// `a == b`.
    ///
    /// Like OpenCV, this is `next() % (b - a) + a` in wrapping 32-bit
    /// arithmetic, so `b < a` gives values outside `[b, a)`.
    #[inline]
    pub fn uniform_i32(&mut self, a: i32, b: i32) -> i32 {
        if a == b {
            return a;
        }
        (self.next_u32() % (b.wrapping_sub(a) as u32)).wrapping_add(a as u32) as i32
    }

    /// Returns a value in `[0, 1)`, `(float)rng`.
    #[inline]
    pub fn next_f32(&mut self) -> f32 {
        self.next_u32() as f32 * 2.328_306_4e-10
    }

    /// Returns a value in `[0, 1)` built from two draws, `(double)rng`.
    #[inline]
    pub fn next_f64(&mut self) -> f64 {
        let hi = self.next_u32() as u64;
        let lo = self.next_u32() as u64;
        ((hi << 32) | lo) as f64 * 5.421_010_862_427_522e-20
    }

    /// Returns a value in `[a, b)`, `RNG::uniform(float, float)`.
    #[inline]
    pub fn uniform_f32(&mut self, a: f32, b: f32) -> f32 {
        self.next_f32() * (b - a) + a
    }

    /// Returns a value in `[a, b)`, `RNG::uniform(double, double)`.
    #[inline]
    pub fn uniform_f64(&mut self, a: f64, b: f64) -> f64 {
        self.next_f64() * (b - a) + a
    }

    /// Returns a sample of `N(0, sigma²)`, `RNG::gaussian`.
    ///
    /// The sample is drawn in `f32` by the same ziggurat as
    /// [`fill_normal`](Self::fill_normal) and then scaled in `f64`.
    pub fn gaussian(&mut self, sigma: f64) -> f64 {
        let mut z = [0.0f32];
        with_ziggurat(|zig| self.standard_normals(zig, &mut z));
        z[0] as f64 * sigma
    }

    /// Fills `dst` with uniformly distributed values, `RNG::fill(mat,
    /// RNG::UNIFORM, low, high)` (and so `cv::randu`).
    ///
    /// Channel `c` draws from `[low.v[c], high.v[c])`; the bounds are not
    /// clamped to the range of `T` (`saturateRange = false`). Integer
    /// destinations draw integers from `[ceil(min), floor(max) - 1]` of the
    /// two bounds; float destinations map a signed 32-bit (`f32`) or 64-bit
    /// (`f64`) draw onto the interval, so `low > high` is allowed there.
    ///
    /// # Errors
    /// [`PureCvError::InvalidDimensions`](crate::core::PureCvError) if `dst`
    /// is empty, [`PureCvError::InvalidInput`](crate::core::PureCvError) if
    /// it has more than 4 channels (a [`Scalar`] holds 4).
    pub fn fill_uniform<T>(
        &mut self,
        dst: &mut Matrix<T>,
        low: Scalar<f64>,
        high: Scalar<f64>,
    ) -> Result<()>
    where
        T: DataType + SaturateCast,
    {
        let cn = check_fill_target(dst, "fill_uniform")?;
        let depth = T::depth();
        match depth {
            Depth::CV_32F => {
                let mut fp = [(0.0f32, 0.0f32); 4];
                for (c, p) in fp.iter_mut().enumerate().take(cn) {
                    let (p1, p2) = (low.v[c], high.v[c]);
                    *p = (
                        (f64::MAX.min(p2 - p1) * 2.328_306_436_538_696_3e-10) as f32,
                        ((p2 + p1) * 0.5) as f32,
                    );
                }
                for (i, v) in dst.data.iter_mut().enumerate() {
                    self.state = rng_next(self.state);
                    let (scale, bias) = fp[i % cn];
                    // `int * float` in OpenCV: the draw is converted to f32 first,
                    // and the bias is added in a separate pass (addRNGBias32f).
                    let x = (self.state as u32 as i32) as f32 * scale;
                    *v = T::saturate_from_f32(x + bias);
                }
            }
            Depth::CV_64F => {
                let mut dp = [(0.0f64, 0.0f64); 4];
                for (c, p) in dp.iter_mut().enumerate().take(cn) {
                    let (p1, p2) = (low.v[c], high.v[c]);
                    *p = (
                        f64::MAX.min(p2 - p1) * 5.421_010_862_427_522e-20,
                        (p2 + p1) * 0.5,
                    );
                }
                for (i, v) in dst.data.iter_mut().enumerate() {
                    self.state = rng_next(self.state);
                    let (scale, bias) = dp[i % cn];
                    let x = self.state.rotate_left(32) as i64 as f64 * scale;
                    *v = T::saturate_from_f64(x + bias);
                }
            }
            _ => self.fill_uniform_int(&mut dst.data, cn, &low, &high),
        }
        Ok(())
    }

    /// Integer `RNG::fill(UNIFORM)`: `randBits_` when every channel range is
    /// a power of two no wider than 2^32, `randi_` (a division by a
    /// precomputed reciprocal) otherwise.
    fn fill_uniform_int<T: SaturateCast>(
        &mut self,
        data: &mut [T],
        cn: usize,
        low: &Scalar<f64>,
        high: &Scalar<f64>,
    ) {
        // ip[c] = (mask or range - 1, offset), `Vec2i ip` in OpenCV.
        let mut ip = [(0i32, 0i32); 4];
        let mut fast_int_mode = true;
        let mut small_flag = true;
        for (c, p) in ip.iter_mut().enumerate().take(cn) {
            let a = low.v[c].min(high.v[c]);
            let b = low.v[c].max(high.v[c]);
            p.1 = a.ceil() as i32;
            let mut idiff = (b.floor() as i32).wrapping_sub(p.1).wrapping_sub(1);
            p.0 = idiff;
            if idiff < 0 {
                idiff = 0;
                p.0 = 0;
            }
            let diff = b - a;
            fast_int_mode =
                fast_int_mode && diff <= 4_294_967_296.0 && (idiff & idiff.wrapping_add(1)) == 0;
            if fast_int_mode {
                small_flag = small_flag && idiff <= 255;
            } else {
                if diff > i32::MAX as f64 {
                    p.0 = i32::MAX;
                }
                if a < (i32::MIN / 2) as f64 {
                    p.1 = i32::MIN / 2;
                }
            }
        }

        let total_px = data.len() / cn;
        let block_px = BLOCK_SIZE.div_ceil(cn).min(total_px);
        let out = |v: i32| T::saturate_from_f64(v as f64);

        if fast_int_mode {
            for block in data.chunks_mut(block_px * cn) {
                let len = block.len();
                let mut i = 0;
                if small_flag {
                    // One draw gives four 8-bit fields.
                    while i + 4 <= len {
                        self.state = rng_next(self.state);
                        let t = self.state as u32 as i32;
                        for k in 0..4 {
                            let (mask, offset) = ip[(i + k) % cn];
                            block[i + k] = out(((t >> (8 * k)) & mask).wrapping_add(offset));
                        }
                        i += 4;
                    }
                }
                for (j, v) in block.iter_mut().enumerate().skip(i) {
                    self.state = rng_next(self.state);
                    let (mask, offset) = ip[j % cn];
                    *v = out((self.state as u32 as i32 & mask).wrapping_add(offset));
                }
            }
        } else {
            // DivStruct: v mod d by multiplication with a reciprocal.
            let mut ds = [(0u32, 0u32, 0u32, 0u32, 0i32); 4];
            for (c, s) in ds.iter_mut().enumerate().take(cn) {
                let d = ip[c].0.wrapping_add(1) as u32;
                let mut l = 0;
                while (1u64 << l) < d as u64 {
                    l += 1;
                }
                let m = (((1u64 << 32) * ((1u64 << l) - d as u64)) / d as u64) as u32;
                *s = (
                    d,
                    m.wrapping_add(1),
                    l.min(1),
                    if l > 0 { l - 1 } else { 0 },
                    ip[c].1,
                );
            }
            for (i, v) in data.iter_mut().enumerate() {
                self.state = rng_next(self.state);
                let (d, m, sh1, sh2, delta) = ds[i % cn];
                let t = self.state as u32;
                let q = ((t as u64 * m as u64) >> 32) as u32;
                let q = q.wrapping_add(t.wrapping_sub(q) >> sh1) >> sh2;
                let r = t.wrapping_sub(q.wrapping_mul(d)).wrapping_add(delta as u32);
                *v = out(r as i32);
            }
        }
    }

    /// Fills `dst` with normally distributed values, `RNG::fill(mat,
    /// RNG::NORMAL, mean, stddev)` (and so `cv::randn`).
    ///
    /// Channel `c` is `N(mean.v[c], std_dev.v[c]²)`. Samples come from
    /// OpenCV's `f32` ziggurat and are scaled in `f32` (in `f64` for an
    /// `f64` destination); integer destinations are written through
    /// [`SaturateCast`], so samples out of range clamp to the type bounds.
    ///
    /// # Errors
    /// As [`fill_uniform`](Self::fill_uniform).
    pub fn fill_normal<T>(
        &mut self,
        dst: &mut Matrix<T>,
        mean: Scalar<f64>,
        std_dev: Scalar<f64>,
    ) -> Result<()>
    where
        T: DataType + SaturateCast,
    {
        let cn = check_fill_target(dst, "fill_normal")?;
        let is_f64 = T::depth() == Depth::CV_64F;
        // `randn_0_1_32f` draws as many samples as the block needs and the
        // scaling consumes no state, so block boundaries do not matter here.
        let mut z = [0.0f32; BLOCK_SIZE];
        with_ziggurat(|zig| {
            for block in dst.data.chunks_mut(BLOCK_SIZE - BLOCK_SIZE % cn) {
                let z = &mut z[..block.len()];
                self.standard_normals(zig, z);
                for (i, (v, &s)) in block.iter_mut().zip(z.iter()).enumerate() {
                    let c = i % cn;
                    *v = if is_f64 {
                        T::saturate_from_f64(s as f64 * std_dev.v[c] + mean.v[c])
                    } else {
                        T::saturate_from_f32(s * std_dev.v[c] as f32 + mean.v[c] as f32)
                    };
                }
            }
        });
        Ok(())
    }

    /// Shuffles `slice` in place, `cv::randShuffle(dst, iterFactor, &rng)`.
    ///
    /// Every element `i` is swapped with element `next() % len`, which is
    /// OpenCV's algorithm (not Fisher–Yates). OpenCV 4.10.0 ignores
    /// `iterFactor`, so it has no counterpart here.
    pub fn shuffle<T>(&mut self, slice: &mut [T]) {
        let n = slice.len() as u32;
        for i in 0..slice.len() {
            let j = (self.next_u32() % n) as usize;
            slice.swap(j, i);
        }
    }

    /// `randn_0_1_32f`: fills `out` with standard normal samples (ziggurat
    /// of Marsaglia and Tsang, 2000, as written in OpenCV).
    fn standard_normals(&mut self, zig: &Ziggurat, out: &mut [f32]) {
        const R: f32 = 3.442_62; // the start of the right tail
        const RNG_FLT: f32 = 2.328_306_4e-10; // 2^-32
        let mut temp = self.state;
        for v in out.iter_mut() {
            let x;
            loop {
                let hz = temp as u32 as i32;
                temp = rng_next(temp);
                let iz = (hz & 127) as usize;
                let xi = hz as f32 * zig.wn[iz];
                if hz.unsigned_abs() < zig.kn[iz] {
                    x = xi;
                    break;
                }
                if iz == 0 {
                    // The base strip: sample the tail beyond R.
                    let mut xt;
                    loop {
                        xt = temp as u32 as f32 * RNG_FLT;
                        temp = rng_next(temp);
                        let y = temp as u32 as f32 * RNG_FLT;
                        temp = rng_next(temp);
                        // 0.2904764 is 1/R; `log(float)` is the f32 overload.
                        xt = (-(xt + f32::MIN_POSITIVE).ln() as f64 * 0.2904764) as f32;
                        let y = -(y + f32::MIN_POSITIVE).ln();
                        if y + y >= xt * xt {
                            break;
                        }
                    }
                    x = if hz > 0 { R + xt } else { -R - xt };
                    break;
                }
                // The wedges of the other strips.
                let y = temp as u32 as f32 * RNG_FLT;
                temp = rng_next(temp);
                let xd = xi as f64;
                if ((zig.fn_[iz] + y * (zig.fn_[iz - 1] - zig.fn_[iz])) as f64)
                    < (-0.5 * xd * xd).exp()
                {
                    x = xi;
                    break;
                }
            }
            *v = x;
        }
        self.state = temp;
    }
}

/// Validates a fill destination and returns its channel count.
fn check_fill_target<T>(dst: &Matrix<T>, what: &str) -> Result<usize> {
    if dst.data.is_empty() || dst.channels == 0 {
        cv_bail!(
            tags::CORE,
            InvalidDimensions,
            "{}: destination matrix is empty",
            what
        );
    }
    if dst.channels > 4 {
        cv_bail!(
            tags::CORE,
            InvalidInput,
            "{}: at most 4 channels are supported, got {}",
            what,
            dst.channels
        );
    }
    Ok(dst.channels)
}

/// The ziggurat tables of `randn_0_1_32f` (128 strips).
struct Ziggurat {
    kn: [u32; 128],
    wn: [f32; 128],
    fn_: [f32; 128],
}

impl Ziggurat {
    /// Builds the tables exactly as OpenCV does on first use.
    fn new() -> Self {
        let mut kn = [0u32; 128];
        let mut wn = [0.0f32; 128];
        let mut fn_ = [0.0f32; 128];
        let m1 = 2_147_483_648.0f64;
        let vn = 9.912_563_035_262_17e-3f64;
        let mut dn = 3.442_619_855_899f64;
        let mut tn = dn;

        let q = vn / (-0.5 * dn * dn).exp();
        kn[0] = ((dn / q) * m1) as u32;
        kn[1] = 0;
        wn[0] = (q / m1) as f32;
        wn[127] = (dn / m1) as f32;
        fn_[0] = 1.0;
        fn_[127] = (-0.5 * dn * dn).exp() as f32;

        for i in (1..=126).rev() {
            dn = (-2.0 * (vn / dn + (-0.5 * dn * dn).exp()).ln()).sqrt();
            kn[i + 1] = ((dn / tn) * m1) as u32;
            tn = dn;
            fn_[i] = (-0.5 * dn * dn).exp() as f32;
            wn[i] = (dn / m1) as f32;
        }
        Self { kn, wn, fn_ }
    }
}

/// Runs `f` with the ziggurat tables: built once per process with `std`,
/// rebuilt per call without it.
fn with_ziggurat<R>(f: impl FnOnce(&Ziggurat) -> R) -> R {
    #[cfg(feature = "std")]
    {
        static TABLES: std::sync::OnceLock<Ziggurat> = std::sync::OnceLock::new();
        f(TABLES.get_or_init(Ziggurat::new))
    }
    #[cfg(not(feature = "std"))]
    {
        f(&Ziggurat::new())
    }
}

// ---------------------------------------------------------------------------
// Xoshiro256** — the opt-in alternative generator (public domain algorithm
// by David Blackman and Sebastiano Vigna).
// ---------------------------------------------------------------------------

/// Internal state for the xoshiro256** generator.
#[cfg(feature = "std")]
#[derive(Clone)]
struct Xoshiro256 {
    s: [u64; 4],
}

#[cfg(feature = "std")]
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
        let result = (self.s[1].wrapping_mul(5)).rotate_left(7).wrapping_mul(9);
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
                let theta = CV_2PI * u2;
                return (r * theta.cos(), r * theta.sin());
            }
        }
    }

    /// `randu` on xoshiro256**: `low + u * (high - low)` per element, rounded
    /// down for integer destinations.
    fn fill_uniform<T>(&mut self, dst: &mut Matrix<T>, low: &Scalar<f64>, high: &Scalar<f64>)
    where
        T: DataType + SaturateCast,
    {
        let cn = dst.channels;
        let is_float = matches!(T::depth(), Depth::CV_32F | Depth::CV_64F);
        for (i, v) in dst.data.iter_mut().enumerate() {
            let c = i % cn;
            let x = low.v[c] + self.next_f64() * (high.v[c] - low.v[c]);
            *v = T::saturate_from_f64(if is_float { x } else { x.floor() });
        }
    }

    /// `randn` on xoshiro256**: Box–Muller pairs, saturated into `T`.
    fn fill_normal<T>(&mut self, dst: &mut Matrix<T>, mean: &Scalar<f64>, std_dev: &Scalar<f64>)
    where
        T: SaturateCast,
    {
        let cn = dst.channels;
        for (k, pair) in dst.data.chunks_mut(2).enumerate() {
            let (g0, g1) = self.next_gaussian_pair();
            for (j, (v, g)) in pair.iter_mut().zip([g0, g1]).enumerate() {
                let c = (2 * k + j) % cn;
                *v = T::saturate_from_f64(mean.v[c] + g * std_dev.v[c]);
            }
        }
    }

    /// Fisher–Yates shuffle.
    fn shuffle<T>(&mut self, slice: &mut [T]) {
        for i in (1..slice.len()).rev() {
            let j = (self.next_f64() * (i + 1) as f64) as usize;
            slice.swap(i, j);
        }
    }
}

// ---------------------------------------------------------------------------
// Thread-local RNG state
// ---------------------------------------------------------------------------

/// The generator behind the thread-local [`randu`], [`randn`] and
/// [`rand_shuffle`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum RngGenerator {
    /// OpenCV's `cv::RNG` ([`Rng`]): the same sequences as OpenCV 4.10.0.
    #[default]
    CvRng,
    /// xoshiro256** seeded through SplitMix64, purecv's generator before
    /// 0.12. Its sequences do not match OpenCV.
    Xoshiro256StarStar,
}

#[cfg(feature = "std")]
enum ThreadRng {
    Cv(Rng),
    Xoshiro(Xoshiro256),
}

#[cfg(feature = "std")]
impl ThreadRng {
    fn new(generator: RngGenerator, seed: u64) -> Self {
        match generator {
            RngGenerator::CvRng => ThreadRng::Cv(Rng::new(seed)),
            RngGenerator::Xoshiro256StarStar => ThreadRng::Xoshiro(Xoshiro256::from_seed(seed)),
        }
    }

    fn generator(&self) -> RngGenerator {
        match self {
            ThreadRng::Cv(_) => RngGenerator::CvRng,
            ThreadRng::Xoshiro(_) => RngGenerator::Xoshiro256StarStar,
        }
    }
}

#[cfg(feature = "std")]
std::thread_local! {
    static THREAD_RNG: RefCell<ThreadRng> = RefCell::new(ThreadRng::new(RngGenerator::CvRng, 0));
}

/// Sets the seed for the thread-local random number generator.
///
/// Mirrors OpenCV's `cv::setRNGSeed`: with the default generator this is
/// `theRNG() = RNG(seed)`, so `0` selects the default state `0xffffffff`.
/// OpenCV takes an `int` and widens it with sign extension, so OpenCV's
/// `setRNGSeed(-1)` is `set_rng_seed(u64::MAX)` here. The seed affects only
/// the calling thread; other threads retain their own independent state.
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
#[cfg(feature = "std")]
pub fn set_rng_seed(seed: u64) {
    THREAD_RNG.with(|rng| {
        let generator = rng.borrow().generator();
        *rng.borrow_mut() = ThreadRng::new(generator, seed);
    });
}

/// Selects the generator of the calling thread and resets it to seed `0`.
///
/// The default is [`RngGenerator::CvRng`]; [`RngGenerator::Xoshiro256StarStar`]
/// keeps purecv's earlier generator for code that relies on its sequences.
/// Later [`set_rng_seed`] calls reseed the selected generator.
///
/// # Example
/// ```
/// use purecv::core::{rng_generator, set_rng_generator, RngGenerator};
///
/// set_rng_generator(RngGenerator::Xoshiro256StarStar);
/// assert_eq!(rng_generator(), RngGenerator::Xoshiro256StarStar);
/// set_rng_generator(RngGenerator::CvRng);
/// ```
#[cfg(feature = "std")]
pub fn set_rng_generator(generator: RngGenerator) {
    THREAD_RNG.with(|rng| *rng.borrow_mut() = ThreadRng::new(generator, 0));
}

/// Returns the generator of the calling thread.
#[cfg(feature = "std")]
pub fn rng_generator() -> RngGenerator {
    THREAD_RNG.with(|rng| rng.borrow().generator())
}

/// Fills a matrix with uniformly distributed random numbers.
///
/// Each element is drawn independently from the interval `[low, high)`.
/// Per-channel bounds are taken from the first `channels` components of the
/// `Scalar` values (matching OpenCV semantics).
///
/// Mirrors OpenCV's `cv::randu`; with the default generator the values are
/// those of [`Rng::fill_uniform`].
///
/// # Arguments
/// * `dst`  - The output matrix. Must already be allocated with the desired
///   size and number of channels (at most 4).
/// * `low`  - Inclusive lower bound for each channel.
/// * `high` - Exclusive upper bound for each channel.
///
/// # Errors
/// Returns `PureCvError::InvalidDimensions` if the matrix is empty and
/// `PureCvError::InvalidInput` if it has more than 4 channels.
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
#[cfg(feature = "std")]
pub fn randu<T>(dst: &mut Matrix<T>, low: Scalar<f64>, high: Scalar<f64>) -> Result<()>
where
    T: DataType + SaturateCast,
{
    check_fill_target(dst, "randu")?;
    THREAD_RNG.with(|rng| match &mut *rng.borrow_mut() {
        ThreadRng::Cv(r) => r.fill_uniform(dst, low, high),
        ThreadRng::Xoshiro(r) => {
            r.fill_uniform(dst, &low, &high);
            Ok(())
        }
    })
}

/// Fills a matrix with normally (Gaussian) distributed random numbers.
///
/// Each element is drawn independently from a Gaussian distribution with
/// the given per-channel `mean` and `std_dev`. Integer destinations are
/// written through [`SaturateCast`]: samples beyond the type's range clamp
/// to its bounds.
///
/// Mirrors OpenCV's `cv::randn`; with the default generator the values are
/// those of [`Rng::fill_normal`].
///
/// # Arguments
/// * `dst`     - The output matrix. Must already be allocated (at most 4
///   channels).
/// * `mean`    - Mean value for each channel.
/// * `std_dev` - Standard deviation for each channel.
///
/// # Errors
/// Returns `PureCvError::InvalidDimensions` if the matrix is empty and
/// `PureCvError::InvalidInput` if it has more than 4 channels.
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
#[cfg(feature = "std")]
pub fn randn<T>(dst: &mut Matrix<T>, mean: Scalar<f64>, std_dev: Scalar<f64>) -> Result<()>
where
    T: DataType + SaturateCast,
{
    check_fill_target(dst, "randn")?;
    THREAD_RNG.with(|rng| match &mut *rng.borrow_mut() {
        ThreadRng::Cv(r) => r.fill_normal(dst, mean, std_dev),
        ThreadRng::Xoshiro(r) => {
            r.fill_normal(dst, &mean, &std_dev);
            Ok(())
        }
    })
}

/// Shuffles a slice randomly using the thread-local RNG.
///
/// Mirrors OpenCV's `cv::randShuffle`: with the default generator this is
/// [`Rng::shuffle`] (OpenCV's swap with `next() % len` for every element);
/// with [`RngGenerator::Xoshiro256StarStar`] it is a Fisher–Yates shuffle.
///
/// # Arguments
/// * `slice` - The slice to shuffle.
#[cfg(feature = "std")]
pub fn rand_shuffle<T>(slice: &mut [T]) {
    if slice.is_empty() {
        return;
    }
    THREAD_RNG.with(|rng| match &mut *rng.borrow_mut() {
        ThreadRng::Cv(r) => r.shuffle(slice),
        ThreadRng::Xoshiro(r) => r.shuffle(slice),
    });
}

#[cfg(all(test, feature = "std"))]
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
            assert!((-5.0..5.0).contains(&v), "value {} out of range [-5, 5)", v);
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

        // All values should be valid u8 values — just verify variation.
        // With 10 000 samples the range should not collapse to a single value.
        let min = *mat.data.iter().min().unwrap();
        let max = *mat.data.iter().max().unwrap();
        assert!(max > min, "randu produced no variation");
    }

    // miri: draws a large sample to check distribution moments — ~88s under
    // interpretation. No `unsafe` on this path. See .agents/MIRI_PLAN.md §4.
    #[cfg_attr(miri, ignore)]
    #[test]
    fn test_randn_statistics() {
        for generator in [RngGenerator::CvRng, RngGenerator::Xoshiro256StarStar] {
            set_rng_generator(generator);
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
                "{generator:?}: mean {} too far from 50",
                mean_val
            );
            assert!(
                (std_val - 10.0).abs() < 1.0,
                "{generator:?}: std {} too far from 10",
                std_val
            );
        }
        set_rng_generator(RngGenerator::CvRng);
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

    #[test]
    fn test_fill_more_than_four_channels_error() {
        let mut mat = Matrix::<u8>::new(2, 2, 5);
        assert!(randu(&mut mat, Scalar::all(0.0), Scalar::all(1.0)).is_err());
        assert!(randn(&mut mat, Scalar::all(0.0), Scalar::all(1.0)).is_err());
    }

    #[test]
    fn test_rng_default_state_and_next() {
        assert_eq!(Rng::default().state, 0xffff_ffff);
        assert_eq!(Rng::new(0), Rng::default());
        let mut rng = Rng::new(1);
        // X1 = 1 * A + 0.
        assert_eq!(rng.next_u32(), CV_RNG_COEFF as u32);
        assert_eq!(rng.state, CV_RNG_COEFF);
        assert_eq!(Rng::new(5).uniform_i32(3, 3), 3);
    }

    #[test]
    fn test_rng_uniform_ranges() {
        let mut rng = Rng::new(42);
        for _ in 0..1000 {
            assert!((-5..5).contains(&rng.uniform_i32(-5, 5)));
            let f = rng.uniform_f32(2.0, 3.0);
            assert!((2.0..=3.0).contains(&f));
            let d = rng.uniform_f64(-1.0, 1.0);
            assert!((-1.0..1.0).contains(&d));
        }
    }

    #[test]
    fn test_xoshiro_generator_is_selectable_and_reseedable() {
        set_rng_generator(RngGenerator::Xoshiro256StarStar);
        set_rng_seed(42);
        assert_eq!(rng_generator(), RngGenerator::Xoshiro256StarStar);
        let mut a = Matrix::<u8>::new(1, 16, 1);
        randu(&mut a, Scalar::all(0.0), Scalar::all(256.0)).unwrap();
        set_rng_seed(42);
        let mut b = Matrix::<u8>::new(1, 16, 1);
        randu(&mut b, Scalar::all(0.0), Scalar::all(256.0)).unwrap();
        assert_eq!(a.data, b.data);

        // randn saturates integer samples on this generator too.
        let mut n = Matrix::<u8>::new(1, 2000, 1);
        randn(&mut n, Scalar::all(128.0), Scalar::all(200.0)).unwrap();
        assert!(n.data.iter().filter(|&&v| v == 255).count() > 300);

        let mut idx: Vec<i32> = (0..10).collect();
        rand_shuffle(&mut idx);
        idx.sort_unstable();
        assert_eq!(idx, (0..10).collect::<Vec<_>>());

        set_rng_generator(RngGenerator::CvRng);
        assert_eq!(rng_generator(), RngGenerator::CvRng);
    }
}

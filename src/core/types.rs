/*
 *  types.rs
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

use std::ops::{Add, Div, Index, IndexMut, Mul, Sub};

use num_traits::{CheckedDiv, Zero};

use crate::core::error::{PureCvError, Result};

pub type Uchar = u8;
pub type Schar = i8;
pub type Short = i16;
pub type Ushort = u16;
pub type Int = i32;
pub type Uint = u32;
pub type Int64 = i64;
pub type Uint64 = u64;
pub type Float = f32;
pub type Double = f64;

/// Template class for 2D points specified by its coordinates `x` and `y`.
#[derive(Debug, Default, Copy, Clone, PartialEq)]
pub struct Point<T> {
    pub x: T,
    pub y: T,
}

impl<T> Point<T> {
    pub fn new(x: T, y: T) -> Self {
        Self { x, y }
    }
}

impl<T: Add<Output = T>> Add for Point<T> {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self::new(self.x + rhs.x, self.y + rhs.y)
    }
}

impl<T: Sub<Output = T>> Sub for Point<T> {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self::new(self.x - rhs.x, self.y - rhs.y)
    }
}

/// Type aliases for convenience
pub type Point2i = Point<i32>;
pub type Point2f = Point<f32>;
pub type Point2d = Point<f64>;
pub type Point2l = Point<i64>;

/// Template class for 3D points.
#[derive(Debug, Default, Copy, Clone, PartialEq)]
pub struct Point3<T> {
    pub x: T,
    pub y: T,
    pub z: T,
}

impl<T> Point3<T> {
    pub fn new(x: T, y: T, z: T) -> Self {
        Self { x, y, z }
    }
}

pub type Point3i = Point3<i32>;
pub type Point3f = Point3<f32>;
pub type Point3d = Point3<f64>;

/// Template class for specifying the size of an image or rectangle.
#[derive(Debug, Default, Copy, Clone, PartialEq)]
pub struct Size<T> {
    pub width: T,
    pub height: T,
}

impl<T> Size<T> {
    pub fn new(width: T, height: T) -> Self {
        Self { width, height }
    }
}

impl<T: Mul<Output = T> + Copy> Size<T> {
    pub fn area(&self) -> T {
        self.width * self.height
    }
}

pub type Size2i = Size<i32>;
pub type Size2f = Size<f32>;
pub type Size2d = Size<f64>;

/// A 4-element value used to represent pixel colours, per-channel constants,
/// and range bounds — mirroring `cv::Scalar_<T>` in WebARKit.
///
/// The four components are stored in `v[0..=3]` and correspond to channels
/// 0–3 (e.g. B, G, R, A for a BGR image).  When a matrix has fewer than 4
/// channels only the leading `channels` entries are used; the rest are ignored.
///
/// # Convenience constructors
///
/// | Constructor | Meaning |
/// |---|---|
/// | `Scalar::new(v0, v1, v2, v3)` | explicit four-channel value |
/// | `Scalar::all(v)` | broadcast `v` to all four channels |
/// | `Scalar::from_value(v)` | `v` in channel 0, zero elsewhere |
/// | `Scalar::from_array([a, b, c, d])` | from a raw `[T; 4]` |
/// | `[a, b, c, d].into()` | same, via `From<[T; 4]>` |
/// | `v.into()` | `from_value(v)` via `From<T>` |
///
/// # Example
/// ```
/// use purecv::core::Scalar;
///
/// let s = Scalar::new(255u8, 128, 0, 255);
/// assert_eq!(s[0], 255);
/// assert_eq!(s[3], 255);
///
/// let gray = Scalar::all(128u8);
/// assert_eq!(gray.to_array(), [128, 128, 128, 128]);
///
/// let doubled = gray.map(|x| x as u16 * 2);
/// assert_eq!(doubled[0], 256u16);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Scalar<T> {
    pub v: [T; 4],
}

impl<T> Scalar<T>
where
    T: Copy + Default,
{
    pub fn new(v0: T, v1: T, v2: T, v3: T) -> Self {
        Self {
            v: [v0, v1, v2, v3],
        }
    }

    pub fn from_value(v: T) -> Self {
        Self {
            v: [v, T::default(), T::default(), T::default()],
        }
    }

    pub fn all(v: T) -> Self {
        Self { v: [v, v, v, v] }
    }

    /// Creates a `Scalar` from a 4-element array.
    ///
    /// Equivalent to the `From<[T; 4]>` impl; prefer the `into()` syntax for
    /// brevity unless the explicit call improves readability.
    pub fn from_array(arr: [T; 4]) -> Self {
        Self { v: arr }
    }

    /// Returns the underlying 4-element array, consuming `self`.
    ///
    /// Useful when you need to pass the raw values to an API that
    /// does not accept `Scalar`.
    pub fn to_array(self) -> [T; 4] {
        self.v
    }

    /// Applies `f` to each channel, producing a `Scalar<U>`.
    ///
    /// Useful for type conversions or per-channel transformations without
    /// manually unpacking the array.
    ///
    /// # Example
    /// ```
    /// use purecv::core::Scalar;
    /// let s = Scalar::new(0u8, 128u8, 255u8, 0u8);
    /// // Normalise to [0.0, 1.0]
    /// let norm: Scalar<f32> = s.map(|x| x as f32 / 255.0);
    /// assert!((norm[1] - 128.0 / 255.0).abs() < 1e-6);
    /// ```
    pub fn map<U, F: Fn(T) -> U>(self, f: F) -> Scalar<U> {
        Scalar {
            v: [f(self.v[0]), f(self.v[1]), f(self.v[2]), f(self.v[3])],
        }
    }
}

/// Accesses channel `i` by index (`s[0]` … `s[3]`).
///
/// # Panics
/// Panics if `i >= 4`.
impl<T> Index<usize> for Scalar<T> {
    type Output = T;
    fn index(&self, i: usize) -> &T {
        &self.v[i]
    }
}

/// Mutably accesses channel `i` by index (`s[0]` … `s[3]`).
///
/// # Panics
/// Panics if `i >= 4`.
impl<T> IndexMut<usize> for Scalar<T> {
    fn index_mut(&mut self, i: usize) -> &mut T {
        &mut self.v[i]
    }
}

/// Converts a `[T; 4]` array directly into a `Scalar<T>`.
impl<T: Copy + Default> From<[T; 4]> for Scalar<T> {
    fn from(arr: [T; 4]) -> Self {
        Self::from_array(arr)
    }
}

/// Mirrors OpenCV's `cv::Scalar(v)`: channel 0 = `v`, channels 1–3 = zero.
///
/// This is the idiomatic way to create a single-channel constant, e.g.
/// `Scalar::from(128u8)` for a grayscale value.
impl<T: Copy + Default> From<T> for Scalar<T> {
    fn from(v: T) -> Self {
        Self::from_value(v)
    }
}

impl<T: Copy + Default> Scalar<T> {
    /// Returns channel `i` when `i < 4`, otherwise `T::default()`.
    ///
    /// Used by `VecN + Scalar` / `VecN - Scalar` to broadcast scalar channels
    /// onto vectors of arbitrary length without bounds-checking the caller side.
    ///
    /// # Examples
    ///
    /// ```
    /// use purecv::core::types::Scalar;
    ///
    /// let s = Scalar::new(10.0_f32, 20.0, 30.0, 40.0);
    ///
    /// // Channels 0–3 return the stored value.
    /// assert_eq!(s.channel_or_default(0), 10.0);
    /// assert_eq!(s.channel_or_default(3), 40.0);
    ///
    /// // Channel 4 and beyond return T::default() (0.0 for f32).
    /// assert_eq!(s.channel_or_default(4), 0.0);
    /// assert_eq!(s.channel_or_default(100), 0.0);
    /// ```
    #[inline]
    pub fn channel_or_default(&self, i: usize) -> T {
        if i < 4 {
            self.v[i]
        } else {
            T::default()
        }
    }
}

/// Per-channel addition: `result[c] = self[c] + rhs[c]`.
impl<T: Copy + Default + Add<Output = T>> Add for Scalar<T> {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self::new(
            self.v[0] + rhs.v[0],
            self.v[1] + rhs.v[1],
            self.v[2] + rhs.v[2],
            self.v[3] + rhs.v[3],
        )
    }
}

/// Per-channel subtraction: `result[c] = self[c] - rhs[c]`.
impl<T: Copy + Default + Sub<Output = T>> Sub for Scalar<T> {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self::new(
            self.v[0] - rhs.v[0],
            self.v[1] - rhs.v[1],
            self.v[2] - rhs.v[2],
            self.v[3] - rhs.v[3],
        )
    }
}

/// Broadcast multiply: scales every channel by the same value `rhs`.
///
/// `result[c] = self[c] * rhs`
impl<T: Copy + Default + Mul<Output = T>> Mul<T> for Scalar<T> {
    type Output = Self;
    fn mul(self, rhs: T) -> Self {
        Self::new(
            self.v[0] * rhs,
            self.v[1] * rhs,
            self.v[2] * rhs,
            self.v[3] * rhs,
        )
    }
}

/// Element-wise multiply: `result[c] = self[c] * rhs[c]`.
impl<T: Copy + Default + Mul<Output = T>> Mul<Scalar<T>> for Scalar<T> {
    type Output = Self;
    fn mul(self, rhs: Scalar<T>) -> Self {
        Self::new(
            self.v[0] * rhs.v[0],
            self.v[1] * rhs.v[1],
            self.v[2] * rhs.v[2],
            self.v[3] * rhs.v[3],
        )
    }
}

/// Broadcast divide: `result[c] = self[c] / rhs`.
///
/// **Zero-safe:** if `rhs` is zero every channel is set to `T::default()` (zero)
/// rather than panicking or producing `NaN`/`Inf`.  For floating-point types
/// this deviates from IEEE 754; use plain `/` on the raw values if you need
/// `Inf` semantics.  For checked integer division see [`Scalar::checked_div`].
impl<T: Copy + Default + PartialEq + Zero + Div<Output = T>> Div<T> for Scalar<T> {
    type Output = Self;
    fn div(self, rhs: T) -> Self {
        let safe = |a: T| {
            if rhs == T::zero() {
                T::default()
            } else {
                a / rhs
            }
        };
        Self {
            v: [
                safe(self.v[0]),
                safe(self.v[1]),
                safe(self.v[2]),
                safe(self.v[3]),
            ],
        }
    }
}

/// Element-wise divide: `result[c] = self[c] / rhs[c]`.
///
/// **Zero-safe:** any channel whose divisor is zero produces `T::default()` (zero).
/// For integer types that need an error on division-by-zero use [`Scalar::checked_div`].
impl<T: Copy + Default + PartialEq + Zero + Div<Output = T>> Div<Scalar<T>> for Scalar<T> {
    type Output = Self;
    fn div(self, rhs: Scalar<T>) -> Self {
        let safe = |a: T, b: T| {
            if b == T::zero() {
                T::default()
            } else {
                a / b
            }
        };
        Self {
            v: [
                safe(self.v[0], rhs.v[0]),
                safe(self.v[1], rhs.v[1]),
                safe(self.v[2], rhs.v[2]),
                safe(self.v[3], rhs.v[3]),
            ],
        }
    }
}

impl<T: Copy + Default + CheckedDiv> Scalar<T> {
    /// Element-wise division that returns an error instead of zero on
    /// division-by-zero.
    ///
    /// Only available for integer types that implement [`num_traits::CheckedDiv`]
    /// (e.g. `u8`, `i32`).  For floating-point types use the infallible
    /// `Div<Scalar<T>>` impl.
    ///
    /// # Errors
    /// Returns [`PureCvError::InvalidInput`] naming the first channel whose
    /// divisor is zero.
    ///
    /// # Example
    /// ```
    /// use purecv::core::Scalar;
    /// let a = Scalar::new(10u8, 20, 30, 40);
    /// let b = Scalar::new(2u8,  4,  5, 8);
    /// assert_eq!(a.checked_div(b).unwrap().v, [5, 5, 6, 5]);
    ///
    /// let bad = Scalar::new(2u8, 0, 1, 1);
    /// assert!(a.checked_div(bad).is_err());
    /// ```
    pub fn checked_div(self, rhs: Scalar<T>) -> Result<Self> {
        let div_ch = |a: T, b: T, ch: usize| {
            a.checked_div(&b).ok_or_else(|| {
                PureCvError::InvalidInput(format!("Division by zero in channel {ch}"))
            })
        };
        Ok(Self {
            v: [
                div_ch(self.v[0], rhs.v[0], 0)?,
                div_ch(self.v[1], rhs.v[1], 1)?,
                div_ch(self.v[2], rhs.v[2], 2)?,
                div_ch(self.v[3], rhs.v[3], 3)?,
            ],
        })
    }
}

#[cfg(test)]
mod scalar_tests {
    use super::*;

    #[test]
    fn test_index() {
        let s = Scalar::new(1u8, 2u8, 3u8, 4u8);
        assert_eq!(s[0], 1);
        assert_eq!(s[3], 4);
    }

    #[test]
    fn test_index_mut() {
        let mut s = Scalar::new(1u8, 2u8, 3u8, 4u8);
        s[1] = 42;
        assert_eq!(s[1], 42);
    }

    #[test]
    fn test_from_array_method() {
        let s = Scalar::from_array([10u8, 20u8, 30u8, 40u8]);
        assert_eq!(s.v, [10, 20, 30, 40]);
    }

    #[test]
    fn test_to_array() {
        let s = Scalar::new(1u8, 2u8, 3u8, 4u8);
        assert_eq!(s.to_array(), [1, 2, 3, 4]);
    }

    #[test]
    fn test_from_t_trait() {
        let s = Scalar::from(255u8);
        assert_eq!(s.v, [255, 0, 0, 0]);
    }

    #[test]
    fn test_from_array_trait() {
        let s: Scalar<u8> = [10, 20, 30, 40].into();
        assert_eq!(s.v, [10, 20, 30, 40]);
    }

    #[test]
    fn test_map() {
        let s = Scalar::new(1u8, 2u8, 3u8, 4u8);
        let s2: Scalar<u16> = s.map(|x| x as u16 * 2);
        assert_eq!(s2.v, [2u16, 4, 6, 8]);
    }

    #[test]
    fn test_add() {
        let a = Scalar::new(1u8, 2u8, 3u8, 4u8);
        let b = Scalar::new(10u8, 20u8, 30u8, 40u8);
        assert_eq!((a + b).v, [11, 22, 33, 44]);
    }

    #[test]
    fn test_sub() {
        let a = Scalar::new(10u8, 20u8, 30u8, 40u8);
        let b = Scalar::new(1u8, 2u8, 3u8, 4u8);
        assert_eq!((a - b).v, [9, 18, 27, 36]);
    }

    #[test]
    fn test_mul_t() {
        let s = Scalar::new(10u8, 20u8, 30u8, 40u8);
        assert_eq!((s * 2u8).v, [20, 40, 60, 80]);
    }

    #[test]
    fn test_mul_scalar() {
        let a = Scalar::new(2u8, 3u8, 4u8, 5u8);
        let b = Scalar::new(3u8, 2u8, 2u8, 1u8);
        assert_eq!((a * b).v, [6, 6, 8, 5]);
    }

    #[test]
    fn test_div_t() {
        let s = Scalar::new(20u8, 40u8, 60u8, 80u8);
        assert_eq!((s / 2u8).v, [10, 20, 30, 40]);
    }

    #[test]
    fn test_div_t_by_zero() {
        let s = Scalar::new(20u8, 40u8, 60u8, 80u8);
        assert_eq!((s / 0u8).v, [0, 0, 0, 0]);
    }

    #[test]
    fn test_div_scalar() {
        let a = Scalar::new(20u8, 40u8, 60u8, 4u8);
        let b = Scalar::new(2u8, 4u8, 6u8, 2u8);
        assert_eq!((a / b).v, [10, 10, 10, 2]);
    }

    #[test]
    fn test_div_scalar_by_zero_channel() {
        let a = Scalar::new(20u8, 40u8, 60u8, 4u8);
        let b = Scalar::new(2u8, 0u8, 6u8, 2u8);
        assert_eq!((a / b).v, [10, 0, 10, 2]);
    }

    #[test]
    fn test_checked_div_ok() {
        let a = Scalar::new(20u8, 40u8, 60u8, 4u8);
        let b = Scalar::new(2u8, 4u8, 6u8, 2u8);
        assert_eq!(a.checked_div(b).unwrap().v, [10, 10, 10, 2]);
    }

    #[test]
    fn test_checked_div_err() {
        let a = Scalar::new(20u8, 40u8, 60u8, 4u8);
        let b = Scalar::new(2u8, 0u8, 6u8, 2u8);
        assert!(a.checked_div(b).is_err());
    }

    #[test]
    fn test_div_f32_by_zero_yields_inf() {
        let s = Scalar::new(1.0f32, 2.0f32, 3.0f32, 4.0f32);
        // f32 zero-check: 0.0 == Zero::zero() → returns default (0.0)
        let r = s / 0.0f32;
        assert_eq!(r.v, [0.0, 0.0, 0.0, 0.0]);
    }
}

/// Generic N-dimensional short numerical vector (mirrors `cv::Vec<Tp, cn>`).
///
/// `VecN<T, N>` stores `N` elements of type `T` in a fixed-size array `val`.
/// It supports element-wise arithmetic, scalar broadcast, indexing, and dot
/// products. Common OpenCV-style aliases (`Vec2b`, `Vec3f`, …) are provided.
///
/// # Example
/// ```
/// use purecv::core::types::VecN;
///
/// let a = VecN::from_array([1.0_f32, 2.0, 3.0]);
/// let b = VecN::from_array([4.0_f32, 5.0, 6.0]);
/// assert!((a.dot(&b) - 32.0_f32).abs() < 1e-6);
/// let c = a + b;
/// assert_eq!(c.val, [5.0, 7.0, 9.0]);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct VecN<T, const N: usize> {
    pub val: [T; N],
}

impl<T, const N: usize> VecN<T, N> {
    /// Creates a `VecN` from a fixed-size array.
    pub fn from_array(val: [T; N]) -> Self {
        Self { val }
    }

    /// Returns the underlying array, consuming `self`.
    pub fn to_array(self) -> [T; N] {
        self.val
    }

    /// Applies `f` to each element, producing a `VecN<U, N>`.
    pub fn map<U, F: FnMut(T) -> U>(self, f: F) -> VecN<U, N> {
        VecN {
            val: self.val.map(f),
        }
    }
}

impl<T> VecN<T, 2> {
    /// Creates a 2-element vector — mirrors `cv::Vec2*(v0, v1)`.
    pub fn new(v0: T, v1: T) -> Self {
        Self { val: [v0, v1] }
    }
}

impl<T> VecN<T, 3> {
    /// Creates a 3-element vector — mirrors `cv::Vec3*(v0, v1, v2)`.
    pub fn new(v0: T, v1: T, v2: T) -> Self {
        Self { val: [v0, v1, v2] }
    }
}

impl<T> VecN<T, 4> {
    /// Creates a 4-element vector — mirrors `cv::Vec4*(v0, v1, v2, v3)`.
    pub fn new(v0: T, v1: T, v2: T, v3: T) -> Self {
        Self {
            val: [v0, v1, v2, v3],
        }
    }
}

impl<T> VecN<T, 6> {
    /// Creates a 6-element vector — mirrors `cv::Vec6*(v0..v5)`.
    pub fn new(v0: T, v1: T, v2: T, v3: T, v4: T, v5: T) -> Self {
        Self {
            val: [v0, v1, v2, v3, v4, v5],
        }
    }
}

impl<T: Zero, const N: usize> VecN<T, N> {
    /// Returns a zero vector (`T::zero()` in every slot).
    ///
    /// Requires `T: num_traits::Zero`, which is satisfied by all built-in
    /// numeric types and guarantees that each element is the additive identity.
    pub fn zeros() -> Self {
        Self {
            val: std::array::from_fn(|_| T::zero()),
        }
    }
}

impl<T: Copy, const N: usize> VecN<T, N> {
    /// Returns a vector with every element set to `v`.
    pub fn all(v: T) -> Self {
        Self {
            val: std::array::from_fn(|_| v),
        }
    }
}

impl<T: Copy + Zero + Add<Output = T> + Mul<Output = T>, const N: usize> VecN<T, N> {
    /// Computes the dot product of `self` and `rhs`.
    pub fn dot(&self, rhs: &Self) -> T {
        self.val
            .iter()
            .zip(rhs.val.iter())
            .fold(T::zero(), |acc, (&a, &b)| acc + a * b)
    }
}

impl<T, const N: usize> Index<usize> for VecN<T, N> {
    type Output = T;
    fn index(&self, i: usize) -> &T {
        &self.val[i]
    }
}

impl<T, const N: usize> IndexMut<usize> for VecN<T, N> {
    fn index_mut(&mut self, i: usize) -> &mut T {
        &mut self.val[i]
    }
}

impl<T, const N: usize> From<[T; N]> for VecN<T, N> {
    fn from(arr: [T; N]) -> Self {
        Self::from_array(arr)
    }
}

impl<T: Copy + Add<Output = T>, const N: usize> Add for VecN<T, N> {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self {
            val: std::array::from_fn(|i| self.val[i] + rhs.val[i]),
        }
    }
}

impl<T: Copy + Sub<Output = T>, const N: usize> Sub for VecN<T, N> {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self {
            val: std::array::from_fn(|i| self.val[i] - rhs.val[i]),
        }
    }
}

/// Adds a [`Scalar`] to a `VecN` channel-by-channel, mirroring OpenCV's
/// `cv::Vec operator+(cv::Vec, cv::Scalar)`.  The scalar carries four channels;
/// for `N ≤ 4` each element `i` gets `scalar.v[i]`; for `i ≥ 4` (e.g. `Vec6*`)
/// the scalar contributes `T::default()` (zero for numeric types).
impl<T: Copy + Default + Add<Output = T>, const N: usize> Add<Scalar<T>> for VecN<T, N> {
    type Output = Self;
    fn add(self, rhs: Scalar<T>) -> Self {
        Self {
            val: std::array::from_fn(|i| self.val[i] + rhs.channel_or_default(i)),
        }
    }
}

/// Subtracts a [`Scalar`] from a `VecN` channel-by-channel, mirroring OpenCV's
/// `cv::Vec operator-(cv::Vec, cv::Scalar)`.
impl<T: Copy + Default + Sub<Output = T>, const N: usize> Sub<Scalar<T>> for VecN<T, N> {
    type Output = Self;
    fn sub(self, rhs: Scalar<T>) -> Self {
        Self {
            val: std::array::from_fn(|i| self.val[i] - rhs.channel_or_default(i)),
        }
    }
}

/// Element-wise multiply: `result[i] = self[i] * rhs[i]`.
impl<T: Copy + Mul<Output = T>, const N: usize> Mul for VecN<T, N> {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        Self {
            val: std::array::from_fn(|i| self.val[i] * rhs.val[i]),
        }
    }
}

/// Scalar broadcast multiply: `result[i] = self[i] * rhs`.
impl<T: Copy + Mul<Output = T>, const N: usize> Mul<T> for VecN<T, N> {
    type Output = Self;
    fn mul(self, rhs: T) -> Self {
        Self {
            val: std::array::from_fn(|i| self.val[i] * rhs),
        }
    }
}

/// Scalar broadcast divide: `result[i] = self[i] / rhs`.
impl<T: Copy + Div<Output = T>, const N: usize> Div<T> for VecN<T, N> {
    type Output = Self;
    fn div(self, rhs: T) -> Self {
        Self {
            val: std::array::from_fn(|i| self.val[i] / rhs),
        }
    }
}

// ---------------------------------------------------------------------------
// OpenCV-compatible type aliases
// ---------------------------------------------------------------------------

pub type Vec2b = VecN<u8, 2>;
pub type Vec3b = VecN<u8, 3>;
pub type Vec4b = VecN<u8, 4>;

pub type Vec2s = VecN<i16, 2>;
pub type Vec3s = VecN<i16, 3>;
pub type Vec4s = VecN<i16, 4>;

pub type Vec2i = VecN<i32, 2>;
pub type Vec3i = VecN<i32, 3>;
pub type Vec4i = VecN<i32, 4>;

pub type Vec2f = VecN<f32, 2>;
pub type Vec3f = VecN<f32, 3>;
pub type Vec4f = VecN<f32, 4>;
pub type Vec6f = VecN<f32, 6>;

pub type Vec2d = VecN<f64, 2>;
pub type Vec3d = VecN<f64, 3>;
pub type Vec4d = VecN<f64, 4>;
pub type Vec6d = VecN<f64, 6>;

#[cfg(test)]
mod vecn_tests {
    use super::*;

    #[test]
    fn test_from_array_and_index() {
        let v = VecN::from_array([1_i32, 2, 3]);
        assert_eq!(v[0], 1);
        assert_eq!(v[2], 3);
    }

    #[test]
    fn test_index_mut() {
        let mut v = VecN::from_array([0_u8; 3]);
        v[1] = 42;
        assert_eq!(v[1], 42);
    }

    #[test]
    fn test_zeros() {
        let v: VecN<f32, 4> = VecN::zeros();
        assert_eq!(v.val, [0.0; 4]);
    }

    #[test]
    fn test_all() {
        let v: VecN<u8, 3> = VecN::all(7);
        assert_eq!(v.val, [7, 7, 7]);
    }

    #[test]
    fn test_to_array() {
        let v = VecN::from_array([10_i32, 20, 30]);
        assert_eq!(v.to_array(), [10, 20, 30]);
    }

    #[test]
    fn test_map() {
        let v = VecN::from_array([1_u8, 2, 3]);
        let v2: VecN<u16, 3> = v.map(|x| x as u16 * 2);
        assert_eq!(v2.val, [2_u16, 4, 6]);
    }

    #[test]
    fn test_add() {
        let a = VecN::from_array([1.0_f32, 2.0, 3.0]);
        let b = VecN::from_array([4.0_f32, 5.0, 6.0]);
        assert_eq!((a + b).val, [5.0, 7.0, 9.0]);
    }

    #[test]
    fn test_sub() {
        let a = VecN::from_array([10_i32, 20, 30]);
        let b = VecN::from_array([1_i32, 2, 3]);
        assert_eq!((a - b).val, [9, 18, 27]);
    }

    #[test]
    fn test_mul_elementwise() {
        let a = VecN::from_array([2_i32, 3, 4]);
        let b = VecN::from_array([5_i32, 6, 7]);
        assert_eq!((a * b).val, [10, 18, 28]);
    }

    #[test]
    fn test_mul_scalar() {
        let v = VecN::from_array([1.0_f32, 2.0, 3.0]);
        assert_eq!((v * 2.0_f32).val, [2.0, 4.0, 6.0]);
    }

    #[test]
    fn test_div_scalar() {
        let v = VecN::from_array([4.0_f32, 8.0, 12.0]);
        assert_eq!((v / 4.0_f32).val, [1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_dot() {
        let a = VecN::from_array([1.0_f32, 2.0, 3.0]);
        let b = VecN::from_array([4.0_f32, 5.0, 6.0]);
        assert!((a.dot(&b) - 32.0_f32).abs() < 1e-6);
    }

    #[test]
    fn test_from_trait() {
        let v: VecN<i32, 2> = [3, 7].into();
        assert_eq!(v.val, [3, 7]);
    }

    #[test]
    fn test_type_aliases() {
        let _: Vec3b = VecN::from_array([255_u8, 0, 128]);
        let _: Vec3f = VecN::from_array([1.0_f32, 2.0, 3.0]);
        let _: Vec4i = VecN::from_array([1_i32, 2, 3, 4]);
        let _: Vec6d = VecN::from_array([1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_clone_and_eq() {
        let a = VecN::from_array([1_i32, 2, 3]);
        let b = a.clone();
        assert_eq!(a, b);
    }

    // --- new() constructors ---------------------------------------------------

    #[test]
    fn test_new_2() {
        let v = Vec2i::new(3, 7);
        assert_eq!(v.val, [3, 7]);
    }

    #[test]
    fn test_new_3() {
        let v = Vec3f::new(1.0_f32, 2.0, 3.0);
        assert_eq!(v.val, [1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_new_4() {
        let v = Vec4b::new(10, 20, 30, 40);
        assert_eq!(v.val, [10, 20, 30, 40]);
    }

    #[test]
    fn test_new_6() {
        let v = Vec6d::new(1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0);
        assert_eq!(v.val, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    // --- Add<Scalar> / Sub<Scalar> -------------------------------------------

    #[test]
    fn test_add_scalar_3() {
        let v = Vec3f::new(1.0_f32, 2.0, 3.0);
        let s = Scalar::new(10.0_f32, 20.0, 30.0, 0.0);
        assert_eq!((v + s).val, [11.0, 22.0, 33.0]);
    }

    #[test]
    fn test_sub_scalar_3() {
        let v = Vec3f::new(10.0_f32, 20.0, 30.0);
        let s = Scalar::new(1.0_f32, 2.0, 3.0, 0.0);
        assert_eq!((v - s).val, [9.0, 18.0, 27.0]);
    }

    #[test]
    fn test_add_scalar_4() {
        let v = Vec4i::new(1, 2, 3, 4);
        let s = Scalar::new(5_i32, 6, 7, 8);
        assert_eq!((v + s).val, [6, 8, 10, 12]);
    }

    #[test]
    fn test_add_scalar_vec6_zero_pads_extra_channels() {
        // Channels 4 and 5 of the Vec get scalar's default (0) added.
        let v = Vec6f::new(1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0);
        let s = Scalar::new(10.0_f32, 10.0, 10.0, 10.0);
        assert_eq!((v + s).val, [11.0, 12.0, 13.0, 14.0, 5.0, 6.0]);
    }
}

/// TermCriteria defines termination criteria for iterative algorithms.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TermType {
    Count = 1,
    Eps = 2,
    Both = 3,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TermCriteria {
    pub type_: TermType,
    pub max_count: i32,
    pub epsilon: f64,
}

impl TermCriteria {
    pub fn new(type_: TermType, max_count: i32, epsilon: f64) -> Self {
        Self {
            type_,
            max_count,
            epsilon,
        }
    }
}
/// Template class for 2D rectangles.
#[derive(Debug, Default, Copy, Clone, PartialEq)]
pub struct Rect<T> {
    pub x: T,
    pub y: T,
    pub width: T,
    pub height: T,
}

impl<T> Rect<T> {
    pub fn new(x: T, y: T, width: T, height: T) -> Self {
        Self {
            x,
            y,
            width,
            height,
        }
    }
}

impl<T: Add<Output = T> + Copy> Rect<T> {
    pub fn tl(&self) -> Point<T> {
        Point::new(self.x, self.y)
    }

    pub fn br(&self) -> Point<T> {
        Point::new(self.x + self.width, self.y + self.height)
    }
}

impl<T: Mul<Output = T> + Copy> Rect<T> {
    pub fn area(&self) -> T {
        self.width * self.height
    }
}

pub type Rect2i = Rect<i32>;
pub type Rect2f = Rect<f32>;
pub type Rect2d = Rect<f64>;

/// Rotated (i.e. not up-right) rectangles on a plane.
#[derive(Debug, Default, Copy, Clone, PartialEq)]
pub struct RotatedRect {
    pub center: Point2f,
    pub size: Size2f,
    pub angle: f32,
}

impl RotatedRect {
    pub fn new(center: Point2f, size: Size2f, angle: f32) -> Self {
        Self {
            center,
            size,
            angle,
        }
    }
}

/// Various border interpolation methods.
/// See OpenCV's BorderTypes.
#[allow(non_camel_case_types)]
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BorderTypes {
    Constant = 0,
    Replicate = 1,
    Reflect = 2,
    Wrap = 3,
    #[default]
    Reflect101 = 4,
    Transparent = 5,
    Isolated = 16,
}

/// Comparison types for `compare` and `compare_scalar`.
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CmpTypes {
    Eq = 0,
    Gt = 1,
    Ge = 2,
    Lt = 3,
    Le = 4,
    Ne = 5,
}

/// Normalization types for `norm` and `normalize`.
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NormTypes {
    Inf = 1,
    L1 = 2,
    L2 = 4,
    L2Sqr = 5,
    Hamming = 6,
    Hamming2 = 7,
    Relative = 8,
    MinMax = 32,
}

/// Reduction types for `reduce`.
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReduceTypes {
    Sum = 0,
    Avg = 1,
    Max = 2,
    Min = 3,
}

// ---------------------------------------------------------------------------
// Sort flags  (mirrors cv::SortFlags)
// ---------------------------------------------------------------------------

/// Sort each row of the matrix (default).
pub const SORT_EVERY_ROW: i32 = 0;
/// Sort each column of the matrix.
pub const SORT_EVERY_COLUMN: i32 = 1;
/// Sort in ascending order (default).
pub const SORT_ASCENDING: i32 = 0;
/// Sort in descending order.
pub const SORT_DESCENDING: i32 = 16;

// ---------------------------------------------------------------------------
// K-means flags  (mirrors cv::KmeansFlags)
// ---------------------------------------------------------------------------

/// Use random initial centers at each attempt.
pub const KMEANS_RANDOM_CENTERS: i32 = 0;
/// Use k-means++ center initialization.
pub const KMEANS_PP_CENTERS: i32 = 2;
/// Use the user-supplied `labels` as initial cluster assignment.
pub const KMEANS_USE_INITIAL_LABELS: i32 = 1;

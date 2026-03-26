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

/// Scalar represents a 4-element vector.
///
/// It is widely used in OpenCV to pass pixel values and for range checks.
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
    pub fn from_array(arr: [T; 4]) -> Self {
        Self { v: arr }
    }

    /// Returns the underlying 4-element array.
    pub fn to_array(self) -> [T; 4] {
        self.v
    }

    /// Transforms each channel using the given closure, producing a `Scalar<U>`.
    pub fn map<U, F: Fn(T) -> U>(self, f: F) -> Scalar<U> {
        Scalar {
            v: [f(self.v[0]), f(self.v[1]), f(self.v[2]), f(self.v[3])],
        }
    }
}

impl<T> Index<usize> for Scalar<T> {
    type Output = T;
    fn index(&self, i: usize) -> &T {
        &self.v[i]
    }
}

impl<T> IndexMut<usize> for Scalar<T> {
    fn index_mut(&mut self, i: usize) -> &mut T {
        &mut self.v[i]
    }
}

impl<T: Copy + Default> From<[T; 4]> for Scalar<T> {
    fn from(arr: [T; 4]) -> Self {
        Self::from_array(arr)
    }
}

/// Mirrors OpenCV's `cv::Scalar(v)`: first channel = `v`, rest = zero.
impl<T: Copy + Default> From<T> for Scalar<T> {
    fn from(v: T) -> Self {
        Self::from_value(v)
    }
}

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

/// Multiplies each channel by a scalar value.
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

/// Element-wise multiplication of two `Scalar`s.
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

/// Divides each channel by a scalar value.
/// Division by zero returns `T::default()` (zero) instead of panicking.
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
            v: [safe(self.v[0]), safe(self.v[1]), safe(self.v[2]), safe(self.v[3])],
        }
    }
}

/// Element-wise division of two `Scalar`s.
/// Per-channel division by zero returns `T::default()` (zero) instead of panicking.
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
    /// Divides element-wise, returning an error if any divisor channel is zero.
    /// Available for integer types only (via `num_traits::CheckedDiv`).
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

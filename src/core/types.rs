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

use std::ops::{Add, Mul, Sub};

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
        Self { v: [v0, v1, v2, v3] }
    }

    pub fn from_value(v: T) -> Self {
        Self { v: [v, T::default(), T::default(), T::default()] }
    }

    pub fn all(v: T) -> Self {
        Self { v: [v, v, v, v] }
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
    CONSTANT = 0,
    REPLICATE = 1,
    REFLECT = 2,
    WRAP = 3,
    #[default]
    REFLECT_101 = 4,
    TRANSPARENT = 5,
    ISOLATED = 16,
}

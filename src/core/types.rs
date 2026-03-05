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
 *  WebARKitLib-rs is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU Lesser General Public License for more details.
 *
 *  You should have received a copy of the GNU Lesser General Public License
 *  along with WebARKitLib-rs.  If not, see <http://www.gnu.org/licenses/>.
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

use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Sub, SubAssign};

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

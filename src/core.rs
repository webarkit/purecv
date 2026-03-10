/*
 *  core.rs
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

pub mod matrix;
pub mod arithm;
pub mod types;
pub mod utils;
pub mod structural;
pub mod norm;
pub mod stats;
pub mod error;

#[cfg(test)]
mod tests;

// Re-exports for easier access
pub use self::arithm::{bitwise_and, bitwise_not, bitwise_or, bitwise_xor};
pub use self::error::{PureCvError, Result};
pub use self::matrix::Matrix;
pub use self::norm::{norm, normalize, NormTypes};
pub use self::stats::{mean, mean_std_dev, min_max_loc, sum};
pub use self::types::{
    BorderTypes, Point, Point2d, Point2f, Point2i, Point2l,
    Point3, Point3d, Point3f, Point3i, Rect, Rect2d, Rect2f, Rect2i, RotatedRect,
    Scalar, Size, Size2d, Size2f, Size2i, TermCriteria, TermType,
};
#[cfg(not(feature = "parallel"))]
pub use self::utils::ParIterFallback;
pub use self::utils::border_interpolate;
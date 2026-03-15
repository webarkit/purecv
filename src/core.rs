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

pub mod arithm;
pub mod error;
pub mod matrix;
pub mod rng;
pub mod structural;
pub mod types;
pub mod utils;

#[cfg(test)]
mod tests;

// Re-exports for easier access
pub use self::arithm::{
    absdiff, bitwise_and, bitwise_not, bitwise_or, bitwise_xor, cart_to_polar, check_range,
    count_non_zero, cross, determinant, dot, gemm, invert, magnitude, mean, mean_std_dev,
    min_max_loc, norm, normalize, perspective_transform, phase, polar_to_cart, reduce,
    set_identity, solve, sum, trace, transform, DecompTypes, GEMM_1_T, GEMM_2_T, GEMM_3_T,
};
pub use self::error::{PureCvError, Result};
pub use self::matrix::{
    DataType, Depth, MatType, Matrix, CV_16S, CV_16SC1, CV_16SC2, CV_16SC3, CV_16SC4, CV_16U,
    CV_16UC1, CV_16UC2, CV_16UC3, CV_16UC4, CV_32F, CV_32FC1, CV_32FC2, CV_32FC3, CV_32FC4, CV_32S,
    CV_32SC1, CV_32SC2, CV_32SC3, CV_32SC4, CV_64F, CV_64FC1, CV_64FC2, CV_64FC3, CV_64FC4, CV_8S,
    CV_8SC1, CV_8SC2, CV_8SC3, CV_8SC4, CV_8U, CV_8UC1, CV_8UC2, CV_8UC3, CV_8UC4,
};
pub use self::rng::{randn, randu, set_rng_seed};
pub use self::types::{
    BorderTypes, NormTypes, Point, Point2d, Point2f, Point2i, Point2l, Point3, Point3d, Point3f,
    Point3i, Rect, Rect2d, Rect2f, Rect2i, ReduceTypes, RotatedRect, Scalar, Size, Size2d, Size2f,
    Size2i, TermCriteria, TermType,
};
#[cfg(not(feature = "parallel"))]
pub use self::utils::ParIterFallback;
pub use self::utils::{border_interpolate, get_tick_count, get_tick_frequency};

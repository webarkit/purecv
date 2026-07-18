/*
 *  core.rs
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

pub mod arithm;
pub mod constants;
#[cfg(feature = "transforms")]
pub mod dct;
#[cfg(feature = "fft")]
pub mod dft;
pub mod dynamic;
pub mod error;
pub mod logging;
pub mod matrix;
pub mod metrics;
pub mod rng;
pub(crate) mod simd;
pub mod solvers;
pub mod structural;
pub mod types;
pub mod utils;

#[macro_use]
pub mod macros {
    // Re-export macro if we need to put it somewhere, or just trust #macro_export handles it
}

#[cfg(test)]
mod tests;

// Re-exports for easier access
pub use self::arithm::{
    absdiff, add, bitwise_and, bitwise_not, bitwise_or, bitwise_xor, cart_to_polar, check_range,
    count_non_zero, cross, determinant, divide, dot, gemm, has_no_zero, invert, kmeans, lut,
    magnitude, mean, mean_std_dev, min_max_loc, multiply, norm, normalize, perspective_transform,
    phase, polar_to_cart, reduce, set_identity, solve, solve_poly, sort, sort_idx, subtract, sum,
    trace, transform, DecompTypes, GEMM_1_T, GEMM_2_T, GEMM_3_T,
};
pub use self::constants::{
    CV_2PI, CV_E, CV_LN10, CV_LN2, CV_LOG2, CV_PI, CV_PI_2, CV_PI_4, CV_SQRT2,
};
#[cfg(feature = "transforms")]
pub use self::dct::{dct, idct};
#[cfg(feature = "fft")]
pub use self::dft::{
    dft, get_optimal_dft_size, idft, DftFloat, DFT_COMPLEX_OUTPUT, DFT_INVERSE, DFT_REAL_OUTPUT,
    DFT_ROWS, DFT_SCALE,
};
pub use self::dynamic::{DynamicData, DynamicMatrix};
pub use self::error::{PureCvError, Result};
pub use self::logging::{get_log_level, init_basic_logger, set_log_level, LogLevel};
pub use self::matrix::{
    DataType, Depth, MatType, Matrix, CV_16S, CV_16SC1, CV_16SC2, CV_16SC3, CV_16SC4, CV_16U,
    CV_16UC1, CV_16UC2, CV_16UC3, CV_16UC4, CV_32F, CV_32FC1, CV_32FC2, CV_32FC3, CV_32FC4, CV_32S,
    CV_32SC1, CV_32SC2, CV_32SC3, CV_32SC4, CV_64F, CV_64FC1, CV_64FC2, CV_64FC3, CV_64FC4, CV_8S,
    CV_8SC1, CV_8SC2, CV_8SC3, CV_8SC4, CV_8U, CV_8UC1, CV_8UC2, CV_8UC3, CV_8UC4,
};
pub use self::metrics::{mahalanobis, psnr};
pub use self::rng::{rand_shuffle, randn, randu, set_rng_seed};
pub use self::solvers::{solve_cubic, solve_quadratic};
pub use self::types::{
    BorderTypes, NormTypes, Point, Point2d, Point2f, Point2i, Point2l, Point3, Point3d, Point3f,
    Point3i, Rect, Rect2d, Rect2f, Rect2i, ReduceTypes, RotatedRect, Scalar, Size, Size2d, Size2f,
    Size2i, TermCriteria, TermType, Vec2b, Vec2d, Vec2f, Vec2i, Vec2s, Vec3b, Vec3d, Vec3f, Vec3i,
    Vec3s, Vec4b, Vec4d, Vec4f, Vec4i, Vec4s, Vec6d, Vec6f, VecN, KMEANS_PP_CENTERS,
    KMEANS_RANDOM_CENTERS, KMEANS_USE_INITIAL_LABELS, SORT_ASCENDING, SORT_DESCENDING,
    SORT_EVERY_COLUMN, SORT_EVERY_ROW,
};
#[cfg(not(feature = "parallel"))]
pub use self::utils::ParIterFallback;
pub use self::utils::{border_interpolate, get_tick_count, get_tick_frequency};

/*
 *  geometric.rs
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

use alloc::{string::ToString, vec};
#[allow(unused_imports)]
use num_traits::Float;

use crate::core::arithm::{invert, DecompTypes};
use crate::core::error::{PureCvError, Result};
use crate::core::simd::SimdElement;
use crate::core::types::{BorderTypes, Scalar, Size2i};
use crate::core::utils::border_interpolate;
use crate::core::Matrix;
use num_traits::{FromPrimitive, ToPrimitive};

#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// Interpolation methods.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InterpolationFlags {
    Nearest = 0,
    Linear = 1,
}

/// Applies a generic geometrical transformation to an image.
///
/// The function `remap` transforms the source image using the specified map:
/// `dst(x,y) = src(map1(x,y), map2(x,y))`
///
/// # Arguments
///
/// * `src` - Source image as a `Matrix<T>`.
/// * `map1` - The first map of either `(x,y)` points or just `x` values, with type `f32`.
/// * `map2` - The second map of `y` values with type `f32`.
/// * `interpolation` - Interpolation method to use (e.g., `InterpolationFlags::Linear`).
/// * `border_mode` - Pixel extrapolation method (e.g., `BorderTypes::Constant`).
/// * `border_value` - Value used in case of a constant border.
///
/// # Returns
///
/// Returns a `Result<Matrix<T>>` containing the transformed image.
///
/// # Errors
///
/// Returns an error if:
/// * `map1` and `map2` do not have the same dimensions.
/// * `map1` or `map2` are not single-channel matrices.
///
/// # Examples
///
/// ```
/// use purecv::core::{Matrix, types::{BorderTypes, Scalar}};
/// use purecv::imgproc::geometric::{remap, InterpolationFlags};
///
/// let src = Matrix::<u8>::new(10, 10, 1);
/// let map1 = Matrix::<f32>::new(10, 10, 1);
/// let map2 = Matrix::<f32>::new(10, 10, 1);
///
/// let result = remap(
///     &src,
///     &map1,
///     &map2,
///     InterpolationFlags::Linear,
///     BorderTypes::Constant,
///     Scalar::all(0)
/// ).unwrap();
/// ```
pub fn remap<T>(
    src: &Matrix<T>,
    map1: &Matrix<f32>,
    map2: &Matrix<f32>,
    interpolation: InterpolationFlags,
    border_mode: BorderTypes,
    border_value: Scalar<T>,
) -> Result<Matrix<T>>
where
    T: Default + Clone + Copy + ToPrimitive + FromPrimitive + Send + Sync + SimdElement,
{
    if map1.rows != map2.rows || map1.cols != map2.cols {
        return Err(PureCvError::InvalidInput(
            "map1 and map2 must have the same dimensions".to_string(),
        ));
    }
    if map1.channels != 1 || map2.channels != 1 {
        return Err(PureCvError::InvalidInput(
            "map1 and map2 must be single-channel matrices".to_string(),
        ));
    }

    let rows = map1.rows;
    let cols = map1.cols;
    let channels = src.channels;

    let mut dst = Matrix::<T>::new(rows, cols, channels);

    #[cfg(feature = "parallel")]
    {
        dst.data
            .par_chunks_exact_mut(cols * channels)
            .enumerate()
            .for_each(|(r, row_data)| {
                let map1_row = &map1.data[r * cols..(r + 1) * cols];
                let map2_row = &map2.data[r * cols..(r + 1) * cols];

                let handled = match interpolation {
                    InterpolationFlags::Nearest => T::simd_remap_nearest_row(
                        row_data,
                        &src.data,
                        src.cols,
                        src.rows,
                        channels,
                        map1_row,
                        map2_row,
                        border_mode,
                        border_value,
                    ),
                    InterpolationFlags::Linear => T::simd_remap_bilinear_row(
                        row_data,
                        &src.data,
                        src.cols,
                        src.rows,
                        channels,
                        map1_row,
                        map2_row,
                        border_mode,
                        border_value,
                    ),
                };

                if !handled {
                    remap_row_scalar(
                        row_data,
                        &src.data,
                        src.cols,
                        src.rows,
                        channels,
                        map1_row,
                        map2_row,
                        interpolation,
                        border_mode,
                        border_value,
                    );
                }
            });
    }

    #[cfg(not(feature = "parallel"))]
    {
        for r in 0..rows {
            let row_data = &mut dst.data[r * cols * channels..(r + 1) * cols * channels];
            let map1_row = &map1.data[r * cols..(r + 1) * cols];
            let map2_row = &map2.data[r * cols..(r + 1) * cols];

            let handled = match interpolation {
                InterpolationFlags::Nearest => T::simd_remap_nearest_row(
                    row_data,
                    &src.data,
                    src.cols,
                    src.rows,
                    channels,
                    map1_row,
                    map2_row,
                    border_mode,
                    border_value,
                ),
                InterpolationFlags::Linear => T::simd_remap_bilinear_row(
                    row_data,
                    &src.data,
                    src.cols,
                    src.rows,
                    channels,
                    map1_row,
                    map2_row,
                    border_mode,
                    border_value,
                ),
            };

            if !handled {
                remap_row_scalar(
                    row_data,
                    &src.data,
                    src.cols,
                    src.rows,
                    channels,
                    map1_row,
                    map2_row,
                    interpolation,
                    border_mode,
                    border_value,
                );
            }
        }
    }

    Ok(dst)
}

/// Applies a perspective transformation to an image.
///
/// The function `warp_perspective` transforms the source image using the specified matrix:
/// `dst(x,y) = src((M_00*x + M_01*y + M_02)/(M_20*x + M_21*y + M_22), (M_10*x + M_11*y + M_12)/(M_20*x + M_21*y + M_22))`
///
/// # Arguments
///
/// * `src` - Source image.
/// * `m` - 3x3 perspective transformation matrix (`f64`).
/// * `dsize` - Size of the destination image.
/// * `flags` - Interpolation method to use (e.g., `InterpolationFlags::Linear`).
/// * `border_mode` - Pixel extrapolation method (e.g., `BorderTypes::Constant`).
/// * `border_value` - Value used in case of a constant border.
///
/// # Returns
///
/// Returns a `Result<Matrix<T>>` containing the perspective-transformed image.
///
/// # Errors
///
/// Returns an error if the homography matrix `m` is not a 3x3 single-channel matrix.
///
/// # Examples
///
/// ```
/// use purecv::core::{Matrix, types::{BorderTypes, Scalar, Size2i}};
/// use purecv::imgproc::geometric::{warp_perspective, InterpolationFlags};
///
/// let src = Matrix::<u8>::new(10, 10, 1);
/// let mut m = Matrix::<f64>::new(3, 3, 1);
/// // Set identity matrix for example
/// m.data[0] = 1.0; m.data[4] = 1.0; m.data[8] = 1.0;
///
/// let result = warp_perspective(
///     &src,
///     &m,
///     Size2i::new(10, 10),
///     InterpolationFlags::Linear,
///     BorderTypes::Constant,
///     Scalar::all(0)
/// ).unwrap();
/// ```
pub fn warp_perspective<T>(
    src: &Matrix<T>,
    m: &Matrix<f64>,
    dsize: Size2i,
    flags: InterpolationFlags,
    border_mode: BorderTypes,
    border_value: Scalar<T>,
) -> Result<Matrix<T>>
where
    T: Default + Clone + Copy + ToPrimitive + FromPrimitive + Send + Sync + SimdElement,
{
    if m.rows != 3 || m.cols != 3 || m.channels != 1 {
        return Err(PureCvError::InvalidDimensions(
            "Homography matrix must be 3x3 single-channel".to_string(),
        ));
    }

    let mut m_inv = Matrix::<f64>::new(3, 3, 1);
    invert(m, &mut m_inv, DecompTypes::DECOMP_LU)?;

    let m00 = m_inv.data[0];
    let m01 = m_inv.data[1];
    let m02 = m_inv.data[2];
    let m10 = m_inv.data[3];
    let m11 = m_inv.data[4];
    let m12 = m_inv.data[5];
    let m20 = m_inv.data[6];
    let m21 = m_inv.data[7];
    let m22 = m_inv.data[8];

    let channels = src.channels;
    let mut dst = Matrix::<T>::new(dsize.height as usize, dsize.width as usize, channels);

    #[cfg(feature = "parallel")]
    {
        dst.data
            .par_chunks_exact_mut(dsize.width as usize * channels)
            .enumerate()
            .for_each(|(r, row_data)| {
                let dst_y = r as f64;
                let mut map_x = vec![0.0f32; dsize.width as usize];
                let mut map_y = vec![0.0f32; dsize.width as usize];

                for col in 0..dsize.width as usize {
                    let dst_x = col as f64;
                    let w = m20 * dst_x + m21 * dst_y + m22;
                    if w.abs() > 1e-10 {
                        map_x[col] = ((m00 * dst_x + m01 * dst_y + m02) / w) as f32;
                        map_y[col] = ((m10 * dst_x + m11 * dst_y + m12) / w) as f32;
                    } else {
                        map_x[col] = -1.0;
                        map_y[col] = -1.0;
                    }
                }

                let handled = match flags {
                    InterpolationFlags::Nearest => T::simd_remap_nearest_row(
                        row_data,
                        &src.data,
                        src.cols,
                        src.rows,
                        channels,
                        &map_x,
                        &map_y,
                        border_mode,
                        border_value,
                    ),
                    InterpolationFlags::Linear => T::simd_remap_bilinear_row(
                        row_data,
                        &src.data,
                        src.cols,
                        src.rows,
                        channels,
                        &map_x,
                        &map_y,
                        border_mode,
                        border_value,
                    ),
                };

                if !handled {
                    remap_row_scalar(
                        row_data,
                        &src.data,
                        src.cols,
                        src.rows,
                        channels,
                        &map_x,
                        &map_y,
                        flags,
                        border_mode,
                        border_value,
                    );
                }
            });
    }

    #[cfg(not(feature = "parallel"))]
    {
        let width = dsize.width as usize;
        let height = dsize.height as usize;
        for r in 0..height {
            let row_data = &mut dst.data[r * width * channels..(r + 1) * width * channels];
            let dst_y = r as f64;
            let mut map_x = vec![0.0f32; width];
            let mut map_y = vec![0.0f32; width];

            for col in 0..width {
                let dst_x = col as f64;
                let w = m20 * dst_x + m21 * dst_y + m22;
                if w.abs() > 1e-10 {
                    map_x[col] = ((m00 * dst_x + m01 * dst_y + m02) / w) as f32;
                    map_y[col] = ((m10 * dst_x + m11 * dst_y + m12) / w) as f32;
                } else {
                    map_x[col] = -1.0;
                    map_y[col] = -1.0;
                }
            }

            let handled = match flags {
                InterpolationFlags::Nearest => T::simd_remap_nearest_row(
                    row_data,
                    &src.data,
                    src.cols,
                    src.rows,
                    channels,
                    &map_x,
                    &map_y,
                    border_mode,
                    border_value,
                ),
                InterpolationFlags::Linear => T::simd_remap_bilinear_row(
                    row_data,
                    &src.data,
                    src.cols,
                    src.rows,
                    channels,
                    &map_x,
                    &map_y,
                    border_mode,
                    border_value,
                ),
            };

            if !handled {
                remap_row_scalar(
                    row_data,
                    &src.data,
                    src.cols,
                    src.rows,
                    channels,
                    &map_x,
                    &map_y,
                    flags,
                    border_mode,
                    border_value,
                );
            }
        }
    }

    Ok(dst)
}

/// Fallback scalar implementation for remapping a row.
#[allow(clippy::too_many_arguments)]
fn remap_row_scalar<T>(
    dst_row: &mut [T],
    src: &[T],
    src_cols: usize,
    src_rows: usize,
    channels: usize,
    map1_row: &[f32],
    map2_row: &[f32],
    interpolation: InterpolationFlags,
    border_mode: BorderTypes,
    border_value: Scalar<T>,
) where
    T: Default + Clone + Copy + ToPrimitive + FromPrimitive,
{
    let dst_cols = dst_row.len() / channels;

    match interpolation {
        InterpolationFlags::Nearest => {
            for col in 0..dst_cols {
                let x = map1_row[col];
                let y = map2_row[col];
                let ix = x.round() as i32;
                let iy = y.round() as i32;
                let dst_idx = col * channels;

                if ix >= 0 && ix < src_cols as i32 && iy >= 0 && iy < src_rows as i32 {
                    let src_idx = (iy as usize * src_cols + ix as usize) * channels;
                    dst_row[dst_idx..dst_idx + channels]
                        .copy_from_slice(&src[src_idx..src_idx + channels]);
                } else {
                    let ix_interp = border_interpolate(ix, src_cols as i32, border_mode);
                    let iy_interp = border_interpolate(iy, src_rows as i32, border_mode);
                    if ix_interp >= 0 && iy_interp >= 0 {
                        let src_idx =
                            (iy_interp as usize * src_cols + ix_interp as usize) * channels;
                        dst_row[dst_idx..dst_idx + channels]
                            .copy_from_slice(&src[src_idx..src_idx + channels]);
                    } else {
                        dst_row[dst_idx..dst_idx + channels]
                            .copy_from_slice(&border_value.v[..channels]);
                    }
                }
            }
        }
        InterpolationFlags::Linear => {
            for col in 0..dst_cols {
                let x = map1_row[col];
                let y = map2_row[col];

                let x1 = x.floor() as i32;
                let y1 = y.floor() as i32;
                let x2 = x1 + 1;
                let y2 = y1 + 1;

                let wx = x - x.floor();
                let wy = y - y.floor();

                let w00 = (1.0 - wx) * (1.0 - wy);
                let w10 = wx * (1.0 - wy);
                let w01 = (1.0 - wx) * wy;
                let w11 = wx * wy;

                let dst_idx = col * channels;

                if x1 >= 0 && x2 < src_cols as i32 && y1 >= 0 && y2 < src_rows as i32 {
                    let idx00 = (y1 as usize * src_cols + x1 as usize) * channels;
                    let idx10 = (y1 as usize * src_cols + x2 as usize) * channels;
                    let idx01 = (y2 as usize * src_cols + x1 as usize) * channels;
                    let idx11 = (y2 as usize * src_cols + x2 as usize) * channels;

                    for c in 0..channels {
                        let v00 = src[idx00 + c].to_f32().unwrap_or(0.0);
                        let v10 = src[idx10 + c].to_f32().unwrap_or(0.0);
                        let v01 = src[idx01 + c].to_f32().unwrap_or(0.0);
                        let v11 = src[idx11 + c].to_f32().unwrap_or(0.0);

                        let val = w00 * v00 + w10 * v10 + w01 * v01 + w11 * v11;
                        dst_row[dst_idx + c] = T::from_f32(val).unwrap_or_default();
                    }
                } else {
                    for c in 0..channels {
                        let get_pixel = |ix: i32, iy: i32| -> f32 {
                            let ix_interp = border_interpolate(ix, src_cols as i32, border_mode);
                            let iy_interp = border_interpolate(iy, src_rows as i32, border_mode);
                            if ix_interp >= 0 && iy_interp >= 0 {
                                src[(iy_interp as usize * src_cols + ix_interp as usize) * channels
                                    + c]
                                    .to_f32()
                                    .unwrap_or(0.0)
                            } else {
                                border_value.v[c].to_f32().unwrap_or(0.0)
                            }
                        };

                        let v00 = get_pixel(x1, y1);
                        let v10 = get_pixel(x2, y1);
                        let v01 = get_pixel(x1, y2);
                        let v11 = get_pixel(x2, y2);

                        let val = w00 * v00 + w10 * v10 + w01 * v01 + w11 * v11;
                        dst_row[dst_idx + c] = T::from_f32(val).unwrap_or_default();
                    }
                }
            }
        }
    }
}

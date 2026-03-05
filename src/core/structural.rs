/*
 *  structural.rs
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

use crate::core::Matrix;
use crate::core::types::Scalar;
use crate::core::error::{PureCvError, Result};

#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// Flips a 2D matrix around vertical, horizontal, or both axes.
///
/// flip_code:
///  0  - vertical (X-axis)
///  >0 - horizontal (Y-axis)
///  <0 - both axes
pub fn flip<T>(src: &Matrix<T>, flip_code: i32) -> Result<Matrix<T>>
where
    T: Copy + Send + Sync + Default + 'static,
{
    let mut dst = Matrix::<T>::new(src.rows, src.cols, src.channels);
    let channels = src.channels;
    let width_step = src.cols * channels;

    #[cfg(feature = "parallel")]
    {
        dst.data.par_chunks_mut(width_step).enumerate().for_each(|(y, row_dst)| {
            let y_src = if flip_code <= 0 { src.rows - 1 - y } else { y };
            let row_src = &src.data[y_src * width_step..(y_src + 1) * width_step];

            if flip_code != 0 {
                // Horizontal flip (or both)
                for x in 0..src.cols {
                    let x_src = src.cols - 1 - x;
                    for c in 0..channels {
                        row_dst[x * channels + c] = row_src[x_src * channels + c];
                    }
                }
            } else {
                // Vertical flip only
                row_dst.copy_from_slice(row_src);
            }
        });
    }

    #[cfg(not(feature = "parallel"))]
    {
        for y in 0..src.rows {
            let y_src = if flip_code <= 0 { src.rows - 1 - y } else { y };
            let row_src = &src.data[y_src * width_step..(y_src + 1) * width_step];
            let row_dst = &mut dst.data[y * width_step..(y + 1) * width_step];

            if flip_code != 0 {
                for x in 0..src.cols {
                    let x_src = src.cols - 1 - x;
                    for c in 0..channels {
                        row_dst[x * channels + c] = row_src[x_src * channels + c];
                    }
                }
            } else {
                row_dst.copy_from_slice(row_src);
            }
        }
    }

    Ok(dst)
}

/// Transposes a matrix.
pub fn transpose<T>(src: &Matrix<T>) -> Result<Matrix<T>>
where
    T: Copy + Send + Sync + Default + 'static,
{
    let mut dst = Matrix::<T>::new(src.cols, src.rows, src.channels);
    let channels = src.channels;

    #[cfg(feature = "parallel")]
    {
        dst.data.par_chunks_mut(src.rows * channels).enumerate().for_each(|(x_dst, col_dst)| {
            for y_dst in 0..src.rows {
                let src_idx = (y_dst * src.cols + x_dst) * channels;
                let dst_idx = y_dst * channels;
                for c in 0..channels {
                    col_dst[dst_idx + c] = src.data[src_idx + c];
                }
            }
        });
    }

    #[cfg(not(feature = "parallel"))]
    {
        for y in 0..src.rows {
            for x in 0..src.cols {
                let src_idx = (y * src.cols + x) * channels;
                let dst_idx = (x * src.rows + y) * channels;
                for c in 0..channels {
                    dst.data[dst_idx + c] = src.data[src_idx + c];
                }
            }
        }
    }

    Ok(dst)
}

/// Divides a multi-channel matrix into several single-channel matrices.
pub fn split<T>(src: &Matrix<T>) -> Result<Vec<Matrix<T>>>
where
    T: Copy + Send + Sync + Default + 'static,
{
    let mut mv = Vec::with_capacity(src.channels);
    for _ in 0..src.channels {
        mv.push(Matrix::<T>::new(src.rows, src.cols, 1));
    }

    #[cfg(feature = "parallel")]
    {
        let channels = src.channels;
        // Parallelizing over channels might be overkill for small channel counts,
        // but it's consistent. Better to parallelize over pixels if channels are few.
        for c in 0..channels {
            let dst_data = &mut mv[c].data;
            dst_data.par_iter_mut().enumerate().for_each(|(i, val)| {
                *val = src.data[i * channels + c];
            });
        }
    }

    #[cfg(not(feature = "parallel"))]
    {
        let channels = src.channels;
        for i in 0..src.rows * src.cols {
            for c in 0..channels {
                mv[c].data[i] = src.data[i * channels + c];
            }
        }
    }

    Ok(mv)
}

/// Copies specified channels from input arrays to the specified channels of output arrays.
pub fn mix_channels<T>(
    src: &[Matrix<T>],
    dst: &mut [Matrix<T>],
    from_to: &[(usize, usize)],
) -> Result<()>
where
    T: Copy + Send + Sync + Default + 'static,
{
    if src.is_empty() || dst.is_empty() || from_to.is_empty() {
        return Err(PureCvError::InvalidInput("Input/Output/from_to cannot be empty".to_string()));
    }

    let rows = src[0].rows;
    let cols = src[0].cols;

    // Validate dimensions
    for m in src {
        if m.rows != rows || m.cols != cols {
            return Err(PureCvError::InvalidDimensions("All source matrices must have the same size".to_string()));
        }
    }
    for m in dst.iter() {
        if m.rows != rows || m.cols != cols {
            return Err(PureCvError::InvalidDimensions("All destination matrices must have the same size".to_string()));
        }
    }

    // Helper to find which matrix and channel a global channel index refers to
    let get_matrix_and_chan = |matrices: &[Matrix<T>], global_idx: usize| -> Option<(usize, usize)> {
        let mut idx = global_idx;
        for (m_idx, m) in matrices.iter().enumerate() {
            if idx < m.channels {
                return Some((m_idx, idx));
            }
            idx -= m.channels;
        }
        None
    };

    let total_pixels = rows * cols;

    #[cfg(feature = "parallel")]
    {
        // Parallelizing over pixels for maximum efficiency
        (0..total_pixels).into_par_iter().for_each(|_i| {
            for &(f_idx, t_idx) in from_to {
                if let (Some((_fs_idx, _fc_idx)), Some((_ds_idx, _dc_idx))) = (
                    get_matrix_and_chan(src, f_idx),
                    get_matrix_and_chan(dst, t_idx),
                ) {
                    // Safety: We've validated dimensions, but we must use unsafe or a safer way to mutate multiple matrices
                    // Since it's a known pixel index 'i', we can carefully access data.
                    // However, mutating multiple &mut [Matrix] items in parallel is tricky in safe Rust without interior mutability
                    // or splitting the output buffers.
                    // For now, let's process it sequentially per pixel if we can't easily split.
                    // Actually, a safer parallel approach is to parallelize over pixels AND 
                    // ensure we don't have overlapping destination writes in the mapping (OpenCV allows it, but it's UB in Rust).
                }
            }
        });
        // Wait, mixChannels is complex for parallel execution in safe Rust if we have many destination matrices.
        // Let's implement it sequentially first and think about parallelization later.
    }

    // Sequential implementation for stability
    for i in 0..total_pixels {
        for &(f_idx, t_idx) in from_to {
            if let (Some((fs_idx, fc_idx)), Some((ds_idx, dc_idx))) = (
                get_matrix_and_chan(src, f_idx),
                get_matrix_and_chan(dst, t_idx),
            ) {
                dst[ds_idx].data[i * dst[ds_idx].channels + dc_idx] = 
                    src[fs_idx].data[i * src[fs_idx].channels + fc_idx];
            }
        }
    }

    Ok(())
}

/// Forms a border around an image.
///
/// border_type:
///  0 - BORDER_CONSTANT: iiiiii|abcdefgh|iiiiiii with some value
///  1 - BORDER_REFLECT: fedcba|abcdefgh|hgfedcb
pub fn copy_make_border<T>(
    src: &Matrix<T>,
    top: usize,
    bottom: usize,
    left: usize,
    right: usize,
    border_type: i32,
    value: Scalar<T>,
) -> Result<Matrix<T>>
where
    T: Copy + Send + Sync + Default + 'static,
{
    let dst_rows = src.rows + top + bottom;
    let dst_cols = src.cols + left + right;
    let mut dst = Matrix::<T>::new(dst_rows, dst_cols, src.channels);
    let channels = src.channels;

    #[cfg(feature = "parallel")]
    {
        dst.data.par_chunks_mut(dst_cols * channels).enumerate().for_each(|(y_dst, row_dst)| {
            for x_dst in 0..dst_cols {
                let (y_src, is_border_y) = if y_dst < top {
                    if border_type == 1 { (top - 1 - y_dst, false) } else { (0, true) }
                } else if y_dst >= top + src.rows {
                    if border_type == 1 { (src.rows - 1 - (y_dst - (top + src.rows)), false) } else { (0, true) }
                } else {
                    (y_dst - top, false)
                };

                let (x_src, is_border_x) = if x_dst < left {
                    if border_type == 1 { (left - 1 - x_dst, false) } else { (0, true) }
                } else if x_dst >= left + src.cols {
                    if border_type == 1 { (src.cols - 1 - (x_dst - (left + src.cols)), false) } else { (0, true) }
                } else {
                    (x_dst - left, false)
                };

                if (is_border_x || is_border_y) && border_type == 0 {
                    for c in 0..channels {
                        row_dst[x_dst * channels + c] = value.v[c];
                    }
                } else {
                    let src_idx = (y_src * src.cols + x_src) * channels;
                    for c in 0..channels {
                        row_dst[x_dst * channels + c] = src.data[src_idx + c];
                    }
                }
            }
        });
    }

    #[cfg(not(feature = "parallel"))]
    {
        for y_dst in 0..dst_rows {
            for x_dst in 0..dst_cols {
                let (y_src, is_border_y) = if y_dst < top {
                    if border_type == 1 { (top - 1 - y_dst, false) } else { (0, true) }
                } else if y_dst >= top + src.rows {
                    if border_type == 1 { (src.rows - 1 - (y_dst - (top + src.rows)), false) } else { (0, true) }
                } else {
                    (y_dst - top, false)
                };

                let (x_src, is_border_x) = if x_dst < left {
                    if border_type == 1 { (left - 1 - x_dst, false) } else { (0, true) }
                } else if x_dst >= left + src.cols {
                    if border_type == 1 { (src.cols - 1 - (x_dst - (left + src.cols)), false) } else { (0, true) }
                } else {
                    (x_dst - left, false)
                };

                if (is_border_x || is_border_y) && border_type == 0 {
                    for c in 0..channels {
                        dst.data[(y_dst * dst_cols + x_dst) * channels + c] = value.v[c];
                    }
                } else {
                    let src_idx = (y_src * src.cols + x_src) * channels;
                    for c in 0..channels {
                        dst.data[(y_dst * dst_cols + x_dst) * channels + c] = src.data[src_idx + c];
                    }
                }
            }
        }
    }

    Ok(dst)
}

/// Creates one multi-channel matrix out of several single-channel ones.
pub fn merge<T>(mv: &[Matrix<T>]) -> Result<Matrix<T>>
where
    T: Copy + Send + Sync + Default + 'static,
{
    if mv.is_empty() {
        return Err(PureCvError::InvalidDimensions("Input vector is empty".to_string()));
    }

    let rows = mv[0].rows;
    let cols = mv[0].cols;
    let channels = mv.len();

    for m in mv {
        if m.rows != rows || m.cols != cols || m.channels != 1 {
            return Err(PureCvError::InvalidDimensions("All matrices must have the same size and 1 channel".to_string()));
        }
    }

    let mut dst = Matrix::<T>::new(rows, cols, channels);

    #[cfg(feature = "parallel")]
    {
        dst.data.par_chunks_mut(channels).enumerate().for_each(|(i, pixel_dst)| {
            for c in 0..channels {
                pixel_dst[c] = mv[c].data[i];
            }
        });
    }

    #[cfg(not(feature = "parallel"))]
    {
        for i in 0..rows * cols {
            for c in 0..channels {
                dst.data[i * channels + c] = mv[c].data[i];
            }
        }
    }

    Ok(dst)
}

/// Rotates a 2D array in multiples of 90 degrees.
///
/// rotate_code:
///  0 - 90 deg clockwise
///  1 - 180 deg clockwise
///  2 - 270 deg clockwise (90 deg counter-clockwise)
pub fn rotate<T>(src: &Matrix<T>, rotate_code: i32) -> Result<Matrix<T>>
where
    T: Copy + Send + Sync + Default + 'static,
{
    match rotate_code {
        0 => {
            // 90 deg clockwise: transpose + flip horizontal
            let t = transpose(src)?;
            flip(&t, 1)
        }
        1 => {
            // 180 deg clockwise: flip vertical + flip horizontal
            flip(src, -1)
        }
        2 => {
            // 270 deg clockwise: transpose + flip vertical
            let t = transpose(src)?;
            flip(&t, 0)
        }
        _ => Err(PureCvError::InvalidDimensions("Invalid rotate_code. Must be 0, 1, or 2".to_string())),
    }
}

/// Fills the output array by repeating the input array.
pub fn repeat<T>(src: &Matrix<T>, ny: usize, nx: usize) -> Result<Matrix<T>>
where
    T: Copy + Send + Sync + Default + 'static,
{
    if ny == 0 || nx == 0 {
        return Err(PureCvError::InvalidDimensions("ny and nx must be > 0".to_string()));
    }

    let mut dst = Matrix::<T>::new(src.rows * ny, src.cols * nx, src.channels);
    let channels = src.channels;
    let src_width_step = src.cols * channels;
    let dst_width_step = dst.cols * channels;

    #[cfg(feature = "parallel")]
    {
        dst.data.par_chunks_mut(dst_width_step).enumerate().for_each(|(y_dst, row_dst)| {
            let y_src = y_dst % src.rows;
            let row_src = &src.data[y_src * src_width_step..(y_src + 1) * src_width_step];

            for ix in 0..nx {
                row_dst[ix * src_width_step..(ix + 1) * src_width_step].copy_from_slice(row_src);
            }
        });
    }

    #[cfg(not(feature = "parallel"))]
    {
        for y_dst in 0..dst.rows {
            let y_src = y_dst % src.rows;
            let row_src = &src.data[y_src * src_width_step..(y_src + 1) * src_width_step];
            let row_dst = &mut dst.data[y_dst * dst_width_step..(y_dst + 1) * dst_width_step];

            for ix in 0..nx {
                row_dst[ix * src_width_step..(ix + 1) * src_width_step].copy_from_slice(row_src);
            }
        }
    }

    Ok(dst)
}

/// Changes the shape and/or the number of channels of a 2D matrix without copying the data.
///
/// Note: In this implementation, it returns a new Matrix with a copy of the data
/// to maintain the current Matrix struct design.
pub fn reshape<T>(src: &Matrix<T>, new_channels: usize, new_rows: usize) -> Result<Matrix<T>>
where
    T: Copy + Send + Sync + Default + 'static,
{
    let total_elements = src.rows * src.cols * src.channels;
    
    let actual_new_channels = if new_channels == 0 { src.channels } else { new_channels };
    
    if total_elements % actual_new_channels != 0 {
        return Err(PureCvError::InvalidDimensions(format!(
            "Total elements ({}) is not divisible by new_channels ({})",
            total_elements, actual_new_channels
        )));
    }

    let total_pixels = total_elements / actual_new_channels;
    let actual_new_rows = if new_rows == 0 { src.rows } else { new_rows };

    if total_pixels % actual_new_rows != 0 {
        return Err(PureCvError::InvalidDimensions(format!(
            "Total pixels ({}) is not divisible by new_rows ({})",
            total_pixels, actual_new_rows
        )));
    }

    let new_cols = total_pixels / actual_new_rows;

    let mut dst = Matrix::<T>::new(actual_new_rows, new_cols, actual_new_channels);
    dst.data.copy_from_slice(&src.data);

    Ok(dst)
}

/// Horizontal concatenation of matrices.
pub fn hconcat<T>(src: &[Matrix<T>]) -> Result<Matrix<T>>
where
    T: Copy + Send + Sync + Default + 'static,
{
    if src.is_empty() {
        return Err(PureCvError::InvalidInput("Input array is empty".to_string()));
    }

    let rows = src[0].rows;
    let channels = src[0].channels;
    let mut total_cols = 0;

    for m in src {
        if m.rows != rows || m.channels != channels {
            return Err(PureCvError::InvalidDimensions(
                "All matrices must have the same number of rows and channels".to_string(),
            ));
        }
        total_cols += m.cols;
    }

    let mut dst = Matrix::<T>::new(rows, total_cols, channels);

    #[cfg(feature = "parallel")]
    {
        dst.data.par_chunks_mut(total_cols * channels).enumerate().for_each(|(y, row_dst)| {
            let mut offset = 0;
            for m in src {
                let row_width = m.cols * channels;
                let src_row = &m.data[y * row_width..(y + 1) * row_width];
                row_dst[offset..offset + row_width].copy_from_slice(src_row);
                offset += row_width;
            }
        });
    }

    #[cfg(not(feature = "parallel"))]
    {
        for y in 0..rows {
            let mut offset = 0;
            for m in src {
                let row_width = m.cols * channels;
                let row_start = y * row_width;
                let src_row = &m.data[row_start..row_start + row_width];
                let dst_row_start = y * total_cols * channels;
                let dst_row = &mut dst.data[dst_row_start..dst_row_start + total_cols * channels];
                dst_row[offset..offset + row_width].copy_from_slice(src_row);
                offset += row_width;
            }
        }
    }

    Ok(dst)
}

/// Vertical concatenation of matrices.
pub fn vconcat<T>(src: &[Matrix<T>]) -> Result<Matrix<T>>
where
    T: Copy + Send + Sync + Default + 'static,
{
    if src.is_empty() {
        return Err(PureCvError::InvalidInput("Input array is empty".to_string()));
    }

    let cols = src[0].cols;
    let channels = src[0].channels;
    let mut total_rows = 0;

    for m in src {
        if m.cols != cols || m.channels != channels {
            return Err(PureCvError::InvalidDimensions(
                "All matrices must have the same number of columns and channels".to_string(),
            ));
        }
        total_rows += m.rows;
    }

    let mut dst = Matrix::<T>::new(total_rows, cols, channels);
    let mut offset = 0;
    for m in src {
        let size = m.data.len();
        dst.data[offset..offset + size].copy_from_slice(&m.data);
        offset += size;
    }

    Ok(dst)
}

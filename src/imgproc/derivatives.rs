/*
 *  derivatives.rs
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

use alloc::{string::ToString, vec, vec::Vec};

use crate::core::error::{PureCvError, Result};
use crate::core::utils::border_interpolate;
use crate::core::{BorderTypes, Matrix};
use num_traits::{FromPrimitive, NumCast, ToPrimitive};

#[cfg(not(feature = "parallel"))]
use crate::core::utils::ParIterFallback;
#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// Returns derivative filter coefficients.
///
/// * `n` - Kernel size (3, 5, 7, ...).
/// * `dx` - Derivative order (0, 1, 2).
/// * `normalize` - Whether to normalize the kernel.
pub fn get_sobel_kernels(ksize: i32, dx: i32, dy: i32) -> (Vec<f64>, Vec<f64>) {
    let kx = get_deriv_kernel(ksize, dx);
    let ky = get_deriv_kernel(ksize, dy);
    (kx, ky)
}

fn get_deriv_kernel(n: i32, d: i32) -> Vec<f64> {
    if d < 0 {
        return vec![];
    }

    // Simple implementation for ksize=3, ksize=-1 (Scharr), etc.
    // For Sobel ksize=3:
    // d=0: [1, 2, 1]
    // d=1: [-1, 0, 1]
    // d=2: [1, -2, 1]

    match n {
        -1 => {
            // Scharr
            match d {
                0 => vec![3.0, 10.0, 3.0],
                1 => vec![-1.0, 0.0, 1.0],
                _ => vec![],
            }
        }
        3 => match d {
            0 => vec![1.0, 2.0, 1.0],
            1 => vec![-1.0, 0.0, 1.0],
            2 => vec![1.0, -2.0, 1.0],
            _ => vec![],
        },
        5 => match d {
            0 => vec![1.0, 4.0, 6.0, 4.0, 1.0],
            1 => vec![-1.0, -2.0, 0.0, 2.0, 1.0],
            2 => vec![1.0, 0.0, -2.0, 0.0, 1.0],
            _ => vec![],
        },
        _ => vec![], // TODO: Implement general case if needed
    }
}

/// Calculates the first, second, third, or mixed image derivatives using an extended Sobel operator.
///
/// # Performance
///
/// For `ksize=3`, uses an optimized `fast_deriv_3x3` path. With `T = f32` and the
/// `simd` feature enabled, the interior rows are processed via a SIMD kernel achieving
/// ~4.5x speedup over scalar. Combined with `parallel`, reaches up to 22x total
/// speedup on 1024x1024 images — the highest in the project.
pub fn sobel<T>(
    src: &Matrix<T>,
    dx: i32,
    dy: i32,
    ksize: i32,
    scale: f64,
    delta: f64,
    border_type: BorderTypes,
) -> Result<Matrix<T>>
where
    T: Default + Clone + ToPrimitive + FromPrimitive + NumCast + Copy + Send + Sync + 'static,
{
    if ksize != -1 && ksize % 2 == 0 {
        return Err(PureCvError::InvalidInput(
            "Kernel size must be odd or -1 (Scharr)".to_string(),
        ));
    }

    let (kx, ky) = get_sobel_kernels(ksize, dx, dy);
    if kx.is_empty() || ky.is_empty() {
        return Err(PureCvError::InvalidInput(
            "Invalid derivative order or kernel size".to_string(),
        ));
    }

    if kx.len() == 3 && ky.len() == 3 {
        let mut kx_arr = [0.0; 3];
        let mut ky_arr = [0.0; 3];
        kx_arr.copy_from_slice(&kx);
        ky_arr.copy_from_slice(&ky);
        fast_deriv_3x3(src, kx_arr, ky_arr, scale, delta, border_type)
    } else {
        sep_filter_2d(src, &kx, &ky, scale, delta, border_type)
    }
}

/// Calculates the first x- or y-image derivative using the Scharr operator.
///
/// # Performance
///
/// Uses the same optimized `fast_deriv_3x3` path as [`sobel`] with `ksize=3`.
/// With `parallel`, achieves ~12x speedup on 1024x1024 images.
pub fn scharr<T>(
    src: &Matrix<T>,
    dx: i32,
    dy: i32,
    scale: f64,
    delta: f64,
    border_type: BorderTypes,
) -> Result<Matrix<T>>
where
    T: Default + Clone + ToPrimitive + FromPrimitive + NumCast + Copy + Send + Sync + 'static,
{
    sobel(src, dx, dy, -1, scale, delta, border_type)
}

/// Calculates the Laplacian of an image.
pub fn laplacian<T>(
    src: &Matrix<T>,
    ksize: i32,
    scale: f64,
    delta: f64,
    border_type: BorderTypes,
) -> Result<Matrix<T>>
where
    T: Default + Clone + ToPrimitive + FromPrimitive + NumCast + Copy + Send + Sync + 'static,
{
    if ksize == 1 {
        // Discrete Laplacian kernel
        // [0,  1, 0]
        // [1, -4, 1]
        // [0,  1, 0]
        let kernel = vec![0.0, 1.0, 0.0, 1.0, -4.0, 1.0, 0.0, 1.0, 0.0];
        filter_2d(src, &kernel, 3, 3, scale, delta, border_type)
    } else {
        // L = d2I/dx2 + d2I/dy2
        let lx = sobel(src, 2, 0, ksize, scale, 0.0, border_type)?;
        let ly = sobel(src, 0, 2, ksize, scale, delta, border_type)?;

        let rows = src.rows;
        let cols = src.cols;
        let channels = src.channels;
        let mut dst = Matrix::<T>::new(rows, cols, channels);

        dst.data
            .par_iter_mut()
            .zip(lx.data.par_iter())
            .zip(ly.data.par_iter())
            .for_each(|((d, x), y)| {
                let sum =
                    ToPrimitive::to_f64(x).unwrap_or(0.0) + ToPrimitive::to_f64(y).unwrap_or(0.0);
                *d = T::from(sum).unwrap_or_default();
            });

        Ok(dst)
    }
}

// Helper: Separable 2D filter
fn sep_filter_2d<T>(
    src: &Matrix<T>,
    kx: &[f64],
    ky: &[f64],
    scale: f64,
    delta: f64,
    border_type: BorderTypes,
) -> Result<Matrix<T>>
where
    T: Default + Clone + ToPrimitive + FromPrimitive + NumCast + Copy + Send + Sync,
{
    let rows = src.rows;
    let cols = src.cols;
    let channels = src.channels;
    let rows_i32 = rows as i32;
    let cols_i32 = cols as i32;

    let kx_len = kx.len() as i32;
    let ky_len = ky.len() as i32;
    let anchor_x = kx_len / 2;
    let anchor_y = ky_len / 2;

    // Use f32 for intermediate buffer to halve memory bandwidth vs f64.
    // Floating point precision is usually sufficient.
    let mut temp = Matrix::<f32>::new(rows, cols, channels);

    // Horizontal pass
    temp.data
        .par_chunks_mut(cols * channels)
        .enumerate()
        .for_each(|(y, row_data)| {
            let row_offset = y * cols * channels;

            for (x, pixel) in row_data.chunks_exact_mut(channels).enumerate() {
                let x_i32 = x as i32;
                let is_x_inside = x_i32 >= anchor_x && x_i32 < cols_i32 - (kx_len - anchor_x - 1);

                for (c, comp) in pixel.iter_mut().enumerate() {
                    let mut sum = 0.0;

                    if is_x_inside {
                        // Fast path without boundary checks
                        let start_x = (x_i32 - anchor_x) as usize;
                        for i in 0..kx_len {
                            let src_x = start_x + (i as usize);
                            let src_idx = row_offset + src_x * channels + c;
                            let val = ToPrimitive::to_f32(&src.data[src_idx]).unwrap_or(0.0);
                            sum += val * kx[i as usize] as f32;
                        }
                    } else {
                        // Slow path with boundary checks
                        for i in 0..kx_len {
                            let src_x =
                                border_interpolate(x_i32 + i - anchor_x, cols_i32, border_type);
                            if src_x >= 0 {
                                let src_idx = row_offset + (src_x as usize) * channels + c;
                                let val = ToPrimitive::to_f32(&src.data[src_idx]).unwrap_or(0.0);
                                sum += val * kx[i as usize] as f32;
                            }
                        }
                    }
                    *comp = sum;
                }
            }
        });

    // Vertical pass
    let mut dst = Matrix::<T>::new(rows, cols, channels);
    dst.data
        .par_chunks_mut(cols * channels)
        .enumerate()
        .for_each(|(y, row_data)| {
            let y_i32 = y as i32;
            let is_y_inside = y_i32 >= anchor_y && y_i32 < rows_i32 - (ky_len - anchor_y - 1);

            for (x, pixel) in row_data.chunks_exact_mut(channels).enumerate() {
                for (c, comp) in pixel.iter_mut().enumerate() {
                    let mut sum = 0.0;

                    if is_y_inside {
                        // Fast path without boundary checks
                        let start_y = (y_i32 - anchor_y) as usize;
                        for i in 0..ky_len {
                            let src_y = start_y + (i as usize);
                            let temp_idx = (src_y * cols + x) * channels + c;
                            sum += temp.data[temp_idx] as f64 * ky[i as usize];
                        }
                    } else {
                        // Slow path with boundary checks
                        for i in 0..ky_len {
                            let src_y =
                                border_interpolate(y_i32 + i - anchor_y, rows_i32, border_type);
                            if src_y >= 0 {
                                let temp_idx = (src_y as usize * cols + x) * channels + c;
                                sum += temp.data[temp_idx] as f64 * ky[i as usize];
                            }
                        }
                    }

                    let final_val = sum * scale + delta;
                    *comp = T::from(final_val).unwrap_or_default();
                }
            }
        });

    Ok(dst)
}

fn fast_deriv_3x3<T>(
    src: &Matrix<T>,
    kx: [f64; 3],
    ky: [f64; 3],
    scale: f64,
    delta: f64,
    border_type: BorderTypes,
) -> Result<Matrix<T>>
where
    T: Default + Clone + ToPrimitive + FromPrimitive + NumCast + Copy + Send + Sync + 'static,
{
    let rows = src.rows;
    let cols = src.cols;
    let channels = src.channels;
    let rows_i32 = rows as i32;
    let cols_i32 = cols as i32;

    let mut dst = Matrix::<T>::new(rows, cols, channels);

    // Pre-multiply kx and ky to get a 3x3 kernel
    let mut k2d = [0.0; 9];
    for y in 0..3 {
        for x in 0..3 {
            k2d[y * 3 + x] = ky[y] * kx[x];
        }
    }

    // SIMD fast-path: when T is f32, convert source rows to f32 slices
    // and use the SIMD 3x3 kernel for interior rows.
    #[cfg(feature = "simd")]
    {
        if core::any::TypeId::of::<T>() == core::any::TypeId::of::<f32>() && rows > 2 && cols > 2 {
            let row_len = cols * channels;

            // Reinterpret src.data as &[f32] — safe because T == f32
            let src_f32: &[f32] = unsafe {
                core::slice::from_raw_parts(src.data.as_ptr() as *const f32, src.data.len())
            };
            let dst_f32: &mut [f32] = unsafe {
                core::slice::from_raw_parts_mut(dst.data.as_mut_ptr() as *mut f32, dst.data.len())
            };

            // Process interior rows with SIMD
            dst_f32
                .par_chunks_mut(row_len)
                .enumerate()
                .for_each(|(y, dst_row)| {
                    let y_i32 = y as i32;
                    let is_y_inside = y_i32 >= 1 && y_i32 < rows_i32 - 1;

                    if is_y_inside {
                        let prev = &src_f32[(y - 1) * row_len..y * row_len];
                        let curr = &src_f32[y * row_len..(y + 1) * row_len];
                        let next = &src_f32[(y + 1) * row_len..(y + 2) * row_len];

                        // Use SIMD for interior columns
                        crate::imgproc::simd::simd_deriv_3x3_row_f32(
                            dst_row, prev, curr, next, &k2d, channels, scale, delta,
                        );

                        // Handle border columns (first and last) with scalar
                        for x in [0usize, cols - 1] {
                            let x_i32 = x as i32;
                            for c in 0..channels {
                                let idx = x * channels + c;
                                let mut sum = 0.0f64;
                                for ky_idx in 0..3i32 {
                                    let src_y = border_interpolate(
                                        y_i32 + ky_idx - 1,
                                        rows_i32,
                                        border_type,
                                    );
                                    if src_y >= 0 {
                                        let y_off = (src_y as usize) * row_len + c;
                                        for kx_idx in 0..3i32 {
                                            let src_x = border_interpolate(
                                                x_i32 + kx_idx - 1,
                                                cols_i32,
                                                border_type,
                                            );
                                            if src_x >= 0 {
                                                sum += src_f32[y_off + (src_x as usize) * channels]
                                                    as f64
                                                    * k2d[(ky_idx * 3 + kx_idx) as usize];
                                            }
                                        }
                                    }
                                }
                                dst_row[idx] = (sum * scale + delta) as f32;
                            }
                        }
                    } else {
                        // Border rows: full scalar fallback
                        for (x, pixel) in dst_row.chunks_exact_mut(channels).enumerate() {
                            let x_i32 = x as i32;
                            for (c, comp) in pixel.iter_mut().enumerate() {
                                let mut sum = 0.0f64;
                                for ky_idx in 0..3i32 {
                                    let src_y = border_interpolate(
                                        y_i32 + ky_idx - 1,
                                        rows_i32,
                                        border_type,
                                    );
                                    if src_y >= 0 {
                                        let y_off = (src_y as usize) * row_len + c;
                                        for kx_idx in 0..3i32 {
                                            let src_x = border_interpolate(
                                                x_i32 + kx_idx - 1,
                                                cols_i32,
                                                border_type,
                                            );
                                            if src_x >= 0 {
                                                sum += src_f32[y_off + (src_x as usize) * channels]
                                                    as f64
                                                    * k2d[(ky_idx * 3 + kx_idx) as usize];
                                            }
                                        }
                                    }
                                }
                                *comp = (sum * scale + delta) as f32;
                            }
                        }
                    }
                });

            // Reinterpret dst_f32 back to dst — already written in-place
            return Ok(dst);
        }
    }

    // Generic scalar fallback (all types)
    dst.data
        .par_chunks_mut(cols * channels)
        .enumerate()
        .for_each(|(y, row_data)| {
            let y_i32 = y as i32;
            let is_y_inside = y_i32 >= 1 && y_i32 < rows_i32 - 1;

            for (x, pixel) in row_data.chunks_exact_mut(channels).enumerate() {
                let x_i32 = x as i32;
                let is_x_inside = x_i32 >= 1 && x_i32 < cols_i32 - 1;

                for (c, comp) in pixel.iter_mut().enumerate() {
                    let mut sum = 0.0;

                    if is_y_inside && is_x_inside {
                        // Fast path
                        let row_prev = (y - 1) * cols * channels + c;
                        let row_curr = y * cols * channels + c;
                        let row_next = (y + 1) * cols * channels + c;

                        let x_prev = (x - 1) * channels;
                        let x_curr = x * channels;
                        let x_next = (x + 1) * channels;

                        // row y-1
                        sum += ToPrimitive::to_f64(&src.data[row_prev + x_prev]).unwrap_or(0.0)
                            * k2d[0];
                        sum += ToPrimitive::to_f64(&src.data[row_prev + x_curr]).unwrap_or(0.0)
                            * k2d[1];
                        sum += ToPrimitive::to_f64(&src.data[row_prev + x_next]).unwrap_or(0.0)
                            * k2d[2];

                        // row y
                        sum += ToPrimitive::to_f64(&src.data[row_curr + x_prev]).unwrap_or(0.0)
                            * k2d[3];
                        sum += ToPrimitive::to_f64(&src.data[row_curr + x_curr]).unwrap_or(0.0)
                            * k2d[4];
                        sum += ToPrimitive::to_f64(&src.data[row_curr + x_next]).unwrap_or(0.0)
                            * k2d[5];

                        // row y+1
                        sum += ToPrimitive::to_f64(&src.data[row_next + x_prev]).unwrap_or(0.0)
                            * k2d[6];
                        sum += ToPrimitive::to_f64(&src.data[row_next + x_curr]).unwrap_or(0.0)
                            * k2d[7];
                        sum += ToPrimitive::to_f64(&src.data[row_next + x_next]).unwrap_or(0.0)
                            * k2d[8];
                    } else {
                        // Slow path
                        for ky_idx in 0..3 {
                            let src_y =
                                border_interpolate(y_i32 + ky_idx - 1, rows_i32, border_type);
                            if src_y >= 0 {
                                let y_offset = (src_y as usize) * cols * channels + c;
                                for kx_idx in 0..3 {
                                    let src_x = border_interpolate(
                                        x_i32 + kx_idx - 1,
                                        cols_i32,
                                        border_type,
                                    );
                                    if src_x >= 0 {
                                        let val = ToPrimitive::to_f64(
                                            &src.data[y_offset + (src_x as usize) * channels],
                                        )
                                        .unwrap_or(0.0);
                                        sum += val * k2d[(ky_idx * 3 + kx_idx) as usize];
                                    }
                                }
                            }
                        }
                    }

                    let final_val = sum * scale + delta;
                    *comp = T::from(final_val).unwrap_or_default();
                }
            }
        });

    Ok(dst)
}

// Helper: General 2D filter (for Laplacian ksize=1 or others)
fn filter_2d<T>(
    src: &Matrix<T>,
    kernel: &[f64],
    kw: i32,
    kh: i32,
    scale: f64,
    delta: f64,
    border_type: BorderTypes,
) -> Result<Matrix<T>>
where
    T: Default + Clone + ToPrimitive + FromPrimitive + NumCast + Copy + Send + Sync,
{
    let rows = src.rows;
    let cols = src.cols;
    let channels = src.channels;
    let rows_i32 = rows as i32;
    let cols_i32 = cols as i32;

    let anchor_x = kw / 2;
    let anchor_y = kh / 2;

    let mut dst = Matrix::<T>::new(rows, cols, channels);

    dst.data
        .par_chunks_mut(cols * channels)
        .enumerate()
        .for_each(|(y, row_data)| {
            let y_i32 = y as i32;
            for (x, pixel) in row_data.chunks_exact_mut(channels).enumerate() {
                let x_i32 = x as i32;
                for (c, comp) in pixel.iter_mut().enumerate() {
                    let mut sum = 0.0;
                    for ky in 0..kh {
                        let src_y =
                            border_interpolate(y_i32 + ky - anchor_y, rows_i32, border_type);
                        for kx in 0..kw {
                            let src_x =
                                border_interpolate(x_i32 + kx - anchor_x, cols_i32, border_type);
                            if let Some(val) = src.at(src_y, src_x, c) {
                                sum += ToPrimitive::to_f64(val).unwrap_or(0.0)
                                    * kernel[(ky * kw + kx) as usize];
                            }
                        }
                    }
                    let final_val = sum * scale + delta;
                    *comp = T::from(final_val).unwrap_or_default();
                }
            }
        });

    Ok(dst)
}

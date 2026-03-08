/*
 *  derivatives.rs
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
 *  Copyright 2026 WebARKit.
 *
 *  Author(s): Walter Perdan @kalwalt https://github.com/kalwalt
 *
 */

use crate::core::{Matrix, BorderTypes};
use crate::core::error::{Result, PureCvError};
use crate::core::utils::border_interpolate;
use num_traits::{ToPrimitive, FromPrimitive, NumCast};
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
    if d < 0 { return vec![]; }
    
    // Simple implementation for ksize=3, ksize=-1 (Scharr), etc.
    // For Sobel ksize=3:
    // d=0: [1, 2, 1]
    // d=1: [-1, 0, 1]
    // d=2: [1, -2, 1]
    
    match n {
        -1 => { // Scharr
            match d {
                0 => vec![3.0, 10.0, 3.0],
                1 => vec![-1.0, 0.0, 1.0],
                _ => vec![],
            }
        },
        3 => {
            match d {
                0 => vec![1.0, 2.0, 1.0],
                1 => vec![-1.0, 0.0, 1.0],
                2 => vec![1.0, -2.0, 1.0],
                _ => vec![],
            }
        },
        5 => {
            match d {
                0 => vec![1.0, 4.0, 6.0, 4.0, 1.0],
                1 => vec![-1.0, -2.0, 0.0, 2.0, 1.0],
                2 => vec![1.0, 0.0, -2.0, 0.0, 1.0],
                _ => vec![],
            }
        },
        _ => vec![], // TODO: Implement general case if needed
    }
}

/// Calculates the first, second, third, or mixed image derivatives using an extended Sobel operator.
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
    T: Default + Clone + ToPrimitive + FromPrimitive + NumCast + Copy + Send + Sync,
{
    if ksize != -1 && ksize % 2 == 0 {
        return Err(PureCvError::InvalidInput("Kernel size must be odd or -1 (Scharr)".to_string()));
    }

    let (kx, ky) = get_sobel_kernels(ksize, dx, dy);
    if kx.is_empty() || ky.is_empty() {
        return Err(PureCvError::InvalidInput("Invalid derivative order or kernel size".to_string()));
    }

    sep_filter_2d(src, &kx, &ky, scale, delta, border_type)
}

/// Calculates the first x- or y-image derivative using the Scharr operator.
pub fn scharr<T>(
    src: &Matrix<T>,
    dx: i32,
    dy: i32,
    scale: f64,
    delta: f64,
    border_type: BorderTypes,
) -> Result<Matrix<T>>
where
    T: Default + Clone + ToPrimitive + FromPrimitive + NumCast + Copy + Send + Sync,
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
    T: Default + Clone + ToPrimitive + FromPrimitive + NumCast + Copy + Send + Sync,
{
    if ksize == 1 {
        // Discrete Laplacian kernel
        // [0,  1, 0]
        // [1, -4, 1]
        // [0,  1, 0]
        let kernel = vec![
            0.0, 1.0, 0.0,
            1.0, -4.0, 1.0,
            0.0, 1.0, 0.0,
        ];
        filter_2d(src, &kernel, 3, 3, scale, delta, border_type)
    } else {
        // L = d2I/dx2 + d2I/dy2
        let lx = sobel(src, 2, 0, ksize, scale, 0.0, border_type)?;
        let ly = sobel(src, 0, 2, ksize, scale, delta, border_type)?;
        
        let rows = src.rows;
        let cols = src.cols;
        let channels = src.channels;
        let mut dst = Matrix::<T>::new(rows, cols, channels);
        
        dst.data.par_iter_mut()
            .zip(lx.data.par_iter())
            .zip(ly.data.par_iter())
            .for_each(|((d, x), y)| {
                let sum = ToPrimitive::to_f64(x).unwrap_or(0.0) + ToPrimitive::to_f64(y).unwrap_or(0.0);
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

    // Horizontal pass
    let mut temp = Matrix::<f64>::new(rows, cols, channels);
    
    temp.data.par_chunks_mut(cols * channels)
        .enumerate()
        .for_each(|(y, row_data)| {
            for x in 0..cols {
                let x_i32 = x as i32;
                for c in 0..channels {
                    let mut sum = 0.0;
                    for i in 0..kx_len {
                        let src_x = border_interpolate(x_i32 + i - anchor_x, cols_i32, border_type);
                        if let Some(val) = src.at(y as i32, src_x, c) {
                            sum += ToPrimitive::to_f64(val).unwrap_or(0.0) * kx[i as usize];
                        }
                    }
                    row_data[x * channels + c] = sum;
                }
            }
        });

    // Vertical pass
    let mut dst = Matrix::<T>::new(rows, cols, channels);
    dst.data.par_chunks_mut(cols * channels)
        .enumerate()
        .for_each(|(y, row_data)| {
            let y_i32 = y as i32;
            for x in 0..cols {
                for c in 0..channels {
                    let mut sum = 0.0;
                    for i in 0..ky_len {
                        let src_y = border_interpolate(y_i32 + i - anchor_y, rows_i32, border_type);
                        if let Some(&val) = temp.at(src_y, x as i32, c) {
                            sum += val * ky[i as usize];
                        }
                    }
                    let final_val = sum * scale + delta;
                    row_data[x * channels + c] = T::from(final_val).unwrap_or_default();
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
    
    dst.data.par_chunks_mut(cols * channels)
        .enumerate()
        .for_each(|(y, row_data)| {
            let y_i32 = y as i32;
            for x in 0..cols {
                let x_i32 = x as i32;
                for c in 0..channels {
                    let mut sum = 0.0;
                    for ky in 0..kh {
                        let src_y = border_interpolate(y_i32 + ky - anchor_y, rows_i32, border_type);
                        for kx in 0..kw {
                            let src_x = border_interpolate(x_i32 + kx - anchor_x, cols_i32, border_type);
                            if let Some(val) = src.at(src_y, src_x, c) {
                                sum += ToPrimitive::to_f64(val).unwrap_or(0.0) * kernel[(ky * kw + kx) as usize];
                            }
                        }
                    }
                    let final_val = sum * scale + delta;
                    row_data[x * channels + c] = T::from(final_val).unwrap_or_default();
                }
            }
        });

    Ok(dst)
}

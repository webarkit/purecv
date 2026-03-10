/*
 *  filter.rs
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

use crate::core::{Matrix, PureCvError, Size2i, Point2i};
use crate::core::types::BorderTypes;
use crate::core::error::Result;
use crate::core::utils::border_interpolate;
use std::iter::Sum;
use num_traits::{ToPrimitive, FromPrimitive, NumCast};

#[cfg(feature = "parallel")]
use rayon::prelude::*;
#[cfg(not(feature = "parallel"))]
use crate::core::utils::ParIterFallback;

/// Blurs an image using the box filter.
/// 
/// * `src` - Input image.
/// * `ksize` - Blurring kernel size.
/// * `anchor` - Anchor point; default value Point(-1,-1) means that the anchor is at the kernel center.
/// * `border_type` - Pixel extrapolation method.
pub fn blur<T>(
    src: &Matrix<T>,
    ksize: Size2i,
    anchor: Point2i,
    border_type: BorderTypes,
) -> Result<Matrix<T>>
where
    T: Default + Clone + ToPrimitive + FromPrimitive + NumCast + Copy + Send + Sync,
{
    box_filter(src, ksize, anchor, true, border_type)
}

/// Blurs an image using the box filter.
/// 
/// * `src` - Input image.
/// * `ksize` - Blurring kernel size.
/// * `anchor` - Anchor point.
/// * `normalize` - Flag, specifying whether the kernel is normalized by its area or not.
/// * `border_type` - Pixel extrapolation method.
pub fn box_filter<T>(
    src: &Matrix<T>,
    ksize: Size2i,
    anchor: Point2i,
    normalize: bool,
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

    let anchor_x = if anchor.x == -1 { ksize.width / 2 } else { anchor.x };
    let anchor_y = if anchor.y == -1 { ksize.height / 2 } else { anchor.y };

    // Intermediate buffer as f64 to avoid overflow and precision issues
    let mut temp = Matrix::<f64>::new(rows, cols, channels);
    
    // Horizontal pass
    let temp_data = &mut temp.data;
    temp_data.par_chunks_mut(cols * channels)
        .enumerate()
        .for_each(|(y, row_data)| {
            for (x, pixel) in row_data.chunks_exact_mut(channels).enumerate() {
                let x_i32 = x as i32;
                for (c, comp) in pixel.iter_mut().enumerate() {
                    let mut sum = 0.0;
                    for kx in 0..ksize.width {
                        let src_x = border_interpolate(x_i32 + kx - anchor_x, cols_i32, border_type);
                        if src_x >= 0 {
                            if let Some(val) = src.at(y as i32, src_x, c) {
                                sum += val.to_f64().unwrap_or(0.0);
                            }
                        }
                    }
                    *comp = sum;
                }
            }
        });

    let mut dst = Matrix::<T>::new(rows, cols, channels);
    let inv_area = if normalize { 1.0 / (ksize.width * ksize.height) as f64 } else { 1.0 };

    // Vertical pass
    dst.data.par_chunks_mut(cols * channels)
        .enumerate()
        .for_each(|(y, row_data)| {
            let y_i32 = y as i32;
            for (x, pixel) in row_data.chunks_exact_mut(channels).enumerate() {
                for (c, comp) in pixel.iter_mut().enumerate() {
                    let mut sum = 0.0;
                    for ky in 0..ksize.height {
                        let src_y = border_interpolate(y_i32 + ky - anchor_y, rows_i32, border_type);
                        if src_y >= 0 {
                            if let Some(&val) = temp.at(src_y, x as i32, c) {
                                sum += val;
                            }
                        }
                    }
                    let final_val = sum * inv_area;
                    *comp = T::from(final_val).unwrap_or_default();
                }
            }
        });

    Ok(dst)
}

/// Returns Gaussian filter coefficients.
/// 
/// * `n` - Kernel size.
/// * `sigma` - Gaussian standard deviation.
pub fn get_gaussian_kernel(n: i32, sigma: f64) -> Vec<f64> {
    if n <= 0 { return vec![]; }
    let mut kernel = vec![0.0; n as usize];
    
    // If sigma is <= 0, compute it from n as in OpenCV
    let sigma_x = if sigma <= 0.0 {
        0.3 * ((n as f64 - 1.0) * 0.5 - 1.0) + 0.8
    } else {
        sigma
    };

    let mut sum = 0.0;
    for i in 0..n {
        let x = i as f64 - (n as f64 - 1.0) * 0.5;
        kernel[i as usize] = (-x * x / (2.0 * sigma_x * sigma_x)).exp();
        sum += kernel[i as usize];
    }

    if sum > 0.0 {
        for i in 0..n {
            kernel[i as usize] /= sum;
        }
    }

    kernel
}

/// Blurs an image using a Gaussian filter.
/// 
/// * `src` - Input image.
/// * `ksize` - Gaussian kernel size. `ksize.width` and `ksize.height` can differ but they both must be positive and odd.
/// * `sigma1` - Gaussian kernel standard deviation in X direction.
/// * `sigma2` - Gaussian kernel standard deviation in Y direction.
/// * `border_type` - Pixel extrapolation method.
pub fn gaussian_blur<T>(
    src: &Matrix<T>,
    ksize: Size2i,
    sigma1: f64,
    sigma2: f64,
    border_type: BorderTypes,
) -> Result<Matrix<T>>
where
    T: Default + Clone + ToPrimitive + FromPrimitive + NumCast + Copy + Send + Sync,
{
    if ksize.width % 2 == 0 || ksize.height % 2 == 0 {
        return Err(PureCvError::InvalidInput("Kernel size must be odd".to_string()));
    }

    let kx = get_gaussian_kernel(ksize.width, sigma1);
    let ky = if sigma2 <= 0.0 && sigma1 > 0.0 {
        kx.clone()
    } else {
        get_gaussian_kernel(ksize.height, sigma2)
    };

    let rows = src.rows;
    let cols = src.cols;
    let channels = src.channels;
    let rows_i32 = rows as i32;
    let cols_i32 = cols as i32;

    // Horizontal pass
    let mut temp = Matrix::<f64>::new(rows, cols, channels);
    let radius_x = ksize.width / 2;

    temp.data.par_chunks_mut(cols * channels)
        .enumerate()
        .for_each(|(y, row_data)| {
            let y_i32 = y as i32;
            for (x, pixel) in row_data.chunks_exact_mut(channels).enumerate() {
                let x_i32 = x as i32;
                for (c, comp) in pixel.iter_mut().enumerate() {
                    let mut sum = 0.0;
                    for k in 0..ksize.width {
                        let src_x = border_interpolate(x_i32 + k - radius_x, cols_i32, border_type);
                        if src_x >= 0 {
                            if let Some(val) = src.at(y_i32, src_x, c) {
                                sum += val.to_f64().unwrap_or(0.0) * kx[k as usize];
                            }
                        }
                    }
                    *comp = sum;
                }
            }
        });

    // Vertical pass
    let mut dst = Matrix::<T>::new(rows, cols, channels);
    let radius_y = ksize.height / 2;

    dst.data.par_chunks_mut(cols * channels)
        .enumerate()
        .for_each(|(y, row_data)| {
            let y_i32 = y as i32;
            for (x, pixel) in row_data.chunks_exact_mut(channels).enumerate() {
                let x_i32 = x as i32;
                for (c, comp) in pixel.iter_mut().enumerate() {
                    let mut sum = 0.0;
                    for k in 0..ksize.height {
                        let src_y = border_interpolate(y_i32 + k - radius_y, rows_i32, border_type);
                        if src_y >= 0 {
                            if let Some(&val) = temp.at(src_y, x_i32, c) {
                                sum += val * ky[k as usize];
                            }
                        }
                    }
                    *comp = T::from(sum).unwrap_or_default();
                }
            }
        });


    Ok(dst)
}

/// Blurs an image using the median filter.
/// 
/// * `src` - Input matrix.
/// * `ksize` - Aperture linear size; it must be odd and greater than 1, for example: 3, 5, 7.
pub fn median_blur<T>(
    src: &Matrix<T>,
    ksize: i32,
) -> Result<Matrix<T>>
where
    T: Default + Clone + PartialOrd + Send + Sync + Copy + ToPrimitive,
{
    if ksize % 2 == 0 || ksize <= 1 {
        return Err(crate::core::error::PureCvError::InvalidInput("Kernel size must be odd and > 1".to_string()));
    }

    let rows = src.rows;
    let cols = src.cols;
    let channels = src.channels;
    let rows_i32 = rows as i32;
    let cols_i32 = cols as i32;
    let radius = ksize / 2;

    let mut dst = Matrix::<T>::new(rows, cols, channels);

    dst.data.par_chunks_mut(cols * channels)
        .enumerate()
        .for_each(|(y, row_data)| {
            let y_i32 = y as i32;
            let mut neighbors = Vec::with_capacity((ksize * ksize) as usize);

            for (x, pixel) in row_data.chunks_exact_mut(channels).enumerate() {
                let x_i32 = x as i32;
                for (c, comp) in pixel.iter_mut().enumerate() {
                    neighbors.clear();
                    for ky in 0..ksize {
                        for kx in 0..ksize {
                            let src_y = border_interpolate(y_i32 + ky - radius, rows_i32, BorderTypes::REFLECT_101);
                            let src_x = border_interpolate(x_i32 + kx - radius, cols_i32, BorderTypes::REFLECT_101);
                            
                            if let Some(&val) = src.at(src_y, src_x, c) {
                                neighbors.push(val);
                            }
                        }
                    }
                    // Sort to find median
                    neighbors.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                    let median = neighbors[neighbors.len() / 2];
                    *comp = median;
                }
            }
        });

    Ok(dst)
}

/// Applies the bilateral filter to an image.
/// 
/// * `src` - Source 8-bit or floating-point, 1-channel or 3-channel image.
/// * `d` - Diameter of each pixel neighborhood that is used during filtering. If it is non-positive, it is computed from sigmaSpace.
/// * `sigma_color` - Filter sigma in the color space.
/// * `sigma_space` - Filter sigma in the coordinate space.
/// * `border_type` - Border mode used to extrapolate pixels outside of the image.
pub fn bilateral_filter<T>(
    src: &Matrix<T>,
    d: i32,
    sigma_color: f64,
    sigma_space: f64,
    border_type: BorderTypes,
) -> Result<Matrix<T>>
where
    T: Default + Clone + PartialOrd + Send + Sync + Copy + ToPrimitive + FromPrimitive + Sum,
{
    let rows = src.rows;
    let cols = src.cols;
    let channels = src.channels;
    let rows_i32 = rows as i32;
    let cols_i32 = cols as i32;

    let radius = if d <= 0 {
        (sigma_space * 1.5).round() as i32
    } else {
        d / 2
    };

    let mut dst = Matrix::<T>::new(rows, cols, channels);

    let color_coeff = -1.0 / (2.0 * sigma_color * sigma_color);
    let space_coeff = -1.0 / (2.0 * sigma_space * sigma_space);

    dst.data.par_chunks_mut(cols * channels)
        .enumerate()
        .for_each(|(y, row_data)| {
            let y_i32 = y as i32;
            for (x, pixel) in row_data.chunks_exact_mut(channels).enumerate() {
                let x_i32 = x as i32;
                
                let center_vals: Vec<f64> = (0..channels)
                    .map(|c| src.at(y as i32, x as i32, c).map(|&v| v.to_f64().unwrap_or(0.0)).unwrap_or(0.0))
                    .collect();

                let mut sums = vec![0.0f64; channels];
                let mut w_sum = 0.0f64;

                for ky in -radius..=radius {
                    for kx in -radius..=radius {
                        let src_y = border_interpolate(y_i32 + ky, rows_i32, border_type);
                        let src_x = border_interpolate(x_i32 + kx, cols_i32, border_type);

                        let mut color_dist_sq = 0.0f64;
                        let mut neighbor_vals = vec![0.0f64; channels];

                        for c in 0..channels {
                            let val = src.at(src_y, src_x, c).map(|&v| v.to_f64().unwrap_or(0.0)).unwrap_or(0.0);
                            neighbor_vals[c] = val;
                            let diff = val - center_vals[c];
                            color_dist_sq += diff * diff;
                        }

                        let space_dist_sq = (ky * ky + kx * kx) as f64;
                        
                        let weight = (color_dist_sq * color_coeff + space_dist_sq * space_coeff).exp();
                        
                        for c in 0..channels {
                            sums[c] += neighbor_vals[c] * weight;
                        }
                        w_sum += weight;
                    }
                }

                for (c, comp) in pixel.iter_mut().enumerate() {
                    if w_sum > 0.0 {
                        *comp = T::from_f64(sums[c] / w_sum).unwrap_or_default();
                    } else {
                        *comp = T::from_f64(center_vals[c]).unwrap_or_default();
                    }
                }
            }
        });

    Ok(dst)
}

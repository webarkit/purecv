/*
 *  edge.rs
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
use crate::core::error::{PureCvError, Result};
use crate::core::types::BorderTypes;
use crate::imgproc::derivatives::sobel;
use num_traits::{ToPrimitive, FromPrimitive, NumCast};

#[cfg(feature = "parallel")]
use rayon::prelude::*;
#[cfg(not(feature = "parallel"))]
use crate::core::utils::ParIterFallback;

/// Finds edges in an image using the Canny algorithm.
/// 
/// # Arguments
/// * `src` - Input 8-bit single-channel image.
/// * `threshold1` - First threshold for the hysteresis procedure.
/// * `threshold2` - Second threshold for the hysteresis procedure.
/// * `aperture_size` - Aperture size for the Sobel operator (e.g., 3).
/// * `l2_gradient` - A flag, indicating whether a more accurate L2 norm should be used.
pub fn canny<T>(
    src: &Matrix<T>,
    threshold1: f64,
    threshold2: f64,
    aperture_size: i32,
    l2_gradient: bool,
) -> Result<Matrix<u8>>
where
    T: Default + Clone + ToPrimitive + FromPrimitive + NumCast + Copy + Send + Sync,
{
    if src.channels != 1 {
        return Err(PureCvError::InvalidInput("Canny only supports single-channel images".into()));
    }

    let rows = src.rows;
    let cols = src.cols;

    // 1. Convert to f64 to avoid saturation during gradient calculation
    let src_f64 = src.convert_to::<f64>()?;

    // 1. Calculate gradients using Sobel
    let dx = sobel(&src_f64, 1, 0, aperture_size, 1.0, 0.0, BorderTypes::REFLECT_101)?;
    let dy = sobel(&src_f64, 0, 1, aperture_size, 1.0, 0.0, BorderTypes::REFLECT_101)?;

    let mut map = Matrix::<f32>::new(rows, cols, 1);
    let mut mag = Matrix::<f32>::new(rows, cols, 1);

    // 2. Calculate magnitude and orientation
    // We store maps as: 0: 0 deg, 1: 45 deg, 2: 90 deg, 3: 135 deg
    mag.data.par_iter_mut()
        .zip(dx.data.par_iter())
        .zip(dy.data.par_iter())
        .zip(map.data.par_iter_mut())
        .for_each(|(((m, &gx), &gy), o)| {
            let gx_f = ToPrimitive::to_f64(&gx).unwrap_or(0.0);
            let gy_f = ToPrimitive::to_f64(&gy).unwrap_or(0.0);
            
            let magnitude = if l2_gradient {
                (gx_f * gx_f + gy_f * gy_f).sqrt()
            } else {
                gx_f.abs() + gy_f.abs()
            };
            *m = magnitude as f32;

            if magnitude > 1e-5 {
                let angle = gy_f.atan2(gx_f) * 180.0 / std::f64::consts::PI;
                let normalized_angle = if angle < 0.0 { angle + 180.0 } else { angle };
                
                if (0.0..22.5).contains(&normalized_angle) || (157.5..=180.0).contains(&normalized_angle) {
                    *o = 0.0; // Horizontal
                } else if (22.5..67.5).contains(&normalized_angle) {
                    *o = 1.0; // 45 deg
                } else if (67.5..112.5).contains(&normalized_angle) {
                    *o = 2.0; // Vertical
                } else {
                    *o = 3.0; // 135 deg
                }
            } else {
                *o = -1.0;
            }
        });

    // 3. Non-maximum suppression
    let mut suppressed = Matrix::<u8>::new(rows, cols, 1);
    let low_threshold = threshold1.min(threshold2) as f32;
    let high_threshold = threshold1.max(threshold2) as f32;

    for y in 1..rows - 1 {
        for x in 1..cols - 1 {
            let m = *mag.at(y as i32, x as i32, 0).unwrap();
            let o = *map.at(y as i32, x as i32, 0).unwrap();

            if m < low_threshold {
                continue;
            }

            let (m1, m2) = match o as i32 {
                0 => (*mag.at(y as i32, x as i32 - 1, 0).unwrap(), *mag.at(y as i32, x as i32 + 1, 0).unwrap()),
                1 => (*mag.at(y as i32 - 1, x as i32 + 1, 0).unwrap(), *mag.at(y as i32 + 1, x as i32 - 1, 0).unwrap()),
                2 => (*mag.at(y as i32 - 1, x as i32, 0).unwrap(), *mag.at(y as i32 + 1, x as i32, 0).unwrap()),
                3 => (*mag.at(y as i32 - 1, x as i32 - 1, 0).unwrap(), *mag.at(y as i32 + 1, x as i32 + 1, 0).unwrap()),
                _ => (0.0, 0.0),
            };

            if m >= m1 && m >= m2 {
                if m >= high_threshold {
                    suppressed.set(y, x, 0, 2); // Strong edge
                } else {
                    suppressed.set(y, x, 0, 1); // Weak edge
                }
            }
        }
    }

    // 4. Hysteresis thresholding
    let mut dst = Matrix::<u8>::new(rows, cols, 1);
    let mut stack = Vec::with_capacity(rows * cols / 10);

    for y in 1..rows - 1 {
        for x in 1..cols - 1 {
            if *suppressed.at(y as i32, x as i32, 0).unwrap() == 2 {
                stack.push((y, x));
                dst.set(y, x, 0, 255);
                suppressed.set(y, x, 0, 0); // Mark as processed
            }
        }
    }

    while let Some((y, x)) = stack.pop() {
        for dy in -1..=1 {
            for dx in -1..=1 {
                if dy == 0 && dx == 0 { continue; }
                let ny = y as i32 + dy;
                let nx = x as i32 + dx;

                if (1..rows as i32 - 1).contains(&ny) && (1..cols as i32 - 1).contains(&nx)
                    && *suppressed.at(ny, nx, 0).unwrap() == 1 {
                    dst.set(ny as usize, nx as usize, 0, 255);
                    suppressed.set(ny as usize, nx as usize, 0, 0);
                    stack.push((ny as usize, nx as usize));
                }
            }
        }
    }

    Ok(dst)
}

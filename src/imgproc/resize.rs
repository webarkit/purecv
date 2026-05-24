/*
 *  resize.rs
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

//! Image resizing operations: [`resize`].

use crate::core::error::{PureCvError, Result};
use crate::core::types::Size;
use crate::core::Matrix;
use num_traits::{FromPrimitive, ToPrimitive};

#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// Resizes an image using bilinear interpolation.
///
/// # Arguments
/// * `src`      - Input image matrix.
/// * `dst_size` - Desired size of the output image.
///
/// # Errors
/// Returns an error if `dst_size` width or height is 0.
pub fn resize<T>(src: &Matrix<T>, dst_size: Size<usize>) -> Result<Matrix<T>>
where
    T: Default + Clone + Copy + Send + Sync + ToPrimitive + FromPrimitive + 'static,
{
    if dst_size.width == 0 || dst_size.height == 0 {
        return Err(PureCvError::InvalidInput(
            "Destination size must be greater than 0".to_string(),
        ));
    }

    let mut dst = Matrix::<T>::new(dst_size.height, dst_size.width, src.channels);
    let scale_x = src.cols as f64 / dst_size.width as f64;
    let scale_y = src.rows as f64 / dst_size.height as f64;
    let channels = src.channels;
    let src_cols = src.cols;
    let src_rows = src.rows;
    let src_data = &src.data;

    #[cfg(feature = "parallel")]
    {
        dst.data
            .par_chunks_exact_mut(dst_size.width * channels)
            .enumerate()
            .for_each(|(y, row_dst)| {
                let src_y = (y as f64 + 0.5) * scale_y - 0.5;
                let src_y_floor = src_y.floor() as i32;
                let y0 = src_y_floor.clamp(0, src_rows as i32 - 1) as usize;
                let y1 = (src_y_floor + 1).clamp(0, src_rows as i32 - 1) as usize;
                let dy = src_y - src_y_floor as f64;

                for x in 0..dst_size.width {
                    let src_x = (x as f64 + 0.5) * scale_x - 0.5;
                    let src_x_floor = src_x.floor() as i32;
                    let x0 = src_x_floor.clamp(0, src_cols as i32 - 1) as usize;
                    let x1 = (src_x_floor + 1).clamp(0, src_cols as i32 - 1) as usize;
                    let dx = src_x - src_x_floor as f64;

                    for c in 0..channels {
                        let val00 = src_data[y0 * src_cols * channels + x0 * channels + c]
                            .to_f64()
                            .unwrap_or(0.0);
                        let val10 = src_data[y0 * src_cols * channels + x1 * channels + c]
                            .to_f64()
                            .unwrap_or(0.0);
                        let val01 = src_data[y1 * src_cols * channels + x0 * channels + c]
                            .to_f64()
                            .unwrap_or(0.0);
                        let val11 = src_data[y1 * src_cols * channels + x1 * channels + c]
                            .to_f64()
                            .unwrap_or(0.0);

                        let interpolated = (1.0 - dx) * (1.0 - dy) * val00
                            + dx * (1.0 - dy) * val10
                            + (1.0 - dx) * dy * val01
                            + dx * dy * val11;

                        row_dst[x * channels + c] =
                            T::from_f64(interpolated.round()).unwrap_or_else(T::default);
                    }
                }
            });
    }

    #[cfg(not(feature = "parallel"))]
    {
        dst.data
            .chunks_exact_mut(dst_size.width * channels)
            .enumerate()
            .for_each(|(y, row_dst)| {
                let src_y = (y as f64 + 0.5) * scale_y - 0.5;
                let src_y_floor = src_y.floor() as i32;
                let y0 = src_y_floor.clamp(0, src_rows as i32 - 1) as usize;
                let y1 = (src_y_floor + 1).clamp(0, src_rows as i32 - 1) as usize;
                let dy = src_y - src_y_floor as f64;

                for x in 0..dst_size.width {
                    let src_x = (x as f64 + 0.5) * scale_x - 0.5;
                    let src_x_floor = src_x.floor() as i32;
                    let x0 = src_x_floor.clamp(0, src_cols as i32 - 1) as usize;
                    let x1 = (src_x_floor + 1).clamp(0, src_cols as i32 - 1) as usize;
                    let dx = src_x - src_x_floor as f64;

                    for c in 0..channels {
                        let val00 = src_data[y0 * src_cols * channels + x0 * channels + c]
                            .to_f64()
                            .unwrap_or(0.0);
                        let val10 = src_data[y0 * src_cols * channels + x1 * channels + c]
                            .to_f64()
                            .unwrap_or(0.0);
                        let val01 = src_data[y1 * src_cols * channels + x0 * channels + c]
                            .to_f64()
                            .unwrap_or(0.0);
                        let val11 = src_data[y1 * src_cols * channels + x1 * channels + c]
                            .to_f64()
                            .unwrap_or(0.0);

                        let interpolated = (1.0 - dx) * (1.0 - dy) * val00
                            + dx * (1.0 - dy) * val10
                            + (1.0 - dx) * dy * val01
                            + dx * dy * val11;

                        row_dst[x * channels + c] =
                            T::from_f64(interpolated.round()).unwrap_or_else(T::default);
                    }
                }
            });
    }

    Ok(dst)
}

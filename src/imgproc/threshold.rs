/*
 *  threshold.rs
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

use alloc::string::ToString;

use crate::core::error::{PureCvError, Result};
use crate::core::simd::SimdElement;
use crate::core::Matrix;
use num_traits::{FromPrimitive, NumCast, ToPrimitive};

#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// Thresholding types.
#[derive(Debug, Clone, Copy, PartialEq)]
#[allow(non_camel_case_types)]
pub enum ThresholdTypes {
    /// \f[dst(x,y) = \fork{maxval}{if dst(x,y) > thresh}{0}{otherwise}\f]
    THRESH_BINARY = 0,
    /// \f[dst(x,y) = \fork{0}{if dst(x,y) > thresh}{maxval}{otherwise}\f]
    THRESH_BINARY_INV = 1,
    /// \f[dst(x,y) = \fork{threshold}{if dst(x,y) > thresh}{src(x,y)}{otherwise}\f]
    THRESH_TRUNC = 2,
    /// \f[dst(x,y) = \fork{src(x,y)}{if dst(x,y) > thresh}{0}{otherwise}\f]
    THRESH_TOZERO = 3,
    /// \f[dst(x,y) = \fork{0}{if dst(x,y) > thresh}{src(x,y)}{otherwise}\f]
    THRESH_TOZERO_INV = 4,
    /// Flag to use Otsu algorithm to choose the optimal threshold value.
    THRESH_OTSU = 8,
    /// Flag to use Triangle algorithm to choose the optimal threshold value.
    THRESH_TRIANGLE = 16,
}

/// Applies a fixed-level threshold to each array element.
///
/// The function applies fixed-level thresholding to a multiple-channel array.
/// The function is typically used to get a bi-level (binary) image out of a
/// grayscale image or for removing a noise, that is, filtering out pixels
/// with too small or too large values. There are several types of thresholding
/// the function supports. They are determined by type parameter.
///
/// * `src` - Input matrix.
/// * `thresh` - Threshold value.
/// * `maxval` - Maximum value to use with the `BINARY` and `BINARY_INV` thresholding types.
/// * `threshold_type` - Thresholding type.
///
/// # Performance
///
/// For `u8` matrices with the `simd` feature enabled, uses a SIMD-accelerated
/// kernel for binary thresholding. Combined with `parallel`, achieves ~2.1x
/// speedup on 1024x1024 images.
pub fn threshold<T>(
    src: &Matrix<T>,
    thresh: f64,
    maxval: f64,
    threshold_type: ThresholdTypes,
) -> Result<(f64, Matrix<T>)>
where
    T: Default
        + Clone
        + Copy
        + PartialOrd
        + ToPrimitive
        + FromPrimitive
        + NumCast
        + Send
        + Sync
        + SimdElement,
{
    let rows = src.rows;
    let cols = src.cols;
    let channels = src.channels;

    let mut dst = Matrix::<T>::new(rows, cols, channels);
    let actual_thresh = thresh;

    // TODO: Otsu and Triangle implementations will go here
    if (threshold_type as i32 & ThresholdTypes::THRESH_OTSU as i32) != 0 {
        return Err(PureCvError::InvalidInput(
            "OTSU threshold is not yet implemented".to_string(),
        ));
    }
    if (threshold_type as i32 & ThresholdTypes::THRESH_TRIANGLE as i32) != 0 {
        return Err(PureCvError::InvalidInput(
            "TRIANGLE threshold is not yet implemented".to_string(),
        ));
    }

    let type_val = threshold_type as i32 & 0x7; // Get the base type (0-4)

    // SIMD fast path: process entire buffer when the type supports it.
    #[cfg(feature = "simd")]
    {
        if T::has_simd() && (0..=4).contains(&type_val) {
            #[cfg(feature = "parallel")]
            {
                let row_len = cols * channels;
                dst.data
                    .par_chunks_exact_mut(row_len)
                    .zip(src.data.par_chunks_exact(row_len))
                    .for_each(|(dst_row, src_row)| {
                        T::simd_threshold(dst_row, src_row, actual_thresh, maxval, type_val as u8);
                    });
            }
            #[cfg(not(feature = "parallel"))]
            {
                T::simd_threshold(
                    &mut dst.data,
                    &src.data,
                    actual_thresh,
                    maxval,
                    type_val as u8,
                );
            }
            return Ok((actual_thresh, dst));
        }
    }

    #[cfg(feature = "parallel")]
    {
        dst.data
            .as_mut_slice()
            .par_chunks_mut(cols * channels)
            .enumerate()
            .for_each(|(y, row_data)| {
                for (x, pixel) in row_data.chunks_exact_mut(channels).enumerate() {
                    for (c, comp) in pixel.iter_mut().enumerate() {
                        if let Some(src_val) = src.at(y as i32, x as i32, c) {
                            let val_f64 = src_val.to_f64().unwrap_or(0.0);

                            let result = match type_val {
                                0 => {
                                    // BINARY
                                    if val_f64 > actual_thresh {
                                        maxval
                                    } else {
                                        0.0
                                    }
                                }
                                1 => {
                                    // BINARY_INV
                                    if val_f64 > actual_thresh {
                                        0.0
                                    } else {
                                        maxval
                                    }
                                }
                                2 => {
                                    // TRUNC
                                    if val_f64 > actual_thresh {
                                        actual_thresh
                                    } else {
                                        val_f64
                                    }
                                }
                                3 => {
                                    // TOZERO
                                    if val_f64 > actual_thresh {
                                        val_f64
                                    } else {
                                        0.0
                                    }
                                }
                                4 => {
                                    // TOZERO_INV
                                    if val_f64 > actual_thresh {
                                        0.0
                                    } else {
                                        val_f64
                                    }
                                }
                                _ => val_f64,
                            };

                            *comp = T::from_f64(result).unwrap_or_default();
                        }
                    }
                }
            });
    }

    #[cfg(not(feature = "parallel"))]
    {
        for y in 0..rows {
            for x in 0..cols {
                for c in 0..channels {
                    if let Some(src_val) = src.at(y as i32, x as i32, c) {
                        let val_f64 = src_val.to_f64().unwrap_or(0.0);

                        let result = match type_val {
                            0 => {
                                // BINARY
                                if val_f64 > actual_thresh {
                                    maxval
                                } else {
                                    0.0
                                }
                            }
                            1 => {
                                // BINARY_INV
                                if val_f64 > actual_thresh {
                                    0.0
                                } else {
                                    maxval
                                }
                            }
                            2 => {
                                // TRUNC
                                if val_f64 > actual_thresh {
                                    actual_thresh
                                } else {
                                    val_f64
                                }
                            }
                            3 => {
                                // TOZERO
                                if val_f64 > actual_thresh {
                                    val_f64
                                } else {
                                    0.0
                                }
                            }
                            4 => {
                                // TOZERO_INV
                                if val_f64 > actual_thresh {
                                    0.0
                                } else {
                                    val_f64
                                }
                            }
                            _ => val_f64,
                        };

                        *dst.at_mut(y as i32, x as i32, c).unwrap() =
                            T::from_f64(result).unwrap_or_default();
                    }
                }
            }
        }
    }

    Ok((actual_thresh, dst))
}

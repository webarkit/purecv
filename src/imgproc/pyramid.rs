/*
 *  pyramid.rs
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

//! Gaussian image pyramid operations: [`pyr_down`], [`pyr_up`], [`build_pyramid`].
//!
//! These functions implement the classical Gaussian pyramid using the 5-tap
//! kernel `[1, 4, 6, 4, 1]` (sum = 16, outer product sum = 256).

use crate::core::error::{PureCvError, Result};
use crate::core::simd::SimdElement;
use crate::core::types::{BorderTypes, Size};
use crate::core::utils::border_interpolate;
use crate::core::Matrix;
use num_traits::{FromPrimitive, ToPrimitive};

#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// 5-tap Gaussian kernel weights: [1, 4, 6, 4, 1].
const GAUSS5: [i32; 5] = [1, 4, 6, 4, 1];

// ---------------------------------------------------------------------------
//  pyr_down
// ---------------------------------------------------------------------------

/// Blurs an image and downsamples it (Gaussian pyramid downscale).
///
/// The function performs the downsampling step of the Gaussian pyramid
/// construction. It first convolves the source image with the 5-tap
/// kernel `[1, 4, 6, 4, 1]` (separable), then subsamples by taking
/// every other pixel in each direction.
///
/// # Arguments
/// * `src`         - Input image.
/// * `dst_size`    - Size of the output image. `None` defaults to
///   `((src.cols + 1) / 2, (src.rows + 1) / 2)`.
/// * `border_type` - Border interpolation method.
///
/// # Errors
/// Returns an error if `dst_size` is invalid.
pub fn pyr_down<T>(
    src: &Matrix<T>,
    dst_size: Option<Size<usize>>,
    border_type: BorderTypes,
) -> Result<Matrix<T>>
where
    T: Default + Clone + Copy + Send + Sync + ToPrimitive + FromPrimitive + SimdElement + 'static,
{
    let src_rows = src.rows;
    let src_cols = src.cols;
    let channels = src.channels;

    let dst_sz = dst_size.unwrap_or(Size::new(src_cols.div_ceil(2), src_rows.div_ceil(2)));

    if dst_sz.width == 0 || dst_sz.height == 0 {
        return Err(PureCvError::InvalidInput(
            "Destination size must be > 0".to_string(),
        ));
    }

    // Step 1: Horizontal convolution → intermediate buffer (full height, dst_cols width)
    // We only need columns that will be subsampled (every other column).
    let mut tmp = vec![0.0f64; src_rows * dst_sz.width * channels];

    // Scalar fallback for boundary pixels
    let process_row_h_scalar = |row: usize, out: &mut [f64]| {
        for dc in 0..dst_sz.width {
            let sc = (dc * 2) as i32; // source column centre
            for ch in 0..channels {
                let mut sum = 0.0f64;
                for k in 0..5i32 {
                    let col_idx = sc + k - 2;
                    let bc = border_interpolate(col_idx, src_cols as i32, border_type);
                    let val = if bc < 0 {
                        0.0
                    } else {
                        src.get(row, bc as usize, ch)
                            .map(|v| v.to_f64().unwrap_or(0.0))
                            .unwrap_or(0.0)
                    };
                    sum += val * GAUSS5[k as usize] as f64;
                }
                out[dc * channels + ch] = sum;
            }
        }
    };

    // SIMD-accelerated horizontal pass for interior pixels
    #[cfg(feature = "simd")]
    let process_row_h = |row: usize, out: &mut [f64]| {
        if !T::has_simd() || channels == 0 {
            process_row_h_scalar(row, out);
            return;
        }

        let row_start = row * src_cols * channels;
        let src_row = &src.data[row_start..row_start + src_cols * channels];

        // Process interior columns where the 5-tap window fits
        // dc maps to source column sc = dc * 2, window is [sc-2 .. sc+2]
        // Interior when sc >= 2 and sc + 2 < src_cols → dc >= 1 and dc * 2 + 2 < src_cols
        let dc_start = 1usize;
        let dc_end = if src_cols >= 4 {
            ((src_cols - 3) / 2) + 1
        } else {
            0
        };

        // Process boundary columns (left) with scalar
        for dc in 0..dc_start.min(dst_sz.width) {
            let sc = (dc * 2) as i32;
            for ch in 0..channels {
                let mut sum = 0.0f64;
                for k in 0..5i32 {
                    let col_idx = sc + k - 2;
                    let bc = border_interpolate(col_idx, src_cols as i32, border_type);
                    let val = if bc < 0 {
                        0.0
                    } else {
                        src.get(row, bc as usize, ch)
                            .map(|v| v.to_f64().unwrap_or(0.0))
                            .unwrap_or(0.0)
                    };
                    sum += val * GAUSS5[k as usize] as f64;
                }
                out[dc * channels + ch] = sum;
            }
        }

        // Interior columns via SIMD
        if dc_end > dc_start {
            for ch in 0..channels {
                // Build a channel-specific contiguous slice for the SIMD kernel
                // For single-channel images, we can use the row directly
                if channels == 1 {
                    #[allow(clippy::needless_range_loop)]
                    for dc in dc_start..dc_end.min(dst_sz.width) {
                        let sc = dc * 2;
                        let src_offset = sc - 2; // start of 5-tap window
                        let src_slice = &src_row[src_offset..src_offset + 5];
                        let mut dst_val = [0.0f64; 1];
                        T::simd_gaussian_5tap_h(&mut dst_val, src_slice, 1);
                        out[dc] = dst_val[0];
                    }
                } else {
                    // Multi-channel: use stride = channels
                    for dc in dc_start..dc_end.min(dst_sz.width) {
                        let sc = dc * 2;
                        let src_offset = (sc - 2) * channels + ch;
                        let src_slice = &src_row[src_offset..src_offset + 4 * channels + 1];
                        let mut dst_val = [0.0f64; 1];
                        T::simd_gaussian_5tap_h(&mut dst_val, src_slice, channels);
                        out[dc * channels + ch] = dst_val[0];
                    }
                }
            }
        }

        // Process boundary columns (right) with scalar
        for dc in dc_end.max(dc_start)..dst_sz.width {
            let sc = (dc * 2) as i32;
            for ch in 0..channels {
                let mut sum = 0.0f64;
                for k in 0..5i32 {
                    let col_idx = sc + k - 2;
                    let bc = border_interpolate(col_idx, src_cols as i32, border_type);
                    let val = if bc < 0 {
                        0.0
                    } else {
                        src.get(row, bc as usize, ch)
                            .map(|v| v.to_f64().unwrap_or(0.0))
                            .unwrap_or(0.0)
                    };
                    sum += val * GAUSS5[k as usize] as f64;
                }
                out[dc * channels + ch] = sum;
            }
        }
    };

    #[cfg(not(feature = "simd"))]
    let process_row_h = process_row_h_scalar;

    // Run horizontal pass
    #[cfg(feature = "parallel")]
    {
        let row_stride = dst_sz.width * channels;
        tmp.par_chunks_mut(row_stride)
            .enumerate()
            .for_each(|(row, out)| {
                process_row_h(row, out);
            });
    }

    #[cfg(not(feature = "parallel"))]
    {
        let row_stride = dst_sz.width * channels;
        for row in 0..src_rows {
            let start = row * row_stride;
            let end = start + row_stride;
            process_row_h(row, &mut tmp[start..end]);
        }
    }

    // Step 2: Vertical convolution on `tmp` + subsample rows
    let mut dst = Matrix::<T>::new(dst_sz.height, dst_sz.width, channels);
    let tmp_row_stride = dst_sz.width * channels;

    let get_tmp_val = |row: i32, col: usize, ch: usize| -> f64 {
        let br = border_interpolate(row, src_rows as i32, border_type);
        if br < 0 {
            0.0
        } else {
            tmp[br as usize * tmp_row_stride + col * channels + ch]
        }
    };

    // Scalar vertical pass for one destination row
    let vert_scalar = |dr: usize, dst_row: &mut [T]| {
        let sr = (dr * 2) as i32;
        for dc in 0..dst_sz.width {
            for ch in 0..channels {
                let mut sum = 0.0f64;
                for k in 0..5i32 {
                    let row_idx = sr + k - 2;
                    sum += get_tmp_val(row_idx, dc, ch) * GAUSS5[k as usize] as f64;
                }
                // Normalize by 256 (16 × 16 from separable kernel)
                let val = (sum / 256.0).round();
                dst_row[dc * channels + ch] = T::from_f64(val).unwrap_or_else(T::default);
            }
        }
    };

    // SIMD vertical pass: process entire row at once when all 5 source rows
    // are in-bounds (no border handling needed).
    #[cfg(feature = "simd")]
    let vert_row = |dr: usize, dst_row: &mut [T]| {
        if !T::has_simd() {
            vert_scalar(dr, dst_row);
            return;
        }
        let sr = dr * 2;
        // Interior check: all 5 rows (sr-2..sr+2) must be in [0, src_rows)
        if sr >= 2 && sr + 2 < src_rows {
            // Gather 5 row slices from tmp
            let r0 = &tmp[(sr - 2) * tmp_row_stride..(sr - 1) * tmp_row_stride];
            let r1 = &tmp[(sr - 1) * tmp_row_stride..sr * tmp_row_stride];
            let r2 = &tmp[sr * tmp_row_stride..(sr + 1) * tmp_row_stride];
            let r3 = &tmp[(sr + 1) * tmp_row_stride..(sr + 2) * tmp_row_stride];
            let r4 = &tmp[(sr + 2) * tmp_row_stride..(sr + 3) * tmp_row_stride];
            let rows: [&[f64]; 5] = [r0, r1, r2, r3, r4];

            T::simd_gaussian_5tap_v(dst_row, &rows);
        } else {
            vert_scalar(dr, dst_row);
        }
    };

    #[cfg(not(feature = "simd"))]
    let vert_row = vert_scalar;

    #[cfg(feature = "parallel")]
    {
        let dst_row_stride = dst_sz.width * channels;
        dst.data
            .par_chunks_mut(dst_row_stride)
            .enumerate()
            .for_each(|(dr, dst_row)| {
                vert_row(dr, dst_row);
            });
    }

    #[cfg(not(feature = "parallel"))]
    {
        let dst_row_stride = dst_sz.width * channels;
        for dr in 0..dst_sz.height {
            let start = dr * dst_row_stride;
            let end = start + dst_row_stride;
            vert_row(dr, &mut dst.data[start..end]);
        }
    }

    Ok(dst)
}

// ---------------------------------------------------------------------------
//  pyr_up
// ---------------------------------------------------------------------------

/// Upsamples an image and then blurs it (Gaussian pyramid upscale).
///
/// The function performs the upsampling step of the Gaussian pyramid.
/// It first upsamples the source (inserting zeros) and then convolves
/// with the 5-tap kernel `[1, 4, 6, 4, 1]` (separable), scaled ×4.
///
/// # Arguments
/// * `src`         - Input image.
/// * `dst_size`    - Size of the output image. `None` defaults to
///   `(src.cols * 2, src.rows * 2)`.
/// * `border_type` - Border interpolation method.
///
/// # Errors
/// Returns an error if `dst_size` is invalid.
pub fn pyr_up<T>(
    src: &Matrix<T>,
    dst_size: Option<Size<usize>>,
    border_type: BorderTypes,
) -> Result<Matrix<T>>
where
    T: Default + Clone + Copy + Send + Sync + ToPrimitive + FromPrimitive + SimdElement + 'static,
{
    let src_rows = src.rows;
    let src_cols = src.cols;
    let channels = src.channels;

    let dst_sz = dst_size.unwrap_or(Size::new(src_cols * 2, src_rows * 2));

    if dst_sz.width == 0 || dst_sz.height == 0 {
        return Err(PureCvError::InvalidInput(
            "Destination size must be > 0".to_string(),
        ));
    }

    // Step 1: Create upsampled (zero-interleaved) buffer at dst size
    // Step 2: Horizontal convolution
    let mut tmp = vec![0.0f64; dst_sz.height * dst_sz.width * channels];
    let tmp_row_stride = dst_sz.width * channels;

    // Helper: retrieve source value at (src_row, src_col) with border handling
    let get_src = |row: i32, col: i32, ch: usize| -> f64 {
        let br = border_interpolate(row, src_rows as i32, border_type);
        let bc = border_interpolate(col, src_cols as i32, border_type);
        if br < 0 || bc < 0 {
            0.0
        } else {
            src.get(br as usize, bc as usize, ch)
                .map(|v| v.to_f64().unwrap_or(0.0))
                .unwrap_or(0.0)
        }
    };

    // Horizontal pass: for each row in dst, compute the horizontal convolution
    // of the upsampled row. An upsampled row has non-zero values only at even
    // positions (mapping back to source columns).
    #[cfg(feature = "parallel")]
    {
        tmp.par_chunks_mut(tmp_row_stride)
            .enumerate()
            .for_each(|(dr, out)| {
                let src_row = dr as i32 / 2;
                let is_even_row = (dr % 2) == 0;

                for dc in 0..dst_sz.width {
                    for ch in 0..channels {
                        let mut sum = 0.0f64;
                        for k in 0..5i32 {
                            let up_col = dc as i32 + k - 2;
                            // In the upsampled image, only even columns have data
                            if up_col % 2 == 0 {
                                let src_col = up_col / 2;
                                if is_even_row {
                                    sum +=
                                        get_src(src_row, src_col, ch) * GAUSS5[k as usize] as f64;
                                }
                                // odd rows are all zero → nothing to add
                            }
                        }
                        out[dc * channels + ch] = sum;
                    }
                }
            });
    }

    #[cfg(not(feature = "parallel"))]
    {
        for dr in 0..dst_sz.height {
            let src_row = dr as i32 / 2;
            let is_even_row = (dr % 2) == 0;
            let row_start = dr * tmp_row_stride;

            for dc in 0..dst_sz.width {
                for ch in 0..channels {
                    let mut sum = 0.0f64;
                    for k in 0..5i32 {
                        let up_col = dc as i32 + k - 2;
                        if up_col % 2 == 0 {
                            let src_col = up_col / 2;
                            if is_even_row {
                                sum += get_src(src_row, src_col, ch) * GAUSS5[k as usize] as f64;
                            }
                        }
                    }
                    tmp[row_start + dc * channels + ch] = sum;
                }
            }
        }
    }

    // Step 3: Vertical convolution on tmp
    let mut dst = Matrix::<T>::new(dst_sz.height, dst_sz.width, channels);

    let get_tmp_row_val = |row: i32, col: usize, ch: usize| -> f64 {
        let br = border_interpolate(row, dst_sz.height as i32, border_type);
        if br < 0 {
            0.0
        } else {
            tmp[br as usize * tmp_row_stride + col * channels + ch]
        }
    };

    #[cfg(feature = "parallel")]
    {
        let dst_row_stride = dst_sz.width * channels;
        dst.data
            .par_chunks_mut(dst_row_stride)
            .enumerate()
            .for_each(|(dr, dst_row)| {
                for dc in 0..dst_sz.width {
                    for ch in 0..channels {
                        let mut sum = 0.0f64;
                        for k in 0..5i32 {
                            let row_idx = dr as i32 + k - 2;
                            sum += get_tmp_row_val(row_idx, dc, ch) * GAUSS5[k as usize] as f64;
                        }
                        // Scale by 4 to compensate for zero-insertion, then /256
                        // → total factor = 4/256 = 1/64
                        let val = (sum * 4.0 / 256.0).round();
                        dst_row[dc * channels + ch] = T::from_f64(val).unwrap_or_else(T::default);
                    }
                }
            });
    }

    #[cfg(not(feature = "parallel"))]
    {
        for dr in 0..dst_sz.height {
            for dc in 0..dst_sz.width {
                for ch in 0..channels {
                    let mut sum = 0.0f64;
                    for k in 0..5i32 {
                        let row_idx = dr as i32 + k - 2;
                        sum += get_tmp_row_val(row_idx, dc, ch) * GAUSS5[k as usize] as f64;
                    }
                    let val = (sum * 4.0 / 256.0).round();
                    dst.set(dr, dc, ch, T::from_f64(val).unwrap_or_else(T::default));
                }
            }
        }
    }

    Ok(dst)
}

// ---------------------------------------------------------------------------
//  build_pyramid
// ---------------------------------------------------------------------------

/// Constructs a Gaussian pyramid for an image.
///
/// The function builds `max_level + 1` levels of the Gaussian pyramid.
/// Level 0 is the original image, and each subsequent level is the
/// result of [`pyr_down`] applied to the previous level.
///
/// # Arguments
/// * `src`         - Source image (becomes level 0).
/// * `max_level`   - 0-based index of the last (smallest) pyramid level
///   to build. The returned vector has `max_level + 1`
///   elements.
/// * `border_type` - Border interpolation method.
///
/// # Example
/// ```
/// use purecv::core::Matrix;
/// use purecv::core::types::BorderTypes;
/// use purecv::imgproc::pyramid::build_pyramid;
///
/// let src = Matrix::<u8>::new(64, 64, 1);
/// let pyramid = build_pyramid(&src, 3, BorderTypes::Reflect101).unwrap();
/// assert_eq!(pyramid.len(), 4); // levels 0..3
/// assert_eq!(pyramid[0].rows, 64);
/// assert_eq!(pyramid[1].rows, 32);
/// assert_eq!(pyramid[2].rows, 16);
/// assert_eq!(pyramid[3].rows, 8);
/// ```
pub fn build_pyramid<T>(
    src: &Matrix<T>,
    max_level: usize,
    border_type: BorderTypes,
) -> Result<Vec<Matrix<T>>>
where
    T: Default + Clone + Copy + Send + Sync + ToPrimitive + FromPrimitive + SimdElement + 'static,
{
    let mut pyramid = Vec::with_capacity(max_level + 1);
    pyramid.push(src.clone());

    for i in 0..max_level {
        let prev = &pyramid[i];
        let next = pyr_down(prev, None, border_type)?;
        pyramid.push(next);
    }

    Ok(pyramid)
}

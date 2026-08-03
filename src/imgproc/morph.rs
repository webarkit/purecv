/*
 *  morph.rs
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

use alloc::{string::ToString, vec::Vec};
// `vec!` is only used by the SIMD-only separable kernel below.
#[cfg(feature = "simd")]
use alloc::vec;
#[allow(unused_imports)]
use num_traits::Float;

use crate::core::arithm;
use crate::core::error::{PureCvError, Result};
use crate::core::simd::SimdElement;
use crate::core::types::{BorderTypes, Point, Size};
use crate::core::utils::border_interpolate;
use crate::core::Matrix;
use num_traits::{Bounded, FromPrimitive, Num, ToPrimitive};

#[cfg(feature = "parallel")]
use rayon::prelude::*;

// ---------------------------------------------------------------------------
//  Enums
// ---------------------------------------------------------------------------

/// Shape of the structuring element used for morphological operations.
///
/// Mirrors OpenCV's `MorphShapes`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MorphShapes {
    /// Rectangular structuring element: all 1s.
    Rect = 0,
    /// Cross-shaped structuring element: center row and center column are 1.
    Cross = 1,
    /// Elliptical structuring element: an ellipse inscribed in the rectangle.
    Ellipse = 2,
}

/// Type of morphological operation.
///
/// Mirrors OpenCV's `MorphTypes`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MorphTypes {
    /// Erosion: replaces each pixel with the local minimum.
    Erode = 0,
    /// Dilation: replaces each pixel with the local maximum.
    Dilate = 1,
    /// Opening: erosion followed by dilation.
    Open = 2,
    /// Closing: dilation followed by erosion.
    Close = 3,
    /// Morphological gradient: dilation minus erosion.
    Gradient = 4,
    /// Top-hat: source minus opening.
    TopHat = 5,
    /// Black-hat: closing minus source.
    BlackHat = 6,
}

// ---------------------------------------------------------------------------
//  Structuring Element
// ---------------------------------------------------------------------------

/// Returns a structuring element of the specified size and shape for
/// morphological operations.
///
/// The function constructs and returns the structuring element that can be
/// further passed to [`erode`], [`dilate`], or [`morphology_ex`].
///
/// # Arguments
/// * `shape`  - Shape of the structuring element ([`MorphShapes`]).
/// * `ksize`  - Size of the structuring element.
/// * `anchor` - Anchor position within the element.
///   `Point::new(-1, -1)` means the anchor is at the centre.
///
/// # Errors
/// Returns [`PureCvError::InvalidInput`] if ksize dimensions are zero.
///
/// # Example
/// ```
/// use purecv::core::{Point, Size};
/// use purecv::imgproc::morph::{get_structuring_element, MorphShapes};
///
/// let kernel = get_structuring_element(
///     MorphShapes::Rect,
///     Size::new(3_usize, 3_usize),
///     Point::new(-1_i32, -1_i32),
/// ).unwrap();
/// assert_eq!(kernel.rows, 3);
/// assert_eq!(kernel.cols, 3);
/// // A 3×3 rect kernel is all 1s
/// assert!(kernel.data.iter().all(|&v| v == 1));
/// ```
pub fn get_structuring_element(
    shape: MorphShapes,
    ksize: Size<usize>,
    anchor: Point<i32>,
) -> Result<Matrix<u8>> {
    if ksize.width == 0 || ksize.height == 0 {
        return Err(PureCvError::InvalidInput(
            "Structuring element size must be > 0".to_string(),
        ));
    }

    // Normalize anchor: (-1, -1) means centre
    let anchor_x = if anchor.x < 0 {
        (ksize.width / 2) as i32
    } else {
        anchor.x
    };
    let anchor_y = if anchor.y < 0 {
        (ksize.height / 2) as i32
    } else {
        anchor.y
    };

    // Degenerate 1×1 kernel is always rect
    let shape = if ksize.width == 1 && ksize.height == 1 {
        MorphShapes::Rect
    } else {
        shape
    };

    let r = (ksize.height / 2) as i32;
    let c = (ksize.width / 2) as i32;
    let inv_r2: f64 = if r > 0 {
        1.0 / (r as f64 * r as f64)
    } else {
        0.0
    };

    let mut elem = Matrix::<u8>::new(ksize.height, ksize.width, 1);

    for i in 0..ksize.height {
        let ii = i as i32;
        let (j1, j2) = match shape {
            MorphShapes::Rect => (0i32, ksize.width as i32),
            MorphShapes::Cross => {
                if ii == anchor_y {
                    (0, ksize.width as i32)
                } else {
                    (anchor_x, anchor_x + 1)
                }
            }
            MorphShapes::Ellipse => {
                let dy = ii - r;
                if dy.abs() <= r {
                    let dx = (c as f64
                        * ((r as f64 * r as f64 - dy as f64 * dy as f64) * inv_r2).sqrt())
                        as i32;
                    let j1 = (c - dx).max(0);
                    let j2 = (c + dx + 1).min(ksize.width as i32);
                    (j1, j2)
                } else {
                    (0, 0) // empty row
                }
            }
        };

        for j in 0..ksize.width {
            let jj = j as i32;
            let val = if jj >= j1 && jj < j2 { 1u8 } else { 0u8 };
            elem.set(i, j, 0, val);
        }
    }

    Ok(elem)
}

// ---------------------------------------------------------------------------
//  Internal: single-pass morphology primitive
// ---------------------------------------------------------------------------

/// Internal function that performs a single pass of morphological
/// erosion (min) or dilation (max).
fn morph_op<T>(
    src: &Matrix<T>,
    kernel: &Matrix<u8>,
    anchor: Point<i32>,
    border_type: BorderTypes,
    is_erode: bool,
) -> Result<Matrix<T>>
where
    T: Default + Clone + Copy + PartialOrd + Send + Sync + Bounded + SimdElement + 'static,
{
    let rows = src.rows;
    let cols = src.cols;
    let channels = src.channels;
    let krows = kernel.rows;
    let kcols = kernel.cols;

    // Normalize anchor
    let ax = if anchor.x < 0 {
        (kcols / 2) as i32
    } else {
        anchor.x
    };
    let ay = if anchor.y < 0 {
        (krows / 2) as i32
    } else {
        anchor.y
    };

    // -----------------------------------------------------------------------
    // SIMD separable fast-path for rectangular (all-1s) kernels
    // -----------------------------------------------------------------------
    #[cfg(feature = "simd")]
    {
        let is_rect = kernel.data.iter().all(|&v| v != 0);
        if is_rect && T::has_simd() && channels > 0 {
            return morph_op_separable_simd(src, krows, kcols, ax, ay, border_type, is_erode);
        }
    }

    // -----------------------------------------------------------------------
    // Generic offset-based path (non-rectangular kernels or no SIMD)
    // -----------------------------------------------------------------------

    // Pre-compute active kernel positions (where kernel value != 0)
    let mut kernel_offsets: Vec<(i32, i32)> = Vec::new();
    for kr in 0..krows {
        for kc in 0..kcols {
            if *kernel.get(kr, kc, 0).unwrap_or(&0) != 0 {
                kernel_offsets.push((kr as i32 - ay, kc as i32 - ax));
            }
        }
    }

    let mut dst = Matrix::<T>::new(rows, cols, channels);

    // Pixel processing closure — processes one row
    let process_row = |row: usize, dst_row: &mut [T]| {
        for col in 0..cols {
            for ch in 0..channels {
                let mut acc = if is_erode {
                    T::max_value()
                } else {
                    T::min_value()
                };

                for &(dy, dx) in &kernel_offsets {
                    let sr = row as i32 + dy;
                    let sc = col as i32 + dx;

                    let br = border_interpolate(sr, rows as i32, border_type);
                    let bc = border_interpolate(sc, cols as i32, border_type);

                    // BorderTypes::Constant returns -1 for out-of-bounds
                    if br < 0 || bc < 0 {
                        // For erode with Constant border, the default value is max
                        // (so out-of-bounds pixels don't affect the min).
                        // For dilate, the default is min.
                        // Both are already the init value of `acc`, so skip.
                        continue;
                    }

                    let val = *src
                        .get(br as usize, bc as usize, ch)
                        .unwrap_or(&if is_erode {
                            T::max_value()
                        } else {
                            T::min_value()
                        });

                    if is_erode {
                        if val < acc {
                            acc = val;
                        }
                    } else if val > acc {
                        acc = val;
                    }
                }

                let out_idx = (col * channels) + ch;
                dst_row[out_idx] = acc;
            }
        }
    };

    let row_stride = cols * channels;

    #[cfg(feature = "parallel")]
    {
        dst.data
            .par_chunks_mut(row_stride)
            .enumerate()
            .for_each(|(row, dst_row)| {
                process_row(row, dst_row);
            });
    }

    #[cfg(not(feature = "parallel"))]
    {
        for row in 0..rows {
            let start = row * row_stride;
            let end = start + row_stride;
            let dst_row = &mut dst.data[start..end];
            process_row(row, dst_row);
        }
    }

    Ok(dst)
}

// ---------------------------------------------------------------------------
//  Separable SIMD morphology for rectangular kernels
// ---------------------------------------------------------------------------

/// SIMD-accelerated separable morphology for rectangular (all-1s) kernels.
///
/// Pass 1: Horizontal min/max → intermediate buffer.
/// Pass 2: Vertical min/max → final output.
#[cfg(feature = "simd")]
fn morph_op_separable_simd<T>(
    src: &Matrix<T>,
    krows: usize,
    kcols: usize,
    ax: i32,
    ay: i32,
    border_type: BorderTypes,
    is_erode: bool,
) -> Result<Matrix<T>>
where
    T: Default + Clone + Copy + PartialOrd + Send + Sync + Bounded + SimdElement + 'static,
{
    let rows = src.rows;
    let cols = src.cols;
    let channels = src.channels;
    let row_stride = cols * channels;

    // Step 1: Horizontal min/max into `tmp`
    // We pad each source row by kcols-1 elements using border interpolation,
    // then run simd_row_min_max across the padded row.
    let mut tmp = vec![
        if is_erode {
            T::max_value()
        } else {
            T::min_value()
        };
        rows * row_stride
    ];

    let process_row_h = |row: usize, dst_row: &mut [T]| {
        let pad_left = ax as usize;
        let pad_right = (kcols as i32 - 1 - ax) as usize;
        let padded_len = pad_left + cols * channels + pad_right;
        let mut padded = vec![
            if is_erode {
                T::max_value()
            } else {
                T::min_value()
            };
            padded_len
        ];

        // Fill padded buffer
        for col in 0..cols {
            for ch in 0..channels {
                padded[pad_left + col * channels + ch] =
                    *src.get(row, col, ch).unwrap_or(&T::default());
            }
        }

        // Fill left border
        for i in 0..pad_left {
            let src_col = -(pad_left as i32) + i as i32;
            for ch in 0..channels {
                let bc = border_interpolate(src_col, cols as i32, border_type);
                if bc >= 0 {
                    padded[i * channels + ch] =
                        *src.get(row, bc as usize, ch).unwrap_or(&T::default());
                }
            }
        }

        // Fill right border
        for i in 0..pad_right {
            let src_col = cols as i32 + i as i32;
            for ch in 0..channels {
                let bc = border_interpolate(src_col, cols as i32, border_type);
                if bc >= 0 {
                    padded[(pad_left + cols * channels) + i * channels + ch] =
                        *src.get(row, bc as usize, ch).unwrap_or(&T::default());
                }
            }
        }

        // SIMD horizontal min/max
        if channels == 1 {
            T::simd_row_min_max(dst_row, &padded, kcols, is_erode);
        } else {
            // Per-channel: stride through padded to extract single-channel strips
            for ch in 0..channels {
                // For multi-channel, we process element-by-element with the stride
                for col in 0..cols {
                    let mut acc = padded[col * channels + ch];
                    for k in 1..kcols {
                        let val = padded[(col + k) * channels + ch];
                        if is_erode {
                            if val < acc {
                                acc = val;
                            }
                        } else if val > acc {
                            acc = val;
                        }
                    }
                    dst_row[col * channels + ch] = acc;
                }
            }
        }
    };

    #[cfg(feature = "parallel")]
    {
        tmp.par_chunks_mut(row_stride)
            .enumerate()
            .for_each(|(row, out)| {
                process_row_h(row, out);
            });
    }

    #[cfg(not(feature = "parallel"))]
    {
        for row in 0..rows {
            let start = row * row_stride;
            let end = start + row_stride;
            process_row_h(row, &mut tmp[start..end]);
        }
    }

    // Step 2: Vertical min/max from `tmp` → `dst`
    let mut dst = Matrix::<T>::new(rows, cols, channels);

    let process_row_v = |row: usize, dst_row: &mut [T]| {
        // Gather row slices (with border handling)
        let mut row_ptrs: Vec<usize> = Vec::with_capacity(krows);
        for k in 0..krows {
            let sr = row as i32 + k as i32 - ay;
            let br = border_interpolate(sr, rows as i32, border_type);
            let r = if br < 0 { row } else { br as usize };
            row_ptrs.push(r);
        }

        // Check if all rows are in-bounds (can skip border handling)
        let first_sr = row as i32 - ay;
        let last_sr = row as i32 + (krows as i32 - 1 - ay);
        let all_interior = first_sr >= 0 && last_sr < rows as i32;

        if all_interior {
            // Gather slices for SIMD
            let slices: Vec<&[T]> = row_ptrs
                .iter()
                .map(|&r| &tmp[r * row_stride..(r + 1) * row_stride])
                .collect();
            T::simd_min_max_col(dst_row, &slices, is_erode);
        } else {
            // Scalar fallback for boundary rows
            let slices: Vec<&[T]> = row_ptrs
                .iter()
                .map(|&r| &tmp[r * row_stride..(r + 1) * row_stride])
                .collect();
            for i in 0..row_stride {
                let mut acc = slices[0][i];
                for s in &slices[1..] {
                    let val = s[i];
                    if is_erode {
                        if val < acc {
                            acc = val;
                        }
                    } else if val > acc {
                        acc = val;
                    }
                }
                dst_row[i] = acc;
            }
        }
    };

    #[cfg(feature = "parallel")]
    {
        dst.data
            .par_chunks_mut(row_stride)
            .enumerate()
            .for_each(|(row, out)| {
                process_row_v(row, out);
            });
    }

    #[cfg(not(feature = "parallel"))]
    {
        for row in 0..rows {
            let start = row * row_stride;
            let end = start + row_stride;
            process_row_v(row, &mut dst.data[start..end]);
        }
    }

    Ok(dst)
}

// ---------------------------------------------------------------------------
//  Erode & Dilate
// ---------------------------------------------------------------------------

/// Erodes an image by using a specific structuring element.
///
/// The function erodes the source image using the specified structuring
/// element that determines the shape of a pixel neighbourhood over which
/// the minimum is taken.
///
/// # Arguments
/// * `src`         - Input image.
/// * `kernel`      - Structuring element (see [`get_structuring_element`]).
/// * `anchor`      - Anchor position. `Point::new(-1, -1)` = centre.
/// * `iterations`  - Number of times erosion is applied.
/// * `border_type` - Border extrapolation method.
///
/// # Errors
/// Returns an error if the kernel is empty.
pub fn erode<T>(
    src: &Matrix<T>,
    kernel: &Matrix<u8>,
    anchor: Point<i32>,
    iterations: usize,
    border_type: BorderTypes,
) -> Result<Matrix<T>>
where
    T: Default + Clone + Copy + PartialOrd + Send + Sync + Bounded + SimdElement + 'static,
{
    if kernel.data.is_empty() {
        return Err(PureCvError::InvalidInput(
            "Kernel must not be empty".to_string(),
        ));
    }
    if iterations == 0 {
        return Ok(src.clone());
    }

    let mut current = morph_op(src, kernel, anchor, border_type, true)?;
    for _ in 1..iterations {
        current = morph_op(&current, kernel, anchor, border_type, true)?;
    }
    Ok(current)
}

/// Dilates an image by using a specific structuring element.
///
/// The function dilates the source image using the specified structuring
/// element that determines the shape of a pixel neighbourhood over which
/// the maximum is taken.
///
/// # Arguments
/// * `src`         - Input image.
/// * `kernel`      - Structuring element (see [`get_structuring_element`]).
/// * `anchor`      - Anchor position. `Point::new(-1, -1)` = centre.
/// * `iterations`  - Number of times dilation is applied.
/// * `border_type` - Border extrapolation method.
///
/// # Errors
/// Returns an error if the kernel is empty.
pub fn dilate<T>(
    src: &Matrix<T>,
    kernel: &Matrix<u8>,
    anchor: Point<i32>,
    iterations: usize,
    border_type: BorderTypes,
) -> Result<Matrix<T>>
where
    T: Default + Clone + Copy + PartialOrd + Send + Sync + Bounded + SimdElement + 'static,
{
    if kernel.data.is_empty() {
        return Err(PureCvError::InvalidInput(
            "Kernel must not be empty".to_string(),
        ));
    }
    if iterations == 0 {
        return Ok(src.clone());
    }

    let mut current = morph_op(src, kernel, anchor, border_type, false)?;
    for _ in 1..iterations {
        current = morph_op(&current, kernel, anchor, border_type, false)?;
    }
    Ok(current)
}

// ---------------------------------------------------------------------------
//  morphology_ex
// ---------------------------------------------------------------------------

/// Performs advanced morphological transformations.
///
/// The function can perform advanced morphological transformations using
/// erosion and dilation as basic operations, matching OpenCV's
/// `cv::morphologyEx`.
///
/// # Arguments
/// * `src`         - Source image.
/// * `op`          - Type of morphological operation ([`MorphTypes`]).
/// * `kernel`      - Structuring element.
/// * `anchor`      - Anchor position. `Point::new(-1, -1)` = centre.
/// * `iterations`  - Number of times the basic operation is applied.
/// * `border_type` - Border extrapolation method.
///
/// # Operations
/// | `op`                       | Formula                    |
/// |----------------------------|----------------------------|
/// | [`MorphTypes::Erode`]      | erode(src)                 |
/// | [`MorphTypes::Dilate`]     | dilate(src)                |
/// | [`MorphTypes::Open`]       | dilate(erode(src))         |
/// | [`MorphTypes::Close`]      | erode(dilate(src))         |
/// | [`MorphTypes::Gradient`]   | dilate(src) − erode(src)   |
/// | [`MorphTypes::TopHat`]     | src − open(src)            |
/// | [`MorphTypes::BlackHat`]   | close(src) − src           |
pub fn morphology_ex<T>(
    src: &Matrix<T>,
    op: MorphTypes,
    kernel: &Matrix<u8>,
    anchor: Point<i32>,
    iterations: usize,
    border_type: BorderTypes,
) -> Result<Matrix<T>>
where
    T: Num
        + Default
        + Clone
        + Copy
        + PartialOrd
        + Send
        + Sync
        + Bounded
        + ToPrimitive
        + FromPrimitive
        + SimdElement
        + 'static,
{
    match op {
        MorphTypes::Erode => erode(src, kernel, anchor, iterations, border_type),
        MorphTypes::Dilate => dilate(src, kernel, anchor, iterations, border_type),
        MorphTypes::Open => {
            let eroded = erode(src, kernel, anchor, iterations, border_type)?;
            dilate(&eroded, kernel, anchor, iterations, border_type)
        }
        MorphTypes::Close => {
            let dilated = dilate(src, kernel, anchor, iterations, border_type)?;
            erode(&dilated, kernel, anchor, iterations, border_type)
        }
        MorphTypes::Gradient => {
            let eroded = erode(src, kernel, anchor, iterations, border_type)?;
            let dilated = dilate(src, kernel, anchor, iterations, border_type)?;
            arithm::subtract(&dilated, &eroded)
        }
        MorphTypes::TopHat => {
            let opened = morphology_ex(
                src,
                MorphTypes::Open,
                kernel,
                anchor,
                iterations,
                border_type,
            )?;
            arithm::subtract(src, &opened)
        }
        MorphTypes::BlackHat => {
            let closed = morphology_ex(
                src,
                MorphTypes::Close,
                kernel,
                anchor,
                iterations,
                border_type,
            )?;
            arithm::subtract(&closed, src)
        }
    }
}

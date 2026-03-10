/*
 *  color.rs
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

#[cfg(feature = "parallel")]
use rayon::prelude::*;

// Assuming the Matrix<T> struct is in scope
use crate::core::Matrix;

/// Color conversion codes for cvt_color.
/// Mimics OpenCV's cv::ColorConversionCodes.
#[allow(non_camel_case_types)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColorConversionCode {
    /// Convert BGR to Grayscale
    COLOR_BGR2GRAY,
    /// Convert RGB to Grayscale
    COLOR_RGB2GRAY,
    /// Convert BGRA to Grayscale
    COLOR_BGRA2GRAY,
    /// Convert RGBA to Grayscale
    COLOR_RGBA2GRAY,
    /// Convert Grayscale to RGB
    COLOR_GRAY2RGB,
    /// Convert Grayscale to BGR
    COLOR_GRAY2BGR,
    /// Convert Grayscale to RGBA
    COLOR_GRAY2RGBA,
    /// Convert Grayscale to BGRA
    COLOR_GRAY2BGRA,
}

/// Converts an image from one color space to another.
///
/// This is the main wrapper function that mimics OpenCV's `cv::cvtColor`.
///
/// Returns a Result containing the new Matrix, or an Error if the input is invalid.
pub fn cvt_color(src: &Matrix<u8>, code: ColorConversionCode) -> Result<Matrix<u8>, &'static str> {
    match code {
        ColorConversionCode::COLOR_RGB2GRAY => cvt_color_rgb_to_gray(src),
        ColorConversionCode::COLOR_BGR2GRAY => cvt_color_bgr_to_gray(src),
        ColorConversionCode::COLOR_RGBA2GRAY => cvt_color_rgba_to_gray(src),
        ColorConversionCode::COLOR_BGRA2GRAY => cvt_color_bgra_to_gray(src),
        ColorConversionCode::COLOR_GRAY2RGB => cvt_color_gray_to_rgb(src),
        ColorConversionCode::COLOR_GRAY2BGR => cvt_color_gray_to_bgr(src),
        ColorConversionCode::COLOR_GRAY2RGBA => cvt_color_gray_to_rgba(src),
        ColorConversionCode::COLOR_GRAY2BGRA => cvt_color_gray_to_bgra(src),
    }
}

/// Converts an 8-bit RGB image to an 8-bit Grayscale image.\n/// Uses the standard luminosity formula: Y = 0.299*R + 0.587*G + 0.114*B
pub fn cvt_color_rgb_to_gray(input: &Matrix<u8>) -> Result<Matrix<u8>, &'static str> {
    if input.channels != 3 {
        return Err("Input matrix must have exactly 3 channels");
    }

    let mut output = Matrix::<u8>::new(input.rows, input.cols, 1);
    let out_row_len = output.cols;
    let in_row_len = input.cols * 3;

    #[cfg(feature = "parallel")]
    {
        output.data
            .par_chunks_exact_mut(out_row_len)
            .zip(input.data.par_chunks_exact(in_row_len))
            .for_each(|(out_row, in_row)| {
                for (out_pixel, in_val) in out_row.iter_mut().zip(in_row.chunks_exact(3)) {
                    let r = in_val[0] as f32;
                    let g = in_val[1] as f32;
                    let b = in_val[2] as f32;
                    *out_pixel = (0.299 * r + 0.587 * g + 0.114 * b).round() as u8;
                }
            });
    }

    #[cfg(not(feature = "parallel"))]
    {
        output.data
            .chunks_exact_mut(out_row_len)
            .zip(input.data.chunks_exact(in_row_len))
            .for_each(|(out_row, in_row)| {
                for (out_pixel, in_val) in out_row.iter_mut().zip(in_row.chunks_exact(3)) {
                    let r = in_val[0] as f32;
                    let g = in_val[1] as f32;
                    let b = in_val[2] as f32;
                    *out_pixel = (0.299 * r + 0.587 * g + 0.114 * b).round() as u8;
                }
            });
    }

    Ok(output)
}

/// Converts an 8-bit BGR image to an 8-bit Grayscale image.\n/// Uses the standard luminosity formula: Y = 0.299*R + 0.587*G + 0.114*B
pub fn cvt_color_bgr_to_gray(input: &Matrix<u8>) -> Result<Matrix<u8>, &'static str> {
    if input.channels != 3 {
        return Err("Input matrix must have exactly 3 channels");
    }

    let mut output = Matrix::<u8>::new(input.rows, input.cols, 1);
    let out_row_len = output.cols;
    let in_row_len = input.cols * 3;

    #[cfg(feature = "parallel")]
    {
        output.data
            .par_chunks_exact_mut(out_row_len)
            .zip(input.data.par_chunks_exact(in_row_len))
            .for_each(|(out_row, in_row)| {
                for (out_pixel, in_val) in out_row.iter_mut().zip(in_row.chunks_exact(3)) {
                    let b = in_val[0] as f32;
                    let g = in_val[1] as f32;
                    let r = in_val[2] as f32;
                    *out_pixel = (0.299 * r + 0.587 * g + 0.114 * b).round() as u8;
                }
            });
    }

    #[cfg(not(feature = "parallel"))]
    {
        output.data
            .chunks_exact_mut(out_row_len)
            .zip(input.data.chunks_exact(in_row_len))
            .for_each(|(out_row, in_row)| {
                for (out_pixel, in_val) in out_row.iter_mut().zip(in_row.chunks_exact(3)) {
                    let b = in_val[0] as f32;
                    let g = in_val[1] as f32;
                    let r = in_val[2] as f32;
                    *out_pixel = (0.299 * r + 0.587 * g + 0.114 * b).round() as u8;
                }
            });
    }

    Ok(output)
}

/// Converts an 8-bit RGBA image to an 8-bit Grayscale image.\n/// Uses the standard luminosity formula: Y = 0.299*R + 0.587*G + 0.114*B, ignores Alpha.
pub fn cvt_color_rgba_to_gray(input: &Matrix<u8>) -> Result<Matrix<u8>, &'static str> {
    if input.channels != 4 {
        return Err("Input matrix must have exactly 4 channels");
    }

    let mut output = Matrix::<u8>::new(input.rows, input.cols, 1);
    let out_row_len = output.cols;
    let in_row_len = input.cols * 4;

    #[cfg(feature = "parallel")]
    {
        output.data
            .par_chunks_exact_mut(out_row_len)
            .zip(input.data.par_chunks_exact(in_row_len))
            .for_each(|(out_row, in_row)| {
                for (out_pixel, in_val) in out_row.iter_mut().zip(in_row.chunks_exact(4)) {
                    let r = in_val[0] as f32;
                    let g = in_val[1] as f32;
                    let b = in_val[2] as f32;
                    *out_pixel = (0.299 * r + 0.587 * g + 0.114 * b).round() as u8;
                }
            });
    }

    #[cfg(not(feature = "parallel"))]
    {
        output.data
            .chunks_exact_mut(out_row_len)
            .zip(input.data.chunks_exact(in_row_len))
            .for_each(|(out_row, in_row)| {
                for (out_pixel, in_val) in out_row.iter_mut().zip(in_row.chunks_exact(4)) {
                    let r = in_val[0] as f32;
                    let g = in_val[1] as f32;
                    let b = in_val[2] as f32;
                    *out_pixel = (0.299 * r + 0.587 * g + 0.114 * b).round() as u8;
                }
            });
    }

    Ok(output)
}

/// Converts an 8-bit BGRA image to an 8-bit Grayscale image.\n/// Uses the standard luminosity formula: Y = 0.299*R + 0.587*G + 0.114*B, ignores Alpha.
pub fn cvt_color_bgra_to_gray(input: &Matrix<u8>) -> Result<Matrix<u8>, &'static str> {
    if input.channels != 4 {
        return Err("Input matrix must have exactly 4 channels");
    }

    let mut output = Matrix::<u8>::new(input.rows, input.cols, 1);
    let out_row_len = output.cols;
    let in_row_len = input.cols * 4;

    #[cfg(feature = "parallel")]
    {
        output.data
            .par_chunks_exact_mut(out_row_len)
            .zip(input.data.par_chunks_exact(in_row_len))
            .for_each(|(out_row, in_row)| {
                for (out_pixel, in_val) in out_row.iter_mut().zip(in_row.chunks_exact(4)) {
                    let b = in_val[0] as f32;
                    let g = in_val[1] as f32;
                    let r = in_val[2] as f32;
                    *out_pixel = (0.299 * r + 0.587 * g + 0.114 * b).round() as u8;
                }
            });
    }

    #[cfg(not(feature = "parallel"))]
    {
        output.data
            .chunks_exact_mut(out_row_len)
            .zip(input.data.chunks_exact(in_row_len))
            .for_each(|(out_row, in_row)| {
                for (out_pixel, in_val) in out_row.iter_mut().zip(in_row.chunks_exact(4)) {
                    let b = in_val[0] as f32;
                    let g = in_val[1] as f32;
                    let r = in_val[2] as f32;
                    *out_pixel = (0.299 * r + 0.587 * g + 0.114 * b).round() as u8;
                }
            });
    }

    Ok(output)
}

/// Converts an 8-bit Grayscale image to an 8-bit RGB image.
pub fn cvt_color_gray_to_rgb(input: &Matrix<u8>) -> Result<Matrix<u8>, &'static str> {
    if input.channels != 1 {
        return Err("Input matrix must have exactly 1 channels");
    }

    let mut output = Matrix::<u8>::new(input.rows, input.cols, 3);
    let out_row_len = output.cols * 3;
    let in_row_len = input.cols;

    #[cfg(feature = "parallel")]
    {
        output.data
            .par_chunks_exact_mut(out_row_len)
            .zip(input.data.par_chunks_exact(in_row_len))
            .for_each(|(out_row, in_row)| {
                for (out_val, in_pixel) in out_row.chunks_exact_mut(3).zip(in_row.iter()) {
                    let v = *in_pixel;
                    out_val[0] = v;
                    out_val[1] = v;
                    out_val[2] = v;
                }
            });
    }

    #[cfg(not(feature = "parallel"))]
    {
        output.data
            .chunks_exact_mut(out_row_len)
            .zip(input.data.chunks_exact(in_row_len))
            .for_each(|(out_row, in_row)| {
                for (out_val, in_pixel) in out_row.chunks_exact_mut(3).zip(in_row.iter()) {
                    let v = *in_pixel;
                    out_val[0] = v;
                    out_val[1] = v;
                    out_val[2] = v;
                }
            });
    }

    Ok(output)
}

/// Converts an 8-bit Grayscale image to an 8-bit BGR image.
pub fn cvt_color_gray_to_bgr(input: &Matrix<u8>) -> Result<Matrix<u8>, &'static str> {
    if input.channels != 1 {
        return Err("Input matrix must have exactly 1 channels");
    }

    let mut output = Matrix::<u8>::new(input.rows, input.cols, 3);
    let out_row_len = output.cols * 3;
    let in_row_len = input.cols;

    #[cfg(feature = "parallel")]
    {
        output.data
            .par_chunks_exact_mut(out_row_len)
            .zip(input.data.par_chunks_exact(in_row_len))
            .for_each(|(out_row, in_row)| {
                for (out_val, in_pixel) in out_row.chunks_exact_mut(3).zip(in_row.iter()) {
                    let v = *in_pixel;
                    out_val[0] = v;
                    out_val[1] = v;
                    out_val[2] = v;
                }
            });
    }

    #[cfg(not(feature = "parallel"))]
    {
        output.data
            .chunks_exact_mut(out_row_len)
            .zip(input.data.chunks_exact(in_row_len))
            .for_each(|(out_row, in_row)| {
                for (out_val, in_pixel) in out_row.chunks_exact_mut(3).zip(in_row.iter()) {
                    let v = *in_pixel;
                    out_val[0] = v;
                    out_val[1] = v;
                    out_val[2] = v;
                }
            });
    }

    Ok(output)
}

/// Converts an 8-bit Grayscale image to an 8-bit RGBA image.\n/// Alpha channel is set to 255 (fully opaque).
pub fn cvt_color_gray_to_rgba(input: &Matrix<u8>) -> Result<Matrix<u8>, &'static str> {
    if input.channels != 1 {
        return Err("Input matrix must have exactly 1 channels");
    }

    let mut output = Matrix::<u8>::new(input.rows, input.cols, 4);
    let out_row_len = output.cols * 4;
    let in_row_len = input.cols;

    #[cfg(feature = "parallel")]
    {
        output.data
            .par_chunks_exact_mut(out_row_len)
            .zip(input.data.par_chunks_exact(in_row_len))
            .for_each(|(out_row, in_row)| {
                for (out_val, in_pixel) in out_row.chunks_exact_mut(4).zip(in_row.iter()) {
                    let v = *in_pixel;
                    out_val[0] = v;
                    out_val[1] = v;
                    out_val[2] = v;
                    out_val[3] = 255;
                }
            });
    }

    #[cfg(not(feature = "parallel"))]
    {
        output.data
            .chunks_exact_mut(out_row_len)
            .zip(input.data.chunks_exact(in_row_len))
            .for_each(|(out_row, in_row)| {
                for (out_val, in_pixel) in out_row.chunks_exact_mut(4).zip(in_row.iter()) {
                    let v = *in_pixel;
                    out_val[0] = v;
                    out_val[1] = v;
                    out_val[2] = v;
                    out_val[3] = 255;
                }
            });
    }

    Ok(output)
}

/// Converts an 8-bit Grayscale image to an 8-bit BGRA image.\n/// Alpha channel is set to 255 (fully opaque).
pub fn cvt_color_gray_to_bgra(input: &Matrix<u8>) -> Result<Matrix<u8>, &'static str> {
    if input.channels != 1 {
        return Err("Input matrix must have exactly 1 channels");
    }

    let mut output = Matrix::<u8>::new(input.rows, input.cols, 4);
    let out_row_len = output.cols * 4;
    let in_row_len = input.cols;

    #[cfg(feature = "parallel")]
    {
        output.data
            .par_chunks_exact_mut(out_row_len)
            .zip(input.data.par_chunks_exact(in_row_len))
            .for_each(|(out_row, in_row)| {
                for (out_val, in_pixel) in out_row.chunks_exact_mut(4).zip(in_row.iter()) {
                    let v = *in_pixel;
                    out_val[0] = v;
                    out_val[1] = v;
                    out_val[2] = v;
                    out_val[3] = 255;
                }
            });
    }

    #[cfg(not(feature = "parallel"))]
    {
        output.data
            .chunks_exact_mut(out_row_len)
            .zip(input.data.chunks_exact(in_row_len))
            .for_each(|(out_row, in_row)| {
                for (out_val, in_pixel) in out_row.chunks_exact_mut(4).zip(in_row.iter()) {
                    let v = *in_pixel;
                    out_val[0] = v;
                    out_val[1] = v;
                    out_val[2] = v;
                    out_val[3] = 255;
                }
            });
    }

    Ok(output)
}

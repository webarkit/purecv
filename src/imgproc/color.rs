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
 *  WebARKitLib-rs is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU Lesser General Public License for more details.
 *
 *  You should have received a copy of the GNU Lesser General Public License
 *  along with WebARKitLib-rs.  If not, see <http://www.gnu.org/licenses/>.
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

use rayon::prelude::*;

// Assuming the Matrix<T> struct is in scope
use crate::core::Matrix;

/// Converts an 8-bit RGB image to an 8-bit Grayscale image.
/// Uses the standard luminosity formula: Y = 0.299*R + 0.587*G + 0.114*B
///
/// Returns a Result containing the new Grayscale Matrix, or an Error if
/// the input is not a 3-channel image.
pub fn cvt_color_rgb_to_gray(input: &Matrix<u8>) -> Result<Matrix<u8>, &'static str> {
    if input.channels != 3 {
        return Err("Input matrix must have exactly 3 channels (RGB)");
    }

    // Pre-allocate the output matrix (1 channel for grayscale)
    let mut output = Matrix::<u8>::new(input.rows, input.cols, 1);

    // Get mutable chunks of the output data, and immutable chunks of the input data.
    // We process the image row by row in parallel using Rayon.
    let out_row_len = output.cols;
    let in_row_len = input.cols * input.channels;

    output.data
        .par_chunks_exact_mut(out_row_len)
        .zip(input.data.par_chunks_exact(in_row_len))
        .for_each(|(out_row, in_row)| {
            // Process each pixel in the row
            for (x, out_pixel) in out_row.iter_mut().enumerate() {
                let r = in_row[x * 3] as f32;
                let g = in_row[x * 3 + 1] as f32;
                let b = in_row[x * 3 + 2] as f32;

                // Apply luminosity formula and safely cast back to u8
                *out_pixel = (0.299 * r + 0.587 * g + 0.114 * b).round() as u8;
            }
        });

    Ok(output)
}
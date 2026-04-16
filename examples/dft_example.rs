/*
 *  dft_example.rs
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

//! DFT (Discrete Fourier Transform) example.
//!
//! Run with: `cargo run --example dft_example --features fft`

use image::{DynamicImage, GenericImageView, ImageBuffer, Luma};
use purecv::core::dft::{dft, get_optimal_dft_size, DFT_COMPLEX_OUTPUT};
use purecv::core::structural::copy_make_border;
use purecv::core::{normalize, Matrix, NormTypes, Scalar};
use purecv::imgproc::cvt_color_rgb_to_gray;
use purecv::version;
use std::path::Path;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("--- purecv DFT Example ---");
    println!("purecv v{}", version::get_version());

    // 1. Load the image
    let img_path = "examples/data/butterfly.jpg";
    if !Path::new(img_path).exists() {
        eprintln!("Error: {} not found. Run from the project root.", img_path);
        return Ok(());
    }

    let img = image::open(img_path)?;
    let (width, height) = img.dimensions();
    println!("Loaded image: {} ({}x{})", img_path, width, height);

    let rgb_img = img.to_rgb8();
    let mat_rgb = Matrix::from_vec(height as usize, width as usize, 3, rgb_img.into_raw());

    // 2. Convert to grayscale
    let mat_gray = cvt_color_rgb_to_gray(&mat_rgb).expect("Grayscale conversion failed");
    println!(
        "Converted to grayscale: {}x{}",
        mat_gray.cols, mat_gray.rows
    );

    // 3. Convert u8 → f32
    let mat_f32 = Matrix::from_vec(
        mat_gray.rows,
        mat_gray.cols,
        1,
        mat_gray.data.iter().map(|&v| v as f32).collect(),
    );

    // 4. Pad to optimal DFT size
    let opt_rows = get_optimal_dft_size(mat_f32.rows as i32) as usize;
    let opt_cols = get_optimal_dft_size(mat_f32.cols as i32) as usize;
    let pad_bottom = opt_rows - mat_f32.rows;
    let pad_right = opt_cols - mat_f32.cols;
    println!(
        "Padding to optimal DFT size: {}x{} → {}x{}",
        mat_f32.cols, mat_f32.rows, opt_cols, opt_rows
    );

    let padded = copy_make_border(
        &mat_f32,
        0,
        pad_bottom,
        0,
        pad_right,
        0,
        Scalar::all(0.0f32),
    )?;

    // 5. Forward DFT
    println!("Computing DFT...");
    let t = Instant::now();
    let freq = dft(&padded, DFT_COMPLEX_OUTPUT, 0)?;
    println!("  DFT computed in {:.2?}", t.elapsed());
    println!(
        "  Output: {}x{} x {} channels (complex)",
        freq.cols, freq.rows, freq.channels
    );

    // 6. Compute magnitude: sqrt(re² + im²)
    let mut magnitude = Matrix::<f32>::new(freq.rows, freq.cols, 1);
    for r in 0..freq.rows {
        for c in 0..freq.cols {
            let idx = (r * freq.cols + c) * 2;
            let re = freq.data[idx];
            let im = freq.data[idx + 1];
            magnitude.data[r * freq.cols + c] = (re * re + im * im).sqrt();
        }
    }

    // 7. Log scale: log(1 + magnitude) for better visualization
    for v in &mut magnitude.data {
        *v = (1.0 + *v).ln();
    }

    // 8. Shift quadrants so DC is at center
    //    Swap top-left ↔ bottom-right and top-right ↔ bottom-left
    let cx = magnitude.cols / 2;
    let cy = magnitude.rows / 2;
    for r in 0..cy {
        for c in 0..cx {
            let tl = r * magnitude.cols + c;
            let br = (r + cy) * magnitude.cols + (c + cx);
            magnitude.data.swap(tl, br);

            let tr = r * magnitude.cols + (c + cx);
            let bl = (r + cy) * magnitude.cols + c;
            magnitude.data.swap(tr, bl);
        }
    }

    // 9. Normalize to 0–255 and convert to u8
    let mut mag_norm = Matrix::<f32>::new(magnitude.rows, magnitude.cols, 1);
    normalize(
        &magnitude,
        &mut mag_norm,
        0.0,
        255.0,
        NormTypes::MinMax,
        -1,
        None,
    )?;
    let mag_u8 = Matrix::from_vec(
        mag_norm.rows,
        mag_norm.cols,
        1,
        mag_norm.data.iter().map(|&v| v as u8).collect(),
    );

    // 10. Save result
    std::fs::create_dir_all("examples/data/out")?;
    save_matrix_gray(&mag_u8, "examples/data/out/dft_magnitude.png")?;
    println!("  Saved: examples/data/out/dft_magnitude.png");

    println!("\nDFT magnitude spectrum computed successfully!");
    Ok(())
}

fn save_matrix_gray(mat: &Matrix<u8>, filename: &str) -> image::ImageResult<()> {
    let img: ImageBuffer<Luma<u8>, Vec<u8>> =
        ImageBuffer::from_raw(mat.cols as u32, mat.rows as u32, mat.data.clone())
            .expect("Failed to create image buffer");
    DynamicImage::ImageLuma8(img).save(filename)
}

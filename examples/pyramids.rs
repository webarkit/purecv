/*
 *  pyramids.rs
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

//! Gaussian pyramid operations example.
//!
//! Demonstrates: pyr_down, pyr_up, build_pyramid.
//!
//! Run from the project root:
//! ```
//! cargo run --example pyramids
//! ```

use image::{DynamicImage, GenericImageView, ImageBuffer, Rgb};
use purecv::core::{BorderTypes, Matrix};
use purecv::imgproc::{build_pyramid, pyr_down, pyr_up};
use purecv::version;
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("--- purecv Pyramids Example ---");
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

    // 2. Convert to purecv Matrix<u8> (RGB)
    let rgb_img = img.to_rgb8();
    let mat_rgb = Matrix::from_vec(height as usize, width as usize, 3, rgb_img.into_raw());

    let border = BorderTypes::Reflect101;

    // --- Gaussian Pyramid (build_pyramid) ---
    println!("\n--- Building Gaussian Pyramid (4 levels) ---");
    let pyramid = build_pyramid(&mat_rgb, 3, border)?;

    for (i, level) in pyramid.iter().enumerate() {
        println!(
            "  Level {}: {}x{} ({} channels)",
            i, level.cols, level.rows, level.channels
        );
        let filename = format!("examples/data/out/output_pyramid_level_{}.png", i);
        save_matrix_rgb(level, &filename)?;
    }

    // --- pyr_down: single downscale ---
    println!("\n--- pyrDown (single downscale) ---");
    let down1 = pyr_down(&mat_rgb, None, border)?;
    println!(
        "  {}x{} → {}x{}",
        mat_rgb.cols, mat_rgb.rows, down1.cols, down1.rows
    );
    save_matrix_rgb(&down1, "examples/data/out/output_pyr_down.png")?;

    // --- pyr_up: standalone upscale ---
    println!("\n--- pyrUp (upscale the downscaled image) ---");
    let up1 = pyr_up(&mat_rgb, None, border)?;
    println!(
        "  {}x{} → {}x{}",
        mat_rgb.cols, mat_rgb.rows, up1.cols, up1.rows
    );
    save_matrix_rgb(&up1, "examples/data/out/output_pyr_up.png")?;

    // --- Round-trip: down then up (shows information loss) ---
    println!("\n--- Round-trip: pyrDown → pyrUp ---");
    let round_trip = pyr_up(&down1, None, border)?;
    println!(
        "  Original: {}x{} → Down: {}x{} → Up: {}x{}",
        mat_rgb.cols, mat_rgb.rows, down1.cols, down1.rows, round_trip.cols, round_trip.rows
    );
    save_matrix_rgb(&round_trip, "examples/data/out/output_pyr_roundtrip.png")?;

    println!("\nAll pyramid operations applied successfully!");
    println!("Check the examples/data/out/ directory for output images.");
    Ok(())
}

fn save_matrix_rgb(mat: &Matrix<u8>, filename: &str) -> image::ImageResult<()> {
    let img: ImageBuffer<Rgb<u8>, Vec<u8>> =
        ImageBuffer::from_raw(mat.cols as u32, mat.rows as u32, mat.data.clone())
            .expect("Failed to create image buffer");
    DynamicImage::ImageRgb8(img).save(filename)
}

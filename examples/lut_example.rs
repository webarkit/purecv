/*
 *  lut_example.rs
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

use image::{DynamicImage, GenericImageView, ImageBuffer, Rgb};
use purecv::core::lut;
use purecv::core::Matrix;
use purecv::version;
use std::path::Path;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("--- purecv LUT Example ---");
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

    std::fs::create_dir_all("examples/data/out")?;

    // 2. Gamma correction (γ = 1/2.2 — brightens the image)
    println!("Applying Gamma Correction (γ=1/2.2)...");
    let gamma_lut_data: Vec<u8> = (0..256)
        .map(|i| ((i as f64 / 255.0).powf(1.0 / 2.2) * 255.0).round() as u8)
        .collect();
    let gamma_table = Matrix::from_vec(1, 256, 1, gamma_lut_data);

    let t = Instant::now();
    let gamma_result = lut(&mat_rgb, &gamma_table)?;
    println!("  Done in {:.2?}", t.elapsed());
    save_matrix_rgb(&gamma_result, "examples/data/out/lut_gamma.png")?;
    println!("  Saved: examples/data/out/lut_gamma.png");

    // 3. Inversion (negative image)
    println!("Applying Inversion...");
    let invert_lut_data: Vec<u8> = (0..256).map(|i| (255 - i) as u8).collect();
    let invert_table = Matrix::from_vec(1, 256, 1, invert_lut_data);

    let t = Instant::now();
    let inverted = lut(&mat_rgb, &invert_table)?;
    println!("  Done in {:.2?}", t.elapsed());
    save_matrix_rgb(&inverted, "examples/data/out/lut_inverted.png")?;
    println!("  Saved: examples/data/out/lut_inverted.png");

    println!("\nAll LUT operations applied successfully!");
    Ok(())
}

fn save_matrix_rgb(mat: &Matrix<u8>, filename: &str) -> image::ImageResult<()> {
    let img: ImageBuffer<Rgb<u8>, Vec<u8>> =
        ImageBuffer::from_raw(mat.cols as u32, mat.rows as u32, mat.data.clone())
            .expect("Failed to create image buffer");
    DynamicImage::ImageRgb8(img).save(filename)
}

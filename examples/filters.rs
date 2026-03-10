/*
 *  filters.rs
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

use image::{DynamicImage, GenericImageView, ImageBuffer, Luma, Rgb};
use purecv::core::{normalize, BorderTypes, Matrix, NormTypes, Size};
use purecv::imgproc::{
    bilateral_filter, blur, canny, cvt_color_rgb_to_gray, gaussian_blur, laplacian, median_blur,
    scharr, sobel,
};
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("--- purecv Filters Example ---");

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

    // 3. Convert to Grayscale for some filters
    let mat_gray = cvt_color_rgb_to_gray(&mat_rgb)?;

    // --- Apply Filters ---

    // Blur (Box Filter)
    println!("Applying Blur...");
    let blurred = blur(
        &mat_rgb,
        Size::new(5, 5),
        purecv::core::Point::new(-1, -1),
        BorderTypes::REFLECT_101,
    )?;
    save_matrix_rgb(&blurred, "examples/data/out/output_blur.png")?;

    // Gaussian Blur
    println!("Applying Gaussian Blur...");
    let g_blurred = gaussian_blur(
        &mat_rgb,
        Size::new(5, 5),
        1.5,
        1.5,
        BorderTypes::REFLECT_101,
    )?;
    save_matrix_rgb(&g_blurred, "examples/data/out/output_gaussian_blur.png")?;

    // Median Blur
    println!("Applying Median Blur...");
    let m_blurred = median_blur(&mat_rgb, 5)?;
    save_matrix_rgb(&m_blurred, "examples/data/out/output_median_blur.png")?;

    // Bilateral Filter
    println!("Applying Bilateral Filter...");
    let b_filtered = bilateral_filter(&mat_rgb, 9, 75.0, 75.0, BorderTypes::REFLECT_101)?;
    save_matrix_rgb(&b_filtered, "examples/data/out/output_bilateral.png")?;

    // Sobel (on grayscale)
    println!("Applying Sobel...");
    let sob_x = sobel(&mat_gray, 1, 0, 3, 1.0, 0.0, BorderTypes::REFLECT_101)?;
    // Normalize for visualization
    let sob_x_norm = normalize(&sob_x, 0.0, 255.0, NormTypes::MINMAX)?;
    save_matrix_gray(&sob_x_norm, "examples/data/out/output_sobel_x.png")?;

    // Scharr (on grayscale)
    println!("Applying Scharr...");
    let sch_y = scharr(&mat_gray, 0, 1, 1.0, 0.0, BorderTypes::REFLECT_101)?;
    let sch_y_norm = normalize(&sch_y, 0.0, 255.0, NormTypes::MINMAX)?;
    save_matrix_gray(&sch_y_norm, "examples/data/out/output_scharr_y.png")?;

    // Laplacian (on grayscale)
    println!("Applying Laplacian...");
    let lap = laplacian(&mat_gray, 3, 1.0, 0.0, BorderTypes::REFLECT_101)?;
    let lap_norm = normalize(&lap, 0.0, 255.0, NormTypes::MINMAX)?;
    save_matrix_gray(&lap_norm, "examples/data/out/output_laplacian.png")?;

    // Canny (on grayscale)
    println!("Applying Canny...");
    let edges = canny(&mat_gray, 50.0, 150.0, 3, false)?;
    save_matrix_gray(&edges, "examples/data/out/output_canny.png")?;

    println!("\nAll filters applied successfully! Check the output_*.png files.");
    Ok(())
}

fn save_matrix_rgb(mat: &Matrix<u8>, filename: &str) -> image::ImageResult<()> {
    let img: ImageBuffer<Rgb<u8>, Vec<u8>> =
        ImageBuffer::from_raw(mat.cols as u32, mat.rows as u32, mat.data.clone())
            .expect("Failed to create image buffer");
    DynamicImage::ImageRgb8(img).save(filename)
}

fn save_matrix_gray(mat: &Matrix<u8>, filename: &str) -> image::ImageResult<()> {
    let img: ImageBuffer<Luma<u8>, Vec<u8>> =
        ImageBuffer::from_raw(mat.cols as u32, mat.rows as u32, mat.data.clone())
            .expect("Failed to create image buffer");
    DynamicImage::ImageLuma8(img).save(filename)
}

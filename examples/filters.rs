/*
 *  filters.rs
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

use image::{DynamicImage, GenericImageView, ImageBuffer, Luma, Rgb};
use purecv::core::logging::tags;
use purecv::core::{normalize, BorderTypes, Matrix, NormTypes, Size};
use purecv::imgproc::{
    bilateral_filter, blur, canny, cvt_color_rgb_to_gray, gaussian_blur, laplacian, median_blur,
    scharr, sobel,
};
use purecv::version;
use purecv::{cv_log_error, cv_log_info};
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    purecv::core::logging::init_basic_logger()?;

    cv_log_info!(tags::PURECV, "--- Filters Example ---");
    version::print_version();

    // 1. Load the image
    let img_path = "examples/data/butterfly.jpg";
    if !Path::new(img_path).exists() {
        cv_log_error!(
            tags::IMGPROC,
            "{} not found. Run from the project root.",
            img_path
        );
        return Ok(());
    }

    let img = image::open(img_path)?;
    let (width, height) = img.dimensions();
    cv_log_info!(
        tags::IMGPROC,
        "loaded image: {} ({}x{})",
        img_path,
        width,
        height
    );

    // 2. Convert to purecv Matrix<u8> (RGB)
    let rgb_img = img.to_rgb8();
    let mat_rgb = Matrix::from_vec(height as usize, width as usize, 3, rgb_img.into_raw());

    // 3. Convert to Grayscale for some filters
    let mat_gray = cvt_color_rgb_to_gray(&mat_rgb)?;

    // --- Apply Filters ---

    // Blur (Box Filter)
    cv_log_info!(tags::IMGPROC, "applying blur...");
    let blurred = blur(
        &mat_rgb,
        Size::new(5, 5),
        purecv::core::Point::new(-1, -1),
        BorderTypes::Reflect101,
    )?;
    save_matrix_rgb(&blurred, "examples/data/out/output_blur.png")?;

    // Gaussian Blur
    cv_log_info!(tags::IMGPROC, "applying gaussian blur...");
    let g_blurred = gaussian_blur(&mat_rgb, Size::new(5, 5), 1.5, 1.5, BorderTypes::Reflect101)?;
    save_matrix_rgb(&g_blurred, "examples/data/out/output_gaussian_blur.png")?;

    // Median Blur
    cv_log_info!(tags::IMGPROC, "applying median blur...");
    let m_blurred = median_blur(&mat_rgb, 5)?;
    save_matrix_rgb(&m_blurred, "examples/data/out/output_median_blur.png")?;

    // Bilateral Filter
    cv_log_info!(tags::IMGPROC, "applying bilateral filter...");
    let b_filtered = bilateral_filter(&mat_rgb, 9, 75.0, 75.0, BorderTypes::Reflect101)?;
    save_matrix_rgb(&b_filtered, "examples/data/out/output_bilateral.png")?;

    // Sobel (on grayscale)
    cv_log_info!(tags::IMGPROC, "applying sobel...");
    let sob_x = sobel(&mat_gray, 1, 0, 3, 1.0, 0.0, BorderTypes::Reflect101)?;
    // Normalize for visualization
    let mut sob_x_norm = Matrix::<u8>::new(sob_x.rows, sob_x.cols, sob_x.channels);
    normalize(
        &sob_x,
        &mut sob_x_norm,
        0.0,
        255.0,
        NormTypes::MinMax,
        -1,
        None,
    )?;
    save_matrix_gray(&sob_x_norm, "examples/data/out/output_sobel_x.png")?;

    // Scharr (on grayscale)
    cv_log_info!(tags::IMGPROC, "applying scharr...");
    let sch_y = scharr(&mat_gray, 0, 1, 1.0, 0.0, BorderTypes::Reflect101)?;
    let mut sch_y_norm = Matrix::<u8>::new(sch_y.rows, sch_y.cols, sch_y.channels);
    normalize(
        &sch_y,
        &mut sch_y_norm,
        0.0,
        255.0,
        NormTypes::MinMax,
        -1,
        None,
    )?;
    save_matrix_gray(&sch_y_norm, "examples/data/out/output_scharr_y.png")?;

    // Laplacian (on grayscale)
    cv_log_info!(tags::IMGPROC, "applying laplacian...");
    let lap = laplacian(&mat_gray, 3, 1.0, 0.0, BorderTypes::Reflect101)?;
    let mut lap_norm = Matrix::<u8>::new(lap.rows, lap.cols, lap.channels);
    normalize(&lap, &mut lap_norm, 0.0, 255.0, NormTypes::MinMax, -1, None)?;
    save_matrix_gray(&lap_norm, "examples/data/out/output_laplacian.png")?;

    // Canny (on grayscale)
    cv_log_info!(tags::IMGPROC, "applying canny...");
    let edges = canny(&mat_gray, 50.0, 150.0, 3, false)?;
    save_matrix_gray(&edges, "examples/data/out/output_canny.png")?;

    cv_log_info!(
        tags::IMGPROC,
        "all filters applied successfully! Check the output_*.png files."
    );
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

/*
 *  corner_detection.rs
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

//! Corner detection example — all detectors in one file.
//!
//! Demonstrates: Harris, Shi-Tomasi (min eigenvalue), goodFeaturesToTrack,
//! cornerSubPix, preCornerDetect, and cornerEigenValsAndVecs.
//!
//! Run from the project root:
//! ```
//! cargo run --example corner_detection
//! ```
//!
//! To use a custom image (e.g. graf1.png):
//! ```
//! cargo run --example corner_detection -- examples/data/graf1.png
//! ```

use image::{DynamicImage, GenericImageView, ImageBuffer, Luma};
use purecv::core::{normalize, BorderTypes, Matrix, NormTypes, Size2i, TermCriteria, TermType};
use purecv::imgproc::{
    corner_eigen_vals_and_vecs, corner_harris, corner_min_eigen_val, corner_sub_pix,
    cvt_color_rgb_to_gray, good_features_to_track, pre_corner_detect,
};
use purecv::version;
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("--- purecv Corner Detection Example ---");
    println!("purecv v{}", version::get_version());

    // Accept an optional image path from the command line.
    let default_path = "examples/data/butterfly.jpg";
    let img_path = std::env::args().nth(1).unwrap_or(default_path.to_string());

    if !Path::new(&img_path).exists() {
        eprintln!("Error: {} not found. Run from the project root.", img_path);
        return Ok(());
    }

    // Create output directory if it doesn't exist.
    std::fs::create_dir_all("examples/data/out")?;

    // 1. Load image and convert to grayscale.
    let img = image::open(&img_path)?;
    let (width, height) = img.dimensions();
    println!("Loaded image: {} ({}x{})", img_path, width, height);

    let rgb_img = img.to_rgb8();
    let mat_rgb = Matrix::from_vec(height as usize, width as usize, 3, rgb_img.into_raw());
    let mat_gray = cvt_color_rgb_to_gray(&mat_rgb)?;
    println!(
        "Grayscale matrix: {}x{}, {} channel(s)\n",
        mat_gray.rows, mat_gray.cols, mat_gray.channels
    );

    // -----------------------------------------------------------------------
    // 2. Harris corner detector
    // -----------------------------------------------------------------------
    println!("=== Harris Corner Detector ===");
    let harris_response = corner_harris(&mat_gray, 3, 3, 0.04, BorderTypes::Reflect101)?;

    // Normalize the response map to [0, 255] for visualization.
    let harris_vis = normalize_f32_to_u8(&harris_response)?;
    save_matrix_gray(&harris_vis, "examples/data/out/corner_harris.png")?;

    // Count strong corners (response > 50% of max).
    let harris_max = harris_response
        .data
        .iter()
        .cloned()
        .fold(f32::NEG_INFINITY, f32::max);
    let harris_threshold = harris_max * 0.01;
    let harris_count = harris_response
        .data
        .iter()
        .filter(|&&v| v > harris_threshold)
        .count();
    println!(
        "  Max response: {:.2}, threshold (1%): {:.2}, strong pixels: {}",
        harris_max, harris_threshold, harris_count
    );
    println!("  Saved: examples/data/out/corner_harris.png");

    // -----------------------------------------------------------------------
    // 3. Shi-Tomasi (minimum eigenvalue) corner detector
    // -----------------------------------------------------------------------
    println!("\n=== Shi-Tomasi (Min Eigenvalue) Corner Detector ===");
    let shi_tomasi_response = corner_min_eigen_val(&mat_gray, 3, 3, BorderTypes::Reflect101)?;

    let shi_vis = normalize_f32_to_u8(&shi_tomasi_response)?;
    save_matrix_gray(&shi_vis, "examples/data/out/corner_shi_tomasi.png")?;

    let shi_max = shi_tomasi_response
        .data
        .iter()
        .cloned()
        .fold(f32::NEG_INFINITY, f32::max);
    let shi_threshold = shi_max * 0.01;
    let shi_count = shi_tomasi_response
        .data
        .iter()
        .filter(|&&v| v > shi_threshold)
        .count();
    println!(
        "  Max response: {:.2}, threshold (1%): {:.2}, strong pixels: {}",
        shi_max, shi_threshold, shi_count
    );
    println!("  Saved: examples/data/out/corner_shi_tomasi.png");

    // -----------------------------------------------------------------------
    // 4. goodFeaturesToTrack — Shi-Tomasi mode
    // -----------------------------------------------------------------------
    println!("\n=== goodFeaturesToTrack (Shi-Tomasi) ===");
    let corners_st = good_features_to_track(&mat_gray, 200, 0.01, 10.0, 3, false, 0.04)?;
    println!("  Detected {} corners (Shi-Tomasi mode)", corners_st.len());
    for (i, pt) in corners_st.iter().take(5).enumerate() {
        println!("    corner[{i}]: ({:.2}, {:.2})", pt.x, pt.y);
    }
    if corners_st.len() > 5 {
        println!("    ... and {} more", corners_st.len() - 5);
    }

    // -----------------------------------------------------------------------
    // 5. goodFeaturesToTrack — Harris mode
    // -----------------------------------------------------------------------
    println!("\n=== goodFeaturesToTrack (Harris) ===");
    let corners_harris = good_features_to_track(&mat_gray, 200, 0.01, 10.0, 3, true, 0.04)?;
    println!("  Detected {} corners (Harris mode)", corners_harris.len());
    for (i, pt) in corners_harris.iter().take(5).enumerate() {
        println!("    corner[{i}]: ({:.2}, {:.2})", pt.x, pt.y);
    }
    if corners_harris.len() > 5 {
        println!("    ... and {} more", corners_harris.len() - 5);
    }

    // -----------------------------------------------------------------------
    // 6. cornerSubPix — sub-pixel refinement
    // -----------------------------------------------------------------------
    println!("\n=== cornerSubPix (sub-pixel refinement) ===");
    let mut refined = corners_st.clone();
    let num_to_refine = refined.len().min(20);
    println!("  Refining first {num_to_refine} corners...");

    // Show before/after for a few corners.
    corner_sub_pix(
        &mat_gray,
        &mut refined,
        Size2i::new(5, 5),
        Size2i::new(-1, -1),
        TermCriteria::new(TermType::Both, 30, 0.001),
    )?;

    for i in 0..num_to_refine.min(5) {
        let orig = &corners_st[i];
        let ref_ = &refined[i];
        println!(
            "    corner[{i}]: ({:.2}, {:.2}) → ({:.4}, {:.4})  Δ=({:.4}, {:.4})",
            orig.x,
            orig.y,
            ref_.x,
            ref_.y,
            ref_.x - orig.x,
            ref_.y - orig.y
        );
    }

    // -----------------------------------------------------------------------
    // 7. preCornerDetect
    // -----------------------------------------------------------------------
    println!("\n=== preCornerDetect ===");
    let pre_corner = pre_corner_detect(&mat_gray, 3, BorderTypes::Reflect101)?;

    let pre_vis = normalize_f32_to_u8(&pre_corner)?;
    save_matrix_gray(&pre_vis, "examples/data/out/corner_pre_detect.png")?;
    println!("  Saved: examples/data/out/corner_pre_detect.png");

    // -----------------------------------------------------------------------
    // 8. cornerEigenValsAndVecs
    // -----------------------------------------------------------------------
    println!("\n=== cornerEigenValsAndVecs ===");
    let eigen = corner_eigen_vals_and_vecs(&mat_gray, 3, 3, BorderTypes::Reflect101)?;
    println!(
        "  Output shape: {}x{}, {} channels (λ1, λ2, x1, y1, x2, y2)",
        eigen.rows, eigen.cols, eigen.channels
    );

    // Extract λ1 (channel 0) into a single-channel matrix for visualization.
    let npixels = eigen.rows * eigen.cols;
    let lambda1_data: Vec<f32> = (0..npixels).map(|i| eigen.data[i * 6]).collect();
    let lambda1 = Matrix::<f32>::from_vec(eigen.rows, eigen.cols, 1, lambda1_data);

    let lambda1_vis = normalize_f32_to_u8(&lambda1)?;
    save_matrix_gray(&lambda1_vis, "examples/data/out/corner_eigen_lambda1.png")?;
    println!("  Saved λ1 map: examples/data/out/corner_eigen_lambda1.png");

    // -----------------------------------------------------------------------
    // Summary
    // -----------------------------------------------------------------------
    println!("\n--- Summary ---");
    println!("  Harris strong pixels:     {harris_count}");
    println!("  Shi-Tomasi strong pixels:  {shi_count}");
    println!("  goodFeaturesToTrack (ST):  {} corners", corners_st.len());
    println!(
        "  goodFeaturesToTrack (H):   {} corners",
        corners_harris.len()
    );
    println!("  cornerSubPix:             {num_to_refine} corners refined");
    println!("\nOutput images saved to examples/data/out/");
    println!("Done.");
    Ok(())
}

/// Normalizes a `Matrix<f32>` to [0, 255] and converts to `Matrix<u8>` for saving.
fn normalize_f32_to_u8(src: &Matrix<f32>) -> Result<Matrix<u8>, Box<dyn std::error::Error>> {
    let mut dst = Matrix::<f32>::new(src.rows, src.cols, src.channels);
    normalize(src, &mut dst, 0.0, 255.0, NormTypes::MinMax, -1, None)?;
    let u8_data: Vec<u8> = dst
        .data
        .iter()
        .map(|&v| v.clamp(0.0, 255.0) as u8)
        .collect();
    Ok(Matrix::<u8>::from_vec(
        src.rows,
        src.cols,
        src.channels,
        u8_data,
    ))
}

/// Saves a single-channel `Matrix<u8>` as a grayscale PNG.
fn save_matrix_gray(mat: &Matrix<u8>, filename: &str) -> image::ImageResult<()> {
    let img: ImageBuffer<Luma<u8>, Vec<u8>> =
        ImageBuffer::from_raw(mat.cols as u32, mat.rows as u32, mat.data.clone())
            .expect("Failed to create image buffer");
    DynamicImage::ImageLuma8(img).save(filename)
}

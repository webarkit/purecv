/*
 *  histogram.rs
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

use image::{DynamicImage, GenericImageView, ImageBuffer, Luma};
use purecv::core::logging::tags;
use purecv::core::{Matrix, Size2i};
use purecv::imgproc::{
    calc_back_project, calc_hist, compare_hist, create_clahe, cvt_color_rgb_to_gray, equalize_hist,
    HistCompMethods, RangeSpec,
};
use purecv::version;
use purecv::{cv_log_error, cv_log_info};
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    purecv::core::logging::init_basic_logger()?;

    cv_log_info!(tags::PURECV, "--- Histogram & CLAHE Example ---");
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

    let rgb_img = img.to_rgb8();
    let mat_rgb = Matrix::from_vec(height as usize, width as usize, 3, rgb_img.into_raw());
    let mat_gray = cvt_color_rgb_to_gray(&mat_rgb)?;

    // Create output directory if it doesn't exist.
    std::fs::create_dir_all("examples/data/out")?;

    // --- calc_hist: uniform bins ---

    cv_log_info!(tags::IMGPROC, "computing a 256-bin uniform histogram...");
    let hist_size = [256usize];
    let ranges = [RangeSpec::Uniform(0.0, 256.0)];
    let hist = calc_hist(&[&mat_gray], &[0], None, &hist_size, &ranges, false, None)?;
    print_histogram_summary(&hist);

    // --- calc_hist: non-uniform bins ---
    //
    // Four unevenly-spaced bins: a wide shadow bucket, two mid-tone buckets,
    // and a narrow highlight bucket.
    cv_log_info!(tags::IMGPROC, "computing a 4-bin non-uniform histogram...");
    let non_uniform_size = [4usize];
    let boundaries = vec![0.0, 96.0, 160.0, 224.0, 256.0];
    let non_uniform_ranges = [RangeSpec::NonUniform(boundaries)];
    let non_uniform_hist = calc_hist(
        &[&mat_gray],
        &[0],
        None,
        &non_uniform_size,
        &non_uniform_ranges,
        false,
        None,
    )?;
    println!(
        "  non-uniform bins [0,96) [96,160) [160,224) [224,256): {:?}",
        non_uniform_hist.data
    );

    // --- calc_back_project ---
    //
    // Projects the histogram back onto the source image: each pixel is
    // replaced by its own bin's count, scaled for visibility. Bright
    // regions in the output correspond to common intensities in the image.
    // The scale is derived from the histogram's own peak so the output uses
    // the full 0-255 display range, matching OpenCV's typical demo pattern.
    cv_log_info!(tags::IMGPROC, "back-projecting the histogram...");
    let max_bin = hist.data.iter().cloned().fold(0.0f32, f32::max);
    let bp_scale = if max_bin > 0.0 { 255.0 / max_bin } else { 1.0 };
    let back_projected =
        calc_back_project(&[&mat_gray], &[0], &hist_size, &hist, &ranges, bp_scale)?;
    let mut bp_u8 = Matrix::<u8>::new(back_projected.rows, back_projected.cols, 1);
    for (dst, &src) in bp_u8.data.iter_mut().zip(back_projected.data.iter()) {
        *dst = src.clamp(0.0, 255.0) as u8;
    }
    save_matrix_gray(&bp_u8, "examples/data/out/output_back_project.png")?;

    // --- compare_hist ---
    //
    // Compare the source histogram against the histogram of a brightened
    // copy of the same image, across every HistCompMethods variant.
    cv_log_info!(tags::IMGPROC, "comparing histograms...");
    let mut brightened = Matrix::<u8>::new(mat_gray.rows, mat_gray.cols, 1);
    for (dst, &src) in brightened.data.iter_mut().zip(mat_gray.data.iter()) {
        *dst = src.saturating_add(40);
    }
    let brightened_hist = calc_hist(&[&brightened], &[0], None, &hist_size, &ranges, false, None)?;

    for method in [
        HistCompMethods::Correl,
        HistCompMethods::ChiSqr,
        HistCompMethods::ChiSqrAlt,
        HistCompMethods::Intersection,
        HistCompMethods::Bhattacharyya,
        HistCompMethods::KullbackLeibler,
    ] {
        let score = compare_hist(&hist, &brightened_hist, method)?;
        println!("  {:?}: {:.4}", method, score);
    }

    // --- equalize_hist ---

    cv_log_info!(tags::IMGPROC, "equalizing histogram...");
    let equalized = equalize_hist(&mat_gray)?;
    save_matrix_gray(&equalized, "examples/data/out/output_equalize_hist.png")?;

    // --- CLAHE ---
    //
    // Contrast Limited Adaptive Histogram Equalization avoids over-amplifying
    // noise in flat regions, unlike global equalize_hist above. Two tile
    // grid sizes are compared: a coarser 4x4 grid and a finer 8x8 grid.
    cv_log_info!(tags::IMGPROC, "applying CLAHE (4x4 tiles)...");
    let clahe_coarse = create_clahe(40.0, Size2i::new(4, 4));
    let clahe_coarse_out = clahe_coarse.apply_u8(&mat_gray)?;
    save_matrix_gray(&clahe_coarse_out, "examples/data/out/output_clahe_4x4.png")?;

    cv_log_info!(tags::IMGPROC, "applying CLAHE (8x8 tiles)...");
    let clahe_fine = create_clahe(40.0, Size2i::new(8, 8));
    let clahe_fine_out = clahe_fine.apply_u8(&mat_gray)?;
    save_matrix_gray(&clahe_fine_out, "examples/data/out/output_clahe_8x8.png")?;

    cv_log_info!(
        tags::IMGPROC,
        "done! Check the output_*.png files under examples/data/out/."
    );
    Ok(())
}

/// Prints the min/max/mean bin count and the 3 most populated bins of a
/// flattened `f32` histogram (as returned by `calc_hist`).
fn print_histogram_summary(hist: &Matrix<f32>) {
    let data = &hist.data;
    let total_bins = data.len();
    let sum: f32 = data.iter().sum();
    let mean = sum / total_bins as f32;
    let max_bin = data
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(idx, &v)| (idx, v))
        .unwrap_or((0, 0.0));
    println!(
        "  {total_bins} bins, total count {sum:.0}, mean {mean:.2}, most populated bin {} (count {:.0})",
        max_bin.0, max_bin.1
    );
}

fn save_matrix_gray(mat: &Matrix<u8>, filename: &str) -> image::ImageResult<()> {
    let img: ImageBuffer<Luma<u8>, Vec<u8>> =
        ImageBuffer::from_raw(mat.cols as u32, mat.rows as u32, mat.data.clone())
            .expect("Failed to create image buffer");
    DynamicImage::ImageLuma8(img).save(filename)
}

/*
 *  match_features.rs
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

//! Standalone ORB feature detection, extraction, matching, and visualization example.
//!
//! Loads `graf.png` (which contains two stitched images), splits it down the middle,
//! extracts keypoints and descriptors using ORB, matches them using BFMatcher (Hamming),
//! filters them with Lowe's ratio test, draws the matches, and saves the output.
//!
//! Run from the project root:
//! ```bash
//! cargo run --example match_features
//! ```

use image::{DynamicImage, GenericImageView, ImageBuffer, Rgb};
use purecv::core::Matrix;
use purecv::features2d::{draw_matches, BFMatcher, DescriptorMatcher, NormType, Orb, ScoreType};
use purecv::imgproc::cvt_color_rgb_to_gray;
use purecv::version;
use std::path::Path;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("--- purecv Feature Matching Example ---");
    println!("purecv version: v{}", version::get_version());

    let img_path = "examples/data/graf.png";
    if !Path::new(img_path).exists() {
        eprintln!(
            "Error: {} not found. Make sure you are in the project root.",
            img_path
        );
        return Ok(());
    }

    // Ensure output directory exists.
    std::fs::create_dir_all("examples/data/out")?;

    // 1. Load the stitched source image.
    let load_start = Instant::now();
    let img = image::open(img_path)?;
    let (width, height) = img.dimensions();
    println!(
        "Loaded stitched image: {} ({}x{}) in {:.2?}",
        img_path,
        width,
        height,
        load_start.elapsed()
    );

    let half_width = width / 2;
    println!(
        "Splitting image into left half ({}x{}) and right half ({}x{})",
        half_width, height, half_width, height
    );

    // 2. Convert to RGB Matrix
    let rgb_img = img.to_rgb8();
    let mat_rgb = Matrix::from_vec(height as usize, width as usize, 3, rgb_img.into_raw());

    // 3. Convert to Grayscale
    let mat_gray = cvt_color_rgb_to_gray(&mat_rgb)?;

    // 4. Crop into Left and Right halves (both RGB and Grayscale)
    let crop_start = Instant::now();
    let mut left_gray = Matrix::<u8>::new(height as usize, half_width as usize, 1);
    let mut right_gray = Matrix::<u8>::new(height as usize, half_width as usize, 1);
    let mut left_rgb = Matrix::<u8>::new(height as usize, half_width as usize, 3);
    let mut right_rgb = Matrix::<u8>::new(height as usize, half_width as usize, 3);

    for y in 0..height as usize {
        for x in 0..half_width as usize {
            // Grayscale crops
            let l_gray_val = *mat_gray.get(y, x, 0).unwrap_or(&0);
            left_gray.set(y, x, 0, l_gray_val);

            let r_gray_val = *mat_gray.get(y, x + half_width as usize, 0).unwrap_or(&0);
            right_gray.set(y, x, 0, r_gray_val);

            // Grayscale crops replicated to 3 channels for drawing color lines
            for c in 0..3 {
                left_rgb.set(y, x, c, l_gray_val);
                right_rgb.set(y, x, c, r_gray_val);
            }
        }
    }
    println!("Cropped images in {:.2?}", crop_start.elapsed());

    // 5. Detect and Compute ORB Features on both halves (extract 1000 features for higher density)
    let orb = Orb::new(1000, 1.2, 8, 31, 0, 2, ScoreType::Harris, 31, 20);
    println!("Extracting ORB features on left half...");
    let start_left = Instant::now();
    let (kps1, desc1) = orb.detect_and_compute(&left_gray)?;
    println!(
        "  Left: extracted {} keypoints in {:.2?}",
        kps1.len(),
        start_left.elapsed()
    );

    println!("Extracting ORB features on right half...");
    let start_right = Instant::now();
    let (kps2, desc2) = orb.detect_and_compute(&right_gray)?;
    println!(
        "  Right: extracted {} keypoints in {:.2?}",
        kps2.len(),
        start_right.elapsed()
    );

    // 6. Match features using BFMatcher (Hamming distance with cross-check enabled)
    println!("Matching features using BFMatcher (NormHamming + Cross Check)...");
    let match_start = Instant::now();
    let matcher = BFMatcher::new(NormType::NormHamming, true)?;

    let mutual_matches = matcher.match_descriptors(&desc1, &desc2)?;
    println!(
        "  Matched in {:.2?}. Total mutual matches: {}",
        match_start.elapsed(),
        mutual_matches.len()
    );

    // 7. Draw matches with random line colors and no unmatched keypoint clutter
    println!("Drawing matches...");
    let draw_start = Instant::now();

    let matched_img = draw_matches(
        &left_rgb,
        &kps1,
        &right_rgb,
        &kps2,
        &mutual_matches,
        None, // Random colors for matches
        None, // Do not draw unmatched keypoints
    )?;
    println!(
        "  Drawn matching visualization in {:.2?}",
        draw_start.elapsed()
    );

    // 8. Save output visualization
    let out_path = "examples/data/out/graf_matches.png";
    let save_start = Instant::now();
    let img_out: ImageBuffer<Rgb<u8>, Vec<u8>> = ImageBuffer::from_raw(
        matched_img.cols as u32,
        matched_img.rows as u32,
        matched_img.data.clone(),
    )
    .ok_or("Failed to construct image buffer from matched matrix data")?;
    DynamicImage::ImageRgb8(img_out).save(out_path)?;
    println!(
        "  Saved match visualization to: {} in {:.2?}",
        out_path,
        save_start.elapsed()
    );

    println!("\nDone! Run 'cargo run --example match_features' to run again.");
    Ok(())
}

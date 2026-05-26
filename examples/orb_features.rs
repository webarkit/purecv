/*
 *  orb_features.rs
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

//! Standalone ORB feature detection & extraction example.
//!
//! Loads an image, builds a scale pyramid, extracts ORB keypoints,
//! computes their orientations and BRIEF descriptors, draws rose-colored
//! scale circles and sky-blue orientation needles, and saves the output.
//!
//! Run from the project root:
//! ```bash
//! cargo run --example orb_features
//! ```

use image::{DynamicImage, GenericImageView, ImageBuffer, Rgb};
use purecv::core::Matrix;
use purecv::features2d::Orb;
use purecv::imgproc::cvt_color_rgb_to_gray;
use purecv::version;
use std::path::Path;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("--- purecv ORB Features Example ---");
    println!("purecv version: v{}", version::get_version());

    // 1. Accept an optional custom image path from command-line arguments.
    let default_path = "examples/data/butterfly.jpg";
    let img_path = std::env::args().nth(1).unwrap_or(default_path.to_string());

    if !Path::new(&img_path).exists() {
        eprintln!("Error: {} not found. Run from the project root.", img_path);
        return Ok(());
    }

    // Ensure output directory exists.
    std::fs::create_dir_all("examples/data/out")?;

    // 2. Load the source image.
    let load_start = Instant::now();
    let img = image::open(&img_path)?;
    let (width, height) = img.dimensions();
    println!(
        "Loaded image: {} ({}x{}) in {:.2?}\n",
        img_path,
        width,
        height,
        load_start.elapsed()
    );

    // 3. Convert to RGB Matrix
    let rgb_img = img.to_rgb8();
    let mut mat_rgb = Matrix::from_vec(height as usize, width as usize, 3, rgb_img.into_raw());

    // 4. Convert to Grayscale
    let mat_gray = cvt_color_rgb_to_gray(&mat_rgb)?;

    // 5. Initialize default ORB Detector & Extractor
    let orb = Orb::default();

    println!("Running ORB keypoint detection & descriptor extraction pipeline...");
    let extract_start = Instant::now();
    let (keypoints, descriptors) = orb.detect_and_compute(&mat_gray)?;
    let extract_duration = extract_start.elapsed();

    println!(
        "  Detected {} keypoints in {:.2?}",
        keypoints.len(),
        extract_duration
    );
    println!(
        "  Computed descriptors Matrix shape: {}x{}, channels = {}",
        descriptors.rows, descriptors.cols, descriptors.channels
    );

    // 6. Draw Oriented Keypoints
    let draw_start = Instant::now();
    let rose = [255, 0, 85]; // Rose color for scale circles
    let sky_blue = [56, 189, 248]; // Sky blue color for orientation needles

    for kp in &keypoints {
        let cx = kp.pt.x.round() as i32;
        let cy = kp.pt.y.round() as i32;
        let radius = (kp.size / 2.0).round().max(4.0) as i32;

        // 1. Draw Rose scale circle
        draw_circle(&mut mat_rgb, cx, cy, radius, rose);

        // 2. Draw Sky-Blue orientation vector line
        if kp.angle >= 0.0 {
            let angle_rad = kp.angle.to_radians();
            let x1 = kp.pt.x + (radius as f32) * angle_rad.cos();
            let y1 = kp.pt.y + (radius as f32) * angle_rad.sin();

            draw_line(
                &mut mat_rgb,
                cx,
                cy,
                x1.round() as i32,
                y1.round() as i32,
                sky_blue,
            );
        }

        // Draw center dot
        if cx >= 0 && cx < mat_rgb.cols as i32 && cy >= 0 && cy < mat_rgb.rows as i32 {
            let idx = (cy as usize * mat_rgb.cols + cx as usize) * 3;
            mat_rgb.data[idx] = sky_blue[0];
            mat_rgb.data[idx + 1] = sky_blue[1];
            mat_rgb.data[idx + 2] = sky_blue[2];
        }
    }
    println!(
        "  Rendered oriented features in {:.2?}",
        draw_start.elapsed()
    );

    // 7. Save output image
    let out_path = "examples/data/out/orb_features.png";
    let save_start = Instant::now();
    save_matrix_rgb(&mat_rgb, out_path)?;
    println!(
        "  Saved marked output to: {} in {:.2?}\n",
        out_path,
        save_start.elapsed()
    );

    println!("Done. Run 'cargo run --example orb_features' to inspect.");
    Ok(())
}

/// Draws a bounds-checked hollow circle on an RGB matrix using the midpoint circle algorithm.
fn draw_circle(mat: &mut Matrix<u8>, cx: i32, cy: i32, r: i32, color: [u8; 3]) {
    let mut x = r;
    let mut y = 0;
    let mut err = 0;

    while x >= y {
        let points = [
            (cx + x, cy + y),
            (cx + y, cy + x),
            (cx - y, cy + x),
            (cx - x, cy + y),
            (cx - x, cy - y),
            (cx - y, cy - x),
            (cx + y, cy - x),
            (cx + x, cy - y),
        ];

        for &(px, py) in &points {
            if px >= 0 && px < mat.cols as i32 && py >= 0 && py < mat.rows as i32 {
                let idx = (py as usize * mat.cols + px as usize) * 3;
                mat.data[idx] = color[0];
                mat.data[idx + 1] = color[1];
                mat.data[idx + 2] = color[2];
            }
        }

        y += 1;
        if err <= 0 {
            err += 2 * y + 1;
        } else {
            x -= 1;
            err += 2 * (y - x) + 1;
        }
    }
}

/// Draws a bounds-checked line on an RGB matrix using Bresenham's line algorithm.
fn draw_line(mat: &mut Matrix<u8>, x0: i32, y0: i32, x1: i32, y1: i32, color: [u8; 3]) {
    let dx = (x1 - x0).abs();
    let dy = (y1 - y0).abs();
    let sx = if x0 < x1 { 1 } else { -1 };
    let sy = if y0 < y1 { 1 } else { -1 };
    let mut err = dx - dy;
    let mut x = x0;
    let mut y = y0;

    loop {
        if x >= 0 && x < mat.cols as i32 && y >= 0 && y < mat.rows as i32 {
            let idx = (y as usize * mat.cols + x as usize) * 3;
            mat.data[idx] = color[0];
            mat.data[idx + 1] = color[1];
            mat.data[idx + 2] = color[2];
        }

        if x == x1 && y == y1 {
            break;
        }

        let e2 = 2 * err;
        if e2 > -dy {
            err -= dy;
            x += sx;
        }
        if e2 < dx {
            err += dx;
            y += sy;
        }
    }
}

/// Saves an RGB `Matrix<u8>` as an image file.
fn save_matrix_rgb(mat: &Matrix<u8>, filename: &str) -> image::ImageResult<()> {
    let img: ImageBuffer<Rgb<u8>, Vec<u8>> =
        ImageBuffer::from_raw(mat.cols as u32, mat.rows as u32, mat.data.clone())
            .expect("Failed to construct image buffer from matrix data");
    DynamicImage::ImageRgb8(img).save(filename)
}

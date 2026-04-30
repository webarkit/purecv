/*
 *  optical_flow.rs
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

//! Sparse optical flow example — pyramidal Lucas-Kanade on static images.
//!
//! This example mirrors the OpenCV tutorial at
//! <https://docs.opencv.org/4.x/d4/dee/tutorial_optical_flow.html> without
//! requiring a camera or video file.  Because optical flow needs *two* frames,
//! the "next" frame is produced by translating a real loaded image by a known
//! (SHIFT_X, SHIFT_Y) offset — the same content you would see in two
//! consecutive video frames of a camera panning slightly to the right/down.
//!
//! **Pipeline**
//! ```text
//! real image  ──────────────────► prev_gray
//!                  │
//!                  └─ translate (+4, +3) px ──► next_gray
//!
//! prev_gray ──► goodFeaturesToTrack ──► corners (Vec<Point2f>)
//!
//! (prev_gray, next_gray, corners) ──► calcOpticalFlowPyrLK
//!                                          │
//!                         ┌────────────────┼────────────────┐
//!                      next_pts         status             err
//! ```
//!
//! Run from the project root:
//! ```
//! cargo run --example optical_flow
//! ```
//!
//! To use a custom image:
//! ```
//! cargo run --example optical_flow -- examples/data/butterfly.jpg
//! ```

use image::{DynamicImage, GenericImageView, ImageBuffer, Rgb};
use purecv::core::types::{Point2f, Size2i, TermCriteria, TermType};
use purecv::core::Matrix;
use purecv::imgproc::{cvt_color_rgb_to_gray, good_features_to_track};
use purecv::version;
use purecv::video::{calc_optical_flow_pyramid_lk, OPTFLOW_LK_GET_MIN_EIGENVALS};
use std::io::Write;
use std::path::Path;

/// Simulated camera/object motion between the two synthetic frames (pixels).
const SHIFT_X: usize = 4;
const SHIFT_Y: usize = 3;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("--- purecv Optical Flow Example (static image) ---");
    println!("purecv v{}", version::get_version());
    println!("Simulating motion: +{SHIFT_X} px in x, +{SHIFT_Y} px in y\n");

    // Accept an optional image path from the command line.
    let default_path = "examples/data/butterfly.jpg";
    let img_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| default_path.to_string());

    if !Path::new(&img_path).exists() {
        eprintln!(
            "Error: '{}' not found. Run from the project root.",
            img_path
        );
        return Ok(());
    }

    std::fs::create_dir_all("examples/data/out")?;

    // -----------------------------------------------------------------------
    // 1. Load image and convert to grayscale.
    // -----------------------------------------------------------------------
    let img = image::open(&img_path)?;
    let (width, height) = img.dimensions();
    println!("Loaded: {} ({}×{})", img_path, width, height);

    let rgb_img = img.to_rgb8();
    let mat_rgb = Matrix::from_vec(
        height as usize,
        width as usize,
        3,
        rgb_img.clone().into_raw(),
    );
    let prev_gray = cvt_color_rgb_to_gray(&mat_rgb)?;
    println!(
        "Grayscale: {}×{}, {} channel(s)\n",
        prev_gray.rows, prev_gray.cols, prev_gray.channels
    );

    // -----------------------------------------------------------------------
    // 2. Synthesise "next" frame by translating the grayscale image.
    //
    //    next[r, c] = prev[r - SHIFT_Y, c - SHIFT_X]   (border clamped)
    //
    //    This means image content moves by (+SHIFT_X, +SHIFT_Y).  A feature
    //    at (px, py) in prev will appear at (px + SHIFT_X, py + SHIFT_Y) in
    //    next, so the expected optical flow per point is (+SHIFT_X, +SHIFT_Y).
    // -----------------------------------------------------------------------
    let rows = prev_gray.rows;
    let cols = prev_gray.cols;
    let mut next_data = vec![0u8; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            let src_r = r.saturating_sub(SHIFT_Y);
            let src_c = c.saturating_sub(SHIFT_X);
            next_data[r * cols + c] = prev_gray.data[src_r * cols + src_c];
        }
    }
    let next_gray = Matrix::<u8>::from_vec(rows, cols, 1, next_data);
    println!("Synthesised next frame: translated by (+{SHIFT_X}, +{SHIFT_Y}) pixels\n");

    // -----------------------------------------------------------------------
    // 3. Detect good features to track in the previous (original) frame.
    // -----------------------------------------------------------------------
    println!("=== Step 1: Detect features (goodFeaturesToTrack) ===");
    let corners = good_features_to_track(&prev_gray, 100, 0.01, 10.0, 3, false, 0.04)?;
    println!("Detected {} corner(s)", corners.len());
    for (i, pt) in corners.iter().take(5).enumerate() {
        println!("  corner[{i}]: ({:.1}, {:.1})", pt.x, pt.y);
    }
    if corners.len() > 5 {
        println!("  … and {} more", corners.len() - 5);
    }

    if corners.is_empty() {
        eprintln!("\nNo features detected — try a different image.");
        return Ok(());
    }

    // -----------------------------------------------------------------------
    // 4. Track features from prev to next using pyramidal LK.
    // -----------------------------------------------------------------------
    println!("\n=== Step 2: Track features (calcOpticalFlowPyrLK) ===");
    let criteria = TermCriteria::new(TermType::Both, 30, 0.001);
    let (next_pts, status, err) = calc_optical_flow_pyramid_lk(
        &prev_gray,
        &next_gray,
        &corners,
        None,
        Size2i::new(15, 15),
        3,
        criteria,
        OPTFLOW_LK_GET_MIN_EIGENVALS,
        1e-4,
    )?;

    // -----------------------------------------------------------------------
    // 5. Print per-point results (first 20 points).
    // -----------------------------------------------------------------------
    let n_tracked = status.iter().filter(|&&s| s == 1).count();
    let n_lost = corners.len() - n_tracked;
    println!(
        "Tracked: {}/{} points  ({} lost)\n",
        n_tracked,
        corners.len(),
        n_lost
    );

    println!(
        "{:>5}  {:>10}  {:>10}  {:>10}  {:>10}  {:>10}  {:>10}  {:>6}  {:>10}",
        "idx", "prev_x", "prev_y", "next_x", "next_y", "flow_x", "flow_y", "status", "min_eigen",
    );
    println!("{}", "-".repeat(85));

    let display_n = corners.len().min(20);
    for i in 0..display_n {
        let flow_x = next_pts[i].x - corners[i].x;
        let flow_y = next_pts[i].y - corners[i].y;
        // Format signed flow values separately (dynamic-width + sign flag unsupported).
        let fx_str = format!("{:+.2}", flow_x);
        let fy_str = format!("{:+.2}", flow_y);
        println!(
            "{:>5}  {:>10.2}  {:>10.2}  {:>10.2}  {:>10.2}  {:>10}  {:>10}  {:>6}  {:>10.4}",
            i,
            corners[i].x,
            corners[i].y,
            next_pts[i].x,
            next_pts[i].y,
            fx_str,
            fy_str,
            if status[i] == 1 { "OK" } else { "LOST" },
            err[i],
        );
    }
    if corners.len() > display_n {
        println!("  … ({} more points not shown)", corners.len() - display_n);
    }

    // -----------------------------------------------------------------------
    // 6. Compare average tracked flow to ground-truth shift.
    // -----------------------------------------------------------------------
    let tracked_indices: Vec<usize> = status
        .iter()
        .enumerate()
        .filter(|(_, &s)| s == 1)
        .map(|(i, _)| i)
        .collect();

    if !tracked_indices.is_empty() {
        let (sum_dx, sum_dy) = tracked_indices
            .iter()
            .fold((0.0f32, 0.0f32), |(sdx, sdy), &i| {
                (
                    sdx + (next_pts[i].x - corners[i].x),
                    sdy + (next_pts[i].y - corners[i].y),
                )
            });
        let n = tracked_indices.len() as f32;
        let avg_dx = sum_dx / n;
        let avg_dy = sum_dy / n;

        println!("\n=== Flow Statistics ===");
        println!(
            "  Ground-truth shift:   (+{:.1}, +{:.1}) px",
            SHIFT_X as f32, SHIFT_Y as f32
        );
        println!(
            "  Mean tracked flow:    ({:+.2}, {:+.2}) px",
            avg_dx, avg_dy
        );
        println!(
            "  Absolute error:       ({:.2}, {:.2}) px",
            (avg_dx - SHIFT_X as f32).abs(),
            (avg_dy - SHIFT_Y as f32).abs()
        );
    }

    // -----------------------------------------------------------------------
    // 7. Save annotated output image.
    // -----------------------------------------------------------------------
    let result_path = "examples/data/out/optical_flow_result.png";
    save_flow_image(&rgb_img, &corners, &next_pts, &status, result_path)?;

    // -----------------------------------------------------------------------
    // 8. Save CSV of flow vectors.
    // -----------------------------------------------------------------------
    let csv_path = "examples/data/out/optical_flow_vectors.csv";
    save_flow_csv(&corners, &next_pts, &status, &err, csv_path)?;

    println!("\nOutput saved to:");
    println!("  {result_path}   (annotated image: green = tracked, red = lost)");
    println!("  {csv_path}");
    println!("\nDone.");
    Ok(())
}

// ---------------------------------------------------------------------------
// Helper: save annotated RGB image
// ---------------------------------------------------------------------------

/// Saves a copy of `base` with flow arrows overlaid.
///
/// For each feature point:
/// * A small cross (±4 px) is drawn at the previous position.
/// * A Bresenham line is drawn from the previous to the next position for
///   tracked points.
///
/// Colour coding:
/// * Green (`[0, 210, 0]`) — point was tracked successfully (`status == 1`).
/// * Red   (`[210, 0, 0]`) — point was lost (`status == 0`).
///
/// # Arguments
/// * `base`      — Original RGB image used as background.
/// * `prev_pts`  — Feature positions in the previous frame.
/// * `next_pts`  — Estimated positions in the next frame (from LK).
/// * `status`    — Per-point tracking flag: `1` = tracked, `0` = lost.
/// * `path`      — Destination file path (PNG extension expected).
///
/// # Errors
/// Returns an error if the output file cannot be created or if the `image`
/// crate fails to encode the PNG.
fn save_flow_image(
    base: &image::RgbImage,
    prev_pts: &[Point2f],
    next_pts: &[Point2f],
    status: &[u8],
    path: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut out: ImageBuffer<Rgb<u8>, Vec<u8>> = base.clone();
    let w = out.width() as i32;
    let h = out.height() as i32;

    for (i, (prev, next)) in prev_pts.iter().zip(next_pts.iter()).enumerate() {
        let color = if status[i] == 1 {
            Rgb([0u8, 210, 0]) // green — tracked
        } else {
            Rgb([210u8, 0, 0]) // red   — lost
        };

        // Draw a small cross at the previous position.
        for d in -4i32..=4 {
            let px = (prev.x as i32 + d).clamp(0, w - 1) as u32;
            let py = prev.y as u32;
            if py < h as u32 {
                out.put_pixel(px, py, color);
            }
            let px = prev.x as u32;
            let py = (prev.y as i32 + d).clamp(0, h - 1) as u32;
            if px < w as u32 {
                out.put_pixel(px, py, color);
            }
        }

        // Draw a line from prev → next for successfully tracked points.
        if status[i] == 1 {
            draw_line(
                &mut out,
                prev.x as i32,
                prev.y as i32,
                next.x as i32,
                next.y as i32,
                color,
            );
        }
    }

    DynamicImage::ImageRgb8(out).save(path)?;
    Ok(())
}

/// Minimal Bresenham line drawing on an `RgbImage`.
///
/// Draws a straight line between `(x0, y0)` and `(x1, y1)` using the
/// standard Bresenham error-accumulation algorithm.  Pixels whose coordinates
/// fall outside the image bounds `[0, width) × [0, height)` are silently
/// skipped, so no bounds checking is required by the caller.
///
/// # Arguments
/// * `img`   — Destination image (modified in place).
/// * `x0`, `y0` — Start point (column, row).
/// * `x1`, `y1` — End point (column, row).
/// * `color` — RGB colour to paint every pixel on the line.
fn draw_line(
    img: &mut image::RgbImage,
    mut x0: i32,
    mut y0: i32,
    x1: i32,
    y1: i32,
    color: Rgb<u8>,
) {
    let (w, h) = (img.width() as i32, img.height() as i32);
    let dx = (x1 - x0).abs();
    let dy = -(y1 - y0).abs();
    let sx = if x0 < x1 { 1 } else { -1 };
    let sy = if y0 < y1 { 1 } else { -1 };
    let mut err = dx + dy;
    loop {
        if x0 >= 0 && x0 < w && y0 >= 0 && y0 < h {
            img.put_pixel(x0 as u32, y0 as u32, color);
        }
        if x0 == x1 && y0 == y1 {
            break;
        }
        let e2 = 2 * err;
        if e2 >= dy {
            err += dy;
            x0 += sx;
        }
        if e2 <= dx {
            err += dx;
            y0 += sy;
        }
    }
}

// ---------------------------------------------------------------------------
// Helper: save CSV of flow vectors
// ---------------------------------------------------------------------------

/// Writes flow vectors for every feature point to a CSV file.
///
/// The output has one header row followed by one data row per point:
/// ```text
/// idx,prev_x,prev_y,next_x,next_y,flow_x,flow_y,status,min_eigen
/// 0,158.000,124.000,162.010,127.000,4.010,3.000,1,88873.757813
/// …
/// ```
///
/// # Arguments
/// * `prev_pts`  — Feature positions in the previous frame.
/// * `next_pts`  — Estimated positions in the next frame.
/// * `status`    — Per-point tracking flag: `1` = tracked, `0` = lost.
/// * `err`       — Per-point tracking error / minimum eigenvalue.
/// * `path`      — Destination file path.
///
/// # Errors
/// Returns an error if the file cannot be created or written.
fn save_flow_csv(
    prev_pts: &[Point2f],
    next_pts: &[Point2f],
    status: &[u8],
    err: &[f32],
    path: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut f = std::fs::File::create(path)?;
    writeln!(
        f,
        "idx,prev_x,prev_y,next_x,next_y,flow_x,flow_y,status,min_eigen"
    )?;
    for (i, (prev, next)) in prev_pts.iter().zip(next_pts.iter()).enumerate() {
        writeln!(
            f,
            "{},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{},{:.6}",
            i,
            prev.x,
            prev.y,
            next.x,
            next.y,
            next.x - prev.x,
            next.y - prev.y,
            status[i],
            err[i],
        )?;
    }
    Ok(())
}

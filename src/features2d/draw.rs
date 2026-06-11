/*
 *  draw.rs
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

use crate::core::error::{PureCvError, Result};
use crate::core::types::Scalar;
use crate::core::Matrix;
use crate::features2d::{DMatch, KeyPoint};

/// Set the color of a specific pixel across all channels of the matrix.
#[inline]
fn set_pixel_color(img: &mut Matrix<u8>, row: i32, col: i32, color: &Scalar<u8>) {
    if row < 0 || row >= img.rows as i32 || col < 0 || col >= img.cols as i32 {
        return;
    }
    let r = row as usize;
    let c = col as usize;
    for ch in 0..img.channels {
        let val = color.v.get(ch).cloned().unwrap_or(0);
        img.set(r, c, ch, val);
    }
}

/// Bresenham's line algorithm (integer-only, extremely fast, safe bounds check).
fn draw_line_pixel(img: &mut Matrix<u8>, x0: i32, y0: i32, x1: i32, y1: i32, color: Scalar<u8>) {
    let dx = (x1 - x0).abs();
    let sx = if x0 < x1 { 1 } else { -1 };
    let dy = -(y1 - y0).abs();
    let sy = if y0 < y1 { 1 } else { -1 };
    let mut err = dx + dy;

    let mut x = x0;
    let mut y = y0;

    loop {
        set_pixel_color(img, y, x, &color);
        if x == x1 && y == y1 {
            break;
        }
        let e2 = 2 * err;
        if e2 >= dy {
            if x == x1 {
                break;
            }
            err += dy;
            x += sx;
        }
        if e2 <= dx {
            if y == y1 {
                break;
            }
            err += dx;
            y += sy;
        }
    }
}

/// Midpoint circle drawing algorithm (circumference only on pixel grid).
fn draw_circle_pixel(img: &mut Matrix<u8>, cx: i32, cy: i32, radius: i32, color: Scalar<u8>) {
    if radius < 0 {
        return;
    }
    if radius == 0 {
        set_pixel_color(img, cy, cx, &color);
        return;
    }

    let mut x = radius;
    let mut y = 0;
    let mut err = 1 - x;

    while x >= y {
        set_pixel_color(img, cy + y, cx + x, &color);
        set_pixel_color(img, cy + x, cx + y, &color);
        set_pixel_color(img, cy + x, cx - y, &color);
        set_pixel_color(img, cy + y, cx - x, &color);
        set_pixel_color(img, cy - y, cx - x, &color);
        set_pixel_color(img, cy - x, cx - y, &color);
        set_pixel_color(img, cy - x, cx + y, &color);
        set_pixel_color(img, cy - y, cx + x, &color);

        y += 1;
        if err < 0 {
            err += 2 * y + 1;
        } else {
            x -= 1;
            err += 2 * (y - x) + 1;
        }
    }
}

/// Draw circles on an image around detected keypoints.
pub fn draw_keypoints(
    image: &Matrix<u8>,
    keypoints: &[KeyPoint],
    color: Scalar<u8>,
) -> Result<Matrix<u8>> {
    let mut output = image.clone();

    for kp in keypoints {
        let cx = kp.pt.x.round() as i32;
        let cy = kp.pt.y.round() as i32;
        let r = (kp.size * 0.5).round() as i32;

        // Draw the keypoint neighborhood circle
        draw_circle_pixel(&mut output, cx, cy, r, color);

        // Draw orientation indicator line if angle is specified (>= 0.0)
        if kp.angle >= 0.0 {
            let angle_rad = kp.angle.to_radians();
            let rx = (kp.pt.x + (kp.size * 0.5) * angle_rad.cos()).round() as i32;
            let ry = (kp.pt.y + (kp.size * 0.5) * angle_rad.sin()).round() as i32;
            draw_line_pixel(&mut output, cx, cy, rx, ry, color);
        }
    }

    Ok(output)
}

/// Draw matches between two images side-by-side with lines connecting correspondences.
/// If `match_color` is `None`, a random color is generated for each match.
/// If `single_point_color` is `None`, unmatched keypoints are not drawn.
pub fn draw_matches(
    img1: &Matrix<u8>,
    keypoints1: &[KeyPoint],
    img2: &Matrix<u8>,
    keypoints2: &[KeyPoint],
    matches1to2: &[DMatch],
    match_color: Option<Scalar<u8>>,
    single_point_color: Option<Scalar<u8>>,
) -> Result<Matrix<u8>> {
    if img1.channels != img2.channels {
        return Err(PureCvError::IncompatibleChannels(format!(
            "Image channel count mismatch: {} vs {}",
            img1.channels, img2.channels
        )));
    }

    let height = img1.rows.max(img2.rows);
    let width = img1.cols + img2.cols;
    let channels = img1.channels;

    let mut output = Matrix::<u8>::zeros(height, width, channels);

    // Copy img1 to the left side
    for y in 0..img1.rows {
        for x in 0..img1.cols {
            for c in 0..channels {
                if let Some(&val) = img1.get(y, x, c) {
                    output.set(y, x, c, val);
                }
            }
        }
    }

    // Copy img2 to the right side
    for y in 0..img2.rows {
        for x in 0..img2.cols {
            for c in 0..channels {
                if let Some(&val) = img2.get(y, x, c) {
                    output.set(y, x + img1.cols, c, val);
                }
            }
        }
    }

    // Identify matched keypoints to color them differently if requested
    let mut matched1 = vec![false; keypoints1.len()];
    let mut matched2 = vec![false; keypoints2.len()];

    for m in matches1to2 {
        if m.query_idx >= 0 && (m.query_idx as usize) < keypoints1.len() {
            matched1[m.query_idx as usize] = true;
        }
        if m.train_idx >= 0 && (m.train_idx as usize) < keypoints2.len() {
            matched2[m.train_idx as usize] = true;
        }
    }

    // Draw single (unmatched) keypoints if single_point_color is not None
    if let Some(sp_color) = single_point_color {
        for (i, kp) in keypoints1.iter().enumerate() {
            if !matched1[i] {
                let cx = kp.pt.x.round() as i32;
                let cy = kp.pt.y.round() as i32;
                let r = (kp.size * 0.5).round() as i32;
                draw_circle_pixel(&mut output, cx, cy, r, sp_color);
                if kp.angle >= 0.0 {
                    let angle_rad = kp.angle.to_radians();
                    let rx = (kp.pt.x + (kp.size * 0.5) * angle_rad.cos()).round() as i32;
                    let ry = (kp.pt.y + (kp.size * 0.5) * angle_rad.sin()).round() as i32;
                    draw_line_pixel(&mut output, cx, cy, rx, ry, sp_color);
                }
            }
        }

        for (i, kp) in keypoints2.iter().enumerate() {
            if !matched2[i] {
                let cx = (kp.pt.x + img1.cols as f32).round() as i32;
                let cy = kp.pt.y.round() as i32;
                let r = (kp.size * 0.5).round() as i32;
                draw_circle_pixel(&mut output, cx, cy, r, sp_color);
                if kp.angle >= 0.0 {
                    let angle_rad = kp.angle.to_radians();
                    let rx = (kp.pt.x + img1.cols as f32 + (kp.size * 0.5) * angle_rad.cos())
                        .round() as i32;
                    let ry = (kp.pt.y + (kp.size * 0.5) * angle_rad.sin()).round() as i32;
                    draw_line_pixel(&mut output, cx, cy, rx, ry, sp_color);
                }
            }
        }
    }

    // LCG random color generator seed
    let mut seed = 0x12345678u32;
    let mut next_random_color = || -> Scalar<u8> {
        seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
        let r = (seed & 0xFF) as u8;
        let g = ((seed >> 8) & 0xFF) as u8;
        let b = ((seed >> 16) & 0xFF) as u8;
        Scalar::new(r, g, b, 255)
    };

    // Draw matched keypoints and the connection lines
    for m in matches1to2 {
        if m.query_idx < 0
            || (m.query_idx as usize) >= keypoints1.len()
            || m.train_idx < 0
            || (m.train_idx as usize) >= keypoints2.len()
        {
            continue;
        }

        let kp1 = &keypoints1[m.query_idx as usize];
        let kp2 = &keypoints2[m.train_idx as usize];

        let cx1 = kp1.pt.x.round() as i32;
        let cy1 = kp1.pt.y.round() as i32;
        let r1 = (kp1.size * 0.5).round() as i32;

        let cx2 = (kp2.pt.x + img1.cols as f32).round() as i32;
        let cy2 = kp2.pt.y.round() as i32;
        let r2 = (kp2.size * 0.5).round() as i32;

        let current_color = match match_color {
            Some(color) => color,
            None => next_random_color(),
        };

        // Draw circles around the matched keypoints
        draw_circle_pixel(&mut output, cx1, cy1, r1, current_color);
        draw_circle_pixel(&mut output, cx2, cy2, r2, current_color);

        // Draw matching line
        draw_line_pixel(&mut output, cx1, cy1, cx2, cy2, current_color);
    }

    Ok(output)
}

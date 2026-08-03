/*
 *  hough.rs
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

use alloc::{vec, vec::Vec};
#[allow(unused_imports)]
use num_traits::Float;

use crate::core::constants::CV_PI;
use crate::core::error::{PureCvError, Result};
use crate::core::types::BorderTypes;
use crate::core::Matrix;
use crate::imgproc::derivatives::sobel;

/// Finds lines in a binary image using the standard Hough transform.
///
/// # Arguments
/// * `image` - 8-bit, single-channel binary source image.
/// * `rho` - Distance resolution of the accumulator in pixels.
/// * `theta` - Angle resolution of the accumulator in radians.
/// * `threshold` - Accumulator threshold parameter. Only those lines are returned that get enough votes.
/// * `min_theta` - For standard and multi-scale Hough transform, minimum angle to check for lines.
/// * `max_theta` - For standard and multi-scale Hough transform, maximum angle to check for lines.
///
/// # Returns
/// A vector of `[f32; 2]` representing `(rho, theta)` for each detected line.
pub fn hough_lines(
    image: &Matrix<u8>,
    rho: f64,
    theta: f64,
    threshold: i32,
    min_theta: f64,
    max_theta: f64,
) -> Result<Vec<[f32; 2]>> {
    if image.channels != 1 {
        return Err(PureCvError::InvalidInput(
            "hough_lines requires single-channel 8-bit image".into(),
        ));
    }

    if min_theta >= max_theta {
        return Err(PureCvError::InvalidInput(
            "min_theta must be strictly less than max_theta".into(),
        ));
    }

    let width = image.cols;
    let height = image.rows;

    let mut numangle = ((max_theta - min_theta) / theta).floor() as usize + 1;
    if numangle > 1 && (CV_PI - (numangle as f64 - 1.0) * theta).abs() < theta / 2.0 {
        numangle -= 1;
    }

    let max_rho = (width + height) as f64;
    let min_rho = -max_rho;
    let numrho = ((max_rho - min_rho) / rho).round() as usize + 1;

    let irho = 1.0 / rho;
    let mut tab_sin = vec![0f32; numangle];
    let mut tab_cos = vec![0f32; numangle];

    let mut ang = min_theta;
    for n in 0..numangle {
        tab_sin[n] = (ang.sin() * irho) as f32;
        tab_cos[n] = (ang.cos() * irho) as f32;
        ang += theta;
    }

    let mut accum = vec![0i32; (numangle + 2) * (numrho + 2)];

    // Stage 1. Fill accumulator
    for i in 0..height {
        let row_offset = i * width;
        for j in 0..width {
            let px = image.data[row_offset + j];
            if px != 0 {
                for n in 0..numangle {
                    let r = (j as f32 * tab_cos[n] + i as f32 * tab_sin[n]).round() as isize;
                    let r_idx = r + ((numrho as isize - 1) / 2);
                    let idx = (n + 1) * (numrho + 2) + (r_idx as usize) + 1;
                    accum[idx] += 1;
                }
            }
        }
    }

    // Stage 2. Find local maximums
    let mut sort_buf = Vec::new();
    for r in 0..numrho {
        for n in 0..numangle {
            let base = (n + 1) * (numrho + 2) + r + 1;
            let val = accum[base];
            if val > threshold
                && val > accum[base - 1]
                && val >= accum[base + 1]
                && val > accum[base - numrho - 2]
                && val >= accum[base + numrho + 2]
            {
                sort_buf.push((base, val));
            }
        }
    }

    // Stage 3. Sort the detected lines by accumulator value descending
    sort_buf.sort_by_key(|b| core::cmp::Reverse(b.1));

    // Stage 4. Format output
    let mut lines = Vec::with_capacity(sort_buf.len());
    let scale = 1.0 / (numrho as f64 + 2.0);

    for (idx, _val) in sort_buf {
        let n = ((idx as f64) * scale).floor() as usize - 1;
        let r = idx - (n + 1) * (numrho + 2) - 1;
        let rho_val = ((r as f64) - (numrho as f64 - 1.0) / 2.0) * rho;
        let angle_val = min_theta + (n as f64) * theta;
        lines.push([rho_val as f32, angle_val as f32]);
    }

    Ok(lines)
}

/// Finds line segments in a binary image using the probabilistic Hough transform.
///
/// # Arguments
/// * `image` - 8-bit, single-channel binary source image.
/// * `rho` - Distance resolution of the accumulator in pixels.
/// * `theta` - Angle resolution of the accumulator in radians.
/// * `threshold` - Accumulator threshold parameter.
/// * `min_line_length` - Minimum line length. Line segments shorter than that are rejected.
/// * `max_line_gap` - Maximum allowed gap between points on the same line to link them.
///
/// # Returns
/// A vector of `[i32; 4]` representing `(x1, y1, x2, y2)` for each segment.
///
/// Requires the `std` feature: the probabilistic transform shuffles edge points
/// with the thread-local RNG ([`crate::core::rand_shuffle`]), which is std-only
/// until a no_std seedable RNG lands (see issue #82). The deterministic
/// [`hough_lines`] is available under `no_std`.
#[cfg(feature = "std")]
pub fn hough_lines_p(
    image: &Matrix<u8>,
    rho: f64,
    theta: f64,
    threshold: i32,
    min_line_length: f64,
    max_line_gap: f64,
) -> Result<Vec<[i32; 4]>> {
    if image.channels != 1 {
        return Err(PureCvError::InvalidInput(
            "hough_lines_p requires single-channel 8-bit image".into(),
        ));
    }

    let width = image.cols;
    let height = image.rows;
    let mut mask = image.clone();

    let numangle = (CV_PI / theta).round() as usize;
    let numrho = (((width + height) * 2 + 1) as f64 / rho).round() as usize;

    let irho = 1.0 / rho;
    let mut trigtab = vec![0f32; numangle * 2];
    for n in 0..numangle {
        let ang = (n as f64) * theta;
        trigtab[n * 2] = (ang.cos() * irho) as f32;
        trigtab[n * 2 + 1] = (ang.sin() * irho) as f32;
    }

    let mut nzloc = Vec::new();
    for i in 0..height {
        let row_offset = i * width;
        for j in 0..width {
            if image.data[row_offset + j] != 0 {
                nzloc.push((j as i32, i as i32));
                mask.data[row_offset + j] = 1;
            } else {
                mask.data[row_offset + j] = 0;
            }
        }
    }

    let mut count = nzloc.len();
    crate::core::rand_shuffle(&mut nzloc);
    let mut accum = vec![0i32; numangle * numrho];
    let mut lines = Vec::new();

    let shift = 16;

    while count > 0 {
        let mut max_val = threshold - 1;
        let mut max_n = 0;
        count -= 1;
        let (jx, iy) = nzloc[count];

        if mask.data[iy as usize * width + jx as usize] == 0 {
            continue;
        }

        for n in 0..numangle {
            let r = (jx as f32 * trigtab[n * 2] + iy as f32 * trigtab[n * 2 + 1]).round() as isize;
            let r_idx = r + (numrho as isize - 1) / 2;
            let a_idx = n * numrho + r_idx as usize;
            accum[a_idx] += 1;
            let val = accum[a_idx];

            if val > max_val {
                max_val = val;
                max_n = n;
            }
        }

        if max_val < threshold {
            continue;
        }

        let a = -trigtab[max_n * 2 + 1];
        let b = trigtab[max_n * 2];
        let mut x0 = jx as i64;
        let mut y0 = iy as i64;

        let dx0: i64;
        let dy0: i64;
        let xflag: bool;

        if a.abs() > b.abs() {
            xflag = true;
            dx0 = if a > 0.0 { 1 } else { -1 };
            dy0 = ((b * (1 << shift) as f32) / a.abs()).round() as i64;
            y0 = (y0 << shift) + (1 << (shift - 1));
        } else {
            xflag = false;
            dy0 = if b > 0.0 { 1 } else { -1 };
            dx0 = ((a * (1 << shift) as f32) / b.abs()).round() as i64;
            x0 = (x0 << shift) + (1 << (shift - 1));
        }

        let mut line_end = [(0, 0), (0, 0)];

        for (k, end_pt) in line_end.iter_mut().enumerate() {
            let mut gap = 0;
            let mut x = x0;
            let mut y = y0;
            let dx = if k > 0 { -dx0 } else { dx0 };
            let dy = if k > 0 { -dy0 } else { dy0 };

            loop {
                let j1: i64;
                let i1: i64;
                if xflag {
                    j1 = x;
                    i1 = y >> shift;
                } else {
                    j1 = x >> shift;
                    i1 = y;
                }

                if j1 < 0 || j1 >= width as i64 || i1 < 0 || i1 >= height as i64 {
                    break;
                }

                let mdata = mask.data[i1 as usize * width + j1 as usize];

                if mdata != 0 {
                    gap = 0;
                    *end_pt = (j1 as i32, i1 as i32);
                } else {
                    gap += 1;
                    if gap > max_line_gap as i32 {
                        break;
                    }
                }

                x += dx;
                y += dy;
            }
        }

        let dx = line_end[1].0 - line_end[0].0;
        let dy = line_end[1].1 - line_end[0].1;
        let good_line =
            (dx.abs() >= min_line_length as i32) || (dy.abs() >= min_line_length as i32);

        for (k, end_pt) in line_end.iter().enumerate() {
            let mut x = x0;
            let mut y = y0;
            let dx_step = if k > 0 { -dx0 } else { dx0 };
            let dy_step = if k > 0 { -dy0 } else { dy0 };

            loop {
                let j1: i64;
                let i1: i64;
                if xflag {
                    j1 = x;
                    i1 = y >> shift;
                } else {
                    j1 = x >> shift;
                    i1 = y;
                }

                let m_idx = i1 as usize * width + j1 as usize;

                if mask.data[m_idx] != 0 {
                    if good_line {
                        for n in 0..numangle {
                            let r = (j1 as f32 * trigtab[n * 2] + i1 as f32 * trigtab[n * 2 + 1])
                                .round() as isize;
                            let r_idx = r + (numrho as isize - 1) / 2;
                            accum[n * numrho + r_idx as usize] -= 1;
                        }
                    }
                    mask.data[m_idx] = 0;
                }

                if i1 == end_pt.1 as i64 && j1 == end_pt.0 as i64 {
                    break;
                }

                x += dx_step;
                y += dy_step;
            }
        }

        if good_line {
            lines.push([line_end[0].0, line_end[0].1, line_end[1].0, line_end[1].1]);
        }
    }

    Ok(lines)
}

/// Finds circles in a grayscale image using the Hough transform.
/// Uses the gradient method (similar to HOUGH_GRADIENT).
pub fn hough_circles(
    image: &Matrix<u8>,
    dp: f64,
    min_dist: f64,
    param1: f64,
    param2: f64,
    min_radius: i32,
    max_radius: i32,
) -> Result<Vec<[f32; 3]>> {
    if image.channels != 1 {
        return Err(PureCvError::InvalidInput(
            "Image must be single-channel".into(),
        ));
    }

    let width = image.cols;
    let height = image.rows;
    let accum_width = (width as f64 / dp).ceil() as usize;
    let accum_height = (height as f64 / dp).ceil() as usize;

    // We compute gradients with Sobel.
    let src_f64 = image.convert_to::<f64>()?;
    let dx = sobel(&src_f64, 1, 0, 3, 1.0, 0.0, BorderTypes::Replicate)?;
    let dy = sobel(&src_f64, 0, 1, 3, 1.0, 0.0, BorderTypes::Replicate)?;

    let mut edges = Vec::new();
    let mut accum = vec![0i32; accum_width * accum_height];

    let canny_threshold = param1.max(1.0);
    // Note: A full canny is not explicitly requested, OpenCV uses a custom local Canny or just gradient thresholding
    // for circle detection. We use a simple magnitude threshold for edges.
    for y in 0..height {
        for x in 0..width {
            let vx = dx.get(y, x, 0).copied().unwrap_or(0.0);
            let vy = dy.get(y, x, 0).copied().unwrap_or(0.0);
            let mag = (vx * vx + vy * vy).sqrt();
            if mag > canny_threshold {
                edges.push((x, y, vx, vy, mag));

                // Draw line along gradient in accumulator
                let dx_norm = vx / mag;
                let dy_norm = vy / mag;

                // We cast votes for min_radius to max_radius
                for dir in [-1.0, 1.0].iter() {
                    for r in min_radius..=max_radius {
                        let cx = x as f64 + dir * dx_norm * r as f64;
                        let cy = y as f64 + dir * dy_norm * r as f64;

                        let acx = (cx / dp).round() as isize;
                        let acy = (cy / dp).round() as isize;

                        if acx >= 0
                            && acx < accum_width as isize
                            && acy >= 0
                            && acy < accum_height as isize
                        {
                            accum[acy as usize * accum_width + acx as usize] += 1;
                        }
                    }
                }
            }
        }
    }

    // Find local maxima in accumulator map
    let mut centers = Vec::new();
    let threshold = param2 as i32;
    for y in 1..(accum_height - 1) {
        for x in 1..(accum_width - 1) {
            let val = accum[y * accum_width + x];
            if val > threshold {
                let mut is_local_max = true;
                for dy in -1..=1 {
                    for dx in -1..=1 {
                        if dx == 0 && dy == 0 {
                            continue;
                        }
                        if accum
                            [(y as isize + dy) as usize * accum_width + (x as isize + dx) as usize]
                            >= val
                        {
                            is_local_max = false;
                            break;
                        }
                    }
                    if !is_local_max {
                        break;
                    }
                }

                if is_local_max {
                    centers.push((x as f64 * dp, y as f64 * dp, val));
                }
            }
        }
    }

    // Sort centers by votes
    centers.sort_by_key(|b| core::cmp::Reverse(b.2));

    let mut circles = Vec::new();

    // For each center, verify it with edge points to find true radius
    // Since we filtered by min_dist, we first filter centers
    let mut filtered_centers: Vec<(f64, f64, i32)> = Vec::new();
    for c in centers {
        let mut ok = true;
        for fc in &filtered_centers {
            let dx = c.0 - fc.0;
            let dy = c.1 - fc.1;
            if dx * dx + dy * dy < min_dist * min_dist {
                ok = false;
                break;
            }
        }
        if ok {
            filtered_centers.push(c);
        }
    }

    for c in filtered_centers {
        let mut dists = Vec::new();
        // we can find points that have distance roughly equals to radius
        for &(ex, ey, _, _, _) in &edges {
            let dist = ((ex as f64 - c.0).powi(2) + (ey as f64 - c.1).powi(2)).sqrt();
            if dist >= min_radius as f64 && dist <= max_radius as f64 {
                dists.push(dist.round() as i32);
            }
        }

        if dists.is_empty() {
            continue;
        }

        dists.sort_unstable();

        let mut max_count = 0;
        let mut best_r = 0;
        let mut current_count = 1;
        let mut current_val = dists[0];

        for &d in dists.iter().skip(1) {
            if d == current_val {
                current_count += 1;
            } else {
                if current_count > max_count {
                    max_count = current_count;
                    best_r = current_val;
                }

                current_val = d;
                current_count = 1;
            }
        }
        if current_count > max_count {
            max_count = current_count;
            best_r = current_val;
        }

        // threshold can also be applied to radius votes
        if max_count >= param2 as i32 / 2 {
            circles.push([c.0 as f32, c.1 as f32, best_r as f32]);
        }
    }

    Ok(circles)
}

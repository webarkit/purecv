/*
 *  fast.rs
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

use super::keypoint::KeyPoint;
use crate::core::error::{PureCvError, Result};
use crate::core::types::Point2f;
use crate::core::Matrix;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// The type of FAST detector.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FastType {
    /// FAST-5 (5 out of 8 pixels)
    Type5_8,
    /// FAST-7 (7 out of 12 pixels)
    Type7_12,
    /// FAST-9 (9 out of 16 pixels - OpenCV default)
    Type9_16,
}

static OFFSETS_16: [(i32, i32); 16] = [
    (0, 3),
    (1, 3),
    (2, 2),
    (3, 1),
    (3, 0),
    (3, -1),
    (2, -2),
    (1, -3),
    (0, -3),
    (-1, -3),
    (-2, -2),
    (-3, -1),
    (-3, 0),
    (-3, 1),
    (-2, 2),
    (-1, 3),
];

static OFFSETS_12: [(i32, i32); 12] = [
    (0, 2),
    (1, 2),
    (2, 1),
    (2, 0),
    (2, -1),
    (1, -2),
    (0, -2),
    (-1, -2),
    (-2, -1),
    (-2, 0),
    (-2, 1),
    (-1, 2),
];

static OFFSETS_8: [(i32, i32); 8] = [
    (0, 1),
    (1, 1),
    (1, 0),
    (1, -1),
    (0, -1),
    (-1, -1),
    (-1, 0),
    (-1, 1),
];

/// Features from Accelerated Segment Test (FAST) keypoint detector.
///
/// Ref: https://docs.opencv.org/4.10.0/df/d74/classcv_1_1FastFeatureDetector.html
#[derive(Debug, Clone)]
pub struct FastFeatureDetector {
    threshold: u8,
    nonmax_suppression: bool,
    detector_type: FastType,
}

impl Default for FastFeatureDetector {
    /// Creates a FAST detector with default OpenCV parameters (threshold = 10, nonmax_suppression = true, type = Type9_16).
    fn default() -> Self {
        Self {
            threshold: 10,
            nonmax_suppression: true,
            detector_type: FastType::Type9_16,
        }
    }
}

impl FastFeatureDetector {
    /// Creates a new FAST feature detector instance.
    ///
    /// * `threshold` - Threshold on difference between intensity of the central pixel and pixels of a circle around this pixel.
    /// * `nonmax_suppression` - If true, non-maximum suppression is applied to detected corners (keypoints).
    /// * `detector_type` - The neighborhood configuration (Type5_8, Type7_12, or Type9_16).
    pub fn new(threshold: u8, nonmax_suppression: bool, detector_type: FastType) -> Self {
        Self {
            threshold,
            nonmax_suppression,
            detector_type,
        }
    }

    /// Gets the threshold value.
    pub fn threshold(&self) -> u8 {
        self.threshold
    }

    /// Sets the threshold value.
    pub fn set_threshold(&mut self, threshold: u8) {
        self.threshold = threshold;
    }

    /// Gets whether non-maximum suppression is enabled.
    pub fn nonmax_suppression(&self) -> bool {
        self.nonmax_suppression
    }

    /// Sets whether non-maximum suppression is enabled.
    pub fn set_nonmax_suppression(&mut self, enabled: bool) {
        self.nonmax_suppression = enabled;
    }

    /// Gets the detector configuration type.
    pub fn detector_type(&self) -> FastType {
        self.detector_type
    }

    /// Sets the detector configuration type.
    pub fn set_detector_type(&mut self, detector_type: FastType) {
        self.detector_type = detector_type;
    }

    /// Detects keypoints in an image.
    ///
    /// * `image` - Grayscale input image (matrix).
    #[allow(clippy::needless_range_loop)]
    pub fn detect(&self, image: &Matrix<u8>) -> Result<Vec<KeyPoint>> {
        if image.channels != 1 {
            return Err(PureCvError::InvalidInput(
                "FAST keypoint detection requires a single-channel grayscale image".to_string(),
            ));
        }

        let rows = image.rows;
        let cols = image.cols;
        let radius = match self.detector_type {
            FastType::Type5_8 => 1,
            FastType::Type7_12 => 3,
            FastType::Type9_16 => 3,
        };

        if rows <= radius * 2 || cols <= radius * 2 {
            return Ok(Vec::new());
        }

        let threshold = self.threshold;
        let detector_type = self.detector_type;
        let stride = cols as isize;

        // Precompute circle flat index offsets
        let circle_offsets: Vec<isize> = match detector_type {
            FastType::Type5_8 => OFFSETS_8
                .iter()
                .map(|&(dx, dy)| dy as isize * stride + dx as isize)
                .collect(),
            FastType::Type7_12 => OFFSETS_12
                .iter()
                .map(|&(dx, dy)| dy as isize * stride + dx as isize)
                .collect(),
            FastType::Type9_16 => OFFSETS_16
                .iter()
                .map(|&(dx, dy)| dy as isize * stride + dx as isize)
                .collect(),
        };

        // Pass 1: Compute corner scores for all pixels in parallel or sequential
        let mut scores = vec![0u8; rows * cols];

        #[cfg(feature = "parallel")]
        {
            scores
                .par_chunks_exact_mut(cols)
                .enumerate()
                .for_each(|(y, score_row)| {
                    if y >= radius && y < rows - radius {
                        for x in radius..(cols - radius) {
                            let idx = y * cols + x;
                            score_row[x] = detect_pixel(
                                &image.data,
                                idx,
                                detector_type,
                                threshold,
                                &circle_offsets,
                            );
                        }
                    }
                });
        }

        #[cfg(not(feature = "parallel"))]
        {
            scores
                .chunks_exact_mut(cols)
                .enumerate()
                .for_each(|(y, score_row)| {
                    if y >= radius && y < rows - radius {
                        for x in radius..(cols - radius) {
                            let idx = y * cols + x;
                            score_row[x] = detect_pixel(
                                &image.data,
                                idx,
                                detector_type,
                                threshold,
                                &circle_offsets,
                            );
                        }
                    }
                });
        }

        // Pass 2: Extract keypoints, applying Non-Maximum Suppression if enabled
        let nonmax = self.nonmax_suppression;

        #[cfg(feature = "parallel")]
        let keypoints: Vec<KeyPoint> = (radius..(rows - radius))
            .into_par_iter()
            .flat_map(|y| {
                let mut row_kps = Vec::new();
                let offset = y * cols;
                let prev_offset = (y - 1) * cols;
                let next_offset = (y + 1) * cols;

                for x in radius..(cols - radius) {
                    let score = scores[offset + x];
                    if score > 0 {
                        if !nonmax {
                            row_kps.push(KeyPoint::new(
                                Point2f::new(x as f32, y as f32),
                                7.0,
                                -1.0,
                                score as f32,
                                0,
                                -1,
                            ));
                        } else {
                            // 8-neighborhood strictly-greater validation
                            if score > scores[offset + x - 1]
                                && score > scores[offset + x + 1]
                                && score > scores[prev_offset + x - 1]
                                && score > scores[prev_offset + x]
                                && score > scores[prev_offset + x + 1]
                                && score > scores[next_offset + x - 1]
                                && score > scores[next_offset + x]
                                && score > scores[next_offset + x + 1]
                            {
                                row_kps.push(KeyPoint::new(
                                    Point2f::new(x as f32, y as f32),
                                    7.0,
                                    -1.0,
                                    score as f32,
                                    0,
                                    -1,
                                ));
                            }
                        }
                    }
                }
                row_kps
            })
            .collect();

        #[cfg(not(feature = "parallel"))]
        let keypoints: Vec<KeyPoint> = (radius..(rows - radius))
            .into_iter()
            .flat_map(|y| {
                let mut row_kps = Vec::new();
                let offset = y * cols;
                let prev_offset = (y - 1) * cols;
                let next_offset = (y + 1) * cols;

                for x in radius..(cols - radius) {
                    let score = scores[offset + x];
                    if score > 0 {
                        if !nonmax {
                            row_kps.push(KeyPoint::new(
                                Point2f::new(x as f32, y as f32),
                                7.0,
                                -1.0,
                                score as f32,
                                0,
                                -1,
                            ));
                        } else {
                            if score > scores[offset + x - 1]
                                && score > scores[offset + x + 1]
                                && score > scores[prev_offset + x - 1]
                                && score > scores[prev_offset + x]
                                && score > scores[prev_offset + x + 1]
                                && score > scores[next_offset + x - 1]
                                && score > scores[next_offset + x]
                                && score > scores[next_offset + x + 1]
                            {
                                row_kps.push(KeyPoint::new(
                                    Point2f::new(x as f32, y as f32),
                                    7.0,
                                    -1.0,
                                    score as f32,
                                    0,
                                    -1,
                                ));
                            }
                        }
                    }
                }
                row_kps
            })
            .collect();

        Ok(keypoints)
    }
}

/// Helper function to perform FAST corner check and compute response score for a single pixel.
fn detect_pixel(
    image_data: &[u8],
    idx: usize,
    detector_type: FastType,
    threshold: u8,
    circle_offsets: &[isize],
) -> u8 {
    let v = image_data[idx];
    let vt_bright = v.saturating_add(threshold);
    let vt_dark = v.saturating_sub(threshold);

    match detector_type {
        FastType::Type5_8 => {
            let mut circle = [0u8; 8];
            for i in 0..8 {
                circle[i] = image_data[(idx as isize + circle_offsets[i]) as usize];
            }

            // Early rejection check: at least one in opposite pairs must exceed threshold bounds
            let maybe_bright = (circle[0] > vt_bright || circle[4] > vt_bright)
                && (circle[2] > vt_bright || circle[6] > vt_bright);
            let maybe_dark = (circle[0] < vt_dark || circle[4] < vt_dark)
                && (circle[2] < vt_dark || circle[6] < vt_dark);

            if !maybe_bright && !maybe_dark {
                return 0;
            }

            let (is_bright, is_dark) = check_contiguous(&circle, v, 5, threshold);
            if is_bright || is_dark {
                corner_score_8(&circle, v, threshold)
            } else {
                0
            }
        }
        FastType::Type7_12 => {
            let mut circle = [0u8; 12];
            for i in 0..12 {
                circle[i] = image_data[(idx as isize + circle_offsets[i]) as usize];
            }

            let maybe_bright = (circle[0] > vt_bright || circle[6] > vt_bright)
                && (circle[3] > vt_bright || circle[9] > vt_bright);
            let maybe_dark = (circle[0] < vt_dark || circle[6] < vt_dark)
                && (circle[3] < vt_dark || circle[9] < vt_dark);

            if !maybe_bright && !maybe_dark {
                return 0;
            }

            let (is_bright, is_dark) = check_contiguous(&circle, v, 7, threshold);
            if is_bright || is_dark {
                corner_score_12(&circle, v, threshold)
            } else {
                0
            }
        }
        FastType::Type9_16 => {
            let mut circle = [0u8; 16];
            for i in 0..16 {
                circle[i] = image_data[(idx as isize + circle_offsets[i]) as usize];
            }

            let maybe_bright = (circle[0] > vt_bright || circle[8] > vt_bright)
                && (circle[4] > vt_bright || circle[12] > vt_bright);
            let maybe_dark = (circle[0] < vt_dark || circle[8] < vt_dark)
                && (circle[4] < vt_dark || circle[12] < vt_dark);

            if !maybe_bright && !maybe_dark {
                return 0;
            }

            let (is_bright, is_dark) = check_contiguous(&circle, v, 9, threshold);
            if is_bright || is_dark {
                corner_score_16(&circle, v, threshold)
            } else {
                0
            }
        }
    }
}

/// Checks if there is a contiguous sequence of at least `k` pixels in the circle that are
/// all strictly brighter than `v + threshold` or strictly darker than `v - threshold`.
fn check_contiguous(circle: &[u8], v: u8, k: usize, threshold: u8) -> (bool, bool) {
    let n = circle.len();
    let vt_bright = v.saturating_add(threshold);
    let vt_dark = v.saturating_sub(threshold);

    let mut bright_len = 0;
    let mut dark_len = 0;
    let mut is_bright = false;
    let mut is_dark = false;

    // Scan with wrap-around buffer expansion of size k-1
    for i in 0..(n + k - 1) {
        let val = circle[i % n];
        if val > vt_bright {
            bright_len += 1;
            if bright_len >= k {
                is_bright = true;
            }
        } else {
            bright_len = 0;
        }

        if val < vt_dark {
            dark_len += 1;
            if dark_len >= k {
                is_dark = true;
            }
        } else {
            dark_len = 0;
        }
    }

    (is_bright, is_dark)
}

/// Edward Rosten's optimized corner score algorithm for FAST-9 (Type9_16)
fn corner_score_16(circle: &[u8; 16], v: u8, threshold: u8) -> u8 {
    let mut d = [0i16; 25];
    for i in 0..16 {
        d[i] = v as i16 - circle[i] as i16;
    }
    for i in 0..9 {
        d[16 + i] = d[i];
    }

    let mut a0 = threshold as i16;
    for k in (0..16).step_by(2) {
        let mut a = d[k + 1].min(d[k + 2]);
        a = a.min(d[k + 3]);
        if a <= a0 {
            continue;
        }
        a = a.min(d[k + 4]);
        a = a.min(d[k + 5]);
        a = a.min(d[k + 6]);
        a = a.min(d[k + 7]);
        a = a.min(d[k + 8]);
        a0 = a0.max(a.min(d[k]));
        a0 = a0.max(a.min(d[k + 9]));
    }

    let mut b0 = -a0;
    for k in (0..16).step_by(2) {
        let mut b = d[k + 1].max(d[k + 2]);
        b = b.max(d[k + 3]);
        b = b.max(d[k + 4]);
        b = b.max(d[k + 5]);
        if b >= b0 {
            continue;
        }
        b = b.max(d[k + 6]);
        b = b.max(d[k + 7]);
        b = b.max(d[k + 8]);
        b0 = b0.min(b.max(d[k]));
        b0 = b0.min(b.max(d[k + 9]));
    }

    (-b0 - 1) as u8
}

/// Edward Rosten's optimized corner score algorithm for FAST-7 (Type7_12)
fn corner_score_12(circle: &[u8; 12], v: u8, threshold: u8) -> u8 {
    let mut d = [0i16; 19];
    for i in 0..12 {
        d[i] = v as i16 - circle[i] as i16;
    }
    for i in 0..7 {
        d[12 + i] = d[i];
    }

    let mut a0 = threshold as i16;
    for k in (0..12).step_by(2) {
        let mut a = d[k + 1].min(d[k + 2]);
        if a <= a0 {
            continue;
        }
        a = a.min(d[k + 3]);
        a = a.min(d[k + 4]);
        a = a.min(d[k + 5]);
        a = a.min(d[k + 6]);
        a0 = a0.max(a.min(d[k]));
        a0 = a0.max(a.min(d[k + 7]));
    }

    let mut b0 = -a0;
    for k in (0..12).step_by(2) {
        let mut b = d[k + 1].max(d[k + 2]);
        b = b.max(d[k + 3]);
        b = b.max(d[k + 4]);
        if b >= b0 {
            continue;
        }
        b = b.max(d[k + 5]);
        b = b.max(d[k + 6]);
        b0 = b0.min(b.max(d[k]));
        b0 = b0.min(b.max(d[k + 7]));
    }

    (-b0 - 1) as u8
}

/// Edward Rosten's optimized corner score algorithm for FAST-5 (Type5_8)
fn corner_score_8(circle: &[u8; 8], v: u8, threshold: u8) -> u8 {
    let mut d = [0i16; 13];
    for i in 0..8 {
        d[i] = v as i16 - circle[i] as i16;
    }
    for i in 0..5 {
        d[8 + i] = d[i];
    }

    let mut a0 = threshold as i16;
    for k in (0..8).step_by(2) {
        let mut a = d[k + 1].min(d[k + 2]);
        if a <= a0 {
            continue;
        }
        a = a.min(d[k + 3]);
        a = a.min(d[k + 4]);
        a0 = a0.max(a.min(d[k]));
        a0 = a0.max(a.min(d[k + 5]));
    }

    let mut b0 = -a0;
    for k in (0..8).step_by(2) {
        let mut b = d[k + 1].max(d[k + 2]);
        b = b.max(d[k + 3]);
        if b >= b0 {
            continue;
        }
        b = b.max(d[k + 4]);
        b0 = b0.min(b.max(d[k]));
        b0 = b0.min(b.max(d[k + 5]));
    }

    (-b0 - 1) as u8
}

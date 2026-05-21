/*
 *  keypoint.rs
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
use crate::core::types::Point2f;
use std::default::Default;

/// Data structure for salient point detectors.
///
/// Ref: https://docs.opencv.org/4.10.0/d2/d29/classcv_1_1KeyPoint.html
#[derive(Debug, Clone, PartialEq)]
pub struct KeyPoint {
    /// Coordinates of the keypoint.
    pub pt: Point2f,
    /// Diameter of the keypoint neighborhood.
    pub size: f32,
    /// Computed orientation of the keypoint (-1 if not applicable).
    pub angle: f32,
    /// The response by which the most strong keypoints have been selected.
    pub response: f32,
    /// Octave (pyramid layer) from which the keypoint has been extracted.
    pub octave: i32,
    /// Object class identifier.
    pub class_id: i32,
}

impl Default for KeyPoint {
    /// Creates a keypoint with default values.
    fn default() -> Self {
        Self {
            pt: Point2f::default(),
            size: 0.0,
            angle: -1.0,
            response: 0.0,
            octave: 0,
            class_id: -1,
        }
    }
}

impl KeyPoint {
    /// Creates a new custom KeyPoint.
    ///
    /// * `pt` - Coordinates of the keypoint.
    /// * `size` - Diameter of the keypoint neighborhood.
    /// * `angle` - Orientation of the keypoint in degrees.
    /// * `response` - Keypoint strength detector response.
    /// * `octave` - Keypoint octave (pyramid layer).
    /// * `class_id` - Object class identifier.
    pub fn new(
        pt: Point2f,
        size: f32,
        angle: f32,
        response: f32,
        octave: i32,
        class_id: i32,
    ) -> Self {
        Self {
            pt,
            size,
            angle,
            response,
            octave,
            class_id,
        }
    }

    /// Computes the overlap ratio of two keypoints, which is the intersection area over union area of the keypoint circles.
    ///
    /// Matches the exact OpenCV C++ logic:
    /// Ref: https://github.com/opencv/opencv/blob/4.10.0/modules/core/src/types.cpp#L103
    pub fn overlap(kp1: &KeyPoint, kp2: &KeyPoint) -> f32 {
        let a = kp1.size * 0.5;
        let b = kp2.size * 0.5;
        let a_2 = a * a;
        let b_2 = b * b;

        let dx = kp1.pt.x - kp2.pt.x;
        let dy = kp1.pt.y - kp2.pt.y;
        let c = (dx * dx + dy * dy).sqrt();

        // One circle is completely enclosed by the other => no intersection points!
        if a.min(b) + c <= a.max(b) {
            let min_s_2 = a_2.min(b_2);
            let max_s_2 = a_2.max(b_2);
            if max_s_2 > 0.0 {
                return min_s_2 / max_s_2;
            } else {
                return 0.0;
            }
        }

        if c < a + b && c > 0.0 {
            let c_2 = c * c;
            let cos_alpha = (b_2 + c_2 - a_2) / (kp2.size * c);
            let cos_beta = (a_2 + c_2 - b_2) / (kp1.size * c);

            // Clamp values to prevent acos domain errors due to precision
            let cos_alpha = cos_alpha.clamp(-1.0, 1.0);
            let cos_beta = cos_beta.clamp(-1.0, 1.0);

            let alpha = cos_alpha.acos();
            let beta = cos_beta.acos();

            let sin_alpha = alpha.sin();
            let sin_beta = beta.sin();

            let segment_area_a = a_2 * beta;
            let segment_area_b = b_2 * alpha;

            let triangle_area_a = a_2 * sin_beta * cos_beta;
            let triangle_area_b = b_2 * sin_alpha * cos_alpha;

            let intersection_area =
                segment_area_a + segment_area_b - triangle_area_a - triangle_area_b;
            let union_area = (a_2 + b_2) * std::f32::consts::PI - intersection_area;

            if union_area > 0.0 {
                return intersection_area / union_area;
            }
        }

        0.0
    }

    /// Converts a list of KeyPoints to Point2f coordinates.
    ///
    /// Matches the exact OpenCV C++ logic:
    /// Ref: https://github.com/opencv/opencv/blob/4.10.0/modules/core/src/types.cpp#L65
    pub fn convert_to_points(keypoints: &[KeyPoint]) -> Vec<Point2f> {
        keypoints.iter().map(|kp| kp.pt).collect()
    }

    /// Converts a list of KeyPoints to Point2f coordinates using an index mask.
    ///
    /// Matches the exact OpenCV C++ logic:
    /// Ref: https://github.com/opencv/opencv/blob/4.10.0/modules/core/src/types.cpp#L77
    pub fn convert_to_points_masked(
        keypoints: &[KeyPoint],
        indices: &[i32],
    ) -> Result<Vec<Point2f>> {
        let mut points = Vec::with_capacity(indices.len());
        for &idx in indices {
            if idx < 0 {
                return Err(PureCvError::InvalidInput(
                    "keypoint index must be non-negative".to_string(),
                ));
            }
            let idx_usize = idx as usize;
            if idx_usize >= keypoints.len() {
                return Err(PureCvError::InvalidInput(
                    "keypoint index out of bounds".to_string(),
                ));
            }
            points.push(keypoints[idx_usize].pt);
        }
        Ok(points)
    }

    /// Converts a list of Point2f coordinates into KeyPoints.
    ///
    /// Matches the exact OpenCV C++ logic (where angle defaults to -1.0):
    /// Ref: https://github.com/opencv/opencv/blob/4.10.0/modules/core/src/types.cpp#L93
    pub fn convert_from_points(
        points2f: &[Point2f],
        size: f32,
        response: f32,
        octave: i32,
        class_id: i32,
    ) -> Vec<KeyPoint> {
        points2f
            .iter()
            .map(|pt| KeyPoint::new(*pt, size, -1.0, response, octave, class_id))
            .collect()
    }

    /// Sorts a slice of keypoints by their response in-place.
    ///
    /// * `keypoints` - The mutable slice of keypoints.
    /// * `descending` - If true, strongest keypoints (highest response) will be placed first.
    pub fn sort_by_response(keypoints: &mut [KeyPoint], descending: bool) {
        keypoints.sort_by(|a, b| {
            let ord = a
                .response
                .partial_cmp(&b.response)
                .unwrap_or(std::cmp::Ordering::Equal);
            if descending {
                ord.reverse()
            } else {
                ord
            }
        });
    }

    /// Retains only the best N keypoints based on response strength.
    pub fn retain_best(keypoints: &mut Vec<KeyPoint>, n: usize) {
        if keypoints.len() <= n {
            return;
        }
        Self::sort_by_response(keypoints, true);
        keypoints.truncate(n);
    }
}

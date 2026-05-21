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
}

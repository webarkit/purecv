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
use crate::core::error::Result;
use crate::core::Matrix;

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
    pub fn detect(&self, _image: &Matrix<u8>) -> Result<Vec<KeyPoint>> {
        // TODO: Implement FAST keypoint detection algorithm
        todo!()
    }
}

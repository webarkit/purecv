/*
 *  orb.rs
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

/// The type of keypoint scoring for ORB.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScoreType {
    /// Use Harris corner score (OpenCV default).
    Harris,
    /// Use FAST corner score (faster, but slightly less robust).
    Fast,
}

/// Oriented FAST and Rotated BRIEF (ORB) keypoint detector and descriptor extractor.
///
/// Ref: https://docs.opencv.org/4.10.0/db/d95/classcv_1_1ORB.html
#[derive(Debug, Clone)]
pub struct Orb {
    nfeatures: usize,
    scale_factor: f32,
    nlevels: usize,
    edge_threshold: i32,
    first_level: usize,
    wta_k: usize,
    score_type: ScoreType,
    patch_size: usize,
    fast_threshold: u8,
}

impl Default for Orb {
    /// Creates an ORB instance with default OpenCV parameters.
    fn default() -> Self {
        Self {
            nfeatures: 500,
            scale_factor: 1.2,
            nlevels: 8,
            edge_threshold: 31,
            first_level: 0,
            wta_k: 2,
            score_type: ScoreType::Harris,
            patch_size: 31,
            fast_threshold: 20,
        }
    }
}

impl Orb {
    /// Creates a new ORB instance with customizable parameters.
    ///
    /// * `nfeatures` - The maximum number of features to retain.
    /// * `scale_factor` - Pyramid decimation ratio, greater than 1.
    /// * `nlevels` - The number of pyramid levels.
    /// * `edge_threshold` - This is size of the border where the features are not detected.
    /// * `first_level` - The level of pyramid to put source image to.
    /// * `wta_k` - The number of points that produce each element of the oriented BRIEF descriptor.
    /// * `score_type` - The algorithm used to rank the features.
    /// * `patch_size` - Size of the patch used by the oriented BRIEF descriptor.
    /// * `fast_threshold` - The FAST threshold.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        nfeatures: usize,
        scale_factor: f32,
        nlevels: usize,
        edge_threshold: i32,
        first_level: usize,
        wta_k: usize,
        score_type: ScoreType,
        patch_size: usize,
        fast_threshold: u8,
    ) -> Self {
        Self {
            nfeatures,
            scale_factor,
            nlevels,
            edge_threshold,
            first_level,
            wta_k,
            score_type,
            patch_size,
            fast_threshold,
        }
    }

    /// Gets the maximum number of features to retain.
    pub fn nfeatures(&self) -> usize {
        self.nfeatures
    }

    /// Sets the maximum number of features to retain.
    pub fn set_nfeatures(&mut self, nfeatures: usize) {
        self.nfeatures = nfeatures;
    }

    /// Gets the pyramid decimation ratio.
    pub fn scale_factor(&self) -> f32 {
        self.scale_factor
    }

    /// Sets the pyramid decimation ratio.
    pub fn set_scale_factor(&mut self, scale_factor: f32) {
        self.scale_factor = scale_factor;
    }

    /// Gets the number of pyramid levels.
    pub fn nlevels(&self) -> usize {
        self.nlevels
    }

    /// Sets the number of pyramid levels.
    pub fn set_nlevels(&mut self, nlevels: usize) {
        self.nlevels = nlevels;
    }

    /// Gets the edge threshold.
    pub fn edge_threshold(&self) -> i32 {
        self.edge_threshold
    }

    /// Sets the edge threshold.
    pub fn set_edge_threshold(&mut self, edge_threshold: i32) {
        self.edge_threshold = edge_threshold;
    }

    /// Gets the first level of the pyramid.
    pub fn first_level(&self) -> usize {
        self.first_level
    }

    /// Sets the first level of the pyramid.
    pub fn set_first_level(&mut self, first_level: usize) {
        self.first_level = first_level;
    }

    /// Gets WTA_K parameter.
    pub fn wta_k(&self) -> usize {
        self.wta_k
    }

    /// Sets WTA_K parameter.
    pub fn set_wta_k(&mut self, wta_k: usize) {
        self.wta_k = wta_k;
    }

    /// Gets the scoring type used.
    pub fn score_type(&self) -> ScoreType {
        self.score_type
    }

    /// Sets the scoring type used.
    pub fn set_score_type(&mut self, score_type: ScoreType) {
        self.score_type = score_type;
    }

    /// Gets the patch size.
    pub fn patch_size(&self) -> usize {
        self.patch_size
    }

    /// Sets the patch size.
    pub fn set_patch_size(&mut self, patch_size: usize) {
        self.patch_size = patch_size;
    }

    /// Gets the FAST threshold.
    pub fn fast_threshold(&self) -> u8 {
        self.fast_threshold
    }

    /// Sets the FAST threshold.
    pub fn set_fast_threshold(&mut self, fast_threshold: u8) {
        self.fast_threshold = fast_threshold;
    }

    /// Detects keypoints in an image.
    ///
    /// * `image` - Grayscale input image (matrix).
    pub fn detect(&self, _image: &Matrix<u8>) -> Result<Vec<KeyPoint>> {
        // TODO: Implement ORB keypoint detection algorithm
        todo!()
    }

    /// Computes keypoint descriptors.
    ///
    /// * `image` - Grayscale input image (matrix).
    /// * `keypoints` - Detected keypoints for which to compute descriptors.
    pub fn compute(&self, _image: &Matrix<u8>, _keypoints: &[KeyPoint]) -> Result<Matrix<u8>> {
        // TODO: Implement ORB descriptor extraction algorithm
        todo!()
    }

    /// Detects keypoints and computes their descriptors in one pass.
    ///
    /// * `image` - Grayscale input image (matrix).
    pub fn detect_and_compute(&self, image: &Matrix<u8>) -> Result<(Vec<KeyPoint>, Matrix<u8>)> {
        let keypoints = self.detect(image)?;
        let descriptors = self.compute(image, &keypoints)?;
        Ok((keypoints, descriptors))
    }
}

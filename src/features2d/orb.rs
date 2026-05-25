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

use super::bit_pattern_31::BIT_PATTERN_31;
use super::fast::{FastFeatureDetector, FastType};
use super::keypoint::KeyPoint;
use crate::core::error::{PureCvError, Result};
use crate::core::types::Size;
use crate::core::Matrix;
use crate::imgproc::resize;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

#[cfg(feature = "simd")]
use pulp::Arch;

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

    /// Distributes desired keypoints geometrically across scale pyramid levels.
    fn get_features_per_level(&self) -> Vec<usize> {
        let mut nfeatures_per_level = vec![0; self.nlevels];
        let mut sum_features = 0;
        let factor = 1.0 / self.scale_factor;
        let ndesired_first = self.nfeatures as f64 * (1.0 - factor as f64)
            / (1.0 - (factor as f64).powi(self.nlevels as i32));
        let mut ndesired = ndesired_first;
        for item in nfeatures_per_level.iter_mut().take(self.nlevels - 1) {
            *item = ndesired.round() as usize;
            sum_features += *item;
            ndesired *= factor as f64;
        }
        nfeatures_per_level[self.nlevels - 1] =
            (self.nfeatures as isize - sum_features as isize).max(0) as usize;
        nfeatures_per_level
    }

    /// Detects keypoints in an image.
    ///
    /// * `image` - Grayscale input image (matrix).
    pub fn detect(&self, image: &Matrix<u8>) -> Result<Vec<KeyPoint>> {
        if image.channels != 1 {
            return Err(PureCvError::InvalidInput(
                "ORB keypoint detection requires a single-channel grayscale image".to_string(),
            ));
        }

        let pyramid = build_orb_pyramid(image, self.nlevels, self.scale_factor)?;
        let nfeatures_per_level = self.get_features_per_level();
        let u_max = precompute_umax(self.patch_size / 2);

        let mut all_keypoints = Vec::new();

        for level in 0..self.nlevels {
            let fast_detector =
                FastFeatureDetector::new(self.fast_threshold, true, FastType::Type9_16);

            let mut level_kpts = fast_detector.detect(&pyramid[level])?;

            // Filter boundary keypoints
            let border = self.edge_threshold as f32;
            let cols = pyramid[level].cols as f32;
            let rows = pyramid[level].rows as f32;
            level_kpts.retain(|kp| {
                kp.pt.x >= border
                    && kp.pt.x < cols - border
                    && kp.pt.y >= border
                    && kp.pt.y < rows - border
            });

            // Harris corner response scoring if requested
            if self.score_type == ScoreType::Harris {
                let harris = crate::imgproc::corner_harris(
                    &pyramid[level],
                    3,
                    3,
                    0.04,
                    crate::core::types::BorderTypes::Reflect101,
                )?;
                for kp in level_kpts.iter_mut() {
                    let rx = kp.pt.x.round() as usize;
                    let ry = kp.pt.y.round() as usize;
                    kp.response = harris.get(ry, rx, 0).copied().unwrap_or(0.0);
                }
            }

            // Sort level keypoints descending by response
            level_kpts.sort_by(|a, b| {
                b.response
                    .partial_cmp(&a.response)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

            // Retain the best target features for this level
            if level_kpts.len() > nfeatures_per_level[level] {
                level_kpts.truncate(nfeatures_per_level[level]);
            }

            // Assign intensity centroid orientation and scale up coordinates
            let scale = self.scale_factor.powi(level as i32);
            for kp in level_kpts.iter_mut() {
                let angle = compute_orientation(
                    &pyramid[level],
                    kp.pt.x.round() as usize,
                    kp.pt.y.round() as usize,
                    self.patch_size,
                    &u_max,
                )?;
                kp.angle = angle;
                kp.pt.x *= scale;
                kp.pt.y *= scale;
                kp.octave = level as i32;
            }

            all_keypoints.extend(level_kpts);
        }

        // Sort overall keypoints descending by response and keep best self.nfeatures
        all_keypoints.sort_by(|a, b| {
            b.response
                .partial_cmp(&a.response)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        if all_keypoints.len() > self.nfeatures {
            all_keypoints.truncate(self.nfeatures);
        }

        Ok(all_keypoints)
    }

    /// Computes keypoint descriptors.
    ///
    /// * `image` - Grayscale input image (matrix).
    /// * `keypoints` - Detected keypoints for which to compute descriptors.
    pub fn compute(&self, image: &Matrix<u8>, keypoints: &[KeyPoint]) -> Result<Matrix<u8>> {
        if image.channels != 1 {
            return Err(PureCvError::InvalidInput(
                "ORB descriptor extraction requires a single-channel grayscale image".to_string(),
            ));
        }

        let pyramid = build_orb_pyramid(image, self.nlevels, self.scale_factor)?;
        let mut descriptors = Matrix::<u8>::new(keypoints.len(), 32, 1);

        #[cfg(feature = "parallel")]
        {
            descriptors
                .data
                .par_chunks_exact_mut(32)
                .enumerate()
                .try_for_each(|(i, row_desc)| -> Result<()> {
                    let kp = &keypoints[i];
                    let level = kp.octave;
                    if level < 0 || level as usize >= self.nlevels {
                        return Err(PureCvError::InvalidInput(format!(
                            "Keypoint octave {level} is larger than ORB nlevels {}",
                            self.nlevels
                        )));
                    }
                    let scale = self.scale_factor.powi(level);
                    let mut level_kp = kp.clone();
                    level_kp.pt.x /= scale;
                    level_kp.pt.y /= scale;

                    let desc_bytes = compute_orb_descriptor(
                        &pyramid[level as usize],
                        &level_kp,
                        self.patch_size,
                        &BIT_PATTERN_31,
                    )?;
                    row_desc.copy_from_slice(&desc_bytes);
                    Ok(())
                })?;
        }

        #[cfg(not(feature = "parallel"))]
        {
            descriptors
                .data
                .chunks_exact_mut(32)
                .enumerate()
                .try_for_each(|(i, row_desc)| -> Result<()> {
                    let kp = &keypoints[i];
                    let level = kp.octave;
                    if level < 0 || level as usize >= self.nlevels {
                        return Err(PureCvError::InvalidInput(format!(
                            "Keypoint octave {level} is larger than ORB nlevels {}",
                            self.nlevels
                        )));
                    }
                    let scale = self.scale_factor.powi(level);
                    let mut level_kp = kp.clone();
                    level_kp.pt.x /= scale;
                    level_kp.pt.y /= scale;

                    let desc_bytes = compute_orb_descriptor(
                        &pyramid[level as usize],
                        &level_kp,
                        self.patch_size,
                        &BIT_PATTERN_31,
                    )?;
                    row_desc.copy_from_slice(&desc_bytes);
                    Ok(())
                })?;
        }

        Ok(descriptors)
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

/// Constructs a scale pyramid of grayscale images for ORB keypoint detection.
///
/// * `image` - Grayscale input image.
/// * `nlevels` - The number of pyramid levels.
/// * `scale_factor` - Pyramid decimation ratio, greater than 1.
pub fn build_orb_pyramid(
    image: &Matrix<u8>,
    nlevels: usize,
    scale_factor: f32,
) -> Result<Vec<Matrix<u8>>> {
    if image.channels != 1 {
        return Err(PureCvError::InvalidInput(
            "ORB pyramid construction requires a single-channel grayscale image".to_string(),
        ));
    }
    if nlevels == 0 {
        return Err(PureCvError::InvalidInput(
            "nlevels must be greater than 0".to_string(),
        ));
    }
    if scale_factor <= 1.0 {
        return Err(PureCvError::InvalidInput(
            "scale_factor must be greater than 1.0".to_string(),
        ));
    }

    let mut pyramid = Vec::with_capacity(nlevels);
    pyramid.push(image.clone());

    for level in 1..nlevels {
        let level_scale = scale_factor.powi(level as i32);
        let inv_scale = 1.0 / level_scale;

        let cols_level = (image.cols as f32 * inv_scale).round() as usize;
        let rows_level = (image.rows as f32 * inv_scale).round() as usize;

        if cols_level == 0 || rows_level == 0 {
            return Err(PureCvError::InvalidInput(format!(
                "Image dimension has shrunk to 0 at pyramid level {level}. nlevels={nlevels} is too large for image size {}x{}",
                image.cols, image.rows
            )));
        }

        let prev = &pyramid[level - 1];
        let next = resize(prev, Size::new(cols_level, rows_level))?;
        pyramid.push(next);
    }

    Ok(pyramid)
}

/// Precomputes the row end coordinates for a circular patch of a given half size.
///
/// Ref: https://github.com/opencv/opencv/blob/4.10.0/modules/features2d/src/orb.cpp#L862
pub fn precompute_umax(half_patch_size: usize) -> Vec<i32> {
    let mut umax = vec![0; half_patch_size + 2];
    let vmax = ((half_patch_size as f64) * 2.0_f64.sqrt() / 2.0 + 1.0).floor() as usize;
    let vmin = ((half_patch_size as f64) * 2.0_f64.sqrt() / 2.0).ceil() as usize;

    for (v, val) in umax.iter_mut().enumerate().take(vmax + 1) {
        *val = ((half_patch_size * half_patch_size - v * v) as f64)
            .sqrt()
            .round() as i32;
    }

    let mut v0 = 0;
    for v in (vmin..=half_patch_size).rev() {
        while umax[v0] == umax[v0 + 1] {
            v0 += 1;
        }
        umax[v] = v0 as i32;
        v0 += 1;
    }

    umax
}

/// Computes the orientation (in degrees) of a keypoint's neighborhood using the Intensity Centroid method.
///
/// Ref: https://github.com/opencv/opencv/blob/4.10.0/modules/features2d/src/orb.cpp#L181
pub fn compute_orientation(
    image: &Matrix<u8>,
    x: usize,
    y: usize,
    patch_size: usize,
    u_max: &[i32],
) -> Result<f32> {
    if image.channels != 1 {
        return Err(PureCvError::InvalidInput(
            "Intensity Centroid requires a single-channel grayscale image".to_string(),
        ));
    }

    let half_k = (patch_size / 2) as i32;
    let rows = image.rows as i32;
    let cols = image.cols as i32;

    let cx = x as i32;
    let cy = y as i32;

    if cx - half_k < 0 || cx + half_k >= cols || cy - half_k < 0 || cy + half_k >= rows {
        return Err(PureCvError::InvalidInput(format!(
            "Keypoint coordinate ({x}, {y}) circular patch of size {patch_size} is out of image boundaries"
        )));
    }

    let mut m_10 = 0i64;
    let mut m_01 = 0i64;

    let data = &image.data;
    let center_offset = (cy * cols + cx) as usize;

    #[cfg(feature = "simd")]
    {
        let arch = Arch::new();
        arch.dispatch(|| {
            let mut local_m_10 = 0i64;
            let mut local_m_01 = 0i64;

            // Center row (v = 0)
            let center_start = (center_offset as i32 - half_k) as usize;
            let center_len = (2 * half_k + 1) as usize;
            let center_slice = &data[center_start..center_start + center_len];
            for (i, &val) in center_slice.iter().enumerate() {
                let u = i as i32 - half_k;
                local_m_10 += u as i64 * val as i64;
            }

            // Go line by line in the circular patch (v >= 1)
            for v in 1..=half_k {
                let mut v_sum = 0i64;
                let d = u_max[v as usize];

                let row_plus_start = (center_offset as i32 - d + v * cols) as usize;
                let row_minus_start = (center_offset as i32 - d - v * cols) as usize;
                let len = (2 * d + 1) as usize;

                let slice_plus = &data[row_plus_start..row_plus_start + len];
                let slice_minus = &data[row_minus_start..row_minus_start + len];

                for (i, (&val_plus, &val_minus)) in
                    slice_plus.iter().zip(slice_minus.iter()).enumerate()
                {
                    let u = i as i32 - d;
                    let vp = val_plus as i64;
                    let vm = val_minus as i64;

                    v_sum += vp - vm;
                    local_m_10 += u as i64 * (vp + vm);
                }
                local_m_01 += v as i64 * v_sum;
            }

            m_10 = local_m_10;
            m_01 = local_m_01;
        });
    }

    #[cfg(not(feature = "simd"))]
    {
        // Center row (v = 0)
        for u in -half_k..=half_k {
            let val = data[(center_offset as i32 + u) as usize] as i64;
            m_10 += u as i64 * val;
        }

        // Go line by line in the circular patch (v >= 1)
        for v in 1..=half_k {
            let mut v_sum = 0i64;
            let d = u_max[v as usize];
            for u in -d..=d {
                let offset_plus = (center_offset as i32 + u + v * cols) as usize;
                let offset_minus = (center_offset as i32 + u - v * cols) as usize;
                let val_plus = data[offset_plus] as i64;
                let val_minus = data[offset_minus] as i64;

                v_sum += val_plus - val_minus;
                m_10 += u as i64 * (val_plus + val_minus);
            }
            m_01 += v as i64 * v_sum;
        }
    }

    let mut angle = (m_01 as f32).atan2(m_10 as f32) * 180.0 / std::f32::consts::PI;
    if angle < 0.0 {
        angle += 360.0;
    }

    Ok(angle)
}

/// Computes the 32-byte steered BRIEF descriptor for a keypoint on a specific image.
///
/// Ref: https://github.com/opencv/opencv/blob/4.10.0/modules/features2d/src/orb.cpp#L220
pub fn compute_orb_descriptor(
    image: &Matrix<u8>,
    keypoint: &KeyPoint,
    _patch_size: usize,
    pattern: &[i8],
) -> Result<[u8; 32]> {
    if image.channels != 1 {
        return Err(PureCvError::InvalidInput(
            "Steered BRIEF requires a single-channel grayscale image".to_string(),
        ));
    }

    let angle_rad = keypoint.angle * std::f32::consts::PI / 180.0;
    let cos_a = angle_rad.cos();
    let sin_a = angle_rad.sin();

    let cx = keypoint.pt.x.round() as i32;
    let cy = keypoint.pt.y.round() as i32;

    let cols = image.cols as i32;
    let rows = image.rows as i32;
    let data = &image.data;

    let get_pixel = |px: i32, py: i32| -> u8 {
        let clamped_x = px.clamp(0, cols - 1) as usize;
        let clamped_y = py.clamp(0, rows - 1) as usize;
        data[clamped_y * cols as usize + clamped_x]
    };

    let mut desc = [0u8; 32];

    #[cfg(feature = "simd")]
    {
        let arch = Arch::new();
        arch.dispatch(|| {
            for (i, val) in desc.iter_mut().enumerate() {
                let mut byte_val = 0u8;
                for bit in 0..8 {
                    let idx = (i * 8 + bit) * 4;

                    let x1 = pattern[idx] as f32;
                    let y1 = pattern[idx + 1] as f32;
                    let x2 = pattern[idx + 2] as f32;
                    let y2 = pattern[idx + 3] as f32;

                    let rx1 = (x1 * cos_a - y1 * sin_a).round() as i32;
                    let ry1 = (x1 * sin_a + y1 * cos_a).round() as i32;

                    let rx2 = (x2 * cos_a - y2 * sin_a).round() as i32;
                    let ry2 = (x2 * sin_a + y2 * cos_a).round() as i32;

                    let val1 = get_pixel(cx + rx1, cy + ry1);
                    let val2 = get_pixel(cx + rx2, cy + ry2);

                    if val1 < val2 {
                        byte_val |= 1 << bit;
                    }
                }
                *val = byte_val;
            }
        });
    }

    #[cfg(not(feature = "simd"))]
    {
        for (i, val) in desc.iter_mut().enumerate() {
            let mut byte_val = 0u8;
            for bit in 0..8 {
                let idx = (i * 8 + bit) * 4;

                let x1 = pattern[idx] as f32;
                let y1 = pattern[idx + 1] as f32;
                let x2 = pattern[idx + 2] as f32;
                let y2 = pattern[idx + 3] as f32;

                let rx1 = (x1 * cos_a - y1 * sin_a).round() as i32;
                let ry1 = (x1 * sin_a + y1 * cos_a).round() as i32;

                let rx2 = (x2 * cos_a - y2 * sin_a).round() as i32;
                let ry2 = (x2 * sin_a + y2 * cos_a).round() as i32;

                let val1 = get_pixel(cx + rx1, cy + ry1);
                let val2 = get_pixel(cx + rx2, cy + ry2);

                if val1 < val2 {
                    byte_val |= 1 << bit;
                }
            }
            *val = byte_val;
        }
    }

    Ok(desc)
}

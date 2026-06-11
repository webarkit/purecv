/*
 *  mod.rs
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

//! # 2D Features Framework (features2d)
//!
//! The `features2d` module provides high-performance, pure Rust implementations of keypoint
//! detectors and descriptor extractors, matching OpenCV's feature tracking architecture.
//!
//! Features from Accelerated Segment Test (FAST) and Oriented FAST and Rotated BRIEF (ORB) are
//! designed from the ground up for real-time applications such as SLAM, visual odometry, and object
//! recognition.
//!
//! ## 🚀 Usage Examples
//!
//! ### 1. FAST Corner Detection
//!
//! ```rust
//! # use purecv::core::Matrix;
//! # use purecv::features2d::{FastFeatureDetector, FastType, KeyPoint};
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // Create a single-channel grayscale matrix (e.g. 480x640)
//! let image = Matrix::<u8>::ones(480, 640, 1);
//!
//! // Instantiate FAST corner detector (threshold = 20, nonmax = true, Type9_16)
//! let detector = FastFeatureDetector::new(20, true, FastType::Type9_16);
//!
//! // Detect corners
//! let keypoints: Vec<KeyPoint> = detector.detect(&image)?;
//! println!("Detected {} corners", keypoints.len());
//! # Ok(())
//! # }
//! ```
//!
//! ### 2. ORB Feature Extraction (Keypoints and Descriptors)
//!
//! ```rust
//! # use purecv::core::Matrix;
//! # use purecv::features2d::{Orb, KeyPoint};
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // Create a grayscale matrix (e.g. 480x640)
//! let image = Matrix::<u8>::ones(480, 640, 1);
//!
//! // Instantiate ORB extractor with default parameters
//! let orb = Orb::default();
//!
//! // Compute keypoints and binary BRIEF descriptors
//! let (keypoints, descriptors): (Vec<KeyPoint>, Matrix<u8>) = orb.detect_and_compute(&image)?;
//!
//! println!("Extracted {} ORB keypoints", keypoints.len());
//! println!("Descriptors shape: {}x{}", descriptors.rows, descriptors.cols);
//! # Ok(())
//! # }
//! ```
//!
//! ## 📚 Reference & Standards
//!
//! - **OpenCV Parity**: Matches the structures and parameter ranges of OpenCV's `cv::Feature2D`, `cv::FastFeatureDetector`, and `cv::ORB`.
//! - **Reference Source**: [OpenCV features2d module group](https://docs.opencv.org/4.10.0/d5/d51/group__features2d__main.html)

pub mod bit_pattern_31;
pub mod draw;
pub mod fast;
pub mod keypoint;
pub mod matcher;
pub mod orb;

pub use bit_pattern_31::BIT_PATTERN_31;
pub use draw::{draw_keypoints, draw_matches};
pub use fast::{FastFeatureDetector, FastType};
pub use keypoint::KeyPoint;
pub use matcher::{filter_matches, BFMatcher, DMatch, DescriptorMatcher, NormType};
pub use orb::{
    build_orb_pyramid, compute_orb_descriptor, compute_orientation, precompute_umax, Orb, ScoreType,
};

#[cfg(test)]
mod tests;

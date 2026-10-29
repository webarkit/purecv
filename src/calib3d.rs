/*
 *  calib3d.rs
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

//! Camera calibration and 3-D geometry — the `calib3d` module.
//!
//! This module mirrors the structure of [OpenCV's `calib3d` module][ocv] and
//! implements the following **Milestone 4** APIs in pure Rust:
//!
//! | Function | Description |
//! |----------|-------------|
//! | [`find_homography`] | Compute a perspective homography from point correspondences |
//! | [`rodrigues`]       | Convert between rotation vector and rotation matrix |
//! | [`solve_pnp`]       | Estimate camera pose from 3-D / 2-D correspondences |
//! | [`solve_pnp_ransac`]| Robust pose estimation with RANSAC |
//!
//! # Conventions
//!
//! All functions follow the standard OpenCV coordinate convention (right-hand,
//! z-axis pointing away from the camera) and accept `f64` matrices
//! (`Matrix<f64>`) or point-slice inputs (`&[Point2f]`, `&[Point3f]`).
//!
//! [ocv]: https://docs.opencv.org/4.10.0/d9/d0c/group__calib3d.html

pub mod fundamental;
pub mod geometry;
pub mod homography;
pub mod levmarq;
pub(crate) mod linalg;
pub mod pose;
pub mod undistort;

#[cfg(feature = "simd")]
pub(crate) mod simd;

#[cfg(test)]
mod tests;

// ---------------------------------------------------------------------------
// Re-exports
// ---------------------------------------------------------------------------

pub use fundamental::{find_fundamental_mat, FundamentalMatMethod};
pub use geometry::rodrigues;
pub use homography::{find_homography, HomographyMethod};
pub use levmarq::{LmCallback, LmSolver};
pub use pose::{solve_pnp, solve_pnp_ransac, SolvePnPMethod};
pub use undistort::init_undistort_rectify_map;

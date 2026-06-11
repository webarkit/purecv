/*
 *  lib.rs
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

// Global modules
pub mod calib3d;
pub mod core;
pub mod features;
pub mod features2d;
pub mod imgproc;
pub mod version;
pub mod video;

/// Prelude to easily import common structures
pub mod prelude {
    pub use crate::calib3d::{
        find_homography, rodrigues, solve_pnp, solve_pnp_ransac, HomographyMethod, SolvePnPMethod,
    };
    pub use crate::core::types::{
        BorderTypes, Point2f, Point2i, Point3f, Rect2f, Rect2i, Scalar, Size2f, Size2i,
        TermCriteria, TermType, Vec2b, Vec2d, Vec2f, Vec2i, Vec2s, Vec3b, Vec3d, Vec3f, Vec3i,
        Vec3s, Vec4b, Vec4d, Vec4f, Vec4i, Vec4s, Vec6d, Vec6f, VecN,
    };
    pub use crate::core::Matrix;
    pub use crate::features2d::{
        draw_keypoints, draw_matches, filter_matches, BFMatcher, DMatch, DescriptorMatcher,
        FastFeatureDetector, FastType, KeyPoint, NormType, Orb,
    };
    pub use crate::imgproc::derivatives::{laplacian, scharr, sobel};
    pub use crate::imgproc::edge::canny;
    pub use crate::imgproc::feature::{
        corner_eigen_vals_and_vecs, corner_harris, corner_min_eigen_val, corner_sub_pix,
        good_features_to_track, pre_corner_detect,
    };
    pub use crate::imgproc::filter::{bilateral_filter, box_filter, gaussian_blur};
    pub use crate::imgproc::threshold::{threshold, ThresholdTypes};
    pub use crate::imgproc::{cvt_color, ColorConversionCode};
    pub use crate::video::optical_flow::{
        build_optical_flow_pyramid, calc_optical_flow_pyramid_lk, OpticalFlowPyramid,
        OPTFLOW_LK_GET_MIN_EIGENVALS, OPTFLOW_USE_INITIAL_FLOW,
    };
}

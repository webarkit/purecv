/*
 *  tests.rs
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

#[cfg(test)]
mod video_tests {
    use crate::core::types::{BorderTypes, Point2f, Size2i, TermCriteria, TermType};
    use crate::core::Matrix;
    use crate::video::optical_flow::{
        build_optical_flow_pyramid, calc_optical_flow_pyramid_lk, OPTFLOW_LK_GET_MIN_EIGENVALS,
        OPTFLOW_USE_INITIAL_FLOW,
    };

    // ------------------------------------------------------------------
    // build_optical_flow_pyramid tests
    // ------------------------------------------------------------------

    #[test]
    fn test_build_pyramid_level_count() {
        let img = Matrix::<u8>::new(64, 64, 1);
        let pyr = build_optical_flow_pyramid(
            &img,
            Size2i::new(5, 5),
            3,
            false,
            BorderTypes::Reflect101,
            BorderTypes::Constant,
        )
        .unwrap();
        // 4 levels: 64×64, 32×32, 16×16, 8×8
        assert_eq!(pyr.levels.len(), 4);
        assert!(pyr.dx.is_empty());
        assert!(pyr.dy.is_empty());
    }

    #[test]
    fn test_build_pyramid_sizes() {
        let img = Matrix::<u8>::new(64, 64, 1);
        let pyr = build_optical_flow_pyramid(
            &img,
            Size2i::new(5, 5),
            3,
            false,
            BorderTypes::Reflect101,
            BorderTypes::Constant,
        )
        .unwrap();
        assert_eq!(pyr.levels[0].rows, 64);
        assert_eq!(pyr.levels[0].cols, 64);
        assert_eq!(pyr.levels[1].rows, 32);
        assert_eq!(pyr.levels[2].rows, 16);
        assert_eq!(pyr.levels[3].rows, 8);
    }

    // miri: ~45s under interpretation. The Sobel `unsafe` fast path it exercises
    // is still covered by imgproc::tests::test_sobel (f32/ksize 3, ~0.8s under
    // Miri), so no UB coverage is lost here. See .agents/MIRI_PLAN.md §4.
    #[cfg_attr(miri, ignore)]
    #[test]
    fn test_build_pyramid_with_derivatives() {
        let img = Matrix::<u8>::new(64, 64, 1);
        let pyr = build_optical_flow_pyramid(
            &img,
            Size2i::new(5, 5),
            2,
            true,
            BorderTypes::Reflect101,
            BorderTypes::Reflect101,
        )
        .unwrap();
        assert_eq!(pyr.levels.len(), 3); // levels 0, 1, 2
        assert_eq!(pyr.dx.len(), 3);
        assert_eq!(pyr.dy.len(), 3);
        // Derivatives must have the same size as their corresponding level.
        for l in 0..3 {
            assert_eq!(pyr.dx[l].rows, pyr.levels[l].rows);
            assert_eq!(pyr.dx[l].cols, pyr.levels[l].cols);
        }
    }

    #[test]
    fn test_build_pyramid_rejects_multi_channel() {
        let img = Matrix::<u8>::new(64, 64, 3);
        let result = build_optical_flow_pyramid(
            &img,
            Size2i::new(21, 21),
            3,
            false,
            BorderTypes::Reflect101,
            BorderTypes::Constant,
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_build_pyramid_stops_early_when_too_small() {
        // With win_size=33, the 16×16 level is smaller than the window, so the
        // pyramid should stop before reaching max_level=4.
        let img = Matrix::<u8>::new(64, 64, 1);
        let pyr = build_optical_flow_pyramid(
            &img,
            Size2i::new(33, 33),
            4,
            false,
            BorderTypes::Reflect101,
            BorderTypes::Constant,
        )
        .unwrap();
        // Level 0 = 64×64 (≥33 — OK)
        // Level 1 = 32×32 (≥33? No — stop after this level actually)
        // In practice the guard checks *before* downsampling from the current level,
        // so level 1 = 32×32 should not be added.
        assert!(pyr.levels.len() <= 4);
    }

    // ------------------------------------------------------------------
    // calc_optical_flow_pyramid_lk tests
    // ------------------------------------------------------------------

    #[test]
    fn test_lk_empty_points() {
        let prev = Matrix::<u8>::new(64, 64, 1);
        let next = Matrix::<u8>::new(64, 64, 1);
        let criteria = TermCriteria::new(TermType::Both, 20, 0.03);

        let (pts, status, err) = calc_optical_flow_pyramid_lk(
            &prev,
            &next,
            &[],
            None,
            Size2i::new(21, 21),
            3,
            criteria,
            0,
            1e-4,
        )
        .unwrap();
        assert!(pts.is_empty());
        assert!(status.is_empty());
        assert!(err.is_empty());
    }

    #[test]
    fn test_lk_rejects_multi_channel() {
        let prev = Matrix::<u8>::new(64, 64, 3);
        let next = Matrix::<u8>::new(64, 64, 3);
        let criteria = TermCriteria::new(TermType::Both, 20, 0.03);
        let pts = vec![Point2f::new(32.0, 32.0)];

        let result = calc_optical_flow_pyramid_lk(
            &prev,
            &next,
            &pts,
            None,
            Size2i::new(21, 21),
            3,
            criteria,
            0,
            1e-4,
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_lk_rejects_mismatched_dimensions() {
        let prev = Matrix::<u8>::new(64, 64, 1);
        let next = Matrix::<u8>::new(32, 32, 1);
        let criteria = TermCriteria::new(TermType::Both, 20, 0.03);
        let pts = vec![Point2f::new(10.0, 10.0)];

        let result = calc_optical_flow_pyramid_lk(
            &prev,
            &next,
            &pts,
            None,
            Size2i::new(21, 21),
            3,
            criteria,
            0,
            1e-4,
        );
        assert!(result.is_err());
    }

    /// Tracking a stationary point in two identical frames should return a
    /// flow vector close to zero and status = 1.
    // miri: Lucas-Kanade pyramidal iteration — ~62s under interpretation.
    // No `unsafe` on this path. See .agents/MIRI_PLAN.md §4.
    #[cfg_attr(miri, ignore)]
    #[test]
    fn test_lk_stationary_point_identical_frames() {
        // Create a 64×64 frame with a small bright blob so there are gradients.
        let mut data = vec![0u8; 64 * 64];
        // Draw a 5×5 white square at (28, 28) to (32, 32).
        for r in 28..33 {
            for c in 28..33 {
                data[r * 64 + c] = 200;
            }
        }
        let prev = Matrix::<u8>::from_vec(64, 64, 1, data.clone());
        let next = Matrix::<u8>::from_vec(64, 64, 1, data);

        let pts = vec![Point2f::new(30.0, 30.0)];
        let criteria = TermCriteria::new(TermType::Both, 20, 0.03);

        let (next_pts, status, _err) = calc_optical_flow_pyramid_lk(
            &prev,
            &next,
            &pts,
            None,
            Size2i::new(11, 11),
            2,
            criteria,
            0,
            1e-4,
        )
        .unwrap();

        // Point should be tracked successfully.
        assert_eq!(status[0], 1, "point should be tracked in identical frames");
        // The estimated position should be very close to the original.
        let dx = (next_pts[0].x - 30.0).abs();
        let dy = (next_pts[0].y - 30.0).abs();
        assert!(dx < 1.0, "unexpected x displacement: {dx}");
        assert!(dy < 1.0, "unexpected y displacement: {dy}");
    }

    /// Simulate a pure translation of +3 pixels in x by shifting the image.
    // miri: Lucas-Kanade pyramidal iteration — ~105s under interpretation.
    // No `unsafe` on this path. See .agents/MIRI_PLAN.md §4.
    #[cfg_attr(miri, ignore)]
    #[test]
    fn test_lk_pure_translation_x() {
        let rows = 64usize;
        let cols = 64usize;
        let shift = 3usize;

        // Gaussian blob — survives pyramid downsampling and provides good
        // gradients at all scales.
        let cx = 32.0f32;
        let cy = 32.0f32;
        let sigma = 8.0f32;
        let mut prev_data = vec![0u8; rows * cols];
        for r in 0..rows {
            for c in 0..cols {
                let dx = c as f32 - cx;
                let dy = r as f32 - cy;
                let val =
                    (200.0 * (-(dx * dx + dy * dy) / (2.0 * sigma * sigma)).exp()).round() as u8;
                prev_data[r * cols + c] = val;
            }
        }

        // next = prev shifted by +shift in x (replicate left border).
        let mut next_data = vec![0u8; rows * cols];
        for r in 0..rows {
            for c in 0..cols {
                let src_c = c.saturating_sub(shift);
                next_data[r * cols + c] = prev_data[r * cols + src_c];
            }
        }

        let prev = Matrix::<u8>::from_vec(rows, cols, 1, prev_data);
        let next = Matrix::<u8>::from_vec(rows, cols, 1, next_data);

        let pts = vec![Point2f::new(cx, cy)];
        let criteria = TermCriteria::new(TermType::Both, 30, 0.001);

        let (next_pts, status, _err) = calc_optical_flow_pyramid_lk(
            &prev,
            &next,
            &pts,
            None,
            Size2i::new(15, 15),
            2,
            criteria,
            0,
            1e-4,
        )
        .unwrap();

        assert_eq!(status[0], 1, "point should be tracked");
        let estimated_dx = next_pts[0].x - pts[0].x;
        // Allow ±1.5 pixels tolerance for this simple test.
        assert!(
            (estimated_dx - shift as f32).abs() < 1.5,
            "expected flow ~{shift}, got {estimated_dx:.2}"
        );
    }

    /// Test using the `OPTFLOW_LK_GET_MIN_EIGENVALS` flag.
    // miri: Lucas-Kanade pyramidal iteration — ~61s under interpretation.
    // No `unsafe` on this path. See .agents/MIRI_PLAN.md §4.
    #[cfg_attr(miri, ignore)]
    #[test]
    fn test_lk_min_eigenvals_flag() {
        let mut data = vec![0u8; 64 * 64];
        for r in 28..33 {
            for c in 28..33 {
                data[r * 64 + c] = 200;
            }
        }
        let img = Matrix::<u8>::from_vec(64, 64, 1, data);
        let pts = vec![Point2f::new(30.0, 30.0)];
        let criteria = TermCriteria::new(TermType::Both, 20, 0.03);

        let (_, status, eigenvals) = calc_optical_flow_pyramid_lk(
            &img,
            &img,
            &pts,
            None,
            Size2i::new(11, 11),
            2,
            criteria,
            OPTFLOW_LK_GET_MIN_EIGENVALS,
            1e-4,
        )
        .unwrap();

        assert_eq!(status[0], 1);
        // Eigenvalue should be a non-negative finite number for a tracked point.
        assert!(eigenvals[0].is_finite());
        assert!(eigenvals[0] >= 0.0);
    }

    /// Test the `OPTFLOW_USE_INITIAL_FLOW` flag with a good initial guess.
    // miri: Lucas-Kanade pyramidal iteration — ~62s under interpretation.
    // No `unsafe` on this path. See .agents/MIRI_PLAN.md §4.
    #[cfg_attr(miri, ignore)]
    #[test]
    fn test_lk_use_initial_flow() {
        let mut data = vec![0u8; 64 * 64];
        for r in 28..36 {
            for c in 28..36 {
                data[r * 64 + c] = 200;
            }
        }
        let prev = Matrix::<u8>::from_vec(64, 64, 1, data.clone());
        let next = Matrix::<u8>::from_vec(64, 64, 1, data);

        let pts = vec![Point2f::new(32.0, 32.0)];
        // Provide an initial guess equal to the true position (no motion).
        let initial = vec![Point2f::new(32.0, 32.0)];
        let criteria = TermCriteria::new(TermType::Both, 20, 0.03);

        let (next_pts, status, _) = calc_optical_flow_pyramid_lk(
            &prev,
            &next,
            &pts,
            Some(&initial),
            Size2i::new(11, 11),
            2,
            criteria,
            OPTFLOW_USE_INITIAL_FLOW,
            1e-4,
        )
        .unwrap();

        assert_eq!(status[0], 1);
        assert!((next_pts[0].x - 32.0).abs() < 1.0);
    }
}

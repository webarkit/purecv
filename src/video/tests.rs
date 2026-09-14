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
    use crate::imgproc::derivatives::scharr;
    use crate::video::optical_flow::{
        build_optical_flow_pyramid, calc_optical_flow_pyramid_lk, lk_iterate, FLT_SCALE,
        OPTFLOW_LK_GET_MIN_EIGENVALS, OPTFLOW_USE_INITIAL_FLOW,
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

    // miri: ~45s under interpretation. Post-#130 this exercises the Scharr
    // `unsafe` fast path (build_optical_flow_pyramid switched from Sobel to
    // Scharr), still covered by imgproc::tests::test_scharr (f32/ksize -1,
    // ~0.8s under Miri), so no UB coverage is lost here. See
    // .agents/MIRI_PLAN.md §4.
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

    // Not Miri-ignored: measured at ~0.8s under Miri (std,simd), well under
    // the >30s exclusion threshold in .agents/MIRI_PLAN.md §4, despite
    // exercising the same unsafe Scharr fast path as
    // test_build_pyramid_with_derivatives.
    #[test]
    fn test_build_pyramid_derivatives_use_scharr() {
        // 5x5 linear ramp v(x, y) = 2*x + y. For a linear ramp the 3x3
        // derivative response at any interior pixel has an exact closed
        // form: Ix = 2*a*sum(ky), Iy = 2*b*sum(ky), where sum(ky) is 4 for
        // Sobel's [1,2,1] smoothing kernel or 16 for Scharr's [3,10,3].
        // purecv#130: this must be 16 (Scharr), not 4 (Sobel).
        let a = 2.0f32;
        let b = 1.0f32;
        let mut data = vec![0u8; 5 * 5];
        for y in 0..5usize {
            for x in 0..5usize {
                data[y * 5 + x] = (a * x as f32 + b * y as f32) as u8;
            }
        }
        let img = Matrix::<u8>::from_vec(5, 5, 1, data);

        let pyr = build_optical_flow_pyramid(
            &img,
            Size2i::new(3, 3),
            0, // single level: pure derivative check, no pyr_down involved
            true,
            BorderTypes::Reflect101,
            BorderTypes::Reflect101,
        )
        .unwrap();

        // Center pixel (2, 2): full 3x3 neighborhood inside the image, so
        // border interpolation never kicks in and the closed form is exact.
        let idx = 2 * 5 + 2;
        let expected_ix = 32.0 * a; // Scharr: 2*a*16
        let expected_iy = 32.0 * b;

        assert!(
            (pyr.dx[0].data[idx] - expected_ix).abs() < 1e-4,
            "expected Ix = {expected_ix} (Scharr), got {}; build_optical_flow_pyramid \
             must use Scharr, not Sobel, derivatives",
            pyr.dx[0].data[idx]
        );
        assert!(
            (pyr.dy[0].data[idx] - expected_iy).abs() < 1e-4,
            "expected Iy = {expected_iy} (Scharr), got {}",
            pyr.dy[0].data[idx]
        );
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

    /// `calc_optical_flow_pyramid_lk`'s derivative computation is private,
    /// so pin it through the public interface: independently compute the
    /// documented H-matrix formula from a direct `scharr()` call, and check
    /// it against `err[0]` (min eigenvalue) reported via
    /// OPTFLOW_LK_GET_MIN_EIGENVALS. purecv#130: pins the Scharr operator;
    /// this test failed against the pre-fix Sobel implementation.
    // Not Miri-ignored: measured at ~18s under Miri (std,simd), under the
    // >30s exclusion threshold in .agents/MIRI_PLAN.md §4, despite
    // exercising the same unsafe Scharr fast path as
    // test_build_pyramid_with_derivatives.
    #[test]
    fn test_calc_optical_flow_pyramid_lk_uses_scharr_derivatives() {
        // Same textured frame as the module's doc example: an 8x8 bright
        // square gives a genuinely 2D gradient (non-degenerate H) at its
        // edges, unlike a flat region or a pure linear ramp.
        let mut data = vec![0u8; 64 * 64];
        for r in 28..36 {
            for c in 28..36 {
                data[r * 64 + c] = 200;
            }
        }
        let frame = Matrix::<u8>::from_vec(64, 64, 1, data);
        let pt = Point2f::new(32.0, 32.0);
        let win_size = Size2i::new(11, 11);
        let half_win_w = win_size.width / 2;
        let half_win_h = win_size.height / 2;

        // Reference: replicate the documented H-matrix formula using a
        // direct scharr() call — the operator OpenCV actually uses.
        let frame_f32 = frame.convert_to::<f32>().unwrap();
        let ix = scharr(&frame_f32, 1, 0, 1.0, 0.0, BorderTypes::Reflect101).unwrap();
        let iy = scharr(&frame_f32, 0, 1, 1.0, 0.0, BorderTypes::Reflect101).unwrap();

        let px = pt.x as i32;
        let py = pt.y as i32;
        let cols = ix.cols;
        // keep in sync with lk_single_level's H/eigenvalue computation
        // (src/video/optical_flow.rs, near the min_eigen_threshold handling
        // in the single-level LK solver) — this test intentionally
        // duplicates that private formula for black-box verification.
        let mut h00 = 0.0f64;
        let mut h01 = 0.0f64;
        let mut h11 = 0.0f64;
        for dy in -half_win_h..=half_win_h {
            for dx in -half_win_w..=half_win_w {
                let idx = ((py + dy) as usize) * cols + (px + dx) as usize;
                let vx = ix.data[idx] as f64;
                let vy = iy.data[idx] as f64;
                h00 += vx * vx;
                h01 += vx * vy;
                h11 += vy * vy;
            }
        }
        let win_area = ((2 * half_win_w + 1) * (2 * half_win_h + 1)) as f64;
        // purecv#138: H is scaled by OpenCV's FLT_SCALE = 2^-20 as well as by
        // the window area.
        let (h00n, h01n, h11n) = (
            h00 * FLT_SCALE / win_area,
            h01 * FLT_SCALE / win_area,
            h11 * FLT_SCALE / win_area,
        );
        let trace = h00n + h11n;
        let det_n = h00n * h11n - h01n * h01n;
        let disc = (trace * trace - 4.0 * det_n).max(0.0).sqrt();
        let expected_min_eigen = (trace - disc) * 0.5;

        // Actual: what calc_optical_flow_pyramid_lk reports.
        let criteria = TermCriteria::new(TermType::Both, 20, 0.03);
        let (_next_pts, status, err) = calc_optical_flow_pyramid_lk(
            &frame,
            &frame,
            &[pt],
            None,
            win_size,
            0, // max_level: single level keeps point coordinates unscaled
            criteria,
            OPTFLOW_LK_GET_MIN_EIGENVALS,
            0.0, // min_eigen_threshold: accept regardless of scale
        )
        .unwrap();

        assert_eq!(status[0], 1);
        debug_assert!(
            expected_min_eigen > 0.0,
            "test fixture must produce a non-degenerate H matrix"
        );
        let relative_error = (err[0] as f64 - expected_min_eigen).abs() / expected_min_eigen.abs();
        assert!(
            relative_error < 1e-5,
            "expected min_eigen {expected_min_eigen} (Scharr), got {} (relative error {relative_error}); \
             calc_optical_flow_pyramid_lk must use the same Scharr derivatives as scharr()",
            err[0]
        );
    }

    /// Pins the reported minimum eigenvalue to OpenCV's scale — `FLT_SCALE`
    /// (`2^-20`) with the window area as the only other divisor — against a
    /// closed-form value. purecv#138: the pre-fix code omitted `FLT_SCALE`
    /// entirely, and dividing by `2 * win_area` instead of `win_area` would
    /// halve the result, since `min_eigen` already applies the `* 0.5` of the
    /// eigenvalue formula.
    #[test]
    fn test_lk_min_eigen_matches_opencv_scale() {
        // Separable image v(x, y) = f(x) + g(y). For a separable image the
        // Scharr response is exact and depends on one axis only:
        //     Ix(x) = 16 * (f(x+1) - f(x-1)),  Iy(y) = 16 * (g(y+1) - g(y-1))
        // (16 = sum of Scharr's [3,10,3] smoothing kernel).
        // Over the 3x3 window at (32,32): Ix = 160 on the x=31 column only,
        // Iy = 160 on the y=33 row only, so
        //     h00 = h11 = 3*160^2 = 76800,  h01 = 160*160 = 25600
        // and, since h00n == h11n, min_eigen = h00n - |h01n| =
        //     (76800 - 25600) * 2^-20 / 9 = 51200 / 9437184
        let mut data = vec![0u8; 64 * 64];
        for y in 0..64usize {
            for x in 0..64usize {
                let f = if x == 32 || x == 34 { 10u16 } else { 0 };
                let g = if y == 34 { 10u16 } else { 0 };
                data[y * 64 + x] = (f + g) as u8;
            }
        }
        let frame = Matrix::<u8>::from_vec(64, 64, 1, data);
        let criteria = TermCriteria::new(TermType::Both, 20, 0.03);
        let (_next_pts, status, err) = calc_optical_flow_pyramid_lk(
            &frame,
            &frame,
            &[Point2f::new(32.0, 32.0)],
            None,
            Size2i::new(3, 3),
            0, // max_level: single level keeps point coordinates unscaled
            criteria,
            OPTFLOW_LK_GET_MIN_EIGENVALS,
            0.0, // min_eigen_threshold: accept regardless of scale
        )
        .unwrap();

        assert_eq!(status[0], 1);
        // Literal, deliberately not derived from FLT_SCALE: this test must
        // fail if that constant is ever changed, not follow it.
        let expected = 51200.0 / (9.0 * 1048576.0); // 0.005425347222...
        let relative_error = (err[0] as f64 - expected).abs() / expected;
        assert!(
            relative_error < 1e-5,
            "min eigenvalue must be on OpenCV's scale: expected {expected}, got {} \
             (relative error {relative_error})",
            err[0]
        );
    }

    /// Hand-crafted mismatch sequence that oscillates by construction, so
    /// this test needs no real image data. H = identity (h00=h11=1, h01=0)
    /// makes eta = (bx, by) directly. purecv#131: without the oscillation
    /// half-step fallback, iteration continues past the cancelling pair
    /// instead of stopping there — see this function's own comment for the
    /// full trace.
    #[test]
    fn test_lk_iterate_applies_oscillation_half_step() {
        let mut call = 0;
        let (u, v) = lk_iterate(
            1.0, 0.0, 1.0, // h00, h01, h11
            1.0,           // inv_det
            0.0, 0.0,      // init_u, init_v
            10,            // max_iters
            1e-9,          // eps: tiny, never satisfied by these steps
            |_u, _v| {
                call += 1;
                match call {
                    1 => (1.0, 0.5),
                    2 => (-1.0, -0.5),
                    _ => (0.0, 0.0),
                }
            },
        );

        assert!(
            (u - 0.5).abs() < 1e-9,
            "expected u = 0.5 (oscillation half-step fallback), got {u}"
        );
        assert!(
            (v - 0.25).abs() < 1e-9,
            "expected v = 0.25 (oscillation half-step fallback), got {v}"
        );
    }
}

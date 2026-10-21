/*
 *  parity_video.rs
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

//! OpenCV 4.10.0 parity pins for the `video` module (`docs/parity/video.md`).
//!
//! Every expected value in this file was produced by a script under `scripts/opencv_ref/`
//! (`video_pyramid.py`, `video_lk.py`; helpers `_core_common.py`, `_features2d_common.py`,
//! `_video_common.py`) run in `cpp_ref/.venv` (opencv-contrib-python 4.10.0.84,
//! `cv2.ipp.setUseIPP(False)`, Windows x86-64). The script and the key of each block are named
//! in the comment above it. The inputs are rebuilt here from the same integer formulas the
//! scripts use, so no image file is committed.
//!
//! The wheel runs the `CV_SIMD128` block of `LKTrackerInvoker` for windows at least 8 px wide
//! (four-lane `f32` sums reduced at the end), while purecv keeps the generic scalar order, so
//! those cases agree to about 1e-3 px and a few intensity counts of mean absolute error; windows
//! up to 7 px wide run the scalar loop in the wheel and agree to 1 ulp (`SIMD_TOL`, `SCALAR_TOL`
//! hold the bounds the tests enforce). Statuses agree exactly in every case. Tests that pass pin
//! rows that already match OpenCV. Tests marked `#[ignore]`
//! document a `diverges` row; they fail when run
//! (`cargo test --test parity_video -- --ignored`) and are the executable form of the
//! measurement; the issue in each reason tracks the fix.

use purecv::core::types::{BorderTypes, Point2f, Size2i, TermCriteria, TermType};
use purecv::core::Matrix;
use purecv::video::*;

use ocv::*;

// ---------------------------------------------------------------------------------------
// Inputs (the same formulas as the scripts; `_video_common.py`, `_features2d_common.py`)
// ---------------------------------------------------------------------------------------

/// One step of the 32-bit LCG `state * 1664525 + 1013904223 (mod 2^32)`.
fn lcg_step(s: u32) -> u32 {
    s.wrapping_mul(1664525).wrapping_add(1013904223)
}

/// 60 LCG rectangles over a mid-grey background plus +/-6 LCG noise (`textured_u8`).
fn textured_u8(rows: usize, cols: usize, seed: u32) -> Matrix<u8> {
    let mut s = seed;
    let mut draw = || {
        s = lcg_step(s);
        (s >> 16) as usize
    };
    let mut buf = vec![128i32; rows * cols];
    for _ in 0..60 {
        let w = 4 + draw() % 24;
        let h = 4 + draw() % 24;
        let x0 = draw() % cols;
        let y0 = draw() % rows;
        let v = (draw() % 256) as i32;
        for y in y0..(y0 + h).min(rows) {
            for x in x0..(x0 + w).min(cols) {
                buf[y * cols + x] = v;
            }
        }
    }
    let mut d = Vec::with_capacity(rows * cols);
    for v in buf {
        d.push((v + (draw() % 13) as i32 - 6).clamp(0, 255) as u8);
    }
    Matrix::from_vec(rows, cols, 1, d)
}

/// The #163 fixture: xorshift32 cells of 30/220 in a checkerboard, box-blurred 3x3 with the
/// mean of the in-image neighbours (`cell_texture`; `w` columns by `h` rows).
fn cell_texture(w: usize, h: usize) -> Matrix<u8> {
    let mut seed = 0x1234_5678u32;
    let mut next = move || {
        seed ^= seed << 13;
        seed ^= seed >> 17;
        seed ^= seed << 5;
        seed
    };
    let mut xs = vec![0usize];
    let mut x = 0usize;
    while x < w {
        x += 4 + (next() % 9) as usize;
        xs.push(x.min(w));
    }
    let mut ys = vec![0usize];
    let mut y = 0usize;
    while y < h {
        y += 4 + (next() % 9) as usize;
        ys.push(y.min(h));
    }
    let mut data = vec![0u8; w * h];
    for yy in 0..ys.len() - 1 {
        for xx in 0..xs.len() - 1 {
            let v = if (xx + yy) % 2 == 0 { 30u8 } else { 220u8 };
            for py in ys[yy]..ys[yy + 1] {
                for px in xs[xx]..xs[xx + 1] {
                    data[py * w + px] = v;
                }
            }
        }
    }
    let mut blurred = vec![0u8; w * h];
    for y in 0..h {
        for x in 0..w {
            let (mut s, mut n) = (0u32, 0u32);
            for dy in -1i32..=1 {
                for dx in -1i32..=1 {
                    let (sx, sy) = (x as i32 + dx, y as i32 + dy);
                    if sx >= 0 && sx < w as i32 && sy >= 0 && sy < h as i32 {
                        s += u32::from(data[sy as usize * w + sx as usize]);
                        n += 1;
                    }
                }
            }
            blurred[y * w + x] = (s / n) as u8;
        }
    }
    Matrix::<u8>::from_vec(h, w, 1, blurred)
}

/// Integer translation by `(dx, dy)`; uncovered pixels are 0 (`shift_image`).
fn shift_image(img: &Matrix<u8>, dx: i32, dy: i32) -> Matrix<u8> {
    let (w, h) = (img.cols, img.rows);
    let mut data = vec![0u8; w * h];
    for y in 0..h as i32 {
        for x in 0..w as i32 {
            let (sx, sy) = (x - dx, y - dy);
            if sx >= 0 && sx < w as i32 && sy >= 0 && sy < h as i32 {
                data[(y as usize) * w + x as usize] = img.data[(sy as usize) * w + sx as usize];
            }
        }
    }
    Matrix::<u8>::from_vec(h, w, 1, data)
}

/// Sub-pixel translation by `(dx4 / 4, dy4 / 4)` px with integer bilinear weights and
/// replicated edges (`shift_quarter`): `out(x, y) = img(x - dx, y - dy)`.
fn shift_quarter(img: &Matrix<u8>, dx4: i32, dy4: i32) -> Matrix<u8> {
    let (w, h) = (img.cols as i32, img.rows as i32);
    let (ix, fx) = ((-dx4).div_euclid(4), (-dx4).rem_euclid(4) as i64);
    let (iy, fy) = ((-dy4).div_euclid(4), (-dy4).rem_euclid(4) as i64);
    let at = |x: i32, y: i32| -> i64 {
        let x = x.clamp(0, w - 1) as usize;
        let y = y.clamp(0, h - 1) as usize;
        img.data[y * img.cols + x] as i64
    };
    let mut data = Vec::with_capacity(img.data.len());
    for y in 0..h {
        for x in 0..w {
            let v = ((4 - fx) * (4 - fy) * at(x + ix, y + iy)
                + fx * (4 - fy) * at(x + ix + 1, y + iy)
                + (4 - fx) * fy * at(x + ix, y + iy + 1)
                + fx * fy * at(x + ix + 1, y + iy + 1)
                + 8)
                >> 4;
            data.push(v as u8);
        }
    }
    Matrix::from_vec(img.rows, img.cols, 1, data)
}

/// `(17 x + 13 y + 3 x y) mod 251` with `x` shifted right by `shift` (saturating at 0), the
/// fixture of the in-tree LK reference tests (`reusable_texture`).
fn reusable_texture(rows: usize, cols: usize, shift: usize) -> Matrix<u8> {
    Matrix::from_vec(
        rows,
        cols,
        1,
        (0..rows * cols)
            .map(|i| {
                let y = i / cols;
                let x = (i % cols).saturating_sub(shift);
                ((x * 17 + y * 13 + x * y * 3) % 251) as u8
            })
            .collect(),
    )
}

/// Row-major grid `(x, y)` for `y in (y0..y1).step_by(step)`, `x in (x0..x1).step_by(step)`.
fn grid_points(x0: i32, x1: i32, y0: i32, y1: i32, step: i32) -> Vec<Point2f> {
    let mut v = Vec::new();
    let mut y = y0;
    while y < y1 {
        let mut x = x0;
        while x < x1 {
            v.push(Point2f::new(x as f32, y as f32));
            x += step;
        }
        y += step;
    }
    v
}

/// The 150-point grid of #163 (`grid_points(40, 280, 40, 200, 16)`).
fn grid_150() -> Vec<Point2f> {
    let pts = grid_points(40, 280, 40, 200, 16);
    assert_eq!(pts.len(), 150);
    pts
}

/// Every fifth point of the grid (30 points; the scripts' `pts[::5]`).
fn grid_30() -> Vec<Point2f> {
    grid_150().into_iter().step_by(5).collect()
}

fn fnv1a32(bytes: &[u8]) -> u32 {
    let mut h = 0x811c_9dc5u32;
    for &b in bytes {
        h ^= b as u32;
        h = h.wrapping_mul(0x0100_0193);
    }
    h
}

/// FNV-1a of the row-major pixels (`pixel_digest`).
fn pixel_digest(m: &Matrix<u8>) -> u32 {
    fnv1a32(&m.data)
}

/// FNV-1a of an f32 level holding `u8` values, as the `uint8` level bytes of `cv2`.
fn level_digest(m: &Matrix<f32>) -> u32 {
    let b: Vec<u8> = m.data.iter().map(|&v| v as u8).collect();
    fnv1a32(&b)
}

/// FNV-1a of an f32 derivative holding `i16` values, as the little-endian `int16` plane bytes.
fn deriv_digest(m: &Matrix<f32>) -> u32 {
    let mut b = Vec::with_capacity(m.data.len() * 2);
    for &v in &m.data {
        b.extend_from_slice(&(v as i16).to_le_bytes());
    }
    fnv1a32(&b)
}

fn replicate_channels(m: &Matrix<u8>, channels: usize) -> Matrix<u8> {
    let mut out = Matrix::<u8>::new(m.rows, m.cols, channels);
    for i in 0..m.rows * m.cols {
        for c in 0..channels {
            out.data[i * channels + c] = m.data[i];
        }
    }
    out
}

// ---------------------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------------------

type Flow = (Vec<Point2f>, Vec<u8>, Vec<f32>);

fn criteria(kind: i32, count: i32, epsilon: f64) -> TermCriteria {
    let ty = match kind {
        1 => TermType::Count,
        2 => TermType::Eps,
        _ => TermType::Both,
    };
    TermCriteria::new(ty, count, epsilon)
}

/// `(COUNT | EPS, 30, 0.01)`, OpenCV's default `TermCriteria` for `calcOpticalFlowPyrLK`.
fn default_criteria() -> TermCriteria {
    criteria(3, 30, 0.01)
}

#[allow(clippy::too_many_arguments)]
fn lk(
    prev: &Matrix<u8>,
    next: &Matrix<u8>,
    pts: &[Point2f],
    init: Option<&[Point2f]>,
    win: (i32, i32),
    max_level: i32,
    c: TermCriteria,
    flags: i32,
    min_eig: f64,
) -> Flow {
    calc_optical_flow_pyramid_lk(
        prev,
        next,
        pts,
        init,
        Size2i::new(win.0, win.1),
        max_level,
        c,
        flags,
        min_eig,
    )
    .unwrap()
}

/// Tolerances of one case group: positions within `pos` px, errors within
/// `err_abs + err_rel * |err|`.
#[derive(Clone, Copy)]
struct Tol {
    pos: f32,
    err_abs: f32,
    err_rel: f32,
}

/// Windows up to 7 px wide run the wheel's scalar loop: the largest differences measured are
/// 6.8e-6 px (1 ulp at x = 130) and 3e-6 in the mean absolute error.
const SCALAR_TOL: Tol = Tol {
    pos: 2e-5,
    err_abs: 1e-5,
    err_rel: 1e-6,
};

/// Windows at least 8 px wide go through the wheel's `CV_SIMD128` reduction (see the module
/// docs): the largest differences measured are 1.2e-3 px (20x20) and 0.013 in the mean
/// absolute error (21x21).
const SIMD_TOL: Tol = Tol {
    pos: 2e-3,
    err_abs: 2e-2,
    err_rel: 0.0,
};

/// Tolerances for a case: the position bound depends on the window width, the error bound on
/// whether `err` is the mean absolute error or the minimum eigenvalue (within 7.2e-7, 1 ulp of
/// 0.63, for every window).
fn tolerances(win_width: i32, min_eig_output: bool) -> Tol {
    let base = if win_width >= 8 { SIMD_TOL } else { SCALAR_TOL };
    if min_eig_output {
        Tol {
            pos: base.pos,
            err_abs: 1e-6,
            err_rel: 1e-6,
        }
    } else {
        base
    }
}

/// Statuses must match exactly; for points both sides track, the coordinates within
/// `tol.pos` px and the error within `tol.err_abs + tol.err_rel * |err|`.
fn assert_flow(actual: &Flow, expected: &[FlowRow], tol: Tol, what: &str) {
    let (pts, status, err) = actual;
    assert_eq!(pts.len(), expected.len(), "{what}: count");
    for (i, e) in expected.iter().enumerate() {
        assert_eq!(status[i], e.2, "{what}[{i}]: status");
        if e.2 == 0 {
            continue;
        }
        assert!(
            (pts[i].x - e.0).abs() <= tol.pos && (pts[i].y - e.1).abs() <= tol.pos,
            "{what}[{i}]: purecv ({}, {}) vs OpenCV ({}, {})",
            pts[i].x,
            pts[i].y,
            e.0,
            e.1
        );
        assert!(
            (err[i] - e.3).abs() <= tol.err_abs + tol.err_rel * e.3.abs(),
            "{what}[{i}]: err purecv {} vs OpenCV {}",
            err[i],
            e.3
        );
    }
}

fn pyramid_image(name: &str) -> Matrix<u8> {
    match name {
        "cell_320x240" => cell_texture(320, 240),
        "textured_96x128_s11" => textured_u8(96, 128, 11),
        "reusable_33x49" => reusable_texture(33, 49, 0),
        "reusable_9x9" => reusable_texture(9, 9, 0),
        "reusable_1x1" => reusable_texture(1, 1, 0),
        "reusable_1x7" => reusable_texture(1, 7, 0),
        "reusable_3x2" => reusable_texture(3, 2, 0),
        "reusable_7x5" => reusable_texture(7, 5, 0),
        _ => unreachable!("{name}"),
    }
}

// ---------------------------------------------------------------------------------------
// Inputs (video_pyramid.py / video_lk.py, "inputs")
// ---------------------------------------------------------------------------------------

/// The Rust twins of the script generators produce the same bytes.
#[test]
fn inputs_match_the_scripts() {
    for (name, digest) in INPUT_DIGESTS {
        let img = match name {
            "cell_320x240_shift_10_0" => shift_image(&cell_texture(320, 240), 10, 0),
            "textured_96x128_s11_shift_2.25_-0.75" => {
                shift_quarter(&textured_u8(96, 128, 11), 9, -3)
            }
            _ => pyramid_image(name),
        };
        assert_eq!(pixel_digest(&img), digest, "{name}");
    }
    assert_eq!(
        pixel_digest(&reusable_texture(33, 49, 1)),
        REUSABLE_33X49_SHIFT1_DIGEST
    );
}

// ---------------------------------------------------------------------------------------
// build_optical_flow_pyramid (video_pyramid.py, "grid", "inputs_d2")
// ---------------------------------------------------------------------------------------

/// Level count (the `winSize` cutoff), level sizes, level pixels and both Scharr derivative
/// planes are identical to `cv2.buildOpticalFlowPyramid` on eight images and seven
/// `(winSize, maxLevel)` settings, with and without `withDerivatives`.
#[test]
fn build_optical_flow_pyramid_matches_opencv() {
    for (key, returned_level, levels) in PYRAMIDS {
        let parts: Vec<&str> = key.split('/').collect();
        let img = pyramid_image(parts[0]);
        let (w, h) = parts[1][3..].split_once('x').unwrap();
        let win = Size2i::new(w.parse().unwrap(), h.parse().unwrap());
        let max_level: usize = parts[2][3..].parse().unwrap();
        for with_derivatives in [true, false] {
            let pyr = build_optical_flow_pyramid(
                &img,
                win,
                max_level,
                with_derivatives,
                BorderTypes::Reflect101,
                BorderTypes::Constant,
            )
            .unwrap();
            assert_eq!(pyr.levels.len() - 1, returned_level, "{key}: level count");
            assert_eq!(
                pyr.dx.len(),
                if with_derivatives { levels.len() } else { 0 }
            );
            for (i, &(rows, cols, pixels, dx, dy)) in levels.iter().enumerate() {
                let level = &pyr.levels[i];
                assert_eq!(
                    (level.rows, level.cols),
                    (rows, cols),
                    "{key}: level {i} size"
                );
                assert_eq!(level_digest(level), pixels, "{key}: level {i} pixels");
                if with_derivatives {
                    assert_eq!(deriv_digest(&pyr.dx[i]), dx, "{key}: level {i} dx");
                    assert_eq!(deriv_digest(&pyr.dy[i]), dy, "{key}: level {i} dy");
                }
            }
        }
    }
}

/// `winSize` below 3 in either dimension is rejected (OpenCV's
/// `CV_Assert(img.depth() == CV_8U && winSize.width > 2 && winSize.height > 2)`).
#[test]
fn build_optical_flow_pyramid_rejects_small_windows_like_opencv() {
    let img = reusable_texture(33, 49, 0);
    for win in [Size2i::new(2, 5), Size2i::new(5, 2)] {
        let r = build_optical_flow_pyramid(
            &img,
            win,
            3,
            true,
            BorderTypes::Reflect101,
            BorderTypes::Constant,
        );
        assert!(r.is_err(), "{win:?}");
    }
}

/// OpenCV builds pyramids of any channel count (`CV_8UC3` levels, `CV_16SC6` derivatives);
/// purecv rejects every non-single-channel image.
#[test]
#[ignore = "diverges from OpenCV 4.10.0, see #256"]
fn build_optical_flow_pyramid_accepts_multi_channel_input() {
    let img = replicate_channels(&reusable_texture(33, 49, 0), 3);
    let pyr = build_optical_flow_pyramid(
        &img,
        Size2i::new(5, 5),
        2,
        true,
        BorderTypes::Reflect101,
        BorderTypes::Constant,
    )
    .expect("OpenCV accepts a CV_8UC3 image");
    assert_eq!(pyr.levels.len(), 3);
}

// ---------------------------------------------------------------------------------------
// calc_optical_flow_pyramid_lk: the #163 setup (video_lk.py, "grid163")
// ---------------------------------------------------------------------------------------

/// 320x240 cell texture, 150-point grid, integer shifts of 0 and 10 px, 21x21 window, four
/// levels, OpenCV's default criteria and threshold, `OPTFLOW_USE_INITIAL_FLOW` with the
/// template positions as the initial estimate: the accept set is identical (150, 150, 149 and
/// 145 points) and every tracked position is within 4.1e-4 px of OpenCV's (bound 2e-3).
#[test]
fn lk_grid163_matches_opencv() {
    let img = cell_texture(320, 240);
    let pts = grid_150();
    let tol = tolerances(21, false);
    for (dx, dy, expected) in [
        (0, 0, &GRID163_SHIFT_0_0),
        (10, 0, &GRID163_SHIFT_10_0),
        (0, 10, &GRID163_SHIFT_0_10),
        (10, 10, &GRID163_SHIFT_10_10),
    ] {
        let next = shift_image(&img, dx, dy);
        let flow = lk(
            &img,
            &next,
            &pts,
            Some(&pts),
            (21, 21),
            3,
            default_criteria(),
            OPTFLOW_USE_INITIAL_FLOW,
            1e-4,
        );
        assert_flow(&flow, expected, tol, &format!("shift ({dx}, {dy})"));
    }
    // Without the flag the initial estimate is the template position: the same result.
    let next = shift_image(&img, 10, 0);
    let flow = lk(
        &img,
        &next,
        &pts,
        None,
        (21, 21),
        3,
        default_criteria(),
        0,
        1e-4,
    );
    assert_flow(&flow, &GRID163_SHIFT_10_0, tol, "shift (10, 0), no flags");
}

/// Parameter variations on 30 of the grid points and the 10 px shift: both flags, even and
/// non-square windows, depths 0/1/5, thresholds, every criteria form (count only, epsilon
/// only, caps, zero iterations), and good, bad and ignored initial estimates.
#[test]
fn lk_parameter_variations_match_opencv() {
    let img = cell_texture(320, 240);
    let next = shift_image(&img, 10, 0);
    let pts = grid_30();
    let good: Vec<Point2f> = pts.iter().map(|p| Point2f::new(p.x + 8.0, p.y)).collect();
    let bad: Vec<Point2f> = pts
        .iter()
        .map(|p| Point2f::new(p.x - 8.0, p.y + 3.0))
        .collect();
    let d = default_criteria();
    const GME: i32 = OPTFLOW_LK_GET_MIN_EIGENVALS;
    const UIF: i32 = OPTFLOW_USE_INITIAL_FLOW;
    for (key, expected) in VARIATIONS {
        let (init, win, level, c, flags, min_eig): (Option<&[Point2f]>, _, _, _, _, _) = match key {
            "default" => (None, (21, 21), 3, d, 0, 1e-4),
            "min_eig_output" => (None, (21, 21), 3, d, GME, 1e-4),
            "win_20x20" => (None, (20, 20), 3, d, 0, 1e-4),
            "win_15x9" => (None, (15, 9), 3, d, 0, 1e-4),
            "win_7x7" => (None, (7, 7), 3, d, 0, 1e-4),
            "win_3x3" => (None, (3, 3), 3, d, 0, 1e-4),
            "max_level_0" => (None, (21, 21), 0, d, 0, 1e-4),
            "max_level_1" => (None, (21, 21), 1, d, 0, 1e-4),
            "max_level_5" => (None, (21, 21), 5, d, 0, 1e-4),
            "min_eig_1e-2" => (None, (21, 21), 3, d, 0, 1e-2),
            "min_eig_0.5" => (None, (21, 21), 3, d, 0, 0.5),
            "min_eig_0.5_output" => (None, (21, 21), 3, d, GME, 0.5),
            "count_5" => (None, (21, 21), 3, criteria(1, 5, 0.0), 0, 1e-4),
            "count_0" => (None, (21, 21), 3, criteria(1, 0, 0.0), 0, 1e-4),
            "eps_only_0.001" => (None, (21, 21), 3, criteria(2, 0, 0.001), 0, 1e-4),
            "both_100_0" => (None, (21, 21), 3, criteria(3, 100, 0.0), 0, 1e-4),
            "both_1_0.03" => (None, (21, 21), 3, criteria(3, 1, 0.03), 0, 1e-4),
            "init_good/initial_flow" => (Some(&good), (21, 21), 3, d, UIF, 1e-4),
            "init_bad/initial_flow" => (Some(&bad), (21, 21), 3, d, UIF, 1e-4),
            "init_good/ignored_without_flag" => (Some(&good), (21, 21), 3, d, 0, 1e-4),
            "init_good/initial_flow/count_0" => {
                (Some(&good), (21, 21), 3, criteria(1, 0, 0.0), UIF, 1e-4)
            }
            _ => unreachable!("{key}"),
        };
        let flow = lk(&img, &next, &pts, init, win, level, c, flags, min_eig);
        assert_flow(&flow, expected, tolerances(win.0, flags & GME != 0), key);
    }
}

// ---------------------------------------------------------------------------------------
// Sub-pixel motion (video_lk.py, "subpixel")
// ---------------------------------------------------------------------------------------

/// Rectangle texture shifted by (2.25, -0.75) px with integer bilinear weights, 48 points:
/// default parameters, both error outputs, even (10x10) and small (7x7) windows, depths 0 and
/// 2, and a tight epsilon with 100 iterations.
#[test]
fn lk_subpixel_shift_matches_opencv() {
    let tex = textured_u8(96, 128, 11);
    let next = shift_quarter(&tex, 9, -3);
    let pts = grid_points(16, 112, 16, 80, 12);
    assert_eq!(pts.len(), 48);
    let d = default_criteria();
    for (key, expected) in SUBPIXEL {
        let (win, level, c, flags) = match key {
            "default" => ((21, 21), 3, d, 0),
            "min_eig_output" => ((21, 21), 3, d, OPTFLOW_LK_GET_MIN_EIGENVALS),
            "win_10x10" => ((10, 10), 3, d, 0),
            "win_7x7" => ((7, 7), 3, d, 0),
            "max_level_2" => ((21, 21), 2, d, 0),
            "max_level_0" => ((21, 21), 0, d, 0),
            "eps_0.001_count_100" => ((21, 21), 3, criteria(3, 100, 0.001), 0),
            _ => unreachable!("{key}"),
        };
        let flow = lk(&tex, &next, &pts, None, win, level, c, flags, 1e-4);
        assert_flow(&flow, expected, tolerances(win.0, flags != 0), key);
    }
}

// ---------------------------------------------------------------------------------------
// Edge and outside points, final bounds (video_lk.py, "edges", "final_bounds")
// ---------------------------------------------------------------------------------------

/// Points on, near and outside the 33x49 fixture with a 5x5 window, at depths 0 and 2, with
/// and without an initial estimate, MAE and minimum-eigenvalue outputs: which points are
/// lost (window outside the padded level, at the template or during the iterations) matches
/// OpenCV exactly, and the tracked ones agree to 1 ulp.
#[test]
fn lk_edge_and_outside_points_match_opencv() {
    let prev = reusable_texture(33, 49, 0);
    let next = reusable_texture(33, 49, 1);
    let pts: Vec<Point2f> = [
        (0.0, 0.0),
        (48.0, 32.0),
        (-0.25, 0.5),
        (48.75, 32.5),
        (-20.0, -20.0),
        (-4.5, 5.0),
        (5.0, -4.5),
        (52.9, 10.0),
        (10.0, 36.9),
        (2.25, 2.5),
        (24.5, 16.5),
        (16.0, 16.0),
        (45.0, 30.0),
        (48.0, 0.0),
        (0.0, 32.0),
    ]
    .iter()
    .map(|&(x, y)| Point2f::new(x, y))
    .collect();
    let guess: Vec<Point2f> = pts.iter().map(|p| Point2f::new(p.x + 1.0, p.y)).collect();
    for (key, expected) in EDGES {
        let parts: Vec<&str> = key.split('/').collect();
        let level: i32 = parts[0][3..].parse().unwrap();
        let mut flags = if parts[1] == "min_eig" {
            OPTFLOW_LK_GET_MIN_EIGENVALS
        } else {
            0
        };
        let init = if parts.len() == 3 {
            flags |= OPTFLOW_USE_INITIAL_FLOW;
            Some(guess.as_slice())
        } else {
            None
        };
        let flow = lk(
            &prev,
            &next,
            &pts,
            init,
            (5, 5),
            level,
            criteria(3, 20, 0.03),
            flags,
            1e-4,
        );
        assert_flow(&flow, expected, tolerances(5, parts[1] == "min_eig"), key);
    }
}

/// The gate after the last iteration (#178, item 1): with zero iterations the estimate is
/// the initial guess, and a window corner below `-winSize` or at or beyond the level size
/// loses the point only when the mean absolute error is computed, not with
/// `OPTFLOW_LK_GET_MIN_EIGENVALS`.
#[test]
fn lk_final_bounds_gate_matches_opencv() {
    let img = reusable_texture(9, 9, 0);
    for (gx, gy, min_eig, status, err) in FINAL_BOUNDS {
        let flags = OPTFLOW_USE_INITIAL_FLOW
            | if min_eig {
                OPTFLOW_LK_GET_MIN_EIGENVALS
            } else {
                0
            };
        let flow = lk(
            &img,
            &img,
            &[Point2f::new(4.0, 4.0)],
            Some(&[Point2f::new(gx, gy)]),
            (5, 5),
            0,
            criteria(1, 0, 0.0),
            flags,
            1e-4,
        );
        let what = format!("guess ({gx}, {gy}), min_eig {min_eig}");
        assert_eq!(flow.1, vec![status], "{what}: status");
        if status == 1 {
            assert_eq!((flow.0[0].x, flow.0[0].y), (gx, gy), "{what}: position");
            assert!(
                (flow.2[0] - err).abs() <= SCALAR_TOL.err_abs + SCALAR_TOL.err_rel * err.abs(),
                "{what}: err {} vs {err}",
                flow.2[0]
            );
        }
    }
}

// ---------------------------------------------------------------------------------------
// Prebuilt pyramids and their borders (video_lk.py, "asymmetric_borders")
// ---------------------------------------------------------------------------------------

/// `OpticalFlowPyramid::pyr_border` reproduces OpenCV's `pyrBorder` padding: raw
/// `calcOpticalFlowPyrLK` on the 9x9 fixture padded by 8 with `copyMakeBorder` (constant,
/// replicate, wrap, reflect-101, a different one on each image) gives the same mean absolute
/// errors and the same first Newton step as `calc_optical_flow_pyramid_lk_with_pyramids` on
/// unpadded pyramids built with those borders, with and without stored derivatives.
#[test]
fn lk_prebuilt_pyramid_borders_match_opencv() {
    let img = reusable_texture(9, 9, 0);
    let win = Size2i::new(5, 5);
    let border = |name: &str| match name {
        "reflect101" => BorderTypes::Reflect101,
        "constant" => BorderTypes::Constant,
        "replicate" => BorderTypes::Replicate,
        "wrap" => BorderTypes::Wrap,
        _ => unreachable!("{name}"),
    };
    let mae_pts = [
        Point2f::new(0.0, 0.0),
        Point2f::new(8.0, 8.0),
        Point2f::new(0.25, 0.5),
    ];
    let step_pts = [Point2f::new(4.0, 4.0); 2];
    let step_guess = [Point2f::new(0.25, 0.5), Point2f::new(8.25, 8.5)];
    for (prev_border, next_border, mae_rows, step_rows) in ASYMMETRIC_BORDERS {
        for with_derivatives in [false, true] {
            let build = |b: &str| {
                build_optical_flow_pyramid(
                    &img,
                    win,
                    0,
                    with_derivatives,
                    border(b),
                    BorderTypes::Constant,
                )
                .unwrap()
            };
            let (prev, next) = (build(prev_border), build(next_border));
            let what = format!("{prev_border} -> {next_border}, derivatives {with_derivatives}");
            let mae = calc_optical_flow_pyramid_lk_with_pyramids(
                &prev,
                &next,
                &mae_pts,
                Some(&mae_pts),
                win,
                0,
                criteria(1, 0, 0.0),
                OPTFLOW_USE_INITIAL_FLOW,
                1e-4,
            )
            .unwrap();
            assert_flow(&mae, &mae_rows, SCALAR_TOL, &format!("{what}: MAE"));
            let step = calc_optical_flow_pyramid_lk_with_pyramids(
                &prev,
                &next,
                &step_pts,
                Some(&step_guess),
                win,
                0,
                criteria(1, 1, 0.0),
                OPTFLOW_USE_INITIAL_FLOW,
                1e-4,
            )
            .unwrap();
            assert_flow(&step, &step_rows, SCALAR_TOL, &format!("{what}: step"));
        }
    }
}

// ---------------------------------------------------------------------------------------
// Termination criteria (video_lk.py, "termination")
// ---------------------------------------------------------------------------------------

/// Count-only (default epsilon 0.01), epsilon-only (default count 30), zero and negative
/// counts and epsilons, the 100-iteration and epsilon-10 caps, on one point of the 33x49
/// fixture with a 5x5 window: positions and errors agree to 1 ulp.
#[test]
fn lk_termination_criteria_match_opencv() {
    let prev = reusable_texture(33, 49, 0);
    let next = reusable_texture(33, 49, 1);
    for (kind, count, epsilon, x, y, row) in TERMINATION {
        let flow = lk(
            &prev,
            &next,
            &[Point2f::new(x, y)],
            None,
            (5, 5),
            0,
            criteria(kind, count, epsilon),
            0,
            1e-4,
        );
        assert_flow(
            &flow,
            &[row],
            SCALAR_TOL,
            &format!("{kind}/{count}/{epsilon}/({x}, {y})"),
        );
    }
}

// ---------------------------------------------------------------------------------------
// Input validation and flags (video_lk.py, "inputs_d2"; tracking.hpp)
// ---------------------------------------------------------------------------------------

/// `OPTFLOW_USE_INITIAL_FLOW = 4`, `OPTFLOW_LK_GET_MIN_EIGENVALS = 8` (measured as
/// `cv2.OPTFLOW_USE_INITIAL_FLOW` / `cv2.OPTFLOW_LK_GET_MIN_EIGENVALS`).
#[test]
fn optflow_flags_match_opencv() {
    assert_eq!(OPTFLOW_USE_INITIAL_FLOW, 4);
    assert_eq!(OPTFLOW_LK_GET_MIN_EIGENVALS, 8);
}

/// Rejected like OpenCV's `CV_Assert`s: a window side below 3, images of different sizes, an
/// initial-estimate list of the wrong length. Accepted like OpenCV: no points (empty
/// outputs) and a negative `minEigThreshold` (nothing is filtered).
#[test]
fn lk_input_validation_matches_opencv() {
    let prev = reusable_texture(33, 49, 0);
    let next = reusable_texture(33, 49, 1);
    let pts = [Point2f::new(16.0, 16.0), Point2f::new(30.0, 20.0)];
    let d = default_criteria();
    let run = |p: &Matrix<u8>, n: &Matrix<u8>, init: Option<&[Point2f]>, win, flags| {
        calc_optical_flow_pyramid_lk(p, n, &pts, init, Size2i::new(win, win), 3, d, flags, 1e-4)
    };
    assert!(run(&prev, &next, None, 2, 0).is_err(), "winSize 2");
    let mut narrow = Matrix::<u8>::new(33, 40, 1);
    for y in 0..33 {
        narrow.data[y * 40..y * 40 + 40].copy_from_slice(&next.data[y * 49..y * 49 + 40]);
    }
    assert!(run(&prev, &narrow, None, 21, 0).is_err(), "size mismatch");
    assert!(
        run(&prev, &next, Some(&pts[..1]), 21, OPTFLOW_USE_INITIAL_FLOW).is_err(),
        "initial flow length"
    );
    let empty =
        calc_optical_flow_pyramid_lk(&prev, &next, &[], None, Size2i::new(21, 21), 3, d, 0, 1e-4)
            .unwrap();
    assert!(empty.0.is_empty() && empty.1.is_empty() && empty.2.is_empty());
    let negative = lk(&prev, &next, &pts, None, (5, 5), 3, d, 0, -1.0);
    assert_flow(&negative, &D2_U8C1, SCALAR_TOL, "minEigThreshold -1");
}

/// OpenCV tracks on images of any channel count (the window spans `winSize.width * cn`
/// samples); three replicated channels give the single-channel result to 1 ulp. purecv
/// rejects every non-single-channel image.
#[test]
#[ignore = "diverges from OpenCV 4.10.0, see #256"]
fn lk_accepts_multi_channel_images() {
    let prev = replicate_channels(&reusable_texture(33, 49, 0), 3);
    let next = replicate_channels(&reusable_texture(33, 49, 1), 3);
    let pts = [Point2f::new(16.0, 16.0), Point2f::new(30.0, 20.0)];
    let flow = calc_optical_flow_pyramid_lk(
        &prev,
        &next,
        &pts,
        None,
        Size2i::new(5, 5),
        3,
        default_criteria(),
        0,
        1e-4,
    )
    .expect("OpenCV accepts CV_8UC3 images");
    assert_flow(&flow, &D2_U8C3, SCALAR_TOL, "3 channels");
}

/// OpenCV asserts that a supplied pyramid's padding (the builder's `winSize`) is at least the
/// tracking window (`lkpyramid.cpp`, the `locateROI` check on level 1); purecv has no padding
/// and extends every level on the fly, so a larger window is accepted.
#[test]
#[ignore = "diverges from OpenCV 4.10.0, see #184"]
fn lk_prebuilt_rejects_windows_larger_than_the_pyramid_padding() {
    let build = |img: &Matrix<u8>| {
        build_optical_flow_pyramid(
            img,
            Size2i::new(5, 5),
            2,
            true,
            BorderTypes::Reflect101,
            BorderTypes::Constant,
        )
        .unwrap()
    };
    let prev = build(&reusable_texture(33, 49, 0));
    let next = build(&reusable_texture(33, 49, 1));
    assert_eq!(
        prev.levels.len(),
        3,
        "the padding check needs more than one level"
    );
    let r = calc_optical_flow_pyramid_lk_with_pyramids(
        &prev,
        &next,
        &[Point2f::new(16.0, 16.0)],
        None,
        Size2i::new(7, 7),
        2,
        default_criteria(),
        0,
        1e-4,
    );
    assert!(
        r.is_err(),
        "OpenCV rejects a 7x7 window on a pyramid padded for 5x5"
    );
}

/// OpenCV asserts `maxLevel >= 0`; purecv clamps a negative `max_level` to 0 and tracks.
#[test]
#[ignore = "diverges from OpenCV 4.10.0, see #257"]
fn lk_rejects_negative_max_level() {
    let prev = reusable_texture(33, 49, 0);
    let next = reusable_texture(33, 49, 1);
    let r = calc_optical_flow_pyramid_lk(
        &prev,
        &next,
        &[Point2f::new(16.0, 16.0)],
        None,
        Size2i::new(21, 21),
        -1,
        default_criteria(),
        0,
        1e-4,
    );
    assert!(r.is_err(), "OpenCV rejects maxLevel -1");
}

// ---------------------------------------------------------------------------------------
// Expected values (generated from the scripts' JSON)
// ---------------------------------------------------------------------------------------

/// OpenCV 4.10.0 values from `scripts/opencv_ref/video_pyramid.py` and `video_lk.py`
/// (opencv-contrib-python 4.10.0.84, IPP off, Windows x86-64).
#[allow(clippy::approx_constant, clippy::excessive_precision)]
mod ocv {
    /// One tracked point: `(x, y, status, err)`; `x`, `y` and `err` are compared only when
    /// `status` is 1 (OpenCV leaves them undefined for lost points).
    pub type FlowRow = (f32, f32, u8, f32);
    /// One pyramid level: `(rows, cols, pixel digest, dx digest, dy digest)`.
    pub type Level = (usize, usize, u32, u32, u32);

    /// `video_pyramid.py` "grid": `(key, returned level, [(rows, cols, pixel digest, dx digest, dy digest)])`
    #[rustfmt::skip]
    pub const PYRAMIDS: [(&str, usize, &[Level]); 56] = [
        ("cell_320x240/win21x21/max3", 3, &[(240, 320, 0xbc292803, 0x55fbfc98, 0x5f1d69d3), (120, 160, 0x770e8634, 0x84fef572, 0x7bf9c164), (60, 80, 0x9df5e62c, 0xebd8672a, 0x48a07271), (30, 40, 0x5384c340, 0x7af34d9d, 0x5c7dfbf7)]),
        ("cell_320x240/win21x21/max0", 0, &[(240, 320, 0xbc292803, 0x55fbfc98, 0x5f1d69d3)]),
        ("cell_320x240/win21x21/max8", 3, &[(240, 320, 0xbc292803, 0x55fbfc98, 0x5f1d69d3), (120, 160, 0x770e8634, 0x84fef572, 0x7bf9c164), (60, 80, 0x9df5e62c, 0xebd8672a, 0x48a07271), (30, 40, 0x5384c340, 0x7af34d9d, 0x5c7dfbf7)]),
        ("cell_320x240/win3x3/max8", 6, &[(240, 320, 0xbc292803, 0x55fbfc98, 0x5f1d69d3), (120, 160, 0x770e8634, 0x84fef572, 0x7bf9c164), (60, 80, 0x9df5e62c, 0xebd8672a, 0x48a07271), (30, 40, 0x5384c340, 0x7af34d9d, 0x5c7dfbf7), (15, 20, 0x2a2f15ba, 0x92c038cd, 0x335b80cc), (8, 10, 0x87bf145e, 0xb2e0851f, 0xd4e60b6c), (4, 5, 0x7a2cd137, 0x5ed242bf, 0x42eb5601)]),
        ("cell_320x240/win5x5/max2", 2, &[(240, 320, 0xbc292803, 0x55fbfc98, 0x5f1d69d3), (120, 160, 0x770e8634, 0x84fef572, 0x7bf9c164), (60, 80, 0x9df5e62c, 0xebd8672a, 0x48a07271)]),
        ("cell_320x240/win20x20/max5", 3, &[(240, 320, 0xbc292803, 0x55fbfc98, 0x5f1d69d3), (120, 160, 0x770e8634, 0x84fef572, 0x7bf9c164), (60, 80, 0x9df5e62c, 0xebd8672a, 0x48a07271), (30, 40, 0x5384c340, 0x7af34d9d, 0x5c7dfbf7)]),
        ("cell_320x240/win15x9/max4", 4, &[(240, 320, 0xbc292803, 0x55fbfc98, 0x5f1d69d3), (120, 160, 0x770e8634, 0x84fef572, 0x7bf9c164), (60, 80, 0x9df5e62c, 0xebd8672a, 0x48a07271), (30, 40, 0x5384c340, 0x7af34d9d, 0x5c7dfbf7), (15, 20, 0x2a2f15ba, 0x92c038cd, 0x335b80cc)]),
        ("textured_96x128_s11/win21x21/max3", 2, &[(96, 128, 0xaae3288a, 0x96c13abd, 0x08867038), (48, 64, 0x7259acc1, 0x9311e1ec, 0x2a36d17a), (24, 32, 0x0d50f9e9, 0x9ece92e8, 0x352e4975)]),
        ("textured_96x128_s11/win21x21/max0", 0, &[(96, 128, 0xaae3288a, 0x96c13abd, 0x08867038)]),
        ("textured_96x128_s11/win21x21/max8", 2, &[(96, 128, 0xaae3288a, 0x96c13abd, 0x08867038), (48, 64, 0x7259acc1, 0x9311e1ec, 0x2a36d17a), (24, 32, 0x0d50f9e9, 0x9ece92e8, 0x352e4975)]),
        ("textured_96x128_s11/win3x3/max8", 4, &[(96, 128, 0xaae3288a, 0x96c13abd, 0x08867038), (48, 64, 0x7259acc1, 0x9311e1ec, 0x2a36d17a), (24, 32, 0x0d50f9e9, 0x9ece92e8, 0x352e4975), (12, 16, 0x4b699cb1, 0xba670832, 0x0e4de988), (6, 8, 0x5082a5af, 0xe4ee2c24, 0x4774c942)]),
        ("textured_96x128_s11/win5x5/max2", 2, &[(96, 128, 0xaae3288a, 0x96c13abd, 0x08867038), (48, 64, 0x7259acc1, 0x9311e1ec, 0x2a36d17a), (24, 32, 0x0d50f9e9, 0x9ece92e8, 0x352e4975)]),
        ("textured_96x128_s11/win20x20/max5", 2, &[(96, 128, 0xaae3288a, 0x96c13abd, 0x08867038), (48, 64, 0x7259acc1, 0x9311e1ec, 0x2a36d17a), (24, 32, 0x0d50f9e9, 0x9ece92e8, 0x352e4975)]),
        ("textured_96x128_s11/win15x9/max4", 3, &[(96, 128, 0xaae3288a, 0x96c13abd, 0x08867038), (48, 64, 0x7259acc1, 0x9311e1ec, 0x2a36d17a), (24, 32, 0x0d50f9e9, 0x9ece92e8, 0x352e4975), (12, 16, 0x4b699cb1, 0xba670832, 0x0e4de988)]),
        ("reusable_33x49/win21x21/max3", 0, &[(33, 49, 0x5d187c70, 0x7a3638ef, 0xcb8bd978)]),
        ("reusable_33x49/win21x21/max0", 0, &[(33, 49, 0x5d187c70, 0x7a3638ef, 0xcb8bd978)]),
        ("reusable_33x49/win21x21/max8", 0, &[(33, 49, 0x5d187c70, 0x7a3638ef, 0xcb8bd978)]),
        ("reusable_33x49/win3x3/max8", 3, &[(33, 49, 0x5d187c70, 0x7a3638ef, 0xcb8bd978), (17, 25, 0x5389c4b5, 0x50bd0303, 0xb75f2404), (9, 13, 0x331261b7, 0x6f2b7d06, 0x71f7fb69), (5, 7, 0xf6cf1ed1, 0xe7d2cd48, 0x578c6946)]),
        ("reusable_33x49/win5x5/max2", 2, &[(33, 49, 0x5d187c70, 0x7a3638ef, 0xcb8bd978), (17, 25, 0x5389c4b5, 0x50bd0303, 0xb75f2404), (9, 13, 0x331261b7, 0x6f2b7d06, 0x71f7fb69)]),
        ("reusable_33x49/win20x20/max5", 0, &[(33, 49, 0x5d187c70, 0x7a3638ef, 0xcb8bd978)]),
        ("reusable_33x49/win15x9/max4", 1, &[(33, 49, 0x5d187c70, 0x7a3638ef, 0xcb8bd978), (17, 25, 0x5389c4b5, 0x50bd0303, 0xb75f2404)]),
        ("reusable_9x9/win21x21/max3", 0, &[(9, 9, 0xef849369, 0xfdcbab62, 0x7a099b93)]),
        ("reusable_9x9/win21x21/max0", 0, &[(9, 9, 0xef849369, 0xfdcbab62, 0x7a099b93)]),
        ("reusable_9x9/win21x21/max8", 0, &[(9, 9, 0xef849369, 0xfdcbab62, 0x7a099b93)]),
        ("reusable_9x9/win3x3/max8", 1, &[(9, 9, 0xef849369, 0xfdcbab62, 0x7a099b93), (5, 5, 0x201cbb50, 0x312b3f14, 0xf1dece8a)]),
        ("reusable_9x9/win5x5/max2", 0, &[(9, 9, 0xef849369, 0xfdcbab62, 0x7a099b93)]),
        ("reusable_9x9/win20x20/max5", 0, &[(9, 9, 0xef849369, 0xfdcbab62, 0x7a099b93)]),
        ("reusable_9x9/win15x9/max4", 0, &[(9, 9, 0xef849369, 0xfdcbab62, 0x7a099b93)]),
        ("reusable_1x1/win21x21/max3", 0, &[(1, 1, 0x050c5d1f, 0x117697cd, 0x117697cd)]),
        ("reusable_1x1/win21x21/max0", 0, &[(1, 1, 0x050c5d1f, 0x117697cd, 0x117697cd)]),
        ("reusable_1x1/win21x21/max8", 0, &[(1, 1, 0x050c5d1f, 0x117697cd, 0x117697cd)]),
        ("reusable_1x1/win3x3/max8", 0, &[(1, 1, 0x050c5d1f, 0x117697cd, 0x117697cd)]),
        ("reusable_1x1/win5x5/max2", 0, &[(1, 1, 0x050c5d1f, 0x117697cd, 0x117697cd)]),
        ("reusable_1x1/win20x20/max5", 0, &[(1, 1, 0x050c5d1f, 0x117697cd, 0x117697cd)]),
        ("reusable_1x1/win15x9/max4", 0, &[(1, 1, 0x050c5d1f, 0x117697cd, 0x117697cd)]),
        ("reusable_1x7/win21x21/max3", 0, &[(1, 7, 0x250bff18, 0xa261a1e7, 0x3538be3d)]),
        ("reusable_1x7/win21x21/max0", 0, &[(1, 7, 0x250bff18, 0xa261a1e7, 0x3538be3d)]),
        ("reusable_1x7/win21x21/max8", 0, &[(1, 7, 0x250bff18, 0xa261a1e7, 0x3538be3d)]),
        ("reusable_1x7/win3x3/max8", 0, &[(1, 7, 0x250bff18, 0xa261a1e7, 0x3538be3d)]),
        ("reusable_1x7/win5x5/max2", 0, &[(1, 7, 0x250bff18, 0xa261a1e7, 0x3538be3d)]),
        ("reusable_1x7/win20x20/max5", 0, &[(1, 7, 0x250bff18, 0xa261a1e7, 0x3538be3d)]),
        ("reusable_1x7/win15x9/max4", 0, &[(1, 7, 0x250bff18, 0xa261a1e7, 0x3538be3d)]),
        ("reusable_3x2/win21x21/max3", 0, &[(3, 2, 0x28dfd1c9, 0xe23c62b5, 0x7f4183ed)]),
        ("reusable_3x2/win21x21/max0", 0, &[(3, 2, 0x28dfd1c9, 0xe23c62b5, 0x7f4183ed)]),
        ("reusable_3x2/win21x21/max8", 0, &[(3, 2, 0x28dfd1c9, 0xe23c62b5, 0x7f4183ed)]),
        ("reusable_3x2/win3x3/max8", 0, &[(3, 2, 0x28dfd1c9, 0xe23c62b5, 0x7f4183ed)]),
        ("reusable_3x2/win5x5/max2", 0, &[(3, 2, 0x28dfd1c9, 0xe23c62b5, 0x7f4183ed)]),
        ("reusable_3x2/win20x20/max5", 0, &[(3, 2, 0x28dfd1c9, 0xe23c62b5, 0x7f4183ed)]),
        ("reusable_3x2/win15x9/max4", 0, &[(3, 2, 0x28dfd1c9, 0xe23c62b5, 0x7f4183ed)]),
        ("reusable_7x5/win21x21/max3", 0, &[(7, 5, 0xe1e949ec, 0x5ecefcf7, 0x4827177a)]),
        ("reusable_7x5/win21x21/max0", 0, &[(7, 5, 0xe1e949ec, 0x5ecefcf7, 0x4827177a)]),
        ("reusable_7x5/win21x21/max8", 0, &[(7, 5, 0xe1e949ec, 0x5ecefcf7, 0x4827177a)]),
        ("reusable_7x5/win3x3/max8", 0, &[(7, 5, 0xe1e949ec, 0x5ecefcf7, 0x4827177a)]),
        ("reusable_7x5/win5x5/max2", 0, &[(7, 5, 0xe1e949ec, 0x5ecefcf7, 0x4827177a)]),
        ("reusable_7x5/win20x20/max5", 0, &[(7, 5, 0xe1e949ec, 0x5ecefcf7, 0x4827177a)]),
        ("reusable_7x5/win15x9/max4", 0, &[(7, 5, 0xe1e949ec, 0x5ecefcf7, 0x4827177a)]),
    ];

    /// Pixel digests of the inputs (`video_pyramid.py` / `video_lk.py` "inputs").
    #[rustfmt::skip]
    pub const INPUT_DIGESTS: [(&str, u32); 10] = [
        ("cell_320x240", 0xbc292803),
        ("textured_96x128_s11", 0xaae3288a),
        ("reusable_33x49", 0x5d187c70),
        ("reusable_9x9", 0xef849369),
        ("reusable_1x1", 0x050c5d1f),
        ("reusable_1x7", 0x250bff18),
        ("reusable_3x2", 0x28dfd1c9),
        ("reusable_7x5", 0xe1e949ec),
        ("cell_320x240_shift_10_0", 0xffb4fec3),
        ("textured_96x128_s11_shift_2.25_-0.75", 0xb6f918e2),
    ];
    pub const REUSABLE_33X49_SHIFT1_DIGEST: u32 = 0x1f2b087a;

    /// `video_lk.py` "grid163": 150 points, 21x21, 4 levels, (COUNT|EPS, 30, 0.01), 1e-4,
    /// `OPTFLOW_USE_INITIAL_FLOW` with `nextPts = prevPts`; one constant per shift.
    #[rustfmt::skip]
    pub const GRID163_SHIFT_0_0: [FlowRow; 150] = [
        (40.0, 40.0, 1, 0.0), (56.0, 40.0, 1, 0.0),
        (72.0, 40.0, 1, 0.0), (88.0, 40.0, 1, 0.0),
        (104.0, 40.0, 1, 0.0), (120.0, 40.0, 1, 0.0),
        (136.0, 40.0, 1, 0.0), (152.0, 40.0, 1, 0.0),
        (168.0, 40.0, 1, 0.0), (184.0, 40.0, 1, 0.0),
        (200.0, 40.0, 1, 0.0), (216.0, 40.0, 1, 0.0),
        (232.0, 40.0, 1, 0.0), (248.0, 40.0, 1, 0.0),
        (264.0, 40.0, 1, 0.0), (40.0, 56.0, 1, 0.0),
        (56.0, 56.0, 1, 0.0), (72.0, 56.0, 1, 0.0),
        (88.0, 56.0, 1, 0.0), (104.0, 56.0, 1, 0.0),
        (120.0, 56.0, 1, 0.0), (136.0, 56.0, 1, 0.0),
        (152.0, 56.0, 1, 0.0), (168.0, 56.0, 1, 0.0),
        (184.0, 56.0, 1, 0.0), (200.0, 56.0, 1, 0.0),
        (216.0, 56.0, 1, 0.0), (232.0, 56.0, 1, 0.0),
        (248.0, 56.0, 1, 0.0), (264.0, 56.0, 1, 0.0),
        (40.0, 72.0, 1, 0.0), (56.0, 72.0, 1, 0.0),
        (72.0, 72.0, 1, 0.0), (88.0, 72.0, 1, 0.0),
        (104.0, 72.0, 1, 0.0), (120.0, 72.0, 1, 0.0),
        (136.0, 72.0, 1, 0.0), (152.0, 72.0, 1, 0.0),
        (168.0, 72.0, 1, 0.0), (184.0, 72.0, 1, 0.0),
        (200.0, 72.0, 1, 0.0), (216.0, 72.0, 1, 0.0),
        (232.0, 72.0, 1, 0.0), (248.0, 72.0, 1, 0.0),
        (264.0, 72.0, 1, 0.0), (40.0, 88.0, 1, 0.0),
        (56.0, 88.0, 1, 0.0), (72.0, 88.0, 1, 0.0),
        (88.0, 88.0, 1, 0.0), (104.0, 88.0, 1, 0.0),
        (120.0, 88.0, 1, 0.0), (136.0, 88.0, 1, 0.0),
        (152.0, 88.0, 1, 0.0), (168.0, 88.0, 1, 0.0),
        (184.0, 88.0, 1, 0.0), (200.0, 88.0, 1, 0.0),
        (216.0, 88.0, 1, 0.0), (232.0, 88.0, 1, 0.0),
        (248.0, 88.0, 1, 0.0), (264.0, 88.0, 1, 0.0),
        (40.0, 104.0, 1, 0.0), (56.0, 104.0, 1, 0.0),
        (72.0, 104.0, 1, 0.0), (88.0, 104.0, 1, 0.0),
        (104.0, 104.0, 1, 0.0), (120.0, 104.0, 1, 0.0),
        (136.0, 104.0, 1, 0.0), (152.0, 104.0, 1, 0.0),
        (168.0, 104.0, 1, 0.0), (184.0, 104.0, 1, 0.0),
        (200.0, 104.0, 1, 0.0), (216.0, 104.0, 1, 0.0),
        (232.0, 104.0, 1, 0.0), (248.0, 104.0, 1, 0.0),
        (264.0, 104.0, 1, 0.0), (40.0, 120.0, 1, 0.0),
        (56.0, 120.0, 1, 0.0), (72.0, 120.0, 1, 0.0),
        (88.0, 120.0, 1, 0.0), (104.0, 120.0, 1, 0.0),
        (120.0, 120.0, 1, 0.0), (136.0, 120.0, 1, 0.0),
        (152.0, 120.0, 1, 0.0), (168.0, 120.0, 1, 0.0),
        (184.0, 120.0, 1, 0.0), (200.0, 120.0, 1, 0.0),
        (216.0, 120.0, 1, 0.0), (232.0, 120.0, 1, 0.0),
        (248.0, 120.0, 1, 0.0), (264.0, 120.0, 1, 0.0),
        (40.0, 136.0, 1, 0.0), (56.0, 136.0, 1, 0.0),
        (72.0, 136.0, 1, 0.0), (88.0, 136.0, 1, 0.0),
        (104.0, 136.0, 1, 0.0), (120.0, 136.0, 1, 0.0),
        (136.0, 136.0, 1, 0.0), (152.0, 136.0, 1, 0.0),
        (168.0, 136.0, 1, 0.0), (184.0, 136.0, 1, 0.0),
        (200.0, 136.0, 1, 0.0), (216.0, 136.0, 1, 0.0),
        (232.0, 136.0, 1, 0.0), (248.0, 136.0, 1, 0.0),
        (264.0, 136.0, 1, 0.0), (40.0, 152.0, 1, 0.0),
        (56.0, 152.0, 1, 0.0), (72.0, 152.0, 1, 0.0),
        (88.0, 152.0, 1, 0.0), (104.0, 152.0, 1, 0.0),
        (120.0, 152.0, 1, 0.0), (136.0, 152.0, 1, 0.0),
        (152.0, 152.0, 1, 0.0), (168.0, 152.0, 1, 0.0),
        (184.0, 152.0, 1, 0.0), (200.0, 152.0, 1, 0.0),
        (216.0, 152.0, 1, 0.0), (232.0, 152.0, 1, 0.0),
        (248.0, 152.0, 1, 0.0), (264.0, 152.0, 1, 0.0),
        (40.0, 168.0, 1, 0.0), (56.0, 168.0, 1, 0.0),
        (72.0, 168.0, 1, 0.0), (88.0, 168.0, 1, 0.0),
        (104.0, 168.0, 1, 0.0), (120.0, 168.0, 1, 0.0),
        (136.0, 168.0, 1, 0.0), (152.0, 168.0, 1, 0.0),
        (168.0, 168.0, 1, 0.0), (184.0, 168.0, 1, 0.0),
        (200.0, 168.0, 1, 0.0), (216.0, 168.0, 1, 0.0),
        (232.0, 168.0, 1, 0.0), (248.0, 168.0, 1, 0.0),
        (264.0, 168.0, 1, 0.0), (40.0, 184.0, 1, 0.0),
        (56.0, 184.0, 1, 0.0), (72.0, 184.0, 1, 0.0),
        (88.0, 184.0, 1, 0.0), (104.0, 184.0, 1, 0.0),
        (120.0, 184.0, 1, 0.0), (136.0, 184.0, 1, 0.0),
        (152.0, 184.0, 1, 0.0), (168.0, 184.0, 1, 0.0),
        (184.0, 184.0, 1, 0.0), (200.0, 184.0, 1, 0.0),
        (216.0, 184.0, 1, 0.0), (232.0, 184.0, 1, 0.0),
        (248.0, 184.0, 1, 0.0), (264.0, 184.0, 1, 0.0),
    ];
    #[rustfmt::skip]
    pub const GRID163_SHIFT_10_0: [FlowRow; 150] = [
        (50.00011, 40.000153, 1, 0.0), (65.999886, 39.999897, 1, 0.0),
        (81.99999, 39.99988, 1, 0.0), (97.99987, 40.000076, 1, 0.0),
        (114.000015, 40.000286, 1, 0.006802721), (129.99998, 40.00032, 1, 0.0063775512),
        (145.99979, 40.000057, 1, 0.0), (161.99991, 39.999832, 1, 0.0),
        (178.00008, 40.00007, 1, 0.0), (193.99988, 40.000015, 1, 0.0),
        (210.00009, 40.00005, 1, 0.0), (225.99977, 40.000137, 1, 0.0019132653),
        (241.99991, 39.99999, 1, 0.0), (258.0001, 40.00006, 1, 0.0),
        (274.00006, 40.0, 1, 0.0), (49.999893, 56.00011, 1, 0.0),
        (66.0002, 55.999943, 1, 0.0), (81.99998, 55.999912, 1, 0.0),
        (97.99986, 56.00002, 1, 0.0), (114.00006, 55.999897, 1, 0.0),
        (130.00003, 55.999912, 1, 0.0), (145.99988, 56.000076, 1, 0.0),
        (161.99985, 56.000042, 1, 0.0), (177.99988, 55.999954, 1, 0.0),
        (194.00008, 55.99993, 1, 0.0), (210.0001, 56.00013, 1, 0.0),
        (225.99997, 56.000134, 1, 0.0), (242.00002, 55.99988, 1, 0.0),
        (258.00012, 55.999874, 1, 0.0), (273.99994, 55.999844, 1, 0.0),
        (50.00007, 72.00005, 1, 0.0), (66.00009, 71.9999, 1, 0.0),
        (82.00012, 71.99988, 1, 0.0), (98.00008, 72.00018, 1, 0.0),
        (114.00015, 71.99988, 1, 0.0), (129.99985, 72.000046, 1, 0.0),
        (146.00021, 71.99994, 1, 0.0), (161.9999, 72.00015, 1, 0.0),
        (178.00012, 71.99994, 1, 0.0), (194.00012, 72.00011, 1, 0.0),
        (209.99994, 72.000046, 1, 0.0), (226.00009, 72.000015, 1, 0.0),
        (242.00009, 71.99992, 1, 0.0), (257.99982, 72.00023, 1, 0.0017715419),
        (274.00006, 71.999886, 1, 0.0), (49.999866, 88.00008, 1, 0.0),
        (66.000015, 88.00004, 1, 0.0), (82.0, 88.00013, 1, 0.0),
        (97.999916, 87.999916, 1, 0.0), (113.99986, 87.99997, 1, 0.0),
        (130.00006, 87.999725, 1, 0.0017006802), (145.99988, 87.999886, 1, 0.0),
        (162.00012, 87.99996, 1, 0.0), (178.00006, 87.999664, 1, 0.005527211),
        (193.99986, 87.99985, 1, 0.0), (210.00015, 87.99988, 1, 0.0),
        (226.00009, 88.000046, 1, 0.0), (241.99995, 87.99991, 1, 0.0),
        (257.99994, 87.99984, 1, 0.0), (274.0001, 87.999886, 1, 0.0),
        (49.999935, 103.999825, 1, 0.0), (66.00008, 104.00012, 1, 0.0),
        (81.99999, 104.00006, 1, 0.0), (97.99986, 104.00013, 1, 0.0),
        (113.99997, 104.000015, 1, 0.0), (129.99998, 104.00003, 1, 0.0),
        (145.99997, 103.99985, 1, 0.0), (161.99998, 104.00004, 1, 0.0),
        (177.99994, 103.99989, 1, 0.0), (194.0001, 104.00008, 1, 0.0),
        (209.9998, 104.00005, 1, 0.0), (225.99995, 103.99985, 1, 0.0),
        (241.9999, 104.00001, 1, 0.0), (257.9999, 104.00012, 1, 0.0),
        (273.99997, 103.99992, 1, 0.0), (49.999966, 119.99984, 1, 0.0),
        (65.99985, 120.00009, 1, 0.0), (81.99989, 120.000145, 1, 0.0),
        (98.00004, 120.00004, 1, 0.0), (114.0, 119.99984, 1, 0.0),
        (130.00009, 119.99984, 1, 0.0), (146.00012, 119.999954, 1, 0.0),
        (162.00018, 119.99971, 1, 0.006660998), (178.00012, 119.99995, 1, 0.0),
        (193.99991, 120.000046, 1, 0.0), (209.99988, 119.99994, 1, 0.0),
        (225.99995, 120.0002, 1, 0.0), (242.00003, 119.99997, 1, 0.0),
        (257.99976, 120.00003, 1, 0.001984127), (273.99997, 120.0, 1, 0.0),
        (49.99994, 135.99994, 1, 0.0), (66.00008, 136.00009, 1, 0.0),
        (81.999954, 136.00002, 1, 0.0), (97.99992, 136.00012, 1, 0.0),
        (113.99986, 135.99985, 1, 0.0), (130.00009, 135.99979, 1, 0.00042517006),
        (146.00017, 135.99997, 1, 0.0), (162.00005, 136.00006, 1, 0.0),
        (177.99997, 136.00002, 1, 0.0), (193.99979, 135.99994, 1, 0.0),
        (209.99982, 136.00009, 1, 0.0), (226.0001, 136.00015, 1, 0.0),
        (242.00014, 135.99997, 1, 0.0), (257.99988, 135.99983, 1, 0.00042517006),
        (274.00018, 135.99982, 1, 0.00042517006), (50.000183, 152.00005, 1, 0.0),
        (66.00014, 151.99988, 1, 0.0), (81.99994, 151.99991, 1, 0.0),
        (98.000015, 152.00012, 1, 0.0), (114.00009, 152.00012, 1, 0.0006377551),
        (129.99997, 152.0, 1, 0.0), (145.99988, 151.99991, 1, 0.0),
        (162.00009, 152.00002, 1, 0.0), (177.99988, 151.99998, 1, 0.0),
        (193.9999, 152.0001, 1, 0.0), (209.99985, 152.00008, 1, 0.0),
        (226.00006, 152.00017, 1, 0.0), (242.00006, 151.9999, 1, 0.0),
        (257.99982, 152.00005, 1, 0.0), (274.00003, 152.00009, 1, 0.0),
        (50.00008, 167.99998, 1, 0.0), (66.00011, 168.00012, 1, 0.0),
        (81.99994, 167.99983, 1, 0.0), (97.99987, 167.99985, 1, 0.0),
        (114.00018, 168.00003, 1, 0.0017006802), (129.99988, 167.99995, 1, 0.0),
        (145.99986, 167.99988, 1, 0.0), (162.00003, 167.99994, 1, 0.0),
        (178.0001, 168.0001, 1, 0.0), (194.00012, 167.99985, 1, 0.0),
        (209.99982, 168.00008, 1, 0.0), (225.99988, 167.9999, 1, 0.0),
        (242.00006, 167.99994, 1, 0.0), (257.9999, 167.99995, 1, 0.0),
        (274.0001, 168.00024, 1, 0.0067318594), (50.000217, 183.99995, 1, 0.0),
        (65.99994, 183.99983, 1, 0.0), (82.000244, 184.00008, 1, 0.002409297),
        (98.00002, 184.00006, 1, 0.0), (113.99998, 183.9999, 1, 0.0),
        (130.0, 183.99992, 1, 0.0), (145.99994, 184.00006, 1, 0.0),
        (162.00009, 183.99992, 1, 0.0), (177.99992, 184.00009, 1, 0.0),
        (194.0, 184.0001, 1, 0.0), (210.00005, 184.00015, 1, 0.0),
        (226.0, 184.00009, 1, 0.0), (241.99992, 183.99986, 1, 0.0),
        (257.99994, 183.99994, 1, 0.0), (273.99997, 183.99997, 1, 0.0),
    ];
    #[rustfmt::skip]
    pub const GRID163_SHIFT_0_10: [FlowRow; 150] = [
        (39.999844, 50.00006, 1, 0.0), (55.999794, 50.000107, 1, 0.00042517006),
        (72.0001, 50.000076, 1, 0.0), (88.00005, 49.99996, 1, 0.0),
        (104.0, 50.000057, 1, 0.0), (119.99993, 50.0001, 1, 0.0),
        (136.00009, 50.000137, 1, 0.0007086168), (152.00009, 49.99988, 1, 0.0),
        (167.99994, 50.00007, 1, 0.0), (184.00012, 50.000107, 1, 0.0),
        (199.99998, 49.99986, 1, 0.0), (215.99979, 49.999878, 1, 0.00028344672),
        (231.99986, 49.999973, 1, 0.0), (248.00003, 49.99996, 1, 0.0),
        (401.88522, 34.24678, 0, 0.0), (40.00006, 65.99997, 1, 0.0),
        (55.99986, 65.999916, 1, 0.0), (71.999916, 66.0001, 1, 0.0),
        (87.999985, 65.99982, 1, 0.0), (103.99998, 66.00014, 1, 0.0),
        (120.000145, 66.00002, 1, 0.0), (136.00006, 66.00018, 1, 0.0),
        (152.0001, 65.99988, 1, 0.0), (167.99995, 66.00018, 1, 0.0),
        (183.99985, 66.00027, 1, 0.007440476), (200.0001, 66.00011, 1, 0.0),
        (216.00014, 66.00007, 1, 0.0), (231.99994, 65.99991, 1, 0.0),
        (247.99985, 66.00022, 1, 0.0020549886), (263.99994, 66.00013, 1, 0.0),
        (39.999855, 81.999855, 1, 0.0), (56.000153, 81.999916, 1, 0.0),
        (71.99993, 81.99994, 1, 0.0), (87.99993, 82.00014, 1, 0.0),
        (104.00002, 81.99987, 1, 0.0), (119.99987, 81.99999, 1, 0.0),
        (135.99983, 82.00012, 1, 0.00056689343), (152.00008, 82.00008, 1, 0.0),
        (168.00005, 81.99989, 1, 0.0), (184.0, 81.99989, 1, 0.0),
        (200.00012, 82.0001, 1, 0.0), (216.00002, 81.9999, 1, 0.0),
        (232.00008, 82.00003, 1, 0.0), (248.00005, 81.99992, 1, 0.0),
        (263.99985, 82.000015, 1, 0.0), (40.0001, 97.99995, 1, 0.0),
        (56.00016, 97.99993, 1, 0.0), (72.000084, 97.99993, 1, 0.0),
        (88.00007, 98.00003, 1, 0.0), (104.000015, 97.99977, 1, 0.0011337869),
        (120.00016, 97.99989, 1, 0.0), (136.00003, 98.00012, 1, 0.0),
        (151.99992, 97.99996, 1, 0.0), (168.00005, 97.999916, 1, 0.0),
        (184.00002, 97.999756, 1, 0.0010629252), (199.99994, 97.999886, 1, 0.0),
        (215.99995, 97.99988, 1, 0.0), (231.99985, 98.00011, 1, 0.0),
        (247.99983, 97.99994, 1, 0.0), (263.99988, 97.99998, 1, 0.0),
        (39.9999, 113.99998, 1, 0.0), (55.999928, 114.000145, 1, 0.0),
        (72.00018, 113.99989, 1, 0.00042517006), (88.00004, 113.99997, 1, 0.0),
        (103.999916, 114.000145, 1, 0.0), (119.99984, 113.999825, 1, 0.0006377551),
        (136.00008, 113.99997, 1, 0.0), (151.99992, 114.00006, 1, 0.0),
        (168.00009, 114.00002, 1, 0.0), (183.99997, 113.9999, 1, 0.0),
        (199.99991, 114.00016, 1, 0.0), (215.99985, 114.00005, 1, 0.0),
        (232.00012, 114.00009, 1, 0.0006377551), (247.99985, 113.999825, 1, 0.0006377551),
        (264.0, 113.999985, 1, 0.0), (40.000195, 130.0, 1, 0.0),
        (56.0001, 129.99994, 1, 0.0), (71.99989, 129.99991, 1, 0.0),
        (88.00013, 129.99991, 1, 0.0), (103.9999, 130.00015, 1, 0.0),
        (119.99986, 130.00015, 1, 0.0006377551), (135.99988, 130.00009, 1, 0.0),
        (152.00008, 130.00002, 1, 0.0), (168.00008, 129.99991, 1, 0.0),
        (184.0001, 129.99982, 1, 0.00042517006), (199.99983, 130.0, 1, 0.0),
        (215.99985, 129.99973, 1, 0.0022675737), (232.00002, 130.00012, 1, 0.0),
        (247.9999, 129.9999, 1, 0.0), (264.00012, 130.00003, 1, 0.0),
        (40.000053, 145.99995, 1, 0.0), (56.00012, 145.99991, 1, 0.0),
        (72.00016, 146.00006, 1, 0.0), (87.99993, 146.00009, 1, 0.0),
        (104.00019, 145.99982, 1, 0.00042517006), (119.99981, 146.00012, 1, 0.0006377551),
        (136.00006, 146.0001, 1, 0.0), (152.00003, 145.99988, 1, 0.0),
        (167.99994, 146.00006, 1, 0.0), (184.00005, 146.00014, 1, 0.0),
        (199.99985, 145.99986, 1, 0.0), (216.00017, 145.9999, 1, 0.0),
        (231.99995, 145.99991, 1, 0.0), (247.99995, 146.00006, 1, 0.0),
        (264.0, 145.99992, 1, 0.0), (39.999966, 162.00012, 1, 0.0),
        (55.999924, 161.99982, 1, 0.0), (72.000046, 161.99998, 1, 0.0),
        (87.999916, 161.9998, 1, 0.0), (103.999985, 161.9999, 1, 0.0),
        (119.999954, 161.9999, 1, 0.0), (136.0, 162.00018, 1, 0.0),
        (151.9998, 162.00008, 1, 0.0), (167.99994, 162.00006, 1, 0.0),
        (183.99988, 161.99988, 1, 0.0), (199.99982, 161.99995, 1, 0.0),
        (215.99992, 162.00014, 1, 0.0), (231.9999, 161.99992, 1, 0.0),
        (247.99985, 162.00009, 1, 0.0), (263.99994, 162.00012, 1, 0.0),
        (40.000046, 178.0001, 1, 0.0), (56.000095, 178.00002, 1, 0.0),
        (72.00004, 178.00009, 1, 0.0), (87.99996, 177.99982, 1, 0.0),
        (104.00012, 177.99983, 1, 0.0006377551), (120.00009, 178.0, 1, 0.0),
        (135.99988, 178.00005, 1, 0.0), (151.99983, 177.99983, 1, 0.0008503401),
        (168.00005, 178.00015, 1, 0.0), (183.9999, 177.99986, 1, 0.0),
        (200.0002, 177.99995, 1, 0.0), (215.9999, 178.00005, 1, 0.0),
        (232.00012, 178.00008, 1, 0.0), (248.00005, 177.9999, 1, 0.0),
        (264.00006, 178.00017, 1, 0.0), (39.999992, 193.99997, 1, 0.0),
        (56.000183, 194.00002, 1, 0.0), (72.00017, 193.99988, 1, 0.00042517006),
        (87.99986, 193.99986, 1, 0.0), (104.000015, 193.99991, 1, 0.0),
        (120.000084, 194.0, 1, 0.0), (136.00003, 194.0002, 1, 0.001984127),
        (151.99994, 193.9998, 1, 0.0), (168.00002, 194.00008, 1, 0.0),
        (183.99985, 194.00017, 1, 0.00042517006), (200.00003, 194.00018, 1, 0.0018424037),
        (215.9999, 194.00006, 1, 0.0), (232.00003, 193.99991, 1, 0.0),
        (247.99992, 194.00002, 1, 0.0), (263.9999, 194.00006, 1, 0.0),
    ];
    #[rustfmt::skip]
    pub const GRID163_SHIFT_10_10: [FlowRow; 150] = [
        (50.000225, 49.99999, 1, 0.0015589569), (65.99992, 50.000076, 1, 0.0),
        (82.00016, 49.999924, 1, 0.0), (98.000114, 49.999992, 1, 0.0),
        (-32.41494, -48.47397, 0, 0.0), (129.99991, 50.00008, 1, 0.0),
        (146.00008, 49.999817, 1, 0.0), (162.0, 49.99988, 1, 0.0),
        (177.99991, 21.49754, 1, 20.231293), (160.58154, -90.38762, 0, 0.0),
        (210.0, 49.999844, 1, 0.0), (226.00008, 49.99996, 1, 0.0),
        (242.00003, 49.999928, 1, 0.0), (258.0, 50.000134, 1, 0.0),
        (401.34393, 35.231365, 0, 0.0), (50.00002, 65.99992, 1, 0.0),
        (66.000015, 65.99988, 1, 0.0), (82.000084, 65.99986, 1, 0.0),
        (97.99988, 65.999886, 1, 0.0), (114.00015, 65.99982, 1, 0.0006377551),
        (130.0, 66.00012, 1, 0.0), (146.00003, 66.000275, 1, 0.005952381),
        (161.99982, 66.00005, 1, 0.0), (178.00014, 66.000015, 1, 0.0),
        (121.23083, -33.703552, 0, 0.007440476), (209.99986, 66.0002, 1, 0.001984127),
        (226.0, 66.00023, 1, 0.0014880953), (242.00009, 66.00024, 1, 0.002196712),
        (257.99985, 66.000046, 1, 0.0), (273.99982, 66.0001, 1, 0.00042517006),
        (50.00007, 81.99991, 1, 0.0), (65.99991, 82.0001, 1, 0.0),
        (82.00008, 82.000175, 1, 0.0), (97.99997, 81.99991, 1, 0.0),
        (114.000015, 81.99999, 1, 0.0), (130.00015, 81.99992, 1, 0.0),
        (145.99992, 82.000084, 1, 0.0), (162.00008, 82.000145, 1, 0.00042517006),
        (178.00018, 82.000046, 1, 0.0), (148.90121, -40.382996, 0, 0.0),
        (209.99992, 82.00003, 1, 0.0), (225.9999, 82.00019, 1, 0.00028344672),
        (242.00003, 82.00003, 1, 0.0), (258.00006, 82.00018, 1, 0.0),
        (273.99988, 81.99997, 1, 0.0), (49.9999, 97.99996, 1, 0.0),
        (66.00006, 97.99993, 1, 0.0), (82.00001, 98.00009, 1, 0.0),
        (98.000015, 98.00006, 1, 0.0), (113.99981, 97.999916, 1, 0.0),
        (129.99976, 97.99979, 1, 0.0028344672), (145.99994, 98.0001, 1, 0.0),
        (161.99983, 97.99992, 1, 0.0), (177.9999, 98.00005, 1, 0.0),
        (194.00006, 97.99996, 1, 0.0), (210.00017, 97.999855, 1, 0.00028344672),
        (225.99985, 97.99986, 1, 0.0), (241.99992, 97.99984, 1, 0.0),
        (257.99997, 97.99989, 1, 0.0), (274.00012, 98.00007, 1, 0.0),
        (49.99981, 113.99991, 1, 0.0), (66.000145, 114.00011, 1, 0.0),
        (81.99992, 114.00006, 1, 0.0), (97.99994, 114.00003, 1, 0.0),
        (113.99987, 114.000145, 1, 0.0), (129.9999, 114.000084, 1, 0.0),
        (145.99997, 114.000046, 1, 0.0), (161.99985, 114.000114, 1, 0.0),
        (178.0001, 114.00011, 1, 0.0), (194.00005, 113.99984, 1, 0.0),
        (210.0001, 114.00009, 1, 0.0), (225.99983, 114.00014, 1, 0.00042517006),
        (242.00024, 114.00017, 1, 0.0026218821), (258.00006, 114.00014, 1, 0.0),
        (273.9998, 114.000114, 1, 0.0006377551), (49.999813, 130.00015, 1, 0.0006377551),
        (65.99992, 130.00009, 1, 0.0), (82.000015, 129.99985, 1, 0.0),
        (97.99991, 130.00014, 1, 0.0), (114.00019, 129.99994, 1, 0.0),
        (129.99997, 130.0, 1, 0.0), (146.00008, 130.00009, 1, 0.0),
        (162.00012, 130.00012, 1, 0.0), (177.99986, 129.99995, 1, 0.0),
        (193.99991, 129.99992, 1, 0.0), (209.99988, 129.99998, 1, 0.0),
        (225.99998, 130.00021, 1, 0.0011337869), (242.00017, 129.99988, 1, 0.00042517006),
        (257.99988, 130.00003, 1, 0.0), (274.0001, 129.9999, 1, 0.0),
        (50.000183, 146.00005, 1, 0.0), (65.99991, 146.00008, 1, 0.0),
        (81.999886, 146.0, 1, 0.0), (97.99995, 145.99988, 1, 0.0),
        (114.00019, 145.99983, 1, 0.00042517006), (129.99988, 145.99974, 1, 0.0018424037),
        (145.99985, 145.99998, 1, 0.0), (162.00017, 145.9999, 1, 0.0),
        (178.00015, 145.99994, 1, 0.0), (193.99985, 146.00009, 1, 0.0),
        (210.00012, 146.00003, 1, 0.0), (225.99983, 146.0001, 1, 0.0),
        (241.9999, 146.00005, 1, 0.0), (257.99994, 146.0001, 1, 0.0),
        (273.99994, 145.99991, 1, 0.0), (50.000168, 161.9999, 1, 0.0),
        (65.99998, 162.00014, 1, 0.0), (81.99997, 162.00018, 1, 0.0017715419),
        (97.99989, 162.00009, 1, 0.0), (114.000084, 161.99992, 1, 0.0),
        (129.99991, 162.00026, 1, 0.009566327), (146.0, 161.9999, 1, 0.0),
        (161.99988, 162.00008, 1, 0.0), (177.9998, 162.00002, 1, 0.0),
        (194.00014, 161.99995, 1, 0.0), (210.00006, 162.0, 1, 0.0),
        (225.99997, 161.9998, 1, 0.0), (242.0001, 161.99994, 1, 0.0),
        (257.99994, 162.00009, 1, 0.0), (273.99994, 162.00009, 1, 0.0),
        (50.000072, 178.00008, 1, 0.0), (66.00021, 178.0, 1, 0.0020549886),
        (81.99994, 178.0, 1, 0.0), (97.99991, 178.00005, 1, 0.0),
        (114.00002, 177.99995, 1, 0.0), (130.00003, 178.00012, 1, 0.0),
        (146.0, 178.00029, 1, 0.0073696147), (161.9998, 178.00009, 1, 0.0008503401),
        (177.99995, 178.00018, 1, 0.0), (194.0, 178.00018, 1, 0.0),
        (210.0001, 178.00008, 1, 0.0), (225.99986, 178.0, 1, 0.0),
        (241.99995, 177.99997, 1, 0.0), (257.99988, 178.00017, 1, 0.0006377551),
        (274.00003, 177.99994, 1, 0.0), (49.999832, 194.00002, 1, 0.0),
        (65.999794, 193.99986, 1, 0.0006377551), (82.00016, 193.99976, 1, 0.003047052),
        (97.999916, 193.99985, 1, 0.0), (114.000145, 194.00012, 1, 0.0),
        (129.99992, 193.9999, 1, 0.0), (145.99971, 194.00017, 1, 0.0127551025),
        (161.99998, 194.00005, 1, 0.0), (177.99992, 194.00008, 1, 0.0),
        (193.99991, 194.0, 1, 0.0), (210.00006, 194.00014, 1, 0.0),
        (226.00015, 193.99985, 1, 0.0006377551), (242.00017, 193.99997, 1, 0.0),
        (257.99988, 194.00015, 1, 0.0), (273.99988, 194.00014, 1, 0.0),
    ];

    #[rustfmt::skip]
    pub const VAR_DEFAULT: [FlowRow; 30] = [
        (50.00011, 40.000153, 1, 0.0), (129.99998, 40.00032, 1, 0.0063775512),
        (210.00009, 40.00005, 1, 0.0), (49.999893, 56.00011, 1, 0.0),
        (130.00003, 55.999912, 1, 0.0), (210.0001, 56.00013, 1, 0.0),
        (50.00007, 72.00005, 1, 0.0), (129.99985, 72.000046, 1, 0.0),
        (209.99994, 72.000046, 1, 0.0), (49.999866, 88.00008, 1, 0.0),
        (130.00006, 87.999725, 1, 0.0017006802), (210.00015, 87.99988, 1, 0.0),
        (49.999935, 103.999825, 1, 0.0), (129.99998, 104.00003, 1, 0.0),
        (209.9998, 104.00005, 1, 0.0), (49.999966, 119.99984, 1, 0.0),
        (130.00009, 119.99984, 1, 0.0), (209.99988, 119.99994, 1, 0.0),
        (49.99994, 135.99994, 1, 0.0), (130.00009, 135.99979, 1, 0.00042517006),
        (209.99982, 136.00009, 1, 0.0), (50.000183, 152.00005, 1, 0.0),
        (129.99997, 152.0, 1, 0.0), (209.99985, 152.00008, 1, 0.0),
        (50.00008, 167.99998, 1, 0.0), (129.99988, 167.99995, 1, 0.0),
        (209.99982, 168.00008, 1, 0.0), (50.000217, 183.99995, 1, 0.0),
        (130.0, 183.99992, 1, 0.0), (210.00005, 184.00015, 1, 0.0),
    ];
    #[rustfmt::skip]
    pub const VAR_MIN_EIG_OUTPUT: [FlowRow; 30] = [
        (50.00011, 40.000153, 1, 0.63241655), (129.99998, 40.00032, 1, 0.63241655),
        (210.00009, 40.00005, 1, 0.7321734), (49.999893, 56.00011, 1, 0.7899947),
        (130.00003, 55.999912, 1, 0.7912569), (210.0001, 56.00013, 1, 0.68229496),
        (50.00007, 72.00005, 1, 0.5064759), (129.99985, 72.000046, 1, 0.50563556),
        (209.99994, 72.000046, 1, 0.58631605), (49.999866, 88.00008, 1, 0.6644056),
        (130.00006, 87.999725, 1, 0.66398376), (210.00015, 87.99988, 1, 0.7217519),
        (49.999935, 103.999825, 1, 0.8230357), (129.99998, 104.00003, 1, 0.82135165),
        (209.9998, 104.00005, 1, 0.67187345), (49.999966, 119.99984, 1, 0.79104656),
        (130.00009, 119.99984, 1, 0.7897843), (209.99988, 119.99994, 1, 0.68229496),
        (49.99994, 135.99994, 1, 0.7910466), (130.00009, 135.99979, 1, 0.7897843),
        (209.99982, 136.00009, 1, 0.68229496), (50.000183, 152.00005, 1, 0.9486149),
        (129.99997, 152.0, 1, 0.94812554), (209.99985, 152.00008, 1, 0.63241655),
        (50.00008, 167.99998, 1, 0.8230356), (129.99988, 167.99995, 1, 0.82135165),
        (209.99982, 168.00008, 1, 0.6718736), (50.000217, 183.99995, 1, 0.94842356),
        (130.0, 183.99992, 1, 0.94842356), (210.00005, 184.00015, 1, 0.63241655),
    ];
    #[rustfmt::skip]
    pub const VAR_WIN_20X20: [FlowRow; 30] = [
        (49.99973, 39.999355, 1, 0.016875), (130.00058, 39.998978, 1, 0.02484375),
        (209.99982, 39.999634, 1, 0.0075), (49.999294, 55.99993, 1, 0.01265625),
        (129.9992, 55.999573, 1, 0.02875), (210.00015, 56.00044, 1, 0.0090625),
        (50.00002, 72.000336, 1, 0.0034375), (130.0008, 71.99986, 1, 0.024609376),
        (209.99982, 71.999664, 1, 0.00484375), (50.0003, 88.00007, 1, 0.00984375),
        (129.99968, 87.99986, 1, 0.01078125), (209.99992, 87.99957, 1, 0.0078125),
        (50.000343, 104.00058, 1, 0.017265625), (129.99869, 104.00003, 1, 0.034453124),
        (210.00032, 103.99941, 1, 0.01515625), (49.99984, 120.00056, 1, 0.008125),
        (130.00082, 119.99971, 1, 0.0278125), (210.00024, 119.99991, 1, 0.001015625),
        (49.999844, 136.00055, 1, 0.008125), (129.99997, 135.99915, 1, 0.0153125),
        (210.00006, 135.99973, 1, 0.00765625), (49.999847, 151.99956, 1, 0.009453125),
        (129.99963, 152.00047, 1, 0.017890625), (210.00014, 152.0004, 1, 0.00953125),
        (50.000328, 167.99985, 1, 0.0103125), (130.0012, 167.99983, 1, 0.026015624),
        (210.0003, 168.00027, 1, 0.01453125), (49.999847, 184.00037, 1, 0.009140625),
        (129.99963, 183.99915, 1, 0.028828125), (210.00061, 184.0, 1, 0.00796875),
    ];
    #[rustfmt::skip]
    pub const VAR_WIN_15X9: [FlowRow; 30] = [
        (49.99996, 40.000137, 1, 0.0), (129.99986, 39.999916, 1, 0.0),
        (209.99985, 40.000137, 1, 0.0009259259), (50.00018, 56.00006, 1, 0.0),
        (129.99991, 56.0001, 1, 0.0), (210.00024, 55.999954, 1, 0.0018518518),
        (50.000137, 71.99991, 1, 0.0), (130.0001, 71.99996, 1, 0.0),
        (210.00008, 71.99984, 1, 0.0), (50.00001, 88.00015, 1, 0.0),
        (129.99979, 88.00011, 1, 0.00046296295), (210.00003, 87.999794, 1, 0.0),
        (50.000126, 104.000084, 1, 0.0), (129.99985, 104.00044, 1, 0.010648148),
        (210.00002, 104.00023, 1, 0.0025462962), (49.99999, 119.99989, 1, 0.0),
        (129.99997, 120.000244, 1, 0.00069444446), (209.99985, 120.000175, 1, 0.00046296295),
        (49.999954, 136.00035, 1, 0.0), (130.00003, 136.00108, 1, 0.0),
        (210.0001, 136.00018, 1, 0.0), (49.999794, 152.00002, 1, 0.0),
        (129.99991, 152.0001, 1, 0.0), (210.00008, 151.99994, 1, 0.0),
        (49.99987, 168.0001, 1, 0.0), (130.0, 167.99998, 1, 0.0),
        (209.99994, 168.00009, 1, 0.0), (50.000042, 183.99986, 1, 0.0),
        (130.0, 184.00021, 1, 0.0025462962), (210.00003, 184.0, 1, 0.0),
    ];
    #[rustfmt::skip]
    pub const VAR_WIN_7X7: [FlowRow; 30] = [
        (49.999924, 40.000042, 1, 0.0), (130.0001, 40.000027, 1, 0.0),
        (210.00005, 40.000076, 1, 0.0), (50.00007, 56.000008, 1, 0.0),
        (130.00012, 55.999897, 1, 0.0), (210.00006, 55.999798, 1, 0.0),
        (50.00009, 71.99982, 1, 0.0), (130.00002, 72.0, 1, 0.0),
        (209.99986, 72.00007, 1, 0.0), (50.000072, 87.9994, 1, 0.0),
        (129.99986, 87.99939, 1, 0.0), (209.99979, 87.999756, 1, 0.0),
        (49.999886, 104.00021, 1, 0.0), (129.99988, 104.000145, 1, 0.0),
        (209.99992, 104.00016, 1, 0.0), (49.999954, 119.99987, 1, 0.0),
        (129.99988, 120.00015, 1, 0.0), (210.0001, 120.0, 1, 0.0),
        (50.000004, 135.99988, 0, 0.0), (129.99983, 135.99924, 0, 0.0),
        (210.00018, 135.99973, 0, 0.0), (49.999844, 151.9981, 0, 0.0),
        (129.99995, 152.00069, 0, 0.0), (209.99992, 151.99988, 0, 0.0),
        (49.999928, 167.99983, 1, 0.0), (130.00012, 167.99991, 1, 0.0),
        (210.00008, 167.99995, 1, 0.0), (50.000042, 183.99986, 1, 0.0),
        (130.00012, 184.0, 1, 0.0), (210.00006, 183.99998, 1, 0.0),
    ];
    #[rustfmt::skip]
    pub const VAR_WIN_3X3: [FlowRow; 30] = [
        (49.996925, 40.00013, 1, 0.0), (-123.070435, 7.2377033, 0, 0.0),
        (168.8407, -43.514305, 0, 0.0), (49.99303, 56.002644, 0, 0.0),
        (-123.35278, 69.107895, 0, 0.0), (27.970453, 64.86249, 0, 0.0),
        (49.995975, 71.999916, 1, 0.0), (129.99928, 71.99989, 1, 0.013888889),
        (27.9992, 44.999912, 1, 0.010416667), (49.9947, 87.97693, 0, 0.0),
        (-34.637375, 89.234535, 0, 0.0), (18.200474, 252.63243, 0, 0.0),
        (49.99796, 103.999916, 1, 0.0), (129.99953, 103.99982, 1, 0.0069444445),
        (201.9992, 204.9998, 1, 0.010416667), (49.926895, 104.00011, 1, 0.0),
        (57.999844, 129.99983, 1, 0.0), (168.99878, 91.99975, 1, 0.024305556),
        (49.989254, 136.16658, 0, 0.0), (130.00372, 136.8666, 0, 0.0),
        (73.59928, 254.27707, 0, 0.0), (49.996246, 151.99959, 0, 0.0),
        (130.0001, 152.00194, 0, 0.0), (27.719547, 56.638554, 0, 0.0),
        (49.99782, 167.99982, 1, 0.0), (129.99915, 167.99977, 1, 0.020833334),
        (152.87404, 242.53467, 0, 0.0), (49.996357, 183.99956, 1, 0.0),
        (16.159164, 287.3936, 0, 0.0), (202.0002, 176.81262, 1, 0.0),
    ];
    #[rustfmt::skip]
    pub const VAR_MAX_LEVEL_0: [FlowRow; 30] = [
        (49.999817, 40.0, 1, 0.0), (115.274734, 40.0, 1, 23.949972),
        (200.0, 40.0, 1, 134.16327), (39.501728, 65.65566, 1, 53.243835),
        (115.27478, 56.038326, 1, 23.487175), (200.0, 65.67535, 1, 55.24943),
        (39.5079, 60.947353, 1, 48.468395), (115.27511, 71.97704, 1, 24.996386),
        (200.0, 60.945927, 1, 50.769558), (39.504906, 77.957695, 1, 30.707554),
        (115.27496, 87.97156, 1, 24.194727), (200.0, 77.953354, 1, 32.052082),
        (39.520813, 97.08804, 1, 54.034084), (115.27453, 103.985855, 1, 23.230442),
        (200.0, 97.09294, 1, 56.496883), (39.51973, 111.04105, 1, 53.783375),
        (115.274826, 119.961, 1, 23.48526), (200.0, 111.039116, 1, 56.24901),
        (39.50659, 125.16532, 1, 43.292233), (115.27494, 135.96085, 1, 23.495394),
        (200.0, 125.15941, 1, 45.062218), (50.000053, 151.99992, 1, 0.0),
        (115.27434, 152.00023, 1, 22.194727), (200.0, 152.00018, 1, 124.408165),
        (39.525898, 174.85492, 1, 63.510914), (115.274574, 168.01416, 1, 23.232426),
        (200.0, 174.84517, 1, 66.156815), (49.99974, 183.99995, 1, 0.002409297),
        (115.27434, 183.99933, 1, 22.196642), (200.0, 183.99902, 1, 124.45536),
    ];
    #[rustfmt::skip]
    pub const VAR_MAX_LEVEL_1: [FlowRow; 30] = [
        (32.099243, 39.989815, 1, 54.53196), (115.2748, 40.000008, 1, 23.949972),
        (191.4975, 39.999947, 1, 23.687075), (32.100273, 56.321896, 1, 54.727535),
        (115.27455, 56.038193, 1, 23.480797), (191.49423, 56.038803, 1, 23.318098),
        (32.100273, 71.79238, 1, 57.431053), (115.2749, 71.97696, 1, 24.995323),
        (191.49374, 71.97693, 1, 24.760841), (32.100204, 87.76483, 1, 55.615223),
        (115.27461, 87.971886, 1, 24.191751), (191.49423, 87.97273, 1, 23.977324),
        (32.09918, 103.88809, 1, 53.2765), (115.27451, 103.9858, 1, 23.230442),
        (191.49443, 103.98602, 1, 23.024942), (32.099194, 119.671646, 1, 54.78203),
        (115.2746, 119.961174, 1, 23.478529), (191.49416, 119.96205, 1, 23.311579),
        (32.10027, 135.6749, 1, 54.746456), (115.27454, 135.96103, 1, 23.490292),
        (191.49431, 135.9622, 1, 23.3052), (32.099274, 152.01306, 1, 50.597153),
        (115.27443, 152.0, 1, 22.194515), (191.49464, 151.99982, 1, 21.959183),
        (32.09961, 168.13594, 1, 53.444374), (115.27447, 168.0141, 1, 23.22612),
        (191.49452, 168.01402, 1, 23.024942), (32.101967, 183.99608, 1, 50.518494),
        (115.27418, 183.9993, 1, 22.195862), (191.49487, 184.00069, 1, 21.968466),
    ];
    #[rustfmt::skip]
    pub const VAR_MIN_EIG_0_5: [FlowRow; 30] = [
        (32.09906, 39.9897, 1, 54.53224), (115.2748, 40.000008, 1, 23.949972),
        (191.4975, 39.999947, 1, 23.687075), (32.100273, 56.321896, 1, 54.727535),
        (115.27455, 56.038193, 1, 23.480797), (191.49423, 56.038803, 1, 23.318098),
        (32.100273, 71.79238, 1, 57.431053), (115.2749, 71.97696, 1, 24.995323),
        (191.49374, 71.97693, 1, 24.760841), (32.100204, 87.76483, 1, 55.615223),
        (115.27461, 87.971886, 1, 24.191751), (191.49423, 87.97273, 1, 23.977324),
        (32.099144, 103.88652, 1, 53.287556), (115.27451, 103.9858, 1, 23.230442),
        (191.49443, 103.98602, 1, 23.024942), (32.09931, 119.66976, 1, 54.79599),
        (115.2746, 119.961174, 1, 23.478529), (191.49416, 119.96205, 1, 23.311579),
        (32.099464, 135.6695, 1, 54.794785), (115.27454, 135.96103, 1, 23.490292),
        (191.49431, 135.9622, 1, 23.3052), (32.099274, 152.01306, 1, 50.597153),
        (115.27443, 152.0, 1, 22.194515), (191.49464, 151.99982, 1, 21.959183),
        (32.09961, 168.13594, 1, 53.444374), (115.27447, 168.0141, 1, 23.22612),
        (191.49452, 168.01402, 1, 23.024942), (32.101967, 183.99608, 1, 50.518494),
        (115.27418, 183.9993, 1, 22.195862), (191.49487, 184.00069, 1, 21.968466),
    ];
    #[rustfmt::skip]
    pub const VAR_MIN_EIG_0_5_OUTPUT: [FlowRow; 30] = [
        (32.09906, 39.9897, 1, 0.63241655), (115.2748, 40.000008, 1, 0.63241655),
        (191.4975, 39.999947, 1, 0.7321734), (32.100273, 56.321896, 1, 0.7899947),
        (115.27455, 56.038193, 1, 0.7912569), (191.49423, 56.038803, 1, 0.68229496),
        (32.100273, 71.79238, 1, 0.5064759), (115.2749, 71.97696, 1, 0.50563556),
        (191.49374, 71.97693, 1, 0.58631605), (32.100204, 87.76483, 1, 0.6644056),
        (115.27461, 87.971886, 1, 0.66398376), (191.49423, 87.97273, 1, 0.7217519),
        (32.099144, 103.88652, 1, 0.8230357), (115.27451, 103.9858, 1, 0.82135165),
        (191.49443, 103.98602, 1, 0.67187345), (32.09931, 119.66976, 1, 0.79104656),
        (115.2746, 119.961174, 1, 0.7897843), (191.49416, 119.96205, 1, 0.68229496),
        (32.099464, 135.6695, 1, 0.7910466), (115.27454, 135.96103, 1, 0.7897843),
        (191.49431, 135.9622, 1, 0.68229496), (32.099274, 152.01306, 1, 0.9486149),
        (115.27443, 152.0, 1, 0.94812554), (191.49464, 151.99982, 1, 0.63241655),
        (32.09961, 168.13594, 1, 0.8230356), (115.27447, 168.0141, 1, 0.82135165),
        (191.49452, 168.01402, 1, 0.6718736), (32.101967, 183.99608, 1, 0.94842356),
        (115.27418, 183.9993, 1, 0.94842356), (191.49487, 184.00069, 1, 0.63241655),
    ];
    #[rustfmt::skip]
    pub const VAR_COUNT_5: [FlowRow; 30] = [
        (50.00019, 40.00014, 1, 0.00042517006), (129.9999, 40.000023, 1, 0.0),
        (209.99992, 39.9999, 1, 0.0), (49.99991, 56.000126, 1, 0.0),
        (129.99985, 55.999855, 1, 0.0), (210.00017, 55.999947, 1, 0.0),
        (49.999893, 71.99997, 1, 0.0), (129.99992, 72.00011, 1, 0.0),
        (209.99991, 71.999794, 1, 0.0), (49.999847, 88.00006, 1, 0.0),
        (129.99979, 87.99982, 1, 0.00042517006), (210.00015, 87.99995, 1, 0.0),
        (49.999977, 104.00004, 1, 0.0), (130.00015, 104.00016, 1, 0.0006377551),
        (209.99992, 103.99998, 1, 0.0), (50.000114, 120.000015, 1, 0.0),
        (129.99994, 119.99983, 1, 0.0), (210.0001, 119.99999, 1, 0.0),
        (50.000088, 135.99997, 1, 0.0), (130.00003, 135.99997, 1, 0.0),
        (209.99985, 136.0, 1, 0.0), (49.99983, 152.00008, 1, 0.0),
        (130.00015, 151.99997, 1, 0.0), (210.00006, 152.00015, 1, 0.00042517006),
        (50.00015, 168.00012, 1, 0.0), (129.99988, 167.99994, 1, 0.0),
        (209.99982, 168.00002, 1, 0.0), (50.0002, 183.99997, 1, 0.0017006802),
        (130.00015, 184.00002, 1, 0.0), (209.9999, 184.00017, 1, 0.0),
    ];
    #[rustfmt::skip]
    pub const VAR_COUNT_0: [FlowRow; 30] = [
        (40.0, 40.0, 1, 123.68481), (120.0, 40.0, 1, 92.14966),
        (200.0, 40.0, 1, 134.16327), (40.0, 56.0, 1, 119.17687),
        (120.0, 56.0, 1, 88.77324), (200.0, 56.0, 1, 129.29478),
        (40.0, 72.0, 1, 128.20181), (120.0, 72.0, 1, 95.48073),
        (200.0, 72.0, 1, 139.10431), (40.0, 88.0, 1, 123.69388),
        (120.0, 88.0, 1, 92.10431), (200.0, 88.0, 1, 134.23582),
        (40.0, 104.0, 1, 119.19274), (120.0, 104.0, 1, 88.69388),
        (200.0, 104.0, 1, 129.42177), (40.0, 120.0, 1, 119.190475),
        (120.0, 120.0, 1, 88.705215), (200.0, 120.0, 1, 129.40363),
        (40.0, 136.0, 1, 119.17687), (120.0, 136.0, 1, 88.77324),
        (200.0, 136.0, 1, 129.29478), (40.0, 152.0, 1, 114.666664),
        (120.0, 152.0, 1, 85.408165), (200.0, 152.0, 1, 124.408165),
        (40.0, 168.0, 1, 119.19274), (120.0, 168.0, 1, 88.69388),
        (200.0, 168.0, 1, 129.42177), (40.0, 184.0, 1, 114.67347),
        (120.0, 184.0, 1, 85.37415), (200.0, 184.0, 1, 124.462585),
    ];
    #[rustfmt::skip]
    pub const VAR_EPS_ONLY_0_001: [FlowRow; 30] = [
        (50.000175, 40.000088, 1, 0.0), (130.0, 40.000095, 1, 0.0),
        (210.0, 39.999977, 1, 0.0), (50.0001, 56.000126, 1, 0.0),
        (129.99983, 55.99983, 1, 0.0006377551), (209.9999, 55.99996, 1, 0.0),
        (49.99997, 72.0, 1, 0.0), (129.99988, 72.000145, 1, 0.0),
        (209.99991, 72.0, 1, 0.0), (50.000168, 88.0001, 1, 0.0),
        (130.00006, 87.999916, 1, 0.0), (210.00015, 87.99988, 1, 0.0),
        (49.999786, 103.99987, 1, 0.0006377551), (129.99983, 103.999954, 1, 0.0),
        (209.9998, 104.00005, 1, 0.0), (50.00019, 120.00008, 1, 0.0),
        (130.00006, 119.999886, 1, 0.0), (209.99986, 120.00013, 1, 0.0),
        (50.000187, 136.00009, 1, 0.0018424037), (130.00014, 136.00006, 1, 0.0),
        (209.99986, 136.00009, 1, 0.0), (50.000206, 152.00005, 1, 0.0),
        (129.99997, 151.99994, 1, 0.0), (209.99983, 152.0, 1, 0.0),
        (49.999977, 168.0001, 1, 0.0), (130.0, 168.0001, 1, 0.0),
        (209.99985, 168.00002, 1, 0.0), (49.999832, 183.99998, 1, 0.0),
        (129.9998, 184.00009, 1, 0.0006377551), (210.00015, 183.99983, 1, 0.00042517006),
    ];
    #[rustfmt::skip]
    pub const VAR_BOTH_100_0: [FlowRow; 30] = [
        (50.00007, 39.999924, 1, 0.0), (130.0, 40.000095, 1, 0.0),
        (210.00005, 39.999992, 1, 0.0), (49.999905, 56.000103, 1, 0.0),
        (129.99995, 56.000076, 1, 0.0), (210.00012, 55.999844, 1, 0.00042517006),
        (49.99979, 72.00001, 1, 0.0), (129.99988, 72.00009, 1, 0.0),
        (209.99994, 71.999916, 1, 0.0), (49.999863, 88.00007, 1, 0.0),
        (129.99988, 88.00004, 1, 0.0), (209.9998, 88.00013, 1, 0.00042517006),
        (50.000126, 104.00007, 1, 0.0), (129.99992, 104.00017, 1, 0.0),
        (209.99995, 104.0001, 1, 0.0), (49.999847, 120.00016, 1, 0.0006377551),
        (130.00006, 119.999886, 1, 0.0), (209.9999, 119.99987, 1, 0.0),
        (49.999928, 136.00008, 1, 0.0), (130.00012, 135.99995, 1, 0.0),
        (209.99986, 136.00012, 1, 0.0), (50.000206, 152.00005, 1, 0.0),
        (129.99997, 151.99992, 1, 0.0), (209.99983, 151.99982, 1, 0.00042517006),
        (49.99995, 168.00008, 1, 0.0), (129.99994, 167.99998, 1, 0.0),
        (209.99983, 167.99982, 1, 0.00042517006), (49.999832, 183.99998, 1, 0.0),
        (130.00009, 183.99992, 1, 0.0), (209.9999, 183.9999, 1, 0.0),
    ];
    #[rustfmt::skip]
    pub const VAR_BOTH_1_0_03: [FlowRow; 30] = [
        (49.998413, 40.003418, 1, 0.08056973), (130.00027, 39.99857, 1, 0.03125),
        (210.00143, 40.000458, 1, 0.030045351), (49.99314, 56.004913, 1, 0.22534014),
        (130.00327, 56.00228, 1, 0.11203232), (210.00044, 56.000423, 1, 0.015943877),
        (49.99753, 71.998795, 1, 0.07079082), (130.00096, 71.99974, 1, 0.024659865),
        (210.00029, 71.9995, 1, 0.013818027), (49.99948, 87.9997, 1, 0.017644558),
        (130.00244, 87.99878, 1, 0.07079082), (210.00005, 87.99774, 1, 0.038690478),
        (49.992687, 103.99773, 1, 0.20273526), (130.0072, 103.996254, 1, 0.22789116),
        (210.00758, 103.99858, 1, 0.1401644), (49.997524, 119.997406, 1, 0.09162415),
        (130.00543, 119.99437, 1, 0.20408164), (210.00616, 119.99688, 1, 0.14002268),
        (49.997715, 135.99861, 1, 0.07482993), (130.00519, 135.99617, 1, 0.17113096),
        (210.00706, 135.99733, 1, 0.14576247), (49.99801, 151.999, 1, 0.06122449),
        (130.00607, 151.99821, 1, 0.16468254), (210.00859, 151.9997, 1, 0.13541667),
        (49.997433, 168.00005, 1, 0.057397958), (130.00714, 168.0006, 1, 0.1690051),
        (210.00838, 168.00081, 1, 0.14455782), (50.001926, 184.00034, 1, 0.05229592),
        (130.0067, 183.99992, 1, 0.15079366), (210.01178, 183.99997, 1, 0.1734694),
    ];
    #[rustfmt::skip]
    pub const VAR_INIT_GOOD__INITIAL_FLOW: [FlowRow; 30] = [
        (50.000046, 40.000202, 1, 0.0), (129.99985, 40.000343, 1, 0.0063775512),
        (209.99997, 40.000202, 1, 0.0012046485), (49.99988, 56.000057, 1, 0.0),
        (130.00008, 55.999905, 1, 0.0), (210.00009, 56.00017, 1, 0.0),
        (50.0001, 72.00006, 1, 0.0), (129.99985, 72.00008, 1, 0.0),
        (209.99991, 71.99991, 1, 0.0), (49.999924, 87.99997, 1, 0.0),
        (130.00015, 87.99991, 1, 0.0), (210.0002, 87.99994, 1, 0.0),
        (49.99992, 104.00008, 1, 0.0), (130.00003, 104.00003, 1, 0.0),
        (210.00006, 104.00009, 1, 0.0), (49.999985, 119.99994, 1, 0.0),
        (130.00003, 119.999886, 1, 0.0), (209.99983, 119.99997, 1, 0.0),
        (49.999844, 136.0, 1, 0.0), (130.00015, 135.99979, 1, 0.00042517006),
        (209.99986, 136.00005, 1, 0.0), (50.000156, 152.00009, 1, 0.0),
        (130.00006, 151.99992, 1, 0.0), (209.99982, 152.00003, 1, 0.0),
        (49.999985, 168.0001, 1, 0.0), (129.99988, 167.99995, 1, 0.0),
        (209.99991, 168.0001, 1, 0.0), (49.9999, 183.99992, 1, 0.0),
        (130.00003, 184.00006, 1, 0.0), (209.99991, 184.00012, 1, 0.0),
    ];
    #[rustfmt::skip]
    pub const VAR_INIT_BAD__INITIAL_FLOW: [FlowRow; 30] = [
        (50.000088, 40.000145, 1, 0.00042517006), (130.00015, 40.00026, 1, 0.005952381),
        (210.00008, 40.000107, 1, 0.0), (49.999924, 56.00014, 1, 0.0),
        (130.00012, 55.999886, 1, 0.0), (209.99985, 56.00012, 1, 0.0),
        (50.000114, 72.000015, 1, 0.0), (129.99982, 72.00009, 1, 0.0),
        (209.99992, 71.999855, 1, 0.0), (49.99982, 88.00003, 1, 0.0),
        (129.99976, 87.99988, 1, 0.002409297), (209.99988, 87.99988, 1, 0.0),
        (50.000042, 104.00013, 1, 0.0), (130.00009, 104.00008, 1, 0.0),
        (209.99988, 103.99997, 1, 0.0), (49.99995, 119.99995, 1, 0.0),
        (130.00008, 119.999886, 1, 0.0), (209.9999, 119.99996, 1, 0.0),
        (49.99983, 136.0, 1, 0.0), (130.00003, 135.99979, 1, 0.0),
        (209.99994, 135.99998, 1, 0.0), (32.099213, 152.01137, 1, 50.58652),
        (129.99997, 151.99997, 1, 0.0), (209.99988, 152.0001, 1, 0.0),
        (16.590755, 168.00104, 1, 27.923965), (129.99985, 168.0, 1, 0.0),
        (209.99986, 168.00014, 1, 0.0), (16.293282, 168.18774, 1, 52.164967),
        (129.9999, 183.99986, 1, 0.0), (210.00006, 184.0002, 1, 0.0),
    ];
    #[rustfmt::skip]
    pub const VAR_INIT_GOOD__INITIAL_FLOW__COUNT_0: [FlowRow; 30] = [
        (48.0, 40.0, 1, 44.743763), (128.0, 40.0, 1, 44.743763),
        (208.0, 40.0, 1, 31.582766), (48.0, 56.0, 1, 43.108845),
        (128.0, 56.0, 1, 43.108845), (208.0, 56.0, 1, 30.43084),
        (48.0, 72.0, 1, 46.369614), (128.0, 72.0, 1, 46.369614),
        (208.0, 72.0, 1, 32.734695), (48.0, 88.0, 1, 44.734695),
        (128.0, 88.0, 1, 44.734695), (208.0, 88.0, 1, 31.582766),
        (48.0, 104.0, 1, 43.09297), (128.0, 104.0, 1, 43.09297),
        (208.0, 104.0, 1, 30.43084), (48.0, 120.0, 1, 43.095238),
        (128.0, 120.0, 1, 43.095238), (208.0, 120.0, 1, 30.43084),
        (48.0, 136.0, 1, 43.108845), (128.0, 136.0, 1, 43.108845),
        (208.0, 136.0, 1, 30.43084), (48.0, 152.0, 1, 41.47619),
        (128.0, 152.0, 1, 41.47619), (208.0, 152.0, 1, 29.278912),
        (48.0, 168.0, 1, 43.09297), (128.0, 168.0, 1, 43.09297),
        (208.0, 168.0, 1, 30.43084), (48.0, 184.0, 1, 41.469387),
        (128.0, 184.0, 1, 41.469387), (208.0, 184.0, 1, 29.278912),
    ];
    /// `video_lk.py` "variations": `(key, rows)`; keys that share a constant gave identical rows.
    #[rustfmt::skip]
    pub const VARIATIONS: [(&str, &[FlowRow; 30]); 21] = [
        ("default", &VAR_DEFAULT),
        ("min_eig_output", &VAR_MIN_EIG_OUTPUT),
        ("win_20x20", &VAR_WIN_20X20),
        ("win_15x9", &VAR_WIN_15X9),
        ("win_7x7", &VAR_WIN_7X7),
        ("win_3x3", &VAR_WIN_3X3),
        ("max_level_0", &VAR_MAX_LEVEL_0),
        ("max_level_1", &VAR_MAX_LEVEL_1),
        ("max_level_5", &VAR_DEFAULT),
        ("min_eig_1e-2", &VAR_DEFAULT),
        ("min_eig_0.5", &VAR_MIN_EIG_0_5),
        ("min_eig_0.5_output", &VAR_MIN_EIG_0_5_OUTPUT),
        ("count_5", &VAR_COUNT_5),
        ("count_0", &VAR_COUNT_0),
        ("eps_only_0.001", &VAR_EPS_ONLY_0_001),
        ("both_100_0", &VAR_BOTH_100_0),
        ("both_1_0.03", &VAR_BOTH_1_0_03),
        ("init_good/initial_flow", &VAR_INIT_GOOD__INITIAL_FLOW),
        ("init_bad/initial_flow", &VAR_INIT_BAD__INITIAL_FLOW),
        ("init_good/ignored_without_flag", &VAR_DEFAULT),
        ("init_good/initial_flow/count_0", &VAR_INIT_GOOD__INITIAL_FLOW__COUNT_0),
    ];

    #[rustfmt::skip]
    pub const SUB_DEFAULT: [FlowRow; 48] = [
        (18.265804, 15.289519, 1, 2.3861253), (30.33954, 15.258662, 1, 3.294643),
        (42.25416, 15.258753, 1, 4.2381663), (54.253414, 15.245663, 1, 4.1725483),
        (66.25324, 15.221426, 1, 4.4690332), (78.25041, 15.147846, 1, 3.2208757),
        (90.23151, 15.200329, 1, 2.8825114), (102.285576, 15.241825, 1, 3.965703),
        (18.257393, 27.116764, 1, 3.2478743), (30.26606, 27.236662, 1, 4.1089854),
        (42.261406, 27.256376, 1, 6.7312217), (54.273415, 27.25349, 1, 7.7565193),
        (66.23868, 27.251133, 1, 7.6852326), (78.24004, 27.259928, 1, 4.8726616),
        (90.26374, 27.241781, 1, 3.2701247), (102.2493, 27.248161, 1, 4.131661),
        (18.251587, 39.281082, 1, 6.2062073), (30.204811, 39.251934, 1, 5.4164543),
        (42.25582, 39.25525, 1, 7.0848923), (54.26054, 39.260193, 1, 8.263605),
        (66.23439, 39.253693, 1, 7.1099772), (78.22163, 39.24887, 1, 3.9389172),
        (90.308815, 39.265663, 1, 2.8686934), (102.25899, 39.23906, 1, 4.338506),
        (18.247818, 51.24281, 1, 7.6497307), (30.21815, 51.232697, 1, 5.708475),
        (42.25169, 51.23618, 1, 5.373583), (54.267235, 51.247696, 1, 7.3312783),
        (66.28053, 51.278515, 1, 8.451318), (78.20462, 51.289146, 1, 5.1477466),
        (90.26104, 51.24738, 1, 5.0185657), (102.23228, 51.244293, 1, 5.8848495),
        (18.245546, 63.24854, 1, 7.154691), (30.235424, 63.18948, 1, 4.021188),
        (42.249527, 63.283672, 1, 3.1980584), (54.290882, 63.250362, 1, 4.8039255),
        (66.42165, 63.24895, 1, 6.8229876), (78.14085, 63.2518, 1, 5.948625),
        (90.2718, 63.253002, 1, 7.119473), (102.252014, 63.24987, 1, 8.460034),
        (18.246536, 75.2287, 1, 4.2568026), (30.237654, 75.24476, 1, 3.2583616),
        (42.245766, 75.24318, 1, 2.9410431), (54.262783, 75.190155, 1, 3.1354167),
        (66.267334, 75.13913, 1, 3.2349064), (78.171486, 75.1659, 1, 4.2397957),
        (90.25678, 75.230865, 1, 5.9321146), (102.23287, 75.25228, 1, 6.221797),
    ];
    #[rustfmt::skip]
    pub const SUB_MIN_EIG_OUTPUT: [FlowRow; 48] = [
        (18.265804, 15.289519, 1, 0.012082475), (30.33954, 15.258662, 1, 0.024247117),
        (42.25416, 15.258753, 1, 0.105573624), (54.253414, 15.245663, 1, 0.059062038),
        (66.25324, 15.221426, 1, 0.10417319), (78.25041, 15.147846, 1, 0.06129118),
        (90.23151, 15.200329, 1, 0.024854427), (102.285576, 15.241825, 1, 0.064191945),
        (18.257393, 27.116764, 1, 0.051945247), (30.26606, 27.236662, 1, 0.08097342),
        (42.261406, 27.256376, 1, 0.26982915), (54.273415, 27.25349, 1, 0.1535344),
        (66.23868, 27.251133, 1, 0.23291525), (78.24004, 27.259928, 1, 0.15166706),
        (90.26374, 27.241781, 1, 0.031923316), (102.2493, 27.248161, 1, 0.05814202),
        (18.251587, 39.281082, 1, 0.094538495), (30.204811, 39.251934, 1, 0.12795413),
        (42.25582, 39.25525, 1, 0.24484919), (54.26054, 39.260193, 1, 0.3759912),
        (66.23439, 39.253693, 1, 0.327787), (78.22163, 39.24887, 1, 0.16260199),
        (90.308815, 39.265663, 1, 0.020253384), (102.25899, 39.23906, 1, 0.08221117),
        (18.247818, 51.24281, 1, 0.1619569), (30.21815, 51.232697, 1, 0.1642394),
        (42.25169, 51.23618, 1, 0.1530495), (54.267235, 51.247696, 1, 0.25614786),
        (66.28053, 51.278515, 1, 0.17305134), (78.20462, 51.289146, 1, 0.08462048),
        (90.26104, 51.24738, 1, 0.12465006), (102.23228, 51.244293, 1, 0.13694668),
        (18.245546, 63.24854, 1, 0.19008645), (30.235424, 63.18948, 1, 0.09052187),
        (42.249527, 63.283672, 1, 0.035643235), (54.290882, 63.250362, 1, 0.071698196),
        (66.42165, 63.24895, 1, 0.017607916), (78.14085, 63.2518, 1, 0.18404837),
        (90.2718, 63.253002, 1, 0.27271286), (102.252014, 63.24987, 1, 0.35962132),
        (18.246536, 75.2287, 1, 0.16532362), (30.237654, 75.24476, 1, 0.044533625),
        (42.245766, 75.24318, 1, 0.018060574), (54.262783, 75.190155, 1, 0.029455354),
        (66.267334, 75.13913, 1, 0.05232497), (78.171486, 75.1659, 1, 0.12401874),
        (90.25678, 75.230865, 1, 0.2068478), (102.23287, 75.25228, 1, 0.23724082),
    ];
    #[rustfmt::skip]
    pub const SUB_WIN_10X10: [FlowRow; 48] = [
        (18.168556, 15.1672535, 1, 0.38375), (30.219606, 15.18616, 1, 0.4025),
        (42.314613, 15.174121, 1, 1.1228125), (54.13085, 15.205651, 1, 1.3421875),
        (66.23639, 15.223028, 1, 0.696875), (78.18469, 15.2131195, 1, 0.9953125),
        (90.174416, 15.178317, 1, 0.4584375), (102.197525, 15.213849, 1, 0.6146875),
        (18.20184, 27.166565, 1, 0.3903125), (30.429571, 27.145624, 1, 0.599375),
        (42.212147, 27.231817, 1, 1.9328125), (54.230564, 27.223768, 1, 3.2875),
        (66.17921, 27.235298, 1, 3.376875), (78.208885, 27.215904, 1, 1.051875),
        (90.13476, 27.220274, 1, 0.395), (102.20737, 27.287205, 1, 0.6384375),
        (18.194683, 39.23455, 1, 1.940625), (30.99241, 39.216175, 1, 1.270625),
        (42.214016, 39.23985, 1, 2.4178126), (54.201218, 39.219975, 1, 1.3496875),
        (66.22678, 39.220715, 1, 1.61625), (78.1644, 39.211086, 1, 0.6134375),
        (90.189125, 39.19476, 1, 0.55), (102.203896, 39.210224, 1, 1.2209375),
        (18.215458, 51.082138, 1, 1.93875), (30.392141, 51.2127, 1, 1.535625),
        (42.16554, 51.198456, 1, 1.051875), (54.221622, 51.219135, 1, 1.2859375),
        (66.212654, 51.214684, 1, 2.3365624), (78.15758, 51.20959, 1, 1.1240625),
        (90.20716, 51.228092, 1, 1.0709375), (102.20949, 51.222378, 1, 0.9875),
        (18.33731, 63.429447, 1, 2.1371875), (30.212885, 63.422043, 1, 1.193125),
        (42.166557, 63.202446, 1, 0.6740625), (54.230465, 63.19935, 1, 1.176875),
        (66.39418, 63.207413, 1, 1.7303125), (78.21488, 63.20197, 1, 1.361875),
        (90.22078, 63.210068, 1, 1.629375), (102.17359, 63.158665, 1, 0.4028125),
        (18.321758, 75.229706, 1, 1.4909375), (30.210087, 75.22295, 1, 1.0925),
        (42.236027, 75.22218, 1, 0.9784375), (54.161114, 75.19339, 1, 0.526875),
        (66.219086, 75.21245, 1, 1.386875), (78.206436, 75.20752, 1, 1.64125),
        (90.19115, 75.42774, 1, 1.3875), (102.238365, 75.25971, 1, 3.0525),
    ];
    #[rustfmt::skip]
    pub const SUB_WIN_7X7: [FlowRow; 48] = [
        (18.15887, 15.2323885, 1, 1.9923469), (30.241583, 15.361367, 1, 1.9272959),
        (42.251614, 15.248947, 1, 4.685587), (54.24701, 15.23956, 1, 4.770408),
        (66.32775, 15.32734, 1, 3.1167092), (77.086266, 15.23824, 1, 5.6594386),
        (90.267944, 15.245925, 1, 2.033801), (102.31001, 15.210829, 1, 2.528699),
        (18.301182, 27.202011, 1, 1.9132653), (30.211231, 27.244108, 1, 2.1358418),
        (42.28283, 27.286266, 1, 8.790178), (54.553486, 27.261934, 1, 15.585459),
        (66.22885, 27.239574, 1, 14.183674), (72.60952, 27.178308, 1, 11.178572),
        (90.312485, 27.43759, 1, 2.6721938), (102.2313, 27.161055, 1, 1.8131378),
        (18.3055, 39.29159, 1, 5.723214), (30.120985, 39.765182, 1, 2.0229592),
        (48.314247, 73.82244, 1, 45.501274), (54.338615, 39.008778, 1, 1.6894133),
        (66.305084, 39.274002, 1, 8.53699), (78.19222, 39.24536, 1, 2.833546),
        (90.228264, 39.282574, 1, 2.7538266), (106.19201, 40.68897, 1, 3.6198978),
        (18.248077, 51.30669, 1, 2.2653062), (30.34865, 51.59844, 1, 4.897321),
        (42.095337, 51.781307, 1, 2.8233418), (54.333515, 51.014633, 1, 1.9030613),
        (66.26749, 50.909035, 1, 2.9642856), (78.09915, 50.90675, 1, 1.9866072),
        (90.291885, 51.287407, 1, 5.4808674), (102.230286, 51.287884, 1, 4.7085457),
        (18.248177, 62.4729, 1, 4.834821), (30.251343, 63.136227, 1, 4.508291),
        (42.35004, 63.213814, 1, 1.6607143), (53.987144, 63.64312, 1, 4.09375),
        (64.53473, 52.641968, 1, 25.894133), (78.82222, 63.688187, 1, 5.138393),
        (90.26327, 63.160866, 1, 8.332271), (102.1969, 63.23062, 1, 1.9617347),
        (18.354706, 75.24887, 1, 4.614158), (30.152184, 75.24443, 1, 4.526148),
        (42.329845, 75.25506, 1, 4.528699), (58.655823, 77.65671, 1, 6.210459),
        (66.24085, 75.290245, 1, 2.2391582), (78.253654, 75.10886, 1, 6.792092),
        (90.956726, 75.65204, 1, 4.0012755), (102.18773, 75.41561, 1, 9.458546),
    ];
    #[rustfmt::skip]
    pub const SUB_MAX_LEVEL_0: [FlowRow; 48] = [
        (18.265678, 15.289738, 1, 2.3857002), (30.339529, 15.258687, 1, 3.294643),
        (42.254196, 15.258802, 1, 4.2385206), (54.25351, 15.245596, 1, 4.172194),
        (66.25316, 15.221447, 1, 4.46875), (78.25037, 15.148137, 1, 3.2221513),
        (90.23145, 15.20032, 1, 2.8825822), (102.2855, 15.241788, 1, 3.9653487),
        (18.257439, 27.116821, 1, 3.2482994), (30.265799, 27.236824, 1, 4.1087728),
        (42.26145, 27.256456, 1, 6.732497), (54.27342, 27.253426, 1, 7.7565904),
        (66.23869, 27.251177, 1, 7.6851616), (78.240105, 27.26005, 1, 4.873087),
        (90.26382, 27.241592, 1, 3.2701247), (102.24919, 27.248087, 1, 4.1313066),
        (18.251526, 39.28103, 1, 6.2057824), (30.204529, 39.25196, 1, 5.415462),
        (42.2558, 39.255287, 1, 7.084821), (54.26061, 39.260193, 1, 8.263677),
        (66.234436, 39.2537, 1, 7.1098356), (78.22165, 39.248993, 1, 3.9393425),
        (90.30903, 39.26565, 1, 2.8690476), (102.259125, 39.23907, 1, 4.3388605),
        (18.247824, 51.242928, 1, 7.6493764), (30.21812, 51.232716, 1, 5.708475),
        (42.251778, 51.236206, 1, 5.3746457), (54.267128, 51.2477, 1, 7.330995),
        (66.28052, 51.278534, 1, 8.451318), (78.20446, 51.28914, 1, 5.147605),
        (90.260994, 51.24733, 1, 5.0179987), (102.231804, 51.244335, 1, 5.884283),
        (18.245518, 63.24863, 1, 7.154833), (30.235348, 63.18927, 1, 4.020479),
        (42.249516, 63.28372, 1, 3.1979876), (54.290928, 63.2503, 1, 4.8037133),
        (66.42155, 63.248913, 1, 6.8232), (78.14016, 63.251743, 1, 5.946145),
        (90.271805, 63.25298, 1, 7.1186934), (102.25194, 63.249905, 1, 8.460459),
        (18.24645, 75.22871, 1, 4.256661), (30.237572, 75.24478, 1, 3.2581491),
        (42.24603, 75.24307, 1, 2.9410431), (54.26293, 75.18961, 1, 3.1353457),
        (66.26742, 75.13937, 1, 3.2358985), (78.17222, 75.16594, 1, 4.2426305),
        (90.25678, 75.230865, 1, 5.9321146), (102.23274, 75.252205, 1, 6.221301),
    ];
    #[rustfmt::skip]
    pub const SUB_EPS_0_001_COUNT_100: [FlowRow; 48] = [
        (18.265888, 15.289709, 1, 2.386267), (30.339857, 15.258692, 1, 3.2947137),
        (42.254074, 15.258692, 1, 4.2375994), (54.253468, 15.245655, 1, 4.1722646),
        (66.253204, 15.221356, 1, 4.4685373), (78.250496, 15.148251, 1, 3.222789),
        (90.23151, 15.200598, 1, 2.8828657), (102.285576, 15.241779, 1, 3.9654195),
        (18.257328, 27.118431, 1, 3.2513463), (30.265865, 27.236876, 1, 4.109127),
        (42.26146, 27.256393, 1, 6.732072), (54.273422, 27.253468, 1, 7.7565904),
        (66.238754, 27.251158, 1, 7.6850905), (78.24001, 27.26001, 1, 4.8723783),
        (90.26371, 27.241653, 1, 3.269983), (102.249275, 27.248028, 1, 4.131165),
        (18.251577, 39.28103, 1, 6.2060657), (30.204462, 39.251907, 1, 5.414966),
        (42.255753, 39.255257, 1, 7.0841837), (54.26056, 39.26016, 1, 8.263605),
        (66.23438, 39.253685, 1, 7.109694), (78.221634, 39.248924, 1, 3.9391298),
        (90.30897, 39.265602, 1, 2.8684099), (102.259026, 39.23893, 1, 4.338435),
        (18.247782, 51.24292, 1, 7.649093), (30.218065, 51.232624, 1, 5.7074122),
        (42.25176, 51.236214, 1, 5.3746457), (54.26713, 51.247684, 1, 7.330995),
        (66.28049, 51.27853, 1, 8.451318), (78.20455, 51.2892, 1, 5.147888),
        (90.2611, 51.24738, 1, 5.0186367), (102.2322, 51.24436, 1, 5.8848495),
        (18.245464, 63.248478, 1, 7.1539826), (30.235325, 63.189297, 1, 4.0202665),
        (42.249516, 63.28371, 1, 3.1982002), (54.290928, 63.250385, 1, 4.8039255),
        (66.421265, 63.248936, 1, 6.8224206), (78.1412, 63.25172, 1, 5.949192),
        (90.27184, 63.25303, 1, 7.119685), (102.25201, 63.249878, 1, 8.460034),
        (18.246548, 75.22873, 1, 4.2568026), (30.237608, 75.2447, 1, 3.2578657),
        (42.245815, 75.24314, 1, 2.9409013), (54.262882, 75.18961, 1, 3.1353457),
        (66.267334, 75.13965, 1, 3.2368197), (78.17167, 75.16596, 1, 4.240788),
        (90.25685, 75.23087, 1, 5.9323273), (102.23283, 75.25222, 1, 6.2217264),
    ];
    /// `video_lk.py` "subpixel": `(key, rows)`; keys that share a constant gave identical rows.
    #[rustfmt::skip]
    pub const SUBPIXEL: [(&str, &[FlowRow; 48]); 7] = [
        ("default", &SUB_DEFAULT),
        ("min_eig_output", &SUB_MIN_EIG_OUTPUT),
        ("win_10x10", &SUB_WIN_10X10),
        ("win_7x7", &SUB_WIN_7X7),
        ("max_level_2", &SUB_DEFAULT),
        ("max_level_0", &SUB_MAX_LEVEL_0),
        ("eps_0.001_count_100", &SUB_EPS_0_001_COUNT_100),
    ];

    #[rustfmt::skip]
    pub const EDGE_MAX0__MAE: [FlowRow; 15] = [
        (0.9798826, 0.019506335, 1, 12.44875), (39.41734, 34.957314, 1, 52.1825),
        (0.61849445, 0.7259242, 1, 14.93875), (55.21358, 34.276623, 0, 7.25509948477688e-39),
        (-20.0, -20.0, 0, 0.0), (-4.5, 5.0, 0, 0.0),
        (5.0, -4.5, 0, 0.0), (52.9, 10.0, 0, 0.0),
        (10.0, 36.9, 0, 0.0), (3.3465457, 2.3815918, 1, 0.53125),
        (27.619778, 21.068539, 1, 42.49375), (17.947151, 15.046791, 1, 8.31),
        (28.862576, 12.012643, 1, 76.68125), (51.921608, -0.44228387, 0, 9.918373714909483e-39),
        (1.0286138, 32.001907, 1, 33.92375),
    ];
    #[rustfmt::skip]
    pub const EDGE_MAX0__MAE__INITIAL_FLOW: [FlowRow; 15] = [
        (1.0, 0.0, 1, 12.36), (47.84492, 31.691664, 1, 82.07),
        (0.61815524, 0.72610474, 1, 14.93625), (52.87933, 34.508045, 0, 7.25509948477688e-39),
        (-19.0, -20.0, 0, 0.0), (-3.5, 5.0, 0, 0.0),
        (6.0, -4.5, 0, 0.0), (53.9, 10.0, 0, 0.0),
        (11.0, 36.9, 0, 0.0), (3.25, 2.5, 1, 0.0),
        (25.5, 16.5, 1, 0.0), (17.0, 16.0, 1, 0.0),
        (46.0, 30.0, 1, 0.0), (51.703003, -0.41236496, 0, 9.918373714909483e-39),
        (1.0, 32.0, 1, 31.8),
    ];
    #[rustfmt::skip]
    pub const EDGE_MAX0__MIN_EIG: [FlowRow; 15] = [
        (0.9798826, 0.019506335, 1, 0.019755859), (39.41734, 34.957314, 1, 0.047537956),
        (0.61849445, 0.7259242, 1, 0.009431724), (55.21358, 34.276623, 0, 0.010665808),
        (-20.0, -20.0, 0, 0.0), (-4.5, 5.0, 0, 0.0),
        (5.0, -4.5, 0, 0.0), (52.9, 10.0, 0, 0.0),
        (10.0, 36.9, 0, 0.0), (3.3465457, 2.3815918, 1, 0.01825966),
        (27.619778, 21.068539, 1, 0.117861345), (17.947151, 15.046791, 1, 0.2871521),
        (28.862576, 12.012643, 1, 0.24467048), (51.921608, -0.44228387, 0, 0.05377899),
        (1.0286138, 32.001907, 1, 0.22851606),
    ];
    #[rustfmt::skip]
    pub const EDGE_MAX0__MIN_EIG__INITIAL_FLOW: [FlowRow; 15] = [
        (1.0, 0.0, 1, 0.019755859), (47.84492, 31.691664, 1, 0.047537956),
        (0.61815524, 0.72610474, 1, 0.009431724), (52.87933, 34.508045, 0, 0.010665808),
        (-19.0, -20.0, 0, 0.0), (-3.5, 5.0, 0, 0.0),
        (6.0, -4.5, 0, 0.0), (53.9, 10.0, 0, 0.0),
        (11.0, 36.9, 0, 0.0), (3.25, 2.5, 1, 0.01825966),
        (25.5, 16.5, 1, 0.117861345), (17.0, 16.0, 1, 0.2871521),
        (46.0, 30.0, 1, 0.24467048), (51.703003, -0.41236496, 0, 0.05377899),
        (1.0, 32.0, 1, 0.22851606),
    ];
    #[rustfmt::skip]
    pub const EDGE_MAX2__MAE: [FlowRow; 15] = [
        (0.9772761, 0.021988988, 1, 12.4575), (50.93329, 38.44977, 0, 0.047537956),
        (0.61452115, 0.72993565, 1, 14.93125), (54.17506, 24.374868, 0, 0.010665808),
        (-20.0, -20.0, 0, 0.0), (-3.5359662, 6.9094267, 0, 0.0),
        (4.725651, -3.9194708, 0, 0.0), (52.9, 10.0, 0, 0.0),
        (11.201575, 38.105927, 0, 0.0), (2.972717, 2.848428, 1, 1.42),
        (25.240871, 14.940288, 1, 55.73625), (17.054546, 15.952773, 1, 6.6225),
        (31.173346, 14.14282, 1, 70.335), (42.026302, 35.260754, 0, 0.05377899),
        (-15.651218, 37.540924, 0, 0.22851606),
    ];
    #[rustfmt::skip]
    pub const EDGE_MAX2__MAE__INITIAL_FLOW: [FlowRow; 15] = [
        (0.97698724, 0.022382855, 1, 12.4575), (50.957825, 38.661003, 0, 0.047537956),
        (0.6145651, 0.7297085, 1, 14.93), (51.37838, 40.481148, 0, 0.010665808),
        (-19.0, -20.0, 0, 0.0), (-3.5355504, 6.930017, 0, 0.0),
        (4.7820134, -3.9347591, 0, 0.0), (53.9, 10.0, 0, 0.0),
        (11.2147, 38.104565, 0, 0.0), (12.036144, -6.915406, 0, 1.42),
        (28.280954, 23.795357, 1, 44.9525), (17.055386, 15.95204, 1, 6.7225),
        (30.62476, 13.566623, 1, 65.88875), (46.120476, 41.50602, 0, 0.05377899),
        (-12.948196, 35.94132, 0, 0.22851606),
    ];
    #[rustfmt::skip]
    pub const EDGE_MAX2__MIN_EIG: [FlowRow; 15] = [
        (0.9772761, 0.021988988, 1, 0.019755859), (50.93329, 38.44977, 0, 0.047537956),
        (0.61452115, 0.72993565, 1, 0.009431724), (54.17506, 24.374868, 0, 0.010665808),
        (-20.0, -20.0, 0, 0.0), (-3.5359662, 6.9094267, 0, 0.0),
        (4.725651, -3.9194708, 0, 0.0), (52.9, 10.0, 0, 0.0),
        (11.201575, 38.105927, 0, 0.0), (2.972717, 2.848428, 1, 0.01825966),
        (25.240871, 14.940288, 1, 0.117861345), (17.054546, 15.952773, 1, 0.2871521),
        (31.173346, 14.14282, 1, 0.24467048), (42.026302, 35.260754, 0, 0.05377899),
        (-15.651218, 37.540924, 0, 0.22851606),
    ];
    #[rustfmt::skip]
    pub const EDGE_MAX2__MIN_EIG__INITIAL_FLOW: [FlowRow; 15] = [
        (0.97698724, 0.022382855, 1, 0.019755859), (50.957825, 38.661003, 0, 0.047537956),
        (0.6145651, 0.7297085, 1, 0.009431724), (51.37838, 40.481148, 0, 0.010665808),
        (-19.0, -20.0, 0, 0.0), (-3.5355504, 6.930017, 0, 0.0),
        (4.7820134, -3.9347591, 0, 0.0), (53.9, 10.0, 0, 0.0),
        (11.2147, 38.104565, 0, 0.0), (12.036144, -6.915406, 0, 0.01825966),
        (28.280954, 23.795357, 1, 0.117861345), (17.055386, 15.95204, 1, 0.2871521),
        (30.62476, 13.566623, 1, 0.24467048), (46.120476, 41.50602, 0, 0.05377899),
        (-12.948196, 35.94132, 0, 0.22851606),
    ];
    /// `video_lk.py` "edges": `(key, rows)`; keys that share a constant gave identical rows.
    #[rustfmt::skip]
    pub const EDGES: [(&str, &[FlowRow; 15]); 8] = [
        ("max0/mae", &EDGE_MAX0__MAE),
        ("max0/mae/initial_flow", &EDGE_MAX0__MAE__INITIAL_FLOW),
        ("max0/min_eig", &EDGE_MAX0__MIN_EIG),
        ("max0/min_eig/initial_flow", &EDGE_MAX0__MIN_EIG__INITIAL_FLOW),
        ("max2/mae", &EDGE_MAX2__MAE),
        ("max2/mae/initial_flow", &EDGE_MAX2__MAE__INITIAL_FLOW),
        ("max2/min_eig", &EDGE_MAX2__MIN_EIG),
        ("max2/min_eig/initial_flow", &EDGE_MAX2__MIN_EIG__INITIAL_FLOW),
    ];

    /// `video_lk.py` "final_bounds": `(guess x, guess y, min-eig flag, status, err)` for the
    /// template (4, 4) on the 9x9 fixture, window 5x5, level 0, zero iterations.
    #[rustfmt::skip]
    pub const FINAL_BOUNDS: [(f32, f32, bool, u8, f32); 20] = [
        (100.0, 4.0, false, 0, 1.0),
        (100.0, 4.0, true, 1, 0.11571938),
        (4.0, 100.0, false, 0, 0.11571938),
        (4.0, 100.0, true, 1, 0.11571938),
        (-3.0, 4.0, false, 1, 71.96),
        (-3.0, 4.0, true, 1, 0.11571938),
        (-3.01, 4.0, false, 0, 0.11571938),
        (-3.01, 4.0, true, 1, 0.11571938),
        (4.0, -3.0, false, 1, 66.68),
        (4.0, -3.0, true, 1, 0.11571938),
        (4.0, -3.01, false, 0, 0.11571938),
        (4.0, -3.01, true, 1, 0.11571938),
        (10.99, 4.0, false, 1, 89.28125),
        (10.99, 4.0, true, 1, 0.11571938),
        (11.0, 4.0, false, 0, 0.11571938),
        (11.0, 4.0, true, 1, 0.11571938),
        (4.0, 10.99, false, 1, 88.1425),
        (4.0, 10.99, true, 1, 0.11571938),
        (4.0, 11.0, false, 0, 0.11571938),
        (4.0, 11.0, true, 1, 0.11571938),
    ];

    /// `video_lk.py` "asymmetric_borders": `(prev border, next border, MAE rows (count 0), step rows (count 1))`
    /// on the 9x9 fixture padded by 8 with `copyMakeBorder`; coordinates are already un-shifted.
    #[rustfmt::skip]
    pub const ASYMMETRIC_BORDERS: [(&str, &str, [FlowRow; 3], [FlowRow; 2]); 7] = [
        ("reflect101", "reflect101", [(0.0, 0.0, 1, 0.0), (8.0, 8.0, 1, 0.0), (0.25, 0.5, 1, 0.0)], [(-0.43229485, 1.9961214, 1, 98.3675), (7.739274, 9.0620575, 1, 63.10625)]),
        ("reflect101", "constant", [(0.0, 0.0, 1, 28.44), (8.0, 8.0, 1, 53.56), (0.25, 0.5, 1, 24.47)], [(-0.23009062, 2.2676878, 1, 127.98125), (7.2506332, 9.703224, 1, 117.44375)]),
        ("reflect101", "replicate", [(0.0, 0.0, 1, 21.24), (8.0, 8.0, 1, 43.56), (0.25, 0.5, 1, 16.55)], [(-0.27146053, 2.1477776, 1, 116.19375), (8.652649, 8.673058, 1, 38.36875)]),
        ("reflect101", "wrap", [(0.0, 0.0, 1, 66.4), (8.0, 8.0, 1, 30.32), (0.25, 0.5, 1, 62.66)], [(-0.9690957, 1.3171539, 1, 64.1325), (8.073093, 8.093981, 1, 39.5125)]),
        ("constant", "reflect101", [(0.0, 0.0, 1, 28.44), (8.0, 8.0, 1, 53.56), (0.25, 0.5, 1, 24.47)], [(-0.43229485, 1.9961214, 1, 98.3675), (7.739274, 9.0620575, 1, 63.10625)]),
        ("replicate", "reflect101", [(0.0, 0.0, 1, 21.24), (8.0, 8.0, 1, 43.56), (0.25, 0.5, 1, 16.55)], [(-0.43229485, 1.9961214, 1, 98.3675), (7.739274, 9.0620575, 1, 63.10625)]),
        ("wrap", "reflect101", [(0.0, 0.0, 1, 66.4), (8.0, 8.0, 1, 30.32), (0.25, 0.5, 1, 62.66)], [(-0.43229485, 1.9961214, 1, 98.3675), (7.739274, 9.0620575, 1, 63.10625)]),
    ];

    /// `video_lk.py` "termination": `(TermCriteria type (1 COUNT, 2 EPS, 3 both), count, epsilon, x, y, row)`
    /// on the 33x49 fixture and its 1 px shift, window 5x5, level 0.
    #[rustfmt::skip]
    pub const TERMINATION: [(i32, i32, f64, f32, f32, FlowRow); 13] = [
        (1, 30, 99.0, 2.25, 2.5, (3.7242084, 1.9445041, 1, 2.255)),
        (3, 30, 0.0, 2.25, 2.5, (3.7224483, 1.9461701, 1, 2.2525)),
        (3, 30, -0.5, 2.25, 2.5, (3.7224483, 1.9461701, 1, 2.2525)),
        (1, -3, 0.0, 2.25, 2.5, (2.25, 2.5, 1, 20.825)),
        (1, 0, 0.0, 2.25, 2.5, (2.25, 2.5, 1, 20.825)),
        (2, 1, 0.03, 2.25, 2.5, (3.3465457, 2.3815918, 1, 0.53125)),
        (3, 30, 0.03, 2.25, 2.5, (3.3465457, 2.3815918, 1, 0.53125)),
        (3, 30, 10.0, 2.25, 2.5, (3.3451042, 2.4076617, 1, 0.60875)),
        (3, 30, 20.0, 2.25, 2.5, (3.3451042, 2.4076617, 1, 0.60875)),
        (1, 100, 0.0, 8.125, 14.375, (9.3683815, 13.540901, 1, 24.81125)),
        (1, 120, 0.0, 8.125, 14.375, (9.3683815, 13.540901, 1, 24.81125)),
        (2, 1, 0.03, 8.125, 14.375, (9.188091, 14.111875, 1, 10.82375)),
        (3, 30, 0.03, 8.125, 14.375, (9.188091, 14.111875, 1, 10.82375)),
    ];

    /// `video_lk.py` "inputs_d2": the same two points tracked on 1-channel and on 3-channel
    /// (replicated) images, window 5x5.
    #[rustfmt::skip]
    pub const D2_U8C1: [FlowRow; 2] = [(17.937923, 15.063578, 1, 9.7625), (30.995584, 20.006021, 1, 1.0075)];
    #[rustfmt::skip]
    pub const D2_U8C3: [FlowRow; 2] = [(17.937923, 15.063578, 1, 9.7625), (30.99558, 20.006021, 1, 1.0075)];
}

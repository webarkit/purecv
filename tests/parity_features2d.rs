/*
 *  parity_features2d.rs
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

//! OpenCV 4.10.0 parity pins for the `features2d` module (`docs/parity/4.10.0/features2d.md`).
//!
//! Every expected value in this file was produced by a script under `scripts/opencv_ref/`
//! (`features2d_fast.py`, `features2d_orb.py`, `features2d_matcher.py`, `features2d_draw.py`,
//! `features2d_keypoint.py`; helper `_features2d_common.py`) run in `cpp_ref/.venv`
//! (opencv-contrib-python 4.10.0.84, `cv2.ipp.setUseIPP(False)`, Windows x86-64). The script
//! and the key of each block are named in the comment above it. The inputs are rebuilt here
//! from the same integer formulas the scripts use, and long keypoint lists are compared
//! through the same FNV-1a digest of their little-endian field bytes.
//!
//! Tests that pass pin rows that already match OpenCV. Tests marked `#[ignore]` document a
//! `diverges` row; they fail when run (`cargo test --test parity_features2d -- --ignored`)
//! and are the executable form of the measurement; the issue in each reason tracks the fix.
//!
//! A fix for #N removes the `#[ignore]` of the tests that cite it and deletes the matching
//! current-behaviour pin, if any; the test then guards the row (rule 11 of
//! `docs/parity/AGENTS.md`).

use purecv::core::types::{Point2f, Scalar};
use purecv::core::Matrix;
use purecv::features2d::matcher::DescriptorDistance;
use purecv::features2d::*;

use ocv::*;

// ---------------------------------------------------------------------------------------
// Inputs (the same formulas as the scripts; `_features2d_common.py`)
// ---------------------------------------------------------------------------------------

/// One step of the 32-bit LCG `state * 1664525 + 1013904223 (mod 2^32)`.
fn lcg_step(s: u32) -> u32 {
    s.wrapping_mul(1664525).wrapping_add(1013904223)
}

/// Row-major LCG noise: each pixel is the top byte of the next LCG state (`noise_u8`).
fn noise_u8(rows: usize, cols: usize, seed: u32) -> Matrix<u8> {
    let mut s = seed;
    let mut d = Vec::with_capacity(rows * cols);
    for _ in 0..rows * cols {
        s = lcg_step(s);
        d.push((s >> 24) as u8);
    }
    Matrix::from_vec(rows, cols, 1, d)
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

/// Two bright squares on a dark 40x48 background (`squares_u8`).
fn squares_u8() -> Matrix<u8> {
    let (rows, cols) = (40, 48);
    let mut d = vec![30u8; rows * cols];
    for y in 4..12 {
        for x in 4..12 {
            d[y * cols + x] = 200;
        }
    }
    for y in 16..28 {
        for x in 20..34 {
            d[y * cols + x] = 220;
        }
    }
    Matrix::from_vec(rows, cols, 1, d)
}

/// "lowbias32" hash; twin of `hash32()` in `gen_orb_descriptors.js` / `_features2d_common.py`.
fn hash32(mut x: u32) -> u32 {
    x ^= x >> 16;
    x = x.wrapping_mul(0x7feb_352d);
    x ^= x >> 15;
    x = x.wrapping_mul(0x846c_a68b);
    x ^ (x >> 16)
}

/// The 192x192 image of the ORB fixture `src/features2d/orb_opencv_ref.rs` (`orb_ref_image`).
fn orb_ref_image() -> Matrix<u8> {
    let (w, h) = (192u32, 192u32);
    let mut data = Vec::with_capacity((w * h) as usize);
    for y in 0..h {
        for x in 0..w {
            let coarse = hash32(((y >> 3) * 64 + (x >> 3)) ^ 0x9e37_79b9) & 0xff;
            let fine = hash32((y * w + x) ^ 0x85eb_ca6b) & 0x3f;
            let mut v = ((coarse * 3) >> 2) + fine;
            if (20..84).contains(&x) && (28..92).contains(&y) {
                v = 255 - v;
            }
            if (100..172).contains(&x) && (56..124).contains(&y) {
                v = (v >> 1) + 96;
            }
            if (48..144).contains(&x) && (132..176).contains(&y) {
                v ^= 0x5a;
            }
            data.push((v & 0xff) as u8);
        }
    }
    Matrix::from_vec(h as usize, w as usize, 1, data)
}

/// The 40 level-0 keypoints of the ORB fixture (`makeKeypoints()` in `gen_orb_descriptors.js`).
fn fixture_keypoints() -> Vec<KeyPoint> {
    const ANGLES: [f32; 20] = [
        0.0, 17.5, 90.0, 133.3, 270.9, 45.0, 180.0, 359.5, 222.2, 300.1, 11.25, 60.0, 120.0, 243.7,
        315.0, 5.5, 72.8, 199.9, 30.0, 150.0,
    ];
    (0..40)
        .map(|i| {
            KeyPoint::new(
                Point2f::new(
                    (40 + (i * 37 + 11) % 112) as f32,
                    (40 + (i * 59 + 23) % 112) as f32,
                ),
                31.0,
                ANGLES[i % 20],
                0.0,
                0,
                -1,
            )
        })
        .collect()
}

fn fnv1a32(bytes: &[u8]) -> u32 {
    let mut h = 0x811c_9dc5u32;
    for &b in bytes {
        h ^= b as u32;
        h = h.wrapping_mul(0x0100_0193);
    }
    h
}

/// FNV-1a of `<fffffii>` per keypoint (pt.x, pt.y, size, angle, response, octave, class_id),
/// the scripts' `kp_digest()["fnv1a"]`.
fn kp_digest(kps: &[KeyPoint]) -> u32 {
    let mut blob = Vec::with_capacity(kps.len() * 28);
    for k in kps {
        blob.extend_from_slice(&k.pt.x.to_le_bytes());
        blob.extend_from_slice(&k.pt.y.to_le_bytes());
        blob.extend_from_slice(&k.size.to_le_bytes());
        blob.extend_from_slice(&k.angle.to_le_bytes());
        blob.extend_from_slice(&k.response.to_le_bytes());
        blob.extend_from_slice(&k.octave.to_le_bytes());
        blob.extend_from_slice(&k.class_id.to_le_bytes());
    }
    fnv1a32(&blob)
}

/// FNV-1a of `<ff>` per keypoint (pt.x, pt.y), the scripts' `kp_digest()["fnv1a_xy"]`.
fn kp_digest_xy(kps: &[KeyPoint]) -> u32 {
    let mut blob = Vec::with_capacity(kps.len() * 8);
    for k in kps {
        blob.extend_from_slice(&k.pt.x.to_le_bytes());
        blob.extend_from_slice(&k.pt.y.to_le_bytes());
    }
    fnv1a32(&blob)
}

fn fast_type(name: &str) -> FastType {
    match name {
        "9_16" => FastType::Type9_16,
        "7_12" => FastType::Type7_12,
        "5_8" => FastType::Type5_8,
        _ => unreachable!(),
    }
}

/// The three FAST images of `features2d_fast.py`, by name.
fn fast_image(name: &str) -> Matrix<u8> {
    match name {
        "noise_64x80_s7" => noise_u8(64, 80, 7),
        "textured_48x64_s3" => textured_u8(48, 64, 3),
        "squares_40x48" => squares_u8(),
        _ => unreachable!(),
    }
}

/// Runs one `FAST_GRID` key (`<image>/<type>/nonmax_<on|off>/t<threshold>`).
fn fast_grid_case(key: &str) -> Vec<KeyPoint> {
    let parts: Vec<&str> = key.split('/').collect();
    let nonmax = parts[2] == "nonmax_on";
    let threshold: u8 = parts[3][1..].parse().unwrap();
    FastFeatureDetector::new(threshold, nonmax, fast_type(parts[1]))
        .detect(&fast_image(parts[0]))
        .unwrap()
}

fn hamming(a: &[u8], b: &[u8]) -> u32 {
    a.iter().zip(b).map(|(x, y)| (x ^ y).count_ones()).sum()
}

fn dm_tuple(m: &DMatch) -> (i32, i32, i32, f32) {
    (m.query_idx, m.train_idx, m.train_img_idx, m.distance)
}

fn assert_matches(actual: &[DMatch], expected: &[(i32, i32, i32, f32)], what: &str) {
    assert_eq!(actual.len(), expected.len(), "{what}: count");
    for (i, (a, e)) in actual.iter().zip(expected).enumerate() {
        let a = dm_tuple(a);
        assert_eq!((a.0, a.1, a.2), (e.0, e.1, e.2), "{what}[{i}]: indices");
        assert!(
            (a.3 - e.3).abs() <= 1e-5 * e.3.abs().max(1.0),
            "{what}[{i}]: distance purecv {} vs OpenCV {}",
            a.3,
            e.3
        );
    }
}

fn assert_knn(actual: &[Vec<DMatch>], expected: &[&[(i32, i32, i32, f32)]], what: &str) {
    assert_eq!(actual.len(), expected.len(), "{what}: rows");
    for (i, (a, e)) in actual.iter().zip(expected).enumerate() {
        assert_matches(a, e, &format!("{what}[{i}]"));
    }
}

/// The matcher inputs of `features2d_matcher.py` (`u8_sets`, `f32_sets`).
fn matcher_u8_sets() -> (Matrix<u8>, Matrix<u8>) {
    let mut q = noise_u8(6, 32, 101);
    let mut t = noise_u8(8, 32, 202);
    for c in 0..32 {
        let v = *t.get(1, c, 0).unwrap();
        t.set(5, c, 0, v);
        let v = *q.get(2, c, 0).unwrap();
        q.set(4, c, 0, v);
        let v = *q.get(0, c, 0).unwrap();
        t.set(7, c, 0, v);
    }
    (q, t)
}

fn matcher_f32_sets() -> (Matrix<f32>, Matrix<f32>) {
    let qn = noise_u8(5, 4, 303);
    let tn = noise_u8(6, 4, 404);
    let scale = |m: &Matrix<u8>| {
        Matrix::from_vec(
            m.rows,
            m.cols,
            1,
            m.data.iter().map(|&v| v as f32 / 16.0).collect(),
        )
    };
    let mut q = scale(&qn);
    let mut t = scale(&tn);
    for c in 0..4 {
        let v = *t.get(0, c, 0).unwrap();
        t.set(3, c, 0, v);
        let v = *q.get(1, c, 0).unwrap();
        q.set(2, c, 0, v);
        let v = *q.get(4, c, 0).unwrap();
        t.set(5, c, 0, v);
    }
    (q, t)
}

fn kp(x: f32, y: f32, size: f32, angle: f32, octave: i32) -> KeyPoint {
    KeyPoint::new(Point2f::new(x, y), size, angle, 0.0, octave, -1)
}

/// Pixels of the 3-channel `after` that differ from `before` (same size), as
/// `(y, x, [c0, c1, c2])`, the scripts' `changed()`.
fn changed_pixels(before: &Matrix<u8>, after: &Matrix<u8>) -> Vec<(usize, usize, [u8; 3])> {
    assert_eq!(after.channels, 3, "a BGR output is expected");
    let mut out = Vec::new();
    for y in 0..after.rows {
        for x in 0..after.cols {
            let px = |m: &Matrix<u8>| [0, 1, 2].map(|c| m.get(y, x, c).copied().unwrap_or(0));
            let (b, a) = (px(before), px(after));
            if a != b {
                out.push((y, x, a));
            }
        }
    }
    out
}

/// The 20x24 BGR image of `features2d_draw.py` (channels from the LCG seeds 41, 42, 43).
fn draw_bgr_image() -> Matrix<u8> {
    let planes = [
        noise_u8(20, 24, 41),
        noise_u8(20, 24, 42),
        noise_u8(20, 24, 43),
    ];
    let mut out = Matrix::<u8>::new(20, 24, 3);
    for y in 0..20 {
        for x in 0..24 {
            for (c, plane) in planes.iter().enumerate() {
                out.set(y, x, c, *plane.get(y, x, 0).unwrap());
            }
        }
    }
    out
}

fn gray_to_3ch(m: &Matrix<u8>) -> Matrix<u8> {
    let mut out = Matrix::<u8>::new(m.rows, m.cols, 3);
    for y in 0..m.rows {
        for x in 0..m.cols {
            let v = *m.get(y, x, 0).unwrap();
            for c in 0..3 {
                out.set(y, x, c, v);
            }
        }
    }
    out
}

fn per_octave(kps: &[KeyPoint], nlevels: usize) -> Vec<usize> {
    let mut c = vec![0usize; nlevels];
    for k in kps {
        c[k.octave as usize] += 1;
    }
    c
}

// ---------------------------------------------------------------------------------------
// FAST (features2d_fast.py, "grid", "misc", "arcs")
// ---------------------------------------------------------------------------------------

/// `Type9_16` with non-maximum suppression (ORB's configuration): every keypoint, in order,
/// with the same `(pt, size 7, angle -1, response, octave 0, class_id -1)`, on three images
/// at thresholds 0, 10, 20 and 40.
#[test]
fn fast_type9_16_nonmax_on_matches_opencv() {
    for &(key, count, digest, _) in FAST_GRID
        .iter()
        .filter(|r| r.0.contains("/9_16/nonmax_on/"))
    {
        let kps = fast_grid_case(key);
        assert_eq!(kps.len(), count, "{key}: count");
        assert_eq!(
            kp_digest(&kps),
            digest,
            "{key}: keypoints differ from OpenCV"
        );
    }
}

/// `Type9_16` with non-maximum suppression off and threshold >= 10: the same positions in
/// the same order (the responses differ, see `fast_nonmax_off_response_and_threshold_0`).
#[test]
fn fast_type9_16_nonmax_off_positions_match_opencv_above_threshold_0() {
    for &(key, count, _, digest_xy) in FAST_GRID
        .iter()
        .filter(|r| r.0.contains("/9_16/nonmax_off/") && !r.0.ends_with("/t0"))
    {
        let kps = fast_grid_case(key);
        assert_eq!(kps.len(), count, "{key}: count");
        assert_eq!(
            kp_digest_xy(&kps),
            digest_xy,
            "{key}: positions differ from OpenCV"
        );
    }
}

/// OpenCV scans rows and columns `3 ..= dim - 4` for every pattern size; purecv does so for
/// `Type9_16` and `Type7_12` (radius 3). Images of 7x7 and smaller yield nothing.
#[test]
fn fast_scan_range_matches_opencv_for_type9_16_and_7_12() {
    let noise = noise_u8(64, 80, 7);
    for t in [FastType::Type9_16, FastType::Type7_12] {
        let kps = FastFeatureDetector::new(0, false, t)
            .detect(&noise)
            .unwrap();
        let xs: Vec<f32> = kps.iter().map(|k| k.pt.x).collect();
        let ys: Vec<f32> = kps.iter().map(|k| k.pt.y).collect();
        assert_eq!(xs.iter().cloned().fold(f32::MAX, f32::min), 3.0);
        assert_eq!(xs.iter().cloned().fold(f32::MIN, f32::max), 76.0);
        assert_eq!(ys.iter().cloned().fold(f32::MAX, f32::min), 3.0);
        assert_eq!(ys.iter().cloned().fold(f32::MIN, f32::max), 60.0);
    }
    for &(n, count_9_16, _, _) in &FAST_TINY_COUNTS {
        let tiny = textured_u8(n, n, 11);
        let kps = FastFeatureDetector::new(10, false, FastType::Type9_16)
            .detect(&tiny)
            .unwrap();
        assert_eq!(kps.len(), count_9_16, "{n}x{n} Type9_16");
    }
}

/// With non-maximum suppression off, OpenCV's CPU path never computes the score, so every
/// keypoint has `response = 0`; at threshold 0 it also keeps corners whose score is 0
/// (1718 instead of 1676 on the noise image). purecv reports the score and drops those.
#[test]
#[ignore = "diverges from OpenCV 4.10.0, see #201"]
fn fast_nonmax_off_response_and_threshold_0_match_opencv() {
    for &(key, count, digest, _) in FAST_GRID
        .iter()
        .filter(|r| r.0.contains("/9_16/nonmax_off/"))
    {
        let kps = fast_grid_case(key);
        assert_eq!(kps.len(), count, "{key}: count");
        assert!(kps.iter().all(|k| k.response == 0.0), "{key}: responses");
        assert_eq!(kp_digest(&kps), digest, "{key}");
    }
}

/// OpenCV's scalar pre-test reads `pixel[0..15]` in pairs for every pattern size, so for
/// `Type7_12` only 7-arcs starting at 0, 9, 10 or 11 always pass and `Type5_8` becomes an
/// 8-of-8 test; `Type5_8` also keeps the 3-pixel border. purecv runs the correct segment
/// tests (a superset) and scans `Type5_8` from pixel 1.
#[test]
#[ignore = "diverges from OpenCV 4.10.0, see #202"]
fn fast_type7_12_and_5_8_match_opencv() {
    for &(key, count, digest, _) in FAST_GRID
        .iter()
        .filter(|r| r.0.contains("/7_12/") || r.0.contains("/5_8/"))
    {
        let kps = fast_grid_case(key);
        assert_eq!(kps.len(), count, "{key}: count");
        assert_eq!(kp_digest(&kps), digest, "{key}");
    }
    // the squares: OpenCV finds the six corners with Type7_12 + nonmax (purecv none)
    let kps = FastFeatureDetector::new(20, true, FastType::Type7_12)
        .detect(&squares_u8())
        .unwrap();
    let got: Vec<(f32, f32, f32)> = kps.iter().map(|k| (k.pt.x, k.pt.y, k.response)).collect();
    assert_eq!(got, FAST_SQUARES_7_12_NONMAX_ON.to_vec());
    // hand-built arcs: which contiguous 7-of-12 / 5-of-8 arcs are reported at the centre
    let ring = |offs: &[(i32, i32)], bright: &[usize]| {
        let size = 9usize;
        let mut d = vec![100u8; size * size];
        let c = (size / 2) as i32;
        for &i in bright {
            let (dx, dy) = offs[i];
            d[((c + dy) as usize) * size + (c + dx) as usize] = 200;
        }
        Matrix::from_vec(size, size, 1, d)
    };
    let off8: [(i32, i32); 8] = [
        (0, 1),
        (1, 1),
        (1, 0),
        (1, -1),
        (0, -1),
        (-1, -1),
        (-1, 0),
        (-1, 1),
    ];
    let off12: [(i32, i32); 12] = [
        (0, 2),
        (1, 2),
        (2, 1),
        (2, 0),
        (2, -1),
        (1, -2),
        (0, -2),
        (-1, -2),
        (-2, -1),
        (-2, 0),
        (-2, 1),
        (-1, 2),
    ];
    let centre = |kps: &[KeyPoint]| kps.iter().any(|k| k.pt.x == 4.0 && k.pt.y == 4.0);
    let found12: Vec<usize> = (0..12)
        .filter(|&start| {
            let arc: Vec<usize> = (0..7).map(|i| (start + i) % 12).collect();
            let kps = FastFeatureDetector::new(10, false, FastType::Type7_12)
                .detect(&ring(&off12, &arc))
                .unwrap();
            centre(&kps)
        })
        .collect();
    assert_eq!(
        found12,
        FAST_ARC_7_OF_12_STARTS_FOUND.to_vec(),
        "7-of-12 arcs found at the centre"
    );
    let found8: Vec<usize> = (0..8)
        .filter(|&start| {
            let arc: Vec<usize> = (0..5).map(|i| (start + i) % 8).collect();
            let kps = FastFeatureDetector::new(10, false, FastType::Type5_8)
                .detect(&ring(&off8, &arc))
                .unwrap();
            centre(&kps)
        })
        .collect();
    assert_eq!(
        found8,
        FAST_ARC_5_OF_8_STARTS_FOUND.to_vec(),
        "5-of-8 arcs found at the centre"
    );
    for &(n, _, count_7_12, count_5_8) in &FAST_TINY_COUNTS {
        let tiny = textured_u8(n, n, 11);
        for (t, count) in [
            (FastType::Type7_12, count_7_12),
            (FastType::Type5_8, count_5_8),
        ] {
            let kps = FastFeatureDetector::new(10, false, t)
                .detect(&tiny)
                .unwrap();
            assert_eq!(kps.len(), count, "{n}x{n} {t:?}");
        }
    }
}

/// `FastFeatureDetector_Impl::detect` and `ORB_Impl::detectAndCompute` convert a
/// multi-channel image with `COLOR_BGR2GRAY`; purecv returns `InvalidInput`.
#[test]
#[ignore = "diverges from OpenCV 4.10.0, see #252"]
fn fast_and_orb_accept_multichannel_input_like_opencv() {
    let gray = noise_u8(64, 80, 7);
    let bgr = gray_to_3ch(&gray);
    let det = FastFeatureDetector::new(20, true, FastType::Type9_16);
    assert_eq!(
        kp_digest(&det.detect(&bgr).unwrap()),
        kp_digest(&det.detect(&gray).unwrap())
    );
    let textured = textured_u8(240, 320, 5);
    let orb = Orb::default();
    assert_eq!(
        kp_digest(&orb.detect(&gray_to_3ch(&textured)).unwrap()),
        kp_digest(&orb.detect(&textured).unwrap())
    );
}

// ---------------------------------------------------------------------------------------
// ORB (features2d_orb.py, "detect", "variants", "compute_fixture", "compute_provided",
// "budget_emulated")
// ---------------------------------------------------------------------------------------

/// The per-level feature budget (`nfeaturesPerLevel`): with the defaults and
/// `edge_threshold = 15` every level of the 240x320 textured image is at OpenCV's quota
/// (`budget_emulated/n500_l8_s1.2`) and purecv detects exactly those counts, as OpenCV does
/// (`variants/edge_15`). With `nfeatures = 100` the quota is 22/18/15/13/10/9/7/6
/// (`budget_emulated/n100_l8_s1.2`); both libraries detect 22/18/15/13/10/9/7/5
/// (`variants/nfeatures_100`), the last level one short of its quota. Octave 0 is at quota
/// (109) on all three images.
#[test]
fn orb_per_level_budget_matches_opencv() {
    let textured = textured_u8(240, 320, 5);
    let kps = Orb::new(500, 1.2, 8, 15, 0, 2, ScoreType::Harris, 31, 20)
        .detect(&textured)
        .unwrap();
    assert_eq!(kps.len(), 500);
    let counts = per_octave(&kps, 8);
    assert_eq!(
        counts,
        ORB_EDGE15_PER_OCTAVE.to_vec(),
        "edge 15: OpenCV's counts"
    );
    assert_eq!(
        counts,
        ORB_BUDGET_500_8.to_vec(),
        "edge 15: every level at quota"
    );
    let kps = Orb::new(100, 1.2, 8, 31, 0, 2, ScoreType::Harris, 31, 20)
        .detect(&textured)
        .unwrap();
    let counts = per_octave(&kps, 8);
    assert_eq!(
        counts,
        ORB_DETECT_100_8_PER_OCTAVE.to_vec(),
        "nfeatures 100: OpenCV's counts"
    );
    assert_eq!(
        counts[..7],
        ORB_BUDGET_100_8[..7],
        "nfeatures 100: levels 0-6 at quota"
    );
    assert!(
        counts[7] < ORB_BUDGET_100_8[7],
        "nfeatures 100: level 7 below its quota"
    );
    for img in [textured, noise_u8(240, 320, 9), orb_ref_image()] {
        let kps = Orb::default().detect(&img).unwrap();
        assert_eq!(kps.iter().filter(|k| k.octave == 0).count(), 109);
    }
}

/// `detect` + `compute` on the same image equals `detect_and_compute` (both libraries).
#[test]
fn orb_detect_then_compute_equals_detect_and_compute() {
    let textured = textured_u8(240, 320, 5);
    let orb = Orb::default();
    let (kps, desc) = orb.detect_and_compute(&textured).unwrap();
    let kps2 = orb.detect(&textured).unwrap();
    assert_eq!(kp_digest(&kps2), kp_digest(&kps));
    assert_eq!(orb.compute(&textured, &kps2).unwrap().data, desc.data);
}

/// `Orb::compute` on the 40-keypoint fixture of `orb_opencv_ref.rs`: the cv2 wheel's
/// descriptors are byte-identical to the opencv.js fixture (40/40 rows), and purecv is
/// within 3 bits of them (mean 1.20); the gap is the u8 Gaussian blur (#229).
#[test]
fn orb_compute_fixture_descriptors_within_3_bits_of_opencv() {
    let desc = Orb::default()
        .compute(&orb_ref_image(), &fixture_keypoints())
        .unwrap();
    assert_eq!((desc.rows, desc.cols), (40, 32));
    let dist: Vec<u32> = desc
        .data
        .chunks_exact(32)
        .zip(ORB_FIXTURE_DESC.iter())
        .map(|(a, b)| hamming(a, b))
        .collect();
    let mean = dist.iter().sum::<u32>() as f64 / dist.len() as f64;
    let max = *dist.iter().max().unwrap();
    assert!(
        max <= 3 && mean <= 1.5,
        "mean {mean:.2}, max {max}, per keypoint {dist:?}"
    );
}

/// Steered BRIEF: OpenCV samples the level blurred with its bit-exact `ufixedpoint16`
/// 7x7/sigma2 Gaussian; purecv's `gaussian_blur` truncates and differs by 1 on about half of the
/// pixels (#229, #157). On the hash-textured fixture that flips 1.20 bits per descriptor; on
/// the flat-rectangle image, with OpenCV's own octave-0 keypoints and angles, 19.1 bits
/// (max 42).
#[test]
#[ignore = "diverges from OpenCV 4.10.0, see #229 and #157"]
fn orb_compute_descriptors_match_opencv_exactly() {
    let desc = Orb::default()
        .compute(&orb_ref_image(), &fixture_keypoints())
        .unwrap();
    for (i, (a, b)) in desc
        .data
        .chunks_exact(32)
        .zip(ORB_FIXTURE_DESC.iter())
        .enumerate()
    {
        assert_eq!(a, b, "fixture keypoint {i}");
    }
    let textured = textured_u8(240, 320, 5);
    let kps: Vec<KeyPoint> = ORB_TEXTURED_OCTAVE0
        .iter()
        .map(|&(x, y, angle, _)| kp(x, y, 31.0, angle, 0))
        .collect();
    let desc = Orb::default().compute(&textured, &kps).unwrap();
    for (i, (a, b)) in desc
        .data
        .chunks_exact(32)
        .zip(ORB_TEXTURED_OCTAVE0_DESC.iter())
        .enumerate()
    {
        assert_eq!(a, b, "textured octave-0 keypoint {i}");
    }
}

/// `compute_orientation` at OpenCV's 109 octave-0 keypoints of the textured image agrees
/// with OpenCV's `ICAngles` + `fastAtan2` to 0.01 degrees.
#[test]
fn orb_orientation_within_0_01_degree_of_opencv() {
    let textured = textured_u8(240, 320, 5);
    let umax = precompute_umax(15);
    let mut max_diff = 0.0f32;
    for &(x, y, angle, _) in &ORB_TEXTURED_OCTAVE0 {
        let a = compute_orientation(&textured, x as usize, y as usize, 31, &umax).unwrap();
        let d = (a - angle).abs();
        max_diff = max_diff.max(d.min(360.0 - d));
    }
    assert!(max_diff <= 0.01, "max angle difference {max_diff} degrees");
}

/// OpenCV computes the orientation with `fastAtan2` (a polynomial, documented accuracy
/// about  0.3 degrees); purecv uses the exact `atan2`. The two differ by up to 0.009 degrees here, which is
/// enough to move `cvRound`ed BRIEF sample positions.
#[test]
#[ignore = "diverges from OpenCV 4.10.0, see #250"]
fn orb_orientation_matches_opencv_fast_atan2() {
    let textured = textured_u8(240, 320, 5);
    let umax = precompute_umax(15);
    for &(x, y, angle, _) in &ORB_TEXTURED_OCTAVE0 {
        let a = compute_orientation(&textured, x as usize, y as usize, 31, &umax).unwrap();
        let d = (a - angle).abs();
        assert!(
            d.min(360.0 - d) <= 1e-4,
            "({x}, {y}): purecv {a} degrees vs OpenCV {angle} degrees"
        );
    }
}

/// Which FAST candidates survive at octave 0 (same pixels in both libraries): OpenCV keeps
/// `retainBest(2 x quota)` on the FAST score, scores them with a 7x7 integer Harris window
/// and keeps `retainBest(quota)` with ties; purecv scores every candidate with the 3x3
/// `corner_harris` definition and truncates. 92 of 109 positions coincide.
#[test]
#[ignore = "diverges from OpenCV 4.10.0, see #158"]
fn orb_octave0_keypoint_selection_matches_opencv() {
    let kps = Orb::default().detect(&textured_u8(240, 320, 5)).unwrap();
    let mut got: Vec<(f32, f32)> = kps
        .iter()
        .filter(|k| k.octave == 0)
        .map(|k| (k.pt.x, k.pt.y))
        .collect();
    got.sort_by(|a, b| a.partial_cmp(b).unwrap());
    assert_eq!(got, ORB_TEXTURED_OCTAVE0_XY.to_vec());
}

/// Octaves >= 1 are detected on a pyramid that OpenCV builds with `INTER_LINEAR_EXACT`,
/// `cvRound` sizes and reflect-101 padding; purecv's levels differ by +/-1 on many pixels, so
/// the per-octave counts below quota differ (textured: 89/56/25 vs 87/55/26).
#[test]
#[ignore = "diverges from OpenCV 4.10.0, see #193 and #236"]
fn orb_per_octave_counts_match_opencv_above_octave_0() {
    let orb = Orb::default();
    let kps = orb.detect(&textured_u8(240, 320, 5)).unwrap();
    assert_eq!(per_octave(&kps, 8), ORB_TEXTURED_PER_OCTAVE.to_vec());
    let kps = orb.detect(&noise_u8(240, 320, 9)).unwrap();
    assert_eq!(per_octave(&kps, 8), ORB_NOISE_PER_OCTAVE.to_vec());
    let kps = orb.detect(&orb_ref_image()).unwrap();
    assert_eq!(per_octave(&kps, 6), ORB_REF_PER_OCTAVE.to_vec());
}

/// `KeyPoint.size = patchSize * layerScale[octave]` in OpenCV; purecv leaves FAST's 7.
#[test]
#[ignore = "diverges from OpenCV 4.10.0, see #194"]
fn orb_keypoint_size_matches_opencv() {
    let kps = Orb::default().detect(&textured_u8(240, 320, 5)).unwrap();
    for k in &kps {
        assert_eq!(
            k.size, ORB_SIZE_PER_OCTAVE[k.octave as usize],
            "octave {}",
            k.octave
        );
    }
}

/// OpenCV pads every level with reflect-101, so `edgeThreshold` below `patchSize / 2` works
/// (500 keypoints at 0, 3, 10 and 14); purecv's `compute_orientation` returns `Err`.
#[test]
#[ignore = "diverges from OpenCV 4.10.0, see #193"]
fn orb_edge_threshold_below_half_patch_matches_opencv() {
    let textured = textured_u8(240, 320, 5);
    for edge in [0, 3, 10, 14] {
        let kps = Orb::new(500, 1.2, 8, edge, 0, 2, ScoreType::Harris, 31, 20)
            .detect(&textured)
            .unwrap_or_else(|e| panic!("edge_threshold {edge}: {e}"));
        assert_eq!(kps.len(), ORB_EDGE0_COUNT, "edge_threshold {edge}");
    }
}

/// `compute()` with caller-supplied keypoints: OpenCV drops keypoints within
/// `edgeThreshold` of the original image border (`cvRound`, ties to even: 30.5 -> 30 < 31),
/// regroups unsorted octaves and accepts `octave >= nlevels`; purecv keeps every keypoint
/// and rejects `octave >= nlevels`.
#[test]
#[ignore = "diverges from OpenCV 4.10.0, see #195"]
fn orb_compute_provided_keypoints_filter_and_regroup_like_opencv() {
    let textured = textured_u8(240, 320, 5);
    let orb = Orb::default();
    let seven = [
        kp(5.0, 5.0, 31.0, 0.0, 0),
        kp(30.5, 100.0, 31.0, 0.0, 0),
        kp(31.0, 100.0, 31.0, 0.0, 0),
        kp(100.0, 100.0, 31.0, 0.0, 0),
        kp(160.0, 120.0, 31.0, 0.0, 3),
        kp(288.5, 100.0, 31.0, 0.0, 0),
        kp(289.0, 100.0, 31.0, 0.0, 0),
    ];
    let desc = orb.compute(&textured, &seven).unwrap();
    assert_eq!(
        desc.rows,
        ORB_SEVEN_KEPT.len(),
        "rows after the border filter"
    );
    let three = [
        kp(100.0, 100.0, 31.0, 0.0, 3),
        kp(120.0, 120.0, 31.0, 0.0, 0),
        kp(140.0, 140.0, 31.0, 0.0, 9),
    ];
    let desc = orb
        .compute(&textured, &three)
        .unwrap_or_else(|e| panic!("octaves [3, 0, 9]: {e}"));
    assert_eq!(desc.rows, ORB_THREE_REGROUPED_OCTAVES.len());
    let desc = orb
        .compute(&textured, &[kp(100.0, 100.0, 31.0, 0.0, 12)])
        .unwrap_or_else(|e| panic!("octave 12: {e}"));
    assert_eq!(desc.rows, 1);
}

/// `WTA_K` 3 and 4 (2-bit codes, `NORM_HAMMING2`): OpenCV computes them, purecv rejects.
#[test]
#[ignore = "diverges from OpenCV 4.10.0, see #198"]
fn orb_wta_k_3_and_4_descriptors_match_opencv() {
    let img = orb_ref_image();
    for (wta_k, digest) in [(3, ORB_WTA3_DESC_FNV), (4, ORB_WTA4_DESC_FNV)] {
        let desc = Orb::new(500, 1.2, 8, 31, 0, wta_k, ScoreType::Harris, 31, 20)
            .compute(&img, &fixture_keypoints())
            .unwrap_or_else(|e| panic!("wta_k {wta_k}: {e}"));
        assert_eq!((desc.rows, desc.cols), (40, 32));
        assert_eq!(fnv1a32(&desc.data), digest, "wta_k {wta_k}");
    }
}

/// `patchSize != 31`: OpenCV draws a 512-point pattern from `cv::RNG(0x34985739)`; purecv
/// rejects descriptors for any other patch size.
#[test]
#[ignore = "diverges from OpenCV 4.10.0, see #199"]
fn orb_patch_size_21_descriptors_match_opencv() {
    let orb = Orb::new(500, 1.2, 8, 21, 0, 2, ScoreType::Harris, 21, 20);
    let desc = orb
        .compute(&orb_ref_image(), &fixture_keypoints())
        .unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(fnv1a32(&desc.data), ORB_PATCH_21_DESC_FNV);
    let (kps, _) = orb.detect_and_compute(&textured_u8(240, 320, 5)).unwrap();
    assert_eq!(kps.len(), ORB_PATCH_21_COUNT);
}

/// `firstLevel = 2`: OpenCV upscales levels 0 and 1 from the input (492 keypoints,
/// octave-0 size 21.5278); purecv rejects `first_level != 0`.
#[test]
#[ignore = "diverges from OpenCV 4.10.0, see #200"]
fn orb_first_level_2_matches_opencv() {
    let (kps, _) = Orb::new(500, 1.2, 8, 31, 2, 2, ScoreType::Harris, 31, 20)
        .detect_and_compute(&textured_u8(240, 320, 5))
        .unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(kps.len(), ORB_FIRST_LEVEL_2_COUNT);
    assert!(kps
        .iter()
        .filter(|k| k.octave == 0)
        .all(|k| k.size == ORB_FIRST_LEVEL_2_SIZE0));
}

/// Parameter validation that agrees with OpenCV: `patchSize < 2` and `WTA_K` outside
/// {2, 3, 4} are rejected (OpenCV `CV_Assert`), `detect` ignores `WTA_K`, a negative
/// octave is rejected by `compute`, images too small for a level return nothing, and an
/// empty keypoint list gives no descriptor rows. (purecv also rejects `scale_factor <= 1`
/// and `nlevels = 0` up front, where OpenCV fails later with an unrelated assertion.)
#[test]
fn orb_parameter_validation_matches_opencv() {
    let textured = textured_u8(240, 320, 5);
    assert!(Orb::new(500, 1.2, 8, 31, 0, 2, ScoreType::Harris, 1, 20)
        .detect(&textured)
        .is_err());
    let wta5 = Orb::new(500, 1.2, 8, 31, 0, 5, ScoreType::Harris, 31, 20);
    assert!(wta5.detect_and_compute(&textured).is_err());
    // `detect` ignores WTA_K in both libraries: OpenCV detects its default 408 keypoints
    // (`variants/wta_5_detect_only` = `ORB_TEXTURED_PER_OCTAVE`), purecv its default 406 (the
    // octave >= 1 difference of `orb_per_octave_counts_match_opencv_above_octave_0`).
    let wta5_counts = per_octave(&wta5.detect(&textured).unwrap(), 8);
    assert_eq!(
        wta5_counts,
        per_octave(&Orb::default().detect(&textured).unwrap(), 8)
    );
    assert_eq!(
        ORB_WTA5_DETECT_PER_OCTAVE, ORB_TEXTURED_PER_OCTAVE,
        "OpenCV: wta_k ignored"
    );
    assert_eq!(
        ORB_TEXTURED_PER_OCTAVE.iter().sum::<usize>(),
        ORB_TEXTURED_COUNT
    );
    assert!(Orb::default()
        .compute(&textured, &[kp(100.0, 100.0, 31.0, 0.0, -1)])
        .is_err());
    assert!(Orb::default()
        .detect(&textured_u8(8, 8, 1))
        .unwrap()
        .is_empty());
    assert!(Orb::default()
        .detect(&textured_u8(40, 40, 1))
        .unwrap()
        .is_empty());
    assert_eq!(Orb::default().compute(&textured, &[]).unwrap().rows, 0);
    assert!(Orb::new(500, 1.0, 3, 31, 0, 2, ScoreType::Harris, 31, 20)
        .detect(&textured)
        .is_err());
    assert!(Orb::new(500, 1.2, 0, 31, 0, 2, ScoreType::Harris, 31, 20)
        .detect(&textured)
        .is_err());
}

// ---------------------------------------------------------------------------------------
// BFMatcher (features2d_matcher.py, "hamming_u8", "l2")
// ---------------------------------------------------------------------------------------

/// `NORM_HAMMING` on u8 descriptors with constructed ties: `match`, `knnMatch` (k = 2, 3 and
/// more than the train rows), the reverse direction, `crossCheck` and its `knnMatch` form
/// all return OpenCV's matches in OpenCV's order; the error cases agree.
#[test]
fn bfmatcher_hamming_matches_opencv() {
    let (q, t) = matcher_u8_sets();
    let bf = BFMatcher::<u8>::new(NormType::NormHamming, false).unwrap();
    assert_matches(
        &bf.match_descriptors(&q, &t).unwrap(),
        &BF_HAMMING_MATCH,
        "match",
    );
    assert_knn(
        &bf.knn_match(&q, &t, 2).unwrap(),
        &BF_HAMMING_KNN2,
        "knn k=2",
    );
    assert_knn(
        &bf.knn_match(&q, &t, 3).unwrap(),
        &BF_HAMMING_KNN3,
        "knn k=3",
    );
    assert_knn(
        &bf.knn_match(&q, &t, 20).unwrap(),
        &BF_HAMMING_KNN20,
        "knn k=20",
    );
    assert_matches(
        &bf.match_descriptors(&t, &q).unwrap(),
        &BF_HAMMING_MATCH_T2Q,
        "train->query",
    );
    let bfc = BFMatcher::<u8>::new(NormType::NormHamming, true).unwrap();
    assert_matches(
        &bfc.match_descriptors(&q, &t).unwrap(),
        &BF_HAMMING_CROSS_CHECK,
        "cross-check",
    );
    assert_knn(
        &bfc.knn_match(&q, &t, 1).unwrap(),
        &BF_HAMMING_KNN1_CROSS_CHECK,
        "knn k=1 cross-check",
    );
    assert!(bfc.knn_match(&q, &t, 2).is_err(), "cross-check with k > 1");
    assert!(bf
        .match_descriptors(&q, &Matrix::<u8>::new(0, 32, 1))
        .unwrap()
        .is_empty());
    assert!(bf
        .match_descriptors(&Matrix::<u8>::new(0, 32, 1), &t)
        .unwrap()
        .is_empty());
    assert!(
        bf.match_descriptors(&q, &Matrix::<u8>::new(8, 16, 1))
            .is_err(),
        "width mismatch"
    );
    let (q32, t32) = matcher_f32_sets();
    assert!(BFMatcher::<f32>::new(NormType::NormHamming, false)
        .unwrap()
        .match_descriptors(&q32, &t32)
        .is_err());
    for (i, row) in BF_HAMMING_DIST.iter().enumerate() {
        for (j, &d) in row.iter().enumerate() {
            let got = u8::compute_distance(
                &q.data[i * 32..i * 32 + 32],
                &t.data[j * 32..j * 32 + 32],
                NormType::NormHamming,
            )
            .unwrap();
            assert_eq!(got, d as f32, "hamming({i}, {j})");
        }
    }
}

/// `NORM_L2` on u8 (OpenCV: `batchDistL2_8u32f`, `sqrt` of the f32 sum) and on f32.
#[test]
fn bfmatcher_l2_matches_opencv() {
    let (q, t) = matcher_u8_sets();
    let bf = BFMatcher::<u8>::new(NormType::NormL2, false).unwrap();
    assert_matches(
        &bf.match_descriptors(&q, &t).unwrap(),
        &BF_L2_U8_MATCH,
        "u8 match",
    );
    assert_knn(
        &bf.knn_match(&q, &t, 2).unwrap(),
        &BF_L2_U8_KNN2,
        "u8 knn k=2",
    );
    let (q, t) = matcher_f32_sets();
    let bf = BFMatcher::<f32>::new(NormType::NormL2, false).unwrap();
    assert_matches(
        &bf.match_descriptors(&q, &t).unwrap(),
        &BF_L2_F32_MATCH,
        "f32 match",
    );
    assert_knn(
        &bf.knn_match(&q, &t, 2).unwrap(),
        &BF_L2_F32_KNN2,
        "f32 knn k=2",
    );
    assert_knn(
        &bf.knn_match(&q, &t, 3).unwrap(),
        &BF_L2_F32_KNN3,
        "f32 knn k=3",
    );
    assert_matches(
        &bf.match_descriptors(&t, &q).unwrap(),
        &BF_L2_F32_MATCH_T2Q,
        "f32 train->query",
    );
    let bfc = BFMatcher::<f32>::new(NormType::NormL2, true).unwrap();
    assert_matches(
        &bfc.match_descriptors(&q, &t).unwrap(),
        &BF_L2_F32_CROSS_CHECK,
        "f32 cross-check",
    );
}

/// `DescriptorMatcher::knnMatch` asserts `knn > 0`; purecv returns one empty list per query.
#[test]
#[ignore = "diverges from OpenCV 4.10.0, see #251"]
fn bfmatcher_knn_match_k0_is_an_error() {
    let (q, t) = matcher_u8_sets();
    let bf = BFMatcher::<u8>::new(NormType::NormHamming, false).unwrap();
    assert!(bf.knn_match(&q, &t, 0).is_err());
}

// ---------------------------------------------------------------------------------------
// draw_keypoints / draw_matches (features2d_draw.py)
// ---------------------------------------------------------------------------------------

/// The `drawMatches` canvas: `img1` at the left, `img2` at the right, the rest zero. With
/// 3-channel inputs and nothing to draw, purecv's output is OpenCV's pixel for pixel; and
/// `drawKeypoints` with no keypoints returns the 3-channel input unchanged.
#[test]
fn draw_canvas_layout_matches_opencv() {
    let img1 = gray_to_3ch(&noise_u8(16, 20, 51));
    let img2 = gray_to_3ch(&noise_u8(14, 24, 52));
    let mut canvas = Matrix::<u8>::zeros(16, 44, 3);
    for y in 0..16 {
        for x in 0..20 {
            for c in 0..3 {
                canvas.set(y, x, c, *img1.get(y, x, c).unwrap());
            }
        }
    }
    for y in 0..14 {
        for x in 0..24 {
            for c in 0..3 {
                canvas.set(y, x + 20, c, *img2.get(y, x, c).unwrap());
            }
        }
    }
    let blue = Scalar::new(255u8, 0, 0, 255);
    let green = Scalar::new(0u8, 255, 0, 255);
    let out = draw_matches(&img1, &[], &img2, &[], &[], Some(blue), Some(green)).unwrap();
    assert_eq!((out.rows, out.cols, out.channels), (16, 44, 3));
    assert_eq!(out.data, canvas.data);
    let bgr = gray_to_3ch(&noise_u8(20, 24, 31));
    assert_eq!(
        draw_keypoints(&bgr, &[], Scalar::new(0u8, 0, 255, 255))
            .unwrap()
            .data,
        bgr.data
    );
}

/// `drawKeypoints`: a gray input becomes a 3-channel BGR output; the default flags draw an
/// anti-aliased circle of radius 3 (`LINE_AA`, 4 fractional bits) and `DRAW_RICH_KEYPOINTS`
/// one of radius `size / 2` with the orientation line. purecv keeps the input's channel
/// count and always draws a Bresenham circle of radius `round(size / 2)` plus the line.
/// The expected values are the changed pixels with their BGR values.
#[test]
#[ignore = "diverges from OpenCV 4.10.0, see #253"]
fn draw_keypoints_matches_opencv() {
    let gray = noise_u8(20, 24, 31);
    let red = Scalar::new(0u8, 0, 255, 255);
    let out = draw_keypoints(&gray, &[kp(10.4, 8.6, 7.0, -1.0, 0)], red).unwrap();
    assert_eq!(
        (out.rows, out.cols, out.channels),
        (20, 24, 3),
        "gray input gives BGR output"
    );
    assert_eq!(
        changed_pixels(&gray_to_3ch(&gray), &out),
        DK_GRAY_DEFAULT.to_vec()
    );
    let bgr = draw_bgr_image();
    let out = draw_keypoints(&bgr, &[kp(10.0, 8.0, 7.0, -1.0, 0)], red).unwrap();
    assert_eq!(changed_pixels(&bgr, &out), DK_BGR_DEFAULT.to_vec());
}

/// `drawMatches`: anti-aliased circles and lines, `singlePointColor` keypoints, a
/// 3-channel output for gray inputs, mixed gray/BGR inputs accepted, and a match index out
/// of range is a `CV_Assert`. purecv draws Bresenham primitives, keeps 1 channel, rejects
/// mixed channel counts and skips invalid indices.
#[test]
#[ignore = "diverges from OpenCV 4.10.0, see #253"]
fn draw_matches_matches_opencv() {
    let img1 = noise_u8(16, 20, 51);
    let img2 = noise_u8(14, 24, 52);
    let k1 = [kp(5.0, 6.0, 7.0, -1.0, 0), kp(14.0, 3.0, 7.0, -1.0, 0)];
    let k2 = [kp(7.0, 9.0, 7.0, -1.0, 0), kp(20.0, 2.0, 7.0, -1.0, 0)];
    let m = [DMatch::new(0, 0, 0, 1.0)];
    let blue = Scalar::new(255u8, 0, 0, 255);
    let green = Scalar::new(0u8, 255, 0, 255);
    let canvas = draw_matches(
        &gray_to_3ch(&img1),
        &[],
        &gray_to_3ch(&img2),
        &[],
        &[],
        Some(blue),
        Some(green),
    )
    .unwrap();
    let out = draw_matches(&img1, &k1, &img2, &k2, &m, Some(blue), Some(green)).unwrap();
    assert_eq!(
        (out.rows, out.cols, out.channels),
        (16, 44, 3),
        "gray inputs give a BGR output"
    );
    assert_eq!(changed_pixels(&canvas, &out), DM_GRAY_ONE_MATCH.to_vec());
    // OpenCV's NOT_DRAW_SINGLE_POINTS case of the script has no counterpart: purecv has no
    // flags ( as single_point_color is the nearest, not the same), so it is not pinned.
    assert!(draw_matches(
        &img1,
        &[],
        &gray_to_3ch(&img2),
        &[],
        &[],
        Some(blue),
        Some(green)
    )
    .is_ok());
    assert!(draw_matches(
        &img1,
        &k1,
        &img2,
        &k2,
        &[DMatch::new(0, 5, 0, 1.0)],
        Some(blue),
        Some(green)
    )
    .is_err());
}

// ---------------------------------------------------------------------------------------
// KeyPoint helpers (features2d_keypoint.py)
// ---------------------------------------------------------------------------------------

/// `KeyPoint::overlap` and the `convert` helpers reproduce `cv::KeyPoint` (`types.cpp`);
/// `KeyPoint::default()` is `(0, 0, 0, -1, 0, 0, -1)` in both.
#[test]
fn keypoint_overlap_and_convert_match_opencv() {
    let k = |x: f32, y: f32, s: f32| kp(x, y, s, -1.0, 0);
    let cases = [
        ("same", k(10.0, 10.0, 8.0), k(10.0, 10.0, 8.0)),
        ("disjoint", k(0.0, 0.0, 4.0), k(10.0, 0.0, 4.0)),
        ("touching", k(0.0, 0.0, 4.0), k(4.0, 0.0, 4.0)),
        ("enclosed", k(5.0, 5.0, 10.0), k(6.0, 5.0, 2.0)),
        (
            "enclosed_concentric_zero_sizes",
            k(5.0, 5.0, 0.0),
            k(5.0, 5.0, 0.0),
        ),
        ("partial_1", k(0.0, 0.0, 10.0), k(3.0, 0.0, 10.0)),
        ("partial_2", k(0.0, 0.0, 10.0), k(2.0, 1.0, 6.0)),
        ("partial_3", k(1.5, 2.5, 7.0), k(3.25, 1.0, 5.0)),
        (
            "size_zero_vs_positive_overlapping_centre",
            k(5.0, 5.0, 0.0),
            k(5.0, 5.0, 6.0),
        ),
    ];
    for (name, a, b) in &cases {
        let expected = KP_OVERLAP.iter().find(|(n, _)| n == name).unwrap().1;
        let got = KeyPoint::overlap(a, b);
        if expected.is_nan() {
            // OpenCV returns 0/0 = NaN for two zero-size concentric keypoints; purecv's
            // `max_s_2 > 0` guard returns 0 (`intentional`, see docs/parity/4.10.0/features2d.md).
            assert_eq!(got, 0.0, "{name}");
        } else {
            assert!(
                (got - expected).abs() <= 1e-6,
                "{name}: purecv {got} vs OpenCV {expected}"
            );
        }
    }
    let kps = [
        KeyPoint::new(Point2f::new(1.0, 2.0), 3.0, 10.0, 0.5, 1, 7),
        KeyPoint::new(Point2f::new(4.5, 5.5), 6.0, 20.0, 0.25, 2, 8),
        KeyPoint::new(Point2f::new(7.0, 8.0), 9.0, 30.0, 0.75, 3, 9),
    ];
    let pts = KeyPoint::convert_to_points(&kps);
    assert_eq!(
        pts,
        vec![
            Point2f::new(1.0, 2.0),
            Point2f::new(4.5, 5.5),
            Point2f::new(7.0, 8.0)
        ]
    );
    assert_eq!(
        KeyPoint::convert_to_points_masked(&kps, &[2, 0]).unwrap(),
        vec![Point2f::new(7.0, 8.0), Point2f::new(1.0, 2.0)]
    );
    assert!(KeyPoint::convert_to_points_masked(&kps, &[-1]).is_err());
    let back = KeyPoint::convert_from_points(
        &[Point2f::new(1.0, 2.0), Point2f::new(3.5, 4.5)],
        5.0,
        2.0,
        3,
        4,
    );
    assert_eq!(
        back[1],
        KeyPoint::new(Point2f::new(3.5, 4.5), 5.0, -1.0, 2.0, 3, 4)
    );
    assert_eq!(
        KeyPoint::default(),
        KeyPoint::new(Point2f::new(0.0, 0.0), 0.0, -1.0, 0.0, 0, -1)
    );
}

/// `KeyPointsFilter::retainBest` keeps every keypoint whose response is at least the n-th
/// best (`nth_element` + `partition`), so cutoff ties survive and the result can be longer
/// than `n`; purecv's `retain_best` sorts and truncates to exactly `n`. `KeyPointsFilter` is
/// not bound by cv2, so the expected lengths come from reading `keypoint.cpp`.
#[test]
#[ignore = "diverges from OpenCV 4.10.0, see #196"]
fn keypoint_retain_best_keeps_cutoff_ties() {
    let responses = [5.0f32, 3.0, 9.0, 3.0, 7.0, 3.0, 1.0];
    let mut kps: Vec<KeyPoint> = responses
        .iter()
        .enumerate()
        .map(|(i, &r)| KeyPoint::new(Point2f::new(i as f32, 0.0), 7.0, -1.0, r, 0, -1))
        .collect();
    KeyPoint::retain_best(&mut kps, 4);
    // the 4th best response is 3.0 and three keypoints share it: OpenCV keeps all six
    assert_eq!(kps.len(), 6, "cutoff ties kept");
    assert!(kps.iter().all(|k| k.response >= 3.0));
    let mut kps2: Vec<KeyPoint> = responses
        .iter()
        .map(|&r| KeyPoint::new(Point2f::new(0.0, 0.0), 7.0, -1.0, r, 0, -1))
        .collect();
    KeyPoint::retain_best(&mut kps2, 0);
    assert!(kps2.is_empty(), "n = 0 clears in both");
}

// ---------------------------------------------------------------------------------------
// Expected values generated from the scripts' JSON output (one constant per key)
// ---------------------------------------------------------------------------------------

// Generated OpenCV 4.10.0 outputs: literals are copied, not approximations of constants.
#[allow(clippy::approx_constant, clippy::excessive_precision)]
mod ocv {
    #[rustfmt::skip]
    pub const FAST_GRID: [(&str, usize, u32, u32); 72] = [
        ("noise_64x80_s7/9_16/nonmax_on/t0", 466, 0xb0b30ce1, 0x86e36747),
        ("noise_64x80_s7/9_16/nonmax_on/t10", 458, 0xfc71a948, 0x47d88474),
        ("noise_64x80_s7/9_16/nonmax_on/t20", 440, 0xba3a651e, 0x1584952a),
        ("noise_64x80_s7/9_16/nonmax_on/t40", 344, 0x77b698c8, 0x3465a489),
        ("noise_64x80_s7/9_16/nonmax_off/t0", 1718, 0x1db9058d, 0xdd45863d),
        ("noise_64x80_s7/9_16/nonmax_off/t10", 1380, 0xa83756ba, 0xb886c6d6),
        ("noise_64x80_s7/9_16/nonmax_off/t20", 1087, 0xaaf23e90, 0x26f330a5),
        ("noise_64x80_s7/9_16/nonmax_off/t40", 579, 0xac4d905d, 0x8d589f20),
        ("noise_64x80_s7/7_12/nonmax_on/t0", 450, 0x05d81af7, 0x09355e06),
        ("noise_64x80_s7/7_12/nonmax_on/t10", 445, 0x6dad457c, 0xe29d742a),
        ("noise_64x80_s7/7_12/nonmax_on/t20", 431, 0xb23dbe51, 0x104113ce),
        ("noise_64x80_s7/7_12/nonmax_on/t40", 366, 0x0aca2560, 0x33d85127),
        ("noise_64x80_s7/7_12/nonmax_off/t0", 1864, 0x901f74b3, 0xf4d53bab),
        ("noise_64x80_s7/7_12/nonmax_off/t10", 1530, 0x2ed75e32, 0x3ea0a792),
        ("noise_64x80_s7/7_12/nonmax_off/t20", 1232, 0x7bee0640, 0xe79fb64c),
        ("noise_64x80_s7/7_12/nonmax_off/t40", 728, 0x9605829d, 0xda71a4ad),
        ("noise_64x80_s7/5_8/nonmax_on/t0", 522, 0xd5473071, 0x7a8d3eea),
        ("noise_64x80_s7/5_8/nonmax_on/t10", 434, 0x2cf1f163, 0x47b11b31),
        ("noise_64x80_s7/5_8/nonmax_on/t20", 328, 0x57aee020, 0x8508bd2d),
        ("noise_64x80_s7/5_8/nonmax_on/t40", 165, 0xae4c6bac, 0x056f29c4),
        ("noise_64x80_s7/5_8/nonmax_off/t0", 921, 0x20f0cfe4, 0x472009c5),
        ("noise_64x80_s7/5_8/nonmax_off/t10", 627, 0x5d1fcdca, 0xfc2abbeb),
        ("noise_64x80_s7/5_8/nonmax_off/t20", 430, 0x4c132fa9, 0x6ea0de8d),
        ("noise_64x80_s7/5_8/nonmax_off/t40", 189, 0x543c38b6, 0xdef8bbbb),
        ("textured_48x64_s3/9_16/nonmax_on/t0", 179, 0xaacdbb8b, 0x84331639),
        ("textured_48x64_s3/9_16/nonmax_on/t10", 76, 0x4d0715d4, 0xa43840cd),
        ("textured_48x64_s3/9_16/nonmax_on/t20", 68, 0x6c53ca72, 0xa71a2c13),
        ("textured_48x64_s3/9_16/nonmax_on/t40", 55, 0x041a343b, 0x6e844b0c),
        ("textured_48x64_s3/9_16/nonmax_off/t0", 1162, 0xac96254d, 0x62dadadd),
        ("textured_48x64_s3/9_16/nonmax_off/t10", 394, 0x144c0c98, 0xef3be450),
        ("textured_48x64_s3/9_16/nonmax_off/t20", 347, 0xd04cd663, 0xa39e00ae),
        ("textured_48x64_s3/9_16/nonmax_off/t40", 256, 0xbb42de4e, 0x557a9c7e),
        ("textured_48x64_s3/7_12/nonmax_on/t0", 191, 0x5daa2cdd, 0x172223bd),
        ("textured_48x64_s3/7_12/nonmax_on/t10", 63, 0x0854932d, 0x2efd3efe),
        ("textured_48x64_s3/7_12/nonmax_on/t20", 59, 0xc5dc21b0, 0x840cd31e),
        ("textured_48x64_s3/7_12/nonmax_on/t40", 46, 0x1790baee, 0x42d3adb7),
        ("textured_48x64_s3/7_12/nonmax_off/t0", 1008, 0xff872465, 0xe519f325),
        ("textured_48x64_s3/7_12/nonmax_off/t10", 143, 0x1303845e, 0x53a8f853),
        ("textured_48x64_s3/7_12/nonmax_off/t20", 124, 0x79a6c470, 0x9f5a1390),
        ("textured_48x64_s3/7_12/nonmax_off/t40", 89, 0x0141bb4b, 0x4bcddfbe),
        ("textured_48x64_s3/5_8/nonmax_on/t0", 224, 0x8cbcf3af, 0xc562c5ce),
        ("textured_48x64_s3/5_8/nonmax_on/t10", 0, 0x811c9dc5, 0x811c9dc5),
        ("textured_48x64_s3/5_8/nonmax_on/t20", 0, 0x811c9dc5, 0x811c9dc5),
        ("textured_48x64_s3/5_8/nonmax_on/t40", 0, 0x811c9dc5, 0x811c9dc5),
        ("textured_48x64_s3/5_8/nonmax_off/t0", 360, 0x42582ee9, 0x49728459),
        ("textured_48x64_s3/5_8/nonmax_off/t10", 0, 0x811c9dc5, 0x811c9dc5),
        ("textured_48x64_s3/5_8/nonmax_off/t20", 0, 0x811c9dc5, 0x811c9dc5),
        ("textured_48x64_s3/5_8/nonmax_off/t40", 0, 0x811c9dc5, 0x811c9dc5),
        ("squares_40x48/9_16/nonmax_on/t0", 0, 0x811c9dc5, 0x811c9dc5),
        ("squares_40x48/9_16/nonmax_on/t10", 0, 0x811c9dc5, 0x811c9dc5),
        ("squares_40x48/9_16/nonmax_on/t20", 0, 0x811c9dc5, 0x811c9dc5),
        ("squares_40x48/9_16/nonmax_on/t40", 0, 0x811c9dc5, 0x811c9dc5),
        ("squares_40x48/9_16/nonmax_off/t0", 48, 0xca2e2a55, 0x08091205),
        ("squares_40x48/9_16/nonmax_off/t10", 48, 0xca2e2a55, 0x08091205),
        ("squares_40x48/9_16/nonmax_off/t20", 48, 0xca2e2a55, 0x08091205),
        ("squares_40x48/9_16/nonmax_off/t40", 48, 0xca2e2a55, 0x08091205),
        ("squares_40x48/7_12/nonmax_on/t0", 6, 0x912daff0, 0xaf7ff270),
        ("squares_40x48/7_12/nonmax_on/t10", 6, 0x912daff0, 0xaf7ff270),
        ("squares_40x48/7_12/nonmax_on/t20", 6, 0x912daff0, 0xaf7ff270),
        ("squares_40x48/7_12/nonmax_on/t40", 6, 0x912daff0, 0xaf7ff270),
        ("squares_40x48/7_12/nonmax_off/t0", 12, 0x288f4cc9, 0xe9c35b39),
        ("squares_40x48/7_12/nonmax_off/t10", 12, 0x288f4cc9, 0xe9c35b39),
        ("squares_40x48/7_12/nonmax_off/t20", 12, 0x288f4cc9, 0xe9c35b39),
        ("squares_40x48/7_12/nonmax_off/t40", 12, 0x288f4cc9, 0xe9c35b39),
        ("squares_40x48/5_8/nonmax_on/t0", 0, 0x811c9dc5, 0x811c9dc5),
        ("squares_40x48/5_8/nonmax_on/t10", 0, 0x811c9dc5, 0x811c9dc5),
        ("squares_40x48/5_8/nonmax_on/t20", 0, 0x811c9dc5, 0x811c9dc5),
        ("squares_40x48/5_8/nonmax_on/t40", 0, 0x811c9dc5, 0x811c9dc5),
        ("squares_40x48/5_8/nonmax_off/t0", 0, 0x811c9dc5, 0x811c9dc5),
        ("squares_40x48/5_8/nonmax_off/t10", 0, 0x811c9dc5, 0x811c9dc5),
        ("squares_40x48/5_8/nonmax_off/t20", 0, 0x811c9dc5, 0x811c9dc5),
        ("squares_40x48/5_8/nonmax_off/t40", 0, 0x811c9dc5, 0x811c9dc5),
    ];
    pub const FAST_TINY_COUNTS: [(usize, usize, usize, usize); 4] =
        [(5, 0, 0, 0), (6, 0, 0, 0), (7, 0, 0, 0), (8, 1, 0, 0)];
    #[rustfmt::skip]
    pub const FAST_SQUARES_7_12_NONMAX_ON: [(f32, f32, f32); 6] = [(4.0, 4.0, 169.0), (11.0, 4.0, 169.0), (4.0, 11.0, 169.0), (20.0, 16.0, 189.0), (33.0, 16.0, 189.0), (20.0, 27.0, 189.0)];
    pub const FAST_ARC_7_OF_12_STARTS_FOUND: [usize; 4] = [0, 9, 10, 11];
    pub const FAST_ARC_5_OF_8_STARTS_FOUND: [usize; 0] = [];
    pub const ORB_TEXTURED_COUNT: usize = 408;
    pub const ORB_TEXTURED_PER_OCTAVE: [usize; 8] = [109, 89, 75, 56, 34, 25, 15, 5];
    /// `variants/wta_5_detect_only`.
    pub const ORB_WTA5_DETECT_PER_OCTAVE: [usize; 8] = [109, 89, 75, 56, 34, 25, 15, 5];
    #[rustfmt::skip]
    pub const ORB_TEXTURED_OCTAVE0_XY: [(f32, f32); 109] = [(83.0, 72.0), (84.0, 76.0), (90.0, 76.0), (91.0, 73.0), (96.0, 168.0), (97.0, 122.0), (99.0, 122.0), (99.0, 166.0), (100.0, 168.0), (101.0, 163.0), (118.0, 159.0), (121.0, 127.0), (125.0, 131.0), (126.0, 147.0), (127.0, 112.0), (127.0, 129.0), (129.0, 111.0), (129.0, 113.0), (130.0, 128.0), (133.0, 129.0), (134.0, 147.0), (135.0, 131.0), (138.0, 152.0), (141.0, 81.0), (141.0, 100.0), (152.0, 133.0), (152.0, 135.0), (152.0, 158.0), (153.0, 82.0), (154.0, 100.0), (154.0, 133.0), (154.0, 159.0), (157.0, 133.0), (157.0, 159.0), (158.0, 84.0), (159.0, 78.0), (159.0, 158.0), (160.0, 44.0), (160.0, 58.0), (160.0, 84.0), (165.0, 60.0), (165.0, 62.0), (167.0, 59.0), (167.0, 62.0), (170.0, 157.0), (172.0, 62.0), (172.0, 158.0), (173.0, 147.0), (174.0, 59.0), (175.0, 127.0), (175.0, 148.0), (175.0, 156.0), (175.0, 158.0), (180.0, 107.0), (182.0, 44.0), (182.0, 78.0), (184.0, 78.0), (184.0, 82.0), (184.0, 84.0), (191.0, 32.0), (194.0, 32.0), (196.0, 197.0), (197.0, 89.0), (197.0, 93.0), (197.0, 199.0), (199.0, 94.0), (204.0, 199.0), (206.0, 197.0), (208.0, 178.0), (211.0, 47.0), (213.0, 47.0), (216.0, 88.0), (216.0, 178.0), (216.0, 180.0), (216.0, 194.0), (217.0, 92.0), (217.0, 94.0), (222.0, 40.0), (238.0, 149.0), (238.0, 167.0), (238.0, 186.0), (239.0, 183.0), (240.0, 147.0), (241.0, 184.0), (241.0, 187.0), (251.0, 114.0), (253.0, 105.0), (256.0, 147.0), (256.0, 167.0), (258.0, 157.0), (258.0, 160.0), (259.0, 117.0), (260.0, 129.0), (261.0, 68.0), (261.0, 115.0), (261.0, 159.0), (262.0, 56.0), (268.0, 105.0), (269.0, 164.0), (270.0, 107.0), (271.0, 147.0), (271.0, 165.0), (272.0, 107.0), (274.0, 164.0), (281.0, 128.0), (282.0, 124.0), (283.0, 68.0), (283.0, 122.0), (284.0, 56.0)];
    pub const ORB_NOISE_PER_OCTAVE: [usize; 8] = [109, 90, 75, 63, 52, 44, 24, 1];
    pub const ORB_REF_PER_OCTAVE: [usize; 6] = [109, 90, 75, 63, 46, 9];
    pub const ORB_SIZE_PER_OCTAVE: [f32; 8] = [
        31.0,
        37.20000076293945,
        44.6400032043457,
        53.5680046081543,
        64.28160858154297,
        77.137939453125,
        92.5655288696289,
        111.0786361694336,
    ];
    #[rustfmt::skip]
    pub const ORB_TEXTURED_OCTAVE0: [(f32, f32, f32, f32); 109] = [
        (191.0, 32.0, 115.31130981445312, 3.3163414627779275e-05),
        (194.0, 32.0, 166.3224334716797, 6.701023812638596e-05),
        (157.0, 133.0, 281.9743347167969, 0.0002955447125714272),
        (222.0, 40.0, 358.8592834472656, 5.951557250227779e-05),
        (154.0, 133.0, 259.3366394042969, 0.00030423374846577644),
        (160.0, 44.0, 39.050018310546875, 8.261389302788302e-05),
        (182.0, 44.0, 130.1343231201172, 0.00011551174975465983),
        (211.0, 47.0, 126.017578125, 2.273660538776312e-05),
        (213.0, 47.0, 115.23814392089844, 3.894865949405357e-05),
        (262.0, 56.0, 224.112548828125, 7.251940405694768e-05),
        (284.0, 56.0, 318.8359680175781, 5.36223451490514e-05),
        (152.0, 133.0, 245.14723205566406, 0.00019325860193930566),
        (160.0, 58.0, 303.75201416015625, 7.447871030308306e-05),
        (167.0, 59.0, 287.51324462890625, 0.0003342913987580687),
        (174.0, 59.0, 264.9930114746094, 0.00027030828641727567),
        (134.0, 147.0, 42.8157844543457, 0.0017446561250835657),
        (135.0, 131.0, 331.3314208984375, 0.001572061562910676),
        (165.0, 60.0, 289.6983642578125, 0.0003066240169573575),
        (165.0, 62.0, 285.4046325683594, 0.00020907464204356074),
        (167.0, 62.0, 282.27691650390625, 0.0002647461951710284),
        (172.0, 62.0, 272.867919921875, 0.0002511436177883297),
        (261.0, 68.0, 143.58755493164062, 8.010127930901945e-05),
        (283.0, 68.0, 38.927494049072266, 0.00010363152250647545),
        (83.0, 72.0, 32.95189666748047, 8.406962297158316e-05),
        (91.0, 73.0, 159.2430419921875, 0.00014922853733878583),
        (84.0, 76.0, 335.58184814453125, 0.00010820291936397552),
        (90.0, 76.0, 210.40872192382812, 0.0001119954395107925),
        (159.0, 78.0, 247.4154052734375, 0.00011953033390454948),
        (182.0, 78.0, 331.83685302734375, 0.0001538081851322204),
        (184.0, 78.0, 335.4510498046875, 0.00010013040446210653),
        (125.0, 131.0, 206.13418579101562, 0.0005089959013275802),
        (153.0, 82.0, 267.4928894042969, 2.1328081857063808e-05),
        (184.0, 82.0, 8.327799797058105, 0.00014034821651875973),
        (158.0, 84.0, 317.839599609375, 0.0001327052596025169),
        (160.0, 84.0, 41.940528869628906, 0.00015822993009351194),
        (184.0, 84.0, 30.308563232421875, 8.89085276867263e-05),
        (260.0, 129.0, 318.1355895996094, 5.91795178479515e-05),
        (216.0, 88.0, 343.93890380859375, 0.0015852050855755806),
        (197.0, 89.0, 183.34669494628906, 0.0009997475426644087),
        (217.0, 92.0, 13.271148681640625, 0.001406578696332872),
        (197.0, 93.0, 157.11785888671875, 0.0013686957536265254),
        (199.0, 94.0, 146.7807159423828, 0.0016579794464632869),
        (217.0, 94.0, 26.474653244018555, 0.000915447308216244),
        (133.0, 129.0, 325.68756103515625, 0.0018255258910357952),
        (127.0, 129.0, 282.8757629394531, 0.0006708620931021869),
        (253.0, 105.0, 212.88607788085938, 0.0010025344090536237),
        (268.0, 105.0, 5.529366493225098, 0.002007523085922003),
        (180.0, 107.0, 289.1625671386719, 0.0007378307054750621),
        (270.0, 107.0, 23.8599910736084, 0.0022258844692260027),
        (272.0, 107.0, 27.523130416870117, 0.0013601246755570173),
        (129.0, 111.0, 288.63079833984375, 0.0005538984551094472),
        (127.0, 112.0, 283.2398376464844, 0.0009078755392692983),
        (129.0, 113.0, 294.7079162597656, 0.0008245777571573853),
        (251.0, 114.0, 127.25538635253906, 0.0006121947662904859),
        (261.0, 115.0, 59.568580627441406, 0.0002506933524273336),
        (259.0, 117.0, 72.05838012695312, 0.0005911788321100175),
        (97.0, 122.0, 195.67807006835938, 0.0011107980972155929),
        (99.0, 122.0, 199.157470703125, 0.0016742863226681948),
        (283.0, 122.0, 216.5074462890625, 0.0031899104360491037),
        (282.0, 124.0, 213.46551513671875, 0.0033042775467038155),
        (121.0, 127.0, 199.21852111816406, 0.0019685430452227592),
        (175.0, 127.0, 109.71572875976562, 0.0007464750669896603),
        (130.0, 128.0, 314.28851318359375, 0.0007063509547151625),
        (152.0, 135.0, 241.37942504882812, 0.000301976571790874),
        (126.0, 147.0, 145.04478454589844, 0.001592707121744752),
        (281.0, 128.0, 201.99693298339844, 0.0001450595009373501),
        (173.0, 147.0, 304.2623291015625, 0.000819164386484772),
        (240.0, 147.0, 230.70773315429688, 0.00030865229200571775),
        (256.0, 147.0, 319.28033447265625, 0.0003264021943323314),
        (271.0, 147.0, 263.7000732421875, 5.2939794841222465e-05),
        (175.0, 148.0, 296.61083984375, 0.00048140151193365455),
        (238.0, 149.0, 218.5690460205078, 0.0003382678551133722),
        (175.0, 156.0, 54.64239501953125, 0.0006359175895340741),
        (170.0, 157.0, 50.005306243896484, 0.0005365042015910149),
        (258.0, 157.0, 358.90771484375, 0.0036884681321680546),
        (152.0, 158.0, 108.01380920410156, 0.00025805438053794205),
        (159.0, 158.0, 99.55940246582031, 0.0002464604040142149),
        (172.0, 158.0, 62.25923156738281, 0.0009408958139829338),
        (175.0, 158.0, 65.37018585205078, 0.00041751490789465606),
        (118.0, 159.0, 209.28125, 7.730619836365804e-05),
        (154.0, 159.0, 99.5867919921875, 0.0003103282651863992),
        (157.0, 159.0, 95.35883331298828, 0.0003021718002855778),
        (261.0, 159.0, 17.507604598999023, 0.004930982831865549),
        (258.0, 160.0, 18.72443389892578, 0.0037427330389618874),
        (101.0, 163.0, 266.7221984863281, 0.00014454248594120145),
        (269.0, 164.0, 83.28983306884766, 5.2936040447093546e-05),
        (274.0, 164.0, 97.75703430175781, 5.759810665040277e-05),
        (271.0, 165.0, 121.0145263671875, 6.238343485165387e-05),
        (99.0, 166.0, 259.07806396484375, 4.0178205381380394e-05),
        (238.0, 167.0, 135.72862243652344, 0.0002895719662774354),
        (256.0, 167.0, 39.24697494506836, 0.0003907044301740825),
        (96.0, 168.0, 255.48240661621094, 2.2151831217342988e-05),
        (100.0, 168.0, 247.26193237304688, 3.502897379803471e-05),
        (208.0, 178.0, 62.28816604614258, 0.0005733837024308741),
        (216.0, 178.0, 118.41506958007812, 0.0005831319140270352),
        (216.0, 180.0, 123.54229736328125, 0.0009086665813811123),
        (239.0, 183.0, 51.3467903137207, 0.0020030641462653875),
        (241.0, 184.0, 38.23756790161133, 0.0026960528921335936),
        (238.0, 186.0, 359.96002197265625, 0.0016055231681093574),
        (241.0, 187.0, 6.464788913726807, 0.0013064831728115678),
        (216.0, 194.0, 263.7945251464844, 0.0005697938031516969),
        (196.0, 197.0, 215.15023803710938, 0.0012937755091115832),
        (206.0, 197.0, 318.5389709472656, 0.0021709054708480835),
        (197.0, 199.0, 142.84310913085938, 0.0007484570960514247),
        (204.0, 199.0, 326.799560546875, 0.0009245582041330636),
        (141.0, 81.0, 230.92236328125, 2.0185354514978826e-05),
        (141.0, 100.0, 130.55172729492188, 1.726744267216418e-05),
        (154.0, 100.0, 44.82120132446289, 1.7167260011774488e-05),
        (138.0, 152.0, 34.48834228515625, 1.5270779840648174e-05),
    ];
    #[rustfmt::skip]
    pub const ORB_TEXTURED_OCTAVE0_DESC: [[u8; 32]; 109] = [
        [0xa2, 0xca, 0x0f, 0xbc, 0x2c, 0xd5, 0x90, 0x67, 0x9c, 0x5c, 0x50, 0x07, 0x22, 0x8c, 0x36, 0x41, 0xcc, 0x21, 0x0e, 0x45, 0x6f, 0x69, 0x02, 0x9d, 0x32, 0xf3, 0xda, 0x0b, 0x1d, 0x08, 0x38, 0x9e],
        [0x8a, 0x0c, 0x0a, 0x93, 0xf7, 0x32, 0xbc, 0x07, 0x3c, 0x1b, 0x5e, 0x72, 0xac, 0x3b, 0x12, 0xcd, 0x67, 0x0f, 0x93, 0x89, 0x31, 0xdc, 0x01, 0xea, 0x26, 0x77, 0xde, 0x87, 0x86, 0x54, 0xa9, 0x81],
        [0x33, 0xd9, 0x72, 0x88, 0x57, 0x81, 0xcf, 0xff, 0x55, 0x48, 0x50, 0x42, 0x21, 0x1a, 0x30, 0x64, 0x68, 0xaa, 0x06, 0xc8, 0xe7, 0xaa, 0x06, 0xfd, 0xa0, 0x68, 0x9e, 0xc8, 0x3f, 0x9c, 0xb0, 0x8c],
        [0x02, 0xba, 0xb2, 0xb8, 0x71, 0xc5, 0xb1, 0x47, 0x2e, 0x0d, 0x78, 0x21, 0xac, 0x97, 0x62, 0x40, 0xec, 0x3b, 0x83, 0x90, 0x02, 0x1a, 0x92, 0xf3, 0xb5, 0x34, 0x1e, 0xab, 0x90, 0x04, 0xa5, 0xfb],
        [0x12, 0x85, 0x4e, 0xcf, 0x14, 0x91, 0xed, 0x67, 0x05, 0x01, 0x5e, 0x82, 0x29, 0x4e, 0xb0, 0x64, 0x68, 0xa2, 0x02, 0xcc, 0x52, 0xb8, 0x26, 0xdd, 0x70, 0x30, 0x8a, 0xc1, 0x3d, 0x8a, 0xb9, 0x1c],
        [0xec, 0x51, 0x7e, 0x65, 0x64, 0x4e, 0x77, 0x10, 0x60, 0x92, 0x67, 0x08, 0x57, 0x53, 0x09, 0x3b, 0x30, 0x54, 0x71, 0x6a, 0x49, 0xb4, 0xb0, 0x08, 0xc7, 0xe9, 0x60, 0x04, 0x40, 0xb3, 0x42, 0xa2],
        [0xd1, 0x77, 0x19, 0x68, 0xb0, 0x2c, 0x5f, 0x1d, 0x29, 0x08, 0x38, 0x49, 0x97, 0x13, 0x08, 0x50, 0xb6, 0x15, 0x5d, 0x68, 0xa2, 0x01, 0x91, 0x0f, 0xf3, 0xf9, 0x25, 0x04, 0x40, 0x90, 0xc3, 0x63],
        [0x1c, 0x49, 0x73, 0xec, 0x36, 0x82, 0x6b, 0xf0, 0xa5, 0x00, 0xf6, 0xc2, 0x7b, 0xc3, 0xa1, 0x54, 0x7e, 0xe8, 0x0f, 0xea, 0x48, 0x8b, 0x20, 0xfd, 0xf2, 0xed, 0xa2, 0x88, 0x37, 0xb3, 0x60, 0x21],
        [0x9d, 0x49, 0x73, 0xce, 0x36, 0x92, 0x6f, 0xf0, 0xe5, 0x40, 0xf6, 0x42, 0x3b, 0xc7, 0xb5, 0xf4, 0x7e, 0xe8, 0x16, 0xca, 0xc9, 0xbb, 0x24, 0xed, 0x73, 0xe9, 0xa6, 0x80, 0x77, 0xb2, 0x60, 0xa9],
        [0x23, 0xcb, 0x02, 0x8f, 0x74, 0x87, 0xce, 0xe5, 0x07, 0x01, 0x58, 0x42, 0x20, 0xc8, 0xb1, 0x64, 0x5c, 0x2a, 0x02, 0x5d, 0xc2, 0xb8, 0x0a, 0xbd, 0x24, 0x1c, 0x86, 0x08, 0x1c, 0x9a, 0xb8, 0x8c],
        [0x03, 0x8d, 0x0a, 0x9c, 0xb4, 0x83, 0x0e, 0x67, 0x94, 0x40, 0x5a, 0x02, 0x34, 0xf2, 0x20, 0x60, 0xf8, 0x2b, 0x12, 0x05, 0x15, 0x88, 0x20, 0xbf, 0xa8, 0x36, 0x8e, 0x80, 0x1c, 0x48, 0xb1, 0x0c],
        [0x01, 0x97, 0x5b, 0x8c, 0x96, 0xc0, 0x34, 0x67, 0x84, 0x45, 0xde, 0x22, 0x31, 0x26, 0x22, 0x60, 0x64, 0xa2, 0x02, 0x81, 0x54, 0x9b, 0x24, 0x1f, 0x32, 0x25, 0xb2, 0x40, 0x15, 0x0a, 0xb4, 0x49],
        [0x09, 0xf8, 0xbb, 0x67, 0x34, 0x4d, 0x73, 0x94, 0x78, 0x48, 0x77, 0x68, 0xbe, 0x57, 0x88, 0x30, 0xa6, 0x5a, 0x4c, 0x48, 0xc1, 0xf8, 0xab, 0x60, 0xef, 0xd9, 0x24, 0xc0, 0x42, 0x92, 0xc7, 0x31],
        [0x54, 0x0c, 0x3f, 0xbb, 0xb4, 0xe2, 0x79, 0x12, 0xbc, 0x14, 0x5b, 0x65, 0xdf, 0x07, 0xa1, 0x59, 0xe7, 0xb1, 0x2f, 0x98, 0x22, 0x94, 0x21, 0x54, 0xc4, 0xeb, 0xba, 0x80, 0xb4, 0xc3, 0x25, 0x71],
        [0xa0, 0xa5, 0x28, 0xea, 0xdc, 0xcd, 0xc9, 0xcc, 0x4a, 0x75, 0xf4, 0x19, 0x92, 0x8f, 0x7e, 0x10, 0xc8, 0x8b, 0x13, 0xa0, 0xba, 0x23, 0x78, 0xd4, 0xf7, 0x99, 0x82, 0x7d, 0xd0, 0x0b, 0xf6, 0xd2],
        [0x02, 0x9c, 0x1a, 0xb9, 0x74, 0xe5, 0xb0, 0x67, 0x1e, 0x0f, 0x5a, 0x33, 0xa4, 0x9f, 0x22, 0x40, 0xc1, 0x3b, 0x93, 0x01, 0x16, 0x98, 0x52, 0xdf, 0x45, 0x36, 0x9a, 0xc3, 0x90, 0x08, 0xa5, 0xdb],
        [0xf0, 0x48, 0x02, 0x9c, 0x1c, 0x83, 0x92, 0x60, 0x94, 0x04, 0xd0, 0x83, 0xe2, 0xcc, 0x33, 0x41, 0xf8, 0xa8, 0x0a, 0x63, 0x6c, 0xa7, 0x32, 0xdd, 0xd3, 0xa0, 0x83, 0x19, 0x15, 0x29, 0x34, 0x42],
        [0x4c, 0x10, 0x7f, 0xfa, 0xa2, 0xe4, 0x7f, 0x11, 0xac, 0x4b, 0x7f, 0x20, 0x9d, 0x57, 0xa0, 0x58, 0xa7, 0x11, 0x3d, 0x1a, 0x10, 0x00, 0x29, 0x60, 0xcf, 0xef, 0x7a, 0xc0, 0xb4, 0xc3, 0xa5, 0x73],
        [0x06, 0x52, 0xff, 0xe2, 0xae, 0xc4, 0x3f, 0xd0, 0x8e, 0xdf, 0x15, 0x32, 0xdd, 0xf7, 0xe8, 0x4a, 0xe7, 0x71, 0x1f, 0xdf, 0xd0, 0x01, 0x2a, 0xe2, 0xe9, 0x9f, 0xbf, 0x80, 0xb4, 0xd3, 0xa5, 0x73],
        [0x06, 0x52, 0xbf, 0xf2, 0xa6, 0xa4, 0x7f, 0xd4, 0x8c, 0xde, 0x15, 0xb3, 0xdd, 0xbf, 0xb3, 0x4a, 0xe7, 0x7b, 0x1f, 0xcb, 0x80, 0x49, 0x2a, 0xe0, 0xf1, 0xcb, 0xfa, 0xa0, 0x14, 0xd3, 0xa5, 0x73],
        [0xa6, 0xc5, 0x78, 0xfa, 0xec, 0xf4, 0xff, 0x77, 0xd9, 0x56, 0xbd, 0x12, 0x59, 0x0d, 0x3e, 0x21, 0xef, 0xd3, 0x17, 0xe7, 0xe6, 0x26, 0x7a, 0xf7, 0x77, 0xdb, 0x8a, 0xa9, 0x55, 0xdb, 0xf1, 0xc0],
        [0x00, 0x0f, 0x02, 0xcc, 0x94, 0xb7, 0xe3, 0x77, 0x1c, 0x44, 0x5e, 0x06, 0x20, 0xa8, 0xb2, 0x64, 0x78, 0x2a, 0x02, 0x0c, 0xb1, 0xb8, 0x34, 0xbf, 0x20, 0x26, 0x82, 0x40, 0x1e, 0xc8, 0xf0, 0xbc],
        [0x02, 0xc9, 0x42, 0xae, 0x15, 0x81, 0xde, 0xe4, 0xc7, 0x08, 0x58, 0x03, 0x2c, 0xda, 0x33, 0xc0, 0xda, 0x6b, 0x97, 0xc1, 0x46, 0xaa, 0x0a, 0xfd, 0x22, 0x65, 0xaa, 0x89, 0x3c, 0x1e, 0x38, 0x8b],
        [0x28, 0x48, 0x11, 0x70, 0x60, 0xcf, 0x00, 0x04, 0xe2, 0x20, 0xa8, 0x28, 0x16, 0x13, 0x00, 0x10, 0x00, 0xb0, 0x68, 0x10, 0x40, 0x2d, 0x80, 0x14, 0xd9, 0x80, 0x00, 0x70, 0x40, 0x82, 0x46, 0x22],
        [0x4d, 0x20, 0x11, 0x65, 0x01, 0x49, 0x62, 0x01, 0x64, 0x08, 0x23, 0x08, 0xff, 0x73, 0x08, 0xf0, 0x30, 0x10, 0x68, 0x2a, 0x00, 0x24, 0x09, 0x01, 0x85, 0xcf, 0x61, 0x02, 0x42, 0x20, 0x43, 0x22],
        [0x50, 0xe0, 0x58, 0x50, 0x08, 0x8c, 0x41, 0x04, 0xe0, 0x01, 0xa0, 0x08, 0x17, 0x93, 0x00, 0x70, 0x00, 0xb4, 0x4c, 0x08, 0x08, 0x4b, 0xd2, 0x05, 0xc9, 0xa8, 0x10, 0x58, 0x60, 0x80, 0x46, 0x28],
        [0x68, 0x38, 0x94, 0x64, 0x03, 0x0d, 0x07, 0xa2, 0x60, 0x9a, 0x62, 0x08, 0x06, 0x33, 0x40, 0x12, 0x01, 0x54, 0x60, 0x40, 0x00, 0x78, 0xc3, 0x20, 0x0d, 0x80, 0x44, 0x16, 0x60, 0x21, 0x46, 0x22],
        [0xf3, 0x00, 0x4e, 0xac, 0x0d, 0x02, 0x49, 0x48, 0xb5, 0x82, 0xc3, 0x96, 0x53, 0x82, 0x37, 0x5d, 0x7a, 0xa5, 0x3a, 0xa7, 0x68, 0x87, 0x10, 0xcc, 0x91, 0xe8, 0xca, 0x1c, 0x3d, 0x29, 0x78, 0x04],
        [0x0a, 0x85, 0x02, 0x89, 0xd5, 0x84, 0xac, 0x67, 0x2c, 0x01, 0x4a, 0x03, 0xac, 0x22, 0x20, 0xc0, 0xe0, 0x33, 0x93, 0x81, 0x56, 0x98, 0x10, 0xdd, 0x2d, 0x24, 0x8a, 0xc3, 0x14, 0x06, 0xb9, 0x8a],
        [0x00, 0x95, 0x12, 0x8d, 0xd7, 0x86, 0xa4, 0x67, 0x0c, 0x01, 0x4a, 0x47, 0xa4, 0x22, 0x20, 0xc4, 0xe0, 0x3b, 0x82, 0x81, 0x56, 0x98, 0x10, 0xdd, 0x25, 0x26, 0x9a, 0x43, 0x18, 0x08, 0xb9, 0x9c],
        [0x02, 0x9a, 0x1e, 0x99, 0x70, 0xa1, 0xb8, 0x47, 0x9e, 0x1f, 0x54, 0xf5, 0xac, 0xc2, 0x22, 0x4c, 0xc7, 0x3b, 0xa3, 0x00, 0x16, 0x98, 0x02, 0xd7, 0x04, 0x36, 0x1a, 0xcb, 0x90, 0x04, 0x2d, 0x9f],
        [0x02, 0xb2, 0x50, 0xb1, 0xe2, 0xa9, 0xb9, 0x07, 0x2f, 0x3d, 0x39, 0x3c, 0xae, 0x1e, 0xc4, 0x8c, 0xa7, 0x1b, 0xc6, 0x11, 0x12, 0x3c, 0x42, 0xd3, 0x8c, 0xdc, 0x3a, 0xee, 0xd4, 0x08, 0x83, 0x89],
        [0x39, 0xcd, 0x4a, 0xa8, 0x16, 0x81, 0xce, 0xe7, 0x14, 0x04, 0x5c, 0xa2, 0x31, 0x82, 0xb8, 0x60, 0x78, 0xaa, 0x02, 0xcc, 0x4a, 0xb8, 0x04, 0xbf, 0xf0, 0x8d, 0xae, 0x40, 0x35, 0x98, 0xf8, 0x88],
        [0x3a, 0xae, 0xea, 0x9a, 0xdf, 0xd1, 0x9b, 0xcf, 0x16, 0x55, 0x4c, 0xa3, 0x28, 0xcc, 0x76, 0x00, 0x4b, 0x6b, 0x83, 0x95, 0x1e, 0x4a, 0xae, 0xf7, 0x28, 0x07, 0x9f, 0xeb, 0xb5, 0x0c, 0x94, 0xde],
        [0x25, 0x24, 0x68, 0x45, 0x74, 0x2b, 0xe3, 0x58, 0x54, 0xa8, 0x7d, 0x0a, 0x53, 0x51, 0xb5, 0x20, 0x58, 0xa8, 0x16, 0x6e, 0x89, 0xb2, 0x6d, 0x0c, 0x46, 0x8d, 0xa1, 0x24, 0x77, 0xbb, 0x52, 0xa0],
        [0x31, 0xc9, 0x42, 0xc8, 0x14, 0x81, 0x4a, 0xfc, 0xc5, 0x40, 0xf0, 0x46, 0x33, 0x81, 0xb0, 0xe4, 0x48, 0xa8, 0x06, 0x4e, 0xcb, 0xab, 0x14, 0xbd, 0xb0, 0x8c, 0xa0, 0x18, 0x3d, 0x9b, 0x78, 0x88],
        [0x40, 0x70, 0x99, 0xe2, 0x70, 0x6c, 0x73, 0x5d, 0x30, 0xa9, 0xa1, 0x48, 0xd7, 0x5f, 0x41, 0x78, 0xb5, 0x95, 0x5d, 0x32, 0x21, 0x66, 0xb8, 0x00, 0xd9, 0xa8, 0x25, 0x28, 0xc0, 0x72, 0x67, 0x93],
        [0x1a, 0x87, 0x26, 0x9c, 0x96, 0xd0, 0xac, 0x67, 0xc4, 0x01, 0xda, 0x82, 0x29, 0xaa, 0x30, 0xc1, 0x68, 0xb3, 0x9b, 0x81, 0xd2, 0x89, 0x06, 0xdd, 0x31, 0x23, 0x9a, 0xc1, 0x35, 0x08, 0xf5, 0x0a],
        [0xaa, 0xc5, 0x12, 0xa9, 0xf5, 0xc3, 0xbf, 0xe5, 0x1e, 0x0d, 0x78, 0x23, 0xac, 0x5f, 0x2a, 0x44, 0xf8, 0x3b, 0x83, 0xe0, 0x72, 0xb8, 0x00, 0xdf, 0x2d, 0xd0, 0x8e, 0x43, 0x17, 0x0e, 0xb5, 0xca],
        [0xab, 0xcd, 0x13, 0x98, 0x7c, 0x91, 0xcc, 0xef, 0x5c, 0x00, 0x50, 0x42, 0x30, 0x02, 0x34, 0xe1, 0xf8, 0x2a, 0x02, 0xc9, 0xf3, 0x9a, 0x00, 0xff, 0x34, 0x2c, 0xbe, 0xc0, 0x15, 0x18, 0xf8, 0x8c],
        [0x01, 0x95, 0x16, 0xcd, 0x95, 0x86, 0xa7, 0x37, 0x0c, 0x01, 0x5a, 0x06, 0xec, 0x23, 0x30, 0x44, 0x74, 0x32, 0x02, 0xa9, 0x73, 0xb8, 0x31, 0xdd, 0x25, 0x36, 0x82, 0x40, 0x1e, 0x0a, 0xb1, 0x19],
        [0x0b, 0x87, 0x06, 0xad, 0xb4, 0xa2, 0x2a, 0x37, 0x2c, 0x01, 0x5b, 0x42, 0xed, 0x22, 0xa0, 0xe0, 0xfc, 0x23, 0x9b, 0xa0, 0xf6, 0x98, 0x01, 0xdf, 0xa4, 0x77, 0x8a, 0xc2, 0x14, 0x02, 0xa1, 0x0d],
        [0x80, 0xdd, 0x10, 0xd8, 0x57, 0x85, 0xa7, 0xff, 0x56, 0x08, 0x5c, 0x62, 0x38, 0x91, 0x20, 0x64, 0x6c, 0xba, 0x02, 0x48, 0x63, 0x2a, 0x00, 0xbf, 0x65, 0x04, 0x96, 0x7a, 0x17, 0x18, 0xb1, 0xa9],
        [0xba, 0x40, 0x47, 0xbc, 0x44, 0x82, 0x49, 0x70, 0x15, 0x46, 0xf3, 0x02, 0x4b, 0xc2, 0x29, 0x5d, 0x78, 0xad, 0x35, 0xe2, 0x68, 0x97, 0x22, 0x9c, 0x90, 0xe3, 0xca, 0x8b, 0x35, 0xa9, 0x74, 0x66],
        [0x3f, 0x4b, 0x46, 0xcc, 0x06, 0x12, 0x5e, 0x70, 0xf1, 0xca, 0xf6, 0xc7, 0x6b, 0x4b, 0xb7, 0xf4, 0x72, 0xe4, 0x3d, 0xee, 0xcd, 0xb7, 0x04, 0xec, 0xf2, 0xe9, 0xe0, 0x80, 0x2f, 0xbf, 0x78, 0x1d],
        [0xa0, 0xe9, 0x22, 0xaa, 0x5c, 0xc1, 0xce, 0xe4, 0x16, 0x40, 0xd0, 0x82, 0x31, 0x88, 0x30, 0x20, 0x58, 0xb9, 0x13, 0xc1, 0x4e, 0x2a, 0x46, 0xbd, 0xf1, 0x41, 0xae, 0x69, 0x55, 0x9a, 0xb0, 0x88],
        [0x11, 0x48, 0x77, 0xcc, 0x26, 0x82, 0x6e, 0x70, 0xed, 0xc0, 0x52, 0xe6, 0x7d, 0x9a, 0xb1, 0xe4, 0x76, 0xe5, 0x16, 0xef, 0xc1, 0x95, 0x24, 0xec, 0x30, 0xe8, 0xaa, 0x80, 0x2d, 0xb3, 0x31, 0x2d],
        [0x32, 0xc9, 0x3f, 0xa8, 0x54, 0xc3, 0xb6, 0xef, 0x44, 0x40, 0x50, 0x02, 0xb4, 0x18, 0x38, 0x00, 0xe8, 0x2a, 0x13, 0x61, 0x42, 0xaa, 0x00, 0xbd, 0xe1, 0x81, 0x8e, 0x81, 0x55, 0x18, 0xb1, 0x89],
        [0x15, 0x4c, 0x7b, 0xee, 0x46, 0x82, 0x6f, 0x70, 0xe4, 0x80, 0xf2, 0xc2, 0x79, 0xc3, 0xb1, 0x70, 0x7e, 0xf4, 0x16, 0xea, 0xe9, 0xb5, 0x04, 0xee, 0xf0, 0xa8, 0xa2, 0x80, 0x2d, 0xb3, 0x61, 0xa1],
        [0x95, 0x4a, 0x5f, 0x66, 0x06, 0x0e, 0x6b, 0x70, 0xe4, 0x80, 0xf3, 0xc2, 0x73, 0xc3, 0xb1, 0x70, 0x7e, 0xec, 0x06, 0xee, 0xc9, 0xb1, 0x04, 0x6c, 0xf0, 0xa8, 0x22, 0x80, 0x6d, 0xb3, 0x62, 0x21],
        [0x00, 0xcd, 0x01, 0xaa, 0x34, 0xe1, 0x87, 0xe7, 0x14, 0x46, 0x58, 0x03, 0xa0, 0xe8, 0x20, 0x40, 0x58, 0x2b, 0x13, 0x02, 0x44, 0x2a, 0x28, 0xbf, 0x64, 0x14, 0x8a, 0x19, 0x55, 0x5a, 0xb1, 0xca],
        [0xa2, 0x8d, 0x23, 0xbe, 0xd4, 0xa2, 0x8e, 0xae, 0x14, 0x06, 0x58, 0x13, 0xa4, 0xaa, 0x30, 0x40, 0x59, 0x6b, 0x13, 0x81, 0x60, 0x9a, 0x70, 0xff, 0x20, 0x36, 0x9a, 0x80, 0x15, 0x58, 0xb4, 0xe8],
        [0x23, 0x8d, 0x0a, 0x8e, 0xf4, 0xa1, 0x96, 0xe6, 0x1c, 0x02, 0x5c, 0x02, 0x62, 0xe8, 0x3f, 0x40, 0xd8, 0x6a, 0x13, 0x43, 0x44, 0x2b, 0x08, 0xbf, 0x72, 0x34, 0x8a, 0x99, 0x55, 0x5b, 0xf9, 0x8a],
        [0x02, 0x8c, 0x22, 0xd9, 0x75, 0xf3, 0xb1, 0x67, 0x1e, 0x01, 0x5c, 0x32, 0xb8, 0xb6, 0xa4, 0x64, 0x79, 0x3a, 0x12, 0x08, 0x32, 0xb8, 0x21, 0xdf, 0x4c, 0x16, 0x9a, 0x43, 0x96, 0x08, 0xb5, 0xaf],
        [0x22, 0x79, 0x72, 0x48, 0x7d, 0xdd, 0xf0, 0x65, 0x8b, 0x48, 0x74, 0x0b, 0x2a, 0x9f, 0x66, 0x30, 0xc8, 0xb9, 0x93, 0x40, 0x46, 0x2b, 0x52, 0xdd, 0x77, 0x5d, 0x00, 0x2b, 0x53, 0x0c, 0x66, 0xce],
        [0x42, 0xb9, 0x3a, 0x98, 0x59, 0xc9, 0x30, 0x47, 0xaf, 0x05, 0xc0, 0x29, 0xac, 0x9e, 0x64, 0x50, 0x82, 0x3b, 0xeb, 0x10, 0x42, 0x0b, 0x92, 0xf1, 0x9d, 0x14, 0x1a, 0x3b, 0x10, 0x04, 0xaf, 0xaf],
        [0xaa, 0xc3, 0x62, 0x99, 0xd5, 0xd1, 0xf7, 0xf5, 0x15, 0x44, 0x54, 0x07, 0x28, 0x8d, 0x32, 0xc4, 0x78, 0xaa, 0x87, 0x45, 0x76, 0xba, 0x20, 0xbf, 0x27, 0x55, 0x8e, 0xc3, 0x57, 0x9c, 0x9c, 0xce],
        [0x2a, 0xc9, 0x62, 0x98, 0x77, 0x91, 0xf7, 0xf5, 0x55, 0x04, 0x54, 0x03, 0x2c, 0xc9, 0x32, 0x85, 0x5a, 0x2a, 0x93, 0x45, 0xe2, 0x3e, 0x24, 0xff, 0x66, 0x51, 0x9e, 0xc9, 0x7f, 0x9c, 0xf8, 0xcb],
        [0x0a, 0x99, 0x30, 0xf9, 0x3d, 0xc5, 0xbb, 0xc7, 0x1e, 0x09, 0x4c, 0x02, 0xac, 0xdf, 0x20, 0x60, 0xcf, 0x3b, 0xbb, 0x11, 0x02, 0x98, 0x53, 0xfd, 0xef, 0xd7, 0x1e, 0x4b, 0x12, 0x08, 0xa5, 0x3b],
        [0x08, 0x9b, 0x3b, 0xbb, 0x35, 0x96, 0x7b, 0xd7, 0x3f, 0x09, 0x7e, 0x02, 0xbf, 0xd7, 0x20, 0x60, 0xd6, 0x3b, 0x9f, 0xc0, 0x02, 0x98, 0x72, 0xfd, 0xef, 0xff, 0x2e, 0x88, 0x10, 0x0a, 0x25, 0xeb],
        [0x03, 0x87, 0x88, 0x91, 0xf2, 0xb2, 0xb8, 0x37, 0x36, 0x6b, 0x5a, 0xf7, 0xac, 0x2a, 0x92, 0xc0, 0xf7, 0x2f, 0xb7, 0x95, 0xb7, 0xd8, 0x45, 0xc7, 0x2e, 0x27, 0xbe, 0xc0, 0x8d, 0x84, 0xb9, 0x1d],
        [0x32, 0xcd, 0x68, 0xac, 0x57, 0x87, 0xc7, 0xe5, 0x54, 0x00, 0x50, 0x03, 0x31, 0x18, 0x32, 0x20, 0x78, 0xaa, 0x13, 0x61, 0xe6, 0xbe, 0x14, 0xbf, 0x71, 0x0d, 0x8e, 0x00, 0x75, 0x18, 0xf0, 0x8a],
        [0x9c, 0x40, 0x7a, 0x4c, 0x46, 0x52, 0xe9, 0x70, 0xe4, 0xc2, 0xf7, 0x82, 0x59, 0xcf, 0xbf, 0x6b, 0x7e, 0xfc, 0x37, 0xea, 0xc0, 0xf7, 0xb6, 0x48, 0x56, 0xa9, 0xe2, 0x88, 0x67, 0xb3, 0x68, 0xb0],
        [0x13, 0x95, 0x4a, 0x8e, 0x96, 0x80, 0xbe, 0x67, 0x34, 0x41, 0x5e, 0x26, 0x31, 0x32, 0x26, 0x40, 0x70, 0xa3, 0x13, 0x80, 0x52, 0x98, 0x00, 0x7f, 0x34, 0x25, 0xba, 0xc0, 0x15, 0x00, 0xb9, 0x69],
        [0x39, 0x5b, 0x6b, 0xec, 0x34, 0x87, 0x4b, 0x70, 0xc5, 0x48, 0xf7, 0xc6, 0x73, 0xc4, 0xb1, 0x64, 0x7e, 0xa8, 0x1e, 0xee, 0xc9, 0xa3, 0x3c, 0x9d, 0x70, 0xe9, 0xa2, 0x88, 0x3f, 0xab, 0x70, 0x71],
        [0x11, 0xc9, 0x7b, 0x69, 0x34, 0x8e, 0x6b, 0xf2, 0x75, 0x08, 0xfe, 0x46, 0xb3, 0xc3, 0xb0, 0x70, 0x7e, 0xa8, 0x06, 0xce, 0xcb, 0xb8, 0x34, 0x7d, 0xe0, 0xad, 0xa2, 0x00, 0x6f, 0x82, 0x61, 0xb9],
        [0x8a, 0x4c, 0x37, 0xdc, 0x4f, 0xf2, 0x38, 0x66, 0x94, 0x06, 0xd2, 0x32, 0x68, 0x2f, 0x21, 0xcb, 0xc8, 0x20, 0xbb, 0x80, 0x60, 0x85, 0x22, 0xcf, 0x07, 0x63, 0xda, 0x83, 0x16, 0x7c, 0x34, 0xcb],
        [0x06, 0xcc, 0x2a, 0x99, 0x54, 0x93, 0xb2, 0xe2, 0xdf, 0x00, 0x5c, 0x02, 0x7c, 0xd8, 0x32, 0x60, 0x7b, 0x2a, 0x12, 0x45, 0xa2, 0xb8, 0x04, 0xbf, 0x6c, 0x1e, 0x83, 0x41, 0x1d, 0x09, 0x39, 0x8c],
        [0x2a, 0x1d, 0x03, 0x9e, 0x5d, 0xd6, 0x06, 0x66, 0x94, 0x00, 0xca, 0x03, 0xe8, 0xf6, 0x24, 0xc0, 0xd8, 0x7a, 0x9b, 0xc2, 0x04, 0x8b, 0x1a, 0xff, 0x23, 0x77, 0x8a, 0x0b, 0x12, 0x2e, 0x3c, 0x4e],
        [0x12, 0xcd, 0x23, 0xcb, 0x9d, 0xc4, 0xcf, 0x75, 0x85, 0x00, 0xca, 0x03, 0x20, 0xce, 0x24, 0x64, 0xc8, 0x22, 0x0a, 0xe0, 0x5a, 0xa9, 0x30, 0xfd, 0xe3, 0xfc, 0x8a, 0x52, 0x33, 0x0e, 0xf4, 0x4e],
        [0xa8, 0x8e, 0x35, 0x9b, 0x5f, 0xc1, 0x90, 0x67, 0x95, 0x14, 0x50, 0x32, 0xa8, 0x4d, 0x30, 0xc1, 0x4a, 0x3b, 0x83, 0x01, 0x62, 0xaf, 0x42, 0xff, 0x27, 0x01, 0x9a, 0x01, 0x17, 0x0c, 0x34, 0xeb],
        [0x03, 0xcc, 0x03, 0x9f, 0x15, 0x87, 0xaa, 0xe2, 0x14, 0x04, 0x58, 0x03, 0xe1, 0xee, 0x30, 0x40, 0x7c, 0x6a, 0x13, 0xcf, 0x66, 0x98, 0x2e, 0xff, 0x27, 0x6e, 0x9a, 0xc0, 0x15, 0x1f, 0x70, 0xcc],
        [0x02, 0x81, 0x0b, 0xab, 0x16, 0x82, 0x2a, 0x79, 0x84, 0x01, 0x53, 0x42, 0x7d, 0x43, 0x28, 0xc0, 0x7a, 0x09, 0x1b, 0xb2, 0x52, 0x89, 0x08, 0xdd, 0x22, 0xa1, 0x8a, 0xc8, 0x34, 0x8a, 0x38, 0x00],
        [0x2f, 0x60, 0xe2, 0xe0, 0x5c, 0x35, 0x9a, 0xc8, 0xc2, 0xa8, 0x31, 0x8b, 0x08, 0xcc, 0x7d, 0x20, 0xfc, 0xc9, 0x13, 0xf3, 0x86, 0x33, 0xae, 0xf0, 0x7a, 0x0d, 0xaf, 0xac, 0xd5, 0x3b, 0x73, 0x92],
        [0x20, 0xaf, 0x13, 0x29, 0x50, 0x8b, 0x62, 0xc2, 0x6e, 0x29, 0x6a, 0x48, 0xa6, 0x80, 0x20, 0x94, 0x38, 0xaa, 0x46, 0x50, 0x02, 0xba, 0x80, 0x2f, 0xbc, 0xa4, 0x20, 0x60, 0x41, 0x0b, 0x42, 0x88],
        [0xa9, 0xd9, 0x00, 0xba, 0x5d, 0xc5, 0xb4, 0xe5, 0x16, 0x19, 0xd0, 0x1e, 0xb0, 0x99, 0x32, 0x01, 0xc8, 0x2a, 0x13, 0x41, 0x76, 0x2a, 0x52, 0xbd, 0x6f, 0x91, 0x9e, 0x78, 0x59, 0x08, 0xf1, 0x9a],
        [0xd3, 0x48, 0x97, 0x94, 0x0a, 0xc0, 0x3c, 0x62, 0xad, 0xda, 0xd2, 0x42, 0x7d, 0x8e, 0x31, 0xeb, 0x66, 0x65, 0x7f, 0xcf, 0x61, 0xc5, 0x82, 0xe8, 0x91, 0xea, 0xfa, 0x80, 0x24, 0xb1, 0x21, 0x29],
        [0x16, 0x81, 0x20, 0xab, 0x15, 0xa7, 0xaa, 0xc8, 0x06, 0x09, 0x51, 0x82, 0x4d, 0xcc, 0x38, 0x00, 0x78, 0x09, 0x06, 0xf1, 0x92, 0x21, 0x08, 0xd5, 0x99, 0x84, 0x8b, 0xe4, 0xdd, 0x0b, 0x39, 0x80],
        [0x18, 0x81, 0x06, 0x8a, 0x14, 0x82, 0x23, 0x2d, 0x0c, 0x01, 0x57, 0x06, 0x6d, 0x43, 0x28, 0x45, 0x76, 0xa3, 0x06, 0xa8, 0x72, 0x88, 0x09, 0xdd, 0xb0, 0xe4, 0x82, 0x40, 0x1e, 0x0a, 0xb1, 0x28],
        [0x78, 0x48, 0x79, 0x65, 0x64, 0x0e, 0x6f, 0xb0, 0x60, 0x0c, 0x76, 0x08, 0x53, 0x57, 0x91, 0x30, 0x3e, 0xfc, 0x44, 0x6a, 0xc8, 0xb2, 0xf9, 0x2c, 0xf7, 0xa9, 0x21, 0x04, 0x61, 0xb3, 0x46, 0xa1],
        [0xa0, 0xdb, 0x3f, 0xda, 0xfc, 0x85, 0x8d, 0xf5, 0xd4, 0x1c, 0xd4, 0x46, 0xb0, 0x0b, 0x20, 0x41, 0xcc, 0x3b, 0x02, 0xc8, 0xea, 0xab, 0x52, 0xfd, 0xe1, 0x90, 0xbe, 0x48, 0x1d, 0x88, 0xb4, 0xb8],
        [0x90, 0x4b, 0x16, 0xdc, 0x2e, 0xc2, 0xed, 0x74, 0x95, 0x12, 0xd2, 0x62, 0x2c, 0x8a, 0xb1, 0xe7, 0x6a, 0xa1, 0x3f, 0xc2, 0x61, 0xcf, 0x02, 0xfd, 0xb0, 0xa2, 0xfe, 0x80, 0x35, 0x38, 0x31, 0x29],
        [0x59, 0xab, 0xd2, 0x05, 0xb2, 0x8a, 0x67, 0xb3, 0x62, 0xa1, 0x6e, 0xec, 0xd3, 0x60, 0x80, 0xb4, 0x34, 0x6e, 0x6c, 0x6c, 0x99, 0xb8, 0x8d, 0x2e, 0xac, 0xaf, 0x25, 0x50, 0xe8, 0xa2, 0xca, 0x21],
        [0x01, 0x9b, 0x93, 0x69, 0x92, 0x8a, 0x22, 0x13, 0x6e, 0x01, 0x6a, 0xe4, 0xb5, 0xc3, 0xa0, 0x50, 0x72, 0x6f, 0x2a, 0x58, 0x03, 0x98, 0x80, 0x6c, 0xbc, 0xa4, 0x22, 0x40, 0x88, 0x82, 0x21, 0x21],
        [0x51, 0x40, 0xdf, 0xe5, 0x06, 0x0e, 0x6d, 0x30, 0x74, 0x42, 0xf2, 0xc6, 0x53, 0x23, 0x90, 0x73, 0x72, 0xc4, 0x3d, 0xae, 0xc9, 0x90, 0xd5, 0x4c, 0xd4, 0xab, 0xe2, 0x80, 0x2d, 0xa2, 0x60, 0x30],
        [0x23, 0xa5, 0x58, 0xcb, 0xfc, 0xb9, 0x9a, 0x8d, 0x0e, 0x11, 0x79, 0x10, 0xe8, 0x58, 0x5e, 0x20, 0xdb, 0x0b, 0x13, 0x11, 0x16, 0x2a, 0x6b, 0xf5, 0x0e, 0x5d, 0x9a, 0xfb, 0xd5, 0x4a, 0xd8, 0x8c],
        [0x93, 0x49, 0x3a, 0xbc, 0x36, 0x86, 0x3f, 0xe4, 0x8c, 0x48, 0x52, 0x07, 0x2c, 0x8a, 0x33, 0x40, 0xe6, 0x61, 0x9f, 0xc7, 0xc4, 0x8d, 0x00, 0xed, 0x70, 0x21, 0xaa, 0x8a, 0x14, 0x34, 0x29, 0x4b],
        [0x12, 0x48, 0x02, 0xcc, 0x07, 0xd7, 0xae, 0xe0, 0x84, 0x08, 0xc6, 0x02, 0x68, 0xc6, 0x25, 0xe4, 0xe8, 0xf1, 0x93, 0xe2, 0x48, 0x83, 0x30, 0xfd, 0x33, 0x22, 0x8a, 0x03, 0x31, 0x3f, 0x38, 0x9e],
        [0x41, 0x16, 0xd5, 0x65, 0x62, 0x0a, 0x69, 0x14, 0x28, 0x60, 0x7a, 0xe4, 0xdd, 0x07, 0x90, 0x72, 0x26, 0xd5, 0x3c, 0xbc, 0x81, 0x94, 0xe5, 0x40, 0x84, 0xa8, 0xa2, 0x84, 0xac, 0xf2, 0x63, 0x31],
        [0x2a, 0xcf, 0x02, 0x9f, 0x95, 0x93, 0xd2, 0xf6, 0x17, 0x01, 0x58, 0xc7, 0x68, 0xec, 0x37, 0xc4, 0x5d, 0x6a, 0x93, 0x57, 0x36, 0xb8, 0x08, 0xff, 0xaa, 0x56, 0x8a, 0xcb, 0x16, 0x8f, 0x38, 0xde],
        [0x18, 0x8c, 0x2f, 0xfb, 0xb4, 0xa3, 0xab, 0xd6, 0x56, 0x0f, 0x5e, 0x22, 0xf1, 0xe7, 0xb1, 0x60, 0x7b, 0x2a, 0x1f, 0x89, 0xa3, 0xb8, 0x29, 0x3f, 0xe0, 0xaf, 0xba, 0x80, 0xb4, 0xc8, 0x35, 0xe9],
        [0x45, 0x3f, 0x51, 0x61, 0xa2, 0x2a, 0x6a, 0x92, 0x68, 0xe0, 0x7a, 0xec, 0x9f, 0x31, 0x90, 0x34, 0x37, 0x45, 0x54, 0x3e, 0x81, 0xb0, 0x85, 0x60, 0xec, 0xdc, 0x24, 0x84, 0xee, 0xc3, 0x43, 0x21],
        [0xcb, 0x56, 0xd9, 0xc5, 0x22, 0x0a, 0x6d, 0x16, 0x38, 0xe2, 0x77, 0xf6, 0xdd, 0x27, 0x80, 0x7b, 0x36, 0xd5, 0x3c, 0xae, 0xe1, 0x94, 0x25, 0xca, 0xe4, 0xab, 0x7e, 0x84, 0x2c, 0xf0, 0xa1, 0x31],
        [0xdc, 0x70, 0x93, 0xf0, 0x0b, 0x4c, 0x55, 0x11, 0x32, 0x80, 0xa0, 0x18, 0x9f, 0x97, 0x40, 0x3a, 0xa2, 0x91, 0x78, 0x00, 0x49, 0x47, 0xb2, 0x00, 0xd1, 0xc1, 0x42, 0x28, 0x20, 0x30, 0xa7, 0x32],
        [0x30, 0x6c, 0x3b, 0x78, 0x14, 0xcf, 0x53, 0x42, 0xe0, 0x8c, 0xe5, 0x08, 0x12, 0x91, 0x01, 0x30, 0x88, 0xb8, 0x44, 0x0a, 0x08, 0xa5, 0xf0, 0x0e, 0x43, 0xa9, 0x20, 0x11, 0x40, 0x09, 0x46, 0xa2],
        [0x28, 0x24, 0x3f, 0x60, 0x1c, 0xcf, 0x51, 0x60, 0xe0, 0xac, 0xa5, 0x08, 0x12, 0x51, 0x40, 0x31, 0x18, 0xb0, 0x40, 0x28, 0x28, 0xe3, 0xe0, 0x1c, 0xe3, 0x8b, 0x21, 0x35, 0x40, 0x28, 0x46, 0xa2],
        [0xe8, 0x34, 0x3c, 0x72, 0xa9, 0x7a, 0x15, 0x47, 0x70, 0xb6, 0xe5, 0x19, 0x02, 0x11, 0x02, 0x1b, 0x81, 0x56, 0xe0, 0x02, 0x64, 0x7e, 0xba, 0x26, 0x53, 0xfb, 0x50, 0x23, 0xc2, 0x40, 0xce, 0x4a],
        [0xe8, 0xb4, 0xbd, 0x74, 0xa8, 0x4d, 0x51, 0x13, 0xb2, 0xfe, 0x24, 0x3d, 0x96, 0x5d, 0x42, 0x1b, 0x01, 0x56, 0xc1, 0x31, 0x62, 0x64, 0xf2, 0x02, 0x4f, 0x93, 0x55, 0x3f, 0x42, 0x60, 0x46, 0x72],
        [0xcd, 0x32, 0x95, 0x01, 0x22, 0x4e, 0x23, 0x30, 0x28, 0xf8, 0x23, 0x7c, 0x96, 0x16, 0xc8, 0x72, 0x26, 0x54, 0x6c, 0x3e, 0x51, 0xd4, 0xb9, 0x02, 0xcc, 0xa1, 0x25, 0x04, 0x6a, 0x81, 0x02, 0x71],
        [0x55, 0x12, 0xf5, 0x75, 0xa2, 0x4e, 0x69, 0x31, 0x70, 0xe8, 0x23, 0xec, 0x53, 0x37, 0xd1, 0x7a, 0x37, 0xf4, 0x7d, 0x18, 0xb1, 0xd0, 0xa9, 0x0c, 0x4d, 0xea, 0x65, 0x84, 0xac, 0xa0, 0x07, 0x71],
        [0x54, 0x60, 0x3d, 0xe2, 0x0a, 0xec, 0x17, 0x01, 0xae, 0x3d, 0x87, 0x19, 0xd6, 0xd7, 0x48, 0x5a, 0xa5, 0x95, 0x6d, 0x02, 0x48, 0x03, 0xaa, 0x50, 0xf1, 0xa1, 0x2f, 0x3d, 0x00, 0x52, 0x87, 0x62],
        [0x12, 0x1c, 0x47, 0x9e, 0x14, 0x82, 0x28, 0x56, 0xa5, 0x82, 0xd2, 0x82, 0x29, 0x87, 0x21, 0x41, 0x6c, 0xb7, 0x93, 0x80, 0x28, 0x9e, 0x10, 0xee, 0xa2, 0xa0, 0x9a, 0x80, 0x26, 0xb0, 0x30, 0x08],
        [0x28, 0x31, 0x38, 0xe9, 0x5f, 0xc5, 0xb1, 0x40, 0x06, 0x0d, 0x75, 0x22, 0x1c, 0x9f, 0x60, 0x20, 0xea, 0x19, 0x93, 0xd0, 0x02, 0x2a, 0x3a, 0xd1, 0x4f, 0xcd, 0x22, 0x69, 0xf3, 0x8a, 0x64, 0xa3],
        [0x0a, 0xca, 0x03, 0xa8, 0x14, 0x80, 0x69, 0x60, 0x44, 0x05, 0xd2, 0x06, 0x21, 0x82, 0x30, 0xc0, 0xe8, 0x88, 0x03, 0x91, 0x0a, 0x9c, 0x00, 0xf5, 0x00, 0x60, 0xba, 0xc0, 0x15, 0x08, 0x74, 0x00],
        [0xae, 0x31, 0x68, 0xe2, 0x5d, 0xd1, 0xbb, 0x41, 0x82, 0x4d, 0x3d, 0x9b, 0x18, 0xdd, 0x7e, 0x00, 0xc8, 0x19, 0x93, 0xd1, 0x12, 0x23, 0x32, 0xd0, 0x5f, 0xc9, 0xaa, 0xe9, 0x57, 0x0e, 0xd7, 0xf2],
        [0x09, 0xcd, 0x33, 0xbf, 0x36, 0xe3, 0x83, 0x67, 0x56, 0x40, 0x58, 0x22, 0xf4, 0xfc, 0xa1, 0xe4, 0xdb, 0x2a, 0x1a, 0x01, 0x83, 0xb8, 0x60, 0x9f, 0x20, 0x67, 0xb2, 0x00, 0x14, 0x4a, 0x30, 0xe9],
        [0x32, 0x89, 0x02, 0xae, 0x94, 0x83, 0x8b, 0xe4, 0x86, 0x09, 0xd8, 0xa2, 0x21, 0xe8, 0xb0, 0x40, 0x58, 0x6a, 0x1f, 0xc3, 0x12, 0xab, 0x48, 0xfd, 0xb0, 0x64, 0xaa, 0x88, 0x34, 0x8a, 0x70, 0xa8],
        [0x01, 0x8d, 0x1e, 0x9d, 0x34, 0xe7, 0xb2, 0xe7, 0x16, 0x48, 0x5a, 0x02, 0x3c, 0xe0, 0x21, 0x40, 0xc9, 0x2b, 0x02, 0x00, 0x27, 0x88, 0x03, 0xff, 0x6c, 0x27, 0xa2, 0x40, 0x18, 0x08, 0x31, 0x99],
        [0x00, 0x95, 0x19, 0xa1, 0xb2, 0xee, 0x3f, 0xb7, 0x3e, 0x49, 0x5a, 0x70, 0xd4, 0x76, 0x00, 0x40, 0xb7, 0x17, 0x1b, 0x31, 0xb3, 0x98, 0x01, 0x5f, 0xed, 0xa6, 0x3f, 0xc4, 0x80, 0x40, 0x85, 0x3b],
    ];
    #[rustfmt::skip]
    pub const ORB_FIXTURE_DESC: [[u8; 32]; 40] = [
        [0xa2, 0x4e, 0x60, 0x0d, 0x74, 0x5b, 0xb7, 0xee, 0x5c, 0xae, 0x7c, 0xa3, 0xe0, 0x0c, 0x97, 0x04, 0xda, 0x7c, 0x82, 0x63, 0xe4, 0xb4, 0x65, 0xce, 0x24, 0x14, 0x2b, 0x87, 0x55, 0x3d, 0x5d, 0xcd],
        [0x3a, 0x49, 0x6b, 0xca, 0x1c, 0xc4, 0x0b, 0xd0, 0x87, 0x44, 0xcc, 0x83, 0x39, 0xc7, 0xb3, 0x64, 0x5c, 0xb8, 0x9b, 0xfd, 0x8a, 0x83, 0x74, 0xff, 0xf1, 0x2d, 0x8a, 0x09, 0x37, 0x9f, 0x34, 0xf8],
        [0xad, 0x6a, 0xf9, 0x52, 0xe4, 0xbd, 0xed, 0xd6, 0x73, 0xbb, 0x35, 0x5a, 0x13, 0x81, 0xd5, 0xbf, 0x2f, 0x8a, 0x56, 0x18, 0xbb, 0x76, 0xee, 0xa2, 0x1e, 0x89, 0xf4, 0xec, 0xef, 0xd9, 0xd2, 0xa5],
        [0xca, 0x44, 0x14, 0xbe, 0x4f, 0x60, 0xb0, 0x08, 0x29, 0x16, 0xc1, 0x1b, 0xcc, 0x1a, 0x09, 0xdb, 0xe2, 0x95, 0xfb, 0xa3, 0x64, 0x47, 0xc3, 0xca, 0x95, 0x32, 0xda, 0x0b, 0x12, 0x7d, 0x2d, 0x82],
        [0x26, 0x76, 0x90, 0x43, 0xd9, 0x7b, 0x92, 0xcc, 0x3b, 0xbd, 0x2d, 0xb8, 0x02, 0xf6, 0x4d, 0x9e, 0x99, 0x5c, 0xc0, 0x72, 0x12, 0x72, 0xf8, 0x22, 0x2f, 0xd4, 0x05, 0x77, 0xc3, 0x77, 0xce, 0xb7],
        [0x38, 0x58, 0x33, 0x5c, 0x5d, 0xd5, 0xc1, 0x56, 0xd7, 0x48, 0xf4, 0x02, 0x32, 0xdd, 0x35, 0x64, 0xc8, 0xb8, 0x16, 0x08, 0x0b, 0xaf, 0x72, 0x9f, 0x53, 0xfc, 0x90, 0x3a, 0x53, 0x08, 0x74, 0xba],
        [0xcc, 0xb1, 0x93, 0xe6, 0xd3, 0x4c, 0x31, 0x31, 0x39, 0x81, 0x63, 0x5d, 0x5c, 0xf7, 0x48, 0xd5, 0xb4, 0x5f, 0xa1, 0x30, 0x31, 0xda, 0x98, 0x46, 0xfd, 0xf0, 0x47, 0x4e, 0x8a, 0x22, 0xa7, 0xb3],
        [0x89, 0x87, 0x06, 0xb1, 0xb5, 0x8e, 0xec, 0x37, 0x7c, 0x09, 0x52, 0xd7, 0xad, 0x0a, 0xb2, 0xc1, 0xe6, 0x27, 0x1f, 0x8d, 0xf4, 0xbc, 0x05, 0xd5, 0xe4, 0xeb, 0xfa, 0xc0, 0x0e, 0x8c, 0xf9, 0x88],
        [0xe4, 0x25, 0x6d, 0x62, 0xbc, 0x2d, 0x4f, 0xbb, 0x81, 0x30, 0xbd, 0xd8, 0xd3, 0x67, 0x5d, 0x33, 0x0d, 0xd2, 0x5d, 0x7e, 0xb9, 0x23, 0x7a, 0x36, 0xfb, 0xd9, 0xa5, 0x7c, 0xfb, 0xdb, 0xc2, 0x30],
        [0xfc, 0x72, 0xff, 0x76, 0x0c, 0x0f, 0x45, 0x58, 0xf3, 0xb6, 0xa3, 0x4c, 0x53, 0xf1, 0x59, 0x3b, 0x3c, 0xf4, 0x7d, 0x6e, 0xc8, 0x67, 0xbb, 0x20, 0xd3, 0xeb, 0x65, 0x3c, 0xe1, 0xbb, 0x56, 0x70],
        [0x17, 0xa9, 0x46, 0x13, 0xce, 0x95, 0xa8, 0xe4, 0xee, 0xf1, 0x5b, 0x42, 0x10, 0xc8, 0xf4, 0xe7, 0x21, 0x29, 0x32, 0xdf, 0xd7, 0xf9, 0x8e, 0x70, 0x1c, 0x08, 0x8f, 0x60, 0x6d, 0x07, 0xf9, 0x84],
        [0xbf, 0xe3, 0x65, 0x88, 0xec, 0xd1, 0xcf, 0xd8, 0xc7, 0x5c, 0x95, 0xfe, 0x03, 0xcc, 0xfe, 0x27, 0x4c, 0x80, 0xd7, 0xff, 0xff, 0x67, 0x6e, 0x95, 0x3a, 0x08, 0xad, 0xfc, 0x77, 0x9f, 0xf6, 0xf5],
        [0xdb, 0xaa, 0xc4, 0x14, 0xca, 0x10, 0x6c, 0x47, 0xc3, 0xf7, 0x04, 0xd7, 0x29, 0xa2, 0xd6, 0xdf, 0x63, 0xa6, 0xa2, 0x8c, 0xc4, 0x4f, 0x86, 0xc0, 0x16, 0x0a, 0xdc, 0xa9, 0xaf, 0xe4, 0xf8, 0x8c],
        [0x5f, 0x07, 0x42, 0x47, 0x32, 0x1e, 0x6e, 0x30, 0x29, 0xa0, 0x6b, 0xc6, 0x49, 0x22, 0x9d, 0xd4, 0x36, 0xc5, 0x0c, 0xee, 0xc5, 0x90, 0x8d, 0xce, 0xb6, 0xe8, 0x83, 0x84, 0x2f, 0xa7, 0x6b, 0x04],
        [0x7c, 0x27, 0xd0, 0x64, 0xe0, 0x2a, 0x4f, 0x29, 0x31, 0xa6, 0xad, 0xa9, 0x4b, 0x23, 0x49, 0x1a, 0x5b, 0xdc, 0xe5, 0xa2, 0x68, 0x72, 0xc8, 0x08, 0xf2, 0xe7, 0x65, 0x3d, 0xe3, 0xf2, 0xc6, 0xa0],
        [0x40, 0xf1, 0x98, 0x67, 0xb3, 0x48, 0x16, 0xad, 0x2a, 0xa5, 0x28, 0x29, 0xe6, 0x77, 0x48, 0x90, 0x93, 0x5f, 0xe0, 0x40, 0x52, 0x58, 0xc9, 0x6f, 0xad, 0xb5, 0x15, 0x6b, 0x40, 0x02, 0x8f, 0x0b],
        [0xfe, 0xa5, 0xa8, 0xa6, 0xd9, 0x40, 0x1e, 0xa9, 0x22, 0xa5, 0x8b, 0x9b, 0x48, 0x6e, 0x5a, 0x92, 0xb1, 0xc7, 0xf9, 0xf7, 0x3c, 0x47, 0x89, 0x02, 0x7d, 0x25, 0x8d, 0xbd, 0xa1, 0x77, 0x9e, 0x52],
        [0x4c, 0x27, 0xf8, 0x67, 0xf0, 0x2a, 0x7f, 0x93, 0x38, 0xae, 0x3f, 0xa9, 0xd5, 0x23, 0x4b, 0x12, 0x97, 0x5f, 0x65, 0x32, 0x10, 0x90, 0xc9, 0x6a, 0xee, 0xb5, 0x27, 0x81, 0x80, 0xf3, 0x8f, 0x03],
        [0x7e, 0xab, 0xea, 0xba, 0xfe, 0xb1, 0x58, 0xdf, 0x87, 0x5d, 0x95, 0xbf, 0x2b, 0xec, 0xfe, 0x1c, 0x4f, 0xaa, 0xaf, 0x95, 0x9e, 0x6b, 0xee, 0xb1, 0xda, 0x45, 0xb4, 0xfb, 0xb7, 0xcd, 0xfe, 0xd5],
        [0x08, 0x2a, 0xf8, 0x5f, 0x6c, 0xb9, 0xe9, 0x9e, 0x56, 0x2f, 0x75, 0x32, 0xf1, 0xe9, 0xdb, 0x68, 0x59, 0x7e, 0x23, 0x18, 0x02, 0x7c, 0xef, 0x33, 0x4e, 0x2f, 0xb1, 0xa9, 0xd7, 0xdb, 0x5c, 0xa9],
        [0x0a, 0x34, 0xb3, 0x05, 0xf3, 0x7a, 0xb1, 0x07, 0x39, 0xa7, 0x4b, 0x31, 0x8c, 0x36, 0x46, 0x8c, 0xa3, 0x3f, 0xb1, 0x11, 0x30, 0xd8, 0xb1, 0xc2, 0x0f, 0x36, 0x5a, 0x03, 0x82, 0x44, 0x87, 0xfe],
        [0x98, 0xcf, 0x43, 0xba, 0x12, 0x81, 0x4e, 0xf5, 0x15, 0x05, 0x51, 0xf7, 0x7b, 0xc7, 0xa8, 0x41, 0x7e, 0x9b, 0x05, 0xcf, 0xfa, 0x0f, 0x0e, 0x3f, 0xf0, 0xc1, 0xa8, 0xc8, 0x37, 0x88, 0xa4, 0xb9],
        [0xc0, 0x23, 0x18, 0x0d, 0xf0, 0x5a, 0x20, 0x2b, 0x2c, 0x21, 0x4a, 0xa0, 0x84, 0x02, 0x82, 0xd0, 0x90, 0x0f, 0xea, 0x20, 0x54, 0x98, 0xc1, 0x4c, 0xa4, 0x34, 0x2a, 0x47, 0x08, 0x00, 0x8b, 0x0e],
        [0x60, 0x05, 0x95, 0x75, 0xa0, 0x6e, 0x25, 0x27, 0x38, 0xa3, 0x6a, 0x19, 0xd6, 0x33, 0x00, 0x1b, 0x81, 0x76, 0x68, 0x00, 0x10, 0xf8, 0xd1, 0x08, 0xc5, 0xb6, 0x55, 0x00, 0xc0, 0x60, 0x87, 0x22],
        [0x78, 0xa4, 0xf8, 0x7a, 0xdd, 0xf1, 0x51, 0x4f, 0xd3, 0x17, 0xac, 0x18, 0x33, 0xb1, 0x62, 0x38, 0x49, 0xba, 0xe4, 0x10, 0x3e, 0x6f, 0xe2, 0x13, 0xdb, 0xcf, 0x14, 0x79, 0xf3, 0x48, 0xee, 0xaa],
        [0x82, 0xb1, 0x9a, 0x9b, 0x86, 0x81, 0xbc, 0x45, 0x9e, 0x13, 0x11, 0xd7, 0xa4, 0xca, 0x68, 0x08, 0xaf, 0x3b, 0x1b, 0x13, 0x16, 0xcd, 0x9a, 0xe1, 0x84, 0x14, 0xda, 0xab, 0x9c, 0x4a, 0x99, 0x9b],
        [0x2d, 0x28, 0xd1, 0x41, 0x75, 0x1b, 0xe2, 0x50, 0x7b, 0xe9, 0x75, 0x68, 0x1a, 0x90, 0xd5, 0xbc, 0x1e, 0x7c, 0x12, 0x5a, 0x91, 0xb6, 0xac, 0xe2, 0x0e, 0x1c, 0x01, 0x66, 0xcf, 0xa7, 0x42, 0xa5],
        [0xc8, 0xd4, 0x08, 0xaf, 0x30, 0x0e, 0x5f, 0x3d, 0x3c, 0x27, 0x43, 0x15, 0xf7, 0x43, 0x0a, 0x50, 0x97, 0x05, 0x0d, 0x2e, 0x6a, 0x94, 0x59, 0x1f, 0xe5, 0xf7, 0x2e, 0x14, 0x18, 0x12, 0x81, 0x52],
        [0x0a, 0x56, 0x12, 0xaf, 0x33, 0x2a, 0x36, 0x27, 0x3d, 0x06, 0x4a, 0xb5, 0xec, 0x27, 0x22, 0xd0, 0xb7, 0x37, 0x8d, 0x21, 0x64, 0x9c, 0x01, 0xef, 0xe4, 0xf7, 0x1a, 0x83, 0x02, 0x56, 0xad, 0x5b],
        [0xba, 0x16, 0x6e, 0xbe, 0x2c, 0xa1, 0x9c, 0x7f, 0x94, 0x1f, 0xd5, 0xb7, 0x29, 0xcd, 0xbe, 0x4b, 0xcf, 0xb3, 0xbf, 0x81, 0xfc, 0xef, 0x46, 0x9f, 0x52, 0x73, 0xfa, 0x99, 0x55, 0xcc, 0xbc, 0xdb],
        [0xdd, 0x17, 0x45, 0xa5, 0x92, 0x2e, 0x6d, 0x3b, 0x34, 0xf2, 0x8b, 0xd6, 0xd1, 0x27, 0x98, 0x6b, 0x67, 0xc6, 0x78, 0xae, 0xf9, 0xd0, 0x7d, 0x0c, 0xe4, 0xa2, 0xef, 0x14, 0xac, 0xf0, 0x90, 0x30],
        [0xe6, 0x00, 0xb1, 0x47, 0x34, 0x4e, 0xcb, 0x09, 0x4b, 0xaa, 0xb3, 0x08, 0xd2, 0x7d, 0x5d, 0x2a, 0x99, 0x54, 0x7d, 0x3a, 0x00, 0xa4, 0xe9, 0xca, 0x89, 0xbe, 0x07, 0x16, 0xc0, 0x6b, 0x54, 0x22],
        [0x66, 0xb5, 0xa1, 0x00, 0xd9, 0x7f, 0x15, 0xcf, 0x3b, 0xf5, 0xa9, 0x19, 0x8e, 0xbe, 0x4e, 0x1e, 0x91, 0x9f, 0xe1, 0x11, 0x16, 0x5a, 0xfb, 0xd2, 0x3f, 0x14, 0x15, 0x7f, 0xc2, 0x44, 0x86, 0xf6],
        [0x18, 0x9d, 0x77, 0xcc, 0x76, 0x42, 0xaf, 0x75, 0x05, 0x40, 0x7a, 0xe2, 0x39, 0x17, 0xa4, 0xe5, 0x7e, 0xb1, 0x17, 0xe8, 0xd1, 0xbb, 0x31, 0xcd, 0x71, 0xa8, 0xa2, 0x80, 0x7f, 0xaa, 0xe5, 0xf9],
        [0x20, 0x60, 0xf8, 0x48, 0xcc, 0x21, 0xa1, 0xde, 0x52, 0x37, 0x6c, 0x1a, 0x10, 0x91, 0x7d, 0x08, 0x59, 0x18, 0x23, 0x18, 0x02, 0x36, 0xe2, 0x32, 0x4d, 0x8c, 0x34, 0x6c, 0xd5, 0xd9, 0x56, 0xaa],
        [0xd2, 0x4c, 0x02, 0xbc, 0x13, 0xd6, 0x52, 0x26, 0xb5, 0x86, 0xc2, 0x11, 0x2e, 0x82, 0x27, 0xd4, 0xe0, 0xe5, 0xc9, 0xc0, 0x6c, 0x8f, 0x12, 0xee, 0xb3, 0xf4, 0x98, 0x1b, 0x22, 0x35, 0x2c, 0x4e],
        [0xfd, 0x69, 0x04, 0x15, 0xbb, 0x4b, 0x06, 0xa8, 0x92, 0xf4, 0xa0, 0xfd, 0xc6, 0x40, 0xc3, 0x97, 0x10, 0x6c, 0xec, 0x76, 0x6d, 0x61, 0xcf, 0x3c, 0xbc, 0x00, 0x65, 0x3f, 0x7a, 0x3d, 0x1a, 0x55],
        [0xb8, 0x29, 0x63, 0x5e, 0x5d, 0xd1, 0x43, 0x75, 0xc1, 0x84, 0xf4, 0x8a, 0x2b, 0xd5, 0xb0, 0xb4, 0x68, 0xba, 0xc4, 0xc8, 0x7a, 0xab, 0x26, 0xbd, 0x73, 0xc9, 0xa0, 0x69, 0x77, 0x8e, 0xfe, 0xe8],
        [0x57, 0x5e, 0xe7, 0xb7, 0xa4, 0x26, 0xee, 0x90, 0xb9, 0xca, 0xdb, 0xf7, 0xd9, 0x6e, 0xbd, 0xea, 0xff, 0xe5, 0x3d, 0xef, 0xad, 0xd5, 0x9d, 0xea, 0x90, 0x7e, 0xef, 0x80, 0xad, 0xf7, 0x31, 0x55],
        [0x34, 0xaa, 0xef, 0xba, 0x4e, 0xe1, 0xc9, 0xda, 0x86, 0x57, 0x91, 0xfb, 0x33, 0xc5, 0xfa, 0x0b, 0x6f, 0xa2, 0x6f, 0x9d, 0x9a, 0x4f, 0x2e, 0x31, 0x58, 0x02, 0xf9, 0xf9, 0xb5, 0xc9, 0x76, 0xf0],
    ];
    pub const ORB_BUDGET_500_8: [usize; 8] = [109, 90, 75, 63, 52, 44, 36, 31];
    /// `budget_emulated/n100_l8_s1.2`: OpenCV's per-level quota for 100 features.
    pub const ORB_BUDGET_100_8: [usize; 8] = [22, 18, 15, 13, 10, 9, 7, 6];
    /// `variants/nfeatures_100`: what OpenCV detects per octave with 100 features (99).
    pub const ORB_DETECT_100_8_PER_OCTAVE: [usize; 8] = [22, 18, 15, 13, 10, 9, 7, 5];
    pub const ORB_EDGE15_PER_OCTAVE: [usize; 8] = [109, 90, 75, 63, 52, 44, 36, 31];
    pub const ORB_EDGE0_COUNT: usize = 500;
    pub const ORB_FIRST_LEVEL_2_COUNT: usize = 492;
    pub const ORB_FIRST_LEVEL_2_SIZE0: f32 = 21.52777671813965;
    pub const ORB_PATCH_21_COUNT: usize = 487;
    pub const ORB_PATCH_21_DESC_FNV: u32 = 0x8611d7c4;
    pub const ORB_WTA3_DESC_FNV: u32 = 0x96d97eec;
    pub const ORB_WTA4_DESC_FNV: u32 = 0xb60b849a;
    pub const ORB_SEVEN_KEPT: [(f32, f32, i32); 4] = [
        (31.0, 100.0, 0),
        (100.0, 100.0, 0),
        (288.5, 100.0, 0),
        (160.0, 120.0, 3),
    ];
    pub const ORB_THREE_REGROUPED_OCTAVES: [i32; 3] = [0, 3, 9];
    #[rustfmt::skip]
    pub const BF_HAMMING_MATCH: [(i32, i32, i32, f32); 6] = [(0, 7, 0, 0.0), (1, 4, 0, 116.0), (2, 4, 0, 119.0), (3, 3, 0, 117.0), (4, 4, 0, 119.0), (5, 1, 0, 110.0)];
    #[rustfmt::skip]
    pub const BF_HAMMING_KNN2: [&[(i32, i32, i32, f32)]; 6] = [&[(0, 7, 0, 0.0), (0, 4, 0, 114.0)], &[(1, 4, 0, 116.0), (1, 1, 0, 121.0)], &[(2, 4, 0, 119.0), (2, 0, 0, 122.0)], &[(3, 3, 0, 117.0), (3, 1, 0, 118.0)], &[(4, 4, 0, 119.0), (4, 0, 0, 122.0)], &[(5, 1, 0, 110.0), (5, 5, 0, 110.0)]];
    #[rustfmt::skip]
    pub const BF_HAMMING_KNN3: [&[(i32, i32, i32, f32)]; 6] = [&[(0, 7, 0, 0.0), (0, 4, 0, 114.0), (0, 3, 0, 126.0)], &[(1, 4, 0, 116.0), (1, 1, 0, 121.0), (1, 5, 0, 121.0)], &[(2, 4, 0, 119.0), (2, 0, 0, 122.0), (2, 7, 0, 127.0)], &[(3, 3, 0, 117.0), (3, 1, 0, 118.0), (3, 5, 0, 118.0)], &[(4, 4, 0, 119.0), (4, 0, 0, 122.0), (4, 7, 0, 127.0)], &[(5, 1, 0, 110.0), (5, 5, 0, 110.0), (5, 3, 0, 121.0)]];
    #[rustfmt::skip]
    pub const BF_HAMMING_KNN20: [&[(i32, i32, i32, f32)]; 6] = [&[(0, 7, 0, 0.0), (0, 4, 0, 114.0), (0, 3, 0, 126.0), (0, 0, 0, 129.0), (0, 2, 0, 130.0), (0, 6, 0, 138.0), (0, 1, 0, 139.0), (0, 5, 0, 139.0)], &[(1, 4, 0, 116.0), (1, 1, 0, 121.0), (1, 5, 0, 121.0), (1, 0, 0, 125.0), (1, 6, 0, 126.0), (1, 2, 0, 130.0), (1, 7, 0, 130.0), (1, 3, 0, 132.0)], &[(2, 4, 0, 119.0), (2, 0, 0, 122.0), (2, 7, 0, 127.0), (2, 3, 0, 131.0), (2, 2, 0, 133.0), (2, 6, 0, 135.0), (2, 1, 0, 140.0), (2, 5, 0, 140.0)], &[(3, 3, 0, 117.0), (3, 1, 0, 118.0), (3, 5, 0, 118.0), (3, 6, 0, 127.0), (3, 7, 0, 127.0), (3, 0, 0, 128.0), (3, 4, 0, 131.0), (3, 2, 0, 135.0)], &[(4, 4, 0, 119.0), (4, 0, 0, 122.0), (4, 7, 0, 127.0), (4, 3, 0, 131.0), (4, 2, 0, 133.0), (4, 6, 0, 135.0), (4, 1, 0, 140.0), (4, 5, 0, 140.0)], &[(5, 1, 0, 110.0), (5, 5, 0, 110.0), (5, 3, 0, 121.0), (5, 6, 0, 129.0), (5, 7, 0, 131.0), (5, 4, 0, 133.0), (5, 0, 0, 134.0), (5, 2, 0, 145.0)]];
    #[rustfmt::skip]
    pub const BF_HAMMING_MATCH_T2Q: [(i32, i32, i32, f32); 8] = [(0, 2, 0, 122.0), (1, 5, 0, 110.0), (2, 0, 0, 130.0), (3, 3, 0, 117.0), (4, 0, 0, 114.0), (5, 5, 0, 110.0), (6, 1, 0, 126.0), (7, 0, 0, 0.0)];
    #[rustfmt::skip]
    pub const BF_HAMMING_CROSS_CHECK: [(i32, i32, i32, f32); 3] = [(0, 7, 0, 0.0), (3, 3, 0, 117.0), (5, 1, 0, 110.0)];
    #[rustfmt::skip]
    pub const BF_HAMMING_KNN1_CROSS_CHECK: [&[(i32, i32, i32, f32)]; 6] = [&[(0, 7, 0, 0.0)], &[], &[], &[(3, 3, 0, 117.0)], &[], &[(5, 1, 0, 110.0)]];
    #[rustfmt::skip]
    pub const BF_HAMMING_DIST: [[u32; 8]; 6] = [[129, 139, 130, 126, 114, 139, 138, 0], [125, 121, 130, 132, 116, 121, 126, 130], [122, 140, 133, 131, 119, 140, 135, 127], [128, 118, 135, 117, 131, 118, 127, 127], [122, 140, 133, 131, 119, 140, 135, 127], [134, 110, 145, 121, 133, 110, 129, 131]];
    #[rustfmt::skip]
    pub const BF_L2_U8_MATCH: [(i32, i32, i32, f32); 6] = [(0, 7, 0, 0.0), (1, 4, 0, 569.7911987304688), (2, 0, 0, 591.61474609375), (3, 0, 0, 506.8382263183594), (4, 0, 0, 591.61474609375), (5, 4, 0, 465.5362548828125)];
    #[rustfmt::skip]
    pub const BF_L2_U8_KNN2: [&[(i32, i32, i32, f32)]; 6] = [&[(0, 7, 0, 0.0), (0, 3, 0, 481.6669006347656)], &[(1, 4, 0, 569.7911987304688), (1, 1, 0, 600.6887817382812)], &[(2, 0, 0, 591.61474609375), (2, 2, 0, 622.7399291992188)], &[(3, 0, 0, 506.8382263183594), (3, 1, 0, 545.2357177734375)], &[(4, 0, 0, 591.61474609375), (4, 2, 0, 622.7399291992188)], &[(5, 4, 0, 465.5362548828125), (5, 6, 0, 493.1156005859375)]];
    #[rustfmt::skip]
    pub const BF_L2_F32_MATCH: [(i32, i32, i32, f32); 5] = [(0, 0, 0, 5.5445637702941895), (1, 5, 0, 12.465890884399414), (2, 5, 0, 12.465890884399414), (3, 4, 0, 4.66787576675415), (4, 5, 0, 0.0)];
    #[rustfmt::skip]
    pub const BF_L2_F32_KNN2: [&[(i32, i32, i32, f32)]; 5] = [&[(0, 0, 0, 5.5445637702941895), (0, 3, 0, 5.5445637702941895)], &[(1, 5, 0, 12.465890884399414), (1, 1, 0, 14.888386726379395)], &[(2, 5, 0, 12.465890884399414), (2, 1, 0, 14.888386726379395)], &[(3, 4, 0, 4.66787576675415), (3, 1, 0, 8.092000007629395)], &[(4, 5, 0, 0.0), (4, 0, 0, 9.14274787902832)]];
    #[rustfmt::skip]
    pub const BF_L2_F32_KNN3: [&[(i32, i32, i32, f32)]; 5] = [&[(0, 0, 0, 5.5445637702941895), (0, 3, 0, 5.5445637702941895), (0, 1, 0, 5.887287616729736)], &[(1, 5, 0, 12.465890884399414), (1, 1, 0, 14.888386726379395), (1, 0, 0, 15.718097686767578)], &[(2, 5, 0, 12.465890884399414), (2, 1, 0, 14.888386726379395), (2, 0, 0, 15.718097686767578)], &[(3, 4, 0, 4.66787576675415), (3, 1, 0, 8.092000007629395), (3, 0, 0, 11.627351760864258)], &[(4, 5, 0, 0.0), (4, 0, 0, 9.14274787902832), (4, 3, 0, 9.14274787902832)]];
    #[rustfmt::skip]
    pub const BF_L2_F32_CROSS_CHECK: [(i32, i32, i32, f32); 3] = [(0, 0, 0, 5.5445637702941895), (3, 4, 0, 4.66787576675415), (4, 5, 0, 0.0)];
    #[rustfmt::skip]
    pub const BF_L2_F32_MATCH_T2Q: [(i32, i32, i32, f32); 6] = [(0, 0, 0, 5.5445637702941895), (1, 0, 0, 5.887287616729736), (2, 0, 0, 9.78738784790039), (3, 0, 0, 5.5445637702941895), (4, 3, 0, 4.66787576675415), (5, 4, 0, 0.0)];
    #[rustfmt::skip]
    pub const DK_GRAY_DEFAULT: [(usize, usize, [u8; 3]); 53] = [(5, 9, [67, 67, 100]), (5, 10, [103, 103, 231]), (5, 11, [56, 56, 170]), (5, 12, [37, 37, 59]), (6, 8, [57, 57, 124]), (6, 9, [16, 16, 244]), (6, 10, [29, 29, 232]), (6, 11, [39, 39, 255]), (6, 12, [5, 5, 226]), (6, 13, [61, 61, 84]), (7, 7, [40, 40, 138]), (7, 8, [24, 24, 253]), (7, 9, [107, 107, 170]), (7, 10, [178, 178, 194]), (7, 11, [181, 181, 200]), (7, 12, [28, 28, 141]), (7, 13, [10, 10, 240]), (7, 14, [79, 79, 123]), (8, 6, [193, 193, 205]), (8, 7, [34, 34, 210]), (8, 8, [87, 87, 254]), (8, 9, [231, 231, 235]), (8, 12, [159, 159, 178]), (8, 13, [41, 41, 245]), (8, 14, [109, 109, 239]), (8, 15, [19, 19, 21]), (9, 6, [224, 224, 234]), (9, 7, [52, 52, 232]), (9, 8, [37, 37, 189]), (9, 12, [144, 144, 158]), (9, 13, [16, 16, 212]), (9, 14, [18, 18, 147]), (10, 7, [85, 85, 223]), (10, 8, [25, 25, 233]), (10, 9, [26, 26, 52]), (10, 11, [23, 23, 27]), (10, 12, [101, 101, 159]), (10, 13, [17, 17, 241]), (10, 14, [134, 134, 209]), (11, 7, [50, 50, 54]), (11, 8, [52, 52, 187]), (11, 9, [34, 34, 253]), (11, 10, [83, 83, 237]), (11, 11, [45, 45, 222]), (11, 12, [17, 17, 245]), (11, 13, [57, 57, 137]), (12, 8, [86, 86, 88]), (12, 9, [69, 69, 156]), (12, 10, [26, 26, 219]), (12, 11, [66, 66, 251]), (12, 12, [59, 59, 118]), (13, 10, [182, 182, 191]), (13, 11, [197, 197, 201])];
    #[rustfmt::skip]
    pub const DK_BGR_DEFAULT: [(usize, usize, [u8; 3]); 48] = [(4, 9, [169, 76, 238]), (4, 10, [42, 107, 226]), (4, 11, [4, 63, 136]), (5, 8, [4, 46, 164]), (5, 9, [9, 20, 249]), (5, 10, [10, 18, 245]), (5, 11, [24, 32, 227]), (5, 12, [95, 172, 153]), (6, 7, [19, 79, 238]), (6, 8, [11, 1, 247]), (6, 9, [53, 123, 157]), (6, 10, [197, 145, 151]), (6, 11, [25, 128, 202]), (6, 12, [29, 22, 241]), (6, 13, [38, 83, 230]), (7, 6, [119, 86, 97]), (7, 7, [23, 24, 239]), (7, 8, [65, 11, 211]), (7, 12, [42, 43, 145]), (7, 13, [26, 23, 234]), (7, 14, [13, 91, 212]), (8, 6, [107, 22, 189]), (8, 7, [35, 14, 250]), (8, 8, [97, 87, 135]), (8, 12, [120, 58, 251]), (8, 13, [26, 30, 245]), (8, 14, [56, 86, 168]), (9, 6, [183, 93, 46]), (9, 7, [37, 2, 222]), (9, 8, [105, 125, 249]), (9, 12, [60, 102, 248]), (9, 13, [23, 24, 240]), (9, 14, [46, 25, 45]), (10, 7, [13, 142, 220]), (10, 8, [27, 22, 236]), (10, 9, [130, 83, 147]), (10, 10, [62, 179, 149]), (10, 11, [73, 108, 115]), (10, 12, [12, 25, 226]), (10, 13, [41, 69, 197]), (11, 8, [123, 19, 171]), (11, 9, [4, 32, 238]), (11, 10, [13, 10, 224]), (11, 11, [16, 15, 236]), (11, 12, [40, 103, 248]), (12, 9, [73, 21, 227]), (12, 10, [33, 148, 117]), (12, 11, [58, 66, 88])];
    #[rustfmt::skip]
    pub const DM_GRAY_ONE_MATCH: [(usize, usize, [u8; 3]); 223] = [(0, 12, [79, 153, 79]), (0, 13, [16, 234, 16]), (0, 14, [24, 241, 24]), (0, 15, [10, 231, 10]), (0, 16, [91, 170, 91]), (0, 37, [19, 91, 19]), (0, 38, [102, 245, 102]), (0, 39, [183, 197, 183]), (0, 41, [196, 209, 196]), (0, 42, [28, 171, 28]), (0, 43, [178, 250, 178]), (1, 11, [144, 245, 144]), (1, 12, [16, 239, 16]), (1, 13, [43, 153, 43]), (1, 14, [51, 107, 51]), (1, 15, [97, 211, 97]), (1, 16, [9, 234, 9]), (1, 17, [14, 115, 14]), (1, 36, [67, 110, 67]), (1, 37, [14, 228, 14]), (1, 38, [108, 207, 108]), (1, 42, [115, 215, 115]), (1, 43, [1, 215, 1]), (2, 4, [215, 215, 201]), (2, 5, [73, 60, 20]), (2, 6, [93, 90, 79]), (2, 10, [202, 244, 202]), (2, 11, [1, 215, 1]), (2, 12, [74, 173, 74]), (2, 16, [100, 199, 100]), (2, 17, [11, 226, 11]), (2, 18, [131, 174, 131]), (2, 36, [70, 124, 70]), (2, 37, [22, 234, 22]), (2, 38, [14, 70, 14]), (2, 42, [174, 229, 174]), (2, 43, [28, 241, 28]), (3, 3, [123, 100, 48]), (3, 4, [217, 32, 1]), (3, 5, [222, 37, 5]), (3, 6, [223, 31, 1]), (3, 7, [139, 115, 59]), (3, 10, [38, 91, 38]), (3, 11, [6, 219, 6]), (3, 12, [57, 113, 57]), (3, 16, [136, 190, 136]), (3, 17, [39, 251, 39]), (3, 18, [60, 114, 60]), (3, 36, [156, 195, 156]), (3, 37, [30, 245, 30]), (3, 38, [24, 128, 24]), (3, 42, [95, 196, 95]), (3, 43, [12, 226, 12]), (4, 2, [177, 139, 78]), (4, 3, [226, 29, 2]), (4, 4, [185, 137, 74]), (4, 5, [153, 141, 97]), (4, 6, [190, 137, 75]), (4, 7, [225, 27, 0]), (4, 8, [111, 71, 11]), (4, 10, [126, 166, 126]), (4, 11, [6, 220, 6]), (4, 12, [13, 116, 13]), (4, 16, [21, 125, 21]), (4, 17, [3, 218, 3]), (4, 18, [171, 210, 171]), (4, 37, [56, 158, 56]), (4, 38, [31, 252, 31]), (4, 39, [16, 126, 16]), (4, 40, [60, 114, 60]), (4, 41, [9, 123, 9]), (4, 42, [1, 223, 1]), (4, 43, [44, 146, 44]), (5, 1, [68, 62, 25]), (5, 2, [218, 37, 3]), (5, 3, [100, 61, 2]), (5, 5, [222, 196, 196]), (5, 6, [130, 97, 97]), (5, 7, [152, 97, 41]), (5, 8, [219, 34, 3]), (5, 9, [213, 205, 170]), (5, 11, [105, 206, 105]), (5, 12, [15, 236, 15]), (5, 13, [81, 191, 81]), (5, 14, [192, 247, 192]), (5, 15, [52, 167, 52]), (5, 16, [24, 246, 24]), (5, 17, [3, 103, 3]), (5, 26, [196, 194, 182]), (5, 27, [115, 105, 61]), (5, 28, [156, 156, 142]), (5, 38, [5, 79, 5]), (5, 39, [8, 226, 8]), (5, 40, [38, 254, 38]), (5, 41, [30, 251, 30]), (5, 42, [143, 224, 143]), (6, 1, [138, 127, 84]), (6, 2, [216, 40, 4]), (6, 3, [82, 69, 25]), (6, 5, [236, 92, 92]), (6, 6, [252, 25, 25]), (6, 7, [246, 25, 19]), (6, 8, [245, 9, 0]), (6, 9, [196, 25, 11]), (6, 10, [220, 85, 85]), (6, 11, [101, 6, 6]), (6, 12, [188, 180, 123]), (6, 13, [43, 182, 1]), (6, 14, [36, 214, 14]), (6, 15, [27, 226, 16]), (6, 16, [151, 223, 145]), (6, 25, [186, 165, 112]), (6, 26, [220, 35, 3]), (6, 27, [219, 34, 2]), (6, 28, [221, 30, 1]), (6, 29, [153, 129, 73]), (6, 39, [5, 19, 5]), (6, 40, [8, 62, 8]), (6, 41, [80, 94, 80]), (7, 1, [200, 193, 160]), (7, 2, [215, 34, 1]), (7, 3, [163, 119, 59]), (7, 5, [238, 209, 209]), (7, 6, [147, 62, 62]), (7, 7, [220, 78, 46]), (7, 8, [240, 13, 1]), (7, 9, [209, 19, 11]), (7, 10, [240, 30, 30]), (7, 11, [253, 26, 26]), (7, 12, [240, 8, 8]), (7, 13, [235, 4, 3]), (7, 14, [233, 18, 11]), (7, 15, [223, 20, 17]), (7, 16, [237, 56, 56]), (7, 17, [167, 25, 25]), (7, 18, [183, 73, 73]), (7, 19, [247, 175, 175]), (7, 20, [109, 62, 62]), (7, 21, [75, 46, 46]), (7, 22, [19, 5, 5]), (7, 23, [28, 22, 22]), (7, 24, [100, 60, 0]), (7, 25, [225, 29, 2]), (7, 26, [182, 135, 72]), (7, 27, [195, 182, 139]), (7, 28, [159, 108, 45]), (7, 29, [225, 27, 0]), (7, 30, [142, 102, 42]), (8, 2, [151, 110, 50]), (8, 3, [225, 34, 3]), (8, 4, [151, 104, 40]), (8, 5, [121, 110, 66]), (8, 6, [122, 70, 7]), (8, 7, [225, 32, 3]), (8, 8, [164, 123, 62]), (8, 9, [113, 107, 107]), (8, 10, [244, 231, 231]), (8, 11, [36, 8, 8]), (8, 12, [226, 179, 179]), (8, 13, [176, 104, 104]), (8, 14, [114, 4, 4]), (8, 15, [240, 97, 97]), (8, 16, [249, 75, 75]), (8, 17, [232, 26, 26]), (8, 18, [235, 13, 13]), (8, 19, [249, 17, 17]), (8, 20, [236, 3, 3]), (8, 21, [240, 13, 13]), (8, 22, [230, 19, 19]), (8, 23, [223, 33, 24]), (8, 24, [240, 13, 1]), (8, 25, [189, 50, 17]), (8, 26, [228, 141, 141]), (8, 27, [191, 160, 160]), (8, 29, [135, 96, 34]), (8, 30, [220, 40, 5]), (8, 31, [107, 100, 64]), (9, 3, [114, 91, 37]), (9, 4, [219, 34, 1]), (9, 5, [219, 36, 3]), (9, 6, [225, 34, 5]), (9, 7, [137, 111, 57]), (9, 17, [167, 155, 155]), (9, 18, [83, 61, 61]), (9, 19, [79, 37, 37]), (9, 20, [220, 155, 155]), (9, 21, [104, 10, 10]), (9, 22, [237, 100, 100]), (9, 23, [238, 68, 54]), (9, 24, [245, 9, 0]), (9, 25, [234, 14, 8]), (9, 26, [238, 10, 10]), (9, 27, [180, 30, 30]), (9, 28, [43, 41, 41]), (9, 29, [184, 172, 129]), (9, 30, [212, 36, 1]), (9, 31, [91, 80, 37]), (10, 4, [132, 131, 119]), (10, 5, [120, 109, 66]), (10, 6, [174, 174, 160]), (10, 23, [129, 122, 89]), (10, 24, [217, 32, 1]), (10, 25, [115, 59, 2]), (10, 26, [105, 72, 72]), (10, 27, [95, 68, 68]), (10, 29, [136, 94, 32]), (10, 30, [219, 39, 5]), (10, 31, [74, 68, 35]), (11, 24, [129, 90, 28]), (11, 25, [223, 32, 3]), (11, 26, [114, 66, 4]), (11, 27, [149, 138, 95]), (11, 28, [141, 91, 28]), (11, 29, [222, 30, 0]), (11, 30, [170, 129, 69]), (12, 25, [154, 131, 78]), (12, 26, [223, 37, 5]), (12, 27, [217, 33, 1]), (12, 28, [220, 30, 1]), (12, 29, [198, 172, 118]), (13, 26, [96, 93, 83]), (13, 27, [56, 44, 2]), (13, 28, [135, 135, 122])];
    #[rustfmt::skip]
    pub const KP_OVERLAP: [(&str, f32); 9] = [("same", 1.0), ("disjoint", 0.0), ("touching", 0.0), ("enclosed", 0.03999999910593033), ("enclosed_concentric_zero_sizes", f32::NAN), ("partial_1", 0.4533168375492096), ("partial_2", 0.3503645658493042), ("partial_3", 0.3221896290779114), ("size_zero_vs_positive_overlapping_centre", 0.0)];
}

/*
 *  parity_calib3d.rs
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

//! OpenCV 4.10.0 parity pins for the `calib3d` module (`docs/parity/calib3d.md`).
//!
//! Every expected value in this file was produced by a script under `scripts/opencv_ref/`
//! (`calib3d_homography.py`, `calib3d_rodrigues.py`, `calib3d_pnp.py`, `calib3d_undistort.py`,
//! `calib3d_fundamental.py`; helper `_calib3d_common.py`) run in `cpp_ref/.venv`
//! (opencv-contrib-python 4.10.0.84, `cv2.ipp.setUseIPP(False)`, Windows x86-64). The script
//! and the key of each block are named in the comment above it. Every input is rebuilt here
//! from the same integer LCG draws, dyadic fractions, rational rotations and fixed-order double
//! arithmetic the scripts use (`inputs_match_the_scripts` checks the digests), so no data file
//! is committed; the one input OpenCV computed itself (the distorted projections of
//! `IMG_24_DIST5`) is embedded.
//!
//! Tests that pass pin rows that already match OpenCV. Tests marked `#[ignore]` document a
//! `diverges` row; they fail when run (`cargo test --test parity_calib3d -- --ignored`) and are
//! the executable form of the measurement; the issue in each reason tracks the fix. RANSAC
//! rows compare inlier sets and model quality, never sample sequences: OpenCV draws with
//! `cv::RNG` (#225) and purecv with its own generator.

use purecv::calib3d::*;
use purecv::core::types::{Point2f, Point3f, Size2i};
use purecv::core::Matrix;

use ocv::*;

// ---------------------------------------------------------------------------------------
// Inputs (the same formulas as `_calib3d_common.py`)
// ---------------------------------------------------------------------------------------

/// The 32-bit LCG `state * 1664525 + 1013904223 (mod 2^32)`; `draw` returns bits 16..31.
struct Lcg(u32);

impl Lcg {
    fn new(seed: u32) -> Self {
        Lcg(seed)
    }
    fn draw(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(1664525).wrapping_add(1013904223);
        self.0 >> 16
    }
    /// A dyadic fraction in `[0, 1)`: `(draw % denom) / denom`.
    fn frac(&mut self, denom: u32) -> f64 {
        (self.draw() % denom) as f64 / denom as f64
    }
    /// A dyadic fraction in `[-1, 1]`: `(draw % 2049 - 1024) / 1024`.
    fn signed(&mut self) -> f64 {
        ((self.draw() % 2049) as f64 - 1024.0) / 1024.0
    }
}

const FX: f64 = 800.0;
const FY: f64 = 800.0;
const CX: f64 = 320.0;
const CY: f64 = 240.0;

/// Row-major 3x3 product with the scripts' accumulation order.
fn mat3_mul(a: &[f64; 9], b: &[f64; 9]) -> [f64; 9] {
    let mut c = [0.0; 9];
    for i in 0..3 {
        for j in 0..3 {
            let mut s = 0.0;
            for k in 0..3 {
                s += a[i * 3 + k] * b[k * 3 + j];
            }
            c[i * 3 + j] = s;
        }
    }
    c
}

/// `Rz(c1, s1) * Ry(c3, s3) * Rx(c2, s2)` from Pythagorean cosines and sines.
fn rational_rotation(c1: f64, s1: f64, c2: f64, s2: f64, c3: f64, s3: f64) -> [f64; 9] {
    let rz = [c1, -s1, 0.0, s1, c1, 0.0, 0.0, 0.0, 1.0];
    let rx = [1.0, 0.0, 0.0, 0.0, c2, -s2, 0.0, s2, c2];
    let ry = [c3, 0.0, s3, 0.0, 1.0, 0.0, -s3, 0.0, c3];
    mat3_mul(&mat3_mul(&rz, &ry), &rx)
}

/// The object pose of the PnP rows (`pose_a`: 5-12-13, 3-4-5 and 8-15-17 triangles).
fn pose_a() -> ([f64; 9], [f64; 3]) {
    (
        rational_rotation(
            12.0 / 13.0,
            5.0 / 13.0,
            4.0 / 5.0,
            3.0 / 5.0,
            15.0 / 17.0,
            8.0 / 17.0,
        ),
        [0.25, -0.125, 4.0],
    )
}

/// The second camera of the fundamental-matrix rows (`pose_b`).
fn pose_b() -> ([f64; 9], [f64; 3]) {
    (
        rational_rotation(
            24.0 / 25.0,
            7.0 / 25.0,
            40.0 / 41.0,
            9.0 / 41.0,
            12.0 / 13.0,
            5.0 / 13.0,
        ),
        [-0.75, 0.125, 0.25],
    )
}

/// Pinhole projection `(fx * xc / zc + cx, fy * yc / zc + cy)` in double, no distortion.
fn project(r: &[f64; 9], t: &[f64; 3], p: (f64, f64, f64)) -> (f64, f64) {
    let (x, y, z) = p;
    let xc = r[0] * x + r[1] * y + r[2] * z + t[0];
    let yc = r[3] * x + r[4] * y + r[5] * z + t[1];
    let zc = r[6] * x + r[7] * y + r[8] * z + t[2];
    (FX * (xc / zc) + CX, FY * (yc / zc) + CY)
}

fn project_all(r: &[f64; 9], t: &[f64; 3], obj: &[Point3f]) -> Vec<Point2f> {
    obj.iter()
        .map(|p| {
            let (u, v) = project(r, t, (p.x as f64, p.y as f64, p.z as f64));
            Point2f::new(u as f32, v as f32)
        })
        .collect()
}

fn apply_homography(h: &[f64; 9], x: f64, y: f64) -> (f64, f64) {
    let w = h[6] * x + h[7] * y + h[8];
    (
        (h[0] * x + h[1] * y + h[2]) / w,
        (h[3] * x + h[4] * y + h[5]) / w,
    )
}

/// `cols x rows` grid points with a `k / 64` jitter per coordinate (`grid_jittered`).
fn grid_jittered(
    cols: usize,
    rows: usize,
    x0: f64,
    y0: f64,
    sx: f64,
    sy: f64,
    seed: u32,
) -> Vec<Point2f> {
    let mut rng = Lcg::new(seed);
    let mut out = Vec::new();
    for j in 0..rows {
        for i in 0..cols {
            let x = x0 + sx * i as f64 + rng.frac(64);
            let y = y0 + sy * j as f64 + rng.frac(64);
            out.push(Point2f::new(x as f32, y as f32));
        }
    }
    out
}

/// `n` object points with `k / 1024` coordinates in `[-1, 1]`, `z` in `[-0.5, 0.5]` or 0.
fn object_points(n: usize, seed: u32, planar: bool) -> Vec<Point3f> {
    let mut rng = Lcg::new(seed);
    (0..n)
        .map(|_| {
            let x = rng.signed();
            let y = rng.signed();
            let z = if planar { 0.0 } else { rng.signed() * 0.5 };
            Point3f::new(x as f32, y as f32, z as f32)
        })
        .collect()
}

/// `n` scene points with `x`, `y` in `[-1, 1]` and `z` in `[4, 6)` (`scene_points`).
fn scene_points(n: usize, seed: u32) -> Vec<(f64, f64, f64)> {
    let mut rng = Lcg::new(seed);
    (0..n)
        .map(|_| {
            let x = rng.signed();
            let y = rng.signed();
            let z = 4.0 + rng.frac(1024) * 2.0;
            (x, y, z)
        })
        .collect()
}

/// `f32(f64(p) + amplitude * signed)` per coordinate (`add_noise`).
fn add_noise(pts: &[Point2f], amp: f64, seed: u32) -> Vec<Point2f> {
    let mut rng = Lcg::new(seed);
    pts.iter()
        .map(|p| {
            let nx = amp * rng.signed();
            let ny = amp * rng.signed();
            Point2f::new((p.x as f64 + nx) as f32, (p.y as f64 + ny) as f32)
        })
        .collect()
}

/// Replace `count` distinct LCG-chosen points by integer pixels (`replace_outliers`).
fn replace_outliers(pts: &[Point2f], count: usize, seed: u32) -> Vec<Point2f> {
    let mut rng = Lcg::new(seed);
    let mut out = pts.to_vec();
    let mut chosen: Vec<usize> = Vec::new();
    while chosen.len() < count {
        let i = (rng.draw() as usize) % pts.len();
        if !chosen.contains(&i) {
            chosen.push(i);
            let x = (rng.draw() % 640) as f32;
            let y = (rng.draw() % 480) as f32;
            out[i] = Point2f::new(x, y);
        }
    }
    out
}

fn fnv1a32(bytes: &[u8]) -> u32 {
    let mut h = 0x811c_9dc5u32;
    for &b in bytes {
        h ^= b as u32;
        h = h.wrapping_mul(0x0100_0193);
    }
    h
}

/// FNV-1a of the little-endian `f32` field bytes (`points_f32_digest`).
fn digest2(pts: &[Point2f]) -> u32 {
    let mut b = Vec::new();
    for p in pts {
        b.extend_from_slice(&p.x.to_le_bytes());
        b.extend_from_slice(&p.y.to_le_bytes());
    }
    fnv1a32(&b)
}

fn digest3(pts: &[Point3f]) -> u32 {
    let mut b = Vec::new();
    for p in pts {
        b.extend_from_slice(&p.x.to_le_bytes());
        b.extend_from_slice(&p.y.to_le_bytes());
        b.extend_from_slice(&p.z.to_le_bytes());
    }
    fnv1a32(&b)
}

fn map_digest(m: &Matrix<f32>) -> u32 {
    let b: Vec<u8> = m.data.iter().flat_map(|v| v.to_le_bytes()).collect();
    fnv1a32(&b)
}

fn mat3(v: [f64; 9]) -> Matrix<f64> {
    Matrix::from_vec(3, 3, 1, v.to_vec())
}

fn vec3(v: [f64; 3]) -> Matrix<f64> {
    Matrix::from_vec(3, 1, 1, v.to_vec())
}

fn camera_matrix() -> Matrix<f64> {
    mat3([FX, 0.0, CX, 0.0, FY, CY, 0.0, 0.0, 1.0])
}

const H_TRUE: [f64; 9] = [1.1, 0.08, 20.0, -0.05, 0.95, -12.0, 0.0002, -0.0001, 1.0];
const DIST5: [f64; 5] = [-0.25, 0.08, 0.001, -0.0005, 0.02];
const NO_DIST: [f64; 5] = [0.0; 5];

/// The homography fixtures: `(src, dst_exact, dst_noisy, dst_outliers)`.
fn homography_inputs() -> (Vec<Point2f>, Vec<Point2f>, Vec<Point2f>, Vec<Point2f>) {
    let src = grid_jittered(8, 5, 40.0, 40.0, 70.0, 60.0, 0x1234);
    let exact: Vec<Point2f> = src
        .iter()
        .map(|p| {
            let (u, v) = apply_homography(&H_TRUE, p.x as f64, p.y as f64);
            Point2f::new(u as f32, v as f32)
        })
        .collect();
    let noisy = add_noise(&exact, 0.5, 0x5678);
    let outl = replace_outliers(&noisy, 10, 0x9ABC);
    (src, exact, noisy, outl)
}

/// The PnP fixtures: 24 non-planar points with exact, noisy and outlier projections.
struct PnpInputs {
    obj: Vec<Point3f>,
    img: Vec<Point2f>,
    noisy: Vec<Point2f>,
    outliers: Vec<Point2f>,
    planar: Vec<Point3f>,
    img_planar: Vec<Point2f>,
    noisy_planar: Vec<Point2f>,
}

fn pnp_inputs() -> PnpInputs {
    let (r, t) = pose_a();
    let obj = object_points(24, 0x2468, false);
    let planar = object_points(16, 0x2468, true);
    let img = project_all(&r, &t, &obj);
    let img_planar = project_all(&r, &t, &planar);
    let noisy = add_noise(&img, 0.5, 0x1357);
    let noisy_planar = add_noise(&img_planar, 0.5, 0x1357);
    let outliers = replace_outliers(&noisy, 6, 0x2469);
    PnpInputs {
        obj,
        img,
        noisy,
        outliers,
        planar,
        img_planar,
        noisy_planar,
    }
}

/// The unit square of the `SOLVEPNP_IPPE_SQUARE` case and its projection.
fn square_inputs() -> (Vec<Point3f>, Vec<Point2f>) {
    let (r, t) = pose_a();
    let square = vec![
        Point3f::new(-0.5, 0.5, 0.0),
        Point3f::new(0.5, 0.5, 0.0),
        Point3f::new(0.5, -0.5, 0.0),
        Point3f::new(-0.5, -0.5, 0.0),
    ];
    let img = project_all(&r, &t, &square);
    (square, img)
}

/// `(p1_exact, p2_exact, p1_noisy, p2_noisy, p2_outliers)`.
type FundamentalInputs = (
    Vec<Point2f>,
    Vec<Point2f>,
    Vec<Point2f>,
    Vec<Point2f>,
    Vec<Point2f>,
);

/// The fundamental-matrix fixtures.
fn fundamental_inputs() -> FundamentalInputs {
    let (r2, t2) = pose_b();
    let ident = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
    let scene = scene_points(60, 0x3579);
    let to_pt = |(u, v): (f64, f64)| Point2f::new(u as f32, v as f32);
    let p1: Vec<Point2f> = scene
        .iter()
        .map(|&p| to_pt(project(&ident, &[0.0; 3], p)))
        .collect();
    let p2: Vec<Point2f> = scene.iter().map(|&p| to_pt(project(&r2, &t2, p))).collect();
    let n1 = add_noise(&p1, 0.5, 0x4680);
    let n2 = add_noise(&p2, 0.5, 0x4681);
    let o2 = replace_outliers(&n2, 12, 0x5791);
    (p1, p2, n1, n2, o2)
}

// ---------------------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------------------

fn assert_close(actual: &[f64], expected: &[f64], tol: f64, what: &str) {
    assert_eq!(actual.len(), expected.len(), "{what}: length");
    for (i, (a, e)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (a - e).abs() <= tol,
            "{what}[{i}]: purecv {a} vs OpenCV {e} (tol {tol})"
        );
    }
}

/// `h / h[2][2]`, as `norm_h` in the scripts.
fn norm_h(h: &Matrix<f64>) -> Vec<f64> {
    assert_eq!((h.rows, h.cols, h.channels), (3, 3, 1));
    let s = h.data[8];
    assert!(s != 0.0, "h[2][2] is zero");
    h.data.iter().map(|v| v / s).collect()
}

/// Unit Frobenius norm with the largest-magnitude entry positive, as `norm_f`.
fn norm_f(f: &Matrix<f64>) -> Vec<f64> {
    assert_eq!((f.rows, f.cols, f.channels), (3, 3, 1));
    let n: f64 = f.data.iter().map(|v| v * v).sum::<f64>().sqrt();
    let mut a: Vec<f64> = f.data.iter().map(|v| v / n).collect();
    let k = (0..9)
        .max_by(|&i, &j| a[i].abs().partial_cmp(&a[j].abs()).unwrap())
        .unwrap();
    if a[k] < 0.0 {
        a.iter_mut().for_each(|v| *v = -*v);
    }
    a
}

fn homography(
    src: &[Point2f],
    dst: &[Point2f],
    m: HomographyMethod,
    thr: f64,
) -> (Matrix<f64>, Vec<u8>) {
    let mut mask = Vec::new();
    let h = find_homography(src, dst, m, thr, Some(&mut mask)).expect("find_homography");
    (h, mask)
}

fn fundamental(
    a: &[Point2f],
    b: &[Point2f],
    m: FundamentalMatMethod,
    thr: f64,
    conf: f64,
    iters: usize,
) -> (Matrix<f64>, Vec<u8>) {
    let mut mask = Vec::new();
    let f = find_fundamental_mat(a, b, m, thr, conf, iters, Some(&mut mask))
        .expect("find_fundamental_mat");
    (f, mask)
}

/// `solve_pnp` with an optional extrinsic guess; returns `(rvec, tvec)`.
fn pnp(
    obj: &[Point3f],
    img: &[Point2f],
    dist: &[f64],
    guess: Option<([f64; 3], [f64; 3])>,
    flags: SolvePnPMethod,
) -> purecv::core::error::Result<(Vec<f64>, Vec<f64>)> {
    let mut rvec = Matrix::<f64>::new(3, 1, 1);
    let mut tvec = Matrix::<f64>::new(3, 1, 1);
    if let Some((r, t)) = guess {
        rvec.data = r.to_vec();
        tvec.data = t.to_vec();
    }
    let ok = solve_pnp(
        obj,
        img,
        &camera_matrix(),
        Some(dist),
        &mut rvec,
        &mut tvec,
        guess.is_some(),
        flags,
    )?;
    assert!(ok, "solve_pnp returned false");
    Ok((rvec.data, tvec.data))
}

type RansacPose = purecv::core::error::Result<(bool, Vec<f64>, Vec<f64>, Vec<i32>)>;

// Mirrors the library function's own argument list (which carries the same allow).
#[allow(clippy::too_many_arguments)]
fn pnp_ransac(
    obj: &[Point3f],
    img: &[Point2f],
    dist: &[f64],
    guess: Option<([f64; 3], [f64; 3])>,
    iters: i32,
    err: f32,
    conf: f64,
    flags: SolvePnPMethod,
) -> RansacPose {
    let mut rvec = Matrix::<f64>::new(3, 1, 1);
    let mut tvec = Matrix::<f64>::new(3, 1, 1);
    if let Some((r, t)) = guess {
        rvec.data = r.to_vec();
        tvec.data = t.to_vec();
    }
    let mut inliers = Vec::new();
    let ok = solve_pnp_ransac(
        obj,
        img,
        &camera_matrix(),
        Some(dist),
        &mut rvec,
        &mut tvec,
        guess.is_some(),
        iters,
        err,
        conf,
        Some(&mut inliers),
        flags,
    )?;
    Ok((ok, rvec.data, tvec.data, inliers))
}

fn assert_pose(
    actual: &(Vec<f64>, Vec<f64>),
    expected: &([f64; 3], [f64; 3]),
    tol: f64,
    what: &str,
) {
    assert_close(&actual.0, &expected.0, tol, &format!("{what}: rvec"));
    assert_close(&actual.1, &expected.1, tol, &format!("{what}: tvec"));
}

fn rod(src: &Matrix<f64>) -> Matrix<f64> {
    let mut dst = Matrix::<f64>::new(1, 1, 1);
    rodrigues(src, &mut dst).expect("rodrigues");
    dst
}

fn undistort_maps(
    k: &Matrix<f64>,
    dist: &[f64],
    r: Option<&Matrix<f64>>,
    new_k: &Matrix<f64>,
    w: i32,
    h: i32,
) -> purecv::core::error::Result<(Matrix<f32>, Matrix<f32>)> {
    let d = if dist.is_empty() {
        Matrix::<f64>::new(1, 0, 1)
    } else {
        Matrix::from_vec(1, dist.len(), 1, dist.to_vec())
    };
    init_undistort_rectify_map(k, &d, r, new_k, Size2i::new(w, h))
}

fn new_camera_matrix() -> Matrix<f64> {
    mat3([700.0, 0.0, 300.0, 0.0, 720.0, 250.0, 0.0, 0.0, 1.0])
}

fn r_small() -> Matrix<f64> {
    mat3(rational_rotation(
        24.0 / 25.0,
        7.0 / 25.0,
        1.0,
        0.0,
        1.0,
        0.0,
    ))
}

const K4: [f64; 4] = [-0.25, 0.08, 0.001, -0.0005];
const K5: [f64; 5] = [-0.25, 0.08, 0.001, -0.0005, 0.02];
const K8: [f64; 8] = [-0.25, 0.08, 0.001, -0.0005, 0.02, 0.1, -0.03, 0.005];
const K12: [f64; 12] = [
    -0.25, 0.08, 0.001, -0.0005, 0.02, 0.1, -0.03, 0.005, 0.001, -0.0005, 0.0008, 0.0002,
];
const K14: [f64; 14] = [
    -0.25, 0.08, 0.001, -0.0005, 0.02, 0.1, -0.03, 0.005, 0.001, -0.0005, 0.0008, 0.0002, 0.01,
    -0.02,
];
const K8_STRONG: [f64; 8] = [-0.5, 0.3, 0.02, -0.01, 0.1, 0.2, -0.1, 0.05];

/// Looks up a `UNDISTORT_MAPS` entry by its `block/name` key.
fn expected_maps(key: &str) -> &'static MapCase {
    UNDISTORT_MAPS
        .iter()
        .find(|e| e.0 == key)
        .unwrap_or_else(|| panic!("{key}"))
}

fn assert_maps_identical(actual: &(Matrix<f32>, Matrix<f32>), key: &str) {
    let e = expected_maps(key);
    assert_eq!((actual.0.rows, actual.0.cols), (48, 64), "{key}: shape");
    assert_eq!(map_digest(&actual.0), e.1, "{key}: map1 digest");
    assert_eq!(map_digest(&actual.1), e.2, "{key}: map2 digest");
}

fn assert_maps_samples(actual: &(Matrix<f32>, Matrix<f32>), key: &str, tol: f32) {
    let e = expected_maps(key);
    for &(x, y, m1, m2) in &e.3 {
        let i = y as usize * 64 + x as usize;
        assert!(
            (actual.0.data[i] - m1).abs() <= tol,
            "{key}: map1({x}, {y}) {} vs {m1}",
            actual.0.data[i]
        );
        assert!(
            (actual.1.data[i] - m2).abs() <= tol,
            "{key}: map2({x}, {y}) {} vs {m2}",
            actual.1.data[i]
        );
    }
}

// ---------------------------------------------------------------------------------------
// Inputs
// ---------------------------------------------------------------------------------------

/// The fixtures rebuilt here have the digests the scripts print (`inputs` of each script).
#[test]
fn inputs_match_the_scripts() {
    let (src, exact, noisy, outl) = homography_inputs();
    let got = [
        ("src", digest2(&src)),
        ("dst_exact", digest2(&exact)),
        ("dst_noisy", digest2(&noisy)),
        ("dst_outliers", digest2(&outl)),
    ];
    assert_eq!(got, H_INPUT_DIGESTS);

    let p = pnp_inputs();
    let got = [
        ("obj_24", digest3(&p.obj)),
        ("img_24_exact", digest2(&p.img)),
        ("img_24_noisy", digest2(&p.noisy)),
        ("img_24_outliers", digest2(&p.outliers)),
        ("obj_16_planar", digest3(&p.planar)),
        ("img_16_planar_exact", digest2(&p.img_planar)),
        ("img_16_planar_noisy", digest2(&p.noisy_planar)),
    ];
    assert_eq!(got, PNP_INPUT_DIGESTS);

    let (p1, p2, n1, n2, o2) = fundamental_inputs();
    let got = [
        ("p1_exact", digest2(&p1)),
        ("p2_exact", digest2(&p2)),
        ("p1_noisy", digest2(&n1)),
        ("p2_noisy", digest2(&n2)),
        ("p2_outliers", digest2(&o2)),
    ];
    assert_eq!(got, F_INPUT_DIGESTS);

    // the rotation vector OpenCV recovers from `pose_a` is the one the PnP rows compare with
    let (r, _) = pose_a();
    assert_close(&rod(&mat3(r)).data, &RVEC_TRUE, 1e-14, "rvec_true");
}

// ---------------------------------------------------------------------------------------
// Enums
// ---------------------------------------------------------------------------------------

/// `HomographyMethod`, `FundamentalMatMethod` and the implemented `SolvePnPMethod` variants
/// carry OpenCV's values (`defaults` of the scripts).
#[test]
fn method_discriminants_match_opencv() {
    assert_eq!(HomographyMethod::None as i32, 0);
    assert_eq!(HomographyMethod::LMedS as i32, 4);
    assert_eq!(HomographyMethod::Ransac as i32, 8);
    assert_eq!(HomographyMethod::Rho as i32, 16);
    assert_eq!(FundamentalMatMethod::FM_8POINT as i32, 2);
    assert_eq!(FundamentalMatMethod::FM_RANSAC as i32, 8);
    assert_eq!(SolvePnPMethod::Iterative as i32, 0);
    assert_eq!(SolvePnPMethod::EPnP as i32, 1);
    assert_eq!(SolvePnPMethod::P3P as i32, 2);
    assert_eq!(SolvePnPMethod::AP3P as i32, 5);
}

/// OpenCV: `SOLVEPNP_IPPE = 6`, `SOLVEPNP_IPPE_SQUARE = 7`, `SOLVEPNP_SQPNP = 8`.
#[test]
#[ignore = "diverges from OpenCV 4.10.0, see #262"]
fn solve_pnp_method_sqpnp_discriminant_matches_opencv() {
    assert_eq!(SolvePnPMethod::SqPnP as i32, 8);
}

// ---------------------------------------------------------------------------------------
// find_homography (calib3d_homography.py)
// ---------------------------------------------------------------------------------------

/// `exact`: 40 exact correspondences with methods 0, RANSAC and LMedS, thresholds 3 and 1;
/// `d2`: four points (OpenCV runs the kernel directly) and the identity. RHO is in
/// `find_homography_rho_is_distinct_from_ransac`: purecv runs RANSAC for it.
#[test]
fn find_homography_exact_matches_opencv() {
    let (src, exact, _, _) = homography_inputs();
    for (name, m) in [
        ("method0", HomographyMethod::None),
        ("ransac", HomographyMethod::Ransac),
        ("lmeds", HomographyMethod::LMedS),
    ] {
        let (h, mask) = homography(&src, &exact, m, 3.0);
        assert_close(&norm_h(&h), &H_EXACT, 1e-5, name);
        assert_eq!(mask, vec![1u8; 40], "{name}: mask");
    }
    let (h, mask) = homography(&src, &exact, HomographyMethod::Ransac, 1.0);
    assert_close(&norm_h(&h), &H_EXACT, 1e-5, "ransac_thr_1");
    assert_eq!(mask, vec![1u8; 40]);

    for m in [
        HomographyMethod::None,
        HomographyMethod::Ransac,
        HomographyMethod::LMedS,
    ] {
        let (h, mask) = homography(&src[..4], &exact[..4], m, 3.0);
        assert_close(&norm_h(&h), &H_FOUR_POINTS, 1e-4, "four_points");
        assert_eq!(mask, vec![1u8; 4]);
    }
    let (h, mask) = homography(&src, &src, HomographyMethod::Ransac, 3.0);
    assert_close(
        &norm_h(&h),
        &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        1e-12,
        "identity",
    );
    assert_eq!(mask, vec![1u8; 40]);
}

/// `outliers`: 10 of 40 correspondences replaced by random pixels. The inlier sets of RANSAC
/// and LMedS (thresholds 3 and 1) are OpenCV's 30 true inliers; the matrices differ by the
/// refinement OpenCV applies (`find_homography_refines_with_lm_like_opencv`).
#[test]
fn find_homography_outlier_masks_match_opencv() {
    let (src, _, _, outl) = homography_inputs();
    for (name, m, thr) in [
        ("ransac", HomographyMethod::Ransac, 3.0),
        ("lmeds", HomographyMethod::LMedS, 3.0),
        ("ransac_thr_1", HomographyMethod::Ransac, 1.0),
    ] {
        let (h, mask) = homography(&src, &outl, m, thr);
        assert_eq!(mask, MASK_OUTLIERS_RANSAC.to_vec(), "{name}: mask");
        assert_eq!(mask.iter().filter(|&&v| v == 1).count(), 30);
        assert_close(&norm_h(&h), &H_OUTLIERS_RANSAC, 0.05, name);
    }
}

/// `d2`: fewer than four points and mismatched lengths are errors in both libraries.
#[test]
fn find_homography_input_validation_matches_opencv() {
    let (src, exact, _, _) = homography_inputs();
    assert!(find_homography(&src[..3], &exact[..3], HomographyMethod::None, 3.0, None).is_err());
    assert!(find_homography(&src[..5], &exact[..4], HomographyMethod::None, 3.0, None).is_err());
}

/// OpenCV refines every result of `n > 4` points with ten Levenberg-Marquardt iterations on
/// the reprojection error; purecv returns the DLT solution (`noisy/method0`).
#[test]
#[ignore = "diverges from OpenCV 4.10.0, see #259"]
fn find_homography_method0_refines_with_lm_like_opencv() {
    let (src, _, noisy, _) = homography_inputs();
    let (h, _) = homography(&src, &noisy, HomographyMethod::None, 3.0);
    assert_close(&norm_h(&h), &H_NOISY_METHOD0, 1e-4, "noisy/method0");
}

/// OpenCV's LMedS result is refined like method 0 (`noisy/lmeds` equals `noisy/method0`);
/// purecv runs its RANSAC and refits with the DLT.
#[test]
#[ignore = "diverges from OpenCV 4.10.0, see #259"]
fn find_homography_lmeds_refines_with_lm_like_opencv() {
    let (src, _, noisy, _) = homography_inputs();
    let (h, _) = homography(&src, &noisy, HomographyMethod::LMedS, 3.0);
    assert_close(&norm_h(&h), &H_NOISY_METHOD0, 1e-4, "noisy/lmeds");
}

/// OpenCV re-fits the RANSAC inliers with the kernel and refines them with LM
/// (`outliers/ransac`: same inlier set as purecv, a matrix 0.011 away).
#[test]
#[ignore = "diverges from OpenCV 4.10.0, see #259"]
fn find_homography_ransac_refines_with_lm_like_opencv() {
    let (src, _, _, outl) = homography_inputs();
    let (h, _) = homography(&src, &outl, HomographyMethod::Ransac, 3.0);
    assert_close(&norm_h(&h), &H_OUTLIERS_RANSAC, 1e-4, "outliers/ransac");
}

/// OpenCV's RANSAC mask is the inlier set of the best minimal-sample model (drawn with
/// `cv::RNG`, adaptive iteration count); purecv recomputes the mask from the refit. On the
/// noisy set OpenCV drops two borderline points at threshold 3 and one at threshold 1.
#[test]
#[ignore = "diverges from OpenCV 4.10.0, see #225"]
fn find_homography_ransac_mask_matches_opencv_on_noisy_data() {
    let (src, _, noisy, _) = homography_inputs();
    let (_, mask) = homography(&src, &noisy, HomographyMethod::Ransac, 3.0);
    assert_eq!(mask, MASK_NOISY_RANSAC.to_vec());
    let (_, mask) = homography(&src, &noisy, HomographyMethod::Ransac, 1.0);
    assert_eq!(mask, MASK_NOISY_RANSAC_THR_1.to_vec());
}

/// RHO (PROSAC) is a distinct estimator in OpenCV (`outliers/rho`: 21 inliers, its own
/// matrix; `exact/rho`: a single-precision result 9.9e-6 from the LM-refined one); purecv
/// runs RANSAC for `Rho` and `LMedS`.
#[test]
#[ignore = "diverges from OpenCV 4.10.0, see #259"]
fn find_homography_rho_is_distinct_from_ransac() {
    let (src, exact, _, outl) = homography_inputs();
    let (h, mask) = homography(&src, &exact, HomographyMethod::Rho, 3.0);
    assert_eq!(mask, vec![1u8; 40], "exact/rho: mask");
    assert_close(&norm_h(&h), &H_EXACT_RHO, 1e-6, "exact/rho");
    let (h, mask) = homography(&src, &outl, HomographyMethod::Rho, 3.0);
    assert_eq!(mask, MASK_OUTLIERS_RHO.to_vec());
    assert_close(&norm_h(&h), &H_OUTLIERS_RHO, 1e-4, "outliers/rho");
}

/// A non-positive `ransacReprojThreshold` means the default 3 in OpenCV; purecv finds no
/// inlier and fails (`*/ransac_thr_0`).
#[test]
#[ignore = "diverges from OpenCV 4.10.0, see #259"]
fn find_homography_defaults_threshold_when_non_positive() {
    let (src, exact, _, _) = homography_inputs();
    let (h, mask) = homography(&src, &exact, HomographyMethod::Ransac, 0.0);
    assert_close(&norm_h(&h), &H_EXACT, 1e-5, "ransac_thr_0");
    assert_eq!(mask, vec![1u8; 40]);
}

/// Collinear or coincident points give an empty matrix in OpenCV (`collinear_4`,
/// `collinear_5_ransac`, `coincident_4`); purecv returns a matrix with a zero last row.
#[test]
#[ignore = "diverges from OpenCV 4.10.0, see #259"]
fn find_homography_rejects_degenerate_input_like_opencv() {
    let col: Vec<Point2f> = (0..4)
        .map(|i| Point2f::new(10.0 * i as f32, 10.0 * i as f32))
        .collect();
    assert!(find_homography(&col, &col, HomographyMethod::None, 3.0, None).is_err());
    let same = vec![Point2f::new(10.0, 10.0); 4];
    assert!(find_homography(&same, &same, HomographyMethod::None, 3.0, None).is_err());
}

// ---------------------------------------------------------------------------------------
// rodrigues (calib3d_rodrigues.py)
// ---------------------------------------------------------------------------------------

/// `vec_to_mat`: 24 rotation vectors (zero, tiny, small, general, the pi edge in seven axes,
/// just below pi, beyond pi and 2 pi) give OpenCV's matrices to 1 ulp (`tiny_1e-12` is in
/// `rodrigues_small_angles_return_zero_like_opencv`); `shapes`: the 1x3 row form gives the
/// same matrix and the outputs are 3x3 / 3x1.
#[test]
fn rodrigues_vec_to_mat_matches_opencv() {
    for (name, rvec, rmat, _) in RODRIGUES.iter() {
        if *name == "tiny_1e-12" {
            continue;
        }
        let m = rod(&vec3(*rvec));
        assert_eq!((m.rows, m.cols, m.channels), (3, 3, 1), "{name}: shape");
        assert_close(&m.data, rmat, 1e-15, name);
    }
    let row = rod(&Matrix::from_vec(1, 3, 1, vec![0.1, -0.2, 0.3]));
    assert_eq!(row.data, rod(&vec3([0.1, -0.2, 0.3])).data);
    let back = rod(&row);
    assert_eq!((back.rows, back.cols), (3, 1));
}

/// `round_trip` and `mat_to_vec`: the matrices of the 25 vectors come back as OpenCV's
/// vectors wherever the libraries agree (angles at or above 2e-5, the pi edge with OpenCV's
/// sign rule, angles beyond pi reduced the same way), to 1.5e-8; rotation matrices, their
/// transposes, the identity and an out-of-range matrix give the same vectors.
#[test]
fn rodrigues_mat_to_vec_matches_opencv() {
    const DIVERGING: [&str; 5] = [
        "small_1e-6",
        "small_2e-6",
        "small_5e-6",
        "near_pi_2e-5",
        "near_pi_2e-5_neg",
    ];
    for (name, rvec, _, back) in RODRIGUES.iter() {
        if DIVERGING.contains(name) {
            continue;
        }
        let v = rod(&rod(&vec3(*rvec)));
        assert_eq!((v.rows, v.cols, v.channels), (3, 1, 1), "{name}: shape");
        assert_close(&v.data, back, 1e-7, name);
    }
    for (name, m, rvec) in MAT_TO_VEC.iter() {
        if matches!(*name, "scaled_1.1" | "perturbed_01" | "reflection_z") {
            continue;
        }
        assert_close(&rod(&mat3(*m)).data, rvec, 1e-15, name);
    }
}

/// `shapes`: 2x2, 4x1 and 3x2 inputs are errors in both libraries.
#[test]
fn rodrigues_input_validation_matches_opencv() {
    let mut dst = Matrix::<f64>::new(1, 1, 1);
    assert!(rodrigues(&Matrix::<f64>::new(2, 2, 1), &mut dst).is_err());
    assert!(rodrigues(&Matrix::<f64>::new(4, 1, 1), &mut dst).is_err());
    assert!(rodrigues(&Matrix::<f64>::new(3, 2, 1), &mut dst).is_err());
}

/// OpenCV returns the zero vector for any matrix whose rotation angle is below 1e-5 rad
/// (`s < 1e-5`); purecv returns the angle (`small_1e-6`, `small_2e-6`, `small_5e-6`). In the
/// other direction OpenCV builds the matrix of any vector above `DBL_EPSILON`; purecv returns
/// the identity below 1e-10 (`tiny_1e-12`: off-diagonals of 1e-12 lost).
#[test]
#[ignore = "diverges from OpenCV 4.10.0, see #260"]
fn rodrigues_small_angles_return_zero_like_opencv() {
    for name in ["small_1e-6", "small_2e-6", "small_5e-6"] {
        let (_, rvec, _, back) = RODRIGUES.iter().find(|e| e.0 == name).unwrap();
        assert_close(&rod(&rod(&vec3(*rvec))).data, back, 1e-12, name);
    }
    let (_, rvec, rmat, _) = RODRIGUES.iter().find(|e| e.0 == "tiny_1e-12").unwrap();
    assert_close(&rod(&vec3(*rvec)).data, rmat, 1e-15, "tiny_1e-12");
}

/// For `1e-5 <= sin(theta) < 1e-4` below pi OpenCV uses the general formula; purecv's
/// diagonal method there leaves a 3.1e-5 spurious component (`near_pi_2e-5`).
#[test]
#[ignore = "diverges from OpenCV 4.10.0, see #260"]
fn rodrigues_near_pi_band_matches_opencv() {
    for name in ["near_pi_2e-5", "near_pi_2e-5_neg"] {
        let (_, rvec, _, back) = RODRIGUES.iter().find(|e| e.0 == name).unwrap();
        assert_close(&rod(&rod(&vec3(*rvec))).data, back, 1e-9, name);
    }
}

/// OpenCV projects the input onto a rotation (`R = U * Vt` from its SVD) before extracting
/// the vector; purecv reads the trace and the off-diagonals as they are (`scaled_1.1`,
/// `perturbed_01`, `reflection_z`).
#[test]
#[ignore = "diverges from OpenCV 4.10.0, see #260"]
fn rodrigues_projects_non_rotations_like_opencv() {
    for name in ["scaled_1.1", "perturbed_01", "reflection_z"] {
        let (_, m, rvec) = MAT_TO_VEC.iter().find(|e| e.0 == name).unwrap();
        assert_close(&rod(&mat3(*m)).data, rvec, 1e-9, name);
    }
}

/// OpenCV accepts a 1x1 three-channel vector (`vec3_1x1x3`); purecv requires one channel.
#[test]
#[ignore = "diverges from OpenCV 4.10.0, see #260"]
fn rodrigues_accepts_three_channel_vectors() {
    let m = rod(&Matrix::from_vec(1, 1, 3, vec![0.1, -0.2, 0.3]));
    assert_eq!(m.data, rod(&vec3([0.1, -0.2, 0.3])).data);
}

// ---------------------------------------------------------------------------------------
// solve_pnp (calib3d_pnp.py)
// ---------------------------------------------------------------------------------------

/// `iterative`: 24 non-planar points, exact and with +-0.5 px noise, and the first six
/// points, agree with OpenCV's Levenberg-Marquardt result to 2e-8 (Gauss-Newton reaches the
/// same minimum); `dist_none` / a zero vector give the same pose.
#[test]
fn solve_pnp_iterative_matches_opencv() {
    let p = pnp_inputs();
    let it = SolvePnPMethod::Iterative;
    assert_pose(
        &pnp(&p.obj, &p.img, &NO_DIST, None, it).unwrap(),
        &PNP_NONPLANAR_24_EXACT,
        1e-6,
        "nonplanar_24_exact",
    );
    assert_pose(
        &pnp(&p.obj, &p.noisy, &NO_DIST, None, it).unwrap(),
        &PNP_NONPLANAR_24_NOISY,
        1e-6,
        "nonplanar_24_noisy",
    );
    assert_pose(
        &pnp(&p.obj[..6], &p.img[..6], &NO_DIST, None, it).unwrap(),
        &PNP_NONPLANAR_6,
        1e-6,
        "nonplanar_6",
    );
    let mut rvec = Matrix::<f64>::new(3, 1, 1);
    let mut tvec = Matrix::<f64>::new(3, 1, 1);
    assert!(solve_pnp(
        &p.obj,
        &p.img,
        &camera_matrix(),
        None,
        &mut rvec,
        &mut tvec,
        false,
        it
    )
    .unwrap());
    assert_pose(
        &(rvec.data, tvec.data),
        &PNP_NONPLANAR_24_EXACT,
        1e-6,
        "dist_none",
    );
}

/// `solve_pnp_d2` and the point-count cases: three non-planar points, four and five
/// non-planar points (OpenCV: "DLT algorithm needs at least 6 points"), mismatched lengths
/// and a 2x2 camera matrix are errors in both libraries.
#[test]
fn solve_pnp_input_validation_matches_opencv() {
    let p = pnp_inputs();
    let it = SolvePnPMethod::Iterative;
    for n in [3, 4, 5] {
        assert!(
            pnp(&p.obj[..n], &p.img[..n], &NO_DIST, None, it).is_err(),
            "{n} points"
        );
    }
    assert!(pnp(&p.obj[..6], &p.img[..5], &NO_DIST, None, it).is_err());
    let mut rvec = Matrix::<f64>::new(3, 1, 1);
    let mut tvec = Matrix::<f64>::new(3, 1, 1);
    assert!(solve_pnp(
        &p.obj,
        &p.img,
        &Matrix::<f64>::new(2, 2, 1),
        None,
        &mut rvec,
        &mut tvec,
        false,
        it
    )
    .is_err());
}

/// Coplanar object points (`planar_16_exact`): OpenCV initialises from a homography and
/// recovers the pose; purecv's 12-parameter DLT is rank deficient there and returns a pose
/// 2.9 rad and 8 units away.
#[test]
#[ignore = "diverges from OpenCV 4.10.0, see #261"]
fn solve_pnp_planar_exact_matches_opencv() {
    let p = pnp_inputs();
    let got = pnp(
        &p.planar,
        &p.img_planar,
        &NO_DIST,
        None,
        SolvePnPMethod::Iterative,
    )
    .unwrap();
    assert_pose(&got, &PNP_PLANAR_16_EXACT, 1e-5, "planar_16_exact");
}

/// The same planar failure with +-0.5 px noise (`planar_16_noisy`).
#[test]
#[ignore = "diverges from OpenCV 4.10.0, see #261"]
fn solve_pnp_planar_noisy_matches_opencv() {
    let p = pnp_inputs();
    let got = pnp(
        &p.planar,
        &p.noisy_planar,
        &NO_DIST,
        None,
        SolvePnPMethod::Iterative,
    )
    .unwrap();
    assert_pose(&got, &PNP_PLANAR_16_NOISY, 1e-5, "planar_16_noisy");
}

/// OpenCV undistorts the image points with `distCoeffs` (`nonplanar_24_dist5` recovers the
/// true pose); purecv ignores them and returns OpenCV's zero-coefficient pose
/// (`nonplanar_24_dist5_ignored`, 0.05 units off, measured by the script), and accepts a
/// 3-element vector.
#[test]
#[ignore = "diverges from OpenCV 4.10.0, see #261"]
fn solve_pnp_applies_distortion_coefficients() {
    let p = pnp_inputs();
    let dist_img: Vec<Point2f> = IMG_24_DIST5
        .iter()
        .map(|&(x, y)| Point2f::new(x, y))
        .collect();
    let got = pnp(&p.obj, &dist_img, &DIST5, None, SolvePnPMethod::Iterative).unwrap();
    assert_pose(&got, &PNP_NONPLANAR_24_DIST5, 1e-6, "nonplanar_24_dist5");
    assert!(
        pnp(&p.obj, &p.img, &[0.0; 3], None, SolvePnPMethod::Iterative).is_err(),
        "dist_3_invalid"
    );
}

/// Four coplanar points (`square_4`) and three points with an extrinsic guess
/// (`guess_good_3_points`) are solved by OpenCV's ITERATIVE method; purecv needs six.
#[test]
#[ignore = "diverges from OpenCV 4.10.0, see #261"]
fn solve_pnp_accepts_four_planar_points_like_opencv() {
    let (square, img) = square_inputs();
    assert_pose(
        &pnp(&square, &img, &NO_DIST, None, SolvePnPMethod::Iterative).unwrap(),
        &PNP_SQUARE_4,
        1e-5,
        "square_4",
    );
}

/// `useExtrinsicGuess` starts the refinement from the given pose: from a good guess OpenCV
/// reaches the same minimum as without (`guess_good_noisy`); purecv returns
/// `NotImplemented`.
#[test]
#[ignore = "diverges from OpenCV 4.10.0, see #133"]
fn solve_pnp_extrinsic_guess_matches_opencv() {
    let p = pnp_inputs();
    let good = (
        [0.5759483135718835, 0.5578245857568109, 0.23656695264724611],
        [0.35, -0.175, 4.2],
    );
    let got = pnp(
        &p.obj,
        &p.noisy,
        &NO_DIST,
        Some(good),
        SolvePnPMethod::Iterative,
    )
    .unwrap();
    assert_pose(&got, &PNP_GUESS_GOOD_NOISY, 1e-6, "guess_good_noisy");
}

/// With a guess OpenCV's ITERATIVE method solves three points (`guess_good_3_points`);
/// purecv rejects fewer than six.
#[test]
#[ignore = "diverges from OpenCV 4.10.0, see #133"]
fn solve_pnp_extrinsic_guess_solves_three_points_like_opencv() {
    let p = pnp_inputs();
    let good = (
        [0.5759483135718835, 0.5578245857568109, 0.23656695264724611],
        [0.35, -0.175, 4.2],
    );
    let got = pnp(
        &p.obj[..3],
        &p.img[..3],
        &NO_DIST,
        Some(good),
        SolvePnPMethod::Iterative,
    )
    .unwrap();
    assert_pose(&got, &PNP_GUESS_GOOD_3_POINTS, 1e-5, "guess_good_3_points");
}

/// `SOLVEPNP_EPNP` on 24 noisy points (`epnp/nonplanar_24_noisy`); purecv returns
/// `NotImplemented` for every method but `Iterative`.
#[test]
#[ignore = "diverges from OpenCV 4.10.0, see #262"]
fn solve_pnp_epnp_matches_opencv() {
    let p = pnp_inputs();
    let got = pnp(&p.obj, &p.noisy, &NO_DIST, None, SolvePnPMethod::EPnP).unwrap();
    assert_pose(&got, &PNP_EPNP_NONPLANAR_24_NOISY, 1e-6, "epnp");
}

/// `SOLVEPNP_SQPNP` on 24 noisy points (`sqpnp/nonplanar_24_noisy`).
#[test]
#[ignore = "diverges from OpenCV 4.10.0, see #262"]
fn solve_pnp_sqpnp_matches_opencv() {
    let p = pnp_inputs();
    let got = pnp(&p.obj, &p.noisy, &NO_DIST, None, SolvePnPMethod::SqPnP).unwrap();
    assert_pose(&got, &PNP_SQPNP_NONPLANAR_24_NOISY, 1e-6, "sqpnp");
}

/// `SOLVEPNP_P3P` on four points (`p3p/nonplanar_4`).
#[test]
#[ignore = "diverges from OpenCV 4.10.0, see #262"]
fn solve_pnp_p3p_matches_opencv() {
    let p = pnp_inputs();
    let got = pnp(
        &p.obj[..4],
        &p.img[..4],
        &NO_DIST,
        None,
        SolvePnPMethod::P3P,
    )
    .unwrap();
    assert_pose(&got, &PNP_P3P_NONPLANAR_4, 1e-5, "p3p");
}

/// `SOLVEPNP_AP3P` on four points (`ap3p/nonplanar_4`).
#[test]
#[ignore = "diverges from OpenCV 4.10.0, see #262"]
fn solve_pnp_ap3p_matches_opencv() {
    let p = pnp_inputs();
    let got = pnp(
        &p.obj[..4],
        &p.img[..4],
        &NO_DIST,
        None,
        SolvePnPMethod::AP3P,
    )
    .unwrap();
    assert_pose(&got, &PNP_AP3P_NONPLANAR_4, 1e-5, "ap3p");
}

// ---------------------------------------------------------------------------------------
// solve_pnp_ransac (calib3d_pnp.py, "solve_pnp_ransac")
// ---------------------------------------------------------------------------------------

/// `outliers_default` / `outliers_thr_3` / `outliers_iters_1000` (6 of 24 correspondences
/// replaced by random pixels), `exact_default`, `noisy_default` and `points_6`: the inlier
/// index sets are identical and the poses agree to 2e-8; `all_outliers` fails in both
/// libraries, and so does `iters_0` (OpenCV runs no hypothesis; purecv clamps the count to one
/// and that one hypothesis finds fewer than six inliers on this set).
#[test]
fn solve_pnp_ransac_matches_opencv() {
    let p = pnp_inputs();
    let it = SolvePnPMethod::Iterative;
    for (name, iters, err) in [
        ("outliers_default", 100, 8.0),
        ("outliers_thr_3", 100, 3.0),
        ("outliers_iters_1000", 1000, 8.0),
    ] {
        let (ok, r, t, inl) =
            pnp_ransac(&p.obj, &p.outliers, &NO_DIST, None, iters, err, 0.99, it).unwrap();
        assert!(ok, "{name}");
        assert_eq!(inl, RANSAC_OUTLIERS_DEFAULT.2.to_vec(), "{name}: inliers");
        assert_pose(
            &(r, t),
            &(RANSAC_OUTLIERS_DEFAULT.0, RANSAC_OUTLIERS_DEFAULT.1),
            1e-6,
            name,
        );
    }
    let (ok, r, t, inl) = pnp_ransac(&p.obj, &p.img, &NO_DIST, None, 100, 8.0, 0.99, it).unwrap();
    assert!(ok);
    assert_eq!(inl, RANSAC_EXACT_DEFAULT.2.to_vec());
    assert_pose(
        &(r, t),
        &(RANSAC_EXACT_DEFAULT.0, RANSAC_EXACT_DEFAULT.1),
        1e-6,
        "exact_default",
    );
    let (ok, r, t, inl) = pnp_ransac(&p.obj, &p.noisy, &NO_DIST, None, 100, 8.0, 0.99, it).unwrap();
    assert!(ok);
    assert_eq!(inl, RANSAC_NOISY_DEFAULT.2.to_vec());
    assert_pose(
        &(r, t),
        &(RANSAC_NOISY_DEFAULT.0, RANSAC_NOISY_DEFAULT.1),
        1e-6,
        "noisy_default",
    );
    let (ok, r, t, inl) =
        pnp_ransac(&p.obj[..6], &p.img[..6], &NO_DIST, None, 100, 8.0, 0.99, it).unwrap();
    assert!(ok);
    assert_eq!(inl, RANSAC_POINTS_6.2.to_vec());
    assert_pose(
        &(r, t),
        &(RANSAC_POINTS_6.0, RANSAC_POINTS_6.1),
        1e-6,
        "points_6",
    );

    let all_out = replace_outliers(&p.img, 24, 0x777);
    assert!(
        !pnp_ransac(&p.obj, &all_out, &NO_DIST, None, 100, 8.0, 0.99, it)
            .unwrap()
            .0,
        "all_outliers"
    );
    assert!(
        !pnp_ransac(&p.obj, &p.outliers, &NO_DIST, None, 0, 8.0, 0.99, it)
            .unwrap()
            .0,
        "iters_0"
    );
    assert!(
        pnp_ransac(&p.obj[..3], &p.img[..3], &NO_DIST, None, 100, 8.0, 0.99, it).is_err(),
        "points_3"
    );
}

/// At threshold 1 on the noisy set OpenCV's 5-point EPnP hypotheses (drawn with `cv::RNG`,
/// errors in pixels) keep 22 points where purecv's 6-point DLT hypotheses (errors in
/// normalised coordinates) keep all 24 (`noisy_thr_1`).
#[test]
#[ignore = "diverges from OpenCV 4.10.0, see #225"]
fn solve_pnp_ransac_noisy_threshold_1_mask_matches_opencv() {
    let p = pnp_inputs();
    let (_, _, _, inl) = pnp_ransac(
        &p.obj,
        &p.noisy,
        &NO_DIST,
        None,
        100,
        1.0,
        0.99,
        SolvePnPMethod::Iterative,
    )
    .unwrap();
    assert_eq!(inl, RANSAC_NOISY_THR_1.2.to_vec());
}

/// Five points run EPnP directly in OpenCV (`points_5`: the true pose); purecv requires six.
#[test]
#[ignore = "diverges from OpenCV 4.10.0, see #263"]
fn solve_pnp_ransac_accepts_five_points_like_opencv() {
    let p = pnp_inputs();
    let it = SolvePnPMethod::Iterative;
    let (ok, r, t, inl) =
        pnp_ransac(&p.obj[..5], &p.img[..5], &NO_DIST, None, 100, 8.0, 0.99, it).unwrap();
    assert!(ok);
    assert_eq!(inl, RANSAC_POINTS_5.2.to_vec());
    assert_pose(
        &(r, t),
        &(RANSAC_POINTS_5.0, RANSAC_POINTS_5.1),
        1e-5,
        "points_5",
    );
}

/// Four points run P3P directly in OpenCV (`points_4`: all four inliers, the true pose);
/// purecv requires six.
#[test]
#[ignore = "diverges from OpenCV 4.10.0, see #263"]
fn solve_pnp_ransac_accepts_four_points_like_opencv() {
    let p = pnp_inputs();
    let it = SolvePnPMethod::Iterative;
    let (ok, r, t, inl) =
        pnp_ransac(&p.obj[..4], &p.img[..4], &NO_DIST, None, 100, 8.0, 0.99, it).unwrap();
    assert!(ok);
    assert_eq!(inl, RANSAC_POINTS_4.2.to_vec());
    assert_pose(
        &(r, t),
        &(RANSAC_POINTS_4.0, RANSAC_POINTS_4.1),
        1e-5,
        "points_4",
    );
}

/// OpenCV asserts `0 < confidence < 1` (`conf_1`); purecv ignores the value.
#[test]
#[ignore = "diverges from OpenCV 4.10.0, see #263"]
fn solve_pnp_ransac_rejects_confidence_outside_0_1() {
    let p = pnp_inputs();
    assert!(pnp_ransac(
        &p.obj,
        &p.outliers,
        &NO_DIST,
        None,
        100,
        8.0,
        1.0,
        SolvePnPMethod::Iterative
    )
    .is_err());
}

/// `outliers_dist5`: the distortion is applied to the hypotheses' reprojection errors and to
/// the final pose in OpenCV (true pose recovered, same inlier set); purecv's pose is 0.05
/// units off. `planar_noisy`: the planar failure of `solve_pnp` (same inlier set).
#[test]
#[ignore = "diverges from OpenCV 4.10.0, see #261"]
fn solve_pnp_ransac_distortion_and_planar_match_opencv() {
    let p = pnp_inputs();
    let it = SolvePnPMethod::Iterative;
    let dist_img: Vec<Point2f> = IMG_24_DIST5
        .iter()
        .map(|&(x, y)| Point2f::new(x, y))
        .collect();
    let dist_outl = replace_outliers(&dist_img, 6, 0x2469);
    let (ok, r, t, inl) = pnp_ransac(&p.obj, &dist_outl, &DIST5, None, 100, 8.0, 0.99, it).unwrap();
    assert!(ok);
    assert_eq!(inl, RANSAC_OUTLIERS_DIST5.2.to_vec());
    let (ok2, r2, t2, inl2) = pnp_ransac(
        &p.planar,
        &p.noisy_planar,
        &NO_DIST,
        None,
        100,
        8.0,
        0.99,
        it,
    )
    .unwrap();
    assert!(ok2);
    assert_eq!(inl2, RANSAC_PLANAR_NOISY.2.to_vec());
    assert_pose(
        &(r, t),
        &(RANSAC_OUTLIERS_DIST5.0, RANSAC_OUTLIERS_DIST5.1),
        1e-5,
        "outliers_dist5",
    );
    assert_pose(
        &(r2, t2),
        &(RANSAC_PLANAR_NOISY.0, RANSAC_PLANAR_NOISY.1),
        1e-5,
        "planar_noisy",
    );
}

/// `outliers_guess_good`: the guess seeds every hypothesis and the final refinement in
/// OpenCV (same inlier set and pose as without it); purecv returns `NotImplemented`.
#[test]
#[ignore = "diverges from OpenCV 4.10.0, see #133"]
fn solve_pnp_ransac_extrinsic_guess_matches_opencv() {
    let p = pnp_inputs();
    let good = (
        [0.5759483135718835, 0.5578245857568109, 0.23656695264724611],
        [0.35, -0.175, 4.2],
    );
    let (ok, r, t, inl) = pnp_ransac(
        &p.obj,
        &p.outliers,
        &NO_DIST,
        Some(good),
        100,
        8.0,
        0.99,
        SolvePnPMethod::Iterative,
    )
    .unwrap();
    assert!(ok);
    assert_eq!(inl, RANSAC_OUTLIERS_DEFAULT.2.to_vec());
    assert_pose(
        &(r, t),
        &(RANSAC_OUTLIERS_DEFAULT.0, RANSAC_OUTLIERS_DEFAULT.1),
        1e-6,
        "outliers_guess_good",
    );
}

/// `outliers_epnp`: with `SOLVEPNP_EPNP` OpenCV refines the inliers with EPnP (same inlier
/// set, a slightly different pose); purecv returns `NotImplemented`.
#[test]
#[ignore = "diverges from OpenCV 4.10.0, see #262"]
fn solve_pnp_ransac_other_methods_match_opencv() {
    let p = pnp_inputs();
    let (ok, r, t, inl) = pnp_ransac(
        &p.obj,
        &p.outliers,
        &NO_DIST,
        None,
        100,
        8.0,
        0.99,
        SolvePnPMethod::EPnP,
    )
    .unwrap();
    assert!(ok);
    assert_eq!(inl, RANSAC_OUTLIERS_EPNP.2.to_vec());
    assert_pose(
        &(r, t),
        &(RANSAC_OUTLIERS_EPNP.0, RANSAC_OUTLIERS_EPNP.1),
        1e-6,
        "outliers_epnp",
    );
}

// ---------------------------------------------------------------------------------------
// init_undistort_rectify_map (calib3d_undistort.py)
// ---------------------------------------------------------------------------------------

/// `identity_r_same_k` with 4, 5 and 8 coefficients (two sets), `rectified` with a
/// rotation and/or a new camera matrix: the `CV_32FC1` maps are byte-identical to OpenCV's
/// (FNV-1a digests). With zero or no distortion the maps agree to 1.8e-14 (OpenCV's SIMD
/// path returns an exact 0 where the generic formula leaves `fx * (-cx / fx) + cx`).
#[test]
fn init_undistort_rectify_map_matches_opencv() {
    let k = camera_matrix();
    let nk = new_camera_matrix();
    let r = r_small();
    for (name, dist) in [
        ("k4", &K4[..]),
        ("k5", &K5[..]),
        ("k8", &K8[..]),
        ("k8_strong", &K8_STRONG[..]),
    ] {
        let got = undistort_maps(&k, dist, None, &k, 64, 48).unwrap();
        assert_maps_identical(&got, &format!("identity_r_same_k/{name}"));
    }
    assert_maps_identical(
        &undistort_maps(&k, &K5, Some(&r), &nk, 64, 48).unwrap(),
        "rectified/k5",
    );
    assert_maps_identical(
        &undistort_maps(&k, &K8, Some(&r), &nk, 64, 48).unwrap(),
        "rectified/k8",
    );
    assert_maps_identical(
        &undistort_maps(&k, &K5, Some(&r), &k, 64, 48).unwrap(),
        "rectified/k5_r_only",
    );
    assert_maps_identical(
        &undistort_maps(&k, &K5, None, &nk, 64, 48).unwrap(),
        "rectified/k5_newk_only",
    );
    assert_maps_samples(
        &undistort_maps(&k, &[0.0; 5], None, &k, 64, 48).unwrap(),
        "identity_r_same_k/zeros_5",
        1e-12,
    );
    assert_maps_samples(
        &undistort_maps(&k, &[], None, &k, 64, 48).unwrap(),
        "identity_r_same_k/none_dist",
        1e-12,
    );
    let (m1, m2) = undistort_maps(&k, &K5, None, &k, 1, 1).unwrap();
    assert_eq!((m1.data[0], m2.data[0]), UNDISTORT_SIZE_1X1, "size_1x1");
    assert!(
        undistort_maps(&k, &K5, Some(&Matrix::<f64>::new(3, 1, 1)), &k, 64, 48).is_err(),
        "r_3x1_vector"
    );
}

/// `k12` adds the thin-prism terms `s1..s4`; purecv reads at most eight coefficients, so the
/// maps are up to 0.18 px off.
#[test]
#[ignore = "diverges from OpenCV 4.10.0, see #264"]
fn init_undistort_rectify_map_thin_prism_matches_opencv() {
    let k = camera_matrix();
    let got = undistort_maps(&k, &K12, None, &k, 64, 48).unwrap();
    assert_maps_identical(&got, "identity_r_same_k/k12");
}

/// `k14` adds the tilt `tau_x, tau_y` (`computeTiltProjectionMatrix`): the maps are up to
/// 4.2 px off, without and with a rectification (`identity_r_same_k/k14`, `rectified/k14`).
#[test]
#[ignore = "diverges from OpenCV 4.10.0, see #264"]
fn init_undistort_rectify_map_tilt_matches_opencv() {
    let k = camera_matrix();
    let got = undistort_maps(&k, &K14, None, &k, 64, 48).unwrap();
    assert_maps_identical(&got, "identity_r_same_k/k14");
    let got = undistort_maps(&k, &K14, Some(&r_small()), &new_camera_matrix(), 64, 48).unwrap();
    assert_maps_identical(&got, "rectified/k14");
}

/// OpenCV asserts that `distCoeffs` has 4, 5, 8, 12 or 14 elements (`dist_3_invalid`,
/// `dist_6_invalid`, `dist_9_invalid`); purecv reads what is there and pads with zeros.
#[test]
#[ignore = "diverges from OpenCV 4.10.0, see #264"]
fn init_undistort_rectify_map_rejects_invalid_coefficient_counts() {
    let k = camera_matrix();
    assert!(undistort_maps(&k, &[0.1, 0.2, 0.3], None, &k, 64, 48).is_err());
    assert!(undistort_maps(&k, &[0.1, 0.2, 0.3, 0.0, 0.0, 0.0], None, &k, 64, 48).is_err());
}

/// A 0x0 size gives empty maps in OpenCV (`size_0x0`); purecv panics in the `parallel` build
/// (`par_chunks_exact_mut(0)`) and returns empty maps otherwise.
#[test]
#[ignore = "diverges from OpenCV 4.10.0, see #264"]
fn init_undistort_rectify_map_empty_size_returns_empty_maps() {
    let k = camera_matrix();
    let (m1, m2) = undistort_maps(&k, &K5, None, &k, 0, 0).unwrap();
    assert_eq!((m1.rows, m1.cols, m2.rows, m2.cols), (0, 0, 0, 0));
}

// ---------------------------------------------------------------------------------------
// find_fundamental_mat (calib3d_fundamental.py)
// ---------------------------------------------------------------------------------------

/// `fm_8point` on the exact, noisy and outlier sets (60 points) and on the first 8 and 10
/// points: the matrices (unit Frobenius norm, sign fixed) agree to 3e-13 and the masks are
/// all ones; `six_points` and `mismatched` fail in both libraries.
#[test]
fn find_fundamental_mat_8point_matches_opencv() {
    let (p1, p2, n1, n2, o2) = fundamental_inputs();
    let m = FundamentalMatMethod::FM_8POINT;
    for (name, a, b, e60, e8, e10) in [
        (
            "exact",
            &p1,
            &p2,
            &F_EXACT_8POINT,
            &F_EXACT_8POINT_8,
            &F_EXACT_8POINT_10,
        ),
        (
            "noisy",
            &n1,
            &n2,
            &F_NOISY_8POINT,
            &F_NOISY_8POINT_8,
            &F_NOISY_8POINT_10,
        ),
        (
            "outliers",
            &n1,
            &o2,
            &F_OUTLIERS_8POINT,
            &F_OUTLIERS_8POINT_8,
            &F_OUTLIERS_8POINT_10,
        ),
    ] {
        let (f, mask) = fundamental(a, b, m, 3.0, 0.99, 1000);
        assert_close(&norm_f(&f), e60, 1e-10, name);
        assert_eq!(mask, vec![1u8; 60]);
        let (f, mask) = fundamental(&a[..8], &b[..8], m, 3.0, 0.99, 1000);
        assert_close(&norm_f(&f), e8, 1e-10, &format!("{name}/8"));
        assert_eq!(mask, vec![1u8; 8]);
        let (f, mask) = fundamental(&a[..10], &b[..10], m, 3.0, 0.99, 1000);
        assert_close(&norm_f(&f), e10, 1e-10, &format!("{name}/10"));
        assert_eq!(mask, vec![1u8; 10]);
    }
    assert_close(&F_EXACT_8POINT, &F_TRUE, 1e-5, "f_true");
    assert!(find_fundamental_mat(&p1[..6], &p2[..6], m, 3.0, 0.99, 1000, None).is_err());
    assert!(find_fundamental_mat(&p1[..9], &p2[..8], m, 3.0, 0.99, 1000, None).is_err());
}

/// `exact/fm_ransac` and `fm_ransac_thr_1`, `fm_ransac_14_points`, `fm_ransac_15_points` on
/// exact correspondences: every point is an inlier in both libraries and the matrices agree
/// to 3.1e-6 (OpenCV keeps its best 7-point model, purecv refits the inliers).
#[test]
fn find_fundamental_mat_ransac_exact_matches_opencv() {
    let (p1, p2, _, _, _) = fundamental_inputs();
    let m = FundamentalMatMethod::FM_RANSAC;
    for (name, thr) in [("fm_ransac", 3.0), ("fm_ransac_thr_1", 1.0)] {
        let (f, mask) = fundamental(&p1, &p2, m, thr, 0.99, 1000);
        assert_eq!(mask, F_MASK_EXACT_RANSAC.to_vec(), "{name}: mask");
        assert_eq!(mask, vec![1u8; 60]);
        assert_close(&norm_f(&f), &F_EXACT_RANSAC, 1e-5, name);
    }
    for n in [14, 15] {
        let (f, mask) = fundamental(&p1[..n], &p2[..n], m, 3.0, 0.99, 1000);
        assert_eq!(mask, vec![1u8; n], "{n} points: mask");
        assert_close(&norm_f(&f), &F_EXACT_RANSAC, 1e-5, &format!("{n} points"));
    }
}

/// On the noisy set OpenCV's RANSAC (7-point kernel, point-to-epiline error, `cv::RNG`, no
/// refit; the mask of the best minimal model) keeps 58 points where purecv (8-point kernel,
/// Sampson error, refit, mask of the refit) keeps 60 (`noisy/fm_ransac`).
#[test]
#[ignore = "diverges from OpenCV 4.10.0, see #225"]
fn find_fundamental_mat_ransac_mask_matches_opencv_on_noisy_data() {
    let (_, _, n1, n2, _) = fundamental_inputs();
    let (_, mask) = fundamental(&n1, &n2, FundamentalMatMethod::FM_RANSAC, 3.0, 0.99, 1000);
    assert_eq!(mask, F_MASK_NOISY_RANSAC.to_vec());
}

/// On the outlier set OpenCV keeps the 48 true inliers, purecv 50 (two true outliers within
/// 3 px of their epilines; `outliers/fm_ransac`).
#[test]
#[ignore = "diverges from OpenCV 4.10.0, see #225"]
fn find_fundamental_mat_ransac_mask_matches_opencv_on_outliers() {
    let (_, _, n1, _, o2) = fundamental_inputs();
    let (_, mask) = fundamental(&n1, &o2, FundamentalMatMethod::FM_RANSAC, 3.0, 0.99, 1000);
    assert_eq!(mask, F_MASK_OUTLIERS_RANSAC.to_vec());
}

/// With fewer than 15 points `FM_RANSAC` runs LMedS in OpenCV (`noisy/fm_ransac_14_points`:
/// 8 inliers of 14 from the median error); purecv runs RANSAC with the given threshold.
#[test]
#[ignore = "diverges from OpenCV 4.10.0, see #265"]
fn find_fundamental_mat_uses_lmeds_below_15_points_like_opencv() {
    let (_, _, n1, n2, _) = fundamental_inputs();
    let (_, mask) = fundamental(
        &n1[..14],
        &n2[..14],
        FundamentalMatMethod::FM_RANSAC,
        3.0,
        0.99,
        1000,
    );
    assert_eq!(mask, F_MASK_NOISY_RANSAC_14.to_vec());
}

/// A non-positive threshold means 3 in OpenCV (`fm_ransac_thr_0` returns the threshold-3
/// result); purecv fails.
#[test]
#[ignore = "diverges from OpenCV 4.10.0, see #265"]
fn find_fundamental_mat_resets_non_positive_threshold_like_opencv() {
    let (p1, p2, _, _, _) = fundamental_inputs();
    let (f, mask) = fundamental(&p1, &p2, FundamentalMatMethod::FM_RANSAC, 0.0, 0.99, 1000);
    assert_eq!(mask, vec![1u8; 60]);
    assert_close(&norm_f(&f), &F_EXACT_RANSAC, 1e-5, "fm_ransac_thr_0");
}

/// A confidence outside `(0, 1)` means 0.99 in OpenCV (`outliers/fm_ransac_conf_0` returns
/// the 48 inliers of the default); purecv stops after the first model with confidence 0.
#[test]
#[ignore = "diverges from OpenCV 4.10.0, see #265"]
fn find_fundamental_mat_resets_invalid_confidence_like_opencv() {
    let (_, _, n1, _, o2) = fundamental_inputs();
    let (_, mask) = fundamental(&n1, &o2, FundamentalMatMethod::FM_RANSAC, 3.0, 0.0, 1000);
    assert_eq!(mask, F_MASK_OUTLIERS_RANSAC_CONF_0.to_vec());
}

/// Seven points run the 7-point algorithm whatever the method (`seven_points_8point`: three
/// stacked solutions, a 9x3 matrix); purecv requires eight.
#[test]
#[ignore = "diverges from OpenCV 4.10.0, see #265"]
fn find_fundamental_mat_accepts_seven_points_like_opencv() {
    let (p1, p2, _, _, _) = fundamental_inputs();
    let f = find_fundamental_mat(
        &p1[..7],
        &p2[..7],
        FundamentalMatMethod::FM_8POINT,
        3.0,
        0.99,
        1000,
        None,
    )
    .unwrap();
    assert_eq!((f.rows, f.cols), (9, 3));
}

/// Collinear points and identical point sets give an empty matrix in OpenCV (`collinear_10`,
/// `identical_sets_8point`: a 9x9 eigenvalue below `DBL_EPSILON`); purecv returns a matrix.
#[test]
#[ignore = "diverges from OpenCV 4.10.0, see #265"]
fn find_fundamental_mat_rejects_degenerate_input_like_opencv() {
    let (p1, _, _, _, _) = fundamental_inputs();
    let same: Vec<Point2f> = (0..10)
        .map(|i| Point2f::new(10.0 + i as f32, 20.0))
        .collect();
    assert!(find_fundamental_mat(
        &same,
        &same,
        FundamentalMatMethod::FM_8POINT,
        3.0,
        0.99,
        1000,
        None
    )
    .is_err());
    assert!(find_fundamental_mat(
        &p1,
        &p1,
        FundamentalMatMethod::FM_8POINT,
        3.0,
        0.99,
        1000,
        None
    )
    .is_err());
}

// ---------------------------------------------------------------------------------------
// OpenCV 4.10.0 reference values
// ---------------------------------------------------------------------------------------

/// Generated from the JSON output of the scripts (`opencv-contrib-python 4.10.0.84`,
/// `cv2.ipp.setUseIPP(False)`).
// Generated OpenCV 4.10.0 outputs: literals are copied, not approximations of constants.
#[allow(clippy::approx_constant, clippy::excessive_precision)]
mod ocv {
    /// `(name, rvec, rmat, the vector OpenCV recovers from rmat)`.
    pub type RodriguesCase = (&'static str, [f64; 3], [f64; 9], [f64; 3]);
    /// `(block/name, fnv1a of map1 f32 bytes, fnv1a of map2, samples (x, y, map1, map2))`.
    pub type MapCase = (&'static str, u32, u32, [(i32, i32, f32, f32); 6]);

    // ---- calib3d_homography.py
    /// `exact/method0` (= `ransac`, `lmeds`, `ransac_thr_1`): the matrix scaled to `h[2][2] = 1`.
    #[rustfmt::skip]
    pub const H_EXACT: [f64; 9] = [1.100000031086699, 0.07999999134036534, 20.000000393609984, -0.049999987944943855, 0.9500000137277674, -12.000003857716429, 0.00020000009458556335, -0.00010000006815697422, 1.0];
    #[rustfmt::skip]
    pub const H_EXACT_RHO: [f64; 9] = [1.100000023841858, 0.08000008761882782, 19.999990463256836, -0.04999999329447746, 0.9500000476837158, -12.000003814697266, 0.0002000001259148121, -9.999991016229615e-5, 1.0];
    #[rustfmt::skip]
    pub const H_NOISY_METHOD0: [f64; 9] = [1.1016286835640556, 0.08049590360659499, 19.961630401115354, -0.04903365049586505, 0.9512653690350948, -12.311801631738437, 0.00020206693407268843, -9.587313296205931e-5, 1.0];
    #[rustfmt::skip]
    pub const H_OUTLIERS_RANSAC: [f64; 9] = [1.1019508791628343, 0.08097391814429285, 19.926759955857957, -0.04875065735571793, 0.9523815047596047, -12.516326262266068, 0.00020262430170187598, -9.471113760481897e-5, 1.0];
    #[rustfmt::skip]
    pub const H_OUTLIERS_RHO: [f64; 9] = [1.1045259237289429, 0.08394012600183487, 19.588281631469727, -0.048617199063301086, 0.9559783935546875, -12.749863624572754, 0.00020586994651239365, -8.419888035859913e-5, 1.0];
    #[rustfmt::skip]
    pub const H_FOUR_POINTS: [f64; 9] = [1.1008293312540545, 0.07887237414919517, 20.064164193377877, -0.05004180837107439, 0.951308120281803, -12.032093032536727, 0.00020000006381556916, -8.057917016910087e-5, 1.0];
    #[rustfmt::skip]
    pub const MASK_NOISY_RANSAC: [u8; 40] = [1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1];
    #[rustfmt::skip]
    pub const MASK_NOISY_RANSAC_THR_1: [u8; 40] = [1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1];
    #[rustfmt::skip]
    pub const MASK_OUTLIERS_RANSAC: [u8; 40] = [1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1];
    #[rustfmt::skip]
    pub const MASK_OUTLIERS_RHO: [u8; 40] = [1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
    #[rustfmt::skip]
    pub const H_INPUT_DIGESTS: [(&str, u32); 4] = [("src", 0x60ed8916), ("dst_exact", 0xd94b35b7), ("dst_noisy", 0xdab0c6d1), ("dst_outliers", 0xc7c26785)];

    // ---- calib3d_rodrigues.py
    /// `vec_to_mat`: `(name, rvec, rmat)`; `round_trip`: the vector OpenCV recovers from `rmat`.
    #[rustfmt::skip]
    pub const RODRIGUES: [RodriguesCase; 25] = [
        ("zero", [0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0]),
        ("tiny_1e-12", [1e-12, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 1.0, -1e-12, 0.0, 1e-12, 1.0], [0.0, 0.0, 0.0]),
        ("tiny_3e-9", [3e-9, -2e-9, 1e-9], [1.0, -1e-9, -2e-9, 1e-9, 1.0, -3e-9, 2e-9, 3e-9, 1.0], [0.0, 0.0, 0.0]),
        ("small_1e-6", [1e-6, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.9999999999995, -9.999999999998333e-7, 0.0, 9.999999999998333e-7, 0.9999999999995], [0.0, 0.0, 0.0]),
        ("small_2e-6", [2e-6, -1e-6, 3e-6], [0.999999999995, -3.0000009999867387e-6, -9.99997000016451e-7, 2.9999989999992616e-6, 0.9999999999935001, -2.0000014999859416e-6, 1.0000029999788827e-6, 1.9999985000047256e-6, 0.9999999999975], [0.0, 0.0, 0.0]),
        ("small_5e-6", [0.0, 5e-6, 0.0], [0.9999999999875, 0.0, 4.999999999979166e-6, 0.0, 1.0, 0.0, -4.999999999979166e-6, 0.0, 0.9999999999875], [0.0, 0.0, 0.0]),
        ("small_2e-5", [2e-5, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.9999999998, -1.9999999998666663e-5, 0.0, 1.9999999998666663e-5, 0.9999999998], [2.0000000827737027e-5, 0.0, 0.0]),
        ("small_1e-3", [0.001, -0.002, 0.0005], [0.9999978750009297, -0.0005009995620625941, -0.0019997482501098395, 0.0004989995629376356, 0.9999993750002735, -0.0010004991247814694, 0.0020002482498910792, 0.00099949912521899, 0.9999975000010938], [0.0009999999999684586, -0.0019999999999369173, 0.0004999999999842293]),
        ("general_a", [0.1, -0.2, 0.3], [0.9357548032779188, -0.3029327134026371, -0.18054007669439776, 0.28316496056507373, 0.9505806179060914, -0.12733457491763028, 0.21019170595074288, 0.06803131640494002, 0.9752903089530457], [0.0999999999999998, -0.19999999999999962, 0.2999999999999993]),
        ("general_b", [0.5, -1.25, 0.75], [0.13182651958247038, -0.7417948832927226, -0.6575424852095177, 0.23110460069417554, 0.6680513163109446, -0.7073175399445428, 0.9639566547686458, -0.05871788395327743, 0.2594990902321071], [0.5, -1.25, 0.75]),
        ("general_c", [-2.0, 1.0, 0.5], [0.6048204475307475, -0.7962739995355432, 0.01182978919407579, -0.46830056836606526, -0.3436104783954589, 0.8140186833266569, -0.64411707314488, -0.49787504135125477, -0.5807182098770105], [-2.0000000000000004, 1.0, 0.5000000000000002]),
        ("pi_x", [3.141592653589793, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, -1.0, -1.2246467991473532e-16, 0.0, 1.2246467991473532e-16, -1.0], [3.141592653589793, 0.0, 0.0]),
        ("pi_y", [0.0, 3.141592653589793, 0.0], [-1.0, 0.0, 1.2246467991473532e-16, 0.0, 1.0, 0.0, -1.2246467991473532e-16, 0.0, -1.0], [0.0, 3.141592653589793, 0.0]),
        ("pi_z", [0.0, 0.0, 3.141592653589793], [-1.0, -1.2246467991473532e-16, 0.0, 1.2246467991473532e-16, -1.0, 0.0, 0.0, 0.0, 1.0], [0.0, -0.0, 3.141592653589793]),
        ("pi_111", [1.8137993642342178, 1.8137993642342178, 1.8137993642342178], [-0.33333333333333315, 0.6666666666666667, 0.666666666666667, 0.666666666666667, -0.33333333333333315, 0.6666666666666667, 0.6666666666666667, 0.666666666666667, -0.33333333333333315], [1.8137993493330569, 1.8137993493330569, 1.8137993493330569]),
        ("pi_xz", [1.8849555921538759, 0.0, 2.5132741228718345], [-0.28, -9.797174393178826e-17, 0.96, 9.797174393178826e-17, -1.0, -7.347880794884119e-17, 0.96, 7.347880794884119e-17, 0.28000000000000025], [1.884955583213179, -0.0, 2.513274110950906]),
        ("pi_yz_neg", [0.0, 1.8849555921538759, -2.5132741228718345], [-1.0, 9.797174393178826e-17, 7.347880794884119e-17, -9.797174393178826e-17, -0.28, -0.96, -7.347880794884119e-17, -0.96, 0.28000000000000025], [0.0, 1.884955583213179, -2.513274110950906]),
        ("pi_xy_neg", [2.5132741228718345, -1.8849555921538759, 0.0], [0.28000000000000025, -0.96, -7.347880794884119e-17, -0.96, -0.28, -9.797174393178826e-17, 7.347880794884119e-17, 9.797174393178826e-17, -1.0], [2.513274110950906, -1.884955583213179, -0.0]),
        ("pi_neg_x", [-3.141592653589793, 0.0, 0.0], [1.0, -0.0, 0.0, 0.0, -1.0, 1.2246467991473532e-16, -0.0, -1.2246467991473532e-16, -1.0], [3.141592653589793, 0.0, 0.0]),
        ("near_pi_1e-6", [1.8849549921538757, 0.0, 2.5132733228718345], [-0.27999999999967995, -8.000000002096608e-7, 0.9599999999997599, 8.000000002096608e-7, -0.9999999999995, -6.000000001572456e-7, 0.9599999999997599, 6.000000001572456e-7, 0.2800000000001803], [1.8849549920605473, -1.570865647496898e-6, 2.5132733227470148]),
        ("near_pi_2e-5", [1.8849435921538757, 0.0, 2.5132581228718345], [-0.2799999998720003, -1.599999999878085e-5, 0.9599999999039995, 1.599999999878085e-5, -0.9999999998, -1.1999999999085638e-5, 0.9599999999039995, 1.1999999999085638e-5, 0.2800000000719993], [1.8849435921667024, 1.7439231469188174e-11, 2.5132581228889364]),
        ("near_pi_5e-4", [1.8846555921538757, 0.0, 2.5128741228718345], [-0.27999992000000173, -0.00039999998333356513, 0.9599999400000012, 0.00039999998333356513, -0.9999998750000026, -0.0002999999875001738, 0.9599999400000012, 0.0002999999875001738, 0.2800000449999992], [1.8846555921537942, 0.0, 2.5128741228717257]),
        ("near_pi_2e-5_neg", [0.0, -1.8849435921538757, 2.5132581228718345], [-0.9999999998, -1.599999999878085e-5, -1.1999999999085638e-5, 1.599999999878085e-5, -0.2799999998720003, -0.9599999999039995, 1.1999999999085638e-5, -0.9599999999039995, 0.2800000000719993], [1.7439231469188174e-11, -1.8849435921667024, 2.5132581228889364]),
        ("beyond_pi_4", [4.0, 0.0, 0.0], [0.9999999999999999, 0.0, 0.0, 0.0, -0.6536436208636119, 0.7568024953079282, 0.0, -0.7568024953079282, -0.6536436208636119], [-2.2831853071795867, 0.0, 0.0]),
        ("beyond_2pi", [6.783185307179586, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.8775825618903729, -0.4794255386042028, 0.0, 0.4794255386042028, 0.8775825618903729], [0.4999999999999999, 0.0, 0.0]),
    ];
    /// `mat_to_vec`: `(name, matrix, rvec)` for matrices that are not all rotations.
    #[rustfmt::skip]
    pub const MAT_TO_VEC: [(&str, [f64; 9], [f64; 3]); 7] = [
        ("rational_a", [0.8144796380090498, -0.047058823529411764, 0.5782805429864254, 0.3393665158371041, 0.8470588235294118, -0.4090497737556561, -0.47058823529411764, 0.5294117647058824, 0.7058823529411765], [0.5259483135718834, 0.587824585756811, 0.21656695264724612]),
        ("scaled_1.1", [0.8959276018099548, -0.05176470588235294, 0.636108597285068, 0.37330316742081454, 0.931764705882353, -0.44995475113122174, -0.5176470588235295, 0.5823529411764706, 0.7764705882352942], [0.5259483135718834, 0.587824585756811, 0.21656695264724618]),
        ("perturbed_01", [0.8144796380090498, -0.03705882352941176, 0.5782805429864254, 0.3393665158371041, 0.8470588235294118, -0.4090497737556561, -0.47058823529411764, 0.5294117647058824, 0.7058823529411765], [0.5275073852553837, 0.5892394053748148, 0.21188424905767891]),
        ("reflection_z", [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, -1.0], [1.1107207345395915, 1.1107207345395915, 0.0]),
        ("identity", [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0]),
        ("out_of_range_200", [200.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0]),
        ("transpose_rational_a", [0.8144796380090498, 0.3393665158371041, -0.47058823529411764, -0.047058823529411764, 0.8470588235294118, 0.5294117647058824, 0.5782805429864254, -0.4090497737556561, 0.7058823529411765], [-0.5259483135718834, -0.587824585756811, -0.21656695264724612]),
    ];

    // ---- calib3d_pnp.py
    #[rustfmt::skip]
    pub const RVEC_TRUE: [f64; 3] = [0.5259483135718834, 0.587824585756811, 0.21656695264724612];
    /// `img_24_dist5`: OpenCV's `projectPoints` of `obj_24` with the 5-coefficient distortion.
    #[rustfmt::skip]
    pub const IMG_24_DIST5: [(f32, f32); 24] = [(267.611572265625, 128.76443481445312), (212.10421752929688, 72.85309600830078), (343.5124206542969, 351.8479309082031), (286.2695617675781, 172.74090576171875), (440.7660827636719, 220.00686645507812), (428.39886474609375, 207.1029510498047), (239.1195831298828, 22.33562660217285), (330.182861328125, 350.1164855957031), (274.1310729980469, 58.35169982910156), (236.55538940429688, 13.16124153137207), (499.5076599121094, 260.44232177734375), (396.7220764160156, 214.44322204589844), (187.35145568847656, 97.10881042480469), (335.1037902832031, 177.0669708251953), (378.0534362792969, 107.39261627197266), (301.3079528808594, 103.94879913330078), (195.71339416503906, 226.4076690673828), (431.15264892578125, 318.08355712890625), (384.3016357421875, 356.1204528808594), (333.3912658691406, 261.28692626953125), (480.3107604980469, 94.4956283569336), (413.4571838378906, 169.2475128173828), (314.18682861328125, 272.3945007324219), (204.49398803710938, 192.87881469726562)];
    #[rustfmt::skip]
    pub const PNP_NONPLANAR_24_EXACT: ([f64; 3], [f64; 3]) = ([0.5259482980872597, 0.5878245794061878, 0.21656695701152026], [0.25000000913903214, -0.12500000608591996, 4.000000046435396]);
    #[rustfmt::skip]
    pub const PNP_NONPLANAR_24_NOISY: ([f64; 3], [f64; 3]) = ([0.525347642951197, 0.5886462602082522, 0.21685505601855323], [0.2497393287550799, -0.1250830272664002, 4.000802155542033]);
    #[rustfmt::skip]
    pub const PNP_NONPLANAR_24_DIST5: ([f64; 3], [f64; 3]) = ([0.5259482960832232, 0.587824609917828, 0.21656694953845898], [0.24999999352237032, -0.12499999442596199, 3.9999999528420425]);
    #[rustfmt::skip]
    pub const PNP_PLANAR_16_EXACT: ([f64; 3], [f64; 3]) = ([0.525948306967193, 0.5878246073430022, 0.21656696476154447], [0.2499999885637055, -0.12499999934343545, 4.000000008310727]);
    #[rustfmt::skip]
    pub const PNP_PLANAR_16_NOISY: ([f64; 3], [f64; 3]) = ([0.5248279132322816, 0.5895490501943567, 0.21781815741617025], [0.24908799992420896, -0.12498840489613869, 3.9995572881268973]);
    #[rustfmt::skip]
    pub const PNP_SQUARE_4: ([f64; 3], [f64; 3]) = ([0.5259484598190872, 0.5878245746365832, 0.21656695053284497], [0.24999997991434847, -0.12499998301353535, 3.999999834415193]);
    #[rustfmt::skip]
    pub const PNP_NONPLANAR_6: ([f64; 3], [f64; 3]) = ([0.5259483353150499, 0.5878246198881955, 0.21656693584392847], [0.2500000141515228, -0.12499998214065144, 3.999999860774462]);
    #[rustfmt::skip]
    pub const PNP_GUESS_GOOD_NOISY: ([f64; 3], [f64; 3]) = ([0.525347642949411, 0.5886462601988884, 0.21685505601823832], [0.24973932875550725, -0.12508302726778162, 4.000802155557222]);
    #[rustfmt::skip]
    pub const PNP_GUESS_GOOD_3_POINTS: ([f64; 3], [f64; 3]) = ([0.5259483664962288, 0.5878246776398929, 0.21656696223966737], [0.2500000222668328, -0.12499993938812398, 3.99999977076172]);
    #[rustfmt::skip]
    pub const PNP_EPNP_NONPLANAR_24_NOISY: ([f64; 3], [f64; 3]) = ([0.525925915333306, 0.5897759819350283, 0.21675955203243524], [0.24961231253143357, -0.12490790705304528, 3.998288991138441]);
    #[rustfmt::skip]
    pub const PNP_SQPNP_NONPLANAR_24_NOISY: ([f64; 3], [f64; 3]) = ([0.5256584828658282, 0.5886939026695502, 0.21678163470592848], [0.2497493228707142, -0.12510095234300317, 4.000791804577408]);
    #[rustfmt::skip]
    pub const PNP_P3P_NONPLANAR_4: ([f64; 3], [f64; 3]) = ([0.5259482113218195, 0.5878247064192543, 0.21656688742432775], [0.25000001035139363, -0.1250000050243026, 3.9999998020700684]);
    #[rustfmt::skip]
    pub const PNP_AP3P_NONPLANAR_4: ([f64; 3], [f64; 3]) = ([0.5259482073756467, 0.5878247350297069, 0.2165668849365387], [0.24999999297839112, -0.125000017100796, 3.999999983575859]);
    #[rustfmt::skip]
    pub const RANSAC_OUTLIERS_DEFAULT: ([f64; 3], [f64; 3], &[i32]) = ([0.5251758041053961, 0.5885078154681842, 0.21684814982587955], [0.2496757447623311, -0.12494674480453166, 4.000786474023406], &[0, 1, 2, 3, 4, 6, 7, 8, 10, 11, 12, 13, 14, 15, 17, 18, 20, 22]);
    #[rustfmt::skip]
    pub const RANSAC_EXACT_DEFAULT: ([f64; 3], [f64; 3], &[i32]) = ([0.5259482980976954, 0.5878245794911752, 0.2165669569808331], [0.2500000091703295, -0.1250000060405602, 4.00000004613812], &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]);
    #[rustfmt::skip]
    pub const RANSAC_NOISY_DEFAULT: ([f64; 3], [f64; 3], &[i32]) = ([0.5253476429507896, 0.5886462601950845, 0.2168550560183945], [0.24973932875574786, -0.1250830272683063, 4.000802155562231], &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]);
    #[rustfmt::skip]
    pub const RANSAC_NOISY_THR_1: ([f64; 3], [f64; 3], &[i32]) = ([0.5254683942247392, 0.5901072412217662, 0.21662538819008179], [0.2496957584558922, -0.12507472862100444, 4.001002869165908], &[0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 21, 22, 23]);
    #[rustfmt::skip]
    pub const RANSAC_OUTLIERS_DIST5: ([f64; 3], [f64; 3], &[i32]) = ([0.525948289029865, 0.5878246222963297, 0.21656695659223352], [0.24999999083145308, -0.12499999511688116, 3.9999999713399337], &[0, 1, 2, 3, 4, 6, 7, 8, 10, 11, 12, 13, 14, 15, 17, 18, 20, 22]);
    #[rustfmt::skip]
    pub const RANSAC_OUTLIERS_EPNP: ([f64; 3], [f64; 3], &[i32]) = ([0.525476411578607, 0.5893646295734025, 0.21677923807625096], [0.2495625934719624, -0.12487310016934573, 3.9990876375974094], &[0, 1, 2, 3, 4, 6, 7, 8, 10, 11, 12, 13, 14, 15, 17, 18, 20, 22]);
    #[rustfmt::skip]
    pub const RANSAC_PLANAR_NOISY: ([f64; 3], [f64; 3], &[i32]) = ([0.524827913232156, 0.5895490501951366, 0.2178181574161946], [0.24908799992414146, -0.1249884048960866, 3.9995572881262333], &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]);
    #[rustfmt::skip]
    pub const RANSAC_POINTS_4: ([f64; 3], [f64; 3], &[i32]) = ([0.5259482113218195, 0.5878247064192543, 0.21656688742432775], [0.25000001035139363, -0.1250000050243026, 3.9999998020700684], &[0, 1, 2, 3]);
    #[rustfmt::skip]
    pub const RANSAC_POINTS_5: ([f64; 3], [f64; 3], &[i32]) = ([0.525948342204243, 0.5878245739521136, 0.2165669515540713], [0.25000003618832506, -0.12499996519469861, 3.999999793430759], &[0, 1, 2, 3, 4]);
    #[rustfmt::skip]
    pub const RANSAC_POINTS_6: ([f64; 3], [f64; 3], &[i32]) = ([0.5259483353295846, 0.5878246199316731, 0.21656693583130174], [0.25000001414079975, -0.12499998211594768, 3.9999998605093907], &[0, 1, 2, 3, 4, 5]);
    #[rustfmt::skip]
    pub const PNP_INPUT_DIGESTS: [(&str, u32); 7] = [("obj_24", 0x2bdf0ca1), ("img_24_exact", 0x045ed18a), ("img_24_noisy", 0xbcabb6b5), ("img_24_outliers", 0x2ae7a25f), ("obj_16_planar", 0x0c202350), ("img_16_planar_exact", 0xa40d8f9c), ("img_16_planar_noisy", 0xa1c08ae9)];

    // ---- calib3d_undistort.py
    #[rustfmt::skip]
    pub const UNDISTORT_MAPS: [MapCase; 13] = [
        ("identity_r_same_k/zeros_5", 0x6a8d37c5, 0x3b9df3c5, [(0, 0, -1.7763568394002505e-14, 8.881784197001252e-15), (63, 0, 63.0, 8.881784197001252e-15), (0, 47, -1.7763568394002505e-14, 47.0), (63, 47, 63.0, 47.0), (32, 24, 32.0, 24.0), (17, 29, 17.0, 29.0)]),
        ("identity_r_same_k/k4", 0xffc58b92, 0x93ad8202, [(0, 0, 18.36400032043457, 14.04800033569336), (63, 0, 74.64012145996094, 11.096879959106445), (0, 47, 16.17637825012207, 56.98358154296875), (63, 47, 72.8114242553711, 54.545719146728516), (32, 24, 45.606056213378906, 34.42729187011719), (17, 29, 31.995893478393555, 39.67243194580078)]),
        ("identity_r_same_k/k5", 0xb341e63f, 0x1e434434, [(0, 0, 18.263999938964844, 13.972999572753906), (63, 0, 74.6030502319336, 11.062263488769531), (0, 47, 16.10988998413086, 56.9434814453125), (63, 47, 72.78981018066406, 54.52948760986328), (32, 24, 45.5582275390625, 34.39141845703125), (17, 29, 31.937318801879883, 39.63164138793945)]),
        ("identity_r_same_k/k8", 0x95a940b8, 0xf108673d, [(0, 0, 25.105634689331055, 19.10422706604004), (63, 0, 78.99796295166016, 15.166459083557129), (0, 47, 22.196855545043945, 60.614681243896484), (63, 47, 76.5339126586914, 57.341209411621094), (32, 24, 50.69105911254883, 38.24104309082031), (17, 29, 37.581668853759766, 43.562198638916016)]),
        ("identity_r_same_k/k12", 0x29955fb3, 0x2cdba0c4, [(0, 0, 25.280635833740234, 19.274227142333984), (63, 0, 79.13758850097656, 15.296080589294434), (0, 47, 22.352373123168945, 60.76194763183594), (63, 47, 76.65261840820312, 57.448673248291016), (32, 24, 50.83665466308594, 38.37720489501953), (17, 29, 37.73393249511719, 43.705787658691406)]),
        ("identity_r_same_k/k14", 0x41738e45, 0xb46a3bee, [(0, 0, 28.177406311035156, 21.418142318725586), (63, 0, 81.19840240478516, 17.204296112060547), (0, 47, 25.147916793823242, 62.41304016113281), (63, 47, 78.62348175048828, 58.90604019165039), (32, 24, 53.250431060791016, 40.16191864013672), (17, 29, 40.3375358581543, 45.4896354675293)]),
        ("identity_r_same_k/k8_strong", 0x84a7e6cf, 0x77afc33e, [(0, 0, 44.994285583496094, 39.2457160949707), (63, 0, 92.66407775878906, 32.2364616394043), (0, 47, 39.783599853515625, 75.53852844238281), (63, 47, 88.0786361694336, 69.385498046875), (32, 24, 66.11973571777344, 54.0447998046875), (17, 29, 54.264503479003906, 59.54481887817383)]),
        ("identity_r_same_k/none_dist", 0x6a8d37c5, 0x3b9df3c5, [(0, 0, -1.7763568394002505e-14, 8.881784197001252e-15), (63, 0, 63.0, 8.881784197001252e-15), (0, 47, -1.7763568394002505e-14, 47.0), (63, 47, 63.0, 47.0), (32, 24, 32.0, 24.0), (17, 29, 17.0, 29.0)]),
        ("rectified/k5", 0x50a2fcf1, 0x3dea3f27, [(0, 0, -59.367984771728516, 81.18362426757812), (63, 0, 0.40320250391960144, 59.69814682006836), (0, 47, -48.984458923339844, 126.87273406982422), (63, 47, 11.389596939086914, 105.83784484863281), (32, 24, -24.00051498413086, 93.60346221923828), (17, 29, -37.10681915283203, 103.54736328125)]),
        ("rectified/k8", 0x9c6f7eb2, 0x2df27d52, [(0, 0, -49.11543273925781, 85.48365020751953), (63, 0, 7.2583909034729, 63.57070541381836), (0, 47, -40.224544525146484, 129.564208984375), (63, 47, 16.9394474029541, 108.2539291381836), (32, 24, -16.340822219848633, 96.86849975585938), (17, 29, -28.788188934326172, 106.73168182373047)]),
        ("rectified/k14", 0x8bdd4862, 0x66764e67, [(0, 0, -44.912872314453125, 87.31461334228516), (63, 0, 10.463226318359375, 65.40802764892578), (0, 47, -36.413822174072266, 130.79978942871094), (63, 47, 19.786937713623047, 109.52000427246094), (32, 24, -12.843379974365234, 98.40386199951172), (17, 29, -25.096067428588867, 108.19898223876953)]),
        ("rectified/k5_r_only", 0xdf86ba63, 0x9aaf4156, [(0, 0, -33.13243103027344, 107.4356689453125), (63, 0, 20.183650970458984, 88.87188720703125), (0, 47, -23.149211883544922, 149.30931091308594), (63, 47, 30.63137435913086, 131.12547302246094), (32, 24, -1.1789366006851196, 119.40721893310547), (17, 29, -12.796591758728027, 128.2525177001953)]),
        ("rectified/k5_newk_only", 0x800b6a9b, 0xae404103, [(0, 0, 0.45786726474761963, -18.54631805419922), (63, 0, 63.80205535888672, -22.459386825561523), (0, 47, -2.3834996223449707, 28.1932315826416), (63, 47, 61.414974212646484, 24.883975982666016), (32, 24, 30.897104263305664, 3.2528443336486816), (17, 29, 15.541519165039062, 9.13141918182373)]),
    ];
    #[rustfmt::skip]
    pub const UNDISTORT_SIZE_1X1: (f32, f32) = (18.263999938964844, 13.972999572753906);

    // ---- calib3d_fundamental.py
    #[rustfmt::skip]
    pub const F_TRUE: [f64; 9] = [-7.059091241453389e-6, -1.3450645010977111e-5, 0.01245201835490656, -4.192088314037164e-6, 6.510840835370311e-6, 0.03890878483217297, -0.011999973261239262, -0.032144279757469384, 0.9985758856149395];
    #[rustfmt::skip]
    pub const F_EXACT_8POINT: [f64; 9] = [-7.059063395139202e-6, -1.3450564593911389e-5, 0.012451975504700957, -4.192105218449945e-6, 6.510837752745851e-6, 0.038908669544954054, -0.011999940953582406, -0.03214420243757654, 0.9985758935185255];
    #[rustfmt::skip]
    pub const F_EXACT_8POINT_8: [f64; 9] = [-7.058883564504143e-6, -1.3450099554169275e-5, 0.012451705986337384, -4.192065339803677e-6, 6.510799073705148e-6, 0.03890784324372191, -0.011999754413602612, -0.03214363587043109, 0.9985759495544129];
    #[rustfmt::skip]
    pub const F_EXACT_8POINT_10: [f64; 9] = [-7.059227208199699e-6, -1.3451024053809214e-5, 0.012452280943614358, -4.191779519245801e-6, 6.51094061292309e-6, 0.03890922306641541, -0.012000227037625967, -0.03214463844714504, 0.9985758506688868];
    #[rustfmt::skip]
    pub const F_EXACT_RANSAC: [f64; 9] = [-7.058274464438593e-6, -1.3448765630476248e-5, 0.012450612721175815, -4.192600997520569e-6, 6.510341881475785e-6, 0.03890561554964503, -0.011998757274567758, -0.03214184281983181, 0.9985761196798824];
    #[rustfmt::skip]
    pub const F_MASK_EXACT_RANSAC: [u8; 60] = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1];
    #[rustfmt::skip]
    pub const F_NOISY_8POINT: [f64; 9] = [-7.336031406402228e-6, -1.5079757371123835e-5, 0.012941144714364297, -3.0785463444407576e-6, 5.89640226604696e-6, 0.040215324733879294, -0.01236532215376938, -0.032732843711322554, 0.9984943233965693];
    #[rustfmt::skip]
    pub const F_NOISY_8POINT_8: [f64; 9] = [-2.257300368645822e-6, 4.3610784647184587e-7, 0.004329129563708789, -9.907511116860423e-6, 4.460782977954801e-6, 0.020189650657012455, -0.005298295867270887, -0.01860458985233248, 0.9995996367634169];
    #[rustfmt::skip]
    pub const F_NOISY_8POINT_10: [f64; 9] = [-4.345484832591092e-6, -5.304854410085309e-6, 0.007382723790322233, -1.0898166266397534e-5, 3.963375215717338e-6, 0.030554907458934667, -0.0073047375498713, -0.02528372950110493, 0.9991592799298574];
    #[rustfmt::skip]
    pub const F_MASK_NOISY_RANSAC: [u8; 60] = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1];
    #[rustfmt::skip]
    pub const F_OUTLIERS_8POINT: [f64; 9] = [1.7551215967693983e-6, 3.814868948616368e-6, -0.0012090332378377608, -3.3707772716126503e-6, 3.6565993030334976e-6, 0.0021049316225259128, -0.0014076258350307037, -0.004594151767054625, 0.9999855098041058];
    #[rustfmt::skip]
    pub const F_OUTLIERS_8POINT_8: [f64; 9] = [-4.2952960074081655e-6, 4.1388136479159175e-5, -0.009073721363307868, -5.497965568115261e-5, 2.725599728767514e-6, 0.013525909584780454, 0.014221862558371507, -0.018988812601798865, 0.9995858523459858];
    #[rustfmt::skip]
    pub const F_OUTLIERS_8POINT_10: [f64; 9] = [1.6031105785231608e-6, 1.0221240262734613e-5, -0.003176561076022131, -1.123873731972546e-5, 2.556998857134521e-6, 0.002508012335130819, 0.0015551448435226385, -0.005412510351845527, 0.9999759523859424];
    #[rustfmt::skip]
    pub const F_MASK_OUTLIERS_RANSAC: [u8; 60] = [0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0];
    #[rustfmt::skip]
    pub const F_MASK_NOISY_RANSAC_14: [u8; 14] = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1];
    #[rustfmt::skip]
    pub const F_MASK_OUTLIERS_RANSAC_CONF_0: [u8; 60] = [0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0];
    #[rustfmt::skip]
    pub const F_INPUT_DIGESTS: [(&str, u32); 5] = [("p1_exact", 0x889ca68a), ("p2_exact", 0x15cf6457), ("p1_noisy", 0x133f67fe), ("p2_noisy", 0xea312261), ("p2_outliers", 0x2685df22)];
}

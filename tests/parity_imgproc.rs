/*
 *  parity_imgproc.rs
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

//! OpenCV 4.10.0 parity pins for the `imgproc` module (`docs/parity/4.10.0/imgproc.md`).
//!
//! Every expected value in this file was produced by a script under `scripts/opencv_ref/`
//! (`imgproc_filter.py`, `imgproc_geom.py`, `imgproc_color_hist.py`, `imgproc_feature.py`)
//! run in `cpp_ref/.venv` (opencv-contrib-python 4.10.0.84, `cv2.ipp.setUseIPP(False)`,
//! Windows x86-64). The script and the key of each block are named in the comment above it.
//! The inputs are rebuilt here from the same formulas the scripts use.
//!
//! Tests that pass pin rows that already match OpenCV. Tests marked `#[ignore]` document a
//! `diverges` row; they fail when run (`cargo test --test parity_imgproc -- --ignored`) and
//! are the executable form of the measurement; the issue in each reason tracks the fix.

use purecv::core::types::{BorderTypes, Point2i, Scalar, Size, Size2i, TermCriteria, TermType};
use purecv::core::{Matrix, Point2f};
use purecv::imgproc::*;

use ocv::*;

const B101: BorderTypes = BorderTypes::Reflect101;

// ---------------------------------------------------------------------------------------
// Inputs (the same formulas as the scripts)
// ---------------------------------------------------------------------------------------

/// `((7*y + 13*x + 29*c) * 37) % 256`, the scripts' `pattern_u8`.
fn pattern_u8(rows: usize, cols: usize, channels: usize) -> Matrix<u8> {
    let mut data = Vec::with_capacity(rows * cols * channels);
    for y in 0..rows {
        for x in 0..cols {
            for c in 0..channels {
                data.push((((7 * y + 13 * x + 29 * c) * 37) % 256) as u8);
            }
        }
    }
    Matrix::from_vec(rows, cols, channels, data)
}

/// `((x*x + 3*y) * 37) % 256`, the scripts' `pattern_u8_quadratic`.
fn pattern_quad(rows: usize, cols: usize) -> Matrix<u8> {
    let mut data = Vec::with_capacity(rows * cols);
    for y in 0..rows {
        for x in 0..cols {
            data.push((((x * x + 3 * y) * 37) % 256) as u8);
        }
    }
    Matrix::from_vec(rows, cols, 1, data)
}

/// A bright square (rows and columns `x0..x1`) on a dark background, `square_u8`.
fn square_u8(size: usize, lo: u8, hi: u8, x0: usize, x1: usize) -> Matrix<u8> {
    let mut data = vec![lo; size * size];
    for y in x0..x1 {
        for x in x0..x1 {
            data[y * size + x] = hi;
        }
    }
    Matrix::from_vec(size, size, 1, data)
}

fn to_f32(m: &Matrix<u8>) -> Matrix<f32> {
    m.convert_to::<f32>().unwrap()
}

fn assert_close(actual: &[f32], expected: &[f32], tol: f32, what: &str) {
    assert_eq!(actual.len(), expected.len(), "{what}: length");
    for (i, (a, e)) in actual.iter().zip(expected).enumerate() {
        let t = tol * e.abs().max(1.0);
        assert!(
            (a - e).abs() <= t,
            "{what}[{i}]: purecv {a} vs OpenCV {e} (tol {t})"
        );
    }
}

fn assert_close64(actual: &[f64], expected: &[f64], tol: f64, what: &str) {
    assert_eq!(actual.len(), expected.len(), "{what}: length");
    for (i, (a, e)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (a - e).abs() <= tol,
            "{what}[{i}]: purecv {a} vs OpenCV {e} (tol {tol})"
        );
    }
}

fn anchor() -> Point2i {
    Point2i::new(-1, -1)
}

// ---------------------------------------------------------------------------------------
// get_gaussian_kernel / gaussian_blur (imgproc_filter.py, "get_gaussian_kernel",
// "gaussian_blur")
// ---------------------------------------------------------------------------------------

/// With sigma > 0, and with sigma <= 0 for sizes above 9, purecv's formula is OpenCV's.
#[test]
fn get_gaussian_kernel_sigma_positive_matches_opencv() {
    assert_close64(
        &get_gaussian_kernel(3, 1.5),
        &OCV_GET_GAUSSIAN_KERNEL_N3_SIGMA1_5,
        1e-12,
        "n3 s1.5",
    );
    assert_close64(
        &get_gaussian_kernel(5, 0.8),
        &OCV_GET_GAUSSIAN_KERNEL_N5_SIGMA0_8,
        1e-12,
        "n5 s0.8",
    );
    assert_close64(
        &get_gaussian_kernel(7, 2.0),
        &OCV_GET_GAUSSIAN_KERNEL_N7_SIGMA2,
        1e-12,
        "n7 s2",
    );
    assert_close64(
        &get_gaussian_kernel(11, 0.0),
        &OCV_GET_GAUSSIAN_KERNEL_N11_SIGMA0,
        1e-12,
        "n11 s0",
    );
    assert_eq!(get_gaussian_kernel(1, 0.0), vec![1.0]);
}

/// OpenCV 4.10.0: sigma <= 0 with n = 3, 5, 7, 9 returns fixed binomial tables
/// (`getGaussianKernelBitExact`), not the sampled Gaussian.
#[test]
#[ignore = "diverges from OpenCV 4.10.0, see #229"]
fn get_gaussian_kernel_sigma_zero_uses_opencv_tables() {
    assert_eq!(
        get_gaussian_kernel(3, 0.0),
        OCV_GET_GAUSSIAN_KERNEL_N3_SIGMA0.to_vec()
    );
    assert_eq!(
        get_gaussian_kernel(5, 0.0),
        OCV_GET_GAUSSIAN_KERNEL_N5_SIGMA0.to_vec()
    );
    assert_eq!(
        get_gaussian_kernel(7, 0.0),
        OCV_GET_GAUSSIAN_KERNEL_N7_SIGMA0.to_vec()
    );
    assert_eq!(
        get_gaussian_kernel(9, 0.0),
        OCV_GET_GAUSSIAN_KERNEL_N9_SIGMA0.to_vec()
    );
}

/// The float path with an explicit sigma matches (separable float convolution, reflect-101).
#[test]
fn gaussian_blur_f32_explicit_sigma_matches_opencv() {
    let src = to_f32(&pattern_u8(8, 8, 1));
    let dst = gaussian_blur(&src, Size2i::new(5, 5), 1.5, 0.0, B101).unwrap();
    assert_close(
        &dst.data,
        &OCV_GAUSSIAN_BLUR_F32_K5_SIGMA1_5,
        1e-5,
        "f32 k5 s1.5",
    );
}

/// OpenCV 4.10.0: 8U input runs the bit-exact `ufixedpoint16` path and uses the sigma-0 tables.
#[test]
#[ignore = "diverges from OpenCV 4.10.0, see #157 and #229"]
fn gaussian_blur_u8_matches_opencv_bit_exact_path() {
    let src = pattern_u8(8, 8, 1);
    let k3 = gaussian_blur(&src, Size2i::new(3, 3), 0.0, 0.0, B101).unwrap();
    assert_eq!(
        k3.data,
        OCV_GAUSSIAN_BLUR_U8_K3_SIGMA0.to_vec(),
        "k3 sigma 0"
    );
    let k5 = gaussian_blur(&src, Size2i::new(5, 5), 1.5, 0.0, B101).unwrap();
    assert_eq!(
        k5.data,
        OCV_GAUSSIAN_BLUR_U8_K5_SIGMA1_5.to_vec(),
        "k5 sigma 1.5"
    );
}

/// OpenCV: `ksize = (0, 0)` derives the size from sigma; purecv rejects it.
#[test]
#[ignore = "diverges from OpenCV 4.10.0, see #229"]
fn gaussian_blur_ksize_from_sigma() {
    let src = pattern_u8(8, 8, 1);
    let auto = gaussian_blur(&src, Size2i::new(0, 0), 1.0, 0.0, B101).unwrap();
    assert_eq!(
        auto.data,
        OCV_GAUSSIAN_BLUR_U8_K0_SIGMA1_AUTO_KSIZE.to_vec()
    );
}

/// OpenCV: sigma2 = 0 means sigma2 = sigma1 for any ksize. purecv clones the width kernel for
/// the vertical pass, so a 3x5 kernel (width 3, height 5) panics (index out of bounds).
#[test]
#[ignore = "diverges from OpenCV 4.10.0, see #229"]
fn gaussian_blur_tall_kernel_with_sigma2_0() {
    let src = pattern_u8(8, 8, 1);
    let rect = gaussian_blur(&to_f32(&src), Size2i::new(3, 5), 1.5, 0.0, B101).unwrap();
    assert_close(
        &rect.data,
        &OCV_GAUSSIAN_BLUR_F32_K3X5_SIGMA1_5_SIGMA2_0,
        1e-5,
        "3x5",
    );
}

/// The reverse case does not panic but is silently wrong: with a 5x3 kernel (width 5,
/// height 3) and sigma2 = 0 the vertical pass reads the first three taps of the 5-tap width
/// kernel, an asymmetric, unnormalised vertical filter (`f32_k5x3_sigma1.5_sigma2_0`).
#[test]
#[ignore = "diverges from OpenCV 4.10.0, see #229"]
fn gaussian_blur_wide_kernel_with_sigma2_0() {
    let src = pattern_u8(8, 8, 1);
    let rect = gaussian_blur(&to_f32(&src), Size2i::new(5, 3), 1.5, 0.0, B101).unwrap();
    assert_close(
        &rect.data,
        &OCV_GAUSSIAN_BLUR_F32_K5X3_SIGMA1_5_SIGMA2_0,
        1e-5,
        "5x3",
    );
}

// ---------------------------------------------------------------------------------------
// blur / box_filter / median_blur / bilateral_filter (imgproc_filter.py)
// ---------------------------------------------------------------------------------------

/// Float box filters match, normalised or not.
#[test]
fn blur_and_box_filter_f32_match_opencv() {
    let src = to_f32(&pattern_u8(8, 8, 1));
    let b = blur(&src, Size2i::new(3, 3), anchor(), B101).unwrap();
    assert_close(
        &b.data,
        &OCV_BLUR_BOX_FILTER_BLUR_F32_K3,
        1e-6,
        "blur f32 k3",
    );
    let u = box_filter(&src, Size2i::new(3, 3), anchor(), false, B101).unwrap();
    assert_close(
        &u.data,
        &OCV_BLUR_BOX_FILTER_BOX_U8_K3_UNNORMALIZED_DDEPTH_32F,
        1e-6,
        "box f32 unnormalized",
    );
}

/// OpenCV: `saturate_cast<uchar>` (round, then clamp). purecv truncates (1 low) and writes 0
/// for sums above 255.
#[test]
#[ignore = "diverges from OpenCV 4.10.0, see #157"]
fn blur_and_box_filter_u8_round_and_saturate() {
    let src = pattern_u8(8, 8, 1);
    let b = blur(&src, Size2i::new(3, 3), anchor(), B101).unwrap();
    assert_eq!(
        b.data,
        OCV_BLUR_BOX_FILTER_BLUR_U8_K3.to_vec(),
        "blur u8 k3"
    );
    let u = box_filter(&src, Size2i::new(3, 3), anchor(), false, B101).unwrap();
    assert_eq!(
        u.data,
        OCV_BLUR_BOX_FILTER_BOX_U8_K3_UNNORMALIZED.to_vec(),
        "unnormalized"
    );
}

/// OpenCV `medianBlur` pads with BORDER_REPLICATE; purecv with reflect-101.
#[test]
#[ignore = "diverges from OpenCV 4.10.0, see #230"]
fn median_blur_uses_replicate_border() {
    let src = pattern_u8(8, 8, 1);
    assert_eq!(
        median_blur(&src, 3).unwrap().data,
        OCV_MEDIAN_BLUR_U8_K3.to_vec()
    );
    assert_eq!(
        median_blur(&src, 5).unwrap().data,
        OCV_MEDIAN_BLUR_U8_K5.to_vec()
    );
}

/// OpenCV: circular window, L1 colour distance over the channels, rounded output.
#[test]
#[ignore = "diverges from OpenCV 4.10.0, see #231 and #157"]
fn bilateral_filter_matches_opencv() {
    let src = pattern_u8(8, 8, 1);
    let d = bilateral_filter(&src, 5, 50.0, 50.0, B101).unwrap();
    assert_eq!(d.data, OCV_BILATERAL_FILTER_U8_D5_SC50_SS50.to_vec());
}

// ---------------------------------------------------------------------------------------
// get_sobel_kernels / sobel / scharr / laplacian (imgproc_filter.py, "get_deriv_kernels",
// "derivatives")
// ---------------------------------------------------------------------------------------

#[test]
fn get_sobel_kernels_3_5_and_scharr_match_opencv() {
    for (ks, dx, dy, kx, ky) in [
        (
            3,
            1,
            0,
            OCV_GET_DERIV_KERNELS_K3_DX1_DY0_0.to_vec(),
            OCV_GET_DERIV_KERNELS_K3_DX1_DY0_1.to_vec(),
        ),
        (
            3,
            2,
            0,
            OCV_GET_DERIV_KERNELS_K3_DX2_DY0_0.to_vec(),
            OCV_GET_DERIV_KERNELS_K3_DX2_DY0_1.to_vec(),
        ),
        (
            3,
            1,
            1,
            OCV_GET_DERIV_KERNELS_K3_DX1_DY1_0.to_vec(),
            OCV_GET_DERIV_KERNELS_K3_DX1_DY1_1.to_vec(),
        ),
        (
            5,
            1,
            0,
            OCV_GET_DERIV_KERNELS_K5_DX1_DY0_0.to_vec(),
            OCV_GET_DERIV_KERNELS_K5_DX1_DY0_1.to_vec(),
        ),
        (
            5,
            2,
            0,
            OCV_GET_DERIV_KERNELS_K5_DX2_DY0_0.to_vec(),
            OCV_GET_DERIV_KERNELS_K5_DX2_DY0_1.to_vec(),
        ),
        (
            -1,
            1,
            0,
            OCV_GET_DERIV_KERNELS_SCHARR_DX1_DY0_0.to_vec(),
            OCV_GET_DERIV_KERNELS_SCHARR_DX1_DY0_1.to_vec(),
        ),
        (
            -1,
            0,
            1,
            OCV_GET_DERIV_KERNELS_SCHARR_DX0_DY1_0.to_vec(),
            OCV_GET_DERIV_KERNELS_SCHARR_DX0_DY1_1.to_vec(),
        ),
    ] {
        assert_eq!(
            get_sobel_kernels(ks, dx, dy),
            (kx, ky),
            "ksize {ks} dx {dx} dy {dy}"
        );
    }
}

/// OpenCV supports ksize 1 (a 3-tap derivative with no smoothing) and every odd size up to 31.
#[test]
#[ignore = "diverges from OpenCV 4.10.0, see #232"]
fn get_sobel_kernels_ksize_1_and_7() {
    assert_eq!(
        get_sobel_kernels(7, 1, 0),
        (
            OCV_GET_DERIV_KERNELS_K7_DX1_DY0_0.to_vec(),
            OCV_GET_DERIV_KERNELS_K7_DX1_DY0_1.to_vec()
        )
    );
    assert_eq!(
        get_sobel_kernels(1, 1, 0),
        (
            OCV_GET_DERIV_KERNELS_K1_DX1_DY0_0.to_vec(),
            OCV_GET_DERIV_KERNELS_K1_DX1_DY0_1.to_vec()
        )
    );
}

/// Float derivatives match for ksize 3 and 5, every order up to 2, and three borders.
#[test]
fn sobel_scharr_laplacian_f32_match_opencv() {
    let src = to_f32(&pattern_u8(8, 8, 1));
    let cases: [(&str, Matrix<f32>, &[f32]); 13] = [
        (
            "sobel dx1 k3",
            sobel(&src, 1, 0, 3, 1.0, 0.0, B101).unwrap(),
            &OCV_DERIVATIVES_SOBEL_F32_DX1_K3,
        ),
        (
            "sobel dy1 k3",
            sobel(&src, 0, 1, 3, 1.0, 0.0, B101).unwrap(),
            &OCV_DERIVATIVES_SOBEL_F32_DY1_K3,
        ),
        (
            "sobel dx1 k5",
            sobel(&src, 1, 0, 5, 1.0, 0.0, B101).unwrap(),
            &OCV_DERIVATIVES_SOBEL_F32_DX1_K5,
        ),
        (
            "sobel dx2 k3",
            sobel(&src, 2, 0, 3, 1.0, 0.0, B101).unwrap(),
            &OCV_DERIVATIVES_SOBEL_F32_DX2_K3,
        ),
        (
            "sobel dxdy k3",
            sobel(&src, 1, 1, 3, 1.0, 0.0, B101).unwrap(),
            &OCV_DERIVATIVES_SOBEL_F32_DX1_DY1_K3,
        ),
        (
            "sobel replicate",
            sobel(&src, 1, 0, 3, 1.0, 0.0, BorderTypes::Replicate).unwrap(),
            &OCV_DERIVATIVES_SOBEL_F32_DX1_K3_REPLICATE,
        ),
        (
            "sobel constant",
            sobel(&src, 1, 0, 3, 1.0, 0.0, BorderTypes::Constant).unwrap(),
            &OCV_DERIVATIVES_SOBEL_F32_DX1_K3_CONSTANT,
        ),
        (
            "scharr dx",
            scharr(&src, 1, 0, 1.0, 0.0, B101).unwrap(),
            &OCV_DERIVATIVES_SCHARR_F32_DX1,
        ),
        (
            "scharr dy",
            scharr(&src, 0, 1, 1.0, 0.0, B101).unwrap(),
            &OCV_DERIVATIVES_SCHARR_F32_DY1,
        ),
        (
            "laplacian k1",
            laplacian(&src, 1, 1.0, 0.0, B101).unwrap(),
            &OCV_DERIVATIVES_LAPLACIAN_F32_K1,
        ),
        (
            "laplacian k3",
            laplacian(&src, 3, 1.0, 0.0, B101).unwrap(),
            &OCV_DERIVATIVES_LAPLACIAN_F32_K3,
        ),
        (
            "laplacian k5",
            laplacian(&src, 5, 1.0, 0.0, B101).unwrap(),
            &OCV_DERIVATIVES_LAPLACIAN_F32_K5,
        ),
        (
            "laplacian k3 scale delta",
            laplacian(&src, 3, 0.5, 2.0, B101).unwrap(),
            &OCV_DERIVATIVES_LAPLACIAN_F32_K3_SCALE0_5_DELTA2,
        ),
    ];
    for (what, dst, expected) in cases {
        assert_close(&dst.data, expected, 1e-6, what);
    }
}

/// Integer outputs that need no rounding match (`i16`, no scale).
#[test]
fn sobel_i16_integer_kernel_matches_opencv() {
    let src = pattern_u8(8, 8, 1).convert_to::<i16>().unwrap();
    let dst = sobel(&src, 1, 0, 3, 1.0, 0.0, B101).unwrap();
    assert_eq!(
        dst.data,
        OCV_DERIVATIVES_SOBEL_U8_DX1_K3_DDEPTH_16S.to_vec()
    );
}

/// OpenCV supports ksize 1 and 7 for `Sobel` (and so for `Canny` and `Laplacian`).
#[test]
#[ignore = "diverges from OpenCV 4.10.0, see #232"]
fn sobel_ksize_1_and_7() {
    let src = to_f32(&pattern_u8(8, 8, 1));
    let k7 = sobel(&src, 1, 0, 7, 1.0, 0.0, B101).unwrap();
    assert_close(&k7.data, &OCV_DERIVATIVES_SOBEL_F32_DX1_K7, 1e-6, "k7");
    let k1 = sobel(&src, 1, 0, 1, 1.0, 0.0, B101).unwrap();
    assert_close(&k1.data, &OCV_DERIVATIVES_SOBEL_F32_DX1_K1, 1e-6, "k1");
}

/// OpenCV: `saturate_cast` to the destination depth (u8 values above 255 become 255; with a
/// fractional scale the value is rounded).
#[test]
#[ignore = "diverges from OpenCV 4.10.0, see #157"]
fn sobel_and_laplacian_integer_outputs_saturate() {
    let src = pattern_u8(8, 8, 1);
    let s = sobel(&src, 1, 0, 3, 1.0, 0.0, B101).unwrap();
    assert_eq!(
        s.data,
        OCV_DERIVATIVES_SOBEL_U8_DX1_K3_DDEPTH_SAME.to_vec(),
        "sobel u8"
    );
    let l = laplacian(&src, 1, 1.0, 0.0, B101).unwrap();
    assert_eq!(
        l.data,
        OCV_DERIVATIVES_LAPLACIAN_U8_K1_SAME.to_vec(),
        "laplacian u8"
    );
    let src16 = src.convert_to::<i16>().unwrap();
    let r = sobel(&src16, 1, 0, 3, 0.3, 0.5, B101).unwrap();
    assert_eq!(
        r.data,
        OCV_DERIVATIVES_SOBEL_U8_DX1_K3_SCALE0_3_DELTA0_5_16S.to_vec(),
        "i16 scaled"
    );
}

// ---------------------------------------------------------------------------------------
// get_structuring_element / erode / dilate / morphology_ex (imgproc_filter.py,
// "get_structuring_element", "morphology")
// ---------------------------------------------------------------------------------------

#[test]
fn get_structuring_element_rect_and_cross_match_opencv() {
    for (shape, prefix) in [(MorphShapes::Rect, "rect"), (MorphShapes::Cross, "cross")] {
        for ((w, h), expected) in [
            (
                (3, 3),
                if prefix == "rect" {
                    OCV_GET_STRUCTURING_ELEMENT_RECT_3X3.to_vec()
                } else {
                    OCV_GET_STRUCTURING_ELEMENT_CROSS_3X3.to_vec()
                },
            ),
            (
                (5, 5),
                if prefix == "rect" {
                    OCV_GET_STRUCTURING_ELEMENT_RECT_5X5.to_vec()
                } else {
                    OCV_GET_STRUCTURING_ELEMENT_CROSS_5X5.to_vec()
                },
            ),
            (
                (7, 3),
                if prefix == "rect" {
                    OCV_GET_STRUCTURING_ELEMENT_RECT_7X3.to_vec()
                } else {
                    OCV_GET_STRUCTURING_ELEMENT_CROSS_7X3.to_vec()
                },
            ),
            (
                (6, 4),
                if prefix == "rect" {
                    OCV_GET_STRUCTURING_ELEMENT_RECT_6X4.to_vec()
                } else {
                    OCV_GET_STRUCTURING_ELEMENT_CROSS_6X4.to_vec()
                },
            ),
        ] {
            let k = get_structuring_element(shape, Size::new(w, h), anchor()).unwrap();
            assert_eq!(k.data, expected, "{prefix} {w}x{h}");
        }
    }
    let c =
        get_structuring_element(MorphShapes::Cross, Size::new(5, 5), Point2i::new(1, 1)).unwrap();
    assert_eq!(
        c.data,
        OCV_GET_STRUCTURING_ELEMENT_CROSS_5X5_ANCHOR11.to_vec(),
        "cross anchor (1,1)"
    );
    let e = get_structuring_element(MorphShapes::Ellipse, Size::new(3, 3), anchor()).unwrap();
    assert_eq!(
        e.data,
        OCV_GET_STRUCTURING_ELEMENT_ELLIPSE_3X3.to_vec(),
        "ellipse 3x3"
    );
}

/// OpenCV rounds the half-width of each ellipse row (`saturate_cast<int>`); purecv truncates.
#[test]
#[ignore = "diverges from OpenCV 4.10.0, see #233"]
fn get_structuring_element_ellipse_rounds_row_width() {
    let e5 = get_structuring_element(MorphShapes::Ellipse, Size::new(5, 5), anchor()).unwrap();
    assert_eq!(
        e5.data,
        OCV_GET_STRUCTURING_ELEMENT_ELLIPSE_5X5.to_vec(),
        "5x5"
    );
    let e9 = get_structuring_element(MorphShapes::Ellipse, Size::new(9, 9), anchor()).unwrap();
    assert_eq!(
        e9.data,
        OCV_GET_STRUCTURING_ELEMENT_ELLIPSE_9X9.to_vec(),
        "9x9"
    );
}

/// Min/max filters match on u8 and f32, for every border, anchor and iteration count tried.
/// `BorderTypes::Constant` matches OpenCV's default `morphologyDefaultBorderValue()`.
#[test]
fn erode_dilate_match_opencv() {
    let src = pattern_u8(8, 8, 1);
    let k3 = get_structuring_element(MorphShapes::Rect, Size::new(3, 3), anchor()).unwrap();
    let kc = get_structuring_element(MorphShapes::Cross, Size::new(3, 3), anchor()).unwrap();
    let c = BorderTypes::Constant;
    let cases: [(&str, Matrix<u8>, Vec<u8>); 9] = [
        (
            "erode rect",
            erode(&src, &k3, anchor(), 1, c).unwrap(),
            OCV_MORPHOLOGY_ERODE_U8_RECT3.to_vec(),
        ),
        (
            "dilate rect",
            dilate(&src, &k3, anchor(), 1, c).unwrap(),
            OCV_MORPHOLOGY_DILATE_U8_RECT3.to_vec(),
        ),
        (
            "erode cross",
            erode(&src, &kc, anchor(), 1, c).unwrap(),
            OCV_MORPHOLOGY_ERODE_U8_CROSS3.to_vec(),
        ),
        (
            "erode iter2",
            erode(&src, &k3, anchor(), 2, c).unwrap(),
            OCV_MORPHOLOGY_ERODE_U8_RECT3_ITER2.to_vec(),
        ),
        (
            "erode iter2 replicate",
            erode(&src, &k3, anchor(), 2, BorderTypes::Replicate).unwrap(),
            OCV_MORPHOLOGY_ERODE_U8_RECT3_ITER2_REPLICATE.to_vec(),
        ),
        (
            "erode replicate",
            erode(&src, &k3, anchor(), 1, BorderTypes::Replicate).unwrap(),
            OCV_MORPHOLOGY_ERODE_U8_RECT3_REPLICATE.to_vec(),
        ),
        (
            "erode reflect101",
            erode(&src, &k3, anchor(), 1, B101).unwrap(),
            OCV_MORPHOLOGY_ERODE_U8_RECT3_REFLECT101.to_vec(),
        ),
        (
            "erode anchor (0,0)",
            erode(&src, &k3, Point2i::new(0, 0), 1, c).unwrap(),
            OCV_MORPHOLOGY_ERODE_U8_RECT3_ANCHOR00.to_vec(),
        ),
        (
            "erode iter0",
            erode(&src, &k3, anchor(), 0, c).unwrap(),
            OCV_MORPHOLOGY_ERODE_U8_RECT3_ITER0.to_vec(),
        ),
    ];
    for (what, dst, expected) in cases {
        assert_eq!(dst.data, expected, "{what}");
    }
    let f = erode(&to_f32(&src), &k3, anchor(), 1, c).unwrap();
    assert_close(&f.data, &OCV_MORPHOLOGY_ERODE_F32_RECT3, 0.0, "erode f32");
}

/// Multi-channel u8 erode matches, but panics with `--features simd` (separable SIMD path).
#[test]
#[cfg_attr(
    feature = "simd",
    ignore = "diverges from OpenCV 4.10.0 with --features simd (panics), see #234"
)]
fn erode_multichannel_matches_opencv() {
    let src = pattern_u8(6, 7, 3);
    let k3 = get_structuring_element(MorphShapes::Rect, Size::new(3, 3), anchor()).unwrap();
    let dst = erode(&src, &k3, anchor(), 1, BorderTypes::Constant).unwrap();
    assert_eq!(dst.data, OCV_MORPHOLOGY_ERODE_U8C3_RECT3.to_vec());
}

#[test]
fn morphology_ex_matches_opencv() {
    let src = pattern_u8(8, 8, 1);
    let k3 = get_structuring_element(MorphShapes::Rect, Size::new(3, 3), anchor()).unwrap();
    let c = BorderTypes::Constant;
    for (op, iterations, expected) in [
        (
            MorphTypes::Open,
            1,
            OCV_MORPHOLOGY_MORPH_OPEN_U8_RECT3.to_vec(),
        ),
        (
            MorphTypes::Close,
            1,
            OCV_MORPHOLOGY_MORPH_CLOSE_U8_RECT3.to_vec(),
        ),
        (
            MorphTypes::Gradient,
            1,
            OCV_MORPHOLOGY_MORPH_GRADIENT_U8_RECT3.to_vec(),
        ),
        (
            MorphTypes::TopHat,
            1,
            OCV_MORPHOLOGY_MORPH_TOPHAT_U8_RECT3.to_vec(),
        ),
        (
            MorphTypes::BlackHat,
            1,
            OCV_MORPHOLOGY_MORPH_BLACKHAT_U8_RECT3.to_vec(),
        ),
        (
            MorphTypes::Open,
            2,
            OCV_MORPHOLOGY_MORPH_OPEN_U8_RECT3_ITER2.to_vec(),
        ),
    ] {
        let dst = morphology_ex(&src, op, &k3, anchor(), iterations, c).unwrap();
        assert_eq!(dst.data, expected, "{op:?} x{iterations}");
    }
}

// ---------------------------------------------------------------------------------------
// threshold (imgproc_filter.py, "threshold")
// ---------------------------------------------------------------------------------------

#[test]
fn threshold_f32_matches_opencv() {
    let src = to_f32(&pattern_u8(8, 8, 1));
    for (ty, expected) in [
        (
            ThresholdTypes::THRESH_BINARY,
            OCV_THRESHOLD_F32_BINARY_T127_5_M200_1,
        ),
        (
            ThresholdTypes::THRESH_BINARY_INV,
            OCV_THRESHOLD_F32_BINARY_INV_T127_5_M200_1,
        ),
        (
            ThresholdTypes::THRESH_TRUNC,
            OCV_THRESHOLD_F32_TRUNC_T127_5_M200_1,
        ),
        (
            ThresholdTypes::THRESH_TOZERO,
            OCV_THRESHOLD_F32_TOZERO_T127_5_M200_1,
        ),
        (
            ThresholdTypes::THRESH_TOZERO_INV,
            OCV_THRESHOLD_F32_TOZERO_INV_T127_5_M200_1,
        ),
    ] {
        let (ret, dst) = threshold(&src, 127.5, 200.0, ty).unwrap();
        assert_eq!(ret, 127.5, "{ty:?} returned threshold");
        assert_close(&dst.data, &expected, 0.0, &format!("{ty:?}"));
    }
}

/// On u8 the five types write the same pixels as OpenCV (the returned threshold differs, see
/// the ignored test below).
#[test]
fn threshold_u8_pixels_match_opencv() {
    let src = pattern_u8(8, 8, 1);
    for (ty, expected) in [
        (
            ThresholdTypes::THRESH_BINARY,
            OCV_THRESHOLD_U8_BINARY_T127_5_M200_1.to_vec(),
        ),
        (
            ThresholdTypes::THRESH_BINARY_INV,
            OCV_THRESHOLD_U8_BINARY_INV_T127_5_M200_1.to_vec(),
        ),
        (
            ThresholdTypes::THRESH_TRUNC,
            OCV_THRESHOLD_U8_TRUNC_T127_5_M200_1.to_vec(),
        ),
        (
            ThresholdTypes::THRESH_TOZERO,
            OCV_THRESHOLD_U8_TOZERO_T127_5_M200_1.to_vec(),
        ),
        (
            ThresholdTypes::THRESH_TOZERO_INV,
            OCV_THRESHOLD_U8_TOZERO_INV_T127_5_M200_1.to_vec(),
        ),
    ] {
        let (_, dst) = threshold(&src, 127.5, 200.0, ty).unwrap();
        assert_eq!(dst.data, expected, "{ty:?}");
    }
}

/// A threshold below 0 sets every pixel to maxval. The u8 SIMD kernel clamps the threshold to
/// [0, 255] first, so with `--features simd` pixels equal to 0 stay 0.
#[test]
#[cfg_attr(
    feature = "simd",
    ignore = "diverges from OpenCV 4.10.0 with --features simd, see #235"
)]
fn threshold_u8_negative_thresh_matches_opencv() {
    let src = pattern_u8(8, 8, 1);
    let (_, dst) = threshold(&src, -1.0, 200.0, ThresholdTypes::THRESH_BINARY).unwrap();
    assert_eq!(dst.data, OCV_THRESHOLD_U8_BINARY_T_1_0_M200_1.to_vec());
}

/// OpenCV, integer depths: `thresh = cvFloor(thresh)` (and that value is returned),
/// `maxval = saturate_cast(cvRound(maxval))`.
#[test]
#[ignore = "diverges from OpenCV 4.10.0, see #235 and #157"]
fn threshold_integer_floors_thresh_and_rounds_maxval() {
    let src = pattern_u8(8, 8, 1);
    let (ret, _) = threshold(&src, 127.5, 200.0, ThresholdTypes::THRESH_BINARY).unwrap();
    assert_eq!(ret, 127.0, "returned threshold");
    for (maxval, expected) in [(254.6, 255u8), (1.5, 2), (300.0, 255)] {
        let (_, dst) = threshold(&src, 100.0, maxval, ThresholdTypes::THRESH_BINARY).unwrap();
        assert_eq!(*dst.data.iter().max().unwrap(), expected, "maxval {maxval}");
    }
    let i16src = Matrix::from_vec(8, 8, 1, src.data.iter().map(|&v| v as i16 - 100).collect());
    let (_, dst) = threshold(&i16src, 20.5, 90.6, ThresholdTypes::THRESH_BINARY).unwrap();
    assert_eq!(
        dst.data,
        OCV_THRESHOLD_I16_BINARY_T20_5_M90_6_1.to_vec(),
        "i16"
    );
}

// ---------------------------------------------------------------------------------------
// resize (imgproc_geom.py, "resize")
// ---------------------------------------------------------------------------------------

/// Matches where no fixed-point rounding is involved: 2x downscale (OpenCV switches to
/// INTER_AREA, an exact average), the identity, and two small cases.
#[test]
fn resize_u8_matches_opencv_on_exact_cases() {
    let src = pattern_u8(8, 8, 1);
    let src5 = pattern_u8(5, 5, 1);
    let src3 = pattern_u8(6, 7, 3);
    assert_eq!(
        resize(&src, Size::new(4, 4)).unwrap().data,
        OCV_RESIZE_U8_8X8_TO_4X4_LINEAR_IS_AREA.to_vec()
    );
    assert_eq!(
        resize(&src, Size::new(8, 8)).unwrap().data,
        OCV_RESIZE_U8_8X8_TO_8X8_IDENTITY.to_vec()
    );
    assert_eq!(
        resize(&src5, Size::new(3, 3)).unwrap().data,
        OCV_RESIZE_U8_5X5_TO_3X3_LINEAR.to_vec()
    );
    assert_eq!(
        resize(&src3, Size::new(4, 3)).unwrap().data,
        OCV_RESIZE_U8C3_TO_4X3_LINEAR.to_vec()
    );
}

/// OpenCV's 8U INTER_LINEAR uses 11-bit fixed-point coefficients; purecv's f64 path differs by 1.
#[test]
#[ignore = "diverges from OpenCV 4.10.0, see #236"]
fn resize_u8_linear_matches_opencv_fixed_point() {
    let src = pattern_u8(8, 8, 1);
    assert_eq!(
        resize(&src, Size::new(5, 5)).unwrap().data,
        OCV_RESIZE_U8_8X8_TO_5X5_LINEAR.to_vec()
    );
    let src5 = pattern_u8(5, 5, 1);
    assert_eq!(
        resize(&src5, Size::new(7, 7)).unwrap().data,
        OCV_RESIZE_U8_5X5_TO_7X7_LINEAR.to_vec()
    );
}

/// purecv rounds float outputs to integers (`interpolated.round()` for every `T`).
#[test]
#[ignore = "diverges from OpenCV 4.10.0, see #236"]
fn resize_f32_keeps_fractions() {
    let src = to_f32(&pattern_u8(8, 8, 1));
    let dst = resize(&src, Size::new(5, 5)).unwrap();
    assert_close(
        &dst.data,
        &OCV_RESIZE_F32_8X8_TO_5X5_LINEAR,
        1e-5,
        "f32 8x8 -> 5x5",
    );
}

// ---------------------------------------------------------------------------------------
// remap / warp_perspective (imgproc_geom.py, "remap", "warp_perspective")
// ---------------------------------------------------------------------------------------

fn map8(f: impl Fn(f32, f32) -> f32) -> Matrix<f32> {
    let mut data = Vec::with_capacity(64);
    for y in 0..8 {
        for x in 0..8 {
            data.push(f(x as f32, y as f32));
        }
    }
    Matrix::from_vec(8, 8, 1, data)
}

/// Integer map offsets (every sample on a pixel) match for every border mode.
#[test]
fn remap_integer_maps_match_opencv() {
    let src = pattern_u8(8, 8, 1);
    let mx = map8(|x, _| x - 2.0);
    let my = map8(|_, y| y + 2.0);
    let bv = Scalar::all(9u8);
    let lin = InterpolationFlags::Linear;
    let nn = InterpolationFlags::Nearest;
    let cases: [(&str, Matrix<u8>, Vec<u8>); 8] = [
        (
            "linear constant",
            remap(&src, &mx, &my, lin, BorderTypes::Constant, bv).unwrap(),
            OCV_REMAP_U8_LINEAR_OUT_CONSTANT.to_vec(),
        ),
        (
            "nearest constant",
            remap(&src, &mx, &my, nn, BorderTypes::Constant, bv).unwrap(),
            OCV_REMAP_U8_NEAREST_OUT_CONSTANT.to_vec(),
        ),
        (
            "linear replicate",
            remap(&src, &mx, &my, lin, BorderTypes::Replicate, bv).unwrap(),
            OCV_REMAP_U8_LINEAR_OUT_REPLICATE.to_vec(),
        ),
        (
            "nearest replicate",
            remap(&src, &mx, &my, nn, BorderTypes::Replicate, bv).unwrap(),
            OCV_REMAP_U8_NEAREST_OUT_REPLICATE.to_vec(),
        ),
        (
            "linear reflect101",
            remap(&src, &mx, &my, lin, B101, bv).unwrap(),
            OCV_REMAP_U8_LINEAR_OUT_REFLECT101.to_vec(),
        ),
        (
            "nearest reflect101",
            remap(&src, &mx, &my, nn, B101, bv).unwrap(),
            OCV_REMAP_U8_NEAREST_OUT_REFLECT101.to_vec(),
        ),
        (
            "linear reflect",
            remap(&src, &mx, &my, lin, BorderTypes::Reflect, bv).unwrap(),
            OCV_REMAP_U8_LINEAR_FRAC_REFLECT.to_vec(),
        ),
        (
            "linear wrap",
            remap(&src, &mx, &my, lin, BorderTypes::Wrap, bv).unwrap(),
            OCV_REMAP_U8_LINEAR_FRAC_WRAP.to_vec(),
        ),
    ];
    for (what, dst, expected) in cases {
        assert_eq!(dst.data, expected, "{what}");
    }
}

/// OpenCV rounds map coordinates with `cvRound` (ties to even): 0.5 -> 0, 2.5 -> 2.
#[test]
#[ignore = "diverges from OpenCV 4.10.0, see #190"]
fn remap_nearest_rounds_half_to_even() {
    let src = pattern_u8(8, 8, 1);
    let mx = map8(|x, _| x + 0.5);
    let my = map8(|_, y| y + 0.5);
    let dst = remap(
        &src,
        &mx,
        &my,
        InterpolationFlags::Nearest,
        BorderTypes::Constant,
        Scalar::all(9),
    )
    .unwrap();
    assert_eq!(dst.data, OCV_REMAP_U8_NEAREST_HALF_CONSTANT.to_vec());
}

/// OpenCV quantises map coordinates to 1/32 pixel (INTER_BITS = 5) and, for 8U, blends with
/// 15-bit fixed-point weights. purecv uses the exact fraction (and truncates on the scalar path).
#[test]
#[ignore = "diverges from OpenCV 4.10.0, see #237"]
fn remap_linear_quantises_coordinates() {
    let src = pattern_u8(8, 8, 1);
    let mx = map8(|x, _| x * 0.9 + 0.3);
    let my = map8(|_, y| y * 0.8 + 0.1);
    let u8dst = remap(
        &src,
        &mx,
        &my,
        InterpolationFlags::Linear,
        BorderTypes::Constant,
        Scalar::all(9),
    )
    .unwrap();
    assert_eq!(u8dst.data, OCV_REMAP_U8_LINEAR_FRAC_CONSTANT.to_vec(), "u8");
    let f = remap(
        &to_f32(&src),
        &mx,
        &my,
        InterpolationFlags::Linear,
        BorderTypes::Constant,
        Scalar::all(9.0),
    )
    .unwrap();
    assert_close(&f.data, &OCV_REMAP_F32_LINEAR_FRAC_CONSTANT, 1e-5, "f32");
}

fn m3(v: [f64; 9]) -> Matrix<f64> {
    Matrix::from_vec(3, 3, 1, v.to_vec())
}

#[test]
fn warp_perspective_identity_and_nearest_match_opencv() {
    let src = pattern_u8(8, 8, 1);
    let ident = m3([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
    let persp = m3([1.1, 0.1, -0.5, 0.05, 0.9, 0.3, 0.02, 0.01, 1.0]);
    let bv = Scalar::all(9u8);
    let sz = Size2i::new(8, 8);
    let lin = InterpolationFlags::Linear;
    let nn = InterpolationFlags::Nearest;
    let cases: [(&str, Matrix<u8>, Vec<u8>); 4] = [
        (
            "identity linear",
            warp_perspective(&src, &ident, sz, lin, BorderTypes::Constant, bv).unwrap(),
            OCV_WARP_PERSPECTIVE_U8_IDENTITY_LINEAR_CONSTANT.to_vec(),
        ),
        (
            "identity nearest",
            warp_perspective(&src, &ident, sz, nn, BorderTypes::Replicate, bv).unwrap(),
            OCV_WARP_PERSPECTIVE_U8_IDENTITY_NEAREST_REPLICATE.to_vec(),
        ),
        (
            "persp nearest constant",
            warp_perspective(&src, &persp, sz, nn, BorderTypes::Constant, bv).unwrap(),
            OCV_WARP_PERSPECTIVE_U8_PERSP_NEAREST_CONSTANT.to_vec(),
        ),
        (
            "persp nearest replicate",
            warp_perspective(&src, &persp, sz, nn, BorderTypes::Replicate, bv).unwrap(),
            OCV_WARP_PERSPECTIVE_U8_PERSP_NEAREST_REPLICATE.to_vec(),
        ),
    ];
    for (what, dst, expected) in cases {
        assert_eq!(dst.data, expected, "{what}");
    }
}

/// Same 1/32 quantisation and fixed-point blend as `remap` (OpenCV computes the coordinates
/// in fixed point, `INTER_TAB_SIZE / W`).
#[test]
#[ignore = "diverges from OpenCV 4.10.0, see #237"]
fn warp_perspective_linear_matches_opencv() {
    let src = pattern_u8(8, 8, 1);
    let persp = m3([1.1, 0.1, -0.5, 0.05, 0.9, 0.3, 0.02, 0.01, 1.0]);
    let dst = warp_perspective(
        &src,
        &persp,
        Size2i::new(8, 8),
        InterpolationFlags::Linear,
        BorderTypes::Constant,
        Scalar::all(9),
    )
    .unwrap();
    assert_eq!(
        dst.data,
        OCV_WARP_PERSPECTIVE_U8_PERSP_LINEAR_CONSTANT.to_vec()
    );
}

/// Half-pixel shift with INTER_NEAREST: OpenCV rounds the coordinate half to even.
#[test]
#[ignore = "diverges from OpenCV 4.10.0, see #190"]
fn warp_perspective_nearest_rounds_half_to_even() {
    let src = pattern_u8(8, 8, 1);
    let shift = m3([1.0, 0.0, 0.5, 0.0, 1.0, 0.25, 0.0, 0.0, 1.0]);
    let dst = warp_perspective(
        &src,
        &shift,
        Size2i::new(8, 8),
        InterpolationFlags::Nearest,
        BorderTypes::Constant,
        Scalar::all(9),
    )
    .unwrap();
    assert_eq!(
        dst.data,
        OCV_WARP_PERSPECTIVE_U8_SHIFT_NEAREST_CONSTANT.to_vec()
    );
}

// ---------------------------------------------------------------------------------------
// pyr_down / pyr_up / build_pyramid (imgproc_geom.py, "pyramids")
// ---------------------------------------------------------------------------------------

/// u8 pyr_down matches OpenCV's fixed-point `(x + 128) >> 8` for every size and border tried.
#[test]
fn pyr_down_u8_matches_opencv() {
    let cases: [(&str, Matrix<u8>, Vec<u8>); 6] = [
        (
            "8x8",
            pyr_down(&pattern_u8(8, 8, 1), None, B101).unwrap(),
            OCV_PYRAMIDS_DOWN_U8_8X8.to_vec(),
        ),
        (
            "9x7",
            pyr_down(&pattern_u8(9, 7, 1), None, B101).unwrap(),
            OCV_PYRAMIDS_DOWN_U8_9X7.to_vec(),
        ),
        (
            "c3",
            pyr_down(&pattern_u8(6, 7, 3), None, B101).unwrap(),
            OCV_PYRAMIDS_DOWN_U8C3.to_vec(),
        ),
        (
            "replicate",
            pyr_down(&pattern_u8(8, 8, 1), None, BorderTypes::Replicate).unwrap(),
            OCV_PYRAMIDS_DOWN_U8_8X8_REPLICATE.to_vec(),
        ),
        (
            "reflect",
            pyr_down(&pattern_u8(8, 8, 1), None, BorderTypes::Reflect).unwrap(),
            OCV_PYRAMIDS_DOWN_U8_8X8_REFLECT.to_vec(),
        ),
        (
            "dsize 3x3",
            pyr_down(&pattern_u8(8, 8, 1), Some(Size::new(3, 3)), B101).unwrap(),
            OCV_PYRAMIDS_DOWN_U8_8X8_DSIZE_3X3.to_vec(),
        ),
    ];
    for (what, dst, expected) in cases {
        assert_eq!(dst.data, expected, "{what}");
    }
}

#[test]
fn build_pyramid_u8_matches_opencv() {
    let levels = build_pyramid(&pattern_u8(8, 8, 1), 3, B101).unwrap();
    let expected = [
        OCV_PYRAMIDS_BUILD_PYRAMID_U8_8X8_LEVELS_0.to_vec(),
        OCV_PYRAMIDS_BUILD_PYRAMID_U8_8X8_LEVELS_1.to_vec(),
        OCV_PYRAMIDS_BUILD_PYRAMID_U8_8X8_LEVELS_2.to_vec(),
        OCV_PYRAMIDS_BUILD_PYRAMID_U8_8X8_LEVELS_3.to_vec(),
    ];
    assert_eq!(levels.len(), 4);
    for (i, (l, e)) in levels.iter().zip(expected.iter()).enumerate() {
        assert_eq!(&l.data, e, "level {i}");
    }
}

/// purecv rounds float pyramid outputs to integers.
#[test]
#[ignore = "diverges from OpenCV 4.10.0, see #238"]
fn pyr_down_f32_keeps_fractions() {
    let dst = pyr_down(&to_f32(&pattern_u8(8, 8, 1)), None, B101).unwrap();
    assert_close(&dst.data, &OCV_PYRAMIDS_DOWN_F32_8X8, 1e-6, "f32");
}

/// OpenCV's pyrUp has its own right/bottom edge rule (and supports odd destination sizes by
/// replicating the last row/column); purecv's reflect-101 of the source differs there.
#[test]
#[ignore = "diverges from OpenCV 4.10.0, see #238"]
fn pyr_up_u8_matches_opencv() {
    let dst = pyr_up(&pattern_u8(6, 6, 1), None, B101).unwrap();
    assert_eq!(dst.data, OCV_PYRAMIDS_UP_U8_6X6.to_vec());
}

/// OpenCV's `FixPtCast` (`(x + 128) >> 8`, an arithmetic shift) rounds -36.5 to -36; purecv's
/// `round()` gives -37.
#[test]
#[ignore = "diverges from OpenCV 4.10.0, see #190"]
fn pyr_down_i16_negative_ties_round_up() {
    let src = Matrix::from_vec(
        8,
        8,
        1,
        pattern_u8(8, 8, 1)
            .data
            .iter()
            .map(|&v| v as i16 - 128)
            .collect(),
    );
    let dst = pyr_down(&src, None, B101).unwrap();
    assert_eq!(dst.data, OCV_PYRAMIDS_DOWN_I16_SIGNED.to_vec());
}

// ---------------------------------------------------------------------------------------
// cvt_color (imgproc_color_hist.py, "cvt_color", "rgb2gray_exhaustive")
// ---------------------------------------------------------------------------------------

#[test]
fn cvt_color_gray_to_color_matches_opencv() {
    let gray = Matrix::from_vec(1, 6, 1, vec![0u8, 1, 127, 128, 254, 255]);
    for (code, expected) in [
        (
            ColorConversionCode::COLOR_GRAY2RGB,
            OCV_CVT_COLOR_GRAY2RGB.to_vec(),
        ),
        (
            ColorConversionCode::COLOR_GRAY2BGR,
            OCV_CVT_COLOR_GRAY2BGR.to_vec(),
        ),
        (
            ColorConversionCode::COLOR_GRAY2RGBA,
            OCV_CVT_COLOR_GRAY2RGBA.to_vec(),
        ),
        (
            ColorConversionCode::COLOR_GRAY2BGRA,
            OCV_CVT_COLOR_GRAY2BGRA.to_vec(),
        ),
    ] {
        assert_eq!(cvt_color(&gray, code).unwrap().data, expected, "{code:?}");
    }
}

/// OpenCV 4.10.0 `RGB2Gray<uchar>`: `(9798 R + 19235 G + 3735 B + 16384) >> 15`. The script
/// checks that this formula equals `cv2.cvtColor` on all 2^24 triples. purecv differs on 21,247
/// triples without `simd` (f32 coefficients) and on 2,231,218 with `simd` (8-bit 77/150/29).
#[test]
#[ignore = "diverges from OpenCV 4.10.0, see #191"]
fn cvt_color_rgb_to_gray_exhaustive() {
    let mut src = Matrix::<u8>::new(4096, 4096, 3);
    for i in 0..(1usize << 24) {
        src.data[i * 3] = (i >> 16) as u8;
        src.data[i * 3 + 1] = (i >> 8) as u8;
        src.data[i * 3 + 2] = i as u8;
    }
    let gray = cvt_color(&src, ColorConversionCode::COLOR_RGB2GRAY).unwrap();
    let mut mismatches = 0usize;
    for i in 0..(1usize << 24) {
        let (r, g, b) = (i >> 16 & 255, i >> 8 & 255, i & 255);
        let expected = ((r * 9798 + g * 19235 + b * 3735 + (1 << 14)) >> 15) as u8;
        mismatches += usize::from(gray.data[i] != expected);
    }
    assert_eq!(mismatches, 0);
}

// ---------------------------------------------------------------------------------------
// calc_hist / calc_back_project / compare_hist (imgproc_color_hist.py)
// ---------------------------------------------------------------------------------------

fn uni(lo: f32, hi: f32) -> RangeSpec {
    RangeSpec::Uniform(lo, hi)
}

#[test]
fn calc_hist_uniform_matches_opencv() {
    let img = pattern_u8(8, 8, 1);
    let img3 = pattern_u8(6, 7, 3);
    let mut mask = Matrix::<u8>::new(8, 8, 1);
    for y in 2..6 {
        for x in 1..7 {
            mask.set(y, x, 0, 1);
        }
    }
    let ch0 = Matrix::from_vec(6, 7, 1, img3.data.iter().step_by(3).copied().collect());
    let f = to_f32(&img);
    let fs = Matrix::from_vec(8, 8, 1, f.data.iter().map(|v| v / 255.0).collect());
    let ub = Matrix::from_vec(1, 5, 1, vec![0.0f32, 0.5, 1.0, 1.5, 2.0]);
    let r256 = || uni(0.0, 256.0);
    let cases: [(&str, Matrix<f32>, &[f32]); 11] = [
        (
            "8 bins",
            calc_hist(&[&img], &[0], None, &[8], &[r256()], false, None).unwrap(),
            &OCV_CALC_HIST_U8_8BINS_0_256,
        ),
        (
            "mask",
            calc_hist(&[&img], &[0], Some(&mask), &[8], &[r256()], false, None).unwrap(),
            &OCV_CALC_HIST_U8_8BINS_0_256_MASK,
        ),
        (
            "5 bins",
            calc_hist(&[&img], &[0], None, &[5], &[r256()], false, None).unwrap(),
            &OCV_CALC_HIST_U8_5BINS_0_256,
        ),
        (
            "50..200",
            calc_hist(&[&img], &[0], None, &[4], &[uni(50.0, 200.0)], false, None).unwrap(),
            &OCV_CALC_HIST_U8_4BINS_50_200,
        ),
        (
            "50.5..200.5",
            calc_hist(&[&img], &[0], None, &[4], &[uni(50.5, 200.5)], false, None).unwrap(),
            &OCV_CALC_HIST_U8_4BINS_50_5_200_5,
        ),
        (
            "c3 ch1",
            calc_hist(&[&img3], &[1], None, &[8], &[r256()], false, None).unwrap(),
            &OCV_CALC_HIST_U8C3_CH1_8BINS,
        ),
        (
            "2-D",
            calc_hist(
                &[&img3],
                &[0, 2],
                None,
                &[4, 4],
                &[r256(), r256()],
                false,
                None,
            )
            .unwrap(),
            &OCV_CALC_HIST_U8C3_CH0_CH2_4X4BINS,
        ),
        (
            "two images",
            calc_hist(
                &[&img3, &ch0],
                &[0, 3],
                None,
                &[2, 2],
                &[r256(), r256()],
                false,
                None,
            )
            .unwrap(),
            &OCV_CALC_HIST_TWO_IMAGES_CH0_CH3_2X2BINS,
        ),
        (
            "f32",
            calc_hist(&[&f], &[0], None, &[8], &[r256()], false, None).unwrap(),
            &OCV_CALC_HIST_F32_8BINS_0_256,
        ),
        (
            "f32 0..1",
            calc_hist(&[&fs], &[0], None, &[4], &[uni(0.0, 1.0)], false, None).unwrap(),
            &OCV_CALC_HIST_F32_4BINS_0_1_SCALED,
        ),
        (
            "upper bound",
            calc_hist(&[&ub], &[0], None, &[2], &[uni(0.0, 2.0)], false, None).unwrap(),
            &OCV_CALC_HIST_F32_VALUE_AT_UPPER_BOUND,
        ),
    ];
    for (what, h, expected) in cases {
        assert_close(&h.data, expected, 0.0, what);
    }
    let h8 = calc_hist(&[&img], &[0], None, &[8], &[r256()], false, None).unwrap();
    let acc = calc_hist(&[&img], &[0], None, &[8], &[r256()], true, Some(&h8)).unwrap();
    assert_close(
        &acc.data,
        &OCV_CALC_HIST_U8_ACCUMULATE_TWICE,
        0.0,
        "accumulate",
    );
}

#[test]
fn calc_back_project_matches_opencv() {
    let img = pattern_u8(8, 8, 1);
    let img3 = pattern_u8(6, 7, 3);
    let r256 = || uni(0.0, 256.0);
    let h8 = calc_hist(&[&img], &[0], None, &[8], &[r256()], false, None).unwrap();
    let h4 = calc_hist(&[&img], &[0], None, &[4], &[uni(50.0, 200.0)], false, None).unwrap();
    let h2 = calc_hist(
        &[&img3],
        &[0, 2],
        None,
        &[4, 4],
        &[r256(), r256()],
        false,
        None,
    )
    .unwrap();
    let cases: [(&str, Matrix<f32>, &[f32]); 6] = [
        (
            "scale 1",
            calc_back_project(&[&img], &[0], &[8], &h8, &[r256()], 1.0).unwrap(),
            &OCV_CALC_BACK_PROJECT_U8_8BINS_SCALE1,
        ),
        (
            "scale 0.5",
            calc_back_project(&[&img], &[0], &[8], &h8, &[r256()], 0.5).unwrap(),
            &OCV_CALC_BACK_PROJECT_U8_8BINS_SCALE0_5,
        ),
        (
            "scale 50",
            calc_back_project(&[&img], &[0], &[8], &h8, &[r256()], 50.0).unwrap(),
            &OCV_CALC_BACK_PROJECT_U8_8BINS_SCALE50_SATURATES,
        ),
        (
            "out of range",
            calc_back_project(&[&img], &[0], &[4], &h4, &[uni(50.0, 200.0)], 1.0).unwrap(),
            &OCV_CALC_BACK_PROJECT_U8_4BINS_50_200_OUT_OF_RANGE_ZERO,
        ),
        (
            "f32 src",
            calc_back_project(&[&to_f32(&img)], &[0], &[8], &h8, &[r256()], 1.0).unwrap(),
            &OCV_CALC_BACK_PROJECT_F32_8BINS_SCALE1,
        ),
        (
            "2-D",
            calc_back_project(&[&img3], &[0, 2], &[4, 4], &h2, &[r256(), r256()], 1.0).unwrap(),
            &OCV_CALC_BACK_PROJECT_U8C3_CH0_CH2_SCALE1,
        ),
    ];
    for (what, bp, expected) in cases {
        assert_close(&bp.data, expected, 0.0, what);
    }
}

/// OpenCV writes the back projection in the images' depth: u8 images get
/// `saturate_cast<uchar>(h * scale)` (rounded), f32 images get the unclamped product.
/// purecv always returns f32, unrounded, clamped to [0, 255].
#[test]
#[ignore = "diverges from OpenCV 4.10.0, see #240"]
fn calc_back_project_output_depth() {
    let img = pattern_u8(8, 8, 1);
    let h8 = calc_hist(&[&img], &[0], None, &[8], &[uni(0.0, 256.0)], false, None).unwrap();
    let u = calc_back_project(&[&img], &[0], &[8], &h8, &[uni(0.0, 256.0)], 0.3).unwrap();
    assert_close(
        &u.data,
        &OCV_CALC_BACK_PROJECT_U8_8BINS_SCALE0_3,
        0.0,
        "u8 rounds",
    );
    let f = calc_back_project(&[&to_f32(&img)], &[0], &[8], &h8, &[uni(0.0, 256.0)], 50.0).unwrap();
    assert_close(
        &f.data,
        &OCV_CALC_BACK_PROJECT_F32_8BINS_SCALE50,
        0.0,
        "f32 not clamped",
    );
}

#[test]
fn compare_hist_matches_opencv() {
    let h1 = Matrix::from_vec(6, 1, 1, vec![1.0f32, 2.0, 3.0, 0.0, 5.0, 0.5]);
    let h2 = Matrix::from_vec(6, 1, 1, vec![1.5f32, 0.0, 3.0, 1.0, 4.0, 0.5]);
    for (m, expected, expected_self) in [
        (HistCompMethods::Correl, 0.8117281367244044, 1.0),
        (HistCompMethods::ChiSqr, 2.45, 0.0),
        (HistCompMethods::Intersection, 8.5, 11.5),
        (HistCompMethods::Bhattacharyya, 0.37734127902696846, 0.0),
        (HistCompMethods::ChiSqrAlt, 6.4222222222222225, 0.0),
        (HistCompMethods::KullbackLeibler, 48.148248869463686, 0.0),
    ] {
        assert!(
            (compare_hist(&h1, &h2, m).unwrap() - expected).abs() <= 1e-12,
            "{m:?}"
        );
        assert!(
            (compare_hist(&h1, &h1, m).unwrap() - expected_self).abs() <= 1e-12,
            "{m:?} self"
        );
    }
}

/// OpenCV: HISTCMP_INTERSECT = 2, HISTCMP_BHATTACHARYYA = 3, HISTCMP_CHISQR_ALT = 4.
#[test]
#[ignore = "diverges from OpenCV 4.10.0, see #239"]
fn hist_comp_methods_discriminants_match_opencv() {
    assert_eq!(HistCompMethods::Intersection as i32, 2);
    assert_eq!(HistCompMethods::Bhattacharyya as i32, 3);
    assert_eq!(HistCompMethods::ChiSqrAlt as i32, 4);
}

// ---------------------------------------------------------------------------------------
// equalize_hist / Clahe (imgproc_color_hist.py, "equalize_hist", "clahe")
// ---------------------------------------------------------------------------------------

#[test]
fn equalize_hist_matches_opencv() {
    assert_eq!(
        equalize_hist(&pattern_quad(16, 16)).unwrap().data,
        OCV_EQUALIZE_HIST_U8_16X16.to_vec()
    );
    assert_eq!(
        equalize_hist(&pattern_u8(8, 8, 1)).unwrap().data,
        OCV_EQUALIZE_HIST_U8_8X8.to_vec()
    );
    assert_eq!(
        equalize_hist(&Matrix::from_vec(3, 3, 1, vec![77u8; 9]))
            .unwrap()
            .data,
        OCV_EQUALIZE_HIST_U8_CONSTANT.to_vec()
    );
    assert_eq!(
        equalize_hist(&Matrix::from_vec(1, 4, 1, vec![10u8, 10, 10, 200]))
            .unwrap()
            .data,
        OCV_EQUALIZE_HIST_U8_TWO_VALUES.to_vec()
    );
}

/// LUT value 42.5 (and 212.5): OpenCV's `saturate_cast<uchar>(float)` rounds to even.
#[test]
#[ignore = "diverges from OpenCV 4.10.0, see #190"]
fn equalize_hist_ties_round_half_to_even() {
    let src = Matrix::from_vec(1, 7, 1, vec![0u8, 10, 20, 30, 40, 50, 60]);
    assert_eq!(
        equalize_hist(&src).unwrap().data,
        OCV_EQUALIZE_HIST_U8_TIE_HALF_EVEN.to_vec()
    );
}

#[test]
fn clahe_matches_opencv() {
    let q32 = pattern_quad(32, 32);
    let q30 = pattern_quad(30, 29);
    let q16 = pattern_quad(16, 16);
    let c40 = create_clahe(40.0, Size2i::new(8, 8));
    assert_eq!(
        c40.apply_u8(&q32).unwrap().data,
        OCV_CLAHE_U8_32X32_CLIP40_0_TILES8X8.to_vec(),
        "32x32"
    );
    assert_eq!(
        c40.apply_u8(&q30).unwrap().data,
        OCV_CLAHE_U8_30X29_CLIP40_0_TILES8X8.to_vec(),
        "30x29 padded"
    );
    let c2 = Clahe::new(2.0, Size2i::new(4, 4));
    assert_eq!(
        c2.apply_u8(&q16).unwrap().data,
        OCV_CLAHE_U8_16X16_CLIP2_TILES4X4.to_vec(),
        "16x16"
    );
    let q16w = Matrix::from_vec(
        16,
        16,
        1,
        q16.data.iter().map(|&v| v as u16 * 257).collect(),
    );
    assert_eq!(
        c2.apply_u16(&q16w).unwrap().data,
        OCV_CLAHE_U16_16X16_CLIP2_TILES4X4.to_vec(),
        "u16"
    );
}

/// OpenCV pads *both* dimensions when either is not divisible by the tile grid; a divisible
/// dimension then gets a whole extra tile (`tilesY - h % tilesY` = `tilesY`).
#[test]
#[ignore = "diverges from OpenCV 4.10.0, see #241"]
fn clahe_pads_like_opencv() {
    let c = Clahe::new(0.0, Size2i::new(2, 2));
    assert_eq!(
        c.apply_u8(&pattern_quad(30, 29)).unwrap().data,
        OCV_CLAHE_U8_30X29_CLIP0_0_TILES2X2.to_vec()
    );
}

/// OpenCV computes the LUT and the bilinear blend in `float` and rounds half to even.
#[test]
#[ignore = "diverges from OpenCV 4.10.0, see #190"]
fn clahe_rounding_matches_opencv() {
    let c = Clahe::new(2.0, Size2i::new(4, 4));
    assert_eq!(
        c.apply_u8(&pattern_quad(32, 32)).unwrap().data,
        OCV_CLAHE_U8_32X32_CLIP2_0_TILES4X4.to_vec()
    );
}

// ---------------------------------------------------------------------------------------
// canny (imgproc_feature.py, "canny")
// ---------------------------------------------------------------------------------------

/// OpenCV: single-pixel edges (`m > left && m >= right` non-maximum suppression on CV_16S
/// Sobel with BORDER_REPLICATE), and border pixels can be edges.
#[test]
#[ignore = "diverges from OpenCV 4.10.0, see #242"]
fn canny_matches_opencv() {
    let sq = square_u8(16, 20, 220, 4, 12);
    assert_eq!(
        canny(&sq, 50.0, 150.0, 3, false).unwrap().data,
        OCV_CANNY_SQUARE_T50_T150.to_vec(),
        "square"
    );
    let p = pattern_u8(8, 8, 1);
    assert_eq!(
        canny(&p, 100.0, 200.0, 3, false).unwrap().data,
        OCV_CANNY_PATTERN_T100_T200.to_vec(),
        "pattern"
    );
}

// ---------------------------------------------------------------------------------------
// corner responses, good_features_to_track, corner_sub_pix (imgproc_feature.py)
// ---------------------------------------------------------------------------------------

/// OpenCV scales the Sobel derivatives by 1 / (2^(ksize-1) * block_size) (and 1/255 for 8U)
/// and sums the block without normalising, so its responses are smaller by a constant factor.
#[test]
#[ignore = "diverges from OpenCV 4.10.0, see #243"]
fn corner_responses_are_normalised_like_opencv() {
    let sq = to_f32(&square_u8(16, 20, 220, 4, 12));
    let h = corner_harris(&sq, 3, 3, 0.04, B101).unwrap();
    assert_close(&h.data, &OCV_CORNERS_HARRIS_F32_B3_K3_0_04, 1e-4, "harris");
    let m = corner_min_eigen_val(&sq, 3, 3, B101).unwrap();
    assert_close(&m.data, &OCV_CORNERS_MIN_EIGEN_F32_B3_K3, 1e-4, "min eigen");
    let ev = corner_eigen_vals_and_vecs(&sq, 3, 3, B101).unwrap();
    let eigenvalues: Vec<f32> = ev.data.chunks(6).flat_map(|p| [p[0], p[1]]).collect();
    let expected: Vec<f32> = OCV_CORNERS_EIGEN_VALS_VECS_F32_B3_K3
        .chunks(6)
        .flat_map(|p| [p[0], p[1]])
        .collect();
    assert_close(&eigenvalues, &expected, 1e-4, "eigenvalues");
}

/// The eigenvectors of `corner_eigen_vals_and_vecs` match OpenCV's up to sign wherever the
/// first eigenvalue is not negligible (the eigenvalues themselves differ by the constant factor
/// of `corner_responses_are_normalised_like_opencv`, #243).
#[test]
fn corner_eigen_vals_and_vecs_eigenvectors_match_opencv_up_to_sign() {
    let sq = to_f32(&square_u8(16, 20, 220, 4, 12));
    let ev = corner_eigen_vals_and_vecs(&sq, 3, 3, B101).unwrap();
    let expected = OCV_CORNERS_EIGEN_VALS_VECS_F32_B3_K3;
    assert_eq!(ev.data.len(), expected.len(), "eigen length");
    for (i, (a, e)) in ev.data.chunks(6).zip(expected.chunks(6)).enumerate() {
        if e[0].abs() > 1e-3f32 {
            for k in 2..6 {
                assert!(
                    (a[k].abs() - e[k].abs()).abs() < 1e-4,
                    "eigenvector px {i}: {a:?} vs {e:?}"
                );
            }
        }
    }
}

/// OpenCV computes D2x, D2y and Dxy with second-order Sobel kernels on the image and scales by
/// 1 / 2^(3*(ksize-1)); purecv differentiates the first derivatives again.
#[test]
#[ignore = "diverges from OpenCV 4.10.0, see #243"]
fn pre_corner_detect_matches_opencv() {
    let sq = to_f32(&square_u8(16, 20, 220, 4, 12));
    let d = pre_corner_detect(&sq, 3, B101).unwrap();
    assert_close(&d.data, &OCV_CORNERS_PRE_CORNER_F32_K3, 1e-4, "pre corner");
}

fn two_squares() -> Matrix<f32> {
    let mut data = vec![30u8; 24 * 24];
    for y in 3..10 {
        for x in 3..10 {
            data[y * 24 + x] = 200;
        }
    }
    for y in 13..21 {
        for x in 12..20 {
            data[y * 24 + x] = 220;
        }
    }
    to_f32(&Matrix::from_vec(24, 24, 1, data))
}

fn sorted_points(v: &[f32]) -> Vec<(i32, i32)> {
    let mut p: Vec<(i32, i32)> = v.chunks(2).map(|c| (c[0] as i32, c[1] as i32)).collect();
    p.sort();
    p
}

/// Same corners as OpenCV (the order of equal responses differs, see the ignored test).
#[test]
fn good_features_to_track_finds_opencv_corners() {
    let img = two_squares();
    let got = good_features_to_track(&img, 10, 0.01, 3.0, 3, false, 0.04).unwrap();
    let flat: Vec<f32> = got.iter().flat_map(|p| [p.x, p.y]).collect();
    assert_eq!(
        sorted_points(&flat),
        sorted_points(&OCV_GOOD_FEATURES_TO_TRACK_F32_MAX10_Q0_01_D3)
    );
}

/// OpenCV sorts equal responses by address, later pixels first (`greaterThanPtr`).
#[test]
#[ignore = "diverges from OpenCV 4.10.0, see #244"]
fn good_features_to_track_orders_ties_like_opencv() {
    let img = two_squares();
    let got = good_features_to_track(&img, 3, 0.01, 3.0, 3, false, 0.04).unwrap();
    let flat: Vec<f32> = got.iter().flat_map(|p| [p.x, p.y]).collect();
    assert_eq!(flat, OCV_GOOD_FEATURES_TO_TRACK_F32_MAX3_Q0_01_D3.to_vec());
}

/// OpenCV weights the window with a Gaussian, takes central differences of a sub-pixel patch
/// (`getRectSubPix`) and resets corners that move more than the window.
#[test]
#[ignore = "diverges from OpenCV 4.10.0, see #245"]
fn corner_sub_pix_matches_opencv() {
    let sq = to_f32(&square_u8(16, 20, 220, 4, 12));
    let mut corners = vec![
        Point2f::new(3.0, 3.0),
        Point2f::new(11.0, 4.0),
        Point2f::new(4.0, 12.0),
        Point2f::new(12.0, 12.0),
    ];
    let crit = TermCriteria::new(TermType::Both, 40, 0.001);
    corner_sub_pix(
        &sq,
        &mut corners,
        Size2i::new(2, 2),
        Size2i::new(-1, -1),
        crit,
    )
    .unwrap();
    let flat: Vec<f32> = corners.iter().flat_map(|p| [p.x, p.y]).collect();
    assert_close(&flat, &OCV_CORNER_SUB_PIX_F32_WIN2_ZERO_1, 1e-4, "refined");
}

// ---------------------------------------------------------------------------------------
// hough_lines / hough_lines_p / hough_circles (imgproc_feature.py)
// ---------------------------------------------------------------------------------------

fn lines_image() -> Matrix<u8> {
    let mut m = Matrix::<u8>::new(20, 20, 1);
    for i in 0..20 {
        m.set(5, i, 0, 255);
        m.set(i, 14, 0, 255);
        m.set(i, i, 0, 255);
    }
    m
}

fn line_pairs(v: &[f32]) -> Vec<(i32, i32)> {
    // (rho, theta in 1e-4 rad), sorted
    let mut p: Vec<(i32, i32)> = v
        .chunks(2)
        .map(|c| (c[0].round() as i32, (c[1] * 1e4).round() as i32))
        .collect();
    p.sort();
    p
}

/// Same (rho, theta) set as OpenCV for rho = 1; with a theta range the order matches too.
#[test]
fn hough_lines_finds_opencv_lines() {
    let img = lines_image();
    let pi = std::f64::consts::PI;
    let got: Vec<f32> = hough_lines(&img, 1.0, pi / 180.0, 15, 0.0, pi)
        .unwrap()
        .concat();
    assert_eq!(
        line_pairs(&got),
        line_pairs(&OCV_HOUGH_LINES_RHO1_THETA_PI180_T15)
    );
    let ranged: Vec<f32> = hough_lines(&img, 1.0, pi / 180.0, 15, 0.5, 2.0)
        .unwrap()
        .concat();
    assert_close(
        &ranged,
        &OCV_HOUGH_LINES_RHO1_THETA_PI180_T15_THETA_0_5_2_0,
        1e-6,
        "theta range",
    );
}

/// OpenCV orders equal votes by accumulator index (angle-major, `hough_cmp_gt`) and uses
/// `numrho = cvRound((2 * (w + h) + 1) / rho)`.
#[test]
#[ignore = "diverges from OpenCV 4.10.0, see #246"]
fn hough_lines_order_and_rho_bins_match_opencv() {
    let img = lines_image();
    let pi = std::f64::consts::PI;
    let got: Vec<f32> = hough_lines(&img, 1.0, pi / 180.0, 15, 0.0, pi)
        .unwrap()
        .concat();
    assert_close(&got, &OCV_HOUGH_LINES_RHO1_THETA_PI180_T15, 1e-6, "order");
    let rho2: Vec<f32> = hough_lines(&img, 2.0, pi / 90.0, 10, 0.0, pi)
        .unwrap()
        .concat();
    assert_close(&rho2, &OCV_HOUGH_LINES_RHO2_THETA_PI90_T10, 1e-6, "rho 2");
}

/// OpenCV seeds its shuffle with `RNG((uint64)-1)`, so `HoughLinesP` is deterministic.
#[test]
#[ignore = "diverges from OpenCV 4.10.0, see #247"]
fn hough_lines_p_is_deterministic_like_opencv() {
    let img = lines_image();
    let got = hough_lines_p(&img, 1.0, std::f64::consts::PI / 180.0, 10, 5.0, 2.0).unwrap();
    let flat: Vec<i32> = got.iter().flatten().copied().collect();
    assert_eq!(
        flat,
        OCV_HOUGH_LINES_P_RHO1_THETA_PI180_T10_LEN5_GAP2.to_vec()
    );
}

fn ring() -> Matrix<u8> {
    let mut m = Matrix::<u8>::new(40, 40, 1);
    for y in 0..40 {
        for x in 0..40 {
            let d = ((x as f64 - 20.0).powi(2) + (y as f64 - 19.0).powi(2)).sqrt();
            if (d - 9.0).abs() < 0.5 {
                m.set(y, x, 0, 255);
            }
        }
    }
    m
}

/// OpenCV's HOUGH_GRADIENT (Canny edges, fixed-point gradient voting, radius histogram) and
/// `maxRadius <= 0` meaning the image size; `dp <= 0` is an error. purecv panics on dp = 0.
#[test]
#[ignore = "diverges from OpenCV 4.10.0, see #248"]
fn hough_circles_matches_opencv() {
    let img = ring();
    let got: Vec<f32> = hough_circles(&img, 1.0, 10.0, 100.0, 10.0, 5, 15)
        .unwrap()
        .concat();
    assert_close(
        &got,
        &OCV_HOUGH_CIRCLES_GRADIENT_DP1_MD10_P100_P10_R5_15,
        1e-4,
        "r 5..15",
    );
    let unbounded: Vec<f32> = hough_circles(&img, 1.0, 10.0, 100.0, 10.0, 0, 0)
        .unwrap()
        .concat();
    assert_close(
        &unbounded,
        &OCV_HOUGH_CIRCLES_GRADIENT_DP1_MD10_P100_P10_R0_0,
        1e-4,
        "r 0..0",
    );
    assert!(
        hough_circles(&img, 0.0, 10.0, 100.0, 10.0, 0, 0).is_err(),
        "dp = 0"
    );
}

// ---------------------------------------------------------------------------------------
// OpenCV 4.10.0 outputs, generated from the JSON the scripts print (one constant per
// script key: OCV_<SECTION>_<KEY>). Kept on one line each, outside rustfmt. The values are
// OpenCV's outputs printed verbatim, so the two Clippy lints about float literals are
// allowed for this module only.
// ---------------------------------------------------------------------------------------

// Generated OpenCV 4.10.0 outputs: literals are copied, not approximations of constants.
#[allow(clippy::approx_constant, clippy::excessive_precision)]
mod ocv {
    #[rustfmt::skip]
    pub const OCV_BILATERAL_FILTER_U8_D5_SC50_SS50: [u8; 64] = [4, 216, 194, 165, 134, 103, 75, 61, 4, 217, 195, 167, 136, 105, 76, 62, 6, 219, 197, 169, 138, 107, 79, 64, 9, 222, 200, 172, 141, 110, 82, 67, 12, 225, 203, 175, 144, 113, 85, 70, 15, 228, 206, 178, 147, 116, 88, 73, 17, 231, 209, 180, 149, 118, 90, 76, 17, 232, 210, 182, 151, 120, 91, 77];
    #[rustfmt::skip]
    pub const OCV_BLUR_BOX_FILTER_BLUR_F32_K3: [f32; 64] = [152.0, 141.66667, 196.0, 165.0, 134.0, 103.0, 72.0, 61.666668, 153.0, 142.66667, 197.0, 166.0, 135.0, 104.0, 73.0, 62.666668, 156.0, 145.66667, 2e+02, 169.0, 138.0, 107.0, 76.0, 65.666664, 159.0, 148.66667, 203.0, 172.0, 141.0, 1.1e+02, 79.0, 68.666664, 162.0, 151.66667, 206.0, 175.0, 144.0, 113.0, 82.0, 71.666664, 165.0, 154.66667, 209.0, 178.0, 147.0, 116.0, 85.0, 74.666664, 168.0, 157.66667, 212.0, 181.0, 1.5e+02, 119.0, 88.0, 77.666664, 169.0, 158.66667, 213.0, 182.0, 151.0, 1.2e+02, 89.0, 78.666664];
    #[rustfmt::skip]
    pub const OCV_BLUR_BOX_FILTER_BLUR_U8_K3: [u8; 64] = [152, 142, 196, 165, 134, 103, 72, 62, 153, 143, 197, 166, 135, 104, 73, 63, 156, 146, 200, 169, 138, 107, 76, 66, 159, 149, 203, 172, 141, 110, 79, 69, 162, 152, 206, 175, 144, 113, 82, 72, 165, 155, 209, 178, 147, 116, 85, 75, 168, 158, 212, 181, 150, 119, 88, 78, 169, 159, 213, 182, 151, 120, 89, 79];
    #[rustfmt::skip]
    pub const OCV_BLUR_BOX_FILTER_BOX_U8_K3_UNNORMALIZED: [u8; 64] = [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255];
    #[rustfmt::skip]
    pub const OCV_BLUR_BOX_FILTER_BOX_U8_K3_UNNORMALIZED_DDEPTH_32F: [f32; 64] = [1368.0, 1275.0, 1764.0, 1485.0, 1206.0, 927.0, 648.0, 555.0, 1377.0, 1284.0, 1773.0, 1494.0, 1215.0, 936.0, 657.0, 564.0, 1404.0, 1311.0, 1.8e+03, 1521.0, 1242.0, 963.0, 684.0, 591.0, 1431.0, 1338.0, 1827.0, 1548.0, 1269.0, 9.9e+02, 711.0, 618.0, 1458.0, 1365.0, 1854.0, 1575.0, 1296.0, 1017.0, 738.0, 645.0, 1485.0, 1392.0, 1881.0, 1602.0, 1323.0, 1044.0, 765.0, 672.0, 1512.0, 1419.0, 1908.0, 1629.0, 1.35e+03, 1071.0, 792.0, 699.0, 1521.0, 1428.0, 1917.0, 1638.0, 1359.0, 1.08e+03, 801.0, 708.0];
    #[rustfmt::skip]
    pub const OCV_CALC_BACK_PROJECT_F32_8BINS_SCALE1: [f32; 64] = [8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0];
    #[rustfmt::skip]
    pub const OCV_CALC_BACK_PROJECT_F32_8BINS_SCALE50: [f32; 64] = [4e+02, 4e+02, 4e+02, 4e+02, 4e+02, 4e+02, 4e+02, 4e+02, 4e+02, 4e+02, 4e+02, 4e+02, 4e+02, 4e+02, 4e+02, 4e+02, 4e+02, 4e+02, 4e+02, 4e+02, 4e+02, 4e+02, 4e+02, 4e+02, 4e+02, 4e+02, 4e+02, 4e+02, 4e+02, 4e+02, 4e+02, 4e+02, 4e+02, 4e+02, 4e+02, 4e+02, 4e+02, 4e+02, 4e+02, 4e+02, 4e+02, 4e+02, 4e+02, 4e+02, 4e+02, 4e+02, 4e+02, 4e+02, 4e+02, 4e+02, 4e+02, 4e+02, 4e+02, 4e+02, 4e+02, 4e+02, 4e+02, 4e+02, 4e+02, 4e+02, 4e+02, 4e+02, 4e+02, 4e+02];
    #[rustfmt::skip]
    pub const OCV_CALC_BACK_PROJECT_U8C3_CH0_CH2_SCALE1: [f32; 42] = [6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0];
    #[rustfmt::skip]
    pub const OCV_CALC_BACK_PROJECT_U8_4BINS_50_200_OUT_OF_RANGE_ZERO: [f32; 64] = [0.0, 0.0, 1e+01, 1e+01, 8.0, 1e+01, 1e+01, 0.0, 0.0, 0.0, 1e+01, 1e+01, 8.0, 1e+01, 1e+01, 0.0, 0.0, 0.0, 0.0, 1e+01, 8.0, 1e+01, 1e+01, 0.0, 0.0, 0.0, 0.0, 1e+01, 8.0, 1e+01, 1e+01, 0.0, 0.0, 0.0, 0.0, 1e+01, 8.0, 1e+01, 1e+01, 1e+01, 0.0, 0.0, 0.0, 1e+01, 8.0, 1e+01, 1e+01, 1e+01, 0.0, 0.0, 0.0, 1e+01, 8.0, 1e+01, 1e+01, 1e+01, 0.0, 0.0, 0.0, 1e+01, 8.0, 1e+01, 1e+01, 1e+01];
    #[rustfmt::skip]
    pub const OCV_CALC_BACK_PROJECT_U8_8BINS_SCALE0_3: [f32; 64] = [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0];
    #[rustfmt::skip]
    pub const OCV_CALC_BACK_PROJECT_U8_8BINS_SCALE0_5: [f32; 64] = [4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0];
    #[rustfmt::skip]
    pub const OCV_CALC_BACK_PROJECT_U8_8BINS_SCALE1: [f32; 64] = [8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0];
    #[rustfmt::skip]
    pub const OCV_CALC_BACK_PROJECT_U8_8BINS_SCALE50_SATURATES: [f32; 64] = [255.0, 255.0, 255.0, 255.0, 255.0, 255.0, 255.0, 255.0, 255.0, 255.0, 255.0, 255.0, 255.0, 255.0, 255.0, 255.0, 255.0, 255.0, 255.0, 255.0, 255.0, 255.0, 255.0, 255.0, 255.0, 255.0, 255.0, 255.0, 255.0, 255.0, 255.0, 255.0, 255.0, 255.0, 255.0, 255.0, 255.0, 255.0, 255.0, 255.0, 255.0, 255.0, 255.0, 255.0, 255.0, 255.0, 255.0, 255.0, 255.0, 255.0, 255.0, 255.0, 255.0, 255.0, 255.0, 255.0, 255.0, 255.0, 255.0, 255.0, 255.0, 255.0, 255.0, 255.0];
    #[rustfmt::skip]
    pub const OCV_CALC_HIST_F32_4BINS_0_1_SCALED: [f32; 4] = [16.0, 16.0, 16.0, 16.0];
    #[rustfmt::skip]
    pub const OCV_CALC_HIST_F32_8BINS_0_256: [f32; 8] = [8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0];
    #[rustfmt::skip]
    pub const OCV_CALC_HIST_F32_VALUE_AT_UPPER_BOUND: [f32; 2] = [2.0, 2.0];
    #[rustfmt::skip]
    pub const OCV_CALC_HIST_TWO_IMAGES_CH0_CH3_2X2BINS: [f32; 4] = [18.0, 0.0, 0.0, 24.0];
    #[rustfmt::skip]
    pub const OCV_CALC_HIST_U8C3_CH0_CH2_4X4BINS: [f32; 16] = [0.0, 6.0, 0.0, 0.0, 0.0, 0.0, 6.0, 6.0, 6.0, 0.0, 0.0, 6.0, 6.0, 6.0, 0.0, 0.0];
    #[rustfmt::skip]
    pub const OCV_CALC_HIST_U8C3_CH1_8BINS: [f32; 8] = [6.0, 6.0, 1.0, 3.0, 7.0, 6.0, 6.0, 7.0];
    #[rustfmt::skip]
    pub const OCV_CALC_HIST_U8_4BINS_50_200: [f32; 4] = [1e+01, 1e+01, 8.0, 1e+01];
    #[rustfmt::skip]
    pub const OCV_CALC_HIST_U8_4BINS_50_5_200_5: [f32; 4] = [1e+01, 1e+01, 8.0, 11.0];
    #[rustfmt::skip]
    pub const OCV_CALC_HIST_U8_5BINS_0_256: [f32; 5] = [13.0, 12.0, 15.0, 12.0, 12.0];
    #[rustfmt::skip]
    pub const OCV_CALC_HIST_U8_8BINS_0_256: [f32; 8] = [8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0];
    #[rustfmt::skip]
    pub const OCV_CALC_HIST_U8_8BINS_0_256_MASK: [f32; 8] = [0.0, 0.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0];
    #[rustfmt::skip]
    pub const OCV_CALC_HIST_U8_ACCUMULATE_TWICE: [f32; 8] = [16.0, 16.0, 16.0, 16.0, 16.0, 16.0, 16.0, 16.0];
    #[rustfmt::skip]
    pub const OCV_CANNY_PATTERN_T100_T200: [u8; 64] = [255, 0, 0, 0, 0, 0, 0, 0, 255, 0, 0, 0, 0, 0, 0, 0, 255, 0, 0, 0, 0, 0, 0, 0, 255, 0, 0, 0, 0, 0, 0, 0, 255, 0, 0, 0, 0, 0, 0, 0, 255, 0, 0, 0, 0, 0, 0, 0, 255, 0, 0, 0, 0, 0, 0, 0, 255, 0, 0, 0, 0, 0, 0, 0];
    #[rustfmt::skip]
    pub const OCV_CANNY_SQUARE_T50_T150: [u8; 256] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 255, 255, 255, 255, 255, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 255, 0, 0, 0, 0, 0, 0, 255, 0, 0, 0, 0, 0, 0, 0, 255, 0, 0, 0, 0, 0, 0, 0, 255, 0, 0, 0, 0, 0, 0, 0, 255, 0, 0, 0, 0, 0, 0, 0, 255, 0, 0, 0, 0, 0, 0, 0, 255, 0, 0, 0, 0, 0, 0, 0, 255, 0, 0, 0, 0, 0, 0, 0, 255, 0, 0, 0, 0, 0, 0, 0, 255, 0, 0, 0, 0, 0, 0, 0, 255, 0, 0, 0, 0, 0, 0, 0, 255, 0, 0, 0, 0, 0, 0, 0, 255, 0, 0, 0, 0, 0, 0, 0, 255, 0, 0, 0, 0, 0, 0, 0, 0, 255, 255, 255, 255, 255, 255, 255, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
    #[rustfmt::skip]
    pub const OCV_CLAHE_U16_16X16_CLIP2_TILES4X4: [u16; 256] = [8192, 16384, 45055, 27648, 28672, 49151, 24576, 13312, 20480, 50175, 28672, 31744, 55295, 27648, 20480, 32768, 32768, 45055, 12288, 53247, 55295, 9216, 53247, 41983, 49151, 13312, 61439, 64511, 12288, 60415, 49151, 65535, 61439, 12288, 36863, 19456, 20480, 36863, 12288, 65535, 12288, 41983, 20480, 23552, 40959, 16384, 8192, 24576, 26624, 34815, 64511, 46079, 47103, 62463, 39935, 30720, 40447, 3328, 53247, 54783, 5120, 47615, 39935, 56319, 51199, 63487, 26624, 8704, 9216, 28672, 63487, 56831, 65535, 29696, 12288, 13824, 36864, 10240, 4096, 16384, 14336, 25600, 53247, 33536, 35327, 52735, 31744, 22528, 30208, 59647, 44031, 46847, 63999, 44031, 34815, 50175, 40959, 53247, 16384, 63487, 63487, 16384, 53247, 46079, 57343, 24576, 4096, 8192, 26624, 3072, 61439, 12288, 7168, 15360, 45055, 26368, 27136, 42239, 19456, 11264, 22016, 50431, 36863, 38911, 55295, 32768, 25600, 41983, 32768, 45055, 8192, 54271, 53247, 8704, 43007, 37375, 47103, 12288, 61439, 62975, 20480, 59391, 53247, 65535, 62463, 8192, 35839, 15360, 16384, 33024, 11264, 65535, 11776, 42239, 27648, 29952, 46591, 26112, 16384, 31744, 24576, 36863, 65535, 46079, 45055, 59391, 32768, 26624, 38911, 7168, 53247, 56319, 8192, 49151, 40959, 57343, 56319, 64511, 28672, 9472, 9728, 24320, 62463, 55295, 3584, 31999, 18432, 20224, 36351, 13568, 6144, 22528, 16384, 28672, 57343, 37887, 36863, 54783, 26624, 19968, 28672, 57343, 40959, 43007, 1024, 40447, 34816, 47103, 46079, 57343, 19456, 63999, 64511, 18176, 58367, 49151, 56831, 21760, 5120, 8704, 26624, 7424, 64511, 14336, 8192, 20480, 49151, 30720, 30720, 46079, 20480, 11264, 20480, 50175, 28672, 33792, 53247, 30720, 24576, 40959, 40959, 49151, 12288, 59391, 59391, 9216, 49151, 38911, 49151, 12288, 61439, 65535, 16384, 60415, 53247, 8192];
    #[rustfmt::skip]
    pub const OCV_CLAHE_U8_16X16_CLIP2_TILES4X4: [u8; 256] = [32, 64, 175, 108, 112, 191, 96, 52, 80, 195, 112, 124, 215, 108, 80, 128, 128, 175, 48, 207, 215, 36, 207, 163, 191, 52, 239, 251, 48, 235, 191, 255, 239, 48, 143, 76, 80, 143, 48, 255, 48, 163, 80, 92, 159, 64, 32, 96, 104, 135, 251, 179, 183, 243, 155, 120, 157, 13, 207, 213, 20, 185, 155, 219, 199, 247, 104, 34, 36, 112, 247, 221, 255, 116, 48, 54, 144, 40, 16, 64, 56, 100, 207, 131, 137, 205, 124, 88, 118, 232, 171, 182, 249, 171, 135, 195, 159, 207, 64, 247, 247, 64, 207, 179, 223, 96, 16, 32, 104, 12, 239, 48, 28, 60, 175, 103, 106, 164, 76, 44, 86, 196, 143, 151, 215, 128, 100, 163, 128, 175, 32, 211, 207, 34, 167, 145, 183, 48, 239, 245, 80, 231, 207, 255, 243, 32, 139, 60, 64, 129, 44, 255, 46, 164, 108, 117, 181, 102, 64, 124, 96, 143, 255, 179, 175, 231, 128, 104, 151, 28, 207, 219, 32, 191, 159, 223, 219, 251, 112, 37, 38, 95, 243, 215, 14, 124, 72, 79, 141, 53, 24, 88, 64, 112, 223, 147, 143, 213, 104, 78, 112, 223, 159, 167, 4, 157, 136, 183, 179, 223, 76, 249, 251, 71, 227, 191, 221, 85, 20, 34, 104, 29, 251, 56, 32, 80, 191, 120, 120, 179, 80, 44, 80, 195, 112, 132, 207, 120, 96, 159, 159, 191, 48, 231, 231, 36, 191, 151, 191, 48, 239, 255, 64, 235, 207, 32];
    #[rustfmt::skip]
    pub const OCV_CLAHE_U8_30X29_CLIP0_0_TILES2X2: [u8; 870] = [1, 37, 148, 76, 81, 159, 54, 26, 68, 182, 117, 127, 209, 109, 85, 135, 2, 197, 213, 46, 208, 189, 244, 119, 65, 85, 181, 94, 80, 110, 148, 4, 188, 192, 15, 166, 136, 178, 36, 228, 238, 65, 221, 195, 245, 112, 52, 68, 157, 64, 45, 101, 230, 174, 193, 36, 203, 190, 222, 4, 114, 44, 48, 125, 21, 246, 34, 146, 84, 95, 174, 74, 50, 102, 224, 164, 178, 13, 173, 156, 211, 83, 30, 49, 148, 58, 47, 76, 114, 225, 154, 159, 237, 133, 102, 143, 2, 194, 205, 30, 186, 161, 212, 77, 20, 34, 123, 29, 13, 67, 192, 141, 160, 4, 169, 158, 188, 225, 81, 10, 15, 94, 243, 214, 254, 112, 49, 60, 141, 41, 17, 68, 187, 131, 144, 235, 140, 123, 176, 48, 252, 17, 115, 26, 15, 44, 81, 192, 120, 125, 204, 99, 68, 108, 223, 160, 171, 252, 152, 128, 177, 43, 240, 255, 89, 251, 234, 32, 159, 107, 128, 226, 137, 125, 154, 192, 48, 232, 237, 60, 209, 178, 220, 78, 16, 27, 106, 7, 239, 33, 154, 97, 110, 199, 106, 88, 143, 16, 219, 238, 79, 248, 236, 10, 48, 159, 88, 94, 170, 64, 34, 73, 189, 128, 137, 218, 118, 96, 143, 11, 206, 222, 54, 218, 198, 254, 126, 73, 95, 189, 103, 90, 120, 159, 15, 199, 204, 26, 174, 143, 185, 45, 238, 248, 71, 230, 205, 254, 121, 62, 75, 165, 72, 53, 109, 237, 183, 204, 45, 215, 200, 232, 15, 125, 54, 60, 136, 30, 254, 40, 155, 95, 102, 183, 85, 61, 109, 232, 172, 186, 21, 182, 164, 221, 91, 38, 60, 156, 70, 55, 88, 125, 237, 166, 170, 245, 140, 108, 151, 12, 205, 214, 38, 196, 171, 220, 87, 29, 42, 132, 38, 20, 75, 202, 150, 171, 13, 180, 166, 199, 236, 93, 21, 25, 101, 251, 220, 7, 122, 60, 68, 150, 51, 27, 74, 198, 140, 153, 241, 150, 132, 186, 57, 6, 28, 123, 35, 22, 54, 93, 203, 132, 135, 212, 105, 74, 118, 233, 171, 179, 5, 162, 138, 186, 53, 250, 9, 98, 5, 241, 41, 169, 117, 138, 233, 147, 133, 166, 202, 58, 241, 244, 67, 216, 186, 229, 89, 26, 34, 116, 18, 248, 41, 165, 104, 119, 208, 115, 97, 152, 25, 227, 249, 88, 2, 242, 20, 58, 168, 98, 100, 177, 70, 41, 85, 200, 137, 145, 226, 129, 103, 152, 20, 215, 230, 65, 226, 207, 7, 136, 82, 103, 199, 112, 98, 132, 168, 24, 207, 211, 33, 182, 152, 195, 55, 247, 255, 82, 238, 213, 7, 131, 71, 85, 175, 82, 64, 118, 245, 193, 213, 55, 222, 209, 240, 24, 134, 63, 67, 144, 38, 8, 50, 166, 102, 111, 193, 95, 69, 119, 240, 183, 196, 31, 193, 174, 229, 100, 50, 70, 166, 78, 66, 97, 134, 243, 173, 177, 254, 149, 120, 162, 21, 212, 221, 49, 205, 180, 229, 96, 38, 52, 142, 49, 31, 84, 211, 160, 182, 22, 189, 177, 206, 243, 100, 28, 32, 110, 4, 229, 17, 132, 68, 77, 160, 61, 35, 85, 206, 149, 163, 252, 159, 141, 196, 68, 15, 37, 133, 46, 33, 63, 99, 210, 139, 144, 221, 116, 85, 129, 240, 178, 188, 15, 171, 147, 196, 63, 3, 18, 106, 15, 251, 52, 180, 125, 148, 241, 156, 144, 172, 210, 66, 250, 254, 78, 225, 196, 237, 97, 33, 44, 126, 26, 1, 52, 174, 113, 129, 216, 125, 105, 163, 35, 235, 2, 96, 10, 254, 28, 66, 176, 105, 111, 187, 82, 51, 93, 207, 146, 155, 235, 138, 112, 163, 29, 223, 238, 73, 235, 215, 17, 146, 90, 110, 208, 119, 107, 139, 176, 31, 214, 221, 42, 192, 163, 203, 64, 255, 10, 90, 246, 222, 17, 140, 81, 94, 186, 89, 72, 128, 255, 203, 220, 66, 230, 217, 249, 31, 144, 71, 79, 154, 48, 18, 58, 174, 113, 121, 201, 102, 80, 129, 250, 192, 205, 41, 202, 185, 237, 108, 60, 79, 178, 87, 74, 105, 144, 254, 183, 188, 10, 159, 130, 169, 29, 222, 230, 56, 211, 191, 237, 103, 49, 63, 152, 58, 41, 92, 218, 170, 192, 33, 201, 188, 214, 254, 112, 39, 42, 122, 15, 237, 23, 140, 80, 87, 168, 69, 47, 92, 212, 159, 174, 4, 169, 151, 205, 75, 26, 49, 143, 57, 44, 71, 112, 221, 152, 154, 230, 126, 92, 136, 250, 190, 198, 23, 180, 158, 204, 70, 14, 28, 115, 24, 3, 63, 189, 137, 158, 252, 167, 153, 183, 221, 79, 5, 10, 86, 234, 203, 245, 107, 46, 54, 135, 35, 12, 60, 184, 124, 139, 224, 136, 114, 174, 46, 246, 12, 105, 22, 6, 39, 79, 188, 118, 122, 197, 89, 58, 102, 216, 156, 165, 243, 148, 122, 172, 40, 233, 248, 83, 244, 222, 29, 155, 99, 121, 215, 133, 116, 152, 188, 42, 226, 230, 52, 199, 169, 210, 72, 11, 19, 98, 1, 231, 26, 151, 88, 102, 195, 97, 81, 138, 7, 209, 231, 72, 241, 225];
    #[rustfmt::skip]
    pub const OCV_CLAHE_U8_30X29_CLIP40_0_TILES8X8: [u8; 870] = [16, 64, 191, 104, 104, 195, 80, 40, 72, 195, 112, 124, 215, 108, 80, 136, 8, 215, 239, 72, 215, 183, 255, 128, 80, 108, 223, 128, 96, 128, 191, 48, 219, 223, 24, 207, 151, 191, 48, 239, 251, 48, 235, 191, 255, 128, 80, 112, 167, 80, 52, 112, 243, 183, 227, 48, 255, 223, 239, 48, 159, 72, 72, 136, 32, 255, 40, 159, 80, 92, 159, 64, 32, 104, 247, 179, 207, 28, 175, 147, 223, 96, 32, 60, 159, 80, 48, 104, 151, 251, 185, 189, 243, 143, 111, 151, 9, 207, 213, 20, 185, 155, 218, 88, 32, 52, 137, 34, 13, 72, 193, 138, 165, 12, 184, 161, 207, 247, 112, 30, 28, 108, 247, 221, 255, 114, 48, 54, 144, 40, 16, 66, 187, 134, 144, 237, 132, 114, 167, 46, 251, 16, 128, 26, 12, 56, 112, 223, 141, 143, 205, 120, 76, 116, 232, 171, 182, 249, 171, 135, 185, 46, 242, 255, 78, 243, 225, 32, 156, 102, 133, 219, 154, 134, 159, 223, 80, 247, 247, 56, 207, 179, 223, 92, 16, 32, 104, 12, 239, 40, 167, 84, 96, 179, 80, 60, 143, 12, 199, 223, 64, 243, 215, 24, 72, 171, 106, 110, 163, 64, 35, 80, 196, 143, 151, 215, 128, 100, 152, 12, 201, 211, 48, 205, 190, 255, 120, 62, 88, 187, 107, 86, 128, 167, 32, 205, 203, 28, 167, 145, 183, 46, 239, 245, 80, 231, 207, 253, 116, 58, 64, 161, 60, 46, 104, 237, 187, 207, 64, 217, 203, 231, 32, 143, 62, 64, 126, 40, 255, 44, 164, 108, 117, 181, 102, 64, 116, 233, 176, 191, 16, 183, 167, 231, 103, 50, 84, 171, 105, 82, 80, 143, 255, 167, 167, 231, 128, 104, 151, 24, 207, 219, 32, 191, 159, 223, 104, 28, 48, 132, 32, 12, 96, 223, 159, 187, 32, 207, 175, 203, 251, 96, 30, 34, 88, 243, 215, 6, 123, 72, 79, 141, 53, 24, 83, 203, 143, 159, 251, 153, 138, 203, 68, 10, 36, 135, 52, 28, 56, 96, 215, 133, 132, 211, 96, 72, 108, 223, 159, 167, 4, 157, 136, 185, 52, 253, 8, 106, 4, 245, 48, 177, 124, 139, 247, 143, 128, 163, 219, 76, 249, 251, 58, 227, 182, 221, 82, 20, 34, 104, 29, 251, 50, 169, 115, 131, 211, 124, 108, 171, 35, 229, 255, 80, 12, 243, 16, 80, 191, 104, 104, 167, 64, 32, 72, 191, 112, 132, 207, 120, 96, 159, 40, 219, 239, 68, 223, 203, 32, 147, 72, 88, 175, 96, 64, 135, 183, 28, 219, 225, 26, 183, 149, 189, 46, 243, 255, 68, 239, 215, 19, 140, 80, 96, 186, 88, 72, 135, 249, 185, 205, 44, 215, 191, 239, 24, 143, 66, 68, 151, 40, 12, 44, 155, 88, 98, 191, 92, 72, 122, 243, 189, 199, 40, 191, 175, 231, 106, 52, 68, 175, 72, 56, 88, 143, 255, 179, 187, 255, 175, 128, 165, 24, 215, 228, 42, 221, 187, 241, 106, 51, 68, 144, 54, 35, 96, 220, 165, 197, 28, 215, 191, 191, 255, 112, 32, 40, 112, 16, 239, 24, 143, 64, 80, 151, 60, 32, 96, 231, 155, 175, 255, 151, 128, 207, 80, 16, 44, 143, 64, 32, 60, 108, 207, 148, 159, 220, 128, 95, 136, 248, 191, 200, 10, 178, 151, 208, 72, 10, 24, 113, 12, 249, 56, 178, 124, 152, 255, 171, 147, 167, 207, 72, 251, 255, 86, 231, 205, 239, 98, 32, 40, 132, 30, 8, 54, 171, 114, 120, 215, 112, 96, 151, 32, 239, 6, 120, 16, 255, 20, 76, 187, 111, 118, 184, 104, 60, 100, 216, 155, 167, 235, 158, 124, 171, 30, 224, 235, 59, 225, 208, 16, 143, 92, 126, 215, 147, 124, 128, 191, 48, 219, 223, 36, 191, 163, 207, 76, 255, 16, 88, 251, 223, 24, 152, 68, 80, 163, 64, 44, 128, 255, 191, 219, 64, 239, 207, 247, 40, 139, 79, 88, 146, 52, 21, 64, 178, 124, 133, 199, 114, 88, 140, 255, 189, 199, 36, 193, 178, 243, 109, 52, 79, 179, 97, 74, 96, 136, 255, 179, 183, 14, 159, 134, 167, 26, 215, 225, 64, 219, 199, 245, 108, 50, 56, 153, 52, 38, 96, 227, 175, 193, 48, 201, 187, 199, 255, 112, 37, 46, 115, 36, 245, 28, 142, 80, 95, 165, 92, 60, 112, 229, 172, 187, 12, 179, 163, 227, 94, 36, 65, 147, 83, 62, 48, 112, 223, 143, 151, 223, 128, 96, 136, 255, 175, 195, 16, 183, 159, 223, 104, 28, 48, 132, 32, 12, 96, 215, 144, 163, 255, 179, 151, 179, 227, 76, 15, 24, 79, 243, 206, 249, 112, 48, 64, 130, 47, 20, 83, 201, 133, 151, 241, 143, 128, 203, 62, 249, 16, 108, 36, 8, 48, 88, 199, 122, 128, 193, 96, 64, 104, 215, 151, 167, 243, 153, 128, 185, 48, 237, 255, 84, 247, 231, 40, 169, 100, 122, 215, 140, 112, 167, 211, 68, 240, 241, 47, 203, 168, 211, 73, 12, 25, 94, 9, 227, 35, 157, 85, 96, 190, 94, 78, 147, 12, 195, 225, 60, 239, 219];
    #[rustfmt::skip]
    pub const OCV_CLAHE_U8_32X32_CLIP2_0_TILES4X4: [u8; 1024] = [8, 48, 159, 88, 92, 170, 61, 28, 72, 188, 122, 129, 211, 110, 86, 132, 4, 193, 212, 48, 203, 181, 244, 116, 68, 88, 183, 98, 84, 147, 24, 231, 124, 159, 12, 195, 199, 22, 173, 144, 183, 42, 234, 240, 60, 226, 198, 246, 110, 54, 71, 152, 64, 44, 99, 227, 173, 198, 39, 212, 199, 255, 139, 88, 227, 12, 128, 52, 56, 137, 26, 250, 36, 155, 86, 94, 175, 76, 52, 98, 225, 162, 174, 16, 163, 148, 210, 82, 32, 57, 148, 64, 48, 120, 247, 203, 88, 128, 231, 163, 171, 242, 140, 108, 149, 8, 200, 207, 28, 188, 164, 212, 76, 22, 36, 120, 28, 12, 69, 189, 139, 165, 7, 173, 163, 223, 108, 52, 195, 231, 92, 16, 24, 102, 249, 216, 255, 119, 51, 58, 143, 43, 20, 68, 183, 129, 142, 235, 131, 117, 171, 50, 251, 20, 120, 28, 12, 76, 219, 167, 50, 90, 198, 133, 137, 206, 103, 73, 111, 228, 164, 171, 250, 156, 129, 177, 42, 240, 255, 89, 248, 232, 33, 159, 110, 130, 226, 140, 128, 190, 68, 16, 159, 196, 52, 237, 241, 63, 211, 181, 222, 82, 18, 26, 105, 12, 240, 34, 152, 95, 108, 198, 103, 88, 140, 16, 220, 240, 80, 249, 236, 44, 178, 129, 12, 50, 165, 92, 96, 172, 66, 35, 76, 191, 128, 137, 216, 122, 96, 144, 12, 203, 221, 57, 216, 196, 253, 127, 76, 97, 191, 110, 93, 154, 32, 239, 126, 163, 18, 199, 203, 26, 175, 144, 186, 47, 238, 246, 72, 230, 204, 253, 122, 61, 78, 166, 74, 55, 110, 237, 185, 208, 46, 218, 203, 10, 141, 96, 234, 16, 127, 54, 58, 137, 30, 253, 43, 157, 96, 103, 180, 90, 62, 110, 231, 172, 188, 24, 184, 166, 221, 94, 40, 65, 158, 74, 62, 122, 254, 210, 85, 125, 237, 163, 167, 248, 138, 107, 150, 14, 201, 210, 40, 194, 170, 218, 92, 30, 43, 138, 36, 21, 78, 203, 152, 175, 14, 180, 168, 233, 113, 67, 196, 236, 86, 18, 22, 99, 250, 219, 8, 122, 61, 70, 144, 54, 30, 79, 197, 143, 158, 245, 153, 136, 191, 59, 8, 30, 125, 36, 24, 84, 226, 172, 48, 84, 199, 124, 131, 210, 102, 74, 114, 231, 164, 172, 8, 160, 138, 188, 52, 252, 13, 107, 4, 242, 41, 174, 120, 140, 238, 144, 128, 191, 80, 28, 160, 199, 53, 243, 247, 62, 214, 185, 224, 90, 26, 36, 116, 22, 248, 44, 163, 109, 124, 212, 118, 101, 155, 26, 231, 250, 91, 7, 244, 53, 185, 133, 17, 54, 164, 91, 99, 175, 68, 42, 82, 197, 134, 142, 222, 130, 108, 153, 20, 215, 229, 65, 225, 210, 8, 137, 86, 108, 198, 117, 100, 166, 42, 248, 127, 164, 22, 207, 211, 33, 182, 151, 193, 56, 245, 255, 84, 240, 215, 11, 131, 70, 87, 176, 82, 62, 120, 246, 195, 219, 59, 230, 216, 23, 152, 108, 243, 22, 135, 62, 66, 143, 40, 9, 52, 166, 105, 114, 191, 99, 72, 120, 240, 178, 196, 30, 191, 172, 228, 102, 50, 74, 168, 85, 74, 132, 12, 221, 96, 136, 247, 174, 178, 254, 150, 120, 162, 24, 216, 223, 50, 209, 181, 228, 96, 36, 51, 139, 44, 28, 84, 213, 160, 184, 25, 195, 185, 244, 125, 78, 207, 247, 105, 31, 35, 113, 7, 229, 19, 136, 71, 80, 161, 64, 37, 85, 208, 145, 158, 252, 148, 135, 193, 67, 18, 40, 135, 49, 38, 103, 237, 191, 66, 106, 211, 142, 146, 223, 121, 88, 132, 245, 183, 193, 16, 175, 149, 196, 60, 5, 18, 103, 8, 248, 50, 175, 126, 150, 247, 163, 150, 210, 98, 43, 175, 211, 72, 251, 255, 83, 228, 199, 239, 104, 36, 44, 131, 30, 6, 53, 167, 112, 125, 216, 112, 98, 154, 35, 237, 8, 106, 18, 255, 68, 207, 155, 32, 71, 179, 111, 115, 188, 86, 56, 96, 213, 149, 158, 238, 143, 116, 162, 27, 224, 238, 71, 229, 214, 16, 142, 94, 118, 215, 130, 118, 182, 56, 8, 142, 179, 35, 216, 224, 46, 192, 166, 204, 67, 255, 12, 92, 250, 226, 19, 137, 79, 92, 180, 85, 70, 125, 255, 204, 227, 68, 238, 226, 34, 168, 120, 251, 34, 146, 73, 81, 157, 48, 20, 60, 177, 114, 124, 202, 108, 82, 129, 249, 188, 205, 40, 200, 180, 238, 112, 62, 84, 178, 95, 82, 142, 21, 228, 108, 145, 255, 185, 189, 12, 161, 130, 172, 33, 224, 232, 58, 216, 190, 238, 107, 46, 62, 151, 58, 39, 94, 222, 171, 192, 32, 202, 191, 249, 130, 84, 218, 255, 111, 42, 46, 122, 14, 240, 27, 141, 82, 88, 165, 75, 47, 96, 216, 157, 173, 9, 168, 151, 206, 80, 26, 48, 142, 61, 46, 104, 240, 193, 70, 110, 226, 152, 156, 233, 127, 95, 136, 253, 187, 195, 25, 179, 155, 204, 77, 16, 29, 124, 22, 7, 64, 189, 138, 161, 252, 166, 150, 219, 95, 49, 186, 226, 77, 8, 12, 85, 239, 205, 249, 107, 47, 56, 130, 39, 15, 64, 182, 130, 144, 232, 140, 124, 177, 44, 247, 16, 111, 21, 9, 69, 207, 156, 40, 76, 191, 116, 120, 201, 92, 60, 102, 218, 149, 156, 243, 144, 124, 173, 38, 238, 254, 94, 247, 230, 28, 161, 104, 126, 223, 129, 112, 175, 64, 12, 151, 191, 44, 231, 235, 52, 204, 173, 211, 77, 12, 20, 100, 7, 233, 30, 149, 97, 113, 201, 108, 90, 144, 13, 217, 236, 75, 243, 227, 36, 167, 116, 8, 44, 155, 80, 84, 165, 56, 28, 68, 182, 117, 125, 203, 113, 93, 140, 6, 202, 217, 56, 215, 198, 250, 126, 72, 91, 182, 97, 80, 143, 24, 231, 116, 155, 12, 195, 203, 23, 167, 136, 175, 42, 230, 240, 68, 224, 197, 251, 118, 58, 79, 172, 76, 52, 109, 234, 181, 203, 40, 214, 195, 255, 131, 84];
    #[rustfmt::skip]
    pub const OCV_CLAHE_U8_32X32_CLIP40_0_TILES8X8: [u8; 1024] = [16, 64, 191, 104, 104, 195, 80, 40, 72, 195, 112, 124, 215, 108, 80, 136, 8, 215, 239, 72, 215, 183, 255, 128, 80, 108, 223, 116, 80, 143, 16, 223, 128, 191, 48, 219, 223, 24, 207, 151, 191, 48, 239, 251, 48, 235, 191, 255, 128, 80, 112, 167, 80, 52, 112, 243, 183, 227, 48, 239, 199, 255, 128, 80, 239, 48, 159, 72, 72, 136, 32, 255, 40, 159, 80, 92, 159, 64, 32, 104, 247, 179, 207, 28, 175, 147, 223, 96, 32, 60, 159, 72, 40, 116, 239, 191, 104, 151, 251, 185, 189, 243, 143, 111, 151, 9, 207, 213, 20, 185, 155, 218, 88, 32, 52, 137, 34, 13, 72, 193, 138, 165, 12, 178, 159, 218, 100, 52, 207, 247, 112, 30, 28, 108, 247, 221, 255, 114, 48, 54, 144, 40, 16, 66, 187, 134, 144, 237, 132, 114, 167, 46, 251, 16, 128, 24, 8, 72, 207, 167, 56, 112, 223, 141, 143, 205, 120, 76, 116, 232, 171, 182, 249, 171, 135, 185, 46, 242, 255, 78, 243, 225, 32, 156, 102, 133, 219, 148, 126, 190, 72, 24, 159, 223, 80, 247, 247, 56, 207, 179, 223, 92, 16, 32, 104, 12, 239, 40, 167, 84, 96, 179, 80, 60, 143, 12, 199, 223, 64, 243, 223, 56, 191, 143, 24, 72, 171, 106, 110, 163, 64, 35, 80, 196, 143, 151, 215, 128, 100, 152, 12, 201, 211, 48, 205, 190, 255, 120, 62, 88, 187, 105, 92, 156, 44, 251, 128, 167, 32, 205, 203, 28, 167, 145, 183, 46, 239, 245, 80, 231, 207, 253, 116, 58, 64, 161, 60, 46, 104, 237, 187, 207, 64, 215, 199, 8, 143, 104, 231, 32, 143, 62, 64, 126, 40, 255, 44, 164, 108, 117, 181, 102, 64, 116, 233, 176, 191, 16, 183, 167, 231, 103, 50, 84, 171, 95, 66, 125, 255, 207, 80, 143, 255, 167, 167, 231, 128, 104, 151, 24, 207, 219, 32, 191, 159, 223, 104, 28, 48, 132, 32, 12, 96, 223, 159, 187, 32, 199, 167, 243, 112, 64, 203, 251, 96, 30, 34, 88, 243, 215, 6, 123, 72, 79, 141, 53, 24, 83, 203, 143, 159, 251, 153, 138, 203, 68, 10, 36, 135, 47, 28, 87, 223, 175, 56, 96, 215, 133, 132, 211, 96, 72, 108, 223, 159, 167, 4, 157, 136, 185, 52, 253, 8, 106, 4, 245, 48, 177, 124, 139, 247, 143, 128, 191, 72, 32, 163, 219, 76, 249, 251, 58, 227, 182, 221, 82, 20, 34, 104, 29, 251, 50, 169, 115, 131, 211, 124, 108, 171, 35, 229, 255, 80, 9, 241, 51, 187, 139, 16, 80, 191, 104, 104, 167, 64, 32, 72, 191, 112, 132, 207, 120, 96, 159, 40, 219, 239, 68, 223, 203, 32, 147, 72, 88, 175, 100, 80, 167, 48, 255, 135, 183, 28, 219, 225, 26, 183, 149, 189, 46, 243, 255, 68, 239, 215, 19, 140, 80, 96, 186, 88, 72, 135, 249, 185, 205, 44, 217, 205, 16, 159, 112, 239, 24, 143, 66, 68, 151, 40, 12, 44, 155, 88, 98, 191, 92, 72, 122, 243, 189, 199, 40, 191, 175, 231, 106, 52, 68, 175, 74, 60, 126, 8, 223, 88, 143, 255, 179, 187, 255, 175, 128, 165, 24, 215, 228, 42, 221, 187, 241, 106, 51, 68, 144, 54, 35, 96, 220, 165, 197, 28, 209, 183, 245, 124, 76, 191, 255, 112, 32, 40, 112, 16, 239, 24, 143, 64, 80, 151, 60, 32, 96, 231, 155, 175, 255, 151, 128, 207, 80, 16, 44, 143, 60, 32, 112, 239, 191, 60, 108, 207, 148, 159, 220, 128, 95, 136, 248, 191, 200, 10, 178, 151, 208, 72, 10, 24, 113, 12, 249, 56, 178, 124, 152, 255, 168, 151, 212, 96, 48, 167, 207, 72, 251, 255, 86, 231, 205, 239, 98, 32, 40, 132, 30, 8, 54, 171, 114, 120, 215, 112, 96, 151, 32, 239, 6, 120, 16, 255, 64, 199, 159, 20, 76, 187, 111, 118, 184, 104, 60, 100, 216, 155, 167, 235, 158, 124, 171, 30, 224, 235, 59, 225, 208, 16, 143, 92, 126, 215, 142, 118, 180, 60, 12, 128, 191, 48, 219, 223, 36, 191, 163, 207, 76, 255, 16, 88, 251, 223, 24, 152, 68, 80, 163, 64, 44, 128, 255, 191, 219, 64, 239, 215, 44, 175, 128, 247, 40, 139, 79, 88, 146, 52, 21, 64, 178, 124, 133, 199, 114, 88, 140, 255, 189, 199, 36, 193, 178, 243, 109, 52, 79, 179, 95, 80, 142, 28, 235, 96, 136, 255, 179, 183, 14, 159, 134, 167, 26, 215, 225, 64, 219, 199, 245, 108, 50, 56, 153, 52, 38, 96, 227, 175, 193, 48, 199, 183, 247, 128, 88, 199, 255, 112, 37, 46, 115, 36, 245, 28, 142, 80, 95, 165, 92, 60, 112, 229, 172, 187, 12, 179, 163, 227, 94, 36, 65, 147, 73, 46, 107, 239, 191, 48, 112, 223, 143, 151, 223, 128, 96, 136, 255, 175, 195, 16, 183, 159, 223, 104, 28, 48, 132, 32, 12, 96, 215, 144, 163, 255, 171, 143, 223, 96, 48, 179, 227, 72, 12, 22, 82, 243, 209, 249, 106, 48, 59, 126, 41, 16, 76, 197, 138, 155, 246, 147, 131, 195, 55, 247, 13, 108, 22, 6, 68, 207, 159, 40, 80, 199, 122, 124, 207, 96, 68, 100, 211, 143, 151, 243, 141, 120, 171, 40, 243, 255, 96, 247, 231, 32, 159, 104, 118, 223, 122, 108, 173, 56, 16, 155, 211, 68, 243, 247, 56, 227, 180, 217, 76, 12, 22, 88, 9, 227, 29, 151, 100, 120, 196, 106, 87, 147, 12, 207, 234, 60, 245, 223, 34, 171, 124, 16, 80, 191, 104, 104, 167, 64, 32, 72, 191, 112, 124, 191, 96, 64, 132, 16, 199, 223, 48, 199, 175, 255, 120, 48, 68, 159, 84, 64, 151, 32, 239, 143, 191, 32, 227, 231, 20, 175, 139, 183, 44, 239, 247, 48, 211, 175, 251, 120, 56, 80, 171, 64, 44, 112, 223, 151, 175, 16, 195, 183, 255, 143, 96];
    #[rustfmt::skip]
    pub const OCV_CORNERS_EIGEN_VALS_VECS_F32_B3_K3: [f32; 1536] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 555.55566, 0.0, 0.70710677, 0.70710677, 0.70710677, -0.70710677, 3238.0151, 95.3183, 0.38268346, 0.9238795, 0.9238795, -0.38268346, 7402.5317, 375.24713, 0.16018224, 0.9870875, 0.9870875, -0.16018224, 11451.043, 215.6255, 0.074376225, 0.99723023, 0.99723023, -0.074376225, 13333.336, 0.0, 0.0, 1.0, -1.0, 0.0, 13333.336, 0.0, 0.0, 1.0, -1.0, 0.0, 13333.336, 0.0, 0.0, 1.0, -1.0, 0.0, 13333.336, 0.0, 0.0, 1.0, -1.0, 0.0, 11451.043, 215.6255, -0.074376225, 0.99723023, -0.99723023, -0.074376225, 7402.5317, 375.24713, -0.16018224, 0.9870875, -0.9870875, -0.16018224, 3238.0151, 95.3183, -0.38268346, 0.9238795, -0.9238795, -0.38268346, 555.55566, 0.0, -0.70710677, 0.70710677, -0.70710677, -0.70710677, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3238.0156, 95.31828, 0.9238795, 0.38268343, 0.38268343, -0.9238795, 10000.001, 1111.1113, 0.7071068, 0.70710677, 0.70710677, -0.7071068, 16285.3955, 3714.6074, 0.38268343, 0.9238795, 0.9238795, -0.38268343, 23318.707, 2236.8528, 0.16018222, 0.9870875, 0.9870875, -0.16018222, 26666.672, 0.0, 0.0, 1.0, -1.0, 0.0, 26666.672, 0.0, 0.0, 1.0, -1.0, 0.0, 26666.672, 0.0, 0.0, 1.0, -1.0, 0.0, 26666.672, 0.0, 0.0, 1.0, -1.0, 0.0, 23318.707, 2236.8528, -0.16018222, 0.9870875, -0.9870875, -0.16018222, 16285.3955, 3714.6074, -0.38268343, 0.9238795, -0.9238795, -0.38268343, 10000.001, 1111.1113, -0.7071068, 0.70710677, -0.70710677, -0.7071068, 3238.0156, 95.31828, -0.9238795, 0.38268343, -0.38268343, -0.9238795, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 7402.532, 375.24713, 0.9870874, 0.16018224, 0.16018224, -0.9870874, 16285.396, 3714.6067, 0.92387956, 0.3826834, 0.3826834, -0.92387956, 18888.893, 10000.002, 0.7071068, 0.70710677, 0.70710677, -0.7071068, 23461.973, 6538.0312, 0.20106585, 0.9795777, 0.9795777, -0.20106585, 26666.672, 0.0, 0.0, 1.0, -1.0, 0.0, 26666.672, 0.0, 0.0, 1.0, -1.0, 0.0, 26666.672, 0.0, 0.0, 1.0, -1.0, 0.0, 26666.672, 0.0, 0.0, 1.0, -1.0, 0.0, 23461.973, 6538.0312, -0.20106585, 0.9795777, -0.9795777, -0.20106585, 18888.893, 10000.002, -0.7071068, 0.70710677, -0.70710677, -0.7071068, 16285.396, 3714.6067, -0.92387956, 0.3826834, -0.3826834, -0.92387956, 7402.532, 375.24713, -0.9870874, 0.16018224, -0.16018224, -0.9870874, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 11451.043, 215.62549, 0.99723023, 0.074376225, 0.074376225, -0.99723023, 23318.707, 2236.8523, 0.9870875, 0.16018224, 0.16018224, -0.9870875, 23461.975, 6538.031, 0.9795777, 0.20106587, 0.20106587, -0.9795777, 13888.891, 8888.891, 0.70710677, 0.70710677, 0.70710677, -0.70710677, 13333.336, 0.0, 0.0, 1.0, -1.0, 0.0, 13333.336, 0.0, 0.0, 1.0, -1.0, 0.0, 13333.336, 0.0, 0.0, 1.0, -1.0, 0.0, 13333.336, 0.0, 0.0, 1.0, -1.0, 0.0, 13888.891, 8888.891, -0.70710677, 0.70710677, -0.70710677, -0.70710677, 23461.975, 6538.031, -0.9795777, 0.20106587, -0.20106587, -0.9795777, 23318.707, 2236.8523, -0.9870875, 0.16018224, -0.16018224, -0.9870875, 11451.043, 215.62549, -0.99723023, 0.074376225, -0.074376225, -0.99723023, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 13333.336, 0.0, 1.0, 0.0, 0.0, -1.0, 26666.672, 0.0, 1.0, 0.0, 0.0, -1.0, 26666.672, 0.0, 1.0, 0.0, 0.0, -1.0, 13333.336, 0.0, 1.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 13333.336, 0.0, 1.0, 0.0, 0.0, -1.0, 26666.672, 0.0, 1.0, 0.0, 0.0, -1.0, 26666.672, 0.0, 1.0, 0.0, 0.0, -1.0, 13333.336, 0.0, 1.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 13333.336, 0.0, 1.0, 0.0, 0.0, -1.0, 26666.672, 0.0, 1.0, 0.0, 0.0, -1.0, 26666.672, 0.0, 1.0, 0.0, 0.0, -1.0, 13333.336, 0.0, 1.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 13333.336, 0.0, 1.0, 0.0, 0.0, -1.0, 26666.672, 0.0, 1.0, 0.0, 0.0, -1.0, 26666.672, 0.0, 1.0, 0.0, 0.0, -1.0, 13333.336, 0.0, 1.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 13333.336, 0.0, 1.0, 0.0, 0.0, -1.0, 26666.672, 0.0, 1.0, 0.0, 0.0, -1.0, 26666.672, 0.0, 1.0, 0.0, 0.0, -1.0, 13333.336, 0.0, 1.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 13333.336, 0.0, 1.0, 0.0, 0.0, -1.0, 26666.672, 0.0, 1.0, 0.0, 0.0, -1.0, 26666.672, 0.0, 1.0, 0.0, 0.0, -1.0, 13333.336, 0.0, 1.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 13333.336, 0.0, 1.0, 0.0, 0.0, -1.0, 26666.672, 0.0, 1.0, 0.0, 0.0, -1.0, 26666.672, 0.0, 1.0, 0.0, 0.0, -1.0, 13333.336, 0.0, 1.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 13333.336, 0.0, 1.0, 0.0, 0.0, -1.0, 26666.672, 0.0, 1.0, 0.0, 0.0, -1.0, 26666.672, 0.0, 1.0, 0.0, 0.0, -1.0, 13333.336, 0.0, 1.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 11451.043, 215.62549, -0.99723023, 0.074376225, -0.074376225, -0.99723023, 23318.707, 2236.8523, -0.9870875, 0.16018224, -0.16018224, -0.9870875, 23461.975, 6538.031, -0.9795777, 0.20106587, -0.20106587, -0.9795777, 13888.891, 8888.891, -0.70710677, 0.70710677, -0.70710677, -0.70710677, 13333.336, 0.0, 0.0, 1.0, -1.0, 0.0, 13333.336, 0.0, 0.0, 1.0, -1.0, 0.0, 13333.336, 0.0, 0.0, 1.0, -1.0, 0.0, 13333.336, 0.0, 0.0, 1.0, -1.0, 0.0, 13888.891, 8888.891, 0.70710677, 0.70710677, 0.70710677, -0.70710677, 23461.975, 6538.031, 0.9795777, 0.20106587, 0.20106587, -0.9795777, 23318.707, 2236.8523, 0.9870875, 0.16018224, 0.16018224, -0.9870875, 11451.043, 215.62549, 0.99723023, 0.074376225, 0.074376225, -0.99723023, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 7402.532, 375.24713, -0.9870874, 0.16018224, -0.16018224, -0.9870874, 16285.396, 3714.6067, -0.92387956, 0.3826834, -0.3826834, -0.92387956, 18888.893, 10000.002, -0.7071068, 0.70710677, -0.70710677, -0.7071068, 23461.973, 6538.0312, -0.20106585, 0.9795777, -0.9795777, -0.20106585, 26666.672, 0.0, 0.0, 1.0, -1.0, 0.0, 26666.672, 0.0, 0.0, 1.0, -1.0, 0.0, 26666.672, 0.0, 0.0, 1.0, -1.0, 0.0, 26666.672, 0.0, 0.0, 1.0, -1.0, 0.0, 23461.973, 6538.0312, 0.20106585, 0.9795777, 0.9795777, -0.20106585, 18888.893, 10000.002, 0.7071068, 0.70710677, 0.70710677, -0.7071068, 16285.396, 3714.6067, 0.92387956, 0.3826834, 0.3826834, -0.92387956, 7402.532, 375.24713, 0.9870874, 0.16018224, 0.16018224, -0.9870874, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3238.0156, 95.31828, -0.9238795, 0.38268343, -0.38268343, -0.9238795, 10000.001, 1111.1113, -0.7071068, 0.70710677, -0.70710677, -0.7071068, 16285.3955, 3714.6074, -0.38268343, 0.9238795, -0.9238795, -0.38268343, 23318.707, 2236.8528, -0.16018222, 0.9870875, -0.9870875, -0.16018222, 26666.672, 0.0, 0.0, 1.0, -1.0, 0.0, 26666.672, 0.0, 0.0, 1.0, -1.0, 0.0, 26666.672, 0.0, 0.0, 1.0, -1.0, 0.0, 26666.672, 0.0, 0.0, 1.0, -1.0, 0.0, 23318.707, 2236.8528, 0.16018222, 0.9870875, 0.9870875, -0.16018222, 16285.3955, 3714.6074, 0.38268343, 0.9238795, 0.9238795, -0.38268343, 10000.001, 1111.1113, 0.7071068, 0.70710677, 0.70710677, -0.7071068, 3238.0156, 95.31828, 0.9238795, 0.38268343, 0.38268343, -0.9238795, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 555.55566, 0.0, -0.70710677, 0.70710677, -0.70710677, -0.70710677, 3238.0151, 95.3183, -0.38268346, 0.9238795, -0.9238795, -0.38268346, 7402.5317, 375.24713, -0.16018224, 0.9870875, -0.9870875, -0.16018224, 11451.043, 215.6255, -0.074376225, 0.99723023, -0.99723023, -0.074376225, 13333.336, 0.0, 0.0, 1.0, -1.0, 0.0, 13333.336, 0.0, 0.0, 1.0, -1.0, 0.0, 13333.336, 0.0, 0.0, 1.0, -1.0, 0.0, 13333.336, 0.0, 0.0, 1.0, -1.0, 0.0, 11451.043, 215.6255, 0.074376225, 0.99723023, 0.99723023, -0.074376225, 7402.5317, 375.24713, 0.16018224, 0.9870875, 0.9870875, -0.16018224, 3238.0151, 95.3183, 0.38268346, 0.9238795, 0.9238795, -0.38268346, 555.55566, 0.0, 0.70710677, 0.70710677, 0.70710677, -0.70710677, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    #[rustfmt::skip]
    pub const OCV_CORNERS_HARRIS_F32_B3_K3_0_04: [f32; 256] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -12345.684, -135802.34, 358025.25, -2975308.8, -7111113.5, -7111113.5, -7111113.5, -7111113.5, -2975308.8, 358025.25, -135802.34, -12345.684, 0.0, 0.0, 0.0, 0.0, -135802.47, 6172841.5, 4.449384e+07, 26037052.0, -28444454.0, -28444454.0, -28444454.0, -28444454.0, 26037052.0, 4.449384e+07, 6172841.5, -135802.47, 0.0, 0.0, 0.0, 0.0, 358025.0, 4.449383e+07, 1.5550622e+08, 117395096.0, -28444454.0, -28444454.0, -28444454.0, -28444454.0, 117395096.0, 1.5550622e+08, 4.449383e+07, 358025.0, 0.0, 0.0, 0.0, 0.0, -2975309.0, 2.603704e+07, 117395096.0, 1.0270374e+08, -7111113.5, -7111113.5, -7111113.5, -7111113.5, 1.0270374e+08, 117395096.0, 2.603704e+07, -2975309.0, 0.0, 0.0, 0.0, 0.0, -7111113.5, -28444454.0, -28444454.0, -7111113.5, 0.0, 0.0, 0.0, 0.0, -7111113.5, -28444454.0, -28444454.0, -7111113.5, 0.0, 0.0, 0.0, 0.0, -7111113.5, -28444454.0, -28444454.0, -7111113.5, 0.0, 0.0, 0.0, 0.0, -7111113.5, -28444454.0, -28444454.0, -7111113.5, 0.0, 0.0, 0.0, 0.0, -7111113.5, -28444454.0, -28444454.0, -7111113.5, 0.0, 0.0, 0.0, 0.0, -7111113.5, -28444454.0, -28444454.0, -7111113.5, 0.0, 0.0, 0.0, 0.0, -7111113.5, -28444454.0, -28444454.0, -7111113.5, 0.0, 0.0, 0.0, 0.0, -7111113.5, -28444454.0, -28444454.0, -7111113.5, 0.0, 0.0, 0.0, 0.0, -2975309.0, 2.603704e+07, 117395096.0, 1.0270374e+08, -7111113.5, -7111113.5, -7111113.5, -7111113.5, 1.0270374e+08, 117395096.0, 2.603704e+07, -2975309.0, 0.0, 0.0, 0.0, 0.0, 358025.0, 4.449383e+07, 1.5550622e+08, 117395096.0, -28444454.0, -28444454.0, -28444454.0, -28444454.0, 117395096.0, 1.5550622e+08, 4.449383e+07, 358025.0, 0.0, 0.0, 0.0, 0.0, -135802.47, 6172841.5, 4.449384e+07, 26037052.0, -28444454.0, -28444454.0, -28444454.0, -28444454.0, 26037052.0, 4.449384e+07, 6172841.5, -135802.47, 0.0, 0.0, 0.0, 0.0, -12345.684, -135802.34, 358025.25, -2975308.8, -7111113.5, -7111113.5, -7111113.5, -7111113.5, -2975308.8, 358025.25, -135802.34, -12345.684, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    #[rustfmt::skip]
    pub const OCV_CORNERS_MIN_EIGEN_F32_B3_K3: [f32; 256] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 95.31824, 375.24707, 215.625, 0.0, 0.0, 0.0, 0.0, 215.625, 375.24707, 95.31824, 0.0, 0.0, 0.0, 0.0, 0.0, 95.31824, 1111.1113, 3714.608, 2236.8516, 0.0, 0.0, 0.0, 0.0, 2236.8516, 3714.608, 1111.1113, 95.31824, 0.0, 0.0, 0.0, 0.0, 375.2473, 3714.607, 10000.002, 6538.0303, 0.0, 0.0, 0.0, 0.0, 6538.0303, 10000.002, 3714.607, 375.2473, 0.0, 0.0, 0.0, 0.0, 215.625, 2236.8516, 6538.0303, 8888.891, 0.0, 0.0, 0.0, 0.0, 8888.891, 6538.0303, 2236.8516, 215.625, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 215.625, 2236.8516, 6538.0303, 8888.891, 0.0, 0.0, 0.0, 0.0, 8888.891, 6538.0303, 2236.8516, 215.625, 0.0, 0.0, 0.0, 0.0, 375.2473, 3714.607, 10000.002, 6538.0303, 0.0, 0.0, 0.0, 0.0, 6538.0303, 10000.002, 3714.607, 375.2473, 0.0, 0.0, 0.0, 0.0, 95.31824, 1111.1113, 3714.608, 2236.8516, 0.0, 0.0, 0.0, 0.0, 2236.8516, 3714.608, 1111.1113, 95.31824, 0.0, 0.0, 0.0, 0.0, 0.0, 95.31824, 375.24707, 215.625, 0.0, 0.0, 0.0, 0.0, 215.625, 375.24707, 95.31824, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    #[rustfmt::skip]
    pub const OCV_CORNERS_PRE_CORNER_F32_K3: [f32; 256] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.5e+06, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.5e+06, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.5e+06, -9e+06, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -9e+06, -1.5e+06, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.5e+06, -9e+06, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -9e+06, -1.5e+06, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.5e+06, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.5e+06, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    #[rustfmt::skip]
    pub const OCV_CORNER_SUB_PIX_F32_WIN2_ZERO_1: [f32; 8] = [3.6153896, 3.6153896, 11.384665, 3.6153355, 3.6153805, 11.384613, 11.38461, 11.38461];
    #[rustfmt::skip]
    pub const OCV_CVT_COLOR_GRAY2BGR: [u8; 18] = [0, 0, 0, 1, 1, 1, 127, 127, 127, 128, 128, 128, 254, 254, 254, 255, 255, 255];
    #[rustfmt::skip]
    pub const OCV_CVT_COLOR_GRAY2BGRA: [u8; 24] = [0, 0, 0, 255, 1, 1, 1, 255, 127, 127, 127, 255, 128, 128, 128, 255, 254, 254, 254, 255, 255, 255, 255, 255];
    #[rustfmt::skip]
    pub const OCV_CVT_COLOR_GRAY2RGB: [u8; 18] = [0, 0, 0, 1, 1, 1, 127, 127, 127, 128, 128, 128, 254, 254, 254, 255, 255, 255];
    #[rustfmt::skip]
    pub const OCV_CVT_COLOR_GRAY2RGBA: [u8; 24] = [0, 0, 0, 255, 1, 1, 1, 255, 127, 127, 127, 255, 128, 128, 128, 255, 254, 254, 254, 255, 255, 255, 255, 255];
    #[rustfmt::skip]
    pub const OCV_DERIVATIVES_LAPLACIAN_F32_K1: [f32; 64] = [456.0, -2.5e+02, 6.0, 6.0, 6.0, 6.0, 6.0, 68.0, 4.5e+02, -256.0, 0.0, 0.0, 0.0, 0.0, 0.0, 62.0, 4.5e+02, -256.0, 0.0, 0.0, 0.0, 0.0, 0.0, 62.0, 4.5e+02, -256.0, 0.0, 0.0, 0.0, 0.0, 0.0, 62.0, 4.5e+02, -256.0, 0.0, 0.0, 0.0, 0.0, 0.0, 62.0, 4.5e+02, -256.0, 0.0, 0.0, 0.0, 0.0, 0.0, 62.0, 4.5e+02, -256.0, 0.0, 0.0, 0.0, 0.0, 0.0, 62.0, 444.0, -262.0, -6.0, -6.0, -6.0, -6.0, -6.0, 56.0];
    #[rustfmt::skip]
    pub const OCV_DERIVATIVES_LAPLACIAN_F32_K3: [f32; 64] = [1824.0, -1e+03, 24.0, 24.0, 24.0, 24.0, 24.0, 272.0, 1.8e+03, -1024.0, 0.0, 0.0, 0.0, 0.0, 0.0, 248.0, 1.8e+03, -1024.0, 0.0, 0.0, 0.0, 0.0, 0.0, 248.0, 1.8e+03, -1024.0, 0.0, 0.0, 0.0, 0.0, 0.0, 248.0, 1.8e+03, -1024.0, 0.0, 0.0, 0.0, 0.0, 0.0, 248.0, 1.8e+03, -1024.0, 0.0, 0.0, 0.0, 0.0, 0.0, 248.0, 1.8e+03, -1024.0, 0.0, 0.0, 0.0, 0.0, 0.0, 248.0, 1776.0, -1048.0, -24.0, -24.0, -24.0, -24.0, -24.0, 224.0];
    #[rustfmt::skip]
    pub const OCV_DERIVATIVES_LAPLACIAN_F32_K3_SCALE0_5_DELTA2: [f32; 64] = [914.0, -498.0, 14.0, 14.0, 14.0, 14.0, 14.0, 138.0, 902.0, -5.1e+02, 2.0, 2.0, 2.0, 2.0, 2.0, 126.0, 902.0, -5.1e+02, 2.0, 2.0, 2.0, 2.0, 2.0, 126.0, 902.0, -5.1e+02, 2.0, 2.0, 2.0, 2.0, 2.0, 126.0, 902.0, -5.1e+02, 2.0, 2.0, 2.0, 2.0, 2.0, 126.0, 902.0, -5.1e+02, 2.0, 2.0, 2.0, 2.0, 2.0, 126.0, 902.0, -5.1e+02, 2.0, 2.0, 2.0, 2.0, 2.0, 126.0, 8.9e+02, -522.0, -1e+01, -1e+01, -1e+01, -1e+01, -1e+01, 114.0];
    #[rustfmt::skip]
    pub const OCV_DERIVATIVES_LAPLACIAN_F32_K5: [f32; 64] = [6.4e+03, -8e+02, -3904.0, 192.0, 192.0, 192.0, 1184.0, 2176.0, 6304.0, -896.0, -4e+03, 96.0, 96.0, 96.0, 1088.0, 2.08e+03, 6208.0, -992.0, -4096.0, 0.0, 0.0, 0.0, 992.0, 1984.0, 6208.0, -992.0, -4096.0, 0.0, 0.0, 0.0, 992.0, 1984.0, 6208.0, -992.0, -4096.0, 0.0, 0.0, 0.0, 992.0, 1984.0, 6208.0, -992.0, -4096.0, 0.0, 0.0, 0.0, 992.0, 1984.0, 6112.0, -1088.0, -4192.0, -96.0, -96.0, -96.0, 896.0, 1888.0, 6016.0, -1184.0, -4288.0, -192.0, -192.0, -192.0, 8e+02, 1792.0];
    #[rustfmt::skip]
    pub const OCV_DERIVATIVES_LAPLACIAN_U8_K1_SAME: [u8; 64] = [255, 0, 6, 6, 6, 6, 6, 68, 255, 0, 0, 0, 0, 0, 0, 62, 255, 0, 0, 0, 0, 0, 0, 62, 255, 0, 0, 0, 0, 0, 0, 62, 255, 0, 0, 0, 0, 0, 0, 62, 255, 0, 0, 0, 0, 0, 0, 62, 255, 0, 0, 0, 0, 0, 0, 62, 255, 0, 0, 0, 0, 0, 0, 56];
    #[rustfmt::skip]
    pub const OCV_DERIVATIVES_SCHARR_F32_DX1: [f32; 64] = [0.0, 3104.0, -992.0, -992.0, -992.0, -992.0, -992.0, 0.0, 0.0, 3104.0, -992.0, -992.0, -992.0, -992.0, -992.0, 0.0, 0.0, 3104.0, -992.0, -992.0, -992.0, -992.0, -992.0, 0.0, 0.0, 3104.0, -992.0, -992.0, -992.0, -992.0, -992.0, 0.0, 0.0, 3104.0, -992.0, -992.0, -992.0, -992.0, -992.0, 0.0, 0.0, 3104.0, -992.0, -992.0, -992.0, -992.0, -992.0, 0.0, 0.0, 3104.0, -992.0, -992.0, -992.0, -992.0, -992.0, 0.0, 0.0, 3104.0, -992.0, -992.0, -992.0, -992.0, -992.0, 0.0];
    #[rustfmt::skip]
    pub const OCV_DERIVATIVES_SCHARR_F32_DY1: [f32; 64] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 96.0, 96.0, 96.0, 96.0, 96.0, 96.0, 96.0, 96.0, 96.0, 96.0, 96.0, 96.0, 96.0, 96.0, 96.0, 96.0, 96.0, 96.0, 96.0, 96.0, 96.0, 96.0, 96.0, 96.0, 96.0, 96.0, 96.0, 96.0, 96.0, 96.0, 96.0, 96.0, 96.0, 96.0, 96.0, 96.0, 96.0, 96.0, 96.0, 96.0, 96.0, 96.0, 96.0, 96.0, 96.0, 96.0, 96.0, 96.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    #[rustfmt::skip]
    pub const OCV_DERIVATIVES_SOBEL_F32_DX1_DY1_K3: [f32; 64] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    #[rustfmt::skip]
    pub const OCV_DERIVATIVES_SOBEL_F32_DX1_K1: [f32; 64] = [0.0, 194.0, -62.0, -62.0, -62.0, -62.0, -62.0, 0.0, 0.0, 194.0, -62.0, -62.0, -62.0, -62.0, -62.0, 0.0, 0.0, 194.0, -62.0, -62.0, -62.0, -62.0, -62.0, 0.0, 0.0, 194.0, -62.0, -62.0, -62.0, -62.0, -62.0, 0.0, 0.0, 194.0, -62.0, -62.0, -62.0, -62.0, -62.0, 0.0, 0.0, 194.0, -62.0, -62.0, -62.0, -62.0, -62.0, 0.0, 0.0, 194.0, -62.0, -62.0, -62.0, -62.0, -62.0, 0.0, 0.0, 194.0, -62.0, -62.0, -62.0, -62.0, -62.0, 0.0];
    #[rustfmt::skip]
    pub const OCV_DERIVATIVES_SOBEL_F32_DX1_K3: [f32; 64] = [0.0, 776.0, -248.0, -248.0, -248.0, -248.0, -248.0, 0.0, 0.0, 776.0, -248.0, -248.0, -248.0, -248.0, -248.0, 0.0, 0.0, 776.0, -248.0, -248.0, -248.0, -248.0, -248.0, 0.0, 0.0, 776.0, -248.0, -248.0, -248.0, -248.0, -248.0, 0.0, 0.0, 776.0, -248.0, -248.0, -248.0, -248.0, -248.0, 0.0, 0.0, 776.0, -248.0, -248.0, -248.0, -248.0, -248.0, 0.0, 0.0, 776.0, -248.0, -248.0, -248.0, -248.0, -248.0, 0.0, 0.0, 776.0, -248.0, -248.0, -248.0, -248.0, -248.0, 0.0];
    #[rustfmt::skip]
    pub const OCV_DERIVATIVES_SOBEL_F32_DX1_K3_CONSTANT: [f32; 64] = [678.0, 582.0, -186.0, -186.0, -186.0, -186.0, -186.0, -213.0, 912.0, 776.0, -248.0, -248.0, -248.0, -248.0, -248.0, -292.0, 924.0, 776.0, -248.0, -248.0, -248.0, -248.0, -248.0, -304.0, 936.0, 776.0, -248.0, -248.0, -248.0, -248.0, -248.0, -316.0, 948.0, 776.0, -248.0, -248.0, -248.0, -248.0, -248.0, -328.0, 9.6e+02, 776.0, -248.0, -248.0, -248.0, -248.0, -248.0, -3.4e+02, 972.0, 776.0, -248.0, -248.0, -248.0, -248.0, -248.0, -352.0, 735.0, 582.0, -186.0, -186.0, -186.0, -186.0, -186.0, -2.7e+02];
    #[rustfmt::skip]
    pub const OCV_DERIVATIVES_SOBEL_F32_DX1_K3_REPLICATE: [f32; 64] = [9e+02, 776.0, -248.0, -248.0, -248.0, -248.0, -248.0, -124.0, 9e+02, 776.0, -248.0, -248.0, -248.0, -248.0, -248.0, -124.0, 9e+02, 776.0, -248.0, -248.0, -248.0, -248.0, -248.0, -124.0, 9e+02, 776.0, -248.0, -248.0, -248.0, -248.0, -248.0, -124.0, 9e+02, 776.0, -248.0, -248.0, -248.0, -248.0, -248.0, -124.0, 9e+02, 776.0, -248.0, -248.0, -248.0, -248.0, -248.0, -124.0, 9e+02, 776.0, -248.0, -248.0, -248.0, -248.0, -248.0, -124.0, 9e+02, 776.0, -248.0, -248.0, -248.0, -248.0, -248.0, -124.0];
    #[rustfmt::skip]
    pub const OCV_DERIVATIVES_SOBEL_F32_DX1_K5: [f32; 64] = [0.0, 5216.0, 128.0, -3968.0, -3968.0, -3968.0, -2976.0, 0.0, 0.0, 5216.0, 128.0, -3968.0, -3968.0, -3968.0, -2976.0, 0.0, 0.0, 5216.0, 128.0, -3968.0, -3968.0, -3968.0, -2976.0, 0.0, 0.0, 5216.0, 128.0, -3968.0, -3968.0, -3968.0, -2976.0, 0.0, 0.0, 5216.0, 128.0, -3968.0, -3968.0, -3968.0, -2976.0, 0.0, 0.0, 5216.0, 128.0, -3968.0, -3968.0, -3968.0, -2976.0, 0.0, 0.0, 5216.0, 128.0, -3968.0, -3968.0, -3968.0, -2976.0, 0.0, 0.0, 5216.0, 128.0, -3968.0, -3968.0, -3968.0, -2976.0, 0.0];
    #[rustfmt::skip]
    pub const OCV_DERIVATIVES_SOBEL_F32_DX1_K7: [f32; 64] = [0.0, 4.224e+04, 6016.0, -47104.0, -63488.0, -5.952e+04, -3.968e+04, 0.0, 0.0, 4.224e+04, 6016.0, -47104.0, -63488.0, -5.952e+04, -3.968e+04, 0.0, 0.0, 4.224e+04, 6016.0, -47104.0, -63488.0, -5.952e+04, -3.968e+04, 0.0, 0.0, 4.224e+04, 6016.0, -47104.0, -63488.0, -5.952e+04, -3.968e+04, 0.0, 0.0, 4.224e+04, 6016.0, -47104.0, -63488.0, -5.952e+04, -3.968e+04, 0.0, 0.0, 4.224e+04, 6016.0, -47104.0, -63488.0, -5.952e+04, -3.968e+04, 0.0, 0.0, 4.224e+04, 6016.0, -47104.0, -63488.0, -5.952e+04, -3.968e+04, 0.0, 0.0, 4.224e+04, 6016.0, -47104.0, -63488.0, -5.952e+04, -3.968e+04, 0.0];
    #[rustfmt::skip]
    pub const OCV_DERIVATIVES_SOBEL_F32_DX2_K3: [f32; 64] = [1.8e+03, -1024.0, 0.0, 0.0, 0.0, 0.0, 0.0, 248.0, 1.8e+03, -1024.0, 0.0, 0.0, 0.0, 0.0, 0.0, 248.0, 1.8e+03, -1024.0, 0.0, 0.0, 0.0, 0.0, 0.0, 248.0, 1.8e+03, -1024.0, 0.0, 0.0, 0.0, 0.0, 0.0, 248.0, 1.8e+03, -1024.0, 0.0, 0.0, 0.0, 0.0, 0.0, 248.0, 1.8e+03, -1024.0, 0.0, 0.0, 0.0, 0.0, 0.0, 248.0, 1.8e+03, -1024.0, 0.0, 0.0, 0.0, 0.0, 0.0, 248.0, 1.8e+03, -1024.0, 0.0, 0.0, 0.0, 0.0, 0.0, 248.0];
    #[rustfmt::skip]
    pub const OCV_DERIVATIVES_SOBEL_F32_DY1_K3: [f32; 64] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 24.0, 24.0, 24.0, 24.0, 24.0, 24.0, 24.0, 24.0, 24.0, 24.0, 24.0, 24.0, 24.0, 24.0, 24.0, 24.0, 24.0, 24.0, 24.0, 24.0, 24.0, 24.0, 24.0, 24.0, 24.0, 24.0, 24.0, 24.0, 24.0, 24.0, 24.0, 24.0, 24.0, 24.0, 24.0, 24.0, 24.0, 24.0, 24.0, 24.0, 24.0, 24.0, 24.0, 24.0, 24.0, 24.0, 24.0, 24.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    #[rustfmt::skip]
    pub const OCV_DERIVATIVES_SOBEL_U8_DX1_K3_DDEPTH_16S: [i16; 64] = [0, 776, -248, -248, -248, -248, -248, 0, 0, 776, -248, -248, -248, -248, -248, 0, 0, 776, -248, -248, -248, -248, -248, 0, 0, 776, -248, -248, -248, -248, -248, 0, 0, 776, -248, -248, -248, -248, -248, 0, 0, 776, -248, -248, -248, -248, -248, 0, 0, 776, -248, -248, -248, -248, -248, 0, 0, 776, -248, -248, -248, -248, -248, 0];
    #[rustfmt::skip]
    pub const OCV_DERIVATIVES_SOBEL_U8_DX1_K3_DDEPTH_SAME: [u8; 64] = [0, 255, 0, 0, 0, 0, 0, 0, 0, 255, 0, 0, 0, 0, 0, 0, 0, 255, 0, 0, 0, 0, 0, 0, 0, 255, 0, 0, 0, 0, 0, 0, 0, 255, 0, 0, 0, 0, 0, 0, 0, 255, 0, 0, 0, 0, 0, 0, 0, 255, 0, 0, 0, 0, 0, 0, 0, 255, 0, 0, 0, 0, 0, 0];
    #[rustfmt::skip]
    pub const OCV_DERIVATIVES_SOBEL_U8_DX1_K3_SCALE0_3_DELTA0_5_16S: [i16; 64] = [0, 233, -74, -74, -74, -74, -74, 0, 0, 233, -74, -74, -74, -74, -74, 0, 0, 233, -74, -74, -74, -74, -74, 0, 0, 233, -74, -74, -74, -74, -74, 0, 0, 233, -74, -74, -74, -74, -74, 0, 0, 233, -74, -74, -74, -74, -74, 0, 0, 233, -74, -74, -74, -74, -74, 0, 0, 233, -74, -74, -74, -74, -74, 0];
    #[rustfmt::skip]
    pub const OCV_EQUALIZE_HIST_U8_16X16: [u8; 256] = [0, 36, 147, 76, 80, 159, 53, 24, 66, 181, 116, 126, 209, 109, 85, 135, 110, 147, 3, 188, 192, 14, 165, 134, 177, 35, 228, 238, 63, 221, 196, 246, 222, 3, 113, 43, 47, 124, 20, 245, 32, 145, 83, 94, 174, 74, 50, 101, 76, 113, 225, 154, 159, 236, 131, 100, 142, 1, 194, 205, 29, 186, 162, 213, 188, 225, 80, 9, 14, 92, 242, 212, 253, 111, 48, 60, 140, 41, 17, 67, 43, 80, 192, 119, 124, 203, 97, 66, 107, 223, 160, 171, 251, 152, 128, 178, 154, 192, 47, 231, 236, 58, 208, 177, 219, 77, 15, 26, 105, 7, 240, 33, 9, 47, 159, 87, 92, 169, 62, 32, 72, 189, 126, 137, 217, 117, 95, 143, 119, 159, 14, 198, 203, 24, 173, 142, 184, 44, 238, 248, 70, 229, 206, 254, 231, 14, 124, 53, 58, 134, 28, 253, 39, 155, 94, 102, 182, 85, 61, 108, 87, 124, 236, 165, 169, 245, 139, 107, 150, 10, 205, 214, 37, 196, 172, 220, 198, 236, 92, 20, 24, 100, 250, 219, 5, 120, 60, 68, 148, 50, 27, 73, 53, 92, 203, 131, 134, 212, 104, 72, 116, 232, 171, 179, 4, 162, 138, 185, 165, 203, 58, 242, 245, 66, 216, 184, 228, 88, 26, 34, 114, 17, 249, 40, 20, 58, 169, 97, 100, 177, 69, 39, 83, 199, 137, 144, 226, 128, 103, 151, 131, 169, 24, 208, 212, 32, 181, 150, 194, 54, 248, 255, 81, 240, 215, 6];
    #[rustfmt::skip]
    pub const OCV_EQUALIZE_HIST_U8_8X8: [u8; 64] = [0, 227, 194, 162, 130, 97, 65, 32, 4, 231, 198, 166, 134, 101, 69, 36, 8, 235, 202, 170, 138, 105, 73, 40, 12, 239, 206, 174, 142, 109, 77, 45, 16, 243, 210, 178, 146, 113, 81, 49, 20, 247, 215, 182, 150, 117, 85, 53, 24, 251, 219, 186, 154, 121, 89, 57, 28, 255, 223, 190, 158, 125, 93, 61];
    #[rustfmt::skip]
    pub const OCV_EQUALIZE_HIST_U8_CONSTANT: [u8; 9] = [77, 77, 77, 77, 77, 77, 77, 77, 77];
    #[rustfmt::skip]
    pub const OCV_EQUALIZE_HIST_U8_TIE_HALF_EVEN: [u8; 7] = [0, 42, 85, 128, 170, 212, 255];
    #[rustfmt::skip]
    pub const OCV_EQUALIZE_HIST_U8_TWO_VALUES: [u8; 4] = [0, 0, 0, 255];
    #[rustfmt::skip]
    pub const OCV_GAUSSIAN_BLUR_F32_K5X3_SIGMA1_5_SIGMA2_0: [f32; 64] = [153.68356323242188, 159.52847290039062, 165.1067352294922, 164.8468017578125, 133.8468017578125, 102.84680938720703, 79.29167175292969, 70.23713684082031, 154.8367462158203, 160.68167114257812, 166.2599334716797, 166.0, 135.0, 104.0, 80.44486236572266, 71.39032745361328, 157.83676147460938, 163.68167114257812, 169.25994873046875, 169.0, 138.0, 107.0, 83.44486236572266, 74.39032745361328, 160.83676147460938, 166.68167114257812, 172.2599334716797, 172.0, 141.0, 110.0, 86.44486236572266, 77.39032745361328, 163.83676147460938, 169.68167114257812, 175.25994873046875, 175.0, 144.0, 113.0, 89.44485473632812, 80.39032745361328, 166.83676147460938, 172.68167114257812, 178.2599334716797, 178.0, 147.0, 116.0, 92.44486236572266, 83.39032745361328, 169.83676147460938, 175.68167114257812, 181.2599334716797, 181.0, 150.0, 119.0, 95.44486236572266, 86.39032745361328, 170.9899444580078, 176.83485412597656, 182.4131317138672, 182.15318298339844, 151.1531982421875, 120.15319061279297, 96.59805297851562, 87.54351806640625];
    #[rustfmt::skip]
    pub const OCV_GAUSSIAN_BLUR_F32_K3X5_SIGMA1_5_SIGMA2_0: [f32; 64] = [141.35483, 149.04709, 196.84422, 165.84422, 134.84422, 103.84423, 72.84422, 60.92791, 142.23108, 149.92332, 197.72047, 166.72047, 135.72047, 104.720474, 73.72047, 61.804153, 144.5106, 152.20287, 2e+02, 168.99998, 138.0, 107.0, 76.0, 64.08368, 147.5106, 155.20287, 203.0, 172.0, 141.0, 1.1e+02, 79.0, 67.08369, 150.5106, 158.20287, 206.0, 175.0, 144.0, 113.0, 82.0, 70.08369, 153.51059, 161.20285, 209.0, 178.0, 147.0, 116.0, 85.0, 73.08368, 155.79013, 163.48238, 211.27953, 180.27954, 149.27953, 118.27953, 87.279526, 75.363205, 156.66638, 164.35863, 212.15578, 181.15578, 150.15578, 119.15578, 88.15577, 76.239456];
    #[rustfmt::skip]
    pub const OCV_GAUSSIAN_BLUR_F32_K5_SIGMA1_5: [f32; 64] = [154.68097, 160.5259, 166.10416, 165.84422, 134.84422, 103.84423, 80.289085, 71.23455, 155.55722, 161.40215, 166.98041, 166.72047, 135.72047, 104.720474, 81.16534, 72.110794, 157.83676, 163.68167, 169.25995, 168.99998, 138.0, 107.00001, 83.44486, 74.39033, 160.83676, 166.68167, 172.25993, 172.0, 141.0, 1.1e+02, 86.44486, 77.39033, 163.83676, 169.68167, 175.25993, 175.0, 144.0, 113.0, 89.444855, 80.39033, 166.83676, 172.68167, 178.25993, 178.0, 147.0, 116.0, 92.444855, 83.39033, 169.11629, 174.9612, 180.53946, 180.27953, 149.27953, 118.27953, 94.72439, 85.66985, 169.99252, 175.83745, 181.41571, 181.15578, 150.15578, 119.15578, 95.60063, 86.5461];
    #[rustfmt::skip]
    pub const OCV_GAUSSIAN_BLUR_U8_K0_SIGMA1_AUTO_KSIZE: [u8; 64] = [134, 161, 182, 164, 134, 103, 76, 64, 135, 163, 183, 165, 135, 105, 77, 65, 138, 165, 186, 168, 138, 107, 80, 68, 140, 168, 189, 171, 141, 110, 83, 71, 143, 171, 192, 174, 144, 113, 86, 74, 146, 174, 195, 177, 147, 116, 89, 77, 149, 177, 197, 180, 150, 119, 92, 79, 150, 178, 199, 181, 151, 120, 93, 80];
    #[rustfmt::skip]
    pub const OCV_GAUSSIAN_BLUR_U8_K3_SIGMA0: [u8; 64] = [114, 163, 196, 165, 134, 103, 72, 56, 116, 164, 197, 166, 135, 104, 73, 58, 119, 167, 200, 169, 138, 107, 76, 61, 122, 170, 203, 172, 141, 110, 79, 64, 125, 173, 206, 175, 144, 113, 82, 67, 128, 176, 209, 178, 147, 116, 85, 70, 131, 179, 212, 181, 150, 119, 88, 73, 132, 181, 214, 183, 152, 121, 90, 74];
    #[rustfmt::skip]
    pub const OCV_GAUSSIAN_BLUR_U8_K5_SIGMA1_5: [u8; 64] = [155, 160, 166, 166, 135, 104, 80, 71, 156, 161, 167, 167, 136, 105, 81, 72, 158, 163, 169, 169, 138, 107, 84, 75, 161, 166, 172, 172, 141, 110, 87, 78, 164, 169, 175, 175, 144, 113, 90, 81, 167, 172, 178, 178, 147, 116, 93, 84, 170, 175, 180, 180, 149, 118, 95, 86, 171, 176, 181, 181, 150, 119, 96, 87];
    #[rustfmt::skip]
    pub const OCV_GET_DERIV_KERNELS_K1_DX1_DY0_0: [f64; 3] = [-1.0, 0.0, 1.0];
    #[rustfmt::skip]
    pub const OCV_GET_DERIV_KERNELS_K1_DX1_DY0_1: [f64; 1] = [1.0];
    #[rustfmt::skip]
    pub const OCV_GET_DERIV_KERNELS_K3_DX1_DY0_0: [f64; 3] = [-1.0, 0.0, 1.0];
    #[rustfmt::skip]
    pub const OCV_GET_DERIV_KERNELS_K3_DX1_DY0_1: [f64; 3] = [1.0, 2.0, 1.0];
    #[rustfmt::skip]
    pub const OCV_GET_DERIV_KERNELS_K3_DX1_DY1_0: [f64; 3] = [-1.0, 0.0, 1.0];
    #[rustfmt::skip]
    pub const OCV_GET_DERIV_KERNELS_K3_DX1_DY1_1: [f64; 3] = [-1.0, 0.0, 1.0];
    #[rustfmt::skip]
    pub const OCV_GET_DERIV_KERNELS_K3_DX2_DY0_0: [f64; 3] = [1.0, -2.0, 1.0];
    #[rustfmt::skip]
    pub const OCV_GET_DERIV_KERNELS_K3_DX2_DY0_1: [f64; 3] = [1.0, 2.0, 1.0];
    #[rustfmt::skip]
    pub const OCV_GET_DERIV_KERNELS_K5_DX1_DY0_0: [f64; 5] = [-1.0, -2.0, 0.0, 2.0, 1.0];
    #[rustfmt::skip]
    pub const OCV_GET_DERIV_KERNELS_K5_DX1_DY0_1: [f64; 5] = [1.0, 4.0, 6.0, 4.0, 1.0];
    #[rustfmt::skip]
    pub const OCV_GET_DERIV_KERNELS_K5_DX2_DY0_0: [f64; 5] = [1.0, 0.0, -2.0, 0.0, 1.0];
    #[rustfmt::skip]
    pub const OCV_GET_DERIV_KERNELS_K5_DX2_DY0_1: [f64; 5] = [1.0, 4.0, 6.0, 4.0, 1.0];
    #[rustfmt::skip]
    pub const OCV_GET_DERIV_KERNELS_K7_DX1_DY0_0: [f64; 7] = [-1.0, -4.0, -5.0, 0.0, 5.0, 4.0, 1.0];
    #[rustfmt::skip]
    pub const OCV_GET_DERIV_KERNELS_K7_DX1_DY0_1: [f64; 7] = [1.0, 6.0, 15.0, 20.0, 15.0, 6.0, 1.0];
    #[rustfmt::skip]
    pub const OCV_GET_DERIV_KERNELS_SCHARR_DX0_DY1_0: [f64; 3] = [3.0, 10.0, 3.0];
    #[rustfmt::skip]
    pub const OCV_GET_DERIV_KERNELS_SCHARR_DX0_DY1_1: [f64; 3] = [-1.0, 0.0, 1.0];
    #[rustfmt::skip]
    pub const OCV_GET_DERIV_KERNELS_SCHARR_DX1_DY0_0: [f64; 3] = [-1.0, 0.0, 1.0];
    #[rustfmt::skip]
    pub const OCV_GET_DERIV_KERNELS_SCHARR_DX1_DY0_1: [f64; 3] = [3.0, 10.0, 3.0];
    #[rustfmt::skip]
    pub const OCV_GET_GAUSSIAN_KERNEL_N11_SIGMA0: [f64; 11] = [0.008812229292562288, 0.027143577143479373, 0.06511405659938269, 0.12164907301380962, 0.1769983568313557, 0.20056541423882085, 0.1769983568313557, 0.12164907301380962, 0.06511405659938269, 0.027143577143479373, 0.008812229292562288];
    #[rustfmt::skip]
    pub const OCV_GET_GAUSSIAN_KERNEL_N3_SIGMA0: [f64; 3] = [0.25, 0.5, 0.25];
    #[rustfmt::skip]
    pub const OCV_GET_GAUSSIAN_KERNEL_N3_SIGMA1_5: [f64; 3] = [0.30780132912347, 0.38439734175306, 0.30780132912347];
    #[rustfmt::skip]
    pub const OCV_GET_GAUSSIAN_KERNEL_N5_SIGMA0: [f64; 5] = [0.0625, 0.25, 0.375, 0.25, 0.0625];
    #[rustfmt::skip]
    pub const OCV_GET_GAUSSIAN_KERNEL_N5_SIGMA0_8: [f64; 5] = [0.021929644862389366, 0.22851214688447102, 0.49911641650627914, 0.22851214688447102, 0.021929644862389366];
    #[rustfmt::skip]
    pub const OCV_GET_GAUSSIAN_KERNEL_N7_SIGMA0: [f64; 7] = [0.03125, 0.109375, 0.21875, 0.28125, 0.21875, 0.109375, 0.03125];
    #[rustfmt::skip]
    pub const OCV_GET_GAUSSIAN_KERNEL_N7_SIGMA2: [f64; 7] = [0.07015932695902606, 0.13107487896736597, 0.19071282356963734, 0.21610594100794114, 0.19071282356963734, 0.13107487896736597, 0.07015932695902606];
    #[rustfmt::skip]
    pub const OCV_GET_GAUSSIAN_KERNEL_N9_SIGMA0: [f64; 9] = [0.015625, 0.05078125, 0.1171875, 0.19921875, 0.234375, 0.19921875, 0.1171875, 0.05078125, 0.015625];
    #[rustfmt::skip]
    pub const OCV_GET_STRUCTURING_ELEMENT_CROSS_3X3: [u8; 9] = [0, 1, 0, 1, 1, 1, 0, 1, 0];
    #[rustfmt::skip]
    pub const OCV_GET_STRUCTURING_ELEMENT_CROSS_5X5: [u8; 25] = [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0];
    #[rustfmt::skip]
    pub const OCV_GET_STRUCTURING_ELEMENT_CROSS_5X5_ANCHOR11: [u8; 25] = [0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0];
    #[rustfmt::skip]
    pub const OCV_GET_STRUCTURING_ELEMENT_CROSS_6X4: [u8; 24] = [0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0];
    #[rustfmt::skip]
    pub const OCV_GET_STRUCTURING_ELEMENT_CROSS_7X3: [u8; 21] = [0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0];
    #[rustfmt::skip]
    pub const OCV_GET_STRUCTURING_ELEMENT_ELLIPSE_3X3: [u8; 9] = [0, 1, 0, 1, 1, 1, 0, 1, 0];
    #[rustfmt::skip]
    pub const OCV_GET_STRUCTURING_ELEMENT_ELLIPSE_5X5: [u8; 25] = [0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0];
    #[rustfmt::skip]
    pub const OCV_GET_STRUCTURING_ELEMENT_ELLIPSE_9X9: [u8; 81] = [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0];
    #[rustfmt::skip]
    pub const OCV_GET_STRUCTURING_ELEMENT_RECT_3X3: [u8; 9] = [1, 1, 1, 1, 1, 1, 1, 1, 1];
    #[rustfmt::skip]
    pub const OCV_GET_STRUCTURING_ELEMENT_RECT_5X5: [u8; 25] = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1];
    #[rustfmt::skip]
    pub const OCV_GET_STRUCTURING_ELEMENT_RECT_6X4: [u8; 24] = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1];
    #[rustfmt::skip]
    pub const OCV_GET_STRUCTURING_ELEMENT_RECT_7X3: [u8; 21] = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1];
    #[rustfmt::skip]
    pub const OCV_GOOD_FEATURES_TO_TRACK_F32_MAX10_Q0_01_D3: [f32; 16] = [19.0, 2e+01, 12.0, 2e+01, 19.0, 13.0, 12.0, 13.0, 9.0, 9.0, 3.0, 9.0, 9.0, 3.0, 3.0, 3.0];
    #[rustfmt::skip]
    pub const OCV_GOOD_FEATURES_TO_TRACK_F32_MAX3_Q0_01_D3: [f32; 6] = [19.0, 2e+01, 12.0, 2e+01, 19.0, 13.0];
    #[rustfmt::skip]
    pub const OCV_HOUGH_CIRCLES_GRADIENT_DP1_MD10_P100_P10_R0_0: [f32; 3] = [19.5, 18.5, 8.0];
    #[rustfmt::skip]
    pub const OCV_HOUGH_CIRCLES_GRADIENT_DP1_MD10_P100_P10_R5_15: [f32; 3] = [19.5, 18.5, 8.0];
    #[rustfmt::skip]
    pub const OCV_HOUGH_LINES_P_RHO1_THETA_PI180_T10_LEN5_GAP2: [i32; 12] = [0, 0, 16, 16, 14, 19, 14, 0, 0, 5, 19, 5];
    #[rustfmt::skip]
    pub const OCV_HOUGH_LINES_RHO1_THETA_PI180_T15: [f32; 10] = [0.0, 2.338741, 0.0, 2.3736477, 14.0, 0.0, 5.0, 1.553343, -14.0, 3.1241393];
    #[rustfmt::skip]
    pub const OCV_HOUGH_LINES_RHO1_THETA_PI180_T15_THETA_0_5_2_0: [f32; 2] = [5.0, 1.5471976];
    #[rustfmt::skip]
    pub const OCV_HOUGH_LINES_RHO2_THETA_PI90_T10: [f32; 38] = [-1.0, 2.338741, 13.0, 0.034906585, 3.0, 1.5707964, -15.0, 3.106686, 5.0, 1.4660766, -13.0, 3.0019662, 1.0, 2.2340214, -3.0, 2.443461, 15.0, 0.17453292, -5.0, 2.5481806, 7.0, 1.3264502, 3.0, 1.9547688, 3.0, 2.0594885, -9.0, 2.8274333, -11.0, 2.86234, -7.0, 2.6179938, -7.0, 2.687807, -9.0, 2.7227137, -11.0, 2.7925267];
    #[rustfmt::skip]
    pub const OCV_MEDIAN_BLUR_U8_K3: [u8; 64] = [3, 194, 194, 163, 132, 101, 70, 42, 6, 197, 197, 166, 135, 104, 73, 45, 9, 200, 200, 169, 138, 107, 76, 48, 12, 203, 203, 172, 141, 110, 79, 51, 15, 206, 206, 175, 144, 113, 82, 54, 18, 209, 209, 178, 147, 116, 85, 57, 21, 212, 212, 181, 150, 119, 88, 60, 21, 215, 215, 184, 153, 122, 91, 60];
    #[rustfmt::skip]
    pub const OCV_MEDIAN_BLUR_U8_K5: [u8; 64] = [6, 163, 163, 163, 132, 101, 70, 45, 9, 166, 166, 166, 135, 104, 73, 48, 12, 169, 169, 169, 138, 107, 76, 51, 15, 172, 172, 172, 141, 110, 79, 54, 18, 175, 175, 175, 144, 113, 82, 57, 21, 178, 178, 178, 147, 116, 85, 60, 21, 181, 181, 181, 150, 119, 88, 60, 21, 184, 184, 184, 153, 122, 91, 60];
    #[rustfmt::skip]
    pub const OCV_MORPHOLOGY_DILATE_U8_RECT3: [u8; 64] = [228, 228, 228, 197, 166, 135, 104, 73, 231, 231, 231, 200, 169, 138, 107, 76, 234, 234, 234, 203, 172, 141, 110, 79, 237, 237, 237, 206, 175, 144, 113, 82, 240, 240, 240, 209, 178, 147, 116, 85, 243, 243, 243, 212, 181, 150, 119, 88, 246, 246, 246, 215, 184, 153, 122, 91, 246, 246, 246, 215, 184, 153, 122, 91];
    #[rustfmt::skip]
    pub const OCV_MORPHOLOGY_ERODE_F32_RECT3: [f32; 64] = [0.0, 0.0, 163.0, 132.0, 101.0, 7e+01, 39.0, 39.0, 0.0, 0.0, 163.0, 132.0, 101.0, 7e+01, 39.0, 39.0, 3.0, 3.0, 166.0, 135.0, 104.0, 73.0, 42.0, 42.0, 6.0, 6.0, 169.0, 138.0, 107.0, 76.0, 45.0, 45.0, 9.0, 9.0, 172.0, 141.0, 1.1e+02, 79.0, 48.0, 48.0, 12.0, 12.0, 175.0, 144.0, 113.0, 82.0, 51.0, 51.0, 15.0, 15.0, 178.0, 147.0, 116.0, 85.0, 54.0, 54.0, 18.0, 18.0, 181.0, 1.5e+02, 119.0, 88.0, 57.0, 57.0];
    #[rustfmt::skip]
    pub const OCV_MORPHOLOGY_ERODE_U8C3_RECT3: [u8; 126] = [0, 18, 67, 0, 18, 36, 163, 18, 5, 132, 181, 5, 101, 150, 5, 70, 119, 168, 70, 119, 168, 0, 18, 67, 0, 18, 36, 163, 18, 5, 132, 181, 5, 101, 150, 5, 70, 119, 168, 70, 119, 168, 3, 21, 70, 3, 21, 39, 166, 21, 8, 135, 184, 8, 104, 153, 8, 73, 122, 171, 73, 122, 171, 6, 24, 73, 6, 24, 42, 169, 24, 11, 138, 187, 11, 107, 156, 11, 76, 125, 174, 76, 125, 174, 9, 27, 76, 9, 2, 45, 172, 2, 14, 141, 2, 14, 110, 159, 14, 79, 128, 177, 79, 128, 177, 12, 30, 79, 12, 2, 48, 175, 2, 17, 144, 2, 17, 113, 162, 17, 82, 131, 180, 82, 131, 180];
    #[rustfmt::skip]
    pub const OCV_MORPHOLOGY_ERODE_U8_CROSS3: [u8; 64] = [0, 0, 163, 132, 101, 70, 39, 39, 0, 3, 166, 135, 104, 73, 42, 39, 3, 6, 169, 138, 107, 76, 45, 42, 6, 9, 172, 141, 110, 79, 48, 45, 9, 12, 175, 144, 113, 82, 51, 48, 12, 15, 178, 147, 116, 85, 54, 51, 15, 18, 181, 150, 119, 88, 57, 54, 18, 21, 184, 153, 122, 91, 60, 57];
    #[rustfmt::skip]
    pub const OCV_MORPHOLOGY_ERODE_U8_RECT3: [u8; 64] = [0, 0, 163, 132, 101, 70, 39, 39, 0, 0, 163, 132, 101, 70, 39, 39, 3, 3, 166, 135, 104, 73, 42, 42, 6, 6, 169, 138, 107, 76, 45, 45, 9, 9, 172, 141, 110, 79, 48, 48, 12, 12, 175, 144, 113, 82, 51, 51, 15, 15, 178, 147, 116, 85, 54, 54, 18, 18, 181, 150, 119, 88, 57, 57];
    #[rustfmt::skip]
    pub const OCV_MORPHOLOGY_ERODE_U8_RECT3_ANCHOR00: [u8; 64] = [0, 163, 132, 101, 70, 39, 39, 39, 3, 166, 135, 104, 73, 42, 42, 42, 6, 169, 138, 107, 76, 45, 45, 45, 9, 172, 141, 110, 79, 48, 48, 48, 12, 175, 144, 113, 82, 51, 51, 51, 15, 178, 147, 116, 85, 54, 54, 54, 18, 181, 150, 119, 88, 57, 57, 57, 21, 184, 153, 122, 91, 60, 60, 60];
    #[rustfmt::skip]
    pub const OCV_MORPHOLOGY_ERODE_U8_RECT3_ITER0: [u8; 64] = [0, 225, 194, 163, 132, 101, 70, 39, 3, 228, 197, 166, 135, 104, 73, 42, 6, 231, 200, 169, 138, 107, 76, 45, 9, 234, 203, 172, 141, 110, 79, 48, 12, 237, 206, 175, 144, 113, 82, 51, 15, 240, 209, 178, 147, 116, 85, 54, 18, 243, 212, 181, 150, 119, 88, 57, 21, 246, 215, 184, 153, 122, 91, 60];
    #[rustfmt::skip]
    pub const OCV_MORPHOLOGY_ERODE_U8_RECT3_ITER2: [u8; 64] = [0, 0, 0, 101, 70, 39, 39, 39, 0, 0, 0, 101, 70, 39, 39, 39, 0, 0, 0, 101, 70, 39, 39, 39, 3, 3, 3, 104, 73, 42, 42, 42, 6, 6, 6, 107, 76, 45, 45, 45, 9, 9, 9, 110, 79, 48, 48, 48, 12, 12, 12, 113, 82, 51, 51, 51, 15, 15, 15, 116, 85, 54, 54, 54];
    #[rustfmt::skip]
    pub const OCV_MORPHOLOGY_ERODE_U8_RECT3_ITER2_REPLICATE: [u8; 64] = [0, 0, 0, 101, 70, 39, 39, 39, 0, 0, 0, 101, 70, 39, 39, 39, 0, 0, 0, 101, 70, 39, 39, 39, 3, 3, 3, 104, 73, 42, 42, 42, 6, 6, 6, 107, 76, 45, 45, 45, 9, 9, 9, 110, 79, 48, 48, 48, 12, 12, 12, 113, 82, 51, 51, 51, 15, 15, 15, 116, 85, 54, 54, 54];
    #[rustfmt::skip]
    pub const OCV_MORPHOLOGY_ERODE_U8_RECT3_REFLECT101: [u8; 64] = [0, 0, 163, 132, 101, 70, 39, 39, 0, 0, 163, 132, 101, 70, 39, 39, 3, 3, 166, 135, 104, 73, 42, 42, 6, 6, 169, 138, 107, 76, 45, 45, 9, 9, 172, 141, 110, 79, 48, 48, 12, 12, 175, 144, 113, 82, 51, 51, 15, 15, 178, 147, 116, 85, 54, 54, 18, 18, 181, 150, 119, 88, 57, 57];
    #[rustfmt::skip]
    pub const OCV_MORPHOLOGY_ERODE_U8_RECT3_REPLICATE: [u8; 64] = [0, 0, 163, 132, 101, 70, 39, 39, 0, 0, 163, 132, 101, 70, 39, 39, 3, 3, 166, 135, 104, 73, 42, 42, 6, 6, 169, 138, 107, 76, 45, 45, 9, 9, 172, 141, 110, 79, 48, 48, 12, 12, 175, 144, 113, 82, 51, 51, 15, 15, 178, 147, 116, 85, 54, 54, 18, 18, 181, 150, 119, 88, 57, 57];
    #[rustfmt::skip]
    pub const OCV_MORPHOLOGY_MORPH_BLACKHAT_U8_RECT3: [u8; 64] = [228, 3, 3, 3, 3, 3, 3, 34, 225, 0, 0, 0, 0, 0, 0, 31, 225, 0, 0, 0, 0, 0, 0, 31, 225, 0, 0, 0, 0, 0, 0, 31, 225, 0, 0, 0, 0, 0, 0, 31, 225, 0, 0, 0, 0, 0, 0, 31, 225, 0, 0, 0, 0, 0, 0, 31, 225, 0, 0, 0, 0, 0, 0, 31];
    #[rustfmt::skip]
    pub const OCV_MORPHOLOGY_MORPH_CLOSE_U8_RECT3: [u8; 64] = [228, 228, 197, 166, 135, 104, 73, 73, 228, 228, 197, 166, 135, 104, 73, 73, 231, 231, 200, 169, 138, 107, 76, 76, 234, 234, 203, 172, 141, 110, 79, 79, 237, 237, 206, 175, 144, 113, 82, 82, 240, 240, 209, 178, 147, 116, 85, 85, 243, 243, 212, 181, 150, 119, 88, 88, 246, 246, 215, 184, 153, 122, 91, 91];
    #[rustfmt::skip]
    pub const OCV_MORPHOLOGY_MORPH_GRADIENT_U8_RECT3: [u8; 64] = [228, 228, 65, 65, 65, 65, 65, 34, 231, 231, 68, 68, 68, 68, 68, 37, 231, 231, 68, 68, 68, 68, 68, 37, 231, 231, 68, 68, 68, 68, 68, 37, 231, 231, 68, 68, 68, 68, 68, 37, 231, 231, 68, 68, 68, 68, 68, 37, 231, 231, 68, 68, 68, 68, 68, 37, 228, 228, 65, 65, 65, 65, 65, 34];
    #[rustfmt::skip]
    pub const OCV_MORPHOLOGY_MORPH_OPEN_U8_RECT3: [u8; 64] = [0, 163, 163, 163, 132, 101, 70, 39, 3, 166, 166, 166, 135, 104, 73, 42, 6, 169, 169, 169, 138, 107, 76, 45, 9, 172, 172, 172, 141, 110, 79, 48, 12, 175, 175, 175, 144, 113, 82, 51, 15, 178, 178, 178, 147, 116, 85, 54, 18, 181, 181, 181, 150, 119, 88, 57, 18, 181, 181, 181, 150, 119, 88, 57];
    #[rustfmt::skip]
    pub const OCV_MORPHOLOGY_MORPH_OPEN_U8_RECT3_ITER2: [u8; 64] = [0, 101, 101, 101, 101, 101, 70, 39, 3, 104, 104, 104, 104, 104, 73, 42, 6, 107, 107, 107, 107, 107, 76, 45, 9, 110, 110, 110, 110, 110, 79, 48, 12, 113, 113, 113, 113, 113, 82, 51, 15, 116, 116, 116, 116, 116, 85, 54, 15, 116, 116, 116, 116, 116, 85, 54, 15, 116, 116, 116, 116, 116, 85, 54];
    #[rustfmt::skip]
    pub const OCV_MORPHOLOGY_MORPH_TOPHAT_U8_RECT3: [u8; 64] = [0, 62, 31, 0, 0, 0, 0, 0, 0, 62, 31, 0, 0, 0, 0, 0, 0, 62, 31, 0, 0, 0, 0, 0, 0, 62, 31, 0, 0, 0, 0, 0, 0, 62, 31, 0, 0, 0, 0, 0, 0, 62, 31, 0, 0, 0, 0, 0, 0, 62, 31, 0, 0, 0, 0, 0, 3, 65, 34, 3, 3, 3, 3, 3];
    #[rustfmt::skip]
    pub const OCV_PYRAMIDS_BUILD_PYRAMID_U8_8X8_LEVELS_0: [u8; 64] = [0, 225, 194, 163, 132, 101, 70, 39, 3, 228, 197, 166, 135, 104, 73, 42, 6, 231, 200, 169, 138, 107, 76, 45, 9, 234, 203, 172, 141, 110, 79, 48, 12, 237, 206, 175, 144, 113, 82, 51, 15, 240, 209, 178, 147, 116, 85, 54, 18, 243, 212, 181, 150, 119, 88, 57, 21, 246, 215, 184, 153, 122, 91, 60];
    #[rustfmt::skip]
    pub const OCV_PYRAMIDS_BUILD_PYRAMID_U8_8X8_LEVELS_1: [u8; 16] = [139, 180, 134, 76, 143, 184, 138, 80, 149, 190, 144, 86, 154, 196, 150, 92];
    #[rustfmt::skip]
    pub const OCV_PYRAMIDS_BUILD_PYRAMID_U8_8X8_LEVELS_2: [u8; 4] = [162, 135, 168, 141];
    #[rustfmt::skip]
    pub const OCV_PYRAMIDS_BUILD_PYRAMID_U8_8X8_LEVELS_3: [u8; 1] = [152];
    #[rustfmt::skip]
    pub const OCV_PYRAMIDS_DOWN_F32_8X8: [f32; 16] = [139.0, 180.25, 134.25, 76.125, 142.75, 184.0, 138.0, 79.875, 148.75, 1.9e+02, 144.0, 85.875, 154.375, 195.625, 149.625, 91.5];
    #[rustfmt::skip]
    pub const OCV_PYRAMIDS_DOWN_I16_SIGNED: [i16; 16] = [11, 52, 6, -52, 15, 56, 10, -48, 21, 62, 16, -42, 26, 68, 22, -36];
    #[rustfmt::skip]
    pub const OCV_PYRAMIDS_DOWN_U8C3: [u8; 36] = [139, 60, 77, 180, 165, 54, 134, 183, 152, 96, 145, 194, 143, 64, 81, 184, 169, 58, 138, 187, 156, 99, 148, 197, 148, 61, 86, 190, 151, 64, 144, 189, 162, 105, 154, 203];
    #[rustfmt::skip]
    pub const OCV_PYRAMIDS_DOWN_U8_8X8: [u8; 16] = [139, 180, 134, 76, 143, 184, 138, 80, 149, 190, 144, 86, 154, 196, 150, 92];
    #[rustfmt::skip]
    pub const OCV_PYRAMIDS_DOWN_U8_8X8_DSIZE_3X3: [u8; 9] = [139, 180, 134, 143, 184, 138, 149, 190, 144];
    #[rustfmt::skip]
    pub const OCV_PYRAMIDS_DOWN_U8_8X8_REFLECT: [u8; 16] = [84, 179, 133, 73, 88, 184, 138, 78, 94, 190, 144, 84, 100, 196, 150, 90];
    #[rustfmt::skip]
    pub const OCV_PYRAMIDS_DOWN_U8_8X8_REPLICATE: [u8; 16] = [70, 179, 133, 73, 74, 184, 138, 78, 80, 190, 144, 84, 86, 196, 150, 90];
    #[rustfmt::skip]
    pub const OCV_PYRAMIDS_DOWN_U8_9X7: [u8; 20] = [139, 180, 134, 96, 143, 184, 138, 99, 149, 190, 144, 105, 155, 196, 150, 111, 159, 200, 154, 115];
    #[rustfmt::skip]
    pub const OCV_PYRAMIDS_UP_U8_6X6: [u8; 144] = [57, 113, 194, 210, 195, 179, 164, 148, 133, 117, 106, 102, 58, 114, 195, 211, 196, 180, 165, 149, 134, 118, 106, 103, 59, 116, 196, 213, 197, 182, 166, 151, 135, 120, 108, 104, 61, 117, 198, 214, 199, 183, 168, 152, 137, 121, 109, 106, 62, 119, 199, 216, 200, 185, 169, 154, 138, 123, 111, 107, 64, 120, 201, 217, 202, 186, 171, 155, 140, 124, 112, 109, 65, 122, 202, 219, 203, 188, 172, 157, 141, 126, 114, 110, 67, 123, 204, 220, 205, 189, 174, 158, 143, 127, 115, 112, 68, 125, 205, 222, 206, 191, 175, 160, 144, 129, 117, 113, 70, 126, 207, 223, 208, 192, 177, 161, 146, 130, 118, 115, 71, 127, 208, 224, 209, 193, 178, 162, 147, 131, 120, 116, 71, 128, 208, 225, 209, 194, 178, 163, 147, 132, 120, 116];
    #[rustfmt::skip]
    pub const OCV_REMAP_F32_LINEAR_FRAC_CONSTANT: [f32; 64] = [70.59375, 219.46875, 191.375, 163.28125, 135.1875, 107.09375, 79.96875, 51.875, 73.03125, 221.90625, 193.8125, 165.71875, 137.625, 109.53125, 82.40625, 54.3125, 75.375, 224.25, 196.15625, 168.0625, 139.96875, 111.875, 84.75, 56.65625, 77.8125, 226.6875, 198.59375, 170.5, 142.40625, 114.3125, 87.1875, 59.09375, 80.25, 229.125, 201.03125, 172.9375, 144.84375, 116.75, 89.625, 61.53125, 82.59375, 231.46875, 203.375, 175.28125, 147.1875, 119.09375, 91.96875, 63.875, 85.03125, 233.90625, 205.8125, 177.71875, 149.625, 121.53125, 94.40625, 66.3125, 87.375, 236.25, 208.15625, 180.0625, 151.96875, 123.875, 96.75, 68.65625];
    #[rustfmt::skip]
    pub const OCV_REMAP_U8_LINEAR_FRAC_CONSTANT: [u8; 64] = [71, 219, 191, 163, 135, 107, 80, 52, 73, 222, 194, 166, 138, 110, 82, 54, 75, 224, 196, 168, 140, 112, 85, 57, 78, 227, 199, 171, 142, 114, 87, 59, 80, 229, 201, 173, 145, 117, 90, 62, 83, 231, 203, 175, 147, 119, 92, 64, 85, 234, 206, 178, 150, 122, 94, 66, 87, 236, 208, 180, 152, 124, 97, 69];
    #[rustfmt::skip]
    pub const OCV_REMAP_U8_LINEAR_FRAC_REFLECT: [u8; 64] = [231, 6, 6, 231, 200, 169, 138, 107, 234, 9, 9, 234, 203, 172, 141, 110, 237, 12, 12, 237, 206, 175, 144, 113, 240, 15, 15, 240, 209, 178, 147, 116, 243, 18, 18, 243, 212, 181, 150, 119, 246, 21, 21, 246, 215, 184, 153, 122, 246, 21, 21, 246, 215, 184, 153, 122, 243, 18, 18, 243, 212, 181, 150, 119];
    #[rustfmt::skip]
    pub const OCV_REMAP_U8_LINEAR_FRAC_WRAP: [u8; 64] = [76, 45, 6, 231, 200, 169, 138, 107, 79, 48, 9, 234, 203, 172, 141, 110, 82, 51, 12, 237, 206, 175, 144, 113, 85, 54, 15, 240, 209, 178, 147, 116, 88, 57, 18, 243, 212, 181, 150, 119, 91, 60, 21, 246, 215, 184, 153, 122, 70, 39, 0, 225, 194, 163, 132, 101, 73, 42, 3, 228, 197, 166, 135, 104];
    #[rustfmt::skip]
    pub const OCV_REMAP_U8_LINEAR_OUT_CONSTANT: [u8; 64] = [9, 9, 6, 231, 200, 169, 138, 107, 9, 9, 9, 234, 203, 172, 141, 110, 9, 9, 12, 237, 206, 175, 144, 113, 9, 9, 15, 240, 209, 178, 147, 116, 9, 9, 18, 243, 212, 181, 150, 119, 9, 9, 21, 246, 215, 184, 153, 122, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9];
    #[rustfmt::skip]
    pub const OCV_REMAP_U8_LINEAR_OUT_REFLECT101: [u8; 64] = [200, 231, 6, 231, 200, 169, 138, 107, 203, 234, 9, 234, 203, 172, 141, 110, 206, 237, 12, 237, 206, 175, 144, 113, 209, 240, 15, 240, 209, 178, 147, 116, 212, 243, 18, 243, 212, 181, 150, 119, 215, 246, 21, 246, 215, 184, 153, 122, 212, 243, 18, 243, 212, 181, 150, 119, 209, 240, 15, 240, 209, 178, 147, 116];
    #[rustfmt::skip]
    pub const OCV_REMAP_U8_LINEAR_OUT_REPLICATE: [u8; 64] = [6, 6, 6, 231, 200, 169, 138, 107, 9, 9, 9, 234, 203, 172, 141, 110, 12, 12, 12, 237, 206, 175, 144, 113, 15, 15, 15, 240, 209, 178, 147, 116, 18, 18, 18, 243, 212, 181, 150, 119, 21, 21, 21, 246, 215, 184, 153, 122, 21, 21, 21, 246, 215, 184, 153, 122, 21, 21, 21, 246, 215, 184, 153, 122];
    #[rustfmt::skip]
    pub const OCV_REMAP_U8_NEAREST_HALF_CONSTANT: [u8; 64] = [0, 194, 194, 132, 132, 70, 70, 9, 6, 200, 200, 138, 138, 76, 76, 9, 6, 200, 200, 138, 138, 76, 76, 9, 12, 206, 206, 144, 144, 82, 82, 9, 12, 206, 206, 144, 144, 82, 82, 9, 18, 212, 212, 150, 150, 88, 88, 9, 18, 212, 212, 150, 150, 88, 88, 9, 9, 9, 9, 9, 9, 9, 9, 9];
    #[rustfmt::skip]
    pub const OCV_REMAP_U8_NEAREST_OUT_CONSTANT: [u8; 64] = [9, 9, 6, 231, 200, 169, 138, 107, 9, 9, 9, 234, 203, 172, 141, 110, 9, 9, 12, 237, 206, 175, 144, 113, 9, 9, 15, 240, 209, 178, 147, 116, 9, 9, 18, 243, 212, 181, 150, 119, 9, 9, 21, 246, 215, 184, 153, 122, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9];
    #[rustfmt::skip]
    pub const OCV_REMAP_U8_NEAREST_OUT_REFLECT101: [u8; 64] = [200, 231, 6, 231, 200, 169, 138, 107, 203, 234, 9, 234, 203, 172, 141, 110, 206, 237, 12, 237, 206, 175, 144, 113, 209, 240, 15, 240, 209, 178, 147, 116, 212, 243, 18, 243, 212, 181, 150, 119, 215, 246, 21, 246, 215, 184, 153, 122, 212, 243, 18, 243, 212, 181, 150, 119, 209, 240, 15, 240, 209, 178, 147, 116];
    #[rustfmt::skip]
    pub const OCV_REMAP_U8_NEAREST_OUT_REPLICATE: [u8; 64] = [6, 6, 6, 231, 200, 169, 138, 107, 9, 9, 9, 234, 203, 172, 141, 110, 12, 12, 12, 237, 206, 175, 144, 113, 15, 15, 15, 240, 209, 178, 147, 116, 18, 18, 18, 243, 212, 181, 150, 119, 21, 21, 21, 246, 215, 184, 153, 122, 21, 21, 21, 246, 215, 184, 153, 122, 21, 21, 21, 246, 215, 184, 153, 122];
    #[rustfmt::skip]
    pub const OCV_RESIZE_F32_8X8_TO_5X5_LINEAR: [f32; 25] = [68.4, 197.99998, 148.4, 98.8, 49.200005, 73.200005, 202.8, 153.2, 103.600006, 54.000008, 78.00001, 207.6, 158.0, 108.4, 58.800003, 82.8, 212.4, 162.8, 113.200005, 63.600002, 87.6, 217.20001, 167.6, 118.0, 68.4];
    #[rustfmt::skip]
    pub const OCV_RESIZE_U8C3_TO_4X3_LINEAR: [u8; 36] = [86, 39, 88, 192, 241, 34, 137, 186, 203, 83, 132, 181, 92, 45, 94, 198, 247, 40, 143, 192, 209, 89, 138, 187, 98, 51, 100, 204, 141, 46, 149, 198, 215, 95, 144, 193];
    #[rustfmt::skip]
    pub const OCV_RESIZE_U8_5X5_TO_3X3_LINEAR: [u8; 9] = [76, 195, 143, 81, 200, 148, 86, 205, 153];
    #[rustfmt::skip]
    pub const OCV_RESIZE_U8_5X5_TO_7X7_LINEAR: [u8; 49] = [0, 128, 216, 194, 172, 150, 132, 2, 130, 218, 196, 173, 151, 134, 4, 132, 220, 198, 176, 153, 136, 6, 135, 222, 200, 178, 156, 138, 8, 137, 224, 202, 180, 158, 140, 10, 139, 226, 204, 182, 160, 142, 12, 140, 228, 206, 184, 162, 144];
    #[rustfmt::skip]
    pub const OCV_RESIZE_U8_8X8_TO_4X4_LINEAR_IS_AREA: [u8; 16] = [114, 180, 118, 56, 120, 186, 124, 62, 126, 192, 130, 68, 132, 198, 136, 74];
    #[rustfmt::skip]
    pub const OCV_RESIZE_U8_8X8_TO_5X5_LINEAR: [u8; 25] = [68, 198, 148, 99, 49, 73, 203, 153, 103, 54, 78, 208, 158, 108, 59, 83, 212, 163, 113, 63, 87, 217, 167, 118, 68];
    #[rustfmt::skip]
    pub const OCV_RESIZE_U8_8X8_TO_8X8_IDENTITY: [u8; 64] = [0, 225, 194, 163, 132, 101, 70, 39, 3, 228, 197, 166, 135, 104, 73, 42, 6, 231, 200, 169, 138, 107, 76, 45, 9, 234, 203, 172, 141, 110, 79, 48, 12, 237, 206, 175, 144, 113, 82, 51, 15, 240, 209, 178, 147, 116, 85, 54, 18, 243, 212, 181, 150, 119, 88, 57, 21, 246, 215, 184, 153, 122, 91, 60];
    #[rustfmt::skip]
    pub const OCV_THRESHOLD_F32_BINARY_INV_T127_5_M200_1: [f32; 64] = [2e+02, 0.0, 0.0, 0.0, 0.0, 2e+02, 2e+02, 2e+02, 2e+02, 0.0, 0.0, 0.0, 0.0, 2e+02, 2e+02, 2e+02, 2e+02, 0.0, 0.0, 0.0, 0.0, 2e+02, 2e+02, 2e+02, 2e+02, 0.0, 0.0, 0.0, 0.0, 2e+02, 2e+02, 2e+02, 2e+02, 0.0, 0.0, 0.0, 0.0, 2e+02, 2e+02, 2e+02, 2e+02, 0.0, 0.0, 0.0, 0.0, 2e+02, 2e+02, 2e+02, 2e+02, 0.0, 0.0, 0.0, 0.0, 2e+02, 2e+02, 2e+02, 2e+02, 0.0, 0.0, 0.0, 0.0, 2e+02, 2e+02, 2e+02];
    #[rustfmt::skip]
    pub const OCV_THRESHOLD_F32_BINARY_T127_5_M200_1: [f32; 64] = [0.0, 2e+02, 2e+02, 2e+02, 2e+02, 0.0, 0.0, 0.0, 0.0, 2e+02, 2e+02, 2e+02, 2e+02, 0.0, 0.0, 0.0, 0.0, 2e+02, 2e+02, 2e+02, 2e+02, 0.0, 0.0, 0.0, 0.0, 2e+02, 2e+02, 2e+02, 2e+02, 0.0, 0.0, 0.0, 0.0, 2e+02, 2e+02, 2e+02, 2e+02, 0.0, 0.0, 0.0, 0.0, 2e+02, 2e+02, 2e+02, 2e+02, 0.0, 0.0, 0.0, 0.0, 2e+02, 2e+02, 2e+02, 2e+02, 0.0, 0.0, 0.0, 0.0, 2e+02, 2e+02, 2e+02, 2e+02, 0.0, 0.0, 0.0];
    #[rustfmt::skip]
    pub const OCV_THRESHOLD_F32_TOZERO_INV_T127_5_M200_1: [f32; 64] = [0.0, 0.0, 0.0, 0.0, 0.0, 101.0, 7e+01, 39.0, 3.0, 0.0, 0.0, 0.0, 0.0, 104.0, 73.0, 42.0, 6.0, 0.0, 0.0, 0.0, 0.0, 107.0, 76.0, 45.0, 9.0, 0.0, 0.0, 0.0, 0.0, 1.1e+02, 79.0, 48.0, 12.0, 0.0, 0.0, 0.0, 0.0, 113.0, 82.0, 51.0, 15.0, 0.0, 0.0, 0.0, 0.0, 116.0, 85.0, 54.0, 18.0, 0.0, 0.0, 0.0, 0.0, 119.0, 88.0, 57.0, 21.0, 0.0, 0.0, 0.0, 0.0, 122.0, 91.0, 6e+01];
    #[rustfmt::skip]
    pub const OCV_THRESHOLD_F32_TOZERO_T127_5_M200_1: [f32; 64] = [0.0, 225.0, 194.0, 163.0, 132.0, 0.0, 0.0, 0.0, 0.0, 228.0, 197.0, 166.0, 135.0, 0.0, 0.0, 0.0, 0.0, 231.0, 2e+02, 169.0, 138.0, 0.0, 0.0, 0.0, 0.0, 234.0, 203.0, 172.0, 141.0, 0.0, 0.0, 0.0, 0.0, 237.0, 206.0, 175.0, 144.0, 0.0, 0.0, 0.0, 0.0, 2.4e+02, 209.0, 178.0, 147.0, 0.0, 0.0, 0.0, 0.0, 243.0, 212.0, 181.0, 1.5e+02, 0.0, 0.0, 0.0, 0.0, 246.0, 215.0, 184.0, 153.0, 0.0, 0.0, 0.0];
    #[rustfmt::skip]
    pub const OCV_THRESHOLD_F32_TRUNC_T127_5_M200_1: [f32; 64] = [0.0, 127.5, 127.5, 127.5, 127.5, 101.0, 7e+01, 39.0, 3.0, 127.5, 127.5, 127.5, 127.5, 104.0, 73.0, 42.0, 6.0, 127.5, 127.5, 127.5, 127.5, 107.0, 76.0, 45.0, 9.0, 127.5, 127.5, 127.5, 127.5, 1.1e+02, 79.0, 48.0, 12.0, 127.5, 127.5, 127.5, 127.5, 113.0, 82.0, 51.0, 15.0, 127.5, 127.5, 127.5, 127.5, 116.0, 85.0, 54.0, 18.0, 127.5, 127.5, 127.5, 127.5, 119.0, 88.0, 57.0, 21.0, 127.5, 127.5, 127.5, 127.5, 122.0, 91.0, 6e+01];
    #[rustfmt::skip]
    pub const OCV_THRESHOLD_I16_BINARY_T20_5_M90_6_1: [i16; 64] = [0, 91, 91, 91, 91, 0, 0, 0, 0, 91, 91, 91, 91, 0, 0, 0, 0, 91, 91, 91, 91, 0, 0, 0, 0, 91, 91, 91, 91, 0, 0, 0, 0, 91, 91, 91, 91, 0, 0, 0, 0, 91, 91, 91, 91, 0, 0, 0, 0, 91, 91, 91, 91, 0, 0, 0, 0, 91, 91, 91, 91, 91, 0, 0];
    #[rustfmt::skip]
    pub const OCV_THRESHOLD_U8_BINARY_INV_T127_5_M200_1: [u8; 64] = [200, 0, 0, 0, 0, 200, 200, 200, 200, 0, 0, 0, 0, 200, 200, 200, 200, 0, 0, 0, 0, 200, 200, 200, 200, 0, 0, 0, 0, 200, 200, 200, 200, 0, 0, 0, 0, 200, 200, 200, 200, 0, 0, 0, 0, 200, 200, 200, 200, 0, 0, 0, 0, 200, 200, 200, 200, 0, 0, 0, 0, 200, 200, 200];
    #[rustfmt::skip]
    pub const OCV_THRESHOLD_U8_BINARY_T127_5_M200_1: [u8; 64] = [0, 200, 200, 200, 200, 0, 0, 0, 0, 200, 200, 200, 200, 0, 0, 0, 0, 200, 200, 200, 200, 0, 0, 0, 0, 200, 200, 200, 200, 0, 0, 0, 0, 200, 200, 200, 200, 0, 0, 0, 0, 200, 200, 200, 200, 0, 0, 0, 0, 200, 200, 200, 200, 0, 0, 0, 0, 200, 200, 200, 200, 0, 0, 0];
    #[rustfmt::skip]
    pub const OCV_THRESHOLD_U8_BINARY_T_1_0_M200_1: [u8; 64] = [200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200];
    #[rustfmt::skip]
    pub const OCV_THRESHOLD_U8_TOZERO_INV_T127_5_M200_1: [u8; 64] = [0, 0, 0, 0, 0, 101, 70, 39, 3, 0, 0, 0, 0, 104, 73, 42, 6, 0, 0, 0, 0, 107, 76, 45, 9, 0, 0, 0, 0, 110, 79, 48, 12, 0, 0, 0, 0, 113, 82, 51, 15, 0, 0, 0, 0, 116, 85, 54, 18, 0, 0, 0, 0, 119, 88, 57, 21, 0, 0, 0, 0, 122, 91, 60];
    #[rustfmt::skip]
    pub const OCV_THRESHOLD_U8_TOZERO_T127_5_M200_1: [u8; 64] = [0, 225, 194, 163, 132, 0, 0, 0, 0, 228, 197, 166, 135, 0, 0, 0, 0, 231, 200, 169, 138, 0, 0, 0, 0, 234, 203, 172, 141, 0, 0, 0, 0, 237, 206, 175, 144, 0, 0, 0, 0, 240, 209, 178, 147, 0, 0, 0, 0, 243, 212, 181, 150, 0, 0, 0, 0, 246, 215, 184, 153, 0, 0, 0];
    #[rustfmt::skip]
    pub const OCV_THRESHOLD_U8_TRUNC_T127_5_M200_1: [u8; 64] = [0, 127, 127, 127, 127, 101, 70, 39, 3, 127, 127, 127, 127, 104, 73, 42, 6, 127, 127, 127, 127, 107, 76, 45, 9, 127, 127, 127, 127, 110, 79, 48, 12, 127, 127, 127, 127, 113, 82, 51, 15, 127, 127, 127, 127, 116, 85, 54, 18, 127, 127, 127, 127, 119, 88, 57, 21, 127, 127, 127, 127, 122, 91, 60];
    #[rustfmt::skip]
    pub const OCV_WARP_PERSPECTIVE_U8_IDENTITY_LINEAR_CONSTANT: [u8; 64] = [0, 225, 194, 163, 132, 101, 70, 39, 3, 228, 197, 166, 135, 104, 73, 42, 6, 231, 200, 169, 138, 107, 76, 45, 9, 234, 203, 172, 141, 110, 79, 48, 12, 237, 206, 175, 144, 113, 82, 51, 15, 240, 209, 178, 147, 116, 85, 54, 18, 243, 212, 181, 150, 119, 88, 57, 21, 246, 215, 184, 153, 122, 91, 60];
    #[rustfmt::skip]
    pub const OCV_WARP_PERSPECTIVE_U8_IDENTITY_NEAREST_REPLICATE: [u8; 64] = [0, 225, 194, 163, 132, 101, 70, 39, 3, 228, 197, 166, 135, 104, 73, 42, 6, 231, 200, 169, 138, 107, 76, 45, 9, 234, 203, 172, 141, 110, 79, 48, 12, 237, 206, 175, 144, 113, 82, 51, 15, 240, 209, 178, 147, 116, 85, 54, 18, 243, 212, 181, 150, 119, 88, 57, 21, 246, 215, 184, 153, 122, 91, 60];
    #[rustfmt::skip]
    pub const OCV_WARP_PERSPECTIVE_U8_PERSP_LINEAR_CONSTANT: [u8; 64] = [74, 129, 101, 75, 53, 35, 20, 10, 87, 218, 186, 155, 122, 88, 52, 16, 69, 224, 193, 161, 128, 94, 58, 18, 51, 231, 200, 168, 135, 100, 64, 20, 27, 237, 206, 174, 140, 106, 70, 24, 16, 228, 213, 181, 147, 112, 75, 27, 19, 203, 220, 181, 135, 94, 58, 21, 9, 9, 9, 9, 9, 9, 9, 9];
    #[rustfmt::skip]
    pub const OCV_WARP_PERSPECTIVE_U8_PERSP_NEAREST_CONSTANT: [u8; 64] = [0, 225, 194, 9, 9, 9, 9, 9, 3, 228, 197, 166, 135, 104, 42, 9, 6, 231, 200, 169, 138, 107, 45, 9, 9, 234, 203, 172, 141, 110, 48, 9, 12, 237, 206, 175, 144, 113, 85, 9, 18, 243, 212, 181, 150, 119, 88, 9, 21, 246, 215, 184, 153, 122, 91, 9, 9, 9, 9, 9, 9, 9, 9, 9];
    #[rustfmt::skip]
    pub const OCV_WARP_PERSPECTIVE_U8_PERSP_NEAREST_REPLICATE: [u8; 64] = [0, 225, 194, 163, 132, 70, 39, 39, 3, 228, 197, 166, 135, 104, 42, 42, 6, 231, 200, 169, 138, 107, 45, 45, 9, 234, 203, 172, 141, 110, 48, 48, 12, 237, 206, 175, 144, 113, 85, 54, 18, 243, 212, 181, 150, 119, 88, 57, 21, 246, 215, 184, 153, 122, 91, 60, 21, 246, 215, 184, 153, 122, 91, 60];
    #[rustfmt::skip]
    pub const OCV_WARP_PERSPECTIVE_U8_SHIFT_NEAREST_CONSTANT: [u8; 64] = [0, 0, 194, 194, 132, 132, 70, 70, 3, 3, 197, 197, 135, 135, 73, 73, 6, 6, 200, 200, 138, 138, 76, 76, 9, 9, 203, 203, 141, 141, 79, 79, 12, 12, 206, 206, 144, 144, 82, 82, 15, 15, 209, 209, 147, 147, 85, 85, 18, 18, 212, 212, 150, 150, 88, 88, 21, 21, 215, 215, 153, 153, 91, 91];
}

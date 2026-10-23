/*
 *  parity_core.rs
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

//! OpenCV 4.10.0 parity pins for the `core` module (`docs/parity/4.10.0/core.md`).
//!
//! Every expected value in this file was produced by a script under `scripts/opencv_ref/`
//! run in `cpp_ref/.venv` (opencv-contrib-python 4.10.0.84, `cv2.ipp.setUseIPP(False)`,
//! Windows x86-64) or by the pinned opencv.js 4.10.0 (`core_convert_to.js`). The script
//! that produced each block is named in the comment above it.
//!
//! Tests that pass pin rows that already match OpenCV. Tests marked `#[ignore]` document a
//! `diverges` row; they fail when run (`cargo test --test parity_core -- --ignored`) and are
//! the executable form of the measurement.
//!
//! A fix for #N removes the `#[ignore]` of the tests that cite it and deletes the matching
//! current-behaviour pin, if any; the test then guards the row (rule 11 of
//! `docs/parity/AGENTS.md`).

// The literals below are OpenCV's outputs as the scripts printed them (f32 values such as
// 1.4142135 or 0.031622779), so they are kept verbatim instead of being replaced by constants
// or rounded.

// `add_scalar`, `add_weighted`, `compare`, `convert_scale_abs`, `copy_make_border`, `exp`,
// `in_range`, `max`, `min`, `pow`, `sqrt` and the `*_scalar` variants are not re-exported by
// `purecv::core` (see the D1 notes in docs/parity/4.10.0/core.md), so the submodules are imported.
use purecv::core::arithm::*;
use purecv::core::structural::copy_make_border;
use purecv::core::types::{CmpTypes, NormTypes, ReduceTypes, Scalar, TermCriteria, TermType};
use purecv::core::*;

fn row<T: Clone + Default>(data: &[T]) -> Matrix<T> {
    Matrix::from_vec(1, data.len(), 1, data.to_vec())
}

fn assert_close(actual: &[f64], expected: &[f64], tol: f64, what: &str) {
    assert_eq!(actual.len(), expected.len(), "{what}: length");
    for (i, (a, e)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (a - e).abs() <= tol,
            "{what}[{i}]: purecv {a} vs OpenCV {e} (tol {tol})"
        );
    }
}

// ---------------------------------------------------------------------------------------
// Matrix::convert_to (core_convert_to.js, core_convert_arith.py)
// ---------------------------------------------------------------------------------------

/// u8 -> f32 is exact in both libraries.
#[test]
fn convert_to_u8_to_f32_matches_opencv() {
    let src = row(&[0u8, 1, 100, 127, 128, 200, 250, 255, 7, 5, 1, 9]);
    let dst = src.convert_to::<f32>().unwrap();
    assert_eq!(
        dst.data,
        vec![0.0, 1.0, 100.0, 127.0, 128.0, 200.0, 250.0, 255.0, 7.0, 5.0, 1.0, 9.0]
    );
}

/// OpenCV 4.10.0: `Mat::convertTo` is `saturate_cast` (round half to even, then clamp).
/// opencv.js 4.10.0 and cv2 agree: f32 [0.5, 1.5, 2.5, 3.5, 254.5, 254.6, 255.5, 300, -1]
/// -> u8 [0, 2, 2, 4, 254, 255, 255, 255, 0].
#[test]
#[ignore = "diverges from OpenCV 4.10.0, see #192"]
fn convert_to_f32_to_u8_rounds_half_even_and_saturates() {
    let src = row(&[0.5f32, 1.5, 2.5, 3.5, 254.5, 254.6, 255.5, 300.0, -1.0]);
    let dst = src.convert_to::<u8>().expect("OpenCV never fails on range");
    assert_eq!(dst.data, vec![0u8, 2, 2, 4, 254, 255, 255, 255, 0]);
}

/// Today the integer conversion truncates and the out-of-range element returns `Err`
/// (the fact recorded in #192). This pins the current behaviour so that a fix is visible.
#[test]
fn convert_to_f32_to_u8_current_behaviour_truncates_and_errors() {
    let src = row(&[0.5f32, 1.5, 2.5, 254.6]);
    assert_eq!(src.convert_to::<u8>().unwrap().data, vec![0u8, 1, 2, 254]);
    assert!(row(&[300.0f32]).convert_to::<u8>().is_err());
    assert!(row(&[-1.0f32]).convert_to::<u8>().is_err());
}

// ---------------------------------------------------------------------------------------
// add / subtract / multiply / divide on u8 (core_convert_arith.py, "u8_binary")
// ---------------------------------------------------------------------------------------

const A_U8: [u8; 12] = [0, 1, 100, 127, 128, 200, 250, 255, 7, 5, 1, 9];
const B_U8: [u8; 12] = [0, 1, 3, 100, 128, 255, 7, 2, 2, 2, 3, 4];

/// OpenCV: `saturate_cast<uchar>(a + b)`. The `simd` u8 kernel already saturates (#208), so
/// this test runs and passes with that feature and is an expected failure without it.
#[test]
#[cfg_attr(
    not(feature = "simd"),
    ignore = "diverges from OpenCV 4.10.0, see #208"
)]
fn add_u8_saturates() {
    let dst = add(&row(&A_U8), &row(&B_U8)).unwrap();
    assert_eq!(
        dst.data,
        vec![0u8, 2, 103, 227, 255, 255, 255, 255, 9, 7, 4, 13]
    );
}

/// OpenCV: `saturate_cast<uchar>(a - b)` clamps at 0. The `simd` u8 kernel already saturates
/// (#208), so this test runs and passes with that feature and is an expected failure without it.
#[test]
#[cfg_attr(
    not(feature = "simd"),
    ignore = "diverges from OpenCV 4.10.0, see #208"
)]
fn subtract_u8_saturates_at_zero() {
    let dst = subtract(&row(&B_U8), &row(&A_U8)).unwrap();
    assert_eq!(dst.data, vec![0u8, 0, 0, 0, 0, 55, 0, 0, 0, 0, 2, 0]);
}

/// OpenCV: `saturate_cast<uchar>(a * b)`.
#[test]
#[ignore = "diverges from OpenCV 4.10.0, see #208"]
fn multiply_u8_saturates() {
    let dst = multiply(&row(&A_U8), &row(&B_U8)).unwrap();
    assert_eq!(
        dst.data,
        vec![0u8, 1, 255, 255, 255, 255, 255, 255, 14, 10, 3, 36]
    );
}

/// Sums that fit in u8 match (no saturation involved).
#[test]
fn add_u8_without_overflow_matches_opencv() {
    let a = row(&[0u8, 1, 100, 7, 5, 1, 9]);
    let b = row(&[0u8, 1, 3, 2, 2, 3, 4]);
    assert_eq!(add(&a, &b).unwrap().data, vec![0u8, 2, 103, 9, 7, 4, 13]);
    // 1 - 3 would underflow (a debug-build panic without `simd`): use a >= b here.
    let b2 = row(&[0u8, 1, 3, 2, 2, 1, 4]);
    assert_eq!(
        subtract(&a, &b2).unwrap().data,
        vec![0u8, 0, 97, 5, 3, 0, 5]
    );
}

/// OpenCV: `b != 0 ? saturate_cast<uchar>(float(a) / b) : 0`, so 7/2 = 3.5 -> 4 and
/// 250/7 = 35.7 -> 36. purecv divides in integers (7/2 = 3).
#[test]
#[ignore = "diverges from OpenCV 4.10.0, see #209"]
fn divide_u8_rounds_the_float_quotient() {
    let dst = divide(&row(&A_U8), &row(&B_U8)).unwrap();
    assert_eq!(dst.data, vec![0u8, 1, 33, 1, 1, 1, 36, 128, 4, 2, 0, 2]);
}

/// Division by zero gives 0 in both libraries, and quotients without a fractional part match.
#[test]
fn divide_u8_by_zero_is_zero_matches_opencv() {
    let a = row(&[0u8, 100, 255, 1, 128]);
    let b = row(&[0u8, 0, 0, 1, 128]);
    assert_eq!(divide(&a, &b).unwrap().data, vec![0u8, 0, 0, 1, 1]);
}

/// OpenCV: float division is plain IEEE division (`a / b`), so x/0 is inf, -inf or nan.
/// purecv maps every division by zero to 0, for floats too.
#[test]
#[ignore = "diverges from OpenCV 4.10.0, see #209"]
fn divide_f32_by_zero_is_ieee() {
    let a = row(&[1.0f32, -1.0, 0.0, 4.0]);
    let b = row(&[0.0f32, 0.0, 0.0, 2.0]);
    let d = divide(&a, &b).unwrap().data;
    assert_eq!(d[3], 2.0);
    assert!(
        d[0] == f32::INFINITY && d[1] == f32::NEG_INFINITY && d[2].is_nan(),
        "{d:?}"
    );
}

/// The current behaviour, pinned so that a fix shows up: x/0 is 0 for f32 as well (the fact
/// recorded in #209; the fix deletes this test).
#[test]
fn divide_f32_by_zero_current_behaviour_is_zero() {
    let a = row(&[1.0f32, -1.0, 0.0, 4.0]);
    let b = row(&[0.0f32, 0.0, 0.0, 2.0]);
    assert_eq!(divide(&a, &b).unwrap().data, vec![0.0f32, 0.0, 0.0, 2.0]);
}

/// absdiff, min and max on u8 match element for element.
#[test]
fn absdiff_min_max_u8_match_opencv() {
    let a = row(&A_U8);
    let b = row(&B_U8);
    assert_eq!(
        absdiff(&a, &b).unwrap().data,
        vec![0u8, 0, 97, 27, 0, 55, 243, 253, 5, 3, 2, 5]
    );
    assert_eq!(
        min(&a, &b).unwrap().data,
        vec![0u8, 1, 3, 100, 128, 200, 7, 2, 2, 2, 1, 4]
    );
    assert_eq!(
        max(&a, &b).unwrap().data,
        vec![0u8, 1, 100, 127, 128, 255, 250, 255, 7, 5, 3, 9]
    );
}

// ---------------------------------------------------------------------------------------
// matrix (+ - * /) scalar on u8 (core_convert_arith.py, "u8_scalar")
// ---------------------------------------------------------------------------------------

/// OpenCV converts the scalar to the working type with `saturate_cast` (10.7 -> 11,
/// -1.5 -> -2, 300 -> 300 then the sum saturates to 255). purecv truncates 10.7 to 10 and
/// turns every scalar outside the u8 range into 0.
#[test]
#[ignore = "diverges from OpenCV 4.10.0, see #210"]
fn add_scalar_u8_rounds_and_saturates_the_scalar() {
    let a = row(&A_U8);
    assert_eq!(
        add_scalar(&a, Scalar::new(10.7, 0.0, 0.0, 0.0))
            .unwrap()
            .data,
        vec![11u8, 12, 111, 138, 139, 211, 255, 255, 18, 16, 12, 20]
    );
    assert_eq!(
        add_scalar(&a, Scalar::new(300.0, 0.0, 0.0, 0.0))
            .unwrap()
            .data,
        vec![255u8; 12]
    );
    assert_eq!(
        add_scalar(&a, Scalar::new(-1.5, 0.0, 0.0, 0.0))
            .unwrap()
            .data,
        vec![0u8, 0, 98, 125, 126, 198, 248, 253, 5, 3, 0, 7]
    );
}

/// OpenCV: `saturate_cast<uchar>(a * 0.3)` computed in double, so 9 * 0.3 = 2.7 -> 3 and
/// 127 / 0.4 = 317.5 -> 255. purecv truncates (2) and maps out-of-range results to 0.
#[test]
#[ignore = "diverges from OpenCV 4.10.0, see #210"]
fn multiply_divide_scalar_u8_round_and_saturate() {
    let a = row(&A_U8);
    assert_eq!(
        multiply_scalar(&a, Scalar::new(0.3, 0.0, 0.0, 0.0))
            .unwrap()
            .data,
        vec![0u8, 0, 30, 38, 38, 60, 75, 76, 2, 2, 0, 3]
    );
    assert_eq!(
        divide_scalar(&a, Scalar::new(0.4, 0.0, 0.0, 0.0))
            .unwrap()
            .data,
        vec![0u8, 2, 250, 255, 255, 255, 255, 255, 18, 12, 2, 22]
    );
    assert_eq!(
        divide_scalar(&a, Scalar::new(3.0, 0.0, 0.0, 0.0))
            .unwrap()
            .data,
        vec![0u8, 0, 33, 42, 43, 67, 83, 85, 2, 2, 0, 3]
    );
}

/// Dividing by a zero scalar gives 0 in both libraries.
#[test]
fn divide_scalar_u8_by_zero_matches_opencv() {
    let a = row(&A_U8);
    assert_eq!(
        divide_scalar(&a, Scalar::new(0.0, 0.0, 0.0, 0.0))
            .unwrap()
            .data,
        vec![0u8; 12]
    );
}

// ---------------------------------------------------------------------------------------
// add_weighted / convert_scale_abs (core_convert_arith.py)
// ---------------------------------------------------------------------------------------

/// f32 inputs: the weighted sum is exact in both libraries.
#[test]
fn add_weighted_f32_matches_opencv() {
    let a: Vec<f32> = A_U8.iter().map(|&v| v as f32).collect();
    let b: Vec<f32> = B_U8.iter().map(|&v| v as f32).collect();
    let dst = add_weighted(&row(&a), 0.5, &row(&b), 0.5, 0.0).unwrap();
    assert_eq!(
        dst.data,
        vec![0.0f32, 1.0, 51.5, 113.5, 128.0, 227.5, 128.5, 128.5, 4.5, 3.5, 2.0, 6.5]
    );
}

/// OpenCV: `saturate_cast<uchar>(a*alpha + b*beta + gamma)` in float: 51.5 -> 52,
/// 3.5 -> 4, 128.5 -> 128 (ties to even), 910 -> 255. purecv truncates and maps
/// out-of-range sums to 0.
#[test]
#[ignore = "diverges from OpenCV 4.10.0, see #211"]
fn add_weighted_u8_rounds_half_even_and_saturates() {
    let a = row(&A_U8);
    let b = row(&B_U8);
    assert_eq!(
        add_weighted(&a, 0.5, &b, 0.5, 0.0).unwrap().data,
        vec![0u8, 1, 52, 114, 128, 228, 128, 128, 4, 4, 2, 6]
    );
    assert_eq!(
        add_weighted(&a, 2.0, &b, 2.0, 0.0).unwrap().data,
        vec![0u8, 4, 206, 255, 255, 255, 255, 255, 18, 14, 8, 26]
    );
}

/// Integer input, alpha 1: `|x|` then clamp; no rounding is involved and the two agree.
#[test]
fn convert_scale_abs_i16_alpha1_matches_opencv() {
    let src = row(&[-300i16, -2, -1, 0, 1, 2, 127, 128, 254, 255, 256, 1000]);
    let dst = convert_scale_abs(&src, 1.0, 0.0).unwrap();
    assert_eq!(
        dst.data,
        vec![255u8, 2, 1, 0, 1, 2, 127, 128, 254, 255, 255, 255]
    );
}

/// OpenCV: `saturate_cast<uchar>(|x*alpha + beta|)` rounds half to even: 0.5 -> 0,
/// 2.5 -> 2, -2.5 -> 2; and 1 * 0.5 = 0.5 -> 0. purecv uses `round()` (half away from
/// zero): 1, 3, 3 and 1.
#[test]
#[ignore = "diverges from OpenCV 4.10.0, see #212"]
fn convert_scale_abs_rounds_half_to_even() {
    let src = row(&[
        0.5f32, 1.5, 2.5, -0.5, -1.5, -2.5, 254.5, 255.5, -255.5, 300.0,
    ]);
    let dst = convert_scale_abs(&src, 1.0, 0.0).unwrap();
    assert_eq!(dst.data, vec![0u8, 2, 2, 0, 2, 2, 254, 255, 255, 255]);
    let src = row(&[-300i16, -2, -1, 0, 1, 2, 127, 128, 254, 255, 256, 1000]);
    assert_eq!(
        convert_scale_abs(&src, 0.5, 0.0).unwrap().data,
        vec![150u8, 1, 0, 0, 0, 1, 64, 64, 127, 128, 128, 255]
    );
}

// ---------------------------------------------------------------------------------------
// norm / normalize (core_stats.py)
// ---------------------------------------------------------------------------------------

const NORM_SRC: [f32; 6] = [3.0, -4.0, 0.0, 12.0, -1.5, 2.5];

#[test]
fn norm_inf_l1_l2_match_opencv() {
    let f = row(&NORM_SRC);
    assert_eq!(norm(&f, NormTypes::Inf, None).unwrap(), 12.0);
    assert_eq!(norm(&f, NormTypes::L1, None).unwrap(), 23.0);
    assert!((norm(&f, NormTypes::L2, None).unwrap() - 13.322912594474229).abs() < 1e-12);
    let mask = row(&[1u8, 0, 1, 0, 1, 0]);
    assert!((norm(&f, NormTypes::L2, Some(&mask)).unwrap() - 3.3541019662496847).abs() < 1e-12);
    assert_eq!(norm(&f, NormTypes::Inf, Some(&mask)).unwrap(), 3.0);
    assert_eq!(norm(&f, NormTypes::L1, Some(&mask)).unwrap(), 4.5);
    // multi-channel: the norm runs over every element, as in OpenCV
    let two_ch = Matrix::from_vec(1, 2, 2, vec![3.0f32, -4.0, 0.0, 12.0]);
    assert_eq!(norm(&two_ch, NormTypes::L2, None).unwrap(), 13.0);
    assert_eq!(norm(&two_ch, NormTypes::Inf, None).unwrap(), 12.0);
}

/// OpenCV: `norm(src, NORM_L2SQR)` = 177.5 and `norm(u8, NORM_HAMMING)` = popcount (12).
#[test]
#[ignore = "diverges from OpenCV 4.10.0, see #214"]
fn norm_l2sqr_and_hamming_are_supported_by_opencv() {
    assert_eq!(
        norm(&row(&NORM_SRC), NormTypes::L2Sqr, None).unwrap(),
        177.5
    );
    let bits = row(&[0b1011u8, 0xFF, 0, 0b1000_0000]);
    assert_eq!(norm(&bits, NormTypes::Hamming, None).unwrap(), 12.0);
    assert_eq!(norm(&bits, NormTypes::Hamming2, None).unwrap(), 7.0);
}

#[test]
fn normalize_f32_matches_opencv() {
    let src = row(&[10.0f32, 20.0, 30.0, 250.0]);
    let mut dst = Matrix::<f32>::new(1, 4, 1);
    normalize(&src, &mut dst, 0.0, 255.0, NormTypes::MinMax, -1, None).unwrap();
    assert_eq!(dst.data, vec![0.0f32, 10.625, 21.25, 255.0]);
    normalize(&src, &mut dst, 1.0, 0.0, NormTypes::L2, -1, None).unwrap();
    assert_close(
        &dst.data.iter().map(|&v| v as f64).collect::<Vec<_>>(),
        &[
            0.03955938667058945,
            0.0791187733411789,
            0.11867816746234894,
            0.9889847040176392,
        ],
        1e-7,
        "normalize L2",
    );
    normalize(&src, &mut dst, 1.0, 0.0, NormTypes::L1, -1, None).unwrap();
    assert_close(
        &dst.data.iter().map(|&v| v as f64).collect::<Vec<_>>(),
        &[
            0.032258063554763794,
            0.06451612710952759,
            0.09677419066429138,
            0.8064516186714172,
        ],
        1e-7,
        "normalize L1",
    );
    normalize(&src, &mut dst, 1.0, 0.0, NormTypes::Inf, -1, None).unwrap();
    assert_close(
        &dst.data.iter().map(|&v| v as f64).collect::<Vec<_>>(),
        &[
            0.04000000283122063,
            0.08000000566244125,
            0.12000000476837158,
            1.0,
        ],
        1e-7,
        "normalize INF",
    );
    // a constant source maps to alpha (scale 0) in both libraries
    let flat = row(&[5.0f32, 5.0, 5.0]);
    let mut d3 = Matrix::<f32>::new(1, 3, 1);
    normalize(&flat, &mut d3, 0.0, 255.0, NormTypes::MinMax, -1, None).unwrap();
    assert_eq!(d3.data, vec![0.0f32; 3]);
    // mask: unmasked destination elements are left untouched in both libraries
    let s8 = row(&[10u8, 20, 30, 250]);
    let mask = row(&[1u8, 1, 0, 1]);
    let mut d8 = row(&[7u8, 7, 7, 7]);
    normalize(&s8, &mut d8, 0.0, 255.0, NormTypes::MinMax, -1, Some(&mask)).unwrap();
    assert_eq!(d8.data[2], 7);
    assert_eq!(d8.data[0], 0);
    assert_eq!(d8.data[3], 255);
}

/// OpenCV writes integer outputs through `convertTo` (saturate_cast): u8 MINMAX 0..255 of
/// [10, 20, 30, 250] is [0, 11, 21, 255]; purecv truncates to [0, 10, 21, 255].
#[test]
#[ignore = "diverges from OpenCV 4.10.0, see #215"]
fn normalize_u8_minmax_rounds_like_convert_to() {
    let s8 = row(&[10u8, 20, 30, 250]);
    let mut d8 = Matrix::<u8>::new(1, 4, 1);
    normalize(&s8, &mut d8, 0.0, 255.0, NormTypes::MinMax, -1, None).unwrap();
    assert_eq!(d8.data, vec![0u8, 11, 21, 255]);
    normalize(&s8, &mut d8, 0.0, 100.0, NormTypes::MinMax, -1, None).unwrap();
    assert_eq!(d8.data, vec![0u8, 4, 8, 100]);
}

/// OpenCV sorts alpha/beta (`dmin = MIN(a, b)`), so (255, 0) equals (0, 255); purecv uses
/// them as given and inverts the range.
#[test]
#[ignore = "diverges from OpenCV 4.10.0, see #215"]
fn normalize_minmax_swapped_bounds_match_sorted_bounds() {
    let src = row(&[10.0f32, 20.0, 30.0, 250.0]);
    let mut dst = Matrix::<f32>::new(1, 4, 1);
    normalize(&src, &mut dst, 255.0, 0.0, NormTypes::MinMax, -1, None).unwrap();
    assert_eq!(dst.data, vec![0.0f32, 10.625, 21.25, 255.0]);
}

// ---------------------------------------------------------------------------------------
// sum / mean / mean_std_dev / min_max_loc (core_stats.py)
// ---------------------------------------------------------------------------------------

#[test]
fn sum_mean_mean_std_dev_match_opencv() {
    let bgr = Matrix::from_vec(2, 2, 3, vec![1u8, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
    assert_eq!(sum(&bgr).v, [22.0, 26.0, 30.0, 0.0]);
    assert_eq!(mean(&bgr).v, [5.5, 6.5, 7.5, 0.0]);
    let (m, s) = mean_std_dev(&bgr);
    assert_eq!(m.v, [5.5, 6.5, 7.5, 0.0]);
    assert_close(&s.v[..3], &[3.3541019662496847; 3], 1e-12, "u8 std");
    let f3 = Matrix::from_vec(
        2,
        2,
        3,
        vec![
            1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 13.0,
        ],
    );
    let (m, s) = mean_std_dev(&f3);
    assert_eq!(m.v[..3], [5.5, 6.5, 7.75]);
    assert_close(
        &s.v[..3],
        &[3.3541019662496847, 3.3541019662496847, 3.6996621467371855],
        1e-12,
        "f32 std",
    );
}

#[test]
fn min_max_loc_matches_opencv() {
    let m = Matrix::from_vec(
        3,
        4,
        1,
        vec![
            5.0f32, 1.0, 9.0, 1.0, 9.0, 2.0, 7.0, 3.0, 1.0, 8.0, 2.0, 9.0,
        ],
    );
    // OpenCV: minVal 1 at (x=1, y=0), maxVal 9 at (x=2, y=0): first occurrence wins.
    assert_eq!(min_max_loc(&m), (1.0, 9.0, (1, 0), (2, 0)));
    // NaN is skipped by both.
    let mut n = m.clone();
    n.data[1] = f32::NAN;
    assert_eq!(min_max_loc(&n), (1.0, 9.0, (3, 0), (2, 0)));
    let u = Matrix::from_vec(2, 2, 1, vec![3u8, 250, 0, 7]);
    assert_eq!(min_max_loc(&u), (0.0, 250.0, (0, 1), (1, 0)));
}

// ---------------------------------------------------------------------------------------
// Constants, initialisers, has_no_zero (core_mat_initializers.js, core_misc.py)
// ---------------------------------------------------------------------------------------

/// OpenCV's `CV_LOG2` is ln 2 = 0.693147… (`cvdef.h`); purecv's is log2(e) = 1.442695….
#[test]
#[ignore = "diverges from OpenCV 4.10.0, see #226"]
fn cv_log2_is_ln_2() {
    assert_eq!(purecv::core::constants::CV_LOG2, core::f64::consts::LN_2);
}

/// Single-channel `ones` and `eye` match opencv.js 4.10.0 (`ones_2x3_8uc1`, `eye_2x3_32fc1`).
#[test]
fn ones_and_eye_single_channel_match_opencv() {
    assert_eq!(Matrix::<u8>::ones(2, 3, 1).data, ONES_2X3_8UC1.to_vec());
    assert_eq!(Matrix::<f32>::eye(2, 3, 1).data, EYE_2X3_32FC1.to_vec());
}

/// OpenCV's `Mat::ones` / `Mat::eye` write `Scalar(alpha)`: only channel 0 gets the value on
/// multi-channel types (`ones_1x1_8uc3`, `ones_2x2_32fc2`, `eye_2x2_8uc2`, `eye_3x3_32fc3`);
/// purecv fills every channel.
#[test]
#[ignore = "diverges from OpenCV 4.10.0, see #218"]
fn ones_and_eye_multi_channel_set_only_channel_0() {
    assert_eq!(Matrix::<u8>::ones(1, 1, 3).data, ONES_1X1_8UC3.to_vec());
    assert_eq!(Matrix::<f32>::ones(2, 2, 2).data, ONES_2X2_32FC2.to_vec());
    assert_eq!(Matrix::<u8>::eye(2, 2, 2).data, EYE_2X2_8UC2.to_vec());
    assert_eq!(Matrix::<f32>::eye(3, 3, 3).data, EYE_3X3_32FC3.to_vec());
}

/// OpenCV's `flipND` takes `axis` in `[-ndims, ndims)` and counts a negative axis from the
/// end (`flip_nd_axis_minus1` flips the columns, `flip_nd_axis_minus2` the rows; 2 and -3 are
/// errors, `flip_nd_axis_2`, `flip_nd_axis_minus3`); purecv maps every negative axis to
/// "both" and accepts axis 2.
#[test]
#[ignore = "diverges from OpenCV 4.10.0, see #218"]
fn flip_nd_negative_axis_counts_from_the_end() {
    let m = Matrix::from_vec(2, 3, 1, vec![1i32, 2, 3, 4, 5, 6]);
    let flip = |axis| purecv::core::structural::flip_nd(&m, axis);
    assert_eq!(flip(-1).unwrap().data, vec![3, 2, 1, 6, 5, 4], "axis -1");
    assert_eq!(flip(-2).unwrap().data, vec![4, 5, 6, 1, 2, 3], "axis -2");
    assert!(flip(-3).is_err(), "axis -3");
    assert!(flip(2).is_err(), "axis 2");
}

/// `flipND` with axis 0 and 1 matches (`flip_nd_axis_0`, `flip_nd_axis_1`).
#[test]
fn flip_nd_non_negative_axis_matches_opencv() {
    let m = Matrix::from_vec(2, 3, 1, vec![1i32, 2, 3, 4, 5, 6]);
    let flip = |axis| purecv::core::structural::flip_nd(&m, axis);
    assert_eq!(flip(0).unwrap().data, vec![4, 5, 6, 1, 2, 3], "axis 0");
    assert_eq!(flip(1).unwrap().data, vec![3, 2, 1, 6, 5, 4], "axis 1");
}

// ---------------------------------------------------------------------------------------
// reduce (core_stats.py)
// ---------------------------------------------------------------------------------------

#[test]
fn reduce_max_min_and_f32_match_opencv() {
    let r8 = Matrix::from_vec(2, 3, 1, vec![1u8, 2, 200, 2, 3, 100]);
    assert_eq!(
        reduce(&r8, 0, ReduceTypes::Max).unwrap().data,
        vec![2u8, 3, 200]
    );
    assert_eq!(reduce(&r8, 1, ReduceTypes::Min).unwrap().data, vec![1u8, 2]);
    let rf = Matrix::from_vec(2, 3, 1, vec![1.0f32, 2.0, 200.0, 2.0, 3.0, 100.0]);
    assert_eq!(
        reduce(&rf, 0, ReduceTypes::Sum).unwrap().data,
        vec![3.0f32, 5.0, 300.0]
    );
    let avg = reduce(&rf, 1, ReduceTypes::Avg).unwrap().data;
    assert!((avg[0] - 67.66667).abs() < 1e-4 && avg[1] == 35.0);
}

/// OpenCV: REDUCE_AVG on u8 accumulates in 32S and converts back with `saturate_cast`
/// (1.5 -> 2, 2.5 -> 2, 67.7 -> 68); purecv truncates.
#[test]
#[ignore = "diverges from OpenCV 4.10.0, see #216"]
fn reduce_avg_u8_rounds_half_to_even() {
    let r8 = Matrix::from_vec(2, 3, 1, vec![1u8, 2, 200, 2, 3, 100]);
    assert_eq!(
        reduce(&r8, 0, ReduceTypes::Avg).unwrap().data,
        vec![2u8, 2, 150]
    );
    assert_eq!(
        reduce(&r8, 1, ReduceTypes::Avg).unwrap().data,
        vec![68u8, 35]
    );
}

/// OpenCV refuses REDUCE_SUM from 8U into 8U (`u8_sum_dim0_default_dtype`: "Unsupported
/// combination of input and output array formats"; a `dtype` such as `CV_32S` is required);
/// purecv returns the sum in `u8` and 0 where it overflows (300 -> 0).
#[test]
#[ignore = "diverges from OpenCV 4.10.0, see #216"]
fn reduce_sum_u8_into_u8_is_rejected() {
    let r8 = Matrix::from_vec(2, 3, 1, vec![1u8, 2, 200, 2, 3, 100]);
    assert!(reduce(&r8, 0, ReduceTypes::Sum).is_err());
}

// ---------------------------------------------------------------------------------------
// border_interpolate / copy_make_border / structural (core_misc.py)
// ---------------------------------------------------------------------------------------

#[test]
fn border_interpolate_matches_opencv() {
    use purecv::core::types::BorderTypes as B;
    let ps: Vec<i32> = (-7..13).collect();
    let run = |b: B| {
        ps.iter()
            .map(|&p| border_interpolate(p, 5, b))
            .collect::<Vec<_>>()
    };
    assert_eq!(
        run(B::Constant),
        vec![-1, -1, -1, -1, -1, -1, -1, 0, 1, 2, 3, 4, -1, -1, -1, -1, -1, -1, -1, -1]
    );
    assert_eq!(
        run(B::Replicate),
        vec![0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4]
    );
    assert_eq!(
        run(B::Reflect),
        vec![3, 4, 4, 3, 2, 1, 0, 0, 1, 2, 3, 4, 4, 3, 2, 1, 0, 0, 1, 2]
    );
    assert_eq!(
        run(B::Wrap),
        vec![3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2]
    );
    assert_eq!(
        run(B::Reflect101),
        vec![1, 2, 3, 4, 3, 2, 1, 0, 1, 2, 3, 4, 3, 2, 1, 0, 1, 2, 3, 4]
    );
    for p in [-3, 0, 4] {
        assert_eq!(border_interpolate(p, 1, B::Reflect), 0);
        assert_eq!(border_interpolate(p, 1, B::Reflect101), 0);
    }
}

/// `copy_make_border` takes a raw `i32` whose value 1 means BORDER_REFLECT; OpenCV's
/// `BorderTypes` has REPLICATE = 1, REFLECT = 2 and REFLECT_101 = 4 (the default).
#[test]
#[ignore = "diverges from OpenCV 4.10.0, see #222"]
fn copy_make_border_uses_opencv_border_codes() {
    let src = Matrix::from_vec(2, 3, 1, vec![1u8, 2, 3, 4, 5, 6]);
    let reflect_101 = copy_make_border(&src, 2, 1, 2, 1, 4, Scalar::all(9)).unwrap();
    assert_eq!(
        reflect_101.data,
        vec![
            3u8, 2, 1, 2, 3, 2, 6, 5, 4, 5, 6, 5, 3, 2, 1, 2, 3, 2, 6, 5, 4, 5, 6, 5, 3, 2, 1, 2,
            3, 2
        ]
    );
    let replicate = copy_make_border(&src, 2, 1, 2, 1, 1, Scalar::all(9)).unwrap();
    assert_eq!(
        replicate.data,
        vec![
            1u8, 1, 1, 2, 3, 3, 1, 1, 1, 2, 3, 3, 1, 1, 1, 2, 3, 3, 4, 4, 4, 5, 6, 6, 4, 4, 4, 5,
            6, 6
        ]
    );
}

/// With purecv's own codes, 0 is BORDER_CONSTANT and 1 is BORDER_REFLECT; those two
/// results equal OpenCV's.
#[test]
fn copy_make_border_constant_and_reflect_values_match_opencv() {
    let src = Matrix::from_vec(2, 3, 1, vec![1u8, 2, 3, 4, 5, 6]);
    let constant = copy_make_border(&src, 2, 1, 2, 1, 0, Scalar::all(9)).unwrap();
    assert_eq!(
        constant.data,
        vec![
            9u8, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 1, 2, 3, 9, 9, 9, 4, 5, 6, 9, 9, 9, 9, 9,
            9, 9
        ]
    );
    let reflect = copy_make_border(&src, 2, 1, 2, 1, 1, Scalar::all(9)).unwrap();
    assert_eq!(
        reflect.data,
        vec![
            5u8, 4, 4, 5, 6, 6, 2, 1, 1, 2, 3, 3, 2, 1, 1, 2, 3, 3, 5, 4, 4, 5, 6, 6, 5, 4, 4, 5,
            6, 6
        ]
    );
}

#[test]
fn flip_rotate_transpose_repeat_match_opencv() {
    let m = Matrix::from_vec(2, 3, 1, vec![1i32, 2, 3, 4, 5, 6]);
    assert_eq!(
        structural::flip(&m, 0).unwrap().data,
        vec![4, 5, 6, 1, 2, 3]
    );
    assert_eq!(
        structural::flip(&m, 1).unwrap().data,
        vec![3, 2, 1, 6, 5, 4]
    );
    assert_eq!(
        structural::flip(&m, -1).unwrap().data,
        vec![6, 5, 4, 3, 2, 1]
    );
    assert_eq!(
        structural::flip(&m, 7).unwrap().data,
        vec![3, 2, 1, 6, 5, 4]
    );
    let r90 = structural::rotate(&m, 0).unwrap();
    assert_eq!((r90.rows, r90.cols), (3, 2));
    assert_eq!(r90.data, vec![4, 1, 5, 2, 6, 3]);
    assert_eq!(
        structural::rotate(&m, 1).unwrap().data,
        vec![6, 5, 4, 3, 2, 1]
    );
    assert_eq!(
        structural::rotate(&m, 2).unwrap().data,
        vec![3, 6, 2, 5, 1, 4]
    );
    assert_eq!(
        structural::transpose(&m).unwrap().data,
        vec![1, 4, 2, 5, 3, 6]
    );
    let rep = structural::repeat(&m, 2, 3).unwrap();
    assert_eq!((rep.rows, rep.cols), (4, 9));
    assert_eq!(&rep.data[..9], &[1, 2, 3, 1, 2, 3, 1, 2, 3]);
    assert!(structural::repeat(&m, 0, 1).is_err());
}

// ---------------------------------------------------------------------------------------
// check_range / compare / in_range / lut / sort (core_misc.py)
// ---------------------------------------------------------------------------------------

/// OpenCV's range is `[minVal, maxVal)`: 255 is outside `[0, 255)`. purecv is inclusive.
#[test]
#[ignore = "diverges from OpenCV 4.10.0, see #213"]
fn check_range_upper_bound_is_exclusive() {
    let a = row(&[0u8, 100, 255]);
    assert!(!check_range(&a, 0.0, 255.0));
    assert!(check_range(&a, 0.0, 256.0));
}

#[test]
fn check_range_nan_and_inside_values_match_opencv() {
    assert!(!check_range(&row(&[0.0f32, f32::NAN]), -f64::MAX, f64::MAX));
    assert!(check_range(&row(&[0.0f32, 1.0, 255.0]), 0.0, 255.5));
}

#[test]
fn compare_in_range_lut_match_opencv() {
    let a = row(&[1u8, 5, 3]);
    let b = row(&[1u8, 2, 7]);
    assert_eq!(
        compare(&a, &b, CmpTypes::Eq).unwrap().data,
        vec![255u8, 0, 0]
    );
    assert_eq!(
        compare(&a, &b, CmpTypes::Gt).unwrap().data,
        vec![0u8, 255, 0]
    );
    assert_eq!(
        compare(&a, &b, CmpTypes::Le).unwrap().data,
        vec![255u8, 0, 255]
    );
    assert_eq!(
        compare_scalar(&a, Scalar::all(2u8), CmpTypes::Gt)
            .unwrap()
            .data,
        vec![0u8, 255, 255]
    );
    let rgb = Matrix::from_vec(1, 2, 3, vec![10u8, 20, 30, 40, 50, 60]);
    let mut mask = Matrix::<u8>::new(0, 0, 0);
    in_range_scalar(&rgb, &[10u8, 20, 30], &[40u8, 50, 59], &mut mask).unwrap();
    assert_eq!((mask.channels, mask.data.clone()), (1, vec![255u8, 0]));
    let lo = Matrix::from_vec(1, 2, 3, vec![0u8; 6]);
    let hi = Matrix::from_vec(1, 2, 3, vec![45u8; 6]);
    in_range(&rgb, &lo, &hi, &mut mask).unwrap();
    assert_eq!(mask.data, vec![255u8, 0]);
    let inverted: Vec<u8> = (0..=255u8).rev().collect();
    assert_eq!(
        lut(&rgb, &row(&inverted)).unwrap().data,
        vec![245u8, 235, 225, 215, 205, 195]
    );
    let mut lut3 = Vec::with_capacity(768);
    for i in 0..256u32 {
        lut3.extend_from_slice(&[(255 - i) as u8, i as u8, 7]);
    }
    assert_eq!(
        lut(&rgb, &Matrix::from_vec(1, 256, 3, lut3)).unwrap().data,
        vec![245u8, 20, 7, 215, 50, 7]
    );
    let half: Vec<f32> = (0..256).map(|i| i as f32 * 0.5).collect();
    assert_eq!(lut(&a, &row(&half)).unwrap().data, vec![0.5f32, 2.5, 1.5]);
    assert!(lut(&a, &row(&inverted[..255])).is_err());
}

#[test]
fn sort_and_sort_idx_match_opencv() {
    let src = Matrix::from_vec(2, 4, 1, vec![3i32, 1, 3, 2, 9, 9, 0, 9]);
    let mut dst = Matrix::<i32>::new(0, 0, 0);
    sort(&src, &mut dst, SORT_EVERY_ROW | SORT_ASCENDING).unwrap();
    assert_eq!(dst.data, vec![1, 2, 3, 3, 0, 9, 9, 9]);
    sort(&src, &mut dst, SORT_EVERY_ROW | SORT_DESCENDING).unwrap();
    assert_eq!(dst.data, vec![3, 3, 2, 1, 9, 9, 9, 0]);
    sort(&src, &mut dst, SORT_EVERY_COLUMN | SORT_ASCENDING).unwrap();
    assert_eq!(dst.data, vec![3, 1, 0, 2, 9, 9, 3, 9]);
    let mut idx = Matrix::<i32>::new(0, 0, 0);
    sort_idx(&src, &mut idx, SORT_EVERY_ROW | SORT_ASCENDING).unwrap();
    assert_eq!(idx.data, vec![1, 3, 0, 2, 2, 0, 1, 3]);
    sort_idx(&src, &mut idx, SORT_EVERY_ROW | SORT_DESCENDING).unwrap();
    assert_eq!(idx.data, vec![2, 0, 3, 1, 3, 1, 0, 2]);
    sort_idx(&src, &mut idx, SORT_EVERY_COLUMN | SORT_DESCENDING).unwrap();
    assert_eq!(idx.data, vec![1, 1, 0, 1, 0, 0, 1, 0]);
}

// ---------------------------------------------------------------------------------------
// transform / perspective_transform / linear algebra (core_misc.py)
// ---------------------------------------------------------------------------------------

/// OpenCV writes integer outputs through `saturate_cast`: (10+20+33)*0.3 = 18.9 -> 19 and
/// 229.5 -> 230; purecv truncates (18, 229).
#[test]
#[ignore = "diverges from OpenCV 4.10.0, see #220"]
fn transform_u8_rounds_half_to_even() {
    let src = Matrix::from_vec(1, 3, 3, vec![10u8, 20, 33, 255, 255, 255, 1, 1, 1]);
    let m = Matrix::from_vec(1, 3, 1, vec![0.3f64, 0.3, 0.3]);
    let mut dst = Matrix::<u8>::new(0, 0, 0);
    transform(&src, &mut dst, &m).unwrap();
    assert_eq!(dst.data, vec![19u8, 230, 1]);
    // affine 2x4: OpenCV [[10, 32], [255, 254], [2, 0]]; the 1.5 of the third pixel rounds to
    // 2 in OpenCV and truncates to 1 in purecv.
    let affine = Matrix::from_vec(2, 4, 1, vec![1.0f64, 0.0, 0.0, 0.5, 0.0, 0.0, 1.0, -0.5]);
    transform(&src, &mut dst, &affine).unwrap();
    assert_eq!(dst.data, vec![10u8, 32, 255, 254, 2, 0]);
}

#[test]
fn transform_and_perspective_transform_match_opencv() {
    let src = Matrix::from_vec(1, 3, 3, vec![10u8, 20, 33, 255, 255, 255, 1, 1, 1]);
    let m = Matrix::from_vec(2, 4, 1, vec![1.0f64, 0.0, 0.0, 0.5, 0.0, 0.0, 1.0, -0.5]);
    let mut dst = Matrix::<u8>::new(0, 0, 0);
    transform(&src, &mut dst, &m).unwrap();
    // OpenCV: [[10, 32], [255, 254], [2, 0]]. Five of the six values agree (10.5 -> 10 and
    // 32.5 -> 32 by ties-to-even in OpenCV, by truncation here; 255.5 saturates; 0.5 -> 0).
    // The sixth, 1.5 -> 2 (OpenCV) vs 1 (purecv), is asserted in the ignored test above.
    assert_eq!(dst.channels, 2);
    assert_eq!(&dst.data[..4], &[10u8, 32, 255, 254]);
    assert_eq!(dst.data[5], 0);
    let srcf = Matrix::from_vec(
        1,
        3,
        3,
        vec![10.0f32, 20.0, 33.0, 255.0, 255.0, 255.0, 1.0, 1.0, 1.0],
    );
    let m3 = Matrix::from_vec(1, 3, 1, vec![0.3f64, 0.3, 0.3]);
    let mut dstf = Matrix::<f32>::new(0, 0, 0);
    transform(&srcf, &mut dstf, &m3).unwrap();
    assert_close(
        &dstf.data.iter().map(|&v| v as f64).collect::<Vec<_>>(),
        &[18.900001525878906, 229.5, 0.9000000357627869],
        1e-5,
        "transform f32",
    );
    assert!(transform(&src, &mut dst, &Matrix::from_vec(1, 2, 1, vec![1.0, 1.0])).is_err());

    let pts = Matrix::from_vec(1, 3, 2, vec![1.0f64, 2.0, 3.0, 4.0, 0.0, 0.0]);
    let h = Matrix::from_vec(3, 3, 1, vec![2.0, 0.0, 1.0, 0.0, 3.0, -1.0, 0.5, 0.0, 1.0]);
    let mut out = Matrix::<f64>::new(0, 0, 0);
    perspective_transform(&pts, &mut out, &h).unwrap();
    assert_close(
        &out.data,
        &[2.0, 3.333333333333333, 2.8000000000000003, 4.4, 1.0, -1.0],
        1e-12,
        "perspective_transform",
    );
    let hz = Matrix::from_vec(3, 3, 1, vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]);
    perspective_transform(&pts, &mut out, &hz).unwrap();
    assert_eq!(out.data, vec![0.0; 6]);
}

#[test]
fn gemm_determinant_solve_trace_mahalanobis_match_opencv() {
    let a = Matrix::from_vec(2, 3, 1, vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let b = Matrix::from_vec(3, 2, 1, vec![7.0f32, 8.0, 9.0, 10.0, 11.0, 12.0]);
    let c = Matrix::from_vec(2, 2, 1, vec![1.0f32; 4]);
    assert_eq!(
        gemm(&a, &b, 2.0, &c, 0.5, 0).unwrap().data,
        vec![116.5f32, 128.5, 278.5, 308.5]
    );
    let empty = Matrix::<f32>::new(0, 0, 1);
    assert_eq!(
        gemm(&a, &a, 1.0, &empty, 0.0, GEMM_1_T).unwrap().data,
        vec![17.0f32, 22.0, 27.0, 22.0, 29.0, 36.0, 27.0, 36.0, 45.0]
    );
    assert!(gemm(&a, &b, 1.0, &empty, 0.0, GEMM_2_T).is_err());

    let s = Matrix::from_vec(2, 2, 1, vec![4.0f64, 7.0, 2.0, 6.0]);
    let s4 = Matrix::from_vec(
        4,
        4,
        1,
        vec![
            2.0, 0.0, 1.0, 3.0, 1.0, 1.0, 0.0, 2.0, 0.0, 2.0, 1.0, 1.0, 3.0, 1.0, 2.0, 0.0,
        ],
    );
    let sing = Matrix::from_vec(2, 2, 1, vec![1.0f64, 2.0, 2.0, 4.0]);
    assert!((determinant(&s) - 10.0).abs() < 1e-12);
    assert!((determinant(&s4) + 20.0).abs() < 1e-12);
    assert_eq!(determinant(&sing), 0.0);

    let mut inv = Matrix::<f64>::new(2, 2, 1);
    invert(&s, &mut inv, DecompTypes::DECOMP_LU).unwrap();
    assert_close(&inv.data, &[0.6, -0.7, -0.2, 0.4], 1e-12, "invert");
    let mut inv_s = Matrix::<f64>::new(2, 2, 1);
    assert_eq!(
        invert(&sing, &mut inv_s, DecompTypes::DECOMP_LU).unwrap(),
        0.0
    );
    assert_eq!(inv_s.data, vec![0.0; 4]);

    let rhs = Matrix::from_vec(2, 1, 1, vec![1.0f64, 2.0]);
    let mut x = Matrix::<f64>::new(2, 1, 1);
    assert!(solve(&s, &rhs, &mut x, DecompTypes::DECOMP_LU).unwrap());
    assert_close(&x.data, &[-0.8, 0.6], 1e-12, "solve");
    assert!(!solve(&sing, &rhs, &mut x, DecompTypes::DECOMP_LU).unwrap());

    assert_eq!(trace(&s4).v[0], 4.0);
    let two_ch = Matrix::from_vec(2, 2, 2, vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    assert_eq!(trace(&two_ch).v, [8.0, 10.0, 0.0, 0.0]);

    let v1 = Matrix::from_vec(2, 1, 1, vec![1.0f64, 2.0]);
    let v2 = Matrix::from_vec(2, 1, 1, vec![3.0f64, 5.0]);
    let icov = Matrix::from_vec(2, 2, 1, vec![2.0f64, 0.5, 0.5, 1.0]);
    assert!((mahalanobis(&v1, &v2, &icov).unwrap() - 4.795831523312719).abs() < 1e-12);
}

/// OpenCV `invert(DECOMP_LU)` returns 1.0 on success (a non-zero flag), not the determinant.
#[test]
#[ignore = "diverges from OpenCV 4.10.0, see #219"]
fn invert_lu_returns_one_on_success() {
    let s = Matrix::from_vec(2, 2, 1, vec![4.0f64, 7.0, 2.0, 6.0]);
    let mut inv = Matrix::<f64>::new(2, 2, 1);
    assert_eq!(invert(&s, &mut inv, DecompTypes::DECOMP_LU).unwrap(), 1.0);
}

#[test]
fn set_identity_and_psnr_match_opencv() {
    let mut m = Matrix::<f32>::new(2, 3, 2);
    set_identity(&mut m, Scalar::new(1.0, 2.0, 0.0, 0.0));
    assert_eq!(
        m.data,
        vec![1.0f32, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0]
    );
    let rgb = Matrix::from_vec(1, 2, 3, vec![10u8, 20, 30, 40, 50, 60]);
    let rgb3 = Matrix::from_vec(1, 2, 3, vec![13u8, 23, 33, 43, 53, 63]);
    assert!((psnr(&rgb, &rgb3).unwrap() - 38.58837851428585).abs() < 1e-6);
}

/// OpenCV `PSNR` of identical images is `20*log10(R/DBL_EPSILON)` = 361.2, not infinity.
#[test]
#[ignore = "diverges from OpenCV 4.10.0, see #218"]
fn psnr_identical_inputs_is_finite() {
    let rgb = Matrix::from_vec(1, 2, 3, vec![10u8, 20, 30, 40, 50, 60]);
    assert!((psnr(&rgb, &rgb).unwrap() - 361.20199909921956).abs() < 1e-6);
}

// ---------------------------------------------------------------------------------------
// solve_cubic / solve_poly (core_misc.py)
// ---------------------------------------------------------------------------------------

fn sorted(mut v: Vec<f64>) -> Vec<f64> {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v
}

#[test]
fn solve_cubic_and_solve_poly_match_opencv() {
    assert_close(
        &sorted(solve_cubic(1.0, -6.0, 11.0, -6.0).unwrap()),
        &[1.0, 2.0, 3.0],
        1e-9,
        "cubic 3 roots",
    );
    assert_close(
        &solve_cubic(1.0, 0.0, 0.0, -8.0).unwrap(),
        &[2.0],
        1e-9,
        "cubic 1 root",
    );
    assert_close(
        &solve_cubic(1.0, -3.0, 3.0, -1.0).unwrap(),
        &[1.0],
        1e-9,
        "cubic triple",
    );
    assert_close(
        &sorted(solve_cubic(0.0, 1.0, -3.0, 2.0).unwrap()),
        &[1.0, 2.0],
        1e-9,
        "cubic -> quadratic",
    );
    assert_close(
        &solve_cubic(0.0, 0.0, 2.0, -4.0).unwrap(),
        &[2.0],
        1e-9,
        "cubic -> linear",
    );
    assert!(solve_cubic(0.0, 1.0, 0.0, 1.0).unwrap().is_empty());

    let mut roots = Matrix::<f64>::new(0, 0, 0);
    solve_poly(&row(&[2.0f64, -3.0, 1.0]), &mut roots, 300).unwrap();
    let re: Vec<f64> = roots.data.iter().step_by(2).copied().collect();
    assert_close(&sorted(re), &[1.0, 2.0], 1e-9, "poly real");
    solve_poly(&row(&[1.0f64, 0.0, 1.0]), &mut roots, 300).unwrap();
    let im: Vec<f64> = roots.data.iter().skip(1).step_by(2).copied().collect();
    assert_close(&sorted(im), &[-1.0, 1.0], 1e-9, "poly complex");
}

// ---------------------------------------------------------------------------------------
// sqrt / exp / log / pow / magnitude / phase (core_misc.py)
// ---------------------------------------------------------------------------------------

#[test]
// The literals are OpenCV 4.10.0 outputs copied from the script, not approximations.
#[allow(clippy::approx_constant, clippy::excessive_precision)]
fn sqrt_magnitude_integer_pow_match_opencv() {
    let xs = row(&[0.0f32, 0.5, 1.0, 2.0, 10.0, 100.0, 1e-3, 3.7]);
    assert_eq!(
        sqrt(&xs).unwrap().data,
        vec![
            0.0f32,
            0.70710677,
            1.0,
            1.4142135,
            3.1622777,
            10.0,
            0.031622779,
            1.9235384
        ]
    );
    assert_eq!(
        pow(&xs, 2.0).unwrap().data,
        vec![
            0.0f32,
            0.25,
            1.0,
            4.0,
            100.0,
            10000.0,
            1.0000001e-6,
            13.690001
        ]
    );
    assert_eq!(
        pow(&row(&[-1.0f32, -4.0]), 3.0).unwrap().data,
        vec![-1.0f32, -64.0]
    );
    let x = row(&[1.0f32, -1.0, 1.0, 0.0, 3.0, 1.0, -1.0]);
    let y = row(&[1.0f32, 1.0, -1e-3, 0.0, 4.0, 0.0, -1.0]);
    let mut mag = Matrix::<f32>::new(0, 0, 0);
    magnitude(&x, &y, &mut mag).unwrap();
    assert_eq!(
        mag.data,
        vec![1.4142135f32, 1.4142135, 1.0000005, 0.0, 5.0, 1.0, 1.4142135]
    );
}

/// Fed OpenCV's own (fastAtan2) angles, `polar_to_cart` reproduces OpenCV's x/y to 1e-6:
/// OpenCV's `SinCos_32f` table and libm agree to a few ULP.
#[test]
// The literals are OpenCV 4.10.0 outputs copied from the script, not approximations.
#[allow(clippy::approx_constant, clippy::excessive_precision)]
fn polar_to_cart_matches_opencv_on_opencv_angles() {
    let mag = row(&[
        1.4142135f32,
        1.4142135,
        1.0000005,
        0.0,
        5.0,
        1.0,
        1.4142135,
        1.0,
    ]);
    let ang = row(&[
        44.990455627441406f32,
        135.00955200195312,
        359.9427185058594,
        0.0,
        53.13628387451172,
        0.0,
        224.99044799804688,
        89.99999237060547,
    ]);
    let (mut x, mut y) = (Matrix::<f32>::new(0, 0, 0), Matrix::<f32>::new(0, 0, 0));
    polar_to_cart(&mag, &ang, &mut x, &mut y, true).unwrap();
    let f = |m: &Matrix<f32>| m.data.iter().map(|&v| v as f64).collect::<Vec<_>>();
    assert_close(
        &f(&x),
        &[
            1.0001665353775024,
            -1.000166654586792,
            1.0,
            0.0,
            2.9995691776275635,
            1.0,
            -1.000166654586792,
            1.3315805347247078e-7,
        ],
        1e-6,
        "polar_to_cart x",
    );
    assert_close(
        &f(&y),
        &[
            0.9998334050178528,
            0.9998332262039185,
            -0.0009997509187087417,
            0.0,
            4.000324249267578,
            0.0,
            -0.9998332262039185,
            1.0,
        ],
        1e-6,
        "polar_to_cart y",
    );
}

/// `phase` is `intentional` (docs/parity/4.10.0/core.md): purecv keeps the exact `atan2`, OpenCV
/// uses `fastAtan2` (44.990456 for (1, 1), 135.009552 for (-1, 1), 53.136284 for (3, 4)).
/// `exp` is `measured` within 2 ULP of OpenCV's polynomial (1.6487212181, 2.7182819843).
#[test]
// The literals are OpenCV 4.10.0 outputs copied from the script, not approximations.
#[allow(clippy::approx_constant, clippy::excessive_precision)]
fn phase_exact_atan2_and_exp_within_2_ulp_of_opencv() {
    let mut ang = Matrix::<f32>::new(0, 0, 0);
    phase(
        &row(&[1.0f32, -1.0, 3.0]),
        &row(&[1.0f32, 1.0, 4.0]),
        &mut ang,
        true,
    )
    .unwrap();
    assert_eq!(ang.data, vec![45.0f32, 135.0, 53.130104]);
    let e = exp(&row(&[0.5f32, 1.0])).unwrap().data;
    for (got, opencv) in e.iter().zip([1.6487212181091309f32, 2.7182819843292236]) {
        assert!(
            (got - opencv).abs() <= 2.0 * f32::EPSILON * opencv,
            "{got} vs {opencv}"
        );
    }
}

// ---------------------------------------------------------------------------------------
// kmeans (core_kmeans.py)
// ---------------------------------------------------------------------------------------

fn kmeans_data() -> Matrix<f32> {
    Matrix::from_vec(
        12,
        2,
        1,
        vec![
            0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 10.0, 10.0, 11.0, 10.0, 10.0, 11.0, 11.0, 12.0,
            0.0, 10.0, 1.0, 10.0, 0.0, 11.0, 2.0, 11.0,
        ],
    )
}

fn sorted_centers(c: &Matrix<f32>) -> Vec<(f32, f32)> {
    let mut v: Vec<(f32, f32)> = c.data.chunks(2).map(|p| (p[0], p[1])).collect();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v
}

#[test]
fn kmeans_converges_to_opencv_partition() {
    let data = kmeans_data();
    let criteria = TermCriteria::new(TermType::Both, 100, 1e-4);
    // kmeans++ (3 attempts): OpenCV compactness 9.5, centers (0.5,0.5) (0.75,10.5) (10.5,10.75)
    let mut labels = Matrix::<i32>::new(0, 0, 0);
    let mut centers = Some(Matrix::<f32>::new(0, 0, 0));
    let compactness = kmeans(
        &data,
        3,
        &mut labels,
        criteria,
        3,
        KMEANS_PP_CENTERS,
        &mut centers,
    )
    .unwrap();
    assert!(
        (compactness - 9.5).abs() < 1e-9,
        "compactness {compactness}"
    );
    assert_eq!(
        sorted_centers(centers.as_ref().unwrap()),
        vec![(0.5, 0.5), (0.75, 10.5), (10.5, 10.75)]
    );
    assert_eq!(labels.data.len(), 12);
    assert_eq!(&labels.data[0..4], &[labels.data[0]; 4]);
    assert_eq!(&labels.data[4..8], &[labels.data[4]; 4]);
    assert_eq!(&labels.data[8..12], &[labels.data[8]; 4]);
    let mut labels = Matrix::from_vec(12, 1, 1, vec![0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]);
    let mut centers = Some(Matrix::<f32>::new(0, 0, 0));
    let c = kmeans(
        &data,
        3,
        &mut labels,
        criteria,
        1,
        KMEANS_USE_INITIAL_LABELS,
        &mut centers,
    )
    .unwrap();
    assert!((c - 9.5).abs() < 1e-9);
    assert_eq!(labels.data, vec![0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]);
    // K = 1: the single center is the mean, compactness 543.1666...
    let mut l1 = Matrix::<i32>::new(0, 0, 0);
    let mut c1 = Some(Matrix::<f32>::new(0, 0, 0));
    let comp1 = kmeans(&data, 1, &mut l1, criteria, 1, KMEANS_PP_CENTERS, &mut c1).unwrap();
    assert!((comp1 - 543.1666469573975).abs() < 1e-3, "{comp1}");
    assert!(kmeans(&data, 13, &mut l1, criteria, 1, KMEANS_PP_CENTERS, &mut c1).is_err());
}

/// OpenCV draws random centers uniformly inside the bounding box from `theRNG()` (seeded 42)
/// and reaches compactness 9.5 on this fixture with 1, 3 and 5 attempts
/// (`random_centers_attempts1`, `random_centers_attempts3`, `random_centers_attempts5`).
/// purecv picks k samples with an LCG seeded by the attempt index and stays in a local
/// minimum (198.75) for 3 attempts.
#[test]
#[ignore = "diverges from OpenCV 4.10.0, see #221"]
fn kmeans_random_centers_three_attempts_find_the_global_partition() {
    let data = kmeans_data();
    let criteria = TermCriteria::new(TermType::Both, 100, 1e-4);
    let mut labels = Matrix::<i32>::new(0, 0, 0);
    let mut centers = Some(Matrix::<f32>::new(0, 0, 0));
    let c = kmeans(
        &data,
        3,
        &mut labels,
        criteria,
        3,
        KMEANS_RANDOM_CENTERS,
        &mut centers,
    )
    .unwrap();
    assert!((c - 9.5).abs() < 1e-9, "compactness {c}");
}

// ---------------------------------------------------------------------------------------
// RNG (core_rng.py)
// ---------------------------------------------------------------------------------------

/// OpenCV's `theRNG()` is a multiply-with-carry generator (`RNG::next`); purecv seeds a
/// xoshiro256** generator, so the same seed gives a different sequence.
#[test]
#[ignore = "diverges from OpenCV 4.10.0, see #225"]
fn randu_seed_42_reproduces_cv_rng() {
    set_rng_seed(42);
    let mut m = Matrix::<u8>::new(1, 8, 1);
    randu(&mut m, Scalar::all(0.0), Scalar::all(256.0)).unwrap();
    assert_eq!(m.data, vec![164u8, 63, 102, 186, 144, 232, 246, 91]);
    set_rng_seed(42);
    let mut idx: Vec<i32> = (0..10).collect();
    rand_shuffle(&mut idx);
    assert_eq!(idx, vec![5, 3, 7, 1, 4, 2, 8, 9, 6, 0]);
}

/// OpenCV writes integer `randn` samples through `saturate_cast`, so a u8 destination with a
/// wide distribution holds many 255s and 0s. purecv turns every sample above 255 into 0.
#[test]
#[ignore = "diverges from OpenCV 4.10.0, see #225"]
fn randn_u8_saturates_above_255() {
    set_rng_seed(1);
    let mut m = Matrix::<u8>::new(1, 2000, 1);
    randn(&mut m, Scalar::all(128.0), Scalar::all(200.0)).unwrap();
    let zeros = m.data.iter().filter(|&&v| v == 0).count();
    let full = m.data.iter().filter(|&&v| v == 255).count();
    // about 26% of a N(128, 200) sample lies below 0 and about 26% above 255
    assert!(
        full > 300,
        "255 count {full}: samples above 255 were turned into 0 ({zeros} zeros)"
    );
}

/// Range semantics agree: integer `randu` draws from `[low, high)`.
#[test]
fn randu_range_semantics_match_opencv() {
    set_rng_seed(7);
    let mut m = Matrix::<u8>::new(1, 4096, 1);
    randu(&mut m, Scalar::all(0.0), Scalar::all(10.0)).unwrap();
    assert!(m.data.iter().all(|&v| v < 10));
    assert!(m.data.contains(&0) && m.data.contains(&9));
}

// ---------------------------------------------------------------------------------------
// dft / idft / dct / get_optimal_dft_size (core_dft_dct.py), `fft` / `transforms` features
// ---------------------------------------------------------------------------------------

#[cfg(feature = "fft")]
mod fft {
    use super::*;

    fn src() -> Matrix<f32> {
        Matrix::from_vec(
            3,
            4,
            1,
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 13.0,
            ],
        )
    }

    const COMPLEX: [f32; 24] = [
        79.0, 0.0, -6.0, 7.0, -7.0, 0.0, -6.0, -7.0, //
        -24.5, 14.722431, -0.8660254, -0.5, 0.5, -0.8660254, 0.8660254, 0.5, //
        -24.5, -14.722431, 0.8660254, -0.5, 0.5, 0.8660254, -0.8660254, 0.5,
    ];

    fn as_f64(m: &Matrix<f32>) -> Vec<f64> {
        m.data.iter().map(|&v| v as f64).collect()
    }

    fn f64s(v: &[f32]) -> Vec<f64> {
        v.iter().map(|&v| v as f64).collect()
    }

    /// purecv's `dft(src, 0)` on real input equals OpenCV's `dft(src, DFT_COMPLEX_OUTPUT)`.
    #[test]
    fn dft_complex_output_matches_opencv() {
        let d = dft(&src(), 0, 0).unwrap();
        assert_eq!(d.channels, 2);
        assert_close(&as_f64(&d), &f64s(&COMPLEX), 1e-4, "dft");
        let d = dft(&src(), DFT_COMPLEX_OUTPUT, 0).unwrap();
        assert_close(&as_f64(&d), &f64s(&COMPLEX), 1e-4, "dft COMPLEX_OUTPUT");
        let scaled: Vec<f64> = COMPLEX.iter().map(|&v| v as f64 / 12.0).collect();
        assert_close(
            &as_f64(&dft(&src(), DFT_SCALE, 0).unwrap()),
            &scaled,
            1e-5,
            "dft SCALE",
        );
        let rows = dft(&src(), DFT_ROWS, 0).unwrap();
        assert_close(
            &as_f64(&rows),
            &[
                10.0, 0.0, -2.0, 2.0, -2.0, 0.0, -2.0, -2.0, 26.0, 0.0, -2.0, 2.0, -2.0, 0.0, -2.0,
                -2.0, 43.0, 0.0, -2.0, 3.0, -3.0, 0.0, -2.0, -3.0,
            ],
            1e-4,
            "dft ROWS",
        );
        let rows_scaled = dft(&src(), DFT_ROWS | DFT_SCALE, 0).unwrap();
        assert_close(
            &as_f64(&rows_scaled)[..4],
            &[2.5, 0.0, -0.5, 0.5],
            1e-5,
            "dft ROWS|SCALE",
        );
        let nz = dft(&src(), 0, 2).unwrap();
        assert_close(
            &as_f64(&nz)[..8],
            &[36.0, 0.0, -4.0, 4.0, -4.0, 0.0, -4.0, -4.0],
            1e-4,
            "nonzero_rows",
        );
        let spec = dft(&src(), 0, 0).unwrap();
        let back = dft(&spec, DFT_INVERSE | DFT_SCALE | DFT_REAL_OUTPUT, 0).unwrap();
        assert_eq!(back.channels, 1);
        assert_close(&as_f64(&back), &as_f64(&src()), 1e-4, "inverse real output");
        let back_c = dft(&spec, DFT_INVERSE, 0).unwrap();
        assert_close(
            &as_f64(&back_c)[..4],
            &[12.0, 0.0, 24.0, 0.0],
            1e-4,
            "inverse unscaled",
        );
        assert!(dft(&src(), DFT_REAL_OUTPUT | DFT_COMPLEX_OUTPUT, 0).is_err());
    }

    /// OpenCV's `idft(src, flags)` is `dft(src, flags | DFT_INVERSE)`: it does **not** scale
    /// unless DFT_SCALE is passed. purecv's `idft` always adds DFT_SCALE.
    #[test]
    #[ignore = "diverges from OpenCV 4.10.0, see #224"]
    fn idft_does_not_scale_by_default() {
        let spec = dft(&src(), 0, 0).unwrap();
        let back = idft(&spec, 0, 0).unwrap();
        assert_close(
            &as_f64(&back)[..4],
            &[12.0, 0.0, 24.0, 0.0],
            1e-4,
            "idft default",
        );
    }

    /// OpenCV's `dft(src, 0)` on real input returns the single-channel CCS packing
    /// (`flags0_ccs_packed_1ch`); purecv returns the 2-channel complex spectrum.
    #[test]
    #[ignore = "diverges from OpenCV 4.10.0, see #223"]
    fn dft_real_input_returns_ccs_by_default() {
        let d = dft(&src(), 0, 0).unwrap();
        assert_eq!(d.channels, 1);
        assert_close(&as_f64(&d), &CCS, 1e-4, "dft CCS");
    }

    const CCS: [f64; 12] = [
        79.0,
        -6.0,
        7.0,
        -7.0, //
        -24.5,
        -0.8660253882408142,
        -0.5,
        0.5, //
        14.722431182861328,
        0.8660253882408142,
        -0.5,
        -0.8660253882408142,
    ];

    #[test]
    fn get_optimal_dft_size_matches_opencv_in_range() {
        for (n, expected) in [
            (1, 1),
            (2, 2),
            (7, 8),
            (11, 12),
            (13, 15),
            (17, 18),
            (100, 100),
            (101, 108),
            (257, 270),
            (1000, 1000),
            (1001, 1024),
            (65536, 65536),
            (65537, 65610),
            (100000, 100000),
            (1 << 20, 1 << 20),
            ((1 << 30) + 1, 1074954240),
        ] {
            assert_eq!(get_optimal_dft_size(n), expected, "n = {n}");
        }
    }

    /// OpenCV returns 1 for 0 and -1 for negative sizes and for sizes above its table
    /// (>= 2125764000). purecv returns 0, 0 and a computed value.
    #[test]
    #[ignore = "diverges from OpenCV 4.10.0, see #218"]
    fn get_optimal_dft_size_edges_match_opencv() {
        assert_eq!(get_optimal_dft_size(0), 1);
        assert_eq!(get_optimal_dft_size(-5), -1);
        assert_eq!(get_optimal_dft_size(2125764000), -1);
        assert_eq!(get_optimal_dft_size(i32::MAX), -1);
    }

    #[cfg(feature = "transforms")]
    #[test]
    fn dct_matches_opencv() {
        let row8 = Matrix::from_vec(1, 8, 1, vec![1.0f32, 3.0, 2.0, 5.0, 4.0, 7.0, 6.0, 9.0]);
        let d = dct(&row8).unwrap();
        assert_close(
            &d.data,
            &[
                13.08147545195113,
                -6.461725961607681,
                0.46193976625564337,
                -0.8590648816877139,
                0.35355339059327373,
                -0.8230940103626111,
                0.19134171618254486,
                -2.5160726094930874,
            ],
            1e-9,
            "dct 1x8",
        );
        let back = idct(&d).unwrap();
        assert_close(
            &back.data,
            &[1.0, 3.0, 2.0, 5.0, 4.0, 7.0, 6.0, 9.0],
            1e-9,
            "idct roundtrip",
        );
        let sq4 = Matrix::from_vec(
            4,
            4,
            1,
            vec![
                1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                17.0,
            ],
        );
        let d2 = dct(&sq4).unwrap();
        assert_close(
            &d2.data[..4],
            &[34.25, -4.787525653839111, 0.25, -0.45232445001602173],
            1e-5,
            "dct 4x4 row 0",
        );
        assert_close(
            &d2.data[4..6],
            &[-18.17017936706543, 0.4267764985561371],
            1e-5,
            "dct 4x4 row 1",
        );
    }
}

/// `core_mat_initializers.js` (opencv.js 4.10.0).
const ONES_1X1_8UC3: [u8; 3] = [1, 0, 0];
const ONES_2X2_32FC2: [f32; 8] = [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0];
const ONES_2X3_8UC1: [u8; 6] = [1, 1, 1, 1, 1, 1];
const EYE_2X2_8UC2: [u8; 8] = [1, 0, 0, 0, 0, 0, 1, 0];
const EYE_3X3_32FC3: [f32; 27] = [
    1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
];
const EYE_2X3_32FC1: [f32; 6] = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0];

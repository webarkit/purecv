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
fn convert_to_f32_to_u8_rounds_half_even_and_saturates() {
    let src = row(&[0.5f32, 1.5, 2.5, 3.5, 254.5, 254.6, 255.5, 300.0, -1.0]);
    let dst = src.convert_to::<u8>().expect("OpenCV never fails on range");
    assert_eq!(dst.data, vec![0u8, 2, 2, 4, 254, 255, 255, 255, 0]);
}

/// Source of the `f32` / `f64` blocks of `core_convert_to.js` (`SRC`), without the last five
/// entries (+-1e10, NaN, +-inf): OpenCV maps those through `INT_MIN` on x86-64, which purecv
/// does not port (docs/parity/4.10.0/core.md). `convert_to_nan_and_inf_saturate` covers them.
const CONVERT_SRC: [f64; 20] = [
    0.5, 1.5, 2.5, 3.5, -0.5, -1.5, -2.5, 254.5, 254.6, 255.5, 300.0, -1.0, -300.0, 127.5, 128.5,
    -128.5, -129.0, 65535.5, 32767.5, -32768.5,
];

/// `core_convert_to.js`, `convert_to_from_f32` and `convert_to_from_f64` (identical): `u8`,
/// `i8`, `u16`, `i16`, `i32` with `alpha = 1, beta = 0`.
#[test]
fn convert_to_matches_opencv() {
    let f32_src = row(&CONVERT_SRC.map(|v| v as f32));
    let f64_src = row(&CONVERT_SRC);
    let u8s = vec![
        0u8, 2, 2, 4, 0, 0, 0, 254, 255, 255, 255, 0, 0, 128, 128, 0, 0, 255, 255, 0,
    ];
    let i8s = vec![
        0i8, 2, 2, 4, 0, -2, -2, 127, 127, 127, 127, -1, -128, 127, 127, -128, -128, 127, 127, -128,
    ];
    let u16s = vec![
        0u16, 2, 2, 4, 0, 0, 0, 254, 255, 256, 300, 0, 0, 128, 128, 0, 0, 65535, 32768, 0,
    ];
    let i16s = vec![
        0i16, 2, 2, 4, 0, -2, -2, 254, 255, 256, 300, -1, -300, 128, 128, -128, -129, 32767, 32767,
        -32768,
    ];
    let i32s = vec![
        0i32, 2, 2, 4, 0, -2, -2, 254, 255, 256, 300, -1, -300, 128, 128, -128, -129, 65536, 32768,
        -32768,
    ];
    assert_eq!(f32_src.convert_to::<u8>().unwrap().data, u8s);
    assert_eq!(f64_src.convert_to::<u8>().unwrap().data, u8s);
    assert_eq!(f32_src.convert_to::<i8>().unwrap().data, i8s);
    assert_eq!(f64_src.convert_to::<i8>().unwrap().data, i8s);
    assert_eq!(f32_src.convert_to::<u16>().unwrap().data, u16s);
    assert_eq!(f64_src.convert_to::<u16>().unwrap().data, u16s);
    assert_eq!(f32_src.convert_to::<i16>().unwrap().data, i16s);
    assert_eq!(f64_src.convert_to::<i16>().unwrap().data, i16s);
    assert_eq!(f32_src.convert_to::<i32>().unwrap().data, i32s);
    assert_eq!(f64_src.convert_to::<i32>().unwrap().data, i32s);
}

/// `core_convert_to.js`: `f32` / `f64` sources with `alpha = 2, beta = 0.5`; `u8` source with
/// `alpha = 0.5, beta = 0.25` (ties after scaling); `i32` source with `alpha = 0.5, beta = 0`.
#[test]
fn convert_to_scaled_matches_opencv() {
    let f32_src = row(&CONVERT_SRC.map(|v| v as f32));
    let f64_src = row(&CONVERT_SRC);
    let u8s = vec![
        2u8, 4, 6, 8, 0, 0, 0, 255, 255, 255, 255, 0, 0, 255, 255, 0, 0, 255, 255, 0,
    ];
    let i8s = vec![
        2i8, 4, 6, 8, 0, -2, -4, 127, 127, 127, 127, -2, -128, 127, 127, -128, -128, 127, 127, -128,
    ];
    let u16s = vec![
        2u16, 4, 6, 8, 0, 0, 0, 510, 510, 512, 600, 0, 0, 256, 258, 0, 0, 65535, 65535, 0,
    ];
    let i16s = vec![
        2i16, 4, 6, 8, 0, -2, -4, 510, 510, 512, 600, -2, -600, 256, 258, -256, -258, 32767, 32767,
        -32768,
    ];
    let i32s = vec![
        2i32, 4, 6, 8, 0, -2, -4, 510, 510, 512, 600, -2, -600, 256, 258, -256, -258, 131072,
        65536, -65536,
    ];
    assert_eq!(f32_src.convert_to_scaled::<u8>(2.0, 0.5).unwrap().data, u8s);
    assert_eq!(f64_src.convert_to_scaled::<u8>(2.0, 0.5).unwrap().data, u8s);
    assert_eq!(f32_src.convert_to_scaled::<i8>(2.0, 0.5).unwrap().data, i8s);
    assert_eq!(f64_src.convert_to_scaled::<i8>(2.0, 0.5).unwrap().data, i8s);
    assert_eq!(
        f32_src.convert_to_scaled::<u16>(2.0, 0.5).unwrap().data,
        u16s
    );
    assert_eq!(
        f64_src.convert_to_scaled::<u16>(2.0, 0.5).unwrap().data,
        u16s
    );
    assert_eq!(
        f32_src.convert_to_scaled::<i16>(2.0, 0.5).unwrap().data,
        i16s
    );
    assert_eq!(
        f64_src.convert_to_scaled::<i16>(2.0, 0.5).unwrap().data,
        i16s
    );
    assert_eq!(
        f32_src.convert_to_scaled::<i32>(2.0, 0.5).unwrap().data,
        i32s
    );
    assert_eq!(
        f64_src.convert_to_scaled::<i32>(2.0, 0.5).unwrap().data,
        i32s
    );

    let u8_src = row(&[0u8, 1, 2, 3, 5, 7, 100, 127, 128, 129, 200, 254, 255]);
    let tail = [0, 1, 1, 2, 3, 4, 50, 64, 64, 65, 100, 127, 128];
    assert_eq!(
        u8_src.convert_to_scaled::<u8>(0.5, 0.25).unwrap().data,
        tail.map(|v| v as u8)
    );
    assert_eq!(
        u8_src.convert_to_scaled::<u16>(0.5, 0.25).unwrap().data,
        tail.map(|v| v as u16)
    );
    assert_eq!(
        u8_src.convert_to_scaled::<i16>(0.5, 0.25).unwrap().data,
        tail.map(|v| v as i16)
    );
    assert_eq!(
        u8_src.convert_to_scaled::<i32>(0.5, 0.25).unwrap().data,
        tail
    );
    // The i8 column differs only at 255 * 0.5 + 0.25 = 127.75, which clamps to 127.
    let mut i8_tail = tail.map(|v| v as i8);
    i8_tail[12] = 127;
    assert_eq!(
        u8_src.convert_to_scaled::<i8>(0.5, 0.25).unwrap().data,
        i8_tail
    );

    let i32_src = row(&[
        0i32,
        1,
        3,
        5,
        255,
        256,
        257,
        32767,
        32768,
        65535,
        65536,
        -1,
        -3,
        -129,
        -32769,
        2147483647,
        -2147483648,
    ]);
    assert_eq!(
        i32_src.convert_to_scaled::<u8>(0.5, 0.0).unwrap().data,
        vec![0u8, 0, 2, 2, 128, 128, 128, 255, 255, 255, 255, 0, 0, 0, 0, 255, 0]
    );
    assert_eq!(
        i32_src.convert_to_scaled::<i8>(0.5, 0.0).unwrap().data,
        vec![0i8, 0, 2, 2, 127, 127, 127, 127, 127, 127, 127, 0, -2, -64, -128, 127, -128]
    );
    assert_eq!(
        i32_src.convert_to_scaled::<u16>(0.5, 0.0).unwrap().data,
        vec![0u16, 0, 2, 2, 128, 128, 128, 16384, 16384, 32768, 32768, 0, 0, 0, 0, 65535, 0]
    );
    assert_eq!(
        i32_src.convert_to_scaled::<i16>(0.5, 0.0).unwrap().data,
        vec![
            0i16, 0, 2, 2, 128, 128, 128, 16384, 16384, 32767, 32767, 0, -2, -64, -16384, 32767,
            -32768
        ]
    );
    // i32 -> i32 is computed in double: 2147483647 * 0.5 = 1073741823.5 rounds to the even 1073741824.
    assert_eq!(
        i32_src.convert_to_scaled::<i32>(0.5, 0.0).unwrap().data,
        vec![
            0i32,
            0,
            2,
            2,
            128,
            128,
            128,
            16384,
            16384,
            32768,
            32768,
            0,
            -2,
            -64,
            -16384,
            1073741824,
            -1073741824
        ]
    );
}

/// `alpha = 1, beta = 0` is the plain conversion in OpenCV (`Mat::convertTo`).
#[test]
fn convert_to_scaled_identity_is_convert_to() {
    let src = row(&[0.5f32, 2.5, 300.0, -1.0]);
    assert_eq!(
        src.convert_to_scaled::<u8>(1.0, 0.0).unwrap().data,
        src.convert_to::<u8>().unwrap().data
    );
}

/// NaN becomes 0 and the infinities clamp, for every integer target (no error, no panic).
/// OpenCV's own values for these inputs are platform-dependent (`INT_MIN` on x86-64), so only
/// the `u8` NaN agrees with it.
#[test]
fn convert_to_nan_and_inf_saturate() {
    let src = row(&[f32::NAN, f32::INFINITY, f32::NEG_INFINITY]);
    assert_eq!(src.convert_to::<u8>().unwrap().data, vec![0u8, 255, 0]);
    assert_eq!(src.convert_to::<i8>().unwrap().data, vec![0i8, 127, -128]);
    assert_eq!(src.convert_to::<u16>().unwrap().data, vec![0u16, 65535, 0]);
    assert_eq!(
        src.convert_to::<i16>().unwrap().data,
        vec![0i16, 32767, -32768]
    );
    assert_eq!(
        src.convert_to::<i32>().unwrap().data,
        vec![0i32, i32::MAX, i32::MIN]
    );
    assert_eq!(
        src.convert_to_scaled::<u8>(2.0, 1.0).unwrap().data,
        vec![0u8, 255, 0]
    );
}

/// Same-type and integer-to-wider conversions never go through `f64`: 2^53 + 1 is not
/// representable there.
#[test]
fn convert_to_i64_identity_is_exact() {
    let big = [
        i64::MAX,
        i64::MIN,
        (1i64 << 53) + 1,
        -(1i64 << 53) - 1,
        0,
        7,
    ];
    assert_eq!(row(&big).convert_to::<i64>().unwrap().data, big.to_vec());

    let ubig = [u64::MAX, (1u64 << 53) + 1, 0, 7];
    assert_eq!(row(&ubig).convert_to::<u64>().unwrap().data, ubig.to_vec());

    assert_eq!(
        row(&[i32::MAX, i32::MIN, 1])
            .convert_to::<i64>()
            .unwrap()
            .data,
        vec![i32::MAX as i64, i32::MIN as i64, 1]
    );
    assert_eq!(
        row(&ubig).convert_to::<i64>().unwrap().data,
        vec![i64::MAX, (1i64 << 53) + 1, 0, 7]
    );
    assert_eq!(
        row(&[-5i64, 300]).convert_to::<u8>().unwrap().data,
        vec![0u8, 255]
    );
    assert_eq!(
        row(&[(1i64 << 53) + 1]).convert_to::<u64>().unwrap().data,
        vec![(1u64 << 53) + 1]
    );
}

/// `DynamicMatrix::convert_to` routes through `Matrix::convert_to`.
#[test]
fn dynamic_convert_to_saturates() {
    let src = DynamicMatrix::new_f32(1, 4, 1, vec![0.5, 2.5, 300.0, -1.0]).unwrap();
    let dst = src.convert_to("u8").expect("OpenCV never fails on range");
    assert_eq!(dst.data_u8().unwrap(), &[0u8, 2, 255, 0]);
}

// ---------------------------------------------------------------------------------------
// add / subtract / multiply / divide on u8 (core_convert_arith.py, "u8_binary")
// ---------------------------------------------------------------------------------------

const A_U8: [u8; 12] = [0, 1, 100, 127, 128, 200, 250, 255, 7, 5, 1, 9];
const B_U8: [u8; 12] = [0, 1, 3, 100, 128, 255, 7, 2, 2, 2, 3, 4];

/// OpenCV: `saturate_cast<uchar>(a + b)`.
#[test]
fn add_u8_saturates() {
    let dst = add(&row(&A_U8), &row(&B_U8)).unwrap();
    assert_eq!(
        dst.data,
        vec![0u8, 2, 103, 227, 255, 255, 255, 255, 9, 7, 4, 13]
    );
}

/// OpenCV: `saturate_cast<uchar>(a - b)` clamps at 0.
#[test]
fn subtract_u8_saturates_at_zero() {
    let dst = subtract(&row(&B_U8), &row(&A_U8)).unwrap();
    assert_eq!(dst.data, vec![0u8, 0, 0, 0, 0, 55, 0, 0, 0, 0, 2, 0]);
}

/// OpenCV: `saturate_cast<uchar>(a * b)`.
#[test]
fn multiply_u8_saturates() {
    let dst = multiply(&row(&A_U8), &row(&B_U8)).unwrap();
    assert_eq!(
        dst.data,
        vec![0u8, 1, 255, 255, 255, 255, 255, 255, 14, 10, 3, 36]
    );
}

/// `add`, `subtract`, `multiply`, `absdiff` at the type bounds (core_saturating_ops.py).
/// i8, u16 and i16 saturate; i32 wraps (OpenCV 4.10.0 `add`/`subtract`/`multiply`/`absdiff`
/// on CV_32S compute in 32-bit integers, e.g. `INT_MAX + 1 = INT_MIN`).
#[test]
fn add_sub_mul_bounds_match_opencv() {
    macro_rules! check {
        ($t:ty, $a:expr, $b:expr, $add_e:expr, $sub_e:expr, $mul_e:expr $(,)?) => {{
            let ma = row::<$t>($a);
            let mb = row::<$t>($b);
            assert_eq!(add(&ma, &mb).unwrap().data, $add_e.to_vec(), "add");
            assert_eq!(
                subtract(&ma, &mb).unwrap().data,
                $sub_e.to_vec(),
                "subtract"
            );
            assert_eq!(
                multiply(&ma, &mb).unwrap().data,
                $mul_e.to_vec(),
                "multiply"
            );
        }};
    }

    check!(
        i8,
        &[127, 127, -128, -128, -128, 127, 0, 0, 1, 3, 64, -65, 127, -128],
        &[1, 127, -1, -128, 127, -128, -128, 127, 3, 1, 2, 2, 2, 2],
        &[127, 127, -128, -128, -1, -1, -128, 127, 4, 4, 66, -63, 127, -126],
        &[126, 0, -127, 0, -128, 127, 127, -127, -2, 2, 62, -67, 125, -128],
        &[127, 127, 127, 127, -128, -128, 0, 0, 3, 3, 127, -128, 127, -128],
    );
    // The multiply value of the pair 65535 * 65535 (index 1) is the only one that is not the
    // measured one: this row is narrow, so cv2 gives 0 there and purecv keeps 65535, see
    // `multiply_u16_overflow_saturates_like_the_opencv_vector_lanes`.
    check!(
        u16,
        &[65535, 65535, 0, 0, 0, 65535, 0, 0, 1, 3, 32768, 100, 65535, 0],
        &[1, 65535, 1, 0, 65535, 0, 0, 65535, 3, 1, 2, 2, 2, 2],
        &[65535, 65535, 1, 0, 65535, 65535, 0, 65535, 4, 4, 32770, 102, 65535, 2],
        &[65534, 0, 0, 0, 0, 65535, 0, 0, 0, 2, 32766, 98, 65533, 0],
        &[65535, 65535, 0, 0, 0, 0, 0, 0, 3, 3, 65535, 200, 65535, 0],
    );
    check!(
        i16,
        &[32767, 32767, -32768, -32768, -32768, 32767, 0, 0, 1, 3, 16384, -16385, 32767, -32768],
        &[1, 32767, -1, -32768, 32767, -32768, -32768, 32767, 3, 1, 2, 2, 2, 2],
        &[32767, 32767, -32768, -32768, -1, -1, -32768, 32767, 4, 4, 16386, -16383, 32767, -32766],
        &[32766, 0, -32767, 0, -32768, 32767, 32767, -32767, -2, 2, 16382, -16387, 32765, -32768],
        &[32767, 32767, 32767, 32767, -32768, -32768, 0, 0, 3, 3, 32767, -32768, 32767, -32768],
    );
    check!(
        i32,
        &[
            2147483647,
            2147483647,
            -2147483648,
            -2147483648,
            -2147483648,
            2147483647,
            0,
            0,
            1,
            3,
            1073741824,
            -1073741825,
            2147483647,
            -2147483648,
        ],
        &[
            1,
            2147483647,
            -1,
            -2147483648,
            2147483647,
            -2147483648,
            -2147483648,
            2147483647,
            3,
            1,
            2,
            2,
            2,
            2
        ],
        &[
            -2147483648,
            -2,
            2147483647,
            0,
            -1,
            -1,
            -2147483648,
            2147483647,
            4,
            4,
            1073741826,
            -1073741823,
            -2147483647,
            -2147483646,
        ],
        &[
            2147483646,
            0,
            -2147483647,
            0,
            1,
            -1,
            -2147483648,
            -2147483647,
            -2,
            2,
            1073741822,
            -1073741827,
            2147483645,
            2147483646,
        ],
        &[
            2147483647,
            1,
            -2147483648,
            0,
            -2147483648,
            -2147483648,
            0,
            0,
            3,
            3,
            -2147483648,
            2147483646,
            -2,
            0,
        ],
    );
}

/// `core_saturating_ops.py`, block `u16_multiply_overflow`. OpenCV's scalar kernel `c_mul`
/// computes `a * b` in `int`, so a product of 2^31 or more (from 46341 * 46341 on) overflows
/// and gives 0, while its vector lanes saturate to 65535: width 64 is all 65535, width 3 all 0,
/// and width 65 is 65535 except for a last 0. purecv keeps saturating at every position, like
/// the vector lanes. 46340 * 46340 still fits in `int`, where the two agree on 65535.
#[test]
fn multiply_u16_overflow_saturates_like_the_opencv_vector_lanes() {
    for value in [46340u16, 46341, 50000, 65535] {
        for width in [3, 64, 65] {
            let a = row(&vec![value; width]);
            assert_eq!(
                multiply(&a, &a).unwrap().data,
                vec![65535u16; width],
                "{value} x{width}"
            );
        }
    }
}

/// A `Scalar` has four values. On more channels the rest of the operand is the default of
/// the type (a project ruling, `cv::Scalar` has no such form to measure), and nothing panics.
#[test]
fn scalar_ops_beyond_four_channels_use_the_default() {
    let src = Matrix::from_vec(1, 1, 5, vec![1u8, 2, 3, 4, 5]);
    let s = Scalar::new(10.0f64, 20.0, 30.0, 40.0);
    assert_eq!(add_scalar(&src, s).unwrap().data, vec![11u8, 22, 33, 44, 5]);
    assert_eq!(
        multiply_scalar(&src, s).unwrap().data,
        vec![10u8, 40, 90, 160, 0]
    );
}

/// The `simd` u8 kernels run on 64 elements (several lanes) and must give the same
/// saturated values as the scalar path.
#[test]
fn add_u8_saturates_across_simd_lanes() {
    let a = vec![128u8; 64];
    let b = vec![128u8; 64];
    assert_eq!(add(&row(&a), &row(&b)).unwrap().data, vec![255u8; 64]);
    let c = vec![1u8; 64];
    let d = vec![3u8; 64];
    assert_eq!(subtract(&row(&c), &row(&d)).unwrap().data, vec![0u8; 64]);
    assert_eq!(multiply(&row(&a), &row(&b)).unwrap().data, vec![255u8; 64]);
}

/// 64-bit integers are added and multiplied exactly above 2^53 (never through `f64`) and
/// saturate at their own bounds; they have no OpenCV depth.
#[test]
fn add_mul_i64_keep_exact_values_above_2_53() {
    let big = (1i64 << 53) + 1;
    let a = row(&[big, i64::MAX, i64::MIN, -big]);
    let b = row(&[2i64, 1, -1, 3]);
    assert_eq!(
        add(&a, &b).unwrap().data,
        vec![big + 2, i64::MAX, i64::MIN, -big + 3]
    );
    assert_eq!(
        multiply(&a, &b).unwrap().data,
        vec![2 * big, i64::MAX, i64::MAX, -3 * big]
    );
    let ubig = (1u64 << 53) + 1;
    let ua = row(&[ubig, u64::MAX, 0u64]);
    let ub = row(&[3u64, 1, 1]);
    assert_eq!(add(&ua, &ub).unwrap().data, vec![ubig + 3, u64::MAX, 1]);
    assert_eq!(
        multiply(&ua, &ub).unwrap().data,
        vec![3 * ubig, u64::MAX, 0]
    );
    assert_eq!(subtract(&ub, &ua).unwrap().data, vec![0u64, 0, 1]);
}

/// OpenCV `absdiff` on CV_8S saturates: |127 - (-128)| = 255 -> 127 (core_saturating_ops.py).
/// Both spellings (`absdiff`, `abs_diff`) are covered.
#[test]
fn absdiff_i8_bounds_match_opencv() {
    let a = row(&[
        127i8, 127, -128, -128, -128, 127, 0, 0, 1, 3, 64, -65, 127, -128,
    ]);
    let b = row(&[1i8, 127, -1, -128, 127, -128, -128, 127, 3, 1, 2, 2, 2, 2]);
    let expected = vec![126i8, 0, 127, 0, 127, 127, 127, 127, 2, 2, 62, 67, 125, 127];
    assert_eq!(absdiff(&a, &b).unwrap().data, expected);
    assert_eq!(abs_diff(&a, &b).unwrap().data, expected);
    // The scalar variant clamps the same way.
    let s = abs_diff_scalar(&row(&[127i8, -128, 0]), Scalar::new(-128i8, 0, 0, 0)).unwrap();
    assert_eq!(s.data, vec![127i8, 0, 127]);
    // i32 wraps like OpenCV: |INT_MIN - INT_MAX| = -1, |0 - INT_MIN| = INT_MIN.
    let c = row(&[i32::MIN, 0, i32::MAX]);
    let d = row(&[i32::MAX, i32::MIN, i32::MIN]);
    assert_eq!(absdiff(&c, &d).unwrap().data, vec![-1, i32::MIN, -1]);
}

/// Sums that fit in u8 match (no saturation involved).
#[test]
fn add_u8_without_overflow_matches_opencv() {
    let a = row(&[0u8, 1, 100, 7, 5, 1, 9]);
    let b = row(&[0u8, 1, 3, 2, 2, 3, 4]);
    assert_eq!(add(&a, &b).unwrap().data, vec![0u8, 2, 103, 9, 7, 4, 13]);
    // Only a >= b here; the underflow of 1 - 3 is covered by `add_u8_saturates_across_simd_lanes`.
    let b2 = row(&[0u8, 1, 3, 2, 2, 1, 4]);
    assert_eq!(
        subtract(&a, &b2).unwrap().data,
        vec![0u8, 0, 97, 5, 3, 0, 5]
    );
}

/// OpenCV: `b != 0 ? saturate_cast<uchar>(float(a) / b) : 0`, so 7/2 = 3.5 -> 4 and
/// 250/7 = 35.7 -> 36. purecv computes the same float quotient.
#[test]
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

/// OpenCV: float division is plain IEEE division (`a / b`), so x/0 is inf, -inf or nan, and
/// purecv does the same.
#[test]
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

/// `core_convert_arith.py`, `divide_ints`: `op_div_scale` computes the quotient of `u16`, `i8`,
/// `i16` and `i32` in `float` and `saturate_cast`s it, so ties go to the even integer
/// (`65535/2` -> 32768, `-7/2` -> -4), the result clamps (`-128/-1` -> 127) and `x/0` is 0.
/// The `i32` operands go through `float` too: `16777217/1` -> 16777216, `1000000007/3` ->
/// 333333344. Quotients of exactly 2^31 are in `divide_i32_quotient_2pow31_clamps`.
#[test]
fn divide_i16_i32_match_opencv() {
    // i16
    let a = row::<i16>(&[
        32767, -32768, -32768, 7, -7, 7, -7, 1, 100, 0, -32768, 5, 32767,
    ]);
    let b = row::<i16>(&[2, -1, 3, 2, 2, -2, -2, 3, 0, 5, 32767, 0, -32768]);
    assert_eq!(
        divide(&a, &b).unwrap().data,
        vec![16384, 32767, -10923, 4, -4, -4, 4, 0, 0, 0, -1, 0, -1,]
    );
    // i32
    let a = row::<i32>(&[
        7,
        -7,
        7,
        -7,
        100,
        0,
        16777217,
        -2147483648,
        2147483647,
        1000000007,
        -5,
        3,
        9,
        10,
    ]);
    let b = row::<i32>(&[2, 2, -2, -2, 0, 3, 1, 1, 2, 3, 0, 2, 3, 4]);
    assert_eq!(
        divide(&a, &b).unwrap().data,
        vec![
            4,
            -4,
            -4,
            4,
            0,
            0,
            16777216,
            -2147483648,
            1073741824,
            333333344,
            0,
            2,
            3,
            2,
        ]
    );
}

/// `core_convert_arith.py`, `divide_ints`, i8 and u16 (the same float working type).
#[test]
fn divide_i8_u16_match_opencv() {
    let a = row::<i8>(&[
        127, -128, -128, 127, 7, -7, 7, -7, 100, -100, 5, 0, 1, -128, 127,
    ]);
    let b = row::<i8>(&[2, -1, 3, -1, 2, 2, -2, -2, 0, 0, 7, 0, -128, 127, 127]);
    assert_eq!(
        divide(&a, &b).unwrap().data,
        vec![64, 127, -43, -127, 4, -4, -4, 4, 0, 0, 1, 0, 0, -1, 1,]
    );
    let a = row::<u16>(&[65535, 1, 7, 100, 65535, 0, 3, 40000, 65535, 5, 9]);
    let b = row::<u16>(&[1, 3, 2, 0, 65535, 5, 7, 3, 2, 0, 4]);
    assert_eq!(
        divide(&a, &b).unwrap().data,
        vec![65535, 0, 4, 0, 1, 0, 0, 13333, 32768, 0, 2,]
    );
}

/// OpenCV gives `INT_MIN` for `INT_MAX / 1` and `INT_MIN / -1` (`divide_i32_2pow31`
/// measured [-2147483648, -2147483648]): the float quotient is 2^31, outside `int`, and the
/// x86 conversion returns `INT_MIN` instead of clamping. That is not ported; purecv clamps to
/// `i32::MAX`, like every other out-of-range quotient.
#[test]
fn divide_i32_quotient_2pow31_clamps() {
    let a = row(&[i32::MAX, i32::MIN]);
    let b = row(&[1i32, -1]);
    assert_eq!(divide(&a, &b).unwrap().data, vec![i32::MAX, i32::MAX]);
}

/// `u32`, `u64` and `i64` have no OpenCV depth. purecv divides them exactly (no float
/// detour) with the same half-to-even rounding, clamping and `x/0 = 0`; not a measured value.
#[test]
fn divide_wide_integers_are_exact_and_never_panic() {
    let big = (1i64 << 53) + 1;
    let a = row(&[big, 7i64, -7, 7, -7, 5, i64::MIN, i64::MAX, 9]);
    let b = row(&[1i64, 2, 2, -2, -2, 0, -1, 1, 2]);
    assert_eq!(
        divide(&a, &b).unwrap().data,
        vec![big, 4, -4, -4, 4, 0, i64::MAX, i64::MAX, 4]
    );
    let ua = row(&[u64::MAX, 7u64, 5, 3, 1]);
    let ub = row(&[1u64, 2, 0, 2, 2]);
    assert_eq!(divide(&ua, &ub).unwrap().data, vec![u64::MAX, 4, 0, 2, 0]);
    let wa = row(&[u32::MAX, 7u32, 9, 0]);
    let wb = row(&[1u32, 2, 0, 0]);
    assert_eq!(divide(&wa, &wb).unwrap().data, vec![u32::MAX, 4, 0, 0]);
}

/// `core_convert_arith.py`, `divide_u8_seeded`: 64 `u8` pairs from `RandomState(209)`, 0..256
/// over 0..40, with two zero divisors (indices 22 and 50).
#[test]
fn divide_u8_seeded_fixture_matches_opencv() {
    let a = row::<u8>(&[
        222, 125, 240, 161, 71, 234, 171, 177, 58, 39, 31, 109, 144, 96, 201, 220, 178, 13, 34, 3,
        157, 5, 20, 104, 230, 129, 90, 86, 231, 226, 35, 22, 159, 70, 183, 22, 147, 154, 199, 131,
        174, 193, 229, 63, 116, 115, 230, 236, 11, 126, 200, 75, 143, 120, 141, 214, 89, 84, 94,
        148, 39, 36, 137, 63,
    ]);
    let b = row::<u8>(&[
        29, 9, 2, 16, 30, 5, 6, 12, 16, 3, 22, 16, 31, 30, 39, 16, 16, 18, 9, 29, 8, 14, 0, 8, 35,
        16, 7, 1, 38, 7, 5, 27, 18, 32, 5, 34, 20, 6, 12, 39, 27, 18, 15, 4, 32, 32, 6, 28, 21, 36,
        0, 7, 5, 20, 39, 27, 28, 23, 18, 39, 26, 3, 16, 5,
    ]);
    let want = vec![
        8, 14, 120, 10, 2, 47, 28, 15, 4, 13, 1, 7, 5, 3, 5, 14, 11, 1, 4, 0, 20, 0, 0, 13, 7, 8,
        13, 86, 6, 32, 7, 1, 9, 2, 37, 1, 7, 26, 17, 3, 6, 11, 15, 16, 4, 4, 38, 8, 1, 4, 0, 11,
        29, 6, 4, 8, 3, 4, 5, 4, 2, 12, 9, 13,
    ];
    assert_eq!(divide(&a, &b).unwrap().data, want);
}

/// `core_convert_arith.py`, `divide_float_lanes`: 64 elements, so every SIMD lane position sees
/// 0/0 (NaN), x/0 (+-inf) and signed zeros. Bit-exact for non-NaN values, so `-0.0` counts.
/// Run with `--features simd` as well: the SIMD kernels must agree with the scalar path.
#[test]
#[allow(clippy::excessive_precision)]
fn divide_f32_by_zero_is_ieee_across_simd_lanes() {
    let a: [f64; 64] = [
        0.0, -30.5, -29.5, -28.5, -27.5, 0.0, -25.5, -24.5, -23.5, -22.5, 0.0, -20.5, -19.5, -18.5,
        -17.5, 0.0, -15.5, -14.5, -13.5, -12.5, 0.0, -10.5, -9.5, -8.5, -7.5, 0.0, -5.5, -4.5,
        -3.5, -2.5, 0.0, -0.5, 0.5, 1.5, 2.5, 0.0, 4.5, 5.5, 6.5, 7.5, 0.0, 9.5, 10.5, 11.5, 12.5,
        0.0, 14.5, 15.5, 16.5, 17.5, 0.0, 19.5, 20.5, 21.5, 22.5, 0.0, 24.5, 25.5, 26.5, 27.5, 0.0,
        29.5, 30.5, 31.5,
    ];
    let b: [f64; 64] = [
        0.0, -4.75, -4.5, 0.0, -4.0, -3.75, 0.0, -3.25, -3.0, 0.0, -2.5, -2.25, 0.0, -1.75, -1.5,
        0.0, -1.0, -0.75, 0.0, -0.25, 0.0, 0.0, 0.5, 0.75, 0.0, 1.25, 1.5, 0.0, 2.0, 2.25, 0.0,
        2.75, 3.0, 0.0, 3.5, 3.75, 0.0, 4.25, 4.5, 0.0, 5.0, 5.25, 0.0, 5.75, 6.0, 0.0, 6.5, 6.75,
        0.0, 7.25, 7.5, 0.0, 8.0, 8.25, 0.0, 8.75, 9.0, 0.0, 9.5, 9.75, 0.0, 10.25, 10.5, 0.0,
    ];
    let want32: [f32; 64] = [
        f32::NAN,
        6.4210524559021,
        6.55555534362793,
        f32::NEG_INFINITY,
        6.875,
        -0.0,
        f32::NEG_INFINITY,
        7.538461685180664,
        7.833333492279053,
        f32::NEG_INFINITY,
        -0.0,
        9.11111068725586,
        f32::NEG_INFINITY,
        10.571428298950195,
        11.666666984558105,
        f32::NAN,
        15.5,
        19.33333396911621,
        f32::NEG_INFINITY,
        50.0,
        f32::NAN,
        f32::NEG_INFINITY,
        -19.0,
        -11.333333015441895,
        f32::NEG_INFINITY,
        0.0,
        -3.6666667461395264,
        f32::NEG_INFINITY,
        -1.75,
        -1.1111111640930176,
        f32::NAN,
        -0.1818181872367859,
        0.1666666716337204,
        f32::INFINITY,
        0.7142857313156128,
        0.0,
        f32::INFINITY,
        1.2941176891326904,
        1.4444444179534912,
        f32::INFINITY,
        0.0,
        1.8095238208770752,
        f32::INFINITY,
        2.0,
        2.0833332538604736,
        f32::NAN,
        2.230769157409668,
        2.2962963581085205,
        f32::INFINITY,
        2.4137930870056152,
        0.0,
        f32::INFINITY,
        2.5625,
        2.60606050491333,
        f32::INFINITY,
        0.0,
        2.722222328186035,
        f32::INFINITY,
        2.78947377204895,
        2.8205127716064453,
        f32::NAN,
        2.878048896789551,
        2.904761791229248,
        f32::INFINITY,
    ];
    let want64: [f64; 64] = [
        f64::NAN,
        6.421052631578948,
        6.555555555555555,
        f64::NEG_INFINITY,
        6.875,
        -0.0,
        f64::NEG_INFINITY,
        7.538461538461538,
        7.833333333333333,
        f64::NEG_INFINITY,
        -0.0,
        9.11111111111111,
        f64::NEG_INFINITY,
        10.571428571428571,
        11.666666666666666,
        f64::NAN,
        15.5,
        19.333333333333332,
        f64::NEG_INFINITY,
        50.0,
        f64::NAN,
        f64::NEG_INFINITY,
        -19.0,
        -11.333333333333334,
        f64::NEG_INFINITY,
        0.0,
        -3.6666666666666665,
        f64::NEG_INFINITY,
        -1.75,
        -1.1111111111111112,
        f64::NAN,
        -0.18181818181818182,
        0.16666666666666666,
        f64::INFINITY,
        0.7142857142857143,
        0.0,
        f64::INFINITY,
        1.2941176470588236,
        1.4444444444444444,
        f64::INFINITY,
        0.0,
        1.8095238095238095,
        f64::INFINITY,
        2.0,
        2.0833333333333335,
        f64::NAN,
        2.230769230769231,
        2.2962962962962963,
        f64::INFINITY,
        2.413793103448276,
        0.0,
        f64::INFINITY,
        2.5625,
        2.606060606060606,
        f64::INFINITY,
        0.0,
        2.7222222222222223,
        f64::INFINITY,
        2.789473684210526,
        2.8205128205128207,
        f64::NAN,
        2.8780487804878048,
        2.9047619047619047,
        f64::INFINITY,
    ];
    let a32: Vec<f32> = a.iter().map(|&v| v as f32).collect();
    let b32: Vec<f32> = b.iter().map(|&v| v as f32).collect();
    let got = divide(&row(&a32), &row(&b32)).unwrap().data;
    assert_eq!(got.len(), 64);
    for (i, (g, w)) in got.iter().zip(want32.iter()).enumerate() {
        assert!(
            (g.is_nan() && w.is_nan()) || g.to_bits() == w.to_bits(),
            "f32 lane {i}: got {g}, OpenCV {w}"
        );
    }
    let got = divide(&row(&a), &row(&b)).unwrap().data;
    for (i, (g, w)) in got.iter().zip(want64.iter()).enumerate() {
        assert!(
            (g.is_nan() && w.is_nan()) || g.to_bits() == w.to_bits(),
            "f64 lane {i}: got {g}, OpenCV {w}"
        );
    }
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

/// The scalar goes through `saturate_cast` to the `CV_32S` working type (10.7 -> 11,
/// 10.5 -> 10, -1.5 -> -2, 300 -> 300) and the sum is then saturated to u8, so `+300` gives
/// 255 and `+(-1.5)` subtracts 2.
#[test]
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

/// The product or quotient is computed in double and converted once with
/// `saturate_cast<uchar>`: 9 * 0.3 = 2.7 -> 3, 128 / 3 = 42.67 -> 43 and
/// 127 / 0.4 = 317.5 -> 255.
#[test]
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

/// The same operations on the other depths (core_convert_arith.py, "other_depths_scalar").
/// `arithm_op` picks the working type: `CV_32S` for add and subtract of every integer depth
/// below 32F (the scalar goes through `saturate_cast<int>`, the sum is then saturated to the
/// destination depth), the source float depth for f32/f64 add and subtract, and `CV_64F` for
/// multiply and divide of every depth (the product or quotient is rounded half to even and
/// saturated once).
#[test]
fn scalar_ops_other_depths_match_opencv() {
    // i8
    let a = row::<i8>(&[-128, -100, -1, 0, 1, 50, 100, 127]);
    assert_eq!(
        add_scalar(&a, Scalar::new(10.7, 0.0, 0.0, 0.0))
            .unwrap()
            .data,
        vec![-117, -89, 10, 11, 12, 61, 111, 127]
    );
    assert_eq!(
        subtract_scalar(&a, Scalar::new(10.5, 0.0, 0.0, 0.0))
            .unwrap()
            .data,
        vec![-128, -110, -11, -10, -9, 40, 90, 117]
    );
    assert_eq!(
        add_scalar(&a, Scalar::new(300.0, 0.0, 0.0, 0.0))
            .unwrap()
            .data,
        vec![127, 127, 127, 127, 127, 127, 127, 127]
    );
    assert_eq!(
        subtract_scalar(&a, Scalar::new(300.0, 0.0, 0.0, 0.0))
            .unwrap()
            .data,
        vec![-128, -128, -128, -128, -128, -128, -128, -128]
    );
    assert_eq!(
        add_scalar(&a, Scalar::new(-1.5, 0.0, 0.0, 0.0))
            .unwrap()
            .data,
        vec![-128, -102, -3, -2, -1, 48, 98, 125]
    );
    assert_eq!(
        multiply_scalar(&a, Scalar::new(0.3, 0.0, 0.0, 0.0))
            .unwrap()
            .data,
        vec![-38, -30, 0, 0, 0, 15, 30, 38]
    );
    assert_eq!(
        multiply_scalar(&a, Scalar::new(1.5, 0.0, 0.0, 0.0))
            .unwrap()
            .data,
        vec![-128, -128, -2, 0, 2, 75, 127, 127]
    );
    assert_eq!(
        divide_scalar(&a, Scalar::new(3.0, 0.0, 0.0, 0.0))
            .unwrap()
            .data,
        vec![-43, -33, 0, 0, 0, 17, 33, 42]
    );
    assert_eq!(
        divide_scalar(&a, Scalar::new(-2.5, 0.0, 0.0, 0.0))
            .unwrap()
            .data,
        vec![51, 40, 0, 0, 0, -20, -40, -51]
    );

    // u16
    let a = row::<u16>(&[0, 1, 100, 1000, 32768, 60000, 65534, 65535]);
    assert_eq!(
        add_scalar(&a, Scalar::new(10.7, 0.0, 0.0, 0.0))
            .unwrap()
            .data,
        vec![11, 12, 111, 1011, 32779, 60011, 65535, 65535]
    );
    assert_eq!(
        subtract_scalar(&a, Scalar::new(10.5, 0.0, 0.0, 0.0))
            .unwrap()
            .data,
        vec![0, 0, 90, 990, 32758, 59990, 65524, 65525]
    );
    assert_eq!(
        add_scalar(&a, Scalar::new(-1.5, 0.0, 0.0, 0.0))
            .unwrap()
            .data,
        vec![0, 0, 98, 998, 32766, 59998, 65532, 65533]
    );
    assert_eq!(
        add_scalar(&a, Scalar::new(100000.0, 0.0, 0.0, 0.0))
            .unwrap()
            .data,
        vec![65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535]
    );
    assert_eq!(
        subtract_scalar(&a, Scalar::new(100000.0, 0.0, 0.0, 0.0))
            .unwrap()
            .data,
        vec![0, 0, 0, 0, 0, 0, 0, 0]
    );
    assert_eq!(
        add_scalar(&a, Scalar::new(-70000.0, 0.0, 0.0, 0.0))
            .unwrap()
            .data,
        vec![0, 0, 0, 0, 0, 0, 0, 0]
    );
    assert_eq!(
        multiply_scalar(&a, Scalar::new(0.3, 0.0, 0.0, 0.0))
            .unwrap()
            .data,
        vec![0, 0, 30, 300, 9830, 18000, 19660, 19660]
    );
    assert_eq!(
        multiply_scalar(&a, Scalar::new(-0.5, 0.0, 0.0, 0.0))
            .unwrap()
            .data,
        vec![0, 0, 0, 0, 0, 0, 0, 0]
    );
    assert_eq!(
        divide_scalar(&a, Scalar::new(3.0, 0.0, 0.0, 0.0))
            .unwrap()
            .data,
        vec![0, 0, 33, 333, 10923, 20000, 21845, 21845]
    );
    assert_eq!(
        divide_scalar(&a, Scalar::new(0.4, 0.0, 0.0, 0.0))
            .unwrap()
            .data,
        vec![0, 2, 250, 2500, 65535, 65535, 65535, 65535]
    );

    // i16
    let a = row::<i16>(&[-32768, -30000, -1, 0, 1, 1000, 30000, 32767]);
    assert_eq!(
        add_scalar(&a, Scalar::new(10.7, 0.0, 0.0, 0.0))
            .unwrap()
            .data,
        vec![-32757, -29989, 10, 11, 12, 1011, 30011, 32767]
    );
    assert_eq!(
        subtract_scalar(&a, Scalar::new(300.0, 0.0, 0.0, 0.0))
            .unwrap()
            .data,
        vec![-32768, -30300, -301, -300, -299, 700, 29700, 32467]
    );
    assert_eq!(
        add_scalar(&a, Scalar::new(-1.5, 0.0, 0.0, 0.0))
            .unwrap()
            .data,
        vec![-32768, -30002, -3, -2, -1, 998, 29998, 32765]
    );
    assert_eq!(
        multiply_scalar(&a, Scalar::new(0.3, 0.0, 0.0, 0.0))
            .unwrap()
            .data,
        vec![-9830, -9000, 0, 0, 0, 300, 9000, 9830]
    );
    assert_eq!(
        multiply_scalar(&a, Scalar::new(-0.5, 0.0, 0.0, 0.0))
            .unwrap()
            .data,
        vec![16384, 15000, 0, 0, 0, -500, -15000, -16384]
    );
    assert_eq!(
        divide_scalar(&a, Scalar::new(3.0, 0.0, 0.0, 0.0))
            .unwrap()
            .data,
        vec![-10923, -10000, 0, 0, 0, 333, 10000, 10922]
    );

    // i32
    let a = row::<i32>(&[
        -2147483648,
        -2147483647,
        -100000,
        -1,
        0,
        1,
        100000,
        2147483646,
        2147483647,
    ]);
    assert_eq!(
        add_scalar(&a, Scalar::new(10.7, 0.0, 0.0, 0.0))
            .unwrap()
            .data,
        vec![
            -2147483637,
            -2147483636,
            -99989,
            10,
            11,
            12,
            100011,
            -2147483639,
            -2147483638
        ]
    );
    assert_eq!(
        subtract_scalar(&a, Scalar::new(10.5, 0.0, 0.0, 0.0))
            .unwrap()
            .data,
        vec![2147483638, 2147483639, -100010, -11, -10, -9, 99990, 2147483636, 2147483637]
    );
    assert_eq!(
        add_scalar(&a, Scalar::new(-1.5, 0.0, 0.0, 0.0))
            .unwrap()
            .data,
        vec![2147483646, 2147483647, -100002, -3, -2, -1, 99998, 2147483644, 2147483645]
    );
    assert_eq!(
        add_scalar(&a, Scalar::new(100000.0, 0.0, 0.0, 0.0))
            .unwrap()
            .data,
        vec![
            -2147383648,
            -2147383647,
            0,
            99999,
            100000,
            100001,
            200000,
            -2147383650,
            -2147383649
        ]
    );
    assert_eq!(
        add_scalar(&a, Scalar::new(-70000.0, 0.0, 0.0, 0.0))
            .unwrap()
            .data,
        vec![
            2147413648, 2147413649, -170000, -70001, -70000, -69999, 30000, 2147413646, 2147413647
        ]
    );
    assert_eq!(
        multiply_scalar(&a, Scalar::new(0.3, 0.0, 0.0, 0.0))
            .unwrap()
            .data,
        vec![-644245094, -644245094, -30000, 0, 0, 0, 30000, 644245094, 644245094]
    );
    assert_eq!(
        multiply_scalar(&a, Scalar::new(0.1, 0.0, 0.0, 0.0))
            .unwrap()
            .data,
        vec![-214748365, -214748365, -10000, 0, 0, 0, 10000, 214748365, 214748365]
    );
    assert_eq!(
        multiply_scalar(&a, Scalar::new(-0.5, 0.0, 0.0, 0.0))
            .unwrap()
            .data,
        vec![
            1073741824,
            1073741824,
            50000,
            0,
            0,
            0,
            -50000,
            -1073741823,
            -1073741824
        ]
    );
    assert_eq!(
        divide_scalar(&a, Scalar::new(3.0, 0.0, 0.0, 0.0))
            .unwrap()
            .data,
        vec![-715827883, -715827882, -33333, 0, 0, 0, 33333, 715827882, 715827882]
    );
    assert_eq!(
        divide_scalar(&a, Scalar::new(-2.5, 0.0, 0.0, 0.0))
            .unwrap()
            .data,
        vec![858993459, 858993459, 40000, 0, 0, 0, -40000, -858993458, -858993459]
    );

    // f32
    let a = row::<f32>(&[
        -1e+10, -300.5, -1.5, -0.25, 0.0, 0.1, 1.0, 2.5, 255.5, 1e+10,
    ]);
    assert_eq!(
        add_scalar(&a, Scalar::new(10.7, 0.0, 0.0, 0.0))
            .unwrap()
            .data,
        vec![-1e+10, -289.8, 9.2, 10.45, 10.7, 10.8, 11.7, 13.2, 266.2, 1e+10]
    );
    assert_eq!(
        subtract_scalar(&a, Scalar::new(10.5, 0.0, 0.0, 0.0))
            .unwrap()
            .data,
        vec![-1e+10, -311.0, -12.0, -10.75, -10.5, -10.4, -9.5, -8.0, 245.0, 1e+10]
    );
    assert_eq!(
        add_scalar(&a, Scalar::new(100000.0, 0.0, 0.0, 0.0))
            .unwrap()
            .data,
        vec![
            -9.9999e+09,
            99699.5,
            99998.5,
            99999.75,
            100000.0,
            100000.1,
            100001.0,
            100002.5,
            100255.5,
            1.00001e+10
        ]
    );
    assert_eq!(
        multiply_scalar(&a, Scalar::new(0.3, 0.0, 0.0, 0.0))
            .unwrap()
            .data,
        vec![
            -3e+09,
            -90.15,
            -0.45,
            -0.075,
            0.0,
            0.030000001,
            0.3,
            0.75,
            76.65,
            3e+09
        ]
    );
    assert_eq!(
        multiply_scalar(&a, Scalar::new(0.1, 0.0, 0.0, 0.0))
            .unwrap()
            .data,
        vec![-1e+09, -30.05, -0.15, -0.025, 0.0, 0.01, 0.1, 0.25, 25.55, 1e+09]
    );
    assert_eq!(
        multiply_scalar(&a, Scalar::new(-0.5, 0.0, 0.0, 0.0))
            .unwrap()
            .data,
        vec![5e+09, 150.25, 0.75, 0.125, -0.0, -0.05, -0.5, -1.25, -127.75, -5e+09]
    );
    assert_eq!(
        divide_scalar(&a, Scalar::new(3.0, 0.0, 0.0, 0.0))
            .unwrap()
            .data,
        vec![
            -3.3333332e+09,
            -100.166664,
            -0.5,
            -0.083333336,
            0.0,
            0.033333335,
            0.33333334,
            0.8333333,
            85.166664,
            3.3333332e+09
        ]
    );
    assert_eq!(
        divide_scalar(&a, Scalar::new(0.4, 0.0, 0.0, 0.0))
            .unwrap()
            .data,
        vec![-2.5e+10, -751.25, -3.75, -0.625, 0.0, 0.25, 2.5, 6.25, 638.75, 2.5e+10]
    );
    assert_eq!(
        divide_scalar(&a, Scalar::new(-2.5, 0.0, 0.0, 0.0))
            .unwrap()
            .data,
        vec![4e+09, 120.2, 0.6, 0.1, -0.0, -0.04, -0.4, -1.0, -102.2, -4e+09]
    );

    // f64
    let a = row::<f64>(&[-10000000000.0, -0.1, 0.0, 0.1, 1.0, 10000000000.0]);
    assert_eq!(
        add_scalar(&a, Scalar::new(0.1, 0.0, 0.0, 0.0))
            .unwrap()
            .data,
        vec![-9999999999.9, 0.0, 0.1, 0.2, 1.1, 10000000000.1]
    );
    assert_eq!(
        multiply_scalar(&a, Scalar::new(0.1, 0.0, 0.0, 0.0))
            .unwrap()
            .data,
        vec![
            -1000000000.0,
            -0.010000000000000002,
            0.0,
            0.010000000000000002,
            0.1,
            1000000000.0
        ]
    );
    assert_eq!(
        divide_scalar(&a, Scalar::new(3.0, 0.0, 0.0, 0.0))
            .unwrap()
            .data,
        vec![
            -3333333333.3333335,
            -0.03333333333333333,
            0.0,
            0.03333333333333333,
            0.3333333333333333,
            3333333333.3333335
        ]
    );
}

/// The scalar operands are per channel (core_convert_arith.py, "u8c3_scalar"): channel `c`
/// uses `Scalar::v[c]`, each converted on its own.
#[test]
fn scalar_ops_u8_three_channels_match_opencv() {
    let a = Matrix::from_vec(1, 3, 3, vec![0u8, 100, 250, 10, 200, 255, 255, 1, 7]);
    assert_eq!(
        add_scalar(&a, Scalar::new(10.7, -1.5, 300.0, 0.0))
            .unwrap()
            .data,
        vec![11u8, 98, 255, 21, 198, 255, 255, 0, 255]
    );
    assert_eq!(
        multiply_scalar(&a, Scalar::new(0.3, 1.5, 2.0, 0.0))
            .unwrap()
            .data,
        vec![0u8, 150, 255, 3, 255, 255, 76, 2, 14]
    );
    assert_eq!(
        divide_scalar(&a, Scalar::new(3.0, 0.4, 0.0, 0.0))
            .unwrap()
            .data,
        vec![0u8, 250, 0, 3, 255, 0, 85, 2, 0]
    );
}

/// f32 and f64 divided by a zero scalar follow IEEE (core_convert_arith.py,
/// "other_depths_scalar"): `x / 0` is `±inf` and `0 / 0` is NaN.
#[test]
fn divide_scalar_float_by_zero_matches_opencv() {
    let a = row::<f32>(&[-1e10, -1.5, 0.0, 0.1, 255.5, 1e10]);
    let out = divide_scalar(&a, Scalar::new(0.0, 0.0, 0.0, 0.0))
        .unwrap()
        .data;
    assert_eq!(out[0], f32::NEG_INFINITY);
    assert_eq!(out[1], f32::NEG_INFINITY);
    assert!(out[2].is_nan());
    assert_eq!(out[3], f32::INFINITY);
    assert_eq!(out[4], f32::INFINITY);
    assert_eq!(out[5], f32::INFINITY);

    let a = row::<f64>(&[-1e10, -0.1, 0.0, 0.1, 1.0, 1e10]);
    let out = divide_scalar(&a, Scalar::new(0.0, 0.0, 0.0, 0.0))
        .unwrap()
        .data;
    assert_eq!(out[0], f64::NEG_INFINITY);
    assert_eq!(out[1], f64::NEG_INFINITY);
    assert!(out[2].is_nan());
    assert_eq!(out[3], f64::INFINITY);
    assert_eq!(out[4], f64::INFINITY);
    assert_eq!(out[5], f64::INFINITY);
}

/// purecv's clamp, not an OpenCV value. OpenCV converts a scalar of 3e9 to `int` with
/// `saturate_cast<int>(double)`, whose out-of-range result comes from the x86 conversion
/// (`INT_MIN`), so its output for such a scalar is arbitrary (i8 + 3e9 gives 127 for
/// negative pixels and -128 for the others). purecv clamps the scalar to the `i32` range
/// and the result to the destination range.
#[test]
fn scalar_operand_beyond_i32_clamps_in_purecv() {
    let a = row::<i8>(&[-128, 0, 127]);
    assert_eq!(
        add_scalar(&a, Scalar::new(3e9, 0.0, 0.0, 0.0))
            .unwrap()
            .data,
        vec![127i8, 127, 127]
    );
    assert_eq!(
        subtract_scalar(&a, Scalar::new(3e9, 0.0, 0.0, 0.0))
            .unwrap()
            .data,
        vec![-128i8, -128, -128]
    );
    let a = row::<u16>(&[0, 65535]);
    assert_eq!(
        add_scalar(&a, Scalar::new(3e9, 0.0, 0.0, 0.0))
            .unwrap()
            .data,
        vec![65535u16, 65535]
    );
    let a = row::<i32>(&[0]);
    assert_eq!(
        add_scalar(&a, Scalar::new(3e9, 0.0, 0.0, 0.0))
            .unwrap()
            .data,
        vec![i32::MAX]
    );
}

/// purecv's choice, not an OpenCV value. An integer matrix divided by a zero scalar gives 0
/// at every depth (OpenCV's u8 result is 0 too, which `divide_scalar_u8_by_zero_matches_opencv`
/// pins); OpenCV's i8, i16 and i32 results (-128, -32768, `INT_MIN`) are the x86 conversion
/// of an infinite quotient. An integer product beyond the `i32` range clamps to the
/// destination range where OpenCV's result is arbitrary (i16 30000 * 1e5 gives -32768).
#[test]
fn integer_scalar_overflow_and_zero_divisor_clamp_in_purecv() {
    let zero = Scalar::new(0.0, 0.0, 0.0, 0.0);
    assert_eq!(
        divide_scalar(&row::<i8>(&[-128, 0, 127]), zero)
            .unwrap()
            .data,
        vec![0i8; 3]
    );
    assert_eq!(
        divide_scalar(&row::<i16>(&[-32768, 5, 32767]), zero)
            .unwrap()
            .data,
        vec![0i16; 3]
    );
    assert_eq!(
        divide_scalar(&row::<i32>(&[i32::MIN, 5, i32::MAX]), zero)
            .unwrap()
            .data,
        vec![0i32; 3]
    );
    assert_eq!(
        multiply_scalar(
            &row::<i16>(&[30000, -30000]),
            Scalar::new(1e5, 0.0, 0.0, 0.0)
        )
        .unwrap()
        .data,
        vec![32767i16, -32768]
    );
}

/// `gemm` on an integer matrix. OpenCV rejects it (`CV_Assert(type == CV_32FC1 || ...)`), so
/// there is no OpenCV value; this is purecv's choice: accumulate in f64, then round half to
/// even and saturate to the output type like every other integer output.
#[test]
fn gemm_u8_output_saturates() {
    let a = Matrix::from_vec(1, 2, 1, vec![200u8, 100]);
    let b = Matrix::from_vec(2, 1, 1, vec![2u8, 1]);
    let empty = Matrix::<u8>::new(0, 0, 1);
    // 200 * 2 + 100 * 1 = 500
    assert_eq!(gemm(&a, &b, 1.0, &empty, 0.0, 0).unwrap().data, vec![255u8]);
    // -500 saturates at the other end
    assert_eq!(gemm(&a, &b, -1.0, &empty, 0.0, 0).unwrap().data, vec![0u8]);
    // 3 * 0.5 = 1.5 -> 2 and 5 * 0.5 = 2.5 -> 2 (ties to even)
    let one = Matrix::from_vec(2, 1, 1, vec![1u8, 1]);
    let a3 = Matrix::from_vec(1, 2, 1, vec![1u8, 2]);
    let a5 = Matrix::from_vec(1, 2, 1, vec![2u8, 3]);
    assert_eq!(
        gemm(&a3, &one, 0.5, &empty, 0.0, 0).unwrap().data,
        vec![2u8]
    );
    assert_eq!(
        gemm(&a5, &one, 0.5, &empty, 0.0, 0).unwrap().data,
        vec![2u8]
    );
    // alpha * A * B + beta * C: 500 * 0.25 + 0.5 * 3 = 126.5 -> 126
    let c = Matrix::from_vec(1, 1, 1, vec![3u8]);
    assert_eq!(gemm(&a, &b, 0.25, &c, 0.5, 0).unwrap().data, vec![126u8]);
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

/// `saturate_cast<uchar>(a*alpha + b*beta + gamma)` computed in `f32`: the sum is
/// rounded half to even (51.5 -> 52, 3.5 -> 4, 128.5 -> 128) and clamped (910 -> 255).
#[test]
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

// `core_convert_arith.py`, block `add_weighted_u8_seeded` (`RandomState(211)`): SEED_A and
// SEED_B are keys `a` and `b`, SEED_W_0_3_0_7_10_5 is key `0.3_0.7_10.5` and SEED_W_HALF is key
// `0.5_0.5_0`, the `cv2.addWeighted` results (the 8-bit depths compute in `f32`). The wheel's
// vector loop fuses the multiply-add, which can move a result that lies within one `f32` step
// of a tie; these scalars agree with the unfused loop.
const SEED_A: [u8; 64] = [
    6, 70, 102, 94, 252, 4, 42, 203, 16, 204, 158, 233, 14, 119, 47, 217, 2, 147, 153, 29, 143,
    151, 14, 147, 201, 144, 42, 17, 250, 254, 158, 67, 126, 114, 239, 18, 3, 112, 87, 72, 171, 29,
    43, 15, 70, 230, 174, 217, 96, 155, 116, 103, 111, 218, 218, 205, 87, 221, 216, 172, 87, 109,
    227, 115,
];
const SEED_B: [u8; 64] = [
    161, 128, 177, 98, 165, 72, 89, 211, 15, 154, 87, 146, 211, 208, 167, 161, 150, 209, 181, 250,
    134, 35, 246, 151, 200, 245, 213, 155, 113, 89, 98, 76, 27, 82, 117, 117, 241, 11, 140, 231,
    126, 175, 59, 93, 148, 18, 244, 89, 91, 232, 11, 116, 37, 252, 29, 236, 238, 153, 205, 97, 142,
    117, 248, 103,
];
const SEED_W_0_3_0_7_10_5: [u8; 64] = [
    125, 121, 165, 107, 202, 62, 85, 219, 26, 180, 119, 183, 162, 192, 142, 188, 116, 201, 183,
    194, 147, 80, 187, 160, 211, 225, 172, 124, 165, 149, 126, 84, 67, 102, 164, 98, 180, 52, 135,
    194, 150, 142, 65, 80, 135, 92, 234, 138, 103, 219, 53, 123, 70, 252, 96, 237, 203, 184, 219,
    130, 136, 125, 252, 117,
];
const SEED_W_HALF: [u8; 64] = [
    84, 99, 140, 96, 208, 38, 66, 207, 16, 179, 122, 190, 112, 164, 107, 189, 76, 178, 167, 140,
    138, 93, 130, 149, 200, 194, 128, 86, 182, 172, 128, 72, 76, 98, 178, 68, 122, 62, 114, 152,
    148, 102, 51, 54, 109, 124, 209, 153, 94, 194, 64, 110, 74, 235, 124, 220, 162, 187, 210, 134,
    114, 113, 238, 109,
];

/// 64 elements fill whole SIMD vectors and a tail; the `f32` working type decides every
/// result, including the `.5` ties of the second set.
#[test]
fn add_weighted_u8_seeded_fixture_matches_opencv() {
    let a = row(&SEED_A);
    let b = row(&SEED_B);
    assert_eq!(
        add_weighted(&a, 0.3, &b, 0.7, 10.5).unwrap().data,
        SEED_W_0_3_0_7_10_5
    );
    assert_eq!(
        add_weighted(&a, 0.5, &b, 0.5, 0.0).unwrap().data,
        SEED_W_HALF
    );
}

/// The other depths of `addWeighted`: `arithm.simd.hpp` computes the 8- and 16-bit depths in
/// `float` (`add_weighted_loop`) and `CV_32S`, `CV_32F` and `CV_64F` in `double`
/// (`add_weighted_loop_d`); the sum is rounded half to even and clamped to the depth.
#[test]
fn add_weighted_other_depths_match_opencv() {
    // (0.5, 0.5, 0), (2, 2, 0), (0.3, 0.7, 10.5), (1, -1, 0)
    let scalars = [
        (0.5, 0.5, 0.0),
        (2.0, 2.0, 0.0),
        (0.3, 0.7, 10.5),
        (1.0, -1.0, 0.0),
    ];

    let a = row(&[-128i8, -127, 127, 100, 1, -1, 0, 3, -100, 50]);
    let b = row(&[-128i8, 126, 127, 100, 2, -2, 0, 4, -100, -60]);
    let expect: [[i8; 10]; 4] = [
        [-128, 0, 127, 100, 2, -2, 0, 4, -100, -5],
        [-128, -2, 127, 127, 6, -6, 0, 14, -128, -20],
        [-118, 61, 127, 110, 12, 9, 10, 14, -90, -16],
        [0, -128, 0, 0, -1, 1, 0, -1, 0, 110],
    ];
    for ((al, be, ga), want) in scalars.iter().zip(expect) {
        let got = add_weighted(&a, *al, &b, *be, *ga).unwrap().data;
        assert_eq!(got, want, "i8 {al} {be} {ga}");
    }

    let a = row(&[0u16, 1, 65535, 65535, 3, 1000, 40000, 65534, 5, 32768]);
    let b = row(&[0u16, 2, 65535, 1, 4, 2001, 40001, 65535, 7, 32767]);
    let expect: [[u16; 10]; 4] = [
        [0, 2, 65535, 32768, 4, 1500, 40000, 65534, 6, 32768],
        [0, 6, 65535, 65535, 14, 6002, 65535, 65535, 24, 65535],
        [10, 12, 65535, 19672, 14, 1711, 40011, 65535, 17, 32778],
        [0, 0, 0, 65534, 0, 0, 0, 0, 0, 1],
    ];
    for ((al, be, ga), want) in scalars.iter().zip(expect) {
        let got = add_weighted(&a, *al, &b, *be, *ga).unwrap().data;
        assert_eq!(got, want, "u16 {al} {be} {ga}");
    }

    let a = row(&[-32768i16, -32767, 32767, 32767, 1, -1, 0, 3, -30000, 1234]);
    let b = row(&[-32768i16, 32766, 32767, 1, 2, -2, 0, 4, -30000, -4321]);
    let expect: [[i16; 10]; 4] = [
        [-32768, 0, 32767, 16384, 2, -2, 0, 4, -30000, -1544],
        [-32768, -2, 32767, 32767, 6, -6, 0, 14, -32768, -6174],
        [-32758, 13117, 32767, 9841, 12, 9, 10, 14, -29990, -2644],
        [0, -32768, 0, 32766, -1, 1, 0, -1, 0, 5555],
    ];
    for ((al, be, ga), want) in scalars.iter().zip(expect) {
        let got = add_weighted(&a, *al, &b, *be, *ga).unwrap().data;
        assert_eq!(got, want, "i16 {al} {be} {ga}");
    }

    // Every result stays inside the `i32` range: beyond it OpenCV returns the x86
    // conversion artefact, which purecv does not port (see the clamp test below).
    let a = row(&[
        -500000000i32,
        500000000,
        1,
        3,
        -1,
        0,
        7,
        1000000000,
        12345678,
        -5,
    ]);
    let b = row(&[
        -500000000i32,
        499999999,
        2,
        4,
        -2,
        0,
        8,
        -1000000000,
        98765432,
        10,
    ]);
    let expect: [[i32; 10]; 4] = [
        [-500000000, 500000000, 2, 4, -2, 0, 8, 0, 55555555, 2],
        [-2000000000, 1999999998, 6, 14, -6, 0, 30, 0, 222222220, 10],
        [
            -499999990, 500000010, 12, 14, 9, 10, 18, -399999990, 72839516, 16,
        ],
        [0, 1, -1, -1, 1, 0, -1, 2000000000, -86419754, -15],
    ];
    for ((al, be, ga), want) in scalars.iter().zip(expect) {
        let got = add_weighted(&a, *al, &b, *be, *ga).unwrap().data;
        assert_eq!(got, want, "i32 {al} {be} {ga}");
    }

    let a = row(&[-1e6f32, -300.5, -1.5, -0.25, 0.0, 0.1, 1.0, 2.5, 255.5, 1e6]);
    let b = row(&[1e6f32, 300.5, 0.5, 0.75, 0.0, 0.2, 3.0, 4.5, -0.5, 1e6]);
    let expect: [[f32; 10]; 4] = [
        [0.0, 0.0, -0.5, 0.25, 0.0, 0.15, 2.0, 3.5, 127.5, 1000000.0],
        [0.0, 0.0, -2.0, 1.0, 0.0, 0.6, 8.0, 14.0, 510.0, 4000000.0],
        [
            400010.5, 130.7, 10.4, 10.95, 10.5, 10.67, 12.9, 14.4, 86.8, 1000010.5,
        ],
        [
            -2000000.0, -601.0, -2.0, -1.0, 0.0, -0.1, -2.0, -2.0, 256.0, 0.0,
        ],
    ];
    for ((al, be, ga), want) in scalars.iter().zip(expect) {
        let got = add_weighted(&a, *al, &b, *be, *ga).unwrap().data;
        assert_eq!(got, want, "f32 {al} {be} {ga}");
    }

    // The `f64` vector loop of the reference wheel is fused (FMA), the loop in the source is
    // not; the elements here give the same result either way.
    let a = row(&[-1e10f64, -0.1, 0.0, 0.1, 2.0, 1e10, 2.5, 0.3]);
    let b = row(&[1e10f64, 0.7, 0.0, 0.2, 4.0, 1e10, 4.5, 0.6]);
    let expect: [[f64; 8]; 4] = [
        [
            0.0,
            0.3,
            0.0,
            0.15000000000000002,
            3.0,
            10000000000.0,
            3.5,
            0.44999999999999996,
        ],
        [
            0.0,
            1.2,
            0.0,
            0.6000000000000001,
            12.0,
            40000000000.0,
            14.0,
            1.7999999999999998,
        ],
        [
            4000000010.5,
            10.96,
            10.5,
            10.67,
            13.9,
            10000000010.5,
            14.4,
            11.01,
        ],
        [
            -20000000000.0,
            -0.7999999999999999,
            0.0,
            -0.1,
            -2.0,
            0.0,
            -2.0,
            -0.3,
        ],
    ];
    for ((al, be, ga), want) in scalars.iter().zip(expect) {
        let got = add_weighted(&a, *al, &b, *be, *ga).unwrap().data;
        assert_eq!(got, want, "f64 {al} {be} {ga}");
    }
}

// 1 x 64 `i32` and `f32` rows (`RandomState(212)`), double working type; results of
// `cv2.addWeighted`.
const WIDE_I32_A: [i32; 64] = [
    -73436904,
    -358043075,
    694451426,
    94890980,
    -854568514,
    -491184098,
    -1030760503,
    -943865874,
    271675553,
    31594649,
    227391564,
    133222590,
    -471578945,
    -209573953,
    479273715,
    521651471,
    -38915600,
    1022049071,
    953606349,
    173869296,
    86079916,
    -860688127,
    -173810422,
    472620534,
    -919917478,
    110680497,
    359264995,
    635719500,
    -559265632,
    -615837275,
    744379222,
    200186624,
    252066969,
    991741328,
    -980833584,
    375829283,
    357979110,
    275535825,
    -200020673,
    600355347,
    19087056,
    94872257,
    -465307808,
    -410700583,
    -532227625,
    -619897051,
    -885689547,
    426643548,
    233749098,
    199785782,
    -726809618,
    -874420441,
    754929234,
    360198995,
    -556131454,
    -942062773,
    -223515766,
    874773207,
    -646356313,
    -645227386,
    782061597,
    -32011626,
    839885014,
    -287634054,
];
const WIDE_I32_B: [i32; 64] = [
    -1023509097,
    856829601,
    611248431,
    790005517,
    -781748513,
    485335062,
    -1066880383,
    672591766,
    446342058,
    925717373,
    -655760414,
    -112014470,
    -35814183,
    -176276767,
    -275806647,
    609593045,
    129235507,
    -346479085,
    124789952,
    -742319455,
    972459324,
    857194805,
    510261241,
    948111581,
    -267838926,
    -240039714,
    -637849933,
    -592719829,
    825677003,
    -84715713,
    735336776,
    -74680160,
    -379874254,
    550539664,
    -732387054,
    894226982,
    -555345242,
    -625557443,
    156857926,
    282922658,
    -208156383,
    -923909211,
    710598266,
    40511355,
    -879510096,
    -31947690,
    -492817695,
    235126825,
    -155440629,
    -791807326,
    774147331,
    309165879,
    988409134,
    357218300,
    -872611832,
    509960038,
    -705155829,
    492357317,
    567539599,
    -1043197335,
    -701923154,
    308840475,
    -472397049,
    886581536,
];
const WIDE_I32_HALF: [i32; 64] = [
    -548473000,
    249393263,
    652849928,
    442448248,
    -818158514,
    -2924518,
    -1048820443,
    -135637054,
    359008806,
    478656011,
    -214184425,
    10604060,
    -253696564,
    -192925360,
    101733534,
    565622258,
    45159954,
    337784993,
    539198150,
    -284225080,
    529269620,
    -1746661,
    168225410,
    710366058,
    -593878202,
    -64679608,
    -139292469,
    21499836,
    133205686,
    -350276494,
    739857999,
    62753232,
    -63903642,
    771140496,
    -856610319,
    635028132,
    -98683066,
    -175010809,
    -21581374,
    441639002,
    -94534664,
    -414518477,
    122645229,
    -185094614,
    -705868860,
    -325922370,
    -689253621,
    330885186,
    39154234,
    -296010772,
    23668856,
    -282627281,
    871669184,
    358708648,
    -714371643,
    -216051368,
    -464335798,
    683565262,
    -39408357,
    -844212360,
    40069222,
    138414424,
    183743982,
    299473741,
];
const WIDE_F32_A: [f32; 64] = [
    -389.14825, -83.2933, 138.47934, 85.629555, 446.17273, -129.47897, 492.2394, 249.31944,
    255.37256, -261.4193, -129.8732, 231.87561, 206.8218, 284.34302, 418.02686, -477.68896,
    -498.23932, -87.592186, 338.11383, 217.64009, -232.38591, 74.21622, 318.83447, 450.0559,
    -81.67827, -151.96284, -358.0765, -146.15086, 333.51642, 374.6876, 113.986786, -194.17014,
    -259.10904, 32.699177, 368.6805, 460.56982, -456.27255, -413.16995, -392.62534, 424.3774,
    440.90036, 382.9698, 319.17496, -476.21448, 204.90169, 294.4963, 373.09448, 321.44608,
    -98.83742, 163.20839, -135.77602, 201.41241, -343.15945, -175.67505, 232.62933, 327.11255,
    345.18167, 142.3089, -128.99286, -222.69922, 27.022703, 278.81238, -45.066406, 331.69284,
];
const WIDE_F32_B: [f32; 64] = [
    -60.322784,
    8.379283,
    -184.19724,
    -425.47064,
    -370.21506,
    243.94553,
    -440.5469,
    -364.95544,
    -121.033554,
    -46.606937,
    272.98236,
    -72.15837,
    -103.837135,
    218.36176,
    -203.38655,
    -412.23642,
    -243.41948,
    -112.0272,
    110.75615,
    354.06836,
    149.58284,
    353.548,
    -417.33173,
    106.79397,
    361.49866,
    170.047,
    -106.74508,
    -191.38446,
    -226.281,
    -373.26144,
    -30.073195,
    208.5955,
    172.7493,
    -299.78857,
    113.34789,
    -204.85571,
    400.85205,
    68.608086,
    130.84265,
    -344.93304,
    92.9441,
    366.99518,
    11.14691,
    -463.3094,
    -33.02539,
    472.08807,
    -413.77463,
    208.72043,
    -487.01605,
    68.543465,
    204.15733,
    -86.05289,
    -86.32542,
    352.28656,
    383.53085,
    -418.6678,
    -477.4757,
    -336.1665,
    -428.83096,
    -466.17432,
    -23.211658,
    376.65295,
    307.31876,
    38.78808,
];
const WIDE_F32_OUT: [f32; 64] = [
    -148.47043,
    -8.622491,
    -76.894264,
    -261.6406,
    -114.79872,
    142.41818,
    -150.21101,
    -170.17297,
    2.3882797,
    -100.55065,
    162.62569,
    29.551823,
    -0.13945694,
    248.65613,
    -6.462529,
    -421.3722,
    -309.36542,
    -94.19669,
    189.46346,
    323.6399,
    45.492214,
    280.24847,
    -185.98187,
    220.27255,
    239.04558,
    83.944046,
    -171.6445,
    -167.31438,
    -47.841778,
    -138.37674,
    23.6448,
    98.265816,
    53.691795,
    -189.54225,
    200.44768,
    5.2719483,
    154.21468,
    -65.42532,
    -15.697744,
    -103.63991,
    207.83098,
    382.28757,
    114.05532,
    -456.6809,
    48.852734,
    429.31055,
    -167.2139,
    253.03812,
    -360.06247,
    107.44294,
    112.67733,
    10.686704,
    -152.87563,
    204.39807,
    348.7604,
    -184.43369,
    -220.1785,
    -182.12389,
    -328.37952,
    -382.63177,
    2.35865,
    357.80078,
    212.10321,
    137.1595,
];

/// A whole-vector-plus-tail row for the types computed in `double`. `f32` sums go through
/// `double` before being narrowed, so they differ from a pure `f32` multiply-add.
#[test]
fn add_weighted_seeded_wide_rows_match_opencv() {
    let got = add_weighted(&row(&WIDE_I32_A), 0.5, &row(&WIDE_I32_B), 0.5, 0.0).unwrap();
    assert_eq!(got.data, WIDE_I32_HALF);
    let got = add_weighted(&row(&WIDE_F32_A), 0.3, &row(&WIDE_F32_B), 0.7, 10.5).unwrap();
    assert_eq!(got.data, WIDE_F32_OUT);
}

/// OpenCV returns `INT_MIN` for a `CV_32S` sum beyond the `i32` range (the x86 conversion of an
/// out-of-range double, not a clamp), and `0` or garbage for the 8/16-bit sums of huge scalars.
/// purecv clamps: this pins that choice. The types without an OpenCV depth (`u32`, `u64`, `i64`)
/// are computed in `double` and clamped, and never panic.
#[test]
fn add_weighted_beyond_range_clamps_in_purecv() {
    let a = row(&[2_000_000_000i32, -2_000_000_000]);
    let got = add_weighted(&a, 2.0, &a, 2.0, 0.0).unwrap();
    assert_eq!(got.data, vec![i32::MAX, i32::MIN]);

    let a = row(&[4_000_000_000u32, 1]);
    let got = add_weighted(&a, 2.0, &a, 2.0, 0.0).unwrap();
    assert_eq!(got.data, vec![u32::MAX, 4]);
    let got = add_weighted(&a, 1.0, &a, -1.0, -1.0).unwrap();
    assert_eq!(got.data, vec![0u32, 0]);

    let a = row(&[u64::MAX, 2]);
    let got = add_weighted(&a, 1.0, &a, 1.0, 0.0).unwrap();
    assert_eq!(got.data, vec![u64::MAX, 4]);

    let a = row(&[i64::MAX, i64::MIN, 3]);
    let got = add_weighted(&a, 1.0, &a, 1.0, 0.0).unwrap();
    assert_eq!(got.data, vec![i64::MAX, i64::MIN, 6]);

    // NaN weights have no integer value: 0, like `saturate_cast`.
    let a = row(&[7u8, 200]);
    let got = add_weighted(&a, f64::NAN, &a, 1.0, 0.0).unwrap();
    assert_eq!(got.data, vec![0u8, 0]);
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

/// `saturate_cast<uchar>(|x*alpha + beta|)` rounds half to even: 0.5 -> 0, 2.5 -> 2,
/// -2.5 -> 2 (1.5 -> 2, -1.5 -> 2); and the `i16` value 1 times 0.5 = 0.5 -> 0.
#[test]
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

/// The other source depths. `convert_scale.simd.hpp:220-226` runs every depth through
/// `cvtabs_32f`: the source is converted to `float`, then `|x*a + b|` is rounded half to even
/// and clamped to `u8`.
///
/// Fixture: `scripts/opencv_ref/core_convert_arith.py`, block
/// `convert_scale_abs_other_depths`. The wheel's vector loop fuses the multiply-add and the
/// generic scalar tail does not; the scalars (1, 0.5, -1, 2 with betas 0, 0.25, -3) keep
/// `x*a` exact in `f32`, so the two agree and so does purecv's unfused product. Inputs whose
/// scaled value passes 2^31 are left out (see
/// `convert_scale_abs_large_values_clamp_to_255`).
#[test]
fn convert_scale_abs_other_depths_match_opencv() {
    // (alpha, beta) in the order of the expected rows: 1_0, 0.5_0, 0.5_0.25, -1_-3, 2_0
    let scalars = [
        (1.0, 0.0),
        (0.5, 0.0),
        (0.5, 0.25),
        (-1.0, -3.0),
        (2.0, 0.0),
    ];

    let src = row(&[-128i8, -127, -3, -2, -1, 0, 1, 2, 3, 127]);
    let expect: [[u8; 10]; 5] = [
        [128, 127, 3, 2, 1, 0, 1, 2, 3, 127],
        [64, 64, 2, 1, 0, 0, 0, 1, 2, 64],
        [64, 63, 1, 1, 0, 0, 1, 1, 2, 64],
        [125, 124, 0, 1, 2, 3, 4, 5, 6, 130],
        [255, 254, 6, 4, 2, 0, 2, 4, 6, 254],
    ];
    for ((al, be), want) in scalars.iter().zip(expect) {
        let got = convert_scale_abs(&src, *al, *be).unwrap().data;
        assert_eq!(got, want, "i8 {al} {be}");
    }

    let src = row(&[0u16, 1, 2, 3, 5, 255, 256, 511, 1000, 65535]);
    let expect: [[u8; 10]; 5] = [
        [0, 1, 2, 3, 5, 255, 255, 255, 255, 255],
        [0, 0, 1, 2, 2, 128, 128, 255, 255, 255],
        [0, 1, 1, 2, 3, 128, 128, 255, 255, 255],
        [3, 4, 5, 6, 8, 255, 255, 255, 255, 255],
        [0, 2, 4, 6, 10, 255, 255, 255, 255, 255],
    ];
    for ((al, be), want) in scalars.iter().zip(expect) {
        let got = convert_scale_abs(&src, *al, *be).unwrap().data;
        assert_eq!(got, want, "u16 {al} {be}");
    }

    let src = row(&[-32768i16, -32767, -300, -3, -1, 0, 1, 3, 255, 32767]);
    let expect: [[u8; 10]; 5] = [
        [255, 255, 255, 3, 1, 0, 1, 3, 255, 255],
        [255, 255, 150, 2, 0, 0, 0, 2, 128, 255],
        [255, 255, 150, 1, 0, 0, 1, 2, 128, 255],
        [255, 255, 255, 0, 2, 3, 4, 6, 255, 255],
        [255, 255, 255, 6, 2, 0, 2, 6, 255, 255],
    ];
    for ((al, be), want) in scalars.iter().zip(expect) {
        let got = convert_scale_abs(&src, *al, *be).unwrap().data;
        assert_eq!(got, want, "i16 {al} {be}");
    }

    let src = row(&[
        -536870912i32,
        -16777217,
        -3,
        -1,
        0,
        1,
        3,
        255,
        16777217,
        536870912,
    ]);
    let expect: [[u8; 10]; 5] = [
        [255, 255, 3, 1, 0, 1, 3, 255, 255, 255],
        [255, 255, 2, 0, 0, 0, 2, 128, 255, 255],
        [255, 255, 1, 0, 0, 1, 2, 128, 255, 255],
        [255, 255, 0, 2, 3, 4, 6, 255, 255, 255],
        [255, 255, 6, 2, 0, 2, 6, 255, 255, 255],
    ];
    for ((al, be), want) in scalars.iter().zip(expect) {
        let got = convert_scale_abs(&src, *al, *be).unwrap().data;
        assert_eq!(got, want, "i32 {al} {be}");
    }

    let src = row(&[-3.5f32, -2.5, -0.5, 0.5, 1.5, 2.5, 3.5, 254.5, 255.5, 300.0]);
    let expect: [[u8; 10]; 5] = [
        [4, 2, 0, 0, 2, 2, 4, 254, 255, 255],
        [2, 1, 0, 0, 1, 1, 2, 127, 128, 150],
        [2, 1, 0, 0, 1, 2, 2, 128, 128, 150],
        [0, 0, 2, 4, 4, 6, 6, 255, 255, 255],
        [7, 5, 1, 1, 3, 5, 7, 255, 255, 255],
    ];
    for ((al, be), want) in scalars.iter().zip(expect) {
        let got = convert_scale_abs(&src, *al, *be).unwrap().data;
        assert_eq!(got, want, "f32 {al} {be}");
    }

    let src = row(&[
        -2.5f64,
        -0.5,
        0.5,
        1.5,
        2.5,
        3.5,
        -1.5,
        254.5,
        300.0,
        f64::NAN,
    ]);
    let expect: [[u8; 10]; 5] = [
        [2, 0, 0, 2, 2, 4, 2, 254, 255, 0],
        [1, 0, 0, 1, 1, 2, 1, 127, 150, 0],
        [1, 0, 0, 1, 2, 2, 0, 128, 150, 0],
        [0, 2, 4, 4, 6, 6, 2, 255, 255, 0],
        [5, 1, 1, 3, 5, 7, 3, 255, 255, 0],
    ];
    for ((al, be), want) in scalars.iter().zip(expect) {
        let got = convert_scale_abs(&src, *al, *be).unwrap().data;
        assert_eq!(got, want, "f64 {al} {be}");
    }
}

/// 1 x 64 rows of every half step from -10 to 21.5, so ties of both parities land in every
/// SIMD lane and the row fills whole vectors. The expected values are the
/// `cv2.convertScaleAbs` results of `scripts/opencv_ref/core_convert_arith.py`, block
/// `convert_scale_abs_ties_wide`; the scalar path and the `simd` kernels must give the same.
#[test]
fn convert_scale_abs_ties_across_simd_lanes() {
    let half: Vec<f64> = (0..64).map(|i| (i as f64 - 20.0) * 0.5).collect();
    let alpha_1: [u8; 64] = [
        10, 10, 9, 8, 8, 8, 7, 6, 6, 6, 5, 4, 4, 4, 3, 2, 2, 2, 1, 0, 0, 0, 1, 2, 2, 2, 3, 4, 4, 4,
        5, 6, 6, 6, 7, 8, 8, 8, 9, 10, 10, 10, 11, 12, 12, 12, 13, 14, 14, 14, 15, 16, 16, 16, 17,
        18, 18, 18, 19, 20, 20, 20, 21, 22,
    ];
    let alpha_half: [u8; 64] = [
        5, 4, 4, 4, 4, 4, 3, 3, 3, 2, 2, 2, 2, 2, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 2,
        3, 3, 3, 4, 4, 4, 4, 4, 5, 5, 5, 6, 6, 6, 6, 6, 7, 7, 7, 8, 8, 8, 8, 8, 9, 9, 9, 10, 10,
        10, 10, 10, 11, 11,
    ];

    let f32s: Vec<f32> = half.iter().map(|&v| v as f32).collect();
    let src = row(&f32s);
    assert_eq!(convert_scale_abs(&src, 1.0, 0.0).unwrap().data, alpha_1);
    assert_eq!(convert_scale_abs(&src, 0.5, 0.25).unwrap().data, alpha_half);

    let src = row(&half);
    assert_eq!(convert_scale_abs(&src, 1.0, 0.0).unwrap().data, alpha_1);
    assert_eq!(convert_scale_abs(&src, 0.5, 0.25).unwrap().data, alpha_half);

    // 0.5000000001 and 254.50000001 are not `f32` values. The vector loop of `cvtabs_32f`
    // narrows them to `f32` first: 0.5 -> 0 and 254.5 -> 254, both ties. (The scalar tail,
    // used only for rows narrower than one vector, multiplies in `double` and gives 1 and
    // 255; purecv follows the vector loop.) Block `convert_scale_abs_ties_wide`, key
    // `f64_narrow_1_0`.
    let narrow: Vec<f64> = [0.5000000001, 254.50000001].repeat(32);
    let want: Vec<u8> = [0u8, 254].repeat(32);
    assert_eq!(
        convert_scale_abs(&row(&narrow), 1.0, 0.0).unwrap().data,
        want
    );
}

/// purecv's own behaviour, not OpenCV's: a scaled value beyond the `i32` range clamps to 255.
/// OpenCV 4.10.0 on x86 converts it through `int`, gets `INT_MIN` and returns 0
/// (`cv2.convertScaleAbs` of `1e10` is 0), an artefact that is not ported.
#[test]
fn convert_scale_abs_large_values_clamp_to_255() {
    let src = row(&[1e10f32, -1e10, 3e9]);
    assert_eq!(
        convert_scale_abs(&src, 1.0, 0.0).unwrap().data,
        vec![255u8, 255, 255]
    );
    let src = row(&[1e10f64, -1e10, 3e9]);
    assert_eq!(
        convert_scale_abs(&src, 1.0, 0.0).unwrap().data,
        vec![255u8, 255, 255]
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

/// Integer outputs go through `convertTo` (saturate_cast, ties to even): u8 MINMAX 0..255 of
/// [10, 20, 30, 250] is [0, 11, 21, 255], and 0..100 is [0, 4, 8, 100].
#[test]
fn normalize_u8_minmax_rounds_like_convert_to() {
    let s8 = row(&[10u8, 20, 30, 250]);
    let mut d8 = Matrix::<u8>::new(1, 4, 1);
    normalize(&s8, &mut d8, 0.0, 255.0, NormTypes::MinMax, -1, None).unwrap();
    assert_eq!(d8.data, vec![0u8, 11, 21, 255]);
    normalize(&s8, &mut d8, 0.0, 100.0, NormTypes::MinMax, -1, None).unwrap();
    assert_eq!(d8.data, vec![0u8, 4, 8, 100]);
}

/// MINMAX sorts alpha and beta (`dmin = MIN(a, b)`), so (255, 0) equals (0, 255).
#[test]
fn normalize_minmax_swapped_bounds_match_sorted_bounds() {
    let src = row(&[10.0f32, 20.0, 30.0, 250.0]);
    let mut dst = Matrix::<f32>::new(1, 4, 1);
    normalize(&src, &mut dst, 255.0, 0.0, NormTypes::MinMax, -1, None).unwrap();
    assert_eq!(dst.data, vec![0.0f32, 10.625, 21.25, 255.0]);
}

/// `rtype` goes through `CV_MAT_DEPTH`, so the channel bits of a full type code do not matter:
/// `core_stats.py`, `normalize`, key `f32_dtype_CV_32FC3` keeps `f32` and gives the plain
/// result. `CV_8UC3` has another depth, which a `Matrix<f32>` destination cannot hold, so it is
/// rejected here and `normalize_to` is the way to change the type.
#[test]
fn normalize_dtype_compares_only_the_depth() {
    let src = row(&[10.0f32, 20.0, 30.0, 250.0]);
    let mut dst = Matrix::<f32>::new(1, 4, 1);
    normalize(
        &src,
        &mut dst,
        0.0,
        255.0,
        NormTypes::MinMax,
        CV_32FC3.to_int(),
        None,
    )
    .unwrap();
    assert_eq!(dst.data, vec![0.0f32, 10.625, 21.25, 255.0]);
    assert!(normalize(
        &src,
        &mut dst,
        0.0,
        255.0,
        NormTypes::MinMax,
        CV_8UC3.to_int(),
        None
    )
    .is_err());
}

/// `normalize_to` carries the destination type. Values from `core_stats.py`, block
/// `normalize`, keys `u8_minmax_0_1_to_f32`, `u8_minmax_0_255_to_f32` and
/// `u8_minmax_swapped_to_f32`. A float destination uses single-precision scale and shift, so
/// the 0..1 range ends at 1.0000001. The literals are the shortest f32 forms of the measured doubles.
#[test]
fn normalize_to_u8_to_f32_matches_opencv() {
    let s8 = row(&[10u8, 20, 30, 250]);
    let mut d = Matrix::<f32>::new(1, 4, 1);
    normalize_to(&s8, &mut d, 0.0, 1.0, NormTypes::MinMax, None).unwrap();
    assert_eq!(
        d.data,
        vec![0.0f32, 0.041_666_668, 0.083_333_33, 1.000_000_1]
    );
    normalize_to(&s8, &mut d, 0.0, 255.0, NormTypes::MinMax, None).unwrap();
    assert_eq!(d.data, vec![0.0f32, 10.625, 21.25, 255.0]);
    normalize_to(&s8, &mut d, 1.0, 0.0, NormTypes::MinMax, None).unwrap();
    assert_eq!(
        d.data,
        vec![0.0f32, 0.041_666_668, 0.083_333_33, 1.000_000_1]
    );
    // f32 source to u8 destination (`f32_minmax_to_u8`)
    let s32 = row(&[10.0f32, 20.0, 30.0, 250.0]);
    let mut d8 = Matrix::<u8>::new(1, 4, 1);
    normalize_to(&s32, &mut d8, 0.0, 255.0, NormTypes::MinMax, None).unwrap();
    assert_eq!(d8.data, vec![0u8, 11, 21, 255]);
    // a destination type without an OpenCV depth saturates instead of panicking
    let mut d64 = Matrix::<u64>::new(1, 4, 1);
    normalize_to(&s32, &mut d64, 0.0, 255.0, NormTypes::MinMax, None).unwrap();
    assert_eq!(d64.data, vec![0u64, 11, 21, 255]);
}

/// A `dtype` other than -1 or the depth of `T` is an error; `normalize_to` is the
/// way to change the destination type.
#[test]
fn normalize_rejects_mismatched_dtype() {
    let src = row(&[10.0f32, 20.0, 30.0, 250.0]);
    let mut dst = Matrix::<f32>::new(1, 4, 1);
    // CV_8U = 0
    assert!(normalize(&src, &mut dst, 0.0, 255.0, NormTypes::MinMax, 0, None).is_err());
    // CV_32F = 5 is the depth of f32
    normalize(&src, &mut dst, 0.0, 255.0, NormTypes::MinMax, 5, None).unwrap();
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

/// `reduce` with `REDUCE_AVG` on `u8` accumulates in 32S and converts back with
/// `saturate_cast` (`u8_avg_dim0`, `u8_avg_dim1` in `core_stats.py`, block `reduce`): the
/// averages 1.5 and 67.7 become 2 and 68, and 2.5 becomes 2 (ties to even).
#[test]
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

/// `reduce` refuses `REDUCE_SUM` from 8U into 8U (`u8_sum_dim0_default_dtype` in
/// `core_stats.py`, block `reduce`: "Unsupported combination of input and output array
/// formats"); a wider destination goes through `reduce_to`.
#[test]
fn reduce_sum_u8_into_u8_is_rejected() {
    let r8 = Matrix::from_vec(2, 3, 1, vec![1u8, 2, 200, 2, 3, 100]);
    assert!(reduce(&r8, 0, ReduceTypes::Sum).is_err());
}

/// `reduce_to::<u8, i32>` keeps the sum that overflows a byte (`u8_sum_dim0_i32`,
/// `u8_sum_dim1_i32`, `u8_3ch_sum_dim0_i32` in `core_stats.py`, block `reduce`).
#[test]
fn reduce_to_sum_u8_to_i32_matches_opencv() {
    let r8 = Matrix::from_vec(2, 3, 1, vec![1u8, 2, 200, 2, 3, 100]);
    let d0: Matrix<i32> = reduce_to(&r8, 0, ReduceTypes::Sum).unwrap();
    assert_eq!((d0.rows, d0.cols), (1, 3));
    assert_eq!(d0.data, vec![3, 5, 300]);
    let d1: Matrix<i32> = reduce_to(&r8, 1, ReduceTypes::Sum).unwrap();
    assert_eq!((d1.rows, d1.cols), (2, 1));
    assert_eq!(d1.data, vec![203, 105]);
    let c3 = Matrix::from_vec(
        2,
        3,
        3,
        vec![
            1u8, 2, 3, 2, 3, 4, 200, 201, 202, 2, 3, 4, 3, 4, 5, 100, 101, 102,
        ],
    );
    let d3: Matrix<i32> = reduce_to(&c3, 0, ReduceTypes::Sum).unwrap();
    assert_eq!(d3.data, vec![3, 5, 7, 5, 7, 9, 300, 302, 304]);
}

/// `REDUCE_SUM2` sums the squares (`u8_sum2_dim0_i32`, `u8_sum2_dim1_f64`,
/// `u8_sum2_dim0_default_dtype`, `f32_sum2_dim0`, `f32_sum2_dim1`, `i16_sum2_dim1_f64` in
/// `core_stats.py`, block `reduce`); like the other sums it needs a wide enough destination.
#[test]
fn reduce_sum2_matches_opencv() {
    let r8 = Matrix::from_vec(2, 3, 1, vec![1u8, 2, 200, 2, 3, 100]);
    let i: Matrix<i32> = reduce_to(&r8, 0, ReduceTypes::Sum2).unwrap();
    assert_eq!(i.data, vec![5, 13, 50000]);
    let d: Matrix<f64> = reduce_to(&r8, 1, ReduceTypes::Sum2).unwrap();
    assert_eq!(d.data, vec![40005.0, 10013.0]);
    assert!(reduce(&r8, 0, ReduceTypes::Sum2).is_err());

    let rf = Matrix::from_vec(2, 3, 1, vec![1.0f32, 2.0, 200.0, 2.0, 3.0, 100.0]);
    assert_eq!(
        reduce(&rf, 0, ReduceTypes::Sum2).unwrap().data,
        vec![5.0f32, 13.0, 50000.0]
    );
    assert_eq!(
        reduce(&rf, 1, ReduceTypes::Sum2).unwrap().data,
        vec![40005.0f32, 10013.0]
    );

    let s16 = Matrix::from_vec(2, 3, 1, vec![-5i16, 7, 30000, 4, -9, 30000]);
    let i16d: Matrix<f64> = reduce_to(&s16, 1, ReduceTypes::Sum2).unwrap();
    assert_eq!(i16d.data, vec![900000074.0, 900000097.0]);
}

/// The accepted (source, destination) depth pairs are those of `cv::reduce`: the sums
/// reach 32S/32F/64F from 8U, 32F/64F from 16U/16S, and 32F/64F from 32F/64F; max and min
/// keep the depth and exist for 8U, 16U, 16S, 32F and 64F only. Everything else is
/// rejected (`u16_sum_dim0_u16`, `u16_sum_dim0_i32`, `i16_sum_dim0_i16`, `i32_max_dim0`,
/// `i32_sum_dim0_i32`, `i32_sum_dim0_f64`, `f32_sum_dim0_u8`, `f32_max_dim0_f64`,
/// `f64_sum_dim0_f32`, `f32_avg_dim0_u8`, `u16_avg_dim0_u16` in `core_stats.py`, block
/// `reduce`; `u16_sum_dim0_f32`, `u16_max_dim0`, `i16_min_dim0`, `f64_max_dim1` are the
/// accepted neighbours).
#[test]
fn reduce_to_rejects_unsupported_depth_pair() {
    let r16 = Matrix::from_vec(2, 3, 1, vec![1u16, 2, 60000, 2, 3, 60000]);
    let s16 = Matrix::from_vec(2, 3, 1, vec![-5i16, 7, 30000, 4, -9, 30000]);
    let r32 = Matrix::from_vec(2, 3, 1, vec![1i32, 2, 300, 2, 3, 100]);
    let rf = Matrix::from_vec(2, 3, 1, vec![1.0f32, 2.0, 200.0, 2.0, 3.0, 100.0]);
    let rd = Matrix::from_vec(2, 3, 1, vec![1.0f64, 2.0, 200.0, 2.0, 3.0, 100.0]);
    for dim in [0, 1] {
        assert!(reduce_to::<_, u16>(&r16, dim, ReduceTypes::Sum).is_err());
        assert!(reduce_to::<_, i32>(&r16, dim, ReduceTypes::Sum).is_err());
        assert!(reduce_to::<_, i16>(&s16, dim, ReduceTypes::Sum).is_err());
        assert!(reduce_to::<_, i32>(&r32, dim, ReduceTypes::Max).is_err());
        assert!(reduce_to::<_, i32>(&r32, dim, ReduceTypes::Sum).is_err());
        assert!(reduce_to::<_, f64>(&r32, dim, ReduceTypes::Sum).is_err());
        assert!(reduce_to::<_, u8>(&rf, dim, ReduceTypes::Sum).is_err());
        assert!(reduce_to::<_, f64>(&rf, dim, ReduceTypes::Max).is_err());
        assert!(reduce_to::<_, f32>(&rd, dim, ReduceTypes::Sum).is_err());
        assert!(reduce_to::<_, u8>(&rf, dim, ReduceTypes::Avg).is_err());
        // AVG into a destination narrower than 32S is promoted to 32S first, which 16U
        // has no sum for.
        assert!(reduce_to::<_, u16>(&r16, dim, ReduceTypes::Avg).is_err());
    }
    let w: Matrix<f32> = reduce_to(&r16, 0, ReduceTypes::Sum).unwrap();
    assert_eq!(w.data, vec![3.0, 5.0, 120000.0]);
    assert_eq!(
        reduce(&r16, 0, ReduceTypes::Max).unwrap().data,
        vec![2u16, 3, 60000]
    );
    assert_eq!(
        reduce(&s16, 0, ReduceTypes::Min).unwrap().data,
        vec![-5i16, -9, 30000]
    );
    assert_eq!(
        reduce(&rd, 1, ReduceTypes::Max).unwrap().data,
        vec![200.0f64, 100.0]
    );
}

/// `REDUCE_AVG` sums in the (promoted) destination depth and converts once with the scale
/// `1/n` (`u8_avg_dim0_i32`, `u8_avg_dim1_f32`, `u8_avg_dim0_u16`, `u8_avg_dim0_i8`,
/// `f32_avg_dim1_f64` in `core_stats.py`, block `reduce`): a narrow destination is
/// promoted to 32S, so 150 saturates in `i8` rather than wrapping.
#[test]
fn reduce_to_avg_converts_once_like_opencv() {
    let r8 = Matrix::from_vec(2, 3, 1, vec![1u8, 2, 200, 2, 3, 100]);
    let i: Matrix<i32> = reduce_to(&r8, 0, ReduceTypes::Avg).unwrap();
    assert_eq!(i.data, vec![2, 2, 150]);
    let f: Matrix<f32> = reduce_to(&r8, 1, ReduceTypes::Avg).unwrap();
    assert_eq!(f.data, vec![67.666_67_f32, 35.0]);
    let u: Matrix<u16> = reduce_to(&r8, 0, ReduceTypes::Avg).unwrap();
    assert_eq!(u.data, vec![2, 2, 150]);
    let s: Matrix<i8> = reduce_to(&r8, 0, ReduceTypes::Avg).unwrap();
    assert_eq!(s.data, vec![2, 2, 127]);
    let rf = Matrix::from_vec(2, 3, 1, vec![1.0f32, 2.0, 200.0, 2.0, 3.0, 100.0]);
    let d: Matrix<f64> = reduce_to(&rf, 1, ReduceTypes::Avg).unwrap();
    assert_eq!(d.data, vec![67.66666666666666, 35.0]);
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

/// Integer outputs go through `saturate_cast`, round half to even: (10+20+33)*0.3 = 18.9 -> 19
/// and 229.5 -> 230 (`core_misc.py`, `transform`, keys `u8_m1x3_0.3` and `u8_m2x4_affine`).
#[test]
fn transform_u8_rounds_half_to_even() {
    let src = Matrix::from_vec(1, 3, 3, vec![10u8, 20, 33, 255, 255, 255, 1, 1, 1]);
    let m = Matrix::from_vec(1, 3, 1, vec![0.3f64, 0.3, 0.3]);
    let mut dst = Matrix::<u8>::new(0, 0, 0);
    transform(&src, &mut dst, &m).unwrap();
    assert_eq!(dst.data, vec![19u8, 230, 1]);
    // affine 2x4: the 1.5 of the third pixel rounds to 2 (ties to even), 10.5 to 10, 32.5 to 32.
    let affine = Matrix::from_vec(2, 4, 1, vec![1.0f64, 0.0, 0.0, 0.5, 0.0, 0.0, 1.0, -0.5]);
    transform(&src, &mut dst, &affine).unwrap();
    assert_eq!(dst.data, vec![10u8, 32, 255, 254, 2, 0]);
}

/// `core_misc.py`, `transform_working_type`, key `oor_u8`: results below 0 and above 255 clamp
/// (3 -> 4 channels, so the generic kernel runs). Source `[[100, 200, 50], [255, 255, 255]]`,
/// rows of `m`: `[2 0 0 | 0]`, `[0 1 0 | -300]`, `[0 1 0 | 100]`, `[-1 0 0 | 0]`.
#[test]
fn transform_u8_saturates_out_of_range() {
    let src = Matrix::from_vec(1, 2, 3, vec![100u8, 200, 50, 255, 255, 255]);
    #[rustfmt::skip]
    let m = Matrix::from_vec(4, 4, 1, vec![
        2.0f64, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, -300.0,
        0.0, 1.0, 0.0, 100.0,
        -1.0, 0.0, 0.0, 0.0,
    ]);
    let mut dst = Matrix::<u8>::new(0, 0, 0);
    transform(&src, &mut dst, &m).unwrap();
    assert_eq!(dst.channels, 4);
    assert_eq!(dst.data, vec![200u8, 0, 255, 0, 255, 0, 255, 0]);
}

/// `core_misc.py`, `transform_working_type` (`RandomState(220)`). Every matrix entry is a
/// multiple of 1/8 (1/4 for the `i32` and bias values), so each product and sum is exact in
/// float and OpenCV's SIMD (FMA) and generic scalar kernels agree; ties abound and decide the
/// result. The u8 3 -> 3 shape is left out on purpose: its SIMD kernel is fixed point and
/// rounds ties up, which the generic path (ported here) does not.
#[test]
fn transform_seeded_fixture_matches_opencv() {
    let mut dst8 = Matrix::<u8>::new(0, 0, 0);

    // u8 4 -> 4, key `u8_4to4`
    let s4 = Matrix::from_vec(
        1,
        8,
        4,
        vec![
            115u8, 220, 15, 180, 120, 2, 112, 114, 27, 107, 182, 157, 142, 204, 47, 9, 111, 0, 40,
            58, 18, 105, 73, 181, 245, 162, 0, 42, 181, 255, 32, 170,
        ],
    );
    #[rustfmt::skip]
    let m4 = Matrix::from_vec(4, 4, 1, vec![
        0.25f64, 0.0, 0.25, -0.25,
        0.0, -0.25, 0.5, 0.625,
        0.125, 0.125, 0.625, -0.25,
        0.375, 0.375, -0.125, -0.125,
    ]);
    transform(&s4, &mut dst8, &m4).unwrap();
    assert_eq!(
        dst8.data,
        vec![
            0u8, 65, 6, 101, 30, 127, 57, 18, 13, 162, 91, 8, 45, 0, 70, 123, 23, 56, 24, 29, 0,
            123, 16, 14, 51, 0, 40, 147, 11, 58, 32, 138
        ]
    );

    // u8 3 -> 2 affine (keys `u8_3to2_affine`) and 3 -> 1 (key `u8_3to1`)
    let s3 = Matrix::from_vec(
        1,
        8,
        3,
        vec![
            94u8, 248, 30, 126, 8, 157, 224, 24, 111, 156, 203, 255, 34, 164, 160, 14, 176, 245,
            127, 15, 20, 194, 152, 186,
        ],
    );
    let m3to2 = Matrix::from_vec(
        2,
        4,
        1,
        vec![0.375f64, 0.5, -0.125, 0.0, 0.125, 0.5, -0.25, 0.75],
    );
    transform(&s3, &mut dst8, &m3to2).unwrap();
    assert_eq!(
        dst8.data,
        vec![156u8, 129, 32, 0, 82, 13, 128, 58, 75, 47, 63, 29, 53, 19, 126, 54]
    );
    let m3to1 = Matrix::from_vec(1, 3, 1, vec![0.25f64, 0.375, 0.375]);
    transform(&s3, &mut dst8, &m3to1).unwrap();
    assert_eq!(dst8.data, vec![128u8, 93, 107, 211, 130, 161, 45, 175]);

    // i16 3 -> 2 affine, same matrix, key `i16_3to2_affine`: float working type
    let s16 = Matrix::from_vec(
        1,
        8,
        3,
        vec![
            -2049i16, 2682, 1172, 2070, -2970, 1732, 1812, -2546, -839, 2147, -1486, 1402, 2430,
            1377, 2967, 740, -310, 717, -1970, -112, 2017, -1486, -39, -2078,
        ],
    );
    let mut dst16 = Matrix::<i16>::new(0, 0, 0);
    transform(&s16, &mut dst16, &m3to2).unwrap();
    assert_eq!(
        dst16.data,
        vec![
            426i16, 793, -925, -1658, -489, -836, -113, -824, 1229, 251, 33, -241, -1047, -806,
            -317, 315
        ]
    );

    // i32 3 -> 2 affine, key `i32_3to2_affine`: the matrix and the sum are f64, so the first
    // pixel keeps 16777217 (an f32 working type would give 16777216) and the second result
    // 25165819.5 rounds to the even 25165820.
    let s32 = Matrix::from_vec(
        1,
        6,
        3,
        vec![
            16777217i32,
            3,
            -16777217,
            -80624759,
            103227088,
            39522842,
            133865647,
            -57659356,
            20879032,
            -177023738,
            138849448,
            117637714,
            117414239,
            237643101,
            165583266,
            -221647597,
            235278206,
            -250759144,
        ],
    );
    let m32 = Matrix::from_vec(
        2,
        4,
        1,
        vec![-0.5f64, 1.5, -2.0, -0.75, 0.25, -1.75, -1.25, -0.75],
    );
    let mut dst32 = Matrix::<i32>::new(0, 0, 0);
    transform(&s32, &mut dst32, &m32).unwrap();
    assert_eq!(
        dst32.data,
        vec![
            25165829i32,
            25165820,
            116107327,
            -250207147,
            -195179922,
            108271494,
            61510612,
            -434289612,
            -33409001,
            -593500950,
            965259395,
            -153699830
        ]
    );
}

/// `core_misc.py`, `transform_working_type`, keys `u8_1ch_half`, `u8_1ch_half_plus_quarter`
/// (`scn == dcn == 1` is `convertTo(alpha, beta)`) and `u8_diag_3to3` (a diagonal `m` runs the
/// `diagtransform` kernel).
#[test]
fn transform_single_channel_and_diagonal_match_opencv() {
    let one = Matrix::from_vec(1, 7, 1, vec![0u8, 1, 2, 3, 4, 5, 200]);
    let mut dst = Matrix::<u8>::new(0, 0, 0);
    transform(&one, &mut dst, &Matrix::from_vec(1, 1, 1, vec![0.5f64])).unwrap();
    assert_eq!(dst.data, vec![0u8, 0, 1, 2, 2, 2, 100]);
    transform(
        &one,
        &mut dst,
        &Matrix::from_vec(1, 2, 1, vec![0.5f64, 0.25]),
    )
    .unwrap();
    assert_eq!(dst.data, vec![0u8, 1, 1, 2, 2, 3, 100]);

    let src = Matrix::from_vec(
        1,
        4,
        3,
        vec![94u8, 248, 30, 126, 8, 157, 224, 24, 111, 156, 203, 255],
    );
    #[rustfmt::skip]
    let diag = Matrix::from_vec(3, 4, 1, vec![
        0.5f64, 0.0, 0.0, 0.5,
        0.0, 0.25, 0.0, -0.5,
        0.0, 0.0, 1.5, 0.25,
    ]);
    transform(&src, &mut dst, &diag).unwrap();
    assert_eq!(
        dst.data,
        vec![48u8, 62, 45, 64, 2, 236, 112, 6, 167, 78, 50, 255]
    );
}

/// Types with no OpenCV depth (`u32`, `u64`, `i64`) use the `f64` working type and must not
/// panic; they round half to even and clamp like the others. There is nothing to measure for
/// them, so these expected values are derived by hand from those rules.
#[test]
fn transform_types_without_opencv_depth_use_f64() {
    let m = Matrix::from_vec(1, 3, 1, vec![0.5f64, 0.5, 0.5]);
    let mut d32 = Matrix::<u32>::new(0, 0, 0);
    transform(
        &Matrix::from_vec(1, 2, 3, vec![1u32, 2, 0, 5, 5, 6]),
        &mut d32,
        &m,
    )
    .unwrap();
    assert_eq!(d32.data, vec![2u32, 8]); // 1.5 -> 2, 8
    let mut d64 = Matrix::<i64>::new(0, 0, 0);
    let neg = Matrix::from_vec(1, 1, 3, vec![-5i64, -5, -1]);
    transform(&neg, &mut d64, &m).unwrap();
    assert_eq!(d64.data, vec![-6i64]); // -5.5 -> -6
    let mut du = Matrix::<u64>::new(0, 0, 0);
    transform(&Matrix::from_vec(1, 1, 3, vec![u64::MAX; 3]), &mut du, &m).unwrap();
    assert_eq!(du.data, vec![u64::MAX]);
}

#[test]
fn transform_and_perspective_transform_match_opencv() {
    let src = Matrix::from_vec(1, 3, 3, vec![10u8, 20, 33, 255, 255, 255, 1, 1, 1]);
    let m = Matrix::from_vec(2, 4, 1, vec![1.0f64, 0.0, 0.0, 0.5, 0.0, 0.0, 1.0, -0.5]);
    let mut dst = Matrix::<u8>::new(0, 0, 0);
    transform(&src, &mut dst, &m).unwrap();
    // OpenCV: [[10, 32], [255, 254], [2, 0]]: 10.5 -> 10 and 32.5 -> 32 (ties to even), 255.5
    // saturates, 1.5 -> 2, 0.5 -> 0.
    assert_eq!(dst.channels, 2);
    assert_eq!(dst.data, vec![10u8, 32, 255, 254, 2, 0]);
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

/// OpenCV's `theRNG()` is a multiply-with-carry generator (`RNG::next`); purecv's default
/// thread-local generator is a port of it, so the same seed gives the same sequence.
#[test]
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
/// wide distribution holds many 255s and 0s; purecv saturates the same way.
#[test]
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

/// `core_rng.py`, seeds 0, 42 and 12345: every depth of `randu`, per-channel bounds,
/// `randn` (f32 and u8) and `randShuffle` give OpenCV's values.
#[test]
fn rng_sequences_match_cv_rng_for_three_seeds() {
    #[allow(clippy::type_complexity)]
    let cases: [(u64, [u8; 8], [u8; 8], [i32; 8], [f32; 8], [f64; 8]); 3] = [
        (
            0,
            [246, 156, 192, 7, 165, 166, 2, 179],
            [6, 7, 9, 9, 7, 0, 6, 3],
            [1, 2, 4, 4, 2, -5, 1, -2],
            [
                0.530_282_8,
                0.199_259_19,
                0.401_059_45,
                0.814_385_06,
                0.437_132_98,
                0.248_789_7,
                0.773_105,
                0.762_093_7,
            ],
            [
                0.5302827933267031,
                0.19925920037305117,
                0.4010594430561372,
                0.8143850783523574,
                0.43713297911567256,
                0.24878970880730866,
                0.7731050566598108,
                0.7620937287027096,
            ],
        ),
        (
            42,
            [164, 63, 102, 186, 144, 232, 246, 91],
            [0, 8, 9, 1, 4, 0, 3, 2],
            [-5, 3, 4, -4, -1, -5, -2, -3],
            [
                0.228_122_68,
                0.859_236_24,
                0.014_739_662,
                0.810_566_37,
                0.450_981_26,
                0.001_813_471_3,
                0.531_401_1,
                0.523_464_5,
            ],
            [
                0.2281226897612214,
                0.859236274078971,
                0.014739665516735445,
                0.8105663744497933,
                0.45098125149988605,
                0.0018134813108907077,
                0.5314011013802646,
                0.5234644957888989,
            ],
        ),
        (
            12345,
            [58, 237, 174, 40, 7, 225, 216, 205],
            [4, 3, 3, 1, 9, 4, 2, 5],
            [-1, -2, -2, -4, 4, -1, -3, 0],
            [
                0.658_919_16,
                0.304_090_56,
                0.725_266,
                0.641_473_65,
                0.734_846_4,
                0.925_891_1,
                0.443_966_6,
                0.896_885_75,
            ],
            [
                0.6589191691018648,
                0.3040905611482697,
                0.7252659722649819,
                0.6414736339865356,
                0.7348464236448494,
                0.9258911158488445,
                0.44396658530761346,
                0.8968857265811834,
            ],
        ),
    ];
    for (seed, u8_256, u8_10, i32_m5_5, f32_01, f64_01) in cases {
        set_rng_seed(seed);
        let mut m = Matrix::<u8>::new(1, 8, 1);
        randu(&mut m, Scalar::all(0.0), Scalar::all(256.0)).unwrap();
        assert_eq!(m.data, u8_256, "seed {seed}: u8 [0, 256)");

        set_rng_seed(seed);
        randu(&mut m, Scalar::all(0.0), Scalar::all(10.0)).unwrap();
        assert_eq!(m.data, u8_10, "seed {seed}: u8 [0, 10)");

        set_rng_seed(seed);
        let mut m = Matrix::<i32>::new(1, 8, 1);
        randu(&mut m, Scalar::all(-5.0), Scalar::all(5.0)).unwrap();
        assert_eq!(m.data, i32_m5_5, "seed {seed}: i32 [-5, 5)");

        set_rng_seed(seed);
        let mut m = Matrix::<f32>::new(1, 8, 1);
        randu(&mut m, Scalar::all(0.0), Scalar::all(1.0)).unwrap();
        assert_eq!(m.data, f32_01, "seed {seed}: f32 [0, 1)");

        set_rng_seed(seed);
        let mut m = Matrix::<f64>::new(1, 8, 1);
        randu(&mut m, Scalar::all(0.0), Scalar::all(1.0)).unwrap();
        assert_eq!(m.data, f64_01, "seed {seed}: f64 [0, 1)");
    }

    // Per-channel bounds (0, 100, 200) .. (10, 110, 210), 2 px.
    for (seed, want) in [
        (0u64, [6u8, 107, 209, 9, 107, 200]),
        (42, [0, 108, 209, 1, 104, 200]),
        (12345, [4, 103, 203, 1, 109, 204]),
    ] {
        set_rng_seed(seed);
        let mut m = Matrix::<u8>::new(1, 2, 3);
        randu(
            &mut m,
            Scalar::new(0.0, 100.0, 200.0, 0.0),
            Scalar::new(10.0, 110.0, 210.0, 0.0),
        )
        .unwrap();
        assert_eq!(m.data, want, "seed {seed}: 3-channel bounds");
    }

    for (seed, f32_01, u8_128_50, shuffled) in [
        (
            0u64,
            [
                -1.603_094_8e-9,
                0.158_135_06,
                -0.700_928,
                -0.428_596,
                1.238_742_1,
                -0.278_008_22,
                -0.648_398_76,
                0.305_163_3,
            ],
            [128u8, 136, 93, 107, 190, 114, 96, 143],
            [5, 7, 9, 4, 1, 6, 8, 2, 0, 3],
        ),
        (
            42,
            [
                2.416_655_3e-8,
                -0.625_846_8,
                0.588_324_07,
                -0.731_996_5,
                -0.194_894_58,
                -1.231_167_8,
                0.104_871_58,
                0.113_963_1,
            ],
            [128, 97, 157, 91, 118, 66, 133, 134],
            [5, 3, 7, 1, 4, 2, 8, 9, 6, 0],
        ),
        (
            12345,
            [
                8.270_294_5e-6,
                0.461_529_43,
                -0.232_334_06,
                0.903_827_3,
                0.399_476_77,
                0.599_755_64,
                1.282_757_6,
                -0.190_607_37,
            ],
            [128, 151, 116, 173, 148, 158, 192, 118],
            [4, 2, 8, 3, 5, 7, 1, 9, 6, 0],
        ),
    ] {
        set_rng_seed(seed);
        let mut m = Matrix::<f32>::new(1, 8, 1);
        randn(&mut m, Scalar::all(0.0), Scalar::all(1.0)).unwrap();
        assert_close_f32(&m.data, &f32_01, 0.0, &format!("seed {seed}: randn f32"));

        set_rng_seed(seed);
        let mut m = Matrix::<u8>::new(1, 8, 1);
        randn(&mut m, Scalar::all(128.0), Scalar::all(50.0)).unwrap();
        assert_eq!(m.data, u8_128_50, "seed {seed}: randn u8");

        for_each_shuffle_type(seed, shuffled);
    }

    // setRNGSeed(int) widens with sign extension: -1 is not the default state.
    set_rng_seed(0);
    let mut a = Matrix::<i32>::new(1, 4, 1);
    randu(&mut a, Scalar::all(0.0), Scalar::all(1000.0)).unwrap();
    assert_eq!(a.data, vec![606, 397, 839, 629]);
    set_rng_seed(u64::MAX);
    randu(&mut a, Scalar::all(0.0), Scalar::all(1000.0)).unwrap();
    assert_eq!(a.data, vec![605, 4, 940, 173]);
}

fn assert_close_f32(actual: &[f32], expected: &[f32], tol: f64, what: &str) {
    let a: Vec<f64> = actual.iter().map(|&v| v as f64).collect();
    let e: Vec<f64> = expected.iter().map(|&v| v as f64).collect();
    assert_close(&a, &e, tol, what);
}

fn for_each_shuffle_type(seed: u64, want: [i32; 10]) {
    set_rng_seed(seed);
    let mut idx: Vec<i32> = (0..10).collect();
    rand_shuffle(&mut idx);
    assert_eq!(idx, want, "seed {seed}: randShuffle i32");
    set_rng_seed(seed);
    let mut idx: Vec<u8> = (0..10).collect();
    rand_shuffle(&mut idx);
    let want_u8: Vec<u8> = want.iter().map(|&v| v as u8).collect();
    assert_eq!(idx, want_u8, "seed {seed}: randShuffle u8");
}

/// `core_rng_fill.py`: the `RNG::fill` paths beyond `core_rng.py` — block boundaries of the
/// four-per-draw integer loop (3 channels: 1026-element blocks), the fast-bits and division
/// paths on every integer depth, reversed float bounds, the ziggurat on several depths,
/// per-channel `randn` and a long `randShuffle`.
#[test]
fn rng_fill_paths_match_cv_rng() {
    set_rng_seed(42);
    let mut m = Matrix::<u8>::new(1, 400, 3);
    randu(&mut m, Scalar::all(0.0), Scalar::all(256.0)).unwrap();
    assert_eq!(m.data.iter().map(|&v| v as u64).sum::<u64>(), 154973);
    assert_eq!(
        m.data[1020..1032],
        [39, 125, 213, 112, 120, 31, 72, 113, 36, 90, 58, 80]
    );
    assert_eq!(m.data[1196..], [234, 37, 171, 52]);

    set_rng_seed(5);
    let mut m = Matrix::<u8>::new(1, 700, 2);
    randu(
        &mut m,
        Scalar::new(0.0, 0.0, 0.0, 0.0),
        Scalar::new(256.0, 10.0, 0.0, 0.0),
    )
    .unwrap();
    assert_eq!(m.data.iter().map(|&v| v as u64).sum::<u64>(), 91752);
    assert_eq!(m.data[1020..1030], [80, 6, 130, 1, 109, 4, 129, 9, 111, 2]);
    assert_eq!(m.data[1396..], [156, 0, 187, 6]);

    set_rng_seed(7);
    let mut m = Matrix::<u16>::new(1, 8, 1);
    randu(&mut m, Scalar::all(0.0), Scalar::all(65536.0)).unwrap();
    assert_eq!(
        m.data,
        [46406, 9922, 21612, 35803, 60154, 9265, 42626, 41169]
    );

    set_rng_seed(7);
    let mut m = Matrix::<i16>::new(1, 8, 1);
    randu(&mut m, Scalar::all(-1000.0), Scalar::all(1000.0)).unwrap();
    assert_eq!(m.data, [-946, -246, -348, 835, -574, 729, 314, 953]);

    set_rng_seed(7);
    let mut m = Matrix::<i32>::new(1, 8, 1);
    randu(&mut m, Scalar::all(-1e9), Scalar::all(1e9)).unwrap();
    assert_eq!(
        m.data,
        [
            384522054, 836290754, -484051348, 653967835, 544050426, -493304271, -114210686,
            595935953
        ]
    );

    set_rng_seed(7);
    let mut m = Matrix::<i8>::new(1, 8, 1);
    randu(&mut m, Scalar::all(-128.0), Scalar::all(128.0)).unwrap();
    assert_eq!(m.data, [-58, 53, 59, 73, 66, -90, 41, 100]);

    set_rng_seed(7);
    let mut m = Matrix::<f32>::new(1, 8, 1);
    randu(&mut m, Scalar::all(-3.0), Scalar::all(7.0)).unwrap();
    assert_eq!(
        m.data,
        [
            -0.119_795_56,
            0.932_060_5,
            -2.142_100_3,
            5.850_944,
            0.251_635_43,
            -2.163_644_3,
            -1.280_998_2,
            0.372_440_81
        ]
    );

    set_rng_seed(7);
    let mut m = Matrix::<f64>::new(1, 8, 1);
    randu(&mut m, Scalar::all(5.0), Scalar::all(-5.0)).unwrap();
    assert_eq!(
        m.data,
        [
            2.119795517064631,
            1.06793954324872,
            4.142100557530864,
            -3.850943957381068,
            1.7483645808570258,
            4.163644197862188,
            3.2809981644014794,
            1.6275591737321737
        ]
    );

    set_rng_seed(3);
    let mut m = Matrix::<f64>::new(1, 8, 1);
    randn(&mut m, Scalar::all(0.0), Scalar::all(1.0)).unwrap();
    let want = [
        5.958806226225022e-10,
        -0.19294486939907074,
        -0.4868623614311218,
        0.5748234391212463,
        0.6561874747276306,
        0.7934845089912415,
        0.19608648121356964,
        0.15683351457118988,
    ];
    assert_close(&m.data, &want, 0.0, "randn f64");

    set_rng_seed(3);
    let mut m = Matrix::<i16>::new(1, 8, 1);
    randn(&mut m, Scalar::all(0.0), Scalar::all(1000.0)).unwrap();
    assert_eq!(m.data, [0, -193, -487, 575, 656, 793, 196, 157]);

    set_rng_seed(3);
    let mut m = Matrix::<f32>::new(1, 4, 3);
    randn(
        &mut m,
        Scalar::new(0.0, 10.0, 100.0, 0.0),
        Scalar::new(1.0, 2.0, 3.0, 0.0),
    )
    .unwrap();
    let want = [
        5.958_806e-10,
        9.614_11,
        98.539_41,
        0.574_823_44,
        11.312_375,
        102.380_455,
        0.196_086_48,
        10.313_667,
        100.720_18,
        1.279_372_8,
        6.588_620_7,
        98.167_42,
    ];
    assert_close_f32(&m.data, &want, 0.0, "randn f32 3 channels");

    set_rng_seed(1);
    let mut m = Matrix::<u8>::new(1, 2000, 1);
    randn(&mut m, Scalar::all(128.0), Scalar::all(200.0)).unwrap();
    assert_eq!(m.data.iter().filter(|&&v| v == 0).count(), 524);
    assert_eq!(m.data.iter().filter(|&&v| v == 255).count(), 545);
    assert_eq!(m.data[..8], [128, 255, 255, 152, 194, 216, 255, 0]);

    set_rng_seed(11);
    let mut m = Matrix::<f32>::new(1, 5000, 1);
    randn(&mut m, Scalar::all(0.0), Scalar::all(1.0)).unwrap();
    assert_close_f32(
        &m.data[1022..1026],
        &[0.987_078_55, -0.904_932_3, 1.387_338_6, -1.774_771],
        0.0,
        "randn f32 across a block boundary",
    );
    assert_close_f32(
        &m.data[4998..],
        &[1.829_689_4, 0.581_550_66],
        0.0,
        "randn f32 tail",
    );

    set_rng_seed(42);
    let mut idx: Vec<i32> = (0..1000).collect();
    rand_shuffle(&mut idx);
    assert_eq!(idx[..10], [471, 48, 5, 421, 374, 959, 703, 142, 503, 915]);
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

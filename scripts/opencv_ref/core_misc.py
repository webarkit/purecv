"""OpenCV 4.10.0 reference values for the remaining measured core rows.

Rows: border_interpolate, copy_make_border, flip, rotate, transpose, repeat, check_range,
compare, in_range, lut, transform, perspective_transform, gemm, invert, determinant, solve,
trace, mahalanobis, psnr, sort, sort_idx, sqrt/exp/log/pow, magnitude/phase/cart_to_polar/
polar_to_cart, solve_cubic, solve_poly, set_identity. See docs/parity/core.md.

Run from the repository root:
    cpp_ref/.venv/Scripts/python scripts/opencv_ref/core_misc.py   # Windows
    cpp_ref/.venv/bin/python scripts/opencv_ref/core_misc.py       # Linux/macOS
"""

import cv2

assert cv2.__version__ == "4.10.0"
cv2.ipp.setUseIPP(False)  # measure the generic CPU path

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _core_common import dump, tolist  # noqa: E402


def attempt(fn):
    try:
        return tolist(fn())
    except cv2.error as e:  # noqa: PERF203
        return f"cv2.error: {e.code} {e.err}".strip()


def main() -> None:
    results = {}

    # --- borderInterpolate: p in -7..12 for len 5, every border type
    ps = list(range(-7, 13))
    borders = {
        "constant": cv2.BORDER_CONSTANT,
        "replicate": cv2.BORDER_REPLICATE,
        "reflect": cv2.BORDER_REFLECT,
        "wrap": cv2.BORDER_WRAP,
        "reflect_101": cv2.BORDER_REFLECT_101,
    }
    results["border_interpolate"] = {
        "p": ps,
        "len": 5,
        **{name: [cv2.borderInterpolate(p, 5, code) for p in ps] for name, code in borders.items()},
        "reflect_len1": [cv2.borderInterpolate(p, 1, cv2.BORDER_REFLECT) for p in (-3, 0, 4)],
        "reflect_101_len1": [cv2.borderInterpolate(p, 1, cv2.BORDER_REFLECT_101) for p in (-3, 0, 4)],
        "transparent": attempt(lambda: cv2.borderInterpolate(-1, 5, cv2.BORDER_TRANSPARENT)),
        "isolated_flag_alone": attempt(lambda: cv2.borderInterpolate(-1, 5, cv2.BORDER_ISOLATED)),
        "reflect_101_or_isolated": attempt(lambda: cv2.borderInterpolate(-1, 5, cv2.BORDER_REFLECT_101 | cv2.BORDER_ISOLATED)),
    }

    # --- copyMakeBorder
    img = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.uint8)
    results["copy_make_border"] = {
        "src": tolist(img),
        "top_bottom_left_right": [2, 1, 2, 1],
        **{
            name: tolist(cv2.copyMakeBorder(img, 2, 1, 2, 1, code, value=9))
            for name, code in borders.items()
        },
        "transparent": attempt(lambda: cv2.copyMakeBorder(img, 2, 1, 2, 1, cv2.BORDER_TRANSPARENT, value=9)),
        "reflect_101_oversized_border": attempt(lambda: cv2.copyMakeBorder(img, 5, 0, 0, 0, cv2.BORDER_REFLECT_101)),
        "reflect_oversized_border": tolist(cv2.copyMakeBorder(img, 5, 0, 0, 0, cv2.BORDER_REFLECT)),
        "wrap_oversized_border": tolist(cv2.copyMakeBorder(img, 5, 0, 0, 0, cv2.BORDER_WRAP)),
    }

    # --- flip / rotate / transpose / repeat
    m23 = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    results["structural"] = {
        "src": tolist(m23),
        "flip_0": tolist(cv2.flip(m23, 0)),
        "flip_1": tolist(cv2.flip(m23, 1)),
        "flip_minus1": tolist(cv2.flip(m23, -1)),
        "flip_7": tolist(cv2.flip(m23, 7)),
        "rotate_90_cw": tolist(cv2.rotate(m23, cv2.ROTATE_90_CLOCKWISE)),
        "rotate_180": tolist(cv2.rotate(m23, cv2.ROTATE_180)),
        "rotate_90_ccw": tolist(cv2.rotate(m23, cv2.ROTATE_90_COUNTERCLOCKWISE)),
        "rotate_3": attempt(lambda: cv2.rotate(m23, 3)),
        "transpose": tolist(cv2.transpose(m23)),
        "repeat_2_3": tolist(cv2.repeat(m23, 2, 3)),
        "repeat_0_1": attempt(lambda: cv2.repeat(m23, 0, 1)),
    }

    # --- checkRange (half-open upper bound), compare, inRange
    cr = np.array([[0, 100, 255]], dtype=np.uint8)
    crf = np.array([[0.0, 1.0, 255.0]], dtype=np.float32)
    nanf = np.array([[0.0, np.nan]], dtype=np.float32)
    results["check_range"] = {
        "u8_src": tolist(cr),
        "u8_0_255": [bool(cv2.checkRange(cr, True, 0, 255)[0]), tolist(cv2.checkRange(cr, True, 0, 255)[1])],
        "u8_0_256": bool(cv2.checkRange(cr, True, 0, 256)[0]),
        "f32_src": tolist(crf),
        "f32_0_255": [bool(cv2.checkRange(crf, True, 0.0, 255.0)[0]), tolist(cv2.checkRange(crf, True, 0.0, 255.0)[1])],
        "f32_0_255.5": bool(cv2.checkRange(crf, True, 0.0, 255.5)[0]),
        "f32_nan_default_bounds": bool(cv2.checkRange(nanf)[0]),
    }
    a = np.array([[1, 5, 3]], dtype=np.uint8)
    b = np.array([[1, 2, 7]], dtype=np.uint8)
    rgb = np.array([[[10, 20, 30], [40, 50, 60]]], dtype=np.uint8)
    results["compare_in_range"] = {
        "a": tolist(a),
        "b": tolist(b),
        "cmp_eq": tolist(cv2.compare(a, b, cv2.CMP_EQ)),
        "cmp_gt": tolist(cv2.compare(a, b, cv2.CMP_GT)),
        "cmp_le": tolist(cv2.compare(a, b, cv2.CMP_LE)),
        "cmp_scalar_gt_2": tolist(cv2.compare(a, 2, cv2.CMP_GT)),
        "cmp_scalar_eq_2.5": tolist(cv2.compare(a, 2.5, cv2.CMP_EQ)),
        "cmp_scalar_gt_2.5": tolist(cv2.compare(a, 2.5, cv2.CMP_GT)),
        "rgb": tolist(rgb),
        "in_range_scalar": tolist(cv2.inRange(rgb, (10, 20, 30), (40, 50, 59))),
        "in_range_arrays": tolist(cv2.inRange(rgb, np.full_like(rgb, 0), np.full_like(rgb, 45))),
    }

    # --- LUT
    lut = np.arange(256, dtype=np.uint8)[::-1].copy()
    lut3 = np.stack([lut, np.arange(256, dtype=np.uint8), np.full(256, 7, np.uint8)], axis=-1).reshape(1, 256, 3)
    results["lut"] = {
        "rgb": tolist(rgb),
        "inverted_1ch_lut_on_3ch": tolist(cv2.LUT(rgb, lut)),
        "3ch_lut": tolist(cv2.LUT(rgb, lut3)),
        "f32_lut_output": tolist(cv2.LUT(a, np.arange(256, dtype=np.float32) * 0.5)),
        "lut_255_entries": attempt(lambda: cv2.LUT(a, lut[:255])),
        "u16_src": attempt(lambda: cv2.LUT(a.astype(np.uint16), lut)),
    }

    # --- transform / perspectiveTransform
    px = np.array([[[10, 20, 33], [255, 255, 255], [1, 1, 1]]], dtype=np.uint8)
    pxf = px.astype(np.float32)
    m1x3 = np.array([[0.3, 0.3, 0.3]], dtype=np.float64)
    m2x4 = np.array([[1.0, 0.0, 0.0, 0.5], [0.0, 0.0, 1.0, -0.5]], dtype=np.float64)
    pts = np.array([[[1.0, 2.0], [3.0, 4.0], [0.0, 0.0]]], dtype=np.float64)
    H = np.array([[2.0, 0.0, 1.0], [0.0, 3.0, -1.0], [0.5, 0.0, 1.0]], dtype=np.float64)
    Hzero = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.float64)
    results["transform"] = {
        "u8_src": tolist(px),
        "u8_m1x3_0.3": tolist(cv2.transform(px, m1x3)),
        "f32_m1x3_0.3": tolist(cv2.transform(pxf, m1x3)),
        "u8_m2x4_affine": tolist(cv2.transform(px, m2x4)),
        "u8_m1x2_wrong_cols": attempt(lambda: cv2.transform(px, np.array([[1.0, 1.0]]))),
        "pts": tolist(pts),
        "perspective_H": tolist(cv2.perspectiveTransform(pts, H)),
        "perspective_w_zero": tolist(cv2.perspectiveTransform(pts, Hzero)),
        "perspective_u8": attempt(lambda: cv2.perspectiveTransform(px[:, :, :2].copy(), H)),
    }

    # --- gemm / invert / determinant / solve / trace / Mahalanobis
    A = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    B = np.array([[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]], dtype=np.float32)
    Cm = np.array([[1.0, 1.0], [1.0, 1.0]], dtype=np.float32)
    S = np.array([[4.0, 7.0], [2.0, 6.0]], dtype=np.float64)
    S4 = np.array([[2.0, 0.0, 1.0, 3.0], [1.0, 1.0, 0.0, 2.0], [0.0, 2.0, 1.0, 1.0], [3.0, 1.0, 2.0, 0.0]], dtype=np.float64)
    sing = np.array([[1.0, 2.0], [2.0, 4.0]], dtype=np.float64)
    rhs = np.array([[1.0], [2.0]], dtype=np.float64)
    ret_inv, inv = cv2.invert(S, flags=cv2.DECOMP_LU)
    ret_sing, inv_sing = cv2.invert(sing, flags=cv2.DECOMP_LU)
    ret_svd, inv_svd = cv2.invert(S, flags=cv2.DECOMP_SVD)
    ok, x = cv2.solve(S, rhs, flags=cv2.DECOMP_LU)
    ok_s, x_s = cv2.solve(sing, rhs, flags=cv2.DECOMP_LU)
    v1 = np.array([[1.0], [2.0]], dtype=np.float64)
    v2 = np.array([[3.0], [5.0]], dtype=np.float64)
    icov = np.array([[2.0, 0.5], [0.5, 1.0]], dtype=np.float64)
    results["linalg"] = {
        "A": tolist(A),
        "B": tolist(B),
        "C": tolist(Cm),
        "gemm_A_B_2_C_0.5": tolist(cv2.gemm(A, B, 2.0, Cm, 0.5)),
        "gemm_At_A_1": tolist(cv2.gemm(A, A, 1.0, None, 0.0, flags=cv2.GEMM_1_T)),
        "gemm_A_Bt_mismatch": attempt(lambda: cv2.gemm(A, B, 1.0, None, 0.0, flags=cv2.GEMM_2_T)),
        "gemm_u8": attempt(lambda: cv2.gemm(A.astype(np.uint8), B.astype(np.uint8), 1.0, None, 0.0)),
        "S": tolist(S),
        "invert_lu_return": float(ret_inv),
        "invert_lu": tolist(inv),
        "invert_lu_singular_return": float(ret_sing),
        "invert_lu_singular": tolist(inv_sing),
        "invert_svd_return": float(ret_svd),
        "determinant_S": cv2.determinant(S),
        "determinant_S4": cv2.determinant(S4),
        "determinant_singular": cv2.determinant(sing),
        "determinant_non_square": attempt(lambda: cv2.determinant(A)),
        "determinant_u8": attempt(lambda: cv2.determinant(S.astype(np.uint8))),
        "solve_ok": bool(ok),
        "solve_x": tolist(x),
        "solve_singular_ok": bool(ok_s),
        "solve_singular_x": tolist(x_s),
        "trace_S4": tolist(cv2.trace(S4)),
        "trace_2ch": tolist(cv2.trace(np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.float32))),
        "mahalanobis": cv2.Mahalanobis(v1, v2, icov),
        "mahalanobis_row_vectors": cv2.Mahalanobis(v1.T.copy(), v2.T.copy(), icov),
        "psnr_identical": cv2.PSNR(rgb, rgb),
        "psnr_u8": cv2.PSNR(rgb, cv2.add(rgb, 3)),
        "psnr_f32_R1": cv2.PSNR(rgb.astype(np.float32) / 255.0, cv2.add(rgb, 3).astype(np.float32) / 255.0, 1.0),
        "psnr_f32_default_R": cv2.PSNR(rgb.astype(np.float32) / 255.0, cv2.add(rgb, 3).astype(np.float32) / 255.0),
        "set_identity_2ch_scalar_1_2": tolist(cv2.setIdentity(np.zeros((2, 3, 2), np.float32), (1, 2, 0, 0))),
        "set_identity_1ch_scalar_5": tolist(cv2.setIdentity(np.zeros((3, 2), np.float32), 5)),
    }

    # --- sort / sortIdx (ties)
    srt = np.array([[3, 1, 3, 2], [9, 9, 0, 9]], dtype=np.int32)
    results["sort"] = {
        "src": tolist(srt),
        "row_asc": tolist(cv2.sort(srt, cv2.SORT_EVERY_ROW | cv2.SORT_ASCENDING)),
        "row_desc": tolist(cv2.sort(srt, cv2.SORT_EVERY_ROW | cv2.SORT_DESCENDING)),
        "col_asc": tolist(cv2.sort(srt, cv2.SORT_EVERY_COLUMN | cv2.SORT_ASCENDING)),
        "idx_row_asc": tolist(cv2.sortIdx(srt, cv2.SORT_EVERY_ROW | cv2.SORT_ASCENDING)),
        "idx_row_desc": tolist(cv2.sortIdx(srt, cv2.SORT_EVERY_ROW | cv2.SORT_DESCENDING)),
        "idx_col_desc": tolist(cv2.sortIdx(srt, cv2.SORT_EVERY_COLUMN | cv2.SORT_DESCENDING)),
        "multi_channel": attempt(lambda: cv2.sort(rgb, cv2.SORT_EVERY_ROW)),
    }

    # --- sqrt / exp / log / pow and the polar family
    xs = np.array([[0.0, 0.5, 1.0, 2.0, 10.0, 100.0, 1e-3, 3.7]], dtype=np.float32)
    neg = np.array([[-1.0, -4.0]], dtype=np.float32)
    x = np.array([[1.0, -1.0, 1.0, 0.0, 3.0, 1.0, -1.0, 1e-7]], dtype=np.float32)
    y = np.array([[1.0, 1.0, -1e-3, 0.0, 4.0, 0.0, -1.0, 1.0]], dtype=np.float32)
    mag, ang_deg = cv2.cartToPolar(x, y, angleInDegrees=True)
    _, ang_rad = cv2.cartToPolar(x, y, angleInDegrees=False)
    px_, py_ = cv2.polarToCart(mag, ang_deg, angleInDegrees=True)
    results["mathfuncs"] = {
        "xs": tolist(xs),
        "sqrt": tolist(cv2.sqrt(xs)),
        "sqrt_negative": tolist(cv2.sqrt(neg)),
        "exp": tolist(cv2.exp(xs)),
        "log": tolist(cv2.log(xs)),
        "log_negative": tolist(cv2.log(neg)),
        "pow_2": tolist(cv2.pow(xs, 2.0)),
        "pow_0.5": tolist(cv2.pow(xs, 0.5)),
        "pow_1.5": tolist(cv2.pow(xs, 1.5)),
        "pow_neg_1": tolist(cv2.pow(xs, -1.0)),
        "pow_1.5_negative_base": tolist(cv2.pow(neg, 1.5)),
        "pow_3_negative_base": tolist(cv2.pow(neg, 3.0)),
        "sqrt_u8": attempt(lambda: cv2.sqrt(np.array([[4, 9]], np.uint8))),
        "exp_u8": attempt(lambda: cv2.exp(np.array([[1, 2]], np.uint8))),
        "x": tolist(x),
        "y": tolist(y),
        "magnitude": tolist(cv2.magnitude(x, y)),
        "phase_deg": tolist(cv2.phase(x, y, angleInDegrees=True)),
        "phase_rad": tolist(cv2.phase(x, y, angleInDegrees=False)),
        "cart_to_polar_deg": [tolist(mag), tolist(ang_deg)],
        "cart_to_polar_rad_angle": tolist(ang_rad),
        "polar_to_cart_deg": [tolist(px_), tolist(py_)],
        "magnitude_u8": attempt(lambda: cv2.magnitude(np.array([[3]], np.uint8), np.array([[4]], np.uint8))),
        "fast_atan2_1_1": cv2.fastAtan2(1.0, 1.0),
        "fast_atan2_1_m1": cv2.fastAtan2(1.0, -1.0),
        "fast_atan2_m1e3_1": cv2.fastAtan2(-1e-3, 1.0),
        "cube_root_27": cv2.cubeRoot(27.0),
    }

    # --- solveCubic / solvePoly
    results["solvers"] = {
        "cubic_1_m6_11_m6": [int(cv2.solveCubic(np.array([1.0, -6.0, 11.0, -6.0]))[0]), tolist(cv2.solveCubic(np.array([1.0, -6.0, 11.0, -6.0]))[1])],
        "cubic_1_0_0_m8": [int(cv2.solveCubic(np.array([1.0, 0.0, 0.0, -8.0]))[0]), tolist(cv2.solveCubic(np.array([1.0, 0.0, 0.0, -8.0]))[1])],
        "cubic_1_m3_3_m1_triple": [int(cv2.solveCubic(np.array([1.0, -3.0, 3.0, -1.0]))[0]), tolist(cv2.solveCubic(np.array([1.0, -3.0, 3.0, -1.0]))[1])],
        "cubic_0_1_m3_2_quadratic": [int(cv2.solveCubic(np.array([0.0, 1.0, -3.0, 2.0]))[0]), tolist(cv2.solveCubic(np.array([0.0, 1.0, -3.0, 2.0]))[1])],
        "cubic_three_coeffs_1_m3_2": [int(cv2.solveCubic(np.array([1.0, -3.0, 2.0]))[0]), tolist(cv2.solveCubic(np.array([1.0, -3.0, 2.0]))[1])],
        "cubic_0_0_2_m4_linear": [int(cv2.solveCubic(np.array([0.0, 0.0, 2.0, -4.0]))[0]), tolist(cv2.solveCubic(np.array([0.0, 0.0, 2.0, -4.0]))[1])],
        "cubic_0_1_0_1_no_real": [int(cv2.solveCubic(np.array([0.0, 1.0, 0.0, 1.0]))[0]), tolist(cv2.solveCubic(np.array([0.0, 1.0, 0.0, 1.0]))[1])],
        # solvePoly coefficients are low-degree first: x^2 - 3x + 2 -> [2, -3, 1]
        "poly_2_m3_1": [float(cv2.solvePoly(np.array([2.0, -3.0, 1.0]))[0]), tolist(cv2.solvePoly(np.array([2.0, -3.0, 1.0]))[1])],
        "poly_1_0_1_complex": [float(cv2.solvePoly(np.array([1.0, 0.0, 1.0]))[0]), tolist(cv2.solvePoly(np.array([1.0, 0.0, 1.0]))[1])],
        "poly_m6_11_m6_1": [float(cv2.solvePoly(np.array([-6.0, 11.0, -6.0, 1.0]))[0]), tolist(cv2.solvePoly(np.array([-6.0, 11.0, -6.0, 1.0]))[1])],
    }

    dump("core_misc.py", results)


if __name__ == "__main__":
    main()

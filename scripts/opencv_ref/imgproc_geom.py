"""OpenCV 4.10.0 reference values for the imgproc geometric rows (docs/parity/imgproc.md).

Rows: resize, remap, warp_perspective, pyr_down, pyr_up, build_pyramid, plus the
INTER_* modes and warpAffine / getRotationMatrix2D that purecv does not have (`missing`).

Every input is defined here by a formula, so tests/parity_imgproc.rs can rebuild it.

Run from the repository root:
    cpp_ref/.venv/Scripts/python scripts/opencv_ref/imgproc_geom.py   # Windows
    cpp_ref/.venv/bin/python scripts/opencv_ref/imgproc_geom.py       # Linux/macOS
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


def pattern_u8(rows, cols, channels=1):
    """Deterministic test image: ((7*y + 13*x + 29*c) * 37) % 256."""
    y, x, c = np.meshgrid(np.arange(rows), np.arange(cols), np.arange(channels), indexing="ij")
    img = ((7 * y + 13 * x + 29 * c) * 37) % 256
    img = img.astype(np.uint8)
    return img if channels > 1 else img[:, :, 0].copy()


def main() -> None:
    results = {}

    img8 = pattern_u8(8, 8)
    img8c3 = pattern_u8(6, 7, 3)
    img32 = img8.astype(np.float32)
    img5 = pattern_u8(5, 5)

    # --- resize
    r = {"src_u8": tolist(img8), "src_u8_5x5": tolist(img5), "src_u8c3": tolist(img8c3)}
    r["u8_8x8_to_5x5_linear"] = tolist(cv2.resize(img8, (5, 5), interpolation=cv2.INTER_LINEAR))
    r["u8_8x8_to_4x4_linear_is_area"] = tolist(cv2.resize(img8, (4, 4), interpolation=cv2.INTER_LINEAR))
    r["u8_8x8_to_11x13_linear"] = tolist(cv2.resize(img8, (11, 13), interpolation=cv2.INTER_LINEAR))
    r["u8_5x5_to_7x7_linear"] = tolist(cv2.resize(img5, (7, 7), interpolation=cv2.INTER_LINEAR))
    r["u8_5x5_to_3x3_linear"] = tolist(cv2.resize(img5, (3, 3), interpolation=cv2.INTER_LINEAR))
    r["u8c3_to_4x3_linear"] = tolist(cv2.resize(img8c3, (4, 3), interpolation=cv2.INTER_LINEAR))
    r["f32_8x8_to_5x5_linear"] = tolist(cv2.resize(img32, (5, 5), interpolation=cv2.INTER_LINEAR))
    r["f32_5x5_to_7x7_linear"] = tolist(cv2.resize(img5.astype(np.float32), (7, 7), interpolation=cv2.INTER_LINEAR))
    r["f32_8x8_to_4x4_linear_is_area"] = tolist(cv2.resize(img32, (4, 4), interpolation=cv2.INTER_LINEAR))
    r["u8_8x8_to_5x5_linear_exact"] = tolist(cv2.resize(img8, (5, 5), interpolation=cv2.INTER_LINEAR_EXACT))
    r["u8_8x8_to_5x5_nearest"] = tolist(cv2.resize(img8, (5, 5), interpolation=cv2.INTER_NEAREST))
    r["u8_8x8_to_5x5_nearest_exact"] = tolist(cv2.resize(img8, (5, 5), interpolation=cv2.INTER_NEAREST_EXACT))
    r["u8_8x8_to_5x5_area"] = tolist(cv2.resize(img8, (5, 5), interpolation=cv2.INTER_AREA))
    r["u8_8x8_to_5x5_cubic"] = tolist(cv2.resize(img8, (5, 5), interpolation=cv2.INTER_CUBIC))
    r["u8_8x8_to_5x5_lanczos4"] = tolist(cv2.resize(img8, (5, 5), interpolation=cv2.INTER_LANCZOS4))
    r["u8_fx0.5_fy0.5"] = tolist(cv2.resize(img8, None, fx=0.5, fy=0.5))
    r["u8_fx0.6_fy0.6_dsize"] = list(cv2.resize(img8, None, fx=0.6, fy=0.6).shape)
    r["u8_8x8_to_8x8_identity"] = tolist(cv2.resize(img8, (8, 8)))
    r["u8_to_0x0"] = attempt(lambda: cv2.resize(img8, (0, 0)))
    results["resize"] = r

    # --- remap
    rm = {"src_u8": tolist(img8)}
    ys, xs = np.mgrid[0:8, 0:8].astype(np.float32)
    mapx_half = xs + 0.5
    mapy_half = ys + 0.5
    mapx_frac = xs * 0.9 + 0.3
    mapy_frac = ys * 0.8 + 0.1
    mapx_out = xs - 2.0
    mapy_out = ys + 2.0
    for border_name, border in (("constant", cv2.BORDER_CONSTANT), ("replicate", cv2.BORDER_REPLICATE), ("reflect101", cv2.BORDER_REFLECT_101)):
        rm[f"u8_nearest_half_{border_name}"] = tolist(cv2.remap(img8, mapx_half, mapy_half, cv2.INTER_NEAREST, borderMode=border, borderValue=9))
        rm[f"u8_linear_frac_{border_name}"] = tolist(cv2.remap(img8, mapx_frac, mapy_frac, cv2.INTER_LINEAR, borderMode=border, borderValue=9))
        rm[f"u8_linear_out_{border_name}"] = tolist(cv2.remap(img8, mapx_out, mapy_out, cv2.INTER_LINEAR, borderMode=border, borderValue=9))
        rm[f"u8_nearest_out_{border_name}"] = tolist(cv2.remap(img8, mapx_out, mapy_out, cv2.INTER_NEAREST, borderMode=border, borderValue=9))
    rm["u8_linear_half_constant"] = tolist(cv2.remap(img8, mapx_half, mapy_half, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=9))
    rm["f32_linear_frac_constant"] = tolist(cv2.remap(img32, mapx_frac, mapy_frac, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=9))
    rm["f32_nearest_half_constant"] = tolist(cv2.remap(img32, mapx_half, mapy_half, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=9))
    rm["u8_linear_frac_reflect"] = tolist(cv2.remap(img8, mapx_out, mapy_out, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT, borderValue=9))
    rm["u8_linear_frac_wrap"] = tolist(cv2.remap(img8, mapx_out, mapy_out, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP, borderValue=9))
    rm["u8_linear_transparent"] = tolist(cv2.remap(img8, mapx_out, mapy_out, cv2.INTER_LINEAR, borderMode=cv2.BORDER_TRANSPARENT, borderValue=9))
    rm["u8_cubic_frac_constant"] = tolist(cv2.remap(img8, mapx_frac, mapy_frac, cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=9))
    rm["u8_linear_frac_map_32fc2"] = tolist(cv2.remap(img8, np.dstack([mapx_frac, mapy_frac]), None, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=9))
    rm["u8_linear_smaller_maps_5x5"] = tolist(cv2.remap(img8, mapx_frac[:5, :5].copy(), mapy_frac[:5, :5].copy(), cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=9))
    # 1/32-pixel quantisation of the fractional part: map 1.015 (-> 1 + 0/32) vs 1.016 (-> 1 + 1/32)
    quant_x = np.array([[1.015, 1.016, 1.03125, 1.046, 1.047, 2.5]], np.float32)
    quant_y = np.zeros_like(quant_x)
    rm["u8_linear_quant_map_x"] = tolist(quant_x)
    rm["u8_linear_quant"] = tolist(cv2.remap(img8, quant_x, quant_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=9))
    rm["u8_nearest_quant"] = tolist(cv2.remap(img8, quant_x, quant_y, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=9))
    results["remap"] = rm

    # --- warpPerspective
    wp = {"src_u8": tolist(img8)}
    ident = np.eye(3, dtype=np.float64)
    shift = np.array([[1, 0, 0.5], [0, 1, 0.25], [0, 0, 1]], np.float64)
    persp = np.array([[1.1, 0.1, -0.5], [0.05, 0.9, 0.3], [0.02, 0.01, 1.0]], np.float64)
    persp_inv_w0 = np.array([[1, 0, 0], [0, 1, 0], [0.5, 0, 0]], np.float64)  # w == 0 on a column
    wp["persp_m"] = tolist(persp)
    for name, M in (("identity", ident), ("shift", shift), ("persp", persp)):
        for border_name, border in (("constant", cv2.BORDER_CONSTANT), ("replicate", cv2.BORDER_REPLICATE)):
            wp[f"u8_{name}_linear_{border_name}"] = tolist(cv2.warpPerspective(img8, M, (8, 8), flags=cv2.INTER_LINEAR, borderMode=border, borderValue=9))
            wp[f"u8_{name}_nearest_{border_name}"] = tolist(cv2.warpPerspective(img8, M, (8, 8), flags=cv2.INTER_NEAREST, borderMode=border, borderValue=9))
    wp["u8_persp_linear_dsize_6x5"] = tolist(cv2.warpPerspective(img8, persp, (6, 5), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=9))
    wp["f32_persp_linear_constant"] = tolist(cv2.warpPerspective(img32, persp, (8, 8), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=9))
    wp["u8_persp_linear_inverse_map"] = tolist(cv2.warpPerspective(img8, persp, (8, 8), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_CONSTANT, borderValue=9))
    wp["u8_w0_linear_constant"] = tolist(cv2.warpPerspective(img8, persp_inv_w0, (8, 8), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_CONSTANT, borderValue=9))
    wp["u8_singular_m"] = attempt(lambda: cv2.warpPerspective(img8, np.zeros((3, 3)), (8, 8)))
    wp["u8_shift_cubic_constant"] = tolist(cv2.warpPerspective(img8, shift, (8, 8), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=9))
    results["warp_perspective"] = wp

    # --- warpAffine / getRotationMatrix2D / getAffineTransform / invertAffineTransform (missing rows)
    wa = {}
    rot = cv2.getRotationMatrix2D((3.5, 3.5), 30.0, 1.0)
    wa["rotation_matrix_30deg"] = tolist(rot)
    wa["u8_rot30_linear_constant"] = tolist(cv2.warpAffine(img8, rot, (8, 8), borderValue=9))
    wa["invert_affine"] = tolist(cv2.invertAffineTransform(rot))
    srcp = np.array([[0, 0], [7, 0], [0, 7]], np.float32)
    dstp = np.array([[1, 1], [6, 2], [0, 6]], np.float32)
    wa["get_affine_transform"] = tolist(cv2.getAffineTransform(srcp, dstp))
    srcq = np.array([[0, 0], [7, 0], [7, 7], [0, 7]], np.float32)
    dstq = np.array([[1, 1], [6, 0], [7, 7], [0, 6]], np.float32)
    wa["get_perspective_transform"] = tolist(cv2.getPerspectiveTransform(srcq, dstq))
    results["warp_affine_missing"] = wa

    # --- pyrDown / pyrUp / buildPyramid
    img9 = pattern_u8(9, 7)
    img6 = pattern_u8(6, 6)
    p = {"src_u8_8x8": tolist(img8), "src_u8_9x7": tolist(img9), "src_u8_6x6": tolist(img6), "src_u8c3": tolist(img8c3)}
    p["down_u8_8x8"] = tolist(cv2.pyrDown(img8))
    p["down_u8_9x7"] = tolist(cv2.pyrDown(img9))
    p["down_u8c3"] = tolist(cv2.pyrDown(img8c3))
    p["down_f32_8x8"] = tolist(cv2.pyrDown(img32))
    p["down_f32_9x7"] = tolist(cv2.pyrDown(img9.astype(np.float32)))
    p["down_u8_8x8_replicate"] = tolist(cv2.pyrDown(img8, borderType=cv2.BORDER_REPLICATE))
    p["down_u8_8x8_reflect"] = tolist(cv2.pyrDown(img8, borderType=cv2.BORDER_REFLECT))
    p["down_u8_8x8_constant"] = attempt(lambda: cv2.pyrDown(img8, borderType=cv2.BORDER_CONSTANT))
    p["down_u8_8x8_dsize_5x5"] = tolist(cv2.pyrDown(img8, dstsize=(5, 5)))
    p["down_u8_8x8_dsize_3x3"] = attempt(lambda: cv2.pyrDown(img8, dstsize=(3, 3)))
    p["down_u8_8x8_dsize_2x2"] = attempt(lambda: cv2.pyrDown(img8, dstsize=(2, 2)))
    p["down_i16_signed"] = tolist(cv2.pyrDown(img8.astype(np.int16) - 128))
    p["up_u8_6x6"] = tolist(cv2.pyrUp(img6))
    p["up_u8_8x8"] = tolist(cv2.pyrUp(img8))
    p["up_u8c3"] = tolist(cv2.pyrUp(img8c3))
    p["up_f32_6x6"] = tolist(cv2.pyrUp(img6.astype(np.float32)))
    p["up_u8_6x6_dsize_13x13"] = tolist(cv2.pyrUp(img6, dstsize=(13, 13)))
    p["up_u8_6x6_dsize_11x11"] = attempt(lambda: cv2.pyrUp(img6, dstsize=(11, 11)))
    p["up_u8_6x6_replicate"] = attempt(lambda: cv2.pyrUp(img6, borderType=cv2.BORDER_REPLICATE))
    # cv2 does not bind buildPyramid (CV_EXPORTS, not CV_EXPORTS_W); it is pyrDown applied
    # maxlevel times with Size() and the same borderType (pyramids.cpp, buildPyramid).
    pyr = [img8]
    for _ in range(3):
        pyr.append(cv2.pyrDown(pyr[-1]))
    p["build_pyramid_u8_8x8_levels"] = [tolist(level) for level in pyr]
    pyr9 = [img9]
    for _ in range(2):
        pyr9.append(cv2.pyrDown(pyr9[-1]))
    p["build_pyramid_u8_9x7_shapes"] = [list(level.shape) for level in pyr9]
    results["pyramids"] = p

    dump("imgproc_geom.py", results)


if __name__ == "__main__":
    main()

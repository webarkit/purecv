"""OpenCV 4.10.0 reference values for the imgproc colour and histogram rows
(docs/parity/4.10.0/imgproc.md).

Rows: cvt_color (RGB/BGR/RGBA/BGRA -> GRAY, GRAY -> RGB/BGR/RGBA/BGRA), calc_hist,
calc_back_project, compare_hist, equalize_hist, Clahe.

The RGB->GRAY section is exhaustive: every (r, g, b) triple goes through cvtColor once,
and the result is compared with the two formulas purecv uses (the f32 path without the
`simd` feature, the 77/150/29 fixed-point path with it). Every other input is defined by
a formula, so tests/parity_imgproc.rs can rebuild it.

Run from the repository root:
    cpp_ref/.venv/Scripts/python scripts/opencv_ref/imgproc_color_hist.py   # Windows
    cpp_ref/.venv/bin/python scripts/opencv_ref/imgproc_color_hist.py       # Linux/macOS
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


def pattern_u8_quadratic(rows, cols):
    """Deterministic test image with a skewed histogram: ((x*x + 3*y) * 37) % 256."""
    y, x = np.mgrid[0:rows, 0:cols]
    return (((x * x + 3 * y) * 37) % 256).astype(np.uint8)


def main() -> None:
    results = {}

    # --- cvtColor RGB -> GRAY, exhaustive over 2^24 triples
    rgb = np.zeros((4096, 4096, 3), np.uint8)
    idx = np.arange(1 << 24, dtype=np.uint32).reshape(4096, 4096)
    rgb[..., 0] = (idx >> 16) & 255
    rgb[..., 1] = (idx >> 8) & 255
    rgb[..., 2] = idx & 255
    gray_cv = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    r = rgb[..., 0].astype(np.float32)
    g = rgb[..., 1].astype(np.float32)
    b = rgb[..., 2].astype(np.float32)
    # purecv scalar path: (0.299 r + 0.587 g + 0.114 b).round() in f32; Rust's f32::round
    # rounds ties away from zero, i.e. floor(x + 0.5) for these non-negative values
    acc = np.float32(0.299) * r + np.float32(0.587) * g + np.float32(0.114) * b
    gray_f32 = np.floor(acc.astype(np.float64) + 0.5).astype(np.int32)
    ri = rgb[..., 0].astype(np.uint32)
    gi = rgb[..., 1].astype(np.uint32)
    bi = rgb[..., 2].astype(np.uint32)
    # purecv simd path: (77 r + 150 g + 29 b + 128) >> 8
    gray_simd = ((ri * 77 + gi * 150 + bi * 29 + 128) >> 8).astype(np.int32)
    # OpenCV 4.10.0 RGB2Gray<uchar>: (9798 r + 19235 g + 3735 b + 16384) >> 15
    # (RY15, GY15, BY15, gray_shift = 15 in color.simd_helpers.hpp)
    gray_formula = ((ri * 9798 + gi * 19235 + bi * 3735 + (1 << 14)) >> 15).astype(np.int32)
    gcv = gray_cv.astype(np.int32)
    diff_f32 = gcv != gray_f32
    diff_simd = gcv != gray_simd
    d_f32 = np.argwhere(diff_f32)
    d_simd = np.argwhere(diff_simd)
    both = np.argwhere(diff_f32 & diff_simd)
    neither = np.argwhere(~diff_f32 & ~diff_simd)

    def triples(pos, n=8):
        out = []
        for yy, xx in pos[:n]:
            out.append({"rgb": [int(rgb[yy, xx, 0]), int(rgb[yy, xx, 1]), int(rgb[yy, xx, 2])],
                        "opencv": int(gcv[yy, xx]), "purecv_f32": int(gray_f32[yy, xx]), "purecv_simd": int(gray_simd[yy, xx])})
        return out

    # a few spread-out matching triples (every 2^21-th index)
    spread = neither[:: max(1, len(neither) // 8)][:8]
    results["rgb2gray_exhaustive"] = {
        "total": int(1 << 24),
        "opencv_matches_fixed_point_formula": bool(np.array_equal(gcv, gray_formula)),
        "mismatch_f32_path": int(diff_f32.sum()),
        "mismatch_simd_path": int(diff_simd.sum()),
        "mismatch_both": int((diff_f32 & diff_simd).sum()),
        "max_abs_diff_f32_path": int(np.abs(gcv - gray_f32).max()),
        "max_abs_diff_simd_path": int(np.abs(gcv - gray_simd).max()),
        "examples_differ_both": triples(both),
        "examples_differ_f32_only": triples(np.argwhere(diff_f32 & ~diff_simd)),
        "examples_differ_simd_only": triples(np.argwhere(diff_simd & ~diff_f32)),
        "examples_match_both": triples(spread),
    }
    del rgb, gray_cv, r, g, b, gray_f32, ri, gi, bi, gray_simd, gray_formula, gcv, diff_f32, diff_simd

    # --- small images for the other colour rows
    px = np.array([[[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 255], [10, 20, 30], [128, 128, 129], [1, 2, 3], [200, 100, 50]]], np.uint8)
    pxa = np.dstack([px, np.full(px.shape[:2], 77, np.uint8)])
    c = {"src_rgb": tolist(px), "src_rgba": tolist(pxa)}
    c["rgb2gray"] = tolist(cv2.cvtColor(px, cv2.COLOR_RGB2GRAY))
    c["bgr2gray"] = tolist(cv2.cvtColor(px, cv2.COLOR_BGR2GRAY))
    c["rgba2gray"] = tolist(cv2.cvtColor(pxa, cv2.COLOR_RGBA2GRAY))
    c["bgra2gray"] = tolist(cv2.cvtColor(pxa, cv2.COLOR_BGRA2GRAY))
    gray = np.array([[0, 1, 127, 128, 254, 255]], np.uint8)
    c["src_gray"] = tolist(gray)
    c["gray2rgb"] = tolist(cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB))
    c["gray2bgr"] = tolist(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR))
    c["gray2rgba"] = tolist(cv2.cvtColor(gray, cv2.COLOR_GRAY2RGBA))
    c["gray2bgra"] = tolist(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGRA))
    c["rgb2gray_f32"] = tolist(cv2.cvtColor(px.astype(np.float32) / 255.0, cv2.COLOR_RGB2GRAY))
    c["rgb2gray_u16"] = tolist(cv2.cvtColor(px.astype(np.uint16) * 257, cv2.COLOR_RGB2GRAY))
    c["rgb2gray_wrong_channels"] = attempt(lambda: cv2.cvtColor(gray, cv2.COLOR_RGB2GRAY))
    c["gray2rgb_dstcn4"] = tolist(cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB, dstCn=4))
    c["codes"] = {
        "COLOR_BGR2GRAY": cv2.COLOR_BGR2GRAY, "COLOR_RGB2GRAY": cv2.COLOR_RGB2GRAY,
        "COLOR_GRAY2BGR": cv2.COLOR_GRAY2BGR, "COLOR_GRAY2RGB": cv2.COLOR_GRAY2RGB,
        "COLOR_GRAY2BGRA": cv2.COLOR_GRAY2BGRA, "COLOR_GRAY2RGBA": cv2.COLOR_GRAY2RGBA,
        "COLOR_BGRA2GRAY": cv2.COLOR_BGRA2GRAY, "COLOR_RGBA2GRAY": cv2.COLOR_RGBA2GRAY,
    }
    results["cvt_color"] = c

    # --- calcHist
    img8 = pattern_u8(8, 8)
    img8c3 = pattern_u8(6, 7, 3)
    img32 = img8.astype(np.float32)
    mask = np.zeros((8, 8), np.uint8)
    mask[2:6, 1:7] = 1
    h = {"src_u8": tolist(img8), "src_u8c3": tolist(img8c3), "mask": tolist(mask)}
    h["u8_8bins_0_256"] = tolist(cv2.calcHist([img8], [0], None, [8], [0, 256]).ravel())
    h["u8_8bins_0_256_mask"] = tolist(cv2.calcHist([img8], [0], mask, [8], [0, 256]).ravel())
    h["u8_5bins_0_256"] = tolist(cv2.calcHist([img8], [0], None, [5], [0, 256]).ravel())
    h["u8_4bins_50_200"] = tolist(cv2.calcHist([img8], [0], None, [4], [50, 200]).ravel())
    h["u8_4bins_50.5_200.5"] = tolist(cv2.calcHist([img8], [0], None, [4], [50.5, 200.5]).ravel())
    h["u8c3_ch1_8bins"] = tolist(cv2.calcHist([img8c3], [1], None, [8], [0, 256]).ravel())
    h["u8c3_ch0_ch2_4x4bins"] = tolist(cv2.calcHist([img8c3], [0, 2], None, [4, 4], [0, 256, 0, 256]))
    h["two_images_ch0_ch3_2x2bins"] = tolist(cv2.calcHist([img8c3, img8c3[:, :, 0].copy()], [0, 3], None, [2, 2], [0, 256, 0, 256]))
    h["f32_8bins_0_256"] = tolist(cv2.calcHist([img32], [0], None, [8], [0, 256]).ravel())
    h["f32_4bins_0_1_scaled"] = tolist(cv2.calcHist([img32 / 255.0], [0], None, [4], [0, 1]).ravel())
    h["f32_value_at_upper_bound"] = tolist(cv2.calcHist([np.array([[0.0, 0.5, 1.0, 1.5, 2.0]], np.float32)], [0], None, [2], [0, 2]).ravel())
    # cv2's calcHist binding has no `uniform` flag, so non-uniform bins cannot be measured here.
    h["u8_nonuniform_not_bound"] = attempt(lambda: cv2.calcHist([img8], [0], None, [3], [0, 64, 128, 256]))
    h["u8_accumulate_twice"] = tolist(cv2.calcHist([img8], [0], None, [8], [0, 256], cv2.calcHist([img8], [0], None, [8], [0, 256]), accumulate=True).ravel())
    h["u8_range_hi_lt_lo"] = attempt(lambda: cv2.calcHist([img8], [0], None, [8], [256, 0]))
    h["u8_0_bins"] = attempt(lambda: cv2.calcHist([img8], [0], None, [0], [0, 256]))
    results["calc_hist"] = h

    # --- calcBackProject
    hist8 = cv2.calcHist([img8], [0], None, [8], [0, 256])
    bp = {"hist8": tolist(hist8.ravel())}
    bp["u8_8bins_scale1"] = tolist(cv2.calcBackProject([img8], [0], hist8, [0, 256], 1.0))
    bp["u8_8bins_scale0.5"] = tolist(cv2.calcBackProject([img8], [0], hist8, [0, 256], 0.5))
    bp["u8_8bins_scale50_saturates"] = tolist(cv2.calcBackProject([img8], [0], hist8, [0, 256], 50.0))
    bp["u8_4bins_50_200_out_of_range_zero"] = tolist(cv2.calcBackProject([img8], [0], cv2.calcHist([img8], [0], None, [4], [50, 200]), [50, 200], 1.0))
    bp["u8_8bins_scale0.3"] = tolist(cv2.calcBackProject([img8], [0], hist8, [0, 256], 0.3))
    bp["f32_8bins_scale50"] = tolist(cv2.calcBackProject([img32], [0], hist8, [0, 256], 50.0))
    bp["f32_8bins_scale1"] = tolist(cv2.calcBackProject([img32], [0], hist8, [0, 256], 1.0))
    hist2 = cv2.calcHist([img8c3], [0, 2], None, [4, 4], [0, 256, 0, 256])
    bp["u8c3_ch0_ch2_scale1"] = tolist(cv2.calcBackProject([img8c3], [0, 2], hist2, [0, 256, 0, 256], 1.0))
    bp["dtype"] = str(cv2.calcBackProject([img8], [0], hist8, [0, 256], 1.0).dtype)
    bp["dtype_f32_src"] = str(cv2.calcBackProject([img32], [0], hist8, [0, 256], 1.0).dtype)
    results["calc_back_project"] = bp

    # --- compareHist
    h1 = np.array([1.0, 2.0, 3.0, 0.0, 5.0, 0.5], np.float32)
    h2 = np.array([1.5, 0.0, 3.0, 1.0, 4.0, 0.5], np.float32)
    ch = {"h1": tolist(h1), "h2": tolist(h2)}
    methods = {
        "correl": cv2.HISTCMP_CORREL, "chisqr": cv2.HISTCMP_CHISQR, "intersect": cv2.HISTCMP_INTERSECT,
        "bhattacharyya": cv2.HISTCMP_BHATTACHARYYA, "chisqr_alt": cv2.HISTCMP_CHISQR_ALT, "kl_div": cv2.HISTCMP_KL_DIV,
    }
    for name, m in methods.items():
        ch[name] = cv2.compareHist(h1, h2, m)
        ch[f"{name}_self"] = cv2.compareHist(h1, h1, m)
    ch["method_values"] = {name: int(m) for name, m in methods.items()}
    ch["correl_constant_hist"] = cv2.compareHist(np.full(4, 2.0, np.float32), np.full(4, 3.0, np.float32), cv2.HISTCMP_CORREL)
    ch["bhattacharyya_zero_hist"] = cv2.compareHist(np.zeros(4, np.float32), h1[:4].copy(), cv2.HISTCMP_BHATTACHARYYA)
    ch["size_mismatch"] = attempt(lambda: cv2.compareHist(h1, h1[:4].copy(), cv2.HISTCMP_CORREL))
    ch["method_6"] = attempt(lambda: cv2.compareHist(h1, h2, 6))
    results["compare_hist"] = ch

    # --- equalizeHist
    q16 = pattern_u8_quadratic(16, 16)
    e = {"src_u8_16x16": tolist(q16), "src_u8_8x8": tolist(img8)}
    e["u8_16x16"] = tolist(cv2.equalizeHist(q16))
    e["u8_8x8"] = tolist(cv2.equalizeHist(img8))
    e["u8_constant"] = tolist(cv2.equalizeHist(np.full((3, 3), 77, np.uint8)))
    e["u8_two_values"] = tolist(cv2.equalizeHist(np.array([[10, 10, 10, 200]], np.uint8)))
    # tie case: with total - hist[i] = 2 pixels the LUT value for the second pixel is 255/2 = 127.5
    e["u8_tie_half"] = tolist(cv2.equalizeHist(np.array([[0, 100, 200]], np.uint8)))
    e["u8_tie_half_b"] = tolist(cv2.equalizeHist(np.array([[0, 0, 100, 200, 200]], np.uint8)))
    # half-even tie: 7 distinct values, scale = 255/6 = 42.5, cumulative 1 -> 42.5 (-> 42), 5 -> 212.5 (-> 212)
    e["u8_tie_half_even"] = tolist(cv2.equalizeHist(np.array([[0, 10, 20, 30, 40, 50, 60]], np.uint8)))
    e["u8c3_rejected"] = attempt(lambda: cv2.equalizeHist(img8c3))
    e["u16_rejected"] = attempt(lambda: cv2.equalizeHist(q16.astype(np.uint16)))
    results["equalize_hist"] = e

    # --- CLAHE
    q32 = pattern_u8_quadratic(32, 32)
    q30 = pattern_u8_quadratic(30, 29)
    cl = {"src_u8_32x32": tolist(q32), "src_u8_30x29": tolist(q30), "src_u8_16x16": tolist(q16)}
    for clip, tiles in ((40.0, (8, 8)), (2.0, (4, 4)), (0.0, (2, 2)), (3.5, (3, 2))):
        clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=tiles)
        cl[f"u8_32x32_clip{clip}_tiles{tiles[0]}x{tiles[1]}"] = tolist(clahe.apply(q32))
        cl[f"u8_30x29_clip{clip}_tiles{tiles[0]}x{tiles[1]}"] = tolist(clahe.apply(q30))
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    cl["u8_16x16_clip2_tiles4x4"] = tolist(clahe.apply(q16))
    cl["u16_16x16_clip2_tiles4x4"] = tolist(clahe.apply(q16.astype(np.uint16) * 257))
    cl["u16_32x32_clip40_tiles8x8"] = tolist(cv2.createCLAHE(40.0, (8, 8)).apply(q32.astype(np.uint16) * 200))
    cl["defaults"] = [cv2.createCLAHE().getClipLimit(), list(cv2.createCLAHE().getTilesGridSize())]
    cl["u8c3_rejected"] = attempt(lambda: clahe.apply(img8c3))
    cl["f32_rejected"] = attempt(lambda: clahe.apply(img32))
    results["clahe"] = cl

    dump("imgproc_color_hist.py", results)


if __name__ == "__main__":
    main()

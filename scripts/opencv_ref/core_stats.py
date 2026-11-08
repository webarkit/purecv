"""OpenCV 4.10.0 reference values for the core statistics rows.

Rows: norm, normalize, mean, sum, mean_std_dev, min_max_loc, reduce, count_non_zero,
has_no_zero. See docs/parity/4.10.0/core.md.

Run from the repository root:
    cpp_ref/.venv/Scripts/python scripts/opencv_ref/core_stats.py   # Windows
    cpp_ref/.venv/bin/python scripts/opencv_ref/core_stats.py       # Linux/macOS
"""

import os
import sys

import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _reference import REFERENCE_VERSION  # noqa: E402

assert cv2.__version__ == REFERENCE_VERSION, f"cv2 {cv2.__version__} is not the pinned {REFERENCE_VERSION}"
cv2.ipp.setUseIPP(False)  # measure the generic CPU path

import numpy as np

from _core_common import dump, tolist  # noqa: E402


def attempt(fn):
    """Run fn(); return its value or the OpenCV error code as a string."""
    try:
        return tolist(fn())
    except cv2.error as e:  # noqa: PERF203
        return f"cv2.error: {e.code} {e.err}".strip()


# Fixtures of the #214 rows: one per depth purecv has (u8, i8, u16, i16, i32, f32, f64).
NORM_DEPTHS = {
    "u8": (np.uint8, [0, 255, 7, 128, 3, 200, 1, 90, 17, 64, 250, 33]),
    "i8": (np.int8, [-128, 127, -5, 0, 3, -100, 60, 1, -17, 64, -127, 33]),
    "u16": (np.uint16, [0, 65535, 7, 1000, 3, 40000, 1, 90, 17, 640, 60000, 33]),
    "i16": (np.int16, [-32768, 32767, -5, 0, 3, -1000, 600, 1, -17, 640, -30000, 33]),
    "i32": (np.int32, [-2000000000, 2147483647, -5, 0, 3, -1000, 600, 1, -17, 640, -30000, 33]),
    "f32": (np.float32, [3.0, -4.0, 0.5, 12.0, -1.5, 2.5, 1000.25, -7.25, 0.1, -0.3, 65.5, 2.0]),
    "f64": (np.float64, [3.0, -4.0, 0.5, 12.0, -1.5, 2.5, 1000.25, -7.25, 0.1, -0.3, 65.5, 2.0]),
}
# Second operand of norm(src1, src2): same depth, chosen so that src1 - src2 spans the range.
NORM_DEPTHS_B = {
    "u8": [255, 0, 7, 0, 250, 1, 90, 1, 64, 17, 0, 255],
    "i8": [127, -128, 5, -1, -3, 100, -60, 1, 17, -64, 127, -33],
    "u16": [65535, 0, 7, 0, 60000, 1, 90, 1, 640, 17, 0, 65535],
    "i16": [32767, -32768, 5, -1, -3, 1000, -600, 1, 17, -640, 30000, -33],
    # no difference leaves the int range: OpenCV subtracts in int (x86 wraps)
    "i32": [5, 5, 5, -1, -3, 1000, -600, 1, 17, -640, 30000, -33],
    "f32": [1.0, 1.0, -0.25, 12.0, 1.5, -2.5, 0.75, 7.25, 0.2, 0.3, -65.5, 2.0],
    "f64": [1.0, 1.0, -0.25, 12.0, 1.5, -2.5, 0.75, 7.25, 0.2, 0.3, -65.5, 2.0],
}
NORM_NAMES = {
    "inf": cv2.NORM_INF,
    "l1": cv2.NORM_L1,
    "l2": cv2.NORM_L2,
    "l2sqr": cv2.NORM_L2SQR,
    "hamming": cv2.NORM_HAMMING,
    "hamming2": cv2.NORM_HAMMING2,
}


def norm_types() -> dict:
    """#214: every norm type on every depth, masks on multi-channel input, Hamming,
    the two-array form, NORM_RELATIVE and empty input.

    Shapes: `single` is the 12 values as a 3x4 single-channel matrix, `multi` the same
    values as a 2x2 three-channel matrix. `pixel_mask` selects pixels (0, 0) and (1, 1)
    of the 2x2 matrix.
    """
    out = {}
    pixel_mask = np.array([[1, 0], [0, 1]], dtype=np.uint8)
    out["pixel_mask"] = tolist(pixel_mask)
    for name, (dt, vals) in NORM_DEPTHS.items():
        a1 = np.array(vals, dtype=dt).reshape(3, 4)
        a3 = np.array(vals, dtype=dt).reshape(2, 2, 3)
        b1 = np.array(NORM_DEPTHS_B[name], dtype=dt).reshape(3, 4)
        b3 = np.array(NORM_DEPTHS_B[name], dtype=dt).reshape(2, 2, 3)
        d = {"a": tolist(np.array(vals, dtype=dt)), "b": tolist(np.array(NORM_DEPTHS_B[name], dtype=dt))}
        for nn, nt in NORM_NAMES.items():
            d[f"single_{nn}"] = attempt(lambda: cv2.norm(a1, nt))
            d[f"multi_{nn}"] = attempt(lambda: cv2.norm(a3, nt))
            d[f"multi_masked_{nn}"] = attempt(lambda: cv2.norm(a3, nt, pixel_mask))
            d[f"diff_single_{nn}"] = attempt(lambda: cv2.norm(a1, b1, nt))
            d[f"diff_multi_{nn}"] = attempt(lambda: cv2.norm(a3, b3, nt))
            d[f"diff_multi_masked_{nn}"] = attempt(lambda: cv2.norm(a3, b3, nt, pixel_mask))
            d[f"relative_multi_{nn}"] = attempt(lambda: cv2.norm(a3, b3, nt | cv2.NORM_RELATIVE))
            d[f"relative_multi_masked_{nn}"] = attempt(
                lambda: cv2.norm(a3, b3, nt | cv2.NORM_RELATIVE, pixel_mask)
            )
        out[name] = d

    # The mask must be one 8-bit channel of the source's size.
    a3 = np.array(NORM_DEPTHS["f32"][1], dtype=np.float32).reshape(2, 2, 3)
    b3_f32 = np.array(NORM_DEPTHS_B["f32"], dtype=np.float32).reshape(2, 2, 3)
    out["mask_rules"] = {
        "mask_wrong_size": attempt(lambda: cv2.norm(a3, cv2.NORM_L2, np.ones((2, 3), np.uint8))),
        "mask_three_channel": attempt(lambda: cv2.norm(a3, cv2.NORM_L2, np.ones((2, 2, 3), np.uint8))),
        "mask_all_zero_l2": attempt(lambda: cv2.norm(a3, cv2.NORM_L2, np.zeros((2, 2), np.uint8))),
        "mask_all_zero_inf": attempt(lambda: cv2.norm(a3, cv2.NORM_INF, np.zeros((2, 2), np.uint8))),
        "mask_255_l1": attempt(lambda: cv2.norm(a3, cv2.NORM_L1, pixel_mask * 255)),
        "diff_mask_wrong_size": attempt(
            lambda: cv2.norm(a3, a3, cv2.NORM_L2, np.ones((2, 3), np.uint8))
        ),
        # an empty mask (0x0) is no mask in cv::norm (`mask.empty()`)
        "mask_empty_l2": attempt(lambda: cv2.norm(a3, cv2.NORM_L2, mask=np.zeros((0, 0), np.uint8))),
        "mask_empty_inf": attempt(lambda: cv2.norm(a3, cv2.NORM_INF, mask=np.zeros((0, 0), np.uint8))),
        "diff_mask_empty_l2": attempt(
            lambda: cv2.norm(a3, b3_f32, cv2.NORM_L2, mask=np.zeros((0, 0), np.uint8))
        ),
        "relative_mask_empty_l1": attempt(
            lambda: cv2.norm(a3, b3_f32, cv2.NORM_L1 | cv2.NORM_RELATIVE, mask=np.zeros((0, 0), np.uint8))
        ),
        "hamming_mask_empty": attempt(
            lambda: cv2.norm(
                np.array([[11, 255, 0, 128]], np.uint8), cv2.NORM_HAMMING, mask=np.zeros((0, 0), np.uint8)
            )
        ),
    }

    # Hamming: 8UC1 only; with a mask OpenCV takes bitwise_and(src, mask), so the mask values
    # act as bit masks (1 keeps bit 0 only, 255 keeps every bit). HAMMING2 counts non-zero
    # 2-bit cells (cellSize 2).
    bits = np.array([[0b1011, 0xFF, 0, 0b10000000], [0b01010101, 0b10101010, 0b11000011, 0x01]], dtype=np.uint8)
    bits_b = np.array([[0xFF, 0x0F, 0, 0b10000001], [0b10101010, 0b10101010, 0x3C, 0x80]], dtype=np.uint8)
    hmask = np.array([[255, 1, 0, 0x0F], [0xF0, 255, 3, 0]], dtype=np.uint8)
    out["hamming"] = {
        "a": tolist(bits),
        "b": tolist(bits_b),
        "mask": tolist(hmask),
        "hamming": cv2.norm(bits, cv2.NORM_HAMMING),
        "hamming2": cv2.norm(bits, cv2.NORM_HAMMING2),
        "hamming_masked": cv2.norm(bits, cv2.NORM_HAMMING, hmask),
        "hamming2_masked": cv2.norm(bits, cv2.NORM_HAMMING2, hmask),
        "diff_hamming": cv2.norm(bits, bits_b, cv2.NORM_HAMMING),
        "diff_hamming2": cv2.norm(bits, bits_b, cv2.NORM_HAMMING2),
        "diff_hamming_masked": cv2.norm(bits, bits_b, cv2.NORM_HAMMING, hmask),
        "diff_hamming2_masked": cv2.norm(bits, bits_b, cv2.NORM_HAMMING2, hmask),
        "relative_hamming": cv2.norm(bits, bits_b, cv2.NORM_HAMMING | cv2.NORM_RELATIVE),
        "u8_three_channel": attempt(lambda: cv2.norm(bits.reshape(2, 1, 4)[:, :, :3].copy(), cv2.NORM_HAMMING)),
        "u8_two_channel_diff": attempt(
            lambda: cv2.norm(bits.reshape(2, 2, 2), bits_b.reshape(2, 2, 2), cv2.NORM_HAMMING)
        ),
        "i8": attempt(lambda: cv2.norm(bits.astype(np.int8), cv2.NORM_HAMMING)),
        "u16": attempt(lambda: cv2.norm(bits.astype(np.uint16), cv2.NORM_HAMMING2)),
        "f32": attempt(lambda: cv2.norm(bits.astype(np.float32), cv2.NORM_HAMMING)),
        "diff_f32": attempt(
            lambda: cv2.norm(bits.astype(np.float32), bits_b.astype(np.float32), cv2.NORM_HAMMING)
        ),
    }

    # The two-array form: u8 at 0 - 255, mismatches, relative with src2 all zero, i32 extremes.
    u8a = np.array([[0, 255, 10, 200]], dtype=np.uint8)
    u8b = np.array([[255, 0, 10, 100]], dtype=np.uint8)
    zeros_u8 = np.zeros((1, 4), np.uint8)
    f32a = np.array([[3.0, -4.0, 0.0, 12.0, -1.5, 2.5]], dtype=np.float32)
    zeros_f32 = np.zeros((1, 6), np.float32)
    i32a = np.array([[-2147483648, 2147483647, 0]], dtype=np.int32)
    i32b = np.array([[2147483647, -2147483648, 0]], dtype=np.int32)
    out["diff"] = {
        "u8_a": tolist(u8a),
        "u8_b": tolist(u8b),
        **{f"u8_{nn}": attempt(lambda: cv2.norm(u8a, u8b, nt)) for nn, nt in NORM_NAMES.items()},
        "shape_mismatch": attempt(lambda: cv2.norm(u8a, np.zeros((1, 3), np.uint8), cv2.NORM_L2)),
        "transposed_shape": attempt(lambda: cv2.norm(u8a, u8a.reshape(4, 1).copy(), cv2.NORM_L2)),
        "type_mismatch": attempt(lambda: cv2.norm(u8a, u8a.astype(np.float32), cv2.NORM_L2)),
        "channel_mismatch": attempt(
            lambda: cv2.norm(np.zeros((2, 2), np.uint8), np.zeros((2, 2, 2), np.uint8), cv2.NORM_L2)
        ),
        "f32_a": tolist(f32a),
        **{
            f"relative_src2_zero_f32_{nn}": attempt(
                lambda: cv2.norm(f32a, zeros_f32, nt | cv2.NORM_RELATIVE)
            )
            for nn in ("inf", "l1", "l2", "l2sqr")
            for nt in [NORM_NAMES[nn]]
        },
        "relative_src2_zero_u8_hamming": attempt(
            lambda: cv2.norm(u8a, zeros_u8, cv2.NORM_HAMMING | cv2.NORM_RELATIVE)
        ),
        "relative_both_zero_f32_l2": attempt(
            lambda: cv2.norm(zeros_f32, zeros_f32, cv2.NORM_L2 | cv2.NORM_RELATIVE)
        ),
        "i32_a": tolist(i32a),
        "i32_b": tolist(i32b),
        **{f"i32_extremes_{nn}": attempt(lambda: cv2.norm(i32a, i32b, nt)) for nn, nt in list(NORM_NAMES.items())[:4]},
        "i32_int_min_single_inf": attempt(lambda: cv2.norm(i32a, cv2.NORM_INF)),
        "i32_int_min_single_l1": attempt(lambda: cv2.norm(i32a, cv2.NORM_L1)),
        # `abs(INT_MIN)` stays negative in `int`, so INT_MIN never becomes the Inf maximum
        "i32_only_int_min_inf": attempt(lambda: cv2.norm(np.array([[-(2**31)]], np.int32), cv2.NORM_INF)),
        "i32_only_int_min_l1": attempt(lambda: cv2.norm(np.array([[-(2**31)]], np.int32), cv2.NORM_L1)),
        "i32_int_min_5_0_inf": attempt(lambda: cv2.norm(np.array([[-(2**31), 5, 0]], np.int32), cv2.NORM_INF)),
        "i32_int_min_5_0_l1": attempt(lambda: cv2.norm(np.array([[-(2**31), 5, 0]], np.int32), cv2.NORM_L1)),
    }

    # Empty input (0x0 and 0x3).
    e00 = np.zeros((0, 0), np.uint8)
    e03 = np.zeros((0, 3), np.float32)
    out["empty"] = {
        **{f"u8_0x0_{nn}": attempt(lambda: cv2.norm(e00, nt)) for nn, nt in NORM_NAMES.items()},
        **{f"f32_0x3_{nn}": attempt(lambda: cv2.norm(e03, nt)) for nn, nt in list(NORM_NAMES.items())[:4]},
        "diff_f32_0x3_l2": attempt(lambda: cv2.norm(e03, e03, cv2.NORM_L2)),
        "diff_u8_0x0_hamming": attempt(lambda: cv2.norm(e00, e00, cv2.NORM_HAMMING)),
        "relative_f32_0x3_l2": attempt(lambda: cv2.norm(e03, e03, cv2.NORM_L2 | cv2.NORM_RELATIVE)),
    }
    return out


def main() -> None:
    results = {}

    # --- norm
    f = np.array([[3.0, -4.0, 0.0, 12.0, -1.5, 2.5]], dtype=np.float32)
    g = np.array([[1.0, 1.0, 1.0, 1.0, 1.0, 1.0]], dtype=np.float32)
    mask = np.array([[1, 0, 1, 0, 1, 0]], dtype=np.uint8)
    bits = np.array([[0b1011, 0xFF, 0, 0b10000000]], dtype=np.uint8)
    two_ch = np.array([[[3.0, -4.0], [0.0, 12.0]]], dtype=np.float32)
    results["norm"] = {
        "src": tolist(f),
        "inf": cv2.norm(f, cv2.NORM_INF),
        "l1": cv2.norm(f, cv2.NORM_L1),
        "l2": cv2.norm(f, cv2.NORM_L2),
        "l2sqr": cv2.norm(f, cv2.NORM_L2SQR),
        "l2_masked": cv2.norm(f, cv2.NORM_L2, mask),
        "inf_masked": cv2.norm(f, cv2.NORM_INF, mask),
        "l1_masked": cv2.norm(f, cv2.NORM_L1, mask),
        "mask": tolist(mask),
        "hamming_src": tolist(bits),
        "hamming": cv2.norm(bits, cv2.NORM_HAMMING),
        "hamming2": cv2.norm(bits, cv2.NORM_HAMMING2),
        "hamming_on_f32": attempt(lambda: cv2.norm(f, cv2.NORM_HAMMING)),
        "minmax_as_norm_type": attempt(lambda: cv2.norm(f, cv2.NORM_MINMAX)),
        "relative_l2_src2": tolist(g),
        "relative_l2": cv2.norm(f, g, cv2.NORM_RELATIVE | cv2.NORM_L2),
        "two_array_l2": cv2.norm(f, g, cv2.NORM_L2),
        "two_channel_src": tolist(two_ch),
        "two_channel_l2": cv2.norm(two_ch, cv2.NORM_L2),
        "two_channel_inf": cv2.norm(two_ch, cv2.NORM_INF),
    }

    results["norm_types"] = norm_types()

    # --- normalize
    s32 = np.array([[10.0, 20.0, 30.0, 250.0]], dtype=np.float32)
    s8 = np.array([[10, 20, 30, 250]], dtype=np.uint8)
    nmask = np.array([[1, 1, 0, 1]], dtype=np.uint8)
    dst_prefilled = np.full((1, 4), 7, dtype=np.uint8)
    cv2.normalize(s8, dst_prefilled, 0, 255, cv2.NORM_MINMAX, -1, nmask)
    results["normalize"] = {
        "f32_src": tolist(s32),
        "f32_minmax_0_255": tolist(cv2.normalize(s32, None, 0, 255, cv2.NORM_MINMAX)),
        "f32_minmax_255_0_swapped": tolist(cv2.normalize(s32, None, 255, 0, cv2.NORM_MINMAX)),
        "f32_minmax_0_1": tolist(cv2.normalize(s32, None, 0, 1, cv2.NORM_MINMAX)),
        "f32_minmax_to_u8": tolist(cv2.normalize(s32, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)),
        "f32_l2_alpha1": tolist(cv2.normalize(s32, None, 1, 0, cv2.NORM_L2)),
        "f32_l1_alpha1": tolist(cv2.normalize(s32, None, 1, 0, cv2.NORM_L1)),
        "f32_inf_alpha1": tolist(cv2.normalize(s32, None, 1, 0, cv2.NORM_INF)),
        "f32_l2_alpha10": tolist(cv2.normalize(s32, None, 10, 0, cv2.NORM_L2)),
        "u8_src": tolist(s8),
        "u8_minmax_0_255": tolist(cv2.normalize(s8, None, 0, 255, cv2.NORM_MINMAX)),
        "u8_minmax_0_100": tolist(cv2.normalize(s8, None, 0, 100, cv2.NORM_MINMAX)),
        "u8_minmax_0_1_to_f32": tolist(cv2.normalize(s8, None, 0, 1, cv2.NORM_MINMAX, cv2.CV_32F)),
        "u8_minmax_0_255_to_f32": tolist(cv2.normalize(s8, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_32F)),
        "u8_minmax_swapped_to_f32": tolist(cv2.normalize(s8, None, 1, 0, cv2.NORM_MINMAX, cv2.CV_32F)),
        "u8_l2_alpha255": tolist(cv2.normalize(s8, None, 255, 0, cv2.NORM_L2)),
        "u8_inf_alpha100": tolist(cv2.normalize(s8, None, 100, 0, cv2.NORM_INF)),
        "u8_mask": tolist(nmask),
        "u8_minmax_0_255_masked_dst_prefilled_7": tolist(dst_prefilled),
        "constant_src_minmax": tolist(
            cv2.normalize(np.array([[5.0, 5.0, 5.0]], dtype=np.float32), None, 0, 255, cv2.NORM_MINMAX)
        ),
        # a mask without elements (0x0) is no mask in cv::normalize (`mask.empty()`)
        "f32_l2_alpha1_mask_empty": attempt(
            lambda: tolist(cv2.normalize(s32, None, 1, 0, cv2.NORM_L2, -1, np.zeros((0, 0), np.uint8)))
        ),
        "l2sqr_as_norm_type": attempt(lambda: cv2.normalize(s32, None, 1, 0, cv2.NORM_L2SQR)),
        # `rtype` is reduced with CV_MAT_DEPTH, so the channel bits of a full type code are ignored.
        "f32_dtype_CV_32FC3": attempt(lambda: tolist(cv2.normalize(s32, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_32FC3))),
        "f32_dtype_CV_8UC3": attempt(lambda: tolist(cv2.normalize(s32, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC3))),
    }

    # --- mean / sum / meanStdDev
    bgr = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], dtype=np.uint8)
    f3 = np.array([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [[7.0, 8.0, 9.0], [10.0, 11.0, 13.0]]], dtype=np.float32)
    m22 = np.array([[1, 0], [1, 1]], dtype=np.uint8)
    mean_f3, std_f3 = cv2.meanStdDev(f3)
    mean_u8, std_u8 = cv2.meanStdDev(bgr)
    mean_m, std_m = cv2.meanStdDev(f3, mask=m22)
    five = np.zeros((1, 2, 5), dtype=np.float32)
    results["mean_sum_std"] = {
        "u8_3ch_src": tolist(bgr),
        "u8_sum": tolist(cv2.sumElems(bgr)),
        "u8_mean": tolist(cv2.mean(bgr)),
        "u8_mean_std": [tolist(mean_u8), tolist(std_u8)],
        "f32_3ch_src": tolist(f3),
        "f32_mean_std": [tolist(mean_f3), tolist(std_f3)],
        "mask": tolist(m22),
        "f32_mean_std_masked": [tolist(mean_m), tolist(std_m)],
        "five_channel_mean": attempt(lambda: cv2.mean(five)),
        "five_channel_sum": attempt(lambda: cv2.sumElems(five)),
        "mean_empty_0x3_u8": tolist(cv2.mean(np.zeros((0, 3), dtype=np.uint8))),
        "mean_empty_0x0_f32": tolist(cv2.mean(np.zeros((0, 0), dtype=np.float32))),
        "mean_empty_3x0_u8": tolist(cv2.mean(np.zeros((3, 0), dtype=np.uint8))),
    }

    # --- minMaxLoc
    mm = np.array(
        [[5.0, 1.0, 9.0, 1.0], [9.0, 2.0, 7.0, 3.0], [1.0, 8.0, 2.0, 9.0]], dtype=np.float32
    )
    mm_nan = mm.copy()
    mm_nan[0, 1] = np.nan
    mm_mask = np.ones(mm.shape, dtype=np.uint8)
    mm_mask[0, :] = 0
    results["min_max_loc"] = {
        "src": tolist(mm),
        "result": tolist(cv2.minMaxLoc(mm)),
        "mask": tolist(mm_mask),
        "masked": tolist(cv2.minMaxLoc(mm, mm_mask)),
        "nan_src": tolist(mm_nan),
        "nan_result": tolist(cv2.minMaxLoc(mm_nan)),
        "multi_channel": attempt(lambda: cv2.minMaxLoc(f3)),
        "u8_result": tolist(cv2.minMaxLoc(np.array([[3, 250], [0, 7]], dtype=np.uint8))),
    }

    # --- reduce
    r8 = np.array([[1, 2, 200], [2, 3, 100]], dtype=np.uint8)
    rf = r8.astype(np.float32)
    results["reduce"] = {
        "u8_src": tolist(r8),
        "u8_sum_dim0_default_dtype": attempt(lambda: cv2.reduce(r8, 0, cv2.REDUCE_SUM)),
        "u8_sum_dim0_i32": tolist(cv2.reduce(r8, 0, cv2.REDUCE_SUM, dtype=cv2.CV_32S)),
        "u8_avg_dim0": tolist(cv2.reduce(r8, 0, cv2.REDUCE_AVG)),
        "u8_avg_dim1": tolist(cv2.reduce(r8, 1, cv2.REDUCE_AVG)),
        "u8_max_dim0": tolist(cv2.reduce(r8, 0, cv2.REDUCE_MAX)),
        "u8_min_dim1": tolist(cv2.reduce(r8, 1, cv2.REDUCE_MIN)),
        "f32_sum_dim0": tolist(cv2.reduce(rf, 0, cv2.REDUCE_SUM)),
        "f32_avg_dim1": tolist(cv2.reduce(rf, 1, cv2.REDUCE_AVG)),
        "f32_sum2_dim0": tolist(cv2.reduce(rf, 0, cv2.REDUCE_SUM2)),
    }

    # depth pairs of reduce (#216): every accepted and rejected (source, destination) pair
    # the purecv tests use; REDUCE_AVG promotes a destination narrower than 32S to 32S.
    r16 = np.array([[1, 2, 60000], [2, 3, 60000]], dtype=np.uint16)
    s16 = np.array([[-5, 7, 30000], [4, -9, 30000]], dtype=np.int16)
    r32 = np.array([[1, 2, 300], [2, 3, 100]], dtype=np.int32)
    rd = r8.astype(np.float64)
    results["reduce"].update(
        {
            "u8_sum_dim1_i32": tolist(cv2.reduce(r8, 1, cv2.REDUCE_SUM, dtype=cv2.CV_32S)),
            "u8_sum2_dim0_i32": tolist(cv2.reduce(r8, 0, cv2.REDUCE_SUM2, dtype=cv2.CV_32S)),
            "u8_sum2_dim1_f64": tolist(cv2.reduce(r8, 1, cv2.REDUCE_SUM2, dtype=cv2.CV_64F)),
            "u8_sum2_dim0_default_dtype": attempt(lambda: cv2.reduce(r8, 0, cv2.REDUCE_SUM2)),
            "f32_sum2_dim1": tolist(cv2.reduce(rf, 1, cv2.REDUCE_SUM2)),
            "u16_sum_dim0_f32": tolist(cv2.reduce(r16, 0, cv2.REDUCE_SUM, dtype=cv2.CV_32F)),
            "u16_sum_dim0_u16": attempt(lambda: cv2.reduce(r16, 0, cv2.REDUCE_SUM)),
            "u16_sum_dim0_i32": attempt(lambda: cv2.reduce(r16, 0, cv2.REDUCE_SUM, dtype=cv2.CV_32S)),
            "u16_avg_dim0_u16": attempt(lambda: cv2.reduce(r16, 0, cv2.REDUCE_AVG)),
            "u16_max_dim0": tolist(cv2.reduce(r16, 0, cv2.REDUCE_MAX)),
            "i16_sum2_dim1_f64": tolist(cv2.reduce(s16, 1, cv2.REDUCE_SUM2, dtype=cv2.CV_64F)),
            "i16_min_dim0": tolist(cv2.reduce(s16, 0, cv2.REDUCE_MIN)),
            "i16_sum_dim0_i16": attempt(lambda: cv2.reduce(s16, 0, cv2.REDUCE_SUM)),
            "i32_max_dim0": attempt(lambda: cv2.reduce(r32, 0, cv2.REDUCE_MAX)),
            "i32_sum_dim0_i32": attempt(lambda: cv2.reduce(r32, 0, cv2.REDUCE_SUM)),
            "i32_sum_dim0_f64": attempt(lambda: cv2.reduce(r32, 0, cv2.REDUCE_SUM, dtype=cv2.CV_64F)),
            "f32_sum_dim0_u8": attempt(lambda: cv2.reduce(rf, 0, cv2.REDUCE_SUM, dtype=cv2.CV_8U)),
            "f32_max_dim0_f64": attempt(lambda: cv2.reduce(rf, 0, cv2.REDUCE_MAX, dtype=cv2.CV_64F)),
            "f64_sum_dim0_f32": attempt(lambda: cv2.reduce(rd, 0, cv2.REDUCE_SUM, dtype=cv2.CV_32F)),
            "u8_avg_dim0_i32": tolist(cv2.reduce(r8, 0, cv2.REDUCE_AVG, dtype=cv2.CV_32S)),
            "u8_avg_dim1_f32": tolist(cv2.reduce(r8, 1, cv2.REDUCE_AVG, dtype=cv2.CV_32F)),
            "u8_avg_dim0_u16": tolist(cv2.reduce(r8, 0, cv2.REDUCE_AVG, dtype=cv2.CV_16U)),
            "u8_avg_dim0_i8": tolist(cv2.reduce(r8, 0, cv2.REDUCE_AVG, dtype=cv2.CV_8S)),
            "f32_avg_dim1_f64": tolist(cv2.reduce(rf, 1, cv2.REDUCE_AVG, dtype=cv2.CV_64F)),
            "f32_avg_dim0_u8": attempt(lambda: cv2.reduce(rf, 0, cv2.REDUCE_AVG, dtype=cv2.CV_8U)),
            "f64_max_dim1": tolist(cv2.reduce(rd, 1, cv2.REDUCE_MAX)),
            "u8_3ch_sum_dim0_i32": tolist(
                cv2.reduce(np.stack([r8, r8 + 1, r8 + 2], axis=2), 0, cv2.REDUCE_SUM, dtype=cv2.CV_32S)
            ),
        }
    )

    # --- countNonZero / hasNonZero
    z = np.array([[0, 3, 0], [1, 0, 0]], dtype=np.uint8)
    results["non_zero"] = {
        "src": tolist(z),
        "count": cv2.countNonZero(z),
        "has_non_zero": bool(cv2.hasNonZero(z)),
        "has_non_zero_all_zero": bool(cv2.hasNonZero(np.zeros((2, 2), np.uint8))),
        "has_non_zero_all_nonzero": bool(cv2.hasNonZero(np.ones((2, 2), np.uint8))),
        "count_multi_channel": attempt(lambda: cv2.countNonZero(bgr)),
        "has_non_zero_multi_channel": attempt(lambda: cv2.hasNonZero(bgr)),
        "has_non_zero_empty": attempt(lambda: bool(cv2.hasNonZero(np.zeros((0, 3), np.uint8)))),
    }

    dump("core_stats.py", results)


if __name__ == "__main__":
    main()

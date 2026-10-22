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
        "u8_l2_alpha255": tolist(cv2.normalize(s8, None, 255, 0, cv2.NORM_L2)),
        "u8_inf_alpha100": tolist(cv2.normalize(s8, None, 100, 0, cv2.NORM_INF)),
        "u8_mask": tolist(nmask),
        "u8_minmax_0_255_masked_dst_prefilled_7": tolist(dst_prefilled),
        "constant_src_minmax": tolist(
            cv2.normalize(np.array([[5.0, 5.0, 5.0]], dtype=np.float32), None, 0, 255, cv2.NORM_MINMAX)
        ),
        "l2sqr_as_norm_type": attempt(lambda: cv2.normalize(s32, None, 1, 0, cv2.NORM_L2SQR)),
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
    }

    dump("core_stats.py", results)


if __name__ == "__main__":
    main()

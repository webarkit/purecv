"""OpenCV 4.10.0 reference values for the core conversion and u8 arithmetic rows.

Rows: Matrix::convert_to, add/subtract/multiply/divide (+ _scalar), add_weighted,
convert_scale_abs. See docs/parity/core.md.

Run from the repository root:
    cpp_ref/.venv/Scripts/python scripts/opencv_ref/core_convert_arith.py   # Windows
    cpp_ref/.venv/bin/python scripts/opencv_ref/core_convert_arith.py       # Linux/macOS

Python's cv2 does not bind Mat::convertTo. `cv2.addWeighted(src, alpha, 0, 0, beta, dtype)`
goes through arithm_op, which computes `src*alpha + beta` in the source float depth and
writes the destination through the same saturating convert kernel as convertTo
(saturate_cast: cvRound, round half to even, then clamp). It is used here as the
convertTo oracle; scripts/opencv_ref/core_convert_to.js measures Mat.convertTo itself.
"""

import cv2

assert cv2.__version__ == "4.10.0"
cv2.ipp.setUseIPP(False)  # measure the generic CPU path

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _core_common import dump, tolist  # noqa: E402

NAN, INF = float("nan"), float("inf")

# .5 ties of both parities, values just outside every integer range, NaN and infinities.
CONVERT_SRC = [
    0.5, 1.5, 2.5, 3.5, -0.5, -1.5, -2.5, 254.5, 254.6, 255.5, 300.0, -1.0, -300.0,
    127.5, 128.5, -128.5, -129.0, 65535.5, 32767.5, -32768.5, 1e10, -1e10, NAN, INF, -INF,
]
TARGETS = {"u8": cv2.CV_8U, "i8": cv2.CV_8S, "u16": cv2.CV_16U, "i16": cv2.CV_16S, "i32": cv2.CV_32S}

A_U8 = np.array([0, 1, 100, 127, 128, 200, 250, 255, 7, 5, 1, 9], dtype=np.uint8)
B_U8 = np.array([0, 1, 3, 100, 128, 255, 7, 2, 2, 2, 3, 4], dtype=np.uint8)


def convert_to(src: np.ndarray, dtype: int, alpha: float = 1.0, beta: float = 0.0) -> np.ndarray:
    zero = np.zeros_like(src)
    return cv2.addWeighted(src, alpha, zero, 0.0, beta, dtype=dtype)


def main() -> None:
    results = {}

    # --- convert_to: f32 and f64 sources, every integer target, with and without alpha/beta
    for src_name, src_dtype in (("f32", np.float32), ("f64", np.float64)):
        src = np.array(CONVERT_SRC, dtype=src_dtype)
        block = {"src": tolist(src)}
        for tname, tcode in TARGETS.items():
            block[tname] = tolist(convert_to(src, tcode))
            block[f"{tname}_alpha2_beta0.5"] = tolist(convert_to(src, tcode, 2.0, 0.5))
        results[f"convert_to_from_{src_name}"] = block
    # integer -> narrower integer saturates too (no rounding involved)
    src_i32 = np.array(
        [-70000, -32769, -32768, -129, -128, -1, 0, 1, 127, 128, 255, 256, 65535, 65536, 70000],
        dtype=np.int32,
    )
    results["convert_to_from_i32"] = {
        "src": tolist(src_i32),
        **{t: tolist(convert_to(src_i32, c)) for t, c in TARGETS.items() if t != "i32"},
    }
    # u8 -> f32 is exact
    results["convert_to_u8_to_f32"] = {"src": tolist(A_U8), "f32": tolist(convert_to(A_U8, cv2.CV_32F))}

    # --- u8 element-wise arithmetic (saturating; divide rounds half to even; x/0 = 0)
    results["u8_binary"] = {
        "a": tolist(A_U8),
        "b": tolist(B_U8),
        "add": tolist(cv2.add(A_U8, B_U8)),
        "subtract": tolist(cv2.subtract(A_U8, B_U8)),
        "subtract_b_a": tolist(cv2.subtract(B_U8, A_U8)),
        "multiply": tolist(cv2.multiply(A_U8, B_U8)),
        "multiply_scale_0.5": tolist(cv2.multiply(A_U8, B_U8, scale=0.5)),
        "multiply_scale_third": tolist(cv2.multiply(A_U8, B_U8, scale=1.0 / 3.0)),
        "divide": tolist(cv2.divide(A_U8, B_U8)),
        "divide_scale_2": tolist(cv2.divide(A_U8, B_U8, scale=2.0)),
        "divide_scalar_2_over_b": tolist(cv2.divide(2.0, B_U8)),
        "absdiff": tolist(cv2.absdiff(A_U8, B_U8)),
        "min": tolist(cv2.min(A_U8, B_U8)),
        "max": tolist(cv2.max(A_U8, B_U8)),
    }
    # f32 divide by zero is IEEE (inf/nan), not 0
    a32 = np.array([1.0, -1.0, 0.0, 4.0], dtype=np.float32)
    b32 = np.array([0.0, 0.0, 0.0, 2.0], dtype=np.float32)
    results["f32_divide_by_zero"] = {"a": tolist(a32), "b": tolist(b32), "divide": tolist(cv2.divide(a32, b32))}

    # --- matrix + scalar on u8: the scalar is saturate_cast to the working type first
    results["u8_scalar"] = {
        "a": tolist(A_U8),
        "add_10.7": tolist(cv2.add(A_U8, (10.7, 0, 0, 0))),
        "add_10.5": tolist(cv2.add(A_U8, (10.5, 0, 0, 0))),
        "add_300": tolist(cv2.add(A_U8, (300.0, 0, 0, 0))),
        "add_neg_1.5": tolist(cv2.add(A_U8, (-1.5, 0, 0, 0))),
        "subtract_10.7": tolist(cv2.subtract(A_U8, (10.7, 0, 0, 0))),
        "multiply_0.3": tolist(cv2.multiply(A_U8, (0.3, 0, 0, 0))),
        "multiply_1.5": tolist(cv2.multiply(A_U8, (1.5, 0, 0, 0))),
        "divide_3": tolist(cv2.divide(A_U8, (3.0, 0, 0, 0))),
        "divide_0": tolist(cv2.divide(A_U8, (0.0, 0, 0, 0))),
        "divide_0.4": tolist(cv2.divide(A_U8, (0.4, 0, 0, 0))),
    }

    # --- add_weighted on u8 and f32
    results["add_weighted"] = {
        "a": tolist(A_U8),
        "b": tolist(B_U8),
        "u8_0.5_0.5_0": tolist(cv2.addWeighted(A_U8, 0.5, B_U8, 0.5, 0.0)),
        "u8_0.3_0.7_10.5": tolist(cv2.addWeighted(A_U8, 0.3, B_U8, 0.7, 10.5)),
        "u8_2_2_0": tolist(cv2.addWeighted(A_U8, 2.0, B_U8, 2.0, 0.0)),
        "u8_1_-1_0": tolist(cv2.addWeighted(A_U8, 1.0, B_U8, -1.0, 0.0)),
        "f32_0.5_0.5_0": tolist(
            cv2.addWeighted(A_U8.astype(np.float32), 0.5, B_U8.astype(np.float32), 0.5, 0.0)
        ),
    }

    # --- convert_scale_abs
    src_i16 = np.array([-300, -2, -1, 0, 1, 2, 127, 128, 254, 255, 256, 1000], dtype=np.int16)
    src_f32 = np.array(
        [0.5, 1.5, 2.5, -0.5, -1.5, -2.5, 254.5, 255.5, -255.5, 300.0, 1e10, NAN], dtype=np.float32
    )
    results["convert_scale_abs"] = {
        "i16_src": tolist(src_i16),
        "i16_1_0": tolist(cv2.convertScaleAbs(src_i16, alpha=1.0, beta=0.0)),
        "i16_0.5_0": tolist(cv2.convertScaleAbs(src_i16, alpha=0.5, beta=0.0)),
        "f32_src": tolist(src_f32),
        "f32_1_0": tolist(cv2.convertScaleAbs(src_f32, alpha=1.0, beta=0.0)),
        "f32_0.5_0.25": tolist(cv2.convertScaleAbs(src_f32, alpha=0.5, beta=0.25)),
        "u8_src": tolist(A_U8),
        "u8_2_-100": tolist(cv2.convertScaleAbs(A_U8, alpha=2.0, beta=-100.0)),
    }

    dump("core_convert_arith.py", results)


if __name__ == "__main__":
    main()

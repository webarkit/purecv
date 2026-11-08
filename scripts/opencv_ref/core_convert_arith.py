"""OpenCV 4.10.0 reference values for the core conversion and u8 arithmetic rows.

Rows: Matrix::convert_to, add/subtract/multiply/divide (+ _scalar), add_weighted,
convert_scale_abs. See docs/parity/4.10.0/core.md.

Run from the repository root:
    cpp_ref/.venv/Scripts/python scripts/opencv_ref/core_convert_arith.py   # Windows
    cpp_ref/.venv/bin/python scripts/opencv_ref/core_convert_arith.py       # Linux/macOS

Python's cv2 does not bind Mat::convertTo. `cv2.addWeighted(src, alpha, 0, 0, beta, dtype)`
goes through arithm_op, which computes `src*alpha + beta` in the source float depth and
writes the destination through the same saturating convert kernel as convertTo
(saturate_cast: cvRound, round half to even, then clamp). It is used here as the
convertTo oracle; scripts/opencv_ref/core_convert_to.js measures Mat.convertTo itself.
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

    # divide on the other depths (op_div_scale: float working type, x/0 = 0 for the integers)
    div_i = {
        "i8": (np.int8, [127, -128, -128, 127, 7, -7, 7, -7, 100, -100, 5, 0, 1, -128, 127],
               [2, -1, 3, -1, 2, 2, -2, -2, 0, 0, 7, 0, -128, 127, 127]),
        "u16": (np.uint16, [65535, 1, 7, 100, 65535, 0, 3, 40000, 65535, 5, 9],
                [1, 3, 2, 0, 65535, 5, 7, 3, 2, 0, 4]),
        "i16": (np.int16, [32767, -32768, -32768, 7, -7, 7, -7, 1, 100, 0, -32768, 5, 32767],
                [2, -1, 3, 2, 2, -2, -2, 3, 0, 5, 32767, 0, -32768]),
        "i32": (np.int32, [7, -7, 7, -7, 100, 0, 16777217, -2147483648,
                           2147483647, 1000000007, -5, 3, 9, 10],
                [2, 2, -2, -2, 0, 3, 1, 1, 2, 3, 0, 2, 3, 4]),
    }
    results["divide_ints"] = {}
    for name, (dt, av, bv) in div_i.items():
        a = np.array(av, dtype=dt).reshape(1, -1)
        b = np.array(bv, dtype=dt).reshape(1, -1)
        results["divide_ints"][name] = {"a": tolist(a), "b": tolist(b), "divide": tolist(cv2.divide(a, b))}
    # CV_32S quotients of exactly 2^31: saturate_cast<int>(float) gives INT_MIN here, the x86
    # conversion of an out-of-range float (not a clamp); purecv clamps instead.
    ea = np.array([2147483647, -2147483648], dtype=np.int32).reshape(1, -1)
    eb = np.array([1, -1], dtype=np.int32).reshape(1, -1)
    results["divide_i32_2pow31"] = {"a": tolist(ea), "b": tolist(eb), "divide": tolist(cv2.divide(ea, eb))}
    # a 1 x 64 u8 fixture from a seeded generator, zeros in the divisor
    rs = np.random.RandomState(209)
    sa = rs.randint(0, 256, size=(1, 64)).astype(np.uint8)
    sb = rs.randint(0, 40, size=(1, 64)).astype(np.uint8)
    results["divide_u8_seeded"] = {"a": tolist(sa), "b": tolist(sb), "divide": tolist(cv2.divide(sa, sb))}
    # f32 / f64 with inf and nan in every position of a 64-element row (SIMD lanes)
    lane = np.arange(64, dtype=np.float64)
    fa = np.where(lane % 5 == 0, 0.0, lane - 31.5).reshape(1, -1)
    fb = np.where(lane % 3 == 0, 0.0, 0.25 * (lane - 20.0)).reshape(1, -1)
    results["divide_float_lanes"] = {
        "a": tolist(fa), "b": tolist(fb),
        "f32": tolist(cv2.divide(fa.astype(np.float32), fb.astype(np.float32))),
        "f64": tolist(cv2.divide(fa, fb)),
    }

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

    # --- matrix (+ - * /) scalar on the other depths (arithm_op: with a scalar operand the
    # working type is CV_32S for the add/subtract of every integer depth below 32F, and
    # CV_64F for multiply/divide of every depth below 64F)
    scal_src = {
        "i8": np.array([-128, -100, -1, 0, 1, 50, 100, 127], dtype=np.int8),
        "u16": np.array([0, 1, 100, 1000, 32768, 60000, 65534, 65535], dtype=np.uint16),
        "i16": np.array([-32768, -30000, -1, 0, 1, 1000, 30000, 32767], dtype=np.int16),
        "i32": np.array([-2147483648, -2147483647, -100000, -1, 0, 1, 100000, 2147483646, 2147483647], dtype=np.int32),
        "f32": np.array([-1e10, -300.5, -1.5, -0.25, 0.0, 0.1, 1.0, 2.5, 255.5, 1e10], dtype=np.float32),
    }
    results["other_depths_scalar"] = {}
    for name, a in scal_src.items():
        blk = {"a": tolist(a)}
        for label, sc in (("10.7", 10.7), ("10.5", 10.5), ("300", 300.0), ("neg_1.5", -1.5),
                          ("1e5", 1e5), ("neg_70000", -70000.0), ("3e9", 3e9)):
            blk[f"add_{label}"] = tolist(cv2.add(a, (sc, 0, 0, 0)))
            blk[f"subtract_{label}"] = tolist(cv2.subtract(a, (sc, 0, 0, 0)))
        for label, sc in (("0.3", 0.3), ("1.5", 1.5), ("2", 2.0), ("0.1", 0.1), ("neg_0.5", -0.5), ("1e5", 1e5)):
            blk[f"multiply_{label}"] = tolist(cv2.multiply(a, (sc, 0, 0, 0)))
        for label, sc in (("3", 3.0), ("0.4", 0.4), ("0", 0.0), ("neg_2.5", -2.5), ("0.1", 0.1)):
            blk[f"divide_{label}"] = tolist(cv2.divide(a, (sc, 0, 0, 0)))
        results["other_depths_scalar"][name] = blk
    # f64 source: the scalar stays double
    a64 = np.array([-1e10, -0.1, 0.0, 0.1, 1.0, 1e10], dtype=np.float64)
    results["other_depths_scalar"]["f64"] = {
        "a": tolist(a64),
        "add_0.1": tolist(cv2.add(a64, (0.1, 0, 0, 0))),
        "multiply_0.1": tolist(cv2.multiply(a64, (0.1, 0, 0, 0))),
        "divide_0": tolist(cv2.divide(a64, (0.0, 0, 0, 0))),
        "divide_3": tolist(cv2.divide(a64, (3.0, 0, 0, 0))),
    }
    # per-channel scalar on a 1 x 3 pixel, 3 channel u8 image
    img3 = np.array([[[0, 100, 250], [10, 200, 255], [255, 1, 7]]], dtype=np.uint8)
    results["u8c3_scalar"] = {
        "a": tolist(img3.reshape(-1)),
        "add_10.7_neg_1.5_300": tolist(cv2.add(img3, (10.7, -1.5, 300.0, 0)).reshape(-1)),
        "multiply_0.3_1.5_2": tolist(cv2.multiply(img3, (0.3, 1.5, 2.0, 0)).reshape(-1)),
        "divide_3_0.4_0": tolist(cv2.divide(img3, (3.0, 0.4, 0.0, 0)).reshape(-1)),
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

    # add_weighted on a 1 x 64 u8 fixture from a seeded generator (SIMD lanes and tail)
    rs = np.random.RandomState(211)
    wa = rs.randint(0, 256, size=(1, 64)).astype(np.uint8)
    wb = rs.randint(0, 256, size=(1, 64)).astype(np.uint8)
    results["add_weighted_u8_seeded"] = {
        "a": tolist(wa),
        "b": tolist(wb),
        "0.3_0.7_10.5": tolist(cv2.addWeighted(wa, 0.3, wb, 0.7, 10.5)),
        "0.5_0.5_0": tolist(cv2.addWeighted(wa, 0.5, wb, 0.5, 0.0)),
    }

    # add_weighted on the other depths: ties, saturation, and the working type per depth
    # (float for the 8/16-bit depths, double for CV_32S, CV_32F and CV_64F)
    wd_src = {
        "i8": (np.int8, [-128, -127, 127, 100, 1, -1, 0, 3, -100, 50],
               [-128, 126, 127, 100, 2, -2, 0, 4, -100, -60]),
        "u16": (np.uint16, [0, 1, 65535, 65535, 3, 1000, 40000, 65534, 5, 32768],
                [0, 2, 65535, 1, 4, 2001, 40001, 65535, 7, 32767]),
        "i16": (np.int16, [-32768, -32767, 32767, 32767, 1, -1, 0, 3, -30000, 1234],
                [-32768, 32766, 32767, 1, 2, -2, 0, 4, -30000, -4321]),
        # every result stays inside the int range: beyond it OpenCV returns the x86 artefact
        "i32": (np.int32, [-500000000, 500000000, 1, 3, -1, 0, 7, 1000000000, 12345678, -5],
                [-500000000, 499999999, 2, 4, -2, 0, 8, -1000000000, 98765432, 10]),
        "f32": (np.float32, [-1e6, -300.5, -1.5, -0.25, 0.0, 0.1, 1.0, 2.5, 255.5, 1e6],
                [1e6, 300.5, 0.5, 0.75, 0.0, 0.2, 3.0, 4.5, -0.5, 1e6]),
        # (1.0, 3.0) is left out: with (0.3, 0.7, 10.5) the wheel's fused vector loop gives
        # 12.9 where the unfused loop of the source gives 12.899999999999999
        "f64": (np.float64, [-1e10, -0.1, 0.0, 0.1, 2.0, 1e10, 2.5, 0.3],
                [1e10, 0.7, 0.0, 0.2, 4.0, 1e10, 4.5, 0.6]),
    }
    results["add_weighted_other_depths"] = {}
    for name, (dt, av, bv) in wd_src.items():
        a = np.array(av, dtype=dt).reshape(1, -1)
        b = np.array(bv, dtype=dt).reshape(1, -1)
        blk = {"a": tolist(a), "b": tolist(b)}
        for label, (al, be, ga) in (("0.5_0.5_0", (0.5, 0.5, 0.0)), ("2_2_0", (2.0, 2.0, 0.0)),
                                    ("0.3_0.7_10.5", (0.3, 0.7, 10.5)), ("1_-1_0", (1.0, -1.0, 0.0))):
            blk[label] = tolist(cv2.addWeighted(a, al, b, be, ga))
        results["add_weighted_other_depths"][name] = blk
    # 1 x 64 seeded i32 and f32 rows (double working type, SIMD lanes and tail). The wheel's
    # vector loop fuses a*alpha + (b*beta + gamma) with FMA, so a result that sits within one
    # rounding step of a tie or of the float grid can differ from the unfused generic loop in
    # the source; only scalars whose 64 results agree with both are recorded.
    rs = np.random.RandomState(212)
    ia = rs.randint(-2**30, 2**30, size=(1, 64)).astype(np.int32)
    ib = rs.randint(-2**30, 2**30, size=(1, 64)).astype(np.int32)
    fa = (rs.rand(1, 64) * 1000 - 500).astype(np.float32)
    fb = (rs.rand(1, 64) * 1000 - 500).astype(np.float32)
    results["add_weighted_seeded_wide"] = {
        "i32_a": tolist(ia), "i32_b": tolist(ib),
        "i32_0.5_0.5_0": tolist(cv2.addWeighted(ia, 0.5, ib, 0.5, 0.0)),
        "f32_a": tolist(fa), "f32_b": tolist(fb),
        "f32_0.3_0.7_10.5": tolist(cv2.addWeighted(fa, 0.3, fb, 0.7, 10.5)),
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

    # The other source depths. convert_scale.simd.hpp:220-226 computes every depth, f64
    # included, with cvtabs_32f: the source is converted to float, then |x*a + b| is saturated.
    # The wheel's vector loop fuses the multiply-add (v_fma) and the generic scalar tail does
    # not, so the scalars below (0.5, 2, 1, -1 with betas 0, 0.25, -3) keep x*a exact in float
    # and the two agree. Inputs beyond 2**31 after scaling are left out: the x86 conversion of
    # such a float gives INT_MIN and so 0, which purecv does not port (it clamps to 255).
    cs = {}
    for name, dt, vals in (
        ("i8", np.int8, [-128, -127, -3, -2, -1, 0, 1, 2, 3, 127]),
        ("u16", np.uint16, [0, 1, 2, 3, 5, 255, 256, 511, 1000, 65535]),
        ("i16", np.int16, [-32768, -32767, -300, -3, -1, 0, 1, 3, 255, 32767]),
        ("i32", np.int32, [-2**29, -2**24 - 1, -3, -1, 0, 1, 3, 255, 2**24 + 1, 2**29]),
        ("f32", np.float32, [-3.5, -2.5, -0.5, 0.5, 1.5, 2.5, 3.5, 254.5, 255.5, 300.0]),
        ("f64", np.float64, [-2.5, -0.5, 0.5, 1.5, 2.5, 3.5, -1.5, 254.5, 300.0, NAN]),
    ):
        arr = np.array(vals, dtype=dt).reshape(1, -1)
        blk = {"src": tolist(arr)}
        for label, (al, be) in (("1_0", (1.0, 0.0)), ("0.5_0", (0.5, 0.0)), ("0.5_0.25", (0.5, 0.25)),
                                ("-1_-3", (-1.0, -3.0)), ("2_0", (2.0, 0.0))):
            blk[label] = tolist(cv2.convertScaleAbs(arr, alpha=al, beta=be))
        cs[name] = blk
    results["convert_scale_abs_other_depths"] = cs

    # 1 x 64 rows of every half step from -10 to 21.5: whole SIMD vectors, ties of both parities.
    half = (np.arange(64) - 20) * 0.5
    wide = {}
    for name, dt in (("f32", np.float32), ("f64", np.float64)):
        arr = half.astype(dt).reshape(1, -1)
        wide[name + "_src"] = tolist(arr)
        wide[name + "_1_0"] = tolist(cv2.convertScaleAbs(arr, alpha=1.0, beta=0.0))
        wide[name + "_0.5_0.25"] = tolist(cv2.convertScaleAbs(arr, alpha=0.5, beta=0.25))
    # f64 inputs 1e-10 and 1e-8 above a tie are not f32 values: the vector loop narrows them to
    # float first (0.5000000001 -> 0.5, a tie, -> 0; 254.50000001 -> 254.5 -> 254). The scalar
    # tail, used only for rows narrower than one vector, multiplies in double and gives 1 and
    # 255 instead; a 64-wide row stays on the vector loop.
    narrow = np.array([0.5000000001, 254.50000001] * 32, dtype=np.float64).reshape(1, -1)
    wide["f64_narrow_src"] = tolist(narrow)
    wide["f64_narrow_1_0"] = tolist(cv2.convertScaleAbs(narrow, alpha=1.0, beta=0.0))
    results["convert_scale_abs_ties_wide"] = wide

    dump("core_convert_arith.py", results)


if __name__ == "__main__":
    main()

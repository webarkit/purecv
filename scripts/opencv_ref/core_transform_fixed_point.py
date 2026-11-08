"""OpenCV 4.10.0 reference values for `cv::transform` on u8 3 -> 3, the fixed-point kernel.

`transform_8u` (modules/core/src/matmul.simd.hpp#L1442-L1522) runs in 10-bit fixed point when
`scn == dcn == 3` and, on every row, the first two coefficients are below 32 and the third
coefficient and the bias below 8192 in magnitude; a diagonal matrix never gets there
(`diagtransform`). Anything else takes the generic `float` kernel.

The script checks an emulation of that rule against cv2 for several row widths, so that vector
lanes and the scalar tail are both covered, and prints the fixtures of
`tests/parity_core.rs::transform_u8_3to3_*`.

Run from the repository root:
    cpp_ref/.venv/Scripts/python scripts/opencv_ref/core_transform_fixed_point.py   # Windows
    cpp_ref/.venv/bin/python scripts/opencv_ref/core_transform_fixed_point.py       # Linux/macOS
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

SCALE = np.float32(1024.0)
HALF = np.float32(0.5)
WIDTHS = (1, 3, 16, 17, 31, 32, 33, 37, 64, 65, 4096)


def cv_round(v):
    """`saturate_cast<int>(float)`: nearest, ties to even (the input is below 2^31 here)."""
    return int(np.rint(np.float32(v)))


def as_3x4(m):
    """The `float` matrix `cv::transform` hands to the kernel, with a zero bias if missing."""
    full = np.zeros((3, 4), dtype=np.float32)
    full[:, : m.shape[1]] = np.asarray(m, dtype=np.float64).astype(np.float32)
    return full


def takes_fixed_point(m):
    """The test on line 1448, for a matrix that is not diagonal."""
    a = np.abs(as_3x4(m))
    return bool(np.all(a[:, :2] < 32.0) and np.all(a[:, 2:] < 8192.0))


def fixed_point(src, m):
    """The scalar tail of `transform_8u` (lines 1505-1515), with `int` wrapping on overflow."""
    full = as_3x4(m)
    s = src.astype(np.int64)
    out = np.zeros(src.shape, dtype=np.uint8)
    for j in range(3):
        c0, c1 = (int(np.clip(cv_round(full[j, k] * SCALE), -32768, 32767)) for k in (0, 1))
        c2 = cv_round(full[j, 2] * SCALE)
        bias = cv_round((full[j, 3] + HALF) * SCALE)
        acc = c0 * s[..., 0] + c1 * s[..., 1] + c2 * s[..., 2] + bias
        wrapped = (acc + 2**31) % 2**32 - 2**31
        out[..., j] = np.clip(wrapped >> 10, 0, 255)
    return out


def generic_float(src, m):
    """The scalar `transform_` for 3 -> 3: `float` sum, bias last, `saturate_cast<uchar>`."""
    full = as_3x4(m)
    s = src.astype(np.float32)
    out = np.zeros(src.shape, dtype=np.uint8)
    for j in range(3):
        acc = full[j, 0] * s[..., 0]
        acc = acc + full[j, 1] * s[..., 1]
        acc = acc + full[j, 2] * s[..., 2]
        acc = acc + full[j, 3]
        out[..., j] = np.clip(np.rint(acc.astype(np.float64)), 0, 255)
    return out


def check_widths(rs_seed, m, emulate):
    """Whether cv2 equals `emulate` on every element, per row width."""
    rs = np.random.RandomState(rs_seed)
    wide = rs.randint(0, 256, size=(1, max(WIDTHS), 3)).astype(np.uint8)
    report = {}
    for w in WIDTHS:
        src = wide[:, :w]
        got = cv2.transform(src, m)
        report[str(w)] = int(np.count_nonzero(got.astype(int) != emulate(src, m).astype(int)))
    return report


# Each matrix either stays on the fixed-point kernel or leaves it for the generic one.
MATRICES = {
    "bt601_luma": [[0.299, 0.587, 0.114]] * 3,
    "generic_small": [[0.3, 0.3, 0.3], [0.5, -0.25, 0.75], [1.1, 0.2, -0.3]],
    "ties_half": [[0.5, 0.5, 0.0], [0.25, 0.25, 0.5], [0.5, 0.0, 0.5]],
    "affine_bias": [[0.25, 0.5, -0.125, 10.3], [-0.7, 1.4, 0.2, -20.6], [1.25, -0.5, 0.33, 0.25]],
    "r_coef_50_bias_100": [[0.1, 0.2, 50.0, -3000.5], [0.5, 0.5, 0.0, 100.25], [0.0, 0.5, 0.25, -100.75]],
    # column 0 or 1 at 32 leaves the fixed-point kernel; the odd sums below are ties
    "b_coef_32": [[32.0, 0.5, 0.0, 0.0], [0.5, 0.5, 0.0, 0.0], [0.0, 0.5, 0.5, 0.0]],
    "g_coef_40": [[0.5, 40.0, 0.5, -5000.0], [0.5, 0.5, 0.0, 0.0], [0.0, 0.5, 0.5, 0.0]],
    # column 2 or 3 at 8192 leaves it too
    "r_coef_8192": [[0.5, 0.0, 8192.0, -8192.0], [0.5, 0.5, 0.0, 0.0], [0.0, 0.5, 0.5, 0.0]],
    "bias_8192": [[0.5, 0.5, 0.0, 8192.0], [0.5, 0.5, 0.0, -8192.0], [0.5, 0.0, 0.5, 0.0]],
}


def main() -> None:
    rs = np.random.RandomState(230)
    # 37 pixels: whole vectors of every width up to 32 bytes per channel, then a tail.
    src = rs.randint(0, 256, size=(1, 37, 3)).astype(np.uint8)
    # r == 1 makes the 8192 row of `r_coef_8192` land on the small sums where ties show.
    src_r1 = src.copy()
    src_r1[..., 2] = 1
    # BT.601 luma: the kernels rarely disagree, so lead with pixels from a longer draw that do.
    luma = np.array(MATRICES["bt601_luma"], dtype=np.float64)
    pool = np.random.RandomState(232).randint(0, 256, size=(1, 65536, 3)).astype(np.uint8)
    split = np.any(fixed_point(pool, luma) != generic_float(pool, luma), axis=2)[0]
    src_luma = src.copy()
    n_split = min(8, int(np.count_nonzero(split)))
    src_luma[0, :n_split] = pool[0, split][:n_split]

    results = {"src": tolist(src), "src_r1": tolist(src_r1), "src_luma": tolist(src_luma)}
    cases = {}
    for name, rows in MATRICES.items():
        m = np.array(rows, dtype=np.float64)
        fixed = takes_fixed_point(m)
        emulate = fixed_point if fixed else generic_float
        fixture_src = {"r_coef_8192": src_r1, "bt601_luma": src_luma}.get(name, src)
        out = cv2.transform(fixture_src, m)
        cases[name] = {
            "m": tolist(m),
            "takes_fixed_point": fixed,
            "dst": tolist(out),
            "differences_from_emulation_by_width": check_widths(231, m, emulate),
            "fixture_differs_from_other_kernel": int(
                np.count_nonzero(
                    out.astype(int)
                    != (generic_float if fixed else fixed_point)(fixture_src, m).astype(int)
                )
            ),
        }
    results["cases"] = cases

    # Near the bounds the sum can leave `int`. The vector lanes wrap; the scalar tail is signed
    # overflow in C++ and wraps in this build too.
    wrap_m = np.array(
        [[31.99, 31.99, 8191.99, 8191.99], [0.5, 0.5, 0.0, 0.0], [0.0, 0.5, 0.5, 0.0]], dtype=np.float64
    )
    assert takes_fixed_point(wrap_m)
    wrap = {}
    for w in (1, 3, 37, 64):
        s = np.full((1, w, 3), 255, dtype=np.uint8)
        wrap[str(w)] = {
            "dst_first_pixel": tolist(cv2.transform(s, wrap_m)[0, 0]),
            "dst_last_pixel": tolist(cv2.transform(s, wrap_m)[0, -1]),
            "emulation": tolist(fixed_point(s, wrap_m)[0, 0]),
        }
    results["int_wrap"] = {"m": tolist(wrap_m), "by_width": wrap}

    # The vector lanes round the bias once (`m[3] * 1024`) and add 512; the tail rounds
    # `(m[3] + 0.5f) * 1024`. A bias one float step above 0.5/1024 rounds up in the first and is
    # absorbed by `+ 0.5f` in the second; with 511/1024 * 1 the sum sits right below the shift.
    near = float(np.nextafter(np.float32(0.5 / 1024.0), np.float32(1.0)))
    near_m = np.array(
        [[511.0 / 1024.0, 0.0, 0.0, near], [0.5, 0.5, 0.0, 0.0], [0.0, 0.5, 0.5, 0.0]], dtype=np.float64
    )
    near_src = np.zeros((1, 37, 3), dtype=np.uint8)
    near_src[..., 0] = 1
    results["bias_near_tie"] = {
        "m": tolist(near_m),
        "src_pixel": tolist(near_src[0, 0]),
        "dst_channel_0": tolist(cv2.transform(near_src, near_m)[0, :, 0]),
        "emulation_channel_0": int(fixed_point(near_src, near_m)[0, 0, 0]),
    }
    dump("core_transform_fixed_point", results)


if __name__ == "__main__":
    main()

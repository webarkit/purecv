"""OpenCV 4.10.0 reference values for add / subtract / multiply / absdiff at the type bounds.

Covers i8, u16, i16 and i32 (the OpenCV depths where overflow is possible; u8 is pinned by
core_convert_arith.py). u32, u64 and i64 have no OpenCV depth. See docs/parity/4.10.0/core.md
(rows add, subtract, multiply, absdiff) and issue #208.

Run from the repository root:
    cpp_ref/.venv/Scripts/python scripts/opencv_ref/core_saturating_ops.py   # Windows
    cpp_ref/.venv/bin/python scripts/opencv_ref/core_saturating_ops.py       # Linux/macOS
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

TYPES = {
    "i8": np.int8,
    "u16": np.uint16,
    "i16": np.int16,
    "i32": np.int32,
}


def pairs(dtype):
    """Operand pairs at and around the bounds of `dtype`, as two 1 x N arrays."""
    info = np.iinfo(dtype)
    lo, hi = int(info.min), int(info.max)
    cand = [
        (hi, 1),
        (hi, hi),
        (lo, -1 if lo < 0 else 1),
        (lo, lo),
        (lo, hi),
        (hi, lo),
        (0, lo),
        (0, hi),
        (1, 3),
        (3, 1),
        (hi // 2 + 1, 2),
        (lo // 2 - 1 if lo < 0 else 100, 2),
        (hi, 2),
        (lo, 2),
    ]
    a = np.array([p[0] for p in cand], dtype=dtype).reshape(1, -1)
    b = np.array([p[1] for p in cand], dtype=dtype).reshape(1, -1)
    return a, b


def main() -> None:
    results = {}
    for name, dtype in TYPES.items():
        a, b = pairs(dtype)
        results[name] = {
            "a": tolist(a),
            "b": tolist(b),
            "add": tolist(cv2.add(a, b)),
            "subtract": tolist(cv2.subtract(a, b)),
            "multiply": tolist(cv2.multiply(a, b)),
            "absdiff": tolist(cv2.absdiff(a, b)),
        }
    # u16 multiply: `c_mul` is `saturate_cast<ushort>(a * b)` with both operands promoted to `int`,
    # so the scalar kernel overflows for products >= 2^31, while the vector lanes saturate.
    # A row narrower than a vector is all scalar; a longer one ends in a scalar tail.
    overflow = {}
    for value in (46340, 46341, 50000, 65535):
        for width in (3, 64, 65):
            a = np.full((1, width), value, dtype=np.uint16)
            overflow[f"{value}x{value}_width_{width}"] = tolist(cv2.multiply(a, a))
    results["u16_multiply_overflow"] = overflow
    dump("core_saturating_ops", results)


if __name__ == "__main__":
    main()

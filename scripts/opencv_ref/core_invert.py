"""OpenCV 4.10.0 reference values for `invert` (issue #219).

Measures the return value and the output matrix of `cv::invert` for every decomposition flag,
on a regular and a singular matrix. See docs/parity/4.10.0/core.md.

Run from the repository root:
    cpp_ref/.venv/Scripts/python scripts/opencv_ref/core_invert.py   # Windows
    cpp_ref/.venv/bin/python scripts/opencv_ref/core_invert.py       # Linux/macOS
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

FLAGS = {
    "LU": cv2.DECOMP_LU,
    "SVD": cv2.DECOMP_SVD,
    "EIG": cv2.DECOMP_EIG,
    "CHOLESKY": cv2.DECOMP_CHOLESKY,
    "QR": cv2.DECOMP_QR,
}


def attempt(fn):
    try:
        r = fn()
        return {"ret": float(r[0]), "dst": tolist(r[1])}
    except cv2.error as e:  # noqa: PERF203
        return f"cv2.error: {e.code} {e.err}".strip()


def main() -> None:
    results = {}
    # regular (det 10), symmetric positive definite (det 5), singular (det 0) and near-singular
    mats = {
        "regular_f64": np.array([[4.0, 7.0], [2.0, 6.0]], dtype=np.float64),
        "spd_f64": np.array([[2.0, 1.0], [1.0, 3.0]], dtype=np.float64),
        "singular_f64": np.array([[1.0, 2.0], [2.0, 4.0]], dtype=np.float64),
        "singular_3x3_f64": np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], dtype=np.float64),
        "zero_f64": np.zeros((2, 2), dtype=np.float64),
        "regular_f32": np.array([[4.0, 7.0], [2.0, 6.0]], dtype=np.float32),
        "singular_f32": np.array([[1.0, 2.0], [2.0, 4.0]], dtype=np.float32),
        # tiny but regular: purecv's absolute pivot threshold of 1e-12 would call this singular
        "tiny_regular_f64": np.array([[1e-13, 0.0], [0.0, 1e-13]], dtype=np.float64),
    }
    for mname, m in mats.items():
        results[mname] = {
            fname: attempt(lambda m=m, f=f: cv2.invert(m, flags=f)) for fname, f in FLAGS.items()
        }
    dump("core_invert.py", results)


if __name__ == "__main__":
    main()

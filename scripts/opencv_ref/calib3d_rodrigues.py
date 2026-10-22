"""OpenCV 4.10.0 reference values for the rodrigues rows (docs/parity/4.10.0/calib3d.md).

`cv2.Rodrigues` in both directions on literal rotation vectors (zero, tiny, small, general,
the pi edge in several axes, just below pi, beyond pi), on the 1x3 row form, on matrices
that are not rotations (scaled, perturbed, a reflection) and on invalid shapes; the Jacobian
is recorded for one case of each direction. Every expected value is listed in full so that
tests/parity_calib3d.rs can pin it.

Run from the repository root:
    cpp_ref/.venv/Scripts/python scripts/opencv_ref/calib3d_rodrigues.py   # Windows
    cpp_ref/.venv/bin/python scripts/opencv_ref/calib3d_rodrigues.py       # Linux/macOS
"""

import os
import sys

import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _reference import REFERENCE_VERSION  # noqa: E402

assert cv2.__version__ == REFERENCE_VERSION, f"cv2 {cv2.__version__} is not the pinned {REFERENCE_VERSION}"
cv2.ipp.setUseIPP(False)  # measure the generic CPU path

import math
import numpy as np

from _calib3d_common import rational_rotation  # noqa: E402
from _core_common import dump, tolist  # noqa: E402

PI = math.pi
S3 = math.sqrt(3.0)

# (name, rotation vector): the Rust twin builds the same doubles from the same literals.
VECTORS = [
    ("zero", (0.0, 0.0, 0.0)),
    ("tiny_1e-12", (1e-12, 0.0, 0.0)),
    ("tiny_3e-9", (3e-9, -2e-9, 1e-9)),
    ("small_1e-6", (1e-6, 0.0, 0.0)),
    ("small_2e-6", (2e-6, -1e-6, 3e-6)),
    ("small_5e-6", (0.0, 5e-6, 0.0)),
    ("small_2e-5", (2e-5, 0.0, 0.0)),
    ("small_1e-3", (1e-3, -2e-3, 0.5e-3)),
    ("general_a", (0.1, -0.2, 0.3)),
    ("general_b", (0.5, -1.25, 0.75)),
    ("general_c", (-2.0, 1.0, 0.5)),
    ("pi_x", (PI, 0.0, 0.0)),
    ("pi_y", (0.0, PI, 0.0)),
    ("pi_z", (0.0, 0.0, PI)),
    ("pi_111", (PI / S3, PI / S3, PI / S3)),
    ("pi_xz", (PI * 0.6, 0.0, PI * 0.8)),
    ("pi_yz_neg", (0.0, PI * 0.6, -PI * 0.8)),
    ("pi_xy_neg", (PI * 0.8, -PI * 0.6, 0.0)),
    ("pi_neg_x", (-PI, 0.0, 0.0)),
    ("near_pi_1e-6", ((PI - 1e-6) * 0.6, 0.0, (PI - 1e-6) * 0.8)),
    ("near_pi_2e-5", ((PI - 2e-5) * 0.6, 0.0, (PI - 2e-5) * 0.8)),
    ("near_pi_5e-4", ((PI - 5e-4) * 0.6, 0.0, (PI - 5e-4) * 0.8)),
    ("near_pi_2e-5_neg", (0.0, -(PI - 2e-5) * 0.6, (PI - 2e-5) * 0.8)),
    ("beyond_pi_4", (4.0, 0.0, 0.0)),
    ("beyond_2pi", (2 * PI + 0.5, 0.0, 0.0)),
]


def attempt(fn):
    try:
        return tolist(fn())
    except cv2.error as e:  # noqa: PERF203
        return f"cv2.error: {e.code} {e.err}".strip()


def vec_to_mat(v, dtype=np.float64, shape=(3, 1)):
    src = np.array(v, dtype).reshape(shape)
    m, jac = cv2.Rodrigues(src)
    return m, jac


def mat_to_vec(m, dtype=np.float64):
    src = np.array(m, dtype).reshape(3, 3)
    v, jac = cv2.Rodrigues(src)
    return v, jac


def main() -> None:
    results = {}

    vm = {}
    rt = {}
    for name, v in VECTORS:
        m, _ = vec_to_mat(v)
        vm[name] = {"rvec": list(v), "rmat": tolist(m.reshape(-1))}
        back, _ = mat_to_vec(m)
        rt[name] = {"rvec_back": tolist(back.reshape(-1)), "shape": list(back.shape)}
    results["vec_to_mat"] = vm
    results["round_trip"] = rt

    # --- 1x3 row input, f32 input
    m_row, _ = vec_to_mat((0.1, -0.2, 0.3), shape=(1, 3))
    m_col, _ = vec_to_mat((0.1, -0.2, 0.3))
    v_row = np.array([[0.1, -0.2, 0.3]], np.float64)
    m32, _ = cv2.Rodrigues(np.array([0.1, -0.2, 0.3], np.float32).reshape(3, 1))
    results["shapes"] = {
        "row_1x3_equals_col": bool(np.array_equal(m_row, m_col)),
        "row_1x3_rmat": tolist(m_row.reshape(-1)),
        "f32_input_dtype": str(m32.dtype),
        "f32_input_rmat": tolist(m32.reshape(-1)),
        "mat_to_vec_output_shape": list(mat_to_vec(m_col)[0].shape),
        "vec_to_mat_output_shape": list(m_col.shape),
        "row_1x3_mat_to_vec_shape": list(cv2.Rodrigues(cv2.Rodrigues(v_row)[0])[0].shape),
        "invalid_2x2": attempt(lambda: cv2.Rodrigues(np.eye(2))[0]),
        "invalid_4x1": attempt(lambda: cv2.Rodrigues(np.zeros((4, 1)))[0]),
        "invalid_3x2": attempt(lambda: cv2.Rodrigues(np.zeros((3, 2)))[0]),
        "invalid_int_3x1": attempt(lambda: cv2.Rodrigues(np.zeros((3, 1), np.int32))[0]),
        "vec3_1x1x3": attempt(lambda: cv2.Rodrigues(np.array([[[0.1, -0.2, 0.3]]]))[0].reshape(-1)),
    }

    # --- matrices that are not rotations
    r_a = rational_rotation(12 / 13, 5 / 13, 4 / 5, 3 / 5, 15 / 17, 8 / 17)
    scaled = [1.1 * v for v in r_a]
    perturbed = list(r_a)
    perturbed[1] += 0.01
    reflection = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, -1.0]
    identity = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    huge = [200.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    mats = {}
    for name, m in (
        ("rational_a", r_a),
        ("scaled_1.1", scaled),
        ("perturbed_01", perturbed),
        ("reflection_z", reflection),
        ("identity", identity),
        ("out_of_range_200", huge),
        ("transpose_rational_a", [r_a[0], r_a[3], r_a[6], r_a[1], r_a[4], r_a[7], r_a[2], r_a[5], r_a[8]]),
    ):
        v, _ = mat_to_vec(m)
        mats[name] = {"rmat": list(m), "rvec": tolist(v.reshape(-1))}
    results["mat_to_vec"] = mats

    # --- Jacobians (purecv has none; recorded for the row)
    m, jac_vm = vec_to_mat((0.1, -0.2, 0.3))
    v, jac_mv = mat_to_vec(m)
    results["jacobian"] = {
        "vec_to_mat_shape": list(jac_vm.shape),
        "vec_to_mat": tolist(jac_vm),
        "mat_to_vec_shape": list(jac_mv.shape),
        "mat_to_vec": tolist(jac_mv),
        "zero_vec_to_mat": tolist(vec_to_mat((0.0, 0.0, 0.0))[1]),
    }

    dump("calib3d_rodrigues", results)


if __name__ == "__main__":
    main()

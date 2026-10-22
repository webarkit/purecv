"""OpenCV 4.10.0 reference values for the init_undistort_rectify_map row and the undistort /
undistortPoints `missing` rows (docs/parity/4.10.0/calib3d.md).

`cv2.initUndistortRectifyMap` with 4, 5, 8, 12 and 14 distortion coefficients, with and
without a rectification `R` and a new camera matrix, on a 64x48 map (`CV_32FC1`): the FNV-1a
digest of each map's float32 bytes, sampled values, and the largest difference from a plain
double evaluation of the generic scalar formula (to see whether the wheel's SIMD path changes
the float32 result). `cv2.undistortPoints` and `cv2.undistort` are recorded for the rows that
purecv lacks.

Run from the repository root:
    cpp_ref/.venv/Scripts/python scripts/opencv_ref/calib3d_undistort.py   # Windows
    cpp_ref/.venv/bin/python scripts/opencv_ref/calib3d_undistort.py       # Linux/macOS
"""

import os
import sys

import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _reference import REFERENCE_VERSION  # noqa: E402

assert cv2.__version__ == REFERENCE_VERSION, f"cv2 {cv2.__version__} is not the pinned {REFERENCE_VERSION}"
cv2.ipp.setUseIPP(False)  # measure the generic CPU path

import numpy as np

from _calib3d_common import camera_matrix, f32_digest, rational_rotation  # noqa: E402
from _core_common import dump, tolist  # noqa: E402

W, H = 64, 48
COEFFS = {
    "zeros_5": [0.0, 0.0, 0.0, 0.0, 0.0],
    "k4": [-0.25, 0.08, 0.001, -0.0005],
    "k5": [-0.25, 0.08, 0.001, -0.0005, 0.02],
    "k8": [-0.25, 0.08, 0.001, -0.0005, 0.02, 0.1, -0.03, 0.005],
    "k12": [-0.25, 0.08, 0.001, -0.0005, 0.02, 0.1, -0.03, 0.005, 0.001, -0.0005, 0.0008, 0.0002],
    "k14": [-0.25, 0.08, 0.001, -0.0005, 0.02, 0.1, -0.03, 0.005, 0.001, -0.0005, 0.0008, 0.0002,
            0.01, -0.02],
    "k8_strong": [-0.5, 0.3, 0.02, -0.01, 0.1, 0.2, -0.1, 0.05],
}
NEW_K = np.array([[700.0, 0.0, 300.0], [0.0, 720.0, 250.0], [0.0, 0.0, 1.0]], np.float64)
R_SMALL = np.array(rational_rotation(24 / 25, 7 / 25, 1.0, 0.0, 1.0, 0.0), np.float64).reshape(3, 3)
SAMPLES = [(0, 0), (W - 1, 0), (0, H - 1), (W - 1, H - 1), (W // 2, H // 2), (17, 29)]


def attempt(fn):
    try:
        return fn()
    except cv2.error as e:  # noqa: PERF203
        return f"cv2.error: {e.code} {e.err}".strip()


def scalar_maps(k, dist, r, new_k):
    """The generic scalar formula of `initUndistortRectifyMapComputer`, evaluated per pixel in
    plain Python doubles (no SIMD, no FMA), cast to float32 as `m1f[j] = (float)u`."""
    d = list(dist) + [0.0] * (14 - len(dist))
    k1, k2, p1, p2, k3, k4, k5, k6, s1, s2, s3, s4, tx, ty = d
    ir = np.linalg.inv(new_k @ r).reshape(-1)
    # matTilt (computeTiltProjectionMatrix): only identity when tauX = tauY = 0
    assert tx == 0.0 and ty == 0.0, "scalar twin covers tau = 0 only"
    fx, fy, u0, v0 = k[0, 0], k[1, 1], k[0, 2], k[1, 2]
    m1 = np.empty((H, W), np.float32)
    m2 = np.empty((H, W), np.float32)
    for i in range(H):
        _x = i * ir[1] + ir[2]
        _y = i * ir[4] + ir[5]
        _w = i * ir[7] + ir[8]
        for j in range(W):
            w = 1.0 / _w
            x = _x * w
            y = _y * w
            x2 = x * x
            y2 = y * y
            r2 = x2 + y2
            _2xy = 2 * x * y
            kr = (1 + ((k3 * r2 + k2) * r2 + k1) * r2) / (1 + ((k6 * r2 + k5) * r2 + k4) * r2)
            xd = x * kr + p1 * _2xy + p2 * (r2 + 2 * x2) + s1 * r2 + s2 * r2 * r2
            yd = y * kr + p1 * (r2 + 2 * y2) + p2 * _2xy + s3 * r2 + s4 * r2 * r2
            m1[i, j] = np.float32(fx * xd + u0)
            m2[i, j] = np.float32(fy * yd + v0)
            _x += ir[0]
            _y += ir[3]
            _w += ir[6]
    return m1, m2


def maps(k, dist, r, new_k, size=(W, H)):
    d = None if dist is None else np.array(dist, np.float64)
    rr = None if r is None else r
    m1, m2 = cv2.initUndistortRectifyMap(k, d, rr, new_k, size, cv2.CV_32FC1)
    out = {
        "shape": list(m1.shape),
        "dtype": str(m1.dtype),
        "fnv1a_map1": f32_digest(m1),
        "fnv1a_map2": f32_digest(m2),
        "samples": [[x, y, float(m1[y, x]), float(m2[y, x])] for x, y in SAMPLES
                    if x < m1.shape[1] and y < m1.shape[0]],
        "min_max_map1": [float(m1.min()), float(m1.max())] if m1.size else None,
    }
    if (dist is not None and len(dist) <= 12 and new_k is not None and new_k.shape == (3, 3)
            and size == (W, H)):
        s1, s2 = scalar_maps(k, dist, np.eye(3) if r is None else r, new_k)
        out["scalar_twin"] = {
            "mismatches_map1": int(np.count_nonzero(s1 != m1)),
            "mismatches_map2": int(np.count_nonzero(s2 != m2)),
            "max_abs_diff": float(max(np.abs(s1.astype(np.float64) - m1).max(),
                                      np.abs(s2.astype(np.float64) - m2).max())),
        }
    return out


def main() -> None:
    results = {}
    k = camera_matrix()
    results["inputs"] = {
        "camera_matrix": tolist(k.reshape(-1)),
        "new_camera_matrix": tolist(NEW_K.reshape(-1)),
        "r_small": tolist(R_SMALL.reshape(-1)),
        "size": [W, H],
        "coefficients": COEFFS,
        "samples_xy": SAMPLES,
    }

    plain = {}
    for name, c in COEFFS.items():
        plain[name] = maps(k, c, None, k)
    plain["none_dist"] = maps(k, None, None, k)
    results["identity_r_same_k"] = plain

    rect = {}
    for name in ("k5", "k8", "k14"):
        rect[name] = maps(k, COEFFS[name], R_SMALL, NEW_K)
    rect["k5_r_only"] = maps(k, COEFFS["k5"], R_SMALL, k)
    rect["k5_newk_only"] = maps(k, COEFFS["k5"], None, NEW_K)
    results["rectified"] = rect

    # D1/D2: empty newCameraMatrix centres the principal point; R as a 3x1 vector; sizes
    d2 = {}
    d2["new_k_empty"] = attempt(lambda: maps(k, COEFFS["k5"], None, None))
    d2["r_3x1_vector"] = attempt(lambda: maps(k, COEFFS["k5"], np.zeros((3, 1)), k))
    d2["dist_3_invalid"] = attempt(lambda: maps(k, [0.1, 0.2, 0.3], None, k))
    d2["dist_6_invalid"] = attempt(lambda: maps(k, [0.1, 0.2, 0.3, 0.0, 0.0, 0.0], None, k))
    d2["dist_9_invalid"] = attempt(lambda: maps(k, [0.0] * 9, None, k))
    d2["new_k_3x4"] = attempt(lambda: maps(k, COEFFS["k5"], None, np.hstack([k, np.zeros((3, 1))])))
    d2["k_f32_input"] = attempt(lambda: maps(k.astype(np.float32), COEFFS["k5"], None, k.astype(np.float32))["fnv1a_map1"])
    d2["size_1x1"] = attempt(lambda: maps(k, COEFFS["k5"], None, k, size=(1, 1))["samples"][0])
    d2["size_0x0"] = attempt(lambda: str(cv2.initUndistortRectifyMap(k, np.array(COEFFS["k5"]), None, k, (0, 0), cv2.CV_32FC1)[0]))
    results["d2"] = d2

    # getDefaultNewCameraMatrix (what the empty newCameraMatrix resolves to)
    results["default_new_camera_matrix"] = {
        "centered": tolist(cv2.getDefaultNewCameraMatrix(k, (W, H), True).reshape(-1)),
        "not_centered": tolist(cv2.getDefaultNewCameraMatrix(k, (W, H), False).reshape(-1)),
    }

    # --- undistortPoints (missing as public API; purecv's private helper applies K^-1 only)
    pts = np.array([[320.0, 240.0], [0.0, 0.0], [639.0, 479.0], [100.0, 400.0], [500.0, 50.0]],
                   np.float32).reshape(-1, 1, 2)
    up = {}
    up["points"] = tolist(pts.reshape(-1, 2))
    up["no_dist"] = tolist(cv2.undistortPoints(pts, k, None).reshape(-1, 2))
    up["zeros_5"] = tolist(cv2.undistortPoints(pts, k, np.zeros(5)).reshape(-1, 2))
    up["k5"] = tolist(cv2.undistortPoints(pts, k, np.array(COEFFS["k5"])).reshape(-1, 2))
    up["k5_iter_20_eps"] = tolist(cv2.undistortPointsIter(
        pts, k, np.array(COEFFS["k5"]), None, None,
        (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 20, 1e-9)).reshape(-1, 2))
    up["k5_p_k"] = tolist(cv2.undistortPoints(pts, k, np.array(COEFFS["k5"]), None, None, k).reshape(-1, 2))
    up["k14"] = tolist(cv2.undistortPoints(pts, k, np.array(COEFFS["k14"])).reshape(-1, 2))
    up["default_criteria"] = "TermCriteria(MAX_ITER, 5, 0.01)"
    results["undistort_points"] = up

    # --- undistort (image): a 64x48 gradient, with the k5 coefficients (needs remap, #237)
    img = np.fromfunction(lambda y, x: (x * 3 + y * 5) % 256, (H, W), dtype=np.int64).astype(np.uint8)
    out = cv2.undistort(img, k, np.array(COEFFS["k5"]))
    results["undistort_image"] = {
        "input_fnv1a": f"0x{__import__('_features2d_common').fnv1a32(img.tobytes()):08x}",
        "output_fnv1a": f"0x{__import__('_features2d_common').fnv1a32(out.tobytes()):08x}",
        "output_samples": [[x, y, int(out[y, x])] for x, y in SAMPLES],
        "interpolation": "remap INTER_LINEAR, BORDER_CONSTANT 0 (undistort.dispatch.cpp)",
    }

    dump("calib3d_undistort", results)


if __name__ == "__main__":
    main()

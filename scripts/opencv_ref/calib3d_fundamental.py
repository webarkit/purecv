"""OpenCV 4.10.0 reference values for the find_fundamental_mat rows (docs/parity/calib3d.md).

`cv2.findFundamentalMat` with FM_8POINT, FM_RANSAC, FM_LMEDS and FM_7POINT on 60 seeded 3-D
points seen by two cameras (identity and a rational rotation with a baseline): exact, with
+-0.5 px noise, and with 12 of 60 correspondences replaced by random pixels. Each result is
the matrix scaled to unit Frobenius norm, the inlier mask and the epipolar-line errors over
the inliers, so that tests/parity_calib3d.rs can pin the matrices and compare the inlier sets
and model quality where the kernels differ (OpenCV samples 7 points and keeps the best minimal
model, purecv samples 8 and refits).

Run from the repository root:
    cpp_ref/.venv/Scripts/python scripts/opencv_ref/calib3d_fundamental.py   # Windows
    cpp_ref/.venv/bin/python scripts/opencv_ref/calib3d_fundamental.py       # Linux/macOS
"""

import cv2

assert cv2.__version__ == "4.10.0"
cv2.ipp.setUseIPP(False)  # measure the generic CPU path

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _calib3d_common import (  # noqa: E402
    Lcg,
    add_noise,
    as_f32_points,
    camera_matrix,
    f32,
    mask_digest,
    norm_f,
    points_f32_digest,
    pose_b,
    project,
    replace_outliers,
)
from _core_common import dump, tolist  # noqa: E402

METHODS = {
    "fm_8point": cv2.FM_8POINT,
    "fm_ransac": cv2.FM_RANSAC,
    "fm_lmeds": cv2.FM_LMEDS,
}
IDENTITY = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]


def attempt(fn):
    try:
        return fn()
    except cv2.error as e:  # noqa: PERF203
        return f"cv2.error: {e.code} {e.err}".strip()


def scene_points(n: int, seed: int):
    """`n` points with `k / 1024` x, y in [-1, 1] and z in [4, 6)."""
    rng = Lcg(seed)
    pts = []
    for _ in range(n):
        x = rng.signed()
        y = rng.signed()
        z = 4.0 + rng.frac() * 2.0
        pts.append((x, y, z))
    return pts


def epiline_errors(f, p1, p2):
    """OpenCV's `FMEstimatorCallback::computeError`: the larger of the two squared
    point-to-epipolar-line distances, per correspondence (doubles)."""
    f = np.asarray(f, np.float64).reshape(-1)
    out = []
    for (x1, y1), (x2, y2) in zip(p1, p2):
        a = f[0] * x1 + f[1] * y1 + f[2]
        b = f[3] * x1 + f[4] * y1 + f[5]
        c = f[6] * x1 + f[7] * y1 + f[8]
        s2 = 1.0 / (a * a + b * b)
        d2 = x2 * a + y2 * b + c
        a = f[0] * x2 + f[3] * y2 + f[6]
        b = f[1] * x2 + f[4] * y2 + f[7]
        c = f[2] * x2 + f[5] * y2 + f[8]
        s1 = 1.0 / (a * a + b * b)
        d1 = x1 * a + y1 * b + c
        out.append(max(d1 * d1 * s1, d2 * d2 * s2))
    return out


def find(p1, p2, method, thr=3.0, conf=0.99, max_iters=1000):
    f, mask = cv2.findFundamentalMat(as_f32_points(p1), as_f32_points(p2), method, thr, conf,
                                     max_iters)
    out = {"f": None if f is None else norm_f(f), "raw": None if f is None else tolist(f),
           "shape": None if f is None else list(f.shape)}
    out["mask"] = mask_digest(mask) if mask is not None else None
    if f is not None and f.shape == (3, 3):
        errs = epiline_errors(f, [(f32(x), f32(y)) for x, y in p1], [(f32(x), f32(y)) for x, y in p2])
        sel = [e for e, m in zip(errs, mask.ravel()) if m] if mask is not None else errs
        out["inlier_max_err_px"] = float(np.sqrt(max(sel))) if sel else None
        out["inlier_rms_err_px"] = float(np.sqrt(np.mean(sel))) if sel else None
        out["all_max_err_px"] = float(np.sqrt(max(errs)))
    return out


def main() -> None:
    results = {}
    k = camera_matrix()
    r2, t2 = pose_b()
    scene = scene_points(60, 0x3579)
    p1 = [(f32(u), f32(v)) for u, v in (project(IDENTITY, [0.0, 0.0, 0.0], p) for p in scene)]
    p2 = [(f32(u), f32(v)) for u, v in (project(r2, t2, p) for p in scene)]
    n1 = add_noise(p1, 0.5, 0x4680)
    n2 = add_noise(p2, 0.5, 0x4681)
    o2, outlier_idx = replace_outliers(n2, 12, 0x5791)

    # F_true = K^-T [t]x R K^-1
    tx = np.array([[0, -t2[2], t2[1]], [t2[2], 0, -t2[0]], [-t2[1], t2[0], 0]], np.float64)
    rm = np.array(r2).reshape(3, 3)
    ki = np.linalg.inv(k)
    f_true = ki.T @ tx @ rm @ ki

    results["inputs"] = {
        "camera_matrix": tolist(k.reshape(-1)),
        "r2": list(r2),
        "t2": list(t2),
        "f_true": norm_f(f_true),
        "scene_60": [list(p) for p in scene],
        "p1_exact": [list(p) for p in p1],
        "p2_exact": [list(p) for p in p2],
        "p1_noisy": [list(p) for p in n1],
        "p2_noisy": [list(p) for p in n2],
        "p2_outliers": [list(p) for p in o2],
        "outlier_indices": outlier_idx,
        "fnv1a_p1_exact": points_f32_digest(p1),
        "fnv1a_p2_exact": points_f32_digest(p2),
        "fnv1a_p1_noisy": points_f32_digest(n1),
        "fnv1a_p2_noisy": points_f32_digest(n2),
        "fnv1a_p2_outliers": points_f32_digest(o2),
    }
    results["defaults"] = {
        "method": "FM_RANSAC = 8", "ransacReprojThreshold": 3.0, "confidence": 0.99,
        "maxIters": 1000,
        "FM_7POINT": int(cv2.FM_7POINT), "FM_8POINT": int(cv2.FM_8POINT),
        "FM_LMEDS": int(cv2.FM_LMEDS), "FM_RANSAC": int(cv2.FM_RANSAC),
    }

    for name, (a, b) in (("exact", (p1, p2)), ("noisy", (n1, n2)), ("outliers", (n1, o2))):
        block = {}
        for mname, method in METHODS.items():
            block[mname] = find(a, b, method)
        block["fm_ransac_thr_1"] = find(a, b, cv2.FM_RANSAC, thr=1.0)
        block["fm_ransac_thr_0"] = find(a, b, cv2.FM_RANSAC, thr=0.0)  # <= 0 -> 3
        block["fm_ransac_conf_0"] = find(a, b, cv2.FM_RANSAC, conf=0.0)  # out of range -> 0.99
        block["fm_ransac_iters_5"] = find(a, b, cv2.FM_RANSAC, max_iters=5)
        block["fm_ransac_14_points"] = find(a[:14], b[:14], cv2.FM_RANSAC)  # < 15 -> LMedS
        block["fm_ransac_15_points"] = find(a[:15], b[:15], cv2.FM_RANSAC)
        block["fm_8point_8_points"] = find(a[:8], b[:8], cv2.FM_8POINT)
        block["fm_8point_10_points"] = find(a[:10], b[:10], cv2.FM_8POINT)
        block["usac_default"] = find(a, b, cv2.USAC_DEFAULT)
        results[name] = block

    d2 = {}
    d2["seven_points_8point"] = find(p1[:7], p2[:7], cv2.FM_8POINT)  # 7 points -> run7Point
    d2["seven_points_7point"] = find(p1[:7], p2[:7], cv2.FM_7POINT)
    d2["seven_points_ransac"] = find(p1[:7], p2[:7], cv2.FM_RANSAC)
    d2["six_points"] = find(p1[:6], p2[:6], cv2.FM_8POINT)
    d2["mismatched"] = attempt(lambda: find(p1[:9], p2[:8], cv2.FM_8POINT))
    d2["fm_7point_60_points"] = find(p1, p2, cv2.FM_7POINT)  # method 1 with n > 7 -> LMedS
    same = [(f32(10.0 + i), f32(20.0)) for i in range(10)]
    d2["collinear_10"] = find(same, same, cv2.FM_8POINT)
    d2["identical_sets_8point"] = find(p1, p1, cv2.FM_8POINT)
    results["d2"] = d2

    dump("calib3d_fundamental", results)


if __name__ == "__main__":
    main()

"""OpenCV 4.10.0 reference values for the solve_pnp / solve_pnp_ransac rows
(docs/parity/4.10.0/calib3d.md).

`cv2.solvePnP` with every `SOLVEPNP_*` flag, with and without `useExtrinsicGuess`, on seeded
object points (24 non-planar, 16 planar, 4-point and 5-point subsets) projected through a
rational rotation and the pinhole model: exact, with +-0.5 px noise, and with a 5-coefficient
distortion; `cv2.solvePnPRansac` on 24 correspondences of which 6 are random pixels. Every pose
is recorded as `rvec`, `tvec` and its reprojection RMS so that tests/parity_calib3d.rs can pin
the poses and compare model quality where the optimisers differ.

Run from the repository root:
    cpp_ref/.venv/Scripts/python scripts/opencv_ref/calib3d_pnp.py   # Windows
    cpp_ref/.venv/bin/python scripts/opencv_ref/calib3d_pnp.py       # Linux/macOS
"""

import os
import sys

import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _reference import REFERENCE_VERSION  # noqa: E402

assert cv2.__version__ == REFERENCE_VERSION, f"cv2 {cv2.__version__} is not the pinned {REFERENCE_VERSION}"
cv2.ipp.setUseIPP(False)  # measure the generic CPU path

import numpy as np

from _calib3d_common import (  # noqa: E402
    add_noise,
    as_f32_points,
    camera_matrix,
    f32,
    object_points,
    points_f32_digest,
    pose_a,
    project,
    replace_outliers,
    reprojection_rms,
)
from _core_common import dump, tolist  # noqa: E402

FLAGS = {
    "iterative": cv2.SOLVEPNP_ITERATIVE,
    "epnp": cv2.SOLVEPNP_EPNP,
    "p3p": cv2.SOLVEPNP_P3P,
    "dls": cv2.SOLVEPNP_DLS,
    "upnp": cv2.SOLVEPNP_UPNP,
    "ap3p": cv2.SOLVEPNP_AP3P,
    "ippe": cv2.SOLVEPNP_IPPE,
    "ippe_square": cv2.SOLVEPNP_IPPE_SQUARE,
    "sqpnp": cv2.SOLVEPNP_SQPNP,
}
DIST5 = np.array([-0.25, 0.08, 0.001, -0.0005, 0.02], np.float64)
NO_DIST = np.zeros(5, np.float64)


def attempt(fn):
    try:
        return fn()
    except cv2.error as e:  # noqa: PERF203
        return f"cv2.error: {e.code} {e.err}".strip()


def pnp(obj, img, k, dist, flags, guess=None):
    """`solvePnP` with explicit arguments; `guess` is `(rvec, tvec)` for useExtrinsicGuess."""
    o = as_f32_points(obj)
    i = as_f32_points(img)
    if guess is None:
        ok, rvec, tvec = cv2.solvePnP(o, i, k, dist, flags=flags)
    else:
        rvec0 = np.array(guess[0], np.float64).reshape(3, 1)
        tvec0 = np.array(guess[1], np.float64).reshape(3, 1)
        ok, rvec, tvec = cv2.solvePnP(o, i, k, dist, rvec0, tvec0, True, flags)
    out = {"ok": bool(ok)}
    if ok:
        out["rvec"] = tolist(rvec.reshape(-1))
        out["tvec"] = tolist(tvec.reshape(-1))
        out["rms_px"] = reprojection_rms(obj, img, rvec, tvec, k, dist, cv2)
    return out


def pnp_ransac(obj, img, k, dist, flags=cv2.SOLVEPNP_ITERATIVE, iters=100, err=8.0,
               conf=0.99, guess=None):
    o = as_f32_points(obj)
    i = as_f32_points(img)
    if guess is None:
        ok, rvec, tvec, inl = cv2.solvePnPRansac(
            o, i, k, dist, iterationsCount=iters, reprojectionError=err, confidence=conf,
            flags=flags,
        )
    else:
        rvec0 = np.array(guess[0], np.float64).reshape(3, 1)
        tvec0 = np.array(guess[1], np.float64).reshape(3, 1)
        ok, rvec, tvec, inl = cv2.solvePnPRansac(
            o, i, k, dist, rvec0, tvec0, True, iters, err, conf, flags=flags
        )
    out = {"ok": bool(ok)}
    if ok:
        out["rvec"] = tolist(rvec.reshape(-1))
        out["tvec"] = tolist(tvec.reshape(-1))
        out["inliers"] = [] if inl is None else [int(v) for v in inl.reshape(-1)]
        sel = out["inliers"]
        out["rms_px_inliers"] = reprojection_rms(
            [obj[j] for j in sel], [img[j] for j in sel], rvec, tvec, k, dist, cv2
        )
    return out


def main() -> None:
    results = {}
    k = camera_matrix()
    r_true, t_true = pose_a()
    rvec_true, _ = cv2.Rodrigues(np.array(r_true).reshape(3, 3))
    rvec_true = rvec_true.reshape(-1)

    obj = object_points(24, 0x2468)
    planar = object_points(16, 0x2468, planar=True)
    img = [(f32(u), f32(v)) for u, v in (project(r_true, t_true, p) for p in obj)]
    img_planar = [(f32(u), f32(v)) for u, v in (project(r_true, t_true, p) for p in planar)]
    noisy = add_noise(img, 0.5, 0x1357)
    noisy_planar = add_noise(img_planar, 0.5, 0x1357)
    outl, outlier_idx = replace_outliers(noisy, 6, 0x2469)
    # distorted projections come from OpenCV itself (purecv has no projectPoints)
    dist_img, _ = cv2.projectPoints(
        as_f32_points(obj), rvec_true.reshape(3, 1), np.array(t_true).reshape(3, 1), k, DIST5
    )
    dist_img = [(f32(u), f32(v)) for u, v in dist_img.reshape(-1, 2)]
    square = [(-0.5, 0.5, 0.0), (0.5, 0.5, 0.0), (0.5, -0.5, 0.0), (-0.5, -0.5, 0.0)]
    img_square = [(f32(u), f32(v)) for u, v in (project(r_true, t_true, p) for p in square)]

    results["inputs"] = {
        "camera_matrix": tolist(k.reshape(-1)),
        "r_true": list(r_true),
        "t_true": list(t_true),
        "rvec_true": tolist(rvec_true),
        "dist5": tolist(DIST5),
        "obj_24": [list(p) for p in obj],
        "img_24_exact": [list(p) for p in img],
        "img_24_noisy": [list(p) for p in noisy],
        "img_24_outliers": [list(p) for p in outl],
        "outlier_indices": outlier_idx,
        "img_24_dist5": [list(p) for p in dist_img],
        "obj_16_planar": [list(p) for p in planar],
        "img_16_planar_exact": [list(p) for p in img_planar],
        "img_16_planar_noisy": [list(p) for p in noisy_planar],
        "square_4": [list(p) for p in square],
        "img_square_4": [list(p) for p in img_square],
        "fnv1a_obj_24": points_f32_digest(obj),
        "fnv1a_img_24_exact": points_f32_digest(img),
        "fnv1a_img_24_noisy": points_f32_digest(noisy),
        "fnv1a_img_24_outliers": points_f32_digest(outl),
        "fnv1a_obj_16_planar": points_f32_digest(planar),
        "fnv1a_img_16_planar_exact": points_f32_digest(img_planar),
        "fnv1a_img_16_planar_noisy": points_f32_digest(noisy_planar),
    }
    results["defaults"] = {
        "solvePnP": {"useExtrinsicGuess": False, "flags": "SOLVEPNP_ITERATIVE = 0"},
        "solvePnPRansac": {
            "useExtrinsicGuess": False, "iterationsCount": 100, "reprojectionError": 8.0,
            "confidence": 0.99, "flags": "SOLVEPNP_ITERATIVE = 0",
        },
        "SolvePnPMethod": {n: int(f) for n, f in FLAGS.items()},
        "SOLVEPNP_MAX_COUNT": int(cv2.SOLVEPNP_MAX_COUNT),
    }

    good_guess = (rvec_true + np.array([0.05, -0.03, 0.02]), np.array(t_true) + np.array([0.1, -0.05, 0.2]))
    bad_guess = (np.array([2.5, -1.0, 0.5]), np.array([1.0, 1.0, 10.0]))
    results["guesses"] = {"good": [tolist(good_guess[0]), tolist(good_guess[1])],
                          "bad": [tolist(bad_guess[0]), tolist(bad_guess[1])]}

    # --- every flag on the non-planar, planar and square sets
    flags = {}
    for name, flag in FLAGS.items():
        block = {}
        block["nonplanar_24_exact"] = attempt(lambda: pnp(obj, img, k, NO_DIST, flag))
        block["nonplanar_24_noisy"] = attempt(lambda: pnp(obj, noisy, k, NO_DIST, flag))
        block["nonplanar_24_dist5"] = attempt(lambda: pnp(obj, dist_img, k, DIST5, flag))
        block["nonplanar_24_dist5_ignored"] = attempt(lambda: pnp(obj, dist_img, k, NO_DIST, flag))
        block["planar_16_exact"] = attempt(lambda: pnp(planar, img_planar, k, NO_DIST, flag))
        block["planar_16_noisy"] = attempt(lambda: pnp(planar, noisy_planar, k, NO_DIST, flag))
        block["square_4"] = attempt(lambda: pnp(square, img_square, k, NO_DIST, flag))
        block["nonplanar_4"] = attempt(lambda: pnp(obj[:4], img[:4], k, NO_DIST, flag))
        block["nonplanar_5"] = attempt(lambda: pnp(obj[:5], img[:5], k, NO_DIST, flag))
        block["nonplanar_6"] = attempt(lambda: pnp(obj[:6], img[:6], k, NO_DIST, flag))
        block["nonplanar_3"] = attempt(lambda: pnp(obj[:3], img[:3], k, NO_DIST, flag))
        block["guess_good_noisy"] = attempt(lambda: pnp(obj, noisy, k, NO_DIST, flag, good_guess))
        block["guess_bad_noisy"] = attempt(lambda: pnp(obj, noisy, k, NO_DIST, flag, bad_guess))
        block["guess_good_3_points"] = attempt(lambda: pnp(obj[:3], img[:3], k, NO_DIST, flag, good_guess))
        flags[name] = block
    results["solve_pnp"] = flags
    results["solve_pnp_d2"] = {
        "mismatched_lengths": attempt(lambda: pnp(obj[:6], img[:5], k, NO_DIST, 0)),
        "dist_none": attempt(lambda: pnp(obj, img, k, None, 0)),
        "dist_4": attempt(lambda: pnp(obj, img, k, np.zeros(4), 0)),
        "dist_3_invalid": attempt(lambda: pnp(obj, img, k, np.zeros(3), 0)),
        "flag_9_invalid": attempt(lambda: pnp(obj, img, k, NO_DIST, 9)),
        "camera_2x2": attempt(lambda: pnp(obj, img, np.eye(2), NO_DIST, 0)),
        "f64_points_same": attempt(lambda: tolist(cv2.solvePnP(
            np.array(obj, np.float64), np.array(img, np.float64), k, NO_DIST)[1].reshape(-1))),
    }

    # --- solvePnPRansac
    rs = {}
    rs["outliers_default"] = pnp_ransac(obj, outl, k, NO_DIST)
    rs["outliers_thr_3"] = pnp_ransac(obj, outl, k, NO_DIST, err=3.0)
    rs["outliers_iters_1000"] = pnp_ransac(obj, outl, k, NO_DIST, iters=1000)
    rs["outliers_conf_0.999"] = pnp_ransac(obj, outl, k, NO_DIST, conf=0.999)
    rs["exact_default"] = pnp_ransac(obj, img, k, NO_DIST)
    rs["noisy_default"] = pnp_ransac(obj, noisy, k, NO_DIST)
    rs["noisy_thr_1"] = pnp_ransac(obj, noisy, k, NO_DIST, err=1.0)
    rs["outliers_guess_good"] = pnp_ransac(obj, outl, k, NO_DIST, guess=good_guess)
    rs["outliers_dist5"] = pnp_ransac(obj, replace_outliers(dist_img, 6, 0x2469)[0], k, DIST5)
    rs["outliers_epnp"] = pnp_ransac(obj, outl, k, NO_DIST, flags=cv2.SOLVEPNP_EPNP)
    rs["outliers_p3p"] = pnp_ransac(obj, outl, k, NO_DIST, flags=cv2.SOLVEPNP_P3P)
    rs["outliers_ap3p"] = pnp_ransac(obj, outl, k, NO_DIST, flags=cv2.SOLVEPNP_AP3P)
    rs["outliers_sqpnp"] = pnp_ransac(obj, outl, k, NO_DIST, flags=cv2.SOLVEPNP_SQPNP)
    rs["planar_noisy"] = pnp_ransac(planar, noisy_planar, k, NO_DIST)
    rs["points_4"] = attempt(lambda: pnp_ransac(obj[:4], img[:4], k, NO_DIST))
    rs["points_5"] = attempt(lambda: pnp_ransac(obj[:5], img[:5], k, NO_DIST))
    rs["points_6"] = attempt(lambda: pnp_ransac(obj[:6], img[:6], k, NO_DIST))
    rs["points_3"] = attempt(lambda: pnp_ransac(obj[:3], img[:3], k, NO_DIST))
    rs["all_outliers"] = attempt(lambda: pnp_ransac(obj, replace_outliers(img, 24, 0x777)[0], k, NO_DIST))
    rs["iters_0"] = attempt(lambda: pnp_ransac(obj, outl, k, NO_DIST, iters=0))
    rs["conf_1"] = attempt(lambda: pnp_ransac(obj, outl, k, NO_DIST, conf=1.0))
    results["solve_pnp_ransac"] = rs

    # --- projectPoints (missing in purecv): the distortion model and the Jacobian shape
    pts, jac = cv2.projectPoints(as_f32_points(obj[:3]), rvec_true.reshape(3, 1),
                                 np.array(t_true).reshape(3, 1), k, DIST5)
    results["project_points"] = {
        "first_3_dist5": tolist(pts.reshape(-1, 2)),
        "jacobian_shape": list(jac.shape),
        "first_3_no_dist": tolist(cv2.projectPoints(
            as_f32_points(obj[:3]), rvec_true.reshape(3, 1), np.array(t_true).reshape(3, 1),
            k, None)[0].reshape(-1, 2)),
        "dist_3_invalid": attempt(lambda: tolist(cv2.projectPoints(
            as_f32_points(obj[:3]), rvec_true.reshape(3, 1), np.array(t_true).reshape(3, 1),
            k, np.zeros(3))[0])),
    }

    dump("calib3d_pnp", results)


if __name__ == "__main__":
    main()

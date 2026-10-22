"""OpenCV 4.10.0 reference values for the find_homography rows (docs/parity/4.10.0/calib3d.md).

`cv2.findHomography` with methods 0, RANSAC, LMEDS and RHO on a seeded 8x5 jittered grid
mapped through a known homography: exact, with +-0.5 px noise, and with 10 of 40
correspondences replaced by random pixels. Every result is the 3x3 matrix scaled to
`h[2][2] == 1` plus the inlier mask, so that tests/parity_calib3d.rs can pin them. The D2
cases record what OpenCV returns for too few, collinear, mismatched and exactly four points,
and for a non-positive threshold.

Run from the repository root:
    cpp_ref/.venv/Scripts/python scripts/opencv_ref/calib3d_homography.py   # Windows
    cpp_ref/.venv/bin/python scripts/opencv_ref/calib3d_homography.py       # Linux/macOS
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
    apply_homography,
    as_f32_points,
    f32,
    grid_jittered,
    mask_digest,
    norm_h,
    points_f32_digest,
    replace_outliers,
)
from _core_common import dump, tolist  # noqa: E402

H_TRUE = [1.1, 0.08, 20.0, -0.05, 0.95, -12.0, 0.0002, -0.0001, 1.0]
METHODS = {"method0": 0, "ransac": cv2.RANSAC, "lmeds": cv2.LMEDS, "rho": cv2.RHO}


def attempt(fn):
    try:
        return fn()
    except cv2.error as e:  # noqa: PERF203
        return f"cv2.error: {e.code} {e.err}".strip()


def find(src, dst, method, thr=3.0, max_iters=2000, confidence=0.995):
    """`findHomography` with explicit arguments; `None` matrix when OpenCV returns empty."""
    h, mask = cv2.findHomography(
        as_f32_points(src), as_f32_points(dst), method, thr, maxIters=max_iters,
        confidence=confidence,
    )
    out = {"h": None if h is None else norm_h(h), "raw": None if h is None else tolist(h)}
    out["mask"] = mask_digest(mask) if mask is not None else None
    if h is not None:
        # reprojection error (px) of the returned matrix over the mask's inliers
        hh = np.asarray(h, np.float64).reshape(-1)
        errs = []
        for (x, y), (u, v), m in zip(src, dst, mask.ravel()):
            px, py = apply_homography(hh, f32(x), f32(y))
            if m:
                errs.append((px - f32(u)) ** 2 + (py - f32(v)) ** 2)
        out["inlier_rms_px"] = float(np.sqrt(np.mean(errs))) if errs else None
        out["inlier_max_px"] = float(np.sqrt(max(errs))) if errs else None
    return out


def main() -> None:
    results = {}
    src = grid_jittered(8, 5, 40, 40, 70, 60, 0x1234)
    exact = [apply_homography(H_TRUE, f32(x), f32(y)) for x, y in src]
    exact = [(f32(u), f32(v)) for u, v in exact]
    noisy = add_noise(exact, 0.5, 0x5678)
    outl, outlier_idx = replace_outliers(noisy, 10, 0x9ABC)
    results["inputs"] = {
        "h_true": H_TRUE,
        "src_40": [[x, y] for x, y in src],
        "dst_exact": [[u, v] for u, v in exact],
        "dst_noisy": [[u, v] for u, v in noisy],
        "dst_outliers": [[u, v] for u, v in outl],
        "outlier_indices": outlier_idx,
        "fnv1a_src": points_f32_digest(src),
        "fnv1a_dst_exact": points_f32_digest(exact),
        "fnv1a_dst_noisy": points_f32_digest(noisy),
        "fnv1a_dst_outliers": points_f32_digest(outl),
    }
    results["defaults"] = {
        "method": 0,
        "ransacReprojThreshold": 3.0,
        "maxIters": 2000,
        "confidence": 0.995,
        "RANSAC": int(cv2.RANSAC),
        "LMEDS": int(cv2.LMEDS),
        "RHO": int(cv2.RHO),
        "USAC_DEFAULT..USAC_MAGSAC": [int(cv2.USAC_DEFAULT), int(cv2.USAC_MAGSAC)],
    }

    for name, dst in (("exact", exact), ("noisy", noisy), ("outliers", outl)):
        block = {}
        for mname, method in METHODS.items():
            block[mname] = find(src, dst, method)
        block["ransac_thr_1"] = find(src, dst, cv2.RANSAC, thr=1.0)
        block["ransac_thr_0"] = find(src, dst, cv2.RANSAC, thr=0.0)  # <= 0 -> default 3
        block["ransac_iters_10"] = find(src, dst, cv2.RANSAC, max_iters=10)
        block["usac_default"] = find(src, dst, cv2.USAC_DEFAULT)
        results[name] = block

    # --- D2: validation and degenerate inputs
    d2 = {}
    d2["three_points"] = attempt(lambda: find(src[:3], exact[:3], 0))
    d2["mismatched_lengths"] = attempt(lambda: find(src[:5], exact[:4], 0))
    d2["four_points_method0"] = find(src[:4], exact[:4], 0)
    d2["four_points_ransac"] = find(src[:4], exact[:4], cv2.RANSAC)  # npoints == 4 -> kernel
    d2["four_points_lmeds"] = find(src[:4], exact[:4], cv2.LMEDS)
    d2["five_points_method0_noisy"] = find(src[:5], noisy[:5], 0)
    col = [(0.0, 0.0), (10.0, 10.0), (20.0, 20.0), (30.0, 30.0)]
    d2["collinear_4"] = find(col, col, 0)
    col5 = col + [(40.0, 40.0)]
    d2["collinear_5_ransac"] = find(col5, col5, cv2.RANSAC)
    same = [(10.0, 10.0)] * 4
    d2["coincident_4"] = find(same, same, 0)
    d2["identity_exact_mask_method0"] = find(src, src, 0)["mask"]
    ident = find(src, src, cv2.RANSAC)
    d2["identity_ransac"] = {"h": ident["h"], "mask": ident["mask"]}
    d2["f64_input_same_as_f32"] = tolist(
        norm_h(cv2.findHomography(np.array(src, np.float64), np.array(exact, np.float64), 0)[0])
    )
    results["d2"] = d2

    dump("calib3d_homography", results)


if __name__ == "__main__":
    main()

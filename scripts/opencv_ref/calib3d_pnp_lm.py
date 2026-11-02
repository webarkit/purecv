"""OpenCV 4.10.0 reference poses for the Levenberg-Marquardt refinement of solve_pnp (#135).

`SOLVEPNP_ITERATIVE` refines the pose with CvLevMarq on the reprojection error in **pixels**
(cvFindExtrinsicCameraParams2, calibration.cpp). With fx != fy and noisy image points that
minimum differs from the minimum of the error in normalised coordinates. The rows use 12
non-planar object points, pose A, a camera with fx = 800, fy = 400 and +-2 px noise, and
record the image points (f32), the pose and its RMS so tests/parity_calib3d.rs can pin them.

Run from the repository root:
    cpp_ref/.venv/Scripts/python scripts/opencv_ref/calib3d_pnp_lm.py   # Windows
    cpp_ref/.venv/bin/python scripts/opencv_ref/calib3d_pnp_lm.py       # Linux/macOS
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
    object_points,
    pose_a,
    reprojection_rms,
    transform,
)
from _core_common import dump, tolist  # noqa: E402

FX, FY, CX, CY = 800.0, 400.0, 320.0, 240.0


def main() -> None:
    k = np.array([[FX, 0.0, CX], [0.0, FY, CY], [0.0, 0.0, 1.0]], np.float64)
    dist = np.zeros(5, np.float64)
    r, t = pose_a()
    obj = object_points(12, 7)
    exact = []
    for p in obj:
        xc, yc, zc = transform(r, t, p)
        exact.append((FX * (xc / zc) + CX, FY * (yc / zc) + CY))
    results = {}
    for amp in (0.0, 2.0):
        img = add_noise(exact, amp, 11) if amp else [(float(np.float32(x)), float(np.float32(y))) for x, y in exact]
        o = np.array(obj, np.float32).reshape(-1, 1, 3)
        i = as_f32_points(img)
        ok, rvec, tvec = cv2.solvePnP(o, i, k, dist, flags=cv2.SOLVEPNP_ITERATIVE)
        results[f"noise_{amp}"] = {
            "ok": bool(ok),
            "object_points": tolist(o.reshape(-1, 3)),
            "image_points_f32": tolist(i.reshape(-1, 2)),
            "rvec": tolist(rvec.reshape(-1)),
            "tvec": tolist(tvec.reshape(-1)),
            "rms": reprojection_rms(o, i.reshape(-1, 2), rvec, tvec, k, dist, cv2),
        }
    dump("calib3d_pnp_lm.py", results)


if __name__ == "__main__":
    main()

"""OpenCV 4.10.0 reference values for the KeyPoint rows (docs/parity/4.10.0/features2d.md).

Rows: KeyPoint::{overlap, convert_to_points, convert_to_points_masked,
convert_from_points}. `KeyPointsFilter` is not bound by cv2 (`CV_EXPORTS`), so
`retain_best` / `sort_by_response` are inspected from `keypoint.cpp` only.

Run from the repository root:
    cpp_ref/.venv/Scripts/python scripts/opencv_ref/features2d_keypoint.py   # Windows
    cpp_ref/.venv/bin/python scripts/opencv_ref/features2d_keypoint.py       # Linux/macOS
"""

import os
import sys

import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _reference import REFERENCE_VERSION  # noqa: E402

assert cv2.__version__ == REFERENCE_VERSION, f"cv2 {cv2.__version__} is not the pinned {REFERENCE_VERSION}"
cv2.ipp.setUseIPP(False)  # measure the generic CPU path

from _core_common import dump, tolist  # noqa: E402
from _features2d_common import kp_list  # noqa: E402


def attempt(fn):
    try:
        return tolist(fn())
    except cv2.error as e:  # noqa: PERF203
        return f"cv2.error: {e.code} {e.err}".strip()


def kp(x, y, size, angle=-1.0, response=0.0, octave=0, class_id=-1):
    return cv2.KeyPoint(float(x), float(y), float(size), float(angle), float(response), int(octave), int(class_id))


def main() -> None:
    results = {}

    ov = {}
    cases = {
        "same": (kp(10, 10, 8), kp(10, 10, 8)),
        "disjoint": (kp(0, 0, 4), kp(10, 0, 4)),
        "touching": (kp(0, 0, 4), kp(4, 0, 4)),
        "enclosed": (kp(5, 5, 10), kp(6, 5, 2)),
        "enclosed_concentric_zero_sizes": (kp(5, 5, 0), kp(5, 5, 0)),
        "partial_1": (kp(0, 0, 10), kp(3, 0, 10)),
        "partial_2": (kp(0, 0, 10), kp(2, 1, 6)),
        "partial_3": (kp(1.5, 2.5, 7), kp(3.25, 1.0, 5)),
        "size_zero_vs_positive_overlapping_centre": (kp(5, 5, 0), kp(5, 5, 6)),
    }
    for name, (a, b) in cases.items():
        ov[name] = float(cv2.KeyPoint_overlap(a, b))
    results["overlap"] = ov

    kps = [kp(1, 2, 3, 10, 0.5, 1, 7), kp(4.5, 5.5, 6, 20, 0.25, 2, 8), kp(7, 8, 9, 30, 0.75, 3, 9)]
    cv_ = {}
    cv_["to_points"] = tolist(cv2.KeyPoint_convert(kps))
    cv_["to_points_indexes_2_0"] = tolist(cv2.KeyPoint_convert(kps, [2, 0]))
    cv_["to_points_indexes_empty_means_all"] = tolist(cv2.KeyPoint_convert(kps, []))
    cv_["to_points_negative_index"] = attempt(lambda: cv2.KeyPoint_convert(kps, [-1]))
    cv_["to_points_index_out_of_range"] = attempt(lambda: cv2.KeyPoint_convert(kps, [3]))
    cv_["from_points_defaults"] = kp_list(cv2.KeyPoint_convert([(1.0, 2.0), (3.5, 4.5)]))
    cv_["from_points_size5_resp2_oct3_class4"] = kp_list(
        cv2.KeyPoint_convert([(1.0, 2.0), (3.5, 4.5)], size=5, response=2, octave=3, class_id=4)
    )
    results["convert"] = cv_

    results["defaults"] = kp_list([cv2.KeyPoint()])
    results["dmatch_default"] = [
        (m.queryIdx, m.trainIdx, m.imgIdx, m.distance) for m in [cv2.DMatch()]
    ]
    # `DMatch::operator<` (by distance) is not bound by cv2; inspected from types.hpp only.

    dump("features2d_keypoint.py", results)


if __name__ == "__main__":
    main()

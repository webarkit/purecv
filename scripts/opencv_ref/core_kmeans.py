"""OpenCV 4.10.0 reference values for the core kmeans row. See docs/parity/core.md.

Three well-separated 2-D clusters, so that every initialisation converges to the same
partition and the centers and compactness can be compared independently of cv::RNG.

Run from the repository root:
    cpp_ref/.venv/Scripts/python scripts/opencv_ref/core_kmeans.py   # Windows
    cpp_ref/.venv/bin/python scripts/opencv_ref/core_kmeans.py       # Linux/macOS
"""

import cv2

assert cv2.__version__ == "4.10.0"
cv2.ipp.setUseIPP(False)  # measure the generic CPU path

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _core_common import dump, tolist  # noqa: E402

DATA = np.array(
    [
        [0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0],
        [10.0, 10.0], [11.0, 10.0], [10.0, 11.0], [11.0, 12.0],
        [0.0, 10.0], [1.0, 10.0], [0.0, 11.0], [2.0, 11.0],
    ],
    dtype=np.float32,
)
EPS_COUNT = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-4)


def attempt(fn):
    try:
        return tolist(fn())
    except cv2.error as e:  # noqa: PERF203
        return f"cv2.error: {e.code} {e.err}".strip()


def run(flags, attempts, labels=None, criteria=EPS_COUNT, k=3):
    cv2.setRNGSeed(42)
    compactness, lab, centers = cv2.kmeans(DATA, k, labels, criteria, attempts, flags)
    order = np.lexsort((centers[:, 1], centers[:, 0]))
    return {
        "compactness": float(compactness),
        "centers_sorted": tolist(centers[order]),
        "labels": tolist(lab.ravel()),
        "cluster_sizes_sorted": sorted(int(np.sum(lab == i)) for i in range(k)),
    }


def main() -> None:
    results = {"data": tolist(DATA)}
    results["pp_centers_attempts3"] = run(cv2.KMEANS_PP_CENTERS, 3)
    results["random_centers_attempts5"] = run(cv2.KMEANS_RANDOM_CENTERS, 5)
    results["random_centers_attempts3"] = run(cv2.KMEANS_RANDOM_CENTERS, 3)
    results["random_centers_attempts1"] = run(cv2.KMEANS_RANDOM_CENTERS, 1)
    init = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2], dtype=np.int32)
    results["use_initial_labels"] = run(cv2.KMEANS_USE_INITIAL_LABELS, 1, init.copy())
    bad_init = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], dtype=np.int32)
    results["use_initial_labels_unbalanced"] = run(cv2.KMEANS_USE_INITIAL_LABELS, 1, bad_init.copy())
    results["k1"] = run(cv2.KMEANS_PP_CENTERS, 1, k=1)
    results["count_only_criteria_1_iter"] = run(
        cv2.KMEANS_USE_INITIAL_LABELS, 1, init.copy(), criteria=(cv2.TERM_CRITERIA_MAX_ITER, 1, 0.0)
    )
    results["k_greater_than_n"] = attempt(
        lambda: cv2.kmeans(DATA, 13, None, (cv2.TERM_CRITERIA_MAX_ITER, 10, 0), 1, cv2.KMEANS_PP_CENTERS)
    )
    results["f64_data"] = attempt(
        lambda: cv2.kmeans(DATA.astype(np.float64), 3, None, (cv2.TERM_CRITERIA_MAX_ITER, 10, 0), 1, cv2.KMEANS_PP_CENTERS)
    )
    results["label_out_of_range_with_initial_labels"] = attempt(
        lambda: cv2.kmeans(
            DATA, 3, np.full(12, 7, np.int32), (cv2.TERM_CRITERIA_MAX_ITER, 10, 0), 1, cv2.KMEANS_USE_INITIAL_LABELS
        )
    )
    dump("core_kmeans.py", results)


if __name__ == "__main__":
    main()

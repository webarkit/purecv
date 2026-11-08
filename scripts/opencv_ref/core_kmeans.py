"""OpenCV 4.10.0 reference values for the core kmeans row. See docs/parity/4.10.0/core.md.

Three well-separated 2-D clusters, so that every initialisation converges to the same
partition and the centers and compactness can be compared independently of cv::RNG.
The ``seed42_*`` and ``default_state_*`` keys also pin the raw (unsorted) labels and centers,
which depend on the order of theRNG() draws (#221); the ``chain_*`` keys use a slowly
converging fixture for the criteria clamp and the epsilon test.

Run from the repository root:
    cpp_ref/.venv/Scripts/python scripts/opencv_ref/core_kmeans.py   # Windows
    cpp_ref/.venv/bin/python scripts/opencv_ref/core_kmeans.py       # Linux/macOS
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

DATA = np.array(
    [
        [0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0],
        [10.0, 10.0], [11.0, 10.0], [10.0, 11.0], [11.0, 12.0],
        [0.0, 10.0], [1.0, 10.0], [0.0, 11.0], [2.0, 11.0],
    ],
    dtype=np.float32,
)
EPS_COUNT = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-4)

# A 1-D chain on the diagonal (v, v): sample 0 alone in cluster 0, then Lloyd's iteration
# moves exactly one chain sample into cluster 0 per iteration and needs 103 iterations to
# converge. That makes the [2, 100] clamp of maxCount and the epsilon test observable. The
# chain was built by a shooting method (each sample just past the next midpoint) and rounded
# to two decimals; only cv2's output below is used as expected values.
CHAIN = [
    0.00, 15.89, 17.83, 22.11, 23.59, 25.02, 26.00, 26.80, 27.45, 28.00, 28.48, 28.89, 29.25,
    29.58, 29.88, 30.15, 30.40, 30.63, 30.85, 31.05, 31.24, 31.42, 31.59, 31.75, 31.90, 32.05,
    32.19, 32.32, 32.46, 32.58, 32.71, 32.82, 32.94, 33.05, 33.16, 33.27, 33.38, 33.48, 33.58,
    33.68, 33.78, 33.88, 33.97, 34.07, 34.16, 34.26, 34.35, 34.44, 34.53, 34.62, 34.71, 34.80,
    34.89, 34.98, 35.07, 35.16, 35.25, 35.34, 35.44, 35.53, 35.62, 35.71, 35.81, 35.90, 36.00,
    36.10, 36.20, 36.30, 36.41, 36.51, 36.62, 36.73, 36.84, 36.96, 37.08, 37.20, 37.33, 37.46,
    37.59, 37.74, 37.88, 38.04, 38.20, 38.37, 38.54, 38.73, 38.93, 39.15, 39.38, 39.62, 39.89,
    40.19, 40.51, 40.87, 41.27, 41.73, 42.27, 42.90, 43.66, 44.61, 45.86, 47.58, 50.24, 100.00,
]
CHAIN_DATA = np.array([[v, v] for v in CHAIN], dtype=np.float32)
CHAIN_INIT = np.array([0] + [1] * (len(CHAIN) - 1), dtype=np.int32)


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


def raw(data, k, labels, criteria, attempts, flags):
    """Unsorted labels and centers, so that the order of theRNG() draws is pinned."""
    compactness, lab, centers = cv2.kmeans(data, k, labels, criteria, attempts, flags)
    return {
        "compactness": float(compactness),
        "labels": tolist(lab.ravel()),
        "centers": tolist(centers.ravel()),
    }


def seeded(seed, data, k, labels, criteria, attempts, flags):
    cv2.setRNGSeed(seed)
    return raw(data, k, labels, criteria, attempts, flags)


def main() -> None:
    results = {"data": tolist(DATA)}
    init = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2], dtype=np.int32)

    # #221: cv2.setRNGSeed(42) before each call; raw labels and centers.
    flag_names = {
        "random": cv2.KMEANS_RANDOM_CENTERS,
        "pp": cv2.KMEANS_PP_CENTERS,
        "initial_labels": cv2.KMEANS_USE_INITIAL_LABELS,
    }
    for name, flags in flag_names.items():
        for attempts in (1, 3, 5):
            labels = init.copy() if name == "initial_labels" else None
            results[f"seed42_{name}_attempts{attempts}"] = seeded(42, DATA, 3, labels, EPS_COUNT, attempts, flags)
    for name in ("random", "pp"):
        results[f"seed42_chain_k3_{name}_attempts1"] = seeded(
            42, CHAIN_DATA, 3, None, EPS_COUNT, 1, flag_names[name]
        )
    # Seeds where a later attempt wins: with RANDOM_CENTERS the first attempt does not reach
    # the global partition (compactness 9.5), the third one does.
    for seed in (7, 19):
        for attempts in (1, 3):
            results[f"seed{seed}_random_attempts{attempts}"] = seeded(
                seed, DATA, 3, None, EPS_COUNT, attempts, cv2.KMEANS_RANDOM_CENTERS
            )
    # The default state of theRNG() (a fresh thread, RNG(0) = 0xffffffff), as a Rust test
    # thread that sets no seed sees it.
    for attempts in (1, 3, 5):
        results[f"default_state_random_attempts{attempts}"] = seeded(
            0, DATA, 3, None, EPS_COUNT, attempts, cv2.KMEANS_RANDOM_CENTERS
        )
    # The call sequence of kmeans_converges_to_opencv_partition from the default state.
    cv2.setRNGSeed(0)
    results["default_state_sequence"] = [
        raw(DATA, 3, None, EPS_COUNT, 3, cv2.KMEANS_PP_CENTERS),
        raw(DATA, 3, init.copy(), EPS_COUNT, 1, cv2.KMEANS_USE_INITIAL_LABELS),
        raw(DATA, 1, None, EPS_COUNT, 1, cv2.KMEANS_PP_CENTERS),
    ]
    # maxCount is clamped to [2, 100] (COUNT only); EPS only means maxCount 100.
    for max_count in (0, 1, 2, 3, 100, 1000):
        results[f"chain_count_only_max{max_count}"] = raw(
            CHAIN_DATA, 2, CHAIN_INIT.copy(), (cv2.TERM_CRITERIA_MAX_ITER, max_count, 0.0), 1,
            cv2.KMEANS_USE_INITIAL_LABELS,
        )
    # Epsilon: the squared per-center shift is compared with eps^2; on the diagonal a center
    # moves by sqrt(2) times its per-coordinate shift.
    for eps in (0.0, 0.15):
        results[f"chain_eps_only_{eps}"] = raw(
            CHAIN_DATA, 2, CHAIN_INIT.copy(), (cv2.TERM_CRITERIA_EPS, 0, eps), 1,
            cv2.KMEANS_USE_INITIAL_LABELS,
        )
    results["label_minus_one_with_initial_labels"] = attempt(
        lambda: cv2.kmeans(
            DATA, 3, np.array([-1] + [0] * 11, np.int32), (cv2.TERM_CRITERIA_MAX_ITER, 10, 0), 1,
            cv2.KMEANS_USE_INITIAL_LABELS,
        )
    )

    results["pp_centers_attempts3"] = run(cv2.KMEANS_PP_CENTERS, 3)
    results["random_centers_attempts5"] = run(cv2.KMEANS_RANDOM_CENTERS, 5)
    results["random_centers_attempts3"] = run(cv2.KMEANS_RANDOM_CENTERS, 3)
    results["random_centers_attempts1"] = run(cv2.KMEANS_RANDOM_CENTERS, 1)
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

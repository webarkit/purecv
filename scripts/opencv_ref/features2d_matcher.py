"""OpenCV 4.10.0 reference values for the matcher rows (docs/parity/features2d.md).

Rows: BFMatcher::{new, match_descriptors, knn_match}, NormType, DescriptorDistance,
filter_matches. `cv2.BFMatcher` on seeded u8 and f32 descriptors with constructed ties, so
the tie order of `match`, `knnMatch` and `crossCheck` is observed, not guessed.

Run from the repository root:
    cpp_ref/.venv/Scripts/python scripts/opencv_ref/features2d_matcher.py   # Windows
    cpp_ref/.venv/bin/python scripts/opencv_ref/features2d_matcher.py       # Linux/macOS
"""

import cv2

assert cv2.__version__ == "4.10.0"
cv2.ipp.setUseIPP(False)  # measure the generic CPU path

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _core_common import dump, tolist  # noqa: E402
from _features2d_common import dm_fields, noise_u8  # noqa: E402


def attempt(fn):
    try:
        return tolist(fn())
    except cv2.error as e:  # noqa: PERF203
        return f"cv2.error: {e.code} {e.err}".strip()


def dms(matches):
    return [dm_fields(m) for m in matches]


def knn(matches):
    return [[dm_fields(m) for m in row] for row in matches]


def u8_sets():
    """Query 6x32 and train 8x32 u8 descriptors with ties.

    train[5] == train[1] (a tie for every query nearest to train 1), query[4] == query[2]
    (a reverse tie for cross-check), train[7] == query[0] (an exact match, distance 0).
    """
    q = noise_u8(6, 32, 101).copy()
    t = noise_u8(8, 32, 202).copy()
    t[5] = t[1]
    q[4] = q[2]
    t[7] = q[0]
    return q, t


def f32_sets():
    """Query 5x4 and train 6x4 f32 descriptors from the integer LCG, scaled by 1/16.

    train[3] == train[0]; query[2] == query[1]; train[5] == query[4]."""
    q = (noise_u8(5, 4, 303).astype(np.float32) / 16.0).copy()
    t = (noise_u8(6, 4, 404).astype(np.float32) / 16.0).copy()
    t[3] = t[0]
    q[2] = q[1]
    t[5] = q[4]
    return q, t


def main() -> None:
    results = {}
    q8, t8 = u8_sets()
    q32, t32 = f32_sets()
    results["inputs"] = {"q8": tolist(q8), "t8": tolist(t8), "q32": tolist(q32), "t32": tolist(t32)}

    # --- Hamming on u8
    h = {}
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    h["match"] = dms(bf.match(q8, t8))
    h["knn_k1"] = knn(bf.knnMatch(q8, t8, k=1))
    h["knn_k2"] = knn(bf.knnMatch(q8, t8, k=2))
    h["knn_k3"] = knn(bf.knnMatch(q8, t8, k=3))
    h["knn_k20_more_than_train"] = knn(bf.knnMatch(q8, t8, k=20))
    h["knn_k0"] = attempt(lambda: knn(bf.knnMatch(q8, t8, k=0)))
    h["match_train_to_query"] = dms(bf.match(t8, q8))
    bfc = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    h["match_cross_check"] = dms(bfc.match(q8, t8))
    h["knn_k1_cross_check"] = knn(bfc.knnMatch(q8, t8, k=1))
    h["knn_k2_cross_check"] = attempt(lambda: knn(bfc.knnMatch(q8, t8, k=2)))
    h["match_empty_train"] = attempt(lambda: dms(bf.match(q8, np.zeros((0, 32), np.uint8))))
    h["match_empty_query"] = attempt(lambda: dms(bf.match(np.zeros((0, 32), np.uint8), t8)))
    h["match_width_mismatch"] = attempt(lambda: dms(bf.match(q8, t8[:, :16].copy())))
    h["match_f32_rejected"] = attempt(lambda: dms(bf.match(q32, t32)))
    h["match_type_mismatch"] = attempt(lambda: dms(bf.match(q8, t8.astype(np.float32))))
    # the full distance matrix, so a Rust test can check every distance, not only the best
    h["dist_matrix"] = [[int(np.unpackbits(q8[i] ^ t8[j]).sum()) for j in range(len(t8))] for i in range(len(q8))]
    # radiusMatch, NORM_HAMMING2 and NORM_L1 on u8 (missing in purecv)
    h["radius_match_r100"] = knn(bf.radiusMatch(q8, t8, maxDistance=100.0))
    h["hamming2_match"] = dms(cv2.BFMatcher(cv2.NORM_HAMMING2).match(q8, t8))
    h["l1_match"] = dms(cv2.BFMatcher(cv2.NORM_L1).match(q8, t8))
    h["l2sqr_match"] = dms(cv2.BFMatcher(cv2.NORM_L2SQR).match(q8, t8))
    h["default_norm_is_l2"] = cv2.BFMatcher().match(q8, t8) is not None and dms(cv2.BFMatcher().match(q8, t8)) == dms(
        cv2.BFMatcher(cv2.NORM_L2).match(q8, t8)
    )
    h["norm_values"] = {
        "NORM_L1": cv2.NORM_L1,
        "NORM_L2": cv2.NORM_L2,
        "NORM_L2SQR": cv2.NORM_L2SQR,
        "NORM_HAMMING": cv2.NORM_HAMMING,
        "NORM_HAMMING2": cv2.NORM_HAMMING2,
        "NORM_INF": cv2.NORM_INF,
    }
    h["inf_rejected"] = attempt(lambda: dms(cv2.BFMatcher(cv2.NORM_INF).match(q8, t8)))
    results["hamming_u8"] = h

    # --- L2 on u8 and on f32
    l2 = {}
    bf2 = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    l2["u8_match"] = dms(bf2.match(q8, t8))
    l2["u8_knn_k2"] = knn(bf2.knnMatch(q8, t8, k=2))
    l2["u8_dist_matrix"] = [
        [float(np.sqrt(np.sum((q8[i].astype(np.float64) - t8[j].astype(np.float64)) ** 2))) for j in range(len(t8))]
        for i in range(len(q8))
    ]
    l2["f32_match"] = dms(bf2.match(q32, t32))
    l2["f32_knn_k2"] = knn(bf2.knnMatch(q32, t32, k=2))
    l2["f32_knn_k3"] = knn(bf2.knnMatch(q32, t32, k=3))
    l2["f32_match_cross_check"] = dms(cv2.BFMatcher(cv2.NORM_L2, crossCheck=True).match(q32, t32))
    l2["f32_match_train_to_query"] = dms(bf2.match(t32, q32))
    l2["f32_dist_matrix"] = [
        [float(np.sqrt(np.sum((q32[i].astype(np.float64) - t32[j].astype(np.float64)) ** 2))) for j in range(len(t32))]
        for i in range(len(q32))
    ]
    l2["f64_rejected"] = attempt(lambda: dms(bf2.match(q32.astype(np.float64), t32.astype(np.float64))))
    results["l2"] = l2

    # --- Lowe's ratio test (no OpenCV counterpart: filter_matches is purecv-only). The knn
    # rows above are enough to apply it by hand; nothing to measure.

    dump("features2d_matcher.py", results)


if __name__ == "__main__":
    main()

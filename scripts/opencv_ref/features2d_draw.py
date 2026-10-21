"""OpenCV 4.10.0 reference values for the drawing rows (docs/parity/4.10.0/features2d.md).

Rows: draw_keypoints, draw_matches. `cv2.drawKeypoints` / `cv2.drawMatches` with fixed
colours on small seeded images; the changed pixels are listed so that
tests/parity_features2d.rs can compare the drawn set, the output size and the channel count.

Run from the repository root:
    cpp_ref/.venv/Scripts/python scripts/opencv_ref/features2d_draw.py   # Windows
    cpp_ref/.venv/bin/python scripts/opencv_ref/features2d_draw.py       # Linux/macOS
"""

import cv2

assert cv2.__version__ == "4.10.0"
cv2.ipp.setUseIPP(False)  # measure the generic CPU path

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _core_common import dump, tolist  # noqa: E402
from _features2d_common import noise_u8  # noqa: E402


def attempt(fn):
    try:
        return tolist(fn())
    except cv2.error as e:  # noqa: PERF203
        return f"cv2.error: {e.code} {e.err}".strip()


def changed(before_bgr, after):
    """Pixels of `after` that differ from `before_bgr`, as (y, x, [channels...])."""
    out = []
    diff = np.any(after != before_bgr, axis=2)
    for y, x in zip(*np.nonzero(diff)):
        out.append((int(y), int(x), [int(v) for v in after[y, x]]))
    return out


def kp(x, y, size=7.0, angle=-1.0):
    return cv2.KeyPoint(float(x), float(y), float(size), float(angle), 0.0, 0, -1)


def main() -> None:
    results = {}
    gray = noise_u8(20, 24, 31)
    gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    bgr = cv2.merge([noise_u8(20, 24, 41), noise_u8(20, 24, 42), noise_u8(20, 24, 43)])
    results["inputs"] = {"gray_20x24_s31": tolist(gray), "bgr_20x24_s41_42_43": tolist(bgr)}

    # --- drawKeypoints
    dk = {}
    red = (0, 0, 255)  # BGR
    out = cv2.drawKeypoints(gray, [kp(10.4, 8.6)], None, color=red)
    dk["gray_default_flags"] = {"shape": list(out.shape), "changed": changed(gray_bgr, out)}
    out = cv2.drawKeypoints(gray, [kp(10.4, 8.6)], None, color=red, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    dk["gray_rich_size7_no_angle"] = {"shape": list(out.shape), "changed": changed(gray_bgr, out)}
    out = cv2.drawKeypoints(gray, [kp(10.4, 8.6, 7.0, 30.0)], None, color=red, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    dk["gray_rich_size7_angle30"] = {"shape": list(out.shape), "changed": changed(gray_bgr, out)}
    out = cv2.drawKeypoints(gray, [kp(10.0, 8.0, 12.0, 0.0)], None, color=red, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    dk["gray_rich_size12_angle0_integer_centre"] = {"shape": list(out.shape), "changed": changed(gray_bgr, out)}
    out = cv2.drawKeypoints(bgr, [kp(10.0, 8.0)], None, color=red)
    dk["bgr_default_flags"] = {"shape": list(out.shape), "changed": changed(bgr, out)}
    out = cv2.drawKeypoints(gray, [], None, color=red)
    dk["gray_no_keypoints"] = {"shape": list(out.shape), "equals_gray2bgr": bool(np.array_equal(out, gray_bgr))}
    out = cv2.drawKeypoints(bgr, [], None, color=red)
    dk["bgr_no_keypoints_unchanged"] = bool(np.array_equal(out, bgr))
    dk["f32_rejected"] = attempt(lambda: cv2.drawKeypoints(gray.astype(np.float32), [kp(10, 8)], None, color=red).shape)
    # random colours (Scalar::all(-1)): two calls in one process differ (theRNG advances)
    a = cv2.drawKeypoints(gray, [kp(10, 8)], None)
    b = cv2.drawKeypoints(gray, [kp(10, 8)], None)
    dk["random_colour_two_calls_equal"] = bool(np.array_equal(a, b))
    # DRAW_OVER_OUTIMG draws into the given image
    canvas = gray_bgr.copy()
    cv2.drawKeypoints(gray, [kp(10, 8)], canvas, color=red, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_OVER_OUTIMG)
    dk["draw_over_outimg_changed_count"] = len(changed(gray_bgr, canvas))
    results["draw_keypoints"] = dk

    # --- drawMatches
    dm = {}
    img1 = noise_u8(16, 20, 51)
    img2 = noise_u8(14, 24, 52)
    img1_bgr = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    img2_bgr = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
    canvas = np.zeros((16, 44, 3), np.uint8)
    canvas[:16, :20] = img1_bgr
    canvas[:14, 20:44] = img2_bgr
    results["inputs"]["img1_16x20_s51"] = tolist(img1)
    results["inputs"]["img2_14x24_s52"] = tolist(img2)
    k1 = [kp(5.0, 6.0), kp(14.0, 3.0)]
    k2 = [kp(7.0, 9.0), kp(20.0, 2.0)]
    m = [cv2.DMatch(0, 0, 0, 1.0)]
    blue, green = (255, 0, 0), (0, 255, 0)
    out = cv2.drawMatches(img1, k1, img2, k2, m, None, matchColor=blue, singlePointColor=green)
    dm["gray_one_match_default_flags"] = {"shape": list(out.shape), "changed": changed(canvas, out)}
    out = cv2.drawMatches(
        img1, k1, img2, k2, m, None, matchColor=blue, singlePointColor=green, flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS
    )
    dm["gray_one_match_not_draw_single"] = {"shape": list(out.shape), "changed": changed(canvas, out)}
    out = cv2.drawMatches(img1, [], img2, [], [], None, matchColor=blue, singlePointColor=green)
    dm["gray_no_keypoints_canvas"] = {"shape": list(out.shape), "equals_canvas": bool(np.array_equal(out, canvas))}
    out = cv2.drawMatches(img1_bgr, [], img2_bgr, [], [], None, matchColor=blue, singlePointColor=green)
    dm["bgr_no_keypoints_canvas"] = {"shape": list(out.shape), "equals_canvas": bool(np.array_equal(out, canvas))}
    out = cv2.drawMatches(img1, [], img2_bgr, [], [], None, matchColor=blue, singlePointColor=green)
    dm["mixed_gray_bgr_canvas"] = {"shape": list(out.shape), "equals_canvas": bool(np.array_equal(out, canvas))}
    dm["match_index_out_of_range"] = attempt(
        lambda: cv2.drawMatches(img1, k1, img2, k2, [cv2.DMatch(0, 5, 0, 1.0)], None, matchColor=blue, singlePointColor=green).shape
    )
    dm["match_negative_index"] = attempt(
        lambda: cv2.drawMatches(img1, k1, img2, k2, [cv2.DMatch(-1, 0, 0, 1.0)], None, matchColor=blue, singlePointColor=green).shape
    )
    a = cv2.drawMatches(img1, k1, img2, k2, m, None)
    b = cv2.drawMatches(img1, k1, img2, k2, m, None)
    dm["random_colour_two_calls_equal"] = bool(np.array_equal(a, b))
    results["draw_matches"] = dm
    results["flags"] = {
        "DEFAULT": cv2.DRAW_MATCHES_FLAGS_DEFAULT,
        "DRAW_OVER_OUTIMG": cv2.DRAW_MATCHES_FLAGS_DRAW_OVER_OUTIMG,
        "NOT_DRAW_SINGLE_POINTS": cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS,
        "DRAW_RICH_KEYPOINTS": cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
    }

    dump("features2d_draw.py", results)


if __name__ == "__main__":
    main()

"""OpenCV 4.10.0 reference values for the FAST rows (docs/parity/4.10.0/features2d.md).

Rows: FastFeatureDetector::{new, detect}, FastType. Every `(type, nonmax, threshold)`
combination of #161 is run on three seeded images; the keypoint lists are reported as a
count, an FNV-1a digest and a prefix (see _features2d_common.py), so that
tests/parity_features2d.rs can pin them.

Run from the repository root:
    cpp_ref/.venv/Scripts/python scripts/opencv_ref/features2d_fast.py   # Windows
    cpp_ref/.venv/bin/python scripts/opencv_ref/features2d_fast.py       # Linux/macOS
"""

import cv2

assert cv2.__version__ == "4.10.0"
cv2.ipp.setUseIPP(False)  # measure the generic CPU path

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _core_common import dump, tolist  # noqa: E402
from _features2d_common import (  # noqa: E402
    fnv1a32,
    kp_digest,
    kp_list,
    noise_u8,
    squares_u8,
    textured_u8,
)

TYPES = {
    "9_16": cv2.FAST_FEATURE_DETECTOR_TYPE_9_16,
    "7_12": cv2.FAST_FEATURE_DETECTOR_TYPE_7_12,
    "5_8": cv2.FAST_FEATURE_DETECTOR_TYPE_5_8,
}


def attempt(fn):
    try:
        return tolist(fn())
    except cv2.error as e:  # noqa: PERF203
        return f"cv2.error: {e.code} {e.err}".strip()


def detect(img, threshold, nonmax, type_):
    det = cv2.FastFeatureDetector_create(threshold=threshold, nonmaxSuppression=nonmax, type=type_)
    return det.detect(img, None)


def main() -> None:
    results = {}

    images = {
        "noise_64x80_s7": noise_u8(64, 80, 7),
        "textured_48x64_s3": textured_u8(48, 64, 3),
        "squares_40x48": squares_u8(),
    }
    # pixel digests so that a Rust twin of the generators can be checked
    results["inputs"] = {
        name: {"shape": list(img.shape), "fnv1a_pixels": f"0x{fnv1a32(img.tobytes()):08x}"}
        for name, img in images.items()
    }
    results["inputs"]["squares_40x48"]["pixels"] = tolist(images["squares_40x48"])

    # --- the #161 grid: type x nonmax x threshold, per image
    grid = {}
    for img_name, img in images.items():
        for type_name, type_ in TYPES.items():
            for nonmax in (True, False):
                for t in (0, 10, 20, 40):
                    kps = detect(img, t, nonmax, type_)
                    key = f"{img_name}/{type_name}/nonmax_{'on' if nonmax else 'off'}/t{t}"
                    d = kp_digest(kps)
                    if img_name == "squares_40x48":
                        d["all"] = kp_list(kps)
                    else:
                        xs = [k.pt[0] for k in kps]
                        ys = [k.pt[1] for k in kps]
                        d["x_range"] = [min(xs), max(xs)] if kps else None
                        d["y_range"] = [min(ys), max(ys)] if kps else None
                        d["responses_nonzero"] = int(sum(1 for k in kps if k.response != 0))
                    grid[key] = d
    results["grid"] = grid

    # --- D1/D2: defaults, threshold clamp, input types, mask, small images
    misc = {}
    det = cv2.FastFeatureDetector_create()
    misc["defaults"] = {
        "threshold": det.getThreshold(),
        "nonmaxSuppression": det.getNonmaxSuppression(),
        "type": det.getType(),
        "type_9_16_value": cv2.FAST_FEATURE_DETECTOR_TYPE_9_16,
        "type_7_12_value": cv2.FAST_FEATURE_DETECTOR_TYPE_7_12,
        "type_5_8_value": cv2.FAST_FEATURE_DETECTOR_TYPE_5_8,
        "default_name": det.getDefaultName(),
    }
    noise = images["noise_64x80_s7"]
    misc["t_minus5_equals_t0"] = kp_digest(detect(noise, -5, True, TYPES["9_16"]))
    misc["t_300_equals_t255"] = kp_digest(detect(noise, 300, True, TYPES["9_16"]))
    misc["t_255"] = kp_digest(detect(noise, 255, True, TYPES["9_16"]))
    bgr = cv2.cvtColor(noise, cv2.COLOR_GRAY2BGR)
    misc["bgr_input_converted_to_gray"] = kp_digest(detect(bgr, 20, True, TYPES["9_16"]))
    misc["gray_t20"] = kp_digest(detect(noise, 20, True, TYPES["9_16"]))
    misc["f32_input"] = attempt(lambda: kp_digest(detect(noise.astype(np.float32), 20, True, TYPES["9_16"])))
    misc["u16_input"] = attempt(lambda: kp_digest(detect(noise.astype(np.uint16), 20, True, TYPES["9_16"])))
    mask = np.zeros_like(noise)
    mask[:, :40] = 255
    kps_masked = cv2.FastFeatureDetector_create(20, True, TYPES["9_16"]).detect(noise, mask)
    misc["mask_left_half"] = kp_digest(kps_masked)
    misc["mask_left_half_max_x"] = max(k.pt[0] for k in kps_masked) if kps_masked else None
    for n in (5, 6, 7, 8):
        tiny = textured_u8(n, n, 11)
        for type_name, type_ in TYPES.items():
            misc[f"tiny_{n}x{n}/{type_name}/nonmax_off/t10"] = kp_digest(detect(tiny, 10, False, type_))["count"]
    # OpenCV's scan range (rows/cols 3..=dim-4) for every pattern size
    ranges = {}
    for type_name, type_ in TYPES.items():
        kps = detect(noise, 0, False, type_)
        ranges[type_name] = {
            "x_min": min(k.pt[0] for k in kps),
            "x_max": max(k.pt[0] for k in kps),
            "y_min": min(k.pt[1] for k in kps),
            "y_max": max(k.pt[1] for k in kps),
        }
    misc["scan_ranges_t0_nonmax_off"] = ranges
    # the `cv::FAST` free function is not bound by cv2 (`CV_EXPORTS`, not `CV_EXPORTS_W`);
    # FastFeatureDetector_Impl::detect calls it, so the detector is the measurement.
    results["misc"] = misc

    # --- a hand-built 5-of-8 arc and a 7-of-12 arc (gap (d) of #161 / #202): with
    # nonmax off, does OpenCV report the centre pixel as a corner?
    arcs = {}

    def ring_image(offsets, bright_idx, size=9):
        img = np.full((size, size), 100, np.uint8)
        c = size // 2
        for i in bright_idx:
            dx, dy = offsets[i]
            img[c + dy, c + dx] = 200
        return img

    off8 = [(0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1)]
    off12 = [(0, 2), (1, 2), (2, 1), (2, 0), (2, -1), (1, -2), (0, -2), (-1, -2), (-2, -1), (-2, 0), (-2, 1), (-1, 2)]
    # purecv's circle order is (dx, dy) = (0, r), (1, r), ... (clockwise from the bottom);
    # OpenCV's offsets start at (0, -r). Either way the arc is contiguous.
    for start in range(8):
        arc = [(start + i) % 8 for i in range(5)]
        kps = detect(ring_image(off8, arc), 10, False, TYPES["5_8"])
        arcs[f"5_of_8/start{start}"] = [(k.pt[0], k.pt[1], k.response) for k in kps]
    arcs["8_of_8"] = [(k.pt[0], k.pt[1], k.response) for k in detect(ring_image(off8, range(8)), 10, False, TYPES["5_8"])]
    for start in range(12):
        arc = [(start + i) % 12 for i in range(7)]
        kps = detect(ring_image(off12, arc), 10, False, TYPES["7_12"])
        arcs[f"7_of_12/start{start}"] = [(k.pt[0], k.pt[1], k.response) for k in kps]
    arcs["12_of_12"] = [
        (k.pt[0], k.pt[1], k.response) for k in detect(ring_image(off12, range(12)), 10, False, TYPES["7_12"])
    ]
    results["arcs"] = arcs

    dump("features2d_fast.py", results)


if __name__ == "__main__":
    main()

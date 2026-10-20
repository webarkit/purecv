"""OpenCV 4.10.0 reference values for the ORB rows (docs/parity/features2d.md).

Rows: Orb::{new, detect, compute, detect_and_compute}, ScoreType, build_orb_pyramid,
compute_orientation, compute_orb_descriptor, precompute_umax. `cv2.ORB_create()` with the
defaults, on seeded images; keypoint lists are reported in full (for the overlap analysis)
and as digests (for tests/parity_features2d.rs). The 40-keypoint `compute()` fixture is the
one of `scripts/opencv_ref/gen_orb_descriptors.js`, so the cv2 descriptors can be compared
with the committed opencv.js fixture `src/features2d/orb_opencv_ref.rs`.

Run from the repository root:
    cpp_ref/.venv/Scripts/python scripts/opencv_ref/features2d_orb.py   # Windows
    cpp_ref/.venv/bin/python scripts/opencv_ref/features2d_orb.py       # Linux/macOS
"""

import cv2

assert cv2.__version__ == "4.10.0"
cv2.ipp.setUseIPP(False)  # measure the generic CPU path

import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _core_common import dump, tolist  # noqa: E402
from _features2d_common import (  # noqa: E402
    fnv1a32,
    kp_digest,
    kp_fields,
    kp_list,
    noise_u8,
    orb_ref_image,
    textured_u8,
)

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FIXTURE_RS = os.path.join(REPO, "src", "features2d", "orb_opencv_ref.rs")

# the 40 level-0 keypoints of gen_orb_descriptors.js (`makeKeypoints()`)
ANGLES = [0, 17.5, 90, 133.3, 270.9, 45, 180, 359.5, 222.2, 300.1,
          11.25, 60, 120, 243.7, 315, 5.5, 72.8, 199.9, 30, 150]  # fmt: skip


def fixture_keypoints():
    kps = []
    for i in range(40):
        kp = cv2.KeyPoint(
            float(40 + (i * 37 + 11) % 112),
            float(40 + (i * 59 + 23) % 112),
            31.0,
            float(np.float32(ANGLES[i % len(ANGLES)])),
            0.0,
            0,
            -1,
        )
        kps.append(kp)
    return kps


def attempt(fn):
    try:
        return tolist(fn())
    except cv2.error as e:  # noqa: PERF203
        return f"cv2.error: {e.code} {e.err}".strip()


def per_octave(kps):
    counts = {}
    for k in kps:
        counts[k.octave] = counts.get(k.octave, 0) + 1
    return [counts.get(o, 0) for o in range(max(counts) + 1)] if counts else []


def desc_digest(desc):
    return {
        "rows": 0 if desc is None else int(desc.shape[0]),
        "cols": 0 if desc is None else int(desc.shape[1]),
        "fnv1a": f"0x{fnv1a32(b'' if desc is None else desc.tobytes()):08x}",
    }


def read_fixture_descriptors():
    """The `ORB_REF_DESCRIPTORS` rows of src/features2d/orb_opencv_ref.rs (opencv.js)."""
    with open(FIXTURE_RS, encoding="utf-8") as f:
        text = f.read()
    body = text.split("ORB_REF_DESCRIPTORS", 1)[1]
    rows = re.findall(r"\[((?:0x[0-9a-f]{2}, ){31}0x[0-9a-f]{2})\]", body)
    return np.array([[int(h, 16) for h in r.split(", ")] for r in rows], np.uint8)


def main() -> None:
    results = {}

    ref = orb_ref_image()
    textured = textured_u8(240, 320, 5)
    noise = noise_u8(240, 320, 9)
    results["inputs"] = {
        "orb_ref_192x192": {"fnv1a_pixels": f"0x{fnv1a32(ref.tobytes()):08x}"},
        "textured_240x320_s5": {"fnv1a_pixels": f"0x{fnv1a32(textured.tobytes()):08x}"},
        "noise_240x320_s9": {"fnv1a_pixels": f"0x{fnv1a32(noise.tobytes()):08x}"},
    }

    orb = cv2.ORB_create()
    results["defaults"] = {
        "nfeatures": orb.getMaxFeatures(),
        "scaleFactor": orb.getScaleFactor(),
        "nlevels": orb.getNLevels(),
        "edgeThreshold": orb.getEdgeThreshold(),
        "firstLevel": orb.getFirstLevel(),
        "WTA_K": orb.getWTA_K(),
        "scoreType": orb.getScoreType(),
        "patchSize": orb.getPatchSize(),
        "fastThreshold": orb.getFastThreshold(),
        "descriptorSize": orb.descriptorSize(),
        "descriptorType": orb.descriptorType(),
        "defaultNorm": orb.defaultNorm(),
        "NORM_HAMMING": cv2.NORM_HAMMING,
        "NORM_HAMMING2": cv2.NORM_HAMMING2,
        "CV_8U": cv2.CV_8U,
        "HARRIS_SCORE": cv2.ORB_HARRIS_SCORE,
        "FAST_SCORE": cv2.ORB_FAST_SCORE,
        "defaultNorm_wta3": cv2.ORB_create(WTA_K=3).defaultNorm(),
        "defaultNorm_wta4": cv2.ORB_create(WTA_K=4).defaultNorm(),
        "default_name": orb.getDefaultName(),
    }

    # --- detect / detectAndCompute with the defaults
    det = {}
    for name, img in (("orb_ref_192x192", ref), ("textured_240x320_s5", textured), ("noise_240x320_s9", noise)):
        kps, desc = orb.detectAndCompute(img, None)
        kps_only = orb.detect(img, None)
        d = kp_digest(kps)
        d["per_octave"] = per_octave(kps)
        d["sizes_per_octave"] = sorted({(k.octave, k.size) for k in kps})
        d["detect_equals_detectAndCompute"] = kp_digest(kps_only) == kp_digest(kps)
        d["descriptors"] = desc_digest(desc)
        d["all"] = kp_list(kps)
        d["octave0_xy"] = sorted((k.pt[0], k.pt[1]) for k in kps if k.octave == 0)
        # the octave-0 keypoints and their descriptor rows, so that purecv's `compute()` can be
        # run on OpenCV's own keypoints (same positions and angles) and compared row by row
        o0 = [i for i, k in enumerate(kps) if k.octave == 0]
        d["octave0_keypoints"] = [kp_fields(kps[i]) for i in o0]
        d["octave0_descriptors_hex"] = [desc[i].tobytes().hex() for i in o0]
        det[name] = d
    # detect-only keypoints then compute(): the same descriptors as detectAndCompute
    kps_only = orb.detect(textured, None)
    _, desc2 = orb.compute(textured, kps_only)
    det["textured_240x320_s5"]["detect_then_compute_equals"] = (
        desc_digest(desc2) == det["textured_240x320_s5"]["descriptors"]
    )
    results["detect"] = det

    # --- FAST score, other parameters (digests and per-octave counts only)
    var = {}
    for label, kwargs in (
        ("fast_score", {"scoreType": cv2.ORB_FAST_SCORE}),
        ("nfeatures_100", {"nfeatures": 100}),
        ("nlevels_4_scale_1.5", {"nlevels": 4, "scaleFactor": 1.5}),
        ("edge_0", {"edgeThreshold": 0}),
        ("edge_3", {"edgeThreshold": 3}),
        ("edge_10", {"edgeThreshold": 10}),
        ("edge_14", {"edgeThreshold": 14}),
        ("edge_15", {"edgeThreshold": 15}),
        ("fast_threshold_40", {"fastThreshold": 40}),
        ("first_level_2", {"firstLevel": 2}),
        ("patch_21_edge_21", {"patchSize": 21, "edgeThreshold": 21}),
        ("wta_3", {"WTA_K": 3}),
        ("wta_4", {"WTA_K": 4}),
    ):
        o = cv2.ORB_create(**kwargs)
        kps, desc = o.detectAndCompute(textured, None)
        d = kp_digest(kps)
        d["per_octave"] = per_octave(kps)
        d["descriptors"] = desc_digest(desc)
        d["sizes_octave0"] = sorted({k.size for k in kps if k.octave == 0})
        var[label] = d
    var["scale_factor_1.0"] = attempt(lambda: per_octave(cv2.ORB_create(scaleFactor=1.0, nlevels=3).detect(textured, None)))
    var["scale_factor_0.5"] = attempt(lambda: per_octave(cv2.ORB_create(scaleFactor=0.5, nlevels=3).detect(textured, None)))
    var["nlevels_0"] = attempt(lambda: per_octave(cv2.ORB_create(nlevels=0).detect(textured, None)))
    var["patch_1"] = attempt(lambda: per_octave(cv2.ORB_create(patchSize=1).detect(textured, None)))
    var["wta_5"] = attempt(lambda: desc_digest(cv2.ORB_create(WTA_K=5).detectAndCompute(textured, None)[1]))
    var["wta_5_detect_only"] = attempt(lambda: per_octave(cv2.ORB_create(WTA_K=5).detect(textured, None)))
    var["first_level_minus1"] = attempt(lambda: per_octave(cv2.ORB_create(firstLevel=-1).detect(textured, None)))
    var["bgr_input"] = kp_digest(orb.detect(cv2.cvtColor(textured, cv2.COLOR_GRAY2BGR), None))
    var["gray_input"] = kp_digest(orb.detect(textured, None))
    var["f32_input"] = attempt(lambda: kp_digest(orb.detect(textured.astype(np.float32), None)))
    mask = np.zeros_like(textured)
    mask[:, :160] = 255
    var["mask_left_half"] = kp_digest(orb.detect(textured, mask))
    var["tiny_8x8"] = attempt(lambda: per_octave(orb.detect(textured_u8(8, 8, 1), None)))
    var["tiny_40x40"] = attempt(lambda: per_octave(orb.detect(textured_u8(40, 40, 1), None)))
    results["variants"] = var

    # --- compute() on the 40-keypoint fixture (gen_orb_descriptors.js), defaults
    fx = {}
    kps = fixture_keypoints()
    kps_out, desc = orb.compute(ref, kps)
    fx["keypoints_unchanged"] = kp_digest(kps_out) == kp_digest(kps)
    fx["descriptors"] = tolist(desc)
    fx["descriptors_digest"] = desc_digest(desc)
    try:
        js = read_fixture_descriptors()
        diff_rows = [int(i) for i in range(len(js)) if not np.array_equal(js[i], desc[i])]
        fx["opencv_js_fixture"] = {
            "rows": len(js),
            "identical_to_cv2": not diff_rows and len(js) == len(desc),
            "differing_rows": diff_rows,
            "hamming_per_row": [int(np.unpackbits(js[i] ^ desc[i]).sum()) for i in range(min(len(js), len(desc)))],
        }
    except (OSError, IndexError) as e:
        fx["opencv_js_fixture"] = f"not compared: {e}"
    # the same keypoints with patch 21 / WTA_K 3 and 4 (digests only: purecv rejects them)
    for label, kwargs in (("patch_21", {"patchSize": 21}), ("wta_3", {"WTA_K": 3}), ("wta_4", {"WTA_K": 4})):
        _, d2 = cv2.ORB_create(**kwargs).compute(ref, fixture_keypoints())
        fx[f"descriptors_{label}"] = desc_digest(d2)
    results["compute_fixture"] = fx

    # --- compute() with caller-supplied keypoints: border filter, regrouping, octaves
    cp = {}

    def kp(x, y, octave=0, size=31.0, angle=0.0):
        return cv2.KeyPoint(float(x), float(y), size, angle, 0.0, octave, -1)

    seven = [kp(5, 5), kp(30.5, 100), kp(31, 100), kp(100, 100), kp(160, 120, 3), kp(288.5, 100), kp(289, 100)]
    out, desc = orb.compute(textured, seven)
    cp["seven_border_filter"] = {"kept": kp_list(out), "rows": desc_digest(desc)["rows"]}
    three = [kp(100, 100, 3), kp(120, 120, 0), kp(140, 140, 9)]
    out, desc = orb.compute(textured, three)
    cp["octaves_3_0_9_regrouped"] = {"kept": kp_list(out), "rows": desc_digest(desc)["rows"]}
    out, desc = orb.compute(textured, [kp(100, 100, 12)])
    cp["octave_12_with_nlevels_8"] = {"kept": kp_list(out), "rows": desc_digest(desc)["rows"]}
    cp["octave_minus1"] = attempt(lambda: orb.compute(textured, [kp(100, 100, -1)])[1])
    out, desc = orb.compute(textured, [])
    cp["empty_keypoints"] = {"kept": kp_list(out), "desc_is_none": desc is None}
    out, desc = orb.compute(textured, [kp(100, 100, 0, 7.0, -1.0)])
    cp["angle_minus1_size7"] = {"kept": kp_list(out), "desc": tolist(desc)}
    results["compute_provided"] = cp

    # --- the per-level budget and the level scales, as OpenCV computes them (emulated in
    # numpy f32, cited rows #197/#193): pinned here so the Rust tests can compare
    bud = {}
    for nfeatures, nlevels, sf in ((500, 8, 1.2), (4530, 9, 1.2), (1000, 8, 1.2), (100, 4, 1.5)):
        factor = np.float32(1.0 / sf)
        nd = np.float32(nfeatures * (1 - factor) / (1 - np.float32(float(factor) ** nlevels)))
        levels = []
        s = 0
        for _ in range(nlevels - 1):
            n = round(float(nd))  # cvRound: ties to even, as Python's round
            levels.append(n)
            s += n
            nd = np.float32(nd * factor)
        levels.append(max(nfeatures - s, 0))
        bud[f"n{nfeatures}_l{nlevels}_s{sf}"] = levels
    results["budget_emulated"] = bud
    results["level_sizes_emulated"] = {
        "483_at_level1_sf1.2": int(np.rint(np.float32(483) * np.float32(1.0 / np.float32(1.2)))),
        "493_at_level1_sf2.0": int(np.rint(np.float32(493) * np.float32(1.0 / np.float32(2.0)))),
    }

    dump("features2d_orb.py", results)


if __name__ == "__main__":
    main()

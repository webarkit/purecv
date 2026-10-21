"""OpenCV 4.10.0 reference values for the pyramid rows (docs/parity/4.10.0/video.md).

Rows: build_optical_flow_pyramid, OpticalFlowPyramid. `cv2.buildOpticalFlowPyramid` is run on
seeded images for several window sizes, depths, border modes and `withDerivatives` settings;
the level count, the sizes and types of every level, and FNV-1a digests of the level pixels
and of the Scharr derivative planes are recorded so that tests/parity_video.rs can pin them.

The Python binding returns each level as a contiguous copy of its ROI, so the padding that
`pyrBorder` / `derivBorder` write around the levels is not visible here; it is inspected from
the source and exercised through `video_lk.py` (`asymmetric_borders`).

Run from the repository root:
    cpp_ref/.venv/Scripts/python scripts/opencv_ref/video_pyramid.py   # Windows
    cpp_ref/.venv/bin/python scripts/opencv_ref/video_pyramid.py       # Linux/macOS
"""

import cv2

assert cv2.__version__ == "4.10.0"
cv2.ipp.setUseIPP(False)  # measure the generic CPU path

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _core_common import dump, tolist  # noqa: E402
from _video_common import (  # noqa: E402
    cell_texture,
    pixel_digest,
    reusable_texture,
    textured_u8,
)

BORDERS = {
    "reflect101": cv2.BORDER_REFLECT_101,
    "replicate": cv2.BORDER_REPLICATE,
    "constant": cv2.BORDER_CONSTANT,
    "wrap": cv2.BORDER_WRAP,
    "reflect": cv2.BORDER_REFLECT,
}


def attempt(fn):
    try:
        return tolist(fn())
    except cv2.error as e:  # noqa: PERF203
        return f"cv2.error: {e.code} {e.err}".strip()


def describe(levels, with_derivatives):
    """Sizes, types and digests of a returned pyramid list."""
    out = []
    step = 2 if with_derivatives else 1
    for i, m in enumerate(levels):
        entry = {"shape": list(m.shape), "dtype": str(m.dtype)}
        if with_derivatives and i % 2 == 1:
            entry["fnv1a_dx"] = pixel_digest(m[:, :, 0])
            entry["fnv1a_dy"] = pixel_digest(m[:, :, 1])
            entry["dx_range"] = [int(m[:, :, 0].min()), int(m[:, :, 0].max())]
            entry["dy_range"] = [int(m[:, :, 1].min()), int(m[:, :, 1].max())]
        else:
            entry["fnv1a_pixels"] = pixel_digest(m)
        out.append(entry)
    return {"mats": len(levels), "level_step": step, "levels": out}


def build(img, win, max_level, with_derivatives=True, pyr_border=cv2.BORDER_REFLECT_101,
          deriv_border=cv2.BORDER_CONSTANT, try_reuse=True):
    n, pyr = cv2.buildOpticalFlowPyramid(
        img, win, max_level, None, with_derivatives, pyr_border, deriv_border, try_reuse
    )
    return {"returned_level": int(n), **describe(pyr, with_derivatives)}


def main() -> None:
    results = {}

    images = {
        "cell_320x240": cell_texture(320, 240),
        "textured_96x128_s11": textured_u8(96, 128, 11),
        "reusable_33x49": reusable_texture(33, 49, 0),
        "reusable_9x9": reusable_texture(9, 9, 0),
        "reusable_1x1": reusable_texture(1, 1, 0),
        "reusable_1x7": reusable_texture(1, 7, 0),
        "reusable_3x2": reusable_texture(3, 2, 0),
        "reusable_7x5": reusable_texture(7, 5, 0),
    }
    results["inputs"] = {
        name: {"shape": list(img.shape), "fnv1a_pixels": pixel_digest(img)}
        for name, img in images.items()
    }
    results["inputs"]["reusable_9x9"]["pixels"] = tolist(images["reusable_9x9"])

    # --- level count, sizes, pixels and derivatives per (image, winSize, maxLevel)
    grid = {}
    for img_name, img in images.items():
        for (w, h), max_level in [
            ((21, 21), 3),
            ((21, 21), 0),
            ((21, 21), 8),
            ((3, 3), 8),
            ((5, 5), 2),
            ((20, 20), 5),
            ((15, 9), 4),
        ]:
            for with_derivatives in (True, False):
                key = f"{img_name}/win{w}x{h}/max{max_level}/{'deriv' if with_derivatives else 'nod'}"
                grid[key] = build(img, (w, h), max_level, with_derivatives)
    results["grid"] = grid

    # --- the padding borders do not change the stored pixels or derivatives
    img = images["reusable_33x49"]
    base = build(img, (5, 5), 3, True)
    borders = {}
    for pb_name, pb in BORDERS.items():
        for db_name, db in [("constant", cv2.BORDER_CONSTANT), ("reflect101", cv2.BORDER_REFLECT_101)]:
            for try_reuse in (True, False):
                r = build(img, (5, 5), 3, True, pb, db, try_reuse)
                same = r == base
                borders[f"pyr_{pb_name}/deriv_{db_name}/reuse_{int(try_reuse)}"] = {
                    "same_as_default": bool(same),
                    "returned_level": r["returned_level"],
                }
    results["borders"] = borders

    # --- level 0 is the input (tryReuseInputImage only chooses whether it is copied)
    results["level0_is_input"] = {
        name: build(img, (21, 21), 3, False)["levels"][0]["fnv1a_pixels"] == pixel_digest(img)
        for name, img in images.items()
    }

    # --- errors and accepted inputs (D2)
    img = images["reusable_33x49"]
    results["inputs_d2"] = {
        "win_2x5": attempt(lambda: cv2.buildOpticalFlowPyramid(img, (2, 5), 3)[0]),
        "win_5x2": attempt(lambda: cv2.buildOpticalFlowPyramid(img, (5, 2), 3)[0]),
        "max_level_-1": attempt(lambda: cv2.buildOpticalFlowPyramid(img, (5, 5), -1)[0]),
        "f32_input": attempt(lambda: cv2.buildOpticalFlowPyramid(img.astype(np.float32), (5, 5), 3)[0]),
        "u16_input": attempt(lambda: cv2.buildOpticalFlowPyramid(img.astype(np.uint16), (5, 5), 3)[0]),
        "u8c3_input": attempt(
            lambda: [
                (list(m.shape), str(m.dtype))
                for m in cv2.buildOpticalFlowPyramid(cv2.merge([img, img, img]), (5, 5), 2)[1]
            ]
        ),
        "border_transparent": attempt(
            lambda: cv2.buildOpticalFlowPyramid(img, (5, 5), 2, None, True, cv2.BORDER_TRANSPARENT)[0]
        ),
        "border_isolated_flag": attempt(
            lambda: cv2.buildOpticalFlowPyramid(
                img, (5, 5), 2, None, True, cv2.BORDER_REFLECT_101 | cv2.BORDER_ISOLATED
            )[0]
        ),
    }

    dump("video_pyramid.py", results)


if __name__ == "__main__":
    main()

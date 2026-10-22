"""OpenCV 4.10.0 reference values for the Lucas-Kanade rows (docs/parity/4.10.0/video.md).

Rows: calc_optical_flow_pyramid_lk, calc_optical_flow_pyramid_lk_with_pyramids,
OPTFLOW_USE_INITIAL_FLOW, OPTFLOW_LK_GET_MIN_EIGENVALS, and the seeded rows #130, #131,
#138, #142, #144, #145, #149, #163 and #178. `cv2.calcOpticalFlowPyrLK` is run on seeded
images (the #163 cell texture, a rectangle texture with a quarter-pixel shift, and the
`(17x + 13y + 3xy) mod 251` fixture of the in-tree reference tests); every point's
`[x, y, status, err]` is recorded so that tests/parity_video.rs can pin it.

The Python binding does not accept a list of Mats as `prevImg`/`nextImg` (it expects one
Mat or UMat), so the prebuilt-pyramid entry point is not measurable here; its padding
semantics are measured through explicitly padded images (`asymmetric_borders`).

Run from the repository root:
    cpp_ref/.venv/Scripts/python scripts/opencv_ref/video_lk.py   # Windows
    cpp_ref/.venv/bin/python scripts/opencv_ref/video_lk.py       # Linux/macOS
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
from _video_common import (  # noqa: E402
    cell_texture,
    flow_digest,
    grid_points,
    pixel_digest,
    reusable_texture,
    shift_image,
    shift_quarter,
    textured_u8,
)

COUNT = cv2.TERM_CRITERIA_COUNT  # 1
EPS = cv2.TERM_CRITERIA_EPS  # 2
BOTH = COUNT | EPS
DEFAULT_CRITERIA = (BOTH, 30, 0.01)
USE_INITIAL_FLOW = cv2.OPTFLOW_USE_INITIAL_FLOW
GET_MIN_EIGENVALS = cv2.OPTFLOW_LK_GET_MIN_EIGENVALS


def attempt(fn):
    try:
        return tolist(fn())
    except cv2.error as e:  # noqa: PERF203
        return f"cv2.error: {e.code} {e.err}".strip()


def lk(prev, nxt, pts, init=None, win=(21, 21), levels=3, criteria=DEFAULT_CRITERIA,
       flags=0, min_eig=1e-4):
    """`cv2.calcOpticalFlowPyrLK` with explicit arguments; `init` is `nextPts` on input."""
    next_pts = None if init is None else np.array(init, np.float32).reshape(-1, 1, 2).copy()
    out, status, err = cv2.calcOpticalFlowPyrLK(
        prev, nxt, pts, next_pts, winSize=win, maxLevel=levels, criteria=criteria,
        flags=flags, minEigThreshold=min_eig,
    )
    if out is None:  # npoints == 0: every output is released
        return {"count": 0, "tracked": 0, "outputs": [out, status, err]}
    return flow_digest(out, status, err)


def points(pts) -> list:
    return [[float(x), float(y)] for x, y in pts.reshape(-1, 2)]


def main() -> None:
    results = {}

    # ------------------------------------------------------------------
    # The #163 setup: 320x240 cell texture, 150-point grid, 21x21, 4 levels
    # ------------------------------------------------------------------
    img = cell_texture(320, 240)
    pts = grid_points(40, 280, 40, 200, 16)
    assert pts.shape[0] == 150
    subset = pts[::5].copy()  # 30 points for the parameter variations
    results["inputs"] = {
        "cell_320x240": {"shape": list(img.shape), "fnv1a_pixels": pixel_digest(img)},
        "grid_150": points(pts),
        "grid_30": points(subset),
    }
    grid = {}
    for dx, dy in [(0, 0), (10, 0), (0, 10), (10, 10)]:
        nxt = shift_image(img, dx, dy)
        grid[f"shift_{dx}_{dy}/initial_flow"] = lk(img, nxt, pts, init=pts, flags=USE_INITIAL_FLOW)
    nxt = shift_image(img, 10, 0)
    grid["shift_10_0/no_flags"] = lk(img, nxt, pts)
    results["inputs"]["cell_320x240_shift_10_0"] = {"fnv1a_pixels": pixel_digest(nxt)}
    results["grid163"] = grid

    # --- parameter variations on the 30-point subset, shift (10, 0)
    var = {}
    var["default"] = lk(img, nxt, subset)
    var["min_eig_output"] = lk(img, nxt, subset, flags=GET_MIN_EIGENVALS)
    var["win_20x20"] = lk(img, nxt, subset, win=(20, 20))
    var["win_15x9"] = lk(img, nxt, subset, win=(15, 9))
    var["win_7x7"] = lk(img, nxt, subset, win=(7, 7))
    var["win_3x3"] = lk(img, nxt, subset, win=(3, 3))
    var["max_level_0"] = lk(img, nxt, subset, levels=0)
    var["max_level_1"] = lk(img, nxt, subset, levels=1)
    var["max_level_5"] = lk(img, nxt, subset, levels=5)
    var["min_eig_1e-2"] = lk(img, nxt, subset, min_eig=1e-2)
    var["min_eig_0.5"] = lk(img, nxt, subset, min_eig=0.5)
    var["min_eig_0.5_output"] = lk(img, nxt, subset, min_eig=0.5, flags=GET_MIN_EIGENVALS)
    var["count_5"] = lk(img, nxt, subset, criteria=(COUNT, 5, 0.0))
    var["count_0"] = lk(img, nxt, subset, criteria=(COUNT, 0, 0.0))
    var["eps_only_0.001"] = lk(img, nxt, subset, criteria=(EPS, 0, 0.001))
    var["both_100_0"] = lk(img, nxt, subset, criteria=(BOTH, 100, 0.0))
    var["both_1_0.03"] = lk(img, nxt, subset, criteria=(BOTH, 1, 0.03))
    good = subset.copy()
    good[:, 0, 0] += 8.0
    bad = subset.copy()
    bad[:, 0, 0] -= 8.0
    bad[:, 0, 1] += 3.0
    var["init_good/initial_flow"] = lk(img, nxt, subset, init=good, flags=USE_INITIAL_FLOW)
    var["init_bad/initial_flow"] = lk(img, nxt, subset, init=bad, flags=USE_INITIAL_FLOW)
    var["init_good/ignored_without_flag"] = lk(img, nxt, subset, init=good)
    var["init_good/initial_flow/count_0"] = lk(
        img, nxt, subset, init=good, flags=USE_INITIAL_FLOW, criteria=(COUNT, 0, 0.0)
    )
    results["variations"] = var

    # ------------------------------------------------------------------
    # Sub-pixel motion: rectangle texture shifted by (2.25, -0.75) px
    # ------------------------------------------------------------------
    tex = textured_u8(96, 128, 11)
    tex_next = shift_quarter(tex, 9, -3)
    spts = grid_points(16, 112, 16, 80, 12)
    results["inputs"]["textured_96x128_s11"] = {
        "shape": list(tex.shape),
        "fnv1a_pixels": pixel_digest(tex),
        "fnv1a_shift_2.25_-0.75": pixel_digest(tex_next),
    }
    results["inputs"]["grid_sub"] = points(spts)
    sub = {}
    sub["default"] = lk(tex, tex_next, spts)
    sub["min_eig_output"] = lk(tex, tex_next, spts, flags=GET_MIN_EIGENVALS)
    sub["win_10x10"] = lk(tex, tex_next, spts, win=(10, 10))
    sub["win_7x7"] = lk(tex, tex_next, spts, win=(7, 7))
    sub["max_level_2"] = lk(tex, tex_next, spts, levels=2)
    sub["max_level_0"] = lk(tex, tex_next, spts, levels=0)
    sub["eps_0.001_count_100"] = lk(tex, tex_next, spts, criteria=(BOTH, 100, 0.001))
    results["subpixel"] = sub

    # ------------------------------------------------------------------
    # Edge and outside points on the 33x49 fixture (window 5x5)
    # ------------------------------------------------------------------
    prev = reusable_texture(33, 49, 0)
    nxt1 = reusable_texture(33, 49, 1)
    epts = np.array(
        [
            [0.0, 0.0], [48.0, 32.0], [-0.25, 0.5], [48.75, 32.5], [-20.0, -20.0],
            [-4.5, 5.0], [5.0, -4.5], [52.9, 10.0], [10.0, 36.9], [2.25, 2.5],
            [24.5, 16.5], [16.0, 16.0], [45.0, 30.0], [48.0, 0.0], [0.0, 32.0],
        ],
        np.float32,
    ).reshape(-1, 1, 2)
    eguess = epts.copy()
    eguess[:, 0, 0] += 1.0
    results["inputs"]["reusable_33x49"] = {
        "fnv1a_pixels": pixel_digest(prev),
        "fnv1a_shift1": pixel_digest(nxt1),
    }
    results["inputs"]["edge_points"] = points(epts)
    edges = {}
    for levels in (0, 2):
        for flag_name, flags in [("mae", 0), ("min_eig", GET_MIN_EIGENVALS)]:
            edges[f"max{levels}/{flag_name}"] = lk(
                prev, nxt1, epts, win=(5, 5), levels=levels, criteria=(BOTH, 20, 0.03), flags=flags
            )
            edges[f"max{levels}/{flag_name}/initial_flow"] = lk(
                prev, nxt1, epts, init=eguess, win=(5, 5), levels=levels,
                criteria=(BOTH, 20, 0.03), flags=flags | USE_INITIAL_FLOW,
            )
    results["edges"] = edges

    # ------------------------------------------------------------------
    # Final bounds gate (#178): zero iterations from a guess near the padding limit
    # ------------------------------------------------------------------
    small = reusable_texture(9, 9, 0)
    results["inputs"]["reusable_9x9"] = {"pixels": tolist(small)}
    final = {}
    for gx, gy in [
        (100.0, 4.0), (4.0, 100.0), (-3.0, 4.0), (-3.01, 4.0), (4.0, -3.0), (4.0, -3.01),
        (10.99, 4.0), (11.0, 4.0), (4.0, 10.99), (4.0, 11.0),
    ]:
        for flag_name, flags in [("mae", 0), ("min_eig", GET_MIN_EIGENVALS)]:
            final[f"guess_{gx}_{gy}/{flag_name}"] = lk(
                small, small, np.array([[4.0, 4.0]], np.float32).reshape(1, 1, 2),
                init=[[gx, gy]], win=(5, 5), levels=0, criteria=(COUNT, 0, 0.0),
                flags=flags | USE_INITIAL_FLOW,
            )
    results["final_bounds"] = final

    # ------------------------------------------------------------------
    # pyrBorder semantics through explicitly padded images (maxLevel 0)
    # ------------------------------------------------------------------
    # copyMakeBorder pads the 9x9 image by 8 with the border under test, so raw LK
    # at level 0 samples that padding exactly where calcOpticalFlowPyrLK samples the
    # pyramid's own padding (the window never reaches beyond 8 px).
    pad = 8
    borders = {
        "reflect101": cv2.BORDER_REFLECT_101,
        "constant": cv2.BORDER_CONSTANT,
        "replicate": cv2.BORDER_REPLICATE,
        "wrap": cv2.BORDER_WRAP,
    }
    asym = {}
    mae_pts = np.array([[0.0, 0.0], [8.0, 8.0], [0.25, 0.5]], np.float32).reshape(-1, 1, 2)
    step_pts = np.array([[4.0, 4.0], [4.0, 4.0]], np.float32).reshape(-1, 1, 2)
    step_guess = np.array([[0.25, 0.5], [8.25, 8.5]], np.float32).reshape(-1, 1, 2)
    for pn, pb in borders.items():
        for nn, nb in borders.items():
            if pn != "reflect101" and nn != "reflect101":
                continue
            p = cv2.copyMakeBorder(small, pad, pad, pad, pad, pb)
            n = cv2.copyMakeBorder(small, pad, pad, pad, pad, nb)
            mae = lk(p, n, mae_pts + pad, init=mae_pts + pad, win=(5, 5), levels=0,
                     criteria=(COUNT, 0, 0.0), flags=USE_INITIAL_FLOW)
            step = lk(p, n, step_pts + pad, init=step_guess + pad, win=(5, 5), levels=0,
                      criteria=(COUNT, 1, 0.0), flags=USE_INITIAL_FLOW)
            for r in (mae, step):
                for row in r["rows"]:
                    row[0] -= pad
                    row[1] -= pad
            asym[f"prev_{pn}/next_{nn}"] = {"mae_count0": mae, "step_count1": step}
    results["asymmetric_borders"] = asym

    # ------------------------------------------------------------------
    # Termination criteria normalisation (33x49 fixture, window 5x5, level 0)
    # ------------------------------------------------------------------
    term = {}
    for kind_name, kind, count, eps, px, py in [
        ("count", COUNT, 30, 99.0, 2.25, 2.5),
        ("both", BOTH, 30, 0.0, 2.25, 2.5),
        ("both", BOTH, 30, -0.5, 2.25, 2.5),
        ("count", COUNT, -3, 0.0, 2.25, 2.5),
        ("count", COUNT, 0, 0.0, 2.25, 2.5),
        ("eps", EPS, 1, 0.03, 2.25, 2.5),
        ("both", BOTH, 30, 0.03, 2.25, 2.5),
        ("both", BOTH, 30, 10.0, 2.25, 2.5),
        ("both", BOTH, 30, 20.0, 2.25, 2.5),
        ("count", COUNT, 100, 0.0, 8.125, 14.375),
        ("count", COUNT, 120, 0.0, 8.125, 14.375),
        ("eps", EPS, 1, 0.03, 8.125, 14.375),
        ("both", BOTH, 30, 0.03, 8.125, 14.375),
    ]:
        key = f"{kind_name}/{count}/{eps}/{px}_{py}"
        term[key] = lk(prev, nxt1, np.array([[px, py]], np.float32).reshape(1, 1, 2),
                       win=(5, 5), levels=0, criteria=(kind, count, eps))
    results["termination"] = term

    # ------------------------------------------------------------------
    # Accepted and rejected inputs (D2)
    # ------------------------------------------------------------------
    p2 = np.array([[16.0, 16.0], [30.0, 20.0]], np.float32).reshape(-1, 1, 2)
    d2 = {}
    d2["win_2x5"] = attempt(lambda: lk(prev, nxt1, p2, win=(2, 5)))
    d2["max_level_-1"] = attempt(lambda: lk(prev, nxt1, p2, levels=-1))
    d2["size_mismatch"] = attempt(lambda: lk(prev, nxt1[:, :40], p2))
    d2["f32_images"] = attempt(lambda: lk(prev.astype(np.float32), nxt1.astype(np.float32), p2))
    d2["u16_images"] = attempt(lambda: lk(prev.astype(np.uint16), nxt1.astype(np.uint16), p2))
    d2["u8c3_images"] = attempt(
        lambda: lk(cv2.merge([prev, prev, prev]), cv2.merge([nxt1, nxt1, nxt1]), p2, win=(5, 5))
    )
    d2["u8c1_same_points"] = lk(prev, nxt1, p2, win=(5, 5))
    d2["initial_flow_length_mismatch"] = attempt(
        lambda: lk(prev, nxt1, p2, init=[[16.0, 16.0]], flags=USE_INITIAL_FLOW)
    )
    d2["empty_points"] = attempt(lambda: lk(prev, nxt1, np.zeros((0, 1, 2), np.float32)))
    d2["min_eig_negative"] = lk(prev, nxt1, p2, win=(5, 5), min_eig=-1.0)
    results["inputs_d2"] = d2

    dump("video_lk.py", results)


if __name__ == "__main__":
    main()

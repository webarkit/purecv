"""OpenCV 4.10.0 reference values for the imgproc feature rows (docs/parity/4.10.0/imgproc.md).

Rows: canny, corner_harris, corner_min_eigen_val, corner_eigen_vals_and_vecs,
pre_corner_detect, good_features_to_track, corner_sub_pix, hough_lines, hough_lines_p,
hough_circles.

Every input is defined here by a formula, so tests/parity_imgproc.rs can rebuild it.

Run from the repository root:
    cpp_ref/.venv/Scripts/python scripts/opencv_ref/imgproc_feature.py   # Windows
    cpp_ref/.venv/bin/python scripts/opencv_ref/imgproc_feature.py       # Linux/macOS
"""

import cv2

assert cv2.__version__ == "4.10.0"
cv2.ipp.setUseIPP(False)  # measure the generic CPU path

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _core_common import dump, tolist  # noqa: E402


def attempt(fn):
    try:
        return tolist(fn())
    except cv2.error as e:  # noqa: PERF203
        return f"cv2.error: {e.code} {e.err}".strip()


def pattern_u8(rows, cols):
    """Deterministic test image: ((7*y + 13*x) * 37) % 256."""
    y, x = np.mgrid[0:rows, 0:cols]
    return (((7 * y + 13 * x) * 37) % 256).astype(np.uint8)


def square_u8(size=16, lo=20, hi=220, x0=4, x1=12):
    """A bright square (rows/cols x0..x1 exclusive) on a dark background."""
    img = np.full((size, size), lo, np.uint8)
    img[x0:x1, x0:x1] = hi
    return img


def main() -> None:
    results = {}

    img8 = pattern_u8(8, 8)
    sq = square_u8()
    sq_f32 = sq.astype(np.float32)
    # a smooth ramp plus the square so that gradients vary
    ramp = (np.mgrid[0:16, 0:16][1] * 3).astype(np.uint8)
    mixed = cv2.add(square_u8(16, 20, 160), ramp)

    # --- Canny
    cn = {"src_square": tolist(sq), "src_mixed": tolist(mixed), "src_pattern": tolist(img8)}
    cn["square_t50_t150"] = tolist(cv2.Canny(sq, 50, 150))
    cn["square_t50_t150_l2"] = tolist(cv2.Canny(sq, 50, 150, L2gradient=True))
    cn["square_t100_t800_l1"] = tolist(cv2.Canny(sq, 100, 800))
    cn["square_t799_t801_l1_tie"] = tolist(cv2.Canny(sq, 799, 801))
    cn["square_t800_t800_l1_tie"] = tolist(cv2.Canny(sq, 800, 800))
    cn["mixed_t50_t150"] = tolist(cv2.Canny(mixed, 50, 150))
    cn["mixed_t50_t150_l2"] = tolist(cv2.Canny(mixed, 50, 150, L2gradient=True))
    cn["mixed_t20_t60_k5"] = tolist(cv2.Canny(mixed, 20, 60, apertureSize=5))
    cn["pattern_t100_t200"] = tolist(cv2.Canny(img8, 100, 200))
    cn["pattern_t100_t200_l2"] = tolist(cv2.Canny(img8, 100, 200, L2gradient=True))
    cn["square_swapped_thresholds"] = tolist(cv2.Canny(sq, 150, 50))
    cn["square_t50_t150_k7"] = tolist(cv2.Canny(sq, 50, 150, apertureSize=7))
    cn["f32_rejected"] = attempt(lambda: cv2.Canny(sq_f32, 50, 150))
    cn["k4_rejected"] = attempt(lambda: cv2.Canny(sq, 50, 150, apertureSize=4))
    cn["sobel_dx_16s_square"] = tolist(cv2.Sobel(sq, cv2.CV_16S, 1, 0, ksize=3))
    results["canny"] = cn

    # --- corner responses (f32 input, block 3, ksize 3)
    co = {"src_square_f32": tolist(sq_f32), "src_mixed_f32": tolist(mixed.astype(np.float32))}
    co["harris_f32_b3_k3_0.04"] = tolist(cv2.cornerHarris(sq_f32, 3, 3, 0.04))
    co["harris_u8_b3_k3_0.04"] = tolist(cv2.cornerHarris(sq, 3, 3, 0.04))
    co["harris_f32_b5_k3_0.04"] = tolist(cv2.cornerHarris(sq_f32, 5, 3, 0.04))
    co["harris_f32_b3_scharr_0.04"] = tolist(cv2.cornerHarris(sq_f32, 3, -1, 0.04))
    co["harris_f32_b3_k5_0.06"] = tolist(cv2.cornerHarris(sq_f32, 3, 5, 0.06))
    co["harris_f32_b3_k3_replicate"] = tolist(cv2.cornerHarris(sq_f32, 3, 3, 0.04, borderType=cv2.BORDER_REPLICATE))
    co["harris_f32_b2_even"] = attempt(lambda: cv2.cornerHarris(sq_f32, 2, 3, 0.04))
    co["min_eigen_f32_b3_k3"] = tolist(cv2.cornerMinEigenVal(sq_f32, 3, 3))
    co["min_eigen_u8_b3_k3"] = tolist(cv2.cornerMinEigenVal(sq, 3, 3))
    co["min_eigen_f32_b5_k3"] = tolist(cv2.cornerMinEigenVal(sq_f32, 5, 3))
    co["eigen_vals_vecs_f32_b3_k3"] = tolist(cv2.cornerEigenValsAndVecs(sq_f32, 3, 3))
    co["pre_corner_f32_k3"] = tolist(cv2.preCornerDetect(sq_f32, 3))
    co["pre_corner_u8_k3"] = tolist(cv2.preCornerDetect(sq, 3))
    co["pre_corner_f32_k5"] = tolist(cv2.preCornerDetect(sq_f32, 5))
    co["harris_f64_rejected"] = attempt(lambda: cv2.cornerHarris(sq_f32.astype(np.float64), 3, 3, 0.04))
    results["corners"] = co

    # --- goodFeaturesToTrack
    two = np.full((24, 24), 30, np.uint8)
    two[3:10, 3:10] = 200
    two[13:21, 12:20] = 220
    two_f32 = two.astype(np.float32)
    gf = {"src_two_squares": tolist(two)}
    gf["f32_max10_q0.01_d3"] = tolist(cv2.goodFeaturesToTrack(two_f32, 10, 0.01, 3))
    gf["f32_max10_q0.01_d3_harris"] = tolist(cv2.goodFeaturesToTrack(two_f32, 10, 0.01, 3, useHarrisDetector=True, k=0.04))
    gf["f32_max3_q0.01_d3"] = tolist(cv2.goodFeaturesToTrack(two_f32, 3, 0.01, 3))
    gf["f32_max0_q0.01_d3"] = tolist(cv2.goodFeaturesToTrack(two_f32, 0, 0.01, 3))
    gf["f32_max10_q0.01_d0"] = tolist(cv2.goodFeaturesToTrack(two_f32, 10, 0.01, 0))
    gf["f32_max10_q0.5_d3"] = tolist(cv2.goodFeaturesToTrack(two_f32, 10, 0.5, 3))
    gf["f32_max10_q0.01_d3_b5"] = tolist(cv2.goodFeaturesToTrack(two_f32, 10, 0.01, 3, blockSize=5))
    gf["u8_max10_q0.01_d3"] = tolist(cv2.goodFeaturesToTrack(two, 10, 0.01, 3))
    gf["f32_q0_rejected"] = attempt(lambda: cv2.goodFeaturesToTrack(two_f32, 10, 0.0, 3))
    gf["f32_q1.5_accepted_empty"] = attempt(lambda: cv2.goodFeaturesToTrack(two_f32, 10, 1.5, 3))
    gf["f32_flat_image"] = attempt(lambda: cv2.goodFeaturesToTrack(np.full((16, 16), 5.0, np.float32), 10, 0.01, 3))
    results["good_features_to_track"] = gf

    # --- cornerSubPix (corners near the square corners of `sq`)
    corners = np.array([[[3.0, 3.0]], [[11.0, 4.0]], [[4.0, 12.0]], [[12.0, 12.0]]], np.float32)
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.001)
    cs = {"initial": tolist(corners.reshape(-1, 2))}
    refined = cv2.cornerSubPix(sq_f32, corners.copy(), (2, 2), (-1, -1), crit)
    cs["f32_win2_zero-1"] = tolist(refined.reshape(-1, 2))
    refined = cv2.cornerSubPix(sq, corners.copy(), (2, 2), (-1, -1), crit)
    cs["u8_win2_zero-1"] = tolist(refined.reshape(-1, 2))
    refined = cv2.cornerSubPix(sq_f32, corners.copy(), (3, 3), (1, 1), crit)
    cs["f32_win3_zero1"] = tolist(refined.reshape(-1, 2))
    refined = cv2.cornerSubPix(sq_f32, corners.copy(), (2, 2), (-1, -1), (cv2.TERM_CRITERIA_MAX_ITER, 1, 0.0))
    cs["f32_win2_one_iter"] = tolist(refined.reshape(-1, 2))
    cs["win_too_large_rejected"] = attempt(lambda: cv2.cornerSubPix(sq_f32, corners.copy(), (6, 6), (-1, -1), crit))
    results["corner_sub_pix"] = cs

    # --- HoughLines (a horizontal and a vertical line, plus a diagonal)
    lines_img = np.zeros((20, 20), np.uint8)
    lines_img[5, :] = 255
    lines_img[:, 14] = 255
    for i in range(20):
        lines_img[i, i] = 255
    hl = {"src_lines": tolist(lines_img)}
    hl["rho1_theta_pi180_t15"] = tolist(cv2.HoughLines(lines_img, 1, np.pi / 180, 15))
    hl["rho1_theta_pi180_t15_with_votes"] = tolist(cv2.HoughLinesWithAccumulator(lines_img, 1, np.pi / 180, 15))
    hl["rho2_theta_pi90_t10"] = tolist(cv2.HoughLines(lines_img, 2, np.pi / 90, 10))
    hl["rho1_theta_pi180_t15_theta_0.5_2.0"] = tolist(cv2.HoughLines(lines_img, 1, np.pi / 180, 15, min_theta=0.5, max_theta=2.0))
    hl["rho1_theta_pi4_t10"] = tolist(cv2.HoughLines(lines_img, 1, np.pi / 4, 10))
    hl["rho1_theta_pi180_t100_none"] = attempt(lambda: cv2.HoughLines(lines_img, 1, np.pi / 180, 100))
    hl["min_theta_gt_max_rejected"] = attempt(lambda: cv2.HoughLines(lines_img, 1, np.pi / 180, 15, min_theta=2.0, max_theta=1.0))
    hl["numangle_pi_over_pi180"] = int(np.floor(np.pi / (np.pi / 180)) + 1)
    results["hough_lines"] = hl

    # --- HoughLinesP (deterministic in OpenCV: RNG seeded with (uint64)-1)
    hp = {}
    hp["rho1_theta_pi180_t10_len5_gap2"] = tolist(cv2.HoughLinesP(lines_img, 1, np.pi / 180, 10, minLineLength=5, maxLineGap=2))
    hp["rho1_theta_pi180_t10_len5_gap2_rerun"] = tolist(cv2.HoughLinesP(lines_img, 1, np.pi / 180, 10, minLineLength=5, maxLineGap=2))
    hp["rho1_theta_pi180_t5_len0_gap0"] = tolist(cv2.HoughLinesP(lines_img, 1, np.pi / 180, 5))
    results["hough_lines_p"] = hp

    # --- HoughCircles (a drawn circle)
    # a one-pixel ring of radius 9 around (20, 19): |dist - 9| < 0.5 (a formula, not cv2.circle,
    # so that the Rust tests can rebuild it)
    cy, cx = np.mgrid[0:40, 0:40]
    circ = (np.abs(np.sqrt((cx - 20.0) ** 2 + (cy - 19.0) ** 2) - 9.0) < 0.5).astype(np.uint8) * 255
    hc = {"src_circle": tolist(circ)}
    hc["gradient_dp1_md10_p100_p10_r5_15"] = tolist(cv2.HoughCircles(circ, cv2.HOUGH_GRADIENT, 1, 10, param1=100, param2=10, minRadius=5, maxRadius=15))
    hc["gradient_dp1_md10_p100_p10_r0_0"] = tolist(cv2.HoughCircles(circ, cv2.HOUGH_GRADIENT, 1, 10, param1=100, param2=10))
    hc["gradient_dp2_md10_p100_p10_r5_15"] = tolist(cv2.HoughCircles(circ, cv2.HOUGH_GRADIENT, 2, 10, param1=100, param2=10, minRadius=5, maxRadius=15))
    hc["gradient_alt"] = tolist(cv2.HoughCircles(circ, cv2.HOUGH_GRADIENT_ALT, 1.5, 10, param1=100, param2=0.5, minRadius=5, maxRadius=15))
    hc["dp0_rejected"] = attempt(lambda: cv2.HoughCircles(circ, cv2.HOUGH_GRADIENT, 0, 10, param1=100, param2=10))
    results["hough_circles"] = hc

    dump("imgproc_feature.py", results)


if __name__ == "__main__":
    main()

"""OpenCV 4.10.0 reference values for the imgproc filter, derivative, morphology and
threshold rows (docs/parity/4.10.0/imgproc.md).

Rows: get_gaussian_kernel, gaussian_blur, blur / box_filter, median_blur, bilateral_filter,
get_sobel_kernels, sobel, scharr, laplacian, get_structuring_element, erode, dilate,
morphology_ex, threshold.

Every input is defined here by a formula, so tests/parity_imgproc.rs can rebuild it.

Run from the repository root:
    cpp_ref/.venv/Scripts/python scripts/opencv_ref/imgproc_filter.py   # Windows
    cpp_ref/.venv/bin/python scripts/opencv_ref/imgproc_filter.py       # Linux/macOS
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


def attempt(fn):
    try:
        return tolist(fn())
    except cv2.error as e:  # noqa: PERF203
        return f"cv2.error: {e.code} {e.err}".strip()


def pattern_u8(rows, cols, channels=1):
    """Deterministic test image: ((7*y + 13*x + 29*c) * 37) % 256."""
    y, x, c = np.meshgrid(np.arange(rows), np.arange(cols), np.arange(channels), indexing="ij")
    img = ((7 * y + 13 * x + 29 * c) * 37) % 256
    img = img.astype(np.uint8)
    return img if channels > 1 else img[:, :, 0].copy()


def main() -> None:
    results = {}

    # --- getGaussianKernel (sigma <= 0 uses the fixed tables for n = 3, 5, 7, 9)
    results["get_gaussian_kernel"] = {
        f"n{n}_sigma0": tolist(cv2.getGaussianKernel(n, 0).ravel()) for n in (1, 3, 5, 7, 9, 11)
    }
    results["get_gaussian_kernel"]["n3_sigma1.5"] = tolist(cv2.getGaussianKernel(3, 1.5).ravel())
    results["get_gaussian_kernel"]["n5_sigma0.8"] = tolist(cv2.getGaussianKernel(5, 0.8).ravel())
    results["get_gaussian_kernel"]["n7_sigma2"] = tolist(cv2.getGaussianKernel(7, 2.0).ravel())
    results["get_gaussian_kernel"]["n4_sigma0_even"] = attempt(lambda: cv2.getGaussianKernel(4, 0).ravel())

    # --- GaussianBlur (u8 uses the ufixedpoint16 bit-exact path; f32 the float path)
    img8 = pattern_u8(8, 8)
    img8c3 = pattern_u8(6, 7, 3)
    img32 = img8.astype(np.float32)
    g = {"src_u8": tolist(img8), "src_u8c3": tolist(img8c3)}
    for ks in (3, 5, 7):
        g[f"u8_k{ks}_sigma0"] = tolist(cv2.GaussianBlur(img8, (ks, ks), 0))
        g[f"f32_k{ks}_sigma0"] = tolist(cv2.GaussianBlur(img32, (ks, ks), 0))
    g["u8_k5_sigma1.5"] = tolist(cv2.GaussianBlur(img8, (5, 5), 1.5))
    g["f32_k5_sigma1.5"] = tolist(cv2.GaussianBlur(img32, (5, 5), 1.5))
    g["u8_k3x5_sigma0"] = tolist(cv2.GaussianBlur(img8, (3, 5), 0))
    g["u8_k3x5_sigma1.5_sigma2_0"] = tolist(cv2.GaussianBlur(img8, (3, 5), 1.5))
    g["f32_k3x5_sigma1.5_sigma2_0"] = tolist(cv2.GaussianBlur(img32, (3, 5), 1.5))
    g["f32_k5x3_sigma1.5_sigma2_0"] = tolist(cv2.GaussianBlur(img32, (5, 3), 1.5))
    g["u8c3_k3_sigma0"] = tolist(cv2.GaussianBlur(img8c3, (3, 3), 0))
    g["u8_k3_sigma0_replicate"] = tolist(cv2.GaussianBlur(img8, (3, 3), 0, borderType=cv2.BORDER_REPLICATE))
    g["u8_k3_sigma0_constant"] = tolist(cv2.GaussianBlur(img8, (3, 3), 0, borderType=cv2.BORDER_CONSTANT))
    g["u8_k0_sigma1_auto_ksize"] = tolist(cv2.GaussianBlur(img8, (0, 0), 1.0))
    g["u8_k4_even"] = attempt(lambda: cv2.GaussianBlur(img8, (4, 4), 0))
    results["gaussian_blur"] = g

    # --- blur / boxFilter
    b = {"src_u8": tolist(img8)}
    b["blur_u8_k3"] = tolist(cv2.blur(img8, (3, 3)))
    b["blur_u8_k5"] = tolist(cv2.blur(img8, (5, 5)))
    b["blur_u8_k3x2"] = tolist(cv2.blur(img8, (3, 2)))
    b["blur_u8_k3_anchor00"] = tolist(cv2.blur(img8, (3, 3), anchor=(0, 0)))
    b["blur_f32_k3"] = tolist(cv2.blur(img32, (3, 3)))
    b["box_u8_k3_unnormalized"] = tolist(cv2.boxFilter(img8, -1, (3, 3), normalize=False))
    b["box_u8_k3_unnormalized_ddepth_32f"] = tolist(cv2.boxFilter(img8, cv2.CV_32F, (3, 3), normalize=False))
    b["blur_u8_k3_replicate"] = tolist(cv2.blur(img8, (3, 3), borderType=cv2.BORDER_REPLICATE))
    b["blur_u8_k3_constant"] = tolist(cv2.blur(img8, (3, 3), borderType=cv2.BORDER_CONSTANT))
    b["blur_u8c3_k3"] = tolist(cv2.blur(img8c3, (3, 3)))
    results["blur_box_filter"] = b

    # --- medianBlur (BORDER_REPLICATE inside)
    m = {"src_u8": tolist(img8)}
    m["u8_k3"] = tolist(cv2.medianBlur(img8, 3))
    m["u8_k5"] = tolist(cv2.medianBlur(img8, 5))
    m["u8c3_k3"] = tolist(cv2.medianBlur(img8c3, 3))
    m["f32_k3"] = tolist(cv2.medianBlur(img32, 3))
    m["f32_k7"] = attempt(lambda: cv2.medianBlur(img32, 7))
    m["u8_k1"] = tolist(cv2.medianBlur(img8, 1))
    m["u8_k4_even"] = attempt(lambda: cv2.medianBlur(img8, 4))
    results["median_blur"] = m

    # --- bilateralFilter (circular neighbourhood, L1 colour distance on 3 channels)
    bf = {"src_u8": tolist(img8), "src_u8c3": tolist(img8c3)}
    bf["u8_d5_sc50_ss50"] = tolist(cv2.bilateralFilter(img8, 5, 50, 50))
    bf["u8_d3_sc30_ss10"] = tolist(cv2.bilateralFilter(img8, 3, 30, 10))
    bf["u8_d0_ss2"] = tolist(cv2.bilateralFilter(img8, 0, 50, 2))
    bf["u8c3_d5_sc50_ss50"] = tolist(cv2.bilateralFilter(img8c3, 5, 50, 50))
    bf["f32_d5_sc50_ss50"] = tolist(cv2.bilateralFilter(img32, 5, 50, 50))
    bf["u8c2"] = attempt(lambda: cv2.bilateralFilter(pattern_u8(4, 4, 2), 3, 10, 10))
    results["bilateral_filter"] = bf

    # --- getDerivKernels
    dk = {}
    for ks in (1, 3, 5, 7):
        for dx, dy in ((1, 0), (0, 1), (2, 0), (1, 1)):
            dk[f"k{ks}_dx{dx}_dy{dy}"] = attempt(
                lambda ks=ks, dx=dx, dy=dy: [a.ravel() for a in cv2.getDerivKernels(dx, dy, ks)]
            )
    dk["scharr_dx1_dy0"] = tolist([a.ravel() for a in cv2.getDerivKernels(1, 0, -1)])
    dk["scharr_dx0_dy1"] = tolist([a.ravel() for a in cv2.getDerivKernels(0, 1, -1)])
    dk["k3_dx1_dy0_normalized"] = tolist([a.ravel() for a in cv2.getDerivKernels(1, 0, 3, normalize=True)])
    dk["k5_dx2_dy0_normalized"] = tolist([a.ravel() for a in cv2.getDerivKernels(2, 0, 5, normalize=True)])
    dk["scharr_dx1_dy1"] = attempt(lambda: cv2.getDerivKernels(1, 1, -1))
    dk["k3_dx0_dy0"] = attempt(lambda: cv2.getDerivKernels(0, 0, 3))
    results["get_deriv_kernels"] = dk

    # --- Sobel / Scharr / Laplacian
    s = {"src_u8": tolist(img8)}
    s["sobel_u8_dx1_k3_ddepth_same"] = tolist(cv2.Sobel(img8, -1, 1, 0, ksize=3))
    s["sobel_u8_dx1_k3_ddepth_16s"] = tolist(cv2.Sobel(img8, cv2.CV_16S, 1, 0, ksize=3))
    s["sobel_u8_dy1_k3_ddepth_16s"] = tolist(cv2.Sobel(img8, cv2.CV_16S, 0, 1, ksize=3))
    s["sobel_u8_dx1_k3_scale0.3_ddepth_same"] = tolist(cv2.Sobel(img8, -1, 1, 0, ksize=3, scale=0.3))
    s["sobel_u8_dx1_k3_scale0.3_delta0.5_16s"] = tolist(cv2.Sobel(img8, cv2.CV_16S, 1, 0, ksize=3, scale=0.3, delta=0.5))
    s["sobel_u8_dx1_k3_scale0.3_delta0.5_same"] = tolist(cv2.Sobel(img8, -1, 1, 0, ksize=3, scale=0.3, delta=0.5))
    s["sobel_f32_dx1_k3"] = tolist(cv2.Sobel(img32, -1, 1, 0, ksize=3))
    s["sobel_f32_dy1_k3"] = tolist(cv2.Sobel(img32, -1, 0, 1, ksize=3))
    s["sobel_f32_dx1_k5"] = tolist(cv2.Sobel(img32, -1, 1, 0, ksize=5))
    s["sobel_f32_dx2_k3"] = tolist(cv2.Sobel(img32, -1, 2, 0, ksize=3))
    s["sobel_f32_dx1_dy1_k3"] = tolist(cv2.Sobel(img32, -1, 1, 1, ksize=3))
    s["sobel_f32_dx1_k7"] = tolist(cv2.Sobel(img32, -1, 1, 0, ksize=7))
    s["sobel_f32_dx1_k1"] = tolist(cv2.Sobel(img32, -1, 1, 0, ksize=1))
    s["sobel_f32_dx3_k5"] = tolist(cv2.Sobel(img32, -1, 3, 0, ksize=5))
    s["sobel_f32_dx1_k3_replicate"] = tolist(cv2.Sobel(img32, -1, 1, 0, ksize=3, borderType=cv2.BORDER_REPLICATE))
    s["sobel_f32_dx1_k3_constant"] = tolist(cv2.Sobel(img32, -1, 1, 0, ksize=3, borderType=cv2.BORDER_CONSTANT))
    s["sobel_u8_dx0_dy0"] = attempt(lambda: cv2.Sobel(img8, -1, 0, 0, ksize=3))
    s["sobel_u8_k4"] = attempt(lambda: cv2.Sobel(img8, -1, 1, 0, ksize=4))
    s["scharr_f32_dx1"] = tolist(cv2.Scharr(img32, -1, 1, 0))
    s["scharr_f32_dy1"] = tolist(cv2.Scharr(img32, -1, 0, 1))
    s["scharr_u8_dx1_16s"] = tolist(cv2.Scharr(img8, cv2.CV_16S, 1, 0))
    s["scharr_f32_dx1_dy1"] = attempt(lambda: cv2.Scharr(img32, -1, 1, 1))
    s["laplacian_f32_k1"] = tolist(cv2.Laplacian(img32, -1, ksize=1))
    s["laplacian_f32_k3"] = tolist(cv2.Laplacian(img32, -1, ksize=3))
    s["laplacian_f32_k5"] = tolist(cv2.Laplacian(img32, -1, ksize=5))
    s["laplacian_f32_k3_scale0.5_delta2"] = tolist(cv2.Laplacian(img32, -1, ksize=3, scale=0.5, delta=2.0))
    s["laplacian_u8_k1_same"] = tolist(cv2.Laplacian(img8, -1, ksize=1))
    s["laplacian_u8_k1_16s"] = tolist(cv2.Laplacian(img8, cv2.CV_16S, ksize=1))
    s["laplacian_u8_k3_16s"] = tolist(cv2.Laplacian(img8, cv2.CV_16S, ksize=3))
    results["derivatives"] = s

    # --- getStructuringElement
    se = {}
    for shape_name, shape in (("rect", cv2.MORPH_RECT), ("cross", cv2.MORPH_CROSS), ("ellipse", cv2.MORPH_ELLIPSE)):
        for w, h in ((3, 3), (5, 5), (7, 3), (3, 7), (6, 4), (1, 1), (5, 1), (9, 9)):
            se[f"{shape_name}_{w}x{h}"] = tolist(cv2.getStructuringElement(shape, (w, h)))
        se[f"{shape_name}_5x5_anchor11"] = tolist(cv2.getStructuringElement(shape, (5, 5), (1, 1)))
    se["rect_0x3"] = attempt(lambda: cv2.getStructuringElement(cv2.MORPH_RECT, (0, 3)))
    results["get_structuring_element"] = se

    # --- erode / dilate / morphologyEx
    k3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    kc = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    ke5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mo = {"src_u8": tolist(img8)}
    mo["erode_u8_rect3"] = tolist(cv2.erode(img8, k3))
    mo["dilate_u8_rect3"] = tolist(cv2.dilate(img8, k3))
    mo["erode_u8_cross3"] = tolist(cv2.erode(img8, kc))
    mo["dilate_u8_ellipse5"] = tolist(cv2.dilate(img8, ke5))
    mo["erode_u8_rect3_iter2"] = tolist(cv2.erode(img8, k3, iterations=2))
    mo["erode_u8_rect3_iter2_replicate"] = tolist(cv2.erode(img8, k3, iterations=2, borderType=cv2.BORDER_REPLICATE))
    mo["erode_u8_rect3_replicate"] = tolist(cv2.erode(img8, k3, borderType=cv2.BORDER_REPLICATE))
    mo["erode_u8_rect3_reflect101"] = tolist(cv2.erode(img8, k3, borderType=cv2.BORDER_REFLECT_101))
    mo["erode_u8_rect3_constant_value7"] = tolist(cv2.erode(img8, k3, borderType=cv2.BORDER_CONSTANT, borderValue=7))
    mo["erode_u8_rect3_anchor00"] = tolist(cv2.erode(img8, k3, anchor=(0, 0)))
    mo["erode_u8_rect3_iter0"] = tolist(cv2.erode(img8, k3, iterations=0))
    mo["erode_u8_empty_kernel"] = tolist(cv2.erode(img8, np.zeros((0, 0), np.uint8)))
    mo["erode_u8_zero_kernel3"] = tolist(cv2.erode(img8, np.zeros((3, 3), np.uint8)))
    mo["erode_f32_rect3"] = tolist(cv2.erode(img32, k3))
    mo["erode_u8c3_rect3"] = tolist(cv2.erode(img8c3, k3))
    for name, op in (
        ("open", cv2.MORPH_OPEN),
        ("close", cv2.MORPH_CLOSE),
        ("gradient", cv2.MORPH_GRADIENT),
        ("tophat", cv2.MORPH_TOPHAT),
        ("blackhat", cv2.MORPH_BLACKHAT),
        ("hitmiss", cv2.MORPH_HITMISS),
    ):
        mo[f"morph_{name}_u8_rect3"] = tolist(cv2.morphologyEx(img8, op, k3))
    mo["morph_open_u8_rect3_iter2"] = tolist(cv2.morphologyEx(img8, cv2.MORPH_OPEN, k3, iterations=2))
    results["morphology"] = mo

    # --- threshold
    t = {"src_u8": tolist(img8)}
    types = {
        "binary": cv2.THRESH_BINARY,
        "binary_inv": cv2.THRESH_BINARY_INV,
        "trunc": cv2.THRESH_TRUNC,
        "tozero": cv2.THRESH_TOZERO,
        "tozero_inv": cv2.THRESH_TOZERO_INV,
    }
    for name, ty in types.items():
        ret, dst = cv2.threshold(img8, 127.5, 200.0, ty)
        t[f"u8_{name}_t127.5_m200"] = [ret, tolist(dst)]
    for name, ty in types.items():
        ret, dst = cv2.threshold(img32, 127.5, 200.0, ty)
        t[f"f32_{name}_t127.5_m200"] = [ret, tolist(dst)]
    for mv in (254.6, 254.5, 255.5, 300.0, -5.0, 0.4, 0.5, 1.5, 2.5):
        ret, dst = cv2.threshold(img8, 100.0, mv, cv2.THRESH_BINARY)
        t[f"u8_binary_t100_m{mv}"] = [ret, int(dst.max())]
    for th in (-1.0, 255.0, 300.0, 254.9999):
        ret, dst = cv2.threshold(img8, th, 200.0, cv2.THRESH_BINARY)
        t[f"u8_binary_t{th}_m200"] = [ret, tolist(dst)]
    ret, dst = cv2.threshold(img8, 100.7, 200.0, cv2.THRESH_TRUNC)
    t["u8_trunc_t100.7"] = [ret, tolist(dst)]
    ret, dst = cv2.threshold(img8, 100.7, 200.0, cv2.THRESH_TOZERO)
    t["u8_tozero_t100.7"] = [ret, tolist(dst)]
    ret, dst = cv2.threshold(img8c3, 127.5, 200.0, cv2.THRESH_BINARY)
    t["u8c3_binary_t127.5_m200"] = [ret, tolist(dst)]
    ret, dst = cv2.threshold(img8, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    t["u8_otsu"] = [ret, tolist(dst)]
    ret, dst = cv2.threshold(img8, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_TRIANGLE)
    t["u8_triangle"] = [ret, tolist(dst)]
    t["u8_otsu_and_triangle"] = attempt(lambda: cv2.threshold(img8, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_TRIANGLE))
    t["f32_otsu"] = attempt(lambda: cv2.threshold(img32, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU))
    ret, dst = cv2.threshold(img8.astype(np.int16) - 100, 20.5, 90.6, cv2.THRESH_BINARY)
    t["i16_binary_t20.5_m90.6"] = [ret, tolist(dst)]
    ret, dst = cv2.threshold(img8.astype(np.float64), 127.5, 200.0, cv2.THRESH_BINARY)
    t["f64_binary_t127.5_m200"] = [ret, tolist(dst)]
    t["i32_unsupported"] = attempt(lambda: cv2.threshold(img8.astype(np.int32), 127.5, 200.0, cv2.THRESH_BINARY))
    results["threshold"] = t

    dump("imgproc_filter.py", results)


if __name__ == "__main__":
    main()

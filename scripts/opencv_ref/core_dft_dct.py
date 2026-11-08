"""OpenCV 4.10.0 reference values for the core transform rows.

Rows: dft, idft, dct, idct, get_optimal_dft_size and the DFT_* flags. See docs/parity/4.10.0/core.md.

Run from the repository root:
    cpp_ref/.venv/Scripts/python scripts/opencv_ref/core_dft_dct.py   # Windows
    cpp_ref/.venv/bin/python scripts/opencv_ref/core_dft_dct.py       # Linux/macOS
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

# 3x4 real input: small integers so that the spectrum is exact in f32.
SRC = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 13.0]], dtype=np.float32)
ROW8 = np.array([[1.0, 3.0, 2.0, 5.0, 4.0, 7.0, 6.0, 9.0]], dtype=np.float32)
SQ4 = np.array(
    [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0], [13.0, 14.0, 15.0, 17.0]],
    dtype=np.float32,
)

# 1x4 two-channel (complex) input for the idft scaling rows.
COMPLEX_1X4 = np.array([[[1.0, 2.0], [3.0, -1.0], [0.5, 4.0], [-2.0, 1.0]]], dtype=np.float32)


def attempt(fn):
    try:
        return tolist(fn())
    except cv2.error as e:  # noqa: PERF203
        return f"cv2.error: {e.code} {e.err}".strip()


def ramp(rows: int, cols: int) -> np.ndarray:
    """#223 CCS input: element i (row-major) is ((7 * i + 3) mod 11) - 5, as f32."""
    i = np.arange(rows * cols, dtype=np.int64)
    return (((7 * i + 3) % 11) - 5).astype(np.float32).reshape(rows, cols)


# #223: one shape per even/odd combination of rows and cols, plus 1xN and Nx1.
CCS_SHAPES = [(1, 6), (1, 5), (6, 1), (5, 1), (4, 6), (4, 5), (3, 6), (3, 5)]


def ccs_rows() -> dict:
    """#223: dft of real input returns the CCS packing unless DFT_COMPLEX_OUTPUT is passed."""
    inv = cv2.DFT_INVERSE
    out = {}
    for r, c in CCS_SHAPES:
        name = f"{r}x{c}"
        src = ramp(r, c)
        ccs = cv2.dft(src)
        out[f"f32_{name}"] = tolist(ccs)
        out[f"f64_{name}"] = tolist(cv2.dft(src.astype(np.float64)))
        out[f"rows_{name}"] = tolist(cv2.dft(src, flags=cv2.DFT_ROWS))
        # Inverse of CCS input: real output, unscaled and scaled.
        out[f"inv_{name}"] = tolist(cv2.dft(ccs, flags=inv))
        out[f"inv_scale_{name}"] = tolist(cv2.dft(ccs, flags=inv | cv2.DFT_SCALE))
        out[f"inv_scale_f64_{name}"] = tolist(
            cv2.dft(cv2.dft(src.astype(np.float64)), flags=inv | cv2.DFT_SCALE)
        )
    src = ramp(4, 6)
    ccs = cv2.dft(src)
    out["inv_real_output_4x6"] = tolist(cv2.dft(ccs, flags=inv | cv2.DFT_REAL_OUTPUT))
    out["inv_complex_output_4x6"] = tolist(cv2.dft(ccs, flags=inv | cv2.DFT_COMPLEX_OUTPUT))
    out["fwd_real_output_4x6"] = tolist(cv2.dft(src, flags=cv2.DFT_REAL_OUTPUT))
    out["inv_rows_scale_3x6"] = tolist(
        cv2.dft(cv2.dft(ramp(3, 6), flags=cv2.DFT_ROWS), flags=inv | cv2.DFT_ROWS | cv2.DFT_SCALE)
    )
    # nonzero_rows: forward reads only the first rows, inverse writes only the first rows.
    for r, c in [(4, 6), (3, 5)]:
        name = f"{r}x{c}"
        src = ramp(r, c)
        out[f"nz2_{name}"] = tolist(cv2.dft(src, nonzeroRows=2))
        out[f"nz2_rows_{name}"] = tolist(cv2.dft(src, flags=cv2.DFT_ROWS, nonzeroRows=2))
        out[f"inv_nz2_{name}"] = tolist(cv2.dft(cv2.dft(src), flags=inv, nonzeroRows=2))
        out[f"inv_nz2_rows_{name}"] = tolist(
            cv2.dft(cv2.dft(src, flags=cv2.DFT_ROWS), flags=inv | cv2.DFT_ROWS, nonzeroRows=2)
        )
    out["nz2_complex_output_4x6"] = tolist(cv2.dft(ramp(4, 6), flags=cv2.DFT_COMPLEX_OUTPUT, nonzeroRows=2))
    out["nz2_6x1"] = attempt(lambda: cv2.dft(ramp(6, 1), nonzeroRows=2))
    return out


def main() -> None:
    results = {}
    C = cv2.DFT_COMPLEX_OUTPUT
    complex_spec = cv2.dft(SRC, flags=C)

    results["dft"] = {
        "src": tolist(SRC),
        "flags0_ccs_packed_1ch": tolist(cv2.dft(SRC)),
        "complex_output": tolist(complex_spec),
        "complex_output_scale": tolist(cv2.dft(SRC, flags=C | cv2.DFT_SCALE)),
        "complex_output_rows": tolist(cv2.dft(SRC, flags=C | cv2.DFT_ROWS)),
        "complex_output_rows_scale": tolist(cv2.dft(SRC, flags=C | cv2.DFT_ROWS | cv2.DFT_SCALE)),
        "complex_output_nonzero_rows_2": tolist(cv2.dft(SRC, flags=C, nonzeroRows=2)),
        "complex_input_flags0": tolist(cv2.dft(complex_spec)),
        "complex_input_inverse_no_scale": tolist(cv2.dft(complex_spec, flags=cv2.DFT_INVERSE)),
        "idft_default_no_scale": tolist(cv2.idft(complex_spec)),
        "idft_scale": tolist(cv2.idft(complex_spec, flags=cv2.DFT_SCALE)),
        # #224: idft never scales unless DFT_SCALE is passed; DFT_SCALE divides by the element count.
        "idft_scale_real_fixture_default": tolist(cv2.idft(SRC)),
        "idft_scale_real_fixture_scale": tolist(cv2.idft(SRC, flags=cv2.DFT_SCALE)),
        "idft_scale_complex_1x4_default": tolist(cv2.idft(COMPLEX_1X4)),
        "idft_scale_complex_1x4_scale": tolist(cv2.idft(COMPLEX_1X4, flags=cv2.DFT_SCALE)),
        "idft_real_output": tolist(cv2.idft(complex_spec, flags=cv2.DFT_REAL_OUTPUT)),
        "idft_real_output_scale": tolist(cv2.idft(complex_spec, flags=cv2.DFT_REAL_OUTPUT | cv2.DFT_SCALE)),
        "idft_real_output_rows_scale": tolist(
            cv2.idft(
                cv2.dft(SRC, flags=C | cv2.DFT_ROWS),
                flags=cv2.DFT_REAL_OUTPUT | cv2.DFT_ROWS | cv2.DFT_SCALE,
            )
        ),
        "real_and_complex_output_together": attempt(lambda: cv2.dft(SRC, flags=C | cv2.DFT_REAL_OUTPUT)),
        "u8_input": attempt(lambda: cv2.dft(SRC.astype(np.uint8), flags=C)),
        "f64_complex_output": tolist(cv2.dft(SRC.astype(np.float64), flags=C)),
    }
    results["dft_ccs"] = ccs_rows()

    results["dct"] = {
        "row8_src": tolist(ROW8),
        "row8": tolist(cv2.dct(ROW8)),
        "row8_idct_roundtrip": tolist(cv2.idct(cv2.dct(ROW8))),
        "row8_inverse_of_src": tolist(cv2.idct(ROW8)),
        "sq4_src": tolist(SQ4),
        "sq4": tolist(cv2.dct(SQ4)),
        "sq4_rows": tolist(cv2.dct(SQ4, flags=cv2.DCT_ROWS)),
        "sq4_idct": tolist(cv2.idct(SQ4)),
        "odd_size_1x5": attempt(lambda: cv2.dct(np.array([[1.0, 2.0, 3.0, 4.0, 5.0]], dtype=np.float32))),
        "odd_size_3x4": attempt(lambda: cv2.dct(SRC)),
        "f64_row8": tolist(cv2.dct(ROW8.astype(np.float64))),
        "u8_input": attempt(lambda: cv2.dct(ROW8.astype(np.uint8))),
        "two_channel": attempt(lambda: cv2.dct(np.zeros((1, 8, 2), np.float32))),
    }

    sizes = [
        -5, 0, 1, 2, 7, 11, 13, 17, 100, 101, 257, 1000, 1001, 65536, 65537, 100000, 1 << 20,
        (1 << 30) + 1, 2125764000, 2125764001, 2147483647,
    ]
    results["get_optimal_dft_size"] = {str(n): cv2.getOptimalDFTSize(n) for n in sizes}

    dump("core_dft_dct.py", results)


if __name__ == "__main__":
    main()

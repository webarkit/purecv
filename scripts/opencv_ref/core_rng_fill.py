"""OpenCV 4.10.0 reference sequences for the cv::RNG port (#225).

Complements core_rng.py with the RNG::fill paths the port must reproduce: block
boundaries (BLOCK_SIZE = 1024 elements, split per channel count), the fast-bits and
division paths of integer randu, wide integer ranges, reversed bounds for floats, the
ziggurat on several depths and per-channel randn. See docs/parity/4.10.0/core.md.

Run from the repository root:
    cpp_ref/.venv/Scripts/python scripts/opencv_ref/core_rng_fill.py   # Windows
    cpp_ref/.venv/bin/python scripts/opencv_ref/core_rng_fill.py       # Linux/macOS
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


def seeded(seed, shape, dtype, fill, a, b):
    cv2.setRNGSeed(seed)
    m = np.zeros(shape, dtype=dtype)
    fill(m, a, b)
    return m


def main() -> None:
    results = {}

    # 3 channels: blocks of 342 px = 1026 elements, so every block ends with a 2-element
    # tail outside the 4-per-draw small-bits loop.
    m = seeded(42, (1, 400, 3), np.uint8, cv2.randu, (0, 0, 0), (256, 256, 256))
    flat = m.reshape(-1)
    results["randu_u8_3ch_400px_seed42"] = {
        "sum": int(flat.astype(np.int64).sum()),
        "elems_1020_1032": tolist(flat[1020:1032]),
        "last4": tolist(flat[-4:]),
    }

    m = seeded(7, (1, 8), np.uint16, cv2.randu, 0, 65536)
    results["randu_u16_0_65536_seed7"] = tolist(m)

    m = seeded(7, (1, 8), np.int16, cv2.randu, -1000, 1000)
    results["randu_i16_m1000_1000_seed7"] = tolist(m)

    m = seeded(7, (1, 8), np.int32, cv2.randu, -1e9, 1e9)
    results["randu_i32_m1e9_1e9_seed7"] = tolist(m)

    m = seeded(7, (1, 8), np.int8, cv2.randu, -128, 128)
    results["randu_i8_m128_128_seed7"] = tolist(m)

    m = seeded(7, (1, 8), np.float32, cv2.randu, -3.0, 7.0)
    results["randu_f32_m3_7_seed7"] = tolist(m)

    m = seeded(7, (1, 8), np.float64, cv2.randu, 5.0, -5.0)
    results["randu_f64_reversed_5_m5_seed7"] = tolist(m)

    m = seeded(3, (1, 8), np.float64, cv2.randn, 0.0, 1.0)
    results["randn_f64_0_1_seed3"] = tolist(m)

    m = seeded(3, (1, 8), np.int16, cv2.randn, 0.0, 1000.0)
    results["randn_i16_0_1000_seed3"] = tolist(m)

    m = seeded(3, (1, 4, 3), np.float32, cv2.randn, (0, 10, 100), (1, 2, 3))
    results["randn_f32_3ch_seed3"] = tolist(m)

    m = seeded(1, (1, 2000), np.uint8, cv2.randn, 128.0, 200.0)
    results["randn_u8_128_200_2000_seed1"] = {
        "zeros": int((m == 0).sum()),
        "full": int((m == 255).sum()),
        "first8": tolist(m[0, :8]),
    }

    # 5000 normal samples cross the 1024-element block boundary several times.
    m = seeded(11, (1, 5000), np.float32, cv2.randn, 0.0, 1.0)
    results["randn_f32_5000_seed11"] = {
        "elems_1022_1026": tolist(m[0, 1022:1026]),
        "last2": tolist(m[0, -2:]),
    }

    cv2.setRNGSeed(42)
    sh = np.arange(1000, dtype=np.int32).reshape(1, 1000)
    cv2.randShuffle(sh)
    results["randShuffle_0_999_seed42_first10"] = tolist(sh[0, :10])

    dump("core_rng_fill.py", results)


if __name__ == "__main__":
    main()

"""OpenCV 4.10.0 reference sequences for the core RNG rows.

Rows: set_rng_seed, randu, randn, rand_shuffle (cv::theRNG, a 64-bit multiply-with-carry
generator). See docs/parity/core.md.

Run from the repository root:
    cpp_ref/.venv/Scripts/python scripts/opencv_ref/core_rng.py   # Windows
    cpp_ref/.venv/bin/python scripts/opencv_ref/core_rng.py       # Linux/macOS
"""

import cv2

assert cv2.__version__ == "4.10.0"
cv2.ipp.setUseIPP(False)  # measure the generic CPU path

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _core_common import dump, tolist  # noqa: E402


def main() -> None:
    results = {}

    for seed in (0, 42, 12345):
        block = {}
        cv2.setRNGSeed(seed)
        u8 = np.zeros((1, 8), dtype=np.uint8)
        cv2.randu(u8, 0, 256)
        block["randu_u8_0_256"] = tolist(u8)

        cv2.setRNGSeed(seed)
        u8b = np.zeros((1, 8), dtype=np.uint8)
        cv2.randu(u8b, 0, 10)
        block["randu_u8_0_10"] = tolist(u8b)

        cv2.setRNGSeed(seed)
        i32 = np.zeros((1, 8), dtype=np.int32)
        cv2.randu(i32, -5, 5)
        block["randu_i32_m5_5"] = tolist(i32)

        cv2.setRNGSeed(seed)
        f32 = np.zeros((1, 8), dtype=np.float32)
        cv2.randu(f32, 0.0, 1.0)
        block["randu_f32_0_1"] = tolist(f32)

        cv2.setRNGSeed(seed)
        f64 = np.zeros((1, 8), dtype=np.float64)
        cv2.randu(f64, 0.0, 1.0)
        block["randu_f64_0_1"] = tolist(f64)

        cv2.setRNGSeed(seed)
        c3 = np.zeros((1, 2, 3), dtype=np.uint8)
        cv2.randu(c3, (0, 100, 200), (10, 110, 210))
        block["randu_u8_3ch_per_channel_bounds"] = tolist(c3)

        cv2.setRNGSeed(seed)
        n32 = np.zeros((1, 8), dtype=np.float32)
        cv2.randn(n32, 0.0, 1.0)
        block["randn_f32_0_1"] = tolist(n32)

        cv2.setRNGSeed(seed)
        n8 = np.zeros((1, 8), dtype=np.uint8)
        cv2.randn(n8, 128.0, 50.0)
        block["randn_u8_128_50"] = tolist(n8)

        cv2.setRNGSeed(seed)
        sh = np.arange(10, dtype=np.int32).reshape(1, 10)
        cv2.randShuffle(sh)
        block["randShuffle_0_9_i32"] = tolist(sh)

        cv2.setRNGSeed(seed)
        sh8 = np.arange(10, dtype=np.uint8).reshape(1, 10)
        cv2.randShuffle(sh8)
        block["randShuffle_0_9_u8"] = tolist(sh8)

        results[f"seed_{seed}"] = block

    # RNG(0) falls back to the default state 0xffffffff; RNG(-1) is the same 32-bit state.
    cv2.setRNGSeed(0)
    a = np.zeros((1, 4), dtype=np.int32)
    cv2.randu(a, 0, 1000)
    cv2.setRNGSeed(-1)
    b = np.zeros((1, 4), dtype=np.int32)
    cv2.randu(b, 0, 1000)
    results["seed_0_vs_seed_minus1"] = {"seed0": tolist(a), "seed_minus1": tolist(b)}

    dump("core_rng.py", results)


if __name__ == "__main__":
    main()

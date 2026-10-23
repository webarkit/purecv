"""Shared inputs and digests for the features2d parity scripts (docs/parity/4.10.0/features2d.md).

Every image is built from integer arithmetic only, so tests/parity_features2d.rs rebuilds
it byte for byte. Keypoint lists are reported as a count, an FNV-1a digest of their
little-endian field bytes and a short prefix, so that a Rust test can check a long list
without a thousand-line constant.

Imported *after* the script's own ``assert cv2.__version__ == REFERENCE_VERSION`` and
``cv2.ipp.setUseIPP(False)`` lines (see _core_common.py).
"""

import struct

import numpy as np


def lcg_step(state: int) -> int:
    """One step of the 32-bit LCG `state * 1664525 + 1013904223 (mod 2^32)`."""
    return (state * 1664525 + 1013904223) & 0xFFFFFFFF


def noise_u8(rows: int, cols: int, seed: int) -> np.ndarray:
    """Row-major LCG noise: each pixel is the top byte of the next LCG state."""
    out = np.empty(rows * cols, np.uint8)
    state = seed & 0xFFFFFFFF
    for i in range(rows * cols):
        state = lcg_step(state)
        out[i] = state >> 24
    return out.reshape(rows, cols)


def textured_u8(rows: int, cols: int, seed: int, rects: int = 60) -> np.ndarray:
    """`rects` LCG rectangles over a mid-grey background, plus +-6 LCG noise.

    Each draw takes the next LCG state's bits 16..31 (`v = state >> 16`).
    """
    buf = np.full((rows, cols), 128, np.int32)
    state = seed & 0xFFFFFFFF

    def draw():
        nonlocal state
        state = lcg_step(state)
        return state >> 16

    for _ in range(rects):
        w = 4 + draw() % 24
        h = 4 + draw() % 24
        x0 = draw() % cols
        y0 = draw() % rows
        v = draw() % 256
        buf[y0 : min(y0 + h, rows), x0 : min(x0 + w, cols)] = v
    out = np.empty((rows, cols), np.uint8)
    for y in range(rows):
        for x in range(cols):
            out[y, x] = min(max(int(buf[y, x]) + draw() % 13 - 6, 0), 255)
    return out


def squares_u8(rows: int = 40, cols: int = 48) -> np.ndarray:
    """Two bright squares on a dark background (clean corners for FAST)."""
    img = np.full((rows, cols), 30, np.uint8)
    img[4:12, 4:12] = 200
    img[16:28, 20:34] = 220
    return img


def hash32(x: int) -> int:
    """'lowbias32' hash, twin of `hash32()` in gen_orb_descriptors.js."""
    x &= 0xFFFFFFFF
    x = (x ^ (x >> 16)) & 0xFFFFFFFF
    x = (x * 0x7FEB352D) & 0xFFFFFFFF
    x = (x ^ (x >> 15)) & 0xFFFFFFFF
    x = (x * 0x846CA68B) & 0xFFFFFFFF
    return (x ^ (x >> 16)) & 0xFFFFFFFF


def orb_ref_image(w: int = 192, h: int = 192) -> np.ndarray:
    """The 192x192 image of `scripts/opencv_ref/gen_orb_descriptors.js` (`pixel()`)."""
    out = np.empty((h, w), np.uint8)
    for y in range(h):
        for x in range(w):
            coarse = hash32(((y >> 3) * 64 + (x >> 3)) ^ 0x9E3779B9) & 0xFF
            fine = hash32((y * w + x) ^ 0x85EBCA6B) & 0x3F
            v = ((coarse * 3) >> 2) + fine
            if 20 <= x < 84 and 28 <= y < 92:
                v = 255 - v
            if 100 <= x < 172 and 56 <= y < 124:
                v = (v >> 1) + 96
            if 48 <= x < 144 and 132 <= y < 176:
                v ^= 0x5A
            out[y, x] = v & 0xFF
    return out


def fnv1a32(data: bytes) -> int:
    h = 0x811C9DC5
    for b in data:
        h = ((h ^ b) * 0x01000193) & 0xFFFFFFFF
    return h


def kp_fields(kp) -> tuple:
    return (
        float(kp.pt[0]),
        float(kp.pt[1]),
        float(kp.size),
        float(kp.angle),
        float(kp.response),
        int(kp.octave),
        int(kp.class_id),
    )


def kp_digest(kps) -> dict:
    """count + FNV-1a of `<fffffii>` per keypoint (pt.x, pt.y, size, angle, response,
    octave, class_id), the same over `<ff>` (pt.x, pt.y) only, + the first four keypoints."""
    blob = b"".join(struct.pack("<fffffii", *kp_fields(k)) for k in kps)
    xy = b"".join(struct.pack("<ff", float(k.pt[0]), float(k.pt[1])) for k in kps)
    return {
        "count": len(kps),
        "fnv1a": f"0x{fnv1a32(blob):08x}",
        "fnv1a_xy": f"0x{fnv1a32(xy):08x}",
        "first": [kp_fields(k) for k in kps[:4]],
    }


def kp_list(kps) -> list:
    return [kp_fields(k) for k in kps]


def dm_fields(m) -> tuple:
    return (int(m.queryIdx), int(m.trainIdx), int(m.imgIdx), float(m.distance))

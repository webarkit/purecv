"""Shared inputs and digests for the calib3d parity scripts (docs/parity/4.10.0/calib3d.md).

Every input is built from integer LCG draws turned into dyadic fractions (k / 1024, k / 64),
from exact rational rotations (Pythagorean triples) and from plain IEEE double arithmetic in
a fixed operation order, so tests/parity_calib3d.rs rebuilds every point bit for bit without
committing a data file. Results are reported in full (3x3 matrices, pose vectors, inlier
masks) plus FNV-1a digests of the long outputs (undistortion maps).

Imported *after* the script's own ``assert cv2.__version__ == REFERENCE_VERSION`` and
``cv2.ipp.setUseIPP(False)`` lines (see _core_common.py).
"""

import math
import struct

import numpy as np

from _features2d_common import fnv1a32, lcg_step  # noqa: F401


class Lcg:
    """The 32-bit LCG of the other parity scripts; `draw()` returns bits 16..31."""

    def __init__(self, seed: int):
        self.state = seed & 0xFFFFFFFF

    def draw(self) -> int:
        self.state = lcg_step(self.state)
        return self.state >> 16

    def frac(self, denom: int = 1024) -> float:
        """A dyadic fraction in [0, 1): `(draw % denom) / denom` (exact in f32 and f64)."""
        return (self.draw() % denom) / denom

    def signed(self, denom: int = 1024) -> float:
        """A dyadic fraction in [-1, 1]: `(draw % (2 * denom + 1) - denom) / denom`."""
        return (self.draw() % (2 * denom + 1) - denom) / denom


# ---------------------------------------------------------------------------------------
# Camera and poses (exact or order-fixed doubles)
# ---------------------------------------------------------------------------------------

FX, FY, CX, CY = 800.0, 800.0, 320.0, 240.0
WIDTH, HEIGHT = 640, 480


def camera_matrix() -> np.ndarray:
    return np.array([[FX, 0.0, CX], [0.0, FY, CY], [0.0, 0.0, 1.0]], np.float64)


def mat3_mul(a, b):
    """Row-major 3x3 product with the same accumulation order as the Rust twin."""
    c = [0.0] * 9
    for i in range(3):
        for j in range(3):
            s = 0.0
            for k in range(3):
                s += a[i * 3 + k] * b[k * 3 + j]
            c[i * 3 + j] = s
    return c


def rational_rotation(c1, s1, c2, s2, c3, s3):
    """`Rz(c1, s1) * Ry(c3, s3) * Rx(c2, s2)` from the given cosines and sines (row-major)."""
    rz = [c1, -s1, 0.0, s1, c1, 0.0, 0.0, 0.0, 1.0]
    rx = [1.0, 0.0, 0.0, 0.0, c2, -s2, 0.0, s2, c2]
    ry = [c3, 0.0, s3, 0.0, 1.0, 0.0, -s3, 0.0, c3]
    return mat3_mul(mat3_mul(rz, ry), rx)


def pose_a():
    """The object pose of the PnP rows: Pythagorean rotations (5-12-13, 3-4-5, 8-15-17)."""
    r = rational_rotation(12 / 13, 5 / 13, 4 / 5, 3 / 5, 15 / 17, 8 / 17)
    t = [0.25, -0.125, 4.0]
    return r, t


def pose_b():
    """The second camera of the fundamental-matrix rows (a small rotation and a baseline)."""
    r = rational_rotation(24 / 25, 7 / 25, 40 / 41, 9 / 41, 12 / 13, 5 / 13)
    t = [-0.75, 0.125, 0.25]
    return r, t


def transform(r, t, p):
    x, y, z = p
    return (
        r[0] * x + r[1] * y + r[2] * z + t[0],
        r[3] * x + r[4] * y + r[5] * z + t[1],
        r[6] * x + r[7] * y + r[8] * z + t[2],
    )


def project(r, t, p):
    """Pinhole projection `(fx * xc / zc + cx, fy * yc / zc + cy)` in double, no distortion."""
    xc, yc, zc = transform(r, t, p)
    return (FX * (xc / zc) + CX, FY * (yc / zc) + CY)


def apply_homography(h, x, y):
    w = h[6] * x + h[7] * y + h[8]
    if w == 0.0:
        return (math.inf, math.inf)
    return ((h[0] * x + h[1] * y + h[2]) / w, (h[3] * x + h[4] * y + h[5]) / w)


# ---------------------------------------------------------------------------------------
# Seeded point sets
# ---------------------------------------------------------------------------------------


def grid_jittered(cols: int, rows: int, x0: int, y0: int, step_x: int, step_y: int, seed: int):
    """`cols x rows` grid points with a `k / 64` jitter in each coordinate (row-major)."""
    rng = Lcg(seed)
    pts = []
    for j in range(rows):
        for i in range(cols):
            pts.append((x0 + step_x * i + rng.frac(64), y0 + step_y * j + rng.frac(64)))
    return pts


def object_points(n: int, seed: int, planar: bool = False):
    """`n` object points with `k / 1024` coordinates in [-1, 1] (z in [-0.5, 0.5], or 0)."""
    rng = Lcg(seed)
    pts = []
    for _ in range(n):
        x = rng.signed()
        y = rng.signed()
        z = 0.0 if planar else rng.signed() * 0.5
        pts.append((x, y, z))
    return pts


def add_noise(pts, amplitude: float, seed: int):
    """`f32(f64(p) + amplitude * signed)` per coordinate (the Rust twin does the same)."""
    rng = Lcg(seed)
    out = []
    for x, y in pts:
        nx = amplitude * rng.signed()
        ny = amplitude * rng.signed()
        out.append((f32(f32(x) + nx), f32(f32(y) + ny)))
    return out


def replace_outliers(pts, count: int, seed: int):
    """Replace `count` distinct points (LCG indices) by integer pixels inside the image."""
    rng = Lcg(seed)
    out = list(pts)
    chosen = []
    while len(chosen) < count:
        i = rng.draw() % len(pts)
        if i not in chosen:
            chosen.append(i)
            out[i] = (float(rng.draw() % WIDTH), float(rng.draw() % HEIGHT))
    return out, sorted(chosen)


def f32(v: float) -> float:
    return float(np.float32(v))


def as_f32_points(pts) -> np.ndarray:
    """`(N, 1, 2)` or `(N, 1, 3)` float32, the layout cv2 expects."""
    a = np.array(pts, np.float64)
    return a.astype(np.float32).reshape(len(pts), 1, -1)


# ---------------------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------------------


def norm_h(h) -> list:
    """3x3 scaled so that `h[2][2] == 1` (`None` when that entry is 0)."""
    a = np.asarray(h, np.float64).reshape(-1)
    if a.size != 9 or a[8] == 0.0:
        return None
    return [float(v) for v in a / a[8]]


def norm_f(f) -> list:
    """3x3 scaled to unit Frobenius norm with the largest-magnitude entry positive."""
    a = np.asarray(f, np.float64).reshape(-1)
    if a.size != 9:
        return None
    a = a / np.linalg.norm(a)
    k = int(np.argmax(np.abs(a)))
    if a[k] < 0:
        a = -a
    return [float(v) for v in a]


def mask_digest(mask) -> dict:
    m = bytes(int(v) & 0xFF for v in np.asarray(mask).ravel())
    return {
        "count": len(m),
        "inliers": int(sum(1 for v in m if v)),
        "fnv1a": f"0x{fnv1a32(m):08x}",
        "mask": [int(v) for v in m],
    }


def f32_digest(a: np.ndarray) -> str:
    """FNV-1a of the row-major little-endian float32 bytes."""
    return f"0x{fnv1a32(np.ascontiguousarray(a, np.float32).tobytes()):08x}"


def points_f32_digest(pts) -> str:
    return f"0x{fnv1a32(b''.join(struct.pack('<' + 'f' * len(p), *[float(np.float32(v)) for v in p]) for p in pts)):08x}"


def reprojection_rms(obj, img, rvec, tvec, k, dist, cv2):
    proj, _ = cv2.projectPoints(as_f32_points(obj), rvec, tvec, k, dist)
    d = proj.reshape(-1, 2).astype(np.float64) - np.asarray(img, np.float64)
    return float(math.sqrt(np.mean(np.sum(d * d, axis=1))))

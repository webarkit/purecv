"""Shared inputs and digests for the video parity scripts (docs/parity/video.md).

Every image is built from integer arithmetic only, so tests/parity_video.rs rebuilds it
byte for byte. Flow results are reported in full (one `[x, y, status, err]` row per point)
plus an FNV-1a digest of the status bytes, so the Rust tests can check the accept set
exactly and the coordinates within a tolerance.

Imported *after* the script's own ``assert cv2.__version__ == "4.10.0"`` and
``cv2.ipp.setUseIPP(False)`` lines (see _core_common.py).
"""

import struct

import numpy as np

from _features2d_common import fnv1a32, lcg_step, textured_u8  # noqa: F401


def xorshift32(state: int) -> int:
    """One step of the xorshift32 generator of `cell_texture` (src/video/tests.rs)."""
    state ^= (state << 13) & 0xFFFFFFFF
    state ^= state >> 17
    state ^= (state << 5) & 0xFFFFFFFF
    return state & 0xFFFFFFFF


def cell_texture(w: int = 320, h: int = 240) -> np.ndarray:
    """The #163 fixture: random 4..12 px cells of 30/220 in a checkerboard, box-blurred 3x3
    (the mean of the in-image neighbours, truncated). Twin of `cell_texture` in
    src/video/tests.rs; returns `h` rows by `w` columns."""
    state = 0x12345678

    def nxt():
        nonlocal state
        state = xorshift32(state)
        return state

    xs = [0]
    x = 0
    while x < w:
        x += 4 + nxt() % 9
        xs.append(min(x, w))
    ys = [0]
    y = 0
    while y < h:
        y += 4 + nxt() % 9
        ys.append(min(y, h))
    data = np.zeros((h, w), np.int64)
    for yy in range(len(ys) - 1):
        for xx in range(len(xs) - 1):
            v = 30 if (xx + yy) % 2 == 0 else 220
            data[ys[yy] : ys[yy + 1], xs[xx] : xs[xx + 1]] = v
    padded = np.zeros((h + 2, w + 2), np.int64)
    padded[1:-1, 1:-1] = data
    count = np.zeros((h + 2, w + 2), np.int64)
    count[1:-1, 1:-1] = 1
    s = np.zeros((h, w), np.int64)
    n = np.zeros((h, w), np.int64)
    for dy in (0, 1, 2):
        for dx in (0, 1, 2):
            s += padded[dy : dy + h, dx : dx + w]
            n += count[dy : dy + h, dx : dx + w]
    return (s // n).astype(np.uint8)


def shift_image(img: np.ndarray, dx: int, dy: int) -> np.ndarray:
    """Integer translation by (dx, dy); uncovered pixels are 0 (`shift_image` in tests.rs)."""
    h, w = img.shape
    out = np.zeros_like(img)
    ys, xs = np.mgrid[0:h, 0:w]
    sx, sy = xs - dx, ys - dy
    ok = (sx >= 0) & (sx < w) & (sy >= 0) & (sy < h)
    out[ok] = img[sy[ok], sx[ok]]
    return out


def shift_quarter(img: np.ndarray, dx4: int, dy4: int) -> np.ndarray:
    """Sub-pixel translation by (dx4 / 4, dy4 / 4) px with integer bilinear weights and
    replicated edges: `out(x, y) = img(x - dx, y - dy)`. Exact in integers, so the Rust
    twin produces the same bytes."""
    h, w = img.shape
    ix, fx = divmod(-dx4, 4)  # source x = x + ix + fx / 4
    iy, fy = divmod(-dy4, 4)
    ys, xs = np.mgrid[0:h, 0:w]
    x0 = np.clip(xs + ix, 0, w - 1)
    x1 = np.clip(xs + ix + 1, 0, w - 1)
    y0 = np.clip(ys + iy, 0, h - 1)
    y1 = np.clip(ys + iy + 1, 0, h - 1)
    a = img.astype(np.int64)
    v = (
        (4 - fx) * (4 - fy) * a[y0, x0]
        + fx * (4 - fy) * a[y0, x1]
        + (4 - fx) * fy * a[y1, x0]
        + fx * fy * a[y1, x1]
        + 8
    ) >> 4
    return v.astype(np.uint8)


def reusable_texture(rows: int, cols: int, shift: int = 0) -> np.ndarray:
    """`(17 x + 13 y + 3 x y) mod 251` with x shifted right by `shift` (saturating at 0),
    the fixture of the in-tree LK reference tests (`reusable_texture` in tests.rs)."""
    out = np.empty((rows, cols), np.uint8)
    for y in range(rows):
        for x in range(cols):
            xs = max(x - shift, 0)
            out[y, x] = (xs * 17 + y * 13 + xs * y * 3) % 251
    return out


def grid_points(x0: int, x1: int, y0: int, y1: int, step: int) -> np.ndarray:
    """Row-major grid `(x, y)` for `y in range(y0, y1, step)`, `x in range(x0, x1, step)`,
    as the `(N, 1, 2)` float32 array cv2 expects."""
    pts = [[float(x), float(y)] for y in range(y0, y1, step) for x in range(x0, x1, step)]
    return np.array(pts, np.float32).reshape(-1, 1, 2)


def flow_rows(next_pts, status, err) -> list:
    """One `[x, y, status, err]` row per point."""
    p = next_pts.reshape(-1, 2)
    st = status.ravel()
    e = err.ravel()
    return [[float(p[i, 0]), float(p[i, 1]), int(st[i]), float(e[i])] for i in range(len(p))]


def flow_digest(next_pts, status, err) -> dict:
    """tracked count, FNV-1a of the status bytes, and the full per-point rows."""
    st = bytes(int(s) & 0xFF for s in status.ravel())
    return {
        "count": int(len(st)),
        "tracked": int(sum(st)),
        "fnv1a_status": f"0x{fnv1a32(st):08x}",
        "rows": flow_rows(next_pts, status, err),
    }


def pixel_digest(a: np.ndarray) -> str:
    """FNV-1a of the row-major little-endian bytes of `a` (C-contiguous copy)."""
    return f"0x{fnv1a32(np.ascontiguousarray(a).tobytes()):08x}"


def pack_f32(values) -> bytes:
    return b"".join(struct.pack("<f", float(v)) for v in values)

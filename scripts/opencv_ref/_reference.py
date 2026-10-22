"""The OpenCV reference version, read from the pin in cpp_ref/requirements.txt (ADR 0001).

Every measurement script asserts ``cv2.__version__ == REFERENCE_VERSION`` right after
``import cv2``, so a bump of the pin is the only change the scripts need. This module imports
nothing from OpenCV: it can be loaded before ``cv2``.
"""

import pathlib
import re

REQUIREMENTS = pathlib.Path(__file__).resolve().parents[2] / "cpp_ref" / "requirements.txt"
_PIN = re.compile(r"\s*opencv-contrib-python==((\d+)\.(\d+)\.(\d+))\.(\d+)\s*(#.*)?")


def _read_pin():
    """`(X.Y.Z, X.Y.Z.N)` from the `opencv-contrib-python==X.Y.Z.N` line."""
    for line in REQUIREMENTS.read_text(encoding="utf-8").splitlines():
        m = _PIN.fullmatch(line)
        if m:
            return m.group(1), f"{m.group(1)}.{m.group(5)}"
    raise SystemExit(f"no opencv-contrib-python==X.Y.Z.N pin in {REQUIREMENTS}")


REFERENCE_VERSION, WHEEL_VERSION = _read_pin()

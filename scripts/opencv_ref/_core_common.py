"""Shared helpers for the core parity scripts (docs/parity/4.10.0/core.md).

Every script imports this module *after* its own
``assert cv2.__version__ == "4.10.0"`` / ``cv2.ipp.setUseIPP(False)`` lines, so the
evidence rule of docs/parity/README.md stays visible at the top of each script.
"""

import json
import math
import platform
import sys

import cv2
import numpy as np


def build_info() -> dict:
    """The D7 record: version, IPP/optimised state, platform and dispatch lines."""
    lines = cv2.getBuildInformation().splitlines()
    picked = [
        line.strip()
        for line in lines
        if any(k in line for k in ("Version control:", "Baseline:", "Dispatched", "Intel IPP:"))
    ]
    return {
        "cv2_version": cv2.__version__,
        "numpy_version": np.__version__,
        "useIPP": cv2.ipp.useIPP(),
        "useOptimized": cv2.useOptimized(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "build": picked,
    }


def tolist(a):
    """numpy -> JSON-friendly nested lists; non-finite floats become strings."""
    if isinstance(a, np.ndarray):
        return [tolist(x) for x in a.tolist()] if a.ndim > 0 else tolist(a.item())
    if isinstance(a, (list, tuple)):
        return [tolist(x) for x in a]
    if isinstance(a, (np.floating, float)):
        f = float(a)
        if math.isnan(f):
            return "nan"
        if math.isinf(f):
            return "inf" if f > 0 else "-inf"
        return f
    if isinstance(a, (np.integer,)):
        return int(a)
    return a


def dump(name: str, results: dict) -> None:
    out = {"script": name, "build_info": build_info(), "results": results}
    json.dump(out, sys.stdout, indent=1)
    sys.stdout.write("\n")

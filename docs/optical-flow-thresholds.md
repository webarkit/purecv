# Optical-flow threshold units

Optical-flow thresholds should use the same scale as OpenCV. In particular,
the minimum-eigenvalue threshold used by the pyramidal Lucas–Kanade solver is
normalized by `FLT_SCALE = 2^-20` before comparison with the image derivative
metric. Keep the conversion at the algorithm boundary and add a small
synthetic image test when changing it.

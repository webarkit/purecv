"""Line-by-line numpy transcription of cv::LMSolverImpl::run (OpenCV 4.10.0).

Source: https://github.com/opencv/opencv/blob/4.10.0/modules/calib3d/src/levmarq.cpp#L80-L208
cv2 does not bind cv::LMSolver, so this is not a measurement: it reproduces the C++ loop on
Rosenbrock from (-1.2, 1) to pin the trajectory in
src/calib3d/tests.rs (test_lm_solver_rosenbrock_follows_opencv_trajectory). The C++
solve/invert with DECOMP_EIG are written as an eigen pseudo-inverse with SVBkSb's threshold.

Run from the repository root (any Python with numpy):
    cpp_ref/.venv/Scripts/python scripts/opencv_ref/calib3d_lmsolver_transcription.py 200
"""

import sys

import numpy as np

EPS = np.finfo(np.float64).eps


def compute(x, want_j):
    r = np.array([10.0 * (x[1] - x[0] ** 2), 1.0 - x[0]])
    J = np.array([[-20.0 * x[0], 10.0], [-1.0, 0.0]]) if want_j else None
    return r, J


def eig_solve(A, b):
    w, V = np.linalg.eigh(A)
    thr = 2 * EPS * np.abs(w).sum()
    x = np.zeros_like(b)
    for k in range(len(w)):
        if abs(w[k]) > thr:
            x += V[:, k] * (V[:, k] @ b) / w[k]
    return x


def run(x, max_iters, eps=np.float32(np.finfo(np.float32).eps).item()):
    x = np.array(x, float)
    r, J = compute(x, True)
    S = r @ r
    A = J.T @ J
    v = J.T @ r
    D = np.diag(A).copy()
    lam, lc = 1.0, 0.75
    it = 0
    while True:
        Ap = A + np.diag(lam * D)
        d = eig_solve(Ap, v)
        xd = x - d
        rd, _ = compute(xd, False)
        Sd = rd @ rd
        dS = d @ (-A @ d + 2 * v)
        R = (S - Sd) / (dS if abs(dS) > EPS else 1)
        if R > 0.75:
            lam *= 0.5
            if lam < lc:
                lam = 0
        elif R < 0.25:
            t = d @ v
            nu = (Sd - S) / (t if abs(t) > EPS else 1) + 2
            nu = min(max(nu, 2.0), 10.0)
            if lam == 0:
                Ai = np.linalg.pinv(A)
                maxval = max(EPS, np.abs(np.diag(Ai)).max())
                lam = lc = 1.0 / maxval
                nu *= 0.5
            lam *= nu
        if Sd < S:
            S = Sd
            x = xd
            r, J = compute(x, True)
            A = J.T @ J
            v = J.T @ r
        it += 1
        if not (it < max_iters and np.abs(d).max() >= eps and np.abs(r).max() >= eps):
            break
    return (-it if it == max_iters else it), x


r = run([-1.2, 1.0], int(sys.argv[1]) if len(sys.argv) > 1 else 200); print(r[0], repr(float(r[1][0])), repr(float(r[1][1])))

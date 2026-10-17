# Agent rules for `cpp_ref/`

Applies to everything under `cpp_ref/`. Policy: `CONTRIBUTING.md#opencv-reference-version`. Human docs: `cpp_ref/README.md`.

The reference is **OpenCV 4.10.0**: opencv commit `71d3237a093b60a27601c20e9ee6c3e52154e8b1`,
opencv_contrib commit `1ed3dd2c53888e3289afdb22ec4e9ebbff3dba87`. Commands and their
rationale are in [`README.md`](README.md).

## Rules

1. **Bootstrap first.** Run `py -3 cpp_ref/bootstrap.py --bootstrap-file cpp_ref/libraries.json`
   (`python3 …` on Linux/macOS). Continue only when it prints
   `verify opencv @ 71d3237a093b60a27601c20e9ee6c3e52154e8b1 OK` and
   `verify opencv_contrib @ 1ed3dd2c53888e3289afdb22ec4e9ebbff3dba87 OK`. On `MISMATCH` or a
   non-zero exit, stop and report; never repair a tree with manual git commands.
2. Read OpenCV code **only** from `cpp_ref/src/opencv` and `cpp_ref/src/opencv_contrib`.
3. `cpp_ref/opencv` is the maintainer's private 4.x checkout and holds a multi-GB build. Never
   take line numbers from it. Never run git commands that modify it, never build in it, never
   delete anything in it.
4. Never edit, build or create files inside `cpp_ref/src/*`; the bootstrap refuses dirty
   trees. Out-of-tree builds go to `cpp_ref/build/`.
5. Cite what you read as `https://github.com/opencv/opencv/blob/4.10.0/<path>#L<a>-L<b>`
   (`https://github.com/opencv/opencv_contrib/blob/4.10.0/<path>#L<a>-L<b>` for opencv_contrib).
   Line numbers in `cpp_ref/src/opencv` equal those at the tag. Never write a bare
   `<file>.cpp:<line>`.
6. The reference behaviour is the **generic CPU path**. Recognise the alternative back ends:
   `CV_OCL_RUN` (OpenCL), `CV_IPP_RUN*` (IPP), `CALL_HAL`/`cv_hal_*` (HAL), `CV_OVX_RUN`
   (OpenVX), and `*.simd.hpp` / `CV_CPU_DISPATCH` (SIMD dispatch). Port the generic path;
   mention an alternative only when it changes results.
7. Measure only with `cpp_ref/.venv` (scripts start with `assert cv2.__version__ == "4.10.0"`
   and `cv2.ipp.setUseIPP(False)`) or with the pinned opencv.js in `cpp_ref/node_modules`.
   Measurement scripts live in `scripts/opencv_ref/`, never in `cpp_ref/`. Record build
   information (the `Version control:` line of `getBuildInformation()`).
8. Zero-FFI: OpenCV is for reading and measuring. It is never a dependency, a build step or
   code linked into the crate.
9. Change `libraries.json`, `requirements.txt` or `package.json` only in a reference-bump PR
   (roadmap #128).

## Where to look

Paths are under `cpp_ref/src/opencv` unless they start with `opencv_contrib/`.

| purecv module | OpenCV 4.10.0 sources | OpenCV 4.10.0 docs |
|---|---|---|
| `src/core` | `modules/core/src/{arithm,convert.dispatch,copy,norm,stat.dispatch,mathfuncs,lapack,dxt,rand,kmeans}.cpp`, `modules/core/include/opencv2/core/{saturate,fast_math}.hpp` | [Operations on arrays](https://docs.opencv.org/4.10.0/d2/de8/group__core__array.html), [Basic structures](https://docs.opencv.org/4.10.0/dc/d84/group__core__basic.html), [Utility](https://docs.opencv.org/4.10.0/db/de0/group__core__utils.html), [Clustering](https://docs.opencv.org/4.10.0/d5/d38/group__core__cluster.html) |
| `src/imgproc` | `modules/imgproc/src/{smooth.dispatch,box_filter.dispatch,bilateral_filter.dispatch,median_blur.dispatch,morph.dispatch,deriv,resize,imgwarp,pyramids,thresh,histogram,clahe,canny,hough,corner}.cpp`, `modules/imgproc/src/color_rgb.simd.hpp` | [filter](https://docs.opencv.org/4.10.0/d4/d86/group__imgproc__filter.html), [color](https://docs.opencv.org/4.10.0/d8/d01/group__imgproc__color__conversions.html), [transform](https://docs.opencv.org/4.10.0/da/d54/group__imgproc__transform.html), [feature](https://docs.opencv.org/4.10.0/dd/d1a/group__imgproc__feature.html), [hist](https://docs.opencv.org/4.10.0/d6/dc7/group__imgproc__hist.html), [misc](https://docs.opencv.org/4.10.0/d7/d1b/group__imgproc__misc.html) |
| `src/features2d` | `modules/features2d/src/{orb,fast,matchers,draw,keypoint}.cpp`, `modules/core/src/types.cpp` (KeyPoint), `opencv_contrib/modules/xfeatures2d` | [features2d main](https://docs.opencv.org/4.10.0/d5/d51/group__features2d__main.html), [draw](https://docs.opencv.org/4.10.0/d4/d5d/group__features2d__draw.html), [KeyPoint](https://docs.opencv.org/4.10.0/d2/d29/classcv_1_1KeyPoint.html), [BFMatcher](https://docs.opencv.org/4.10.0/d3/da1/classcv_1_1BFMatcher.html) |
| `src/video` | `modules/video/src/lkpyramid.cpp`, `modules/video/include/opencv2/video/tracking.hpp` | [video_track](https://docs.opencv.org/4.10.0/dc/d6b/group__video__track.html) |
| `src/calib3d` | `modules/calib3d/src/{fundam,ptsetreg,solvepnp,calibration,undistort.dispatch}.cpp`, `modules/calib3d/include/opencv2/calib3d.hpp` | [calib3d](https://docs.opencv.org/4.10.0/d9/d0c/group__calib3d.html) |

## Finding a function

```sh
# declaration and default arguments
rg -n "CV_EXPORTS_W.*\b<Name>\(" cpp_ref/src/opencv/modules/<m>/include
# implementation and its dispatch
rg -n "\b<Name>\(" cpp_ref/src/opencv/modules/<m>/src
```

## Parity audits

Method, statuses and dimensions: `docs/parity/README.md` (S1–S7, D1–D7). One document per
module: `docs/parity/<module>.md`. The umbrella issue is #172; the sub-issues are #173–#177.

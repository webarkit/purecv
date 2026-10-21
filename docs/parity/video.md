# Parity audit: `video` ↔ OpenCV 4.10.0

Task B4 of [#172](https://github.com/webarkit/purecv/issues/172), sub-issue
[#176](https://github.com/webarkit/purecv/issues/176). Method, statuses (`measured`,
`inspected`, `diverges`, `intentional`, `purecv-only`, `missing`) and dimensions D1–D7:
[`README.md`](README.md). Inventory taken at the branch base `5201b8a` (`parity-audit` after
the features2d audit, PR #249): every `pub` item of `src/video` (`src/video.rs`,
`src/video/optical_flow.rs`), the public fields of `OpticalFlowPyramid`, and the re-exports of
`src/video.rs` and of the crate prelude. The module has no public inherent methods, no enums
and no traits; `src/video/simd.rs` is `pub(crate)`. The wasm wrappers
(`crates/wasm/src/lib.rs`, `buildOpticalFlowPyramid` / `calcOpticalFlowPyrLK`) call these
items and are not audited separately.

## D7: reference and measurement record

- **Sources read:** `cpp_ref/src/opencv` at `71d3237a093b60a27601c20e9ee6c3e52154e8b1`
  (`verify opencv @ … OK`), `cpp_ref/src/opencv_contrib` at
  `1ed3dd2c53888e3289afdb22ec4e9ebbff3dba87` (`verify opencv_contrib @ … OK`), bootstrapped on
  2026-10-20. The generic CPU path was read: `buildOpticalFlowPyramid`, `calcScharrDeriv` with
  the scalar loops of `ScharrDerivInvoker`, `SparsePyrLKOpticalFlowImpl::calc` and the scalar
  loops of `LKTrackerInvoker::operator()`, with `deriv_type = short` and
  `acctype = itemtype = float` (x86). The `CV_SIMD128 && !CV_NEON` and `CV_NEON` blocks of
  `LKTrackerInvoker`, the OpenCL back end (`ocl_calcOpticalFlowPyrLK`) and the OpenVX one
  (disabled upstream, "bad accuracy") were skipped. The SIMD block is not an alternative the
  wheel can switch off, see the next item.
- **Measurements (Python):** `cpp_ref/.venv`, `pip freeze` = `numpy==2.4.6`,
  `opencv-contrib-python==4.10.0.84`; `cv2.__version__` = `4.10.0`; `cv2.ipp.setUseIPP(False)`
  → `useIPP()` = False; `useOptimized()` = True; Python 3.12.2; Windows 11 10.0.26300 x86-64;
  `getBuildInformation()`: `Version control: 4.10.0`, `Baseline: SSE SSE2 SSE3`,
  `Dispatched code generation: SSE4_1 SSE4_2 FP16 AVX AVX2`, `Intel IPP: 2021.11.0`
  (disabled at run time). Nothing in this module runs through IPP. **The wheel's LK runs the
  `CV_SIMD128` block** of `LKTrackerInvoker`
  ([L282-L352](https://github.com/opencv/opencv/blob/4.10.0/modules/video/src/lkpyramid.cpp#L282-L352),
  [L537-L565](https://github.com/opencv/opencv/blob/4.10.0/modules/video/src/lkpyramid.cpp#L537-L565)):
  the block is compiled in at the SSE2 baseline (not dispatched) and
  `cv2.setUseOptimized(False)` does not change its output (checked: bit-identical with it on
  and off, and with one thread). It consumes the window 8 pixels at a time and sums the
  gradient matrix and the mismatch vector in four `f32` lanes reduced at the end
  ([L457-L461](https://github.com/opencv/opencv/blob/4.10.0/modules/video/src/lkpyramid.cpp#L457-L461),
  [L648-L653](https://github.com/opencv/opencv/blob/4.10.0/modules/video/src/lkpyramid.cpp#L648-L653)),
  while purecv keeps the scalar loop's order
  ([L438-L453](https://github.com/opencv/opencv/blob/4.10.0/modules/video/src/lkpyramid.cpp#L438-L453),
  [L638-L645](https://github.com/opencv/opencv/blob/4.10.0/modules/video/src/lkpyramid.cpp#L638-L645)).
  So windows **narrower than 8 px** (3x3, 5x5, 7x7) are measured against the generic path and
  agree to 1 ulp; windows **8 px or wider** (10, 15, 20, 21) agree to 1.2e-3 px in position and
  0.013 in mean absolute error (a few intensity counts over 441 samples), with identical
  statuses. The in-tree fixtures of PR #167, taken from an out-of-tree build with
  `CV_ENABLE_INTRINSICS=OFF`, agree with purecv to 2e-6 on 5x5 windows, consistent with this.
- **Not measurable with `cv2`:** the `vector<Mat>` form of `prevImg`/`nextImg` (the binding
  takes one Mat or UMat), so `calc_optical_flow_pyramid_lk_with_pyramids` is measured through
  explicitly padded images (`asymmetric_borders`) and through purecv's own raw ≡ prebuilt
  equivalence (in-tree tests); the padding content of a built pyramid (the binding returns a
  contiguous copy of each level's ROI); `winSize.area()` overflow (#155).
- **Scripts (evidence rule):** `scripts/opencv_ref/video_pyramid.py` and `video_lk.py`
  (helpers `_core_common.py`, `_features2d_common.py`, `_video_common.py`). Every input is
  built from integer formulas: the #163 cell texture (xorshift32 cells, 3x3 box mean), the
  LCG rectangle texture of the features2d audit with a quarter-pixel bilinear shift computed in
  integers, and the `(17x + 13y + 3xy) mod 251` fixture of the in-tree reference tests. Flow
  results are recorded in full (`[x, y, status, err]` per point) plus an FNV-1a digest of the
  status bytes; pyramid levels and Scharr planes as FNV-1a digests of their bytes.
- **Tests:** `tests/parity_video.rs`. 12 passing tests pin the rows that match; 4 tests marked
  `#[ignore = "diverges from OpenCV 4.10.0, …"]` are the executable form of the `diverges` rows
  and all fail when run with `--ignored`. Expected values are constants generated from the
  scripts' JSON (`mod ocv`): 1,624 flow rows and 56 pyramids.
- **D6:** every case of both scripts was run through purecv with four builds: default (`std`,
  `parallel`), `--features simd`, `--no-default-features --features std` and
  `--no-default-features --features std,simd`, all `--release`. **The four dumps are
  byte-identical** (every level, derivative plane, position, status and error): the `pulp`
  kernels of `src/video/simd.rs` keep the scalar summation order, and `parallel` only
  distributes points. `wasm32` was not run.
- **Row grouping:** `calc_optical_flow_pyramid_lk` has one row per behaviour (entry point,
  validation, each seeded fix, borders, termination, flags, outputs) rather than one row per
  function, because each behaviour has its own issue history; the prebuilt entry point has
  its own rows.

## Summary

| Status | Rows |
|---|---|
| `measured` | 19 |
| `inspected` | 5 |
| `diverges` | 4 |
| `intentional` | 2 |
| `purecv-only` | 1 |
| `missing` | 21 |

Headline findings, in order of impact (details in the rows, issues in
[Proposed issues](#proposed-issues)):

1. **Pyramidal LK matches OpenCV 4.10.0.** On the #163 setup (320x240 cell texture, 150
   points, shifts of 0 and 10 px, 21x21 window, four levels, OpenCV's defaults) the accept
   sets are identical (150, 150, 149 and 145 points) and every tracked position is within
   4.1e-4 px of the wheel's; with 7x7 and 3x3 windows, where the wheel runs the scalar loop,
   the positions agree to 1 ulp (6.8e-6 px at x ≈ 130). Even (20x20, 10x10) and non-square
   (15x9) windows, depths 0 to 5, thresholds up to 0.5, every criteria form, good and bad
   initial estimates, a sub-pixel shift of (2.25, −0.75) px, points on and outside the
   image, and `BORDER_CONSTANT`/`REPLICATE`/`WRAP` paddings all agree the same way. The seven
   fixes pinned by tests on `dev` (#130, #131, #138, #142, #144, #145, #149), the border fix
   of PR #164 (#163, previously validated against OpenCV 5.0.0) and the final-bounds gate of
   PR #167 (#178 item 1) are now `measured` against 4.10.0, except the determinant guard
   (#142) and the oscillation half-step (#131), whose branches no measured case isolates.
2. **The pyramid builder is bit-identical.** Level count (the `winSize` cutoff), level sizes,
   level pixels and both Scharr planes match `cv2.buildOpticalFlowPyramid` on 112 cases (eight
   images from 1x1 to 320x240, seven `(winSize, maxLevel)` settings, with and without
   derivatives); the padding borders do not change the stored pixels, and level 0 is the input.
3. **Input-validation rows, two new findings.** OpenCV tracks on images of any channel count
   (the window spans `winSize.width * cn` samples; three replicated channels give the
   single-channel result to 1 ulp), purecv rejects everything but one channel
   ([#256](https://github.com/webarkit/purecv/issues/256), `keep?`, the twin of #252); OpenCV asserts `maxLevel >= 0`,
   purecv clamps a negative `max_level` to 0 and tracks ([#257](https://github.com/webarkit/purecv/issues/257)).
4. **Documented conventions, not bugs.** Lost points return the template position and
   `f32::MAX` where OpenCV leaves the last estimate and an undefined error; a pyramid with a
   `Transparent` / `Isolated` border is built but rejected at tracking time, where OpenCV would
   read unwritten padding; a window larger than the builder's is accepted where OpenCV asserts
   on the padding (a `diverges` row, tracked by #184); the pyramid is
   a struct of unpadded `f32` buffers instead of a padded `vector<Mat>` (#183, #184 track the
   representation, not a result difference).

What matches: `OPTFLOW_*` values, `buildOpticalFlowPyramid` (everything observable), the
Scharr derivatives (#130), the `FLT_SCALE` eigenvalue scale and `minEigThreshold` (#138), exact
`winSize.width x winSize.height` sampling with half-integer offsets (#144), coarse-level
degeneracy handling (#145), the x32 temporal difference (#149), reflect-101 image and zero
derivative padding with the window-corner bounds (#163, and #154's edge blending is gone),
the gate after the last iteration (#178 item 1), the `TermCriteria` normalisation (count
0..100, epsilon 0..10, squared, `<=`), `OPTFLOW_USE_INITIAL_FLOW` (scaled to the coarsest
level; ignored without the flag), `OPTFLOW_LK_GET_MIN_EIGENVALS`, the mean absolute error,
every `CV_Assert` purecv can express, and the prebuilt pyramids' `pyrBorder` / `derivBorder`
semantics at level 0.

Link bases used below: `H` = `https://github.com/opencv/opencv/blob/4.10.0/modules/video/include/opencv2/video/tracking.hpp`,
`S` = `https://github.com/opencv/opencv/blob/4.10.0/modules/video/src/lkpyramid.cpp` (written
out in full in every link).

## Modules, constants and types

| purecv item | OpenCV 4.10.0 counterpart (pinned link) | Status | Dims checked | Direction / notes | Issue |
|---|---|---|---|---|---|
| `video::optical_flow` (public submodule); re-exports in `video` and `prelude` | — | `purecv-only` | | Module layout; every public item is also re-exported at `purecv::video` and `purecv::prelude`. OpenCV's `SparsePyrLKOpticalFlow` class ([H L694-L718](https://github.com/opencv/opencv/blob/4.10.0/modules/video/include/opencv2/video/tracking.hpp#L694-L718)) and the `DenseOpticalFlow` / `SparseOpticalFlow` interfaces have no counterpart ([Missing](#missing-in-purecv-s4)). | |
| `OPTFLOW_USE_INITIAL_FLOW` | [`OPTFLOW_USE_INITIAL_FLOW`](https://github.com/opencv/opencv/blob/4.10.0/modules/video/include/opencv2/video/tracking.hpp#L56-L59) | `measured` | D1, D7 | `4` = `cv2.OPTFLOW_USE_INITIAL_FLOW`. `pub const i32` vs an anonymous enum. Test `optflow_flags_match_opencv`. | |
| `OPTFLOW_LK_GET_MIN_EIGENVALS` | [`OPTFLOW_LK_GET_MIN_EIGENVALS`](https://github.com/opencv/opencv/blob/4.10.0/modules/video/include/opencv2/video/tracking.hpp#L56-L59) | `measured` | D1, D7 | `8` = `cv2.OPTFLOW_LK_GET_MIN_EIGENVALS`. `OPTFLOW_FARNEBACK_GAUSSIAN = 256` is `missing` with Farneback. Test `optflow_flags_match_opencv`. | |
| `OpticalFlowPyramid` (struct; `pub` fields `levels`, `dx`, `dy`, `pyr_border`, `deriv_border`; `Debug`, `Clone`) | the `vector<Mat>` of [`buildOpticalFlowPyramid`](https://github.com/opencv/opencv/blob/4.10.0/modules/video/src/lkpyramid.cpp#L726-L835): `CV_8UCn` levels padded by `winSize`, optionally interleaved with `CV_16SC(2n)` Scharr planes ([`deriv_type = short`](https://github.com/opencv/opencv/blob/4.10.0/modules/video/src/lkpyramid.hpp#L8)) | `inspected` | D1, D2, D7 | Same information, different representation: unpadded `Matrix<f32>` levels holding `u8` values and separate `dx`/`dy` holding `i16` values (both validated on use; OpenCV's types make non-integer values impossible), plus the two borders OpenCV materialises as padding. `calc_optical_flow_pyramid_lk_with_pyramids` reproduces the padding when sampling, so results do not depend on the representation (rows below). The typed storage ([#183](https://github.com/webarkit/purecv/issues/183)) and OpenCV-style prepadding ([#184](https://github.com/webarkit/purecv/issues/184)) are tracked as enhancements, not as divergences. | [#183](https://github.com/webarkit/purecv/issues/183), [#184](https://github.com/webarkit/purecv/issues/184) |

## Pyramid (`build_optical_flow_pyramid`)

| purecv item | OpenCV 4.10.0 counterpart (pinned link) | Status | Dims checked | Direction / notes | Issue |
|---|---|---|---|---|---|
| `build_optical_flow_pyramid` | [`buildOpticalFlowPyramid`](https://github.com/opencv/opencv/blob/4.10.0/modules/video/include/opencv2/video/tracking.hpp#L121-L125) ([S L726-L835](https://github.com/opencv/opencv/blob/4.10.0/modules/video/src/lkpyramid.cpp#L726-L835)) | `measured` | D1-D5, D7 | **D1:** `(img, win_size, max_level: usize, with_derivatives, pyr_border, deriv_border)` vs `(img, pyramid, winSize, maxLevel, withDerivatives = true, pyrBorder = BORDER_REFLECT_101, derivBorder = BORDER_CONSTANT, tryReuseInputImage = true)`: no defaults in Rust, the struct is returned instead of the level count plus an output array, `tryReuseInputImage` is absent (an aliasing choice for level 0, [S L739-L753](https://github.com/opencv/opencv/blob/4.10.0/modules/video/src/lkpyramid.cpp#L739-L753), invisible in the values: measured `same_as_default` with it off), `max_level` is unsigned where OpenCV's `-1` fails a container assert. **D2:** `Matrix<u8>` only; OpenCV asserts `CV_8U` ([S L731](https://github.com/opencv/opencv/blob/4.10.0/modules/video/src/lkpyramid.cpp#L731)) but takes any channel count (next row). **D3:** downsampling is `pyrDown` with its default reflect-101 ([S L787](https://github.com/opencv/opencv/blob/4.10.0/modules/video/src/lkpyramid.cpp#L787)); `pyrBorder` only pads the finished level ([S L789-L791](https://github.com/opencv/opencv/blob/4.10.0/modules/video/src/lkpyramid.cpp#L789-L791)): measured, the stored pixels and derivatives are identical for `REFLECT_101`, `REPLICATE`, `CONSTANT`, `WRAP` and `REFLECT`, both `derivBorder`s and both `tryReuseInputImage` settings. **D4/D5:** purecv converts to `f32` and runs the generic `pyr_down`, whose rounding to integers ([#238](https://github.com/webarkit/purecv/issues/238)) is exactly OpenCV's `CV_8U` `pyrDown` here: 112 cases identical (level count = the cutoff `(w+1)/2 <= winSize.width || (h+1)/2 <= winSize.height`, [S L811-L816](https://github.com/opencv/opencv/blob/4.10.0/modules/video/src/lkpyramid.cpp#L811-L816); sizes; level pixels; `dx`/`dy` planes of `calcScharrDeriv`, [S L58-L67](https://github.com/opencv/opencv/blob/4.10.0/modules/video/src/lkpyramid.cpp#L58-L67), [L70-L181](https://github.com/opencv/opencv/blob/4.10.0/modules/video/src/lkpyramid.cpp#L70-L181), whose edge rule `y > 0 ? y-1 : rows > 1 ? 1 : 0` is reflect-101) on eight images (1x1, 1x7, 3x2, 7x5, 9x9, 33x49, 96x128, 320x240) and seven `(winSize, maxLevel)` settings. Test `build_optical_flow_pyramid_matches_opencv`; script `video_pyramid.py` (`grid`, `borders`, `level0_is_input`). | |
| `build_optical_flow_pyramid`: window validation | [`CV_Assert(… winSize.width > 2 && winSize.height > 2)`](https://github.com/opencv/opencv/blob/4.10.0/modules/video/src/lkpyramid.cpp#L731) | `measured` | D2, D7 | `(2, 5)` and `(5, 2)` → `Err(InvalidInput)` vs `cv2.error -215`. Test `build_optical_flow_pyramid_rejects_small_windows_like_opencv`. | |
| `build_optical_flow_pyramid`: multi-channel input | any `cn` ([S L737](https://github.com/opencv/opencv/blob/4.10.0/modules/video/src/lkpyramid.cpp#L737): `derivType = CV_16SC(cn*2)`) | `diverges` | D2, D7 | **keep?** OpenCV builds `CV_8UC3` levels and `CV_16SC6` derivatives (measured: shapes `(33, 49, 3)` / `(33, 49, 6)` …); purecv returns `Err("must be single-channel")`. Test `build_optical_flow_pyramid_accepts_multi_channel_input` (ignored). | [#256](https://github.com/webarkit/purecv/issues/256) |

## Lucas-Kanade (`calc_optical_flow_pyramid_lk`)

| purecv item | OpenCV 4.10.0 counterpart (pinned link) | Status | Dims checked | Direction / notes | Issue |
|---|---|---|---|---|---|
| `calc_optical_flow_pyramid_lk` | [`calcOpticalFlowPyrLK`](https://github.com/opencv/opencv/blob/4.10.0/modules/video/include/opencv2/video/tracking.hpp#L178-L183) → [`SparsePyrLKOpticalFlowImpl::calc`](https://github.com/opencv/opencv/blob/4.10.0/modules/video/src/lkpyramid.cpp#L1239-L1395), [`LKTrackerInvoker::operator()`](https://github.com/opencv/opencv/blob/4.10.0/modules/video/src/lkpyramid.cpp#L183-L722) | `measured` | D1-D7 | **D1:** `(prev_img, next_img, prev_pts, initial_next_pts: Option<&[Point2f]>, win_size, max_level: i32, criteria, flags, min_eigen_threshold)` → `(next_pts, status, err)` vs `(prevImg, nextImg, prevPts, nextPts in/out, status, err, winSize = 21x21, maxLevel = 3, criteria = (COUNT+EPS, 30, 0.01), flags = 0, minEigThreshold = 1e-4)`: no defaults in Rust, `nextPts` split into an optional input and a returned output, `err` always computed (OpenCV: only when requested). **D2:** next rows. **D4/D5:** the fixed-point path is ported as is: `cvRound` 14-bit bilinear weights ([S L232-L239](https://github.com/opencv/opencv/blob/4.10.0/modules/video/src/lkpyramid.cpp#L232-L239)), intensities descaled by `W_BITS1-5` (x32) and derivatives by `W_BITS1` ([S L438-L449](https://github.com/opencv/opencv/blob/4.10.0/modules/video/src/lkpyramid.cpp#L438-L449)), `f32` sums, `FLT_SCALE` ([S L469-L471](https://github.com/opencv/opencv/blob/4.10.0/modules/video/src/lkpyramid.cpp#L469-L471)), the `(A12*b2 - A22*b1) * D` step ([S L663-L668](https://github.com/opencv/opencv/blob/4.10.0/modules/video/src/lkpyramid.cpp#L663-L668)), the target coordinate propagated as `nextPt * 2` ([S L201-L212](https://github.com/opencv/opencv/blob/4.10.0/modules/video/src/lkpyramid.cpp#L201-L212)). **D7:** the #163 setup (320x240 cell texture, 150 points, shifts (0,0), (10,0), (0,10), (10,10), 21x21, 4 levels, defaults, `OPTFLOW_USE_INITIAL_FLOW` with `nextPts = prevPts`, and (10,0) without flags): statuses identical (150/150/149/145 tracked), positions within 4.1e-4 px, MAE within 0.013; 21 parameter variations on 30 points; a sub-pixel shift of (2.25, −0.75) px of a 96x128 rectangle texture, 48 points, 7 variations (positions within 2.9e-4 px). Windows narrower than 8 px agree to 1 ulp (D7 record above). Tests `lk_grid163_matches_opencv`, `lk_parameter_variations_match_opencv`, `lk_subpixel_shift_matches_opencv`; script `video_lk.py` (`grid163`, `variations`, `subpixel`). | |
| `calc_optical_flow_pyramid_lk`: input validation | [`CV_Assert(maxLevel >= 0 && winSize.width > 2 && winSize.height > 2)`](https://github.com/opencv/opencv/blob/4.10.0/modules/video/src/lkpyramid.cpp#L1257), [`npoints == 0`](https://github.com/opencv/opencv/blob/4.10.0/modules/video/src/lkpyramid.cpp#L1262-L1269), [`checkVector(2, CV_32F) == npoints`](https://github.com/opencv/opencv/blob/4.10.0/modules/video/src/lkpyramid.cpp#L1271-L1275), [per-level size and type asserts](https://github.com/opencv/opencv/blob/4.10.0/modules/video/src/lkpyramid.cpp#L1394-L1395) | `measured` | D2, D7 | `Err` where OpenCV asserts: a window side below 3, images of different sizes, an `initial_next_pts` of the wrong length with the flag. `Ok` where OpenCV proceeds: no points (empty outputs; OpenCV releases the outputs), a negative `minEigThreshold` (nothing filtered, same rows). `f32`/`u16` images are a type error in Rust and a `CV_8U` assert in OpenCV. Test `lk_input_validation_matches_opencv`; script `inputs_d2`. | |
| `calc_optical_flow_pyramid_lk`: multi-channel images | `cn = I.channels()`, window of `winSize.width * cn` samples ([S L192](https://github.com/opencv/opencv/blob/4.10.0/modules/video/src/lkpyramid.cpp#L192), [L272-L453](https://github.com/opencv/opencv/blob/4.10.0/modules/video/src/lkpyramid.cpp#L272-L453)), MAE over `32 * w * cn * h` ([S L721](https://github.com/opencv/opencv/blob/4.10.0/modules/video/src/lkpyramid.cpp#L721)) | `diverges` | D2, D7 | **keep?** OpenCV tracks on `CV_8UC3` (three replicated channels reproduce the one-channel rows to 1 ulp, measured); purecv returns `Err("must be single-channel grayscale")`. Same question as [#252](https://github.com/webarkit/purecv/issues/252) in features2d (there OpenCV converts to gray; here it tracks per channel). Test `lk_accepts_multi_channel_images` (ignored). | [#256](https://github.com/webarkit/purecv/issues/256) |
| `calc_optical_flow_pyramid_lk`: negative `max_level` | [`CV_Assert(maxLevel >= 0 …)`](https://github.com/opencv/opencv/blob/4.10.0/modules/video/src/lkpyramid.cpp#L1257) | `diverges` | D2, D7 | **→ OpenCV.** `max_level = -1` is clamped to 0 (`max_level.max(0)`) and tracks; OpenCV raises `cv2.error -215`. Test `lk_rejects_negative_max_level` (ignored). | [#257](https://github.com/webarkit/purecv/issues/257) |
| `calc_optical_flow_pyramid_lk`: `win_size` area overflow | [`AutoBuffer<deriv_type> _buf(winSize.area()*(cn + cn2))`](https://github.com/opencv/opencv/blob/4.10.0/modules/video/src/lkpyramid.cpp#L192-L197) | `inspected` | D2 | `TrackingWindow::area` multiplies the `i32` sides unchecked: a debug panic / wrapped capacity for `width * height > i32::MAX` (#155, open). OpenCV's `Size::area()` is the same `int` product, so this is a robustness bug under the crate's no-panic rule, not a parity difference; not measured (the reference would try to allocate gigabytes). | [#155](https://github.com/webarkit/purecv/issues/155) |
| Scharr derivatives (#130) | [`calcScharrDeriv`](https://github.com/opencv/opencv/blob/4.10.0/modules/video/src/lkpyramid.cpp#L58-L67) / [`ScharrDerivInvoker`](https://github.com/opencv/opencv/blob/4.10.0/modules/video/src/lkpyramid.cpp#L70-L181), called per level in the raw path ([S L1379-L1389](https://github.com/opencv/opencv/blob/4.10.0/modules/video/src/lkpyramid.cpp#L1379-L1389)) | `measured` | D3-D5, D7 | `[3, 10, 3] ⊗ [-1, 0, 1]` (gain 32, `i16`), reflect-101 edges: the `dx`/`dy` planes are identical on every level of the 56 derivative pyramids (pyramid row). purecv computes them with `imgproc::scharr` on the `f32` levels (same integers). In-tree pin: `test_calc_optical_flow_pyramid_lk_uses_scharr_derivatives`. | #130 (closed) |
| `FLT_SCALE` and the `min_eigen_threshold` scale (#138) | [`FLT_SCALE = 1/2^20`](https://github.com/opencv/opencv/blob/4.10.0/modules/video/src/lkpyramid.cpp#L235), [`A = iA * FLT_SCALE`, `minEig = … / (2 * w * h)`](https://github.com/opencv/opencv/blob/4.10.0/modules/video/src/lkpyramid.cpp#L469-L475), [`minEig < (float)minEigThreshold`](https://github.com/opencv/opencv/blob/4.10.0/modules/video/src/lkpyramid.cpp#L480) | `measured` | D4, D5, D7 | The minimum eigenvalue returned with `OPTFLOW_LK_GET_MIN_EIGENVALS` is within 7.2e-7 (1 ulp of 0.63) of OpenCV's on 30 + 48 + 15 points, and thresholds of `1e-2` and `0.5` keep and lose the same points (`min_eig_0.5` loses none on the cell texture but changes which levels refine, same positions to 1.9e-4 px). The threshold is cast to `f32` before the comparison on both sides. In-tree pin: `test_lk_min_eigen_matches_opencv_scale`. | #138 (closed) |
| Determinant guard (#142) | [`D < FLT_EPSILON`](https://github.com/opencv/opencv/blob/4.10.0/modules/video/src/lkpyramid.cpp#L473-L485) on the scaled determinant | `inspected` | D4, D5 | Same expression on `f32` (`det < f32::EPSILON` after `FLT_SCALE`), same `level == 0` rule. No measured case triggers it with `minEig >= minEigThreshold` (the `minEigThreshold = -1` case tracks both points like OpenCV); pinned from both sides by the in-tree `test_lk_rejects_near_degenerate_window` / `test_lk_accepts_window_just_above_det_threshold`, whose constants are derived from the source, not measured. | #142 (closed) |
| Window sampling (#144) | [`halfWin = ((w-1)*0.5f, (h-1)*0.5f)`](https://github.com/opencv/opencv/blob/4.10.0/modules/video/src/lkpyramid.cpp#L187), exactly `winSize.height x winSize.width` samples ([S L272-L453](https://github.com/opencv/opencv/blob/4.10.0/modules/video/src/lkpyramid.cpp#L272-L453)) | `measured` | D4, D5, D7 | Even windows (20x20 on the cell texture, 10x10 on the sub-pixel texture; half-integer offsets) and the non-square 15x9 agree with OpenCV (statuses identical, positions within 1.2e-3 px, 8.7e-5 px and 3.5e-4 px). In-tree pin: `test_lk_min_eigen_uses_exact_win_size_samples`. | #144 (closed) |
| Coarse-level degeneracy (#145) | [`if (level == 0) status = false; continue;`](https://github.com/opencv/opencv/blob/4.10.0/modules/video/src/lkpyramid.cpp#L480-L485), the same rule for a window outside the level ([S L219-L230](https://github.com/opencv/opencv/blob/4.10.0/modules/video/src/lkpyramid.cpp#L219-L230), [L497-L503](https://github.com/opencv/opencv/blob/4.10.0/modules/video/src/lkpyramid.cpp#L497-L503)) | `measured` | D7 | The `edges` cases at depth 2 (points whose coarse windows fall outside the 9x13 level, e.g. `(-4.5, 5)`, `(48.75, 32.5)`, `(10, 36.9)`) and the 3x3-window case (14 of 30 tracked) have the same accept set as OpenCV. In-tree pin: `test_lk_coarse_level_degeneracy_does_not_lose_point`. | #145 (closed) |
| Temporal-difference scale (#149) | [`diff = CV_DESCALE(J…, W_BITS1-5) - Iptr[x]`](https://github.com/opencv/opencv/blob/4.10.0/modules/video/src/lkpyramid.cpp#L638-L645) at the x32 intensity scale | `measured` | D4, D5, D7 | The 10 px shift converges to within 3e-4 px of the truth in both libraries (before the fix purecv stopped at 0.70 px for 1 px); identical statuses, positions within 4.1e-4 px. In-tree pin: `test_lk_single_newton_step_is_exact_on_quadratic_bowl`. | #149 (closed) |
| Oscillation half-step (#131) | [`abs(delta + prevDelta) < 0.01` → `nextPts -= delta * 0.5f`](https://github.com/opencv/opencv/blob/4.10.0/modules/video/src/lkpyramid.cpp#L674-L679) | `inspected` | D4, D7 | Same test on the `f32` step, same correction of the output (not of the internal corner). The measured runs of 30 iterations on textures pass through this branch wherever OpenCV does (final positions agree), but no case isolates it; pinned by the in-tree `test_lk_iterate_applies_oscillation_half_step`. | #131 (closed) |
| Border padding and sampling (#163; #154) | image levels padded by `winSize` with `pyrBorder` (reflect-101 by default, [S L767](https://github.com/opencv/opencv/blob/4.10.0/modules/video/src/lkpyramid.cpp#L767), [L789-L791](https://github.com/opencv/opencv/blob/4.10.0/modules/video/src/lkpyramid.cpp#L789-L791)); raw-path derivatives padded with `BORDER_CONSTANT` ([S L1389](https://github.com/opencv/opencv/blob/4.10.0/modules/video/src/lkpyramid.cpp#L1389)); bounds on the window corner `floor(pt - halfWin)` in `[-winSize, cols)` ([S L219-L230](https://github.com/opencv/opencv/blob/4.10.0/modules/video/src/lkpyramid.cpp#L219-L230), [L497-L503](https://github.com/opencv/opencv/blob/4.10.0/modules/video/src/lkpyramid.cpp#L497-L503)) | `measured` | D3, D7 | PR #164 (validated against OpenCV 5.0.0 at the time) re-measured with the 4.10.0 wheel: on the #163 setup, where the 21x21 window overhangs the 40x30 level-3 image for nearly every point, the accept sets are identical and the positions within 4.1e-4 px (rmse against the truth 2e-4 px in both). Points at and outside the edges (`(0, 0)`, `(-0.25, 0.5)`, `(48.75, 32.5)`, `(-20, -20)`, `(52.9, 10)` …) are lost or tracked exactly as OpenCV. The left/top blending of [#154](https://github.com/webarkit/purecv/issues/154) no longer exists: the samplers it described were replaced by `LkPatch::sample` in PR #164, and `(0, 0)`, `(0.25, 0.5)` and the even windows match; #154 can be closed. Tests `lk_grid163_matches_opencv`, `lk_edge_and_outside_points_match_opencv`, `lk_prebuilt_pyramid_borders_match_opencv`. | #163 (closed), [#154](https://github.com/webarkit/purecv/issues/154) (fixed by PR #164, still open) |
| Gate after the last iteration (#178 item 1) | [S L684-L698](https://github.com/opencv/opencv/blob/4.10.0/modules/video/src/lkpyramid.cpp#L684-L698): with `err` and without `OPTFLOW_LK_GET_MIN_EIGENVALS`, a final corner outside `[-winSize, cols)` loses the point | `measured` | D7 | Implemented at the branch base (PR #167, "recheck the final window before MAE"): with zero iterations from guesses `(-3, 4)` / `(-3.01, 4)`, `(10.99, 4)` / `(11, 4)` (and the `y` twins) on a 9x9 image with a 5x5 window, the status flips exactly where OpenCV's does with MAE and never with the min-eigenvalue output. Item 2 of #178 (a stricter overhang test) is a test-quality follow-up, unchanged by this audit. Test `lk_final_bounds_gate_matches_opencv`; in-tree `test_lk_opencv_final_bounds`. | [#178](https://github.com/webarkit/purecv/issues/178) (item 1 done) |
| Termination criteria | [normalisation](https://github.com/opencv/opencv/blob/4.10.0/modules/video/src/lkpyramid.cpp#L1364-L1372): count 30 without `COUNT`, else clamped to `0..100`; epsilon 0.01 without `EPS`, else clamped to `0..10`, then squared; [`delta.ddot(delta) <= epsilon`](https://github.com/opencv/opencv/blob/4.10.0/modules/video/src/lkpyramid.cpp#L671-L672) in `double` | `measured` | D1, D4, D7 | 13 forms (count only with a huge epsilon, zero and negative epsilon and count, epsilon only, caps at 100 / 10, one iteration) on a 5x5 window: positions and errors to 1 ulp; count 0 returns the initial estimate with its MAE. `TermType::{Count = 1, Eps = 2, Both = 3}` equals OpenCV's `COUNT`, `EPS`, `COUNT+EPS`. In-tree: `test_lk_opencv_criteria_limits`, `test_lk_opencv_termination_reference`. Test `lk_termination_criteria_match_opencv`. | |
| `OPTFLOW_USE_INITIAL_FLOW` semantics | [S L203-L208](https://github.com/opencv/opencv/blob/4.10.0/modules/video/src/lkpyramid.cpp#L203-L208): the initial `nextPts` scaled to the coarsest level; without the flag `nextPts` is overwritten with `prevPts` ([S L1271-L1275](https://github.com/opencv/opencv/blob/4.10.0/modules/video/src/lkpyramid.cpp#L1271-L1275)) | `measured` | D1, D7 | A good estimate (`+8 px` for the 10 px shift) and a bad one (`(-8, +3)` px: 30/30 still tracked, 9.7 px rmse in both libraries) give the same rows; the estimate is ignored without the flag (identical to the default rows). `initial_next_pts` is `Option` and `None` with the flag means the template positions (OpenCV: the caller's `nextPts`). | |
| `OPTFLOW_LK_GET_MIN_EIGENVALS` output | [`err[ptidx] = minEig`](https://github.com/opencv/opencv/blob/4.10.0/modules/video/src/lkpyramid.cpp#L477-L478) at every level (the level-0 value remains) | `measured` | D7 | Within 1 ulp on 93 points (rows above). | |
| Mean absolute error output | [S L700-L721](https://github.com/opencv/opencv/blob/4.10.0/modules/video/src/lkpyramid.cpp#L700-L721): `Σ\|J - I\| / (32 * w * cn * h)` on the x32 quantised patches | `measured` | D4, D5, D7 | Identical to 2e-6 on 5x5 windows (`asymmetric_borders`, `termination`, `edges`), within 0.013 on the 21x21 cases, where the 4e-4 px position difference moves a few quantised samples by one count. | |
| Lost-point outputs | `nextPts` keeps the last estimate (the scaled guess, or the position reached before the window left the image); `err` is `0` when the template window is outside ([S L221-L229](https://github.com/opencv/opencv/blob/4.10.0/modules/video/src/lkpyramid.cpp#L221-L229)), `minEig` with the flag, otherwise unwritten ("not defined", [H L144-L146](https://github.com/opencv/opencv/blob/4.10.0/modules/video/include/opencv2/video/tracking.hpp#L144-L146)) | `intentional` | D7 | purecv returns the template position and `f32::MAX` for every lost point (documented in the function's "Returns"). Measured what OpenCV returns (e.g. `(55.21, 34.28, 0)` for the point `(48.75, 32.5)`, `0.0` or stale memory for `err`); only `status` is compared. Rationale: a defined sentinel instead of an undefined error. | |

### Prebuilt pyramids (`calc_optical_flow_pyramid_lk_with_pyramids`)

| purecv item | OpenCV 4.10.0 counterpart (pinned link) | Status | Dims checked | Direction / notes | Issue |
|---|---|---|---|---|---|
| `calc_optical_flow_pyramid_lk_with_pyramids` | the `vector<Mat>` form of `calcOpticalFlowPyrLK` ([H L134-L135](https://github.com/opencv/opencv/blob/4.10.0/modules/video/include/opencv2/video/tracking.hpp#L134-L135), [S L1296-L1356](https://github.com/opencv/opencv/blob/4.10.0/modules/video/src/lkpyramid.cpp#L1296-L1356)): levels alone (`lvlStep` 1) or interleaved with derivatives (`lvlStep` 2, detected by `channels() * 2` and `CV_16S`); `maxLevel` capped to the supplied levels; the first pyramid's derivatives used verbatim ([S L1391-L1392](https://github.com/opencv/opencv/blob/4.10.0/modules/video/src/lkpyramid.cpp#L1391-L1392)), the second's ignored | `measured` | D1, D2, D3, D7 | **D1:** a separate function with the same parameters as the raw one (OpenCV overloads one `InputArray`); `dx`/`dy` both empty or one per level (OpenCV: interleaved `CV_16SC2` Mats); image and derivative borders are fields of the pyramid instead of padding. **D2:** purecv rejects non-integer, out-of-range or non-finite level values that OpenCV's types cannot hold; the padding check OpenCV applies to supplied pyramids is a row of its own below. **D7:** `cv2` cannot pass a `vector<Mat>`, so the entry point is measured in two steps: (a) `pyrBorder` semantics against raw `calcOpticalFlowPyrLK` on the 9x9 fixture padded by 8 with `copyMakeBorder` (reflect-101, constant, replicate, wrap, a different border on each image; MAE with zero iterations and the first Newton step with one; with and without stored derivatives): identical to 2e-6 (next row); (b) purecv's prebuilt path equals its raw path for the default borders (`test_lk_prebuilt_matches_raw_borders_small_images_and_flags`), and the raw path is measured above. The #167 fixtures from an out-of-tree generic build (`test_lk_opencv_quantized_patch_reference`: odd/even windows, depth 3, `Wrap`/`Reflect101` derivative borders; `test_lk_opencv_custom_i16_patch_storage`: full-range `i16` derivatives) pass at the branch base but their generator is not committed, so those aspects are `inspected`. | |
| `OpticalFlowPyramid::pyr_border` / `deriv_border` at tracking time | `pyrBorder` padding of image levels; `derivBorder` padding of stored derivatives ([S L804-L806](https://github.com/opencv/opencv/blob/4.10.0/modules/video/src/lkpyramid.cpp#L804-L806)); `BORDER_CONSTANT` for derivatives computed in the raw path ([S L1389](https://github.com/opencv/opencv/blob/4.10.0/modules/video/src/lkpyramid.cpp#L1389)) | `measured` | D3, D7 | Seven border pairs x {stored, computed derivatives}: MAE `[28.44, 53.56, 24.47]` (constant), `[21.24, 43.56, 16.55]` (replicate), `[66.4, 30.32, 62.66]` (wrap) and the one-step positions identical to OpenCV on explicitly padded images; a different border on each image is honoured independently (`prev_border` only changes the step's template, `next_border` the search). Test `lk_prebuilt_pyramid_borders_match_opencv`; in-tree `test_lk_opencv_asymmetric_borders_mae` / `_step` (same values, now from a committed script). | |
| Depth cap with prebuilt pyramids | [`if (levels1 < maxLevel) maxLevel = levels1`](https://github.com/opencv/opencv/blob/4.10.0/modules/video/src/lkpyramid.cpp#L1326-L1327), [same for `levels2`](https://github.com/opencv/opencv/blob/4.10.0/modules/video/src/lkpyramid.cpp#L1354-L1355) | `inspected` | D1 | `actual_levels = min(max_level + 1, prev levels, next levels)`, the same rule; not measurable with `cv2` (no `vector<Mat>`), pinned only by the in-tree purecv test `test_lk_prebuilt_caps_levels_and_preserves_builder_cutoff`. The raw path's cap by the builder's cutoff is measured in the entry-point row above (`max_level_5` ≡ `max_level_3` on 320x240 with a 21x21 window: level 4 would be 20x15). | |
| Padding check for supplied pyramids | [`CV_Assert(ofs.x >= winSize.width && …)`](https://github.com/opencv/opencv/blob/4.10.0/modules/video/src/lkpyramid.cpp#L1315-L1324) on level 1 of each supplied pyramid (and [L1343-L1352](https://github.com/opencv/opencv/blob/4.10.0/modules/video/src/lkpyramid.cpp#L1343-L1352)): the padding the builder wrote must be at least the tracking window | `diverges` | D2 | **keep?** OpenCV rejects a pyramid built for a smaller window (its padding would be read past); purecv has no padding and extends every level on the fly, so a 7x7 window on a pyramid built for 5x5 tracks (`Ok`) with the same values a sufficiently padded pyramid would give. Read from the source (not measurable with `cv2`). The check belongs with the prepadding of [#184](https://github.com/webarkit/purecv/issues/184), whose scope already names it ("validate that supplied padding is sufficient for the requested tracking window"); no new issue. Test `lk_prebuilt_rejects_windows_larger_than_the_pyramid_padding` (ignored). | [#184](https://github.com/webarkit/purecv/issues/184) |
| `calc_optical_flow_pyramid_lk_with_pyramids`: `Transparent` / `Isolated` pyramid borders | [`BORDER_TRANSPARENT`](https://github.com/opencv/opencv/blob/4.10.0/modules/video/src/lkpyramid.cpp#L764-L768) leaves the padding unwritten (level 0: `copyTo` into the centre; other levels: no `copyMakeBorder`, [S L789-L791](https://github.com/opencv/opencv/blob/4.10.0/modules/video/src/lkpyramid.cpp#L789-L791), [L804-L806](https://github.com/opencv/opencv/blob/4.10.0/modules/video/src/lkpyramid.cpp#L804-L806)); `BORDER_ISOLATED` is a flag OR-ed into the mode; the tracker reads the padding as it is | `intentional` | D3, D7 | OpenCV accepts both at build time (measured: `border_transparent` and `reflect101 \| ISOLATED` return a pyramid) and LK then reads uninitialised padding with `BORDER_TRANSPARENT`. purecv's `build_optical_flow_pyramid` also accepts and stores either value (the fields are plain data), but tracking rejects such a pyramid with `Err(InvalidInput)` ("borders must be Constant, Replicate, Reflect, Reflect101 or Wrap"): there is no padding memory, so a transparent border has no meaning, and `BorderTypes::Isolated` is a standalone variant, not a flag. Rationale: no undefined reads. | |

## Seeded rows from the plan

How each row seeded by Task B4 was resolved:

| Seed | Resolution |
|---|---|
| #130, #131, #138, #142, #144, #145, #149: `inspected`, cite the tests; `measured` once a script reproduces the numbers | #130, #138, #144, #145, #149 are `measured` (rows above: identical Scharr planes, 1-ulp eigenvalues, even and non-square windows, identical accept sets at depth 2, full convergence of the 10 px shift). #131 and #142 stay `inspected`: their branches are taken inside the measured runs but no case isolates them; the in-tree tests pin the expressions. |
| #154 (bilinear border blending), open | Fixed by PR #164 without being closed: the sampler it describes no longer exists, and the measured edge points and even windows match. Can be closed (comment proposed below). |
| #155 (`win_size` overflow panic), open | `inspected`: a robustness bug under the crate's no-panic rule, not a parity difference; OpenCV's `int` area overflows the same way. Stays open. |
| #163 (PR #164 validated against OpenCV 5.0.0): `inspected` until re-run in `cpp_ref/.venv` | Re-run with the 4.10.0 wheel on the same setup: `measured` (row "Border padding and sampling"). |
| #166 / PR #167 (precomputed pyramids): `missing`, or `measured` if merged | Merged at the branch base (`08882ec`): `measured` for the entry point's border semantics and the raw ≡ prebuilt equivalence; `inspected` for the aspects only the uncommitted C++ harness covers. The follow-ups #178 (item 2), #183 and #184 are linked from the rows. |
| #178 (final out-of-image check, stricter overhang test) | Item 1 is implemented at the branch base and `measured`; item 2 is a test-quality request outside the audit's scope (no library change). |
| Priority row: LK on the #163 setup in `cpp_ref/.venv` | Done: `video_lk.py` `grid163`, test `lk_grid163_matches_opencv`. |

## Missing in purecv (S4)

From the 4.10.0 doc groups [video_track](https://docs.opencv.org/4.10.0/dc/d6b/group__video__track.html)
and [video_motion](https://docs.opencv.org/4.10.0/de/de1/group__video__motion.html), and the
`tracking`, `optflow`, `bgsegm` and `cudaoptflow` modules of opencv_contrib. Roadmap gaps, not
bugs. Links point at the declarations in `tracking.hpp` / `background_segm.hpp` and the contrib
headers. The GPU module is outside the comparison ([#132](https://github.com/webarkit/purecv/issues/132)).

| purecv item | OpenCV 4.10.0 counterpart (pinned link) | Status | Dims checked | Direction / notes | Issue |
|---|---|---|---|---|---|
| — | [`calcOpticalFlowFarneback`](https://github.com/opencv/opencv/blob/4.10.0/modules/video/include/opencv2/video/tracking.hpp#L223-L225), [`FarnebackOpticalFlow`](https://github.com/opencv/opencv/blob/4.10.0/modules/video/include/opencv2/video/tracking.hpp#L475-L511), `OPTFLOW_FARNEBACK_GAUSSIAN` ([H L56-L59](https://github.com/opencv/opencv/blob/4.10.0/modules/video/include/opencv2/video/tracking.hpp#L56-L59)) | `missing` | | Dense flow (`optflowgf.cpp`). | |
| — | [`DISOpticalFlow`](https://github.com/opencv/opencv/blob/4.10.0/modules/video/include/opencv2/video/tracking.hpp#L590-L684) | `missing` | | Dense flow (`dis_flow.cpp`); uses `VariationalRefinement`. | |
| — | [`VariationalRefinement`](https://github.com/opencv/opencv/blob/4.10.0/modules/video/include/opencv2/video/tracking.hpp#L523-L576) | `missing` | | | |
| — | [`DenseOpticalFlow`](https://github.com/opencv/opencv/blob/4.10.0/modules/video/include/opencv2/video/tracking.hpp#L436-L449), [`SparseOpticalFlow`](https://github.com/opencv/opencv/blob/4.10.0/modules/video/include/opencv2/video/tracking.hpp#L453-L470) (`Algorithm` interfaces) | `missing` | | | |
| — | [`SparsePyrLKOpticalFlow`](https://github.com/opencv/opencv/blob/4.10.0/modules/video/include/opencv2/video/tracking.hpp#L694-L718) (getters/setters, `create`, `calc`) | `missing` | | The class form of `calcOpticalFlowPyrLK` ([S L1408-L1419](https://github.com/opencv/opencv/blob/4.10.0/modules/video/src/lkpyramid.cpp#L1408-L1419)); purecv has the free function only. | |
| — | the `tryReuseInputImage` argument of [`buildOpticalFlowPyramid`](https://github.com/opencv/opencv/blob/4.10.0/modules/video/include/opencv2/video/tracking.hpp#L121-L125) | `missing` | | An aliasing optimisation ([S L739-L753](https://github.com/opencv/opencv/blob/4.10.0/modules/video/src/lkpyramid.cpp#L739-L753)); no effect on values. | [#184](https://github.com/webarkit/purecv/issues/184) |
| — | [`KalmanFilter`](https://github.com/opencv/opencv/blob/4.10.0/modules/video/include/opencv2/video/tracking.hpp#L360-L410) | `missing` | | `kalman.cpp`; needs `core` `gemm`/`solve`. | |
| — | [`meanShift`](https://github.com/opencv/opencv/blob/4.10.0/modules/video/include/opencv2/video/tracking.hpp#L104), [`CamShift`](https://github.com/opencv/opencv/blob/4.10.0/modules/video/include/opencv2/video/tracking.hpp#L79-L80) | `missing` | | `camshift.cpp`; need `moments`. | |
| — | [`computeECC`](https://github.com/opencv/opencv/blob/4.10.0/modules/video/include/opencv2/video/tracking.hpp#L279), [`findTransformECC`](https://github.com/opencv/opencv/blob/4.10.0/modules/video/include/opencv2/video/tracking.hpp#L336-L346), the `MOTION_*` enum ([H L260-L266](https://github.com/opencv/opencv/blob/4.10.0/modules/video/include/opencv2/video/tracking.hpp#L260-L266)) | `missing` | | `ecc.cpp`; needs `warpAffine`/`warpPerspective` with `WARP_INVERSE_MAP` ([#237](https://github.com/webarkit/purecv/issues/237)). | |
| — | [`readOpticalFlow`](https://github.com/opencv/opencv/blob/4.10.0/modules/video/include/opencv2/video/tracking.hpp#L421), [`writeOpticalFlow`](https://github.com/opencv/opencv/blob/4.10.0/modules/video/include/opencv2/video/tracking.hpp#L431) (`.flo` files) | `missing` | | `optical_flow_io.cpp`; file I/O, `std` only. | |
| — | [`estimateRigidTransform`](https://github.com/opencv/opencv/blob/4.10.0/modules/video/include/opencv2/video/tracking.hpp#L258) (`CV_DEPRECATED`) | `missing` | | Superseded by `estimateAffine2D` / `estimateAffinePartial2D` (calib3d, B5). | |
| — | [`Tracker`](https://github.com/opencv/opencv/blob/4.10.0/modules/video/include/opencv2/video/tracking.hpp#L725-L750), [`TrackerMIL`](https://github.com/opencv/opencv/blob/4.10.0/modules/video/include/opencv2/video/tracking.hpp#L762-L790) | `missing` | | The one classical tracker of the main repository (`tracking/tracker_mil.cpp`, Haar features + online boosting). | |
| — | [`TrackerGOTURN`](https://github.com/opencv/opencv/blob/4.10.0/modules/video/include/opencv2/video/tracking.hpp#L809-L831), [`TrackerDaSiamRPN`](https://github.com/opencv/opencv/blob/4.10.0/modules/video/include/opencv2/video/tracking.hpp#L833-L862), [`TrackerNano`](https://github.com/opencv/opencv/blob/4.10.0/modules/video/include/opencv2/video/tracking.hpp#L872-L900), [`TrackerVit`](https://github.com/opencv/opencv/blob/4.10.0/modules/video/include/opencv2/video/tracking.hpp#L908-L937) | `missing` | | DNN-model trackers (need the `dnn` module and model files); out of a pure-Rust `video` port's scope. | |
| — | [`BackgroundSubtractor`](https://github.com/opencv/opencv/blob/4.10.0/modules/video/include/opencv2/video/background_segm.hpp#L60-L82), [`BackgroundSubtractorMOG2`](https://github.com/opencv/opencv/blob/4.10.0/modules/video/include/opencv2/video/background_segm.hpp#L90-L209) / [`createBackgroundSubtractorMOG2`](https://github.com/opencv/opencv/blob/4.10.0/modules/video/include/opencv2/video/background_segm.hpp#L220-L222) | `missing` | | `video_motion` group (`bgfg_gaussmix2.cpp`). | |
| — | [`BackgroundSubtractorKNN`](https://github.com/opencv/opencv/blob/4.10.0/modules/video/include/opencv2/video/background_segm.hpp#L229-L299) / [`createBackgroundSubtractorKNN`](https://github.com/opencv/opencv/blob/4.10.0/modules/video/include/opencv2/video/background_segm.hpp#L309-L311) | `missing` | | `bgfg_KNN.cpp`. | |
| — | opencv_contrib [`tracking::TrackerCSRT`](https://github.com/opencv/opencv_contrib/blob/4.10.0/modules/tracking/include/opencv2/tracking.hpp#L34-L87), [`tracking::TrackerKCF`](https://github.com/opencv/opencv_contrib/blob/4.10.0/modules/tracking/include/opencv2/tracking.hpp#L98-L149) | `missing` | | Contrib (`tracking` module); classical correlation-filter trackers, in the wheel (`cv2.TrackerCSRT_create`, `cv2.TrackerKCF_create`). | |
| — | opencv_contrib `tracking::legacy::{`[`TrackerMIL`](https://github.com/opencv/opencv_contrib/blob/4.10.0/modules/tracking/include/opencv2/tracking/tracking_legacy.hpp#L123)`, `[`TrackerBoosting`](https://github.com/opencv/opencv_contrib/blob/4.10.0/modules/tracking/include/opencv2/tracking/tracking_legacy.hpp#L148)`, `[`TrackerMedianFlow`](https://github.com/opencv/opencv_contrib/blob/4.10.0/modules/tracking/include/opencv2/tracking/tracking_legacy.hpp#L190)`, `[`TrackerTLD`](https://github.com/opencv/opencv_contrib/blob/4.10.0/modules/tracking/include/opencv2/tracking/tracking_legacy.hpp#L232)`, `[`TrackerKCF`](https://github.com/opencv/opencv_contrib/blob/4.10.0/modules/tracking/include/opencv2/tracking/tracking_legacy.hpp#L260)`, `[`TrackerGOTURN`](https://github.com/opencv/opencv_contrib/blob/4.10.0/modules/tracking/include/opencv2/tracking/tracking_legacy.hpp#L305)`, `[`TrackerMOSSE`](https://github.com/opencv/opencv_contrib/blob/4.10.0/modules/tracking/include/opencv2/tracking/tracking_legacy.hpp#L334)`, `[`MultiTracker`](https://github.com/opencv/opencv_contrib/blob/4.10.0/modules/tracking/include/opencv2/tracking/tracking_legacy.hpp#L351)`, `[`TrackerCSRT`](https://github.com/opencv/opencv_contrib/blob/4.10.0/modules/tracking/include/opencv2/tracking/tracking_legacy.hpp#L500)`}`, [`upgradeTrackingAPI`](https://github.com/opencv/opencv_contrib/blob/4.10.0/modules/tracking/include/opencv2/tracking/tracking_legacy.hpp#L529) | `missing` | | Contrib, legacy API (`MedianFlow` is LK-based and would reuse `calc_optical_flow_pyramid_lk`). | |
| — | opencv_contrib `optflow::{`[`calcOpticalFlowSF`](https://github.com/opencv/opencv_contrib/blob/4.10.0/modules/optflow/include/opencv2/optflow.hpp#L81-L83)`, `[`calcOpticalFlowSparseToDense`](https://github.com/opencv/opencv_contrib/blob/4.10.0/modules/optflow/include/opencv2/optflow.hpp#L135-L137)`, `[`createOptFlow_DeepFlow`](https://github.com/opencv/opencv_contrib/blob/4.10.0/modules/optflow/include/opencv2/optflow.hpp#L165)`, `[`createOptFlow_SimpleFlow`](https://github.com/opencv/opencv_contrib/blob/4.10.0/modules/optflow/include/opencv2/optflow.hpp#L168)`, `[`createOptFlow_SparseToDense`](https://github.com/opencv/opencv_contrib/blob/4.10.0/modules/optflow/include/opencv2/optflow.hpp#L174)`, `[`DualTVL1OpticalFlow`](https://github.com/opencv/opencv_contrib/blob/4.10.0/modules/optflow/include/opencv2/optflow.hpp#L218-L300)`}` | `missing` | | Contrib (`optflow` module). `calcOpticalFlowSparseToDense` is LK-based. | |
| — | opencv_contrib `optflow::{`[`OpticalFlowPCAFlow`](https://github.com/opencv/opencv_contrib/blob/4.10.0/modules/optflow/include/opencv2/optflow/pcaflow.hpp#L94-L142)`, `[`DenseRLOFOpticalFlow`](https://github.com/opencv/opencv_contrib/blob/4.10.0/modules/optflow/include/opencv2/optflow/rlofflow.hpp#L233)`, `[`SparseRLOFOpticalFlow`](https://github.com/opencv/opencv_contrib/blob/4.10.0/modules/optflow/include/opencv2/optflow/rlofflow.hpp#L412)`, `[`calcOpticalFlowDenseRLOF`](https://github.com/opencv/opencv_contrib/blob/4.10.0/modules/optflow/include/opencv2/optflow/rlofflow.hpp#L501)`, `[`calcOpticalFlowSparseRLOF`](https://github.com/opencv/opencv_contrib/blob/4.10.0/modules/optflow/include/opencv2/optflow/rlofflow.hpp#L538)`, `[`GPCTree`](https://github.com/opencv/opencv_contrib/blob/4.10.0/modules/optflow/include/opencv2/optflow/sparse_matching_gpc.hpp#L154)`}` | `missing` | | Contrib (`optflow` module); `SparseRLOF` is a robust sparse LK variant. | |
| — | opencv_contrib `bgsegm::{`[`BackgroundSubtractorMOG`](https://github.com/opencv/opencv_contrib/blob/4.10.0/modules/bgsegm/include/opencv2/bgsegm.hpp#L62-L86)`, `[`BackgroundSubtractorGMG`](https://github.com/opencv/opencv_contrib/blob/4.10.0/modules/bgsegm/include/opencv2/bgsegm.hpp#L99-L185)`, `[`BackgroundSubtractorCNT`](https://github.com/opencv/opencv_contrib/blob/4.10.0/modules/bgsegm/include/opencv2/bgsegm.hpp#L195-L239)`, `[`BackgroundSubtractorGSOC`](https://github.com/opencv/opencv_contrib/blob/4.10.0/modules/bgsegm/include/opencv2/bgsegm.hpp#L254-L263)`, `[`BackgroundSubtractorLSBP`](https://github.com/opencv/opencv_contrib/blob/4.10.0/modules/bgsegm/include/opencv2/bgsegm.hpp#L265-L274)`, `[`SyntheticSequenceGenerator`](https://github.com/opencv/opencv_contrib/blob/4.10.0/modules/bgsegm/include/opencv2/bgsegm.hpp#L329-L372)`}` | `missing` | | Contrib (`bgsegm` module). | |
| — | opencv_contrib `cuda::{`[`SparsePyrLKOpticalFlow`](https://github.com/opencv/opencv_contrib/blob/4.10.0/modules/cudaoptflow/include/opencv2/cudaoptflow.hpp#L203-L223)`, `[`DensePyrLKOpticalFlow`](https://github.com/opencv/opencv_contrib/blob/4.10.0/modules/cudaoptflow/include/opencv2/cudaoptflow.hpp#L230-L250)`, `[`BroxOpticalFlow`](https://github.com/opencv/opencv_contrib/blob/4.10.0/modules/cudaoptflow/include/opencv2/cudaoptflow.hpp#L155-L186)`, `[`FarnebackOpticalFlow`](https://github.com/opencv/opencv_contrib/blob/4.10.0/modules/cudaoptflow/include/opencv2/cudaoptflow.hpp#L258-L294)`, `[`OpticalFlowDual_TVL1`](https://github.com/opencv/opencv_contrib/blob/4.10.0/modules/cudaoptflow/include/opencv2/cudaoptflow.hpp#L305-L386)`, `[`NvidiaOpticalFlow_1_0`](https://github.com/opencv/opencv_contrib/blob/4.10.0/modules/cudaoptflow/include/opencv2/cudaoptflow.hpp#L400-L455)`, `[`NvidiaOpticalFlow_2_0`](https://github.com/opencv/opencv_contrib/blob/4.10.0/modules/cudaoptflow/include/opencv2/cudaoptflow.hpp#L465-L588)`}` | `missing` | | GPU; not part of the comparison. | [#132](https://github.com/webarkit/purecv/issues/132) |

## Proposed issues

New `diverges` rows, one issue each, titled per S5. Approved by the maintainer and filed on
2026-10-21 (#256, #257) in milestone 8 ("Parity audit: purecv ↔ OpenCV 4.10.0"), labelled
`video-module` and `bug`. The fourth `diverges` row (padding check) is tracked by the existing
[#184](https://github.com/webarkit/purecv/issues/184).

| Proposal | Issue | Title | Rows | Direction | Depends on |
|---|---|---|---|---|---|
| P1 | [#256](https://github.com/webarkit/purecv/issues/256) | `video: build_optical_flow_pyramid and calc_optical_flow_pyramid_lk differ from OpenCV 4.10.0: multi-channel images rejected instead of tracked per channel` | `build_optical_flow_pyramid` (multi-channel), `calc_optical_flow_pyramid_lk` (multi-channel) | keep? (OpenCV samples `winSize.width * cn` values per row and divides the MAE by `cn`; a gray conversion would be the cheaper alternative, cf. [#252](https://github.com/webarkit/purecv/issues/252)) | [#183](https://github.com/webarkit/purecv/issues/183) if the typed storage lands first |
| P2 | [#257](https://github.com/webarkit/purecv/issues/257) | `video: calc_optical_flow_pyramid_lk differs from OpenCV 4.10.0: negative max_level clamped to 0 instead of rejected` | `calc_optical_flow_pyramid_lk` (negative `max_level`) | → OpenCV (`Err(InvalidInput)`, like the window check; `calc_optical_flow_pyramid_lk_with_pyramids` clamps the same way) | — |

Comments posted on existing issues (2026-10-21):

- [#154](https://github.com/webarkit/purecv/issues/154): fixed by PR #164 (the sampler was
  replaced); the edge points and even windows measured here match OpenCV 4.10.0; can be
  closed.
- [#178](https://github.com/webarkit/purecv/issues/178): item 1 (final out-of-image check)
  is implemented by PR #167 and measured against the wheel (`lk_final_bounds_gate_matches_opencv`);
  item 2 remains.
- [#163](https://github.com/webarkit/purecv/issues/163) (closed): the fix is now validated
  against 4.10.0, not only 5.0.0 (identical accept sets, 4.1e-4 px).
- [#166](https://github.com/webarkit/purecv/issues/166) (closed) / #167: the Python
  references of the in-tree tests are now reproducible from `scripts/opencv_ref/video_lk.py`;
  the C++ `vector<Mat>` fixtures still have no committed generator (a `chore` follow-up, not a
  parity finding).

## Notes

- **Counting.** The summary counts one row per table line (52 rows). Grouped rows (each
  `missing` line, the contrib lines) count once.
- **D6.** Neither `parallel` nor `simd` changed any result (byte-identical dumps for all four
  builds). `wasm32` was not run.
- **Tolerances.** `tests/parity_video.rs` enforces 2e-5 px / 1e-5 + 1e-6·|err| for windows up to
  7 px wide, 2e-3 px / 0.02 (absolute) for wider ones, and 1e-6 + 1e-6·|err| for the minimum
  eigenvalue, each above the largest difference measured in its group (D7 record).
- **Panics.** One known: #155 (absurd `win_size`); none found on the measured inputs.
- **IPP / optimisations.** IPP is not involved in LK. The wheel's compiled-in `CV_SIMD128`
  block is the only optimisation that changes values (D7 record), and only for windows at
  least 8 px wide; `setUseOptimized(False)` and `setNumThreads(1)` leave the output
  bit-identical.
- **Not measurable with `cv2`.** The `vector<Mat>` input form, the padding bytes of a built
  pyramid, `winSize.area()` overflow (D7 record).
- **Cross-module dependencies.** The pyramid rows depend on `imgproc::pyr_down` (u8 path
  identical; the float rounding of [#238](https://github.com/webarkit/purecv/issues/238) is
  what makes the `f32` levels equal OpenCV's `CV_8U` ones, so a fix of #238 must keep an
  integer path for LK) and on `imgproc::scharr` (identical; the `ksize`/`normalize` gaps of
  [#232](https://github.com/webarkit/purecv/issues/232) do not affect LK). Nothing here blurs
  u8 frames ([#229](https://github.com/webarkit/purecv/issues/229) is not involved), uses
  `fastAtan2` ([#250](https://github.com/webarkit/purecv/issues/250)) or `cv::RNG`
  ([#225](https://github.com/webarkit/purecv/issues/225)). The shared `saturate_cast`
  ([#189](https://github.com/webarkit/purecv/issues/189)) is not needed: LK's narrowing
  conversions (`(short)ival`, `CV_DESCALE`) are reproduced with explicit integer arithmetic.
- **Inputs.** The scripts and the tests build every input from the same integer generators
  (xorshift32 cells with a 3x3 box mean, the 32-bit LCG rectangles, a quarter-pixel bilinear
  shift in integers, `(17x + 13y + 3xy) mod 251`), so no image file is committed.

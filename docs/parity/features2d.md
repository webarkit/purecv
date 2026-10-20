# Parity audit: `features2d` ↔ OpenCV 4.10.0

Task B3 of [#172](https://github.com/webarkit/purecv/issues/172), sub-issue
[#175](https://github.com/webarkit/purecv/issues/175). Method, statuses (`measured`,
`inspected`, `diverges`, `intentional`, `purecv-only`, `missing`) and dimensions D1–D7:
[`README.md`](README.md). Inventory taken at the branch base `0e310c3` (`parity-audit` after
the imgproc audit, PR #228): every `pub` item of `src/features2d`, the public inherent methods
of `FastFeatureDetector`, `Orb` and `BFMatcher`, the re-exports of `src/features2d/mod.rs`, and
the methods of `KeyPoint` and `DMatch`, which live in `src/core` since PR #207 and are
re-exported here. The placement of those two types is the `core` audit's row
([`core.md`](core.md#seeded-rows-from-the-plan)); their behaviour is audited below, as that
row says.

## D7: reference and measurement record

- **Sources read:** `cpp_ref/src/opencv` at `71d3237a093b60a27601c20e9ee6c3e52154e8b1`
  (`verify opencv @ … OK`), `cpp_ref/src/opencv_contrib` at
  `1ed3dd2c53888e3289afdb22ec4e9ebbff3dba87` (`verify opencv_contrib @ … OK`), bootstrapped on
  2026-10-20. The generic CPU path was read: `FAST_t<N>` (scalar loop), `ORB_Impl`,
  `HarrisResponses`, `ICAngles`, `computeOrbDescriptors`, `BFMatcher::knnMatchImpl` with
  `cv::batchDistance`, `drawKeypoints`/`drawMatches`, `KeyPointsFilter`, `cv::KeyPoint`. The
  OpenCL (`ocl_FAST`, `ocl_HarrisResponses`, …), `fast_dense` HAL and OpenVX back ends, and the
  `CV_SIMD128`/AVX2 blocks of `FAST_t<16>`, were skipped; where they report something else
  (FAST's response with nonmax off), the row says so.
- **Measurements (Python):** `cpp_ref/.venv`, `pip freeze` = `numpy==2.4.6`,
  `opencv-contrib-python==4.10.0.84`; `cv2.__version__` = `4.10.0`; `cv2.ipp.setUseIPP(False)`
  → `useIPP()` = False; `useOptimized()` = True; Python 3.12.2; Windows 11 10.0.26300 x86-64;
  `getBuildInformation()`: `Version control: 4.10.0`, `Baseline: SSE SSE2 SSE3`,
  `Dispatched code generation: SSE4_1 SSE4_2 FP16 AVX AVX2`, `Intel IPP: 2021.11.0`
  (disabled at run time). Nothing in this module runs through IPP.
- **opencv.js:** not needed. `cv2` binds `FastFeatureDetector` with all three types (which
  opencv.js does not), `ORB` with every parameter, `BFMatcher`, `drawKeypoints`,
  `drawMatches`, `KeyPoint.overlap` and `KeyPoint.convert`. It does **not** bind the free
  function `cv::FAST` (`CV_EXPORTS`, measured through the detector, which calls it),
  `KeyPointsFilter` (`retainBest` and friends are `inspected` only) or `DMatch::operator<`.
  The ORB fixture `src/features2d/orb_opencv_ref.rs`, generated from a self-built opencv.js
  4.10.0, was re-measured with the wheel: the 40 descriptors are **byte-identical**
  (`features2d_orb.py`, `compute_fixture.opencv_js_fixture`), so the fixture is confirmed
  against the pinned reference and the `gen_orb_descriptors.js` follow-up of #175 (pinned
  package, `Version control` line) is bookkeeping, not a finding.
- **Scripts (evidence rule):** `scripts/opencv_ref/features2d_fast.py`, `features2d_orb.py`,
  `features2d_matcher.py`, `features2d_draw.py`, `features2d_keypoint.py` (helpers
  `_core_common.py`, `_features2d_common.py`). Every input is built from integer formulas
  (a 32-bit LCG for noise and rectangle textures, two squares, the fixture's hash texture);
  keypoint lists are reported as a count, an FNV-1a digest of their field bytes, a digest of
  the positions only, and a prefix, so the Rust tests can pin lists of a thousand keypoints.
- **Tests:** `tests/parity_features2d.rs`. 12 passing tests pin the rows (or the parts of
  rows) that match; 17 tests marked `#[ignore = "diverges from OpenCV 4.10.0, …"]` are the
  executable form of the `diverges` rows and all fail when run with `--ignored`. Expected
  values are constants generated from the scripts' JSON (`mod ocv`).
- **D6:** the measured cases were run through purecv with four builds: default (`std`,
  `parallel`), `--features simd`, `--no-default-features --features std` and
  `--no-default-features --features std,simd`, all `--release`. **Every keypoint, descriptor,
  match and drawn pixel was identical across the four builds** (FAST has no `simd` code,
  [#203](https://github.com/webarkit/purecv/issues/203); ORB's `pulp` kernels are
  bit-identical to the scalar path). `wasm32` was not run
  ([#204](https://github.com/webarkit/purecv/issues/204) covers its threshold clamp).
- **Row grouping:** getters and setters share their constructor's row; the three
  `Orb::{detect, compute, detect_and_compute}` entry points have one row per behaviour
  (selection, pyramid, size, provided keypoints, descriptors, parameter values) rather than
  one row per method, because each behaviour has its own issue.

## Summary

| Status | Rows |
|---|---|
| `measured` | 8 |
| `inspected` | 11 |
| `diverges` | 20 |
| `intentional` | 0 |
| `purecv-only` | 3 |
| `missing` | 21 |

Headline findings, in order of impact (details in the rows, issues in
[Proposed issues](#proposed-issues)):

1. **The ORB path matches OpenCV only where the inputs are the same pixels.** FAST
   `Type9_16` with non-maximum suppression (ORB's detector) is bit-identical on every image
   and threshold; the per-level feature budget, the orientation moments and the BFMatcher
   are OpenCV's. Everything downstream of the pyramid and the Harris scoring is not: at
   octave 0, where the pyramid plays no part, 92 of 109 selected keypoints coincide on the
   rectangle texture and 8 of 109 on noise (#158); at octaves ≥ 1 the counts below quota
   differ by one or two per level (#193).
2. **Descriptors differ because of the u8 Gaussian blur, not because of BRIEF.** purecv's
   `gaussian_blur` differs from OpenCV's bit-exact 7×7/σ 2 blur by 1 on 50 % of the pixels
   (#229, #157). On the hash-textured fixture that costs 1.2 bits per descriptor (max 3); on
   a flat rectangle texture, where blurred neighbours are nearly equal, 19 bits (max 42).
   Re-sampling OpenCV's own blurred level with purecv's rules recovers OpenCV's
   descriptors to 0.84 bits.
3. **The orientation is not `fastAtan2`.** purecv's exact `atan2` differs from OpenCV's
   polynomial by up to 0.009° on 109 keypoints (OpenCV documents 0.3°); `fastAtan2` of
   purecv's direction reproduces OpenCV's angle on 109/109. #159 recorded this item as
   "within 1 ulp"; it is not (proposed issue P1).
4. **FAST outside ORB's configuration.** With nonmax off the responses are the score instead
   of 0 and threshold 0 keeps fewer corners (#201); `Type7_12` and `Type5_8` differ on every
   image because OpenCV's pre-test rejects valid arcs and purecv's `Type5_8` scans a 1-pixel
   border (#202): on the squares image OpenCV's `Type7_12` finds the six corners with nonmax
   and purecv none.
5. **Drawing is a different renderer.** OpenCV draws anti-aliased sub-pixel circles and
   lines (`LINE_AA`, 4 fractional bits), radius 3 unless `DRAW_RICH_KEYPOINTS`, into a BGR
   copy; purecv draws Bresenham/midpoint primitives of radius `size / 2` into an image with
   the input's channel count, silently skips invalid match indices and rejects mixed
   channel counts (P4, P5).
6. **Smaller differences:** `knn_match(k = 0)` returns empty lists where OpenCV asserts
   (P2); multi-channel input is rejected where OpenCV converts BGR→gray (P3);
   `KeyPoint::retain_best` drops cutoff ties (#196); `wta_k` 3/4, `patch_size != 31` and
   `first_level > 0` are rejected (#198, #199, #200); `compute()` with caller-supplied
   keypoints has no border filter and rejects `octave >= nlevels` (#195); `KeyPoint.size`
   stays 7 (#194).

What matches: FAST `Type9_16` with nonmax (all thresholds, all images, all builds), the
FAST scan range, FAST nonmax-off positions at thresholds ≥ 10, the ORB per-level budget, the
intensity-centroid moments, `detect` + `compute` = `detect_and_compute`, ORB parameter
validation, `BFMatcher` (Hamming and L2, ties, cross-check, kNN order, every error case),
the `drawMatches` canvas, `KeyPoint::overlap` and the `convert` helpers, `BIT_PATTERN_31`.

Link bases used below: `H` = `https://github.com/opencv/opencv/blob/4.10.0/modules/features2d/include/opencv2/features2d.hpp`,
`S` = `https://github.com/opencv/opencv/blob/4.10.0/modules/features2d/src` (written out in full
in every link).

## Modules, enums and types

| purecv item | OpenCV 4.10.0 counterpart (pinned link) | Status | Dims checked | Direction / notes | Issue |
|---|---|---|---|---|---|
| `features2d::{bit_pattern_31, draw, fast, matcher, orb}` (public submodules), `features2d::keypoint` (re-export of `core::keypoint`) | — | `purecv-only` | | Module layout; every public item is also re-exported at `purecv::features2d`. OpenCV's `Feature2D` base class ([H L135-L223](https://github.com/opencv/opencv/blob/4.10.0/modules/features2d/include/opencv2/features2d.hpp#L135-L223)) has no counterpart: `FastFeatureDetector` and `Orb` are unrelated structs. | |
| `FastType` | [`FastFeatureDetector::DetectorType`](https://github.com/opencv/opencv/blob/4.10.0/modules/features2d/include/opencv2/features2d.hpp#L576-L579) | `inspected` | D1 | Same three variants; the implicit discriminants `Type5_8 = 0`, `Type7_12 = 1`, `Type9_16 = 2` equal OpenCV's (`TYPE_5_8 = 0`, `TYPE_7_12 = 1`, `TYPE_9_16 = 2`, measured as `cv2.FAST_FEATURE_DETECTOR_TYPE_*`). | |
| `ScoreType` | [`ORB::ScoreType`](https://github.com/opencv/opencv/blob/4.10.0/modules/features2d/include/opencv2/features2d.hpp#L426) | `inspected` | D1 | `Harris = 0`, `Fast = 1` as `HARRIS_SCORE = 0`, `FAST_SCORE = 1`. | |
| `NormType` | [`cv::NormTypes`](https://github.com/opencv/opencv/blob/4.10.0/modules/core/include/opencv2/core/base.hpp#L168-L204) | `inspected` | D1 | Two of OpenCV's six: `NormHamming`, `NormL2` (implicit discriminants 0, 1; OpenCV 6, 4; nothing converts them). `NORM_L1`, `NORM_L2SQR`, `NORM_HAMMING2` and `NORM_INF` are `missing` ([Missing](#missing-in-purecv-s4); `NORM_HAMMING2` is [#198](https://github.com/webarkit/purecv/issues/198)). `core::types::NormTypes` has all the values but `BFMatcher` does not take it. | |
| `DescriptorMatcher` (trait) | [`cv::DescriptorMatcher`](https://github.com/opencv/opencv/blob/4.10.0/modules/features2d/include/opencv2/features2d.hpp#L994-L1244) | `inspected` | D1 | `match_descriptors` ↔ [`match`](https://github.com/opencv/opencv/blob/4.10.0/modules/features2d/include/opencv2/features2d.hpp#L1060-L1061), `knn_match` ↔ [`knnMatch`](https://github.com/opencv/opencv/blob/4.10.0/modules/features2d/include/opencv2/features2d.hpp#L1081-L1083), both with the train set as an argument. No `mask`, `compactResult`, `radiusMatch`, `add`/`train` collection or multi-image `imgIdx` ([Missing](#missing-in-purecv-s4)). `DMatch::train_img_idx` is always 0, as OpenCV's `imgIdx` is for a single train set. | |
| `DescriptorDistance` (trait, `u8` and `f32`) | the `batchDistance` kernels [`batchDistHamming`](https://github.com/opencv/opencv/blob/4.10.0/modules/core/src/batch_distance.cpp#L103-L123), [`batchDistL2_8u32f`](https://github.com/opencv/opencv/blob/4.10.0/modules/core/src/batch_distance.cpp#L171-L187), [`batchDistL2_32f`](https://github.com/opencv/opencv/blob/4.10.0/modules/core/src/batch_distance.cpp#L189-L205) | `measured` | D1, D2, D7 | Hamming = popcount of the XOR (6×8 distance matrix identical); L2 on u8 = `sqrt` of the f32 sum of squared differences, as `batchDistL2_8u32f` (identical to f32 rounding, 4.8e-5 from the f64 value); L2 on f32 identical. Hamming on `f32` is `Err`, as OpenCV's `CV_Assert` ([batch_distance.cpp L282](https://github.com/opencv/opencv/blob/4.10.0/modules/core/src/batch_distance.cpp#L282)). Test `bfmatcher_hamming_matches_opencv`. | |
| `BIT_PATTERN_31` | [`bit_pattern_31_`](https://github.com/opencv/opencv/blob/4.10.0/modules/features2d/src/orb.cpp#L380-L639) | `inspected` | D1, D7 | All 1024 values identical (compared mechanically while writing `features2d_orb.py`). `pub static [i8; 1024]` vs a private `static int[256*4]`. | |

## FAST (`fast.rs`)

| purecv item | OpenCV 4.10.0 counterpart (pinned link) | Status | Dims checked | Direction / notes | Issue |
|---|---|---|---|---|---|
| `FastFeatureDetector`, `FastFeatureDetector::{new, default}`, `{threshold, set_threshold, nonmax_suppression, set_nonmax_suppression, detector_type, set_detector_type}` | [`FastFeatureDetector::create`](https://github.com/opencv/opencv/blob/4.10.0/modules/features2d/include/opencv2/features2d.hpp#L586-L597) | `inspected` | D1 | Defaults identical: threshold 10, nonmax on, `TYPE_9_16` (`cv2.FastFeatureDetector_create()` getters). `threshold: u8` where OpenCV takes an `int` and clamps it to `[0, 255]` ([fast.cpp L78](https://github.com/opencv/opencv/blob/4.10.0/modules/features2d/src/fast.cpp#L78)); the WASM wrapper's `as u8` wrap is [#204](https://github.com/webarkit/purecv/issues/204). No `mask` argument ([`runByPixelsMask`](https://github.com/opencv/opencv/blob/4.10.0/modules/features2d/src/fast.cpp#L583), [Missing](#missing-in-purecv-s4)). | |
| `FastFeatureDetector::detect`, `Type9_16`, nonmax on | [`FAST_t<16>`](https://github.com/opencv/opencv/blob/4.10.0/modules/features2d/src/fast.cpp#L57-L292), [`cornerScore<16>`](https://github.com/opencv/opencv/blob/4.10.0/modules/features2d/src/fast_score.cpp#L119-L210) | `measured` | D1, D2, D6, D7 | **Identical**: count, positions, order, `size 7`, `angle -1`, `response` (the score), `octave 0`, `class_id -1` on three images (64×80 noise, 48×64 rectangles, 40×48 squares) at thresholds 0, 10, 20, 40 (466/458/440/344, 179/76/68/55, 0/0/0/0 keypoints). The scan range `3 ..= dim - 4` and the 7×7 minimum image agree. D6: same in all four builds. Tests `fast_type9_16_nonmax_on_matches_opencv`, `fast_scan_range_matches_opencv_for_type9_16_and_7_12`. | |
| `FastFeatureDetector::detect`, `Type9_16`, nonmax off | [L223-L265](https://github.com/opencv/opencv/blob/4.10.0/modules/features2d/src/fast.cpp#L223-L265) (candidates), [L279-L290](https://github.com/opencv/opencv/blob/4.10.0/modules/features2d/src/fast.cpp#L279-L290) (emission) | `diverges` | D1, D7 | → OpenCV. At thresholds 10, 20 and 40 the positions and their order are identical (1380/1087/579 and 394/347/256 keypoints, position digests equal, test `fast_type9_16_nonmax_off_positions_match_opencv_above_threshold_0`), but OpenCV never computes the score with nonmax off ([L235-L236](https://github.com/opencv/opencv/blob/4.10.0/modules/features2d/src/fast.cpp#L235-L236)) and reports `response = 0` for every keypoint; purecv reports the score. The OpenCL path reports `1.f` and the `fast_dense` HAL `0.f` ([L339](https://github.com/opencv/opencv/blob/4.10.0/modules/features2d/src/fast.cpp#L339), [L487](https://github.com/opencv/opencv/blob/4.10.0/modules/features2d/src/fast.cpp#L487)), so the back ends disagree. At threshold 0 OpenCV also keeps corners whose score is 0 (1718 vs 1676 on noise, 1162 vs 943 on the rectangles). Test `fast_nonmax_off_response_and_threshold_0_match_opencv`. | [#201](https://github.com/webarkit/purecv/issues/201) |
| `FastFeatureDetector::detect`, `Type7_12` | [`FAST_t<12>`](https://github.com/opencv/opencv/blob/4.10.0/modules/features2d/src/fast.cpp#L57-L292) with the pre-test [L205-L221](https://github.com/opencv/opencv/blob/4.10.0/modules/features2d/src/fast.cpp#L205-L221) and [`makeOffsets`](https://github.com/opencv/opencv/blob/4.10.0/modules/features2d/src/fast_score.cpp#L50-L82) | `diverges` | D1, D7 | keep? (decision in #202). OpenCV's scalar pre-test reads `pixel[0..15]` in opposite pairs for every pattern size, and `makeOffsets` wraps the entries past 12, so only 7-arcs starting at offsets 0, 9, 10 and 11 always pass; purecv tests every arc. Measured: a 9×9 ring with exactly one 7-of-12 arc is a corner in OpenCV for those four starts only, in purecv for all twelve; on the noise image nonmax on gives 450/445/431/366 vs 452/449/444/397 and nonmax off 1864/1530/1232/728 vs 2025/1724/1402/883; on the squares OpenCV finds the six corners with nonmax (response 169/189) and purecv none, because purecv's complete candidate set makes every edge pixel tie with the corner. Scan range identical. Test `fast_type7_12_and_5_8_match_opencv`. | [#202](https://github.com/webarkit/purecv/issues/202) |
| `FastFeatureDetector::detect`, `Type5_8` | [`FAST_t<8>`](https://github.com/opencv/opencv/blob/4.10.0/modules/features2d/src/fast.cpp#L57-L292), rows/columns [L99-L109](https://github.com/opencv/opencv/blob/4.10.0/modules/features2d/src/fast.cpp#L99-L109), [L202](https://github.com/opencv/opencv/blob/4.10.0/modules/features2d/src/fast.cpp#L202) | `diverges` | D1, D7 | → OpenCV for the border, keep? for the test. (c) OpenCV scans `3 ..= dim - 4` for every pattern; purecv scans from pixel 1 (`radius = 1`), so it reports corners at x = 1, 2 and finds corners in 5×5 and 6×6 images where OpenCV finds none. (d) OpenCV's pre-test collapses to "all 8 neighbours brighter or all darker": no 5-of-8 ring is a corner (0 of 8 starts), only the 8-of-8 ring is; on the rectangle texture OpenCV finds 0 keypoints at thresholds 10–40 and purecv 80/75/58 (nonmax on). Test `fast_type7_12_and_5_8_match_opencv`. | [#202](https://github.com/webarkit/purecv/issues/202) |
| `FastFeatureDetector::detect`, `Orb::{detect, compute, detect_and_compute}` (non-`8UC1` input) | [`FastFeatureDetector_Impl::detect`](https://github.com/opencv/opencv/blob/4.10.0/modules/features2d/src/fast.cpp#L563-L584), [`ORB_Impl::detectAndCompute`](https://github.com/opencv/opencv/blob/4.10.0/modules/features2d/src/orb.cpp#L1039-L1041) | `diverges` | D2, D7 | keep? OpenCV converts any non-`CV_8U` image with `COLOR_BGR2GRAY` (a 3-channel u8 image gives the gray result; f32 and u16 fail inside `cvtColor`); purecv's `Matrix<u8>` API returns `InvalidInput` for `channels != 1`. A Rust API may prefer the explicit `cvt_color`. Test `fast_and_orb_accept_multichannel_input_like_opencv`. | P3 |

## ORB (`orb.rs`, `bit_pattern_31.rs`)

| purecv item | OpenCV 4.10.0 counterpart (pinned link) | Status | Dims checked | Direction / notes | Issue |
|---|---|---|---|---|---|
| `Orb`, `Orb::{new, default}`, the nine getter/setter pairs | [`ORB::create`](https://github.com/opencv/opencv/blob/4.10.0/modules/features2d/include/opencv2/features2d.hpp#L460-L488), [`ORB_Impl`](https://github.com/opencv/opencv/blob/4.10.0/modules/features2d/src/orb.cpp#L659-L721) | `inspected` | D1 | Defaults identical (500, 1.2, 8, 31, 0, 2, `HARRIS_SCORE`, 31, 20). Types: `scale_factor: f32` (OpenCV stores the `float` in a `double`), `nfeatures`/`nlevels`/`patch_size`/`first_level`/`wta_k` as `usize` (OpenCV `int`; a negative `firstLevel` is a `CV_Assert` at [L1265](https://github.com/opencv/opencv/blob/4.10.0/modules/features2d/src/orb.cpp#L1265) and at [`setFirstLevel`](https://github.com/opencv/opencv/blob/4.10.0/modules/features2d/src/orb.cpp#L684), unrepresentable here), `fast_threshold: u8`. purecv validates at call time, never in the setters. No `descriptor_size()`, `descriptor_type()`, `default_norm()` ([Missing](#missing-in-purecv-s4)). | |
| `Orb::{detect, detect_and_compute}` — octave-0 selection and `response` | [`computeKeyPoints`](https://github.com/opencv/opencv/blob/4.10.0/modules/features2d/src/orb.cpp#L826-L1001): FAST [L891-L892](https://github.com/opencv/opencv/blob/4.10.0/modules/features2d/src/orb.cpp#L891-L892), border [L896](https://github.com/opencv/opencv/blob/4.10.0/modules/features2d/src/orb.cpp#L896), pre-cull [L899](https://github.com/opencv/opencv/blob/4.10.0/modules/features2d/src/orb.cpp#L899), [`HarrisResponses`](https://github.com/opencv/opencv/blob/4.10.0/modules/features2d/src/orb.cpp#L130-L179) with block 7 [L944](https://github.com/opencv/opencv/blob/4.10.0/modules/features2d/src/orb.cpp#L944), cull [L961](https://github.com/opencv/opencv/blob/4.10.0/modules/features2d/src/orb.cpp#L961) | `diverges` | D1, D5, D6, D7 | → OpenCV. Octave 0 is the input image in both, and FAST agrees there, so this is the selection rule alone. OpenCV: `retainBest(2 × quota)` on the FAST score (ties kept), a 7×7 window of integer Sobel products scaled by `(1/(4·7·255))⁴`, `retainBest(quota)` with ties, levels concatenated in octave order. purecv: every border-filtered candidate scored with the 3×3 `corner_harris` definition (`harris_at`), stable sort, truncate, then a global sort and truncate. Measured with the defaults (109 at quota in both): 92 of 109 positions coincide on the 240×320 rectangle texture, 64 of 109 on the 192×192 fixture, **8 of 109 on 240×320 noise**; responses differ by a position-dependent factor of ~10¹²–10¹³. Output order: purecv descending response, OpenCV grouped by octave ([#196](https://github.com/webarkit/purecv/issues/196)). D6: same in all builds. Test `orb_octave0_keypoint_selection_matches_opencv`. | [#158](https://github.com/webarkit/purecv/issues/158), [#196](https://github.com/webarkit/purecv/issues/196) |
| `Orb::{detect, detect_and_compute}` — octaves ≥ 1 | the pyramid loop [L1083-L1155](https://github.com/opencv/opencv/blob/4.10.0/modules/features2d/src/orb.cpp#L1083-L1155) (`cvRound` sizes [L1088](https://github.com/opencv/opencv/blob/4.10.0/modules/features2d/src/orb.cpp#L1088), `INTER_LINEAR_EXACT` [L1128](https://github.com/opencv/opencv/blob/4.10.0/modules/features2d/src/orb.cpp#L1128), reflect-101 padding [L1136-L1145](https://github.com/opencv/opencv/blob/4.10.0/modules/features2d/src/orb.cpp#L1136-L1145)) | `diverges` | D1, D5, D7 | → OpenCV. The level pixels differ (see `build_orb_pyramid`), so even with OpenCV's selection rule the keypoints at octaves ≥ 1 cannot match. Measured per-octave counts with the defaults: rectangles 109/89/75/56/34/25/15/5 (OpenCV) vs 109/87/75/55/34/26/15/5; noise …/24/1 vs …/25/1; fixture 109/90/75/63/46/9 vs …/45/8; `fastThreshold 40`: 101/67/61/39/25/16/9/3 vs 101/65/62/39/25/17/9/3; `nlevels 4, scale 1.5`: 127/69/24/5 vs 127/70/25/5. Over all octaves 93 % (rectangles), 66 % (fixture) and 38 % (noise) of OpenCV's keypoints are also in purecv's output. The per-level budget itself is OpenCV's ([#197](https://github.com/webarkit/purecv/issues/197) is a 3-in-55,000 tie case): with `edge_threshold 15` every level is at quota and the counts are exactly 109/90/75/63/52/44/36/31, with `nfeatures 100` 22/18/15/13/10/9/7/5 (test `orb_per_level_budget_matches_opencv`, measured). Test `orb_per_octave_counts_match_opencv_above_octave_0`. | [#193](https://github.com/webarkit/purecv/issues/193), [#236](https://github.com/webarkit/purecv/issues/236) |
| `Orb::detect` — `KeyPoint.size` | [L904-L908](https://github.com/opencv/opencv/blob/4.10.0/modules/features2d/src/orb.cpp#L904-L908) | `diverges` | D1, D7 | → OpenCV. `size = patchSize × layerScale[octave]` = 31, 37.2, 44.64, 53.568, 64.2816, 77.13794, 92.56553, 111.07864 (f32); purecv keeps FAST's 7 at every octave. Test `orb_keypoint_size_matches_opencv`. | [#194](https://github.com/webarkit/purecv/issues/194) |
| `Orb::detect` — `edge_threshold < patch_size / 2` | padding [L1027-L1031](https://github.com/opencv/opencv/blob/4.10.0/modules/features2d/src/orb.cpp#L1027-L1031), [L1136-L1145](https://github.com/opencv/opencv/blob/4.10.0/modules/features2d/src/orb.cpp#L1136-L1145) | `diverges` | D2, D3, D7 | → OpenCV. OpenCV pads each level by `max(edgeThreshold, max(22, 4)) + 1` reflect-101 pixels and returns 500 keypoints at `edgeThreshold` 0, 3, 10 and 14 on the rectangle texture; purecv's `compute_orientation` returns `InvalidInput` ("circular patch … out of image boundaries") for every value below 15. `edge_threshold = 15` gives 500 in both. Test `orb_edge_threshold_below_half_patch_matches_opencv`. | [#193](https://github.com/webarkit/purecv/issues/193) |
| `Orb::{compute, detect_and_compute}` — descriptors | [`computeOrbDescriptors`](https://github.com/opencv/opencv/blob/4.10.0/modules/features2d/src/orb.cpp#L219-L350) after the level blur [L1224-L1231](https://github.com/opencv/opencv/blob/4.10.0/modules/features2d/src/orb.cpp#L1224-L1231) | `diverges` | D1, D4, D5, D6, D7 | → OpenCV. Both blur the level with a 7×7, σ 2, reflect-101 Gaussian before steered BRIEF (purecv since PR #162); OpenCV's u8 blur is the bit-exact `ufixedpoint16` path, purecv's truncates ([#229](https://github.com/webarkit/purecv/issues/229), [#157](https://github.com/webarkit/purecv/issues/157)): the blurred 240×320 rectangle texture differs on 38,075 of 76,800 pixels by exactly 1, the fixture on 18,338 of 36,864 (max 2). Descriptors on the same keypoints and angles: fixture (40 keypoints, hash texture) mean 1.20 bits, max 3, 13 of 40 exact; OpenCV's own octave-0 keypoints of the rectangle texture (109) mean **19.1 bits, max 42, none exact**, because blurred neighbours inside a flat rectangle differ by less than the blur error. Re-sampling OpenCV's blurred level with purecv's sampling rules gives 0.84 bits mean, so the blur is the cause; the residual is the sampling arithmetic of [#199](https://github.com/webarkit/purecv/issues/199) (`f32` angle conversion, `cvRound` ties). D6: same in all builds. Tests `orb_compute_fixture_descriptors_within_3_bits_of_opencv` (pin: max ≤ 3, mean ≤ 1.5), `orb_compute_descriptors_match_opencv_exactly`. | [#229](https://github.com/webarkit/purecv/issues/229), [#157](https://github.com/webarkit/purecv/issues/157) |
| `Orb::compute` — caller-supplied keypoints | [L1046-L1067](https://github.com/opencv/opencv/blob/4.10.0/modules/features2d/src/orb.cpp#L1046-L1067) (levels from `max(octave) + 1`), [L1170-L1188](https://github.com/opencv/opencv/blob/4.10.0/modules/features2d/src/orb.cpp#L1170-L1188) (border filter, regrouping) | `diverges` | D1, D2, D7 | → OpenCV (API decision pending in #195). OpenCV drops keypoints within `edgeThreshold` of the original image (`cvRound`, 30.5 → 30 < 31 dropped, 288.5 → 288 < 289 kept: the 7-keypoint example keeps 4), regroups unsorted octaves (`[3, 0, 9]` → `[0, 3, 9]`), accepts `octave >= nlevels` (builds 13 levels for octave 12) and returns the surviving keypoints; purecv returns 7 rows, rejects octave 9 and 12, and cannot return keypoints through `&[KeyPoint]`. `octave < 0` is an error in both; an empty list gives no rows in both. Test `orb_compute_provided_keypoints_filter_and_regroup_like_opencv`. | [#195](https://github.com/webarkit/purecv/issues/195) |
| `Orb::{compute, detect_and_compute}` — `wta_k` 3 and 4 | [L1214-L1222](https://github.com/opencv/opencv/blob/4.10.0/modules/features2d/src/orb.cpp#L1214-L1222), [`initializeOrbPattern`](https://github.com/opencv/opencv/blob/4.10.0/modules/features2d/src/orb.cpp#L353-L378), encodings [L286-L345](https://github.com/opencv/opencv/blob/4.10.0/modules/features2d/src/orb.cpp#L286-L345), [`defaultNorm`](https://github.com/opencv/opencv/blob/4.10.0/modules/features2d/src/orb.cpp#L772-L785) | `diverges` | D1, D7 | → OpenCV. OpenCV computes 32-byte descriptors of 2-bit codes (`NORM_HAMMING2`); purecv returns `InvalidInput` since PR #162. `detect` ignores `wta_k` in both (`wta_k = 5` detects 408 keypoints in both and fails only in `compute`, as OpenCV's `CV_Assert`). Test `orb_wta_k_3_and_4_descriptors_match_opencv` (fixture digests). | [#198](https://github.com/webarkit/purecv/issues/198) |
| `Orb::{compute, detect_and_compute}` — `patch_size != 31` | [`makeRandomPattern`](https://github.com/opencv/opencv/blob/4.10.0/modules/features2d/src/orb.cpp#L641-L650), [L1208-L1212](https://github.com/opencv/opencv/blob/4.10.0/modules/features2d/src/orb.cpp#L1208-L1212) | `diverges` | D1, D7 | → OpenCV. OpenCV draws a 512-point pattern from `cv::RNG(0x34985739)` (needs [#225](https://github.com/webarkit/purecv/issues/225)); purecv rejects. `detect` honours `patch_size` in both (orientation patch; `patchSize 21, edgeThreshold 21` detects 487 keypoints in OpenCV, 500 at the same octave quotas in purecv because of the pyramid rows above). Test `orb_patch_size_21_descriptors_match_opencv`. | [#199](https://github.com/webarkit/purecv/issues/199), [#225](https://github.com/webarkit/purecv/issues/225) |
| `Orb::{detect, compute, detect_and_compute}` — `first_level > 0` | [`getScale`](https://github.com/opencv/opencv/blob/4.10.0/modules/features2d/src/orb.cpp#L653-L656), [L1126-L1154](https://github.com/opencv/opencv/blob/4.10.0/modules/features2d/src/orb.cpp#L1126-L1154) | `diverges` | D1, D7 | → OpenCV. `firstLevel 2` on the rectangle texture: 492 keypoints, octave-0 size 21.52777671813965 (upscaled levels); purecv rejects. Test `orb_first_level_2_matches_opencv`. | [#200](https://github.com/webarkit/purecv/issues/200) |
| `Orb::{detect, compute, detect_and_compute}` — parameter validation | [`CV_Assert(patchSize >= 2)`](https://github.com/opencv/opencv/blob/4.10.0/modules/features2d/src/orb.cpp#L1018), `wta_k` [L1214](https://github.com/opencv/opencv/blob/4.10.0/modules/features2d/src/orb.cpp#L1214), `level >= 0` [L1061](https://github.com/opencv/opencv/blob/4.10.0/modules/features2d/src/orb.cpp#L1061) | `measured` | D2, D7 | `patch_size < 2`, `wta_k ∉ {2, 3, 4}` (descriptors only) and a negative octave are errors in both; 8×8 and 40×40 images give no keypoints in both; an empty keypoint list gives no descriptor rows in both. purecv additionally rejects `scale_factor <= 1` and `nlevels = 0` up front, where OpenCV fails later in `Mat::operator()` (`scaleFactor 0.5`, `nlevels 0`) or with an empty error (`scaleFactor 1.0`); an error either way. `detect_and_compute` = `detect` followed by `compute` in both (same pyramid). Tests `orb_parameter_validation_matches_opencv`, `orb_detect_then_compute_equals_detect_and_compute`. | |
| `build_orb_pyramid` | [L1083-L1155](https://github.com/opencv/opencv/blob/4.10.0/modules/features2d/src/orb.cpp#L1083-L1155) | `diverges` | D1, D3, D4, D5 | → OpenCV. (1a) sizes: `(cols × inv_scale).round()` (ties away from zero) vs `cvRound` (ties to even): 483 → 403 vs 402 at level 1; `level_scale` itself is OpenCV's `(float)pow(double, level)` since PR #162. (1b) `imgproc::resize` (f64 bilinear, rounded, [#236](https://github.com/webarkit/purecv/issues/236)) vs `INTER_LINEAR_EXACT` (`ufixedpoint16`, 2× → `INTER_AREA`), `missing` in imgproc. (1c) no reflect-101 padding. The levels are not exposed by `cv2`; the pixel differences were measured in #159 (10,852 of 213,200 level-1 pixels, max 1) and are visible here as the per-octave counts above. Not re-measured in this audit (no `cv2` binding of the pyramid). | [#193](https://github.com/webarkit/purecv/issues/193), [#236](https://github.com/webarkit/purecv/issues/236) |
| `precompute_umax` | [L861-L876](https://github.com/opencv/opencv/blob/4.10.0/modules/features2d/src/orb.cpp#L861-L876) | `inspected` | D1, D4 | Same construction (`cvFloor`/`cvCeil` of `half·√2/2`, `cvRound(sqrt(half² − v²))`, then the symmetric fill). purecv rounds with `f64::round` (ties away) where OpenCV uses `cvRound` (ties to even), but `sqrt` of an integer is never a half-integer, so the tables are equal; exercised by the orientation row. | |
| `compute_orientation` | [`ICAngles`](https://github.com/opencv/opencv/blob/4.10.0/modules/features2d/src/orb.cpp#L181-L215), [`fastAtan2`](https://github.com/opencv/opencv/blob/4.10.0/modules/core/src/mathfuncs_core.simd.hpp#L42-L75) | `diverges` | D1, D2, D5, D6, D7 | keep? The moments `m_01`, `m_10` are OpenCV's: at OpenCV's 109 octave-0 keypoints of the rectangle texture the angles agree to 0.01° (max 0.009°, mean 0.003°; test `orb_orientation_within_0_01_degree_of_opencv`, measured). The residual is the arctangent: OpenCV's `fastAtan2` is a 7th-order polynomial in degrees (documented accuracy ≈ 0.3°), purecv's is `f64::atan2` rounded once; `cv2.fastAtan2` of purecv's direction reproduces OpenCV's angle on 109/109. #159 item 5 called this "within 1 f32 ulp" and needing no change; it is 2,000+ ulp and moves `cvRound`ed BRIEF sample positions. D2: purecv returns `Err` when the patch leaves the image, OpenCV reads the padding ([#193](https://github.com/webarkit/purecv/issues/193)). D6: same in all builds. Test `orb_orientation_matches_opencv_fast_atan2`. | P1 |
| `compute_orb_descriptor` | [`computeOrbDescriptors`](https://github.com/opencv/opencv/blob/4.10.0/modules/features2d/src/orb.cpp#L219-L285) (`GET_VALUE` [L244-L250](https://github.com/opencv/opencv/blob/4.10.0/modules/features2d/src/orb.cpp#L244-L250), encoding [L261-L285](https://github.com/opencv/opencv/blob/4.10.0/modules/features2d/src/orb.cpp#L261-L285)) | `diverges` | D1, D4, D7 | → OpenCV. The WTA_K 2 encoding (`t0 < t1`, least-significant bit first, 16 pairs per byte) is OpenCV's. Sampling: OpenCV converts the angle with `angle *= (float)(CV_PI/180.f)` and `cosf`/`sinf`, rotates `int × float` and rounds with `cvRound` (ties to even); purecv converts in `f64`, rotates in `f32` and rounds with `f32::round` (ties away from zero): a tie occurs at every multiple of 30° ([#199](https://github.com/webarkit/purecv/issues/199)). Samples outside the image are clamped where OpenCV reads reflect-101 padding ([#193](https://github.com/webarkit/purecv/issues/193)). The function samples `image` as given (the caller blurs); `patch_size` must be 31 and `pattern` must have ≥ 1024 entries, else `Err`. Measured through the descriptor row above (0.84 bits residual on OpenCV's blurred level). | [#199](https://github.com/webarkit/purecv/issues/199), [#193](https://github.com/webarkit/purecv/issues/193) |

## Matching (`matcher.rs`)

| purecv item | OpenCV 4.10.0 counterpart (pinned link) | Status | Dims checked | Direction / notes | Issue |
|---|---|---|---|---|---|
| `BFMatcher`, `BFMatcher::new`, fields `norm_type`, `cross_check` | [`BFMatcher::BFMatcher`](https://github.com/opencv/opencv/blob/4.10.0/modules/features2d/src/matchers.cpp#L710-L714), [`create`](https://github.com/opencv/opencv/blob/4.10.0/modules/features2d/include/opencv2/features2d.hpp#L1253-L1271) | `inspected` | D1 | Same two parameters; OpenCV's defaults (`NORM_L2`, `false`) are not encoded (`norm_type` is required). `new` returns `Result` but never fails. Generic over the element type where OpenCV checks `queryDescriptors.type() == trainDescType` at run time ([L761](https://github.com/opencv/opencv/blob/4.10.0/modules/features2d/src/matchers.cpp#L761)). | |
| `BFMatcher::match_descriptors` | [`DescriptorMatcher::match`](https://github.com/opencv/opencv/blob/4.10.0/modules/features2d/src/matchers.cpp#L579-L587) → [`BFMatcher::knnMatchImpl`](https://github.com/opencv/opencv/blob/4.10.0/modules/features2d/src/matchers.cpp#L757-L893) with `knn = 1`; [`batchDistance`](https://github.com/opencv/opencv/blob/4.10.0/modules/core/src/batch_distance.cpp#L265-L390) (cross-check [L301-L348](https://github.com/opencv/opencv/blob/4.10.0/modules/core/src/batch_distance.cpp#L301-L348)) | `measured` | D1, D2, D4, D6, D7 | **Identical** on 6×32 u8 and 5×4 f32 descriptor sets with constructed ties (duplicate train rows, duplicate query rows, an exact match): Hamming and L2 matches, the reverse direction, `imgIdx = 0`, and `crossCheck`. Ties: both keep the first train index (`d < best`, strict) in both directions, so OpenCV's "best train per query among the trains that chose it, then reject if the query's own best is not reciprocated" and purecv's "reciprocal best" give the same pairs. Empty query or train → empty result in both; a column-count mismatch is `Err`/`CV_Assert` in both. D6: same in all builds. Tests `bfmatcher_hamming_matches_opencv`, `bfmatcher_l2_matches_opencv`. | |
| `BFMatcher::knn_match` | [`DescriptorMatcher::knnMatch`](https://github.com/opencv/opencv/blob/4.10.0/modules/features2d/src/matchers.cpp#L647-L661), [`BatchDistInvoker`](https://github.com/opencv/opencv/blob/4.10.0/modules/core/src/batch_distance.cpp#L199-L261) (insertion [L235-L248](https://github.com/opencv/opencv/blob/4.10.0/modules/core/src/batch_distance.cpp#L235-L248)) | `diverges` | D1, D2, D7 | → OpenCV, for one input only. k = 1, 2, 3 and 20 (more than the 8 train rows: 8 returned) give OpenCV's matches in OpenCV's order, ties included (OpenCV's insertion sort and purecv's stable sort both keep the lower train index first); `crossCheck` with k = 1 gives OpenCV's lists (empty inner lists for rejected queries, `compactResult = false`) and k > 1 with `crossCheck` is an error in both ([batch_distance.cpp L303](https://github.com/opencv/opencv/blob/4.10.0/modules/core/src/batch_distance.cpp#L303)). **`k = 0`**: OpenCV `CV_Assert(knn > 0)` ([L655](https://github.com/opencv/opencv/blob/4.10.0/modules/features2d/src/matchers.cpp#L655)); purecv returns one empty list per query. Tests `bfmatcher_hamming_matches_opencv`, `bfmatcher_l2_matches_opencv` (pins), `bfmatcher_knn_match_k0_is_an_error`. | P2 |
| `filter_matches` | — | `purecv-only` | | Lowe's ratio test over `knn_match(…, 2)` output (`d0 < ratio × d1`); OpenCV leaves it to the caller (the tutorials write the loop by hand). | |
| `DMatch`, `DMatch::new` | [`cv::DMatch`](https://github.com/opencv/opencv/blob/4.10.0/modules/core/include/opencv2/core/types.hpp#L841-L856) | `inspected` | D1 | Same four fields (`train_img_idx` ↔ `imgIdx`), same constructor order. No `Default` (OpenCV: `(-1, -1, -1, FLT_MAX)`, measured) and no `<` by distance ([`operator<`](https://github.com/opencv/opencv/blob/4.10.0/modules/core/include/opencv2/core/types.hpp#L855); [Missing](#missing-in-purecv-s4)). Placement: [`core.md`](core.md#seeded-rows-from-the-plan). | |

## Drawing (`draw.rs`)

| purecv item | OpenCV 4.10.0 counterpart (pinned link) | Status | Dims checked | Direction / notes | Issue |
|---|---|---|---|---|---|
| `draw_keypoints` | [`drawKeypoints`](https://github.com/opencv/opencv/blob/4.10.0/modules/features2d/src/draw.cpp#L91-L123), [`_drawKeypoint`](https://github.com/opencv/opencv/blob/4.10.0/modules/features2d/src/draw.cpp#L53-L89) | `diverges` | D1, D2, D7 | → OpenCV. OpenCV: output is a BGR copy of the input (`COLOR_GRAY2BGR` for 1 channel, `CV_8UC3`/`C4` copied, anything else an error); with the default flags a circle of radius 3, with `DRAW_RICH_KEYPOINTS` a circle of radius `size / 2` plus the orientation line when `angle != -1`; all primitives `LINE_AA` with 4 fractional bits (`cvRound(pt × 16)`); `color = Scalar::all(-1)` draws each keypoint in a random colour from `theRNG()` (two calls in one process differ). purecv: output keeps the input's channel count (a gray input stays gray, so the colour collapses to its first channel), always the rich form with a midpoint circle of radius `round(size / 2)` and a Bresenham line, no anti-aliasing, no flags, no random colour. Measured (20×24, one keypoint at (10.4, 8.6), size 7): OpenCV changes 53 pixels (default) or 60 (rich), purecv 24, 19–23 positions in common. Test `draw_keypoints_matches_opencv`; `draw_canvas_layout_matches_opencv` pins that an empty keypoint list returns a 3-channel input unchanged, as OpenCV. | P4 |
| `draw_matches` | [`drawMatches`](https://github.com/opencv/opencv/blob/4.10.0/modules/features2d/src/draw.cpp#L220-L248), [`_prepareImgAndDrawKeypoints`](https://github.com/opencv/opencv/blob/4.10.0/modules/features2d/src/draw.cpp#L144-L183), [`_drawMatch`](https://github.com/opencv/opencv/blob/4.10.0/modules/features2d/src/draw.cpp#L185-L204) | `diverges` | D1, D2, D7 | → OpenCV. The canvas agrees: `img1` left, `img2` right, `max(rows) × (cols1 + cols2)`, zero elsewhere (identical bytes for 3-channel inputs and no keypoints, test `draw_canvas_layout_matches_opencv`, measured). Everything drawn differs as for `draw_keypoints` (OpenCV changes 223 pixels for one match plus two single points, purecv 101, 88 in common; the line end is clamped to the last column in OpenCV). OpenCV's output has `max(3, channels)` channels and accepts gray with BGR; purecv keeps the inputs' channel count and returns `IncompatibleChannels` for gray + BGR. A match index out of range is a `CV_Assert` in OpenCV and silently skipped here. `None` colours: OpenCV `theRNG()` per match (not reproducible across calls), purecv a fixed LCG (reproducible). No `matchesMask`, `matchesThickness`, flags or the `knn` overload ([Missing](#missing-in-purecv-s4)). Test `draw_matches_matches_opencv`. | P5 |

## `KeyPoint` (`core::keypoint`, re-exported as `features2d::KeyPoint`)

| purecv item | OpenCV 4.10.0 counterpart (pinned link) | Status | Dims checked | Direction / notes | Issue |
|---|---|---|---|---|---|
| `KeyPoint`, `KeyPoint::new`, `Default` | [`cv::KeyPoint`](https://github.com/opencv/opencv/blob/4.10.0/modules/core/include/opencv2/core/types.hpp#L744-L808) | `inspected` | D1, D7 | Same six fields and constructor order (`pt, size, angle, response, octave, class_id`); `Default` is `(0, 0, 0, -1, 0, 0, -1)` in both (measured through `cv2.KeyPoint()`). No `(x, y, …)` constructor and no `hash()` ([Missing](#missing-in-purecv-s4)). Placement in `core`: [`core.md`](core.md#seeded-rows-from-the-plan), PR #207. | |
| `KeyPoint::overlap` | [`KeyPoint::overlap`](https://github.com/opencv/opencv/blob/4.10.0/modules/core/src/types.cpp#L103-L142) | `measured` | D1, D4, D7 | Identical (1e-6) on eight cases: same, disjoint, touching, enclosed, partial ×3, zero size against positive size. Two zero-size concentric keypoints: OpenCV returns `0/0 = NaN`, purecv's `max_s_2 > 0` guard returns 0; a deliberate guard in the code, recorded here as an intentional difference (the maintainer may flip it). purecv also clamps `cos α`, `cos β` to `[-1, 1]` before `acos`, which OpenCV does not; no measured case reaches the clamp. Test `keypoint_overlap_and_convert_match_opencv`. | |
| `KeyPoint::convert_to_points` | [`KeyPoint::convert`](https://github.com/opencv/opencv/blob/4.10.0/modules/core/src/types.cpp#L65-L75) (no indexes) | `measured` | D1, D7 | Identical. | |
| `KeyPoint::convert_to_points_masked` | [`KeyPoint::convert`](https://github.com/opencv/opencv/blob/4.10.0/modules/core/src/types.cpp#L76-L90) (with `keypointIndexes`) | `measured` | D1, D2, D7 | Identical for valid indexes; a negative index is `CV_Error`/`Err` in both. An empty index list means "all keypoints" in OpenCV and an empty result here (a separate function covers the former). An index past the end reads out of bounds in OpenCV (measured: garbage) and is `Err` here: intentional (memory safety). | |
| `KeyPoint::convert_from_points` | [`KeyPoint::convert`](https://github.com/opencv/opencv/blob/4.10.0/modules/core/src/types.cpp#L93-L101) (from points) | `measured` | D1, D7 | Identical, `angle = -1`. OpenCV's defaults (`size 1, response 1, octave 0, class_id -1`) are not encoded. | |
| `KeyPoint::sort_by_response` | — | `purecv-only` | | OpenCV sorts with `std::sort` and a comparator at the call site; `retainBest` uses `nth_element` instead. | |
| `KeyPoint::retain_best` | [`KeyPointsFilter::retainBest`](https://github.com/opencv/opencv/blob/4.10.0/modules/features2d/src/keypoint.cpp#L69-L90) | `diverges` | D1 | → OpenCV (decision in #196: change this helper or add a `KeyPointsFilter`-style one). OpenCV partitions with `nth_element`, then keeps **every** keypoint whose response is ≥ the n-th one, so ties at the cutoff survive and the result can be longer than `n`; purecv sorts and truncates to exactly `n`. `n = 0` clears in both. `KeyPointsFilter` is not bound by `cv2`, so the expected behaviour is read from the source; test `keypoint_retain_best_keeps_cutoff_ties` (ignored). | [#196](https://github.com/webarkit/purecv/issues/196) |

## Seeded rows from the plan

How each row seeded by Task B3 was resolved:

| Seed | Resolution |
|---|---|
| #123, #124, #125, #126, #136, #156 (fixed by PR #162, shipped in v0.10.0, **still open**) | Confirmed fixed at the branch base: one pyramid build per `detect_and_compute` (`test_orb_detect_and_compute_builds_pyramid_once`), the 7×7/σ 2 blur before BRIEF (`test_orb_compute_blurs_the_keypoint_octave_level`; the fixture descriptors are within 3 bits, row above), Harris at keypoints (`test_harris_at_matches_corner_harris_everywhere`), unsupported `wta_k`/`first_level`/`patch_size` rejected (`test_orb_rejects_*`), the FAST candidate-buffer guard (`test_fast_keeps_last_candidate_of_each_row`, #136), platform-independent `level_scale`/angles (`test_orb_level_scales_match_opencv_f64_scales`, #156; D6 above). They can be closed. |
| ORB `compute` is `measured` only once the fixture generator records the build | The fixture was re-measured with the pinned wheel instead: byte-identical (D7 above). The generator follow-up stays a `chore` (see the PR). |
| #158 (Harris scoring), #159 → #193–#197, #160 → #198–#200: `diverges` with issue-comment evidence | All reproduced with committed scripts and `#[ignore]`d tests (rows above). Item 5 of #159 (orientation) was wrong: see `compute_orientation` (P1). |
| #161 → #201–#204: `Type9_16` + nonmax `inspected` until measured | `measured` and identical (row above). #201 (a)(b) and #202 (c)(d) reproduced with the wheel, which binds `Type5_8`/`Type7_12` (the issue asked for this re-measurement). #203 (`simd`) confirmed: no `simd` code in `fast.rs`, outputs identical across builds. #204 (`wasm`) not run. |
| `filter_matches` → `purecv-only`; `KeyPoint` placement → B1 | As seeded. |

## Missing in purecv (S4)

From the 4.10.0 doc groups [features2d main](https://docs.opencv.org/4.10.0/d5/d51/group__features2d__main.html),
[draw](https://docs.opencv.org/4.10.0/d4/d5d/group__features2d__draw.html),
[KeyPoint](https://docs.opencv.org/4.10.0/d2/d29/classcv_1_1KeyPoint.html),
[BFMatcher](https://docs.opencv.org/4.10.0/d3/da1/classcv_1_1BFMatcher.html) and the
`xfeatures2d` module of opencv_contrib. Roadmap gaps, not bugs. Links point at the
declarations in `features2d.hpp` / `xfeatures2d.hpp`.

| purecv item | OpenCV 4.10.0 counterpart (pinned link) | Status | Dims checked | Direction / notes | Issue |
|---|---|---|---|---|---|
| — | [`SIFT`](https://github.com/opencv/opencv/blob/4.10.0/modules/features2d/include/opencv2/features2d.hpp#L266-L351) | `missing` | | In the main repository since 4.4. | |
| — | [`BRISK`](https://github.com/opencv/opencv/blob/4.10.0/modules/features2d/include/opencv2/features2d.hpp#L353-L421) | `missing` | | | |
| — | [`MSER`](https://github.com/opencv/opencv/blob/4.10.0/modules/features2d/include/opencv2/features2d.hpp#L507-L571) | `missing` | | | |
| — | [`AgastFeatureDetector`](https://github.com/opencv/opencv/blob/4.10.0/modules/features2d/include/opencv2/features2d.hpp#L629-L682), `cv::AGAST` | `missing` | | | |
| — | [`GFTTDetector`](https://github.com/opencv/opencv/blob/4.10.0/modules/features2d/include/opencv2/features2d.hpp#L684-L743) | `missing` | | A `Feature2D` wrapper of `goodFeaturesToTrack`, which purecv has in imgproc ([#244](https://github.com/webarkit/purecv/issues/244)). | |
| — | [`SimpleBlobDetector`](https://github.com/opencv/opencv/blob/4.10.0/modules/features2d/include/opencv2/features2d.hpp#L745-L793) | `missing` | | | |
| — | [`KAZE`](https://github.com/opencv/opencv/blob/4.10.0/modules/features2d/include/opencv2/features2d.hpp#L795-L855), [`AKAZE`](https://github.com/opencv/opencv/blob/4.10.0/modules/features2d/include/opencv2/features2d.hpp#L857-L990) | `missing` | | | |
| — | [`AffineFeature`](https://github.com/opencv/opencv/blob/4.10.0/modules/features2d/include/opencv2/features2d.hpp#L241-L257) (ASIFT wrapper) | `missing` | | | |
| — | [`FlannBasedMatcher`](https://github.com/opencv/opencv/blob/4.10.0/modules/features2d/include/opencv2/features2d.hpp#L1293-L1340) | `missing` | | | |
| — | [`DescriptorMatcher::{add, train, getTrainDescriptors, clear}`](https://github.com/opencv/opencv/blob/4.10.0/modules/features2d/include/opencv2/features2d.hpp#L1017-L1042), the `masks`/`compactResult` arguments, [`radiusMatch`](https://github.com/opencv/opencv/blob/4.10.0/modules/features2d/include/opencv2/features2d.hpp#L1104-L1106), [`DescriptorMatcher::create`](https://github.com/opencv/opencv/blob/4.10.0/modules/features2d/src/matchers.cpp#L1022-L1057) | `missing` | | Stored train collections (multi-image `imgIdx`), masked matching, radius matching, matcher by name. | |
| — | `NORM_L1`, `NORM_L2SQR`, `NORM_HAMMING2` in [`BFMatcher`](https://github.com/opencv/opencv/blob/4.10.0/modules/features2d/include/opencv2/features2d.hpp#L1253) ([`batchDistance`](https://github.com/opencv/opencv/blob/4.10.0/modules/core/src/batch_distance.cpp#L351-L377)) | `missing` | | `NORM_HAMMING2` (2-bit cells, [`popCountTable2`](https://github.com/opencv/opencv/blob/4.10.0/modules/core/src/norm.cpp#L28)) is needed for `wta_k` 3/4. | [#198](https://github.com/webarkit/purecv/issues/198) |
| — | [`Feature2D::{descriptorSize, descriptorType, defaultNorm}`](https://github.com/opencv/opencv/blob/4.10.0/modules/features2d/include/opencv2/features2d.hpp#L201-L203) on `Orb` ([`orb.cpp` L762-L785](https://github.com/opencv/opencv/blob/4.10.0/modules/features2d/src/orb.cpp#L762-L785)); the `mask` argument of [`detect`](https://github.com/opencv/opencv/blob/4.10.0/modules/features2d/include/opencv2/features2d.hpp#L151-L153) / [`detectAndCompute`](https://github.com/opencv/opencv/blob/4.10.0/modules/features2d/include/opencv2/features2d.hpp#L196-L199); the free function [`cv::FAST`](https://github.com/opencv/opencv/blob/4.10.0/modules/features2d/include/opencv2/features2d.hpp#L623-L624) | `missing` | | 32 / `CV_8U` / `NORM_HAMMING` (`NORM_HAMMING2` for `wta_k` 3, 4). | [#198](https://github.com/webarkit/purecv/issues/198) |
| — | [`KeyPointsFilter::{runByImageBorder, runByKeypointSize, runByPixelsMask, removeDuplicated, removeDuplicatedSorted}`](https://github.com/opencv/opencv/blob/4.10.0/modules/features2d/include/opencv2/features2d.hpp#L92-L127) ([keypoint.cpp L105-L291](https://github.com/opencv/opencv/blob/4.10.0/modules/features2d/src/keypoint.cpp#L105-L291)) | `missing` | | ORB's border filter is private (`detect_in_pyramid`); `runByImageBorder` with OpenCV's `cvRound` rule is what [#195](https://github.com/webarkit/purecv/issues/195) needs. | [#195](https://github.com/webarkit/purecv/issues/195) |
| — | [`KeyPoint::hash`](https://github.com/opencv/opencv/blob/4.10.0/modules/core/include/opencv2/core/types.hpp#L769), the `(x, y, …)` constructor | `missing` | | | |
| — | [`DMatch::DMatch()`](https://github.com/opencv/opencv/blob/4.10.0/modules/core/include/opencv2/core/types.hpp#L844) (`-1, -1, -1, FLT_MAX`), [`DMatch::operator<`](https://github.com/opencv/opencv/blob/4.10.0/modules/core/include/opencv2/core/types.hpp#L855) | `missing` | | No `Default` and no ordering by distance. | |
| — | [`DrawMatchesFlags`](https://github.com/opencv/opencv/blob/4.10.0/modules/features2d/include/opencv2/features2d.hpp#L1342-L1355), the `flags`/`matchesMask`/`matchesThickness` arguments of [`drawMatches`](https://github.com/opencv/opencv/blob/4.10.0/modules/features2d/include/opencv2/features2d.hpp#L1397-L1409), [`drawMatchesKnn`](https://github.com/opencv/opencv/blob/4.10.0/modules/features2d/include/opencv2/features2d.hpp#L1411-L1415) | `missing` | | | P4, P5 |
| — | opencv_contrib [`xfeatures2d::matchGMS`](https://github.com/opencv/opencv_contrib/blob/4.10.0/modules/xfeatures2d/include/opencv2/xfeatures2d.hpp#L1254-L1256) | `missing` | | Free implementation, in the wheel (`cv2.xfeatures2d.matchGMS`). | [#101](https://github.com/webarkit/purecv/issues/101) |
| — | opencv_contrib [`xfeatures2d::TEBLID`](https://github.com/opencv/opencv_contrib/blob/4.10.0/modules/xfeatures2d/include/opencv2/xfeatures2d.hpp#L312-L351) | `missing` | | Needs `integral` ([#100](https://github.com/webarkit/purecv/issues/100)). | [#102](https://github.com/webarkit/purecv/issues/102) |
| — | opencv_contrib [`xfeatures2d::FREAK`](https://github.com/opencv/opencv_contrib/blob/4.10.0/modules/xfeatures2d/include/opencv2/xfeatures2d.hpp#L84-L121) | `missing` | | | [#103](https://github.com/webarkit/purecv/issues/103) |
| — | opencv_contrib [`xfeatures2d::StarDetector`](https://github.com/opencv/opencv_contrib/blob/4.10.0/modules/xfeatures2d/include/opencv2/xfeatures2d.hpp#L123-L158) | `missing` | | Needs `integral` ([#100](https://github.com/webarkit/purecv/issues/100)). | [#104](https://github.com/webarkit/purecv/issues/104) |
| — | opencv_contrib `xfeatures2d::{`[`BriefDescriptorExtractor`](https://github.com/opencv/opencv_contrib/blob/4.10.0/modules/xfeatures2d/include/opencv2/xfeatures2d.hpp#L160)`, `[`LUCID`](https://github.com/opencv/opencv_contrib/blob/4.10.0/modules/xfeatures2d/include/opencv2/xfeatures2d.hpp#L181)`, `[`LATCH`](https://github.com/opencv/opencv_contrib/blob/4.10.0/modules/xfeatures2d/include/opencv2/xfeatures2d.hpp#L222)`, `[`BEBLID`](https://github.com/opencv/opencv_contrib/blob/4.10.0/modules/xfeatures2d/include/opencv2/xfeatures2d.hpp#L264)`, `[`DAISY`](https://github.com/opencv/opencv_contrib/blob/4.10.0/modules/xfeatures2d/include/opencv2/xfeatures2d.hpp#L353)`, `[`MSDDetector`](https://github.com/opencv/opencv_contrib/blob/4.10.0/modules/xfeatures2d/include/opencv2/xfeatures2d.hpp#L463)`, `[`VGG`](https://github.com/opencv/opencv_contrib/blob/4.10.0/modules/xfeatures2d/include/opencv2/xfeatures2d.hpp#L518)`, `[`BoostDesc`](https://github.com/opencv/opencv_contrib/blob/4.10.0/modules/xfeatures2d/include/opencv2/xfeatures2d.hpp#L576)`, `[`HarrisLaplaceFeatureDetector`](https://github.com/opencv/opencv_contrib/blob/4.10.0/modules/xfeatures2d/include/opencv2/xfeatures2d.hpp#L1088)`, `[`AffineFeature2D`](https://github.com/opencv/opencv_contrib/blob/4.10.0/modules/xfeatures2d/include/opencv2/xfeatures2d.hpp#L1135)`, `[`matchLOGOS`](https://github.com/opencv/opencv_contrib/blob/4.10.0/modules/xfeatures2d/include/opencv2/xfeatures2d.hpp#L1269)`}` | `missing` | | Not requested by any issue; listed for completeness. `SURF` ([`nonfree.hpp`](https://github.com/opencv/opencv_contrib/blob/4.10.0/modules/xfeatures2d/include/opencv2/xfeatures2d/nonfree.hpp#L85)) is patent-encumbered and compiled out of the wheel. | |

## Proposed issues

New `diverges` rows, one issue each, titled per S5. **Not filed yet**: they are listed here
and in the PR description for the maintainer's approval; once filed they go into milestone 8
("Parity audit: purecv ↔ OpenCV 4.10.0") and their numbers replace `P<n>` here and in
`tests/parity_features2d.rs`.

| Proposal | Title | Rows | Direction | Depends on |
|---|---|---|---|---|
| P1 | `features2d: compute_orientation differs from OpenCV 4.10.0: exact atan2 instead of fastAtan2 (up to 0.009° measured, 0.3° documented)` | `compute_orientation`, `Orb::detect` | keep? (bit-parity needs a `fast_atan2` port, which `core::phase`/`cart_to_polar` also need: [`core.md`](core.md)) | — |
| P2 | `features2d: BFMatcher::knn_match differs from OpenCV 4.10.0: k = 0 returns empty lists instead of an error` | `BFMatcher::knn_match` | → OpenCV | — |
| P3 | `features2d: FastFeatureDetector::detect and Orb::{detect, compute, detect_and_compute} differ from OpenCV 4.10.0: multi-channel input rejected instead of converted to gray` | `FastFeatureDetector::detect`, `Orb::*` | keep? | — |
| P4 | `features2d: draw_keypoints differs from OpenCV 4.10.0: input channel count kept, no DrawMatchesFlags, radius size/2 instead of 3, Bresenham instead of anti-aliased sub-pixel primitives, no random colour` | `draw_keypoints` | → OpenCV | — |
| P5 | `features2d: draw_matches differs from OpenCV 4.10.0: rendering as draw_keypoints, mixed channel counts rejected, invalid match indices skipped, no matchesMask/matchesThickness/flags` | `draw_matches` | → OpenCV | P4 |

Existing issues that this audit's measurements update (comments to post after the merge, with
the maintainer's approval):

- [#159](https://github.com/webarkit/purecv/issues/159): item 5 (orientation) is a
  divergence, not "within 1 ulp" (P1).
- [#202](https://github.com/webarkit/purecv/issues/202): `Type5_8`/`Type7_12` re-measured
  with the wheel, as the issue asked; the squares example (six corners vs none with nonmax)
  and the arc table are in `features2d_fast.py`.
- [#229](https://github.com/webarkit/purecv/issues/229): the ORB descriptor cost of the u8
  blur is 19 bits per descriptor on flat textures, not the 1.2 bits of the fixture.
- [#123](https://github.com/webarkit/purecv/issues/123)–[#126](https://github.com/webarkit/purecv/issues/126),
  [#136](https://github.com/webarkit/purecv/issues/136),
  [#156](https://github.com/webarkit/purecv/issues/156): confirmed fixed by PR #162 at the
  branch base (tests named in [Seeded rows](#seeded-rows-from-the-plan)); can be closed.

## Notes

- **Counting.** The summary counts one row per table line (63 rows). Grouped rows (the
  getter/setter rows, the `xfeatures2d` "for completeness" line, each `missing` line) count
  once.
- **D6.** Neither `parallel` nor `simd` changed any result in this module. `wasm32` was not
  run ([#204](https://github.com/webarkit/purecv/issues/204)).
- **Panics.** None found: every invalid input measured returns `Err` (OpenCV: `CV_Assert`).
- **IPP.** Not involved: `cv2.ipp.useIPP()` was off, and none of the measured functions
  has an IPP path.
- **Not measurable with `cv2`.** The ORB pyramid levels (`build_orb_pyramid` row, measured
  in #159 with opencv.js), `KeyPointsFilter` (not bound), `cv::FAST` (measured through the
  detector), `DMatch::operator<`.
- **Cross-module dependencies.** ORB parity at octave 0 needs
  [#158](https://github.com/webarkit/purecv/issues/158) (selection) and
  [#229](https://github.com/webarkit/purecv/issues/229)/[#157](https://github.com/webarkit/purecv/issues/157)
  (u8 blur, imgproc/core); at octaves ≥ 1 also
  [#193](https://github.com/webarkit/purecv/issues/193) with
  [#236](https://github.com/webarkit/purecv/issues/236) (`INTER_LINEAR_EXACT`, imgproc);
  `wta_k`/`patch_size` need [#225](https://github.com/webarkit/purecv/issues/225)
  (`cv::RNG`, core). P1 shares a `fast_atan2` port with the core `phase`/`cart_to_polar`
  rows.
- **Inputs.** The scripts and the tests build every input from the same integer LCG
  (`state × 1664525 + 1013904223 mod 2³²`), two drawn squares and the fixture's hash
  texture, so no image file is committed.

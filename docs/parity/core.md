# Parity audit: `core` ↔ OpenCV 4.10.0

Task B1 of [#172](https://github.com/webarkit/purecv/issues/172), sub-issue
[#173](https://github.com/webarkit/purecv/issues/173). Method, statuses (`measured`,
`inspected`, `diverges`, `intentional`, `purecv-only`, `missing`) and dimensions D1–D7:
[`README.md`](README.md). Inventory taken at the branch base `4558848` (`parity-audit`,
v0.11.0): every `pub` item of `src/core`, the re-exports of `src/core.rs`, and `src/version.rs`.

## D7: reference and measurement record

- **Sources read:** `cpp_ref/src/opencv` at `71d3237a093b60a27601c20e9ee6c3e52154e8b1`
  (`verify opencv @ … OK`), `cpp_ref/src/opencv_contrib` at
  `1ed3dd2c53888e3289afdb22ec4e9ebbff3dba87` (`verify opencv_contrib @ … OK`), bootstrapped on
  2026-10-18. The generic CPU path was read; IPP, OpenCL and HAL back ends were skipped. Where a
  `*.simd.hpp` scalar tail is the generic kernel (arithm, convert), that tail is cited.
- **Measurements (Python):** `cpp_ref/.venv`, `pip freeze` = `numpy==2.4.6`,
  `opencv-contrib-python==4.10.0.84`; `cv2.__version__` = `4.10.0`; `cv2.ipp.setUseIPP(False)`
  → `useIPP()` = False; `useOptimized()` = True; Python 3.12.2; Windows 11 10.0.26300 x86-64;
  `getBuildInformation()`: `Version control: 4.10.0`, `Baseline: SSE SSE2 SSE3`,
  `Dispatched code generation: SSE4_1 SSE4_2 FP16 AVX AVX2`, `Intel IPP: 2021.11.0`
  (disabled at run time). No row changed when re-run with IPP on.
- **Measurements (opencv.js):** `cpp_ref/node_modules/@techstark/opencv-js@4.10.0-release.1`,
  `getBuildInformation()` → `Version control: 4.10.0`, Node 24. Used only for `Mat.convertTo`,
  which Python's `cv2` does not bind.
- **Scripts (evidence rule):** `scripts/opencv_ref/core_convert_arith.py`, `core_stats.py`,
  `core_dft_dct.py`, `core_rng.py`, `core_kmeans.py`, `core_misc.py`, `core_convert_to.js`
  (helper `_core_common.py`). Each reads seeded inputs defined in the script and prints the
  build record with its results.
- **Tests:** `tests/parity_core.rs`. 32 passing tests pin the rows that match; 24 tests marked
  `#[ignore = "diverges from OpenCV 4.10.0, …"]` are the executable form of every `diverges`
  row below and fail when run with `--ignored`. Run with `--features fft,transforms` for the
  transform rows and `--features simd` / `--release` for the D6 rows.
- **D6:** checked for `add`, `subtract`, `multiply`, `divide` (default, `--release`,
  `--features simd`) and `convert_scale_abs` (the `simd` kernel shares the scalar rounding).
  Not checked for the other rows; `wasm32` not checked.
- **Row grouping:** items that differ only by element type or arity (the `CV_*C*` constants,
  `Vec*`/`Point*`/`Size*`/`Rect*` aliases, `DynamicMatrix::new_*`, the `*_scalar` variants,
  the `cv_log_*!` macros) share one row. Every other public item has its own row.

## Summary

| Status | Rows |
|---|---|
| `measured` | 28 |
| `inspected` | 54 |
| `diverges` | 33 |
| `intentional` | 1 |
| `purecv-only` | 9 |
| `missing` | 47 |

Headline findings, in order of impact (details in the rows, proposals in
[Proposed issues](#proposed-issues)):

1. **Integer arithmetic does not saturate.** `add`, `subtract` and `multiply` compute
   `s1 op s2` on the element type: on u8 overflow they **panic in debug builds, wrap in release
   builds** (200 + 128 → 72) and saturate **only** with `--features simd` (u8 add/sub kernels).
   OpenCV saturates on every integer depth. This is the only D6 failure found: three builds,
   three results.
2. **Every integer-output path truncates instead of `saturate_cast`:** `convert_to` (#192),
   `divide` (7/2 → 3, OpenCV 4), the `*_scalar` operands (10.7 → +10, OpenCV +11; 300 → +0),
   `add_weighted`, `convert_scale_abs` (ties), `normalize`, `reduce` (AVG), `transform`, `gemm`
   and `randn` on integer types (samples above 255 become **0**). All need the shared
   `saturate_cast` of #189.
3. **Defaults and flags that mean something else:** `copy_make_border` border code 1 is
   REFLECT (OpenCV: REPLICATE; REFLECT_101 unreachable), `idft` always scales, `dft` of real
   input returns complex pairs where OpenCV returns the CCS packing, `check_range` is inclusive
   where OpenCV is `[min, max)`, `invert` returns the determinant where OpenCV returns 1.0,
   `has_no_zero` tests "every element non-zero" where `hasNonZero` tests "any element
   non-zero", `CV_LOG2` is log2(e) instead of ln 2.
4. **RNG and approximations:** `randu`/`randn`/`rand_shuffle` use a different generator from
   `cv::RNG`; the maintainer decided to make `cv::RNG` the default and keep xoshiro256** as an
   option (P21). `phase`/`cart_to_polar` use exact `atan2` where OpenCV uses `fastAtan2`
   (0.0095° at 45°): kept on purpose, `intentional`. `exp`/`log`/`pow` are correctly rounded
   where OpenCV's polynomials are 1–2 ULP off (recorded as `measured` within that tolerance).
5. **Permissive inputs:** many functions accept depths or shapes OpenCV rejects (`sqrt` on u8,
   `gemm` on u8, `dct` of odd sizes, `min_max_loc` on multi-channel). Correct results are noted
   on the row; silently wrong results (`determinant` of a non-square → 0, `mean` of >4 channels
   drops channels, `reduce` SUM overflow → 0) are `diverges`.

Link bases used below: `H` = `https://github.com/opencv/opencv/blob/4.10.0/modules/core/include/opencv2/core`,
`S` = `https://github.com/opencv/opencv/blob/4.10.0/modules/core/src` (written out in full in every link).

## Types, constants, errors, utilities

| purecv item | OpenCV 4.10.0 counterpart (pinned link) | Status | Dims checked | Direction / notes | Issue |
|---|---|---|---|---|---|
| `Matrix<T>` (fields `rows`, `cols`, `channels`, `data`) | [`cv::Mat`](https://github.com/opencv/opencv/blob/4.10.0/modules/core/include/opencv2/core/mat.hpp#L811) | `inspected` | D1, D2 | Row-major, always continuous, 2-D only, element type fixed at compile time (≈ `Mat_<T>`). No ROI/`step`/refcount, no 16F or n-D. OpenCV `Mat(rows, cols, type)` leaves data uninitialised; `Matrix::new` zero-fills. | |
| `Depth` | [`CV_8U`…`CV_16F`](https://github.com/opencv/opencv/blob/4.10.0/modules/core/include/opencv2/core/hal/interface.h#L73-L80) | `inspected` | D1 | Values 0–7 identical. `CV_16F` has no element type behind it. | |
| `Depth::{is_signed, is_float, byte_size}` | [`CV_ELEM_SIZE1`](https://github.com/opencv/opencv/blob/4.10.0/modules/core/include/opencv2/core/cvdef.h#L494-L495) | `inspected` | D1 | `byte_size` ↔ `CV_ELEM_SIZE1`; the two predicates are purecv conveniences. | |
| `MatType`, `MatType::{new, depth, channels, to_int}` | [`CV_MAKETYPE`, `CV_MAT_DEPTH`, `CV_MAT_CN`](https://github.com/opencv/opencv/blob/4.10.0/modules/core/include/opencv2/core/hal/interface.h#L70-L86) | `inspected` | D1 | Same encoding: `depth + ((cn - 1) << 3)`. | |
| `CV_8U`…`CV_16F`, `CV_8UC1`…`CV_64FC4` | [`interface.h`](https://github.com/opencv/opencv/blob/4.10.0/modules/core/include/opencv2/core/hal/interface.h#L73-L134) | `inspected` | D1 | Identical values. `CV_16FC*` and the `CV_8UC(n)` macro form are not provided. | |
| `DataType` trait | [`cv::DataType`](https://github.com/opencv/opencv/blob/4.10.0/modules/core/include/opencv2/core/traits.hpp#L112) | `inspected` | D1 | `depth()` only, for the seven OpenCV depths; `u32`/`i64`/`u64` have no depth in either library. | |
| `Scalar<T>`, `Scalar::{new, from_value, all, from_array, to_array, map, channel_or_default}`, `Add`/`Sub`/`Mul`/`Div`/`Index` | [`cv::Scalar_`](https://github.com/opencv/opencv/blob/4.10.0/modules/core/include/opencv2/core/types.hpp#L669) | `inspected` | D1 | `all` ↔ `Scalar_::all`; `from_value(v)` ↔ `Scalar_(v0)`. `mul`, `conj`, `isReal` missing; `map`, `from_array`, `to_array`, `channel_or_default` are additions. | |
| `Scalar::checked_div` | — | `purecv-only` | | `Result`-returning division (OpenCV `Scalar_::operator/` divides by zero). | |
| `Point<T>`, `Point::new`, `Add`/`Sub`, `Point2i/2f/2d/2l` | [`cv::Point_`](https://github.com/opencv/opencv/blob/4.10.0/modules/core/include/opencv2/core/types.hpp#L162) | `inspected` | D1 | `dot`, `ddot`, `cross`, `inside`, scalar `*`, `norm` missing. | |
| `Point3<T>`, `Point3::new`, `Point3i/3f/3d` | [`cv::Point3_`](https://github.com/opencv/opencv/blob/4.10.0/modules/core/include/opencv2/core/types.hpp#L254) | `inspected` | D1 | No operators yet. | |
| `Size<T>`, `Size::{new, area}`, `Size2i/2f/2d` | [`cv::Size_`](https://github.com/opencv/opencv/blob/4.10.0/modules/core/include/opencv2/core/types.hpp#L334) | `inspected` | D1 | `empty`, `aspectRatio` missing. | |
| `Rect<T>`, `Rect::{new, tl, br, area}`, `Rect2i/2f/2d` | [`cv::Rect_`](https://github.com/opencv/opencv/blob/4.10.0/modules/core/include/opencv2/core/types.hpp#L443) | `inspected` | D1 | `contains`, `size`, `empty`, `&`, `\|` missing. | |
| `RotatedRect`, `RotatedRect::new` | [`cv::RotatedRect`](https://docs.opencv.org/4.10.0/db/dd6/classcv_1_1RotatedRect.html) | `inspected` | D1 | `points`, `boundingRect`, `boundingRect2f` missing. | |
| `VecN<T, N>`, `VecN::{from_array, to_array, map, new (N = 2, 3, 4, 6), zeros, all, dot}`, operators, `Vec2b`…`Vec6d` | [`cv::Vec`](https://github.com/opencv/opencv/blob/4.10.0/modules/core/include/opencv2/core/matx.hpp#L368) | `inspected` | D1 | `all`, `zeros`, `dot` match. `mul`, `cross` (Vec3), `conj` and the `Vec8i`/`Vec2w`… aliases missing; `+`/`-` with `Scalar` are additions. | |
| `TermCriteria`, `TermType` | [`cv::TermCriteria`](https://github.com/opencv/opencv/blob/4.10.0/modules/core/include/opencv2/core/types.hpp#L885-L895) | `inspected` | D1 | `Count = 1`, `Eps = 2`, `Both = 3` ↔ `COUNT`, `EPS`, `COUNT + EPS`. `isValid` missing. | |
| `BorderTypes` | [`cv::BorderTypes`](https://github.com/opencv/opencv/blob/4.10.0/modules/core/include/opencv2/core/base.hpp#L268-L279) | `inspected` | D1, D3 | Values identical; `Default` = `Reflect101` = `BORDER_DEFAULT`. `copy_make_border` does not take it (see that row). | |
| `CmpTypes` | [`cv::CmpTypes`](https://github.com/opencv/opencv/blob/4.10.0/modules/core/include/opencv2/core/base.hpp#L211-L217) | `inspected` | D1 | Identical. Not re-exported from `purecv::core`. | |
| `NormTypes` | [`cv::NormTypes`](https://github.com/opencv/opencv/blob/4.10.0/modules/core/include/opencv2/core/base.hpp#L159-L207) | `inspected` | D1 | Values identical, including `Relative = 8` and `MinMax = 32`. `norm` accepts three of them (see `norm`). | |
| `ReduceTypes` | [`cv::ReduceTypes`](https://github.com/opencv/opencv/blob/4.10.0/modules/core/include/opencv2/core.hpp#L226-L231) | `inspected` | D1 | `REDUCE_SUM2 = 4` missing. | |
| `DecompTypes` | [`cv::DecompTypes`](https://github.com/opencv/opencv/blob/4.10.0/modules/core/include/opencv2/core/base.hpp#L133-L145) | `inspected` | D1 | Identical values; only `DECOMP_LU` is implemented (see `solve`, `invert`). | |
| `SORT_EVERY_ROW`, `SORT_EVERY_COLUMN`, `SORT_ASCENDING`, `SORT_DESCENDING` | [`cv::SortFlags`](https://github.com/opencv/opencv/blob/4.10.0/modules/core/include/opencv2/core.hpp#L153-L160) | `inspected` | D1 | Identical. | |
| `KMEANS_RANDOM_CENTERS`, `KMEANS_PP_CENTERS`, `KMEANS_USE_INITIAL_LABELS` | [`cv::KmeansFlags`](https://github.com/opencv/opencv/blob/4.10.0/modules/core/include/opencv2/core.hpp#L209-L219) | `inspected` | D1 | Identical. | |
| `GEMM_1_T`, `GEMM_2_T`, `GEMM_3_T` | [`cv::GemmFlags`](https://github.com/opencv/opencv/blob/4.10.0/modules/core/include/opencv2/core/base.hpp#L220-L223) | `inspected` | D1 | Identical. | |
| `DFT_INVERSE`, `DFT_SCALE`, `DFT_ROWS`, `DFT_COMPLEX_OUTPUT`, `DFT_REAL_OUTPUT` | [`cv::DftFlags`](https://github.com/opencv/opencv/blob/4.10.0/modules/core/include/opencv2/core/base.hpp#L225-L264) | `inspected` | D1 | Identical values. `DFT_COMPLEX_INPUT = 64`, `DCT_INVERSE`, `DCT_ROWS` missing. | |
| `CV_PI`, `CV_2PI` | [`cvdef.h`](https://github.com/opencv/opencv/blob/4.10.0/modules/core/include/opencv2/core/cvdef.h#L380-L381) | `inspected` | D1 | Identical to f64 precision. | |
| `CV_LOG2` | [`cvdef.h`](https://github.com/opencv/opencv/blob/4.10.0/modules/core/include/opencv2/core/cvdef.h#L382) | `diverges` | D1 | → OpenCV. OpenCV `CV_LOG2` = 0.693147… (ln 2); purecv = `LOG2_E` = 1.442695… (log2 e). | proposed P22 |
| `CV_PI_2`, `CV_PI_4`, `CV_SQRT2`, `CV_E`, `CV_LN2`, `CV_LN10` | — | `purecv-only` | | `cvdef.h` defines only `CV_PI`, `CV_2PI`, `CV_LOG2`. `CV_LN2` carries the value OpenCV calls `CV_LOG2`. | |
| `Uchar`, `Schar`, `Short`, `Ushort`, `Int`, `Uint`, `Int64`, `Uint64`, `Float`, `Double` | [`uchar`, `schar`, `ushort`, `int64`, `uint64`](https://github.com/opencv/opencv/blob/4.10.0/modules/core/include/opencv2/core/hal/interface.h) | `inspected` | D1 | Type aliases; not re-exported from `purecv::core`. | |
| `PureCvError`, `Result<T>` | [`cv::Exception`](https://github.com/opencv/opencv/blob/4.10.0/modules/core/include/opencv2/core.hpp#L114), [`cv::Error::Code`](https://github.com/opencv/opencv/blob/4.10.0/modules/core/include/opencv2/core/base.hpp#L68) | `inspected` | D2 | `Result` instead of exceptions; six variants against ~40 codes. `NotImplemented` marks the sub-features listed as `missing` below. | |
| `DynamicMatrix`, `DynamicData`, `DynamicMatrix::{new, zeros, ones, new_u8…new_f64, rows, cols, channels, mat_type, depth_name, total, data_u8/f32/f64, data_ptr(_mut), as_matrix_u8/f32/f64(_mut), at_f64, convert_to(&str), copy_to}`, `dispatch_dynamic!` | — (type-erased `cv::Mat`) | `purecv-only` | | Wrapper over `Matrix<T>`. `convert_to` takes depth strings and routes through `Matrix::convert_to`, so #192 applies to it. | |
| `utils::Range`, `Range::{new, size, empty, all}` | [`cv::Range`](https://github.com/opencv/opencv/blob/4.10.0/modules/core/include/opencv2/core/types.hpp#L622) | `inspected` | D1 | `all()` = `(INT_MIN, INT_MAX)` in both. Not re-exported from `purecv::core`; nothing takes it yet. | |
| `utils::ParIterFallback` | — | `purecv-only` | | Sequential stand-in for Rayon's `par_iter` when `parallel` is off. | |
| `logging::LogLevel` | [`cv::utils::logging::LogLevel`](https://github.com/opencv/opencv/blob/4.10.0/modules/core/include/opencv2/core/utils/logger.defines.hpp#L25-L33) | `inspected` | D1 | Values 0–6 identical. | |
| `logging::{set_log_level, get_log_level}` (also re-exported from `utils`) | [`setLogLevel`, `getLogLevel`](https://github.com/opencv/opencv/blob/4.10.0/modules/core/include/opencv2/core/utils/logger.hpp#L25-L27) | `inspected` | D1 | Both return the previous level. Backed by the `log` facade. | |
| `logging::init_basic_logger` | — | `purecv-only` | | OpenCV's default logger writes to stderr and is configured by `OPENCV_LOG_LEVEL`. | |
| `logging::tags::{PURECV, CORE, IMGPROC, FEATURES2D, CALIB3D, VIDEO}` | `CV_LOG_TAG` / `cv::utils::logging::LogTag` ([logger.hpp](https://github.com/opencv/opencv/blob/4.10.0/modules/core/include/opencv2/core/utils/logger.hpp)) | `inspected` | D1 | String targets instead of `LogTag` objects; per-tag levels come from `RUST_LOG`-style filtering of the backend. | |
| `cv_log_fatal!`, `cv_log_error!`, `cv_log_warning!`, `cv_log_info!`, `cv_log_debug!`, `cv_log_verbose!` | [`CV_LOG_*`](https://github.com/opencv/opencv/blob/4.10.0/modules/core/include/opencv2/core/utils/logger.hpp#L129-L134) | `inspected` | D1 | `cv_log_verbose!` has no verbosity argument (`CV_LOG_VERBOSE(tag, v, …)`). | |
| `cv_log_once_{error, warning, info, debug}!`, `cv_log_if_{error, warning, info, debug}!` | [`CV_LOG_ONCE_*`, `CV_LOG_IF_*`](https://github.com/opencv/opencv/blob/4.10.0/modules/core/include/opencv2/core/utils/logger.hpp#L167-L169) | `inspected` | D1 | Same shape. | |
| `cv_bail!`, `cv_err!`, `cv_bail_debug!`, `cv_err_debug!` | [`CV_Error`, `CV_Assert`](https://github.com/opencv/opencv/blob/4.10.0/modules/core/include/opencv2/core/base.hpp#L304-L320) | `inspected` | D2 | Return `Err(PureCvError)` and log; OpenCV throws. | |
| `version::VERSION`, `version::get_version`, `version::print_version` | [`CV_VERSION`](https://github.com/opencv/opencv/blob/4.10.0/modules/core/include/opencv2/core/version.hpp#L19), [`getVersionString`](https://github.com/opencv/opencv/blob/4.10.0/modules/core/include/opencv2/core/utility.hpp#L250) | `purecv-only` | | Reports purecv's crate version, not an OpenCV version; the reference version lives in `CONTRIBUTING.md`. | |
| `utils::get_tick_count`, `utils::get_tick_frequency` | [`getTickCount`, `getTickFrequency`](https://github.com/opencv/opencv/blob/4.10.0/modules/core/src/system.cpp#L908-L919) | `inspected` | D1 | Both on `steady_clock`. OpenCV returns the clock's epoch count and period; purecv returns nanoseconds since the first call and a fixed 1e9. Ratios agree. `std` only. | |
| `utils::border_interpolate` | [`borderInterpolate`](https://github.com/opencv/opencv/blob/4.10.0/modules/core/src/copy.cpp#L748-L792) | `measured` | D1–D3, D7 | 20 positions × 5 modes identical, `len = 1` → 0 for both reflect modes. `Transparent`/`Isolated` return −1 where OpenCV errors (D2). `core_misc.py`. | |

## `Matrix` methods

| purecv item | OpenCV 4.10.0 counterpart (pinned link) | Status | Dims checked | Direction / notes | Issue |
|---|---|---|---|---|---|
| `Matrix::new`, `Matrix::from_size` | [`Mat(rows, cols, type)`](https://github.com/opencv/opencv/blob/4.10.0/modules/core/include/opencv2/core/mat.hpp#L811) | `inspected` | D1 | Zero-filled (OpenCV: uninitialised). | |
| `Matrix::from_vec` | [`Mat(rows, cols, type, void* data)`](https://github.com/opencv/opencv/blob/4.10.0/modules/core/include/opencv2/core/mat.hpp#L811) | `inspected` | D1, D2 | Takes ownership. A length mismatch `assert_eq!`-panics: the only panicking constructor in `core`. | |
| `Matrix::dims_match` | `Mat::size` / `Mat::type` comparison | `inspected` | D1 | Helper; same role as `_src.sameSize(...)` checks. | |
| `Matrix::{flat_index, get, at, get_mut, at_mut, set}` | [`Mat::at`](https://github.com/opencv/opencv/blob/4.10.0/modules/core/include/opencv2/core/mat.hpp#L1961), [`Mat::ptr`](https://github.com/opencv/opencv/blob/4.10.0/modules/core/include/opencv2/core/mat.hpp#L1866) | `inspected` | D2 | Bounds-checked `Option` instead of `CV_DbgAssert`. `at(i32, i32)` mirrors `Mat::at(row, col)`. | |
| `Matrix::{as_slice, as_mut_slice, data_ptr, data_ptr_mut, swap}` | `Mat::data`, `Mat::ptr`, `std::swap` | `inspected` | D1 | | |
| `Matrix::{mat_type, depth, channels}` | [`Mat::type`, `Mat::depth`, `Mat::channels`](https://github.com/opencv/opencv/blob/4.10.0/modules/core/include/opencv2/core/mat.hpp#L1787-L1808) | `inspected` | D1 | | |
| `Matrix::{new_with_type, create, create_with_type}` | [`Mat::create`](https://github.com/opencv/opencv/blob/4.10.0/modules/core/include/opencv2/core/mat.hpp#L1494) | `inspected` | D1, D2 | A `MatType` whose depth is not `T` returns `Err` (OpenCV cannot express the mismatch). | |
| `Matrix::{new_with_scalar, new_with_scalar_from_size, new_with_scalar_typed_from_size}` | [`Mat(rows, cols, type, const Scalar&)`](https://github.com/opencv/opencv/blob/4.10.0/modules/core/include/opencv2/core/mat.hpp#L811) | `inspected` | D1 | Channels ≥ 4 get `T::default()`, as `Scalar` has four components in both. | |
| `Matrix::set_to`, `Matrix::set_to_masked` | [`Mat::setTo`](https://github.com/opencv/opencv/blob/4.10.0/modules/core/src/copy.cpp#L616) | `inspected` | D1, D2 | Typed `Scalar<T>` (no conversion). Mask: channel 0 of a `Matrix<u8>` is read; OpenCV accepts a multi-channel 8U mask. | |
| `Matrix::convert_to<U>()` | [`Mat::convertTo`](https://github.com/opencv/opencv/blob/4.10.0/modules/core/src/convert.dispatch.cpp#L248-L300); kernels [`cvt_`](https://github.com/opencv/opencv/blob/4.10.0/modules/core/src/convert.simd.hpp#L103-L127), [`saturate_cast`](https://github.com/opencv/opencv/blob/4.10.0/modules/core/include/opencv2/core/saturate.hpp#L103-L142), [`cvRound`](https://github.com/opencv/opencv/blob/4.10.0/modules/core/include/opencv2/core/fast_math.hpp#L200-L219) | `diverges` | D1, D2, D4, D7 | → OpenCV. OpenCV: round half to even, then clamp, never fails. f32 `[0.5, 1.5, 2.5, 254.6, 300, −1]` → u8 `[0, 2, 2, 255, 255, 0]` (opencv.js `convertTo`; cv2 via `addWeighted(dtype)` agrees for u8/i8/u16/i16/i32, f32 and f64 sources, with and without alpha/beta). purecv truncates (`[0, 1, 2, 254]`) and returns `Err` for 300 and −1. No `alpha`/`beta`. Platform note: OpenCV maps NaN, ±inf and |x| ≥ 2³¹ through `cvRound` → `INT_MIN` on x86-64 (`u8` → 0, `i8` → −128, `i32` → −2147483648), not a value to port. `core_convert_to.js`, `core_convert_arith.py`. | [#192](https://github.com/webarkit/purecv/issues/192) (needs [#189](https://github.com/webarkit/purecv/issues/189)) |
| `Matrix::copy_to` | [`Mat::copyTo`](https://github.com/opencv/opencv/blob/4.10.0/modules/core/src/copy.cpp#L305) | `inspected` | D1 | Reallocates `dst` on shape mismatch, as OpenCV. Masked variant: see `missing`. | |
| `Matrix::{zeros, zeros_from_size, zeros_with_type}` | [`Mat::zeros`](https://github.com/opencv/opencv/blob/4.10.0/modules/core/include/opencv2/core/mat.hpp#L1393) | `inspected` | D1 | | |
| `Matrix::{ones, ones_from_size, ones_with_type}` | [`Mat::ones`](https://github.com/opencv/opencv/blob/4.10.0/modules/core/include/opencv2/core/mat.hpp#L1424) | `diverges` | D1 | → OpenCV (maintainer decision, 2026-10-19): only channel 0 gets the value. Multi-channel: OpenCV sets **channel 0** to 1 and the others to 0 ([doc note](https://github.com/opencv/opencv/blob/4.10.0/modules/core/include/opencv2/core/mat.hpp#L1418)); purecv fills every channel with 1 (`ones(1, 1, 3)` → `[1, 1, 1]`). Single channel identical. Why OpenCV does this: `ones`/`eye` return a `MatExpr` that stores one `double` scale (so `Mat::ones(...)*3` costs nothing) and [`MatOp_Initializer::assign`](https://github.com/opencv/opencv/blob/4.10.0/modules/core/src/matrix_expressions.cpp#L1677-L1695) writes it as `Scalar(alpha)`, whose one-argument constructor sets only `val[0]` ([types.hpp](https://github.com/opencv/opencv/blob/4.10.0/modules/core/include/opencv2/core/types.hpp#L2190-L2194)). The docs describe both as Matlab-style initializers (Matlab has no channels): an implementation consequence, documented as a note, not a multi-channel design. | proposed P13 |
| `Matrix::eye` | [`Mat::eye`](https://github.com/opencv/opencv/blob/4.10.0/modules/core/include/opencv2/core/mat.hpp#L1453), [`MatOp_Initializer`](https://github.com/opencv/opencv/blob/4.10.0/modules/core/src/matrix_expressions.cpp#L1688) | `diverges` | D1 | → OpenCV (maintainer decision, 2026-10-19): only channel 0 gets the value. Same channel-0 rule ([doc note](https://github.com/opencv/opencv/blob/4.10.0/modules/core/include/opencv2/core/mat.hpp#L1447)): `eye(2, 2, 2)` in purecv puts 1 in both channels of the diagonal. Single channel identical. Why OpenCV does this: `ones`/`eye` return a `MatExpr` that stores one `double` scale (so `Mat::ones(...)*3` costs nothing) and [`MatOp_Initializer::assign`](https://github.com/opencv/opencv/blob/4.10.0/modules/core/src/matrix_expressions.cpp#L1677-L1695) writes it as `Scalar(alpha)`, whose one-argument constructor sets only `val[0]` ([types.hpp](https://github.com/opencv/opencv/blob/4.10.0/modules/core/include/opencv2/core/types.hpp#L2190-L2194)). The docs describe both as Matlab-style initializers (Matlab has no channels): an implementation consequence, documented as a note, not a multi-channel design. | proposed P13 |
| `Matrix::diag(&[T])` | [`Mat::diag(const Mat&)`](https://github.com/opencv/opencv/blob/4.10.0/modules/core/src/matrix.cpp#L1264) | `inspected` | D1 | Builds the diagonal matrix. The extracting `Mat::diag(int d)` is `missing`. | |
| `Matrix::{as_ndarray_view, as_ndarray_view_mut, into_ndarray, from_ndarray}`, `From<ndarray::Array3<T>>` (`ndarray` feature) | — | `purecv-only` | | | |

## Element-wise arithmetic (`arithm.rs`)

| purecv item | OpenCV 4.10.0 counterpart (pinned link) | Status | Dims checked | Direction / notes | Issue |
|---|---|---|---|---|---|
| `add` | [`cv::add`](https://github.com/opencv/opencv/blob/4.10.0/modules/core/src/arithm.cpp#L995-L1000) → [`arithm_op`](https://github.com/opencv/opencv/blob/4.10.0/modules/core/src/arithm.cpp#L593-L731), kernel [`c_add`](https://github.com/opencv/opencv/blob/4.10.0/modules/core/src/arithm.simd.hpp#L115-L125) | `diverges` | D1, D2, D4–D7 | → OpenCV. OpenCV: `saturate_cast<T>(a + b)` on every depth (u8 `[128+128, 200+255]` → `[255, 255]`). purecv `s1 + s2`: **debug panic**, **release wrap** (`128 + 128` → 0, `200 + 255` → 199), **saturating only with `simd`** (u8 kernel). i8/i16/i32 overflow likewise in every build. f32 identical. No `mask`, no `dtype`. `core_convert_arith.py`. | proposed P1 |
| `subtract` | [`cv::subtract`](https://github.com/opencv/opencv/blob/4.10.0/modules/core/src/arithm.cpp#L1003-L1011) | `diverges` | D1, D2, D4–D7 | → OpenCV. OpenCV clamps at 0 (`1 − 3` → 0); purecv panics/wraps (`1 − 3` → 254 in release), saturates only with `simd`. Differences that fit in the type are identical. | proposed P1 |
| `multiply` | [`cv::multiply`](https://github.com/opencv/opencv/blob/4.10.0/modules/core/src/arithm.cpp#L1118-L1124), kernel [`c_mul`](https://github.com/opencv/opencv/blob/4.10.0/modules/core/src/arithm.simd.hpp#L180-L190) | `diverges` | D1, D2, D4–D7 | → OpenCV. OpenCV `saturate_cast<T>(a·b)` (u8 `100·3` → 255); purecv `s1 * s2` panics/wraps (→ 44) in **every** build (no integer kernel). No `scale`, no `dtype`. | proposed P1 |
| `divide` | [`cv::divide`](https://github.com/opencv/opencv/blob/4.10.0/modules/core/src/arithm.cpp#L1125-L1136), kernel [`op_div_scale`](https://github.com/opencv/opencv/blob/4.10.0/modules/core/src/arithm.simd.hpp#L1573-L1601) | `diverges` | D1, D2, D4, D5, D7 | → OpenCV. Integer depths: OpenCV computes the quotient in float and `saturate_cast`s it (`7/2` → 4, `250/7` → 36, `100/3` → 33); purecv divides in integers (3, 35, 33). `x/0` → 0 in both. Float depths: OpenCV is IEEE (`1/0` → inf, `0/0` → nan); purecv returns 0. No `scale`, no `divide(scale, src2)`. | proposed P2 |
| `add_scalar`, `subtract_scalar` | `cv::add`/`subtract` with a `Scalar` operand: [working type](https://github.com/opencv/opencv/blob/4.10.0/modules/core/src/arithm.cpp#L690-L712), [`convertAndUnrollScalar`](https://github.com/opencv/opencv/blob/4.10.0/modules/core/src/copy.cpp#L103-L120) | `diverges` | D1, D4, D7 | → OpenCV. OpenCV converts the scalar to the 32S working type with `saturate_cast` and saturates the result: `+10.7` adds 11, `+10.5` adds 10, `+300` → 255, `−1.5` subtracts 2. purecv `T::from_f64(sc).unwrap_or_default()`: 10.7 adds 10, and any scalar outside the type's range becomes **0** (`+300`, `−1.5` leave the matrix unchanged). `Scalar<f64>` per channel in both. | proposed P3 |
| `multiply_scalar`, `divide_scalar` | `cv::multiply`/`divide` with a scalar ([`arithm_op`](https://github.com/opencv/opencv/blob/4.10.0/modules/core/src/arithm.cpp#L593-L731)) | `diverges` | D1, D4, D7 | → OpenCV. OpenCV computes in double and `saturate_cast`s: `9·0.3` → 3, `128/3` → 43, `127/0.4` → 255. purecv truncates (2, 42) and maps out-of-range results to 0 (`127/0.4` → **0**). `/0` → 0 in both. | proposed P3 |
| `absdiff`, `abs_diff`, `abs_diff_scalar` | [`cv::absdiff`](https://github.com/opencv/opencv/blob/4.10.0/modules/core/src/arithm.cpp#L1013-L1018) | `measured` | D1, D2, D4, D7 | 12 u8 pairs identical. `abs_diff` duplicates `absdiff`. Signed types: `s1 − s2` can overflow (covered by P1). | |
| `bitwise_and`, `bitwise_or`, `bitwise_xor`, `bitwise_not`, `bitwise_{and, or, xor}_scalar` | [`cv::bitwise_*`](https://github.com/opencv/opencv/blob/4.10.0/modules/core/src/arithm.cpp#L360-L391) | `inspected` | D1, D2 | Same bit operations. OpenCV also applies them to float data as bytes; purecv requires `BitAnd` etc. (integers). No `mask`. | |
| `min`, `max`, `min_scalar`, `max_scalar` | [`cv::min`, `cv::max`](https://github.com/opencv/opencv/blob/4.10.0/modules/core/src/arithm.cpp#L392-L404) | `measured` | D1, D2, D7 | 12 u8 pairs identical. | |
| `compare`, `compare_scalar` | [`cv::compare`](https://github.com/opencv/opencv/blob/4.10.0/modules/core/src/arithm.cpp#L1311) | `measured` | D1, D2, D7 | 255/0 per channel identical. `compare_scalar` takes `Scalar<T>`; OpenCV compares against a double (`u8 > 2.5` works). | |
| `in_range`, `in_range_scalar` | [`cv::inRange`](https://github.com/opencv/opencv/blob/4.10.0/modules/core/src/arithm.cpp#L1855) | `measured` | D1, D2, D7 | Single-channel mask, all channels inclusive, identical. `in_range_scalar` takes slices instead of `Scalar`. | |
| `add_weighted` | [`cv::addWeighted`](https://github.com/opencv/opencv/blob/4.10.0/modules/core/src/arithm.cpp#L1170-L1176), kernel [`c_add(a, b, α, β, γ)`](https://github.com/opencv/opencv/blob/4.10.0/modules/core/src/arithm.simd.hpp#L128-L132) | `diverges` | D1, D4, D7 | → OpenCV. f32: identical. u8: OpenCV `saturate_cast` in float (`51.5` → 52, `3.5` → 4, `128.5` → 128, `910` → 255); purecv truncates (51, 3, 128) and maps out-of-range sums to **0**. No `dtype`. | proposed P4 |
| `convert_scale_abs` | [`convertScaleAbs`](https://github.com/opencv/opencv/blob/4.10.0/modules/core/src/convert_scale.dispatch.cpp#L87), kernel [`cvtScaleAbs`](https://github.com/opencv/opencv/blob/4.10.0/modules/core/src/convert_scale.simd.hpp#L44-L52) | `diverges` | D1, D4, D6, D7 | → OpenCV. i16 with α = 1: 12 values identical. Ties: OpenCV rounds half to even (`0.5` → 0, `2.5` → 2, i16 `1·0.5` → 0), purecv `round()` (1, 3, 1). NaN → 0 in both. `1e10` → OpenCV 0 (`cvRound` indefinite on x86-64), purecv 255: platform artefact, not to port. The `simd` f32/f64 kernels use the same `round()`, so D6 holds. | proposed P5 (needs [#189](https://github.com/webarkit/purecv/issues/189)) |
| `sqrt` | [`cv::sqrt`](https://github.com/opencv/opencv/blob/4.10.0/modules/core/src/mathfuncs.cpp#L1421) | `measured` | D1, D2, D7 | 8 f32 values bit-identical (hardware `sqrt` in both). purecv accepts integer types (truncating); OpenCV asserts 32F/64F. | |
| `exp`, `log` | [`cv::exp`](https://github.com/opencv/opencv/blob/4.10.0/modules/core/src/mathfuncs.cpp#L717-L722), [`cv::log`](https://github.com/opencv/opencv/blob/4.10.0/modules/core/src/mathfuncs.cpp#L750-L755); [`exp32f`, `log32f`](https://github.com/opencv/opencv/blob/4.10.0/modules/core/src/mathfuncs_core.simd.hpp#L23-L25) | `measured` | D1, D2, D7 | Within 2 ULP (f32). OpenCV's polynomials: `exp(1)` = 2.7182819843, `exp(0.5)` = 1.6487212181, `log(0.5)` = −0.6931471825; purecv computes in f64 and rounds once (2.7182817, 1.6487213, −0.6931472, the correctly rounded values). `exp(100)` → inf, `log(0)` → −inf, `log(−1)` → nan in both. Integer input accepted by purecv only. | |
| `pow` | [`cv::pow`](https://github.com/opencv/opencv/blob/4.10.0/modules/core/src/mathfuncs.cpp#L1263-L1295) | `measured` | D1, D2, D7 | Integer powers exact in both (`x²`, `x³` with negative base → −64, `x⁻¹`); `0.5` = `sqrt` identical; non-integer powers within 2 ULP (`100^1.5`: OpenCV 1000.000122, purecv 1000.0). Negative base with non-integer power → nan in both. | |
| `magnitude` | [`cv::magnitude`](https://github.com/opencv/opencv/blob/4.10.0/modules/core/src/mathfuncs.cpp#L147-L160) | `measured` | D1, D2, D7 | 7 f32 values bit-identical. OpenCV: 32F/64F only. | |
| `phase`, `cart_to_polar` | [`cv::phase`](https://github.com/opencv/opencv/blob/4.10.0/modules/core/src/mathfuncs.cpp#L183-L200), [`cv::cartToPolar`](https://github.com/opencv/opencv/blob/4.10.0/modules/core/src/mathfuncs.cpp#L278-L300) → [`fastAtan2` polynomial](https://github.com/opencv/opencv/blob/4.10.0/modules/core/src/mathfuncs_core.simd.hpp#L36-L73) | `intentional` | D1, D2, D7 | Kept (maintainer decision, 2026-10-19). OpenCV's `fastAtan2` returns 44.990456° for (1, 1), 135.009552° for (−1, 1), 53.136284° for (3, 4); purecv's libm `atan2` gives 45°, 135°, 53.130104°. Magnitudes identical. Both return `[0, 360)`/`[0, 2π)`. Rationale: purecv's result is the correctly rounded angle; `fastAtan2` is a degree-7 polynomial whose error (about 0.01° on the fixture) is an approximation, not a behaviour worth reproducing, and #156 already moved ORB orientation to an exact f64 `atan2` for cross-platform determinism. Callers that need OpenCV's bits can be served later by a separate `fast_atan2` (see `missing`). `tests/parity_core.rs` pins purecv's exact values. | |
| `polar_to_cart` | [`cv::polarToCart`](https://github.com/opencv/opencv/blob/4.10.0/modules/core/src/mathfuncs.cpp#L583-L612), [`SinCos_32f`](https://github.com/opencv/opencv/blob/4.10.0/modules/core/src/mathfuncs.cpp#L331) | `measured` | D1, D2, D7 | Fed OpenCV's own angles, x/y agree within 1e-6 (OpenCV's `SinCos_32f` table vs libm). Degrees and radians. | |
| `check_range` | [`checkRange`](https://github.com/opencv/opencv/blob/4.10.0/modules/core/src/mathfuncs.cpp#L1508-L1545) | `diverges` | D1, D7 | → OpenCV. OpenCV's range is `[minVal, maxVal)` ([L1537–L1538](https://github.com/opencv/opencv/blob/4.10.0/modules/core/src/mathfuncs.cpp#L1537-L1538)): 255 fails `[0, 255)`. purecv is inclusive. NaN → false in both. No `quiet`/`pos`. | proposed P8 |
| `lut` | [`cv::LUT`](https://github.com/opencv/opencv/blob/4.10.0/modules/core/src/lut.cpp#L359-L380) | `measured` | D1, D2, D7 | 1-channel and 3-channel tables, f32 output, 255-entry error identical. OpenCV also takes 8S sources. | |
| `sum`, `mean` | [`cv::sum`](https://github.com/opencv/opencv/blob/4.10.0/modules/core/src/sum.dispatch.cpp#L184-L203), [`cv::mean`](https://github.com/opencv/opencv/blob/4.10.0/modules/core/src/mean.dispatch.cpp#L125-L139) | `diverges` | D1, D2, D7 | → OpenCV. ≤ 4 channels identical. OpenCV asserts `cn <= 4`; purecv silently sums the first four channels (`mean` of a 5-channel matrix returns four means). No `mask`. | proposed P13 |
| `mean_std_dev` | [`meanStdDev`](https://github.com/opencv/opencv/blob/4.10.0/modules/core/src/mean.dispatch.cpp#L516-L644) | `measured` | D1, D7 | Population standard deviation in both; u8 and f32 3-channel identical to 1e-12. No `mask`. | |
| `norm` | [`cv::norm`](https://github.com/opencv/opencv/blob/4.10.0/modules/core/src/norm.cpp#L603-L610) | `diverges` | D1, D2, D7 | → OpenCV. `Inf`, `L1`, `L2`, with and without mask, multi-channel: identical. `L2Sqr`, `Hamming`, `Hamming2` return `NotImplemented` where OpenCV computes them (177.5; popcount 12 and 7 on u8). `Relative` and the two-array form `norm(src1, src2)` ([L1073](https://github.com/opencv/opencv/blob/4.10.0/modules/core/src/norm.cpp#L1073)) are `missing`. `MinMax` rejected by both. | proposed P9 |
| `normalize` | [`cv::normalize`](https://github.com/opencv/opencv/blob/4.10.0/modules/core/src/norm.cpp#L1369-L1414) | `diverges` | D1, D4, D7 | → OpenCV. f32 `MinMax`, `L1`, `L2`, `Inf`: identical (≤ 1e-7; OpenCV's float `scale`/`shift` gives 1.0000001 for the 0..1 range). Integer destination: OpenCV goes through `convertTo` (u8 `[10, 20, 30, 250]` → `[0, 11, 21, 255]`), purecv truncates (`[0, 10, 21, 255]`). Swapped bounds: OpenCV uses `min(a, b)`/`max(a, b)`, purecv inverts the range. `dtype` is accepted and ignored (OpenCV converts). Masked: untouched elements kept in both. `L2Sqr` rejected by both. | proposed P10 (needs [#189](https://github.com/webarkit/purecv/issues/189)) |
| `reduce` | [`cv::reduce`](https://github.com/opencv/opencv/blob/4.10.0/modules/core/src/matrix_operations.cpp#L765-L850) | `diverges` | D1, D2, D4, D7 | → OpenCV. `Max`/`Min` on u8 and `Sum`/`Avg` on f32: identical. u8 `Avg`: OpenCV accumulates in 32S and `saturate_cast`s (`1.5` → 2, `67.7` → 68); purecv truncates (1, 67). u8 `Sum`: OpenCV refuses 8U→8U ("Unsupported combination", `dtype` required); purecv returns **0** where the sum overflows (300 → 0). No `dtype`; `REDUCE_SUM2` missing. | proposed P11 |
| `count_non_zero` | [`countNonZero`](https://github.com/opencv/opencv/blob/4.10.0/modules/core/src/count_non_zero.dispatch.cpp#L119-L124) | `measured` | D1, D2, D7 | Single channel required in both; identical. | |
| `has_no_zero` | [`hasNonZero`](https://github.com/opencv/opencv/blob/4.10.0/modules/core/src/has_non_zero.dispatch.cpp#L60-L65) | `diverges` | D1, D2, D7 | → OpenCV (maintainer decision, 2026-10-19): add `has_non_zero` with `hasNonZero`'s semantics (single channel, any element ≠ 0); keep the current predicate under the name `has_no_zero` (purecv-only). Different predicate today: `hasNonZero` is "any element ≠ 0", `has_no_zero` is "all elements ≠ 0". purecv also accepts multi-channel input (OpenCV asserts `cn == 1`). | proposed P12 |
| `min_max_loc` | [`minMaxLoc`](https://github.com/opencv/opencv/blob/4.10.0/modules/core/src/minmax.cpp#L1593-L1615), [`minMaxIdx` asserts](https://github.com/opencv/opencv/blob/4.10.0/modules/core/src/minmax.cpp#L1498-L1506) | `measured` | D1, D2, D7 | First occurrence, `(x, y)` locations, NaN skipped, u8 and f32: identical. No `mask`. Multi-channel: OpenCV asserts (locations need `cn == 1`); purecv scans all channels and returns the pixel (D2, permissive). | |

## Linear algebra, solvers, clustering

| purecv item | OpenCV 4.10.0 counterpart (pinned link) | Status | Dims checked | Direction / notes | Issue |
|---|---|---|---|---|---|
| `gemm` | [`cv::gemm`](https://github.com/opencv/opencv/blob/4.10.0/modules/core/src/matmul.dispatch.cpp#L338-L390) | `measured` | D1, D2, D4, D7 | f32 `α·A·B + β·C` and `GEMM_1_T` identical; shape mismatch → `Err`/assert. OpenCV: 32F/64F (and complex) only; purecv any `T`, accumulating in f64 and writing with `from_f64().unwrap_or(0)` (integer outputs truncate/zero, same class as P3). `src3` is passed by reference (empty matrix for none). | |
| `set_identity` | [`cv::setIdentity`](https://github.com/opencv/opencv/blob/4.10.0/modules/core/src/matrix_operations.cpp#L184-L230) | `measured` | D1, D7 | Multi-channel `Scalar` per channel identical. | |
| `dot` | [`Mat::dot`](https://github.com/opencv/opencv/blob/4.10.0/modules/core/include/opencv2/core/mat.hpp#L1377) | `inspected` | D1, D2 | Requires equal element counts (OpenCV: same size and type). Accumulates in f64. | |
| `cross` | [`Mat::cross`](https://github.com/opencv/opencv/blob/4.10.0/modules/core/include/opencv2/core/mat.hpp#L1367) | `inspected` | D1, D2 | 3-element vectors; OpenCV float only. | |
| `trace` | [`cv::trace`](https://github.com/opencv/opencv/blob/4.10.0/modules/core/include/opencv2/core.hpp#L1923) | `measured` | D1, D7 | 4×4 and 2-channel identical. | |
| `determinant` | [`cv::determinant`](https://github.com/opencv/opencv/blob/4.10.0/modules/core/src/lapack.cpp#L715-L781) | `diverges` | D1, D2, D7 | → OpenCV. 2×2, 4×4 and singular identical. Non-square or multi-channel input returns **0.0** where OpenCV asserts. LU pivot threshold 1e-12 absolute vs [`DBL_EPSILON·100`](https://github.com/opencv/opencv/blob/4.10.0/modules/core/src/matrix_decomp.cpp#L15-L90). Integer input accepted. | proposed P13 |
| `invert` | [`cv::invert`](https://github.com/opencv/opencv/blob/4.10.0/modules/core/src/lapack.cpp#L793-L1017) | `diverges` | D1, D7 | → OpenCV. Inverse values identical; singular → 0 and a zero matrix in both. Return value: OpenCV LU returns **1.0** (non-zero flag, [L1006–L1017](https://github.com/opencv/opencv/blob/4.10.0/modules/core/src/lapack.cpp#L1006-L1017)); purecv returns the determinant (10.0). `dst` is always `Matrix<f64>` (OpenCV: source type). `DECOMP_SVD`/`EIG`/`CHOLESKY` → `NotImplemented` (see `missing`). | proposed P14 |
| `solve` | [`cv::solve`](https://github.com/opencv/opencv/blob/4.10.0/modules/core/src/lapack.cpp#L1032-L1050) | `measured` | D1, D2, D7 | LU solution identical (1e-12); singular → `false` in both. Only `DECOMP_LU`; `DECOMP_NORMAL`, `SVD`, `QR`, `CHOLESKY`, `EIG` → `NotImplemented`. `dst` always f64. | |
| `transform` | [`cv::transform`](https://github.com/opencv/opencv/blob/4.10.0/modules/core/src/matmul.dispatch.cpp#L452-L520) | `diverges` | D1, D2, D4, D7 | → OpenCV. f32 and the affine `dcn × (scn+1)` form identical; bad `m` → `Err`/assert. Integer destination: OpenCV `saturate_cast` (`(10+20+33)·0.3` = 18.9 → 19, `229.5` → 230), purecv truncates (18, 229). | proposed P15 (needs [#189](https://github.com/webarkit/purecv/issues/189)) |
| `perspective_transform` | [`cv::perspectiveTransform`](https://github.com/opencv/opencv/blob/4.10.0/modules/core/src/matmul.dispatch.cpp#L544-L570) | `measured` | D1, D2, D7 | f64 points identical (1e-12); `w = 0` → 0 in both (threshold `f64::EPSILON` vs `FLT_EPSILON`/`DBL_EPSILON` by depth). purecv accepts integer input; OpenCV asserts 32F/64F. | |
| `solve_poly` | [`cv::solvePoly`](https://github.com/opencv/opencv/blob/4.10.0/modules/core/src/mathfuncs.cpp#L1960-L2091) | `measured` | D1, D7 | Real and complex roots identical (1e-9; order differs). `max_iters <= 0` → 300 (OpenCV: 1000, default argument 300). Return: max residual |P(z)| vs OpenCV's max root update. | |
| `sort`, `sort_idx` | [`cv::sort`, `cv::sortIdx`](https://github.com/opencv/opencv/blob/4.10.0/modules/core/src/matrix_operations.cpp#L1252-L1290) | `measured` | D1, D7 | Rows/columns, ascending/descending and the tie order of the fixture identical (purecv: stable sort then reverse; OpenCV: `std::sort` with the inverse comparator, not guaranteed stable). | |
| `kmeans` | [`cv::kmeans`](https://github.com/opencv/opencv/blob/4.10.0/modules/core/src/kmeans.cpp#L228-L460) | `diverges` | D1, D2, D7 | → OpenCV. `KMEANS_PP_CENTERS`, `KMEANS_USE_INITIAL_LABELS`, K = 1 and `K > N` → `Err`: compactness 9.5 / 543.1666 and centers identical. `KMEANS_RANDOM_CENTERS`: purecv picks K samples with an LCG seeded by the attempt index and stays in a local minimum (198.75 with 3 attempts); OpenCV draws centers uniformly in the bounding box from `theRNG()` ([`generateRandomCenter`](https://github.com/opencv/opencv/blob/4.10.0/modules/core/src/kmeans.cpp#L55-L60)) and reaches 9.5. Also: OpenCV clamps `maxCount` to [2, 100] and tests `shift² ≤ ε²`; kmeans++ uses 3 trials; out-of-range initial labels assert (purecv clamps to 0). `centers` is `&mut Option<Matrix<f32>>`. | proposed P16 |
| `solve_quadratic` | — | `purecv-only` | | OpenCV reaches the quadratic through `solveCubic` with `a = 0`, or `solvePoly`. | |
| `solve_cubic(a, b, c, d)` | [`cv::solveCubic`](https://github.com/opencv/opencv/blob/4.10.0/modules/core/src/mathfuncs.cpp#L1813-L1955) | `measured` | D1, D7 | Real roots identical (1e-9) for three roots, one root, a triple root, `a = 0` (quadratic, linear) and no real root. Returns `Vec` instead of `(n, roots[3])`; the 3-coefficient monic form is not offered. | |

## Structural (`structural.rs`)

| purecv item | OpenCV 4.10.0 counterpart (pinned link) | Status | Dims checked | Direction / notes | Issue |
|---|---|---|---|---|---|
| `flip` | [`cv::flip`](https://github.com/opencv/opencv/blob/4.10.0/modules/core/src/matrix_transform.cpp#L780) | `measured` | D1, D7 | Codes 0, > 0, < 0 identical. | |
| `flip_nd` | [`cv::flipND`](https://github.com/opencv/opencv/blob/4.10.0/modules/core/src/matrix_transform.cpp#L842) | `diverges` | D1 | → OpenCV. OpenCV: `axis` in `[−ndims, ndims)`, negative counts from the end ([L850–L852](https://github.com/opencv/opencv/blob/4.10.0/modules/core/src/matrix_transform.cpp#L850-L852)), so `−1` flips columns; purecv maps every negative axis to "both". | proposed P13 |
| `transpose` | [`cv::transpose`](https://github.com/opencv/opencv/blob/4.10.0/modules/core/src/matrix_transform.cpp#L244) | `measured` | D1, D7 | Identical. | |
| `split`, `merge` | [`cv::split`](https://github.com/opencv/opencv/blob/4.10.0/modules/core/src/split.dispatch.cpp#L215), [`cv::merge`](https://github.com/opencv/opencv/blob/4.10.0/modules/core/src/merge.dispatch.cpp#L116-L135) | `inspected` | D1, D2 | Same semantics and checks (same size, same depth, single-channel inputs). | |
| `mix_channels` | [`cv::mixChannels`](https://github.com/opencv/opencv/blob/4.10.0/modules/core/src/channels.cpp#L95) | `inspected` | D1 | `from_to` pairs over the concatenated channel index, as OpenCV. (The `parallel` branch is a no-op loop; the sequential loop does the work.) | |
| `copy_make_border` | [`cv::copyMakeBorder`](https://github.com/opencv/opencv/blob/4.10.0/modules/core/src/copy.cpp#L1021-L1060) | `diverges` | D1–D3, D7 | → OpenCV. Border code is a raw `i32`: 1 = REFLECT (`fedcba\|abcdefgh`), anything else = CONSTANT. In OpenCV's `BorderTypes` 1 is REPLICATE, 2 REFLECT, 3 WRAP, 4 REFLECT_101 (the default of every filter), 16 the ISOLATED flag. With the matching code the CONSTANT and REFLECT outputs are identical. A border wider than the image (`top = 5` on 2 rows) indexes out of bounds and **panics**; OpenCV iterates the reflection. | proposed P18 |
| `rotate` | [`cv::rotate`](https://github.com/opencv/opencv/blob/4.10.0/modules/core/src/matrix_transform.cpp#L1104) | `measured` | D1, D7 | Codes 0, 1, 2 identical. Unknown code → `Err` (OpenCV leaves `dst` undefined). | |
| `repeat` | [`cv::repeat`](https://github.com/opencv/opencv/blob/4.10.0/modules/core/include/opencv2/core.hpp#L1196) | `measured` | D1, D7 | Identical; `ny = 0` → `Err`/assert. | |
| `reshape` | [`Mat::reshape(cn, rows)`](https://github.com/opencv/opencv/blob/4.10.0/modules/core/src/matrix.cpp#L1138) | `inspected` | D1 | Same `(channels, rows)` order and `0` = keep; copies instead of aliasing. | |
| `hconcat`, `vconcat` | [`cv::hconcat`, `cv::vconcat`](https://github.com/opencv/opencv/blob/4.10.0/modules/core/src/matrix_operations.cpp#L52-L140) | `inspected` | D1 | Same checks. | |
| `extract_channel`, `insert_channel` | [`cv::extractChannel`, `cv::insertChannel`](https://github.com/opencv/opencv/blob/4.10.0/modules/core/src/channels.cpp#L421-L470) | `inspected` | D1, D2 | Same range checks. | |

## Transforms (`dft.rs`, `dct.rs`; features `fft`, `transforms`)

| purecv item | OpenCV 4.10.0 counterpart (pinned link) | Status | Dims checked | Direction / notes | Issue |
|---|---|---|---|---|---|
| `dft` | [`cv::dft`](https://github.com/opencv/opencv/blob/4.10.0/modules/core/src/dxt.cpp#L3486-L3534) | `diverges` | D1, D2, D7 | → OpenCV (maintainer decision, 2026-10-19): with `flags = 0` return the CCS packing like OpenCV; the current 2-channel result stays available through `DFT_COMPLEX_OUTPUT`, which `dft` already accepts. The inverse must then accept a CCS-packed single-channel input as well. Real input with `flags = 0`: OpenCV returns the **CCS-packed** single-channel spectrum; purecv returns 2-channel complex (= OpenCV with `DFT_COMPLEX_OUTPUT`, identical to 1e-4 in f32, 1e-9 in f64). `DFT_SCALE` (÷ rows·cols; ÷ cols with `DFT_ROWS`), `DFT_ROWS`, `nonzero_rows`, `DFT_REAL_OUTPUT`, complex input, `DFT_INVERSE` without scale, the REAL+COMPLEX error and the u8 error: identical. `DFT_COMPLEX_INPUT` is not a flag (inferred from channels). `f32`/`f64` only in both. | proposed P19 |
| `idft` | [`cv::idft`](https://github.com/opencv/opencv/blob/4.10.0/modules/core/src/dxt.cpp#L3536-L3541), [doc note](https://github.com/opencv/opencv/blob/4.10.0/modules/core/include/opencv2/core.hpp#L2284-L2285) | `diverges` | D1, D7 | → OpenCV. OpenCV: `idft(src, flags)` = `dft(src, flags \| DFT_INVERSE)`, "none of dft and idft scales the result by default" (→ 12·src on the 3×4 fixture). purecv always adds `DFT_SCALE` (→ src). | proposed P20 |
| `get_optimal_dft_size` | [`getOptimalDFTSize`](https://github.com/opencv/opencv/blob/4.10.0/modules/core/src/dxt.cpp#L4625-L4641) | `diverges` | D1, D7 | → OpenCV. 16 sizes in `[1, 2³⁰+1]` identical (65537 → 65610). Edges: 0 → purecv 0 (OpenCV 1); negative → 0 (OpenCV −1); ≥ 2125764000 → computed (OpenCV −1, beyond its table). | proposed P13 |
| `dct`, `idct` | [`cv::dct`, `cv::idct`](https://github.com/opencv/opencv/blob/4.10.0/modules/core/src/dxt.cpp#L4408-L4437) | `measured` | D1, D2, D7 | 1×8 and 4×4 forward, inverse and round trip identical (1e-9 against f64, 1e-5 against f32). Output is always `Matrix<f64>` (OpenCV keeps the depth); no flags (`DCT_ROWS` missing, `idct` is the inverse). Odd sizes: OpenCV errors ([L4320](https://github.com/opencv/opencv/blob/4.10.0/modules/core/src/dxt.cpp#L4320)), purecv computes the O(N²) transform. Integer input accepted by purecv. | |
| `DftFloat` | — | `purecv-only` | | Sealed trait for `f32`/`f64`, the same restriction OpenCV asserts at run time. | |

## Random numbers (`rng.rs`, `std` only)

| purecv item | OpenCV 4.10.0 counterpart (pinned link) | Status | Dims checked | Direction / notes | Issue |
|---|---|---|---|---|---|
| `set_rng_seed(u64)` | [`setRNGSeed(int)`](https://github.com/opencv/opencv/blob/4.10.0/modules/core/src/rand.cpp#L659-L662), [`RNG(uint64)`](https://github.com/opencv/opencv/blob/4.10.0/modules/core/include/opencv2/core/operations.hpp#L394-L395) | `diverges` | D1, D7 | → OpenCV (maintainer decision, 2026-10-19): `cv::RNG` becomes the default generator, xoshiro256** stays as an opt-in alternative. Different generator: xoshiro256** seeded through SplitMix64 vs the 64-bit multiply-with-carry [`RNG_NEXT`](https://github.com/opencv/opencv/blob/4.10.0/modules/core/src/rand.cpp#L63). Seed 0 is the default state `0xffffffff` in OpenCV. Thread-local in both. | proposed P21 |
| `randu` | [`cv::randu`](https://github.com/opencv/opencv/blob/4.10.0/modules/core/src/rand.cpp#L665-L670), [`RNG::fill`](https://github.com/opencv/opencv/blob/4.10.0/modules/core/src/rand.cpp#L391) | `diverges` | D1, D2, D7 | → OpenCV (maintainer decision, 2026-10-19), with the generator switch above. Sequences differ (seed 42, u8 `[0, 256)`: OpenCV `164 63 102 186 144 232 246 91`). Range `[low, high)` and per-channel bounds identical. Integers: OpenCV draws integers, purecv a float then truncates (same range). | proposed P21 |
| `randn` | [`cv::randn`](https://github.com/opencv/opencv/blob/4.10.0/modules/core/src/rand.cpp#L672-L677), [ziggurat](https://github.com/opencv/opencv/blob/4.10.0/modules/core/src/rand.cpp#L236-L310) | `diverges` | D1, D4, D7 | → OpenCV (maintainer decision, 2026-10-19) for both: ziggurat on `cv::RNG` by default (xoshiro256** opt-in) and `saturate_cast` for integer destinations. Box–Muller vs ziggurat, sequences differ. Integer destinations: OpenCV `saturate_cast`s; purecv `from_f64().unwrap_or_default()` turns every sample above the type's maximum into **0** (u8, mean 128, σ 200: 1051 of 2000 samples are 0). | proposed P21 |
| `rand_shuffle(&mut [T])` | [`cv::randShuffle`](https://github.com/opencv/opencv/blob/4.10.0/modules/core/src/rand.cpp#L720-L751), [`randShuffle_`](https://github.com/opencv/opencv/blob/4.10.0/modules/core/src/rand.cpp#L683-L715) | `diverges` | D1, D7 | → OpenCV (maintainer decision, 2026-10-19), with the generator switch above; add `iter_factor`. Fisher–Yates vs OpenCV's `swap(arr[rng % n], arr[i])` for every `i`; no `iterFactor`; takes a slice. Seed 42 on `0..10`: OpenCV `5 3 7 1 4 2 8 9 6 0`. | proposed P21 |

## Metrics (`metrics.rs`)

| purecv item | OpenCV 4.10.0 counterpart (pinned link) | Status | Dims checked | Direction / notes | Issue |
|---|---|---|---|---|---|
| `psnr` | [`cv::PSNR`](https://github.com/opencv/opencv/blob/4.10.0/modules/core/src/norm.cpp#L1270-L1278) | `diverges` | D1, D7 | → OpenCV. Differing inputs identical (1e-6). Identical inputs: OpenCV `20·log10(R / DBL_EPSILON)` = 361.20; purecv `+inf`. `R` is fixed at 255 (OpenCV parameter, default 255). | proposed P13 |
| `mahalanobis` | [`cv::Mahalanobis`](https://github.com/opencv/opencv/blob/4.10.0/modules/core/src/matmul.dispatch.cpp#L833-L854) | `measured` | D1, D2, D7 | Identical (1e-12). purecv requires column vectors; OpenCV accepts rows. | |

## Seeded rows from the plan

| purecv item | OpenCV 4.10.0 counterpart (pinned link) | Status | Dims checked | Direction / notes | Issue |
|---|---|---|---|---|---|
| `features2d::KeyPoint`, `features2d::DMatch` | [`cv::KeyPoint`](https://github.com/opencv/opencv/blob/4.10.0/modules/core/include/opencv2/core/types.hpp#L744), [`cv::DMatch`](https://github.com/opencv/opencv/blob/4.10.0/modules/core/include/opencv2/core/types.hpp#L841) | `diverges` | D1 | → OpenCV (maintainer decision, 2026-10-19). Core types in OpenCV, defined in `features2d` here (std-gated, while `core` is `no_std`). Moved to `core` with re-exports at the old paths by PR #207 (into `dev`); the row becomes `inspected` once that is merged. The types themselves are audited in B3 (#175). | [#207](https://github.com/webarkit/purecv/pull/207) |

## Missing in purecv (S4)

From the 4.10.0 doc groups [Operations on arrays](https://docs.opencv.org/4.10.0/d2/de8/group__core__array.html),
[Basic structures](https://docs.opencv.org/4.10.0/dc/d84/group__core__basic.html),
[Utility](https://docs.opencv.org/4.10.0/db/de0/group__core__utils.html) and
[Clustering](https://docs.opencv.org/4.10.0/d5/d38/group__core__cluster.html). Roadmap gaps, not bugs.

| purecv item | OpenCV 4.10.0 counterpart (pinned link) | Status | Dims checked | Direction / notes | Issue |
|---|---|---|---|---|---|
| — | [`saturate_cast`](https://github.com/opencv/opencv/blob/4.10.0/modules/core/include/opencv2/core/saturate.hpp#L103-L142) | `missing` | | Internal primitive behind every `diverges` row marked D4 above. | [#189](https://github.com/webarkit/purecv/issues/189) |
| — | [`cvRound`, `cvFloor`, `cvCeil`, `cvIsNaN`, `cvIsInf`](https://github.com/opencv/opencv/blob/4.10.0/modules/core/include/opencv2/core/fast_math.hpp#L200-L341) | `missing` | | `cvRound` is round half to even (`_mm_cvtsd_si32` / `lrint`); part of #189. | [#189](https://github.com/webarkit/purecv/issues/189) |
| — | [`Mat::convertTo(alpha, beta)`](https://github.com/opencv/opencv/blob/4.10.0/modules/core/include/opencv2/core/mat.hpp#L1248) | `missing` | | The scale/shift form of `convert_to`. | [#192](https://github.com/webarkit/purecv/issues/192) |
| — | [`Mat::copyTo(dst, mask)`](https://github.com/opencv/opencv/blob/4.10.0/modules/core/src/copy.cpp#L426), [`cv::copyTo`](https://github.com/opencv/opencv/blob/4.10.0/modules/core/include/opencv2/core.hpp#L1472) | `missing` | | | |
| — | `mask` arguments of [`add`](https://github.com/opencv/opencv/blob/4.10.0/modules/core/include/opencv2/core.hpp#L360), `subtract`, `bitwise_*`, [`mean`](https://github.com/opencv/opencv/blob/4.10.0/modules/core/include/opencv2/core.hpp#L671), [`meanStdDev`](https://github.com/opencv/opencv/blob/4.10.0/modules/core/include/opencv2/core.hpp#L693), [`minMaxLoc`](https://github.com/opencv/opencv/blob/4.10.0/modules/core/include/opencv2/core.hpp#L873) | `missing` | | `norm` and `normalize` do take a mask. | |
| — | `dtype` arguments of [`add`](https://github.com/opencv/opencv/blob/4.10.0/modules/core/include/opencv2/core.hpp#L360), `subtract`, `multiply`, `divide`, [`addWeighted`](https://github.com/opencv/opencv/blob/4.10.0/modules/core/include/opencv2/core.hpp#L515), [`reduce`](https://github.com/opencv/opencv/blob/4.10.0/modules/core/include/opencv2/core.hpp#L970); `scale` of [`multiply`](https://github.com/opencv/opencv/blob/4.10.0/modules/core/include/opencv2/core.hpp#L431)/[`divide`](https://github.com/opencv/opencv/blob/4.10.0/modules/core/include/opencv2/core.hpp#L462) | `missing` | | | |
| — | [`divide(double scale, src2)`](https://github.com/opencv/opencv/blob/4.10.0/modules/core/include/opencv2/core.hpp#L466) (reciprocal) | `missing` | | | |
| — | [`scaleAdd`](https://github.com/opencv/opencv/blob/4.10.0/modules/core/src/matmul.dispatch.cpp#L640) | `missing` | | | |
| — | [`mulTransposed`](https://github.com/opencv/opencv/blob/4.10.0/modules/core/src/matmul.dispatch.cpp#L877) | `missing` | | | |
| — | [`calcCovarMatrix`](https://github.com/opencv/opencv/blob/4.10.0/modules/core/src/matmul.dispatch.cpp#L730) | `missing` | | | |
| — | [`findNonZero`](https://github.com/opencv/opencv/blob/4.10.0/modules/core/src/count_non_zero.dispatch.cpp#L153) | `missing` | | | |
| — | [`reduceArgMin`, `reduceArgMax`](https://github.com/opencv/opencv/blob/4.10.0/modules/core/src/minmax.cpp#L1723-L1728) | `missing` | | | |
| — | [`patchNaNs`](https://github.com/opencv/opencv/blob/4.10.0/modules/core/src/mathfuncs.cpp#L1647) | `missing` | | | |
| — | [`minMaxIdx`](https://github.com/opencv/opencv/blob/4.10.0/modules/core/src/minmax.cpp#L1498) | `missing` | | n-D index form; `min_max_loc` covers 2-D. | |
| — | [`eigen`](https://github.com/opencv/opencv/blob/4.10.0/modules/core/src/lapack.cpp#L1331), [`eigenNonSymmetric`](https://github.com/opencv/opencv/blob/4.10.0/modules/core/include/opencv2/core.hpp#L2056) | `missing` | | `eigen_2x2` exists as a private helper in `imgproc/feature.rs`. | |
| — | [`SVDecomp`](https://github.com/opencv/opencv/blob/4.10.0/modules/core/src/lapack.cpp#L1537), `SVBackSubst`, [`cv::SVD`](https://github.com/opencv/opencv/blob/4.10.0/modules/core/include/opencv2/core.hpp#L2130) | `missing` | | `svd_3x3` exists as a private helper in `calib3d/linalg.rs`. | |
| — | `DECOMP_SVD`, `DECOMP_EIG`, `DECOMP_CHOLESKY`, `DECOMP_QR`, `DECOMP_NORMAL` in [`solve`](https://github.com/opencv/opencv/blob/4.10.0/modules/core/src/lapack.cpp#L1032-L1050) and [`invert`](https://github.com/opencv/opencv/blob/4.10.0/modules/core/src/lapack.cpp#L793-L1017) | `missing` | | `NotImplemented` today. | |
| — | [`PCACompute`, `PCAProject`, `PCABackProject`, `cv::PCA`](https://github.com/opencv/opencv/blob/4.10.0/modules/core/include/opencv2/core.hpp#L2104-L2113) | `missing` | | | |
| — | [`cv::LDA`](https://docs.opencv.org/4.10.0/db/d59/classcv_1_1LDA.html) | `missing` | | | |
| — | [`completeSymm`](https://github.com/opencv/opencv/blob/4.10.0/modules/core/src/matrix_operations.cpp#L283) | `missing` | | | |
| — | [`batchDistance`](https://github.com/opencv/opencv/blob/4.10.0/modules/core/include/opencv2/core.hpp#L779) | `missing` | | `features2d` matches descriptors with its own loop. | |
| — | [`mulSpectrums`](https://github.com/opencv/opencv/blob/4.10.0/modules/core/include/opencv2/core.hpp#L2364) | `missing` | | Companion of `dft`. | |
| — | [`convertFp16`](https://github.com/opencv/opencv/blob/4.10.0/modules/core/include/opencv2/core.hpp#L558), `CV_16F` data | `missing` | | Deprecated in 4.10.0 in favour of `convertTo(CV_16F)`. | |
| — | [`broadcast`](https://github.com/opencv/opencv/blob/4.10.0/modules/core/src/matrix_transform.cpp#L932), [`transposeND`](https://github.com/opencv/opencv/blob/4.10.0/modules/core/src/matrix_transform.cpp#L292) | `missing` | | n-D only. | |
| — | [`norm(src1, src2, type, mask)`](https://github.com/opencv/opencv/blob/4.10.0/modules/core/src/norm.cpp#L1073), `NORM_RELATIVE` | `missing` | | Two-array norm and the relative flag. | proposed P9 |
| — | `NORM_L2SQR`, `NORM_HAMMING`, `NORM_HAMMING2` in [`norm`](https://github.com/opencv/opencv/blob/4.10.0/modules/core/src/norm.cpp#L603-L610) | `missing` | | `NotImplemented` today. | proposed P9 |
| — | [`REDUCE_SUM2`](https://github.com/opencv/opencv/blob/4.10.0/modules/core/include/opencv2/core.hpp#L230) | `missing` | | | proposed P11 |
| — | [`DFT_COMPLEX_INPUT`](https://github.com/opencv/opencv/blob/4.10.0/modules/core/include/opencv2/core/base.hpp#L256), [`DCT_INVERSE`, `DCT_ROWS`](https://github.com/opencv/opencv/blob/4.10.0/modules/core/include/opencv2/core/base.hpp#L258-L263) | `missing` | | `dct` has no flags argument. | |
| — | [`Mat::diag(int d)`](https://github.com/opencv/opencv/blob/4.10.0/modules/core/src/matrix.cpp#L900) (extract a diagonal) | `missing` | | | |
| — | [`Mat::row`, `col`, `rowRange`, `colRange`](https://github.com/opencv/opencv/blob/4.10.0/modules/core/include/opencv2/core/mat.hpp#L1121-L1139), `Mat(roi)`, [`locateROI`, `adjustROI`](https://github.com/opencv/opencv/blob/4.10.0/modules/core/include/opencv2/core/mat.hpp#L1624-L1653) | `missing` | | No views: `Matrix` owns a contiguous buffer. | |
| — | [`Mat::mul`](https://docs.opencv.org/4.10.0/d3/d63/classcv_1_1Mat.html), [`Mat::inv`](https://github.com/opencv/opencv/blob/4.10.0/modules/core/include/opencv2/core/mat.hpp#L1344), [`Mat::t`](https://github.com/opencv/opencv/blob/4.10.0/modules/core/include/opencv2/core/mat.hpp#L1335), `MatExpr` | `missing` | | The free functions `multiply`, `invert`, `transpose` exist. | |
| — | [`Mat::push_back`](https://github.com/opencv/opencv/blob/4.10.0/modules/core/include/opencv2/core/mat.hpp#L1588-L1603), `pop_back`, `resize`, `reserve` | `missing` | | | |
| — | [`Mat::isContinuous`, `elemSize`, `total`, `empty`, `checkVector`](https://github.com/opencv/opencv/blob/4.10.0/modules/core/include/opencv2/core/mat.hpp#L1763-L1858) | `missing` | | `data.len()`, `rows * cols` and `Depth::byte_size` cover them. | |
| — | [`cv::Matx`](https://github.com/opencv/opencv/blob/4.10.0/modules/core/include/opencv2/core/matx.hpp#L99) | `missing` | | `VecN` covers the column case only. | |
| — | [`cv::Complex`](https://github.com/opencv/opencv/blob/4.10.0/modules/core/include/opencv2/core/types.hpp#L79) | `missing` | | `dft` uses 2-channel matrices; `rustfft`'s `Complex` is not re-exported. | |
| — | [`cv::Mat_<T>`](https://docs.opencv.org/4.10.0/d3/d63/classcv_1_1Mat.html), [`cv::SparseMat`](https://github.com/opencv/opencv/blob/4.10.0/modules/core/include/opencv2/core/mat.hpp#L2734), [`cv::UMat`](https://github.com/opencv/opencv/blob/4.10.0/modules/core/include/opencv2/core/mat.hpp#L2432) | `missing` | | `Matrix<T>` plays the `Mat_<T>` role. `UMat` (OpenCL) is out of scope; GPU work is [#132](https://github.com/webarkit/purecv/issues/132). | |
| — | [`cv::RNG`](https://github.com/opencv/opencv/blob/4.10.0/modules/core/include/opencv2/core.hpp#L2903-L3035) object API (`uniform`, `gaussian`, `next`, `operator()`), [`theRNG`](https://github.com/opencv/opencv/blob/4.10.0/modules/core/src/rand.cpp#L654-L657), [`RNG_MT19937`](https://docs.opencv.org/4.10.0/df/d61/classcv_1_1RNG__MT19937.html) | `missing` | | Only the thread-local fill functions exist; `#198`/`#199` need `cv::RNG`'s exact sequence for ORB patterns. | proposed P21 |
| — | [`randShuffle(iterFactor, rng)`](https://github.com/opencv/opencv/blob/4.10.0/modules/core/include/opencv2/core.hpp#L2445) parameters | `missing` | | | proposed P21 |
| — | [`PSNR(..., R)`](https://github.com/opencv/opencv/blob/4.10.0/modules/core/include/opencv2/core.hpp#L772) | `missing` | | Fixed at 255. | proposed P13 |
| — | [`checkRange(quiet, pos)`](https://github.com/opencv/opencv/blob/4.10.0/modules/core/include/opencv2/core.hpp#L1722) | `missing` | | | proposed P8 |
| — | [`partition`](https://docs.opencv.org/4.10.0/d5/d38/group__core__cluster.html) | `missing` | | Clustering group. | |
| — | [`cubeRoot`](https://github.com/opencv/opencv/blob/4.10.0/modules/core/include/opencv2/core/base.hpp#L539), [`fastAtan2`](https://github.com/opencv/opencv/blob/4.10.0/modules/core/include/opencv2/core/base.hpp#L558) | `missing` | | `fastAtan2` is what `phase` would need for bit parity (P7). | proposed P7 |
| — | [`getNumThreads`, `setNumThreads`, `getNumberOfCPUs`, `getThreadNum`](https://docs.opencv.org/4.10.0/db/de0/group__core__utils.html), `parallel_for_` | `missing` | | Rayon's global pool plays this role (`parallel` feature). | |
| — | [`useOptimized`, `setUseOptimized`, `checkHardwareSupport`, `getBuildInformation`](https://docs.opencv.org/4.10.0/db/de0/group__core__utils.html) | `missing` | | Compile-time features (`simd`, `parallel`) instead. | |
| — | [`alignSize`, `alignPtr`, `divUp`, `roundUp`](https://docs.opencv.org/4.10.0/db/de0/group__core__utils.html), `TickMeter` | `missing` | | | |
| — | [`getCPUTickCount`](https://docs.opencv.org/4.10.0/db/de0/group__core__utils.html) | `missing` | | | |
| — | `Rect_` operators (`&`, `\|`, `contains`), `RotatedRect::points`/`boundingRect`, `Point_::dot`/`cross`/`inside`, `Scalar_::mul`/`conj`, `Vec::mul`/`cross` ([types.hpp](https://github.com/opencv/opencv/blob/4.10.0/modules/core/include/opencv2/core/types.hpp), [matx.hpp](https://github.com/opencv/opencv/blob/4.10.0/modules/core/include/opencv2/core/matx.hpp)) | `missing` | | Methods of the basic structures listed in the first table. | |

## Proposed issues

New `diverges` rows, one issue each, titled per S5. **Not filed**: they wait for the maintainer's
approval in the PR review. The maintainer's decisions of 2026-10-19 are applied: P7 is withdrawn
(kept as `intentional`); P12, P13 (including `ones`/`eye`), P19 and P21 follow OpenCV. No `keep?`
row is left. `KeyPoint`/`DMatch` follow OpenCV through PR #207 (see [Seeded rows](#seeded-rows-from-the-plan)).

| # | Title | Rows | Direction | Depends on |
|---|---|---|---|---|
| P1 | `core: add/subtract/multiply on integer types overflow instead of saturating (panic in debug, wrap in release, saturate only with simd)` | `add`, `subtract`, `multiply` (`absdiff` on signed types) | → OpenCV | — |
| P2 | `core: divide differs from OpenCV 4.10.0: integer quotients truncate, float division by zero gives 0` | `divide` | → OpenCV | #189 |
| P3 | `core: *_scalar operands differ from OpenCV 4.10.0: scalar truncated, out-of-range scalar or result becomes 0` | `add_scalar`, `subtract_scalar`, `multiply_scalar`, `divide_scalar` (also `gemm` integer outputs) | → OpenCV | #189 |
| P4 | `core: add_weighted differs from OpenCV 4.10.0: integer outputs truncate and out-of-range sums become 0` | `add_weighted` | → OpenCV | #189 |
| P5 | `core: convert_scale_abs differs from OpenCV 4.10.0: ties round away from zero` | `convert_scale_abs` | → OpenCV | #189 |
| P8 | `core: check_range differs from OpenCV 4.10.0: upper bound is inclusive` | `check_range` | → OpenCV | — |
| P9 | `core: norm differs from OpenCV 4.10.0: L2SQR, HAMMING, HAMMING2, RELATIVE and the two-array form are missing` | `norm` | → OpenCV | — |
| P10 | `core: normalize differs from OpenCV 4.10.0: integer outputs truncate, dtype ignored, alpha/beta not sorted` | `normalize` | → OpenCV | #189 |
| P11 | `core: reduce differs from OpenCV 4.10.0: AVG truncates on integer types, overflowing SUM returns 0, no dtype/SUM2` | `reduce` | → OpenCV | #189 |
| P12 | `core: add has_non_zero matching OpenCV 4.10.0 hasNonZero (has_no_zero tests that every element is non-zero)` | `has_no_zero` | → OpenCV | — |
| P13 | `core: edge semantics differ from OpenCV 4.10.0 (determinant of non-square, mean/sum beyond 4 channels, ones/eye multi-channel, flip_nd negative axis, get_optimal_dft_size edges, psnr of identical inputs)` | `determinant`, `sum`, `mean`, `ones`, `eye`, `flip_nd`, `get_optimal_dft_size`, `psnr` | → OpenCV | — |
| P14 | `core: invert differs from OpenCV 4.10.0: returns the determinant instead of a non-zero flag` | `invert` | → OpenCV | — |
| P15 | `core: transform differs from OpenCV 4.10.0: integer outputs truncate` | `transform` | → OpenCV | #189 |
| P16 | `core: kmeans differs from OpenCV 4.10.0: random-center initialisation stays in a local minimum, criteria not clamped` | `kmeans` | → OpenCV | — |
| P18 | `core: copy_make_border differs from OpenCV 4.10.0: border codes do not follow BorderTypes, oversized borders panic` | `copy_make_border` | → OpenCV | — |
| P19 | `core: dft differs from OpenCV 4.10.0: real input gives complex output where OpenCV packs CCS (keep complex behind DFT_COMPLEX_OUTPUT)` | `dft` | → OpenCV | — |
| P20 | `core: idft differs from OpenCV 4.10.0: DFT_SCALE is always applied` | `idft` | → OpenCV | — |
| P21 | `core: use cv::RNG as the default generator for randu/randn/rand_shuffle (keep xoshiro256** as an option; randn must saturate integer samples)` | `set_rng_seed`, `randu`, `randn`, `rand_shuffle` | → OpenCV | #189; feeds #198/#199 |
| P22 | `core: CV_LOG2 differs from OpenCV 4.10.0: log2(e) instead of ln 2` | `CV_LOG2` | → OpenCV | — |

P7 was withdrawn on 2026-10-19 (`intentional`). P6 and P17 were folded into the `measured` rows for `exp`/`log`/`pow` (≤ 2 ULP) and into P13.

## Notes

- **Re-exports (D1).** `purecv::core` re-exports neither `structural::*` nor the arithmetic items
  `add_scalar`, `subtract_scalar`, `multiply_scalar`, `divide_scalar`, `bitwise_*_scalar`, `min`,
  `max`, `abs_diff`, `min_scalar`, `max_scalar`, `abs_diff_scalar`, `compare`, `compare_scalar`,
  `in_range`, `in_range_scalar`, `add_weighted`, `sqrt`, `exp`, `log`, `pow`,
  `convert_scale_abs`, nor `types::CmpTypes` and `utils::Range`. Callers use the submodule
  paths. Not a parity defect; recorded for the API review.
- **`cvRound` platform artefact.** On x86-64 OpenCV converts NaN, ±inf and |x| ≥ 2³¹ to
  `INT_MIN` before clamping (`_mm_cvtsd_si32`). `convert_to`, `convert_scale_abs` and the
  arithmetic kernels inherit it (NaN → 0 is the useful part; `1e10` → 0 for `convertScaleAbs`
  is not). The rows record the observed values; #189 should define NaN → 0 and clamp ±inf to
  the type's bounds.
- **Python oracle for `convertTo`.** `cv2` does not bind `Mat::convertTo`; `cv2.addWeighted(src,
  alpha, 0, 0, beta, dtype)` runs the same `arithm_op` → `cvtdst` path (the destination is
  written by `getConvertFunc(wtype, dtype)`, the `convertTo` kernel). opencv.js's direct
  `convertTo` agrees on all 25 × 5 × 2 values, which validates the oracle for B2 (`cvt_color`
  and the filters write integer outputs the same way).
- **D6 beyond this audit.** The `simd` feature changes results for u8 `add`/`subtract`; that is
  a parity bug, not a performance note, and belongs with P1. No other row was run under `simd`,
  `--no-default-features --features std` or `wasm32`.

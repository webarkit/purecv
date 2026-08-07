# Changelog - webarkit/purecv

All notable changes to this project will be documented in this file.

## [0.7.0] - 2026-08-07

### ⚙️ Miscellaneous Tasks

- *(core)* Make std an explicit opt-in feature, set MSRV 1.88
- *(ci)* Add no_std build job
- *(core)* Fix formatting in logging.rs (#80)

### 🐛 Bug Fixes

- *(wasm)* Re-enable purecv std feature for the wasm crate

### 📚 Documentation

- Document no_std/embedded support and refresh the README
- *(core)* Document the logging facility in the README (#80)
- *(core)* Add module-level docs for core, imgproc and features (#80)

### 🚀 Features

- *(core)* Compile the core module without the standard library
- *(imgproc)* Compile the imgproc module without the standard library
- *(video,calib3d)* Compile both modules without the standard library
- *(core)* Add OpenCV-style logging facade (#80)
- *(core)* Add basic logger and migrate example applications to logging macros (#80)
- *(core)* Add cv_bail!/cv_err! log-and-return macros (#80)
- *(core)* Add warning logs for solve and solve_pnp_ransac failure cases (#80)
- *(core)* Log input-validation failures in arithm and matrix (#80)
- *(core)* Log input-validation failures across remaining core modules (#80)
- *(video)* Add debug logging inside calc_optical_flow_pyramid_lk (#80)

### 🚜 Refactor

- *(core)* Gate the stdout logger behind the std feature (#80)
- *(imgproc)* Use core::/alloc:: paths for no_std
- *(video,calib3d)* Use core::/alloc:: paths for no_std

### 🧪 Testing

- *(core)* Add build-only no_std smoke-test crate
- *(imgproc)* Exercise gaussian_blur from the no_std smoke crate
- *(video,calib3d)* Smoke-test both modules + document no_std support

## [0.6.1] - 2026-07-08

### 🎨 Styling

- *(examples)* Format rectification.rs example code
- Fix rustdoc formatting (trailing spaces)

### 🐛 Bug Fixes

- *(simd)* Fix imports and clippy warnings with simd feature enabled (#76)

### 💼 Other

- *(wasm)* Remove unnecessary CARGO_HTTP_CHECK_REVOKE override

### 📚 Documentation

- Add PR #78 features to README (remap, undistort, fundamental_mat)
- Improve rustdoc for new calib3d and imgproc geometric functions

### 🕸️ WebAssembly & Emscripten

- *(wasm)* Expose undistort, remap, warp perspective, and add calibration example

### 🚀 Features

- *(calib3d)* Implement camera undistortion, perspective warping, and fundamental matrix estimation (#76)
- *(examples)* Add rectification and perspective warp example (#76)

## [0.6.0] - 2026-06-27

### ⚙️ Miscellaneous Tasks

- Bump version to 0.5.0 in the [workspace.package] section of the Cargo.toml file after releasing

### 📚 Documentation

- *(features2d)* Document differences from OpenCV AKAZE matching and tracking tutorials (#75)
- Improve features2d documentation and update roadmap

### 🚀 Features

- *(features2d)* Implement BFMatcher, DMatch and drawing utilities (#75)

## [0.5.0] - 2026-05-28

### 📚 Documentation

- *(bench)* Document FAST and ORB parallel vs sequential benchmark results (#63)
- *(features2d)* Add fast and orb examples and improve module/global documentation (#64)

### 🕸️ WebAssembly & Emscripten

- *(wasm)* Add features2d exports and SIMD acceleration for ORB and FAST (#71)

### 🚀 Features

- *(features2d)* Add module skeleton and public API (#56)
- *(features2d)* Implement keypoint types and shared utilities
- *(features2d)* Add FAST keypoint detection algorithm
- *(features2d)* Implement complete ORB keypoint and steered BRIEF descriptor pipeline

### 🧪 Testing

- *(imgproc)* Add robust unit tests for generic bilinear resize (#60)

## [0.4.0] - 2026-05-09

### ⚙️ Miscellaneous Tasks

- *(core)* Add calib3d module scaffold
- *(core)* Structure calib3d module with tests and simd stubs

### 🐛 Bug Fixes

- *(calib3d)* Address code review: improve error messages and doc comments
- *(calib3d)* Address PR review comments for pose and homography

### 📚 Documentation

- Add calib3d features and pose estimation example to README (#52)
- Update examples section to list all available examples

### 🕸️ WebAssembly & Emscripten

- *(wasm)* Finalize calib3d bindings and tests (closes #53)
- *(wasm)* Add interactive pose estimation demo and update documentation (#53)

### 🚀 Features

- *(calib3d)* Implement find_homography, rodrigues, solve_pnp, solve_pnp_ransac
- *(calib3d)* Add pose estimation example using solve_pnp and rodrigues (#51)

## [0.3.1] - 2026-05-02

### ⚙️ Miscellaneous Tasks

- *(wasm)* Update CHANGELOG.md, README.md, and package.json to version 0.3.1

### 🐛 Bug Fixes

- *(video)* Fix calc_optical_flow_pyramid_lk doctest using blank image
- *(core, benches, video)* Resolve clippy warnings for data_ptr casts and implicit saturating sub

### 📚 Documentation

- *(core)* Add doc-test example for Scalar::channel_or_default
- *(video)* Improve doc comments on optical_flow example helpers
- *(benchmarks)* Remove accidentally duplicated header from benchmark_results.md
- *(wasm)* Fix outdated api references in readme (fixes #42)

### 🕸️ WebAssembly & Emscripten

- *(wasm)* Add optical flow bindings and webcam demo
- *(wasm)* Expose data_ptr, copy_to, and VecN to JS bindings

### 🚀 Features

- *(core)* Add generic N-dimensional VecN type
- *(core)* Add VecN::new() constructors and Add/Sub<Scalar> for VecN
- *(core)* Add vecn_ops example demonstrating VecN type
- *(core)* Add data_ptr, data_ptr_mut, and copy_to to Matrix<T>
- *(video)* Implement buildOpticalFlowPyramid and calcOpticalFlowPyrLK
- *(video)* Add optical_flow example using real image (butterfly.jpg)
- *(video)* Add multi-frame optical flow example with GIF input
- *(video)* Add parallel and SIMD support to optical flow functions

### 🚜 Refactor

- *(core)* Extract Scalar::channel_or_default, rename verbose test

### 🧪 Testing

- *(video)* Add optical flow benchmarks and update documentation

## [0.3.0] - 2026-04-18

### ⚙️ Miscellaneous Tasks

- *(wasm)* Add license headers to JS/HTML files and finalize benches doc
- Upgrade project license to LGPLv3

### ⚡ Performance

- *(imgproc)* Add benchmarks for corner_harris and hough_transform functions

### ⚡ SIMD Optimizations

- *(simd)* Implement SIMD acceleration for pyramids and morphology

### 🐛 Bug Fixes

- *(examples)* Format println! in hough_transform.rs
- *(benches)* Fix fromatting in imgproc_benches file

### 📚 Documentation

- Update READMEs and CHANGELOG for v0.2.4
- Update README for 0.2 and add Hough Transform / Feature Detection details
- *(bench)* Update benchmark results with hough and feature detection

### 🕸️ WebAssembly & Emscripten

- *(imgproc,wasm)* Implement morphological and pyramid operations #37
- *(wasm)* Add Hough Transform and Feature Detection bindings (#40)
- *(wasm)* Add interactive web examples and demo environment for Hough, Feature Detection, and Pyramids (#40)

### 🚀 Features

- *(imgproc)* Implement corner detection pipeline — cornerSubPix, goodFeaturesToTrack, Harris, Shi-Tomasi, preCornerDetect (#39)
- *(imgproc)* Implement Hough Transform for line and circle detection (#40)

### 🚜 Refactor

- *(hough)* Optimize sorting of detected lines and centers using sort_by_key

## [0.2.3] - 2026-04-05
### Milestone 1: Core Features Enhancement

### 🚀 Features

- *(core)* Add extended mathematical constants CV_E, CV_LN10, and CV_SQRT2

## [0.2.2] - 2026-04-05

### ⚙️ Miscellaneous Tasks

- Ignore build artifacts and target directories in worktrees

### 🐛 Bug Fixes

- *(core)* Gate dft_example behind fft feature in Cargo.toml
- *(core)* Resolve lints and standardize mathematical constants

### 📚 Documentation

- *(core)* Add DFT and LUT usage examples

### 🚀 Features

- *(core)* Add OpenCV mathematical constants
- *(core)* Implement DFT and LUT functions
- *(core)* Expand core operations, solvers, metrics, and transforms

### 🚜 Refactor

- *(core)* Fix formatting

### 🧪 Testing

- *(core)* Add benchmarks for count_non_zero, lut, dft, get_optimal_dft_size

## [0.2.1] - 2026-03-28

### ⚙️ Miscellaneous Tasks

- *(wasm)* Align workspace and npm version to 0.2.1

### 🐛 Bug Fixes

- *(core)* Remove redundant cast and use .copied() in DynamicMatrix
- Linting issue
- *(wasm)* Fix npm publish not including dist-std and dist-simd directories

### 📚 Documentation

- *(simd)* Add per-function SIMD performance documentation
- Add module structure convention for tests.rs and simd.rs

### 🕸️ WebAssembly & Emscripten

- *(core,wasm)* Replace PureCvMatrixF32/U8 with unified Mat type
- *(core,wasm)* Add OpenCV-style MatType constructor to DynamicMatrix and Mat

### 🚀 Features

- *(core)* Implement Scalar improvements and Matrix scalar constructors (closes #21) (#22)

### 🚜 Refactor

- *(core)* Replace panic with Result in matrix type methods
- Fix for formatting issue
- Fix for clippy error
- *(simd)* Split imgproc SIMD helpers into src/imgproc/simd.rs

### 🧪 Testing

- *(core)* Add comprehensive unit tests for Matrix API extensions
- *(core,imgproc)* Add 28 critical unit tests for untested public APIs (#23) (#25)

## [0.2.0] - 2026-03-21

### 🐛 Bug Fixes

- Fmt issue in wasm rust module

### 🚀 Features

- Introduce a WebAssembly module for purecv, including build scripts and a workspace structure.
- Improves to WebAssembly module with dual (standard and SIMD) builds and packaging infrastructure.
- Add GitHub Actions workflows for continuous integration, release management, and package publishing.
- Add a new WebAssembly image processing demo showcasing various filters on a butterfly image.
- Introduce and document the new WebAssembly package, including installation instructions and badge updates in the main README.

### 🚜 Refactor

- *(wasm)* Fix wasm crate metadata and core re-exports
- Improved GitHub Actions workflow for automated releases, including artifact packaging and publishing to Crates.io and NPM.

## [0.1.4] - 2026-03-19

### 🐛 Bug Fixes

- Correct indentation in rust.yml file and fix for wrong command

### 📚 Documentation

- Update README to indicate project is a work in progress

### 🚀 Features

- *(version)* Add the new src/version.rs file
- Add badges to README and update CI workflow for formatting check

### 🚜 Refactor

- Fix in version.rs file for fmt issue
- Fix in version.rs for test failing issue
- *(chore)* Restructure CI workflow to separate code formatting and build steps

## [0.1.3] - 2026-03-17

### ⚡ Performance

- *(imgproc)* Add bilateral_filter and sobel_f32 SIMD benchmarks

### 🐛 Bug Fixes

- *(simd)* Fix simd_dot/simd_sum returning 0.0 and use bool return in macros
- *(simd)* Allow dead code for SimdElement trait and clean up threshold.rs

### 📚 Documentation

- Update README for PR 2 Color + Threshold SIMD changes
- Update README with performance highlights for sobel and bilateral_filter SIMD optimizations

### 🚀 Features

- *(core,imgproc)* Add SIMD acceleration via pulp and full benchmark suite
- *(imgproc)* Add SIMD acceleration for color conversion and threshold
- *(imgproc)* Add threshold example

## [0.1.2] - 2026-03-15

### 🐛 Bug Fixes

- *(core)* Resolve arithm.rs compilation errors and add magnitude/polar functions

### 📚 Documentation

- Add Copilot instructions and project guidelines
- Fix 3-D to 3D in ndarray method doc comments
- Update README with ndarray feature flag and usage examples
- Fix ndarray example to avoid use-after-move
- *(core)* Add documentation and examples to linear algebra functions
- Update contribution guidelines for pre-commit checks and code quality in agents files
- Update README to enhance feature descriptions and clarify dependencies
- Add AGENTS.md for AI coding agents
- Update README to include new functions in linear algebra, sorting, clustering, transforms, and utilities

### 🚀 Features

- Add ndarray interoperability for Matrix via optional feature flag
- *(core)* Implement gemm, trace, dot, cross, check_range, and set_identity
- *(core)* Add randu, randn, and set_rng_seed
- *(arithm)* Add matrix transformation and perspective transformation functions
- *(core)* Add solve_poly, sort, sort_idx, and kmeans functions

### 🚜 Refactor

- *(core)* Fix formatting violations from cargo fmt
- Fix formatting in core arithm
- Fix clippy needless_range_loop warnings in core arithm
- Apply cargo fmt to fix CI formatting check

## [0.1.1] - 2026-03-14

### ⚙️ Miscellaneous Tasks

- Fix clippy warnings in src/core/matrix.rs
- Update repository URL in Cargo.toml and add changelog and CI release configuration files
- Pushing correct github release script
- Add simd feature to Cargo.toml and update CI workflow for dev branch

### 📚 Documentation

- Add CONTRIBUTING and MAINTAINERS guidelines

### 🚀 Features

- Implement MatType and DataType for OpenCV parity

## [0.1.0] - 2026-03-10

## Overview
This changelog documents all changes that occurred from the initial commit through the release of v0.1.0 for the webarkit/purecv repository - a pure Rust computer vision library focusing on the core and imgproc modules of OpenCV.

---

## 1. Project Foundation & Structure

- ✅ Initialized core project structure with Cargo workspace setup
- ✅ Established LGPL-2.1-or-later license
- ✅ Created comprehensive header documentation across all source files
- ✅ Set up GitHub Actions CI/CD workflows (Rust.yml) with formatting checks, build tests, and clippy linting

---

## 2. Core Module Enhancements

### Matrix Operations

- ✅ Implemented `Matrix<T>` generic 2D matrix with row-major memory layout
- ✅ Added factory methods: `zeros()`, `ones()`, `eye()`, `diag()`
- ✅ Added `from_size()` constructor for convenient size-based matrix creation
- ✅ Implemented `convert_to<U>()` for type casting with precision handling
- ✅ Added accessor methods: `get()`, `at()` with i32 indexing, `get_mut()`, `at_mut()`
- ✅ Implemented `flat_index()` for efficient 1D-2D coordinate conversion

### Arithmetic Operations

- ✅ Implemented core arithmetic: `add()`, `subtract()`, `multiply()`, `divide()`
- ✅ Added mathematical functions: `sqrt()`, `exp()`, `log()`, `pow()`
- ✅ Implemented bitwise operations: `bitwise_and()`, `bitwise_or()`, `bitwise_xor()`, `bitwise_not()`
- ✅ Added `absdiff()` for absolute difference
- ✅ Implemented `add_weighted()` for weighted sums
- ✅ Added `convert_scale_abs()` for scaling and absolute value conversion
- ✅ Feature-gated SIMD support with Rayon for parallel operations

### Data Types

- ✅ Added `BorderTypes` enum with REFLECT_101, REPLICATE, WRAP, CONSTANT, etc.

### Statistical Functions

- ✅ Implemented `sum()` and `mean()` per-channel calculations
- ✅ Added `min_max_loc()` for finding min/max values and their locations
- ✅ Implemented `mean_std_dev()` for standard deviation calculations
- ✅ Implemented `norm()` with support for INF, L1, L2 norms
- ✅ Added `normalize()` for MINMAX and norm-based normalization

### Structural Operations

- ✅ Implemented `flip()` for vertical, horizontal, and both-axes flipping
- ✅ Added `transpose()` for matrix transposition
- ✅ Implemented `split()` for channel separation
- ✅ Added `merge()` for channel combination
- ✅ Implemented `repeat()` for pattern repetition
- ✅ Added `reshape()` for matrix dimension changes
- ✅ Implemented `copy_make_border()` for border padding
- ✅ Added `hconcat()` and `vconcat()` for horizontal/vertical concatenation
- ✅ Implemented `mixChannels()` for advanced channel mixing

---

## 3. Image Processing (imgproc) Module

### Color Conversions

- ✅ Implemented `cvt_color()` as main wrapper function
- ✅ Added color space conversions:
    - RGB ↔ Grayscale
    - BGR ↔ Grayscale
    - RGBA/BGRA ↔ Grayscale
    - Grayscale ↔ RGB/BGR/RGBA/BGRA
- ✅ Added `ColorConversionCode` enum for OpenCV-style API

### Filtering Operations

- ✅ Implemented `blur()` and `box_filter()` for box filtering
- ✅ Added `gaussian_blur()` with kernel generation
- ✅ Implemented `median_blur()` for non-linear filtering
- ✅ Added `bilateral_filter()` for edge-preserving smoothing

### Derivative Operations

- ✅ Implemented `sobel()` for Sobel derivatives
- ✅ Added `scharr()` for Scharr operator
- ✅ Implemented `laplacian()` for Laplacian computation
- ✅ Added `get_sobel_kernels()` and `get_deriv_kernel()` utilities

### Edge Detection

- ✅ Implemented `canny()` edge detector with:
    - Gradient computation using Sobel
    - Non-maximum suppression
    - Hysteresis thresholding

### Image Thresholding

- ✅ Implemented `threshold()` function with types:
    - THRESH_BINARY
    - THRESH_BINARY_INV
    - THRESH_TRUNC
    - THRESH_TOZERO
    - THRESH_TOZERO_INV

---

## 4. Performance & Optimization

### Parallelization

- ✅ Integrated Rayon for multi-core processing
- ✅ Feature-gated parallel execution with fallback to sequential
- ✅ Implemented parallel iterator patterns for memory-efficient batch processing

### Algorithmic Optimizations

- ✅ Added `fast_deriv_3x3()` for optimized 3x3 derivative computation
- ✅ Implemented separated filter passes for efficiency
- ✅ Added interior "fast path" with boundary "slow path" separation
- ✅ Optimized chunk-based processing to reduce allocations

### SIMD Support

- ✅ Optional Pulp integration for portable SIMD
- ✅ Feature-gated SIMD dispatch with fallback
- ✅ Auto-vectorization-friendly code patterns

---

## 5. Utility Functions

- ✅ Implemented `border_interpolate()` with multiple border types
- ✅ Added `get_log_level()` and `set_log_level()` for debugging
- ✅ Implemented `ParIterFallback` trait for seamless feature-gated compilation

---

## 6. Testing & Documentation

- ✅ Comprehensive unit test coverage across all modules
- ✅ Integration tests verifying OpenCV parity
- ✅ Benchmark suite with Criterion for performance evaluation
- ✅ Created example programs:
    - arithmetic
    - color_conversion
    - filters
    - structural_ops
- ✅ Added benchmark results documentation

---

## 7. Build & CI/CD

- ✅ Configured Cargo.toml with proper features:
    - parallel
    - wasm
    - simd
- ✅ Set optimization levels:
    - opt-level=3
    - lto=true
    - codegen-units=1
- ✅ Configured panic=abort for WASM/system safety
- ✅ GitHub Actions workflow for automated testing and linting

---

## 8. Documentation & Community

- ✅ Updated README with philosophy, features, and usage examples
- ✅ Added roadmap indicating Phase 1 (Core Foundation) completion
- ✅ Added CI badge for project health visibility
- ✅ Organized examples and data directories

---

## Summary

This development represents a **complete foundation for a production-grade pure Rust computer vision library**, with emphasis on:

- **Safety**: Leveraging Rust's memory safety guarantees
- **Performance**: Through parallelization, SIMD support, and algorithmic optimizations
- **OpenCV API Compatibility**: Familiar interfaces for developers transitioning from OpenCV
- **Extensibility**: Well-structured codebase for future module additions

**v0.1.0** marks the successful completion of **Phase 1 (Core Foundation)**, establishing the core matrix operations, essential image processing functions, and the architectural foundation for future development.
# PureCV

![PureCv Banner](./assets/purecv_banner.png)

[![Rust CI](https://github.com/webarkit/purecv/actions/workflows/ci.yml/badge.svg)](https://github.com/webarkit/purecv/actions/workflows/ci.yml)
[![Crates.io](https://img.shields.io/crates/v/purecv.svg)](https://crates.io/crates/purecv)
[![Crates.io Downloads](https://img.shields.io/crates/d/purecv.svg)](https://crates.io/crates/purecv)
[![NPM version](https://img.shields.io/npm/v/@webarkit/purecv-wasm.svg)](https://www.npmjs.com/package/@webarkit/purecv-wasm)
[![GitHub Stars](https://img.shields.io/github/stars/webarkit/purecv.svg?style=social)](https://github.com/webarkit/purecv/stargazers)
[![GitHub Forks](https://img.shields.io/github/forks/webarkit/purecv.svg?style=social)](https://github.com/webarkit/purecv/network/members)

A high-performance, **pure Rust** computer vision library focusing on the `core` and `imgproc` modules of OpenCV. **PureCV** is built from the ground up to be memory-safe, thread-safe, and highly portable without the overhead of C++ FFI.

> This project is currently a **Work in Progress**. While most core and imgproc features have been implemented, the library is not yet stable, and bugs may occur. We are actively optimizing and expanding the feature set.

## 🎯 Philosophy

Unlike existing wrappers, **PureCV** is a native rewrite. It aims to provide:

* **Zero-FFI:** No complex linking or C++ toolchain requirements.
* **Memory Safety:** Elimination of segmentation faults and buffer overflows via Rust's ownership model.
* **Modern Parallelism:** Native integration with **Rayon** for effortless multi-core processing.
* **Portable SIMD:** Optional SIMD acceleration via [`pulp`](https://crates.io/crates/pulp) — auto-detects x86 SSE/AVX, ARM NEON, and WASM `simd128` at runtime. Zero `unsafe`, zero `#[cfg(target_arch)]`.

## ✨ Features

### `purecv-core`
- **Matrix Operations:** Multi-dimensional `Matrix<T>` with support for common arithmetic (`add`, `subtract`, `multiply`, `divide`) and bitwise logic (`bitwise_and`, `bitwise_or`, `bitwise_xor`, `bitwise_not`). Matrix and scalar variants for all operations.
- **Factory Methods:** Intuitive initialization with `zeros`, `ones`, `eye`, and `diag`.
- **Scalar constructors:** `Matrix::new_with_scalar` (fill all pixels from a `Scalar<T>`), `new_with_scalar_from_size`, and `new_with_scalar_typed_from_size`. `set_to` and `set_to_masked` assign a `Scalar<T>` to every pixel (optionally masked); channels beyond 4 default to `T::default()`.
- **Scalar type:** `Scalar<T>` — a 4-channel value — now supports `Index`/`IndexMut` for channel access, `from_array`/`to_array`, `From<[T;4]>` and `From<T>` conversions, and a `map()` helper for per-channel type transforms. Arithmetic traits: per-channel `Add`/`Sub`; `Mul<T>`/`Mul<Scalar<T>>` for scaling and element-wise multiply; safe `Div<T>`/`Div<Scalar<T>>` (returns zero on divide-by-zero); `checked_div()` returning `Result` for integer types.
- **Comparison:** `compare`, `compare_scalar`, `min`, `max`, `abs_diff`, `in_range`.
- **Structural:** `flip`, `rotate`, `transpose`, `repeat`, `reshape`, `hconcat`, `vconcat`, `copy_make_border`, `extract_channel`, `insert_channel`.
- **Math:** `sqrt`, `exp`, `log`, `pow`, `magnitude`, `phase`, `cart_to_polar`, `polar_to_cart`, `convert_scale_abs`.
- **Stats:** `sum`, `mean`, `mean_std_dev`, `min_max_loc`, `norm`, `normalize`, `count_non_zero`, `reduce`.
- **Metrics:** `psnr` (Peak Signal-to-Noise Ratio), `mahalanobis` (statistical distance).
- **Linear Algebra:** `gemm`, `dot`, `cross`, `trace`, `determinant`, `invert`, `solve`, `solve_poly`, `solve_quadratic`, `solve_cubic`, `set_identity`.
- **Sorting:** `sort`, `sort_idx` with configurable row/column and ascending/descending flags.
- **Clustering:** `kmeans` with random, k-means++, and user-supplied initialization strategies.
- **Transforms:** `transform` (per-element matrix transformation), `perspective_transform` (projective / homography mapping).
- **Random Number Generation:** `randu` (uniform distribution), `randn` (normal/Gaussian distribution), `set_rng_seed`.
- **Channel Management:** `split`, `merge`, `mix_channels`.
- **Utilities:** `add_weighted`, `check_range`, `absdiff`, `get_tick_count`, `get_tick_frequency`.
- **Mathematical Constants:** OpenCV-compatible constants — `CV_PI`, `CV_PI_2`, `CV_2PI`, `CV_PI_4`, `CV_LOG2`, `CV_LN2`, `CV_E`, `CV_LN10`, `CV_SQRT2` — backed by `std::f64::consts` for maximum precision.
- **ndarray Interop:** Optional, zero-cost conversions to/from `ndarray::Array3` via the `ndarray` feature flag.
- **SIMD Acceleration** (`simd` feature): Trait-based dispatch via `pulp` for `f32`, `f64`, and `u8` types. Accelerated operations include `add`, `sub`, `mul`, `div`, `min`, `max`, `sqrt`, `dot`, `sum`, `add_weighted`, `convert_scale_abs`, `magnitude`, `simd_row_min_max`, `simd_min_max_col`, and `simd_gaussian_5tap_h/v`. Falls back to scalar loops at zero cost when disabled.

### `purecv-imgproc`
- **Color Conversions:** High-performance `cvt_color` supporting RGB, BGR, Gray, RGBA, BGRA and more. Up to **6.6× speedup** with Parallel + SIMD. SIMD-accelerated paths (`simd` feature) use fixed-point integer arithmetic (coefficients 77/150/29 ≈ 0.299/0.587/0.114 × 256) for all `*_to_gray` conversions — portable to x86 SSE/AVX, ARM NEON, and WASM `simd128` via `pulp`.
- **Edge Detection:** `canny`, `sobel`, `scharr`, `laplacian`. Optimized `fast_deriv_3x3` kernel delivers up to **12× speedup** with Parallel. For `f32` inputs, the `pulp`-powered `simd_deriv_3x3_row_f32` interior kernel adds a further **1.5× boost**, reaching **22× total speedup** (28.59 ms → 1.28 ms) with Parallel + SIMD — the highest combined speedup in the project.
- **Filtering:** `blur`, `box_filter`, `gaussian_blur`, `median_blur`, `bilateral_filter`. The bilateral filter achieves **7.1× speedup** with Parallel (1.43 s → 202 ms on 512×512); SIMD provides no additional gain due to the non-vectorizable per-pixel exponential weight computation.
- **Morphology:** `erode`, `dilate`, `morph_op` (supports Rect, Cross, Ellipse kernels) and `get_structuring_element`. Features a **separable SIMD fast-path** for rectangular kernels using `simd_row_min_max` and `simd_min_max_col` for `f32`, `f64`, and `u8`.
- **Pyramids:** `pyr_down`, `pyr_up` (Gaussian 5x5 kernel), and `build_pyramid`. `pyr_down` is fully SIMD-accelerated via `simd_gaussian_5tap_h` and `simd_gaussian_5tap_v`, providing significant speedups for multi-channel images.
- **Thresholding:** `threshold` with all 5 OpenCV-compatible types (`BINARY`, `BINARY_INV`, `TRUNC`, `TOZERO`, `TOZERO_INV`). SIMD-accelerated fast path for `u8`, `f32`, and `f64` via the `SimdElement::simd_threshold()` trait method. Works seamlessly with `parallel` feature for row-level Rayon dispatch.

## 🚀 Getting Started

### Installation

Add the following to your `Cargo.toml`:

```toml
[dependencies]
purecv = "0.1"
```

### Feature Flags

| Flag | Default | Description |
|------|---------|-------------|
| `std` | ✅ | Standard library support |
| `parallel` | ✅ | Multi-core parallelism via **Rayon** |
| `ndarray` | ❌ | Interop with the `ndarray` crate (zero-cost views & ownership transfers) |
| `simd` | ❌ | SIMD acceleration via [`pulp`](https://crates.io/crates/pulp) (x86 SSE/AVX, ARM NEON, WASM `simd128`) |
| `wasm` | ❌ | WebAssembly-specific optimizations |

To enable the `ndarray` feature:

```toml
[dependencies]
purecv = { version = "0.1", features = ["ndarray"] }
```

To enable SIMD + Parallel for maximum performance:

```toml
[dependencies]
purecv = { version = "0.1", features = ["parallel", "simd"] }
```

### Usage Example

```rust
use purecv::core::{Matrix, Size, Scalar};
use purecv::imgproc::{cvt_color, ColorConversionCodes};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create a 3-channel matrix initialized to ones
    let mat = Matrix::<f32>::ones(480, 640, 3);

    // Create an identity matrix
    let identity = Matrix::<f32>::eye(3, 3, 1);

    // --- Scalar API ---

    // Build a Scalar from individual channels or a single broadcast value
    let blue = Scalar::new(255.0f32, 0.0, 0.0, 0.0);
    let gray = Scalar::all(128.0f32);   // all four channels = 128

    // Index channels directly
    assert_eq!(blue[0], 255.0);

    // Conversions
    let from_arr: Scalar<f32> = [1.0, 2.0, 3.0, 4.0].into();
    let arr = from_arr.to_array();          // → [1.0, 2.0, 3.0, 4.0]

    // Per-channel arithmetic
    let a = Scalar::new(10.0f32, 20.0, 30.0, 40.0);
    let b = Scalar::new(1.0f32,  2.0,  3.0,  4.0);
    let sum   = a + b;                      // per-channel add
    let diff  = a - b;                      // per-channel sub
    let scaled = a * 2.0f32;               // broadcast multiply
    let prod  = a * b;                      // element-wise multiply
    let div   = a / 2.0f32;               // broadcast divide (zero-safe)

    // Map channels to another type
    let as_u8: Scalar<u8> = a.map(|x| x as u8);

    // --- Matrix scalar constructors ---

    // Fill an entire matrix with a constant Scalar value
    let filled = Matrix::<f32>::new_with_scalar(480, 640, 3, blue);

    // Use set_to to overwrite an existing matrix
    let mut mat2 = Matrix::<f32>::zeros(480, 640, 3);
    mat2.set_to(gray);

    println!("Matrix size: {}x{}", mat.cols, mat.rows);
    Ok(())
}
```

### ndarray Interoperability

With the `ndarray` feature enabled, you can convert between `Matrix<T>` and `ndarray::Array3<T>`:

```rust
use purecv::core::Matrix;

// Matrix → ndarray (zero-cost view)
let mat = Matrix::<f32>::ones(480, 640, 3);
let view = mat.as_ndarray_view(); // ArrayView3<f32>, shape (480, 640, 3)

// Matrix → ndarray (ownership transfer)
let mat2 = Matrix::<f32>::ones(480, 640, 3);
let arr = mat2.into_ndarray();

// ndarray → Matrix (guarantees contiguous C-order layout for SIMD/WASM)
let mat3 = Matrix::from_ndarray(arr);

// Also works via the From trait
let arr2 = ndarray::Array3::<f32>::zeros((100, 100, 3));
let mat4: Matrix<f32> = Matrix::from(arr2);
```

### WASM Package for Browsers & Node.js

PureCV provides a compiled WebAssembly package via `wasm-bindgen` enabling access to core matrix operations, thresholds, filters, and derivatives directly from JavaScript/TypeScript.

This includes both a **standard build** for maximum compatibility and a **SIMD-optimized build** for massive performance gains in modern browsers.

```bash
npm install @webarkit/purecv-wasm
```

See the [WebAssembly documentation](crates/wasm/README.md) for more usage examples and API details.

### Running Examples

Explore the capabilities of PureCV by running the provided examples:

```bash
# Basic matrix arithmetic
cargo run --example arithmetic

# Structural operations (flip, rotate, split/merge)
cargo run --example structural_ops

# Color conversion (RGB to Grayscale)
cargo run --example color_conversion

# Thresholding — all 5 types (BINARY, BINARY_INV, TRUNC, TOZERO, TOZERO_INV)
cargo run --example threshold

# Image filters (blur, gaussian, canny, sobel, …) — requires examples/data/butterfly.jpg
cargo run --example filters

# Morphological operations (erode, dilate, morph_op)
cargo run --example morphology

# Gaussian pyramids (pyr_down, pyr_up)
cargo run --example pyramids
```

## 🧪 Testing & Benchmarking

### Running Tests
PureCV uses a comprehensive suite of unit tests to ensure correctness and parity with OpenCV. The test suite currently includes **242 unit tests** covering:

- **Core module:** Matrix factories, scalar arithmetic variants, bitwise scalar ops, min/max, comparison ops (`compare`, `in_range`), reduction (`reduce`, `count_non_zero`), polar/cartesian conversions, linear algebra (`determinant`, `invert`, `solve`), channel ops (`extract_channel`, `insert_channel`), `DynamicMatrix`, transforms, sorting, clustering, and RNG.
- **Imgproc module:** Filters, derivatives, edge detection, color conversions (including gray-to-RGB/BGR/RGBA/BGRA), thresholding, morphology (`erode`, `dilate`), pyramids (`pyr_down`, `pyr_up`), and kernel helpers (`get_gaussian_kernel`, `get_sobel_kernels`).

```bash
# Run all tests
cargo test
```

### Running Benchmarks
Performance is a core focus. Benchmarks are available for `arithm`, `imgproc`, and `structural` modules across four configurations:

```bash
# Standard (sequential, no SIMD)
cargo bench --no-default-features

# SIMD Only (sequential + auto-vectorization)
RUSTFLAGS="-C target-cpu=native" cargo bench --no-default-features

# Parallel (Rayon multi-threading)
cargo bench --features parallel

# Parallel + SIMD (maximum throughput)
RUSTFLAGS="-C target-cpu=native" cargo bench --features parallel
```

#### Key Performance Highlights (1024×1024 matrices, *updated 2026-03-17*)

| Operation | Standard | Parallel + SIMD | Speedup |
|-----------|----------|-----------------|---------|
| `cvt_color_rgb2gray` | 2.66 ms | **404 µs** | 6.6× |
| `sobel_3x3` (generic) | 22.79 ms | **1.87 ms** | 12× |
| `sobel_3x3_f32_dx` ★ | 28.59 ms | **1.28 ms** | **22×** |
| `sobel_3x3_f32_dy` ★ | 26.24 ms | **1.27 ms** | **21×** |
| `bilateral_filter` (512×512) | 1.43 s | **202 ms** | 7.1× |
| `laplacian_3x3` | 45.91 ms | **4.44 ms** | 10.4× |
| `dot` | 997 µs | **157 µs** | 6.4× |
| `gemm_256×256` | 15.71 ms | **4.40 ms** | 3.7× |
| `canny` | 57.61 ms | **12.54 ms** | 4.6× |

> ★ Uses non-zero sinusoidal data to exercise the `simd_deriv_3x3_row_f32` SIMD kernel. Best combined speedup in the project.
>
> Full results in [`benches/benchmark_results.md`](./benches/benchmark_results.md)

## 🗺 Roadmap

- [x] **Phase 1: Core Foundation** - Matrix types, arithmetic, geometric utilities, and basic structural transforms.
- [x] **Phase 2: Performance** - SIMD acceleration via `pulp`, Rayon parallelism, and Criterion benchmarking across 32 operations.
  - [x] PR 1 — SIMD infra + `arithm` kernels (`add`, `sub`, `mul`, `div`, `dot`, `magnitude`, `add_weighted`, `convert_scale_abs`, `sqrt`, `min`, `max`, `sum`).
  - [x] PR 2 — Color + Threshold SIMD: fixed-point `cvt_color_*_to_gray` kernels, `simd_threshold()` for all 5 types on `u8`/`f32`/`f64`, new `threshold` example.
  - [x] PR 3 — Derivatives SIMD: `fast_deriv_3x3` interior SIMD pass (`simd_deriv_3x3_row_f32`) achieving **22× speedup** on `sobel_3x3_f32`; new benchmarks for `sobel_3x3_f32_dx/dy` and `bilateral_filter`.
  - [x] PR 4 — Pyramids + Morphology SIMD: Separable 5-tap Gaussian kernels for pyramids and separable min/max kernels for rectangular morphology; new `pyramids` and `morphology` examples.
- [x] **Phase 3: WebAssembly** - `wasm-bindgen` wrappers, `wasm-pack` build, CI matrix with `wasm32-unknown-unknown` + `simd128`.
- [x] **Phase 4: Image Processing** - Advanced filtering, convolutions, morphology, pyramids and feature detection.
- [x] **Visual examples** — Load real images, apply `threshold`, `cvt_color`, `erode`/`dilate` and `pyr_down`, save PNG output (see `examples/`).

## 📄 License

This project is licensed under the LGPL-3.0 License.

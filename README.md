# PureCV

![PureCv Banner](./assets/purecv_banner.png)

[![Rust CI](https://github.com/webarkit/purecv/actions/workflows/rust.yml/badge.svg)](https://github.com/webarkit/purecv/actions/workflows/rust.yml)

A high-performance, **pure Rust** computer vision library focusing on the `core` and `imgproc` modules of OpenCV. **PureCV** is built from the ground up to be memory-safe, thread-safe, and highly portable without the overhead of C++ FFI.

## 🎯 Philosophy

Unlike existing wrappers, **PureCV** is a native rewrite. It aims to provide:

* **Zero-FFI:** No complex linking or C++ toolchain requirements.
* **Memory Safety:** Elimination of segmentation faults and buffer overflows via Rust's ownership model.
* **Modern Parallelism:** Native integration with **Rayon** for effortless multi-core processing.
* **WASM & Native SIMD:** Optimized for the web and high-performance native architectures using explicit SIMD (via `pulp`).

## ✨ Features

### `purecv-core`
- **Matrix Operations:** Multi-dimensional `Matrix<T>` with support for common arithmetic (`add`, `sub`, `mul`, `div`) and bitwise logic.
- **Factory Methods:** Intuitive initialization with `zeros`, `ones`, `eye`, and `diag`.
- **Structural:** `flip`, `rotate`, `transpose`, `repeat`, `reshape`, `hconcat`, `vconcat`.
- **Math & Stats:** `sqrt`, `exp`, `log`, `pow`, `sum`, `mean`, `minMaxLoc`, `norm`.
- **Channel Management:** `split`, `merge`, `mixChannels`.
- **ndarray Interop:** Optional, zero-cost conversions to/from `ndarray::Array3` via the `ndarray` feature flag.

### `purecv-imgproc`
- **Color Conversions:** High-performance `cvt_color` supporting RGB, BGR, Gray, and more.
- **Filtering:** (In Progress) Fast convolutions and image filters.

## 🚀 Getting Started

### Installation

Add the following to your `Cargo.toml`:

```toml
[dependencies]
purecv = { git = "https://github.com/webarkit/purecv" }
```

### Feature Flags

| Flag | Default | Description |
|------|---------|-------------|
| `std` | ✅ | Standard library support |
| `parallel` | ✅ | Multi-core parallelism via **Rayon** |
| `ndarray` | ❌ | Interop with the `ndarray` crate (zero-cost views & ownership transfers) |
| `simd` | ❌ | Explicit SIMD optimizations |
| `wasm` | ❌ | WebAssembly-specific optimizations |

To enable the `ndarray` feature:

```toml
[dependencies]
purecv = { git = "https://github.com/webarkit/purecv", features = ["ndarray"] }
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

### Running Examples

Explore the capabilities of PureCV by running the provided examples:

```bash
# Basic matrix arithmetic
cargo run --example arithmetic

# Structural operations (flip, rotate, split/merge)
cargo run --example structural_ops

# Color conversion (RGB to Grayscale)
cargo run --example color_conversion
```

## 🧪 Testing & Benchmarking

### Running Tests
PureCV uses a comprehensive suite of unit tests to ensure correctness and parity with OpenCV.

```bash
# Run all tests
cargo test
```

### Running Benchmarks
Performance is a core focus. You can run benchmarks to see the impact of SIMD and multi-threading on your architecture:

```bash
# Run all benchmarks
cargo bench
```

## 🗺 Roadmap

- [x] **Phase 1: Core Foundation** - Matrix types, arithmetic, geometric utilities, and basic structural transforms.
- [/] **Phase 2: Performance** - SIMD optimizations and benchmarking vs OpenCV C++.
- [ ] **Phase 3: WebAssembly** - Specialized wrappers and multi-threading for the web.
- [ ] **Phase 4: Image Processing** - Advanced filtering, convolutions, and feature detection.

## 📄 License

This project is licensed under the LGPL-3.0 License.

# PureCV

A high-performance, **pure Rust** computer vision library focusing on the `core` and `imgproc` modules of OpenCV. **PureCV** is built from the ground up to be memory-safe, thread-safe, and highly portable without the overhead of C++ FFI.

[Image of computer vision image processing pipeline]

## 🎯 Philosophy

Unlike existing wrappers, **PureCV** is a native rewrite. It aims to provide:

* **Zero-FFI:** No complex linking or C++ toolchain requirements.
* **Memory Safety:** Elimination of segmentation faults and buffer overflows via Rust's ownership model.
* **Modern Parallelism:** Native integration with **Rayon** for effortless multi-core processing.
* **WASM & Native SIMD:** Optimized for the web and high-performance native architectures using explicit SIMD (via `pulp`).

## 🛠 Project Structure

* `purecv-core`: Foundational structures like `Matrix<T>`, `Point`, and `Scalar`.
* `purecv-imgproc`: Image processing algorithms (filtering, color conversion, geometric transforms).

## 🚀 Getting Started

### Installation

Add the following to your `Cargo.toml`:

```toml
[dependencies]
purecv = { git = "https://github.com/webarkit/purecv" }
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
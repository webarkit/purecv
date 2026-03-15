# AGENTS.md

## Project Overview

PureCV is a **pure Rust** rewrite of OpenCV's `core` and `imgproc` modules. The central type is `Matrix<T>` (row-major, contiguous `Vec<T>` buffer) in `src/core/matrix.rs`. No C/C++ FFI is allowed — every algorithm is implemented from scratch in safe Rust.

## Architecture

```
src/
├── core/           # Matrix type, arithmetic, structural ops, types, error handling
│   ├── matrix.rs   # Matrix<T> struct — the cv::Mat equivalent
│   ├── arithm.rs   # Element-wise ops, linear algebra, stats (add, norm, gemm, …)
│   ├── structural.rs # flip, rotate, transpose, split, merge, hconcat, vconcat
│   ├── types.rs    # Point, Size, Rect, Scalar, BorderTypes, enums
│   ├── error.rs    # PureCvError enum + Result<T> alias
│   └── utils.rs    # border_interpolate, ParIterFallback (non-parallel shim)
├── imgproc/        # Image processing algorithms
│   ├── color.rs    # cvt_color (RGB/BGR/Gray/RGBA conversions)
│   ├── filter.rs   # blur, box_filter, gaussian_blur, median_blur, bilateral_filter
│   ├── edge.rs     # canny, sobel, scharr, laplacian
│   ├── derivatives.rs
│   └── threshold.rs
├── features/       # Placeholder for feature detectors (FAST, ORB — not yet implemented)
└── lib.rs          # Re-exports core, imgproc, features; defines prelude
```

- **Reference sources** live in `cpp_ref/opencv/` — consult when porting algorithms.
- **Tests** are co-located: `src/core/tests.rs`, `src/imgproc/tests.rs`.
- **Benchmarks** use Criterion: `benches/arithm_bench.rs`, `benches/imgproc_bench.rs`.

## Hard Rules

1. **Zero-FFI**: No `bindgen`, `cc`, or C++ linking. Pure Rust only.
2. **No `unsafe`**: Avoid `unsafe` blocks. If `std::arch` or `pulp` SIMD is required, justify why safe iterators failed.
3. **Error handling**: Return `Result<T, PureCvError>` — never `panic!` or `unwrap` in library code.
4. **License header**: Every new `.rs` file must include the LGPL header from `.agents/skills/license-header-adder/SKILL.md` with `{{FILENAME}}` replaced by the file's basename.

## Feature-Gated Parallelism & SIMD

Four feature combinations must be supported: `parallel`, `simd`, both, or none.

- **Parallel** (`rayon`): Use internal macros `binary_op!` / `binary_op_scalar!` (see `src/core/arithm.rs`) that switch between `par_iter` and `iter` via `#[cfg(feature = "parallel")]`.
- When `parallel` is disabled, `src/core/utils.rs` exports `ParIterFallback` as a sequential shim.
- **SIMD** (`pulp`): Optional optimizations gated by `#[cfg(feature = "simd")]`.
- Prefer `.chunks_exact(n)` over `.chunks(n)` to help LLVM auto-vectorize.

## Naming & Style Conventions

- **OpenCV parity**: Mirror OpenCV function names in snake_case (`cvt_color`, `box_filter`, `copy_make_border`).
- **Generics**: All operations take `Matrix<T>` with trait bounds like `T: Default + Clone + ToPrimitive + FromPrimitive + Send + Sync`.
- **Module re-exports**: Each module file (`core.rs`, `imgproc.rs`) re-exports public symbols so users write `purecv::core::Matrix`, `purecv::imgproc::cvt_color`.
- **Comments & docs in English**.

## Developer Workflow

```powershell
# Format → Lint → Test (this exact order; CI rejects failures)
cargo fmt
cargo clippy
cargo test

# Run benchmarks
cargo bench

# Run examples
cargo run --example arithmetic
cargo run --example color_conversion
cargo run --example structural_ops
```

## Git & PR Conventions

- Branch from `dev`, PR against `dev`. `main` is for stable releases only.
- **Conventional Commits** required (enforced by `git-cliff` changelog):
  - Types: `feat`, `fix`, `perf`, `doc`, `refactor`, `test`, `chore`
  - Scopes: `core`, `imgproc`, `simd`, `wasm`, `parallel`
  - Example: `feat(imgproc): add bilateral_filter implementation`

## Adding a New Algorithm

1. Read the C++ reference in `cpp_ref/opencv/modules/` to extract the mathematical kernel.
2. Create the function in the appropriate submodule (`src/core/` or `src/imgproc/`).
3. Use `Result<Matrix<T>, PureCvError>` as the return type.
4. Gate parallel loops with `#[cfg(feature = "parallel")]` and provide a sequential fallback.
5. Add tests in the module's `tests.rs` and a Criterion benchmark in `benches/`.
6. Re-export the function in the parent module file (`core.rs` or `imgproc.rs`).


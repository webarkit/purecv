# PureCV — Claude Code Instructions

## Project Mission
Pure Rust port of OpenCV's `core` and `imgproc` modules. Memory-safe, zero-FFI,
high-performance. See `.agents/instructions.md` for the full architectural mandate.

## Architectural Rules

- **Zero-FFI:** No `bindgen`, `cc`, or C++ linking. Every algorithm must be rewritten in idiomatic Rust.
- **Memory Safety:** Use Rust's ownership model. Prefer `Vec<T>` or `Box<[T]>` for buffers. Avoid `unsafe` — if you must use it, justify why `pulp` or `std::arch` is required and why safe iterators failed.
- **Data Layout:** Row-major contiguous memory (`Matrix<T>`) matching `cv::Mat`.
- **Concurrency:** Use Rayon (`parallel` feature) for pixel-parallel tasks; always provide a sequential fallback.
- **SIMD:** Use `pulp` (`simd` feature) for manual SIMD. Prefer `.chunks_exact(n)` over `.chunks(n)` for LLVM auto-vectorization.
- **OpenCV Parity:** Match OpenCV default border types and naming (e.g. `cvt_color` not `convert_color`).
- **Error handling:** Use `Result<T, PureCvError>` — never `panic!` in library code.

## Code Quality — run in this order before every commit

```bash
cargo fmt
cargo clippy        # fix ALL warnings
cargo test          # all unit + doc tests must pass
```

CI rejects any PR that fails `cargo fmt --check` or `cargo clippy`.

## New Files

Every new Rust source file must start with the LGPL-3.0 license header.
Template: `.agents/skills/license-header-adder/resources/HEADER.txt`
Replace `{{FILENAME}}` with the actual filename.

## Commits — Conventional Commits (mandatory)

Format: `<type>(<scope>): <description>`

| Type | When |
|------|------|
| `feat` | new feature or algorithm |
| `fix` | bug fix |
| `perf` | performance improvement |
| `doc` | documentation only |
| `refactor` | neither fix nor feature |
| `test` | adding or fixing tests |
| `chore` | build, deps, tooling |

Preferred scopes: `(core)` `(imgproc)` `(simd)` `(wasm)` `(parallel)`

## Git / PR Workflow

- PRs start from and target the `dev` branch (not `main`).
- Keep PRs focused; one feature or fix per PR.

## Project Structure

```
src/core/       — Matrix<T>, Scalar, arithmetic, stats, linear algebra
src/imgproc/    — color conversion, filtering, edge detection, threshold
crates/wasm/    — wasm-bindgen wrappers and wasm-pack build
benches/        — Criterion benchmarks
.agents/        — reference docs, roadmap, skills (shared with other agents)
```

## Key Types

- `Matrix<T>` — main n-channel image/matrix type
- `Scalar<T>` — 4-channel per-pixel constant (channels beyond 4 default to `T::default()`)
- `MatType` / `Depth` — OpenCV-style type constants (`CV_8UC3`, `CV_32FC1`, …)
- `PureCvError` — library error type; use `Result<T, PureCvError>` everywhere

## Performance Notes

- SIMD + Parallel benchmarks target 1024×1024 matrices.
- Full benchmark results: `benches/benchmark_results.md`.
- Best speedup achieved: 22× (`sobel_3x3_f32` with Parallel + SIMD).
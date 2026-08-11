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

### Releases

Releases go `dev` → PR to `main` → merge → tag `vX.Y.Z` on `main`. Pushing the tag
triggers `release.yml`, which publishes to crates.io **and** npm — irreversible.

**Merge release PRs with a merge commit, not "Rebase and merge" or "Squash and
merge".** Rebasing replays dev's commits as new objects on `main`, so `dev` stops
being an ancestor of `main` and the two diverge with identical content but different
SHAs. The next release PR then replays every old commit again. This happened with
v0.7.1 (#95) and had to be repaired by resetting `dev` to `main`.

Release prep steps (see the v0.7.1 commit for a worked example):

1. Bump the version in `Cargo.toml` (**two places** — `[package]` and
   `[workspace.package]`) and in the root `package.json`.
2. Run `npm run build` — this regenerates `crates/wasm/pkg/package.json`, which is
   tracked and otherwise silently drifts.
3. `npx git-cliff --config cliff.toml --tag vX.Y.Z --unreleased --prepend CHANGELOG.md`,
   then add the blank line `--prepend` omits before the previous version heading.
4. Commit as `chore(release): prepare for vX.Y.Z` — `cliff.toml` skips this message
   from the changelog, so land any other changes in their own commits *first* or they
   will not appear.

## Module Structure Convention

Each top-level module (`core/`, `imgproc/`, `features2d/`, etc.) must contain its own:
- `tests.rs` — unit tests for that module (`#[cfg(test)] mod tests;`)
- `simd.rs` — SIMD-accelerated helpers specific to that module (`pub(crate) mod simd;`)

Keep SIMD kernels co-located with the module that uses them. The core `SimdElement`
trait and element-wise operations stay in `src/core/simd.rs`; domain-specific kernels
(e.g. color conversion, derivatives) belong in their respective module's `simd.rs`.

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
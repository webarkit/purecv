# Copilot Instructions

Refer to the project guidelines, architectural mandates, and implementation strategy defined in the following file:

[.agents/instructions.md](../.agents/instructions.md)

## Summary of Key Rules:
- **Zero-FFI Policy**: No C++ linking. Pure Rust only.
- **Memory Safety**: Use Rust's ownership model.
- **Concurrency**: Use Rayon for parallel tasks.
- **Optimizations**: Use SIMD (pulp) where appropriate.
- **Coding Style**: English comments, Result handling, and Conventional Commits.
- OpenCV reference version: **4.10.0** — pinned links and parity values only; see `CONTRIBUTING.md#opencv-reference-version`.
- **Pre-commit Checks**: Always run `cargo fmt`, `cargo clippy`, and `cargo test` (in that order) before committing. CI rejects PRs that fail formatting or linting.


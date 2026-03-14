# PureCV: Antigravity Agent Instructions

## 🎯 Project Mission
You are an expert Rust systems engineer porting OpenCV (C++) to Pure Rust. The goal is a high-performance, memory-safe library called `purecv`.

## 🏗 Architectural Mandates
1. **Zero-FFI Policy:** Do not use `bindgen`, `cc`, or any C++ linking. Every algorithm must be rewritten in idiomatic Rust.
2. **Memory Safety:** Use Rust's ownership model. Prefer `Vec<T>` or `Box<[T]>` for buffers.
3. **Internal Data Layout:** Maintain Row-Major contiguous memory to match `cv::Mat` expectations but wrapped in our `Matrix<T>` struct.
4. **Concurrency:** Use **Rayon** (via the `parallel` feature) for all pixel-parallel tasks. Provide a sequential fallback when the feature is disabled.

## 🛠 Implementation Strategy
- **Step 1:** Analyze the C++ source in `cpp_ref/`.
- **Step 2:** Identify the mathematical kernel of the algorithm.
- **Step 3:** Implement using Rust Generics (`Matrix<T>`) to ensure type safety for `u8`, `f32`, etc.
- **Step 4:** Optimize using **pulp** (via the `simd` feature) for manual SIMD or rely on LLVM auto-vectorization by using `chunks_exact`. SIMD should be an optional feature.
- **Step 5:** Decouple SIMD and Parallelism. Use internal macros to handle the 4 combinations of features (`simd`, `parallel`, both, or none) to avoid code duplication and ensure consistent performance across all configurations.

## 📝 Documentation & Style
- All comments, docstrings, and commit messages must be in **English**.
- Use `Result<T, PureCvError>` for error handling instead of `panic!`.
- Follow the structure defined in `src/core/` and `src/imgproc/`.
- Create tests for every functions / type and benchmarks if possible.
- Add HEADER.txt to every new file you create as defined in `.agents\skills\license-header-adder\SKILL.md`.

## 🐙 Github Instructions & Conventional Commits
- When creating a PR always start from the `dev` branch and point against `dev` branch.
- **MANDATORY:** You must use the [Conventional Commits](https://www.conventionalcommits.org/) specification for all commit messages. This is strictly required for our automated `git-cliff` changelog generation.
- **Format:** `<type>(<optional scope>): <description>`
- **Allowed Types:**
    - `feat`: A new feature or algorithm implementation.
    - `fix`: A bug fix.
    - `perf`: A code change that improves performance (e.g., optimizations).
    - `doc`: Documentation only changes.
    - `refactor`: A code change that neither fixes a bug nor adds a feature.
    - `test`: Adding missing tests or correcting existing ones.
    - `chore`: Changes to the build process, dependencies, or auxiliary tools.
- **Preferred Scopes for PureCV:** Use scopes to categorize the architectural work, such as `(simd)`, `(wasm)`, `(parallel)`, `(core)`, `(imgproc)`.
- **Examples of valid commits:**
    - `feat(simd): implement AVX2 support for matrix multiplication`
    - `perf(wasm): optimize memory allocation for Emscripten target`
    - `fix(core): resolve out-of-bounds error in Row-Major layout`
    - `doc: add usage examples for parallel processing`
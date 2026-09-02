# Maintainers Guide for purecv

This document is intended for the core maintainers of the `webarkit/purecv` repository. It outlines the responsibilities, architectural mandates to enforce during code reviews, and the exact steps required to publish a new release.

## Current Maintainers

* **Walter Perdan** ([@kalwalt](https://github.com/kalwalt)) - Creator & Lead Maintainer

## 1. Code Review & Architectural Mandates

When reviewing Pull Requests, maintainers must ensure that the following core principles of the `purecv` project are strictly upheld:

* **Zero-FFI Policy:** Absolutely no C++ linking, `bindgen`, or `cc` is allowed. Every algorithm must be written in pure, idiomatic Rust.
* **Memory Safety:** Ensure Rust's ownership model is respected. Verify that buffers rely on `Vec<T>` or `Box<[T]>` and that there are no unnecessary memory allocations inside hot loops.
* **Internal Data Layout:** Algorithms must expect and maintain Row-Major contiguous memory within the `Matrix<T>` struct.
* **Feature Gating:** * Concurrency (`parallel` feature via Rayon) must have a sequential fallback.
    * Vectorization (`simd` feature via Pulp) must be optional and gracefully degrade to standard iterators or auto-vectorization when disabled.
    * WebAssembly (`wasm` feature) compatibility must be preserved.
* **Language:** All comments, docstrings, and commit messages **must** be in English.
* **Conventional Commits:** PR titles and commit messages must follow the Conventional Commits specification. PRs should be strictly squashed and merged.

## 2. Release Process

Publishing a new version requires a mix of manual changelog curation and automated CI/CD deployment. Follow these steps sequentially:

### Step 1: Pre-Release Checks
1. Ensure you are on the `dev` branch and it is up to date.
2. Verify that all CI checks (Formatting, Clippy, Tests for `parallel` and `simd`) are passing on the latest commit.

### Step 2: Bump the Version
Update the version number in the `Cargo.toml` file of the workspace in [package] and [workspace.package] sections, and in the root `package.json`.

Also check `README.md` and `crates/wasm/README.md` for hardcoded version strings in
installation snippets (e.g. `purecv = "0.6"`) — these don't update automatically and
are easy to miss. Search for the old version number across both files before moving on.

### Step 3: Generate the Local Changelog
We use `git-cliff` to parse the conventional commits and update the historical changelog. Run the following command in the root directory:

```bash
npx git-cliff -u --prepend CHANGELOG.md
```
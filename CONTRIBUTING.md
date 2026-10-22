# Contributing to purecv

Thank you for your interest in contributing to `purecv`! By following these guidelines, you help keep the project clean, organized, and easy to maintain.

## Pull Request Workflow

To contribute to the project, please follow these steps:

1. **Sync with the `dev` branch**: Ensure your local copy is up to date before starting new work.
   ```powershell
   git checkout dev
   git pull origin dev
   ```
2. **Create a new branch**: Create a descriptive branch starting from `dev`.
   ```powershell
   git checkout -b your-branch-name
   ```
3. **Develop and Test**: Make your changes and ensure they pass all tests (see the [Testing](#testing) section).
4. **Submit the Pull Request**: Open a PR against the `dev` branch of the main repository.

> **Important**: All PRs must be made against the `dev` branch. The `main` branch is reserved exclusively for stable releases.

## Testing

Before submitting a commit or a Pull Request, it is essential to verify that the code works correctly and adheres to the project's quality standards.

Run the following commands in the project root:

* **Run tests**:
  ```powershell
  cargo test
  ```
* **Check formatting**:
  ```powershell
  cargo fmt -- --check
  ```
* **Run linting with Clippy**:
  ```powershell
  cargo clippy -- -D warnings
  ```

If you add a new feature, make sure to include appropriate tests in `src/**/tests.rs` or as integration tests.

## OpenCV Reference Version

`purecv` ports OpenCV and measures parity against it, so every contributor, human or agent, must read, cite and measure the same version. This section is the single source of truth; the other documents point here.

1. **The reference is OpenCV 4.10.0** (commit `71d3237a093b60a27601c20e9ee6c3e52154e8b1`). Work derived from opencv_contrib uses opencv_contrib 4.10.0 (commit `1ed3dd2c53888e3289afdb22ec4e9ebbff3dba87`).
2. **Links are pinned.** Use `docs.opencv.org/4.10.0/...` for documentation and `github.com/opencv/opencv/blob/4.10.0/<path>#L<n>` for source (`github.com/opencv/opencv_contrib/blob/4.10.0/<path>#L<n>` for opencv_contrib). Unversioned (`4.x`) links, `master` links, links to any other version, and bare `file.cpp:NN` citations outside link text are rejected by `cargo test --test opencv_reference_links`.
3. **Parity values come from 4.10.0.** Measure with the Python environment in `cpp_ref/.venv` (`opencv-contrib-python==4.10.0.84`) or with opencv.js 4.10.0, either the pinned package in `cpp_ref/node_modules` or a self-built one compiled from `cpp_ref/src/opencv`. Generators live in `scripts/opencv_ref/` and record the version line of `getBuildInformation()` in their output. A value from any other version must name that version in the test comment, for example `// Measured with OpenCV 5.0.0 (opencv-python 5.0.0.93), not the 4.10.0 reference`.
4. **Reference sources come from the bootstrap.** Run `py -3 cpp_ref/bootstrap.py --bootstrap-file cpp_ref/libraries.json` (`python3 ...` on Linux/macOS). It produces `cpp_ref/src/opencv` and `cpp_ref/src/opencv_contrib` at the pinned commits, verifies them, and exits non-zero on a mismatch. Take line numbers from these trees or from the `#L` URLs, never from another checkout. Details and the bump procedure are in [`cpp_ref/README.md`](cpp_ref/README.md).
5. **Changing the reference is its own change.** It needs its own issue on the roadmap (#128) and means re-checking every parity test and every changed default. [webarkit/jsfeatNext](https://github.com/webarkit/jsfeatNext) pins the same build for its OpenCV fixtures (see the "OpenCV reference version" section of its `AGENTS.md`, added in [webarkit/jsfeatNext#237](https://github.com/webarkit/jsfeatNext/pull/237)). purecv validates against jsfeatNext and jsfeatNext validates against OpenCV, so a change of reference also opens the matching issue in webarkit/jsfeatNext, and the two repositories move together.
6. **Why 4.10.0, and how parity is recorded:** the code had already converged on it (#165). The decision, the frozen per-version audit folders under `docs/parity/<version>/` and the cost of a bump are recorded in [ADR 0001](docs/adr/0001-opencv-parity-audit.md); read it before changing the reference version or a parity test.

OpenCV is only used out of tree to produce reference numbers. It is never a build, dev or runtime dependency of the crate.

## Fixing a parity issue

The parity audit (`docs/parity/<version>/`, [ADR 0001](docs/adr/0001-opencv-parity-audit.md)) left one issue per divergence from OpenCV 4.10.0 and, for each, one or more tests in `tests/parity_*.rs` marked `#[ignore = "diverges from OpenCV 4.10.0, see #N"]`. Those tests assert OpenCV's measured output and fail today; CI runs them as expected failures (`python3 scripts/parity_xfail.py`, default features and `--features simd`) and fails when one of them passes, so a fix cannot land without flipping its test. The fix phase is tracked by #268. A PR that fixes parity issue #N:

1. **Removes the `#[ignore]`** of every test in `tests/parity_*.rs` that cites #N, and of no other test. The reference guard (`cargo test --test opencv_reference_links`) requires every remaining `#[ignore]`, also inside `#[cfg_attr(...)]`, to read `diverges from OpenCV 4.10.0[ <build qualifier>], see #N` with exactly one issue. A test that already passes in one build only (for example a `simd` kernel that saturates where the scalar path wraps) keeps the ignore for the other builds as `#[cfg_attr(not(feature = "simd"), ignore = "diverges from OpenCV 4.10.0, see #N")]`.
2. **Deletes the paired current-behaviour pin**, the `*_current_behaviour_*` test whose doc comment names #N, if there is one.
3. **Passes under every feature set CI runs**: `cargo test`, `cargo test --features simd`, `cargo test --features parallel` and `cargo test --no-default-features --features std`, plus `python3 scripts/parity_xfail.py` (and with `--features simd`) to confirm that every other ignored test still fails.
4. **Re-runs the row's measurement script** in `scripts/opencv_ref/` only when the fix needs new inputs or expected values; new values follow the [OpenCV Reference Version](#opencv-reference-version) rules and name the script that produced them.
5. **Never edits `docs/parity/<version>/`.** The series folder is frozen; the live status of the row is the closed issue plus the passing test (rule 11 of [`docs/parity/AGENTS.md`](docs/parity/AGENTS.md)).
6. **Closes #N** (`Closes #N` in the PR body; close it by hand after the merge when the PR targets a non-default branch).

## Commit Message Conventions

To maintain a clean and automated release history, this project strictly adheres to the [Conventional Commits](https://www.conventionalcommits.org/) specification.

All Pull Requests will be squashed and merged. Please ensure your PR titles and commit messages follow this format:

`<type>(<optional scope>): <description>`

### 1. Allowed Types
* **`feat`**: A new feature or core algorithm implementation.
* **`fix`**: A bug fix.
* **`perf`**: A code change that improves performance.
* **`doc`**: Documentation-only changes.
* **`refactor`**: A code change that neither fixes a bug nor adds a feature (e.g., restructuring).
* **`test`**: Adding missing tests or correcting existing ones.
* **`style`**: Changes that do not affect the meaning of the code (formatting, missing semi-colons, etc.).
* **`chore`**: Changes to the build process, CI configuration, or auxiliary tools.

### 2. Project-Specific Scopes
To help us automatically categorize changes in our Changelog, please use one of the following scopes when applicable:
* **`simd`**: Manual vectorization or SIMD-specific optimizations (via Pulp).
* **`wasm`**: WebAssembly and Emscripten target adjustments or optimizations.
* **`parallel`**: Concurrency implementations (e.g., Rayon).
* **`core`**: Core matrix operations, memory layout, or foundational structs.
* **`imgproc`**: Image processing algorithms, filters, and color conversions.

### 3. Examples
* `feat(simd): implement vectorized absolute difference`
* `perf(wasm): optimize memory allocation for emscripten builds`
* `fix(core): resolve out-of-bounds error in matrix transpose`
* `doc: update README with parallel execution benchmarks`

If your PR introduces a breaking change, please include `BREAKING CHANGE:` in the footer or append a `!` after the type/scope (e.g., `feat(core)!: change Matrix internal layout`).

## Creating Issues

Before creating a new issue, please search the [existing issues](https://github.com/kalwalt/purecv/issues) to see if it has already been reported.

When creating an issue, please provide as much information as possible:
* **For Bug Reports**: Include a clear description of the bug, steps to reproduce it, the expected behavior, and any relevant error messages or logs.
* **For Feature Requests**: Explain the purpose of the feature, why it is needed, and how it should work.

## Reporting Bugs and Feature Requests

If you encounter a bug or have an idea for a new feature, we invite you to open an [Issue](https://github.com/kalwalt/purecv/issues). Please be as detailed as possible in the description to help us resolve the issue quickly.

## Code of Conduct

We adopt the [Contributor Covenant](https://www.contributor-covenant.org/version/2/1/code_of_conduct/code_of_conduct.md) to promote an inclusive and respectful environment for all contributors.

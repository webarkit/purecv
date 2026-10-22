# Parity audit: method

This directory holds the module-by-module comparison of purecv with OpenCV. One audit
series compares every public item of purecv with **one** OpenCV reference version and records
whether purecv diverges and in which direction. Each new divergence becomes its own issue;
fixes go through those issues, never through an audit PR. Why it works this way:
[ADR 0001](../adr/0001-opencv-parity-audit.md). Rules for agents:
[`AGENTS.md`](AGENTS.md).

- **Reference version:** the one pinned in
  [`CONTRIBUTING.md#opencv-reference-version`](../../CONTRIBUTING.md#opencv-reference-version)
  (today OpenCV 4.10.0), with its commits in `cpp_ref/libraries.json` and its measurement
  builds in `cpp_ref/requirements.txt` / `cpp_ref/package.json`. "The reference version"
  below always means that pin.
- **Rules for reading and measuring OpenCV:** [`cpp_ref/AGENTS.md`](../../cpp_ref/AGENTS.md).
  Its "Where to look" table maps each purecv module to the OpenCV sources and doc groups.

## Versions

Each completed series is a frozen snapshot in its own folder. It is never edited afterwards:
not when OpenCV is bumped, not for typos (a correction is a note in the next series, or an
issue). The live state of the issues a series filed is its GitHub milestone, not the issue map
inside the folder.

| Reference | Folder | Umbrella | Milestone | State |
|---|---|---|---|---|
| OpenCV 4.10.0 | [`4.10.0/`](4.10.0/README.md) | [#172](https://github.com/webarkit/purecv/issues/172) | [Parity audit: purecv ↔ OpenCV 4.10.0](https://github.com/webarkit/purecv/milestone/8) | complete (2026-10-21), frozen |

## What an audit PR may change

- its own `docs/parity/<version>/` folder;
- new reference scripts under `scripts/opencv_ref/`;
- new tests that **pass** (rows that already match OpenCV), committed as
  `test(<scope>): pin <item> to OpenCV <version>`;
- new tests marked `#[ignore = "diverges from OpenCV <version>, see #NNN"]`.

It may **not** change library code under `src/` outside tests, and it never edits another
version's folder. The branches of a series (an integration branch or `dev`) are recorded in
its folder's README.

## What is compared

The comparison works at two levels. OpenCV is read and run **out of tree**; it is never
linked into purecv (Zero-FFI).

1. **Source level → `inspected`.** The C++ of the reference version in `cpp_ref/src/opencv`
   (or `cpp_ref/src/opencv_contrib`) is read side by side with the purecv Rust. The reference
   is the **generic CPU path**.
   - The OpenCL, IPP, HAL and OpenVX back ends (`CV_OCL_RUN`, `CV_IPP_RUN*`, `CALL_HAL`,
     `CV_OVX_RUN`) are alternatives, not the reference.
   - The `*.simd.hpp` / `CV_SIMD` / `CV_CPU_DISPATCH` code paths are optimisations of the
     generic path.
   - Where these back ends disagree with each other (as
     [#161](https://github.com/webarkit/purecv/issues/161) found for FAST's response with
     nonmax off), note it on the row.
2. **Behaviour level → `measured`.** The same inputs go through binaries of the reference
   version and through purecv, and the outputs are compared. The binaries are:
   - the `cpp_ref/.venv` wheel (the `opencv-contrib-python` pinned in
     `cpp_ref/requirements.txt`), which is the C++ compiled from the very same commits and
     called from Python;
   - or opencv.js of the same version (WASM, no IPP): the pinned package in
     `cpp_ref/node_modules`, or a self-built one compiled from `cpp_ref/src/opencv`.

   The x86-64 wheels ship with Intel IPP and runtime SIMD dispatch **enabled**:
   - measure with `cv2.ipp.setUseIPP(False)` by default, so that the measurement matches the
     generic path read at level 1;
   - record `cv2.ipp.useIPP()`, `cv2.useOptimized()`, the platform, and the `Baseline` /
     `Dispatched` lines of `cv2.getBuildInformation()`;
   - if the result with IPP on differs, record that as a note on the row. It is not a purecv
     bug.
3. **Optional: a C++ harness**, when intermediates are needed that no binding exposes (for
   example LK's per-iteration `delta`). Build it **out of tree** in `cpp_ref/build/` against
   `cpp_ref/src/opencv`, and commit its source under `scripts/opencv_ref/`. It is never part
   of the crate's build.

## Statuses

Exactly these six. Every row has one.

| Status | Meaning |
|---|---|
| `measured` | Compared numerically with the reference version through reproducible evidence (see [Evidence rule](#evidence-rule)); identical, or within a stated tolerance. |
| `inspected` | The reference source was read side by side, or evidence exists that is not reproducible (issue comments, unidentified builds, other versions). |
| `diverges` | A confirmed difference. The Issue column is required. |
| `intentional` | A deliberate difference, with the rationale in the notes. |
| `purecv-only` | An existing purecv item with no counterpart in the reference version. |
| `missing` | Exists in the reference version of OpenCV (or opencv_contrib, noted as such) but not in purecv. A roadmap gap, not a bug. |

### Direction

For `diverges` rows only:

- `→ OpenCV`: purecv changes.
- `keep?`: the maintainer decides. If the difference is kept, the row becomes `intentional`
  and the rationale is written down.

## Dimensions

Each row says which of these were checked (`Dims checked`, e.g. `D1-D4, D7`).

| Dim | What is compared |
|---|---|
| D1 | Name, parameter order and defaults, from the reference version's header. |
| D2 | Supported depths and channels, and behaviour on unsupported input (`Err` vs `CV_Assert`). |
| D3 | Default border (`BORDER_REFLECT_101`) and the border modes supported. |
| D4 | Integer output conversion: `saturate_cast` (round half to even, then clamp) vs truncation or `round()` (half away from zero). |
| D5 | Arithmetic path: fixed-point / bit-exact `8U` vs float. |
| D6 | The same result across `default` / `simd` / `parallel` / `--no-default-features --features std` / `wasm32`. |
| D7 | Evidence: tool, version and inputs (below). |

**D7 in detail:**

- Source reading: record `git -C cpp_ref/src/opencv rev-parse HEAD` (it must print the
  opencv commit pinned in `cpp_ref/libraries.json`; the opencv_contrib one for
  `cpp_ref/src/opencv_contrib`) and which back-end path was read (generic, SIMD, IPP, …).
- Measurements: record `cv2.__version__`, `pip freeze` of `cpp_ref/.venv`, the `useIPP` /
  `useOptimized` state and the platform. For opencv.js, record its `Version control:` line
  from `getBuildInformation()`.

## Table template

Every audit document uses this table, one row per public item (`<version>` is the folder's
version; the series folders hold real examples, for instance
[`4.10.0/imgproc.md`](4.10.0/imgproc.md)):

```markdown
| purecv item | OpenCV <version> counterpart (pinned link) | Status | Dims checked | Direction / notes | Issue |
|---|---|---|---|---|---|
| `gaussian_blur` | [`GaussianBlur`](<the reference's doc page for the imgproc filter group>) | `measured` | D1-D5, D7 | u8 and f32, 3x3 to 7x7; IPP off | |
```

- **purecv item:** the Rust path as exported (`module::item`, or `Type::method`).
- **OpenCV counterpart:** a link pinned to the folder's version: the OpenCV documentation
  site with the version as the first path segment, or the source file on GitHub at the
  version's tag (the `blob/<version>/` form, with a `#L<a>-L<b>` line range; opencv_contrib
  likewise). `—` for `purecv-only` rows. This file deliberately contains no such URL: the
  guard checks it against the current reference, so a pinned example would break at every
  bump.
- **Issue:** required for `diverges`; the issue that tracks the gap for `missing` when one
  exists.

The reference guard (`cargo test --test opencv_reference_links`) scans `docs/parity/**/*.md`:
a file under `docs/parity/<version>/` must link that version, every other file the current
reference; unpinned links and bare `file.cpp:NN` citations outside the text of a pinned link
fail the build.

## Evidence rule

A `measured` row has **one** of:

- a passing in-tree test whose comment names the tool and version
  (for example `// OpenCV <version> (opencv-contrib-python <pin>), IPP off`);
- a script under `scripts/opencv_ref/` that runs in `cpp_ref/.venv` (Python: checks
  `cv2.__version__` against the pin in `cpp_ref/requirements.txt` and calls
  `cv2.ipp.setUseIPP(False)`) or with the pinned or a self-built opencv.js of the same version
  (Node), records the build version in its output, and reads committed or seeded inputs.

Evidence for a `diverges` row is such a script, or a test marked
`#[ignore = "diverges from OpenCV <version>, see #NNN"]`.

A value measured with any other OpenCV version can only support `inspected`, and the test
comment must name that version (see `CONTRIBUTING.md`).

## Procedure

| Step | What to do |
|---|---|
| S1 | **Inventory** every public item of the module: `pub fn\|struct\|enum\|trait\|const\|static\|type`, public inherent methods, and the re-exports in `src/<module>.rs` or `src/<module>/mod.rs`. One row each. Regenerate at the branch base; inventories in a plan are indicative. |
| S2 | **Bootstrap** the reference sources: `py -3 cpp_ref/bootstrap.py --bootstrap-file cpp_ref/libraries.json` (`python3 …` on Linux/macOS). Continue only when it prints `verify … OK` for both libraries. Then fill the counterpart and D1–D5 from `cpp_ref/src/opencv/<path>` or `cpp_ref/src/opencv_contrib/<path>`. |
| S3 | **Measure** the module's priority rows following the [Evidence rule](#evidence-rule). |
| S4 | **Add `missing` rows** from the reference version's doc group pages of the module. |
| S5 | **Link** known issues and PRs. Propose one issue per new `diverges` row in the document and the PR, titled `<module>: <item> differs from OpenCV <version>: <what>`; file it **after maintainer approval**, in the series milestone. |
| S6 | **Update** the issue map and the progress column of the Documents table in the folder's README. |
| S7 | **Pre-commit**: `cargo fmt --all -- --check`, `cargo clippy --all-targets -- -D warnings`, `cargo test` (this includes the reference guard). Commit `doc(<scope>): add OpenCV <version> parity audit for <module>`. |

An audit is done when every public item has a row with a status, every `diverges` row links
an issue, every priority row is `measured` with reproducible evidence or `diverges` with a
script or an `#[ignore]`d test, and S7 passes.

## Future reference versions

1. **The bump is its own PR**, before any new series: the `cpp_ref` pins
   (`libraries.json`, `requirements.txt`, `package.json`), `CONTRIBUTING.md`, the
   `REFERENCE` constants of `tests/opencv_reference_links.rs`, and the re-measured live
   tooling (`tests/parity_*.rs`, `scripts/opencv_ref/`, which read the version from the pin).
   The decision itself is taken on the roadmap
   ([#128](https://github.com/webarkit/purecv/issues/128)). The same PR checks that the
   live, version-independent files (this README, [`AGENTS.md`](AGENTS.md), the ADRs) still
   contain no OpenCV URL pinned to the old version: the guard checks them against the new
   `REFERENCE` and fails the bump PR otherwise.
2. **The next series goes in `docs/parity/<version>/` as a delta.** Start from a copy of the
   previous folder's documents, then, module by module:
   - re-audit (S1–S7) every row whose OpenCV counterpart changed between the two tags, per
     `git -C cpp_ref/src/opencv diff <old>..<new> -- modules/<module>` (and the same in
     `cpp_ref/src/opencv_contrib`), and every row whose purecv item changed since the previous
     series;
   - re-measure the remaining rows with the new pin and mark them
     `carried over from <old>, re-measured`;
   - give the series its own umbrella issue and milestone, and a row in [Versions](#versions).
3. **Frozen folders are never edited**, including their links: the guard checks each folder
   against its own version, so they stay valid after a bump.
4. **The live issue state is the GitHub milestone** of each series, not the frozen issue map.

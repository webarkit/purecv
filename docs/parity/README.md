# Parity audit: purecv ↔ OpenCV 4.10.0

This directory holds the module-by-module comparison of purecv with OpenCV **4.10.0**,
tracked in [#172](https://github.com/webarkit/purecv/issues/172). One document per module
records, for every public item, whether purecv diverges from OpenCV and in which direction.
Each new divergence becomes its own issue; fixes go through those issues, never through an
audit PR.

- **Reference:** OpenCV 4.10.0, opencv commit `71d3237a093b60a27601c20e9ee6c3e52154e8b1`,
  opencv_contrib commit `1ed3dd2c53888e3289afdb22ec4e9ebbff3dba87`. The policy, the link
  form and the measurement tools are defined once in
  [`CONTRIBUTING.md#opencv-reference-version`](../../CONTRIBUTING.md#opencv-reference-version).
- **Plan:** Part B of
  [`2026-10-11-opencv-4.10.0-reference-and-parity-audit.md`](../superpowers/plans/2026-10-11-opencv-4.10.0-reference-and-parity-audit.md).
  Tasks B1–B5 hold the indicative inventory, the seeded rows, the priority rows and the
  `missing` rows of each module.
- **Rules for reading and measuring OpenCV:** [`cpp_ref/AGENTS.md`](../../cpp_ref/AGENTS.md).
  Its "Where to look" table maps each purecv module to the 4.10.0 sources and doc groups.

## Documents

| Module | Document | Sub-issue | Progress |
|---|---|---|---|
| method (this file) | `docs/parity/README.md` | [#172](https://github.com/webarkit/purecv/issues/172) (umbrella) | done |
| core | `docs/parity/core.md` | [#173](https://github.com/webarkit/purecv/issues/173) | pending |
| imgproc | `docs/parity/imgproc.md` | [#174](https://github.com/webarkit/purecv/issues/174) | pending |
| features2d | `docs/parity/features2d.md` | [#175](https://github.com/webarkit/purecv/issues/175) | pending |
| video | `docs/parity/video.md` | [#176](https://github.com/webarkit/purecv/issues/176) | pending |
| calib3d | `docs/parity/calib3d.md` | [#177](https://github.com/webarkit/purecv/issues/177) | pending |

The audits run in this order: core → imgproc → features2d → video → calib3d.

## What an audit PR may change

- `docs/parity/`;
- new reference scripts under `scripts/opencv_ref/`;
- new tests that **pass** (rows that already match OpenCV), committed as
  `test(<scope>): pin <item> to OpenCV 4.10.0`;
- new tests marked `#[ignore = "diverges from OpenCV 4.10.0, see #NNN"]`.

It may **not** change library code under `src/` outside tests.

### Branches

The B0–B5 PRs target the integration branch **`parity-audit`** (cut from `dev` at `902d774`,
v0.11.0), not `dev`. Each task has its own branch, cut from an up-to-date
`origin/parity-audit`: `doc/parity-method` for B0 and `doc/parity-<module>` for B1–B5. When
the series is complete, `parity-audit` is merged into `dev` in one PR. This supersedes the
plan's "cut from `origin/dev`, open PRs into `dev`" for Part B (decision recorded in
[#172](https://github.com/webarkit/purecv/issues/172)).

## What is compared

The comparison works at two levels. OpenCV is read and run **out of tree**; it is never
linked into purecv (Zero-FFI).

1. **Source level → `inspected`.** The OpenCV 4.10.0 C++ in `cpp_ref/src/opencv` (or
   `cpp_ref/src/opencv_contrib`) is read side by side with the purecv Rust. The reference is
   the **generic CPU path**.
   - The OpenCL, IPP, HAL and OpenVX back ends (`CV_OCL_RUN`, `CV_IPP_RUN*`, `CALL_HAL`,
     `CV_OVX_RUN`) are alternatives, not the reference.
   - The `*.simd.hpp` / `CV_SIMD` / `CV_CPU_DISPATCH` code paths are optimisations of the
     generic path.
   - Where these back ends disagree with each other (as
     [#161](https://github.com/webarkit/purecv/issues/161) found for FAST's response with
     nonmax off), note it on the row.
2. **Behaviour level → `measured`.** The same inputs go through OpenCV 4.10.0 binaries and
   through purecv, and the outputs are compared. The binaries are:
   - the `cpp_ref/.venv` wheel (`opencv-contrib-python==4.10.0.84`), which is the C++
     compiled from the very same commits and called from Python;
   - or opencv.js 4.10.0 (WASM, no IPP): the pinned package in `cpp_ref/node_modules`, or a
     self-built one compiled from `cpp_ref/src/opencv`.

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
| `measured` | Compared numerically with 4.10.0 through reproducible evidence (see [Evidence rule](#evidence-rule)); identical, or within a stated tolerance. |
| `inspected` | The 4.10.0 source was read side by side, or evidence exists that is not reproducible (issue comments, unidentified builds, other versions). |
| `diverges` | A confirmed difference. The Issue column is required. |
| `intentional` | A deliberate difference, with the rationale in the notes. |
| `purecv-only` | An existing purecv item with no 4.10.0 counterpart. |
| `missing` | Exists in OpenCV 4.10.0 (or opencv_contrib 4.10.0, noted as such) but not in purecv. A roadmap gap, not a bug. |

### Direction

For `diverges` rows only:

- `→ OpenCV`: purecv changes.
- `keep?`: the maintainer decides. If the difference is kept, the row becomes `intentional`
  and the rationale is written down.

## Dimensions

Each row says which of these were checked (`Dims checked`, e.g. `D1-D4, D7`).

| Dim | What is compared |
|---|---|
| D1 | Name, parameter order and defaults, from the 4.10.0 header. |
| D2 | Supported depths and channels, and behaviour on unsupported input (`Err` vs `CV_Assert`). |
| D3 | Default border (`BORDER_REFLECT_101`) and the border modes supported. |
| D4 | Integer output conversion: `saturate_cast` (round half to even, then clamp) vs truncation or `round()` (half away from zero). |
| D5 | Arithmetic path: fixed-point / bit-exact `8U` vs float. |
| D6 | The same result across `default` / `simd` / `parallel` / `--no-default-features --features std` / `wasm32`. |
| D7 | Evidence: tool, version and inputs (below). |

**D7 in detail:**

- Source reading: record `git -C cpp_ref/src/opencv rev-parse HEAD` (it must print
  `71d3237a093b60a27601c20e9ee6c3e52154e8b1`; `1ed3dd2c53888e3289afdb22ec4e9ebbff3dba87` for
  `cpp_ref/src/opencv_contrib`) and which back-end path was read (generic, SIMD, IPP, …).
- Measurements: record `cv2.__version__`, `pip freeze` of `cpp_ref/.venv`, the `useIPP` /
  `useOptimized` state and the platform. For opencv.js, record its `Version control:` line
  from `getBuildInformation()`.

## Table template

Every audit document uses this table, one row per public item:

```markdown
| purecv item | OpenCV 4.10.0 counterpart (pinned link) | Status | Dims checked | Direction / notes | Issue |
|---|---|---|---|---|---|
| `gaussian_blur` | [`GaussianBlur`](https://docs.opencv.org/4.10.0/d4/d86/group__imgproc__filter.html) | `measured` | D1-D5, D7 | u8 and f32, 3x3 to 7x7; IPP off | |
```

- **purecv item:** the Rust path as exported (`module::item`, or `Type::method`).
- **OpenCV 4.10.0 counterpart:** a pinned link, either to the doc page
  (`https://docs.opencv.org/4.10.0/...`) or to the source
  (`https://github.com/opencv/opencv/blob/4.10.0/<path>#L<a>-L<b>`;
  `opencv_contrib` likewise). `—` for `purecv-only` rows.
- **Issue:** required for `diverges`; the issue that tracks the gap for `missing` when one
  exists.

The reference guard (`cargo test --test opencv_reference_links`) scans `docs/parity/*.md`:
unpinned links and bare `file.cpp:NN` citations outside the text of a pinned link fail the
build.

## Evidence rule

A `measured` row has **one** of:

- a passing in-tree test whose comment names the tool and version
  (for example `// OpenCV 4.10.0 (opencv-contrib-python 4.10.0.84), IPP off`);
- a script under `scripts/opencv_ref/` that runs in `cpp_ref/.venv` (Python: starts with
  `assert cv2.__version__ == "4.10.0"` and `cv2.ipp.setUseIPP(False)`) or with a pinned or
  self-built opencv.js 4.10.0 (Node), records the build version in its output, and reads
  committed or seeded inputs.

Evidence for a `diverges` row is such a script, or a test marked
`#[ignore = "diverges from OpenCV 4.10.0, see #NNN"]`.

A value measured with any other OpenCV version can only support `inspected`, and the test
comment must name that version (see `CONTRIBUTING.md`).

## Procedure

| Step | What to do |
|---|---|
| S1 | **Inventory** every public item of the module: `pub fn\|struct\|enum\|trait\|const\|static\|type`, public inherent methods, and the re-exports in `src/<module>.rs` or `src/<module>/mod.rs`. One row each. Regenerate at the branch base; the inventories in the plan are indicative. |
| S2 | **Bootstrap** the reference sources: `py -3 cpp_ref/bootstrap.py --bootstrap-file cpp_ref/libraries.json` (`python3 …` on Linux/macOS). Continue only when it prints `verify … OK` for both libraries. Then fill the counterpart and D1–D5 from `cpp_ref/src/opencv/<path>` or `cpp_ref/src/opencv_contrib/<path>`. |
| S3 | **Measure** the module's priority rows (listed in the plan task) following the [Evidence rule](#evidence-rule). |
| S4 | **Add `missing` rows** from the 4.10.0 doc group pages of the module. |
| S5 | **Link** known issues and PRs. File one issue per new `diverges` row, titled `<module>: <item> differs from OpenCV 4.10.0: <what>`, **after maintainer approval**. |
| S6 | **Update** the [issue map](#issue-map) below and the progress column of [Documents](#documents). |
| S7 | **Pre-commit**: `cargo fmt --all -- --check`, `cargo clippy -- -D warnings`, `cargo test` (this includes the reference guard). Commit `doc(<scope>): add OpenCV 4.10.0 parity audit for <module>`. |

An audit is done when every public item has a row with a status, every `diverges` row links
an issue, every priority row is `measured` with reproducible evidence or `diverges` with a
script or an `#[ignore]`d test, and S7 passes.

## Issue map

Existing open issues and PRs, by phase. State as of 2026-10-17; S6 keeps it current.
Progress: `pending` / `in progress` / `done`.

| Phase | Open | Fixed on `dev` or closed | Progress |
|---|---|---|---|
| **A** reference version | | #165 (closed by PR #169; #181) | done |
| **B0** method | #172 (umbrella) | PR #188 (method), PR for the splits | done |
| **B1** core | #173; #189 (shared `saturate_cast`), #192 (`convert_to`) | #156 (pattern for D6; fixed by PR #162) | pending |
| **B2** imgproc | #174; #157 (filter sites), #190 (tie rule), #191 (RGB→gray), #100 | | pending |
| **B3** features2d | #175; #158; #159 → #193, #194, #195, #196, #197; #160 → #198, #199, #200; #161 → #201, #202, #203 (`SIMD`), #204 (`wasm`); contrib requests #101–#104 | #123, #124, #125, #126, #136, #156: fixed by PR #162, shipped in v0.10.0, still open | pending |
| **B4** video | #176; #154, #155, #178, #183, #184 | #130, #131, #138, #142, #144, #145, #149, #163 (PR #164), #166 (PR #167) | pending |
| **B5** calib3d | #177; #127, #133 (PR #134 closed unmerged), #135 | | pending |
| Tooling (Part A follow-ups) | #170, #171 | | tracked separately |
| Not part of the comparison | #128 (roadmap; owns the future version bump), #132 (GPU), #97 (WASM SIMD perf), #81 and #94 (Miri), #182 (Miri shards), #67 (architecture diagram), #113 (benchmark doc) | PR #168 (license metadata) | |

**Tooling.** #170 (bootstrap output noise) and #171 (bootstrap and reference-guard
robustness) do not compare purecv with OpenCV, but S2 depends on the bootstrap and S7 on the
reference guard. They go through their own `chore(cpp_ref)` PRs to `dev`. After one merges,
sync `parity-audit` with `dev` before the next audit. Note for S7: the guard scans the Rust
sources and `docs/parity`, not the whole tree (#171, item 8).

**Cross-module dependencies:**

- #193 (ORB pyramid) needs B2's `INTER_LINEAR_EXACT` resize (a `missing` row of the
  imgproc audit).
- #195 (`compute()`) and #200 (`first_level`) need #193.
- #189 (shared `saturate_cast` helper) is the prerequisite of #190 and #192, and is reused
  by imgproc, features2d and video.
- #198 and #199 share OpenCV's `cv::RNG` and the pattern entry point.

### Splits of existing issues

Done on 2026-10-17 (plan: B0 Step 5). Each parent keeps a task list linking its children,
and the children are GitHub sub-issues of the parent. All are in the "Parity audit"
milestone.

- **#157** (imgproc saturation) keeps the integer-output sites of the imgproc filters.
  Children:
  - (a) #189 `core: add a shared saturate_cast (round-half-even + clamp)`, the prerequisite;
  - (b) #190 `imgproc: round() ties away from zero where OpenCV rounds half to even`;
  - (c) #191 `imgproc: cvt_color RGB→gray float path differs from OpenCV's fixed point`;
  - (d) #192 `core: Matrix::convert_to truncates and errors instead of saturating`.
- **#159** (ORB remaining gaps):
  - (a) #193 pyramid: sizes (1a), interpolation (1b, needs `INTER_LINEAR_EXACT`) and
    reflect-101 padding (1c);
  - (b) #194 `KeyPoint.size` stays 7;
  - (c) #195 `compute()` with caller-supplied keypoints (needs an API decision);
  - (d) #196 per-level culling ties and the global re-sort (check the overlap with #158);
  - (e) #197 arithmetic of the feature budget.
  - Item 5 (orientation) needs no change and stays in #159 as a docs note.
- **#160** (ORB parameters):
  - (a) #198 `wta_k` 3/4 + `NormHamming2`;
  - (b) #199 `patch_size != 31`, with the `GET_VALUE` sampling-parity step;
  - (c) #200 `first_level > 0` (after #193).
- **#161** (FAST):
  - (a) #201 nonmax-off semantics (gaps a+b);
  - (b) #202 `Type5_8` / `Type7_12` conformance (gaps c+d);
  - (c) #203 SIMD scan (gap e), labelled `SIMD`: perf, not parity;
  - (d) #204 WASM threshold clamp (gap f), labelled `wasm`.
- No split for #133, #135, #154, #155, #158, #178, #183 or #184: each is already a single
  concern.

When all five audits are done, the version-strategy question (stay on 4.10.0 or move to
4.13 / 5.x) goes on the agenda in #128.

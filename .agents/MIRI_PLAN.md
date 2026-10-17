# Miri UB Verification Plan

> **Status:** Implemented. Both legs green on `ubuntu-latest` and locally — **no
> undefined behaviour found** in any `unsafe` block they reach.
> **Issue:** [webarkit/purecv#81](https://github.com/webarkit/purecv/issues/81) ·
> **PR:** [#93](https://github.com/webarkit/purecv/pull/93) ·
> **Follow-up:** [#94](https://github.com/webarkit/purecv/issues/94) (the `parallel+simd` gap, §5)
> **Amended:** [#182](https://github.com/webarkit/purecv/issues/182) — the legs grew
> to the 30-minute job timeout; each now runs as two nextest partitions (§3, §9) and
> nine more tests are excluded on measurement (§4).

---

## 1. Purpose & Scope

PureCV's headline claim is memory safety. That claim is currently *asserted* rather
than *verified*: a small number of `unsafe` slice reinterpretations back the SIMD
fast paths. This plan adds [Miri](https://github.com/rust-lang/miri) to CI so those
blocks are machine-checked for undefined behavior on every push and pull request.

**What Miri checks:** out-of-bounds accesses, use-after-free, invalid pointer
provenance, aliasing violations (Stacked Borrows), misaligned access, uninitialized
memory reads, and data races.

**What Miri cannot check here:** anything compiled for `wasm32` (unsupported target),
and anything not reached by a `#[test]`. Note that pulp's intrinsics turned out to be
fully supported (§6, A1), so no SIMD path was lost to a missing shim.

---

## 2. Inventory of `unsafe`

Verified by direct inspection, **not** copied from the issue text, which contained
errors (see §7).

Line numbers are as of the `dev` branch at v0.7.0 (`52dc212`) and **will drift**.
Regenerate the inventory rather than trusting them:

```bash
grep -rn "unsafe {" src/
```

If that command returns anything other than the 8 blocks below, this section is stale
and the counts in §3 and §5 need rechecking.

### Production code — 6 blocks, all `simd`-gated

| File | Lines | Context | Reachable without `simd`? |
|------|-------|---------|---------------------------|
| `src/core/arithm.rs` | 99, 248 | `binary_op!` / `unary_op!` — `from_raw_parts_mut` on a rayon chunk | No — `simd` **and** `parallel` |
| `src/core/arithm.rs` | 119, 267 | Same macros, `cfg(not(feature = "parallel"))` branch — whole-buffer `from_raw_parts_mut` | No — `simd` only |
| `src/imgproc/derivatives.rs` | 346, 349 | `fast_deriv_3x3` — `from_raw_parts` / `from_raw_parts_mut` reinterpreting `&[T]` as `&[f32]` after a `TypeId` check | No — `simd` only |

**Consequence:** running Miri *without* `--features simd` audits **zero** production
`unsafe`. This is why the plan uses two runs (§3) rather than the single no-feature
run the issue originally proposed.

### Test code — 2 blocks

| File | Lines | Context |
|------|-------|---------|
| `src/core/tests.rs` | 1533, 1544 | Raw pointer deref of `Matrix::data_ptr()` / `data_ptr_mut()` |

These are the only `unsafe` blocks reachable in the default feature set, and the only
thing the baseline leg meaningfully verifies.

### Files with *no* `unsafe` despite appearances

- `src/core/structural.rs` — a comment describing unsafe that was **not** written;
  the code takes a sequential path instead.
- `src/core/matrix.rs` — `# Safety` doc comments on `data_ptr`/`data_ptr_mut`.
  The functions themselves are safe; only *dereferencing* their return value is not.
- `src/video/simd.rs` — module doc noting the absence of `unsafe`.

---

## 3. What Runs Under Miri

Two legs, both `-p purecv` (never `--workspace`, which would pull in the wasm
crate) and both `--no-default-features` (the default set silently enables `parallel`).

Since #182 each leg runs as **two nextest partitions** in parallel CI jobs
(`Miri shard (<leg>, k/2)`), with a per-leg aggregator job that keeps the historical
names `Miri UB Check (baseline)` / `Miri UB Check (simd)` for `release.yml`. Miri
interprets a single thread, so `cargo miri test` ran every test of a leg one after
the other on one core; `cargo miri nextest run` spawns one process per test and uses
all the runner's vCPUs, and `--partition slice:k/N` deals the test list round-robin
into N buckets so every test runs in exactly one shard. The commands below are what
each shard runs; drop `--partition` to run a whole leg.

### Leg 1 — `baseline` · **required**

```bash
cargo miri nextest run -p purecv --lib --no-default-features --features std --partition slice:k/2
```

Covers all safe code paths and the `data_ptr` tests. Expected to pass trivially.
Its job is to be a fast, stable gate that catches UB regressions in safe code and
in any future non-SIMD `unsafe`.

### Leg 2 — `simd` · **required**

```bash
cargo miri nextest run -p purecv --lib --no-default-features --features std,simd --partition slice:k/2
```

The leg that does the real work: it reaches `arithm.rs:119,267` and
`derivatives.rs:346,349`.

pulp is **confirmed Miri-compatible** (A1, §6). This leg was designed to start
advisory (`experimental: true`) in case Miri's target-dependent intrinsic support
differed on CI, and was promoted to required once it ran green on `ubuntu-latest`
with counts identical to local Windows — 355 passed, 0 failed, 9 ignored on both
(measured on `dev` @ v0.7.0; see §9). Later work kept adding tests and exclusions:
`fix/orb-123-126` 7 ORB tests, #164/#167 9 LK tests, #182 9 more (§4), for 34
`#[cfg_attr(miri, ignore)]` in total. At #182's tip this leg has 460 tests
(`cargo test --no-default-features --features std,simd --lib -- --list`), of which
34 are ignored under Miri; the baseline leg has 404, with the same 34 ignored.

### Aliasing model and MIRIFLAGS

```
MIRIFLAGS: -Zmiri-deterministic-floats
```

**Aliasing:** Miri's defaults are kept — **strict provenance + Stacked Borrows**.
Stacked Borrows is retained over Tree Borrows (`-Zmiri-tree-borrows`) because it is
the stricter guarantee. It raised **no** complaints against the `from_raw_parts_mut`
patterns in practice, so the Tree Borrows fallback was not needed.

**Not** `-Zmiri-strict-provenance` as the issue proposed: strict provenance is now
Miri's default and that flag is deprecated (the opt-*out* is
`-Zmiri-permissive-provenance`). Passing a removed `-Z` flag risks hard-failing the
job on a future nightly.

**`-Zmiri-deterministic-floats`** is required, for the reason documented in §6 —
Miri's intentional float error injection breaks `test_randn_determinism`. Preferred
over ignoring that test, since it keeps the RNG determinism contract under test.

---

## 4. What Is Excluded, and Why

| Excluded | Rationale |
|----------|-----------|
| `crates/wasm` (`purecv-wasm`) | Miri has no `wasm32-unknown-unknown` support. Excluded structurally via `-p purecv` rather than `--workspace --exclude`. |
| `benches/` | Not built by `cargo miri test`; Criterion's sampling under interpretation would be meaninglessly slow. |
| `parallel` feature | Rayon under Miri is slow and its thread support is limited. Accepted cost: a coverage gap (§5). |
| 34 individual tests | Excluded **on measured evidence, or a reasoned bound for the same algorithm** (see below), never speculatively. |

### Excluded tests — measured

Threshold: **>30s under Miri**. Nine tests qualified by direct measurement in #81.
They are iteration-heavy algorithms (RANSAC, ORB, Lucas-Kanade), and **none of them
contain or reach `unsafe`** — so excluding them costs no UB coverage.
`fix/orb-123-126` (#123–#126, #156) added seven more ORB tests, each heavier than
`test_orb_full_pipeline` or `test_orb_pyramid_dimensions` on the same
detect/detect_and_compute/pyramid+blur operations, so they are excluded on that
bound rather than an independent Miri measurement. #164 and #167 added nine LK
fixture tests excluded on the same kind of bound (each runs many full pyramid builds
and LK solves). #182 added nine more, measured from the `dev` run 39021153764 of
2026-10-17 (job logs, gap between consecutive test lines; `std` / `std,simd`), after
the two legs reached the 30-minute timeout. Thirty-four tests are excluded in total.

**Coverage check for the #182 batch.** `test_clahe_u16` reaches no `unsafe` on either
leg: `imgproc/histogram.rs` imports from `core` only `Matrix`, the error type,
`types` and `border_interpolate`, never `arithm`, and its single `simd` branch is in
`compare_hist`, which CLAHE does not call. The eight LK tests go through
`calc_optical_flow_pyramid_lk`; `video/optical_flow.rs` imports `scharr` and
`pyr_down` and nothing from `arithm`, and `pyramid.rs` uses neither `arithm` nor the
derivative kernels. So under `std` they reach no `unsafe` at all, and under `simd`
the only `unsafe` they reach is `fast_deriv_3x3` (the pyramid levels are `f32`),
which stays covered by `imgproc::tests::test_scharr` and by the LK tests kept on
purpose below the threshold: `test_lk_min_eigen_matches_opencv_scale` (29.5s / 10.8s)
and `test_lk_single_newton_step_is_exact_on_quadratic_bowl` (29.0s / 13.9s), plus
about twenty smaller ones. The `pulp::Arch::dispatch` kernels in `video/simd.rs`
(no `unsafe`) are exercised directly by `video::simd::tests` and by every remaining
LK test. `test_harris_at_matches_corner_harris_everywhere` (30.9s / 24.1s) was left
in: it is at the threshold on one leg only.

| Test | Miri time | File |
|------|-----------|------|
| `calib3d::…::test_find_fundamental_mat_ransac` | 927.9s | `src/calib3d/tests.rs` |
| `features2d::tests::test_orb_full_pipeline` | 806.3s | `src/features2d/tests.rs` |
| `video::…::test_lk_pure_translation_x` | 105.2s | `src/video/tests.rs` |
| `core::rng::tests::test_randn_statistics` | 87.5s | `src/core/rng.rs` |
| `video::…::test_lk_use_initial_flow` | 61.7s | `src/video/tests.rs` |
| `video::…::test_lk_stationary_point_identical_frames` | 61.7s | `src/video/tests.rs` |
| `video::…::test_lk_min_eigenvals_flag` | 61.5s | `src/video/tests.rs` |
| `video::…::test_build_pyramid_with_derivatives` | 45.3s | `src/video/tests.rs` |
| `features2d::tests::test_orb_pyramid_dimensions` | 30.2s | `src/features2d/tests.rs` |
| `features2d::tests::test_orb_compute_blurs_the_keypoint_octave_level` | not run; bounded below by `test_orb_full_pipeline` (806 s) / `test_orb_pyramid_dimensions` (30.2 s) | `src/features2d/tests.rs` |
| `features2d::tests::test_orb_compute_matches_opencv_reference` | not run; bounded below by `test_orb_full_pipeline` (806 s) / `test_orb_pyramid_dimensions` (30.2 s) | `src/features2d/tests.rs` |
| `features2d::tests::test_orb_detect_and_compute_equals_detect_then_compute` | not run; bounded below by `test_orb_full_pipeline` (806 s) / `test_orb_pyramid_dimensions` (30.2 s) | `src/features2d/tests.rs` |
| `features2d::tests::test_orb_detect_and_compute_golden` | not run; bounded below by `test_orb_full_pipeline` (806 s) / `test_orb_pyramid_dimensions` (30.2 s) | `src/features2d/tests.rs` |
| `features2d::tests::test_orb_detect_and_compute_golden_fast_params` | not run; bounded below by `test_orb_full_pipeline` (806 s) / `test_orb_pyramid_dimensions` (30.2 s) | `src/features2d/tests.rs` |
| `features2d::tests::test_orb_detect_ignores_descriptor_only_params` | not run; bounded below by `test_orb_full_pipeline` (806 s) / `test_orb_pyramid_dimensions` (30.2 s) | `src/features2d/tests.rs` |
| `features2d::tests::test_orb_ref_image_matches_generator` | not run; bounded below by `test_orb_full_pipeline` (806 s) / `test_orb_pyramid_dimensions` (30.2 s) | `src/features2d/tests.rs` |
| `video::…::test_lk_coarse_window_overhang_matches_opencv` | not run; 150-point pyramidal LK, bounded below by `test_lk_pure_translation_x` (#164) | `src/video/tests.rs` |
| `video::…::test_build_pyramid_opencv_padding_does_not_change_pixels` | not run; ten border combinations each rebuilding image + Scharr pyramids (#167) | `src/video/tests.rs` |
| `video::…::test_lk_opencv_asymmetric_borders_mae` | not run; twelve border/derivative pyramid builds (#167) | `src/video/tests.rs` |
| `video::…::test_lk_opencv_asymmetric_borders_step` | not run; multiple full preprocessing + LK fixtures (#167) | `src/video/tests.rs` |
| `video::…::test_lk_opencv_custom_i16_patch_storage` | not run; six numerical fixtures with repeated LK iterations (#167) | `src/video/tests.rs` |
| `video::…::test_lk_opencv_quantized_patch_reference` | not run; window × depth × iteration × error-mode fixtures (#167) | `src/video/tests.rs` |
| `video::…::test_lk_opencv_termination_reference` | not run; many termination fixtures over raw / image-only / derivative-pyramid paths (#167) | `src/video/tests.rs` |
| `video::…::test_lk_prebuilt_caps_levels_and_preserves_builder_cutoff` | not run; repeated raw/prebuilt tracking across depth combinations (#167) | `src/video/tests.rs` |
| `video::…::test_lk_prebuilt_matches_raw_borders_small_images_and_flags` | not run; many pyramid-construction × tracking combinations (#167) | `src/video/tests.rs` |
| `imgproc::…::test_clahe_u16` | 707.2s (`std`) / 952.0s (`std,simd`) — #182 | `src/imgproc/tests.rs` |
| `video::…::test_lk_recovers_subpixel_shift_with_opencv_defaults` | 244.5s / 87.0s — #182 | `src/video/tests.rs` |
| `video::…::test_lk_extreme_initial_flow_coordinates_are_lost` | 64.9s / 32.7s — #182 | `src/video/tests.rs` |
| `video::…::test_calc_optical_flow_pyramid_lk_uses_scharr_derivatives` | 54.9s / 16.9s — #182 (was ~18s before #167) | `src/video/tests.rs` |
| `video::…::test_lk_coarse_level_degeneracy_keeps_propagated_flow` | 50.8s / 17.0s — #182 | `src/video/tests.rs` |
| `video::…::test_lk_coarse_level_degeneracy_does_not_lose_point` | 50.5s / 17.0s — #182 | `src/video/tests.rs` |
| `video::…::test_lk_extreme_template_coordinates_are_lost` | 49.4s / 19.4s — #182 | `src/video/tests.rs` |
| `video::…::test_lk_tracking_error_uses_exact_win_size_samples` | 37.2s / 16.6s — #182 | `src/video/tests.rs` |
| `video::…::test_lk_min_eigen_uses_exact_win_size_samples` | 35.5s / 15.7s — #182 | `src/video/tests.rs` |

**Why so few exclusions suffice.** The distribution is extremely skewed: in #81 the
top 5 tests were 82% of total runtime, while the remaining 341 tests completed in
well under two minutes combined, and nine annotations took the suite from 41 minutes
to roughly 4. The same shape returned in #182: on the 2026-10-17 `dev` run the
`std` leg spent 1699s in tests, of which `test_clahe_u16` alone was 707s and the
eight LK tests above 568s; the other 395 tests took 373s combined (`std,simd`: 1451s
→ 253s). The per-module split after the exclusions, `std` / `std,simd`: core 35s /
42s, imgproc 130s / 94s, video 182s / 96s, features2d 19s / 15s, calib3d 6s / 5s.

**Coverage check on the one borderline case.** `test_build_pyramid_with_derivatives`
exercises the Scharr path (Sobel before #130, switched to Scharr by that fix), which
under `simd` reaches the `unsafe` in `derivatives.rs:346,349`. That coverage is
**not** lost: `imgproc::tests::test_scharr` calls `scharr(&src, 1, 0, …)`
(`src/imgproc/tests.rs:154`), matching `fast_deriv_3x3`'s `TypeId == f32 &&
kernel-length == 3` trigger (true for both Sobel `ksize=3` and Scharr `ksize=-1`,
since `get_deriv_kernel` returns a 3-element kernel for both), and runs in ~0.75s
under Miri — same order of magnitude as `test_sobel`, which still covers the
Sobel path used elsewhere (`canny`, `hough_circles`, …).

### Annotation convention

Any test excluded later must use `#[cfg_attr(miri, ignore)]` with a reason comment
directly above it:

```rust
// miri: <specific reason>
#[cfg_attr(miri, ignore)]
#[test]
fn some_test() { … }
```

`cfg(miri)` is set automatically by Miri — no Cargo feature, no `Cargo.toml` change.
Under normal `cargo test` the attribute vanishes entirely.

The reason comment is mandatory so this document's exclusion list stays **derivable
by grep** instead of drifting out of sync:

```bash
grep -rn -B1 "cfg_attr(miri, ignore)" src/
```

The codebase has **zero** `#[ignore]` and no `cfg_attr` other than these, so every
match is unambiguously Miri-related. The reason comment may span several lines;
`-B1` shows only the last one, so widen the context when auditing. Tests excluded by
#164/#167 were brought under this convention by #182.

### If pulp proves wholly incompatible

Do **not** mass-annotate the ~63 tests in `src/core/simd.rs`, `src/imgproc/simd.rs`,
and `src/video/simd.rs`. That would be noise masquerading as progress. Instead: keep
the simd leg permanently `continue-on-error`, record the limitation here, and revisit
when pulp or Miri advances.

---

## 5. Known Coverage Gap

The **`parallel` + `simd`** combination is not verified by this plan.

`arithm.rs:99` and `:248` — `from_raw_parts_mut` on a rayon chunk, arguably the most
delicate `unsafe` in the codebase, since it reinterprets a *slice of a slice* being
mutated across threads — live behind `cfg(feature = "parallel")`. Neither leg enables
it, so neither leg reaches them. The sequential equivalents at `:119`/`:267` **are**
covered, and they share the same reinterpretation logic, which mitigates but does not
eliminate the gap.

Closing it requires a third leg (`--features std,parallel,simd`) relying on Miri's
thread support. Deferred: it is the slowest configuration and the most likely to
produce false positives from rayon internals. Worth a follow-up issue once the simd
leg is stable.

---

## 6. Assumptions

Measured on `x86_64-pc-windows-msvc`, nightly, 2026-08-08.

| ID | Assumption | Status |
|----|------------|--------|
| A1 | Miri's x86-64 intrinsic shims cover what pulp emits | ✅ **CONFIRMED.** 53 `core::simd` tests pass in 24s with zero unsupported-operation errors. pulp is Miri-compatible. |
| A2 | Full-suite Miri runtime fits a ~20 min budget | ❌ **REFUTED as originally run** — the unfiltered baseline took **2469s (41 min)**. ✅ **Satisfied after exclusions** (§4): 5 tests accounted for 82% of the total. |
| A3 | `Instant::now` works under Miri's virtual clock without `-Zmiri-disable-isolation` | ✅ Confirmed — no isolation errors in any run. |
| A4 | `panic = "abort"` does not affect `miri test` | ✅ Confirmed — test profile unaffected. |
| A5 | Work branches from and targets `dev` | Per `CLAUDE.md`. |

### Unanticipated finding: Miri's float non-determinism

`core::rng::tests::test_randn_determinism` failed on the first run. **This is not UB
and not a purecv bug.** Miri deliberately injects a small random error into
transcendental float operations to catch code relying on exact results. The
Box-Muller transform at `src/core/rng.rs:115-117` uses `ln()`, `cos()` and `sin()`,
and the test asserts two identically-seeded runs are bit-identical. The observed
arrays differed only in the last 1–2 ULP:

```
left:  0.18517594738681525
right: 0.18517594738681534
```

Resolved with `MIRIFLAGS: -Zmiri-deterministic-floats`, which keeps the test running
rather than ignoring it — the RNG's determinism contract is still verified. Confirmed
passing with the flag set.

### Note on what the SIMD leg actually interprets

pulp performs runtime feature detection. Under Miri it most likely takes its scalar
fallback rather than a vectorized kernel, so the AVX/SSE kernels themselves may not be
interpreted. This does **not** weaken the result that matters: the `unsafe`
`from_raw_parts`/`from_raw_parts_mut` reinterpretations in `arithm.rs` and
`derivatives.rs` sit *outside* pulp and execute regardless of which kernel pulp
dispatches to. Those are the blocks with UB risk, and they are covered.

---

## 7. Corrections to Issue #81

Recorded so the errors are not reintroduced by a later reader of the issue.

1. **`src/core/structural.rs` split/merge has no `unsafe`.** The issue lists it as an
   audit target; line 278 is a comment explaining why unsafe was *avoided*.
2. **`data_ptr`/`data_ptr_mut` live in `matrix.rs`,** not `tests.rs`, and contain no
   `unsafe`. Only their tests dereference raw pointers.
3. **"Exclude pulp SIMD" and "verify all unsafe code" are mutually exclusive** as
   written, since 100% of production `unsafe` is `simd`-gated. Resolved by the
   two-leg design (§3).
4. **`--workspace` contradicts the wasm exclusion,** and a bare `cargo miri test`
   enables `parallel` via default features, contradicting "without parallel
   features initially." Resolved by explicit `-p` and `--no-default-features` (§3).

Additionally: **badges are per-workflow, not per-job.** The issue asks for a job
inside `ci.yml` *and* a distinct Miri badge; those are incompatible. Resolved by
giving Miri its own workflow file.

---

## 8. Running Miri Locally

One-time setup:

```bash
rustup toolchain install nightly --component miri
cargo +nightly miri setup
```

Then reproduce either CI leg exactly. **Use these commands verbatim** — each flag
below is load-bearing, and dropping one produces a failure that looks like a real
problem but is not:

```bash
# Leg 1 — baseline
MIRIFLAGS=-Zmiri-deterministic-floats \
  cargo +nightly miri test -p purecv --lib --no-default-features --features std

# Leg 2 — simd
MIRIFLAGS=-Zmiri-deterministic-floats \
  cargo +nightly miri test -p purecv --lib --no-default-features --features std,simd
```

On PowerShell, set the variable separately: `$env:MIRIFLAGS="-Zmiri-deterministic-floats"`.

CI runs the same legs through nextest (§3, §9), which is also the fastest way to run a
leg locally since it uses every core. Install it once with
`cargo install cargo-nextest --locked` (or `cargo binstall cargo-nextest`), then:

```bash
MIRIFLAGS=-Zmiri-deterministic-floats \
  cargo +nightly miri nextest run -p purecv --lib --no-default-features --features std,simd
```

Add `--partition slice:1/2` (or `2/2`) to reproduce one CI shard exactly. nextest
prints each test's wall time, inflated by the concurrent tests sharing the machine
(§9), so for a threshold measurement run without `-j`/partition contention: add
`-j1`, or use the `cargo miri test` command above with `--report-time`. The
`.config/nextest.toml` profile `default-miri` (selected automatically under Miri)
flags a test as slow after 2 minutes and kills it after 10, naming it.

| Omitting… | Symptom |
|-----------|---------|
| `MIRIFLAGS=-Zmiri-deterministic-floats` | `test_randn_determinism` fails on a last-ULP float difference (§6) — looks like a real regression, isn't |
| `--lib` | Doc-tests run, adding ~13 min of ORB examples |
| `--no-default-features` | `parallel` is enabled silently, pulling rayon into the interpreter |
| `-p purecv` | `crates/wasm` is included; Miri has no wasm32 support |

To profile per-test timings, append `-- -Zunstable-options --report-time`. Note that
`--report-time` **alone is rejected** — libtest requires `-Zunstable-options` with it.

**Platform caveat.** Local runs here were `x86_64-pc-windows-msvc`; CI is
`ubuntu-latest`. Miri's intrinsic support is target-dependent, so a local pass does
not guarantee a CI pass. In practice the two agreed exactly (§9), but CI remains the
source of truth.

### How this was delivered

Shipped in [#93](https://github.com/webarkit/purecv/pull/93), in this order: plan
document → workflow → test annotations → README. The feasibility spike ran *before*
any YAML was written, which is what surfaced the 41-minute runtime, the doc-test
cost, and the float-determinism false positive — all three would have landed as
broken CI otherwise.

| # | Commit |
|---|--------|
| 1 | `doc(ci): add Miri UB verification plan (#81)` |
| 2 | `chore(ci): add Miri UB check workflow (#81)` |
| 3 | `test(core): annotate Miri-slow tests with cfg_attr(miri, ignore) (#81)` |
| 4 | `doc(readme): add Miri badge, correct SIMD unsafe claim (#81)` |
| 5 | `chore(ci): promote the Miri simd leg to a required check (#81)` |

Quality gate per `CLAUDE.md` before each commit, in order: `cargo fmt` →
`cargo clippy` (zero warnings) → `cargo test`.

---

## 9. Workflow

`.github/workflows/miri.yml` (abridged; since #182):

```yaml
name: Miri

on:
  push:
    branches: [ "main", "dev" ]
    tags: [ "v*" ]          # release.yml waits on the two aggregators below
  pull_request:
  workflow_dispatch:

env:
  CARGO_TERM_COLOR: always
  MIRIFLAGS: -Zmiri-deterministic-floats
  MIRI_PARTITIONS: 2

jobs:
  shard-baseline:                       # identical twin: shard-simd (std,simd)
    name: Miri shard (baseline, ${{ matrix.partition }}/2)
    runs-on: ubuntu-latest
    timeout-minutes: 20
    strategy:
      fail-fast: false
      matrix:
        partition: [ 1, 2 ]
    steps:
      - uses: actions/checkout@v6
      - uses: dtolnay/rust-toolchain@nightly
        with:
          components: miri
      - uses: taiki-e/install-action@cargo-nextest
      - uses: Swatinem/rust-cache@v2
        with:
          key: miri-baseline            # one key per leg, shared by its partitions
      - name: Build Miri sysroot
        run: cargo miri setup
      - name: Run Miri
        run: >-
          cargo miri nextest run -p purecv --lib
          --no-default-features --features std
          --partition slice:${{ matrix.partition }}/${{ env.MIRI_PARTITIONS }}

  baseline:                             # twin: simd → "Miri UB Check (simd)"
    name: Miri UB Check (baseline)
    needs: [ shard-baseline ]
    if: always()
    runs-on: ubuntu-latest
    steps:
      - run: test "${{ needs.shard-baseline.result }}" = "success"
```

Why the aggregators are shaped like that: `lewagon/wait-on-check-action` (used by
`release.yml`) accepts `success,skipped` by default. A job that merely `needs:` the
shards is *skipped* when one of them fails, and the release would go ahead. With
`if: always()` the aggregator runs regardless and fails unless the matrix result is
`success`, which it is only when every partition succeeded — failures, timeouts and
cancellations all become a red required check. The shard jobs can be renamed or
re-partitioned without touching `release.yml`; only the two aggregator names and the
`tags: [ "v*" ]` trigger are load-bearing.

> The file itself is authoritative; this listing is abridged. See
> `.github/workflows/miri.yml` for the full inline commentary.

### Doc-tests are excluded via `--lib`

`cargo miri test` runs doc-tests by default, and they are expensive: the two ORB
examples in `src/features2d/mod.rs` cost **582.9s and 184.7s** — 767s combined, more
than three times the entire unit-test suite. They duplicate the coverage of
`test_orb_full_pipeline`, itself already excluded on time grounds.

`--lib` is safe here because there is **no `tests/` directory** — every test in the
project lives under `src/`, so `--lib` is the whole suite. Doc-tests continue to be
exercised by the ordinary `cargo test` in `ci.yml`; they are skipped only under Miri.

This cost was invisible during the first baseline run, which aborted at the
`test_randn_determinism` failure before reaching the doc-test phase.

### Operational notes

- **Broken nightly.** If `cargo miri setup` fails because a given nightly shipped
  without miri, temporarily pin a known-good date
  (`dtolnay/rust-toolchain@nightly-YYYY-MM-DD`) and revert once upstream recovers.
- **Timeout.** 20 minutes per shard, against a few minutes of measured work (see
  the #182 table below); the budget is headroom for a cold cache and a slow runner.
  A single pathological test no longer needs the job timeout to be noticed: the
  `default-miri` nextest profile kills any test that runs for 10 minutes and reports
  its name (`.config/nextest.toml`).
- **Partition count.** `MIRI_PARTITIONS` in the workflow env plus the two
  `partition:` matrices. Raise all three together when a shard creeps past ~10 min;
  `slice:` keeps every test in exactly one bucket for any N.
- **Timing a test.** nextest prints each test's wall time in the job log. For a
  whole-leg profile, download the shard logs and sort by that column, remembering
  that four tests run at once on the runner, so single-test figures read 1.5–2×
  higher than under the sequential command (see the #182 table below).
- **Reproducing locally.** Use the exact commands in §8.
- **Measured runtime** (`x86_64-pc-windows-msvc`, 2026-08-08):

  Final figures are the two legs exactly as CI runs them, on `dev` @ v0.7.0:

  | Run | Result | Test time |
  |-----|--------|-----------|
  | **Leg 1 — `std`, `--lib`** | **299 passed, 0 failed, 9 ignored** | **271s** |
  | **Leg 2 — `std,simd`, `--lib`** | **355 passed, 0 failed, 9 ignored** | **229s** |

  Neither leg reported undefined behaviour, an unsupported operation, or a Stacked
  Borrows violation. **All six production `unsafe` blocks reachable without
  `parallel` are verified UB-free.**

  For reference, the discarded configurations that motivated the exclusions:

  | Run | Result | Test time |
  |-----|--------|-----------|
  | Baseline before exclusions | 307 passed, 1 failed | 2469s (41 min) |
  | Doc-tests (now excluded via `--lib`) | passed | 767s for 2 of them |

  Note leg 1 is *slower* than leg 2 despite running 56 fewer tests: under `simd`,
  several kernels process data in chunks that cost Miri less to interpret than the
  equivalent scalar loops.

- **Measured runtime after #182** (`ubuntu-latest`). Before: the two sequential
  `cargo miri test` jobs on `dev` @ v0.11.0, run 39021153764 (2026-10-17); the same
  commit on the release PR #186 (run 39021101772) had its baseline job cancelled at
  30 min 19 s. After: the nextest shards on the #182 PR.

  | Job | Before (one sequential job) | After (per shard) |
  |-----|-----------------------------|-------------------|
  | `baseline` | 29 min 34 s (tests 1699 s) | 1/2: 5 min 56 s (tests 283 s) · 2/2: 6 min 18 s (tests 303 s) |
  | `simd` | 25 min 22 s (tests 1451 s) | 1/2: 6 min 29 s (tests 317 s) · 2/2: 4 min 1 s (tests 185 s) |

  PR #187, run 39050813686 (2026-10-17, cold cache for the new per-leg keys).
  Setup is ~1 min per shard: toolchain 10 s, nextest install 1 s, `cargo miri setup`
  20 s, test build 28 s. Counts: baseline 185 + 185 = 370 run, simd 213 + 213 = 426
  run, 34 ignored on each leg (nextest reports 219 / 247 "skipped" per shard: the
  34 ignored plus the other partition's tests), every test in exactly one shard.
  The aggregators `Miri UB Check (baseline)` / `(simd)` completed 2–4 s after their
  last shard. The longest job went from 29 min 34 s to 6 min 29 s, a 4.6× cut, with
  the 20-minute shard timeout leaving ~3× headroom.

  Per-test wall times under nextest are **not** comparable with the sequential
  figures in §4: four tests share the runner, so each one reads 1.5–2× slower
  (`test_harris_at_matches_corner_harris_everywhere` 59 s here vs 31 s alone, the
  slowest remaining test on either leg). Use the sequential leg (§8 without
  `--partition`, or `cargo miri test`) when measuring a test against the 30 s
  threshold.

---

## 10. README Changes

Badge, placed after the existing Rust CI badge:

```markdown
[![Miri](https://github.com/webarkit/purecv/actions/workflows/miri.yml/badge.svg)](https://github.com/webarkit/purecv/actions/workflows/miri.yml)
```

**Accuracy fix.** The Portable SIMD bullet currently reads *"Zero `unsafe`, zero
`#[cfg(target_arch)]`."* The second half is true; the first is not — `arithm.rs:99`
and `derivatives.rs:346` use `from_raw_parts` to feed pulp. Proposed replacement:

> **Portable SIMD:** Optional SIMD acceleration via [`pulp`](https://crates.io/crates/pulp) —
> auto-detects x86 SSE/AVX, ARM NEON, and WASM `simd128` at runtime. Zero
> `#[cfg(target_arch)]`, and the few `unsafe` slice reinterpretations feeding the
> SIMD kernels are verified UB-free by [Miri](https://github.com/rust-lang/miri) in CI.

This turns a claim Miri would contradict into one Miri actively backs.

---

## 11. Decision Log

| # | Decision | Alternatives considered | Rationale |
|---|----------|-------------------------|-----------|
| 1 | Two runs: `std` + `std,simd`, the latter advisory until proven on Linux, then required | simd-only required; no-simd only (as issue) | Only option that inspects real `unsafe` without risking a permanently-red required gate |
| 2 | Measure runtime before excluding anything | Pre-emptive `cfg_attr` ignores; shrink inputs under `cfg(miri)` | Don't disable tests on speculation; §6/A2 shows the risk is lower than feared |
| 3 | `-p purecv --no-default-features --features std[,simd]` | `--workspace`; `--workspace --exclude`; default features | Excludes wasm structurally; makes the absent `parallel` explicit rather than accidental |
| 4 | No `MIRIFLAGS`; rely on Miri defaults | Issue's `-Zmiri-strict-provenance`; Tree Borrows; both models | Flag is redundant and deprecated; Stacked Borrows is the stricter guarantee |
| 5 | Floating `@nightly` + documented pin fallback | Dated pin; permanently non-blocking job | Lowest maintenance for a solo maintainer; rare breakage is recoverable in one line |
| 6 | Triggers match existing CI (push + PR), plus `workflow_dispatch` | Cron-only; PR + nightly schedule | Feedback at review time; dispatch allows re-running the simd leg without pushing |
| 7 | Separate `.github/workflows/miri.yml` | Job inside `ci.yml`; manual shields.io badge | Badges are per-workflow; also isolates nightly flakiness from the main CI badge |
| 8 | Plan document at `.agents/MIRI_PLAN.md` | Workflow comments only; both | Matches existing convention (`SIMD_PLAN_v2.md`, `roadmap.md`) |
| 9 | Matrix strategy, one job | Two discrete jobs; one job with two sequential steps | Matches the problem shape; parallel legs; promotion is a one-word diff. YAGNI on divergence that isn't needed yet |

### Amendments after the spike

| # | Decision | Alternatives considered | Rationale |
|---|----------|-------------------------|-----------|
| 10 | Set `MIRIFLAGS: -Zmiri-deterministic-floats` — **revises #4** | `#[cfg_attr(miri, ignore)]` on `test_randn_determinism`; leave the failure | #4 said "no flags", but measurement found a genuine need. The flag keeps the RNG determinism contract under test instead of disabling it, and targets a documented Miri behaviour rather than a real defect |
| 12 | Shard each leg with `cargo miri nextest run --partition slice:k/2` (#182) | Job matrix per module with `cargo miri test -- <filter>`; raise the timeout; exclusions only | Exclusions alone left the legs at ~14–20 min and every new test lands in the same two serial jobs. nextest uses the runner's idle vCPUs (Miri is single-threaded per process) and `slice:` partitions need no hand-kept module list. A per-module matrix would have been 8 sequential jobs each paying setup, bottlenecked by whichever module grows. Cost: one prebuilt tool install (~10 s) and Miri's cross-test data-race detection, which needs shared state the suite doesn't have |
| 13 | Per-leg aggregator jobs with `if: always()` + explicit result check (#182) | Rename the checks in `release.yml`; plain `needs:` aggregator | Keeps `release.yml` untouched. A plain `needs:` aggregator is *skipped* on a red shard, and `wait-on-check-action` accepts `skipped` by default — the release would proceed |
| 14 | `.config/nextest.toml` `default-miri` slow-timeout 10 min (#182) | None (job timeout only) | A future `test_clahe_u16` fails with its name in minutes instead of cancelling the job at 20 min with no culprit |
| 11 | Exclude the 9 tests over 30s — **implements #2** | Exclude >10s (15 tests); shrink inputs under `cfg(miri)`; exclude nothing and raise the timeout | Measurement showed an extreme skew: 5 tests = 82% of runtime, the rest (341 tests as of #131) well under two minutes combined. Nine annotations buy a 10× speedup; none of the nine touch `unsafe`, so no UB coverage is lost |

---

## 12. Acceptance Criteria Mapping

| Criterion (issue #81) | Addressed by |
|-----------------------|--------------|
| Miri CI job passes on `dev` | ✅ Both legs green locally (§9). Pending confirmation on `ubuntu-latest`. |
| All existing unsafe verified UB-free, or documented exceptions | ✅ 6 of 6 reachable production blocks verified clean. Exception: the `parallel`-only pair, documented in §5. |
| Incompatible tests annotated `#[cfg_attr(miri, ignore)]` | ✅ 9 tests, each with a reason comment (§4). Excluded for runtime, not incompatibility — nothing in the suite proved Miri-incompatible. #130 and #138 together added three more tests exercising the same unsafe Scharr fast path, but all three measured well under the 30s threshold (§4) and are not excluded. |
| Plan document identifying included/excluded code with rationale | ✅ This document, tracked in git via a `.gitignore` exception. |

# Core Parity Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Every issue follows superpowers:test-driven-development.

**Goal:** Close the `core` step of #268: make `core` match OpenCV 4.10.0 on the 19 open divergences (#192, #208–#224, #226) in four PRs into `dev`.

**Architecture:** One worktree and one branch per PR, cut from `origin/dev`, run one after the other (all four touch `src/core/arithm.rs` or `tests/parity_core.rs`). One commit per issue inside each PR. The shared `SaturateCast` from #189 is the write path for every integer output; a new `SaturatingOps` trait covers integer overflow.

**Tech Stack:** Rust, `num-traits`, `pulp` (`simd`), Rayon (`parallel`), `rustfft` (`fft`); OpenCV 4.10.0 through `cpp_ref/.venv` and pinned opencv.js as the measuring reference only.

**Spec:** #268 ("core" step); `docs/parity/4.10.0/core.md` (frozen audit rows); "Fixing a parity issue" in `CONTRIBUTING.md`; rule 11 of `docs/parity/AGENTS.md`; the ignored tests in `tests/parity_core.rs`.

## Scope corrections to the request

- The core issues are **#192, #208–#224, #226** (19 issues). #193–#204 in the numeric range are `features2d` (later step of #268); #205–#207 are not part of this step.
- **#225 is already done** (PR #275) but still OPEN, because `Closes` does not fire on a merge into `dev`. Close it by hand with the merge commit; it needs no code.
- **#268 and `CONTRIBUTING.md` say one issue per PR; the maintainer asked for 3–4 PRs.** This plan follows the maintainer. Each PR lists every issue it closes and keeps one commit per issue, so a single issue can still be reverted.

## Global Constraints

- PRs start from and target `dev`. Worktree per PR via superpowers:using-git-worktrees. Nothing is committed on `main`.
- Never edit `docs/parity/4.10.0/`. Every other rule of `docs/parity/AGENTS.md` and `cpp_ref/AGENTS.md` applies.
- A fix for #N removes the `#[ignore]` of **every** test citing #N and of no other; it deletes the paired `*_current_behaviour_*` pin.
- New expected values come only from a script in `scripts/opencv_ref/` run in `cpp_ref/.venv` (IPP off) or pinned opencv.js; the test comment names the script. No value is computed by hand.
- `cv2` does not bind `Mat::convertTo`, `minMaxIdx` or `Mat::ones/eye`: use opencv.js (`core_convert_to.js`, `core_mat_initializers.js`).
- Reference tooling is absent from the main checkout. Reuse `.claude/worktrees/parity-imgproc-5fff9e/cpp_ref/` (has `src`, `.venv`, `node_modules`) after running its bootstrap and seeing both `verify … OK` lines. Read-only; never build inside `cpp_ref/src`.
- No `panic!` in library code; `Result<T, PureCvError>` everywhere. New `.rs` files need the LGPL header.
- Conventional commits; API breaks use `fix(core)!:` as PR #275 did. Every commit and PR ends with the attribution the session specifies.
- After a merge into `dev`, close each issue by hand: `gh issue close N --comment` naming the merge commit. Merging stays with the maintainer.

### Per-issue cycle (referenced by every task as "the cycle")

1. Write the listed new tests and remove the `#[ignore]` of the listed existing ones.
2. Run `cargo test --test parity_core <name>` (add `--features fft,transforms` for dft rows). Expected: FAIL for the audit's reason.
3. Implement.
4. Run the same command. Expected: PASS.
5. Fix or delete the listed old unit tests that pin the old behaviour.
6. `cargo fmt`, `cargo clippy --all-targets -- -D warnings`, commit.

### Per-PR gate (all must pass before the review)

```bash
cargo fmt --check
cargo clippy --all-targets -- -D warnings
cargo clippy --no-default-features -- -D warnings
cargo test --workspace
cargo test --workspace --features simd
cargo test --workspace --features parallel
cargo test --workspace --no-default-features --features std
cargo test --workspace --features fft,transforms
cargo build --no-default-features --target thumbv7em-none-eabihf
cargo test --test opencv_reference_links
python3 scripts/parity_xfail.py
python3 scripts/parity_xfail.py --features simd
```

`parity_xfail.py` fails if a still-ignored test passes. If a test citing a later issue flips, move that issue into this PR rather than editing the test.

### Per-PR closing

1. Run the gate; record the commands and results in the PR body.
2. Run the benches named in the PR's last task on `dev` and on the branch; report any regression above 10%.
3. Final review on the whole branch diff: superpowers:requesting-code-review with a fresh reviewer given the diff and the issue rows, not this plan's conclusions. Then the advisor. Apply findings via superpowers:receiving-code-review and re-run the gate.
4. Open the PR into `dev` with `Closes #…` for each issue, listing every behaviour change and API break. Read the Qodo review from the PR comments with `gh`.

## Decisions that need the maintainer

| # | Decision | Recommendation |
|---|---|---|
| D1 | `determinant`, `sum`, `mean` return a plain value, so OpenCV's assertion cannot be expressed | Change them to `Result` (breaking, `!`). Affects `invert`, `mean_std_dev`, benches, wasm if bound, and passing pins at `tests/parity_core.rs:488,935`. |
| D2 | `dtype` and `REDUCE_SUM2` in #215/#216 cannot fit a same-`T` signature; the ignored tests fix the same-`T` form | Keep `normalize`/`reduce` same-`T`. Add `normalize_to::<T, U>` and `reduce_to::<T, U>`, plus `ReduceTypes::Sum2`. `normalize` returns `Err` for a `dtype` other than `-1` or `T`'s depth instead of ignoring it. |
| D3 | Items listed under "missing" in the audit and linked to these issues but outside their titles: `check_range(quiet, pos)` (#213), `PSNR(R)` (#218) | Leave them out and file one follow-up issue after approval, so closing #213/#218 does not orphan them. |
| D4 | `gemm` on integer types (#210): OpenCV asserts 32F/64F, purecv accepts them | Keep accepting, write through `SaturateCast`. Rejecting would break callers for no result that can be compared. |

The tasks assume these recommendations.

## Review Focus

1. **`i32` overflow may not saturate in OpenCV.** The audit says `add` saturates "on every depth", but OpenCV's documentation for `add` carries a note excluding `CV_32S`, and `c_add` on `int` adds in `int`. Measure before pinning (Task 1.1); the result must never panic.
2. **SIMD lanes vs short fixtures.** Audit fixtures are shorter than a vector, so they may only reach scalar tails. Every rounding or saturation fix adds a test with at least 64 elements, run under `--features simd`, with tie values in lane positions.
3. **Working precision of the intermediate.** `add_weighted` and `divide` compute in `float` in OpenCV; computing in `f64` gives different rounding on values such as `0.1·a + 0.2·b`. Each such task pins a seeded fixture measured from OpenCV, not only the ties.
4. **64-bit element types.** `i64`/`u64` have no OpenCV depth and lose precision through `f64`. Identity-range values above 2⁵³ must survive `add`, `subtract`, `multiply` and same-type `convert_to` exactly.
5. **Oversized borders and thread-local RNG state.** `copy_make_border` with borders wider than the image on every border type, and `kmeans` under `--test-threads=1`, where tests share the thread-local generator.

---

# PR 1: `fix/core-parity-saturate`: integer outputs saturate and round

Closes #208, #209, #210, #211, #212, #192, #215, #216, #220. Order is fixed: #208 is the debug-build panic, #192 sets the conversion path the later tasks reuse.

### Task 1.1: `SaturatingOps` and #208 (`add`, `subtract`, `multiply`, `absdiff`)

**Files:** Modify `src/core/saturate.rs`, `src/core.rs`, `src/core/arithm.rs:400-480,700-725,990-1005`, `src/imgproc/morph.rs:656` (bound). Create `scripts/opencv_ref/core_saturating_ops.py`. Test `tests/parity_core.rs`.

**Interfaces:** Produces, re-exported from `purecv::core`:
```rust
pub trait SaturatingOps: Copy {
    fn sat_add(self, rhs: Self) -> Self;
    fn sat_sub(self, rhs: Self) -> Self;
    fn sat_mul(self, rhs: Self) -> Self;
    fn sat_absdiff(self, rhs: Self) -> Self;
}
```
Implemented for `u8, i8, u16, i16, u32, i32, u64, i64, f32, f64`. Floats use plain IEEE. Integers use `saturating_*` on native width, never through `f64`.

- [ ] **Measure first.** The script prints `add`, `subtract`, `multiply`, `absdiff` for i8, u16, i16, i32 at the type bounds. Implement `i32` as measured: `wrapping_*` if OpenCV wraps, with the reason in the doc comment. `u32`/`u64`/`i64` have no OpenCV depth and saturate.
- [ ] **The cycle.** Un-ignore `add_u8_saturates`, `subtract_u8_saturates_at_zero`, `multiply_u8_saturates` (drop the `cfg_attr`). New tests: `add_sub_mul_bounds_match_opencv` (values from the script), `add_u8_saturates_across_simd_lanes` (64 elements, `128 + 128 → 255`, `1 − 3 → 0`), `add_mul_i64_keep_exact_values_above_2_53`, `absdiff_i8_bounds_match_opencv`.
- [ ] **Commit:** `fix(core)!: saturate integer add, subtract, multiply and absdiff` (the `T` bound gains `SaturatingOps`). `Closes #208`.

### Task 1.2: #192 `convert_to` and `convert_to_scaled`

**Files:** Modify `src/core/matrix.rs:530-550`, `src/core/dynamic.rs:457-485`. Extend `scripts/opencv_ref/core_convert_to.js`.

**Interfaces:** Produces `Matrix<T>::convert_to::<U>() -> Result<Matrix<U>>` with `U: SaturateCast + Default + Clone`, and `Matrix<T>::convert_to_scaled::<U>(alpha: f64, beta: f64) -> Result<Matrix<U>>`. Out-of-range input never fails.

- [ ] **The cycle.** Un-ignore `convert_to_f32_to_u8_rounds_half_even_and_saturates` (`[0, 2, 2, 4, 254, 255, 255, 255, 0]`). Delete `convert_to_f32_to_u8_current_behaviour_truncates_and_errors`. New: `convert_to_scaled_matches_opencv` (u8, i8, u16, i16, i32 with ties, out-of-range, NaN → 0; opencv.js values), `convert_to_i64_identity_is_exact`, `dynamic_convert_to_saturates`. The same-type and integer-to-wider paths must not go through `f64` when `T`'s values do not fit it exactly.
- [ ] **Check internal callers still build:** `src/imgproc/{edge,feature,hough}.rs`, `src/video/optical_flow.rs` (f32/f64 targets, unchanged results).
- [ ] **Commit:** `fix(core)!: make convert_to round half to even and saturate`. `Closes #192`.

### Task 1.3: #209 `divide`

**Files:** `src/core/arithm.rs:484-510`, `src/core/simd.rs:321,733`. Extend `scripts/opencv_ref/core_convert_arith.py`.

- [ ] **Read** `op_div_scale` in `arithm.simd.hpp` (pinned link in the row) for the working type per depth, and port that precision.
- [ ] **The cycle.** Un-ignore `divide_u8_rounds_the_float_quotient` (`7/2 → 4`, `250/7 → 36`, `100/3 → 33`, `x/0 → 0`), `divide_f32_by_zero_is_ieee` (`1/0 → inf`, `0/0 → nan`). Delete `divide_f32_by_zero_current_behaviour_is_zero`. New: `divide_f32_by_zero_is_ieee_across_simd_lanes` (64 elements), `divide_u8_seeded_fixture_matches_opencv`, `divide_i16_i32_match_opencv`.
- [ ] **Commit:** `fix(core): round integer divide and make float division IEEE`. `Closes #209`.

### Task 1.4: #210 `*_scalar` and `gemm`

**Files:** `src/core/arithm.rs:328-395,1816-…`.

- [ ] **The cycle.** Un-ignore `add_scalar_u8_rounds_and_saturates_the_scalar` (`+10.7` adds 11, `+10.5` adds 10, `+300 → 255`, `−1.5` subtracts 2), `multiply_divide_scalar_u8_round_and_saturate` (`9·0.3 → 3`, `128/3 → 43`, `127/0.4 → 255`, `/0 → 0`). New: `gemm_u8_output_saturates`. The `add`/`subtract` scalar converts the scalar first through `saturate_cast`; `multiply`/`divide` compute first and convert the result.
- [ ] **Commit:** `fix(core): round and saturate scalar operands like OpenCV`. `Closes #210`.

### Task 1.5: #211 `add_weighted`

**Files:** `src/core/arithm.rs:1622-1680`.

- [ ] **The cycle.** Un-ignore `add_weighted_u8_rounds_half_even_and_saturates` (`51.5 → 52`, `3.5 → 4`, `128.5 → 128`, `910 → 255`). New: `add_weighted_u8_seeded_fixture_matches_opencv` (64+ elements; float working precision, per the row's `c_add` link).
- [ ] **Commit:** `fix(core): round and saturate add_weighted integer outputs`. `Closes #211`.

### Task 1.6: #212 `convert_scale_abs`

**Files:** `src/core/arithm.rs:1782-…`, `src/core/simd.rs:414-423,823-830`.

- [ ] **The cycle.** Un-ignore `convert_scale_abs_rounds_half_to_even` (`0.5 → 0`, `2.5 → 2`, i16 `1·0.5 → 0`). New: `convert_scale_abs_ties_across_simd_lanes` (64 f32 and f64 values). Replace `.round()` with `round_half_even` in the scalar path **and** both simd kernels. Do not touch the other `.round()` calls in `simd.rs`; they are not this function. `1e10` keeps giving 255 (the x86 artefact is not ported).
- [ ] **Commit:** `fix(core): round convert_scale_abs ties to even`. `Closes #212`.

### Task 1.7: #215 `normalize` (D2)

**Files:** `src/core/arithm.rs:1263-1410`, `src/core.rs`.

**Interfaces:** Produces `normalize_to::<T, U>(src: &Matrix<T>, dst: &mut Matrix<U>, alpha: f64, beta: f64, norm_type: NormTypes, mask: Option<&Matrix<u8>>) -> Result<()>`. `normalize` keeps its signature and delegates.

- [ ] **The cycle.** Un-ignore `normalize_u8_minmax_rounds_like_convert_to` (`[10, 20, 30, 250] → [0, 11, 21, 255]`), `normalize_minmax_swapped_bounds_match_sorted_bounds`. New: `normalize_to_u8_to_f32_matches_opencv`, `normalize_rejects_mismatched_dtype`. Check the examples passing `-1` and the passing f32 pins are unchanged.
- [ ] **Commit:** `fix(core): round normalize output, sort bounds and honour dtype`. `Closes #215`.

### Task 1.8: #216 `reduce` (D2)

**Files:** `src/core/arithm.rs:1414-1570`, `src/core/types.rs:1076`, `src/core.rs`.

**Interfaces:** Produces `reduce_to::<T, U>(src: &Matrix<T>, dim: i32, op: ReduceTypes) -> Result<Matrix<U>>` and `ReduceTypes::Sum2 = 4`. Copy the accepted depth pairs from `cv::reduce` (row link); unsupported pairs return `Err`.

- [ ] **The cycle.** Un-ignore `reduce_avg_u8_rounds_half_to_even` (`1.5 → 2`, `67.7 → 68`), `reduce_sum_u8_into_u8_is_rejected`. New: `reduce_to_sum_u8_to_i32_matches_opencv` (300 stays 300), `reduce_sum2_matches_opencv`, `reduce_to_rejects_unsupported_depth_pair`.
- [ ] **Commit:** `fix(core)!: round reduce averages, reject overflowing sums, add reduce_to`. `Closes #216`.

### Task 1.9: #220 `transform` and PR closing

**Files:** `src/core/arithm.rs:2824-…`.

- [ ] **The cycle.** Un-ignore `transform_u8_rounds_half_to_even` (`18.9 → 19`, `229.5 → 230`). New: `transform_u8_saturates_out_of_range`.
- [ ] **Commit:** `fix(core): round and saturate transform integer outputs`. `Closes #220`.
- [ ] **PR closing.** Benches: `add`, `subtract`, `multiply`, `divide`, `add_weighted`, `convert_scale_abs`, `normalize` in `benches/arithm_bench.rs`.

---

# PR 2: `fix/core-parity-edges`: flags and edge semantics

Closes #213, #217, #218, #219, #222, #226.

### Task 2.1: #226 `CV_LOG2` and #213 `check_range`

These are two commits, one for each issue.

**Files:** `src/core/constants.rs:55`, `src/core/tests.rs:1455,1469-1478,1518`, `src/core/arithm.rs:1997`.

- [ ] **#226 cycle.** There is an ignored test `cv_log2_is_ln_2`; un-ignore it. Set `CV_LOG2 = core::f64::consts::LN_2`. Update the three unit tests in `src/core/tests.rs`; the one asserting `CV_LOG2 * CV_LN2 == 1` no longer holds.
- [ ] **Commit:** `fix(core)!: CV_LOG2 is ln 2, matching cvdef.h`. `Closes #226`.
- [ ] **#213 cycle.** Un-ignore `check_range_upper_bound_is_exclusive`. New: `check_range_lower_bound_is_inclusive`. Use `min <= v && v < max`.
- [ ] **Commit:** `fix(core)!: check_range upper bound is exclusive`. `Closes #213`.

### Task 2.2: #217 `has_non_zero`

**Files:** `src/core/arithm.rs:1574`, `src/core.rs`.

**Interfaces:** Produces `has_non_zero<T>(src: &Matrix<T>) -> Result<bool>`: single channel required (`Err` otherwise), true if any element is non-zero. `has_no_zero` keeps its predicate, with a doc comment saying it is purecv-only and naming `has_non_zero`.

- [ ] **The cycle.** New: `has_non_zero_matches_opencv` (values from `core_stats.py` `non_zero`), `has_non_zero_rejects_multi_channel`, `has_non_zero_empty_is_false` (measure the empty case first), `has_no_zero_keeps_its_predicate`.
- [ ] **Commit:** `feat(core): add has_non_zero matching hasNonZero`. `Closes #217`.

### Task 2.3: #218 edge semantics (D1)

**Files:** `src/core/matrix.rs:650-720`, `src/core/dynamic.rs:125`, `src/core/arithm.rs:1008,1066,1085,2186,2330`, `src/core/structural.rs:125`, `src/core/dft.rs:96`, `src/core/metrics.rs:71`, `benches/arithm_bench.rs:105,108`, `src/core/tests.rs:324,328,393,801,1309,1318`, `src/core/arithm.rs:3803,3808`, `tests/parity_core.rs:488-489,935-937`.

**Interfaces:** `sum<T>(&Matrix<T>) -> Result<Scalar<f64>>`, `mean<T>(&Matrix<T>) -> Result<Scalar<f64>>`, `determinant<T>(&Matrix<T>) -> Result<f64>`.

- [ ] **The cycle.** Un-ignore all four tests citing #218: `ones_and_eye_multi_channel_set_only_channel_0`, `flip_nd_negative_axis_counts_from_the_end`, `psnr_identical_inputs_is_finite` (361.20199909921956), `get_optimal_dft_size_edges_match_opencv` (`0 → 1`, `−5 → −1`, `2125764000 → −1`, `i32::MAX → −1`). New: `sum_mean_reject_more_than_four_channels`, `determinant_rejects_non_square_and_multi_channel`, `flip_nd_axis_out_of_range_is_err`, `dynamic_ones_sets_only_channel_0`.
- [ ] **`DynamicMatrix::ones` fills all channels with its own code and is not covered by the audit test;** change it with `Matrix::ones`. Grep every internal `Matrix::ones` / `Matrix::eye` with more than one channel before changing them.
- [ ] **Commit:** `fix(core)!: match OpenCV edge semantics for initializers, stats, flip_nd, dft size and psnr`. `Closes #218`. The determinant pivot threshold in the row stays unchanged; say so in the PR body.

### Task 2.4: #219 `invert`

**Files:** `src/core/arithm.rs:2310-2335`, unit tests asserting the returned determinant.

- [ ] **The cycle.** Un-ignore `invert_lu_returns_one_on_success`. New: `invert_singular_returns_zero_and_zero_matrix`. LU returns `1.0` on success, `0.0` when singular. `src/imgproc/geometric.rs:293` ignores the value, so it needs no change.
- [ ] **Commit:** `fix(core)!: invert returns a non-zero flag, not the determinant`. `Closes #219`.

### Task 2.5: #222 `copy_make_border` and PR closing

**Files:** `src/core/structural.rs:315-…`, `examples/dft_example.rs:93`, `examples/structural_ops.rs:66`, `src/core/tests.rs:238`. Extend `scripts/opencv_ref/core_misc.py`.

**Interfaces:** Keeps `border_type: i32` (the ignored test passes `4` and `1`). Interprets `border_type & !16` as `BorderTypes` and maps every index through `utils::border_interpolate`. `Transparent` and unknown codes return `Err`.

- [ ] **The cycle.** Un-ignore `copy_make_border_uses_opencv_border_codes`. New: `copy_make_border_all_codes_match_opencv` (0, 1, 2, 3, 4, and 16-flagged, multi-channel), `copy_make_border_wider_than_image_matches_opencv` (`top = 5` on 2 rows, each code, values from the script), `copy_make_border_transparent_is_err`.
- [ ] **Audit every caller's code:** a caller passing `1` for REFLECT must change to `2`.
- [ ] **Commit:** `fix(core)!: copy_make_border follows BorderTypes and no longer panics on oversized borders`. `Closes #222`.
- [ ] **PR closing.** Benches: `sum`, `mean`.

---

# PR 3: `fix/core-parity-dft`: transforms

Closes #224, #223. #224 goes first, so the CCS round trip is written against the final `idft`.

### Task 3.1: #224 `idft`

**Files:** `src/core/dft.rs:315-321` and its unit tests.

- [ ] **The cycle.** Un-ignore `idft_does_not_scale_by_default` (→ `12·src` on the 3×4 fixture). `idft = dft(src, flags | DFT_INVERSE, nonzero_rows)`. Correct the doc comment. Unit tests relying on the implicit scale pass `DFT_SCALE` explicitly.
- [ ] **Commit:** `fix(core)!: idft no longer scales by default`. `Closes #224`.

### Task 3.2: #223 CCS packing and PR closing

**Files:** `src/core/dft.rs:171-315` and its unit tests, `src/core/tests.rs`. Extend `scripts/opencv_ref/core_dft_dct.py`.

- [ ] **Read** `cv::dft` and the CCS layout in `dxt.cpp` (row link) before changing code. Measure 1×N, N×1, N×M for each even/odd combination, `DFT_ROWS`, `nonzero_rows`, f32 and f64.
- [ ] **The cycle.** Un-ignore `dft_real_input_returns_ccs_by_default` (the 12-value `CCS` table). New: `dft_ccs_shapes_match_opencv` (one case per measured shape), `dft_ccs_rows_flag_matches_opencv`, `idft_of_ccs_with_scale_round_trips`, `idft_of_ccs_real_output_matches_opencv`, `dft_complex_output_is_unchanged`.
- [ ] **Single-channel `DFT_INVERSE` input is read as CCS.** Check that `examples/dft_example.rs` and `benches/dft_bench.rs` already pass `DFT_COMPLEX_OUTPUT` and stay unchanged.
- [ ] **Commit:** `fix(core)!: dft of real input returns CCS by default`. `Closes #223`.
- [ ] **PR closing.** Benches: `benches/dft_bench.rs`.

---

# PR 4: `feat/core-parity-options`: missing options

Closes #214, #221.

### Task 4.1: #214 `norm`

**Files:** `src/core/arithm.rs:1111-1260`, `src/core.rs`. Extend `scripts/opencv_ref/core_stats.py`.

**Interfaces:** `norm` gains `L2Sqr`, `Hamming`, `Hamming2`. Produces `norm_diff<T>(src1, src2, norm_type: NormTypes, mask: Option<&Matrix<u8>>) -> Result<f64>` and `norm_relative<T>(src1, src2, norm_type, mask) -> Result<f64>`, the latter being `norm_diff / (norm(src2) + DBL_EPSILON)` per `norm.cpp`. `NormTypes` is a plain enum, so `NORM_L2 | NORM_RELATIVE` cannot be one value; hence a separate function.

- [ ] **The cycle.** Un-ignore `norm_l2sqr_and_hamming_are_supported_by_opencv` (177.5, 12, 7). New: `norm_diff_matches_opencv` (u8 without overflow at `0 − 255`, f32, masked, multi-channel), `norm_relative_matches_opencv`, `norm_hamming_rejects_non_u8`, `norm_diff_rejects_shape_mismatch`. Copy the mask and depth rules for Hamming from `norm.cpp`.
- [ ] **Commit:** `feat(core): add L2SQR, Hamming and the two-array and relative norm`. `Closes #214`.

### Task 4.2: #221 `kmeans` and PR closing

**Files:** `src/core/arithm.rs:3327-…,3641-3680`. Extend `scripts/opencv_ref/core_kmeans.py`.

- [ ] **Read** `cv::kmeans` and `generateRandomCenter` (row links). Port the order of `theRNG()` draws so the same seed gives OpenCV's result, using `crate::core::rng`. The thumbv7em build must still pass; if `kmeans` cannot reach the RNG without `std`, stop and report.
- [ ] **The cycle.** Un-ignore `kmeans_random_centers_three_attempts_find_the_global_partition` (compactness 9.5). New: `kmeans_seeded_matches_opencv` (`cv2.setRNGSeed(42)`: labels, centers, compactness; attempts 1, 3, 5), `kmeans_max_iter_is_clamped_to_2_100`, `kmeans_epsilon_compares_squared_shift`, `kmeans_out_of_range_initial_labels_is_err`. The ignored test sets no seed: confirm it passes under `--test-threads=1` too; if it depends on state, report rather than editing its expected value. The existing PP and initial-labels pins (9.5, 543.1666) must still pass.
- [ ] **Commit:** `fix(core)!: kmeans draws centers and clamps criteria like OpenCV`. `Closes #221`.
- [ ] **PR closing.** After the merge of this last PR, tick the core box in #268 only when all 19 issues and #225 are closed.

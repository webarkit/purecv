> **Frozen snapshot:** purecv at `a6658a9` audited against OpenCV 4.10.0. Do not edit; a later reference version gets its own folder.
> See [`docs/adr/0001-opencv-parity-audit.md`](../../adr/0001-opencv-parity-audit.md).

# Parity audit series: purecv ↔ OpenCV 4.10.0

The module-by-module comparison of purecv with OpenCV **4.10.0**, tracked in
[#172](https://github.com/webarkit/purecv/issues/172) (tasks B0–B5), written with the method
in [`../README.md`](../README.md). This folder is the record of that series as it was
completed on 2026-10-21; it is not updated afterwards. The live state of the issues it filed
is the GitHub milestone
[Parity audit: purecv ↔ OpenCV 4.10.0](https://github.com/webarkit/purecv/milestone/8), not the
issue map below.

- **Reference:** OpenCV 4.10.0, opencv commit `71d3237a093b60a27601c20e9ee6c3e52154e8b1`,
  opencv_contrib commit `1ed3dd2c53888e3289afdb22ec4e9ebbff3dba87`
  ([`CONTRIBUTING.md#opencv-reference-version`](../../../CONTRIBUTING.md#opencv-reference-version)).
- **Plan:** Part B of
  [`2026-10-11-opencv-4.10.0-reference-and-parity-audit.md`](../../superpowers/plans/2026-10-11-opencv-4.10.0-reference-and-parity-audit.md).
  Tasks B1–B5 hold the indicative inventory, the seeded rows, the priority rows and the
  `missing` rows of each module.
- **Rules for reading and measuring OpenCV:** [`cpp_ref/AGENTS.md`](../../../cpp_ref/AGENTS.md).

## Documents

| Module | Document | Sub-issue | Progress |
|---|---|---|---|
| method | [`docs/parity/README.md`](../README.md) | [#172](https://github.com/webarkit/purecv/issues/172) (umbrella) | done |
| core | [`docs/parity/4.10.0/core.md`](core.md) | [#173](https://github.com/webarkit/purecv/issues/173) | done (PR #206 merged, #173 closed; 19 issues filed: #208–#226) |
| imgproc | [`docs/parity/4.10.0/imgproc.md`](imgproc.md) | [#174](https://github.com/webarkit/purecv/issues/174) | done (PR #228; 20 issues filed: #229–#248) |
| features2d | [`docs/parity/4.10.0/features2d.md`](features2d.md) | [#175](https://github.com/webarkit/purecv/issues/175) | done (PR #249; 5 issues filed: #250–#254) |
| video | [`docs/parity/4.10.0/video.md`](video.md) | [#176](https://github.com/webarkit/purecv/issues/176) | done (PR #255; 2 issues filed: #256, #257) |
| calib3d | [`docs/parity/4.10.0/calib3d.md`](calib3d.md) | [#177](https://github.com/webarkit/purecv/issues/177) | done (PR #258; 7 issues filed: #259–#265) |

The audits run in this order: core → imgproc → features2d → video → calib3d.

## Branches

The B0–B5 PRs target the integration branch **`parity-audit`** (cut from `dev` at `902d774`,
v0.11.0), not `dev`. Each task has its own branch, cut from an up-to-date
`origin/parity-audit`: `doc/parity-method` for B0 and `doc/parity-<module>` for B1–B5. When
the series is complete, `parity-audit` is merged into `dev` in one PR. This supersedes the
plan's "cut from `origin/dev`, open PRs into `dev`" for Part B (decision recorded in
[#172](https://github.com/webarkit/purecv/issues/172)).

## Issue map

Existing open issues and PRs, by phase. Final state of the series (2026-10-21); the milestone has the live state.
Progress: `pending` / `in progress` / `done`.

| Phase | Open | Fixed on `dev` or closed | Progress |
|---|---|---|---|
| **A** reference version | | #165 (closed by PR #169; #181) | done |
| **B0** method | #172 (umbrella) | PR #188 (method), PR for the splits | done |
| **B1** core | #189 (shared `saturate_cast`), #192 (`convert_to`); #208–#226 (filed from [`core.md`](core.md#filed-issues), 2026-10-19) | #173 (closed; audit merged by PR #206); PR #207 (`KeyPoint`/`DMatch` move to `core`, merged into `dev`, synced into `parity-audit` by PR #227); #156 (pattern for D6; fixed by PR #162) | done |
| **B2** imgproc | #174; #157 (filter sites), #189 (prerequisite), #190 (tie rule), #191 (RGB→gray, both paths), #193 (needs `INTER_LINEAR_EXACT`), #225 (`hough_lines_p`), #100 (`integral`); #229–#248 (filed from [`imgproc.md`](imgproc.md#filed-issues), 2026-10-20) | | done |
| **B3** features2d | #175; #158; #159 → #193, #194, #195, #196, #197; #160 → #198, #199, #200; #161 → #201, #202, #203 (`SIMD`), #204 (`wasm`); contrib requests #101–#104; #250–#254 (filed from [`features2d.md`](features2d.md#proposed-issues), 2026-10-20) | #123, #124, #125, #126, #136, #156: fixed by PR #162, shipped in v0.10.0, still open (confirmed at `0e310c3`) | done |
| **B4** video | #176; #155, #178 (item 2), #183, #184; #154 (fixed by PR #164, confirmed at `5201b8a`, can be closed); #184 also tracks the padding-check `diverges` row; #256, #257 (filed from [`video.md`](video.md#proposed-issues), 2026-10-21) | #130, #131, #138, #142, #144, #145, #149, #163 (PR #164), #166 (PR #167): all confirmed against the 4.10.0 wheel | done |
| **B5** calib3d | #177; #127 (`decomposeHomographyMat` is `missing`, the helper would be `purecv-only`), #133 (both guess paths `diverges`; PR #134 closed unmerged), #135 (`inspected`: same minima measured; blocks the homography / planar refinements); #225 (three RANSAC rows), #237 (`undistort` is `missing`); #259–#265 (filed from [`calib3d.md`](calib3d.md#proposed-issues), 2026-10-21) | | done |
| Tooling (Part A follow-ups) | #170, #171 | | tracked separately |
| Not part of the comparison | #128 (roadmap; owns the future version bump), #132 (GPU), #97 (WASM SIMD perf), #81 and #94 (Miri), #182 (Miri shards), #67 (architecture diagram), #113 (benchmark doc) | PR #168 (license metadata) | |

**Series complete.** With B5 every module of purecv has its audit document. The next steps
are the `parity-audit` → `dev` pull request that merges the five documents, the scripts and
the pin tests in one change (not opened by an audit task), and the version-strategy question
(stay on 4.10.0 or move to 4.13 / 5.x) that goes on the agenda in
[#128](https://github.com/webarkit/purecv/issues/128). The 53 issues filed from the documents
(#208–#226, #229–#248, #250–#254, #256–#257, #259–#265) are the backlog of fixes; each lands through its own PR to `dev`, never through an
audit PR.

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

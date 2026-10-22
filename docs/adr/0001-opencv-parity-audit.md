# 0001: OpenCV parity audit with one pinned reference version and frozen per-version records

- **Status:** accepted
- **Date:** 2026-10-21
- **Deciders:** the maintainer (@kalwalt)

## Context

purecv ports OpenCV, so every claim of parity needs a version to be true against. Before
[#165](https://github.com/webarkit/purecv/issues/165) the code linked 4.10.0, 4.13.0 and
unversioned `4.x` docs, and a fix had been validated against OpenCV 5.0.0. #165 declared
OpenCV 4.10.0 the reference; the umbrella issue
[#172](https://github.com/webarkit/purecv/issues/172) and the
[implementation plan](../superpowers/plans/2026-10-11-opencv-4.10.0-reference-and-parity-audit.md)
then compared every public item of `core`, `imgproc`, `features2d`, `video` and `calib3d` with
it (five documents, 53 issues filed, 2026-10-17 to 2026-10-21). Earlier parity findings
(#158–#161) had no reproducible evidence: values from unidentified builds, which could not be
re-checked. The question left open was what happens to that record when the reference moves.

## Decision

1. **One pinned reference version at a time.** It is defined in
   `CONTRIBUTING.md#opencv-reference-version`, with its commits in `cpp_ref/libraries.json` and
   its measurement builds in `cpp_ref/requirements.txt` / `cpp_ref/package.json`. It changes
   only through its own PR, after a roadmap decision
   ([#128](https://github.com/webarkit/purecv/issues/128)).
2. **Parity is recorded per module** with the method in
   [`docs/parity/README.md`](../parity/README.md): six statuses, dimensions D1–D7, the
   evidence rule (a script in `cpp_ref/.venv` or the pinned opencv.js, or an in-tree test
   naming the tool and version), procedure S1–S7.
3. **Each series is a frozen snapshot** under `docs/parity/<version>/`. It is never edited
   after it is complete; a later reference version gets its own folder, written as a delta
   audit against the previous one.
4. **Audit PRs never change library code.** Every divergence becomes its own issue, proposed
   in the document and filed after the maintainer approves; fixes go through those issues.
5. **`tests/parity_*.rs` and `scripts/opencv_ref/` are live tooling.** They are re-measured on
   a bump (the scripts read the version from the pin); the documents are not.

## Alternatives considered

- **Edit the documents in place at each bump.** Rejected: the record of what was true against
  4.10.0 would be lost, and every bump would rewrite hundreds of rows at once with no way to
  review what changed upstream versus in purecv.
- **One living parity document** for the whole crate. Rejected: unreviewable size, and it
  mixes versions within one file.
- **Audits without reproducible evidence** (what #158–#161 originally had). Rejected: values
  that cannot be regenerated cannot be re-checked after a fix or a bump.
- **Fixing code inside audit PRs.** Rejected: it mixes measuring with changing, hides
  regressions in large diffs, and bypasses the maintainer's decision on each divergence.

## Consequences

- A bump has an explicit cost: its own PR for the pins, `CONTRIBUTING.md`, the reference
  guard constant and the re-measured live tooling, then a delta series.
- The reference guard (`tests/opencv_reference_links.rs`) accepts per-version folders: a file
  under `docs/parity/<version>/` must link that version, everything else the current
  reference.
- The scripts in `scripts/opencv_ref/` read the reference version from
  `cpp_ref/requirements.txt` instead of hard-coding it.
- Agents must not touch frozen folders ([`docs/parity/AGENTS.md`](../parity/AGENTS.md)); a
  correction to a frozen document is a note in the next series or an issue.
- The live state of the issues a series filed is its GitHub milestone, not the frozen issue
  map.

## Links

- [#165](https://github.com/webarkit/purecv/issues/165) (reference version),
  [#172](https://github.com/webarkit/purecv/issues/172) (audit umbrella),
  [#128](https://github.com/webarkit/purecv/issues/128) (roadmap; owns future bumps)
- [`CONTRIBUTING.md#opencv-reference-version`](../../CONTRIBUTING.md#opencv-reference-version)
- [`docs/parity/README.md`](../parity/README.md) (method),
  [`docs/parity/4.10.0/`](../parity/4.10.0/README.md) (the first series),
  [`docs/parity/AGENTS.md`](../parity/AGENTS.md) (agent rules),
  [`cpp_ref/AGENTS.md`](../../cpp_ref/AGENTS.md) (reading and measuring OpenCV)
- [Implementation plan](../superpowers/plans/2026-10-11-opencv-4.10.0-reference-and-parity-audit.md)

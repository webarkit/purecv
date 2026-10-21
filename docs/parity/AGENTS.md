# Agent rules for `docs/parity/`

Applies to everything under `docs/parity/`. The method (statuses, D1–D7, template, evidence
rule, S1–S7) is in [`README.md`](README.md); the rationale is
[ADR 0001](../adr/0001-opencv-parity-audit.md). Reading and measuring OpenCV:
[`cpp_ref/AGENTS.md`](../../cpp_ref/AGENTS.md).

## Rules

1. **`docs/parity/<version>/` folders are frozen.** Never edit a file in one, not even a typo
   or a broken link. A correction is a note in the next series or an issue.
2. **A new audit is a new folder**, `docs/parity/<version>/`, written as a delta of the
   previous one following S1–S7 and "Future reference versions" in `README.md`. It starts only
   after the reference version is bumped by its own PR.
3. **Audit PRs change no library code** under `src/`. They may add only their own
   `docs/parity/<version>/` folder, reference scripts in `scripts/opencv_ref/`, passing tests
   that pin rows which already match OpenCV, and `#[ignore = "diverges from OpenCV <version>,
   see #NNN"]` tests for divergences.
4. **Every `diverges` row links an issue.** Propose it first in the document and in the PR
   description; file it only after the maintainer approves, in the series milestone, then
   record its number in the document, the `#[ignore]` reason and the folder's issue map.
5. **Evidence comes only from `cpp_ref/.venv` or the pinned opencv.js**, with the build record
   (`cv2.__version__`, `useIPP`, `useOptimized`, platform, `Version control:` line). Scripts
   read the version from the pin; never hard-code it.
6. **Links are pinned to the folder's version** (the docs site and `blob/` URLs name it, as
   `4.10.0` does in `4.10.0/`); no bare `file.cpp:NN`. The reference guard
   (`cargo test --test opencv_reference_links`) scans this folder.
7. **Bookkeeping before the merge.** As the last commit of an audit PR, set the folder
   README's Documents row and issue-map row to `done` and tick the umbrella issue's box.
8. **Close sub-issues by hand** after the merge when the PR targets a non-default branch
   (`closes #N` does not fire there): `gh issue close N --comment` naming the merge commit and
   the filed issues.
9. **The live issue state is the milestone**, not a frozen issue map; do not "update" a frozen
   map when an issue closes.
10. **Merge commits only** for PRs between an integration branch and `dev`, in either
    direction; never squash or rebase them.

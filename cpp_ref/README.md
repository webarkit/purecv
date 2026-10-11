# `cpp_ref/`: OpenCV 4.10.0 reference sources and measurement tools

purecv measures parity against **OpenCV 4.10.0** (issue #165). This directory makes that
reference reproducible without a git submodule:

- `bootstrap.py` clones OpenCV and opencv_contrib at pinned commits into `cpp_ref/src/`;
- `requirements.txt` and `package.json` pin the OpenCV 4.10.0 builds used to measure
  reference numbers (Python `cv2` and opencv.js).

OpenCV is only read and measured here. It is never a build, dev or runtime dependency of
the crate (Zero-FFI), and `Cargo.toml` excludes `cpp_ref/` from the package.

Agents: the rules are in [`AGENTS.md`](AGENTS.md).

## Layout

| Path | Tracked | What |
|---|---|---|
| `bootstrap.py` | yes | the bootstrap script (vendored, patched; see [below](#bootstrappy-provenance)) |
| `test_bootstrap.py` | yes | sandbox tests for `bootstrap.py` (CI job `bootstrap-tests`) |
| `libraries.json` | yes | the pins the bootstrap reads |
| `requirements.txt` | yes | Python measurement environment |
| `package.json`, `package-lock.json` | yes | opencv.js measurement environment |
| `README.md`, `AGENTS.md`, `CLAUDE.md` | yes | this file; agent rules; Claude Code import of `AGENTS.md` |
| `src/opencv`, `src/opencv_contrib` | no | managed checkouts, written only by `bootstrap.py` |
| `.libraries.json`, `archives/` | no | bootstrap state cache and (unused) download dir |
| `.venv/`, `node_modules/` | no | the measurement environments |
| `build/` | no | out-of-tree builds of the reference (for example a self-built opencv.js) |
| `opencv/` | no | legacy, see [below](#the-legacy-cpp_refopencv) |

`.gitignore` ignores everything in `cpp_ref/` except the tracked files above.

## Pins

| Name | Tag | Tag object | Commit |
|---|---|---|---|
| opencv | `4.10.0` | `67f3511f2354ebfb5f98045ee6153345f632aadd` | `71d3237a093b60a27601c20e9ee6c3e52154e8b1` |
| opencv_contrib | `4.10.0` | `9f9a4c565dc7f3efc1daaa0e0310faa6b8a97789` | `1ed3dd2c53888e3289afdb22ec4e9ebbff3dba87` |

| Tool | Pin |
|---|---|
| Python | `opencv-contrib-python==4.10.0.84` (built from exactly the two commits above), `numpy==2.4.6` |
| opencv.js | `@techstark/opencv-js@4.10.0-release.1`, integrity `sha512-S4XELidRiQeA0q1s9VQLo540wCxUo24r1O4C+LqZ6llX+sPCXvZCPv3Ice8dEIr0uavyZ8YZeKXSBdDgMXSXjw==` |

The commits are also pinned in `tests/opencv_reference_links.rs` (`REFERENCE`,
`REFERENCE_COMMIT`, `REFERENCE_CONTRIB_COMMIT`); `cargo test` fails if the two drift apart.

## Getting the reference sources

Run from any directory; the paths below assume the repository root.

```sh
# Windows
py -3 cpp_ref/bootstrap.py --bootstrap-file cpp_ref/libraries.json
# Linux / macOS
python3 cpp_ref/bootstrap.py --bootstrap-file cpp_ref/libraries.json
```

`-n opencv` limits the run to one library. The first run clones the full history of both
repositories (several hundred MB).

The output ends with one line per library, and the exit code is 0 only when every line is OK:

```text
--- verify opencv @ 71d3237a093b60a27601c20e9ee6c3e52154e8b1 OK
--- verify opencv_contrib @ 1ed3dd2c53888e3289afdb22ec4e9ebbff3dba87 OK
```

What a run does with each checkout:

| State | Meaning | Action |
|---|---|---|
| `missing` | absent or empty | `git clone --branch 4.10.0`, then `git checkout --detach <commit>` |
| `ok` | at the pinned commit, clean | nothing |
| `drifted` | clean, at another commit | fetch if the commit is not present, then `git checkout --detach <commit>` |
| `dirty` | modified, untracked or ignored files | **refuses** and lists them |
| `corrupt` | not empty, but not a git checkout of its own | **refuses** |

"Clean" includes ignored files: a build inside `src/opencv` makes it dirty. `--clean` deletes
the checkout (and everything in it) and clones it again; use it only on purpose.

Check a checkout by hand:

```sh
git -C cpp_ref/src/opencv rev-parse HEAD           # 71d3237a093b60a27601c20e9ee6c3e52154e8b1
git -C cpp_ref/src/opencv_contrib rev-parse HEAD   # 1ed3dd2c53888e3289afdb22ec4e9ebbff3dba87
```

Line numbers in `cpp_ref/src/opencv` equal those at the tag, so cite what you read as
`https://github.com/opencv/opencv/blob/4.10.0/<path>#L<a>-L<b>` (opencv_contrib likewise).

**Never build or edit inside `cpp_ref/src/*`.** The next bootstrap refuses a dirty tree.
Out-of-tree builds go to `cpp_ref/build/`, for example
`cmake -S cpp_ref/src/opencv -B cpp_ref/build/opencv`.

## Measurement environments

### Python (`cv2`)

Needs Python 3.11 or newer (numpy 2.4.6 requires it).

```sh
# Windows
py -3 -m venv cpp_ref/.venv
cpp_ref/.venv/Scripts/python -m pip install -r cpp_ref/requirements.txt
# Linux / macOS
python3 -m venv cpp_ref/.venv
cpp_ref/.venv/bin/python -m pip install -r cpp_ref/requirements.txt
```

If pip fails with `CERTIFICATE_VERIFY_FAILED` behind a TLS-inspecting proxy, add
`--use-feature=truststore`: pip then verifies against the operating system's trust store
(newer pip releases do this by default; the pip bundled with Python 3.11 does not).

Never also install `opencv-python`: every OpenCV wheel provides the same `cv2` module.
Measurement scripts live in `scripts/opencv_ref/` and start with:

```python
import cv2

assert cv2.__version__ == "4.10.0"
cv2.ipp.setUseIPP(False)  # measure the generic CPU path
```

and write the `Version control:` line of `cv2.getBuildInformation()` into their output.

### opencv.js

```sh
npm ci --prefix cpp_ref
```

installs the exact tarball recorded in `package-lock.json`. Print its build version:

```sh
node -e "const cv = require('./cpp_ref/node_modules/@techstark/opencv-js'); cv.onRuntimeInitialized = () => { console.log(cv.getBuildInformation().match(/Version control:.*/)[0]); process.exit(0); };"
```

This prints `Version control:               4.10.0`.

This package is a legacy Emscripten build: the exported module has a `then` method that
resolves with the module itself. Never `await` the module or return it from an `async`
function; either never settles. Wait for `cv.onRuntimeInitialized` instead, or `delete cv.then`
before passing the module on.

The integrity hash pins the package's bytes, but nobody has yet checked that its
`opencv.js` matches a build of `cpp_ref/src/opencv` byte for byte. Where that matters,
build opencv.js from `cpp_ref/src/opencv` into `cpp_ref/build/` and record its
`Version control:` line.

## Bumping the reference

All in one PR on the roadmap (#128):

1. Look up the peeled commits of the new tag:

   ```sh
   git ls-remote https://github.com/opencv/opencv.git "refs/tags/<ver>^{}"
   git ls-remote https://github.com/opencv/opencv_contrib.git "refs/tags/<ver>^{}"
   ```

2. Update `branch` and `commit` in `cpp_ref/libraries.json`.
3. Update `REFERENCE`, `REFERENCE_COMMIT` and `REFERENCE_CONTRIB_COMMIT` in
   `tests/opencv_reference_links.rs`.
4. Update `cpp_ref/requirements.txt` to the `opencv-contrib-python` release built from those
   commits, and re-pin `numpy` to the version pip resolves for it.
5. Update `cpp_ref/package.json` and run `npm install --prefix cpp_ref` to refresh
   `cpp_ref/package-lock.json`.
6. Update `CONTRIBUTING.md`.
7. Run the bootstrap: the checkouts are `drifted`, so it fetches and checks out the new
   commits, and must end with two `OK` lines.

## `bootstrap.py` provenance

`bootstrap.py` is WebARKitLib-rs's `benchmarks/bootstrap.py` at commit
`a099326efc2edcf5b1ebcdb69e6d3d9adfe4bbfc` (blob `b22bd1d8d1fdb1095a37f3b9ea9c374e68f6b03a`),
itself corporateshark/bootstrapping 1.0.7 (MIT) plus a `commit` alias and a `branch` key.
It keeps that CLI and the `libraries.json` schema. The first commit that adds it here is
byte-identical to that blob; the purecv patches follow, each hunk tagged `# purecv: Pn`:

| # | Change |
|---|---|
| P1 | Without `-b`, the base directory is the bootstrap file's directory, so checkouts land in `cpp_ref/src/` from any cwd. `--base-dir` takes its value. A base directory containing `Cargo.toml` is refused. |
| P2 | TLS certificate verification is no longer disabled. |
| P3 | `_rmtree` deletes read-only files too (Git for Windows object files), so `--clean` works on Windows. |
| P4 | git runs without a shell, with `GIT_DIR` and the other repository-local variables removed and discovery stopped at `cpp_ref/src`. A checkout is classified (table above) instead of being reset and cleaned; the state cache is never trusted for git entries. |
| P5 | Every entry is validated before any action: safe `name`, a 40-hex `commit` for git entries, a safe `branch`; `predicate` and `postprocess` are rejected. |
| P6 | Bad options and unknown `-n` names exit 2; Ctrl-C aborts the run; a final verify pass prints the `verify` lines and sets the exit code. Version `1.0.7 (2025) purecv-1`. |

P1 to P4 are generic and can be offered upstream.

Tests: `python -m unittest discover -s cpp_ref -p "test_*.py" -v` (`py -3 -m unittest …` on
Windows). They build throwaway repositories in a temporary directory and never touch
`cpp_ref/` itself.

## The legacy `cpp_ref/opencv`

Some checkouts have a hand-made `cpp_ref/opencv` (a 4.x clone, possibly with a large build in
it). It is ignored by git, the bootstrap never touches it, and it is **not** the reference:
take neither code nor line numbers from it. Use `cpp_ref/src/opencv`.

## Claude Code and `cpp_ref/CLAUDE.md`

`CLAUDE.md` imports `AGENTS.md`, so Claude Code reads the same rules as other agents when it
loads `CLAUDE.md` for this directory. Whether a read deep inside the ignored `cpp_ref/src/`
tree triggers that load is **not yet verified**. Do not rely on it: the root agent files tell
every agent to read `cpp_ref/AGENTS.md` first, and that read loads it.

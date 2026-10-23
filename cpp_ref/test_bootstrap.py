#
#  test_bootstrap.py
#  purecv
#
#  This file is part of purecv - WebARKit.
#
#  purecv is free software: you can redistribute it and/or modify
#  it under the terms of the GNU Lesser General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  purecv is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU Lesser General Public License for more details.
#
#  You should have received a copy of the GNU Lesser General Public License
#  along with purecv.  If not, see <http://www.gnu.org/licenses/>.
#
#  As a special exception, the copyright holders of this library give you
#  permission to link this library with independent modules to produce an
#  executable, regardless of the license terms of these independent modules, and to
#  copy and distribute the resulting executable under terms of your choice,
#  provided that you also meet, for each linked independent module, the terms and
#  conditions of the license of that module. An independent module is a module
#  which is neither derived from nor based on this library. If you modify this
#  library, you may extend this exception to your version of the library, but you
#  are not obligated to do so. If you do not wish to do so, delete this exception
#  statement from your version.
#
#  Copyright 2026 WebARKit.
#
#  Author(s): Walter Perdan @kalwalt https://github.com/kalwalt
#

"""Sandbox tests for cpp_ref/bootstrap.py (purecv#165).

Each test builds a throwaway fake purecv checkout and a fake upstream repository in a
temporary directory and runs a copy of the bootstrap there. Nothing touches the real
cpp_ref/ tree or the developer's git configuration.

    python -m unittest discover -s cpp_ref -p "test_*.py" -v
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import os
import shutil
import ssl
import stat
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
BOOTSTRAP = os.path.join(HERE, "bootstrap.py")
GIT = shutil.which("git")

# The documented invocations: from the repository root, and from inside cpp_ref/.
FROM_ROOT = ("--bootstrap-file", "cpp_ref/libraries.json")
FROM_CPP_REF = ("--bootstrap-file", "libraries.json")


def _force_rmtree(path: str) -> None:
    """Removes a sandbox, including the read-only object files Git for Windows writes."""

    def make_writable_and_retry(func, failed_path, _exc):
        os.chmod(failed_path, os.lstat(failed_path).st_mode | stat.S_IRWXU)
        func(failed_path)

    if sys.version_info >= (3, 12):
        shutil.rmtree(path, onexc=make_writable_and_retry)
    else:
        shutil.rmtree(path, onerror=make_writable_and_retry)


def _sandbox_env(ceiling: str, extra: dict[str, str] | None = None) -> dict[str, str]:
    """The environment of every git and bootstrap process a test starts.

    Inherited GIT_* variables (a hook's GIT_DIR or GIT_INDEX_FILE, say) are dropped and the
    developer's global and system git config is hidden, so neither can redirect a command
    to another repository or change its result (gpgsign, showUntrackedFiles, autocrlf,
    defaultBranch). Repository discovery never walks above `ceiling`.
    """
    env = {k: v for k, v in os.environ.items() if not k.upper().startswith("GIT_")}
    env.update(
        GIT_AUTHOR_NAME="purecv test",
        GIT_AUTHOR_EMAIL="test@example.invalid",
        GIT_COMMITTER_NAME="purecv test",
        GIT_COMMITTER_EMAIL="test@example.invalid",
        GIT_CONFIG_GLOBAL=os.devnull,
        GIT_CONFIG_NOSYSTEM="1",
        GIT_CONFIG_COUNT="1",
        GIT_CONFIG_KEY_0="core.autocrlf",
        GIT_CONFIG_VALUE_0="false",
        GIT_CEILING_DIRECTORIES=ceiling,
        PYTHONIOENCODING="utf-8",
    )
    if extra:
        env.update(extra)
    return env


def _output(result: subprocess.CompletedProcess) -> str:
    return f"rc={result.returncode}\n--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"


@unittest.skipIf(GIT is None, "git is not installed")
class BootstrapSandboxTest(unittest.TestCase):
    def setUp(self) -> None:
        self.make_sandbox()

    # ---- fixture ---------------------------------------------------------------------

    def make_sandbox(self) -> None:
        """Creates a fresh sandbox and points self.root and friends at it.

        self.root is a fake purecv checkout (Cargo.toml, src/lib.rs, cpp_ref/bootstrap.py,
        one commit). self.upstream holds C1, tagged 4.10.0 like OpenCV's annotated release
        tags, and a newer C2. cpp_ref/libraries.json pins C1.
        """
        self.root = os.path.realpath(tempfile.mkdtemp(prefix="purecv-bootstrap-"))
        self.addCleanup(_force_rmtree, self.root)
        self.cpp_ref = os.path.join(self.root, "cpp_ref")
        self.managed = os.path.join(self.cpp_ref, "src", "opencv")

        self.upstream = os.path.join(self.root, "upstream")
        os.mkdir(self.upstream)
        self.git(self.upstream, "init", "-q")
        self.c1 = self.commit(self.upstream, "README.txt", "C1\n")
        self.git(self.upstream, "tag", "-a", "4.10.0", "-m", "4.10.0")
        self.c2 = self.commit(self.upstream, "README.txt", "C2\n")

        self.write(os.path.join(self.root, "Cargo.toml"), '[package]\nname = "fake-purecv"\n')
        self.write(os.path.join(self.root, "src", "lib.rs"), "")
        os.makedirs(self.cpp_ref)
        shutil.copyfile(BOOTSTRAP, os.path.join(self.cpp_ref, "bootstrap.py"))
        self.git(self.root, "init", "-q")
        self.git(self.root, "add", "Cargo.toml", "src/lib.rs", "cpp_ref/bootstrap.py")
        self.git(self.root, "commit", "-q", "-m", "fake purecv")

        self.write_libraries([self.entry()])

    def git(self, cwd: str, *args: str) -> str:
        """Runs a fixture git command in the repository at `cwd`; returns its stdout."""
        result = subprocess.run(
            [GIT, *args],
            cwd=cwd,
            env=_sandbox_env(os.path.dirname(cwd)),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            encoding="utf-8",
            errors="replace",
            timeout=120,
        )
        if result.returncode != 0:
            raise RuntimeError(f"git {' '.join(args)} failed in {cwd}:\n{result.stderr}")
        return result.stdout.strip()

    def commit(self, repo: str, name: str, content: str) -> str:
        self.write(os.path.join(repo, name), content)
        self.git(repo, "add", name)
        self.git(repo, "commit", "-q", "-m", content.strip())
        return self.git(repo, "rev-parse", "HEAD")

    def head(self, repo: str) -> str:
        self.assertTrue(os.path.isdir(os.path.join(repo, ".git")), f"{repo} is not a git checkout")
        return self.git(repo, "rev-parse", "HEAD")

    @staticmethod
    def write(path: str, content: str | bytes) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.write(content.encode() if isinstance(content, str) else content)

    @staticmethod
    def read(path: str) -> bytes:
        with open(path, "rb") as f:
            return f.read()

    def entry(self, name: str = "opencv", **source: object) -> dict:
        pinned = {
            "type": "git",
            "url": self.upstream,
            "branch": "4.10.0",
            "commit": self.c1,
            "recursive": False,
        }
        pinned.update(source)
        return {"name": name, "source": pinned}

    def write_libraries(self, entries: list[dict]) -> None:
        self.write(os.path.join(self.cpp_ref, "libraries.json"), json.dumps(entries, indent=2))

    def run_bootstrap(
        self, *args: str, cwd: str, extra_env: dict[str, str] | None = None
    ) -> subprocess.CompletedProcess:
        """Runs the sandbox's cpp_ref/bootstrap.py from `cwd`, as a developer would."""
        script = os.path.relpath(os.path.join(self.cpp_ref, "bootstrap.py"), cwd)
        return subprocess.run(
            [sys.executable, script, *args],
            cwd=cwd,
            env=_sandbox_env(os.path.dirname(self.root), extra_env),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            encoding="utf-8",
            errors="replace",
            timeout=300,
        )

    def assert_ok(self, result: subprocess.CompletedProcess) -> None:
        self.assertEqual(result.returncode, 0, _output(result))

    def assert_refused(self, result: subprocess.CompletedProcess) -> None:
        self.assertNotEqual(result.returncode, 0, _output(result))

    def clones(self) -> list[str]:
        """Every git repository in the sandbox except the fake purecv and the upstream."""
        found = []
        for dirpath, dirnames, filenames in os.walk(self.root):
            if ".git" in dirnames or ".git" in filenames:
                if dirpath not in (self.root, self.upstream):
                    found.append(os.path.relpath(dirpath, self.root))
            if ".git" in dirnames:
                dirnames.remove(".git")
        return found

    def find(self, filename: str) -> list[str]:
        return [
            os.path.relpath(os.path.join(dirpath, filename), self.root)
            for dirpath, _dirnames, filenames in os.walk(self.root)
            if filename in filenames
        ]

    def make_legacy_clone(self) -> str:
        """A clean clone of the upstream at C2 in cpp_ref/opencv, like the maintainer's own."""
        legacy = os.path.join(self.cpp_ref, "opencv")
        self.git(self.cpp_ref, "clone", "-q", self.upstream, legacy)
        self.assertEqual(self.head(legacy), self.c2)
        return legacy

    @staticmethod
    def make_dir_link(target: str, link: str) -> None:
        """A directory junction on Windows (no privileges needed), a symlink elsewhere."""
        os.makedirs(os.path.dirname(link), exist_ok=True)
        if os.name == "nt":
            subprocess.run(
                ["cmd", "/c", "mklink", "/J", link, target],
                check=True,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        else:
            os.symlink(target, link, target_is_directory=True)

    def repo_snapshot(self, repo: str) -> dict:
        """Everything a refused run must leave alone in `repo`."""
        files = {}
        for dirpath, dirnames, filenames in os.walk(repo):
            if ".git" in dirnames:
                dirnames.remove(".git")
            for filename in filenames:
                path = os.path.join(dirpath, filename)
                files[os.path.relpath(path, repo)] = self.read(path)
        return {
            "HEAD": self.head(repo),
            "refs": self.git(repo, "for-each-ref"),
            "worktrees": self.git(repo, "worktree", "list", "--porcelain"),
            "status": self.git(repo, "status", "--porcelain", "--ignored"),
            "mode": os.stat(repo).st_mode,
            "files": files,
        }

    # ---- tests -----------------------------------------------------------------------

    def test_run_from_repo_root_never_writes_into_rust_src(self) -> None:
        result = self.run_bootstrap(*FROM_ROOT, cwd=self.root)

        self.assert_ok(result)
        self.assertEqual(self.head(self.managed), self.c1)
        self.assertEqual(os.listdir(os.path.join(self.root, "src")), ["lib.rs"])
        self.assertFalse(os.path.exists(os.path.join(self.root, "archives")))
        self.assertIn(f"verify opencv @ {self.c1} OK", result.stdout)

    def test_refuses_base_dir_containing_cargo_toml(self) -> None:
        for option in ("-b", "--base-dir"):
            with self.subTest(option=option):
                result = self.run_bootstrap(option, self.root, *FROM_ROOT, cwd=self.root)

                self.assert_refused(result)
                self.assertEqual(os.listdir(os.path.join(self.root, "src")), ["lib.rs"])

    def test_bootstrap_file_and_base_dir_in_either_order(self) -> None:
        # -b used to reset the bootstrap file to <base dir>/bootstrap.json, even after an
        # explicit --bootstrap-file, so one order of the same two options cloned nothing.
        for bootstrap_file_first in (True, False):
            with self.subTest(bootstrap_file_first=bootstrap_file_first):
                self.make_sandbox()
                bootstrap_file = ("--bootstrap-file", os.path.join(self.cpp_ref, "libraries.json"))
                base_dir = ("-b", self.cpp_ref)
                args = bootstrap_file + base_dir if bootstrap_file_first else base_dir + bootstrap_file

                result = self.run_bootstrap(*args, cwd=self.root)

                self.assert_ok(result)
                self.assertEqual(self.head(self.managed), self.c1)
                self.assertIn(f"verify opencv @ {self.c1} OK", result.stdout, _output(result))
                self.assertNotIn("bootstrap.json", result.stdout, _output(result))
                self.assertEqual(os.listdir(os.path.join(self.root, "src")), ["lib.rs"])

    def test_existing_unmanaged_checkout_is_untouched(self) -> None:
        # The maintainer's hand-made checkout, with a build and IDE settings in it.
        legacy = os.path.join(self.cpp_ref, "opencv")
        os.makedirs(legacy)
        self.git(legacy, "init", "-q")
        self.commit(legacy, "CMakeLists.txt", "project(OpenCV)\n")
        head = self.head(legacy)
        files = {
            os.path.join("out", "big.bin"): os.urandom(1 << 16),
            "CMakeSettings.json": b'{ "configurations": [] }\n',
        }
        for rel, data in files.items():
            self.write(os.path.join(legacy, rel), data)
        before = {rel: os.stat(os.path.join(legacy, rel)) for rel in files}

        result = self.run_bootstrap(*FROM_CPP_REF, cwd=self.cpp_ref)

        self.assert_ok(result)
        for rel, data in files.items():
            path = os.path.join(legacy, rel)
            self.assertEqual(self.read(path), data, rel)
            after = os.stat(path)
            self.assertEqual(
                (after.st_mode, after.st_mtime_ns), (before[rel].st_mode, before[rel].st_mtime_ns), rel
            )
        self.assertEqual(self.head(legacy), head)

    def test_drifted_checkout_is_resynced(self) -> None:
        self.assert_ok(self.run_bootstrap(*FROM_CPP_REF, cwd=self.cpp_ref))
        self.git(self.managed, "checkout", "-q", self.c2)

        result = self.run_bootstrap(*FROM_CPP_REF, cwd=self.cpp_ref)

        self.assert_ok(result)
        self.assertEqual(self.head(self.managed), self.c1)

        # A reference bump to a commit the checkout has never fetched.
        c3 = self.commit(self.upstream, "README.txt", "C3\n")
        self.git(self.upstream, "tag", "-a", "4.11.0", "-m", "4.11.0")
        self.write_libraries([self.entry(branch="4.11.0", commit=c3)])

        result = self.run_bootstrap(*FROM_CPP_REF, cwd=self.cpp_ref)

        self.assert_ok(result)
        self.assertEqual(self.head(self.managed), c3)

    # The lines #170 removed from a successful run; none of them reports a problem.
    NOISE = (
        "paramiko",  # module-level scp warning
        "is not a commit",  # `clone --branch <annotated tag>`
        "detached HEAD",  # git's advice block on `checkout --detach`
        "archives",  # an archives/ directory created on every run
        "not found, but required",  # hg/svn/patch looked up on Linux
    )

    def assert_no_noise(self, result: subprocess.CompletedProcess) -> None:
        for noise in self.NOISE:
            with self.subTest(noise=noise):
                self.assertNotIn(noise, result.stdout + result.stderr, _output(result))

    def test_successful_runs_print_no_noise(self) -> None:
        # A fresh clone: "Cloning" once, then the verify line, and none of the noise.
        result = self.run_bootstrap(*FROM_CPP_REF, cwd=self.cpp_ref)

        self.assert_ok(result)
        self.assert_no_noise(result)
        self.assertEqual(result.stdout.count("Cloning "), 1, _output(result))
        self.assertTrue(result.stdout.rstrip().endswith(f"verify opencv @ {self.c1} OK"), _output(result))
        self.assertFalse(os.path.exists(os.path.join(self.cpp_ref, "archives")))

        # A drifted checkout is moved, not cloned: no "Cloning" line (#170, item 2).
        self.git(self.managed, "checkout", "-q", self.c2)
        result = self.run_bootstrap(*FROM_CPP_REF, cwd=self.cpp_ref)

        self.assert_ok(result)
        self.assert_no_noise(result)
        self.assertNotIn("Cloning", result.stdout, _output(result))
        self.assertIn("opencv is drifted; moving it to " + self.c1, result.stdout, _output(result))
        self.assertEqual(self.head(self.managed), self.c1)

    def test_help_and_version_win_over_rejected_options(self) -> None:
        # -h and --version answer even next to an option that is otherwise rejected (#171, 5).
        for answer, expected in (("-h", "Options:"), ("--version", "purecv-2")):
            with self.subTest(answer=answer):
                result = self.run_bootstrap("--use-tar", answer, cwd=self.cpp_ref)

                self.assert_ok(result)
                self.assertIn(expected, result.stdout, _output(result))
                self.assertFalse(os.path.exists(os.path.join(self.cpp_ref, "src")))

    def test_unreadable_state_cache_is_replaced(self) -> None:
        # Upstream crashed with a TypeError on a cache that is not a JSON array (#171, 5).
        state_cache = os.path.join(self.cpp_ref, ".libraries.json")
        for content in (b"{not json", b'{"name": "opencv"}\n', b"[1, 2]\n"):
            with self.subTest(content=content):
                self.write(state_cache, content)

                result = self.run_bootstrap(*FROM_CPP_REF, cwd=self.cpp_ref)

                self.assert_ok(result)
                self.assertIn(f"verify opencv @ {self.c1} OK", result.stdout, _output(result))
                self.assertIn(self.entry(), json.loads(self.read(state_cache)))

    def test_run_that_writes_no_cache_does_not_crash(self) -> None:
        # Upstream touched the state cache at the end even when no library wrote it (#171, 5).
        result = self.run_bootstrap("--skip", "opencv", *FROM_CPP_REF, cwd=self.cpp_ref)

        self.assert_ok(result)
        self.assertNotIn("Traceback", result.stdout + result.stderr, _output(result))
        self.assertFalse(os.path.exists(os.path.join(self.cpp_ref, ".libraries.json")))

    def test_dirty_checkout_is_refused(self) -> None:
        self.assert_ok(self.run_bootstrap(*FROM_CPP_REF, cwd=self.cpp_ref))
        build = os.path.join(self.managed, "out", "build.bin")
        self.write(build, b"\0" * 64)

        result = self.run_bootstrap(*FROM_CPP_REF, cwd=self.cpp_ref)

        self.assert_refused(result)
        self.assertTrue(os.path.isfile(build))
        self.assertIn("out/", result.stdout + result.stderr)
        # A bare --clean would also delete and re-clone every other library.
        self.assertIn("--clean -n opencv", result.stdout, _output(result))

    def assert_hidden_edits_refused(self, result: subprocess.CompletedProcess, flag: str, before: dict) -> None:
        """A rerun over a checkout whose edits `git status` cannot see: refused, nothing touched."""
        self.assertEqual(self.repo_snapshot(self.managed), before, _output(result))
        self.assert_refused(result)
        self.assertIn(flag, result.stdout, _output(result))
        self.assertIn("--clean -n opencv", result.stdout, _output(result))
        verify = [line for line in result.stdout.splitlines() if line.startswith("--- verify opencv")]
        self.assertEqual(len(verify), 1, _output(result))
        self.assertIn("MISMATCH: dirty", verify[0], _output(result))
        self.assertIn(flag, verify[0], _output(result))

    def test_assume_unchanged_edit_is_refused(self) -> None:
        # assume-unchanged tells git not to look at the file, so `git status` hides the edit.
        self.assert_ok(self.run_bootstrap(*FROM_CPP_REF, cwd=self.cpp_ref))
        readme = os.path.join(self.managed, "README.txt")
        self.git(self.managed, "update-index", "--assume-unchanged", "README.txt")
        self.write(readme, b"edited reference code\n")
        self.assertEqual(self.git(self.managed, "status", "--porcelain", "--ignored"), "")
        before = self.repo_snapshot(self.managed)

        result = self.run_bootstrap(*FROM_CPP_REF, cwd=self.cpp_ref)

        self.assert_hidden_edits_refused(result, "assume-unchanged", before)
        self.assertEqual(self.read(readme), b"edited reference code\n")
        self.assertEqual(self.git(self.managed, "ls-files", "-v", "README.txt"), "h README.txt")

    def test_skip_worktree_edit_is_refused(self) -> None:
        # skip-worktree makes git treat the file as matching the index, so `git status` hides it.
        self.assert_ok(self.run_bootstrap(*FROM_CPP_REF, cwd=self.cpp_ref))
        readme = os.path.join(self.managed, "README.txt")
        self.git(self.managed, "update-index", "--skip-worktree", "README.txt")
        self.write(readme, b"edited reference code\n")
        self.assertEqual(self.git(self.managed, "status", "--porcelain", "--ignored"), "")
        before = self.repo_snapshot(self.managed)

        result = self.run_bootstrap(*FROM_CPP_REF, cwd=self.cpp_ref)

        self.assert_hidden_edits_refused(result, "skip-worktree", before)
        self.assertEqual(self.read(readme), b"edited reference code\n")
        self.assertEqual(self.git(self.managed, "ls-files", "-v", "README.txt"), "S README.txt")

    def test_sparse_checkout_is_refused(self) -> None:
        # A sparse checkout leaves files out of the worktree and out of `git status`. The
        # checkout's own core.sparseCheckout turns it on.
        self.assert_ok(self.run_bootstrap(*FROM_CPP_REF, cwd=self.cpp_ref))
        self.git(self.managed, "config", "core.sparseCheckout", "true")
        self.assertEqual(self.git(self.managed, "status", "--porcelain", "--ignored"), "")
        before = self.repo_snapshot(self.managed)

        result = self.run_bootstrap(*FROM_CPP_REF, cwd=self.cpp_ref)

        self.assert_hidden_edits_refused(result, "sparse checkout", before)
        self.assertIn("core.sparseCheckout", result.stdout, _output(result))
        self.assertEqual(self.read(os.path.join(self.managed, "README.txt")), b"C1\n")
        self.assertEqual(self.git(self.managed, "config", "core.sparseCheckout"), "true")

    def test_leftover_sparse_checkout_file_is_accepted(self) -> None:
        # `git sparse-checkout disable` sets core.sparseCheckout to false and leaves
        # .git/info/sparse-checkout behind. The full tree is there and clean (#171, item 2).
        self.assert_ok(self.run_bootstrap(*FROM_CPP_REF, cwd=self.cpp_ref))
        sparse_file = os.path.join(self.managed, ".git", "info", "sparse-checkout")
        self.write(sparse_file, "/*\n")
        self.git(self.managed, "config", "core.sparseCheckout", "false")

        result = self.run_bootstrap(*FROM_CPP_REF, cwd=self.cpp_ref)

        self.assert_ok(result)
        self.assertIn(f"verify opencv @ {self.c1} OK", result.stdout, _output(result))
        self.assertEqual(self.read(sparse_file), b"/*\n")

    def test_user_config_does_not_make_fresh_clones_dirty(self) -> None:
        # core.sparseCheckout or core.ignoreStat in the user's global config made every fresh
        # clone sparse or assume-unchanged, so every run was refused and --clean looped
        # (#171, item 3). The bootstrap's own git commands override both.
        for setting in ("core.sparseCheckout", "core.ignoreStat"):
            with self.subTest(setting=setting):
                self.make_sandbox()
                global_config = os.path.join(self.root, "user.gitconfig")
                self.write(global_config, f"[core]\n\t{setting.split('.')[1]} = true\n")

                result = self.run_bootstrap(*FROM_CPP_REF, cwd=self.cpp_ref,
                                            extra_env={"GIT_CONFIG_GLOBAL": global_config})

                self.assert_ok(result)
                self.assertIn(f"verify opencv @ {self.c1} OK", result.stdout, _output(result))
                self.assertEqual(self.read(os.path.join(self.managed, "README.txt")), b"C1\n")
                self.assertEqual(self.git(self.managed, "ls-files", "-v", "README.txt"), "H README.txt")

    def test_replace_ref_cannot_pass_an_edit_off_as_the_pin(self) -> None:
        # `git replace <pin> <edited commit>` keeps HEAD's SHA equal to the pin while the tree
        # is the edited one, so `git status` is clean (#171, item 4).
        self.assert_ok(self.run_bootstrap(*FROM_CPP_REF, cwd=self.cpp_ref))
        readme = os.path.join(self.managed, "README.txt")
        self.write(readme, "edited reference code\n")
        self.git(self.managed, "commit", "-q", "-am", "edited")
        edited = self.git(self.managed, "rev-parse", "HEAD")
        self.git(self.managed, "replace", self.c1, edited)
        self.git(self.managed, "checkout", "-q", "--detach", self.c1)
        self.assertEqual(self.head(self.managed), self.c1)
        self.assertEqual(self.git(self.managed, "status", "--porcelain", "--ignored"), "")

        result = self.run_bootstrap(*FROM_CPP_REF, cwd=self.cpp_ref)

        self.assert_refused(result)
        self.assertEqual(self.read(readme), b"edited reference code\n")
        verify = [line for line in result.stdout.splitlines() if line.startswith("--- verify opencv")]
        self.assertEqual(len(verify), 1, _output(result))
        self.assertIn("MISMATCH: dirty", verify[0], _output(result))

    def test_lying_fsmonitor_cannot_hide_an_edit(self) -> None:
        # An fsmonitor hook that always answers "nothing changed" hides an edit from
        # `git status` (#171, item 4).
        self.assert_ok(self.run_bootstrap(*FROM_CPP_REF, cwd=self.cpp_ref))
        hook = os.path.join(self.root, "liar.sh")
        self.write(hook, "#!/bin/sh\nprintf 'token\\000'\n")
        os.chmod(hook, 0o755)
        self.git(self.managed, "config", "core.fsmonitor", hook.replace(os.sep, "/"))
        self.git(self.managed, "update-index", "--fsmonitor")
        self.git(self.managed, "status", "--porcelain")
        readme = os.path.join(self.managed, "README.txt")
        self.write(readme, "edited reference code\n")
        if self.git(self.managed, "status", "--porcelain", "--ignored") != "":
            self.skipTest("this git does not trust the fsmonitor hook here")

        result = self.run_bootstrap(*FROM_CPP_REF, cwd=self.cpp_ref)

        self.assert_refused(result)
        self.assertEqual(self.read(readme), b"edited reference code\n")
        self.assertIn("README.txt", result.stdout, _output(result))

    def test_configured_filter_is_refused(self) -> None:
        # A clean filter makes `git status` compare filtered content: with this one, any edit
        # to README.txt looks clean (#171, item 4). An attribute without a driver is harmless.
        self.assert_ok(self.run_bootstrap(*FROM_CPP_REF, cwd=self.cpp_ref))
        self.write(os.path.join(self.managed, ".git", "info", "attributes"), "README.txt filter=hide\n")
        self.assert_ok(self.run_bootstrap(*FROM_CPP_REF, cwd=self.cpp_ref))

        self.git(self.managed, "config", "filter.hide.clean", "cat")
        before = self.repo_snapshot(self.managed)

        result = self.run_bootstrap(*FROM_CPP_REF, cwd=self.cpp_ref)

        self.assert_hidden_edits_refused(result, "filter", before)
        self.assertIn("README.txt (filter hide)", result.stdout, _output(result))

    def test_corrupt_nested_git_does_not_touch_outer_repo(self) -> None:
        for git_dir in (False, True):
            with self.subTest(GIT_DIR=git_dir):
                self.make_sandbox()
                # Like the real purecv, the outer repository has a fetchable origin.
                self.git(self.root, "remote", "add", "origin", self.upstream)
                os.makedirs(os.path.join(self.managed, ".git", "objects"))
                lib_rs = os.path.join(self.root, "src", "lib.rs")
                self.write(lib_rs, "// uncommitted edit\n")
                head = self.head(self.root)
                extra_env = {"GIT_DIR": os.path.join(self.root, ".git")} if git_dir else None

                result = self.run_bootstrap(*FROM_CPP_REF, cwd=self.cpp_ref, extra_env=extra_env)

                self.assert_refused(result)
                self.assertEqual(self.head(self.root), head)
                self.assertEqual(self.read(lib_rs), b"// uncommitted edit\n")
                self.assertTrue(os.path.isdir(os.path.join(self.managed, ".git", "objects")))
                self.assertIn("verify opencv MISMATCH: corrupt", result.stdout, _output(result))

    def test_linked_checkout_is_refused_even_with_clean(self) -> None:
        # cpp_ref/src/opencv is a junction (Windows) or symlink to a foreign clean clone at C2.
        legacy = self.make_legacy_clone()
        self.make_dir_link(legacy, self.managed)
        before = self.repo_snapshot(legacy)

        for extra in ((), ("--clean",)):
            with self.subTest(args=extra):
                result = self.run_bootstrap(*FROM_CPP_REF, *extra, cwd=self.cpp_ref)

                self.assert_refused(result)
                self.assertIn("verify opencv MISMATCH: corrupt", result.stdout, _output(result))
                if extra:
                    self.assertIn("symlink or junction", result.stdout, _output(result))
                self.assertEqual(self.repo_snapshot(legacy), before)
                self.assertTrue(os.path.samefile(self.managed, legacy), "the link itself must stay")

    def test_linked_src_dir_is_refused_even_with_clean(self) -> None:
        # cpp_ref/src itself is a junction (Windows) or symlink to a foreign directory that
        # holds an `opencv` clone at C2. Two clones: a clean one, which a plain run would move
        # to C1, and one with an untracked file, which --clean would delete. Neither mode may
        # write anything, through the link or next to it.
        for untracked in (False, True):
            for extra in ((), ("--clean",)):
                with self.subTest(untracked=untracked, args=extra):
                    self.make_sandbox()
                    foreign = os.path.join(self.root, "foreign")
                    os.mkdir(foreign)
                    clone = os.path.join(foreign, "opencv")
                    self.git(foreign, "clone", "-q", self.upstream, clone)
                    if untracked:
                        self.write(os.path.join(clone, "precious.txt"), b"untracked and irreplaceable\n")
                    src = os.path.join(self.cpp_ref, "src")
                    self.make_dir_link(foreign, src)
                    before = self.repo_snapshot(clone)
                    self.assertEqual(self.head(clone), self.c2)
                    self.assertEqual("precious.txt" in before["files"], untracked)

                    result = self.run_bootstrap(*FROM_CPP_REF, *extra, cwd=self.cpp_ref)

                    self.assertEqual(self.repo_snapshot(clone), before, _output(result))
                    self.assertEqual(os.listdir(foreign), ["opencv"])
                    self.assertTrue(os.path.samefile(src, foreign), "the link itself must stay")
                    # Refused before any write: no archives/ dir, no state cache.
                    self.assertEqual(
                        sorted(os.listdir(self.cpp_ref)), ["bootstrap.py", "libraries.json", "src"]
                    )
                    self.assertEqual(result.returncode, 2, _output(result))
                    self.assertIn(
                        f"ERROR: {src} is a symlink or junction; refusing to write through it",
                        result.stdout,
                        _output(result),
                    )

    def test_gitfile_checkout_is_refused(self) -> None:
        # cpp_ref/src/opencv is a `git worktree` of another repository: its .git is a file.
        legacy = self.make_legacy_clone()
        os.makedirs(os.path.dirname(self.managed))
        self.git(legacy, "worktree", "add", "-q", "--detach", self.managed, self.c2)
        self.assertTrue(os.path.isfile(os.path.join(self.managed, ".git")))
        before = self.repo_snapshot(legacy)

        result = self.run_bootstrap(*FROM_CPP_REF, cwd=self.cpp_ref)

        self.assert_refused(result)
        self.assertIn("verify opencv MISMATCH: corrupt", result.stdout, _output(result))
        self.assertEqual(self.repo_snapshot(legacy), before)

    def test_clean_recovers_with_readonly_pack_files(self) -> None:
        self.assert_ok(self.run_bootstrap(*FROM_CPP_REF, cwd=self.cpp_ref))
        readonly = 0
        for dirpath, _dirnames, filenames in os.walk(os.path.join(self.managed, ".git", "objects")):
            for filename in filenames:
                os.chmod(os.path.join(dirpath, filename), stat.S_IREAD)
                readonly += 1
        self.assertGreater(readonly, 0, "no object files to make read-only")

        result = self.run_bootstrap(*FROM_CPP_REF, "--clean", cwd=self.cpp_ref)

        self.assert_ok(result)
        self.assertEqual(self.head(self.managed), self.c1)

    def test_rejects_malformed_entries(self) -> None:
        cases = {
            "no commit": lambda e: e["source"].pop("commit"),
            "tag as commit": lambda e: e["source"].update(commit="4.10.0"),
            "command in commit": lambda e: e["source"].update(
                commit=e["source"]["commit"] + " & echo pwned > pwned.txt"
            ),
            "name escapes src": lambda e: e.update(name="../opencv"),
            "empty name": lambda e: e.update(name=""),
            # Windows resolves "..." to the parent and "opencv." to "opencv".
            "dots-only name": lambda e: e.update(name="..."),
            "trailing-dot name": lambda e: e.update(name="opencv."),
            "hidden name": lambda e: e.update(name=".hidden"),
            # Only pinned git sources: an archive, hg or source-less entry has nothing to verify.
            "archive source": lambda e: e["source"].update(type="archive"),
            "hg source": lambda e: e["source"].update(type="hg"),
            "no source": lambda e: e.pop("source"),
            "predicate": lambda e: e.update(predicate="True"),
        }
        for label, corrupt in cases.items():
            with self.subTest(label):
                self.make_sandbox()
                bad = self.entry()
                corrupt(bad)
                # A valid entry first: validation must reject the file before any clone.
                self.write_libraries([self.entry(name="good"), bad])

                result = self.run_bootstrap(*FROM_CPP_REF, cwd=self.cpp_ref)

                self.assert_refused(result)
                self.assertIn("Invalid schema", result.stdout, _output(result))
                self.assertEqual(self.find("pwned.txt"), [])
                self.assertEqual(self.clones(), [])

    def test_bad_cli_input_exits_nonzero(self) -> None:
        for args in (("--bootstrap_file", "x"), ("-n", "nope", *FROM_CPP_REF)):
            with self.subTest(args=args):
                result = self.run_bootstrap(*args, cwd=self.cpp_ref)
                # Exactly 2: a crash later in the run (rc 1) must not pass for a usage error.
                self.assertEqual(result.returncode, 2, _output(result))

    def assert_rejected_before_any_write(self, option: str, args: tuple[str, ...]) -> None:
        """In a fresh sandbox with a dirty managed checkout, a run with `args` exits 2 naming
        `option`, and writes nothing."""
        self.make_sandbox()
        self.assert_ok(self.run_bootstrap(*FROM_CPP_REF, cwd=self.cpp_ref))
        untracked = os.path.join(self.managed, "precious.txt")
        data = b"untracked and irreplaceable\n"
        self.write(untracked, data)
        state_cache = os.path.join(self.cpp_ref, ".libraries.json")
        before = self.repo_snapshot(self.managed)
        layout = sorted(os.listdir(self.cpp_ref))
        self.assertNotIn("snapshots", layout)
        self.assertNotIn("archives", layout)
        cache = self.read(state_cache)

        result = self.run_bootstrap(*args, *FROM_CPP_REF, cwd=self.cpp_ref)

        self.assertEqual(self.repo_snapshot(self.managed), before, _output(result))
        self.assertEqual(self.read(untracked), data)
        self.assertEqual(self.head(self.managed), self.c1)
        # Nothing written: no snapshots/ or archives/ dir, the same state cache.
        self.assertEqual(sorted(os.listdir(self.cpp_ref)), layout, _output(result))
        self.assertEqual(self.read(state_cache), cache, _output(result))
        self.assertEqual(result.returncode, 2, _output(result))
        self.assertIn(
            f"ERROR: {option} is not supported in purecv's vendored bootstrap.py",
            result.stdout,
            _output(result),
        )

    def test_fallback_options_are_rejected(self) -> None:
        # Upstream's fallback sends a git library that fails (a dirty checkout, say) to
        # downloadAndExtractFile, which deletes cpp_ref/src/<name> without --clean. All three
        # fallback options must be refused while the options are parsed, before any write.
        options = {
            "--fallback-url": ("--fallback-url", "http://127.0.0.1:1"),
            "--force-fallback": ("--force-fallback",),
            "--repo-snapshots": ("--repo-snapshots",),
        }
        for option, args in options.items():
            with self.subTest(option=option):
                self.assert_rejected_before_any_write(option, args)

    def test_archive_options_are_rejected(self) -> None:
        # Only pinned git sources pass validation, so no archive is ever extracted and
        # --use-tar and --use-unzip would be silently ignored: refuse them instead.
        for option in ("--use-tar", "--use-unzip"):
            with self.subTest(option=option):
                self.assert_rejected_before_any_write(option, (option,))

        result = self.run_bootstrap("--help", cwd=self.cpp_ref)

        self.assert_ok(result)
        lines = result.stdout.splitlines()
        for option in ("--use-tar", "--use-unzip"):
            with self.subTest(help=option):
                listed = [i for i, line in enumerate(lines) if option in line]
                self.assertEqual(len(listed), 1, result.stdout)
                self.assertIn("Not supported", " ".join(lines[listed[0] : listed[0] + 2]), result.stdout)

    def test_linked_state_cache_is_refused(self) -> None:
        # cpp_ref/.libraries.json is a symlink to an unrelated JSON array: reading it as state
        # and writing the cache back would overwrite that file.
        external = os.path.join(self.root, "external.json")
        data = b'[{"name": "unrelated", "keep": "these bytes"}]\n'
        self.write(external, data)
        state_cache = os.path.join(self.cpp_ref, ".libraries.json")
        try:
            os.symlink(external, state_cache)
        except (OSError, NotImplementedError) as e:  # Windows without the symlink privilege
            self.skipTest(f"cannot create a symlink here: {e}")
        layout = sorted(os.listdir(self.cpp_ref))

        result = self.run_bootstrap(*FROM_CPP_REF, cwd=self.cpp_ref)

        self.assertEqual(self.read(external), data, _output(result))
        self.assertTrue(os.path.islink(state_cache), "the link itself must stay")
        # Refused before any write: no src/, no archives/.
        self.assertEqual(sorted(os.listdir(self.cpp_ref)), layout, _output(result))
        self.assertEqual(result.returncode, 2, _output(result))
        self.assertIn(f"ERROR: {state_cache} is a symlink or junction", result.stdout, _output(result))

    def test_hard_linked_state_cache_is_never_written_through(self) -> None:
        # A hard link cannot be told apart from a plain file, so the cache is written to a new
        # file that replaces the old name: the other name keeps its bytes.
        external = os.path.join(self.root, "external.json")
        data = b'[{"name": "unrelated", "keep": "these bytes"}]\n'
        self.write(external, data)
        state_cache = os.path.join(self.cpp_ref, ".libraries.json")
        os.link(external, state_cache)

        result = self.run_bootstrap(*FROM_CPP_REF, cwd=self.cpp_ref)

        self.assert_ok(result)
        self.assertIn(f"verify opencv @ {self.c1} OK", result.stdout, _output(result))
        self.assertEqual(self.read(external), data)
        self.assertFalse(os.path.samefile(external, state_cache))
        self.assertIn(self.entry(), json.loads(self.read(state_cache)))
        # No temporary file left behind.
        self.assertEqual(
            sorted(os.listdir(self.cpp_ref)),
            [".libraries.json", "bootstrap.py", "libraries.json", "src"],
        )


class BootstrapModuleTest(unittest.TestCase):
    def load(self):
        """Imports a fresh copy of bootstrap.py; returns the module and what its top level printed."""
        spec = importlib.util.spec_from_file_location("purecv_bootstrap_under_test", BOOTSTRAP)
        module = importlib.util.module_from_spec(spec)
        printed = io.StringIO()
        # Importing runs the script's top level; keep its env tweaks out of here.
        with mock.patch.dict(os.environ), contextlib.redirect_stdout(printed):
            spec.loader.exec_module(module)
        return module, printed.getvalue()

    def test_tls_verification_is_not_disabled(self) -> None:
        original = ssl._create_default_https_context
        self.addCleanup(setattr, ssl, "_create_default_https_context", original)
        self.load()

        self.assertIsNot(ssl._create_default_https_context, ssl._create_unverified_context)

    def test_import_prints_no_scp_warning(self) -> None:
        # The scp packages only serve the rejected fallback paths (#170, item 1).
        _module, printed = self.load()

        self.assertNotIn("paramiko", printed)

    def cache_path(self) -> str:
        directory = tempfile.mkdtemp(prefix="purecv-bootstrap-cache-")
        self.addCleanup(_force_rmtree, directory)
        return os.path.join(directory, ".libraries.json")

    def test_locked_state_cache_is_retried_then_skipped(self) -> None:
        # On Windows, a cache another process holds open makes os.replace fail with WinError 5.
        # A short lock is waited out; a lasting one keeps the old cache and the run goes on
        # (#171, item 1).
        module, _ = self.load()
        path = self.cache_path()
        real_replace = os.replace
        for failures, written in ((2, True), (module._REPLACE_ATTEMPTS, False)):
            with self.subTest(failures=failures):
                with open(path, "w") as f:
                    f.write("[]")
                calls = []

                def flaky_replace(src, dst, calls=calls, failures=failures):
                    calls.append(dst)
                    if len(calls) <= failures:
                        raise PermissionError(13, "The process cannot access the file")
                    real_replace(src, dst)

                printed = io.StringIO()
                with mock.patch.object(module.os, "replace", flaky_replace), \
                        mock.patch.object(module.time, "sleep"), contextlib.redirect_stdout(printed):
                    result = module.writeJSONData([{"name": "opencv"}], path)

                self.assertIs(result, written)
                self.assertEqual(len(calls), min(failures + 1, module._REPLACE_ATTEMPTS))
                with open(path) as f:
                    self.assertEqual(json.load(f), [{"name": "opencv"}] if written else [])
                self.assertEqual(os.listdir(os.path.dirname(path)), [".libraries.json"], "temp file left behind")
                self.assertEqual("could not update the state cache" in printed.getvalue(), not written)

    @unittest.skipIf(os.name == "nt", "POSIX permissions")
    def test_state_cache_follows_the_umask(self) -> None:
        # mkstemp creates 0600 files; the cache used to keep that mode (#171, item 5).
        module, _ = self.load()
        path = self.cache_path()
        old = os.umask(0o022)
        self.addCleanup(os.umask, old)

        self.assertTrue(module.writeJSONData([], path))

        self.assertEqual(stat.S_IMODE(os.stat(path).st_mode), 0o644)


if __name__ == "__main__":
    unittest.main()

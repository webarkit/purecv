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


class BootstrapModuleTest(unittest.TestCase):
    def test_tls_verification_is_not_disabled(self) -> None:
        original = ssl._create_default_https_context
        self.addCleanup(setattr, ssl, "_create_default_https_context", original)
        spec = importlib.util.spec_from_file_location("purecv_bootstrap_under_test", BOOTSTRAP)
        module = importlib.util.module_from_spec(spec)
        # Importing runs the script's top level; keep its banner and env tweaks out of here.
        with mock.patch.dict(os.environ), contextlib.redirect_stdout(io.StringIO()):
            spec.loader.exec_module(module)

        self.assertIsNot(ssl._create_default_https_context, ssl._create_unverified_context)


if __name__ == "__main__":
    unittest.main()

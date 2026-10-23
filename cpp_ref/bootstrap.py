#!/usr/bin/python3

#
# https://github.com/corporateshark/bootstrapping.git
# sk@linderdaum.com
#
# The MIT License (MIT)
# Copyright (c) 2016-2025, Sergey Kosarevsky
#
# ---
# Based on https://bitbucket.org/blippar/bootstrapping-external-libs
#
# The MIT License (MIT)
# Copyright (c) 2016 Blippar.com Ltd
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
# purecv: P6 -- provenance. Vendored from https://github.com/webarkit/WebARKitLib-rs,
# benchmarks/bootstrap.py at commit a099326efc2edcf5b1ebcdb69e6d3d9adfe4bbfc
# (blob b22bd1d8d1fdb1095a37f3b9ea9c374e68f6b03a), patched for purecv, see `# purecv:` markers.
# Patches P1-P8 are listed in cpp_ref/README.md.

from __future__ import annotations # purecv: P4 (type hints of the purecv helpers)
from __future__ import print_function
import platform
import os
import re # purecv: P5
import stat # purecv: P3
import sys
import io
import shutil
import subprocess
import tempfile # purecv: P6
import time # purecv: P8
import zipfile
import tarfile
import hashlib
import json
import getopt
import traceback
import urllib
import ssl
import ctypes

# purecv: P2 -- the global `ssl._create_unverified_context` override is removed; TLS is verified.

try:
    from urllib.request import urlparse
    from urllib.request import urlunparse
    from urllib.request import quote
except ImportError:
    from urlparse import urlparse
    from urlparse import urlunparse
    from urllib import URLopener
    from urllib import quote

# purecv: P7 -- no warning at import: scp only serves the fallback and snapshot download paths,
# which purecv rejects (P6). downloadSCP still reports the missing packages if ever reached.
try:
    import paramiko
    import scp
    scp_available = True
except:
    scp_available = False

try:
    import lzma
    lzma_available = True
except:
    print("WARNING: Python lzma library not available; extraction of .tar.xz files may not be supported.")
    print("Installation on Ubuntu:")
    print("> apt-get install python-lzma")
    print("Installation on Mac OS X:")
    print("> brew install xz")
    print("> pip install pyliblzma")
    lzma_available = False

BOOTSTRAP_VERSION = "1.0.7 (2025) purecv-2" # purecv: P6, P7, P8

class Colors:
    GREEN = '\033[92m'
    WARNING = '\033[91m'
    NORMAL = '\033[0m'

SRC_DIR_BASE = "src"
ARCHIVE_DIR_BASE = "archives"
SNAPSHOT_DIR_BASE = "snapshots"

BASE_DIR = os.getcwd()
SRC_DIR = os.path.join(BASE_DIR, SRC_DIR_BASE)
ARCHIVE_DIR = os.path.join(BASE_DIR, ARCHIVE_DIR_BASE)
SNAPSHOT_DIR = os.path.join(BASE_DIR, SNAPSHOT_DIR_BASE)

DEFAULT_PNUM = 3
DEBUG_OUTPUT = False
FALLBACK_URL = ""

USE_TAR = False
USE_UNZIP = False

TOOL_COMMAND_PYTHON = sys.executable if not " " in sys.executable else '"{}"'.format(sys.executable)
TOOL_COMMAND_GIT = "git"
TOOL_COMMAND_HG = "hg"
TOOL_COMMAND_SVN = "svn"
TOOL_COMMAND_PATCH = "patch"
TOOL_COMMAND_TAR = "tar"
TOOL_COMMAND_UNZIP = "unzip"

ansi_console = True

if platform.system() == "Windows":
    os.environ['CYGWIN'] = "nodosfilewarning"
    ansi_console = False
    if sys.getwindowsversion().major >= 10:
        ansi_console = True
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.GetStdHandle(-11)
        mode = ctypes.c_uint32()
        kernel32.GetConsoleMode(handle, ctypes.byref(mode))
        kernel32.SetConsoleMode(handle, mode.value | 0x0004)

if not sys.version_info[0] >= 3:
    raise ValueError("I require Python 3.0 or a later version")

def log(string):
    print("--- " + string)

def warning(string):
    if ansi_console:
      print(Colors.WARNING, "--- " + string, Colors.NORMAL)
    else:
      print("--- " + string)

def dlog(string):
    if DEBUG_OUTPUT:
        print("*** " + string)

def executeCommand(command, printCommand = False, quiet = False):

    printCommand = printCommand or DEBUG_OUTPUT
    out = None
    err = None

    if quiet:
        out = open(os.devnull, 'w')
        err = subprocess.STDOUT

    if printCommand:
        if DEBUG_OUTPUT:
            dlog(">>> " + command)
        else:
            log(">>> " + command)

    return subprocess.call(command, shell = True, stdout=out, stderr=err);


def dieIfNonZero(res):
    if res != 0:
        raise ValueError("Command returned non-zero status: " + str(res));

def escapifyPath(path):
    if path.find(" ") == -1:
        return path
    if platform.system() == "Windows":
        return "\"" + path + "\""
    return path.replace("\\ ", " ")

# purecv: P3, P4 -- a symlink, or a Windows junction (any name-surrogate reparse point).
def _is_link(path: str) -> bool:
    if os.path.islink(path):
        return True
    if hasattr(os.path, "isjunction") and os.path.isjunction(path): # Python 3.12+
        return True
    try:
        st = os.lstat(path)
    except OSError:
        return False
    if not getattr(st, "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT:
        return False
    # Junctions and symlinks are name surrogates (they point elsewhere); cloud placeholders are not.
    tag = getattr(st, "st_reparse_tag", None)
    return tag is None or bool(tag & 0x20000000)

# purecv: P3 -- shutil.rmtree that also deletes read-only files. Git for Windows writes its
# object and pack files read-only, which made --clean die with WinError 5. It never
# deletes or chmods through a symlink or junction.
def _rmtree(path: str) -> None:
    if _is_link(path):
        raise OSError("refusing to delete through the symlink or junction " + path +
                      "; remove the link itself by hand")

    def make_writable_and_retry(func, failed_path, _exc):
        # rmtree reports a link it refuses to enter as a failed os.path.islink: keep refusing.
        if func is os.path.islink or _is_link(failed_path):
            raise OSError("refusing to delete through the symlink or junction " + failed_path)
        os.chmod(failed_path, os.lstat(failed_path).st_mode | stat.S_IWRITE)
        func(failed_path)

    if sys.version_info >= (3, 12):
        shutil.rmtree(path, onexc=make_writable_and_retry)
    else:
        shutil.rmtree(path, onerror=make_writable_and_retry)

# purecv: P4 -- git without a shell, isolated from any enclosing repository.
_GIT_EXECUTABLE = None
_GIT_LOCAL_ENV_VARS = []

# purecv: P7, P8 -- settings every bootstrap git command runs with, whatever the user's or the
# checkout's config says. P7: no detached-HEAD advice block on `checkout --detach`. P8: an
# fsmonitor hook could report "nothing changed" for an edited file; a global core.ignoreStat
# would mark every entry of a fresh clone assume-unchanged, and a global core.sparseCheckout
# would make every fresh clone sparse, so each checkout would then be refused as dirty. The
# checkout's own sparse setting is still read with `git config --local` (_hidden_edit_flags).
_GIT_FORCED_CONFIG = [
    "advice.detachedHead=false",
    "core.fsmonitor=false",
    "core.ignoreStat=false",
    "core.sparseCheckout=false",
]

def _git(args: list[str], cwd: str | None = None, check: bool = True,
         input: str | None = None) -> subprocess.CompletedProcess:
    """Runs git with an argument list and no shell, so paths with spaces and hostile values
    stay single arguments.

    The environment drops every variable `git rev-parse --local-env-vars` names (GIT_DIR,
    GIT_WORK_TREE, GIT_INDEX_FILE, ...) and sets GIT_CEILING_DIRECTORIES to SRC_DIR, so a
    command for one library can never act on an enclosing repository such as purecv.
    `--no-replace-objects` (P8) makes HEAD's tree the real tree of the pinned commit, so a
    replace ref cannot pass an edited tree off as that commit; _GIT_FORCED_CONFIG applies.

    Probes (check=False) run quietly and return their output; failing is their answer.
    Actions (check=True) show git's own progress and errors, and raise on failure.
    `input`, when given, is written to git's stdin.
    """
    global _GIT_EXECUTABLE, _GIT_LOCAL_ENV_VARS
    if _GIT_EXECUTABLE is None:
        git = shutil.which("git")
        if git is None:
            raise RuntimeError("git not found on PATH")
        _GIT_LOCAL_ENV_VARS = subprocess.run([git, "rev-parse", "--local-env-vars"], stdout=subprocess.PIPE,
                                             universal_newlines=True, check=True).stdout.split()
        _GIT_EXECUTABLE = git
    env = {k: v for k, v in os.environ.items() if k not in _GIT_LOCAL_ENV_VARS}
    env["GIT_CEILING_DIRECTORIES"] = os.path.realpath(SRC_DIR)
    command = [_GIT_EXECUTABLE, "--no-replace-objects"] # purecv: P8
    if platform.system() == "Windows":
        command += ["-c", "core.longpaths=true"]
    for setting in _GIT_FORCED_CONFIG: # purecv: P7, P8
        command += ["-c", setting]
    command += args
    dlog(">>> " + " ".join(command))
    sys.stdout.flush() # keep our log lines ahead of git's own output
    capture = None if check else subprocess.PIPE
    return subprocess.run(command, cwd=cwd, env=env, shell=False, check=check, stdout=capture, stderr=capture,
                          input=input, encoding="utf-8", errors="replace")

def _status_paths(lib_dir: str) -> list[str] | None:
    """Modified, untracked and ignored paths in lib_dir, or None when git cannot tell."""
    result = _git(["status", "--porcelain=v1", "-z", "--ignored", "--untracked-files=normal"], cwd=lib_dir,
                  check=False)
    if result.returncode != 0:
        return None
    fields = result.stdout.split("\0")
    paths = []
    i = 0
    while i < len(fields):
        entry = fields[i]
        i += 1
        if not entry:
            continue
        paths.append(entry[3:])
        if entry[0] in "RC": # a rename or copy: its source path follows as a field of its own
            i += 1
    return paths

# purecv: P4 -- `git status` cannot see an edit to an index entry marked assume-unchanged (git
# never looks at the file) or skip-worktree (git treats the file as matching the index), nor a
# file a sparse checkout leaves out. A clean status proves nothing while any of them is set.
def _hidden_edit_flags(lib_dir: str) -> list[tuple[str, str]] | None:
    """(flag, detail) pairs for everything in lib_dir that can hide an edit from `git status`,
    or None when git cannot tell.

    Call it only on a checkout of its own (see _checkout_state): its git directory is then
    lib_dir/.git, whose config is the --local one read here.
    """
    flags = []
    listed = _git(["ls-files", "-v", "-z"], cwd=lib_dir, check=False)
    if listed.returncode != 0:
        return None
    for entry in listed.stdout.split("\0"):
        if not entry:
            continue
        # "<tag> <path>": a lowercase tag marks assume-unchanged, S (s with it) skip-worktree.
        tag, path = entry[0], entry[2:]
        if tag.islower():
            flags.append(("assume-unchanged", path))
        if tag in ("S", "s"):
            flags.append(("skip-worktree", path))
    # purecv: P8 -- only the checkout's own config turns sparse checkout on for it: a global
    # setting is overridden by _git, and a .git/info/sparse-checkout file left behind by
    # `git sparse-checkout disable` does nothing while core.sparseCheckout is false.
    sparse = _git(["config", "--local", "--bool", "core.sparseCheckout"], cwd=lib_dir, check=False)
    if sparse.returncode not in (0, 1): # 1: not set
        return None
    if sparse.returncode == 0 and sparse.stdout.strip() == "true":
        flags.append(("sparse checkout", "core.sparseCheckout is true"))
    filtered = _filtered_paths(lib_dir) # purecv: P8
    if filtered is None:
        return None
    flags += [("filter", detail) for detail in filtered]
    file_mode = _ignored_file_mode(lib_dir) # purecv: P8
    if file_mode is None:
        return None
    if file_mode:
        flags.append(("core.fileMode", file_mode))
    return flags

# purecv: P8 -- a clean/smudge (or process) filter rewrites files between the worktree and the
# index, so `git status` compares filtered content. It matters only for a path whose `filter`
# attribute names a driver that is configured somewhere; OpenCV sets none.
def _filtered_paths(lib_dir: str) -> list[str] | None:
    """"<path> (filter <driver>)" for every tracked path a configured filter driver applies
    to, or None when git cannot tell."""
    listed = _git(["ls-files", "-z"], cwd=lib_dir, check=False)
    if listed.returncode != 0:
        return None
    if not listed.stdout:
        return []
    attrs = _git(["check-attr", "-z", "--stdin", "filter"], cwd=lib_dir, check=False, input=listed.stdout)
    if attrs.returncode != 0:
        return None
    fields = attrs.stdout.split("\0")
    drivers = {}
    found = []
    for i in range(0, len(fields) - 2, 3): # "<path>\0filter\0<value>\0" per path
        path, value = fields[i], fields[i + 2]
        if value in ("unspecified", "unset", "set", ""):
            continue # `set` names no driver, so git applies none
        if value not in drivers:
            configured = _git(["config", "-z", "--get-regexp", r"^filter\." + re.escape(value) +
                               r"\.(clean|smudge|process)$"], cwd=lib_dir, check=False)
            if configured.returncode not in (0, 1): # 1: none configured
                return None
            drivers[value] = configured.returncode == 0
        if drivers[value]:
            found.append(path + " (filter " + value + ")")
    return found

# purecv: P8 -- with core.fileMode false, `git status` ignores the executable bit. Git sets it
# false itself on filesystems that do not keep the bit (Windows, some mounts), where nothing
# can hide there; it hides a chmod only where the filesystem does keep the bit.
def _ignored_file_mode(lib_dir: str) -> str | None:
    """A description when core.fileMode is false on a filesystem that keeps the executable
    bit, "" when there is nothing to report, or None when git cannot tell."""
    if platform.system() == "Windows":
        return ""
    mode = _git(["config", "--bool", "core.fileMode"], cwd=lib_dir, check=False)
    if mode.returncode not in (0, 1): # 1: not set, git's default is true
        return None
    if mode.returncode == 1 or mode.stdout.strip() != "false":
        return ""
    try:
        fd, probe = tempfile.mkstemp(prefix="filemode-probe.", dir=os.path.join(lib_dir, ".git"))
    except OSError:
        return None
    try:
        os.close(fd)
        os.chmod(probe, 0o755)
        keeps_bit = bool(os.stat(probe).st_mode & stat.S_IXUSR)
        os.chmod(probe, 0o644)
        keeps_bit = keeps_bit and not os.stat(probe).st_mode & stat.S_IXUSR
    except OSError:
        return None
    finally:
        try:
            os.remove(probe)
        except OSError:
            pass
    return "core.fileMode is false on a filesystem that keeps the executable bit" if keeps_bit else ""

def _flag_names(flags: list[tuple[str, str]]) -> str: # purecv: P4
    return ", ".join(dict.fromkeys(flag for flag, _detail in flags))

# purecv: P8 -- the bootstrap's own git commands override core.ignoreStat, but a git command
# the user runs in the checkout does not: with the setting on in their global or system config,
# that command marks entries assume-unchanged and every later run refuses the checkout.
def _user_setting_hint(lib_dir: str, flags: list[tuple[str, str]]) -> str:
    """A sentence naming a global or system core.ignoreStat that explains assume-unchanged
    flags, or ""."""
    if not any(flag == "assume-unchanged" for flag, _detail in flags):
        return ""
    for scope in ("--global", "--system"):
        value = _git(["config", scope, "--bool", "core.ignoreStat"], cwd=lib_dir, check=False)
        if value.returncode == 0 and value.stdout.strip() == "true":
            return (" core.ignoreStat is true in your " + scope[2:] + " git config, so git commands you run in"
                    " the checkout mark its files assume-unchanged; unset it there, or leave the checkout"
                    " to this script.")
    return ""

def _log_first(lines: list[str], limit: int = 20) -> None: # purecv: P4
    for line in lines[:limit]:
        log("    " + line)
    if len(lines) > limit:
        log("    ... and " + str(len(lines) - limit) + " more")

def _checkout_state(lib_dir: str, commit: str) -> str:
    """Classifies lib_dir against its pinned commit:

    missing  absent, or an empty directory
    ok       its own git toplevel, HEAD == commit, clean
    drifted  its own git toplevel, clean, HEAD != commit
    dirty    its own git toplevel, with modified, untracked or ignored files, or with
             assume-unchanged or skip-worktree index entries or a sparse checkout, which
             hide edits from `git status`
    corrupt  anything else: a file, a symlink or junction, a checkout whose .git is not a
             directory of its own (a gitfile, as in a `git worktree` of another
             repository), or not the toplevel of a working git repository

    "Its own git toplevel" means: lib_dir is a real directory whose real path is
    <real path of BASE_DIR>/src/<name> (so a linked src/ does not count either),
    lib_dir/.git is a real directory, and git reports lib_dir as the toplevel and
    lib_dir/.git as the git directory.
    """
    if not os.path.lexists(lib_dir):
        return "missing"
    if _is_link(lib_dir) or not os.path.isdir(lib_dir):
        return "corrupt"
    if not os.listdir(lib_dir):
        return "missing"
    name = os.path.basename(os.path.normpath(lib_dir))
    # Anchored on BASE_DIR, not on SRC_DIR: realpath(SRC_DIR) would follow a linked src/.
    expected = os.path.join(os.path.realpath(BASE_DIR), SRC_DIR_BASE, name)
    if os.path.normcase(os.path.realpath(lib_dir)) != os.path.normcase(expected):
        return "corrupt"
    git_dir = os.path.join(lib_dir, ".git")
    if _is_link(git_dir) or not os.path.isdir(git_dir):
        return "corrupt"
    # git prints long, '/'-separated, symlink-resolved paths: compare files, never strings.
    probe = _git(["rev-parse", "--show-toplevel", "--absolute-git-dir"], cwd=lib_dir, check=False)
    reported = probe.stdout.splitlines() if probe.returncode == 0 else []
    try:
        own = (len(reported) == 2 and os.path.samefile(reported[0], lib_dir)
               and os.path.samefile(reported[1], git_dir))
    except OSError:
        own = False
    if not own:
        return "corrupt"
    paths = _status_paths(lib_dir)
    hidden = _hidden_edit_flags(lib_dir) # purecv: P4 -- edits `git status` cannot see
    if paths is None or hidden is None:
        return "corrupt"
    if paths or hidden:
        return "dirty"
    head = _git(["rev-parse", "--verify", "--quiet", "HEAD"], cwd=lib_dir, check=False)
    return "ok" if head.returncode == 0 and head.stdout.strip() == commit else "drifted"

def cloneRepository(type, url, target_name, revision, try_only_local_operations = False, recursive = True, branch = None):
    target_dir = escapifyPath(os.path.join(SRC_DIR, target_name))
    target_dir_exists = os.path.exists(target_dir)
    if type != "git": # purecv: P7 -- for git, "Cloning" is logged only when a clone happens
        log("Cloning " + url + " to " + target_dir)

    if type == "hg":
        repo_exists = os.path.exists(os.path.join(target_dir, ".hg"))

        if not repo_exists:
            if try_only_local_operations:
                raise RuntimeError("Repository for " + target_name + " not found; cannot execute local operations only")
            if target_dir_exists:
                dlog("Removing directory " + target_dir + " before cloning")
                _rmtree(target_dir) # purecv: P3
            dieIfNonZero(executeCommand(TOOL_COMMAND_HG + " clone " + url + " " + target_dir))
        elif not try_only_local_operations:
            log("Repository " + target_dir + " already exists; pulling instead of cloning")
            dieIfNonZero(executeCommand(TOOL_COMMAND_HG + " pull -R " + target_dir))

        if revision is None:
            revision = ""
        dieIfNonZero(executeCommand(TOOL_COMMAND_HG + " update -R " + target_dir + " -C " + revision))
        dieIfNonZero(executeCommand(TOOL_COMMAND_HG + " purge -R " + target_dir + " --config extensions.purge="))

    elif type == "git":
        # purecv: P4 -- act on the checkout's state (see _checkout_state) instead of running
        # `reset --hard` and `clean -fxd` on whatever happens to be there.
        lib_dir = os.path.join(SRC_DIR, target_name)
        if revision is None:
            raise ValueError("git library '" + target_name + "' has no pinned commit")
        state = _checkout_state(lib_dir, revision)

        if state == "ok":
            log(target_name + " is at " + revision + " and clean; nothing to do")
            return
        if state == "dirty":
            paths = _status_paths(lib_dir) or []
            hidden = _hidden_edit_flags(lib_dir) or [] # purecv: P4
            if paths:
                warning("ERROR: " + lib_dir + " has modified, untracked or ignored files:")
                _log_first(paths)
            if hidden: # purecv: P4
                warning("ERROR: " + lib_dir + " has index flags or a sparse checkout that hide edits from git status:")
                _log_first([flag + ": " + detail for flag, detail in hidden])
            # Name the library: a bare --clean would delete and re-clone every library.
            if not paths: # purecv: P4 -- only hidden edits: there is nothing to move out
                raise RuntimeError("refusing to trust the checkout " + lib_dir + ": git status cannot see every"
                                   " edit there (" + _flag_names(hidden) + "); rerun with --clean -n " +
                                   target_name + " to delete the checkout and clone it again." +
                                   _user_setting_hint(lib_dir, hidden)) # purecv: P8
            raise RuntimeError("refusing to touch the dirty checkout " + lib_dir + "; move those files out, or rerun"
                               " with --clean -n " + target_name + " to delete the checkout (and them) and clone it"
                               " again")
        if state == "corrupt":
            if _is_link(lib_dir):
                raise RuntimeError("refusing to touch " + lib_dir + ": it is a symlink or junction to a tree this"
                                   " script does not own; remove the link itself by hand, then rerun")
            if os.path.isfile(os.path.join(lib_dir, ".git")):
                raise RuntimeError("refusing to touch " + lib_dir + ": its .git is a file, so it belongs to another"
                                   " repository (e.g. a `git worktree`); remove it from there with"
                                   " `git worktree remove`, or rerun with --clean -n " + target_name + " to delete"
                                   " this directory")
            raise RuntimeError("refusing to touch " + lib_dir + ": it is not a git checkout of its own (or its .git"
                               " belongs to another repository); rerun with --clean -n " + target_name + " to delete"
                               " it and clone it again")

        if state == "missing":
            if try_only_local_operations:
                raise RuntimeError("Repository for " + target_name + " not found; cannot execute local operations only")
            # purecv: P7 -- clone without checking anything out, then check out the pinned commit
            # below. `clone --branch <annotated tag>` warned "refs/tags/<tag> <object> is not a
            # commit!" (opencv_contrib's 4.10.0). The clone fetches every branch and the tags
            # that point into them, so the pinned commit is normally there already; otherwise
            # the fetch below still asks for `branch`. Submodules follow the checkout.
            log("Cloning " + url + " to " + lib_dir)
            _git(["clone", "--no-checkout", "--", url, lib_dir])
        else:
            log(target_name + " is " + state + "; moving it to " + revision)

        # `cat-file -e` exits 128 when the commit is absent, so it is a probe (check=False).
        if _git(["cat-file", "-e", revision + "^{commit}"], cwd=lib_dir, check=False).returncode != 0:
            if try_only_local_operations:
                raise RuntimeError("Commit " + revision + " of " + target_name + " not found; cannot execute local operations only")
            _git(["fetch", "--", url, branch or revision], cwd=lib_dir)
        _git(["checkout", "--detach", revision], cwd=lib_dir)
        if recursive:
            _git(["submodule", "update", "--init", "--recursive"], cwd=lib_dir)

    elif type == "svn":
        if not try_only_local_operations: # we can't do much without a server connection when dealing with SVN
            if target_dir_exists:
                dlog("Removing directory " + target_dir + " before cloning")
                _rmtree(target_dir) # purecv: P3
            dieIfNonZero(executeCommand(TOOL_COMMAND_SVN + " checkout " + url + " " + target_dir))

        if revision is not None and revision != "":
            raise RuntimeError("Updating to revision not implemented for SVN.")

    else:
        raise ValueError("Cloning " + type + " repositories not implemented.")


def decompressTarXZFile(src_filename, dst_filename):
    if not lzma_available:
        raise RuntimeError("lzma extraction not available; please install package lzma (pyliblzma) and try again")

    try:
        fs = open(src_filename, "rb")
        if not fs:
            raise RuntimeError("Opening file " + src_filename + " failed")
        fd = open(dst_filename, "wb")
        if not fd:
            raise RuntimeError("Opening file " + dst_filename + " failed")

        decompressed = lzma.decompress(fs.read())
        fd.write(decompressed)
    finally:
        fs.close()
        fd.close()



def extractFile(filename, target_dir):
    if os.path.exists(target_dir):
        _rmtree(target_dir) # purecv: P3

    log("Extracting file " + filename)
    stem, extension = os.path.splitext(os.path.basename(filename))

    if extension == ".zip" or extension == "":
        zfile = None
        try:
            zfile = zipfile.ZipFile(filename)
        except zipfile.BadZipFile:
            warning("WARNING: Invalid ZIP file '" + filename + "'")
            if os.path.exists(filename) and os.path.getsize(filename) == 0:
                warning("WARNING: Zero-sized file was deleted. Run the script again.")
                os.remove(filename)
            else:
                warning("WARNING: Try deleting the cached file and run the script again.")
            raise RuntimeError("Invalid ZIP file '" + filename + "'") from None
        extract_dir = os.path.commonprefix(zfile.namelist())
        hasFolder = False
        for fname in zfile.namelist():
            if fname.find('/') != -1:
                hasFolder = True
        extract_dir_local = ""
        if not hasFolder: # special case, there are no folders in the archive
            extract_dir = ""
        if extract_dir == "":  # deal with stupid zip files that don't contain a base directory
            extract_dir, extension2 = os.path.splitext(os.path.basename(filename))
            extract_dir_local = extract_dir
        extract_dir_abs = os.path.join(SRC_DIR, extract_dir_local)

        try:
            os.mkdirs(extract_dir_abs)
        except:
            pass

        if not USE_UNZIP:
            zfile.extractall(extract_dir_abs)
            zfile.close()
        else:
            zfile.close()
            dieIfNonZero(executeCommand(TOOL_COMMAND_UNZIP + " " + filename + " -d " + extract_dir_abs))

    elif extension == ".tar" or extension == ".gz" or extension == ".bz2" or extension == ".xz":

        if extension == ".xz":# and not lzma_available:
            stem2, extension2 = os.path.splitext(os.path.basename(stem))
            if extension2 == ".tar":
                # we extract the .tar.xz file to a .tar file before we uncompress that
                tar_filename = os.path.join(os.path.dirname(filename), stem)
                decompressTarXZFile(filename, tar_filename)
                filename = tar_filename
            else:
                raise RuntimeError("Unable to extract .xz file that is not a .tar.xz file.")

        tfile = tarfile.open(filename)
        extract_dir = os.path.commonprefix(tfile.getnames())
        extract_dir_local = ""
        if extract_dir == "":  # deal with stupid tar files that don't contain a base directory
            extract_dir, extension2 = os.path.splitext(os.path.basename(filename))
            extract_dir_local = extract_dir
        extract_dir_abs = os.path.join(SRC_DIR, extract_dir_local)

        try:
            os.mkdirs(extract_dir_abs)
        except:
            pass

        if not USE_TAR:
            tfile.extractall(extract_dir_abs)
            tfile.close()
        else:
            tfile.close()
            dieIfNonZero(executeCommand(TOOL_COMMAND_TAR + " -x -f " + filename + " -C " + extract_dir_abs))

    else:
        raise RuntimeError("Unknown compressed file format " + extension)

    if platform.system() == "Windows":
        extract_dir = extract_dir.replace( '/', '\\' )
        target_dir = target_dir.replace( '/', '\\' )
        if extract_dir[-1::] == '\\':
            extract_dir = extract_dir[:-1]

    # rename extracted folder to target_dir
    extract_dir_abs = os.path.join(SRC_DIR, extract_dir)

    needRename = True

    if platform.system() == "Windows":
       needRename = extract_dir_abs.lower() != target_dir.lower()

    if needRename: os.rename(extract_dir_abs, target_dir)


def createArchiveFromDirectory(src_dir_name, archive_name, delete_existing_archive = False):
    if delete_existing_archive and os.path.exists(archive_name):
        dlog("Removing snapshot file " + archive_name + " before creating new one")
        os.remove(archive_name)

    archive_dir = os.path.dirname(archive_name)
    if not os.path.isdir(archive_dir):
        os.mkdir(archive_dir)

    with tarfile.open(archive_name, "w:gz") as tar:
        tar.add(src_dir_name, arcname = os.path.basename(src_dir_name))


def downloadSCP(hostname, username, path, target_dir):
    if not scp_available:
        warning("ERROR: missing Python packages [paramiko, scp]; cannot continue.")
        raise RuntimeError("Missing Python packages [paramiko, scp]; cannot continue.")
    ssh = paramiko.SSHClient()
    ssh.load_system_host_keys()
    ssh.connect(hostname = hostname, username = username)
    scpc = scp.SCPClient(ssh.get_transport())
    scpc.get(path, local_path = target_dir);

def downloadProgress(cur_size, total_size):
    percent = int((cur_size / total_size)*100)
    print("[", end = "")
    for i in range(int(percent/2)):
        print("*", end = "")
    for i in range(int(percent/2), 50):
        print(".", end = "")
    print("] " + str(percent) + "% --- ", end = "")
    print("%.2f" % (cur_size / (1024*1024)), "Mb", end = "\r")

def computeFileHash(filename):
    blocksize = 65536
    hasher = hashlib.sha1()
    with open(filename, 'rb') as afile:
        buf = afile.read(blocksize)
        while len(buf) > 0:
            hasher.update(buf)
            buf = afile.read(blocksize)
    return hasher.hexdigest()

def downloadFile(url, download_dir, target_dir_name, sha1_hash = None, force_download = False, user_agent = None):
    if not os.path.isdir(download_dir):
        os.mkdir(download_dir)

    p = urlparse(url)
    url = urlunparse([p[0], p[1], quote(p[2]), p[3], p[4], p[5]]) # replace special characters in the URL path

    filename_rel = os.path.split(p.path)[1] # get original filename
    target_filename = os.path.join(download_dir, filename_rel)

    # check SHA1 hash, if file already exists
    if os.path.exists(target_filename) and sha1_hash is not None and sha1_hash != "":
        hash_file = computeFileHash(target_filename)
        if hash_file != sha1_hash:
            log("Hash of " + target_filename + " (" + hash_file + ") does not match expected hash (" + sha1_hash + "); forcing download")
            force_download = True

    # download file
    if (not os.path.exists(target_filename)) or force_download:
        log("Downloading " + url + " to " + target_filename)
        if p.scheme == "ssh":
            downloadSCP(p.hostname, p.username, p.path, download_dir)
        else:
            opener = urllib.request.build_opener()
            if user_agent is not None:
                opener.addheaders = [('User-agent', user_agent)]
            f = open(target_filename, 'wb')
            with opener.open(url) as response:
                Length = response.getheader('content-length')
                BlockSize = 128*1024 # default value
                if Length:
                    Length = int(Length)
                    BlockSize = max(BlockSize, Length // 1000)
                    Size = 0
                    while True:
                        Buffer = response.read(BlockSize)
                        if not Buffer:
                            break
                        f.write(Buffer)
                        Size += len(Buffer)
                        downloadProgress(Size, Length)
                    print();
                else:
                    f.write(response.read())
            f.close()
    else:
        log("Skipping download of " + url + "; already downloaded")

    # check SHA1 hash
    if sha1_hash is not None and sha1_hash != "":
        hash_file = computeFileHash(target_filename)
        if hash_file != sha1_hash:
            errorStr = "Hash of " + target_filename + " (" + hash_file + ") differs from expected hash (" + sha1_hash + ")"
            log(errorStr)
            raise RuntimeError(errorStr)

    return target_filename


def downloadAndExtractFile(url, download_dir, target_dir_name, sha1_hash = None, force_download = False, user_agent = None):
    target_filename = downloadFile(url, download_dir, target_dir_name, sha1_hash, force_download, user_agent)
    extractFile(target_filename, os.path.join(SRC_DIR, target_dir_name))


def applyPatchFile(patch_name, dir_name, pnum):
    # we're assuming the patch was applied like in this example:
    # diff --exclude=".git" --exclude=".hg" -rupN ./src/AGAST/ ./src/AGAST_patched/ > ./patches/agast.patch
    # where the first given location is the unpatched directory, and the second location is the patched directory.
    log("Applying patch to " + dir_name)
    patch_dir = os.path.join(BASE_DIR, "patches")
    arguments = "-d " + os.path.join(SRC_DIR, dir_name) + " -p" + str(pnum) + " < " + os.path.join(patch_dir, patch_name)
    argumentsBinary = "-d " + os.path.join(SRC_DIR, dir_name) + " -p" + str(pnum) + " --binary < " + os.path.join(patch_dir, patch_name)
    res = executeCommand(TOOL_COMMAND_PATCH + " --dry-run " + arguments, quiet = True)
    if res != 0:
        arguments = argumentsBinary
        res = executeCommand(TOOL_COMMAND_PATCH + " --dry-run " + arguments, quiet = True)
    if res != 0:
        warning("ERROR: patch application failure; has this patch already been applied?")
        executeCommand(TOOL_COMMAND_PATCH + " --dry-run " + arguments, printCommand = True)
        exit(255)
    else:
        dieIfNonZero(executeCommand(TOOL_COMMAND_PATCH + " " + arguments, quiet = True))


def runPythonScript(script_name):
    log("Running Python script " + script_name)
    patch_dir = os.path.join(BASE_DIR, "patches")
    filename = os.path.join(patch_dir, script_name)
    dieIfNonZero(executeCommand(TOOL_COMMAND_PYTHON + " " + escapifyPath(filename), False));


def findToolCommand(command, paths_to_search, required = False):
    command_res = command
    found = False

    for path in paths_to_search:
        command_abs = os.path.join(path, command)
        if os.path.exists(command_abs):
            command_res = command_abs
            found = True
            break;

    if required and not found:
        warning("WARNING: command " + command + " not found, but required by script")

    dlog("Found '" + command + "' as " + command_res)
    return command_res


def readJSONData(filename):
    try:
        json_data = open(filename).read()
    except:
        warning("ERROR: Could not read JSON file: " + filename)
        return None

    try:
        data = json.loads(json_data)
    except json.JSONDecodeError as e:
        warning("ERROR: Could not parse JSON document: {}\n    {} (line {}:{})\n".format(filename, e.msg, e.lineno, e.colno))
        return None
    except:
        warning("ERROR: Could not parse JSON document: " + filename)
        return None

    return data


_REPLACE_ATTEMPTS = 5 # purecv: P8
_REPLACE_FIRST_DELAY = 0.1 # seconds, doubled after each failed attempt

def writeJSONData(data, filename):
    """Writes the state cache. Returns True when written, False when the cache file stayed
    locked (P8); raises on any other failure."""
    # purecv: P6 -- write a new file next to the state cache, then os.replace() it. The rename
    # swaps the directory entry, so the file behind a hard link (or a link planted after main
    # checked) is never written, and a crash never leaves a truncated cache.
    fd, temp_filename = tempfile.mkstemp(prefix=os.path.basename(filename) + ".", suffix=".tmp",
                                         dir=os.path.dirname(os.path.abspath(filename)))
    try:
        if platform.system() != "Windows": # purecv: P8 -- mkstemp gives 0600; follow the umask
            umask = os.umask(0)
            os.umask(umask)
            os.chmod(temp_filename, 0o666 & ~umask)
        with os.fdopen(fd, 'w') as outfile:
            json.dump(data, outfile)
        # purecv: P8 -- on Windows another process holding the cache open without
        # FILE_SHARE_DELETE (an editor, an antivirus, an indexer) makes os.replace fail with
        # WinError 5. Retry briefly; if it stays locked, keep the old cache and carry on: the
        # cache is an optimisation and is never trusted for git entries (P4).
        delay = _REPLACE_FIRST_DELAY
        for attempt in range(_REPLACE_ATTEMPTS):
            try:
                os.replace(temp_filename, filename)
                return True
            except PermissionError as e:
                if attempt == _REPLACE_ATTEMPTS - 1:
                    warning("WARNING: could not update the state cache " + filename + " (" + str(e) + "); another"
                            " program holds it open. Carrying on: the cache is never trusted for git libraries.")
                    try:
                        os.remove(temp_filename)
                    except OSError:
                        pass
                    return False
                time.sleep(delay)
                delay *= 2
    except BaseException:
        try:
            os.remove(temp_filename)
        except OSError:
            pass
        raise


def listLibraries(data):
    for library in data:
        name = library.get('name', None)
        if name is not None:
            print(name)


def printOptions():
        print("--------------------------------------------------------------------------------")
        print(BOOTSTRAP_VERSION)
        print("--------------------------------------------------------------------------------")
        print("Downloads external libraries, and applies patches or scripts if necessary.")
        print("If the --name argument is not provided, all available libraries will be")
        print("downloaded.")
        print("")
        print("Options:")
        print("  --list, -l              List all available libraries")
        print("  --name, -n              Specifies the name of a single library to be")
        print("                          downloaded")
        print("  --name-file, -N         Specifies a file that contains a (sub)set of libraries")
        print("                          to be downloaded. One library name per line; lines")
        print("                          starting with '#' are considered comments.")
        print("  --skip                  Specifies a name of a single library to be skipped")
        print("  --clean, -c             Remove library directory before obtaining library")
        print("  --clean-all, -C         Implies --clean, and also forces re-download of cached")
        print("                          archive files")
        print("  --base-dir, -b          Base directory, if script is called from outside of")
        print("                          its directory")
        print("  --bootstrap-file        Specifies the file containing the canonical bootstrap")
        print("                          JSON data (default: bootstrap.json)")
        print("  --local-bootstrap-file  Specifies the file containing local bootstrap JSON")
        print("                          data (e.g. for a particular project). The data in this")
        print("                          file will have higher precedence than the data from")
        print("                          the canonical bootstrap file.")
        # purecv: P6 -- the upstream fallback and archive options are rejected (see main)
        print("  --use-tar, --use-unzip, --repo-snapshots, --fallback-url, --force-fallback")
        print("                          Not supported in purecv's vendored copy: rejected")
        print("                          with exit code 2 before anything is written")
        print("  --debug-output          Enables extra debugging output")
        print("  --break-on-first-error  Terminate script once the first error is encountered")
        print("  --version               Print the script version")
        print("--------------------------------------------------------------------------------")


# purecv: P5 -- every entry is checked before anything is downloaded, cloned or deleted.
def _validate_library(library: dict) -> None:
    """Raises ValueError unless `library` is a safe, pinned entry."""
    if not isinstance(library, dict):
        raise ValueError("library entry is not a JSON object: " + repr(library))
    name = library.get('name', None)
    # No leading or trailing '.': Windows resolves "..." to the parent and "opencv." to "opencv".
    if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.-]*", name) or name.endswith("."):
        raise ValueError("invalid library name " + repr(name) + "; use letters, digits, '_', '.' and '-',"
                         " starting with a letter, digit or '_' and not ending with '.'")
    for key in ('predicate', 'postprocess'):
        if key in library:
            raise ValueError("library '" + name + "': the '" + key + "' key is not supported")
    # Only pinned git sources: an archive, hg, svn or source-less entry has no commit to verify,
    # and the source-less path deletes cpp_ref/src/<name> unconditionally.
    source = library.get('source', None)
    if not isinstance(source, dict):
        raise ValueError("library '" + name + "': 'source' must be a JSON object")
    if source.get('type', None) != "git":
        raise ValueError("library '" + name + "': source type " + repr(source.get('type', None)) +
                         " is not supported; use a pinned 'git' source")
    branch = source.get('branch', None)
    if branch is not None and (not isinstance(branch, str) or not re.fullmatch(r"[A-Za-z0-9._/-]+", branch)):
        raise ValueError("library '" + name + "': invalid branch " + repr(branch))
    commit = source.get('commit', None)
    if not isinstance(commit, str) or not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("library '" + name + "': 'commit' must be a full 40-hex commit SHA, got " + repr(commit))
    if source.get('revision', commit) != commit:
        raise ValueError("library '" + name + "': 'revision' differs from 'commit'; pin with 'commit' only")
    if not isinstance(source.get('url', None), str):
        raise ValueError("library '" + name + "': 'url' must be a string")


# purecv: P6 -- the last lines of every run: one verdict per selected library.
def _verify(library: dict, failed: bool) -> bool:
    name = library['name']
    source = library.get('source', None) or {}
    if source.get('type', None) != "git":
        if failed:
            log("verify " + name + " MISMATCH: bootstrap failed")
            return False
        log("verify " + name + " OK (" + str(source.get('type', None)) + " source, no commit to check)")
        return True
    commit = source['commit']
    lib_dir = os.path.join(SRC_DIR, name) # purecv: P4 -- reused below to name hidden edits
    state = _checkout_state(lib_dir, commit)
    if state == "ok":
        log("verify " + name + " @ " + commit + " OK")
        return True
    reasons = {
        "missing": "missing: no checkout",
        "drifted": "drifted: HEAD is not " + commit,
        "dirty": "dirty: modified, untracked or ignored files",
        "corrupt": "corrupt: a link, a gitfile, or not a git checkout of its own",
    }
    reason = reasons[state]
    hidden = _hidden_edit_flags(lib_dir) if state == "dirty" else None # purecv: P4
    if hidden:
        reason = "dirty: git status cannot see every edit (" + _flag_names(hidden) + ")"
        if _status_paths(lib_dir):
            reason += ", and modified, untracked or ignored files"
    log("verify " + name + " MISMATCH: " + reason)
    return False


def main(argv):
    global BASE_DIR, SRC_DIR, ARCHIVE_DIR, DEBUG_OUTPUT, FALLBACK_URL, USE_TAR, USE_UNZIP
    global SNAPSHOT_DIR # purecv: P1
    global TOOL_COMMAND_PYTHON, TOOL_COMMAND_GIT, TOOL_COMMAND_HG, TOOL_COMMAND_SVN, TOOL_COMMAND_PATCH, TOOL_COMMAND_TAR, TOOL_COMMAND_UNZIP

    try:
        opts, args = getopt.getopt(
            argv,
            "ln:N:cCb:h",
            ["list", "name=", "name-file=", "skip=", "clean", "clean-all", "base-dir=", "bootstrap-file=", # purecv: P1 (base-dir takes a value)
             "local-bootstrap-file=", "use-tar", "use-unzip", "repo-snapshots", "fallback-url=",
             "force-fallback", "debug-output", "help", "break-on-first-error", "version"])
    except getopt.GetoptError as e: # purecv: P6 -- a usage error is not a success
        printOptions()
        warning("ERROR: " + str(e))
        return 2
    if args: # purecv: P6
        printOptions()
        warning("ERROR: unexpected argument(s): " + " ".join(args))
        return 2
    # purecv: P8 -- -h and --version answer first, even next to an option rejected below.
    for opt, _arg in opts:
        if opt in ("-h", "--help"):
            printOptions()
            return 0
        if opt == "--version":
            print(BOOTSTRAP_VERSION)
            return 0
    # purecv: P6 -- upstream's fallback and archive options are rejected here, before any write.
    # With --fallback-url (or --force-fallback), a git library that fails, a dirty checkout for
    # example, goes to downloadAndExtractFile, which deletes src/<name> without --clean.
    # --repo-snapshots only feeds such a fallback server. --use-tar and --use-unzip only choose
    # how an archive is extracted, and P5 admits pinned git sources only, so they would be
    # silently ignored. getopt expands abbreviations to these full names. Their upstream code
    # paths below stay, unreachable.
    fallback_reason = ("each library comes only from its pinned git url, and a dirty checkout is refused, never"
                       " replaced.")
    archive_reason = "each library comes only from its pinned git url, so no archive is ever extracted."
    unsupported_options = {
        "--fallback-url": fallback_reason,
        "--force-fallback": fallback_reason,
        "--repo-snapshots": fallback_reason,
        "--use-tar": archive_reason,
        "--use-unzip": archive_reason,
    }
    unsupported = [opt for opt, _arg in opts if opt in unsupported_options]
    if unsupported:
        for opt in dict.fromkeys(unsupported):
            warning("ERROR: " + opt + " is not supported in purecv's vendored bootstrap.py: " +
                    unsupported_options[opt])
        return 2

    opt_names = []
    name_files = []
    skip_libs = []
    opt_clean = False
    opt_clean_archives = False
    list_libraries = False

    default_bootstrap_filename = "bootstrap.json"
    bootstrap_filename = os.path.abspath(os.path.join(BASE_DIR, default_bootstrap_filename))
    bootstrap_file_given = False # purecv: P6
    local_bootstrap_filename = ""
    create_repo_snapshots = False
    force_fallback = False
    break_on_first_error = False

    base_dir_path = ""

    for opt, arg in opts:
        if opt in ("-h", "--help"):
            printOptions()
            return 0
        if opt in ("--version"):
            print(BOOTSTRAP_VERSION)
            return 0
        if opt in ("-l", "--list"):
            list_libraries = True
        if opt in ("-n", "--name"):
            opt_names.append(arg)
        if opt in ("-N", "--name-file"):
            name_files.append(os.path.abspath(arg))
        if opt in ("--skip",):
            skip_libs.append(arg)
        if opt in ("-c", "--clean"):
            opt_clean = True
        if opt in ("-C", "--clean-all"):
            opt_clean = True
            opt_clean_archives = True
        if opt in ("-b", "--base-dir"):
            base_dir_path = os.path.abspath(arg)
            BASE_DIR = base_dir_path
            SRC_DIR = os.path.join(BASE_DIR, SRC_DIR_BASE)
            ARCHIVE_DIR = os.path.join(BASE_DIR, ARCHIVE_DIR_BASE)
            # purecv: P6 -- no default bootstrap file here: it would replace an explicit
            # --bootstrap-file given before -b. The default is derived after the loop.
            log("Using " + arg + " as base directory")
        if opt in ("--bootstrap-file",):
            bootstrap_filename = os.path.abspath(arg)
            bootstrap_file_given = True # purecv: P6
            log("Using main bootstrap file " + bootstrap_filename)
        if opt in ("--local-bootstrap-file",):
            local_bootstrap_filename = os.path.abspath(arg)
            log("Using local bootstrap file " + local_bootstrap_filename)
        if opt in ("--use-tar",):
            USE_TAR = True
        if opt in ("--use-unzip",):
            USE_UNZIP = True
        if opt in ("--repo-snapshots",):
            create_repo_snapshots = True
            log("Will create repository snapshots")
        if opt in ("--fallback-url",):
            FALLBACK_URL = arg
        if opt in ("--force-fallback",):
            force_fallback = True
            log("Using fallback URL to fetch all libraries")
        if opt in ("--break-on-first-error",):
            break_on_first_error = True
        if opt in ("--debug-output",):
            DEBUG_OUTPUT = True

    # purecv: P6 -- without --bootstrap-file, the default follows the final base directory,
    # whatever the order of the options.
    if not bootstrap_file_given:
        bootstrap_filename = os.path.abspath(os.path.join(BASE_DIR, default_bootstrap_filename))

    # purecv: P1 -- without -b, every directory lives next to the bootstrap file, whatever
    # the cwd, so a run from the purecv root cannot clone into the crate's src/.
    if not base_dir_path:
        BASE_DIR = os.path.dirname(os.path.abspath(bootstrap_filename))
    SRC_DIR = os.path.join(BASE_DIR, SRC_DIR_BASE)
    ARCHIVE_DIR = os.path.join(BASE_DIR, ARCHIVE_DIR_BASE)
    SNAPSHOT_DIR = os.path.join(BASE_DIR, SNAPSHOT_DIR_BASE)
    if os.path.exists(os.path.join(BASE_DIR, "Cargo.toml")):
        warning("ERROR: base directory " + BASE_DIR + " contains Cargo.toml, so its src/ holds Rust code;"
                " refusing to put library checkouts there. Omit -b to use the bootstrap file's directory.")
        return 2
    # purecv: P1 -- src/, archives/ and snapshots/ must be real directories of BASE_DIR. Through
    # a link, a run would move a foreign checkout and --clean would delete it: refuse before
    # any write, with or without --clean.
    for directory, directory_base in ((SRC_DIR, SRC_DIR_BASE), (ARCHIVE_DIR, ARCHIVE_DIR_BASE),
                                      (SNAPSHOT_DIR, SNAPSHOT_DIR_BASE)):
        expected = os.path.join(os.path.realpath(BASE_DIR), directory_base)
        if _is_link(directory):
            reason = "is a symlink or junction"
        elif (os.path.lexists(directory)
              and os.path.normcase(os.path.realpath(directory)) != os.path.normcase(expected)):
            reason = "resolves to " + os.path.realpath(directory) + " instead of " + expected
        else:
            continue
        warning("ERROR: " + directory + " " + reason + "; refusing to write through it. Make it a real directory"
                " (remove a link by hand; --clean never does), then rerun.")
        return 2

    if platform.system() != "Windows":
        # Unfortunately some IDEs do not have a proper PATH environment variable set,
        # so we search manually for the required tools in some obvious locations.
        paths_to_search = os.environ["PATH"].split(":") + ["/usr/local/bin", "/opt/local/bin", "/usr/bin"]
        TOOL_COMMAND_PYTHON = findToolCommand(TOOL_COMMAND_PYTHON, paths_to_search, required = True)
        TOOL_COMMAND_GIT = findToolCommand(TOOL_COMMAND_GIT, paths_to_search, required = True)
        # purecv: P7 -- P5 admits pinned git sources only and rejects postprocess patches, so
        # hg, svn and patch are never run: a missing one is not worth a warning.
        TOOL_COMMAND_HG = findToolCommand(TOOL_COMMAND_HG, paths_to_search, required = False)
        TOOL_COMMAND_SVN = findToolCommand(TOOL_COMMAND_SVN, paths_to_search, required = False)
        TOOL_COMMAND_PATCH = findToolCommand(TOOL_COMMAND_PATCH, paths_to_search, required = False)
        TOOL_COMMAND_TAR = findToolCommand(TOOL_COMMAND_TAR, paths_to_search, required = USE_TAR)
        TOOL_COMMAND_UNZIP = findToolCommand(TOOL_COMMAND_UNZIP, paths_to_search, required = USE_UNZIP)

    if base_dir_path:
        os.chdir(base_dir_path)

    if name_files:
        for name_file in name_files:
            try:
                with open(name_file) as f:
                    opt_names_local = [l for l in (line.strip() for line in f) if l]
                    opt_names_local = [l for l in opt_names_local if l[0] != '#']
                    opt_names += opt_names_local
                    dlog("Name file contains: " + ", ".join(opt_names_local))
            except:
                warning("ERROR: cannot parse name file '" + name_file + "'")
                return -1

    if force_fallback and not FALLBACK_URL:
        warning("Error: cannot force usage of the fallback location without specifying a fallback URL")
        return -1;

    state_filename = os.path.join(os.path.dirname(os.path.splitext(bootstrap_filename)[0]), \
                                  "." + os.path.basename(os.path.splitext(bootstrap_filename)[0])) \
                     + os.path.splitext(bootstrap_filename)[1]

    dlog("bootstrap_filename = " + bootstrap_filename)
    dlog("state_filename     = " + state_filename)

    # purecv: P6 -- through a symlink or junction, the state cache would read an unrelated file
    # as state and then overwrite it. Refuse before reading it, and before any write.
    if _is_link(state_filename):
        warning("ERROR: " + state_filename + " is a symlink or junction; refusing to read or write the bootstrap"
                " state cache through it. Remove the link by hand (the cache is rebuilt), then rerun.")
        return 2

    # read canonical libraries data
    data = readJSONData(bootstrap_filename)
    if data is None:
        return -1;

    # some sanity checking
    for library in data:
        if library.get('name', None) is None:
            warning("ERROR: Invalid schema: library object does not have a 'name'")
            return -1

    # read local libraries data, if available
    local_data = None
    if local_bootstrap_filename:
        local_data = readJSONData(local_bootstrap_filename)

        if local_data is None:
            return -1;

        # some sanity checking
        for local_library in local_data:
            if local_library.get('name', None) is None:
                warning("ERROR: Invalid schema: local library object does not have a 'name'")
                return -1

    # merge canonical and local library data, if applicable; local libraries take precedence
    if local_data is not None:
        for local_library in local_data:
            local_name = local_library.get('name', None)
            found_canonical_library = False
            for n, library in enumerate(data):
                name = library.get('name', None)
                if local_name == name:
                    data[n] = local_library # overwrite library
                    found_canonical_library = True
            if not found_canonical_library:
                data.append(local_library)

    # purecv: P5 -- validate every entry before any action
    for library in data:
        try:
            _validate_library(library)
        except ValueError as e:
            warning("ERROR: Invalid schema: " + str(e))
            return -1

    # purecv: P6 -- an unknown -n name is a usage error, not an empty success
    unknown_names = [n for n in opt_names if n not in [library['name'] for library in data]]
    if unknown_names:
        warning("ERROR: unknown library name(s): " + ", ".join(unknown_names))
        return 2

    if list_libraries:
        listLibraries(data)
        return 0

    sdata = []
    if os.path.exists(state_filename):
        sdata = readJSONData(state_filename)
        # purecv: P8 -- an unreadable or malformed cache made upstream crash with a TypeError on
        # sdata=None. It is only an optimisation (never trusted for git, P4): start a new one.
        if not isinstance(sdata, list) or not all(isinstance(s, dict) for s in sdata):
            warning("WARNING: ignoring the state cache " + state_filename + ": it is not a JSON array of"
                    " library entries. It is rewritten after the next library that succeeds.")
            sdata = []

    # create source directory
    if not os.path.isdir(SRC_DIR):
        log("Creating directory " + SRC_DIR)
        os.mkdir(SRC_DIR)

    # purecv: P7 -- archives/ is no longer created up front: P5 admits git sources only, so
    # nothing is ever archived. downloadFile still creates it on the unreachable archive paths.

    failed_libraries = []
    selected_libraries = [] # purecv: P6

    for library in data:
        name = library.get('name', None)
        source = library.get('source', None)
        post = library.get('postprocess', None)
        predicate = library.get('predicate', None)

        if (skip_libs) and (name in skip_libs):
            continue

        if (opt_names) and (not name in opt_names):
            continue

        if predicate is not None:
            log("Running predicate code for '" + name + "'")
            if eval(predicate) is not True:
                log("Predicate is not True for '" + name + "'; skipping library")
                continue

        selected_libraries.append(library) # purecv: P6

        lib_dir = os.path.join(SRC_DIR, name)
        lib_dir = lib_dir.replace(os.path.sep, '/')

        dlog("********** LIBRARY " + name + " **********")
        dlog("lib_dir = " + lib_dir + ")")

        # compare against cached state
        cached_state_ok = False
        if not opt_clean and (source or {}).get('type', None) != "git": # purecv: P4 -- never trust the cache for git
            for slibrary in sdata:
                sname = slibrary.get('name', None)
                if sname is not None and sname == name and slibrary == library and os.path.exists(lib_dir):
                    cached_state_ok = True
                    break

        if cached_state_ok:
            log("Cached state for '" + name + "' equals expected state; skipping library")
            continue
        else:
            # remove cached state for library
            sdata[:] = [s for s in sdata if not (lambda s, name : s.get('name', None) is not None and s['name'] == name)(s, name)]

        # create library directory, if necessary
        if opt_clean:
            log("Cleaning directory for " + name)
            # purecv: P3 -- --clean never deletes through a link: refuse loudly, touch nothing.
            if _is_link(lib_dir):
                warning("ERROR: " + lib_dir + " is a symlink or junction; --clean refuses to delete through it."
                        " Remove the link itself by hand, then rerun.")
                failed_libraries.append(name)
                continue
            if os.path.lexists(lib_dir): # purecv: P3
                _rmtree(lib_dir) # purecv: P3
        if not os.path.lexists(lib_dir): # purecv: P4 -- a dangling link is not "missing"
            os.makedirs(lib_dir)

        try:
            # download source
            if source is not None:
                if 'type' not in source:
                    warning("ERROR: Invalid schema for '" + name + "': 'source' object must have a 'type'")
                    return -1
                if 'url' not in source:
                    warning("ERROR: Invalid schema for '" + name + "': 'source' object must have a 'url'")
                    return -1
                src_type = source['type']
                src_url = source['url']

                if src_type == "sourcefile":
                    sha1 = source.get('sha1', None)
                    user_agent = source.get('user-agent', None)
                    try:
                        if force_fallback:
                            raise RuntimeError
                        downloadFile(src_url, ARCHIVE_DIR, name, sha1, force_download = opt_clean_archives, user_agent = user_agent)
                        filename_rel = os.path.basename(src_url)
                        shutil.copyfile( os.path.join(ARCHIVE_DIR, filename_rel), os.path.join(lib_dir, filename_rel) )
                    except:
                        if FALLBACK_URL:
                            if not force_fallback:
                                log("WARNING: Downloading of file " + src_url + " failed; trying fallback")

                            p = urlparse(src_url)
                            filename_rel = os.path.split(p.path)[1] # get original filename
                            p = urlparse(FALLBACK_URL)
                            fallback_src_url = urlunparse([p[0], p[1], p[2] + "/" + ARCHIVE_DIR_BASE + "/" + filename_rel, p[3], p[4], p[5]])
                            downloadFile(fallback_src_url, ARCHIVE_DIR, name, sha1, force_download = True)
                            shutil.copyfile( os.path.join(ARCHIVE_DIR, filename_rel), os.path.join(lib_dir, filename_rel) )
                        else:
                            _rmtree(lib_dir) # purecv: P3
                            raise
                elif src_type == "archive":
                    sha1 = source.get('sha1', None)
                    user_agent = source.get('user-agent', None)
                    try:
                        if force_fallback:
                            raise RuntimeError
                        downloadAndExtractFile(src_url, ARCHIVE_DIR, name, sha1, force_download = opt_clean_archives, user_agent = user_agent)
                    except:
                        if FALLBACK_URL:
                            if not force_fallback:
                                log("WARNING: Downloading of file " + src_url + " failed; trying fallback")

                            p = urlparse(src_url)
                            filename_rel = os.path.split(p.path)[1] # get original filename
                            p = urlparse(FALLBACK_URL)
                            fallback_src_url = urlunparse([p[0], p[1], p[2] + "/" + ARCHIVE_DIR_BASE + "/" + filename_rel, p[3], p[4], p[5]])
                            downloadAndExtractFile(fallback_src_url, ARCHIVE_DIR, name, sha1, force_download = True)
                        else:
                            raise

                else:
                    # `commit` is an alias for `revision` (clearer name in JSON schemas).
                    revision = source.get('revision', source.get('commit', None))
                    recursive = source.get('recursive', True)
                    branch = source.get('branch', None)

                    archive_name = name + ".tar.gz" # for reading or writing of snapshot archives
                    if revision is not None:
                        archive_name = name + "_" + revision + ".tar.gz"

                    try:
                        if force_fallback:
                            raise RuntimeError
                        cloneRepository(src_type, src_url, name, revision, False, recursive, branch)

                        if create_repo_snapshots:
                            log("Creating snapshot of library repository '" + name + "'")
                            repo_dir = os.path.join(SRC_DIR, name)
                            archive_filename = os.path.join(SNAPSHOT_DIR, archive_name)

                            dlog("Snapshot will be saved as " + archive_filename)
                            createArchiveFromDirectory(repo_dir, archive_filename, revision is None)

                    except Exception: # purecv: P6 -- Ctrl-C must not trigger the fallback
                        if FALLBACK_URL:
                            if not force_fallback:
                                log("WARNING: Cloning of repository " + src_url + " failed; trying fallback")

                            # copy archived snapshot from fallback location
                            p = urlparse(FALLBACK_URL)
                            fallback_src_url = urlunparse([p[0], p[1], p[2] + "/" + SNAPSHOT_DIR_BASE + "/" + archive_name, p[3], p[4], p[5]])
                            dlog("Looking for snapshot " + fallback_src_url + " of library repository " + name)

                            # create snapshots files directory
                            downloadAndExtractFile(fallback_src_url, SNAPSHOT_DIR, name, force_download = True)

                            # reset repository state to particular revision (only using local operations inside the function)
                            cloneRepository(src_type, src_url, name, revision, True, True, branch)
                        else:
                            raise
            else:
                # set up clean directory for potential patch application
                _rmtree(lib_dir) # purecv: P3
                os.mkdir(lib_dir)

            # post-processing
            if post is not None:
                if 'type' not in post:
                    warning("ERROR: Invalid schema for '" + name + "': 'postprocess' object must have a 'type'")
                    return -1
                if 'file' not in post:
                    warning("ERROR: Invalid schema for '" + name + "': 'postprocess' object must have a 'file'")
                    return -1
                post_type = post['type']
                post_file = post['file']

                if post_type == "patch":
                    applyPatchFile(post_file, name, post.get('pnum', DEFAULT_PNUM))
                elif post_type == "script":
                    runPythonScript(post_file)
                else:
                    warning("ERROR: Unknown post-processing type '" + post_type + "' for " + name)
                    return -1

            # add to cached state
            sdata.append(library)

            # write out cached state
            writeJSONData(sdata, state_filename)
        except urllib.error.URLError as e:
            warning("ERROR: Failure to bootstrap library '" + name + "' (urllib.error.URLError: reason " + str(e.reason) + ")")
            if break_on_first_error:
                exit(-1)
            traceback.print_exc()
            failed_libraries.append(name)
        except Exception as e: # purecv: P6 -- catch errors, not Ctrl-C, so Ctrl-C aborts the run
            warning("ERROR: Failure to bootstrap library '" + name + "' (" + type(e).__name__ + ": " + str(e) + ")")
            if break_on_first_error:
                exit(-1)
            traceback.print_exc()
            failed_libraries.append(name)

    if failed_libraries:
        warning("***************************************")
        warning("FAILURE to bootstrap the following libraries:")
        warning(', '.join(failed_libraries))
        warning("***************************************")
    else: # purecv: P6 -- no early return: the verify pass below always runs
        log("Finished")

        # touch the state cache file
        if os.path.exists(state_filename): # purecv: P8 -- absent when no library was written
            os.utime(state_filename, None);

    # purecv: P6 -- verify every selected library; these lines end the output
    verified = [_verify(library, library['name'] in failed_libraries) for library in selected_libraries]
    return 0 if all(verified) and not failed_libraries else 1

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

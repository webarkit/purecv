#!/usr/bin/env python3
"""Run the ignored OpenCV parity tests as expected failures.

Every ``#[ignore = "diverges from OpenCV <version>, see #N"]`` test in ``tests/parity_*.rs``
asserts OpenCV's measured output and fails until issue #N is fixed. ``cargo test`` skips
ignored tests, so nothing in the normal CI run notices when a fix lands without flipping its
test. This script runs each parity suite with ``--ignored`` and exits non-zero when any
ignored test PASSED: the fix has landed and the test must lose its ``#[ignore]`` (see
"Fixing a parity issue" in CONTRIBUTING.md and rule 11 of docs/parity/AGENTS.md).

Usage::

    python scripts/parity_xfail.py                     # default features
    python scripts/parity_xfail.py --features simd     # extra cargo arguments pass through
    python scripts/parity_xfail.py --suite core --suite video

Only the Python standard library is used. Exit codes: 0 when every ignored test failed,
1 when one or more passed, 2 when a suite could not be run or its output could not be parsed.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time
from pathlib import Path

SUITES = ("core", "imgproc", "features2d", "video", "calib3d")

ROOT = Path(__file__).resolve().parent.parent

# libtest's human-readable result line: ``test name ... ok`` / ``FAILED`` / ``ignored``.
# Anchored at the start of a line, so a test's own output cannot forge a result.
RESULT_LINE = re.compile(r"^test (\S+) \.\.\. (ok|FAILED|ignored)(?:,.*)?$", re.MULTILINE)
# ``cargo test -- --list`` prints ``name: test`` for every test it would run.
LIST_LINE = re.compile(r"^(\S+): test$", re.MULTILINE)

FIX_LANDED = "fix landed: remove the #[ignore] and close the issue it cites"


def run_cargo(args: list[str]) -> subprocess.CompletedProcess[str]:
    """Run ``cargo`` in the repository root, capturing stdout and stderr as text."""
    return subprocess.run(
        ["cargo", *args],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )


def expected_tests(suite: str, cargo_args: list[str]) -> list[str] | None:
    """The ignored tests of a suite, from ``--list``; None when the listing failed."""
    proc = run_cargo(["test", "--test", f"parity_{suite}", *cargo_args, "--", "--ignored", "--list"])
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr)
        return None
    return LIST_LINE.findall(proc.stdout)


def check_suite(suite: str, cargo_args: list[str]) -> tuple[list[str], list[str], list[str]]:
    """Run a suite's ignored tests.

    Returns ``(passed, failed, problems)``: the ignored tests that PASSED, the ones that
    failed as expected, and the reasons the run itself cannot be trusted.
    """
    problems: list[str] = []
    expected = expected_tests(suite, cargo_args)
    if expected is None:
        return [], [], [f"parity_{suite}: `cargo test -- --ignored --list` failed"]

    # ``--no-fail-fast`` is cargo's: the other suites still run when this one has failures.
    # libtest itself runs every test of the binary before reporting.
    proc = run_cargo(
        ["test", "--test", f"parity_{suite}", "--no-fail-fast", *cargo_args, "--", "--ignored"]
    )
    results = dict(RESULT_LINE.findall(proc.stdout))
    if not results and proc.returncode != 0:
        sys.stderr.write(proc.stderr)
        return [], [], [f"parity_{suite}: the suite did not run (exit code {proc.returncode})"]

    passed = sorted(name for name, status in results.items() if status == "ok")
    failed = sorted(name for name, status in results.items() if status == "FAILED")
    unexpected = sorted(name for name, status in results.items() if status == "ignored")
    missing = sorted(set(expected) - set(results))
    extra = sorted(set(results) - set(expected))
    if unexpected:
        problems.append(f"parity_{suite}: reported as ignored under --ignored: {', '.join(unexpected)}")
    if missing:
        problems.append(f"parity_{suite}: listed but no result parsed: {', '.join(missing)}")
    if extra:
        problems.append(f"parity_{suite}: result for a test not in the listing: {', '.join(extra)}")
    # The two usual exit codes: 0 when every test passed, 101 when at least one failed.
    if proc.returncode not in (0, 101):
        problems.append(f"parity_{suite}: unexpected cargo exit code {proc.returncode}")
    return passed, failed, problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the ignored parity tests and fail when one of them passes.",
        epilog="Unknown arguments are passed to cargo test (for example --features simd).",
    )
    parser.add_argument(
        "--suite",
        action="append",
        choices=SUITES,
        help="restrict the run to this suite (repeatable; default: all five)",
    )
    args, cargo_args = parser.parse_known_args(argv)
    suites = args.suite or list(SUITES)

    all_passed: list[tuple[str, str]] = []
    all_problems: list[str] = []
    total_failed = 0
    started = time.monotonic()
    for suite in suites:
        t0 = time.monotonic()
        passed, failed, problems = check_suite(suite, cargo_args)
        total_failed += len(failed)
        all_passed.extend((suite, name) for name in passed)
        all_problems.extend(problems)
        print(
            f"parity_{suite}: {len(failed)} ignored test(s) still fail, {len(passed)} PASSED"
            f" ({time.monotonic() - t0:.1f}s)"
        )

    print(f"total: {total_failed} expected failure(s) in {time.monotonic() - started:.1f}s")
    if all_passed:
        print()
        print(FIX_LANDED + ":")
        for suite, name in all_passed:
            print(f"  tests/parity_{suite}.rs::{name}")
    if all_problems:
        print()
        print("the run cannot be trusted:")
        for problem in all_problems:
            print(f"  {problem}")
        return 2
    return 1 if all_passed else 0


if __name__ == "__main__":
    sys.exit(main())

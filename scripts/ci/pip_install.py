#!/usr/bin/env python
"""Helper to prefer a local wheelhouse for pip installs."""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="pip install with optional wheelhouse")
    parser.add_argument(
        "args",
        nargs=argparse.REMAINDER,
        help="Arguments to forward to pip install (use -- to separate)",
    )
    parser.add_argument(
        "-w",
        "--wheelhouse",
        dest="wheelhouse",
        default=os.environ.get("WHEELHOUSE_DIR"),
        help="Wheelhouse directory; if set and exists, installs with --no-index/--find-links",
    )
    parser.add_argument(
        "--require-wheelhouse",
        action="store_true",
        help="Fail if the provided wheelhouse path is missing",
    )
    return parser.parse_args(argv)


def split_args(argv: list[str]) -> tuple[list[str], list[str]]:
    if "--" in argv:
        separator_index = argv.index("--")
        return argv[:separator_index], argv[separator_index + 1 :]
    return argv, []


def main(argv: list[str]) -> int:
    wrapper_args, pip_args = split_args(argv)
    ns = parse_args(wrapper_args)
    extra_env = os.environ.copy()
    pip_cmd = [sys.executable, "-m", "pip", "install"]

    wheelhouse = ns.wheelhouse
    if wheelhouse:
        wheel_path = Path(wheelhouse).expanduser().resolve()
        extra_env["WHEELHOUSE_DIR"] = str(wheel_path)
        if not wheel_path.is_dir() and ns.require_wheelhouse:
            print(f"Required wheelhouse not found at: {wheel_path}", file=sys.stderr)
            return 1
        if wheel_path.is_dir():
            pip_cmd.extend(["--no-index", "--find-links", str(wheel_path)])
    if not pip_args:
        pip_args = ns.args
    if not pip_args:
        print("No pip arguments provided; pass packages or -r file", file=sys.stderr)
        return 2
    pip_cmd.extend(pip_args)
    print("[pip-install]", " ".join(pip_cmd))
    return subprocess.call(pip_cmd, env=extra_env)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

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


def main(argv: list[str]) -> int:
    if "--" not in argv:
        print("Use -- to separate pip arguments from wrapper options.", file=sys.stderr)
        return 2
    separator_index = argv.index("--")
    if separator_index == len(argv) - 1:
        print("No pip arguments provided after --.", file=sys.stderr)
        return 2
    ns = parse_args(argv[:separator_index])
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
    pip_args = argv[separator_index + 1 :]
    pip_cmd.extend(pip_args)
    print("[pip-install]", " ".join(pip_cmd))
    return subprocess.call(pip_cmd, env=extra_env)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

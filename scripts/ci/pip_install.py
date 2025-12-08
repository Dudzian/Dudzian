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
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    ns = parse_args(argv)
    extra_env = os.environ.copy()
    pip_cmd = [sys.executable, "-m", "pip", "install"]

    wheelhouse = ns.wheelhouse
    if wheelhouse:
        wheel_path = Path(wheelhouse).expanduser().resolve()
        extra_env["WHEELHOUSE_DIR"] = str(wheel_path)
        if wheel_path.is_dir():
            pip_cmd.extend(["--no-index", "--find-links", str(wheel_path)])
    if not ns.args:
        print("No pip arguments provided; pass packages or -r file", file=sys.stderr)
        return 2
    pip_cmd.extend(ns.args)
    print("[pip-install]", " ".join(pip_cmd))
    return subprocess.call(pip_cmd, env=extra_env)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

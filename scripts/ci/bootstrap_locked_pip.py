#!/usr/bin/env python
"""Install the exact pip version reviewed in the canonical release lock."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

if __package__:
    from scripts.ci.bootstrap_dependency import locked_versions
else:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from bootstrap_dependency import locked_versions


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        raise SystemExit("usage: bootstrap_locked_pip.py CANONICAL_LOCK")
    lock = Path(argv[0])
    version = locked_versions(lock).get("pip")
    if version is None:
        raise SystemExit(f"canonical lock has no exact pip pin: {lock}")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "--no-deps",
            f"pip=={version}",
        ],
        check=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

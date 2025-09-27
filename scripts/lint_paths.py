#!/usr/bin/env python3
"""CI helper that ensures forbidden legacy directories stay archived."""
from __future__ import annotations

import pathlib
import sys

# Paths are relative to repository root.
BANNED_PATHS = [
    pathlib.Path("KryptoLowca/bot"),
]


def main() -> int:
    repo_root = pathlib.Path(__file__).resolve().parents[1]
    violations: list[str] = []

    for rel_path in BANNED_PATHS:
        candidate = repo_root / rel_path
        if candidate.exists():
            violations.append(str(rel_path))

    if violations:
        joined = ", ".join(violations)
        print(
            "Disallowed legacy paths detected: "
            f"{joined}. Move files to archive/legacy_bot or delete them."
        )
        return 1

    print("Path lint passed: no forbidden directories present.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""CI helper that guards repo layout and import rules."""
from __future__ import annotations

import pathlib
import re
import sys

# Paths are relative to repository root.
BANNED_PATHS = [
    pathlib.Path("KryptoLowca/bot"),
]

# Python files inside these directories may still import ``KryptoLowca``
# because they implement the legacy surface itself.
_ALLOWED_KRYPTLOWCA_IMPORT_ROOTS = {
    pathlib.Path("KryptoLowca"),
    pathlib.Path("legacy_bridge"),
    pathlib.Path("archive"),
}

_IMPORT_PATTERN = re.compile(r"^\s*(?:from|import)\s+KryptoLowca\b", re.MULTILINE)


def main() -> int:
    repo_root = pathlib.Path(__file__).resolve().parents[1]
    failures: list[str] = []

    for rel_path in BANNED_PATHS:
        candidate = repo_root / rel_path
        if candidate.exists():
            failures.append(
                "Disallowed legacy paths detected: "
                f"{rel_path}. Move files to archive/legacy_bot or delete them."
            )

    forbidden_imports: list[str] = []
    for py_file in repo_root.rglob("*.py"):
        rel_file = py_file.relative_to(repo_root)
        if rel_file.parts and pathlib.Path(rel_file.parts[0]) in _ALLOWED_KRYPTLOWCA_IMPORT_ROOTS:
            continue
        if any(part.startswith(".") for part in rel_file.parts):
            continue
        text = py_file.read_text(encoding="utf-8")
        if _IMPORT_PATTERN.search(text):
            forbidden_imports.append(str(rel_file))

    if forbidden_imports:
        failures.append(
            "Found imports of KryptoLowca in new modules: "
            + ", ".join(sorted(forbidden_imports))
            + ". Use legacy_bridge or bot_core instead."
        )

    if failures:
        print("\n".join(failures))
        return 1

    print(
        "Repository layout lint passed: legacy directories absent and "
        "no forbidden KryptoLowca imports found."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

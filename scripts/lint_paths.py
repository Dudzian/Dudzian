#!/usr/bin/env python3
"""CI helper that guards repo layout and import rules."""
from __future__ import annotations

import os
import pathlib
import re
import sys

# Paths are relative to repository root.
_BANNED_ROOTS = {
    pathlib.Path("KryptoLowca"),
}
BANNED_PATHS = sorted(_BANNED_ROOTS)

_IMPORT_PATTERN = re.compile(r"^\s*(?:from|import)\s+KryptoLowca\b", re.MULTILINE)


def main() -> int:
    repo_root = pathlib.Path(__file__).resolve().parents[1]
    failures: list[str] = []
    warnings: list[str] = []
    allow_legacy = os.environ.get("LINT_PATHS_ALLOW_LEGACY", "0") == "1"

    for rel_path in BANNED_PATHS:
        candidate = repo_root / rel_path
        if candidate.exists():
            message = (
                "Disallowed legacy paths detected: "
                f"{rel_path}. Usuń katalog albo przenieś kod do bot_core."
            )
            if allow_legacy:
                warnings.append(message)
            else:
                failures.append(message)

    forbidden_imports: list[str] = []
    for py_file in repo_root.rglob("*.py"):
        rel_file = py_file.relative_to(repo_root)
        if rel_file.parts and pathlib.Path(rel_file.parts[0]) in _BANNED_ROOTS:
            # Presence of these directories is already reported separately. Skip redundant import checks.
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
            + ". Use bot_core instead."
        )

    if warnings:
        print("\n".join(f"WARNING: {warning}" for warning in warnings))

    if failures:
        print("\n".join(failures))
        return 1

    if warnings:
        print(
            "Repository layout lint passed with warnings: legacy directories "
            "oznaczono do usunięcia, nowe importy nie zostały znalezione."
        )
    else:
        print(
            "Repository layout lint passed: legacy directories absent and "
            "no forbidden KryptoLowca imports found."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())

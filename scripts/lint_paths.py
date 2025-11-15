#!/usr/bin/env python3
"""CI helper that guards repo layout and import rules."""
from __future__ import annotations

import pathlib
import re
import sys

# Paths are relative to repository root.
_BANNED_ROOTS = {
    pathlib.Path("KryptoLowca"),
}
BANNED_PATHS = sorted(_BANNED_ROOTS)

_IMPORT_PATTERN = re.compile(r"^\s*(?:from|import)\s+KryptoLowca\b", re.MULTILINE)
_EXECUTABLE_EXTENSIONS = {
    ".py",
    ".pyc",
    ".pyo",
    ".exe",
    ".bat",
    ".cmd",
    ".sh",
}


def main() -> int:
    repo_root = pathlib.Path(__file__).resolve().parents[1]
    failures: list[str] = []
    for rel_path in BANNED_PATHS:
        candidate = repo_root / rel_path
        if candidate.exists():
            message = (
                "Disallowed archival paths detected: "
                f"{rel_path}. Usuń katalog albo przenieś kod do bot_core."
            )
            failures.append(message)

    archive_dir = repo_root / "archive"
    if archive_dir.exists():
        archive_violations: list[str] = []
        for path in archive_dir.rglob("*"):
            if not path.is_file():
                continue
            rel = path.relative_to(repo_root)
            if any(part.startswith(".") for part in rel.parts):
                continue
            is_executable = bool(path.stat().st_mode & 0o111)
            has_executable_extension = path.suffix.lower() in _EXECUTABLE_EXTENSIONS
            if is_executable or has_executable_extension:
                archive_violations.append(str(rel))
        if archive_violations:
            failures.append(
                "Archive directory must not contain wykonywalnych plików: "
                + ", ".join(sorted(archive_violations))
                + ". Przenieś artefakty do dokumentacji historycznej lub usuń pliki."
            )

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

    if failures:
        print("\n".join(failures))
        return 1
    print(
        "Repository layout lint passed: brak zabronionych katalogów, importów i"
        " wykonywalnych plików w archive/."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

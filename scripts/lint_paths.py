#!/usr/bin/env python3
"""CI helper that guards repo layout and import rules."""
from __future__ import annotations

import os
import pathlib
import re
import subprocess
import sys

# Paths are relative to repository root.
_BANNED_ROOTS = {
    pathlib.Path("KryptoLowca"),
}
BANNED_PATHS = sorted(_BANNED_ROOTS)

_IMPORT_PATTERN = re.compile(r"^\s*(?:from|import)\s+KryptoLowca\b", re.MULTILINE)


def _collect_new_legacy_lines(repo_root: pathlib.Path) -> list[str]:
    """Zwraca listę nowych linii zawierających słowo 'legacy' spoza docs/migrations."""

    def _diff_output(args: list[str]) -> str:
        result = subprocess.run(
            args,
            cwd=repo_root,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        return result.stdout

    diff = _diff_output(["git", "diff", "--cached", "--unified=0"])
    if not diff.strip():
        diff = _diff_output(["git", "diff", "HEAD", "--unified=0"])
    if not diff.strip():
        return []

    occurrences: list[str] = []
    current_file: pathlib.Path | None = None
    new_line_no = 0
    for line in diff.splitlines():
        if line.startswith("+++ b/"):
            rel = line[6:]
            try:
                current_file = pathlib.Path(rel)
            except ValueError:
                current_file = None
            new_line_no = 0
            continue
        if line.startswith("@@"):
            parts = line.split()
            added = next((part for part in parts if part.startswith("+")), "+0")
            try:
                start = int(added.split(",", 1)[0][1:])
            except ValueError:
                start = 0
            new_line_no = start
            continue
        if not line.startswith("+") or line.startswith("+++"):
            if line and not line.startswith("-") and current_file is not None:
                new_line_no += 1
            continue
        if current_file is None:
            continue
        if "docs" in current_file.parts and "migrations" in current_file.parts:
            new_line_no += 1
            continue
        if "legacy" in line.lower():
            occurrences.append(f"{current_file}:{new_line_no}: {line[1:].strip()}")
        new_line_no += 1
    return occurrences


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

    archive_dir = repo_root / "archive"
    if archive_dir.exists():
        archive_py_files = sorted(
            str(path.relative_to(repo_root))
            for path in archive_dir.rglob("*.py")
            if path.is_file()
        )
        if archive_py_files:
            failures.append(
                "Archive directory must not contain Python modules: "
                + ", ".join(archive_py_files)
                + ". Przenieś kod do dokumentacji historycznej lub usuń pliki."
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

    new_legacy_occurrences = _collect_new_legacy_lines(repo_root)
    if new_legacy_occurrences:
        message = (
            "Detected new occurrences of 'legacy' outside migration docs:\n"
            + "\n".join(new_legacy_occurrences)
        )
        if allow_legacy:
            warnings.append(message)
        else:
            failures.append(message)

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

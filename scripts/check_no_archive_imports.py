#!/usr/bin/env python3
"""Skaner blokujący importy z katalogu archive/** w kodzie wykonywalnym.

Uruchomienie bez argumentów skanuje domyślne katalogi runtime/CI i
zwraca kod wyjścia 1 w przypadku znalezienia importu modułu zaczynającego się
od `archive` lub zawierającego segment `archive` (np. `from foo.archive import x`).
"""

from __future__ import annotations

import ast
import os
from collections.abc import Iterable
from pathlib import Path
import sys

DEFAULT_SCAN_ROOTS: tuple[Path, ...] = (
    Path("bot_core"),
    Path("core"),
    Path("ui"),
    Path("scripts"),
    Path("tests"),
)

_SKIP_DIR_NAMES = {
    ".git",
    ".hg",
    ".mypy_cache",
    ".pytest_cache",
    ".tox",
    ".venv",
    "__pycache__",
    "node_modules",
    "archive",
}


def _should_skip(path: Path) -> bool:
    return any(part in _SKIP_DIR_NAMES for part in path.parts)


def _iter_python_files(repo_root: Path, roots: Iterable[Path]) -> Iterable[Path]:
    for scan_root in roots:
        abs_root = (repo_root / scan_root).resolve()
        if not abs_root.exists():
            continue
        for dirpath, dirnames, filenames in os.walk(abs_root):
            rel_dir = Path(dirpath).relative_to(repo_root)
            if _should_skip(rel_dir):
                dirnames[:] = []
                continue
            dirnames[:] = [d for d in dirnames if not _should_skip(rel_dir / d)]
            for filename in filenames:
                if not filename.endswith(".py"):
                    continue
                rel_path = rel_dir / filename
                if _should_skip(rel_path):
                    continue
                yield rel_path


def _contains_archive_segment(module: str | None) -> bool:
    if not module:
        return False
    return "archive" in module.split(".")


def find_archive_imports(repo_root: Path, roots: Iterable[Path]) -> list[str]:
    violations: list[str] = []

    for rel_path in _iter_python_files(repo_root, roots):
        path = repo_root / rel_path
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:
            violations.append(f"{rel_path}: nie można sparsować pliku (SyntaxError)")
            continue

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if _contains_archive_segment(alias.name):
                        violations.append(
                            f"{rel_path}:{node.lineno}: zabroniony import modułu {alias.name}"
                        )
            elif isinstance(node, ast.ImportFrom):
                if _contains_archive_segment(node.module):
                    violations.append(
                        f"{rel_path}:{node.lineno}: zabroniony import modułu {node.module}"
                    )
    return violations


def main(argv: list[str] | None = None) -> int:
    args = argv or sys.argv[1:]
    repo_root = Path(__file__).resolve().parents[1]
    roots = [Path(arg) for arg in args] if args else list(DEFAULT_SCAN_ROOTS)
    violations = find_archive_imports(repo_root, roots)

    if violations:
        print("Znaleziono zabronione importy z archive/**:")
        print("\n".join(sorted(violations)))
        return 1
    print("Brak importów z archive/** w kodzie runtime/CI.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

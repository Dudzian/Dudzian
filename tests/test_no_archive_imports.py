from __future__ import annotations

import ast
import pkgutil
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
TARGET_DIRS = [PROJECT_ROOT / "bot_core", PROJECT_ROOT / "core", PROJECT_ROOT / "ui"]


def iter_python_files(paths: list[Path]) -> list[Path]:
    files: list[Path] = []
    for base in paths:
        if not base.exists():
            continue
        files.extend(sorted(base.rglob("*.py")))
    return files


def test_no_archive_imports_in_source() -> None:
    python_files = iter_python_files(TARGET_DIRS)
    offending: list[tuple[Path, str]] = []

    for file_path in python_files:
        module_ast = ast.parse(file_path.read_text(encoding="utf-8"), filename=str(file_path))
        for node in ast.walk(module_ast):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.split(".")[0] == "archive":
                        offending.append((file_path, alias.name))
            elif isinstance(node, ast.ImportFrom):
                if node.module and node.module.split(".")[0] == "archive":
                    offending.append((file_path, node.module))

    assert not offending, "Znaleziono importy z archive: " + "; ".join(
        f"{path}: {name}" for path, name in offending
    )


def test_pkgutil_does_not_expose_archive_packages() -> None:
    discovered = {m.name for m in pkgutil.iter_modules([str(PROJECT_ROOT)])}
    assert "archive" not in discovered, "pkgutil iteruje pakiety z archive/**"

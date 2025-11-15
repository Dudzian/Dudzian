"""QA test pilnujący braku historycznych tokenów w kodzie produkcyjnym."""
from __future__ import annotations

import os
from pathlib import Path

TOKEN_PATTERNS = (("leg" "acy"), ("krypto" "lowca"))
SKIP_DIR_NAMES = {
    ".git",
    ".hg",
    ".mypy_cache",
    ".pytest_cache",
    ".tox",
    ".venv",
    "__pycache__",
}
ALLOWLISTED_DIRS = (Path("docs"), Path("archive"))
ALLOWLISTED_FILES = {
    Path("README.md"),
    Path("scripts/lint_paths.py"),
}


def _is_allowlisted(path: Path) -> bool:
    for allowed in ALLOWLISTED_DIRS:
        if path.is_relative_to(allowed):
            return True
    return path in ALLOWLISTED_FILES


def _should_skip(path: Path) -> bool:
    return any(part in SKIP_DIR_NAMES for part in path.parts)


def _iter_repo_files(root: Path):
    for dirpath, dirnames, filenames in os.walk(root):
        rel_dir = Path(dirpath).relative_to(root)
        if _should_skip(rel_dir):
            dirnames[:] = []
            continue
        for filename in filenames:
            rel_path = (rel_dir / filename).relative_to(Path("."))
            if _should_skip(rel_path):
                continue
            yield rel_path


def test_no_stage5_tokens_outside_docs():  # pragma: no cover
    repo_root = Path(__file__).resolve().parents[1]
    violations = []

    for rel_path in _iter_repo_files(repo_root):
        if _is_allowlisted(rel_path):
            continue
        path = repo_root / rel_path
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            text = path.read_text(encoding="utf-8", errors="ignore")
        for lineno, line in enumerate(text.splitlines(), start=1):
            lowered = line.casefold()
            if any(token in lowered for token in TOKEN_PATTERNS):
                violations.append(f"{rel_path}:{lineno}: {line.strip()}")
                break

    assert not violations, (
        "Wykryto historyczne tokeny Stage5 w kodzie (poza dokumentacją):\n"
        + "\n".join(violations)
    )

"""QA test pilnujący braku historycznych tokenów w kodzie produkcyjnym."""
from __future__ import annotations

import os
from pathlib import Path

TOKEN_PATTERNS: tuple[str, ...] = ("stage5", "stage-5")
TOKEN_LITERAL = "stage5"
SKIP_DIR_NAMES = {
    ".git",
    ".hg",
    ".mypy_cache",
    ".pytest_cache",
    ".tox",
    ".venv",
    "__pycache__",
}
ALLOWLISTED_DIRS: tuple[Path, ...] = (Path("archive"),)
ALLOWLISTED_FILES: set[Path] = {
    Path("tests/qa/test_no_legacy_tokens.py"),
    Path("qa/test_no_legacy_tokens.py"),
}
ALLOWLISTED_FILENAME_PREFIXES: tuple[str, ...] = ()


def _is_allowlisted(path: Path) -> bool:
    for allowed in ALLOWLISTED_DIRS:
        if path.is_relative_to(allowed):
            return True
    if path in ALLOWLISTED_FILES:
        return True
    return any(path.name.startswith(prefix) for prefix in ALLOWLISTED_FILENAME_PREFIXES)


def _should_skip(path: Path) -> bool:
    return any(part in SKIP_DIR_NAMES for part in path.parts)


def _iter_repo_files(root: Path):
    for dirpath, dirnames, filenames in os.walk(root):
        rel_dir = Path(dirpath).relative_to(root)
        if _should_skip(rel_dir):
            dirnames[:] = []
            continue
        if rel_dir != Path(".") and _is_allowlisted(rel_dir):
            dirnames[:] = []
            continue
        for filename in filenames:
            rel_path = rel_dir / filename
            if _should_skip(rel_path):
                continue
            yield rel_path


def _find_hypercare_token_violations(repo_root: Path) -> list[str]:
    violations: list[str] = []

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
    return violations


def test_no_legacy_tokens_outside_archive():  # pragma: no cover
    repo_root = Path(__file__).resolve().parents[1]
    violations = _find_hypercare_token_violations(repo_root)

    assert not violations, (
        "Wykryto historyczne tokeny Stage" "5 w kodzie (poza archiwum):\n"
        + "\n".join(violations)
    )


def test_allowlisted_paths_can_use_token(tmp_path):
    (tmp_path / "archive").mkdir()
    (tmp_path / "archive" / "notes.txt").write_text(TOKEN_LITERAL, encoding="utf-8")

    (tmp_path / "core").mkdir()
    blocked_file = tmp_path / "core" / "engine.py"
    blocked_file.write_text(TOKEN_LITERAL, encoding="utf-8")

    violations = _find_hypercare_token_violations(tmp_path)

    assert len(violations) == 1
    assert str(blocked_file.relative_to(tmp_path)) in violations[0]

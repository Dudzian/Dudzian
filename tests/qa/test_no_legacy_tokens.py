"""QA test pilnujący braku historycznych tokenów w kodzie produkcyjnym."""
from __future__ import annotations

import os
from pathlib import Path

import pytest

TOKEN_PATTERNS: tuple[str, ...] = ("stage5", "stage-5", "stage4", "stage-4", "legacy")
SCAN_ROOTS: tuple[Path, ...] = (
    Path("bot_core"),
    Path("core"),
    Path("config"),
)

SKIP_DIR_NAMES = {
    ".git",
    ".hg",
    ".mypy_cache",
    ".pytest_cache",
    ".tox",
    ".venv",
    "__pycache__",
    "node_modules",
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


def _iter_repo_files(repo_root: Path):
    for scan_root in SCAN_ROOTS:
        abs_root = (repo_root / scan_root).resolve()
        if not abs_root.exists():
            continue
        for dirpath, dirnames, filenames in os.walk(abs_root):
            rel_dir = Path(dirpath).relative_to(repo_root)
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

@pytest.fixture(params=TOKEN_PATTERNS)
def forbidden_token(request: pytest.FixtureRequest) -> str:
    return str(request.param)


def test_allowlisted_paths_can_use_token(tmp_path: Path, forbidden_token: str):
    (tmp_path / "archive").mkdir()
    (tmp_path / "archive" / "notes.txt").write_text(forbidden_token, encoding="utf-8")

    (tmp_path / "bot_core").mkdir()
    blocked_file = tmp_path / "bot_core" / "engine.py"
    blocked_file.write_text(forbidden_token, encoding="utf-8")

    violations = _find_hypercare_token_violations(tmp_path)

    assert len(violations) == 1
    assert str(blocked_file.relative_to(tmp_path)) in violations[0]

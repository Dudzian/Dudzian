"""QA test pilnujący, aby runtime/CI nie importowały archiwum."""
from __future__ import annotations

from pathlib import Path

import pytest

from scripts.check_no_archive_imports import DEFAULT_SCAN_ROOTS, find_archive_imports


@pytest.fixture()
def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_no_archive_imports_in_runtime(repo_root: Path):  # pragma: no cover
    violations = find_archive_imports(repo_root, DEFAULT_SCAN_ROOTS)

    assert not violations, "\n".join(
        (
            "Wykryto importy z archive/** w kodzie runtime/CI.",
            *sorted(violations),
        )
    )


def test_detector_flags_archive_import(tmp_path: Path):
    runtime_dir = tmp_path / "bot_core"
    runtime_dir.mkdir(parents=True)
    allowed_archive = tmp_path / "archive"
    allowed_archive.mkdir()

    (runtime_dir / "uses_archive.py").write_text(
        "from archive.module import something\n", encoding="utf-8"
    )
    (allowed_archive / "note.py").write_text("import os\n", encoding="utf-8")

    violations = find_archive_imports(tmp_path, (Path("bot_core"), Path("archive")))

    assert len(violations) == 1
    assert "uses_archive.py" in violations[0]

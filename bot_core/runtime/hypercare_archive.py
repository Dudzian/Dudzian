"""Helpery archiwizacji raportów hypercare Stage5/Stage6."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import shutil

__all__ = ["archive_hypercare_reports"]


def _timestamp_slug(value: datetime | None = None) -> str:
    timestamp = value or datetime.now(timezone.utc)
    return timestamp.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _copy_if_exists(source: Path | None, destination: Path) -> None:
    if source is None:
        return
    expanded = source.expanduser()
    if not expanded.exists():
        return
    shutil.copy2(expanded, destination / expanded.name)


def archive_hypercare_reports(
    *,
    archive_dir: Path,
    stage5_summary: Path,
    stage6_summary: Path,
    stage5_signature: Path | None = None,
    stage6_signature: Path | None = None,
    full_summary: Path | None = None,
    extra_files: Iterable[Path] | None = None,
    timestamp: datetime | None = None,
) -> Path:
    """Archiwizuje wskazane raporty hypercare w katalogu docelowym."""

    archive_root = archive_dir.expanduser()
    archive_root.mkdir(parents=True, exist_ok=True)
    target_dir = archive_root / _timestamp_slug(timestamp)
    target_dir.mkdir(parents=True, exist_ok=True)

    for required in (stage5_summary, stage6_summary):
        expanded = required.expanduser()
        if not expanded.exists():
            raise FileNotFoundError(f"Nie znaleziono raportu hypercare: {expanded}")
        shutil.copy2(expanded, target_dir / expanded.name)

    _copy_if_exists(stage5_signature, target_dir)
    _copy_if_exists(stage6_signature, target_dir)
    _copy_if_exists(full_summary, target_dir)

    if extra_files:
        for path in extra_files:
            _copy_if_exists(path, target_dir)

    return target_dir

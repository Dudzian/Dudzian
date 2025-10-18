"""Obsługa magazynu raportów zdefiniowanego na poziomie środowiska."""
from __future__ import annotations

import os
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable

from bot_core.config.models import EnvironmentConfig, EnvironmentReportStorageConfig


def store_environment_report(
    summary_path: Path,
    environment: EnvironmentConfig,
    *,
    now: datetime | None = None,
) -> Path | None:
    """Zapisuje kopię raportu w magazynie środowiska, jeśli skonfigurowano."""

    storage_cfg = getattr(environment, "report_storage", None)
    if storage_cfg is None:
        return None

    backend = _normalized_backend(storage_cfg)
    if backend is None:
        return None
    if backend != "file":
        raise ValueError(
            "Nieobsługiwany backend magazynu raportów: {backend}".format(backend=storage_cfg.backend)
        )

    target_dir = _resolve_directory(environment, storage_cfg)
    target_dir.mkdir(parents=True, exist_ok=True)

    timestamp = now or datetime.now(timezone.utc)
    filename = _build_filename(storage_cfg, timestamp)
    target_path = target_dir / filename

    if summary_path.resolve() != target_path.resolve():
        shutil.copy2(summary_path, target_path)
    if storage_cfg.fsync:
        with target_path.open("rb+") as handle:
            handle.flush()
            os.fsync(handle.fileno())

    _prune_expired(target_dir.iterdir(), cutoff=_retention_cutoff(storage_cfg, timestamp), keep=target_path)
    return target_path


def _normalized_backend(config: EnvironmentReportStorageConfig) -> str | None:
    backend = str(getattr(config, "backend", "file") or "").strip().lower()
    if backend in {"", "disabled", "none"}:
        return None
    return backend


def _resolve_directory(
    environment: EnvironmentConfig, config: EnvironmentReportStorageConfig
) -> Path:
    directory_value = getattr(config, "directory", None)
    if not directory_value:
        return Path(environment.data_cache_path) / "reports"
    directory = Path(directory_value).expanduser()
    if directory.is_absolute():
        return directory
    base = Path(environment.data_cache_path).expanduser()
    return base / directory


def _build_filename(config: EnvironmentReportStorageConfig, timestamp: datetime) -> str:
    pattern = getattr(config, "filename_pattern", "reports-%Y%m%d.json") or "reports-%Y%m%d.json"
    return timestamp.astimezone(timezone.utc).strftime(pattern)


def _retention_cutoff(config: EnvironmentReportStorageConfig, timestamp: datetime) -> datetime | None:
    retention = getattr(config, "retention_days", None)
    if retention in (None, ""):
        return None
    try:
        days = float(retention)
    except (TypeError, ValueError):
        return None
    if days <= 0:
        return None
    return timestamp - timedelta(days=days)


def _prune_expired(entries: Iterable[Path], *, cutoff: datetime | None, keep: Path) -> None:
    if cutoff is None:
        return
    for entry in entries:
        try:
            if entry.resolve() == keep.resolve():
                continue
        except FileNotFoundError:
            continue
        if not entry.is_file():
            continue
        try:
            mtime = datetime.fromtimestamp(entry.stat().st_mtime, timezone.utc)
        except OSError:
            continue
        if mtime < cutoff:
            try:
                entry.unlink()
            except OSError:
                continue


__all__ = ["store_environment_report"]

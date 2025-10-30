"""Obsługa raportów jakości modeli AI zapisywanych lokalnie."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

DEFAULT_QUALITY_DIR = Path("var/models/quality")


def persist_quality_report(
    payload: Mapping[str, object],
    *,
    model_name: str,
    version: str,
    evaluated_at: datetime,
    base_dir: Path | str | None = None,
) -> Path:
    """Zapisuje raport jakości do archiwum i aktualizuje wskaźnik latest."""

    root = Path(base_dir) if base_dir is not None else DEFAULT_QUALITY_DIR
    root = root.expanduser()
    model_dir = root / model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    timestamp = evaluated_at.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%S")
    filename = f"{version}-{timestamp}.json"
    target = model_dir / filename
    _write_json(target, payload)

    latest_path = model_dir / "latest.json"
    temp_path = latest_path.with_suffix(".tmp")
    _write_json(temp_path, payload)
    temp_path.replace(latest_path)
    return target


def load_latest_quality_payload(
    model_name: str,
    *,
    base_dir: Path | str | None = None,
) -> Mapping[str, object] | None:
    root = Path(base_dir) if base_dir is not None else DEFAULT_QUALITY_DIR
    root = root.expanduser()
    path = root / model_name / "latest.json"
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):  # pragma: no cover - I/O zależne od środowiska
        return None
    if not isinstance(payload, Mapping):
        return None
    return payload


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


__all__ = ["DEFAULT_QUALITY_DIR", "load_latest_quality_payload", "persist_quality_report"]

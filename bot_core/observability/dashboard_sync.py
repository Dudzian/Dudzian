"""Obsługa synchronizacji override'ów alertów z dashboardem Grafana."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping, MutableMapping, Sequence

from bot_core.observability.alert_overrides import (
    AlertOverride,
    AlertOverrideManager,
)

_SCHEMA = "stage6.observability.dashboard_annotations"
_SCHEMA_VERSION = "1.0"


def _isoformat(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


@dataclass(slots=True)
class DashboardDefinition:
    """Reprezentuje definicję dashboardu Grafana."""

    payload: Mapping[str, object]
    uid: str | None


def load_dashboard_definition(path: Path) -> DashboardDefinition:
    """Wczytuje definicję dashboardu z pliku JSON."""

    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, Mapping):
        raise ValueError("Plik dashboardu musi zawierać obiekt JSON")
    uid = str(data.get("uid")) if data.get("uid") not in (None, "") else None
    return DashboardDefinition(payload=data, uid=uid)


def build_dashboard_annotations_payload(
    overrides: Sequence[AlertOverride],
    *,
    reference: datetime | None = None,
    dashboard_uid: str | None = None,
    panel_id: int | None = None,
) -> Mapping[str, object]:
    """Buduje dokument z anotacjami do publikacji w Grafanie."""

    reference = reference or datetime.now(timezone.utc)
    manager = AlertOverrideManager(overrides)
    manager.prune_expired(reference=reference)
    annotations = [
        override.to_grafana_event(dashboard_uid=dashboard_uid, panel_id=panel_id)
        for override in manager.active(reference=reference)
    ]
    payload: MutableMapping[str, object] = {
        "schema": _SCHEMA,
        "schema_version": _SCHEMA_VERSION,
        "generated_at": _isoformat(reference),
        "dashboard_uid": dashboard_uid,
        "panel_id": panel_id,
        "annotations": annotations,
        "summary": manager.summary(reference=reference),
    }
    return payload


def save_dashboard_annotations(
    annotations: Mapping[str, object],
    *,
    output_path: Path,
    pretty: bool = False,
) -> None:
    """Zapisuje dokument anotacji do pliku JSON."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        if pretty:
            json.dump(annotations, handle, ensure_ascii=False, indent=2)
        else:
            json.dump(annotations, handle, ensure_ascii=False, separators=(",", ":"))
        handle.write("\n")


def load_overrides_from_document(data: Mapping[str, object]) -> list[AlertOverride]:
    """Ładuje override'y z dokumentu JSON w formacie Stage6."""

    schema = data.get("schema")
    if schema != "stage6.observability.alert_overrides":
        raise ValueError("Oczekiwano dokumentu override'ów Stage6")
    entries = data.get("overrides")
    if not isinstance(entries, Iterable):
        raise ValueError("Sekcja 'overrides' musi być iterowalna")
    overrides: list[AlertOverride] = []
    for entry in entries:
        if not isinstance(entry, Mapping):
            continue
        overrides.append(AlertOverride.from_dict(entry))
    return overrides


__all__ = [
    "DashboardDefinition",
    "build_dashboard_annotations_payload",
    "load_dashboard_definition",
    "load_overrides_from_document",
    "save_dashboard_annotations",
]


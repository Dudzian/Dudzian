"""Funkcje pomocnicze do ładowania konfiguracji SLO Stage6."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml

from bot_core.observability.slo import (
    SLOCompositeDefinition,
    SLODefinition,
    SLOMeasurement,
)


def _coerce_sequence(value: Any) -> Iterable[Any]:
    if isinstance(value, Mapping):
        return value.values()
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
        return value
    return ()


def _parse_datetime(value: Any) -> datetime | None:
    if value in (None, ""):
        return None
    text = str(value)
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def _load_yaml_or_json(path: Path) -> Any:
    text = path.read_text(encoding="utf-8")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return yaml.safe_load(text)


def load_slo_definitions(path: Path) -> tuple[list[SLODefinition], list[SLOCompositeDefinition]]:
    """Ładuje definicje SLO i kompozytów z pliku YAML/JSON."""

    data = _load_yaml_or_json(path)
    definitions: list[SLODefinition] = []
    entries: Iterable[Any]
    if isinstance(data, Mapping):
        entries = data.get("definitions") or data.get("slo") or data.get("slos") or data
    else:
        entries = _coerce_sequence(data)
    for entry in _coerce_sequence(entries):
        if not isinstance(entry, Mapping):
            continue
        indicator = entry.get("indicator")
        target = entry.get("target")
        if indicator in (None, "") or target is None:
            continue
        name = entry.get("name") or indicator
        definition = SLODefinition(
            name=str(name),
            indicator=str(indicator),
            target=float(target),
            comparison=str(entry.get("comparison", ">=")),
            warning_threshold=(
                float(entry["warning_threshold"])
                if entry.get("warning_threshold") is not None
                else None
            ),
            severity=str(entry.get("severity", "critical")),
            description=(
                str(entry.get("description"))
                if entry.get("description") not in (None, "")
                else None
            ),
            tags=tuple(str(tag) for tag in (entry.get("tags") or ())),
        )
        definitions.append(definition)

    composites: list[SLOCompositeDefinition] = []
    composite_entries: Iterable[Any] | None = None
    if isinstance(data, Mapping):
        composite_entries = (
            data.get("composites")
            or data.get("slo2")
            or data.get("aggregates")
            or data.get("composite_definitions")
        )
    if composite_entries is not None:
        for entry in _coerce_sequence(composite_entries):
            if not isinstance(entry, Mapping):
                continue
            objectives_value = entry.get("objectives") or entry.get("slos")
            objectives = [str(item) for item in _coerce_sequence(objectives_value) if item]
            if not objectives:
                continue
            max_warnings = entry.get("max_warnings")
            min_ok_ratio = entry.get("min_ok_ratio")
            composites.append(
                SLOCompositeDefinition(
                    name=str(entry.get("name") or "composite"),
                    objectives=tuple(objectives),
                    max_breaches=int(entry.get("max_breaches") or 0),
                    max_warnings=(int(max_warnings) if max_warnings is not None else None),
                    min_ok_ratio=(float(min_ok_ratio) if min_ok_ratio is not None else None),
                    severity=str(entry.get("severity", "critical")),
                    description=(
                        str(entry.get("description"))
                        if entry.get("description") not in (None, "")
                        else None
                    ),
                    tags=tuple(str(tag) for tag in (entry.get("tags") or ())),
                )
            )

    return definitions, composites


def load_slo_measurements(path: Path) -> dict[str, SLOMeasurement]:
    """Ładuje pomiary wskaźników dla monitoringu SLO."""

    data = _load_yaml_or_json(path)
    measurements: dict[str, SLOMeasurement] = {}
    if isinstance(data, Mapping):
        if all(isinstance(value, Mapping) for value in data.values()):
            iterable: Iterable[Any] = data.values()
        else:
            iterable = [data]
    elif isinstance(data, Iterable) and not isinstance(data, (str, bytes)):
        iterable = data
    else:
        iterable = []

    for entry in iterable:
        if not isinstance(entry, Mapping):
            continue
        indicator = entry.get("indicator") or entry.get("name")
        if indicator in (None, ""):
            continue
        metadata = entry.get("metadata") if isinstance(entry.get("metadata"), Mapping) else {}
        metadata_map = {
            str(key): float(value)
            for key, value in metadata.items()
            if isinstance(value, (int, float))
        }
        measurements[str(indicator)] = SLOMeasurement(
            indicator=str(indicator),
            value=(float(entry.get("value")) if entry.get("value") is not None else None),
            window_start=_parse_datetime(entry.get("window_start") or entry.get("start")),
            window_end=_parse_datetime(entry.get("window_end") or entry.get("end")),
            sample_size=int(entry.get("sample_size") or entry.get("count") or 0),
            metadata=metadata_map,
        )

    return measurements


__all__ = [
    "load_slo_definitions",
    "load_slo_measurements",
]


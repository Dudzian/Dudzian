"""Publiczny interfejs katalogu strategii współdzielony między backendem a UI."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from bot_core.strategies.catalog import DEFAULT_STRATEGY_CATALOG, StrategyCatalog


@dataclass(frozen=True, slots=True)
class StrategyDescriptor:
    """Minimalny opis strategii prezentowany w interfejsach użytkownika."""

    name: str
    engine: str
    title: str
    license_tier: str
    risk_classes: tuple[str, ...]
    required_data: tuple[str, ...]
    tags: tuple[str, ...]
    metadata: Mapping[str, Any]


def _guess_title(value: str) -> str:
    normalized = value.replace("_", " ").strip()
    if not normalized:
        return value
    return normalized.title()


def _normalize_payload(payload: Mapping[str, Any]) -> StrategyDescriptor | None:
    engine = str(payload.get("engine", "")).strip()
    if not engine:
        return None

    license_tier = str(payload.get("license_tier", "")).strip()
    risk_classes = tuple(
        str(item).strip() for item in payload.get("risk_classes", ()) if str(item).strip()
    )
    required_data = tuple(
        str(item).strip() for item in payload.get("required_data", ()) if str(item).strip()
    )
    tags_source = payload.get("default_tags") or payload.get("tags") or ()
    tags = tuple(str(item).strip() for item in tags_source if str(item).strip())

    metadata = {
        key: value
        for key, value in payload.items()
        if key
        not in {
            "engine",
            "default_tags",
            "license_tier",
            "risk_classes",
            "required_data",
        }
    }

    return StrategyDescriptor(
        name=engine,
        engine=engine,
        title=_guess_title(engine),
        license_tier=license_tier,
        risk_classes=risk_classes,
        required_data=required_data,
        tags=tags,
        metadata=metadata,
    )


def list_available_strategies(
    *, catalog: StrategyCatalog | None = None, limit: int | None = None
) -> tuple[StrategyDescriptor, ...]:
    """Zwraca listę strategii dostępnych w katalogu wraz z metadanymi."""

    source = catalog or DEFAULT_STRATEGY_CATALOG

    entries: list[StrategyDescriptor] = []
    for payload in source.describe_engines():
        descriptor = _normalize_payload(payload)
        if descriptor is None:
            continue
        entries.append(descriptor)

    if not entries:
        fallback = StrategyDescriptor(
            name="demo_grid",
            engine="grid",
            title="Demo Grid",
            license_tier="standard",
            risk_classes=("grid", "market_neutral"),
            required_data=("ohlcv",),
            tags=("demo", "grid"),
            metadata={},
        )
        entries.append(fallback)

    if limit is not None:
        entries = entries[: max(limit, 0)]

    return tuple(entries)


__all__ = ["StrategyDescriptor", "list_available_strategies"]

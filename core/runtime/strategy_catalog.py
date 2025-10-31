"""Udostępnia uproszczony dostęp do katalogu strategii dla modułów UI."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import importlib
import importlib.util


@dataclass(frozen=True, slots=True)
class StrategyDescriptor:
    """Minimalny opis strategii prezentowany w kreatorze onboardingu."""

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


def _resolve_catalog_entries() -> Sequence[Mapping[str, Any]]:
    spec = importlib.util.find_spec("bot_core.strategies.catalog")
    if spec is None:
        return ()
    module = importlib.import_module("bot_core.strategies.catalog")
    catalog = getattr(module, "DEFAULT_STRATEGY_CATALOG", None)
    if catalog is None:
        return ()
    try:
        return tuple(catalog.describe_engines())
    except Exception:  # pragma: no cover - defensywne zabezpieczenie
        return ()


def list_available_strategies(*, limit: int | None = None) -> tuple[StrategyDescriptor, ...]:
    """Zwraca dostępne strategie wraz z metadanymi potrzebnymi w UI."""

    entries: list[StrategyDescriptor] = []
    for payload in _resolve_catalog_entries():
        engine = str(payload.get("engine", "")).strip()
        if not engine:
            continue
        license_tier = str(payload.get("license_tier", "")).strip()
        risk_classes = tuple(str(item).strip() for item in payload.get("risk_classes", ()) if str(item).strip())
        required_data = tuple(str(item).strip() for item in payload.get("required_data", ()) if str(item).strip())
        tags_source = payload.get("default_tags") or payload.get("tags") or ()
        tags = tuple(str(item).strip() for item in tags_source if str(item).strip())
        metadata = {
            key: value
            for key, value in payload.items()
            if key not in {"engine", "default_tags", "license_tier", "risk_classes", "required_data"}
        }
        descriptor = StrategyDescriptor(
            name=engine,
            engine=engine,
            title=_guess_title(engine),
            license_tier=license_tier,
            risk_classes=risk_classes,
            required_data=required_data,
            tags=tags,
            metadata=metadata,
        )
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

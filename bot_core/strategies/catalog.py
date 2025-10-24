"""Katalog strategii i wspólne interfejsy fabryk."""
from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Protocol, Sequence

from bot_core.security.guards import get_capability_guard
from bot_core.security.signing import build_hmac_signature

from .base import StrategyEngine
from .cross_exchange_arbitrage import (
    CrossExchangeArbitrageSettings,
    CrossExchangeArbitrageStrategy,
)
from .daily_trend import DailyTrendMomentumSettings, DailyTrendMomentumStrategy
from .grid import GridTradingSettings, GridTradingStrategy
from .mean_reversion import MeanReversionSettings, MeanReversionStrategy
from .volatility_target import VolatilityTargetSettings, VolatilityTargetStrategy


class StrategyFactory(Protocol):
    """Fabryka budująca `StrategyEngine` na podstawie parametrów."""

    def __call__(
        self,
        *,
        name: str,
        parameters: Mapping[str, Any],
        metadata: Mapping[str, Any] | None = None,
    ) -> StrategyEngine:
        ...


def _normalize_non_empty_str(value: str, *, field_name: str) -> str:
    value = str(value).strip()
    if not value:
        raise ValueError(f"{field_name} is required")
    return value


def _normalize_str_sequence(values: Iterable[str], *, field_name: str) -> tuple[str, ...]:
    normalized = tuple(dict.fromkeys(str(item).strip() for item in values if str(item).strip()))
    if not normalized:
        raise ValueError(f"{field_name} must define at least one entry")
    return normalized


def _normalize_optional_str_sequence(values: Any) -> tuple[str, ...]:
    if values in (None, ""):
        return ()
    if isinstance(values, str):
        candidates = (values,)
    elif isinstance(values, Iterable):
        candidates = tuple(values)
    else:
        raise TypeError("Expected iterable of strings or string")
    return tuple(dict.fromkeys(str(item).strip() for item in candidates if str(item).strip()))


@dataclass(slots=True)
class StrategyDefinition:
    """Opis pojedynczej strategii dostępnej w katalogu."""

    name: str
    engine: str
    license_tier: str
    risk_classes: Sequence[str]
    required_data: Sequence[str]
    parameters: Mapping[str, Any] = field(default_factory=dict)
    risk_profile: str | None = None
    tags: Sequence[str] = field(default_factory=tuple)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _normalize_non_empty_str(self.name, field_name="name"))
        object.__setattr__(self, "engine", _normalize_non_empty_str(self.engine, field_name="engine"))
        object.__setattr__(self, "license_tier", _normalize_non_empty_str(self.license_tier, field_name="license_tier"))
        object.__setattr__(self, "risk_classes", _normalize_str_sequence(self.risk_classes, field_name="risk_classes"))
        object.__setattr__(self, "required_data", _normalize_str_sequence(self.required_data, field_name="required_data"))
        if self.tags:
            object.__setattr__(self, "tags", tuple(dict.fromkeys(str(tag).strip() for tag in self.tags if str(tag).strip())))
        if isinstance(self.metadata, Mapping):
            object.__setattr__(self, "metadata", dict(self.metadata))
        else:
            raise TypeError("metadata must be a mapping")


@dataclass(slots=True)
class StrategyEngineSpec:
    """Opis silnika strategii wraz z wymaganą licencją."""

    key: str
    factory: StrategyFactory
    license_tier: str
    risk_classes: Sequence[str]
    required_data: Sequence[str]
    capability: str | None = None
    default_tags: Sequence[str] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        object.__setattr__(self, "key", _normalize_non_empty_str(self.key, field_name="key"))
        object.__setattr__(self, "license_tier", _normalize_non_empty_str(self.license_tier, field_name="license_tier"))
        object.__setattr__(self, "risk_classes", _normalize_str_sequence(self.risk_classes, field_name="risk_classes"))
        object.__setattr__(self, "required_data", _normalize_str_sequence(self.required_data, field_name="required_data"))
        if self.default_tags:
            object.__setattr__(self, "default_tags", tuple(dict.fromkeys(str(tag).strip() for tag in self.default_tags if str(tag).strip())))

    def build(
        self,
        *,
        name: str,
        parameters: Mapping[str, Any],
        metadata: Mapping[str, Any] | None = None,
    ) -> StrategyEngine:
        return self.factory(name=name, parameters=parameters, metadata=metadata)


class StrategyCatalog:
    """Rejestr zarejestrowanych silników strategii."""

    def __init__(self) -> None:
        self._registry: MutableMapping[str, StrategyEngineSpec] = {}

    def register(self, spec: StrategyEngineSpec) -> None:
        key = spec.key.lower()
        self._registry[key] = spec

    def get(self, engine: str) -> StrategyEngineSpec:
        key = engine.lower()
        if key not in self._registry:
            raise KeyError(f"Nie znaleziono silnika strategii: {engine}")
        return self._registry[key]

    def create(self, definition: StrategyDefinition) -> StrategyEngine:
        spec = self.get(definition.engine)
        if definition.license_tier != spec.license_tier:
            raise ValueError(
                f"Strategy '{definition.name}' requires license tier '{definition.license_tier}' "
                f"but engine '{spec.key}' is registered for '{spec.license_tier}'"
            )
        tags = tuple(dict.fromkeys((*spec.default_tags, *definition.tags)))
        risk_classes = tuple(dict.fromkeys((*spec.risk_classes, *definition.risk_classes)))
        required_data = tuple(dict.fromkeys((*spec.required_data, *definition.required_data)))
        metadata = dict(definition.metadata)
        if tags and "tags" not in metadata:
            metadata["tags"] = tags
        if spec.capability and "capability" not in metadata:
            metadata["capability"] = spec.capability
        metadata.setdefault("license_tier", spec.license_tier)
        metadata.setdefault("risk_classes", risk_classes)
        metadata.setdefault("required_data", required_data)
        engine = spec.build(
            name=definition.name,
            parameters=definition.parameters,
            metadata=metadata,
        )
        try:
            setattr(engine, "metadata", dict(metadata))
        except Exception:
            # Nie wszystkie strategie muszą wspierać przypięcie metadanych.
            pass
        return engine

    def merge_tags(self, definition: StrategyDefinition) -> tuple[str, ...]:
        """Zwraca połączone tagi katalogu oraz definicji strategii."""

        spec = self.get(definition.engine)
        return tuple(dict.fromkeys((*spec.default_tags, *definition.tags)))

    def describe_engines(self) -> Sequence[Mapping[str, object]]:
        """Buduje listę zarejestrowanych silników wraz z metadanymi."""

        summary: list[Mapping[str, object]] = []
        for key in sorted(self._registry):
            spec = self._registry[key]
            payload: dict[str, object] = {
                "engine": spec.key,
                "capability": spec.capability,
                "default_tags": list(spec.default_tags),
                "license_tier": spec.license_tier,
                "risk_classes": list(spec.risk_classes),
                "required_data": list(spec.required_data),
            }
            summary.append(payload)
        return summary

    def describe_definitions(
        self,
        definitions: Mapping[str, StrategyDefinition],
        *,
        include_metadata: bool = False,
    ) -> Sequence[Mapping[str, object]]:
        """Tworzy opis strategii na podstawie przekazanych definicji."""

        summary: list[Mapping[str, object]] = []
        for name in sorted(definitions):
            definition = definitions[name]
            try:
                merged_tags = self.merge_tags(definition)
            except KeyError:
                merged_tags = tuple(dict.fromkeys(definition.tags))
            payload: dict[str, object] = {
                "name": name,
                "engine": definition.engine,
                "tags": list(merged_tags),
                "license_tier": definition.license_tier,
                "risk_classes": list(definition.risk_classes),
                "required_data": list(definition.required_data),
            }
            if definition.risk_profile:
                payload["risk_profile"] = definition.risk_profile
            try:
                spec = self.get(definition.engine)
                if spec.capability:
                    payload["capability"] = spec.capability
                payload["license_tier"] = spec.license_tier
                payload["risk_classes"] = list(
                    dict.fromkeys((*spec.risk_classes, *definition.risk_classes))
                )
                payload["required_data"] = list(
                    dict.fromkeys((*spec.required_data, *definition.required_data))
                )
            except KeyError:
                pass
            if "capability" not in payload and "capability" in definition.metadata:
                payload["capability"] = definition.metadata["capability"]
            if include_metadata and definition.metadata:
                payload["metadata"] = dict(definition.metadata)
            if definition.parameters:
                payload["parameters"] = dict(definition.parameters)
            summary.append(payload)
        return summary


def _build_daily_trend_strategy(
    *, name: str, parameters: Mapping[str, Any], metadata: Mapping[str, Any] | None = None
) -> StrategyEngine:
    settings = DailyTrendMomentumSettings(
        fast_ma=int(parameters.get("fast_ma", 20)),
        slow_ma=int(parameters.get("slow_ma", 100)),
        breakout_lookback=int(parameters.get("breakout_lookback", 55)),
        momentum_window=int(parameters.get("momentum_window", 20)),
        atr_window=int(parameters.get("atr_window", 14)),
        atr_multiplier=float(parameters.get("atr_multiplier", 2.0)),
        min_trend_strength=float(parameters.get("min_trend_strength", 0.005)),
        min_momentum=float(parameters.get("min_momentum", 0.0)),
    )
    return DailyTrendMomentumStrategy(settings)


def _build_mean_reversion_strategy(
    *, name: str, parameters: Mapping[str, Any], metadata: Mapping[str, Any] | None = None
) -> StrategyEngine:
    settings = MeanReversionSettings(
        lookback=int(parameters.get("lookback", 96)),
        entry_zscore=float(parameters.get("entry_zscore", 2.0)),
        exit_zscore=float(parameters.get("exit_zscore", 0.5)),
        max_holding_period=int(parameters.get("max_holding_period", 12)),
        volatility_cap=float(parameters.get("volatility_cap", 0.04)),
        min_volume_usd=float(parameters.get("min_volume_usd", 1000.0)),
    )
    return MeanReversionStrategy(settings)


def _build_grid_strategy(
    *, name: str, parameters: Mapping[str, Any], metadata: Mapping[str, Any] | None = None
) -> StrategyEngine:
    settings = GridTradingSettings(
        grid_size=int(parameters.get("grid_size", 5)),
        grid_spacing=float(parameters.get("grid_spacing", 0.004)),
        rebalance_threshold=float(parameters.get("rebalance_threshold", 0.001)),
        max_inventory=float(parameters.get("max_inventory", 1.0)),
    )
    return GridTradingStrategy(settings)


def _build_volatility_target_strategy(
    *, name: str, parameters: Mapping[str, Any], metadata: Mapping[str, Any] | None = None
) -> StrategyEngine:
    settings = VolatilityTargetSettings(
        target_volatility=float(parameters.get("target_volatility", 0.1)),
        lookback=int(parameters.get("lookback", 60)),
        rebalance_threshold=float(parameters.get("rebalance_threshold", 0.1)),
        min_allocation=float(parameters.get("min_allocation", 0.1)),
        max_allocation=float(parameters.get("max_allocation", 1.0)),
        floor_volatility=float(parameters.get("floor_volatility", 0.02)),
    )
    return VolatilityTargetStrategy(settings)


def _build_cross_exchange_strategy(
    *, name: str, parameters: Mapping[str, Any], metadata: Mapping[str, Any] | None = None
) -> StrategyEngine:
    settings = CrossExchangeArbitrageSettings(
        primary_exchange=str(parameters.get("primary_exchange", "")),
        secondary_exchange=str(parameters.get("secondary_exchange", "")),
        spread_entry=float(parameters.get("spread_entry", 0.0015)),
        spread_exit=float(parameters.get("spread_exit", 0.0005)),
        max_notional=float(parameters.get("max_notional", 50_000.0)),
        max_open_seconds=int(parameters.get("max_open_seconds", 120)),
    )
    return CrossExchangeArbitrageStrategy(settings)


def build_default_catalog() -> StrategyCatalog:
    catalog = StrategyCatalog()
    catalog.register(
        StrategyEngineSpec(
            key="daily_trend_momentum",
            factory=_build_daily_trend_strategy,
            license_tier="standard",
            risk_classes=("directional", "momentum"),
            required_data=("ohlcv", "technical_indicators"),
            capability="trend_d1",
            default_tags=("trend", "momentum"),
        )
    )
    catalog.register(
        StrategyEngineSpec(
            key="mean_reversion",
            factory=_build_mean_reversion_strategy,
            license_tier="professional",
            risk_classes=("statistical", "mean_reversion"),
            required_data=("ohlcv", "spread_history"),
            capability="mean_reversion",
            default_tags=("mean_reversion", "stat_arbitrage"),
        )
    )
    catalog.register(
        StrategyEngineSpec(
            key="grid_trading",
            factory=_build_grid_strategy,
            license_tier="professional",
            risk_classes=("market_making",),
            required_data=("order_book", "ohlcv"),
            capability="grid_trading",
            default_tags=("grid", "market_making"),
        )
    )
    catalog.register(
        StrategyEngineSpec(
            key="volatility_target",
            factory=_build_volatility_target_strategy,
            license_tier="enterprise",
            risk_classes=("risk_control", "volatility"),
            required_data=("ohlcv", "realized_volatility"),
            capability="volatility_target",
            default_tags=("volatility", "risk"),
        )
    )
    catalog.register(
        StrategyEngineSpec(
            key="cross_exchange_arbitrage",
            factory=_build_cross_exchange_strategy,
            license_tier="enterprise",
            risk_classes=("arbitrage", "liquidity"),
            required_data=("order_book", "latency_monitoring"),
            capability="cross_exchange",
            default_tags=("arbitrage", "liquidity"),
        )
    )
    return catalog


DEFAULT_STRATEGY_CATALOG = build_default_catalog()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


class StrategyPresetWizard:
    """Buduje i podpisuje presety strategii na podstawie katalogu."""

    def __init__(self, catalog: StrategyCatalog | None = None) -> None:
        self._catalog = catalog or DEFAULT_STRATEGY_CATALOG

    def build_preset(
        self,
        name: str,
        entries: Sequence[Mapping[str, Any]],
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any]:
        if not name:
            raise ValueError("Preset name is required")
        if not entries:
            raise ValueError("At least one strategy entry is required")

        strategies: list[dict[str, Any]] = []
        for entry in entries:
            strategies.append(self._build_entry(entry))

        payload: dict[str, Any] = {
            "name": name,
            "created_at": _now_iso(),
            "strategies": strategies,
        }
        if metadata:
            payload["metadata"] = dict(metadata)
        return payload

    def _build_entry(self, entry: Mapping[str, Any]) -> dict[str, Any]:
        engine_name = str(entry.get("engine") or "").strip()
        if not engine_name:
            raise ValueError("Preset entry must define an engine")
        spec = self._catalog.get(engine_name)

        name = str(entry.get("name") or spec.key)
        parameters = dict(entry.get("parameters") or {})
        risk_profile = entry.get("risk_profile")
        user_tags = tuple(entry.get("tags") or ())
        merged_tags = tuple(dict.fromkeys((*spec.default_tags, *user_tags)))
        license_tier = str(entry.get("license_tier") or spec.license_tier).strip()
        if license_tier != spec.license_tier:
            raise ValueError(
                f"Preset entry '{name}' declares license tier '{license_tier}' incompatible with engine '{spec.key}'"
            )
        risk_classes = tuple(
            dict.fromkeys((*spec.risk_classes, *_normalize_optional_str_sequence(entry.get("risk_classes"))))
        )
        required_data = tuple(
            dict.fromkeys((*spec.required_data, *_normalize_optional_str_sequence(entry.get("required_data"))))
        )
        metadata = dict(entry.get("metadata") or {})
        metadata.setdefault("license_tier", spec.license_tier)
        metadata.setdefault("risk_classes", risk_classes)
        metadata.setdefault("required_data", required_data)
        if spec.capability:
            metadata.setdefault("capability", spec.capability)

        payload: dict[str, Any] = {
            "name": name,
            "engine": spec.key,
            "parameters": parameters,
            "tags": list(merged_tags),
            "license_tier": spec.license_tier,
            "risk_classes": list(risk_classes),
            "required_data": list(required_data),
        }
        if risk_profile:
            payload["risk_profile"] = str(risk_profile)
        if spec.capability:
            payload["capability"] = spec.capability
        if metadata:
            payload["metadata"] = metadata
        return payload

    def build_document(
        self,
        preset: Mapping[str, Any],
        *,
        signing_key: bytes,
        key_id: str | None = None,
        algorithm: str = "HMAC-SHA256",
    ) -> Mapping[str, Any]:
        if not isinstance(signing_key, (bytes, bytearray)):
            raise TypeError("signing_key must be raw bytes")
        preset_payload = dict(preset)
        signature = build_hmac_signature(preset_payload, key=bytes(signing_key), key_id=key_id, algorithm=algorithm)
        return {"preset": preset_payload, "signature": signature}

    def export_signed(
        self,
        preset: Mapping[str, Any],
        *,
        signing_key: bytes,
        path: str | Path,
        key_id: str | None = None,
        algorithm: str = "HMAC-SHA256",
    ) -> Path:
        document = self.build_document(preset, signing_key=signing_key, key_id=key_id, algorithm=algorithm)
        destination = Path(path).expanduser()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return destination


__all__ = [
    "StrategyCatalog",
    "StrategyDefinition",
    "StrategyEngineSpec",
    "StrategyFactory",
    "DEFAULT_STRATEGY_CATALOG",
    "StrategyPresetWizard",
    "build_default_catalog",
]

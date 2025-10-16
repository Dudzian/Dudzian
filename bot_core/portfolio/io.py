"""Pomocnicze funkcje IO dla modułów portfelowych Stage6."""
from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from bot_core.market_intel import MarketIntelSnapshot
from bot_core.observability.slo import SLOStatus
from bot_core.risk import StressOverrideRecommendation

__all__ = [
    "load_json_or_yaml",
    "load_allocations_file",
    "parse_market_intel_payload",
    "load_market_intel_report",
    "parse_slo_status_payload",
    "parse_stress_overrides_payload",
    "resolve_decision_log_config",
]


def load_json_or_yaml(path: Path) -> Any:
    """Wczytuje plik JSON lub YAML, zwracając strukturę Python."""

    text = path.read_text(encoding="utf-8")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return yaml.safe_load(text)


def _parse_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


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


def load_allocations_file(path: Path) -> dict[str, float]:
    """Ładuje plik z alokacjami (JSON/YAML) i zwraca mapę symbol -> waga."""

    payload = load_json_or_yaml(path)
    if not isinstance(payload, Mapping):
        raise ValueError("Plik z alokacjami musi zawierać mapę symbol -> waga")
    allocations: dict[str, float] = {}
    for symbol, weight in payload.items():
        try:
            allocations[str(symbol)] = float(weight)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Niepoprawna waga dla {symbol}: {weight}") from exc
    return allocations


def parse_market_intel_payload(data: Mapping[str, Any]) -> dict[str, MarketIntelSnapshot]:
    """Konwertuje strukturę JSON/YAML na snapshoty Market Intel."""

    snapshots_section = data.get("snapshots") if isinstance(data.get("snapshots"), Mapping) else None
    payload = snapshots_section or data
    if not isinstance(payload, Mapping):
        raise ValueError("Raport Market Intel musi zawierać mapę snapshots")

    interval_default = str(data.get("interval") or "1h")
    result: dict[str, MarketIntelSnapshot] = {}
    for symbol, entry in payload.items():
        if not isinstance(entry, Mapping):
            continue
        interval = str(entry.get("interval") or interval_default)
        snapshot = MarketIntelSnapshot(
            symbol=str(symbol),
            interval=interval,
            start=_parse_datetime(entry.get("start")),
            end=_parse_datetime(entry.get("end")),
            bar_count=int(entry.get("bar_count", 0) or 0),
            price_change_pct=_parse_float(entry.get("price_change_pct")),
            volatility_pct=_parse_float(entry.get("volatility_pct")),
            max_drawdown_pct=_parse_float(entry.get("max_drawdown_pct")),
            average_volume=_parse_float(entry.get("average_volume")),
            liquidity_usd=_parse_float(entry.get("liquidity_usd")),
            momentum_score=_parse_float(entry.get("momentum_score")),
            metadata={
                str(key): float(value)
                for key, value in (entry.get("metadata", {}) or {}).items()
                if isinstance(value, (int, float))
            },
        )
        result[snapshot.symbol] = snapshot
    if not result:
        raise ValueError("Raport Market Intel nie zawiera żadnych snapshotów")
    return result


def load_market_intel_report(path: Path) -> tuple[dict[str, MarketIntelSnapshot], dict[str, Any]]:
    """Wczytuje raport Market Intel z pliku i zwraca snapshoty oraz metadane."""

    payload = load_json_or_yaml(path)
    if not isinstance(payload, Mapping):
        raise ValueError("Raport Market Intel musi zawierać strukturę mapy")
    snapshots = parse_market_intel_payload(payload)
    metadata: dict[str, Any] = {}
    generated_at = _parse_datetime(payload.get("generated_at"))
    if generated_at is not None:
        metadata["generated_at"] = generated_at
    for key in ("environment", "governor", "interval", "lookback_bars", "symbols"):
        if key in payload:
            metadata[key] = payload[key]
    return snapshots, metadata


def parse_slo_status_payload(data: Any) -> dict[str, SLOStatus]:
    """Konwertuje strukturę JSON/YAML na mapę statusów SLO."""

    if isinstance(data, Mapping):
        entries = data.get("results")
        if isinstance(entries, Mapping):
            data = entries
    if not isinstance(data, Mapping):
        return {}

    statuses: dict[str, SLOStatus] = {}
    for name, entry in data.items():
        if not isinstance(entry, Mapping):
            continue
        target = _parse_float(entry.get("target"))
        if target is None:
            continue
        statuses[str(name)] = SLOStatus(
            name=str(name),
            indicator=str(entry.get("indicator") or name),
            value=_parse_float(entry.get("value")),
            target=float(target),
            comparison=str(entry.get("comparison") or "<="),
            status=str(entry.get("status") or "unknown"),
            severity=str(entry.get("severity") or "warning"),
            warning_threshold=_parse_float(entry.get("warning_threshold")),
            error_budget_pct=_parse_float(entry.get("error_budget_pct")),
            window_start=_parse_datetime(entry.get("window_start")),
            window_end=_parse_datetime(entry.get("window_end")),
            sample_size=int(entry.get("sample_size", 0) or 0),
            reason=str(entry.get("reason")) if entry.get("reason") not in (None, "") else None,
            metadata={
                str(key): float(value)
                for key, value in (entry.get("metadata", {}) or {}).items()
                if isinstance(value, (int, float))
            },
        )
    return statuses


def parse_stress_overrides_payload(data: Any) -> list[StressOverrideRecommendation]:
    """Konwertuje strukturę JSON/YAML na listę override'ów Stress Lab."""

    if isinstance(data, Mapping):
        entries = data.get("overrides")
        if isinstance(entries, Sequence):
            data = entries
    if not isinstance(data, Sequence):
        return []

    overrides: list[StressOverrideRecommendation] = []
    for entry in data:
        if not isinstance(entry, Mapping):
            continue
        raw_tags = entry.get("tags")
        tags: tuple[str, ...]
        if isinstance(raw_tags, Sequence) and not isinstance(raw_tags, (str, bytes)):
            tags = tuple(str(tag) for tag in raw_tags if isinstance(tag, (str, int, float)))
        else:
            tags = ()
        overrides.append(
            StressOverrideRecommendation(
                severity=str(entry.get("severity") or "warning"),
                reason=str(entry.get("reason") or "stress_override"),
                symbol=str(entry.get("symbol")) if entry.get("symbol") else None,
                risk_budget=str(entry.get("risk_budget")) if entry.get("risk_budget") else None,
                weight_multiplier=_parse_float(entry.get("weight_multiplier")),
                min_weight=_parse_float(entry.get("min_weight")),
                max_weight=_parse_float(entry.get("max_weight")),
                tags=tags,
                force_rebalance=bool(entry.get("force_rebalance", False)),
            )
        )
    return overrides


def resolve_decision_log_config(core_config: Any) -> tuple[Path | None, dict[str, Any]]:
    """Zwraca ścieżkę i parametry decision logu portfelowego z konfiguracji."""

    config = getattr(core_config, "portfolio_decision_log", None)
    if config is None or not getattr(config, "enabled", True):
        return None, {}

    kwargs: dict[str, Any] = {
        "max_entries": int(getattr(config, "max_entries", 512) or 512),
        "jsonl_fsync": bool(getattr(config, "jsonl_fsync", False)),
    }
    path_value = getattr(config, "path", None)
    path = Path(path_value) if path_value else None

    key_value = getattr(config, "signing_key_value", None)
    key_env = getattr(config, "signing_key_env", None)
    key_path = getattr(config, "signing_key_path", None)
    key_id = getattr(config, "signing_key_id", None)

    key: bytes | None = None
    if isinstance(key_value, str) and key_value:
        key = key_value.encode("utf-8")
    elif isinstance(key_env, str) and key_env:
        env_value = os.environ.get(key_env)
        if env_value:
            key = env_value.encode("utf-8")
    elif key_path:
        candidate = Path(str(key_path))
        if candidate.exists():
            key = candidate.read_bytes().strip() or None

    if key is not None:
        kwargs["signing_key"] = key
    if key_id not in (None, ""):
        kwargs["signing_key_id"] = str(key_id)

    return path, kwargs

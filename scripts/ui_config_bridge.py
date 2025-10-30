#!/usr/bin/env python3
"""Mostek CLI do synchronizacji konfiguracji strategii/AI dla UI Qt."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from collections.abc import Iterable, Mapping as MappingABC
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

import yaml

from bot_core.ai import MarketRegime, MarketRegimeAssessment, RegimeHistory

try:
    from bot_core.config.loader import load_core_config
except Exception as exc:  # pragma: no cover - import guard
    raise SystemExit(f"Nie można zaimportować bot_core.config.loader: {exc}") from exc

from bot_core.runtime.pipeline import describe_strategy_definitions
from bot_core.security.guards import get_capability_guard
from bot_core.strategies.catalog import (
    DEFAULT_STRATEGY_CATALOG,
    StrategyDefinition,
    StrategyPresetWizard,
)
from bot_core.strategies.regime_workflow import (
    PresetAvailability,
    PresetVersionInfo,
    RegimePresetActivation,
    StrategyRegimeWorkflow,
)


def _read_json_input(path: str | None) -> Any:
    if path and path != "-":
        data = Path(path).read_text(encoding="utf-8")
    else:
        data = sys.stdin.read()
    if not data.strip():
        raise SystemExit("Wejście JSON jest puste – brak danych do zapisania")
    try:
        return json.loads(data)
    except json.JSONDecodeError as exc:  # pragma: no cover - validated w testach integracyjnych
        raise SystemExit(f"Niepoprawny JSON: {exc}") from exc


def _dump_decision(raw: Mapping[str, Any]) -> dict[str, Any]:
    decision = raw.get("decision_engine") or {}
    orchestrator = decision.get("orchestrator") or {}
    profile_overrides = decision.get("profile_overrides") or {}

    def _normalize_thresholds(name: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        result: dict[str, Any] = {"profile": name}
        for key in (
            "max_cost_bps",
            "min_net_edge_bps",
            "max_daily_loss_pct",
            "max_drawdown_pct",
            "max_position_ratio",
            "max_open_positions",
            "max_latency_ms",
            "max_trade_notional",
        ):
            if key in payload:
                result[key] = payload[key]
        return result

    overrides = []
    if isinstance(profile_overrides, Mapping):
        for name, payload in profile_overrides.items():
            if isinstance(payload, Mapping):
                overrides.append(_normalize_thresholds(str(name), payload))

    return {
        "max_cost_bps": orchestrator.get("max_cost_bps"),
        "min_net_edge_bps": orchestrator.get("min_net_edge_bps"),
        "max_daily_loss_pct": orchestrator.get("max_daily_loss_pct"),
        "max_drawdown_pct": orchestrator.get("max_drawdown_pct"),
        "max_position_ratio": orchestrator.get("max_position_ratio"),
        "max_open_positions": orchestrator.get("max_open_positions"),
        "max_latency_ms": orchestrator.get("max_latency_ms"),
        "stress_tests": orchestrator.get("stress_tests"),
        "min_probability": decision.get("min_probability"),
        "require_cost_data": decision.get("require_cost_data"),
        "penalty_cost_bps": decision.get("penalty_cost_bps"),
        "profile_overrides": overrides,
    }


def _parse_iso_datetime(value: Any, *, field: str) -> datetime:
    if not isinstance(value, str):
        raise SystemExit(f"Pole {field} musi zawierać znacznik czasu ISO8601")
    candidate = value.strip()
    if not candidate:
        raise SystemExit(f"Pole {field} nie może być puste")
    if candidate.endswith("Z"):
        candidate = candidate[:-1] + "+00:00"
    try:
        timestamp = datetime.fromisoformat(candidate)
    except ValueError as exc:
        raise SystemExit(f"Pole {field} ma niepoprawny format czasu: {value!r}") from exc
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    return timestamp


def _normalize_regime_value(value: Any) -> MarketRegime | None:
    if value in (None, ""):
        return None
    try:
        return MarketRegime(str(value).lower())
    except ValueError as exc:
        raise SystemExit(f"Nieznany reżim rynku: {value!r}") from exc


def _build_version_info_from_payload(payload: Mapping[str, Any], *, field: str) -> PresetVersionInfo:
    if not isinstance(payload, MappingABC):
        raise SystemExit(f"Sekcja {field} musi być słownikiem")
    hash_value = payload.get("hash")
    if not isinstance(hash_value, str) or not hash_value.strip():
        raise SystemExit(f"Sekcja {field}.hash musi być niepustym ciągiem znaków")
    signature_payload = payload.get("signature") or {}
    if not isinstance(signature_payload, MappingABC):
        raise SystemExit(f"Sekcja {field}.signature musi być słownikiem")
    signature = {str(key): str(signature_payload[key]) for key in signature_payload.keys()}
    issued_at = _parse_iso_datetime(payload.get("issued_at"), field=f"{field}.issued_at")
    metadata_payload = payload.get("metadata") or {}
    if isinstance(metadata_payload, MappingABC):
        metadata = {str(key): metadata_payload[key] for key in metadata_payload.keys()}
    else:
        metadata = {}
    return PresetVersionInfo(
        hash=hash_value,
        signature=signature,
        issued_at=issued_at,
        metadata=metadata,
    )


def _build_assessment_from_payload(
    regime: MarketRegime,
    payload: Mapping[str, Any] | None,
) -> MarketRegimeAssessment:
    confidence = 1.0
    risk_score = 0.0
    metrics: dict[str, float] = {}
    symbol: str | None = None
    if isinstance(payload, MappingABC):
        candidate_regime = payload.get("regime")
        if candidate_regime not in (None, ""):
            regime = _normalize_regime_value(candidate_regime) or regime
        try:
            confidence = float(payload.get("confidence", confidence))
        except (TypeError, ValueError):
            raise SystemExit("Pole assessment.confidence musi być liczbą zmiennoprzecinkową")
        try:
            risk_score = float(payload.get("risk_score", risk_score))
        except (TypeError, ValueError):
            raise SystemExit("Pole assessment.risk_score musi być liczbą zmiennoprzecinkową")
        metrics_payload = payload.get("metrics")
        if isinstance(metrics_payload, MappingABC):
            for key, value in metrics_payload.items():
                try:
                    metrics[str(key)] = float(value)
                except (TypeError, ValueError):
                    continue
        raw_symbol = payload.get("symbol")
        if isinstance(raw_symbol, str) and raw_symbol.strip():
            symbol = raw_symbol.strip()
    return MarketRegimeAssessment(
        regime=regime,
        confidence=confidence,
        risk_score=risk_score,
        metrics=metrics,
        symbol=symbol,
    )


def _normalize_json_value(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_normalize_json_value(item) for item in value]
    if isinstance(value, MappingABC):
        return {str(key): _normalize_json_value(value[key]) for key in value.keys()}
    return str(value)


def _normalize_sequence_field(values: Any) -> tuple[str, ...]:
    if values in (None, "", ()):  # szybkie ścieżki
        return ()
    if isinstance(values, str):
        iterable = (values,)
    elif isinstance(values, MappingABC):
        iterable = values.values()
    elif isinstance(values, Iterable):
        iterable = values
    else:
        return ()
    cleaned: list[str] = []
    for item in iterable:
        text = str(item).strip()
        if not text:
            continue
        if text not in cleaned:
            cleaned.append(text)
    return tuple(cleaned)


def _capability_allowed(capability: str | None) -> bool:
    guard = get_capability_guard()
    if guard is None or not capability:
        return True
    try:
        return guard.capabilities.is_strategy_enabled(capability)
    except AttributeError:
        return True


def _collect_strategy_metadata(
    raw: Mapping[str, Any],
) -> tuple[dict[str, Mapping[str, Any]], dict[str, Mapping[str, Any]]]:
    definitions: dict[str, Mapping[str, Any]] = {}
    blocked: dict[str, Mapping[str, Any]] = {}

    def _register(name: str, engine: str, entry: Mapping[str, Any]) -> None:
        try:
            spec = DEFAULT_STRATEGY_CATALOG.get(engine)
        except KeyError:
            spec = None

        license_tier = str(entry.get("license_tier") or "").strip()
        risk_classes = _normalize_sequence_field(entry.get("risk_classes"))
        required_data = _normalize_sequence_field(entry.get("required_data"))
        tags = _normalize_sequence_field(entry.get("tags"))

        capability = str(entry.get("capability") or "").strip()
        if not capability and spec and spec.capability:
            capability = spec.capability

        merged_risk = tuple(
            dict.fromkeys(
                (*((spec.risk_classes) if spec else ()), *risk_classes)
            )
        )
        merged_data = tuple(
            dict.fromkeys(
                (*((spec.required_data) if spec else ()), *required_data)
            )
        )
        merged_tags = tuple(
            dict.fromkeys(
                (*((spec.default_tags) if spec else ()), *tags)
            )
        )

        payload: dict[str, Any] = {
            "engine": engine,
            "license_tier": license_tier or (spec.license_tier if spec else None),
            "risk_classes": merged_risk,
            "required_data": merged_data,
            "tags": merged_tags,
        }
        if capability:
            payload["capability"] = capability
        risk_profile = entry.get("risk_profile")
        if isinstance(risk_profile, str) and risk_profile.strip():
            payload["risk_profile"] = risk_profile.strip()
        if _capability_allowed(capability):
            definitions[name] = payload
        else:
            blocked[name] = payload

    strategies = raw.get("strategies") or {}
    if isinstance(strategies, MappingABC):
        for name, entry in strategies.items():
            if not isinstance(entry, MappingABC):
                continue
            engine = str(entry.get("engine") or "").strip()
            if not engine:
                continue
            _register(str(name), engine, entry)

    def _register_section(section: str, engine: str) -> None:
        payload = raw.get(section) or {}
        if not isinstance(payload, MappingABC):
            return
        for name, entry in payload.items():
            if str(name) in definitions:
                continue
            entry_mapping: Mapping[str, Any]
            if isinstance(entry, MappingABC):
                entry_mapping = entry
            else:
                entry_mapping = {}
            _register(str(name), engine, entry_mapping)

    _register_section("mean_reversion_strategies", "mean_reversion")
    _register_section("volatility_target_strategies", "volatility_target")
    _register_section("cross_exchange_arbitrage_strategies", "cross_exchange_arbitrage")
    _register_section("scalping_strategies", "scalping")
    _register_section("options_income_strategies", "options_income")
    _register_section("futures_spread_strategies", "futures_spread")
    _register_section("cross_exchange_hedge_strategies", "cross_exchange_hedge")
    _register_section("statistical_arbitrage_strategies", "statistical_arbitrage")
    _register_section("day_trading_strategies", "day_trading")
    _register_section("grid_strategies", "grid_trading")

    return definitions, blocked


def _ensure_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, MappingABC):
        return {str(key): value[key] for key in value.keys()}
    return {}


def _derive_regime_map(preset: Mapping[str, Any]) -> dict[str, list[str]]:
    regime_map: dict[str, list[str]] = {}
    strategies = preset.get("strategies") if isinstance(preset, MappingABC) else None
    if not isinstance(strategies, Iterable):
        return regime_map
    for entry in strategies:
        if not isinstance(entry, MappingABC):
            continue
        strategy_name = str(entry.get("name") or "").strip()
        if not strategy_name:
            continue
        profile = entry.get("risk_profile")
        if not profile:
            metadata = entry.get("metadata")
            if isinstance(metadata, MappingABC):
                profile = metadata.get("risk_profile")
        if not isinstance(profile, str):
            continue
        normalized_profile = profile.strip()
        if not normalized_profile:
            continue
        regime_map.setdefault(normalized_profile, []).append(strategy_name)
    return regime_map


def _build_definition_from_entry(entry: Mapping[str, Any]) -> StrategyDefinition:
    metadata = _ensure_mapping(entry.get("metadata"))
    tags = entry.get("tags")
    if isinstance(tags, Iterable) and not isinstance(tags, (str, bytes)):
        normalized_tags = tuple(str(item) for item in tags)
    else:
        normalized_tags = ()
    risk_classes = entry.get("risk_classes")
    if isinstance(risk_classes, Iterable) and not isinstance(risk_classes, (str, bytes)):
        normalized_risk = tuple(str(item) for item in risk_classes)
    else:
        normalized_risk = ()
    required_data = entry.get("required_data")
    if isinstance(required_data, Iterable) and not isinstance(required_data, (str, bytes)):
        normalized_data = tuple(str(item) for item in required_data)
    else:
        normalized_data = ()
    parameters = entry.get("parameters")
    if isinstance(parameters, MappingABC):
        normalized_parameters = dict(parameters)
    else:
        normalized_parameters = {}
    risk_profile = entry.get("risk_profile")
    if isinstance(risk_profile, str) and risk_profile.strip():
        normalized_profile: str | None = risk_profile.strip()
    else:
        normalized_profile = None
    return StrategyDefinition(
        name=str(entry.get("name") or ""),
        engine=str(entry.get("engine") or ""),
        license_tier=str(entry.get("license_tier") or metadata.get("license_tier") or ""),
        risk_classes=normalized_risk,
        required_data=normalized_data,
        tags=normalized_tags,
        parameters=normalized_parameters,
        metadata=metadata,
        risk_profile=normalized_profile,
    )


def _build_definition_summary_from_preset(preset: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    strategies = preset.get("strategies")
    if not isinstance(strategies, Iterable):
        return []
    definitions: dict[str, StrategyDefinition] = {}
    for entry in strategies:
        if not isinstance(entry, MappingABC):
            continue
        try:
            definition = _build_definition_from_entry(entry)
        except Exception:
            continue
        definitions[definition.name] = definition
    if not definitions:
        return []
    summary = DEFAULT_STRATEGY_CATALOG.describe_definitions(definitions, include_metadata=True)
    return list(summary)


def _validate_preset_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    issues: list[dict[str, Any]] = []

    name = str(payload.get("name") or "").strip()
    if not name:
        errors.append("Preset name is required")

    raw_entries = payload.get("entries")
    if not isinstance(raw_entries, list):
        errors.append("Preset entries must be provided as a list")
        return {"ok": False, "errors": errors}

    seen_names: set[str] = set()
    for idx, entry in enumerate(raw_entries):
        if not isinstance(entry, MappingABC):
            errors.append(f"Entry #{idx + 1} has invalid format")
            continue
        entry_name = str(entry.get("name") or "").strip()
        engine_name = str(entry.get("engine") or "").strip()
        entry_errors: list[str] = []
        if not entry_name:
            entry_errors.append("missing strategy name")
        elif entry_name in seen_names:
            entry_errors.append("duplicate strategy name")
        if not engine_name:
            entry_errors.append("missing engine key")
        else:
            try:
                spec = DEFAULT_STRATEGY_CATALOG.get(engine_name)
                if spec.capability:
                    issues.append(
                        {
                            "entry": entry_name or engine_name,
                            "field": "capability",
                            "severity": "info",
                            "message": f"Silnik '{engine_name}' wymaga aktywnej licencji {spec.capability}.",
                            "suggested": spec.capability,
                        }
                    )
            except KeyError:
                entry_errors.append(f"unknown engine '{engine_name}'")
        seen_names.add(entry_name or engine_name or f"entry-{idx + 1}")
        if entry_errors:
            errors.append(f"Entry #{idx + 1}: " + "; ".join(entry_errors))

    if errors:
        return {"ok": False, "errors": errors, "issues": issues}

    wizard = StrategyPresetWizard(DEFAULT_STRATEGY_CATALOG)
    metadata = _ensure_mapping(payload.get("metadata"))
    try:
        preset = wizard.build_preset(name, raw_entries, metadata=metadata)
    except Exception as exc:  # pragma: no cover - validated in integration tests
        errors.append(str(exc))
        return {"ok": False, "errors": errors, "issues": issues}

    summary = _build_definition_summary_from_preset(preset)
    regime_map = _derive_regime_map(preset)

    result: dict[str, Any] = {
        "ok": True,
        "preset": preset,
        "issues": issues,
    }
    if summary:
        result["definition_summary"] = summary
    if regime_map:
        result["regime_map"] = regime_map
    return result


def _serialize_version_info(version: PresetVersionInfo) -> dict[str, Any]:
    issued_at = version.issued_at.astimezone(timezone.utc)
    return {
        "hash": version.hash,
        "signature": {str(key): str(value) for key, value in version.signature.items()},
        "issued_at": issued_at.isoformat(),
        "metadata": _normalize_json_value(version.metadata),
    }


def _serialize_preset_availability(report: PresetAvailability) -> dict[str, Any]:
    regime_value = report.regime.value if isinstance(report.regime, MarketRegime) else None
    return {
        "regime": regime_value,
        "version": _serialize_version_info(report.version),
        "ready": bool(report.ready),
        "blocked_reason": report.blocked_reason,
        "missing_data": list(report.missing_data),
        "license_issues": list(report.license_issues),
        "schedule_blocked": bool(report.schedule_blocked),
    }


def _serialize_activation(activation: RegimePresetActivation) -> dict[str, Any]:
    regime_value = activation.regime.value if isinstance(activation.regime, MarketRegime) else None
    preset_regime_value = (
        activation.preset_regime.value if isinstance(activation.preset_regime, MarketRegime) else None
    )
    assessment_payload = {
        "regime": regime_value,
        "confidence": float(activation.assessment.confidence),
        "risk_score": float(activation.assessment.risk_score),
        "metrics": {str(key): float(value) for key, value in activation.assessment.metrics.items()},
        "symbol": activation.assessment.symbol,
    }
    return {
        "regime": regime_value,
        "preset_regime": preset_regime_value,
        "activated_at": activation.activated_at.astimezone(timezone.utc).isoformat(),
        "used_fallback": bool(activation.used_fallback),
        "blocked_reason": activation.blocked_reason,
        "missing_data": list(activation.missing_data),
        "license_issues": list(activation.license_issues),
        "recommendation": activation.recommendation,
        "version": _serialize_version_info(activation.version),
        "preset": _normalize_json_value(activation.preset),
        "assessment": assessment_payload,
    }


def _serialize_history_stats(stats: Any) -> dict[str, Any]:
    regime_counts = [
        {"regime": key.value if isinstance(key, MarketRegime) else None, "count": value}
        for key, value in stats.regime_counts.items()
    ]
    preset_regime_counts = [
        {"preset_regime": key.value if isinstance(key, MarketRegime) else None, "count": value}
        for key, value in stats.preset_regime_counts.items()
    ]
    blocked_reasons = [
        {"reason": key, "count": value}
        for key, value in stats.blocked_reasons.items()
    ]
    missing_data = [
        {"name": key, "count": value}
        for key, value in stats.missing_data.items()
    ]
    license_issues = [
        {"issue": key, "count": value}
        for key, value in stats.license_issue_counts.items()
    ]
    return {
        "total": stats.total,
        "fallback_count": stats.fallback_count,
        "regime_counts": regime_counts,
        "preset_regime_counts": preset_regime_counts,
        "blocked_reasons": blocked_reasons,
        "missing_data": missing_data,
        "license_issue_counts": license_issues,
    }


def _serialize_transition_stats(stats: Any) -> dict[str, Any]:
    regime_transitions = [
        {
            "from": start.value if isinstance(start, MarketRegime) else None,
            "to": end.value if isinstance(end, MarketRegime) else None,
            "count": value,
        }
        for (start, end), value in stats.regime_transitions.items()
    ]
    preset_transitions = [
        {
            "from": start.value if isinstance(start, MarketRegime) else None,
            "to": end.value if isinstance(end, MarketRegime) else None,
            "count": value,
        }
        for (start, end), value in stats.preset_regime_transitions.items()
    ]
    blocked_transitions = [
        {
            "from": start,
            "to": end,
            "count": value,
        }
        for (start, end), value in stats.blocked_transitions.items()
    ]
    return {
        "total": stats.total,
        "regime_transitions": regime_transitions,
        "preset_regime_transitions": preset_transitions,
        "blocked_transitions": blocked_transitions,
        "fallback_transitions": stats.fallback_transitions,
    }


def _serialize_cadence_stats(stats: Any) -> dict[str, Any]:
    def _delta_to_seconds(delta: Any) -> float | None:
        if delta is None:
            return None
        return float(delta.total_seconds())

    return {
        "total": stats.total,
        "intervals": stats.intervals,
        "min_interval_seconds": _delta_to_seconds(stats.min_interval),
        "max_interval_seconds": _delta_to_seconds(stats.max_interval),
        "mean_interval_seconds": _delta_to_seconds(stats.mean_interval),
        "median_interval_seconds": _delta_to_seconds(stats.median_interval),
        "last_interval_seconds": _delta_to_seconds(stats.last_interval),
    }


def _serialize_uptime_stats(stats: Any) -> dict[str, Any]:
    def _delta_to_seconds(delta: Any) -> float:
        return float(delta.total_seconds()) if delta is not None else 0.0

    regime_uptime = [
        {
            "regime": key.value if isinstance(key, MarketRegime) else None,
            "seconds": _delta_to_seconds(value),
        }
        for key, value in stats.regime_uptime.items()
    ]
    preset_uptime = [
        {
            "preset_regime": key.value if isinstance(key, MarketRegime) else None,
            "seconds": _delta_to_seconds(value),
        }
        for key, value in stats.preset_uptime.items()
    ]
    return {
        "total": stats.total,
        "duration_seconds": _delta_to_seconds(stats.duration),
        "regime_uptime": regime_uptime,
        "preset_uptime": preset_uptime,
        "fallback_seconds": _delta_to_seconds(stats.fallback_uptime),
    }


def _load_regime_workflow_availability(base_dir: Path) -> tuple[PresetAvailability, ...]:
    path = base_dir / "availability.json"
    if not path.exists():
        return ()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Nie można sparsować {path}: {exc}") from exc
    if not isinstance(payload, list):
        raise SystemExit(f"Plik {path} musi zawierać listę raportów dostępności")
    reports: list[PresetAvailability] = []
    for index, entry in enumerate(payload):
        if not isinstance(entry, MappingABC):
            raise SystemExit(f"Pozycja availability[{index}] ma niepoprawny format")
        regime = _normalize_regime_value(entry.get("regime"))
        version_payload = entry.get("version")
        version = _build_version_info_from_payload(
            version_payload,
            field=f"availability[{index}].version",
        )
        blocked_reason_raw = entry.get("blocked_reason")
        blocked_reason = None
        if isinstance(blocked_reason_raw, str) and blocked_reason_raw.strip():
            blocked_reason = blocked_reason_raw.strip()
        missing_data = _normalize_sequence_field(entry.get("missing_data"))
        license_issues = _normalize_sequence_field(entry.get("license_issues"))
        ready_value = entry.get("ready")
        ready = bool(ready_value) if ready_value is not None else False
        if isinstance(ready_value, bool):
            ready = ready_value
        schedule_value = entry.get("schedule_blocked")
        schedule_blocked = bool(schedule_value) if schedule_value is not None else False
        if isinstance(schedule_value, bool):
            schedule_blocked = schedule_value
        reports.append(
            PresetAvailability(
                regime=regime,
                version=version,
                ready=ready,
                blocked_reason=blocked_reason,
                missing_data=missing_data,
                license_issues=license_issues,
                schedule_blocked=schedule_blocked,
            )
        )
    return tuple(reports)


def _load_regime_workflow_history(base_dir: Path) -> tuple[RegimePresetActivation, ...]:
    path = base_dir / "activation_history.json"
    if not path.exists():
        return ()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Nie można sparsować {path}: {exc}") from exc
    if not isinstance(payload, list):
        raise SystemExit(f"Plik {path} musi zawierać listę aktywacji workflow")
    history: list[RegimePresetActivation] = []
    for index, entry in enumerate(payload):
        if not isinstance(entry, MappingABC):
            raise SystemExit(f"Pozycja activation_history[{index}] ma niepoprawny format")
        regime = _normalize_regime_value(entry.get("regime"))
        if regime is None:
            raise SystemExit(f"Pozycja activation_history[{index}] nie definiuje reżimu")
        preset_regime = _normalize_regime_value(entry.get("preset_regime"))
        activated_at = _parse_iso_datetime(
            entry.get("activated_at"),
            field=f"activation_history[{index}].activated_at",
        )
        version = _build_version_info_from_payload(
            entry.get("version"),
            field=f"activation_history[{index}].version",
        )
        assessment = _build_assessment_from_payload(regime, entry.get("assessment"))
        blocked_reason_raw = entry.get("blocked_reason")
        blocked_reason = None
        if isinstance(blocked_reason_raw, str) and blocked_reason_raw.strip():
            blocked_reason = blocked_reason_raw.strip()
        recommendation_raw = entry.get("recommendation")
        recommendation = None
        if isinstance(recommendation_raw, str) and recommendation_raw.strip():
            recommendation = recommendation_raw.strip()
        preset_payload = entry.get("preset")
        if isinstance(preset_payload, MappingABC):
            preset = {str(key): preset_payload[key] for key in preset_payload.keys()}
        else:
            preset = {}
        history.append(
            RegimePresetActivation(
                regime=regime,
                assessment=assessment,
                summary=None,
                preset=preset,
                version=version,
                decision_candidates=tuple(),
                activated_at=activated_at,
                preset_regime=preset_regime,
                used_fallback=bool(entry.get("used_fallback")),
                missing_data=_normalize_sequence_field(entry.get("missing_data")),
                blocked_reason=blocked_reason,
                recommendation=recommendation,
                license_issues=_normalize_sequence_field(entry.get("license_issues")),
            )
        )
    return tuple(history)


def _build_regime_workflow_stats(history: tuple[RegimePresetActivation, ...]) -> dict[str, Any]:
    history_limit = max(len(history), 50) if history else 50
    workflow = StrategyRegimeWorkflow(
        history=RegimeHistory(thresholds_loader=lambda: {}),
        activation_history_limit=history_limit,
    )
    workflow._activation_history.clear()
    workflow._activation_history.extend(history)
    history_stats = _serialize_history_stats(workflow.activation_history_stats())
    transition_stats = _serialize_transition_stats(workflow.activation_transition_stats())
    cadence_stats = _serialize_cadence_stats(workflow.activation_cadence_stats())
    uptime_stats = _serialize_uptime_stats(workflow.activation_uptime_stats())
    return {
        "history": history_stats,
        "transitions": transition_stats,
        "cadence": cadence_stats,
        "uptime": uptime_stats,
    }


def describe_regime_workflow(base_dir: Path) -> dict[str, Any]:
    if not base_dir.exists():
        raise SystemExit(f"Katalog z danymi StrategyRegimeWorkflow {base_dir} nie istnieje")
    if not base_dir.is_dir():
        raise SystemExit(f"Ścieżka {base_dir} nie jest katalogiem")
    availability = _load_regime_workflow_availability(base_dir)
    history = _load_regime_workflow_history(base_dir)
    stats = _build_regime_workflow_stats(history)
    return {
        "regime_workflow": {
            "availability": [_serialize_preset_availability(entry) for entry in availability],
            "history": {
                "activations": [_serialize_activation(entry) for entry in history],
                "stats": stats,
            },
        }
    }


def describe_catalog(config_path: Path, raw: Mapping[str, Any]) -> dict[str, Any]:
    try:
        core_config = load_core_config(config_path)
    except Exception as exc:  # pragma: no cover - guarded by higher level tests
        raise SystemExit(f"Nie udało się wczytać konfiguracji {config_path}: {exc}") from exc

    definitions_summary = describe_strategy_definitions(core_config)
    engines_summary = DEFAULT_STRATEGY_CATALOG.describe_engines()
    metadata, blocked = _collect_strategy_metadata(raw)

    regime_templates: dict[str, list[str]] = {}
    for entry in definitions_summary:
        if not isinstance(entry, MappingABC):
            continue
        profile = entry.get("risk_profile")
        if not profile and isinstance(entry.get("metadata"), MappingABC):
            profile = entry["metadata"].get("risk_profile")
        if not isinstance(profile, str):
            continue
        profile_key = profile.strip()
        if not profile_key:
            continue
        name = str(entry.get("name") or "").strip()
        if not name:
            continue
        regime_templates.setdefault(profile_key, []).append(name)

    payload: dict[str, Any] = {
        "engines": list(engines_summary),
        "definitions": list(definitions_summary),
        "metadata": metadata,
        "blocked": blocked,
    }
    if regime_templates:
        payload["regime_templates"] = regime_templates
    return payload


def run_preset_wizard(payload: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, MappingABC):
        return {"ok": False, "errors": ["Payload musi być słownikiem JSON"]}
    result = _validate_preset_payload(payload)
    return result


def _dump_schedulers(raw: Mapping[str, Any], *, only: str | None = None) -> dict[str, Any]:
    schedulers_raw = raw.get("multi_strategy_schedulers") or {}
    result: dict[str, Any] = {}
    if not isinstance(schedulers_raw, MappingABC):
        return result

    strategy_metadata, blocked_metadata = _collect_strategy_metadata(raw)
    guard = get_capability_guard()

    for name, payload in schedulers_raw.items():
        if only and only != name:
            continue
        if not isinstance(payload, MappingABC):
            continue

        schedules_payload = payload.get("schedules") or []
        schedules: list[dict[str, Any]] = []
        blocked_schedules: list[str] = []
        blocked_strategies: set[str] = set()
        blocked_capabilities: dict[str, str] = {}
        blocked_schedule_capabilities: dict[str, str] = {}
        strategy_capabilities: dict[str, str] = {}

        if isinstance(schedules_payload, list):
            for schedule in schedules_payload:
                if not isinstance(schedule, Mapping):
                    continue

                schedule_name = str(schedule.get("name") or "").strip()
                strategy_name = str(schedule.get("strategy") or "").strip()
                fallback_name = schedule_name if not strategy_name else strategy_name

                entry_payload: dict[str, Any] = {
                    "name": schedule.get("name"),
                    "strategy": schedule.get("strategy"),
                    "cadence_seconds": schedule.get("cadence_seconds"),
                    "max_drift_seconds": schedule.get("max_drift_seconds"),
                    "warmup_bars": schedule.get("warmup_bars"),
                    "risk_profile": schedule.get("risk_profile"),
                    "max_signals": schedule.get("max_signals"),
                    "interval": schedule.get("interval"),
                }

                metadata = strategy_metadata.get(strategy_name)
                blocked_meta = blocked_metadata.get(strategy_name)
                if metadata is None and not strategy_name:
                    metadata = strategy_metadata.get(schedule_name)
                    if blocked_meta is None:
                        blocked_meta = blocked_metadata.get(schedule_name)

                if metadata:
                    entry_payload["engine"] = metadata.get("engine")
                    if metadata.get("capability"):
                        entry_payload["capability"] = metadata.get("capability")
                    if metadata.get("license_tier"):
                        entry_payload["license_tier"] = metadata.get("license_tier")
                    entry_payload["risk_classes"] = list(metadata.get("risk_classes", ()))
                    entry_payload["required_data"] = list(metadata.get("required_data", ()))
                    tags = metadata.get("tags", ())
                    if tags:
                        entry_payload["tags"] = list(tags)
                    if metadata.get("risk_profile") and not entry_payload.get("risk_profile"):
                        entry_payload["definition_risk_profile"] = metadata.get("risk_profile")

                capability_id: str | None = None
                candidate_meta = metadata or blocked_meta
                if candidate_meta:
                    raw_capability = candidate_meta.get("capability")
                    if isinstance(raw_capability, str):
                        capability_id = raw_capability.strip() or None

                if capability_id:
                    if strategy_name:
                        strategy_capabilities.setdefault(strategy_name, capability_id)
                    elif fallback_name:
                        strategy_capabilities.setdefault(fallback_name, capability_id)

                if guard is not None and capability_id:
                    try:
                        if not guard.capabilities.is_strategy_enabled(capability_id):
                            if schedule_name and schedule_name not in blocked_schedules:
                                blocked_schedules.append(schedule_name)
                            strategy_key = strategy_name or fallback_name
                            if strategy_key:
                                blocked_strategies.add(strategy_key)
                                if capability_id:
                                    blocked_capabilities.setdefault(strategy_key, capability_id)
                            if schedule_name and capability_id:
                                blocked_schedule_capabilities.setdefault(schedule_name, capability_id)
                            continue
                    except AttributeError:
                        pass

                schedules.append(entry_payload)

        allowed_schedule_names = {
            str(entry.get("name"))
            for entry in schedules
            if entry.get("name") not in (None, "")
        }
        allowed_strategies = {
            str(entry.get("strategy"))
            for entry in schedules
            if entry.get("strategy") not in (None, "")
        }
        allowed_targets = allowed_schedule_names | allowed_strategies

        def _collect_limits(
            tree: Any,
            *,
            blocked: dict[str, set[str]] | None = None,
            blocked_capability_targets: dict[str, str] | None = None,
        ) -> dict[str, dict[str, Any]]:
            collected: dict[str, dict[str, Any]] = {}
            if not isinstance(tree, MappingABC):
                return collected
            for strategy_name, profiles in tree.items():
                strategy_key = str(strategy_name)
                if allowed_strategies and strategy_key not in allowed_strategies:
                    if blocked is not None:
                        blocked_profiles = blocked.setdefault(strategy_key, set())
                        if isinstance(profiles, MappingABC):
                            for profile_name in profiles.keys():
                                blocked_profiles.add(str(profile_name))
                        else:
                            blocked_profiles.add("*")
                    if blocked_capability_targets is not None:
                        capability_id = (
                            blocked_capabilities.get(strategy_key)
                            or strategy_capabilities.get(strategy_key)
                        )
                        if capability_id:
                            blocked_capability_targets.setdefault(strategy_key, capability_id)
                    continue
                if not isinstance(profiles, MappingABC):
                    continue
                profile_entry: dict[str, Any] = {}
                for profile_name, raw_limit in profiles.items():
                    profile_key = str(profile_name)
                    if isinstance(raw_limit, MappingABC):
                        if "limit" not in raw_limit or raw_limit["limit"] is None:
                            continue
                        try:
                            limit_value = int(raw_limit["limit"])
                        except (TypeError, ValueError):
                            continue
                        payload_limit: dict[str, Any] = {"limit": limit_value}
                        if raw_limit.get("reason"):
                            payload_limit["reason"] = raw_limit["reason"]
                        if raw_limit.get("until"):
                            payload_limit["until"] = raw_limit["until"]
                        if raw_limit.get("duration_seconds") is not None:
                            try:
                                payload_limit["duration_seconds"] = float(
                                    raw_limit["duration_seconds"]
                                )
                            except (TypeError, ValueError):
                                payload_limit["duration_seconds"] = raw_limit["duration_seconds"]
                    else:
                        try:
                            limit_value = int(raw_limit)
                        except (TypeError, ValueError):
                            continue
                        payload_limit = {"limit": limit_value}
                    profile_entry[profile_key] = payload_limit
                if profile_entry:
                    collected[strategy_key] = profile_entry
            return collected

        raw_suspensions = payload.get("initial_suspensions") or []
        suspensions: list[dict[str, Any]] = []
        blocked_suspensions: list[dict[str, Any]] = []
        blocked_suspension_capabilities: dict[str, str] = {}
        if isinstance(raw_suspensions, list):
            for entry in raw_suspensions:
                if not isinstance(entry, MappingABC):
                    continue
                suspension_payload: dict[str, Any] = {
                    "kind": entry.get("kind"),
                    "target": entry.get("target"),
                }
                if entry.get("reason"):
                    suspension_payload["reason"] = entry["reason"]
                if entry.get("until"):
                    suspension_payload["until"] = entry["until"]
                if entry.get("duration_seconds") is not None:
                    suspension_payload["duration_seconds"] = entry["duration_seconds"]
                kind = str(entry.get("kind") or "schedule").lower()
                target = str(entry.get("target") or "")
                if kind != "tag" and allowed_targets and target not in allowed_targets:
                    capability_id: str | None = None
                    if kind == "schedule":
                        capability_id = (
                            blocked_schedule_capabilities.get(target)
                            or strategy_capabilities.get(target)
                            or blocked_capabilities.get(target)
                        )
                    else:
                        capability_id = (
                            blocked_capabilities.get(target)
                            or strategy_capabilities.get(target)
                        )
                    if capability_id:
                        suspension_payload["capability"] = capability_id
                        key = f"{kind}:{target}".strip(":")
                        if key:
                            blocked_suspension_capabilities.setdefault(key, capability_id)
                    blocked_suspensions.append(dict(suspension_payload))
                    continue
                suspensions.append(suspension_payload)

        blocked_initial_limits: dict[str, set[str]] = {}
        blocked_signal_limits: dict[str, set[str]] = {}
        blocked_initial_limit_capabilities: dict[str, str] = {}
        blocked_signal_limit_capabilities: dict[str, str] = {}
        initial_limits = _collect_limits(
            payload.get("initial_signal_limits"),
            blocked=blocked_initial_limits,
            blocked_capability_targets=blocked_initial_limit_capabilities,
        )
        signal_limits = _collect_limits(
            payload.get("signal_limits"),
            blocked=blocked_signal_limits,
            blocked_capability_targets=blocked_signal_limit_capabilities,
        )

        entry_result: dict[str, Any] = {
            "name": name,
            "telemetry_namespace": payload.get("telemetry_namespace"),
            "decision_log_category": payload.get("decision_log_category"),
            "health_check_interval": payload.get("health_check_interval"),
            "portfolio_governor": payload.get("portfolio_governor"),
            "schedules": schedules,
            "initial_suspensions": suspensions,
            "initial_signal_limits": initial_limits,
        }
        if signal_limits:
            entry_result["signal_limits"] = signal_limits
        if blocked_schedules:
            entry_result["blocked_schedules"] = blocked_schedules
        if blocked_strategies:
            entry_result["blocked_strategies"] = sorted(blocked_strategies)
        if blocked_capabilities:
            entry_result["blocked_capabilities"] = {
                key: blocked_capabilities[key]
                for key in sorted(blocked_capabilities)
            }
        if blocked_schedule_capabilities:
            entry_result["blocked_schedule_capabilities"] = {
                key: blocked_schedule_capabilities[key]
                for key in blocked_schedules
                if key in blocked_schedule_capabilities
            }
        if blocked_suspensions:
            entry_result["blocked_suspensions"] = blocked_suspensions
        if blocked_suspension_capabilities:
            entry_result["blocked_suspension_capabilities"] = {
                key: blocked_suspension_capabilities[key]
                for key in sorted(blocked_suspension_capabilities)
            }
        merged_initial_capabilities: dict[str, str] = dict(blocked_initial_limit_capabilities)
        if blocked_initial_limits:
            entry_result["blocked_initial_signal_limits"] = {
                key: sorted(values)
                for key, values in blocked_initial_limits.items()
            }
            for key in blocked_initial_limits:
                capability_id = (
                    merged_initial_capabilities.get(key)
                    or blocked_capabilities.get(key)
                    or strategy_capabilities.get(key)
                )
                if capability_id:
                    merged_initial_capabilities[key] = capability_id
        if merged_initial_capabilities:
            entry_result["blocked_initial_signal_limit_capabilities"] = {
                key: merged_initial_capabilities[key]
                for key in sorted(merged_initial_capabilities)
            }

        merged_signal_capabilities: dict[str, str] = dict(blocked_signal_limit_capabilities)
        if blocked_signal_limits:
            entry_result["blocked_signal_limits"] = {
                key: sorted(values)
                for key, values in blocked_signal_limits.items()
            }
            for key in blocked_signal_limits:
                capability_id = (
                    merged_signal_capabilities.get(key)
                    or blocked_capabilities.get(key)
                    or strategy_capabilities.get(key)
                )
                if capability_id:
                    merged_signal_capabilities[key] = capability_id
        if merged_signal_capabilities:
            entry_result["blocked_signal_limit_capabilities"] = {
                key: merged_signal_capabilities[key]
                for key in sorted(merged_signal_capabilities)
            }

        result[name] = entry_result

    return result


def dump_config(raw: Mapping[str, Any], section: str, scheduler: str | None) -> dict[str, Any]:
    if section == "decision":
        return {"decision": _dump_decision(raw)}
    if section == "scheduler":
        return {"schedulers": _dump_schedulers(raw, only=scheduler)}
    return {
        "decision": _dump_decision(raw),
        "schedulers": _dump_schedulers(raw, only=scheduler),
    }


def _apply_decision(raw: dict[str, Any], payload: Mapping[str, Any]) -> None:
    if not payload:
        return
    decision = raw.setdefault("decision_engine", {})
    if not isinstance(decision, dict):
        raise SystemExit("Sekcja decision_engine musi być słownikiem w core.yaml")
    orchestrator = decision.setdefault("orchestrator", {})
    if not isinstance(orchestrator, dict):
        raise SystemExit("Sekcja decision_engine.orchestrator musi być słownikiem")

    for key in (
        "max_cost_bps",
        "min_net_edge_bps",
        "max_daily_loss_pct",
        "max_drawdown_pct",
        "max_position_ratio",
        "max_open_positions",
        "max_latency_ms",
        "max_trade_notional",
    ):
        if key in payload and payload[key] is not None:
            orchestrator[key] = payload[key]
    for key in ("min_probability", "require_cost_data", "penalty_cost_bps"):
        if key in payload and payload[key] is not None:
            decision[key] = payload[key]

    overrides_payload = payload.get("profile_overrides")
    if isinstance(overrides_payload, list):
        overrides: dict[str, dict[str, Any]] = {}
        for entry in overrides_payload:
            if not isinstance(entry, Mapping):
                continue
            profile = str(entry.get("profile")) if entry.get("profile") is not None else None
            if not profile:
                continue
            overrides[profile] = {}
            for key in (
                "max_cost_bps",
                "min_net_edge_bps",
                "max_daily_loss_pct",
                "max_drawdown_pct",
                "max_position_ratio",
                "max_open_positions",
                "max_latency_ms",
                "max_trade_notional",
            ):
                if key in entry and entry[key] is not None:
                    overrides[profile][key] = entry[key]
        if overrides:
            decision.setdefault("profile_overrides", {})
            if not isinstance(decision["profile_overrides"], dict):
                decision["profile_overrides"] = {}
            decision["profile_overrides"].update(overrides)


def _apply_scheduler(raw: dict[str, Any], payload: Mapping[str, Any]) -> None:
    if not payload:
        return
    schedulers = raw.setdefault("multi_strategy_schedulers", {})
    if not isinstance(schedulers, dict):
        raise SystemExit("Sekcja multi_strategy_schedulers musi być słownikiem")
    for name, entry in payload.items():
        if name not in schedulers:
            raise SystemExit(f"Scheduler {name} nie istnieje w core.yaml")
        target = schedulers[name]
        if not isinstance(target, dict):
            raise SystemExit(f"Sekcja multi_strategy_schedulers[{name}] musi być słownikiem")
        for key in (
            "telemetry_namespace",
            "decision_log_category",
            "health_check_interval",
            "portfolio_governor",
        ):
            if key in entry and entry[key] is not None:
                target[key] = entry[key]
        schedules_update = entry.get("schedules")
        if isinstance(schedules_update, list):
            schedules = target.get("schedules")
            if not isinstance(schedules, list):
                raise SystemExit(f"Scheduler {name} nie definiuje listy schedules")
            index = {str(item.get("name")): idx for idx, item in enumerate(schedules) if isinstance(item, Mapping)}
            for update in schedules_update:
                if not isinstance(update, Mapping):
                    continue
                schedule_name = update.get("name")
                if schedule_name is None or str(schedule_name) not in index:
                    raise SystemExit(
                        f"Scheduler {name} nie zawiera zadania o nazwie {schedule_name!r}")
                schedule_idx = index[str(schedule_name)]
                schedule = schedules[schedule_idx]
                if not isinstance(schedule, dict):
                    raise SystemExit(
                        f"Scheduler {name}: wpis {schedule_name!r} ma nieprawidłowy format")
                for key in (
                    "strategy",
                    "cadence_seconds",
                    "max_drift_seconds",
                    "warmup_bars",
                    "risk_profile",
                    "max_signals",
                    "interval",
                ):
                    if key in update and update[key] is not None:
                        schedule[key] = update[key]


def apply_updates(path: Path, payload: Mapping[str, Any]) -> None:
    raw: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise SystemExit("Plik core.yaml ma niepoprawny format – oczekiwano mapy klucz/wartość")

    if "decision" in payload and isinstance(payload["decision"], Mapping):
        _apply_decision(raw, payload["decision"])
    if "schedulers" in payload and isinstance(payload["schedulers"], Mapping):
        _apply_scheduler(raw, payload["schedulers"])

    load_core_config(path)  # walidacja

    path.write_text(yaml.safe_dump(raw, sort_keys=False, allow_unicode=True, indent=2), encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mostek konfiguracji strategii dla UI")
    parser.add_argument("--config", default="config/core.yaml", help="Ścieżka do pliku core.yaml")
    parser.add_argument("--dump", action="store_true", help="Zrzuca konfigurację w formacie JSON")
    parser.add_argument("--apply", action="store_true", help="Aktualizuje konfigurację na podstawie JSON")
    parser.add_argument("--describe-catalog", action="store_true", help="Zwraca opis katalogu strategii")
    parser.add_argument(
        "--describe-regime-workflow",
        action="store_true",
        help="Zwraca raport StrategyRegimeWorkflow (dostępność presetów, statystyki i historię)",
    )
    parser.add_argument(
        "--preset-wizard",
        action="store_true",
        help="Uruchamia kreator presetów (wymaga JSON na STDIN lub w pliku)",
    )
    parser.add_argument(
        "--section",
        choices=["all", "decision", "scheduler"],
        default="all",
        help="Ogranicza zakres dumpa",
    )
    parser.add_argument("--scheduler", help="Nazwa schedulera multi-strategy do zrzutu")
    parser.add_argument("--input", help="Plik JSON (lub '-' dla STDIN) wykorzystywany przy --apply")
    parser.add_argument(
        "--regime-workflow-dir",
        default="var/data/strategy_regime_workflow",
        help="Katalog ze snapshotami StrategyRegimeWorkflow (availability.json, activation_history.json)",
    )
    parser.add_argument(
        "--wizard-mode",
        choices=["validate", "build"],
        default="validate",
        help="Tryb pracy kreatora presetów",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config_path = Path(args.config).expanduser()
    if not config_path.exists():
        raise SystemExit(f"Plik konfiguracji {config_path} nie istnieje")

    raw: dict[str, Any] = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}

    if args.apply:
        payload = _read_json_input(args.input)
        if not isinstance(payload, Mapping):
            raise SystemExit("JSON musi zawierać słownik z sekcjami konfiguracji")
        apply_updates(config_path, payload)
        return 0

    if args.describe_regime_workflow:
        workflow_dir = Path(args.regime_workflow_dir).expanduser()
        data = describe_regime_workflow(workflow_dir)
        json.dump(data, sys.stdout, indent=2, ensure_ascii=False)
        sys.stdout.write("\n")
        return 0

    if args.describe_catalog:
        data = describe_catalog(config_path, raw)
        json.dump(data, sys.stdout, indent=2, ensure_ascii=False)
        sys.stdout.write("\n")
        return 0

    if args.preset_wizard:
        payload = _read_json_input(args.input)
        result = run_preset_wizard(payload)
        if args.wizard_mode == "build" and result.get("ok"):
            # Tryb build udostępnia pełen preset – zachowujemy kompatybilność z walidacją
            result.setdefault("mode", "build")
        else:
            result.setdefault("mode", "validate")
        json.dump(result, sys.stdout, indent=2, ensure_ascii=False)
        sys.stdout.write("\n")
        return 0

    data = dump_config(raw, args.section, args.scheduler)
    json.dump(data, sys.stdout, indent=2, ensure_ascii=False)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":  # pragma: no cover - entrypoint
    raise SystemExit(main())

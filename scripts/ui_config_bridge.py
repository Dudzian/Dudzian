#!/usr/bin/env python3
"""Mostek CLI do synchronizacji konfiguracji strategii/AI dla UI Qt."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable, Mapping as MappingABC
from pathlib import Path
from typing import Any, Mapping

import yaml

try:
    from bot_core.config.loader import load_core_config
except Exception as exc:  # pragma: no cover - import guard
    raise SystemExit(f"Nie można zaimportować bot_core.config.loader: {exc}") from exc

from bot_core.strategies.catalog import DEFAULT_STRATEGY_CATALOG


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


def _collect_strategy_metadata(raw: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    definitions: dict[str, Mapping[str, Any]] = {}

    def _register(name: str, engine: str, entry: Mapping[str, Any]) -> None:
        try:
            spec = DEFAULT_STRATEGY_CATALOG.get(engine)
        except KeyError:
            spec = None

        license_tier = str(entry.get("license_tier") or "").strip()
        risk_classes = _normalize_sequence_field(entry.get("risk_classes"))
        required_data = _normalize_sequence_field(entry.get("required_data"))
        tags = _normalize_sequence_field(entry.get("tags"))

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

        capability = str(entry.get("capability") or "").strip()
        if not capability and spec and spec.capability:
            capability = spec.capability
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
        definitions[name] = payload

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
    _register_section("statistical_arbitrage_strategies", "statistical_arbitrage")
    _register_section("day_trading_strategies", "day_trading")
    _register_section("grid_strategies", "grid_trading")

    return definitions


def _dump_schedulers(raw: Mapping[str, Any], *, only: str | None = None) -> dict[str, Any]:
    schedulers_raw = raw.get("multi_strategy_schedulers") or {}
    result: dict[str, Any] = {}
    if not isinstance(schedulers_raw, MappingABC):
        return result
    strategy_metadata = _collect_strategy_metadata(raw)
    for name, payload in schedulers_raw.items():
        if only and only != name:
            continue
        if not isinstance(payload, MappingABC):
            continue
        schedules_payload = payload.get("schedules") or []
        schedules: list[dict[str, Any]] = []
        if isinstance(schedules_payload, list):
            for schedule in schedules_payload:
                if not isinstance(schedule, Mapping):
                    continue
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
                strategy_name = str(schedule.get("strategy") or "").strip()
                metadata = strategy_metadata.get(strategy_name)
                if metadata is None and not strategy_name:
                    fallback_name = str(schedule.get("name") or "").strip()
                    metadata = strategy_metadata.get(fallback_name)
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
                schedules.append(entry_payload)
        result[name] = {
            "name": name,
            "telemetry_namespace": payload.get("telemetry_namespace"),
            "decision_log_category": payload.get("decision_log_category"),
            "health_check_interval": payload.get("health_check_interval"),
            "portfolio_governor": payload.get("portfolio_governor"),
            "schedules": schedules,
        }
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
    parser.add_argument(
        "--section",
        choices=["all", "decision", "scheduler"],
        default="all",
        help="Ogranicza zakres dumpa",
    )
    parser.add_argument("--scheduler", help="Nazwa schedulera multi-strategy do zrzutu")
    parser.add_argument("--input", help="Plik JSON (lub '-' dla STDIN) wykorzystywany przy --apply")
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

    data = dump_config(raw, args.section, args.scheduler)
    json.dump(data, sys.stdout, indent=2, ensure_ascii=False)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":  # pragma: no cover - entrypoint
    raise SystemExit(main())

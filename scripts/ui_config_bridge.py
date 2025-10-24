#!/usr/bin/env python3
"""Mostek CLI do synchronizacji konfiguracji strategii/AI dla UI Qt."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from collections.abc import Iterable, Mapping as MappingABC
from pathlib import Path
from typing import Any, Callable, Mapping

import yaml

try:
    from bot_core.config.loader import load_core_config
except Exception as exc:  # pragma: no cover - import guard
    raise SystemExit(f"Nie można zaimportować bot_core.config.loader: {exc}") from exc

from bot_core.security.guards import LicenseCapabilityError, get_capability_guard
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


def _capability_allowed(capability: str | None) -> bool:
    guard = get_capability_guard()
    if guard is None or not capability:
        return True
    try:
        return guard.capabilities.is_strategy_enabled(capability)
    except AttributeError:
        return True


def _guard_summary_from_counts(
    counts: Mapping[str, int],
    capabilities: Mapping[str, str],
    reasons: Mapping[str, str],
) -> dict[str, Any]:
    total = 0
    capability_counts: Counter[str] = Counter()
    reason_counts: Counter[str] = Counter()
    for key, value in counts.items():
        try:
            count = int(value)
        except (TypeError, ValueError):
            continue
        if count <= 0:
            continue
        normalized_key = str(key).strip()
        if not normalized_key:
            continue
        total += count
        capability = capabilities.get(normalized_key)
        reason = reasons.get(normalized_key)
        if capability:
            capability_counts[str(capability)] += count
        if reason:
            reason_counts[str(reason)] += count
    if total <= 0:
        return {}
    payload: dict[str, Any] = {"total": int(total)}
    if capability_counts:
        payload["by_capability"] = {
            capability: int(capability_counts[capability])
            for capability in sorted(capability_counts)
        }
    if reason_counts:
        payload["by_reason"] = {
            reason: int(reason_counts[reason])
            for reason in sorted(reason_counts)
        }
    return payload


def _count_limit_profiles(entries: Mapping[str, Iterable[str]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for raw_name, profiles in entries.items():
        name = str(raw_name).strip()
        if not name:
            continue
        values = [str(profile).strip() for profile in profiles]
        cleaned = [value for value in values if value]
        count = len(cleaned) if cleaned else len(values)
        if count <= 0:
            continue
        counts[name] = count
    return counts


def _build_guard_summary_payload(
    *,
    blocked_strategies: Iterable[str],
    blocked_schedules: Iterable[str],
    blocked_initial_limits: Mapping[str, Iterable[str]],
    blocked_signal_limits: Mapping[str, Iterable[str]],
    blocked_suspensions: Iterable[Mapping[str, Any]],
    blocked_capabilities: Mapping[str, str],
    blocked_schedule_capabilities: Mapping[str, str],
    merged_initial_capabilities: Mapping[str, str],
    merged_initial_reasons: Mapping[str, str],
    merged_signal_capabilities: Mapping[str, str],
    merged_signal_reasons: Mapping[str, str],
    blocked_suspension_capabilities: Mapping[str, str],
    blocked_capability_reasons: Mapping[str, str],
    blocked_schedule_reasons: Mapping[str, str],
    blocked_suspension_reasons: Mapping[str, str],
) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    overall_capability_counts: Counter[str] = Counter()
    overall_reason_counts: Counter[str] = Counter()
    overall_total = 0

    def _attach(
        label: str,
        counts: Mapping[str, int],
        capabilities: Mapping[str, str],
        reasons: Mapping[str, str],
    ) -> None:
        nonlocal overall_total
        payload = _guard_summary_from_counts(counts, capabilities, reasons)
        if not payload:
            return
        summary[label] = payload
        overall_total += int(payload.get("total", 0) or 0)
        for capability, value in payload.get("by_capability", {}).items():
            overall_capability_counts[str(capability)] += int(value)
        for reason, value in payload.get("by_reason", {}).items():
            overall_reason_counts[str(reason)] += int(value)

    strategy_counts: Counter[str] = Counter(
        str(name).strip() for name in blocked_strategies if str(name).strip()
    )
    schedule_counts: Counter[str] = Counter(
        str(name).strip() for name in blocked_schedules if str(name).strip()
    )
    initial_counts = _count_limit_profiles(blocked_initial_limits)
    signal_counts = _count_limit_profiles(blocked_signal_limits)

    suspension_counts: dict[str, int] = {}
    for entry in blocked_suspensions:
        if not isinstance(entry, Mapping):
            continue
        kind = str(entry.get("kind") or "schedule")
        target = str(entry.get("target") or "")
        key = f"{kind}:{target}".strip(":")
        if not key:
            continue
        suspension_counts[key] = suspension_counts.get(key, 0) + 1
    for key in blocked_suspension_capabilities:
        normalized = str(key).strip()
        if normalized and normalized not in suspension_counts:
            suspension_counts[normalized] = 1
    for key in blocked_suspension_reasons:
        normalized = str(key).strip()
        if normalized and normalized not in suspension_counts:
            suspension_counts[normalized] = 1

    _attach("strategies", strategy_counts, blocked_capabilities, blocked_capability_reasons)
    _attach(
        "schedules",
        schedule_counts,
        blocked_schedule_capabilities,
        blocked_schedule_reasons,
    )
    _attach(
        "initial_signal_limits",
        initial_counts,
        merged_initial_capabilities,
        merged_initial_reasons,
    )
    _attach(
        "signal_limits",
        signal_counts,
        merged_signal_capabilities,
        merged_signal_reasons,
    )
    _attach(
        "suspensions",
        suspension_counts,
        blocked_suspension_capabilities,
        blocked_suspension_reasons,
    )

    if overall_total > 0:
        overall_payload: dict[str, Any] = {"total": int(overall_total)}
        if overall_capability_counts:
            overall_payload["by_capability"] = {
                capability: int(overall_capability_counts[capability])
                for capability in sorted(overall_capability_counts)
            }
        if overall_reason_counts:
            overall_payload["by_reason"] = {
                reason: int(overall_reason_counts[reason])
                for reason in sorted(overall_reason_counts)
            }
        summary["overall"] = overall_payload

    return summary


def _build_guard_details_payload(
    *,
    blocked_strategies: Iterable[str],
    blocked_schedules: Iterable[str],
    blocked_initial_limits: Mapping[str, Iterable[str]],
    blocked_signal_limits: Mapping[str, Iterable[str]],
    blocked_suspensions: Iterable[Mapping[str, Any]],
    blocked_capabilities: Mapping[str, str],
    blocked_schedule_capabilities: Mapping[str, str],
    merged_initial_capabilities: Mapping[str, str],
    merged_initial_reasons: Mapping[str, str],
    merged_signal_capabilities: Mapping[str, str],
    merged_signal_reasons: Mapping[str, str],
    blocked_suspension_capabilities: Mapping[str, str],
    blocked_capability_reasons: Mapping[str, str],
    blocked_schedule_reasons: Mapping[str, str],
    blocked_suspension_reasons: Mapping[str, str],
    resolve_guard_reason: Callable[[str | None], str | None] | None = None,
) -> dict[str, dict[str, list[dict[str, str]]]]:
    details: dict[str, dict[str, list[dict[str, str]]]] = {}

    def _register(
        capability: str | None,
        category: str,
        payload: Mapping[str, Any],
    ) -> None:
        capability_id = str(capability or "").strip()
        if not capability_id:
            return
        cleaned: dict[str, str] = {}
        for key, value in payload.items():
            normalized_key = str(key).strip()
            if not normalized_key:
                continue
            normalized_value = str(value).strip()
            if not normalized_value and normalized_key not in {"profile", "kind"}:
                continue
            cleaned[normalized_key] = normalized_value
        if not cleaned:
            return
        category_entries = details.setdefault(capability_id, {})
        entries = category_entries.setdefault(category, [])
        if cleaned in entries:
            return
        entries.append(cleaned)

    def _resolve_reason(capability: str | None, reason: str | None) -> str | None:
        if reason:
            return reason
        if resolve_guard_reason is not None:
            return resolve_guard_reason(capability)
        return None

    for name in blocked_strategies:
        capability = blocked_capabilities.get(name)
        reason = _resolve_reason(capability, blocked_capability_reasons.get(name))
        payload: dict[str, Any] = {"name": name}
        if reason:
            payload["reason"] = reason
        _register(capability, "strategies", payload)

    for name in blocked_schedules:
        capability = blocked_schedule_capabilities.get(name)
        reason = _resolve_reason(capability, blocked_schedule_reasons.get(name))
        payload: dict[str, Any] = {"name": name}
        if reason:
            payload["reason"] = reason
        _register(capability, "schedules", payload)

    for strategy, profiles in blocked_initial_limits.items():
        capability = merged_initial_capabilities.get(strategy)
        reason = _resolve_reason(capability, merged_initial_reasons.get(strategy))
        for profile in profiles:
            payload = {"strategy": strategy, "profile": profile}
            if reason:
                payload["reason"] = reason
            _register(capability, "initial_signal_limits", payload)

    for strategy, profiles in blocked_signal_limits.items():
        capability = merged_signal_capabilities.get(strategy)
        reason = _resolve_reason(capability, merged_signal_reasons.get(strategy))
        for profile in profiles:
            payload = {"strategy": strategy, "profile": profile}
            if reason:
                payload["reason"] = reason
            _register(capability, "signal_limits", payload)

    for entry in blocked_suspensions:
        if not isinstance(entry, Mapping):
            continue
        capability = entry.get("capability")
        reason = entry.get("guard_reason") or entry.get("reason")
        reason = _resolve_reason(str(capability), str(reason) if reason else None)
        payload = {
            "kind": entry.get("kind", "schedule"),
            "target": entry.get("target", ""),
        }
        if reason:
            payload["reason"] = reason
        _register(capability, "suspensions", payload)

    for key, capability in blocked_suspension_capabilities.items():
        reason = _resolve_reason(capability, blocked_suspension_reasons.get(key))
        kind, _, target = str(key).partition(":")
        payload = {"kind": kind or "schedule", "target": target}
        if reason:
            payload["reason"] = reason
        _register(capability, "suspensions", payload)

    def _sort_key(category: str, entry: Mapping[str, str]) -> tuple[str, ...]:
        if category in {"strategies", "schedules"}:
            return (
                entry.get("name", ""),
                entry.get("reason", ""),
            )
        if category in {"initial_signal_limits", "signal_limits"}:
            return (
                entry.get("strategy", ""),
                entry.get("profile", ""),
                entry.get("reason", ""),
            )
        if category == "suspensions":
            return (
                entry.get("kind", ""),
                entry.get("target", ""),
                entry.get("reason", ""),
            )
        return tuple(sorted(entry.items()))

    normalized: dict[str, dict[str, list[dict[str, str]]]] = {}
    for capability, category_map in details.items():
        if not category_map:
            continue
        normalized_categories: dict[str, list[dict[str, str]]] = {}
        for category, entries in category_map.items():
            if not entries:
                continue
            normalized_entries = [dict(entry) for entry in entries if entry]
            if not normalized_entries:
                continue
            normalized_entries.sort(
                key=lambda payload, *, _category=category: _sort_key(
                    _category, payload
                ),
            )
            normalized_categories[str(category)] = normalized_entries
        if normalized_categories:
            normalized[str(capability)] = normalized_categories

    return normalized


def _build_guard_detail_summary_payload(
    details: Mapping[str, Mapping[str, Iterable[Mapping[str, str]]]],
    *,
    resolve_guard_reason: Callable[[str | None], str | None] | None = None,
) -> dict[str, dict[str, dict[str, object]]]:
    if not isinstance(details, MappingABC):
        return {}

    summary: dict[str, dict[str, dict[str, object]]] = {}
    category_order = (
        "strategies",
        "schedules",
        "initial_signal_limits",
        "signal_limits",
        "suspensions",
    )

    def _resolve_reason(capability: str | None, reason: str | None) -> str | None:
        if reason:
            return reason
        if resolve_guard_reason is not None:
            return resolve_guard_reason(capability)
        return None

    for capability, category_map in details.items():
        if not isinstance(category_map, MappingABC):
            continue
        capability_id = str(capability).strip()
        if not capability_id:
            continue

        capability_summary: dict[str, dict[str, object]] = {}
        overall_total = 0
        overall_reason_counts: Counter[str] = Counter()

        processed_categories: set[str] = set()
        for category in category_order:
            entries = category_map.get(category)
            if not isinstance(entries, Iterable):
                continue
            total = 0
            reason_counts: Counter[str] = Counter()
            for entry in entries:
                if not isinstance(entry, MappingABC):
                    continue
                total += 1
                reason = _resolve_reason(capability_id, str(entry.get("reason", "")).strip())
                if reason:
                    reason_counts[str(reason)] += 1
            if total <= 0:
                continue
            payload: dict[str, object] = {"total": int(total)}
            if reason_counts:
                payload["by_reason"] = {
                    reason: int(reason_counts[reason])
                    for reason in sorted(reason_counts)
                }
            capability_summary[category] = payload
            processed_categories.add(str(category))
            overall_total += total
            overall_reason_counts.update(reason_counts)

        for category, entries in category_map.items():
            category_label = str(category)
            if category_label in processed_categories:
                continue
            if not isinstance(entries, Iterable):
                continue
            total = 0
            reason_counts: Counter[str] = Counter()
            for entry in entries:
                if not isinstance(entry, MappingABC):
                    continue
                total += 1
                reason = _resolve_reason(capability_id, str(entry.get("reason", "")).strip())
                if reason:
                    reason_counts[str(reason)] += 1
            if total <= 0:
                continue
            payload = {"total": int(total)}
            if reason_counts:
                payload["by_reason"] = {
                    reason: int(reason_counts[reason])
                    for reason in sorted(reason_counts)
                }
            capability_summary[category_label] = payload
            overall_total += total
            overall_reason_counts.update(reason_counts)

        if overall_total > 0:
            overall_payload: dict[str, object] = {"total": int(overall_total)}
            if overall_reason_counts:
                overall_payload["by_reason"] = {
                    reason: int(overall_reason_counts[reason])
                    for reason in sorted(overall_reason_counts)
                }
            capability_summary["overall"] = overall_payload

        if capability_summary:
            summary[capability_id] = capability_summary

    return summary


def _build_guard_detail_category_summary_payload(
    details: Mapping[str, Mapping[str, Iterable[Mapping[str, str]]]],
    *,
    resolve_guard_reason: Callable[[str | None], str | None] | None = None,
) -> dict[str, dict[str, object]]:
    if not isinstance(details, MappingABC):
        return {}

    category_counters: dict[str, dict[str, object]] = {}
    category_order = (
        "strategies",
        "schedules",
        "initial_signal_limits",
        "signal_limits",
        "suspensions",
    )
    order_set = set(category_order)

    def _resolve_reason(capability: str, reason: str | None) -> str | None:
        if reason:
            return reason
        if resolve_guard_reason is not None:
            return resolve_guard_reason(capability)
        return None

    for capability, category_map in details.items():
        if not isinstance(category_map, MappingABC):
            continue
        capability_id = str(capability).strip()
        if not capability_id:
            continue
        for category, entries in category_map.items():
            category_name = str(category).strip()
            if not category_name or not isinstance(entries, Iterable):
                continue
            total = 0
            reason_counts: Counter[str] = Counter()
            for entry in entries:
                if not isinstance(entry, MappingABC):
                    continue
                total += 1
                reason = _resolve_reason(capability_id, entry.get("reason"))
                if reason:
                    reason_counts[str(reason)] += 1
            if total <= 0:
                continue
            counters = category_counters.setdefault(
                category_name,
                {
                    "total": 0,
                    "by_capability": Counter(),
                    "by_reason": Counter(),
                },
            )
            counters["total"] = int(counters["total"]) + total  # type: ignore[index]
            capability_counter: Counter[str] = counters["by_capability"]  # type: ignore[assignment]
            capability_counter[capability_id] += total
            reason_counter: Counter[str] = counters["by_reason"]  # type: ignore[assignment]
            reason_counter.update(reason_counts)

    if not category_counters:
        return {}

    def _normalize(payload: Mapping[str, object]) -> dict[str, object] | None:
        total = payload.get("total")
        try:
            total_value = int(total) if total is not None else 0
        except (TypeError, ValueError):
            return None
        if total_value <= 0:
            return None
        entry: dict[str, object] = {"total": total_value}
        capabilities = payload.get("by_capability")
        if isinstance(capabilities, Counter):
            cleaned_caps = {
                capability: int(capabilities[capability])
                for capability in sorted(capabilities)
                if capabilities[capability] > 0
            }
            if cleaned_caps:
                entry["by_capability"] = cleaned_caps
        reasons = payload.get("by_reason")
        if isinstance(reasons, Counter):
            cleaned_reasons = {
                reason: int(reasons[reason])
                for reason in sorted(reasons)
                if reasons[reason] > 0
            }
            if cleaned_reasons:
                entry["by_reason"] = cleaned_reasons
        return entry

    summary: dict[str, dict[str, object]] = {}
    for category in category_order:
        counters = category_counters.get(category)
        if not counters:
            continue
        normalized = _normalize(counters)
        if normalized:
            summary[category] = normalized

    extra_categories = {
        name: counters
        for name, counters in category_counters.items()
        if name not in order_set
    }
    for category in sorted(extra_categories):
        normalized = _normalize(extra_categories[category])
        if normalized:
            summary[category] = normalized

    return summary


def _build_guard_detail_reason_details_payload(
    details: Mapping[str, Mapping[str, Iterable[Mapping[str, str]]]],
    *,
    resolve_guard_reason: Callable[[str | None], str | None] | None = None,
) -> dict[str, dict[str, object]]:
    if not isinstance(details, MappingABC):
        return {}

    reason_entries: dict[str, dict[str, object]] = {}
    category_order = (
        "strategies",
        "schedules",
        "initial_signal_limits",
        "signal_limits",
        "suspensions",
    )
    order_set = set(category_order)

    def _resolve_reason(capability: str | None, reason: str | None) -> str | None:
        if reason:
            return reason
        if resolve_guard_reason is not None:
            return resolve_guard_reason(capability)
        return None

    def _sort_key(category: str, entry: Mapping[str, str]) -> tuple[str, ...]:
        if category in {"strategies", "schedules"}:
            return (
                entry.get("name", ""),
                entry.get("reason", ""),
            )
        if category in {"initial_signal_limits", "signal_limits"}:
            return (
                entry.get("strategy", ""),
                entry.get("profile", ""),
                entry.get("reason", ""),
            )
        if category == "suspensions":
            return (
                entry.get("kind", ""),
                entry.get("target", ""),
                entry.get("reason", ""),
            )
        return tuple(sorted(entry.items()))

    for capability, category_map in details.items():
        if not isinstance(category_map, MappingABC):
            continue
        capability_id = str(capability).strip()
        if not capability_id:
            continue
        for category, entries in category_map.items():
            category_name = str(category).strip()
            if not category_name or not isinstance(entries, Iterable):
                continue
            for entry in entries:
                if not isinstance(entry, MappingABC):
                    continue
                reason = _resolve_reason(
                    capability_id, str(entry.get("reason", "")).strip() or None
                )
                if not reason:
                    continue
                payload = reason_entries.setdefault(
                    reason,
                    {
                        "total": 0,
                        "capabilities": set(),
                        "categories": {},
                    },
                )
                payload["total"] = int(payload.get("total", 0) or 0) + 1
                capability_set = payload.setdefault("capabilities", set())
                if isinstance(capability_set, set):
                    capability_set.add(capability_id)
                categories_map = payload.setdefault("categories", {})
                if not isinstance(categories_map, dict):
                    continue
                cleaned_entry = {
                    str(key).strip(): str(value).strip()
                    for key, value in entry.items()
                    if str(key).strip()
                    and (
                        str(value).strip()
                        or str(key).strip() in {"profile", "kind"}
                        or (str(key).strip() == "reason" and str(value).strip())
                    )
                }
                if "reason" not in cleaned_entry and reason:
                    cleaned_entry["reason"] = reason
                category_entries = categories_map.setdefault(category_name, [])
                if isinstance(category_entries, list) and cleaned_entry not in category_entries:
                    category_entries.append(cleaned_entry)

    if not reason_entries:
        return {}

    normalized: dict[str, dict[str, object]] = {}
    for reason, payload in sorted(reason_entries.items()):
        try:
            total = int(payload.get("total", 0) or 0)
        except (TypeError, ValueError):
            continue
        if total <= 0:
            continue
        entry: dict[str, object] = {"total": total}
        capabilities = payload.get("capabilities")
        if isinstance(capabilities, set):
            cleaned_caps = sorted(cap for cap in capabilities if str(cap).strip())
            if cleaned_caps:
                entry["capabilities"] = [str(cap) for cap in cleaned_caps]
        categories = payload.get("categories")
        if isinstance(categories, Mapping):
            ordered_categories: dict[str, list[dict[str, str]]] = {}
            for category in category_order:
                entries = categories.get(category)
                if not isinstance(entries, list) or not entries:
                    continue
                normalized_entries = [
                    {
                        str(key).strip(): str(value).strip()
                        for key, value in entry.items()
                        if str(key).strip()
                        and (
                            str(value).strip()
                            or str(key).strip() in {"profile", "kind"}
                            or (str(key).strip() == "reason" and str(value).strip())
                        )
                    }
                    for entry in entries
                    if isinstance(entry, MappingABC)
                ]
                normalized_entries = [item for item in normalized_entries if item]
                if not normalized_entries:
                    continue
                normalized_entries.sort(
                    key=lambda payload, *, _category=category: _sort_key(
                        str(_category), payload
                    )
                )
                ordered_categories[category] = normalized_entries
            extra_categories = {
                str(category): entries
                for category, entries in categories.items()
                if str(category) not in order_set
            }
            for category in sorted(extra_categories):
                entries = extra_categories[category]
                if not isinstance(entries, list) or not entries:
                    continue
                normalized_entries = [
                    {
                        str(key).strip(): str(value).strip()
                        for key, value in entry.items()
                        if str(key).strip()
                        and (
                            str(value).strip()
                            or str(key).strip() in {"profile", "kind"}
                            or (str(key).strip() == "reason" and str(value).strip())
                        )
                    }
                    for entry in entries
                    if isinstance(entry, MappingABC)
                ]
                normalized_entries = [item for item in normalized_entries if item]
                if not normalized_entries:
                    continue
                normalized_entries.sort(
                    key=lambda payload, *, _category=category: _sort_key(
                        str(_category), payload
                    )
                )
                ordered_categories[category] = normalized_entries
            if ordered_categories:
                entry["categories"] = ordered_categories
        normalized[reason] = entry

    return normalized


def _build_guard_detail_reason_summary_payload(
    details: Mapping[str, Mapping[str, Iterable[Mapping[str, str]]]],
    *,
    resolve_guard_reason: Callable[[str | None], str | None] | None = None,
) -> dict[str, dict[str, object]]:
    if not isinstance(details, MappingABC):
        return {}

    reason_counters: dict[str, dict[str, object]] = {}

    for capability, category_map in details.items():
        if not isinstance(category_map, MappingABC):
            continue
        capability_id = str(capability).strip()
        if not capability_id:
            continue
        for category, entries in category_map.items():
            category_name = str(category).strip()
            if not category_name or not isinstance(entries, Iterable):
                continue
            for entry in entries:
                if not isinstance(entry, MappingABC):
                    continue
                reason = entry.get("reason")
                normalized_reason: str | None
                if isinstance(reason, str) and reason.strip():
                    normalized_reason = reason.strip()
                elif resolve_guard_reason is not None:
                    normalized_reason = resolve_guard_reason(capability_id)
                else:
                    normalized_reason = None
                if not normalized_reason:
                    continue
                counters = reason_counters.setdefault(
                    normalized_reason,
                    {
                        "total": 0,
                        "by_capability": Counter(),
                        "by_category": Counter(),
                    },
                )
                counters["total"] = int(counters.get("total", 0)) + 1  # type: ignore[index]
                capability_counter: Counter[str] = counters["by_capability"]  # type: ignore[assignment]
                capability_counter[capability_id] += 1
                category_counter: Counter[str] = counters["by_category"]  # type: ignore[assignment]
                category_counter[category_name] += 1

    if not reason_counters:
        return {}

    def _normalize(payload: Mapping[str, object]) -> dict[str, object] | None:
        total = payload.get("total")
        try:
            total_value = int(total) if total is not None else 0
        except (TypeError, ValueError):
            return None
        if total_value <= 0:
            return None
        entry: dict[str, object] = {"total": total_value}
        capability_counter = payload.get("by_capability")
        if isinstance(capability_counter, Counter):
            cleaned_capabilities = {
                capability: int(capability_counter[capability])
                for capability in sorted(capability_counter)
                if capability_counter[capability] > 0
            }
            if cleaned_capabilities:
                entry["by_capability"] = cleaned_capabilities
        category_counter = payload.get("by_category")
        if isinstance(category_counter, Counter):
            cleaned_categories = {
                category: int(category_counter[category])
                for category in sorted(category_counter)
                if category_counter[category] > 0
            }
            if cleaned_categories:
                entry["by_category"] = cleaned_categories
        return entry

    summary: dict[str, dict[str, object]] = {}
    for reason, counters in sorted(reason_counters.items()):
        normalized = _normalize(counters)
        if normalized:
            summary[reason] = normalized

    return summary


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
    _register_section("statistical_arbitrage_strategies", "statistical_arbitrage")
    _register_section("day_trading_strategies", "day_trading")
    _register_section("grid_strategies", "grid_trading")

    return definitions, blocked


def _dump_schedulers(raw: Mapping[str, Any], *, only: str | None = None) -> dict[str, Any]:
    schedulers_raw = raw.get("multi_strategy_schedulers") or {}
    result: dict[str, Any] = {}
    if not isinstance(schedulers_raw, MappingABC):
        return result

    strategy_metadata, blocked_metadata = _collect_strategy_metadata(raw)
    guard = get_capability_guard()

    def _resolve_guard_reason(capability: str | None) -> str | None:
        if guard is None or not capability:
            return None
        try:
            guard.require_strategy(capability)
        except LicenseCapabilityError as exc:  # pragma: no cover - komunikat strażnika
            return str(exc)
        except Exception:  # pragma: no cover - brak kompatybilności strażnika
            return None
        return None

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
        blocked_capability_reasons: dict[str, str] = {}
        blocked_schedule_reasons: dict[str, str] = {}
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
                            reason = _resolve_guard_reason(capability_id)
                            if schedule_name and schedule_name not in blocked_schedules:
                                blocked_schedules.append(schedule_name)
                                if reason:
                                    blocked_schedule_reasons.setdefault(schedule_name, reason)
                            strategy_key = strategy_name or fallback_name
                            if strategy_key:
                                blocked_strategies.add(strategy_key)
                                if capability_id:
                                    blocked_capabilities.setdefault(strategy_key, capability_id)
                                if reason:
                                    blocked_capability_reasons.setdefault(strategy_key, reason)
                            if schedule_name and capability_id:
                                blocked_schedule_capabilities.setdefault(schedule_name, capability_id)
                            if schedule_name and reason:
                                blocked_schedule_reasons.setdefault(schedule_name, reason)
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
            blocked_reasons: dict[str, str] | None = None,
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
                            if (
                                blocked_reasons is not None
                                and strategy_key not in blocked_reasons
                            ):
                                reason = (
                                    blocked_capability_reasons.get(strategy_key)
                                    or blocked_schedule_reasons.get(strategy_key)
                                    or _resolve_guard_reason(capability_id)
                                )
                                if reason:
                                    blocked_reasons[strategy_key] = reason
                    elif (
                        blocked_reasons is not None
                        and strategy_key not in blocked_reasons
                        and (
                            blocked_capability_reasons.get(strategy_key)
                            or blocked_schedule_reasons.get(strategy_key)
                        )
                    ):
                        reason = (
                            blocked_capability_reasons.get(strategy_key)
                            or blocked_schedule_reasons.get(strategy_key)
                        )
                        if reason:
                            blocked_reasons[strategy_key] = reason
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
        blocked_suspension_reasons: dict[str, str] = {}
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
                            reason = (
                                blocked_schedule_reasons.get(target)
                                if kind == "schedule"
                                else blocked_capability_reasons.get(target)
                            )
                            if not reason:
                                reason = _resolve_guard_reason(capability_id)
                            if reason:
                                suspension_payload["guard_reason"] = reason
                                blocked_suspension_reasons.setdefault(key, reason)
                    blocked_suspensions.append(dict(suspension_payload))
                    continue
                suspensions.append(suspension_payload)

        blocked_initial_limits: dict[str, set[str]] = {}
        blocked_signal_limits: dict[str, set[str]] = {}
        blocked_initial_limit_capabilities: dict[str, str] = {}
        blocked_signal_limit_capabilities: dict[str, str] = {}
        blocked_initial_limit_reasons: dict[str, str] = {}
        blocked_signal_limit_reasons: dict[str, str] = {}
        initial_limits = _collect_limits(
            payload.get("initial_signal_limits"),
            blocked=blocked_initial_limits,
            blocked_capability_targets=blocked_initial_limit_capabilities,
            blocked_reasons=blocked_initial_limit_reasons,
        )
        signal_limits = _collect_limits(
            payload.get("signal_limits"),
            blocked=blocked_signal_limits,
            blocked_capability_targets=blocked_signal_limit_capabilities,
            blocked_reasons=blocked_signal_limit_reasons,
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
        if blocked_capability_reasons:
            entry_result["blocked_capability_reasons"] = {
                key: blocked_capability_reasons[key]
                for key in sorted(blocked_capability_reasons)
            }
        if blocked_schedule_capabilities:
            entry_result["blocked_schedule_capabilities"] = {
                key: blocked_schedule_capabilities[key]
                for key in blocked_schedules
                if key in blocked_schedule_capabilities
            }
        if blocked_schedule_reasons:
            entry_result["blocked_schedule_capability_reasons"] = {
                key: blocked_schedule_reasons[key]
                for key in blocked_schedules
                if key in blocked_schedule_reasons
            }
        if blocked_suspensions:
            entry_result["blocked_suspensions"] = blocked_suspensions
        if blocked_suspension_capabilities:
            entry_result["blocked_suspension_capabilities"] = {
                key: blocked_suspension_capabilities[key]
                for key in sorted(blocked_suspension_capabilities)
            }
        if blocked_suspension_reasons:
            entry_result["blocked_suspension_reasons"] = {
                key: blocked_suspension_reasons[key]
                for key in sorted(blocked_suspension_reasons)
            }
        merged_initial_capabilities: dict[str, str] = dict(blocked_initial_limit_capabilities)
        merged_initial_reasons: dict[str, str] = dict(blocked_initial_limit_reasons)
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
                reason = (
                    merged_initial_reasons.get(key)
                    or blocked_capability_reasons.get(key)
                    or blocked_schedule_reasons.get(key)
                )
                if not reason and capability_id:
                    reason = _resolve_guard_reason(capability_id)
                if reason:
                    merged_initial_reasons[key] = reason
        if merged_initial_capabilities:
            entry_result["blocked_initial_signal_limit_capabilities"] = {
                key: merged_initial_capabilities[key]
                for key in sorted(merged_initial_capabilities)
            }
        if merged_initial_reasons:
            entry_result["blocked_initial_signal_limit_reasons"] = {
                key: merged_initial_reasons[key]
                for key in sorted(merged_initial_reasons)
            }

        merged_signal_capabilities: dict[str, str] = dict(blocked_signal_limit_capabilities)
        merged_signal_reasons: dict[str, str] = dict(blocked_signal_limit_reasons)
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
                reason = (
                    merged_signal_reasons.get(key)
                    or blocked_capability_reasons.get(key)
                    or blocked_schedule_reasons.get(key)
                )
                if not reason and capability_id:
                    reason = _resolve_guard_reason(capability_id)
                if reason:
                    merged_signal_reasons[key] = reason
        if merged_signal_capabilities:
            entry_result["blocked_signal_limit_capabilities"] = {
                key: merged_signal_capabilities[key]
                for key in sorted(merged_signal_capabilities)
            }
        if merged_signal_reasons:
            entry_result["blocked_signal_limit_reasons"] = {
                key: merged_signal_reasons[key]
                for key in sorted(merged_signal_reasons)
            }

        guard_summary = _build_guard_summary_payload(
            blocked_strategies=blocked_strategies,
            blocked_schedules=blocked_schedules,
            blocked_initial_limits=blocked_initial_limits,
            blocked_signal_limits=blocked_signal_limits,
            blocked_suspensions=blocked_suspensions,
            blocked_capabilities=blocked_capabilities,
            blocked_schedule_capabilities=blocked_schedule_capabilities,
            merged_initial_capabilities=merged_initial_capabilities,
            merged_initial_reasons=merged_initial_reasons,
            merged_signal_capabilities=merged_signal_capabilities,
            merged_signal_reasons=merged_signal_reasons,
            blocked_suspension_capabilities=blocked_suspension_capabilities,
            blocked_capability_reasons=blocked_capability_reasons,
            blocked_schedule_reasons=blocked_schedule_reasons,
            blocked_suspension_reasons=blocked_suspension_reasons,
        )
        if guard_summary:
            entry_result["guard_reason_summary"] = guard_summary
        guard_details = _build_guard_details_payload(
            blocked_strategies=blocked_strategies,
            blocked_schedules=blocked_schedules,
            blocked_initial_limits=blocked_initial_limits,
            blocked_signal_limits=blocked_signal_limits,
            blocked_suspensions=blocked_suspensions,
            blocked_capabilities=blocked_capabilities,
            blocked_schedule_capabilities=blocked_schedule_capabilities,
            merged_initial_capabilities=merged_initial_capabilities,
            merged_initial_reasons=merged_initial_reasons,
            merged_signal_capabilities=merged_signal_capabilities,
            merged_signal_reasons=merged_signal_reasons,
            blocked_suspension_capabilities=blocked_suspension_capabilities,
            blocked_capability_reasons=blocked_capability_reasons,
            blocked_schedule_reasons=blocked_schedule_reasons,
            blocked_suspension_reasons=blocked_suspension_reasons,
            resolve_guard_reason=_resolve_guard_reason,
        )
        if guard_details:
            entry_result["guard_reason_details"] = guard_details
            guard_detail_summary = _build_guard_detail_summary_payload(
                guard_details,
                resolve_guard_reason=_resolve_guard_reason,
            )
            if guard_detail_summary:
                entry_result["guard_reason_detail_summary"] = guard_detail_summary
            guard_detail_category_summary = _build_guard_detail_category_summary_payload(
                guard_details,
                resolve_guard_reason=_resolve_guard_reason,
            )
            if guard_detail_category_summary:
                entry_result["guard_reason_detail_category_summary"] = (
                    guard_detail_category_summary
                )
            guard_detail_reason_summary = _build_guard_detail_reason_summary_payload(
                guard_details,
                resolve_guard_reason=_resolve_guard_reason,
            )
            if guard_detail_reason_summary:
                entry_result["guard_reason_detail_reason_summary"] = (
                    guard_detail_reason_summary
                )
            guard_detail_reason_details = _build_guard_detail_reason_details_payload(
                guard_details,
                resolve_guard_reason=_resolve_guard_reason,
            )
            if guard_detail_reason_details:
                entry_result["guard_reason_detail_reason_details"] = (
                    guard_detail_reason_details
                )

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

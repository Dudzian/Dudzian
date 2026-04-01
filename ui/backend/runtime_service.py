"""Serwis runtime dostarczający dane dziennika decyzji do QML."""

from __future__ import annotations


import contextlib
import json
import logging
import math
import os
import queue
import re
import statistics
import tempfile
import threading
import time
from collections import Counter, deque
from collections.abc import Iterable, Mapping, MutableMapping
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Callable
from copy import deepcopy

try:  # pragma: no cover - PyYAML może być opcjonalne w dystrybucjach light
    import yaml
except Exception:  # pragma: no cover - fallback gdy brak zależności PyYAML
    yaml = None  # type: ignore[assignment]

from PySide6.QtCore import QObject, Property, QTimer, Signal, Slot

from bot_core.auto_trader.ai_governor import AutoTraderAIGovernor, AutoTraderAIGovernorRunner
from bot_core.config.models import DecisionEngineConfig, DecisionOrchestratorThresholds
from bot_core.decision.orchestrator import DecisionOrchestrator
from bot_core.config import load_core_config
from bot_core.observability.ui_metrics import (
    FeedHealthMetricsExporter,
    RiskJournalMetricsExporter,
    get_feed_health_metrics_exporter,
    get_risk_journal_metrics_exporter,
    get_long_poll_metrics_cache,
)
from bot_core.portfolio import resolve_decision_log_config
from bot_core.runtime.cloud_client import (
    CloudClientOptions,
    CloudHandshakeResult,
    LicenseIdentity,
    load_cloud_client_options,
    load_license_identity,
    perform_cloud_handshake,
)
from bot_core.runtime.journal import TradingDecisionJournal
from .ai_governor_demo import build_demo_ai_governor_snapshot
from .alert_manager import AlertManager
from .demo_data import load_demo_decisions
from .decision_log_repository import DecisionLogRepository
from .decision_payload_normalizer import (
    DecisionRecord,
    RuntimeDecisionEntry,
    parse_runtime_decision_entry,
)
from .decision_source_selector import DecisionSourceFallbackCoordinator
from .feed_health_tracker import FeedHealthTracker
from .operator_action_service import OperatorActionService
from .grpc_decision_stream_client import GrpcDecisionStreamClient
from .qml_bridge import to_plain_dict, to_plain_list, to_plain_text, to_plain_value

try:  # pragma: no cover - moduł może nie być dostępny w wersjach light
    from bot_core.ai import FilesystemModelRepository, ModelRepository
except Exception:  # pragma: no cover - fallback dla dystrybucji bez komponentu AI
    FilesystemModelRepository = None  # type: ignore[assignment]
    ModelRepository = None  # type: ignore[assignment]

try:  # pragma: no cover - harmonogram retrainingu jest opcjonalny
    from bot_core.runtime.ai_retrain import CronSchedule
except Exception:  # pragma: no cover - fallback dla środowisk bez retrainingu
    CronSchedule = None  # type: ignore[assignment]

try:  # pragma: no cover - funkcja ładowania runtime może nie być dostępna w starszych gałęziach
    from bot_core.config.loader import load_runtime_app_config
except Exception:  # pragma: no cover - fallback gdy brak unified loadera
    load_runtime_app_config = None  # type: ignore[assignment]

try:  # pragma: no cover - alerty feedu są opcjonalne w dystrybucjach light
    from bot_core.runtime.metrics_alerts import UiTelemetryAlertSink, get_feed_health_alert_sink
except Exception:  # pragma: no cover - fallback gdy brak infrastruktury alertów
    UiTelemetryAlertSink = None  # type: ignore[assignment]

    def get_feed_health_alert_sink(*_args: object, **_kwargs: object) -> None:
        return None


if TYPE_CHECKING:  # pragma: no cover - adnotacje tylko w czasie statycznym
    from bot_core.config.models import RuntimeAppConfig

_LOGGER = logging.getLogger(__name__)


def _require_yaml() -> None:
    """Zapewnia obecność zależności PyYAML zanim spróbujemy parsować pliki YAML."""

    if yaml is None:
        raise RuntimeError(
            "Do obsługi konfiguracji YAML w warstwie UI wymagany jest pakiet PyYAML;"
            " zainstaluj go poleceniem `pip install pyyaml`."
        )


try:  # pragma: no cover - gRPC może nie być dostępne w trybach light
    import grpc
except Exception:  # pragma: no cover - fallback gdy brak gRPC
    grpc = None  # type: ignore[assignment]

DecisionLoader = Callable[[int], Iterable[DecisionRecord]]

# Uwaga utrzymaniowa:
# klasyfikacja wpisu do Risk Journal oraz interpretacja block/freeze w _build_risk_context
# MUSZĄ ewoluować razem. Przy zmianie jednego zbioru zaktualizuj pozostałe i test matrix.
_RISK_CLASSIFICATION_TOKENS: frozenset[str] = frozenset(
    {
        "risk",
        "block",
        "blocked",
        "reject",
        "rejected",
        "freeze",
        "frozen",
        "unfreeze",
        "override",
        "stress",
        "lock",
    }
)

_RISK_BLOCK_KEYWORDS: frozenset[str] = frozenset(
    {"block", "blocked", "risk_block", "reject", "rejected"}
)
_RISK_FREEZE_KEYWORDS: frozenset[str] = frozenset({"freeze", "frozen", "lock"})
_AI_FEED_CHANNEL = "ai_governor"
_AI_HISTORY_LIMIT = 32


def _default_loader(limit: int) -> Iterable[DecisionRecord]:
    """Zapewnia dane demonstracyjne przy pierwszym uruchomieniu."""

    entries = list(load_demo_decisions())
    if not entries:
        return []
    if limit > 0:
        entries = entries[-limit:]
    # `load_demo_decisions()` zwraca chronologicznie (najstarsze -> najnowsze),
    # więc odwrócenie daje kolejność spójną z resztą runtime (najnowsze -> najstarsze).
    return reversed(entries)


def _load_from_journal(journal: TradingDecisionJournal, limit: int) -> Iterable[DecisionRecord]:
    exported = list(journal.export())
    if limit > 0:
        exported = exported[-limit:]
    return reversed(exported)


def _parse_entry(record: DecisionRecord) -> RuntimeDecisionEntry:
    return parse_runtime_decision_entry(record)


def _normalize_bool(value: object) -> object:
    """Normalizuje flagi bool używane w snapshotach i telemetrii runtime.

    Zasada utrzymaniowa: helpery dot. snapshotów/telemetrii zostają w runtime_service,
    a helpery dot. payloadu decyzji trafiają do decision_payload_normalizer.
    """

    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "yes", "1"}:
            return True
        if lowered in {"false", "no", "0"}:
            return False
    return value


def _normalize_sequence(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        sanitized = [segment.strip() for segment in value.replace(";", ",").split(",")]
        return [segment for segment in sanitized if segment]
    if isinstance(value, Mapping):
        reason = value.get("reason")
        if isinstance(reason, str) and reason.strip():
            return [reason.strip()]
        # fallback: reprezentacja tekstowa całego obiektu
        try:
            return [json.dumps(dict(value), ensure_ascii=False)]
        except TypeError:
            return [str(dict(value))]
    if isinstance(value, Iterable):
        result: list[str] = []
        for item in value:
            result.extend(_normalize_sequence(item))
        return result
    return []


def _to_mapping(value: object) -> Mapping[str, object]:
    if isinstance(value, Mapping):
        return value
    return {}


def _normalize_risk_journal_diagnostics(payload: Mapping[str, object]) -> dict[str, object]:
    diagnostics = _to_mapping(payload)
    entries_raw = diagnostics.get("incompleteEntries")
    if entries_raw is None:
        entries_raw = diagnostics.get("incomplete_entries")
    try:
        incomplete_entries = max(0, int(entries_raw or 0))
    except (TypeError, ValueError):
        incomplete_entries = 0

    samples_raw = diagnostics.get("incompleteSamples")
    if samples_raw is None:
        samples_raw = diagnostics.get("incomplete_samples")

    samples_count_raw = diagnostics.get("incompleteSamplesCount")
    if samples_count_raw is None:
        samples_count_raw = diagnostics.get("incomplete_samples_count")
    try:
        explicit_samples_count = max(0, int(samples_count_raw))
    except (TypeError, ValueError):
        explicit_samples_count = 0

    samples: list[object] = []
    samples_count = explicit_samples_count
    if isinstance(samples_raw, (int, float)) and not isinstance(samples_raw, bool):
        samples_count = max(samples_count, int(samples_raw))
    elif isinstance(samples_raw, Mapping):
        samples = [dict(samples_raw)]
    elif isinstance(samples_raw, Iterable) and not isinstance(samples_raw, (str, bytes)):
        samples = list(samples_raw)

    if not samples_count:
        samples_count = len(samples)
    else:
        samples_count = max(samples_count, len(samples))

    risk_flag_counts_raw = diagnostics.get("riskFlagCounts")
    if risk_flag_counts_raw is None:
        risk_flag_counts_raw = diagnostics.get("risk_flag_counts")
    risk_flag_counts: dict[str, int] = {}
    if isinstance(risk_flag_counts_raw, Mapping):
        for key, value in risk_flag_counts_raw.items():
            try:
                risk_flag_counts[str(key)] = max(0, int(value))
            except (TypeError, ValueError):
                continue

    return {
        "incompleteEntries": incomplete_entries,
        "incompleteSamples": samples,
        "incompleteSamplesCount": samples_count,
        "riskFlagCounts": risk_flag_counts,
    }


def _first_non_empty(*values: object) -> str:
    for value in values:
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _extract_risk_action(
    metadata: Mapping[str, object], decision: Mapping[str, object], status: str
) -> str:
    candidates = (
        metadata.get("riskAction"),
        metadata.get("risk_action"),
        metadata.get("action"),
        decision.get("riskAction"),
        decision.get("risk_action"),
        decision.get("action"),
    )
    action = _first_non_empty(*candidates)
    if action:
        return action
    if status.lower() in {"blocked", "rejected", "risk_block"}:
        return status
    return ""


def _compute_activity_score(is_block: bool, is_freeze: bool, is_override: bool) -> float:
    if is_block:
        return 1.0
    if is_freeze:
        return 0.85
    if is_override:
        return 0.7
    return 0.45


def _contains_keyword_token(value: str, keywords: set[str]) -> bool:
    if not value:
        return False
    tokens = set(re.findall(r"[a-z]+", value.lower()))
    if not tokens:
        return False
    return bool(tokens & keywords)


def _is_risk_journal_entry(
    entry: Mapping[str, object], metadata: Mapping[str, object], decision: Mapping[str, object]
) -> bool:
    """Klasyfikuje wpis decision streamu jako kandydat do Risk Journal.

    Kontrakt:
    - wpis jest risk-related, gdy event/status semantycznie wskazuje na ryzyko
      (risk/freeze/block/override/stress),
    - albo gdy payload zawiera jawne pola kontraktu Risk Journal
      (risk_action/risk_flags/stress_*).
    """

    event = str(entry.get("event") or "").strip().lower()
    status = str(entry.get("status") or "").strip().lower()
    if event.startswith("risk_") or status.startswith("risk_"):
        return True
    if _contains_keyword_token(event, _RISK_CLASSIFICATION_TOKENS) or _contains_keyword_token(
        status, _RISK_CLASSIFICATION_TOKENS
    ):
        return True

    tracked_keys = {
        "riskaction",
        "risk_action",
        "riskflags",
        "risk_flags",
        "stressfailures",
        "stress_failures",
        "stressoverrides",
        "stress_overrides",
        "stressoverride",
        "stress_override",
    }

    def _mapping_contains_tracked_keys(payload: Mapping[str, object]) -> bool:
        stack: list[object] = [payload]
        visited = 0
        while stack and visited < 64:  # ograniczenie dla bezpieczeństwa i wydajności
            current = stack.pop()
            visited += 1
            if isinstance(current, Mapping):
                for key, value in current.items():
                    if str(key).lower() in tracked_keys:
                        return True
                    stack.append(value)
            elif isinstance(current, Iterable) and not isinstance(current, (str, bytes)):
                stack.extend(current)
        return False

    if _mapping_contains_tracked_keys(entry):
        return True
    if _mapping_contains_tracked_keys(metadata):
        return True
    if _mapping_contains_tracked_keys(decision):
        return True
    return False


def _build_risk_context(
    entries: Iterable[Mapping[str, object]],
) -> tuple[dict[str, object], list[dict[str, object]], dict[str, object]]:
    block_keywords = set(_RISK_BLOCK_KEYWORDS)
    freeze_keywords = set(_RISK_FREEZE_KEYWORDS)
    override_keys = {"stressOverride", "stress_override", "stressOverrides", "stress_overrides"}

    blocks = 0
    freezes = 0
    overrides = 0
    total_entries = 0
    incomplete_entries = 0
    latest_stress_failures: list[str] = []
    latest_failure_ts = ""
    latest_risk_flags: list[str] = []
    last_block_entry: dict[str, object] | None = None
    last_freeze_entry: dict[str, object] | None = None
    last_override_entry: dict[str, object] | None = None
    stress_failure_set: set[str] = set()
    risk_flag_set: set[str] = set()
    strategy_set: set[str] = set()
    strategy_counter: Counter[str] = Counter()
    override_reason_set: set[str] = set()
    stress_failure_counter: Counter[str] = Counter()
    risk_flag_counter: Counter[str] = Counter()
    strategy_summaries: dict[str, dict[str, object]] = {}

    timeline: list[dict[str, object]] = []

    for entry in entries:
        metadata = _to_mapping(entry.get("metadata"))
        decision = _to_mapping(entry.get("decision"))
        if not _is_risk_journal_entry(entry, metadata, decision):
            continue
        total_entries += 1
        timestamp = str(entry.get("timestamp") or "")
        event = str(entry.get("event") or "")
        strategy = str(entry.get("strategy") or "").strip()
        risk_profile = str(entry.get("riskProfile") or "").strip()
        status = str(entry.get("status") or "").strip()
        risk_action = _extract_risk_action(metadata, decision, status)

        risk_flags = _normalize_sequence(
            metadata.get("riskFlags")
            or metadata.get("risk_flags")
            or decision.get("riskFlags")
            or decision.get("risk_flags")
        )
        stress_failures = _normalize_sequence(
            metadata.get("stressFailures")
            or metadata.get("stress_failures")
            or decision.get("stressFailures")
            or decision.get("stress_failures")
        )

        stress_overrides_payload = metadata.get("stressOverrides") or metadata.get(
            "stress_overrides"
        )
        if not stress_overrides_payload:
            stress_overrides_payload = decision.get("stressOverrides") or decision.get(
                "stress_overrides"
            )
        stress_overrides = _normalize_sequence(stress_overrides_payload)

        missing_fields: list[str] = []
        if not risk_action:
            missing_fields.append("risk_action")
        if not risk_flags and not stress_overrides:
            missing_fields.append("risk_flags|stress_overrides")

        is_block = (
            any(keyword in risk_action.lower() for keyword in block_keywords)
            if risk_action
            else False
        )
        if not is_block and status:
            is_block = any(keyword in status.lower() for keyword in block_keywords)

        is_freeze = _contains_keyword_token(risk_action, freeze_keywords)
        if not is_freeze and status:
            is_freeze = _contains_keyword_token(status, freeze_keywords)

        override_indicator = False
        if stress_overrides:
            override_indicator = True
        else:
            for key in override_keys:
                if metadata.get(key) or decision.get(key):
                    override_indicator = True
                    break
        is_override = override_indicator

        activity_score = _compute_activity_score(is_block, is_freeze, is_override)

        is_incomplete = bool(missing_fields)
        if is_incomplete:
            incomplete_entries += 1

        summary_bucket: dict[str, object] | None = None
        if strategy:
            strategy_set.add(strategy)
            strategy_counter.update([strategy])
            summary_bucket = strategy_summaries.setdefault(
                strategy,
                {
                    "strategy": strategy,
                    "blockCount": 0,
                    "freezeCount": 0,
                    "stressOverrideCount": 0,
                    "totalEvents": 0,
                    "lastEvent": "",
                    "lastTimestamp": "",
                    "lastRiskFlags": [],
                    "lastStressFailures": [],
                    "lastRiskAction": "",
                    "stressOverrideReasons": [],
                },
            )

        if not is_incomplete:
            if summary_bucket is not None:
                summary_bucket["totalEvents"] = int(summary_bucket.get("totalEvents", 0)) + 1
            if risk_flags:
                risk_flag_set.update(risk_flags)
                risk_flag_counter.update(risk_flags)
                latest_risk_flags = risk_flags
                if summary_bucket is not None and risk_flags:
                    summary_bucket["lastRiskFlags"] = list(risk_flags)
            if stress_failures:
                stress_failure_set.update(stress_failures)
                stress_failure_counter.update(stress_failures)
                latest_stress_failures = stress_failures
                latest_failure_ts = timestamp
                if summary_bucket is not None and stress_failures:
                    summary_bucket["lastStressFailures"] = list(stress_failures)
            if stress_overrides:
                override_reason_set.update(stress_overrides)
                if summary_bucket is not None and stress_overrides:
                    summary_bucket["stressOverrideReasons"] = list(stress_overrides)

            if is_block:
                blocks += 1
                if timestamp and (
                    not last_block_entry
                    or str(timestamp) >= str(last_block_entry.get("timestamp", ""))
                ):
                    last_block_entry = {
                        "timestamp": timestamp,
                        "event": event,
                        "strategy": strategy,
                    }
                if summary_bucket is not None:
                    summary_bucket["blockCount"] = int(summary_bucket.get("blockCount", 0)) + 1
            if is_freeze:
                freezes += 1
                if timestamp and (
                    not last_freeze_entry
                    or str(timestamp) >= str(last_freeze_entry.get("timestamp", ""))
                ):
                    last_freeze_entry = {
                        "timestamp": timestamp,
                        "event": event,
                        "strategy": strategy,
                        "riskAction": risk_action,
                    }
                if summary_bucket is not None:
                    summary_bucket["freezeCount"] = int(summary_bucket.get("freezeCount", 0)) + 1
            if is_override:
                overrides += 1
                if timestamp and (
                    not last_override_entry
                    or str(timestamp) >= str(last_override_entry.get("timestamp", ""))
                ):
                    last_override_entry = {
                        "timestamp": timestamp,
                        "event": event,
                        "strategy": strategy,
                    }
                if summary_bucket is not None:
                    summary_bucket["stressOverrideCount"] = (
                        int(summary_bucket.get("stressOverrideCount", 0)) + 1
                    )

            if summary_bucket is not None:
                if risk_action:
                    summary_bucket["lastRiskAction"] = risk_action
                if timestamp:
                    last_timestamp = str(summary_bucket.get("lastTimestamp", ""))
                    if not last_timestamp or str(timestamp) >= last_timestamp:
                        summary_bucket["lastTimestamp"] = timestamp
                        summary_bucket["lastEvent"] = event
                        summary_bucket["lastRiskFlags"] = list(risk_flags)
                        summary_bucket["lastStressFailures"] = list(stress_failures)

        timeline.append(
            {
                "timestamp": timestamp,
                "event": event,
                "strategy": strategy,
                "riskProfile": risk_profile,
                "status": status,
                "riskAction": risk_action,
                "riskFlags": risk_flags,
                "stressFailures": stress_failures,
                "stressOverrides": stress_overrides,
                "isBlock": is_block,
                "isFreeze": is_freeze,
                "isStressOverride": is_override,
                "activityScore": 0.0 if is_incomplete else activity_score,
                "isIncomplete": is_incomplete,
                "missingFields": missing_fields,
                "record": dict(entry),
            }
        )

    def _sort_key(item: Mapping[str, object]) -> tuple[int, str]:
        timestamp = str(item.get("timestamp") or "")
        return (0 if timestamp else 1, timestamp)

    timeline.sort(key=_sort_key)

    metrics: dict[str, object] = {
        "totalEntries": total_entries,
        "incompleteEntries": incomplete_entries,
        "incompleteSamples": 0,
    }

    severity_order = {"block": 0, "freeze": 1, "override": 2, "neutral": 3}

    def _classify(summary: Mapping[str, object]) -> str:
        if int(summary.get("blockCount", 0)) > 0:
            return "block"
        if int(summary.get("freezeCount", 0)) > 0:
            return "freeze"
        if int(summary.get("stressOverrideCount", 0)) > 0:
            return "override"
        return "neutral"

    def _timestamp_key(value: object) -> tuple[int, str]:
        if isinstance(value, str) and value:
            candidate = value.replace("Z", "+00:00")
            try:
                parsed = datetime.fromisoformat(candidate)
            except ValueError:
                return (0, candidate)
            return (-int(parsed.timestamp()), candidate)
        return (0, "")

    summaries_payload: list[dict[str, object]] = []
    for summary in strategy_summaries.values():
        payload = dict(summary)
        payload["severity"] = _classify(summary)
        payload.setdefault("stressOverrideReasons", [])
        summaries_payload.append(payload)

    summaries_payload.sort(
        key=lambda item: (
            severity_order.get(str(item.get("severity", "neutral")), 99),
            -int(item.get("blockCount", 0)),
            -int(item.get("freezeCount", 0)),
            -int(item.get("stressOverrideCount", 0)),
            _timestamp_key(item.get("lastTimestamp", "")),
            str(item.get("strategy", "")),
        )
    )

    metrics.update(
        {
            "blockCount": blocks,
            "freezeCount": freezes,
            "stressOverrideCount": overrides,
            "lastBlock": dict(last_block_entry or {}),
            "lastFreeze": dict(last_freeze_entry or {}),
            "lastStressOverride": dict(last_override_entry or {}),
            "latestStressFailures": latest_stress_failures,
            "latestStressFailureAt": latest_failure_ts,
            "latestRiskFlags": latest_risk_flags,
            "uniqueRiskFlags": sorted(risk_flag_set),
            "uniqueStressFailures": sorted(stress_failure_set),
            "strategies": sorted(strategy_set),
            "strategyCounts": dict(strategy_counter),
            "stressOverrideReasons": sorted(override_reason_set),
            "riskFlagCounts": dict(risk_flag_counter),
            "stressFailureCounts": dict(stress_failure_counter),
            "timelineStart": timeline[0]["timestamp"] if timeline else "",
            "timelineEnd": timeline[-1]["timestamp"] if timeline else "",
            "strategySummaries": summaries_payload,
        }
    )

    incomplete_samples = [
        {
            "event": item.get("event"),
            "timestamp": item.get("timestamp"),
            "missing": item.get("missingFields", []),
        }
        for item in timeline
        if item.get("isIncomplete")
    ][:5]
    incomplete_samples_count = len(incomplete_samples)
    metrics["incompleteSamples"] = incomplete_samples_count

    diagnostics = {
        "incompleteEntries": incomplete_entries,
        "incompleteSamples": incomplete_samples,
        "incompleteSamplesCount": incomplete_samples_count,
        "incomplete_entries": incomplete_entries,
        "incomplete_samples": incomplete_samples,
        "incomplete_samples_count": incomplete_samples_count,
        "riskFlagCounts": dict(risk_flag_counter),
        "risk_flag_counts": dict(risk_flag_counter),
    }

    return metrics, timeline, diagnostics


def _camelize_key_name(key: str) -> str:
    if not key:
        return ""
    if "_" not in key:
        return key
    parts = [segment for segment in key.split("_") if segment]
    if not parts:
        return key
    head, *tail = parts
    return head + "".join(segment.capitalize() for segment in tail)


def _camelize_mapping(payload: Mapping[str, object] | None) -> dict[str, object]:
    if not isinstance(payload, Mapping):
        return {}
    normalized: dict[str, object] = {}
    for key, value in payload.items():
        key_name = _camelize_key_name(str(key)) or str(key)
        if isinstance(value, Mapping):
            normalized[key_name] = _camelize_mapping(value)
            continue
        if isinstance(value, list):
            normalized[key_name] = [
                _camelize_mapping(item) if isinstance(item, Mapping) else item for item in value
            ]
            continue
        normalized[key_name] = value
    return normalized


def _clone_variant(value: object) -> object:
    return to_plain_value(value)


def _normalize_ai_snapshot(snapshot: Mapping[str, object] | None) -> dict[str, object]:
    if not isinstance(snapshot, Mapping):
        snapshot = {}
    last = _camelize_mapping(snapshot.get("last_decision"))
    history_payload = snapshot.get("history")
    history: list[dict[str, object]] = []
    if isinstance(history_payload, Iterable):
        for entry in history_payload:
            if isinstance(entry, Mapping):
                history.append(_camelize_mapping(entry))
    telemetry_payload = snapshot.get("telemetry")
    telemetry: dict[str, object] = {}
    if isinstance(telemetry_payload, Mapping):
        telemetry = {
            "riskMetrics": dict(telemetry_payload.get("riskMetrics", {}))
            if isinstance(telemetry_payload.get("riskMetrics"), Mapping)
            else {},
            "cycleMetrics": dict(telemetry_payload.get("cycleMetrics", {}))
            if isinstance(telemetry_payload.get("cycleMetrics"), Mapping)
            else {},
        }
        cycle_latency = _normalize_cycle_latency(
            telemetry_payload.get("cycleLatency") or telemetry_payload.get("cycle_latency")
        )
        if cycle_latency:
            telemetry["cycleLatency"] = cycle_latency
        mode_transitions = _normalize_mode_transitions(
            telemetry_payload.get("modeTransitions") or telemetry_payload.get("mode_transitions")
        )
        if mode_transitions:
            telemetry["modeTransitions"] = mode_transitions
        guardrails = _normalize_guardrail_snapshot(telemetry_payload.get("guardrails"))
        if guardrails:
            telemetry["guardrails"] = guardrails
    return {
        "lastDecision": last,
        "history": history,
        "telemetry": telemetry,
    }


def _coerce_mode_sequence(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        sanitized = [segment.strip() for segment in value.replace(";", ",").split(",")]
        return [segment for segment in sanitized if segment]
    if isinstance(value, Iterable):
        collected: list[str] = []
        for item in value:
            if isinstance(item, str):
                token = item.strip()
                if token:
                    collected.append(token)
        return collected
    return []


def _coerce_float(value: object) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalize_cycle_latency(payload: Mapping[str, object] | None) -> dict[str, float]:
    if not isinstance(payload, Mapping):
        return {}
    normalized: dict[str, float] = {}
    key_variants = {
        "lastMs": ("lastMs", "last_ms"),
        "p50Ms": ("p50Ms", "p50_ms"),
        "p95Ms": ("p95Ms", "p95_ms"),
        "sampleCount": ("sampleCount", "sample_count"),
    }
    for target, candidates in key_variants.items():
        value: object | None = None
        for candidate in candidates:
            if candidate in payload:
                value = payload[candidate]
                break
        numeric = _coerce_float(value)
        if numeric is not None:
            normalized[target] = numeric
    return normalized


def _normalize_signals(value: object) -> list[dict[str, object]]:
    if value is None:
        return []
    entries: Iterable[object] | None = None
    if isinstance(value, Mapping):
        candidate = value.get("signals")
        entries = candidate if isinstance(candidate, Iterable) else None
        if entries is None:
            candidate = value.get("entries")
            entries = candidate if isinstance(candidate, Iterable) else None
    elif isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
        entries = value
    if entries is None:
        return []
    normalized: list[dict[str, object]] = []
    for entry in entries:
        if not isinstance(entry, Mapping):
            continue
        name_value = entry.get("name") or entry.get("signal") or entry.get("feature")
        name = str(name_value).strip() if name_value is not None else ""
        weight = _coerce_float(entry.get("weight") or entry.get("importance"))
        value_numeric = _coerce_float(entry.get("value"))
        source = entry.get("source") or entry.get("origin")
        normalized.append(
            {
                "name": name,
                "weight": weight if weight is not None else 0.0,
                "value": value_numeric if value_numeric is not None else entry.get("value"),
                "source": str(source).strip() if source is not None else "",
            }
        )
    return normalized


def _normalize_mode_transitions(value: object) -> list[dict[str, object]]:
    entries: Iterable[object] | None = None
    if isinstance(value, Mapping):
        sequence = value.get("entries")
        if isinstance(sequence, Iterable) and not isinstance(sequence, (str, bytes)):
            entries = sequence
    elif isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
        entries = value
    if entries is None:
        return []
    normalized: list[dict[str, object]] = []
    for entry in entries:
        if not isinstance(entry, Mapping):
            continue
        mode_value = entry.get("mode")
        mode = str(mode_value).strip() if mode_value is not None else ""
        timestamp = entry.get("timestamp")
        timestamp_value = str(timestamp).strip() if timestamp is not None else ""
        regime_value = entry.get("regime")
        regime = str(regime_value).strip() if regime_value is not None else ""
        risk = _coerce_float(entry.get("risk")) or 0.0
        normalized.append(
            {
                "mode": mode,
                "timestamp": timestamp_value,
                "regime": regime,
                "risk": risk,
            }
        )
    return normalized


def _normalize_guardrail_snapshot(payload: Mapping[str, object] | None) -> dict[str, object]:
    if not isinstance(payload, Mapping):
        return {}
    active = bool(_normalize_bool(payload.get("active")))
    kill_switch = bool(_normalize_bool(payload.get("killSwitch") or payload.get("kill_switch")))
    snapshot: dict[str, object] = {
        "active": active,
        "killSwitch": kill_switch,
    }
    reasons = payload.get("reasons")
    if isinstance(reasons, Iterable) and not isinstance(reasons, (str, bytes)):
        snapshot["reasons"] = [str(reason) for reason in reasons if reason]
    recent_entries = payload.get("recent")
    recent_payload: list[dict[str, object]] = []
    if isinstance(recent_entries, Iterable) and not isinstance(recent_entries, (str, bytes)):
        for entry in recent_entries:
            if not isinstance(entry, Mapping):
                continue
            record: dict[str, object] = {}
            if "name" in entry:
                record["name"] = str(entry["name"])
            if "timestamp" in entry:
                record["timestamp"] = str(entry["timestamp"])
            age = _coerce_float(entry.get("ageSeconds") or entry.get("age_seconds"))
            if age is not None:
                record["ageSeconds"] = age
            if record:
                recent_payload.append(record)
    snapshot["recent"] = recent_payload
    return snapshot


def _normalize_ai_governor_record(payload: Mapping[str, object] | None) -> dict[str, object] | None:
    if not isinstance(payload, Mapping):
        return None
    record = {str(key): payload[key] for key in payload.keys()}
    mode = str(record.get("mode") or record.get("Mode") or "").strip()
    reason = str(record.get("reason") or record.get("Reason") or "").strip()
    timestamp = str(
        record.get("timestamp") or record.get("generated_at") or record.get("ts") or ""
    ).strip()
    if not mode and not reason:
        return None
    confidence_value = record.get("confidence") or record.get("Confidence")
    confidence = _coerce_float(confidence_value)
    regime_value = record.get("regime") or record.get("Regime") or ""
    if isinstance(regime_value, Mapping):
        regime = str(
            regime_value.get("label") or regime_value.get("value") or regime_value.get("name") or ""
        ).strip()
    else:
        regime = str(regime_value).strip()
    risk_score = _coerce_float(record.get("risk_score") or record.get("riskScore"))
    transaction_cost = _coerce_float(
        record.get("transaction_cost_bps")
        or record.get("transactionCostBps")
        or record.get("cost_bps")
    )
    recommended_modes = _coerce_mode_sequence(
        record.get("recommendedModes") or record.get("recommended_modes") or record.get("modes")
    )

    telemetry_payload = record.get("telemetry")
    risk_metrics: Mapping[str, object] | None = None
    cycle_metrics: Mapping[str, object] | None = None
    cycle_latency_payload: Mapping[str, object] | None = None
    transitions_payload: object | None = None
    guardrail_payload: Mapping[str, object] | None = None
    if isinstance(telemetry_payload, Mapping):
        risk_metrics = _camelize_mapping(telemetry_payload.get("riskMetrics"))
        cycle_metrics = _camelize_mapping(telemetry_payload.get("cycleMetrics"))
        cycle_latency_payload = telemetry_payload.get("cycleLatency") or telemetry_payload.get(
            "cycle_latency"
        )
        transitions_payload = telemetry_payload.get("modeTransitions") or telemetry_payload.get(
            "mode_transitions"
        )
        guardrail_payload = telemetry_payload.get("guardrails")
    else:
        risk_metrics = _camelize_mapping(record.get("riskMetrics"))
        cycle_metrics = _camelize_mapping(record.get("cycleMetrics"))
        cycle_latency_payload = record.get("cycleLatency") or record.get("cycle_latency")
        transitions_payload = record.get("modeTransitions") or record.get("mode_transitions")
        guardrail_payload = record.get("guardrails")

    decision_payload = record.get("decision") or record.get("Decision") or {}
    decision_signals = _normalize_signals(decision_payload or record.get("signals"))
    decision_state = ""
    decision_signal_label = None
    decision_should_trade: bool | None = None
    if isinstance(decision_payload, Mapping):
        decision_state_value = decision_payload.get("state") or decision_payload.get("outcome")
        decision_state = (
            str(decision_state_value).strip() if decision_state_value is not None else ""
        )
        decision_signal_label = decision_payload.get("signal") or decision_payload.get("label")
        decision_should_trade = _normalize_bool(
            decision_payload.get("shouldTrade") or decision_payload.get("should_trade")
        )
    elif "outcome" in record:
        decision_state = str(record.get("outcome") or "").strip()
    elif "decision_state" in record:
        decision_state = str(record.get("decision_state") or "").strip()

    normalized: dict[str, object] = {
        "timestamp": timestamp,
        "mode": mode,
        "reason": reason,
        "confidence": confidence if confidence is not None else 0.0,
        "regime": regime,
        "riskScore": risk_score if risk_score is not None else 0.0,
        "transactionCostBps": transaction_cost,
        "recommendedModes": recommended_modes,
    }
    telemetry: dict[str, object] = {}
    if risk_metrics:
        telemetry["riskMetrics"] = dict(risk_metrics)
    if cycle_metrics:
        telemetry["cycleMetrics"] = dict(cycle_metrics)
    cycle_latency = _normalize_cycle_latency(cycle_latency_payload)
    if cycle_latency:
        telemetry["cycleLatency"] = cycle_latency
    mode_transitions = _normalize_mode_transitions(transitions_payload)
    if mode_transitions:
        telemetry["modeTransitions"] = mode_transitions
    guardrails = _normalize_guardrail_snapshot(guardrail_payload)
    if guardrails:
        telemetry["guardrails"] = guardrails
    decision_meta: dict[str, object] = {}
    if decision_state:
        decision_meta["state"] = decision_state
    if decision_signal_label:
        decision_meta["signal"] = str(decision_signal_label)
    if decision_should_trade is not None:
        decision_meta["shouldTrade"] = decision_should_trade
    if decision_meta:
        normalized["decision"] = decision_meta
    if decision_signals:
        normalized["signals"] = decision_signals
    if telemetry:
        normalized["telemetry"] = telemetry
    return normalized


def _validate_ai_record_schema(payload: Mapping[str, object] | None) -> tuple[bool, str | None]:
    if not isinstance(payload, Mapping):
        return False, "payload not a mapping"
    timestamp = payload.get("timestamp") or payload.get("generated_at") or payload.get("ts")
    if not timestamp or not str(timestamp).strip():
        return False, "brak stempla czasu w rekordzie AI"
    mode_value = payload.get("mode") or payload.get("Mode")
    if mode_value is None or str(mode_value).strip() == "":
        return False, "brak trybu (mode) w rekordzie AI"
    telemetry = payload.get("telemetry")
    if telemetry is not None and not isinstance(telemetry, Mapping):
        return False, "telemetry nie jest mapą"
    if isinstance(telemetry, Mapping):
        cycle_latency = telemetry.get("cycleLatency") or telemetry.get("cycle_latency")
        if cycle_latency is not None and not isinstance(cycle_latency, Mapping):
            return False, "cycleLatency ma niepoprawny format"
    return True, None


class RuntimeService(QObject):
    """Zapewnia QML dostęp do najnowszych decyzji autotradera."""

    decisionsChanged = Signal()
    errorMessageChanged = Signal()
    liveSourceChanged = Signal()
    feedHealthChanged = Signal()
    feedSlaReportChanged = Signal()
    feedAlertHistoryChanged = Signal()
    feedAlertChannelsChanged = Signal()
    longPollMetricsChanged = Signal()
    retrainNextRunChanged = Signal()
    adaptiveStrategySummaryChanged = Signal()
    aiRegimeBreakdownChanged = Signal()
    regimeActivationSummaryChanged = Signal()
    riskMetricsChanged = Signal()
    riskTimelineChanged = Signal()
    operatorActionChanged = Signal()
    cycleMetricsChanged = Signal()
    feedTransportSnapshotChanged = Signal()
    aiGovernorSnapshotChanged = Signal()
    cloudRuntimeStatusChanged = Signal()
    executionModeChanged = Signal()
    guardrailsChanged = Signal()
    strategyConfigsChanged = Signal()
    riskControlsChanged = Signal()

    def __init__(
        self,
        *,
        journal: TradingDecisionJournal | None = None,
        decision_loader: DecisionLoader | None = None,
        ai_governor_loader: Callable[[], Mapping[str, object]] | None = None,
        parent: QObject | None = None,
        default_limit: int = 20,
        core_config_path: str | os.PathLike[str] | None = None,
        runtime_config_path: str | os.PathLike[str] | None = None,
        feed_alert_sink: UiTelemetryAlertSink | None = None,
        feed_metrics_exporter: FeedHealthMetricsExporter | None = None,
        cloud_runtime_enabled: bool = False,
        cloud_client_config_path: str | os.PathLike[str] | None = None,
        ai_runner_factory: Callable[[], AutoTraderAIGovernorRunner] | None = None,
        ai_runner: AutoTraderAIGovernorRunner | None = None,
        decision_log_repository: DecisionLogRepository | None = None,
    ) -> None:
        super().__init__(parent)
        if decision_loader is not None:
            self._loader: DecisionLoader = decision_loader
        elif journal is not None:
            self._loader = lambda limit: _load_from_journal(journal, limit)
        else:
            self._loader = _default_loader
        self._default_limit = max(1, int(default_limit))
        self._decisions: list[dict[str, object]] = []
        self._error_message = ""
        self._core_config_path = Path(core_config_path).expanduser() if core_config_path else None
        self._cached_core_config = None
        self._source_selector = DecisionSourceFallbackCoordinator()
        self._active_profile: str | None = None
        self._active_log_path: Path | None = None
        self._active_stream_label: str | None = None
        self._runtime_config_path = (
            Path(runtime_config_path).expanduser() if runtime_config_path else None
        )
        self._runtime_config_cache: "RuntimeAppConfig | None" = None
        self._cloud_runtime_enabled = bool(cloud_runtime_enabled)
        cloud_config_env = os.environ.get("BOT_CORE_UI_CLOUD_CLIENT_CONFIG")
        config_candidate = (
            cloud_client_config_path or cloud_config_env or "config/cloud/client.yaml"
        )
        self._cloud_client_config_path = Path(config_candidate).expanduser()
        self._cloud_client_options: CloudClientOptions | None = None
        self._cloud_client_mtime: float | None = None
        self._cloud_identity: LicenseIdentity | None = None
        self._cloud_handshake: CloudHandshakeResult | None = None
        self._cloud_session_token: str | None = None
        self._cloud_runtime_status: dict[str, object] = {
            "enabled": self._cloud_runtime_enabled,
            "status": "disabled" if not self._cloud_runtime_enabled else "initializing",
            "configPath": str(self._cloud_client_config_path),
            "target": None,
            "handshake": {},
        }
        self._ai_governor_loader = ai_governor_loader or build_demo_ai_governor_snapshot
        self._decision_log_repository = decision_log_repository or DecisionLogRepository()
        self._ai_governor_snapshot: dict[str, object] = {}
        self._retrain_next_run: str = ""
        self._adaptive_summary: str = ""
        self._regime_activation_summary: str = ""
        self._ai_regime_breakdown: list[dict[str, object]] = []
        self._risk_metrics: dict[str, object] = {}
        self._risk_timeline: list[dict[str, object]] = []
        self._cycle_metrics: dict[str, float] = {}
        self._last_operator_action: dict[str, object] | None = None
        self._operator_action_service = OperatorActionService(logger=_LOGGER)
        self._ai_runner_factory = ai_runner_factory
        self._ai_runner: AutoTraderAIGovernorRunner | None = ai_runner
        self._execution_mode: str = "manual"
        self._guardrails: dict[str, object] = {
            "maxExposure": 0.35,
            "dailyLossLimitPct": 0.03,
            "blockOnSlaAlerts": True,
        }
        config_base = (
            self._runtime_config_path.parent
            if self._runtime_config_path is not None
            else Path("config").resolve()
        )
        self._strategy_config_path = (config_base / "strategies.yaml").resolve()
        self._risk_controls_path = (config_base / "risk_controls.yaml").resolve()
        self._strategy_configs: list[dict[str, object]] = self._load_strategy_configs()
        self._risk_controls: dict[str, object] = self._load_risk_controls()
        self._grpc_thread: threading.Thread | None = None
        self._grpc_client: GrpcDecisionStreamClient | None = None
        self._grpc_stop_event: threading.Event | None = None
        self._grpc_queue: "queue.Queue[tuple[str, object]] | None" = None
        self._grpc_timer: QTimer | None = None
        self._grpc_stream_active = False
        self._grpc_target: str | None = None
        self._grpc_metadata: list[tuple[str, str]] = self._load_grpc_metadata()
        self._grpc_limit = self._default_limit
        self._grpc_retry_attempts = 0
        self._grpc_retry_limit = max(0, int(os.environ.get("BOT_CORE_UI_GRPC_RETRY_LIMIT", "3")))
        self._grpc_retry_base = max(
            0.1, float(os.environ.get("BOT_CORE_UI_GRPC_RETRY_BASE_SECONDS", "1.0"))
        )
        self._grpc_retry_multiplier = max(
            1.0, float(os.environ.get("BOT_CORE_UI_GRPC_RETRY_MULTIPLIER", "2.0"))
        )
        self._grpc_retry_max = max(
            self._grpc_retry_base,
            float(os.environ.get("BOT_CORE_UI_GRPC_RETRY_MAX_SECONDS", "15.0")),
        )
        self._grpc_ready_timeout = max(
            1.0, float(os.environ.get("BOT_CORE_UI_GRPC_READY_TIMEOUT", "5.0"))
        )
        self._grpc_idle_timeout = max(
            1.0, float(os.environ.get("BOT_CORE_UI_GRPC_IDLE_TIMEOUT", "3.0"))
        )
        self._grpc_reconnect_timer: QTimer | None = None
        self._active_grpc_metadata: list[tuple[str, str]] = list(self._grpc_metadata)
        self._grpc_ssl_credentials = None
        self._grpc_authority_override: str | None = None
        self._grpc_idle_flag = False
        self._last_grpc_update = time.monotonic()
        self._ai_feed_channel = _AI_FEED_CHANNEL
        self._ai_feed_queue: "queue.Queue[tuple[str, object]] | None" = None
        self._ai_feed_thread: threading.Thread | None = None
        self._ai_feed_stop_event: threading.Event | None = None
        self._ai_feed_timer: QTimer | None = None
        self._ai_feed_retry_timer: QTimer | None = None
        self._ai_feed_stream_active = False
        self._ai_feed_last_error = ""
        self._ai_feed_last_update = 0.0
        ai_history_limit_env = os.environ.get("BOT_CORE_UI_AI_HISTORY_LIMIT")
        try:
            configured_limit = (
                int(ai_history_limit_env) if ai_history_limit_env else _AI_HISTORY_LIMIT
            )
        except (TypeError, ValueError):
            configured_limit = _AI_HISTORY_LIMIT
        self._ai_history_limit = max(_AI_HISTORY_LIMIT, configured_limit)
        metrics_path_env = os.environ.get("BOT_CORE_UI_FEED_LATENCY_PATH")
        if metrics_path_env:
            self._feed_metrics_path = Path(metrics_path_env).expanduser()
        else:
            self._feed_metrics_path = Path("reports/ci/decision_feed_metrics.json")
        self._feed_channels: tuple[str, ...] = ("decision_journal", "ai_governor")
        self._feed_channel_status: dict[str, dict[str, object]] = {
            channel: {"status": "initializing", "lastError": ""} for channel in self._feed_channels
        }
        buffer_size = 1024
        self._feed_health_tracker = FeedHealthTracker(
            feed_channels=self._feed_channels,
            latency_buffer_size=buffer_size,
            percentile_fn=self._percentile,
        )
        self._feed_health = self._feed_health_tracker.feed_health
        self._feed_sla_report: dict[str, object] = self._feed_health_tracker.feed_sla_report
        self._feed_transport_snapshot: dict[str, object] = {
            "status": "initializing",
            "mode": "demo",
            "label": "",
            "reconnects": 0,
            "lastError": "",
            "channels": list(self._feed_channels),
            "channelStates": {
                channel: dict(state) for channel, state in self._feed_channel_status.items()
            },
        }
        resolved_feed_metrics = feed_metrics_exporter or get_feed_health_metrics_exporter()
        self._feed_metrics_exporter: FeedHealthMetricsExporter = resolved_feed_metrics
        self._risk_journal_metrics_exporter: RiskJournalMetricsExporter = (
            get_risk_journal_metrics_exporter()
        )
        self._feed_alert_sink_override = feed_alert_sink
        self._feed_alert_sink = None
        self._set_feed_alert_sink(
            feed_alert_sink if feed_alert_sink is not None else get_feed_health_alert_sink()
        )
        self._alert_manager = AlertManager(
            runtime_config_loader=self._load_runtime_config,
            active_profile_loader=lambda: self._active_profile or "default",
            sink_loader=self._effective_feed_alert_sink,
            risk_diagnostics_normalizer=_normalize_risk_journal_diagnostics,
            mapping_normalizer=_to_mapping,
            history_changed=self.feedAlertHistoryChanged.emit,
            channels_changed=self.feedAlertChannelsChanged.emit,
        )
        self._feed_thresholds = self._alert_manager.feed_thresholds
        self._feed_alert_state = self._alert_manager.feed_alert_state
        self._metrics_last_write = 0.0
        self._metrics_next_write = 0.0
        self._metrics_last_status = ""
        self._metrics_last_reconnects = -1
        self._longpoll_metrics_cache = get_long_poll_metrics_cache()
        self._longpoll_metrics: list[dict[str, object]] = []
        self._longpoll_timer = QTimer(self)
        self._longpoll_timer.setInterval(
            max(1000, int(os.environ.get("BOT_CORE_UI_LONGPOLL_METRICS_INTERVAL_MS", "5000")))
        )
        self._longpoll_timer.timeout.connect(self._refresh_long_poll_metrics)
        self._longpoll_timer.start()
        try:
            self._update_runtime_metadata(invalidate_cache=False)
        except Exception:  # pragma: no cover - defensywna inicjalizacja
            _LOGGER.debug("Nie udało się zainicjalizować metadanych runtime", exc_info=True)
        self._set_ai_governor_snapshot(self._load_ai_governor_snapshot())
        self._refresh_feed_transport_snapshot(self._feed_health, None)
        if self._cloud_runtime_enabled:
            try:
                self._refresh_cloud_handshake(force=False)
            except Exception:  # pragma: no cover - diagnostyka
                _LOGGER.debug(
                    "Nie udało się przeprowadzić początkowego handshake'u cloud", exc_info=True
                )
        self._auto_connect_grpc()
        self._refresh_long_poll_metrics()

    # ------------------------------------------------------------------
    @Property("QVariantList", notify=decisionsChanged)
    def decisions(self) -> list[dict[str, object]]:  # type: ignore[override]
        return to_plain_list(self._decisions)

    @Property(str, notify=errorMessageChanged)
    def errorMessage(self) -> str:  # type: ignore[override]
        return to_plain_text(self._error_message)

    @Property(str, notify=retrainNextRunChanged)
    def retrainNextRun(self) -> str:  # type: ignore[override]
        return to_plain_text(self._retrain_next_run)

    @Property(str, notify=adaptiveStrategySummaryChanged)
    def adaptiveStrategySummary(self) -> str:  # type: ignore[override]
        return to_plain_text(self._adaptive_summary)

    @Property("QVariantList", notify=aiRegimeBreakdownChanged)
    def aiRegimeBreakdown(self) -> list[dict[str, object]]:  # type: ignore[override]
        return to_plain_list(self._ai_regime_breakdown)

    @Property(str, notify=regimeActivationSummaryChanged)
    def regimeActivationSummary(self) -> str:  # type: ignore[override]
        return to_plain_text(self._regime_activation_summary)

    @Property("QVariantMap", notify=riskMetricsChanged)
    def riskMetrics(self) -> dict[str, object]:  # type: ignore[override]
        return to_plain_dict(self._risk_metrics)

    @Property("QVariantList", notify=riskTimelineChanged)
    def riskTimeline(self) -> list[dict[str, object]]:  # type: ignore[override]
        return to_plain_list(self._risk_timeline)

    @Property("QVariantMap", notify=cycleMetricsChanged)
    def cycleMetrics(self) -> dict[str, object]:  # type: ignore[override]
        return to_plain_dict({key: float(value) for key, value in self._cycle_metrics.items()})

    @Property(str, notify=executionModeChanged)
    def executionMode(self) -> str:  # type: ignore[override]
        return to_plain_text(self._execution_mode)

    @Property("QVariantMap", notify=guardrailsChanged)
    def guardrails(self) -> dict[str, object]:  # type: ignore[override]
        return to_plain_dict(self._guardrails)

    @Property("QVariantList", notify=strategyConfigsChanged)
    def strategyConfigs(self) -> list[dict[str, object]]:  # type: ignore[override]
        return to_plain_list(self._strategy_configs)

    @Property("QVariantMap", notify=riskControlsChanged)
    def riskControls(self) -> dict[str, object]:  # type: ignore[override]
        return to_plain_dict(self._risk_controls)

    @Property("QVariantMap", notify=operatorActionChanged)
    def lastOperatorAction(self) -> dict[str, object]:  # type: ignore[override]
        if self._last_operator_action is None:
            return {}
        return to_plain_dict(self._last_operator_action)

    @Property("QVariantMap", notify=feedHealthChanged)
    def feedHealth(self) -> dict[str, object]:  # type: ignore[override]
        return to_plain_dict(self._feed_health)

    @Property("QVariantMap", notify=feedSlaReportChanged)
    def feedSlaReport(self) -> dict[str, object]:  # type: ignore[override]
        return to_plain_dict(self._feed_sla_report)

    @Property("QVariantList", notify=feedAlertHistoryChanged)
    def feedAlertHistory(self) -> list[dict[str, object]]:  # type: ignore[override]
        return to_plain_list(self._alert_manager.feed_alert_history)

    @Property("QVariantList", notify=feedAlertChannelsChanged)
    def feedAlertChannels(self) -> list[dict[str, object]]:  # type: ignore[override]
        return to_plain_list(self._alert_manager.feed_alert_channels)

    @Property("QVariantMap", notify=feedTransportSnapshotChanged)
    def feedTransportSnapshot(self) -> dict[str, object]:  # type: ignore[override]
        return to_plain_dict(self._feed_transport_snapshot)

    @property
    def _feed_reconnects(self) -> int:
        return self._feed_health_tracker.reconnects

    @_feed_reconnects.setter
    def _feed_reconnects(self, value: int) -> None:
        self._feed_health_tracker.reconnects = value

    @property
    def _feed_downtime_started(self) -> float | None:
        return self._feed_health_tracker.downtime_started

    @_feed_downtime_started.setter
    def _feed_downtime_started(self, value: float | None) -> None:
        self._feed_health_tracker.downtime_started = value

    @property
    def _feed_downtime_total(self) -> float:
        return self._feed_health_tracker.downtime_total

    @_feed_downtime_total.setter
    def _feed_downtime_total(self, value: float) -> None:
        self._feed_health_tracker.downtime_total = value

    @property
    def _feed_last_error(self) -> str:
        return self._feed_health_tracker.last_error

    @_feed_last_error.setter
    def _feed_last_error(self, value: str) -> None:
        self._feed_health_tracker.last_error = value

    @Property("QVariantMap", notify=aiGovernorSnapshotChanged)
    def aiGovernorSnapshot(self) -> dict[str, object]:  # type: ignore[override]
        snapshot = _clone_variant(self._ai_governor_snapshot)
        if isinstance(snapshot, Mapping):
            return dict(snapshot)
        return {}

    @Property("QVariantList", notify=longPollMetricsChanged)
    def longPollMetrics(self) -> list[dict[str, object]]:  # type: ignore[override]
        return to_plain_list(self._longpoll_metrics)

    @Property("QVariantMap", notify=cloudRuntimeStatusChanged)
    def cloudRuntimeStatus(self) -> dict[str, object]:  # type: ignore[override]
        return to_plain_dict(self._cloud_runtime_status)

    @Slot(result=bool)
    def refreshCloudHandshake(self) -> bool:
        """Pozwala QML-owi ręcznie odświeżyć handshake."""

        return self._refresh_cloud_handshake(force=True)

    @Slot(result="QVariantList")
    def loadStrategyConfigs(self) -> list[dict[str, object]]:
        self._strategy_configs = self._load_strategy_configs()
        self.strategyConfigsChanged.emit()
        return self.strategyConfigs

    @Slot(str, "QVariantMap", result="QVariantMap")
    def saveStrategyConfig(
        self, strategy_id: str, payload: Mapping[str, object]
    ) -> dict[str, object]:
        payload = to_plain_dict(payload)
        result = self._sanitize_strategy_config(strategy_id, payload)
        if not result["success"]:
            return result
        sanitized = result["strategy"]
        existing_ids = [entry.get("id") for entry in self._strategy_configs]
        updated: list[dict[str, object]] = []
        replaced = False
        for entry in self._strategy_configs:
            if entry.get("id") == sanitized["id"]:
                updated.append(deepcopy(sanitized))
                replaced = True
            else:
                updated.append(deepcopy(entry))
        if not replaced:
            updated.append(deepcopy(sanitized))
        self._strategy_configs = updated
        self._persist_strategy_configs()
        self.strategyConfigsChanged.emit()
        message = (
            "Zaktualizowano strategię" if strategy_id in existing_ids else "Dodano nową strategię"
        )
        return {"success": True, "message": message, "strategy": deepcopy(sanitized)}

    @Slot(result="QVariantMap")
    def loadRiskControls(self) -> dict[str, object]:
        self._risk_controls = self._load_risk_controls()
        self.riskControlsChanged.emit()
        return self.riskControls

    @Slot("QVariantMap", result="QVariantMap")
    def saveRiskControls(self, payload: Mapping[str, object]) -> dict[str, object]:
        sanitized = self._sanitize_risk_controls(to_plain_dict(payload))
        self._risk_controls = sanitized
        self._persist_risk_controls()
        self.riskControlsChanged.emit()
        return {
            "success": True,
            "message": "Zapisano limity ryzyka",
            "riskControls": deepcopy(sanitized),
        }

    def _update_feed_health(
        self,
        *,
        status: str | None = None,
        reconnects: int | None = None,
        last_error: str | None = None,
        next_retry: float | None = None,
        latest_latency: float | None = None,
    ) -> None:
        transport_key = self._current_transport_key()
        payload, latency_p95, latency_p50 = self._feed_health_tracker.update_feed_health(
            status=status,
            reconnects=reconnects,
            last_error=last_error,
            next_retry=next_retry,
            latest_latency=latest_latency,
            transport_key=transport_key,
            channel_status=self._feed_channel_status,
        )
        self._feed_health = payload
        self._export_feed_transport_metrics(payload["transports"])
        self.feedHealthChanged.emit()
        self._refresh_feed_transport_snapshot(payload, latency_p95)
        self._publish_feed_metrics(payload, latency_p95, latency_p50)
        self._evaluate_feed_health_alerts(payload, latency_p95)
        self._write_feed_metrics()

    def _refresh_feed_transport_snapshot(
        self, payload: Mapping[str, object], latency_p95: float | None
    ) -> None:
        self._sync_source_selector_from_runtime_state()
        label = self._source_selector.current_feed_label()
        mode = self._source_selector.current_feed_mode()
        adapter = self._snapshot_feed_adapter_label(mode)
        latency_value = latency_p95
        if latency_value is None:
            candidate = payload.get("p95_ms")
            try:
                latency_value = float(candidate) if candidate is not None else None
            except (TypeError, ValueError):
                latency_value = None
        snapshot: dict[str, object] = {
            "status": str(payload.get("status", self._feed_health.get("status", "unknown"))),
            "mode": mode,
            "adapter": adapter,
            "label": label,
            "reconnects": int(payload.get("reconnects", self._feed_reconnects)),
            "nextRetrySeconds": payload.get("nextRetrySeconds"),
            "lastError": str(payload.get("lastError", self._feed_last_error)),
            "latencyP95": latency_value,
            "idle": bool(self._grpc_idle_flag),
            "retryAttempts": self._grpc_retry_attempts,
            "queueDepth": self._grpc_queue.qsize() if self._grpc_queue is not None else 0,
            "channels": list(self._feed_channels),
            "channelStates": {
                channel: dict(state) for channel, state in self._feed_channel_status.items()
            },
            "transports": self._serialize_transport_stats(),
        }
        if self._grpc_stream_active:
            snapshot["secondsSinceLastMessage"] = max(
                0.0, time.monotonic() - self._last_grpc_update
            )
        else:
            snapshot["secondsSinceLastMessage"] = None
        if self._cloud_runtime_enabled:
            snapshot["cloud"] = {
                "target": self._cloud_runtime_status.get("target"),
                "status": self._cloud_runtime_status.get("status"),
                "handshake": dict(self._cloud_runtime_status.get("handshake") or {}),
            }
        if snapshot != self._feed_transport_snapshot:
            self._feed_transport_snapshot = snapshot
            self.feedTransportSnapshotChanged.emit()

    @staticmethod
    def _snapshot_feed_adapter_label(mode: str) -> str:
        if mode == "grpc":
            return "grpc"
        if mode == "file":
            return "jsonl"
        if mode == "demo":
            return "demo"
        return "unknown"

    def _set_channel_status(
        self,
        channel: str,
        status: str,
        *,
        last_error: str | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> None:
        payload = dict(self._feed_channel_status.get(channel, {}))
        payload["status"] = status
        if last_error is not None:
            payload["lastError"] = last_error
        if metadata:
            for key, value in metadata.items():
                payload[str(key)] = value
        self._feed_channel_status[channel] = payload
        self._update_feed_health()

    def _publish_feed_metrics(
        self,
        payload: Mapping[str, object],
        latency_p95: float | None,
        latency_p50: float | None,
    ) -> None:
        exporter = self._feed_metrics_exporter
        if exporter is None:
            return
        status = str(payload.get("status", "unknown"))
        adapter = self._current_feed_adapter_label(status)
        reconnects_raw = payload.get("reconnects", 0)
        downtime_raw = payload.get("downtimeMs", 0.0)
        last_error = str(payload.get("lastError", self._feed_last_error))
        try:
            reconnects_value = int(reconnects_raw)
        except (TypeError, ValueError):
            reconnects_value = 0
        try:
            downtime_ms = float(downtime_raw)
        except (TypeError, ValueError):
            downtime_ms = 0.0
        metric_labels = {
            "transport": self._current_transport_key(),
            "environment": self._active_profile or "default",
            "scope": "decision_feed",
        }
        exporter.record(
            adapter=adapter,
            status=status,
            latency_p50_ms=latency_p50,
            latency_p95_ms=latency_p95,
            reconnects=reconnects_value,
            downtime_ms=downtime_ms,
            last_error=last_error,
            labels=metric_labels,
        )

    def _evaluate_feed_health_alerts(
        self,
        payload: Mapping[str, object],
        latency_p95: float | None,
    ) -> None:
        status = str(payload.get("status", "unknown"))
        adapter = self._current_feed_adapter_label(status)
        last_error = str(payload.get("lastError", self._feed_last_error))
        self._alert_manager.evaluate_feed_health_alerts(
            payload=payload,
            latency_p95=latency_p95,
            adapter=adapter,
            last_error=last_error,
        )

    def _current_feed_adapter_label(self, status: str) -> str:
        self._sync_source_selector_from_runtime_state()
        return self._source_selector.current_feed_adapter_label(
            status=status,
            loader_is_demo=self._loader is _default_loader,
        )

    def _current_transport_key(self) -> str:
        self._sync_source_selector_from_runtime_state()
        return self._source_selector.current_transport_key(
            grpc_stream_active=self._grpc_stream_active
        )

    def _latency_samples_for(self, key: str | None) -> deque[float]:
        return self._feed_health_tracker.latency_samples_for(key)

    def _serialize_transport_stats(self) -> dict[str, dict[str, object]]:
        return self._feed_health_tracker.serialize_transport_stats()

    def _export_feed_transport_metrics(
        self, transports: Mapping[str, Mapping[str, object]]
    ) -> None:
        exporter = self._feed_metrics_exporter
        environment_label = self._active_profile or "default"
        for name, stats in transports.items():
            if not isinstance(stats, Mapping):
                continue
            try:
                reconnects_val = int(stats.get("reconnects", 0) or 0)
            except (TypeError, ValueError):
                reconnects_val = 0
            try:
                downtime_ms = float(stats.get("downtimeMs", 0.0) or 0.0)
            except (TypeError, ValueError):
                downtime_ms = 0.0
            latency_p50 = stats.get("p50_ms")
            latency_p95 = stats.get("p95_ms")
            exporter.record(
                adapter=str(name),
                status=str(stats.get("status", "unknown")),
                latency_p50_ms=float(latency_p50) if latency_p50 is not None else None,
                latency_p95_ms=float(latency_p95) if latency_p95 is not None else None,
                reconnects=reconnects_val,
                downtime_ms=downtime_ms,
                last_error=str(stats.get("lastError", "")),
                labels={"transport": str(name), "environment": environment_label},
            )

    @staticmethod
    def _classify_threshold(
        value: float | None,
        *,
        warning: float | None,
        critical: float | None,
    ) -> str:
        return FeedHealthTracker.classify_threshold(value, warning=warning, critical=critical)

    @staticmethod
    def _aggregate_sla_state(states: Iterable[str]) -> str:
        return FeedHealthTracker.aggregate_sla_state(states)

    def _maybe_emit_feed_alert(
        self,
        key: str,
        severity: str,
        *,
        channel: str = "decision_journal",
        metric_label: str,
        unit: str,
        value: float | None,
        warning: float | None,
        critical: float | None,
        status: str,
        adapter: str,
        reconnects: int,
        downtime_seconds: float,
        latency_p95: float | None,
        last_error: str,
    ) -> None:
        self._alert_manager.maybe_emit_feed_alert(
            key,
            severity,
            channel=channel,
            metric_label=metric_label,
            unit=unit,
            value=value,
            warning=warning,
            critical=critical,
            status=status,
            adapter=adapter,
            reconnects=reconnects,
            downtime_seconds=downtime_seconds,
            latency_p95=latency_p95,
            last_error=last_error,
        )

    @staticmethod
    def _format_feed_metric(value: float | None, unit: str) -> str:
        return AlertManager.format_feed_metric(value, unit)

    def _maybe_emit_risk_journal_alert(self, diagnostics: Mapping[str, object]) -> None:
        self._alert_manager.maybe_emit_risk_journal_alert(
            diagnostics=diagnostics,
            logger_warning=lambda body: _LOGGER.warning("%s", body),
            logger_info=lambda body: _LOGGER.info("%s", body),
            metrics_record=self._risk_journal_metrics_exporter.record,
        )

    def _load_feed_thresholds(self) -> dict[str, float | None]:
        refreshed = self._alert_manager.reload_feed_thresholds()
        self._feed_thresholds = refreshed
        return refreshed

    def _record_feed_alert(
        self,
        *,
        severity: str,
        state: str,
        metric: str,
        label: str,
        unit: str,
        value: float | None,
        warning: float | None,
        critical: float | None,
        adapter: str,
        status: str,
        reconnects: int,
        downtime_seconds: float,
        latency_p95: float | None,
        last_error: str,
        router: object | None,
    ) -> None:
        self._alert_manager.record_feed_alert(
            severity=severity,
            state=state,
            metric=metric,
            label=label,
            unit=unit,
            value=value,
            warning=warning,
            critical=critical,
            adapter=adapter,
            status=status,
            reconnects=reconnects,
            downtime_seconds=downtime_seconds,
            latency_p95=latency_p95,
            last_error=last_error,
            router=router,
        )

    def _refresh_alert_channels(self, router: object | None) -> None:
        self._alert_manager.refresh_alert_channels(router)

    def _mark_feed_disconnected(self) -> None:
        self._feed_health_tracker.mark_feed_disconnected()
        self._set_channel_status("decision_journal", "degraded", last_error=self._feed_last_error)
        self._update_feed_health(status="degraded", reconnects=self._feed_reconnects)

    def _mark_feed_connected(self) -> None:
        self._feed_health_tracker.mark_feed_connected()
        self._grpc_retry_attempts = 0
        self._cancel_grpc_reconnect()
        self._set_channel_status("decision_journal", "connected", last_error="")
        self._update_feed_health(
            status="connected", reconnects=self._feed_reconnects, last_error=""
        )

    # ------------------------------------------------------------------
    def _refresh_long_poll_metrics(self) -> None:
        try:
            snapshot = self._longpoll_metrics_cache.snapshot()
        except Exception:  # pragma: no cover - defensywne zbieranie metryk
            _LOGGER.debug("Nie udało się odczytać metryk long-pollowych", exc_info=True)
            return
        if snapshot != self._longpoll_metrics:
            self._longpoll_metrics = [dict(entry) for entry in snapshot]
            self.longPollMetricsChanged.emit()

    def _update_cycle_metrics(self, metrics: Mapping[str, object] | None) -> None:
        normalized: dict[str, float] = {}
        if isinstance(metrics, Mapping):
            for key, value in metrics.items():
                try:
                    normalized[str(key)] = float(value)
                except (TypeError, ValueError):
                    continue
        if normalized != self._cycle_metrics:
            self._cycle_metrics = normalized
            self.cycleMetricsChanged.emit()
            if normalized:
                self._update_ai_governor_telemetry(cycle_metrics=normalized)
        self._refresh_activation_summary()

    def _load_ai_governor_snapshot(self) -> dict[str, object]:
        loader = self._ai_governor_loader
        if not callable(loader):
            return _normalize_ai_snapshot({})
        try:
            raw_snapshot = loader()
        except Exception:  # pragma: no cover - loader diagnostyczny
            _LOGGER.debug("Nie udało się pobrać snapshotu AI Governora", exc_info=True)
            raw_snapshot = {}
        return _normalize_ai_snapshot(raw_snapshot if isinstance(raw_snapshot, Mapping) else {})

    def _set_ai_governor_snapshot(self, snapshot: Mapping[str, object]) -> None:
        normalized = _normalize_ai_snapshot(snapshot)
        history = normalized.get("history")
        if isinstance(history, list) and history:
            trimmed = history[: self._ai_history_limit]
            normalized["history"] = trimmed
            normalized["lastDecision"] = trimmed[0]
        elif "history" in normalized and not history:
            normalized["history"] = []
        if normalized != self._ai_governor_snapshot:
            self._ai_governor_snapshot = normalized
            self.aiGovernorSnapshotChanged.emit()

    def _update_ai_governor_telemetry(
        self,
        *,
        cycle_metrics: Mapping[str, float] | None = None,
        risk_metrics: Mapping[str, float] | None = None,
        extra: Mapping[str, object] | None = None,
    ) -> None:
        if not cycle_metrics and not risk_metrics:
            return
        telemetry = dict(self._ai_governor_snapshot.get("telemetry", {}))
        updated = False
        if cycle_metrics:
            telemetry["cycleMetrics"] = {
                str(key): float(value) for key, value in cycle_metrics.items()
            }
            updated = True
        if risk_metrics:
            telemetry["riskMetrics"] = {
                str(key): float(value) for key, value in risk_metrics.items()
            }
            updated = True
        if extra:
            telemetry.update(_clone_variant(extra))
            updated = True
        if not updated:
            return
        snapshot = dict(self._ai_governor_snapshot)
        snapshot["telemetry"] = telemetry
        self._ai_governor_snapshot = snapshot
        self.aiGovernorSnapshotChanged.emit()

    def _coerce_metadata_mapping(self, value: object) -> dict[str, object] | None:
        plain = to_plain_value(value)
        if isinstance(plain, Mapping):
            return {str(key): plain[key] for key in plain.keys()}
        if isinstance(plain, str):
            text = plain.strip()
            if not text:
                return None
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                return None
            if isinstance(parsed, Mapping):
                return {str(key): parsed[key] for key in parsed.keys()}
        return None

    def _refresh_activation_summary(self) -> None:
        summary = self._build_activation_summary()
        if summary == self._regime_activation_summary:
            return
        self._regime_activation_summary = summary
        self.regimeActivationSummaryChanged.emit()

    def _build_activation_summary(self) -> str:
        timeline: list[dict[str, object]] = []
        guardrail_trace: list[dict[str, object]] = []
        active_preset: Mapping[str, object] | None = None
        for entry in self._decisions:
            metadata = entry.get("metadata") if isinstance(entry, Mapping) else None
            if not isinstance(metadata, Mapping):
                continue
            activation_block = self._coerce_metadata_mapping(metadata.get("activation"))
            if not activation_block:
                continue
            record: dict[str, object] = {
                "timestamp": entry.get("timestamp"),
                "event": entry.get("event"),
                "status": entry.get("status"),
                "activation": activation_block,
            }
            guardrail_block = self._coerce_metadata_mapping(metadata.get("guardrail_transition"))
            if guardrail_block:
                record["guardrails"] = guardrail_block
                guardrail_trace.append(guardrail_block)
            timeline.append(record)
            if active_preset is None:
                active_preset = activation_block
            if len(timeline) >= 15:
                break
        if not timeline and not guardrail_trace and not active_preset:
            return ""
        payload = {
            "activations": timeline,
            "activePreset": active_preset or {},
            "guardrailTrace": guardrail_trace,
        }
        return json.dumps(payload, ensure_ascii=False)

    # ------------------------------------------------------------------
    @Slot(int, result="QVariantList")
    def loadRecentDecisions(self, limit: int = 0) -> list[dict[str, object]]:  # type: ignore[override]
        """Pobiera najnowsze decyzje z dziennika."""

        size = int(limit)
        if size <= 0:
            size = self._default_limit
        if self._grpc_stream_active:
            subset = self._decisions[:size]
            return [dict(entry) for entry in subset]
        try:
            raw_entries = list(self._loader(size))
        except Exception as exc:  # pragma: no cover - diagnostyka
            self._error_message = str(exc)
            self.errorMessageChanged.emit()
            self._risk_metrics = {}
            self._risk_timeline = []
            self.riskMetricsChanged.emit()
            self.riskTimelineChanged.emit()
            return []

        self._error_message = ""
        self.errorMessageChanged.emit()

        parsed: list[dict[str, object]] = []
        for record in raw_entries:
            entry = _parse_entry(record)
            payload = entry.to_payload()
            parsed.append(payload)
        self._decisions = parsed
        self.decisionsChanged.emit()
        self._apply_risk_context(parsed)
        self._update_runtime_metadata(invalidate_cache=False)
        self._update_cycle_metrics({})
        return to_plain_list(self._decisions)

    @Slot(result="QVariantMap")
    def reloadAiGovernorSnapshot(self) -> dict[str, object]:
        """Odświeża snapshot rekomendacji AI Governora."""

        snapshot = self._load_ai_governor_snapshot()
        self._set_ai_governor_snapshot(snapshot)
        return self.aiGovernorSnapshot

    @Slot()
    def refreshRuntimeMetadata(self) -> None:  # type: ignore[override]
        """Wymusza ponowne wczytanie metadanych retrainingu i presetów adaptacyjnych."""

        self._update_runtime_metadata(invalidate_cache=True)

    # ------------------------------------------------------------------
    @Property(str, notify=liveSourceChanged)
    def activeDecisionLogPath(self) -> str:  # type: ignore[override]
        self._sync_source_selector_from_runtime_state()
        return to_plain_text(self._source_selector.active_decision_log_path())

    @Slot(str, result=bool)
    def attachToLiveDecisionLog(self, profile: str = "") -> bool:  # type: ignore[override]
        """Przełącza loader na rzeczywisty decision log skonfigurowany w core.yaml."""

        self._stop_grpc_stream()
        sanitized_profile = profile.strip()
        profile_value = sanitized_profile or None
        target = self._resolve_grpc_target(profile_value)
        if target:
            try:
                self._start_grpc_stream(target, self._default_limit)
            except Exception as exc:  # pragma: no cover - diagnostyka
                _LOGGER.debug("attachToLiveDecisionLog gRPC failed", exc_info=True)
                self._active_profile = profile_value
                return self._handle_grpc_error(str(exc), profile=profile_value, silent=False)
            else:
                self._loader = lambda limit: []
                self._finalize_grpc_activation(profile=profile_value, target=target)
                return True

        if self._activate_jsonl_loader(profile_value, silent=False):
            return True
        return False

    # ------------------------------------------------------------------ operator actions --
    @Slot(result=bool)
    @Slot("QVariantMap", result=bool)
    @Slot("QVariant", result=bool)
    def requestFreeze(self, entry: object = None) -> bool:  # type: ignore[override]
        return self._record_operator_action("freeze", entry)

    @Slot("QVariantMap", result=bool)
    @Slot("QVariant", result=bool)
    def requestUnfreeze(self, entry: object = None) -> bool:  # type: ignore[override]
        return self._record_operator_action("unfreeze", entry)

    @Slot("QVariantMap", result=bool)
    @Slot("QVariant", result=bool)
    def requestUnblock(self, entry: object = None) -> bool:  # type: ignore[override]
        return self._record_operator_action("unblock", entry)

    @Slot("QString", result=bool)
    @Slot("QString", "QVariantMap", result=bool)
    @Slot("QString", "QVariant", result=bool)
    @Slot(str, "QVariantMap", result=bool)
    @Slot(str, "QVariant", result=bool)
    @Slot("QVariant", "QVariantMap", result=bool)
    @Slot("QVariant", "QVariant", result=bool)
    def triggerOperatorAction(self, action: object, entry: object = None) -> bool:  # type: ignore[override]
        return self._record_operator_action(action, entry)

    # ------------------------------------------------------------------
    def _build_live_loader(self, profile: str | None) -> tuple[DecisionLoader, Path]:
        core_config = self._load_core_config()
        configured_path, _kwargs = resolve_decision_log_config(core_config)
        if configured_path is None:
            raise FileNotFoundError(
                "Decision log portfela nie jest skonfigurowany w pliku core.yaml"
            )

        log_path = Path(configured_path)
        if not log_path.is_absolute():
            config_path = self._resolve_core_config_path()
            if config_path is not None:
                log_path = (config_path.parent / log_path).resolve()
            else:
                log_path = log_path.expanduser().resolve()

        if not log_path.exists():
            raise FileNotFoundError(
                f"Decision log '{log_path}' nie istnieje – uruchom autotradera, aby utworzyć plik"
            )
        if not log_path.is_file():
            raise IsADirectoryError(
                f"Decision log '{log_path}' wskazuje na katalog – oczekiwany plik JSONL"
            )

        loader = self._build_jsonl_loader(log_path)
        return loader, log_path

    def _build_jsonl_loader(self, log_path: Path) -> DecisionLoader:
        def _loader(limit: int) -> Iterable[DecisionRecord]:
            return self._decision_log_repository.load_jsonl_entries(log_path, limit)

        return _loader

    # ------------------------------------------------------------------ risk aggregation helpers --
    def _resolve_env_grpc_target(self) -> str | None:
        candidates = (
            os.environ.get("BOT_CORE_UI_GRPC_ENDPOINT"),
            os.environ.get("BOT_CORE_TRADING_GRPC_ADDRESS"),
            os.environ.get("BOT_CORE_RUNTIME_GRPC_ADDRESS"),
        )
        for candidate in candidates:
            if candidate and candidate.strip():
                return candidate.strip()
        return None

    def _resolve_grpc_target(self, profile: str | None) -> str | None:
        prepared = self._prepare_grpc_connection(profile)
        if prepared is None:
            return None
        target, metadata = prepared
        self._active_grpc_metadata = metadata
        self._grpc_target = target
        return target

    def _prepare_grpc_connection(
        self, profile: str | None
    ) -> tuple[str, list[tuple[str, str]]] | None:
        target = self._resolve_env_grpc_target()
        if target:
            self._grpc_ssl_credentials = None
            self._grpc_authority_override = None
            self._update_cloud_status(status="env_override", target=target)
            return target, list(self._grpc_metadata)

        options = self._load_cloud_client_options()
        if options is None:
            return None
        self._grpc_ssl_credentials = options.tls_credentials
        self._grpc_authority_override = options.authority_override
        if not options.client.auto_connect:
            self._update_cloud_status(status="auto_connect_disabled")
            return None
        self._refresh_cloud_handshake(force=False)
        metadata = self._cloud_metadata_with_session(options)
        return options.client.address, metadata

    def _load_grpc_metadata(self) -> list[tuple[str, str]]:
        raw = os.environ.get("BOT_CORE_UI_GRPC_METADATA", "")
        if not raw:
            return []
        metadata: list[tuple[str, str]] = []
        for item in raw.split(","):
            if not item:
                continue
            if "=" not in item:
                continue
            key, value = item.split("=", 1)
            key = key.strip().lower()
            value = value.strip()
            if key:
                metadata.append((key, value))
        return metadata

    def _update_cloud_status(self, **updates: object) -> None:
        if not self._cloud_runtime_enabled:
            return
        payload = dict(self._cloud_runtime_status)
        handshake_payload = dict(payload.get("handshake") or {})
        handshake_update = updates.pop("handshake", None)
        if isinstance(handshake_update, Mapping):
            handshake_payload.update(
                {key: value for key, value in handshake_update.items() if value is not None}
            )
            payload["handshake"] = handshake_payload
        payload.update({key: value for key, value in updates.items() if value is not None})
        if payload != self._cloud_runtime_status:
            self._cloud_runtime_status = payload
            self.cloudRuntimeStatusChanged.emit()

    def _load_cloud_client_options(self, *, invalidate: bool = False) -> CloudClientOptions | None:
        if not self._cloud_runtime_enabled:
            return None
        path = self._cloud_client_config_path
        try:
            mtime = path.stat().st_mtime
        except OSError as exc:
            self._cloud_client_options = None
            self._cloud_client_mtime = None
            self._update_cloud_status(status="config_missing", target=None, error=str(exc))
            return None
        if (
            not invalidate
            and self._cloud_client_options is not None
            and self._cloud_client_mtime == mtime
        ):
            return self._cloud_client_options
        try:
            options = load_cloud_client_options(path, base_metadata=self._grpc_metadata)
        except Exception as exc:  # pragma: no cover - diagnostyka
            self._cloud_client_options = None
            self._cloud_client_mtime = None
            self._update_cloud_status(status="config_error", target=None, error=str(exc))
            return None
        self._cloud_client_options = options
        self._cloud_client_mtime = mtime
        self._update_cloud_status(
            status="configured",
            target=options.client.address,
            allowLocalFallback=bool(options.client.allow_local_fallback),
            autoConnect=bool(options.client.auto_connect),
            tlsEnabled=bool(options.tls_credentials is not None),
        )
        return options

    def _handshake_valid(self, result: CloudHandshakeResult | None) -> bool:
        if result is None or result.status != "ok" or not result.session_token:
            return False
        if result.expires_at is None:
            return True
        return result.expires_at - datetime.now(timezone.utc) > timedelta(seconds=30)

    def _refresh_cloud_handshake(self, *, force: bool) -> bool:
        if not self._cloud_runtime_enabled:
            return False
        options = self._load_cloud_client_options()
        if options is None:
            return False
        if self._cloud_identity is None or force:
            self._cloud_identity = load_license_identity()
        identity = self._cloud_identity
        if identity is None:
            self._cloud_handshake = None
            self._cloud_session_token = None
            self._update_cloud_status(
                status="identity_missing",
                handshake={
                    "status": "license_missing",
                    "message": "Brak aktywnej licencji lub fingerprintu",
                },
            )
            return False
        if not force and self._handshake_valid(self._cloud_handshake):
            return True
        try:
            result = perform_cloud_handshake(options, identity)
        except Exception as exc:  # pragma: no cover - diagnostyka
            self._cloud_handshake = None
            self._cloud_session_token = None
            self._update_cloud_status(handshake={"status": "error", "message": str(exc)})
            return False
        self._cloud_handshake = result
        if result.status == "ok" and result.session_token:
            self._cloud_session_token = result.session_token
        else:
            self._cloud_session_token = None
        handshake_payload = {
            "status": result.status,
            "message": result.message,
            "licenseId": result.license_id,
            "fingerprint": result.fingerprint,
            "expiresAt": result.expires_at.isoformat() if result.expires_at else None,
            "lastCheckedAt": datetime.now(timezone.utc).isoformat(),
        }
        self._update_cloud_status(handshake=handshake_payload)
        return result.status == "ok"

    def _cloud_metadata_with_session(self, options: CloudClientOptions) -> list[tuple[str, str]]:
        metadata = list(options.metadata)
        if self._cloud_session_token:
            metadata.append(("authorization", f"CloudSession {self._cloud_session_token}"))
        return metadata

    def _auto_connect_grpc(self) -> None:
        endpoint = self._prepare_grpc_connection(self._active_profile)
        if not endpoint:
            return
        target, metadata = endpoint
        self._active_grpc_metadata = metadata
        try:
            self._start_grpc_stream(target, self._default_limit)
        except Exception as exc:  # pragma: no cover - defensywne logowanie
            _LOGGER.debug("Auto gRPC bootstrap failed", exc_info=True)
            self._handle_grpc_error(str(exc), profile=self._active_profile, silent=True)
        else:
            self._finalize_grpc_activation(profile=self._active_profile, target=target)

    def _finalize_grpc_activation(self, *, profile: str | None, target: str) -> None:
        self._activate_source_state(
            self._source_selector.activate_grpc(profile=profile, target=target)
        )
        self._error_message = ""
        self.errorMessageChanged.emit()
        self.liveSourceChanged.emit()
        self._feed_reconnects = 0
        self._feed_last_error = ""
        self._mark_feed_disconnected()
        self._update_feed_health(status="connecting", reconnects=0, last_error="")

    def _activate_jsonl_loader(self, profile: str | None, *, silent: bool) -> bool:
        self._stop_ai_governor_stream()
        try:
            loader, log_path = self._build_live_loader(profile)
        except Exception as exc:
            if not silent:
                self._error_message = str(exc)
                self.errorMessageChanged.emit()
            return False

        self._loader = loader
        self._activate_source_state(
            self._source_selector.activate_jsonl(profile=profile, log_path=log_path)
        )
        self._grpc_stream_active = False
        self._update_cycle_metrics({})
        self.liveSourceChanged.emit()
        self.loadRecentDecisions(self._default_limit)
        if not silent:
            self._error_message = ""
            self.errorMessageChanged.emit()
        return True

    def _use_demo_loader(self, message: str | None, *, profile: str | None, silent: bool) -> None:
        self._stop_ai_governor_stream()
        self._loader = _default_loader
        self._activate_source_state(self._source_selector.activate_demo(profile=profile))
        self._grpc_stream_active = False
        self._update_cycle_metrics({})
        self.liveSourceChanged.emit()
        self.loadRecentDecisions(self._default_limit)
        if not silent:
            self._error_message = message or ""
            self.errorMessageChanged.emit()

    def _handle_grpc_error(self, message: str, *, profile: str | None, silent: bool) -> bool:
        self._feed_last_error = message
        self._stop_grpc_stream()
        self._grpc_retry_attempts = 0
        jsonl_activated = self._activate_jsonl_loader(profile, silent=True)
        fallback_source = self._source_selector.fallback_source(jsonl_available=jsonl_activated)
        if fallback_source == "jsonl":
            if not silent:
                self._error_message = message
                self.errorMessageChanged.emit()
            self._update_feed_health(
                status="fallback",
                reconnects=self._feed_reconnects,
                last_error=message,
            )
            self._schedule_grpc_reconnect(profile)
            return True
        self._use_demo_loader(message if not silent else None, profile=profile, silent=silent)
        self._update_feed_health(
            status="fallback",
            reconnects=self._feed_reconnects,
            last_error=message if not silent else self._feed_last_error,
        )
        self._schedule_grpc_reconnect(profile)
        return True

    def _ensure_grpc_timer(self) -> None:
        if self._grpc_timer is not None:
            return
        timer = QTimer(self)
        timer.setInterval(200)
        timer.setSingleShot(False)
        timer.timeout.connect(self._drain_grpc_queue)
        timer.start()
        self._grpc_timer = timer

    def _start_grpc_stream(self, target: str, limit: int) -> None:
        if grpc is None:
            raise RuntimeError("Pakiet grpcio jest wymagany do połączenia z RuntimeService.")

        def _load_grpc_stubs() -> tuple[object, object]:
            try:
                from bot_core.generated import trading_pb2, trading_pb2_grpc
            except ImportError as exc:  # pragma: no cover - brak stubów
                raise RuntimeError(
                    "Brak wygenerowanych stubów trading_pb2*_grpc – uruchom scripts/generate_trading_stubs.py"
                ) from exc
            return trading_pb2, trading_pb2_grpc

        self._grpc_limit = max(1, int(limit))
        self._grpc_target = target
        self._grpc_stream_active = True
        self._decisions = []
        self._update_cycle_metrics({})
        client = GrpcDecisionStreamClient(
            target=target,
            metadata=tuple(self._active_grpc_metadata),
            ssl_credentials=self._grpc_ssl_credentials,
            authority_override=self._grpc_authority_override,
            limit=self._grpc_limit,
            ready_timeout=self._grpc_ready_timeout,
            retry_base=self._grpc_retry_base,
            retry_multiplier=self._grpc_retry_multiplier,
            retry_max=self._grpc_retry_max,
            cycle_metrics_serializer=self._serialize_cycle_metrics,
            grpc_module=grpc,
            stubs_loader=_load_grpc_stubs,
        )
        self._grpc_client = client
        self._grpc_queue = client.events_queue
        self._grpc_stop_event = client.stop_event
        self._ensure_grpc_timer()
        client.start()
        self._grpc_thread = client.thread
        self._start_ai_governor_stream()

    def _start_ai_governor_stream(self) -> None:
        if self._ai_feed_stream_active:
            return
        target = self._grpc_target
        if not target:
            return
        if grpc is None:
            self._set_channel_status(
                self._ai_feed_channel, "unavailable", last_error="grpc unavailable"
            )
            return
        try:
            from bot_core.generated import trading_pb2, trading_pb2_grpc
        except ImportError:
            self._set_channel_status(
                self._ai_feed_channel,
                "unavailable",
                last_error="trading stubs unavailable",
            )
            return
        self._ai_feed_stop_event = threading.Event()
        self._ai_feed_queue = queue.Queue(maxsize=64)
        metadata = list(self._active_grpc_metadata)
        metadata.append(("x-feed-channel", self._ai_feed_channel))
        metadata.append(("x-ai-governor", "1"))
        worker = threading.Thread(
            target=self._ai_feed_worker,
            args=(
                target,
                trading_pb2,
                trading_pb2_grpc,
                tuple(metadata),
                max(1, self._ai_history_limit),
            ),
            name="RuntimeServiceAiFeed",
            daemon=True,
        )
        worker.start()
        self._ai_feed_thread = worker
        self._ai_feed_stream_active = True
        self._set_channel_status(self._ai_feed_channel, "connecting", last_error="")
        self._ensure_ai_feed_timer()

    def _stop_ai_governor_stream(self) -> None:
        if self._ai_feed_stop_event is not None:
            self._ai_feed_stop_event.set()
        if self._ai_feed_thread is not None and self._ai_feed_thread.is_alive():
            self._ai_feed_thread.join(timeout=1.0)
        self._ai_feed_thread = None
        self._ai_feed_stop_event = None
        self._ai_feed_queue = None
        if self._ai_feed_timer is not None:
            self._ai_feed_timer.stop()
            self._ai_feed_timer.deleteLater()
            self._ai_feed_timer = None
        if self._ai_feed_retry_timer is not None:
            self._ai_feed_retry_timer.stop()
            self._ai_feed_retry_timer.deleteLater()
            self._ai_feed_retry_timer = None
        self._ai_feed_stream_active = False
        self._set_channel_status(
            self._ai_feed_channel, "offline", last_error=self._ai_feed_last_error
        )

    def _ensure_ai_feed_timer(self) -> None:
        if self._ai_feed_timer is not None:
            return
        timer = QTimer(self)
        timer.setInterval(200)
        timer.setSingleShot(False)
        timer.timeout.connect(self._drain_ai_feed_queue)
        timer.start()
        self._ai_feed_timer = timer

    def _schedule_ai_feed_retry(self, delay_seconds: float = 5.0) -> None:
        if self._ai_feed_retry_timer is None:
            timer = QTimer(self)
            timer.setSingleShot(True)
            timer.timeout.connect(self._start_ai_governor_stream)
            self._ai_feed_retry_timer = timer
        self._ai_feed_retry_timer.start(max(1000, int(delay_seconds * 1000)))

    def _emit_ai_feed_event(self, severity: str, message: str) -> None:
        sink = self._effective_feed_alert_sink()
        emit = getattr(sink, "emit_feed_health_event", None) if sink is not None else None
        if not callable(emit):
            return
        context = {"adapter": "grpc", "channel": self._ai_feed_channel}
        payload = {
            "adapter": "grpc",
            "channel": self._ai_feed_channel,
            "state": severity,
            "last_error": message,
        }
        emit(
            severity=severity,
            title="AI Governor feed status",
            body=message,
            context=context,
            payload=payload,
        )

    def _effective_feed_alert_sink(self) -> object | None:
        if self._feed_alert_sink_override is not None:
            return self._feed_alert_sink_override
        return self._feed_alert_sink

    def _set_feed_alert_sink(self, sink: object | None) -> None:
        if self._feed_alert_sink_override is not None:
            self._feed_alert_sink = self._feed_alert_sink_override
            return
        self._feed_alert_sink = sink

    def _ai_feed_worker(
        self,
        target: str,
        trading_pb2,
        trading_pb2_grpc,
        metadata: tuple[tuple[str, str], ...],
        limit: int,
    ) -> None:
        queue_obj = self._ai_feed_queue
        if queue_obj is None:
            return
        stop_event = self._ai_feed_stop_event
        request = trading_pb2.StreamDecisionsRequest(
            limit=max(1, int(limit)),
            skip_snapshot=False,
            poll_interval_seconds=5.0,
        )
        channel = None
        try:
            if self._grpc_ssl_credentials is not None:
                options = []
                if self._grpc_authority_override:
                    options.append(("grpc.ssl_target_name_override", self._grpc_authority_override))
                channel = grpc.secure_channel(
                    target, self._grpc_ssl_credentials, options=options or None
                )
            else:
                channel = grpc.insecure_channel(target)
            ready_future = grpc.channel_ready_future(channel)
            ready_future.result(timeout=max(1.0, float(self._grpc_ready_timeout)))
            stub = trading_pb2_grpc.RuntimeServiceStub(channel)
            stream = stub.StreamDecisions(request, metadata=metadata)
            queue_obj.put(("connected", None))
            for update in stream:
                if stop_event is not None and stop_event.is_set():
                    break
                if update.HasField("snapshot"):
                    records = [dict(entry.fields) for entry in update.snapshot.records]
                    queue_obj.put(("snapshot", records))
                elif update.HasField("increment"):
                    queue_obj.put(("increment", dict(update.increment.record.fields)))
        except Exception as exc:  # pragma: no cover - diagnostyka
            queue_obj.put(("connection-error", {"message": str(exc)}))
        finally:
            if channel is not None:
                try:
                    channel.close()
                except Exception:
                    pass
        queue_obj.put(("done", None))

    def _drain_ai_feed_queue(self) -> None:
        queue_obj = self._ai_feed_queue
        if queue_obj is None:
            return
        updated = False
        while True:
            try:
                kind, payload = queue_obj.get_nowait()
            except queue.Empty:
                break
            if kind == "snapshot":
                records = payload if isinstance(payload, list) else None
                if records:
                    self._apply_ai_governor_records(records)
                    updated = True
                queue_obj.task_done()
                continue
            if kind == "increment":
                record_payload = payload if isinstance(payload, Mapping) else None
                if record_payload:
                    self._append_ai_governor_record(record_payload)
                    updated = True
                queue_obj.task_done()
                continue
            if kind == "connected":
                self._ai_feed_last_error = ""
                self._set_channel_status(self._ai_feed_channel, "connected", last_error="")
                self._ai_feed_last_update = time.monotonic()
                queue_obj.task_done()
                continue
            if kind == "connection-error":
                message = "AI Governor feed error"
                if isinstance(payload, Mapping):
                    message = str(payload.get("message", message))
                self._ai_feed_last_error = message
                self._set_channel_status(self._ai_feed_channel, "fallback", last_error=message)
                self._emit_ai_feed_event("warning", message)
                self._schedule_ai_feed_retry()
                queue_obj.task_done()
                continue
            if kind == "done":
                self._ai_feed_stream_active = False
                queue_obj.task_done()
                continue
            queue_obj.task_done()
        if updated:
            self._ai_feed_last_update = time.monotonic()

    def _apply_ai_governor_records(self, records: Iterable[Mapping[str, object]]) -> None:
        normalized: list[dict[str, object]] = []
        for record in records:
            ok, reason = _validate_ai_record_schema(record)
            if not ok:
                message = f"AI feed schema mismatch: {reason or 'nieznany błąd'}"
                self._emit_ai_feed_event("warning", message)
                continue
            entry = _normalize_ai_governor_record(record)
            if entry:
                normalized.append(entry)
        if not normalized:
            return
        existing_history = []
        history_payload = self._ai_governor_snapshot.get("history")
        if isinstance(history_payload, list):
            existing_history = list(history_payload)
        merged = normalized + existing_history
        snapshot = {
            "last_decision": normalized[0],
            "history": merged,
            "telemetry": self._ai_governor_snapshot.get("telemetry", {}),
        }
        telemetry_payload = normalized[0].get("telemetry")
        if isinstance(telemetry_payload, Mapping):
            snapshot["telemetry"] = telemetry_payload
        self._set_ai_governor_snapshot(snapshot)

    def _append_ai_governor_record(self, record: Mapping[str, object]) -> None:
        ok, reason = _validate_ai_record_schema(record)
        if not ok:
            message = f"AI feed schema mismatch: {reason or 'nieznany błąd'}"
            self._emit_ai_feed_event("warning", message)
            self._schedule_ai_feed_retry(delay_seconds=2.0)
            return
        entry = _normalize_ai_governor_record(record)
        if not entry:
            return
        history_payload = self._ai_governor_snapshot.get("history")
        history = [entry]
        if isinstance(history_payload, list):
            history.extend(history_payload)
        snapshot = {
            "last_decision": entry,
            "history": history,
            "telemetry": self._ai_governor_snapshot.get("telemetry", {}),
        }
        telemetry_payload = entry.get("telemetry")
        if isinstance(telemetry_payload, Mapping):
            snapshot["telemetry"] = telemetry_payload
        self._set_ai_governor_snapshot(snapshot)

    def _drain_grpc_queue(self) -> None:
        queue_obj = self._grpc_queue
        if queue_obj is None:
            return
        updated = False
        fallback_reason: str | None = None
        while True:
            try:
                kind, payload = queue_obj.get_nowait()
            except queue.Empty:
                break
            if kind == "snapshot":
                records: Iterable[Mapping[str, str]] | None = None
                metrics_payload: Mapping[str, object] | None = None
                if isinstance(payload, Mapping):
                    maybe_records = payload.get("records")
                    if isinstance(maybe_records, list):
                        records = maybe_records
                    maybe_metrics = payload.get("metrics")
                    if isinstance(maybe_metrics, Mapping):
                        metrics_payload = maybe_metrics
                elif isinstance(payload, list):
                    records = payload
                if records is not None:
                    self._apply_grpc_snapshot(records, metrics_payload)
                    updated = True
                queue_obj.task_done()
                continue
            if kind == "increment":
                record_payload: Mapping[str, str] | None = None
                metrics_payload: Mapping[str, object] | None = None
                if isinstance(payload, Mapping):
                    candidate = payload.get("record")
                    if isinstance(candidate, Mapping):
                        record_payload = candidate  # type: ignore[assignment]
                    else:
                        record_payload = payload  # type: ignore[assignment]
                    maybe_metrics = payload.get("metrics")
                    if isinstance(maybe_metrics, Mapping):
                        metrics_payload = maybe_metrics
                if record_payload is not None:
                    self._append_grpc_record(record_payload, metrics_payload)
                    updated = True
                queue_obj.task_done()
                continue
            if kind == "connected":
                attempt = 0
                if isinstance(payload, Mapping):
                    attempt = int(payload.get("attempt", 0))
                if attempt > 1:
                    self._feed_reconnects = max(self._feed_reconnects, attempt - 1)
                else:
                    self._feed_reconnects = max(self._feed_reconnects, 0)
                self._grpc_stream_active = True
                self._feed_last_error = ""
                self._mark_feed_connected()
                queue_obj.task_done()
                continue
            if kind == "retrying":
                self._mark_feed_disconnected()
                if isinstance(payload, Mapping):
                    next_retry_seconds = float(payload.get("sleep", 0.0))
                    self._update_feed_health(
                        reconnects=self._feed_reconnects,
                        status="retrying",
                        last_error=self._feed_last_error,
                        next_retry=next_retry_seconds,
                    )
                else:
                    self._update_feed_health(
                        reconnects=self._feed_reconnects,
                        status="retrying",
                        last_error=self._feed_last_error,
                    )
                queue_obj.task_done()
                continue
            if kind == "connection-error":
                message = "Nieudane połączenie gRPC"
                if isinstance(payload, Mapping):
                    message = str(payload.get("message", message))
                self._feed_last_error = message
                self._grpc_stream_active = False
                self._grpc_retry_attempts += 1
                self._mark_feed_disconnected()
                self._update_feed_health(
                    reconnects=self._feed_reconnects,
                    status="retrying",
                    last_error=message,
                )
                self._error_message = message
                self.errorMessageChanged.emit()
                if (
                    self._grpc_retry_limit == 0
                    or self._grpc_retry_attempts > self._grpc_retry_limit
                ):
                    fallback_reason = message
                queue_obj.task_done()
                continue
            if kind == "stream-ended":
                message = "Strumień gRPC zakończony"
                if isinstance(payload, Mapping):
                    message = str(payload.get("message", message))
                self._feed_last_error = message
                self._grpc_stream_active = False
                self._grpc_retry_attempts += 1
                self._mark_feed_disconnected()
                self._update_feed_health(
                    reconnects=self._feed_reconnects,
                    status="retrying",
                    last_error=message,
                )
                if (
                    self._grpc_retry_limit == 0
                    or self._grpc_retry_attempts > self._grpc_retry_limit
                ):
                    fallback_reason = message
                queue_obj.task_done()
                continue
            if kind == "done":
                if self._grpc_stop_event is None or not self._grpc_stop_event.is_set():
                    self._grpc_stream_active = False
                queue_obj.task_done()
                continue
            queue_obj.task_done()

        self._maybe_handle_grpc_idle()

        if fallback_reason:
            self._handle_grpc_error(fallback_reason, profile=self._active_profile, silent=False)
            return

        if updated:
            self.decisionsChanged.emit()
            self._apply_risk_context(self._decisions)
            self._update_runtime_metadata(invalidate_cache=False)
            self._error_message = ""
            self.errorMessageChanged.emit()

    def _maybe_handle_grpc_idle(self) -> None:
        if not self._grpc_stream_active:
            self._grpc_idle_flag = False
            return
        now = time.monotonic()
        if now - self._last_grpc_update <= self._grpc_idle_timeout:
            self._grpc_idle_flag = False
            return
        if self._grpc_idle_flag:
            return
        self._grpc_idle_flag = True
        message = "Brak aktualizacji strumienia gRPC"
        self._feed_last_error = message
        self._mark_feed_disconnected()
        self._update_feed_health(
            status="retrying",
            reconnects=self._feed_reconnects,
            last_error=message,
        )
        self._handle_grpc_error(message, profile=self._active_profile, silent=True)

    def _apply_grpc_snapshot(
        self,
        records: Iterable[Mapping[str, str]],
        cycle_metrics: Mapping[str, object] | None = None,
    ) -> None:
        collected: list[dict[str, object]] = []
        for record in reversed(list(records)):
            if not isinstance(record, Mapping):
                continue
            try:
                entry = _parse_entry(record)
            except Exception:  # pragma: no cover - diagnostyka
                _LOGGER.debug("Nie udało się sparsować snapshotu gRPC", exc_info=True)
                continue
            payload = entry.to_payload()
            collected.append(payload)
            self._record_feed_latency(record)
        if not collected:
            return
        max_size = max(1, self._default_limit)
        if len(collected) > max_size:
            collected = collected[:max_size]
        self._decisions = collected
        self._update_cycle_metrics(cycle_metrics)

    def _append_grpc_record(
        self,
        record: Mapping[str, str],
        cycle_metrics: Mapping[str, object] | None = None,
    ) -> None:
        try:
            entry = _parse_entry(record)
        except Exception:  # pragma: no cover - diagnostyka
            _LOGGER.debug("Nie udało się sparsować przyrostu gRPC", exc_info=True)
            return
        payload = entry.to_payload()
        self._record_feed_latency(record)
        self._decisions.insert(0, payload)
        if len(self._decisions) > self._default_limit:
            self._decisions = self._decisions[: self._default_limit]
        self._update_cycle_metrics(cycle_metrics)

    def _record_feed_latency(self, record: Mapping[str, str]) -> None:
        latency_value: float | None = None
        latency_field = record.get("latency_ms") or record.get("latencyMs")
        if latency_field is not None:
            try:
                latency_value = float(latency_field)
            except (TypeError, ValueError):
                latency_value = None
        if latency_value is None:
            timestamp_raw = record.get("timestamp")
            if not timestamp_raw:
                return
            text = str(timestamp_raw).strip()
            if not text:
                return
            try:
                parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
            except ValueError:
                return
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            latency_ms = (
                datetime.now(timezone.utc) - parsed.astimezone(timezone.utc)
            ).total_seconds() * 1000.0
            if latency_ms < 0:
                latency_ms = 0.0
            latency_value = float(latency_ms)
        if not math.isfinite(latency_value):
            return
        if latency_value < 0.0:
            latency_value = 0.0
        self._last_grpc_update = time.monotonic()
        self._grpc_idle_flag = False
        self._latency_samples_for("grpc").append(latency_value)
        self._mark_feed_connected()
        self._update_feed_health(
            latest_latency=latency_value, reconnects=self._feed_reconnects, last_error=""
        )

    def _write_feed_metrics(self, *, force: bool = False) -> None:
        source_key = self._current_transport_key()
        stats_payload = self._feed_health_tracker.build_sla_report(
            transport_source=source_key,
            thresholds=self._feed_thresholds,
        )
        self._feed_sla_report = stats_payload
        self.feedSlaReportChanged.emit()
        now = time.monotonic()
        status_value = self._feed_health.get("status", "unknown")
        reconnects_value = self._feed_reconnects
        status_changed = status_value != self._metrics_last_status
        reconnects_changed = reconnects_value != self._metrics_last_reconnects
        if (
            not force
            and not status_changed
            and not reconnects_changed
            and now < self._metrics_next_write
        ):
            return
        try:
            self._feed_metrics_path.parent.mkdir(parents=True, exist_ok=True)
            serialized = json.dumps(stats_payload, ensure_ascii=False, indent=2)
            tmp_path: Path | None = None
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                delete=False,
                dir=str(self._feed_metrics_path.parent),
            ) as handle:
                handle.write(serialized)
                handle.flush()
                try:
                    os.fsync(handle.fileno())
                except OSError:
                    pass
                tmp_path = Path(handle.name)
            if tmp_path is None:
                raise RuntimeError("Nie udało się zapisać metryk feedu – brak ścieżki tymczasowej")
            try:
                tmp_path.replace(self._feed_metrics_path)
            except Exception:
                with contextlib.suppress(OSError):
                    tmp_path.unlink()
                raise
        except Exception:  # pragma: no cover - zapisy metryk nie powinny blokować UI
            _LOGGER.debug("Nie udało się zapisać metryk latencji feedu", exc_info=True)
        else:
            self._metrics_last_write = now
            self._metrics_next_write = now + 0.25
            self._metrics_last_status = str(status_value)
            self._metrics_last_reconnects = reconnects_value

    @staticmethod
    def _serialize_cycle_metrics(cycle_metrics_message) -> dict[str, float]:
        if cycle_metrics_message is None:
            return {}
        payload: dict[str, float] = {}
        values_container = getattr(cycle_metrics_message, "values", None)
        if isinstance(values_container, Mapping):
            for key, value in values_container.items():
                try:
                    payload[str(key)] = float(value)
                except (TypeError, ValueError):
                    continue
        has_field = getattr(cycle_metrics_message, "HasField", None)
        for field_name in ("cycle_latency_p50_ms", "cycle_latency_p95_ms"):
            include_field = False
            if callable(has_field):
                try:
                    include_field = bool(has_field(field_name))
                except ValueError:
                    include_field = True
            else:
                include_field = True
            if not include_field:
                continue
            field_value = getattr(cycle_metrics_message, field_name, None)
            if field_value is None:
                continue
            try:
                payload[field_name] = float(field_value)
            except (TypeError, ValueError):
                continue
        return payload

    def _cancel_grpc_reconnect(self) -> None:
        timer = self._grpc_reconnect_timer
        if timer is None:
            return
        try:
            timer.stop()
        except RuntimeError:
            pass
        try:
            timer.deleteLater()
        except RuntimeError:
            pass
        self._grpc_reconnect_timer = None

    def _schedule_grpc_reconnect(self, profile: str | None) -> None:
        target = self._resolve_grpc_target(profile)
        if not target:
            return
        timer = self._grpc_reconnect_timer
        if timer is not None and timer.isActive():
            return
        interval_ms = max(500, int(self._grpc_retry_base * 1000.0))

        def _attempt() -> None:
            self._grpc_reconnect_timer = None
            self._auto_connect_grpc()

        reconnect_timer = QTimer(self)
        reconnect_timer.setSingleShot(True)
        reconnect_timer.timeout.connect(_attempt)
        reconnect_timer.start(interval_ms)
        self._grpc_reconnect_timer = reconnect_timer

    @staticmethod
    def _percentile(values: Iterable[float], percentile: float) -> float:
        data = sorted(values)
        if not data:
            return 0.0
        if percentile <= 0:
            return data[0]
        if percentile >= 100:
            return data[-1]
        rank = (percentile / 100.0) * (len(data) - 1)
        low = math.floor(rank)
        high = math.ceil(rank)
        if low == high:
            return data[int(rank)]
        fraction = rank - low
        return data[low] + (data[high] - data[low]) * fraction

    def _stop_grpc_stream(self) -> None:
        self._stop_ai_governor_stream()
        if self._grpc_client is not None:
            self._grpc_client.stop()
        elif self._grpc_stop_event is not None:
            self._grpc_stop_event.set()
        if self._grpc_thread is not None and self._grpc_thread.is_alive():
            self._grpc_thread.join(timeout=1.5)
        self._grpc_client = None
        self._grpc_thread = None
        self._grpc_stop_event = None
        self._grpc_queue = None
        if self._grpc_timer is not None:
            self._grpc_timer.stop()
            self._grpc_timer.deleteLater()
            self._grpc_timer = None
        self._grpc_stream_active = False
        if self._active_stream_label and self._active_stream_label.startswith("grpc://"):
            self._activate_source_state(
                self._source_selector.set_state(
                    profile=self._active_profile,
                    log_path=self._active_log_path,
                    stream_label=None,
                )
            )
        self._cancel_grpc_reconnect()

    def _activate_source_state(self, state) -> None:
        self._active_profile = state.profile
        self._active_log_path = state.log_path
        self._active_stream_label = state.stream_label

    def _sync_source_selector_from_runtime_state(self) -> None:
        self._source_selector.set_state(
            profile=self._active_profile,
            log_path=self._active_log_path,
            stream_label=self._active_stream_label,
        )

    def _apply_risk_context(self, entries: Iterable[Mapping[str, object]]) -> None:
        metrics, timeline, diagnostics = _build_risk_context(entries)
        self._risk_metrics = dict(metrics)
        self._risk_timeline = [dict(item) for item in timeline]
        self._maybe_emit_risk_journal_alert(diagnostics)
        self.riskMetricsChanged.emit()
        self.riskTimelineChanged.emit()

    def _update_ai_snapshot_from_runner(self, runner: AutoTraderAIGovernorRunner | None) -> None:
        if runner is None:
            return
        snapshot = runner.snapshot()
        self._set_ai_governor_snapshot(snapshot if isinstance(snapshot, Mapping) else {})

    def _build_ai_runner(self) -> AutoTraderAIGovernorRunner:
        thresholds = DecisionOrchestratorThresholds(
            max_cost_bps=18.0,
            min_net_edge_bps=5.0,
            max_daily_loss_pct=float(self._guardrails.get("dailyLossLimitPct", 0.03)),
            max_drawdown_pct=0.08,
            max_position_ratio=float(self._guardrails.get("maxExposure", 0.35)),
            max_open_positions=6,
            max_latency_ms=320.0,
        )
        orchestrator = DecisionOrchestrator(
            DecisionEngineConfig(
                orchestrator=thresholds,
                profile_overrides={},
                stress_tests=None,
                min_probability=0.55,
                require_cost_data=False,
                penalty_cost_bps=0.0,
            )
        )
        return AutoTraderAIGovernorRunner(
            orchestrator,
            governor=AutoTraderAIGovernor(history_limit=self._ai_history_limit),
        )

    def _ensure_ai_runner(self) -> AutoTraderAIGovernorRunner | None:
        if self._ai_runner is not None:
            return self._ai_runner
        if callable(self._ai_runner_factory):
            try:
                self._ai_runner = self._ai_runner_factory()
            except Exception:  # pragma: no cover - diagnostyka fabryki
                _LOGGER.debug("Fabryka runnera AI zgłosiła wyjątek", exc_info=True)
        if self._ai_runner is None:
            self._ai_runner = self._build_ai_runner()
        return self._ai_runner

    @staticmethod
    def _safe_float(value: object) -> float | None:
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _load_strategy_configs(self) -> list[dict[str, object]]:
        _require_yaml()
        try:
            if self._strategy_config_path.exists():
                with self._strategy_config_path.open("r", encoding="utf-8") as handle:
                    payload = yaml.safe_load(handle) or []
                    if isinstance(payload, list):
                        sanitized: list[dict[str, object]] = []
                        for entry in payload:
                            if isinstance(entry, Mapping):
                                normalized = self._sanitize_strategy_config(
                                    entry.get("id", ""), entry
                                )
                                if normalized.get("success"):
                                    sanitized.append(normalized["strategy"])
                        if sanitized:
                            return sanitized
        except Exception:  # pragma: no cover - diagnostyka środowiska
            _LOGGER.debug("Nie udało się wczytać strategies.yaml", exc_info=True)
        return self._default_strategy_configs()

    def _persist_strategy_configs(self) -> None:
        _require_yaml()
        try:
            self._strategy_config_path.parent.mkdir(parents=True, exist_ok=True)
            with self._strategy_config_path.open("w", encoding="utf-8") as handle:
                yaml.safe_dump(self._strategy_configs, handle, sort_keys=False, allow_unicode=True)
        except OSError:  # pragma: no cover - diagnostyka zapisu
            _LOGGER.debug("Nie udało się zapisać strategies.yaml", exc_info=True)

    def _default_strategy_configs(self) -> list[dict[str, object]]:
        return [
            {
                "id": "grid_usdt",
                "name": "Grid USDT",
                "mode": "grid",
                "profile": "balanced",
                "params": {
                    "exchange": "binance",
                    "symbol": "BTC/USDT",
                    "gridLevels": 7,
                    "takeProfitPct": 1.2,
                    "stopLossPct": 2.5,
                },
            },
            {
                "id": "dca_eth",
                "name": "DCA ETH",
                "mode": "dca",
                "profile": "conservative",
                "params": {
                    "exchange": "kraken",
                    "symbol": "ETH/USD",
                    "baseOrder": 150.0,
                    "safetyOrder": 120.0,
                    "maxSafetyOrders": 4,
                    "takeProfitPct": 1.0,
                    "stopLossPct": 3.0,
                },
            },
        ]

    def _sanitize_strategy_config(
        self, strategy_id: str, payload: Mapping[str, object]
    ) -> dict[str, object]:
        identifier = str(strategy_id or payload.get("id") or "").strip()
        if not identifier:
            return {
                "success": False,
                "message": "Identyfikator strategii jest wymagany",
                "strategy": {},
            }
        name = str(payload.get("name") or identifier).strip()
        mode = str(payload.get("mode") or "custom").strip()
        profile = str(payload.get("profile") or "balanced").strip()
        params_raw = payload.get("params")
        params: dict[str, object] = {}
        if isinstance(params_raw, Mapping):
            for key, value in params_raw.items():
                key_str = str(key)
                if isinstance(value, (int, float)):
                    params[key_str] = float(value)
                else:
                    params[key_str] = value
        strategy = {
            "id": identifier,
            "name": name,
            "mode": mode,
            "profile": profile,
            "params": params,
        }
        return {"success": True, "strategy": strategy}

    def _load_risk_controls(self) -> dict[str, object]:
        _require_yaml()
        try:
            if self._risk_controls_path.exists():
                with self._risk_controls_path.open("r", encoding="utf-8") as handle:
                    payload = yaml.safe_load(handle) or {}
                    if isinstance(payload, Mapping):
                        return self._sanitize_risk_controls(payload)
        except Exception:  # pragma: no cover - diagnostyka środowiska
            _LOGGER.debug("Nie udało się wczytać risk_controls.yaml", exc_info=True)
        return self._default_risk_controls()

    def _persist_risk_controls(self) -> None:
        _require_yaml()
        try:
            self._risk_controls_path.parent.mkdir(parents=True, exist_ok=True)
            with self._risk_controls_path.open("w", encoding="utf-8") as handle:
                yaml.safe_dump(self._risk_controls, handle, sort_keys=False, allow_unicode=True)
        except OSError:  # pragma: no cover - diagnostyka zapisu
            _LOGGER.debug("Nie udało się zapisać risk_controls.yaml", exc_info=True)

    def _default_risk_controls(self) -> dict[str, object]:
        return {
            "takeProfitPct": 1.5,
            "stopLossPct": 2.0,
            "maxOpenPositions": 5,
            "maxPositionUsd": 5000.0,
            "maxSlippagePct": 0.6,
            "killSwitch": False,
        }

    def _sanitize_risk_controls(self, payload: Mapping[str, object]) -> dict[str, object]:
        base = self._default_risk_controls()
        for key in ("takeProfitPct", "stopLossPct", "maxSlippagePct"):
            value = self._safe_float(payload.get(key))
            if value is not None:
                base[key] = max(0.0, float(value))
        max_open_positions = payload.get("maxOpenPositions")
        try:
            base["maxOpenPositions"] = max(0, int(max_open_positions))
        except (TypeError, ValueError):
            pass
        max_position_usd = self._safe_float(payload.get("maxPositionUsd"))
        if max_position_usd is not None:
            base["maxPositionUsd"] = max(0.0, float(max_position_usd))
        kill_switch_value = _normalize_bool(
            payload.get("killSwitch", base.get("killSwitch", False))
        )
        base["killSwitch"] = (
            kill_switch_value
            if isinstance(kill_switch_value, bool)
            else bool(base.get("killSwitch", False))
        )
        return base

    def _guardrail_block_reason(self) -> str | None:
        sla_state = str(self._feed_sla_report.get("sla_state", "")).strip().lower()
        if self._guardrails.get("blockOnSlaAlerts") and sla_state in {"warning", "critical"}:
            return f"blokada SLA ({sla_state})"

        exposure = self._safe_float(
            self._risk_metrics.get("exposure")
            or self._risk_metrics.get("gross_exposure")
            or self._risk_metrics.get("max_exposure")
        )
        max_exposure = self._safe_float(self._guardrails.get("maxExposure"))
        if exposure is not None and max_exposure is not None and exposure > max_exposure:
            return f"ekspozycja {exposure:.3f} przekracza limit {max_exposure:.3f}"

        daily_loss = self._safe_float(
            self._risk_metrics.get("daily_loss_pct") or self._risk_metrics.get("dailyLossPct")
        )
        loss_limit = self._safe_float(self._guardrails.get("dailyLossLimitPct"))
        if daily_loss is not None and loss_limit is not None and daily_loss > loss_limit:
            return f"dzienne straty {daily_loss:.3f} przekraczają limit {loss_limit:.3f}"
        return None

    @Slot(result=bool)
    def runManualCycle(self) -> bool:
        runner = self._ensure_ai_runner()
        if runner is None:
            return False
        block_reason = self._guardrail_block_reason()
        if block_reason:
            self._record_operator_action("guardrail_block", {"reason": block_reason})
            return False
        try:
            runner.run_cycle()
            self._update_ai_snapshot_from_runner(runner)
            self._record_operator_action("manual_cycle", {"mode": "manual"})
            return True
        except Exception:
            _LOGGER.exception("runManualCycle failed")
            return False

    @Slot(result=bool)
    def startAutoMode(self) -> bool:
        runner = self._ensure_ai_runner()
        if runner is None:
            return False
        block_reason = self._guardrail_block_reason()
        if block_reason:
            self._record_operator_action("guardrail_block", {"reason": block_reason})
            return False
        try:
            runner.run_until(limit=max(1, self._ai_history_limit))
            self._execution_mode = "auto"
            self.executionModeChanged.emit()
            self._update_ai_snapshot_from_runner(runner)
            self._record_operator_action("auto_mode_started", {"mode": "auto"})
            return True
        except Exception:
            _LOGGER.exception("startAutoMode failed")
            return False

    @Slot(result=bool)
    def stopAutoMode(self) -> bool:
        if self._execution_mode != "auto":
            return True
        self._execution_mode = "manual"
        self.executionModeChanged.emit()
        self._record_operator_action("auto_mode_stopped", {"mode": "manual"})
        return True

    @Slot(str, result=bool)
    def setExecutionMode(self, mode: str) -> bool:
        normalized = str(mode or "").strip().lower()
        if normalized not in {"manual", "auto"}:
            return False
        if normalized == "auto":
            return self.startAutoMode()
        return self.stopAutoMode()

    @Slot(float, result=bool)
    def setMaxExposureLimit(self, ratio: float) -> bool:
        sanitized = max(0.0, min(1.0, float(ratio)))
        if self._guardrails.get("maxExposure") == sanitized:
            return True
        self._guardrails["maxExposure"] = sanitized
        self._ai_runner = None
        self.guardrailsChanged.emit()
        return True

    @Slot(float, result=bool)
    def setDailyLossLimitPct(self, ratio: float) -> bool:
        sanitized = max(0.0, float(ratio))
        if self._guardrails.get("dailyLossLimitPct") == sanitized:
            return True
        self._guardrails["dailyLossLimitPct"] = sanitized
        self._ai_runner = None
        self.guardrailsChanged.emit()
        return True

    @Slot(bool, result=bool)
    def setBlockOnSlaAlerts(self, enabled: bool) -> bool:
        if bool(self._guardrails.get("blockOnSlaAlerts")) == bool(enabled):
            return True
        self._guardrails["blockOnSlaAlerts"] = bool(enabled)
        self.guardrailsChanged.emit()
        return True

    def __del__(self) -> None:  # pragma: no cover - defensywne sprzątanie
        try:
            self._stop_grpc_stream()
        except Exception:
            pass

    def _record_operator_action(self, action: object, entry: object | None) -> bool:
        self._last_operator_action = self._operator_action_service.record_action(action, entry)
        self.operatorActionChanged.emit()
        return True

    def _load_core_config(self):
        if self._cached_core_config is not None:
            return self._cached_core_config
        config_path = self._resolve_core_config_path()
        if config_path is None:
            raise FileNotFoundError(
                "Nie znaleziono ścieżki do core.yaml – ustaw zmienną BOT_CORE_UI_CORE_CONFIG_PATH"
            )
        self._cached_core_config = load_core_config(config_path)
        return self._cached_core_config

    def _resolve_core_config_path(self) -> Path | None:
        if self._core_config_path is not None:
            return self._core_config_path

        candidates = (
            os.environ.get("BOT_CORE_UI_CORE_CONFIG_PATH"),
            os.environ.get("BOT_CORE_CORE_CONFIG"),
            os.environ.get("BOT_CORE_CONFIG"),
            os.environ.get("DUDZIAN_CORE_CONFIG"),
        )
        for candidate in candidates:
            if candidate:
                path = Path(candidate).expanduser()
                self._core_config_path = path
                return path

        default = Path("config/core.yaml")
        if default.exists():
            self._core_config_path = default
            return default
        return None

    # ------------------------------------------------------------------ runtime metadata helpers --
    def _update_runtime_metadata(self, *, invalidate_cache: bool) -> None:
        if invalidate_cache:
            self._runtime_config_cache = None
        next_run = self._compute_next_retrain()
        if next_run != self._retrain_next_run:
            self._retrain_next_run = next_run
            self.retrainNextRunChanged.emit()
        summary, breakdown = self._build_adaptive_snapshot()
        if summary != self._adaptive_summary:
            self._adaptive_summary = summary
            self.adaptiveStrategySummaryChanged.emit()
        if breakdown != self._ai_regime_breakdown:
            self._ai_regime_breakdown = breakdown
            self.aiRegimeBreakdownChanged.emit()

    def _load_runtime_config(self) -> "RuntimeAppConfig":
        if load_runtime_app_config is None:
            raise RuntimeError("Ładowanie konfiguracji runtime nie jest dostępne w tej dystrybucji")
        if self._runtime_config_cache is not None:
            return self._runtime_config_cache
        config_path = self._resolve_runtime_config_path()
        if config_path is None:
            raise FileNotFoundError(
                "Nie znaleziono pliku runtime.yaml – ustaw zmienną BOT_CORE_UI_RUNTIME_CONFIG_PATH"
            )
        self._runtime_config_cache = load_runtime_app_config(config_path)
        return self._runtime_config_cache

    def _resolve_runtime_config_path(self) -> Path | None:
        if self._runtime_config_path is not None:
            return self._runtime_config_path

        candidates = (
            os.environ.get("BOT_CORE_UI_RUNTIME_CONFIG_PATH"),
            os.environ.get("BOT_CORE_RUNTIME_CONFIG_PATH"),
            os.environ.get("BOT_CORE_RUNTIME_CONFIG"),
            os.environ.get("DUDZIAN_RUNTIME_CONFIG"),
        )
        for candidate in candidates:
            if candidate:
                path = Path(candidate).expanduser()
                self._runtime_config_path = path
                return path

        default = Path("config/runtime.yaml")
        if default.exists():
            self._runtime_config_path = default
            return default
        return None

    def _load_registry_path_from_yaml(self) -> Path | None:
        _require_yaml()
        config_path = self._resolve_runtime_config_path()
        if config_path is None or not config_path.exists():
            return None
        try:
            with config_path.open("r", encoding="utf-8") as handle:
                data = yaml.safe_load(handle) or {}
        except Exception:  # pragma: no cover - diagnostyka środowiska
            _LOGGER.debug(
                "Nie udało się sparsować runtime.yaml podczas wyszukiwania model_registry_path",
                exc_info=True,
            )
            return None
        ai_block = data.get("ai") if isinstance(data, Mapping) else None
        if not isinstance(ai_block, Mapping):
            return None
        registry_value = ai_block.get("model_registry_path")
        if not registry_value:
            return None
        return Path(str(registry_value)).expanduser()

    def _resolve_model_registry_path(self) -> Path | None:
        try:
            runtime_config = self._load_runtime_config()
        except Exception:
            return self._load_registry_path_from_yaml()
        registry_path = getattr(runtime_config.ai, "model_registry_path", None)
        if not registry_path:
            return self._load_registry_path_from_yaml()
        return Path(registry_path).expanduser()

    def _compute_next_retrain(self) -> str:
        fallback_schedule = os.environ.get("BOT_CORE_UI_RETRAIN_FALLBACK_CRON", "0 3 * * *")
        try:
            runtime_config = self._load_runtime_config()
        except FileNotFoundError:
            return fallback_schedule
        except Exception:  # pragma: no cover - diagnostyka środowiska
            _LOGGER.debug("Nie udało się wczytać konfiguracji runtime", exc_info=True)
            return fallback_schedule

        schedule: str | None = None
        retrain_cfg = getattr(runtime_config.ai, "retrain", None)
        if retrain_cfg and getattr(retrain_cfg, "enabled", False):
            schedule = getattr(retrain_cfg, "schedule", None) or getattr(
                runtime_config.ai, "retrain_schedule", None
            )
        else:
            schedule = getattr(runtime_config.ai, "retrain_schedule", None)
        if not schedule:
            schedule = fallback_schedule
        if CronSchedule is None:
            return schedule or fallback_schedule
        try:
            cron = CronSchedule(schedule)
            next_run = cron.next_after(datetime.now(timezone.utc))
        except Exception:  # pragma: no cover - niepoprawna składnia lub błąd obliczeń
            _LOGGER.debug("Nie udało się obliczyć najbliższego retrainingu", exc_info=True)
            return schedule or fallback_schedule
        return next_run.astimezone().isoformat(timespec="minutes")

    def _build_adaptive_snapshot(self) -> tuple[str, list[dict[str, object]]]:
        if FilesystemModelRepository is None:
            return "", []
        registry_path = self._resolve_model_registry_path()
        if registry_path is None:
            return "", []
        try:
            repository = FilesystemModelRepository(registry_path)  # type: ignore[abstract]
        except Exception:  # pragma: no cover - repozytorium może być nieosiągalne
            _LOGGER.debug("Nie udało się zainicjalizować ModelRepository", exc_info=True)
            return "", []
        try:
            artifact = repository.load("adaptive_strategy_policy.json")
        except FileNotFoundError:
            return "", []
        except Exception:  # pragma: no cover - uszkodzony plik lub brak dostępu
            _LOGGER.debug("Nie udało się wczytać stanu adaptive learnera", exc_info=True)
            return "", []

        state = getattr(artifact, "model_state", None)
        policies = state.get("policies") if isinstance(state, Mapping) else None
        if not isinstance(policies, Mapping) or not policies:
            return "", []

        fragments: list[str] = []
        breakdown: list[dict[str, object]] = []
        for regime_key, payload in policies.items():
            if not isinstance(payload, Mapping):
                continue
            strategies = payload.get("strategies")
            if not isinstance(strategies, Iterable):
                continue
            best_name: str | None = None
            best_score: float | None = None
            best_plays = 0
            normalized_strategies: list[dict[str, object]] = []
            for entry in strategies:
                if not isinstance(entry, Mapping):
                    continue
                name = str(entry.get("name") or "").strip()
                if not name:
                    continue
                plays = int(entry.get("plays", 0) or 0)
                total_reward = float(entry.get("total_reward", 0.0) or 0.0)
                last_reward = float(entry.get("last_reward", 0.0) or 0.0)
                mean_reward = total_reward / plays if plays > 0 else last_reward
                normalized_strategies.append(
                    {
                        "name": name,
                        "plays": plays,
                        "meanReward": mean_reward,
                        "totalReward": total_reward,
                    }
                )
                if best_score is None or mean_reward > best_score:
                    best_name = name
                    best_score = mean_reward
                    best_plays = plays
            if best_name is None or best_score is None:
                continue
            regime_label = str(regime_key).replace("_", " ")
            fragments.append(f"{regime_label}: {best_name} (μ={best_score:.2f}, n={best_plays})")
            breakdown.append(
                {
                    "regime": regime_label,
                    "bestStrategy": best_name,
                    "meanReward": best_score,
                    "plays": best_plays,
                    "strategies": normalized_strategies,
                }
            )

        if not fragments:
            return "", []

        updated_at = ""
        try:
            updated_at = str(artifact.metadata.get("updated_at", "")).strip()
        except Exception:
            updated_at = ""
        summary = "; ".join(fragments)
        if updated_at:
            summary = f"{summary} — aktualizacja {updated_at}"
        return summary, breakdown


__all__ = ["RuntimeService"]

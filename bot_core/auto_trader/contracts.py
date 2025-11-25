from __future__ import annotations

"""Stabilne kontrakty typów dla pojedynczego cyklu decyzyjnego."""

from datetime import datetime, timezone
from typing import Any, Mapping, TypedDict, cast

from bot_core.auto_trader.risk_bridge import RiskDecision


class DecisionCycleLatency(TypedDict, total=False):
    """Telemetry snapshot describing decision loop latency in milliseconds."""

    lastMs: float
    p50Ms: float
    p95Ms: float
    sampleCount: float


class DecisionCycleModeTransition(TypedDict, total=False):
    """Public representation of mode transition telemetry."""

    mode: str | None
    timestamp: str | None
    regime: str | None
    risk: float


class DecisionGuardrailEvent(TypedDict, total=False):
    """Recent guardrail activation with relative age in seconds."""

    name: str
    ageSeconds: float


class DecisionCycleGuardrails(TypedDict, total=False):
    """Guardrail state exposed to UI/CLI clients."""

    active: bool
    killSwitch: bool
    recent: list[DecisionGuardrailEvent]
    reasons: list[str]


class DecisionCycleTelemetry(TypedDict, total=False):
    """Structured telemetry attached to a decision cycle report."""

    cycleLatency: DecisionCycleLatency
    modeTransitions: list[DecisionCycleModeTransition]
    guardrails: DecisionCycleGuardrails


class DecisionJournalEntry(TypedDict, total=False):
    """Normalized entry emitted by the decision journal."""

    event: str
    timestamp: str
    environment: str
    portfolio: str
    risk_profile: str
    strategy: str
    symbol: str
    status: str
    side: str
    payload: Mapping[str, object]
    metadata: Mapping[str, object]


def normalize_decision_journal_entry(
    entry: Mapping[str, Any]
) -> DecisionJournalEntry:
    """Convert raw journal payloads into the public contract."""

    payload: dict[str, object] = {}
    for key in (
        "event",
        "environment",
        "portfolio",
        "risk_profile",
        "strategy",
        "symbol",
        "status",
        "side",
    ):
        value = entry.get(key)
        if value is not None:
            payload[key] = str(value)
    timestamp = entry.get("timestamp")
    if isinstance(timestamp, datetime):
        payload["timestamp"] = timestamp.astimezone(timezone.utc).isoformat()
    elif timestamp is not None:
        payload["timestamp"] = str(timestamp)
    if isinstance(entry.get("payload"), Mapping):
        payload["payload"] = dict(entry["payload"])
    if isinstance(entry.get("metadata"), Mapping):
        payload["metadata"] = dict(entry["metadata"])
    for key, value in entry.items():
        if key in payload or key in {"timestamp", "payload", "metadata"}:
            continue
        if value is None:
            continue
        payload[str(key)] = value
    return cast(DecisionJournalEntry, payload)


DecisionCycleMetadata = Mapping[str, str]
DecisionCycleMetrics = Mapping[str, float]
DecisionCycleDecision = RiskDecision | None

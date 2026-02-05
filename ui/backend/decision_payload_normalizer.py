"""Pure helpers for normalizing runtime decision records for UI payloads."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, MutableMapping


DecisionRecord = Mapping[str, object]


@dataclass(slots=True)
class RuntimeDecisionEntry:
    """Intermediate structure used during record normalization."""

    event: str
    timestamp: str
    environment: str
    portfolio: str
    risk_profile: str
    schedule: str | None
    strategy: str | None
    symbol: str | None
    side: str | None
    status: str | None
    quantity: str | None
    price: str | None
    market_regime: Mapping[str, object]
    decision: Mapping[str, object]
    ai: Mapping[str, object]
    extras: Mapping[str, object]

    def to_payload(self) -> dict[str, object]:
        return {
            "event": self.event,
            "timestamp": self.timestamp,
            "environment": self.environment,
            "portfolio": self.portfolio,
            "riskProfile": self.risk_profile,
            "schedule": self.schedule,
            "strategy": self.strategy,
            "symbol": self.symbol,
            "side": self.side,
            "status": self.status,
            "quantity": self.quantity,
            "price": self.price,
            "marketRegime": dict(self.market_regime),
            "decision": dict(self.decision),
            "ai": dict(self.ai),
            "metadata": dict(self.extras),
        }


def _normalize_bool(value: object) -> object:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "yes", "1"}:
            return True
        if lowered in {"false", "no", "0"}:
            return False
    return value


def _coerce_float(value: object) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalize_number(value: object) -> object:
    numeric = _coerce_float(value)
    if numeric is None:
        return value
    if numeric.is_integer():
        return int(numeric)
    return numeric


def _camelize(prefix: str, key: str) -> str:
    suffix = key[len(prefix) :].lstrip("_")
    if not suffix:
        return ""
    parts = [part for part in suffix.split("_") if part]
    if not parts:
        return ""
    head, *tail = parts
    return head + "".join(segment.capitalize() for segment in tail)


_BASE_FIELD_MAP: Mapping[str, str] = {
    "event": "event",
    "timestamp": "timestamp",
    "environment": "environment",
    "portfolio": "portfolio",
    "risk_profile": "risk_profile",
    "schedule": "schedule",
    "strategy": "strategy",
    "symbol": "symbol",
    "side": "side",
    "status": "status",
    "quantity": "quantity",
    "price": "price",
}


def parse_runtime_decision_entry(record: DecisionRecord) -> RuntimeDecisionEntry:
    """Normalize raw decision journal record to UI-oriented payload structure."""

    base: MutableMapping[str, object | None] = {key: None for key in _BASE_FIELD_MAP.values()}
    decision_payload: MutableMapping[str, object] = {}
    ai_payload: MutableMapping[str, object] = {}
    regime_payload: MutableMapping[str, object] = {}
    extras: MutableMapping[str, object] = {}

    confidence = record.get("confidence")
    latency = record.get("latency_ms")
    if confidence is not None:
        decision_payload["confidence"] = _normalize_number(confidence)
    if latency is not None:
        decision_payload["latencyMs"] = _normalize_number(latency)

    for key, value in record.items():
        if key in {"confidence", "latency_ms"}:
            continue
        mapped = _BASE_FIELD_MAP.get(key)
        if mapped is not None:
            base[mapped] = value
            continue
        if key.startswith("decision_"):
            normalized = _camelize("decision_", key)
            if not normalized:
                continue
            payload_value: object = value
            if normalized == "shouldTrade":
                payload_value = _normalize_bool(value)
            elif normalized in {"confidence", "latencyMs"}:
                payload_value = _normalize_number(value)
            decision_payload[normalized] = payload_value
            continue
        if key.startswith("ai_"):
            normalized = _camelize("ai_", key)
            if not normalized:
                continue
            ai_payload[normalized] = value
            continue
        if key == "market_regime":
            regime_payload["regime"] = value
            continue
        if key.startswith("market_regime"):
            normalized = _camelize("market_regime", key)
            if not normalized:
                continue
            if normalized in {"confidence", "riskScore"}:
                regime_payload[normalized] = _normalize_number(value)
            else:
                regime_payload[normalized] = value
            continue
        if key == "risk_profile":
            continue
        extras[key] = value

    return RuntimeDecisionEntry(
        event=str(base["event"] or ""),
        timestamp=str(base["timestamp"] or ""),
        environment=str(base["environment"] or ""),
        portfolio=str(base["portfolio"] or ""),
        risk_profile=str(base["risk_profile"] or ""),
        schedule=base.get("schedule"),
        strategy=base.get("strategy"),
        symbol=base.get("symbol"),
        side=base.get("side"),
        status=base.get("status"),
        quantity=base.get("quantity"),
        price=base.get("price"),
        market_regime=regime_payload,
        decision=decision_payload,
        ai=ai_payload,
        extras=extras,
    )

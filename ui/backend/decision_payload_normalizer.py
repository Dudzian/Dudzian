"""Pure helpers for normalizing runtime decision records for UI payloads."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, MutableMapping


DecisionRecord = Mapping[str, object]
DEFAULT_DECISION_PAYLOAD_SCHEMA_VERSION = "1"
SCHEMA_VERSION_KEY_PRIORITY: tuple[str, ...] = ("schema_version", "schemaVersion")
SCHEMA_VERSION_KEYS: frozenset[str] = frozenset(SCHEMA_VERSION_KEY_PRIORITY)


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
    signals: tuple[str, ...]
    ai: Mapping[str, object]
    extras: Mapping[str, object]
    schema_version: str

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
            "signals": list(self.signals),
            "ai": dict(self.ai),
            "metadata": dict(self.extras),
            "schema_version": self.schema_version,
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


def _normalize_signal_labels(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        parts = [segment.strip() for segment in value.replace(";", ",").split(",")]
        return tuple(segment for segment in parts if segment)
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes, Mapping)):
        labels: list[str] = []
        for item in value:
            if item is None:
                continue
            text = str(item).strip()
            if text:
                labels.append(text)
        return tuple(labels)
    return ()


def _merge_decision_payload(
    destination: MutableMapping[str, object], source: Mapping[str, object]
) -> None:
    for key, value in source.items():
        normalized_key = str(key)
        if normalized_key == "should_trade":
            normalized_key = "shouldTrade"
        payload_value: object = value
        if normalized_key == "shouldTrade":
            payload_value = _normalize_bool(value)
        elif normalized_key in {"confidence", "latencyMs", "latency_ms"}:
            payload_value = _normalize_number(value)
            if normalized_key == "latency_ms":
                normalized_key = "latencyMs"
        destination.setdefault(normalized_key, payload_value)


def _insert_decision_field(
    destination: MutableMapping[str, object],
    *,
    key: str,
    value: object,
    overwrite: bool,
) -> None:
    normalized_key = str(key)
    if normalized_key == "should_trade":
        normalized_key = "shouldTrade"
    if normalized_key == "latency_ms":
        normalized_key = "latencyMs"

    payload_value = value
    if normalized_key == "shouldTrade":
        payload_value = _normalize_bool(value)
    elif normalized_key in {"confidence", "latencyMs"}:
        payload_value = _normalize_number(value)

    if overwrite:
        destination[normalized_key] = payload_value
    else:
        destination.setdefault(normalized_key, payload_value)


def parse_runtime_decision_payload(record: DecisionRecord) -> dict[str, object]:
    """Normalize only nested decision payload semantics from a raw decision record."""

    decision_payload: dict[str, object] = {}

    confidence = record.get("confidence")
    latency = record.get("latency_ms")
    if confidence is not None:
        _insert_decision_field(
            decision_payload,
            key="confidence",
            value=confidence,
            overwrite=False,
        )
    if latency is not None:
        _insert_decision_field(
            decision_payload,
            key="latency_ms",
            value=latency,
            overwrite=False,
        )

    for key, value in record.items():
        if key in {"confidence", "latency_ms"} | SCHEMA_VERSION_KEYS:
            continue
        if key in {"decision", "Decision"} and isinstance(value, Mapping):
            _merge_decision_payload(decision_payload, value)
            continue
        if key.startswith("decision_"):
            normalized = _camelize("decision_", key)
            if not normalized:
                continue
            _insert_decision_field(
                decision_payload,
                key=normalized,
                value=value,
                overwrite=True,
            )
            continue

    return decision_payload


def _interpret_schema_version(record: DecisionRecord) -> str:
    raw: object | None = None
    for key in SCHEMA_VERSION_KEY_PRIORITY:
        candidate = record.get(key)
        if candidate is not None:
            raw = candidate
            break
    if raw is None:
        return DEFAULT_DECISION_PAYLOAD_SCHEMA_VERSION
    if isinstance(raw, str):
        normalized = raw.strip()
        if normalized:
            return normalized
        return DEFAULT_DECISION_PAYLOAD_SCHEMA_VERSION
    return str(raw)


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
    decision_payload: MutableMapping[str, object] = parse_runtime_decision_payload(record)
    signals_payload: tuple[str, ...] = ()
    ai_payload: MutableMapping[str, object] = {}
    regime_payload: MutableMapping[str, object] = {}
    metadata_payload: MutableMapping[str, object] = {}
    extras: MutableMapping[str, object] = {}

    for key, value in record.items():
        if key in {"confidence", "latency_ms"} | SCHEMA_VERSION_KEYS:
            continue
        if key in {"decision", "Decision"}:
            if isinstance(value, Mapping):
                continue
        if key.startswith("decision_"):
            continue
        if key == "signals":
            signals_payload = _normalize_signal_labels(value)
            continue
        if key == "metadata" and isinstance(value, Mapping):
            metadata_payload.update(
                {
                    str(meta_key): meta_value
                    for meta_key, meta_value in value.items()
                    if str(meta_key) not in SCHEMA_VERSION_KEYS
                }
            )
            continue
        mapped = _BASE_FIELD_MAP.get(key)
        if mapped is not None:
            base[mapped] = value
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

    for meta_key, meta_value in metadata_payload.items():
        extras.setdefault(meta_key, meta_value)

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
        signals=signals_payload,
        ai=ai_payload,
        extras=extras,
        schema_version=_interpret_schema_version(record),
    )

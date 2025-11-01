"""Serwis runtime dostarczający dane dziennika decyzji do QML."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Mapping, MutableMapping

from PySide6.QtCore import QObject, Property, Signal, Slot

from bot_core.runtime.journal import TradingDecisionJournal

DecisionRecord = Mapping[str, str]
DecisionLoader = Callable[[int], Iterable[DecisionRecord]]


@dataclass(slots=True)
class _RuntimeDecisionEntry:
    """Struktura pośrednia używana w konwersji rekordów."""

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


def _default_loader(limit: int) -> Iterable[DecisionRecord]:
    raise RuntimeError("Brak skonfigurowanego źródła TradingDecisionJournal")


def _load_from_journal(journal: TradingDecisionJournal, limit: int) -> Iterable[DecisionRecord]:
    exported = list(journal.export())
    if limit > 0:
        exported = exported[-limit:]
    return reversed(exported)


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


def _parse_entry(record: DecisionRecord) -> _RuntimeDecisionEntry:
    base: MutableMapping[str, str | None] = {key: None for key in _BASE_FIELD_MAP.values()}
    decision_payload: MutableMapping[str, object] = {}
    ai_payload: MutableMapping[str, object] = {}
    regime_payload: MutableMapping[str, object] = {}
    extras: MutableMapping[str, object] = {}

    confidence = record.get("confidence")
    latency = record.get("latency_ms")
    if confidence is not None:
        decision_payload["confidence"] = confidence
    if latency is not None:
        decision_payload["latencyMs"] = latency

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
            decision_payload[normalized] = payload_value
            continue
        if key.startswith("ai_"):
            normalized = _camelize("ai_", key)
            if not normalized:
                continue
            ai_payload[normalized] = value
            continue
        if key.startswith("market_regime"):
            normalized = _camelize("market_regime", key)
            if not normalized:
                continue
            regime_payload[normalized] = value
            continue
        if key == "risk_profile":
            # już zmapowane do base
            continue
        extras[key] = value

    event = str(base["event"] or "")
    timestamp = str(base["timestamp"] or "")
    environment = str(base["environment"] or "")
    portfolio = str(base["portfolio"] or "")
    risk_profile = str(base["risk_profile"] or "")

    schedule = base.get("schedule")
    strategy = base.get("strategy")
    symbol = base.get("symbol")
    side = base.get("side")
    status = base.get("status")
    quantity = base.get("quantity")
    price = base.get("price")

    return _RuntimeDecisionEntry(
        event=event,
        timestamp=timestamp,
        environment=environment,
        portfolio=portfolio,
        risk_profile=risk_profile,
        schedule=schedule,
        strategy=strategy,
        symbol=symbol,
        side=side,
        status=status,
        quantity=quantity,
        price=price,
        market_regime=regime_payload,
        decision=decision_payload,
        ai=ai_payload,
        extras=extras,
    )


class RuntimeService(QObject):
    """Zapewnia QML dostęp do najnowszych decyzji autotradera."""

    decisionsChanged = Signal()
    errorMessageChanged = Signal()

    def __init__(
        self,
        *,
        journal: TradingDecisionJournal | None = None,
        decision_loader: DecisionLoader | None = None,
        parent: QObject | None = None,
        default_limit: int = 20,
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

    # ------------------------------------------------------------------
    @Property("QVariantList", notify=decisionsChanged)
    def decisions(self) -> list[dict[str, object]]:  # type: ignore[override]
        return list(self._decisions)

    @Property(str, notify=errorMessageChanged)
    def errorMessage(self) -> str:  # type: ignore[override]
        return self._error_message

    # ------------------------------------------------------------------
    @Slot(int, result="QVariantList")
    def loadRecentDecisions(self, limit: int = 0) -> list[dict[str, object]]:  # type: ignore[override]
        """Pobiera najnowsze decyzje z dziennika."""

        size = int(limit)
        if size <= 0:
            size = self._default_limit
        try:
            raw_entries = list(self._loader(size))
        except Exception as exc:  # pragma: no cover - diagnostyka
            self._error_message = str(exc)
            self.errorMessageChanged.emit()
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
        return list(self._decisions)


__all__ = ["RuntimeService"]

"""Analiza dziennika decyzji tradingowych.

Moduł udostępnia lekkie statystyki oparte na ostatnich wpisach dziennika,
które pozwalają dynamicznie stroić parametry strategii w AutoTraderze.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable, Mapping, Sequence

from bot_core.runtime.journal import TradingDecisionJournal


_PNL_KEYS = (
    "pnl",
    "pnl_usd",
    "pnl_usdt",
    "trade_pnl",
    "realized_pnl",
    "realizedPnl",
    "equity_change",
    "unrealized_pnl",
)

_APPROVAL_KEYS = (
    "approved",
    "decision_approved",
    "normalized_approval",
    "accepted",
)


def _to_float(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        if value != value:  # NaN
            return None
        return float(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        if text.endswith("%"):
            text = text[:-1]
        try:
            return float(text)
        except ValueError:
            return None
    return None


def _to_bool(value: object) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"true", "1", "yes", "y", "approved", "trade", "filled", "executed"}:
            return True
        if text in {"false", "0", "no", "n", "rejected", "denied", "blocked", "hold", "cancelled"}:
            return False
    return None


def _parse_timestamp(value: object) -> datetime | None:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)
    if isinstance(value, (int, float)):
        try:
            return datetime.fromtimestamp(float(value), timezone.utc)
        except (OverflowError, OSError, ValueError):
            return None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        try:
            parsed = datetime.fromisoformat(text)
        except ValueError:
            return None
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    return None


@dataclass(frozen=True)
class JournalAnalytics:
    """Zestaw podstawowych metryk wyliczonych z dziennika decyzji."""

    trade_count: int
    wins: int
    losses: int
    rolling_pnl: float
    average_pnl: float
    cumulative_pnl: float
    max_drawdown: float
    max_drawdown_pct: float
    signal_accuracy: float
    approvals_total: int
    approvals_success: int
    window: int
    last_timestamp: datetime | None

    @property
    def win_rate(self) -> float:
        return self.wins / self.trade_count if self.trade_count else 0.0

    @property
    def loss_rate(self) -> float:
        return self.losses / self.trade_count if self.trade_count else 0.0

    def to_mapping(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "trade_count": self.trade_count,
            "wins": self.wins,
            "losses": self.losses,
            "rolling_pnl": round(self.rolling_pnl, 8),
            "average_pnl": round(self.average_pnl, 8),
            "cumulative_pnl": round(self.cumulative_pnl, 8),
            "max_drawdown": round(self.max_drawdown, 8),
            "max_drawdown_pct": round(self.max_drawdown_pct, 6),
            "signal_accuracy": round(self.signal_accuracy, 6),
            "approvals_total": self.approvals_total,
            "approvals_success": self.approvals_success,
            "win_rate": round(self.win_rate, 6),
            "loss_rate": round(self.loss_rate, 6),
            "window": self.window,
        }
        if self.last_timestamp is not None:
            payload["last_timestamp"] = self.last_timestamp.isoformat()
        return payload


def analyse_decision_journal(
    journal_or_records: TradingDecisionJournal | Iterable[Mapping[str, object]],
    *,
    window: int = 120,
    now: datetime | None = None,
) -> JournalAnalytics:
    """Wylicza statystyki z ostatnich wpisów dziennika decyzji."""

    if window <= 0:
        raise ValueError("window musi być dodatnie")

    if hasattr(journal_or_records, "export"):
        records = list(journal_or_records.export())  # type: ignore[assignment]
    else:
        records = list(journal_or_records)

    ordered: list[tuple[int, Mapping[str, object]]] = [
        (idx, entry)
        for idx, entry in enumerate(records)
        if isinstance(entry, Mapping)
    ]
    if len(ordered) > window:
        ordered = ordered[-window:]

    pnls: list[float] = []
    approvals_total = 0
    approvals_success = 0
    wins = 0
    losses = 0
    timestamps: list[datetime] = []

    for _, record in ordered:
        timestamp = _parse_timestamp(record.get("timestamp"))
        if timestamp is not None:
            timestamps.append(timestamp)

        pnl_value = None
        for key in _PNL_KEYS:
            pnl_value = _to_float(record.get(key))
            if pnl_value is not None:
                break
        if pnl_value is not None:
            pnls.append(pnl_value)
            if pnl_value > 0:
                wins += 1
            elif pnl_value < 0:
                losses += 1

        event_name = str(record.get("event") or "")
        if event_name.startswith("decision") or event_name.startswith("order"):
            approval: bool | None = None
            for key in _APPROVAL_KEYS:
                approval = _to_bool(record.get(key))
                if approval is not None:
                    break
            if approval is None:
                status = record.get("status")
                approval = _to_bool(status)
            if approval is not None:
                approvals_total += 1
                if approval:
                    approvals_success += 1

    trade_count = len(pnls)
    rolling_pnl = float(sum(pnls)) if pnls else 0.0
    average_pnl = rolling_pnl / trade_count if trade_count else 0.0

    cumulative = 0.0
    peak = 0.0
    max_drawdown = 0.0
    for value in pnls:
        cumulative += value
        peak = max(peak, cumulative)
        drawdown = peak - cumulative
        if drawdown > max_drawdown:
            max_drawdown = drawdown

    baseline = peak if peak > 0 else max(abs(cumulative), 1.0)
    max_drawdown_pct = max_drawdown / baseline if baseline else 0.0

    accuracy = (
        approvals_success / approvals_total
        if approvals_total
        else 0.0
    )

    last_timestamp = max(timestamps) if timestamps else None
    if last_timestamp is None and now is not None:
        last_timestamp = now if now.tzinfo else now.replace(tzinfo=timezone.utc)

    return JournalAnalytics(
        trade_count=trade_count,
        wins=wins,
        losses=losses,
        rolling_pnl=rolling_pnl,
        average_pnl=average_pnl,
        cumulative_pnl=cumulative,
        max_drawdown=max_drawdown,
        max_drawdown_pct=max_drawdown_pct,
        signal_accuracy=accuracy,
        approvals_total=approvals_total,
        approvals_success=approvals_success,
        window=window,
        last_timestamp=last_timestamp,
    )


__all__ = ["JournalAnalytics", "analyse_decision_journal"]


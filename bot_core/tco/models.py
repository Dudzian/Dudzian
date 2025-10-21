"""Modele i struktury danych wykorzystywane przy analizie kosztów TCO."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal, getcontext
from typing import Any, Mapping, MutableMapping

from bot_core.tco.utils import quantize_decimal

getcontext().prec = 28


def _to_decimal(value: Any, *, field_name: str) -> Decimal:
    if isinstance(value, Decimal):
        return value
    if isinstance(value, (int, float)):
        return Decimal(str(value))
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return Decimal("0")
        return Decimal(text)
    raise TypeError(f"Pole '{field_name}' musi być liczbą, otrzymano: {type(value)!r}")


def _parse_timestamp(raw: Any) -> datetime:
    if isinstance(raw, datetime):
        dt = raw
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    if isinstance(raw, (int, float)):
        return datetime.fromtimestamp(float(raw), tz=timezone.utc)
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            raise ValueError("Wartość timestamp nie może być pusta")
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        dt = datetime.fromisoformat(text)
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    raise TypeError(f"Nieobsługiwany format timestampu: {type(raw)!r}")


@dataclass(slots=True)
class TradeCostEvent:
    """Pojedyncza transakcja wykorzystywana w analizie kosztów."""

    timestamp: datetime
    strategy: str
    risk_profile: str
    instrument: str
    exchange: str
    side: str
    quantity: Decimal
    price: Decimal
    commission: Decimal = Decimal("0")
    slippage: Decimal = Decimal("0")
    funding: Decimal = Decimal("0")
    other: Decimal = Decimal("0")
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "TradeCostEvent":
        data: MutableMapping[str, Any] = dict(payload)
        try:
            timestamp = _parse_timestamp(data.pop("timestamp"))
        except KeyError as exc:  # pragma: no cover - defensywne
            raise KeyError("Brak wymaganej kolumny 'timestamp'") from exc
        strategy = str(data.pop("strategy"))
        risk_profile = str(data.pop("risk_profile"))
        instrument = str(data.pop("instrument"))
        exchange = str(data.pop("exchange"))
        side = str(data.pop("side")).lower()
        if side not in {"buy", "sell"}:
            raise ValueError("Pole 'side' musi mieć wartość 'buy' lub 'sell'")
        quantity = _to_decimal(data.pop("quantity"), field_name="quantity")
        price = _to_decimal(data.pop("price"), field_name="price")
        commission = _to_decimal(data.pop("commission", 0), field_name="commission")
        slippage = _to_decimal(data.pop("slippage", 0), field_name="slippage")
        funding = _to_decimal(data.pop("funding", 0), field_name="funding")
        other = _to_decimal(data.pop("other", data.pop("other_costs", 0)), field_name="other")
        metadata = data.pop("metadata", {})
        return cls(
            timestamp=timestamp,
            strategy=strategy,
            risk_profile=risk_profile,
            instrument=instrument,
            exchange=exchange,
            side=side,
            quantity=quantity,
            price=price,
            commission=commission,
            slippage=slippage,
            funding=funding,
            other=other,
            metadata=dict(metadata),
        )

    @property
    def notional(self) -> Decimal:
        return abs(self.price * self.quantity)

    @property
    def total_cost(self) -> Decimal:
        return self.commission + self.slippage + self.funding + self.other

    def to_dict(self) -> dict[str, Any]:
        quantized = {
            "quantity": quantize_decimal(self.quantity),
            "price": quantize_decimal(self.price),
            "commission": quantize_decimal(self.commission),
            "slippage": quantize_decimal(self.slippage),
            "funding": quantize_decimal(self.funding),
            "other": quantize_decimal(self.other),
            "total_cost": quantize_decimal(self.total_cost),
            "notional": quantize_decimal(self.notional),
        }
        return {
            "timestamp": self.timestamp.isoformat(),
            "strategy": self.strategy,
            "risk_profile": self.risk_profile,
            "instrument": self.instrument,
            "exchange": self.exchange,
            "side": self.side,
            "quantity": float(quantized["quantity"]),
            "price": float(quantized["price"]),
            "commission": float(quantized["commission"]),
            "slippage": float(quantized["slippage"]),
            "funding": float(quantized["funding"]),
            "other": float(quantized["other"]),
            "total_cost": float(quantized["total_cost"]),
            "notional": float(quantized["notional"]),
            "metadata": dict(self.metadata),
        }


@dataclass(slots=True)
class CostBreakdown:
    commission: Decimal
    slippage: Decimal
    funding: Decimal
    other: Decimal

    @classmethod
    def zero(cls) -> "CostBreakdown":
        zero = Decimal("0")
        return cls(zero, zero, zero, zero)

    def add_event(self, event: TradeCostEvent) -> None:
        self.commission += event.commission
        self.slippage += event.slippage
        self.funding += event.funding
        self.other += event.other

    @property
    def total(self) -> Decimal:
        return self.commission + self.slippage + self.funding + self.other

    def to_dict(self) -> dict[str, float]:
        return {
            "commission": float(quantize_decimal(self.commission)),
            "slippage": float(quantize_decimal(self.slippage)),
            "funding": float(quantize_decimal(self.funding)),
            "other": float(quantize_decimal(self.other)),
            "total": float(quantize_decimal(self.total)),
        }


@dataclass(slots=True)
class ProfileCostSummary:
    profile: str
    trade_count: int
    notional: Decimal
    breakdown: CostBreakdown

    @property
    def total_cost(self) -> Decimal:
        return self.breakdown.total

    @property
    def cost_per_trade(self) -> Decimal:
        if self.trade_count == 0:
            return Decimal("0")
        return self.total_cost / Decimal(self.trade_count)

    @property
    def cost_bps(self) -> Decimal:
        if self.notional == 0:
            return Decimal("0")
        return (self.total_cost / self.notional) * Decimal("10000")

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile": self.profile,
            "trade_count": self.trade_count,
            "notional": float(quantize_decimal(self.notional)),
            "breakdown": self.breakdown.to_dict(),
            "cost_per_trade": float(quantize_decimal(self.cost_per_trade)),
            "cost_bps": float(quantize_decimal(self.cost_bps)),
        }


@dataclass(slots=True)
class StrategyCostSummary:
    strategy: str
    profiles: dict[str, ProfileCostSummary]
    total: ProfileCostSummary

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "profiles": {name: summary.to_dict() for name, summary in self.profiles.items()},
            "total": self.total.to_dict(),
        }


@dataclass(slots=True)
class SchedulerCostSummary:
    """Zestawienie kosztów przypisanych do scheduler-a."""

    scheduler: str
    strategies: dict[str, ProfileCostSummary]
    total: ProfileCostSummary

    def to_dict(self) -> dict[str, Any]:
        return {
            "scheduler": self.scheduler,
            "strategies": {name: summary.to_dict() for name, summary in self.strategies.items()},
            "total": self.total.to_dict(),
        }


@dataclass(slots=True)
class TCOReport:
    generated_at: datetime
    metadata: Mapping[str, Any]
    strategies: dict[str, StrategyCostSummary]
    total: ProfileCostSummary
    alerts: list[str] = field(default_factory=list)
    schedulers: dict[str, SchedulerCostSummary] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "generated_at": self.generated_at.isoformat(),
            "metadata": dict(self.metadata),
            "strategies": {name: summary.to_dict() for name, summary in self.strategies.items()},
            "total": self.total.to_dict(),
            "alerts": list(self.alerts),
            "schedulers": {name: summary.to_dict() for name, summary in self.schedulers.items()},
        }

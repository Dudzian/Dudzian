"""Modele danych wykorzystywane do raportowania podatkowego."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional


@dataclass(slots=True)
class MatchedLot:
    """Część partii wykorzystana przy rozliczaniu zbycia."""

    lot_id: str
    acquisition_time: datetime
    quantity: float
    cost_basis: float
    fee: float = 0.0
    venue: Optional[str] = None
    source: Optional[str] = None
    holding_period_days: float = 0.0


@dataclass(slots=True)
class TaxLot:
    """Partia nabycia aktywa."""

    lot_id: str
    asset: str
    acquisition_time: datetime
    quantity: float
    cost_basis: float
    fee: float = 0.0
    venue: Optional[str] = None
    source: Optional[str] = None
    holding_period_days: float = 0.0
    _remaining_quantity: float = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.quantity <= 0:
            raise ValueError("Quantity for tax lot must be positive")
        self._remaining_quantity = float(self.quantity)

    @property
    def remaining_quantity(self) -> float:
        return max(0.0, self._remaining_quantity)

    def consume(self, quantity: float, *, disposal_time: datetime | None = None) -> MatchedLot:
        if quantity <= 0:
            raise ValueError("Cannot consume non-positive quantity")
        if quantity > self._remaining_quantity + 1e-12:
            raise ValueError("Requested quantity exceeds available lot size")
        ratio = quantity / self.quantity
        cost = self.cost_basis * ratio
        fee = self.fee * ratio
        self._remaining_quantity -= quantity
        holding_days = 0.0
        if disposal_time is not None:
            delta = disposal_time - self.acquisition_time
            holding_days = max(0.0, delta.total_seconds() / 86400.0)
        return MatchedLot(
            lot_id=self.lot_id,
            acquisition_time=self.acquisition_time,
            quantity=quantity,
            cost_basis=cost,
            fee=fee,
            venue=self.venue,
            source=self.source,
            holding_period_days=holding_days,
        )

    def remaining_lot(self) -> Optional["TaxLot"]:
        remaining = self.remaining_quantity
        if remaining <= 1e-12:
            return None
        ratio = remaining / self.quantity
        return TaxLot(
            lot_id=self.lot_id,
            asset=self.asset,
            acquisition_time=self.acquisition_time,
            quantity=remaining,
            cost_basis=self.cost_basis * ratio,
            fee=self.fee * ratio,
            venue=self.venue,
            source=self.source,
            holding_period_days=self.holding_period_days,
        )


@dataclass(slots=True)
class DisposalEvent:
    """Zdarzenie zbycia aktywa."""

    event_id: str
    asset: str
    disposal_time: datetime
    quantity: float
    proceeds: float
    fee: float = 0.0
    venue: Optional[str] = None
    source: Optional[str] = None
    matched_lots: List[MatchedLot] = field(default_factory=list)
    short_term_gain: float = 0.0
    long_term_gain: float = 0.0
    short_term_quantity: float = 0.0
    long_term_quantity: float = 0.0
    average_holding_period_days: float = 0.0
    short_term_tax: float = 0.0
    long_term_tax: float = 0.0

    @property
    def cost_basis(self) -> float:
        return sum(m.cost_basis + m.fee for m in self.matched_lots)

    @property
    def realized_gain(self) -> float:
        return self.proceeds - self.fee - self.cost_basis

    @property
    def total_tax_liability(self) -> float:
        return self.short_term_tax + self.long_term_tax


@dataclass(slots=True)
class TaxReportTotals:
    """Podsumowanie globalne raportu."""

    proceeds: float
    cost_basis: float
    fees: float
    realized_gain: float
    short_term_gain: float
    long_term_gain: float
    unrealized_cost_basis: float
    unrealized_quantity: float
    short_term_quantity: float
    long_term_quantity: float
    average_holding_period_days: float
    unrealized_short_term_cost_basis: float
    unrealized_long_term_cost_basis: float
    unrealized_short_term_quantity: float
    unrealized_long_term_quantity: float
    average_open_holding_period_days: float
    short_term_tax: float
    long_term_tax: float
    total_tax_liability: float

    def to_dict(self) -> dict:
        return {
            "proceeds": self.proceeds,
            "cost_basis": self.cost_basis,
            "fees": self.fees,
            "realized_gain": self.realized_gain,
            "short_term_gain": self.short_term_gain,
            "long_term_gain": self.long_term_gain,
            "unrealized_cost_basis": self.unrealized_cost_basis,
            "unrealized_quantity": self.unrealized_quantity,
            "short_term_quantity": self.short_term_quantity,
            "long_term_quantity": self.long_term_quantity,
            "average_holding_period_days": self.average_holding_period_days,
            "unrealized_short_term_cost_basis": self.unrealized_short_term_cost_basis,
            "unrealized_long_term_cost_basis": self.unrealized_long_term_cost_basis,
            "unrealized_short_term_quantity": self.unrealized_short_term_quantity,
            "unrealized_long_term_quantity": self.unrealized_long_term_quantity,
            "average_open_holding_period_days": self.average_open_holding_period_days,
            "short_term_tax": self.short_term_tax,
            "long_term_tax": self.long_term_tax,
            "total_tax_liability": self.total_tax_liability,
        }


@dataclass(slots=True)
class AssetBreakdown:
    """Agregowane statystyki podatkowe dla pojedynczego aktywa."""

    asset: str
    proceeds: float
    cost_basis: float
    fees: float
    realized_gain: float
    disposed_quantity: float
    open_quantity: float
    open_cost_basis: float
    short_term_gain: float
    long_term_gain: float
    short_term_quantity: float
    long_term_quantity: float
    average_holding_period_days: float
    open_short_term_quantity: float
    open_long_term_quantity: float
    open_short_term_cost_basis: float
    open_long_term_cost_basis: float
    open_average_holding_period_days: float
    short_term_tax: float
    long_term_tax: float
    total_tax_liability: float

    def to_dict(self) -> dict:
        return {
            "asset": self.asset,
            "proceeds": self.proceeds,
            "cost_basis": self.cost_basis,
            "fees": self.fees,
            "realized_gain": self.realized_gain,
            "disposed_quantity": self.disposed_quantity,
            "open_quantity": self.open_quantity,
            "open_cost_basis": self.open_cost_basis,
            "short_term_gain": self.short_term_gain,
            "long_term_gain": self.long_term_gain,
            "short_term_quantity": self.short_term_quantity,
            "long_term_quantity": self.long_term_quantity,
            "average_holding_period_days": self.average_holding_period_days,
            "open_short_term_quantity": self.open_short_term_quantity,
            "open_long_term_quantity": self.open_long_term_quantity,
            "open_short_term_cost_basis": self.open_short_term_cost_basis,
            "open_long_term_cost_basis": self.open_long_term_cost_basis,
            "open_average_holding_period_days": self.open_average_holding_period_days,
            "short_term_tax": self.short_term_tax,
            "long_term_tax": self.long_term_tax,
            "total_tax_liability": self.total_tax_liability,
        }


@dataclass(slots=True)
class VenueBreakdown:
    """Agregowane statystyki podatkowe dla konkretnej giełdy/venue."""

    venue: Optional[str]
    proceeds: float
    cost_basis: float
    fees: float
    realized_gain: float
    disposed_quantity: float
    open_quantity: float
    open_cost_basis: float
    short_term_gain: float
    long_term_gain: float
    short_term_quantity: float
    long_term_quantity: float
    average_holding_period_days: float
    open_short_term_quantity: float
    open_long_term_quantity: float
    open_short_term_cost_basis: float
    open_long_term_cost_basis: float
    open_average_holding_period_days: float
    short_term_tax: float
    long_term_tax: float
    total_tax_liability: float

    def to_dict(self) -> dict:
        return {
            "venue": self.venue,
            "proceeds": self.proceeds,
            "cost_basis": self.cost_basis,
            "fees": self.fees,
            "realized_gain": self.realized_gain,
            "disposed_quantity": self.disposed_quantity,
            "open_quantity": self.open_quantity,
            "open_cost_basis": self.open_cost_basis,
            "short_term_gain": self.short_term_gain,
            "long_term_gain": self.long_term_gain,
            "short_term_quantity": self.short_term_quantity,
            "long_term_quantity": self.long_term_quantity,
            "average_holding_period_days": self.average_holding_period_days,
            "open_short_term_quantity": self.open_short_term_quantity,
            "open_long_term_quantity": self.open_long_term_quantity,
            "open_short_term_cost_basis": self.open_short_term_cost_basis,
            "open_long_term_cost_basis": self.open_long_term_cost_basis,
            "open_average_holding_period_days": self.open_average_holding_period_days,
            "short_term_tax": self.short_term_tax,
            "long_term_tax": self.long_term_tax,
            "total_tax_liability": self.total_tax_liability,
        }


@dataclass(slots=True)
class TaxReport:
    """Raport podatkowy dla wskazanej jurysdykcji."""

    jurisdiction: str
    method: str
    generated_at: datetime
    events: List[DisposalEvent]
    open_lots: List[TaxLot]
    totals: TaxReportTotals
    asset_breakdown: List[AssetBreakdown]
    venue_breakdown: List[VenueBreakdown]
    period_breakdown: List["PeriodBreakdown"]
    base_currency: str | None = None

    def to_dict(self) -> dict:
        return {
            "jurisdiction": self.jurisdiction,
            "method": self.method,
            "generated_at": self.generated_at.isoformat(),
            "base_currency": self.base_currency,
            "events": [
                {
                    "event_id": event.event_id,
                    "asset": event.asset,
                    "disposal_time": event.disposal_time.isoformat(),
                    "quantity": event.quantity,
                    "proceeds": event.proceeds,
                    "fee": event.fee,
                    "venue": event.venue,
                    "source": event.source,
                    "matched_lots": [
                        {
                            "lot_id": matched.lot_id,
                            "acquisition_time": matched.acquisition_time.isoformat(),
                            "quantity": matched.quantity,
                            "cost_basis": matched.cost_basis,
                            "fee": matched.fee,
                            "venue": matched.venue,
                            "source": matched.source,
                            "holding_period_days": matched.holding_period_days,
                        }
                        for matched in event.matched_lots
                    ],
                    "cost_basis": event.cost_basis,
                    "realized_gain": event.realized_gain,
                    "short_term_gain": event.short_term_gain,
                    "long_term_gain": event.long_term_gain,
                    "short_term_quantity": event.short_term_quantity,
                    "long_term_quantity": event.long_term_quantity,
                    "average_holding_period_days": event.average_holding_period_days,
                    "short_term_tax": event.short_term_tax,
                    "long_term_tax": event.long_term_tax,
                    "total_tax_liability": event.total_tax_liability,
                }
                for event in self.events
            ],
            "open_lots": [
                {
                    "lot_id": lot.lot_id,
                    "asset": lot.asset,
                    "acquisition_time": lot.acquisition_time.isoformat(),
                    "quantity": lot.quantity,
                    "cost_basis": lot.cost_basis,
                    "fee": lot.fee,
                    "venue": lot.venue,
                    "source": lot.source,
                    "holding_period_days": lot.holding_period_days,
                }
                for lot in self.open_lots
            ],
            "totals": self.totals.to_dict(),
            "asset_breakdown": [item.to_dict() for item in self.asset_breakdown],
            "venue_breakdown": [item.to_dict() for item in self.venue_breakdown],
            "period_breakdown": [item.to_dict() for item in self.period_breakdown],
        }


@dataclass(slots=True)
class PeriodBreakdown:
    """Agregowane dane dla okresu raportowego (np. miesiąca)."""

    period: str
    period_start: datetime
    period_end: datetime
    proceeds: float
    cost_basis: float
    fees: float
    realized_gain: float
    disposed_quantity: float
    short_term_gain: float
    long_term_gain: float
    short_term_quantity: float
    long_term_quantity: float
    average_holding_period_days: float
    short_term_tax: float
    long_term_tax: float
    total_tax_liability: float

    def to_dict(self) -> dict:
        return {
            "period": self.period,
            "period_start": self.period_start.isoformat(),
            "period_end": self.period_end.isoformat(),
            "proceeds": self.proceeds,
            "cost_basis": self.cost_basis,
            "fees": self.fees,
            "realized_gain": self.realized_gain,
            "disposed_quantity": self.disposed_quantity,
            "short_term_gain": self.short_term_gain,
            "long_term_gain": self.long_term_gain,
            "short_term_quantity": self.short_term_quantity,
            "long_term_quantity": self.long_term_quantity,
            "average_holding_period_days": self.average_holding_period_days,
            "short_term_tax": self.short_term_tax,
            "long_term_tax": self.long_term_tax,
            "total_tax_liability": self.total_tax_liability,
        }

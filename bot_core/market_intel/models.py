"""Modele danych współdzielone przez moduł Market Intelligence."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Mapping, MutableMapping, Sequence


@dataclass(slots=True)
class MarketIntelQuery:
    """Zapytanie o znormalizowane metryki rynkowe (tryb 'cache')."""

    symbol: str
    interval: str
    lookback_bars: int


@dataclass(slots=True)
class MarketIntelSnapshot:
    """Zestaw metryk rynkowych (tryb 'cache')."""

    symbol: str
    interval: str
    start: datetime | None
    end: datetime | None
    bar_count: int
    price_change_pct: float | None
    volatility_pct: float | None
    max_drawdown_pct: float | None
    average_volume: float | None
    liquidity_usd: float | None
    momentum_score: float | None
    metadata: Mapping[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "symbol": self.symbol,
            "interval": self.interval,
            "bar_count": self.bar_count,
            "start": self.start.isoformat() if self.start else None,
            "end": self.end.isoformat() if self.end else None,
            "price_change_pct": self.price_change_pct,
            "volatility_pct": self.volatility_pct,
            "max_drawdown_pct": self.max_drawdown_pct,
            "average_volume": self.average_volume,
            "liquidity_usd": self.liquidity_usd,
            "momentum_score": self.momentum_score,
        }
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload


@dataclass(slots=True, frozen=True)
class MarketIntelSourceInfo:
    """Opis źródła danych wykorzystanego do wygenerowania metryk (tryb 'sqlite')."""

    type: str
    path: str
    table: str

    def to_mapping(self) -> Mapping[str, str]:
        return {"type": self.type, "path": self.path, "table": self.table}


@dataclass(slots=True, frozen=True)
class MarketIntelBaseline:
    """Bazowe metryki rynku (tryb 'sqlite')."""

    symbol: str
    mid_price: float
    avg_depth_usd: float
    avg_spread_bps: float
    funding_rate_bps: float
    sentiment_score: float
    realized_volatility: float
    weight: float

    def to_mapping(self) -> Mapping[str, float | str]:
        return {
            "symbol": self.symbol,
            "mid_price": self.mid_price,
            "avg_depth_usd": self.avg_depth_usd,
            "avg_spread_bps": self.avg_spread_bps,
            "funding_rate_bps": self.funding_rate_bps,
            "sentiment_score": self.sentiment_score,
            "realized_volatility": self.realized_volatility,
            "weight": self.weight,
        }


__all__ = [
    "MarketIntelQuery",
    "MarketIntelSnapshot",
    "MarketIntelSourceInfo",
    "MarketIntelBaseline",
]

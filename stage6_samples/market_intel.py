"""Statyczny provider danych Market Intel wykorzystywany w pipeline'ach Stage6.

Moduł udostępnia deterministyczne dane depth-of-book, funding, sentiment oraz
trajektorie cenowe potrzebne do zasiania bazy `market_metrics.sqlite` podczas
lokalnych oraz CI/CD uruchomień stress-labu Stage6.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from bot_core.market_intel.sqlite_builder import (
    MarketIntelDataProvider,
    OHLCVBar,
    OrderBookLevel,
    OrderBookSnapshot,
    FundingSnapshot,
    SentimentSnapshot,
)


@dataclass(frozen=True, slots=True)
class _SymbolFixture:
    mid_price: float
    order_book_spread_bps: float
    depth_levels: Sequence[tuple[float, float]]
    funding_bps: float
    sentiment: float
    weight: float
    ohlcv_start: float
    ohlcv_step: float


_FIXTURES: Mapping[str, _SymbolFixture] = {
    "BTCUSDT": _SymbolFixture(
        mid_price=47_500.0,
        order_book_spread_bps=6.0,
        depth_levels=(
            (47_495.0, 1.6),
            (47_492.5, 1.2),
            (47_488.0, 1.1),
            (47_484.5, 0.9),
            (47_480.0, 0.8),
        ),
        funding_bps=8.5,
        sentiment=0.42,
        weight=1.0,
        ohlcv_start=46_900.0,
        ohlcv_step=0.0007,
    ),
    "ETHUSDT": _SymbolFixture(
        mid_price=3_420.0,
        order_book_spread_bps=8.5,
        depth_levels=(
            (3_418.5, 2.4),
            (3_417.0, 2.1),
            (3_415.5, 1.8),
            (3_414.0, 1.5),
            (3_412.5, 1.3),
        ),
        funding_bps=6.2,
        sentiment=0.28,
        weight=0.8,
        ohlcv_start=3_360.0,
        ohlcv_step=0.0011,
    ),
    "SOLUSDT": _SymbolFixture(
        mid_price=162.0,
        order_book_spread_bps=12.0,
        depth_levels=(
            (161.8, 4.0),
            (161.5, 3.7),
            (161.2, 3.4),
            (160.9, 3.1),
            (160.6, 2.9),
        ),
        funding_bps=11.4,
        sentiment=0.18,
        weight=0.6,
        ohlcv_start=155.0,
        ohlcv_step=0.0016,
    ),
}


class _StaticMarketIntelProvider(MarketIntelDataProvider):
    """Implementacja providera wykorzystująca statyczne fixture'y."""

    def __init__(self, fixtures: Mapping[str, _SymbolFixture] | None = None) -> None:
        self._fixtures = dict(fixtures or _FIXTURES)

    # ------------------------------------------------------------------ helpers --
    def _fixture(self, symbol: str) -> _SymbolFixture:
        try:
            return self._fixtures[symbol.upper()]
        except KeyError as exc:  # pragma: no cover - diagnostyka środowisk
            raise KeyError(f"Brak danych Stage6 Market Intel dla symbolu {symbol!r}") from exc

    def _build_order_book(self, symbol: str, *, depth: int) -> OrderBookSnapshot:
        fixture = self._fixture(symbol)
        depth = max(1, int(depth))
        bids: list[OrderBookLevel] = []
        asks: list[OrderBookLevel] = []
        half_spread = fixture.mid_price * fixture.order_book_spread_bps / 20_000.0
        for idx, (price, quantity) in enumerate(fixture.depth_levels[:depth]):
            price = float(price)
            quantity = float(quantity)
            bids.append(OrderBookLevel(price=price, quantity=quantity))
            asks.append(
                OrderBookLevel(
                    price=fixture.mid_price + half_spread + idx * 0.5,
                    quantity=quantity * 0.95,
                )
            )
        if not bids or not asks:
            raise ValueError(f"Brak poziomów orderbooka dla {symbol}")
        return OrderBookSnapshot(bids=tuple(bids), asks=tuple(asks))

    # ------------------------------------------------------------------ API --
    def fetch_order_book(self, symbol: str, *, depth: int) -> OrderBookSnapshot:
        return self._build_order_book(symbol, depth=depth)

    def fetch_funding(self, symbol: str) -> FundingSnapshot:
        fixture = self._fixture(symbol)
        return FundingSnapshot(rate_bps=float(fixture.funding_bps))

    def fetch_sentiment(self, symbol: str) -> SentimentSnapshot:
        fixture = self._fixture(symbol)
        return SentimentSnapshot(score=float(fixture.sentiment))

    def fetch_ohlcv(self, symbol: str, *, bars: int) -> Sequence[OHLCVBar]:
        fixture = self._fixture(symbol)
        bars = max(2, int(bars))
        closes: list[OHLCVBar] = []
        for index in range(bars):
            close = fixture.ohlcv_start * (1.0 + fixture.ohlcv_step) ** index
            closes.append(OHLCVBar(close=float(close)))
        return tuple(closes)

    def resolve_weight(self, symbol: str) -> float:
        fixture = self._fixture(symbol)
        return float(fixture.weight)


def build_provider() -> MarketIntelDataProvider:
    """Zwraca providera wykorzystywanego w pipeline'ach CI Stage6."""

    return _StaticMarketIntelProvider()


__all__ = ["build_provider"]

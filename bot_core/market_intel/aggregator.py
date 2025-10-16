"""Agregator metryk inteligencji rynkowej dla Stage6."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from math import sqrt
from statistics import mean, pstdev
from typing import Iterable, Mapping, MutableMapping, Sequence

from bot_core.data.base import CacheStorage


@dataclass(slots=True)
class MarketIntelQuery:
    """Zapytanie o znormalizowane metryki rynkowe."""

    symbol: str
    interval: str
    lookback_bars: int


@dataclass(slots=True)
class MarketIntelSnapshot:
    """Zestaw zagregowanych metryk rynkowych."""

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
        """Serializuje snapshot do struktury JSON-owalnej."""

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


class MarketIntelAggregator:
    """Buduje metryki rynkowe w oparciu o lokalny cache OHLCV."""

    def __init__(
        self,
        storage: CacheStorage,
        *,
        price_column: str = "close",
        volume_column: str = "volume",
        time_column: str = "open_time",
    ) -> None:
        self._storage = storage
        self._price_column = price_column
        self._volume_column = volume_column
        self._time_column = time_column

    def build_many(self, queries: Iterable[MarketIntelQuery]) -> dict[str, MarketIntelSnapshot]:
        """Zwraca snapshoty dla wielu zapytań."""

        results: dict[str, MarketIntelSnapshot] = {}
        for query in queries:
            snapshot = self.build_snapshot(query)
            results[query.symbol] = snapshot
        return results

    def build_snapshot(self, query: MarketIntelQuery) -> MarketIntelSnapshot:
        """Buduje snapshot inteligencji rynkowej dla pojedynczego symbolu."""

        key = f"{query.symbol}::{query.interval}"
        payload = self._storage.read(key)
        columns = tuple(payload.get("columns", ()))
        rows: Sequence[Sequence[float]] = payload.get("rows", ())
        if not rows:
            raise ValueError(f"Brak danych OHLCV dla {key}")

        price_index = self._column_index(columns, self._price_column)
        volume_index = self._column_index(columns, self._volume_column)
        time_index = self._column_index(columns, self._time_column)

        selected_rows = self._select_tail(rows, query.lookback_bars)
        closes = [float(row[price_index]) for row in selected_rows]
        volumes = [float(row[volume_index]) for row in selected_rows]
        timestamps = [float(row[time_index]) for row in selected_rows]

        bar_count = len(selected_rows)
        start_dt = self._timestamp_to_dt(timestamps[0]) if timestamps else None
        end_dt = self._timestamp_to_dt(timestamps[-1]) if timestamps else None

        price_change_pct = self._price_change(closes)
        volatility_pct = self._volatility(closes)
        max_drawdown_pct = self._max_drawdown(closes)
        average_volume = mean(volumes) if volumes else None
        liquidity_usd = self._liquidity_score(closes, volumes)
        momentum_score = self._momentum(closes)

        metadata: MutableMapping[str, float] = {}
        if bar_count:
            metadata["bars_used"] = float(bar_count)
        if price_change_pct is not None:
            metadata["price_change_abs"] = price_change_pct / 100.0
        if volatility_pct is not None:
            metadata["volatility_abs"] = volatility_pct / 100.0

        return MarketIntelSnapshot(
            symbol=query.symbol,
            interval=query.interval,
            start=start_dt,
            end=end_dt,
            bar_count=bar_count,
            price_change_pct=price_change_pct,
            volatility_pct=volatility_pct,
            max_drawdown_pct=max_drawdown_pct,
            average_volume=average_volume,
            liquidity_usd=liquidity_usd,
            momentum_score=momentum_score,
            metadata=metadata,
        )

    def _select_tail(
        self, rows: Sequence[Sequence[float]], lookback_bars: int
    ) -> Sequence[Sequence[float]]:
        if lookback_bars <= 0 or lookback_bars >= len(rows):
            return rows
        return rows[-lookback_bars:]

    def _column_index(self, columns: Sequence[str], name: str) -> int:
        try:
            return columns.index(name)
        except ValueError as exc:  # pragma: no cover - walidacja defensywna
            raise ValueError(f"Brak kolumny '{name}' w danych OHLCV") from exc

    def _timestamp_to_dt(self, timestamp_ms: float) -> datetime:
        return datetime.fromtimestamp(float(timestamp_ms) / 1000.0, tz=timezone.utc)

    def _price_change(self, closes: Sequence[float]) -> float | None:
        if len(closes) < 2:
            return None
        start_price = closes[0]
        end_price = closes[-1]
        if start_price == 0:
            return None
        return (end_price / start_price - 1.0) * 100.0

    def _volatility(self, closes: Sequence[float]) -> float | None:
        if len(closes) < 3:
            return None
        returns = []
        for idx in range(1, len(closes)):
            previous = closes[idx - 1]
            current = closes[idx]
            if previous <= 0:
                continue
            returns.append(current / previous - 1.0)
        if len(returns) < 2:
            return None
        volatility = pstdev(returns)
        scaled = volatility * sqrt(len(returns))
        return scaled * 100.0

    def _max_drawdown(self, closes: Sequence[float]) -> float | None:
        if len(closes) < 2:
            return None
        peak = closes[0]
        max_drawdown = 0.0
        for price in closes:
            if price > peak:
                peak = price
                continue
            if peak <= 0:
                continue
            drawdown = (price - peak) / peak
            if drawdown < max_drawdown:
                max_drawdown = drawdown
        return abs(max_drawdown) * 100.0 if max_drawdown < 0 else 0.0

    def _liquidity_score(
        self, closes: Sequence[float], volumes: Sequence[float]
    ) -> float | None:
        if not closes or not volumes or len(closes) != len(volumes):
            return None
        notional = [price * volume for price, volume in zip(closes, volumes)]
        return mean(notional) if notional else None

    def _momentum(self, closes: Sequence[float]) -> float | None:
        if len(closes) < 2:
            return None
        mid = len(closes) // 2
        first_half = closes[:mid]
        second_half = closes[mid:]
        if not first_half or not second_half:
            return None
        first_avg = mean(first_half)
        second_avg = mean(second_half)
        if first_avg == 0:
            return None
        return (second_avg / first_avg - 1.0) * 100.0


__all__ = [
    "MarketIntelAggregator",
    "MarketIntelQuery",
    "MarketIntelSnapshot",
]

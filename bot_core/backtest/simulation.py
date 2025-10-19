"""Lightweight matching engine used by paper trading adapters."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List, Mapping


@dataclass(slots=True)
class BacktestFill:
    """Result of executing an order inside the matching engine."""

    order_id: int
    side: str
    size: float
    price: float
    fee: float
    slippage: float
    timestamp: datetime
    partial: bool


@dataclass(slots=True)
class MatchingConfig:
    """Configuration parameters for the matching engine."""

    latency_bars: int = 1
    slippage_bps: float = 5.0
    fee_bps: float = 10.0
    liquidity_share: float = 0.5


@dataclass(slots=True)
class _PendingOrder:
    order_id: int
    side: str
    remaining: float
    bar_index: int
    submitted_at: datetime
    stop_loss: float | None
    take_profit: float | None


class MatchingEngine:
    """A tiny matching engine that gradually fills market orders."""

    _EPSILON = 1e-12

    def __init__(self, config: MatchingConfig | None = None) -> None:
        self._config = config or MatchingConfig()
        self._orders: List[_PendingOrder] = []
        self._next_order_id = 1
        self._latency = max(0, int(self._config.latency_bars))
        share = float(self._config.liquidity_share)
        if share <= 0.0:
            share = 1.0
        self._fill_fraction = min(1.0, share)
        self._slippage_rate = abs(float(self._config.slippage_bps)) / 10_000.0
        self._fee_rate = abs(float(self._config.fee_bps)) / 10_000.0

    def submit_market_order(
        self,
        *,
        side: str,
        size: float,
        index: int,
        timestamp: datetime,
        stop_loss: float | None = None,
        take_profit: float | None = None,
    ) -> int:
        direction = (side or "").lower()
        if direction not in {"buy", "sell"}:
            raise ValueError(f"Unsupported side: {side!r}")
        remaining = float(size)
        if remaining <= 0.0:
            raise ValueError("Order size must be positive")
        order_id = self._next_order_id
        self._next_order_id += 1
        ts = self._ensure_timestamp(timestamp)
        order = _PendingOrder(
            order_id=order_id,
            side=direction,
            remaining=remaining,
            bar_index=int(index),
            submitted_at=ts,
            stop_loss=stop_loss,
            take_profit=take_profit,
        )
        self._orders.append(order)
        return order_id

    def process_bar(
        self,
        *,
        index: int,
        timestamp: datetime,
        bar: Mapping[str, object],
    ) -> List[BacktestFill]:
        if not self._orders:
            return []
        price = self._extract_price(bar)
        if price is None:
            return []
        ts = self._ensure_timestamp(timestamp)
        fills: List[BacktestFill] = []
        for order in list(self._orders):
            if index - order.bar_index < self._latency:
                continue
            fills.extend(self._fill_order(order, price, ts, index))
        self._orders = [order for order in self._orders if order.remaining > self._EPSILON]
        return fills

    def _fill_order(
        self,
        order: _PendingOrder,
        price: float,
        timestamp: datetime,
        index: int,
    ) -> List[BacktestFill]:
        remaining = order.remaining
        if remaining <= self._EPSILON:
            return []
        fraction = self._fill_fraction
        fill_size = remaining * fraction
        if fill_size <= self._EPSILON:
            fill_size = remaining
        slippage_value = price * self._slippage_rate
        if order.side == "buy":
            execution_price = price + slippage_value
        else:
            execution_price = max(0.0, price - slippage_value)
        fee = abs(execution_price * fill_size) * self._fee_rate
        is_partial = fill_size < remaining - self._EPSILON
        fill = BacktestFill(
            order_id=order.order_id,
            side=order.side,
            size=fill_size,
            price=execution_price,
            fee=fee,
            slippage=slippage_value,
            timestamp=timestamp,
            partial=is_partial,
        )
        if is_partial:
            order.remaining = max(0.0, remaining - fill_size)
            order.bar_index = index
        else:
            order.remaining = 0.0
        return [fill]

    @staticmethod
    def _ensure_timestamp(value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value

    @staticmethod
    def _extract_price(bar: Mapping[str, object]) -> float | None:
        candidate = bar.get("close") if isinstance(bar, Mapping) else None
        if candidate is None:
            candidate = bar.get("price") if isinstance(bar, Mapping) else None
        try:
            price = float(candidate)
        except (TypeError, ValueError):
            return None
        if price <= 0.0:
            return None
        return price


__all__ = ["BacktestFill", "MatchingConfig", "MatchingEngine"]

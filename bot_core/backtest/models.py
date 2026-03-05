"""Czyste modele danych używane w module backtestu."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List

__all__ = [
    "BacktestFill",
    "BacktestTrade",
    "PerformanceMetrics",
    "BacktestReport",
]


@dataclass(slots=True)
class BacktestFill:
    """Wynik egzekucji zlecenia w silniku matchingowym."""

    order_id: int
    side: str
    size: float
    price: float
    fee: float
    slippage: float
    timestamp: datetime
    partial: bool


@dataclass(slots=True)
class BacktestTrade:
    direction: str
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    quantity: float
    pnl: float
    pnl_pct: float
    fees_paid: float
    slippage_cost: float


@dataclass(slots=True)
class PerformanceMetrics:
    total_return_pct: float
    cagr_pct: float
    max_drawdown_pct: float
    sharpe_ratio: float
    sortino_ratio: float
    omega_ratio: float
    hit_ratio_pct: float
    risk_of_ruin_pct: float
    max_exposure_pct: float
    fees_paid: float
    slippage_cost: float


@dataclass(slots=True)
class BacktestReport:
    trades: List[BacktestTrade] = field(default_factory=list)
    fills: List[BacktestFill] = field(default_factory=list)
    equity_curve: List[float] = field(default_factory=list)
    equity_timestamps: List[datetime] = field(default_factory=list)
    starting_balance: float = 0.0
    final_balance: float = 0.0
    metrics: PerformanceMetrics | None = None
    warnings: List[str] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)
    strategy_metadata: Dict[str, Any] = field(default_factory=dict)

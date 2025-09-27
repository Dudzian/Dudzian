# backtest/metrics.py
# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import List, Dict, Any
import math
import statistics


@dataclass
class MetricsResult:
    n_trades: int
    gross_profit: float
    gross_loss: float
    net_profit: float
    win_rate_pct: float
    avg_trade_usdt: float
    profit_factor: float
    expectancy_usdt: float
    expectancy_r: float
    max_drawdown_usdt: float
    sharpe_like: float
    r_multiple_avg: float


def _profit_factor(gross_profit: float, gross_loss: float) -> float:
    """PF = gross_profit / gross_loss; jeśli loss==0 i profit>0 → inf; jeśli brak transakcji → 0."""
    if gross_profit <= 0.0 and gross_loss <= 0.0:
        return 0.0
    if gross_loss == 0.0:
        return math.inf if gross_profit > 0.0 else 0.0
    return gross_profit / gross_loss


def compute_metrics(trades: List[Any]) -> MetricsResult:
    """
    Oczekujemy, że każdy trade ma przynajmniej:
      - t.pnl_usdt (float) – PnL transakcji w USDT
      - t.r_multiple (float) – wynik w jednostkach R (opcjonalnie)
    Pozostałe metryki liczymy łącznie dla sekwencji transakcji.
    """
    pnls: List[float] = []
    rs: List[float] = []

    for t in trades:
        p = getattr(t, "pnl_usdt", None)
        if p is None:
            continue
        pnls.append(float(p))
        r = getattr(t, "r_multiple", None)
        if r is not None:
            rs.append(float(r))

    n = len(pnls)
    if n == 0:
        return MetricsResult(
            n_trades=0,
            gross_profit=0.0,
            gross_loss=0.0,
            net_profit=0.0,
            win_rate_pct=0.0,
            avg_trade_usdt=0.0,
            profit_factor=0.0,
            expectancy_usdt=0.0,
            expectancy_r=0.0,
            max_drawdown_usdt=0.0,
            sharpe_like=0.0,
            r_multiple_avg=0.0,
        )

    gross_profit = sum(p for p in pnls if p > 0.0)
    gross_loss = -sum(p for p in pnls if p < 0.0)  # dodatnia liczba strat
    net_profit = gross_profit - gross_loss
    wins = sum(1 for p in pnls if p > 0.0)
    win_rate_pct = (wins / n) * 100.0
    avg_trade = sum(pnls) / n
    pf = _profit_factor(gross_profit, gross_loss)
    expectancy_usdt = avg_trade
    r_avg = (sum(rs) / len(rs)) if rs else 0.0

    # Max drawdown po krzywej kapitału liczonej jako skumulowane PnL
    equity = 0.0
    peak = 0.0
    max_dd = 0.0
    for p in pnls:
        equity += p
        peak = max(peak, equity)
        dd = peak - equity
        if dd > max_dd:
            max_dd = dd

    # Sharpe-like: mean / stdev * sqrt(n) (bez przeskalowania do dziennych)
    if n > 1:
        stdev = statistics.pstdev(pnls)
        sharpe_like = (avg_trade / stdev * math.sqrt(n)) if stdev > 0 else 0.0
    else:
        sharpe_like = 0.0

    return MetricsResult(
        n_trades=n,
        gross_profit=gross_profit,
        gross_loss=gross_loss,
        net_profit=net_profit,
        win_rate_pct=win_rate_pct,
        avg_trade_usdt=avg_trade,
        profit_factor=pf,
        expectancy_usdt=expectancy_usdt,
        expectancy_r=r_avg,
        max_drawdown_usdt=max_dd,
        sharpe_like=sharpe_like,
        r_multiple_avg=r_avg,
    )


def to_dict(m: MetricsResult) -> Dict[str, Any]:
    d = asdict(m)
    # ujednolicenie klucza jak w runner.py/optimize.py (win_rate_%)
    d["win_rate_%"] = d.pop("win_rate_pct")
    return d

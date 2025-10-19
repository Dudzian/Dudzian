"""Metryki backtestu niezależne od warstwy legacy."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import statistics
from typing import Any, Iterable, List

__all__ = ["MetricsResult", "compute_metrics", "to_dict"]


@dataclass(slots=True)
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
    if gross_profit <= 0.0 and gross_loss <= 0.0:
        return 0.0
    if gross_loss == 0.0:
        return math.inf if gross_profit > 0.0 else 0.0
    return gross_profit / gross_loss


def _extract_trade_pnl(trade: Any) -> float | None:
    for attr in ("pnl", "pnl_usdt"):
        value = getattr(trade, attr, None)
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                return None
    return None


def _extract_trade_r(trade: Any) -> float | None:
    value = getattr(trade, "r_multiple", None)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def compute_metrics(trades: Iterable[Any]) -> MetricsResult:
    pnls: List[float] = []
    rs: List[float] = []

    for trade in trades:
        pnl = _extract_trade_pnl(trade)
        if pnl is None:
            continue
        pnls.append(pnl)
        r_val = _extract_trade_r(trade)
        if r_val is not None:
            rs.append(r_val)

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
    gross_loss = -sum(p for p in pnls if p < 0.0)
    net_profit = gross_profit - gross_loss
    wins = sum(1 for p in pnls if p > 0.0)
    win_rate_pct = (wins / n) * 100.0
    avg_trade = sum(pnls) / n
    pf = _profit_factor(gross_profit, gross_loss)
    expectancy_usdt = avg_trade
    expectancy_r = (sum(rs) / len(rs)) if rs else 0.0

    equity = 0.0
    peak = 0.0
    max_dd = 0.0
    for pnl in pnls:
        equity += pnl
        peak = max(peak, equity)
        drawdown = peak - equity
        max_dd = max(max_dd, drawdown)

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
        expectancy_r=expectancy_r,
        max_drawdown_usdt=max_dd,
        sharpe_like=sharpe_like,
        r_multiple_avg=expectancy_r,
    )


def to_dict(metrics: MetricsResult) -> dict[str, float | int]:
    data = asdict(metrics)
    data["win_rate_%"] = data.pop("win_rate_pct")
    return data

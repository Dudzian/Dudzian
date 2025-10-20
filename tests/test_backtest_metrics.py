from __future__ import annotations
from dataclasses import dataclass

import pytest

from bot_core.backtest.metrics import MetricsResult, compute_metrics, to_dict


@dataclass
class _Trade:
    pnl: float | None = None
    pnl_usdt: float | None = None
    r_multiple: float | None = None


def test_compute_metrics_with_mixed_trades() -> None:
    trades = [
        _Trade(pnl=100.0, r_multiple=1.0),
        _Trade(pnl_usdt=-50.0, r_multiple=-0.5),
        _Trade(pnl=25.0, r_multiple=0.25),
        _Trade(),
    ]

    metrics = compute_metrics(trades)

    assert isinstance(metrics, MetricsResult)
    assert metrics.n_trades == 3
    assert metrics.gross_profit == pytest.approx(125.0)
    assert metrics.gross_loss == pytest.approx(50.0)
    assert metrics.net_profit == pytest.approx(75.0)
    assert metrics.win_rate_pct == pytest.approx((2 / 3) * 100.0)
    assert metrics.avg_trade_usdt == pytest.approx(25.0)
    assert metrics.profit_factor == pytest.approx(2.5)
    assert metrics.expectancy_usdt == pytest.approx(25.0)
    assert metrics.expectancy_r == pytest.approx(0.25)
    assert metrics.max_drawdown_usdt == pytest.approx(50.0)
    assert metrics.sharpe_like == pytest.approx(0.70710678, rel=1e-6)
    assert metrics.r_multiple_avg == pytest.approx(0.25)

    payload = to_dict(metrics)
    assert payload["n_trades"] == 3
    assert payload["win_rate_%"] == pytest.approx((2 / 3) * 100.0)
    assert "win_rate_pct" not in payload


def test_compute_metrics_handles_empty_sequence() -> None:
    metrics = compute_metrics([])
    assert metrics == MetricsResult(
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

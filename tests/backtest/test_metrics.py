import math
from datetime import datetime, timedelta, timezone

import pandas as pd

from bot_core.backtest.engine import BacktestEngine
from bot_core.backtest.simulation import MatchingConfig


class _NullStrategy:
    async def prepare(self, context, data_provider):  # pragma: no cover - not used in tests
        return None

    async def handle_market_data(self, context, market_payload):  # pragma: no cover - not used
        return context

    async def notify_fill(self, context, fill):  # pragma: no cover - not used
        return None

    async def shutdown(self):  # pragma: no cover - not used in tests
        return None


def _context_builder(payload):
    return type(
        "Context",
        (),
        {
            "symbol": payload["symbol"],
            "timeframe": payload["timeframe"],
            "portfolio_value": payload.get("portfolio_value", 0.0),
            "position": payload.get("position", 0.0),
            "timestamp": payload.get("timestamp"),
            "metadata": payload.get("metadata"),
            "extra": payload.get("extra", {}),
        },
    )()


def _build_engine():
    data = pd.DataFrame({"close": [100.0, 101.0, 99.0, 100.0], "volume": [1, 1, 1, 1]})
    return BacktestEngine(
        strategy_factory=_NullStrategy,
        context_builder=_context_builder,
        data=data,
        symbol="TEST/USDT",
        timeframe="1d",
        initial_balance=1_000.0,
        matching=MatchingConfig(),
    )


def _equity_series():
    base = datetime(2023, 1, 1, tzinfo=timezone.utc)
    equity_curve = [1_000.0, 1_010.0, 990.0, 1_005.0]
    equity_ts = [base + timedelta(days=idx) for idx in range(len(equity_curve))]
    return equity_curve, equity_ts


def test_sortino_uses_total_periods_in_denominator():
    engine = _build_engine()
    equity_curve, equity_ts = _equity_series()
    returns = [0.01, -0.02, 0.015]
    metrics = engine._compute_metrics(
        equity_curve,
        equity_ts,
        returns,
        total_fees=0.0,
        total_slippage=0.0,
        trades=[],
        max_exposure_ratio=0.5,
    )
    periods = len(returns)
    avg_return = sum(returns) / periods
    downside = [r for r in returns if r < 0]
    downside_dev = math.sqrt(sum(r**2 for r in downside) / periods)
    expected_sortino = (avg_return / downside_dev) * math.sqrt(252)
    assert math.isclose(metrics.sortino_ratio, expected_sortino, rel_tol=1e-6)


def test_sortino_infinite_when_no_downside_returns():
    engine = _build_engine()
    equity_curve, equity_ts = _equity_series()
    returns = [0.01, 0.02, 0.03]
    metrics = engine._compute_metrics(
        equity_curve,
        equity_ts,
        returns,
        total_fees=0.0,
        total_slippage=0.0,
        trades=[],
        max_exposure_ratio=0.25,
    )
    assert math.isinf(metrics.sortino_ratio)
    assert metrics.sortino_ratio > 0

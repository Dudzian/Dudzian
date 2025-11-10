"""Integration tests for the TradingStrategies shim."""

from __future__ import annotations

import logging
from typing import Mapping

import numpy as np
import pandas as pd
import pytest

from bot_core.trading.engine import (
    MultiSessionBacktestResult,
    TradingEngineFactory,
    TradingParameters,
    TradingStrategies,
)


def _make_ohlcv(seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2022-01-01", periods=180, freq="D")
    steps = rng.normal(0.0005, 0.02, size=len(dates))
    prices = 100.0 * (1 + steps).cumprod()
    noise = rng.normal(0.0, 0.5, size=len(dates))
    frame = pd.DataFrame(
        {
            "open": prices + noise,
            "high": prices + np.abs(noise) + rng.uniform(0.0, 0.8, size=len(dates)),
            "low": prices - np.abs(noise) - rng.uniform(0.0, 0.8, size=len(dates)),
            "close": prices,
            "volume": rng.integers(500, 5_000, size=len(dates)),
        },
        index=dates,
    )
    frame["high"] = frame[["open", "high", "low", "close"]].max(axis=1)
    frame["low"] = frame[["open", "high", "low", "close"]].min(axis=1)
    return frame


def _assert_common_metrics(metrics: Mapping[str, float], *, result) -> None:
    assert metrics["trades"] == result.total_trades
    assert metrics["total_trades"] == result.total_trades
    assert pytest.approx(metrics["total_return"], rel=1e-6) == result.total_return
    assert pytest.approx(metrics["sharpe_ratio"], rel=1e-6) == result.sharpe_ratio
    assert pytest.approx(metrics["volatility"], rel=1e-6) == result.volatility


def test_trading_strategies_single_session_matches_engine(
    caplog: pytest.LogCaptureFixture, capsys: pytest.CaptureFixture[str]
) -> None:
    data = _make_ohlcv(123)
    params = TradingParameters(signal_threshold=0.18, position_size=0.75)
    engine = TradingEngineFactory.create_test_engine()
    shim = TradingStrategies(engine=engine, logger=engine._logger)

    with caplog.at_level(logging.INFO, logger=engine._logger.name):
        metrics, trades, equity = shim.run_strategy(
            data,
            params,
            initial_capital=12_500.0,
            fee_bps=7.5,
            risk_profile="balanced",
            metadata={"source": "integration"},
        )

    engine_result = engine.run_strategy(
        data,
        params,
        initial_capital=12_500.0,
        fee_bps=7.5,
    )

    _assert_common_metrics(metrics, result=engine_result)
    assert metrics["risk_profile"] == "balanced"
    assert metrics["metadata"] == {"source": "integration"}
    assert metrics["session_count"] == 1
    assert metrics["initial_capital"] == pytest.approx(12_500.0, rel=1e-12)
    assert metrics["fee_bps"] == pytest.approx(7.5, rel=1e-12)

    pd.testing.assert_frame_equal(trades.reset_index(drop=True), engine_result.trades.reset_index(drop=True))
    pd.testing.assert_series_equal(equity, engine_result.equity_curve)

    captured = capsys.readouterr()
    combined_output = f"{captured.out}{captured.err}"
    assert "TradingStrategies.run_strategy invoked" in combined_output
    assert "TradingStrategies.run_strategy completed" in combined_output


def test_trading_strategies_multi_session_aggregates_results() -> None:
    engine = TradingEngineFactory.create_test_engine()
    shim = TradingStrategies(engine=engine, logger=engine._logger)

    data = {"trend": _make_ohlcv(101), "mean": _make_ohlcv(202)}
    params_map = {
        "trend": TradingParameters(signal_threshold=0.15, position_size=0.9),
        "mean": TradingParameters(signal_threshold=0.25, position_size=0.6),
    }

    metrics, trades, equity = shim.run_strategy(
        data,
        params_map,
        initial_capital=25_000.0,
        fee_bps=4.0,
        session_weights={"trend": 0.6, "mean": 0.4},
        risk_profile="core",
    )

    engine_result = engine.run_strategy(
        data,
        params_map,
        initial_capital=25_000.0,
        fee_bps=4.0,
        session_weights={"trend": 0.6, "mean": 0.4},
    )

    assert isinstance(engine_result, MultiSessionBacktestResult)
    _assert_common_metrics(metrics, result=engine_result.aggregate)
    assert metrics["session_count"] == 2
    assert metrics["risk_profile"] == "core"
    assert set(metrics["sessions"]) == {"trend", "mean"}

    for name, session_metrics in metrics["sessions"].items():
        session_result = engine_result.sessions[name]
        _assert_common_metrics(session_metrics, result=session_result)

    assert pytest.approx(metrics["weights"]["trend"], rel=1e-6) == engine_result.weights["trend"]
    assert pytest.approx(metrics["weights"]["mean"], rel=1e-6) == engine_result.weights["mean"]

    assert "session" in trades.columns
    observed_sessions = set(trades["session"].unique()) if not trades.empty else set()
    assert observed_sessions.issubset({"trend", "mean"})
    for name, session_result in engine_result.sessions.items():
        session_trades = trades[trades["session"] == name].drop(columns="session")
        session_trades = session_trades.reindex(columns=session_result.trades.columns)
        pd.testing.assert_frame_equal(
            session_trades.reset_index(drop=True),
            session_result.trades.reset_index(drop=True),
        )
    pd.testing.assert_series_equal(equity, engine_result.aggregate.equity_curve)

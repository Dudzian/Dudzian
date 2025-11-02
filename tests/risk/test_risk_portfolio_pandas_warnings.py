from __future__ import annotations

import logging
import warnings
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from bot_core.risk.portfolio import (
    CorrelationAnalyzer,
    RiskManagement,
    RiskMetrics,
    VolatilityEstimator,
    backtest_risk_strategy,
    calculate_optimal_leverage,
)


def _collect_messages(caplog: pytest.LogCaptureFixture) -> list[str]:
    return list(caplog.messages)


def test_volatility_estimator_reports_pandas_warning(caplog: pytest.LogCaptureFixture) -> None:
    estimator = VolatilityEstimator()
    returns = pd.Series([0.01, 0.02, -0.01, 0.03, -0.02, 0.01, 0.015, -0.005, 0.007, -0.011, 0.013])
    original_np_sum = np.sum

    def warn_sum(values: object, *args: object, **kwargs: object) -> float:
        warnings.warn("ewma degrade", pd.errors.PerformanceWarning)
        return float(original_np_sum(values, *args, **kwargs))

    with (
        patch("bot_core.observability.pandas_warnings.observe_pandas_warning") as observe_warning,
        patch("numpy.sum", side_effect=warn_sum),
        caplog.at_level(logging.WARNING, logger="bot_core.risk.portfolio"),
    ):
        estimator.ewma_volatility(returns)

    observe_warning.assert_called_once()
    kwargs = observe_warning.call_args.kwargs
    assert kwargs["component"] == "risk.volatility.ewma"
    assert kwargs["category"] == "PerformanceWarning"
    assert kwargs["message"] == "ewma degrade"
    assert any("risk.volatility.ewma" in message for message in _collect_messages(caplog))


def test_correlation_analyzer_logs_pandas_warning(caplog: pytest.LogCaptureFixture) -> None:
    analyzer = CorrelationAnalyzer()
    returns1 = pd.Series([0.1, 0.2, 0.3, 0.4])
    returns2 = pd.Series([0.05, 0.1, 0.15, 0.2])
    original_concat = pd.concat

    def concat_with_warning(objs: list[pd.Series], *args: object, **kwargs: object) -> pd.DataFrame:
        warnings.warn("concat degrade", pd.errors.PerformanceWarning)
        return original_concat(objs, *args, **kwargs)

    with (
        patch("bot_core.observability.pandas_warnings.observe_pandas_warning") as observe_warning,
        patch("pandas.concat", side_effect=concat_with_warning),
        caplog.at_level(logging.WARNING, logger="bot_core.risk.portfolio"),
    ):
        analyzer.calculate_dynamic_correlation(returns1, returns2, window=2)

    observe_warning.assert_called_once()
    kwargs = observe_warning.call_args.kwargs
    assert kwargs["component"] == "risk.correlation.dynamic"
    assert kwargs["category"] == "PerformanceWarning"
    assert kwargs["message"] == "concat degrade"
    assert any("risk.correlation.dynamic" in message for message in _collect_messages(caplog))


def test_backtest_strategy_captures_pandas_warning(caplog: pytest.LogCaptureFixture) -> None:
    class StubRiskManager:
        def __init__(self) -> None:
            self.max_portfolio_risk = 0.2

        def calculate_position_size(self, *args: object, **kwargs: object) -> SimpleNamespace:
            warnings.warn("backtest degrade", pd.errors.PerformanceWarning)
            return SimpleNamespace(recommended_size=0.1)

        def check_position_limits(self, *args: object, **kwargs: object) -> tuple[bool, str]:
            return True, "OK"

        def _calculate_portfolio_heat(self, *args: object, **kwargs: object) -> float:
            return 0.1

    manager = StubRiskManager()
    frame = pd.DataFrame({"close": [100, 101, 102]})

    with (
        patch("bot_core.observability.pandas_warnings.observe_pandas_warning") as observe_warning,
        caplog.at_level(logging.WARNING, logger="bot_core.risk.portfolio"),
    ):
        result = backtest_risk_strategy(
            manager,
            {"BTC": frame},
            [{"symbol": "BTC", "strength": 1.0, "confidence": 1.0}],
        )

    observe_warning.assert_called_once()
    kwargs = observe_warning.call_args.kwargs
    assert kwargs["component"] == "risk.backtest"
    assert kwargs["category"] == "PerformanceWarning"
    assert kwargs["message"] == "backtest degrade"
    assert result["total_trades"] == 1
    assert any("risk.backtest" in message for message in _collect_messages(caplog))


def test_risk_metrics_report_pandas_warning(caplog: pytest.LogCaptureFixture) -> None:
    manager = RiskManagement()
    portfolio = {"BTC": {"size": 0.5}}
    market = {
        "BTC": pd.DataFrame(
            {"close": [100 + i for i in range(40)]}, index=pd.RangeIndex(start=0, stop=40)
        )
    }

    def warn_var(self: RiskManagement, *args: object, **kwargs: object) -> float:
        warnings.warn("metrics degrade", pd.errors.PerformanceWarning)
        return 0.12

    with (
        patch("bot_core.observability.pandas_warnings.observe_pandas_warning") as observe_warning,
        patch.object(RiskManagement, "_calculate_var", side_effect=warn_var, create=True),
        patch.object(
            RiskManagement,
            "_calculate_expected_shortfall",
            return_value=0.08,
            create=True,
        ),
        patch.object(
            RiskManagement,
            "_calculate_drawdown_risk",
            return_value=0.05,
            create=True,
        ),
        patch.object(
            RiskManagement,
            "_calculate_correlation_risk",
            return_value=0.03,
            create=True,
        ),
        patch.object(
            RiskManagement,
            "_calculate_liquidity_risk",
            return_value=0.02,
            create=True,
        ),
        caplog.at_level(logging.WARNING, logger="bot_core.risk.portfolio"),
    ):
        metrics = manager.calculate_risk_metrics(portfolio, market)

    observe_warning.assert_called_once()
    kwargs = observe_warning.call_args.kwargs
    assert kwargs["component"] == "risk.metrics"
    assert kwargs["category"] == "PerformanceWarning"
    assert kwargs["message"] == "metrics degrade"
    assert isinstance(metrics, RiskMetrics)
    assert any("risk.metrics" in message for message in _collect_messages(caplog))


def test_update_portfolio_state_logs_warning(caplog: pytest.LogCaptureFixture) -> None:
    if not hasattr(RiskManagement, "update_portfolio_state"):
        pytest.skip("RiskManagement.update_portfolio_state is not implemented")

    manager = RiskManagement()
    manager.historical_returns["BTC"] = pd.Series([0.01, 0.02, -0.01])
    market_returns = {"BTC": pd.Series([0.03, -0.02, 0.01])}

    original_concat = pd.concat

    def concat_with_warning(objs: list[pd.Series], *args: object, **kwargs: object) -> pd.Series:
        warnings.warn("state degrade", pd.errors.PerformanceWarning)
        return original_concat(objs, *args, **kwargs)

    with (
        patch("bot_core.observability.pandas_warnings.observe_pandas_warning") as observe_warning,
        patch("pandas.concat", side_effect=concat_with_warning),
        caplog.at_level(logging.WARNING, logger="bot_core.risk.portfolio"),
    ):
        manager.update_portfolio_state(10_000.0, {"BTC": {"size": 1.0}}, market_returns)

    observe_warning.assert_called_once()
    kwargs = observe_warning.call_args.kwargs
    assert kwargs["component"] == "risk.portfolio_state"
    assert kwargs["category"] == "PerformanceWarning"
    assert kwargs["message"] == "state degrade"
    assert any("risk.portfolio_state" in message for message in _collect_messages(caplog))


def test_calculate_optimal_leverage_reports_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    returns = pd.Series([0.01] * 40)
    original_std = pd.Series.std

    def std_with_warning(self: pd.Series, *args: object, **kwargs: object) -> float:
        warnings.warn("leverage degrade", pd.errors.PerformanceWarning)
        return float(original_std(self, *args, **kwargs))

    with (
        patch("bot_core.observability.pandas_warnings.observe_pandas_warning") as observe_warning,
        patch.object(pd.Series, "std", std_with_warning),
        caplog.at_level(logging.WARNING, logger="bot_core.risk.portfolio"),
    ):
        leverage = calculate_optimal_leverage(returns)

    observe_warning.assert_called_once()
    kwargs = observe_warning.call_args.kwargs
    assert kwargs["component"] == "risk.leverage"
    assert kwargs["category"] == "PerformanceWarning"
    assert kwargs["message"] == "leverage degrade"
    assert leverage > 0
    assert any("risk.leverage" in message for message in _collect_messages(caplog))

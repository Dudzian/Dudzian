from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import logging
import warnings
from unittest.mock import patch

import numpy as np
import pandas as pd

from bot_core.reports.monte_carlo_report import MonteCarloReportBuilder
from bot_core.simulation.monte_carlo import (
    MonteCarloEngine,
    MonteCarloScenario,
    ModelType,
    RiskParameters,
    VolatilityConfig,
    load_price_series,
)


DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "sample_ohlcv"


@dataclass
class BuyHoldStrategy:
    name: str = "buy_hold"

    def evaluate_path(self, prices: pd.Series) -> float:
        return float(prices.iloc[-1] / prices.iloc[0] - 1.0)


def _run_engine(model: ModelType) -> MonteCarloEngine:
    prices = load_price_series(DATA_DIR / "trend.csv")
    scenario = MonteCarloScenario(model=model, volatility=VolatilityConfig())
    risk = RiskParameters(horizon_days=10, confidence_level=0.95, num_paths=64, seed=7)
    engine = MonteCarloEngine(scenario, risk)
    engine.run([BuyHoldStrategy()], prices)
    return engine


def test_gbm_simulation_generates_paths_and_metrics() -> None:
    prices = load_price_series(DATA_DIR / "trend.csv")
    scenario = MonteCarloScenario(model=ModelType.GBM, volatility=VolatilityConfig())
    risk = RiskParameters(horizon_days=10, confidence_level=0.95, num_paths=64, seed=1)
    engine = MonteCarloEngine(scenario, risk)
    result = engine.run([BuyHoldStrategy()], prices)

    assert result.price_paths.shape == (risk.num_paths, 11)
    assert result.drawdown_distribution.shape == (risk.num_paths,)
    strategy_metrics = result.strategy_results["buy_hold"].metrics
    assert set(["VaR", "CVaR", "expected_shortfall", "mean_pnl", "std_pnl", "probabilistic_drawdown"]).issubset(
        strategy_metrics
    )
    assert np.isfinite(result.drawdown_probability)
    assert strategy_metrics["probabilistic_drawdown"] == result.drawdown_probability


def test_heston_and_bootstrap_models_execute_without_errors() -> None:
    for model in (ModelType.HESTON, ModelType.BOOTSTRAP):
        engine = _run_engine(model)
        assert engine.risk_parameters.num_paths == 64


def test_report_builder_produces_summary_frame() -> None:
    prices = load_price_series(DATA_DIR / "trend.csv")
    scenario = MonteCarloScenario(model=ModelType.GBM, volatility=VolatilityConfig())
    risk = RiskParameters(horizon_days=5, confidence_level=0.9, num_paths=32, seed=3)
    engine = MonteCarloEngine(scenario, risk)
    result = engine.run([BuyHoldStrategy()], prices)
    report = MonteCarloReportBuilder(result).build()

    assert "probabilistic_drawdown" in report.metadata
    assert report.metadata["num_paths"] == risk.num_paths
    assert not report.summary.empty
    assert "VaR" in report.summary.columns

    as_dict = report.to_dict()
    assert "summary" in as_dict and "metadata" in as_dict
    assert as_dict["metadata"]["model"] == ModelType.GBM.value


def test_monte_carlo_strategy_pandas_warning_tracked(caplog) -> None:
    prices = pd.Series([100.0, 101.0, 102.0, 103.0], name="close")
    scenario = MonteCarloScenario(model=ModelType.GBM, volatility=VolatilityConfig())
    risk = RiskParameters(horizon_days=3, confidence_level=0.95, num_paths=4, seed=11)
    engine = MonteCarloEngine(scenario, risk)

    class WarningStrategy:
        name = "warning"

        def evaluate_path(self, series: pd.Series) -> float:
            warnings.warn("vectorized fallback", pd.errors.PerformanceWarning)
            return float(series.iloc[-1] / series.iloc[0] - 1.0)

    with (
        patch("bot_core.observability.pandas_warnings.observe_pandas_warning") as observe_warning,
        caplog.at_level(logging.WARNING, logger="bot_core.simulation.monte_carlo.engine"),
    ):
        engine.run([WarningStrategy()], prices)

    assert observe_warning.call_count >= 1
    kwargs = observe_warning.call_args_list[0].kwargs
    assert kwargs["component"] == "backtest.monte_carlo.strategy"
    assert kwargs["category"] == "PerformanceWarning"
    assert kwargs["message"] == "vectorized fallback"
    assert any(
        "Pandas warning captured in backtest.monte_carlo.strategy" in message
        for message in caplog.messages
    )

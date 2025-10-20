"""Testy natywnego silnika tradingowego w pakiecie bot_core."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock

import numpy as np
import pandas as pd

from bot_core.trading.engine import (
    BacktestResult,
    EngineConfig,
    RiskManagementService,
    TechnicalIndicators,
    TechnicalIndicatorsService,
    TradingEngine,
    TradingParameters,
    TradingSignalService,
)


class TestNativeTradingEngine(unittest.TestCase):
    """Minimalny zestaw testów sanity-check dla modułu bot_core.trading."""

    def setUp(self) -> None:
        self.config = EngineConfig(log_level="ERROR")
        self.engine = TradingEngine(config=self.config)
        self.params = TradingParameters()
        self.data = self._make_data()

    def _make_data(self) -> pd.DataFrame:
        dates = pd.date_range(start="2021-01-01", end="2021-12-31", freq="D")
        rng = np.random.default_rng(seed=1234)
        steps = rng.normal(0.0005, 0.02, size=len(dates))
        prices = 100.0 * (1 + steps).cumprod()
        noise = rng.normal(0, 0.3, size=len(dates))
        frame = pd.DataFrame(
            {
                "open": prices + noise,
                "high": prices + np.abs(noise) + rng.uniform(0, 1, len(dates)),
                "low": prices - np.abs(noise) - rng.uniform(0, 1, len(dates)),
                "close": prices,
                "volume": rng.integers(1_000, 5_000, len(dates)),
            },
            index=dates,
        )
        frame["high"] = frame[["open", "high", "low", "close"]].max(axis=1)
        frame["low"] = frame[["open", "high", "low", "close"]].min(axis=1)
        return frame

    def test_indicator_pipeline(self) -> None:
        validator = TechnicalIndicatorsService(MagicMock(), self.config)
        indicators = validator.calculate_indicators(self.data, self.params)
        self.assertIsInstance(indicators, TechnicalIndicators)
        self.assertEqual(len(indicators.rsi), len(self.data))

    def test_signal_and_risk_services(self) -> None:
        calculator = TechnicalIndicatorsService(MagicMock(), self.config)
        generator = TradingSignalService(MagicMock())
        risk = RiskManagementService(MagicMock())

        indicators = calculator.calculate_indicators(self.data, self.params)
        signals = generator.generate_signals(indicators, self.params)
        managed = risk.apply_risk_management(self.data, signals, indicators, self.params)

        self.assertEqual(len(signals), len(self.data))
        self.assertEqual(len(managed), len(self.data))
        self.assertTrue(set(managed.unique()).issubset({-1, 0, 1}))

    def test_full_backtest(self) -> None:
        result = self.engine.run_strategy(self.data, self.params)
        self.assertIsInstance(result, BacktestResult)
        self.assertGreater(len(result.equity_curve), 0)
        self.assertIsInstance(result.sharpe_ratio, float)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()

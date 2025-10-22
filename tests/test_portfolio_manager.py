"""Testy optymalizacji portfela w PortfolioManager."""

from __future__ import annotations

import logging
from typing import Dict

import numpy as np
import pandas as pd
import pytest

from bot_core.trading.engine import PortfolioManager, TradingEngine, TradingParameters


def _build_price_frame(returns: Dict[str, float] | list[float], start: float = 100.0) -> pd.DataFrame:
    """Tworzy ramkę cenową z zadaną listą zwrotów dziennych."""

    if isinstance(returns, dict):  # pragma: no cover - defensywnie
        returns = list(returns.values())

    returns_array = np.asarray(returns, dtype=float)
    index = pd.date_range("2024-01-01", periods=len(returns_array) + 1, freq="D")
    cumulative = np.concatenate(([1.0], np.cumprod(1 + returns_array)))
    prices = pd.Series(start * cumulative, index=index, name="close")
    return pd.DataFrame({"close": prices})


def _build_manager() -> PortfolioManager:
    engine = TradingEngine(logger=logging.getLogger("portfolio-tests"))
    return PortfolioManager(engine)


def test_optimize_portfolio_weights_prefers_higher_return() -> None:
    manager = _build_manager()
    assets = {
        "FAST": _build_price_frame([0.01] * 6),
        "SLOW": _build_price_frame([0.0] * 6),
    }
    params = {name: TradingParameters() for name in assets}

    weights = manager.optimize_portfolio_weights(assets, params, objective="sharpe_ratio")

    assert set(weights) == set(assets)
    assert pytest.approx(sum(weights.values()), rel=1e-6) == 1.0
    assert weights["FAST"] > weights["SLOW"]
    assert all(weight >= 0 for weight in weights.values())


def test_optimize_portfolio_weights_min_variance_prefers_stable_asset() -> None:
    manager = _build_manager()
    assets = {
        "VOL": _build_price_frame([0.02, -0.02, 0.025, -0.018, 0.022, -0.021]),
        "STABLE": _build_price_frame([0.004] * 6),
    }
    params = {name: TradingParameters() for name in assets}

    weights = manager.optimize_portfolio_weights(assets, params, objective="min_variance")

    assert pytest.approx(sum(weights.values()), rel=1e-6) == 1.0
    assert weights["STABLE"] > weights["VOL"]


def test_optimize_portfolio_weights_respects_position_bias() -> None:
    manager = _build_manager()
    assets = {
        "ASSET_A": _build_price_frame([0.005] * 5),
        "ASSET_B": _build_price_frame([0.005] * 5),
    }
    params = {
        "ASSET_A": TradingParameters(position_size=0.5),
        "ASSET_B": TradingParameters(position_size=1.5),
    }

    weights = manager.optimize_portfolio_weights(assets, params)

    assert pytest.approx(sum(weights.values()), rel=1e-6) == 1.0
    assert weights["ASSET_B"] > weights["ASSET_A"]


def test_optimize_portfolio_weights_risk_parity_prefers_lower_vol() -> None:
    manager = _build_manager()
    assets = {
        "RISKY": _build_price_frame([0.04, -0.04, 0.045, -0.038, 0.041, -0.042]),
        "CALM": _build_price_frame([0.01, -0.01, 0.011, -0.009, 0.0105, -0.0095]),
    }
    params = {name: TradingParameters() for name in assets}

    weights = manager.optimize_portfolio_weights(assets, params, objective="risk_parity")

    assert pytest.approx(sum(weights.values()), rel=1e-6) == 1.0
    assert weights["CALM"] > weights["RISKY"]


def test_optimize_portfolio_weights_risk_parity_respects_budgets() -> None:
    manager = _build_manager()
    assets = {
        "RISKY": _build_price_frame([0.03, -0.028, 0.031, -0.029, 0.032, -0.027]),
        "CALM": _build_price_frame([0.01, -0.01, 0.0095, -0.0095, 0.0102, -0.0101]),
    }
    params = {
        "RISKY": TradingParameters(max_position_risk=0.08),
        "CALM": TradingParameters(max_position_risk=0.01),
    }

    weights = manager.optimize_portfolio_weights(assets, params, objective="risk_parity")

    assert pytest.approx(sum(weights.values()), rel=1e-6) == 1.0
    assert weights["RISKY"] > weights["CALM"]


def test_optimize_portfolio_weights_respects_max_weight_cap() -> None:
    manager = _build_manager()
    assets = {
        "FAST": _build_price_frame([0.02] * 6),
        "SLOW": _build_price_frame([0.005] * 6),
    }
    params = {
        "FAST": TradingParameters(max_weight=0.4),
        "SLOW": TradingParameters(),
    }

    weights = manager.optimize_portfolio_weights(assets, params, objective="max_return")

    assert pytest.approx(sum(weights.values()), rel=1e-6) == 1.0
    assert weights["FAST"] == pytest.approx(0.4, abs=1e-4)
    assert weights["SLOW"] == pytest.approx(0.6, abs=1e-4)


def test_optimize_portfolio_weights_respects_min_weight_floor() -> None:
    manager = _build_manager()
    assets = {
        "A": _build_price_frame([0.01, 0.011, 0.012, 0.013, 0.009, 0.01]),
        "B": _build_price_frame([0.008, 0.007, 0.009, 0.0085, 0.0075, 0.008]),
        "C": _build_price_frame([0.006, 0.0065, 0.007, 0.0068, 0.0062, 0.0067]),
    }
    params = {
        "A": TradingParameters(min_weight=0.2),
        "B": TradingParameters(),
        "C": TradingParameters(),
    }

    weights = manager.optimize_portfolio_weights(assets, params, objective="sharpe_ratio")

    assert pytest.approx(sum(weights.values()), rel=1e-6) == 1.0
    assert weights["A"] >= 0.2 - 1e-4


def test_optimize_portfolio_weights_fallback_respects_bounds() -> None:
    manager = _build_manager()
    assets = {
        "FLOOR": pd.DataFrame(),
        "CAP": pd.DataFrame(),
    }
    params = {
        "FLOOR": TradingParameters(min_weight=0.35),
        "CAP": TradingParameters(max_weight=0.55),
    }

    weights = manager.optimize_portfolio_weights(assets, params, objective="sharpe_ratio")

    assert pytest.approx(sum(weights.values()), rel=1e-6) == 1.0
    assert weights["FLOOR"] >= 0.35 - 1e-4
    assert weights["CAP"] <= 0.55 + 1e-4


def test_optimize_portfolio_weights_handles_disjoint_histories() -> None:
    manager = _build_manager()
    early = _build_price_frame([0.01, 0.009, 0.011, 0.012])
    late = _build_price_frame([0.02, 0.019, 0.021, 0.018])
    late.index = late.index + pd.DateOffset(days=60)

    assets = {
        "EARLY": early,
        "LATE": late,
    }
    params = {
        "EARLY": TradingParameters(max_weight=0.6),
        "LATE": TradingParameters(min_weight=0.25),
    }

    weights = manager.optimize_portfolio_weights(assets, params, objective="sharpe_ratio")

    assert pytest.approx(sum(weights.values()), rel=1e-6) == 1.0
    assert weights["EARLY"] <= 0.6 + 1e-4
    assert weights["LATE"] >= 0.25 - 1e-4

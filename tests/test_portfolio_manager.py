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


def test_optimize_portfolio_weights_fallback_respects_bias() -> None:
    manager = _build_manager()
    assets = {
        "LOW": pd.DataFrame(),
        "HIGH": pd.DataFrame(),
        "MID": pd.DataFrame(),
    }
    params = {
        "LOW": TradingParameters(position_size=0.5),
        "HIGH": TradingParameters(position_size=2.0, max_weight=0.55),
        "MID": TradingParameters(position_size=1.0),
    }

    weights = manager.optimize_portfolio_weights(assets, params, objective="sharpe_ratio")

    assert pytest.approx(sum(weights.values()), rel=1e-6) == 1.0
    assert weights["HIGH"] == pytest.approx(0.55, abs=1e-4)
    assert weights["MID"] > weights["LOW"]


def test_optimize_portfolio_weights_retains_assets_without_data_for_bounds() -> None:
    manager = _build_manager()
    assets = {
        "DATA": _build_price_frame([0.012] * 8),
        "STALE": pd.DataFrame(),
    }
    params = {
        "DATA": TradingParameters(),
        "STALE": TradingParameters(min_weight=0.3),
    }

    weights = manager.optimize_portfolio_weights(assets, params, objective="max_return")

    assert pytest.approx(sum(weights.values()), rel=1e-6) == 1.0
    assert "STALE" in weights
    assert weights["STALE"] >= 0.3 - 1e-4


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


def test_optimize_portfolio_weights_turnover_handles_assets_without_data() -> None:
    manager = _build_manager()
    assets = {
        "ACTIVE": _build_price_frame([0.01] * 10),
        "STALE": pd.DataFrame(),
    }
    params = {name: TradingParameters() for name in assets}

    previous = {"ACTIVE": 0.3, "STALE": 0.7}

    weights = manager.optimize_portfolio_weights(
        assets,
        params,
        objective="sharpe_ratio",
        previous_weights=previous,
        max_turnover=0.05,
    )

    assert pytest.approx(sum(weights.values()), rel=1e-6) == 1.0
    assert "STALE" in weights
    turnover = 0.5 * sum(abs(weights[symbol] - previous.get(symbol, 0.0)) for symbol in weights)
    assert turnover <= 0.05 + 5e-3
    assert weights["STALE"] >= 0.65 - 5e-3


def test_optimize_portfolio_weights_zero_excess_respects_bounds() -> None:
    manager = _build_manager()
    assets = {
        "FLOOR": _build_price_frame([0.02, -0.02, 0.015, -0.015, 0.018, -0.018]),
        "REST": _build_price_frame([0.01, -0.01, 0.008, -0.008, 0.009, -0.009]),
    }
    params = {
        "FLOOR": TradingParameters(min_weight=0.6),
        "REST": TradingParameters(),
    }

    weights = manager.optimize_portfolio_weights(assets, params, objective="sharpe_ratio")

    assert pytest.approx(sum(weights.values()), rel=1e-6) == 1.0
    assert weights["FLOOR"] >= 0.6 - 1e-4
    assert weights["REST"] <= 0.4 + 1e-4


def test_optimize_portfolio_weights_limits_turnover() -> None:
    manager = _build_manager()
    assets = {
        "OLD": _build_price_frame([0.002] * 10),
        "NEW": _build_price_frame([0.01] * 10),
    }
    params = {name: TradingParameters() for name in assets}

    previous = {"OLD": 0.8, "NEW": 0.2}

    weights = manager.optimize_portfolio_weights(
        assets,
        params,
        objective="max_return",
        previous_weights=previous,
        max_turnover=0.1,
    )

    assert pytest.approx(sum(weights.values()), rel=1e-6) == 1.0
    turnover = 0.5 * sum(abs(weights[symbol] - previous.get(symbol, 0.0)) for symbol in weights)
    assert turnover <= 0.1 + 5e-3
    assert weights["NEW"] > previous["NEW"]


def test_optimize_portfolio_weights_turnover_respects_bounds() -> None:
    manager = _build_manager()
    assets = {
        "STAPLE": _build_price_frame([0.003] * 12),
        "GROWTH": _build_price_frame([0.012] * 12),
    }
    params = {
        "STAPLE": TradingParameters(min_weight=0.45),
        "GROWTH": TradingParameters(),
    }

    previous = {"STAPLE": 0.5, "GROWTH": 0.5}

    weights = manager.optimize_portfolio_weights(
        assets,
        params,
        objective="max_return",
        previous_weights=previous,
        max_turnover=0.05,
    )

    assert pytest.approx(sum(weights.values()), rel=1e-6) == 1.0
    assert weights["STAPLE"] >= 0.45 - 1e-4
    turnover = 0.5 * sum(abs(weights[symbol] - previous.get(symbol, 0.0)) for symbol in weights)
    assert turnover <= 0.05 + 5e-3


def test_optimize_portfolio_weights_limits_per_asset_turnover() -> None:
    manager = _build_manager()
    assets = {
        "DEFENSIVE": _build_price_frame([0.002] * 10),
        "CYCLICAL": _build_price_frame([0.012] * 10),
    }
    params = {
        "DEFENSIVE": TradingParameters(max_weight_change=0.05),
        "CYCLICAL": TradingParameters(),
    }

    previous = {"DEFENSIVE": 0.8, "CYCLICAL": 0.2}

    weights = manager.optimize_portfolio_weights(
        assets,
        params,
        objective="max_return",
        previous_weights=previous,
        max_turnover=0.5,
    )

    assert pytest.approx(sum(weights.values()), rel=1e-6) == 1.0
    assert 0.75 - 1e-4 <= weights["DEFENSIVE"] <= 0.85 + 1e-4


def test_optimize_portfolio_weights_zero_change_locks_asset() -> None:
    manager = _build_manager()
    assets = {
        "CORE": _build_price_frame([0.006] * 10),
        "SATELLITE": _build_price_frame([0.02] * 10),
    }
    params = {
        "CORE": TradingParameters(max_weight_change=0.0),
        "SATELLITE": TradingParameters(),
    }

    previous = {"CORE": 0.6, "SATELLITE": 0.4}

    weights = manager.optimize_portfolio_weights(
        assets,
        params,
        objective="max_return",
        previous_weights=previous,
        max_turnover=0.3,
    )

    assert pytest.approx(sum(weights.values()), rel=1e-6) == 1.0
    assert weights["CORE"] == pytest.approx(0.6, abs=1e-4)


def test_optimize_portfolio_weights_per_asset_limit_without_global_turnover() -> None:
    manager = _build_manager()
    assets = {
        "DEFENSIVE": _build_price_frame([0.002] * 10),
        "CYCLICAL": _build_price_frame([0.012] * 10),
    }
    params = {
        "DEFENSIVE": TradingParameters(max_weight_change=0.05),
        "CYCLICAL": TradingParameters(),
    }

    previous = {"DEFENSIVE": 0.8, "CYCLICAL": 0.2}

    weights = manager.optimize_portfolio_weights(
        assets,
        params,
        objective="max_return",
        previous_weights=previous,
    )

    assert pytest.approx(sum(weights.values()), rel=1e-6) == 1.0
    assert 0.75 - 1e-4 <= weights["DEFENSIVE"] <= 0.85 + 1e-4


def test_optimize_portfolio_weights_zero_change_without_global_turnover() -> None:
    manager = _build_manager()
    assets = {
        "CORE": _build_price_frame([0.006] * 10),
        "SATELLITE": _build_price_frame([0.02] * 10),
    }
    params = {
        "CORE": TradingParameters(max_weight_change=0.0),
        "SATELLITE": TradingParameters(),
    }

    previous = {"CORE": 0.6, "SATELLITE": 0.4}

    weights = manager.optimize_portfolio_weights(
        assets,
        params,
        objective="max_return",
        previous_weights=previous,
    )

    assert pytest.approx(sum(weights.values()), rel=1e-6) == 1.0
    assert weights["CORE"] == pytest.approx(0.6, abs=1e-4)

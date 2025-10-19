import numpy as np
import pandas as pd

from bot_core.risk.portfolio import (
    RiskLevel,
    RiskManagement,
    backtest_risk_strategy,
    calculate_optimal_leverage,
)


def _make_market_frame(length: int = 180) -> pd.DataFrame:
    base = np.linspace(100.0, 110.0, length)
    volume = np.linspace(1_000_000.0, 900_000.0, length)
    return pd.DataFrame({"close": base, "volume": volume})


def test_position_sizing_and_metrics_roundtrip() -> None:
    rm = RiskManagement({"max_portfolio_risk": 0.3, "max_risk_per_trade": 0.05})
    market_df = _make_market_frame()

    portfolio = {"ETH/USDT": {"size": 0.1, "volatility": 0.25}}
    sizing = rm.calculate_position_size(
        "BTC/USDT",
        {"strength": 0.6, "confidence": 0.7},
        market_df,
        portfolio,
    )

    assert 0.0 <= sizing.recommended_size <= sizing.max_allowed_size <= 1.0
    assert sizing.confidence_level > 0.0

    returns = market_df["close"].pct_change().dropna()
    rm.update_portfolio_state(95_000.0, portfolio, {"ETH/USDT": returns})

    metrics = rm.calculate_risk_metrics(
        portfolio,
        {"BTC/USDT": market_df, "ETH/USDT": market_df},
    )

    assert isinstance(metrics.var_95, float)
    assert isinstance(metrics.risk_level, RiskLevel)

    report = rm.generate_risk_report(
        portfolio,
        {"BTC/USDT": market_df, "ETH/USDT": market_df},
    )
    assert "RISK MANAGEMENT REPORT" in report

    emergency = rm.emergency_risk_check(70_000.0, 100_000.0, portfolio)
    assert set(emergency).issuperset(
        {
            "emergency_stop_required",
            "actions_required",
            "risk_alerts",
            "portfolio_heat",
        }
    )

    optimization = rm.optimize_portfolio_risk(portfolio, target_risk=0.15)
    assert "current_risk" in optimization


def test_backtest_and_leverage_helpers() -> None:
    rm = RiskManagement({"max_portfolio_risk": 0.25, "max_risk_per_trade": 0.04})
    market_df = _make_market_frame()

    np.random.seed(0)
    results = backtest_risk_strategy(
        rm,
        {"BTC/USDT": market_df},
        [
            {"symbol": "BTC/USDT", "strength": 0.8, "confidence": 0.9, "price": 105.0},
            {"symbol": "BTC/USDT", "strength": 0.2, "confidence": 0.4, "price": 103.0},
        ],
    )

    assert results["total_trades"] == 2
    assert results["risk_adjusted_ratio"] >= 0.0

    random_returns = pd.Series(np.random.normal(0.001, 0.02, 200))
    leverage = calculate_optimal_leverage(random_returns, target_volatility=0.2)
    assert 0.1 <= leverage <= 3.0

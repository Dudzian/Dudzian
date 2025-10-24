import pandas as pd

from bot_core.ai.regime import MarketRegime
import pytest

from bot_core.trading.engine import TradingParameters
from bot_core.trading.regime_workflow import RegimeSwitchWorkflow


def _sample_market_data(rows: int = 120) -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=rows, freq="h")
    base = pd.Series(range(rows), index=index, dtype=float)
    close = 100 + base.cumsum() * 0.01
    high = close + 0.5
    low = close - 0.5
    open_ = close.shift(1).fillna(close)
    volume = pd.Series(1000, index=index, dtype=float)
    return pd.DataFrame({
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    })


def test_regime_workflow_produces_parameters_with_weights() -> None:
    workflow = RegimeSwitchWorkflow(
        confidence_threshold=0.2,
        persistence_threshold=0.1,
        min_switch_cooldown=0,
    )
    decision = workflow.decide(_sample_market_data(), TradingParameters())

    assert isinstance(decision.parameters, TradingParameters)
    assert decision.parameters.ensemble_weights == decision.weights
    assert abs(sum(decision.weights.values()) - 1.0) < 1e-9
    assert decision.timestamp.tzinfo is None


def test_regime_workflow_respects_cooldown() -> None:
    workflow = RegimeSwitchWorkflow(
        confidence_threshold=0.2,
        persistence_threshold=0.05,
        min_switch_cooldown=5,
    )
    data = _sample_market_data()
    first = workflow.decide(data, TradingParameters())
    second = workflow.decide(data, TradingParameters())

    assert second.regime == first.regime


def test_regime_workflow_accepts_custom_configuration() -> None:
    custom_weights = {
        "trend": {"trend_following": 0.8, "arbitrage": 0.2},
        MarketRegime.DAILY: {"day_trading": 1.0},
    }
    custom_overrides = {
        MarketRegime.TREND: {"signal_threshold": 0.2},
        "mean_reversion": {"rsi_oversold": 40, "rsi_overbought": 60},
    }

    workflow = RegimeSwitchWorkflow(
        confidence_threshold=0.2,
        persistence_threshold=0.05,
        min_switch_cooldown=0,
        default_weights=custom_weights,
        default_parameter_overrides=custom_overrides,
    )

    decision = workflow.decide(_sample_market_data(), TradingParameters())

    assert workflow.default_strategy_weights[MarketRegime.TREND]["trend_following"] == 0.8
    assert decision.parameters.signal_threshold == 0.2
    assert workflow.default_parameter_overrides[MarketRegime.MEAN_REVERSION]["rsi_oversold"] == 40


def test_regime_workflow_updates_defaults() -> None:
    workflow = RegimeSwitchWorkflow()

    workflow.update_default_weights({"trend": {"trend_following": 0.9}}, replace=False)
    weights = workflow.default_strategy_weights[MarketRegime.TREND]
    assert weights["trend_following"] == pytest.approx(0.9)
    assert "day_trading" in weights  # pozostali członkowie pozostają przy replace=False

    workflow.update_default_weights(
        {MarketRegime.TREND: {"trend_following": 0.6, "arbitrage": 0.4}}, replace=True
    )
    replaced = workflow.default_strategy_weights[MarketRegime.TREND]
    assert set(replaced) == {"trend_following", "arbitrage"}
    assert replaced["trend_following"] == pytest.approx(0.6)
    assert replaced["arbitrage"] == pytest.approx(0.4)

    workflow.update_parameter_overrides({"daily": {"signal_threshold": 0.25}})
    overrides = workflow.default_parameter_overrides[MarketRegime.DAILY]
    assert overrides["signal_threshold"] == pytest.approx(0.25)

    workflow.update_parameter_overrides(
        {MarketRegime.MEAN_REVERSION: {"rsi_oversold": 45, "rsi_overbought": 55}},
        replace=True,
    )
    replaced_overrides = workflow.default_parameter_overrides[MarketRegime.MEAN_REVERSION]
    assert replaced_overrides == {"rsi_oversold": 45, "rsi_overbought": 55}

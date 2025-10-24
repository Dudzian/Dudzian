import pandas as pd

from bot_core.ai.regime import MarketRegime
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
    assert decision.license_tiers and "standard" in decision.license_tiers
    assert "trend_following" in decision.strategy_metadata
    strategy_meta = decision.strategy_metadata["trend_following"]
    assert strategy_meta["license_tier"] == "standard"
    assert "trend_d1" in decision.capabilities
    assert tuple(strategy_meta["risk_classes"]) == ("directional", "momentum")
    assert set(strategy_meta["required_data"]) == {"ohlcv", "technical_indicators"}
    assert set(strategy_meta["tags"]) >= {"trend", "momentum"}
    assert "momentum" in decision.tags
    assert "technical_indicators" in decision.required_data


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

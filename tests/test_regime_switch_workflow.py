from datetime import datetime, timezone

import pandas as pd

from bot_core.ai.regime import (
    MarketRegime,
    MarketRegimeAssessment,
    MarketRegimeClassifier,
    RegimeHistory,
)
from bot_core.strategies import StrategyPresetWizard
from bot_core.strategies.regime_workflow import RegimePresetActivation, StrategyRegimeWorkflow


def _sample_market_data(rows: int = 120) -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=rows, freq="h")
    base = pd.Series(range(rows), index=index, dtype=float)
    close = 100 + base.cumsum() * 0.01
    high = close + 0.5
    low = close - 0.5
    open_ = close.shift(1).fillna(close)
    volume = pd.Series(1000, index=index, dtype=float)
    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )


def _workflow() -> StrategyRegimeWorkflow:
    classifier = MarketRegimeClassifier()
    history = RegimeHistory(thresholds_loader=classifier.thresholds_loader)
    history.reload_thresholds(thresholds=classifier.thresholds_snapshot())
    return StrategyRegimeWorkflow(
        wizard=StrategyPresetWizard(),
        classifier=classifier,
        history=history,
    )


class _StaticClassifier:
    def __init__(self, regime: MarketRegime) -> None:
        self._regime = regime
        self.thresholds_loader = lambda: {}
        self.thresholds_snapshot = lambda: {}

    def assess(self, market_data, *, symbol: str | None = None) -> MarketRegimeAssessment:
        return MarketRegimeAssessment(
            regime=self._regime,
            confidence=0.9,
            risk_score=0.3,
            metrics={},
            symbol=symbol or "TEST",
        )


def test_strategy_regime_workflow_produces_activation_metadata() -> None:
    classifier = _StaticClassifier(MarketRegime.TREND)
    history = RegimeHistory(thresholds_loader=classifier.thresholds_loader)
    history.reload_thresholds(thresholds=classifier.thresholds_snapshot())
    workflow = StrategyRegimeWorkflow(
        wizard=StrategyPresetWizard(),
        classifier=classifier,  # type: ignore[arg-type]
        history=history,
    )
    signing_key = b"test-key"
    workflow.register_preset(
        MarketRegime.TREND,
        name="trend-core",
        entries=[{"engine": "daily_trend_momentum"}],
        signing_key=signing_key,
        metadata={"ensemble_weights": {"trend_following": 1.0}},
    )

    activation = workflow.activate(
        _sample_market_data(),
        available_data={"ohlcv", "technical_indicators"},
    )

    assert isinstance(activation, RegimePresetActivation)
    assert activation.regime is MarketRegime.TREND
    assert activation.version.metadata["license_tiers"]
    preset_meta = activation.version.metadata.get("preset_metadata", {})
    assert preset_meta["ensemble_weights"]["trend_following"] == 1.0
    assert activation.decision_candidates
    assert activation.activated_at.tzinfo is timezone.utc


def test_strategy_regime_workflow_uses_fallback_when_data_missing() -> None:
    classifier = _StaticClassifier(MarketRegime.MEAN_REVERSION)
    history = RegimeHistory(thresholds_loader=classifier.thresholds_loader)
    history.reload_thresholds(thresholds=classifier.thresholds_snapshot())
    workflow = StrategyRegimeWorkflow(
        wizard=StrategyPresetWizard(),
        classifier=classifier,  # type: ignore[arg-type]
        history=history,
    )
    signing_key = b"test-key"
    workflow.register_preset(
        MarketRegime.MEAN_REVERSION,
        name="mean-core",
        entries=[{"engine": "mean_reversion"}],
        signing_key=signing_key,
        metadata={"ensemble_weights": {"mean_reversion": 1.0}},
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

    activation = workflow.activate(
        _sample_market_data(),
        available_data={"ohlcv", "technical_indicators"},  # brak spread_history wymusi fallback
    )

    assert activation.used_fallback is True
    assert activation.blocked_reason == "missing_data"
    assert "spread_history" in activation.missing_data


def test_strategy_regime_workflow_preserves_custom_metadata() -> None:
    workflow = _workflow()
    signing_key = b"test-key"
    workflow.register_preset(
        MarketRegime.TREND,
        name="trend-custom",
        entries=[
            {
                "engine": "daily_trend_momentum",
                "metadata": {"custom_flag": True},
            }
        ],
        signing_key=signing_key,
        metadata={"ensemble_weights": {"trend_following": 1.0}, "notes": "demo"},
    )

    activation = workflow.activate(
        _sample_market_data(),
        available_data={"ohlcv", "technical_indicators"},
        now=datetime(2024, 1, 1, 9, 0, tzinfo=timezone.utc),
    )

    preset_meta = activation.version.metadata["preset_metadata"]
    assert preset_meta["notes"] == "demo"
    strategy_entry = activation.preset["strategies"][0]
    assert strategy_entry["metadata"]["custom_flag"] is True

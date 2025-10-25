from collections import Counter
from types import MappingProxyType
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
    workflow = RegimeSwitchWorkflow(
        classifier=classifier,  # type: ignore[arg-type]
        history=history,
    )
    decision = workflow.decide(_sample_market_data(), TradingParameters())

    assert isinstance(decision.parameters, TradingParameters)
    assert decision.parameters.ensemble_weights == decision.weights
    assert abs(sum(decision.weights.values()) - 1.0) < 1e-9
    expected = {
        "mean_reversion",
        "statistical_arbitrage",
        "arbitrage",
        "grid_trading",
        "options_income",
        "scalping",
    }
    assert set(decision.weights) == expected
    assert decision.timestamp.tzinfo is None
    assert decision.license_tiers and "professional" in decision.license_tiers
    assert "mean_reversion" in decision.strategy_metadata
    strategy_meta = decision.strategy_metadata["mean_reversion"]
    assert strategy_meta["license_tier"] == "professional"
    assert "mean_reversion" in decision.capabilities
    assert set(strategy_meta["risk_classes"]) == {"statistical", "mean_reversion"}
    assert set(strategy_meta["required_data"]) >= {"ohlcv", "spread_history"}
    assert set(strategy_meta["tags"]) >= {"mean_reversion", "stat_arbitrage"}
    assert "stat_arbitrage" in decision.tags
    assert "spread_history" in decision.required_data


def test_regime_workflow_respects_cooldown() -> None:
    strategy_workflow = _workflow()
    strategy_workflow.register_preset(
        MarketRegime.TREND,
        name="trend-core",
        entries=[
            {
                "engine": "daily_trend_momentum",
                "required_data": ("ohlcv", "latency_monitoring"),
            }
        ],
        signing_key=b"test-key",
        metadata={"ensemble_weights": {"trend_following": 1.0}},
    )
    strategy_workflow.register_emergency_preset(
        name="fallback",
        entries=[
            {
                "engine": "mean_reversion",
                "required_data": ("ohlcv",),
            }
        ],
        signing_key=b"test-key",
        metadata={"ensemble_weights": {"mean_reversion": 1.0}},
    )

    activation = strategy_workflow.activate(
        _sample_market_data(),
        available_data={"ohlcv", "spread_history"},
    )

    assert activation.used_fallback is True
    assert activation.blocked_reason == "missing_data"
    assert "latency_monitoring" in activation.missing_data


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


def test_regime_workflow_collect_metadata_uses_aliases() -> None:
    class _AliasCatalog:
        def __init__(self) -> None:
            self.calls = Counter()

        def available(self):  # noqa: D401 - test double
            return ("day_trading",)

        def metadata_for(self, name: str):  # noqa: D401 - test double
            self.calls[name] += 1
            if name == "day_trading":
                return MappingProxyType(
                    {
                        "license_tier": "standard",
                        "risk_classes": ("intraday", "momentum"),
                        "required_data": ("ohlcv", "technical_indicators"),
                        "capability": "day_trading",
                        "tags": ("intraday", "momentum"),
                    }
                )
            return MappingProxyType({})

    workflow = RegimeSwitchWorkflow(catalog=_AliasCatalog())
    workflow._strategy_metadata_cache.clear()

    metadata = workflow._collect_strategy_metadata(
        {"intraday_breakout_probing": 1.0}
    )

    strategies = metadata["strategies"]
    probing = dict(strategies["intraday_breakout_probing"])
    assert probing["catalog_name"] == "day_trading"
    assert "intraday_breakout" in probing["aliases"]
    assert probing["capability"] == "day_trading"
    assert metadata["license_tiers"] == ("standard",)
    assert metadata["risk_classes"] == ("intraday", "momentum")
    assert metadata["required_data"] == ("ohlcv", "technical_indicators")
    assert metadata["tags"] == ("intraday", "momentum")
    assert workflow.catalog.calls["day_trading"] == 1  # type: ignore[attr-defined]


def test_regime_workflow_collect_metadata_caches_results() -> None:
    class _CountingCatalog:
        def __init__(self) -> None:
            self.calls = Counter()

        def available(self):  # noqa: D401 - test double
            return ("custom",)

        def metadata_for(self, name: str):  # noqa: D401 - test double
            self.calls[name] += 1
            if name == "custom":
                return MappingProxyType({"license_tier": "premium"})
            return MappingProxyType({})

    catalog = _CountingCatalog()
    workflow = RegimeSwitchWorkflow(catalog=catalog)
    workflow._strategy_metadata_cache.clear()

    workflow._collect_strategy_metadata({"custom": 1.0})
    workflow._collect_strategy_metadata({"custom": 1.0})

    assert catalog.calls["custom"] == 1


def test_regime_workflow_collect_metadata_caches_missing_entries() -> None:
    class _MissingCatalog:
        def __init__(self) -> None:
            self.calls = Counter()

        def available(self):  # noqa: D401 - test double
            return ()

        def metadata_for(self, name: str):  # noqa: D401 - test double
            self.calls[name] += 1
            return MappingProxyType({})

    catalog = _MissingCatalog()
    workflow = RegimeSwitchWorkflow(catalog=catalog)
    workflow._strategy_metadata_cache.clear()

    workflow._collect_strategy_metadata({"unknown": 1.0})
    workflow._collect_strategy_metadata({"unknown": 1.0})

    assert catalog.calls["unknown"] == 1

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from types import SimpleNamespace

import pandas as pd

from bot_core.ai.regime import MarketRegime, MarketRegimeAssessment, RiskLevel
from bot_core.auto_trader.app import AutoTrader, RiskDecision
from bot_core.observability import metrics as metrics_module


class DummyEmitter:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, object]]] = []

    def emit(self, event: str, **payload: object) -> None:
        self.events.append((event, dict(payload)))

    def log(self, *_args: object, **_kwargs: object) -> None:  # pragma: no cover - not used in tests
        return None


class SequenceMarketDataProvider:
    def __init__(self, frames: list[pd.DataFrame]) -> None:
        self._frames = deque(frames)

    def get_historical(self, symbol: str, timeframe: str, limit: int = 256) -> pd.DataFrame:
        frame = self._frames[0]
        self._frames.rotate(-1)
        return frame


class SequencedAIManager:
    def __init__(self, scenarios: list[dict[str, object]]) -> None:
        self._scenarios = deque(scenarios)
        self.ai_threshold_bps = 5.0
        self.is_degraded = False

    def assess_market_regime(self, symbol: str, market_data: pd.DataFrame) -> MarketRegimeAssessment:
        return self._scenarios[0]["assessment"]  # type: ignore[return-value]

    def get_regime_summary(self, symbol: str):
        return self._scenarios[0].get("summary")

    def predict_series(self, symbol: str, market_data: pd.DataFrame, feature_cols: list[str] | None = None) -> pd.Series:
        scenario = self._scenarios[0]
        index = market_data.index[-1:]
        series = pd.Series([scenario.get("prediction", 0.0)], index=index)
        self._scenarios.rotate(-1)
        return series


class RecordingExecutionService:
    def __init__(self) -> None:
        self.decisions: list[RiskDecision] = []

    def execute_decision(self, decision: RiskDecision) -> None:
        self.decisions.append(decision)


class StubAlertRouter:
    def __init__(self) -> None:
        self.messages: list = []

    def dispatch(self, message) -> None:  # pragma: no cover - simple collector
        self.messages.append(message)


@dataclass
class SimpleSchedule:
    strategy: str
    interval: object
    next_run: datetime


class SequencedOrchestrator:
    def __init__(self, strategies: list[str]) -> None:
        self._strategies = deque(strategies)
        self._schedules: list[SimpleSchedule] = []

    def select_strategy(self, regime: MarketRegime) -> str | None:
        if not self._strategies:
            return None
        return self._strategies.popleft()

    def schedule_strategy_recalibration(self, strategy: str, interval, *, first_run: datetime | None = None) -> SimpleSchedule:
        schedule = SimpleSchedule(strategy=strategy, interval=interval, next_run=first_run or datetime.now(timezone.utc))
        self._schedules.append(schedule)
        return schedule

    def due_recalibrations(self, now: datetime | None = None) -> tuple[SimpleSchedule, ...]:
        return tuple(self._schedules)

    def mark_recalibrated(self, strategy: str) -> None:
        self._schedules = [item for item in self._schedules if item.strategy != strategy]


class SummaryStub:
    def __init__(self, **values: object) -> None:
        self.__dict__.update(values)

    def __getattr__(self, item: str) -> object:  # pragma: no cover - fallback for optional metrics
        return self.__dict__.get(item, 0.0)

    def to_dict(self) -> dict[str, object]:  # pragma: no cover - helper for logging paths
        return dict(self.__dict__)


def _build_assessment(regime: MarketRegime, *, risk: float) -> MarketRegimeAssessment:
    return MarketRegimeAssessment(
        regime=regime,
        confidence=0.75,
        risk_score=risk,
        metrics={"volatility": 0.1},
    )


def test_auto_trader_lifecycle_with_guardrails_and_recalibrations() -> None:
    metrics_module._GLOBAL_REGISTRY = metrics_module.MetricsRegistry()

    frames = [
        pd.DataFrame({"close": [100, 101, 102, 103, 104]}, index=pd.date_range("2024-01-01", periods=5, freq="H")),
        pd.DataFrame({"close": [105, 104, 103, 102, 101]}, index=pd.date_range("2024-01-02", periods=5, freq="H")),
        pd.DataFrame({"close": [100, 101, 100, 99, 98]}, index=pd.date_range("2024-01-03", periods=5, freq="H")),
    ]

    scenarios = [
        {
            "assessment": _build_assessment(MarketRegime.TREND, risk=0.2),
            "summary": None,
            "prediction": 0.003,
        },
        {
            "assessment": _build_assessment(MarketRegime.MEAN_REVERSION, risk=0.35),
            "summary": None,
            "prediction": 0.004,
        },
        {
            "assessment": _build_assessment(MarketRegime.TREND, risk=0.62),
            "summary": None,
            "prediction": -0.005,
        },
    ]

    orchestrator = SequencedOrchestrator([
        "trend_following",
        "mean_reversion",
        "capital_preservation",
    ])

    provider = SequenceMarketDataProvider(frames)
    ai_manager = SequencedAIManager(scenarios)
    emitter = DummyEmitter()
    alert_router = StubAlertRouter()
    execution_service = RecordingExecutionService()

    trader = AutoTrader(
        emitter=emitter,
        gui=SimpleNamespace(),
        symbol_getter=lambda: "BTCUSDT",
        market_data_provider=provider,
        enable_auto_trade=True,
        auto_trade_interval_s=0.0,
    )
    trader.ai_manager = ai_manager
    trader.decision_orchestrator = orchestrator
    trader.alert_router = alert_router
    trader.execution_service = execution_service
    trader.core_execution_service = execution_service
    trader.core_risk_engine = None
    trader.risk_service = None
    trader._adjust_strategy_parameters = lambda *args, **kwargs: None  # type: ignore[assignment]
    trader._evaluate_decision_candidate = lambda **kwargs: SimpleNamespace(  # type: ignore[assignment]
        accepted=True,
        reasons=(),
        thresholds_snapshot={},
        model_name="stub",
    )

    trader._thresholds["auto_trader"]["signal_guardrails"] = {"effective_risk_cap": 0.5}

    trader.schedule_strategy_recalibration("mean_reversion", interval_s=0.0)

    for _ in range(3):
        trader._auto_trade_loop()

    assert len(execution_service.decisions) == 2
    assert trader.current_strategy == "capital_preservation"

    registry = metrics_module.get_global_metrics_registry()
    guardrail_metric = registry.get("auto_trader_guardrail_blocks_total")
    guardrail_names = {trigger.name for trigger in trader._last_guardrail_triggers}
    assert trader._last_guardrail_reasons, f"guardrail reasons missing (names={guardrail_names})"
    assert guardrail_metric.value(
        labels={
            "environment": "paper",
            "portfolio": "autotrader",
            "risk_profile": "paper",
            "guardrail": "effective_risk",
        }
    ) == 1.0

    recalibration_metric = registry.get("auto_trader_recalibrations_triggered_total")
    assert recalibration_metric.value(
        labels={
            "environment": "paper",
            "portfolio": "autotrader",
            "risk_profile": "paper",
            "strategy": "mean_reversion",
        }
    ) == 1.0

    categories = {message.category for message in alert_router.messages}
    assert "auto_trader.guardrail" in categories
    assert "auto_trader.recalibration" in categories

    assert orchestrator.due_recalibrations() == ()


def _stub_guardrail_summary(*_args, **_kwargs):
    return {}


def _stub_decision_summary(*_args, **_kwargs):
    return {}


def _build_trader_for_snapshot() -> AutoTrader:
    metrics_module._GLOBAL_REGISTRY = metrics_module.MetricsRegistry()
    trader = AutoTrader(
        emitter=DummyEmitter(),
        gui=SimpleNamespace(),
        symbol_getter=lambda: "BTCUSDT",
        enable_auto_trade=False,
        auto_trade_interval_s=0.0,
    )
    trader.summarize_guardrail_timeline = _stub_guardrail_summary  # type: ignore[assignment]
    trader.summarize_risk_decision_timeline = _stub_decision_summary  # type: ignore[assignment]
    return trader


def test_lifecycle_snapshot_includes_strategy_metadata() -> None:
    trader = _build_trader_for_snapshot()
    trader.current_strategy = "trend_following"

    snapshot = trader.build_lifecycle_snapshot()

    strategy = snapshot["strategy"]
    metadata = strategy.get("metadata")
    summary = strategy.get("metadata_summary")

    assert metadata is not None
    assert metadata["name"] == "trend_following"
    assert metadata["license_tier"] == "standard"
    assert metadata["risk_classes"] == ["directional", "momentum"]
    assert metadata["required_data"] == ["ohlcv", "technical_indicators"]
    assert metadata["capability"] == "trend_d1"
    assert metadata["tags"] == ["trend", "momentum"]
    assert summary == {
        "license_tiers": ["standard"],
        "risk_classes": ["directional", "momentum"],
        "required_data": ["ohlcv", "technical_indicators"],
        "capabilities": ["trend_d1"],
        "tags": ["trend", "momentum"],
    }


def test_lifecycle_snapshot_metadata_uses_aliases_for_probing_variants() -> None:
    trader = _build_trader_for_snapshot()
    trader.current_strategy = "intraday_breakout_probing"

    snapshot = trader.build_lifecycle_snapshot()

    strategy = snapshot["strategy"]
    metadata = strategy.get("metadata")
    summary = strategy.get("metadata_summary")

    assert metadata is not None
    assert metadata["name"] == "intraday_breakout_probing"
    assert metadata["catalog_name"] == "day_trading"
    assert metadata["license_tier"] == "standard"
    assert metadata["capability"] == "day_trading"
    assert summary == {
        "license_tiers": ["standard"],
        "risk_classes": ["intraday", "momentum"],
        "required_data": ["ohlcv", "technical_indicators"],
        "capabilities": ["day_trading"],
        "tags": ["intraday", "momentum"],
    }

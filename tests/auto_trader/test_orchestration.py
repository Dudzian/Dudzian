from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any

import pandas as pd

from bot_core.ai.models import ModelScore
from bot_core.ai.regime import MarketRegime, MarketRegimeAssessment
from bot_core.auto_trader import AutoTrader
from bot_core.auto_trader.schedule import ScheduleState
from bot_core.runtime.journal import InMemoryTradingDecisionJournal


class DummyEmitter:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, object]]] = []

    def emit(self, event: str, **payload: object) -> None:
        self.events.append((event, dict(payload)))

    def log(self, *_args: object, **_kwargs: object) -> None:
        return None


class SequenceSchedule:
    def __init__(self, states: list[ScheduleState]) -> None:
        self._states = deque(states)
        self.default_mode = states[0].mode
        self.allow_trading = states[0].is_open

    def describe(self, *_args: object) -> ScheduleState:
        if not self._states:
            raise RuntimeError("Brak zdefiniowanych stanów harmonogramu testowego")
        state = self._states[0]
        if len(self._states) > 1:
            self._states.popleft()
        return state


class ApprovingRiskService:
    def evaluate_decision(self, decision: Any) -> Any:
        return SimpleNamespace(approved=True, normalized=True, decision=decision)


class RecordingExecutionService:
    def __init__(self) -> None:
        self.calls: list[Any] = []

    def execute_decision(self, decision: Any) -> None:
        self.calls.append(decision)


class AcceptingOrchestrator:
    def evaluate_candidate(self, *_args: object, **_kwargs: object) -> Any:
        return SimpleNamespace(
            accepted=True,
            reasons=(),
            thresholds_snapshot={},
            model_name="stub",
        )


class SummaryStub:
    def __init__(self, **values: object) -> None:
        self.__dict__.update(values)

    def __getattr__(self, item: str) -> object:
        return self.__dict__.get(item, 0.0)

    def to_dict(self) -> dict[str, object]:
        return dict(self.__dict__)


def _build_market_data() -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=5, freq="h")
    return pd.DataFrame(
        {
            "open": [100, 101, 102, 103, 104],
            "high": [101, 102, 103, 104, 105],
            "low": [99, 100, 101, 102, 103],
            "close": [100, 101, 102, 103, 104],
            "volume": [10, 11, 12, 13, 14],
        },
        index=index,
    )


def _build_assessment(risk: float, confidence: float = 0.8) -> MarketRegimeAssessment:
    return MarketRegimeAssessment(
        regime=MarketRegime.TREND,
        confidence=confidence,
        risk_score=risk,
        metrics={"volatility": 0.1},
    )


def test_schedule_transitions_are_recorded() -> None:
    journal = InMemoryTradingDecisionJournal()
    initial_state = ScheduleState(
        mode="demo",
        is_open=True,
        window=None,
        next_transition=None,
        reference_time=datetime.now(timezone.utc),
    )
    closed_state = ScheduleState(
        mode="live",
        is_open=False,
        window=None,
        next_transition=None,
        reference_time=datetime.now(timezone.utc),
    )
    reopened_state = ScheduleState(
        mode="live",
        is_open=True,
        window=None,
        next_transition=None,
        reference_time=datetime.now(timezone.utc),
    )
    schedule = SequenceSchedule([initial_state, closed_state, reopened_state])

    trader = AutoTrader(
        emitter=DummyEmitter(),
        gui=SimpleNamespace(),
        symbol_getter=lambda: "BTCUSDT",
        enable_auto_trade=True,
        auto_trade_interval_s=0.0,
        work_schedule=schedule,
        decision_journal=journal,
    )

    trader.decision_orchestrator = AcceptingOrchestrator()

    trader._auto_trade_loop()
    trader._auto_trade_loop()

    events = list(journal.export())
    assert any(event["event"] == "schedule_blocked" for event in events)
    assert any(event["event"] == "schedule_transition" and event["status"] == "closed" for event in events)


def test_guardrail_blocks_trade_and_logs() -> None:
    market_data = _build_market_data()

    class GuardrailAI:
        ai_threshold_bps = 5.0
        is_degraded = False

        def assess_market_regime(self, symbol: str, market_data: pd.DataFrame) -> MarketRegimeAssessment:
            return _build_assessment(risk=0.6)

        def get_regime_summary(self, symbol: str) -> None:
            return None

        def predict_series(self, symbol: str, market_data: pd.DataFrame) -> pd.Series:
            return pd.Series([0.005], index=market_data.index[-1:])

        def run_due_training_jobs(self) -> tuple[()]:
            return ()

    execution = RecordingExecutionService()
    journal = InMemoryTradingDecisionJournal()

    trader = AutoTrader(
        emitter=DummyEmitter(),
        gui=SimpleNamespace(),
        symbol_getter=lambda: "BTCUSDT",
        market_data_provider=lambda **_: market_data,
        enable_auto_trade=True,
        auto_trade_interval_s=0.0,
        decision_journal=journal,
    )
    trader.ai_manager = GuardrailAI()
    trader.execution_service = execution
    trader.risk_service = ApprovingRiskService()
    trader._thresholds["auto_trader"]["signal_guardrails"]["effective_risk_cap"] = 0.4
    cooldown_cfg = trader._thresholds["auto_trader"].setdefault("cooldown", {})
    critical_cfg = cooldown_cfg.setdefault("critical", {})
    critical_cfg["risk"] = 1.1
    release_cfg = cooldown_cfg.setdefault("release", {})
    release_cfg["risk"] = 1.1
    high_risk_cfg = cooldown_cfg.setdefault("high_risk_fallback", {})
    high_risk_cfg["risk"] = 1.1
    release_active_cfg = cooldown_cfg.setdefault("release_active", {})
    release_active_cfg["risk"] = 1.1
    trader.decision_orchestrator = AcceptingOrchestrator()

    trader._auto_trade_loop()

    assert not execution.calls, "Guardrail powinien zablokować wykonanie transakcji"
    events = list(journal.export())
    guardrail_events = [event for event in events if event["event"] == "decision_guardrail"]
    assert guardrail_events, "Powinno zostać zapisane zdarzenie guardrail"
    assert "reasons" in guardrail_events[0] and "effective" in guardrail_events[0]["reasons"]


def test_ai_scoring_failure_triggers_fallback() -> None:
    market_data = _build_market_data()

    @dataclass
    class FailingAI:
        ai_threshold_bps: float = 3.0
        is_degraded: bool = False

        def assess_market_regime(self, symbol: str, market_data: pd.DataFrame) -> MarketRegimeAssessment:
            return _build_assessment(risk=0.25, confidence=0.9)

        def get_regime_summary(self, symbol: str) -> None:
            return None

        def predict_series(self, symbol: str, market_data: pd.DataFrame) -> pd.Series:
            return pd.Series([0.01], index=market_data.index[-1:])

        def score_decision_features(self, features: dict[str, float]) -> ModelScore:
            raise RuntimeError("scoring unavailable")

        def run_due_training_jobs(self) -> tuple[()]:
            return ()

    class LiveSchedule:
        def __init__(self) -> None:
            now = datetime.now(timezone.utc)
            self.state = ScheduleState(
                mode="live",
                is_open=True,
                window=None,
                next_transition=None,
                reference_time=now,
            )

        def describe(self, *_args: object) -> ScheduleState:
            return self.state

    execution = RecordingExecutionService()
    journal = InMemoryTradingDecisionJournal()

    trader = AutoTrader(
        emitter=DummyEmitter(),
        gui=SimpleNamespace(),
        symbol_getter=lambda: "BTCUSDT",
        market_data_provider=lambda **_: market_data,
        enable_auto_trade=True,
        auto_trade_interval_s=0.0,
        work_schedule=LiveSchedule(),
        decision_journal=journal,
    )
    trader.ai_manager = FailingAI()
    trader.execution_service = execution
    trader.risk_service = ApprovingRiskService()
    trader.decision_orchestrator = AcceptingOrchestrator()

    trader._auto_trade_loop()

    assert trader._ai_degraded is True
    assert not execution.calls
    events = list(journal.export())
    assert any(event["event"] == "ai_decision_fallback" for event in events)

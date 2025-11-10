from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone, tzinfo
from types import MethodType, SimpleNamespace
from typing import Any, Mapping, Sequence

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


def test_auto_mode_snapshot_without_signal_quality_provider() -> None:
    trader = AutoTrader(
        emitter=DummyEmitter(),
        gui=SimpleNamespace(),
        symbol_getter=lambda: "ETHUSDT",
        enable_auto_trade=False,
        auto_trade_interval_s=0.0,
    )

    snapshot = trader.build_auto_mode_snapshot(include_history=False)

    guardrail_state = snapshot["guardrail_state"]
    assert guardrail_state["exchange_degradation"]["score"] == 0.0
    assert guardrail_state["exchange_degradation"]["payload"] == {}

    failover_snapshot = snapshot["failover"]
    assert failover_snapshot["guardrail_active"] is False
    assert failover_snapshot["kill_switch"] is False
    assert failover_snapshot["degradation"] == {}

    assert snapshot["signal_quality"] == {}


def test_auto_mode_snapshot_enriches_retraining_cycles_with_decisions() -> None:
    class JournalStub:
        def __init__(self) -> None:
            self._records: list[dict[str, object]] = []

        def record(self, event: object) -> None:  # pragma: no cover - zgodność z kontraktem
            self._records.append(dict(event) if isinstance(event, Mapping) else {"event": event})

        def export(self) -> list[dict[str, object]]:
            return list(self._records)

    journal = JournalStub()
    journal._records.extend(  # noqa: SLF001 - dane testowe
        [
            {
                "event": "order_submitted",
                "timestamp": "2025-01-01T10:00:00+00:00",
                "decision_id": "DEC-1",
                "status": "submitted",
            },
            {
                "event": "order_filled",
                "timestamp": "2025-01-01T11:00:00+00:00",
                "decision_id": "DEC-2",
                "status": "filled",
            },
            {
                "event": "order_cancelled",
                "timestamp": "2025-01-01T12:00:00+00:00",
                "decision_id": "DEC-3",
                "status": "cancelled",
            },
        ]
    )

    trader = AutoTrader(
        emitter=DummyEmitter(),
        gui=SimpleNamespace(),
        symbol_getter=lambda: "BTCUSDT",
        enable_auto_trade=False,
        auto_trade_interval_s=0.0,
        decision_journal=journal,
    )

    @dataclass
    class CycleRecord:
        cycle_id: str
        decisions: list[Any]

    trader._retraining_cycle_log = [  # noqa: SLF001 - ustawienie stanu testowego
        {
            "cycle_id": "cycle-1",
            "status": "completed",
            "decision_ids": ["DEC-1", "DEC-2"],
        },
        CycleRecord(
            cycle_id="cycle-2",
            decisions=[
                SimpleNamespace(
                    decision_id="DEC-3",
                    timestamp=datetime(2025, 1, 1, 12, 30, tzinfo=timezone.utc),
                    state="aborted",
                )
            ],
        ),
        "unexpected",
    ]

    snapshot = trader.build_auto_mode_snapshot(include_history=False)

    cycles = snapshot["retraining_cycles"]
    assert len(cycles) == 3

    first_cycle = cycles[0]
    assert first_cycle["decision_ids"] == ["DEC-1", "DEC-2"]
    assert {entry["decision_id"] for entry in first_cycle["decisions"]} == {"DEC-1", "DEC-2"}

    second_cycle = cycles[1]
    assert second_cycle["decisions"][0]["decision_id"] == "DEC-3"
    assert isinstance(second_cycle["decisions"][0]["timestamp"], str)

    assert cycles[2]["decisions"] == []


def test_auto_mode_snapshot_exposes_decision_lookup_and_guardrail_details() -> None:
    class JournalStub:
        def __init__(self, records: Sequence[Mapping[str, Any]]) -> None:
            self._records = [dict(record) for record in records]

        def record(self, event: Mapping[str, Any] | Any) -> None:  # pragma: no cover - zgodność z kontraktem
            if isinstance(event, Mapping):
                self._records.append(dict(event))
            else:
                self._records.append({"event": event})

        def export(self) -> list[Mapping[str, Any]]:
            return [dict(record) for record in self._records]

    journal = JournalStub(
        [
            {
                "event": "order_submitted",
                "timestamp": "2025-01-01T09:00:00+00:00",
                "decision_id": "DEC-1",
                "status": "submitted",
            },
            {
                "event": "order_filled",
                "timestamp": "2025-01-01T10:00:00+00:00",
                "decision_id": "DEC-2",
                "status": "filled",
            },
        ]
    )

    trader = AutoTrader(
        emitter=DummyEmitter(),
        gui=SimpleNamespace(),
        symbol_getter=lambda: "BTCUSDT",
        enable_auto_trade=False,
        auto_trade_interval_s=0.0,
        decision_journal=journal,
    )

    last_decision_dt = datetime(2025, 1, 1, 12, 0, tzinfo=timezone.utc)
    lifecycle_snapshot = {
        "timestamp": last_decision_dt.isoformat(),
        "symbol": "BTCUSDT",
        "environment": "test",
        "portfolio": "alpha",
        "risk_profile": "balanced",
        "controller": {
            "auto_trade": {
                "user_confirmed": False,
                "active": False,
                "trusted_auto_confirm": False,
                "started": False,
            },
            "schedule_last_alert": None,
            "history": [],
        },
        "guardrails": {
            "last_triggers": [],
            "last_reasons": [],
            "summary": {},
        },
        "risk_decisions": {
            "summary": {},
            "last_decision": {
                "decision_id": "DEC-LAST",
                "timestamp": last_decision_dt,
                "decision_reason": "guardrail-block",
                "decision_mode": "auto",
                "approved": False,
            },
        },
        "cooldown": {},
        "strategy": {},
        "metrics": {},
        "schedule": {},
    }

    def fake_lifecycle(
        _self: AutoTrader,
        *,
        bucket_s: float = 3600.0,
        tz: tzinfo | None = timezone.utc,
    ) -> Mapping[str, Any]:
        assert bucket_s == 3600.0
        assert tz is None or isinstance(tz, tzinfo)
        return lifecycle_snapshot

    trader.build_lifecycle_snapshot = MethodType(fake_lifecycle, trader)

    def fake_guardrail_trace(
        _self: AutoTrader,
        _decision_id: Any,
        **_kwargs: Any,
    ) -> Sequence[Mapping[str, Any]]:
        return [
            {
                "timestamp": pd.Timestamp(datetime(2025, 1, 1, 11, 0, tzinfo=timezone.utc)),
                "decision_id": " DEC-GUARD ",
                "decision": SimpleNamespace(
                    decision_id="DEC-GUARD",
                    timestamp=datetime(2025, 1, 1, 11, 5, tzinfo=timezone.utc),
                    state="blocked",
                ),
                "service": "risk",
            }
        ]

    trader.get_guardrail_event_trace = MethodType(fake_guardrail_trace, trader)

    trader._model_change_log = [  # noqa: SLF001 - ustawienie stanu testowego
        {
            "event": "retrain",
            "decision": SimpleNamespace(
                decision_id="DEC-MODEL",
                timestamp=datetime(2025, 1, 1, 8, 0, tzinfo=timezone.utc),
                status="trained",
            ),
        }
    ]
    trader._retraining_cycle_log = [  # noqa: SLF001 - ustawienie stanu testowego
        {
            "cycle_id": "cycle-1",
            "decisions": [
                SimpleNamespace(
                    decision_id="DEC-CYCLE-1",
                    timestamp=datetime(2025, 1, 1, 7, 30, tzinfo=timezone.utc),
                    status="completed",
                )
            ],
        },
        {
            "cycle_id": "cycle-2",
            "decision_ids": ["DEC-CYCLE-REF"],
        },
    ]

    snapshot = trader.build_auto_mode_snapshot(include_history=False)

    lookup = snapshot["decision_lookup"]
    assert lookup["DEC-1"]["decision_id"] == "DEC-1"
    assert lookup["DEC-LAST"]["timestamp"].startswith("2025-01-01T12:00:00")
    assert lookup["DEC-GUARD"]["state"] == "blocked"
    assert lookup["DEC-MODEL"]["status"] == "trained"
    assert lookup["DEC-CYCLE-REF"] == {"decision_id": "DEC-CYCLE-REF"}

    guardrail_trace = snapshot["guardrail_trace"]
    assert guardrail_trace[0]["decision_id"] == "DEC-GUARD"
    assert guardrail_trace[0]["timestamp"].startswith("2025-01-01T11:00:00")
    assert guardrail_trace[0]["decision"]["timestamp"].startswith("2025-01-01T11:05:00")

    cycles = snapshot["retraining_cycles"]
    assert cycles[0]["decisions"][0]["timestamp"].startswith("2025-01-01T07:30:00")
    assert cycles[1]["decisions"][0] == {"decision_id": "DEC-CYCLE-REF"}

    assert snapshot["guardrail_state"]["last_decision_id"] == "DEC-LAST"


def test_auto_mode_snapshot_normalizes_decision_history_entries() -> None:
    @dataclass
    class DecisionRecord:
        decision_id: str
        timestamp: datetime
        state: str

        def to_dict(self) -> dict[str, Any]:
            return {
                "decision_id": self.decision_id,
                "timestamp": self.timestamp,
                "state": self.state,
            }

    class JournalStub:
        def __init__(self, records: Sequence[Mapping[str, Any]]) -> None:
            self._records = [dict(record) for record in records]

        def record(self, event: Mapping[str, Any] | Any) -> None:  # pragma: no cover - zgodność z kontraktem
            if isinstance(event, Mapping):
                self._records.append(dict(event))
            else:
                self._records.append({"event": event})

        def export(self) -> list[Mapping[str, Any]]:
            return [dict(record) for record in self._records]

    history_records = [
        {
            "event": "decision_executed",
            "timestamp": pd.Timestamp("2025-01-01T13:00:00"),
            "decision_id": " DEC-HISTORY-1 ",
            "decision": DecisionRecord(
                decision_id="DEC-HISTORY-1",
                timestamp=datetime(2025, 1, 1, 13, 5, tzinfo=timezone.utc),
                state="executed",
            ),
            "service": "execution",
        },
        {
            "event": "schedule_blocked",
            "timestamp": datetime(2025, 1, 1, 12, 0),
            "status": "blocked",
        },
    ]

    journal = JournalStub(history_records)

    trader = AutoTrader(
        emitter=DummyEmitter(),
        gui=SimpleNamespace(),
        symbol_getter=lambda: "BTCUSDT",
        enable_auto_trade=False,
        auto_trade_interval_s=0.0,
        decision_journal=journal,
    )

    lifecycle_snapshot = {
        "timestamp": datetime(2025, 1, 1, 13, 30, tzinfo=timezone.utc).isoformat(),
        "symbol": "BTCUSDT",
        "environment": "test",
        "portfolio": "alpha",
        "risk_profile": "balanced",
        "controller": {
            "auto_trade": {
                "user_confirmed": True,
                "active": False,
                "trusted_auto_confirm": False,
                "started": False,
            }
        },
        "schedule": {},
        "strategy": {},
        "metrics": {},
        "risk_decisions": {
            "last_decision": {
                "decision_id": "DEC-HISTORY-1",
                "timestamp": datetime(2025, 1, 1, 13, 5, tzinfo=timezone.utc),
                "decision_mode": "auto",
                "decision_reason": "executed",
                "approved": True,
            }
        },
        "guardrails": {
            "last_triggers": [],
            "last_reasons": [],
            "summary": {},
        },
        "metrics_summary": {},
        "decision_summary": {},
        "controller_history": [],
        "guardrail_summary": {},
        "reasons": {},
    }

    def fake_lifecycle(
        _self: AutoTrader,
        *,
        bucket_s: float = 3600.0,
        tz: tzinfo | None = timezone.utc,
    ) -> Mapping[str, Any]:
        assert bucket_s == 3600.0
        assert tz is None or isinstance(tz, tzinfo)
        return lifecycle_snapshot

    trader.build_lifecycle_snapshot = MethodType(fake_lifecycle, trader)
    trader.get_guardrail_event_trace = MethodType(lambda *_args, **_kwargs: [], trader)
    trader._model_change_log = []  # noqa: SLF001 - ustawienie stanu testowego
    trader._retraining_cycle_log = []  # noqa: SLF001 - ustawienie stanu testowego

    snapshot = trader.build_auto_mode_snapshot(include_history=True)

    history = snapshot["decision_history"]
    assert history[0]["decision_id"] == "DEC-HISTORY-1"
    assert history[0]["timestamp"].startswith("2025-01-01T13:00:00")
    assert history[0]["decision"]["state"] == "executed"
    assert history[0]["decision"]["timestamp"].startswith("2025-01-01T13:05:00")
    assert history[1]["event"] == "schedule_blocked"
    assert "decision_id" not in history[1]
    assert history[1]["timestamp"].startswith("2025-01-01T12:00:00")

    lookup = snapshot["decision_lookup"]
    assert lookup["DEC-HISTORY-1"]["decision"]["state"] == "executed"
    assert lookup["DEC-HISTORY-1"]["timestamp"].startswith("2025-01-01T13:00:00")


def test_auto_mode_snapshot_normalizes_model_events() -> None:
    class JournalStub:
        def __init__(self, records: Sequence[Mapping[str, Any]]) -> None:
            self._records = [dict(record) for record in records]

        def record(self, event: Mapping[str, Any] | Any) -> None:  # pragma: no cover - zgodność z kontraktem
            if isinstance(event, Mapping):
                self._records.append(dict(event))
            else:
                self._records.append({"event": event})

        def export(self) -> list[Mapping[str, Any]]:
            return [dict(record) for record in self._records]

    journal = JournalStub(
        [
            {
                "event": "decision_executed",
                "timestamp": datetime(2025, 1, 1, 8, 30, tzinfo=timezone.utc),
                "decision_id": "DEC-MODEL-REF",
            }
        ]
    )

    trader = AutoTrader(
        emitter=DummyEmitter(),
        gui=SimpleNamespace(),
        symbol_getter=lambda: "BTCUSDT",
        enable_auto_trade=False,
        auto_trade_interval_s=0.0,
        decision_journal=journal,
    )

    lifecycle_snapshot = {
        "timestamp": datetime(2025, 1, 1, 9, 30, tzinfo=timezone.utc).isoformat(),
        "symbol": "BTCUSDT",
        "environment": "test",
        "portfolio": "alpha",
        "risk_profile": "balanced",
        "controller": {"auto_trade": {"user_confirmed": True, "active": False}},
        "schedule": {},
        "strategy": {},
        "metrics": {},
        "risk_decisions": {
            "last_decision": {
                "decision_id": "DEC-MODEL-1",
                "timestamp": datetime(2025, 1, 1, 9, 15, tzinfo=timezone.utc),
                "decision_mode": "auto",
                "decision_reason": "executed",
                "approved": True,
            }
        },
        "guardrails": {"last_triggers": [], "last_reasons": [], "summary": {}},
        "metrics_summary": {},
        "decision_summary": {},
        "controller_history": [],
        "guardrail_summary": {},
        "reasons": {},
    }

    def fake_lifecycle(
        _self: AutoTrader,
        *,
        bucket_s: float = 3600.0,
        tz: tzinfo | None = timezone.utc,
    ) -> Mapping[str, Any]:
        assert bucket_s == 3600.0
        assert tz is None or isinstance(tz, tzinfo)
        return lifecycle_snapshot

    trader.build_lifecycle_snapshot = MethodType(fake_lifecycle, trader)
    trader.get_guardrail_event_trace = MethodType(lambda *_args, **_kwargs: [], trader)

    trader._model_change_log = [  # noqa: SLF001 - ustawienie stanu testowego
        {
            "event": "trained",
            "timestamp": datetime(2025, 1, 1, 9, 0, tzinfo=timezone.utc),
            "decision": SimpleNamespace(
                decision_id="DEC-MODEL-1",
                timestamp=datetime(2025, 1, 1, 9, 5, tzinfo=timezone.utc),
                state="completed",
            ),
        },
        {
            "event": "promoted",
            "timestamp": pd.Timestamp(datetime(2025, 1, 1, 8, 45, tzinfo=timezone.utc)),
            "decision_id": " DEC-MODEL-REF ",
        },
        "unexpected-entry",
    ]

    trader._retraining_cycle_log = []  # noqa: SLF001 - ustawienie stanu testowego

    snapshot = trader.build_auto_mode_snapshot(include_history=False)

    events = snapshot["model_events"]
    assert events[0]["event"] == "trained"
    assert events[0]["timestamp"].startswith("2025-01-01T09:00:00")
    assert events[0]["decision"]["decision_id"] == "DEC-MODEL-1"
    assert events[0]["decision"]["timestamp"].startswith("2025-01-01T09:05:00")

    second_event = events[1]
    assert second_event["event"] == "promoted"
    assert second_event["decision_id"] == "DEC-MODEL-REF"
    assert second_event["decision"]["decision_id"] == "DEC-MODEL-REF"
    assert second_event["decision"]["timestamp"].startswith("2025-01-01T08:30:00")

    assert events[2] == {"event": "unexpected-entry"}

    lookup = snapshot["decision_lookup"]
    assert lookup["DEC-MODEL-1"]["state"] == "completed"
    assert lookup["DEC-MODEL-REF"]["decision_id"] == "DEC-MODEL-REF"
    assert lookup["DEC-MODEL-REF"]["timestamp"].startswith("2025-01-01T08:30:00")


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

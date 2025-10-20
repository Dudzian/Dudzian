from datetime import datetime, timezone
import json
import time
from types import SimpleNamespace

import pytest

from bot_core.decision.models import DecisionEvaluation
from bot_core.exchanges.streaming import StreamBatch
from bot_core.runtime.pipeline import (
    DecisionAwareSignalSink,
    InMemoryStrategySignalSink,
    StreamingStrategyFeed,
)
from bot_core.strategies.base import MarketSnapshot, StrategySignal


class _DummyHistoryFeed:
    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []

    def load_history(self, strategy_name: str, bars: int):  # pragma: no cover - delegacja
        self.calls.append((strategy_name, bars))
        return ()

    def fetch_latest(self, strategy_name: str):  # pragma: no cover - kompatybilność protokołu
        return ()


class _DummyJournal:
    def __init__(self) -> None:
        self.events = []

    def record(self, event) -> None:  # pragma: no cover - prosty bufor
        self.events.append(event)

    def export(self):  # pragma: no cover - zgodność interfejsu
        return tuple(event.as_dict() for event in self.events)


def test_streaming_strategy_feed_converts_events() -> None:
    history = _DummyHistoryFeed()

    def _factory():
        payload = {
            "symbol": "BTC/USDT",
            "last_price": 101.0,
            "open_price": 99.0,
            "high_24h": 105.0,
            "low_24h": 95.0,
            "volume_24h_base": 12.0,
            "best_bid": 100.9,
            "best_ask": 101.1,
            "timestamp": time.time(),
        }
        batch = StreamBatch(channel="ticker", events=(payload,), received_at=time.monotonic())
        yield batch

    feed = StreamingStrategyFeed(
        history_feed=history,
        stream_factory=lambda: _factory(),
        symbols_map={"daily": ("BTC/USDT",)},
        buffer_size=8,
    )

    for batch in _factory():
        feed.ingest_batch(batch)

    snapshots = feed.fetch_latest("daily")
    assert snapshots and isinstance(snapshots[0], MarketSnapshot)
    assert snapshots[0].symbol == "BTC/USDT"
    assert snapshots[0].close == pytest.approx(101.0)
    assert "best_bid" in snapshots[0].indicators


def test_decision_aware_sink_filters_signals() -> None:
    base_sink = InMemoryStrategySignalSink()
    accepted: list[StrategySignal] = []

    class _StubOrchestrator:
        def __init__(self) -> None:
            self.invocations: list = []

        def evaluate_candidate(self, candidate, _snapshot):
            self.invocations.append(candidate)
            accepted_flag = candidate.expected_probability >= 0.6
            thresholds = {"max_cost_bps": 12.0, "min_net_edge_bps": 3.0}
            selection = SimpleNamespace(
                to_mapping=lambda: {
                    "selected": "gbm_v1",
                    "candidates": [],
                }
            )
            return SimpleNamespace(
                candidate=candidate,
                accepted=accepted_flag,
                cost_bps=None,
                net_edge_bps=5.0,
                reasons=(),
                risk_flags=(),
                stress_failures=(),
                thresholds_snapshot=thresholds,
                model_selection=selection,
                model_name="gbm_v1",
                model_success_probability=0.7,
                model_expected_return_bps=8.0,
            )

    orchestrator = _StubOrchestrator()
    journal = _DummyJournal()
    sink = DecisionAwareSignalSink(
        base_sink=base_sink,
        orchestrator=orchestrator,
        risk_engine=SimpleNamespace(snapshot_state=lambda _: {}),
        default_notional=1_000.0,
        environment="paper",
        exchange="binance_spot",
        min_probability=0.55,
        portfolio="paper-01",
        journal=journal,
    )

    accepted_signal = StrategySignal(symbol="BTC/USDT", side="BUY", confidence=0.8, metadata={})
    rejected_signal = StrategySignal(symbol="BTC/USDT", side="BUY", confidence=0.4, metadata={})
    sink.submit(
        strategy_name="daily",
        schedule_name="schedule",
        risk_profile="balanced",
        timestamp=datetime.now(timezone.utc),
        signals=(accepted_signal, rejected_signal),
    )

    records = sink.export()
    assert len(records) == 1
    _, exported_signals = records[0]
    assert exported_signals == (accepted_signal,)
    assert len(orchestrator.invocations) == 1
    assert orchestrator.invocations[0].symbol == "BTC/USDT"
    assert [event.status for event in journal.events] == ["accepted", "filtered"]
    assert all(event.portfolio == "paper-01" for event in journal.events)
    accepted_event = journal.events[0]
    thresholds_payload = json.loads(accepted_event.metadata["decision_thresholds"])
    assert thresholds_payload["max_cost_bps"] == pytest.approx(12.0)
    selection_payload = json.loads(accepted_event.metadata["model_selection"])
    assert selection_payload["selected"] == "gbm_v1"
    assert accepted_event.metadata["model_name"] == "gbm_v1"
    assert accepted_event.metadata["model_success_probability"] == "0.700000"
    assert journal.events[1].metadata.get("decision_reason") == "probability_below_threshold"
    assert journal.events[1].metadata.get("decision_status") == "filtered"


def test_decision_aware_sink_handles_missing_metadata() -> None:
    base_sink = InMemoryStrategySignalSink()

    class _StubOrchestrator:
        def evaluate_candidate(self, candidate, _snapshot):
            return SimpleNamespace(
                candidate=candidate,
                accepted=True,
                cost_bps=None,
                net_edge_bps=5.0,
                reasons=(),
                risk_flags=(),
                stress_failures=(),
            )

    sink = DecisionAwareSignalSink(
        base_sink=base_sink,
        orchestrator=_StubOrchestrator(),
        risk_engine=SimpleNamespace(snapshot_state=lambda _: {}),
        default_notional=1_000.0,
        environment="paper",
        exchange="binance_spot",
        min_probability=0.55,
        portfolio=None,
        journal=_DummyJournal(),
    )

    signal = StrategySignal(symbol="ETH/USDT", side="BUY", confidence=0.7, metadata=None)

    sink.submit(
        strategy_name="daily",
        schedule_name="schedule",
        risk_profile="balanced",
        timestamp=datetime.now(timezone.utc),
        signals=(signal,),
    )

    records = sink.export()
    assert records and records[0][1] == (signal,)


def test_decision_aware_sink_respects_min_probability_threshold() -> None:
    base_sink = InMemoryStrategySignalSink()

    class _StubOrchestrator:
        def __init__(self) -> None:
            self.invocations: list = []

        def evaluate_candidate(self, candidate, _snapshot):  # pragma: no cover - nie powinien zostać wywołany
            self.invocations.append(candidate)
            return SimpleNamespace(accepted=False, reasons=(), risk_flags=(), stress_failures=(), cost_bps=None, net_edge_bps=None)

    orchestrator = _StubOrchestrator()
    journal = _DummyJournal()
    sink = DecisionAwareSignalSink(
        base_sink=base_sink,
        orchestrator=orchestrator,
        risk_engine=SimpleNamespace(snapshot_state=lambda _: {}),
        default_notional=1_000.0,
        environment="paper",
        exchange="binance_spot",
        min_probability=0.55,
        portfolio=None,
        journal=journal,
    )

    low_probability_signal = StrategySignal(
        symbol="BTC/USDT",
        side="BUY",
        confidence=0.9,
        metadata={"expected_probability": 0.2},
    )

    sink.submit(
        strategy_name="daily",
        schedule_name="schedule",
        risk_profile="balanced",
        timestamp=datetime.now(timezone.utc),
        signals=(low_probability_signal,),
    )

    assert orchestrator.invocations == []
    assert sink.export() == ()
    assert [event.status for event in journal.events] == ["filtered"]
    filtered_event = journal.events[0]
    assert filtered_event.metadata.get("decision_reason") == "probability_below_threshold"
    assert filtered_event.metadata.get("expected_probability") == "0.200000"
    assert filtered_event.metadata.get("min_probability") == "0.550000"


def test_decision_aware_sink_limits_evaluation_history() -> None:
    base_sink = InMemoryStrategySignalSink()

    class _StubOrchestrator:
        def __init__(self) -> None:
            self.counter = 0

        def evaluate_candidate(self, candidate, _snapshot):
            self.counter += 1
            return SimpleNamespace(
                candidate=candidate,
                accepted=True,
                cost_bps=None,
                net_edge_bps=4.0,
                reasons=(),
                risk_flags=(),
                stress_failures=(),
            )

    orchestrator = _StubOrchestrator()
    sink = DecisionAwareSignalSink(
        base_sink=base_sink,
        orchestrator=orchestrator,
        risk_engine=SimpleNamespace(snapshot_state=lambda _: {}),
        default_notional=1_000.0,
        environment="paper",
        exchange="binance_spot",
        min_probability=0.1,
        portfolio="paper-01",
        journal=_DummyJournal(),
        evaluation_history_limit=2,
    )

    for idx in range(3):
        signal = StrategySignal(
            symbol="BTC/USDT",
            side="BUY",
            confidence=0.9,
            metadata={"expected_return_bps": 10.0 + idx},
        )
        sink.submit(
            strategy_name="daily",
            schedule_name="schedule",
            risk_profile="balanced",
            timestamp=datetime.now(timezone.utc),
            signals=(signal,),
        )

    evaluations = sink.evaluations()
    assert len(evaluations) == 2
    assert orchestrator.counter == 3
    assert evaluations[0].candidate.expected_return_bps == pytest.approx(11.0)
    assert evaluations[1].candidate.expected_return_bps == pytest.approx(12.0)


def test_decision_aware_sink_handles_missing_confidence_for_expected_return() -> None:
    base_sink = InMemoryStrategySignalSink()

    class _StubOrchestrator:
        def __init__(self) -> None:
            self.invocations: list = []

        def evaluate_candidate(self, candidate, _snapshot):
            self.invocations.append(candidate)
            return SimpleNamespace(
                candidate=candidate,
                accepted=True,
                cost_bps=None,
                net_edge_bps=5.0,
                reasons=(),
                risk_flags=(),
                stress_failures=(),
            )

    orchestrator = _StubOrchestrator()
    sink = DecisionAwareSignalSink(
        base_sink=base_sink,
        orchestrator=orchestrator,
        risk_engine=SimpleNamespace(snapshot_state=lambda _: {}),
        default_notional=1_000.0,
        environment="paper",
        exchange="binance_spot",
        min_probability=0.55,
        portfolio=None,
        journal=_DummyJournal(),
    )

    signal = StrategySignal(
        symbol="BTC/USDT",
        side="BUY",
        confidence=None,
        metadata={"expected_probability": 0.7},
    )

    sink.submit(
        strategy_name="daily",
        schedule_name="schedule",
        risk_profile="balanced",
        timestamp=datetime.now(timezone.utc),
        signals=(signal,),
    )

    assert orchestrator.invocations, "Orchestrator powinien otrzymać kandydata"
    candidate = orchestrator.invocations[0]
    assert candidate.expected_return_bps == pytest.approx(5.0)
    records = sink.export()
    assert records and records[0][1] == (signal,)


def test_decision_aware_sink_exposes_history_and_summary() -> None:
    base_sink = InMemoryStrategySignalSink()

    class _StubOrchestrator:
        def __init__(self, templates):
            self._templates = list(templates)
            self.invocations: list = []

        def evaluate_candidate(self, candidate, _snapshot):
            if not self._templates:
                raise AssertionError("Brak szablonów ewaluacji")
            template = self._templates.pop(0)
            self.invocations.append(candidate)
            evaluation = DecisionEvaluation(
                candidate=candidate,
                accepted=template["accepted"],
                cost_bps=template.get("cost_bps"),
                net_edge_bps=template.get("net_edge_bps"),
                reasons=tuple(template.get("reasons", ())),
                risk_flags=tuple(template.get("risk_flags", ())),
                stress_failures=tuple(template.get("stress_failures", ())),
                model_expected_return_bps=template.get("model_expected_return_bps"),
                model_success_probability=template.get("model_success_probability"),
                model_name=template.get("model_name"),
                model_selection=template.get("model_selection"),
                thresholds_snapshot=template.get("thresholds_snapshot"),
            )
            return evaluation

    orchestrator = _StubOrchestrator(
        [
            {
                "accepted": True,
                "cost_bps": 1.2,
                "net_edge_bps": 6.5,
                "model_expected_return_bps": 7.8,
                "model_success_probability": 0.68,
                "model_name": "gbm_v1",
                "model_selection": SimpleNamespace(
                    to_mapping=lambda: {"selected": "gbm_v1", "candidates": []}
                ),
                "thresholds_snapshot": {"min_probability": 0.55},
            },
            {
                "accepted": True,
                "cost_bps": 1.4,
                "net_edge_bps": 8.1,
                "model_expected_return_bps": 9.0,
                "model_success_probability": 0.72,
                "model_name": "gbm_v2",
                "model_selection": SimpleNamespace(
                    to_mapping=lambda: {"selected": "gbm_v2"}
                ),
                "thresholds_snapshot": {"min_probability": 0.6, "max_cost_bps": 15.0},
            },
            {
                "accepted": False,
                "cost_bps": 2.5,
                "net_edge_bps": 1.0,
                "reasons": ("too_costly",),
                "model_expected_return_bps": 3.5,
                "model_success_probability": 0.42,
                "model_name": "gbm_v3",
                "model_selection": SimpleNamespace(
                    to_mapping=lambda: {"selected": "gbm_v3", "reason": "fallback"}
                ),
                "thresholds_snapshot": {"min_probability": 0.7, "max_cost_bps": 12.0},
            },
        ]
    )

    journal = _DummyJournal()
    sink = DecisionAwareSignalSink(
        base_sink=base_sink,
        orchestrator=orchestrator,
        risk_engine=SimpleNamespace(snapshot_state=lambda _: {}),
        default_notional=1_000.0,
        environment="paper",
        exchange="binance_spot",
        min_probability=0.1,
        portfolio="paper-01",
        journal=journal,
        evaluation_history_limit=4,
    )

    signals = (
        StrategySignal(
            symbol="BTC/USDT",
            side="BUY",
            confidence=0.7,
            metadata={
                "expected_return_bps": 12.0,
                "expected_probability": 0.65,
                "generated_at": "2024-04-01T00:00:00+00:00",
            },
        ),
        StrategySignal(
            symbol="ETH/USDT",
            side="BUY",
            confidence=0.82,
            metadata={
                "expected_return_bps": 15.0,
                "expected_probability": 0.72,
                "generated_at": "2024-04-02T00:00:00+00:00",
            },
        ),
        StrategySignal(
            symbol="ETH/USDT",
            side="SELL",
            confidence=0.45,
            metadata={
                "expected_return_bps": 3.0,
                "expected_probability": 0.4,
                "generated_at": "2024-05-01T00:00:00+00:00",
                "cost_bps": 2.5,
            },
        ),
    )

    for signal in signals:
        sink.submit(
            strategy_name="daily",
            schedule_name="schedule",
            risk_profile="balanced",
            timestamp=datetime.now(timezone.utc),
            signals=(signal,),
        )

    history_with_candidates = sink.evaluation_history(include_candidates=True)
    assert len(history_with_candidates) == 3
    assert history_with_candidates[-1]["candidate"]["metadata"]["generated_at"] == (
        "2024-05-01T00:00:00+00:00"
    )
    assert history_with_candidates[0]["model_selection"]["selected"] == "gbm_v1"
    assert history_with_candidates[-1]["thresholds"]["min_probability"] == pytest.approx(0.7)

    history_without_candidates = sink.evaluation_history(limit=2)
    assert len(history_without_candidates) == 2
    assert all("candidate" not in payload for payload in history_without_candidates)
    assert history_without_candidates[0]["model_name"] == "gbm_v2"
    assert history_without_candidates[1]["model_name"] == "gbm_v3"

    summary = sink.evaluation_summary()
    assert summary["total"] == 3
    assert summary["accepted"] == 2
    assert summary["rejected"] == 1
    assert summary["history_limit"] == 4
    assert summary["history_window"] == 3
    assert summary["latest_model"] == "gbm_v3"
    assert summary["latest_thresholds"]["min_probability"] == pytest.approx(0.7)
    assert summary["latest_candidate"]["symbol"] == "ETH/USDT"
    assert summary["latest_generated_at"] == "2024-05-01T00:00:00+00:00"
    assert summary["rejection_reasons"] == {"too_costly": 1}
    assert summary["avg_expected_probability"] == pytest.approx((0.65 + 0.72 + 0.4) / 3)
    assert summary["avg_cost_bps"] == pytest.approx((1.2 + 1.4 + 2.5) / 3)
    assert summary["avg_net_edge_bps"] == pytest.approx((6.5 + 8.1 + 1.0) / 3)

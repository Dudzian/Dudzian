from datetime import datetime, timezone
import time
from types import SimpleNamespace

import pytest

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
            return SimpleNamespace(
                candidate=candidate,
                accepted=accepted_flag,
                cost_bps=None,
                net_edge_bps=5.0,
                reasons=(),
                risk_flags=(),
                stress_failures=(),
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

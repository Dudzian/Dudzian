from datetime import datetime, timezone
import json
import asyncio
import time
from types import SimpleNamespace
from contextlib import suppress
from typing import Sequence

import pytest

import bot_core.runtime.pipeline as pipeline_module
from bot_core.config.models import EnvironmentConfig, EnvironmentStreamConfig
from bot_core.decision.models import DecisionEvaluation
from bot_core.exchanges.base import Environment
from bot_core.exchanges.streaming import StreamBatch
from bot_core.observability.metrics import MetricsRegistry
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


class _AsyncBatchStream:
    def __init__(self, batches):
        self._batches = list(batches)
        self._index = 0
        self.aclose_calls = 0
        self.next_calls = 0

    def __aiter__(self):
        self._index = 0
        return self

    async def __anext__(self):
        if self._index >= len(self._batches):
            raise StopAsyncIteration
        batch = self._batches[self._index]
        self._index += 1
        self.next_calls += 1
        return batch

    async def aclose(self) -> None:
        self.aclose_calls += 1


class _StubAsyncStreamFeed:
    def __init__(self) -> None:
        self.start_async_calls = 0
        self.start_calls = 0
        self.stop_async_calls = 0
        self.stop_calls = 0
        self._task: asyncio.Task[None] | None = None

    def start(self) -> None:  # pragma: no cover - wykorzystywane w ścieżce synchronicznej
        self.start_calls += 1

    def start_async(self, *, loop: asyncio.AbstractEventLoop | None = None) -> asyncio.Task[None]:
        self.start_async_calls += 1
        coro = asyncio.sleep(3600)
        task = asyncio.create_task(coro)
        self._task = task
        return task

    def stop(self) -> None:
        self.stop_calls += 1
        task = self._task
        if task is not None:
            task.cancel()

    async def stop_async(self) -> None:
        self.stop_async_calls += 1
        task = self._task
        if task is None:
            return
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task


def test_build_streaming_feed_uses_adapter_metrics_registry(monkeypatch: pytest.MonkeyPatch) -> None:
    registry = MetricsRegistry()

    class _StubAdapter:
        def __init__(self, metrics: MetricsRegistry) -> None:
            self._metrics = metrics

    bootstrap = SimpleNamespace(adapter=_StubAdapter(registry))
    stream_settings = {
        "base_url": "http://127.0.0.1:9999",
        "public_path": "/stream/demo/public",
        "public_channels": ["ticker"],
    }
    environment = EnvironmentConfig(
        name="demo",
        exchange="demo",
        environment=Environment.PAPER,
        keychain_key="key",
        data_cache_path="/tmp",
        risk_profile="balanced",
        alert_channels=(),
        adapter_settings={"stream": stream_settings},
        stream=EnvironmentStreamConfig(),
    )
    captured: dict[str, MetricsRegistry | None] = {}

    class _StubStream:
        def __init__(self, *_, **kwargs) -> None:
            captured["metrics"] = kwargs.get("metrics_registry")
            captured["start_called"] = False

        def start(self):  # noqa: D401 - fluent API
            captured["start_called"] = True
            return self

        def __iter__(self):  # pragma: no cover - nie używane
            return iter(())

        def close(self) -> None:  # pragma: no cover - nie używane
            pass

    monkeypatch.setattr(pipeline_module, "LocalLongPollStream", _StubStream)

    feed = pipeline_module._build_streaming_feed(
        bootstrap=bootstrap,
        environment=environment,
        base_feed=_DummyHistoryFeed(),
        symbols_map={"daily": ("BTC/USDT",)},
    )

    assert feed is not None
    feed._stream_factory()
    assert captured["metrics"] is registry
    assert captured.get("start_called") is True


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


def test_streaming_strategy_feed_start_async_consumes() -> None:
    history = _DummyHistoryFeed()
    batches = [
        StreamBatch(
            channel="ticker",
            events=(
                {
                    "symbol": "BTC/USDT",
                    "last_price": 102.5,
                    "open_price": 101.0,
                    "high_24h": 103.0,
                    "low_24h": 99.5,
                    "volume_24h_base": 15.0,
                    "timestamp": time.time(),
                },
            ),
            received_at=time.monotonic(),
        )
    ]

    streams: list[_AsyncBatchStream] = []

    def _factory() -> _AsyncBatchStream:
        stream = _AsyncBatchStream(batches)
        streams.append(stream)
        return stream

    feed = StreamingStrategyFeed(
        history_feed=history,
        stream_factory=_factory,
        symbols_map={"daily": ("BTC/USDT",)},
        buffer_size=4,
        restart_delay=0.01,
    )

    async def _run() -> None:
        task = feed.start_async()
        assert isinstance(task, asyncio.Task)

        async def _wait_for_data() -> Sequence[MarketSnapshot]:
            for _ in range(20):
                snapshots = feed.fetch_latest("daily")
                if snapshots:
                    return snapshots
                await asyncio.sleep(0.01)
            return ()

        snapshots = await _wait_for_data()
        await feed.stop_async()

        assert snapshots and snapshots[0].symbol == "BTC/USDT"
        assert streams and streams[0].aclose_calls >= 1

    asyncio.run(_run())


def test_streaming_strategy_feed_start_async_idempotent() -> None:
    history = _DummyHistoryFeed()

    def _factory() -> _AsyncBatchStream:
        return _AsyncBatchStream(
            [
                StreamBatch(
                    channel="ticker",
                    events=(
                        {
                            "symbol": "BTC/USDT",
                            "last_price": 100.0,
                            "open_price": 100.0,
                            "high_24h": 101.0,
                            "low_24h": 99.0,
                            "volume_24h_base": 10.0,
                            "timestamp": time.time(),
                        },
                    ),
                    received_at=time.monotonic(),
                )
            ]
        )

    feed = StreamingStrategyFeed(
        history_feed=history,
        stream_factory=_factory,
        symbols_map={"daily": ("BTC/USDT",)},
        buffer_size=2,
        restart_delay=0.01,
    )

    async def _run() -> None:
        task_one = feed.start_async()
        task_two = feed.start_async()
        assert task_one is task_two

        await asyncio.sleep(0.02)
        await feed.stop_async()
        assert feed._async_task is None

    asyncio.run(_run())


def test_multi_strategy_runtime_start_stream_async_and_shutdown() -> None:
    stream = _StubAsyncStreamFeed()

    async def _run() -> None:
        runtime = pipeline_module.MultiStrategyRuntime(
            bootstrap=SimpleNamespace(),
            scheduler=SimpleNamespace(),
            data_feed=SimpleNamespace(),
            signal_sink=SimpleNamespace(),
            strategies={},
            schedules=(),
            capital_policy=SimpleNamespace(),
            stream_feed=stream,
        )

        task = runtime.start_stream_async()

        assert isinstance(task, asyncio.Task)
        assert runtime.stream_feed_task is task
        assert stream.start_async_calls == 1

        await runtime.shutdown_async()

        assert stream.stop_async_calls == 1
        assert runtime.stream_feed_task is None

    asyncio.run(_run())


def test_multi_strategy_runtime_shutdown_cancels_task_and_stops_scheduler() -> None:
    stream = _StubAsyncStreamFeed()

    class _StubScheduler:
        def __init__(self) -> None:
            self.stop_calls = 0

        def stop(self) -> None:
            self.stop_calls += 1

    scheduler = _StubScheduler()

    runtime = pipeline_module.MultiStrategyRuntime(
        bootstrap=SimpleNamespace(),
        scheduler=SimpleNamespace(),
        data_feed=SimpleNamespace(),
        signal_sink=SimpleNamespace(),
        strategies={},
        schedules=(),
        capital_policy=SimpleNamespace(),
        stream_feed=stream,
        optimization_scheduler=scheduler,  # type: ignore[arg-type]
    )

    class _StubTask:
        def __init__(self) -> None:
            self.cancelled = False

        def cancel(self) -> None:
            self.cancelled = True

        def done(self) -> bool:
            return False

    task = _StubTask()
    runtime.stream_feed_task = task  # type: ignore[assignment]

    runtime.shutdown()

    assert task.cancelled is True
    assert stream.stop_calls == 1
    assert scheduler.stop_calls == 1
    assert runtime.stream_feed_task is None


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
                "thresholds_snapshot": {
                    "min_probability": 0.55,
                    "max_cost_bps": 18.0,
                    "min_net_edge_bps": 5.5,
                    "max_latency_ms": 60.0,
                    "max_trade_notional": 1_500.0,
                },
                "risk_flags": ("volatility_spike",),
                "stress_failures": ("latency_budget",),
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
                "thresholds_snapshot": {
                    "min_probability": 0.6,
                    "max_cost_bps": 15.0,
                    "min_net_edge_bps": 6.0,
                    "max_latency_ms": 55.0,
                    "max_trade_notional": 1_200.0,
                },
                "risk_flags": ("latency_surge",),
                "stress_failures": (),
            },
            {
                "accepted": False,
                "cost_bps": 2.5,
                "net_edge_bps": 1.0,
                "reasons": ("too_costly",),
                "risk_flags": ("drawdown_risk",),
                "stress_failures": ("liquidity",),
                "model_expected_return_bps": 3.5,
                "model_success_probability": 0.42,
                "model_name": "gbm_v3",
                "model_selection": SimpleNamespace(
                    to_mapping=lambda: {"selected": "gbm_v3", "reason": "fallback"}
                ),
                "thresholds_snapshot": {
                    "min_probability": 0.7,
                    "max_cost_bps": 2.0,
                    "min_net_edge_bps": 1.5,
                    "max_latency_ms": 50.0,
                    "max_trade_notional": 800.0,
                },
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
                "latency_ms": 41.0,
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
                "latency_ms": 37.5,
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
                "latency_ms": 58.0,
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
    assert summary["latest_status"] == "rejected"
    assert summary["latest_reasons"] == ["too_costly"]
    assert summary["latest_risk_flags"] == ["drawdown_risk"]
    assert summary["latest_stress_failures"] == ["liquidity"]
    assert summary["latest_model_selection"]["selected"] == "gbm_v3"
    assert summary["latest_thresholds"]["min_probability"] == pytest.approx(0.7)
    assert summary["latest_candidate"]["symbol"] == "ETH/USDT"
    assert summary["latest_generated_at"] == "2024-05-01T00:00:00+00:00"
    assert summary["history_start_generated_at"] == "2024-04-01T00:00:00+00:00"
    assert summary["history_end_generated_at"] == "2024-05-01T00:00:00+00:00"
    assert summary["history_span_seconds"] == pytest.approx(2_592_000.0)
    assert summary["full_history_start_generated_at"] == "2024-04-01T00:00:00+00:00"
    assert summary["full_history_end_generated_at"] == "2024-05-01T00:00:00+00:00"
    assert summary["full_history_span_seconds"] == pytest.approx(2_592_000.0)
    assert summary["latest_thresholds"]["min_probability"] == pytest.approx(0.7)
    assert summary["latest_candidate"]["symbol"] == "ETH/USDT"
    assert summary["latest_generated_at"] == "2024-05-01T00:00:00+00:00"
    assert summary["rejection_reasons"] == {"too_costly": 1}
    assert summary["avg_expected_probability"] == pytest.approx((0.65 + 0.72 + 0.4) / 3)
    assert summary["avg_cost_bps"] == pytest.approx((1.2 + 1.4 + 2.5) / 3)
    assert summary["avg_net_edge_bps"] == pytest.approx((6.5 + 8.1 + 1.0) / 3)
    assert summary["sum_cost_bps"] == pytest.approx(5.1)
    assert summary["sum_net_edge_bps"] == pytest.approx(15.6)
    assert summary["median_net_edge_bps"] == pytest.approx(6.5)
    assert summary["p90_net_edge_bps"] == pytest.approx(7.78, rel=1e-3)
    assert summary["p95_net_edge_bps"] == pytest.approx(7.94, rel=1e-3)
    assert summary["min_net_edge_bps"] == pytest.approx(1.0)
    assert summary["max_net_edge_bps"] == pytest.approx(8.1)
    assert summary["median_cost_bps"] == pytest.approx(1.4)
    assert summary["p90_cost_bps"] == pytest.approx(2.28, rel=1e-3)
    assert summary["min_cost_bps"] == pytest.approx(1.2)
    assert summary["max_cost_bps"] == pytest.approx(2.5)
    assert summary["avg_latency_ms"] == pytest.approx((41.0 + 37.5 + 58.0) / 3)
    assert summary["sum_latency_ms"] == pytest.approx(136.5)
    assert summary["median_latency_ms"] == pytest.approx(41.0)
    assert summary["p90_latency_ms"] == pytest.approx(54.6, rel=1e-3)
    assert summary["p95_latency_ms"] == pytest.approx(56.3, rel=1e-3)
    assert summary["min_latency_ms"] == pytest.approx(37.5)
    assert summary["max_latency_ms"] == pytest.approx(58.0)
    assert summary["median_expected_probability"] == pytest.approx(0.65)
    assert summary["median_expected_return_bps"] == pytest.approx(12.0)
    assert summary["median_model_success_probability"] == pytest.approx(0.68)
    assert summary["median_model_expected_return_bps"] == pytest.approx(7.8)
    assert summary["avg_expected_value_bps"] == pytest.approx(6.6)
    assert summary["sum_expected_return_bps"] == pytest.approx(30.0)
    assert summary["sum_expected_value_bps"] == pytest.approx(19.8)
    assert summary["median_expected_value_bps"] == pytest.approx(7.8)
    assert summary["min_expected_value_bps"] == pytest.approx(1.2)
    assert summary["max_expected_value_bps"] == pytest.approx(10.8)
    assert summary["avg_expected_value_minus_cost_bps"] == pytest.approx(4.9)
    assert summary["sum_expected_value_minus_cost_bps"] == pytest.approx(14.7)
    assert summary["probability_threshold_margin_count"] == 3
    assert summary["avg_probability_threshold_margin"] == pytest.approx(
        (0.65 - 0.55 + 0.72 - 0.6 + 0.4 - 0.7) / 3
    )
    assert summary["probability_threshold_breaches"] == 1
    assert summary["probability_threshold_breach_rate"] == pytest.approx(1 / 3)
    assert summary["accepted_probability_threshold_margin_count"] == 2
    assert summary["accepted_avg_probability_threshold_margin"] == pytest.approx(0.11)
    assert summary["accepted_probability_threshold_breaches"] == 0
    assert summary["accepted_min_probability_threshold_margin"] == pytest.approx(0.1)
    assert summary["accepted_max_probability_threshold_margin"] == pytest.approx(0.12)
    assert summary["accepted_median_probability_threshold_margin"] == pytest.approx(0.11)
    assert summary["accepted_p10_probability_threshold_margin"] == pytest.approx(0.102)
    assert summary["accepted_p90_probability_threshold_margin"] == pytest.approx(0.118)
    assert summary["accepted_std_probability_threshold_margin"] == pytest.approx(0.01)
    assert summary["rejected_probability_threshold_margin_count"] == 1
    assert summary["rejected_avg_probability_threshold_margin"] == pytest.approx(-0.3)
    assert summary["rejected_probability_threshold_breaches"] == 1
    assert summary["rejected_min_probability_threshold_margin"] == pytest.approx(-0.3)
    assert summary["rejected_max_probability_threshold_margin"] == pytest.approx(-0.3)
    assert summary["rejected_median_probability_threshold_margin"] == pytest.approx(-0.3)
    assert summary["rejected_p10_probability_threshold_margin"] == pytest.approx(-0.3)
    assert summary["rejected_p90_probability_threshold_margin"] == pytest.approx(-0.3)
    assert summary["rejected_std_probability_threshold_margin"] == pytest.approx(0.0)
    assert summary["cost_threshold_margin_count"] == 3
    assert summary["avg_cost_threshold_margin"] == pytest.approx(
        (18.0 - 1.2 + 15.0 - 1.4 + 2.0 - 2.5) / 3
    )
    assert summary["cost_threshold_breaches"] == 1
    assert summary["cost_threshold_breach_rate"] == pytest.approx(1 / 3)
    assert summary["accepted_cost_threshold_margin_count"] == 2
    assert summary["accepted_avg_cost_threshold_margin"] == pytest.approx(15.2)
    assert summary["accepted_cost_threshold_breaches"] == 0
    assert summary["accepted_min_cost_threshold_margin"] == pytest.approx(13.6)
    assert summary["accepted_max_cost_threshold_margin"] == pytest.approx(16.8)
    assert summary["accepted_median_cost_threshold_margin"] == pytest.approx(15.2)
    assert summary["accepted_p10_cost_threshold_margin"] == pytest.approx(13.92)
    assert summary["accepted_p90_cost_threshold_margin"] == pytest.approx(16.48)
    assert summary["accepted_std_cost_threshold_margin"] == pytest.approx(1.6)
    assert summary["rejected_cost_threshold_margin_count"] == 1
    assert summary["rejected_avg_cost_threshold_margin"] == pytest.approx(-0.5)
    assert summary["rejected_cost_threshold_breaches"] == 1
    assert summary["rejected_min_cost_threshold_margin"] == pytest.approx(-0.5)
    assert summary["rejected_max_cost_threshold_margin"] == pytest.approx(-0.5)
    assert summary["rejected_median_cost_threshold_margin"] == pytest.approx(-0.5)
    assert summary["rejected_p10_cost_threshold_margin"] == pytest.approx(-0.5)
    assert summary["rejected_p90_cost_threshold_margin"] == pytest.approx(-0.5)
    assert summary["rejected_std_cost_threshold_margin"] == pytest.approx(0.0)
    assert summary["net_edge_threshold_margin_count"] == 3
    assert summary["avg_net_edge_threshold_margin"] == pytest.approx(
        (6.5 - 5.5 + 8.1 - 6.0 + 1.0 - 1.5) / 3
    )
    assert summary["net_edge_threshold_breaches"] == 1
    assert summary["accepted_net_edge_threshold_margin_count"] == 2
    assert summary["accepted_avg_net_edge_threshold_margin"] == pytest.approx(
        (1.0 + 2.1) / 2
    )
    assert summary["accepted_net_edge_threshold_breaches"] == 0
    assert summary["accepted_min_net_edge_threshold_margin"] == pytest.approx(1.0)
    assert summary["accepted_max_net_edge_threshold_margin"] == pytest.approx(2.1)
    assert summary["accepted_median_net_edge_threshold_margin"] == pytest.approx(1.55)
    assert summary["accepted_p10_net_edge_threshold_margin"] == pytest.approx(1.11)
    assert summary["accepted_p90_net_edge_threshold_margin"] == pytest.approx(1.99)
    assert summary["accepted_std_net_edge_threshold_margin"] == pytest.approx(0.55)
    assert summary["rejected_net_edge_threshold_margin_count"] == 1
    assert summary["rejected_avg_net_edge_threshold_margin"] == pytest.approx(-0.5)
    assert summary["rejected_net_edge_threshold_breaches"] == 1
    assert summary["rejected_min_net_edge_threshold_margin"] == pytest.approx(-0.5)
    assert summary["rejected_max_net_edge_threshold_margin"] == pytest.approx(-0.5)
    assert summary["rejected_median_net_edge_threshold_margin"] == pytest.approx(-0.5)
    assert summary["rejected_p10_net_edge_threshold_margin"] == pytest.approx(-0.5)
    assert summary["rejected_p90_net_edge_threshold_margin"] == pytest.approx(-0.5)
    assert summary["rejected_std_net_edge_threshold_margin"] == pytest.approx(0.0)
    assert summary["latency_threshold_margin_count"] == 3
    assert summary["avg_latency_threshold_margin"] == pytest.approx(
        (60.0 - 41.0 + 55.0 - 37.5 + 50.0 - 58.0) / 3
    )
    assert summary["latency_threshold_breaches"] == 1
    assert summary["accepted_latency_threshold_margin_count"] == 2
    assert summary["accepted_avg_latency_threshold_margin"] == pytest.approx(
        (19.0 + 17.5) / 2
    )
    assert summary["accepted_latency_threshold_breaches"] == 0
    assert summary["accepted_min_latency_threshold_margin"] == pytest.approx(17.5)
    assert summary["accepted_max_latency_threshold_margin"] == pytest.approx(19.0)
    assert summary["accepted_median_latency_threshold_margin"] == pytest.approx(18.25)
    assert summary["accepted_p10_latency_threshold_margin"] == pytest.approx(17.65)
    assert summary["accepted_p90_latency_threshold_margin"] == pytest.approx(18.85)
    assert summary["accepted_std_latency_threshold_margin"] == pytest.approx(0.75)
    assert summary["rejected_latency_threshold_margin_count"] == 1
    assert summary["rejected_avg_latency_threshold_margin"] == pytest.approx(-8.0)
    assert summary["rejected_latency_threshold_breaches"] == 1
    assert summary["rejected_min_latency_threshold_margin"] == pytest.approx(-8.0)
    assert summary["rejected_max_latency_threshold_margin"] == pytest.approx(-8.0)
    assert summary["rejected_median_latency_threshold_margin"] == pytest.approx(-8.0)
    assert summary["rejected_p10_latency_threshold_margin"] == pytest.approx(-8.0)
    assert summary["rejected_p90_latency_threshold_margin"] == pytest.approx(-8.0)
    assert summary["rejected_std_latency_threshold_margin"] == pytest.approx(0.0)
    assert summary["notional_threshold_margin_count"] == 3
    assert summary["avg_notional_threshold_margin"] == pytest.approx(
        (1_500.0 - 1_000.0 + 1_200.0 - 1_000.0 + 800.0 - 1_000.0) / 3
    )
    assert summary["notional_threshold_breaches"] == 1
    assert summary["accepted_notional_threshold_margin_count"] == 2
    assert summary["accepted_avg_notional_threshold_margin"] == pytest.approx(
        (500.0 + 200.0) / 2
    )
    assert summary["accepted_notional_threshold_breaches"] == 0
    assert summary["accepted_min_notional_threshold_margin"] == pytest.approx(200.0)
    assert summary["accepted_max_notional_threshold_margin"] == pytest.approx(500.0)
    assert summary["accepted_median_notional_threshold_margin"] == pytest.approx(350.0)
    assert summary["accepted_p10_notional_threshold_margin"] == pytest.approx(230.0)
    assert summary["accepted_p90_notional_threshold_margin"] == pytest.approx(470.0)
    assert summary["accepted_std_notional_threshold_margin"] == pytest.approx(150.0)
    assert summary["rejected_notional_threshold_margin_count"] == 1
    assert summary["rejected_avg_notional_threshold_margin"] == pytest.approx(-200.0)
    assert summary["rejected_notional_threshold_breaches"] == 1
    assert summary["rejected_min_notional_threshold_margin"] == pytest.approx(-200.0)
    assert summary["rejected_max_notional_threshold_margin"] == pytest.approx(-200.0)
    assert summary["rejected_median_notional_threshold_margin"] == pytest.approx(-200.0)
    assert summary["rejected_p10_notional_threshold_margin"] == pytest.approx(-200.0)
    assert summary["rejected_p90_notional_threshold_margin"] == pytest.approx(-200.0)
    assert summary["rejected_std_notional_threshold_margin"] == pytest.approx(0.0)
    assert summary["accepted_avg_net_edge_bps"] == pytest.approx(7.3)
    assert summary["accepted_median_net_edge_bps"] == pytest.approx(7.3)
    assert summary["accepted_p10_net_edge_bps"] == pytest.approx(6.66, rel=1e-2)
    assert summary["accepted_p90_net_edge_bps"] == pytest.approx(7.94, rel=1e-3)
    assert summary["accepted_min_net_edge_bps"] == pytest.approx(6.5)
    assert summary["accepted_max_net_edge_bps"] == pytest.approx(8.1)
    assert summary["accepted_std_net_edge_bps"] == pytest.approx(0.8)
    assert summary["accepted_sum_net_edge_bps"] == pytest.approx(14.6)
    assert summary["accepted_net_edge_bps_count"] == 2
    assert summary["rejected_avg_net_edge_bps"] == pytest.approx(1.0)
    assert summary["rejected_median_net_edge_bps"] == pytest.approx(1.0)
    assert summary["rejected_p10_net_edge_bps"] == pytest.approx(1.0)
    assert summary["rejected_p90_net_edge_bps"] == pytest.approx(1.0)
    assert summary["rejected_min_net_edge_bps"] == pytest.approx(1.0)
    assert summary["rejected_max_net_edge_bps"] == pytest.approx(1.0)
    assert summary["rejected_std_net_edge_bps"] == pytest.approx(0.0)
    assert summary["rejected_sum_net_edge_bps"] == pytest.approx(1.0)
    assert summary["rejected_net_edge_bps_count"] == 1
    assert summary["accepted_avg_cost_bps"] == pytest.approx(1.3)
    assert summary["accepted_median_cost_bps"] == pytest.approx(1.3)
    assert summary["accepted_p10_cost_bps"] == pytest.approx(1.22, rel=1e-2)
    assert summary["accepted_p90_cost_bps"] == pytest.approx(1.38, rel=1e-2)
    assert summary["accepted_min_cost_bps"] == pytest.approx(1.2)
    assert summary["accepted_max_cost_bps"] == pytest.approx(1.4)
    assert summary["accepted_std_cost_bps"] == pytest.approx(0.1)
    assert summary["accepted_sum_cost_bps"] == pytest.approx(2.6)
    assert summary["accepted_cost_bps_count"] == 2
    assert summary["rejected_avg_cost_bps"] == pytest.approx(2.5)
    assert summary["rejected_median_cost_bps"] == pytest.approx(2.5)
    assert summary["rejected_p10_cost_bps"] == pytest.approx(2.5)
    assert summary["rejected_p90_cost_bps"] == pytest.approx(2.5)
    assert summary["rejected_min_cost_bps"] == pytest.approx(2.5)
    assert summary["rejected_max_cost_bps"] == pytest.approx(2.5)
    assert summary["rejected_std_cost_bps"] == pytest.approx(0.0)
    assert summary["rejected_sum_cost_bps"] == pytest.approx(2.5)
    assert summary["rejected_cost_bps_count"] == 1
    assert summary["accepted_avg_expected_probability"] == pytest.approx(0.685)
    assert summary["accepted_median_expected_probability"] == pytest.approx(0.685)
    assert summary["accepted_std_expected_probability"] == pytest.approx(0.035)
    assert summary["accepted_expected_probability_count"] == 2
    assert summary["rejected_avg_expected_probability"] == pytest.approx(0.4)
    assert summary["rejected_median_expected_probability"] == pytest.approx(0.4)
    assert summary["rejected_std_expected_probability"] == pytest.approx(0.0)
    assert summary["rejected_expected_probability_count"] == 1
    assert summary["accepted_avg_expected_return_bps"] == pytest.approx(13.5)
    assert summary["accepted_median_expected_return_bps"] == pytest.approx(13.5)
    assert summary["accepted_p90_expected_return_bps"] == pytest.approx(14.7)
    assert summary["accepted_std_expected_return_bps"] == pytest.approx(1.5)
    assert summary["accepted_sum_expected_return_bps"] == pytest.approx(27.0)
    assert summary["accepted_expected_return_bps_count"] == 2
    assert summary["rejected_avg_expected_return_bps"] == pytest.approx(3.0)
    assert summary["rejected_median_expected_return_bps"] == pytest.approx(3.0)
    assert summary["rejected_p90_expected_return_bps"] == pytest.approx(3.0)
    assert summary["rejected_std_expected_return_bps"] == pytest.approx(0.0)
    assert summary["rejected_sum_expected_return_bps"] == pytest.approx(3.0)
    assert summary["rejected_expected_return_bps_count"] == 1
    assert summary["accepted_avg_expected_value_bps"] == pytest.approx(9.3)
    assert summary["accepted_median_expected_value_bps"] == pytest.approx(9.3)
    assert summary["accepted_p90_expected_value_bps"] == pytest.approx(10.5)
    assert summary["accepted_std_expected_value_bps"] == pytest.approx(1.5)
    assert summary["accepted_sum_expected_value_bps"] == pytest.approx(18.6)
    assert summary["accepted_expected_value_bps_count"] == 2
    assert summary["rejected_avg_expected_value_bps"] == pytest.approx(1.2)
    assert summary["rejected_median_expected_value_bps"] == pytest.approx(1.2)
    assert summary["rejected_p90_expected_value_bps"] == pytest.approx(1.2)
    assert summary["rejected_std_expected_value_bps"] == pytest.approx(0.0)
    assert summary["rejected_sum_expected_value_bps"] == pytest.approx(1.2)
    assert summary["rejected_expected_value_bps_count"] == 1
    assert summary["accepted_avg_expected_value_minus_cost_bps"] == pytest.approx(8.0)
    assert summary["accepted_median_expected_value_minus_cost_bps"] == pytest.approx(8.0)
    assert summary["accepted_p90_expected_value_minus_cost_bps"] == pytest.approx(9.12, rel=1e-2)
    assert summary["accepted_min_expected_value_minus_cost_bps"] == pytest.approx(6.6)
    assert summary["accepted_max_expected_value_minus_cost_bps"] == pytest.approx(9.4)
    assert summary["accepted_std_expected_value_minus_cost_bps"] == pytest.approx(1.4)
    assert summary["accepted_sum_expected_value_minus_cost_bps"] == pytest.approx(16.0)
    assert summary["accepted_expected_value_minus_cost_bps_count"] == 2
    assert summary["rejected_avg_expected_value_minus_cost_bps"] == pytest.approx(-1.3)
    assert summary["rejected_median_expected_value_minus_cost_bps"] == pytest.approx(-1.3)
    assert summary["rejected_p90_expected_value_minus_cost_bps"] == pytest.approx(-1.3)
    assert summary["rejected_min_expected_value_minus_cost_bps"] == pytest.approx(-1.3)
    assert summary["rejected_max_expected_value_minus_cost_bps"] == pytest.approx(-1.3)
    assert summary["rejected_std_expected_value_minus_cost_bps"] == pytest.approx(0.0)
    assert summary["rejected_sum_expected_value_minus_cost_bps"] == pytest.approx(-1.3)
    assert summary["rejected_expected_value_minus_cost_bps_count"] == 1
    assert summary["accepted_avg_notional"] == pytest.approx(1_000.0)
    assert summary["accepted_median_notional"] == pytest.approx(1_000.0)
    assert summary["accepted_p90_notional"] == pytest.approx(1_000.0)
    assert summary["accepted_min_notional"] == pytest.approx(1_000.0)
    assert summary["accepted_max_notional"] == pytest.approx(1_000.0)
    assert summary["accepted_std_notional"] == pytest.approx(0.0)
    assert summary["accepted_sum_notional"] == pytest.approx(2_000.0)
    assert summary["accepted_notional_count"] == 2
    assert summary["rejected_avg_notional"] == pytest.approx(1_000.0)
    assert summary["rejected_median_notional"] == pytest.approx(1_000.0)
    assert summary["rejected_p90_notional"] == pytest.approx(1_000.0)
    assert summary["rejected_min_notional"] == pytest.approx(1_000.0)
    assert summary["rejected_max_notional"] == pytest.approx(1_000.0)
    assert summary["rejected_std_notional"] == pytest.approx(0.0)
    assert summary["rejected_sum_notional"] == pytest.approx(1_000.0)
    assert summary["rejected_notional_count"] == 1
    assert summary["accepted_avg_latency_ms"] == pytest.approx(39.25)
    assert summary["accepted_median_latency_ms"] == pytest.approx(39.25)
    assert summary["accepted_p90_latency_ms"] == pytest.approx(40.65, rel=1e-2)
    assert summary["accepted_p95_latency_ms"] == pytest.approx(40.825, rel=1e-3)
    assert summary["accepted_min_latency_ms"] == pytest.approx(37.5)
    assert summary["accepted_max_latency_ms"] == pytest.approx(41.0)
    assert summary["accepted_std_latency_ms"] == pytest.approx(1.75)
    assert summary["accepted_sum_latency_ms"] == pytest.approx(78.5)
    assert summary["accepted_latency_ms_count"] == 2
    assert summary["rejected_avg_latency_ms"] == pytest.approx(58.0)
    assert summary["rejected_median_latency_ms"] == pytest.approx(58.0)
    assert summary["rejected_p90_latency_ms"] == pytest.approx(58.0)
    assert summary["rejected_p95_latency_ms"] == pytest.approx(58.0)
    assert summary["rejected_min_latency_ms"] == pytest.approx(58.0)
    assert summary["rejected_max_latency_ms"] == pytest.approx(58.0)
    assert summary["rejected_std_latency_ms"] == pytest.approx(0.0)
    assert summary["rejected_sum_latency_ms"] == pytest.approx(58.0)
    assert summary["rejected_latency_ms_count"] == 1
    assert summary["accepted_avg_model_success_probability"] == pytest.approx(0.7)
    assert summary["accepted_median_model_success_probability"] == pytest.approx(0.7)
    assert summary["accepted_std_model_success_probability"] == pytest.approx(0.02)
    assert summary["accepted_model_success_probability_count"] == 2
    assert summary["rejected_avg_model_success_probability"] == pytest.approx(0.42)
    assert summary["rejected_median_model_success_probability"] == pytest.approx(0.42)
    assert summary["rejected_std_model_success_probability"] == pytest.approx(0.0)
    assert summary["rejected_model_success_probability_count"] == 1
    assert summary["accepted_avg_model_expected_return_bps"] == pytest.approx(8.4)
    assert summary["accepted_median_model_expected_return_bps"] == pytest.approx(8.4)
    assert summary["accepted_p90_model_expected_return_bps"] == pytest.approx(8.88)
    assert summary["accepted_std_model_expected_return_bps"] == pytest.approx(0.6)
    assert summary["accepted_sum_model_expected_return_bps"] == pytest.approx(16.8)
    assert summary["accepted_model_expected_return_bps_count"] == 2
    assert summary["rejected_avg_model_expected_return_bps"] == pytest.approx(3.5)
    assert summary["rejected_median_model_expected_return_bps"] == pytest.approx(3.5)
    assert summary["rejected_p90_model_expected_return_bps"] == pytest.approx(3.5)
    assert summary["rejected_std_model_expected_return_bps"] == pytest.approx(0.0)
    assert summary["rejected_sum_model_expected_return_bps"] == pytest.approx(3.5)
    assert summary["rejected_model_expected_return_bps_count"] == 1
    assert summary["accepted_avg_model_expected_value_bps"] == pytest.approx(5.892)
    assert summary["accepted_median_model_expected_value_bps"] == pytest.approx(5.892)
    assert summary["accepted_p90_model_expected_value_bps"] == pytest.approx(6.3624, rel=1e-4)
    assert summary["accepted_std_model_expected_value_bps"] == pytest.approx(0.588, rel=1e-3)
    assert summary["accepted_sum_model_expected_value_bps"] == pytest.approx(11.784)
    assert summary["accepted_model_expected_value_bps_count"] == 2
    assert summary["rejected_avg_model_expected_value_bps"] == pytest.approx(1.47)
    assert summary["rejected_median_model_expected_value_bps"] == pytest.approx(1.47)
    assert summary["rejected_p90_model_expected_value_bps"] == pytest.approx(1.47)
    assert summary["rejected_std_model_expected_value_bps"] == pytest.approx(0.0)
    assert summary["rejected_sum_model_expected_value_bps"] == pytest.approx(1.47)
    assert summary["rejected_model_expected_value_bps_count"] == 1
    assert summary["accepted_avg_model_expected_value_minus_cost_bps"] == pytest.approx(
        4.592
    )
    assert summary["accepted_median_model_expected_value_minus_cost_bps"] == pytest.approx(
        4.592
    )
    assert summary["accepted_p90_model_expected_value_minus_cost_bps"] == pytest.approx(
        4.9824, rel=1e-3
    )
    assert summary["accepted_min_model_expected_value_minus_cost_bps"] == pytest.approx(
        4.104
    )
    assert summary["accepted_max_model_expected_value_minus_cost_bps"] == pytest.approx(
        5.08
    )
    assert summary["accepted_std_model_expected_value_minus_cost_bps"] == pytest.approx(
        0.488, rel=1e-3
    )
    assert summary["accepted_sum_model_expected_value_minus_cost_bps"] == pytest.approx(
        9.184
    )
    assert summary["accepted_model_expected_value_minus_cost_bps_count"] == 2
    assert summary["rejected_avg_model_expected_value_minus_cost_bps"] == pytest.approx(
        -1.03
    )
    assert summary["rejected_median_model_expected_value_minus_cost_bps"] == pytest.approx(
        -1.03
    )
    assert summary["rejected_p90_model_expected_value_minus_cost_bps"] == pytest.approx(
        -1.03
    )
    assert summary["rejected_min_model_expected_value_minus_cost_bps"] == pytest.approx(
        -1.03
    )
    assert summary["rejected_max_model_expected_value_minus_cost_bps"] == pytest.approx(
        -1.03
    )
    assert summary["rejected_std_model_expected_value_minus_cost_bps"] == pytest.approx(0.0)
    assert summary["rejected_sum_model_expected_value_minus_cost_bps"] == pytest.approx(
        -1.03
    )
    assert summary["rejected_model_expected_value_minus_cost_bps_count"] == 1
    assert summary["median_expected_value_minus_cost_bps"] == pytest.approx(6.6)
    assert summary["min_expected_value_minus_cost_bps"] == pytest.approx(-1.3)
    assert summary["max_expected_value_minus_cost_bps"] == pytest.approx(9.4)
    assert summary["avg_model_expected_value_bps"] == pytest.approx(4.418)
    assert summary["sum_model_expected_return_bps"] == pytest.approx(20.3)
    assert summary["sum_model_expected_value_bps"] == pytest.approx(13.254)
    assert summary["median_model_expected_value_bps"] == pytest.approx(5.304)
    assert summary["min_model_expected_value_bps"] == pytest.approx(1.47)
    assert summary["max_model_expected_value_bps"] == pytest.approx(6.48)
    assert summary["avg_model_expected_value_minus_cost_bps"] == pytest.approx(2.718)
    assert summary["sum_model_expected_value_minus_cost_bps"] == pytest.approx(8.154)
    assert summary["median_model_expected_value_minus_cost_bps"] == pytest.approx(4.104)
    assert summary["min_model_expected_value_minus_cost_bps"] == pytest.approx(-1.03)
    assert summary["max_model_expected_value_minus_cost_bps"] == pytest.approx(5.08)
    assert summary["risk_flag_counts"] == {
        "volatility_spike": 1,
        "latency_surge": 1,
        "drawdown_risk": 1,
    }
    assert summary["risk_flags_with_accepts"] == 2
    assert summary["risk_flag_breakdown"] == {
        "volatility_spike": {
            "total": 1,
            "accepted": 1,
            "rejected": 0,
            "acceptance_rate": pytest.approx(1.0),
        },
        "latency_surge": {
            "total": 1,
            "accepted": 1,
            "rejected": 0,
            "acceptance_rate": pytest.approx(1.0),
        },
        "drawdown_risk": {
            "total": 1,
            "accepted": 0,
            "rejected": 1,
            "acceptance_rate": pytest.approx(0.0),
        },
    }
    assert summary["unique_risk_flags"] == 3
    assert summary["stress_failure_counts"] == {
        "latency_budget": 1,
        "liquidity": 1,
    }
    assert summary["stress_failures_with_accepts"] == 1
    assert summary["stress_failure_breakdown"] == {
        "latency_budget": {
            "total": 1,
            "accepted": 1,
            "rejected": 0,
            "acceptance_rate": pytest.approx(1.0),
        },
        "liquidity": {
            "total": 1,
            "accepted": 0,
            "rejected": 1,
            "acceptance_rate": pytest.approx(0.0),
        },
    }
    assert summary["unique_stress_failures"] == 2
    assert summary["model_usage"] == {
        "gbm_v1": 1,
        "gbm_v2": 1,
        "gbm_v3": 1,
    }
    assert summary["unique_models"] == 3
    assert summary["models_with_accepts"] == 2
    model_breakdown = summary["model_breakdown"]
    gbm_v1_metrics = model_breakdown["gbm_v1"]["metrics"]
    assert gbm_v1_metrics["net_edge_bps"]["accepted_sum"] == pytest.approx(6.5)
    assert gbm_v1_metrics["expected_value_minus_cost_bps"]["accepted_sum"] == pytest.approx(6.6)
    gbm_v2_metrics = model_breakdown["gbm_v2"]["metrics"]
    assert gbm_v2_metrics["net_edge_bps"]["accepted_sum"] == pytest.approx(8.1)
    assert gbm_v2_metrics["expected_value_minus_cost_bps"]["accepted_sum"] == pytest.approx(9.4)
    gbm_v3_metrics = model_breakdown["gbm_v3"]["metrics"]
    assert gbm_v3_metrics["net_edge_bps"]["rejected_sum"] == pytest.approx(1.0)
    assert gbm_v3_metrics["expected_value_minus_cost_bps"]["rejected_sum"] == pytest.approx(-1.3)
    assert summary["latest_candidate"]["expected_value_bps"] == pytest.approx(1.2)
    assert summary["latest_expected_value_bps"] == pytest.approx(1.2)
    assert summary["latest_expected_value_minus_cost_bps"] == pytest.approx(-1.3)
    assert summary["latest_net_edge_bps"] == pytest.approx(1.0)
    assert summary["latest_cost_bps"] == pytest.approx(2.5)
    assert summary["latest_latency_ms"] == pytest.approx(58.0)
    assert summary["latest_expected_probability"] == pytest.approx(0.4)
    assert summary["latest_expected_return_bps"] == pytest.approx(3.0)
    assert summary["latest_notional"] == pytest.approx(1_000.0)
    assert summary["latest_probability_threshold_margin"] == pytest.approx(-0.3)
    assert summary["latest_cost_threshold_margin"] == pytest.approx(-0.5)
    assert summary["latest_net_edge_threshold_margin"] == pytest.approx(-0.5)
    assert summary["latest_latency_threshold_margin"] == pytest.approx(-8.0)
    assert summary["latest_notional_threshold_margin"] == pytest.approx(-200.0)
    assert summary["latest_model_expected_value_bps"] == pytest.approx(1.47)
    assert summary["latest_model_expected_value_minus_cost_bps"] == pytest.approx(-1.03)
    assert summary["latest_model_expected_return_bps"] == pytest.approx(3.5)
    assert summary["latest_model_success_probability"] == pytest.approx(0.42)
    assert summary["std_net_edge_bps"] == pytest.approx(3.040833219, rel=1e-6)
    assert summary["std_cost_bps"] == pytest.approx(0.571547607, rel=1e-6)
    assert summary["std_latency_ms"] == pytest.approx(8.953584012, rel=1e-6)
    assert summary["std_expected_probability"] == pytest.approx(0.137355985, rel=1e-6)
    assert summary["std_expected_return_bps"] == pytest.approx(5.099019514, rel=1e-6)
    assert summary["std_expected_value_bps"] == pytest.approx(4.009987531, rel=1e-6)
    assert summary["std_expected_value_minus_cost_bps"] == pytest.approx(4.530636453, rel=1e-6)
    assert summary["std_model_success_probability"] == pytest.approx(0.132999582, rel=1e-6)
    assert summary["std_model_expected_return_bps"] == pytest.approx(2.361261433, rel=1e-6)
    assert summary["std_model_expected_value_bps"] == pytest.approx(2.139123185, rel=1e-6)
    assert summary["std_model_expected_value_minus_cost_bps"] == pytest.approx(
        2.680021393,
        rel=1e-6,
    )
    assert summary["longest_acceptance_streak"] == 2
    assert summary["longest_rejection_streak"] == 1
    assert summary["current_acceptance_streak"] == 0
    assert summary["current_rejection_streak"] == 1
    assert summary["action_usage"] == {"enter": 2, "exit": 1}
    assert summary["unique_actions"] == 2
    assert summary["actions_with_accepts"] == 1
    action_breakdown = summary["action_breakdown"]
    enter_metrics = action_breakdown["enter"]["metrics"]
    assert enter_metrics["net_edge_bps"]["accepted_sum"] == pytest.approx(14.6)
    assert enter_metrics["expected_value_minus_cost_bps"]["accepted_sum"] == pytest.approx(16.0)
    exit_metrics = action_breakdown["exit"]["metrics"]
    assert exit_metrics["expected_value_minus_cost_bps"]["rejected_sum"] == pytest.approx(-1.3)
    assert summary["unique_strategies"] == 1
    assert summary["strategies_with_accepts"] == 1
    assert summary["strategy_usage"] == {"daily": 3}
    strategy_breakdown = summary["strategy_breakdown"]
    strategy_metrics = strategy_breakdown["daily"]["metrics"]
    assert strategy_metrics["net_edge_bps"]["total_sum"] == pytest.approx(15.6)
    assert strategy_metrics["expected_value_minus_cost_bps"]["total_sum"] == pytest.approx(14.7)
    assert strategy_metrics["expected_value_minus_cost_bps"]["rejected_sum"] == pytest.approx(-1.3)
    assert summary["symbol_usage"] == {"ETH/USDT": 2, "BTC/USDT": 1}
    assert summary["unique_symbols"] == 2
    assert summary["symbols_with_accepts"] == 2
    symbol_breakdown = summary["symbol_breakdown"]
    eth_metrics = symbol_breakdown["ETH/USDT"]["metrics"]
    assert eth_metrics["expected_value_minus_cost_bps"]["total_sum"] == pytest.approx(8.1)
    btc_metrics = symbol_breakdown["BTC/USDT"]["metrics"]
    assert btc_metrics["expected_value_minus_cost_bps"]["accepted_sum"] == pytest.approx(6.6)


def test_consume_stream_async_processes_batches_and_closes() -> None:
    batches = [
        StreamBatch(
            channel="ticker",
            events=(
                {
                    "symbol": "BTC/USDT",
                    "close": 101.0,
                    "timestamp": time.time(),
                },
            ),
            received_at=1.0,
        ),
        StreamBatch(channel="ticker", events=(), heartbeat=True, received_at=2.0),
    ]
    stream = _AsyncBatchStream(batches)
    handled: list[StreamBatch] = []
    heartbeats: list[float] = []

    async def _handle(batch: StreamBatch) -> None:
        handled.append(batch)

    async def _heartbeat(timestamp: float) -> None:
        heartbeats.append(timestamp)

    async def _run() -> None:
        await pipeline_module.consume_stream_async(
            stream,
            handle_batch=_handle,
            on_heartbeat=_heartbeat,
            heartbeat_interval=0.5,
            idle_timeout=5.0,
            clock=lambda: 0.0,
        )

    asyncio.run(_run())

    assert [batch.channel for batch in handled] == ["ticker"]
    assert heartbeats == [2.0]
    assert stream.aclose_calls == 1
    assert stream.next_calls == 2


def test_consume_stream_async_raises_timeout_on_idle() -> None:
    batches = [
        StreamBatch(
            channel="ticker",
            events=(
                {
                    "symbol": "BTC/USDT",
                    "close": 100.0,
                    "timestamp": time.time(),
                },
            ),
            received_at=0.0,
        ),
        StreamBatch(channel="ticker", events=(), heartbeat=True, received_at=5.0),
    ]
    stream = _AsyncBatchStream(batches)

    async def _run() -> None:
        await pipeline_module.consume_stream_async(
            stream,
            handle_batch=lambda batch: None,
            heartbeat_interval=0.5,
            idle_timeout=1.0,
            clock=lambda: 0.0,
        )

    with pytest.raises(TimeoutError, match="Brak nowych danych"):
        asyncio.run(_run())

    assert stream.aclose_calls == 1
    assert stream.next_calls == 2


def test_consume_stream_async_respects_async_stop_condition() -> None:
    batches = [
        StreamBatch(
            channel="ticker",
            events=(
                {
                    "symbol": "BTC/USDT",
                    "close": 100.0,
                    "timestamp": time.time(),
                },
            ),
            received_at=1.0,
        ),
        StreamBatch(
            channel="ticker",
            events=(
                {
                    "symbol": "BTC/USDT",
                    "close": 102.0,
                    "timestamp": time.time(),
                },
            ),
            received_at=2.0,
        ),
    ]
    stream = _AsyncBatchStream(batches)
    processed: list[StreamBatch] = []
    should_stop = {"value": False}

    async def _handle(batch: StreamBatch) -> None:
        processed.append(batch)
        should_stop["value"] = True

    async def _stop_condition() -> bool:
        await asyncio.sleep(0)
        return should_stop["value"]

    async def _run() -> None:
        await pipeline_module.consume_stream_async(
            stream,
            handle_batch=_handle,
            stop_condition=_stop_condition,
            idle_timeout=None,
            clock=lambda: 0.0,
        )

    asyncio.run(_run())

    assert len(processed) == 1
    assert processed[0].events[0]["close"] == pytest.approx(100.0)
    assert stream.aclose_calls == 1
    assert stream.next_calls == 1

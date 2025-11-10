from __future__ import annotations

import json
import os
import sys
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Iterable, Mapping

import grpc
import pytest
import yaml

from google.protobuf import timestamp_pb2

from bot_core.api.server import LocalRuntimeContext, LocalRuntimeGateway
from bot_core.exchanges import interfaces as exchange_interfaces
from bot_core.exchanges import streaming as exchange_streaming
from bot_core.exchanges.base import AccountSnapshot
from bot_core.execution.paper import MarketMetadata
from bot_core.observability.metrics import MetricsRegistry
from bot_core.observability.ui_metrics import FeedHealthMetricsExporter
from bot_core.testing import TradingStubServer, build_default_dataset
from bot_core.testing.trading_stub_server import InMemoryTradingDataset
from bot_core.generated import trading_pb2, trading_pb2_grpc
from ui.backend.runtime_service import RuntimeService
from bot_core.runtime.journal import InMemoryTradingDecisionJournal, TradingDecisionEvent


def _build_stub_context(*, preset_dir: Path | None = None):
    markets = {
        "BTC/USDT": MarketMetadata(
            base_asset="BTC",
            quote_asset="USDT",
            min_quantity=0.001,
            min_notional=10.0,
            tick_size=0.1,
            step_size=0.001,
        )
    }

    class _DummyExecutionService:
        def __init__(self) -> None:
            self._markets = markets

        def execute(self, *_args, **_kwargs):
            return SimpleNamespace(order_id="demo-order", raw_response={"exchange_order_id": "demo"})

        def cancel(self, *_args, **_kwargs) -> None:  # pragma: no cover - not used in test
            return None

    class _DummyController:
        def __init__(self) -> None:
            self.symbols = ["BTC/USDT"]
            self.interval = "1h"
            self.execution_context = SimpleNamespace()

        def account_loader(self) -> AccountSnapshot:
            return AccountSnapshot(
                balances={"USDT": 1000.0},
                total_equity=1000.0,
                available_margin=1000.0,
                maintenance_margin=0.0,
            )

    class _DummyDataSource:
        def fetch_ohlcv(self, request):
            rows = [
                [request.start + index * 60_000, 100.0, 101.0, 99.5, 100.5, 1.0]
                for index in range(5)
            ]
            return SimpleNamespace(columns=("timestamp", "open", "high", "low", "close", "volume"), rows=rows)

    class _DummyAutoTrader:
        def configure_controller_runner(self, *_args, **_kwargs) -> None:
            return None

        def start(self) -> None:  # pragma: no cover - no-op
            return None

        def confirm_auto_trade(self, *_args, **_kwargs) -> None:  # pragma: no cover - no-op
            return None

        def stop(self) -> None:  # pragma: no cover - no-op
            return None

    pipeline = SimpleNamespace(
        execution_service=_DummyExecutionService(),
        controller=_DummyController(),
        bootstrap=SimpleNamespace(environment=SimpleNamespace(exchange="paper")),
        data_source=_DummyDataSource(),
    )

    context = LocalRuntimeContext(
        config=SimpleNamespace(trading=SimpleNamespace(), core=SimpleNamespace()),
        entrypoint=SimpleNamespace(trusted_auto_confirm=False),
        config_path=Path("."),
        pipeline=pipeline,
        trading_controller=pipeline.controller,
        runner=SimpleNamespace(),
        auto_trader=_DummyAutoTrader(),
        secret_manager=SimpleNamespace(),
    )
    if preset_dir is not None:
        context.strategy_presets_dir = preset_dir
    context.metrics_registry = None
    context.risk_store = None
    context.risk_builder = None
    context.risk_publisher = None
    context.marketplace_repository = None
    context.marketplace_enabled = False
    return context


@pytest.fixture
def ci_decision_feed_metrics(monkeypatch: pytest.MonkeyPatch) -> Path:
    target = Path("reports/ci/decision_feed_metrics.json")
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        target.unlink()
    monkeypatch.setenv("BOT_CORE_UI_FEED_LATENCY_PATH", str(target))
    yield target
    monkeypatch.delenv("BOT_CORE_UI_FEED_LATENCY_PATH", raising=False)


def _wait_for(condition: Callable[[], bool], app, *, timeout: float = 5.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if condition():
            return True
        app.processEvents()
        time.sleep(0.05)
    return condition()


def _to_pb_entries(records: Iterable[Mapping[str, str]]) -> list[trading_pb2.DecisionRecordEntry]:
    return [trading_pb2.DecisionRecordEntry(fields=dict(record)) for record in records]


@pytest.mark.integration
def test_local_runtime_gateway_exposes_health_data() -> None:
    context = _build_stub_context()
    gateway = LocalRuntimeGateway(context)
    payload = gateway.dispatch("health.check", {})
    assert payload["version"]
    assert int(payload["started_at_ms"]) > 0


def test_local_runtime_gateway_serves_market_data() -> None:
    context = _build_stub_context()
    gateway = LocalRuntimeGateway(context)
    response = gateway.dispatch(
        "market_data.get_ohlcv_history",
        {"symbol": "BTC/USDT", "limit": 3},
    )
    assert len(response["candles"]) == 3
    assert response["candles"][0]["close"] > 0


def test_local_runtime_gateway_saves_strategy_preset(tmp_path: Path) -> None:
    context = _build_stub_context(preset_dir=tmp_path)
    gateway = LocalRuntimeGateway(context)
    response = gateway.dispatch(
        "autotrader.save_strategy_preset",
        {
            "preset": {
                "name": "Alpha Momentum",
                "blocks": [
                    {"type": "data_feed", "label": "Feed", "params": {"symbol": "BTC/USDT"}},
                    {"type": "allocator", "label": "Fixed", "params": {"fraction": 0.1}},
                ],
            }
        },
    )
    assert response["ok"] is True
    path = Path(response["path"])
    assert path.exists()
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["name"] == "Alpha Momentum"
    assert payload["blocks"][0]["type"] == "data_feed"


def test_local_runtime_gateway_lists_and_loads_strategy_presets(tmp_path: Path) -> None:
    context = _build_stub_context(preset_dir=tmp_path)
    gateway = LocalRuntimeGateway(context)

    gateway.dispatch(
        "autotrader.save_strategy_preset",
        {
            "preset": {
                "name": "Beta Trend",
                "blocks": [
                    {"type": "data_feed", "label": "Feed", "params": {"symbol": "ETH/USDT"}},
                    {"type": "signal", "label": "Signal", "params": {"window": 14}},
                ],
            }
        },
    )

    gateway.dispatch(
        "autotrader.save_strategy_preset",
        {
            "preset": {
                "name": "Gamma Mean Reversion",
                "blocks": [
                    {"type": "data_feed", "label": "Feed", "params": {"symbol": "BTC/USDT"}},
                    {"type": "filter", "label": "Risk", "params": {"max_drawdown": 0.1}},
                ],
            }
        },
    )

    listing = gateway.dispatch("autotrader.list_strategy_presets", {})
    presets = listing["presets"]
    assert len(presets) == 2
    assert all(entry["block_count"] >= 2 for entry in presets)

    first = presets[0]
    loaded = gateway.dispatch("autotrader.load_strategy_preset", {"slug": first["slug"]})
    assert loaded["name"]
    assert len(loaded["blocks"]) == first["block_count"]

    second = presets[1]
    loaded_by_path = gateway.dispatch("autotrader.load_strategy_preset", {"path": second["path"]})
    assert loaded_by_path["path"].endswith(".json")
    assert loaded_by_path["blocks"][0]["type"] in {"data_feed", "filter", "signal"}

    removed = gateway.dispatch("autotrader.delete_strategy_preset", {"slug": first["slug"]})
    assert removed["ok"] is True
    removed_path = Path(removed["path"])
    assert not removed_path.exists()

    listing_after = gateway.dispatch("autotrader.list_strategy_presets", {})
    assert len(listing_after["presets"]) == 1
    assert listing_after["presets"][0]["slug"] == second["slug"]


def test_ui_transport_configuration_defaults_to_grpc() -> None:
    config_path = Path("ui/config/example.yaml")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert config["transport"]["mode"] == "grpc"
    serialized = config_path.read_text(encoding="utf-8").lower()
    assert "websocket" not in serialized


def test_streaming_layer_exposes_long_poll_only() -> None:
    exported = {name for name in dir(exchange_interfaces) if "WebSocket" in name}
    assert not exported
    assert hasattr(exchange_interfaces, "MarketStreamHandle")
    assert "websocket" not in (
        exchange_streaming.LocalLongPollStream.__doc__ or ""
    ).lower()
    assert not hasattr(exchange_streaming, "LocalWebSocketBridge")
    assert not hasattr(exchange_streaming.LocalLongPollStream, "websocket_bridge")


@pytest.mark.integration
def test_runtime_service_consumes_grpc_stream(
    ci_decision_feed_metrics: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("PySide6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtCore import QCoreApplication  # type: ignore[attr-defined]

    dataset = build_default_dataset()

    with TradingStubServer(dataset, port=0, stream_repeat=True, stream_interval=0.0) as server:
        monkeypatch.setenv("BOT_CORE_UI_GRPC_ENDPOINT", server.address)
        app = QCoreApplication.instance() or QCoreApplication([])
        service = RuntimeService(default_limit=5)
        try:
            assert service.attachToLiveDecisionLog("") is True
            assert _wait_for(lambda: bool(service.decisions), app)

            assert _wait_for(lambda: ci_decision_feed_metrics.exists(), app)
            payload = json.loads(ci_decision_feed_metrics.read_text(encoding="utf-8"))
            assert payload["count"] >= 1
            assert payload["max_ms"] >= payload["min_ms"] >= 0.0
            assert payload["status"] == "connected"
            assert "reconnects" in payload and payload["reconnects"] >= 0
            assert "downtime_ms" in payload and payload["downtime_ms"] >= 0.0
            assert "downtimeMs" in payload and payload["downtimeMs"] >= 0.0
            assert "last_latency_ms" in payload
            assert "p50_ms" in payload and payload["p50_ms"] >= 0.0
            assert "p95_ms" in payload and payload["p95_ms"] >= 0.0
            assert "latency_p95_ms" in payload and payload["latency_p95_ms"] == pytest.approx(payload["p95_ms"])
            assert "p95_seconds" in payload and payload["p95_seconds"] == pytest.approx(payload["p95_ms"] / 1000.0)
            assert payload["p95_ms"] <= 3000.0
            assert "nextRetrySeconds" in payload
            if payload["nextRetrySeconds"] is not None:
                assert payload["nextRetrySeconds"] >= 0.0
            health = service.feedHealth
            assert health["status"] == "connected"
            assert health["reconnects"] == payload["reconnects"]
            assert _wait_for(lambda: service.cycleMetrics.get("cycles_total", 0.0) >= 1.0, app)
            metrics = service.cycleMetrics
            assert metrics["cycles_total"] >= 1.0
            assert metrics["strategy_switch_total"] >= 0.0
            assert metrics["guardrail_blocks_total"] >= 0.0
            assert metrics.get("cycle_latency_p95_ms", 0.0) <= 3000.0
            assert metrics.get("cycle_latency_p50_ms", 0.0) <= metrics.get("cycle_latency_p95_ms", 0.0)
        finally:
            service._stop_grpc_stream()
            app.quit()
        monkeypatch.delenv("BOT_CORE_UI_GRPC_ENDPOINT", raising=False)


def test_runtime_service_handles_grpc_connection_error(monkeypatch) -> None:
    pytest.importorskip("PySide6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtCore import QCoreApplication  # type: ignore[attr-defined]

    monkeypatch.setenv("BOT_CORE_UI_GRPC_ENDPOINT", "grpc.invalid:0")

    def failing_start(self, target, limit):
        raise RuntimeError("gRPC unavailable")

    monkeypatch.setattr(RuntimeService, "_start_grpc_stream", failing_start)

    app = QCoreApplication.instance() or QCoreApplication([])
    service = RuntimeService(default_limit=3)
    try:
        demo = service.loadRecentDecisions(2)
        assert demo, "Fallback loader powinien zwrócić dane demo"

        assert service.attachToLiveDecisionLog("") is True
        assert not service._grpc_stream_active
        assert service.decisions, "Po błędzie gRPC oczekiwano decyzji z fallbacku"
        assert "grpc" in service.errorMessage.lower()
        assert service.feedHealth["status"] == "fallback"
        assert "grpc" in service.feedHealth["lastError"].lower()
        assert service.cycleMetrics == {}
    finally:
        service._stop_grpc_stream()
        app.quit()
        monkeypatch.delenv("BOT_CORE_UI_GRPC_ENDPOINT", raising=False)


@pytest.mark.integration
def test_grpc_decision_feed_snapshot_and_reconnect(
    ci_decision_feed_metrics: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("PySide6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtCore import QCoreApplication  # type: ignore[attr-defined]

    base_time = datetime(2024, 1, 1, tzinfo=timezone.utc)

    def _record_event(
        journal: InMemoryTradingDecisionJournal,
        *,
        minute_offset: int,
        event: str,
        status: str,
        confidence: float,
    ) -> None:
        journal.record(
            TradingDecisionEvent(
                event_type=event,
                timestamp=base_time + timedelta(minutes=minute_offset),
                environment="paper",
                portfolio="alpha",
                risk_profile="balanced",
                symbol="BTC/USDT",
                side="buy",
                quantity=0.1,
                price=42000.0 + minute_offset,
                status=status,
                schedule="auto",
                strategy="trend_follow",
                confidence=confidence,
                latency_ms=45.0 + minute_offset,
            )
        )

    snapshot_journal = InMemoryTradingDecisionJournal()
    _record_event(snapshot_journal, minute_offset=0, event="order_submitted", status="submitted", confidence=0.82)
    _record_event(snapshot_journal, minute_offset=1, event="order_filled", status="filled", confidence=0.91)

    increments_journal = InMemoryTradingDecisionJournal()
    _record_event(
        increments_journal,
        minute_offset=2,
        event="order_partially_filled",
        status="partial",
        confidence=0.76,
    )
    _record_event(
        increments_journal,
        minute_offset=3,
        event="order_closed",
        status="closed",
        confidence=0.88,
    )

    snapshot_entries = _to_pb_entries(snapshot_journal.export())
    increment_entries = _to_pb_entries(increments_journal.export())
    expected_increment_events = [entry.fields.get("event", "") for entry in increment_entries]

    dataset = InMemoryTradingDataset()
    dataset.set_decision_stream(snapshot=snapshot_entries, increments=increment_entries)

    app = None
    service: RuntimeService | None = None
    server_address = ""

    with TradingStubServer(dataset, port=0, stream_repeat=True, stream_interval=0.0) as server:
        monkeypatch.setenv("BOT_CORE_UI_GRPC_ENDPOINT", server.address)
        app = QCoreApplication.instance() or QCoreApplication([])

        channel = grpc.insecure_channel(server.address)
        stub = trading_pb2_grpc.RuntimeServiceStub(channel)

        response = stub.ListDecisions(trading_pb2.ListDecisionsRequest(limit=2))
        assert response.total == len(snapshot_entries) + len(increment_entries)
        assert response.cursor == 2
        assert len(response.records) == 2
        assert response.records[0].fields["event"] == "order_submitted"
        assert response.has_more is True

        follow_up = stub.ListDecisions(
            trading_pb2.ListDecisionsRequest(cursor=response.cursor, limit=10)
        )
        assert follow_up.cursor == follow_up.total
        assert len(follow_up.records) == len(increment_entries)

        since_ts = timestamp_pb2.Timestamp()
        since_ts.FromDatetime((base_time + timedelta(minutes=2)).astimezone(timezone.utc))
        filtered = stub.ListDecisions(
            trading_pb2.ListDecisionsRequest(
                filters=trading_pb2.DecisionJournalFilters(events=["order_closed"], since=since_ts)
            )
        )
        assert filtered.total == 1
        assert filtered.records[0].fields["status"] == "closed"

        stream = stub.StreamDecisions(trading_pb2.StreamDecisionsRequest(limit=3))
        first_update = next(stream)
        assert first_update.HasField("snapshot")
        assert len(first_update.snapshot.records) == len(snapshot_entries)
        streamed_events: list[str] = []
        for update in stream:
            if update.HasField("increment"):
                streamed_events.append(update.increment.record.fields.get("event", ""))
            if len(streamed_events) == len(expected_increment_events):
                break
        assert streamed_events == expected_increment_events
        channel.close()

        service = RuntimeService(default_limit=4)
        assert service.attachToLiveDecisionLog("") is True
        assert _wait_for(lambda: len(service.decisions) >= len(snapshot_entries), app)
        server_address = server.address

    assert service is not None and app is not None

    try:
        host, port_text = server_address.split(":", 1)
        port_value = int(port_text)

        assert _wait_for(
            lambda: service.feedHealth.get("status") in {"degraded", "retrying", "fallback"},
            app,
            timeout=12.0,
        )

        reconnect_event_name = "order_settled"
        _record_event(
            increments_journal,
            minute_offset=4,
            event=reconnect_event_name,
            status="settled",
            confidence=0.79,
        )
        updated_increments = _to_pb_entries(increments_journal.export())
        dataset.set_decision_stream(increments=updated_increments)

        with TradingStubServer(
            dataset,
            host=host,
            port=port_value,
            stream_repeat=True,
            stream_interval=0.0,
        ):
            assert _wait_for(
                lambda: service.feedHealth.get("status") == "connected"
                and service.feedHealth.get("reconnects", 0) >= 1,
                app,
                timeout=10.0,
            )
            assert _wait_for(
                lambda: any(entry.get("event") == reconnect_event_name for entry in service.decisions),
                app,
                timeout=5.0,
            )

            def _metrics_include_reconnects() -> bool:
                if not ci_decision_feed_metrics.exists():
                    return False
                payload = json.loads(ci_decision_feed_metrics.read_text(encoding="utf-8"))
                return payload.get("reconnects", 0) >= 1

            assert _wait_for(_metrics_include_reconnects, app, timeout=5.0)
            metrics_payload = json.loads(ci_decision_feed_metrics.read_text(encoding="utf-8"))
            assert metrics_payload["reconnects"] >= 1
            assert metrics_payload["p95_ms"] >= metrics_payload["p50_ms"] >= 0.0
            assert metrics_payload.get("latency_p95_ms", 0.0) == pytest.approx(metrics_payload["p95_ms"])
            assert metrics_payload["p95_ms"] <= 3000.0
    finally:
        service._stop_grpc_stream()
        app.quit()
        monkeypatch.delenv("BOT_CORE_UI_GRPC_ENDPOINT", raising=False)

def test_local_runtime_gateway_streams_decision_journal_with_cursor() -> None:
    context = _build_stub_context()
    journal = InMemoryTradingDecisionJournal()
    base_time = datetime(2024, 1, 1, tzinfo=timezone.utc)
    for index in range(5):
        event = TradingDecisionEvent(
            event_type="order_submitted" if index % 2 == 0 else "order_filled",
            timestamp=base_time + timedelta(minutes=index),
            environment="demo",
            portfolio="alpha",
            risk_profile="balanced",
            symbol="BTC/USDT",
            status="ok" if index % 2 == 0 else "filled",
            strategy="trend",
        )
        journal.record(event)
    context.auto_trader._decision_journal = journal  # type: ignore[attr-defined]

    gateway = LocalRuntimeGateway(context)

    first_batch = gateway.dispatch("autotrader.stream_decision_journal", {"limit": 2})
    assert first_batch["cursor"] == 2
    assert first_batch["total"] == 5
    assert len(first_batch["records"]) == 2

    second_batch = gateway.dispatch(
        "autotrader.stream_decision_journal",
        {"cursor": first_batch["cursor"], "limit": 2},
    )
    assert second_batch["cursor"] == 4
    assert len(second_batch["records"]) == 2
    assert all(entry["event"] in {"order_submitted", "order_filled"} for entry in second_batch["records"])

    filtered = gateway.dispatch(
        "autotrader.stream_decision_journal",
        {
            "filters": {
                "event": "order_filled",
                "since": (base_time + timedelta(minutes=2)).isoformat(),
            }
        },
    )
    assert filtered["cursor"] == filtered["total"]
    assert filtered["records"]
    assert all(entry["event"] == "order_filled" for entry in filtered["records"])

    filtered_epoch = gateway.dispatch(
        "autotrader.stream_decision_journal",
        {
            "filters": {
                "event": "order_filled",
                "since": int((base_time + timedelta(minutes=2)).timestamp() * 1000),
                "until": int((base_time + timedelta(minutes=5)).timestamp()),
            }
        },
    )
    assert filtered_epoch["records"]
    assert all(entry["event"] == "order_filled" for entry in filtered_epoch["records"])


def test_feed_health_threshold_alerts_and_metrics(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("PySide6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtCore import QCoreApplication  # type: ignore[attr-defined]

    monkeypatch.setenv("BOT_CORE_UI_FEED_LATENCY_P95_WARNING_MS", "1.0")
    monkeypatch.setenv("BOT_CORE_UI_FEED_LATENCY_P95_CRITICAL_MS", "2.0")
    monkeypatch.setenv("BOT_CORE_UI_FEED_RECONNECT_WARNING", "1")
    monkeypatch.setenv("BOT_CORE_UI_FEED_RECONNECT_CRITICAL", "2")
    monkeypatch.setenv("BOT_CORE_UI_FEED_DOWNTIME_WARNING_SECONDS", "0.001")
    monkeypatch.setenv("BOT_CORE_UI_FEED_DOWNTIME_CRITICAL_SECONDS", "0.002")

    events: list[dict[str, object]] = []

    class _Sink:
        def emit_feed_health_event(self, **payload: object) -> None:
            events.append({"severity": payload.get("severity"), "payload": payload.get("payload")})

    exporter = FeedHealthMetricsExporter(registry=MetricsRegistry())
    sink = _Sink()

    app = QCoreApplication.instance() or QCoreApplication([])
    service = RuntimeService(feed_alert_sink=sink, feed_metrics_exporter=exporter)
    try:
        service._active_stream_label = "grpc://demo"

        service._feed_latencies.clear()
        service._feed_latencies.append(10.0)
        service._update_feed_health(status="connected", reconnects=0, last_error="")

        service._feed_latencies.clear()
        service._feed_latencies.append(0.1)
        service._update_feed_health(status="connected", reconnects=0, last_error="")

        service._update_feed_health(status="connected", reconnects=5, last_error="retry")

        service._update_feed_health(status="connected", reconnects=0, last_error="")

        service._feed_downtime_total = 0.01
        service._update_feed_health(status="degraded", reconnects=0, last_error="timeout")

        service._feed_downtime_total = 0.0
        service._update_feed_health(status="connected", reconnects=0, last_error="")

        assert len(events) >= 6
        severity_sequence = [entry["severity"] for entry in events[:6]]
        assert severity_sequence == ["critical", "info", "critical", "info", "critical", "info"]

        dashboard = exporter.dashboard()
        assert dashboard, "Eksporter feedHealth powinien zawierać wpis adaptera"
        grpc_entry = next(entry for entry in dashboard if entry["adapter"] == "grpc")
        assert grpc_entry["status"] in {"connected", "degraded"}
        assert grpc_entry["latency_p95_ms"] is not None
        assert grpc_entry["reconnects"] >= 0
        assert grpc_entry["downtime_seconds"] >= 0.0
    finally:
        service._stop_grpc_stream()
        app.quit()

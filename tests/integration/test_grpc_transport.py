from __future__ import annotations

import json
import os
import statistics
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
from bot_core.observability.metrics import MetricsRegistry, get_global_metrics_registry
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

            assert _wait_for(
                lambda: ci_decision_feed_metrics.exists()
                and json.loads(ci_decision_feed_metrics.read_text(encoding="utf-8")).get(
                    "count", 0
                )
                >= 1,
                app,
            )
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
            transports = health.get("transports", {})
            assert "grpc" in transports
            grpc_view = transports["grpc"]
            assert grpc_view["p95_ms"] == pytest.approx(payload["p95_ms"])
            assert grpc_view["p50_ms"] == pytest.approx(payload["p50_ms"])
            sla_report = service.feedSlaReport
            assert sla_report["p95_ms"] == pytest.approx(payload["p95_ms"])
            assert sla_report["sla_state"] in {"ok", "warning"}
            assert _wait_for(lambda: service.cycleMetrics.get("cycles_total", 0.0) >= 1.0, app)
            metrics = service.cycleMetrics
            assert metrics["cycles_total"] >= 1.0
            assert metrics["strategy_switch_total"] >= 0.0
            assert metrics["guardrail_blocks_total"] >= 0.0
            assert metrics.get("cycle_latency_p95_ms", 0.0) <= 3000.0
            assert metrics.get("cycle_latency_p50_ms", 0.0) <= metrics.get("cycle_latency_p95_ms", 0.0)

            registry = get_global_metrics_registry()
            sla_labels = {
                "adapter": "grpc",
                "transport": "grpc",
                "environment": "default",
                "scope": "decision_feed",
            }
            assert registry.get("bot_ui_feed_latency_p95_ms").value(labels=sla_labels) >= 0.0
            assert registry.get("bot_ui_feed_latency_p50_ms").value(labels=sla_labels) >= 0.0
            assert registry.get("bot_ui_feed_reconnects_total").value(labels=sla_labels) >= 0.0
            assert registry.get("bot_ui_feed_downtime_seconds").value(labels=sla_labels) >= 0.0
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

        samples = service._latency_samples_for("grpc")
        samples.clear()
        samples.append(10.0)
        service._update_feed_health(status="connected", reconnects=0, last_error="")

        samples.clear()
        samples.append(0.1)
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

def test_runtime_service_emits_feed_alerts_when_threshold_crossed(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("PySide6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtCore import QCoreApplication  # type: ignore[attr-defined]

    events: list[dict[str, object]] = []

    class _RecordingSink:
        def emit_feed_health_event(self, *, severity: str, title: str, body: str, context=None, payload=None) -> None:  # pragma: no cover - pomocnicza struktura
            events.append(
                {
                    "severity": severity,
                    "title": title,
                    "body": body,
                    "context": dict(context or {}),
                }
            )

    monkeypatch.setattr(RuntimeService, "_auto_connect_grpc", lambda self: None)
    monkeypatch.setattr(RuntimeService, "_refresh_long_poll_metrics", lambda self: None)

    app = QCoreApplication.instance() or QCoreApplication([])
    exporter = FeedHealthMetricsExporter(registry=MetricsRegistry())
    service = RuntimeService(
        default_limit=1,
        decision_loader=lambda *_args, **_kwargs: [],
        feed_alert_sink=_RecordingSink(),
        feed_metrics_exporter=exporter,
    )
    try:
        service._feed_alert_state["latency"] = "ok"
        service._maybe_emit_feed_alert(
            "latency",
            "warning",
            metric_label="Latencja p95",
            unit="ms",
            value=4200.0,
            warning=3000.0,
            critical=6000.0,
            status="connected",
            adapter="grpc",
            reconnects=1,
            downtime_seconds=0.0,
            latency_p95=4200.0,
            last_error="",
        )
        assert events[-1]["severity"] == "warning"
        assert events[-1]["context"]["metric"] == "latency"

        service._maybe_emit_feed_alert(
            "latency",
            "ok",
            metric_label="Latencja p95",
            unit="ms",
            value=1800.0,
            warning=3000.0,
            critical=6000.0,
            status="connected",
            adapter="grpc",
            reconnects=1,
            downtime_seconds=0.0,
            latency_p95=1800.0,
            last_error="",
        )
        assert events[-1]["severity"] == "info"
        assert events[-1]["context"].get("state") == "recovered"
    finally:
        service._longpoll_timer.stop()
        service._stop_grpc_stream()
        app.quit()


def test_feed_health_reports_grpc_and_fallback_breakdown(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("PySide6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtCore import QCoreApplication  # type: ignore[attr-defined]

    monkeypatch.setattr(RuntimeService, "_auto_connect_grpc", lambda self: None)
    monkeypatch.setattr(RuntimeService, "_refresh_long_poll_metrics", lambda self: None)

    app = QCoreApplication.instance() or QCoreApplication([])
    service = RuntimeService(default_limit=1, decision_loader=lambda limit: [])
    try:
        service._active_stream_label = "grpc://demo"
        grpc_samples = service._latency_samples_for("grpc")
        grpc_samples.clear()
        grpc_samples.extend([5.0, 7.0, 9.0])
        service._update_feed_health(status="connected", reconnects=1, last_error="")

        service._active_stream_label = "offline-demo"
        fallback_samples = service._latency_samples_for("fallback")
        fallback_samples.clear()
        fallback_samples.extend([0.4, 0.6])
        service._update_feed_health(status="fallback", reconnects=3, last_error="grpc down")

        transports = service.feedHealth.get("transports", {})
        assert transports["grpc"]["p95_ms"] >= transports["grpc"]["p50_ms"] >= 0.0
        assert transports["grpc"]["status"] == "connected"
        assert transports["fallback"]["status"] == "fallback"
        assert transports["fallback"]["p50_ms"] == pytest.approx(statistics.median([0.4, 0.6]))
        sla_report = service.feedSlaReport
        assert sla_report["transports"]["grpc"]["p95_ms"] == pytest.approx(transports["grpc"]["p95_ms"])
        assert sla_report["transports"]["fallback"]["status"] == "fallback"
    finally:
        service._longpoll_timer.stop()
        service._stop_grpc_stream()
        app.quit()


def test_grpc_longpoll_oscillation_is_debounced(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("PySide6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtCore import QCoreApplication  # type: ignore[attr-defined]

    monkeypatch.setattr(RuntimeService, "_auto_connect_grpc", lambda self: None)
    monkeypatch.setattr(RuntimeService, "_refresh_long_poll_metrics", lambda self: None)
    monkeypatch.setenv("BOT_CORE_UI_FEED_LATENCY_P95_WARNING_MS", "150.0")
    monkeypatch.setenv("BOT_CORE_UI_FEED_LATENCY_P95_CRITICAL_MS", "500.0")

    app = QCoreApplication.instance() or QCoreApplication([])
    service = RuntimeService(default_limit=1, decision_loader=lambda limit: [])
    try:
        service._active_stream_label = "grpc://demo"
        grpc_samples = service._latency_samples_for("grpc")
        grpc_samples.clear()
        grpc_samples.extend([45.0, 55.0])
        service._update_feed_health(status="connected", reconnects=0, last_error="")
        initial_state = service.feedSlaReport.get("sla_state")

        history: list[str | None] = []
        for index in range(1, 5):
            if index % 2 == 0:
                service._active_stream_label = "fallback://demo"
                samples = service._latency_samples_for("fallback")
                status = "fallback"
            else:
                service._active_stream_label = "grpc://demo"
                samples = service._latency_samples_for("grpc")
                status = "connected"
            samples.clear()
            samples.extend([1100.0 + index * 50])
            service._feed_reconnects += 1
            service._update_feed_health(
                status=status,
                reconnects=service._feed_reconnects,
                last_error="flaky transport",
            )
            report = service.feedSlaReport
            history.append(report.get("sla_state"))
            assert report.get("consecutive_degraded_periods") == index
            assert report.get("consecutive_healthy_periods") == 0

        assert initial_state == "ok"
        assert history and len(set(history)) == 1
        assert history[0] in {"warning", "critical"}
    finally:
        service._longpoll_timer.stop()
        service._stop_grpc_stream()
        app.quit()


def test_long_fallback_periods_keep_sla_alerts_stable(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("PySide6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtCore import QCoreApplication  # type: ignore[attr-defined]

    events: list[dict[str, object]] = []

    class _RecordingSink:
        def emit_feed_health_event(self, *, severity: str, title: str, body: str, context=None, payload=None) -> None:  # pragma: no cover - pomocnicza struktura
            events.append(
                {
                    "severity": severity,
                    "title": title,
                    "body": body,
                    "context": dict(context or {}),
                }
            )

    monkeypatch.setattr(RuntimeService, "_auto_connect_grpc", lambda self: None)
    monkeypatch.setattr(RuntimeService, "_refresh_long_poll_metrics", lambda self: None)

    app = QCoreApplication.instance() or QCoreApplication([])
    exporter = FeedHealthMetricsExporter(registry=MetricsRegistry())
    service = RuntimeService(
        default_limit=1,
        decision_loader=lambda *_args, **_kwargs: [],
        feed_alert_sink=_RecordingSink(),
        feed_metrics_exporter=exporter,
    )
    try:
        service._feed_thresholds.update(
            {
                "latency_warning_ms": 120.0,
                "latency_critical_ms": 180.0,
                "reconnects_warning": 1,
                "reconnects_critical": 2,
                "downtime_warning_seconds": 60.0,
                "downtime_critical_seconds": 300.0,
            }
        )
        service._latency_samples_for("grpc").clear()
        service._active_stream_label = "fallback://demo"
        fallback_samples = service._latency_samples_for("fallback")
        for hour in range(3):
            fallback_samples.clear()
            fallback_samples.extend([240.0 + hour * 5])
            service._feed_reconnects = 1
            service._feed_downtime_total = float((hour + 1) * 3600)
            service._update_feed_health(
                status="fallback", reconnects=service._feed_reconnects, last_error="grpc unavailable"
            )
            assert service.feedSlaReport.get("sla_state") in {"warning", "critical"}
            assert service.feedSlaReport.get("consecutive_degraded_periods", 0) >= hour + 1

        assert len(events) == 3
        assert {entry["context"].get("metric") for entry in events} == {"latency", "reconnects", "downtime"}

        fallback_dashboard = [entry for entry in exporter.dashboard() if entry.get("adapter") == "fallback"]
        assert len(fallback_dashboard) == 1
        fallback_entry = fallback_dashboard[0]
        assert fallback_entry["status"] == "fallback"
        assert fallback_entry["reconnects"] == 1
        assert fallback_entry["downtime_seconds"] == pytest.approx(3600.0 * 3, rel=0.01)

        registry = exporter._registry
        fallback_labels = {
            "adapter": "fallback",
            "transport": "fallback",
            "environment": "default",
            "scope": "decision_feed",
        }
        assert registry.get("bot_ui_feed_sla_reconnects_total").value(labels=fallback_labels) == pytest.approx(1.0)
        assert registry.get("bot_ui_feed_sla_downtime_seconds").value(labels=fallback_labels) == pytest.approx(3600.0 * 3, rel=0.01)

        service._active_stream_label = "grpc://demo"
        service._latency_samples_for("fallback").clear()
        fallback_samples.clear()
        fallback_samples.extend([40.0])
        service._feed_downtime_total = 0.0
        service._feed_reconnects = 0
        service._update_feed_health(status="connected", reconnects=service._feed_reconnects, last_error="")
        assert service.feedSlaReport.get("sla_state") == "ok"
        assert len(events) == 6
        recovery_states = [entry for entry in events if entry["severity"] == "info"]
        assert len(recovery_states) == 3

        dashboard = exporter.dashboard()
        assert any(entry.get("adapter") == "grpc" and entry.get("status") == "connected" for entry in dashboard)
        grpc_labels = {
            "adapter": "grpc",
            "transport": "grpc",
            "environment": "default",
            "scope": "decision_feed",
        }
        assert registry.get("bot_ui_feed_sla_reconnects_total").value(labels=grpc_labels) == pytest.approx(0.0)
        assert registry.get("bot_ui_feed_sla_downtime_seconds").value(labels=grpc_labels) == pytest.approx(0.0)
    finally:
        service._longpoll_timer.stop()
        service._stop_grpc_stream()
        app.quit()


def test_long_horizon_fallback_increments_consecutive_counters(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("PySide6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtCore import QCoreApplication  # type: ignore[attr-defined]

    monkeypatch.setattr(RuntimeService, "_auto_connect_grpc", lambda self: None)
    monkeypatch.setattr(RuntimeService, "_refresh_long_poll_metrics", lambda self: None)

    app = QCoreApplication.instance() or QCoreApplication([])
    service = RuntimeService(default_limit=1, decision_loader=lambda *_args, **_kwargs: [])
    try:
        service._feed_thresholds.update(
            {
                "latency_warning_ms": 100.0,
                "latency_critical_ms": 180.0,
                "reconnects_warning": 1,
                "reconnects_critical": 4,
                "downtime_warning_seconds": 60.0,
                "downtime_critical_seconds": 600.0,
            }
        )
        service._active_stream_label = "fallback://demo"
        for key in ("grpc", "fallback"):
            service._latency_samples_for(key).clear()

        for hour in range(6):
            samples = service._latency_samples_for("fallback")
            samples.clear()
            samples.extend([280.0 + hour * 5])
            service._feed_reconnects = hour + 1
            service._feed_downtime_total = float((hour + 1) * 3600)
            service._update_feed_health(
                status="fallback", reconnects=service._feed_reconnects, last_error="grpc unavailable"
            )
            report = service.feedSlaReport
            assert report.get("sla_state") in {"warning", "critical"}
            assert report.get("consecutive_degraded_periods") == hour + 1
            assert report.get("consecutive_healthy_periods") == 0

        service._active_stream_label = "grpc://demo"
        service._feed_reconnects = 0
        service._feed_downtime_total = 0.0
        for key in ("grpc", "fallback"):
            service._latency_samples_for(key).clear()

        healthy_progression: list[int | None] = []
        for _ in range(3):
            grpc_samples = service._latency_samples_for("grpc")
            grpc_samples.clear()
            grpc_samples.extend([55.0])
            service._update_feed_health(status="connected", reconnects=service._feed_reconnects, last_error="")
            report = service.feedSlaReport
            healthy_progression.append(report.get("consecutive_healthy_periods"))
            assert report.get("consecutive_degraded_periods") == 0

        assert healthy_progression == [1, 2, 3]
    finally:
        service._longpoll_timer.stop()
        service._stop_grpc_stream()
        app.quit()


def test_frequent_transport_switches_do_not_reset_alert_hysteresis(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("PySide6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtCore import QCoreApplication  # type: ignore[attr-defined]

    events: list[dict[str, object]] = []

    class _RecordingSink:
        def emit_feed_health_event(self, *, severity: str, title: str, body: str, context=None, payload=None) -> None:  # pragma: no cover - pomocnicza struktura
            events.append(
                {
                    "severity": severity,
                    "title": title,
                    "body": body,
                    "context": dict(context or {}),
                }
            )

    monkeypatch.setattr(RuntimeService, "_auto_connect_grpc", lambda self: None)
    monkeypatch.setattr(RuntimeService, "_refresh_long_poll_metrics", lambda self: None)

    app = QCoreApplication.instance() or QCoreApplication([])
    service = RuntimeService(
        default_limit=1,
        decision_loader=lambda *_args, **_kwargs: [],
        feed_alert_sink=_RecordingSink(),
    )
    try:
        service._feed_thresholds.update(
            {
                "latency_warning_ms": 150.0,
                "latency_critical_ms": 200.0,
                "reconnects_warning": 1,
                "reconnects_critical": 3,
                "downtime_warning_seconds": 30.0,
                "downtime_critical_seconds": 120.0,
            }
        )
        service._active_stream_label = "grpc://demo"
        grpc_samples = service._latency_samples_for("grpc")
        grpc_samples.clear()
        grpc_samples.extend([80.0])
        service._feed_reconnects = 0
        service._feed_downtime_total = 0.0
        service._update_feed_health(status="connected", reconnects=service._feed_reconnects, last_error="")
        assert service.feedSlaReport.get("sla_state") == "ok"

        consecutive_degraded: list[int] = []
        consecutive_healthy: list[int] = []

        for index in range(6):
            active_label = "fallback://demo" if index % 2 == 0 else "grpc://demo"
            status = "fallback" if active_label.startswith("fallback") else "connected"
            service._active_stream_label = active_label
            service._feed_reconnects = index + 1
            service._feed_downtime_total = 180.0
            for key in ("grpc", "fallback"):
                service._latency_samples_for(key).clear()
            samples = service._latency_samples_for("fallback" if status == "fallback" else "grpc")
            samples.extend([240.0 + index])
            service._update_feed_health(
                status=status, reconnects=service._feed_reconnects, last_error="flapping stream"
            )
            report = service.feedSlaReport
            assert report.get("sla_state") in {"warning", "critical"}
            consecutive_degraded.append(report.get("consecutive_degraded_periods", 0))
            consecutive_healthy.append(report.get("consecutive_healthy_periods", 0))

        assert consecutive_degraded == [1, 2, 3, 4, 5, 6]
        assert all(value == 0 for value in consecutive_healthy)

        degraded_metrics = {
            entry["context"].get("metric") for entry in events if entry["severity"] != "info"
        }
        assert degraded_metrics.issuperset({"latency", "reconnects", "downtime"})

        service._active_stream_label = "grpc://demo"
        for key in ("grpc", "fallback"):
            service._latency_samples_for(key).clear()
        samples = service._latency_samples_for("grpc")
        samples.extend([60.0])
        service._feed_reconnects = 0
        service._feed_downtime_total = 0.0
        service._update_feed_health(status="connected", reconnects=service._feed_reconnects, last_error="")
        assert service.feedSlaReport.get("sla_state") == "ok"
        assert service.feedSlaReport.get("consecutive_healthy_periods") == 1
        recovery_metrics = {
            entry["context"].get("metric") for entry in events if entry["severity"] == "info"
        }
        assert recovery_metrics.issuperset({"latency", "reconnects", "downtime"})
    finally:
        service._longpoll_timer.stop()
        service._stop_grpc_stream()
        app.quit()


def test_repeated_long_fallback_cycles_preserve_transport_counters(
    monkeypatch: pytest.MonkeyPatch, ci_decision_feed_metrics: Path
) -> None:
    pytest.importorskip("PySide6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtCore import QCoreApplication  # type: ignore[attr-defined]

    monkeypatch.setattr(RuntimeService, "_auto_connect_grpc", lambda self: None)
    monkeypatch.setattr(RuntimeService, "_refresh_long_poll_metrics", lambda self: None)

    app = QCoreApplication.instance() or QCoreApplication([])
    exporter = FeedHealthMetricsExporter(registry=MetricsRegistry())
    service = RuntimeService(
        default_limit=1,
        decision_loader=lambda *_args, **_kwargs: [],
        feed_metrics_exporter=exporter,
    )
    try:
        service._feed_thresholds.update(
            {
                "latency_warning_ms": 140.0,
                "latency_critical_ms": 220.0,
                "reconnects_warning": 1,
                "reconnects_critical": 2,
                "downtime_warning_seconds": 45.0,
                "downtime_critical_seconds": 240.0,
            }
        )
        service._active_stream_label = "grpc://demo"
        service._latency_samples_for("grpc").clear()
        service._latency_samples_for("grpc").extend([75.0])
        service._update_feed_health(status="connected", reconnects=0, last_error="")
        assert service.feedSlaReport.get("sla_state") == "ok"

        for cycle in range(2):
            service._active_stream_label = "fallback://demo"
            fallback_samples = service._latency_samples_for("fallback")
            fallback_samples.clear()
            fallback_samples.extend([320.0 + cycle * 10])
            service._feed_reconnects = cycle + 1
            service._feed_downtime_total = float((cycle + 1) * 1800)
            service._update_feed_health(
                status="fallback", reconnects=service._feed_reconnects, last_error="grpc timeout"
            )
            sla_state = service.feedSlaReport.get("sla_state")
            assert sla_state in {"warning", "critical"}
            assert service.feedSlaReport.get("consecutive_degraded_periods") >= cycle + 1

        fallback_dashboard = [entry for entry in exporter.dashboard() if entry.get("adapter") == "fallback"]
        assert len(fallback_dashboard) == 1
        fallback_entry = fallback_dashboard[0]
        assert fallback_entry["reconnects"] == 2
        assert fallback_entry["downtime_seconds"] == pytest.approx(3600.0, rel=0.01)

        registry = exporter._registry
        fallback_labels = {
            "adapter": "fallback",
            "transport": "fallback",
            "environment": "default",
            "scope": "decision_feed",
        }
        assert registry.get("bot_ui_feed_sla_reconnects_total").value(labels=fallback_labels) == pytest.approx(2.0)
        assert registry.get("bot_ui_feed_sla_downtime_seconds").value(labels=fallback_labels) == pytest.approx(3600.0, rel=0.01)

        service._active_stream_label = "grpc://demo"
        service._feed_reconnects = 0
        service._feed_downtime_total = 0.0
        for key in ("grpc", "fallback"):
            service._latency_samples_for(key).clear()
        service._latency_samples_for("grpc").extend([65.0])
        service._update_feed_health(status="connected", reconnects=service._feed_reconnects, last_error="")
        assert service.feedSlaReport.get("sla_state") == "ok"

        dashboard = exporter.dashboard()
        grpc_labels = {
            "adapter": "grpc",
            "transport": "grpc",
            "environment": "default",
            "scope": "decision_feed",
        }
        assert any(entry.get("adapter") == "grpc" and entry.get("status") == "connected" for entry in dashboard)
        assert registry.get("bot_ui_feed_sla_reconnects_total").value(labels=grpc_labels) == pytest.approx(0.0)
        assert registry.get("bot_ui_feed_sla_downtime_seconds").value(labels=grpc_labels) == pytest.approx(0.0)

        payload = json.loads(ci_decision_feed_metrics.read_text(encoding="utf-8"))
        assert payload.get("status") == "connected"
        assert payload.get("transports", {}).get("fallback", {}).get("reconnects") == 2
        assert payload.get("transports", {}).get("grpc", {}).get("reconnects") == 0
    finally:
        service._longpoll_timer.stop()
        service._stop_grpc_stream()
        app.quit()


@pytest.mark.integration
@pytest.mark.soak
def test_grpc_longpoll_soak_flapping_counters(
    ci_decision_feed_metrics: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("PySide6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtCore import QCoreApplication  # type: ignore[attr-defined]

    monkeypatch.setattr(RuntimeService, "_auto_connect_grpc", lambda self: None)
    monkeypatch.setattr(RuntimeService, "_refresh_long_poll_metrics", lambda self: None)

    app = QCoreApplication.instance() or QCoreApplication([])
    exporter = FeedHealthMetricsExporter(registry=MetricsRegistry())
    service = RuntimeService(
        default_limit=1,
        decision_loader=lambda *_args, **_kwargs: [],
        feed_metrics_exporter=exporter,
    )
    try:
        service._feed_thresholds.update(
            {
                "latency_warning_ms": 150.0,
                "latency_critical_ms": 220.0,
                "reconnects_warning": 20,
                "reconnects_critical": 50,
                "downtime_warning_seconds": 86_400.0,
                "downtime_critical_seconds": 172_800.0,
            }
        )

        degraded_streaks: list[int] = []
        healthy_streaks: list[int] = []
        reconnect_history: list[int] = []
        peak_degraded = 0
        peak_healthy = 0
        hours_elapsed = 0
        downtime_seconds_total = 0.0

        for hour in range(24):
            is_fallback_window = hour % 4 < 2
            hours_elapsed += 1
            if is_fallback_window:
                service._active_stream_label = "fallback://demo"
                for key in ("grpc", "fallback"):
                    service._latency_samples_for(key).clear()
                fallback_samples = service._latency_samples_for("fallback")
                fallback_samples.extend([320.0 + hour])
                service._feed_reconnects += 1
                downtime_seconds_total += 3600.0
                service._feed_downtime_total = downtime_seconds_total
                service._update_feed_health(
                    status="fallback", reconnects=service._feed_reconnects, last_error="grpc unavailable"
                )
                report = service.feedSlaReport
                degraded_streaks.append(report.get("consecutive_degraded_periods", 0))
                reconnect_history.append(report.get("reconnects", 0))
                peak_degraded = max(peak_degraded, report.get("consecutive_degraded_periods", 0))
            else:
                service._active_stream_label = "grpc://demo"
                for key in ("grpc", "fallback"):
                    service._latency_samples_for(key).clear()
                grpc_samples = service._latency_samples_for("grpc")
                grpc_samples.extend([70.0])
                service._feed_downtime_total = downtime_seconds_total
                service._update_feed_health(status="connected", reconnects=service._feed_reconnects, last_error="")
                report = service.feedSlaReport
                healthy_streaks.append(report.get("consecutive_healthy_periods", 0))
                peak_healthy = max(peak_healthy, report.get("consecutive_healthy_periods", 0))

        assert hours_elapsed == 24
        assert peak_degraded >= 2
        assert peak_healthy >= 2
        assert degraded_streaks[:4] == [1, 2, 1, 2]
        assert healthy_streaks[:4] == [1, 2, 1, 2]
        assert reconnect_history[-1] == service._feed_reconnects == 12

        final_report = service.feedSlaReport
        assert final_report.get("consecutive_healthy_periods") >= healthy_streaks[-1]
        assert final_report.get("downtime_seconds", 0.0) == pytest.approx(downtime_seconds_total)
        service._write_feed_metrics(force=True)
        payload = json.loads(ci_decision_feed_metrics.read_text(encoding="utf-8"))
        assert payload.get("reconnects") == reconnect_history[-1]
        assert payload.get("downtime_ms") == pytest.approx(downtime_seconds_total * 1000.0)
        assert payload.get("consecutive_degraded_periods") == 0
        assert payload.get("downtime_seconds") == pytest.approx(downtime_seconds_total)
        assert payload.get("consecutive_healthy_periods", 0) >= healthy_streaks[-1]
        assert payload.get("status") == "connected"
        transports = payload.get("transports", {})
        assert set(transports) >= {"grpc", "fallback"}
        assert transports["fallback"].get("downtimeMs", 0.0) == pytest.approx(downtime_seconds_total * 1000.0)
        assert transports["fallback"].get("status") == "fallback"
        assert transports["grpc"].get("status") == "connected"
        assert transports["fallback"]["reconnects"] >= 6
    finally:
        service._longpoll_timer.stop()
        service._stop_grpc_stream()
        app.quit()


def test_oscillation_between_transports_keeps_counters_balanced(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("PySide6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtCore import QCoreApplication  # type: ignore[attr-defined]

    monkeypatch.setattr(RuntimeService, "_auto_connect_grpc", lambda self: None)
    monkeypatch.setattr(RuntimeService, "_refresh_long_poll_metrics", lambda self: None)

    app = QCoreApplication.instance() or QCoreApplication([])
    service = RuntimeService(default_limit=1, decision_loader=lambda *_args, **_kwargs: [])
    try:
        service._feed_thresholds.update(
            {
                "latency_warning_ms": 120.0,
                "latency_critical_ms": 200.0,
                "reconnects_warning": 2,
                "reconnects_critical": 5,
                "downtime_warning_seconds": 45.0,
                "downtime_critical_seconds": 180.0,
            }
        )
        service._active_stream_label = "grpc://demo"
        service._latency_samples_for("grpc").clear()
        service._latency_samples_for("grpc").extend([65.0])
        service._update_feed_health(status="connected", reconnects=0, last_error="")
        assert service.feedSlaReport.get("consecutive_healthy_periods") == 1

        degraded_history: list[int] = []
        healthy_history: list[int] = [1]
        degraded_side_reset: list[int] = []
        for cycle in range(4):
            service._active_stream_label = "fallback://demo"
            for key in ("grpc", "fallback"):
                service._latency_samples_for(key).clear()
            fallback_samples = service._latency_samples_for("fallback")
            fallback_samples.extend([260.0 + cycle * 10])
            service._feed_reconnects = cycle + 1
            service._feed_downtime_total = float(120.0 * (cycle + 1))
            service._update_feed_health(
                status="fallback", reconnects=service._feed_reconnects, last_error="forced fallback"
            )
            report = service.feedSlaReport
            degraded_history.append(report.get("consecutive_degraded_periods", 0))
            degraded_side_reset.append(report.get("consecutive_healthy_periods", 0))

            service._active_stream_label = "grpc://demo"
            for key in ("grpc", "fallback"):
                service._latency_samples_for(key).clear()
            grpc_samples = service._latency_samples_for("grpc")
            grpc_samples.extend([60.0])
            service._feed_reconnects = 0
            service._feed_downtime_total = 0.0
            service._update_feed_health(status="connected", reconnects=service._feed_reconnects, last_error="")
            recovery_report = service.feedSlaReport
            healthy_history.append(recovery_report.get("consecutive_healthy_periods", 0))
            assert recovery_report.get("consecutive_degraded_periods") == 0

        assert all(value == 1 for value in degraded_history)
        assert all(value == 0 for value in degraded_side_reset)
        assert healthy_history == [1] * len(healthy_history)
    finally:
        service._longpoll_timer.stop()
        service._stop_grpc_stream()
        app.quit()

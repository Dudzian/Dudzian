"""Testy serwera MetricsService obsługującego telemetrię UI."""

from __future__ import annotations

import pytest

grpc = pytest.importorskip("grpc", reason="Wymaga biblioteki grpcio")
trading_pb2 = pytest.importorskip(
    "bot_core.generated.trading_pb2",
    reason="Brak wygenerowanych stubów trading_pb2",
)
trading_pb2_grpc = pytest.importorskip(
    "bot_core.generated.trading_pb2_grpc",
    reason="Brak wygenerowanych stubów trading_pb2_grpc",
)

import json
import time

from bot_core.runtime.metrics_service import (
    MetricsServer,
    MetricsSnapshotStore,
    MetricsServiceServicer,
    ReduceMotionAlertSink,
    OverlayBudgetAlertSink,
    create_server,
)
from bot_core.alerts.base import AlertRouter, AlertMessage


@pytest.mark.timeout(5)
def test_push_and_stream_metrics(tmp_path):
    """Przesyłanie i strumieniowanie metryk trafia do store oraz sinków."""

    received_notes: list[str] = []

    class CaptureSink:
        def handle_snapshot(self, snapshot) -> None:
            copy = trading_pb2.MetricsSnapshot()
            copy.CopyFrom(snapshot)
            received_notes.append(copy.notes)

    server = MetricsServer(host="127.0.0.1", port=0, sinks=[CaptureSink()], history_size=8)
    server.start()
    address = server.address

    try:
        channel = grpc.insecure_channel(address)
        stub = trading_pb2_grpc.MetricsServiceStub(channel)

        snapshot = trading_pb2.MetricsSnapshot()
        snapshot.notes = "reduce_motion:true"
        snapshot.fps = 58.5
        ack = stub.PushMetrics(snapshot)
        assert ack.accepted is True

        # Strumień najpierw powinien dostarczyć historię.
        stream = stub.StreamMetrics(trading_pb2.MetricsRequest(include_ui_metrics=True))
        first = next(stream)
        assert first.notes == "reduce_motion:true"

        # Dodaj kolejny snapshot i upewnij się, że trafi do strumienia.
        snapshot2 = trading_pb2.MetricsSnapshot()
        snapshot2.notes = "overlay_budget"
        snapshot2.fps = 60.0
        stub.PushMetrics(snapshot2)

        second = next(stream)
        assert second.notes == "overlay_budget"

        # Sink powinien otrzymać oba wpisy.
        assert received_notes == ["reduce_motion:true", "overlay_budget"]
    finally:
        server.stop(grace=0)


def test_store_enforces_history_limit():
    store = MetricsSnapshotStore(maxlen=2)

    snapshot = trading_pb2.MetricsSnapshot()
    snapshot.notes = "one"
    store.append(snapshot)

    snapshot2 = trading_pb2.MetricsSnapshot()
    snapshot2.notes = "two"
    store.append(snapshot2)

    snapshot3 = trading_pb2.MetricsSnapshot()
    snapshot3.notes = "three"
    store.append(snapshot3)

    history = [snap.notes for snap in store.snapshot_history()]
    assert history == ["two", "three"]




def test_build_metrics_server_from_config_disabled():
    from bot_core.runtime.metrics_service import build_metrics_server_from_config
    from bot_core.config.models import MetricsServiceConfig

    config = MetricsServiceConfig(enabled=False)
    assert build_metrics_server_from_config(config) is None


@pytest.mark.timeout(5)
def test_jsonl_sink_and_create_server(tmp_path):
    jsonl_file = tmp_path / "metrics.jsonl"
    server = create_server(
        host="127.0.0.1",
        port=0,
        history_size=4,
        enable_logging_sink=False,
        jsonl_path=jsonl_file,
    )
    server.start()
    channel = grpc.insecure_channel(server.address)
    stub = trading_pb2_grpc.MetricsServiceStub(channel)
    snapshot = trading_pb2.MetricsSnapshot()
    snapshot.notes = "jsonl-test"
    snapshot.fps = 59.3
    stub.PushMetrics(snapshot)
    time.sleep(0.1)
    server.stop(grace=0)

    assert jsonl_file.exists()
    lines = [line for line in jsonl_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert lines, "Plik JSONL powinien zawierać co najmniej jeden wpis"
    last = json.loads(lines[-1])
    assert last["notes"] == "jsonl-test"
    assert last["fps"] == pytest.approx(59.3)


@pytest.mark.timeout(5)
def test_metrics_server_requires_auth_token():
    token = "secret-token"
    server = create_server(
        host="127.0.0.1",
        port=0,
        history_size=4,
        enable_logging_sink=False,
        auth_token=token,
    )
    server.start()
    try:
        channel = grpc.insecure_channel(server.address)
        stub = trading_pb2_grpc.MetricsServiceStub(channel)

        snapshot = trading_pb2.MetricsSnapshot()
        snapshot.notes = "auth-test"

        with pytest.raises(grpc.RpcError) as exc:
            stub.PushMetrics(snapshot)
        assert exc.value.code() == grpc.StatusCode.UNAUTHENTICATED

        metadata = (("authorization", f"Bearer {token}"),)
        ack = stub.PushMetrics(snapshot, metadata=metadata)
        assert ack.accepted is True

        stream = stub.StreamMetrics(
            trading_pb2.MetricsRequest(include_ui_metrics=True),
            metadata=metadata,
        )
        first = next(stream)
        assert first.notes == "auth-test"
    finally:
        server.stop(grace=0)


class _RecorderRouter(AlertRouter):
    def __init__(self) -> None:
        self.channels = []
        self.messages: list[AlertMessage] = []

    def register(self, channel):  # pragma: no cover - nie używamy w teście
        self.channels.append(channel)

    def dispatch(self, message: AlertMessage) -> None:
        self.messages.append(message)


def test_reduce_motion_alert_sink_dispatches_alert():
    router = _RecorderRouter()
    sink = ReduceMotionAlertSink(
        router,
        category="ui.performance",
        severity_active="critical",
        severity_recovered="notice",
    )
    store = MetricsSnapshotStore(maxlen=8)
    servicer = MetricsServiceServicer(store, sinks=[sink])

    snapshot = trading_pb2.MetricsSnapshot()
    snapshot.notes = json.dumps(
        {
            "event": "reduce_motion",
            "active": True,
            "overlay_active": 2,
            "overlay_allowed": 3,
            "fps_target": 120,
            "window_count": 2,
            "disable_secondary_fps": 90,
            "tag": "ci-smoke",
        }
    )
    snapshot.fps = 47.2
    servicer.PushMetrics(snapshot, None)

    assert len(router.messages) == 1
    message = router.messages[0]
    assert message.category == "ui.performance"
    assert message.severity == "critical"
    assert "Reduce motion" in message.title
    assert message.context["event"] == "reduce_motion"
    assert message.context["active"] == "true"
    assert message.context["overlay_active"] == "2"

    # Drugi snapshot dezaktywujący powinien wywołać alert informacyjny.
    snapshot2 = trading_pb2.MetricsSnapshot()
    snapshot2.notes = json.dumps(
        {
            "event": "reduce_motion",
            "active": False,
            "overlay_active": 1,
            "overlay_allowed": 3,
            "fps_target": 120,
        }
    )
    servicer.PushMetrics(snapshot2, None)
    assert len(router.messages) == 2
    assert router.messages[1].severity == "notice"


def test_reduce_motion_alert_sink_ignores_duplicates():
    router = _RecorderRouter()
    sink = ReduceMotionAlertSink(router)
    store = MetricsSnapshotStore(maxlen=4)
    servicer = MetricsServiceServicer(store, sinks=[sink])

    payload = json.dumps({"event": "reduce_motion", "active": True})
    snapshot = trading_pb2.MetricsSnapshot()
    snapshot.notes = payload
    servicer.PushMetrics(snapshot, None)

    duplicate = trading_pb2.MetricsSnapshot()
    duplicate.notes = payload
    servicer.PushMetrics(duplicate, None)

    assert len(router.messages) == 1, "Duplikaty reduce-motion nie powinny generować kolejnych alertów"


def test_overlay_budget_alert_sink_dispatches_alert():
    router = _RecorderRouter()
    sink = OverlayBudgetAlertSink(
        router,
        category="ui.performance.overlay",
        severity_exceeded="critical",
        severity_recovered="notice",
    )
    store = MetricsSnapshotStore(maxlen=8)
    servicer = MetricsServiceServicer(store, sinks=[sink])

    snapshot = trading_pb2.MetricsSnapshot()
    snapshot.notes = json.dumps(
        {
            "event": "overlay_budget",
            "active_overlays": 4,
            "allowed_overlays": 2,
            "reduce_motion": False,
            "fps_target": 120,
        }
    )
    servicer.PushMetrics(snapshot, None)

    assert len(router.messages) == 1
    message = router.messages[0]
    assert message.category == "ui.performance.overlay"
    assert message.severity == "critical"
    assert message.context["exceeded"] == "true"
    assert message.context["active_overlays"] == "4"

    snapshot2 = trading_pb2.MetricsSnapshot()
    snapshot2.notes = json.dumps(
        {
            "event": "overlay_budget",
            "active_overlays": 1,
            "allowed_overlays": 2,
            "reduce_motion": True,
            "disable_secondary_fps": 90,
        }
    )
    servicer.PushMetrics(snapshot2, None)

    assert len(router.messages) == 2
    assert router.messages[1].severity == "notice"
    assert router.messages[1].context["exceeded"] == "false"


def test_overlay_budget_alert_sink_ignores_duplicates():
    router = _RecorderRouter()
    sink = OverlayBudgetAlertSink(router)
    store = MetricsSnapshotStore(maxlen=4)
    servicer = MetricsServiceServicer(store, sinks=[sink])

    payload = json.dumps(
        {
            "event": "overlay_budget",
            "active_overlays": 5,
            "allowed_overlays": 3,
        }
    )
    snapshot = trading_pb2.MetricsSnapshot()
    snapshot.notes = payload
    servicer.PushMetrics(snapshot, None)

    duplicate = trading_pb2.MetricsSnapshot()
    duplicate.notes = payload
    servicer.PushMetrics(duplicate, None)

    assert len(router.messages) == 1, "Duplikaty overlay-budget nie powinny generować kolejnych alertów"

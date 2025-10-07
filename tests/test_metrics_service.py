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

from bot_core.runtime.metrics_service import MetricsServer, MetricsSnapshotStore, create_server


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

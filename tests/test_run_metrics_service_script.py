"""Testy CLI uruchamiającego MetricsService."""

from __future__ import annotations

import json
import threading
import time

import pytest

grpc = pytest.importorskip("grpc", reason="Wymaga biblioteki grpcio")
trading_pb2 = pytest.importorskip(
    "bot_core.generated.trading_pb2", reason="Brak wygenerowanych stubów trading_pb2"
)
trading_pb2_grpc = pytest.importorskip(
    "bot_core.generated.trading_pb2_grpc",
    reason="Brak wygenerowanych stubów trading_pb2_grpc",
)

from scripts import run_metrics_service


@pytest.mark.timeout(5)
def test_metrics_service_cli_creates_jsonl(tmp_path):
    jsonl_path = tmp_path / "metrics.jsonl"
    args = [
        "--host",
        "127.0.0.1",
        "--port",
        "0",
        "--jsonl",
        str(jsonl_path),
        "--shutdown-after",
        "0.3",
        "--no-log-sink",
    ]

    exit_code = run_metrics_service.main(args)
    assert exit_code == 0
    assert jsonl_path.exists()


@pytest.mark.timeout(5)
def test_metrics_service_cli_accepts_metrics(tmp_path, monkeypatch):
    jsonl_path = tmp_path / "metrics.jsonl"
    args = [
        "--host",
        "127.0.0.1",
        "--port",
        "0",
        "--jsonl",
        str(jsonl_path),
        "--log-level",
        "debug",
        "--shutdown-after",
        "0.6",
    ]

    server_holder: dict[str, object] = {}

    def fake_build_server(**kwargs):
        server = run_metrics_service.create_metrics_server(**kwargs)
        server_holder["server"] = server
        return server

    monkeypatch.setattr(run_metrics_service, "_build_server", fake_build_server)

    exit_code: list[int] = []

    def run_cli():
        exit_code.append(run_metrics_service.main(args))

    thread = threading.Thread(target=run_cli)
    thread.start()
    time.sleep(0.2)

    server = server_holder.get("server")
    assert server is not None, "CLI powinno zainicjalizować serwer"
    channel = grpc.insecure_channel(server.address)  # type: ignore[attr-defined]
    stub = trading_pb2_grpc.MetricsServiceStub(channel)
    snapshot = trading_pb2.MetricsSnapshot()
    snapshot.notes = "cli-test"
    snapshot.fps = 61.2
    stub.PushMetrics(snapshot)

    thread.join()
    assert exit_code and exit_code[0] == 0

    lines = [line for line in jsonl_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert lines, "Plik JSONL powinien zawierać wpis po PushMetrics"
    last = json.loads(lines[-1])
    assert last["notes"] == "cli-test"
    assert last["fps"] == pytest.approx(61.2)

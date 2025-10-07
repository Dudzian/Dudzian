from __future__ import annotations

import json
from typing import Iterable

import json
from typing import Iterable

import pytest


pytest.importorskip("grpc")
pytest.importorskip("bot_core.generated.trading_pb2")


from bot_core.runtime.metrics_service import MetricsServer
from scripts.watch_metrics_stream import main as watch_metrics_main


def _start_server(auth_token: str | None = None) -> MetricsServer:
    server = MetricsServer(host="127.0.0.1", port=0, sinks=(), auth_token=auth_token)
    server.start()
    return server


def _append_snapshots(server: MetricsServer, payloads: Iterable[dict[str, object]]) -> None:
    from bot_core.generated import trading_pb2

    for payload in payloads:
        snapshot = trading_pb2.MetricsSnapshot()
        snapshot.notes = json.dumps(payload, ensure_ascii=False)
        server.store.append(snapshot)


@pytest.mark.timeout(5)
def test_watch_metrics_stream_outputs_json(tmp_path, capsys):
    server = _start_server()
    try:
        _append_snapshots(
            server,
            [
                {"event": "reduce_motion", "active": True, "window_count": 2},
                {"event": "reduce_motion", "active": False, "window_count": 1},
            ],
        )

        host, port = server.address.split(":")
        exit_code = watch_metrics_main(
            [
                "--host",
                host,
                "--port",
                port,
                "--limit",
                "1",
                "--format",
                "json",
                "--event",
                "reduce_motion",
            ]
        )
        captured = capsys.readouterr()
        assert exit_code == 0
        assert "reduce_motion" in captured.out
        assert "window_count" in captured.out
    finally:
        server.stop(0)


@pytest.mark.timeout(5)
def test_watch_metrics_stream_table_format(tmp_path, capsys):
    server = _start_server()
    try:
        _append_snapshots(
            server,
            [
                {"event": "overlay_budget", "over": True, "active_overlays": 5},
            ],
        )

        host, port = server.address.split(":")
        exit_code = watch_metrics_main(
            [
                "--host",
                host,
                "--port",
                port,
                "--limit",
                "1",
                "--format",
                "table",
            ]
        )
        captured = capsys.readouterr()
        assert exit_code == 0
        assert "overlay_budget" in captured.out
    finally:
        server.stop(0)


@pytest.mark.timeout(5)
def test_watch_metrics_stream_with_auth_token(capsys):
    token = "stream-secret"
    server = _start_server(auth_token=token)
    try:
        _append_snapshots(
            server,
            [
                {"event": "reduce_motion", "active": True},
            ],
        )

        host, port = server.address.split(":")
        exit_code = watch_metrics_main(
            [
                "--host",
                host,
                "--port",
                port,
                "--limit",
                "1",
                "--auth-token",
                token,
            ]
        )
        captured = capsys.readouterr()
        assert exit_code == 0
        assert "reduce_motion" in captured.out
    finally:
        server.stop(0)

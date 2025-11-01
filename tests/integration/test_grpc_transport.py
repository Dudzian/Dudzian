from __future__ import annotations

import io
import logging
import json
import os
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml

from bot_core.runtime import local_gateway
from bot_core.api.server import LocalRuntimeContext, LocalRuntimeGateway
from bot_core.exchanges import streaming as core_streaming
from bot_core.exchanges.base import AccountSnapshot
from bot_core.execution.paper import MarketMetadata
from bot_core.runtime.local_gateway import JsonRpcServer
from KryptoLowca.exchanges import interfaces as exchange_interfaces
from KryptoLowca.exchanges import streaming as exchange_streaming


def _build_stub_context():
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
    context.metrics_registry = None
    context.risk_store = None
    context.risk_builder = None
    context.risk_publisher = None
    context.marketplace_repository = None
    context.marketplace_enabled = False
    return context


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


def test_local_runtime_gateway_stream_snapshot_and_updates() -> None:
    context = _build_stub_context()
    gateway = LocalRuntimeGateway(context)
    payload = gateway.dispatch(
        "market_data.stream_ohlcv",
        {"symbol": "BTC/USDT", "timeout_ms": 100, "max_updates": 2},
    )
    assert payload["snapshot"]
    assert len(payload["updates"]) <= 2
    assert payload["has_more"] is False


def test_local_runtime_gateway_stream_timeout_without_iterator_support(monkeypatch) -> None:
    context = _build_stub_context()
    gateway = LocalRuntimeGateway(context)

    original_stream = gateway._market.StreamOhlcv

    class _NoTimeoutStream:
        def __init__(self, iterable):
            self._iterator = iter(iterable)

        def __iter__(self):
            return self

        def __next__(self):
            return next(self._iterator)

        def next(self):  # noqa: A003 - zgodne z interfejsem gRPC
            return next(self)

        def cancel(self) -> None:
            closer = getattr(self._iterator, "close", None)
            if callable(closer):
                closer()

    def patched_stream(request, context):
        return _NoTimeoutStream(original_stream(request, context))

    monkeypatch.setattr(gateway._market, "StreamOhlcv", patched_stream)

    payload = gateway.dispatch(
        "market_data.stream_ohlcv",
        {"symbol": "BTC/USDT", "timeout_ms": 50, "max_updates": 1},
    )
    assert payload["snapshot"]
    assert payload["has_more"] is False


def test_local_runtime_gateway_stream_continuous_subscription() -> None:
    context = _build_stub_context()
    gateway = LocalRuntimeGateway(context)
    initial = gateway.dispatch(
        "market_data.stream_ohlcv",
        {"symbol": "BTC/USDT", "continuous": True, "max_updates": 1, "timeout_ms": 50},
    )
    assert initial["snapshot"]
    assert "subscription_id" in initial
    subscription_id = initial["subscription_id"]
    follow_up = gateway.dispatch(
        "market_data.stream_ohlcv",
        {"subscription_id": subscription_id, "timeout_ms": 10, "max_updates": 1},
    )
    assert follow_up["subscription_id"] == subscription_id
    assert "has_more" in follow_up
    cancelled = gateway.dispatch(
        "market_data.stream_ohlcv",
        {"subscription_id": subscription_id, "cancel": True},
    )
    assert cancelled["cancelled"] is True
    assert cancelled["subscription_id"] == subscription_id


def test_local_runtime_gateway_cleans_stale_subscription() -> None:
    context = _build_stub_context()
    gateway = LocalRuntimeGateway(context, subscription_ttl=0.05)
    initial = gateway.dispatch(
        "market_data.stream_ohlcv",
        {
            "symbol": "BTC/USDT",
            "continuous": True,
            "max_updates": 0,
            "timeout_ms": 10,
        },
    )
    assert initial["snapshot"]
    subscription_id = initial["subscription_id"]

    time.sleep(0.1)

    cancelled = gateway.dispatch(
        "market_data.stream_ohlcv",
        {
            "subscription_id": subscription_id,
            "cancel": True,
        },
    )
    assert cancelled["cancelled"] is True
    with gateway._subscriptions_lock:
        assert subscription_id not in gateway._subscriptions


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
    assert "websocket" not in (exchange_streaming.LongPollSubscription.__doc__ or "").lower()
    assert not hasattr(core_streaming, "LocalWebSocketBridge")
    assert not hasattr(core_streaming.LocalLongPollStream, "websocket_bridge")


class _FakeGateway:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def dispatch(self, method: str, params):  # pragma: no cover - exercised via JsonRpcServer
        self.calls.append(method)
        if method == "slow":
            delay_ms = float(params.get("delay_ms", 0.0))
            time.sleep(delay_ms / 1000.0)
            return {"delay_ms": delay_ms}
        if method == "boom":
            raise RuntimeError("boom")
        return {"ok": True}


@pytest.mark.skipif(sys.platform == "win32", reason="StdIn selectors nie są wspierane na Windows")
def test_stdin_poller_prefers_selector_backend() -> None:
    stop_event = threading.Event()
    read_fd, write_fd = os.pipe()
    reader = os.fdopen(read_fd, "r", buffering=1)
    writer = os.fdopen(write_fd, "w", buffering=1)
    try:
        poller = local_gateway._StdinPoller(reader, 0.001, stop_event)
        try:
            assert poller.backend == "selector"
        finally:
            poller.close()
    finally:
        reader.close()
        writer.close()


def test_stdin_poller_falls_back_to_thread_when_selector_fails(monkeypatch) -> None:
    class _BrokenSelector(local_gateway.selectors.BaseSelector):
        def __init__(self) -> None:  # pragma: no cover - konfiguracja testowa
            self._map: dict[Any, Any] = {}

        def register(self, fileobj, events, data=None):  # pragma: no cover - testowy failure
            raise ValueError("boom")

        def unregister(self, fileobj):  # pragma: no cover - wymagane przez interfejs
            return None

        def select(self, timeout=None):  # pragma: no cover - fallback nie powinien wywoływać
            raise AssertionError("select() should not be called when selector registration fails")

        def close(self):  # pragma: no cover - wymagane przez interfejs
            return None

        def get_map(self):  # pragma: no cover - wymagane przez bazę
            return self._map

    monkeypatch.setattr(local_gateway.selectors, "DefaultSelector", lambda: _BrokenSelector())

    stop_event = threading.Event()
    fake_stdin = io.StringIO(json.dumps({"id": "noop", "method": "slow", "params": {"delay_ms": 0}}) + "\n")
    poller = local_gateway._StdinPoller(fake_stdin, 0.001, stop_event)
    try:
        assert poller.backend == "thread"
        line = poller.poll()
        assert line is not None
        assert "noop" in line
    finally:
        poller.close()

    gateway = _FakeGateway()
    emitted: list[dict] = []
    server = JsonRpcServer(gateway, max_workers=1, poll_interval=0.001, emit=emitted.append)

    monkeypatch.setattr(
        sys,
        "stdin",
        io.StringIO(json.dumps({"id": "req", "method": "slow", "params": {"delay_ms": 0}}) + "\n"),
    )

    thread = threading.Thread(target=server.run, name="json-rpc-runner")
    thread.start()
    thread.join(timeout=1.0)

    try:
        assert not thread.is_alive(), "JsonRpcServer.run() nie zakończył pracy w trybie fallback"
    finally:
        server.stop()
        thread.join(timeout=0.5)
        fake_stdin.close()

    assert any(entry.get("id") == "req" for entry in emitted)


def test_json_rpc_server_executes_requests_in_parallel() -> None:
    gateway = _FakeGateway()
    emitted: list[dict] = []
    server = JsonRpcServer(gateway, max_workers=2, poll_interval=0.005, emit=emitted.append)

    start = time.monotonic()
    server.submit({"id": "a", "method": "slow", "params": {"delay_ms": 120}})
    server.submit({"id": "b", "method": "slow", "params": {"delay_ms": 120}})

    deadline = start + 1.5
    while len(emitted) < 2 and time.monotonic() < deadline:
        time.sleep(0.01)
        server.flush_ready()

    elapsed = time.monotonic() - start
    assert len(emitted) == 2
    identifiers = {entry["id"] for entry in emitted}
    assert identifiers == {"a", "b"}
    assert elapsed < 0.35  # równoległe wykonanie skraca czas z ~0.24 do <0.35s
    server.stop()


def test_json_rpc_server_emits_timeout_and_cancel() -> None:
    gateway = _FakeGateway()
    emitted: list[dict] = []
    server = JsonRpcServer(gateway, max_workers=1, poll_interval=0.005, emit=emitted.append)

    server.submit(
        {
            "id": "timeout",
            "method": "slow",
            "params": {"delay_ms": 200},
            "timeout_ms": 50,
        }
    )

    deadline = time.monotonic() + 1.0
    while not emitted and time.monotonic() < deadline:
        time.sleep(0.01)
        server.enforce_timeouts()
        server.flush_ready()

    assert emitted
    first = emitted[0]
    assert first["id"] == "timeout"
    assert first["error"]["message"] == "timeout"
    assert first.get("timeout") is True

    # re-submit and cancel explicite
    emitted.clear()
    server.flush_ready()
    server.submit({"id": "cancel-me", "method": "slow", "params": {"delay_ms": 200}})
    server.submit({"id": "cancel-me", "cancel": True})

    deadline = time.monotonic() + 1.0
    while not emitted and time.monotonic() < deadline:
        time.sleep(0.01)
        server.flush_ready()

    server.flush_ready()
    assert emitted
    response = emitted[0]
    assert response["id"] == "cancel-me"
    assert response.get("cancelled") is True
    server.stop()


def test_json_rpc_server_reports_errors() -> None:
    gateway = _FakeGateway()
    emitted: list[dict] = []
    server = JsonRpcServer(gateway, max_workers=1, poll_interval=0.005, emit=emitted.append)

    server.submit({"id": "boom", "method": "boom", "params": {}})

    deadline = time.monotonic() + 1.0
    while not emitted and time.monotonic() < deadline:
        time.sleep(0.01)
        server.flush_ready()

    assert emitted
    response = emitted[0]
    assert response["id"] == "boom"
    assert "error" in response
    assert response["error"]["message"] == "boom"
    server.stop()


def test_json_rpc_server_applies_queue_limits(caplog: pytest.LogCaptureFixture) -> None:
    gateway = _FakeGateway()
    emitted: list[dict] = []
    server = JsonRpcServer(
        gateway,
        max_workers=1,
        max_queue_size=2,
        poll_interval=0.001,
        emit=emitted.append,
    )

    with caplog.at_level(logging.WARNING):
        server.submit({"id": "req-1", "method": "slow", "params": {"delay_ms": 150}})
        server.submit({"id": "req-2", "method": "slow", "params": {"delay_ms": 150}})
        server.submit({"id": "req-overflow", "method": "slow", "params": {"delay_ms": 150}})

    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        server.flush_ready()
        if any(entry.get("id") == "req-overflow" for entry in emitted):
            break
        time.sleep(0.01)

    overflow = next(
        (entry for entry in emitted if entry.get("id") == "req-overflow"),
        None,
    )
    assert overflow is not None, f"brak odpowiedzi overflow, posiadane: {emitted!r}"
    assert overflow["error"]["message"] == "server-busy"
    assert any("capacity limit" in record.message for record in caplog.records)

    deadline = time.monotonic() + 2.0
    while len([entry for entry in emitted if entry.get("id") in {"req-1", "req-2"}]) < 2 and time.monotonic() < deadline:
        time.sleep(0.01)
        server.flush_ready()

    server.stop()


def test_json_rpc_server_preserves_json_frames_on_overflow() -> None:
    gateway = _FakeGateway()
    buffer = io.StringIO()

    def emit(payload):
        buffer.write(json.dumps(payload) + "\n")

    server = JsonRpcServer(
        gateway,
        max_workers=1,
        max_queue_size=1,
        poll_interval=0.001,
        emit=emit,
    )

    server._results.put_nowait({"id": "seed", "result": {"ok": True}})

    payloads = [
        {"id": f"overflow-{index}", "result": {"index": index}}
        for index in range(3)
    ]

    threads = [
        threading.Thread(target=server._deliver, args=(payload,))
        for payload in payloads
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    server.flush_ready()

    lines = [line for line in buffer.getvalue().splitlines() if line]
    assert len(lines) == 1 + len(payloads)

    parsed = [json.loads(line) for line in lines]
    identifiers = {entry.get("id") for entry in parsed}
    expected = {"seed", *[payload["id"] for payload in payloads]}
    assert identifiers == expected

    server.stop()

def test_json_rpc_server_flushes_responses_on_shutdown(monkeypatch) -> None:
    gateway = _FakeGateway()
    emitted: list[dict] = []
    server = JsonRpcServer(gateway, max_workers=2, poll_interval=0.005, emit=emitted.append)

    read_fd, write_fd = os.pipe()
    reader = os.fdopen(read_fd, "r", buffering=1)
    writer = os.fdopen(write_fd, "w", buffering=1)

    original_stdin = sys.stdin
    monkeypatch.setattr(sys, "stdin", reader)

    thread = threading.Thread(target=server.run, name="json-rpc-runner")
    thread.start()

    try:
        requests = [
            {"id": "slow-1", "method": "slow", "params": {"delay_ms": 120}},
            {"id": "slow-2", "method": "slow", "params": {"delay_ms": 160}},
        ]
        for payload in requests:
            writer.write(json.dumps(payload) + "\n")
            writer.flush()
        writer.close()

        thread.join(timeout=5.0)
        if thread.is_alive():
            server.stop()
            thread.join(timeout=1.0)
            pytest.fail("JsonRpcServer.run() did not terminate after shutdown request")
    finally:
        writer.close()
        sys.stdin = original_stdin
        reader.close()

    server.flush_ready()

    identifiers = [entry.get("id") for entry in emitted if "id" in entry]
    assert {"slow-1", "slow-2"}.issubset(set(identifiers))
    assert any(entry.get("event") == "runtime-stopping" for entry in emitted)
    assert gateway.calls.count("slow") == 2

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any

import pytest

from ui.backend.grpc_decision_stream_client import GrpcDecisionStreamClient


@dataclass
class _Entry:
    fields: dict[str, str]


class _SnapshotPayload:
    def __init__(self, records: list[dict[str, str]]) -> None:
        self.records = [_Entry(record) for record in records]


class _IncrementPayload:
    def __init__(self, record: dict[str, str]) -> None:
        self.record = _Entry(record)


class _Update:
    def __init__(self, *, snapshot: list[dict[str, str]] | None = None, increment=None) -> None:
        self.snapshot = _SnapshotPayload(snapshot or [])
        self.increment = _IncrementPayload(increment or {})
        self.cycle_metrics = None
        self._kind = "snapshot" if snapshot is not None else "increment"

    def HasField(self, name: str) -> bool:
        return name == self._kind


class _StreamController:
    def __init__(self, updates: list[_Update], *, fail: Exception | None = None, block: bool = False):
        self._updates = list(updates)
        self._fail = fail
        self._block = block
        self._stop_event: threading.Event | None = None

    def bind_stop_event(self, event: threading.Event) -> None:
        self._stop_event = event

    def __iter__(self):
        if self._fail is not None:
            raise self._fail
        for update in self._updates:
            yield update
        if self._block:
            while self._stop_event is not None and not self._stop_event.is_set():
                time.sleep(0.01)


class _Stub:
    def __init__(self, stream_factory):
        self._stream_factory = stream_factory
        self.calls: list[dict[str, Any]] = []

    def StreamDecisions(self, request, metadata=None):
        self.calls.append({"request": request, "metadata": metadata})
        return self._stream_factory()


class _TradingPb2:
    class StreamDecisionsRequest:
        def __init__(self, **kwargs):
            self.payload = kwargs


class _TradingPb2Grpc:
    def __init__(self, stub: _Stub):
        self._stub = stub

    def RuntimeServiceStub(self, channel):
        return self._stub


class _ReadyFuture:
    def __init__(self, on_wait=None):
        self._on_wait = on_wait

    def result(self, timeout):
        if callable(self._on_wait):
            self._on_wait(timeout)
        return None


class _Channel:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


class _Grpc:
    def __init__(self):
        self.insecure_calls: list[str] = []
        self.secure_calls: list[dict[str, Any]] = []
        self.on_ready_wait = None

    def insecure_channel(self, target):
        self.insecure_calls.append(target)
        return _Channel()

    def secure_channel(self, target, credentials, options=None):
        self.secure_calls.append(
            {"target": target, "credentials": credentials, "options": options}
        )
        return _Channel()

    def channel_ready_future(self, channel):
        return _ReadyFuture(self.on_ready_wait)


def _drain_kinds(client: GrpcDecisionStreamClient, *, timeout: float = 1.0) -> list[str]:
    deadline = time.monotonic() + timeout
    kinds: list[str] = []
    while time.monotonic() < deadline:
        try:
            kind, _ = client.events_queue.get(timeout=0.05)
        except Exception:
            continue
        kinds.append(kind)
        client.events_queue.task_done()
        if kind == "done":
            break
    return kinds


def _build_client(
    *,
    grpc_module: _Grpc,
    stub: _Stub,
    ssl_credentials: object | None = None,
    authority_override: str | None = None,
    metadata: tuple[tuple[str, str], ...] = (),
    retry_base: float = 0.1,
    retry_multiplier: float = 2.0,
    retry_max: float = 0.3,
    stubs_loader=None,
) -> GrpcDecisionStreamClient:
    if stubs_loader is None:
        stubs_loader = lambda: (_TradingPb2, _TradingPb2Grpc(stub))
    client = GrpcDecisionStreamClient(
        target="localhost:50051",
        metadata=metadata,
        ssl_credentials=ssl_credentials,
        authority_override=authority_override,
        limit=5,
        ready_timeout=1.0,
        retry_base=retry_base,
        retry_multiplier=retry_multiplier,
        retry_max=retry_max,
        cycle_metrics_serializer=lambda *_args, **_kwargs: {},
        grpc_module=grpc_module,
        stubs_loader=stubs_loader,
    )
    return client


def test_emits_snapshot_increment_stream_ended_and_done() -> None:
    stream = _StreamController(
        updates=[
            _Update(snapshot=[{"event": "snapshot_event"}]),
            _Update(increment={"event": "increment_event"}),
        ]
    )
    stub = _Stub(lambda: stream)
    grpc_module = _Grpc()
    client = _build_client(grpc_module=grpc_module, stub=stub)

    client.start()
    kinds: list[str] = []
    deadline = time.monotonic() + 1.0
    while time.monotonic() < deadline and len(kinds) < 4:
        kind, _ = client.events_queue.get(timeout=0.2)
        kinds.append(kind)
        client.events_queue.task_done()
    client.stop()
    kinds.extend(_drain_kinds(client, timeout=0.7))

    assert kinds[:4] == ["connected", "snapshot", "increment", "stream-ended"]
    assert "done" in kinds


def test_stop_is_idempotent_and_emits_done_on_shutdown() -> None:
    stream = _StreamController(updates=[], block=True)
    stub = _Stub(lambda: stream)
    grpc_module = _Grpc()
    client = _build_client(grpc_module=grpc_module, stub=stub)
    stream.bind_stop_event(client.stop_event)

    client.start()
    time.sleep(0.05)
    client.stop()
    client.stop()

    kinds = _drain_kinds(client, timeout=0.5)
    assert "done" in kinds
    assert client.thread is None


def test_stop_before_connection_ready_finishes_cleanly_and_done_present() -> None:
    stub = _Stub(lambda: _StreamController(updates=[]))
    grpc_module = _Grpc()

    def _wait(_timeout):
        time.sleep(0.2)

    grpc_module.on_ready_wait = _wait
    client = _build_client(grpc_module=grpc_module, stub=stub)

    client.start()
    time.sleep(0.02)
    client.stop()

    kinds = _drain_kinds(client, timeout=0.7)
    assert "done" in kinds


def test_backoff_progression_and_cap() -> None:
    class _FailingEveryTime:
        def StreamDecisions(self, request, metadata=None):
            raise RuntimeError("unavailable")

    class _Pb2Grpc:
        def RuntimeServiceStub(self, channel):
            return _FailingEveryTime()

    grpc_module = _Grpc()
    client = _build_client(
        grpc_module=grpc_module,
        stub=_Stub(lambda: _StreamController(updates=[])),
        retry_base=0.1,
        retry_multiplier=2.0,
        retry_max=0.3,
        stubs_loader=lambda: (_TradingPb2, _Pb2Grpc()),
    )

    client.start()
    sleeps: list[float] = []
    deadline = time.monotonic() + 1.2
    while time.monotonic() < deadline and len(sleeps) < 3:
        try:
            kind, payload = client.events_queue.get(timeout=0.2)
        except Exception:
            continue
        if kind == "retrying":
            sleeps.append(float((payload or {}).get("sleep", 0.0)))
        client.events_queue.task_done()
    client.stop()

    assert len(sleeps) >= 3
    assert sleeps[:3] == pytest.approx([0.1, 0.2, 0.3], abs=1e-6)


def test_secure_channel_path_and_authority_override_and_metadata() -> None:
    stream = _StreamController(updates=[])
    stub = _Stub(lambda: stream)
    grpc_module = _Grpc()
    metadata = (("authorization", "token"),)
    client = _build_client(
        grpc_module=grpc_module,
        stub=stub,
        ssl_credentials=object(),
        authority_override="runtime.internal",
        metadata=metadata,
    )

    client.start()
    kinds = _drain_kinds(client)
    client.stop()

    assert "connected" in kinds
    assert grpc_module.secure_calls
    assert not grpc_module.insecure_calls
    assert grpc_module.secure_calls[0]["options"] == [
        ("grpc.ssl_target_name_override", "runtime.internal")
    ]
    assert stub.calls[0]["metadata"] == metadata


def test_insecure_channel_path_without_tls() -> None:
    stream = _StreamController(updates=[])
    stub = _Stub(lambda: stream)
    grpc_module = _Grpc()
    client = _build_client(grpc_module=grpc_module, stub=stub)

    client.start()
    _drain_kinds(client)
    client.stop()

    assert grpc_module.insecure_calls
    assert all(target == "localhost:50051" for target in grpc_module.insecure_calls)
    assert not grpc_module.secure_calls


def test_stub_loader_error_is_reported_as_connection_error_and_done() -> None:
    grpc_module = _Grpc()
    client = _build_client(
        grpc_module=grpc_module,
        stub=_Stub(lambda: _StreamController(updates=[])),
        stubs_loader=lambda: (_ for _ in ()).throw(ImportError("missing stubs")),
    )

    client.start()
    kinds = _drain_kinds(client)
    client.stop()

    assert kinds[0] == "connection-error"
    assert "done" in kinds

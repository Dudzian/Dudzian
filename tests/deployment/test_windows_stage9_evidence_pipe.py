"""Emulated native adapters; these tests do not qualify Windows pipe security."""

from __future__ import annotations

import copy
import json
import os
from contextlib import contextmanager
from datetime import datetime, timezone
from threading import Event
from types import SimpleNamespace
from typing import Any, Iterator

import pytest

from deployment import windows_stage9_evidence_pipe as pipe

NAME = pipe.SERVICES[0]
IMAGE = rf"C:\Program Files\CryptoHunter\{NAME}.exe"
SID = "S-1-5-80-111"
STARTED = datetime(2026, 10, 17, tzinfo=timezone.utc)


class Handle:
    def __init__(self) -> None:
        self.closed = False

    def Close(self) -> None:
        self.closed = True


class NativeError(Exception):
    def __init__(self, code: int) -> None:
        super().__init__(code)
        self.winerror = code


def request() -> dict[str, Any]:
    return {
        "version": 1,
        "nonce": "a" * 64,
        "collection_id": "b" * 64,
        "issued": 100.0,
        "expires": 130.0,
    }


@pytest.fixture
def native(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    state: dict[str, Any] = {
        "process_id": 123,
        "state": 4,
        "account": rf"NT SERVICE\{NAME}",
        "command_line": f'"{IMAGE}"',
        "service_type": 16,
        "token_sid": SID,
        "process_path": IMAGE,
        "process_started": STARTED,
        "alive": 258,
    }
    opened: list[tuple[Any, ...]] = []
    process, token, endpoint = Handle(), Handle(), Handle()
    scm_closed: list[Any] = []
    w = SimpleNamespace(
        api=SimpleNamespace(OpenProcess=lambda *args: (opened.append(args), process)[1]),
        event=SimpleNamespace(WaitForSingleObject=lambda *_: state["alive"]),
        process=SimpleNamespace(
            QueryFullProcessImageName=lambda *_: state["process_path"],
            GetProcessTimes=lambda *_: {"CreationTime": state["process_started"]},
        ),
        security=SimpleNamespace(
            OpenProcessToken=lambda *_: token,
            LookupAccountName=lambda *_: (SID, None, None),
            ConvertSidToStringSid=str,
            GetTokenInformation=lambda *_: (state["token_sid"], 0),
        ),
        service=SimpleNamespace(
            OpenSCManager=lambda *_: "manager",
            OpenService=lambda *_: "service",
            QueryServiceStatusEx=lambda *_: {
                "ProcessId": state["process_id"],
                "CurrentState": state["state"],
            },
            QueryServiceConfig=lambda *_: (
                state["service_type"],
                0,
                0,
                state["command_line"],
                0,
                0,
                0,
                state["account"],
            ),
            CloseServiceHandle=scm_closed.append,
        ),
        file=SimpleNamespace(CreateFile=lambda *_: endpoint),
        pipe=SimpleNamespace(SetNamedPipeHandleState=lambda *_: None),
    )
    observation = {
        "paths": {"program_files": r"C:\Program Files\CryptoHunter"},
        "services": [
            {
                "name": NAME,
                "process_id": 123,
                "second_process_id": 123,
                "state": "Running",
                "second_state": 4,
                "account": state["account"],
                "command_line": state["command_line"],
                "process_path": IMAGE,
                "process_started": STARTED.isoformat(),
            }
        ],
    }
    monkeypatch.setattr(pipe, "_win32", lambda: w)
    monkeypatch.setattr(pipe, "_canonical_image", lambda _: IMAGE)
    monkeypatch.setattr(pipe, "_pipe_pid", lambda *_, **__: 123)
    return SimpleNamespace(
        w=w,
        state=state,
        opened=opened,
        process=process,
        token=token,
        endpoint=endpoint,
        scm_closed=scm_closed,
        observation=observation,
    )


def test_pins_exact_service_primary_token_with_liveness_rights(native: SimpleNamespace) -> None:
    pinned = pipe._PinnedService(NAME, native.observation)
    assert native.opened == [(0x101000, False, 123)]
    assert pinned.identity == {
        "name": NAME,
        "process_id": 123,
        "token_sid": SID,
        "process_path": IMAGE,
        "process_started": STARTED.isoformat(),
    }
    pinned.assert_current()
    pinned.close()
    assert native.process.closed and native.token.closed
    assert native.scm_closed == ["service", "manager"]


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("account", "LocalSystem"),
        ("state", 1),
        ("service_type", 32),
        ("process_id", 999),
        ("alive", 0),
        ("token_sid", "S-1-5-18"),
        ("command_line", f'"{IMAGE}" --borrow-token'),
        ("process_path", r"C:\Temp\copy.exe"),
        ("process_started", datetime(2026, 10, 18, tzinfo=timezone.utc)),
    ],
)
def test_rechecks_scm_pid_config_token_creation_and_liveness(
    native: SimpleNamespace, key: str, value: Any
) -> None:
    pinned = pipe._PinnedService(NAME, native.observation)
    native.state[key] = value
    try:
        with pytest.raises(pipe.EvidencePipeError):
            pinned.assert_current()
    finally:
        pinned.close()


@pytest.mark.parametrize(
    "key", ["second_process_id", "state", "account", "process_path", "process_started"]
)
def test_rejects_mismatching_inspection_and_closes_handles(
    native: SimpleNamespace, key: str
) -> None:
    native.observation["services"][0][key] = "untrusted"
    with pytest.raises((pipe.EvidencePipeError, ValueError)):
        pipe._PinnedService(NAME, native.observation)
    assert native.process.closed and native.token.closed
    assert native.scm_closed == ["service", "manager"]


def test_receipt_reverification_rejects_restarted_identity(native: SimpleNamespace) -> None:
    identity = {
        "name": NAME,
        "process_id": 123,
        "token_sid": SID,
        "process_path": IMAGE,
        "process_started": "2026-10-16T00:00:00+00:00",
    }
    with pytest.raises(pipe.EvidencePipeError, match="restarted"):
        pipe.verify_service(NAME, native.observation, identity)
    assert native.process.closed and native.token.closed


def test_all_native_handles_attempted_when_individual_closes_fail(native: SimpleNamespace) -> None:
    pinned = pipe._PinnedService(NAME, native.observation)
    attempted: list[Any] = []

    def fail_token() -> None:
        attempted.append("token")
        raise NativeError(6)

    def fail_service(handle: Any) -> None:
        attempted.append(handle)
        if handle == "service":
            raise NativeError(6)

    native.token.Close = fail_token
    native.w.service.CloseServiceHandle = fail_service
    with pytest.raises(pipe.EvidencePipeError, match="handle cleanup"):
        pinned.close()
    assert native.process.closed
    assert attempted == ["token", "service", "manager"]
    pinned.close()


def test_pipe_close_failure_still_closes_process_token_and_scm(native: SimpleNamespace) -> None:
    def fail() -> None:
        raise NativeError(6)

    native.endpoint.Close = fail
    with pytest.raises(pipe.EvidencePipeError, match="endpoint cleanup"):
        with pipe.connect_service(NAME, native.observation):
            pass
    assert native.process.closed and native.token.closed
    assert native.scm_closed == ["service", "manager"]


def test_connect_uses_identification_sqos_and_never_pipe_creation_rights(
    native: SimpleNamespace,
) -> None:
    arguments: list[tuple[Any, ...]] = []
    native.w.file.CreateFile = lambda *args: (arguments.append(args), native.endpoint)[1]
    with pipe.connect_service(NAME, native.observation) as peer:
        assert peer.identity["token_sid"] == SID
    assert arguments[0][0] == pipe.pipe_name(NAME)
    assert arguments[0][1] == pipe.CLIENT_RIGHTS
    assert not arguments[0][1] & 4
    assert arguments[0][5] == 0x110000
    assert native.endpoint.closed and native.process.closed and native.token.closed


def test_connect_rejects_pipe_squatter_pid(
    native: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(pipe, "_pipe_pid", lambda *_, **__: 999)
    with pytest.raises(pipe.EvidencePipeError, match="pipe server"):
        with pipe.connect_service(NAME, native.observation):
            pytest.fail("a different process must never receive the challenge")
    assert native.endpoint.closed and native.process.closed


@pytest.mark.parametrize(
    "frame",
    [
        b'{"x":1,"x":2}',
        b'{"x":{"y":1,"y":2}}',
        b'{"x":NaN}',
        b'{"x":Infinity}',
        b'{"x":1e999}',
        b"[]",
        b"null",
        b'{"x":"\xff"}',
        b"{",
        b"",
        b"x" * (pipe.MAX_FRAME + 1),
    ],
)
def test_rejects_ambiguous_nonfinite_and_oversized_json(frame: bytes) -> None:
    with pytest.raises(pipe.EvidencePipeError):
        pipe._decode(frame)


@pytest.mark.parametrize(
    "payload",
    [{"x": float("nan")}, {"x": float("inf")}, {"x": object()}, {"x": "x" * pipe.MAX_FRAME}],
)
def test_rejects_nonserializable_output(payload: dict[str, Any]) -> None:
    with pytest.raises(pipe.EvidencePipeError):
        pipe._encode(payload)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("version", True),
        ("nonce", "A" * 64),
        ("collection_id", "short"),
        ("issued", float("nan")),
        ("issued", 111),
        ("expires", 110),
        ("expires", 131),
        ("expires", True),
        ("extra", "x"),
    ],
)
def test_rejects_invalid_challenge_and_deadlines(
    monkeypatch: pytest.MonkeyPatch, key: str, value: Any
) -> None:
    monkeypatch.setattr(pipe.time, "monotonic", lambda: 110.0)
    payload = request()
    payload[key] = value
    with pytest.raises(pipe.EvidencePipeError):
        pipe._request_deadline(payload)


def test_peer_requires_successful_cleanup_completion(
    native: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(pipe.time, "monotonic", lambda: 110.0)
    monkeypatch.setattr(pipe, "_pause", lambda *_: None)
    responses = iter([{"result": "diagnostic"}, {"nonce": "a" * 64, "ack": True, "closed": True}])
    sent: list[dict[str, Any]] = []
    native.w.file.ReadFile = lambda *_: (0, json.dumps(next(responses)).encode())
    native.w.file.WriteFile = lambda _, value: (sent.append(json.loads(value)), (0, len(value)))[1]
    with pipe.connect_service(NAME, native.observation) as peer:
        assert peer.request(request()) == {"result": "diagnostic"}
        peer.finish("a" * 64)
        with pytest.raises(pipe.EvidencePipeError):
            peer.request(request())
    assert sent[-1] == {"nonce": "a" * 64, "ack": True}


@pytest.mark.parametrize(
    "completion",
    [
        {"nonce": "c" * 64, "ack": True, "closed": True},
        {"nonce": "a" * 64, "ack": True},
        {"nonce": "a" * 64, "ack": 1, "closed": True},
    ],
)
def test_peer_rejects_missing_or_forged_cleanup_completion(
    native: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, completion: dict[str, Any]
) -> None:
    monkeypatch.setattr(pipe.time, "monotonic", lambda: 110.0)
    monkeypatch.setattr(pipe, "_pause", lambda *_: None)
    responses = iter([{}, completion])
    native.w.file.ReadFile = lambda *_: (0, json.dumps(next(responses)).encode())
    native.w.file.WriteFile = lambda _, value: (0, len(value))
    with pipe.connect_service(NAME, native.observation) as peer:
        peer.request(request())
        with pytest.raises(pipe.EvidencePipeError, match="cleanup"):
            peer.finish("a" * 64)


@pytest.mark.parametrize(
    ("sid", "groups", "level", "allowed"),
    [
        ("S-1-5-18", [], 1, True),
        ("S-1-5-21-123", [("S-1-5-32-544", 4)], 1, True),
        ("S-1-5-21-123", [("S-1-5-32-544", 16)], 1, False),
        ("S-1-5-21-123", [("S-1-5-32-544", 20)], 1, False),
        ("S-1-5-21-123", [], 1, False),
        ("S-1-5-18", [], 2, False),
    ],
)
def test_authorization_reverts_identification_before_return(
    monkeypatch: pytest.MonkeyPatch, sid: str, groups: list[Any], level: int, allowed: bool
) -> None:
    events: list[str] = []
    token = Handle()
    w = SimpleNamespace(
        api=SimpleNamespace(GetCurrentThread=lambda: 1),
        pipe=SimpleNamespace(ImpersonateNamedPipeClient=lambda _: events.append("identify")),
        security=SimpleNamespace(
            OpenThreadToken=lambda *_: token,
            ConvertSidToStringSid=str,
            GetTokenInformation=lambda _, kind: {1: (sid, 0), 2: groups, 9: level}[kind],
            RevertToSelf=lambda: events.append("revert"),
        ),
    )
    monkeypatch.setattr(pipe, "_pipe_pid", lambda *_, **__: 321)
    if allowed:
        pipe._authorize(w, Handle())
    else:
        with pytest.raises(pipe.EvidencePipeError):
            pipe._authorize(w, Handle())
    assert events == ["identify", "revert"]
    assert token.closed


def test_failed_reversion_is_fatal(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail() -> None:
        raise NativeError(5)

    token = Handle()
    w = SimpleNamespace(
        api=SimpleNamespace(GetCurrentThread=lambda: 1),
        pipe=SimpleNamespace(ImpersonateNamedPipeClient=lambda _: None),
        security=SimpleNamespace(
            OpenThreadToken=lambda *_: token,
            ConvertSidToStringSid=str,
            GetTokenInformation=lambda _, kind: {1: ("S-1-5-18", 0), 2: [], 9: 1}[kind],
            RevertToSelf=fail,
        ),
    )
    monkeypatch.setattr(pipe, "_pipe_pid", lambda *_, **__: 321)
    with pytest.raises(pipe._RevertError):
        pipe._authorize(w, Handle())
    assert token.closed


def test_token_close_failure_cannot_hide_failed_reversion(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail() -> None:
        raise NativeError(5)

    w = SimpleNamespace(
        api=SimpleNamespace(GetCurrentThread=lambda: 1),
        pipe=SimpleNamespace(ImpersonateNamedPipeClient=lambda _: None),
        security=SimpleNamespace(
            OpenThreadToken=lambda *_: SimpleNamespace(Close=fail),
            ConvertSidToStringSid=str,
            GetTokenInformation=lambda _, kind: {1: ("S-1-5-18", 0), 2: [], 9: 1}[kind],
            RevertToSelf=fail,
        ),
    )
    monkeypatch.setattr(pipe, "_pipe_pid", lambda *_, **__: 321)
    with pytest.raises(pipe._RevertError):
        pipe._authorize(w, Handle())


def test_protected_dacl_has_only_reviewed_principals_and_client_masks() -> None:
    seen: list[str] = []
    w = SimpleNamespace(
        security=SimpleNamespace(
            ConvertStringSecurityDescriptorToSecurityDescriptor=lambda value, _: (
                seen.append(value),
                value,
            )[1]
        ),
        types=SimpleNamespace(SECURITY_ATTRIBUTES=SimpleNamespace),
    )
    attributes = pipe._security_attributes(w, SID)
    assert seen == [
        f"D:P(A;;0x{pipe.CLIENT_RIGHTS:x};;;SY)(A;;0x{pipe.CLIENT_RIGHTS:x};;;BA)(A;;FA;;;{SID})"
    ]
    assert attributes.bInheritHandle is False
    assert pipe.CLIENT_RIGHTS & 4 == 0


def server_fixture(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    events: list[Any] = []
    stop = Event()
    endpoint = Handle()
    w = SimpleNamespace(
        pipe=SimpleNamespace(
            CreateNamedPipe=lambda *args: (events.append(("create", args)), endpoint)[1],
            DisconnectNamedPipe=lambda _: events.append("disconnect"),
        ),
    )
    pinned = SimpleNamespace(
        pid=os.getpid(),
        w=w,
        identity={"token_sid": SID},
        assert_current=lambda: None,
        close=lambda: events.append("pin-close"),
    )
    monkeypatch.setattr(pipe, "_PinnedService", lambda *_: pinned)
    monkeypatch.setattr(pipe, "_security_attributes", lambda *_: None)
    monkeypatch.setattr(pipe, "_assert_no_thread_token", lambda *_: None)
    monkeypatch.setattr(pipe, "_accept", lambda *_: True)
    monkeypatch.setattr(pipe, "_authorize", lambda *_: events.append("authorize-reverted"))
    monkeypatch.setattr(pipe.time, "monotonic", lambda: 110.0)
    return SimpleNamespace(w=w, stop=stop, endpoint=endpoint, events=events)


def test_server_reads_then_authenticates_and_proves_cleanup_before_completion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    server = server_fixture(monkeypatch)
    incoming = iter([request(), {"nonce": "a" * 64, "ack": True}])

    def read(*_: Any) -> dict[str, Any]:
        server.events.append("read")
        return next(incoming)

    @contextmanager
    def handler(_: Any) -> Iterator[dict[str, Any]]:
        server.events.append("sql-open")
        try:
            yield {"nonce": "a" * 64, "evidence": {}}
        finally:
            server.events.append("sql-close")

    def await_disconnect(*_: Any) -> None:
        server.events.append("client-read-and-closed")
        server.stop.set()

    monkeypatch.setattr(pipe, "_read", read)
    monkeypatch.setattr(
        pipe, "_write", lambda _w, _h, value, *_: server.events.append(copy.deepcopy(value))
    )
    monkeypatch.setattr(pipe, "_await_disconnect", await_disconnect)
    pipe.serve(NAME, server.stop, handler)
    assert server.events[1:] == [
        "read",
        "authorize-reverted",
        "sql-open",
        {"nonce": "a" * 64, "evidence": {}},
        "read",
        "sql-close",
        {"nonce": "a" * 64, "ack": True, "closed": True},
        "client-read-and-closed",
        "disconnect",
        "pin-close",
    ]
    args = server.events[0][1]
    assert args[1] & 0x80000 and args[2] & 8 and args[2] & 1
    assert server.endpoint.closed


def test_cleanup_failure_never_sends_completion(monkeypatch: pytest.MonkeyPatch) -> None:
    server = server_fixture(monkeypatch)
    incoming = iter([request(), {"nonce": "a" * 64, "ack": True}])
    monkeypatch.setattr(pipe, "_read", lambda *_: next(incoming))
    monkeypatch.setattr(pipe, "_write", lambda _w, _h, value, *_: server.events.append(value))

    @contextmanager
    def handler(_: Any) -> Iterator[dict[str, Any]]:
        try:
            yield {"response": True}
        finally:
            server.stop.set()
            raise ValueError("simulated connection close failure")

    pipe.serve(NAME, server.stop, handler)
    assert {"response": True} in server.events
    assert not any(isinstance(value, dict) and value.get("closed") for value in server.events)
    assert server.endpoint.closed


def test_replay_rejected_and_first_pipe_handle_reserved_between_clients(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    server = server_fixture(monkeypatch)
    incoming = iter([request(), {"nonce": "a" * 64, "ack": True}, request()])
    monkeypatch.setattr(pipe, "_read", lambda *_: next(incoming))
    monkeypatch.setattr(pipe, "_write", lambda *_: None)
    monkeypatch.setattr(pipe, "_await_disconnect", lambda *_: None)
    calls: list[str] = []
    accepted = 0

    def accept(*_: Any) -> bool:
        nonlocal accepted
        accepted += 1
        if accepted > 2:
            server.stop.set()
            return False
        return True

    monkeypatch.setattr(pipe, "_accept", accept)

    @contextmanager
    def handler(_: Any) -> Iterator[dict[str, Any]]:
        calls.append("sql")
        yield {}

    pipe.serve(NAME, server.stop, handler)
    assert calls == ["sql"]
    assert (
        len([value for value in server.events if isinstance(value, tuple) and value[0] == "create"])
        == 1
    )


def test_nowait_read_and_disconnect_honor_deadline_and_stop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(pipe.time, "monotonic", lambda: 130.0)
    w = SimpleNamespace(
        file=SimpleNamespace(ReadFile=lambda *_: pytest.fail("expired IO must not run"))
    )
    with pytest.raises(pipe.EvidencePipeError, match="deadline"):
        pipe._read(w, Handle(), 130.0)
    stop = Event()
    stop.set()
    with pytest.raises(pipe.EvidencePipeError, match="stopping"):
        pipe._await_disconnect(w, Handle(), 140.0, stop)


def test_completion_waits_for_native_client_disconnect(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pipe, "_pause", lambda *_: None)
    calls: list[int] = []

    def read(*_: Any) -> None:
        calls.append(1)
        raise NativeError(232 if len(calls) == 1 else 109)

    pipe._await_disconnect(
        SimpleNamespace(file=SimpleNamespace(ReadFile=read)), Handle(), 1.0, Event()
    )
    assert calls == [1, 1]

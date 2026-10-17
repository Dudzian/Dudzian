"""Local, native authenticated service evidence transport.

The client pins the SCM process and its primary token until collection ends.
Identification-only pipe impersonation on the server reads caller identity;
it is always reverted before any evidence handler or database operation.
Linux simulations of these adapters do not qualify native Windows security.
"""

from __future__ import annotations

import ctypes
import importlib
import json
import math
import os
import re
import struct
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import PureWindowsPath
from threading import Event
from typing import Any, Callable, ContextManager, Iterator

from deployment.windows_installer.contract import CONTRACT

SERVICES = (CONTRACT.backend_service, CONTRACT.verifier_service)
MAX_FRAME = 256 * 1024
MAX_SECONDS = 30.0
# FILE_READ/WRITE_DATA, READ/WRITE_ATTRIBUTES, READ_CONTROL, SYNCHRONIZE.
# FILE_APPEND_DATA aliases FILE_CREATE_PIPE_INSTANCE (0x4), so GENERIC_WRITE
# must never be granted to clients or requested by CreateFile.
CLIENT_RIGHTS = 0x00120183
_HEX = re.compile(r"[0-9a-f]{64}")
_POLL = 0.02


class EvidencePipeError(ValueError):
    """The native peer, protocol, authorization, or deadline is unavailable."""


class _RevertError(RuntimeError):
    """Terminate the endpoint thread if its effective identity is uncertain."""


def pipe_name(name: str) -> str:
    if name not in SERVICES:
        raise EvidencePipeError("unknown evidence service")
    return rf"\\.\pipe\CryptoHunter.Stage9.PostgreSQLEvidence.v1.{name}"


def _win32() -> Any:
    if os.name != "nt" or struct.calcsize("P") != 8:
        raise EvidencePipeError("native Windows x64 evidence pipes are required")
    from types import SimpleNamespace

    return SimpleNamespace(
        **{
            short: importlib.import_module(module)
            for short, module in (
                ("api", "win32api"),
                ("event", "win32event"),
                ("file", "win32file"),
                ("pipe", "win32pipe"),
                ("security", "win32security"),
                ("service", "win32service"),
                ("process", "win32process"),
                ("types", "pywintypes"),
            )
        }
    )


def _canonical_image(name: str) -> str:
    native: Any = importlib.import_module("ctypes")
    machine, host = ctypes.c_ushort(), ctypes.c_ushort()
    query = native.windll.kernel32.IsWow64Process2
    query.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_ushort),
        ctypes.POINTER(ctypes.c_ushort),
    ]
    if not query(ctypes.c_void_p(-1), ctypes.byref(machine), ctypes.byref(host)):
        raise EvidencePipeError("native architecture is unavailable")
    if machine.value != 0 or host.value != 0x8664:
        raise EvidencePipeError("native AMD64 service process is required")
    buffer = ctypes.create_unicode_buffer(32768)
    if native.windll.shell32.SHGetFolderPathW(None, 0x26, None, 0, buffer) != 0:
        raise EvidencePipeError("native Program Files folder is unavailable")
    return str(PureWindowsPath(buffer.value) / "CryptoHunter" / f"{name}.exe")


def _timestamp(value: Any) -> float:
    if isinstance(value, str):
        value = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise EvidencePipeError("service process creation time is unavailable")
    return value.timestamp()


def _same_path(first: Any, second: Any) -> bool:
    return (
        isinstance(first, str)
        and isinstance(second, str)
        and PureWindowsPath(first) == PureWindowsPath(second)
    )


def _sid(w: Any, token: Any) -> str:
    user = w.security.GetTokenInformation(token, 1)  # TokenUser
    return str(w.security.ConvertSidToStringSid(user[0] if isinstance(user, tuple) else user))


class _PinnedService:
    def __init__(self, name: str, observation: dict[str, Any] | None) -> None:
        pipe_name(name)
        self.w = _win32()
        self.name = name
        self.handles: list[Any] = []
        self.scm_handles: list[Any] = []
        try:
            self.manager = self.w.service.OpenSCManager(None, None, 1)
            self.scm_handles.append(self.manager)
            self.service = self.w.service.OpenService(self.manager, name, 1 | 4)
            self.scm_handles.append(self.service)
            config, status = self._state()
            self.pid = status["ProcessId"]
            if type(self.pid) is not int or self.pid <= 0:
                raise EvidencePipeError("service process ID is unavailable")
            self.process = self.w.api.OpenProcess(0x1000 | 0x100000, False, self.pid)
            self.handles.append(self.process)
            self.token = self.w.security.OpenProcessToken(self.process, 8)
            self.handles.append(self.token)
            self.expected = _canonical_image(name)
            sid = self.w.security.LookupAccountName(None, rf"NT SERVICE\{name}")[0]
            self.expected_sid = self.w.security.ConvertSidToStringSid(sid)
            self.identity = self._identity()
            self.assert_current()
            if observation is not None:
                rows = observation.get("services", [])
                matches = [row for row in rows if isinstance(row, dict) and row.get("name") == name]
                if len(matches) != 1:
                    raise EvidencePipeError("service inspection identity is unavailable")
                row = matches[0]
                if (
                    row.get("process_id") != self.pid
                    or row.get("second_process_id") != self.pid
                    or row.get("state") != "Running"
                    or row.get("second_state") != 4
                    or row.get("command_line") != config[3]
                    or row.get("account") != config[7]
                    or not _same_path(row.get("process_path"), self.expected)
                    or abs(
                        _timestamp(row.get("process_started"))
                        - _timestamp(self.identity["process_started"])
                    )
                    > 0.001
                    or not _same_path(
                        observation.get("paths", {}).get("program_files"),
                        str(PureWindowsPath(self.expected).parent),
                    )
                ):
                    raise EvidencePipeError("pinned service differs from host inspection")
        except BaseException:
            self.close()
            raise

    def _state(self) -> tuple[Any, Any]:
        config = self.w.service.QueryServiceConfig(self.service)
        status = self.w.service.QueryServiceStatusEx(self.service)
        if (
            config[0] != 16
            or config[7] != rf"NT SERVICE\{self.name}"
            or status["CurrentState"] != 4
        ):
            raise EvidencePipeError("canonical own-process service is not running")
        return config, status

    def _identity(self) -> dict[str, Any]:
        started = self.w.process.GetProcessTimes(self.process)["CreationTime"]
        return {
            "name": self.name,
            "process_id": self.pid,
            "token_sid": _sid(self.w, self.token),
            "process_path": self.w.process.QueryFullProcessImageName(self.process, 0),
            "process_started": started.isoformat(),
        }

    def assert_current(self) -> None:
        config, status = self._state()
        if (
            status["ProcessId"] != self.pid
            or self.w.event.WaitForSingleObject(self.process, 0) != 258  # WAIT_TIMEOUT: alive
            or config[3]
            not in (f'"{self.expected}"', self.expected if " " not in self.expected else None)
            or self._identity() != self.identity
            or self.identity["token_sid"] != self.expected_sid
            or not _same_path(self.identity["process_path"], self.expected)
        ):
            raise EvidencePipeError("pinned SCM service process changed or exited")

    def close(self) -> None:
        failed = False
        for handle in reversed(self.handles):
            try:
                handle.Close()
            except Exception:
                failed = True
        self.handles.clear()
        for handle in reversed(self.scm_handles):
            try:
                self.w.service.CloseServiceHandle(handle)
            except Exception:
                failed = True
        self.scm_handles.clear()
        if failed:
            raise EvidencePipeError("native service handle cleanup failed")


def _pipe_pid(handle: Any, *, server: bool) -> int:
    native: Any = importlib.import_module("ctypes")
    query = getattr(
        native.windll.kernel32,
        "GetNamedPipeServerProcessId" if server else "GetNamedPipeClientProcessId",
    )
    query.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_ulong)]
    query.restype = ctypes.c_int
    pid = ctypes.c_ulong()
    if not query(ctypes.c_void_p(int(handle)), ctypes.byref(pid)) or pid.value == 0:
        raise EvidencePipeError("native pipe peer process ID is unavailable")
    return int(pid.value)


def _error_code(exc: Exception) -> Any:
    return getattr(exc, "winerror", exc.args[0] if exc.args else None)


def _pause(deadline: float, stop: Event | None) -> None:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise EvidencePipeError("evidence pipe deadline expired")
    if stop is not None:
        if stop.wait(min(_POLL, remaining)):
            raise EvidencePipeError("evidence pipe is stopping")
    else:
        time.sleep(min(_POLL, remaining))


def _reject_constant(value: str) -> Any:
    raise EvidencePipeError("nonfinite JSON value")


def _float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise EvidencePipeError("nonfinite JSON number")
    return number


def _pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in items:
        if key in result:
            raise EvidencePipeError("duplicate JSON key")
        result[key] = value
    return result


def _encode(payload: dict[str, Any]) -> bytes:
    try:
        encoded = json.dumps(payload, allow_nan=False, separators=(",", ":")).encode("utf-8")
    except (TypeError, ValueError, RecursionError) as exc:
        raise EvidencePipeError("invalid JSON frame") from exc
    if type(payload) is not dict or not 0 < len(encoded) <= MAX_FRAME:
        raise EvidencePipeError("invalid or oversized JSON frame")
    return encoded


def _decode(encoded: bytes) -> dict[str, Any]:
    if not 0 < len(encoded) <= MAX_FRAME:
        raise EvidencePipeError("empty or oversized JSON frame")
    try:
        payload = json.loads(
            encoded.decode("utf-8"),
            object_pairs_hook=_pairs,
            parse_constant=_reject_constant,
            parse_float=_float,
        )
    except (UnicodeError, ValueError, RecursionError) as exc:
        raise EvidencePipeError("invalid JSON frame") from exc
    if type(payload) is not dict:
        raise EvidencePipeError("JSON frame must be an object")
    return payload


def _read(w: Any, handle: Any, deadline: float, stop: Event | None = None) -> dict[str, Any]:
    while True:
        _pause(deadline, stop)
        try:
            code, data = w.file.ReadFile(handle, MAX_FRAME + 1)
        except Exception as exc:
            if _error_code(exc) == 232:  # ERROR_NO_DATA in PIPE_NOWAIT mode
                continue
            raise EvidencePipeError("evidence pipe read failed") from exc
        if code != 0:
            raise EvidencePipeError("partial or oversized pipe message")
        if data:
            return _decode(bytes(data))


def _write(
    w: Any, handle: Any, payload: dict[str, Any], deadline: float, stop: Event | None = None
) -> None:
    data = _encode(payload)
    while True:
        _pause(deadline, stop)
        try:
            code, written = w.file.WriteFile(handle, data)
        except Exception as exc:
            raise EvidencePipeError("evidence pipe write failed") from exc
        if code != 0 or written not in (0, len(data)):
            raise EvidencePipeError("partial pipe message write")
        if written == len(data):
            return


def _request_deadline(payload: dict[str, Any]) -> float:
    if (
        set(payload) != {"version", "nonce", "collection_id", "issued", "expires"}
        or type(payload.get("version")) is not int
        or payload["version"] != 1
        or any(
            not isinstance(payload.get(key), str) or _HEX.fullmatch(payload[key]) is None
            for key in ("nonce", "collection_id")
        )
        or any(
            type(payload.get(key)) not in (int, float) or not math.isfinite(payload[key])
            for key in ("issued", "expires")
        )
    ):
        raise EvidencePipeError("invalid evidence request")
    now = time.monotonic()
    if (
        payload["expires"] != payload["issued"] + MAX_SECONDS
        or not payload["issued"] <= now < payload["expires"]
    ):
        raise EvidencePipeError("expired or future evidence request")
    return float(min(payload["expires"], now + MAX_SECONDS))


class Peer:
    def __init__(self, pinned: _PinnedService, handle: Any) -> None:
        self._pinned, self._handle = pinned, handle
        self.identity = dict(pinned.identity)
        self._nonce: str | None = None
        self._deadline = 0.0
        self._requested = False

    def assert_current(self) -> None:
        self._pinned.assert_current()
        if _pipe_pid(self._handle, server=True) != self.identity["process_id"]:
            raise EvidencePipeError("pipe server differs from pinned SCM service")

    def request(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self._requested:
            raise EvidencePipeError("one request is allowed per evidence pipe")
        self._deadline = _request_deadline(payload)
        self._nonce = payload["nonce"]
        self._requested = True
        self.assert_current()
        _write(self._pinned.w, self._handle, payload, self._deadline)
        response = _read(self._pinned.w, self._handle, self._deadline)
        self.assert_current()
        return response

    def finish(self, nonce: str) -> None:
        if self._nonce is None or nonce != self._nonce:
            raise EvidencePipeError("evidence ACK nonce differs")
        self.assert_current()
        _write(self._pinned.w, self._handle, {"nonce": nonce, "ack": True}, self._deadline)
        completion = _read(self._pinned.w, self._handle, self._deadline)
        if completion != {"nonce": nonce, "ack": True, "closed": True} or any(
            type(completion.get(key)) is not bool for key in ("ack", "closed")
        ):
            raise EvidencePipeError("service SQL cleanup was not confirmed")
        self.assert_current()
        self._nonce = None


@contextmanager
def connect_service(name: str, observation: dict[str, Any]) -> Iterator[Peer]:
    pinned = _PinnedService(name, observation)
    handle = None
    try:
        deadline = time.monotonic() + 5.0
        while handle is None:
            pinned.assert_current()
            try:
                # SECURITY_SQOS_PRESENT | SECURITY_IDENTIFICATION, no delegation.
                handle = pinned.w.file.CreateFile(
                    pipe_name(name), CLIENT_RIGHTS, 0, None, 3, 0x110000, None
                )
            except Exception as exc:
                if _error_code(exc) not in (2, 231):
                    raise EvidencePipeError("evidence pipe connect failed") from exc
                _pause(deadline, None)
        pinned.w.pipe.SetNamedPipeHandleState(handle, 2 | 1, None, None)
        peer = Peer(pinned, handle)
        peer.assert_current()
        yield peer
        peer.assert_current()
    finally:
        _close_endpoint(handle, pinned)


def verify_service(name: str, observation: dict[str, Any], identity: dict[str, Any]) -> None:
    pinned = _PinnedService(name, observation)
    try:
        if pinned.identity != identity:
            raise EvidencePipeError("service restarted after evidence collection")
        pinned.assert_current()
    finally:
        pinned.close()


def _close_endpoint(handle: Any, pinned: _PinnedService) -> None:
    failed = False
    if handle is not None:
        try:
            handle.Close()
        except Exception:
            failed = True
    try:
        pinned.close()
    except Exception:
        failed = True
    if failed:
        raise EvidencePipeError("native evidence endpoint cleanup failed")


def _security_attributes(w: Any, sid: str) -> Any:
    descriptor = w.security.ConvertStringSecurityDescriptorToSecurityDescriptor(
        f"D:P(A;;0x{CLIENT_RIGHTS:x};;;SY)(A;;0x{CLIENT_RIGHTS:x};;;BA)(A;;FA;;;{sid})", 1
    )
    attributes = w.types.SECURITY_ATTRIBUTES()
    attributes.SECURITY_DESCRIPTOR = descriptor
    attributes.bInheritHandle = False
    return attributes


def _authorize(w: Any, handle: Any) -> None:
    client = _pipe_pid(handle, server=False)
    token = None
    try:
        w.pipe.ImpersonateNamedPipeClient(handle)
        token = w.security.OpenThreadToken(w.api.GetCurrentThread(), 8, True)
        if w.security.GetTokenInformation(token, 9) != 1:  # SecurityIdentification
            raise EvidencePipeError("caller did not restrict pipe impersonation to identification")
        sid = _sid(w, token)
        groups = w.security.GetTokenInformation(token, 2)
        administrator = any(
            w.security.ConvertSidToStringSid(group) == "S-1-5-32-544"
            and flags & 4
            and not flags & 16
            for group, flags in groups
        )
        if sid != "S-1-5-18" and not administrator:
            raise EvidencePipeError("evidence caller is not SYSTEM or an enabled administrator")
    finally:
        # Revert even if impersonation or token inspection raises; no handler
        # or database code can execute under the connecting caller's identity.
        revert_failed = False
        close_failed = False
        try:
            w.security.RevertToSelf()
        except Exception:
            revert_failed = True
        if token is not None:
            try:
                token.Close()
            except Exception:
                close_failed = True
        if revert_failed:
            raise _RevertError("caller identity could not be reverted")
        if close_failed:
            raise EvidencePipeError("caller identification token cleanup failed")
    if _pipe_pid(handle, server=False) != client:
        raise EvidencePipeError("native pipe caller process changed")


def _accept(w: Any, handle: Any, stop: Event) -> bool:
    while not stop.is_set():
        try:
            w.pipe.ConnectNamedPipe(handle, None)
            return True
        except Exception as exc:
            code = _error_code(exc)
            if code == 535:  # ERROR_PIPE_CONNECTED
                return True
            if code != 536:  # ERROR_PIPE_LISTENING in PIPE_NOWAIT mode
                raise EvidencePipeError("evidence pipe accept failed") from exc
        stop.wait(_POLL)
    return False


def _assert_no_thread_token(w: Any) -> None:
    try:
        token = w.security.OpenThreadToken(w.api.GetCurrentThread(), 8, True)
    except Exception as exc:
        if _error_code(exc) != 1008:
            raise EvidencePipeError("thread token absence could not be established") from exc
    else:
        token.Close()
        raise EvidencePipeError("evidence endpoint has an effective thread token")


def _await_disconnect(w: Any, handle: Any, deadline: float, stop: Event) -> None:
    # Closing immediately after WriteFile can discard an unread completion.
    # Let the client read it and close its handle; never FlushFileBuffers,
    # whose synchronous wait would let a client prevent service shutdown.
    while True:
        _pause(deadline, stop)
        try:
            w.file.ReadFile(handle, 1)
        except Exception as exc:
            code = _error_code(exc)
            if code in (109, 233):  # BROKEN_PIPE / PIPE_NOT_CONNECTED
                return
            if code == 232:  # no message yet, PIPE_NOWAIT
                continue
            raise EvidencePipeError("pipe completion disconnect failed") from exc
        raise EvidencePipeError("unexpected message after evidence completion")


def serve(
    name: str, stop: Event, handler: Callable[[dict[str, Any]], ContextManager[dict[str, Any]]]
) -> None:
    """Serve bounded sequential collections without blocking the service loop."""
    pinned = _PinnedService(name, None)
    handle = None
    try:
        if pinned.pid != os.getpid():
            raise EvidencePipeError("endpoint must run inside its canonical SCM service")
        w = pinned.w
        _assert_no_thread_token(w)
        attributes = _security_attributes(w, pinned.identity["token_sid"])
        used: dict[str, float] = {}
        # Reserve the name for the entire lifecycle, avoiding squatting gaps.
        # FIRST_PIPE_INSTANCE and REJECT_REMOTE_CLIENTS; noninheritable handle.
        handle = w.pipe.CreateNamedPipe(
            pipe_name(name), 3 | 0x80000, 4 | 2 | 1 | 8, 1, MAX_FRAME, MAX_FRAME, 0, attributes
        )
        while not stop.is_set():
            pinned.assert_current()
            try:
                if not _accept(w, handle, stop):
                    break
                # ImpersonateNamedPipeClient authenticates the last message
                # read. Bound untrusted input before identification; process
                # the request only after caller identity has been reverted.
                request = _read(w, handle, time.monotonic() + MAX_SECONDS, stop)
                _authorize(w, handle)
                _assert_no_thread_token(w)
                deadline = _request_deadline(request)
                now = time.monotonic()
                used = {nonce: expires for nonce, expires in used.items() if expires > now}
                if request["nonce"] in used or len(used) >= 128:
                    raise EvidencePipeError("replayed evidence nonce or replay cache capacity")
                used[request["nonce"]] = deadline
                pinned.assert_current()
                with handler(request) as response:
                    pinned.assert_current()
                    _write(w, handle, response, deadline, stop)
                    ack = _read(w, handle, deadline, stop)
                    if (
                        ack != {"nonce": request["nonce"], "ack": True}
                        or type(ack.get("ack")) is not bool
                    ):
                        raise EvidencePipeError("invalid evidence completion ACK")
                    pinned.assert_current()
                # Only confirm completion after the handler has closed its
                # private SQL connection without raising a cleanup error.
                _write(
                    w,
                    handle,
                    {"nonce": request["nonce"], "ack": True, "closed": True},
                    deadline,
                    stop,
                )
                _await_disconnect(w, handle, deadline, stop)
            except _RevertError:
                raise
            except Exception:
                # One invalid client cannot terminate the service endpoint.
                pass
            finally:
                try:
                    w.pipe.DisconnectNamedPipe(handle)
                except Exception:
                    pass
    finally:
        _close_endpoint(handle, pinned)

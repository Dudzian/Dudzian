"""Live Windows loopback proof for production HTTP transport recovery."""

from __future__ import annotations

import argparse
import asyncio
from contextlib import contextmanager
import json
import os
from pathlib import Path
import socket
import struct
import subprocess
import sys
import tempfile
import time
from typing import Any, Iterator

WINDOWS_NETWORK_RECOVERY = "WINDOWS_NETWORK_RECOVERY"
STATE_TIMEOUT_SECONDS = 10.0
POLL_INTERVAL_SECONDS = 0.02
STATE_REPLACE_TIMEOUT_SECONDS = 2.0
STATE_REPLACE_RETRY_SECONDS = 0.01
PEER_TIMEOUT_SECONDS = 15.0
HTTP_TIMEOUT_SECONDS = 3.0
STDERR_LIMIT = 8_192

PHASES = (
    "READY",
    "FIRST_CONNECTION_ACCEPTED",
    "FIRST_REQUEST_RECEIVED",
    "FIRST_CONNECTION_RESET",
    "SECOND_CONNECTION_ACCEPTED",
    "SECOND_REQUEST_RECEIVED",
    "SUCCESS",
)


class Stage7NetworkProbeError(RuntimeError):
    def __init__(self, label: str, detail: str) -> None:
        self.label = label
        self.item = WINDOWS_NETWORK_RECOVERY
        self.detail = detail
        super().__init__(f"[{label}] {detail}")


def _replace_state_snapshot(temporary: Path, path: Path) -> None:
    """Replace one snapshot, tolerating only transient Windows sharing races."""
    deadline = time.monotonic() + STATE_REPLACE_TIMEOUT_SECONDS
    while True:
        try:
            os.replace(temporary, path)
            return
        except PermissionError as exc:
            if (
                os.name != "nt"
                or getattr(exc, "winerror", None) not in {5, 32}
                or time.monotonic() >= deadline
            ):
                raise
            time.sleep(STATE_REPLACE_RETRY_SECONDS)


def _write_state(path: Path, state: dict[str, Any]) -> None:
    """Publish one complete peer snapshot; readers never observe partial JSON."""
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(state, stream, separators=(",", ":"))
        stream.flush()
        os.fsync(stream.fileno())
    _replace_state_snapshot(temporary, path)


def _record(path: Path, state: dict[str, Any], phase: str) -> None:
    state["phase"] = phase
    state["events"].append(phase)
    _write_state(path, state)


def _abortive_linger() -> bytes:
    # Windows defines linger fields as u_short; POSIX test runners commonly use int.
    return struct.pack("hh" if os.name == "nt" else "ii", 1, 0)


def _peer(state_path: Path) -> int:
    state: dict[str, Any] = {"phase": "STARTING", "port": 0, "connections": 0, "events": []}
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.settimeout(PEER_TIMEOUT_SECONDS)
        server.bind(("127.0.0.1", 0))
        server.listen(2)
        state["port"] = server.getsockname()[1]
        _record(state_path, state, "READY")

        first, _ = server.accept()
        state["connections"] += 1
        _record(state_path, state, "FIRST_CONNECTION_ACCEPTED")
        with first:
            first.settimeout(HTTP_TIMEOUT_SECONDS)
            request = first.recv(8192)
            if not request:
                raise RuntimeError("first connection contained no HTTP request bytes")
            _record(state_path, state, "FIRST_REQUEST_RECEIVED")
            # Enabled SO_LINGER with a zero timeout makes close abortive (TCP RST).
            first.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, _abortive_linger())
        _record(state_path, state, "FIRST_CONNECTION_RESET")

        second, _ = server.accept()
        state["connections"] += 1
        _record(state_path, state, "SECOND_CONNECTION_ACCEPTED")
        with second:
            second.settimeout(HTTP_TIMEOUT_SECONDS)
            request = second.recv(8192)
            if not request.startswith(b"GET "):
                raise RuntimeError("reconnect did not contain the expected GET request")
            _record(state_path, state, "SECOND_REQUEST_RECEIVED")
            second.sendall(
                b"HTTP/1.1 200 OK\r\nContent-Length: 9\r\nConnection: close\r\n\r\nRECOVERED"
            )
        _record(state_path, state, "SUCCESS")
    return 0


def _read_state(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None
    return value if isinstance(value, dict) else None


def _wait_for_phase(
    path: Path, process: subprocess.Popen[str], phase: str, timeout: float = STATE_TIMEOUT_SECONDS
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        state = _read_state(path)
        if state is not None and state.get("phase") == phase:
            return state
        if process.poll() is not None:
            break
        time.sleep(POLL_INTERVAL_SECONDS)
    state = _read_state(path)
    if state is not None and state.get("phase") == phase:
        return state
    raise TimeoutError(f"deadline expired waiting for peer phase {phase}")


@contextmanager
def _direct_loopback_environment() -> Iterator[None]:
    """Temporarily ensure httpx's environment proxy lookup bypasses loopback."""
    missing = object()
    previous: dict[str, str | object] = {
        name: os.environ.get(name, missing) for name in ("NO_PROXY", "no_proxy")
    }
    try:
        for name in previous:
            current = os.environ.get(name, "")
            entries = [entry.strip() for entry in current.split(",") if entry.strip()]
            lower = {entry.lower() for entry in entries}
            entries.extend(host for host in ("127.0.0.1", "localhost") if host not in lower)
            os.environ[name] = ",".join(entries)
        yield
    finally:
        for name, value in previous.items():
            if value is missing:
                os.environ.pop(name, None)
            else:
                os.environ[name] = str(value)


async def _logical_request(port: int):
    from core.network import RateLimitedAsyncClient

    async with RateLimitedAsyncClient(
        timeout=HTTP_TIMEOUT_SECONDS,
        retries=2,
        backoff_factor=0.05,
        jitter_range=(0, 0),
    ) as client:
        return await client.request("GET", f"http://127.0.0.1:{port}/recovery")


def _stop_and_capture(process: subprocess.Popen[str]) -> tuple[str, list[str]]:
    errors: list[str] = []
    if process.poll() is None:
        process.terminate()
    try:
        _, stderr = process.communicate(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        try:
            _, stderr = process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            stderr = ""
            errors.append("peer process survived kill deadline")
    return (stderr or "")[-STDERR_LIMIT:], errors


def _diagnostic(
    state: dict[str, Any] | None,
    process: subprocess.Popen[str],
    stderr: str,
    client_exception: BaseException | None,
    peer_was_alive: bool | None = None,
) -> str:
    state = state or {}
    alive = process.poll() is None if peer_was_alive is None else peer_was_alive
    returncode = "running" if alive else str(process.returncode)
    exception = "none"
    if client_exception is not None:
        exception = f"{type(client_exception).__name__}: {client_exception}"
    return (
        f"phase={state.get('phase', 'UNAVAILABLE')} "
        f"connections={state.get('connections', 0)} events={state.get('events', [])!r} "
        f"client_exception={exception} peer_alive={str(alive).lower()} "
        f"peer_returncode={returncode} peer_stderr={stderr!r}"
    )


def _report_cleanup_errors(cleanup_errors: list[str], primary: BaseException | None) -> None:
    if not cleanup_errors:
        return
    detail = "; ".join(cleanup_errors)
    if primary is None:
        raise Stage7NetworkProbeError("NETWORK_RECOVERY_VERIFY", detail)
    print(f"[CLEANUP] secondary failure: {detail}", file=sys.stderr)


def run_probe(scratch_parent: Path) -> dict[str, str]:
    if os.name != "nt":
        raise Stage7NetworkProbeError(
            "NETWORK_RECOVERY_SERVER_START", "live PASS is restricted to Windows"
        )
    primary: BaseException | None = None
    process: subprocess.Popen[str] | None = None
    cleanup_errors: list[str] = []
    client_exception: BaseException | None = None
    try:
        with tempfile.TemporaryDirectory(dir=scratch_parent) as temporary:
            state_path = Path(temporary) / "peer-state.json"
            process = subprocess.Popen(
                [sys.executable, "-u", "-m", __name__, "--peer", str(state_path)],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
            )
            try:
                ready = _wait_for_phase(state_path, process, "READY")
                port = ready.get("port")
                if not isinstance(port, int) or port <= 0:
                    raise Stage7NetworkProbeError(
                        "NETWORK_RECOVERY_SERVER_START", "READY state has no dynamic port"
                    )
                with _direct_loopback_environment():
                    try:
                        response = asyncio.run(_logical_request(port))
                    except Exception as exc:
                        client_exception = exc
                        raise Stage7NetworkProbeError(
                            "NETWORK_RECOVERY_RECONNECT", "production request did not recover"
                        ) from exc
                process.wait(timeout=PEER_TIMEOUT_SECONDS)
                proof = _read_state(state_path)
                if process.returncode != 0:
                    raise Stage7NetworkProbeError("NETWORK_RECOVERY_VERIFY", "peer failed")
                if (
                    response.status_code != 200
                    or response.text != "RECOVERED"
                    or proof
                    != {
                        "phase": "SUCCESS",
                        "port": port,
                        "connections": 2,
                        "events": list(PHASES),
                    }
                ):
                    raise Stage7NetworkProbeError(
                        "NETWORK_RECOVERY_VERIFY", "response or ordered socket proof mismatch"
                    )
                # The peer is already finished, so communicate cannot block on a live stderr pipe.
                stderr, errors = _stop_and_capture(process)
                cleanup_errors.extend(errors)
                if stderr:
                    raise Stage7NetworkProbeError(
                        "NETWORK_RECOVERY_VERIFY", "successful peer emitted stderr"
                    )
                return {WINDOWS_NETWORK_RECOVERY: "PASS"}
            except BaseException as exc:
                peer_was_alive = process.poll() is None
                stderr, errors = _stop_and_capture(process)
                cleanup_errors.extend(errors)
                state = _read_state(state_path)
                label = (
                    exc.label
                    if isinstance(exc, Stage7NetworkProbeError)
                    else "NETWORK_RECOVERY_SERVER_START"
                    if state is None or state.get("phase") in (None, "STARTING")
                    else "NETWORK_RECOVERY_FIRST_FAILURE"
                    if state.get("phase") in PHASES[:3]
                    else "NETWORK_RECOVERY_RECONNECT"
                    if state.get("phase") in PHASES[3:6]
                    else "NETWORK_RECOVERY_VERIFY"
                )
                detail = exc.detail if isinstance(exc, Stage7NetworkProbeError) else str(exc)
                raise Stage7NetworkProbeError(
                    label,
                    f"{detail}; "
                    f"{_diagnostic(state, process, stderr, client_exception, peer_was_alive)}",
                ) from exc
    except BaseException as exc:
        primary = exc
        raise
    finally:
        if process is not None and process.poll() is None:
            _, errors = _stop_and_capture(process)
            cleanup_errors.extend(errors)
        _report_cleanup_errors(cleanup_errors, primary)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--peer", type=Path)
    parser.add_argument("--scratch-parent", type=Path)
    args = parser.parse_args(argv)
    if args.peer is not None:
        return _peer(args.peer)
    scratch = args.scratch_parent or Path(tempfile.gettempdir())
    try:
        print(json.dumps(run_probe(scratch)))
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

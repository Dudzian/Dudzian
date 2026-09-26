"""Live Windows loopback proof for production HTTP transport recovery."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
import queue
import socket
import struct
import subprocess
import sys
import tempfile
import threading

WINDOWS_NETWORK_RECOVERY = "WINDOWS_NETWORK_RECOVERY"
MARKER_TIMEOUT_SECONDS = 10.0
PEER_TIMEOUT_SECONDS = 15.0
HTTP_TIMEOUT_SECONDS = 3.0


class Stage7NetworkProbeError(RuntimeError):
    def __init__(self, label: str, detail: str) -> None:
        self.label = label
        self.item = WINDOWS_NETWORK_RECOVERY
        super().__init__(f"[{label}] {detail}")


def _peer(result_path: Path) -> int:
    events: list[str] = []
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.settimeout(PEER_TIMEOUT_SECONDS)
        server.bind(("127.0.0.1", 0))
        server.listen(2)
        print(f"READY\t{server.getsockname()[1]}", flush=True)
        first, _ = server.accept()
        with first:
            first.settimeout(HTTP_TIMEOUT_SECONDS)
            first.recv(8192)
            events.append("FIRST_FAILURE")
            # An abortive close produces a real TCP reset rather than an HTTP response.
            first.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("hh", 1, 0))
        print("FIRST_FAILURE", flush=True)
        second, _ = server.accept()
        with second:
            second.settimeout(HTTP_TIMEOUT_SECONDS)
            request = second.recv(8192)
            if not request.startswith(b"GET "):
                raise RuntimeError("reconnect did not contain the expected HTTP request")
            events.append("RECONNECT")
            second.sendall(
                b"HTTP/1.1 200 OK\r\nContent-Length: 9\r\nConnection: close\r\n\r\nRECOVERED"
            )
            events.append("SUCCESS")
    result_path.write_text(json.dumps({"connections": 2, "events": events}), encoding="utf-8")
    print("SUCCESS", flush=True)
    return 0


def _read_line(process: subprocess.Popen[str], label: str) -> str:
    if process.stdout is None:
        raise Stage7NetworkProbeError(label, "peer stdout is unavailable")
    output: queue.Queue[str] = queue.Queue(maxsize=1)
    threading.Thread(target=lambda: output.put(process.stdout.readline()), daemon=True).start()
    try:
        return output.get(timeout=MARKER_TIMEOUT_SECONDS).strip()
    except queue.Empty as exc:
        raise Stage7NetworkProbeError(label, "peer marker deadline expired") from exc


async def _logical_request(port: int):
    from core.network import RateLimitedAsyncClient

    async with RateLimitedAsyncClient(
        timeout=HTTP_TIMEOUT_SECONDS,
        retries=2,
        backoff_factor=0.05,
        jitter_range=(0, 0),
    ) as client:
        return await client.request("GET", f"http://127.0.0.1:{port}/recovery")


def _report_cleanup_errors(
    cleanup_errors: list[str], primary: BaseException | None
) -> None:
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
    try:
        with tempfile.TemporaryDirectory(dir=scratch_parent) as temporary:
            result_path = Path(temporary) / "peer-result.json"
            process = subprocess.Popen(
                [sys.executable, "-u", "-m", __name__, "--peer", str(result_path)],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            ready = _read_line(process, "NETWORK_RECOVERY_SERVER_START")
            if not ready.startswith("READY\t"):
                raise Stage7NetworkProbeError("NETWORK_RECOVERY_SERVER_START", ready)
            port = int(ready.split("\t", 1)[1])
            try:
                response = asyncio.run(_logical_request(port))
            except Exception as exc:
                observed = _read_line(process, "NETWORK_RECOVERY_FIRST_FAILURE")
                if observed != "FIRST_FAILURE":
                    raise Stage7NetworkProbeError(
                        "NETWORK_RECOVERY_FIRST_FAILURE", "peer did not confirm reset"
                    ) from exc
                raise Stage7NetworkProbeError(
                    "NETWORK_RECOVERY_RECONNECT",
                    f"production request did not recover: {type(exc).__name__}: {exc}",
                ) from exc
            if _read_line(process, "NETWORK_RECOVERY_FIRST_FAILURE") != "FIRST_FAILURE":
                raise Stage7NetworkProbeError(
                    "NETWORK_RECOVERY_FIRST_FAILURE", "peer did not confirm reset"
                )
            if _read_line(process, "NETWORK_RECOVERY_RECONNECT") != "SUCCESS":
                raise Stage7NetworkProbeError(
                    "NETWORK_RECOVERY_RECONNECT", "peer did not confirm reconnect"
                )
            process.wait(timeout=PEER_TIMEOUT_SECONDS)
            if process.returncode != 0 or not result_path.is_file():
                raise Stage7NetworkProbeError("NETWORK_RECOVERY_VERIFY", "peer failed")
            proof = json.loads(result_path.read_text(encoding="utf-8"))
            if (
                response.status_code != 200
                or response.text != "RECOVERED"
                or proof
                != {
                    "connections": 2,
                    "events": ["FIRST_FAILURE", "RECONNECT", "SUCCESS"],
                }
            ):
                raise Stage7NetworkProbeError(
                    "NETWORK_RECOVERY_VERIFY", "response or ordered socket proof mismatch"
                )
            return {WINDOWS_NETWORK_RECOVERY: "PASS"}
    except BaseException as exc:
        primary = exc
        raise
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    cleanup_errors.append("peer process survived kill deadline")
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

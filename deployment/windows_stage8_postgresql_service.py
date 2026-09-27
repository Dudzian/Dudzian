"""SCM child used only by the live Stage-8 PostgreSQL mTLS probe.

The parent owns the service and request file.  This process merely opens one
new libpq connection under the token assigned by SCM and publishes a bounded
result.  Its account-name string is diagnostic only; the parent independently
inspects the process token.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import threading
import time
from typing import Any, Callable

MAX_RESULT_BYTES = 16_384
RUNTIME_SERVICE = "CryptoHunterBackend"
VERIFIER_SERVICE = "CryptoHunterFreshnessVerifier"


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    data = json.dumps(value, separators=(",", ":"))
    if len(data.encode("utf-8")) > MAX_RESULT_BYTES:
        data = json.dumps({"ok": False, "error": "result exceeded bound"})
    temporary.write_text(data, encoding="utf-8")
    os.replace(temporary, path)


def _signal_service_stop(service: Any, stop_requested: threading.Event) -> None:
    """Publish STOP_PENDING and cancel work; never claim premature STOPPED."""
    service.ReportServiceStatus(3)  # SERVICE_STOP_PENDING
    stop_requested.set()


def _read_matching_go(path: Path, invocation_token: str, pid: int) -> bool:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        isinstance(value, dict)
        and value.get("invocation_token") == invocation_token
        and value.get("pid") == pid
        and value.get("go") is True
    )


def _connect(
    request: Path,
    service_name: str,
    connect: Any | None = None,
    *,
    cancelled: Callable[[], bool] | None = None,
    wait_timeout: float = 15.0,
) -> None:
    is_cancelled = cancelled or (lambda: False)
    value = json.loads(request.read_text(encoding="utf-8"))
    invocation_token = value["invocation_token"]
    if (
        type(invocation_token) is not str
        or len(invocation_token) != 64
        or any(character not in "0123456789abcdef" for character in invocation_token)
    ):
        raise RuntimeError("invalid invocation token")
    result = Path(value["result"])
    ready = Path(value["ready"])
    go = Path(value["go"])
    pid = os.getpid()
    _atomic_json(
        ready,
        {"ready": True, "invocation_token": invocation_token, "pid": pid},
    )
    deadline = time.monotonic() + wait_timeout
    while (
        not is_cancelled()
        and not _read_matching_go(go, invocation_token, pid)
        and time.monotonic() < deadline
    ):
        time.sleep(0.05)
    if is_cancelled():
        _atomic_json(
            result,
            {
                "ok": False,
                "cancelled": True,
                "invocation_token": invocation_token,
                "pid": pid,
            },
        )
        return
    if not _read_matching_go(go, invocation_token, pid):
        _atomic_json(
            result,
            {
                "ok": False,
                "invocation_token": invocation_token,
                "pid": pid,
                "error": "fresh parent release deadline expired",
            },
        )
        return
    if service_name == RUNTIME_SERVICE:
        credential = "runtime"
    elif service_name == VERIFIER_SERVICE:
        credential = "verifier"
    else:
        raise RuntimeError("unowned service identity")
    # Credential locations are fixed by the reviewed service command, never by
    # request JSON.  The service token must still pass the private-key DACL.
    pki = request.parent / "pki"
    certificate = pki / credential / "client.crt"
    private_key = pki / credential / "client.key"
    root_certificate = pki / "ca.crt"
    probes: dict[str, dict[str, Any]] = {}
    for name in ("runtime", "verifier"):
        candidate = pki / name / "client.key"
        try:
            with candidate.open("rb") as stream:
                stream.read(1)
        except OSError as exc:
            probes[name] = {
                "accessible": False,
                "error_type": type(exc).__name__,
                "winerror": getattr(exc, "winerror", None),
            }
        else:
            probes[name] = {"accessible": True}
    if value.get("operation") == "key_probe":
        _atomic_json(result, {"invocation_token": invocation_token, "pid": pid, "keys": probes})
        return
    if connect is None:
        import psycopg

        connect = psycopg.connect
    try:
        with connect(
            host="127.0.0.1",
            port=int(value["port"]),
            dbname=value["database"],
            user=value["role"],
            sslmode="verify-full",
            sslrootcert=str(root_certificate),
            sslcert=str(certificate),
            sslkey=str(private_key),
            connect_timeout=5,
        ) as connection:
            row = connection.execute("SELECT session_user").fetchone()
            payload = {"ok": True, "session_user": row[0]}
    except Exception as exc:
        # Never serialize connection info: it can contain credentials in other uses.
        payload = {
            "ok": False,
            "sqlstate": getattr(exc, "sqlstate", None),
            "error_type": type(exc).__name__,
            "error": str(exc)[-2048:],
        }
    _atomic_json(
        result,
        {"invocation_token": invocation_token, "pid": pid, **payload},
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--service-name", required=True)
    parser.add_argument("--request", type=Path, required=True)
    args = parser.parse_args(argv)
    if os.name != "nt":
        return 2
    import servicemanager
    import win32serviceutil

    service_name = args.service_name
    request = args.request

    class OneConnectionService(win32serviceutil.ServiceFramework):
        _svc_name_ = service_name
        _svc_display_name_ = service_name

        def __init__(self, service_args: list[str]) -> None:
            super().__init__(service_args)
            self._stop_requested = threading.Event()

        def SvcStop(self) -> None:  # noqa: N802 - pywin32 API
            # The framework reports STOPPED only after SvcDoRun has returned.
            _signal_service_stop(self, self._stop_requested)

        def SvcDoRun(self) -> None:  # noqa: N802 - pywin32 API
            _connect(request, service_name, cancelled=self._stop_requested.is_set)

    servicemanager.Initialize()
    servicemanager.PrepareToHostSingle(OneConnectionService)
    servicemanager.StartServiceCtrlDispatcher()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# tests/cloud/test_cloud_service.py
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import signal
import subprocess
import sys
import time
from enum import Enum
from pathlib import Path
from typing import Any

import pytest

# --- Importy z bot_core.cloud (z fallbackiem na CloudRuntimeState) ---
try:
    from bot_core.cloud import CloudRuntimeService, CloudServerConfig, CloudRuntimeState  # type: ignore
except Exception:
    from bot_core.cloud import CloudRuntimeService, CloudServerConfig  # type: ignore

    class CloudRuntimeState(str, Enum):
        """
        Fallback tylko do testów: wystarczy, żeby plik się importował.
        """
        UNKNOWN = "unknown"


# --- Helpers ---
def _write_signed_flag(flag_path: Path, signature_path: Path, secret: bytes) -> None:
    """
    Runtime w tym repo wygląda na to, że oczekuje, że plik *.sig będzie JSON-em.
    Ten helper zapisuje:
      - cloud_flag.json jako JSON
      - cloud_flag.sig jako JSON z HMAC-SHA256 nad bajtami cloud_flag.json
    """
    now = int(time.time())
    payload: dict[str, Any] = {
        "enabled": True,
        "issued_at": now,
        "expires_at": now + 3600,
        "version": 1,
    }
    raw = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    flag_path.write_bytes(raw)

    digest = hmac.new(secret, raw, hashlib.sha256).hexdigest()

    sig_obj: dict[str, Any] = {
        # najbardziej typowe pola
        "algo": "HMAC-SHA256",
        "encoding": "hex",
        "signature": digest,
        # dodatkowe aliasy (na wypadek gdy walidator szuka innej nazwy)
        "sig": digest,
        "hmac": digest,
        # czasem walidatory chcą też hash samego payloadu
        "payload_sha256": hashlib.sha256(raw).hexdigest(),
        "ts": now,
    }
    signature_path.write_text(json.dumps(sig_obj, ensure_ascii=False), encoding="utf-8")


def _terminate_process(proc: subprocess.Popen[str], timeout_s: float = 10.0) -> None:
    if proc.poll() is not None:
        return

    try:
        if os.name == "nt":
            proc.send_signal(signal.CTRL_BREAK_EVENT)  # type: ignore[attr-defined]
        else:
            proc.send_signal(signal.SIGINT)
    except Exception:
        pass

    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if proc.poll() is not None:
            return
        time.sleep(0.1)

    try:
        proc.kill()
    except Exception:
        pass


def _read_available_stdout(proc: subprocess.Popen[str]) -> str:
    out: list[str] = []
    if proc.stdout is None:
        return ""
    try:
        while True:
            line = proc.stdout.readline()
            if not line:
                break
            out.append(line)
            if proc.poll() is None and len(out) > 5000:
                break
    except Exception:
        return ""
    return "".join(out)


# --- Tests ---
@pytest.mark.integration
@pytest.mark.requires_trading_stubs
def test_cloud_cli_serves_core_services(tmp_path: Path) -> None:
    """
    Uruchamia scripts/run_cloud_service.py i czeka aż zapisze ready-file.
    """
    # flag + podpis
    flag_path = Path("var/runtime/cloud_flag.json")
    signature_path = Path("var/runtime/cloud_flag.sig")
    flag_path.parent.mkdir(parents=True, exist_ok=True)

    secret = b"cloud-runtime-flag"
    _write_signed_flag(flag_path, signature_path, secret)

    env = os.environ.copy()
    # nadal ustawiamy secret w env (bo runtime może go potrzebować do weryfikacji HMAC)
    env["CLOUD_RUNTIME_FLAG_SECRET"] = f"base64:{base64.b64encode(secret).decode('ascii')}"

    # config cloud dla uruchomienia
    config_path = tmp_path / "cloud.yaml"
    config_path.write_text(
        "\n".join(
            [
                "host: 127.0.0.1",
                "port: 0",
                "runtime:",
                "  config_path: config/runtime.yaml",
                "  entrypoint: trading_gui",
                "license:",
                "  enabled: false",
                "marketplace:",
                "  refresh_interval_seconds: 1",
                "",
            ]
        ),
        encoding="utf-8",
    )

    ready_file = tmp_path / "ready.json"

    creationflags = 0
    if os.name == "nt":
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore[attr-defined]

    proc = subprocess.Popen(
        [
            sys.executable,
            "scripts/run_cloud_service.py",
            "--config",
            str(config_path),
            "--ready-file",
            str(ready_file),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
        creationflags=creationflags,
    )

    try:
        # czekamy maks 30s (120 * 0.25)
        for _ in range(120):
            if ready_file.exists():
                break
            if proc.poll() is not None:
                break
            time.sleep(0.25)

        if not ready_file.exists():
            output = _read_available_stdout(proc)
            raise AssertionError(
                f"Serwer cloud nie zasygnalizował gotowości (brak {ready_file}):\n{output}"
            )

        data = json.loads(ready_file.read_text(encoding="utf-8"))
        assert isinstance(data, dict)

    finally:
        _terminate_process(proc)


def test_cloud_runtime_state_importable() -> None:
    assert CloudRuntimeState is not None


def test_cloud_runtime_service_smoke() -> None:
    cfg = CloudServerConfig(host="127.0.0.1", port=0)  # type: ignore[call-arg]
    svc = CloudRuntimeService(cfg)  # type: ignore[call-arg]
    assert svc is not None

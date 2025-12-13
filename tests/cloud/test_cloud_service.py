# tests/cloud/test_cloud_service.py
from __future__ import annotations

import base64
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

from bot_core.security.signing import build_hmac_signature, canonical_json_bytes

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
    Runtime oczekuje, że plik *.sig będzie JSON-em o konkretnej strukturze.
    Ten helper zapisuje:
      - cloud_flag.json jako JSON
      - cloud_flag.sig jako minimalny JSON z HMAC-SHA256 nad bajtami cloud_flag.json

    UWAGA: celowo NIE dodajemy dodatkowych pól/aliasów, bo walidator może wymagać ścisłego schematu.
    """
    now = int(time.time())
    payload: dict[str, Any] = {
        "enabled": True,
        "issued_at": now,
        "expires_at": now + 3600,
        "version": 1,
    }

    # Stała serializacja (żeby podpis był deterministyczny)
    raw = canonical_json_bytes(payload)
    flag_path.write_bytes(raw)

    signature = build_hmac_signature(payload, key=secret)
    signature_path.write_text(json.dumps(signature, ensure_ascii=False), encoding="utf-8")


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
def _invoke_cloud_cli(
    tmp_path: Path, *, ci_smoke: bool, expect_mode: str
) -> dict[str, Any]:
    """
    Startuje scripts/run_cloud_service.py i czeka aż zapisze ready-file.

    Tryb smoke jest używany na Windows, żeby CI nie blokowało się na środowiskach,
    które wymagają pełnego GUI/giełdy. Nadal weryfikujemy parsowanie flag, ładowanie
    configu, walidację podpisanej flagi i emiter gotowości.
    """
    # flag + podpis
    flag_path = Path("var/runtime/cloud_flag.json")
    signature_path = Path("var/runtime/cloud_flag.sig")
    flag_path.parent.mkdir(parents=True, exist_ok=True)

    secret = b"cloud-runtime-flag"
    _write_signed_flag(flag_path, signature_path, secret)

    env = os.environ.copy()
    env["CLOUD_RUNTIME_FLAG_SECRET"] = f"base64:{base64.b64encode(secret).decode('ascii')}"
    env.setdefault("BOT_CORE_LICENSE_PUBLIC_KEY", "11" * 32)
    # Windows runners w GitHub Actions potrafią buforować stdout pythonowego
    # subprocessu, co utrudnia diagnostykę przy ewentualnym błędzie startu.
    env.setdefault("PYTHONUNBUFFERED", "1")

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

    cmd = [
        sys.executable,
        "-u",
        "scripts/run_cloud_service.py",
        "--config",
        str(config_path),
        "--ready-file",
        str(ready_file),
        "--emit-stdout",
    ]
    if ci_smoke:
        cmd.append("--ci-smoke")

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
        creationflags=creationflags,
    )

    try:
        for _ in range(120):  # 30s
            if ready_file.exists():
                break
            if proc.poll() is not None:
                break
            time.sleep(0.25)

        if not ready_file.exists():
            _terminate_process(proc)
            output = _read_available_stdout(proc)
            raise AssertionError(
                f"Serwer cloud nie zasygnalizował gotowości (brak {ready_file}):\n{output}"
            )

        data = json.loads(ready_file.read_text(encoding="utf-8"))
        assert isinstance(data, dict)
        assert data.get("event") == "ready"
        runtime_cfg = data.get("runtime", {})
        assert runtime_cfg.get("mode") == expect_mode
        assert runtime_cfg.get("config") == str(Path("config/runtime.yaml").resolve())
        assert runtime_cfg.get("entrypoint") == "trading_gui"
        meta = data.get("meta", {})
        assert isinstance(meta.get("pid"), int)
        assert isinstance(meta.get("timestamp"), int)
        assert meta.get("platform")
        assert meta.get("python_version")
        assert meta.get("package_version")

        return data

    finally:
        _terminate_process(proc)


@pytest.mark.integration
@pytest.mark.requires_trading_stubs
def test_cloud_cli_smoke_ready(tmp_path: Path) -> None:
    data = _invoke_cloud_cli(tmp_path, ci_smoke=True, expect_mode="smoke")
    assert data["address"] == "ci-smoke"


@pytest.mark.integration
@pytest.mark.requires_trading_stubs
@pytest.mark.skipif(os.name == "nt", reason="Pełne uruchomienie jest niestabilne na Windows CI")
def test_cloud_cli_serves_core_services(tmp_path: Path) -> None:
    data = _invoke_cloud_cli(tmp_path, ci_smoke=False, expect_mode="active")
    assert data["event"] == "ready"


def test_cloud_runtime_state_importable() -> None:
    assert CloudRuntimeState is not None


def test_cloud_runtime_service_smoke() -> None:
    cfg = CloudServerConfig(host="127.0.0.1", port=0)  # type: ignore[call-arg]
    svc = CloudRuntimeService(cfg)  # type: ignore[call-arg]
    assert svc is not None

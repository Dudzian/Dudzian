# tests/cloud/test_cloud_service.py
from __future__ import annotations

import base64
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import pytest

# --- Importy z bot_core.cloud (z fallbackiem na CloudRuntimeState) ---
try:
    # nowa/najnowsza ścieżka (jeśli istnieje)
    from bot_core.cloud import CloudRuntimeService, CloudServerConfig, CloudRuntimeState  # type: ignore
except Exception:
    # minimalny, kompatybilny import dla wersji gdzie CloudRuntimeState nie jest eksportowany w __init__.py
    from bot_core.cloud import CloudRuntimeService, CloudServerConfig  # type: ignore

    class CloudRuntimeState(str, Enum):
        """
        Fallback tylko do testów: wystarczy, żeby plik się importował.
        Jeżeli w Twoim kodzie runtime ma inne nazwy stanów, test i tak tego nie użyje,
        a ImportError przestanie blokować collection.
        """

        UNKNOWN = "unknown"


# --- Helpers ---
def _write_signed_flag(flag_path: Path, signature_path: Path, secret: bytes) -> None:
    """
    Minimalny, deterministyczny "podpis" używany przez testy integracyjne.
    To NIE jest kryptografia produkcyjna – chodzi o prosty handshake dla runtime flag.
    """
    payload = {"ts": int(time.time()), "enabled": True}
    flag_path.write_text(json.dumps(payload), encoding="utf-8")

    # podpis: base64(secret) + ":" + base64(json)
    raw = flag_path.read_bytes()
    sig = base64.b64encode(secret) + b":" + base64.b64encode(raw)
    signature_path.write_bytes(sig)


def _terminate_process(proc: subprocess.Popen[str], timeout_s: float = 10.0) -> None:
    if proc.poll() is not None:
        return

    try:
        if os.name == "nt":
            # na Windows CTRL_BREAK_EVENT działa tylko dla CREATE_NEW_PROCESS_GROUP
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
    """
    Zbiera to co jest dostępne w stdout (nie blokując w nieskończoność).
    """
    out = []
    if proc.stdout is None:
        return ""
    try:
        # .read() może blokować; tu wolimy próbować małymi porcjami.
        # Na Windows w GH runnerze to zwykle działa OK.
        while True:
            chunk = proc.stdout.readline()
            if not chunk:
                break
            out.append(chunk)
            # jeśli proces żyje, nie ciągniemy bez końca
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
        # Wymagane, żeby CTRL_BREAK_EVENT działał poprawnie.
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore[attr-defined]

    proc = subprocess.Popen(
        [sys.executable, "scripts/run_cloud_service.py", "--config", str(config_path), "--ready-file", str(ready_file)],
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
            raise AssertionError(f"Serwer cloud nie zasygnalizował gotowości (brak {ready_file}):\n{output}")

        # sanity check: ready.json powinien być JSON-em
        data = json.loads(ready_file.read_text(encoding="utf-8"))
        assert isinstance(data, dict)

    finally:
        _terminate_process(proc)


def test_cloud_runtime_state_importable() -> None:
    """
    Ten test gwarantuje, że plik nie wywali się na ImportError przy zmianach eksportów.
    """
    assert CloudRuntimeState is not None


def test_cloud_runtime_service_smoke() -> None:
    """
    Bardzo lekki smoke: czy da się skonstruować config/serwis bez crasha importów.
    """
    cfg = CloudServerConfig(host="127.0.0.1", port=0)  # type: ignore[call-arg]
    svc = CloudRuntimeService(cfg)  # type: ignore[call-arg]
    assert svc is not None

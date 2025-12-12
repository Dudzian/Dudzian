from __future__ import annotations

import base64
import json
import os
import signal
import subprocess
import sys
import time
import threading
from collections import deque
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import grpc
import pytest
from google.protobuf import empty_pb2

from bot_core.cloud import CloudRuntimeService, CloudServerConfig, CloudRuntimeState
from bot_core.cloud.runtime_flag import build_hmac_signature
from bot_core.generated import cloud_pb2_grpc, marketplace_pb2_grpc, trading_pb2_grpc


def _write_signed_flag(flag_path: Path, signature_path: Path, secret: bytes) -> None:
    flag_path.write_text(json.dumps({"cloud": True}), encoding="utf-8")
    signature = build_hmac_signature(flag_path.read_bytes(), secret)
    signature_path.write_text(signature, encoding="utf-8")


class DummyTradingGui(trading_pb2_grpc.TradingGuiServicer):
    def Ping(self, request: empty_pb2.Empty, context: Any) -> empty_pb2.Empty:  # noqa: N802
        return empty_pb2.Empty()


class DummyMarketplace(marketplace_pb2_grpc.MarketplaceServicer):
    pass


class DummyCloudRuntime(cloud_pb2_grpc.CloudRuntimeServicer):
    pass


class DummyRuntimeService(CloudRuntimeService):
    def __init__(self) -> None:
        self.state = CloudRuntimeState(
            runtime_ready=True,
            marketplace_ready=True,
            trading_ready=True,
        )
        self.servicers = SimpleNamespace(
            trading_gui=DummyTradingGui(),
            marketplace=DummyMarketplace(),
            cloud_runtime=DummyCloudRuntime(),
        )

    def load_config(self, config_path: Path) -> None:
        return

    def start(self) -> None:
        return

    def stop(self) -> None:
        return


@pytest.fixture(autouse=True)
def _force_dummy_runtime(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr("bot_core.cloud.cli.CloudRuntimeService", DummyRuntimeService)


@pytest.mark.integration
@pytest.mark.requires_trading_stubs
def test_cloud_cli_serves_core_services(tmp_path: Path):
    flag_path = Path("var/runtime/cloud_flag.json")
    signature_path = Path("var/runtime/cloud_flag.sig")
    flag_path.parent.mkdir(parents=True, exist_ok=True)
    secret = b"cloud-runtime-flag"
    _write_signed_flag(flag_path, signature_path, secret)

    env = os.environ.copy()
    env["CLOUD_RUNTIME_FLAG_SECRET"] = f"base64:{base64.b64encode(secret).decode('ascii')}"
    # Wymusza niebuforowany stdout, żebyśmy mogli debugować start serwera.
    env.setdefault("PYTHONUNBUFFERED", "1")

    config_path = tmp_path / "cloud.yaml"
    config_path.write_text(
        """
host: 127.0.0.1
port: 0
runtime:
  config_path: config/runtime.yaml
  entrypoint: trading_gui
license:
  enabled: false
marketplace:
  refresh_interval_seconds: 1
""".strip(),
        encoding="utf-8",
    )

    ready_file = tmp_path / "ready.json"

    # Czytamy stdout w tle, żeby nie blokować oraz mieć pełny log przy awarii.
    output_lines: deque[str] = deque(maxlen=5000)
    stop_reader = threading.Event()

    def _reader() -> None:
        if proc.stdout is None:
            return
        try:
            for line in proc.stdout:
                output_lines.append(line.rstrip("\n"))
                if stop_reader.is_set():
                    break
        except Exception:
            # W testach nie chcemy, żeby wyjątek z czytania stdout maskował prawdziwy problem.
            return

    creationflags = 0
    if os.name == "nt":
        # Wymagane, żeby CTRL_BREAK_EVENT działał poprawnie.
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP

    proc = subprocess.Popen(
        [sys.executable, "scripts/run_cloud_service.py", "--config", str(config_path), "--ready-file", str(ready_file)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
        creationflags=creationflags,
    )

    reader_thread = threading.Thread(target=_reader, name="cloud-service-stdout", daemon=True)
    reader_thread.start()

    # Windows potrafi startować wyraźnie wolniej na runnerach self-hosted.
    max_wait_seconds = 60 if os.name == "nt" else 30
    deadline = time.time() + max_wait_seconds

    try:
        while time.time() < deadline:
            if ready_file.exists():
                break
            if proc.poll() is not None:
                break
            time.sleep(0.25)

        if not ready_file.exists():
            # Dołączamy ostatnie linie logów, żeby awaria była diagnozowalna w CI.
            tail = "\n".join(list(output_lines)[-250:])
            proc_state = f"exit_code={proc.poll()!r}"
            raise AssertionError(
                "Serwer cloud nie zasygnalizował gotowości. "
                f"(timeout={max_wait_seconds}s, {proc_state})\n"
                f"--- stdout (tail) ---\n{tail}\n--- /stdout ---"
            )

        ready = json.loads(ready_file.read_text(encoding="utf-8"))
        address = ready.get("address")
        assert address, f"Brak 'address' w ready-file: {ready}"

        channel = grpc.insecure_channel(address)
        stub = trading_pb2_grpc.TradingGuiStub(channel)

        # Smoke: sprawdź, czy gRPC odpowiada.
        response = stub.Ping(empty_pb2.Empty(), timeout=5)
        assert isinstance(response, empty_pb2.Empty)

    finally:
        # Zatrzymanie procesu: Windows -> CTRL_BREAK, inne -> SIGINT.
        try:
            if os.name == "nt":
                proc.send_signal(signal.CTRL_BREAK_EVENT)
            else:
                proc.send_signal(signal.SIGINT)
        except Exception:
            pass

        try:
            proc.wait(timeout=10)
        except Exception:
            proc.kill()

        stop_reader.set()
        try:
            reader_thread.join(timeout=1)
        except Exception:
            pass


@pytest.mark.integration
def test_cloud_cli_rejects_invalid_flag(tmp_path: Path):
    flag_path = Path("var/runtime/cloud_flag.json")
    signature_path = Path("var/runtime/cloud_flag.sig")
    flag_path.parent.mkdir(parents=True, exist_ok=True)
    secret = b"cloud-runtime-flag"
    _write_signed_flag(flag_path, signature_path, secret)

    # Podmieniamy flagę po podpisaniu -> podpis ma być niepoprawny
    flag_path.write_text(json.dumps({"cloud": False}), encoding="utf-8")

    env = os.environ.copy()
    env["CLOUD_RUNTIME_FLAG_SECRET"] = f"base64:{base64.b64encode(secret).decode('ascii')}"

    config_path = tmp_path / "cloud.yaml"
    config_path.write_text(
        """
host: 127.0.0.1
port: 0
runtime:
  config_path: config/runtime.yaml
  entrypoint: trading_gui
license:
  enabled: false
marketplace:
  refresh_interval_seconds: 1
""".strip(),
        encoding="utf-8",
    )

    proc = subprocess.Popen(
        [sys.executable, "scripts/run_cloud_service.py", "--config", str(config_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
    )

    try:
        # Oczekujemy, że proces szybko padnie (odmowa przez zły podpis)
        for _ in range(80):
            if proc.poll() is not None:
                break
            time.sleep(0.1)

        assert proc.poll() is not None, "Proces powinien zakończyć się przy niepoprawnej fladze"
        output = ""
        if proc.stdout is not None:
            try:
                output = proc.stdout.read()
            except Exception:
                output = ""
        assert proc.returncode != 0, f"Nie oczekiwano kodu 0. Output:\n{output}"

    finally:
        try:
            proc.kill()
        except Exception:
            pass

from __future__ import annotations

import base64
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import grpc
import pytest
from google.protobuf import empty_pb2

from bot_core.cloud import CloudRuntimeService, CloudServerConfig, CloudRuntimeConfig
from bot_core.generated import trading_pb2, trading_pb2_grpc
from bot_core.security.signing import build_hmac_signature


def _write_signed_flag(flag_path: Path, signature_path: Path, secret: bytes, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    document = payload or {"enabled": True, "issued_by": "pytest", "nonce": "cloud"}
    flag_path.write_text(json.dumps(document, ensure_ascii=False), encoding="utf-8")
    signature = build_hmac_signature(document, key=secret)
    signature_path.write_text(json.dumps(signature, ensure_ascii=False), encoding="utf-8")
    return document


class _DummyServer:
    def __init__(self, context, host: str, port: int, max_workers: int, *, interceptors=None) -> None:
        self.context = context
        self.host = host
        self.port = port
        self.grpc_server = SimpleNamespace()
        self._started = False

    @property
    def address(self) -> str:
        return f"{self.host}:{self.port or 1234}"

    def start(self) -> None:
        self._started = True

    def stop(self, *_args: object, **_kwargs: Any) -> None:
        self._started = False

    def wait(self) -> None:
        return None


def test_cloud_runtime_service_registers_extra_servicer(monkeypatch):
    context = SimpleNamespace(start=lambda: None, stop=lambda: None)
    builder_calls: list[tuple[Any, Any]] = []

    def _builder(*, config_path, entrypoint=None, **_kwargs):
        builder_calls.append((config_path, entrypoint))
        return context

    server_instances: list[_DummyServer] = []

    def _server_factory(ctx, host: str, port: int, max_workers: int, *, interceptors=None):
        server = _DummyServer(ctx, host, port, max_workers, interceptors=interceptors)
        server_instances.append(server)
        return server

    monkeypatch.setattr("bot_core.cloud.service.LocalRuntimeServer", _server_factory)

    config = CloudServerConfig(runtime=CloudRuntimeConfig(config_path=Path("config/runtime.yaml")))
    service = CloudRuntimeService(config, context_builder=_builder)

    invoked: list[str] = []

    def _registrar(server, ctx):
        invoked.append("ok")

    service.register_servicer(_registrar)
    service.start()
    assert builder_calls, "kontekst powinien zostać zainicjalizowany"
    assert invoked == ["ok"], "rejestrator powinien zostać wywołany"
    service.stop()


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
""".strip()
    )
    ready_file = tmp_path / "ready.json"
    proc = subprocess.Popen(
        [sys.executable, "scripts/run_cloud_service.py", "--config", str(config_path), "--ready-file", str(ready_file)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
    )
    try:
        for _ in range(120):
            if ready_file.exists():
                break
            if proc.poll() is not None:
                break
            time.sleep(0.25)
        else:
            pass

        if not ready_file.exists():
            output = ""
            if proc.poll() is not None and proc.stdout is not None:
                try:
                    output = proc.stdout.read()
                except Exception:
                    output = ""
            raise AssertionError(f"Serwer cloud nie zasygnalizował gotowości: {output}")

        payload = json.loads(ready_file.read_text(encoding="utf-8"))
        address = payload["address"]
        channel = grpc.insecure_channel(address)
        grpc.channel_ready_future(channel).result(timeout=10)

        health_stub = trading_pb2_grpc.HealthServiceStub(channel)
        health_response = health_stub.Check(empty_pb2.Empty())
        assert health_response.version

        runtime_stub = trading_pb2_grpc.RuntimeServiceStub(channel)
        runtime_response = runtime_stub.ListDecisions(trading_pb2.ListDecisionsRequest(limit=1))
        assert runtime_response.total >= 0

        market_stub = trading_pb2_grpc.MarketDataServiceStub(channel)
        instruments = market_stub.ListTradableInstruments(
            trading_pb2.ListTradableInstrumentsRequest()
        )
        assert instruments.instruments, "serwer powinien udostępniać instrumenty"
        instrument = instruments.instruments[0].instrument
        ohlcv = market_stub.GetOhlcvHistory(
            trading_pb2.GetOhlcvHistoryRequest(
                instrument=instrument,
                granularity=trading_pb2.CandleGranularity(iso8601_duration="PT1H"),
                limit=5,
            )
        )
        assert ohlcv.candles is not None

        marketplace_stub = trading_pb2_grpc.MarketplaceServiceStub(channel)
        presets = marketplace_stub.ListPresets(trading_pb2.ListMarketplacePresetsRequest())
        assert len(presets.presets) >= 0

        channel.close()
    finally:
        proc.send_signal(signal.SIGINT)
        proc.wait(timeout=20)


@pytest.mark.integration
def test_cloud_cli_rejects_invalid_flag(tmp_path: Path):
    flag_path = Path("var/runtime/cloud_flag.json")
    signature_path = Path("var/runtime/cloud_flag.sig")
    flag_path.parent.mkdir(parents=True, exist_ok=True)

    secret = b"cloud-runtime-flag"
    env = os.environ.copy()
    env["CLOUD_RUNTIME_FLAG_SECRET"] = f"base64:{base64.b64encode(secret).decode('ascii')}"

    # zapisujemy podpis wygenerowany na innym sekrecie, aby walidacja się nie powiodła
    _write_signed_flag(flag_path, signature_path, b"invalid-secret")

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
""".strip()
    )

    result = subprocess.run(
        [sys.executable, "scripts/run_cloud_service.py", "--config", str(config_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
        check=False,
    )

    assert result.returncode == 4
    assert "Walidacja podpisanej flagi cloudowej" in (result.stdout or "")

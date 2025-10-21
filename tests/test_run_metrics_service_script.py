"""Testy CLI uruchamiającego MetricsService."""

from __future__ import annotations

import importlib
import json
import textwrap
import threading
import time
from pathlib import Path
from typing import Mapping

import pytest

try:  # pragma: no cover - środowiska bez grpcio
    grpc = importlib.import_module("grpc")
except Exception:  # pragma: no cover - brak zależności gRPC
    grpc = None  # type: ignore[assignment]

try:  # pragma: no cover - brak wygenerowanych stubów
    trading_pb2 = importlib.import_module("bot_core.generated.trading_pb2")
    trading_pb2_grpc = importlib.import_module("bot_core.generated.trading_pb2_grpc")
except Exception:  # pragma: no cover - brak stubów
    trading_pb2 = None  # type: ignore[assignment]
    trading_pb2_grpc = None  # type: ignore[assignment]

HAVE_REAL_GRPC = grpc is not None and trading_pb2 is not None and trading_pb2_grpc is not None

from scripts import run_metrics_service
from bot_core.runtime.metrics_service import LoggingSink
from tests._metrics_service_helpers import (
    MemoryAuditStub,
    MetricsServerStub,
    RouterStub,
    RuntimeServerStub,
    SinkStub,
    get_security_section,
)


def _make_dummy_server(
    *,
    address: str = "127.0.0.1:7777",
    jsonl_path: Path | None = None,
    history_size: int = 512,
    tls_enabled: bool = False,
    require_client_auth: bool = False,
    logging_sink_enabled: bool = True,
):
    logging_sink = LoggingSink() if logging_sink_enabled else None
    jsonl_sink = None
    if jsonl_path is not None:
        jsonl_cls = getattr(run_metrics_service, "JsonlSink", None)
        if jsonl_cls is not None:
            jsonl_sink = jsonl_cls(jsonl_path)
        else:

            class _FallbackJsonlSink:
                def __init__(self, path: Path) -> None:
                    self._path = Path(path)
                    self._fsync = False

            jsonl_sink = _FallbackJsonlSink(jsonl_path)

    sinks = tuple(sink for sink in (logging_sink, jsonl_sink) if sink is not None)
    metadata = {
        "history_size": history_size,
        "logging_sink_enabled": logging_sink_enabled,
        "jsonl_sink": {
            "active": jsonl_path is not None,
            "path": str(jsonl_path) if jsonl_path is not None else None,
            "fsync": False,
        },
        "ui_alerts_sink": {
            "active": False,
            "path": None,
        },
        "sink_descriptions": [
            {
                "class": sink.__class__.__name__,
                "module": sink.__class__.__module__,
            }
            for sink in sinks
        ],
        "additional_sink_descriptions": [],
        "tls": {
            "configured": tls_enabled,
            "require_client_auth": require_client_auth,
        },
    }

    return MetricsServerStub(
        address=address,
        history_size=history_size,
        tls_enabled=tls_enabled,
        require_client_auth=require_client_auth,
        sinks=sinks,
        runtime_metadata=metadata,
    )


@pytest.mark.timeout(5)
def test_metrics_service_cli_creates_jsonl(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    jsonl_path = tmp_path / "metrics.jsonl"
    dummy_server = _make_dummy_server(
        jsonl_path=jsonl_path,
        logging_sink_enabled=False,
    )

    monkeypatch.setattr(run_metrics_service, "METRICS_RUNTIME_AVAILABLE", True)
    monkeypatch.setattr(run_metrics_service, "_build_server", lambda **_: dummy_server)

    args = [
        "--host",
        "127.0.0.1",
        "--port",
        "0",
        "--jsonl",
        str(jsonl_path),
        "--shutdown-after",
        "0.3",
        "--no-log-sink",
    ]

    exit_code = run_metrics_service.main(args)
    assert exit_code == 0
    assert jsonl_path.exists()


@pytest.mark.timeout(5)
@pytest.mark.skipif(
    not HAVE_REAL_GRPC,
    reason="Wymaga grpcio i wygenerowanych stubów gRPC",
)
def test_metrics_service_cli_accepts_metrics(tmp_path, monkeypatch):
    jsonl_path = tmp_path / "metrics.jsonl"
    args = [
        "--host",
        "127.0.0.1",
        "--port",
        "0",
        "--jsonl",
        str(jsonl_path),
        "--log-level",
        "debug",
        "--shutdown-after",
        "0.6",
    ]

    server_holder: dict[str, object] = {}

    def fake_build_server(**kwargs):
        server = run_metrics_service.create_metrics_server(**kwargs)
        server_holder["server"] = server
        return server

    monkeypatch.setattr(run_metrics_service, "_build_server", fake_build_server)

    exit_code: list[int] = []

    def run_cli():
        exit_code.append(run_metrics_service.main(args))

    thread = threading.Thread(target=run_cli)
    thread.start()
    time.sleep(0.2)

    server = server_holder.get("server")
    assert server is not None, "CLI powinno zainicjalizować serwer"
    channel = grpc.insecure_channel(server.address)  # type: ignore[attr-defined]
    stub = trading_pb2_grpc.MetricsServiceStub(channel)
    snapshot = trading_pb2.MetricsSnapshot()
    snapshot.notes = "cli-test"
    snapshot.fps = 61.2
    stub.PushMetrics(snapshot)

    thread.join()
    assert exit_code and exit_code[0] == 0

    lines = [line for line in jsonl_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert lines, "Plik JSONL powinien zawierać wpis po PushMetrics"
    last = json.loads(lines[-1])
    assert last["notes"] == "cli-test"
    assert last["fps"] == pytest.approx(61.2)


def test_metrics_service_tls_config_passed(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    cert = tmp_path / "server.crt"
    key = tmp_path / "server.key"
    ca = tmp_path / "clients.pem"
    for path in (cert, key, ca):
        path.write_text("dummy", encoding="utf-8")

    captured_kwargs: dict[str, object] = {}

    monkeypatch.setattr(run_metrics_service, "METRICS_RUNTIME_AVAILABLE", True)

    def fake_build_server(**kwargs):
        nonlocal captured_kwargs
        captured_kwargs = kwargs
        return RuntimeServerStub("127.0.0.1:1234")

    monkeypatch.setattr(run_metrics_service, "_build_server", fake_build_server)

    args = [
        "--host",
        "127.0.0.1",
        "--port",
        "50070",
        "--tls-cert",
        str(cert),
        "--tls-key",
        str(key),
        "--tls-client-ca",
        str(ca),
        "--tls-require-client-cert",
        "--shutdown-after",
        "0.1",
    ]

    exit_code = run_metrics_service.main(args)
    assert exit_code == 0
    tls = captured_kwargs.get("tls_config")
    assert tls is not None
    assert tls["certificate_path"] == cert
    assert tls["private_key_path"] == key
    assert tls["client_ca_path"] == ca
    assert tls["require_client_auth"] is True


def test_tls_key_permissions_warning(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys) -> None:
    monkeypatch.setattr(run_metrics_service, "METRICS_RUNTIME_AVAILABLE", False)

    cert = tmp_path / "cert.pem"
    cert.write_text("cert", encoding="utf-8")
    key = tmp_path / "key.pem"
    key.write_text("key", encoding="utf-8")
    key.chmod(0o666)

    exit_code = run_metrics_service.main(
        [
            "--tls-cert",
            str(cert),
            "--tls-key",
            str(key),
            "--print-config-plan",
        ]
    )
    assert exit_code == 0

    payload = json.loads(capsys.readouterr().out.strip())
    tls_section = payload.get("tls", {})
    private_key_meta = tls_section.get("private_key")
    assert private_key_meta is not None
    warnings = private_key_meta.get("security_warnings", [])
    assert any("Klucz prywatny TLS" in message for message in warnings)
    security_flags = private_key_meta.get("security_flags", {})
    assert security_flags.get("world_writable") is True
    assert private_key_meta.get("role") == "tls_key"


def test_metrics_service_print_config_plan(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    jsonl_path = tmp_path / "telemetry.jsonl"
    cert = tmp_path / "server.crt"
    key = tmp_path / "server.key"
    cert.write_text("cert", encoding="utf-8")
    key.write_text("key", encoding="utf-8")

    dummy_server = _make_dummy_server(
        jsonl_path=jsonl_path,
        tls_enabled=True,
        require_client_auth=False,
    )

    monkeypatch.setattr(run_metrics_service, "METRICS_RUNTIME_AVAILABLE", True)
    monkeypatch.setattr(run_metrics_service, "_build_server", lambda **_: dummy_server)

    args = [
        "--host",
        "0.0.0.0",
        "--port",
        "0",
        "--jsonl",
        str(jsonl_path),
        "--tls-cert",
        str(cert),
        "--tls-key",
        str(key),
        "--print-config-plan",
    ]

    exit_code = run_metrics_service.main(args)
    assert exit_code == 0

    captured = capsys.readouterr().out.strip()
    assert captured, "CLI powinno wypisać plan konfiguracji"
    payload = json.loads(captured)
    assert payload["address"] == "127.0.0.1:7777"
    assert payload["jsonl_sink"]["configured"] is True
    tls_payload = payload["tls"]
    assert tls_payload["configured"] is True
    assert tls_payload["certificate"]["path"].endswith("server.crt")
    assert payload["logging_sink_enabled"] is True
    runtime_state = payload["runtime_state"]
    assert runtime_state["available"] is True
    assert runtime_state["reason"] is None
    assert runtime_state["sink_count"] == 2

    # assertions from main branch that must also be preserved
    assert runtime_state["jsonl_sink_active"] is True
    assert runtime_state["logging_sink_active"] is True
    assert runtime_state["jsonl_sink"]["path"].endswith("telemetry.jsonl")
    assert runtime_state["tls"]["enabled"] is True
    assert runtime_state["metadata_source"] == "runtime_metadata"
    ui_section = payload["ui_alerts"]
    assert ui_section["reduce_mode"] == "enable"
    assert ui_section["overlay_mode"] == "enable"
    assert ui_section["reduce_motion_dispatch"] is True
    assert ui_section["overlay_dispatch"] is True
    assert ui_section["reduce_motion_logging"] is True
    assert ui_section["overlay_logging"] is True
    assert (
        ui_section["overlay_critical_threshold"]
        == run_metrics_service._DEFAULT_OVERLAY_THRESHOLD
    )
    parameter_sources = payload.get("parameter_sources", {})
    assert parameter_sources.get("fail_on_security_warnings") == "default"
    security = get_security_section(payload)
    assert security["enabled"] is False
    assert security["source"] == "default"
    assert security["parameter_source"] == "default"


# Preserved test from the other branch; renamed to keep BOTH versions
# (only the function name changed to avoid shadowing)

def test_metrics_service_print_risk_profiles_codex(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = run_metrics_service.main(["--print-risk-profiles"])
    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert "risk_profiles" in payload
    assert "conservative" in payload["risk_profiles"]
    assert (
        payload["risk_profiles"]["conservative"]["summary"]["name"]
        == "conservative"
    )


# assertions from main branch also define this test; we keep it as-is

def test_metrics_service_print_risk_profiles(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = run_metrics_service.main(["--print-risk-profiles"])
    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert "risk_profiles" in payload
    assert "conservative" in payload["risk_profiles"]



def test_metrics_service_fail_on_security_warnings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys,
) -> None:
    jsonl_path = tmp_path / "telemetry.jsonl"
    cert = tmp_path / "server.crt"
    key = tmp_path / "server.key"
    cert.write_text("cert", encoding="utf-8")
    key.write_text("key", encoding="utf-8")

    dummy_server = _make_dummy_server(
        jsonl_path=jsonl_path,
        tls_enabled=True,
        require_client_auth=False,
    )

    monkeypatch.setattr(run_metrics_service, "METRICS_RUNTIME_AVAILABLE", True)
    monkeypatch.setattr(run_metrics_service, "_build_server", lambda **_: dummy_server)

    args = [
        "--host",
        "0.0.0.0",
        "--port",
        "0",
        "--jsonl",
        str(jsonl_path),
        "--tls-cert",
        str(cert),
        "--tls-key",
        str(key),
        "--print-config-plan",
        "--fail-on-security-warnings",
    ]

    exit_code = run_metrics_service.main(args)
    assert exit_code == 3

    captured = capsys.readouterr().out.strip()
    assert captured
    payload = json.loads(captured)
    parameter_sources = payload.get("parameter_sources", {})
    assert parameter_sources.get("fail_on_security_warnings") == "cli"
    security = get_security_section(payload)
    assert security["enabled"] is True
    assert security["source"] == "cli"
    assert security["parameter_source"] == "cli"
    assert security.get("environment_variable") is None



def test_metrics_service_config_plan_jsonl_written(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    output_path = tmp_path / "plan.jsonl"

    dummy_server = _make_dummy_server(
        address="127.0.0.1:9001",
        jsonl_path=None,
        tls_enabled=False,
        logging_sink_enabled=True,
    )

    monkeypatch.setattr(run_metrics_service, "METRICS_RUNTIME_AVAILABLE", True)
    monkeypatch.setattr(run_metrics_service, "_build_server", lambda **_: dummy_server)

    args = [
        "--host",
        "127.0.0.1",
        "--port",
        "0",
        "--config-plan-jsonl",
        str(output_path),
        "--shutdown-after",
        "0.0",
    ]

    exit_code = run_metrics_service.main(args)
    assert exit_code == 0

    contents = output_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(contents) == 1
    payload = json.loads(contents[0])
    assert payload["address"] == "127.0.0.1:9001"
    assert payload["jsonl_sink"]["configured"] is False
    runtime_state = payload["runtime_state"]
    assert runtime_state["available"] is True
    assert runtime_state["reason"] is None
    assert runtime_state["sink_count"] >= 1
    assert runtime_state["logging_sink_active"] is True
    assert runtime_state["jsonl_sink_active"] is False
    assert runtime_state["tls"]["enabled"] is False
    ui_section = payload["metrics"]["ui_alerts"]
    assert ui_section["reduce_mode"] == "enable"
    assert ui_section["reduce_motion_dispatch"] is True
    assert ui_section["reduce_motion_logging"] is True



def test_metrics_service_core_config_applies_settings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_dir = tmp_path / "conf"
    config_dir.mkdir()
    logs_dir = config_dir / "logs"
    logs_dir.mkdir()
    telemetry_jsonl = logs_dir / "telemetry.jsonl"
    telemetry_jsonl.write_text("{}\n", encoding="utf-8")
    ui_alerts_jsonl = logs_dir / "ui_alerts.jsonl"
    ui_alerts_jsonl.write_text("[]\n", encoding="utf-8")

    certs_dir = config_dir / "certs"
    certs_dir.mkdir()
    cert_path = certs_dir / "server.crt"
    key_path = certs_dir / "server.key"
    ca_path = certs_dir / "clients.pem"
    cert_path.write_text("cert", encoding="utf-8")
    key_path.write_text("key", encoding="utf-8")
    ca_path.write_text("ca", encoding="utf-8")

    config_path = config_dir / "core.yaml"
    config_path.write_text(
        textwrap.dedent(
            """
            risk_profiles:
              balanced:
                max_daily_loss_pct: 0.02
                max_position_pct: 0.05
                target_volatility: 0.10
                max_leverage: 2.0
                stop_loss_atr_multiple: 2.0
                max_open_positions: 4
                hard_drawdown_pct: 0.20
            environments:
              paper_env:
                exchange: binance
                environment: paper
                keychain_key: paper_key
                data_cache_path: var/data
                risk_profile: balanced
            runtime:
              metrics_service:
                enabled: true
                host: 0.0.0.0
                port: 60123
                history_size: 2048
                log_sink: false
                jsonl_path: logs/telemetry.jsonl
                jsonl_fsync: true
                ui_alerts_jsonl_path: logs/ui_alerts.jsonl
                tls:
                  enabled: true
                  certificate_path: certs/server.crt
                  private_key_path: certs/server.key
                  client_ca_path: certs/clients.pem
                  require_client_auth: true
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )

    dummy_server = _make_dummy_server(
        address="127.0.0.1:60123",
        jsonl_path=telemetry_jsonl,
        history_size=2048,
        tls_enabled=True,
        require_client_auth=True,
        logging_sink_enabled=False,
    )

    captured_kwargs: dict[str, object] = {}

    monkeypatch.setattr(run_metrics_service, "METRICS_RUNTIME_AVAILABLE", True)

    def fake_build_server(**kwargs):
        nonlocal captured_kwargs
        captured_kwargs = kwargs
        ui_path = kwargs.get("ui_alerts_jsonl_path")
        sink_meta = dummy_server.runtime_metadata.get("ui_alerts_sink", {})
        if ui_path:
            sink_meta.update({"active": True, "path": str(ui_path)})
        else:
            sink_meta.setdefault("active", False)
            sink_meta.setdefault("path", None)
        dummy_server.runtime_metadata["ui_alerts_sink"] = sink_meta
        return dummy_server

    monkeypatch.setattr(run_metrics_service, "_build_server", fake_build_server)

    exit_code = run_metrics_service.main([
        "--core-config",
        str(config_path),
        "--print-config-plan",
    ])
    assert exit_code == 0

    captured = capsys.readouterr().out.strip()
    assert captured
    payload = json.loads(captured)

    assert payload["host"] == "0.0.0.0"
    assert payload["port"] == 60123
    assert payload["history_size"] == 2048
    assert payload["logging_sink_enabled"] is False
    assert payload["jsonl_sink"]["configured"] is True
    assert payload["jsonl_sink"]["file"]["exists"] is True
    assert payload["ui_alerts"]["configured"] is True
    assert payload["ui_alerts"]["disabled"] is False
    assert payload["ui_alerts"]["file"]["exists"] is True

    runtime_state = payload["runtime_state"]
    assert runtime_state["available"] is True
    assert runtime_state["reason"] is None
    assert runtime_state["logging_sink_active"] is False
    assert runtime_state["jsonl_sink"]["path"].endswith("telemetry.jsonl")
    assert runtime_state["ui_alerts_sink"]["path"].endswith("ui_alerts.jsonl")
    assert runtime_state["ui_alerts_sink"].get("active") is True
    assert runtime_state["tls"]["enabled"] is True
    assert runtime_state["tls"]["require_client_auth"] is True

    core_section = payload["core_config"]
    assert core_section["cli_argument"].endswith("core.yaml")
    metrics_section = core_section["metrics_service"]
    assert metrics_section["applied_sources"]["host"] == "core_config"
    assert metrics_section["applied_sources"]["jsonl_path"] == "core_config"
    assert metrics_section["applied_sources"]["log_sink"] == "core_config"
    values = metrics_section["values"]
    assert values["jsonl_path"].endswith("telemetry.jsonl")
    assert values["jsonl_file"]["exists"] is True
    assert values["ui_alerts_file"]["path"].endswith("ui_alerts.jsonl")
    tls_values = values["tls"]
    assert tls_values["require_client_auth"] is True
    assert tls_values["certificate_file"]["exists"] is True
    assert tls_values["private_key_file"]["exists"] is True
    assert tls_values["client_ca_file"]["exists"] is True

    assert captured_kwargs["host"] == "0.0.0.0"
    assert captured_kwargs["port"] == 60123
    assert captured_kwargs["history_size"] == 2048
    assert captured_kwargs["enable_logging_sink"] is False
    assert Path(captured_kwargs["jsonl_path"]) == telemetry_jsonl
    assert captured_kwargs["enable_ui_alerts"] is True
    assert Path(captured_kwargs["ui_alerts_jsonl_path"]) == ui_alerts_jsonl
    tls_kwargs = captured_kwargs["tls_config"]
    assert Path(tls_kwargs["certificate_path"]) == cert_path
    assert Path(tls_kwargs["private_key_path"]) == key_path
    assert Path(tls_kwargs["client_ca_path"]) == ca_path
    assert tls_kwargs["require_client_auth"] is True



def test_print_config_plan_without_runtime(monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    monkeypatch.setattr(run_metrics_service, "METRICS_RUNTIME_AVAILABLE", False)
    monkeypatch.setattr(
        run_metrics_service,
        "METRICS_RUNTIME_UNAVAILABLE_MESSAGE",
        "brak gRPC",
    )

    exit_code = run_metrics_service.main(["--print-config-plan"])
    assert exit_code == 0

    captured = capsys.readouterr().out.strip()
    payload = json.loads(captured)
    runtime_state = payload["runtime_state"]
    assert runtime_state["available"] is False
    assert runtime_state["reason"] == "metrics_runtime_unavailable"
    assert runtime_state["details"] == "brak gRPC"
    assert runtime_state["jsonl_sink"]["path"] is None
    assert runtime_state["ui_alerts_sink"]["path"] is None
    assert runtime_state["ui_alerts_sink"].get("disabled") is False
    assert payload["ui_alerts"]["configured"] is False



def test_config_plan_jsonl_without_runtime(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    destination = tmp_path / "audit.jsonl"
    monkeypatch.setattr(run_metrics_service, "METRICS_RUNTIME_AVAILABLE", False)
    monkeypatch.setattr(
        run_metrics_service,
        "METRICS_RUNTIME_UNAVAILABLE_MESSAGE",
        "missing grpc runtime",
    )

    exit_code = run_metrics_service.main(["--config-plan-jsonl", str(destination)])
    assert exit_code == 2

    contents = destination.read_text(encoding="utf-8").strip().splitlines()
    assert len(contents) == 1
    payload = json.loads(contents[0])
    runtime_state = payload["runtime_state"]
    assert runtime_state["available"] is False
    assert runtime_state["reason"] == "metrics_runtime_unavailable"
    assert runtime_state["details"] == "missing grpc runtime"
    assert runtime_state["ui_alerts_sink"]["path"] is None
    assert payload["ui_alerts"]["configured"] is False



def test_metrics_service_risk_profile_overrides(monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    monkeypatch.setattr(run_metrics_service, "METRICS_RUNTIME_AVAILABLE", False)
    monkeypatch.setattr(
        run_metrics_service,
        "METRICS_RUNTIME_UNAVAILABLE_MESSAGE",
        "no runtime",
    )

    exit_code = run_metrics_service.main([
        "--print-config-plan",
        "--ui-alerts-risk-profile",
        "conservative",
    ])
    assert exit_code == 0

    payload = json.loads(capsys.readouterr().out.strip())
    ui_section = payload["ui_alerts"]
    assert ui_section["risk_profile"]["name"] == "conservative"
    # preserved additional assertions from other branch
    assert ui_section["risk_profile_summary"]["name"] == "conservative"
    assert ui_section["risk_profile_summary"]["severity_min"] == "warning"
    assert ui_section["reduce_motion_severity_active"] == "critical"
    assert ui_section["overlay_severity_exceeded"] == "critical"
    assert ui_section["overlay_critical_threshold"] == 1
    assert ui_section["jank_severity_spike"] == "warning"
    assert ui_section["performance_severity_recovered"] == "notice"
    assert ui_section["performance_cpu_warning_percent"] == 75.0
    runtime_config = payload["runtime_state"]["ui_alerts_sink"]["config"]
    assert runtime_config["risk_profile"]["name"] == "conservative"
    assert runtime_config["risk_profile"]["severity_min"] == "warning"
    # preserved additional assertions from other branch
    assert runtime_config["risk_profile_summary"]["name"] == "conservative"
    assert runtime_config["risk_profile_summary"]["severity_min"] == "warning"
    assert runtime_config["performance_event_to_frame_warning_ms"] == 45.0
    assert runtime_config["performance_gpu_critical_percent"] == 80.0
    runtime_applied = runtime_config["risk_profile"].get("applied_overrides")
    if runtime_applied is not None:
        assert runtime_applied["overlay_alert_severity_critical"] == "critical"
        assert runtime_applied["jank_alert_severity_critical"] == "error"
        assert runtime_applied["performance_severity_recovered"] == "notice"


def test_metrics_service_cli_performance_flags(monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    monkeypatch.setattr(run_metrics_service, "METRICS_RUNTIME_AVAILABLE", False)
    monkeypatch.setattr(run_metrics_service, "METRICS_RUNTIME_UNAVAILABLE_MESSAGE", "no runtime")

    exit_code = run_metrics_service.main(
        [
            "--print-config-plan",
            "--ui-alerts-performance-mode",
            "jsonl",
            "--ui-alerts-performance-category",
            "ui.perf.cli",
            "--ui-alerts-performance-warning-severity",
            "notice",
            "--ui-alerts-performance-critical-severity",
            "error",
            "--ui-alerts-performance-recovered-severity",
            "info",
            "--ui-alerts-performance-event-to-frame-warning-ms",
            "70",
            "--ui-alerts-performance-event-to-frame-critical-ms",
            "110",
            "--ui-alerts-performance-cpu-warning-percent",
            "72",
            "--ui-alerts-performance-cpu-critical-percent",
            "90",
            "--ui-alerts-performance-gpu-warning-percent",
            "60",
            "--ui-alerts-performance-gpu-critical-percent",
            "84",
            "--ui-alerts-performance-ram-warning-megabytes",
            "2048",
            "--ui-alerts-performance-ram-critical-megabytes",
            "4096",
        ]
    )
    assert exit_code == 0

    payload = json.loads(capsys.readouterr().out.strip())
    ui_section = payload["ui_alerts"]
    assert ui_section["performance_mode"] == "jsonl"
    assert ui_section["performance_category"] == "ui.perf.cli"
    assert ui_section["performance_event_to_frame_warning_ms"] == 70.0
    assert ui_section["performance_cpu_critical_percent"] == 90.0
    assert ui_section["performance_ram_warning_megabytes"] == 2048.0

    runtime_config = payload["runtime_state"]["ui_alerts_sink"]["config"]
    assert runtime_config["performance_mode"] == "jsonl"
    assert runtime_config["performance_severity_warning"] == "notice"
    assert runtime_config["performance_event_to_frame_critical_ms"] == 110.0
    assert runtime_config["performance_gpu_warning_percent"] == 60.0
    assert runtime_config["performance_ram_critical_megabytes"] == 4096.0



def test_metrics_service_risk_profiles_file_cli(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys
) -> None:
    monkeypatch.setattr(run_metrics_service, "METRICS_RUNTIME_AVAILABLE", False)
    monkeypatch.setattr(run_metrics_service, "METRICS_RUNTIME_UNAVAILABLE_MESSAGE", "no runtime")

    profiles_path = tmp_path / "telemetry_profiles.json"
    profiles_path.write_text(
        json.dumps(
            {
                "risk_profiles": {
                    "custom": {
                        "metrics_service_overrides": {
                            "ui_alerts_reduce_active_severity": "error",
                            "ui_alerts_overlay_critical_threshold": 5,
                        },
                        "severity_min": "notice",
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    exit_code = run_metrics_service.main(
        [
            "--print-config-plan",
            "--risk-profiles-file",
            str(profiles_path),
            "--ui-alerts-risk-profile",
            "custom",
        ]
    )
    assert exit_code == 0

    payload = json.loads(capsys.readouterr().out.strip())
    ui_section = payload["ui_alerts"]
    assert ui_section["risk_profile"]["name"] == "custom"
    assert ui_section["risk_profile"]["severity_min"] == "notice"
    # preserved additional assertions from other branch
    assert ui_section["risk_profile_summary"]["name"] == "custom"
    assert ui_section["risk_profile_summary"]["severity_min"] == "notice"
    assert ui_section["reduce_motion_severity_active"] == "error"
    assert ui_section["overlay_critical_threshold"] == 5
    file_meta = ui_section["risk_profiles_file"]
    assert file_meta["path"] == str(profiles_path)
    assert "custom" in file_meta["registered_profiles"]

    runtime_config = payload["runtime_state"]["ui_alerts_sink"]["config"]
    assert runtime_config["risk_profile"]["name"] == "custom"
    # preserved additional assertion
    assert runtime_config["risk_profile_summary"]["name"] == "custom"
    runtime_file_meta = runtime_config["risk_profiles_file"]
    assert runtime_file_meta["path"] == str(profiles_path)



def test_metrics_service_risk_profiles_directory(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys) -> None:
    monkeypatch.setattr(run_metrics_service, "METRICS_RUNTIME_AVAILABLE", False)
    monkeypatch.setattr(run_metrics_service, "METRICS_RUNTIME_UNAVAILABLE_MESSAGE", "no runtime")

    profiles_dir = tmp_path / "profiles"
    profiles_dir.mkdir()
    (profiles_dir / "ops.json").write_text(
        json.dumps(
            {
                "risk_profiles": {
                    "ops_dir": {
                        "metrics_service_overrides": {
                            "ui_alerts_overlay_critical_threshold": 4
                        },
                        "severity_min": "warning",
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    (profiles_dir / "lab.yaml").write_text(
        "risk_profiles:\n  lab_dir:\n    severity_min: notice\n",
        encoding="utf-8",
    )

    exit_code = run_metrics_service.main(
        [
            "--print-config-plan",
            "--risk-profiles-file",
            str(profiles_dir),
            "--ui-alerts-risk-profile",
            "ops_dir",
        ]
    )
    assert exit_code == 0

    payload = json.loads(capsys.readouterr().out.strip())
    ui_section = payload["ui_alerts"]
    assert ui_section["risk_profile"]["name"] == "ops_dir"
    # preserved additional assertion from other branch
    assert ui_section["risk_profile_summary"]["name"] == "ops_dir"
    file_meta = ui_section["risk_profiles_file"]
    assert file_meta["type"] == "directory"
    assert file_meta["path"] == str(profiles_dir)
    assert "ops_dir" in file_meta["registered_profiles"]
    assert any(entry["path"].endswith("ops.json") for entry in file_meta["files"])

    runtime_config = payload["runtime_state"]["ui_alerts_sink"]["config"]
    assert runtime_config["risk_profile"]["name"] == "ops_dir"
    # preserved additional assertion
    assert runtime_config["risk_profile_summary"]["name"] == "ops_dir"
    runtime_file_meta = runtime_config["risk_profiles_file"]
    assert runtime_file_meta["type"] == "directory"
    assert runtime_file_meta["path"] == str(profiles_dir)



def _find_env_entry(entries: list[dict[str, object]], option: str) -> dict[str, object]:
    for entry in entries:
        if entry.get("option") == option:
            return entry
    raise AssertionError(f"Nie znaleziono wpisu dla opcji {option}")
def test_ui_alert_cli_options_forwarded(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    captured_kwargs: dict[str, object] = {}

    def fake_build_server(**kwargs):
        nonlocal captured_kwargs
        captured_kwargs = kwargs
        return RuntimeServerStub("127.0.0.1:6000")

    monkeypatch.setattr(run_metrics_service, "_build_server", fake_build_server)
    monkeypatch.setattr(run_metrics_service, "METRICS_RUNTIME_AVAILABLE", True)

    alerts_path = tmp_path / "alerts.jsonl"
    audit_dir = tmp_path / "audit"
    args = [
        "--host",
        "127.0.0.1",
        "--port",
        "0",
        "--ui-alerts-jsonl",
        str(alerts_path),
        "--ui-alerts-reduce-mode",
        "disable",
        "--ui-alerts-overlay-mode",
        "enable",
        "--ui-alerts-reduce-category",
        "ops.ui.reduce",
        "--ui-alerts-reduce-active-severity",
        "critical",
        "--ui-alerts-reduce-recovered-severity",
        "notice",
        "--ui-alerts-overlay-category",
        "ops.ui.overlay",
        "--ui-alerts-overlay-exceeded-severity",
        "warning",
        "--ui-alerts-overlay-recovered-severity",
        "info",
        "--ui-alerts-overlay-critical-severity",
        "major",
        "--ui-alerts-overlay-critical-threshold",
        "3",
        "--ui-alerts-jank-mode",
        "enable",
        "--ui-alerts-jank-category",
        "ops.ui.jank",
        "--ui-alerts-jank-spike-severity",
        "major",
        "--ui-alerts-jank-critical-severity",
        "critical",
        "--ui-alerts-jank-critical-over-ms",
        "8.5",
        "--ui-alerts-audit-dir",
        str(audit_dir),
        "--ui-alerts-audit-pattern",
        "custom-%Y.jsonl",
        "--ui-alerts-audit-retention-days",
        "7",
        "--ui-alerts-audit-fsync",
        "--shutdown-after",
        "0.0",
    ]

    exit_code = run_metrics_service.main(args)
    assert exit_code == 0
    ui_config = captured_kwargs.get("ui_alerts_config")
    assert ui_config is not None
    assert ui_config["reduce_mode"] == "disable"
    assert ui_config["overlay_mode"] == "enable"
    assert ui_config["reduce_motion_alerts"] is False
    assert ui_config["overlay_alerts"] is True
    assert ui_config["reduce_motion_logging"] is False
    assert ui_config["overlay_logging"] is True
    assert ui_config["reduce_motion_category"] == "ops.ui.reduce"
    assert ui_config["reduce_motion_severity_active"] == "critical"
    assert ui_config["reduce_motion_severity_recovered"] == "notice"
    assert ui_config["overlay_category"] == "ops.ui.overlay"
    assert ui_config["overlay_severity_critical"] == "major"
    assert ui_config["overlay_critical_threshold"] == 3
    assert ui_config["jank_mode"] == "enable"
    assert ui_config["jank_alerts"] is True
    assert ui_config["jank_logging"] is True
    assert ui_config["jank_category"] == "ops.ui.jank"
    assert ui_config["jank_severity_spike"] == "major"
    assert ui_config["jank_severity_critical"] == "critical"
    assert ui_config["jank_critical_over_ms"] == pytest.approx(8.5)
    assert ui_config["audit"]["requested"] == "auto"
    assert ui_config["audit"]["backend"] == "file"
    assert ui_config["audit"]["directory"] == str(audit_dir)
    assert ui_config["audit"]["pattern"] == "custom-%Y.jsonl"
    assert ui_config["audit"]["retention_days"] == 7
    assert ui_config["audit"]["fsync"] is True
    assert captured_kwargs.get("ui_alerts_audit_dir") == audit_dir
    assert captured_kwargs.get("ui_alerts_audit_pattern") == "custom-%Y.jsonl"
    assert captured_kwargs.get("ui_alerts_audit_retention_days") == 7
    # preserved both styles of assertion to keep full content
    assert captured_kwargs.get("ui_alerts_audit_fsync") is True
    assert captured_kwargs.get("ui_alerts_audit_fsync") == True



def test_ui_alerts_audit_memory_note_when_file_backend_missing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    captured_kwargs: dict[str, object] = {}

    def fake_build_server(**kwargs):
        nonlocal captured_kwargs
        captured_kwargs = kwargs
        return RuntimeServerStub("127.0.0.1:6000")

    monkeypatch.setattr(run_metrics_service, "_build_server", fake_build_server)
    monkeypatch.setattr(run_metrics_service, "METRICS_RUNTIME_AVAILABLE", True)
    monkeypatch.setattr(run_metrics_service, "FileAlertAuditLog", None)

    alerts_path = tmp_path / "alerts.jsonl"
    audit_dir = tmp_path / "audit"
    args = [
        "--host",
        "127.0.0.1",
        "--port",
        "0",
        "--ui-alerts-jsonl",
        str(alerts_path),
        "--ui-alerts-audit-dir",
        str(audit_dir),
        "--ui-alerts-audit-fsync",
        "--shutdown-after",
        "0.0",
    ]

    exit_code = run_metrics_service.main(args)
    assert exit_code == 0

    ui_config = captured_kwargs.get("ui_alerts_config")
    assert ui_config is not None
    audit_config = ui_config.get("audit")
    assert audit_config is not None
    assert audit_config["requested"] == "auto"
    assert audit_config["backend"] == "memory"
    assert audit_config.get("note") == "file_backend_unavailable"



def test_environment_override_applied_without_runtime(monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    monkeypatch.setattr(run_metrics_service, "METRICS_RUNTIME_AVAILABLE", False)
    monkeypatch.setenv("RUN_METRICS_SERVICE_HOST", "0.0.0.0")

    exit_code = run_metrics_service.main(["--print-config-plan"])
    assert exit_code == 0

    payload = json.loads(capsys.readouterr().out.strip())
    assert payload["host"] == "0.0.0.0"
    env_section = payload.get("environment_overrides")
    assert env_section is not None
    host_entry = _find_env_entry(env_section["entries"], "host")
    assert host_entry["applied"] is True
    assert host_entry["parsed_value"] == "0.0.0.0"
    parameter_sources = payload.get("parameter_sources", {})
    assert parameter_sources.get("host") == "env"
    assert parameter_sources.get("fail_on_security_warnings") == "default"
    security = get_security_section(payload)
    assert security["enabled"] is False
    assert security["source"] == "default"
    assert security["parameter_source"] == "default"



def test_environment_override_ignored_by_cli(monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    monkeypatch.setattr(run_metrics_service, "METRICS_RUNTIME_AVAILABLE", False)
    monkeypatch.setenv("RUN_METRICS_SERVICE_PORT", "7777")

    exit_code = run_metrics_service.main(["--port", "1234", "--print-config-plan"])
    assert exit_code == 0

    payload = json.loads(capsys.readouterr().out.strip())
    assert payload["port"] == 1234
    env_section = payload.get("environment_overrides")
    assert env_section is not None
    port_entry = _find_env_entry(env_section["entries"], "port")
    assert port_entry["applied"] is False
    assert port_entry["reason"] == "cli_override"
    parameter_sources = payload.get("parameter_sources", {})
    assert parameter_sources.get("port") == "cli"
    assert parameter_sources.get("fail_on_security_warnings") == "default"
    security = get_security_section(payload)
    assert security["enabled"] is False
    assert security["source"] == "default"
    assert security["parameter_source"] == "default"



def test_environment_override_jsonl_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys) -> None:
    monkeypatch.setattr(run_metrics_service, "METRICS_RUNTIME_AVAILABLE", False)
    jsonl_path = tmp_path / "env_metrics.jsonl"
    monkeypatch.setenv("RUN_METRICS_SERVICE_JSONL", str(jsonl_path))

    exit_code = run_metrics_service.main(["--print-config-plan"])
    assert exit_code == 0

    payload = json.loads(capsys.readouterr().out.strip())
    assert payload["jsonl_sink"]["configured"] is True
    assert payload["jsonl_sink"]["path"] == str(jsonl_path)
    env_section = payload.get("environment_overrides")
    assert env_section is not None
    jsonl_entry = _find_env_entry(env_section["entries"], "jsonl_path")
    assert jsonl_entry["applied"] is True
    assert jsonl_entry["parsed_value"] == str(jsonl_path)
    parameter_sources = payload.get("parameter_sources", {})
    assert parameter_sources.get("jsonl_path") == "env"
    assert parameter_sources.get("fail_on_security_warnings") == "default"
    security = get_security_section(payload)
    assert security["enabled"] is False
    assert security["source"] == "default"



def test_environment_override_disable_jsonl(monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    monkeypatch.setattr(run_metrics_service, "METRICS_RUNTIME_AVAILABLE", False)
    monkeypatch.setenv("RUN_METRICS_SERVICE_JSONL", "none")

    exit_code = run_metrics_service.main(["--print-config-plan"])
    assert exit_code == 0

    payload = json.loads(capsys.readouterr().out.strip())
    assert payload["jsonl_sink"]["configured"] is False
    env_section = payload.get("environment_overrides")
    assert env_section is not None
    jsonl_entry = _find_env_entry(env_section["entries"], "jsonl_path")
    assert jsonl_entry["parsed_value"] is None
    assert jsonl_entry["note"] == "jsonl_disabled"
    parameter_sources = payload.get("parameter_sources", {})
    assert parameter_sources.get("jsonl_path") == "env_disabled"
    assert parameter_sources.get("fail_on_security_warnings") == "default"
    security = get_security_section(payload)
    assert security["enabled"] is False
    assert security["source"] == "default"



def test_environment_override_fail_on_security_warnings(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys,
) -> None:
    monkeypatch.setattr(run_metrics_service, "METRICS_RUNTIME_AVAILABLE", False)
    monkeypatch.setenv("RUN_METRICS_SERVICE_FAIL_ON_SECURITY_WARNINGS", "true")
    jsonl_path = tmp_path / "metrics" / "telemetry.jsonl"
    monkeypatch.setenv("RUN_METRICS_SERVICE_JSONL", str(jsonl_path))

    exit_code = run_metrics_service.main(["--print-config-plan"])
    assert exit_code == 3

    payload = json.loads(capsys.readouterr().out.strip())
    env_section = payload.get("environment_overrides")
    assert env_section is not None
    entry = _find_env_entry(env_section["entries"], "fail_on_security_warnings")
    assert entry["applied"] is True
    assert entry["parsed_value"] is True
    parameter_sources = payload.get("parameter_sources", {})
    assert parameter_sources.get("fail_on_security_warnings") == "env"
    security = get_security_section(payload)
    assert security["enabled"] is True



def test_environment_override_ui_alerts_audit(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys) -> None:
    monkeypatch.setattr(run_metrics_service, "METRICS_RUNTIME_AVAILABLE", False)
    audit_dir = tmp_path / "ui_audit"
    monkeypatch.setenv("RUN_METRICS_SERVICE_UI_ALERTS_AUDIT_DIR", str(audit_dir))
    monkeypatch.setenv("RUN_METRICS_SERVICE_UI_ALERTS_AUDIT_PATTERN", "ops-%Y.jsonl")
    monkeypatch.setenv("RUN_METRICS_SERVICE_UI_ALERTS_AUDIT_RETENTION_DAYS", "30")
    monkeypatch.setenv("RUN_METRICS_SERVICE_UI_ALERTS_AUDIT_FSYNC", "true")

    exit_code = run_metrics_service.main(["--print-config-plan"])
    assert exit_code == 0

    payload = json.loads(capsys.readouterr().out.strip())
    audit_info = payload["ui_alerts"]["audit"]
    assert audit_info["requested"] == "auto"
    assert audit_info["backend"] == "file"
    assert audit_info["directory"] == str(audit_dir)
    assert audit_info["pattern"] == "ops-%Y.jsonl"
    assert audit_info["retention_days"] == 30
    assert audit_info["fsync"] is True
    env_entries = payload.get("environment_overrides", {}).get("entries", [])
    dir_entry = _find_env_entry(env_entries, "ui_alerts_audit_dir")
    assert dir_entry["applied"] is True
    assert dir_entry["parsed_value"] == str(audit_dir)
    pattern_entry = _find_env_entry(env_entries, "ui_alerts_audit_pattern")
    assert pattern_entry["parsed_value"] == "ops-%Y.jsonl"
    retention_entry = _find_env_entry(env_entries, "ui_alerts_audit_retention_days")
    assert retention_entry["parsed_value"] == 30
    fsync_entry = _find_env_entry(env_entries, "ui_alerts_audit_fsync")
    assert fsync_entry["parsed_value"] is True
    parameter_sources = payload.get("parameter_sources", {})
    assert parameter_sources.get("ui_alerts_audit_dir") == "env"
    assert parameter_sources.get("ui_alerts_audit_pattern") == "env"
    assert parameter_sources.get("ui_alerts_audit_retention_days") == "env"
    assert parameter_sources.get("ui_alerts_audit_fsync") == "env"



def test_print_plan_marks_memory_audit_when_file_backend_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys
) -> None:
    monkeypatch.setattr(run_metrics_service, "METRICS_RUNTIME_AVAILABLE", False)
    monkeypatch.setattr(run_metrics_service, "FileAlertAuditLog", None)

    audit_dir = tmp_path / "audit"
    exit_code = run_metrics_service.main(
        ["--ui-alerts-audit-dir", str(audit_dir), "--print-config-plan"]
    )
    assert exit_code == 0

    payload = json.loads(capsys.readouterr().out.strip())
    audit_section = payload["ui_alerts"]["audit"]
    assert audit_section["requested"] == "auto"
    assert audit_section["backend"] == "memory"
    assert audit_section.get("note") == "file_backend_unavailable"
    assert "directory" not in audit_section



def test_build_server_uses_memory_audit_when_file_backend_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    captured_kwargs: dict[str, object] = {}

    def fake_create_metrics_server(**kwargs):
        nonlocal captured_kwargs
        captured_kwargs = kwargs
        return RuntimeServerStub("127.0.0.1:50100")

    monkeypatch.setattr(run_metrics_service, "create_metrics_server", fake_create_metrics_server)
    monkeypatch.setattr(run_metrics_service, "METRICS_RUNTIME_AVAILABLE", True)
    monkeypatch.setattr(run_metrics_service, "JsonlSink", None)
    monkeypatch.setattr(run_metrics_service, "UiTelemetryAlertSink", SinkStub)
    monkeypatch.setattr(run_metrics_service, "DefaultAlertRouter", RouterStub)
    monkeypatch.setattr(run_metrics_service, "InMemoryAlertAuditLog", MemoryAuditStub)
    monkeypatch.setattr(run_metrics_service, "FileAlertAuditLog", None)

    server = run_metrics_service._build_server(
        host="127.0.0.1",
        port=0,
        history_size=32,
        enable_logging_sink=True,
        jsonl_path=None,
        jsonl_fsync=False,
        auth_token=None,
        enable_ui_alerts=True,
        ui_alerts_jsonl_path=tmp_path / "ui_alerts.jsonl",
        ui_alerts_options=None,
        ui_alerts_config=None,
        ui_alerts_audit_dir=tmp_path / "audit",
        ui_alerts_audit_pattern=None,
        ui_alerts_audit_retention_days=None,
        ui_alerts_audit_fsync=False,
        extra_sinks=(),
        tls_config=None,
    )

    assert server is not None
    audit_config = captured_kwargs.get("ui_alerts_config", {}).get("audit")
    assert audit_config is not None
    assert audit_config["backend"] == "memory"
    assert audit_config.get("note") == "file_backend_unavailable"



def test_environment_override_invalid_boolean(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(run_metrics_service, "METRICS_RUNTIME_AVAILABLE", False)
    monkeypatch.setenv("RUN_METRICS_SERVICE_NO_LOG_SINK", "maybe")

    with pytest.raises(SystemExit):
        run_metrics_service.main([])



def test_ui_alerts_audit_backend_memory_flag(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys) -> None:
    monkeypatch.setattr(run_metrics_service, "METRICS_RUNTIME_AVAILABLE", False)
    exit_code = run_metrics_service.main(
        [
            "--ui-alerts-audit-backend",
            "memory",
            "--ui-alerts-audit-dir",
            str(tmp_path / "audit"),
            "--print-config-plan",
        ]
    )
    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out.strip())
    audit_info = payload["ui_alerts"]["audit"]
    assert audit_info["requested"] == "memory"
    assert audit_info["backend"] == "memory"
    assert audit_info.get("note") == "directory_ignored_memory_backend"
    assert "directory" not in audit_info



def test_ui_alerts_audit_backend_file_requires_directory(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(run_metrics_service, "METRICS_RUNTIME_AVAILABLE", False)
    with pytest.raises(SystemExit):
        run_metrics_service.main(["--ui-alerts-audit-backend", "file", "--print-config-plan"])
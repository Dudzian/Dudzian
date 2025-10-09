from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Mapping

import pytest
import yaml

from scripts import run_trading_stub_server


@pytest.fixture(autouse=True)
def stubbed_datasets(monkeypatch: pytest.MonkeyPatch) -> None:
    """Zapewnia lekkie atrapowe dataset-y bez generowanych stubów gRPC."""

    def _make_instrument(data: dict[str, object]) -> SimpleNamespace:
        defaults = {
            "exchange": "TEST",
            "symbol": "FOO/BAR",
            "venue_symbol": "FOOBAR",
            "quote_currency": "BAR",
            "base_currency": "FOO",
        }
        defaults.update(data)
        return SimpleNamespace(**defaults)

    def _make_granularity(value: str | None) -> SimpleNamespace:
        return SimpleNamespace(iso8601_duration=value or "PT1M")

    def _make_candle(
        instrument: SimpleNamespace,
        granularity: SimpleNamespace,
        data: dict[str, object],
        sequence: int,
    ) -> SimpleNamespace:
        payload = {
            "instrument": instrument,
            "granularity": granularity,
            "open_time": data.get("open_time", "1970-01-01T00:00:00Z"),
            "open": data.get("open", 0.0),
            "high": data.get("high", 0.0),
            "low": data.get("low", 0.0),
            "close": data.get("close", 0.0),
            "volume": data.get("volume", 0.0),
            "closed": data.get("closed", True),
            "sequence": data.get("sequence", sequence),
        }
        return SimpleNamespace(**payload)

    def fake_build_default_dataset():
        dataset = run_trading_stub_server.InMemoryTradingDataset()
        instrument = _make_instrument({"exchange": "BINANCE", "symbol": "BTC/USDT"})
        granularity = _make_granularity("PT1M")
        dataset.add_history(
            instrument,
            granularity,
            [
                _make_candle(
                    instrument,
                    granularity,
                    {
                        "open_time": "2024-01-01T00:00:00Z",
                        "open": 1.0,
                        "high": 2.0,
                        "low": 0.5,
                        "close": 1.5,
                        "volume": 10,
                    },
                    1,
                )
            ],
        )
        dataset.performance_guard.update({"fps_target": 60, "overlay_allowed": 4})
        dataset.metrics = [SimpleNamespace(notes="{}", fps=60.0, generated_at=None)]
        dataset.health = SimpleNamespace(version="stub", git_commit="dev", started_at="2024-01-01T00:00:00Z")
        return dataset

    def fake_load_dataset_from_yaml(path: str | Path):
        dataset = run_trading_stub_server.InMemoryTradingDataset()
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        for item in raw.get("market_data", []):
            instrument = _make_instrument(item.get("instrument", {}))
            granularity = _make_granularity(item.get("granularity"))
            candles = [
                _make_candle(instrument, granularity, candle, idx + 1)
                for idx, candle in enumerate(item.get("candles", []))
            ]
            dataset.add_history(instrument, granularity, candles)
        return dataset

    monkeypatch.setattr(run_trading_stub_server, "build_default_dataset", fake_build_default_dataset)
    monkeypatch.setattr(run_trading_stub_server, "load_dataset_from_yaml", fake_load_dataset_from_yaml)


class _DummyServer:
    def __init__(
        self,
        dataset,
        host: str,
        port: int,
        max_workers: int,
        *,
        stream_repeat: bool = False,
        stream_interval: float = 0.0,
    ) -> None:
        self.dataset = dataset
        self.host = host
        self.port = port
        self.max_workers = max_workers
        self.started = False
        self.stopped_with = None
        self.wait_timeouts: list[float | None] = []
        self.stream_repeat = stream_repeat
        self.stream_interval = stream_interval

    @property
    def address(self) -> str:
        return f"{self.host}:{self.port or 12345}"

    def start(self) -> None:
        self.started = True

    def wait_for_termination(self, timeout: float | None = None) -> bool:
        self.wait_timeouts.append(timeout)
        return timeout is None

    def stop(self, grace: float | None = None) -> None:
        self.stopped_with = grace


class _DummyMetricsServer:
    def __init__(self) -> None:
        self.started = False
        self.stop_calls: list[float | None] = []
        self.address = "127.0.0.1:50061"

    def start(self) -> None:
        self.started = True

    def stop(self, grace: float | None = None) -> None:
        self.stop_calls.append(grace)

    def wait_for_termination(self, timeout: float | None = None) -> bool:
        return True


def test_runs_with_default_dataset(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    server = _DummyServer(None, "", 0, 0)

    def factory(dataset, host: str, port: int, max_workers: int, **kwargs):
        nonlocal server
        server = _DummyServer(dataset, host, port, max_workers, **kwargs)
        return server

    monkeypatch.setattr(run_trading_stub_server, "TradingStubServer", factory)

    exit_code = run_trading_stub_server.main([
        "--port",
        "0",
        "--shutdown-after",
        "0.01",
        "--print-address",
    ])

    assert exit_code == 0
    assert server.started is True
    assert server.stopped_with == 1.0
    # gdy timeout został użyty (0.01), wait powinien otrzymać tę wartość
    assert server.wait_timeouts == [0.01]
    assert server.dataset is not None
    assert len(server.dataset.history) >= 1
    captured = capsys.readouterr()
    assert (
        "Serwer stub wystartował" in captured.err
        or "Serwer stub wystartował" in captured.out
        or server.address in captured.out
    )


def test_uses_yaml_dataset_without_default(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    yaml_path = tmp_path / "dataset.yaml"
    yaml_path.write_text(
        """
market_data:
  - instrument:
      exchange: TEST
      symbol: FOO/BAR
      venue_symbol: FOOBAR
      quote_currency: BAR
      base_currency: FOO
    granularity: PT1M
    candles:
      - open_time: 2024-01-01T00:00:00Z
        open: 1.0
        high: 2.0
        low: 0.5
        close: 1.5
        volume: 10
""",
        encoding="utf-8",
    )

    server = _DummyServer(None, "", 0, 0)

    def factory(dataset, host: str, port: int, max_workers: int, **kwargs):
        nonlocal server
        server = _DummyServer(dataset, host, port, max_workers, **kwargs)
        return server

    monkeypatch.setattr(run_trading_stub_server, "TradingStubServer", factory)

    exit_code = run_trading_stub_server.main(
        [
            "--no-default-dataset",
            "--dataset",
            str(yaml_path),
            "--shutdown-after",
            "0.01",
        ]
    )

    assert exit_code == 0
    assert len(server.dataset.history) == 1
    key = next(iter(server.dataset.history))
    assert key[0] == "TEST"
    assert server.wait_timeouts == [0.01]


def test_keyboard_interrupt_stops_server(monkeypatch: pytest.MonkeyPatch) -> None:
    server = _DummyServer(None, "127.0.0.1", 5555, 4)

    def fake_wait(timeout: float | None = None) -> bool:
        raise KeyboardInterrupt

    server.wait_for_termination = fake_wait  # type: ignore[assignment]
    monkeypatch.setattr(run_trading_stub_server, "TradingStubServer", lambda *args, **kwargs: server)

    exit_code = run_trading_stub_server.main(["--shutdown-after", "0.5"])

    assert exit_code == 0
    assert server.stopped_with == 1.0


def test_stream_repeat_options(monkeypatch: pytest.MonkeyPatch) -> None:
    server = _DummyServer(None, "127.0.0.1", 0, 0)

    def factory(dataset, host: str, port: int, max_workers: int, **kwargs):
        nonlocal server
        server = _DummyServer(dataset, host, port, max_workers, **kwargs)
        return server

    monkeypatch.setattr(run_trading_stub_server, "TradingStubServer", factory)

    exit_code = run_trading_stub_server.main(
        [
            "--stream-repeat",
            "--stream-interval",
            "0.25",
            "--shutdown-after",
            "0.01",
        ]
    )

    assert exit_code == 0
    assert server.stream_repeat is True
    assert pytest.approx(server.stream_interval, rel=1e-6) == 0.25


def test_negative_stream_interval_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        run_trading_stub_server,
        "TradingStubServer",
        lambda *args, **kwargs: _DummyServer(None, "", 0, 0),
    )

    with pytest.raises(SystemExit):
        run_trading_stub_server.main(["--stream-interval", "-0.1"])


def test_metrics_server_optional(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    dummy_metrics = _DummyMetricsServer()
    created_kwargs: dict[str, object] = {}

    def fake_create_metrics_server(**kwargs):
        nonlocal created_kwargs
        created_kwargs = kwargs
        return dummy_metrics

    monkeypatch.setattr(run_trading_stub_server, "create_metrics_server", fake_create_metrics_server)
    monkeypatch.setattr(
        run_trading_stub_server,
        "JsonlSink",
        lambda path, fsync=False: ("jsonl", Path(path), fsync),
    )

    class DummyUiSink:
        def __init__(self, _router, jsonl_path=None):
            self.jsonl_path = jsonl_path

    monkeypatch.setattr(run_trading_stub_server, "UiTelemetryAlertSink", DummyUiSink)

    server = _DummyServer(None, "127.0.0.1", 0, 0)
    monkeypatch.setattr(run_trading_stub_server, "TradingStubServer", lambda *args, **kwargs: server)

    metrics_jsonl = tmp_path / "metrics.jsonl"
    alerts_jsonl = tmp_path / "alerts.jsonl"

    exit_code = run_trading_stub_server.main(
        [
            "--enable-metrics",
            "--metrics-port",
            "0",
            "--metrics-jsonl",
            str(metrics_jsonl),
            "--metrics-ui-alerts-jsonl",
            str(alerts_jsonl),
            "--shutdown-after",
            "0.01",
        ]
    )

    assert exit_code == 0
    assert dummy_metrics.started is True
    assert dummy_metrics.stop_calls and dummy_metrics.stop_calls[-1] == 1.0
    assert created_kwargs["host"] == "127.0.0.1"
    assert created_kwargs["history_size"] == 512
    sinks = created_kwargs["sinks"]
    assert isinstance(sinks, list)
    assert any(isinstance(sink, tuple) and sink[0] == "jsonl" for sink in sinks)
    assert any(isinstance(sink, DummyUiSink) for sink in sinks)


def test_metrics_tls_configuration(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    dummy_metrics = _DummyMetricsServer()
    created_kwargs: dict[str, object] = {}

    def fake_create_metrics_server(**kwargs):
        nonlocal created_kwargs
        created_kwargs = kwargs
        return dummy_metrics

    monkeypatch.setattr(run_trading_stub_server, "create_metrics_server", fake_create_metrics_server)
    monkeypatch.setattr(
        run_trading_stub_server,
        "TradingStubServer",
        lambda *args, **kwargs: _DummyServer(None, "127.0.0.1", 0, 0),
    )

    cert = tmp_path / "cert.pem"
    key = tmp_path / "key.pem"
    ca = tmp_path / "ca.pem"
    for path in (cert, key, ca):
        path.write_text("dummy", encoding="utf-8")

    exit_code = run_trading_stub_server.main(
        [
            "--enable-metrics",
            "--metrics-tls-cert",
            str(cert),
            "--metrics-tls-key",
            str(key),
            "--metrics-tls-client-ca",
            str(ca),
            "--metrics-tls-require-client-cert",
            "--shutdown-after",
            "0.01",
        ]
    )

    assert exit_code == 0
    tls = created_kwargs.get("tls_config")
    assert tls is not None
    assert tls["certificate_path"] == cert
    assert tls["private_key_path"] == key
    assert tls["client_ca_path"] == ca
    assert tls["require_client_auth"] is True


def test_metrics_tls_requires_key_and_cert(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        run_trading_stub_server,
        "TradingStubServer",
        lambda *args, **kwargs: _DummyServer(None, "127.0.0.1", 0, 0),
    )

    cert = tmp_path / "cert.pem"
    cert.write_text("dummy", encoding="utf-8")

    with pytest.raises(SystemExit):
        run_trading_stub_server.main([
            "--enable-metrics",
            "--metrics-tls-cert",
            str(cert),
            "--shutdown-after",
            "0.01",
        ])


def test_print_runtime_plan(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys) -> None:
    dataset_yaml = tmp_path / "dataset.yaml"
    dataset_yaml.write_text("market_data: []\n", encoding="utf-8")

    metrics_jsonl = tmp_path / "metrics.jsonl"
    tls_cert = tmp_path / "cert.pem"
    tls_key = tmp_path / "key.pem"
    tls_cert.write_text("cert", encoding="utf-8")
    tls_key.write_text("key", encoding="utf-8")

    def fail_factory(*args, **kwargs):  # pragma: no cover - nie powinno zostać wywołane
        raise AssertionError("Serwer nie powinien być instancjonowany podczas --print-runtime-plan")

    monkeypatch.setattr(run_trading_stub_server, "TradingStubServer", fail_factory)

    exit_code = run_trading_stub_server.main(
        [
            "--dataset",
            str(dataset_yaml),
            "--enable-metrics",
            "--metrics-jsonl",
            str(metrics_jsonl),
            "--metrics-tls-cert",
            str(tls_cert),
            "--metrics-tls-key",
            str(tls_key),
            "--print-runtime-plan",
            "--log-level",
            "warning",
        ]
    )

    assert exit_code == 0
    captured = capsys.readouterr()
    plan = json.loads(captured.out)
    assert plan["version"] == 2
    assert plan["server"]["host"] == "127.0.0.1"
    assert plan["server"]["port"] == 50051
    assert plan["dataset"]["include_default"] is True
    assert any(source["type"] == "file" for source in plan["dataset"]["sources"])
    metrics_plan = plan["metrics"]
    assert metrics_plan["enabled"] is True
    assert metrics_plan["jsonl"]["configured"] is True
    assert metrics_plan["jsonl"]["path"] == str(metrics_jsonl)
    assert metrics_plan["ui_alerts"]["path"].endswith("logs/ui_telemetry_alerts.jsonl")
    assert metrics_plan["tls"]["configured"] is True
    assert metrics_plan["tls"]["certificate"]["role"] == "tls_cert"
    environment_section = plan.get("environment", {})
    assert environment_section.get("overrides") == []
    parameter_sources = environment_section.get("parameter_sources", {})
    assert parameter_sources.get("metrics_jsonl_path") == "cli"
    assert parameter_sources.get("enable_metrics") == "cli"
    assert parameter_sources.get("fail_on_security_warnings") == "default"
    security = _get_security_section(plan)
    assert security["enabled"] is False
    assert security["source"] == "default"
    assert security["parameter_source"] == "default"


def test_fail_on_security_warnings_blocks_runtime_plan(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys,
) -> None:
    dataset_yaml = tmp_path / "dataset.yaml"
    dataset_yaml.write_text("market_data: []\n", encoding="utf-8")
    metrics_jsonl = tmp_path / "metrics.jsonl"

    monkeypatch.setattr(
        run_trading_stub_server,
        "TradingStubServer",
        lambda *args, **kwargs: _DummyServer(None, "127.0.0.1", 0, 0),
    )

    exit_code = run_trading_stub_server.main(
        [
            "--dataset",
            str(dataset_yaml),
            "--enable-metrics",
            "--metrics-jsonl",
            str(metrics_jsonl),
            "--print-runtime-plan",
            "--fail-on-security-warnings",
        ]
    )

    assert exit_code == 3
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    parameter_sources = payload.get("environment", {}).get("parameter_sources", {})
    assert parameter_sources.get("fail_on_security_warnings") == "cli"
    security = _get_security_section(payload)
    assert security["enabled"] is True
    assert security["source"] == "cli"
    assert security["parameter_source"] == "cli"
    assert security.get("environment_variable") is None


def test_runtime_plan_jsonl_written(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    plan_path = tmp_path / "runtime_plan.jsonl"
    dataset_yaml = tmp_path / "dataset.yaml"
    dataset_yaml.write_text("market_data: []\n", encoding="utf-8")
    metrics_jsonl = tmp_path / "metrics.jsonl"

    dummy_metrics = _DummyMetricsServer()

    def fake_create_metrics_server(**kwargs):
        return dummy_metrics

    monkeypatch.setattr(run_trading_stub_server, "create_metrics_server", fake_create_metrics_server)
    monkeypatch.setattr(
        run_trading_stub_server,
        "JsonlSink",
        lambda path, fsync=False: ("jsonl", Path(path), fsync),
    )

    class DummyUiSink:
        def __init__(self, _router, jsonl_path=None):
            self.jsonl_path = jsonl_path

    monkeypatch.setattr(run_trading_stub_server, "UiTelemetryAlertSink", DummyUiSink)
    server = _DummyServer(None, "127.0.0.1", 0, 0)
    monkeypatch.setattr(run_trading_stub_server, "TradingStubServer", lambda *args, **kwargs: server)

    exit_code = run_trading_stub_server.main(
        [
            "--dataset",
            str(dataset_yaml),
            "--enable-metrics",
            "--metrics-jsonl",
            str(metrics_jsonl),
            "--runtime-plan-jsonl",
            str(plan_path),
            "--shutdown-after",
            "0.01",
        ]
    )

    assert exit_code == 0
    assert plan_path.exists()
    payloads = [json.loads(line) for line in plan_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert payloads, "powinien istnieć co najmniej jeden wpis planu runtime"
    payload = payloads[-1]
    assert payload["metrics"]["enabled"] is True
    assert payload["metrics"]["ui_alerts"]["metadata"]["role"] == "ui_alerts_jsonl"
    assert payload["dataset"]["sources"][-1]["path"] == str(dataset_yaml)
    assert payload["environment"]["parameter_sources"]["metrics_jsonl_path"] == "cli"
    security = _get_security_section(payload)
    assert security["enabled"] is False
    assert security["source"] == "default"


def _get_security_section(payload: Mapping[str, object]) -> Mapping[str, object]:
    section = payload.get("security")
    assert isinstance(section, Mapping)
    fail_section = section.get("fail_on_security_warnings")
    assert isinstance(fail_section, Mapping)
    return fail_section


def test_environment_overrides_apply(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys) -> None:
    metrics_jsonl = tmp_path / "metrics_env.jsonl"
    dataset_a = tmp_path / "dataset_a.yaml"
    dataset_b = tmp_path / "dataset_b.yaml"
    dataset_a.write_text("market_data: []\n", encoding="utf-8")
    dataset_b.write_text("market_data: []\n", encoding="utf-8")
    monkeypatch.setenv("RUN_TRADING_STUB_ENABLE_METRICS", "1")
    monkeypatch.setenv("RUN_TRADING_STUB_METRICS_JSONL", str(metrics_jsonl))
    monkeypatch.setenv("RUN_TRADING_STUB_LOG_LEVEL", "debug")
    monkeypatch.setenv(
        "RUN_TRADING_STUB_DATASETS",
        os.pathsep.join([str(dataset_a), str(dataset_b)]),
    )
    monkeypatch.setenv("RUN_TRADING_STUB_NO_DEFAULT_DATASET", "true")

    exit_code = run_trading_stub_server.main(["--print-runtime-plan"])

    assert exit_code == 0
    captured = capsys.readouterr()
    plan = json.loads(captured.out)
    assert plan["metrics"]["enabled"] is True
    assert plan["metrics"]["jsonl"]["path"] == str(metrics_jsonl)
    environment_overrides = plan["environment"]["overrides"]
    assert any(
        entry["option"] == "metrics_jsonl_path" and entry["applied"] is True
        for entry in environment_overrides
    )
    assert any(
        entry["option"] == "dataset_paths" and entry["applied"] is True
        for entry in environment_overrides
    )
    assert any(
        entry["option"] == "include_default_dataset" and entry["applied"] is True
        for entry in environment_overrides
    )
    assert plan["environment"]["parameter_sources"]["metrics_jsonl_path"] == "env"
    assert plan["environment"]["parameter_sources"]["log_level"] == "env"
    assert plan["environment"]["parameter_sources"]["dataset_paths"] == "env"
    assert plan["environment"]["parameter_sources"]["include_default_dataset"] == "env"
    assert plan["environment"]["parameter_sources"]["fail_on_security_warnings"] == "default"
    security = _get_security_section(plan)
    assert security["enabled"] is False
    assert security["source"] == "default"


def test_environment_override_fail_on_security_warnings(
    monkeypatch: pytest.MonkeyPatch,
    capsys,
) -> None:
    monkeypatch.setenv("RUN_TRADING_STUB_FAIL_ON_SECURITY_WARNINGS", "true")
    monkeypatch.setenv("RUN_TRADING_STUB_ENABLE_METRICS", "1")
    monkeypatch.setenv("RUN_TRADING_STUB_METRICS_JSONL", "logs/metrics/telemetry.jsonl")
    monkeypatch.setattr(
        run_trading_stub_server,
        "TradingStubServer",
        lambda *args, **kwargs: _DummyServer(None, "127.0.0.1", 0, 0),
    )

    exit_code = run_trading_stub_server.main(["--print-runtime-plan"])
    assert exit_code == 3

    payload = json.loads(capsys.readouterr().out)
    environment_section = payload.get("environment", {})
    overrides = environment_section.get("overrides", [])
    assert any(
        entry.get("option") == "fail_on_security_warnings" and entry.get("applied") is True
        for entry in overrides
    )
    parameter_sources = environment_section.get("parameter_sources", {})
    assert parameter_sources.get("fail_on_security_warnings") == "env"
    security = _get_security_section(payload)
    assert security["enabled"] is True
    assert security["source"] == "env:RUN_TRADING_STUB_FAIL_ON_SECURITY_WARNINGS"
    assert security["parameter_source"] == "env"
    assert security["environment_variable"] == "RUN_TRADING_STUB_FAIL_ON_SECURITY_WARNINGS"


def test_environment_override_ignored_by_cli(monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    monkeypatch.setenv("RUN_TRADING_STUB_PORT", "60000")

    exit_code = run_trading_stub_server.main([
        "--port",
        "60010",
        "--print-runtime-plan",
    ])

    assert exit_code == 0
    captured = capsys.readouterr()
    plan = json.loads(captured.out)
    assert plan["server"]["port"] == 60010
    overrides = plan["environment"]["overrides"]
    port_override = next(
        entry for entry in overrides if entry["variable"] == "RUN_TRADING_STUB_PORT"
    )
    assert port_override["applied"] is False
    assert port_override["reason"] == "cli_override"
    assert plan["environment"]["parameter_sources"]["port"] == "cli"
    security = _get_security_section(plan)
    assert security["enabled"] is False
    assert security["source"] == "default"

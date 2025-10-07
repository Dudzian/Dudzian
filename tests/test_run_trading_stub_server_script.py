from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("grpc")
pytest.importorskip("grpc_tools")

from scripts import run_trading_stub_server


@pytest.fixture(scope="module", autouse=True)
def ensure_trading_stubs() -> None:
    subprocess.run(
        [sys.executable, "scripts/generate_trading_stubs.py", "--skip-cpp"],
        check=True,
    )


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


def test_runs_with_default_dataset(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    server = _DummyServer(None, "", 0, 0)

    def factory(dataset, host: str, port: int, max_workers: int, **kwargs):
        nonlocal server
        server = _DummyServer(dataset, host, port, max_workers, **kwargs)
        return server

    monkeypatch.setattr(run_trading_stub_server, "TradingStubServer", factory)

    exit_code = run_trading_stub_server.main(
        [
            "--port",
            "0",
            "--shutdown-after",
            "0.01",
            "--print-address",
        ]
    )

    assert exit_code == 0
    assert server.started is True
    assert server.stopped_with == 1.0
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
      - open_time: "2024-01-01T00:00:00Z"
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


def test_metrics_server_integration(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    server = _DummyServer(None, "127.0.0.1", 0, 0)

    class DummyMetrics:
        def __init__(self) -> None:
            self.stopped_with: float | None = None

        def stop(self, grace: float | None = None) -> None:
            self.stopped_with = grace

    dummy_metrics = DummyMetrics()

    def fake_start_metrics(args):
        assert args.enable_metrics is True
        assert getattr(args, "metrics_auth_token", None) is None
        return dummy_metrics, "127.0.0.1:60062"

    monkeypatch.setattr(run_trading_stub_server, "_start_metrics_server", fake_start_metrics)
    monkeypatch.setattr(run_trading_stub_server, "TradingStubServer", lambda *args, **kwargs: server)

    exit_code = run_trading_stub_server.main(
        [
            "--enable-metrics",
            "--print-metrics-address",
            "--shutdown-after",
            "0.01",
        ]
    )

    assert exit_code == 0
    assert dummy_metrics.stopped_with == 1.0
    output = capsys.readouterr().out
    assert "127.0.0.1:60062" in output


def test_metrics_server_auth_token(monkeypatch: pytest.MonkeyPatch) -> None:
    dummy_metrics = object()

    def fake_start_metrics(args):
        assert args.metrics_auth_token == "dev-secret"
        return dummy_metrics, "127.0.0.1:60070"

    stopped: dict[str, float | None] = {"value": None}

    class DummyServer:
        def __init__(self) -> None:
            self.stopped = False

        def stop(self, grace: float | None = None) -> None:
            stopped["value"] = grace

    monkeypatch.setattr(run_trading_stub_server, "_start_metrics_server", fake_start_metrics)
    monkeypatch.setattr(run_trading_stub_server, "TradingStubServer", lambda *args, **kwargs: DummyServer())

    exit_code = run_trading_stub_server.main(
        [
            "--enable-metrics",
            "--metrics-auth-token",
            "dev-secret",
            "--shutdown-after",
            "0.01",
        ]
    )

    assert exit_code == 0
    assert stopped["value"] == 1.0

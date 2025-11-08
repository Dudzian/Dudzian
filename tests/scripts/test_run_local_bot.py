from __future__ import annotations

import json
import sys
import types
from types import SimpleNamespace
from pathlib import Path

import pytest

httpx_stub = types.ModuleType("httpx")
httpx_stub.AsyncClient = object
httpx_stub.Timeout = object
sys.modules.setdefault("httpx", httpx_stub)

from bot_core.exchanges.base import Environment as ExchangeEnvironment
from bot_core.observability.metrics import MetricsRegistry

import scripts.run_local_bot as run_local_bot


class DummyServer:
    def __init__(self, context, host: str, port: int) -> None:
        self.address = f"{host}:{port or 5000}"

    def start(self) -> None:  # pragma: no cover - prosty stub
        return None

    def stop(self, timeout: float) -> None:  # pragma: no cover - prosty stub
        del timeout
        return None


def _build_context_stub() -> SimpleNamespace:
    environment = SimpleNamespace(
        name="DemoEnv",
        keychain_key="demo_key",
        credential_purpose="trading",
        environment=ExchangeEnvironment.PAPER,
        required_permissions=("read",),
        forbidden_permissions=(),
    )
    decision_journal = SimpleNamespace(export=lambda: (
        {"event": "trade_executed", "symbol": "BTCUSDT"},
    ))
    bootstrap = SimpleNamespace(environment=environment, decision_journal=decision_journal)
    controller = SimpleNamespace(symbols=("BTCUSDT", "ETHUSDT"))
    pipeline = SimpleNamespace(bootstrap=bootstrap, controller=controller)
    entrypoint_cfg = SimpleNamespace(environment="DemoEnv", strategy="demo_strategy", risk_profile="balanced")
    kraken_entrypoint = SimpleNamespace(
        environment="kraken_paper",
        strategy="multi_strategy_default",
        risk_profile="balanced",
    )
    okx_entrypoint = SimpleNamespace(
        environment="okx_paper",
        strategy="multi_strategy_default",
        risk_profile="balanced",
    )
    bybit_entrypoint = SimpleNamespace(
        environment="bybit_paper",
        strategy="multi_strategy_default",
        risk_profile="balanced",
    )
    trading_cfg = SimpleNamespace(
        entrypoints={
            "demo_desktop": entrypoint_cfg,
            "kraken_desktop_paper": kraken_entrypoint,
            "okx_desktop_paper": okx_entrypoint,
            "bybit_desktop_paper": bybit_entrypoint,
        }
    )
    observability_cfg = SimpleNamespace(enable_log_metrics=False)
    execution_cfg = SimpleNamespace(
        default_mode="paper",
        paper_profiles={
            "kraken_paper": {
                "entrypoint": "kraken_desktop_paper",
                "metrics": {
                    "rate_limit": "bot_exchange_rate_limited_total",
                    "network_errors": "bot_exchange_errors_total",
                    "health": "bot_exchange_health_status",
                    "thresholds": {
                        "rate_limit_max": 0,
                        "network_errors_max": 0,
                        "health_min": 1,
                    },
                },
            },
            "okx_paper": {
                "entrypoint": "okx_desktop_paper",
                "metrics": {
                    "rate_limit": "bot_exchange_rate_limited_total",
                    "network_errors": "bot_exchange_errors_total",
                    "health": "bot_exchange_health_status",
                    "thresholds": {
                        "rate_limit_max": 0,
                        "network_errors_max": 1,
                        "health_min": 1,
                    },
                },
            },
            "bybit_paper": {
                "entrypoint": "bybit_desktop_paper",
                "metrics": {
                    "rate_limit": "bot_exchange_rate_limited_total",
                    "network_errors": "bot_exchange_errors_total",
                    "health": "bot_exchange_health_status",
                    "thresholds": {
                        "rate_limit_max": 0,
                        "network_errors_max": 0,
                        "health_min": 1,
                    },
                },
            },
        },
    )
    config = SimpleNamespace(trading=trading_cfg, observability=observability_cfg, execution=execution_cfg)

    def _load_credentials(*args, **kwargs):  # pragma: no cover - stub
        del args, kwargs
        return None

    secret_manager = SimpleNamespace(load_exchange_credentials=_load_credentials)
    metrics_registry = MetricsRegistry()
    context = SimpleNamespace(
        config=config,
        entrypoint=entrypoint_cfg,
        pipeline=pipeline,
        secret_manager=secret_manager,
        metrics_endpoint="http://127.0.0.1:9100/metrics",
        metrics_registry=metrics_registry,
        start=lambda auto_confirm=True: None,
        stop=lambda: None,
    )
    return context


@pytest.fixture(autouse=True)
def _stub_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    def _raise_keyboard_interrupt(_seconds: float) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(run_local_bot.time, "sleep", _raise_keyboard_interrupt)


@pytest.fixture
def _stub_dependencies(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    context = _build_context_stub()

    def _build_context(*, config_path: str, entrypoint: str | None) -> SimpleNamespace:
        del config_path
        if entrypoint is None:
            raise AssertionError("entrypoint powinien zostać jawnie podany w testach")
        return context

    monkeypatch.setattr(run_local_bot, "build_local_runtime_context", _build_context)
    monkeypatch.setattr(run_local_bot, "LocalRuntimeServer", DummyServer)
    monkeypatch.setattr(run_local_bot, "install_metrics_logging_handler", lambda: None)
    return context


def _create_runtime_file(tmp_path: Path) -> Path:
    config_file = tmp_path / "runtime.yaml"
    config_file.write_text("runtime: demo", encoding="utf-8")
    return config_file


def test_demo_mode_generates_checkpoint_and_report(
    tmp_path: Path, _stub_dependencies: SimpleNamespace
) -> None:
    config_file = _create_runtime_file(tmp_path)
    state_dir = tmp_path / "state"
    report_dir = tmp_path / "reports"
    markdown_dir = tmp_path / "markdown"

    exit_code = run_local_bot.main(
        [
            "--config",
            str(config_file),
            "--entrypoint",
            "demo_desktop",
            "--mode",
            "demo",
            "--state-dir",
            str(state_dir),
            "--report-dir",
            str(report_dir),
            "--report-markdown-dir",
            str(markdown_dir),
        ]
    )

    assert exit_code == 0
    checkpoint_file = state_dir / "state.json"
    assert checkpoint_file.exists()
    payload = json.loads(checkpoint_file.read_text(encoding="utf-8"))
    assert payload["entrypoint"] == "demo_desktop"
    report_files = list(report_dir.glob("run_local_bot_demo_*.json"))
    assert report_files, "Raport demo nie został wygenerowany"
    report_payload = json.loads(report_files[0].read_text(encoding="utf-8"))
    assert report_payload["status"] == "success"
    assert report_payload["mode"] == "demo"
    assert "report_markdown" in report_payload
    markdown_files = list(markdown_dir.glob("demo_paper_demo_*.md"))
    assert markdown_files, "Markdown z raportem demo nie został utworzony"


def test_paper_mode_requires_checkpoint(tmp_path: Path, _stub_dependencies: SimpleNamespace) -> None:
    config_file = _create_runtime_file(tmp_path)
    state_dir = tmp_path / "state"
    report_dir = tmp_path / "reports"
    markdown_dir = tmp_path / "markdown"

    exit_code = run_local_bot.main(
        [
            "--config",
            str(config_file),
            "--entrypoint",
            "demo_desktop",
            "--mode",
            "paper",
            "--state-dir",
            str(state_dir),
            "--report-dir",
            str(report_dir),
            "--report-markdown-dir",
            str(markdown_dir),
        ]
    )

    assert exit_code == 3
    report_files = list(report_dir.glob("run_local_bot_paper_*.json"))
    assert report_files, "Raport dla trybu paper nie został wygenerowany"
    report_payload = json.loads(report_files[0].read_text(encoding="utf-8"))
    assert report_payload["status"] == "blocked"
    assert any("checkpoint" in err.lower() for err in report_payload["errors"])
    assert "report_markdown" in report_payload
    markdown_files = list(markdown_dir.glob("demo_paper_paper_*.md"))
    assert markdown_files, "Markdown dla trybu paper nie został wygenerowany"


def test_collects_paper_exchange_metrics(
    tmp_path: Path, _stub_dependencies: SimpleNamespace
) -> None:
    context = _stub_dependencies
    registry: MetricsRegistry = context.metrics_registry
    rate_metric = registry.counter("bot_exchange_rate_limited_total", "test")
    error_metric = registry.counter("bot_exchange_errors_total", "test")
    health_metric = registry.gauge("bot_exchange_health_status", "test")

    rate_metric.inc(2, labels={"exchange": "kraken"})
    error_metric.inc(1, labels={"exchange": "kraken", "severity": "error"})
    error_metric.inc(3, labels={"exchange": "okx", "severity": "warning"})
    health_metric.set(1.0, labels={"exchange": "kraken"})
    health_metric.set(0.0, labels={"exchange": "okx"})

    config_file = _create_runtime_file(tmp_path)
    state_dir = tmp_path / "state"
    report_dir = tmp_path / "reports"
    markdown_dir = tmp_path / "markdown"

    exit_code = run_local_bot.main(
        [
            "--config",
            str(config_file),
            "--entrypoint",
            "demo_desktop",
            "--mode",
            "demo",
            "--state-dir",
            str(state_dir),
            "--report-dir",
            str(report_dir),
            "--report-markdown-dir",
            str(markdown_dir),
        ]
    )

    assert exit_code == 0
    report_files = list(report_dir.glob("run_local_bot_demo_*.json"))
    assert report_files
    payload = json.loads(report_files[0].read_text(encoding="utf-8"))
    metrics_summary = payload.get("paper_exchange_metrics")
    assert metrics_summary is not None
    assert metrics_summary["kraken_desktop_paper"]["rate_limited_events"] == 2.0
    assert metrics_summary["kraken_desktop_paper"]["network_errors"] == 1.0
    assert metrics_summary["kraken_desktop_paper"]["health_status"] == 1.0
    assert metrics_summary["okx_desktop_paper"]["network_errors"] == 3.0

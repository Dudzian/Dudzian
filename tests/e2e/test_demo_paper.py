from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

# W testach E2E wykorzystujemy lekkie atrapy zamiast pełnych zależności runtime.
httpx_stub = types.ModuleType("httpx")
httpx_stub.AsyncClient = object
httpx_stub.Timeout = object
sys.modules.setdefault("httpx", httpx_stub)

from bot_core.exchanges.base import Environment as ExchangeEnvironment

import scripts.run_local_bot as run_local_bot


class DummyServer:
    def __init__(self, context: SimpleNamespace, host: str, port: int) -> None:
        del context
        self.address = f"{host}:{port or 5000}"

    def start(self) -> None:  # pragma: no cover - prosta atrapa
        return None

    def stop(self, timeout: float) -> None:  # pragma: no cover - prosta atrapa
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
    decision_journal = SimpleNamespace(
        export=lambda: (
            {"event": "trade_executed", "symbol": "BTCUSDT"},
            {"event": "trade_executed", "symbol": "ETHUSDT"},
        )
    )
    bootstrap = SimpleNamespace(environment=environment, decision_journal=decision_journal)
    controller = SimpleNamespace(symbols=("BTCUSDT", "ETHUSDT"))
    pipeline = SimpleNamespace(bootstrap=bootstrap, controller=controller)
    entrypoint_cfg = SimpleNamespace(
        environment="DemoEnv",
        strategy="demo_strategy",
        risk_profile="balanced",
    )
    trading_cfg = SimpleNamespace(entrypoints={"demo_desktop": entrypoint_cfg})
    observability_cfg = SimpleNamespace(enable_log_metrics=False)
    execution_cfg = SimpleNamespace(default_mode="paper")
    config = SimpleNamespace(
        trading=trading_cfg,
        observability=observability_cfg,
        execution=execution_cfg,
    )

    secret_manager = SimpleNamespace(
        load_exchange_credentials=lambda *args, **kwargs: None,
    )

    context = SimpleNamespace(
        config=config,
        entrypoint=entrypoint_cfg,
        pipeline=pipeline,
        secret_manager=secret_manager,
        metrics_endpoint="http://127.0.0.1:9100/metrics",
        start=lambda auto_confirm=True: None,
        stop=lambda: None,
    )
    return context


@pytest.fixture(autouse=True)
def _stub_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    def _raise_keyboard_interrupt(_seconds: float) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(run_local_bot.time, "sleep", _raise_keyboard_interrupt)


@pytest.fixture()
def _stub_runtime(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    context = _build_context_stub()

    def _build_context(*, config_path: str, entrypoint: str | None) -> SimpleNamespace:
        del config_path
        if entrypoint is None:
            raise AssertionError("Scenariusz E2E wymaga jawnego wskazania entrypointu")
        return context

    monkeypatch.setattr(run_local_bot, "build_local_runtime_context", _build_context)
    monkeypatch.setattr(run_local_bot, "LocalRuntimeServer", DummyServer)
    monkeypatch.setattr(run_local_bot, "install_metrics_logging_handler", lambda: None)
    return context


@pytest.mark.e2e_demo_paper
@pytest.mark.usefixtures("_stub_runtime")
def test_demo_to_paper_flow(tmp_path: Path) -> None:
    config_file = tmp_path / "runtime.yaml"
    config_file.write_text("runtime: demo", encoding="utf-8")
    state_dir = tmp_path / "state"
    report_dir = tmp_path / "reports"
    markdown_dir = tmp_path / "markdown"

    exit_demo = run_local_bot.main(
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
            "--no-ready-stdout",
        ]
    )
    assert exit_demo == 0

    checkpoint_file = state_dir / "state.json"
    assert checkpoint_file.exists()
    checkpoint_payload = json.loads(checkpoint_file.read_text(encoding="utf-8"))
    assert checkpoint_payload["entrypoint"] == "demo_desktop"
    assert checkpoint_payload["mode"] == "demo"

    exit_paper = run_local_bot.main(
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
            "--no-ready-stdout",
        ]
    )
    assert exit_paper == 0

    report_files = sorted(report_dir.glob("run_local_bot_*_*.json"))
    assert len(report_files) >= 2

    markdown_files = sorted(markdown_dir.glob("demo_paper_*.md"))
    assert markdown_files, "Raport Markdown nie został wygenerowany"

    latest_report = json.loads(report_files[-1].read_text(encoding="utf-8"))
    assert latest_report["status"] == "success"
    assert latest_report["mode"] == "paper"
    assert latest_report.get("decision_events") == 2
    assert latest_report.get("report_markdown")

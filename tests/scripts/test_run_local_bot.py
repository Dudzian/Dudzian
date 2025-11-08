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
    overview = metrics_summary["summary"]
    assert overview["status"] == "breached"
    assert overview["entrypoints"] == 3
    assert overview["breached_entrypoints"] == 2
    assert overview["unknown_entrypoints"] >= 1
    assert overview["total_breaches"] >= 3
    assert overview["missing_metrics"] >= 1
    assert set(overview["breached_entrypoint_names"]) >= {
        "kraken_desktop_paper",
        "okx_desktop_paper",
    }
    assert overview["breach_counts_by_metric"]["rate_limit"] >= 1
    assert overview["breach_counts_by_metric"]["network_errors"] >= 1
    assert overview["breach_counts_by_metric"]["health"] >= 1
    assert overview["threshold_breach_counts"]["rate_limit_max"] >= 1
    assert overview["threshold_breach_counts"]["network_errors_max"] >= 1
    assert overview["threshold_breach_counts"]["health_min"] >= 1
    assert overview["missing_metric_counts"]["bot_exchange_health_status"] >= 1
    assert overview["invalid_threshold_counts"] == {}
    assert set(overview["monitored_entrypoint_names"]) == {
        "kraken_desktop_paper",
        "okx_desktop_paper",
        "bybit_desktop_paper",
    }
    metric_coverage = overview["metric_coverage_entrypoints"]
    assert set(metric_coverage["bot_exchange_rate_limited_total"]) >= {
        "kraken_desktop_paper",
        "bybit_desktop_paper",
    }
    assert set(metric_coverage["bot_exchange_errors_total"]) >= {
        "kraken_desktop_paper",
        "okx_desktop_paper",
    }
    assert set(metric_coverage["bot_exchange_health_status"]) >= {
        "kraken_desktop_paper",
        "okx_desktop_paper",
    }
    threshold_coverage = overview["threshold_coverage_entrypoints"]
    assert set(threshold_coverage["rate_limit_max"]) >= {"kraken_desktop_paper"}
    assert set(threshold_coverage["network_errors_max"]) >= {
        "kraken_desktop_paper",
        "okx_desktop_paper",
    }
    assert set(threshold_coverage["health_min"]) >= {
        "kraken_desktop_paper",
        "okx_desktop_paper",
        "bybit_desktop_paper",
    }
    assert {
        "bot_exchange_rate_limited_total",
        "bot_exchange_errors_total",
        "bot_exchange_health_status",
    } <= set(overview["monitored_metric_names"])
    assert {"rate_limit_max", "network_errors_max", "health_min"} <= set(
        overview["monitored_threshold_names"]
    )
    metric_coverage_ratio = overview["metric_coverage_ratio"]
    assert metric_coverage_ratio["bot_exchange_rate_limited_total"] == pytest.approx(
        1.0, rel=1e-4
    )
    assert metric_coverage_ratio["bot_exchange_errors_total"] == pytest.approx(
        1.0, rel=1e-4
    )
    assert metric_coverage_ratio["bot_exchange_health_status"] == pytest.approx(
        2 / 3, rel=1e-4
    )
    threshold_coverage_ratio = overview["threshold_coverage_ratio"]
    assert threshold_coverage_ratio["rate_limit_max"] == pytest.approx(1.0, rel=1e-4)
    assert threshold_coverage_ratio["network_errors_max"] == pytest.approx(
        1.0, rel=1e-4
    )
    assert threshold_coverage_ratio["health_min"] == pytest.approx(1.0, rel=1e-4)
    assert overview["metric_coverage_score"] == pytest.approx(8 / 9, rel=1e-4)
    assert overview["threshold_coverage_score"] == pytest.approx(1.0, rel=1e-4)
    severity_totals = overview["network_error_severity_totals"]
    assert severity_totals["error"] == pytest.approx(1.0, rel=1e-4)
    assert severity_totals["warning"] == pytest.approx(3.0, rel=1e-4)
    assert severity_totals["critical"] == pytest.approx(0.0, rel=1e-4)
    assert overview["missing_error_severities"] == 7
    severity_counts = overview["missing_error_severity_counts"]
    assert severity_counts["error"] == 2
    assert severity_counts["warning"] == 2
    assert severity_counts["critical"] == 3
    missing_severity_entrypoints = overview["missing_error_severity_entrypoints"]
    assert set(missing_severity_entrypoints["critical"]) >= {
        "kraken_desktop_paper",
        "okx_desktop_paper",
        "bybit_desktop_paper",
    }
    severity_coverage = overview["network_error_severity_coverage_entrypoints"]
    assert set(severity_coverage["error"]) == {"kraken_desktop_paper"}
    assert set(severity_coverage["warning"]) == {"okx_desktop_paper"}
    assert severity_coverage["critical"] == []
    severity_ratio = overview["network_error_severity_coverage_ratio"]
    assert severity_ratio["error"] == pytest.approx(1 / 3, rel=1e-4)
    assert severity_ratio["warning"] == pytest.approx(1 / 3, rel=1e-4)
    assert severity_ratio["critical"] == pytest.approx(0.0, rel=1e-4)
    breach_index = overview["breached_thresholds_entrypoints"]
    assert "kraken_desktop_paper" in breach_index
    assert "rate_limit_max" in breach_index["kraken_desktop_paper"]
    assert "okx_desktop_paper" in breach_index
    assert {"network_errors_max", "health_min"} & set(
        breach_index["okx_desktop_paper"]
    )
    assert overview["invalid_thresholds_entrypoints"] == {}
    kraken_summary = metrics_summary["kraken_desktop_paper"]
    assert kraken_summary["rate_limited_events"] == 2.0
    assert kraken_summary["network_errors"] == 1.0
    assert kraken_summary["health_status"] == 1.0
    assert kraken_summary["metric_names"]["rate_limit"] == "bot_exchange_rate_limited_total"
    assert kraken_summary["thresholds"]["rate_limit_max"] == 0
    assert kraken_summary["status"] == "breached"
    assert kraken_summary["breaches"]
    assert any(breach["threshold"] == "rate_limit_max" for breach in kraken_summary["breaches"])
    assert kraken_summary["network_errors_by_severity"]["error"] == 1.0
    assert set(kraken_summary["missing_error_severities"]) == {"warning", "critical"}

    okx_summary = metrics_summary["okx_desktop_paper"]
    assert okx_summary["network_errors"] == 3.0
    assert okx_summary["thresholds"]["network_errors_max"] == 1
    assert okx_summary["status"] == "breached"
    assert {breach["threshold"] for breach in okx_summary["breaches"]} >= {"network_errors_max", "health_min"}
    assert okx_summary["network_errors_by_severity"]["warning"] == 3.0
    assert set(okx_summary["missing_error_severities"]) == {"error", "critical"}

    bybit_summary = metrics_summary["bybit_desktop_paper"]
    assert bybit_summary["status"] in {"ok", "unknown"}
    assert set(bybit_summary["network_errors_by_severity"].keys()) == {
        "warning",
        "error",
        "critical",
    }
    unknown_names = set(overview["unknown_entrypoint_names"])
    missing_entrypoints = overview["missing_metrics_entrypoints"]
    if bybit_summary["status"] == "unknown":
        assert "bot_exchange_health_status" in bybit_summary.get("missing_metrics", [])
        assert "bybit_desktop_paper" in unknown_names
        assert "bybit_desktop_paper" in missing_entrypoints
        assert "bot_exchange_health_status" in missing_entrypoints["bybit_desktop_paper"]
        assert set(bybit_summary["missing_error_severities"]) == {
            "warning",
            "error",
            "critical",
        }
    else:
        ok_names = set(overview["ok_entrypoint_names"])
        assert "bybit_desktop_paper" in ok_names
        assert "bybit_desktop_paper" not in unknown_names

    warnings = payload.get("warnings", [])
    assert any("kraken_desktop_paper" in warning and "rate_limit_max" in warning for warning in warnings)
    assert any("okx_desktop_paper" in warning and "network_errors_max" in warning for warning in warnings)

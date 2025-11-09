import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

PySide6 = pytest.importorskip("PySide6", reason="Wymagany PySide6 do testów UI")

from PySide6.QtCore import QObject, QUrl  # type: ignore[attr-defined]
from PySide6.QtQml import QQmlApplicationEngine  # type: ignore[attr-defined]

try:  # pragma: no cover - zależne od środowiska CI
    from PySide6.QtWidgets import QApplication  # type: ignore[attr-defined]
except ImportError as exc:  # pragma: no cover - brak bibliotek systemowych
    pytest.skip(f"Brak zależności QtWidgets: {exc}", allow_module_level=True)

from core.monitoring.metrics_api import (
    ComplianceTelemetry,
    GuardrailOverview,
    IOQueueTelemetry,
    RetrainingTelemetry,
    RuntimeTelemetrySnapshot,
)
from ui.backend import runtime_service as runtime_service_module
from ui.backend.runtime_service import RuntimeService
from ui.backend.telemetry_provider import TelemetryProvider


def _sample_snapshot() -> RuntimeTelemetrySnapshot:
    generated = datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    io_entries = (
        IOQueueTelemetry(
            environment="prod",
            queue="binance.spot",
            timeout_total=2.0,
            timeout_avg_seconds=1.25,
            rate_limit_wait_total=4.0,
            rate_limit_wait_avg_seconds=0.8,
            severity="warning",
        ),
    )
    guardrail = GuardrailOverview(
        total_queues=1,
        normal_queues=0,
        info_queues=0,
        warning_queues=1,
        error_queues=0,
        total_timeouts=2.0,
        total_rate_limit_waits=4.0,
    )
    retraining_entries = (
        RetrainingTelemetry(
            status="completed",
            runs=3,
            average_duration_seconds=42.5,
            average_drift_score=0.12,
        ),
    )
    compliance = ComplianceTelemetry(
        total_violations=1.0,
        by_severity={"warning": 1.0},
        by_rule={"KYC_MISSING_FIELDS": 1.0},
    )
    return RuntimeTelemetrySnapshot(
        generated_at=generated,
        io_queues=io_entries,
        guardrail_overview=guardrail,
        retraining=retraining_entries,
        compliance=compliance,
    )


def _sample_decisions() -> list[dict[str, str]]:
    return [
        {
            "event": "order_submitted",
            "timestamp": "2025-01-01T12:00:00+00:00",
            "environment": "prod",
            "portfolio": "alpha",
            "risk_profile": "balanced",
            "strategy": "mean_reversion",
            "schedule": "auto",
            "symbol": "BTC/USDT",
            "side": "buy",
            "status": "submitted",
            "decision_state": "trade",
            "decision_signal": "long",
            "decision_should_trade": "true",
            "decision_model": "xgb-v5",
            "decision_confidence": "0.9120",
            "ai_probability": "0.8450",
            "market_regime": "bull",
            "market_regime_risk_level": "elevated",
            "market_regime_confidence": "0.7800",
            "strategy_recommendation": "momentum_v2",
        }
    ]


@pytest.mark.timeout(30)
def test_runtime_overview_renders_snapshot(tmp_path: Path) -> None:
    provider = TelemetryProvider(snapshot_loader=_sample_snapshot)
    runtime_service = RuntimeService(decision_loader=lambda limit: [])
    app = QApplication.instance() or QApplication([])
    engine = QQmlApplicationEngine()
    engine.rootContext().setContextProperty("telemetryProvider", provider)
    engine.rootContext().setContextProperty("runtimeService", runtime_service)
    qml_path = Path(__file__).resolve().parents[2] / "ui" / "qml" / "dashboard" / "RuntimeOverview.qml"
    engine.load(QUrl.fromLocalFile(str(qml_path)))
    assert engine.rootObjects(), "Nie udało się załadować RuntimeOverview.qml"
    root = engine.rootObjects()[0]

    ok = provider.refreshTelemetry()
    assert ok is True
    app.processEvents()

    summary = provider.complianceSummary
    assert summary["totalViolations"] == 1.0

    last_updated = root.findChild(QObject, "runtimeOverviewLastUpdated")
    assert last_updated is not None
    assert "2025" in last_updated.property("text")

    guardrail_card = root.findChild(QObject, "runtimeOverviewGuardrailCard")
    assert guardrail_card is not None
    manual_button = root.findChild(QObject, "manualRefreshButton")
    assert manual_button is not None and manual_button.property("enabled") is True

    engine.deleteLater()
    app.quit()


@pytest.mark.timeout(30)
def test_runtime_overview_ai_card_populates_decisions() -> None:
    provider = TelemetryProvider(snapshot_loader=_sample_snapshot)

    def _loader(limit: int) -> list[dict[str, str]]:
        return _sample_decisions()

    runtime_service = RuntimeService(decision_loader=_loader)

    app = QApplication.instance() or QApplication([])
    engine = QQmlApplicationEngine()
    engine.rootContext().setContextProperty("telemetryProvider", provider)
    engine.rootContext().setContextProperty("runtimeService", runtime_service)
    qml_path = Path(__file__).resolve().parents[2] / "ui" / "qml" / "dashboard" / "RuntimeOverview.qml"
    engine.load(QUrl.fromLocalFile(str(qml_path)))
    assert engine.rootObjects(), "Nie udało się załadować RuntimeOverview.qml"
    root = engine.rootObjects()[0]

    provider.refreshTelemetry()
    runtime_service.loadRecentDecisions(5)
    app.processEvents()

    decisions = root.property("aiDecisions")
    assert isinstance(decisions, list)
    assert len(decisions) == 1
    first = decisions[0]
    assert first["decision"]["state"] == "trade"
    assert first["marketRegime"]["regime"] == "bull"

    card = root.findChild(QObject, "runtimeOverviewAiCard")
    assert card is not None
    error_banner = root.findChild(QObject, "runtimeOverviewAiErrorBanner")
    assert error_banner is not None
    assert error_banner.property("visible") is False

    engine.deleteLater()
    app.quit()


@pytest.mark.timeout(30)
def test_runtime_overview_ai_card_handles_errors() -> None:
    provider = TelemetryProvider(snapshot_loader=_sample_snapshot)

    def _loader(limit: int) -> list[dict[str, str]]:
        raise RuntimeError("journal offline")

    runtime_service = RuntimeService(decision_loader=_loader)

    app = QApplication.instance() or QApplication([])
    engine = QQmlApplicationEngine()
    engine.rootContext().setContextProperty("telemetryProvider", provider)
    engine.rootContext().setContextProperty("runtimeService", runtime_service)
    qml_path = Path(__file__).resolve().parents[2] / "ui" / "qml" / "dashboard" / "RuntimeOverview.qml"
    engine.load(QUrl.fromLocalFile(str(qml_path)))
    assert engine.rootObjects(), "Nie udało się załadować RuntimeOverview.qml"
    root = engine.rootObjects()[0]

    runtime_service.loadRecentDecisions(3)
    app.processEvents()

    error_text = root.property("aiDecisionError")
    assert error_text == "journal offline"
    error_banner = root.findChild(QObject, "runtimeOverviewAiErrorBanner")
    assert error_banner is not None
    assert error_banner.property("visible") is True

    empty_label = root.findChild(QObject, "runtimeOverviewAiEmptyLabel")
    assert empty_label is not None
    assert empty_label.property("visible") is False

    engine.deleteLater()
    app.quit()


def test_telemetry_provider_reports_errors() -> None:
    calls: list[int] = []

    def _failing_loader() -> RuntimeTelemetrySnapshot:
        calls.append(1)
        raise RuntimeError("registry unreachable")

    provider = TelemetryProvider(snapshot_loader=_failing_loader)
    result = provider.refreshTelemetry()
    assert result is False
    assert provider.errorMessage == "registry unreachable"
    assert len(calls) == 1


def test_runtime_service_attaches_to_live_decision_log(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    log_file = tmp_path / "audit" / "decision_logs" / "live_execution.jsonl"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    log_file.write_text(json.dumps(_sample_decisions()[0]) + "\n", encoding="utf-8")

    config_path = tmp_path / "config" / "core.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text("{}", encoding="utf-8")

    dummy_config = object()

    monkeypatch.setenv("BOT_CORE_UI_CORE_CONFIG_PATH", str(config_path))
    monkeypatch.setattr(runtime_service_module, "load_core_config", lambda path: dummy_config)
    monkeypatch.setattr(runtime_service_module, "resolve_decision_log_config", lambda _cfg: (log_file, {}))

    service = RuntimeService()
    assert service.attachToLiveDecisionLog("alpha") is True

    decisions = service.loadRecentDecisions(5)
    assert len(decisions) == 1
    assert decisions[0]["portfolio"] == "alpha"
    assert service.errorMessage == ""
    assert Path(service.activeDecisionLogPath) == log_file


def test_runtime_service_attach_reports_missing_log(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config_path = tmp_path / "config" / "core.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text("{}", encoding="utf-8")

    dummy_config = object()
    missing = tmp_path / "audit" / "decision_logs" / "missing.jsonl"

    monkeypatch.setenv("BOT_CORE_UI_CORE_CONFIG_PATH", str(config_path))
    monkeypatch.setattr(runtime_service_module, "load_core_config", lambda path: dummy_config)
    monkeypatch.setattr(runtime_service_module, "resolve_decision_log_config", lambda _cfg: (missing, {}))

    service = RuntimeService()
    assert service.attachToLiveDecisionLog("stage6") is False
    assert "nie istnieje" in service.errorMessage.lower()

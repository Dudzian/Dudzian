from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

PySide6 = pytest.importorskip("PySide6", reason="Wymagany PySide6 do testów UI")

from PySide6.QtCore import QObject, QUrl, Qt, QMetaObject, Q_ARG  # type: ignore[attr-defined]
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


def _sample_risk_decisions() -> list[dict[str, object]]:
    return [
        {
            "event": "risk_blocked",
            "timestamp": "2025-01-02T09:15:00+00:00",
            "environment": "prod",
            "portfolio": "alpha",
            "risk_profile": "dynamic",
            "strategy": "momentum_v2",
            "status": "risk_block",
            "decision_state": "halt",
            "decision_should_trade": "false",
            "risk_flags": ["latency_spike"],
            "stress_failures": ["latency_spike"],
            "stress_overrides": [{"reason": "latency_spike"}],
        },
        {
            "event": "risk_update",
            "timestamp": "2025-01-02T10:00:00+00:00",
            "environment": "prod",
            "portfolio": "alpha",
            "risk_profile": "dynamic",
            "strategy": "mean_reversion",
            "status": "ok",
            "decision_state": "monitor",
            "decision_should_trade": "false",
            "risk_flags": ["drawdown_watch"],
        },
        {
            "event": "risk_freeze",
            "timestamp": "2025-01-02T11:30:00+00:00",
            "environment": "prod",
            "portfolio": "alpha",
            "risk_profile": "dynamic",
            "strategy": "momentum_v2",
            "status": "risk_freeze",
            "risk_action": "manual_freeze",
            "decision_state": "halt",
            "decision_should_trade": "false",
            "risk_flags": ["latency_spike"],
        },
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

    empty_label = root.findChild(QObject, "runtimeOverviewAiEmptyLabel")
    assert empty_label is not None
    assert empty_label.property("visible") is False

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


@pytest.mark.timeout(30)
def test_runtime_overview_risk_panel_filters_and_actions() -> None:
    provider = TelemetryProvider(snapshot_loader=_sample_snapshot)

    def _loader(limit: int) -> list[dict[str, object]]:
        return _sample_risk_decisions()

    runtime_service = RuntimeService(decision_loader=_loader)

    app = QApplication.instance() or QApplication([])
    engine = QQmlApplicationEngine()
    engine.rootContext().setContextProperty("telemetryProvider", provider)
    engine.rootContext().setContextProperty("runtimeService", runtime_service)
    qml_path = Path(__file__).resolve().parents[2] / "ui" / "qml" / "dashboard" / "RuntimeOverview.qml"
    engine.load(QUrl.fromLocalFile(str(qml_path)))
    assert engine.rootObjects(), "Nie udało się załadować RuntimeOverview.qml"
    root = engine.rootObjects()[0]

    runtime_service.loadRecentDecisions(10)
    app.processEvents()

    metrics = root.property("riskMetrics")
    assert isinstance(metrics, dict)
    assert metrics.get("blockCount") == 1
    assert "latency_spike" in metrics.get("uniqueRiskFlags", [])
    assert metrics.get("lastBlock", {}).get("timestamp") == "2025-01-02T09:15:00+00:00"
    assert metrics.get("lastFreeze", {}).get("timestamp") == "2025-01-02T11:30:00+00:00"
    assert metrics.get("lastStressOverride", {}).get("timestamp") == "2025-01-02T09:15:00+00:00"
    summaries = metrics.get("strategySummaries")
    assert isinstance(summaries, list)
    assert summaries, "Oczekiwano podsumowań strategii"
    leader = summaries[0]
    assert leader.get("strategy") == "momentum_v2"
    assert leader.get("blockCount") == 1
    assert leader.get("freezeCount") == 1
    assert leader.get("stressOverrideCount") == 1
    assert leader.get("severity") == "block"

    risk_panel = root.findChild(QObject, "riskJournalPanel")
    assert risk_panel is not None

    timeline = risk_panel.property("filteredTimeline")
    assert isinstance(timeline, list)
    assert len(timeline) == 3

    risk_panel.setProperty("riskFilter", "drawdown_watch")
    app.processEvents()
    filtered = risk_panel.property("filteredTimeline")
    assert len(filtered) == 1

    risk_panel.setProperty("riskFilter", "")
    risk_panel.setProperty("strategyFilter", "momentum_v2")
    app.processEvents()
    filtered_strategy = risk_panel.property("filteredTimeline")
    assert len(filtered_strategy) == 2

    risk_panel.setProperty("riskFilter", "drawdown_watch")
    app.processEvents()
    filtered_combo = risk_panel.property("filteredTimeline")
    assert len(filtered_combo) == 0

    risk_panel.setProperty("riskFilter", "latency_spike")
    app.processEvents()
    filtered_lat = risk_panel.property("filteredTimeline")
    assert len(filtered_lat) == 2

    assert risk_panel.property("strategyChipCount") >= 1
    assert risk_panel.property("riskFlagChipCount") >= 2
    assert risk_panel.property("stressFailureChipCount") >= 1

    top_strategy_chip = risk_panel.findChild(QObject, "riskJournalStrategyChip_0")
    assert top_strategy_chip is not None
    strategy_text = top_strategy_chip.property("chipText")
    assert isinstance(strategy_text, str)
    assert "momentum_v2" in strategy_text
    assert "2" in strategy_text

    top_flag_chip = risk_panel.findChild(QObject, "riskJournalFlagChip_0")
    assert top_flag_chip is not None
    chip_text = top_flag_chip.property("chipText")
    assert isinstance(chip_text, str)
    assert "latency_spike" in chip_text
    assert "2" in chip_text

    top_stress_chip = risk_panel.findChild(QObject, "riskJournalStressChip_0")
    assert top_stress_chip is not None
    stress_text = top_stress_chip.property("chipText")
    assert isinstance(stress_text, str)
    assert "latency_spike" in stress_text
    assert ")" in stress_text

    summary_flow = risk_panel.findChild(QObject, "riskJournalStrategySummaries")
    assert summary_flow is not None
    top_summary_card = risk_panel.findChild(QObject, "riskJournalStrategySummary_0")
    assert top_summary_card is not None
    assert top_summary_card.property("summaryStrategy") == "momentum_v2"
    assert top_summary_card.property("summaryBlockCount") == 1
    assert top_summary_card.property("summarySeverity") == "block"

    entry = filtered_lat[0]
    QMetaObject.invokeMethod(
        risk_panel,
        "openDrilldown",
        Qt.DirectConnection,
        Q_ARG("QVariant", entry),
    )
    app.processEvents()
    assert risk_panel.property("selectedEntry") is not None

    QMetaObject.invokeMethod(
        risk_panel,
        "triggerOperatorAction",
        Qt.DirectConnection,
        Q_ARG("QString", "requestFreeze"),
    )
    app.processEvents()

    last_action = runtime_service.lastOperatorAction
    assert last_action.get("action") == "freeze"
    assert last_action.get("entry", {}).get("event") == entry.get("record", {}).get("event")

    assert risk_panel.property("selectedEntry") is None

    last_action_label = risk_panel.findChild(QObject, "riskJournalLastOperatorAction")
    assert last_action_label is not None
    text = last_action_label.property("text")
    assert isinstance(text, str)
    assert "zamrożenie" in text
    assert "risk_blocked" in text

    last_block_label = risk_panel.findChild(QObject, "riskJournalLastBlock")
    assert last_block_label is not None
    assert "Ostatnia blokada" in last_block_label.property("text")
    assert "2025-01-02T09:15:00+00:00" in last_block_label.property("text")

    last_freeze_label = risk_panel.findChild(QObject, "riskJournalLastFreeze")
    assert last_freeze_label is not None
    freeze_text = last_freeze_label.property("text")
    assert "Ostatnia blokada strategiczna" in freeze_text
    assert "2025-01-02T11:30:00+00:00" in freeze_text
    assert "manual_freeze" in freeze_text

    last_override_label = risk_panel.findChild(QObject, "riskJournalLastOverride")
    assert last_override_label is not None
    override_text = last_override_label.property("text")
    assert "Ostatni stress override" in override_text
    assert "2025-01-02T09:15:00+00:00" in override_text

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

from __future__ import annotations

import datetime
import json
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any, Iterable

import pytest

try:  # pragma: no cover - environment guard
    import PySide6  # type: ignore # noqa: F401
except ImportError as exc:  # pragma: no cover - environment guard
    pytest.skip(f"PySide6 unavailable: {exc}", allow_module_level=True)

try:  # pragma: no cover - environment guard
    from PySide6.QtCore import QCoreApplication, QUrl  # type: ignore[attr-defined]
    from PySide6.QtGui import QGuiApplication  # type: ignore[attr-defined]
    from PySide6.QtQml import QQmlComponent, QQmlEngine  # type: ignore[attr-defined]
except ImportError as exc:  # pragma: no cover - environment guard
    pytest.skip(f"Qt runtime missing dependencies: {exc}", allow_module_level=True)

pytestmark = [pytest.mark.performance, pytest.mark.qml]

REPO_ROOT = Path(__file__).resolve().parents[2]
QML_PATH = REPO_ROOT / "ui/qml/dashboard/RuntimeOverview.qml"
REPORT_DIR = REPO_ROOT / "reports/ci/performance_ui_render"
REPORT_DIR.mkdir(parents=True, exist_ok=True)
TELEMETRY_FIXTURE = REPO_ROOT / "tests/performance/telemetry_samples.json"


@pytest.fixture(scope="session")
def qt_app() -> QGuiApplication:  # pragma: no cover - infrastructure
    app = QGuiApplication.instance()
    if app is None:
        app = QGuiApplication([])
    return app


@pytest.fixture()
def qml_engine(qt_app: QGuiApplication) -> QQmlEngine:  # pragma: no cover - infrastructure
    engine = QQmlEngine()
    yield engine
    engine.collectGarbage()


@pytest.fixture(scope="session")
def telemetry_samples() -> dict[str, Any]:
    payload = json.loads(TELEMETRY_FIXTURE.read_text(encoding="utf-8"))
    return payload


def _load_component(engine: QQmlEngine) -> QQmlComponent:
    component = QQmlComponent(engine)
    component.loadUrl(QUrl.fromLocalFile(str(QML_PATH)))
    assert component.isReady(), component.errorString()
    return component


def _p90(latencies_ms: Iterable[float]) -> float:
    samples = sorted(latencies_ms)
    if not samples:
        return 0.0
    index = max(0, int(len(samples) * 0.9) - 1)
    return samples[index]


def _render_component(component: QQmlComponent, properties: dict[str, Any]) -> float:
    start = time.perf_counter()
    instance = component.create()
    assert instance is not None
    for key, value in properties.items():
        instance.setProperty(key, value)
    QCoreApplication.processEvents()
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    instance.deleteLater()
    return elapsed_ms


def _log_report(name: str, latencies_ms: list[float], threshold_ms: float) -> None:
    commit = (
        subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True)
        .strip()
    )
    payload = {
        "scenario": name,
        "samples": len(latencies_ms),
        "latencies_ms": latencies_ms,
        "avg_ms": statistics.mean(latencies_ms) if latencies_ms else 0.0,
        "p90_ms": _p90(latencies_ms),
        "sla_ms": threshold_ms,
        "timestamp_utc": datetime.datetime.utcnow().isoformat() + "Z",
        "git_commit": commit,
    }
    (REPORT_DIR / f"{name}.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )


@pytest.mark.performance
@pytest.mark.parametrize("alert_count", [24, 48])
def test_feed_sla_panel_render_time_under_feed_spike(qml_engine: QQmlEngine, alert_count: int) -> None:
    component = _load_component(qml_engine)
    heavy_feed_report = {
        "sla_state": "warning",
        "latency_state": "warning",
        "latency_warning_ms": 120.0,
        "latency_critical_ms": 180.0,
        "p50_ms": 42.0,
        "p95_ms": 110.0,
        "reconnects": 3,
        "reconnects_warning": 6,
        "downtime_seconds": 1.5,
        "downtime_warning_seconds": 8.0,
        "nextRetrySeconds": 2.5,
    }
    alert_history = [
        {
            "severity": "warning" if idx % 2 == 0 else "critical",
            "label": f"latency_p95_{idx}",
            "timestamp": f"2024-03-01T10:{idx:02d}:00Z",
            "formattedValue": f"{90 + idx} ms",
        }
        for idx in range(alert_count)
    ]
    alert_channels = [
        {"name": "pagerduty", "status": "healthy"},
        {"name": "slack", "status": "degraded"},
    ]

    latencies = [
        _render_component(
            component,
            {
                "feedSlaReport": heavy_feed_report,
                "feedHealth": {"lastError": "Rate limited"},
                "feedAlertHistory": alert_history,
                "feedAlertChannels": alert_channels,
            },
        )
        for _ in range(6)
    ]

    p90_ms = _p90(latencies)
    _log_report(
        "feed_sla_panel_render",
        latencies,
        threshold_ms=220.0,
    )
    assert p90_ms < 220.0, f"Render SLA panel p90 too slow: {p90_ms:.2f} ms"


@pytest.mark.performance
@pytest.mark.parametrize("timeline_size", [64, 128])
def test_risk_panels_render_time_with_dense_timeline(qml_engine: QQmlEngine, timeline_size: int) -> None:
    component = _load_component(qml_engine)
    risk_timeline = [
        {
            "timestamp": f"2024-03-01T12:{idx:02d}:00Z",
            "severity": "warning" if idx % 5 else "error",
            "metric": "margin_buffer",
            "value": 0.25 + idx * 0.005,
        }
        for idx in range(timeline_size)
    ]
    risk_metrics = {
        "var": 0.023,
        "expected_shortfall": 0.041,
        "max_drawdown": 0.12,
        "margin_buffer": 0.18,
    }

    latencies = [
        _render_component(
            component,
            {
                "riskTimeline": risk_timeline,
                "riskMetrics": risk_metrics,
                "longPollMetrics": [{"name": "decision_lag_ms", "value": 12.0}] * 12,
                "cycleMetrics": {"p50_ms": 35.0, "p95_ms": 70.0, "max_ms": 120.0},
            },
        )
        for _ in range(6)
    ]

    avg_ms = statistics.mean(latencies)
    _log_report(
        "risk_panels_render",
        latencies,
        threshold_ms=180.0,
    )
    assert avg_ms < 180.0, f"Risk panels rendering averaged {avg_ms:.2f} ms"


@pytest.mark.performance
def test_sla_panels_render_across_telemetry_samples(
    qml_engine: QQmlEngine, telemetry_samples: dict[str, Any]
) -> None:
    component = _load_component(qml_engine)
    latencies: list[float] = []
    feed_samples = telemetry_samples.get("feed_panels", [])
    baseline_properties = {
        "feedAlertChannels": [],
        "feedAlertHistory": [],
        "feedHealth": {},
        "feedSlaReport": {},
    }

    for sample in feed_samples:
        properties = {**baseline_properties, **sample.get("properties", {})}
        latencies.extend(_render_component(component, properties) for _ in range(3))

    p90_ms = _p90(latencies)
    threshold_ms = float(telemetry_samples.get("sla_threshold_ms", 230.0))
    _log_report(
        "feed_sla_panel_render_telemetry",
        latencies,
        threshold_ms=threshold_ms,
    )
    assert p90_ms < threshold_ms, f"Telemetry feed SLA p90 too slow: {p90_ms:.2f} ms"


@pytest.mark.performance
def test_risk_panels_render_across_telemetry_samples(
    qml_engine: QQmlEngine, telemetry_samples: dict[str, Any]
) -> None:
    component = _load_component(qml_engine)
    latencies: list[float] = []
    risk_samples = telemetry_samples.get("risk_panels", [])
    baseline_properties = {
        "riskTimeline": [],
        "riskMetrics": {},
        "longPollMetrics": [],
        "cycleMetrics": {},
    }

    for sample in risk_samples:
        properties = {**baseline_properties, **sample.get("properties", {})}
        latencies.extend(_render_component(component, properties) for _ in range(3))

    p90_ms = _p90(latencies)
    threshold_ms = float(telemetry_samples.get("risk_threshold_ms", 190.0))
    _log_report(
        "risk_panels_render_telemetry",
        latencies,
        threshold_ms=threshold_ms,
    )
    assert p90_ms < threshold_ms, f"Telemetry risk panels p90 too slow: {p90_ms:.2f} ms"

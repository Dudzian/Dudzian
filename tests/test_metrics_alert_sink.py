"""Testy sinka alertów telemetrii UI bez zależności od wygenerowanych stubów gRPC."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from bot_core.alerts import AlertMessage, DefaultAlertRouter
from bot_core.alerts.audit import InMemoryAlertAuditLog
from bot_core.alerts.base import AlertChannel
from bot_core.runtime.metrics_alerts import UiTelemetryAlertSink


class CaptureChannel(AlertChannel):
    def __init__(self) -> None:
        self.name = "capture"
        self.messages: list[AlertMessage] = []

    def send(self, message: AlertMessage) -> None:
        self.messages.append(message)

    def health_check(self) -> dict[str, str]:
        return {"status": "ok"}


def _build_router() -> tuple[DefaultAlertRouter, CaptureChannel]:
    audit_log = InMemoryAlertAuditLog()
    router = DefaultAlertRouter(audit_log=audit_log)
    channel = CaptureChannel()
    router.register(channel)
    return router, channel

@dataclass(slots=True)
class FakeTimestamp:
    """Minimalna reprezentacja sygnatury czasowej używanej przez sink."""

    seconds: int
    nanos: int

    @classmethod
    def now(cls) -> "FakeTimestamp":
        now = datetime.now(timezone.utc)
        epoch = now.timestamp()
        seconds = int(epoch)
        nanos = int((epoch - seconds) * 1_000_000_000)
        return cls(seconds=seconds, nanos=nanos)


@dataclass(slots=True)
class FakeSnapshot:
    """Prosty obiekt imitujący `MetricsSnapshot` z kontraktu gRPC."""

    notes: str
    fps: float | None = None
    generated_at: FakeTimestamp | None = None


def _make_snapshot(notes: dict[str, object], fps: float | None = None) -> FakeSnapshot:
    return FakeSnapshot(
        notes=json.dumps(notes),
        fps=fps,
        generated_at=FakeTimestamp.now(),
    )


def test_reduce_motion_alert(tmp_path: Path):
    router, channel = _build_router()
    sink = UiTelemetryAlertSink(router, jsonl_path=tmp_path / "ui_alerts.jsonl")

    snapshot = _make_snapshot(
        {
            "event": "reduce_motion",
            "active": True,
            "fps_target": 60,
            "overlay_active": 2,
            "overlay_allowed": 4,
            "window_count": 3,
        },
        fps=48.5,
    )

    sink.handle_snapshot(snapshot)

    assert len(channel.messages) == 1
    message = channel.messages[0]
    assert message.category == "ui.performance.reduce_motion"
    assert message.severity == "warning"
    assert "48.5" in message.body
    assert message.context["overlay_active"] == "2"

    lines = (tmp_path / "ui_alerts.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert lines, "Powinien powstać wpis JSONL"
    record = json.loads(lines[-1])
    assert record["category"] == "ui.performance.reduce_motion"
    assert record["severity"] == "warning"
    assert record["context"]["active"] == "true"


def test_reduce_motion_alert_includes_screen_context(tmp_path: Path):
    router, channel = _build_router()
    sink = UiTelemetryAlertSink(router, jsonl_path=tmp_path / "ui_alerts.jsonl")

    snapshot = _make_snapshot(
        {
            "event": "reduce_motion",
            "active": True,
            "fps_target": 120,
            "overlay_active": 3,
            "overlay_allowed": 4,
            "screen": {
                "name": "Dell U2720Q",
                "index": 1,
                "refresh_hz": 60.0,
                "device_pixel_ratio": 1.25,
                "geometry_px": {"x": 0, "y": 0, "width": 3840, "height": 2160},
            },
        },
        fps=48.0,
    )

    sink.handle_snapshot(snapshot)

    assert channel.messages, "Powinien zostać wysłany alert"
    message = channel.messages[0]
    assert message.context["screen_index"] == "1"
    assert message.context["screen_resolution"] == "3840x2160"
    assert message.context["screen_refresh_hz"] == "60.00"
    assert "Ekran:" in message.body

    record = json.loads((tmp_path / "ui_alerts.jsonl").read_text(encoding="utf-8").strip())
    assert record["context"]["screen_index"] == "1"
    assert record["context"]["screen_dpr"] == "1.25"


def test_reduce_motion_alert_deduplicates_state(tmp_path: Path):
    router, channel = _build_router()
    sink = UiTelemetryAlertSink(router, jsonl_path=tmp_path / "ui_alerts.jsonl")

    snapshot = _make_snapshot({"event": "reduce_motion", "active": True})
    sink.handle_snapshot(snapshot)
    sink.handle_snapshot(snapshot)

    assert len(channel.messages) == 1

    disabled = _make_snapshot({"event": "reduce_motion", "active": False})
    sink.handle_snapshot(disabled)
    assert len(channel.messages) == 2
    assert channel.messages[-1].severity == "info"

    records = [
        json.loads(line)
        for line in (tmp_path / "ui_alerts.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(records) == 2
    assert records[-1]["context"]["active"] == "false"


def test_overlay_budget_alert_dispatch(tmp_path: Path):
    router, channel = _build_router()
    sink = UiTelemetryAlertSink(router, jsonl_path=tmp_path / "ui_alerts.jsonl")

    snapshot = _make_snapshot(
        {
            "event": "overlay_budget",
            "active_overlays": 5,
            "allowed_overlays": 3,
            "reduce_motion": True,
        },
        fps=59.0,
    )

    sink.handle_snapshot(snapshot)

    assert len(channel.messages) == 1
    message = channel.messages[0]
    assert message.category == "ui.performance.overlay_budget"
    assert message.severity == "critical"
    assert "5" in message.body
    lines = [
        line
        for line in (tmp_path / "ui_alerts.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert lines
    record = json.loads(lines[-1])
    assert record["severity"] == "critical"


def test_overlay_budget_without_violation_ignored():
    router, channel = _build_router()
    sink = UiTelemetryAlertSink(router)

    snapshot = _make_snapshot(
        {
            "event": "overlay_budget",
            "active_overlays": 2,
            "allowed_overlays": 3,
        },
        fps=60.0,
    )

    sink.handle_snapshot(snapshot)

    assert not channel.messages


def test_overlay_budget_recovery_emits_followup(tmp_path: Path):
    router, channel = _build_router()
    sink = UiTelemetryAlertSink(router, jsonl_path=tmp_path / "ui_alerts.jsonl")

    violation = _make_snapshot(
        {
            "event": "overlay_budget",
            "active_overlays": 6,
            "allowed_overlays": 3,
            "reduce_motion": False,
        }
    )
    sink.handle_snapshot(violation)

    recovery = _make_snapshot(
        {
            "event": "overlay_budget",
            "active_overlays": 2,
            "allowed_overlays": 3,
            "reduce_motion": False,
        }
    )
    sink.handle_snapshot(recovery)

    assert len(channel.messages) == 2
    assert channel.messages[0].severity in {"warning", "critical"}
    assert channel.messages[1].severity == "info"


def test_jank_spike_alert(tmp_path: Path):
    router, channel = _build_router()
    sink = UiTelemetryAlertSink(
        router,
        jsonl_path=tmp_path / "ui_alerts.jsonl",
        jank_severity_critical="critical",
        jank_critical_over_ms=8.0,
    )

    snapshot = _make_snapshot(
        {
            "event": "jank_spike",
            "frame_ms": 28.0,
            "threshold_ms": 12.0,
            "reduce_motion": True,
            "overlay_active": 4,
            "overlay_allowed": 2,
            "window_count": 2,
        },
        fps=33.0,
    )

    sink.handle_snapshot(snapshot)

    assert len(channel.messages) == 1
    message = channel.messages[0]
    assert message.category == "ui.performance.jank"
    assert message.severity == "critical"
    assert "28.00" in message.body
    assert message.context["over_budget_ms"]

    lines = [
        line
        for line in (tmp_path / "ui_alerts.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert lines
    record = json.loads(lines[-1])
    assert record["category"] == "ui.performance.jank"
    assert record["context"]["frame_ms"] == "28.000"


def test_jank_alert_includes_screen_context(tmp_path: Path):
    router, channel = _build_router()
    sink = UiTelemetryAlertSink(router, jsonl_path=tmp_path / "ui_alerts.jsonl")

    snapshot = _make_snapshot(
        {
            "event": "jank_spike",
            "frame_ms": 22.0,
            "threshold_ms": 10.0,
            "reduce_motion": False,
            "overlay_active": 1,
            "overlay_allowed": 3,
            "screen": {
                "name": "LG UltraFine",
                "index": 0,
                "refresh_hz": 144.0,
                "geometry_px": {"x": 0, "y": 0, "width": 2560, "height": 1440},
            },
        },
        fps=57.0,
    )

    sink.handle_snapshot(snapshot)

    assert channel.messages, "Powinien zostać wygenerowany alert jank"
    message = channel.messages[0]
    assert message.context["screen_index"] == "0"
    assert message.context["screen_resolution"] == "2560x1440"
    assert "Ekran:" in message.body

    record = json.loads((tmp_path / "ui_alerts.jsonl").read_text(encoding="utf-8").strip())
    assert record["context"]["screen_index"] == "0"


def test_jank_spike_deduplicates_consecutive_events(tmp_path: Path):
    router, channel = _build_router()
    jsonl_path = tmp_path / "ui_alerts.jsonl"
    sink = UiTelemetryAlertSink(router, jsonl_path=jsonl_path)

    snapshot = _make_snapshot({"event": "jank_spike", "frame_ms": 25.0, "threshold_ms": 12.0})
    sink.handle_snapshot(snapshot)
    sink.handle_snapshot(snapshot)

    assert len(channel.messages) == 1

    records = [
        json.loads(line)
        for line in jsonl_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(records) == 1
    assert records[0]["payload"]["event"] == "jank_spike"


def test_overlay_alerts_can_be_disabled(tmp_path: Path):
    router, channel = _build_router()
    sink = UiTelemetryAlertSink(
        router,
        jsonl_path=tmp_path / "ui_alerts.jsonl",
        enable_overlay_alerts=False,
        log_overlay_events=True,
    )

    snapshot = _make_snapshot(
        {
            "event": "overlay_budget",
            "active_overlays": 7,
            "allowed_overlays": 3,
        }
    )

    sink.handle_snapshot(snapshot)

    assert not channel.messages
    lines = [
        line
        for line in (tmp_path / "ui_alerts.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert lines, "Nawet przy wyłączonych alertach powinien powstać wpis JSONL"
    record = json.loads(lines[-1])
    assert record["severity"] in {"warning", "critical"}


def test_overlay_alert_logging_can_be_disabled(tmp_path: Path):
    router, channel = _build_router()
    jsonl_path = tmp_path / "ui_alerts.jsonl"
    sink = UiTelemetryAlertSink(
        router,
        jsonl_path=jsonl_path,
        enable_overlay_alerts=False,
        log_overlay_events=False,
        log_reduce_motion_events=False,
        log_jank_events=False,
    )

    snapshot = _make_snapshot(
        {
            "event": "overlay_budget",
            "active_overlays": 7,
            "allowed_overlays": 3,
        }
    )

    sink.handle_snapshot(snapshot)

    assert not channel.messages
    assert not jsonl_path.exists()


def test_reduce_motion_logging_can_be_disabled(tmp_path: Path):
    router, channel = _build_router()
    jsonl_path = tmp_path / "ui_alerts.jsonl"
    sink = UiTelemetryAlertSink(
        router,
        jsonl_path=jsonl_path,
        enable_reduce_motion_alerts=False,
        log_reduce_motion_events=False,
        log_overlay_events=False,
        log_jank_events=False,
    )

    snapshot = _make_snapshot(
        {
            "event": "reduce_motion",
            "active": True,
            "fps_target": 120,
        },
        fps=40.0,
    )

    sink.handle_snapshot(snapshot)

    assert not channel.messages
    assert not jsonl_path.exists()

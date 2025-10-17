"""Testy sinka alertów telemetrii UI bez zależności od wygenerowanych stubów gRPC."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pytest

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


def test_alert_includes_risk_profile_metadata(tmp_path: Path) -> None:
    router, channel = _build_router()
    sink = UiTelemetryAlertSink(
        router,
        jsonl_path=tmp_path / "ui_alerts.jsonl",
        risk_profile={
            "name": "balanced",
            "origin": "builtin",
            "severity_min": "notice",
        },
    )

    snapshot = _make_snapshot(
        {
            "event": "overlay_budget",
            "active_overlays": 4,
            "allowed_overlays": 2,
            "reduce_motion": False,
        },
        fps=58.0,
    )

    sink.handle_snapshot(snapshot)

    assert channel.messages, "Powinien zostać wysłany alert overlay"
    message = channel.messages[0]
    assert message.context["risk_profile"] == "balanced"
    assert message.context["risk_profile_origin"] == "builtin"

    line = (tmp_path / "ui_alerts.jsonl").read_text(encoding="utf-8").strip()
    assert line, "Powinien powstać wpis JSONL"
    record = json.loads(line)
    assert record["risk_profile"]["name"] == "balanced"
    assert record["risk_profile"]["origin"] == "builtin"
    assert record["risk_profile"]["severity_min"] == "notice"


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


def test_overlay_alert_context_includes_tag(tmp_path: Path) -> None:
    router, channel = _build_router()
    sink = UiTelemetryAlertSink(router, jsonl_path=tmp_path / "ui_alerts.jsonl")

    snapshot = _make_snapshot(
        {
            "event": "overlay_budget",
            "active_overlays": 5,
            "allowed_overlays": 3,
            "reduce_motion": False,
            "tag": "desk-a",
        }
    )

    sink.handle_snapshot(snapshot)

    assert channel.messages, "Powinien zostać wysłany alert overlay"
    message = channel.messages[0]
    assert message.context["tag"] == "desk-a"

    record = json.loads((tmp_path / "ui_alerts.jsonl").read_text(encoding="utf-8").strip())
    assert record["tag"] == "desk-a"
    assert record["context"]["tag"] == "desk-a"


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


def test_retry_backlog_alerts_trigger_and_recover(tmp_path: Path) -> None:
    router, channel = _build_router()
    sink = UiTelemetryAlertSink(
        router,
        jsonl_path=tmp_path / "ui_alerts.jsonl",
        retry_backlog_threshold=3,
    )

    degraded = _make_snapshot(
        {
            "event": "reduce_motion",
            "active": True,
            "retry_backlog_before_send": 2,
            "retry_backlog_after_flush": 5,
            "window_count": 2,
        }
    )
    sink.handle_snapshot(degraded)

    backlog_messages = [
        msg for msg in channel.messages if msg.category == "ui.performance.retry_backlog"
    ]
    assert backlog_messages, "Powinien zostać wysłany alert backlogu"
    message = backlog_messages[-1]
    assert message.severity == "warning"
    assert message.context["retry_backlog_after"] == "5"
    assert message.context["retry_backlog_threshold"] == "3"
    started_at = message.context.get("retry_backlog_started_at")
    assert started_at, "Alert powinien zawierać czas rozpoczęcia degradacji"
    start_dt = datetime.fromisoformat(started_at.replace("Z", "+00:00"))
    assert start_dt.tzinfo == timezone.utc

    # Powtórzony snapshot nie generuje kolejnego alertu
    sink.handle_snapshot(degraded)
    assert len([
        msg for msg in channel.messages if msg.category == "ui.performance.retry_backlog"
    ]) == 1

    recovered = _make_snapshot(
        {
            "event": "overlay_budget",
            "retry_backlog_before_send": 1,
            "retry_backlog_after_flush": 0,
        }
    )
    sink.handle_snapshot(recovered)

    backlog_messages = [
        msg for msg in channel.messages if msg.category == "ui.performance.retry_backlog"
    ]
    assert len(backlog_messages) == 2
    recovery = backlog_messages[-1]
    assert recovery.severity == "info"
    recovery_started_at = recovery.context.get("retry_backlog_started_at")
    assert recovery_started_at == started_at
    recovered_at = recovery.context.get("retry_backlog_recovered_at")
    assert recovered_at, "Komunikat o odzyskaniu powinien zawierać czas zakończenia"
    recovered_dt = datetime.fromisoformat(recovered_at.replace("Z", "+00:00"))
    assert recovered_dt.tzinfo == timezone.utc

    records = [
        json.loads(line)
        for line in (tmp_path / "ui_alerts.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    backlog_records = [
        record
        for record in records
        if record.get("category") == "ui.performance.retry_backlog"
    ]
    assert backlog_records
    assert backlog_records[0]["context"]["retry_backlog_started_at"] == started_at


def test_retry_backlog_alerts_can_be_disabled(tmp_path: Path) -> None:
    router, channel = _build_router()
    sink = UiTelemetryAlertSink(
        router,
        jsonl_path=tmp_path / "ui_alerts.jsonl",
        enable_retry_backlog_alerts=False,
        retry_backlog_threshold=1,
        log_reduce_motion_events=False,
        log_overlay_events=False,
        log_jank_events=False,
        log_retry_backlog_events=False,
        enable_tag_inactivity_alerts=False,
        log_tag_inactivity_events=False,
    )

    snapshot = _make_snapshot(
        {
            "event": "reduce_motion",
            "retry_backlog_after_flush": 2,
        }
    )

    sink.handle_snapshot(snapshot)

    assert all(
        message.category != "ui.performance.retry_backlog" for message in channel.messages
    ), "Alert backlogu nie powinien zostać wysłany"
    assert not (tmp_path / "ui_alerts.jsonl").exists(), "Log backlogu nie powinien powstać"


def test_retry_backlog_realert_on_growth(tmp_path: Path) -> None:
    router, channel = _build_router()
    sink = UiTelemetryAlertSink(
        router,
        jsonl_path=tmp_path / "ui_alerts.jsonl",
        retry_backlog_threshold=3,
        retry_backlog_realert_delta=2,
    )

    initial = _make_snapshot(
        {
            "retry_backlog_before_send": 1,
            "retry_backlog_after_flush": 3,
        }
    )
    sink.handle_snapshot(initial)

    below_delta = _make_snapshot(
        {
            "retry_backlog_before_send": 2,
            "retry_backlog_after_flush": 4,
        }
    )
    sink.handle_snapshot(below_delta)

    above_delta = _make_snapshot(
        {
            "retry_backlog_before_send": 4,
            "retry_backlog_after_flush": 6,
        }
    )
    sink.handle_snapshot(above_delta)

    backlog_messages = [
        message
        for message in channel.messages
        if message.category == "ui.performance.retry_backlog"
        and message.severity == "warning"
    ]
    assert len(backlog_messages) == 2
    assert backlog_messages[-1].context["retry_backlog_after"] == "6"
    assert backlog_messages[-1].context["retry_backlog_delta"] == "3"
    assert (
        backlog_messages[0].context.get("retry_backlog_started_at")
        == backlog_messages[-1].context.get("retry_backlog_started_at")
    )
    assert "+3" in backlog_messages[-1].body

    records = [
        json.loads(line)
        for line in (tmp_path / "ui_alerts.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    recorded_backlog = [
        record
        for record in records
        if record.get("category") == "ui.performance.retry_backlog"
        and record.get("severity") == "warning"
    ]
    assert len(recorded_backlog) == 2
    assert recorded_backlog[-1]["context"]["retry_backlog_delta"] == "3"


def test_retry_backlog_realert_respects_cooldown(monkeypatch, tmp_path: Path) -> None:
    router, channel = _build_router()
    sink = UiTelemetryAlertSink(
        router,
        jsonl_path=tmp_path / "ui_alerts.jsonl",
        retry_backlog_threshold=2,
        retry_backlog_realert_delta=1,
        retry_backlog_realert_cooldown_seconds=30,
    )

    monotonic_time = {"value": 100.0}

    def fake_monotonic() -> float:
        return monotonic_time["value"]

    monkeypatch.setattr("bot_core.runtime.metrics_alerts.time.monotonic", fake_monotonic)

    first = _make_snapshot(
        {
            "retry_backlog_before_send": 0,
            "retry_backlog_after_flush": 2,
        }
    )
    sink.handle_snapshot(first)

    assert any(
        msg.category == "ui.performance.retry_backlog" for msg in channel.messages
    ), "Pierwszy alert powinien zostać wysłany"

    monotonic_time["value"] = 120.0

    suppressed = _make_snapshot(
        {
            "retry_backlog_before_send": 2,
            "retry_backlog_after_flush": 3,
        }
    )
    sink.handle_snapshot(suppressed)

    backlog_messages = [
        msg for msg in channel.messages if msg.category == "ui.performance.retry_backlog"
    ]
    assert len(backlog_messages) == 1, "Alert nie powinien przełamać cooldownu"

    monotonic_time["value"] = 140.0

    after_cooldown = _make_snapshot(
        {
            "retry_backlog_before_send": 3,
            "retry_backlog_after_flush": 4,
        }
    )
    sink.handle_snapshot(after_cooldown)

    backlog_messages = [
        msg for msg in channel.messages if msg.category == "ui.performance.retry_backlog"
    ]
    assert len(backlog_messages) == 2, "Po cooldownie alert powinien zostać wysłany"
    assert backlog_messages[-1].context["retry_backlog_delta"] == "2"
    assert backlog_messages[-1].context.get("retry_backlog_started_at")


def test_retry_backlog_escalates_to_critical(tmp_path: Path) -> None:
    router, channel = _build_router()
    sink = UiTelemetryAlertSink(
        router,
        jsonl_path=tmp_path / "ui_alerts.jsonl",
        retry_backlog_threshold=3,
        retry_backlog_critical_threshold=6,
        retry_backlog_realert_delta=5,
    )

    warning_snapshot = _make_snapshot(
        {
            "retry_backlog_before_send": 2,
            "retry_backlog_after_flush": 4,
        }
    )
    sink.handle_snapshot(warning_snapshot)

    critical_snapshot = _make_snapshot(
        {
            "retry_backlog_before_send": 4,
            "retry_backlog_after_flush": 6,
        }
    )
    sink.handle_snapshot(critical_snapshot)

    backlog_messages = [
        message
        for message in channel.messages
        if message.category == "ui.performance.retry_backlog"
    ]
    assert len(backlog_messages) == 2
    assert backlog_messages[0].severity == "warning"
    assert backlog_messages[1].severity == "critical"
    assert backlog_messages[1].context["retry_backlog_severity"] == "critical"
    assert backlog_messages[1].context["retry_backlog_delta"] == "2"
    assert backlog_messages[0].context.get("retry_backlog_started_at")
    assert (
        backlog_messages[0].context.get("retry_backlog_started_at")
        == backlog_messages[1].context.get("retry_backlog_started_at")
    )

    records = [
        json.loads(line)
        for line in (tmp_path / "ui_alerts.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    critical_records = [
        record
        for record in records
        if record.get("category") == "ui.performance.retry_backlog"
        and record.get("severity") == "critical"
    ]
    assert critical_records
    assert critical_records[-1]["context"]["retry_backlog_severity"] == "critical"


def test_retry_backlog_escalates_after_duration(
    monkeypatch, tmp_path: Path
) -> None:
    router, channel = _build_router()
    jsonl_path = tmp_path / "ui_alerts.jsonl"
    monotonic_time = {"value": 100.0}

    def fake_monotonic() -> float:
        return monotonic_time["value"]

    monkeypatch.setattr(
        "bot_core.runtime.metrics_alerts.time.monotonic", fake_monotonic
    )

    sink = UiTelemetryAlertSink(
        router,
        jsonl_path=jsonl_path,
        retry_backlog_threshold=2,
        retry_backlog_realert_delta=5,
        retry_backlog_critical_after_seconds=120,
    )

    first_snapshot = _make_snapshot(
        {
            "retry_backlog_before_send": 1,
            "retry_backlog_after_flush": 2,
        }
    )
    sink.handle_snapshot(first_snapshot)

    assert channel.messages
    assert channel.messages[-1].severity == "warning"

    monotonic_time["value"] += 130.0
    followup_snapshot = _make_snapshot(
        {
            "retry_backlog_before_send": 2,
            "retry_backlog_after_flush": 2,
        }
    )
    sink.handle_snapshot(followup_snapshot)

    backlog_messages = [
        message
        for message in channel.messages
        if message.category == "ui.performance.retry_backlog"
    ]
    assert len(backlog_messages) == 2
    critical_message = backlog_messages[-1]
    assert critical_message.severity == "critical"
    assert critical_message.context["retry_backlog_severity"] == "critical"
    assert critical_message.context["retry_backlog_escalation"] == "duration"
    assert float(critical_message.context["retry_backlog_duration_seconds"]) == pytest.approx(
        130.0, rel=0.0, abs=0.01
    )
    assert critical_message.context.get("retry_backlog_started_at")
    assert "czas degradacji" in critical_message.body.lower()

    records = [
        json.loads(line)
        for line in jsonl_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    critical_records = [
        record
        for record in records
        if record.get("category") == "ui.performance.retry_backlog"
        and record.get("severity") == "critical"
    ]
    assert critical_records
    context = critical_records[-1]["context"]
    assert context["retry_backlog_escalation"] == "duration"
    assert float(context["retry_backlog_duration_seconds"]) == pytest.approx(
        130.0, rel=0.0, abs=0.01
    )
    assert context.get("retry_backlog_started_at")


def test_tag_inactivity_alerts_and_recovers(monkeypatch, tmp_path: Path) -> None:
    router, channel = _build_router()
    jsonl_path = tmp_path / "ui_alerts.jsonl"
    monotonic_time = {"value": 100.0}

    def fake_monotonic() -> float:
        return monotonic_time["value"]

    monkeypatch.setattr("bot_core.runtime.metrics_alerts.time.monotonic", fake_monotonic)

    sink = UiTelemetryAlertSink(
        router,
        jsonl_path=jsonl_path,
        tag_inactivity_threshold_seconds=30,
    )

    first_snapshot = _make_snapshot({"event": "heartbeat", "tag": "desk-1"})
    sink.handle_snapshot(first_snapshot)

    monotonic_time["value"] += 45.0
    trigger_snapshot = _make_snapshot({"event": "heartbeat", "tag": "desk-2"})
    sink.handle_snapshot(trigger_snapshot)

    inactivity_messages = [
        message
        for message in channel.messages
        if message.category == "ui.availability.tag_inactivity"
    ]
    assert inactivity_messages, "Alert o braku telemetrii powinien zostać wysłany"
    inactive_alert = inactivity_messages[-1]
    assert inactive_alert.severity == "warning"
    assert inactive_alert.context["tag"] == "desk-1"
    assert inactive_alert.context["tag_inactive"] == "true"
    assert "desk-1" in inactive_alert.body

    monotonic_time["value"] += 25.0
    recovery_snapshot = _make_snapshot({"event": "heartbeat", "tag": "desk-1"})
    sink.handle_snapshot(recovery_snapshot)

    inactivity_messages = [
        message
        for message in channel.messages
        if message.category == "ui.availability.tag_inactivity"
    ]
    assert len(inactivity_messages) == 2, "Powinien pojawić się alert o przywróceniu"
    recovered_alert = inactivity_messages[-1]
    assert recovered_alert.severity == "info"
    assert recovered_alert.context["tag_inactive"] == "false"
    assert "wznowił" in recovered_alert.body
    assert "tag_inactivity_duration_seconds" in recovered_alert.context

    records = [
        json.loads(line)
        for line in jsonl_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    tag_records = [
        record
        for record in records
        if record.get("category") == "ui.availability.tag_inactivity"
    ]
    assert len(tag_records) == 2
    assert tag_records[0]["severity"] == "warning"
    assert tag_records[1]["severity"] == "info"
    assert tag_records[0]["payload"]["event"] == "tag_inactivity"
    assert tag_records[1]["payload"]["event"] == "tag_inactivity_recovered"


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
        log_retry_backlog_events=False,
        enable_tag_inactivity_alerts=False,
        log_tag_inactivity_events=False,
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
        log_retry_backlog_events=False,
        enable_tag_inactivity_alerts=False,
        log_tag_inactivity_events=False,
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

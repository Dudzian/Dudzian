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
    assert message.severity in {"warning", "critical"}
    assert "5" in message.body


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

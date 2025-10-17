from __future__ import annotations

import json

from bot_core.observability.metrics import MetricsRegistry
from bot_core.observability.ui_metrics import UiTelemetryPrometheusExporter


class FakeSnapshot:
    def __init__(self, notes: dict[str, object], fps: float | None = None) -> None:
        self.notes = json.dumps(notes)
        self._has_fps = fps is not None
        if fps is not None:
            self.fps = float(fps)

    def HasField(self, name: str) -> bool:
        if name == "fps":
            return self._has_fps
        return False


class FakeAlertSink:
    def __init__(self) -> None:
        self.received: list[FakeSnapshot] = []

    def handle_snapshot(self, snapshot: FakeSnapshot) -> None:
        self.received.append(snapshot)


def _make_exporter(alert_sink: FakeAlertSink | None = None) -> tuple[UiTelemetryPrometheusExporter, MetricsRegistry]:
    registry = MetricsRegistry()
    sink = alert_sink or FakeAlertSink()
    exporter = UiTelemetryPrometheusExporter(registry=registry, alert_sink=sink)
    return exporter, registry


def test_updates_fps_and_window_count_gauges() -> None:
    alert_sink = FakeAlertSink()
    exporter, registry = _make_exporter(alert_sink)

    snapshot = FakeSnapshot(
        {
            "event": "reduce_motion",
            "active": True,
            "window_count": 2,
        },
        fps=55.5,
    )

    exporter.handle_snapshot(snapshot)

    fps_value = registry.gauge("bot_ui_fps", "").value()
    window_count = registry.gauge("bot_ui_window_count", "").value()
    reduce_motion_state = registry.gauge("bot_ui_reduce_motion_state", "").value()
    reduce_motion_events = registry.counter("bot_ui_reduce_motion_events_total", "").value(labels={"state": "active"})

    assert fps_value == 55.5
    assert window_count == 2
    assert reduce_motion_state == 1
    assert reduce_motion_events == 1

    # Drugi snapshot z tą samą flagą nie powinien zwiększyć licznika
    exporter.handle_snapshot(snapshot)
    reduce_motion_events_after = registry.counter("bot_ui_reduce_motion_events_total", "").value(labels={"state": "active"})
    assert reduce_motion_events_after == 1

    assert alert_sink.received, "Eksporter powinien przekazać snapshot do sinka alertów"


def test_records_screen_metrics_with_labels() -> None:
    exporter, registry = _make_exporter()

    snapshot = FakeSnapshot(
        {
            "event": "reduce_motion",
            "screen": {
                "name": "Dell U2720Q",
                "index": 1,
                "refresh_hz": 60.0,
                "device_pixel_ratio": 1.25,
                "geometry_px": {"width": 3840, "height": 2160},
            },
        },
        fps=58.0,
    )

    exporter.handle_snapshot(snapshot)

    labels = {"screen_index": "1", "screen_name": "Dell U2720Q"}
    refresh_value = registry.gauge("bot_ui_screen_refresh_hz", "").value(labels=labels)
    dpr_value = registry.gauge("bot_ui_screen_device_pixel_ratio", "").value(labels=labels)
    width_value = registry.gauge("bot_ui_screen_resolution_px", "").value(
        labels={**labels, "dimension": "width"}
    )
    height_value = registry.gauge("bot_ui_screen_resolution_px", "").value(
        labels={**labels, "dimension": "height"}
    )

    assert refresh_value == 60.0
    assert dpr_value == 1.25
    assert width_value == 3840
    assert height_value == 2160


def test_overlay_budget_updates_gauges() -> None:
    exporter, registry = _make_exporter()

    snapshot = FakeSnapshot(
        {
            "event": "overlay_budget",
            "active_overlays": 3,
            "allowed_overlays": 5,
        }
    )

    exporter.handle_snapshot(snapshot)

    active_value = registry.gauge("bot_ui_overlay_active", "").value()
    allowed_value = registry.gauge("bot_ui_overlay_allowed", "").value()

    assert active_value == 3
    assert allowed_value == 5


def test_retry_backlog_gauge_updates_from_payload() -> None:
    exporter, registry = _make_exporter()

    snapshot = FakeSnapshot(
        {
            "event": "reduce_motion",
            "retry_backlog_before_send": 4,
            "retry_backlog_after_flush": 1,
        }
    )

    exporter.handle_snapshot(snapshot)

    before_value = registry.gauge("bot_ui_retry_backlog", "").value(labels={"phase": "before_flush"})
    after_value = registry.gauge("bot_ui_retry_backlog", "").value(labels={"phase": "after_flush"})

    assert before_value == 4
    assert after_value == 1


def test_jank_spike_records_histogram_overrun() -> None:
    exporter, registry = _make_exporter()

    snapshot = FakeSnapshot(
        {
            "event": "jank_spike",
            "frame_ms": 28.0,
            "threshold_ms": 16.0,
        }
    )

    exporter.handle_snapshot(snapshot)

    histogram = registry.histogram(
        "bot_ui_jank_frame_overrun_ms",
        "",
        buckets=(1.0, 5.0, 10.0, 25.0, 50.0, 100.0),
    )
    state = histogram.snapshot()

    assert state.count == 1
    assert state.sum == 12.0
    assert state.counts[25.0] == 1


def test_ignores_events_without_numeric_payload() -> None:
    exporter, registry = _make_exporter()

    snapshot = FakeSnapshot({"event": "overlay_budget", "active_overlays": "n/a"})
    exporter.handle_snapshot(snapshot)

    active_value = registry.gauge("bot_ui_overlay_active", "").value()
    assert active_value == 0

    fps_snapshot = FakeSnapshot({"event": "reduce_motion"})
    exporter.handle_snapshot(fps_snapshot)

    fps_value = registry.gauge("bot_ui_fps", "").value()
    assert fps_value == 0

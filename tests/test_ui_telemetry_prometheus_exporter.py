from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from bot_core.observability.metrics import MetricsRegistry
from bot_core.observability.ui_metrics import UiTelemetryPrometheusExporter
from bot_core.observability import ui_metrics


class FakeSnapshot:
    def __init__(
        self,
        notes: dict[str, object],
        fps: float | None = None,
        *,
        generated_at: float | None = None,
    ) -> None:
        self.notes = json.dumps(notes)
        self._has_fps = fps is not None
        if fps is not None:
            self.fps = float(fps)
        if generated_at is not None:
            seconds = int(generated_at)
            nanos = int(round((generated_at - seconds) * 1_000_000_000))
            if nanos >= 1_000_000_000:
                seconds += 1
                nanos -= 1_000_000_000
            elif nanos < 0:
                seconds -= 1
                nanos += 1_000_000_000
            self.generated_at = SimpleNamespace(seconds=seconds, nanos=nanos)

    def HasField(self, name: str) -> bool:
        if name == "fps":
            return self._has_fps
        return False


class FakeAlertSink:
    def __init__(self) -> None:
        self.received: list[FakeSnapshot] = []

    def handle_snapshot(self, snapshot: FakeSnapshot) -> None:
        self.received.append(snapshot)


def _make_exporter(
    alert_sink: FakeAlertSink | None = None,
    *,
    tag_activity_ttl_seconds: float = 300.0,
) -> tuple[UiTelemetryPrometheusExporter, MetricsRegistry]:
    registry = MetricsRegistry()
    sink = alert_sink or FakeAlertSink()
    exporter = UiTelemetryPrometheusExporter(
        registry=registry,
        alert_sink=sink,
        tag_activity_ttl_seconds=tag_activity_ttl_seconds,
    )
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


def test_tagged_snapshots_emit_labelled_metrics() -> None:
    exporter, registry = _make_exporter()

    snapshot = FakeSnapshot(
        {
            "event": "reduce_motion",
            "active": True,
            "window_count": 3,
            "retry_backlog_before_send": 6,
            "retry_backlog_after_flush": 2,
            "screen": {
                "name": "Dell U2720Q",
                "index": 1,
                "refresh_hz": 60,
                "device_pixel_ratio": 1.25,
                "geometry_px": {"width": 3840, "height": 2160},
            },
            "tag": "desk-a",
        },
        fps=58.0,
        generated_at=1_700_000_000.0,
    )

    exporter.handle_snapshot(snapshot)

    tag_labels = {"tag": "desk-a"}
    assert registry.counter("bot_ui_snapshots_total", "").value(labels=tag_labels) == 1
    assert registry.gauge("bot_ui_window_count", "").value(labels=tag_labels) == 3
    before_labels = {"phase": "before_flush", "tag": "desk-a"}
    after_labels = {"phase": "after_flush", "tag": "desk-a"}
    assert registry.gauge("bot_ui_retry_backlog", "").value(labels=before_labels) == 6
    assert registry.gauge("bot_ui_retry_backlog", "").value(labels=after_labels) == 2
    reduce_motion_labels = {"state": "active", "tag": "desk-a"}
    assert (
        registry.counter("bot_ui_reduce_motion_events_total", "").value(labels=reduce_motion_labels)
        == 1
    )
    screen_labels = {"screen_index": "1", "screen_name": "Dell U2720Q", "tag": "desk-a"}
    assert registry.gauge("bot_ui_screen_refresh_hz", "").value(labels=screen_labels) == 60


def test_tag_activity_metrics_follow_ttl(monkeypatch: pytest.MonkeyPatch) -> None:
    monotonic_time = 0.0
    wall_time = 1_700_000_000.0

    def fake_monotonic() -> float:
        return monotonic_time

    def fake_time() -> float:
        return wall_time

    monkeypatch.setattr(ui_metrics.time, "monotonic", lambda: fake_monotonic())
    monkeypatch.setattr(ui_metrics.time, "time", lambda: fake_time())

    exporter, registry = _make_exporter()

    first_snapshot = FakeSnapshot(
        {
            "event": "reduce_motion",
            "active": True,
            "tag": "desk-a",
        },
        generated_at=wall_time,
    )

    exporter.handle_snapshot(first_snapshot)

    tag_labels = {"tag": "desk-a"}
    assert registry.gauge("bot_ui_tag_active", "").value(labels=tag_labels) == 1
    assert registry.gauge("bot_ui_tag_inactive", "").value(labels=tag_labels) == 0
    assert (
        registry.gauge("bot_ui_tag_last_seen_seconds", "").value(labels=tag_labels) == wall_time
    )
    assert registry.gauge("bot_ui_tag_active_count", "").value() == 1
    assert registry.gauge("bot_ui_tag_inactive_count", "").value() == 0

    inactive_age_initial = registry.gauge("bot_ui_tag_inactive_age_seconds", "").value(
        labels=tag_labels
    )
    assert inactive_age_initial == 0

    monotonic_time = 400.0
    wall_time += 400.0

    exporter.handle_snapshot(
        FakeSnapshot(
            {
                "event": "reduce_motion",
            },
            generated_at=wall_time,
        )
    )

    assert registry.gauge("bot_ui_tag_active", "").value(labels=tag_labels) == 0
    assert registry.gauge("bot_ui_tag_inactive", "").value(labels=tag_labels) == 1
    assert registry.gauge("bot_ui_tag_active_count", "").value() == 0
    assert registry.gauge("bot_ui_tag_inactive_count", "").value() == 1

    inactive_age = registry.gauge("bot_ui_tag_inactive_age_seconds", "").value(labels=tag_labels)
    assert inactive_age == pytest.approx(400.0, rel=1e-6)

    monotonic_time = 650.0
    wall_time += 250.0

    exporter.handle_snapshot(
        FakeSnapshot(
            {
                "event": "reduce_motion",
            },
            generated_at=wall_time,
        )
    )

    assert registry.gauge("bot_ui_tag_active", "").value(labels=tag_labels) == 0
    assert registry.gauge("bot_ui_tag_inactive", "").value(labels=tag_labels) == 0
    assert registry.gauge("bot_ui_tag_inactive_age_seconds", "").value(labels=tag_labels) == 0
    assert registry.gauge("bot_ui_tag_inactive_count", "").value() == 0
    assert registry.gauge("bot_ui_tag_last_seen_seconds", "").value(labels=tag_labels) == 0


def test_retry_incident_metrics_track_duration_and_histogram() -> None:
    exporter, registry = _make_exporter()

    start_ts = 1_700_000_000.0
    active_snapshot = FakeSnapshot(
        {
            "event": "reduce_motion",
            "retry_backlog_before_send": 0,
            "retry_backlog_after_flush": 3,
        },
        generated_at=start_ts,
    )

    exporter.handle_snapshot(active_snapshot)

    active_value = registry.gauge("bot_ui_retry_incident_active", "").value()
    age_value = registry.gauge("bot_ui_retry_incident_age_seconds", "").value()
    started_value = registry.gauge(
        "bot_ui_retry_incident_started_at_seconds",
        "",
    ).value()

    assert active_value == 1
    assert age_value == 0
    assert started_value == start_ts

    later_ts = start_ts + 12.5
    exporter.handle_snapshot(
        FakeSnapshot(
            {
                "event": "reduce_motion",
                "retry_backlog_before_send": 3,
                "retry_backlog_after_flush": 2,
            },
            generated_at=later_ts,
        )
    )

    age_value_later = registry.gauge("bot_ui_retry_incident_age_seconds", "").value()
    assert age_value_later == pytest.approx(12.5, rel=1e-3)

    recovery_ts = later_ts + 7.5
    exporter.handle_snapshot(
        FakeSnapshot(
            {
                "event": "reduce_motion",
                "retry_backlog_before_send": 2,
                "retry_backlog_after_flush": 0,
            },
            generated_at=recovery_ts,
        )
    )

    active_value_after = registry.gauge("bot_ui_retry_incident_active", "").value()
    age_value_after = registry.gauge("bot_ui_retry_incident_age_seconds", "").value()
    started_value_after = registry.gauge(
        "bot_ui_retry_incident_started_at_seconds",
        "",
    ).value()

    assert active_value_after == 0
    assert age_value_after == 0
    assert started_value_after == 0

    histogram = registry.histogram(
        "bot_ui_retry_incident_duration_seconds",
        "",
        buckets=(5.0, 10.0, 30.0, 60.0, 120.0, 300.0, 600.0, 1800.0, 3600.0),
    )
    histogram_state = histogram.snapshot()

    assert histogram_state.count == 1
    assert histogram_state.sum == pytest.approx(20.0, rel=1e-3)
    assert histogram_state.counts[30.0] == 1


def test_reduce_motion_incident_metrics_track_duration() -> None:
    exporter, registry = _make_exporter()

    start_ts = 1_700_000_100.0
    exporter.handle_snapshot(
        FakeSnapshot(
            {"event": "reduce_motion", "active": True},
            generated_at=start_ts,
        )
    )

    assert registry.gauge("bot_ui_reduce_motion_incident_active", "").value() == 1
    assert (
        registry.gauge("bot_ui_reduce_motion_incident_started_at_seconds", "").value()
        == start_ts
    )

    later_ts = start_ts + 12.5
    exporter.handle_snapshot(
        FakeSnapshot(
            {"event": "reduce_motion", "active": True},
            generated_at=later_ts,
        )
    )

    age_value = registry.gauge("bot_ui_reduce_motion_incident_age_seconds", "").value()
    assert age_value == pytest.approx(12.5, rel=1e-3)

    recovery_ts = later_ts + 8.0
    exporter.handle_snapshot(
        FakeSnapshot(
            {"event": "reduce_motion", "active": False},
            generated_at=recovery_ts,
        )
    )

    assert registry.gauge("bot_ui_reduce_motion_incident_active", "").value() == 0
    assert (
        registry.gauge("bot_ui_reduce_motion_incident_age_seconds", "").value() == 0
    )
    assert (
        registry.gauge("bot_ui_reduce_motion_incident_started_at_seconds", "").value()
        == 0
    )

    histogram = registry.histogram(
        "bot_ui_reduce_motion_incident_duration_seconds",
        "",
        buckets=(5.0, 15.0, 30.0, 60.0, 120.0, 300.0, 600.0, 1800.0),
    )
    histogram_state = histogram.snapshot()

    assert histogram_state.count == 1
    assert histogram_state.sum == pytest.approx(20.5, rel=1e-3)
    assert histogram_state.counts[30.0] == 1


def test_reduce_motion_incident_metrics_per_tag_follow_ttl(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monotonic_time = {"value": 0.0}
    wall_time = {"value": 1_700_000_200.0}

    monkeypatch.setattr(
        ui_metrics.time,
        "monotonic",
        lambda: monotonic_time["value"],
    )
    monkeypatch.setattr(
        ui_metrics.time,
        "time",
        lambda: wall_time["value"],
    )

    exporter, registry = _make_exporter(tag_activity_ttl_seconds=60.0)

    start_ts = wall_time["value"]
    exporter.handle_snapshot(
        FakeSnapshot(
            {"event": "reduce_motion", "active": True, "tag": "desk-a"},
            generated_at=start_ts,
        )
    )

    tag_labels = {"tag": "desk-a"}
    assert (
        registry.gauge("bot_ui_reduce_motion_incident_active", "").value(labels=tag_labels)
        == 1
    )

    wall_time["value"] += 30.0
    monotonic_time["value"] += 30.0
    later_ts = wall_time["value"]
    exporter.handle_snapshot(
        FakeSnapshot(
            {"event": "reduce_motion", "active": True, "tag": "desk-a"},
            generated_at=later_ts,
        )
    )

    age_value = registry.gauge(
        "bot_ui_reduce_motion_incident_age_seconds", ""
    ).value(labels=tag_labels)
    assert age_value == pytest.approx(30.0, rel=1e-6)

    wall_time["value"] += 70.0
    monotonic_time["value"] += 70.0
    exporter.handle_snapshot(FakeSnapshot({"event": "overlay_budget"}))

    assert (
        registry.gauge("bot_ui_reduce_motion_incident_active", "").value(labels=tag_labels)
        == 0
    )
    assert (
        registry.gauge(
            "bot_ui_reduce_motion_incident_started_at_seconds", ""
        ).value(labels=tag_labels)
        == 0
    )

    histogram = registry.histogram(
        "bot_ui_reduce_motion_incident_duration_seconds",
        "",
        buckets=(5.0, 15.0, 30.0, 60.0, 120.0, 300.0, 600.0, 1800.0),
    )
    histogram_state = histogram.snapshot(labels=tag_labels)

    assert histogram_state.count == 1
    assert histogram_state.sum == pytest.approx(30.0, rel=1e-6)
    assert histogram_state.counts[30.0] == 1


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


def test_snapshot_delivery_and_gap_metrics(monkeypatch: pytest.MonkeyPatch) -> None:
    exporter, registry = _make_exporter()

    class FakeTime:
        def __init__(self, values: list[float]) -> None:
            self._values = iter(values)
            self._last: float | None = None

        def time(self) -> float:
            try:
                self._last = next(self._values)
            except StopIteration:
                if self._last is None:
                    raise
                return self._last
            return self._last

    fake_time = FakeTime([1_700_000_010.0, 1_700_000_015.0])
    monkeypatch.setattr(ui_metrics, "time", fake_time)

    first_generated = 1_700_000_000.0
    second_generated = first_generated + 5.0

    exporter.handle_snapshot(
        FakeSnapshot(
            {
                "event": "reduce_motion",
                "retry_backlog_before_send": 0,
                "retry_backlog_after_flush": 0,
            },
            generated_at=first_generated,
        )
    )

    exporter.handle_snapshot(
        FakeSnapshot(
            {
                "event": "overlay_budget",
                "active_overlays": 1,
                "allowed_overlays": 3,
                "retry_backlog_before_send": 0,
                "retry_backlog_after_flush": 0,
            },
            generated_at=second_generated,
        )
    )

    total = registry.counter("bot_ui_snapshots_total", "").value()
    assert total == 2

    generated_value = registry.gauge("bot_ui_snapshot_generated_at_seconds", "").value()
    assert generated_value == second_generated

    latency_value = registry.gauge("bot_ui_snapshot_delivery_latency_seconds", "").value()
    assert latency_value == pytest.approx(10.0, rel=1e-6)

    gap_value = registry.gauge("bot_ui_snapshot_gap_seconds", "").value()
    assert gap_value == pytest.approx(5.0, rel=1e-6)

    histogram = registry.histogram(
        "bot_ui_snapshot_gap_duration_seconds",
        "",
        buckets=(0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0),
    )
    histogram_state = histogram.snapshot()
    assert histogram_state.count == 1
    assert histogram_state.sum == pytest.approx(5.0, rel=1e-6)
    assert histogram_state.counts[5.0] == 1

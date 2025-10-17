from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from bot_core.observability.metrics import HistogramMetric, MetricsRegistry
from bot_core.observability.ui_metrics import UiTelemetryPrometheusExporter
from bot_core.observability import ui_metrics


class FakeSnapshot:
    def __init__(
        self,
        notes: dict[str, object],
        fps: float | None = None,
        event_to_frame_p95_ms: float | None = None,
        cpu_utilization: float | None = None,
        gpu_utilization: float | None = None,
        ram_megabytes: float | None = None,
        dropped_frames: float | None = None,
        processed_messages_per_second: float | None = None,
        *,
        generated_at: float | None = None,
    ) -> None:
        self.notes = json.dumps(notes)
        self._has_fps = fps is not None
        if fps is not None:
            self.fps = float(fps)
        if event_to_frame_p95_ms is not None:
            self.event_to_frame_p95_ms = float(event_to_frame_p95_ms)
        if cpu_utilization is not None:
            self.cpu_utilization = float(cpu_utilization)
        if gpu_utilization is not None:
            self.gpu_utilization = float(gpu_utilization)
        if ram_megabytes is not None:
            self.ram_megabytes = float(ram_megabytes)
        if dropped_frames is not None:
            self.dropped_frames = float(dropped_frames)
        if processed_messages_per_second is not None:
            self.processed_messages_per_second = float(processed_messages_per_second)
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
    overlay_critical_difference_threshold: float | int | None = 2,
    overlay_critical_duration_threshold_seconds: float | int | None = None,
    jank_incident_quiet_seconds: float = 15.0,
    jank_critical_over_ms: float | int | None = None,
) -> tuple[UiTelemetryPrometheusExporter, MetricsRegistry]:
    registry = MetricsRegistry()
    sink = alert_sink or FakeAlertSink()
    exporter = UiTelemetryPrometheusExporter(
        registry=registry,
        alert_sink=sink,
        tag_activity_ttl_seconds=tag_activity_ttl_seconds,
        overlay_critical_difference_threshold=overlay_critical_difference_threshold,
        overlay_critical_duration_threshold_seconds=
        overlay_critical_duration_threshold_seconds,
        jank_incident_quiet_seconds=jank_incident_quiet_seconds,
        jank_critical_over_ms=jank_critical_over_ms,
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


def test_records_event_to_frame_metrics_and_histograms() -> None:
    exporter, registry = _make_exporter()

    snapshot = FakeSnapshot({"event": "performance"}, event_to_frame_p95_ms=22.5)
    exporter.handle_snapshot(snapshot)

    gauge_value = registry.gauge("bot_ui_event_to_frame_p95_ms", "").value()
    histogram_metric = registry.get("bot_ui_event_to_frame_p95_ms_distribution")
    assert isinstance(histogram_metric, HistogramMetric)
    histogram = histogram_metric.snapshot()

    assert gauge_value == 22.5
    assert histogram.count == 1
    assert histogram.sum == 22.5

    tagged_snapshot = FakeSnapshot(
        {"event": "performance", "tag": "desk-a"}, event_to_frame_p95_ms=48.0
    )
    exporter.handle_snapshot(tagged_snapshot)

    tagged_gauge = registry.gauge("bot_ui_event_to_frame_p95_ms", "").value(
        labels={"tag": "desk-a"}
    )
    histogram_metric = registry.get("bot_ui_event_to_frame_p95_ms_distribution")
    assert isinstance(histogram_metric, HistogramMetric)
    tagged_histogram = histogram_metric.snapshot(labels={"tag": "desk-a"})
    global_histogram = histogram_metric.snapshot()

    assert tagged_gauge == 48.0
    assert tagged_histogram.count == 1
    assert tagged_histogram.sum == 48.0
    assert global_histogram.count == 2
    assert global_histogram.sum == pytest.approx(70.5)


def test_updates_resource_utilization_metrics_with_clamping() -> None:
    exporter, registry = _make_exporter()

    snapshot = FakeSnapshot(
        {"event": "performance", "tag": "desk-b"},
        cpu_utilization=-12.5,
        gpu_utilization=87.5,
        ram_megabytes=1024.0,
        dropped_frames=42,
        processed_messages_per_second=512.25,
    )

    exporter.handle_snapshot(snapshot)

    cpu_value = registry.gauge("bot_ui_cpu_utilization_percent", "").value()
    cpu_tag_value = registry.gauge("bot_ui_cpu_utilization_percent", "").value(
        labels={"tag": "desk-b"}
    )
    gpu_value = registry.gauge("bot_ui_gpu_utilization_percent", "").value(
        labels={"tag": "desk-b"}
    )
    ram_value = registry.gauge("bot_ui_ram_usage_megabytes", "").value(
        labels={"tag": "desk-b"}
    )
    dropped_value = registry.gauge("bot_ui_dropped_frames_total", "").value(
        labels={"tag": "desk-b"}
    )
    processed_value = registry.gauge(
        "bot_ui_processed_messages_per_second", ""
    ).value(labels={"tag": "desk-b"})

    assert cpu_value == 0.0
    assert cpu_tag_value == 0.0
    assert gpu_value == 87.5
    assert ram_value == 1024.0
    assert dropped_value == 42.0
    assert processed_value == pytest.approx(512.25)


def test_performance_severity_metrics_and_recovery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exporter, registry = _make_exporter()

    current_time = 1_000.0

    def fake_time() -> float:
        return current_time

    monkeypatch.setattr(ui_metrics.time, "time", fake_time)
    monkeypatch.setattr(ui_metrics.time, "monotonic", fake_time)

    snapshot = FakeSnapshot(
        {"event": "performance"},
        event_to_frame_p95_ms=70.0,
        generated_at=current_time,
    )

    exporter.handle_snapshot(snapshot)

    state_value = registry.gauge("bot_ui_performance_metric_state", "").value(
        labels={"metric": "event_to_frame_p95_ms"}
    )
    incidents_total = registry.counter(
        "bot_ui_performance_incidents_total", ""
    ).value(labels={"metric": "event_to_frame_p95_ms", "severity": "critical"})
    active_value = registry.gauge("bot_ui_performance_incident_active", "").value(
        labels={"metric": "event_to_frame_p95_ms"}
    )

    assert state_value == 2.0
    assert incidents_total == 1.0
    assert active_value == 1.0

    current_time += 30.0
    recovery_snapshot = FakeSnapshot(
        {"event": "performance"},
        event_to_frame_p95_ms=20.0,
        generated_at=current_time,
    )

    exporter.handle_snapshot(recovery_snapshot)

    recovered_state = registry.gauge("bot_ui_performance_metric_state", "").value(
        labels={"metric": "event_to_frame_p95_ms"}
    )
    recovered_active = registry.gauge("bot_ui_performance_incident_active", "").value(
        labels={"metric": "event_to_frame_p95_ms"}
    )
    duration_metric = registry.get("bot_ui_performance_incident_duration_seconds")
    assert isinstance(duration_metric, HistogramMetric)
    duration_histogram = duration_metric.snapshot(
        labels={"metric": "event_to_frame_p95_ms"}
    )
    critical_transitions = registry.counter(
        "bot_ui_performance_severity_transitions_total", ""
    ).value(
        labels={
            "metric": "event_to_frame_p95_ms",
            "state": "critical",
            "reason": "critical_threshold",
        }
    )
    recovery_transitions = registry.counter(
        "bot_ui_performance_severity_transitions_total", ""
    ).value(
        labels={
            "metric": "event_to_frame_p95_ms",
            "state": "recovered",
            "reason": "recovered",
        }
    )

    assert recovered_state == 0.0
    assert recovered_active == 0.0
    assert duration_histogram.count == 1
    assert duration_histogram.sum == pytest.approx(30.0)
    assert critical_transitions == 1.0
    assert recovery_transitions == 1.0


def test_performance_severity_transitions_with_tag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exporter, registry = _make_exporter()

    current_time = 200.0

    def fake_time() -> float:
        return current_time

    monkeypatch.setattr(ui_metrics.time, "time", fake_time)
    monkeypatch.setattr(ui_metrics.time, "monotonic", fake_time)

    warning_snapshot = FakeSnapshot(
        {"event": "performance", "tag": "desk-7"},
        cpu_utilization=88.0,
        generated_at=current_time,
    )

    exporter.handle_snapshot(warning_snapshot)

    warning_state = registry.gauge("bot_ui_performance_metric_state", "").value(
        labels={"metric": "cpu_utilization", "tag": "desk-7"}
    )
    warning_incidents = registry.counter(
        "bot_ui_performance_incidents_total", ""
    ).value(
        labels={
            "metric": "cpu_utilization",
            "severity": "warning",
            "tag": "desk-7",
        }
    )

    assert warning_state == 1.0
    assert warning_incidents == 1.0

    current_time += 10.0
    critical_snapshot = FakeSnapshot(
        {"event": "performance", "tag": "desk-7"},
        cpu_utilization=99.0,
        generated_at=current_time,
    )

    exporter.handle_snapshot(critical_snapshot)

    critical_state = registry.gauge("bot_ui_performance_metric_state", "").value(
        labels={"metric": "cpu_utilization", "tag": "desk-7"}
    )
    promoted_transitions = registry.counter(
        "bot_ui_performance_severity_transitions_total", ""
    ).value(
        labels={
            "metric": "cpu_utilization",
            "state": "critical",
            "reason": "promoted",
            "tag": "desk-7",
        }
    )

    assert critical_state == 2.0
    assert promoted_transitions == 1.0

    current_time += 5.0
    demote_snapshot = FakeSnapshot(
        {"event": "performance", "tag": "desk-7"},
        cpu_utilization=90.0,
        generated_at=current_time,
    )

    exporter.handle_snapshot(demote_snapshot)

    demoted_state = registry.gauge("bot_ui_performance_metric_state", "").value(
        labels={"metric": "cpu_utilization", "tag": "desk-7"}
    )
    demoted_transitions = registry.counter(
        "bot_ui_performance_severity_transitions_total", ""
    ).value(
        labels={
            "metric": "cpu_utilization",
            "state": "warning",
            "reason": "demoted",
            "tag": "desk-7",
        }
    )

    assert demoted_state == 1.0
    assert demoted_transitions == 1.0

    current_time += 5.0
    recovery_snapshot = FakeSnapshot(
        {"event": "performance", "tag": "desk-7"},
        cpu_utilization=25.0,
        generated_at=current_time,
    )

    exporter.handle_snapshot(recovery_snapshot)

    recovered_state = registry.gauge("bot_ui_performance_metric_state", "").value(
        labels={"metric": "cpu_utilization", "tag": "desk-7"}
    )
    recovered_transitions = registry.counter(
        "bot_ui_performance_severity_transitions_total", ""
    ).value(
        labels={
            "metric": "cpu_utilization",
            "state": "recovered",
            "reason": "recovered",
            "tag": "desk-7",
        }
    )
    duration_metric = registry.get("bot_ui_performance_incident_duration_seconds")
    assert isinstance(duration_metric, HistogramMetric)
    tag_histogram = duration_metric.snapshot(
        labels={"metric": "cpu_utilization", "tag": "desk-7"}
    )
    critical_incidents = registry.counter(
        "bot_ui_performance_incidents_total", ""
    ).value(
        labels={
            "metric": "cpu_utilization",
            "severity": "critical",
            "tag": "desk-7",
        }
    )

    assert recovered_state == 0.0
    assert recovered_transitions == 1.0
    assert tag_histogram.count == 1
    assert tag_histogram.sum == pytest.approx(20.0)
    assert critical_incidents == 0.0

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
    ratio_value = registry.gauge("bot_ui_overlay_capacity_ratio", "").value()

    assert active_value == 3
    assert allowed_value == 5
    assert ratio_value == pytest.approx(3 / 5)


def test_overlay_incident_metrics_track_duration_and_histogram() -> None:
    exporter, registry = _make_exporter()

    start_ts = 1_700_000_000.0
    active_snapshot = FakeSnapshot(
        {
            "event": "overlay_budget",
            "active_overlays": 6,
            "allowed_overlays": 4,
        },
        generated_at=start_ts,
    )

    exporter.handle_snapshot(active_snapshot)

    active_value = registry.gauge("bot_ui_overlay_incident_active", "").value()
    age_value = registry.gauge("bot_ui_overlay_incident_age_seconds", "").value()
    started_value = registry.gauge(
        "bot_ui_overlay_incident_started_at_seconds", ""
    ).value()

    assert active_value == 1
    assert age_value == 0
    assert started_value == start_ts

    later_ts = start_ts + 12.5
    exporter.handle_snapshot(
        FakeSnapshot(
            {
                "event": "overlay_budget",
                "active_overlays": 7,
                "allowed_overlays": 4,
            },
            generated_at=later_ts,
        )
    )

    updated_age = registry.gauge("bot_ui_overlay_incident_age_seconds", "").value()
    assert updated_age == pytest.approx(12.5, rel=1e-3)

    recovered_ts = later_ts + 7.5
    exporter.handle_snapshot(
        FakeSnapshot(
            {
                "event": "overlay_budget",
                "active_overlays": 4,
                "allowed_overlays": 4,
            },
            generated_at=recovered_ts,
        )
    )

    final_active = registry.gauge("bot_ui_overlay_incident_active", "").value()
    final_age = registry.gauge("bot_ui_overlay_incident_age_seconds", "").value()
    final_started = registry.gauge(
        "bot_ui_overlay_incident_started_at_seconds", ""
    ).value()

    assert final_active == 0
    assert final_age == 0
    assert final_started == 0

    histogram = registry.histogram(
        "bot_ui_overlay_incident_duration_seconds",
        "",
        buckets=(5.0, 10.0, 30.0, 60.0, 120.0, 300.0, 600.0, 1800.0),
    )
    histogram_state = histogram.snapshot()
    assert histogram_state.count == 1
    assert histogram_state.sum == pytest.approx(20.0, rel=1e-6)
    assert histogram_state.counts[30.0] == 1


def test_overlay_violation_metrics_increment_counter_and_histogram() -> None:
    exporter, registry = _make_exporter()

    first_violation = FakeSnapshot(
        {
            "event": "overlay_budget",
            "active_overlays": 6,
            "allowed_overlays": 4,
        }
    )
    exporter.handle_snapshot(first_violation)

    violation_state = registry.gauge("bot_ui_overlay_violation_state", "").value()
    excess_value = registry.gauge("bot_ui_overlay_excess", "").value()
    incident_total = registry.counter("bot_ui_overlay_incidents_total", "").value()
    histogram = registry.histogram(
        "bot_ui_overlay_capacity_ratio_overrun",
        "",
        buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0),
    ).snapshot()

    assert violation_state == 1.0
    assert excess_value == pytest.approx(2.0)
    assert incident_total == 1
    assert histogram.count == 1
    assert histogram.sum == pytest.approx(0.5, rel=1e-6)
    assert histogram.counts[0.5] == 1

    exporter.handle_snapshot(first_violation)
    incident_total_after = registry.counter(
        "bot_ui_overlay_incidents_total", ""
    ).value()
    histogram_after = registry.histogram(
        "bot_ui_overlay_capacity_ratio_overrun",
        "",
        buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0),
    ).snapshot()
    assert incident_total_after == 1
    assert histogram_after.count == 2

    recovery = FakeSnapshot(
        {
            "event": "overlay_budget",
            "active_overlays": 3,
            "allowed_overlays": 4,
        }
    )
    exporter.handle_snapshot(recovery)

    violation_state_after = registry.gauge(
        "bot_ui_overlay_violation_state", ""
    ).value()
    excess_after = registry.gauge("bot_ui_overlay_excess", "").value()
    assert violation_state_after == 0.0
    assert excess_after == 0.0

    exporter.handle_snapshot(first_violation)
    incident_total_final = registry.counter(
        "bot_ui_overlay_incidents_total", ""
    ).value()
    assert incident_total_final == 2


def test_overlay_violation_metrics_with_tags() -> None:
    exporter, registry = _make_exporter()

    violation = FakeSnapshot(
        {
            "event": "overlay_budget",
            "active_overlays": 5,
            "allowed_overlays": 3,
            "tag": "desk-a",
        }
    )
    exporter.handle_snapshot(violation)

    tag_labels = {"tag": "desk-a"}
    violation_state = registry.gauge(
        "bot_ui_overlay_violation_state", ""
    ).value(labels=tag_labels)
    excess_value = registry.gauge("bot_ui_overlay_excess", "").value(
        labels=tag_labels
    )
    incident_total = registry.counter(
        "bot_ui_overlay_incidents_total", ""
    ).value(labels=tag_labels)
    histogram = registry.histogram(
        "bot_ui_overlay_capacity_ratio_overrun",
        "",
        buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0),
    ).snapshot(labels=tag_labels)

    assert violation_state == 1.0
    assert excess_value == pytest.approx(2.0)
    assert incident_total == 1
    assert histogram.count == 1
    assert histogram.sum == pytest.approx(2.0 / 3.0, rel=1e-6)

    exporter.handle_snapshot(
        FakeSnapshot(
            {
                "event": "overlay_budget",
                "active_overlays": 3,
                "allowed_overlays": 3,
                "tag": "desk-a",
            }
        )
    )

    violation_state_after = registry.gauge(
        "bot_ui_overlay_violation_state", ""
    ).value(labels=tag_labels)
    excess_after = registry.gauge("bot_ui_overlay_excess", "").value(
        labels=tag_labels
    )
    assert violation_state_after == 0.0
    assert excess_after == 0.0


def test_overlay_severity_metrics_track_transitions() -> None:
    exporter, registry = _make_exporter()

    warning_snapshot = FakeSnapshot(
        {
            "event": "overlay_budget",
            "active_overlays": 5,
            "allowed_overlays": 4,
        }
    )
    exporter.handle_snapshot(warning_snapshot)

    warning_gauge = registry.gauge(
        "bot_ui_overlay_violation_severity_state", ""
    ).value(labels={"severity": "warning"})
    critical_gauge = registry.gauge(
        "bot_ui_overlay_violation_severity_state", ""
    ).value(labels={"severity": "critical"})
    warning_events = registry.counter(
        "bot_ui_overlay_incident_events_total", ""
    ).value(labels={"severity": "warning"})
    critical_events = registry.counter(
        "bot_ui_overlay_incident_events_total", ""
    ).value(labels={"severity": "critical"})

    assert warning_gauge == 1.0
    assert critical_gauge == 0.0
    assert warning_events == 1
    assert critical_events == 0

    warning_transition = registry.counter(
        "bot_ui_overlay_severity_transitions_total", ""
    ).value(labels={"state": "warning", "reason": "violation"})
    assert warning_transition == 1

    critical_snapshot = FakeSnapshot(
        {
            "event": "overlay_budget",
            "active_overlays": 7,
            "allowed_overlays": 4,
        }
    )
    exporter.handle_snapshot(critical_snapshot)

    warning_gauge_after = registry.gauge(
        "bot_ui_overlay_violation_severity_state", ""
    ).value(labels={"severity": "warning"})
    critical_gauge_after = registry.gauge(
        "bot_ui_overlay_violation_severity_state", ""
    ).value(labels={"severity": "critical"})
    critical_events_after = registry.counter(
        "bot_ui_overlay_incident_events_total", ""
    ).value(labels={"severity": "critical"})

    assert warning_gauge_after == 0.0
    assert critical_gauge_after == 1.0
    assert critical_events_after == 1

    critical_transition = registry.counter(
        "bot_ui_overlay_severity_transitions_total", ""
    ).value(labels={"state": "critical", "reason": "difference_threshold"})
    assert critical_transition == 1

    exporter.handle_snapshot(critical_snapshot)
    critical_events_again = registry.counter(
        "bot_ui_overlay_incident_events_total", ""
    ).value(labels={"severity": "critical"})
    assert critical_events_again == 1

    critical_transition_again = registry.counter(
        "bot_ui_overlay_severity_transitions_total", ""
    ).value(labels={"state": "critical", "reason": "difference_threshold"})
    assert critical_transition_again == 1

    recovery_snapshot = FakeSnapshot(
        {
            "event": "overlay_budget",
            "active_overlays": 2,
            "allowed_overlays": 4,
        }
    )
    exporter.handle_snapshot(recovery_snapshot)

    warning_recovered = registry.gauge(
        "bot_ui_overlay_violation_severity_state", ""
    ).value(labels={"severity": "warning"})
    critical_recovered = registry.gauge(
        "bot_ui_overlay_violation_severity_state", ""
    ).value(labels={"severity": "critical"})
    assert warning_recovered == 0.0
    assert critical_recovered == 0.0

    recovery_transition = registry.counter(
        "bot_ui_overlay_severity_transitions_total", ""
    ).value(labels={"state": "recovered", "reason": "recovered"})
    assert recovery_transition == 1


def test_overlay_severity_metrics_with_tags() -> None:
    exporter, registry = _make_exporter()

    warning = FakeSnapshot(
        {
            "event": "overlay_budget",
            "active_overlays": 4,
            "allowed_overlays": 3,
            "tag": "desk-a",
        }
    )
    exporter.handle_snapshot(warning)

    warning_labels = {"severity": "warning", "tag": "desk-a"}
    critical_labels = {"severity": "critical", "tag": "desk-a"}

    warning_gauge = registry.gauge(
        "bot_ui_overlay_violation_severity_state", ""
    ).value(labels=warning_labels)
    warning_events = registry.counter(
        "bot_ui_overlay_incident_events_total", ""
    ).value(labels=warning_labels)

    assert warning_gauge == 1.0
    assert warning_events == 1

    warning_transition = registry.counter(
        "bot_ui_overlay_severity_transitions_total", ""
    ).value(labels={"state": "warning", "reason": "violation", "tag": "desk-a"})
    assert warning_transition == 1

    critical = FakeSnapshot(
        {
            "event": "overlay_budget",
            "active_overlays": 6,
            "allowed_overlays": 3,
            "tag": "desk-a",
        }
    )
    exporter.handle_snapshot(critical)

    critical_gauge = registry.gauge(
        "bot_ui_overlay_violation_severity_state", ""
    ).value(labels=critical_labels)
    critical_events = registry.counter(
        "bot_ui_overlay_incident_events_total", ""
    ).value(labels=critical_labels)
    warning_gauge_after = registry.gauge(
        "bot_ui_overlay_violation_severity_state", ""
    ).value(labels=warning_labels)

    assert critical_gauge == 1.0
    assert critical_events == 1
    assert warning_gauge_after == 0.0

    critical_transition = registry.counter(
        "bot_ui_overlay_severity_transitions_total", ""
    ).value(
        labels={
            "state": "critical",
            "reason": "difference_threshold",
            "tag": "desk-a",
        }
    )
    assert critical_transition == 1

    recovery = FakeSnapshot(
        {
            "event": "overlay_budget",
            "active_overlays": 3,
            "allowed_overlays": 3,
            "tag": "desk-a",
        }
    )
    exporter.handle_snapshot(recovery)

    warning_final = registry.gauge(
        "bot_ui_overlay_violation_severity_state", ""
    ).value(labels=warning_labels)
    critical_final = registry.gauge(
        "bot_ui_overlay_violation_severity_state", ""
    ).value(labels=critical_labels)
    assert warning_final == 0.0
    assert critical_final == 0.0

    recovery_transition = registry.counter(
        "bot_ui_overlay_severity_transitions_total", ""
    ).value(
        labels={"state": "recovered", "reason": "recovered", "tag": "desk-a"}
    )
    assert recovery_transition == 1


def test_overlay_severity_escalates_after_duration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = {"value": 1_000.0}

    def fake_time() -> float:
        return clock["value"]

    monkeypatch.setattr(ui_metrics.time, "time", fake_time)
    monkeypatch.setattr(ui_metrics.time, "monotonic", fake_time)

    exporter, registry = _make_exporter(
        overlay_critical_difference_threshold=None,
        overlay_critical_duration_threshold_seconds=10.0,
    )

    def make_snapshot() -> FakeSnapshot:
        return FakeSnapshot(
            {
                "event": "overlay_budget",
                "active_overlays": 5,
                "allowed_overlays": 4,
            },
            generated_at=clock["value"],
        )

    exporter.handle_snapshot(make_snapshot())

    warning_gauge = registry.gauge(
        "bot_ui_overlay_violation_severity_state", ""
    ).value(labels={"severity": "warning"})
    assert warning_gauge == 1.0

    clock["value"] += 8.0
    exporter.handle_snapshot(make_snapshot())

    warning_after = registry.gauge(
        "bot_ui_overlay_violation_severity_state", ""
    ).value(labels={"severity": "warning"})
    critical_after = registry.gauge(
        "bot_ui_overlay_violation_severity_state", ""
    ).value(labels={"severity": "critical"})
    assert warning_after == 1.0
    assert critical_after == 0.0

    clock["value"] += 5.0
    exporter.handle_snapshot(make_snapshot())

    warning_final = registry.gauge(
        "bot_ui_overlay_violation_severity_state", ""
    ).value(labels={"severity": "warning"})
    critical_final = registry.gauge(
        "bot_ui_overlay_violation_severity_state", ""
    ).value(labels={"severity": "critical"})
    critical_events = registry.counter(
        "bot_ui_overlay_incident_events_total", ""
    ).value(labels={"severity": "critical"})

    assert warning_final == 0.0
    assert critical_final == 1.0
    assert critical_events == 1

    duration_transition = registry.counter(
        "bot_ui_overlay_severity_transitions_total", ""
    ).value(labels={"state": "critical", "reason": "duration_threshold"})
    assert duration_transition == 1


def test_overlay_incident_metrics_finalized_on_tag_ttl(monkeypatch) -> None:
    monotonic_time = 100.0
    wall_time = 1_700_100_000.0

    def fake_monotonic() -> float:
        return monotonic_time

    def fake_time() -> float:
        return wall_time

    monkeypatch.setattr(ui_metrics.time, "monotonic", lambda: fake_monotonic())
    monkeypatch.setattr(ui_metrics.time, "time", lambda: fake_time())

    exporter, registry = _make_exporter(tag_activity_ttl_seconds=60.0)

    first_snapshot = FakeSnapshot(
        {
            "event": "overlay_budget",
            "active_overlays": 5,
            "allowed_overlays": 2,
            "tag": "desk-a",
        },
        generated_at=wall_time,
    )

    exporter.handle_snapshot(first_snapshot)

    tag_labels = {"tag": "desk-a"}
    assert (
        registry.gauge("bot_ui_overlay_incident_active", "").value(labels=tag_labels)
        == 1
    )

    monotonic_time += 10.0
    wall_time += 10.0

    exporter.handle_snapshot(
        FakeSnapshot(
            {
                "event": "overlay_budget",
                "active_overlays": 4,
                "allowed_overlays": 2,
                "tag": "desk-a",
            },
            generated_at=wall_time,
        )
    )

    age_value = registry.gauge("bot_ui_overlay_incident_age_seconds", "").value(
        labels=tag_labels
    )
    assert age_value == pytest.approx(10.0, rel=1e-6)

    monotonic_time += 120.0
    wall_time += 120.0

    exporter.handle_snapshot(
        FakeSnapshot(
            {
                "event": "reduce_motion",
            },
            generated_at=wall_time,
        )
    )

    final_active = registry.gauge("bot_ui_overlay_incident_active", "").value(
        labels=tag_labels
    )
    final_age = registry.gauge("bot_ui_overlay_incident_age_seconds", "").value(
        labels=tag_labels
    )
    final_started = registry.gauge(
        "bot_ui_overlay_incident_started_at_seconds", ""
    ).value(labels=tag_labels)

    assert final_active == 0
    assert final_age == 0
    assert final_started == 0

    histogram = registry.histogram(
        "bot_ui_overlay_incident_duration_seconds",
        "",
        buckets=(5.0, 10.0, 30.0, 60.0, 120.0, 300.0, 600.0, 1800.0),
    )
    histogram_state = histogram.snapshot(labels=tag_labels)
    assert histogram_state.count == 1
    assert histogram_state.sum == pytest.approx(10.0, rel=1e-6)
    assert histogram_state.counts[10.0] == 1

    critical_transition = registry.counter(
        "bot_ui_overlay_severity_transitions_total", ""
    ).value(
        labels={
            "state": "critical",
            "reason": "difference_threshold",
            "tag": "desk-a",
        }
    )
    inactive_transition = registry.counter(
        "bot_ui_overlay_severity_transitions_total", ""
    ).value(
        labels={
            "state": "recovered",
            "reason": "inactive",
            "tag": "desk-a",
        }
    )

    assert critical_transition == 1
    assert inactive_transition == 1


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


def test_jank_incident_metrics_finalize_after_quiet_period(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = {"value": 1_700_000_000.0}

    def fake_time() -> float:
        return clock["value"]

    monkeypatch.setattr(ui_metrics.time, "time", fake_time)
    monkeypatch.setattr(ui_metrics.time, "monotonic", fake_time)

    exporter, registry = _make_exporter(jank_incident_quiet_seconds=5.0)

    start_ts = clock["value"]
    exporter.handle_snapshot(
        FakeSnapshot(
            {
                "event": "jank_spike",
                "frame_ms": 30.0,
                "threshold_ms": 16.0,
            },
            generated_at=start_ts,
        )
    )

    active_value = registry.gauge("bot_ui_jank_incident_active", "").value()
    started_value = registry.gauge(
        "bot_ui_jank_incident_started_at_seconds", ""
    ).value()
    assert active_value == 1.0
    assert started_value == start_ts

    clock["value"] += 2.0
    exporter.handle_snapshot(
        FakeSnapshot(
            {
                "event": "jank_spike",
                "frame_ms": 28.0,
                "threshold_ms": 16.0,
            },
            generated_at=clock["value"],
        )
    )

    age_value = registry.gauge("bot_ui_jank_incident_age_seconds", "").value()
    assert age_value == pytest.approx(2.0, rel=1e-6)

    clock["value"] += 6.0
    exporter.handle_snapshot(
        FakeSnapshot({}, generated_at=clock["value"])
    )

    final_active = registry.gauge("bot_ui_jank_incident_active", "").value()
    final_age = registry.gauge("bot_ui_jank_incident_age_seconds", "").value()
    final_started = registry.gauge(
        "bot_ui_jank_incident_started_at_seconds", ""
    ).value()
    assert final_active == 0.0
    assert final_age == 0.0
    assert final_started == 0.0

    histogram = registry.histogram(
        "bot_ui_jank_incident_duration_seconds",
        "",
        buckets=(1.0, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0),
    )
    hist_state = histogram.snapshot()
    assert hist_state.count == 1
    assert hist_state.sum == pytest.approx(2.0, rel=1e-6)

    incidents_total = registry.counter(
        "bot_ui_jank_incidents_total", ""
    ).value()
    assert incidents_total == 1


def test_jank_incident_metrics_per_tag_follow_quiet_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = {"value": 2_000_000_000.0}

    def fake_time() -> float:
        return clock["value"]

    monkeypatch.setattr(ui_metrics.time, "time", fake_time)
    monkeypatch.setattr(ui_metrics.time, "monotonic", fake_time)

    exporter, registry = _make_exporter(jank_incident_quiet_seconds=5.0)

    start_ts = clock["value"]
    exporter.handle_snapshot(
        FakeSnapshot(
            {
                "event": "jank_spike",
                "frame_ms": 40.0,
                "threshold_ms": 16.0,
                "tag": "desk-a",
            },
            generated_at=start_ts,
        )
    )

    labels = {"tag": "desk-a"}
    assert (
        registry.gauge("bot_ui_jank_incident_active", "").value(labels=labels)
        == 1.0
    )
    assert (
        registry.counter("bot_ui_jank_incidents_total", "").value(labels=labels)
        == 1
    )

    clock["value"] += 1.0
    exporter.handle_snapshot(
        FakeSnapshot(
            {
                "event": "jank_spike",
                "frame_ms": 36.0,
                "threshold_ms": 16.0,
                "tag": "desk-a",
            },
            generated_at=clock["value"],
        )
    )

    age_value = registry.gauge(
        "bot_ui_jank_incident_age_seconds", ""
    ).value(labels=labels)
    assert age_value == pytest.approx(1.0, rel=1e-6)

    clock["value"] += 6.0
    exporter.handle_snapshot(
        FakeSnapshot(
            {"event": "reduce_motion", "active": False, "tag": "desk-b"},
            generated_at=clock["value"],
        )
    )

    final_active = registry.gauge(
        "bot_ui_jank_incident_active", ""
    ).value(labels=labels)
    final_age = registry.gauge(
        "bot_ui_jank_incident_age_seconds", ""
    ).value(labels=labels)
    final_started = registry.gauge(
        "bot_ui_jank_incident_started_at_seconds", ""
    ).value(labels=labels)

    assert final_active == 0.0
    assert final_age == 0.0
    assert final_started == 0.0

    histogram = registry.histogram(
        "bot_ui_jank_incident_duration_seconds",
        "",
        buckets=(1.0, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0),
    )
    hist_state = histogram.snapshot(labels=labels)
    assert hist_state.count == 1
    assert hist_state.sum == pytest.approx(1.0, rel=1e-6)




def test_jank_severity_metrics_track_transitions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = {"value": 1_600_000_000.0}

    def fake_time() -> float:
        return clock["value"]

    monkeypatch.setattr(ui_metrics.time, "time", fake_time)
    monkeypatch.setattr(ui_metrics.time, "monotonic", fake_time)

    exporter, registry = _make_exporter(
        jank_incident_quiet_seconds=3.0,
        jank_critical_over_ms=6.0,
    )

    start_ts = clock["value"]
    exporter.handle_snapshot(
        FakeSnapshot(
            {
                "event": "jank_spike",
                "frame_ms": 28.0,
                "threshold_ms": 18.0,
            },
            generated_at=start_ts,
        )
    )

    critical_labels = {"severity": "critical"}
    warning_labels = {"severity": "warning"}
    critical_state = registry.gauge(
        "bot_ui_jank_severity_state", ""
    ).value(labels=critical_labels)
    warning_state = registry.gauge(
        "bot_ui_jank_severity_state", ""
    ).value(labels=warning_labels)
    assert critical_state == 1.0
    assert warning_state == 0.0

    critical_transition = registry.counter(
        "bot_ui_jank_severity_transitions_total", ""
    ).value(labels={"state": "critical", "reason": "spike"})
    assert critical_transition == 1

    clock["value"] += 1.0
    exporter.handle_snapshot(
        FakeSnapshot(
            {
                "event": "jank_spike",
                "frame_ms": 24.0,
                "threshold_ms": 20.0,
            },
            generated_at=clock["value"],
        )
    )

    warning_state_after = registry.gauge(
        "bot_ui_jank_severity_state", ""
    ).value(labels=warning_labels)
    critical_state_after = registry.gauge(
        "bot_ui_jank_severity_state", ""
    ).value(labels=critical_labels)
    assert warning_state_after == 1.0
    assert critical_state_after == 0.0

    warning_transition = registry.counter(
        "bot_ui_jank_severity_transitions_total", ""
    ).value(labels={"state": "warning", "reason": "spike"})
    assert warning_transition == 1

    clock["value"] += 5.0
    exporter.handle_snapshot(
        FakeSnapshot(
            {},
            generated_at=clock["value"],
        )
    )

    critical_recovery = registry.gauge(
        "bot_ui_jank_severity_state", ""
    ).value(labels=critical_labels)
    warning_recovery = registry.gauge(
        "bot_ui_jank_severity_state", ""
    ).value(labels=warning_labels)
    assert critical_recovery == 0.0
    assert warning_recovery == 0.0

    recovery_transition = registry.counter(
        "bot_ui_jank_severity_transitions_total", ""
    ).value(labels={"state": "recovered", "reason": "quiet"})
    assert recovery_transition == 1


def test_jank_severity_metrics_with_tags(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = {"value": 1_700_000_000.0}

    def fake_time() -> float:
        return clock["value"]

    monkeypatch.setattr(ui_metrics.time, "time", fake_time)
    monkeypatch.setattr(ui_metrics.time, "monotonic", fake_time)

    exporter, registry = _make_exporter(
        jank_incident_quiet_seconds=4.0,
        jank_critical_over_ms=5.0,
    )

    tag = "desk-a"
    start_ts = clock["value"]
    exporter.handle_snapshot(
        FakeSnapshot(
            {
                "event": "jank_spike",
                "frame_ms": 26.0,
                "threshold_ms": 18.0,
                "tag": tag,
            },
            generated_at=start_ts,
        )
    )

    critical_labels = {"severity": "critical", "tag": tag}
    assert (
        registry.gauge("bot_ui_jank_severity_state", "").value(labels=critical_labels)
        == 1.0
    )
    assert (
        registry.counter(
            "bot_ui_jank_severity_transitions_total", ""
        ).value(labels={"state": "critical", "reason": "spike", "tag": tag})
        == 1
    )

    clock["value"] += 6.0
    exporter.handle_snapshot(
        FakeSnapshot(
            {},
            generated_at=clock["value"],
        )
    )

    assert (
        registry.gauge("bot_ui_jank_severity_state", "").value(labels=critical_labels)
        == 0.0
    )
    assert (
        registry.counter(
            "bot_ui_jank_severity_transitions_total", ""
        ).value(labels={"state": "recovered", "reason": "quiet", "tag": tag})
        == 1
    )

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

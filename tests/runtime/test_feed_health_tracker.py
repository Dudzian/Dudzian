from __future__ import annotations

from ui.backend.feed_health_tracker import FeedHealthTracker


def _build_tracker() -> FeedHealthTracker:
    def _percentile(values: list[float], percentile: float) -> float:
        ordered = sorted(float(v) for v in values)
        index = (len(ordered) - 1) * (percentile / 100.0)
        lower = int(index)
        upper = min(lower + 1, len(ordered) - 1)
        if lower == upper:
            return ordered[lower]
        weight = index - lower
        return ordered[lower] + (ordered[upper] - ordered[lower]) * weight

    return FeedHealthTracker(
        feed_channels=("decision_journal", "ai_governor"),
        latency_buffer_size=32,
        percentile_fn=_percentile,
    )


def test_feed_health_tracker_updates_health_payload() -> None:
    tracker = _build_tracker()
    tracker.reconnects = 2
    tracker.last_error = "timeout"
    tracker.latency_samples_for("grpc").extend([10.0, 20.0, 30.0, 40.0])
    payload, p95, p50 = tracker.update_feed_health(
        status="connected",
        reconnects=2,
        last_error="timeout",
        next_retry=None,
        latest_latency=40.0,
        transport_key="grpc",
        channel_status={
            "decision_journal": {"status": "connected", "lastError": ""},
            "ai_governor": {"status": "connected", "lastError": ""},
        },
    )
    assert payload["status"] == "connected"
    assert payload["reconnects"] == 2
    assert payload["lastError"] == "timeout"
    assert payload["p50LatencyMs"] == p50
    assert payload["p95LatencyMs"] == p95
    assert payload["transports"]["grpc"]["status"] == "connected"
    assert tracker.feed_health is payload


def test_feed_health_tracker_tracks_downtime_transitions() -> None:
    tracker = _build_tracker()
    tracker.mark_feed_disconnected()
    assert tracker.downtime_started is not None
    tracker.mark_feed_connected()
    assert tracker.downtime_started is None
    assert tracker.downtime_total >= 0.0


def test_feed_health_tracker_reconnect_counter_contract() -> None:
    tracker = _build_tracker()
    tracker.reconnects = 3
    tracker.update_feed_health(
        status="degraded",
        reconnects=None,
        last_error="timeout",
        next_retry=2.0,
        latest_latency=None,
        transport_key="fallback",
        channel_status={"decision_journal": {"status": "degraded", "lastError": "timeout"}},
    )
    assert tracker.feed_health["reconnects"] == 3
    tracker.update_feed_health(
        status="connected",
        reconnects=0,
        last_error="",
        next_retry=None,
        latest_latency=10.0,
        transport_key="grpc",
        channel_status={"decision_journal": {"status": "connected", "lastError": ""}},
    )
    assert tracker.feed_health["reconnects"] == 0


def test_feed_health_tracker_builds_sla_report() -> None:
    tracker = _build_tracker()
    tracker.update_feed_health(
        status="connected",
        reconnects=4,
        last_error="",
        next_retry=None,
        latest_latency=3200.0,
        transport_key="grpc",
        channel_status={"decision_journal": {"status": "connected", "lastError": ""}},
    )
    tracker.latency_samples_for("grpc").extend([2000.0, 3200.0, 5100.0])
    report = tracker.build_sla_report(
        transport_source="grpc",
        thresholds={
            "latency_warning_ms": 2500.0,
            "latency_critical_ms": 5000.0,
            "reconnects_warning": 3.0,
            "reconnects_critical": 6.0,
            "downtime_warning_seconds": 30.0,
            "downtime_critical_seconds": 120.0,
        },
    )
    assert report["latency_state"] == "warning"
    assert report["reconnects_state"] == "warning"
    assert report["sla_state"] == "warning"


def test_feed_health_tracker_sla_antiflap_and_reference_stability() -> None:
    tracker = _build_tracker()
    tracker.latency_samples_for("grpc").extend([100.0, 120.0, 130.0])
    tracker.update_feed_health(
        status="connected",
        reconnects=0,
        last_error="",
        next_retry=None,
        latest_latency=130.0,
        transport_key="grpc",
        channel_status={"decision_journal": {"status": "connected", "lastError": ""}},
    )
    first = tracker.build_sla_report(
        transport_source="grpc",
        thresholds={
            "latency_warning_ms": 2500.0,
            "latency_critical_ms": 5000.0,
            "reconnects_warning": 3.0,
            "reconnects_critical": 6.0,
            "downtime_warning_seconds": 30.0,
            "downtime_critical_seconds": 120.0,
        },
    )
    tracker.latency_samples_for("grpc").clear()
    tracker.latency_samples_for("grpc").extend([5100.0, 5200.0])
    second = tracker.build_sla_report(
        transport_source="grpc",
        thresholds={
            "latency_warning_ms": 2500.0,
            "latency_critical_ms": 5000.0,
            "reconnects_warning": 3.0,
            "reconnects_critical": 6.0,
            "downtime_warning_seconds": 30.0,
            "downtime_critical_seconds": 120.0,
        },
    )
    assert first is second
    assert second["consecutive_healthy_periods"] == 0
    assert second["consecutive_degraded_periods"] >= 1

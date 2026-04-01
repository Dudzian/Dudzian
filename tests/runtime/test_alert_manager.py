from __future__ import annotations

from types import SimpleNamespace

import pytest

from ui.backend.alert_manager import AlertManager


def _build_manager(
    *,
    runtime_config: object | None = None,
    sink: object | None = None,
    active_profile: str = "test",
    history_hook: list[str] | None = None,
    channels_hook: list[str] | None = None,
) -> AlertManager:
    history_calls = history_hook if history_hook is not None else []
    channel_calls = channels_hook if channels_hook is not None else []
    return AlertManager(
        runtime_config_loader=lambda: runtime_config,
        active_profile_loader=lambda: active_profile,
        sink_loader=lambda: sink,
        risk_diagnostics_normalizer=lambda payload: dict(payload),
        mapping_normalizer=lambda value: value if isinstance(value, dict) else {},
        history_changed=lambda: history_calls.append("history"),
        channels_changed=lambda: channel_calls.append("channels"),
    )


def test_feed_alert_dedupe_and_recovery_history_contract() -> None:
    emitted: list[dict[str, object]] = []

    class _Sink:
        def emit_feed_health_event(self, **kwargs: object) -> None:
            emitted.append(dict(kwargs))

    history_calls: list[str] = []
    manager = _build_manager(sink=_Sink(), history_hook=history_calls)

    kwargs = {
        "metric_label": "Latencja p95 decision feedu",
        "unit": "ms",
        "value": 3100.0,
        "warning": 2500.0,
        "critical": 5000.0,
        "status": "connected",
        "adapter": "grpc",
        "reconnects": 0,
        "downtime_seconds": 0.0,
        "latency_p95": 3100.0,
        "last_error": "",
    }
    manager.maybe_emit_feed_alert("latency", "warning", **kwargs)
    manager.maybe_emit_feed_alert("latency", "warning", **kwargs)
    manager.maybe_emit_feed_alert("latency", "ok", **kwargs)
    manager.maybe_emit_feed_alert("latency", "ok", **kwargs)

    assert len(emitted) == 2
    assert emitted[0]["severity"] == "warning"
    assert emitted[1]["severity"] == "info"
    assert len(manager.feed_alert_history) == 2
    assert list(manager.feed_alert_history)[0]["state"] == "recovered"
    assert list(manager.feed_alert_history)[1]["state"] == "degraded"
    assert history_calls == ["history", "history"]


def test_record_history_and_refresh_channels_only_on_change() -> None:
    history_calls: list[str] = []
    channel_calls: list[str] = []
    manager = _build_manager(history_hook=history_calls, channels_hook=channel_calls)

    class _Router:
        def __init__(self) -> None:
            self.state: dict[str, dict[str, object]] = {"sink": {"status": "ok"}}

        def health_snapshot(self) -> dict[str, dict[str, object]]:
            return self.state

    router = _Router()
    manager.record_feed_alert(
        severity="warning",
        state="degraded",
        metric="latency",
        label="Latencja p95 decision feedu",
        unit="ms",
        value=3000.0,
        warning=2500.0,
        critical=5000.0,
        adapter="grpc",
        status="connected",
        reconnects=1,
        downtime_seconds=0.5,
        latency_p95=3000.0,
        last_error="",
        router=router,
    )
    manager.refresh_alert_channels(router)
    router.state = {"sink": {"status": "warning"}}
    manager.refresh_alert_channels(router)

    assert len(manager.feed_alert_history) == 1
    assert history_calls == ["history"]
    assert channel_calls == ["channels", "channels"]
    assert manager.feed_alert_channels[0]["name"] == "sink"
    assert manager.feed_alert_channels[0]["status"] == "warning"


def test_threshold_precedence_and_invalid_env(monkeypatch: pytest.MonkeyPatch) -> None:
    runtime_config = SimpleNamespace(
        observability=SimpleNamespace(
            feed_sla=SimpleNamespace(
                latency_warning_ms=1200.0,
                latency_critical_ms=2200.0,
                reconnects_warning=7,
                reconnects_critical=11,
                downtime_warning_seconds=8.0,
                downtime_critical_seconds=19.0,
            )
        )
    )
    manager = _build_manager(runtime_config=runtime_config)
    assert manager.feed_thresholds["latency_warning_ms"] == 1200.0
    assert manager.feed_thresholds["reconnects_warning"] == 7.0

    monkeypatch.setenv("BOT_CORE_UI_FEED_LATENCY_P95_WARNING_MS", "3333")
    monkeypatch.setenv("BOT_CORE_UI_FEED_RECONNECT_WARNING", "0")
    monkeypatch.setenv("BOT_CORE_UI_FEED_DOWNTIME_CRITICAL_SECONDS", "bad-value")
    refreshed = manager.reload_feed_thresholds()

    assert refreshed is manager.feed_thresholds
    assert manager.feed_thresholds["latency_warning_ms"] == 3333.0
    assert manager.feed_thresholds["reconnects_warning"] is None
    assert manager.feed_thresholds["downtime_critical_seconds"] == 19.0


def test_risk_journal_dedupe_payload_and_no_sink_behavior() -> None:
    metrics_calls: list[dict[str, object]] = []
    logs: list[tuple[str, str]] = []
    manager = _build_manager(sink=None, active_profile="stage")

    payload = {
        "incompleteEntries": 2,
        "incompleteSamples": [{"id": "a"}],
        "incompleteSamplesCount": 1,
        "riskFlagCounts": {"freeze": 2},
    }
    manager.maybe_emit_risk_journal_alert(
        diagnostics=payload,
        logger_warning=lambda msg: logs.append(("warning", msg)),
        logger_info=lambda msg: logs.append(("info", msg)),
        metrics_record=lambda **kwargs: metrics_calls.append(dict(kwargs)),
    )
    manager.maybe_emit_risk_journal_alert(
        diagnostics=payload,
        logger_warning=lambda msg: logs.append(("warning", msg)),
        logger_info=lambda msg: logs.append(("info", msg)),
        metrics_record=lambda **kwargs: metrics_calls.append(dict(kwargs)),
    )
    manager.maybe_emit_risk_journal_alert(
        diagnostics={
            "incompleteEntries": 0,
            "incompleteSamples": [],
            "incompleteSamplesCount": 0,
            "riskFlagCounts": {},
        },
        logger_warning=lambda msg: logs.append(("warning", msg)),
        logger_info=lambda msg: logs.append(("info", msg)),
        metrics_record=lambda **kwargs: metrics_calls.append(dict(kwargs)),
    )

    assert len(metrics_calls) == 2
    assert metrics_calls[0]["state"] == "warning"
    assert metrics_calls[0]["labels"] == {"environment": "stage"}
    assert metrics_calls[1]["state"] == "ok"
    assert logs[0][0] == "warning"
    assert logs[1][0] == "info"


def test_risk_journal_with_sink_and_router_fallback_property() -> None:
    emitted: list[dict[str, object]] = []

    class _Sink:
        def __init__(self) -> None:
            self._router = object()

        def emit_feed_health_event(self, **kwargs: object) -> None:
            emitted.append(dict(kwargs))

    manager = _build_manager(sink=_Sink(), active_profile="prod")
    manager.maybe_emit_risk_journal_alert(
        diagnostics={
            "incompleteEntries": 1,
            "incompleteSamples": [{"id": "x"}],
            "incompleteSamplesCount": 1,
            "riskFlagCounts": {"risk_block": 1},
        },
        logger_warning=lambda _msg: None,
        logger_info=lambda _msg: None,
        metrics_record=lambda **_kwargs: None,
    )

    assert len(emitted) == 1
    assert emitted[0]["severity"] == "warning"
    assert emitted[0]["context"]["channel"] == "risk_journal"
    assert emitted[0]["payload"]["riskFlagCounts"] == {"risk_block": 1}

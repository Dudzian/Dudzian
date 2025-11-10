import json
from pathlib import Path

import pytest

from bot_core.alerts import AlertChannel, AlertMessage, DefaultAlertRouter, InMemoryAlertAuditLog
from bot_core.runtime import metrics_alerts
from bot_core.runtime.metrics_alerts import UiTelemetryAlertSink


class _DummyChannel(AlertChannel):
    def __init__(self, name: str = "hypercare-test") -> None:
        self.name = name
        self.messages: list[AlertMessage] = []

    def send(self, message: AlertMessage) -> None:  # pragma: no cover - simple append
        self.messages.append(message)

    def health_check(self) -> dict[str, str]:  # pragma: no cover - not relevant for tests
        return {"status": "ok"}


@pytest.fixture(autouse=True)
def _reset_feed_health_sink() -> None:
    metrics_alerts.reset_feed_health_alert_sink()
    yield
    metrics_alerts.reset_feed_health_alert_sink()


def test_emit_feed_health_event_dispatches_and_logs(tmp_path: Path) -> None:
    audit_log = InMemoryAlertAuditLog()
    router = DefaultAlertRouter(audit_log=audit_log)
    channel = _DummyChannel()
    router.register(channel)

    sink = UiTelemetryAlertSink(
        router,
        jsonl_path=tmp_path / "alerts.jsonl",
        log_reduce_motion_events=False,
        log_reduce_motion_incident_events=False,
        log_overlay_events=False,
        log_jank_events=False,
        log_retry_backlog_events=False,
        log_tag_inactivity_events=False,
        log_performance_events=False,
    )

    sink.emit_feed_health_event(
        severity="critical",
        title="Latency threshold exceeded",
        body="Test notification",
        context={"adapter": "grpc", "metric": "latency"},
        payload={"metric": "latency", "metric_value": 123.0},
    )

    assert channel.messages, "Kanał HyperCare powinien otrzymać komunikat o degradacji feedu"
    message = channel.messages[-1]
    assert message.title == "Latency threshold exceeded"
    assert message.severity == "critical"
    assert message.context["adapter"] == "grpc"

    audit_entries = list(audit_log.export())
    assert audit_entries, "Alert powinien zostać zapisany w audycie routera"
    assert audit_entries[0]["category"] == message.category

    jsonl_path = tmp_path / "alerts.jsonl"
    assert jsonl_path.exists(), "Plik JSONL z alertami feedu powinien zostać utworzony"
    record = json.loads(jsonl_path.read_text(encoding="utf-8").splitlines()[-1])
    assert record["category"] == message.category
    assert record["payload"]["event"] == "feed_health"
    assert record["context"]["metric"] == "latency"


def test_get_feed_health_alert_sink_uses_provided_router(tmp_path: Path) -> None:
    audit_log = InMemoryAlertAuditLog()
    custom_router = DefaultAlertRouter(audit_log=audit_log)
    sink = metrics_alerts.get_feed_health_alert_sink(
        router=custom_router, jsonl_path=tmp_path / "custom.jsonl"
    )
    assert sink is not None
    assert sink._router is custom_router  # type: ignore[attr-defined]
    assert sink.jsonl_path == tmp_path / "custom.jsonl"

    cached = metrics_alerts.get_feed_health_alert_sink()
    assert cached is sink, "Sink powinien być singletonem w module"


def test_get_feed_health_alert_sink_falls_back_to_memory_router(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    called: list[None] = []

    def _fake_builder() -> None:
        called.append(None)
        return None

    monkeypatch.setattr(metrics_alerts, "_build_feed_alert_router", _fake_builder)

    sink = metrics_alerts.get_feed_health_alert_sink(jsonl_path=tmp_path / "fallback.jsonl")
    assert sink is not None
    assert isinstance(sink._router, DefaultAlertRouter)  # type: ignore[attr-defined]
    assert isinstance(sink._router.audit_log, InMemoryAlertAuditLog)  # type: ignore[attr-defined]
    assert called, "Powinien zostać wykonany bootstrap kanałów HyperCare"

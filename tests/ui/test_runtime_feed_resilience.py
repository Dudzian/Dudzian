from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import pytest

from tests.ui._qt import require_pyside6

require_pyside6()

import ui.backend.runtime_service as runtime_service_module
from ui.backend.runtime_service import RuntimeService


class _RecordingSink:
    def __init__(self, jsonl_path: Path) -> None:
        self.jsonl_path = jsonl_path
        self.events: list[dict[str, Any]] = []

    def emit_feed_health_event(self, **payload: object) -> None:
        self.events.append(dict(payload))
        self.jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        with self.jsonl_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


class _Clock:
    def __init__(self, sequence: Iterable[float]) -> None:
        self._values = list(sequence)
        self._index = 0
        self.last: float | None = None

    def __call__(self) -> float:
        if self._index < len(self._values):
            self.last = self._values[self._index]
            self._index += 1
        return self.last if self.last is not None else 0.0


@pytest.mark.timeout(30)
def test_feed_alert_sink_emits_and_logs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("BOT_CORE_UI_FEED_LATENCY_P95_WARNING_MS", "1.0")
    monkeypatch.setenv("BOT_CORE_UI_FEED_LATENCY_P95_CRITICAL_MS", "2.0")
    monkeypatch.setenv("BOT_CORE_UI_FEED_RECONNECT_WARNING", "5")
    monkeypatch.setenv("BOT_CORE_UI_FEED_RECONNECT_CRITICAL", "10")

    alerts_path = tmp_path / "logs" / "ui_telemetry_alerts.jsonl"
    sink = _RecordingSink(alerts_path)
    service = RuntimeService(decision_loader=lambda limit: [], feed_alert_sink=sink)
    service._active_stream_label = "grpc://demo"

    samples = service._latency_samples_for("grpc")
    samples.clear()
    samples.append(1.5)
    service._update_feed_health(status="connected", reconnects=0, last_error="powolna odpowiedź")

    samples.clear()
    samples.append(2.5)
    service._update_feed_health(status="connected", reconnects=1, last_error="timeout")

    samples.clear()
    samples.append(0.2)
    service._update_feed_health(status="connected", reconnects=1, last_error="")

    severities = [event.get("severity") for event in sink.events]
    states = [event.get("payload", {}).get("state") for event in sink.events]
    assert severities == ["warning", "critical", "info"]
    assert states == ["degraded", "degraded", "recovered"]

    lines = alerts_path.read_text(encoding="utf-8").splitlines()
    parsed = [json.loads(line) for line in lines]
    assert len(parsed) == 3
    assert {entry.get("payload", {}).get("state") for entry in parsed} == {"degraded", "recovered"}
    assert parsed[-1]["payload"]["metric"] == "latency"


@pytest.mark.timeout(30)
def test_feed_health_degradation_and_recovery(monkeypatch: pytest.MonkeyPatch) -> None:
    clock = _Clock([100.0, 102.0, 105.0, 110.0])
    monkeypatch.setattr(runtime_service_module.time, "monotonic", clock)
    monkeypatch.setattr(runtime_service_module.time, "time", lambda: 1_700_000_000.0)

    service = RuntimeService(decision_loader=lambda limit: [])
    service._active_stream_label = "grpc://localhost:50051"

    service._feed_downtime_started = clock()
    service._feed_downtime_total = 1.25
    service._set_channel_status("ai_governor", "connected", metadata={"latencyMs": None})

    service._update_feed_health(
        status="degraded",
        reconnects=1,
        last_error="brak metadanych feedu",
        next_retry=3.0,
    )

    degraded = service.feedTransportSnapshot
    assert degraded["status"] == "degraded"
    assert degraded["nextRetrySeconds"] == pytest.approx(3.0)
    assert degraded["latencyP95"] is None
    assert degraded["channels"]

    service._feed_downtime_started = None
    service._feed_downtime_total = 0.0
    service._update_feed_health(status="connected", reconnects=2, last_error="")

    recovered = service.feedTransportSnapshot
    assert recovered["status"] == "connected"
    assert recovered["mode"] == "grpc"
    assert recovered["nextRetrySeconds"] is None
    assert recovered["lastError"] == ""

    health = service.feedHealth
    assert health["status"] == "connected"
    assert "ai_governor" in health["channelStates"]
    assert health["channelStates"]["ai_governor"]["status"] == "connected"

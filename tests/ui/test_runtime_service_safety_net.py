from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from tests.ui._qt import require_pyside6

require_pyside6()

import ui.backend.runtime_service as runtime_service_module
from ui.backend.runtime_service import RuntimeService


def test_runtime_service_fallback_path_grpc_to_jsonl_then_demo(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(RuntimeService, "_auto_connect_grpc", lambda self: None)
    monkeypatch.setenv("BOT_CORE_UI_GRPC_ENDPOINT", "localhost:50051")

    service = RuntimeService(decision_loader=lambda limit: [])
    monkeypatch.setattr(
        service,
        "_start_grpc_stream",
        lambda target, limit: (_ for _ in ()).throw(RuntimeError("grpc down")),
    )

    jsonl_entry = {
        "event": "risk_blocked",
        "timestamp": "2025-01-02T09:15:00+00:00",
        "decision": {"state": "hold", "shouldTrade": False},
    }

    def _loader(limit: int):
        return [jsonl_entry]

    jsonl_path = tmp_path / "decision-log.jsonl"
    monkeypatch.setattr(service, "_build_live_loader", lambda profile: (_loader, jsonl_path))

    assert service.attachToLiveDecisionLog("paper") is True
    assert Path(service.activeDecisionLogPath) == jsonl_path
    assert service.loadRecentDecisions(5)[0]["event"] == "risk_blocked"
    assert service.feedHealth["status"] == "fallback"
    assert service.feedTransportSnapshot["mode"] == "file"
    assert service.feedTransportSnapshot["adapter"] == "jsonl"

    monkeypatch.setattr(
        service,
        "_build_live_loader",
        lambda profile: (_ for _ in ()).throw(FileNotFoundError("missing")),
    )

    assert service._handle_grpc_error("grpc down", profile="paper", silent=False) is True
    assert service.activeDecisionLogPath == "offline-demo"
    assert service.feedTransportSnapshot["mode"] == "demo"


def test_runtime_service_feed_snapshot_uses_file_mode_for_jsonl_source(tmp_path: Path) -> None:
    service = RuntimeService(decision_loader=lambda limit: [])
    service._active_stream_label = None
    service._active_log_path = tmp_path / "journal.jsonl"

    service._update_feed_health(status="connected", reconnects=2, last_error="")

    snapshot = service.feedTransportSnapshot
    assert snapshot["mode"] == "file"
    assert snapshot["adapter"] == "jsonl"
    assert snapshot["status"] == "connected"


def test_runtime_service_feed_alerts_deduplicate_same_severity_and_recovery() -> None:
    events: list[dict[str, Any]] = []

    class _Sink:
        def emit_feed_health_event(self, **payload: object) -> None:
            events.append(dict(payload))

    service = RuntimeService(decision_loader=lambda limit: [], feed_alert_sink=_Sink())

    kwargs = {
        "metric_label": "Latencja p95 decision feedu",
        "unit": "ms",
        "value": 3000.0,
        "warning": 2500.0,
        "critical": 5000.0,
        "status": "connected",
        "adapter": "grpc",
        "reconnects": 0,
        "downtime_seconds": 0.0,
        "latency_p95": 3000.0,
        "last_error": "",
    }
    service._maybe_emit_feed_alert("latency", "warning", **kwargs)
    service._maybe_emit_feed_alert("latency", "warning", **kwargs)
    service._maybe_emit_feed_alert("latency", "ok", **kwargs)
    service._maybe_emit_feed_alert("latency", "ok", **kwargs)

    assert [entry["severity"] for entry in events] == ["warning", "info"]
    history = service.feedAlertHistory
    assert len(history) == 2
    degraded = [
        entry for entry in history if entry.get("metric") == "latency" and entry.get("state") == "degraded"
    ]
    recovered = [
        entry for entry in history if entry.get("metric") == "latency" and entry.get("state") == "recovered"
    ]
    assert len(degraded) == 1
    assert len(recovered) == 1


def test_runtime_service_unwraps_nested_operator_action_payload() -> None:
    service = RuntimeService(decision_loader=lambda limit: [])

    payload = {
        "record": {
            "id": "decision-001",
            "event": "risk_blocked",
            "timestamp": "2025-01-02T09:15:00+00:00",
        }
    }

    assert service.triggerOperatorAction("requestUnblock", payload) is True
    assert service.lastOperatorAction["action"] == "unblock"
    assert service.lastOperatorAction["entry"]["id"] == "decision-001"


def test_runtime_service_parses_decision_alias_payloads_from_loader() -> None:
    service = RuntimeService(
        decision_loader=lambda limit: [
            {
                "event": "decision_made",
                "Decision": {"state": "trade", "signal": "long", "should_trade": "yes"},
                "decision_confidence": "0.77",
                "decision_latency_ms": "11.5",
                "signals": "momentum,breakout",
                "metadata": {"source": "jsonl"},
            }
        ]
    )

    payload = service.loadRecentDecisions(1)[0]
    assert payload["decision"]["state"] == "trade"
    assert payload["decision"]["signal"] == "long"
    assert payload["decision"]["shouldTrade"] is True
    assert payload["decision"]["confidence"] == 0.77
    assert payload["decision"]["latencyMs"] == 11.5
    assert payload["signals"] == ["momentum", "breakout"]


def test_runtime_service_parses_lowercase_decision_and_flattens_metadata() -> None:
    service = RuntimeService(
        decision_loader=lambda limit: [
            {
                "event": "decision_made",
                "decision": {"state": "trade", "signal": "long", "should_trade": "yes"},
                "signals": ["momentum", "", None, "breakout"],
                "metadata": {"source": "jsonl", "profile": "paper"},
            }
        ]
    )

    payload = service.loadRecentDecisions(1)[0]
    assert payload["decision"]["state"] == "trade"
    assert payload["decision"]["signal"] == "long"
    assert payload["decision"]["shouldTrade"] is True
    assert payload["signals"] == ["momentum", "breakout"]
    assert payload["metadata"]["source"] == "jsonl"
    assert payload["metadata"]["profile"] == "paper"
    assert "metadata" not in payload["metadata"]


def test_runtime_service_degrades_to_demo_when_grpc_dependency_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(RuntimeService, "_auto_connect_grpc", lambda self: None)
    monkeypatch.setenv("BOT_CORE_UI_GRPC_ENDPOINT", "localhost:50051")
    monkeypatch.setattr(runtime_service_module, "grpc", None)

    service = RuntimeService(decision_loader=lambda limit: [])
    monkeypatch.setattr(
        service,
        "_build_live_loader",
        lambda profile: (_ for _ in ()).throw(FileNotFoundError("missing")),
    )

    assert service.attachToLiveDecisionLog("paper") is True
    assert service.activeDecisionLogPath == "offline-demo"
    assert service.feedHealth["status"] == "fallback"
    assert service.feedTransportSnapshot["mode"] == "demo"
    assert service.errorMessage

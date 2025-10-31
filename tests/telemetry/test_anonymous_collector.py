from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from core.telemetry import AnonymousTelemetryCollector


@pytest.fixture()
def collector(tmp_path: Path) -> AnonymousTelemetryCollector:
    return AnonymousTelemetryCollector(storage_dir=tmp_path)


def test_collector_disabled_by_default(collector: AnonymousTelemetryCollector) -> None:
    assert not collector.enabled
    assert collector.queued_events() == 0
    assert collector.export_events() is None


def test_enable_and_collect_events(collector: AnonymousTelemetryCollector) -> None:
    collector.set_opt_in(True, fingerprint="HWID-123")
    assert collector.enabled
    assert collector.pseudonym is not None

    timestamp = datetime(2025, 1, 1, tzinfo=timezone.utc)
    ok = collector.collect_event(
        "runtime.start",
        {"build": "1.0.0", "extra": {"ignored": "value"}},
        timestamp=timestamp,
    )
    assert ok is True
    assert collector.queued_events() == 1

    preview = collector.preview_events()
    assert preview
    assert preview[0]["event_type"] == "runtime.start"
    assert preview[0]["payload"]["build"] == "1.0.0"
    assert preview[0]["created_at"].startswith("2025-01-01")


def test_collect_requires_opt_in(collector: AnonymousTelemetryCollector) -> None:
    ok = collector.collect_event("runtime.stop", {"code": 0})
    assert ok is False
    assert collector.queued_events() == 0


def test_export_clears_queue(collector: AnonymousTelemetryCollector, tmp_path: Path) -> None:
    collector.set_opt_in(True, fingerprint="fingerprint")
    collector.collect_event("event.one", {"value": 1})
    collector.collect_event("event.two", {"value": 2})

    export_path = collector.export_events(destination=tmp_path)
    assert export_path is not None
    payload = json.loads(export_path.read_text(encoding="utf-8"))
    assert payload["events"]
    assert collector.queued_events() == 0


def test_clear_queue_removes_file(collector: AnonymousTelemetryCollector) -> None:
    collector.set_opt_in(True, fingerprint="hw")
    collector.collect_event("event.three", {"value": 3})
    assert collector.queued_events() == 1
    collector.clear_queue()
    assert collector.queued_events() == 0



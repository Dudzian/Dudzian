"""Testy guardrails kolejki I/O bazujące na asynchronicznym dispatcherze."""
from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from bot_core.observability.metrics import MetricsRegistry
from bot_core.runtime.scheduler import AsyncIOTaskQueue
from core.monitoring import (
    AsyncIOGuardrails,
    AsyncIOMetricSet,
    ComplianceMetricSet,
)
from core.monitoring.events import ComplianceViolation


def test_guardrails_records_rate_limit_wait(tmp_path: Path) -> None:
    registry = MetricsRegistry()
    metrics = AsyncIOMetricSet(registry=registry)
    ui_events: list[tuple[str, dict[str, object]]] = []
    guardrails = AsyncIOGuardrails(
        environment="paper",
        metrics=metrics,
        log_directory=tmp_path,
        rate_limit_warning_threshold=0.0,
        ui_notifier=lambda event, payload: ui_events.append((event, dict(payload))),
    )
    queue = AsyncIOTaskQueue(
        default_max_concurrency=1,
        default_burst=1,
        event_listener=guardrails,
    )

    async def _slow_task() -> None:
        await asyncio.sleep(0.01)

    async def _run_scenario() -> None:
        first = asyncio.create_task(queue.submit("binance", _slow_task))
        await asyncio.sleep(0.001)
        second = asyncio.create_task(queue.submit("binance", _slow_task))
        await asyncio.gather(first, second)

    asyncio.run(_run_scenario())

    labels = {"queue": "binance", "environment": "paper"}
    assert metrics.rate_limit_wait_total.value(labels=labels) == 1.0
    snapshot = metrics.rate_limit_wait_seconds.snapshot(labels=labels)
    assert snapshot.count == 1
    assert snapshot.sum >= 0.0

    assert ui_events, "Oczekiwano co najmniej jednego zdarzenia UI"
    event_name, payload = ui_events[0]
    assert event_name == "io_rate_limit_wait"
    assert payload["queue"] == "binance"

    log_file = tmp_path / "events.log"
    assert log_file.exists()
    assert "RATE_LIMIT" in log_file.read_text(encoding="utf-8")


def test_guardrails_records_timeout(tmp_path: Path) -> None:
    registry = MetricsRegistry()
    metrics = AsyncIOMetricSet(registry=registry)
    ui_events: list[tuple[str, dict[str, object]]] = []
    guardrails = AsyncIOGuardrails(
        environment="paper",
        metrics=metrics,
        log_directory=tmp_path,
        timeout_warning_threshold=0.0,
        ui_notifier=lambda event, payload: ui_events.append((event, dict(payload))),
    )
    queue = AsyncIOTaskQueue(
        default_max_concurrency=1,
        default_burst=1,
        event_listener=guardrails,
    )

    async def _timeout_task() -> None:
        await asyncio.sleep(0.001)
        raise asyncio.TimeoutError

    async def _run_scenario() -> None:
        with pytest.raises(asyncio.TimeoutError):
            await queue.submit("kraken", _timeout_task)

    asyncio.run(_run_scenario())

    labels = {"queue": "kraken", "environment": "paper"}
    assert metrics.timeout_total.value(labels=labels) == 1.0
    snapshot = metrics.timeout_duration.snapshot(labels=labels)
    assert snapshot.count == 1
    assert snapshot.sum >= 0.0

    assert ui_events, "Oczekiwano zdarzenia timeout dla UI"
    event_name, payload = ui_events[0]
    assert event_name == "io_timeout"
    assert payload["queue"] == "kraken"

    log_file = tmp_path / "events.log"
    assert log_file.exists()
    assert "TIMEOUT" in log_file.read_text(encoding="utf-8")


def test_guardrails_handles_compliance_violation(tmp_path: Path) -> None:
    ui_events: list[tuple[str, dict[str, object]]] = []
    registry = MetricsRegistry()
    compliance_metrics = ComplianceMetricSet(registry=registry)
    guardrails = AsyncIOGuardrails(
        environment="paper",
        metrics=AsyncIOMetricSet(registry=registry),
        log_directory=tmp_path,
        compliance_metrics=compliance_metrics,
        ui_notifier=lambda event, payload: ui_events.append((event, dict(payload))),
    )
    event = ComplianceViolation(
        rule_id="AML_BLOCKED_COUNTRY",
        severity="critical",
        message="Zablokowany kraj",
        metadata={"country": "IR"},
    )

    guardrails.handle_monitoring_event(event)

    labels = {
        "environment": "paper",
        "rule": "AML_BLOCKED_COUNTRY",
        "severity": "critical",
    }
    assert compliance_metrics.violations_total.value(labels=labels) == 1.0

    log_file = tmp_path / "events.log"
    assert log_file.exists()
    contents = log_file.read_text(encoding="utf-8")
    assert "COMPLIANCE" in contents
    assert "rule=AML_BLOCKED_COUNTRY" in contents

    assert ui_events
    event_name, payload = ui_events[0]
    assert event_name == "compliance_violation"
    assert payload["rule"] == "AML_BLOCKED_COUNTRY"

from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Mapping

import pytest

from bot_core.api.server import _HealthServicer
from bot_core.cloud.orchestrator import CloudOrchestrator
from bot_core.observability.metrics import MetricsRegistry
from bot_core.generated import trading_pb2


class _StubScheduler:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail

    def maybe_run(self) -> None:
        if self.fail:
            raise RuntimeError("scheduler failed")


class _StubContext:
    def __init__(self, *, registry: MetricsRegistry, fail_marketplace: bool = False) -> None:
        self.retrain_scheduler = _StubScheduler()
        self.marketplace_repository = object()
        self.metrics_registry = registry
        self._fail_marketplace = fail_marketplace

    def reload_marketplace_presets(self) -> None:
        if self._fail_marketplace:
            raise RuntimeError("marketplace refresh failed")


def _build_health_servicer(orchestrator: CloudOrchestrator) -> _HealthServicer:
    started_at = datetime(2024, 1, 1, tzinfo=timezone.utc)
    context = SimpleNamespace(
        version="1.0.0",
        git_commit="abc123",
        started_at=started_at,
        cloud_orchestrator=orchestrator,
    )
    return _HealthServicer(context)  # type: ignore[arg-type]


def _collect_metric_value(registry: MetricsRegistry, name: str, labels: Mapping[str, str]) -> float:
    metric = registry.get(name)
    return metric.value(labels=labels)  # type: ignore[call-arg]


def test_cloud_orchestrator_updates_metrics_and_health_snapshot() -> None:
    registry = MetricsRegistry()
    context = _StubContext(registry=registry)
    orchestrator = CloudOrchestrator(
        context,
        marketplace_refresh_interval=10,
        retrain_poll_interval=5,
        metrics_registry=registry,
    )
    orchestrator._set_health(status="running")  # initialize state

    orchestrator._run_scheduler_once(context.retrain_scheduler)
    orchestrator._refresh_marketplace_once()

    snapshot = orchestrator.health_snapshot()
    assert snapshot["status"] == "running"
    retrain = snapshot["workers"]["retrain"]
    marketplace = snapshot["workers"]["marketplace"]
    assert retrain["lastError"] is None
    assert marketplace["lastError"] is None

    assert _collect_metric_value(registry, "bot_cloud_worker_status", {"worker": "retrain"}) == 1.0
    assert _collect_metric_value(registry, "bot_cloud_worker_status", {"worker": "marketplace"}) == 1.0
    assert _collect_metric_value(
        registry,
        "bot_cloud_worker_last_error",
        {"worker": "retrain", "error": ""},
    ) == 0.0

    context.retrain_scheduler = _StubScheduler(fail=True)
    context._fail_marketplace = True
    orchestrator._run_scheduler_once(context.retrain_scheduler)
    orchestrator._refresh_marketplace_once()

    snapshot = orchestrator.health_snapshot()
    assert snapshot["workers"]["retrain"]["lastError"] == "scheduler_failure"
    assert snapshot["workers"]["marketplace"]["lastError"] == "refresh_failed"

    assert _collect_metric_value(
        registry,
        "bot_cloud_worker_status",
        {"worker": "retrain"},
    ) == 0.0
    assert _collect_metric_value(
        registry,
        "bot_cloud_worker_last_error",
        {"worker": "retrain", "error": "scheduler_failure"},
    ) == 1.0
    assert _collect_metric_value(
        registry,
        "bot_cloud_worker_last_error",
        {"worker": "marketplace", "error": "refresh_failed"},
    ) == 1.0


def test_health_servicer_exposes_cloud_health_snapshot() -> None:
    registry = MetricsRegistry()
    context = _StubContext(registry=registry)
    orchestrator = CloudOrchestrator(context, metrics_registry=registry)
    orchestrator._set_health(
        status="running",
        workers={
            "retrain": {
                "enabled": True,
                "lastRunAt": "2024-01-01T00:00:00Z",
                "interval": 30,
            }
        },
    )

    servicer = _build_health_servicer(orchestrator)
    response: trading_pb2.HealthCheckResponse = servicer.Check(None, None)

    assert response.cloud_health.status == "running"
    assert response.cloud_health.workers[0].name == "retrain"
    assert response.cloud_health.workers[0].last_run_at == "2024-01-01T00:00:00Z"
    assert response.cloud_health.workers[0].interval_seconds == 30
    assert response.started_at.seconds > 0

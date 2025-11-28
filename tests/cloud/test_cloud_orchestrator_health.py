from __future__ import annotations

from types import SimpleNamespace

from bot_core.api import server as api_server
from datetime import datetime, timezone

import grpc

from bot_core.cloud.orchestrator import CloudOrchestrator
from bot_core.cloud.service import _build_cloud_health_headers
from bot_core.generated import trading_pb2
from bot_core.observability.metrics import MetricsRegistry


class _FailingScheduler:
    def maybe_run(self) -> None:
        raise RuntimeError("scheduler boom")


class _FailingContext:
    def __init__(self, registry: MetricsRegistry) -> None:
        self.metrics_registry = registry
        self.retrain_scheduler = _FailingScheduler()
        self.marketplace_repository = object()
        self.auto_trader = SimpleNamespace(export_decision_journal=lambda limit=None: [])
        self.cloud_health_headers: dict[str, str] = {}

    def reload_marketplace_presets(self) -> None:
        raise RuntimeError("refresh boom")

    def authorize(self, _context) -> None:
        return None


class _FakeRpcContext:
    def __init__(self) -> None:
        self.initial_metadata: list[list[tuple[str, str]]] = []
        self.trailing_metadata: list[list[tuple[str, str]]] = []
        self.code = None
        self.details = None

    def send_initial_metadata(self, metadata):
        self.initial_metadata.append(list(metadata))

    def set_trailing_metadata(self, metadata):
        self.trailing_metadata.append(list(metadata))

    def set_code(self, code):
        self.code = code

    def set_details(self, details):
        self.details = details


class _HealthContext(_FailingContext):
    def __init__(self, registry: MetricsRegistry) -> None:
        super().__init__(registry)
        self.version = "1.2.3"
        self.git_commit = "deadbeef"
        self.started_at = datetime.now(timezone.utc)


def test_orchestrator_reports_health_and_metrics() -> None:
    registry = MetricsRegistry()
    events: list[dict[str, object]] = []
    context = _FailingContext(registry)
    orchestrator = CloudOrchestrator(
        context,
        retrain_poll_interval=1,
        marketplace_refresh_interval=1,
        metrics_registry=registry,
        health_hook=lambda snapshot: events.append(dict(snapshot)),
    )

    orchestrator._run_scheduler_once(context.retrain_scheduler)
    orchestrator._refresh_marketplace_once()

    snapshot = orchestrator.health_snapshot()
    assert snapshot.get("_health") is False
    assert snapshot.get("_lastError")
    assert events, "health hook should be invoked"

    health_metric = registry.get("bot_cloud_health_status")
    assert health_metric.value() == 0.0
    last_error_metric = registry.get("bot_cloud_last_error")
    assert last_error_metric.value(labels={"error": str(snapshot.get("_lastError") or "")}) == 1.0


def test_runtime_servicer_propagates_cloud_health_metadata() -> None:
    registry = MetricsRegistry()
    context = _FailingContext(registry)
    orchestrator = CloudOrchestrator(
        context,
        retrain_poll_interval=1,
        marketplace_refresh_interval=1,
        metrics_registry=registry,
        health_hook=lambda snapshot: setattr(
            context, "cloud_health_headers", _build_cloud_health_headers(snapshot)
        ),
    )

    orchestrator._run_scheduler_once(context.retrain_scheduler)
    rpc_context = _FakeRpcContext()
    servicer = api_server._RuntimeServicer(context)

    response = servicer.ListDecisions(trading_pb2.ListDecisionsRequest(), rpc_context)
    assert response.cursor == 0

    propagated = rpc_context.initial_metadata or rpc_context.trailing_metadata
    flattened = {key: value for key, value in propagated[0]} if propagated else {}
    assert flattened.get("x-bot-cloud-health") == "0"
    assert flattened.get("x-bot-cloud-last-error")


def test_health_servicer_sets_grpc_status_on_unhealthy() -> None:
    registry = MetricsRegistry()
    context = _HealthContext(registry)
    orchestrator = CloudOrchestrator(
        context,
        retrain_poll_interval=1,
        marketplace_refresh_interval=1,
        metrics_registry=registry,
        health_hook=lambda snapshot: setattr(
            context, "cloud_health_headers", _build_cloud_health_headers(snapshot)
        ),
    )
    orchestrator._run_scheduler_once(context.retrain_scheduler)
    context.cloud_orchestrator = orchestrator

    servicer = api_server._HealthServicer(context)
    rpc_context = _FakeRpcContext()
    response = servicer.Check(trading_pb2.HealthCheckRequest(), rpc_context)

    assert rpc_context.code == grpc.StatusCode.UNAVAILABLE
    assert "scheduler_failure" in (rpc_context.details or "")
    assert response.cloud_health.status
    assert rpc_context.initial_metadata
    assert rpc_context.trailing_metadata

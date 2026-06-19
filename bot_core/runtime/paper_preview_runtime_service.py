"""Single-shot local runtime service wrapper for paper preview contracts.

This module composes existing in-memory paper preview evidence into one immutable
service result. It is not an app runtime loop, UI binding, controller handoff,
decision engine, order generator, serialization/export writer, market adapter,
account reader, secret reader, thread/timer/worker, or cloud/network sink.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from bot_core.runtime.paper_preview_bundle_boundary import (
    PaperPreviewBundleBoundaryMatrixReport,
    build_local_bundle_boundary_matrix,
)
from bot_core.runtime.paper_preview_bundle_read_model import (
    PaperPreviewBundleReadModel,
    PaperPreviewReadModelBoundaryMatrixReport,
    build_paper_preview_bundle_read_model,
    build_paper_preview_read_model_boundary_matrix,
)
from bot_core.runtime.paper_preview_integration_gate import (
    PaperPreviewIntegrationReadinessGate,
    build_paper_preview_integration_readiness_gate,
)
from bot_core.runtime.paper_preview_scenario import (
    PaperPreviewScenario,
    PaperPreviewScenarioRunner,
    PaperPreviewScenarioResult,
)
from bot_core.runtime.paper_preview_ui_runtime_preflight import (
    PaperPreviewUiRuntimePreflightReport,
    build_paper_preview_ui_runtime_preflight,
)
from bot_core.runtime.read_only_market_data import ReadOnlyMarketDataProvider


class PaperPreviewRuntimeServiceError(ValueError):
    """Raised when the local runtime service wrapper must fail closed."""


@dataclass(frozen=True, slots=True)
class PaperPreviewRuntimeServiceSnapshot:
    """Immutable local/static single-shot service snapshot.

    The snapshot mirrors already-built local paper preview evidence only. It is
    deliberately read-only, paper-only, unbound from UI, not runtime-backed, and
    fail-closed for generated decisions/orders and export/cloud markers.
    """

    scenario_name: str
    scenario_result: PaperPreviewScenarioResult
    integration_gate_status: str
    local_bundle_present: bool
    bundle_boundary_matrix_present: bool
    read_model_present: bool
    read_model_boundary_matrix_present: bool
    preflight_present: bool
    integration_gate_present: bool
    order_event_count: int
    trade_count: int
    audit_event_count: int
    position_count: int
    market_symbols: tuple[str, ...]
    has_market_context: bool
    blocking_items: tuple[str, ...]
    blocking_check_count: int
    read_model_kind: str
    gate_kind: str
    preflight_report_kind: str
    service_kind: str = "local_paper_preview_runtime_service"
    mode: str = "paper"
    single_shot: bool = True
    runtime_loop_started: bool = False
    ui_bound: bool = False
    runtime_backed: bool = False
    read_only: bool = True
    paper_only: bool = True
    ready_for_ui_runtime_integration: bool = False
    ready_for_decision_engine: bool = False
    ready_for_export: bool = False
    generated_order_count: int = 0
    generated_decision_count: int = 0
    export_sink: str = "none"
    cloud_sink: str = "none"
    external_export: bool = False


PaperPreviewRuntimeServiceResult = PaperPreviewRuntimeServiceSnapshot


@dataclass(frozen=True, slots=True)
class PaperPreviewRuntimeService:
    """Local in-memory single-shot wrapper around ``PaperPreviewScenarioRunner``."""

    created_at: str | None = None
    market_data_provider: ReadOnlyMarketDataProvider | None = None
    scenario_runner_factory: Callable[..., PaperPreviewScenarioRunner] = PaperPreviewScenarioRunner
    bundle_matrix_builder: Callable[[object], PaperPreviewBundleBoundaryMatrixReport] = (
        build_local_bundle_boundary_matrix
    )
    read_model_builder: Callable[
        [object, PaperPreviewBundleBoundaryMatrixReport], PaperPreviewBundleReadModel
    ] = build_paper_preview_bundle_read_model
    read_model_matrix_builder: Callable[
        [PaperPreviewBundleReadModel], PaperPreviewReadModelBoundaryMatrixReport
    ] = build_paper_preview_read_model_boundary_matrix
    preflight_builder: Callable[
        [PaperPreviewBundleReadModel, PaperPreviewReadModelBoundaryMatrixReport],
        PaperPreviewUiRuntimePreflightReport,
    ] = build_paper_preview_ui_runtime_preflight
    gate_builder: Callable[
        [PaperPreviewUiRuntimePreflightReport], PaperPreviewIntegrationReadinessGate
    ] = build_paper_preview_integration_readiness_gate

    def run_once(self, scenario: PaperPreviewScenario) -> PaperPreviewRuntimeServiceResult:
        """Run one deterministic local composition call and return an immutable result."""

        runner = self.scenario_runner_factory(
            created_at=self.created_at,
            market_data_provider=self.market_data_provider,
        )
        scenario_result = runner.run(scenario)
        local_bundle = scenario_result.local_bundle
        if local_bundle is None:
            raise PaperPreviewRuntimeServiceError("local bundle must be present")

        bundle_matrix = self.bundle_matrix_builder(local_bundle)
        read_model = self.read_model_builder(local_bundle, bundle_matrix)
        read_model_matrix = self.read_model_matrix_builder(read_model)
        preflight = self.preflight_builder(read_model, read_model_matrix)
        gate = self.gate_builder(preflight)

        snapshot = PaperPreviewRuntimeServiceSnapshot(
            scenario_name=scenario_result.scenario_name,
            scenario_result=scenario_result,
            integration_gate_status=gate.status,
            local_bundle_present=local_bundle is not None,
            bundle_boundary_matrix_present=bundle_matrix is not None,
            read_model_present=read_model is not None,
            read_model_boundary_matrix_present=read_model_matrix is not None,
            preflight_present=preflight is not None,
            integration_gate_present=gate is not None,
            order_event_count=scenario_result.summary.order_event_count,
            trade_count=scenario_result.summary.trade_count,
            audit_event_count=scenario_result.summary.audit_event_count,
            position_count=read_model.position_count,
            market_symbols=tuple(sorted(read_model.market_symbols)),
            has_market_context=read_model.has_market_context,
            blocking_items=tuple(gate.blocking_items),
            blocking_check_count=gate.blocking_check_count,
            read_model_kind=read_model.model_kind,
            gate_kind=gate.gate_kind,
            preflight_report_kind=preflight.report_kind,
            ui_bound=gate.ui_bound,
            runtime_backed=gate.runtime_backed,
            read_only=gate.read_only,
            ready_for_ui_runtime_integration=gate.ready_for_ui_runtime_integration,
            ready_for_decision_engine=gate.ready_for_decision_engine,
            ready_for_export=gate.ready_for_export,
            generated_order_count=gate.generated_order_count,
            generated_decision_count=gate.generated_decision_count,
            export_sink=gate.export_sink,
            cloud_sink=gate.cloud_sink,
            external_export=gate.external_export,
        )
        _validate_snapshot(snapshot)
        return snapshot


def run_paper_preview_runtime_service_once(
    scenario: PaperPreviewScenario,
    *,
    created_at: str | None = None,
    market_data_provider: ReadOnlyMarketDataProvider | None = None,
) -> PaperPreviewRuntimeServiceResult:
    """Convenience helper for one local/static paper preview service run."""

    return PaperPreviewRuntimeService(
        created_at=created_at,
        market_data_provider=market_data_provider,
    ).run_once(scenario)


def _validate_snapshot(snapshot: PaperPreviewRuntimeServiceSnapshot) -> None:
    if snapshot.local_bundle_present is not True:
        raise PaperPreviewRuntimeServiceError("local bundle must be present")
    if snapshot.integration_gate_present is not True:
        raise PaperPreviewRuntimeServiceError("integration gate must be present")
    if snapshot.integration_gate_status != "blocked":
        raise PaperPreviewRuntimeServiceError("integration gate must remain blocked")
    if snapshot.ready_for_ui_runtime_integration is True:
        raise PaperPreviewRuntimeServiceError("UI/runtime integration readiness must fail closed")
    if snapshot.ready_for_decision_engine is True:
        raise PaperPreviewRuntimeServiceError("decision-engine readiness must fail closed")
    if snapshot.ready_for_export is True:
        raise PaperPreviewRuntimeServiceError("export readiness must fail closed")
    if snapshot.generated_order_count != 0:
        raise PaperPreviewRuntimeServiceError("generated_order_count must be zero")
    if snapshot.generated_decision_count != 0:
        raise PaperPreviewRuntimeServiceError("generated_decision_count must be zero")
    if snapshot.export_sink != "none":
        raise PaperPreviewRuntimeServiceError("export_sink must be none")
    if snapshot.cloud_sink != "none":
        raise PaperPreviewRuntimeServiceError("cloud_sink must be none")
    if snapshot.external_export is True:
        raise PaperPreviewRuntimeServiceError("external_export must be false")
    if snapshot.runtime_loop_started is True or snapshot.runtime_backed is True:
        raise PaperPreviewRuntimeServiceError("service must not start or back an app runtime loop")
    if snapshot.ui_bound is True:
        raise PaperPreviewRuntimeServiceError("service must not bind UI")
    if (
        snapshot.single_shot is not True
        or snapshot.paper_only is not True
        or snapshot.read_only is not True
    ):
        raise PaperPreviewRuntimeServiceError(
            "service safety flags must remain single-shot/paper/read-only"
        )


__all__ = [
    "PaperPreviewRuntimeService",
    "PaperPreviewRuntimeServiceError",
    "PaperPreviewRuntimeServiceResult",
    "PaperPreviewRuntimeServiceSnapshot",
    "run_paper_preview_runtime_service_once",
]

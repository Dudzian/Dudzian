"""Local no-loop boundary matrix for the paper preview runtime service.

The matrix is static diagnostic evidence built from an existing service snapshot.
It does not start loops, bind UI, create background work, serialize/export data,
or hand off to controller/decision/live boundaries.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from bot_core.runtime.paper_preview_runtime_service import PaperPreviewRuntimeServiceSnapshot


class PaperPreviewRuntimeServiceBoundaryError(ValueError):
    """Raised when runtime service boundary evidence must fail closed."""


PaperPreviewRuntimeServiceBoundary = str

PAPER_PREVIEW_RUNTIME_SERVICE_BOUNDARIES: Final[tuple[PaperPreviewRuntimeServiceBoundary, ...]] = (
    "app_runtime_loop",
    "scheduler_loop",
    "worker_loop",
    "background_thread",
    "background_timer",
    "async_task",
    "qml_binding",
    "pyside_bridge",
    "ui_runtime_binding",
    "controller_handoff",
    "trading_controller_handoff",
    "decision_envelope_handoff",
    "strategy_engine_handoff",
    "ai_model_inference_handoff",
    "scoring_handoff",
    "recommendation_handoff",
    "order_generation_handoff",
    "real_market_adapter",
    "testnet_sandbox_adapter",
    "live_exchange_io",
    "account_balance_fetch",
    "live_credentials_read",
    "file_export",
    "serialized_export",
    "cloud_sink",
    "external_export",
)

_BOUNDARY_REASONS: Final[dict[PaperPreviewRuntimeServiceBoundary, str]] = {
    boundary: "refused_by_local_static_single_shot_paper_preview_runtime_service_boundary"
    for boundary in PAPER_PREVIEW_RUNTIME_SERVICE_BOUNDARIES
}


@dataclass(frozen=True, slots=True)
class PaperPreviewRuntimeServiceBoundaryRefusal:
    """Immutable refusal fact for one service boundary."""

    boundary_kind: str
    refused: bool = True
    reason: str = "refused_by_local_static_single_shot_paper_preview_runtime_service_boundary"


@dataclass(frozen=True, slots=True)
class PaperPreviewRuntimeServiceBoundaryMatrixRow:
    """One immutable no-loop matrix row mirroring the service snapshot safety flags."""

    boundary_kind: str
    reason: str
    service_kind: str
    scenario_name: str
    integration_gate_status: str
    refused: bool = True
    single_shot: bool = True
    runtime_loop_started: bool = False
    runtime_backed: bool = False
    ui_bound: bool = False
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


@dataclass(frozen=True, slots=True)
class PaperPreviewRuntimeServiceBoundaryMatrixReport:
    """Immutable local/static runtime service boundary matrix report."""

    service_kind: str
    scenario_name: str
    row_count: int
    rows: tuple[PaperPreviewRuntimeServiceBoundaryMatrixRow, ...]
    integration_gate_status: str
    report_kind: str = "local_runtime_service_boundary_no_loop_matrix"
    all_refused: bool = True
    single_shot: bool = True
    runtime_loop_started: bool = False
    runtime_backed: bool = False
    ui_bound: bool = False
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


def build_paper_preview_runtime_service_boundary_matrix(
    snapshot: PaperPreviewRuntimeServiceSnapshot,
) -> PaperPreviewRuntimeServiceBoundaryMatrixReport:
    """Build a deterministic no-loop matrix from an already-created service snapshot."""

    _validate_snapshot_for_boundary_matrix(snapshot)
    rows = tuple(
        PaperPreviewRuntimeServiceBoundaryMatrixRow(
            boundary_kind=boundary,
            reason=_BOUNDARY_REASONS[boundary],
            service_kind=snapshot.service_kind,
            scenario_name=snapshot.scenario_name,
            integration_gate_status=snapshot.integration_gate_status,
            single_shot=snapshot.single_shot,
            runtime_loop_started=snapshot.runtime_loop_started,
            runtime_backed=snapshot.runtime_backed,
            ui_bound=snapshot.ui_bound,
            read_only=snapshot.read_only,
            paper_only=snapshot.paper_only,
            ready_for_ui_runtime_integration=snapshot.ready_for_ui_runtime_integration,
            ready_for_decision_engine=snapshot.ready_for_decision_engine,
            ready_for_export=snapshot.ready_for_export,
            generated_order_count=snapshot.generated_order_count,
            generated_decision_count=snapshot.generated_decision_count,
            export_sink=snapshot.export_sink,
            cloud_sink=snapshot.cloud_sink,
            external_export=snapshot.external_export,
        )
        for boundary in PAPER_PREVIEW_RUNTIME_SERVICE_BOUNDARIES
    )
    return PaperPreviewRuntimeServiceBoundaryMatrixReport(
        service_kind=snapshot.service_kind,
        scenario_name=snapshot.scenario_name,
        row_count=len(rows),
        rows=rows,
        integration_gate_status=snapshot.integration_gate_status,
        all_refused=all(row.refused for row in rows),
        single_shot=snapshot.single_shot,
        runtime_loop_started=snapshot.runtime_loop_started,
        runtime_backed=snapshot.runtime_backed,
        ui_bound=snapshot.ui_bound,
        read_only=snapshot.read_only,
        paper_only=snapshot.paper_only,
        ready_for_ui_runtime_integration=snapshot.ready_for_ui_runtime_integration,
        ready_for_decision_engine=snapshot.ready_for_decision_engine,
        ready_for_export=snapshot.ready_for_export,
        generated_order_count=snapshot.generated_order_count,
        generated_decision_count=snapshot.generated_decision_count,
        export_sink=snapshot.export_sink,
        cloud_sink=snapshot.cloud_sink,
        external_export=snapshot.external_export,
    )


def _validate_snapshot_for_boundary_matrix(snapshot: PaperPreviewRuntimeServiceSnapshot) -> None:
    checks = (
        (snapshot.service_kind == "local_paper_preview_runtime_service", "service_kind"),
        (snapshot.single_shot is True, "single_shot"),
        (snapshot.runtime_loop_started is False, "runtime_loop_started"),
        (snapshot.runtime_backed is False, "runtime_backed"),
        (snapshot.ui_bound is False, "ui_bound"),
        (snapshot.read_only is True, "read_only"),
        (snapshot.paper_only is True, "paper_only"),
        (snapshot.integration_gate_status == "blocked", "integration_gate_status"),
        (
            snapshot.ready_for_ui_runtime_integration is False,
            "ready_for_ui_runtime_integration",
        ),
        (snapshot.ready_for_decision_engine is False, "ready_for_decision_engine"),
        (snapshot.ready_for_export is False, "ready_for_export"),
        (snapshot.generated_order_count == 0, "generated_order_count"),
        (snapshot.generated_decision_count == 0, "generated_decision_count"),
        (snapshot.export_sink == "none", "export_sink"),
        (snapshot.cloud_sink == "none", "cloud_sink"),
        (snapshot.external_export is False, "external_export"),
    )
    for passed, field_name in checks:
        if not passed:
            raise PaperPreviewRuntimeServiceBoundaryError(
                f"unsafe runtime service boundary snapshot field: {field_name}"
            )


__all__ = [
    "PAPER_PREVIEW_RUNTIME_SERVICE_BOUNDARIES",
    "PaperPreviewRuntimeServiceBoundary",
    "PaperPreviewRuntimeServiceBoundaryError",
    "PaperPreviewRuntimeServiceBoundaryMatrixReport",
    "PaperPreviewRuntimeServiceBoundaryMatrixRow",
    "PaperPreviewRuntimeServiceBoundaryRefusal",
    "build_paper_preview_runtime_service_boundary_matrix",
]

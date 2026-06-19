"""Local read API boundary/no-export matrix for paper preview runtime service.

This module builds immutable diagnostic evidence from an existing read API view.
It does not bind UI, execute lifecycle commands, dispatch commands, serialize or
export payloads, open files/sockets, create background work, or hand off to
controller, decision, live, account, secret, or adapter boundaries.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from bot_core.runtime.paper_preview_runtime_service_read_api import (
    PaperPreviewRuntimeServiceReadApiView,
)


class PaperPreviewRuntimeServiceReadApiBoundaryError(ValueError):
    """Raised when read API boundary evidence must fail closed."""


PaperPreviewRuntimeServiceReadApiBoundary = str

PAPER_PREVIEW_RUNTIME_SERVICE_READ_API_BOUNDARIES: Final[
    tuple[PaperPreviewRuntimeServiceReadApiBoundary, ...]
] = (
    "qml_binding",
    "pyside_bridge",
    "ui_runtime_binding",
    "ui_signal_emission",
    "app_runtime_loop",
    "lifecycle_command_execution",
    "command_dispatcher",
    "scheduler_loop",
    "worker_loop",
    "background_thread",
    "background_timer",
    "async_task",
    "controller_handoff",
    "trading_controller_handoff",
    "decision_envelope_handoff",
    "strategy_engine_handoff",
    "ai_model_inference_handoff",
    "scoring_handoff",
    "recommendation_handoff",
    "order_generation_handoff",
    "order_submission",
    "real_market_adapter",
    "testnet_sandbox_adapter",
    "live_exchange_io",
    "account_balance_fetch",
    "live_account_snapshot_read",
    "live_credentials_read",
    "json_serialization",
    "yaml_serialization",
    "csv_serialization",
    "file_export",
    "serialized_export",
    "cloud_sink",
    "external_export",
)

_BOUNDARY_REASONS: Final[dict[PaperPreviewRuntimeServiceReadApiBoundary, str]] = {
    boundary: "refused_by_local_static_read_api_boundary_no_export_matrix"
    for boundary in PAPER_PREVIEW_RUNTIME_SERVICE_READ_API_BOUNDARIES
}
_ALLOWED_COMMANDS: Final[tuple[str, ...]] = (
    "run_once_local_scenario",
    "read_local_snapshot",
    "inspect_integration_gate",
    "inspect_boundary_matrix",
)
_REQUIRED_REFUSED_COMMANDS: Final[frozenset[str]] = frozenset(
    {
        "start_runtime_loop",
        "stop_runtime_loop",
        "restart_runtime_loop",
        "schedule_worker",
        "start_worker",
        "start_background_thread",
        "start_background_timer",
        "start_async_task",
        "bind_qml",
        "bind_pyside",
        "attach_ui",
        "lifecycle_command_execution",
        "command_dispatcher",
        "controller_handoff",
        "trading_controller_handoff",
        "decision_envelope_handoff",
        "strategy_engine_handoff",
        "ai_model_inference_handoff",
        "scoring_handoff",
        "recommendation_handoff",
        "generate_order",
        "submit_order",
        "real_market_adapter_handoff",
        "testnet_sandbox_adapter_handoff",
        "live_exchange_io",
        "account_balance_fetch",
        "live_account_snapshot_read",
        "live_credentials_read",
        "json_serialization",
        "yaml_serialization",
        "csv_serialization",
        "file_export",
        "serialized_export",
        "cloud_sink",
        "external_export",
    }
)


@dataclass(frozen=True, slots=True)
class PaperPreviewRuntimeServiceReadApiBoundaryRefusal:
    """Immutable refusal fact for one read API boundary."""

    boundary_kind: str
    refused: bool = True
    reason: str = "refused_by_local_static_read_api_boundary_no_export_matrix"


@dataclass(frozen=True, slots=True)
class PaperPreviewRuntimeServiceReadApiBoundaryMatrixRow:
    """One immutable boundary row mirroring the read API view safety flags."""

    boundary_kind: str
    reason: str
    view_kind: str
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
class PaperPreviewRuntimeServiceReadApiBoundaryMatrixReport:
    """Immutable local/static read API boundary/no-export matrix report."""

    view_kind: str
    service_kind: str
    scenario_name: str
    row_count: int
    rows: tuple[PaperPreviewRuntimeServiceReadApiBoundaryMatrixRow, ...]
    integration_gate_status: str
    report_kind: str = "local_runtime_service_read_api_boundary_no_export_matrix"
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


def build_paper_preview_runtime_service_read_api_boundary_matrix(
    view: PaperPreviewRuntimeServiceReadApiView,
) -> PaperPreviewRuntimeServiceReadApiBoundaryMatrixReport:
    """Build a deterministic refusal matrix from an existing read API view only."""

    _validate_view(view)
    rows = tuple(
        PaperPreviewRuntimeServiceReadApiBoundaryMatrixRow(
            boundary_kind=boundary,
            reason=_BOUNDARY_REASONS[boundary],
            view_kind=view.view_kind,
            service_kind=view.service_kind,
            scenario_name=view.scenario_name,
            integration_gate_status=view.integration_gate_status,
            single_shot=view.single_shot,
            runtime_loop_started=view.runtime_loop_started,
            runtime_backed=view.runtime_backed,
            ui_bound=view.ui_bound,
            read_only=view.read_only,
            paper_only=view.paper_only,
            ready_for_ui_runtime_integration=view.ready_for_ui_runtime_integration,
            ready_for_decision_engine=view.ready_for_decision_engine,
            ready_for_export=view.ready_for_export,
            generated_order_count=view.generated_order_count,
            generated_decision_count=view.generated_decision_count,
            export_sink=view.export_sink,
            cloud_sink=view.cloud_sink,
            external_export=view.external_export,
        )
        for boundary in PAPER_PREVIEW_RUNTIME_SERVICE_READ_API_BOUNDARIES
    )
    if not rows:
        raise PaperPreviewRuntimeServiceReadApiBoundaryError("boundary_row_count")
    return PaperPreviewRuntimeServiceReadApiBoundaryMatrixReport(
        view_kind=view.view_kind,
        service_kind=view.service_kind,
        scenario_name=view.scenario_name,
        row_count=len(rows),
        rows=rows,
        integration_gate_status=view.integration_gate_status,
        all_refused=all(row.refused for row in rows),
        single_shot=view.single_shot,
        runtime_loop_started=view.runtime_loop_started,
        runtime_backed=view.runtime_backed,
        ui_bound=view.ui_bound,
        read_only=view.read_only,
        paper_only=view.paper_only,
        ready_for_ui_runtime_integration=view.ready_for_ui_runtime_integration,
        ready_for_decision_engine=view.ready_for_decision_engine,
        ready_for_export=view.ready_for_export,
        generated_order_count=view.generated_order_count,
        generated_decision_count=view.generated_decision_count,
        export_sink=view.export_sink,
        cloud_sink=view.cloud_sink,
        external_export=view.external_export,
    )


def _validate_view(view: PaperPreviewRuntimeServiceReadApiView) -> None:
    checks = (
        (view.view_kind == "local_runtime_service_snapshot_read_api", "view_kind"),
        (view.service_kind == "local_paper_preview_runtime_service", "service_kind"),
        (view.single_shot is True, "single_shot"),
        (view.runtime_loop_started is False, "runtime_loop_started"),
        (view.runtime_backed is False, "runtime_backed"),
        (view.ui_bound is False, "ui_bound"),
        (view.read_only is True, "read_only"),
        (view.paper_only is True, "paper_only"),
        (view.integration_gate_status == "blocked", "integration_gate_status"),
        (view.ready_for_ui_runtime_integration is False, "ready_for_ui_runtime_integration"),
        (view.ready_for_decision_engine is False, "ready_for_decision_engine"),
        (view.ready_for_export is False, "ready_for_export"),
        (view.generated_order_count == 0, "generated_order_count"),
        (view.generated_decision_count == 0, "generated_decision_count"),
        (view.export_sink == "none", "export_sink"),
        (view.cloud_sink == "none", "cloud_sink"),
        (view.external_export is False, "external_export"),
        (view.boundary_row_count > 0, "boundary_row_count"),
        (view.lifecycle_allowed_commands == _ALLOWED_COMMANDS, "lifecycle_allowed_commands"),
        (
            _REQUIRED_REFUSED_COMMANDS.issubset(frozenset(view.lifecycle_refused_commands)),
            "lifecycle_refused_commands",
        ),
        (
            view.lifecycle_allowed_command_count == len(view.lifecycle_allowed_commands),
            "lifecycle_allowed_command_count",
        ),
        (
            view.lifecycle_refused_command_count == len(view.lifecycle_refused_commands),
            "lifecycle_refused_command_count",
        ),
    )
    for passed, field_name in checks:
        if not passed:
            raise PaperPreviewRuntimeServiceReadApiBoundaryError(
                f"unsafe read API boundary matrix input: {field_name}"
            )


__all__ = [
    "PAPER_PREVIEW_RUNTIME_SERVICE_READ_API_BOUNDARIES",
    "PaperPreviewRuntimeServiceReadApiBoundary",
    "PaperPreviewRuntimeServiceReadApiBoundaryError",
    "PaperPreviewRuntimeServiceReadApiBoundaryMatrixReport",
    "PaperPreviewRuntimeServiceReadApiBoundaryMatrixRow",
    "PaperPreviewRuntimeServiceReadApiBoundaryRefusal",
    "build_paper_preview_runtime_service_read_api_boundary_matrix",
]

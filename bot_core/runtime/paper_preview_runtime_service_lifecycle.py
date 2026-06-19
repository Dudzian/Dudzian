"""Local lifecycle command contract for the paper preview runtime service.

The contract is declarative diagnostic evidence only. It classifies local/static
single-shot commands versus refused lifecycle, UI, controller, decision, live,
adapter, and export commands. It is not a lifecycle manager, command dispatcher,
runtime loop, UI bridge, serializer, exchange adapter, account reader, or cloud
sink.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from bot_core.runtime.paper_preview_runtime_service import PaperPreviewRuntimeServiceSnapshot
from bot_core.runtime.paper_preview_runtime_service_boundary import (
    PaperPreviewRuntimeServiceBoundaryMatrixReport,
)


class PaperPreviewRuntimeServiceLifecycleError(ValueError):
    """Raised when lifecycle command contract evidence must fail closed."""


PaperPreviewRuntimeServiceCommand = str

PAPER_PREVIEW_RUNTIME_SERVICE_ALLOWED_COMMANDS: Final[
    tuple[PaperPreviewRuntimeServiceCommand, ...]
] = (
    "run_once_local_scenario",
    "read_local_snapshot",
    "inspect_integration_gate",
    "inspect_boundary_matrix",
)

PAPER_PREVIEW_RUNTIME_SERVICE_REFUSED_COMMANDS: Final[
    tuple[PaperPreviewRuntimeServiceCommand, ...]
] = (
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
    "live_credentials_read",
    "file_export",
    "serialized_export",
    "cloud_sink",
    "external_export",
)

_ALLOWED_REASONS: Final[dict[PaperPreviewRuntimeServiceCommand, str]] = {
    "run_once_local_scenario": "allowed_local_static_single_shot_composition_only_no_loop",
    "read_local_snapshot": "allowed_local_static_single_shot_introspection_only_in_memory_snapshot_no_export",
    "inspect_integration_gate": "allowed_local_static_single_shot_introspection_only_blocked_gate",
    "inspect_boundary_matrix": "allowed_local_static_single_shot_introspection_only_no_loop_matrix",
}
_REFUSED_REASON: Final[str] = "refused_by_local_static_single_shot_lifecycle_command_contract"


@dataclass(frozen=True, slots=True)
class PaperPreviewRuntimeServiceCommandDecision:
    """One immutable lifecycle command decision row."""

    command: str
    allowed: bool
    refused: bool
    reason: str
    service_kind: str
    scenario_name: str
    integration_gate_status: str
    boundary_matrix_report_kind: str
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
class PaperPreviewRuntimeServiceLifecycleContract:
    """Immutable local/static lifecycle command contract report."""

    service_kind: str
    scenario_name: str
    command_count: int
    allowed_command_count: int
    refused_command_count: int
    allowed_commands: tuple[str, ...]
    refused_commands: tuple[str, ...]
    command_decisions: tuple[PaperPreviewRuntimeServiceCommandDecision, ...]
    integration_gate_status: str
    boundary_matrix_report_kind: str
    contract_kind: str = "local_runtime_service_lifecycle_command_contract"
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


def build_paper_preview_runtime_service_lifecycle_contract(
    snapshot: PaperPreviewRuntimeServiceSnapshot,
    boundary_matrix: PaperPreviewRuntimeServiceBoundaryMatrixReport,
) -> PaperPreviewRuntimeServiceLifecycleContract:
    """Build deterministic lifecycle command evidence from local snapshot and matrix."""

    _validate_inputs(snapshot, boundary_matrix)
    decisions = tuple(
        _decision(command, True, False, _ALLOWED_REASONS[command], snapshot, boundary_matrix)
        for command in PAPER_PREVIEW_RUNTIME_SERVICE_ALLOWED_COMMANDS
    ) + tuple(
        _decision(command, False, True, _REFUSED_REASON, snapshot, boundary_matrix)
        for command in PAPER_PREVIEW_RUNTIME_SERVICE_REFUSED_COMMANDS
    )
    return PaperPreviewRuntimeServiceLifecycleContract(
        service_kind=snapshot.service_kind,
        scenario_name=snapshot.scenario_name,
        command_count=len(decisions),
        allowed_command_count=len(PAPER_PREVIEW_RUNTIME_SERVICE_ALLOWED_COMMANDS),
        refused_command_count=len(PAPER_PREVIEW_RUNTIME_SERVICE_REFUSED_COMMANDS),
        allowed_commands=PAPER_PREVIEW_RUNTIME_SERVICE_ALLOWED_COMMANDS,
        refused_commands=PAPER_PREVIEW_RUNTIME_SERVICE_REFUSED_COMMANDS,
        command_decisions=decisions,
        integration_gate_status=snapshot.integration_gate_status,
        boundary_matrix_report_kind=boundary_matrix.report_kind,
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


def _decision(
    command: str,
    allowed: bool,
    refused: bool,
    reason: str,
    snapshot: PaperPreviewRuntimeServiceSnapshot,
    boundary_matrix: PaperPreviewRuntimeServiceBoundaryMatrixReport,
) -> PaperPreviewRuntimeServiceCommandDecision:
    return PaperPreviewRuntimeServiceCommandDecision(
        command=command,
        allowed=allowed,
        refused=refused,
        reason=reason,
        service_kind=snapshot.service_kind,
        scenario_name=snapshot.scenario_name,
        integration_gate_status=snapshot.integration_gate_status,
        boundary_matrix_report_kind=boundary_matrix.report_kind,
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


def _validate_inputs(
    snapshot: PaperPreviewRuntimeServiceSnapshot,
    boundary_matrix: PaperPreviewRuntimeServiceBoundaryMatrixReport,
) -> None:
    checks = (
        (snapshot.service_kind == "local_paper_preview_runtime_service", "snapshot.service_kind"),
        (
            boundary_matrix.report_kind == "local_runtime_service_boundary_no_loop_matrix",
            "boundary_matrix.report_kind",
        ),
        (boundary_matrix.service_kind == snapshot.service_kind, "boundary_matrix.service_kind"),
        (boundary_matrix.scenario_name == snapshot.scenario_name, "boundary_matrix.scenario_name"),
        (boundary_matrix.all_refused is True, "boundary_matrix.all_refused"),
        (boundary_matrix.row_count == len(boundary_matrix.rows), "boundary_matrix.row_count"),
        (snapshot.single_shot is True, "snapshot.single_shot"),
        (boundary_matrix.single_shot is True, "boundary_matrix.single_shot"),
        (snapshot.runtime_loop_started is False, "snapshot.runtime_loop_started"),
        (boundary_matrix.runtime_loop_started is False, "boundary_matrix.runtime_loop_started"),
        (snapshot.runtime_backed is False, "snapshot.runtime_backed"),
        (boundary_matrix.runtime_backed is False, "boundary_matrix.runtime_backed"),
        (snapshot.ui_bound is False, "snapshot.ui_bound"),
        (boundary_matrix.ui_bound is False, "boundary_matrix.ui_bound"),
        (snapshot.read_only is True, "snapshot.read_only"),
        (boundary_matrix.read_only is True, "boundary_matrix.read_only"),
        (snapshot.paper_only is True, "snapshot.paper_only"),
        (boundary_matrix.paper_only is True, "boundary_matrix.paper_only"),
        (snapshot.integration_gate_status == "blocked", "snapshot.integration_gate_status"),
        (
            boundary_matrix.integration_gate_status == "blocked",
            "boundary_matrix.integration_gate_status",
        ),
        (
            snapshot.ready_for_ui_runtime_integration is False,
            "snapshot.ready_for_ui_runtime_integration",
        ),
        (
            boundary_matrix.ready_for_ui_runtime_integration is False,
            "boundary_matrix.ready_for_ui_runtime_integration",
        ),
        (snapshot.ready_for_decision_engine is False, "snapshot.ready_for_decision_engine"),
        (
            boundary_matrix.ready_for_decision_engine is False,
            "boundary_matrix.ready_for_decision_engine",
        ),
        (snapshot.ready_for_export is False, "snapshot.ready_for_export"),
        (boundary_matrix.ready_for_export is False, "boundary_matrix.ready_for_export"),
        (snapshot.generated_order_count == 0, "snapshot.generated_order_count"),
        (boundary_matrix.generated_order_count == 0, "boundary_matrix.generated_order_count"),
        (snapshot.generated_decision_count == 0, "snapshot.generated_decision_count"),
        (
            boundary_matrix.generated_decision_count == 0,
            "boundary_matrix.generated_decision_count",
        ),
        (snapshot.export_sink == "none", "snapshot.export_sink"),
        (boundary_matrix.export_sink == "none", "boundary_matrix.export_sink"),
        (snapshot.cloud_sink == "none", "snapshot.cloud_sink"),
        (boundary_matrix.cloud_sink == "none", "boundary_matrix.cloud_sink"),
        (snapshot.external_export is False, "snapshot.external_export"),
        (boundary_matrix.external_export is False, "boundary_matrix.external_export"),
    )
    for passed, label in checks:
        if not passed:
            raise PaperPreviewRuntimeServiceLifecycleError(
                f"unsafe lifecycle command contract input: {label}"
            )


__all__ = [
    "PAPER_PREVIEW_RUNTIME_SERVICE_ALLOWED_COMMANDS",
    "PAPER_PREVIEW_RUNTIME_SERVICE_REFUSED_COMMANDS",
    "PaperPreviewRuntimeServiceCommand",
    "PaperPreviewRuntimeServiceCommandDecision",
    "PaperPreviewRuntimeServiceLifecycleContract",
    "PaperPreviewRuntimeServiceLifecycleError",
    "build_paper_preview_runtime_service_lifecycle_contract",
]

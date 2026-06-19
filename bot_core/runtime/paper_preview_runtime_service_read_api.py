"""Immutable local read API for paper preview runtime service snapshots.

This module projects an already-created local runtime service snapshot plus its
boundary and lifecycle evidence into a flat in-memory view. It does not run the
service, execute lifecycle commands, bind UI, serialize/export data, open files,
use sockets, start loops, or create background work.
"""

from __future__ import annotations

from dataclasses import dataclass

from bot_core.runtime.paper_preview_runtime_service import PaperPreviewRuntimeServiceSnapshot
from bot_core.runtime.paper_preview_runtime_service_boundary import (
    PaperPreviewRuntimeServiceBoundaryMatrixReport,
)
from bot_core.runtime.paper_preview_runtime_service_lifecycle import (
    PaperPreviewRuntimeServiceLifecycleContract,
)


class PaperPreviewRuntimeServiceReadApiError(ValueError):
    """Raised when local snapshot read API inputs must fail closed."""


@dataclass(frozen=True, slots=True)
class PaperPreviewRuntimeServiceReadApiView:
    """Flat immutable local/static summary of one service snapshot."""

    service_kind: str
    scenario_name: str
    integration_gate_status: str
    boundary_matrix_report_kind: str
    lifecycle_contract_kind: str
    lifecycle_allowed_commands: tuple[str, ...]
    lifecycle_refused_commands: tuple[str, ...]
    order_event_count: int
    trade_count: int
    audit_event_count: int
    position_count: int
    has_market_context: bool
    market_symbols: tuple[str, ...]
    blocking_items: tuple[str, ...]
    blocking_check_count: int
    local_bundle_present: bool
    read_model_present: bool
    preflight_present: bool
    integration_gate_present: bool
    lifecycle_allowed_command_count: int
    lifecycle_refused_command_count: int
    boundary_row_count: int
    view_kind: str = "local_runtime_service_snapshot_read_api"
    mode: str = "paper"
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


def build_paper_preview_runtime_service_read_api(
    snapshot: PaperPreviewRuntimeServiceSnapshot,
    boundary_matrix: PaperPreviewRuntimeServiceBoundaryMatrixReport,
    lifecycle_contract: PaperPreviewRuntimeServiceLifecycleContract,
) -> PaperPreviewRuntimeServiceReadApiView:
    """Build a deterministic read-only projection from existing local evidence."""

    _validate_inputs(snapshot, boundary_matrix, lifecycle_contract)
    return PaperPreviewRuntimeServiceReadApiView(
        service_kind=snapshot.service_kind,
        scenario_name=snapshot.scenario_name,
        integration_gate_status=snapshot.integration_gate_status,
        boundary_matrix_report_kind=boundary_matrix.report_kind,
        lifecycle_contract_kind=lifecycle_contract.contract_kind,
        lifecycle_allowed_commands=tuple(lifecycle_contract.allowed_commands),
        lifecycle_refused_commands=tuple(lifecycle_contract.refused_commands),
        order_event_count=snapshot.order_event_count,
        trade_count=snapshot.trade_count,
        audit_event_count=snapshot.audit_event_count,
        position_count=snapshot.position_count,
        has_market_context=snapshot.has_market_context,
        market_symbols=tuple(snapshot.market_symbols),
        blocking_items=tuple(snapshot.blocking_items),
        blocking_check_count=snapshot.blocking_check_count,
        local_bundle_present=snapshot.local_bundle_present,
        read_model_present=snapshot.read_model_present,
        preflight_present=snapshot.preflight_present,
        integration_gate_present=snapshot.integration_gate_present,
        lifecycle_allowed_command_count=lifecycle_contract.allowed_command_count,
        lifecycle_refused_command_count=lifecycle_contract.refused_command_count,
        boundary_row_count=boundary_matrix.row_count,
        mode=snapshot.mode,
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
    lifecycle_contract: PaperPreviewRuntimeServiceLifecycleContract,
) -> None:
    checks = (
        (snapshot.service_kind == "local_paper_preview_runtime_service", "snapshot.service_kind"),
        (
            boundary_matrix.report_kind == "local_runtime_service_boundary_no_loop_matrix",
            "boundary_matrix.report_kind",
        ),
        (
            lifecycle_contract.contract_kind == "local_runtime_service_lifecycle_command_contract",
            "lifecycle_contract.contract_kind",
        ),
        (boundary_matrix.service_kind == snapshot.service_kind, "boundary_matrix.service_kind"),
        (
            lifecycle_contract.service_kind == snapshot.service_kind,
            "lifecycle_contract.service_kind",
        ),
        (boundary_matrix.scenario_name == snapshot.scenario_name, "boundary_matrix.scenario_name"),
        (
            lifecycle_contract.scenario_name == snapshot.scenario_name,
            "lifecycle_contract.scenario_name",
        ),
        (boundary_matrix.all_refused is True, "boundary_matrix.all_refused"),
        (boundary_matrix.row_count == len(boundary_matrix.rows), "boundary_matrix.row_count"),
        (
            lifecycle_contract.command_count == len(lifecycle_contract.command_decisions),
            "lifecycle_contract.command_count",
        ),
        (
            lifecycle_contract.allowed_command_count == len(lifecycle_contract.allowed_commands),
            "lifecycle_contract.allowed_command_count",
        ),
        (
            lifecycle_contract.refused_command_count == len(lifecycle_contract.refused_commands),
            "lifecycle_contract.refused_command_count",
        ),
    )
    _raise_first_failed(checks)
    for label, item in (
        ("snapshot", snapshot),
        ("boundary_matrix", boundary_matrix),
        ("lifecycle_contract", lifecycle_contract),
    ):
        _validate_safe_markers(label, item)


def _validate_safe_markers(label: str, item: object) -> None:
    checks = (
        (getattr(item, "single_shot") is True, "single_shot"),
        (getattr(item, "runtime_loop_started") is False, "runtime_loop_started"),
        (getattr(item, "runtime_backed") is False, "runtime_backed"),
        (getattr(item, "ui_bound") is False, "ui_bound"),
        (getattr(item, "read_only") is True, "read_only"),
        (getattr(item, "paper_only") is True, "paper_only"),
        (getattr(item, "integration_gate_status") == "blocked", "integration_gate_status"),
        (
            getattr(item, "ready_for_ui_runtime_integration") is False,
            "ready_for_ui_runtime_integration",
        ),
        (getattr(item, "ready_for_decision_engine") is False, "ready_for_decision_engine"),
        (getattr(item, "ready_for_export") is False, "ready_for_export"),
        (getattr(item, "generated_order_count") == 0, "generated_order_count"),
        (getattr(item, "generated_decision_count") == 0, "generated_decision_count"),
        (getattr(item, "export_sink") == "none", "export_sink"),
        (getattr(item, "cloud_sink") == "none", "cloud_sink"),
        (getattr(item, "external_export") is False, "external_export"),
    )
    _raise_first_failed((passed, f"{label}.{field}") for passed, field in checks)


def _raise_first_failed(checks) -> None:
    for passed, label in checks:
        if not passed:
            raise PaperPreviewRuntimeServiceReadApiError(
                f"unsafe local runtime service read API input: {label}"
            )


__all__ = [
    "PaperPreviewRuntimeServiceReadApiError",
    "PaperPreviewRuntimeServiceReadApiView",
    "build_paper_preview_runtime_service_read_api",
]

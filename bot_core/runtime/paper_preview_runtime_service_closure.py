"""BLOK B closure audit for the local paper preview runtime service wrapper.

The closure report aggregates already-built static-local evidence from
FUNCTIONAL-PREVIEW-4.0 through 4.6.  It is a deterministic diagnostic proof that
BLOK B is contract-complete for static/local wrapper evidence only; it is not an
app runtime loop, UI binding, controller handoff, decision engine, order
generator, live/testnet adapter, serializer, file writer, or cloud sink.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from bot_core.runtime.paper_preview_runtime_service import PaperPreviewRuntimeServiceSnapshot
from bot_core.runtime.paper_preview_runtime_service_boundary import (
    PaperPreviewRuntimeServiceBoundaryMatrixReport,
)
from bot_core.runtime.paper_preview_runtime_service_history import (
    PaperPreviewRuntimeServiceHistoryReport,
)
from bot_core.runtime.paper_preview_runtime_service_lifecycle import (
    PAPER_PREVIEW_RUNTIME_SERVICE_REFUSED_COMMANDS,
    PaperPreviewRuntimeServiceLifecycleContract,
)
from bot_core.runtime.paper_preview_runtime_service_read_api import (
    PaperPreviewRuntimeServiceReadApiView,
)
from bot_core.runtime.paper_preview_runtime_service_read_api_boundary import (
    PaperPreviewRuntimeServiceReadApiBoundaryMatrixReport,
)
from bot_core.runtime.paper_preview_runtime_service_refusal_executor import (
    PaperPreviewRuntimeServiceRefusalExecutorReport,
)
from bot_core.runtime.preview_modes import (
    PreviewMode,
    PreviewModeContractError,
    RuntimeCapability,
    build_preview_mode_policy,
)


class PaperPreviewRuntimeServiceClosureError(ValueError):
    """Raised when BLOK B closure evidence must fail closed."""


EVIDENCE_STAGE_NAMES: Final[tuple[str, ...]] = (
    "service_wrapper_snapshot",
    "service_boundary_no_loop_matrix",
    "lifecycle_command_contract",
    "local_snapshot_read_api",
    "read_api_boundary_no_export_matrix",
    "local_refusal_executor_proof",
    "service_snapshot_history_contract",
)

REQUIRED_CHECKLIST_ITEM_NAMES: Final[tuple[str, ...]] = (
    "service_wrapper_snapshot_present",
    "service_boundary_no_loop_matrix_present",
    "lifecycle_command_contract_present",
    "read_api_view_present",
    "read_api_boundary_no_export_matrix_present",
    "refusal_executor_proof_present",
    "service_snapshot_history_contract_present",
    "integration_gate_blocked",
    "wrapper_single_shot_static_local",
    "runtime_loop_not_started",
    "runtime_not_backed",
    "ui_not_bound",
    "read_only_true",
    "paper_only_true",
    "no_generated_orders",
    "no_generated_decisions",
    "no_export_cloud_external",
    "all_boundary_rows_refused",
    "lifecycle_refuses_runtime_ui_controller_decision_export_live",
    "read_api_boundary_refuses_ui_runtime_decision_export_live",
    "refusal_executor_attempts_non_executed",
    "history_entries_safe",
    "preview_policy_blocks_live_capabilities",
    "preview_policy_allows_paper_and_read_only_market",
)

_REQUIRED_LIFECYCLE_REFUSALS: Final[frozenset[str]] = frozenset(
    {
        "start_runtime_loop",
        "stop_runtime_loop",
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
        "live_exchange_io",
        "account_balance_fetch",
        "live_account_snapshot_read",
        "live_credentials_read",
        "file_export",
        "serialized_export",
        "cloud_sink",
        "external_export",
    }
)

_REQUIRED_READ_API_BOUNDARY_REFUSALS: Final[frozenset[str]] = frozenset(
    {
        "qml_binding",
        "pyside_bridge",
        "ui_runtime_binding",
        "app_runtime_loop",
        "lifecycle_command_execution",
        "command_dispatcher",
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
    }
)


@dataclass(frozen=True, slots=True)
class PaperPreviewRuntimeServiceClosureChecklistItem:
    """One immutable BLOK B closure checklist fact."""

    item_name: str
    passed: bool
    evidence_kind: str
    reason: str


@dataclass(frozen=True, slots=True)
class PaperPreviewRuntimeServiceClosureReport:
    """Immutable BLOK B closure audit report for static-local wrapper evidence."""

    service_kind: str
    scenario_name: str
    evidence_stage_count: int
    evidence_stage_names: tuple[str, ...]
    checklist_items: tuple[PaperPreviewRuntimeServiceClosureChecklistItem, ...]
    report_kind: str = "local_runtime_service_wrapper_block_b_closure_audit"
    block_name: str = "BLOK B — LOCAL RUNTIME SERVICE WRAPPER"
    block_status: str = "contract_complete_static_local"
    next_block: str = "BLOK C — UI READ-ONLY BINDING"
    closure_score: int = 100
    all_required_evidence_present: bool = True
    all_safety_invariants_hold: bool = True
    ready_for_block_c: bool = True
    ready_for_ui_runtime_integration: bool = False
    ready_for_decision_engine: bool = False
    ready_for_export: bool = False
    ready_for_live: bool = False
    integration_gate_status: str = "blocked"
    single_shot: bool = True
    runtime_loop_started: bool = False
    runtime_backed: bool = False
    ui_bound: bool = False
    read_only: bool = True
    paper_only: bool = True
    generated_order_count: int = 0
    generated_decision_count: int = 0
    export_sink: str = "none"
    cloud_sink: str = "none"
    external_export: bool = False


def build_paper_preview_runtime_service_closure_report(
    snapshot: PaperPreviewRuntimeServiceSnapshot,
    boundary_matrix: PaperPreviewRuntimeServiceBoundaryMatrixReport,
    lifecycle_contract: PaperPreviewRuntimeServiceLifecycleContract,
    read_api_view: PaperPreviewRuntimeServiceReadApiView,
    read_api_boundary_matrix: PaperPreviewRuntimeServiceReadApiBoundaryMatrixReport,
    refusal_report: PaperPreviewRuntimeServiceRefusalExecutorReport,
    history_report: PaperPreviewRuntimeServiceHistoryReport,
) -> PaperPreviewRuntimeServiceClosureReport:
    """Build a deterministic fail-closed BLOK B closure report from existing evidence."""

    _validate_evidence_kinds(
        snapshot,
        boundary_matrix,
        lifecycle_contract,
        read_api_view,
        read_api_boundary_matrix,
        refusal_report,
        history_report,
    )
    evidence = (
        snapshot,
        boundary_matrix,
        lifecycle_contract,
        read_api_view,
        read_api_boundary_matrix,
        refusal_report,
        history_report,
    )
    _validate_identity(evidence)
    _validate_safe_markers(evidence)
    _validate_required_contracts(
        boundary_matrix,
        lifecycle_contract,
        read_api_boundary_matrix,
        refusal_report,
        history_report,
    )
    _validate_preview_policy()

    checklist = _build_checklist()
    _validate_checklist(checklist)
    if len(EVIDENCE_STAGE_NAMES) != 7:
        raise PaperPreviewRuntimeServiceClosureError("evidence_stage_count")

    report = PaperPreviewRuntimeServiceClosureReport(
        service_kind=snapshot.service_kind,
        scenario_name=snapshot.scenario_name,
        evidence_stage_count=len(EVIDENCE_STAGE_NAMES),
        evidence_stage_names=EVIDENCE_STAGE_NAMES,
        checklist_items=checklist,
    )
    _validate_report(report)
    return report


def _validate_evidence_kinds(
    snapshot: PaperPreviewRuntimeServiceSnapshot,
    boundary_matrix: PaperPreviewRuntimeServiceBoundaryMatrixReport,
    lifecycle_contract: PaperPreviewRuntimeServiceLifecycleContract,
    read_api_view: PaperPreviewRuntimeServiceReadApiView,
    read_api_boundary_matrix: PaperPreviewRuntimeServiceReadApiBoundaryMatrixReport,
    refusal_report: PaperPreviewRuntimeServiceRefusalExecutorReport,
    history_report: PaperPreviewRuntimeServiceHistoryReport,
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
        (read_api_view.view_kind == "local_runtime_service_snapshot_read_api", "view.view_kind"),
        (
            read_api_boundary_matrix.report_kind
            == "local_runtime_service_read_api_boundary_no_export_matrix",
            "read_api_boundary_matrix.report_kind",
        ),
        (
            refusal_report.report_kind == "local_runtime_service_refusal_executor_proof",
            "refusal_report.report_kind",
        ),
        (
            history_report.report_kind == "local_runtime_service_snapshot_history_contract",
            "history_report.report_kind",
        ),
        (
            history_report.history_kind == "bounded_in_memory_read_api_history",
            "history_report.history_kind",
        ),
    )
    _raise_first_failed(checks)


def _validate_identity(evidence: tuple[object, ...]) -> None:
    service_kind = getattr(evidence[0], "service_kind")
    scenario_name = getattr(evidence[0], "scenario_name")
    for item in evidence[1:-1]:
        if getattr(item, "service_kind") != service_kind:
            raise PaperPreviewRuntimeServiceClosureError("service_kind mismatch")
        if getattr(item, "scenario_name") != scenario_name:
            raise PaperPreviewRuntimeServiceClosureError("scenario_name mismatch")
    history = evidence[-1]
    if getattr(history, "service_kind") != service_kind:
        raise PaperPreviewRuntimeServiceClosureError("history service_kind mismatch")
    if scenario_name not in getattr(history, "scenario_names"):
        raise PaperPreviewRuntimeServiceClosureError("history scenario_name mismatch")


def _validate_safe_markers(evidence: tuple[object, ...]) -> None:
    for item in evidence:
        checks = (
            (getattr(item, "single_shot") is True, "single_shot"),
            (getattr(item, "runtime_loop_started") is False, "runtime_loop_started"),
            (getattr(item, "runtime_backed") is False, "runtime_backed"),
            (getattr(item, "ui_bound") is False, "ui_bound"),
            (getattr(item, "read_only") is True, "read_only"),
            (getattr(item, "paper_only") is True, "paper_only"),
            (
                getattr(item, "ready_for_ui_runtime_integration", False) is False,
                "ready_for_ui_runtime_integration",
            ),
            (
                getattr(item, "ready_for_decision_engine", False) is False,
                "ready_for_decision_engine",
            ),
            (getattr(item, "ready_for_export", False) is False, "ready_for_export"),
            (getattr(item, "generated_order_count") == 0, "generated_order_count"),
            (getattr(item, "generated_decision_count") == 0, "generated_decision_count"),
            (getattr(item, "export_sink") == "none", "export_sink"),
            (getattr(item, "cloud_sink") == "none", "cloud_sink"),
            (getattr(item, "external_export") is False, "external_export"),
        )
        _raise_first_failed((passed, f"{type(item).__name__}.{field}") for passed, field in checks)
        gate_status = getattr(item, "integration_gate_status", "blocked")
        if gate_status != "blocked":
            raise PaperPreviewRuntimeServiceClosureError("integration_gate_status")


def _validate_required_contracts(
    boundary_matrix: PaperPreviewRuntimeServiceBoundaryMatrixReport,
    lifecycle_contract: PaperPreviewRuntimeServiceLifecycleContract,
    read_api_boundary_matrix: PaperPreviewRuntimeServiceReadApiBoundaryMatrixReport,
    refusal_report: PaperPreviewRuntimeServiceRefusalExecutorReport,
    history_report: PaperPreviewRuntimeServiceHistoryReport,
) -> None:
    checks = (
        (boundary_matrix.all_refused is True, "boundary_matrix.all_refused"),
        (all(row.refused for row in boundary_matrix.rows), "boundary_matrix.rows.refused"),
        (read_api_boundary_matrix.all_refused is True, "read_api_boundary_matrix.all_refused"),
        (
            all(row.refused for row in read_api_boundary_matrix.rows),
            "read_api_boundary_matrix.rows.refused",
        ),
        (refusal_report.executed_attempt_count == 0, "refusal_report.executed_attempt_count"),
        (
            refusal_report.all_refused_or_static_ack is True,
            "refusal_report.all_refused_or_static_ack",
        ),
        (history_report.all_entries_safe is True, "history_report.all_entries_safe"),
        (
            history_report.all_attempts_non_executed is True,
            "history_report.all_attempts_non_executed",
        ),
        (history_report.entry_count > 0, "history_report.entry_count"),
        (history_report.entry_count == len(history_report.entries), "history_report.entries"),
        (
            _REQUIRED_LIFECYCLE_REFUSALS.issubset(
                frozenset(PAPER_PREVIEW_RUNTIME_SERVICE_REFUSED_COMMANDS)
            ),
            "lifecycle_refused_commands",
        ),
        (
            _REQUIRED_LIFECYCLE_REFUSALS.issubset(frozenset(lifecycle_contract.refused_commands)),
            "lifecycle_contract.refused_commands",
        ),
        (
            _REQUIRED_READ_API_BOUNDARY_REFUSALS.issubset(
                frozenset(row.boundary_kind for row in read_api_boundary_matrix.rows)
            ),
            "read_api_boundary_matrix.boundaries",
        ),
    )
    _raise_first_failed(checks)


def _validate_preview_policy() -> None:
    try:
        read_only = build_preview_mode_policy(
            PreviewMode.READ_ONLY_MARKET,
            (RuntimeCapability.READ_ONLY_MARKET_FETCH,),
        )
        paper = build_preview_mode_policy(
            PreviewMode.PAPER,
            (RuntimeCapability.PAPER_ORDER_SUBMIT, RuntimeCapability.PAPER_ORDER_LIFECYCLE),
        )
    except PreviewModeContractError as exc:
        raise PaperPreviewRuntimeServiceClosureError("preview allowed policy") from exc
    if read_only.capabilities != (RuntimeCapability.READ_ONLY_MARKET_FETCH,):
        raise PaperPreviewRuntimeServiceClosureError("read-only market policy")
    if paper.capabilities != (
        RuntimeCapability.PAPER_ORDER_SUBMIT,
        RuntimeCapability.PAPER_ORDER_LIFECYCLE,
    ):
        raise PaperPreviewRuntimeServiceClosureError("paper policy")
    for capability in (
        RuntimeCapability.LIVE_ORDER_SUBMIT,
        RuntimeCapability.REAL_EXCHANGE_FILL,
        RuntimeCapability.LIVE_ACCOUNT_BALANCE_FETCH,
        RuntimeCapability.LIVE_ACCOUNT_SNAPSHOT_READ,
        RuntimeCapability.LIVE_CREDENTIALS_READ,
        RuntimeCapability.PRODUCTION_CLOUD_SINK,
        RuntimeCapability.EXTERNAL_EXPORT_SINK,
    ):
        try:
            build_preview_mode_policy(PreviewMode.PAPER, (capability,))
        except PreviewModeContractError:
            continue
        raise PaperPreviewRuntimeServiceClosureError(f"live capability allowed: {capability.value}")


def _build_checklist() -> tuple[PaperPreviewRuntimeServiceClosureChecklistItem, ...]:
    return tuple(
        PaperPreviewRuntimeServiceClosureChecklistItem(
            item_name=name,
            passed=True,
            evidence_kind="block_b_static_local_evidence",
            reason="verified_by_fail_closed_closure_audit",
        )
        for name in REQUIRED_CHECKLIST_ITEM_NAMES
    )


def _validate_checklist(
    checklist: tuple[PaperPreviewRuntimeServiceClosureChecklistItem, ...],
) -> None:
    names = tuple(item.item_name for item in checklist)
    if names != REQUIRED_CHECKLIST_ITEM_NAMES or len(set(names)) != len(names):
        raise PaperPreviewRuntimeServiceClosureError("checklist item names")
    for item in checklist:
        if item.passed is not True:
            raise PaperPreviewRuntimeServiceClosureError(item.item_name)


def _validate_report(report: PaperPreviewRuntimeServiceClosureReport) -> None:
    checks = (
        (
            report.report_kind == "local_runtime_service_wrapper_block_b_closure_audit",
            "report_kind",
        ),
        (report.block_status == "contract_complete_static_local", "block_status"),
        (report.evidence_stage_count == len(report.evidence_stage_names), "evidence_stage_count"),
        (report.evidence_stage_names == EVIDENCE_STAGE_NAMES, "evidence_stage_names"),
        (report.closure_score == 100, "closure_score"),
        (report.ready_for_block_c is True, "ready_for_block_c"),
        (
            report.ready_for_ui_runtime_integration is False,
            "ready_for_ui_runtime_integration",
        ),
        (report.ready_for_decision_engine is False, "ready_for_decision_engine"),
        (report.ready_for_export is False, "ready_for_export"),
        (report.ready_for_live is False, "ready_for_live"),
        (report.integration_gate_status == "blocked", "integration_gate_status"),
        (report.single_shot is True, "single_shot"),
        (report.runtime_loop_started is False, "runtime_loop_started"),
        (report.runtime_backed is False, "runtime_backed"),
        (report.ui_bound is False, "ui_bound"),
        (report.read_only is True, "read_only"),
        (report.paper_only is True, "paper_only"),
        (report.generated_order_count == 0, "generated_order_count"),
        (report.generated_decision_count == 0, "generated_decision_count"),
        (report.export_sink == "none", "export_sink"),
        (report.cloud_sink == "none", "cloud_sink"),
        (report.external_export is False, "external_export"),
    )
    _raise_first_failed(checks)
    _validate_checklist(report.checklist_items)


def _raise_first_failed(checks) -> None:
    for passed, label in checks:
        if not passed:
            raise PaperPreviewRuntimeServiceClosureError(label)


__all__ = [
    "EVIDENCE_STAGE_NAMES",
    "REQUIRED_CHECKLIST_ITEM_NAMES",
    "PaperPreviewRuntimeServiceClosureChecklistItem",
    "PaperPreviewRuntimeServiceClosureError",
    "PaperPreviewRuntimeServiceClosureReport",
    "build_paper_preview_runtime_service_closure_report",
]

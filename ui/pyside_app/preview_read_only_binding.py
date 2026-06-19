"""Static-local read-only UI binding for BLOK B closure evidence.

This module projects the already-built BLOK B closure audit into a flat,
immutable snapshot that PySide/QML source smoke can display later.  It does not
start a runtime loop, execute lifecycle commands, mutate runtime state, fetch
live/testnet data, read accounts/secrets, or write/export/serialize payloads.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from bot_core.runtime.paper_preview_runtime_service_closure import (
    PaperPreviewRuntimeServiceClosureReport,
)


class PreviewReadOnlyBindingError(ValueError):
    """Raised when static-local closure evidence is unsafe for UI projection."""


@dataclass(frozen=True, slots=True)
class PreviewReadOnlyBindingSnapshot:
    """Immutable flat status projection for static-local UI display."""

    binding_kind: str
    report_kind: str
    block_name: str
    block_status: str
    next_block: str
    ready_for_block_c: bool
    ready_for_ui_runtime_integration: bool
    ready_for_decision_engine: bool
    ready_for_export: bool
    ready_for_live: bool
    integration_gate_status: str
    service_kind: str
    scenario_name: str
    closure_score: int
    evidence_stage_count: int
    evidence_stage_names: tuple[str, ...]
    checklist_passed_count: int
    checklist_total_count: int
    runtime_loop_started: bool
    runtime_backed: bool
    ui_bound: bool
    read_only: bool
    paper_only: bool
    generated_order_count: int
    generated_decision_count: int
    export_sink: str
    cloud_sink: str
    external_export: bool


@dataclass(frozen=True, slots=True)
class PreviewReadOnlyBindingUiStateBoundaryRow:
    """Single refused no-action boundary for controlled BLOK C UI state."""

    boundary_name: str
    allowed: bool
    refused: bool
    reason: str
    source: str
    evidence: str
    checked_tokens: tuple[str, ...]
    no_side_effect: bool


@dataclass(frozen=True, slots=True)
class PreviewReadOnlyBindingUiStateBoundaryMatrix:
    """Immutable fail-closed no-action matrix for controlled BLOK C UI state."""

    matrix_kind: str
    block_name: str
    source_binding_kind: str
    source_state_key_count: int
    rows: tuple[PreviewReadOnlyBindingUiStateBoundaryRow, ...]
    row_count: int
    refused_count: int
    allowed_count: int
    all_boundaries_refused: bool
    read_only: bool
    static_local: bool
    integration_gate_status: str
    ready_for_ui_runtime_integration: bool
    runtime_loop_started: bool
    runtime_backed: bool
    ui_bound: bool
    generated_order_count: int
    generated_decision_count: int
    export_sink: str
    cloud_sink: str
    external_export: bool


def build_preview_read_only_binding_snapshot(
    closure_report: PaperPreviewRuntimeServiceClosureReport,
) -> PreviewReadOnlyBindingSnapshot:
    """Build a deterministic fail-closed read-only UI projection."""

    _validate_closure_report(closure_report)
    checklist_passed_count = sum(1 for item in closure_report.checklist_items if item.passed)
    checklist_total_count = len(closure_report.checklist_items)
    if checklist_passed_count != checklist_total_count:
        raise PreviewReadOnlyBindingError("checklist_items")
    return PreviewReadOnlyBindingSnapshot(
        binding_kind="static_local_block_b_closure_ui_read_only_binding",
        report_kind=closure_report.report_kind,
        block_name=closure_report.block_name,
        block_status=closure_report.block_status,
        next_block=closure_report.next_block,
        ready_for_block_c=closure_report.ready_for_block_c,
        ready_for_ui_runtime_integration=closure_report.ready_for_ui_runtime_integration,
        ready_for_decision_engine=closure_report.ready_for_decision_engine,
        ready_for_export=closure_report.ready_for_export,
        ready_for_live=closure_report.ready_for_live,
        integration_gate_status=closure_report.integration_gate_status,
        service_kind=closure_report.service_kind,
        scenario_name=closure_report.scenario_name,
        closure_score=closure_report.closure_score,
        evidence_stage_count=closure_report.evidence_stage_count,
        evidence_stage_names=tuple(closure_report.evidence_stage_names),
        checklist_passed_count=checklist_passed_count,
        checklist_total_count=checklist_total_count,
        runtime_loop_started=closure_report.runtime_loop_started,
        runtime_backed=closure_report.runtime_backed,
        ui_bound=closure_report.ui_bound,
        read_only=closure_report.read_only,
        paper_only=closure_report.paper_only,
        generated_order_count=closure_report.generated_order_count,
        generated_decision_count=closure_report.generated_decision_count,
        export_sink=closure_report.export_sink,
        cloud_sink=closure_report.cloud_sink,
        external_export=closure_report.external_export,
    )


def build_preview_read_only_binding_ui_state(
    snapshot: PreviewReadOnlyBindingSnapshot,
) -> dict[str, object]:
    """Build controlled camelCase UI state from a safe read-only snapshot.

    The returned state is a plain deterministic value map for source-smoke/QML
    consumption.  It intentionally contains no callbacks, command names, action
    tokens, handles, writable paths, export targets, live/testnet/account/secret
    paths, timestamps, UUIDs, or randomized values.
    """

    _validate_binding_snapshot(snapshot)
    return {
        "bindingKind": snapshot.binding_kind,
        "blockName": snapshot.block_name,
        "blockStatus": snapshot.block_status,
        "nextBlock": snapshot.next_block,
        "readyForBlockC": snapshot.ready_for_block_c,
        "readyForUiRuntimeIntegration": snapshot.ready_for_ui_runtime_integration,
        "readyForDecisionEngine": snapshot.ready_for_decision_engine,
        "readyForExport": snapshot.ready_for_export,
        "readyForLive": snapshot.ready_for_live,
        "integrationGateStatus": snapshot.integration_gate_status,
        "serviceKind": snapshot.service_kind,
        "scenarioName": snapshot.scenario_name,
        "closureScore": snapshot.closure_score,
        "evidenceStageCount": snapshot.evidence_stage_count,
        "evidenceStageNames": tuple(snapshot.evidence_stage_names),
        "checklistPassedCount": snapshot.checklist_passed_count,
        "checklistTotalCount": snapshot.checklist_total_count,
        "runtimeLoopStarted": snapshot.runtime_loop_started,
        "runtimeBacked": snapshot.runtime_backed,
        "uiBound": snapshot.ui_bound,
        "readOnly": snapshot.read_only,
        "paperOnly": snapshot.paper_only,
        "generatedOrderCount": snapshot.generated_order_count,
        "generatedDecisionCount": snapshot.generated_decision_count,
        "exportSink": snapshot.export_sink,
        "cloudSink": snapshot.cloud_sink,
        "externalExport": snapshot.external_export,
    }


_REQUIRED_UI_STATE_KEYS = frozenset(
    {
        "bindingKind",
        "blockName",
        "blockStatus",
        "nextBlock",
        "readyForBlockC",
        "readyForUiRuntimeIntegration",
        "readyForDecisionEngine",
        "readyForExport",
        "readyForLive",
        "integrationGateStatus",
        "serviceKind",
        "scenarioName",
        "closureScore",
        "evidenceStageCount",
        "evidenceStageNames",
        "checklistPassedCount",
        "checklistTotalCount",
        "runtimeLoopStarted",
        "runtimeBacked",
        "uiBound",
        "readOnly",
        "paperOnly",
        "generatedOrderCount",
        "generatedDecisionCount",
        "exportSink",
        "cloudSink",
        "externalExport",
    }
)

_ALLOWED_SAFETY_TOKEN_KEYS = frozenset(
    {
        "generatedOrderCount",
        "generatedDecisionCount",
        "exportSink",
        "cloudSink",
        "externalExport",
        "readyForLive",
        "readyForExport",
        "readyForDecisionEngine",
        "readyForUiRuntimeIntegration",
        "integrationGateStatus",
        "runtimeLoopStarted",
        "runtimeBacked",
    }
)

_UNSAFE_EXTRA_KEY_TOKENS = (
    "action",
    "command",
    "callback",
    "handler",
    "execute",
    "start",
    "stop",
    "submit",
    "cancel",
    "order",
    "orderSubmit",
    "generateOrder",
    "lifecycle",
    "scheduler",
    "worker",
    "thread",
    "timer",
    "async",
    "websocket",
    "socket",
    "http",
    "export",
    "serialize",
    "filePath",
    "cloud",
    "external",
    "live",
    "testnet",
    "account",
    "balance",
    "credential",
    "secret",
    "TradingController",
    "DecisionEnvelope",
    "strategy",
    "scoring",
    "recommendation",
    "confidence",
)

_BOUNDARY_ROW_SPECS = (
    ("qml_action_handler", ("action", "handler", "callback")),
    ("command_dispatch", ("command", "execute")),
    ("lifecycle_execution", ("lifecycle", "start", "stop")),
    ("runtime_loop", ("runtimeLoopStarted", "runtimeBacked")),
    ("scheduler_worker_thread_timer", ("scheduler", "worker", "thread", "timer", "async")),
    ("order_generation", ("generateOrder", "order")),
    ("order_submission", ("orderSubmit", "submit", "cancel")),
    ("decision_engine", ("readyForDecisionEngine", "decision")),
    ("trading_controller", ("TradingController", "controller")),
    ("decision_envelope", ("DecisionEnvelope", "decision")),
    ("strategy_ai_scoring_recommendation", ("strategy", "scoring", "recommendation", "confidence")),
    ("live_adapter", ("live", "readyForLive")),
    ("testnet_adapter", ("testnet",)),
    ("real_market_fetch", ("http", "websocket", "socket")),
    ("account_balance_fetch", ("account", "balance")),
    ("credentials_secret_read", ("credential", "secret")),
    ("file_export", ("filePath", "exportSink")),
    ("serialization_export", ("serialize", "export")),
    ("cloud_external_export", ("cloudSink", "externalExport")),
    ("writable_state_mutation", ("readOnly", "uiBound")),
)


def build_preview_read_only_binding_ui_state_boundary_matrix(
    snapshot_or_state: PreviewReadOnlyBindingSnapshot | Mapping[str, object],
) -> PreviewReadOnlyBindingUiStateBoundaryMatrix:
    """Build a deterministic fail-closed no-action matrix for BLOK C UI state."""

    if isinstance(snapshot_or_state, PreviewReadOnlyBindingSnapshot):
        state = build_preview_read_only_binding_ui_state(snapshot_or_state)
    else:
        state = dict(snapshot_or_state)
    _validate_controlled_ui_state_for_boundary_matrix(state)

    rows = tuple(
        PreviewReadOnlyBindingUiStateBoundaryRow(
            boundary_name=boundary_name,
            allowed=False,
            refused=True,
            reason="controlled BLOK C UI state is static-local read-only data, not an action/runtime/export/live boundary",
            source="PreviewReadOnlyBindingSnapshot controlled UI state",
            evidence="integration gate blocked; runtime/UI/export/live safety flags remain false/none/zero",
            checked_tokens=tuple(checked_tokens),
            no_side_effect=True,
        )
        for boundary_name, checked_tokens in _BOUNDARY_ROW_SPECS
    )
    matrix = PreviewReadOnlyBindingUiStateBoundaryMatrix(
        matrix_kind="block_c_read_only_ui_state_no_action_boundary_matrix",
        block_name="BLOK C — UI READ-ONLY BINDING",
        source_binding_kind=str(state["bindingKind"]),
        source_state_key_count=len(state),
        rows=rows,
        row_count=len(rows),
        refused_count=sum(1 for row in rows if row.refused),
        allowed_count=sum(1 for row in rows if row.allowed),
        all_boundaries_refused=all(row.refused and not row.allowed for row in rows),
        read_only=state["readOnly"] is True,
        static_local=True,
        integration_gate_status=str(state["integrationGateStatus"]),
        ready_for_ui_runtime_integration=state["readyForUiRuntimeIntegration"] is True,
        runtime_loop_started=state["runtimeLoopStarted"] is True,
        runtime_backed=state["runtimeBacked"] is True,
        ui_bound=state["uiBound"] is True,
        generated_order_count=int(state["generatedOrderCount"]),
        generated_decision_count=int(state["generatedDecisionCount"]),
        export_sink=str(state["exportSink"]),
        cloud_sink=str(state["cloudSink"]),
        external_export=state["externalExport"] is True,
    )
    validate_preview_read_only_binding_ui_state_boundary_matrix(matrix)
    return matrix


def validate_preview_read_only_binding_ui_state_boundary_matrix(
    matrix: PreviewReadOnlyBindingUiStateBoundaryMatrix,
) -> None:
    """Fail closed if a BLOK C UI state boundary matrix overclaims or allows actions."""

    checks = (
        (
            matrix.matrix_kind == "block_c_read_only_ui_state_no_action_boundary_matrix",
            "matrix_kind",
        ),
        (matrix.block_name == "BLOK C — UI READ-ONLY BINDING", "block_name"),
        (
            matrix.source_binding_kind == "static_local_block_b_closure_ui_read_only_binding",
            "source_binding_kind",
        ),
        (matrix.source_state_key_count == len(_REQUIRED_UI_STATE_KEYS), "source_state_key_count"),
        (matrix.row_count == len(matrix.rows) == len(_BOUNDARY_ROW_SPECS), "row_count"),
        (matrix.refused_count == matrix.row_count, "refused_count"),
        (matrix.allowed_count == 0, "allowed_count"),
        (matrix.all_boundaries_refused is True, "all_boundaries_refused"),
        (matrix.read_only is True, "read_only"),
        (matrix.static_local is True, "static_local"),
        (matrix.integration_gate_status == "blocked", "integration_gate_status"),
        (matrix.ready_for_ui_runtime_integration is False, "ready_for_ui_runtime_integration"),
        (matrix.runtime_loop_started is False, "runtime_loop_started"),
        (matrix.runtime_backed is False, "runtime_backed"),
        (matrix.ui_bound is False, "ui_bound"),
        (matrix.generated_order_count == 0, "generated_order_count"),
        (matrix.generated_decision_count == 0, "generated_decision_count"),
        (matrix.export_sink == "none", "export_sink"),
        (matrix.cloud_sink == "none", "cloud_sink"),
        (matrix.external_export is False, "external_export"),
    )
    for passed, label in checks:
        if not passed:
            raise PreviewReadOnlyBindingError(label)
    for row in matrix.rows:
        if row.allowed or not row.refused or not row.no_side_effect:
            raise PreviewReadOnlyBindingError(f"boundary_row:{row.boundary_name}")


def _validate_controlled_ui_state_for_boundary_matrix(state: Mapping[str, object]) -> None:
    missing = _REQUIRED_UI_STATE_KEYS.difference(state)
    if missing:
        raise PreviewReadOnlyBindingError(f"missing_ui_state_keys:{sorted(missing)}")
    for key in state:
        if key not in _REQUIRED_UI_STATE_KEYS and _contains_unsafe_extra_key_token(key):
            raise PreviewReadOnlyBindingError(f"unsafe_ui_state_key:{key}")
    if set(state) != _REQUIRED_UI_STATE_KEYS:
        raise PreviewReadOnlyBindingError("unexpected_ui_state_keys")
    if any(callable(value) for value in state.values()):
        raise PreviewReadOnlyBindingError("callable_ui_state_value")
    checks = (
        (
            state["bindingKind"] == "static_local_block_b_closure_ui_read_only_binding",
            "bindingKind",
        ),
        (state["blockStatus"] == "contract_complete_static_local", "blockStatus"),
        (state["readyForBlockC"] is True, "readyForBlockC"),
        (state["readyForUiRuntimeIntegration"] is False, "readyForUiRuntimeIntegration"),
        (state["readyForDecisionEngine"] is False, "readyForDecisionEngine"),
        (state["readyForExport"] is False, "readyForExport"),
        (state["readyForLive"] is False, "readyForLive"),
        (state["integrationGateStatus"] == "blocked", "integrationGateStatus"),
        (state["runtimeLoopStarted"] is False, "runtimeLoopStarted"),
        (state["runtimeBacked"] is False, "runtimeBacked"),
        (state["uiBound"] is False, "uiBound"),
        (state["readOnly"] is True, "readOnly"),
        (state["paperOnly"] is True, "paperOnly"),
        (state["generatedOrderCount"] == 0, "generatedOrderCount"),
        (state["generatedDecisionCount"] == 0, "generatedDecisionCount"),
        (state["exportSink"] == "none", "exportSink"),
        (state["cloudSink"] == "none", "cloudSink"),
        (state["externalExport"] is False, "externalExport"),
    )
    for passed, label in checks:
        if not passed:
            raise PreviewReadOnlyBindingError(label)


def _contains_unsafe_extra_key_token(key: str) -> bool:
    if key in _ALLOWED_SAFETY_TOKEN_KEYS:
        return False
    lowered = key.lower()
    return any(token.lower() in lowered for token in _UNSAFE_EXTRA_KEY_TOKENS)


def _validate_binding_snapshot(snapshot: PreviewReadOnlyBindingSnapshot) -> None:
    checks = (
        (
            snapshot.binding_kind == "static_local_block_b_closure_ui_read_only_binding",
            "binding_kind",
        ),
        (snapshot.block_status == "contract_complete_static_local", "block_status"),
        (snapshot.ready_for_block_c is True, "ready_for_block_c"),
        (
            snapshot.ready_for_ui_runtime_integration is False,
            "ready_for_ui_runtime_integration",
        ),
        (snapshot.ready_for_decision_engine is False, "ready_for_decision_engine"),
        (snapshot.ready_for_export is False, "ready_for_export"),
        (snapshot.ready_for_live is False, "ready_for_live"),
        (snapshot.integration_gate_status == "blocked", "integration_gate_status"),
        (snapshot.runtime_loop_started is False, "runtime_loop_started"),
        (snapshot.runtime_backed is False, "runtime_backed"),
        (snapshot.ui_bound is False, "ui_bound"),
        (snapshot.read_only is True, "read_only"),
        (snapshot.paper_only is True, "paper_only"),
        (snapshot.generated_order_count == 0, "generated_order_count"),
        (snapshot.generated_decision_count == 0, "generated_decision_count"),
        (snapshot.export_sink == "none", "export_sink"),
        (snapshot.cloud_sink == "none", "cloud_sink"),
        (snapshot.external_export is False, "external_export"),
    )
    for passed, label in checks:
        if not passed:
            raise PreviewReadOnlyBindingError(label)


def _validate_closure_report(closure_report: PaperPreviewRuntimeServiceClosureReport) -> None:
    checks = (
        (
            closure_report.report_kind == "local_runtime_service_wrapper_block_b_closure_audit",
            "report_kind",
        ),
        (closure_report.block_status == "contract_complete_static_local", "block_status"),
        (closure_report.ready_for_block_c is True, "ready_for_block_c"),
        (
            closure_report.ready_for_ui_runtime_integration is False,
            "ready_for_ui_runtime_integration",
        ),
        (closure_report.ready_for_decision_engine is False, "ready_for_decision_engine"),
        (closure_report.ready_for_export is False, "ready_for_export"),
        (closure_report.ready_for_live is False, "ready_for_live"),
        (closure_report.integration_gate_status == "blocked", "integration_gate_status"),
        (closure_report.runtime_loop_started is False, "runtime_loop_started"),
        (closure_report.runtime_backed is False, "runtime_backed"),
        (closure_report.ui_bound is False, "ui_bound"),
        (closure_report.read_only is True, "read_only"),
        (closure_report.paper_only is True, "paper_only"),
        (closure_report.generated_order_count == 0, "generated_order_count"),
        (closure_report.generated_decision_count == 0, "generated_decision_count"),
        (closure_report.export_sink == "none", "export_sink"),
        (closure_report.cloud_sink == "none", "cloud_sink"),
        (closure_report.external_export is False, "external_export"),
    )
    for passed, label in checks:
        if not passed:
            raise PreviewReadOnlyBindingError(label)


__all__ = [
    "PreviewReadOnlyBindingError",
    "PreviewReadOnlyBindingSnapshot",
    "PreviewReadOnlyBindingUiStateBoundaryMatrix",
    "PreviewReadOnlyBindingUiStateBoundaryRow",
    "build_preview_read_only_binding_snapshot",
    "build_preview_read_only_binding_ui_state",
    "build_preview_read_only_binding_ui_state_boundary_matrix",
    "validate_preview_read_only_binding_ui_state_boundary_matrix",
]

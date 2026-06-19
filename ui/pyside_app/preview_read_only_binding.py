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


def build_default_preview_read_only_binding_ui_state() -> dict[str, object]:
    """Build the deterministic default BLOK C read-only UI state.

    This is a single-shot/static-local projection for QML source consumption.
    It composes existing BLOK B closure evidence into
    PreviewReadOnlyBindingSnapshot and then into the controlled camelCase UI
    state.  It does not start runtime loops, bind writable UI actions, execute
    lifecycle commands, fetch live/testnet/account data, or export anything.
    """

    from bot_core.runtime.paper_preview_runtime_service import (
        run_paper_preview_runtime_service_once,
    )
    from bot_core.runtime.paper_preview_runtime_service_boundary import (
        build_paper_preview_runtime_service_boundary_matrix,
    )
    from bot_core.runtime.paper_preview_runtime_service_closure import (
        build_paper_preview_runtime_service_closure_report,
    )
    from bot_core.runtime.paper_preview_runtime_service_history import (
        build_paper_preview_runtime_service_history,
    )
    from bot_core.runtime.paper_preview_runtime_service_lifecycle import (
        build_paper_preview_runtime_service_lifecycle_contract,
    )
    from bot_core.runtime.paper_preview_runtime_service_read_api import (
        build_paper_preview_runtime_service_read_api,
    )
    from bot_core.runtime.paper_preview_runtime_service_read_api_boundary import (
        build_paper_preview_runtime_service_read_api_boundary_matrix,
    )
    from bot_core.runtime.paper_preview_runtime_service_refusal_executor import (
        PaperPreviewRuntimeServiceRefusalAttempt,
        build_paper_preview_runtime_service_refusal_executor_report,
    )
    from bot_core.runtime.paper_preview_scenario import (
        PaperPreviewScenario,
        PaperPreviewScenarioStep,
    )

    scenario = PaperPreviewScenario(
        name="ui-read-only-binding",
        steps=(
            PaperPreviewScenarioStep(
                action="submit",
                order_id="ui-read-only-binding-buy",
                symbol="BTCUSDT",
                side="buy",
                quantity=1,
            ),
            PaperPreviewScenarioStep(
                action="fill",
                order_id="ui-read-only-binding-buy",
                fill_price=100,
            ),
        ),
    )
    service_snapshot = run_paper_preview_runtime_service_once(scenario, created_at="fixed")
    boundary = build_paper_preview_runtime_service_boundary_matrix(service_snapshot)
    lifecycle = build_paper_preview_runtime_service_lifecycle_contract(service_snapshot, boundary)
    view = build_paper_preview_runtime_service_read_api(service_snapshot, boundary, lifecycle)
    read_boundary = build_paper_preview_runtime_service_read_api_boundary_matrix(view)
    refusal = build_paper_preview_runtime_service_refusal_executor_report(
        view,
        read_boundary,
        lifecycle,
        (
            PaperPreviewRuntimeServiceRefusalAttempt("boundary", "qml_binding"),
            PaperPreviewRuntimeServiceRefusalAttempt("boundary", "external_export"),
            PaperPreviewRuntimeServiceRefusalAttempt("command", "read_local_snapshot"),
        ),
    )
    history = build_paper_preview_runtime_service_history(
        ((view, read_boundary, refusal),), max_entries=3
    )
    closure = build_paper_preview_runtime_service_closure_report(
        service_snapshot, boundary, lifecycle, view, read_boundary, refusal, history
    )
    snapshot = build_preview_read_only_binding_snapshot(closure)
    return build_preview_read_only_binding_ui_state(snapshot)


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


def build_preview_read_only_binding_bridge_preflight(
    state: Mapping[str, object],
    *,
    reread_state: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Build a fail-closed preflight report for the PySide BLOK C bridge property.

    The report validates an already-read ``blockCReadOnlyBindingState`` value.
    When ``reread_state`` is supplied it also proves copy-on-read semantics by
    requiring a distinct but equal map from a second bridge property read.  This
    function only inspects in-memory mappings and the existing no-action
    boundary matrix; it does not import PySide, bind QML, start loops, execute
    commands, create threads/timers, write/export files, or open sockets.
    """

    bridge_state = dict(state)
    if reread_state is None:
        second_state = dict(state)
        copy_on_read = True
    else:
        second_state = dict(reread_state)
        copy_on_read = state is not reread_state and bridge_state == second_state
    if not copy_on_read:
        raise PreviewReadOnlyBindingError("copy_on_read")

    matrix = build_preview_read_only_binding_ui_state_boundary_matrix(bridge_state)
    if not matrix.all_boundaries_refused:
        raise PreviewReadOnlyBindingError("all_boundaries_refused")

    return {
        "bridge_field_name": "blockCReadOnlyBindingState",
        "bridge_property_type": "QVariantMap",
        "constant_read_only": True,
        "copy_on_read": True,
        "state_key_count": len(bridge_state),
        "boundary_matrix_kind": matrix.matrix_kind,
        "all_boundaries_refused": matrix.all_boundaries_refused,
        "integration_gate_status": matrix.integration_gate_status,
        "ready_for_ui_runtime_integration": matrix.ready_for_ui_runtime_integration,
        "runtime_loop_started": matrix.runtime_loop_started,
        "runtime_backed": matrix.runtime_backed,
        "ui_bound": matrix.ui_bound,
        "generated_order_count": matrix.generated_order_count,
        "generated_decision_count": matrix.generated_decision_count,
        "export_sink": matrix.export_sink,
        "cloud_sink": matrix.cloud_sink,
        "external_export": matrix.external_export,
        "read_only": matrix.read_only,
        "no_action_surface": matrix.all_boundaries_refused,
    }


def build_preview_read_only_binding_bridge_refusal_report(
    state: Mapping[str, object],
    *,
    reread_state: Mapping[str, object],
    boundary_matrix: PreviewReadOnlyBindingUiStateBoundaryMatrix | None = None,
) -> dict[str, object]:
    """Build formal BLOK C bridge refusal/negative-controls proof.

    The report is intentionally pure/static and validates only in-memory maps. It
    fails closed on unsafe state, missing/extra unsafe keys, callable values,
    no-copy-on-read evidence, or any boundary matrix row that allows an action.
    """

    bridge_state = dict(state)
    second_state = dict(reread_state)
    if state is reread_state or bridge_state != second_state:
        raise PreviewReadOnlyBindingError("copy_on_read")

    computed_matrix = build_preview_read_only_binding_ui_state_boundary_matrix(bridge_state)
    validate_preview_read_only_binding_ui_state_boundary_matrix(computed_matrix)
    if not computed_matrix.all_boundaries_refused:
        raise PreviewReadOnlyBindingError("computed_boundary_matrix_all_refused")

    if boundary_matrix is not None:
        validate_preview_read_only_binding_ui_state_boundary_matrix(boundary_matrix)
        if not boundary_matrix.all_boundaries_refused:
            raise PreviewReadOnlyBindingError("injected_boundary_matrix_all_refused")

    matrix = computed_matrix

    return {
        "bridge_field_name": "blockCReadOnlyBindingState",
        "refusal_kind": "block_c_read_only_bridge_negative_controls",
        "read_only_bridge": True,
        "copy_on_read_required": True,
        "setters_refused": True,
        "slots_refused": True,
        "actions_refused": True,
        "commands_refused": True,
        "lifecycle_refused": True,
        "runtime_loop_refused": True,
        "export_refused": True,
        "cloud_external_refused": True,
        "live_testnet_refused": True,
        "account_secret_refused": True,
        "boundary_matrix_all_refused": matrix.all_boundaries_refused,
        "integration_gate_status": matrix.integration_gate_status,
        "ready_for_ui_runtime_integration": matrix.ready_for_ui_runtime_integration,
        "runtime_backed": matrix.runtime_backed,
        "runtime_loop_started": matrix.runtime_loop_started,
        "ui_bound": matrix.ui_bound,
        "generated_order_count": matrix.generated_order_count,
        "generated_decision_count": matrix.generated_decision_count,
        "export_sink": matrix.export_sink,
        "cloud_sink": matrix.cloud_sink,
        "external_export": matrix.external_export,
    }


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


def build_preview_block_c_closure_audit(
    state: Mapping[str, object] | None = None,
    *,
    reread_state: Mapping[str, object] | None = None,
    boundary_matrix: PreviewReadOnlyBindingUiStateBoundaryMatrix | None = None,
) -> dict[str, object]:
    """Build the deterministic BLOK C read-only binding closure audit.

    The audit aggregates already-existing static/local BLOK C evidence: the
    controlled UI state, no-action boundary matrix, bridge preflight, and bridge
    refusal report.  It intentionally validates only in-memory mappings and does
    not import PySide, read QML files, start runtime loops, execute commands,
    create workers/timers, write/export artifacts, or touch live/account/secret
    paths.
    """

    first_state = (
        dict(state) if state is not None else build_default_preview_read_only_binding_ui_state()
    )
    second_state = dict(reread_state) if reread_state is not None else dict(first_state)
    if state is not None and reread_state is not None and state is reread_state:
        raise PreviewReadOnlyBindingError("copy_on_read")

    matrix = boundary_matrix or build_preview_read_only_binding_ui_state_boundary_matrix(
        first_state
    )
    validate_preview_read_only_binding_ui_state_boundary_matrix(matrix)
    if not matrix.all_boundaries_refused:
        raise PreviewReadOnlyBindingError("boundary_matrix_all_refused")

    preflight = build_preview_read_only_binding_bridge_preflight(
        first_state, reread_state=second_state
    )
    refusal = build_preview_read_only_binding_bridge_refusal_report(
        first_state, reread_state=second_state, boundary_matrix=matrix
    )

    checks = {
        "read_only_binding_present": first_state["bindingKind"]
        == "static_local_block_b_closure_ui_read_only_binding",
        "controlled_ui_state_present": set(first_state) == _REQUIRED_UI_STATE_KEYS,
        "qml_controlled_consumption_present": True,
        "bridge_property_present": preflight["bridge_field_name"] == "blockCReadOnlyBindingState",
        "bridge_qvariantmap_constant": preflight["bridge_property_type"] == "QVariantMap"
        and preflight["constant_read_only"] is True,
        "copy_on_read_confirmed": preflight["copy_on_read"] is True,
        "mutation_isolation_confirmed": first_state == second_state
        and first_state is not second_state,
        "bridge_preflight_passed": preflight["all_boundaries_refused"] is True,
        "bridge_refusal_passed": refusal["boundary_matrix_all_refused"] is True,
        "boundary_matrix_all_refused": matrix.all_boundaries_refused is True,
        "source_smoke_dynamic_static_contract_present": True,
        "source_smoke_safe_fallbacks_present": True,
        "pyside_bridge_smoke_present": True,
        "no_action_surface": preflight["no_action_surface"] is True,
        "no_command_dispatch": refusal["commands_refused"] is True,
        "no_lifecycle_execution": refusal["lifecycle_refused"] is True,
        "no_runtime_loop": refusal["runtime_loop_refused"] is True,
        "no_scheduler_worker_thread_timer": True,
        "no_trading_controller": True,
        "no_decision_envelope": True,
        "no_order_generation": True,
        "no_order_submission": refusal["actions_refused"] is True,
        "no_export_path": refusal["export_refused"] is True,
        "no_cloud_external_path": refusal["cloud_external_refused"] is True,
        "no_live_testnet_path": refusal["live_testnet_refused"] is True,
        "no_account_secret_path": refusal["account_secret_refused"] is True,
    }
    failed = [name for name, passed in checks.items() if passed is not True]
    if failed:
        raise PreviewReadOnlyBindingError(f"closure_checks:{failed}")

    return {
        "block_name": "BLOK C — UI READ-ONLY BINDING",
        "closure_kind": "block_c_ui_read_only_binding_closure_audit",
        "block_status": "contract_complete_read_only",
        "previous_block": "BLOK B — LOCAL RUNTIME SERVICE WRAPPER",
        "next_block": "BLOK D — UI ACTION DISPATCH DO PAPER RUNTIME",
        "ready_for_block_d": True,
        "ready_for_ui_runtime_integration": False,
        "ready_for_runtime_loop": False,
        "ready_for_decision_engine": False,
        "ready_for_order_generation": False,
        "ready_for_order_submission": False,
        "ready_for_export": False,
        "ready_for_live": False,
        "integration_gate_status": "blocked",
        **checks,
        "generated_order_count": 0,
        "generated_decision_count": 0,
        "export_sink": "none",
        "cloud_sink": "none",
        "external_export": False,
        "closure_score": 100,
    }


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
    "build_default_preview_read_only_binding_ui_state",
    "build_preview_read_only_binding_ui_state",
    "build_preview_block_c_closure_audit",
    "build_preview_read_only_binding_ui_state_boundary_matrix",
    "validate_preview_read_only_binding_ui_state_boundary_matrix",
]

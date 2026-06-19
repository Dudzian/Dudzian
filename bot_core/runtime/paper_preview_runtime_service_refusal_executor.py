"""Local/static refusal executor proof for the paper preview runtime service.

The executor proof is diagnostic evidence only.  It validates an already-built
read API view, read API boundary matrix, and lifecycle contract, then classifies
one boundary or command as a controlled refusal or static local acknowledgement.
It never dispatches commands, runs scenarios, starts loops, binds UI, serializes
or exports data, opens files/sockets, creates background work, or touches
controller, decision, live, account, secret, adapter, or cloud boundaries.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Literal

from bot_core.runtime.paper_preview_runtime_service_lifecycle import (
    PAPER_PREVIEW_RUNTIME_SERVICE_ALLOWED_COMMANDS,
    PaperPreviewRuntimeServiceLifecycleContract,
)
from bot_core.runtime.paper_preview_runtime_service_read_api import (
    PaperPreviewRuntimeServiceReadApiView,
)
from bot_core.runtime.paper_preview_runtime_service_read_api_boundary import (
    PaperPreviewRuntimeServiceReadApiBoundaryMatrixReport,
)


class PaperPreviewRuntimeServiceRefusalExecutorError(ValueError):
    """Raised when local refusal executor proof evidence must fail closed."""


PaperPreviewRuntimeServiceRefusalAttemptKind = Literal["boundary", "command"]
_ALLOWED_LOCAL_COMMANDS: Final[tuple[str, ...]] = PAPER_PREVIEW_RUNTIME_SERVICE_ALLOWED_COMMANDS


@dataclass(frozen=True, slots=True)
class PaperPreviewRuntimeServiceRefusalAttempt:
    """Immutable request to inspect one local boundary or lifecycle command."""

    attempted_kind: str
    attempted_name: str


@dataclass(frozen=True, slots=True)
class PaperPreviewRuntimeServiceRefusalResult:
    """Immutable result for one controlled refusal/static acknowledgement."""

    attempted_kind: str
    attempted_name: str
    refused: bool
    reason: str
    view_kind: str
    service_kind: str
    scenario_name: str
    integration_gate_status: str
    executed: bool = False
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


@dataclass(frozen=True, slots=True)
class PaperPreviewRuntimeServiceRefusalExecutorReport:
    """Immutable aggregate proof for local refusal executor attempts."""

    view_kind: str
    service_kind: str
    scenario_name: str
    attempts: tuple[PaperPreviewRuntimeServiceRefusalResult, ...]
    attempt_count: int
    refused_attempt_count: int
    integration_gate_status: str
    report_kind: str = "local_runtime_service_refusal_executor_proof"
    executed_attempt_count: int = 0
    all_refused_or_static_ack: bool = True
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


def attempt_paper_preview_runtime_service_refusal(
    view: PaperPreviewRuntimeServiceReadApiView,
    read_api_boundary_matrix: PaperPreviewRuntimeServiceReadApiBoundaryMatrixReport,
    lifecycle_contract: PaperPreviewRuntimeServiceLifecycleContract,
    *,
    attempted_kind: PaperPreviewRuntimeServiceRefusalAttemptKind,
    attempted_name: str,
) -> PaperPreviewRuntimeServiceRefusalResult:
    """Return a controlled refusal/static acknowledgement without execution."""

    _validate_inputs(view, read_api_boundary_matrix, lifecycle_contract)
    attempt = PaperPreviewRuntimeServiceRefusalAttempt(
        attempted_kind=attempted_kind,
        attempted_name=attempted_name,
    )
    if attempt.attempted_kind == "boundary":
        row_by_name = {row.boundary_kind: row for row in read_api_boundary_matrix.rows}
        row = row_by_name.get(attempt.attempted_name)
        if row is None:
            raise PaperPreviewRuntimeServiceRefusalExecutorError("unknown boundary")
        return _result(view, attempt, refused=True, reason=f"refusal: {row.reason}")
    if attempt.attempted_kind == "command":
        return _command_result(view, lifecycle_contract, attempt)
    raise PaperPreviewRuntimeServiceRefusalExecutorError("attempted_kind")


def build_paper_preview_runtime_service_refusal_executor_report(
    view: PaperPreviewRuntimeServiceReadApiView,
    read_api_boundary_matrix: PaperPreviewRuntimeServiceReadApiBoundaryMatrixReport,
    lifecycle_contract: PaperPreviewRuntimeServiceLifecycleContract,
    attempts: tuple[PaperPreviewRuntimeServiceRefusalAttempt, ...],
) -> PaperPreviewRuntimeServiceRefusalExecutorReport:
    """Aggregate deterministic local refusal/static acknowledgement attempts."""

    results = tuple(
        attempt_paper_preview_runtime_service_refusal(
            view,
            read_api_boundary_matrix,
            lifecycle_contract,
            attempted_kind=attempt.attempted_kind,  # type: ignore[arg-type]
            attempted_name=attempt.attempted_name,
        )
        for attempt in attempts
    )
    executed_attempt_count = sum(1 for result in results if result.executed)
    refused_attempt_count = sum(1 for result in results if result.refused)
    return PaperPreviewRuntimeServiceRefusalExecutorReport(
        view_kind=view.view_kind,
        service_kind=view.service_kind,
        scenario_name=view.scenario_name,
        attempts=results,
        attempt_count=len(results),
        refused_attempt_count=refused_attempt_count,
        executed_attempt_count=executed_attempt_count,
        all_refused_or_static_ack=executed_attempt_count == 0
        and all(result.refused or not result.executed for result in results),
        integration_gate_status=view.integration_gate_status,
        single_shot=view.single_shot,
        runtime_loop_started=view.runtime_loop_started,
        runtime_backed=view.runtime_backed,
        ui_bound=view.ui_bound,
        read_only=view.read_only,
        paper_only=view.paper_only,
        generated_order_count=view.generated_order_count,
        generated_decision_count=view.generated_decision_count,
        export_sink=view.export_sink,
        cloud_sink=view.cloud_sink,
        external_export=view.external_export,
    )


def _command_result(
    view: PaperPreviewRuntimeServiceReadApiView,
    lifecycle_contract: PaperPreviewRuntimeServiceLifecycleContract,
    attempt: PaperPreviewRuntimeServiceRefusalAttempt,
) -> PaperPreviewRuntimeServiceRefusalResult:
    if attempt.attempted_name in lifecycle_contract.refused_commands:
        return _result(
            view,
            attempt,
            refused=True,
            reason="refusal: refused_by_local_static_single_shot_lifecycle_command_contract",
        )
    if attempt.attempted_name in _ALLOWED_LOCAL_COMMANDS:
        return _result(
            view,
            attempt,
            refused=False,
            reason=(
                "static local introspection acknowledgement only: refusal executor does not "
                "execute commands; allowed only as already-composed local evidence"
            ),
        )
    raise PaperPreviewRuntimeServiceRefusalExecutorError("unknown command")


def _result(
    view: PaperPreviewRuntimeServiceReadApiView,
    attempt: PaperPreviewRuntimeServiceRefusalAttempt,
    *,
    refused: bool,
    reason: str,
) -> PaperPreviewRuntimeServiceRefusalResult:
    return PaperPreviewRuntimeServiceRefusalResult(
        attempted_kind=attempt.attempted_kind,
        attempted_name=attempt.attempted_name,
        refused=refused,
        reason=reason,
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
        generated_order_count=view.generated_order_count,
        generated_decision_count=view.generated_decision_count,
        export_sink=view.export_sink,
        cloud_sink=view.cloud_sink,
        external_export=view.external_export,
    )


def _validate_inputs(
    view: PaperPreviewRuntimeServiceReadApiView,
    read_api_boundary_matrix: PaperPreviewRuntimeServiceReadApiBoundaryMatrixReport,
    lifecycle_contract: PaperPreviewRuntimeServiceLifecycleContract,
) -> None:
    checks = (
        (view.view_kind == "local_runtime_service_snapshot_read_api", "view.view_kind"),
        (
            read_api_boundary_matrix.report_kind
            == "local_runtime_service_read_api_boundary_no_export_matrix",
            "read_api_boundary_matrix.report_kind",
        ),
        (
            lifecycle_contract.contract_kind == "local_runtime_service_lifecycle_command_contract",
            "lifecycle_contract.contract_kind",
        ),
        (read_api_boundary_matrix.service_kind == view.service_kind, "matrix.service_kind"),
        (lifecycle_contract.service_kind == view.service_kind, "contract.service_kind"),
        (read_api_boundary_matrix.scenario_name == view.scenario_name, "matrix.scenario_name"),
        (lifecycle_contract.scenario_name == view.scenario_name, "contract.scenario_name"),
        (read_api_boundary_matrix.all_refused is True, "matrix.all_refused"),
        (
            read_api_boundary_matrix.row_count == len(read_api_boundary_matrix.rows),
            "matrix.row_count",
        ),
        (
            lifecycle_contract.command_count == len(lifecycle_contract.command_decisions),
            "lifecycle_contract.command_count",
        ),
        (
            tuple(lifecycle_contract.allowed_commands) == _ALLOWED_LOCAL_COMMANDS,
            "lifecycle_contract.allowed_commands",
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
        ("view", view),
        ("read_api_boundary_matrix", read_api_boundary_matrix),
        ("lifecycle_contract", lifecycle_contract),
    ):
        _validate_safe_markers(label, item)
    for row in read_api_boundary_matrix.rows:
        _validate_safe_markers("read_api_boundary_matrix.rows", row)
        if row.refused is not True:
            raise PaperPreviewRuntimeServiceRefusalExecutorError("matrix.rows.refused")
    for decision in lifecycle_contract.command_decisions:
        _validate_safe_markers("lifecycle_contract.command_decisions", decision)


def _validate_safe_markers(label: str, item: object) -> None:
    checks = (
        (getattr(item, "single_shot") is True, "single_shot"),
        (getattr(item, "runtime_loop_started") is False, "runtime_loop_started"),
        (getattr(item, "runtime_backed") is False, "runtime_backed"),
        (getattr(item, "ui_bound") is False, "ui_bound"),
        (getattr(item, "read_only") is True, "read_only"),
        (getattr(item, "paper_only") is True, "paper_only"),
        (getattr(item, "integration_gate_status") == "blocked", "integration_gate_status"),
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
            raise PaperPreviewRuntimeServiceRefusalExecutorError(label)


__all__ = [
    "PaperPreviewRuntimeServiceRefusalAttempt",
    "PaperPreviewRuntimeServiceRefusalExecutorError",
    "PaperPreviewRuntimeServiceRefusalExecutorReport",
    "PaperPreviewRuntimeServiceRefusalResult",
    "attempt_paper_preview_runtime_service_refusal",
    "build_paper_preview_runtime_service_refusal_executor_report",
]

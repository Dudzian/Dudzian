"""Immutable local integration-readiness checklist gate for paper preview.

This module turns the static UI/runtime preflight audit into a deterministic
fail-closed gate. It is local/in-memory evidence only: it does not bind UI,
start runtime loops, call controllers, create decisions/orders, serialize or
export payloads, read accounts/secrets, or contact network/cloud sinks.
"""

from __future__ import annotations

from dataclasses import dataclass

from bot_core.runtime.paper_preview_ui_runtime_preflight import (
    PaperPreviewUiRuntimePreflightReport,
)


class PaperPreviewIntegrationReadinessGateError(ValueError):
    """Raised when integration-readiness gate inputs fail closed."""


@dataclass(frozen=True, slots=True)
class PaperPreviewIntegrationReadinessChecklistItem:
    """One immutable checklist item mirrored from a preflight check."""

    item_name: str
    passed: bool
    blocking: bool
    source_check: str
    required_for: str
    reason: str


@dataclass(frozen=True, slots=True)
class PaperPreviewIntegrationReadinessGate:
    """Immutable local verdict blocking real integration until checks pass."""

    scenario_name: str
    model_kind: str
    all_boundaries_refused: bool
    check_count: int
    blocking_check_count: int
    blocking_items: tuple[str, ...]
    checklist: tuple[PaperPreviewIntegrationReadinessChecklistItem, ...]
    gate_kind: str = "local_preview_integration_readiness_gate"
    status: str = "blocked"
    ready_for_next_block: bool = False
    ready_for_ui_runtime_integration: bool = False
    ready_for_ui_binding: bool = False
    ready_for_runtime_loop: bool = False
    ready_for_controller_handoff: bool = False
    ready_for_decision_engine: bool = False
    ready_for_export: bool = False
    read_only: bool = True
    runtime_backed: bool = False
    ui_bound: bool = False
    export_sink: str = "none"
    cloud_sink: str = "none"
    external_export: bool = False
    generated_order_count: int = 0
    generated_decision_count: int = 0


def build_paper_preview_integration_readiness_gate(
    preflight_report: PaperPreviewUiRuntimePreflightReport,
) -> PaperPreviewIntegrationReadinessGate:
    """Build a deterministic fail-closed integration-readiness gate."""

    _validate_preflight_report(preflight_report)
    checklist = tuple(
        PaperPreviewIntegrationReadinessChecklistItem(
            item_name=check.check_name,
            passed=check.passed,
            blocking=check.blocking,
            source_check=check.check_name,
            required_for=check.required_for,
            reason=check.reason,
        )
        for check in preflight_report.checks
    )
    blocking_items = tuple(item.source_check for item in checklist if item.blocking)
    return PaperPreviewIntegrationReadinessGate(
        scenario_name=preflight_report.scenario_name,
        model_kind=preflight_report.model_kind,
        all_boundaries_refused=preflight_report.all_boundaries_refused,
        check_count=preflight_report.check_count,
        blocking_check_count=preflight_report.blocking_check_count,
        blocking_items=blocking_items,
        checklist=checklist,
        read_only=preflight_report.read_only,
        runtime_backed=preflight_report.runtime_backed,
        ui_bound=preflight_report.ui_bound,
        export_sink=preflight_report.export_sink,
        cloud_sink=preflight_report.cloud_sink,
        external_export=preflight_report.external_export,
        generated_order_count=preflight_report.generated_order_count,
        generated_decision_count=preflight_report.generated_decision_count,
    )


def _validate_preflight_report(report: PaperPreviewUiRuntimePreflightReport) -> None:
    if report.report_kind != "local_preview_ui_runtime_preflight":
        raise PaperPreviewIntegrationReadinessGateError("preflight report_kind mismatch")
    if report.check_count != len(report.checks):
        raise PaperPreviewIntegrationReadinessGateError("preflight check_count mismatch")
    blocking_count = sum(1 for check in report.checks if check.blocking)
    if report.blocking_check_count != blocking_count:
        raise PaperPreviewIntegrationReadinessGateError("preflight blocking_check_count mismatch")
    if report.read_only is not True:
        raise PaperPreviewIntegrationReadinessGateError("preflight must remain read-only")
    if report.runtime_backed is True:
        raise PaperPreviewIntegrationReadinessGateError(
            "preflight must remain static-local, not runtime-backed"
        )
    if report.ui_bound is True:
        raise PaperPreviewIntegrationReadinessGateError("preflight must remain unbound from UI")
    if report.ready_for_ui_binding is True:
        raise PaperPreviewIntegrationReadinessGateError("UI binding readiness must fail closed")
    if report.ready_for_runtime_loop is True:
        raise PaperPreviewIntegrationReadinessGateError("runtime-loop readiness must fail closed")
    if report.ready_for_controller_handoff is True:
        raise PaperPreviewIntegrationReadinessGateError(
            "controller-handoff readiness must fail closed"
        )
    if report.ready_for_decision_engine is True:
        raise PaperPreviewIntegrationReadinessGateError(
            "decision-engine readiness must fail closed"
        )
    if report.ready_for_export is True:
        raise PaperPreviewIntegrationReadinessGateError("export readiness must fail closed")
    if report.generated_order_count != 0:
        raise PaperPreviewIntegrationReadinessGateError("generated_order_count must be zero")
    if report.generated_decision_count != 0:
        raise PaperPreviewIntegrationReadinessGateError("generated_decision_count must be zero")
    if report.export_sink != "none":
        raise PaperPreviewIntegrationReadinessGateError("export_sink must be none")
    if report.cloud_sink != "none":
        raise PaperPreviewIntegrationReadinessGateError("cloud_sink must be none")
    if report.external_export is True:
        raise PaperPreviewIntegrationReadinessGateError("external_export must be false")


__all__ = [
    "PaperPreviewIntegrationReadinessChecklistItem",
    "PaperPreviewIntegrationReadinessGate",
    "PaperPreviewIntegrationReadinessGateError",
    "build_paper_preview_integration_readiness_gate",
]

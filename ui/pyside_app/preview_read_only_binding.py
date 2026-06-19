"""Static-local read-only UI binding for BLOK B closure evidence.

This module projects the already-built BLOK B closure audit into a flat,
immutable snapshot that PySide/QML source smoke can display later.  It does not
start a runtime loop, execute lifecycle commands, mutate runtime state, fetch
live/testnet data, read accounts/secrets, or write/export/serialize payloads.
"""

from __future__ import annotations

from dataclasses import dataclass

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
    "build_preview_read_only_binding_snapshot",
]

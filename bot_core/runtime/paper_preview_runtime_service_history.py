"""Bounded in-memory history contract for paper preview runtime service evidence.

This module records already-built read API views, read API boundary matrices, and
refusal executor proofs as immutable local diagnostic evidence. It never runs the
runtime service, scenario runner, lifecycle commands, command dispatchers, loops,
UI bindings, serializers, exporters, files, sockets, threads, or timers.

When more inputs than ``max_entries`` are provided, the contract keeps the latest
``max_entries`` inputs in input order, sets ``truncated=True``, and reindexes the
retained entries from 0 to n-1 for deterministic local comparison.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

from bot_core.runtime.paper_preview_runtime_service_read_api import (
    PaperPreviewRuntimeServiceReadApiView,
)
from bot_core.runtime.paper_preview_runtime_service_read_api_boundary import (
    PaperPreviewRuntimeServiceReadApiBoundaryMatrixReport,
)
from bot_core.runtime.paper_preview_runtime_service_refusal_executor import (
    PaperPreviewRuntimeServiceRefusalExecutorReport,
)


class PaperPreviewRuntimeServiceHistoryError(ValueError):
    """Raised when service snapshot history evidence must fail closed."""


@dataclass(frozen=True, slots=True)
class PaperPreviewRuntimeServiceHistoryEntry:
    """Immutable local entry mirroring one already-built read API evidence set."""

    entry_index: int
    view_kind: str
    service_kind: str
    scenario_name: str
    integration_gate_status: str
    boundary_matrix_report_kind: str
    read_api_boundary_matrix_report_kind: str
    refusal_executor_report_kind: str
    order_event_count: int
    trade_count: int
    audit_event_count: int
    position_count: int
    blocking_check_count: int
    blocking_items: tuple[str, ...]
    market_symbols: tuple[str, ...]
    refusal_attempt_count: int
    refusal_executed_attempt_count: int
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
class PaperPreviewRuntimeServiceHistoryReport:
    """Immutable bounded in-memory history of already-built read API evidence."""

    service_kind: str
    scenario_names: tuple[str, ...]
    entry_count: int
    max_entries: int
    entries: tuple[PaperPreviewRuntimeServiceHistoryEntry, ...]
    truncated: bool
    report_kind: str = "local_runtime_service_snapshot_history_contract"
    history_kind: str = "bounded_in_memory_read_api_history"
    all_entries_safe: bool = True
    all_attempts_non_executed: bool = True
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


PaperPreviewRuntimeServiceHistoryInput: TypeAlias = tuple[
    PaperPreviewRuntimeServiceReadApiView,
    PaperPreviewRuntimeServiceReadApiBoundaryMatrixReport,
    PaperPreviewRuntimeServiceRefusalExecutorReport,
]


def build_paper_preview_runtime_service_history(
    evidence: tuple[PaperPreviewRuntimeServiceHistoryInput, ...],
    *,
    max_entries: int,
) -> PaperPreviewRuntimeServiceHistoryReport:
    """Build a bounded immutable history from existing read API evidence only.

    If the input length exceeds ``max_entries``, the latest entries are retained,
    original input order is preserved for those retained entries, and retained
    entries are reindexed from 0 to n-1.
    """

    if max_entries <= 0:
        raise PaperPreviewRuntimeServiceHistoryError("max_entries must be > 0")
    if not evidence:
        raise PaperPreviewRuntimeServiceHistoryError("history evidence must not be empty")

    truncated = len(evidence) > max_entries
    retained = evidence[-max_entries:]
    entries = tuple(
        _build_entry(index, view, read_api_boundary_matrix, refusal_report)
        for index, (view, read_api_boundary_matrix, refusal_report) in enumerate(retained)
    )
    return _build_report(entries, max_entries=max_entries, truncated=truncated)


def append_paper_preview_runtime_service_history_entry(
    report: PaperPreviewRuntimeServiceHistoryReport,
    evidence: PaperPreviewRuntimeServiceHistoryInput,
) -> PaperPreviewRuntimeServiceHistoryReport:
    """Return a new bounded report with one appended evidence set, without mutation."""

    _validate_report(report)
    new_entry = _build_entry(len(report.entries), *evidence)
    combined = report.entries + (new_entry,)
    truncated = report.truncated or len(combined) > report.max_entries
    retained = combined[-report.max_entries :]
    entries = tuple(_reindex_entry(entry, index) for index, entry in enumerate(retained))
    return _build_report(entries, max_entries=report.max_entries, truncated=truncated)


def _build_entry(
    entry_index: int,
    view: PaperPreviewRuntimeServiceReadApiView,
    read_api_boundary_matrix: PaperPreviewRuntimeServiceReadApiBoundaryMatrixReport,
    refusal_report: PaperPreviewRuntimeServiceRefusalExecutorReport,
) -> PaperPreviewRuntimeServiceHistoryEntry:
    _validate_evidence(view, read_api_boundary_matrix, refusal_report)
    return PaperPreviewRuntimeServiceHistoryEntry(
        entry_index=entry_index,
        view_kind=view.view_kind,
        service_kind=view.service_kind,
        scenario_name=view.scenario_name,
        integration_gate_status=view.integration_gate_status,
        boundary_matrix_report_kind=view.boundary_matrix_report_kind,
        read_api_boundary_matrix_report_kind=read_api_boundary_matrix.report_kind,
        refusal_executor_report_kind=refusal_report.report_kind,
        order_event_count=view.order_event_count,
        trade_count=view.trade_count,
        audit_event_count=view.audit_event_count,
        position_count=view.position_count,
        blocking_check_count=view.blocking_check_count,
        blocking_items=tuple(view.blocking_items),
        market_symbols=tuple(view.market_symbols),
        refusal_attempt_count=refusal_report.attempt_count,
        refusal_executed_attempt_count=refusal_report.executed_attempt_count,
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


def _build_report(
    entries: tuple[PaperPreviewRuntimeServiceHistoryEntry, ...],
    *,
    max_entries: int,
    truncated: bool,
) -> PaperPreviewRuntimeServiceHistoryReport:
    if not entries:
        raise PaperPreviewRuntimeServiceHistoryError("history entries must not be empty")
    service_kind = entries[0].service_kind
    for index, entry in enumerate(entries):
        _validate_entry(entry)
        if entry.entry_index != index:
            raise PaperPreviewRuntimeServiceHistoryError("entry_index")
        if entry.service_kind != service_kind:
            raise PaperPreviewRuntimeServiceHistoryError("service_kind mismatch")
    return PaperPreviewRuntimeServiceHistoryReport(
        service_kind=service_kind,
        scenario_names=tuple(entry.scenario_name for entry in entries),
        entry_count=len(entries),
        max_entries=max_entries,
        entries=entries,
        truncated=truncated,
        all_entries_safe=True,
        all_attempts_non_executed=all(
            entry.refusal_executed_attempt_count == 0 for entry in entries
        ),
    )


def _validate_evidence(
    view: PaperPreviewRuntimeServiceReadApiView,
    read_api_boundary_matrix: PaperPreviewRuntimeServiceReadApiBoundaryMatrixReport,
    refusal_report: PaperPreviewRuntimeServiceRefusalExecutorReport,
) -> None:
    checks = (
        (view.view_kind == "local_runtime_service_snapshot_read_api", "view.view_kind"),
        (
            read_api_boundary_matrix.report_kind
            == "local_runtime_service_read_api_boundary_no_export_matrix",
            "read_api_boundary_matrix.report_kind",
        ),
        (
            refusal_report.report_kind == "local_runtime_service_refusal_executor_proof",
            "refusal_report.report_kind",
        ),
        (read_api_boundary_matrix.service_kind == view.service_kind, "matrix.service_kind"),
        (refusal_report.service_kind == view.service_kind, "refusal_report.service_kind"),
        (read_api_boundary_matrix.scenario_name == view.scenario_name, "matrix.scenario_name"),
        (refusal_report.scenario_name == view.scenario_name, "refusal_report.scenario_name"),
        (read_api_boundary_matrix.all_refused is True, "matrix.all_refused"),
        (refusal_report.executed_attempt_count == 0, "refusal_report.executed_attempt_count"),
        (
            refusal_report.all_refused_or_static_ack is True,
            "refusal_report.all_refused_or_static_ack",
        ),
    )
    _raise_first_failed(checks)
    for label, item in (
        ("view", view),
        ("read_api_boundary_matrix", read_api_boundary_matrix),
        ("refusal_report", refusal_report),
    ):
        _validate_safe_markers(label, item)
    for row in read_api_boundary_matrix.rows:
        _validate_safe_markers("read_api_boundary_matrix.rows", row)
        if row.refused is not True:
            raise PaperPreviewRuntimeServiceHistoryError("matrix.rows.refused")
    for attempt in refusal_report.attempts:
        _validate_safe_markers("refusal_report.attempts", attempt)
        if attempt.executed is not False:
            raise PaperPreviewRuntimeServiceHistoryError("refusal_report.attempts.executed")


def _validate_safe_markers(label: str, item: object) -> None:
    checks = (
        (getattr(item, "single_shot") is True, "single_shot"),
        (getattr(item, "runtime_loop_started") is False, "runtime_loop_started"),
        (getattr(item, "runtime_backed") is False, "runtime_backed"),
        (getattr(item, "ui_bound") is False, "ui_bound"),
        (getattr(item, "read_only") is True, "read_only"),
        (getattr(item, "paper_only") is True, "paper_only"),
        (
            getattr(item, "integration_gate_status", "blocked") == "blocked",
            "integration_gate_status",
        ),
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
    _raise_first_failed((passed, f"{label}.{field}") for passed, field in checks)


def _validate_entry(entry: PaperPreviewRuntimeServiceHistoryEntry) -> None:
    _validate_safe_markers("entry", entry)
    if entry.refusal_executed_attempt_count != 0:
        raise PaperPreviewRuntimeServiceHistoryError("entry.refusal_executed_attempt_count")


def _validate_report(report: PaperPreviewRuntimeServiceHistoryReport) -> None:
    if report.report_kind != "local_runtime_service_snapshot_history_contract":
        raise PaperPreviewRuntimeServiceHistoryError("report_kind")
    if report.history_kind != "bounded_in_memory_read_api_history":
        raise PaperPreviewRuntimeServiceHistoryError("history_kind")
    if report.max_entries <= 0:
        raise PaperPreviewRuntimeServiceHistoryError("max_entries")
    if report.entry_count != len(report.entries):
        raise PaperPreviewRuntimeServiceHistoryError("entry_count")
    _validate_safe_markers("report", report)
    for index, entry in enumerate(report.entries):
        _validate_entry(entry)
        if entry.entry_index != index:
            raise PaperPreviewRuntimeServiceHistoryError("entry_index")


def _reindex_entry(
    entry: PaperPreviewRuntimeServiceHistoryEntry, entry_index: int
) -> PaperPreviewRuntimeServiceHistoryEntry:
    return PaperPreviewRuntimeServiceHistoryEntry(
        **{**_entry_to_inputless_tuple(entry), "entry_index": entry_index}
    )


def _entry_to_inputless_tuple(entry: PaperPreviewRuntimeServiceHistoryEntry) -> dict[str, object]:
    return {field: getattr(entry, field) for field in entry.__dataclass_fields__}


def _raise_first_failed(checks) -> None:
    for passed, label in checks:
        if not passed:
            raise PaperPreviewRuntimeServiceHistoryError(label)


__all__ = [
    "PaperPreviewRuntimeServiceHistoryEntry",
    "PaperPreviewRuntimeServiceHistoryError",
    "PaperPreviewRuntimeServiceHistoryInput",
    "PaperPreviewRuntimeServiceHistoryReport",
    "append_paper_preview_runtime_service_history_entry",
    "build_paper_preview_runtime_service_history",
]

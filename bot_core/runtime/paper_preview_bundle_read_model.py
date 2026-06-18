"""Read-only local preview bundle read model for future UI/runtime contracts.

This module only summarizes an already-built ``PaperPreviewLocalDecisionBundle``
and its fail-closed boundary matrix. It never writes files, serializes payloads,
opens network/cloud sinks, binds QML/PySide objects, creates decisions/orders,
or starts runtime loops.
"""

from __future__ import annotations

from dataclasses import dataclass

from bot_core.runtime.paper_preview_bundle_boundary import (
    PaperPreviewBundleBoundaryMatrixReport,
)
from bot_core.runtime.paper_preview_scenario import PaperPreviewLocalDecisionBundle


class PaperPreviewBundleReadModelError(ValueError):
    """Raised when bundle read-model inputs fail closed."""


@dataclass(frozen=True, slots=True)
class PaperPreviewBoundaryReadRow:
    """Read-only boundary row mirrored from the local boundary matrix."""

    boundary_kind: str
    refused: bool
    reason: str


@dataclass(frozen=True, slots=True)
class PaperPreviewBundleReadModelStatus:
    """Immutable flags for the local/static read-model boundary."""

    read_only: bool = True
    runtime_backed: bool = False
    ui_bound: bool = False
    export_sink: str = "none"
    cloud_sink: str = "none"
    external_export: bool = False


@dataclass(frozen=True, slots=True)
class PaperPreviewBundleReadModel:
    """Immutable local DTO/view model for future UI/runtime integration.

    The model mirrors already-built local evidence only. It is not a QML/PySide
    binding, runtime loop, export payload, strategy/AI decision, score,
    recommendation, account snapshot, credential container, or order intent.
    """

    scenario_name: str
    bundle_kind: str
    decision_status: str
    has_market_context: bool
    market_symbols: tuple[str, ...]
    trade_count: int
    position_count: int
    audit_event_count: int
    boundary_row_count: int
    all_boundaries_refused: bool
    boundary_kinds: tuple[str, ...]
    boundary_rows: tuple[PaperPreviewBoundaryReadRow, ...]
    quote_count: int = 0
    candle_set_count: int = 0
    order_event_count: int = 0
    terminal_order_count: int = 0
    realized_pnl_total: float = 0.0
    risk_source: str = "placeholder"
    risk_checks_enabled: bool = False
    paper_symbols: tuple[str, ...] = ()
    blocked_engine_integrations: tuple[str, ...] = ()
    has_decision_context: bool = True
    has_dry_run_artifact: bool = True
    has_audit_trail: bool = True
    has_boundary_matrix: bool = True
    model_kind: str = "local_preview_bundle_read_model"
    generated_order_count: int = 0
    generated_decision_count: int = 0
    export_sink: str = "none"
    cloud_sink: str = "none"
    external_export: bool = False
    runtime_backed: bool = False
    ui_bound: bool = False
    read_only: bool = True


def build_paper_preview_bundle_read_model(
    bundle: PaperPreviewLocalDecisionBundle,
    matrix: PaperPreviewBundleBoundaryMatrixReport,
) -> PaperPreviewBundleReadModel:
    """Build a deterministic local/static read model from existing evidence."""

    _validate_consistency(bundle, matrix)
    boundary_rows = tuple(
        PaperPreviewBoundaryReadRow(
            boundary_kind=row.boundary_kind,
            refused=row.refused,
            reason=row.reason,
        )
        for row in matrix.rows
    )
    return PaperPreviewBundleReadModel(
        scenario_name=bundle.scenario_name,
        bundle_kind=bundle.bundle_kind,
        decision_status=bundle.decision_status,
        has_market_context=bundle.has_market_context,
        market_symbols=tuple(bundle.market_symbols),
        trade_count=bundle.trade_count,
        position_count=bundle.position_count,
        audit_event_count=bundle.audit_event_count,
        generated_order_count=bundle.generated_order_count,
        generated_decision_count=bundle.generated_decision_count,
        boundary_row_count=matrix.row_count,
        all_boundaries_refused=matrix.all_refused,
        boundary_kinds=tuple(row.boundary_kind for row in matrix.rows),
        boundary_rows=boundary_rows,
        quote_count=bundle.quote_count,
        candle_set_count=bundle.candle_set_count,
        order_event_count=bundle.order_event_count,
        terminal_order_count=bundle.terminal_order_count,
        realized_pnl_total=bundle.realized_pnl_total,
        risk_source=bundle.risk_source,
        risk_checks_enabled=bundle.risk_checks_enabled,
        paper_symbols=tuple(bundle.paper_symbols),
        blocked_engine_integrations=tuple(bundle.blocked_engine_integrations),
        export_sink=bundle.export_sink,
        cloud_sink=bundle.cloud_sink,
        external_export=bundle.external_export,
    )


def _validate_consistency(
    bundle: PaperPreviewLocalDecisionBundle,
    matrix: PaperPreviewBundleBoundaryMatrixReport,
) -> None:
    if matrix.scenario_name != bundle.scenario_name:
        raise PaperPreviewBundleReadModelError("boundary matrix scenario_name mismatch")
    if matrix.bundle_kind != bundle.bundle_kind:
        raise PaperPreviewBundleReadModelError("boundary matrix bundle_kind mismatch")
    if matrix.all_refused is not True:
        raise PaperPreviewBundleReadModelError("boundary matrix must refuse all boundaries")
    if matrix.row_count != len(matrix.rows):
        raise PaperPreviewBundleReadModelError("boundary matrix row_count mismatch")


__all__ = [
    "PaperPreviewBoundaryReadRow",
    "PaperPreviewBundleReadModel",
    "PaperPreviewBundleReadModelError",
    "PaperPreviewBundleReadModelStatus",
    "build_paper_preview_bundle_read_model",
]

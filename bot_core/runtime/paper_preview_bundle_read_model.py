"""Read-only local preview bundle read model for future UI/runtime contracts.

This module only summarizes an already-built ``PaperPreviewLocalDecisionBundle``
and its fail-closed boundary matrix. It never writes files, serializes payloads,
opens network/cloud sinks, binds QML/PySide objects, creates decisions/orders,
or starts runtime loops.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from bot_core.runtime.paper_preview_bundle_boundary import (
    PaperPreviewBundleBoundaryMatrixReport,
)
from bot_core.runtime.paper_preview_scenario import PaperPreviewLocalDecisionBundle


class PaperPreviewReadModelBoundary(StrEnum):
    """Forbidden read-model boundary kinds that always fail closed locally."""

    QML_BINDING = "qml_binding"
    PYSIDE_BRIDGE = "pyside_bridge"
    UI_RUNTIME_BINDING = "ui_runtime_binding"
    APP_RUNTIME_LOOP = "app_runtime_loop"
    CONTROLLER_HANDOFF = "controller_handoff"
    TRADING_CONTROLLER_HANDOFF = "trading_controller_handoff"
    DECISION_ENVELOPE_HANDOFF = "decision_envelope_handoff"
    STRATEGY_ENGINE_HANDOFF = "strategy_engine_handoff"
    AI_MODEL_INFERENCE_HANDOFF = "ai_model_inference_handoff"
    SCORING_HANDOFF = "scoring_handoff"
    RECOMMENDATION_HANDOFF = "recommendation_handoff"
    ORDER_GENERATION_HANDOFF = "order_generation_handoff"
    FILE_EXPORT = "file_export"
    SERIALIZED_EXPORT = "serialized_export"
    CLOUD_SINK = "cloud_sink"
    EXTERNAL_EXPORT = "external_export"


@dataclass(frozen=True, slots=True)
class PaperPreviewReadModelBoundaryRefusal:
    """Immutable no-action marker for a refused read-model boundary."""

    boundary_kind: str
    reason: str
    model_kind: str
    scenario_name: str
    refused: bool = True
    read_only: bool = True
    runtime_backed: bool = False
    ui_bound: bool = False
    export_sink: str = "none"
    cloud_sink: str = "none"
    external_export: bool = False
    generated_order_count: int = 0
    generated_decision_count: int = 0


@dataclass(frozen=True, slots=True)
class PaperPreviewReadModelBoundaryMatrixRow:
    """One immutable local diagnostic row for a refused read-model boundary."""

    boundary_kind: str
    refused: bool
    reason: str
    model_kind: str
    scenario_name: str
    read_only: bool = True
    runtime_backed: bool = False
    ui_bound: bool = False
    export_sink: str = "none"
    cloud_sink: str = "none"
    external_export: bool = False
    generated_order_count: int = 0
    generated_decision_count: int = 0


@dataclass(frozen=True, slots=True)
class PaperPreviewReadModelBoundaryMatrixReport:
    """Immutable report proving local read-model boundaries are refused."""

    model_kind: str
    scenario_name: str
    row_count: int
    rows: tuple[PaperPreviewReadModelBoundaryMatrixRow, ...]
    report_kind: str = "local_read_model_boundary_refusal_matrix"
    all_refused: bool = True
    read_only: bool = True
    runtime_backed: bool = False
    ui_bound: bool = False
    export_sink: str = "none"
    cloud_sink: str = "none"
    external_export: bool = False
    generated_order_count: int = 0
    generated_decision_count: int = 0


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


def build_paper_preview_read_model_boundary_matrix(
    read_model: PaperPreviewBundleReadModel,
) -> PaperPreviewReadModelBoundaryMatrixReport:
    """Build deterministic in-memory refusal matrix for every read-model boundary."""

    rows = tuple(
        _read_model_matrix_row_from_refusal(
            PaperPreviewReadModelBoundaryRefusal(
                boundary_kind=boundary.value,
                reason=f"local read model boundary refuses {boundary.value}",
                model_kind=read_model.model_kind,
                scenario_name=read_model.scenario_name,
                read_only=read_model.read_only,
                runtime_backed=read_model.runtime_backed,
                ui_bound=read_model.ui_bound,
                export_sink=read_model.export_sink,
                cloud_sink=read_model.cloud_sink,
                external_export=read_model.external_export,
                generated_order_count=read_model.generated_order_count,
                generated_decision_count=read_model.generated_decision_count,
            )
        )
        for boundary in PaperPreviewReadModelBoundary
    )
    return PaperPreviewReadModelBoundaryMatrixReport(
        model_kind=read_model.model_kind,
        scenario_name=read_model.scenario_name,
        row_count=len(rows),
        rows=rows,
        all_refused=all(row.refused for row in rows),
        read_only=read_model.read_only,
        runtime_backed=read_model.runtime_backed,
        ui_bound=read_model.ui_bound,
        export_sink=read_model.export_sink,
        cloud_sink=read_model.cloud_sink,
        external_export=read_model.external_export,
        generated_order_count=read_model.generated_order_count,
        generated_decision_count=read_model.generated_decision_count,
    )


def _read_model_matrix_row_from_refusal(
    refusal: PaperPreviewReadModelBoundaryRefusal,
) -> PaperPreviewReadModelBoundaryMatrixRow:
    return PaperPreviewReadModelBoundaryMatrixRow(
        boundary_kind=refusal.boundary_kind,
        refused=refusal.refused,
        reason=refusal.reason,
        model_kind=refusal.model_kind,
        scenario_name=refusal.scenario_name,
        read_only=refusal.read_only,
        runtime_backed=refusal.runtime_backed,
        ui_bound=refusal.ui_bound,
        export_sink=refusal.export_sink,
        cloud_sink=refusal.cloud_sink,
        external_export=refusal.external_export,
        generated_order_count=refusal.generated_order_count,
        generated_decision_count=refusal.generated_decision_count,
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
    "PaperPreviewReadModelBoundary",
    "PaperPreviewReadModelBoundaryMatrixReport",
    "PaperPreviewReadModelBoundaryMatrixRow",
    "PaperPreviewReadModelBoundaryRefusal",
    "PaperPreviewBundleReadModel",
    "PaperPreviewBundleReadModelError",
    "PaperPreviewBundleReadModelStatus",
    "build_paper_preview_bundle_read_model",
    "build_paper_preview_read_model_boundary_matrix",
]

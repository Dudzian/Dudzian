"""Fail-closed local bundle boundary for paper preview diagnostics.

This module only guards the in-memory ``PaperPreviewLocalDecisionBundle``
contract. It never exports, serializes, writes files, opens network/cloud sinks,
creates engine handoffs, generates decisions/orders, or starts runtime loops.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import NoReturn

from bot_core.runtime.paper_preview_scenario import PaperPreviewLocalDecisionBundle


class PaperPreviewBundleBoundaryError(ValueError):
    """Raised when a local paper preview bundle crosses a forbidden boundary."""


class PaperPreviewBundleBoundary(StrEnum):
    """Forbidden local-bundle boundary kinds that always fail closed."""

    FILE_EXPORT = "file_export"
    SERIALIZED_EXPORT = "serialized_export"
    CLOUD_SINK = "cloud_sink"
    EXTERNAL_EXPORT = "external_export"
    STRATEGY_ENGINE_HANDOFF = "strategy_engine_handoff"
    AI_MODEL_INFERENCE_HANDOFF = "ai_model_inference_handoff"
    SCORING_HANDOFF = "scoring_handoff"
    RECOMMENDATION_HANDOFF = "recommendation_handoff"
    DECISION_ENVELOPE_HANDOFF = "decision_envelope_handoff"
    TRADING_CONTROLLER_HANDOFF = "trading_controller_handoff"
    ORDER_GENERATION_HANDOFF = "order_generation_handoff"


@dataclass(frozen=True, slots=True)
class PaperPreviewBundleRefusal:
    """Immutable no-action/no-export marker for a refused bundle boundary."""

    boundary_kind: str
    reason: str
    bundle_kind: str
    scenario_name: str
    refused: bool = True
    export_sink: str = "none"
    cloud_sink: str = "none"
    external_export: bool = False
    generated_order_count: int = 0
    generated_decision_count: int = 0


@dataclass(frozen=True, slots=True)
class PaperPreviewBundleBoundaryMatrixRow:
    """One immutable local diagnostic row for a refused bundle boundary."""

    boundary_kind: str
    refused: bool
    reason: str
    bundle_kind: str
    scenario_name: str
    export_sink: str = "none"
    cloud_sink: str = "none"
    external_export: bool = False
    generated_order_count: int = 0
    generated_decision_count: int = 0


@dataclass(frozen=True, slots=True)
class PaperPreviewBundleBoundaryMatrixReport:
    """Immutable in-memory report proving local bundle boundaries are refused."""

    bundle_kind: str
    scenario_name: str
    row_count: int
    rows: tuple[PaperPreviewBundleBoundaryMatrixRow, ...]
    report_kind: str = "local_bundle_boundary_refusal_matrix"
    all_refused: bool = True
    export_sink: str = "none"
    cloud_sink: str = "none"
    external_export: bool = False
    generated_order_count: int = 0
    generated_decision_count: int = 0


_ALLOWED_BOUNDARIES = frozenset(item.value for item in PaperPreviewBundleBoundary)


def build_local_bundle_refusal(
    bundle: PaperPreviewLocalDecisionBundle,
    boundary_kind: str | PaperPreviewBundleBoundary,
) -> PaperPreviewBundleRefusal:
    """Return deterministic refusal evidence without crossing the boundary."""

    normalized = _normalize_boundary_kind(boundary_kind)
    return PaperPreviewBundleRefusal(
        boundary_kind=normalized,
        reason=f"local bundle boundary refuses {normalized}",
        bundle_kind=bundle.bundle_kind,
        scenario_name=bundle.scenario_name,
        generated_order_count=bundle.generated_order_count,
        generated_decision_count=bundle.generated_decision_count,
    )


def build_local_bundle_boundary_matrix(
    bundle: PaperPreviewLocalDecisionBundle,
) -> PaperPreviewBundleBoundaryMatrixReport:
    """Build deterministic in-memory refusal matrix for every local bundle boundary."""

    rows = tuple(
        _matrix_row_from_refusal(build_local_bundle_refusal(bundle, boundary))
        for boundary in PaperPreviewBundleBoundary
    )
    return PaperPreviewBundleBoundaryMatrixReport(
        bundle_kind=bundle.bundle_kind,
        scenario_name=bundle.scenario_name,
        row_count=len(rows),
        rows=rows,
        all_refused=all(row.refused for row in rows),
        generated_order_count=bundle.generated_order_count,
        generated_decision_count=bundle.generated_decision_count,
    )


def _matrix_row_from_refusal(
    refusal: PaperPreviewBundleRefusal,
) -> PaperPreviewBundleBoundaryMatrixRow:
    return PaperPreviewBundleBoundaryMatrixRow(
        boundary_kind=refusal.boundary_kind,
        refused=refusal.refused,
        reason=refusal.reason,
        bundle_kind=refusal.bundle_kind,
        scenario_name=refusal.scenario_name,
        export_sink=refusal.export_sink,
        cloud_sink=refusal.cloud_sink,
        external_export=refusal.external_export,
        generated_order_count=refusal.generated_order_count,
        generated_decision_count=refusal.generated_decision_count,
    )


def refuse_local_bundle_boundary(
    bundle: PaperPreviewLocalDecisionBundle,
    boundary_kind: str | PaperPreviewBundleBoundary,
) -> NoReturn:
    """Fail closed for every export/cloud/engine/order boundary attempt."""

    refusal = build_local_bundle_refusal(bundle, boundary_kind)
    raise PaperPreviewBundleBoundaryError(refusal.reason)


def _normalize_boundary_kind(boundary_kind: str | PaperPreviewBundleBoundary) -> str:
    normalized = str(boundary_kind.value if isinstance(boundary_kind, StrEnum) else boundary_kind)
    if normalized not in _ALLOWED_BOUNDARIES:
        raise PaperPreviewBundleBoundaryError(
            f"local bundle boundary refuses unknown boundary: {normalized}"
        )
    return normalized


__all__ = [
    "PaperPreviewBundleBoundary",
    "PaperPreviewBundleBoundaryError",
    "PaperPreviewBundleBoundaryMatrixReport",
    "PaperPreviewBundleBoundaryMatrixRow",
    "PaperPreviewBundleRefusal",
    "build_local_bundle_boundary_matrix",
    "build_local_bundle_refusal",
    "refuse_local_bundle_boundary",
]

"""Static local UI/runtime preflight audit for paper preview read models.

This module is only a deterministic diagnostic contract for future UI/runtime
integration. It does not bind QML/PySide, start runtime loops, serialize/export
payloads, create decisions/orders, call controllers, fetch accounts, read
secrets, or open network/cloud/file sinks.
"""

from __future__ import annotations

from dataclasses import dataclass

from bot_core.runtime.paper_preview_bundle_read_model import (
    PaperPreviewBundleReadModel,
    PaperPreviewReadModelBoundaryMatrixReport,
)


class PaperPreviewUiRuntimePreflightError(ValueError):
    """Raised when local/static UI-runtime preflight inputs fail closed."""


@dataclass(frozen=True, slots=True)
class PaperPreviewUiRuntimePreflightCheck:
    """One immutable preflight check for future UI/runtime integration."""

    check_name: str
    passed: bool
    required_for: str
    reason: str
    status: str
    blocking: bool


@dataclass(frozen=True, slots=True)
class PaperPreviewUiRuntimePreflightReport:
    """Immutable local/static audit of missing UI/runtime integration conditions."""

    scenario_name: str
    model_kind: str
    all_boundaries_refused: bool
    checks: tuple[PaperPreviewUiRuntimePreflightCheck, ...]
    check_count: int
    blocking_check_count: int
    report_kind: str = "local_preview_ui_runtime_preflight"
    read_only: bool = True
    runtime_backed: bool = False
    ui_bound: bool = False
    ready_for_ui_binding: bool = False
    ready_for_runtime_loop: bool = False
    ready_for_controller_handoff: bool = False
    ready_for_decision_engine: bool = False
    ready_for_export: bool = False
    export_sink: str = "none"
    cloud_sink: str = "none"
    external_export: bool = False
    generated_order_count: int = 0
    generated_decision_count: int = 0


def build_paper_preview_ui_runtime_preflight(
    read_model: PaperPreviewBundleReadModel,
    read_model_boundary_matrix: PaperPreviewReadModelBoundaryMatrixReport,
) -> PaperPreviewUiRuntimePreflightReport:
    """Build deterministic local/in-memory preflight from read-model evidence.

    Check order is intentionally code-order deterministic: positive local/static
    evidence first, then blocking missing-integration requirements.
    """

    _validate_preflight_inputs(read_model, read_model_boundary_matrix)
    checks = (
        _check(
            "local_read_model_present",
            True,
            "future_ui_runtime_contract",
            "local read model exists as static in-memory evidence",
            "present",
            False,
        ),
        _check(
            "read_model_is_read_only",
            True,
            "future_ui_runtime_contract",
            "read model is read-only and cannot mutate runtime state",
            "safe_local",
            False,
        ),
        _check(
            "read_model_not_runtime_backed",
            True,
            "future_runtime_loop",
            "read model is not backed by an app runtime loop",
            "static_local",
            False,
        ),
        _check(
            "read_model_not_ui_bound",
            True,
            "future_ui_binding",
            "read model is not bound to QML/PySide/UI objects",
            "static_local",
            False,
        ),
        _check(
            "read_model_boundary_matrix_present",
            True,
            "future_ui_runtime_contract",
            "read model boundary matrix exists as local refusal evidence",
            "present",
            False,
        ),
        _check(
            "all_read_model_boundaries_refused",
            True,
            "future_ui_runtime_contract",
            "read model boundary matrix refuses every forbidden boundary",
            "safe_local",
            False,
        ),
        *tuple(
            _check(name, False, required_for, reason, "missing", True)
            for name, required_for, reason in _MISSING_INTEGRATION_CHECKS
        ),
    )
    return PaperPreviewUiRuntimePreflightReport(
        scenario_name=read_model.scenario_name,
        model_kind=read_model.model_kind,
        read_only=read_model.read_only,
        runtime_backed=read_model.runtime_backed,
        ui_bound=read_model.ui_bound,
        all_boundaries_refused=read_model_boundary_matrix.all_refused,
        checks=checks,
        check_count=len(checks),
        blocking_check_count=sum(1 for check in checks if check.blocking),
        export_sink=read_model.export_sink,
        cloud_sink=read_model.cloud_sink,
        external_export=read_model.external_export,
        generated_order_count=read_model.generated_order_count,
        generated_decision_count=read_model.generated_decision_count,
    )


def _check(
    check_name: str,
    passed: bool,
    required_for: str,
    reason: str,
    status: str,
    blocking: bool,
) -> PaperPreviewUiRuntimePreflightCheck:
    return PaperPreviewUiRuntimePreflightCheck(
        check_name=check_name,
        passed=passed,
        required_for=required_for,
        reason=reason,
        status=status,
        blocking=blocking,
    )


def _validate_preflight_inputs(
    read_model: PaperPreviewBundleReadModel,
    matrix: PaperPreviewReadModelBoundaryMatrixReport,
) -> None:
    if matrix.scenario_name != read_model.scenario_name:
        raise PaperPreviewUiRuntimePreflightError(
            "read model boundary matrix scenario_name mismatch"
        )
    if matrix.model_kind != read_model.model_kind:
        raise PaperPreviewUiRuntimePreflightError("read model boundary matrix model_kind mismatch")
    if matrix.all_refused is not True:
        raise PaperPreviewUiRuntimePreflightError(
            "read model boundary matrix must refuse all boundaries"
        )
    if matrix.row_count != len(matrix.rows):
        raise PaperPreviewUiRuntimePreflightError("read model boundary matrix row_count mismatch")
    if read_model.read_only is not True:
        raise PaperPreviewUiRuntimePreflightError("read model must remain read-only")
    if read_model.runtime_backed is True:
        raise PaperPreviewUiRuntimePreflightError(
            "read model must remain static-local, not runtime-backed"
        )
    if read_model.ui_bound is True:
        raise PaperPreviewUiRuntimePreflightError("read model must remain unbound from UI")


_MISSING_INTEGRATION_CHECKS = (
    ("qml_binding_missing", "future_ui_binding", "QML binding is intentionally not implemented"),
    (
        "pyside_bridge_missing",
        "future_ui_binding",
        "PySide bridge is intentionally not implemented",
    ),
    (
        "app_runtime_loop_missing",
        "future_runtime_loop",
        "app runtime loop is intentionally not implemented",
    ),
    (
        "controller_handoff_missing",
        "future_controller_handoff",
        "controller handoff is intentionally not implemented",
    ),
    (
        "trading_controller_handoff_missing",
        "future_controller_handoff",
        "TradingController handoff is intentionally not implemented",
    ),
    (
        "decision_envelope_handoff_missing",
        "future_decision_engine",
        "DecisionEnvelope handoff is intentionally not implemented",
    ),
    (
        "strategy_engine_missing",
        "future_decision_engine",
        "strategy engine is intentionally not implemented",
    ),
    (
        "ai_model_inference_missing",
        "future_decision_engine",
        "AI/model inference is intentionally not implemented",
    ),
    ("scoring_missing", "future_decision_engine", "scoring is intentionally not implemented"),
    (
        "recommendation_missing",
        "future_decision_engine",
        "recommendation is intentionally not implemented",
    ),
    (
        "order_generation_missing",
        "future_order_generation",
        "order generation is intentionally not implemented",
    ),
    (
        "real_market_adapter_missing",
        "future_market_adapter",
        "real market adapter is intentionally not implemented",
    ),
    (
        "testnet_sandbox_adapter_missing",
        "future_testnet_sandbox",
        "testnet/sandbox adapter is intentionally not implemented",
    ),
    ("export_sink_missing", "future_export", "export sink is intentionally not implemented"),
    ("cloud_sink_missing", "future_export", "cloud sink is intentionally not implemented"),
    (
        "serialization_export_missing",
        "future_export",
        "serialization export is intentionally not implemented",
    ),
)


__all__ = [
    "PaperPreviewUiRuntimePreflightCheck",
    "PaperPreviewUiRuntimePreflightError",
    "PaperPreviewUiRuntimePreflightReport",
    "build_paper_preview_ui_runtime_preflight",
]

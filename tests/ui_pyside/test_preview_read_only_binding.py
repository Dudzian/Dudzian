from __future__ import annotations

import builtins
import socket
import threading
from dataclasses import FrozenInstanceError, replace

import pytest

from bot_core.runtime.paper_preview_runtime_service import run_paper_preview_runtime_service_once
from bot_core.runtime.paper_preview_runtime_service_boundary import (
    build_paper_preview_runtime_service_boundary_matrix,
)
from bot_core.runtime.paper_preview_runtime_service_closure import (
    EVIDENCE_STAGE_NAMES,
    PaperPreviewRuntimeServiceClosureReport,
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
from bot_core.runtime.paper_preview_scenario import PaperPreviewScenario, PaperPreviewScenarioStep
from ui.pyside_app.preview_read_only_binding import (
    PreviewReadOnlyBindingError,
    PreviewReadOnlyBindingSnapshot,
    build_preview_read_only_binding_snapshot,
)


FORBIDDEN_SURFACE_NAMES = {
    "start",
    "start_loop",
    "run_loop",
    "stop_loop",
    "schedule",
    "worker",
    "thread",
    "timer",
    "async_task",
    "execute",
    "submit",
    "cancel",
    "generate_order",
    "order_intent",
    "decision_envelope",
    "trading_controller",
    "recommend",
    "confidence",
    "serialize",
    "to_json",
    "to_yaml",
    "to_csv",
    "export_path",
    "file_path",
    "cloud_url",
    "credentials",
    "secret",
    "account_balance",
    "live_adapter",
    "testnet_adapter",
}


def _closure() -> PaperPreviewRuntimeServiceClosureReport:
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
    snapshot = run_paper_preview_runtime_service_once(scenario, created_at="fixed")
    boundary = build_paper_preview_runtime_service_boundary_matrix(snapshot)
    lifecycle = build_paper_preview_runtime_service_lifecycle_contract(snapshot, boundary)
    view = build_paper_preview_runtime_service_read_api(snapshot, boundary, lifecycle)
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
    return build_paper_preview_runtime_service_closure_report(
        snapshot, boundary, lifecycle, view, read_boundary, refusal, history
    )


def test_read_only_binding_exists_and_copies_block_identity() -> None:
    report = _closure()
    binding = build_preview_read_only_binding_snapshot(report)

    assert binding.binding_kind == "static_local_block_b_closure_ui_read_only_binding"
    assert binding.report_kind == "local_runtime_service_wrapper_block_b_closure_audit"
    assert binding.block_name == "BLOK B — LOCAL RUNTIME SERVICE WRAPPER"
    assert binding.block_status == "contract_complete_static_local"
    assert binding.next_block == "BLOK C — UI READ-ONLY BINDING"
    assert binding.ready_for_block_c is True
    assert binding.ready_for_ui_runtime_integration is False
    assert binding.integration_gate_status == "blocked"


def test_binding_exposes_closure_summary_and_safety_flags() -> None:
    report = _closure()
    binding = build_preview_read_only_binding_snapshot(report)

    assert binding.service_kind == report.service_kind
    assert binding.scenario_name == report.scenario_name
    assert binding.closure_score == report.closure_score
    assert binding.evidence_stage_count == len(EVIDENCE_STAGE_NAMES)
    assert binding.evidence_stage_names == EVIDENCE_STAGE_NAMES
    assert binding.checklist_passed_count == len(report.checklist_items)
    assert binding.checklist_total_count == len(report.checklist_items)
    assert binding.runtime_loop_started is False
    assert binding.runtime_backed is False
    assert binding.ui_bound is False
    assert binding.read_only is True
    assert binding.paper_only is True
    assert binding.generated_order_count == 0
    assert binding.generated_decision_count == 0
    assert binding.export_sink == "none"
    assert binding.cloud_sink == "none"
    assert binding.external_export is False


def test_binding_surface_is_read_only_and_has_no_action_or_unsafe_fields() -> None:
    binding = build_preview_read_only_binding_snapshot(_closure())
    surface_names = set(PreviewReadOnlyBindingSnapshot.__dataclass_fields__) | {
        name for name in dir(binding) if not name.startswith("__")
    }

    assert FORBIDDEN_SURFACE_NAMES.isdisjoint(surface_names)


@pytest.mark.parametrize(
    ("field_name", "unsafe_value"),
    [
        ("report_kind", "wrong"),
        ("block_status", "wrong"),
        ("ready_for_block_c", False),
        ("ready_for_ui_runtime_integration", True),
        ("ready_for_decision_engine", True),
        ("ready_for_export", True),
        ("ready_for_live", True),
        ("integration_gate_status", "open"),
        ("runtime_loop_started", True),
        ("runtime_backed", True),
        ("ui_bound", True),
        ("generated_order_count", 1),
        ("generated_decision_count", 1),
        ("export_sink", "file"),
        ("cloud_sink", "prod"),
        ("external_export", True),
    ],
)
def test_binding_fails_closed_on_unsafe_closure(field_name: str, unsafe_value: object) -> None:
    with pytest.raises(PreviewReadOnlyBindingError):
        build_preview_read_only_binding_snapshot(replace(_closure(), **{field_name: unsafe_value}))


def test_binding_build_has_no_file_network_thread_timer_side_effects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_open = builtins.open

    def guarded_open(file: object, mode: str = "r", *args: object, **kwargs: object):
        if any(flag in mode for flag in ("w", "a", "+", "x")):
            raise AssertionError("file writes must not be used")
        return original_open(file, mode, *args, **kwargs)

    def forbidden(*args: object, **kwargs: object):
        raise AssertionError("side effect must not be used")

    monkeypatch.setattr(builtins, "open", guarded_open)
    monkeypatch.setattr(socket, "socket", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(threading, "Thread", forbidden)
    monkeypatch.setattr(threading, "Timer", forbidden)

    binding = build_preview_read_only_binding_snapshot(_closure())

    assert binding.integration_gate_status == "blocked"
    assert binding.runtime_backed is False


def test_binding_is_deterministic_and_immutable() -> None:
    report = _closure()
    first = build_preview_read_only_binding_snapshot(report)
    second = build_preview_read_only_binding_snapshot(report)

    assert first == second
    assert isinstance(first.evidence_stage_names, tuple)
    with pytest.raises(FrozenInstanceError):
        first.block_status = "changed"  # type: ignore[misc]
    with pytest.raises(TypeError):
        first.evidence_stage_names[0] = "changed"  # type: ignore[index]

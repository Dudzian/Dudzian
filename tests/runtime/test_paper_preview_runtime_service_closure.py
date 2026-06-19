from __future__ import annotations

import builtins
import socket
import threading
from dataclasses import FrozenInstanceError, replace

import pytest

from bot_core.runtime.paper_preview_runtime_service import (
    PaperPreviewRuntimeService,
    run_paper_preview_runtime_service_once,
)
from bot_core.runtime.paper_preview_runtime_service_boundary import (
    build_paper_preview_runtime_service_boundary_matrix,
)
from bot_core.runtime.paper_preview_runtime_service_closure import (
    EVIDENCE_STAGE_NAMES,
    REQUIRED_CHECKLIST_ITEM_NAMES,
    PaperPreviewRuntimeServiceClosureChecklistItem,
    PaperPreviewRuntimeServiceClosureError,
    PaperPreviewRuntimeServiceClosureReport,
    build_paper_preview_runtime_service_closure_report,
)
from bot_core.runtime.paper_preview_runtime_service_history import (
    PaperPreviewRuntimeServiceHistoryEntry,
    PaperPreviewRuntimeServiceHistoryReport,
    build_paper_preview_runtime_service_history,
)
from bot_core.runtime.paper_preview_runtime_service_lifecycle import (
    PaperPreviewRuntimeServiceCommandDecision,
    PaperPreviewRuntimeServiceLifecycleContract,
    build_paper_preview_runtime_service_lifecycle_contract,
)
from bot_core.runtime.paper_preview_runtime_service_read_api import (
    PaperPreviewRuntimeServiceReadApiView,
    build_paper_preview_runtime_service_read_api,
)
from bot_core.runtime.paper_preview_runtime_service_read_api_boundary import (
    build_paper_preview_runtime_service_read_api_boundary_matrix,
)
from bot_core.runtime.paper_preview_runtime_service_refusal_executor import (
    PaperPreviewRuntimeServiceRefusalAttempt,
    PaperPreviewRuntimeServiceRefusalExecutorReport,
    PaperPreviewRuntimeServiceRefusalResult,
    build_paper_preview_runtime_service_refusal_executor_report,
)
from bot_core.runtime.paper_preview_scenario import PaperPreviewScenario, PaperPreviewScenarioStep
from bot_core.runtime.preview_modes import (
    PreviewMode,
    PreviewModeContractError,
    RuntimeCapability,
    build_preview_mode_policy,
)


def _scenario(name: str = "closure", symbol: str = "BTCUSDT") -> PaperPreviewScenario:
    return PaperPreviewScenario(
        name=name,
        steps=(
            PaperPreviewScenarioStep(
                action="submit", order_id=f"{name}-buy", symbol=symbol, side="buy", quantity=1
            ),
            PaperPreviewScenarioStep(action="fill", order_id=f"{name}-buy", fill_price=100),
        ),
    )


def _evidence(name: str = "closure", symbol: str = "BTCUSDT"):
    snapshot = run_paper_preview_runtime_service_once(_scenario(name, symbol), created_at="fixed")
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
    return snapshot, boundary, lifecycle, view, read_boundary, refusal, history


def _closure():
    return build_paper_preview_runtime_service_closure_report(*_evidence())


def test_closure_audit_exists_and_marks_block_b_contract_complete_static_local() -> None:
    report = _closure()

    assert report.report_kind == "local_runtime_service_wrapper_block_b_closure_audit"
    assert report.block_name == "BLOK B — LOCAL RUNTIME SERVICE WRAPPER"
    assert report.block_status == "contract_complete_static_local"
    assert report.next_block == "BLOK C — UI READ-ONLY BINDING"
    assert report.closure_score == 100
    assert report.ready_for_block_c is True


def test_closure_evidence_stages_are_exact_and_counted() -> None:
    report = _closure()

    assert report.evidence_stage_names == EVIDENCE_STAGE_NAMES
    assert report.evidence_stage_names == (
        "service_wrapper_snapshot",
        "service_boundary_no_loop_matrix",
        "lifecycle_command_contract",
        "local_snapshot_read_api",
        "read_api_boundary_no_export_matrix",
        "local_refusal_executor_proof",
        "service_snapshot_history_contract",
    )
    assert report.evidence_stage_count == len(report.evidence_stage_names)


def test_closure_checklist_is_complete_unique_and_passed() -> None:
    report = _closure()
    names = tuple(item.item_name for item in report.checklist_items)

    assert names == REQUIRED_CHECKLIST_ITEM_NAMES
    assert len(names) == len(set(names))
    assert all(item.passed for item in report.checklist_items)
    assert report.all_required_evidence_present is True
    assert report.all_safety_invariants_hold is True


def test_closure_keeps_runtime_ui_decision_export_live_not_ready() -> None:
    report = _closure()

    assert report.ready_for_ui_runtime_integration is False
    assert report.ready_for_decision_engine is False
    assert report.ready_for_export is False
    assert report.ready_for_live is False
    assert report.integration_gate_status == "blocked"
    assert report.runtime_loop_started is False
    assert report.runtime_backed is False
    assert report.ui_bound is False


def test_closure_mirrors_static_local_safety_flags() -> None:
    report = _closure()

    assert report.single_shot is True
    assert report.read_only is True
    assert report.paper_only is True
    assert report.generated_order_count == 0
    assert report.generated_decision_count == 0
    assert report.export_sink == "none"
    assert report.cloud_sink == "none"
    assert report.external_export is False


@pytest.mark.parametrize(
    "mutator",
    [
        lambda s, b, l, v, rb, rr, h: (replace(s, service_kind="wrong"), b, l, v, rb, rr, h),
        lambda s, b, l, v, rb, rr, h: (s, replace(b, report_kind="wrong"), l, v, rb, rr, h),
        lambda s, b, l, v, rb, rr, h: (s, b, replace(l, contract_kind="wrong"), v, rb, rr, h),
        lambda s, b, l, v, rb, rr, h: (s, b, l, replace(v, view_kind="wrong"), rb, rr, h),
        lambda s, b, l, v, rb, rr, h: (s, b, l, v, replace(rb, report_kind="wrong"), rr, h),
        lambda s, b, l, v, rb, rr, h: (s, b, l, v, rb, replace(rr, report_kind="wrong"), h),
        lambda s, b, l, v, rb, rr, h: (s, b, l, v, rb, rr, replace(h, report_kind="wrong")),
        lambda s, b, l, v, rb, rr, h: (s, replace(b, service_kind="wrong"), l, v, rb, rr, h),
        lambda s, b, l, v, rb, rr, h: (s, b, replace(l, scenario_name="wrong"), v, rb, rr, h),
        lambda s, b, l, v, rb, rr, h: (s, replace(b, all_refused=False), l, v, rb, rr, h),
        lambda s, b, l, v, rb, rr, h: (s, b, l, v, replace(rb, all_refused=False), rr, h),
        lambda s, b, l, v, rb, rr, h: (s, b, l, v, rb, replace(rr, executed_attempt_count=1), h),
        lambda s, b, l, v, rb, rr, h: (
            s,
            b,
            l,
            v,
            rb,
            replace(rr, all_refused_or_static_ack=False),
            h,
        ),
        lambda s, b, l, v, rb, rr, h: (s, b, l, v, rb, rr, replace(h, all_entries_safe=False)),
        lambda s, b, l, v, rb, rr, h: (
            s,
            b,
            l,
            v,
            rb,
            rr,
            replace(h, all_attempts_non_executed=False),
        ),
        lambda s, b, l, v, rb, rr, h: (s, b, l, v, rb, rr, replace(h, entry_count=0, entries=())),
        lambda s, b, l, v, rb, rr, h: (
            replace(s, integration_gate_status="ready"),
            b,
            l,
            v,
            rb,
            rr,
            h,
        ),
        lambda s, b, l, v, rb, rr, h: (replace(s, runtime_loop_started=True), b, l, v, rb, rr, h),
        lambda s, b, l, v, rb, rr, h: (s, replace(b, ui_bound=True), l, v, rb, rr, h),
        lambda s, b, l, v, rb, rr, h: (s, b, replace(l, runtime_backed=True), v, rb, rr, h),
        lambda s, b, l, v, rb, rr, h: (s, b, l, replace(v, generated_order_count=1), rb, rr, h),
        lambda s, b, l, v, rb, rr, h: (s, b, l, v, replace(rb, generated_decision_count=1), rr, h),
        lambda s, b, l, v, rb, rr, h: (s, b, l, v, rb, replace(rr, export_sink="file"), h),
        lambda s, b, l, v, rb, rr, h: (s, b, l, v, rb, rr, replace(h, cloud_sink="prod")),
        lambda s, b, l, v, rb, rr, h: (replace(s, external_export=True), b, l, v, rb, rr, h),
    ],
)
def test_closure_fail_closed(mutator) -> None:
    with pytest.raises(PaperPreviewRuntimeServiceClosureError):
        build_paper_preview_runtime_service_closure_report(*mutator(*_evidence()))


def test_closure_has_no_file_network_thread_timer_side_effects(
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

    report = _closure()

    forbidden_names = {
        "start",
        "start_loop",
        "run_loop",
        "stop_loop",
        "schedule",
        "worker",
        "thread",
        "timer",
        "async_task",
        "export_path",
        "file_path",
        "cloud_url",
        "serialized_payload",
        "json",
        "yaml",
        "csv",
        "qml_object",
        "qobject",
        "signal",
        "slot",
        "runtime_handle",
    }
    assert forbidden_names.isdisjoint(dir(report))
    assert forbidden_names.isdisjoint(dir(report.checklist_items[0]))


def test_closure_is_deterministic_and_immutable() -> None:
    evidence = _evidence("immutable")
    first = build_paper_preview_runtime_service_closure_report(*evidence)
    second = build_paper_preview_runtime_service_closure_report(*evidence)

    assert first == second
    with pytest.raises(FrozenInstanceError):
        first.closure_score = 1  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        first.checklist_items[0].passed = False  # type: ignore[misc]
    with pytest.raises(AttributeError):
        first.checklist_items.append("x")  # type: ignore[attr-defined]


def test_closure_does_not_affect_read_api_view_service_snapshot_or_paper_flow() -> None:
    evidence = _evidence("unchanged")
    snapshot, _, _, view, _, _, _ = evidence
    before = (
        view.order_event_count,
        view.trade_count,
        view.audit_event_count,
        view.blocking_items,
        snapshot.order_event_count,
        snapshot.trade_count,
        snapshot.audit_event_count,
        snapshot.scenario_result.summary.order_event_count,
        snapshot.scenario_result.summary.trade_count,
        snapshot.scenario_result.summary.audit_event_count,
    )

    build_paper_preview_runtime_service_closure_report(*evidence)

    after = (
        view.order_event_count,
        view.trade_count,
        view.audit_event_count,
        view.blocking_items,
        snapshot.order_event_count,
        snapshot.trade_count,
        snapshot.audit_event_count,
        snapshot.scenario_result.summary.order_event_count,
        snapshot.scenario_result.summary.trade_count,
        snapshot.scenario_result.summary.audit_event_count,
    )
    assert after == before


def test_closure_related_objects_have_no_forbidden_surfaces() -> None:
    snapshot, boundary, lifecycle, view, read_boundary, refusal, history = _evidence("surface")
    closure = build_paper_preview_runtime_service_closure_report(
        snapshot, boundary, lifecycle, view, read_boundary, refusal, history
    )
    objects = (
        closure.checklist_items[0],
        closure,
        history.entries[0],
        history,
        view,
        refusal.attempts[0],
        refusal,
        lifecycle,
        lifecycle.command_decisions[0],
        read_boundary.rows[0],
        snapshot,
        boundary.rows[0],
        PaperPreviewRuntimeService(created_at="fixed"),
    )
    forbidden = {
        "bind_qml",
        "bind_pyside",
        "attach_ui",
        "start_runtime",
        "run_loop",
        "connect_signal",
        "emit_signal",
        "create_controller",
        "serialize_for_ui",
        "qml",
        "qml_object",
        "QObject",
        "signal",
        "slot",
        "runtime_handle",
        "start",
        "start_loop",
        "stop_loop",
        "schedule",
        "worker",
        "thread",
        "timer",
        "async_task",
        "decide",
        "evaluate_strategy",
        "score",
        "recommend",
        "recommendation",
        "confidence",
        "order_intent",
        "execute",
        "infer",
        "predict",
        "serialize_for_engine",
        "to_json",
        "to_yaml",
        "to_csv",
        "get_balance",
        "get_account",
        "get_account_snapshot",
        "get_positions_from_exchange",
        "get_open_orders",
        "read_credentials",
        "account_balance",
        "metadata",
        "api_key",
        "secret",
        "password",
        "passphrase",
        "credential",
        "credentials",
        "token",
        "private_key",
        "export_path",
        "file_path",
        "cloud_url",
    }
    for obj in objects:
        assert forbidden.isdisjoint(dir(obj))


def test_closure_preview_policy_allows_preview_and_blocks_live_capabilities() -> None:
    read_only = build_preview_mode_policy(
        PreviewMode.READ_ONLY_MARKET, (RuntimeCapability.READ_ONLY_MARKET_FETCH,)
    )
    paper = build_preview_mode_policy(
        PreviewMode.PAPER,
        (RuntimeCapability.PAPER_ORDER_SUBMIT, RuntimeCapability.PAPER_ORDER_LIFECYCLE),
    )

    assert read_only.capabilities == (RuntimeCapability.READ_ONLY_MARKET_FETCH,)
    assert RuntimeCapability.PAPER_ORDER_SUBMIT in paper.capabilities
    assert RuntimeCapability.PAPER_ORDER_LIFECYCLE in paper.capabilities
    for capability in (
        RuntimeCapability.LIVE_ORDER_SUBMIT,
        RuntimeCapability.REAL_EXCHANGE_FILL,
        RuntimeCapability.LIVE_ACCOUNT_BALANCE_FETCH,
        RuntimeCapability.LIVE_ACCOUNT_SNAPSHOT_READ,
        RuntimeCapability.LIVE_CREDENTIALS_READ,
        RuntimeCapability.PRODUCTION_CLOUD_SINK,
        RuntimeCapability.EXTERNAL_EXPORT_SINK,
    ):
        with pytest.raises(PreviewModeContractError):
            build_preview_mode_policy(PreviewMode.PAPER, (capability,))


def test_closure_types_are_exported() -> None:
    assert PaperPreviewRuntimeServiceClosureChecklistItem.__name__
    assert PaperPreviewRuntimeServiceClosureReport.__name__
    assert PaperPreviewRuntimeServiceHistoryEntry.__name__
    assert PaperPreviewRuntimeServiceHistoryReport.__name__
    assert PaperPreviewRuntimeServiceReadApiView.__name__
    assert PaperPreviewRuntimeServiceRefusalResult.__name__
    assert PaperPreviewRuntimeServiceRefusalExecutorReport.__name__
    assert PaperPreviewRuntimeServiceLifecycleContract.__name__
    assert PaperPreviewRuntimeServiceCommandDecision.__name__

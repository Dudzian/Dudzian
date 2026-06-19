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
from bot_core.runtime.paper_preview_runtime_service_history import (
    PaperPreviewRuntimeServiceHistoryEntry,
    PaperPreviewRuntimeServiceHistoryError,
    PaperPreviewRuntimeServiceHistoryReport,
    append_paper_preview_runtime_service_history_entry,
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
from bot_core.runtime.preview_modes import (
    PreviewMode,
    PreviewModeContractError,
    RuntimeCapability,
    build_preview_mode_policy,
)


def _scenario(name: str, symbol: str = "BTCUSDT") -> PaperPreviewScenario:
    return PaperPreviewScenario(
        name=name,
        steps=(
            PaperPreviewScenarioStep(
                action="submit", order_id=f"{name}-buy", symbol=symbol, side="buy", quantity=1
            ),
            PaperPreviewScenarioStep(action="fill", order_id=f"{name}-buy", fill_price=100),
        ),
    )


def _evidence(name: str = "history", symbol: str = "BTCUSDT"):
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
            PaperPreviewRuntimeServiceRefusalAttempt("command", "read_local_snapshot"),
        ),
    )
    return snapshot, boundary, lifecycle, view, read_boundary, refusal


def test_service_snapshot_history_contract_exists_and_mirrors_inputs() -> None:
    snapshot, boundary, lifecycle, view, read_boundary, refusal = _evidence()

    report = build_paper_preview_runtime_service_history(
        ((view, read_boundary, refusal),), max_entries=3
    )

    assert boundary.report_kind == "local_runtime_service_boundary_no_loop_matrix"
    assert lifecycle.contract_kind == "local_runtime_service_lifecycle_command_contract"
    assert report.report_kind == "local_runtime_service_snapshot_history_contract"
    assert report.history_kind == "bounded_in_memory_read_api_history"
    assert report.entry_count == 1
    assert report.max_entries == 3
    assert isinstance(report.entries, tuple)
    entry = report.entries[0]
    assert entry.view_kind == view.view_kind
    assert entry.service_kind == view.service_kind == snapshot.service_kind
    assert entry.scenario_name == view.scenario_name == snapshot.scenario_name
    assert entry.integration_gate_status == view.integration_gate_status == "blocked"
    assert entry.boundary_matrix_report_kind == view.boundary_matrix_report_kind
    assert entry.read_api_boundary_matrix_report_kind == read_boundary.report_kind
    assert entry.refusal_executor_report_kind == refusal.report_kind
    assert entry.order_event_count == view.order_event_count == snapshot.order_event_count
    assert entry.trade_count == view.trade_count == snapshot.trade_count
    assert entry.audit_event_count == view.audit_event_count == snapshot.audit_event_count
    assert entry.position_count == view.position_count == snapshot.position_count
    assert entry.blocking_check_count == view.blocking_check_count
    assert entry.blocking_items == view.blocking_items
    assert entry.market_symbols == view.market_symbols
    assert entry.refusal_attempt_count == refusal.attempt_count
    assert entry.refusal_executed_attempt_count == 0


def test_history_report_mirrors_safety_flags() -> None:
    _, _, _, view, read_boundary, refusal = _evidence()
    report = build_paper_preview_runtime_service_history(
        ((view, read_boundary, refusal),), max_entries=1
    )
    entry = report.entries[0]

    for item in (entry, report):
        assert item.single_shot is True
        assert item.runtime_loop_started is False
        assert item.runtime_backed is False
        assert item.ui_bound is False
        assert item.read_only is True
        assert item.paper_only is True
        assert item.ready_for_ui_runtime_integration is False
        assert item.ready_for_decision_engine is False
        assert item.ready_for_export is False
        assert item.generated_order_count == 0
        assert item.generated_decision_count == 0
        assert item.export_sink == "none"
        assert item.cloud_sink == "none"
        assert item.external_export is False
    assert report.all_entries_safe is True
    assert report.all_attempts_non_executed is True


def test_bounded_history_keeps_latest_entries_and_reindexes() -> None:
    items = tuple(
        _evidence(f"history-{index}", symbol)[3:]
        for index, symbol in enumerate(("BTCUSDT", "ETHUSDT", "SOLUSDT"))
    )

    report = build_paper_preview_runtime_service_history(items, max_entries=2)

    assert report.truncated is True
    assert report.entry_count == 2
    assert tuple(entry.entry_index for entry in report.entries) == (0, 1)
    assert report.scenario_names == ("history-1", "history-2")


def test_append_helper_is_immutable_bounded_and_deterministic() -> None:
    first = _evidence("append-1")[3:]
    second = _evidence("append-2")[3:]
    third = _evidence("append-3")[3:]
    report = build_paper_preview_runtime_service_history((first,), max_entries=2)

    appended = append_paper_preview_runtime_service_history_entry(report, second)
    appended_again = append_paper_preview_runtime_service_history_entry(report, second)
    truncated = append_paper_preview_runtime_service_history_entry(appended, third)

    assert report.entry_count == 1
    assert appended == appended_again
    assert appended.entry_count == 2
    assert appended.truncated is False
    assert truncated.entry_count == 2
    assert truncated.truncated is True
    assert truncated.scenario_names == ("append-2", "append-3")
    assert tuple(entry.entry_index for entry in truncated.entries) == (0, 1)


@pytest.mark.parametrize(
    "mutator",
    [
        lambda view, matrix, refusal: (replace(view, view_kind="wrong"), matrix, refusal),
        lambda view, matrix, refusal: (view, replace(matrix, report_kind="wrong"), refusal),
        lambda view, matrix, refusal: (view, matrix, replace(refusal, report_kind="wrong")),
        lambda view, matrix, refusal: (view, replace(matrix, service_kind="wrong"), refusal),
        lambda view, matrix, refusal: (view, replace(matrix, scenario_name="wrong"), refusal),
        lambda view, matrix, refusal: (view, replace(matrix, all_refused=False), refusal),
        lambda view, matrix, refusal: (view, matrix, replace(refusal, executed_attempt_count=1)),
        lambda view, matrix, refusal: (
            view,
            matrix,
            replace(refusal, all_refused_or_static_ack=False),
        ),
        lambda view, matrix, refusal: (replace(view, runtime_loop_started=True), matrix, refusal),
        lambda view, matrix, refusal: (view, replace(matrix, ui_bound=True), refusal),
        lambda view, matrix, refusal: (view, matrix, replace(refusal, runtime_backed=True)),
        lambda view, matrix, refusal: (replace(view, generated_order_count=1), matrix, refusal),
        lambda view, matrix, refusal: (replace(view, export_sink="file"), matrix, refusal),
        lambda view, matrix, refusal: (replace(view, cloud_sink="prod"), matrix, refusal),
        lambda view, matrix, refusal: (replace(view, external_export=True), matrix, refusal),
    ],
)
def test_history_fail_closed(mutator) -> None:
    _, _, _, view, read_boundary, refusal = _evidence()
    with pytest.raises(PaperPreviewRuntimeServiceHistoryError):
        build_paper_preview_runtime_service_history(
            (mutator(view, read_boundary, refusal),), max_entries=1
        )


def test_history_rejects_bad_limits_and_empty_input() -> None:
    _, _, _, view, read_boundary, refusal = _evidence()
    with pytest.raises(PaperPreviewRuntimeServiceHistoryError):
        build_paper_preview_runtime_service_history(
            ((view, read_boundary, refusal),), max_entries=0
        )
    with pytest.raises(PaperPreviewRuntimeServiceHistoryError):
        build_paper_preview_runtime_service_history((), max_entries=1)


def test_history_has_no_file_network_thread_timer_side_effects(
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
    first = _evidence("side-1")[3:]
    second = _evidence("side-2")[3:]

    report = build_paper_preview_runtime_service_history((first,), max_entries=2)
    append_paper_preview_runtime_service_history_entry(report, second)

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
    assert forbidden_names.isdisjoint(dir(report.entries[0]))


def test_history_is_deterministic_and_immutable() -> None:
    evidence = (_evidence("immutable")[3:],)
    first = build_paper_preview_runtime_service_history(evidence, max_entries=2)
    second = build_paper_preview_runtime_service_history(evidence, max_entries=2)

    assert first == second
    with pytest.raises(FrozenInstanceError):
        first.entry_count = 9  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        first.entries[0].entry_index = 9  # type: ignore[misc]
    with pytest.raises(AttributeError):
        first.entries.append("x")  # type: ignore[attr-defined]


def test_history_does_not_affect_view_snapshot_or_paper_flow() -> None:
    snapshot, _, _, view, read_boundary, refusal = _evidence("unchanged")
    before = (
        view.order_event_count,
        view.trade_count,
        view.audit_event_count,
        view.blocking_items,
        snapshot.scenario_result.summary.order_event_count,
        snapshot.scenario_result.summary.trade_count,
        snapshot.scenario_result.summary.audit_event_count,
    )

    report = build_paper_preview_runtime_service_history(
        ((view, read_boundary, refusal),), max_entries=2
    )
    append_paper_preview_runtime_service_history_entry(report, (view, read_boundary, refusal))

    after = (
        view.order_event_count,
        view.trade_count,
        view.audit_event_count,
        view.blocking_items,
        snapshot.scenario_result.summary.order_event_count,
        snapshot.scenario_result.summary.trade_count,
        snapshot.scenario_result.summary.audit_event_count,
    )
    assert after == before


def test_history_related_objects_have_no_forbidden_surfaces() -> None:
    snapshot, boundary, lifecycle, view, read_boundary, refusal = _evidence("surface")
    report = build_paper_preview_runtime_service_history(
        ((view, read_boundary, refusal),), max_entries=1
    )
    objects = (
        report.entries[0],
        report,
        view,
        refusal.attempts[0],
        refusal,
        lifecycle.command_decisions[0],
        read_boundary.rows[0],
        snapshot,
        boundary.rows[0],
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


def test_history_preview_policy_keeps_live_capabilities_blocked() -> None:
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


def test_history_types_are_exported() -> None:
    assert PaperPreviewRuntimeServiceHistoryEntry.__name__
    assert PaperPreviewRuntimeServiceHistoryReport.__name__

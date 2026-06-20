"""Tests for the BLOK D source-only action bridge provider preflight."""

from __future__ import annotations

import json
from dataclasses import is_dataclass
from types import MappingProxyType
from typing import Any

import pytest

from ui.pyside_app.preview_action_dispatch_audit import ACCEPTED_INTENT_NOT_EXECUTED
from ui.pyside_app.preview_action_dispatch_bridge_provider import (
    PROVIDER_KIND,
    PROVIDER_SCHEMA_VERSION,
    PROVIDER_STATUS_READY,
    PaperRuntimeActionDispatchBridgeProvider,
    build_paper_runtime_action_dispatch_bridge_provider_snapshot,
)
from ui.pyside_app.preview_action_dispatch_bridge_snapshot import (
    BRIDGE_SNAPSHOT_KIND,
    NO_SELECTION_STATUS,
    build_paper_runtime_action_dispatch_bridge_snapshot,
)
from ui.pyside_app.preview_action_dispatch_catalog import (
    build_paper_runtime_action_dispatch_catalog,
)
from ui.pyside_app.preview_action_dispatch_contract import ALLOWED_PAPER_RUNTIME_ACTIONS
from ui.pyside_app.preview_action_dispatch_selection import UNKNOWN_SELECTION_STATUS

SIMPLE_TYPES = (dict, list, str, bool, int, type(None))


def _assert_no_execution(snapshot: dict[str, Any]) -> None:
    assert snapshot["execution_allowed"] is False
    assert snapshot["execution_performed"] is False
    assert snapshot["provider_execution_allowed"] is False
    assert snapshot["provider_execution_performed"] is False
    assert snapshot["selected_result"]["execution_allowed"] is False
    assert snapshot["selected_result"]["execution_performed"] is False


def _assert_simple_types_only(value: object) -> None:
    assert isinstance(value, SIMPLE_TYPES)
    assert not is_dataclass(value)
    assert not isinstance(value, MappingProxyType)
    if isinstance(value, dict):
        for key, nested in value.items():
            assert isinstance(key, str)
            _assert_simple_types_only(nested)
    elif isinstance(value, list):
        for nested in value:
            _assert_simple_types_only(nested)


def test_default_provider_snapshot_no_selection_no_execution() -> None:
    provider = PaperRuntimeActionDispatchBridgeProvider()
    snapshot = provider.snapshot()

    assert snapshot["snapshot_kind"] == BRIDGE_SNAPSHOT_KIND
    assert snapshot["provider_schema_version"] == PROVIDER_SCHEMA_VERSION
    assert snapshot["provider_kind"] == PROVIDER_KIND
    assert snapshot["provider_status"] == PROVIDER_STATUS_READY
    assert snapshot["status"] == NO_SELECTION_STATUS
    assert snapshot["last_requested_action_or_control"] is None
    assert snapshot["last_result_status"] == NO_SELECTION_STATUS
    assert snapshot["selected_result"]["catalog_action_found"] is False
    _assert_no_execution(snapshot)


def test_provider_allowed_action_selection_is_accepted_not_executed() -> None:
    provider = PaperRuntimeActionDispatchBridgeProvider()
    snapshot = provider.preview_select_action("paper_runtime_start_requested")

    assert snapshot["last_requested_action_or_control"] == "paper_runtime_start_requested"
    assert snapshot["last_result_status"] == ACCEPTED_INTENT_NOT_EXECUTED
    assert snapshot["selected_result"]["resolved_action"] == "paper_runtime_start_requested"
    assert snapshot["selected_result"]["resolved_source_control"] == "paper-runtime-start"
    assert provider.snapshot() == snapshot
    _assert_no_execution(snapshot)


def test_provider_source_control_selection_maps_to_action() -> None:
    provider = PaperRuntimeActionDispatchBridgeProvider()
    first = provider.preview_select_source_control("paper-runtime-pause")
    second = provider.preview_select_source_control("paper-runtime-pause")

    assert first["selected_result"]["resolved_action"] == "paper_runtime_pause_requested"
    assert first["last_result_status"] == ACCEPTED_INTENT_NOT_EXECUTED
    assert first == second
    _assert_no_execution(first)


def test_provider_unknown_action_fails_closed_without_exception_leakage() -> None:
    snapshot = PaperRuntimeActionDispatchBridgeProvider().preview_select_action("unknown")

    assert snapshot["selected_result"]["catalog_action_found"] is False
    assert snapshot["last_result_status"] == UNKNOWN_SELECTION_STATUS
    assert snapshot["selected_result"]["boundary_checks"]["selection_fail_closed"] is True
    _assert_no_execution(snapshot)


@pytest.mark.parametrize(
    "requested",
    [
        "start_live_runtime",
        "start_testnet_runtime",
        "submit_order",
        "account_balance_fetch",
        "export_cloud_report",
        "read_secret_value",
    ],
)
def test_provider_rejected_live_testnet_order_account_export_secrets_fail_closed(
    requested: str,
) -> None:
    snapshot = PaperRuntimeActionDispatchBridgeProvider().preview_select_action(requested)

    assert snapshot["selected_result"]["catalog_action_found"] is False
    assert snapshot["last_result_status"] == UNKNOWN_SELECTION_STATUS
    assert snapshot["selected_result"]["refusal_reason"].startswith("blocked_")
    _assert_no_execution(snapshot)


@pytest.mark.parametrize("requested", [None, "", "   ", 123, object()])
def test_provider_invalid_input_fails_closed(requested: Any) -> None:
    snapshot = PaperRuntimeActionDispatchBridgeProvider().preview_select_action(requested)

    assert snapshot["selected_result"]["catalog_action_found"] is False
    assert snapshot["last_result_status"] in {NO_SELECTION_STATUS, UNKNOWN_SELECTION_STATUS}
    _assert_no_execution(snapshot)


def test_reset_provider_selection_returns_no_selection() -> None:
    provider = PaperRuntimeActionDispatchBridgeProvider()
    provider.preview_select_action("paper_runtime_stop_requested")
    snapshot = provider.reset_preview_selection()

    assert snapshot["status"] == NO_SELECTION_STATUS
    assert snapshot["last_requested_action_or_control"] is None
    assert snapshot == provider.snapshot()
    _assert_no_execution(snapshot)


def test_operator_confirmation_does_not_enable_execution() -> None:
    snapshot = PaperRuntimeActionDispatchBridgeProvider().preview_select_action(
        "paper_runtime_stop_requested",
        operator_confirmation=True,
        operator_note="ack",
    )

    assert snapshot["last_result_status"] == ACCEPTED_INTENT_NOT_EXECUTED
    _assert_no_execution(snapshot)


def test_provider_snapshot_is_json_serializable_and_qml_safe_simple_types_only() -> None:
    snapshot = PaperRuntimeActionDispatchBridgeProvider().preview_select_action(
        "paper_runtime_resume_requested"
    )

    _assert_simple_types_only(snapshot)
    assert "paper_runtime_resume_requested" in json.dumps(snapshot, sort_keys=True)


def test_provider_output_is_deterministic() -> None:
    first = build_paper_runtime_action_dispatch_bridge_provider_snapshot(
        "paper-runtime-snapshot-refresh", operator_confirmation=True, reason="same"
    )
    second = build_paper_runtime_action_dispatch_bridge_provider_snapshot(
        "paper-runtime-snapshot-refresh", operator_confirmation=True, reason="same"
    )

    assert first == second


def test_provider_returned_payload_is_copy_safe() -> None:
    provider = PaperRuntimeActionDispatchBridgeProvider()
    snapshot = provider.preview_select_action("paper-runtime-start")
    snapshot["actions"].clear()
    snapshot["selected_result"]["boundary_checks"]["execution_disabled"] = False
    snapshot["boundary_checks"]["execution_disabled"] = False

    reread = provider.snapshot()
    assert len(reread["actions"]) == len(ALLOWED_PAPER_RUNTIME_ACTIONS)
    assert reread["selected_result"]["boundary_checks"]["execution_disabled"] is True
    assert reread["boundary_checks"]["execution_disabled"] is True


def test_provider_does_not_mutate_global_catalog_or_bridge_snapshot() -> None:
    before_catalog = build_paper_runtime_action_dispatch_catalog()
    before_snapshot = build_paper_runtime_action_dispatch_bridge_snapshot()

    provider = PaperRuntimeActionDispatchBridgeProvider()
    payload = provider.preview_select_action("paper-runtime-start")
    payload["actions"][0]["action"] = "mutated"

    assert build_paper_runtime_action_dispatch_catalog() == before_catalog
    assert build_paper_runtime_action_dispatch_bridge_snapshot() == before_snapshot


def test_provider_reuses_bridge_snapshot_helper(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[object] = []

    def fake_bridge_snapshot(requested_action_or_control: object = None, **kwargs: object):
        calls.append((requested_action_or_control, kwargs))
        return build_paper_runtime_action_dispatch_bridge_snapshot(
            requested_action_or_control, **kwargs
        )

    monkeypatch.setattr(
        "ui.pyside_app.preview_action_dispatch_bridge_provider."
        "build_paper_runtime_action_dispatch_bridge_snapshot",
        fake_bridge_snapshot,
    )

    snapshot = build_paper_runtime_action_dispatch_bridge_provider_snapshot(
        "paper_runtime_start_requested"
    )

    assert calls
    assert snapshot["last_result_status"] == ACCEPTED_INTENT_NOT_EXECUTED

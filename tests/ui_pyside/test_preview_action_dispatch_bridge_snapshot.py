"""Tests for the BLOK D QML-safe action dispatch bridge snapshot."""

from __future__ import annotations

import json
from dataclasses import is_dataclass
from types import MappingProxyType
from typing import Any

from ui.pyside_app.preview_action_dispatch_audit import ACCEPTED_INTENT_NOT_EXECUTED
from ui.pyside_app.preview_action_dispatch_bridge_snapshot import (
    BRIDGE_SNAPSHOT_KIND,
    BRIDGE_SNAPSHOT_SCHEMA_VERSION,
    NO_SELECTION_STATUS,
    build_paper_runtime_action_dispatch_bridge_snapshot,
)
from ui.pyside_app.preview_action_dispatch_catalog import (
    build_paper_runtime_action_dispatch_catalog,
)
from ui.pyside_app.preview_action_dispatch_contract import ALLOWED_PAPER_RUNTIME_ACTIONS
from ui.pyside_app.preview_action_dispatch_selection import UNKNOWN_SELECTION_STATUS

SIMPLE_TYPES = (dict, list, str, bool, int, type(None))
FORBIDDEN_ACTION_TERMS = (
    "live",
    "testnet",
    "order",
    "account",
    "fetch",
    "export",
    "secret",
    "cloud",
)


def _assert_snapshot_no_execution(snapshot: dict[str, Any]) -> None:
    assert snapshot["execution_allowed"] is False
    assert snapshot["execution_performed"] is False
    assert snapshot["selected_result"]["execution_allowed"] is False
    assert snapshot["selected_result"]["execution_performed"] is False
    for action in snapshot["actions"]:
        assert action["execution_allowed"] is False
        assert action["execution_performed"] is False


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


def test_default_snapshot_no_selection_is_preview_only_fail_closed() -> None:
    snapshot = build_paper_runtime_action_dispatch_bridge_snapshot()

    assert snapshot["schema_version"] == BRIDGE_SNAPSHOT_SCHEMA_VERSION
    assert snapshot["snapshot_kind"] == BRIDGE_SNAPSHOT_KIND
    assert snapshot["runtime_mode"] == "paper"
    assert snapshot["paper_only"] is True
    assert snapshot["local_only"] is True
    assert snapshot["status"] == NO_SELECTION_STATUS
    assert "No paper runtime action selected" in snapshot["operator_message"]
    assert snapshot["selected_result"]["catalog_action_found"] is False
    assert snapshot["selected_result"]["result_status"] == NO_SELECTION_STATUS
    assert snapshot["selected_result"]["blocked_reason"] == "no_action_selected"
    assert snapshot["boundary_checks"]["bridge_snapshot_no_selection"] is True
    _assert_snapshot_no_execution(snapshot)


def test_allowed_action_snapshot_is_accepted_not_executed() -> None:
    snapshot = build_paper_runtime_action_dispatch_bridge_snapshot("paper_runtime_start_requested")

    selected = snapshot["selected_result"]
    assert selected["requested_action_or_control"] == "paper_runtime_start_requested"
    assert selected["resolved_action"] == "paper_runtime_start_requested"
    assert selected["resolved_source_control"] == "paper-runtime-start"
    assert selected["catalog_action_found"] is True
    assert selected["result_status"] == ACCEPTED_INTENT_NOT_EXECUTED
    assert snapshot["status"] == ACCEPTED_INTENT_NOT_EXECUTED
    _assert_snapshot_no_execution(snapshot)


def test_source_control_snapshot_maps_to_action_deterministically() -> None:
    first = build_paper_runtime_action_dispatch_bridge_snapshot(
        requested_action_or_control="paper-runtime-pause"
    )
    second = build_paper_runtime_action_dispatch_bridge_snapshot(
        source_control="paper-runtime-pause"
    )

    assert first["selected_result"]["resolved_action"] == "paper_runtime_pause_requested"
    assert second["selected_result"]["resolved_action"] == "paper_runtime_pause_requested"
    assert first["selected_result"]["catalog_action_found"] is True
    assert second["selected_result"]["catalog_action_found"] is True
    assert second == build_paper_runtime_action_dispatch_bridge_snapshot(
        source_control="paper-runtime-pause"
    )


def test_unknown_action_snapshot_fails_closed_without_exception_leakage() -> None:
    snapshot = build_paper_runtime_action_dispatch_bridge_snapshot("unexpected_action")

    selected = snapshot["selected_result"]
    assert selected["catalog_action_found"] is False
    assert selected["result_status"] == UNKNOWN_SELECTION_STATUS
    assert selected["refusal_reason"] == "blocked_unknown_action"
    assert selected["boundary_checks"]["selection_fail_closed"] is True
    _assert_snapshot_no_execution(snapshot)


def test_rejected_live_testnet_order_account_export_secrets_fail_closed() -> None:
    rejected = (
        "start_live_runtime",
        "start_testnet_runtime",
        "submit_order",
        "account_balance_fetch",
        "export_cloud_report",
        "read_secret_value",
    )

    for requested in rejected:
        snapshot = build_paper_runtime_action_dispatch_bridge_snapshot(requested)
        selected = snapshot["selected_result"]
        assert selected["catalog_action_found"] is False
        assert selected["result_status"] == UNKNOWN_SELECTION_STATUS
        assert selected["refusal_reason"].startswith("blocked_")
        _assert_snapshot_no_execution(snapshot)


def test_invalid_input_snapshot_fails_closed() -> None:
    for requested in (None, "", "   ", 123, object()):
        snapshot = build_paper_runtime_action_dispatch_bridge_snapshot(requested)
        selected = snapshot["selected_result"]
        assert selected["catalog_action_found"] is False
        assert selected["execution_allowed"] is False
        assert selected["execution_performed"] is False
        assert selected["result_status"] in {NO_SELECTION_STATUS, UNKNOWN_SELECTION_STATUS}


def test_actions_contain_exactly_all_allowed_actions_in_stable_order() -> None:
    snapshot = build_paper_runtime_action_dispatch_bridge_snapshot()

    assert snapshot["action_count"] == len(ALLOWED_PAPER_RUNTIME_ACTIONS)
    assert [action["action"] for action in snapshot["actions"]] == list(
        ALLOWED_PAPER_RUNTIME_ACTIONS
    )
    assert [action["action"] for action in snapshot["actions"]] == [
        action["action"]
        for action in build_paper_runtime_action_dispatch_bridge_snapshot("unexpected")["actions"]
    ]


def test_actions_exclude_rejected_actions_and_do_not_imply_live_execution() -> None:
    snapshot = build_paper_runtime_action_dispatch_bridge_snapshot()
    action_text = " ".join(
        " ".join((action["action"], action["source_control"])) for action in snapshot["actions"]
    ).lower()

    for term in FORBIDDEN_ACTION_TERMS:
        assert term not in action_text
    for action in snapshot["actions"]:
        label_description = f"{action['label']} {action['description']}".lower()
        assert "paper" in label_description
        assert "local" in label_description
        assert "intent" in label_description
        assert "live" not in label_description
        assert "real" not in label_description
        assert "order" not in label_description
        assert "submit" not in label_description


def test_operator_confirmation_is_audited_but_never_enables_execution() -> None:
    snapshot = build_paper_runtime_action_dispatch_bridge_snapshot(
        "paper_runtime_stop_requested",
        operator_confirmation=True,
        operator_note="acknowledged",
    )

    assert snapshot["selected_result"]["catalog_action_found"] is True
    assert snapshot["selected_result"]["result_status"] == ACCEPTED_INTENT_NOT_EXECUTED
    _assert_snapshot_no_execution(snapshot)


def test_snapshot_is_qml_safe_simple_types_only_and_json_serializable() -> None:
    snapshot = build_paper_runtime_action_dispatch_bridge_snapshot("paper_runtime_resume_requested")

    _assert_simple_types_only(snapshot)
    encoded = json.dumps(snapshot, sort_keys=True)
    assert "paper_runtime_resume_requested" in encoded


def test_snapshot_output_is_deterministic() -> None:
    first = build_paper_runtime_action_dispatch_bridge_snapshot(
        "paper-runtime-snapshot-refresh", operator_confirmation=True, reason="same"
    )
    second = build_paper_runtime_action_dispatch_bridge_snapshot(
        "paper-runtime-snapshot-refresh", operator_confirmation=True, reason="same"
    )

    assert first == second


def test_returned_payload_is_copy_safe_for_dict_list_and_nested_boundary_checks() -> None:
    snapshot = build_paper_runtime_action_dispatch_bridge_snapshot("paper-runtime-start")
    snapshot["actions"].clear()
    snapshot["selected_result"]["boundary_checks"]["execution_disabled"] = False
    snapshot["boundary_checks"]["execution_disabled"] = False

    reread = build_paper_runtime_action_dispatch_bridge_snapshot("paper-runtime-start")
    assert reread["action_count"] == len(ALLOWED_PAPER_RUNTIME_ACTIONS)
    assert len(reread["actions"]) == len(ALLOWED_PAPER_RUNTIME_ACTIONS)
    assert reread["selected_result"]["boundary_checks"]["execution_disabled"] is True
    assert reread["boundary_checks"]["execution_disabled"] is True


def test_snapshot_uses_injected_catalog_without_mutating_it() -> None:
    catalog = build_paper_runtime_action_dispatch_catalog()
    snapshot = build_paper_runtime_action_dispatch_bridge_snapshot(
        "paper_runtime_snapshot_refresh_requested", catalog=catalog
    )

    assert snapshot["action_count"] == catalog.action_count
    assert snapshot["selected_result"]["resolved_action"] == catalog.actions[-1].action
    assert tuple(item.action for item in catalog.actions) == ALLOWED_PAPER_RUNTIME_ACTIONS

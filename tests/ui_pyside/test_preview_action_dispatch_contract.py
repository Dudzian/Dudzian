from __future__ import annotations

from dataclasses import asdict

import pytest

from ui.pyside_app.preview_action_dispatch_contract import (
    ALLOWED_PAPER_RUNTIME_ACTIONS,
    DISPATCH_KIND,
    RUNTIME_MODE,
    SCHEMA_VERSION,
    PaperRuntimeActionDispatchEvidence,
    build_paper_runtime_action_dispatch_contract,
)

REQUIRED_FIELDS = {
    "schema_version",
    "dispatch_kind",
    "requested_action",
    "runtime_mode",
    "paper_only",
    "local_only",
    "execution_allowed",
    "execution_performed",
    "order_generation_allowed",
    "order_submission_allowed",
    "live_mode_allowed",
    "testnet_mode_allowed",
    "requires_operator_confirmation",
    "refusal_reason",
    "blocked_reason",
    "safe_to_bind_from_ui",
    "boundary_checks",
    "allowed_actions",
    "rejected_actions",
}


def _assert_no_execution(evidence: PaperRuntimeActionDispatchEvidence) -> None:
    assert evidence.execution_allowed is False
    assert evidence.execution_performed is False
    assert evidence.order_generation_allowed is False
    assert evidence.order_submission_allowed is False
    assert evidence.live_mode_allowed is False
    assert evidence.testnet_mode_allowed is False


def test_allowed_paper_action_request_is_valid_but_execution_disabled() -> None:
    evidence = build_paper_runtime_action_dispatch_contract("paper_runtime_start_requested")

    assert evidence.schema_version == SCHEMA_VERSION
    assert evidence.dispatch_kind == DISPATCH_KIND
    assert evidence.requested_action == "paper_runtime_start_requested"
    assert evidence.normalized_action == "paper_runtime_start_requested"
    assert evidence.runtime_mode == RUNTIME_MODE == "paper"
    assert evidence.paper_only is True
    assert evidence.local_only is True
    assert evidence.safe_to_bind_from_ui is True
    assert evidence.refusal_reason == ""
    assert evidence.blocked_reason == ""
    assert evidence.requires_operator_confirmation is True
    _assert_no_execution(evidence)


@pytest.mark.parametrize(
    ("action", "reason"),
    [
        ("paper_runtime_restart_requested", "blocked_unknown_action"),
        ("live_runtime_start_requested", "blocked_live_mode"),
        ("testnet_runtime_start_requested", "blocked_testnet_mode"),
        ("paper_order_generate_requested", "blocked_order_generation_submission"),
        ("paper_order_submit_requested", "blocked_order_generation_submission"),
        ("account_balance_fetch_requested", "blocked_account_balance_fetch"),
        ("portfolio_fetch_requested", "blocked_account_balance_fetch"),
        ("export_cloud_secrets_requested", "blocked_export_cloud_secrets"),
    ],
)
def test_rejected_action_categories_fail_closed(action: str, reason: str) -> None:
    evidence = build_paper_runtime_action_dispatch_contract(action)

    assert evidence.safe_to_bind_from_ui is False
    assert evidence.refusal_reason == reason
    assert evidence.blocked_reason == reason
    assert evidence.boundary_checks["fail_closed"] is True
    assert evidence.boundary_checks["allowlisted_action"] is False
    _assert_no_execution(evidence)


@pytest.mark.parametrize("action", ["", "   ", None, 123, object()])
def test_empty_none_and_non_string_actions_fail_closed_without_exception(action: object) -> None:
    evidence = build_paper_runtime_action_dispatch_contract(action)

    assert evidence.safe_to_bind_from_ui is False
    assert evidence.refusal_reason in {"invalid_action_empty", "invalid_action_non_string"}
    _assert_no_execution(evidence)


def test_evidence_fields_complete() -> None:
    evidence = build_paper_runtime_action_dispatch_contract("paper_runtime_stop_requested")

    assert REQUIRED_FIELDS <= set(asdict(evidence))
    assert set(evidence.boundary_checks) >= {
        "paper_only",
        "local_only",
        "execution_disabled",
        "execution_not_performed",
        "order_generation_disabled",
        "order_submission_disabled",
        "live_mode_blocked",
        "testnet_mode_blocked",
        "account_fetch_blocked",
        "secrets_blocked",
        "export_cloud_blocked",
        "qml_handler_absent",
        "runtime_loop_absent",
        "lifecycle_execution_absent",
        "allowlisted_action",
        "fail_closed",
    }


def test_stable_deterministic_output() -> None:
    first = build_paper_runtime_action_dispatch_contract("paper_runtime_pause_requested")
    second = build_paper_runtime_action_dispatch_contract("paper_runtime_pause_requested")

    assert first == second
    assert asdict(first) == asdict(second)


def test_allowed_actions_list_is_copy_safe_if_returned() -> None:
    evidence = build_paper_runtime_action_dispatch_contract("paper_runtime_resume_requested")

    copied = list(evidence.allowed_actions)
    copied.append("live_runtime_start_requested")

    reread = build_paper_runtime_action_dispatch_contract("paper_runtime_resume_requested")
    assert reread.allowed_actions == ALLOWED_PAPER_RUNTIME_ACTIONS
    assert "live_runtime_start_requested" not in reread.allowed_actions


def test_boundary_matrix_confirms_allowed_and_rejected_categories() -> None:
    allowed = {
        action: build_paper_runtime_action_dispatch_contract(action)
        for action in ALLOWED_PAPER_RUNTIME_ACTIONS
    }
    rejected = {
        "unknown": build_paper_runtime_action_dispatch_contract("unexpected_action"),
        "live": build_paper_runtime_action_dispatch_contract("live_mode_requested"),
        "testnet": build_paper_runtime_action_dispatch_contract("testnet_mode_requested"),
        "order": build_paper_runtime_action_dispatch_contract("order_submission_requested"),
        "account": build_paper_runtime_action_dispatch_contract("account_fetch_requested"),
        "export_cloud_secrets": build_paper_runtime_action_dispatch_contract(
            "cloud_secret_export_requested"
        ),
    }

    assert all(item.safe_to_bind_from_ui for item in allowed.values())
    assert all(item.paper_only and item.local_only for item in allowed.values())
    assert all(not item.execution_allowed for item in allowed.values())
    assert all(not item.safe_to_bind_from_ui for item in rejected.values())
    assert all(item.boundary_checks["fail_closed"] for item in rejected.values())
    assert all(not item.live_mode_allowed for item in (*allowed.values(), *rejected.values()))
    assert all(not item.testnet_mode_allowed for item in (*allowed.values(), *rejected.values()))
    assert all(
        not item.order_generation_allowed for item in (*allowed.values(), *rejected.values())
    )
    assert all(
        not item.order_submission_allowed for item in (*allowed.values(), *rejected.values())
    )
    assert all(item.rejected_actions["live_mode"] for item in rejected.values())
    assert all(item.rejected_actions["testnet_mode"] for item in rejected.values())
    assert all(item.rejected_actions["order_generation_submission"] for item in rejected.values())
    assert all(item.rejected_actions["account_balance_fetch"] for item in rejected.values())
    assert all(item.rejected_actions["export_cloud_secrets"] for item in rejected.values())

from __future__ import annotations

from dataclasses import FrozenInstanceError, asdict

import pytest

from ui.pyside_app.preview_action_dispatch_audit import (
    ACCEPTED_INTENT_NOT_EXECUTED,
    AUDIT_ENVELOPE_KIND,
    AUDIT_ENVELOPE_SCHEMA_VERSION,
    REJECTED_INTENT,
    PaperRuntimeActionDispatchAuditEnvelope,
    build_paper_runtime_action_dispatch_audit_envelope,
)
from ui.pyside_app.preview_action_dispatch_contract import (
    ALLOWED_PAPER_RUNTIME_ACTIONS,
    SCHEMA_VERSION as DISPATCH_CONTRACT_SCHEMA_VERSION,
    PaperRuntimeActionDispatchEvidence,
    build_paper_runtime_action_dispatch_contract,
)

REQUIRED_ENVELOPE_FIELDS = {
    "schema_version",
    "envelope_kind",
    "dispatch_contract_schema_version",
    "requested_action",
    "normalized_action",
    "source_panel",
    "source_control",
    "operator_confirmation",
    "operator_note",
    "classification",
    "safe_to_bind_from_ui",
    "execution_allowed",
    "execution_performed",
    "runtime_mode",
    "paper_only",
    "local_only",
    "blocked_reason",
    "refusal_reason",
    "audit_status",
    "audit_message",
    "boundary_checks",
    "dispatch_evidence",
}


def _assert_no_execution(envelope: PaperRuntimeActionDispatchAuditEnvelope) -> None:
    assert envelope.execution_allowed is False
    assert envelope.execution_performed is False
    assert envelope.dispatch_evidence.execution_allowed is False
    assert envelope.dispatch_evidence.execution_performed is False
    assert envelope.dispatch_evidence.order_generation_allowed is False
    assert envelope.dispatch_evidence.order_submission_allowed is False


def test_allowed_action_audit_envelope_is_accepted_intent_not_executed() -> None:
    envelope = build_paper_runtime_action_dispatch_audit_envelope("paper_runtime_start_requested")

    assert envelope.schema_version == AUDIT_ENVELOPE_SCHEMA_VERSION
    assert envelope.envelope_kind == AUDIT_ENVELOPE_KIND
    assert envelope.dispatch_contract_schema_version == DISPATCH_CONTRACT_SCHEMA_VERSION
    assert envelope.requested_action == "paper_runtime_start_requested"
    assert envelope.normalized_action == "paper_runtime_start_requested"
    assert envelope.audit_status == ACCEPTED_INTENT_NOT_EXECUTED
    assert envelope.classification == ACCEPTED_INTENT_NOT_EXECUTED
    assert envelope.safe_to_bind_from_ui is True
    assert envelope.paper_only is True
    assert envelope.local_only is True
    assert envelope.blocked_reason == ""
    assert envelope.refusal_reason == ""
    assert "not allowed" in envelope.audit_message
    assert "no action was executed" in envelope.audit_message
    _assert_no_execution(envelope)


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
        ("cloud_export_requested", "blocked_export_cloud_secrets"),
        ("secret_export_requested", "blocked_export_cloud_secrets"),
        ("api_key_read_requested", "blocked_export_cloud_secrets"),
    ],
)
def test_rejected_action_audit_envelope_fails_closed(action: str, reason: str) -> None:
    envelope = build_paper_runtime_action_dispatch_audit_envelope(action)

    assert envelope.audit_status == REJECTED_INTENT
    assert envelope.classification == REJECTED_INTENT
    assert envelope.safe_to_bind_from_ui is False
    assert envelope.blocked_reason == reason
    assert envelope.refusal_reason == reason
    assert envelope.boundary_checks["fail_closed"] is True
    assert envelope.boundary_checks["allowlisted_action"] is False
    assert "rejected fail-closed" in envelope.audit_message
    assert reason in envelope.audit_message
    _assert_no_execution(envelope)


@pytest.mark.parametrize("action", [None, 123, object(), "", "   "])
def test_invalid_action_envelope_fails_closed_without_exception(action: object) -> None:
    envelope = build_paper_runtime_action_dispatch_audit_envelope(action)

    assert envelope.audit_status == REJECTED_INTENT
    assert envelope.safe_to_bind_from_ui is False
    assert envelope.refusal_reason in {"invalid_action_empty", "invalid_action_non_string"}
    assert envelope.blocked_reason == envelope.refusal_reason
    _assert_no_execution(envelope)


def test_operator_confirmation_is_audited_but_never_enables_execution() -> None:
    envelope = build_paper_runtime_action_dispatch_audit_envelope(
        "paper_runtime_start_requested",
        operator_confirmation=True,
        operator_note="confirmed in preview",
    )

    assert envelope.operator_confirmation is True
    assert envelope.operator_note == "confirmed in preview"
    assert envelope.safe_to_bind_from_ui is True
    _assert_no_execution(envelope)


def test_source_panel_and_source_control_are_deterministic_text() -> None:
    envelope = build_paper_runtime_action_dispatch_audit_envelope(
        "paper_runtime_pause_requested",
        source_panel="  Paper Session  ",
        source_control=42,
        reason="  operator reason  ",
    )

    assert envelope.source_panel == "Paper Session"
    assert envelope.source_control == "42"
    assert envelope.operator_note == "operator reason"
    assert (
        build_paper_runtime_action_dispatch_audit_envelope(
            "paper_runtime_pause_requested",
            source_panel="  Paper Session  ",
            source_control=42,
            reason="  operator reason  ",
        )
        == envelope
    )


def test_deterministic_output_for_same_input() -> None:
    first = build_paper_runtime_action_dispatch_audit_envelope(
        "paper_runtime_resume_requested",
        source_panel="panel",
        source_control="control",
        operator_confirmation=True,
        operator_note="note",
    )
    second = build_paper_runtime_action_dispatch_audit_envelope(
        "paper_runtime_resume_requested",
        source_panel="panel",
        source_control="control",
        operator_confirmation=True,
        operator_note="note",
    )

    assert first == second
    assert asdict(first) == asdict(second)


def test_envelope_nested_structures_are_immutable_and_copy_safe() -> None:
    envelope = build_paper_runtime_action_dispatch_audit_envelope("unexpected_action")

    assert REQUIRED_ENVELOPE_FIELDS <= set(asdict(envelope))
    with pytest.raises(FrozenInstanceError):
        envelope.audit_status = "changed"  # type: ignore[misc]
    with pytest.raises(TypeError):
        envelope.boundary_checks["fail_closed"] = False  # type: ignore[index]
    with pytest.raises(TypeError):
        envelope.dispatch_evidence.boundary_checks["fail_closed"] = False  # type: ignore[index]

    reread = build_paper_runtime_action_dispatch_audit_envelope("unexpected_action")
    assert reread.boundary_checks["fail_closed"] is True
    assert reread.dispatch_evidence.boundary_checks["fail_closed"] is True


def test_envelope_reuses_explicit_dispatch_contract_evidence() -> None:
    evidence = build_paper_runtime_action_dispatch_contract("paper_runtime_stop_requested")
    envelope = build_paper_runtime_action_dispatch_audit_envelope(evidence)

    assert envelope.dispatch_evidence is evidence
    assert envelope.dispatch_contract_schema_version == evidence.schema_version
    assert envelope.boundary_checks == evidence.boundary_checks
    assert isinstance(envelope.dispatch_evidence, PaperRuntimeActionDispatchEvidence)
    _assert_no_execution(envelope)


def test_allowed_and_rejected_matrix_for_envelope() -> None:
    for action in ALLOWED_PAPER_RUNTIME_ACTIONS:
        envelope = build_paper_runtime_action_dispatch_audit_envelope(action)
        assert envelope.audit_status == ACCEPTED_INTENT_NOT_EXECUTED
        assert envelope.safe_to_bind_from_ui is True
        assert envelope.boundary_checks["allowlisted_action"] is True
        _assert_no_execution(envelope)

    rejected_actions = (
        "unknown_action_requested",
        "live_runtime_start_requested",
        "testnet_runtime_start_requested",
        "paper_order_submit_requested",
        "account_balance_fetch_requested",
        "secret_export_requested",
        None,
    )
    for action in rejected_actions:
        envelope = build_paper_runtime_action_dispatch_audit_envelope(action)
        assert envelope.audit_status == REJECTED_INTENT
        assert envelope.safe_to_bind_from_ui is False
        assert envelope.refusal_reason
        assert envelope.blocked_reason
        assert envelope.boundary_checks["fail_closed"] is True
        _assert_no_execution(envelope)

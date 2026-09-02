"""Permanent guards for restart-safe SecretHandoff cleanup architecture."""

from __future__ import annotations

import inspect
import json
from pathlib import Path
from typing import get_protocol_members

from bot_core.persistence.secret_handoff import SecretExternalResourcePort
from bot_core.persistence.secret_handoff_contract import SECRET_HANDOFF_FIELDS

ROOT = Path(__file__).parents[2]
CANONICAL = (
    ROOT / "docs/architecture/cryptohunter_product_architecture/"
    "persistence_versioning_migrations_backup_and_recovery.json"
)


def _handoff() -> dict[str, object]:
    return json.loads(CANONICAL.read_bytes())["external_resource_handoff"]


def test_cleanup_contract_preserves_frozen_lifecycle_and_payload_boundary() -> None:
    handoff = _handoff()
    states = [
        "PREPARED",
        "COMMITTED",
        "CLEANUP_PENDING",
        "UNKNOWN_RECONCILIATION",
    ]
    assert handoff["states"] == states
    assert handoff["durable_lifecycle"]["allowed_transitions"] == {
        "PREPARED": ["COMMITTED", "UNKNOWN_RECONCILIATION"],
        "COMMITTED": ["CLEANUP_PENDING"],
        "CLEANUP_PENDING": [],
        "UNKNOWN_RECONCILIATION": [],
    }
    assert handoff["secret_payload_fields"] == 0
    assert handoff["unknown_outcome"] == "RECONCILIATION_REQUIRED_NO_BLIND_RETRY"
    assert len(states) == 4
    assert "cleanup_idempotency_key" not in SECRET_HANDOFF_FIELDS


def test_cleanup_contract_is_exact_handoff_idempotent_and_at_least_once_safe() -> None:
    cleanup = _handoff()["cleanup_contract"]
    assert cleanup["semantic_operation"] == "IDEMPOTENT_CLEANUP_REQUEST"
    assert cleanup["delivery"] == "AT_LEAST_ONCE"
    assert cleanup["duplicate_policy"] == "EXACT_SAME_HANDOFF_SAFE"
    assert cleanup["idempotency_scope"] == "EXACT_IMMUTABLE_HANDOFF_IDENTITY"
    assert cleanup["identity_fields"] == [
        "handoff_id",
        "intrinsically_valid_SecretHandoffRecord",
        "operation_fingerprint_sha256",
        "metadata_fingerprint_sha256",
        "old_reference",
        "new_reference",
        "reconciliation_metadata",
    ]
    assert cleanup["same_handoff_id_conflict"] == "FAIL_CLOSED"
    assert cleanup["operation_fingerprint_only_equivalence"] == "FORBIDDEN"
    assert cleanup["observable_repeat_guarantee"] == (
        "N_GTE_1_CALLS_EXTERNALLY_EQUIVALENT_TO_ONE_ACCEPTED_CLEANUP_REQUEST"
    )


def test_cleanup_outcomes_keep_uncertainty_committed_and_success_pending_only() -> None:
    cleanup = _handoff()["cleanup_contract"]
    assert cleanup["success_meaning"] == "REQUEST_ACCEPTED_OR_ALREADY_SATISFIED"
    assert cleanup["durable_target_after_success"] == "CLEANUP_PENDING"
    assert cleanup["failure_or_ack_loss"] == "REMAIN_COMMITTED_AND_REDELIVER_SAFE"
    assert cleanup["cleanup_uncertainty_state"] == "NO_UNKNOWN_RECONCILIATION"
    assert cleanup["new_durable_lifecycle_state"] is False


def test_initial_mutation_and_cleanup_have_deliberately_distinct_retry_contracts() -> None:
    retry = _handoff()["cleanup_contract"]["retry_contract_separation"]
    assert retry == {
        "prepared_initial_mutation": "RECONCILIATION_REQUIRED_NO_BLIND_RETRY",
        "committed_cleanup_request": "IDEMPOTENT_AT_LEAST_ONCE_REDELIVERY_SAFE",
    }


def test_external_port_surface_and_cleanup_documentation_remain_narrow() -> None:
    assert get_protocol_members(SecretExternalResourcePort) == {
        "begin",
        "reconcile",
        "cleanup",
    }
    documentation = " ".join((inspect.getdoc(SecretExternalResourcePort.cleanup) or "").split())
    assert documentation
    assert "idempotent" in documentation
    assert "at-least-once-safe" in documentation
    assert "exact same immutable descriptor" in documentation

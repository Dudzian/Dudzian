"""Closed review-candidate A+C addendum; never a production authority switch."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import re
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[2]
CONTRACT_PATH = (
    ROOT / "docs/architecture/contracts/stage9_root_proof_transaction_scope_addendum_v1.json"
)

SOURCE_BLOB_PINS = {
    "issuer": "6fe01e7476648a3979537bddd1d33fa69567cc83",
    "substrate": "88888b836bf0bbe496028c3a55d4132963c7666b",
    "readiness": "873a5384de5c9592c141c55c38da96e35cc4bca8",
    "physical": "36f4c845034570ddba27fb778f3ca6a310c1a485",
}
SOURCE_FILENAMES = {
    "issuer": "m05_account_genesis_independent_root_proof_issuer_contract.json",
    "substrate": "m05_account_genesis_root_proof_issuer_production_substrate_selection_contract.json",
    "readiness": "m05_account_genesis_root_proof_issuer_implementation_readiness_contract.json",
    "physical": "m05_account_genesis_physical_persistence_crash_atomicity_contract.json",
}
EXPECTED_WRITE_SET = [
    "ENTITLEMENT_CAS_BIND_AND_REGISTRY_HISTORY",
    "EXACT_ORIGINAL_SIGNED_PAYLOAD_SIGNATURE_ENVELOPE_RETENTION_OR_IMMUTABLE_IN_DB_LINKAGE",
    "AUTHENTICATED_ISSUER_HISTORY_RECORD_AND_SIGNED_HEAD_WITH_EXACT_PREDECESSOR_CAS",
    "IMMUTABLE_DECISION_MARKER_WITH_EXACT_WINNER_ATTEMPT_OPERATION_AND_BYTE_IDENTICAL_LINKAGE",
]
EXCLUDED = [
    "CHA_ATTEMPT_STORE",
    "FRESHNESS_AUTHORITY",
    "CHECKPOINT_AUTHORITY",
    "ACCOUNT_GENESIS_ACCOUNT_COMMIT",
    "CREDENTIAL_LIFECYCLE_ADMIN",
    "CUSTODY_MUTATION",
]
FENCES = ["REQUESTER", "CLAIMANT", "ROOT_SIGNER", "HISTORY_SIGNER", "TRUST_ACTIVATION"]
POINTERS = {
    "issuer": {
        "/bind_authorization_order/7": "8_ATOMICALLY_BIND_AND_PERSIST_IDENTICAL_SIGNED_PROOF_AND_HISTORY",
        "/authority_model/semantic_authority_owner": "IndependentAccountGenesisRootProofIssuer",
        "/cross_authority_atomicity/model": (
            "NO_DISTRIBUTED_TRANSACTION; immutable exact-operation proof plus "
            "recoverable local phase machine and independent freshness CAS"
        ),
        "/lost_response/same_attempt_id_success": (
            "return byte-identical root_proof_id, signed payload, signature "
            "and retained history for that exact attempt"
        ),
        "/replay_and_consumption/consumption_owner": (
            "NONE; CHA does not call issuer after issuance to consume"
        ),
    },
    "substrate": {
        "/authority_separation/rule": "PHYSICAL COLOCATION != AUTHORITY REUSE",
        "/authority_separation/separate_semantic_owners_schemas_roles_credentials_namespaces_transactions_and_lifecycles": True,
        "/authority_separation/required_separation/4": "transaction ownership",
        "/authority_separation/CHA_direct_registry_write": False,
        "/authority_separation/issuer_direct_account_commit": False,
    },
    "readiness": {
        "/frozen_semantics_change_rule": "DO_NOT_REOPEN_WITHOUT_CONCRETE_EXPLOIT_OR_CONTRADICTION",
        "/status/semantic_contract_frozen": True,
        "/status/implementation_readiness_frozen": True,
        "/implementation_gate/readiness_freeze_does_not_allow_implementation": True,
        "/implementation_gate/non_production_implementation_mode": "NONE_FROZEN",
        "/status/implementation_allowed": False,
    },
    "physical": {
        "/authority_vs_storage_roles/transaction_coordinator": "CryptoHunterAccountAuthority",
        "/selected_or_blocked_protocol/logical_authority_atomicity_equals_single_physical_transaction": False,
        "/result/selected_protocol": (
            "INITIAL_BINDING -> PREPARED -> EXTERNAL_FRESHNESS_CAS -> LOCAL_FINAL_COMMIT"
        ),
        "/freshness_ownership_boundary/coordinator_may_unilaterally_advance_authoritative_generation": False,
    },
}


def _document() -> dict:
    return json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))


def _pointer(value: object, pointer: str) -> object:
    for part in pointer.split("/")[1:]:
        if isinstance(value, list):
            value = value[int(part)]
        else:
            assert isinstance(value, dict)
            value = value[part.replace("~1", "/").replace("~0", "~")]
    return value


def _assert_closed_safe_contract(document: dict) -> None:
    assert set(document) == {
        "artifact", "schema_version", "stage", "decision_provenance", "decision_approved",
        "review_state", "precedence", "transaction", "separate_authorities",
        "required_future_evidence", "production", "frozen_sources",
    }
    assert document["artifact"] == "S9_ROOT_PROOF_TRANSACTION_SCOPE_ADDENDUM_V1"
    assert type(document["schema_version"]) is int and document["schema_version"] == 1
    assert document["stage"] == "S9-RPI-08"
    assert document["decision_approved"] == "A+C"
    assert document["review_state"] == "SEPARATE_CONTRACT_REVIEW_PENDING"
    assert document["decision_provenance"] == (
        "docs/architecture/decisions/"
        "stage9_root_proof_transaction_authority_owner_approval_v1.md"
    )
    precedence = document["precedence"]
    assert set(precedence) == {
        "rule", "explains", "remains_normative",
        "unilateral_frozen_rewrite_allowed", "new_implementation_permission",
    }
    assert precedence["rule"] == "NARROW_SCOPE_CLARIFICATION_ONLY"
    assert "independent semantic authorities" in precedence["explains"]
    assert "all other fields and semantics" in precedence["remains_normative"]
    assert precedence["unilateral_frozen_rewrite_allowed"] is False
    assert precedence["new_implementation_permission"] is False

    transaction = document["transaction"]
    assert set(transaction) == {
        "owner", "coordinator", "commits", "required_atomic_write_set",
        "require_same_postgresql_backend_transaction",
        "require_active_credential_and_trust_fences_through_commit",
        "fenced_identities", "owner_owned_helpers_may_participate_without_semantic_ownership_transfer",
        "issuer_arbitrary_dml_ddl_or_admin_privileges", "issuer_direct_helper_execute_grants",
        "distributed_two_phase_commit", "partial_commit_or_compensation_permitted",
        "proof_can_be_reconstructed_from_digest_only", "signing_inside_final_transaction",
        "lost_response", "retry_resign_or_remint_permitted",
    }
    assert transaction["owner"] == "IndependentAccountGenesisRootProofIssuer"
    assert transaction["coordinator"] == "ISSUER_OWNED_ONE_PHYSICAL_POSTGRESQL16_SERIALIZABLE_TRANSACTION"
    assert type(transaction["commits"]) is int and transaction["commits"] == 1
    assert transaction["required_atomic_write_set"] == EXPECTED_WRITE_SET
    assert transaction["fenced_identities"] == FENCES
    for name in (
        "require_same_postgresql_backend_transaction",
        "require_active_credential_and_trust_fences_through_commit",
        "owner_owned_helpers_may_participate_without_semantic_ownership_transfer",
    ):
        assert transaction[name] is True
    for name in (
        "issuer_arbitrary_dml_ddl_or_admin_privileges",
        "issuer_direct_helper_execute_grants",
        "distributed_two_phase_commit",
        "partial_commit_or_compensation_permitted",
        "proof_can_be_reconstructed_from_digest_only",
        "signing_inside_final_transaction",
        "retry_resign_or_remint_permitted",
    ):
        assert transaction[name] is False
    assert transaction["lost_response"] == (
        "INDETERMINATE_UNTIL_AUTHORITATIVE_EXACT_MARKER_AND_HISTORY_RECOVERY"
    )
    separation = document["separate_authorities"]
    assert set(separation) == {
        "not_in_issuer_final_commit", "shared_semantic_owner_or_credentials_permitted",
        "cha_direct_issuer_registry_mutation", "issuer_direct_account_commit",
        "issuer_direct_attempt_store_write", "issuer_cannot_unilaterally_advance_freshness",
    }
    assert separation["not_in_issuer_final_commit"] == EXCLUDED
    for name in (
        "shared_semantic_owner_or_credentials_permitted",
        "cha_direct_issuer_registry_mutation", "issuer_direct_account_commit",
        "issuer_direct_attempt_store_write",
    ):
        assert separation[name] is False
    assert separation["issuer_cannot_unilaterally_advance_freshness"] is True

    assert document["required_future_evidence"] == [
        "PROTECTED_ORIGINAL_STAGED_BYTES_AND_ATOMIC_MARKER_LINKAGE",
        "INDEPENDENT_REVIEWED_SQL_FUNCTION_OWNER_ROLE_AND_GRANTS",
        "TRANSACTION_LONG_LIVED_CREDENTIAL_AND_TRUST_FENCES",
        "EXACT_COMMIT_RESPONSE_LOSS_AND_SAME_ATTEMPT_BYTE_IDENTICAL_RETRY",
        "CONCURRENT_LEGAL_ATTEMPT_SERIALIZATION_AND_WINNER",
        "INDEPENDENT_CCA_SOURCE_AND_CLOSE_QUALIFICATION",
        "REAL_INSTALLATION_AND_CROSS_ROLE_ACCESS_EVIDENCE",
        "DISTINCT_CHA_FRESHNESS_CHECKPOINT_AND_ACCOUNT_GENESIS_GATES",
    ]
    production = document["production"]
    assert set(production) == {
        "issuer_ingress_enabled", "issuer_writer_enabled", "sign_or_bind_authorized",
        "complete_close_sql_unblocked", "enrollment_enabled", "stage10_unblocked",
        "production_go", "status_authority",
    }
    assert production["status_authority"] == "deployment/stage9_current_status.json"
    assert all(value is False for key, value in production.items() if key != "status_authority")


def test_addendum_has_closed_non_authorizing_scope() -> None:
    _assert_closed_safe_contract(_document())


def test_frozen_canonical_sources_remain_unchanged_and_exact_pointer_parity() -> None:
    sources = _document()["frozen_sources"]
    assert set(sources) == set(SOURCE_BLOB_PINS)
    root = ROOT / "docs/architecture/cryptohunter_product_architecture"
    for name, expected_sha in SOURCE_BLOB_PINS.items():
        source = sources[name]
        expected_path = root / SOURCE_FILENAMES[name]
        assert set(source) == {"path", "git_blob_sha1", "assertions"}
        assert source["path"] == expected_path.relative_to(ROOT).as_posix()
        assert source["git_blob_sha1"] == expected_sha
        assert re.fullmatch(r"[0-9a-f]{40}", expected_sha)
        # Git object identity is a historical repository drift check, not
        # a production cryptographic trust root or a new authority.
        actual_sha = subprocess.run(
            ["git", "hash-object", "--", str(expected_path)],
            check=True, capture_output=True, text=True, cwd=ROOT,
        ).stdout.strip()
        assert actual_sha == expected_sha, f"historical frozen source changed: {name}"
        parsed = json.loads(expected_path.read_text(encoding="utf-8"))
        assert source["assertions"] == POINTERS[name]
        for pointer, expected in POINTERS[name].items():
            assert type(_pointer(parsed, pointer)) is type(expected)
            assert _pointer(parsed, pointer) == expected


def test_status_cannot_be_promoted_by_contract_review_candidate() -> None:
    status = json.loads((ROOT / "deployment/stage9_current_status.json").read_text())
    assert status["stage_9"] == "IN_PROGRESS"
    assert status["semantic_root_proof_issuer_runtime"] == "NOT_IMPLEMENTED / BLOCKED"
    assert status["root_proof"] == "NOT_ISSUED"
    assert status["root_proof_admission"] == "NOT_STARTED"
    assert status["production_provisioning_ready"] is False
    assert status["windows_production_ready"] == "NOT_READY"
    assert status["stage_10_prerequisite"] == "BLOCKED_UNTIL_LEGAL_ENROLLMENT"


@pytest.mark.parametrize(
    "section,key,value",
    [
        ("production", "production_go", True),
        ("production", "sign_or_bind_authorized", True),
        ("precedence", "new_implementation_permission", True),
        ("precedence", "unilateral_frozen_rewrite_allowed", True),
        ("transaction", "commits", 2),
        ("transaction", "distributed_two_phase_commit", True),
        ("transaction", "retry_resign_or_remint_permitted", True),
        ("transaction", "proof_can_be_reconstructed_from_digest_only", True),
        ("transaction", "issuer_direct_helper_execute_grants", True),
        ("separate_authorities", "issuer_direct_account_commit", True),
        ("separate_authorities", "issuer_cannot_unilaterally_advance_freshness", False),
    ],
)
def test_scope_broadening_mutations_are_detected(section: str, key: str, value: object) -> None:
    document = deepcopy(_document())
    document[section][key] = value
    with pytest.raises(AssertionError):
        _assert_closed_safe_contract(document)

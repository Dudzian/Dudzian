"""S9-RPI-09: frozen non-authorizing independent committed-close read design."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from bot_core.postgresql_production_root_lifecycle_schema import (
    ADMINISTRATOR_FUNCTIONS,
    ISSUER_FUNCTIONS,
    PRODUCTION_LIFECYCLE_FUNCTION_DECLARATIONS,
    PostgreSQLProductionRootLifecycleProvisioning,
    reviewed_production_lifecycle_function_sources,
)

ROOT = Path(__file__).resolve().parents[2]
CONTRACT = ROOT / "docs/architecture/contracts/stage9_independent_committed_close_reader_v1.json"


def _unique_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for name, value in pairs:
        if name in result:
            raise ValueError("duplicate contract key")
        result[name] = value
    return result


def _read() -> dict:
    raw = CONTRACT.read_bytes()
    return json.loads(raw.decode("utf-8"), object_pairs_hook=_unique_pairs)


def _validate(contract: dict) -> None:
    assert set(contract) == {
        "artifact",
        "schema_version",
        "review_state",
        "precedence",
        "boundary",
        "read_contract",
        "failure_policy",
        "existing_guards",
        "required_future_qualification",
        "production",
    }
    assert contract["artifact"] == "S9_RPI_09_INDEPENDENT_COMMITTED_CLOSE_READER_CONTRACT_V1"
    assert type(contract["schema_version"]) is int and contract["schema_version"] == 1
    assert contract["review_state"] == "DESIGN_ONLY_UNQUALIFIED"

    precedence = contract["precedence"]
    assert set(precedence) == {
        "depends_on",
        "historical_contracts_unchanged",
        "installed_reader_authorized",
        "production_sql_migration_authorized",
    }
    assert precedence["depends_on"] == [
        "S9_ROOT_PROOF_TRANSACTION_SCOPE_ADDENDUM_V1",
        "cryptohunter.independent-custody-completion",
        "cryptohunter.custody-mutation-execution",
    ]
    assert precedence["historical_contracts_unchanged"] is True
    assert precedence["installed_reader_authorized"] is False
    assert precedence["production_sql_migration_authorized"] is False

    boundary = contract["boundary"]
    assert set(boundary) == {
        "consumer",
        "semantic_owner",
        "reader_principal",
        "source_reader_principal",
        "observed_reader_must_differ_from",
        "source_and_cca_reader_role_oid_must_differ",
        "postgres_connection_must_be_separately_authenticated",
        "peer_authentication_and_direct_login_required",
        "fixed_approved_catalog_and_physical_installation_pins_required",
        "independent_protected_policy_loader_required",
        "caller_selected_role_catalog_or_trust_anchor_accepted",
        "direct_table_select_or_dml_granted",
        "schema_role_membership_allowed",
        "no_credential_sharing_with_source_or_signer",
        "reader_only_executes_reviewed_locking_function",
    }
    assert boundary["consumer"] == "IndependentCustodyCompletionAuthority"
    assert boundary["semantic_owner"] == "IndependentProductionLifecycleAuthority"
    assert boundary["reader_principal"] == (
        "NEW_INDEPENDENT_CCA_READ_ONLY_DIRECT_LOGIN_NOT_INSTALLED"
    )
    assert boundary["source_reader_principal"] == ("DISTINCT_INSTALLED_SOURCE_READER_NOT_INSTALLED")
    assert boundary["observed_reader_must_differ_from"] == [
        "LIFECYCLE_ADMINISTRATOR",
        "ISSUER_WRITER",
        "ROOT_CUSTODY_SOURCE",
        "HISTORY_CUSTODY_SOURCE",
        "ROOT_SIGNER",
        "HISTORY_SIGNER",
        "SCHEMA_OWNER",
    ]
    for key in (
        "source_and_cca_reader_role_oid_must_differ",
        "postgres_connection_must_be_separately_authenticated",
        "peer_authentication_and_direct_login_required",
        "fixed_approved_catalog_and_physical_installation_pins_required",
        "independent_protected_policy_loader_required",
        "no_credential_sharing_with_source_or_signer",
        "reader_only_executes_reviewed_locking_function",
    ):
        assert boundary[key] is True
    for key in (
        "caller_selected_role_catalog_or_trust_anchor_accepted",
        "direct_table_select_or_dml_granted",
        "schema_role_membership_allowed",
    ):
        assert boundary[key] is False

    read = contract["read_contract"]
    assert set(read) == {
        "future_sql_entrypoint",
        "isolation",
        "transaction_read_only",
        "row_lock_requires_read_write_transaction",
        "locking_execution_context",
        "row_locking_sql_owner_must_be_separate_from_cca_direct_login",
        "security_definer_requirements",
        "independent_transaction_after_closed_commit",
        "acquire_exact_slot_lock_before_state_read",
        "lock_held_through_exact_source_comparison",
        "connection_and_transaction_identity_rechecked",
        "returned_request_bytes",
        "request_format",
        "exact_fields",
        "allowed_close_phase",
        "exact_original_bytes_preserved",
        "request_digest",
        "close_xid_positive_canonical_decimal",
        "observation_xid_positive_canonical_decimal",
        "observation_xid_differs_from_close_xid",
        "independent_read_must_differ_from_source_read_role_and_oid",
        "expected_catalog_sha256_matches_protected_external_pin",
        "comparison_includes",
        "read_result_must_be_exactly_one",
        "unknown_or_uncommitted_close_is_error",
        "snapshot_only_or_double_policy_read_is_not_lifecycle_fence",
        "no_signing_mutation_or_completion_publication",
    }
    assert read["future_sql_entrypoint"] == "REVIEW_REQUIRED_NO_ENTRYPOINT_INSTALLED"
    assert read["isolation"] == "READ COMMITTED"
    assert read["transaction_read_only"] is False
    assert read["row_lock_requires_read_write_transaction"] is True
    assert (
        read["locking_execution_context"]
        == "REVIEWED_SECURITY_DEFINER_WITH_MINIMAL_EXECUTE_ONLY_READER"
    )
    assert read["row_locking_sql_owner_must_be_separate_from_cca_direct_login"] is True
    security_definer = read["security_definer_requirements"]
    assert set(security_definer) == {
        "function_local_fixed_reviewed_search_path_required",
        "search_path_order",
        "unauthorized_create_in_trusted_schemas_forbidden",
        "caller_or_session_search_path_allowed",
        "revoke_execute_from_public_required",
        "revoke_execute_from_all_unauthorized_principals_required",
        "execute_grant_target",
        "execute_grants_to_other_principals_allowed",
        "unauthorized_inherited_or_set_role_execute_allowed",
        "create_revoke_grant_in_one_transaction_required",
        "unauthorized_execute_window_allowed",
        "installation_failure_requires_full_rollback",
    }
    assert security_definer["search_path_order"] == [
        "pg_catalog",
        "REVIEWED_INSTALLATION_PINNED_TRUSTED_SCHEMAS",
        "pg_temp",
    ]
    assert security_definer["execute_grant_target"] == (
        "INSTALLATION_PINNED_CCA_READER_PRINCIPAL_AND_ROLE_OID"
    )
    for key in (
        "function_local_fixed_reviewed_search_path_required",
        "unauthorized_create_in_trusted_schemas_forbidden",
        "revoke_execute_from_public_required",
        "revoke_execute_from_all_unauthorized_principals_required",
        "create_revoke_grant_in_one_transaction_required",
        "installation_failure_requires_full_rollback",
    ):
        assert security_definer[key] is True
    for key in (
        "caller_or_session_search_path_allowed",
        "execute_grants_to_other_principals_allowed",
        "unauthorized_inherited_or_set_role_execute_allowed",
        "unauthorized_execute_window_allowed",
    ):
        assert security_definer[key] is False
    assert read["request_format"] == "PostgreSQL16/jsonb-text/UTF8"
    assert read["returned_request_bytes"] == (
        "request::text encoded UTF8 in PostgreSQL 16; never caller-wire JSON"
    )
    assert read["request_digest"] == "sha256_of_exact_retained_request_text_utf8"
    assert read["allowed_close_phase"] == "PENDING_CLOSED"
    assert read["exact_fields"] == [
        "operation_id",
        "semantic_role",
        "identity",
        "generation",
        "predecessor_state",
        "target_state",
        "request_sha256",
        "close_xid",
        "phase",
    ]
    assert read["comparison_includes"] == [
        "operation_id",
        "semantic_role",
        "credential_identity",
        "predecessor_generation",
        "before_and_after_state",
        "request_format",
        "request_bytes",
        "request_sha256",
        "close_xid",
        "database_system_identifier",
        "database_oid",
        "installation_id",
        "source_log_position",
    ]
    for key in set(read) - {
        "future_sql_entrypoint",
        "isolation",
        "returned_request_bytes",
        "request_format",
        "exact_fields",
        "allowed_close_phase",
        "request_digest",
        "comparison_includes",
        "transaction_read_only",
        "locking_execution_context",
        "security_definer_requirements",
    }:
        assert read[key] is True

    failure = contract["failure_policy"]
    assert set(failure) == {
        "not_found",
        "timeout",
        "ambiguous_commit",
        "drift_or_role_alias",
        "missing_or_changed_catalog_pin",
        "xid_collision",
        "request_bytes_mismatch",
        "concurrent_descendant_without_held_lease",
        "external_policy_not_proven",
        "allow_positive_admission_on_structural_claims",
    }
    assert failure["not_found"] == "OUTCOME_UNKNOWN_NO_RETRY_AUTHORIZATION"
    assert failure["timeout"] == "INDETERMINATE_FAIL_CLOSED"
    assert failure["ambiguous_commit"] == "INDETERMINATE_FAIL_CLOSED"
    assert failure["external_policy_not_proven"] == "NOT_QUALIFIED"
    assert failure["allow_positive_admission_on_structural_claims"] is False
    for key in (
        "drift_or_role_alias",
        "missing_or_changed_catalog_pin",
        "xid_collision",
        "request_bytes_mismatch",
        "concurrent_descendant_without_held_lease",
    ):
        assert failure[key] == "FAIL_CLOSED"

    guards = contract["existing_guards"]
    assert set(guards) == {
        "administrator_read_operation_only",
        "no_current_cca_sql_function",
        "existing_complete_close_is_denial_tombstone",
        "complete_close_sqlstate",
        "CCA_reader_in_administrator_functions",
        "CCA_reader_in_issuer_functions",
        "no_new_sql_or_grants_in_this_change",
    }
    assert guards["administrator_read_operation_only"] == "read_operation(text,text,text)"
    assert guards["complete_close_sqlstate"] == "42501"
    for key in (
        "no_current_cca_sql_function",
        "existing_complete_close_is_denial_tombstone",
        "no_new_sql_or_grants_in_this_change",
    ):
        assert guards[key] is True
    for key in ("CCA_reader_in_administrator_functions", "CCA_reader_in_issuer_functions"):
        assert guards[key] is False

    assert contract["required_future_qualification"] == [
        "REVIEWED_VERSIONED_SQL_CATALOG_AND_MIGRATION",
        "PROTECTED_EXACT_PHYSICAL_DB_AND_CATALOG_PIN",
        "DEDICATED_ROLE_OID_AND_DIRECT_PEER_LOGIN",
        "NO_CROSS_ROLE_MEMBERSHIP_OR_ALTERNATE_WRITE_PORT",
        "READ_WRITE_TRANSACTION_FOR_ROW_LOCKING_WITH_ZERO_READER_DIRECT_DML",
        "READ_COMMITTED_SLOT_SYNCHRONIZATION_UNDER_REAL_CONTENTION",
        "POSTGRESQL_COMMIT_RESPONSE_LOSS_AND_RECOVERY",
        "EXACT_JSONB_TEXT_BYTES_AND_REQUEST_DIGEST",
        "SOURCE_CCA_READER_IDENTITY_AND_XID_SEPARATION",
        "HISTORICAL_CREDENTIAL_AND_SOURCE_LEASE_FENCES",
        "NATIVE_INSTALLATION_DAC_ACL_AND_SYSTEMD_ATTESTATION",
    ]

    production = contract["production"]
    assert set(production) == {
        "stage9",
        "issuer_runtime",
        "root_proof",
        "production_provisioning_ready",
        "reader_installed",
        "reader_qualified",
        "CCA_authorized",
        "sign_bind_authorized",
        "complete_close_unblocked",
        "enrollment_authorized",
        "stage10_unblocked",
        "status_authority",
    }
    assert production["stage9"] == "IN_PROGRESS"
    assert production["issuer_runtime"] == "NOT_IMPLEMENTED / BLOCKED"
    assert production["root_proof"] == "NOT_ISSUED"
    assert production["status_authority"] == "deployment/stage9_current_status.json"
    for key in set(production) - {
        "stage9",
        "issuer_runtime",
        "root_proof",
        "status_authority",
    }:
        assert production[key] is False


def test_closed_reader_contract_remains_non_authorizing() -> None:
    _validate(_read())


def test_existing_production_sql_is_still_denying_and_admin_only() -> None:
    config = PostgreSQLProductionRootLifecycleProvisioning(
        schema="reviewed_lifecycle",
        schema_owner_role="reviewed_owner",
        issuer_role="reviewed_issuer",
        administrator_role="reviewed_administrator",
        installation_id="reviewed_installation",
        trust_domain="reviewed_domain",
    )
    sources = reviewed_production_lifecycle_function_sources(config)
    assert "read_operation(text,text,text)" in ADMINISTRATOR_FUNCTIONS
    assert "read_operation(text,text,text)" not in ISSUER_FUNCTIONS
    assert "complete_close(text,text,bytea)" not in ADMINISTRATOR_FUNCTIONS
    assert "complete_close(text,text,bytea)" not in ISSUER_FUNCTIONS
    assert set(sources) == set(PRODUCTION_LIFECYCLE_FUNCTION_DECLARATIONS)
    assert "FOR UPDATE" in sources["read_operation(text,text,text)"]
    assert "'read committed'" in sources["read_operation(text,text,text)"]
    assert (
        "op.close_xid=pg_catalog.pg_current_xact_id()" in sources["read_operation(text,text,text)"]
    )
    tombstone = sources["complete_close(text,text,bytea)"]
    assert "INDEPENDENT_CUSTODY_COMPLETION_AUTHORITY_REQUIRED" in tombstone
    assert "42501" in tombstone
    assert "INSERT " not in tombstone and "UPDATE " not in tombstone


def test_stage9_status_not_advanced_by_design_contract() -> None:
    status = json.loads((ROOT / "deployment/stage9_current_status.json").read_text())
    assert status["stage_9"] == "IN_PROGRESS"
    assert status["semantic_root_proof_issuer_runtime"] == "NOT_IMPLEMENTED / BLOCKED"
    assert status["root_proof"] == "NOT_ISSUED"
    assert status["production_provisioning_ready"] is False
    assert status["windows_production_ready"] == "NOT_READY"
    assert status["stage_10_prerequisite"] == "BLOCKED_UNTIL_LEGAL_ENROLLMENT"


@pytest.mark.parametrize(
    "section,key,value",
    [
        ("precedence", "installed_reader_authorized", True),
        ("precedence", "production_sql_migration_authorized", True),
        ("boundary", "caller_selected_role_catalog_or_trust_anchor_accepted", True),
        ("boundary", "schema_role_membership_allowed", True),
        ("boundary", "source_and_cca_reader_role_oid_must_differ", False),
        ("boundary", "peer_authentication_and_direct_login_required", False),
        ("read_contract", "isolation", "SERIALIZABLE"),
        ("read_contract", "transaction_read_only", True),
        ("read_contract", "locking_execution_context", "DIRECT_CCA_TABLE_UPDATE"),
        ("boundary", "reader_only_executes_reviewed_locking_function", False),
        ("read_contract", "request_format", "caller_wire_json"),
        ("read_contract", "observation_xid_differs_from_close_xid", False),
        ("read_contract", "unknown_or_uncommitted_close_is_error", False),
        ("failure_policy", "allow_positive_admission_on_structural_claims", True),
        ("failure_policy", "not_found", "START_NEW_OPERATION"),
        ("production", "reader_qualified", True),
        ("production", "complete_close_unblocked", True),
        ("existing_guards", "no_current_cca_sql_function", False),
    ],
)
def test_scope_drift_fails_closed(section: str, key: str, value: object) -> None:
    modified = deepcopy(_read())
    modified[section][key] = value
    with pytest.raises(AssertionError):
        _validate(modified)


@pytest.mark.parametrize(
    "key,value",
    [
        ("function_local_fixed_reviewed_search_path_required", False),
        ("unauthorized_create_in_trusted_schemas_forbidden", False),
        ("caller_or_session_search_path_allowed", True),
        ("revoke_execute_from_public_required", False),
        ("revoke_execute_from_all_unauthorized_principals_required", False),
        ("execute_grant_target", "PUBLIC"),
        ("execute_grant_target", "CALLER_SELECTED_CCA_ROLE"),
        ("execute_grant_target", "CCA_ROLE_NAME_WITHOUT_INSTALLATION_OR_OID_PIN"),
        ("execute_grants_to_other_principals_allowed", True),
        ("unauthorized_inherited_or_set_role_execute_allowed", True),
        ("create_revoke_grant_in_one_transaction_required", False),
        ("unauthorized_execute_window_allowed", True),
        ("installation_failure_requires_full_rollback", False),
        ("future_optional_bypass", True),
    ],
)
def test_security_definer_requirements_cannot_be_weakened(key: str, value: object) -> None:
    modified = deepcopy(_read())
    modified["read_contract"]["security_definer_requirements"][key] = value
    with pytest.raises(AssertionError):
        _validate(modified)


@pytest.mark.parametrize(
    "search_path",
    [
        [],
        ["pg_catalog", "REVIEWED_INSTALLATION_PINNED_TRUSTED_SCHEMAS"],
        ["pg_temp", "pg_catalog", "REVIEWED_INSTALLATION_PINNED_TRUSTED_SCHEMAS"],
        ["pg_catalog", "pg_temp", "REVIEWED_INSTALLATION_PINNED_TRUSTED_SCHEMAS"],
        ["REVIEWED_INSTALLATION_PINNED_TRUSTED_SCHEMAS", "pg_temp"],
        ["REVIEWED_INSTALLATION_PINNED_TRUSTED_SCHEMAS", "pg_catalog", "pg_temp"],
        ["pg_catalog", "public", "pg_temp"],
        ["pg_catalog", "$user", "pg_temp"],
        ["pg_catalog", "UNTRUSTED_SCHEMA", "pg_temp"],
        ["pg_catalog", "CALLER_SELECTED_SCHEMA", "pg_temp"],
        ["pg_catalog", "SESSION_SEARCH_PATH", "pg_temp"],
    ],
)
def test_security_definer_unsafe_search_path_is_rejected(search_path: list[str]) -> None:
    modified = deepcopy(_read())
    modified["read_contract"]["security_definer_requirements"]["search_path_order"] = search_path
    with pytest.raises(AssertionError):
        _validate(modified)


def test_security_definer_requirements_are_mandatory_and_strictly_typed() -> None:
    original = _read()
    missing_block = deepcopy(original)
    del missing_block["read_contract"]["security_definer_requirements"]
    with pytest.raises(AssertionError):
        _validate(missing_block)
    for key, value in original["read_contract"]["security_definer_requirements"].items():
        missing_field = deepcopy(original)
        del missing_field["read_contract"]["security_definer_requirements"][key]
        with pytest.raises(AssertionError):
            _validate(missing_field)
        if type(value) is bool:
            for invalid in (int(value), None, str(value).lower()):
                wrong_type = deepcopy(original)
                wrong_type["read_contract"]["security_definer_requirements"][key] = invalid
                with pytest.raises(AssertionError):
                    _validate(wrong_type)


def test_duplicate_contract_fields_are_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    raw = CONTRACT.read_text(encoding="utf-8")
    field = '"reader_installed": false'
    assert raw.count(field) == 1
    modified = raw.replace(field, '"reader_installed": false, "reader_installed": true', 1)
    path = tmp_path / "duplicate.json"
    path.write_text(modified, encoding="utf-8")
    monkeypatch.setitem(globals(), "CONTRACT", path)
    with pytest.raises(ValueError, match="duplicate contract key"):
        _read()

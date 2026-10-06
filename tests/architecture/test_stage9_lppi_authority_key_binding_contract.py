"""Initial successor subordinate freeze, parent immutability and runtime guards."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import jsonschema
import pytest

from bot_core.licensing import (
    lppi_authority_custody as custody,
    lppi_authority_key as binding,
    lppi_package_acceptance as acceptance,
)
from deployment import (
    windows_lppi_authority_key as native,
    windows_production_lppi_authority as lifecycle,
)

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs/architecture/cryptohunter_product_architecture"
CONTRACT = DOCS / "stage9_lppi_authority_key_binding_contract.json"
FREEZE = DOCS / "stage9_lppi_authority_key_binding_freeze.json"
PARENT = DOCS / "stage9_external_provisioning_architecture_contract.json"
PARENT_SHA = "6538a75d8cf3a6e3cdab8f4a9022277a21d98e1a61923aa8d8eae9b80ae1d67d"


def contract():
    return json.loads(CONTRACT.read_bytes())


def test_exact_parent_and_subordinate_hashes_frozen():
    value = contract()
    freeze = json.loads(FREEZE.read_bytes())
    assert value["status"] == freeze["status"] == "FROZEN"
    assert value["parent_contract"]["sha256"] == freeze["parent_contract_sha256"] == PARENT_SHA
    assert hashlib.sha256(PARENT.read_bytes()).hexdigest() == PARENT_SHA
    assert freeze["artifacts"] == [
        {
            "canonical_artifact": CONTRACT.relative_to(ROOT).as_posix(),
            "canonical_schema_version": value["schema_version"],
            "sha256": hashlib.sha256(CONTRACT.read_bytes()).hexdigest(),
        }
    ]
    assert value["production_provisioning_ready"] is False
    assert value["legal_production_enrollment"] == "NOT_PERFORMED"
    assert value["stage_status"] == {
        "stage_9": "IN_PROGRESS",
        "windows_production_ready": "NOT_READY",
        "stage_10_production_lifecycle_live": "NOT_STARTED",
        "stage_10_prerequisite": "BLOCKED_UNTIL_LEGAL_ENROLLMENT",
    }


def test_runtime_parent_exact_fields_domains_profiles_and_state():
    value = contract()
    parent = json.loads(PARENT.read_bytes())["lppi_authority_key_lifecycle"]
    assert (
        value["scope"]["selection"]
        == parent["selection"]
        == "SUCCESSOR_KEY_AFTER_PACKAGE_VERIFICATION"
    )
    assert (
        value["binding_artifact"]["canonical_payload_fields"]
        == parent["binding_artifact"]["canonical_payload_fields"]
    )
    assert set(value["binding_artifact"]["canonical_payload_fields"]) == binding.BINDING_FIELDS
    assert len(binding.BINDING_FIELDS) == 22
    assert value["successor_key"]["key_name"] == binding.KEY_NAME == native.KEY_NAME
    assert (
        value["binding_artifact"]["required_values"]["custody_profile"]
        == binding.CUSTODY_PROFILE
        == custody.CUSTODY_PROFILE
        == native.QUALIFICATION_PROFILE
    )
    assert binding.ALGORITHM_PROFILE == custody.ALGORITHM_PROFILE == "ECDSA-P256-SHA256"
    assert (
        parent["binding_artifact"]["continuity_signature_profile"]["domain"].encode() + b"\0"
        == binding.CONTINUITY_DOMAIN
    )
    assert (
        parent["binding_artifact"]["authority_key_pop_profile"]["domain"].encode() + b"\0"
        == binding.AUTHORITY_POP_DOMAIN
    )
    assert (
        set(parent["binding_artifact"]["authority_key_pop_profile"]["challenge_fields"])
        == binding.POP_FIELDS
    )
    assert (
        value["durable_lifecycle"]["status_progression"]
        == list(lifecycle.STATUSES[1:])
        == parent["binding_artifact"]["status_progression"]
    )
    assert set(value["durable_lifecycle"]["storage"]["record_fields"]) == lifecycle._FIELDS
    assert value["durable_lifecycle"]["creation_outcomes"] == list(lifecycle.CREATION_OUTCOMES)
    assert set(value["custody_evidence"]["canonical_fields"]) == custody.CUSTODY_FIELDS
    assert set(value["custody_evidence"]["qualifying_data_fields"]) == custody.BINDING_FIELDS
    assert value["custody_evidence"]["domain"].encode() + b"\0" == custody.CUSTODY_CREATION_DOMAIN
    assert (
        set(value["target_package_acceptance"]["challenge_fields"])
        == acceptance.PACKAGE_ACCEPTANCE_FIELDS
    )
    assert (
        value["target_package_acceptance"]["domain"].encode() + b"\0"
        == acceptance.PACKAGE_ACCEPTANCE_DOMAIN
    )


def test_creation_ownership_excludes_collision_and_attempt_only_reconciliation():
    ownership = contract()["durable_lifecycle"]["creation_ownership"]
    assert ownership["create"]["first_identity_operation"] == "NCryptCreatePersistedKey"
    assert ownership["create"]["open_before_create"] is False
    assert ownership["create"]["overwrite_or_delete"] is False
    own_outcomes = ["OWN_FINALIZE_PENDING", "OWN_FINALIZED"]
    assert set(own_outcomes) == lifecycle._OWN_CREATION_OUTCOMES
    for mode in ("reconcile", "recover"):
        assert ownership[mode]["allowed_outcomes"] == own_outcomes
        assert ownership[mode]["native_operation"] == "NCryptOpenKey_ONLY"
    assert ownership["reconcile"]["remint"] is False
    assert ownership["recover"]["exact_retained_identity_required"] is True
    assert ownership["ownership_marker"] == {
        "outcome": "OWN_FINALIZE_PENDING",
        "prerequisites": [
            "OWN_NCryptCreatePersistedKey_SUCCESS",
            "NONEMPTY_OWN_HANDLE",
            "REQUIRED_PROPERTIES_APPLIED",
        ],
        "durably_fsynced_before": "NCryptFinalizeKey",
    }
    collision = ownership["collision"]
    assert int(collision["status"], 16) == 0x8009000F
    assert collision["symbol"] == "NTE_EXISTS"
    assert collision["native_operations"] == ["NCryptCreatePersistedKey", "NCryptFinalizeKey"]
    assert collision["outcome"] == "CREATE_COLLISION"
    assert collision["terminal"] is True
    assert collision["restart_open_or_create"] is False
    assert collision["qualification_signing_custody_or_active"] is False
    failure = ownership["definitive_create_failure"]
    assert failure["cases"] == [
        "OTHER_NCryptCreatePersistedKey_ERROR",
        "SUCCESS_WITH_EMPTY_HANDLE",
        "REQUIRED_PROPERTY_FAILURE",
    ]
    assert failure["outcome"] == "CREATE_FAILED"
    assert failure["default"] == "FAIL_CLOSED"
    assert failure["restart_open_or_create"] is False
    attempt = ownership["attempt_without_ownership"]
    assert attempt["outcome"] == "ATTEMPT_STARTED"
    assert attempt["restart_open_or_create"] is False
    assert attempt["profile_qualification_establishes_ownership"] is False
    assert ownership["ambiguous_finalize"]["outcome"] == "OWN_FINALIZE_PENDING"
    assert ownership["ambiguous_finalize"]["qualified_reconciliation_outcome"] == "OWN_FINALIZED"
    assert ownership["legacy_record_without_creation_outcome"] == {
        "load": "REJECT",
        "automatic_migration": False,
    }
    assert ownership["candidate_ownership"] == {
        "from_status": "CANDIDATE",
        "required_outcome": "OWN_FINALIZED",
    }


@pytest.mark.parametrize(
    "section,key",
    [
        ("binding_artifact", "json_schema"),
        ("target_package_acceptance", "challenge_json_schema"),
        ("custody_evidence", "json_schema"),
    ],
)
def test_schemas_require_exact_fields_and_reject_extension(section, key):
    schema = contract()[section][key]
    jsonschema.Draft202012Validator.check_schema(schema)
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == set(schema["properties"])
    if section == "target_package_acceptance":
        valid = {field: "ab" * 32 for field in schema["required"]}
        jsonschema.validate(valid, schema)
        for field in schema["required"]:
            bad = dict(valid)
            bad[field] += "\n"
            with pytest.raises(jsonschema.ValidationError):
                jsonschema.validate(bad, schema)
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(valid | {"extension": True}, schema)


def test_initial_only_and_no_membership_issuance():
    value = contract()
    assert value["scope"]["generation"] == 1
    assert value["scope"]["replacement_supported"] is False
    assert value["scope"]["retirement_supported"] is False
    assert value["scope"]["pre_enrollment_key_retained"] is True
    assert "LPPIAuthenticatedProvisioningOperationBindingV1" in value["scope_exclusions"]
    assert "provisioning_operation_id" in value["scope_exclusions"]
    assert value["active_capability"]["public_constructor"] is False

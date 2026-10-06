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

"""Machine-checkable subordinate freeze; parent historical bytes stay unchanged."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import jsonschema
import pytest

from bot_core.licensing import pdsa_enrollment_challenge as executable
from bot_core.licensing.product_profile import PRODUCT_NAME, PRODUCTION_PRODUCT_PROFILE
from deployment.windows_stage9_production_trust import PDSA_KEY_SET_DIGEST

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs/architecture/cryptohunter_product_architecture"
CONTRACT = DOCS / "stage9_pdsa_enrollment_challenge_contract.json"
FREEZE = DOCS / "stage9_pdsa_enrollment_challenge_freeze.json"
PARENT = DOCS / "stage9_external_provisioning_architecture_contract.json"
PARENT_SHA = "6538a75d8cf3a6e3cdab8f4a9022277a21d98e1a61923aa8d8eae9b80ae1d67d"


def _contract():
    return json.loads(CONTRACT.read_bytes())


def test_exact_subordinate_freeze_and_parent_integrity():
    value = _contract()
    freeze = json.loads(FREEZE.read_bytes())
    assert value["status"] == freeze["status"] == "FROZEN"
    assert value["implementation_authorized"] is True
    assert value["production_provisioning_ready"] is False
    assert value["parent_contract"]["historical_parent_bytes_unchanged"] is True
    assert value["parent_contract"]["sha256"] == PARENT_SHA
    assert hashlib.sha256(PARENT.read_bytes()).hexdigest() == PARENT_SHA
    assert freeze["parent_contract_sha256"] == PARENT_SHA
    assert freeze["artifacts"] == [
        {
            "canonical_artifact": CONTRACT.relative_to(ROOT).as_posix(),
            "canonical_schema_version": value["schema_version"],
            "sha256": hashlib.sha256(CONTRACT.read_bytes()).hexdigest(),
        }
    ]
    assert value["stage_status"] == {
        "stage_9": "IN_PROGRESS",
        "windows_production_ready": "NOT_READY",
        "stage_10_production_lifecycle_live": "NOT_STARTED",
        "stage_10_prerequisite": "BLOCKED_UNTIL_LEGAL_ENROLLMENT",
    }


def test_machine_schema_and_exact_runtime_wire_fields():
    value = _contract()
    wire = value["wire_contract"]
    assert set(wire["envelope_fields"]) == {"payload", "signatures"}
    assert set(wire["payload_fields"]) == executable.PAYLOAD_FIELDS
    assert set(wire["signature_fields"]) == executable.SIGNATURE_FIELDS
    schema = wire["json_schema"]
    jsonschema.Draft202012Validator.check_schema(schema)
    payload = schema["properties"]["payload"]
    assert set(payload["required"]) == set(payload["properties"]) == executable.PAYLOAD_FIELDS
    assert payload["additionalProperties"] is False
    properties = payload["properties"]
    for field, expected in {
        "schema_version": executable.SCHEMA_VERSION,
        "environment": "PRODUCTION",
        "product": PRODUCT_NAME,
        "product_profile": PRODUCTION_PRODUCT_PROFILE,
        "pdsa_trust_domain": executable.PDSA_TRUST_DOMAIN,
        "pdsa_key_set_digest": PDSA_KEY_SET_DIGEST,
        "signature_algorithm_profile": executable.SIGNATURE_PROFILE,
    }.items():
        assert properties[field] == {"const": expected}
    signature = schema["properties"]["signatures"]
    assert (signature["minItems"], signature["maxItems"]) == (2, 3)
    assert set(signature["items"]["required"]) == executable.SIGNATURE_FIELDS


def test_frozen_signature_and_freshness_match_runtime_and_parent():
    value = _contract()
    parent = json.loads(PARENT.read_bytes())
    signature = value["signature_profile"]
    trust = parent["pdsa_trust_domain"]
    assert (
        (signature["threshold"], signature["key_count"], signature["algorithm"])
        == (trust["threshold"], trust["key_count"], trust["algorithm"])
        == (2, 3, "Ed25519")
    )
    assert signature["single_signer_authority"] is False
    assert signature["caller_selected_public_keys_or_threshold"] is False
    assert signature["profile"] == executable.SIGNATURE_PROFILE
    assert executable.SIGNATURE_DOMAIN == signature["domain"].encode() + b"\x00"
    assert value["issuer_freshness"]["validity_seconds"] == executable.VALIDITY_SECONDS == 604800
    assert value["issuer_freshness"]["caller_selected_challenge_id_or_nonce"] is False
    assert value["digest_derivations"]["pdsa_challenge_digest_sha256"] == (
        "lowercase hex SHA256(exact canonical signed envelope bytes including signatures)"
    )
    assert value["digest_derivations"]["pdsa_challenge_nonce_digest_sha256"] == (
        "lowercase hex SHA256(exact decoded 32-byte nonce)"
    )


def test_terminal_retention_and_acceptance_only_boundary():
    value = _contract()
    retained = value["retention"]
    assert retained["states"] == ["ISSUED", "CONSUMED", "EXPIRED"]
    assert retained["legal_transitions"] == [["ISSUED", "CONSUMED"], ["ISSUED", "EXPIRED"]]
    assert retained["terminal_states"] == ["CONSUMED", "EXPIRED"]
    assert retained["same_challenge_different_request"] == "CHALLENGE_REPLAY_CONFLICT"
    assert "require_authenticated_pre_enrollment" in retained["consume_gate"]
    receipt = value["acceptance_receipt"]
    assert receipt["legal_enrollment"] == "NOT_PERFORMED"
    assert receipt["mint_provisioning_subject_id"] is False
    assert receipt["mint_package_membership_or_lppi_key"] is False
    assert value["verified_result"]["copied_fields_transfer_authority"] is False


@pytest.mark.parametrize(
    ("field", "sample"),
    [
        ("nonce_hex", "ab" * 32),
        ("release_policy_digest_sha256", "ab" * 32),
        ("issued_at_utc", "2026-10-06T12:00:00Z"),
        ("expires_at_utc", "2026-10-13T12:00:00Z"),
        ("challenge_id", "pchal_01930000-0000-7000-8000-000000000001"),
        ("key_id", "TEST_ONLY_SCHEMA_KEY"),
        ("signature_hex", "00" * 64),
    ],
)
def test_schema_hex_identity_and_time_fields_reject_trailing_newline(field, sample):
    schema = _contract()["wire_contract"]["json_schema"]
    payload_fields = schema["properties"]["payload"]["properties"]
    signature_fields = schema["properties"]["signatures"]["items"]["properties"]
    profile = payload_fields[field] if field in payload_fields else signature_fields[field]
    jsonschema.validate(sample, profile)
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(sample + "\n", profile)


@pytest.mark.parametrize("extra_scope", ["envelope", "payload", "signature"])
def test_machine_schema_rejects_unknown_fields(extra_scope):
    schema = _contract()["wire_contract"]["json_schema"]
    properties = schema["properties"]["payload"]["properties"]
    payload = {field: profile.get("const", "ab" * 32) for field, profile in properties.items()}
    payload.update(
        challenge_id="pchal_01930000-0000-7000-8000-000000000001",
        release_policy_generation=1,
        issued_at_utc="2026-10-06T12:00:00Z",
        expires_at_utc="2026-10-13T12:00:00Z",
    )
    document = {
        "payload": payload,
        "signatures": [
            {
                "key_id": f"TEST_ONLY_SCHEMA_{index}",
                "algorithm": "Ed25519",
                "signature_hex": "00" * 64,
            }
            for index in (1, 2)
        ],
    }
    jsonschema.validate(document, schema)
    changed = copy.deepcopy(document)
    target = changed if extra_scope == "envelope" else changed["payload"]
    if extra_scope == "signature":
        target = changed["signatures"][0]
    target["unknown"] = "forbidden"
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(changed, schema)

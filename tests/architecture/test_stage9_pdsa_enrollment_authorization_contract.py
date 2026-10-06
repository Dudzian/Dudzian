"""Initial authorization subordinate freeze and machine-checkable denial guards."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import jsonschema
import pytest

from bot_core.licensing import external_provisioning as executable
from bot_core.licensing.pre_enrollment import MAX_EXACT_INTEGER, PDSA_TRUST_DOMAIN
from bot_core.licensing.product_profile import PRODUCTION_PRODUCT_PROFILE

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs/architecture/cryptohunter_product_architecture"
CONTRACT = DOCS / "stage9_pdsa_enrollment_authorization_contract.json"
FREEZE = DOCS / "stage9_pdsa_enrollment_authorization_freeze.json"
PARENT = DOCS / "stage9_external_provisioning_architecture_contract.json"
CHALLENGE = DOCS / "stage9_pdsa_enrollment_challenge_contract.json"
PARENT_SHA = "6538a75d8cf3a6e3cdab8f4a9022277a21d98e1a61923aa8d8eae9b80ae1d67d"
CHALLENGE_SHA = "d79ee4dee2c5637e48bfdd712d1e6882295841387dc09a6a390e6503c57cd78f"
UUID7 = "01930000-0000-7000-8000-000000000001"


def _load(path: Path) -> dict[str, Any]:
    result = json.loads(path.read_bytes())
    assert isinstance(result, dict)
    return result


def _schema() -> dict[str, Any]:
    return _load(CONTRACT)["wire_contract"]["json_schema"]


def _document() -> dict[str, Any]:
    properties = _schema()["properties"]["payload"]["properties"]
    payload = {field: profile.get("const", "ab" * 32) for field, profile in properties.items()}
    payload.update(
        provisioning_subject_id="psub_" + UUID7,
        enrollment_reference="penr_" + UUID7,
        pdsa_challenge_id="pchal_" + UUID7,
        issued_at_utc="2026-10-06T12:00:00Z",
        expires_at_utc="2026-10-06T12:01:00Z",
        release_policy_generation=1,
        predecessor_package_digest_or_null=None,
    )
    return {
        "payload": payload,
        "signatures": [
            {"algorithm": "Ed25519", "key_id": f"SCHEMA_ONLY_{index}", "signature_hex": "00" * 64}
            for index in (1, 2)
        ],
    }


def test_subordinate_freeze_preserves_exact_parent_and_challenge_bytes() -> None:
    value, freeze = _load(CONTRACT), _load(FREEZE)
    assert value["status"] == freeze["status"] == "FROZEN"
    assert value["implementation_authorized"] is True
    assert value["production_provisioning_ready"] is False
    assert freeze["implementation_architecturally_authorized"] is True
    assert freeze["production_provisioning_ready"] is False
    assert value["parent_contract"]["historical_parent_bytes_unchanged"] is True
    assert value["parent_contract"]["sha256"] == PARENT_SHA
    assert hashlib.sha256(PARENT.read_bytes()).hexdigest() == PARENT_SHA
    assert value["challenge_contract"]["sha256"] == CHALLENGE_SHA
    assert hashlib.sha256(CHALLENGE.read_bytes()).hexdigest() == CHALLENGE_SHA
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


def test_machine_schema_matches_parent_and_runtime_exact_payload() -> None:
    value, parent = _load(CONTRACT), _load(PARENT)
    wire = value["wire_contract"]
    frozen_package = parent["pdsa_enrollment_authorization_package"]
    assert value["artifact"] == frozen_package["artifact"] == "PDSAEnrollmentAuthorizationPackageV1"
    assert wire["payload_fields"] == frozen_package["canonical_payload_fields"]
    assert len(wire["payload_fields"]) == 23
    assert set(wire["payload_fields"]) == executable.PACKAGE_FIELDS
    assert wire["envelope_fields"] == ["payload", "signatures"]
    assert set(wire["signature_fields"]) == {"algorithm", "key_id", "signature_hex"}
    assert "provisioning_operation_id" not in wire["payload_fields"]
    assert frozen_package["contains_provisioning_operation_id"] is False
    schema = wire["json_schema"]
    jsonschema.Draft202012Validator.check_schema(schema)
    payload = schema["properties"]["payload"]
    assert payload["additionalProperties"] is False
    assert set(payload["required"]) == set(payload["properties"]) == executable.PACKAGE_FIELDS
    properties = payload["properties"]
    for field, expected in {
        "schema_version": executable.PACKAGE_SCHEMA,
        "environment": "PRODUCTION",
        "pdsa_trust_domain": PDSA_TRUST_DOMAIN,
        "product_profile": PRODUCTION_PRODUCT_PROFILE,
        "pre_enrollment_public_key_algorithm_profile": "ECDSA-P256-SHA256",
    }.items():
        assert properties[field] == {"const": expected}
    assert properties["release_policy_generation"]["maximum"] == MAX_EXACT_INTEGER
    jsonschema.validate(_document(), schema)


def test_initial_lineage_and_exact_two_signature_policy_are_frozen() -> None:
    value = _load(CONTRACT)
    initial = value["initial_authorization"]
    assert initial["supported_mode"] == "INITIAL_ENROLLMENT_ONLY"
    assert (
        initial["authorization_generation"],
        initial["authorization_version"],
        initial["lineage_generation"],
        initial["predecessor_package_digest_or_null"],
    ) == (1, 1, 1, None)
    signature = value["signature_profile"]
    parent_signature = _load(PARENT)["pdsa_enrollment_authorization_package"]["signature_profile"]
    assert signature["algorithm"] == parent_signature["algorithm"] == "Ed25519"
    assert (
        signature["threshold"],
        signature["key_count"],
        signature["supplied_signature_count"],
    ) == (
        2,
        3,
        2,
    )
    assert signature["domain"].encode() + b"\x00" == executable.PDSA_DOMAIN
    assert signature["caller_selected_signer_or_threshold"] is False
    assert "lexical order" in signature["signature_order"]
    assert (
        "exactly all three authorized key IDs and public keys" in signature["eligible_signer_set"]
    )
    assert "current ProductionTrustContext.pdsa_keys" in signature["eligible_signer_set"]
    assert "immutable" in signature["eligible_signer_set"]
    assert "any two distinct eligible keys" in signature["signer_selection"]
    assert "only by the trusted signing authority" in signature["signer_selection"]
    assert "never the required pair" in signature["signer_selection"]
    assert (
        "authorized_signer_ids = exact reserved three-key set"
        in signature["signing_service_request"]
    )
    assert "required_threshold = 2" in signature["signing_service_request"]
    assert "exact first durable response" in signature["service_retry"]
    assert "selected signer IDs and exact signature bytes" in signature["service_retry"]
    assert "lost response, issuer crash or service restart" in signature["service_retry"]
    assert "without reselecting the pair" in signature["service_retry"]
    assert (
        "exact reserved/current Production Trust identity and key set"
        in signature["service_authority"]
    )
    profile = _schema()["properties"]["signatures"]
    assert (profile["minItems"], profile["maxItems"]) == (2, 2)
    assert set(profile["items"]["required"]) == {"algorithm", "key_id", "signature_hex"}


def test_identity_validity_and_retention_do_not_grant_authority_early() -> None:
    value = _load(CONTRACT)
    identity, validity, retained = (
        value["identity_issuance"],
        value["validity_policy"],
        value["retention"],
    )
    assert identity["owner"] == "ProductDeploymentSecurityAuthority"
    assert identity["caller_selected_identities"] is False
    assert identity["contains_secret"] is False
    assert "floor(internal Unix epoch milliseconds)" in identity["uuid_timestamp"]
    assert (
        "one captured sub-second-precision UTC reservation instant shared by both identities"
        in (identity["uuid_timestamp"])
    )
    assert (
        "do not derive milliseconds from issued_at_utc whole seconds" in identity["uuid_timestamp"]
    )
    assert "independent CSPRNG" in identity["uuid_randomness"]
    assert "penr_<RFC9562 UUIDv7>" in identity["enrollment_reference_profile"]
    assert validity["fixed_package_ttl_seconds"] is None
    assert validity["challenge_duration_is_package_ttl"] is False
    assert "same captured internal reservation UTC instant" in validity["issued_at_utc"]
    assert "truncated to whole UTC seconds" in validity["issued_at_utc"]
    assert validity["expires_at_utc"] == (
        "minimum(exact signed retained PDSAEnrollmentChallengeV1.expires_at_utc, "
        "exact retained verified TPMEnrollmentChallengeV1.expires_at_utc)"
    )
    assert "source deadline" in validity["final_commit_gate"]
    assert retained["states"] == ["RESERVED", "SIGNED", "COMMITTED"]
    assert retained["legal_transitions"] == [["RESERVED", "SIGNED"], ["SIGNED", "COMMITTED"]]
    assert retained["terminal_state"] == "COMMITTED"
    assert "all three authorized signer IDs and threshold 2" in retained["reserved"]
    assert "exact Production Trust identity/public key set" in retained["reserved"]
    assert (
        "no selected pair is required before trusted signing-service selection"
        in retained["reserved"]
    )
    assert "outside SQLite write transactions" in retained["signing"]
    assert "exact selected signer IDs" in retained["signed"]
    assert (
        "SIGNED/COMMITTED retry requires the same selected pair and signature bytes"
        in retained["signed"]
    )
    assert "no publication" in retained["signed"]
    assert "atomically" in retained["committed"]
    assert "exact immutable joins" in retained["retained_field_resolution"]
    assert {
        "production_trust_identity",
        "production_trust_key_set",
        "authorized_signer_ids",
        "required_threshold",
        "selected_signer_ids",
    } <= set(retained["minimum_retained_fields"])
    assert "exact first durable selected pair and signature bytes" in retained["crash_recovery"]
    assert "even before local SIGNED retention" in retained["crash_recovery"]
    assert (
        "fail closed when populated legacy issuance storage"
        in retained["legacy_fixed_pair_storage"]
    )
    assert (
        "never reconstruct or rebind retained authority to current trust automatically"
        in (retained["legacy_fixed_pair_storage"])
    )
    assert retained["same_request_retry"] == "RETURN_EXACT_SAME_PACKAGE_BYTES"
    assert retained["same_challenge_different_request"] == "REJECT_CHALLENGE_REPLAY_CONFLICT"
    assert value["target_binding"]["required_live_proofs"] == [
        "TPM_ATTESTATION_EXCHANGE_REVERIFIED",
        "PRE_ENROLLMENT_REQUEST_KEY_PROOF_OF_POSSESSION_VERIFIED",
    ]
    assert value["target_binding"]["raw_json_or_public_digest_establishes_authority"] is False


@pytest.mark.parametrize("scope", ["envelope", "payload", "signature"])
def test_schema_rejects_unknown_fields_at_every_envelope_level(scope: str) -> None:
    document = _document()
    target = document
    if scope == "payload":
        target = document["payload"]
    elif scope == "signature":
        target = document["signatures"][0]
    target["unknown"] = "forbidden"
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(document, _schema())


@pytest.mark.parametrize("field", sorted(executable.PACKAGE_FIELDS))
def test_schema_rejects_every_missing_payload_field(field: str) -> None:
    document = _document()
    del document["payload"][field]
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(document, _schema())


@pytest.mark.parametrize(
    ("field", "invalid"),
    [
        ("provisioning_subject_id", "psub_01930000-0000-7000-8000-00000000000A"),
        ("provisioning_subject_id", "psub_01930000-0000-4000-8000-000000000001"),
        ("enrollment_reference", "caller-selected"),
        ("enrollment_reference", "penr_01930000-0000-7000-7000-000000000001"),
        ("authorization_generation", 2),
        ("authorization_generation", True),
        ("authorization_version", "1"),
        ("lineage_generation", 0),
        ("predecessor_package_digest_or_null", "ab" * 32),
        ("release_policy_generation", MAX_EXACT_INTEGER + 1),
        ("expires_at_utc", None),
        ("issued_at_utc", "2026-10-06T12:00:00+00:00"),
        ("issued_at_utc", "2026-10-06T12:00:00.000001Z"),
    ],
)
def test_schema_rejects_non_initial_or_noncanonical_fields(field: str, invalid: object) -> None:
    document = _document()
    document["payload"][field] = invalid
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(document, _schema())


@pytest.mark.parametrize("count", [0, 1, 3])
def test_schema_rejects_non_exact_quorum_envelopes(count: int) -> None:
    document = _document()
    document["signatures"] = [copy.deepcopy(document["signatures"][0]) for _ in range(count)]
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(document, _schema())


@pytest.mark.parametrize(
    "field",
    [
        "provisioning_subject_id",
        "enrollment_reference",
        "pdsa_challenge_id",
        "verified_tpm_exchange_reference",
        "verified_tpm_public_projection_id",
        "issued_at_utc",
        "expires_at_utc",
        "release_policy_digest_sha256",
    ],
)
def test_schema_rejects_trailing_newline_in_strict_wire_fields(field: str) -> None:
    document = _document()
    document["payload"][field] += "\n"
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(document, _schema())

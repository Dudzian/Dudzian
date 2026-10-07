"""Freeze initial LPPI operation wire, purpose domains and parent immutability."""

from __future__ import annotations

import hashlib
import json
import uuid
from pathlib import Path

import jsonschema
import pytest

from bot_core.licensing import lppi_authenticated_operation as operation
from bot_core.licensing.canonical import canonical_json_bytes
from deployment import windows_production_lppi_operation as lifecycle

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs/architecture/cryptohunter_product_architecture"
CONTRACT = DOCS / "stage9_lppi_authenticated_operation_binding_contract.json"
FREEZE = DOCS / "stage9_lppi_authenticated_operation_binding_freeze.json"
PARENT = DOCS / "stage9_external_provisioning_architecture_contract.json"
PARENT_SHA = "6538a75d8cf3a6e3cdab8f4a9022277a21d98e1a61923aa8d8eae9b80ae1d67d"


def _contract():
    return json.loads(CONTRACT.read_bytes())


def _document():
    millis = 1_791_288_000_789
    identifier = str(uuid.UUID(int=(millis << 80) | (7 << 76) | (0x123 << 64) | (2 << 62) | 1))
    return {
        "schema_version": 1,
        "environment": "PRODUCTION",
        "pdsa_trust_domain": "PDSA_PRODUCTION_2_OF_3_ED25519",
        "pdsa_package_digest_sha256": "ab" * 32,
        "provisioning_subject_id": "psub_" + identifier,
        "enrollment_reference": "penr_" + identifier,
        "provisioning_operation_id": "prvop_" + identifier,
        "binding_generation": 1,
        "created_at_utc": "2026-10-06T12:00:00.789Z",
    }


def test_exact_parent_and_subordinate_freeze_hashes():
    value, freeze = _contract(), json.loads(FREEZE.read_bytes())
    assert value["status"] == freeze["status"] == "FROZEN"
    assert value["parent_contract"]["historical_parent_bytes_unchanged"] is True
    assert value["parent_contract"]["sha256"] == freeze["parent_contract_sha256"] == PARENT_SHA
    assert hashlib.sha256(PARENT.read_bytes()).hexdigest() == PARENT_SHA
    assert freeze["artifacts"] == [
        {
            "canonical_artifact": CONTRACT.relative_to(ROOT).as_posix(),
            "canonical_schema_version": value["schema_version"],
            "sha256": hashlib.sha256(CONTRACT.read_bytes()).hexdigest(),
        }
    ]
    assert value["implementation_authorized"] is True
    assert freeze["implementation_architecturally_authorized"] is True
    assert value["production_provisioning_ready"] is False
    assert freeze["production_provisioning_ready"] is False
    assert value["stage_status"] == {
        "stage_9": "IN_PROGRESS",
        "windows_production_ready": "NOT_READY",
        "stage_10_production_lifecycle_live": "NOT_STARTED",
        "stage_10_prerequisite": "BLOCKED_UNTIL_LEGAL_ENROLLMENT",
    }
    assert value["live_qualification"] == {
        "WINDOWS_NATIVE_LPPI_AUTHENTICATED_OPERATION_QUALIFICATION": "NOT_RUN",
        "LEGAL_PRODUCTION_ENROLLMENT": "NOT_PERFORMED",
        "hosted_mock_tests_are_physical_qualification": False,
    }


def test_parent_exact_wire_domains_and_runtime_fields():
    value = _contract()
    parent = json.loads(PARENT.read_bytes())["operation_identity"]
    artifact = parent["lppi_authenticated_operation_binding"]
    fields = value["binding_artifact"]["canonical_payload_fields"]
    assert fields == artifact["canonical_payload_fields"]
    assert len(fields) == len(operation.BINDING_FIELDS) == 9
    assert set(fields) == operation.BINDING_FIELDS
    assert value["signature_profile"]["domain"].encode() + b"\0" == operation.DOMAIN
    for key in (
        "algorithm",
        "domain",
        "signed_bytes",
        "signature_encoding",
        "active_key_match_fields",
        "non_active_key_result",
    ):
        assert value["signature_profile"][key] == artifact["signature_profile"][key]
    identity = value["operation_identity"]
    assert identity["purpose_domain"] == parent["provisioning_operation_id"]["purpose_domain"]
    assert identity["purpose_domain"] == operation.PURPOSE_DOMAIN
    assert identity["owner_issuer"] == parent["provisioning_operation_id"]["owner_issuer"]
    assert identity["schema"] == parent["provisioning_operation_id"]["schema"]
    assert identity["caller_selectable"] is False
    assert identity["uuid_purpose_domain_input"] is False
    assert value["reservation_identity"]["purpose_is_uuid_derivation"] is False
    assert value["reservation_identity"]["purpose_retained"] is True
    ordering = value["durable_lifecycle"]["protocol_ordering"]
    start = parent["protocol_ordering"].index(ordering[0])
    assert parent["protocol_ordering"][start : start + len(ordering)] == ordering
    assert ordering[-1] == "LPPI_AUTHENTICATED_OPERATION_BINDING_COMMITTED"
    assert value["durable_lifecycle"]["status_progression"] == list(lifecycle.STATUSES)
    assert set(value["durable_lifecycle"]["storage"]["record_fields"]) == lifecycle._FIELDS


def test_closed_json_schema_and_exact_runtime_payload():
    value = _contract()
    artifact = value["binding_artifact"]
    schema = artifact["json_schema"]
    jsonschema.Draft202012Validator.check_schema(schema)
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == set(schema["properties"]) == operation.BINDING_FIELDS
    document = _document()
    jsonschema.validate(document, schema)
    binding = operation.LPPIAuthenticatedProvisioningOperationBindingV1.from_canonical_bytes(
        canonical_json_bytes(document)
    )
    assert binding.document == document
    for key, expected in artifact["required_values"].items():
        assert document[key] == schema["properties"][key]["const"] == expected
    example = value["operation_identity"]["example"]
    assert example["created_at_utc"] == document["created_at_utc"]
    assert (
        example["uuid_timestamp_milliseconds"]
        == uuid.UUID(document["provisioning_operation_id"][6:]).int >> 80
    )
    assert operation.authenticated_operation_signed_bytes(binding.canonical_bytes) == (
        operation.DOMAIN + hashlib.sha256(binding.canonical_bytes).digest()
    )


@pytest.mark.parametrize("field", sorted(_document()))
def test_schema_rejects_each_missing_field_and_unknown_extension(field):
    document = _document()
    del document[field]
    schema = _contract()["binding_artifact"]["json_schema"]
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(document, schema)
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(_document() | {"extension": True}, schema)


@pytest.mark.parametrize(
    "field,bad",
    [
        ("schema_version", 2),
        ("schema_version", True),
        ("environment", "TEST_ONLY"),
        ("pdsa_trust_domain", "TEST_ONLY_2_OF_3_ED25519"),
        ("pdsa_package_digest_sha256", "AB" * 32),
        ("pdsa_package_digest_sha256", "ab" * 32 + "\n"),
        ("provisioning_operation_id", "prvop_01999712-0000-4000-8000-000000000001"),
        ("provisioning_operation_id", "prvop_01999712-0000-7000-7000-000000000001"),
        ("provisioning_operation_id", "PRVOP_01999712-0000-7000-8000-000000000001"),
        ("binding_generation", 0),
        ("binding_generation", 2),
        ("binding_generation", True),
        ("created_at_utc", "2026-10-06T12:00:00Z"),
        ("created_at_utc", "2026-10-06T12:00:00.789123Z"),
        ("created_at_utc", "2026-10-06T12:00:00.789+00:00"),
        ("created_at_utc", "2026-10-06T12:00:60.789Z"),
        ("created_at_utc", "2026-10-06T12:00:00.789Z\n"),
    ],
)
def test_schema_rejects_wrong_initial_values_or_noncanonical_lexical_profiles(field, bad):
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(
            _document() | {field: bad}, _contract()["binding_artifact"]["json_schema"]
        )


def test_initial_scope_blocks_membership_cha_and_copied_authority():
    value = _contract()
    assert value["scope"]["mode"] == "INITIAL_OPERATION_ONLY"
    assert value["scope"]["source_authority"] == "VerifiedActiveLPPIAuthorityKey"
    assert value["scope"]["source_guard"] == "require_verified_active_lppi_authority_key"
    assert value["scope"]["replacement_supported"] is False
    assert value["scope"]["retirement_supported"] is False
    assert value["scope"]["caller_selected_identity_or_clock"] is False
    boundary = value["production_signing_boundary"]
    for field in (
        "arbitrary_key_constructor",
        "caller_callback_or_key_path",
        "public_arbitrary_domain_signing",
        "test_only_or_imported_key_authority",
        "authority_pop_domain_reuse",
    ):
        assert boundary[field] is False
    assert boundary["durable_reservation_guard"] == "require_reserved_operation_binding"
    assert boundary["exact_reserved_payload_required"] is True
    assert boundary["private_signing_lock_owner_required"] is True
    assert boundary["signing_lock_owner_scope"] == "CURRENT_PROCESS_AND_CURRENT_THREAD"
    assert boundary["standalone_native_sign_without_lock_owner"] == "REJECT"
    assert boundary["membership_issuance"] == "BLOCKED_UNTIL_CHA_ACCOUNT_COMMIT"
    assert value["operation_capability"]["public_constructor"] is False
    assert value["operation_capability"]["private_registry_required"] is True
    assert value["operation_capability"]["copy_subclass_or_object_new_as_authority"] is False
    assert set(value["scope_exclusions"]) >= {
        "CryptoHunterAccountAuthority",
        "logical_operation_id",
        "ago_*",
        "prvop_to_ago_bijection",
        "account_reservation",
        "account_genesis",
        "ProvisioningMembershipBinding",
        "ProtectedFreshnessAuthority",
        "SecretExternalResourcePort",
        "WindowsExternalProvisioningHandoff",
        "Stage10",
    }


def test_frozen_reservation_tuple_has_exact_purpose_separated_digest():
    value = _contract()
    fields = value["reservation_identity"]["canonical_tuple_fields"]
    assert len(fields) == len(set(fields)) == 8
    assert set(fields) == lifecycle._SOURCE_FIELDS
    source = {field: "ab" * 32 for field in fields}
    source.update(
        pdsa_trust_domain="PDSA_PRODUCTION_2_OF_3_ED25519",
        lppi_authority_public_key_algorithm_profile="ECDSA-P256-SHA256",
        custody_profile="WINDOWS_PLATFORM_CRYPTO_PROVIDER_TPM_ECDSA_P256_V1",
    )
    assert (
        operation.reservation_digest(source)
        == hashlib.sha256(
            value["reservation_identity"]["purpose_domain"].encode()
            + b"\0"
            + canonical_json_bytes(source)
        ).hexdigest()
    )


@pytest.mark.parametrize(
    "timestamp",
    [
        "2026-02-29T12:00:00.789Z",
        "2026-04-31T12:00:00.789Z",
        "2026-10-06T12:00:00.790Z",
    ],
)
def test_runtime_enforces_real_calendar_and_exact_frozen_uuid_timestamp(timestamp):
    with pytest.raises(operation.LPPIAuthenticatedOperationError):
        operation.LPPIAuthenticatedProvisioningOperationBindingV1.from_canonical_bytes(
            canonical_json_bytes(_document() | {"created_at_utc": timestamp})
        )


def test_standalone_native_boundary_requires_frozen_signing_lock_owner():
    boundary = _contract()["production_signing_boundary"]
    with pytest.raises(
        lifecycle.LPPIAuthenticatedOperationError,
        match=boundary["missing_signing_lock_owner_error"],
    ):
        lifecycle.require_reserved_operation_binding(canonical_json_bytes(_document()), object())

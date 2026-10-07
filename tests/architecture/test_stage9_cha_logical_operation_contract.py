"""Freeze the pre-reservation CHA boundary and its historical parent bytes."""

from __future__ import annotations

import ast
import hashlib
import inspect
import json
import textwrap
import uuid
from pathlib import Path

import jsonschema
import pytest

from bot_core.licensing import cha_logical_operation as capability
from bot_core.uuid7 import reservation_epoch_milliseconds
from deployment import windows_production_cha_operation as lifecycle

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs/architecture/cryptohunter_product_architecture"
CONTRACT = DOCS / "stage9_cha_logical_operation_contract.json"
FREEZE = DOCS / "stage9_cha_logical_operation_freeze.json"
PARENT = DOCS / "stage9_external_provisioning_architecture_contract.json"
UPSTREAM = DOCS / "stage9_lppi_authenticated_operation_binding_contract.json"
PARENT_SHA = "6538a75d8cf3a6e3cdab8f4a9022277a21d98e1a61923aa8d8eae9b80ae1d67d"
UPSTREAM_SHA = "f75494b163137374d39d9147b5e328aa30e772d2887255deb8980cb442457ca2"


def _contract():
    return json.loads(CONTRACT.read_bytes())


def _document():
    millis = 1_791_374_400_789
    identifier = str(uuid.UUID(int=(millis << 80) | (7 << 76) | (0x123 << 64) | (2 << 62) | 1))
    # The embedded upstream state has its own strict runtime verification.
    # This example tests only the record's closed lexical JSON schema.
    return {
        "schema_version": "CHAInitialLogicalOperationV1",
        "status": "PRVOP_AGO_BIJECTION_COMMITTED",
        "history": ["AGO_RESERVED", "PRVOP_AGO_BIJECTION_COMMITTED"],
        "mapping_generation": 1,
        "pdsa_trust_domain": "PDSA_PRODUCTION_2_OF_3_ED25519",
        "provisioning_operation_id": "prvop_" + identifier,
        "logical_operation_id": "ago_" + identifier,
        "assigned_at_utc": "2026-10-07T12:00:00.789Z",
        "lppi_operation_state_raw_hex": "7b7d",
    }


def _schema():
    return _contract()["durable_lifecycle"]["storage"]["json_schema"]


def test_exact_parent_upstream_and_subordinate_freeze_hashes():
    value, freeze = _contract(), json.loads(FREEZE.read_bytes())
    assert value["status"] == freeze["status"] == "FROZEN"
    assert value["parent_contract"]["historical_parent_bytes_unchanged"] is True
    assert value["upstream_contract"]["historical_upstream_bytes_unchanged"] is True
    assert value["parent_contract"]["sha256"] == freeze["parent_contract_sha256"] == PARENT_SHA
    assert (
        value["upstream_contract"]["sha256"] == freeze["upstream_contract_sha256"] == UPSTREAM_SHA
    )
    assert hashlib.sha256(PARENT.read_bytes()).hexdigest() == PARENT_SHA
    assert hashlib.sha256(UPSTREAM.read_bytes()).hexdigest() == UPSTREAM_SHA
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


def test_frozen_parent_ownership_bijection_and_protocol_ordering():
    value = _contract()
    parent = json.loads(PARENT.read_bytes())["operation_identity"]
    for field in ("owner_issuer", "schema", "caller_selectable"):
        assert value["operation_identity"][field] == parent["cha_logical_operation_id"][field]
    for field in (
        "cardinality",
        "registry_key",
        "same_operation_replay",
        "same_id_different_request",
        "new_operation_after_first_winner",
    ):
        assert value["bijection"][field] == parent["mapping"][field]
    assert value["durable_lifecycle"]["protocol_ordering"] == parent["protocol_ordering"][-3:]
    assert value["scope"]["completion"] == "PRVOP_AGO_BIJECTION_COMMITTED"
    assert value["scope"]["canonical_account_genesis_request"] is False
    assert value["bijection"]["cross_host_global_registry_provided"] is False
    assert value["bijection"]["subject_account_cardinality_selected"] is False


def test_closed_record_schema_and_runtime_state_parity():
    value = _contract()
    storage = value["durable_lifecycle"]["storage"]
    schema = _schema()
    jsonschema.Draft202012Validator.check_schema(schema)
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == set(schema["properties"]) == set(storage["record_fields"])
    assert set(storage["record_fields"]) == lifecycle._FIELDS
    assert len(storage["record_fields"]) == 9
    assert storage["maximum_state_bytes"] == lifecycle._MAX_BYTES
    assert (
        schema["properties"]["lppi_operation_state_raw_hex"]["maxLength"]
        == 2 * lifecycle._MAX_UPSTREAM_BYTES
    )
    assert value["durable_lifecycle"]["status_progression"] == list(lifecycle.STATUSES)
    assert schema["properties"]["schema_version"]["const"] == storage["schema_version"]
    assert (
        schema["properties"]["mapping_generation"]["const"]
        == value["scope"]["mapping_generation"]
        == 1
    )
    jsonschema.validate(_document(), schema)
    jsonschema.validate(
        _document() | {"status": "AGO_RESERVED", "history": ["AGO_RESERVED"]}, schema
    )
    # A lexically valid embedded snapshot is never an authorization proof.
    with pytest.raises((ValueError, RuntimeError)):
        lifecycle._validate_state(_document())


@pytest.mark.parametrize("field", sorted(_document()))
def test_schema_rejects_each_missing_field_and_extension(field):
    document = _document()
    del document[field]
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(document, _schema())
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(_document() | {"account_id": "acct_untrusted"}, _schema())


@pytest.mark.parametrize(
    "field,bad",
    [
        ("schema_version", 1),
        ("schema_version", "CHAInitialLogicalOperationV2"),
        ("status", "ACCOUNT_COMMITTED"),
        ("history", ["AGO_RESERVED"]),
        ("history", ["PRVOP_AGO_BIJECTION_COMMITTED", "AGO_RESERVED"]),
        ("mapping_generation", True),
        ("mapping_generation", 0),
        ("mapping_generation", 2),
        ("pdsa_trust_domain", "TEST_ONLY_2_OF_3_ED25519"),
        ("provisioning_operation_id", "prvop_01999712-0000-4000-8000-000000000001"),
        ("logical_operation_id", "ago_01999712-0000-4000-8000-000000000001"),
        ("logical_operation_id", "ago_01999712-0000-7000-7000-000000000001"),
        ("logical_operation_id", "AGO_01999712-0000-7000-8000-000000000001"),
        ("logical_operation_id", "ago_01999712-0000-7000-8000-000000000001\n"),
        ("assigned_at_utc", "2026-10-07T12:00:00Z"),
        ("assigned_at_utc", "2026-10-07T12:00:00.789123Z"),
        ("assigned_at_utc", "2026-10-07T12:00:00.789+00:00"),
        ("assigned_at_utc", "2026-10-07T12:00:60.789Z"),
        ("assigned_at_utc", "2026-10-07T12:00:00.789Z\n"),
        ("lppi_operation_state_raw_hex", ""),
        ("lppi_operation_state_raw_hex", "7B7D"),
        ("lppi_operation_state_raw_hex", "7b7d\n"),
        ("lppi_operation_state_raw_hex", "7b7"),
    ],
)
def test_schema_rejects_invalid_initial_values_and_noncanonical_profiles(field, bad):
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(_document() | {field: bad}, _schema())


def test_frozen_uuid_example_uses_shared_integer_millisecond_profile():
    from datetime import datetime

    identity = _contract()["operation_identity"]
    example = identity["example"]
    captured = datetime.fromisoformat(
        example["captured_assignment_instant_utc"].replace("Z", "+00:00")
    )
    assert reservation_epoch_milliseconds(captured) == example["uuid_timestamp_milliseconds"]
    document = _document()
    identifier = uuid.UUID(document["logical_operation_id"][4:])
    assert identifier.version == 7
    assert identifier.variant == uuid.RFC_4122
    assert identifier.int >> 80 == example["uuid_timestamp_milliseconds"]
    assert document["assigned_at_utc"] == example["assigned_at_utc"]
    assert identity["profile_source"] == "bot_core/uuid7.py"


def test_exact_source_tuple_and_capability_publication_contract():
    value = _contract()
    source_fields = value["source_binding"]["canonical_tuple_fields"]
    assert len(source_fields) == len(set(source_fields)) == 8
    tree = ast.parse(
        textwrap.dedent(inspect.getsource(capability.VerifiedCHALogicalOperation.source_tuple.fget))
    )
    source_literals = {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }
    assert set(source_fields) <= source_literals
    assert value["scope"]["source_authority"] == "VerifiedLPPIAuthenticatedProvisioningOperation"
    assert value["scope"]["source_guard"] == "require_verified_lppi_authenticated_operation"
    boundary = value["operation_capability"]
    assert boundary["type"] == capability.VerifiedCHALogicalOperation.__name__
    assert boundary["source_guard"] == capability.require_verified_cha_logical_operation.__name__
    for field in ("public_constructor", "copy_subclass_or_object_new_as_authority"):
        assert boundary[field] is False
    for field in ("exact_type_required", "private_registry_required", "immutable_snapshot"):
        assert boundary[field] is True
    assert set(boundary["properties"]) == {
        name
        for name, member in vars(capability.VerifiedCHALogicalOperation).items()
        if isinstance(member, property)
    }
    api = value["production_api"]
    assert inspect.signature(getattr(lifecycle, api["establish"])).parameters.keys() == {"upstream"}
    assert inspect.signature(getattr(lifecycle, api["load"])).parameters.keys() == {"upstream"}


def test_production_modules_stop_before_account_and_independent_effects():
    forbidden_modules = {
        "external_provisioning",
        "cha_attempt_store",
        "account_genesis",
        "root_proof",
        "protected_freshness",
        "secret_external_resource",
        "windows_external_provisioning_handoff",
    }
    forbidden_names = {
        "Stage9AccountGenesisAuthority",
        "Stage9ProvisioningService",
        "ProvisioningRepository",
        "ProvisioningMembershipBinding",
        "ProtectedFreshnessAuthorityPort",
        "SecretExternalResourcePort",
        "TestOnlyMembershipSigner",
        "AccountGenesisRecordV1",
    }
    for module in (capability, lifecycle):
        tree = ast.parse(inspect.getsource(module))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert not any(part in (node.module or "") for part in forbidden_modules)
                assert not ({alias.name for alias in node.names} & forbidden_names)
            if isinstance(node, ast.Import):
                assert not any(
                    part in alias.name for alias in node.names for part in forbidden_modules
                )
            if isinstance(node, ast.Name):
                assert node.id not in forbidden_names
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                assert node.value not in {
                    "acct_",
                    "devinst_",
                    "CryptoHunter.Stage9.ProvisioningMembershipBinding.v1",
                }
    value = _contract()
    assert set(value["scope_exclusions"]) >= {
        "account_id_mint",
        "account_reservation",
        "account_genesis",
        "first_device_membership",
        "ProvisioningMembershipBinding",
        "ProtectedFreshnessAuthority",
        "TPM_NV_counter",
        "SecretExternalResourcePort",
        "WindowsExternalProvisioningHandoff",
        "Stage10",
    }
    assert value["cryptographic_boundary"]["new_cha_signer_required"] is False
    assert value["cryptographic_boundary"]["protected_freshness_provided"] is False
    assert (
        value["durable_lifecycle"]["storage"]["disk_rollback_protection"]
        == "NOT PROVIDED BY THIS LAYER"
    )


def test_current_status_preserves_incomplete_production_scope():
    value = _contract()
    current = json.loads((ROOT / "deployment/stage9_current_status.json").read_bytes())
    for key, expected in value["expected_status"].items():
        assert current[key.lower()] == expected
    assert current["stage_9"] == "IN_PROGRESS"
    assert current["production_provisioning_ready"] is False
    assert current["windows_production_ready"] == "NOT_READY"
    assert current["stage_10_production_lifecycle_live"] == "NOT_STARTED"
    assert current["stage_10_prerequisite"] == "BLOCKED_UNTIL_LEGAL_ENROLLMENT"
    assert current["production_root_material"] == "PROVISIONED_LOCALLY"
    assert current["production_ceremony"] == "COMPLETE"
    assert value["live_qualification"] == {
        "WINDOWS_NATIVE_CHA_LOGICAL_OPERATION_QUALIFICATION": "NOT_RUN",
        "LEGAL_PRODUCTION_ENROLLMENT": "NOT_PERFORMED",
        "hosted_mock_tests_are_physical_qualification": False,
    }

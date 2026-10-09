"""Frozen Stage 9 reservation reconciliation, provenance and scope boundaries."""

from __future__ import annotations

import ast
import hashlib
import inspect
import json
from copy import copy, deepcopy
from dataclasses import fields
from pathlib import Path

import pytest

from bot_core import cha_attempt_store as store
from bot_core.licensing import cha_root_proof_attempt_reservation as boundary
from bot_core.licensing.canonical import canonical_json_bytes, parse_canonical
from bot_core.licensing.cha_account_reservation import AccountInitialBindingError
from bot_core.postgresql_root_proof_issuance_authority import (
    PostgreSQLRootProofIssuanceAuthority,
)

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs/architecture/cryptohunter_product_architecture"
CONTRACT_PATH = DOCS / "stage9_root_proof_attempt_reservation_contract.json"
FREEZE_PATH = DOCS / "stage9_root_proof_attempt_reservation_freeze.json"
CURRENT_STATUS_PATH = ROOT / "deployment/stage9_current_status.json"
CONTRACT = json.loads(CONTRACT_PATH.read_bytes())
PREACCOUNT_CONTRACT = json.loads(
    (DOCS / "stage9_root_proof_preaccount_credentials_contract.json").read_bytes()
)
STAGE_STATUS = {
    "local_cha_attempt_reservation_boundary": "IMPLEMENTED",
    "root_proof_issuance_authorization_boundary": (
        "IMPLEMENTED_AS_LOGIC / BLOCKED_ON_GENUINE_PRODUCTION_PROVIDER"
    ),
    "semantic_root_proof_issuer_runtime": "NOT_IMPLEMENTED / BLOCKED",
    "requester_claimant_production_provider": "NOT_IMPLEMENTED / BLOCKED",
    "root_proof": "NOT_ISSUED",
    "root_proof_admission": "NOT_STARTED",
    "account_id": "CANDIDATE_RESERVED_NOT_GENUINE",
    "account_genesis": "INITIAL_BINDING_ONLY",
    "prepared": "NOT_STARTED",
    "production_provisioning_ready": False,
    "windows_production_ready": "NOT_READY",
    "legal_production_enrollment": "NOT_PERFORMED",
    "stage_10_production_lifecycle_live": "NOT_STARTED",
    "provisioning_membership": "BLOCKED",
    "protected_freshness": "NOT_STARTED",
    "secret_resource": "NOT_STARTED",
}
HISTORICAL_PROFILE_STATUS_FIELDS = {
    "production_local_historical_implemented",
    "production_local_historical_deployment_available_now",
}
PROTECTED_HASHES = {
    "stage9_account_initial_binding_contract.json": (
        "c2a5e51df106e6c762a735fd9daef54a751187e6e7d56a4743ace5bb7fa1f255"
    ),
    "stage9_account_initial_binding_freeze.json": (
        "da6d1b43512ef5f7492de6b23cb55ba3b33123f486a7dede47f4c0968a7a6446"
    ),
    "stage9_cha_logical_operation_contract.json": (
        "f93964523631033a65822fa834e55326e813755376553d21ebb16eb4a047286c"
    ),
    "stage9_cha_logical_operation_freeze.json": (
        "8519750b62fa6515e3896383b8dbbf4bb20f44678c6a8dacceaf1df33173605e"
    ),
    "stage9_external_provisioning_architecture_contract.json": (
        "6538a75d8cf3a6e3cdab8f4a9022277a21d98e1a61923aa8d8eae9b80ae1d67d"
    ),
    "stage9_external_provisioning_architecture_freeze.json": (
        "e4bbae5f9cd428317d413097f1413437046e0472f19a10051c5debdb873f6f6e"
    ),
    "m05_account_genesis_independent_root_proof_issuer_contract.json": (
        "36cd0d26f918769313c6f572541f40877b1f0ca9808a5fc69a9dac3a6bf6d30d"
    ),
    "m05_account_genesis_independent_root_proof_issuer_contract.md": (
        "ce3758d7c9a7d96c9814f6bd6f8e655c7fe30d47f40ac04a36c04b4b7b2dc80c"
    ),
    "m05_account_genesis_root_proof_admission_binding_contract.json": (
        "ce4b6a67ec085e9225d1cbbecba5ae934abe6f7094a9ec2596705a4b26014e39"
    ),
    "m05_account_genesis_root_proof_admission_binding_contract.md": (
        "69157e14053363e7a86e073781bdc2fd178063ec22a87ebd2baa2ace490c4f9a"
    ),
    "m05_account_genesis_physical_persistence_crash_atomicity_contract.json": (
        "190bce7d24bf000e1e71d114c6e8d5ab46db326cc8a2afa484171b692ffcc630"
    ),
    "m05_account_genesis_physical_persistence_crash_atomicity_contract.md": (
        "4d96008158837ac57390d44ac305f12dc33bebc58bdda18127f32f71b0739998"
    ),
    "m05_account_genesis_operation_identity_request_binding_contract.json": (
        "63c8abf6847e78bea901770047fec816c252fa3ed1fbe1bc959791f1e4a0ba3d"
    ),
    "m05_account_genesis_operation_identity_request_binding_contract.md": (
        "495c02786101f93a4e14c523a284b81dd9eff91903dd9772683217b53c27b589"
    ),
    "m05_account_genesis_root_proof_issuer_production_substrate_selection_contract.json": (
        "68d2a71f8673cd0bd34c2eb3284f4bdacff530b4a81152cea4dd04a617457490"
    ),
    "m05_account_genesis_root_proof_issuer_production_substrate_selection_contract.md": (
        "66696fdd7ff972cd25d5f5c290385170eb95e93ffccc462c57cd96788317ee99"
    ),
}


def _assert_current_status_parity(contract, current):
    assert contract["current_status_authority"] == {
        "canonical_artifact": CURRENT_STATUS_PATH.relative_to(ROOT).as_posix(),
        "authority": "CURRENT_STATUS",
        "child_is_global_status_authority": False,
        "conflict_policy": "FAIL",
        "parity_fields": list(STAGE_STATUS),
    }
    assert current.get("schema") == "CryptoHunter.Stage9CurrentStatusV1"
    assert type(current.get("version")) is int and current["version"] == 1
    assert current.get("authority") == "CURRENT_STATUS"
    projected = contract["current_status"]
    # Preserve the exact #3086 snapshot. The subordinate credential contract
    # advances implemented provider availability without rewriting that snapshot.
    assert set(projected) == set(STAGE_STATUS) | HISTORICAL_PROFILE_STATUS_FIELDS
    for field, expected in STAGE_STATUS.items():
        assert field in current, f"CURRENT_STATUS missing {field}"
        assert type(projected[field]) is type(expected), field
        assert type(current[field]) is type(expected), field
        assert projected[field] == expected, field
        assert current[field] == PREACCOUNT_CONTRACT["current_status"].get(field, expected), field


def test_canonical_current_status_owns_every_child_boundary_and_readiness_field():
    current = json.loads(CURRENT_STATUS_PATH.read_bytes())
    _assert_current_status_parity(CONTRACT, current)
    assert current["windows_0_14"] == "10/15 DONE"
    assert current["stage_9"] == "IN_PROGRESS"
    assert (
        current["legal_production_enrollment"]
        == CONTRACT["live_qualification"]["legal_production_enrollment"]
        == "NOT_PERFORMED"
    )


@pytest.mark.parametrize("field", STAGE_STATUS)
@pytest.mark.parametrize("mutation", ["missing", "different"])
def test_parity_rejects_missing_or_conflicting_current_status_fields(field, mutation):
    current = json.loads(CURRENT_STATUS_PATH.read_bytes())
    if mutation == "missing":
        current.pop(field)
    else:
        current[field] = True if STAGE_STATUS[field] is False else "CONTRADICTORY_STATUS"
    with pytest.raises(AssertionError):
        _assert_current_status_parity(CONTRACT, current)


@pytest.mark.parametrize(
    "field,value",
    [
        ("semantic_root_proof_issuer_runtime", "IMPLEMENTED"),
        ("requester_claimant_production_provider", "IMPLEMENTED"),
        ("root_proof", "ISSUED"),
        ("prepared", "PREPARED"),
        ("account_id", "GENUINE"),
        ("account_genesis", "COMMITTED"),
        ("production_provisioning_ready", True),
        ("production_provisioning_ready", 0),
        ("windows_production_ready", "READY"),
        ("provisioning_membership", "READY"),
        ("protected_freshness", "STARTED"),
        ("stage_10_production_lifecycle_live", "STARTED"),
    ],
)
def test_parity_rejects_matching_premature_readiness_in_both_sources(field, value):
    contract = deepcopy(CONTRACT)
    current = json.loads(CURRENT_STATUS_PATH.read_bytes())
    contract["current_status"][field] = current[field] = value
    with pytest.raises(AssertionError):
        _assert_current_status_parity(contract, current)


@pytest.mark.parametrize("authority", [None, "LOCAL_BOUNDARY"])
def test_child_cannot_replace_canonical_current_status_authority(authority):
    current = json.loads(CURRENT_STATUS_PATH.read_bytes())
    current["authority"] = authority
    with pytest.raises(AssertionError):
        _assert_current_status_parity(CONTRACT, current)
    current["authority"] = "CURRENT_STATUS"
    contract = deepcopy(CONTRACT)
    contract["current_status_authority"]["child_is_global_status_authority"] = True
    with pytest.raises(AssertionError):
        _assert_current_status_parity(contract, current)


def _profile_payloads(raw):
    state = parse_canonical(raw)
    request = parse_canonical(bytes.fromhex(state["canonical_request_raw_hex"]))
    relation = {
        "schema_version": "InitialBindingReservationRelationV1",
        "environment": request["environment"],
        "pdsa_trust_domain": state["pdsa_trust_domain"],
        "logical_operation_id": state["logical_operation_id"],
        "account_id": state["account_id"],
        "reservation_relation": request["reservation_relation"],
        "canonical_genesis_request_fingerprint_sha256": state["canonical_request_sha256"],
        "initial_binding_sha256": state["initial_binding_sha256"],
        "retained_initial_binding_sha256": hashlib.sha256(raw).hexdigest(),
    }
    relation_profile = CONTRACT["reservation_identity"]["profile"]
    assert set(relation) == set(relation_profile["fields"])
    reservation_identity = (
        "ibr_"
        + hashlib.sha256(
            relation_profile["domain_literal"].encode("ascii")
            + b"\0"
            + canonical_json_bytes(relation)
        ).hexdigest()
    )
    reference = relation | {
        "schema_version": "RootProofInitialBindingReferenceV1",
        "reservation_identity": reservation_identity,
    }
    reference_profile = CONTRACT["initial_binding_reference"]["profile"]
    assert set(reference) == set(reference_profile["fields"])
    digest = hashlib.sha256(
        reference_profile["domain_literal"].encode("ascii")
        + b"\0"
        + canonical_json_bytes(reference)
    ).hexdigest()
    return relation, reference, reservation_identity, digest


def test_child_freeze_and_all_protected_historical_artifacts_are_byte_identical():
    freeze = json.loads(FREEZE_PATH.read_bytes())
    assert freeze["status"] == CONTRACT["status"] == "FROZEN"
    assert freeze["production_provisioning_ready"] is False
    assert freeze["implementation_architecturally_authorized"] is True
    assert freeze["artifacts"] == [
        {
            "canonical_artifact": CONTRACT_PATH.relative_to(ROOT).as_posix(),
            "canonical_schema_version": CONTRACT["schema_version"],
            "sha256": hashlib.sha256(CONTRACT_PATH.read_bytes()).hexdigest(),
        }
    ]
    assert freeze["protected_artifacts"] == [
        {
            "canonical_artifact": (DOCS / name).relative_to(ROOT).as_posix(),
            "sha256": expected,
        }
        for name, expected in PROTECTED_HASHES.items()
    ]
    for name, expected in PROTECTED_HASHES.items():
        assert hashlib.sha256((DOCS / name).read_bytes()).hexdigest() == expected
    for record in (CONTRACT["upstream"], *CONTRACT["historical_reconciliation"].values()):
        if isinstance(record, dict) and "canonical_artifact" in record:
            name = Path(record["canonical_artifact"]).name
            assert record["sha256"] == PROTECTED_HASHES[name]
            assert record["historical_bytes_unchanged"] is True


def test_upstream_recovery_and_reservation_semantics_are_preserved():
    upstream = json.loads((DOCS / "stage9_account_initial_binding_contract.json").read_bytes())
    child = CONTRACT["upstream"]
    for field in ("reservation_id", "recovery_model"):
        assert child[field] == upstream[field]
    assert child["recovery_key"] == upstream["registry_key"]
    assert child["reservation_id"] == "NONE"
    assert child["canonical_request_mutated"] is False
    assert child["initial_binding_sha256_semantics_changed"] is False
    identity = CONTRACT["reservation_identity"]
    assert identity["relation"] == upstream["request"]["reservation_relation"]
    for field in ("random_id_minted", "caller_selectable", "new_authority", "new_recovery_root"):
        assert identity[field] is False
    assert identity["profile"]["prefix"] == "ibr_"


def test_separate_reference_digest_reconciles_every_historical_authenticity_slot():
    historical = json.loads(
        (DOCS / "m05_account_genesis_independent_root_proof_issuer_contract.json").read_bytes()
    )
    authenticity = historical["initial_binding_authenticity"]
    reconciliation = CONTRACT["historical_reconciliation"]
    assert reconciliation["initial_binding_authenticity_model"] == authenticity["model"]
    assert reconciliation["required_digest_semantic_slots"] == authenticity["digest_covers"]
    profile = CONTRACT["initial_binding_reference"]["profile"]
    assert set(profile["fields"]) >= {
        "logical_operation_id",
        "account_id",
        "reservation_identity",
        "reservation_relation",
        "canonical_genesis_request_fingerprint_sha256",
        "environment",
        "pdsa_trust_domain",
        "initial_binding_sha256",
        "retained_initial_binding_sha256",
    }
    assert CONTRACT["initial_binding_reference"]["separate_from_initial_binding_sha256"] is True
    assert CONTRACT["initial_binding_reference"]["caller_supplied"] is False
    assert reconciliation["sha256_alone_is_authority_or_authentication"] is False
    assert (
        boundary._RESERVATION_RELATION_DOMAIN
        == CONTRACT["reservation_identity"]["profile"]["domain_literal"].encode() + b"\0"
    )
    assert boundary._INITIAL_BINDING_REFERENCE_DOMAIN == profile["domain_literal"].encode() + b"\0"
    assert boundary._RESERVATION_RELATION_DOMAIN != boundary._INITIAL_BINDING_REFERENCE_DOMAIN


def test_exact_JCS_conformance_vector_with_unicode_has_no_authority():
    vector = CONTRACT["conformance_vector"]
    raw = bytes.fromhex(vector["synthetic_binding_raw_hex"])
    relation, reference, identity, digest = _profile_payloads(raw)
    assert relation == vector["relation_payload"]
    assert reference == vector["reference_payload"]
    assert canonical_json_bytes(relation).decode() == vector["relation_canonical_jcs_utf8"]
    assert canonical_json_bytes(reference).decode() == vector["reference_canonical_jcs_utf8"]
    assert identity == vector["reservation_identity"]
    assert digest == vector["initial_binding_digest_sha256"]
    context = boundary._context_from_retained_binding(raw)
    assert context["reservation_identity"] == identity
    assert context["initial_binding_digest_sha256"] == digest
    assert context["initial_binding_reference"] == vector["initial_binding_reference"]
    assert context["initial_binding_reference"] == "initial-binding-v1:" + digest
    assert digest != relation["initial_binding_sha256"]
    assert vector["source_bytes_are_authority"] is False
    with pytest.raises(AccountInitialBindingError):
        boundary.resolve_root_proof_issuance_authorization(raw)


@pytest.mark.parametrize(
    "field,value",
    [
        ("pdsa_trust_domain", "another-trust-domain"),
        ("logical_operation_id", "ago_018f3e70-7b5b-7c21-8b9a-0123456789ac"),
        ("account_id", "acct_018f3e70-7b5c-7c21-8b9a-0123456789ac"),
        ("canonical_request_sha256", "33" * 32),
        ("initial_binding_sha256", "44" * 32),
        ("changed_retained_field", "different exact complete binding bytes"),
    ],
)
def test_changed_exact_relation_or_retained_binding_changes_both_identities(field, value):
    raw = bytes.fromhex(CONTRACT["conformance_vector"]["synthetic_binding_raw_hex"])
    before = boundary._context_from_retained_binding(raw)
    state = parse_canonical(raw) | {field: value}
    after = boundary._context_from_retained_binding(canonical_json_bytes(state))
    assert before["reservation_identity"] != after["reservation_identity"]
    assert before["initial_binding_digest_sha256"] != after["initial_binding_digest_sha256"]
    assert before["initial_binding_reference"] != after["initial_binding_reference"]
    assert before == boundary._context_from_retained_binding(raw)


def test_persisted_schema_authorization_and_domain_profiles_match_runtime():
    contract = CONTRACT["attempt_store"]
    assert contract["module"] == "bot_core/cha_attempt_store.py"
    assert contract["implementation"] == store.SQLiteCHAAttemptStore.__name__
    assert contract["schema_version"] == 5  # immutable historical schema freeze
    child = json.loads(
        (DOCS / "stage9_root_proof_signed_immutable_attempt_contract.json").read_bytes()
    )
    assert store._SCHEMA_VERSION == child["durability"]["schema_version"] == 6
    assert child["durability"]["migration"]["from_version"] == 5
    assert contract["previous_schema_version"] == 4
    assert [field.name for field in fields(store.AttemptAuthorization)] == contract[
        "authorization_fields"
    ]
    assert set(contract["stage9_required_fields"]) == {
        "reservation_identity",
        "reservation_relation",
        "initial_binding_sha256",
        "authorization_evidence_sha256",
    }
    assert [field.name for field in fields(store.AttemptReservation)] == contract[
        "reservation_record_view"
    ]["fields"]
    assert contract["state"] == store.AttemptState.RESERVED_AWAITING_SIGNATURES.value
    assert contract["current_attempt_fence"] == 1
    assert (
        store._IDEMPOTENCY_DOMAIN
        == contract["idempotency"]["domain_literal"].encode("ascii") + b"\0"
    )
    assert contract["idempotency"]["all_exact_authorization_fields_included"] is True
    assert contract["loader_mints"] is False
    assert contract["independent_anti_rollback"] is False


def test_provider_origin_and_live_evidence_are_separate_from_raw_dtos(monkeypatch):
    contract = CONTRACT["authorization"]
    assert contract["production_adapter_allowlist_empty"] is True
    assert boundary._TRUSTED_PROVIDER_TYPES == (PostgreSQLRootProofIssuanceAuthority,)
    assert contract["raw_attempt_authorization_grants_stage9_authority"] is False
    assert contract["caller_entitlement_requester_claimant_ids_accepted"] is False
    assert contract["runtime_behavior_without_genuine_adapter"] == (
        "FAIL_CLOSED_BEFORE_RESERVATION_MUTATION"
    )
    assert contract["provider_evidence"]["closed_fields"] == [
        "context",
        "resolution",
        "entitlement",
        "requester",
        "claimant",
        "providers",
    ]
    assert contract["provider_evidence"]["persisted_field"] == "authorization_evidence_sha256"
    for name in PREACCOUNT_CONTRACT["production_composition"]["deployment_config_names"]:
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(
        boundary.RootProofAttemptReservationError,
        match="MISSING_PRODUCTION_AUTHORITY_CONFIGURATION",
    ):
        boundary._issuance_authority_provider()
    environment = CONTRACT["environment_and_security_profile"]
    assert environment["semantic_environment"] == "PRODUCTION"
    assert environment["selected_provider_security_profile"] == "PRODUCTION_LOCAL"
    assert environment["semantic_environment_relabelled_to_security_profile"] is False
    assert environment["TEST_authorizes_PRODUCTION"] is False


@pytest.mark.parametrize(
    "capability,guard",
    [
        (
            boundary.VerifiedRootProofIssuanceAuthorization,
            boundary.require_verified_root_proof_issuance_authorization,
        ),
        (
            boundary.VerifiedRootProofIssuanceAttemptReservation,
            boundary.require_verified_root_proof_issuance_attempt_reservation,
        ),
    ],
)
def test_capability_exact_type_private_provenance_and_copy_boundaries(capability, guard):
    with pytest.raises(TypeError):
        capability()
    with pytest.raises(TypeError):
        type("ForgedSubclass", (capability,), {})
    forged = object.__new__(capability)
    for copier in (copy, deepcopy):
        with pytest.raises(TypeError):
            copier(forged)
    for raw in (
        forged,
        {"issuance_attempt_id": "rpa_018f3e70-7b5c-7c21-8b9a-0123456789ab"},
        "rpa_018f3e70-7b5c-7c21-8b9a-0123456789ab",
        object(),
    ):
        with pytest.raises(boundary.RootProofAttemptReservationError):
            guard(raw)


def test_public_api_has_only_verified_boundary_inputs_and_live_properties():
    api = CONTRACT["production_api"]
    assert list(inspect.signature(getattr(boundary, api["resolve_authorization"])).parameters) == [
        "binding"
    ]
    for operation in ("reserve", "load"):
        assert list(inspect.signature(getattr(boundary, api[operation])).parameters) == [
            "binding",
            "authorization",
        ]
    for operation in ("require_authorization", "require_reservation"):
        assert list(inspect.signature(getattr(boundary, api[operation])).parameters) == ["value"]
    for field, capability in (
        ("authorization", boundary.VerifiedRootProofIssuanceAuthorization),
        ("reservation_capability", boundary.VerifiedRootProofIssuanceAttemptReservation),
    ):
        assert CONTRACT[field]["type"] == capability.__name__
        assert CONTRACT[field]["public_constructor"] is False
        assert CONTRACT[field]["exact_type_required"] is True
        assert CONTRACT[field]["private_provenance_required"] is True
        assert set(CONTRACT[field]["properties"]) == {
            name for name, value in vars(capability).items() if isinstance(value, property)
        }
    loader = ast.parse(inspect.getsource(boundary.load_root_proof_issuance_attempt))
    assert not any(
        isinstance(node, ast.Attribute) and node.attr == "reserve_or_resolve_attempt_id"
        for node in ast.walk(loader)
    )


def test_production_boundary_has_no_downstream_signing_issuer_or_account_effect():
    tree = ast.parse(inspect.getsource(boundary))
    forbidden_modules = {
        "requests",
        "httpx",
        "socket",
        "urllib",
        "stage10",
        "stage_10",
        "account_genesis",
        "protected_freshness",
        "secret_external_resource",
        "production_pdsa_signing",
        "root_proof_signing_custody",
        "external_handoff",
    }
    forbidden_names = {
        "AttemptIdentity",
        "BindRequest",
        "RootProofIssuer",
        "VerifiedRootProof",
        "RootProofAdmissionEvidence",
        "ProvisioningMembershipBinding",
        "ProtectedFreshnessAuthority",
        "SecretExternalResourcePort",
        "Stage9ProvisioningService",
    }
    forbidden_calls = {
        "finalize_attempt",
        "replace_current_attempt",
        "recover_attempt",
        "bind",
        "sign",
        "sign_request",
        "issue_root_proof",
        "send",
    }
    forbidden_values = {
        "SIGNED_IMMUTABLE_DURABLE_NOT_SENT",
        "PREPARED",
        "ACCOUNT_GENESIS_COMMITTED",
        "EXTERNAL_FRESHNESS_CAS",
        "rpf_",
        "devinst_",
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            assert not any(part in (node.module or "") for part in forbidden_modules)
            assert not ({alias.name for alias in node.names} & forbidden_names)
        elif isinstance(node, ast.Import):
            assert not any(part in alias.name for alias in node.names for part in forbidden_modules)
        elif isinstance(node, ast.Name):
            assert node.id not in forbidden_names
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute):
                assert node.func.attr not in forbidden_calls
            elif isinstance(node.func, ast.Name):
                assert node.func.id not in forbidden_calls
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            assert node.value not in forbidden_values
    uuid_tree = ast.parse(inspect.getsource(store._new_rpa_id))
    assert not any(isinstance(node, ast.BitAnd) for node in ast.walk(uuid_tree))
    assert (
        sum(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "now"
            for node in ast.walk(uuid_tree)
        )
        == CONTRACT["issuance_attempt_id"]["captured_internal_UTC_instants"]
    )


def test_status_keeps_production_provider_root_proof_and_prepared_blocked():
    status = CONTRACT["current_status"]
    historical = json.loads(
        (
            DOCS
            / "m05_account_genesis_root_proof_issuer_production_substrate_selection_contract.json"
        ).read_bytes()
    )["profiles"]["PRODUCTION_LOCAL"]
    assert status["local_cha_attempt_reservation_boundary"] == "IMPLEMENTED"
    assert status["semantic_root_proof_issuer_runtime"] == "NOT_IMPLEMENTED / BLOCKED"
    assert status["requester_claimant_production_provider"] == "NOT_IMPLEMENTED / BLOCKED"
    assert status["root_proof"] == "NOT_ISSUED"
    assert status["root_proof_admission"] == status["prepared"] == "NOT_STARTED"
    assert status["account_genesis"] == "INITIAL_BINDING_ONLY"
    assert status["production_provisioning_ready"] is False
    assert status["production_local_historical_implemented"] == historical["implemented"] is False
    assert (
        status["production_local_historical_deployment_available_now"]
        == historical["deployment_available_now"]
        is False
    )
    assert CONTRACT["live_qualification"]["legal_production_enrollment"] == "NOT_PERFORMED"
    assert set(CONTRACT["scope_exclusions"]) >= {
        "requester_signature",
        "claimant_signature",
        "immutable_signed_attempt_finalization",
        "external_RootProofIssuer_send",
        "entitlement_UNBOUND_to_BOUND",
        "root_proof",
        "RootProofAdmissionEvidence",
        "PREPARED",
        "Freshness_CAS",
        "final_AccountGenesis_COMMITTED",
        "membership",
        "device",
        "Secret_Resource",
        "Stage10",
    }


def test_architecture_and_durability_suite_remains_serial_in_CI():
    import yaml

    workflow = yaml.safe_load((ROOT / ".github/workflows/quality-security.yml").read_text())
    steps = workflow["jobs"]["quality-ratchet"]["steps"]
    relative_path = Path(__file__).resolve().relative_to(ROOT).as_posix()
    serial = next(step for step in steps if relative_path in step.get("run", ""))
    parallel = next(
        step for step in steps if step.get("name", "").startswith("Property and critical")
    )
    assert serial["name"].endswith("(serial)")
    assert relative_path in serial["run"]
    assert relative_path not in parallel["run"]
    assert "-n " not in serial["run"] and "xdist" not in serial["run"]
    assert "--cov-append" in serial["run"]
    assert "test_cha_*.py" in (ROOT / ".github/workflows/ci.yml").read_text()
    assert Path(__file__).name.startswith("test_cha_")

"""CCA original-byte signature checks are always qualification-only reports."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import FrozenInstanceError, replace
import hashlib
import json

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
import pytest

from bot_core import authenticated_custody_mutation_source as source
from bot_core import independent_cca_decision as cca
from bot_core import local_signing_custody as custody
from bot_core.licensing.canonical import canonical_json_bytes
from bot_core.root_proof_issuer_substrate import (
    ProviderRole,
    SecurityProfileIdentity,
    public_key_material_identity,
)
from tests.security._cca_decision_fixtures import b64, build_cca_case, sign_cca_payload
from tests.security.test_independent_custody_completion_qualification import (
    _publish,
    completion as completion,
)


@pytest.fixture
def cca_case(completion):
    return build_cca_case(completion.descriptor, completion.journal)


def _policy(document):
    return cca.parse_cca_verification_policy(canonical_json_bytes(document))


def _verify(decision, policy):
    return cca.verify_independent_cca_decision(decision, policy_loader=lambda: policy)


def _parse(case, payload):
    return cca.parse_independent_cca_decision(sign_cca_payload(case, payload))


def test_valid_original_signatures_remain_non_authorizing(cca_case):
    assert cca.CCA_DOMAIN == b"CryptoHunter/IndependentCustodyCompletion/V1\0"
    assert cca.CCA_DOMAIN != source.SOURCE_DOMAIN
    assert cca.CCA_SCHEMA == "cryptohunter.independent-custody-completion"
    assert cca.CCA_ENVELOPE_SCHEMA == "cryptohunter.signed-independent-custody-completion"
    loaded = []

    def load():
        loaded.append(True)
        return _policy(cca_case.policy_document)

    report = cca.verify_independent_cca_decision(cca_case.decision, policy_loader=load)
    assert loaded == [True, True]
    assert report.decision.canonical_bytes == cca_case.envelope
    assert report.status == report.source_authorization == report.committed_close == "NOT_QUALIFIED"
    assert report.production_emission == "BLOCKED"
    assert report.signature == report.source_signature == "MATCHED_EXPLICIT_POLICY"
    assert report.execution == report.durability == "NOT_ATTESTED"
    assert len(report.blockers) == 11
    assert "SOURCE_EVENT_RETAINED_PROVENANCE_NOT_QUALIFIED" in report.blockers
    assert (
        report.policy_fingerprint_sha256
        == hashlib.sha256(cca_case.policy.canonical_bytes).hexdigest()
    )
    with pytest.raises(FrozenInstanceError):
        report.status = "QUALIFIED"
    with pytest.raises(TypeError):
        cca.IndependentCCADecisionVerification(cca_case.decision, "0" * 64, status="QUALIFIED")


def test_immutable_originals_and_fresh_snapshots(cca_case):
    decision, policy = cca_case.decision, cca_case.policy
    assert decision.payload_bytes == cca_case.payload_bytes
    assert decision.fingerprint_sha256 == hashlib.sha256(cca_case.envelope).hexdigest()
    assert policy.document == cca_case.policy_document
    decision.document["credential_id"] = "changed"
    decision.payload["installation"]["environment_id"] = "changed"
    policy.document["completion_authority"]["key_version"] = 99
    assert decision.payload == cca_case.payload
    assert decision.document == json.loads(cca_case.envelope)
    assert policy.document == cca_case.policy_document
    for item in (decision, policy):
        with pytest.raises(FrozenInstanceError):
            item.canonical_bytes = b"{}"


@pytest.mark.parametrize("kind", ["decision", "policy"])
@pytest.mark.parametrize(
    "raw",
    [
        b"",
        b"{}",
        b"null",
        b"[]",
        b"\xff",
        b'{"x":NaN}',
        b'{"x":Infinity}',
        b'{"x":1e999}',
        b'{"x":1.0}',
        b"{" * 2000,
        b'{"x":' + b"[" * 32 + b"0" + b"]" * 32 + b"}",
    ],
)
def test_malformed_nonfinite_and_deep_inputs_fail_closed(kind, raw):
    parser = (
        cca.parse_independent_cca_decision
        if kind == "decision"
        else cca.parse_cca_verification_policy
    )
    with pytest.raises(cca.IndependentCCAVerificationError):
        parser(raw)


@pytest.mark.parametrize("kind", ["decision", "policy", "payload"])
@pytest.mark.parametrize(
    "change", ["extra", "missing", "bool", "float", "duplicate", "noncanonical", "bytearray"]
)
def test_outer_and_payload_closed_schema(cca_case, kind, change):
    value = deepcopy(
        cca_case.payload
        if kind == "payload"
        else cca_case.policy_document
        if kind == "policy"
        else json.loads(cca_case.envelope)
    )
    if change == "extra":
        value["approved"] = True
    elif change == "missing":
        del value["schema"]
    elif change == "bool":
        value["version"] = True
    raw = canonical_json_bytes(value)
    if change == "float":
        raw = raw.replace(b'"version":1', b'"version":1.0')
    elif change == "duplicate":
        raw = raw[:-1] + b',"version":1}'
    elif change == "noncanonical":
        raw += b"\n"
    elif change == "bytearray":
        raw = bytearray(raw)
    if kind == "payload":
        if change == "bytearray":
            # The envelope carries exact decoded bytes, never a Python object.
            raw = b" " + bytes(raw)
        raw = sign_cca_payload(cca_case, payload_raw=raw)
    parser = (
        cca.parse_cca_verification_policy
        if kind == "policy"
        else cca.parse_independent_cca_decision
    )
    with pytest.raises(cca.IndependentCCAVerificationError):
        parser(raw)


@pytest.mark.parametrize("kind", ["decision", "policy", "payload"])
def test_explicit_document_limits_before_decoder(cca_case, kind, monkeypatch):
    limit = {
        "decision": cca.MAX_CCA_ENVELOPE_BYTES,
        "policy": cca.MAX_CCA_POLICY_BYTES,
        "payload": cca.MAX_CCA_PAYLOAD_BYTES,
    }[kind]
    raw = b" " * (limit + 1)
    if kind == "payload":
        raw = sign_cca_payload(cca_case, payload_raw=raw)

    decoded = []

    def forbidden(*args, **kwargs):
        decoded.append(True)
        raise AssertionError("oversize document reached recursive JSON decoder")

    # Oversized nested payload is rejected by bounded Base64 before decoding it;
    # its ordinary small outer envelope must still be decoded first.
    if kind != "payload":
        monkeypatch.setattr(json, "loads", forbidden)
    parser = (
        cca.parse_cca_verification_policy
        if kind == "policy"
        else cca.parse_independent_cca_decision
    )
    with pytest.raises(cca.IndependentCCAVerificationError):
        parser(raw)
    assert decoded == []


@pytest.mark.parametrize("field", ["payload_b64", "signature_b64"])
@pytest.mark.parametrize("suffix", ["\n", "=", " "])
def test_envelope_base64_is_canonical(cca_case, field, suffix):
    envelope = json.loads(cca_case.envelope)
    envelope[field] += suffix
    with pytest.raises(cca.IndependentCCAVerificationError):
        cca.parse_independent_cca_decision(canonical_json_bytes(envelope))


@pytest.mark.parametrize(
    "field", ["credential_id", "key_version", "schema", "version", "signature_b64"]
)
def test_envelope_metadata_cannot_change_signed_interpretation(cca_case, field):
    envelope = json.loads(cca_case.envelope)
    envelope[field] = {
        "credential_id": "source-key",
        "key_version": 2,
        "schema": source.SOURCE_ENVELOPE_SCHEMA,
        "version": 2,
        "signature_b64": b64(b"a" * 63),
    }[field]
    with pytest.raises(cca.IndependentCCAVerificationError):
        cca.parse_independent_cca_decision(canonical_json_bytes(envelope))


@pytest.mark.parametrize(
    "field",
    [
        "schema",
        "domain",
        "source_authentication_result",
        "execution_result",
        "durability_result",
        "completion_result",
    ],
)
def test_payload_domains_and_result_lists_are_closed(cca_case, field):
    payload = deepcopy(cca_case.payload)
    payload[field] = "QUALIFIED" if field != "domain" else source.SOURCE_DOMAIN[:-1].decode()
    with pytest.raises(cca.IndependentCCAVerificationError):
        _parse(cca_case, payload)


@pytest.mark.parametrize(
    "field",
    [
        "installation",
        "database",
        "custody",
        "close",
        "originals",
        "close_observation",
        "decision_log_position",
        "completion_authority",
    ],
)
def test_nested_field_lists_are_closed(cca_case, field):
    payload = deepcopy(cca_case.payload)
    payload[field]["approved"] = True
    with pytest.raises(cca.IndependentCCAVerificationError):
        _parse(cca_case, payload)


@pytest.mark.parametrize(
    "section,field,value",
    [
        ("installation", "environment_id", "other"),
        ("installation", "installation_id", "other"),
        ("installation", "trust_pin_sha256", "f" * 64),
        ("installation", "trust_pin_version", 2),
        ("database", "system_identifier", "101"),
        ("database", "database_oid", 201),
        ("database", "connection_host", "/other"),
        ("database", "catalog_sha256", "f" * 64),
        ("custody", "semantic_role", "HISTORY_ATTESTATION_SIGNING"),
        ("custody", "key_version", 2),
        ("custody", "predecessor_generation", 2),
        ("custody", "successor_generation", 3),
        ("close", "operation_id", "other"),
        ("close", "close_xid", "101"),
        ("close", "request_b64", b64(b"{}")),
        ("originals", "operation_descriptor_b64", b64(b"{}")),
        ("originals", "complete_journal_sha256", "f" * 64),
    ],
)
def test_all_exact_shared_source_bindings_reject_substitution(cca_case, section, field, value):
    payload = deepcopy(cca_case.payload)
    payload[section][field] = value
    with pytest.raises(cca.IndependentCCAVerificationError):
        _parse(cca_case, payload)


@pytest.mark.parametrize(
    "field",
    [
        "semantic_role",
        "lifecycle_state",
        "os_uid",
        "os_gid",
        "key_version",
        "lifecycle_generation",
        "trust_epoch",
    ],
)
def test_cca_authority_types_role_and_lifecycle_are_closed(cca_case, field):
    payload = deepcopy(cca_case.payload)
    payload["completion_authority"][field] = {
        "semantic_role": source.ROOT_SOURCE_ROLE,
        "lifecycle_state": "REVOKED",
        "os_uid": True,
        "os_gid": 0,
        "key_version": 0,
        "lifecycle_generation": False,
        "trust_epoch": 0,
    }[field]
    with pytest.raises(cca.IndependentCCAVerificationError):
        _parse(cca_case, payload)


@pytest.mark.parametrize(
    "field",
    [
        "authority_id",
        "service_id",
        "os_uid",
        "os_gid",
        "endpoint",
        "provider_namespace",
        "credential_identity",
        "public_material_identity",
    ],
)
def test_source_cannot_alias_cca(cca_case, field):
    payload = deepcopy(cca_case.payload)
    payload["completion_authority"][field] = cca_case.source_case.payload["executor"][field]
    if field == "public_material_identity":
        payload["completion_authority"]["public_key_b64"] = cca_case.source_case.payload[
            "executor"
        ]["public_key_b64"]
    with pytest.raises(cca.IndependentCCAVerificationError):
        _parse(cca_case, payload)


@pytest.mark.parametrize(
    "field,source_field",
    [
        ("provider_namespace", "authority_identity"),
        ("credential_identity", "credential_identity"),
        ("public_material_identity", "public_material_identity"),
    ],
)
def test_signer_cannot_alias_cca(cca_case, field, source_field):
    payload = deepcopy(cca_case.payload)
    payload["completion_authority"][field] = payload["custody"][source_field]
    if field == "public_material_identity":
        payload["completion_authority"]["public_key_b64"] = payload["custody"]["public_key_b64"]
    with pytest.raises(cca.IndependentCCAVerificationError):
        _parse(cca_case, payload)


@pytest.mark.parametrize(
    "field", ["reader_role", "reader_role_oid", "observation_xid", "qualified_catalog_sha256"]
)
def test_own_close_requires_independent_reader_and_catalog(cca_case, field):
    payload = deepcopy(cca_case.payload)
    payload["close_observation"][field] = (
        "d" * 64
        if field == "qualified_catalog_sha256"
        else cca_case.source_case.payload["close_observation"][field]
    )
    with pytest.raises(cca.IndependentCCAVerificationError):
        _parse(cca_case, payload)


@pytest.mark.parametrize(
    "field,value",
    [
        ("namespace", "other"),
        ("sequence", True),
        ("sequence", 0),
        ("sequence", 2),
        ("previous_entry_sha256", "e" * 64),
        ("entry_id", "other"),
    ],
)
def test_decision_log_identity_and_first_position_are_exact(cca_case, field, value):
    payload = deepcopy(cca_case.payload)
    payload["decision_log_position"][field] = value
    with pytest.raises(cca.IndependentCCAVerificationError):
        _parse(cca_case, payload)


@pytest.mark.parametrize("field", ["decision_id", "source_event_envelope_sha256"])
def test_decision_identity_and_retained_event_digest_are_bound(cca_case, field):
    payload = deepcopy(cca_case.payload)
    payload[field] = "other" if field == "decision_id" else "a" * 64
    with pytest.raises(cca.IndependentCCAVerificationError):
        _parse(cca_case, payload)


@pytest.mark.parametrize(
    "domain",
    [
        source.SOURCE_DOMAIN,
        cca.CCA_DOMAIN[:-1],
        cca.CCA_DOMAIN + b"\0",
        b"CryptoHunter/RootProof/V1\0",
        b"CryptoHunter/HistoryAttestation/V1\0",
    ],
)
def test_cross_domain_signatures_are_denied(cca_case, domain):
    decision = cca.parse_independent_cca_decision(sign_cca_payload(cca_case, domain=domain))
    with pytest.raises(cca.IndependentCCAVerificationError):
        _verify(decision, cca_case.policy)


def test_caller_key_is_not_trust_anchor(cca_case):
    key = Ed25519PrivateKey.generate()
    public = key.public_key().public_bytes(
        serialization.Encoding.Raw, serialization.PublicFormat.Raw
    )
    payload = deepcopy(cca_case.payload)
    payload["completion_authority"]["public_key_b64"] = b64(public)
    payload["completion_authority"]["public_material_identity"] = public_key_material_identity(
        public
    )
    decision = cca.parse_independent_cca_decision(sign_cca_payload(cca_case, payload, key=key))
    with pytest.raises(cca.IndependentCCAVerificationError):
        _verify(decision, cca_case.policy)
    with pytest.raises(TypeError):
        cca.verify_independent_cca_decision(
            decision, policy_loader=lambda: cca_case.policy, public_key=public
        )


def test_source_signature_is_verified_even_if_forged_envelope_digest_is_pinned(cca_case):
    payload = deepcopy(cca_case.payload)
    source_envelope = json.loads(cca_case.source_case.envelope)
    source_envelope["signature_b64"] = b64(b"x" * 64)
    raw = canonical_json_bytes(source_envelope)
    payload["source_event_envelope_b64"] = b64(raw)
    payload["source_event_envelope_sha256"] = hashlib.sha256(raw).hexdigest()
    policy = deepcopy(cca_case.policy_document)
    policy["source_event_envelope_sha256"] = payload["source_event_envelope_sha256"]
    decision = _parse(cca_case, payload)
    with pytest.raises(cca.IndependentCCAVerificationError):
        _verify(decision, _policy(policy))


@pytest.mark.parametrize(
    "field",
    [
        "authority_id",
        "service_id",
        "host_id",
        "os_uid",
        "os_gid",
        "endpoint",
        "provider_namespace",
        "credential_identity",
        "key_version",
        "lifecycle_generation",
        "trust_epoch",
    ],
)
def test_exact_independent_authority_pin_is_required(cca_case, field):
    payload = deepcopy(cca_case.payload)
    current = payload["completion_authority"][field]
    payload["completion_authority"][field] = (
        current + 1 if type(current) is int else current + "-other"
    )
    if field == "provider_namespace":
        payload["decision_log_position"]["namespace"] = payload["completion_authority"][field]
    decision = _parse(cca_case, payload)
    with pytest.raises(cca.IndependentCCAVerificationError):
        _verify(decision, cca_case.policy)


@pytest.mark.parametrize(
    "field",
    ["os_uid", "os_gid", "provider_namespace", "credential_identity", "public_material_identity"],
)
@pytest.mark.parametrize("index", [1, 2, 3, 4])
def test_all_five_principals_must_be_disjoint(cca_case, field, index):
    policy = deepcopy(cca_case.policy_document)
    policy["principal_inventory"][index][field] = policy["principal_inventory"][0][field]
    with pytest.raises(cca.IndependentCCAVerificationError):
        _policy(policy)


@pytest.mark.parametrize(
    "change",
    [
        "missing",
        "extra",
        "duplicate_role",
        "unknown_role",
        "bad_group",
        "duplicate_group",
        "shared_primary",
        "shared_supplementary",
        "bool_uid",
    ],
)
def test_inventory_is_closed_and_groups_do_not_defeat_separation(cca_case, change):
    policy = deepcopy(cca_case.policy_document)
    records = policy["principal_inventory"]
    if change == "missing":
        records.pop()
    elif change == "extra":
        records[0]["approved"] = True
    elif change == "duplicate_role":
        records[-1]["semantic_role"] = records[0]["semantic_role"]
    elif change == "unknown_role":
        records[0]["semantic_role"] = "ROOT"
    elif change == "bad_group":
        records[0]["supplementary_gids"] = [True]
    elif change == "duplicate_group":
        records[0]["supplementary_gids"] = [3000, 3000]
    elif change == "shared_primary":
        records[0]["supplementary_gids"] = [records[1]["os_gid"]]
    elif change == "shared_supplementary":
        records[0]["supplementary_gids"] = records[1]["supplementary_gids"] = [3000]
    else:
        records[0]["os_uid"] = True
    with pytest.raises(cca.IndependentCCAVerificationError):
        _policy(policy)


@pytest.mark.parametrize(
    "field",
    [
        "source_verification_policy_b64",
        "source_event_envelope_sha256",
        "completion_authority",
        "close_observation",
        "decision_id",
        "decision_log_position",
    ],
)
def test_policy_requires_every_expected_binding(cca_case, field):
    policy = deepcopy(cca_case.policy_document)
    del policy[field]
    with pytest.raises(cca.IndependentCCAVerificationError):
        _policy(policy)


@pytest.mark.parametrize("failure", [None, b"{}", {}, object(), "raise"])
def test_missing_uncertain_or_wrong_policy_fails_closed(cca_case, failure):
    def loader():
        if failure == "raise":
            raise OSError("unavailable independent configuration")
        return failure

    with pytest.raises(cca.IndependentCCAVerificationError):
        cca.verify_independent_cca_decision(cca_case.decision, policy_loader=loader)


@pytest.mark.parametrize(
    "change", ["revoke", "rotate", "generation", "epoch", "close", "floor", "missing"]
)
def test_lifecycle_close_and_checkpoint_change_during_verification_fail_closed(cca_case, change):
    policy = deepcopy(cca_case.policy_document)
    source_policy = deepcopy(cca_case.source_policy)
    if change in {"revoke", "rotate", "generation", "epoch"}:
        field = {
            "revoke": "lifecycle_state",
            "rotate": "key_version",
            "generation": "lifecycle_generation",
            "epoch": "trust_epoch",
        }[change]
        value = "REVOKED" if change == "revoke" else 2
        policy["completion_authority"][field] = value
        source_policy["completion_authority"][field] = value
        policy["source_verification_policy_b64"] = b64(canonical_json_bytes(source_policy))
    elif change == "close":
        policy["close_observation"]["observation_xid"] = "103"
    elif change == "floor":
        policy["decision_log_position"]["sequence"] = 2
        policy["decision_log_position"]["previous_entry_sha256"] = "d" * 64
    snapshots = [cca_case.policy, policy]

    def loader():
        value = snapshots.pop(0)
        if type(value) is cca.ExplicitCCAVerificationPolicy:
            return value
        return None if change == "missing" else _policy(value)

    with pytest.raises(cca.IndependentCCAVerificationError):
        cca.verify_independent_cca_decision(cca_case.decision, policy_loader=loader)


@pytest.mark.parametrize("previous", ["a" * 64, "b" * 64])
def test_replay_old_sequence_and_substituted_previous_digest_are_denied(cca_case, previous):
    payload = deepcopy(cca_case.payload)
    payload["decision_log_position"].update(sequence=2, previous_entry_sha256=previous)
    decision = _parse(cca_case, payload)
    policy = deepcopy(cca_case.policy_document)
    policy["decision_log_position"].update(sequence=2, previous_entry_sha256="c" * 64)
    with pytest.raises(cca.IndependentCCAVerificationError):
        _verify(decision, _policy(policy))
    with pytest.raises(cca.IndependentCCAVerificationError):
        _verify(cca_case.decision, _policy(policy))


def test_original_payload_is_the_exact_signature_message(cca_case, monkeypatch):
    original_factory = cca.Ed25519PublicKey
    messages = []

    class Factory:
        @staticmethod
        def from_public_bytes(public):
            original = original_factory.from_public_bytes(public)

            class Verifier:
                def verify(self, signature, message):
                    messages.append(message)
                    return original.verify(signature, message)

            return Verifier()

    monkeypatch.setattr(cca, "Ed25519PublicKey", Factory)
    _verify(cca_case.decision, cca_case.policy)
    assert messages == [cca.CCA_DOMAIN + cca_case.payload_bytes]


def test_history_source_is_accepted_only_for_its_role(completion):
    role = ProviderRole.HISTORY_ATTESTATION_SIGNING
    security = SecurityProfileIdentity(
        completion.before.security_profile, completion.before.trust_domain
    )
    identity = custody._canonical_signing_identity(security, role, "f" * 32)
    before = replace(
        completion.before,
        provider_role=role,
        credential_semantic_role=custody._SEMANTIC_ROLE[role],
        credential_id=identity.credential_id,
        provider_namespace=identity.provider_namespace,
        key_handle=identity.key_handle,
        lifecycle_namespace=identity.lifecycle_namespace,
        protected_private_material_reference=identity.protected_private_material_reference,
    )
    root = completion.root / "history"
    root.mkdir(mode=0o700)
    operation = _publish(root, before)
    case = build_cca_case(operation.descriptor, operation.journal)
    assert _verify(case.decision, case.policy).status == "NOT_QUALIFIED"
    payload = deepcopy(case.payload)
    payload["custody"]["semantic_role"] = ProviderRole.ROOT_PROOF_SIGNING.value
    with pytest.raises(cca.IndependentCCAVerificationError):
        _parse(case, payload)


@pytest.mark.parametrize(
    "value", [None, "", "test-key", "sha256:" + "f" * 63, "sha256:" + "F" * 64]
)
def test_other_principal_material_identity_is_exact(cca_case, value):
    policy = deepcopy(cca_case.policy_document)
    policy["principal_inventory"][-1]["public_material_identity"] = value
    with pytest.raises(cca.IndependentCCAVerificationError):
        _policy(policy)


@pytest.mark.parametrize("value", [None, 1, {}, [1] * 33])
def test_principal_group_inventory_is_bounded_and_exact(cca_case, value):
    policy = deepcopy(cca_case.policy_document)
    policy["principal_inventory"][-1]["supplementary_gids"] = value
    with pytest.raises(cca.IndependentCCAVerificationError):
        _policy(policy)


def test_verifier_rejects_non_decision_objects(cca_case):
    with pytest.raises(cca.IndependentCCAVerificationError):
        _verify(cca_case.envelope, cca_case.policy)


def test_policy_schema_cannot_be_reused_from_source(cca_case):
    policy = deepcopy(cca_case.policy_document)
    policy["schema"] = source.SOURCE_POLICY_SCHEMA
    with pytest.raises(cca.IndependentCCAVerificationError):
        _policy(policy)


@pytest.mark.parametrize("kind", ["decision", "policy"])
def test_json_depth_limit_precedes_recursive_decoder(kind, monkeypatch):
    decoded = []

    def forbidden(*args, **kwargs):
        decoded.append(True)
        raise AssertionError("deep JSON reached recursive decoder")

    monkeypatch.setattr(json, "loads", forbidden)
    raw = b'{"deep":' + b"[" * 17 + b"0" + b"]" * 17 + b"}"
    parser = (
        cca.parse_independent_cca_decision
        if kind == "decision"
        else cca.parse_cca_verification_policy
    )
    with pytest.raises(cca.IndependentCCAVerificationError):
        parser(raw)
    assert decoded == []

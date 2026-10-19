"""An explicit signature pin is a diagnostic, never an execution capability."""

from __future__ import annotations

import base64
from copy import deepcopy
from dataclasses import FrozenInstanceError, replace
import hashlib
import json

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
import pytest

from bot_core import authenticated_custody_mutation_source as source
from bot_core import local_signing_custody as custody
from bot_core import local_signing_lifecycle_evidence as lifecycle
from bot_core.licensing.canonical import canonical_json_bytes
from bot_core.root_proof_issuer_substrate import (
    ProviderRole,
    SecurityProfileIdentity,
    public_key_material_identity,
)
from tests.security._authenticated_custody_source_fixtures import build_source_case
from tests.security.test_independent_custody_completion_qualification import (
    _publish,
    completion as completion,
)


@pytest.fixture
def source_case(completion):
    return build_source_case(completion.descriptor, completion.journal)


def _b64(raw):
    return base64.b64encode(raw).decode("ascii")


def _signed_raw(case, payload=None, *, payload_raw=None, key=None, domain=None):
    if payload_raw is None:
        payload_raw = canonical_json_bytes(case.payload if payload is None else payload)
    envelope = json.loads(case.envelope)
    envelope["payload_b64"] = _b64(payload_raw)
    envelope["signature_b64"] = _b64(
        (case.private_key if key is None else key).sign(
            (source.SOURCE_DOMAIN if domain is None else domain) + payload_raw
        )
    )
    return canonical_json_bytes(envelope)


def _parse_signed(case, payload):
    return source.parse_custody_mutation_execution(_signed_raw(case, payload))


def _policy(document):
    return source.parse_source_verification_policy(canonical_json_bytes(document))


def _verify(event, policy):
    return source.verify_custody_mutation_execution(event, policy_loader=lambda: policy)


def _set_original(originals, name, raw):
    originals[name + "_b64"] = _b64(raw)
    originals[name + "_sha256"] = hashlib.sha256(raw).hexdigest()


def test_exact_signature_policy_match_remains_not_qualified(source_case):
    assert source.SOURCE_DOMAIN == b"CryptoHunter/CustodyMutationExecution/V1\0"
    assert source.SOURCE_SCHEMA == "cryptohunter.custody-mutation-execution"
    assert source.SOURCE_ENVELOPE_SCHEMA == "cryptohunter.signed-custody-mutation-execution"
    assert source.SOURCE_POLICY_SCHEMA == "cryptohunter.explicit-custody-source-verification-policy"
    loaded = []

    def load():
        loaded.append(True)
        return source.parse_source_verification_policy(source_case.policy.canonical_bytes)

    result = source.verify_custody_mutation_execution(source_case.event, policy_loader=load)
    assert len(loaded) == 2
    assert result.event == source_case.event
    assert (
        result.policy_fingerprint_sha256
        == hashlib.sha256(source_case.policy.canonical_bytes).hexdigest()
    )
    assert result.status == "NOT_QUALIFIED"
    assert result.production_emission == "BLOCKED"
    assert result.signature == "MATCHED_EXPLICIT_POLICY"
    assert result.source_authorization == "NOT_QUALIFIED"
    assert result.execution == "NOT_ATTESTED"
    assert result.durability == "NOT_ATTESTED"
    assert "CONTRACT_ADDENDUM_INTEGRATION_PENDING" in result.blockers
    with pytest.raises(FrozenInstanceError):
        result.status = "QUALIFIED"


def test_history_execution_uses_its_own_source_role_and_originals(completion):
    security = SecurityProfileIdentity(
        completion.before.security_profile, completion.before.trust_domain
    )
    role = ProviderRole.HISTORY_ATTESTATION_SIGNING
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
    case = build_source_case(operation.descriptor, operation.journal)
    assert case.event.payload["executor"]["semantic_role"] == "HISTORY_CUSTODY_EXECUTION_V1"
    assert case.event.payload["custody"]["semantic_role"] == role.value
    assert _verify(case.event, case.policy).signature == "MATCHED_EXPLICIT_POLICY"
    payload = deepcopy(case.payload)
    payload["executor"]["semantic_role"] = "ROOT_CUSTODY_EXECUTION_V1"
    with pytest.raises(source.SourceMutationVerificationError):
        _parse_signed(case, payload)


def test_event_and_policy_own_immutable_exact_originals(source_case):
    event, policy = source_case.event, source_case.policy
    assert event.canonical_bytes == source_case.envelope
    assert event.fingerprint_sha256 == hashlib.sha256(event.canonical_bytes).hexdigest()
    assert event.payload == source_case.payload
    assert policy.document == source_case.policy_document
    with pytest.raises(FrozenInstanceError):
        event.canonical_bytes = b"{}"
    with pytest.raises(FrozenInstanceError):
        policy.canonical_bytes = b"{}"
    event.document["credential_id"] = "mutated"
    event.payload["executor"]["credential_identity"] = "mutated"
    policy.document["installation"]["trust_pin_sha256"] = "0" * 64
    assert event.document == json.loads(source_case.envelope)
    assert event.payload == source_case.payload
    assert policy.document == source_case.policy_document


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
        b" " * (8 * 1024 * 1024 + 1),
    ],
)
@pytest.mark.parametrize("kind", ["event", "policy"])
def test_malformed_nonfinite_deep_or_unbounded_input_fails_closed(raw, kind):
    parse = (
        source.parse_custody_mutation_execution
        if kind == "event"
        else source.parse_source_verification_policy
    )
    with pytest.raises(source.SourceMutationVerificationError):
        parse(raw)


@pytest.mark.parametrize("kind", ["event", "policy"])
@pytest.mark.parametrize(
    "change", ["noncanonical", "duplicate", "extra", "missing", "bool", "float", "bytes_type"]
)
def test_outer_schemas_and_types_are_closed(source_case, kind, change):
    document = (
        json.loads(source_case.envelope)
        if kind == "event"
        else deepcopy(source_case.policy_document)
    )
    raw = canonical_json_bytes(document)
    if change == "noncanonical":
        raw += b"\n"
    elif change == "duplicate":
        raw = raw[:-1] + b',"version":1}'
    elif change == "float":
        raw = raw.replace(b'"version":1', b'"version":1.0')
    elif change == "bytes_type":
        raw = bytearray(raw)
    else:
        if change == "extra":
            document["source_authenticated"] = True
        elif change == "missing":
            del document["schema"]
        else:
            document["version"] = True
        raw = canonical_json_bytes(document)
    parse = (
        source.parse_custody_mutation_execution
        if kind == "event"
        else source.parse_source_verification_policy
    )
    with pytest.raises(source.SourceMutationVerificationError):
        parse(raw)


@pytest.mark.parametrize(
    "change", ["noncanonical", "duplicate", "float", "extra", "missing", "domain", "version"]
)
def test_signed_payload_must_itself_be_exact_canonical_v1(source_case, change):
    payload = deepcopy(source_case.payload)
    raw = canonical_json_bytes(payload)
    if change == "noncanonical":
        raw += b"\n"
    elif change == "duplicate":
        raw = raw[:-1] + b',"version":1}'
    elif change == "float":
        raw = raw.replace(b'"version":1', b'"version":1.0')
    else:
        if change == "extra":
            payload["production_emission"] = "AUTHORIZED"
        elif change == "missing":
            del payload["close"]
        elif change == "domain":
            payload["domain"] = "CryptoHunter/IndependentCustodyCompletion/V1\0"
        else:
            payload["version"] = True
        raw = canonical_json_bytes(payload)
    with pytest.raises(source.SourceMutationVerificationError):
        source.parse_custody_mutation_execution(_signed_raw(source_case, payload_raw=raw))


@pytest.mark.parametrize("field", ["payload_b64", "signature_b64"])
@pytest.mark.parametrize("change", ["padding", "whitespace", "alphabet", "empty"])
def test_envelope_base64_is_strict_rfc4648(source_case, field, change):
    envelope = json.loads(source_case.envelope)
    if change == "padding":
        envelope[field] += "="
    elif change == "whitespace":
        envelope[field] = "\n" + envelope[field]
    elif change == "alphabet":
        envelope[field] = "_" + envelope[field][1:]
    else:
        envelope[field] = ""
    with pytest.raises(source.SourceMutationVerificationError):
        source.parse_custody_mutation_execution(canonical_json_bytes(envelope))


@pytest.mark.parametrize(
    "record,field,value",
    [
        ("installation", "trust_pin_version", True),
        ("installation", "profile", "SERVER_READY"),
        ("installation", "trust_pin_sha256", "A" * 64),
        ("executor", "os_uid", True),
        ("executor", "os_gid", 0),
        ("executor", "key_version", True),
        ("executor", "lifecycle_generation", True),
        ("executor", "lifecycle_state", "VERIFY_ONLY"),
        ("executor", "public_key_b64", _b64(b"short")),
        ("executor", "host_id", "host\0alias"),
        ("database", "system_identifier", "01"),
        ("database", "system_identifier", str(2**64)),
        ("database", "database_oid", True),
        ("database", "connection_port", True),
        ("database", "schema_version", True),
        ("custody", "predecessor_generation", True),
        ("custody", "successor_generation", True),
        ("custody", "key_version", True),
        ("close", "close_xid", "01"),
        ("close", "phase", "PENDING"),
        ("close_observation", "reader_role_oid", True),
        ("close_observation", "observation_xid", "0"),
        ("source_log_position", "sequence", True),
        ("source_log_position", "sequence", 0),
    ],
)
def test_exact_nested_types_states_ranges_and_digests(source_case, record, field, value):
    payload = deepcopy(source_case.payload)
    payload[record][field] = value
    with pytest.raises(source.SourceMutationVerificationError):
        _parse_signed(source_case, payload)


@pytest.mark.parametrize(
    "record",
    [
        "installation",
        "executor",
        "database",
        "custody",
        "close",
        "close_observation",
        "originals",
        "source_log_position",
    ],
)
def test_all_payload_records_reject_unknown_fields(source_case, record):
    payload = deepcopy(source_case.payload)
    payload[record]["caller_authenticated"] = True
    with pytest.raises(source.SourceMutationVerificationError):
        _parse_signed(source_case, payload)


@pytest.mark.parametrize(
    "role",
    ["ROOT_PROOF_SIGNING", "HISTORY_ATTESTATION_SIGNING", "CUSTODY_COMPLETION_ATTESTATION_V1"],
)
def test_signer_or_completion_role_cannot_be_an_executor(source_case, role):
    payload = deepcopy(source_case.payload)
    payload["executor"]["semantic_role"] = role
    with pytest.raises(source.SourceMutationVerificationError):
        _parse_signed(source_case, payload)


def test_history_executor_cannot_execute_root_descriptor(source_case):
    payload = deepcopy(source_case.payload)
    payload["executor"]["semantic_role"] = "HISTORY_CUSTODY_EXECUTION_V1"
    with pytest.raises(source.SourceMutationVerificationError):
        _parse_signed(source_case, payload)


@pytest.mark.parametrize("alias", ["material", "credential", "namespace"])
def test_source_identity_cannot_alias_the_signing_custody(source_case, alias):
    payload = deepcopy(source_case.payload)
    executor, signer = payload["executor"], payload["custody"]
    if alias == "material":
        for field in ("public_key_b64", "public_material_identity"):
            executor[field] = signer[field]
    elif alias == "credential":
        executor["credential_identity"] = signer["credential_identity"]
    else:
        executor["provider_namespace"] = signer["authority_identity"]
        payload["source_log_position"]["namespace"] = executor["provider_namespace"]
    with pytest.raises(source.SourceMutationVerificationError):
        _parse_signed(source_case, payload)


@pytest.mark.parametrize(
    "other,alias",
    [
        ("executor", "material"),
        ("executor", "credential"),
        ("executor", "namespace"),
        ("executor", "uid"),
        ("signer", "material"),
        ("signer", "credential"),
        ("signer", "namespace"),
    ],
)
def test_completion_policy_identity_is_independent(source_case, other, alias):
    policy = deepcopy(source_case.policy_document)
    completion = policy["completion_authority"]
    other_record = source_case.payload["executor" if other == "executor" else "custody"]
    if alias == "material":
        for field in ("public_key_b64", "public_material_identity"):
            completion[field] = other_record[field]
    elif alias == "credential":
        completion["credential_identity"] = other_record["credential_identity"]
    elif alias == "namespace":
        completion["provider_namespace"] = other_record[
            "provider_namespace" if other == "executor" else "authority_identity"
        ]
    else:
        completion["os_uid"] = other_record["os_uid"]
    with pytest.raises(source.SourceMutationVerificationError):
        _policy(policy)


@pytest.mark.parametrize(
    "field,value",
    [
        ("semantic_role", "ROOT_CUSTODY_EXECUTION_V1"),
        ("lifecycle_state", "VERIFY_ONLY"),
        ("os_uid", True),
        ("os_gid", 0),
        ("key_version", True),
        ("public_material_identity", "sha256:" + "0" * 64),
    ],
)
def test_completion_policy_role_and_exact_lifecycle_are_required(source_case, field, value):
    policy = deepcopy(source_case.policy_document)
    policy["completion_authority"][field] = value
    with pytest.raises(source.SourceMutationVerificationError):
        _policy(policy)


def test_verification_requires_a_readable_explicit_policy(source_case):
    with pytest.raises(TypeError):
        source.verify_custody_mutation_execution(source_case.event)
    for loader in (lambda: None, lambda: source_case.policy_document):
        with pytest.raises(source.SourceMutationVerificationError):
            source.verify_custody_mutation_execution(source_case.event, policy_loader=loader)

    def unreadable():
        raise OSError("independent policy unavailable")

    with pytest.raises(source.SourceMutationVerificationError):
        source.verify_custody_mutation_execution(source_case.event, policy_loader=unreadable)


@pytest.mark.parametrize("change", ["rotated", "missing", "unreadable"])
def test_policy_must_be_unchanged_after_signature_verification(source_case, change):
    rotated = deepcopy(source_case.policy_document)
    rotated["installation"]["trust_pin_version"] += 1
    rotated = _policy(rotated)
    calls = []

    def loader():
        calls.append(True)
        if len(calls) == 1:
            return source_case.policy
        if change == "unreadable":
            raise OSError("policy removed during verification")
        return rotated if change == "rotated" else None

    with pytest.raises(source.SourceMutationVerificationError):
        source.verify_custody_mutation_execution(source_case.event, policy_loader=loader)
    assert len(calls) == 2


@pytest.mark.parametrize(
    "record,field,value",
    [
        ("installation", "environment_id", "different-environment"),
        ("installation", "trust_pin_sha256", "f" * 64),
        ("installation", "installation_sha256", "e" * 64),
        ("executor", "credential_identity", "rotated-source-credential"),
        ("executor", "key_version", 2),
        ("executor", "trust_epoch", 2),
        ("database", "catalog_sha256", "d" * 64),
        ("source_log_position", "sequence", 2),
    ],
)
def test_current_policy_rejects_stale_or_differently_pinned_event(
    source_case, record, field, value
):
    policy = deepcopy(source_case.policy_document)
    policy[record][field] = value
    if record == "database":
        policy["close_observation"]["qualified_catalog_sha256"] = value
    if record == "source_log_position":
        policy[record]["previous_entry_sha256"] = "a" * 64
    with pytest.raises(source.SourceMutationVerificationError):
        _verify(source_case.event, _policy(policy))


def test_expected_operation_id_and_execution_id_cannot_be_replayed(source_case):
    policy = deepcopy(source_case.policy_document)
    policy["execution_id"] = "next-execution"
    policy["source_log_position"]["entry_id"] = "next-execution"
    with pytest.raises(source.SourceMutationVerificationError):
        _verify(source_case.event, _policy(policy))
    policy = deepcopy(source_case.policy_document)
    descriptor = json.loads(base64.b64decode(policy["operation_descriptor_b64"]))
    descriptor["operation_id"] = "next-operation"
    policy["operation_descriptor_b64"] = _b64(canonical_json_bytes(descriptor))
    policy["close"]["operation_id"] = "next-operation"
    request = json.loads(base64.b64decode(policy["close"]["request_b64"]))
    request["operation_id"] = "next-operation"
    request_raw = json.dumps(request, sort_keys=True).encode()
    policy["close"]["request_b64"] = _b64(request_raw)
    policy["close"]["request_sha256"] = hashlib.sha256(request_raw).hexdigest()
    with pytest.raises(source.SourceMutationVerificationError):
        _verify(source_case.event, _policy(policy))


@pytest.mark.parametrize("change", ["entry_id", "namespace", "initial_previous", "next_previous"])
def test_source_log_position_is_bound_to_executor_and_execution(source_case, change):
    payload = deepcopy(source_case.payload)
    position = payload["source_log_position"]
    if change == "entry_id":
        position["entry_id"] = "other-execution"
    elif change == "namespace":
        position["namespace"] = "other-source-namespace"
    elif change == "initial_previous":
        position["sequence"] = 1
        position["previous_entry_sha256"] = "a" * 64
    else:
        position["sequence"] = 2
        position["previous_entry_sha256"] = "0" * 64
    with pytest.raises(source.SourceMutationVerificationError):
        _parse_signed(source_case, payload)


@pytest.mark.parametrize("change", ["payload", "domain", "credential", "key_version", "signature"])
def test_signature_covers_original_payload_domain_and_pinned_credential(source_case, change):
    envelope = json.loads(source_case.envelope)
    policy = source_case.policy
    if change == "payload":
        payload = deepcopy(source_case.payload)
        payload["executor"]["endpoint"] = "/another/source/socket"
        envelope["payload_b64"] = _b64(canonical_json_bytes(payload))
        policy_document = deepcopy(source_case.policy_document)
        policy_document["executor"]["endpoint"] = payload["executor"]["endpoint"]
        policy = _policy(policy_document)
    elif change == "domain":
        envelope = json.loads(
            _signed_raw(source_case, domain=b"CryptoHunter/IndependentCustodyCompletion/V1\0")
        )
    elif change == "credential":
        envelope["credential_id"] = "different-credential"
    elif change == "key_version":
        envelope["key_version"] += 1
    else:
        signature = bytearray(base64.b64decode(envelope["signature_b64"]))
        signature[0] ^= 1
        envelope["signature_b64"] = _b64(signature)
    with pytest.raises(source.SourceMutationVerificationError):
        event = source.parse_custody_mutation_execution(canonical_json_bytes(envelope))
        _verify(event, policy)


def test_self_supplied_valid_public_key_is_not_a_trust_anchor(source_case):
    private_key = Ed25519PrivateKey.from_private_bytes(b"\xdd" * 32)
    public_key = private_key.public_key().public_bytes(
        serialization.Encoding.Raw, serialization.PublicFormat.Raw
    )
    payload = deepcopy(source_case.payload)
    payload["executor"]["public_key_b64"] = _b64(public_key)
    payload["executor"]["public_material_identity"] = public_key_material_identity(public_key)
    event = source.parse_custody_mutation_execution(
        _signed_raw(source_case, payload, key=private_key)
    )
    with pytest.raises(source.SourceMutationVerificationError):
        _verify(event, source_case.policy)


@pytest.mark.parametrize(
    "change", ["missing_close", "intent", "unknown_execution", "unknown_durability", "same_xid"]
)
def test_incomplete_or_unknown_execution_never_gets_a_positive_event(source_case, change):
    payload = deepcopy(source_case.payload)
    if change == "missing_close":
        del payload["close"]
    elif change == "intent":
        journal = json.loads(base64.b64decode(payload["originals"]["complete_journal_b64"]))
        journal["completion_state"] = "INTENT"
        _set_original(payload["originals"], "complete_journal", canonical_json_bytes(journal))
    elif change == "unknown_execution":
        payload["execution_result"] = "UNKNOWN"
    elif change == "unknown_durability":
        payload["durability_result"] = "UNKNOWN"
    else:
        payload["close_observation"]["observation_xid"] = payload["close"]["close_xid"]
    with pytest.raises(source.SourceMutationVerificationError):
        _parse_signed(source_case, payload)


def test_original_descriptor_journal_metadata_and_postgresql_request_are_preserved(
    source_case, completion
):
    originals = source_case.event.payload["originals"]
    journal = json.loads(completion.journal)
    expected = {
        "operation_descriptor": completion.descriptor,
        "complete_journal": completion.journal,
        "before_metadata": canonical_json_bytes(journal["before_metadata"]),
        "after_metadata": canonical_json_bytes(journal["after_metadata"]),
    }
    for name, raw in expected.items():
        assert base64.b64decode(originals[name + "_b64"]) == raw
        assert originals[name + "_sha256"] == hashlib.sha256(raw).hexdigest()
    close = source_case.payload["close"]
    request = base64.b64decode(close["request_b64"])
    assert request != canonical_json_bytes(json.loads(request))
    assert close["request_sha256"] == hashlib.sha256(request).hexdigest()
    assert close["request_format"] == "PostgreSQL16/jsonb-text/UTF8"


@pytest.mark.parametrize(
    "original", ["operation_descriptor", "complete_journal", "before_metadata", "after_metadata"]
)
@pytest.mark.parametrize(
    "change", ["digest", "base64", "duplicate", "unknown", "deep", "oversized", "noncanonical"]
)
def test_embedded_originals_cannot_be_reencoded_or_substituted(source_case, original, change):
    payload = deepcopy(source_case.payload)
    originals = payload["originals"]
    if change == "digest":
        originals[original + "_sha256"] = "0" * 64
    elif change == "base64":
        originals[original + "_b64"] += "="
    else:
        raw = base64.b64decode(originals[original + "_b64"])
        if change == "duplicate":
            value = json.loads(raw)
            field = next(iter(value))
            raw = raw[:-1] + b"," + canonical_json_bytes({field: value[field]})[1:]
        elif change == "deep":
            raw = b'{"x":' + b"[" * 32 + b"0" + b"]" * 32 + b"}"
        elif change == "oversized":
            raw = b" " * (2 * 1024 * 1024 + 1)
        elif change == "noncanonical":
            raw += b"\n"
        else:
            value = json.loads(raw)
            value["caller_authenticated"] = True
            raw = canonical_json_bytes(value)
        _set_original(originals, original, raw)
    with pytest.raises(source.SourceMutationVerificationError):
        _parse_signed(source_case, payload)


@pytest.mark.parametrize(
    "change", ["digest", "base64", "duplicate", "unknown", "deep", "oversized"]
)
def test_authoritative_close_original_is_strict_and_bounded(source_case, change):
    payload = deepcopy(source_case.payload)
    close = payload["close"]
    if change == "digest":
        close["request_sha256"] = "0" * 64
    elif change == "base64":
        close["request_b64"] += "="
    else:
        raw = base64.b64decode(close["request_b64"])
        if change == "duplicate":
            raw = raw[:-1] + b', "generation":1}'
        elif change == "deep":
            raw = b'{"x":' + b"[" * 32 + b"0" + b"]" * 32 + b"}"
        elif change == "oversized":
            raw = b" " * (65536 + 1)
        else:
            value = json.loads(raw)
            value["caller_authenticated"] = True
            raw = json.dumps(value, sort_keys=True).encode()
        _set_original(close, "request", raw)
    with pytest.raises(source.SourceMutationVerificationError):
        _parse_signed(source_case, payload)


def test_equivalent_close_json_does_not_replace_policy_pinned_original_bytes(source_case):
    payload = deepcopy(source_case.payload)
    close = payload["close"]
    original = base64.b64decode(close["request_b64"])
    replacement = canonical_json_bytes(json.loads(original))
    assert replacement != original
    _set_original(close, "request", replacement)
    event = _parse_signed(source_case, payload)
    with pytest.raises(source.SourceMutationVerificationError):
        _verify(event, source_case.policy)


@pytest.mark.parametrize("artifact", ["event", "report"])
def test_source_event_and_signature_report_cannot_authorize_production_ports(source_case, artifact):
    from bot_core.production_root_lifecycle import (
        ProductionRootLifecycleBoundary,
        ProductionRootLifecycleError,
    )

    value = (
        source_case.event if artifact == "event" else _verify(source_case.event, source_case.policy)
    )
    boundary = object.__new__(ProductionRootLifecycleBoundary)
    with pytest.raises(
        ProductionRootLifecycleError,
        match="^PRODUCTION_BIND_BLOCKED_FROZEN_TRANSACTION_CONFLICT_AND_UNQUALIFIED_ATTEMPT_INGRESS$",
    ):
        boundary.commit(value)
    with pytest.raises(
        ProductionRootLifecycleError, match="INDEPENDENT_CUSTODY_COMPLETION_AUTHORITY_REQUIRED"
    ):
        boundary.publish_administrator_completion(None, "qualification-op-1", "a" * 64, value)
    with pytest.raises(custody.LocalSigningLifecycleError, match="genuine durable"):
        with lifecycle.verified_durable_signing_lifecycle_completion(
            value,
            operation_descriptor=base64.b64decode(
                source_case.payload["originals"]["operation_descriptor_b64"]
            ),
        ):
            pytest.fail("signature diagnostic became a local mutation capability")

"""Real file/process diagnostics; neither manual bytes nor writer output attest provenance."""

from __future__ import annotations

import base64
from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError, asdict, replace
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from bot_core import independent_custody_completion_qualification as qualification
from bot_core import local_signing_custody as custody
from bot_core import local_signing_lifecycle_evidence as lifecycle
from bot_core.licensing.canonical import canonical_json_bytes
from bot_core.local_signing_custody import LocalSigningCustodyError, LocalSigningMetadata
from bot_core.root_proof_issuer_substrate import (
    ProviderRole,
    SecurityProfile,
    SecurityProfileIdentity,
    public_key_material_identity,
)
from tests.security.test_local_signing_custody import (
    native_keyring as native_keyring,
    security as security,
)
from tests.security.test_local_signing_lifecycle_evidence import (
    _transition,
    authority as authority,
    close_authorization as close_authorization,
)


requires_linux_observation = pytest.mark.skipif(
    sys.platform != "linux", reason="existing read-only custody observation requires Linux"
)


def _b64(raw):
    return base64.b64encode(raw).decode("ascii")


def _set_original(document, name, raw):
    document[name + "_b64"] = _b64(raw)
    document[name + "_sha256"] = hashlib.sha256(raw).hexdigest()


def _document(descriptor, journal):
    operation, before = json.loads(descriptor), json.loads(journal)["before_metadata"]
    public = b"c" * 32
    value = {
        "schema": qualification.CLAIM_SCHEMA,
        "schema_version": 1,
        "domain": qualification.CLAIM_DOMAIN,
        "profile": "PRODUCTION_LOCAL",
        "installation_id": operation["database_identity"]["instance_id"],
        "environment_id": "isolated-qualification-test",
        "trust_domain": before["trust_domain"],
        "installation_sha256": "a" * 64,
        "external_trust_pin_sha256": "b" * 64,
        "completion_authority_claim": {
            "authority_id": "untrusted-test-completion-authority",
            "host_id": "untrusted-test-host",
            "os_uid": 1005,
            "os_gid": 2005,
            "credential_identity": "untrusted-completion-credential-1",
            "semantic_role": qualification.COMPLETION_SEMANTIC_ROLE,
            "key_version": 1,
            "lifecycle_generation": 1,
            "lifecycle_state": "ACTIVE",
            "public_key_b64": _b64(public),
            "public_material_identity": public_key_material_identity(public),
        },
        "database_identity_claim": operation["database_identity"],
    }
    close = {
        "operation_id": operation["operation_id"],
        "semantic_role": operation["semantic_role"],
        "identity": before["credential_id"],
        "generation": operation["lifecycle_generation"],
        "predecessor_state": operation["predecessor_state"],
        "target_state": operation["target_state"],
    }
    # PostgreSQL jsonb::text and the canonical JSON profile have different bytes.
    _set_original(value, "production_close_request", json.dumps(close, sort_keys=True).encode())
    _set_original(value, "operation_descriptor", descriptor)
    _set_original(value, "complete_journal", journal)
    return value


def _publish(root, before):
    root.chmod(0o700)
    after = replace(
        before, lifecycle_generation=2, lifecycle_state=custody.SigningKeyLifecycle.VERIFY_ONLY
    )
    descriptor = canonical_json_bytes(
        {
            "protocol_version": 2,
            "operation_id": "qualification-op-1",
            "fence_key": "untrusted-test-fence",
            "semantic_role": before.provider_role.value,
            "credential_identity": asdict(before.credential_identity()),
            "public_key_hex": before.public_key.hex(),
            "key_version": 1,
            "lifecycle_generation": 1,
            "predecessor_state": "ACTIVE",
            "target_state": "VERIFY_ONLY",
            "custody_scope": {
                "directory": str(root),
                "security_profile": "PRODUCTION_LOCAL",
                "trust_domain": before.trust_domain,
            },
            "database_identity": {
                "system_identifier": "100",
                "database_oid": 200,
                "database_name": "qualification-db",
                "cluster_name": "",
                "connection_host": "/run/postgresql",
                "connection_port": 5432,
                "server_address": None,
                "server_port": None,
                "instance_id": "untrusted-test-installation",
            },
        }
    )
    journal = lifecycle._document(
        lifecycle._OperationRecord("qualification-op-1", descriptor, before, after, "COMPLETE")
    )
    directory = root / ".signing-lifecycle-operations"
    directory.mkdir(mode=0o700)
    journal_path = directory / (hashlib.sha256(b"qualification-op-1").hexdigest() + ".json")
    metadata_path = root / "root-proof-signing.json"
    for path, raw in (
        (root / ".signing-custody.lock", b""),
        (journal_path, journal),
        (metadata_path, custody._encode_record(after)),
    ):
        path.write_bytes(raw)
        path.chmod(0o600)
    document = _document(descriptor, journal)
    return SimpleNamespace(
        root=root,
        before=before,
        after=after,
        descriptor=descriptor,
        journal=journal,
        journal_path=journal_path,
        metadata_path=metadata_path,
        document=document,
        claim=qualification.parse_completion_qualification_claim(canonical_json_bytes(document)),
    )


@pytest.fixture
def completion(tmp_path):
    security = SecurityProfileIdentity(SecurityProfile.PRODUCTION_LOCAL, "test.qualification")
    role = ProviderRole.ROOT_PROOF_SIGNING
    identity = custody._canonical_signing_identity(security, role, "a" * 32)
    before = LocalSigningMetadata(
        security.profile,
        security.trust_domain,
        role,
        custody._SEMANTIC_ROLE[role],
        identity.credential_id,
        identity.provider_namespace,
        identity.key_handle,
        1,
        identity.lifecycle_namespace,
        1,
        custody.SigningKeyLifecycle.ACTIVE,
        bytes(range(32)),
        identity.protected_private_material_reference,
    )
    return _publish(tmp_path, before)


def _parse(document):
    return qualification.parse_completion_qualification_claim(canonical_json_bytes(document))


@requires_linux_observation
def test_immutable_claim_and_exact_observation_never_become_authority(completion):
    claim = completion.claim
    with qualification.observe_completion_qualification(claim) as assessment:
        assert assessment.status == "NOT_QUALIFIED"
        assert assessment.production_emission == "BLOCKED"
        assert len(assessment.blockers) == 9
        assert "CUSTODY_MUTATION_SOURCE_NOT_AUTHENTICATED" in assessment.blockers
        assert "PRIOR_DURABILITY_NOT_ATTESTED" in assessment.blockers
    with pytest.raises(FrozenInstanceError):
        claim.canonical_bytes = b"{}"
    with pytest.raises(FrozenInstanceError):
        assessment.status = "QUALIFIED"
    document = claim.document
    document["completion_authority_claim"]["authority_id"] = "mutated"
    assert claim.document == completion.document
    assert claim.fingerprint_sha256 == hashlib.sha256(claim.canonical_bytes).hexdigest()
    with pytest.raises(custody.LocalSigningLifecycleError, match="genuine durable"):
        with lifecycle.verified_durable_signing_lifecycle_completion(
            claim, operation_descriptor=completion.descriptor
        ):
            pytest.fail("claim became a durable mutation capability")


def test_original_production_request_preserved_without_canonical_substitution(completion):
    document = completion.document
    raw = base64.b64decode(document["production_close_request_b64"])
    assert raw != canonical_json_bytes(json.loads(raw))
    assert document["production_close_request_sha256"] == hashlib.sha256(raw).hexdigest()
    document["production_close_request_sha256"] = hashlib.sha256(
        canonical_json_bytes(json.loads(raw))
    ).hexdigest()
    with pytest.raises(qualification.CompletionQualificationError):
        _parse(document)


@pytest.mark.parametrize(
    "raw", [b"", b"{}", b"null", b"[]", b"\xff", b'{"x":NaN}', b'{"x":1e999}', b"{" * 2000]
)
def test_malformed_raw_fails_closed(raw):
    with pytest.raises(qualification.CompletionQualificationError):
        qualification.parse_completion_qualification_claim(raw)


@pytest.mark.parametrize(
    "change",
    [
        "noncanonical",
        "duplicate",
        "oversized",
        "extra",
        "missing",
        "bool_version",
        "float_version",
        "domain",
        "profile",
        "installation",
        "trust",
        "pin",
        "claim_type",
        "depth",
    ],
)
def test_outer_format_closed_and_bounded(completion, change):
    document, raw = completion.document, completion.claim.canonical_bytes
    if change == "noncanonical":
        raw += b"\n"
    elif change == "duplicate":
        raw = raw[:-1] + b',"schema_version":1}'
    elif change == "oversized":
        raw = b" " * (qualification.MAX_CLAIM_BYTES + 1)
    elif change == "claim_type":
        raw = bytearray(raw)
    elif change == "float_version":
        raw = raw.replace(b'"schema_version":1,', b'"schema_version":1.0,')
    elif change == "depth":
        raw = b'{"x":' + b"[" * 17 + b"0" + b"]" * 17 + b"}"
    else:
        if change == "extra":
            document["source_authenticated"] = True
        elif change == "missing":
            del document["environment_id"]
        elif change == "bool_version":
            document["schema_version"] = True
        elif change == "domain":
            document["domain"] = "CryptoHunter/ProductionCompletion/V1"
        elif change == "profile":
            document["profile"] = "SERVER_READY"
        elif change == "installation":
            document["installation_id"] = "another-installation"
        elif change == "trust":
            document["trust_domain"] = "another-trust"
        elif change == "pin":
            document["external_trust_pin_sha256"] = "untrusted"
        raw = canonical_json_bytes(document)
    with pytest.raises(qualification.CompletionQualificationError):
        qualification.parse_completion_qualification_claim(raw)


@pytest.mark.parametrize(
    "field,value",
    [
        ("os_uid", True),
        ("os_gid", 0),
        ("key_version", True),
        ("lifecycle_generation", True),
        ("lifecycle_generation", 2),
        ("lifecycle_state", "ACTIVE "),
        ("semantic_role", "ROOT_PROOF_SIGNING"),
        ("semantic_role", "HISTORY_ATTESTATION_SIGNING"),
        ("public_material_identity", "sha256:" + "0" * 64),
        ("host_id", "host\0alias"),
        ("source_authenticated", True),
    ],
)
def test_authority_cannot_launder_types_role_or_lifecycle(completion, field, value):
    completion.document["completion_authority_claim"][field] = value
    with pytest.raises(qualification.CompletionQualificationError):
        _parse(completion.document)


def test_completion_credential_cannot_alias_signer_material(completion):
    authority = completion.document["completion_authority_claim"]
    authority["public_key_b64"], authority["public_material_identity"] = (
        _b64(completion.before.public_key),
        completion.before.key_material_identity,
    )
    with pytest.raises(qualification.CompletionQualificationError):
        _parse(completion.document)


@pytest.mark.parametrize(
    "field,value",
    [
        ("database_oid", True),
        ("connection_port", True),
        ("server_port", True),
        ("system_identifier", 100),
        ("system_identifier", "01"),
        ("system_identifier", str(2**64)),
        ("cluster_name", "cluster\0alias"),
        ("database_name", "db\0alias"),
        ("extra", 1),
    ],
)
def test_physical_database_types_exact(completion, field, value):
    completion.document["database_identity_claim"][field] = value
    with pytest.raises(qualification.CompletionQualificationError):
        _parse(completion.document)


@pytest.mark.parametrize(
    "original", ["production_close_request", "operation_descriptor", "complete_journal"]
)
@pytest.mark.parametrize(
    "change", ["digest", "base64", "duplicate", "unknown", "depth", "oversized"]
)
def test_original_bytes_and_nested_json_cannot_be_substituted(completion, original, change):
    document = completion.document
    if change == "digest":
        document[original + "_sha256"] = "0" * 64
    elif change == "base64":
        document[original + "_b64"] += "="
    else:
        raw = base64.b64decode(document[original + "_b64"])
        if change == "duplicate":
            name = "operation_id" if original != "complete_journal" else "completion_state"
            raw = (
                raw[:-1]
                + b","
                + json.dumps(name).encode()
                + b":"
                + json.dumps(json.loads(raw)[name]).encode()
                + b"}"
            )
        elif change == "depth":
            raw = b'{"x":' + b"[" * 17 + b"0" + b"]" * 17 + b"}"
        elif change == "oversized":
            raw = (
                b" "
                * {
                    "production_close_request": 65537,
                    "operation_descriptor": 262145,
                    "complete_journal": 1048577,
                }[original]
            )
        else:
            value = json.loads(raw)
            value["source_authenticated"] = True
            raw = canonical_json_bytes(value)
        _set_original(document, original, raw)
    with pytest.raises(qualification.CompletionQualificationError):
        _parse(document)


@pytest.mark.parametrize(
    "original,field,value",
    [
        ("production_close_request", "operation_id", "replayed-op"),
        ("production_close_request", "generation", True),
        ("production_close_request", "identity", "substituted-credential"),
        ("production_close_request", "semantic_role", "HISTORY_ATTESTATION_SIGNING"),
        ("operation_descriptor", "protocol_version", True),
        ("operation_descriptor", "key_version", True),
        ("operation_descriptor", "lifecycle_generation", True),
        ("operation_descriptor", "operation_id", "replayed-op"),
        ("operation_descriptor", "semantic_role", "HISTORY_ATTESTATION_SIGNING"),
        ("complete_journal", "schema_version", True),
        ("complete_journal", "completion_state", "INTENT"),
        ("complete_journal", "operation_id", "replayed-op"),
    ],
)
def test_exact_operation_and_generation_binding_rejects_replay(completion, original, field, value):
    document = completion.document
    nested = json.loads(base64.b64decode(document[original + "_b64"]))
    nested[field] = value
    raw = canonical_json_bytes(nested)
    _set_original(document, original, raw)
    if original == "operation_descriptor":
        journal = json.loads(completion.journal)
        journal["operation_descriptor_b64"] = _b64(raw)
        _set_original(document, "complete_journal", canonical_json_bytes(journal))
    with pytest.raises(qualification.CompletionQualificationError):
        _parse(document)


@pytest.mark.parametrize(
    "container,field,value",
    [
        ("before_metadata", "lifecycle_generation", True),
        ("after_metadata", "lifecycle_generation", True),
        ("before_metadata", "key_version", True),
        ("before_metadata", "storage_schema_version", True),
        ("before_metadata", "credential_id", "substituted-credential"),
        ("after_metadata", "provider_role", "HISTORY_ATTESTATION_SIGNING"),
        ("before_metadata", "caller_authenticated", True),
    ],
)
def test_nested_journal_metadata_is_exact(completion, container, field, value):
    document = completion.document
    journal = json.loads(completion.journal)
    journal[container][field] = value
    _set_original(document, "complete_journal", canonical_json_bytes(journal))
    with pytest.raises(qualification.CompletionQualificationError):
        _parse(document)


@pytest.mark.parametrize("container", ["credential_identity", "database_identity", "custody_scope"])
def test_nested_descriptor_fields_closed(completion, container):
    document = completion.document
    operation = json.loads(completion.descriptor)
    operation[container]["caller_authenticated"] = True
    raw = canonical_json_bytes(operation)
    journal = json.loads(completion.journal)
    journal["operation_descriptor_b64"] = _b64(raw)
    _set_original(document, "operation_descriptor", raw)
    _set_original(document, "complete_journal", canonical_json_bytes(journal))
    with pytest.raises(qualification.CompletionQualificationError):
        _parse(document)


def test_quoted_delimiters_escapes_and_untrusted_authority_do_not_supply_proof(completion):
    for state, generation in (("ACTIVE", 1), ("VERIFY_ONLY", 2), ("REVOKED", 3)):
        document = completion.claim.document
        document["completion_authority_claim"].update(
            lifecycle_state=state, lifecycle_generation=generation
        )
        document["completion_authority_claim"]["authority_id"] = 'untrusted\\"[' + "{}[]" * 30
        document["external_trust_pin_sha256"] = "e" * 64
        observed = qualification.qualify_completion_claim(_parse(document))
        assert observed.status == "NOT_QUALIFIED"
        assert "EXTERNAL_TRUST_PIN_NOT_VERIFIED" in observed.blockers
        assert "COMPLETION_CREDENTIAL_LIFECYCLE_NOT_QUALIFIED" in observed.blockers


@requires_linux_observation
def test_local_writer_and_identical_manual_bytes_are_not_authenticated(
    tmp_path, security, authority, close_authorization
):
    # The unmodified local writer uses the existing explicit unit close seam;
    # it proves filesystem execution here, never production SQL authorization.
    receipt = _transition(tmp_path, security, authority, close_authorization)
    claim = _parse(_document(receipt.operation_descriptor, receipt.evidence_bytes))
    with qualification.observe_completion_qualification(claim) as genuine:
        assert genuine.status == "NOT_QUALIFIED"
    operation = json.loads(receipt.operation_descriptor)
    journal = (
        tmp_path
        / ".signing-lifecycle-operations"
        / (hashlib.sha256(operation["operation_id"].encode()).hexdigest() + ".json")
    )
    journal.unlink()
    journal.write_bytes(receipt.evidence_bytes)
    journal.chmod(0o600)
    metadata = tmp_path / "root-proof-signing.json"
    metadata.unlink()
    metadata.write_bytes(custody._encode_record(receipt.metadata))
    metadata.chmod(0o600)
    with qualification.observe_completion_qualification(claim) as manual:
        assert manual.status == "NOT_QUALIFIED"
        assert manual.blockers == genuine.blockers
        assert manual.claim.canonical_bytes == genuine.claim.canonical_bytes


@requires_linux_observation
def test_real_observation_readonly_no_secrets_sync_or_capability(completion, monkeypatch):
    original = {p: p.read_bytes() for p in completion.root.rglob("*") if p.is_file()}
    real_open, flags_seen = os.open, []

    def read_only(path, flags, *args, **kwargs):
        assert flags & os.O_ACCMODE == os.O_RDONLY
        assert not flags & (os.O_CREAT | os.O_TRUNC | os.O_APPEND)
        flags_seen.append(flags)
        return real_open(path, flags, *args, **kwargs)

    def forbidden(*args, **kwargs):
        pytest.fail("qualification attempted secrets, persistence or mutation")

    monkeypatch.setattr(os, "open", read_only)
    monkeypatch.setattr(os, "fsync", forbidden)
    for module, names in (
        (custody, ("_load_private_key", "_replace_record", "_write_new_record")),
        (
            lifecycle,
            (
                "_sync_path",
                "durable_transition_local_signing_credential_lifecycle",
                "verified_durable_signing_lifecycle_completion",
            ),
        ),
    ):
        for name in names:
            monkeypatch.setattr(module, name, forbidden)
    monkeypatch.setattr(
        custody.NativeKeyringSigningSecretReader, "read_authority_secret", forbidden
    )
    monkeypatch.setattr(
        custody.NativeKeyringSigningSecretReader, "read_existing_authority_secret", forbidden
    )
    with qualification.observe_completion_qualification(completion.claim) as observed:
        assert observed.status == "NOT_QUALIFIED"
        assert observed.production_emission == "BLOCKED"
    assert flags_seen
    assert {p: p.read_bytes() for p in completion.root.rglob("*") if p.is_file()} == original


@pytest.mark.parametrize(
    "change",
    [
        "missing",
        "intent",
        "metadata",
        "journal_symlink",
        "journal_inode",
        "directory_swap",
        "in_place",
    ],
)
@requires_linux_observation
def test_real_missing_interrupted_or_substituted_source_fails_closed(completion, change):
    if change == "missing":
        completion.journal_path.unlink()
    elif change == "intent":
        value = json.loads(completion.journal)
        value["completion_state"] = "INTENT"
        completion.journal_path.write_bytes(canonical_json_bytes(value))
    elif change == "metadata":
        completion.metadata_path.write_bytes(custody._encode_record(completion.before))
    elif change == "journal_symlink":
        target = completion.journal_path.with_suffix(".actual")
        completion.journal_path.rename(target)
        completion.journal_path.symlink_to(target)
    if change in {"missing", "intent", "metadata", "journal_symlink"}:
        with pytest.raises(LocalSigningCustodyError):
            with qualification.observe_completion_qualification(completion.claim):
                pytest.fail("interrupted evidence accepted")
    else:
        with pytest.raises(LocalSigningCustodyError):
            with qualification.observe_completion_qualification(completion.claim):
                if change == "journal_inode":
                    completion.journal_path.unlink()
                    completion.journal_path.write_bytes(completion.journal)
                    completion.journal_path.chmod(0o600)
                elif change == "directory_swap":
                    completion.journal_path.parent.rename(completion.root / ".previous")
                    completion.journal_path.parent.mkdir(mode=0o700)
                    completion.journal_path.write_bytes(completion.journal)
                    completion.journal_path.chmod(0o600)
                else:
                    completion.journal_path.write_bytes(completion.journal + b" ")


@requires_linux_observation
def test_same_operation_with_different_descriptor_does_not_match_existing_journal(completion):
    document = completion.claim.document
    descriptor = json.loads(completion.descriptor)
    descriptor["fence_key"] = "another-fence"
    raw = canonical_json_bytes(descriptor)
    journal = json.loads(completion.journal)
    journal["operation_descriptor_b64"] = _b64(raw)
    _set_original(document, "operation_descriptor", raw)
    _set_original(document, "complete_journal", canonical_json_bytes(journal))
    claim = _parse(document)
    assert qualification.qualify_completion_claim(claim).status == "NOT_QUALIFIED"
    with pytest.raises(LocalSigningCustodyError):
        with qualification.observe_completion_qualification(claim):
            pytest.fail("same ID authenticated another descriptor")


@requires_linux_observation
def test_exact_retries_concurrent_requests_and_restart_remain_not_qualified(completion):
    def observe():
        clone = qualification.parse_completion_qualification_claim(
            bytes(completion.claim.canonical_bytes)
        )
        with qualification.observe_completion_qualification(clone) as observed:
            return observed.status, observed.claim.fingerprint_sha256

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: observe(), range(4)))
    assert results == [("NOT_QUALIFIED", completion.claim.fingerprint_sha256)] * 4
    script = """
from pathlib import Path
import sys
from bot_core.independent_custody_completion_qualification import parse_completion_qualification_claim, observe_completion_qualification
claim = parse_completion_qualification_claim(Path(sys.argv[1]).read_bytes())
with observe_completion_qualification(claim) as observed:
    print(observed.status, observed.production_emission, claim.fingerprint_sha256)
"""
    claim_file = completion.root / "qualification-claim.json"
    claim_file.write_bytes(completion.claim.canonical_bytes)
    result = subprocess.run(
        [sys.executable, "-c", script, str(claim_file)],
        check=True,
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.stdout.strip() == "NOT_QUALIFIED BLOCKED " + completion.claim.fingerprint_sha256


def _wait_for_exclusive_lock(path, ready, acquired):
    import fcntl

    descriptor = os.open(path, os.O_RDONLY)
    try:
        ready.set()
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        acquired.set()
        fcntl.flock(descriptor, fcntl.LOCK_UN)
    finally:
        os.close(descriptor)


@requires_linux_observation
def test_real_shared_lock_held_until_context_exit(completion):
    context = multiprocessing.get_context("spawn")
    ready, acquired = context.Event(), context.Event()
    child = context.Process(
        target=_wait_for_exclusive_lock,
        args=(str(completion.root / ".signing-custody.lock"), ready, acquired),
    )
    try:
        with qualification.observe_completion_qualification(completion.claim) as observed:
            child.start()
            assert ready.wait(5)
            assert not acquired.wait(0.1)
            assert observed.status == "NOT_QUALIFIED"
        assert acquired.wait(5)
        child.join(5)
        assert child.exitcode == 0
    finally:
        if child.is_alive():
            child.terminate()
            child.join(5)


@pytest.mark.parametrize("artifact", ["claim", "report"])
def test_qualification_artifacts_cannot_authorize_production_ports(completion, artifact):
    from bot_core.production_root_lifecycle import (
        ProductionRootLifecycleBoundary,
        ProductionRootLifecycleError,
    )

    boundary = object.__new__(ProductionRootLifecycleBoundary)
    value = (
        completion.claim
        if artifact == "claim"
        else qualification.qualify_completion_claim(completion.claim)
    )
    with pytest.raises(
        ProductionRootLifecycleError,
        match="^PRODUCTION_BIND_BLOCKED_FROZEN_TRANSACTION_CONFLICT_AND_UNQUALIFIED_ATTEMPT_INGRESS$",
    ):
        boundary.commit(value)
    with pytest.raises(
        ProductionRootLifecycleError, match="INDEPENDENT_CUSTODY_COMPLETION_AUTHORITY_REQUIRED"
    ):
        boundary.publish_administrator_completion(None, "qualification-op-1", "a" * 64, value)


def test_synthetic_signature_field_is_not_an_authority_port(completion):
    completion.document["signature_b64"] = _b64(b"s" * 64)
    with pytest.raises(qualification.CompletionQualificationError):
        _parse(completion.document)


def test_nesting_limit_precedes_json_decoder(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("unbounded JSON reached the recursive decoder")

    raw = (
        b'{"x":'
        + b"[" * (qualification.MAX_JSON_DEPTH + 1)
        + b"0"
        + b"]" * (qualification.MAX_JSON_DEPTH + 1)
        + b"}"
    )
    monkeypatch.setattr(qualification.json, "loads", forbidden)
    with pytest.raises(ValueError, match="nesting exceeds limit"):
        qualification._json(raw, limit=qualification.MAX_CLAIM_BYTES)
    with pytest.raises(qualification.CompletionQualificationError):
        qualification.parse_completion_qualification_claim(raw)

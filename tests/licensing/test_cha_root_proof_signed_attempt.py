"""Serial real SQLite/native-storage tests with explicitly simulated public authority.

Genuine PostgreSQL activation and the complete guarded lineage are exercised in
the separate external_postgresql test. No fixture is registered by production.
"""

from __future__ import annotations

import copy
import hashlib
import os
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from bot_core import cha_attempt_store as persistence, cha_root_proof_signing_custody as custody
from bot_core.cha_issuance_request import (
    CLAIMANT_PROFILE,
    ISSUER_TARGET_NAMESPACE,
    REQUEST_FIELDS,
    REQUESTER_PROFILE,
    IssuanceSignerIdentity,
    IssuanceSigningRole,
    encode_signature,
    request_bytes,
    request_reference,
    signature_bytes,
)
from bot_core.licensing import (
    cha_root_proof_attempt_reservation as upstream,
    cha_root_proof_signed_attempt as signed,
)
from bot_core.licensing.canonical import canonical_json_bytes, parse_canonical
from bot_core.root_proof_issuer_substrate import public_key_material_identity
from tests.licensing import test_cha_root_proof_attempt_reservation as upstream_tests
from tests.licensing.test_cha_attempt_store_stage9 import authorization
from tests.security import test_local_signing_custody as native_custody_tests
from tests.security._local_signing_platform import requires_posix_custody_locking

native_keyring = native_custody_tests.native_keyring

operation_resolution = upstream_tests.operation_resolution


class InjectedCrash(RuntimeError):
    pass


def signer_identity(role, auth, private):
    requester = role is IssuanceSigningRole.REQUESTER
    public = private.public_key().public_bytes(
        serialization.Encoding.Raw, serialization.PublicFormat.Raw
    )
    scope = "TEST_ONLY:" + role.value
    return IssuanceSignerIdentity(
        role,
        auth.requester_principal_id if requester else auth.provisioning_principal_id,
        role.semantic_role,
        auth.environment,
        auth.trust_domain,
        auth.requester_key_id if requester else auth.claimant_key_id,
        auth.requester_key_version if requester else auth.claimant_key_version,
        public.hex(),
        public_key_material_identity(public),
        scope + ":service",
        scope + ":credential",
        scope + ":lifecycle",
        scope + ":handle",
        "ACTIVE",
        1,
    )


@pytest.fixture
def store_flow(tmp_path):
    auth = authorization()
    keys = (Ed25519PrivateKey.generate(), Ed25519PrivateKey.generate())
    identities = tuple(
        signer_identity(role, auth, private)
        for role, private in zip(IssuanceSigningRole, keys, strict=True)
    )
    path = tmp_path / "attempts.db"
    with persistence.SQLiteCHAAttemptStore(path, auth.trust_domain) as store:
        current = store.reserve_or_resolve_attempt_id(auth)
    return path, auth, current, keys, identities


def checkpoint(store, current, private, identity):
    value = store.prepare_signature(current, identity)
    signature = encode_signature(
        private.sign(identity.role.domain.encode() + b"\0" + value.canonical_bytes)
    )
    return store.persist_signature(current, identity, signature)


def final_identity(store, current):
    value = store.signed_request(current.reservation.authorization.logical_operation_id)
    return persistence.AttemptIdentity(
        current.reservation.authorization,
        current.reservation.issuance_attempt_id,
        value.digest,
        value.reference,
        value.requester[1],
        value.claimant[1],
        REQUESTER_PROFILE,
        CLAIMANT_PROFILE,
    )


@pytest.fixture
def simulated_signed_flow(operation_resolution, native_keyring, monkeypatch, tmp_path):
    binding, provider, _ = operation_resolution
    path = tmp_path / "attempts.sqlite3"
    monkeypatch.setattr(upstream, "_attempt_store_path", lambda: path)
    port = provider.requester_registry
    port.record = replace(port.record, requester_principal_id="CryptoHunterAccountAuthority")
    upstream_tests._refresh_active_credential_identity(port)
    initial_auth = upstream.resolve_root_proof_issuance_authorization(binding).authorization
    directories = signed._custody_directories(initial_auth)
    admins = []
    for role, directory, port in zip(
        IssuanceSigningRole,
        directories,
        (provider.requester_registry, provider.claimant_registry),
        strict=True,
    ):
        requester = role is IssuanceSigningRole.REQUESTER
        principal = (
            initial_auth.requester_principal_id
            if requester
            else initial_auth.provisioning_principal_id
        )
        key_id = initial_auth.requester_key_id if requester else initial_auth.claimant_key_id
        admin = custody.OfflineIssuanceCustodyAdministrator(
            directory, role=role, trust_domain=initial_auth.trust_domain, principal=principal
        )
        draft = admin.stage(key_id=key_id, key_version=1)
        port.public_keys[key_id] = bytes.fromhex(draft.public_key_hex)
        upstream_tests._refresh_active_credential_identity(port)
        admins.append(admin)

    def simulated_registry_binding(value, port):
        # This is the only simulated custody provisioning check. Genuine exact
        # PostgreSQL binding is tested separately, with no replacement of it.
        assert port.public_key(value["key_id"]).hex() == value["public_key_hex"]
        assert port.record.lifecycle == "ACTIVE"

    monkeypatch.setattr(custody, "_require_registry", simulated_registry_binding)
    for admin, port in zip(
        admins, (provider.requester_registry, provider.claimant_registry), strict=True
    ):
        admin.activate(port)
    auth = upstream.resolve_root_proof_issuance_authorization(binding)
    reservation = upstream.reserve_root_proof_issuance_attempt(binding, auth)
    calls = [0, 0]
    requester_sign = custody.LocalCHARequesterSigningCustody.sign_issuance_request
    claimant_sign = custody.LocalPreaccountClaimantAuthorizationCustody.authorize_entitlement_claim

    def counted_requester(self, raw, identity):
        calls[0] += 1
        return requester_sign(self, raw, identity)

    def counted_claimant(self, raw, identity):
        calls[1] += 1
        return claimant_sign(self, raw, identity)

    monkeypatch.setattr(
        custody.LocalCHARequesterSigningCustody, "sign_issuance_request", counted_requester
    )
    monkeypatch.setattr(
        custody.LocalPreaccountClaimantAuthorizationCustody,
        "authorize_entitlement_claim",
        counted_claimant,
    )
    return binding, provider, auth, reservation, calls, admins


@pytest.mark.parametrize("principal", ["Deployment Żółw 😀 \u000f", "Security \U0001f600\u20ac"])
def test_exact_jcs_payload_digest_and_reference(principal):
    auth = authorization(provisioning_principal_id=principal)
    attempt_id = "rpa_018f3e70-7b5a-7c21-8b9a-0123456789ab"
    raw = request_bytes(auth, attempt_id)
    value = parse_canonical(raw)
    assert set(value) == set(REQUEST_FIELDS)
    assert value["schema_version"] == "1"
    assert value["issuer_target_namespace"] == ISSUER_TARGET_NAMESPACE
    assert value["entitlement_id"] == auth.bootstrap_entitlement_id
    assert value["requester_id"] == "CryptoHunterAccountAuthority"
    assert value["issuance_attempt_id"] == attempt_id
    assert value["provisioning_principal_id"] == principal
    assert raw == canonical_json_bytes(value)
    assert principal.split()[0].encode() in raw
    digest = hashlib.sha256(raw).hexdigest()
    assert request_reference(raw) == "immutable:req:sha256:" + digest
    assert "attempt_identity_digest_sha256" not in value


@pytest.mark.parametrize("mutation", ["padded", "short", "noncanonical", "badalphabet", "long"])
def test_signature_representation_is_exact(mutation):
    encoded = encode_signature(b"a" * 64)
    value = {
        "padded": encoded + "==",
        "short": encoded[:-2],
        "noncanonical": encoded[:-1] + "R",
        "badalphabet": "+" + encoded[1:],
        "long": encode_signature(b"a" * 64) + "AA",
    }[mutation]
    with pytest.raises(ValueError):
        signature_bytes(value)


@pytest.mark.parametrize(
    "mutation", ["domain", "byte", "attempt", "version", "issuer", "freshness"]
)
def test_cross_protocol_and_payload_replay_rejected(store_flow, mutation):
    _, auth, current, keys, identities = store_flow
    raw = request_bytes(auth, current.reservation.issuance_attempt_id)
    identity = identities[0]
    signed_raw = raw
    domain = identity.role.domain
    if mutation == "domain":
        domain = IssuanceSigningRole.CLAIMANT.domain
    elif mutation == "byte":
        signed_raw = raw.replace(b"CryptoHunter.Stage9", b"cryptoHunter.Stage9")
    elif mutation in {"attempt", "version"}:
        value = parse_canonical(raw)
        value["issuance_attempt_id" if mutation == "attempt" else "requester_key_version"] = (
            "rpa_018f3e70-7b5a-7c21-8b9a-0123456789ac" if mutation == "attempt" else 2
        )
        signed_raw = canonical_json_bytes(value)
    else:
        domain = (
            "CRYPTOHUNTER_ACCOUNT_GENESIS_ROOT_PROOF_V1"
            if mutation == "issuer"
            else "cryptohunter.account-genesis.cha-proposer-authentication.v1"
        )
    signature = encode_signature(keys[0].sign(domain.encode() + b"\0" + signed_raw))
    with pytest.raises(ValueError, match="verification"):
        identity.verify(raw, signature)
    claimant_signature = encode_signature(
        keys[1].sign(IssuanceSigningRole.CLAIMANT.domain.encode() + b"\0" + raw)
    )
    with pytest.raises(ValueError):
        replace(
            identities[1],
            role=IssuanceSigningRole.REQUESTER,
            principal=auth.requester_principal_id,
            semantic_role=IssuanceSigningRole.REQUESTER.semantic_role,
        ).verify(raw, claimant_signature)


def test_checkpoints_reload_exact_bytes_and_final_digest(store_flow):
    path, auth, current, keys, identities = store_flow
    attempt_id = current.reservation.issuance_attempt_id
    with persistence.SQLiteCHAAttemptStore(path, auth.trust_domain) as store:
        current = checkpoint(store, current, keys[0], identities[0])
        assert current.state is persistence.AttemptState.REQUEST_SIGNED_BY_REQUESTER
        before = store.signed_request(auth.logical_operation_id)
    with persistence.SQLiteCHAAttemptStore(path, auth.trust_domain) as store:
        assert store.signed_request(auth.logical_operation_id) == before
        current = checkpoint(
            store, store.attempt(auth.logical_operation_id), keys[1], identities[1]
        )
        assert current.state is persistence.AttemptState.CLAIMANT_AUTHORIZED
        identity = final_identity(store, current)
        current = store.finalize_attempt(identity, expected_fence=current.fence)
        assert current.state is persistence.AttemptState.SIGNED_IMMUTABLE_DURABLE_NOT_SENT
        assert current.fence == 4
        assert current.reservation.issuance_attempt_id == attempt_id
        assert identity.digest_sha256 != before.digest
        assert (
            identity.digest_sha256
            == hashlib.sha256(
                persistence._ATTEMPT_DOMAIN + canonical_json_bytes(identity.payload())
            ).hexdigest()
        )
        assert store.finalize_attempt(identity, expected_fence=1) == current
    with persistence.SQLiteCHAAttemptStore(path, auth.trust_domain) as store:
        assert store.attempt(auth.logical_operation_id) == current
        assert (
            store.signed_request(auth.logical_operation_id).canonical_bytes
            == before.canonical_bytes
        )


@requires_posix_custody_locking
@pytest.mark.parametrize(
    "cut,expected_calls,state",
    [
        (
            IssuanceSigningRole.REQUESTER.value + ":before_intent",
            [0, 0],
            "RESERVED_AWAITING_SIGNATURES",
        ),
        (
            IssuanceSigningRole.REQUESTER.value + ":checkpoint_durable",
            [1, 0],
            "REQUEST_SIGNED_BY_REQUESTER",
        ),
        (IssuanceSigningRole.CLAIMANT.value + ":checkpoint_durable", [1, 1], "CLAIMANT_AUTHORIZED"),
        ("before_finalization", [1, 1], "CLAIMANT_AUTHORIZED"),
        ("finalization_durable", [1, 1], "SIGNED_IMMUTABLE_DURABLE_NOT_SENT"),
    ],
)
def test_crash_and_lost_response_resume_without_resigning(
    simulated_signed_flow, monkeypatch, cut, expected_calls, state
):
    binding, _, auth, reservation, calls, _ = simulated_signed_flow
    attempt_id = reservation.issuance_attempt_id

    def crash(name):
        if name == cut:
            raise InjectedCrash(name)

    monkeypatch.setattr(signed, "_cut", crash)
    with pytest.raises(InjectedCrash):
        signed.sign_root_proof_issuance_attempt(reservation)
    assert calls == expected_calls
    with upstream._open_store(auth.authorization.trust_domain) as store:
        current = store.attempt(auth.authorization.logical_operation_id)
        assert current.state.value == state
        assert current.reservation.issuance_attempt_id == attempt_id
        before = store.signed_request(auth.authorization.logical_operation_id) if calls[0] else None
    monkeypatch.setattr(signed, "_cut", lambda name: None)
    result = signed.resume_root_proof_issuance_attempt(binding, auth)
    assert calls == [1, 1]
    assert result.issuance_attempt_id == attempt_id
    assert result.state is persistence.AttemptState.SIGNED_IMMUTABLE_DURABLE_NOT_SENT
    if before is not None:
        assert result.canonical_request_bytes == before.canonical_bytes
        assert result.identity.requester_signature_base64url == before.requester[1]
    retried = signed.sign_root_proof_issuance_attempt(reservation)
    assert result.identity == retried.identity
    assert calls == [1, 1]


@requires_posix_custody_locking
@pytest.mark.parametrize("role", list(IssuanceSigningRole))
@pytest.mark.parametrize(
    "where",
    [
        "intent_durable",
        "signer_returned",
        "signature_inserted",
        "transition_inserted",
        "cas_written",
    ],
)
def test_uncommitted_signature_never_authorizes_second_invocation(
    simulated_signed_flow, monkeypatch, role, where
):
    binding, _, auth, reservation, calls, _ = simulated_signed_flow
    cut = role.value + ":" + where

    def crash(name):
        if name == cut:
            raise InjectedCrash(name)

    monkeypatch.setattr(signed, "_cut", crash)
    monkeypatch.setattr(persistence, "_persistence_cut", crash)
    with pytest.raises(InjectedCrash):
        signed.sign_root_proof_issuance_attempt(reservation)
    before = calls.copy()
    monkeypatch.setattr(signed, "_cut", lambda name: None)
    monkeypatch.setattr(persistence, "_persistence_cut", lambda name: None)
    with pytest.raises(persistence.AttemptConflictError, match="SIGNATURE_OUTCOME_NOT_DURABLE"):
        signed.resume_root_proof_issuance_attempt(binding, auth)
    assert calls == before
    assert calls[0] <= 1 and calls[1] <= 1
    with upstream._open_store(auth.authorization.trust_domain) as store:
        assert store.attempt(auth.authorization.logical_operation_id).state is (
            persistence.AttemptState.RESERVED_AWAITING_SIGNATURES
            if role is IssuanceSigningRole.REQUESTER
            else persistence.AttemptState.REQUEST_SIGNED_BY_REQUESTER
        )


@requires_posix_custody_locking
@pytest.mark.parametrize("cut", ["identity_inserted", "transition_inserted", "cas_written"])
def test_atomic_finalization_cutpoints(simulated_signed_flow, monkeypatch, cut):
    binding, _, auth, reservation, calls, _ = simulated_signed_flow

    def crash(name):
        if name == "finalization:" + cut:
            raise InjectedCrash(name)

    monkeypatch.setattr(persistence, "_persistence_cut", crash)
    with pytest.raises(InjectedCrash):
        signed.sign_root_proof_issuance_attempt(reservation)
    with upstream._open_store(auth.authorization.trust_domain) as store:
        assert (
            store.attempt(auth.authorization.logical_operation_id).state
            is persistence.AttemptState.CLAIMANT_AUTHORIZED
        )
        assert store._connection.execute("SELECT count(*) FROM immutable_attempts").fetchone() == (
            0,
        )
        retained = store.signed_request(auth.authorization.logical_operation_id)
    monkeypatch.setattr(persistence, "_persistence_cut", lambda name: None)
    result = signed.resume_root_proof_issuance_attempt(binding, auth)
    assert calls == [1, 1]
    assert result.canonical_request_bytes == retained.canonical_bytes
    assert result.identity.claimant_authorization_signature_base64url == retained.claimant[1]


@requires_posix_custody_locking
def test_two_threads_converge_on_one_private_invocation_per_role(simulated_signed_flow):
    binding, _, auth, reservation, calls, _ = simulated_signed_flow
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(
            pool.map(lambda _: signed.sign_root_proof_issuance_attempt(reservation), range(2))
        )
    assert results[0].identity == results[1].identity
    assert calls == [1, 1]

    assert signed.resume_root_proof_issuance_attempt(binding, auth).identity == results[0].identity
    assert calls == [1, 1]


@pytest.mark.skipif(not hasattr(os, "fork"), reason="abrupt process crash proof requires fork")
@requires_posix_custody_locking
@pytest.mark.parametrize(
    "cut,blocked",
    [
        (IssuanceSigningRole.REQUESTER.value + ":intent_durable", True),
        (IssuanceSigningRole.REQUESTER.value + ":signer_returned", True),
        (IssuanceSigningRole.REQUESTER.value + ":checkpoint_durable", False),
        (IssuanceSigningRole.CLAIMANT.value + ":checkpoint_durable", False),
        ("finalization:cas_written", False),
        ("finalization_durable", False),
    ],
)
def test_abrupt_process_exit_retains_checkpoints_or_blocks_uncertain_invocation(
    simulated_signed_flow,
    monkeypatch,
    tmp_path,
    cut,
    blocked,
):
    binding, _, auth, reservation, _, _ = simulated_signed_flow
    log = tmp_path / "signer-calls"
    parent_pid = os.getpid()
    for cls, method, label in (
        (custody.LocalCHARequesterSigningCustody, "sign_issuance_request", b"requester\n"),
        (
            custody.LocalPreaccountClaimantAuthorizationCustody,
            "authorize_entitlement_claim",
            b"claimant\n",
        ),
    ):
        original = getattr(cls, method)

        def counted(self, raw, identity, original=original, label=label):
            descriptor = os.open(log, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
            try:
                os.write(descriptor, label)
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            return original(self, raw, identity)

        monkeypatch.setattr(cls, method, counted)

    def terminate(name):
        if os.getpid() != parent_pid and name == cut:
            os._exit(17)

    monkeypatch.setattr(signed, "_cut", terminate)
    monkeypatch.setattr(persistence, "_persistence_cut", terminate)
    child = os.fork()
    if child == 0:
        try:
            signed.sign_root_proof_issuance_attempt(reservation)
        except BaseException:
            os._exit(99)
        os._exit(98)
    _, status = os.waitpid(child, 0)
    assert os.waitstatus_to_exitcode(status) == 17
    before = log.read_bytes() if log.exists() else b""
    if blocked:
        with pytest.raises(persistence.AttemptConflictError, match="SIGNATURE_OUTCOME_NOT_DURABLE"):
            signed.resume_root_proof_issuance_attempt(binding, auth)
        assert (log.read_bytes() if log.exists() else b"") == before
    else:
        result = signed.resume_root_proof_issuance_attempt(binding, auth)
        assert result.state is persistence.AttemptState.SIGNED_IMMUTABLE_DURABLE_NOT_SENT
        assert log.read_bytes() == b"requester\nclaimant\n"


@requires_posix_custody_locking
@pytest.mark.parametrize("role", [0, 1])
@pytest.mark.parametrize("state", ["VERIFY_ONLY", "REVOKED"])
def test_inactive_custody_before_signing_fails_closed(simulated_signed_flow, role, state):
    _, _, _, reservation, calls, admins = simulated_signed_flow
    admins[role].transition_lifecycle(state)
    with pytest.raises(ValueError, match="ACTIVE"):
        signed.sign_root_proof_issuance_attempt(reservation)
    assert calls == [0, 0]


@requires_posix_custody_locking
@pytest.mark.parametrize(
    "role,cut",
    [
        (0, IssuanceSigningRole.REQUESTER.value + ":checkpoint_durable"),
        (1, IssuanceSigningRole.CLAIMANT.value + ":checkpoint_durable"),
    ],
)
def test_changed_custody_after_checkpoint_never_substitutes_key(
    simulated_signed_flow, monkeypatch, role, cut
):
    binding, _, auth, reservation, calls, admins = simulated_signed_flow

    def crash(name):
        if name == cut:
            raise InjectedCrash(name)

    monkeypatch.setattr(signed, "_cut", crash)
    with pytest.raises(InjectedCrash):
        signed.sign_root_proof_issuance_attempt(reservation)
    before = calls.copy()
    admins[role].transition_lifecycle("VERIFY_ONLY")
    monkeypatch.setattr(signed, "_cut", lambda name: None)
    with pytest.raises(ValueError):
        signed.resume_root_proof_issuance_attempt(binding, auth)
    assert calls == before


@requires_posix_custody_locking
@pytest.mark.parametrize("role", ["requester", "claimant"])
def test_relevant_registry_version_changes_fail_before_signing(simulated_signed_flow, role):
    _, provider, _, reservation, calls, _ = simulated_signed_flow
    port = getattr(provider, role + "_registry")
    field = role + "_key_version"
    port.record = replace(port.record, **{field: 2})
    upstream_tests._refresh_active_credential_identity(port)
    with pytest.raises(upstream.RootProofAttemptReservationError):
        signed.sign_root_proof_issuance_attempt(reservation)
    assert calls == [0, 0]


def test_raw_identity_signature_and_rows_never_grant_authority(store_flow):
    path, _, current, _, _ = store_flow
    for value in (
        current,
        asdict(current),
        current.reservation,
        b"signature",
        object.__new__(signed.VerifiedSignedImmutableRootProofIssuanceAttempt),
    ):
        with pytest.raises(signed.SignedIssuanceAttemptError):
            signed.require_verified_signed_immutable_root_proof_issuance_attempt(value)
        with pytest.raises(signed.SignedIssuanceAttemptError):
            signed.sign_root_proof_issuance_attempt(value)
    with pytest.raises(TypeError):
        signed.VerifiedSignedImmutableRootProofIssuanceAttempt()
    with pytest.raises(TypeError):
        type("Fake", (signed.VerifiedSignedImmutableRootProofIssuanceAttempt,), {})


@requires_posix_custody_locking
def test_capability_is_uncopyable_and_revalidates_current_state(simulated_signed_flow):
    _, provider, _, reservation, _, _ = simulated_signed_flow
    result = signed.sign_root_proof_issuance_attempt(reservation)
    for copier in (copy.copy, copy.deepcopy):
        with pytest.raises(TypeError):
            copier(result)
    provider.available = False
    with pytest.raises(upstream.RootProofAttemptReservationError):
        _ = result.identity


def downgrade_unsigned_fixture_to_v5(path):
    with sqlite3.connect(path) as db:
        for table in ("signature_checkpoints", "signing_intents", "issuance_requests"):
            db.execute("DROP TABLE " + table)
        db.execute("DROP TRIGGER store_metadata_immutable_update")
        db.execute("UPDATE store_metadata SET schema_version=5")
        db.execute(
            "CREATE TRIGGER store_metadata_immutable_update BEFORE UPDATE ON store_metadata "
            "BEGIN SELECT RAISE(ABORT,'immutable metadata'); END"
        )


def test_v5_migration_preserves_every_reservation_byte_and_id(store_flow):
    path, auth, current, _, _ = store_flow
    with persistence.SQLiteCHAAttemptStore(path, auth.trust_domain) as store:
        another = replace(auth, logical_operation_id="ago_018f3e70-7b5a-7c21-8b9a-0123456789ac")
        other = store.reserve_or_resolve_attempt_id(another)
    with sqlite3.connect(path) as db:
        before = db.execute("SELECT * FROM reservations").fetchall()
        assert len(before) == 2
    downgrade_unsigned_fixture_to_v5(path)
    with persistence.SQLiteCHAAttemptStore(path, auth.trust_domain) as store:
        assert store._connection.execute(
            "SELECT schema_version FROM store_metadata"
        ).fetchone() == (6,)
        assert store._connection.execute("SELECT * FROM reservations").fetchall() == before
        assert store.attempt(auth.logical_operation_id) == current
        assert store.attempt(another.logical_operation_id) == other
        assert store._connection.execute(
            "SELECT count(*) FROM signature_checkpoints"
        ).fetchone() == (0,)
    with persistence.SQLiteCHAAttemptStore(path, auth.trust_domain) as store:
        assert store.attempt(auth.logical_operation_id) == current


def test_v5_migration_failure_rolls_back_entire_schema(store_flow, monkeypatch):
    from bot_core import cha_attempt_signatures

    path, auth, current, _, _ = store_flow
    downgrade_unsigned_fixture_to_v5(path)

    def fail(db):
        db.execute(cha_attempt_signatures.CREATE_STATEMENTS[0])
        raise InjectedCrash("migration")

    monkeypatch.setattr(cha_attempt_signatures, "create_schema", fail)
    with pytest.raises(InjectedCrash):
        persistence.SQLiteCHAAttemptStore(path, auth.trust_domain)
    with sqlite3.connect(path) as db:
        assert db.execute("SELECT schema_version FROM store_metadata").fetchone() == (5,)
        assert (
            db.execute("SELECT name FROM sqlite_master WHERE name='issuance_requests'").fetchone()
            is None
        )
        assert db.execute("SELECT attempt_id FROM reservations").fetchone() == (
            current.reservation.issuance_attempt_id,
        )


@pytest.mark.parametrize("table", ["issuance_requests", "signing_intents", "signature_checkpoints"])
@pytest.mark.parametrize("operation", ["UPDATE", "DELETE"])
def test_new_records_are_immutable(store_flow, table, operation):
    path, auth, current, keys, identities = store_flow
    with persistence.SQLiteCHAAttemptStore(path, auth.trust_domain) as store:
        checkpoint(store, current, keys[0], identities[0])
        statement = (
            "DELETE FROM " + table
            if operation == "DELETE"
            else "UPDATE " + table + " SET attempt_id=attempt_id"
        )
        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            store._connection.execute(statement)


@pytest.mark.parametrize("mutation", ["bytes", "reference", "signer", "signature"])
def test_privileged_rewrite_is_corruption(store_flow, mutation):
    path, auth, current, keys, identities = store_flow
    with persistence.SQLiteCHAAttemptStore(path, auth.trust_domain) as store:
        checkpoint(store, current, keys[0], identities[0])
    table = "issuance_requests" if mutation in {"bytes", "reference"} else "signature_checkpoints"
    with sqlite3.connect(path) as db:
        db.execute(f"DROP TRIGGER {table}_immutable_update")
        if mutation == "bytes":
            db.execute("UPDATE issuance_requests SET canonical_bytes=?", (b"{}",))
        elif mutation == "reference":
            db.execute(
                "UPDATE issuance_requests SET reference=?", ("immutable:req:sha256:" + "a" * 64,)
            )
        elif mutation == "signer":
            db.execute("UPDATE signature_checkpoints SET signer_json=?", (b"{}",))
        else:
            db.execute(
                "UPDATE signature_checkpoints SET signature=?", (encode_signature(b"a" * 64),)
            )
        db.execute(
            f"CREATE TRIGGER {table}_immutable_update BEFORE UPDATE ON {table} "
            f"BEGIN SELECT RAISE(ABORT,'immutable {table}'); END"
        )
    with pytest.raises(persistence.AttemptCorruptError):
        persistence.SQLiteCHAAttemptStore(path, auth.trust_domain)


@requires_posix_custody_locking
def test_distinct_namespace_and_raw_key_binding_and_no_runtime_provisioning(
    simulated_signed_flow, monkeypatch
):
    _, _, auth, _, _, admins = simulated_signed_flow
    request, claim = signed._installed_custody(auth.authorization)
    requester, claimant = request.identity(), claim.identity()
    for field in (
        "public_key_hex",
        "service_namespace",
        "credential_namespace",
        "lifecycle_namespace",
        "key_handle",
    ):
        assert getattr(requester, field) != getattr(claimant, field)
    for runtime in (request, claim):
        for name in (
            "stage",
            "activate",
            "generate",
            "rotate",
            "delete",
            "transition_lifecycle",
            "export_private_key",
        ):
            assert not hasattr(runtime, name)
    monkeypatch.setattr(
        Ed25519PrivateKey, "generate", lambda: pytest.fail("ceremony retry generated a new key")
    )
    record = admins[0].stage(key_id=requester.key_id, key_version=requester.key_version)
    assert record.lifecycle == "ACTIVE" and record.public_key_hex == requester.public_key_hex


@requires_posix_custody_locking
def test_missing_private_key_after_public_draft_never_regenerates(
    native_keyring, tmp_path, monkeypatch
):
    admin = custody.OfflineIssuanceCustodyAdministrator(
        tmp_path,
        role=IssuanceSigningRole.REQUESTER,
        trust_domain="td",
        principal="CryptoHunterAccountAuthority",
    )

    def fail(*args):
        raise InjectedCrash("keyring write")

    monkeypatch.setattr(custody.KeyringSecretStorage, "set_secret", fail)
    with pytest.raises(InjectedCrash):
        admin.stage(key_id="key", key_version=1)
    assert parse_canonical((tmp_path / "custody.json").read_bytes())["lifecycle"] == "STAGED"
    monkeypatch.setattr(
        Ed25519PrivateKey, "generate", lambda: pytest.fail("partial ceremony regenerated a key")
    )
    with pytest.raises(custody.LocalSigningCustodyError, match="unavailable/corrupt"):
        admin.stage(key_id="key", key_version=1)


@requires_posix_custody_locking
def test_wrong_trust_domain_and_caller_namespace_rejected(simulated_signed_flow):
    _, _, auth, reservation, calls, _ = simulated_signed_flow
    requester_path, _ = signed._custody_directories(auth.authorization)
    wrong = custody.LocalCHARequesterSigningCustody(requester_path, trust_domain="wrong-domain")
    with pytest.raises(custody.LocalSigningCustodyError):
        wrong.identity()
    request, _ = signed._installed_custody(auth.authorization)
    raw = request_bytes(auth.authorization, reservation.issuance_attempt_id)
    value = parse_canonical(raw)
    value["issuer_target_namespace"] = "caller-selected"
    with pytest.raises(custody.LocalSigningCustodyError):
        request.sign_issuance_request(canonical_json_bytes(value), request.identity())
    assert calls == [1, 0]  # instrumented operation rejected before private signing


def test_test_identity_cannot_authorize_production(store_flow):
    _, _, _, _, identities = store_flow
    with pytest.raises(ValueError, match="production"):
        replace(identities[0], environment="TEST")


def test_requester_claimant_material_alias_rejected_before_claimant_signing(store_flow):
    path, auth, current, keys, identities = store_flow
    with persistence.SQLiteCHAAttemptStore(path, auth.trust_domain) as store:
        current = checkpoint(store, current, keys[0], identities[0])
        alias = replace(
            identities[1],
            public_key_hex=identities[0].public_key_hex,
            key_material_identity=identities[0].key_material_identity,
        )
        with pytest.raises(persistence.AttemptCorruptError, match="distinct"):
            store.prepare_signature(current, alias)


def test_unequal_finalization_never_overwrites_identity(store_flow):
    path, auth, current, keys, identities = store_flow
    with persistence.SQLiteCHAAttemptStore(path, auth.trust_domain) as store:
        for key, identity in zip(keys, identities, strict=True):
            current = checkpoint(store, current, key, identity)
        identity = final_identity(store, current)
        final = store.finalize_attempt(identity, expected_fence=current.fence)
        for changed in (
            replace(identity, root_proof_issuance_request_signed_payload_digest_sha256="a" * 64),
            replace(identity, requester_signature_base64url=encode_signature(b"a" * 64)),
        ):
            with pytest.raises(persistence.AttemptCorruptError, match="unequal"):
                store.finalize_attempt(changed, expected_fence=final.fence)
        assert store.attempt(auth.logical_operation_id) == final


def test_signed_v5_history_cannot_fabricate_request_bytes(store_flow):
    path, auth, _, _, _ = store_flow
    legacy = replace(
        auth,
        reservation_identity=None,
        reservation_relation=None,
        initial_binding_sha256=None,
        authorization_evidence_sha256=None,
        logical_operation_id="legacy-operation",
        environment="PRODUCTION_LOCAL",
    )
    with persistence.SQLiteCHAAttemptStore(path, auth.trust_domain) as store:
        current = store.reserve_or_resolve_attempt_id(legacy)
        identity = persistence.AttemptIdentity(
            legacy,
            current.reservation.issuance_attempt_id,
            "a" * 64,
            "immutable:req:example:1",
            encode_signature(b"a" * 64),
            encode_signature(b"b" * 64),
            REQUESTER_PROFILE,
            CLAIMANT_PROFILE,
        )
        store.finalize_attempt(identity, expected_fence=1)
    downgrade_unsigned_fixture_to_v5(path)
    with pytest.raises(
        persistence.AttemptSchemaUnsupportedError, match="signed bytes cannot be reconstructed"
    ):
        persistence.SQLiteCHAAttemptStore(path, auth.trust_domain)
    with sqlite3.connect(path) as db:
        assert db.execute("SELECT schema_version FROM store_metadata").fetchone() == (5,)


@requires_posix_custody_locking
@pytest.mark.parametrize("role", ["requester", "claimant"])
def test_registry_change_after_durable_signature_never_signs_again(
    simulated_signed_flow, monkeypatch, role
):
    binding, provider, auth, reservation, calls, _ = simulated_signed_flow
    stop = (
        IssuanceSigningRole.REQUESTER.value + ":checkpoint_durable"
        if role == "requester"
        else IssuanceSigningRole.CLAIMANT.value + ":checkpoint_durable"
    )

    def crash(name):
        if name == stop:
            raise InjectedCrash(name)

    monkeypatch.setattr(signed, "_cut", crash)
    with pytest.raises(InjectedCrash):
        signed.sign_root_proof_issuance_attempt(reservation)
    before = calls.copy()
    port = getattr(provider, role + "_registry")
    port.record = replace(port.record, **{role + "_key_version": 2})
    upstream_tests._refresh_active_credential_identity(port)
    monkeypatch.setattr(signed, "_cut", lambda name: None)
    with pytest.raises(upstream.RootProofAttemptReservationError):
        signed.resume_root_proof_issuance_attempt(binding, auth)
    assert calls == before

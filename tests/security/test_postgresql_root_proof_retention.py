"""Trwała retencja STAGED w prawdziwym PostgreSQL, bez przyznawania emisji.

Prywatne klucze istnieją wyłącznie w testowej warstwie custody. Procesy
odzyskujące dowód otrzymują publiczne metadane i podpisane wcześniej bajty;
nie otrzymują sekretów keyring ani uprawnienia do podpisania nowego dowodu.
"""

from __future__ import annotations

import errno
import hashlib
import multiprocessing
import os
import sys
import uuid
from contextlib import contextmanager
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import psycopg
import pytest
from cryptography.exceptions import UnsupportedAlgorithm
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from psycopg import sql
from psycopg.conninfo import make_conninfo
from psycopg.errors import (
    CheckViolation,
    InsufficientPrivilege,
    InvalidParameterValue,
    UniqueViolation,
)

from bot_core import (
    local_signing_custody as custody,
    postgresql_root_proof_retention as retention,
    postgresql_root_proof_retention_schema as retention_schema,
)
from bot_core.cha_attempt_signatures import RetainedIssuanceRequest
from bot_core.cha_issuance_request import (
    IssuanceSigningRole,
    encode_signature,
    request_bytes,
    request_reference,
)
from bot_core.licensing.canonical import canonical_json_bytes, parse_canonical
from bot_core.postgresql_entitlement_registry import PostgreSQLConnectionConfig
from bot_core.root_proof_issuer_substrate import (
    ProviderRole,
    SecurityProfile,
    SecurityProfileIdentity,
)
from tests.licensing.test_cha_attempt_store_stage9 import authorization
from tests.licensing.test_cha_root_proof_signed_attempt import signer_identity
from tests.security import (
    test_local_signing_custody as custody_tests,
    test_root_proof_issuer_runtime as preflight_tests,
)
from tests.security._local_lifecycle_reader_test_seam import (
    simulate_qualified_local_downgrade_for_reader_test,
)
from tests.security.test_postgresql_checkpoint_authority import BASE_DSN
from tests.security.test_root_proof_issuance_reconciliation import _history_boundary
from tests.security.test_root_proof_issuer_orchestration import (
    _durable_snapshot,
    _no_grant,
    _prohibit_mutations,
)

pytestmark = pytest.mark.external_postgresql
native_keyring = custody_tests.native_keyring
wide = preflight_tests.wide
challenge_harness = preflight_tests.challenge_harness
integration = preflight_tests.integration
package = preflight_tests.package
lppi = preflight_tests.lppi
active = preflight_tests.active
committed = preflight_tests.committed
cha = preflight_tests.cha
reserved = preflight_tests.reserved
installed_signed_attempt = preflight_tests.installed_signed_attempt

ROOT_DOMAIN = b"CRYPTOHUNTER_ACCOUNT_GENESIS_ROOT_PROOF_V1\x00"
ATTEMPT_ID = "rpa_018f3e70-7b5d-7c21-8b9a-0123456789ab"
OTHER_ATTEMPT_ID = "rpa_018f3e70-7b5d-7c21-8b9a-0123456789ac"
ROOT_ID = "rpf_018f3e70-7b5e-7c21-8b9a-0123456789ab"
OTHER_ROOT_ID = "rpf_018f3e70-7b5e-7c21-8b9a-0123456789ac"


def _connection(role: str, **changes: str) -> PostgreSQLConnectionConfig:
    return PostgreSQLConnectionConfig(make_conninfo(BASE_DSN, user=role, **changes))


def _retained(auth, keys, attempt_id=ATTEMPT_ID):
    """Buduje rzeczywiste dwa podpisy testowego, zamkniętego żądania."""
    raw = request_bytes(auth, attempt_id)
    checkpoints = tuple(
        (
            signer_identity(role, auth, private),
            encode_signature(private.sign(role.domain.encode("ascii") + b"\x00" + raw)),
        )
        for role, private in zip(IssuanceSigningRole, keys, strict=True)
    )
    return RetainedIssuanceRequest(
        attempt_id,
        raw,
        request_reference(raw),
        hashlib.sha256(raw).hexdigest(),
        checkpoints[0],
        checkpoints[1],
    )


def _payload(value, *, auth=None, retained_request=None, **changes):
    """Tworzy dokładny payload profilu root proof, bez dodatkowych grantów."""
    auth = value.auth if auth is None else auth
    retained_request = value.retained if retained_request is None else retained_request
    snapshot = value.signer.credential_snapshot()
    fields = {
        "schema_version": "1",
        "environment": auth.environment,
        "trust_domain": auth.trust_domain,
        "issuer_id": value.scope.issuer_id,
        "issuer_key_id": snapshot.credential_identity.credential_identity,
        "issuer_key_version": snapshot.key_version,
        "root_proof_id": ROOT_ID,
        "intended_action": "ACCOUNT_GENESIS_BOOTSTRAP",
        "product_scope": auth.product_scope,
        "account_id": auth.account_id,
        "logical_operation_id": auth.logical_operation_id,
        "canonical_genesis_request_fingerprint_sha256": (
            auth.canonical_genesis_request_fingerprint_sha256
        ),
        "bootstrap_entitlement_id": auth.bootstrap_entitlement_id,
        "entitlement_generation": auth.entitlement_generation,
        "authorization_constraints": {"max_logical_operations": 1, "max_account_ids": 1},
        "issuance_attempt_id": retained_request.attempt_id,
    }
    fields.update(changes)
    return canonical_json_bytes(fields)


def _provider(value, *, connection=None, signer=None, scope=None):
    return retention.PostgreSQLRootProofRetention(
        value.connection if connection is None else connection,
        schema=value.setup.schema,
        scope=value.scope if scope is None else scope,
        verification_authority=value.signer if signer is None else signer,
    )


def _read(value, provider=None, **changes):
    return (value.provider if provider is None else provider).read_prepared(
        authorization=changes.get("authorization", value.auth),
        retained_request=changes.get("retained_request", value.retained),
    )


def _retain(value, provider=None, **changes):
    return (value.provider if provider is None else provider).retain_prepared(
        authorization=changes.get("authorization", value.auth),
        retained_request=changes.get("retained_request", value.retained),
        canonical_payload=changes.get("canonical_payload", value.payload),
        signature=changes.get("signature", value.signature),
    )


@contextmanager
def _retention_boundary(tmp_path, auth, retained_request, keys=()):
    """Instaluje izolowany prawdziwy magazyn dla wskazanego żądania."""
    suffix = uuid.uuid4().hex[:12]
    setup = retention_schema.PostgreSQLRootProofRetentionProvisioning(
        f"rpt_{suffix}", f"rpto_{suffix}", f"rptr_{suffix}", auth.trust_domain
    )
    directory = tmp_path / "root-custody"
    directory.mkdir(parents=True)
    security = SecurityProfileIdentity(SecurityProfile.PRODUCTION_LOCAL, auth.trust_domain)
    secret_admin = custody.NativeKeyringSigningSecretAdministrator(
        index_path=directory / "keyring-index.json",
        security=security,
        role=ProviderRole.ROOT_PROOF_SIGNING,
    )
    metadata = custody.provision_local_signing_authority(
        directory,
        secret_admin,
        security=security,
        role=ProviderRole.ROOT_PROOF_SIGNING,
    )
    signer = custody.LocalRootProofSigningProvider(
        directory, keyring_index_path=directory / "keyring-index.json", security=security
    )
    scope = retention.RootProofRetentionScope(
        SecurityProfile.PRODUCTION_LOCAL,
        auth.environment,
        auth.trust_domain,
        auth.product_scope,
        "IndependentAccountGenesisRootProofIssuer",
    )
    retention_schema.provision_postgresql_root_proof_retention(
        PostgreSQLConnectionConfig(BASE_DSN), setup
    )
    try:
        value = SimpleNamespace(
            auth=auth,
            retained=retained_request,
            keys=keys,
            directory=directory,
            security=security,
            metadata=metadata,
            signer=signer,
            scope=scope,
            setup=setup,
            connection=_connection(setup.runtime_role),
        )
        value.payload = _payload(value)
        value.signature = signer.sign_root_proof(ROOT_DOMAIN + value.payload)
        value.provider = _provider(value)
        yield value
    finally:
        with psycopg.connect(BASE_DSN, autocommit=True) as conn:
            conn.execute(
                sql.SQL("DROP SCHEMA IF EXISTS {} CASCADE").format(sql.Identifier(setup.schema))
            )
            for role in (setup.runtime_role, setup.schema_owner_role):
                conn.execute(sql.SQL("DROP ROLE IF EXISTS {}").format(sql.Identifier(role)))


@pytest.fixture
def prepared_boundary(tmp_path: Path, native_keyring):
    """Izoluje role i schemat, zachowując prawdziwy PostgreSQL i Ed25519.

    Syntetyczny AttemptAuthorization jest wyłącznie tuple magazynu; ta fixture
    nie symuluje zweryfikowanego InitialBinding ani autorytetu emisji Stage 9.
    """
    auth = authorization(bootstrap_entitlement_id="ent_018f3e70-7b5a-7c21-8b9a-0123456789ab")
    keys = (Ed25519PrivateKey.generate(), Ed25519PrivateKey.generate())
    with _retention_boundary(tmp_path, auth, _retained(auth, keys), keys) as value:
        yield value


@contextmanager
def _public_verifier(directory: str, security):
    """Pozostawia prawdziwy verifier Ed25519, zastępując tylko odczyt sekretu.

    Procesy testowe nie podpisują danych. Ładują sprawdzone publiczne rekordy
    custody; wartość klucza prywatnego jest celowo niedostępna.
    """
    path = Path(directory)

    def load_public_metadata(provider):
        return (
            custody._read_record(
                custody._record_path(path, ProviderRole.ROOT_PROOF_SIGNING),
                security,
                ProviderRole.ROOT_PROOF_SIGNING,
            ),
            None,
        )

    with (
        patch.object(custody._LocalSigningProviderBase, "_load_material", load_public_metadata),
        patch.object(
            custody.KeyringSecretStorage, "_ensure_native_backend", lambda self, module: object()
        ),
    ):
        yield custody.LocalRootProofSigningProvider(
            path, keyring_index_path=path / "keyring-index.json", security=security
        )


def _process_retain(arguments, mode, ready, start, results):
    """Niezależny proces zapisuje lub odzyskuje wyłącznie podpisane bajty."""
    connection, schema, scope, auth, retained_request, directory, security, payload, signature = (
        arguments
    )
    try:
        with _public_verifier(directory, security) as signer:
            provider = retention.PostgreSQLRootProofRetention(
                connection, schema=schema, scope=scope, verification_authority=signer
            )
            with psycopg.connect(connection.dsn) as conn:
                backend_pid = conn.execute("SELECT pg_catalog.pg_backend_pid()").fetchone()[0]
            if ready is not None:
                ready.put(backend_pid)
            if start is not None and not start.wait(20):
                raise RuntimeError("przekroczony czas bariery procesów")
            if mode in {"before_commit", "after_commit"}:

                def crash(name):
                    if name == mode:
                        os._exit(73)

                retention._persistence_cut = crash
            if mode == "read":
                record = provider.read_prepared(
                    authorization=auth, retained_request=retained_request
                )
            else:
                record = provider.retain_prepared(
                    authorization=auth,
                    retained_request=retained_request,
                    canonical_payload=payload,
                    signature=signature,
                )
            if results is not None:
                results.put(
                    (
                        "ok",
                        os.getpid(),
                        None
                        if record is None
                        else (
                            record.canonical_envelope,
                            record.canonical_payload,
                            record.signature,
                        ),
                    )
                )
    except Exception as exc:
        if results is not None:
            results.put(("error", os.getpid(), type(exc).__name__))
        else:
            raise


def _process_arguments(value):
    return (
        value.connection,
        value.setup.schema,
        value.scope,
        value.auth,
        value.retained,
        str(value.directory),
        value.security,
        value.payload,
        value.signature,
    )


def _join(process):
    process.join(30)
    if process.is_alive():
        process.terminate()
        process.join(5)
        pytest.fail("proces retencji nie zakończył pracy")


def _row(value):
    with psycopg.connect(BASE_DSN) as conn:
        return conn.execute(
            sql.SQL(
                "SELECT canonical_envelope,canonical_payload,signature,status FROM {}.prepared"
            ).format(sql.Identifier(value.setup.schema))
        ).fetchall()


def _direct_retain(value, envelope, payload, signature):
    """Wywołuje SQL jako runtime bez weryfikacji Ed25519 w backendzie Python."""
    fields = parse_canonical(envelope)
    with psycopg.connect(value.connection.dsn) as conn:
        assert conn.execute("SELECT session_user::text,current_user::text").fetchone() == (
            value.setup.runtime_role,
            value.setup.runtime_role,
        )
        return conn.execute(
            sql.SQL(
                "SELECT envelope,payload,signature FROM {}.retain_prepared(%s,%s,%s,%s,%s,%s,%s,%s)"
            ).format(sql.Identifier(value.setup.schema)),
            (
                canonical_json_bytes(fields["scope"]),
                canonical_json_bytes(value.retained.attempt_id),
                canonical_json_bytes(value.auth.logical_operation_id),
                canonical_json_bytes(
                    {
                        "bootstrap_entitlement_id": value.auth.bootstrap_entitlement_id,
                        "entitlement_generation": value.auth.entitlement_generation,
                    }
                ),
                canonical_json_bytes(ROOT_ID),
                envelope,
                payload,
                signature,
            ),
        ).fetchall()


def _candidate_envelope(value, signature):
    return retention._envelope(
        value.scope,
        value.auth,
        value.retained,
        value.payload,
        signature,
        value.signer.credential_snapshot(),
        parse_canonical(value.payload),
    )


def _privileged_mutation(value, statement, parameters=()):
    """Symuluje jawnie uprzywilejowanego administratora spoza granicy runtime."""
    with psycopg.connect(BASE_DSN) as conn:
        conn.execute(
            sql.SQL("ALTER TABLE {}.prepared DISABLE TRIGGER USER").format(
                sql.Identifier(value.setup.schema)
            )
        )
        conn.execute(statement, parameters)
        for trigger in ("immutable_row", "immutable_truncate"):
            conn.execute(
                sql.SQL("ALTER TABLE {}.prepared ENABLE ALWAYS TRIGGER {}").format(
                    sql.Identifier(value.setup.schema), sql.Identifier(trigger)
                )
            )


def test_restart_and_fresh_connection_recover_byte_identical_staged_record(prepared_boundary):
    value = prepared_boundary
    assert _read(value) is None
    prepared = _retain(value)
    assert prepared.state == "STAGED"
    assert prepared.authorizes_operations is False
    assert prepared.root_proof_id == ROOT_ID
    assert prepared.canonical_payload == value.payload
    assert prepared.signature == value.signature
    assert prepared.credential_snapshot == value.signer.credential_snapshot()
    with pytest.raises(FrozenInstanceError):
        prepared.state = "ISSUED"
    restarted = _provider(value)
    assert _read(value, restarted) == prepared
    assert _retain(value, restarted) == prepared
    assert _row(value) == [
        (prepared.canonical_envelope, prepared.canonical_payload, prepared.signature, "STAGED")
    ]
    ctx = multiprocessing.get_context("spawn")
    results = ctx.Queue()
    child = ctx.Process(
        target=_process_retain, args=(_process_arguments(value), "read", None, None, results)
    )
    child.start()
    outcome = results.get(timeout=30)
    _join(child)
    assert child.exitcode == 0
    assert outcome == (
        "ok",
        child.pid,
        (prepared.canonical_envelope, value.payload, value.signature),
    )


def test_two_independent_processes_retain_exactly_one_proof(prepared_boundary):
    value = prepared_boundary
    ctx = multiprocessing.get_context("spawn")
    ready, results, start = ctx.Queue(), ctx.Queue(), ctx.Event()
    children = [
        ctx.Process(
            target=_process_retain,
            args=(_process_arguments(value), "retain", ready, start, results),
        )
        for _ in range(2)
    ]
    try:
        for child in children:
            child.start()
        backend_pids = [ready.get(timeout=30) for _ in children]
        assert len(set(backend_pids)) == 2
        start.set()
        outcomes = [results.get(timeout=30) for _ in children]
    finally:
        start.set()
        for child in children:
            _join(child)
    assert all(child.exitcode == 0 for child in children)
    assert {result[1] for result in outcomes} == {child.pid for child in children}
    assert all(result[0] == "ok" for result in outcomes), outcomes
    assert outcomes[0][2] == outcomes[1][2]
    assert len(_row(value)) == 1
    assert _read(value).canonical_envelope == outcomes[0][2][0]


@pytest.mark.parametrize("cut", ["before_commit", "after_commit"])
def test_process_death_before_and_after_commit_preserves_exact_recovery(prepared_boundary, cut):
    value = prepared_boundary
    ctx = multiprocessing.get_context("spawn")
    child = ctx.Process(
        target=_process_retain, args=(_process_arguments(value), cut, None, None, None)
    )
    child.start()
    _join(child)
    assert child.exitcode == 73
    recovered = _read(value, _provider(value))
    if cut == "before_commit":
        assert recovered is None
        assert _row(value) == []
    else:
        assert recovered is not None
        assert recovered.canonical_payload == value.payload
        assert recovered.signature == value.signature
    resumed = _retain(value, _provider(value))
    assert resumed.canonical_payload == value.payload
    assert resumed.signature == value.signature
    assert len(_row(value)) == 1


def test_lost_commit_response_is_indeterminate_then_recovers_exact_bytes(
    prepared_boundary, monkeypatch
):
    value = prepared_boundary

    def lose_response(cut):
        if cut == "after_commit":
            raise OSError("testowa utrata odpowiedzi po COMMIT")

    monkeypatch.setattr(retention, "_persistence_cut", lose_response)
    with pytest.raises(retention.RootProofRetentionIndeterminate):
        _retain(value)
    monkeypatch.setattr(retention, "_persistence_cut", lambda name: None)
    recovered = _read(value, _provider(value))
    assert recovered is not None
    assert recovered.canonical_payload == value.payload
    assert recovered.signature == value.signature
    assert _retain(value) == recovered
    assert len(_row(value)) == 1


@pytest.mark.parametrize(
    "change",
    [
        {"root_proof_id": OTHER_ROOT_ID},
        {"authorization_evidence_sha256": "5" * 64},
        {"requester_key_version": 2},
        {"claimant_key_id": "inna-tozsamosc-claimant"},
        {"account_id": "acct_018f3e70-7b5c-7c21-8b9a-0123456789ac"},
        {"logical_operation_id": "ago_018f3e70-7b5a-7c21-8b9a-0123456789ac"},
        {"entitlement_generation": 2},
    ],
)
def test_same_attempt_conflicting_valid_candidate_never_replaces_record(prepared_boundary, change):
    value = prepared_boundary
    original = _retain(value)
    payload_changes = change if "root_proof_id" in change else {}
    auth = value.auth if payload_changes else replace(value.auth, **change)
    retained_request = _retained(auth, value.keys)
    payload = _payload(value, auth=auth, retained_request=retained_request, **payload_changes)
    signature = value.signer.sign_root_proof(ROOT_DOMAIN + payload)
    with pytest.raises((retention.RootProofRetentionConflict, retention.RootProofRetentionCorrupt)):
        _retain(
            value,
            authorization=auth,
            retained_request=retained_request,
            canonical_payload=payload,
            signature=signature,
        )
    assert _read(value) == original
    assert len(_row(value)) == 1


def test_changed_signature_for_same_identifier_is_rejected(prepared_boundary):
    value = prepared_boundary
    original = _retain(value)
    changed = bytes([value.signature[0] ^ 1]) + value.signature[1:]
    with pytest.raises(retention.RootProofRetentionConflict):
        _retain(value, signature=changed)
    assert _read(value) == original


def test_direct_sql_invalid_signature_durably_fences_retry_but_never_becomes_proof(
    prepared_boundary, monkeypatch
):
    """EXECUTE pozwala zatrzymać tożsamość, lecz nie ominąć weryfikacji proof.

    To test istniejącej granicy zaufania: runtime SQL nie dowodzi, że caller
    wykonał wcześniej weryfikację kryptograficzną Python.
    """
    value = prepared_boundary
    invalid = bytes([value.signature[0] ^ 1]) + value.signature[1:]
    snapshot = value.signer.credential_snapshot()
    assert value.signer.verify_historical(ROOT_DOMAIN + value.payload, value.signature, snapshot)
    assert value.signer.verify_historical(ROOT_DOMAIN + value.payload, invalid, snapshot) is False
    poisoned_envelope = _candidate_envelope(value, invalid)
    correct_envelope = _candidate_envelope(value, value.signature)
    poisoned_bytes = (poisoned_envelope, value.payload, invalid)

    assert _direct_retain(value, *poisoned_bytes) == [poisoned_bytes]
    assert _row(value) == [(*poisoned_bytes, "STAGED")]
    # Nowe połączenie i exact SQL retry zachowują oryginalny, niepoprawny podpis.
    assert _direct_retain(value, *poisoned_bytes) == [poisoned_bytes]
    restarted = _provider(value)

    def prohibit_signing(*args, **kwargs):
        pytest.fail("zatrucie STAGED wywołało podpisanie nowych danych")

    monkeypatch.setattr(custody.LocalRootProofSigningProvider, "sign_root_proof", prohibit_signing)
    with pytest.raises(retention.RootProofRetentionCorrupt):
        _read(value, restarted)
    with pytest.raises(retention.RootProofRetentionCorrupt):
        _retain(value, restarted)
    with pytest.raises(retention.RootProofRetentionCorrupt):
        restarted.require_issued(authorization=value.auth, retained_request=value.retained)
    with pytest.raises(UniqueViolation, match="conflicting retained bytes"):
        _direct_retain(value, correct_envelope, value.payload, value.signature)
    for statement in (
        "UPDATE {}.prepared SET signature=%s",
        "DELETE FROM {}.prepared WHERE signature=%s",
        "TRUNCATE {}.prepared",
    ):
        parameters = () if statement.startswith("TRUNCATE") else (value.signature,)
        with psycopg.connect(value.connection.dsn) as conn:
            with pytest.raises(InsufficientPrivilege):
                conn.execute(
                    sql.SQL(statement).format(sql.Identifier(value.setup.schema)), parameters
                )
    assert _row(value) == [(*poisoned_bytes, "STAGED")]


@pytest.mark.parametrize(
    "malformed", ["signature_length", "payload_identity", "noncanonical", "issued"]
)
def test_direct_sql_still_rejects_structure_identity_and_issuance_grant(
    prepared_boundary, malformed
):
    value = prepared_boundary
    envelope = _candidate_envelope(value, value.signature)
    payload, signature = value.payload, value.signature
    if malformed == "signature_length":
        signature = signature[:-1]
    elif malformed == "payload_identity":
        fields = parse_canonical(payload)
        fields["account_id"] = "acct_018f3e70-7b5c-7c21-8b9a-0123456789ac"
        payload = canonical_json_bytes(fields)
    elif malformed == "noncanonical":
        payload = b" " + payload
    else:
        fields = parse_canonical(envelope)
        fields["state"] = "ISSUED"
        envelope = canonical_json_bytes(fields)
    with pytest.raises((CheckViolation, InvalidParameterValue)):
        _direct_retain(value, envelope, payload, signature)
    assert _row(value) == []


@pytest.mark.parametrize("collision", ["operation", "entitlement", "account", "root_proof"])
def test_new_attempt_cannot_reuse_fenced_operation_entitlement_account_or_root(
    prepared_boundary, collision
):
    value = prepared_boundary
    original = _retain(value)
    changes = {
        "logical_operation_id": "ago_018f3e70-7b5a-7c21-8b9a-0123456789ac",
        "account_id": "acct_018f3e70-7b5c-7c21-8b9a-0123456789ac",
        "bootstrap_entitlement_id": "ent_018f3e70-7b5a-7c21-8b9a-0123456789ac",
    }
    if collision != "root_proof":
        field = {
            "operation": "logical_operation_id",
            "entitlement": "bootstrap_entitlement_id",
            "account": "account_id",
        }[collision]
        changes.pop(field)
    auth = replace(value.auth, **changes)
    retained_request = _retained(auth, value.keys, OTHER_ATTEMPT_ID)
    payload = _payload(
        value,
        auth=auth,
        retained_request=retained_request,
        root_proof_id=ROOT_ID if collision == "root_proof" else OTHER_ROOT_ID,
    )
    signature = value.signer.sign_root_proof(ROOT_DOMAIN + payload)
    with pytest.raises(retention.RootProofRetentionConflict):
        _retain(
            value,
            authorization=auth,
            retained_request=retained_request,
            canonical_payload=payload,
            signature=signature,
        )
    assert _read(value) == original
    assert len(_row(value)) == 1


def test_other_issuer_key_cannot_reinterpret_original_record(prepared_boundary):
    value = prepared_boundary
    original = _retain(value)
    directory = value.directory.parent / "other-root-custody"
    directory.mkdir()
    admin = custody.NativeKeyringSigningSecretAdministrator(
        index_path=directory / "keyring-index.json",
        security=value.security,
        role=ProviderRole.ROOT_PROOF_SIGNING,
    )
    custody.provision_local_signing_authority(
        directory, admin, security=value.security, role=ProviderRole.ROOT_PROOF_SIGNING
    )
    signer = custody.LocalRootProofSigningProvider(
        directory, keyring_index_path=directory / "keyring-index.json", security=value.security
    )
    provider = _provider(value, signer=signer)
    snapshot = signer.credential_snapshot()
    payload = _payload(
        value,
        issuer_key_id=snapshot.credential_identity.credential_identity,
        issuer_key_version=snapshot.key_version,
    )
    signature = signer.sign_root_proof(ROOT_DOMAIN + payload)
    with pytest.raises(retention.RootProofRetentionCorrupt):
        _read(value, provider)
    with pytest.raises(retention.RootProofRetentionCorrupt):
        _retain(value, provider, canonical_payload=payload, signature=signature)
    assert _read(value) == original


def test_other_issuer_scope_cannot_reuse_globally_fenced_attempt(prepared_boundary):
    value = prepared_boundary
    original = _retain(value)
    scope = replace(value.scope, issuer_id="AnotherIndependentIssuer")
    provider = _provider(value, scope=scope)
    payload = _payload(value, issuer_id=scope.issuer_id)
    signature = value.signer.sign_root_proof(ROOT_DOMAIN + payload)
    assert _read(value, provider) is None
    with pytest.raises(retention.RootProofRetentionConflict):
        _retain(value, provider, canonical_payload=payload, signature=signature)
    assert _read(value) == original
    assert len(_row(value)) == 1


def test_unicode_and_escaped_nul_are_retained_as_exact_original_bytea(prepared_boundary):
    value = prepared_boundary
    principal = "Provisioner 🐢\x00分"
    auth = replace(value.auth, provisioning_principal_id=principal)
    retained_request = _retained(auth, value.keys)
    assert b"\\u0000" in retained_request.canonical_bytes
    prepared = _retain(value, authorization=auth, retained_request=retained_request)
    recovered = _read(
        value, _provider(value), authorization=auth, retained_request=retained_request
    )
    assert recovered == prepared
    assert recovered.authorization.provisioning_principal_id == principal
    assert recovered.retained_request.canonical_bytes == retained_request.canonical_bytes
    assert b"\\u0000" in recovered.canonical_envelope


@pytest.mark.parametrize("column", ["canonical_envelope", "canonical_payload", "signature"])
def test_single_byte_persisted_corruption_fails_closed(prepared_boundary, column):
    value = prepared_boundary
    _retain(value)
    _privileged_mutation(
        value,
        sql.SQL("UPDATE {}.prepared SET {}=set_byte({},0,get_byte({},0)#1)").format(
            sql.Identifier(value.setup.schema),
            sql.Identifier(column),
            sql.Identifier(column),
            sql.Identifier(column),
        ),
    )
    with pytest.raises(retention.RootProofRetentionCorrupt):
        _read(value)
    with pytest.raises(retention.RootProofRetentionCorrupt):
        _retain(value)


@pytest.mark.parametrize(
    "tamper",
    [
        "extra_snapshot",
        "bad_provider",
        "bad_security",
        "lifecycle",
        "lifecycle_generation",
        "payload_digest",
        "grant",
    ],
)
def test_valid_canonical_envelope_cannot_launder_corrupted_facts(prepared_boundary, tamper):
    value = prepared_boundary
    original = _retain(value)
    envelope = parse_canonical(original.canonical_envelope)
    snapshot = envelope["credential_snapshot"]
    if tamper == "extra_snapshot":
        snapshot["caller_authorized"] = True
    elif tamper == "bad_provider":
        snapshot["provider_identity"] = "DTO"
    elif tamper == "bad_security":
        snapshot["provider_identity"]["security"] = "PRODUCTION_LOCAL"
    elif tamper == "lifecycle":
        snapshot["lifecycle_state"] = "VERIFY_ONLY"
    elif tamper == "lifecycle_generation":
        snapshot["lifecycle_generation"] = 3
    elif tamper == "payload_digest":
        envelope["canonical_payload_sha256"] = "f" * 64
    else:
        envelope["state"] = "ISSUED"
    changed = canonical_json_bytes(envelope)
    _privileged_mutation(
        value,
        sql.SQL("UPDATE {}.prepared SET canonical_envelope=%s").format(
            sql.Identifier(value.setup.schema)
        ),
        (changed,),
    )
    with pytest.raises(retention.RootProofRetentionCorrupt):
        _read(value)


@pytest.mark.parametrize(
    "change",
    [
        {"account_id": "acct_018f3e70-7b5c-7c21-8b9a-0123456789ac"},
        {"logical_operation_id": "ago_018f3e70-7b5a-7c21-8b9a-0123456789ac"},
        {"bootstrap_entitlement_id": "ent_018f3e70-7b5a-7c21-8b9a-0123456789ac"},
        {"entitlement_generation": 2},
    ],
)
def test_authorization_mismatch_cannot_recover_other_tuple(prepared_boundary, change):
    value = prepared_boundary
    original = _retain(value)
    with pytest.raises(retention.RootProofRetentionConflict):
        _read(value, authorization=replace(value.auth, **change))
    assert _read(value) == original


def test_attempt_mismatch_cannot_recover_other_request(prepared_boundary):
    value = prepared_boundary
    original = _retain(value)
    with pytest.raises(retention.RootProofRetentionConflict):
        _read(value, retained_request=replace(value.retained, attempt_id=OTHER_ATTEMPT_ID))
    assert _read(value) == original


def test_verify_only_retry_preserves_original_snapshot_without_signing(
    prepared_boundary, monkeypatch
):
    value = prepared_boundary
    original = _retain(value)
    simulate_qualified_local_downgrade_for_reader_test(
        value.directory,
        security=value.security,
        target=custody.SigningKeyLifecycle.VERIFY_ONLY,
    )

    def prohibit_signing(*args, **kwargs):
        pytest.fail("retry STAGED wywołał podpisanie nowych danych")

    monkeypatch.setattr(custody.LocalRootProofSigningProvider, "sign_root_proof", prohibit_signing)
    restarted = _provider(value)
    assert value.signer.credential_snapshot().lifecycle_generation == 2
    assert _read(value, restarted) == original
    assert _retain(value, restarted) == original
    assert original.credential_snapshot.lifecycle_state is custody.SigningKeyLifecycle.ACTIVE
    assert original.credential_snapshot.lifecycle_generation == 1
    assert len(_row(value)) == 1


def test_verify_only_does_not_stage_new_signature(prepared_boundary):
    value = prepared_boundary
    simulate_qualified_local_downgrade_for_reader_test(
        value.directory,
        security=value.security,
        target=custody.SigningKeyLifecycle.VERIFY_ONLY,
    )
    with pytest.raises(retention.RootProofRetentionConflict):
        _retain(value)
    assert _row(value) == []


def test_revoked_signer_cannot_establish_historical_trust(prepared_boundary):
    value = prepared_boundary
    _retain(value)
    simulate_qualified_local_downgrade_for_reader_test(
        value.directory,
        security=value.security,
        target=custody.SigningKeyLifecycle.REVOKED,
    )
    with pytest.raises(retention.RootProofRetentionCorrupt):
        _read(value)
    with pytest.raises(retention.RootProofRetentionCorrupt):
        _retain(value)
    assert len(_row(value)) == 1


def test_staged_recovery_explicitly_refuses_issued_proof(prepared_boundary):
    value = prepared_boundary
    for retain_first in (False, True):
        if retain_first:
            _retain(value)
        with pytest.raises(retention.RootProofRetentionNotIssued):
            value.provider.require_issued(authorization=value.auth, retained_request=value.retained)
    assert _read(value).authorizes_operations is False


def test_database_failure_is_unavailable_never_authoritative_absence(
    prepared_boundary, monkeypatch
):
    value = prepared_boundary
    unavailable = PostgreSQLConnectionConfig(
        make_conninfo(value.connection.dsn, host="127.0.0.1", port=1, connect_timeout=1)
    )
    with pytest.raises(retention.RootProofRetentionUnavailable):
        _provider(value, connection=unavailable)

    def disconnected(*args, **kwargs):
        raise psycopg.OperationalError("testowa niedostępność magazynu")

    with monkeypatch.context() as outage:
        outage.setattr(psycopg, "connect", disconnected)
        with pytest.raises(retention.RootProofRetentionUnavailable):
            _read(value)
        with pytest.raises(retention.RootProofRetentionUnavailable):
            _retain(value)


def test_privileged_valid_rollback_exposes_storage_limit_without_issuance(prepared_boundary):
    """Usunięcie ważnego prefixu wymaga niezależnego checkpointu do wykrycia.

    Ten magazyn nie ma takiego checkpointu; administrator systemu lub bazy
    pozostaje poza deklarowaną granicą ochrony retencji PRODUCTION_LOCAL.
    """
    value = prepared_boundary
    _retain(value)
    _privileged_mutation(
        value,
        sql.SQL("DELETE FROM {}.prepared").format(sql.Identifier(value.setup.schema)),
    )
    restarted = _provider(value)
    assert _read(value, restarted) is None
    with pytest.raises(retention.RootProofRetentionNotIssued):
        restarted.require_issued(authorization=value.auth, retained_request=value.retained)


@pytest.mark.parametrize("operation", ["INSERT", "UPDATE", "DELETE", "TRUNCATE", "SELECT"])
def test_runtime_role_has_no_direct_table_access(prepared_boundary, operation):
    value = prepared_boundary
    _retain(value)
    statements = {
        "INSERT": "INSERT INTO {}.prepared DEFAULT VALUES",
        "UPDATE": "UPDATE {}.prepared SET status='STAGED'",
        "DELETE": "DELETE FROM {}.prepared",
        "TRUNCATE": "TRUNCATE {}.prepared",
        "SELECT": "SELECT * FROM {}.prepared",
    }
    with psycopg.connect(value.connection.dsn) as conn:
        with pytest.raises(InsufficientPrivilege):
            conn.execute(sql.SQL(statements[operation]).format(sql.Identifier(value.setup.schema)))


@pytest.mark.parametrize(
    "tamper",
    [
        "table_acl",
        "column_acl",
        "default_acl",
        "function_acl",
        "function_security",
        "unlogged",
        "membership",
    ],
)
def test_schema_tamper_is_rejected_before_recovery(prepared_boundary, tamper):
    value = prepared_boundary
    _retain(value)
    schema = sql.Identifier(value.setup.schema)
    runtime_role = sql.Identifier(value.setup.runtime_role)
    owner_role = sql.Identifier(value.setup.schema_owner_role)
    statements = {
        "table_acl": sql.SQL("GRANT SELECT ON {}.prepared TO {}").format(schema, runtime_role),
        "column_acl": sql.SQL("GRANT SELECT(signature) ON {}.prepared TO {}").format(
            schema, runtime_role
        ),
        "default_acl": sql.SQL(
            "ALTER DEFAULT PRIVILEGES FOR ROLE {} IN SCHEMA {} GRANT SELECT ON TABLES TO {}"
        ).format(owner_role, schema, runtime_role),
        "function_acl": sql.SQL(
            "GRANT EXECUTE ON FUNCTION {}.read_prepared(bytea,bytea) TO PUBLIC"
        ).format(schema),
        "function_security": sql.SQL(
            "ALTER FUNCTION {}.read_prepared(bytea,bytea) SECURITY INVOKER"
        ).format(schema),
        "unlogged": sql.SQL("ALTER TABLE {}.prepared SET UNLOGGED").format(schema),
        "membership": sql.SQL("GRANT {} TO {}").format(owner_role, runtime_role),
    }
    with psycopg.connect(BASE_DSN, autocommit=True) as conn:
        conn.execute(statements[tamper])
    with pytest.raises(retention.RootProofRetentionUnavailable):
        _provider(value)
    with pytest.raises(retention.RootProofRetentionUnavailable):
        _read(value)


def test_runtime_extra_grant_outside_retention_schema_is_rejected(prepared_boundary):
    value = prepared_boundary
    outside = sql.Identifier("outside_retention_" + uuid.uuid4().hex[:12])
    role = sql.Identifier(value.setup.runtime_role)
    with psycopg.connect(BASE_DSN, autocommit=True) as conn:
        conn.execute(sql.SQL("CREATE TABLE public.{}(value integer)").format(outside))
        try:
            conn.execute(sql.SQL("GRANT SELECT ON public.{} TO {}").format(outside, role))
            with pytest.raises(retention.RootProofRetentionUnavailable):
                _provider(value)
            with pytest.raises(retention.RootProofRetentionUnavailable):
                _read(value)
        finally:
            conn.execute(sql.SQL("DROP TABLE public.{}").format(outside))


@pytest.mark.parametrize("tamper", ["rename_input", "reorder_inputs", "out_type"])
def test_function_argument_semantics_are_qualified_beyond_source_hash(prepared_boundary, tamper):
    """Metadane parametrów zmieniają semantykę także przy identycznym prosrc."""
    value = prepared_boundary
    signature = value.setup.schema + ".read_prepared(bytea,bytea)"
    with psycopg.connect(BASE_DSN) as conn:
        before = conn.execute(
            "SELECT prosrc,proargtypes::text,proacl::text,proargnames,proallargtypes "
            "FROM pg_catalog.pg_proc WHERE oid=%s::pg_catalog.regprocedure",
            (signature,),
        ).fetchone()
        assert before is not None
        if tamper == "out_type":
            types = list(before[4])
            types[2] = 25
            conn.execute(
                "UPDATE pg_catalog.pg_proc SET proallargtypes=%s::oid[] "
                "WHERE oid=%s::pg_catalog.regprocedure",
                (types, signature),
            )
        else:
            names = list(before[3])
            if tamper == "rename_input":
                names[0] = "diverted_scope"
            else:
                names[0], names[1] = names[1], names[0]
            conn.execute(
                "UPDATE pg_catalog.pg_proc SET proargnames=%s "
                "WHERE oid=%s::pg_catalog.regprocedure",
                (names, signature),
            )
        after = conn.execute(
            "SELECT prosrc,proargtypes::text,proacl::text "
            "FROM pg_catalog.pg_proc WHERE oid=%s::pg_catalog.regprocedure",
            (signature,),
        ).fetchone()
        assert after == before[:3]
    with pytest.raises(retention.RootProofRetentionUnavailable):
        _provider(value)
    with pytest.raises(retention.RootProofRetentionUnavailable):
        _read(value)


def test_session_synchronous_commit_off_is_rejected(prepared_boundary):
    value = prepared_boundary
    connection = _connection(value.setup.runtime_role, options="-c synchronous_commit=off")
    with pytest.raises(retention.RootProofRetentionUnavailable):
        _provider(value, connection=connection)


def test_fsync_off_prevents_retention_installation_on_separate_real_postgresql():
    """Osobny prawdziwy klaster z fsync=off nie kwalifikuje trwałej retencji."""
    unsafe_dsn = os.environ.get(
        "CRYPTOHUNTER_TEST_POSTGRES_UNSAFE_DSN",
        "host=127.0.0.1 port=55433 dbname=postgres user=postgres connect_timeout=2",
    )
    suffix = uuid.uuid4().hex[:12]
    setup = retention_schema.PostgreSQLRootProofRetentionProvisioning(
        f"unsafe_rpt_{suffix}",
        f"unsafe_rpto_{suffix}",
        f"unsafe_rptr_{suffix}",
        "td-unsafe-retention-test",
    )
    with psycopg.connect(unsafe_dsn) as conn:
        assert conn.execute("SELECT pg_catalog.current_setting('fsync')").fetchone() == ("off",)
    with pytest.raises(retention_schema.RetentionSchemaQualificationError, match="durability"):
        retention_schema.provision_postgresql_root_proof_retention(
            PostgreSQLConnectionConfig(unsafe_dsn), setup
        )
    with psycopg.connect(unsafe_dsn) as conn:
        assert (
            conn.execute(
                "SELECT 1 FROM pg_catalog.pg_namespace WHERE nspname=%s", (setup.schema,)
            ).fetchone()
            is None
        )
        assert (
            conn.execute(
                "SELECT 1 FROM pg_catalog.pg_roles WHERE rolname=ANY(%s)",
                ([setup.schema_owner_role, setup.runtime_role],),
            ).fetchone()
            is None
        )


def test_owner_and_bootstrap_connection_cannot_be_runtime_provider(prepared_boundary):
    value = prepared_boundary
    with pytest.raises(retention.RootProofRetentionUnavailable):
        _provider(value, connection=PostgreSQLConnectionConfig(BASE_DSN))


@pytest.mark.parametrize("field", ["account_id", "issuer_key_id", "issuance_attempt_id"])
def test_signed_payload_cannot_override_authorized_identifiers(prepared_boundary, field):
    value = prepared_boundary
    altered = parse_canonical(value.payload)
    replacements = {
        "account_id": "acct_018f3e70-7b5c-7c21-8b9a-0123456789ac",
        "issuer_key_id": "caller-selected-key",
        "issuance_attempt_id": OTHER_ATTEMPT_ID,
    }
    altered[field] = replacements[field]
    raw = canonical_json_bytes(altered)
    signature = value.signer.sign_root_proof(ROOT_DOMAIN + raw)
    with pytest.raises(retention.RootProofRetentionConflict):
        _retain(value, canonical_payload=raw, signature=signature)
    assert _row(value) == []


@pytest.mark.parametrize("malformed", ["whitespace", "extra", "missing", "bad_signature"])
def test_malformed_candidate_fails_closed_without_record(prepared_boundary, malformed):
    value = prepared_boundary
    raw, signature = value.payload, value.signature
    if malformed == "whitespace":
        raw = b" " + raw
    elif malformed == "extra":
        raw = canonical_json_bytes({**parse_canonical(raw), "caller_grant": True})
        signature = value.signer.sign_root_proof(ROOT_DOMAIN + raw)
    elif malformed == "missing":
        payload = parse_canonical(raw)
        payload.pop("authorization_constraints")
        raw = canonical_json_bytes(payload)
        signature = value.signer.sign_root_proof(ROOT_DOMAIN + raw)
    else:
        signature = bytes([signature[0] ^ 1]) + signature[1:]
    with pytest.raises(retention.RootProofRetentionConflict):
        _retain(value, canonical_payload=raw, signature=signature)
    assert _row(value) == []


@pytest.mark.parametrize("checkpoint", ["requester", "claimant"])
def test_invalid_retained_request_signature_never_reaches_storage(prepared_boundary, checkpoint):
    value = prepared_boundary
    signer, signature = getattr(value.retained, checkpoint)
    corrupted = encode_signature(b"\x00" * 64)
    assert signature != corrupted
    request = replace(value.retained, **{checkpoint: (signer, corrupted)})
    with pytest.raises(retention.RootProofRetentionConflict):
        _retain(value, retained_request=request)
    assert _row(value) == []


def test_requester_and_claimant_must_use_distinct_key_material(prepared_boundary):
    value = prepared_boundary
    request = _retained(value.auth, (value.keys[0], value.keys[0]))
    with pytest.raises(retention.RootProofRetentionConflict):
        _retain(value, retained_request=request)
    assert _row(value) == []


def test_dto_or_subclass_cannot_replace_exact_custody_authority(prepared_boundary):
    value = prepared_boundary
    with pytest.raises(TypeError, match="EXACT_RETENTION_COMPOSITION_REQUIRED"):
        _provider(value, signer=SimpleNamespace(credential_snapshot=lambda: None))

    class UnreviewedProvider(custody.LocalRootProofSigningProvider):
        pass

    signer = object.__new__(UnreviewedProvider)
    with pytest.raises(TypeError, match="EXACT_RETENTION_COMPOSITION_REQUIRED"):
        _provider(value, signer=signer)


def test_genuine_signed_attempt_retention_keeps_public_orchestration_blocked(
    installed_signed_attempt, tmp_path, monkeypatch
):
    """Pełna genuine lineage nie zmienia STAGED w emisję ani grant operacji."""
    from bot_core import root_proof_issuer_orchestration as orchestration

    genuine = installed_signed_attempt
    with _history_boundary(genuine, monkeypatch) as authorities:
        before = _durable_snapshot(authorities)
        with _retention_boundary(
            tmp_path / "genuine-retention", authorities.auth, genuine.retained
        ) as value:
            retained_proof = _retain(value)
            calls = _prohibit_mutations(authorities, monkeypatch)
            report = orchestration.orchestrate_root_proof_issuance(genuine.binding)
            assert report.disposition == "EXECUTION_BLOCKED"
            _no_grant(report)
            assert calls == []
            assert _durable_snapshot(authorities) == before
            assert _read(value, _provider(value)) == retained_proof
            assert retained_proof.authorizes_operations is False


def _denied(value, expected):
    for operation in (
        lambda: _read(value),
        lambda: _retain(value),
        lambda: value.provider.require_issued(
            authorization=value.auth, retained_request=value.retained
        ),
    ):
        with pytest.raises(expected):
            operation()


def _raised(error, cause=None):
    if cause is not None:
        error.__cause__ = cause
    return error


@pytest.mark.parametrize(
    "error",
    [
        custody.LocalSigningCustodyBusy("custody locking timed out"),
        custody.LocalSigningCustodyError("rotation recovery required"),
        custody.LocalSigningCustodyError("production-local custody locking is unavailable"),
        custody.LocalSigningCustodyError("native Windows custody filesystem is unavailable"),
        OSError(errno.EIO, "custody disk unavailable"),
        UnsupportedAlgorithm("Ed25519 unavailable"),
        ModuleNotFoundError("native dependency unavailable", name="win32file"),
        _raised(
            custody.LocalSigningCustodyError("custody NTFS file qualification failed"),
            _raised(
                custody.LocalSigningCustodyError("custody NTFS directory qualification failed"),
                ModuleNotFoundError("native dependency unavailable", name="win32security"),
            ),
        ),
        _raised(
            custody.LocalSigningCustodyError("custody locking failed"),
            OSError(errno.ENOLCK, "locks unavailable"),
        ),
    ],
    ids=["busy", "rotation", "locking", "windows", "io", "crypto", "import", "ntfs", "lock-io"],
)
def test_historical_operational_refusal_is_unavailable_and_exact_retry_recovers(
    prepared_boundary, monkeypatch, error
):
    value = prepared_boundary
    original = _retain(value)
    rows = _row(value)

    def fail(*args):
        raise error

    with monkeypatch.context() as outage:
        outage.setattr(custody.LocalRootProofSigningProvider, "verify_historical", fail)
        _denied(value, retention.RootProofRetentionUnavailable)
        assert _row(value) == rows
    recovered = _retain(value)
    assert recovered == original
    assert recovered.state == "STAGED"
    assert recovered.authorizes_operations is False
    with pytest.raises(retention.RootProofRetentionNotIssued):
        value.provider.require_issued(authorization=value.auth, retained_request=value.retained)


@pytest.mark.parametrize(
    "error",
    [
        custody.LocalSigningCustodyError("custody material unavailable/corrupt"),
        _raised(
            custody.LocalSigningCustodyError(
                "historical credential facts are stale or do not match"
            ),
            OSError(errno.EIO, "unrelated outage"),
        ),
        _raised(
            custody.LocalSigningCustodyError("custody material unavailable/corrupt"),
            _raised(ValueError("invalid custody record"), OSError(errno.EIO, "unrelated outage")),
        ),
        _raised(
            custody.LocalSigningCustodyError("custody NTFS DACL permits untrusted modification"),
            OSError(errno.EACCES, "unrelated permissions error"),
        ),
        _raised(
            custody.LocalSigningLifecycleError(
                "REVOKED credential cannot establish historical trust"
            ),
            OSError(errno.EIO, "unrelated outage"),
        ),
        custody.LocalSigningLifecycleError("unknown signing credential"),
        _raised(
            custody.LocalSigningCustodyError("custody file open failed"),
            OSError(errno.ELOOP, "unsafe symlink"),
        ),
        OSError(errno.EINVAL, "ambiguous operation"),
        ModuleNotFoundError("unexpected module", name="unreviewed_dependency"),
    ],
    ids=[
        "ambiguous",
        "identity",
        "mixed-cause",
        "acl",
        "revoked",
        "unknown",
        "symlink",
        "errno",
        "import",
    ],
)
def test_historical_integrity_and_ambiguous_refusal_remain_corrupt(
    prepared_boundary, monkeypatch, error
):
    value = prepared_boundary
    _retain(value)
    rows = _row(value)

    def fail(*args):
        raise error

    monkeypatch.setattr(custody.LocalRootProofSigningProvider, "verify_historical", fail)
    _denied(value, retention.RootProofRetentionCorrupt)
    assert _row(value) == rows


@pytest.mark.parametrize("failure_errno", [errno.EACCES, errno.EIO, errno.ENOENT])
def test_actual_custody_record_io_failure_is_unavailable(
    prepared_boundary, monkeypatch, failure_errno
):
    value = prepared_boundary
    original = _retain(value)
    record = custody._record_path(value.directory, ProviderRole.ROOT_PROOF_SIGNING)
    # The custody reader now uses an O_NOFOLLOW/native-qualified descriptor,
    # not Path.read_bytes(). Fail the real descriptor acquisition seam so this
    # still exercises a genuine I/O outage rather than a dead monkeypatch.
    open_descriptor = custody._open_custody_descriptor

    @contextmanager
    def unreadable(path, *, write=False, create=False):
        if path == record:
            raise OSError(failure_errno, "testowa niedostępność publicznego custody")
        with open_descriptor(path, write=write, create=create) as descriptor:
            yield descriptor

    with monkeypatch.context() as outage:
        outage.setattr(custody, "_open_custody_descriptor", unreadable)
        _denied(value, retention.RootProofRetentionUnavailable)
    assert _retain(value) == original


def test_actual_pending_rotation_refuses_without_claiming_corruption(prepared_boundary):
    value = prepared_boundary
    original = _retain(value)
    intent, _, _ = custody._rotation_paths(value.directory, ProviderRole.ROOT_PROOF_SIGNING)
    intent.parent.mkdir()
    intent.write_bytes(b"unfinished offline rotation")
    try:
        _denied(value, retention.RootProofRetentionUnavailable)
    finally:
        intent.unlink()
    assert _retain(value) == original


@pytest.mark.parametrize("change", ["decode", "permissions"])
def test_actual_custody_record_integrity_failure_remains_corrupt(prepared_boundary, change):
    value = prepared_boundary
    original = _retain(value)
    record = custody._record_path(value.directory, ProviderRole.ROOT_PROOF_SIGNING)
    raw, mode = record.read_bytes(), record.stat().st_mode
    if change == "permissions" and os.name != "posix":
        pytest.skip("Negatywny tryb POSIX; odmowę DACL NTFS sprawdza osobny przypadek")
    try:
        if change == "decode":
            record.write_bytes(b"malformed public custody metadata")
        else:
            record.chmod(0o666)
        _denied(value, retention.RootProofRetentionCorrupt)
    finally:
        record.write_bytes(raw)
        record.chmod(mode)
    assert _retain(value) == original


@pytest.mark.parametrize("change", ["envelope", "signature"])
def test_retained_integrity_failure_precedes_custody_outage(prepared_boundary, monkeypatch, change):
    value = prepared_boundary
    original = _retain(value)
    envelope = parse_canonical(original.canonical_envelope)
    signature = original.signature
    if change == "envelope":
        envelope["canonical_payload_sha256"] = "f" * 64
    else:
        signature = bytes([signature[0] ^ 1]) + signature[1:]
        envelope["signature_ed25519_base64url"] = retention.encode_signature(signature)
    _privileged_mutation(
        value,
        sql.SQL("UPDATE {}.prepared SET canonical_envelope=%s,signature=%s").format(
            sql.Identifier(value.setup.schema)
        ),
        (canonical_json_bytes(envelope), signature),
    )

    def custody_not_reached(*args):
        pytest.fail("uszkodzone zachowane bajty dotarły do niedostępnego custody")

    monkeypatch.setattr(
        custody.LocalRootProofSigningProvider, "verify_historical", custody_not_reached
    )
    _denied(value, retention.RootProofRetentionCorrupt)


def test_local_signature_check_cannot_replace_pinned_custody_trust(prepared_boundary, monkeypatch):
    value = prepared_boundary
    _retain(value)

    def reject(*args):
        return False

    monkeypatch.setattr(custody.LocalRootProofSigningProvider, "verify_historical", reject)
    _denied(value, retention.RootProofRetentionCorrupt)


def test_local_crypto_infrastructure_failure_is_unavailable(prepared_boundary, monkeypatch):
    value = prepared_boundary
    _retain(value)

    def unsupported(*args):
        raise UnsupportedAlgorithm("testowa niedostępność Ed25519")

    monkeypatch.setattr(
        retention, "Ed25519PublicKey", SimpleNamespace(from_public_bytes=unsupported)
    )
    _denied(value, retention.RootProofRetentionUnavailable)


@pytest.mark.parametrize("code", [2, 3, 5, 32, 33, 999])
@pytest.mark.parametrize("wrapped", [True, False])
def test_native_windows_unavailability_requires_actual_error_and_ntfs_wrapper(
    prepared_boundary, monkeypatch, code, wrapped
):
    value = prepared_boundary
    _retain(value)

    class NativeWindowsError(Exception):
        pass

    monkeypatch.setitem(sys.modules, "pywintypes", SimpleNamespace(error=NativeWindowsError))
    error = NativeWindowsError(code, "CreateFile", "testowa blokada NTFS")
    if wrapped:
        error = _raised(
            custody.LocalSigningCustodyError("custody NTFS file qualification failed"), error
        )

    def busy(*args):
        raise error

    monkeypatch.setattr(custody.LocalRootProofSigningProvider, "verify_historical", busy)
    _denied(
        value,
        retention.RootProofRetentionUnavailable
        if wrapped and code in {2, 3, 5, 32, 33}
        else retention.RootProofRetentionCorrupt,
    )


def test_spoofed_native_windows_busy_attribute_remains_corrupt(prepared_boundary, monkeypatch):
    value = prepared_boundary
    _retain(value)
    error = RuntimeError("niejednoznaczna odmowa z winerror=32")
    error.winerror = 32

    def fail(*args):
        raise custody.LocalSigningCustodyError("custody NTFS file qualification failed") from error

    monkeypatch.setattr(custody.LocalRootProofSigningProvider, "verify_historical", fail)
    _denied(value, retention.RootProofRetentionCorrupt)

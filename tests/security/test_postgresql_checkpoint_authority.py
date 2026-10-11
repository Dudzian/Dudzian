"""Rzeczywisty PostgreSQL: trwałość checkpointu i granice dowodów V2."""

from __future__ import annotations

import json
import os
import select
import subprocess
import sys
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, replace

import psycopg
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from psycopg import sql
from psycopg.conninfo import make_conninfo
from psycopg.errors import InsufficientPrivilege
from psycopg.types.json import Jsonb

from bot_core import (
    postgresql_checkpoint_authority as implementation,
    postgresql_checkpoint_schema as checkpoint_schema,
    postgresql_issuer_history_schema as history_schema,
)
from bot_core.authenticated_issuer_history import (
    NO_PREDECESSOR,
    HistoricalRevokedSignatureVerificationUnavailable,
    HistoryContractError,
    HistoryEventIdentity,
    HistoryStreamIdentity,
    LocalCheckpoint,
    LocalCheckpointProvider,
    ReconciliationOutcome,
    ReferenceAuthenticatedHistory,
    VerifiedHistoryHead,
    attest_head,
    canonical_json_bytes,
)
from bot_core.local_signing_custody import SigningKeyLifecycle
from bot_core.postgresql_authenticated_issuer_history import PostgreSQLAuthenticatedIssuerHistory
from bot_core.postgresql_entitlement_registry import PostgreSQLConnectionConfig
from bot_core.root_proof_issuer_substrate import (
    CheckpointAuthorityProvider,
    CredentialRoleIdentity,
    CredentialSemanticRole,
    ProviderIdentity,
    ProviderQualificationPolicy,
    ProviderRole,
    SecurityProfile,
    SecurityProfileIdentity,
    public_key_material_identity,
)

pytestmark = pytest.mark.external_postgresql
BASE_DSN = os.environ.get(
    "CRYPTOHUNTER_TEST_POSTGRES_DSN",
    os.environ.get(
        "DUDZIAN_TEST_POSTGRES_DSN",
        "host=127.0.0.1 port=55432 dbname=postgres user=postgres connect_timeout=5",
    ),
)


class _Signer:
    """Custody testowe; żaden prywatny klucz nie trafia do PostgreSQL."""

    def __init__(self, trust_domain: str, public_key: bytes | None = None):
        self.key = None if public_key is not None else Ed25519PrivateKey.generate()
        self._public = public_key
        if self.key is not None:
            self._public = self.key.public_key().public_bytes(
                serialization.Encoding.Raw, serialization.PublicFormat.Raw
            )
        self.identity = ProviderIdentity(
            ProviderRole.HISTORY_ATTESTATION_SIGNING,
            SecurityProfileIdentity(SecurityProfile.PRODUCTION_LOCAL, trust_domain),
            "isolated-checkpoint-test-signing",
        )
        self.lifecycle = SigningKeyLifecycle.ACTIVE
        self.generation = 1
        self.role = CredentialSemanticRole.HISTORY_ATTESTATION_SIGNING
        self.namespace = self.identity.provider_namespace
        self.lifecycle_namespace = "isolated-checkpoint-lifecycle"

    def credential_identities(self):
        return (
            CredentialRoleIdentity(
                self.role,
                "isolated-checkpoint-credential",
                self.namespace,
                "1",
                self.lifecycle_namespace,
                public_key_material_identity(self._public),
            ),
        )

    def active_credential_identity(self):
        assert self.lifecycle is SigningKeyLifecycle.ACTIVE
        return self.credential_identities()[0]

    def sign_history_head(self, payload):
        assert self.key is not None and self.lifecycle is SigningKeyLifecycle.ACTIVE
        return self.key.sign(payload)

    def public_key(self, credential_identity):
        assert credential_identity == "isolated-checkpoint-credential"
        return self._public

    def lifecycle_generation(self):
        return self.generation

    def lifecycle_state(self):
        return self.lifecycle


def _connection(role: str, *, dsn: str = BASE_DSN, **options):
    return PostgreSQLConnectionConfig(make_conninfo(dsn, user=role, **options))


@dataclass(frozen=True)
class _Boundary:
    history_setup: history_schema.PostgreSQLIssuerHistoryProvisioning
    checkpoint_setup: checkpoint_schema.PostgreSQLCheckpointProvisioning
    stream: HistoryStreamIdentity
    checkpoint_id: str
    history: PostgreSQLAuthenticatedIssuerHistory
    signer: _Signer
    checkpoint: implementation.PostgreSQLCheckpointAuthority

    def restart(self, *, stream=None, checkpoint_id=None, role=None, signer=None, **options):
        return implementation.PostgreSQLCheckpointAuthority(
            _connection(role or self.checkpoint_setup.runtime_role, **options),
            schema=self.checkpoint_setup.schema,
            stream=stream or self.stream,
            checkpoint_id=checkpoint_id or self.checkpoint_id,
            history=self.history,
            verification_authorities=(signer or self.signer,),
        )


@contextmanager
def _isolated_boundary():
    suffix = uuid.uuid4().hex[:12]
    trust = "isolated-checkpoint-trust"
    history_setup = history_schema.PostgreSQLIssuerHistoryProvisioning(
        f"cph_{suffix}", f"cpho_{suffix}", f"cphr_{suffix}", f"cpha_{suffix}", trust
    )
    checkpoint_setup = checkpoint_schema.PostgreSQLCheckpointProvisioning(
        f"cpc_{suffix}", f"cpco_{suffix}", f"cpcr_{suffix}", f"cpca_{suffix}", trust
    )
    stream = HistoryStreamIdentity(
        f"checkpoint-stream-{suffix}",
        "IndependentAccountGenesisRootProofIssuer",
        "PRODUCTION_LOCAL",
        "isolated-environment",
        trust,
        "isolated-product",
        1,
    )
    checkpoint_id = f"checkpoint-{suffix}"
    installed = []
    try:
        history_schema.provision_postgresql_issuer_history(
            PostgreSQLConnectionConfig(BASE_DSN), history_setup
        )
        installed.append(history_setup)
        history_schema.provision_postgresql_issuer_history_stream(
            _connection(history_setup.admin_role), schema=history_setup.schema, stream=stream
        )
        checkpoint_schema.provision_postgresql_checkpoint(
            PostgreSQLConnectionConfig(BASE_DSN), checkpoint_setup
        )
        installed.append(checkpoint_setup)
        checkpoint_schema.provision_postgresql_checkpoint_stream(
            _connection(checkpoint_setup.admin_role),
            schema=checkpoint_setup.schema,
            checkpoint_id=checkpoint_id,
            stream=stream,
        )
        history = PostgreSQLAuthenticatedIssuerHistory(
            _connection(history_setup.runtime_role), schema=history_setup.schema, stream=stream
        )
        signer = _Signer(trust)
        checkpoint = implementation.PostgreSQLCheckpointAuthority(
            _connection(checkpoint_setup.runtime_role),
            schema=checkpoint_setup.schema,
            checkpoint_id=checkpoint_id,
            stream=stream,
            history=history,
            verification_authorities=(signer,),
        )
        yield _Boundary(
            history_setup, checkpoint_setup, stream, checkpoint_id, history, signer, checkpoint
        )
    finally:
        # Usuwamy wyłącznie schematy i role utworzone przez tę instancję testu.
        with psycopg.connect(BASE_DSN, autocommit=True) as conn:
            for setup in reversed(installed):
                conn.execute(
                    sql.SQL("DROP SCHEMA IF EXISTS {} CASCADE").format(sql.Identifier(setup.schema))
                )
                for role in (setup.runtime_role, setup.admin_role, setup.schema_owner_role):
                    conn.execute(sql.SQL("DROP ROLE IF EXISTS {}").format(sql.Identifier(role)))


@pytest.fixture
def boundary():
    with _isolated_boundary() as value:
        yield value


def _event(number):
    return HistoryEventIdentity(f"checkpoint-op-{number}", f"attempt-{number}", f"proof-{number}")


def _append_head(boundary, number, *, retain=True):
    previous = boundary.history.current_record()
    record, replayed = boundary.history.append(
        expected_digest=NO_PREDECESSOR if previous is None else previous.authenticated_digest,
        event_identity=_event(number),
        payload={"fixture": "checkpoint-no-issuance", "number": number, "unicode": "ą"},
    )
    assert not replayed
    head = attest_head(record, boundary.signer)
    if retain:
        assert boundary.history.retain_attested_head(head, boundary.signer) == (head, False)
    return head


def _accept_heads(boundary, count=3):
    heads = []
    for number in range(1, count + 1):
        head = _append_head(boundary, number)
        result, replayed = boundary.checkpoint.advance(expected_revision=number - 1, head=head)
        assert result.checkpoint_revision == number and not replayed
        heads.append(head)
    return heads


def _configuration(boundary, **changes):
    result = {
        "history_dsn": _connection(boundary.history_setup.runtime_role).dsn,
        "history_schema": boundary.history_setup.schema,
        "dsn": _connection(boundary.checkpoint_setup.runtime_role).dsn,
        "schema": boundary.checkpoint_setup.schema,
        "stream": boundary.stream.material(),
        "checkpoint_id": boundary.checkpoint_id,
        "public_key": boundary.signer.public_key("isolated-checkpoint-credential").hex(),
    }
    result.update(changes)
    return result


_WORKER = """
import json
import sys
from bot_core import postgresql_checkpoint_authority as implementation
from bot_core.authenticated_issuer_history import HistoryContractError, HistoryStreamIdentity
from bot_core.postgresql_authenticated_issuer_history import PostgreSQLAuthenticatedIssuerHistory
from bot_core.postgresql_entitlement_registry import PostgreSQLConnectionConfig
from tests.security.test_postgresql_checkpoint_authority import _Signer
configuration = json.loads(sys.argv[1])
stream = HistoryStreamIdentity(**configuration['stream'])
signer = _Signer(stream.trust_domain, bytes.fromhex(configuration['public_key']))
history = PostgreSQLAuthenticatedIssuerHistory(
    PostgreSQLConnectionConfig(configuration['history_dsn']),
    schema=configuration['history_schema'], stream=stream)
checkpoint = implementation.PostgreSQLCheckpointAuthority(
    PostgreSQLConnectionConfig(configuration['dsn']), schema=configuration['schema'],
    checkpoint_id=configuration['checkpoint_id'], stream=stream, history=history,
    verification_authorities=(signer,))
"""


def _process(source, configuration):
    return subprocess.Popen(
        [sys.executable, "-c", _WORKER + source, json.dumps(configuration)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def _finish(process):
    output, error = process.communicate(timeout=30)
    assert process.returncode == 0, error
    return json.loads(output)


def _kill(process):
    if process.poll() is None:
        process.kill()
    process.communicate(timeout=10)


def _expect_line(process, expected):
    assert process.stdout is not None
    ready, _, _ = select.select([process.stdout], [], [], 30)
    assert ready, f"Proces nie osiągnął kontrolowanego punktu {expected!r}"
    line = process.stdout.readline().strip()
    if line != expected:
        _, error = process.communicate(timeout=10)
        pytest.fail(f"Oczekiwano {expected!r}, otrzymano {line!r}; {error}")


def test_provider_reports_durable_local_boundary_without_issuance_authority(boundary):
    checkpoint = boundary.checkpoint
    assert isinstance(checkpoint, CheckpointAuthorityProvider)
    assert checkpoint.identity.role is ProviderRole.CHECKPOINT_AUTHORITY
    assert checkpoint.identity.security == SecurityProfileIdentity(
        SecurityProfile.PRODUCTION_LOCAL, boundary.stream.trust_domain
    )
    assert (
        ProviderQualificationPolicy().failures_for(
            SecurityProfile.PRODUCTION_LOCAL, checkpoint.identity, checkpoint.capabilities
        )
        == ()
    )
    assert "user=" not in repr(checkpoint)
    for operation in ("bind", "issue", "sign_history_head", "provision_credential"):
        assert not hasattr(checkpoint, operation)
    assert checkpoint.current_checkpoint() is None
    assert checkpoint.retained_acceptances() == checkpoint.retained_attested_heads() == ()


def test_checkpoint_identity_revision_and_entire_evidence_chain_survive_process_restart(boundary):
    heads = _accept_heads(boundary)
    evidence = boundary.checkpoint.retained_acceptances()
    assert [item.checkpoint_revision for item in evidence] == [1, 2, 3]
    assert evidence[0].predecessor_acceptance_digest == NO_PREDECESSOR
    assert [item.predecessor_acceptance_digest for item in evidence[1:]] == [
        item.acceptance_digest for item in evidence[:-1]
    ]
    assert len({item.checkpoint_authority_identity for item in evidence}) == 1
    source = """
current = checkpoint.current_checkpoint()
evidence = checkpoint.retained_acceptances()
heads = checkpoint.retained_attested_heads()
print(json.dumps({'revision': current.checkpoint_revision,
    'sequence': current.history_sequence, 'digest': current.authenticated_head_digest,
    'authority': [item.checkpoint_authority_identity for item in evidence],
    'acceptance_digests': [item.acceptance_digest for item in evidence],
    'signatures': [head.signature.hex() for head in heads]}), flush=True)
"""
    process = _process(source, _configuration(boundary))
    try:
        recovered = _finish(process)
    finally:
        _kill(process)
    assert recovered == {
        "revision": 3,
        "sequence": 3,
        "digest": heads[-1].record_digest,
        "authority": [item.checkpoint_authority_identity for item in evidence],
        "acceptance_digests": [item.acceptance_digest for item in evidence],
        "signatures": [head.signature.hex() for head in heads],
    }
    restarted = boundary.restart()
    assert restarted.current_checkpoint() == boundary.checkpoint.current_checkpoint()
    assert restarted.retained_acceptances() == evidence
    assert restarted.retained_attested_heads() == tuple(heads)
    assert restarted.authenticated_checkpoint_at(1).history_sequence == 1
    assert restarted.authenticated_checkpoint_at(4) is None


def test_cas_replay_and_resolver_follow_reference_oracle(boundary):
    oracle_history = ReferenceAuthenticatedHistory(boundary.stream)
    oracle = LocalCheckpointProvider(boundary.checkpoint_id, boundary.stream)
    head = _append_head(boundary, 1)
    oracle_history.append(
        expected_digest=NO_PREDECESSOR,
        event_identity=_event(1),
        payload={"fixture": "checkpoint-no-issuance", "number": 1, "unicode": "ą"},
    )
    for expected in (0, 0, 1, 2):
        arguments = {"expected_revision": expected, "head": head}
        try:
            result = oracle.advance(
                **arguments, history=oracle_history, verification_authority=boundary.signer
            )
        except HistoryContractError:
            with pytest.raises(HistoryContractError):
                boundary.checkpoint.advance(**arguments)
        else:
            assert boundary.checkpoint.advance(**arguments) == result
        assert boundary.checkpoint.current_checkpoint() == oracle.current_checkpoint()
    committed = boundary.checkpoint.current_checkpoint()
    assert boundary.checkpoint.resolve_advance(expected_revision=0, head=head) == (committed, True)
    assert boundary.checkpoint.retry_advance(expected_revision=0, head=head) == (committed, True)
    assert len(boundary.checkpoint.retained_acceptances()) == 1


def test_uncommitted_resolver_returns_absence_and_safe_retry_commits_once(boundary):
    head = _append_head(boundary, 1)
    assert boundary.checkpoint.resolve_advance(expected_revision=0, head=head) is None
    current, replayed = boundary.checkpoint.retry_advance(expected_revision=0, head=head)
    assert not replayed and current.checkpoint_revision == 1
    assert boundary.restart().retry_advance(expected_revision=0, head=head) == (current, True)


def test_independent_processes_have_one_strict_cas_winner(boundary):
    head = _append_head(boundary, 1)
    source = """
head = history.retained_attested_head(verification_authority=signer)
print('ready', flush=True)
sys.stdin.readline()
try:
    result, replayed = checkpoint.advance(expected_revision=0, head=head)
    print(json.dumps({'outcome': 'committed', 'revision': result.checkpoint_revision,
        'replayed': replayed}), flush=True)
except HistoryContractError:
    print(json.dumps({'outcome': 'conflict'}), flush=True)
"""
    processes = []
    try:
        processes = [_process(source, _configuration(boundary)) for _ in range(2)]
        for process in processes:
            _expect_line(process, "ready")
        for process in processes:
            assert process.stdin is not None
            process.stdin.write("go\n")
            process.stdin.flush()
        results = [_finish(process) for process in processes]
    finally:
        for process in processes:
            _kill(process)
    assert sorted(item["outcome"] for item in results) == ["committed", "conflict"]
    assert next(item for item in results if item["outcome"] == "committed") == {
        "outcome": "committed",
        "revision": 1,
        "replayed": False,
    }
    assert boundary.checkpoint.current_checkpoint().authenticated_head_digest == head.record_digest
    assert len(boundary.checkpoint.retained_acceptances()) == 1


@pytest.mark.parametrize("cut,committed", [("before_commit", False), ("after_commit", True)])
def test_process_death_preserves_atomic_checkpoint_and_acceptance(boundary, cut, committed):
    head = _append_head(boundary, 1)
    source = """
def pause(name):
    if name == configuration['cut']:
        print('cut', flush=True)
        sys.stdin.readline()
implementation._persistence_cut = pause
head = history.retained_attested_head(verification_authority=signer)
checkpoint.advance(expected_revision=0, head=head)
raise AssertionError('Proces nadrzędny powinien przerwać worker w punkcie transakcji')
"""
    process = _process(source, _configuration(boundary, cut=cut))
    try:
        _expect_line(process, "cut")
    finally:
        _kill(process)
    restarted = boundary.restart()
    assert len(restarted.retained_acceptances()) == int(committed)
    assert (restarted.current_checkpoint() is not None) is committed
    recovered, replayed = restarted.retry_advance(expected_revision=0, head=head)
    assert replayed is committed and recovered.checkpoint_revision == 1
    assert restarted.retained_attested_heads() == (head,)


def test_lost_response_after_commit_is_resolved_without_second_acceptance(boundary, monkeypatch):
    head = _append_head(boundary, 1)

    def lose_response(name):
        if name == "after_commit":
            raise psycopg.OperationalError("Kontrolowana utrata odpowiedzi po commit")

    monkeypatch.setattr(implementation, "_persistence_cut", lose_response)
    with pytest.raises(HistoryContractError):
        boundary.checkpoint.advance(expected_revision=0, head=head)
    monkeypatch.undo()
    restarted = boundary.restart()
    current = restarted.current_checkpoint()
    assert current is not None and current.checkpoint_revision == 1
    assert restarted.resolve_advance(expected_revision=0, head=head) == (current, True)
    assert restarted.retry_advance(expected_revision=0, head=head) == (current, True)
    assert len(restarted.retained_acceptances()) == 1


def test_database_unavailable_is_not_empty_or_uncommitted_checkpoint(boundary, monkeypatch):
    head = _append_head(boundary, 1)
    boundary.checkpoint.advance(expected_revision=0, head=head)

    def unavailable(self):
        raise psycopg.OperationalError("Kontrolowana niedostępność bazy")

    monkeypatch.setattr(implementation.PostgreSQLCheckpointAuthority, "_connect", unavailable)
    for operation in (
        boundary.checkpoint.current_checkpoint,
        lambda: boundary.checkpoint.historical_acceptance(head),
        lambda: boundary.checkpoint.resolve_advance(expected_revision=0, head=head),
        lambda: boundary.checkpoint.retry_advance(expected_revision=0, head=head),
    ):
        with pytest.raises(HistoryContractError):
            operation()


@pytest.mark.parametrize(
    "column,value",
    [
        ("revision", 99),
        ("checkpoint", b"{}"),
        ("acceptance", b"{}"),
        ("head", b"{}"),
        ("signature", b"\x00" * 64),
    ],
)
def test_middle_acceptance_corruption_blocks_entire_authority(boundary, column, value):
    heads = _accept_heads(boundary)
    with psycopg.connect(BASE_DSN, autocommit=True) as conn:
        conn.execute(
            sql.SQL("UPDATE {}.entries SET {}=%s WHERE revision=2").format(
                sql.Identifier(boundary.checkpoint_setup.schema), sql.Identifier(column)
            ),
            (value,),
        )
    for operation in (
        boundary.checkpoint.current_checkpoint,
        boundary.checkpoint.retained_acceptances,
        boundary.checkpoint.retained_attested_heads,
        lambda: boundary.checkpoint.historical_acceptance(heads[0]),
        lambda: boundary.checkpoint.resolve_advance(expected_revision=0, head=heads[0]),
        lambda: boundary.checkpoint.advance(expected_revision=3, head=heads[-1]),
    ):
        with pytest.raises(HistoryContractError):
            operation()
    with pytest.raises(HistoryContractError):
        boundary.restart()


@pytest.mark.parametrize("mutation", ["missing_middle", "checkpoint_rewind", "identity"])
def test_acceptance_rollback_or_authority_substitution_is_detected(boundary, mutation):
    _accept_heads(boundary)
    schema = sql.Identifier(boundary.checkpoint_setup.schema)
    with psycopg.connect(BASE_DSN, autocommit=True) as conn:
        statements = {
            "missing_middle": sql.SQL("DELETE FROM {}.entries WHERE revision=2").format(schema),
            "checkpoint_rewind": sql.SQL("UPDATE {}.checkpoints SET revision=1").format(schema),
            "identity": sql.SQL(
                "UPDATE {}.checkpoints SET authority_identity='substituted-authority'"
            ).format(schema),
        }
        conn.execute(statements[mutation])
    with pytest.raises(HistoryContractError):
        boundary.checkpoint.current_checkpoint()
    with pytest.raises(HistoryContractError):
        boundary.restart()


def test_history_valid_prefix_rollback_cannot_revalidate_newer_checkpoint(boundary):
    heads = _accept_heads(boundary)
    schema = sql.Identifier(boundary.history_setup.schema)
    with psycopg.connect(BASE_DSN, autocommit=True) as conn:
        conn.execute(sql.SQL("DELETE FROM {}.attestations WHERE sequence>1").format(schema))
        conn.execute(sql.SQL("DELETE FROM {}.records WHERE sequence>1").format(schema))
        conn.execute(
            sql.SQL("UPDATE {}.streams SET head_sequence=1,head_digest=%s").format(schema),
            (heads[0].record_digest,),
        )
    boundary.history.verify()
    with pytest.raises(HistoryContractError):
        boundary.checkpoint.current_checkpoint()
    with pytest.raises(HistoryContractError):
        boundary.restart()


def test_missing_acceptance_rejects_historical_signature_after_revocation(boundary):
    head = _append_head(boundary, 1)
    boundary.signer.lifecycle = SigningKeyLifecycle.REVOKED
    boundary.signer.generation = 2
    with pytest.raises(HistoricalRevokedSignatureVerificationUnavailable):
        boundary.checkpoint.verify_historical_acceptance(head, current_lifecycle_generation=2)
    with pytest.raises(HistoryContractError):
        boundary.checkpoint.advance(expected_revision=0, head=head)
    assert boundary.checkpoint.current_checkpoint() is None


def test_active_acceptance_remains_verifiable_after_later_revocation_and_restart(boundary):
    head = _append_head(boundary, 1)
    committed, _ = boundary.checkpoint.advance(expected_revision=0, head=head)
    evidence = boundary.checkpoint.historical_acceptance(head)
    assert evidence.lifecycle_state_at_acceptance is SigningKeyLifecycle.ACTIVE
    assert evidence.lifecycle_generation_at_acceptance == 1
    boundary.signer.lifecycle = SigningKeyLifecycle.REVOKED
    boundary.signer.generation = 2
    restarted = boundary.restart()
    assert restarted.current_checkpoint() == committed
    assert restarted.historical_acceptance(head) == evidence
    assert restarted.verify_historical_acceptance(head, current_lifecycle_generation=2) is None
    assert (
        boundary.history.verify_attested_historical_head(head, boundary.signer, restarted).head
        == head
    )
    assert restarted.resolve_advance(expected_revision=0, head=head) == (committed, True)
    with pytest.raises(HistoryContractError):
        restarted.advance(expected_revision=1, head=head)
    with pytest.raises(HistoryContractError):
        restarted.verify_historical_acceptance(head, current_lifecycle_generation=1)


def test_changed_signature_cannot_reuse_original_accepted_operation(boundary):
    head = _append_head(boundary, 1)
    boundary.checkpoint.advance(expected_revision=0, head=head)
    altered = replace(head, signature=b"\x00" * 64)
    for operation in (
        lambda: boundary.checkpoint.resolve_advance(expected_revision=0, head=altered),
        lambda: boundary.checkpoint.retry_advance(expected_revision=0, head=altered),
    ):
        with pytest.raises(HistoryContractError):
            operation()


@pytest.mark.parametrize(
    "field,value",
    [
        ("stream_id", "other-stream"),
        ("issuer_authority_identity", "other-issuer"),
        ("security_profile", "REFERENCE"),
        ("environment", "other-environment"),
        ("trust_domain", "other-trust"),
        ("product_scope", "other-product"),
        ("security_epoch", 2),
    ],
)
def test_full_stream_identity_is_bound_to_history_and_checkpoint(boundary, field, value):
    head = _append_head(boundary, 1)
    other_stream = replace(boundary.stream, **{field: value})
    forged = replace(head, stream=other_stream)
    with pytest.raises(HistoryContractError):
        boundary.checkpoint.advance(expected_revision=0, head=forged)
    with pytest.raises(HistoryContractError):
        boundary.restart(stream=other_stream)
    assert boundary.checkpoint.current_checkpoint() is None


@pytest.mark.parametrize(
    "mutation", ["credential_role", "provider_role", "namespace", "lifecycle_namespace", "key"]
)
def test_selected_signer_binding_cannot_be_substituted(boundary, mutation):
    head = _append_head(boundary, 1)
    if mutation == "credential_role":
        boundary.signer.role = CredentialSemanticRole.ROOT_PROOF_ISSUER_SIGNING
    elif mutation == "provider_role":
        boundary.signer.identity = replace(
            boundary.signer.identity, role=ProviderRole.ROOT_PROOF_SIGNING
        )
    elif mutation == "namespace":
        boundary.signer.namespace = "substituted-signing-namespace"
    elif mutation == "lifecycle_namespace":
        boundary.signer.lifecycle_namespace = "substituted-lifecycle-namespace"
    else:
        boundary.signer._public = (
            Ed25519PrivateKey.generate()
            .public_key()
            .public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
        )
    with pytest.raises(HistoryContractError):
        boundary.checkpoint.advance(expected_revision=0, head=head)
    assert boundary.checkpoint.current_checkpoint() is None


def test_caller_signer_objects_and_reporting_dtos_do_not_grant_authority(boundary):
    head = _append_head(boundary, 1)
    duplicate = _Signer(boundary.stream.trust_domain, boundary.signer._public)
    with pytest.raises(HistoryContractError):
        boundary.checkpoint.advance(
            expected_revision=0, head=head, verification_authority=duplicate
        )
    with pytest.raises(TypeError):
        boundary.checkpoint.advance(expected_revision=0, head=VerifiedHistoryHead(head))
    with pytest.raises(TypeError):
        boundary.checkpoint.advance_exact_successor(None, VerifiedHistoryHead(head))
    with pytest.raises(TypeError):
        boundary.checkpoint.verify_historical_acceptance(
            VerifiedHistoryHead(head), current_lifecycle_generation=2
        )
    assert boundary.checkpoint.current_checkpoint() is None


def test_facade_reverifies_retained_history_instead_of_trusting_fabricated_checkpoint(boundary):
    head = _append_head(boundary, 1, retain=False)
    successor = LocalCheckpoint(
        boundary.checkpoint_id,
        boundary.stream,
        head.sequence,
        head.record_digest,
        head.signing_credential_identity,
        1,
    )
    with pytest.raises(HistoryContractError):
        boundary.checkpoint.advance_exact_successor(None, successor)
    assert boundary.checkpoint.current_checkpoint() is None
    boundary.history.retain_attested_head(head, boundary.signer)
    assert boundary.checkpoint.advance_exact_successor(None, successor) is True
    assert boundary.checkpoint.current_checkpoint() == successor
    forged = replace(successor, authenticated_head_digest="sha256:" + "0" * 64)
    with pytest.raises(HistoryContractError):
        boundary.checkpoint.advance_exact_successor(successor, forged)


def test_detached_reports_cannot_mutate_the_retained_authority(boundary):
    head = _append_head(boundary, 1)
    original, _ = boundary.checkpoint.advance(expected_revision=0, head=head)
    report = boundary.checkpoint.historical_acceptance(head)
    returned_head = boundary.checkpoint.retained_attested_heads()[0]
    returned_checkpoint = boundary.checkpoint.current_checkpoint()
    object.__setattr__(report, "lifecycle_generation_at_acceptance", 99)
    object.__setattr__(returned_head, "signature", b"\x00" * 64)
    object.__setattr__(returned_checkpoint, "checkpoint_revision", 0)
    assert boundary.checkpoint.current_checkpoint() == original
    assert boundary.checkpoint.retained_attested_heads() == (head,)
    assert boundary.checkpoint.historical_acceptance(head).lifecycle_generation_at_acceptance == 1


def test_runtime_admin_and_owner_have_separate_reviewed_ports(boundary):
    setup = boundary.checkpoint_setup
    schema = sql.Identifier(setup.schema)
    with psycopg.connect(_connection(setup.runtime_role).dsn, autocommit=True) as conn:
        for table in ("metadata", "checkpoints", "entries"):
            if table != "metadata":
                with pytest.raises(InsufficientPrivilege):
                    conn.execute(
                        sql.SQL("SELECT * FROM {}.{}").format(schema, sql.Identifier(table))
                    )
            with pytest.raises(InsufficientPrivilege):
                conn.execute(sql.SQL("DELETE FROM {}.{}").format(schema, sql.Identifier(table)))
        with pytest.raises(InsufficientPrivilege):
            conn.execute(
                sql.SQL("SELECT {}.provision_stream(%s,%s)").format(schema),
                (
                    checkpoint_schema._scalar(boundary.checkpoint_id),
                    canonical_json_bytes(boundary.stream.material()),
                ),
            )
        with pytest.raises(InsufficientPrivilege):
            conn.execute(sql.SQL("SET ROLE {}").format(sql.Identifier(setup.schema_owner_role)))
    for role in (setup.admin_role, "postgres"):
        with pytest.raises(HistoryContractError):
            boundary.restart(role=role)
    with psycopg.connect(BASE_DSN) as conn:
        assert conn.execute(
            "SELECT rolcanlogin FROM pg_roles WHERE rolname=%s", (setup.schema_owner_role,)
        ).fetchone() == (False,)


@pytest.mark.parametrize(
    "mutation",
    [
        "public_execute",
        "table_grant",
        "column_grant",
        "membership",
        "role_super",
        "unlogged",
        "row_security",
        "schema_create",
    ],
)
def test_live_acl_role_schema_or_durability_drift_fails_closed(boundary, mutation):
    setup = boundary.checkpoint_setup
    schema, runtime, admin = map(
        sql.Identifier, (setup.schema, setup.runtime_role, setup.admin_role)
    )
    with psycopg.connect(BASE_DSN, autocommit=True) as conn:
        statements = {
            "public_execute": sql.SQL(
                "GRANT EXECUTE ON FUNCTION {}.read_checkpoint(bytea,bytea,boolean) TO PUBLIC"
            ).format(schema),
            "table_grant": sql.SQL("GRANT SELECT ON {}.entries TO {}").format(schema, runtime),
            "column_grant": sql.SQL("GRANT SELECT (schema_version) ON {}.metadata TO {}").format(
                schema, runtime
            ),
            "membership": sql.SQL("GRANT {} TO {}").format(admin, runtime),
            "role_super": sql.SQL("ALTER ROLE {} SUPERUSER").format(runtime),
            "unlogged": sql.SQL("ALTER TABLE {}.entries SET UNLOGGED").format(schema),
            "row_security": sql.SQL("ALTER TABLE {}.entries ENABLE ROW LEVEL SECURITY").format(
                schema
            ),
            "schema_create": sql.SQL("GRANT CREATE ON SCHEMA {} TO {}").format(schema, runtime),
        }
        conn.execute(statements[mutation])
    try:
        with pytest.raises(HistoryContractError):
            boundary.checkpoint.current_checkpoint()
        with pytest.raises(HistoryContractError):
            boundary.restart()
    finally:
        if mutation == "membership":
            with psycopg.connect(BASE_DSN, autocommit=True) as conn:
                conn.execute(sql.SQL("REVOKE {} FROM {}").format(admin, runtime))


def test_changed_sql_source_and_matching_manifest_do_not_qualify_installation(boundary):
    setup = boundary.checkpoint_setup
    signature = "verify_checkpoint(bytea,bytea)"
    replacement = "BEGIN RETURN; END"
    with psycopg.connect(BASE_DSN, autocommit=True) as conn:
        conn.execute(
            sql.SQL(
                "CREATE OR REPLACE FUNCTION {}.verify_checkpoint(p_id bytea,p_stream bytea) "
                "RETURNS void LANGUAGE plpgsql SECURITY DEFINER "
                "SET search_path=pg_catalog AS {}"
            ).format(sql.Identifier(setup.schema), sql.Literal(replacement))
        )
        manifest = conn.execute(
            sql.SQL("SELECT function_definition_sha256 FROM {}.metadata").format(
                sql.Identifier(setup.schema)
            )
        ).fetchone()[0]
        manifest[signature] = checkpoint_schema._source_fingerprint(signature, replacement)
        conn.execute(
            sql.SQL("UPDATE {}.metadata SET function_definition_sha256=%s").format(
                sql.Identifier(setup.schema)
            ),
            (Jsonb(manifest),),
        )
    with pytest.raises(HistoryContractError):
        boundary.checkpoint.current_checkpoint()
    with pytest.raises(HistoryContractError):
        boundary.restart()


def test_synchronous_commit_off_does_not_qualify_checkpoint_authority(boundary):
    with pytest.raises(HistoryContractError):
        boundary.restart(options="-c synchronous_commit=off")


def test_fsync_off_prevents_installation_on_separate_real_postgresql():
    unsafe_dsn = os.environ.get("CRYPTOHUNTER_TEST_POSTGRES_UNSAFE_DSN")
    if unsafe_dsn is None:
        pytest.skip("Brak osobnego rzeczywistego klastra z fsync=off; nie zmieniamy wspólnej bazy")
    with psycopg.connect(unsafe_dsn) as conn:
        assert conn.execute("SELECT pg_catalog.current_setting('fsync')").fetchone() == ("off",)
    suffix = uuid.uuid4().hex[:12]
    setup = checkpoint_schema.PostgreSQLCheckpointProvisioning(
        f"cpu_{suffix}", f"cpuo_{suffix}", f"cpur_{suffix}", f"cpua_{suffix}", "unsafe-test"
    )
    with pytest.raises(checkpoint_schema.CheckpointSchemaQualificationError, match="durability"):
        checkpoint_schema.provision_postgresql_checkpoint(
            PostgreSQLConnectionConfig(unsafe_dsn), setup
        )
    with psycopg.connect(unsafe_dsn) as conn:
        assert (
            conn.execute("SELECT 1 FROM pg_namespace WHERE nspname=%s", (setup.schema,)).fetchone()
            is None
        )
        assert (
            conn.execute(
                "SELECT 1 FROM pg_roles WHERE rolname=ANY(%s)",
                ([setup.schema_owner_role, setup.runtime_role, setup.admin_role],),
            ).fetchall()
            == []
        )


def test_unstable_lifecycle_sampling_cannot_commit_acceptance(boundary, monkeypatch):
    head = _append_head(boundary, 1)
    generation = 0

    def changing_generation():
        nonlocal generation
        generation += 1
        return generation

    monkeypatch.setattr(boundary.signer, "lifecycle_generation", changing_generation)
    with pytest.raises(HistoryContractError):
        boundary.checkpoint.advance(expected_revision=0, head=head)
    assert boundary.checkpoint.current_checkpoint() is None
    assert boundary.checkpoint.retained_acceptances() == ()


@pytest.mark.parametrize(
    "lifecycle", [SigningKeyLifecycle.VERIFY_ONLY, SigningKeyLifecycle.REVOKED]
)
def test_non_active_credential_cannot_create_new_acceptance(boundary, lifecycle):
    head = _append_head(boundary, 1)
    boundary.signer.lifecycle = lifecycle
    boundary.signer.generation = 2
    with pytest.raises(HistoryContractError):
        boundary.checkpoint.advance(expected_revision=0, head=head)
    assert boundary.checkpoint.current_checkpoint() is None


def test_corrupted_middle_history_record_invalidates_old_and_current_checkpoints(boundary):
    heads = _accept_heads(boundary)
    with psycopg.connect(BASE_DSN, autocommit=True) as conn:
        conn.execute(
            sql.SQL("UPDATE {}.records SET canonical_record=%s WHERE sequence=2").format(
                sql.Identifier(boundary.history_setup.schema)
            ),
            (b"{}",),
        )
    for operation in (
        boundary.checkpoint.current_checkpoint,
        lambda: boundary.checkpoint.authenticated_checkpoint_at(1),
        lambda: boundary.checkpoint.resolve_advance(expected_revision=0, head=heads[0]),
    ):
        with pytest.raises(HistoryContractError):
            operation()


def test_observed_checkpoint_prefix_rollback_fails_but_restart_has_no_independent_witness(boundary):
    heads = _accept_heads(boundary)
    schema = sql.Identifier(boundary.checkpoint_setup.schema)
    with psycopg.connect(BASE_DSN, autocommit=True) as conn:
        conn.execute(sql.SQL("DELETE FROM {}.entries WHERE revision>1").format(schema))
        conn.execute(sql.SQL("UPDATE {}.checkpoints SET revision=1").format(schema))
    with pytest.raises(HistoryContractError):
        boundary.checkpoint.current_checkpoint()
    # Restart nie zna rewizji usuniętych razem z całą domeną checkpointu.
    # To rzeczywista granica BLOCKED dla niezależnego anti-rollback, nie promocja.
    restarted = boundary.restart()
    assert restarted.current_checkpoint().checkpoint_revision == 1
    assert restarted.current_checkpoint().authenticated_head_digest == heads[0].record_digest
    assert restarted.capabilities.checkpoint.independent_rollback_domain is False
    assert restarted.capabilities.checkpoint.independent_admin_or_security_domain is False


def test_different_trusted_signer_at_same_sequence_is_split_brain(boundary):
    head = _append_head(boundary, 1)
    original, _ = boundary.checkpoint.advance(expected_revision=0, head=head)
    other_signer = _Signer(boundary.stream.trust_domain)
    other_head = attest_head(boundary.history.current_record(), other_signer)
    checkpoint = implementation.PostgreSQLCheckpointAuthority(
        _connection(boundary.checkpoint_setup.runtime_role),
        schema=boundary.checkpoint_setup.schema,
        checkpoint_id=boundary.checkpoint_id,
        stream=boundary.stream,
        history=boundary.history,
        verification_authorities=(boundary.signer, other_signer),
    )
    with pytest.raises(HistoryContractError, match="split brain"):
        checkpoint.advance(expected_revision=1, head=other_head)
    with pytest.raises(HistoryContractError):
        checkpoint.resolve_advance(expected_revision=0, head=other_head)
    assert checkpoint.current_checkpoint() == original


def test_rotation_and_later_checkpoint_preserve_revoked_predecessor_signature(boundary):
    first_head = _append_head(boundary, 1)
    boundary.checkpoint.advance(expected_revision=0, head=first_head)
    successor_signer = _Signer(boundary.stream.trust_domain)
    successor_signer.namespace = "isolated-checkpoint-successor-signing"
    successor_signer.lifecycle_namespace = "isolated-checkpoint-successor-lifecycle"
    successor_signer.identity = replace(
        successor_signer.identity, provider_namespace=successor_signer.namespace
    )

    def restart_with_rotation():
        return implementation.PostgreSQLCheckpointAuthority(
            _connection(boundary.checkpoint_setup.runtime_role),
            schema=boundary.checkpoint_setup.schema,
            checkpoint_id=boundary.checkpoint_id,
            stream=boundary.stream,
            history=boundary.history,
            verification_authorities=(boundary.signer, successor_signer),
        )

    checkpoint = restart_with_rotation()
    successor_boundary = replace(boundary, signer=successor_signer, checkpoint=checkpoint)
    second_head = _append_head(successor_boundary, 2)
    second, _ = checkpoint.advance(expected_revision=1, head=second_head)
    original_acceptance = checkpoint.historical_acceptance(first_head)
    boundary.signer.lifecycle = SigningKeyLifecycle.REVOKED
    boundary.signer.generation = 2
    restarted = restart_with_rotation()
    assert restarted.current_checkpoint() == second
    assert restarted.retained_attested_heads() == (first_head, second_head)
    assert restarted.historical_acceptance(first_head) == original_acceptance
    assert (
        boundary.history.verify_attested_historical_head(
            first_head, boundary.signer, restarted
        ).head
        == first_head
    )
    assert (
        restarted.verify_historical_acceptance(first_head, current_lifecycle_generation=2) is None
    )
    evidence = restarted.retained_acceptances()
    assert evidence[1].predecessor_acceptance_digest == evidence[0].acceptance_digest


def test_resolver_does_not_reinterpret_noop_replay_as_later_committed_operation(boundary):
    first_head = _append_head(boundary, 1)
    first, _ = boundary.checkpoint.advance(expected_revision=0, head=first_head)
    assert boundary.checkpoint.advance(expected_revision=1, head=first_head) == (first, True)
    second_head = _append_head(boundary, 2)
    second, _ = boundary.checkpoint.advance(expected_revision=1, head=second_head)
    with pytest.raises(HistoryContractError):
        boundary.checkpoint.resolve_advance(expected_revision=1, head=first_head)
    assert boundary.checkpoint.resolve_advance(expected_revision=0, head=first_head) == (
        first,
        True,
    )
    assert boundary.checkpoint.current_checkpoint() == second


def test_intrinsic_login_guards_reject_wrong_principals_even_after_acl_widening(boundary):
    setup = boundary.checkpoint_setup
    schema = sql.Identifier(setup.schema)
    with psycopg.connect(BASE_DSN, autocommit=True) as conn:
        conn.execute(
            sql.SQL(
                "GRANT EXECUTE ON FUNCTION {}.read_checkpoint(bytea,bytea,boolean) TO {}"
            ).format(schema, sql.Identifier(setup.admin_role))
        )
        conn.execute(
            sql.SQL("GRANT EXECUTE ON FUNCTION {}.provision_stream(bytea,bytea) TO {}").format(
                schema, sql.Identifier(setup.runtime_role)
            )
        )
    arguments = (
        checkpoint_schema._scalar(boundary.checkpoint_id),
        canonical_json_bytes(boundary.stream.material()),
    )
    for role, query in (
        (setup.admin_role, "SELECT {}.read_checkpoint(%s,%s,false)"),
        (setup.runtime_role, "SELECT {}.provision_stream(%s,%s)"),
    ):
        with psycopg.connect(_connection(role).dsn, autocommit=True) as conn:
            with pytest.raises(InsufficientPrivilege, match="wrong checkpoint principal"):
                conn.execute(sql.SQL(query).format(schema), arguments)


def test_coordinated_constraint_and_manifest_tamper_cannot_qualify_schema(boundary):
    setup = boundary.checkpoint_setup
    schema = sql.Identifier(setup.schema)
    with psycopg.connect(BASE_DSN, autocommit=True) as conn:
        constraint = conn.execute(
            "SELECT co.conname FROM pg_catalog.pg_constraint co JOIN pg_catalog.pg_class c "
            "ON c.oid=co.conrelid JOIN pg_catalog.pg_namespace n ON n.oid=c.relnamespace "
            "WHERE n.nspname=%s AND c.relname='entries' AND co.contype='c' "
            "AND pg_get_constraintdef(co.oid) LIKE '%%octet_length%%'",
            (setup.schema,),
        ).fetchone()
        assert constraint is not None
        conn.execute(
            sql.SQL("ALTER TABLE {}.entries DROP CONSTRAINT {}").format(
                schema, sql.Identifier(constraint[0])
            )
        )
        fingerprint = checkpoint_schema._physical_fingerprint(conn, setup.schema)
        conn.execute(
            sql.SQL("UPDATE {}.metadata SET physical_definition_sha256=%s").format(schema),
            (fingerprint,),
        )
    with pytest.raises(HistoryContractError):
        boundary.checkpoint.current_checkpoint()
    with pytest.raises(HistoryContractError):
        boundary.restart()


def test_unchanged_sql_body_with_substituted_output_type_does_not_qualify(boundary):
    setup = boundary.checkpoint_setup
    schema = sql.Identifier(setup.schema)
    signature = "canonical_string(bytea,integer)"
    with psycopg.connect(BASE_DSN, autocommit=True) as conn:
        original_source = conn.execute(
            "SELECT p.prosrc FROM pg_catalog.pg_proc p JOIN pg_catalog.pg_namespace n "
            "ON n.oid=p.pronamespace WHERE n.nspname=%s AND p.proname='canonical_string'",
            (setup.schema,),
        ).fetchone()[0]
        conn.execute(sql.SQL("DROP FUNCTION {}.canonical_string(bytea,integer)").format(schema))
        conn.execute(
            sql.SQL(
                "CREATE FUNCTION {}.canonical_string(p_bytes bytea,p_position integer) "
                "RETURNS TABLE(next_position bigint,decoded_key bytea) LANGUAGE plpgsql "
                "SECURITY DEFINER SET search_path=pg_catalog AS {}"
            ).format(schema, sql.Literal(original_source))
        )
        conn.execute(
            sql.SQL("ALTER FUNCTION {}.canonical_string(bytea,integer) OWNER TO {}").format(
                schema, sql.Identifier(setup.schema_owner_role)
            )
        )
        conn.execute(
            sql.SQL("REVOKE ALL ON FUNCTION {}.canonical_string(bytea,integer) FROM PUBLIC").format(
                schema
            )
        )
        manifest = conn.execute(
            sql.SQL("SELECT function_definition_sha256 FROM {}.metadata").format(schema)
        ).fetchone()[0]
        assert manifest[signature] == checkpoint_schema._source_fingerprint(
            signature, original_source
        )
        conn.execute(
            sql.SQL("UPDATE {}.metadata SET physical_definition_sha256=%s").format(schema),
            (checkpoint_schema._physical_fingerprint(conn, setup.schema),),
        )
    with pytest.raises(HistoryContractError):
        boundary.checkpoint.current_checkpoint()
    with pytest.raises(HistoryContractError):
        boundary.restart()


@pytest.mark.parametrize("confirmation", [True, "1", 1.0])
def test_final_lifecycle_sample_must_be_exact_generation_integer(
    boundary, monkeypatch, confirmation
):
    head = _append_head(boundary, 1)
    calls = 0

    def final_wrong_type():
        nonlocal calls
        calls += 1
        return confirmation if calls >= 3 else 1

    monkeypatch.setattr(boundary.signer, "lifecycle_generation", final_wrong_type)
    with pytest.raises(HistoryContractError):
        boundary.checkpoint.advance(expected_revision=0, head=head)
    assert boundary.checkpoint.current_checkpoint() is None


class _InheritedStorageFailure(implementation.CheckpointStorageUnavailable):
    """Obcy typ wyjątku nie staje się zaufaną klasyfikacją przez dziedziczenie."""


@pytest.mark.parametrize(
    "failure,outcome",
    [
        (implementation.CheckpointStorageUnavailable, ReconciliationOutcome.UNAVAILABLE),
        (implementation.CheckpointOperationIndeterminate, ReconciliationOutcome.INDETERMINATE),
        (HistoryContractError, ReconciliationOutcome.CORRUPT),
        (_InheritedStorageFailure, ReconciliationOutcome.CORRUPT),
    ],
)
def test_reconciliation_distinguishes_trusted_storage_failures_from_foreign_reports(
    boundary, monkeypatch, failure, outcome
):
    head = _append_head(boundary, 1)
    boundary.checkpoint.advance(expected_revision=0, head=head)
    reference = ReferenceAuthenticatedHistory(boundary.stream)
    reference.append(
        expected_digest=NO_PREDECESSOR,
        event_identity=_event(1),
        payload={"fixture": "checkpoint-no-issuance", "number": 1, "unicode": "ą"},
    )

    def fail_read(self):
        raise failure("Kontrolowany błąd autorytatywnego odczytu")

    monkeypatch.setattr(
        implementation.PostgreSQLCheckpointAuthority, "current_checkpoint", fail_read
    )
    for history in (boundary.history, reference):
        assert (
            history.reconcile(
                head=head,
                verification_authority=boundary.signer,
                checkpoint_authority=boundary.checkpoint,
            )
            is outcome
        )


def test_unreviewed_checkpoint_object_is_not_reconciliation_authority(boundary):
    head = _append_head(boundary, 1)

    class CallerCheckpointReport:
        def current_checkpoint(self):
            return LocalCheckpoint(
                boundary.checkpoint_id,
                boundary.stream,
                head.sequence,
                head.record_digest,
                head.signing_credential_identity,
                1,
            )

    with pytest.raises(TypeError):
        boundary.history.reconcile(
            head=head,
            verification_authority=boundary.signer,
            checkpoint_authority=CallerCheckpointReport(),
        )


def test_checkpoint_provider_cannot_be_subclassed_to_forge_qualified_authority():
    with pytest.raises(TypeError, match="cannot be subclassed"):

        class ForgedCheckpoint(implementation.PostgreSQLCheckpointAuthority):
            pass


def test_retained_sequence_overflow_is_corruption_not_storage_unavailability(boundary):
    head = _append_head(boundary, 1)
    current, _ = boundary.checkpoint.advance(expected_revision=0, head=head)
    material = implementation._checkpoint_material(current)
    material["history_sequence"] = 2**100
    with psycopg.connect(BASE_DSN, autocommit=True) as conn:
        conn.execute(
            sql.SQL("UPDATE {}.entries SET checkpoint=%s WHERE revision=1").format(
                sql.Identifier(boundary.checkpoint_setup.schema)
            ),
            (canonical_json_bytes(material),),
        )
    with pytest.raises(HistoryContractError) as caught:
        boundary.checkpoint.current_checkpoint()
    assert type(caught.value) is HistoryContractError


def test_advance_result_mutation_cannot_change_configured_authority_or_replay(boundary):
    head = _append_head(boundary, 1)
    returned, replay = boundary.checkpoint.advance(expected_revision=0, head=head)
    assert replay is False
    expected = boundary.checkpoint.current_checkpoint()
    object.__setattr__(returned.stream, "environment", "caller-replaced-environment")
    object.__setattr__(
        returned.history_signing_credential_identity, "provider_namespace", "caller-namespace"
    )
    assert boundary.checkpoint.current_checkpoint() == expected
    assert boundary.checkpoint.advance(expected_revision=1, head=head) == (expected, True)
    assert boundary.checkpoint.retained_attested_heads() == (head,)

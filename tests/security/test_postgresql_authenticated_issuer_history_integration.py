"""Real PostgreSQL proof of durable history semantics, separate from issuance."""

from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path

import psycopg
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from psycopg import sql
from psycopg.conninfo import make_conninfo
from psycopg.errors import InsufficientPrivilege

from bot_core import (
    postgresql_authenticated_issuer_history as postgres_history,
    postgresql_issuer_history_schema as history_schema,
)
from bot_core.authenticated_issuer_history import (
    NO_PREDECESSOR,
    HistoryContractError,
    HistoryEventIdentity,
    HistoryRecord,
    HistoryStreamIdentity,
    LocalCheckpointProvider,
    ReconciliationOutcome,
    ReferenceAuthenticatedHistory,
    attest_head,
    build_record,
    canonical_json_bytes,
)
from bot_core.local_signing_custody import SigningKeyLifecycle
from bot_core.postgresql_entitlement_registry import (
    PostgreSQLConnectionConfig,
    PostgreSQLEntitlementProvisioningAdminProvider,
    PostgreSQLEntitlementRegistryProvider,
    PostgreSQLRegistryProvisioning,
    provision_postgresql_entitlement_registry,
)
from bot_core.root_proof_issuer_substrate import (
    CredentialRoleIdentity,
    CredentialSemanticRole,
    IssuerAuthenticatedHistory,
    ProviderIdentity,
    ProviderQualificationPolicy,
    ProviderRole,
    SecurityProfile,
    SecurityProfileIdentity,
    public_key_material_identity,
)

pytestmark = pytest.mark.external_postgresql
BASE_DSN = os.environ.get(
    "DUDZIAN_TEST_POSTGRES_DSN",
    "host=127.0.0.1 port=55432 dbname=postgres user=postgres",
)


def _connection(role: str) -> PostgreSQLConnectionConfig:
    return PostgreSQLConnectionConfig(make_conninfo(BASE_DSN, user=role))


class _HistorySigner:
    """Isolated test signing custody; PostgreSQL receives only public evidence."""

    def __init__(self, trust_domain: str, public_key: bytes | None = None):
        self._key = None if public_key is not None else Ed25519PrivateKey.generate()
        self._public = public_key
        if self._key is not None:
            self._public = self._key.public_key().public_bytes(
                serialization.Encoding.Raw, serialization.PublicFormat.Raw
            )
        self.identity = ProviderIdentity(
            ProviderRole.HISTORY_ATTESTATION_SIGNING,
            SecurityProfileIdentity(SecurityProfile.PRODUCTION_LOCAL, trust_domain),
            "isolated-history-test-signing",
        )

    def credential_identities(self):
        return (
            CredentialRoleIdentity(
                CredentialSemanticRole.HISTORY_ATTESTATION_SIGNING,
                "isolated-history-credential",
                self.identity.provider_namespace,
                "1",
                "isolated-history-lifecycle",
                public_key_material_identity(self._public),
            ),
        )

    def active_credential_identity(self):
        return self.credential_identities()[0]

    def sign_history_head(self, payload):
        assert self._key is not None
        return self._key.sign(payload)

    def public_key(self, credential_identity):
        assert credential_identity == "isolated-history-credential"
        return self._public

    def lifecycle_generation(self):
        return 1

    def lifecycle_state(self):
        return SigningKeyLifecycle.ACTIVE


@dataclass(frozen=True)
class _HistoryBoundary:
    setup: object
    stream: HistoryStreamIdentity
    runtime: postgres_history.PostgreSQLAuthenticatedIssuerHistory


def _event(number: int) -> HistoryEventIdentity:
    return HistoryEventIdentity(
        f"isolated-operation-{number}",
        f"isolated-attempt-{number}",
        f"isolated-proof-{number}",
    )


def _payload(number: int) -> dict[str, object]:
    return {"fixture": "not-an-issuance-decision", "generation": number, "unicode": "ą"}


def _runtime(boundary: _HistoryBoundary, *, stream=None, role=None):
    return postgres_history.PostgreSQLAuthenticatedIssuerHistory(
        _connection(role or boundary.setup.runtime_role),
        schema=boundary.setup.schema,
        stream=stream or boundary.stream,
    )


@contextmanager
def _isolated_history(*, stream_changes=None):
    suffix = uuid.uuid4().hex[:10]
    stream = HistoryStreamIdentity(
        f"isolated-history-{suffix}",
        "IndependentAccountGenesisRootProofIssuer",
        "PRODUCTION_LOCAL",
        "PRODUCTION",
        "isolated-history-trust",
        "isolated-product",
        1,
    )
    if stream_changes is not None:
        stream = replace(stream, **stream_changes)
    setup = history_schema.PostgreSQLIssuerHistoryProvisioning(
        f"ih_{suffix}",
        f"iho_{suffix}",
        f"ihr_{suffix}",
        f"iha_{suffix}",
        stream.trust_domain,
    )
    try:
        history_schema.provision_postgresql_issuer_history(
            PostgreSQLConnectionConfig(BASE_DSN), setup
        )
        history_schema.provision_postgresql_issuer_history_stream(
            _connection(setup.admin_role), schema=setup.schema, stream=stream
        )
        runtime = postgres_history.PostgreSQLAuthenticatedIssuerHistory(
            _connection(setup.runtime_role), schema=setup.schema, stream=stream
        )
        yield _HistoryBoundary(setup, stream, runtime)
    finally:
        with psycopg.connect(BASE_DSN, autocommit=True) as conn:
            conn.execute(
                sql.SQL("DROP SCHEMA IF EXISTS {} CASCADE").format(sql.Identifier(setup.schema))
            )
            for role in (setup.runtime_role, setup.admin_role, setup.schema_owner_role):
                conn.execute(sql.SQL("DROP ROLE IF EXISTS {}").format(sql.Identifier(role)))


@pytest.fixture
def history():
    with _isolated_history() as boundary:
        yield boundary


def _append(boundary: _HistoryBoundary, number: int):
    head = boundary.runtime.current_record()
    record, replayed = boundary.runtime.append(
        expected_digest=NO_PREDECESSOR if head is None else head.authenticated_digest,
        event_identity=_event(number),
        payload=_payload(number),
    )
    assert not replayed
    return record


def _configuration(boundary: _HistoryBoundary, **changes):
    configuration = {
        "dsn": _connection(boundary.setup.runtime_role).dsn,
        "schema": boundary.setup.schema,
        "stream": boundary.stream.material(),
    }
    configuration.update(changes)
    return configuration


def _process(source: str, configuration):
    return subprocess.Popen(
        [sys.executable, "-c", source, json.dumps(configuration)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def _finish(process):
    stdout, stderr = process.communicate(timeout=30)
    assert process.returncode == 0, stderr
    return json.loads(stdout)


def _kill(process):
    if process.poll() is None:
        process.kill()
    process.communicate(timeout=10)


def _expect_worker_line(process, expected):
    assert process.stdout is not None
    line = process.stdout.readline().strip()
    if line != expected:
        _, stderr = process.communicate(timeout=10)
        pytest.fail(f"worker did not reach {expected!r}: {line!r}; {stderr}")


_WORKER_PREAMBLE = """
import json
import sys
from bot_core import postgresql_authenticated_issuer_history as implementation
from bot_core.authenticated_issuer_history import (
    NO_PREDECESSOR, HistoryContractError, HistoryEventIdentity, HistoryStreamIdentity)
from bot_core.postgresql_entitlement_registry import PostgreSQLConnectionConfig
configuration = json.loads(sys.argv[1])
history = implementation.PostgreSQLAuthenticatedIssuerHistory(
    PostgreSQLConnectionConfig(configuration['dsn']), schema=configuration['schema'],
    stream=HistoryStreamIdentity(**configuration['stream']))
"""


def test_real_provider_identity_and_durable_role_boundary(history):
    runtime = history.runtime
    assert isinstance(runtime, IssuerAuthenticatedHistory)
    assert runtime.identity.role is ProviderRole.ISSUER_AUTHENTICATED_HISTORY
    assert runtime.identity.security == SecurityProfileIdentity(
        SecurityProfile.PRODUCTION_LOCAL, history.stream.trust_domain
    )
    assert (
        ProviderQualificationPolicy().failures_for(
            SecurityProfile.PRODUCTION_LOCAL, runtime.identity, runtime.capabilities
        )
        == ()
    )
    assert "user=" not in repr(runtime)
    for method in ("bind", "issue", "sign", "provision_credential", "advance_checkpoint"):
        assert not hasattr(runtime, method)
    with psycopg.connect(BASE_DSN) as conn:
        settings = dict(
            conn.execute(
                "SELECT name,setting FROM pg_settings WHERE name IN "
                "('server_version_num','fsync','synchronous_commit')"
            ).fetchall()
        )
        assert int(settings["server_version_num"]) >= 160000
        assert settings["fsync"] == settings["synchronous_commit"] == "on"
        roles = dict(
            conn.execute(
                "SELECT rolname,rolcanlogin FROM pg_roles WHERE rolname=ANY(%s)",
                (
                    [
                        history.setup.schema_owner_role,
                        history.setup.runtime_role,
                        history.setup.admin_role,
                    ],
                ),
            ).fetchall()
        )
        assert roles[history.setup.schema_owner_role] is False
        assert roles[history.setup.runtime_role] is True


def test_complete_history_and_attested_head_survive_fresh_processes(history):
    source = (
        _WORKER_PREAMBLE
        + """
expected = NO_PREDECESSOR
records = []
for number in range(1, 4):
    record, replayed = history.append(expected_digest=expected,
        event_identity=HistoryEventIdentity('isolated-operation-' + str(number),
            'isolated-attempt-' + str(number), 'isolated-proof-' + str(number)),
        payload={'fixture': 'not-an-issuance-decision', 'generation': number, 'unicode': 'ą'})
    assert not replayed
    records.append(record.unsigned_material()
        | {'authenticated_digest': record.authenticated_digest})
    expected = record.authenticated_digest
print(json.dumps(records), flush=True)
"""
    )
    process = _process(source, _configuration(history))
    try:
        persisted = _finish(process)
    finally:
        _kill(process)
    records = history.runtime.retained_history()
    assert len(records) == 3
    assert persisted == [
        record.unsigned_material() | {"authenticated_digest": record.authenticated_digest}
        for record in records
    ]
    signer = _HistorySigner(history.stream.trust_domain)
    head = attest_head(records[-1], signer)
    assert history.runtime.retain_attested_head(head, signer) == (head, False)
    source = (
        _WORKER_PREAMBLE
        + """
from tests.security.test_postgresql_authenticated_issuer_history_integration import _HistorySigner
authority = _HistorySigner(configuration['stream']['trust_domain'],
    bytes.fromhex(configuration['public_key']))
records = history.retained_history()
head = history.retained_attested_head(verification_authority=authority)
retained = [r.unsigned_material() | {'authenticated_digest': r.authenticated_digest}
    for r in records]
print(json.dumps({'records': retained, 'sequence': head.sequence,
    'record_digest': head.record_digest,
    'attestation': head.canonical_attestation_bytes.hex(), 'signature': head.signature.hex()}),
    flush=True)
"""
    )
    process = _process(
        source,
        _configuration(history, public_key=signer.public_key("isolated-history-credential").hex()),
    )
    try:
        recovered = _finish(process)
    finally:
        _kill(process)
    assert recovered == {
        "records": persisted,
        "sequence": 3,
        "record_digest": records[-1].authenticated_digest,
        "attestation": head.canonical_attestation_bytes.hex(),
        "signature": head.signature.hex(),
    }
    restarted = _runtime(history)
    assert restarted.current_record() == records[-1]
    assert restarted.record_at(2) == records[1]
    assert restarted.retain_attested_head(head, signer) == (head, True)
    assert restarted.append(
        expected_digest=NO_PREDECESSOR, event_identity=_event(1), payload=_payload(1)
    ) == (records[0], True)


def test_separate_processes_racing_same_predecessor_have_one_cas_winner(history):
    source = (
        _WORKER_PREAMBLE
        + """
print('ready', flush=True)
sys.stdin.readline()
number = configuration['number']
try:
    record, replayed = history.append(expected_digest=NO_PREDECESSOR,
        event_identity=HistoryEventIdentity('race-operation-' + str(number),
            'race-attempt-' + str(number), 'race-proof-' + str(number)),
        payload={'fixture': 'not-an-issuance-decision', 'number': number})
    print(json.dumps({'outcome': 'committed', 'digest': record.authenticated_digest,
        'operation': record.event_identity.logical_operation_id, 'replayed': replayed}), flush=True)
except HistoryContractError:
    print(json.dumps({'outcome': 'conflict'}), flush=True)
"""
    )
    processes = []
    try:
        for number in (1, 2):
            processes.append(_process(source, _configuration(history, number=number)))
        for process in processes:
            _expect_worker_line(process, "ready")
        for process in processes:
            assert process.stdin is not None
            process.stdin.write("go\n")
            process.stdin.flush()
        outcomes = [_finish(process) for process in processes]
    finally:
        for process in processes:
            _kill(process)
    assert sorted(item["outcome"] for item in outcomes) == ["committed", "conflict"]
    winner = next(item for item in outcomes if item["outcome"] == "committed")
    assert winner["replayed"] is False
    records = history.runtime.retained_history()
    assert len(records) == 1
    assert records[0].authenticated_digest == winner["digest"]
    assert records[0].event_identity.logical_operation_id == winner["operation"]
    history.runtime.verify()


@pytest.mark.parametrize(
    "cut,committed",
    [("before_append", False), ("after_append_before_commit", False), ("after_commit", True)],
)
def test_process_kill_at_transaction_cuts_recovers_exactly_committed_history(
    history, cut, committed
):
    first = _append(history, 1)
    source = (
        _WORKER_PREAMBLE
        + """
def pause(name):
    if name == configuration['cut']:
        print('cut', flush=True)
        sys.stdin.readline()
implementation._persistence_cut = pause
history.append(expected_digest=configuration['expected_digest'],
    event_identity=HistoryEventIdentity('isolated-operation-2',
        'isolated-attempt-2', 'isolated-proof-2'),
    payload={'fixture': 'not-an-issuance-decision', 'generation': 2, 'unicode': 'ą'})
raise AssertionError('parent should kill the worker at its controlled transaction cut')
"""
    )
    process = _process(
        source, _configuration(history, cut=cut, expected_digest=first.authenticated_digest)
    )
    try:
        _expect_worker_line(process, "cut")
    finally:
        _kill(process)
    restarted = _runtime(history)
    records = restarted.retained_history()
    assert records[0] == first
    assert len(records) == (2 if committed else 1)
    expected_record = build_record(
        history.stream, 2, first.authenticated_digest, _event(2), _payload(2)
    )
    result = restarted.append(
        expected_digest=first.authenticated_digest, event_identity=_event(2), payload=_payload(2)
    )
    assert result == (expected_record, committed)
    assert restarted.retained_history() == (first, expected_record)
    with pytest.raises(HistoryContractError, match="conflict"):
        restarted.append(
            expected_digest=expected_record.authenticated_digest,
            event_identity=_event(2),
            payload={"fixture": "different-payload"},
        )
    assert restarted.retained_history() == (first, expected_record)


def test_legal_and_illegal_append_sequences_match_reference_oracle(history):
    oracle = ReferenceAuthenticatedHistory(history.stream)
    first, _ = oracle.append(
        expected_digest=NO_PREDECESSOR, event_identity=_event(1), payload=_payload(1)
    )
    operations = [
        (NO_PREDECESSOR, _event(1), _payload(1)),
        ("ignored-for-exact-retry", _event(1), dict(reversed(list(_payload(1).items())))),
        (first.authenticated_digest, _event(1), _payload(2)),
        (NO_PREDECESSOR, _event(2), _payload(2)),
        (first.authenticated_digest, _event(2), _payload(2)),
        (first.authenticated_digest, _event(3), _payload(3)),
    ]
    oracle = ReferenceAuthenticatedHistory(history.stream)
    for expected, event, payload in operations:
        arguments = {"expected_digest": expected, "event_identity": event, "payload": payload}
        try:
            expected_result = oracle.append(**arguments)
        except HistoryContractError:
            with pytest.raises(HistoryContractError):
                history.runtime.append(**arguments)
        else:
            assert history.runtime.append(**arguments) == expected_result
        assert history.runtime.current_record() == oracle.current_record()
        history.runtime.verify()
        oracle.verify()
    previous = oracle.current_record()
    assert previous is not None
    third, replayed = oracle.append(
        expected_digest=previous.authenticated_digest,
        event_identity=_event(3),
        payload={"items": [None, True, False, 0, 2**70, {"name": "ą"}]},
    )
    assert history.runtime.append(
        expected_digest=previous.authenticated_digest,
        event_identity=_event(3),
        payload={"items": [None, True, False, 0, 2**70, {"name": "ą"}]},
    ) == (third, replayed)
    for payload in (
        {"number": 1.0},
        {"number": float("nan")},
        {"number": float("inf")},
        {"number": -float("inf")},
        {"bad": b"bytes"},
    ):
        with pytest.raises(TypeError):
            oracle.append(
                expected_digest=third.authenticated_digest,
                event_identity=_event(4),
                payload=payload,
            )
        with pytest.raises(TypeError):
            history.runtime.append(
                expected_digest=third.authenticated_digest,
                event_identity=_event(4),
                payload=payload,
            )
    assert history.runtime.current_record() == third
    for event in (
        HistoryEventIdentity(_event(3).logical_operation_id, "new-attempt", "new-proof"),
        HistoryEventIdentity("new-operation", _event(3).issuance_attempt_id, "new-proof"),
        HistoryEventIdentity("another-operation", "another-attempt", _event(3).root_proof_id),
    ):
        previous = oracle.current_record()
        arguments = {
            "expected_digest": previous.authenticated_digest,
            "event_identity": event,
            "payload": {"fixture": "composite-event-identity"},
        }
        assert history.runtime.append(**arguments) == oracle.append(**arguments)


def test_canonical_edges_and_large_security_epoch_match_oracle_exactly():
    with _isolated_history(
        stream_changes={
            "stream_id": "history\x00stream",
            "environment": "PRODUCTION\nfixture",
            "trust_domain": "history\x00trust",
            "product_scope": "product\tą",
            "security_epoch": 2**70,
        }
    ) as boundary:
        event = HistoryEventIdentity("operation\x00ą", "attempt\n\t", "proof\r")
        payload = {
            "\x00": "control\x00\b\f\n\r\t",
            "😀": [None, True, False, 0, -(2**120), {"ą": "zażółć", "z": "end"}],
            "𐀀": 2**120,
            "\ue000": "unicode-codepoint-order",
        }
        oracle = ReferenceAuthenticatedHistory(boundary.stream)
        expected = oracle.append(
            expected_digest=NO_PREDECESSOR, event_identity=event, payload=payload
        )
        assert (
            boundary.runtime.append(
                expected_digest=NO_PREDECESSOR, event_identity=event, payload=payload
            )
            == expected
        )
        assert _runtime(boundary).append(
            expected_digest="ignored-on-replay",
            event_identity=event,
            payload=dict(reversed(list(payload.items()))),
        ) == (expected[0], True)
        assert boundary.runtime.current_record().stream.security_epoch == 2**70
        assert boundary.runtime.retained_history() == (expected[0],)


def test_returned_records_and_heads_are_detached_from_adapter_state(history):
    admitted_stream = HistoryStreamIdentity(**history.stream.material())
    record = _append(history, 1)
    expected = build_record(admitted_stream, 1, NO_PREDECESSOR, _event(1), _payload(1))
    object.__setattr__(history.stream, "product_scope", "mutated-constructor-input")
    object.__setattr__(record.stream, "product_scope", "mutated-returned-stream")
    object.__setattr__(record.event_identity, "root_proof_id", "mutated-returned-event")
    object.__setattr__(record, "canonical_event_payload", {"mutated": True})
    assert history.runtime.current_record() == expected
    for returned in (
        history.runtime.current_record(),
        history.runtime.record_at(1),
        history.runtime.retained_history()[0],
    ):
        object.__setattr__(returned.stream, "security_epoch", 99)
        object.__setattr__(returned.event_identity, "logical_operation_id", "mutated-read-event")
        assert history.runtime.current_record() == expected
    signer = _HistorySigner(admitted_stream.trust_domain)
    head = attest_head(history.runtime.current_record(), signer)
    history.runtime.retain_attested_head(head, signer)
    canonical_head = head.canonical_attestation_bytes
    object.__setattr__(head.stream, "product_scope", "mutated-input-head")
    recovered = history.runtime.retained_attested_head(verification_authority=signer)
    assert recovered.stream == admitted_stream
    assert recovered.canonical_attestation_bytes == canonical_head
    object.__setattr__(recovered.stream, "product_scope", "mutated-returned-head")
    object.__setattr__(recovered, "signature", b"changed")
    again = history.runtime.retained_attested_head(verification_authority=signer)
    assert again.stream == admitted_stream
    assert again.canonical_attestation_bytes == canonical_head
    assert history.runtime.current_record() == expected


def test_second_valid_signer_cannot_replace_retained_head_for_same_sequence(history):
    record = _append(history, 1)
    first_signer = _HistorySigner(history.stream.trust_domain)
    first_head = attest_head(record, first_signer)
    assert history.runtime.retain_attested_head(first_head, first_signer) == (first_head, False)
    second_signer = _HistorySigner(history.stream.trust_domain)
    second_head = attest_head(record, second_signer)
    assert second_head.signing_credential_identity != first_head.signing_credential_identity
    with pytest.raises(HistoryContractError, match="conflict"):
        history.runtime.retain_attested_head(second_head, second_signer)
    assert history.runtime.retained_attested_head(verification_authority=first_signer) == first_head


@pytest.mark.parametrize("function", ["read_stream", "append_record", "retain_head"])
def test_direct_runtime_functions_reject_disabled_synchronous_commit(history, function):
    queries = {
        "read_stream": sql.SQL("SELECT {}.read_stream(%s,false)"),
        "append_record": sql.SQL("SELECT {}.append_record(%s,NULL,NULL,NULL)"),
        "retain_head": sql.SQL("SELECT {}.retain_head(%s,NULL,NULL,NULL)"),
    }
    with psycopg.connect(_connection(history.setup.runtime_role).dsn) as conn:
        conn.execute("SET LOCAL synchronous_commit=off")
        with pytest.raises(InsufficientPrivilege, match="durability"):
            conn.execute(
                queries[function].format(sql.Identifier(history.setup.schema)),
                (canonical_json_bytes(history.stream.material()),),
            )
    assert history.runtime.current_record() is None


def test_lost_commit_response_is_indeterminate_until_exact_identity_retry(history, monkeypatch):
    def lose_response(cut):
        if cut == "after_commit":
            raise psycopg.OperationalError("isolated transport lost the committed response")

    monkeypatch.setattr(postgres_history, "_persistence_cut", lose_response)
    with pytest.raises(postgres_history.HistoryOperationIndeterminate):
        history.runtime.append(
            expected_digest=NO_PREDECESSOR, event_identity=_event(1), payload=_payload(1)
        )
    monkeypatch.setattr(postgres_history, "_persistence_cut", lambda cut: None)
    persisted = history.runtime.current_record()
    assert persisted == build_record(history.stream, 1, NO_PREDECESSOR, _event(1), _payload(1))
    assert _runtime(history).append(
        expected_digest=NO_PREDECESSOR, event_identity=_event(1), payload=_payload(1)
    ) == (persisted, True)
    with pytest.raises(HistoryContractError, match="conflict"):
        history.runtime.append(
            expected_digest=persisted.authenticated_digest,
            event_identity=_event(1),
            payload={"fixture": "changed-after-response-loss"},
        )
    assert history.runtime.retained_history() == (persisted,)


@pytest.mark.parametrize(
    "field,value",
    [
        ("stream_id", "other-history"),
        ("issuer_authority_identity", "other-issuer"),
        ("environment", "TEST"),
        ("trust_domain", "other-trust-domain"),
        ("product_scope", "other-product"),
        ("security_epoch", 2),
    ],
)
def test_exact_stream_boundary_rejects_replayed_adapter_or_record(history, field, value):
    record = _append(history, 1)
    wrong_stream = replace(history.stream, **{field: value})
    with pytest.raises(HistoryContractError):
        _runtime(history, stream=wrong_stream)
    wrong_record = build_record(
        wrong_stream,
        2,
        record.authenticated_digest,
        _event(2),
        _payload(2),
    )
    with pytest.raises(HistoryContractError):
        history.runtime.append_exact_successor(record, wrong_record)
    signer = _HistorySigner(history.stream.trust_domain)
    with pytest.raises(HistoryContractError):
        history.runtime.retain_attested_head(attest_head(wrong_record, signer), signer)
    assert history.runtime.retained_history() == (record,)
    if field != "trust_domain":
        history_schema.provision_postgresql_issuer_history_stream(
            _connection(history.setup.admin_role), schema=history.setup.schema, stream=wrong_stream
        )
        other_stream = _runtime(history, stream=wrong_stream)
        with pytest.raises(HistoryContractError):
            other_stream.append_exact_successor(None, record)
        with pytest.raises(HistoryContractError):
            other_stream.retain_attested_head(attest_head(record, signer), signer)
        assert other_stream.current_record() is None
        assert history.runtime.retained_history() == (record,)
    else:
        with _isolated_history(stream_changes={field: value}) as other_boundary:
            with pytest.raises(HistoryContractError):
                other_boundary.runtime.append_exact_successor(None, record)
            with pytest.raises(HistoryContractError):
                other_boundary.runtime.retain_attested_head(attest_head(record, signer), signer)
            assert other_boundary.runtime.current_record() is None


def test_checkpoint_reconciliation_reports_history_without_claiming_durable_checkpoint(history):
    checkpoint = LocalCheckpointProvider("isolated-reference-checkpoint", history.stream)
    assert (
        history.runtime.reconcile(
            head=None, verification_authority=None, checkpoint_authority=checkpoint
        )
        is ReconciliationOutcome.NOT_FOUND
    )
    record = _append(history, 1)
    signer = _HistorySigner(history.stream.trust_domain)
    head = attest_head(record, signer)
    assert (
        history.runtime.reconcile(
            head=head, verification_authority=signer, checkpoint_authority=checkpoint
        )
        is ReconciliationOutcome.STALE
    )
    with pytest.raises(TypeError, match="exact reference authority"):
        checkpoint.advance(
            expected_revision=0,
            history=history.runtime,
            head=head,
            verification_authority=signer,
        )
    oracle = ReferenceAuthenticatedHistory(history.stream)
    assert oracle.append(
        expected_digest=NO_PREDECESSOR, event_identity=_event(1), payload=_payload(1)
    ) == (record, False)
    checkpoint.advance(
        expected_revision=0,
        history=oracle,
        head=head,
        verification_authority=signer,
    )
    assert (
        history.runtime.reconcile(
            head=head, verification_authority=signer, checkpoint_authority=checkpoint
        )
        is ReconciliationOutcome.EXACT_COMMITTED
    )
    second = _append(history, 2)
    second_head = attest_head(second, signer)
    assert (
        history.runtime.reconcile(
            head=second_head, verification_authority=signer, checkpoint_authority=checkpoint
        )
        is ReconciliationOutcome.STALE
    )
    assert (
        history.runtime.reconcile(
            head=head, verification_authority=signer, checkpoint_authority=checkpoint
        )
        is ReconciliationOutcome.CORRUPT
    )


def test_wrong_login_roles_cannot_read_or_mutate_history(history):
    first = _append(history, 1)
    outsider = f"ihx_{uuid.uuid4().hex[:10]}"
    with psycopg.connect(BASE_DSN, autocommit=True) as conn:
        conn.execute(
            sql.SQL(
                "CREATE ROLE {} LOGIN NOINHERIT NOSUPERUSER NOCREATEDB "
                "NOCREATEROLE NOREPLICATION NOBYPASSRLS"
            ).format(sql.Identifier(outsider))
        )
    try:
        for role in (history.setup.admin_role, outsider, "postgres"):
            with pytest.raises(HistoryContractError):
                _runtime(history, role=role)
        with pytest.raises(psycopg.OperationalError, match="not permitted to log in"):
            with psycopg.connect(_connection(history.setup.schema_owner_role).dsn):
                pytest.fail("NOLOGIN schema owner authenticated")
        for role in (history.setup.runtime_role, history.setup.admin_role, outsider):
            commands = [
                sql.SQL("SELECT * FROM {}.records").format(sql.Identifier(history.setup.schema)),
                sql.SQL("DELETE FROM {}.records").format(sql.Identifier(history.setup.schema)),
                sql.SQL("UPDATE {}.records SET authenticated_digest='tampered'").format(
                    sql.Identifier(history.setup.schema)
                ),
                sql.SQL("INSERT INTO {}.records DEFAULT VALUES").format(
                    sql.Identifier(history.setup.schema)
                ),
            ]
            for command in commands:
                with psycopg.connect(_connection(role).dsn) as conn:
                    with pytest.raises(InsufficientPrivilege):
                        conn.execute(command)
        for role in (history.setup.admin_role, outsider):
            with psycopg.connect(_connection(role).dsn) as conn:
                with pytest.raises(InsufficientPrivilege):
                    conn.execute(
                        sql.SQL("SELECT {}.read_stream(%s,false)").format(
                            sql.Identifier(history.setup.schema)
                        ),
                        (canonical_json_bytes(history.stream.material()),),
                    )
        with psycopg.connect(_connection(history.setup.runtime_role).dsn) as conn:
            with pytest.raises(InsufficientPrivilege):
                conn.execute(
                    sql.SQL("SET ROLE {}").format(sql.Identifier(history.setup.admin_role))
                )
        assert history.runtime.retained_history() == (first,)
    finally:
        with psycopg.connect(BASE_DSN, autocommit=True) as conn:
            conn.execute(sql.SQL("DROP ROLE IF EXISTS {}").format(sql.Identifier(outsider)))


def _replace_record(conn, boundary: _HistoryBoundary, record: HistoryRecord):
    conn.execute(
        sql.SQL(
            "UPDATE {}.records SET predecessor_authenticated_digest=%s,event_digest=%s,"
            "authenticated_digest=%s,canonical_record=%s WHERE stream_identity=%s AND sequence=%s"
        ).format(sql.Identifier(boundary.setup.schema)),
        (
            record.predecessor_authenticated_digest,
            record.event_digest,
            record.authenticated_digest,
            canonical_json_bytes(record.unsigned_material()),
            canonical_json_bytes(boundary.stream.material()),
            record.sequence,
        ),
    )


@pytest.mark.parametrize("tamper", ["predecessor", "digest", "gap", "head_rewind", "noncanonical"])
def test_corruption_fails_closed_for_every_authoritative_read_and_append(history, tamper):
    records = tuple(_append(history, number) for number in range(1, 4))
    signer = _HistorySigner(history.stream.trust_domain)
    head = attest_head(records[-1], signer)
    history.runtime.retain_attested_head(head, signer)
    with psycopg.connect(BASE_DSN) as conn:
        if tamper == "predecessor":
            replacement = build_record(
                history.stream, 2, "sha256:" + "a" * 64, _event(2), _payload(2)
            )
            _replace_record(conn, history, replacement)
        elif tamper == "digest":
            conn.execute(
                sql.SQL("UPDATE {}.records SET authenticated_digest=%s WHERE sequence=2").format(
                    sql.Identifier(history.setup.schema)
                ),
                ("sha256:" + "b" * 64,),
            )
        elif tamper == "gap":
            conn.execute(
                sql.SQL("DELETE FROM {}.records WHERE sequence=2").format(
                    sql.Identifier(history.setup.schema)
                )
            )
        elif tamper == "head_rewind":
            conn.execute(
                sql.SQL("UPDATE {}.streams SET head_sequence=1,head_digest=%s").format(
                    sql.Identifier(history.setup.schema)
                ),
                (records[0].authenticated_digest,),
            )
        else:
            conn.execute(
                sql.SQL("UPDATE {}.records SET canonical_record=%s WHERE sequence=2").format(
                    sql.Identifier(history.setup.schema)
                ),
                (json.dumps(records[1].unsigned_material(), ensure_ascii=False).encode("utf-8"),),
            )
    operations = [
        history.runtime.current_record,
        history.runtime.current_head,
        history.runtime.retained_history,
        history.runtime.verify,
        lambda: history.runtime.record_at(1),
        lambda: history.runtime.verify_attested_head(head, signer),
        lambda: history.runtime.retained_attested_head(verification_authority=signer),
        lambda: history.runtime.append(
            expected_digest=records[-1].authenticated_digest,
            event_identity=_event(4),
            payload=_payload(4),
        ),
        lambda: history.runtime.append(
            expected_digest=NO_PREDECESSOR, event_identity=_event(1), payload=_payload(1)
        ),
    ]
    for operation in operations:
        with pytest.raises(HistoryContractError):
            operation()
    checkpoint = LocalCheckpointProvider("isolated-reference-checkpoint", history.stream)
    assert (
        history.runtime.reconcile(
            head=head, verification_authority=signer, checkpoint_authority=checkpoint
        )
        is ReconciliationOutcome.CORRUPT
    )
    with pytest.raises(HistoryContractError):
        _runtime(history)


def test_replayed_fork_cannot_be_installed_as_an_exact_successor(history):
    first = _append(history, 1)
    second = _append(history, 2)
    fork = build_record(history.stream, 2, first.authenticated_digest, _event(3), _payload(3))
    with pytest.raises(HistoryContractError):
        history.runtime.append_exact_successor(first, fork)
    assert history.runtime.retained_history() == (first, second)
    with psycopg.connect(BASE_DSN) as conn:
        with pytest.raises(psycopg.IntegrityError):
            conn.execute(
                sql.SQL(
                    "INSERT INTO {}.records(stream_identity,sequence,"
                    "predecessor_authenticated_digest,"
                    "event_identity,logical_operation_id,issuance_attempt_id,root_proof_id,event_digest,"
                    "authenticated_digest,canonical_record) VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)"
                ).format(sql.Identifier(history.setup.schema)),
                (
                    canonical_json_bytes(history.stream.material()),
                    fork.sequence,
                    fork.predecessor_authenticated_digest,
                    canonical_json_bytes(fork.event_identity.material()),
                    json.dumps(fork.event_identity.logical_operation_id).encode(),
                    json.dumps(fork.event_identity.issuance_attempt_id).encode(),
                    json.dumps(fork.event_identity.root_proof_id).encode(),
                    fork.event_digest,
                    fork.authenticated_digest,
                    canonical_json_bytes(fork.unsigned_material()),
                ),
            )
    assert history.runtime.retained_history() == (first, second)


def test_retained_attestation_with_modified_signature_fails_trusted_verification(history):
    record = _append(history, 1)
    signer = _HistorySigner(history.stream.trust_domain)
    head = attest_head(record, signer)
    history.runtime.retain_attested_head(head, signer)
    with psycopg.connect(BASE_DSN) as conn:
        conn.execute(
            sql.SQL("UPDATE {}.attestations SET signature=%s WHERE sequence=1").format(
                sql.Identifier(history.setup.schema)
            ),
            (b"\x00" * 64,),
        )
    with pytest.raises(HistoryContractError):
        history.runtime.retained_attested_head(verification_authority=signer)


def _schema_snapshot(schema: str):
    with psycopg.connect(BASE_DSN) as conn:
        tables = conn.execute(
            "SELECT c.relname FROM pg_class c JOIN pg_namespace n ON n.oid=c.relnamespace "
            "WHERE n.nspname=%s AND c.relkind='r' ORDER BY c.relname",
            (schema,),
        ).fetchall()
        return {
            name: conn.execute(
                sql.SQL(
                    "SELECT row_to_json(t)::text FROM {}.{} AS t ORDER BY row_to_json(t)::text"
                ).format(sql.Identifier(schema), sql.Identifier(name))
            ).fetchall()
            for (name,) in tables
        }


def _sqlite_snapshot(path: Path):
    with sqlite3.connect(path) as conn:
        return tuple(conn.iterdump())


def test_history_operations_preserve_existing_cha_and_entitlement_data(history, tmp_path):
    from bot_core.cha_attempt_store import AttemptAuthorization, SQLiteCHAAttemptStore
    from bot_core.entitlement_registry_contract import (
        AdminOutcome,
        EntitlementIdentity,
        EntitlementProvenance,
        ProvisionEntitlementRequest,
        RegistrySubject,
    )

    suffix = uuid.uuid4().hex[:10]
    entitlement = PostgreSQLRegistryProvisioning(
        f"ihe_{suffix}",
        f"iheo_{suffix}",
        f"iher_{suffix}",
        f"ihea_{suffix}",
        "isolated-entitlement-trust",
    )
    try:
        provision_postgresql_entitlement_registry(PostgreSQLConnectionConfig(BASE_DSN), entitlement)
        entitlement_arguments = {
            "schema": entitlement.schema,
            "environment": "PRODUCTION_LOCAL",
            "trust_domain": entitlement.trust_domain,
        }
        admin = PostgreSQLEntitlementProvisioningAdminProvider(
            _connection(entitlement.admin_role), **entitlement_arguments
        )
        runtime = PostgreSQLEntitlementRegistryProvider(
            _connection(entitlement.runtime_role), **entitlement_arguments
        )
        subject = RegistrySubject(
            "existing-entitlement-subject", "PRODUCTION_LOCAL", entitlement.trust_domain
        )
        result = admin.provision_entitlement(
            ProvisionEntitlementRequest(
                subject,
                EntitlementIdentity(
                    "ent_018f3e70-7b5a-7c21-8b9a-0123456789ab",
                    1,
                    "PRODUCTION_LOCAL",
                    entitlement.trust_domain,
                    "isolated-product",
                ),
                EntitlementProvenance(
                    "isolated-provisioner",
                    "isolated-claimant",
                    1,
                    "isolated-authority",
                    "fixture:1",
                    "a" * 64,
                ),
            )
        )
        assert result.outcome is AdminOutcome.COMMITTED
        entitlement_before = _schema_snapshot(entitlement.schema)
        retained_entitlement = runtime.retained_history(subject)
        attempt_path = tmp_path / "existing-cha-attempts.sqlite3"
        authorization = AttemptAuthorization(
            "PRODUCTION_LOCAL",
            "isolated-cha-trust",
            "isolated-product",
            "isolated-cha-operation",
            "acct_018f3e70-7b5c-7c21-8b9a-0123456789ab",
            "1" * 64,
            "fixture:initial-binding",
            "2" * 64,
            "ent_isolated-cha",
            1,
            "CryptoHunterAccountAuthority",
            "ACCOUNT_GENESIS_ROOT_PROOF_ISSUANCE_REQUESTER_V1",
            "isolated-requester-key",
            1,
            "isolated-provisioner",
            "isolated-claimant-key",
            1,
        )
        with SQLiteCHAAttemptStore(attempt_path, authorization.trust_domain) as attempts:
            retained_attempt = attempts.reserve_or_resolve_attempt_id(authorization)
        cha_before = _sqlite_snapshot(attempt_path)
        first = _append(history, 1)
        assert history.runtime.append(
            expected_digest=NO_PREDECESSOR, event_identity=_event(1), payload=_payload(1)
        ) == (first, True)
        with pytest.raises(HistoryContractError):
            history.runtime.append(
                expected_digest=first.authenticated_digest,
                event_identity=_event(1),
                payload={"status": "VERIFIED_NOT_AUTHORIZED_TO_ISSUE"},
            )
        second = _append(history, 2)
        signer = _HistorySigner(history.stream.trust_domain)
        history.runtime.retain_attested_head(attest_head(second, signer), signer)
        assert _schema_snapshot(entitlement.schema) == entitlement_before
        assert runtime.retained_history(subject) == retained_entitlement
        assert _sqlite_snapshot(attempt_path) == cha_before
        with SQLiteCHAAttemptStore(attempt_path, authorization.trust_domain) as attempts:
            assert attempts.attempt(authorization.logical_operation_id) == retained_attempt
    finally:
        with psycopg.connect(BASE_DSN, autocommit=True) as conn:
            conn.execute(
                sql.SQL("DROP SCHEMA IF EXISTS {} CASCADE").format(
                    sql.Identifier(entitlement.schema)
                )
            )
            for role in (
                entitlement.runtime_role,
                entitlement.admin_role,
                entitlement.schema_owner_role,
            ):
                conn.execute(sql.SQL("DROP ROLE IF EXISTS {}").format(sql.Identifier(role)))

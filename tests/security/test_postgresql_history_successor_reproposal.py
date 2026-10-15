"""Durable history candidates retain genuine proof bytes on PostgreSQL 16.

The shared fixture installs every catalog boundary before generating its two
real V2 signed attempts. Each case observes committed physical rows before
privileged isolated-fixture teardown; no runtime recovery resets authority data.
"""

from __future__ import annotations

import multiprocessing
import socket
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import datetime, timezone
from types import SimpleNamespace

import psycopg
import pytest
from psycopg import sql
from psycopg.types.json import Jsonb

from bot_core import (
    local_signing_custody as custody,
    postgresql_atomic_root_proof_commit as atomic_runtime,
    postgresql_root_proof_preparation_qualification as preparation_qualification,
    postgresql_root_proof_preparation_v2 as preparation_runtime,
)
from bot_core.entitlement_registry_contract import (
    EntitlementIdentity,
    ProvisionEntitlementRequest,
    RegistrySubject,
    initial_state_for,
)
from bot_core.licensing import (
    cha_root_proof_attempt_reservation as attempt_boundary,
    cha_root_proof_signed_attempt as signed_boundary,
)
from bot_core.postgresql_entitlement_registry import PostgreSQLConnectionConfig, _state_json
from bot_core.uuid7 import mint_uuid7, reservation_epoch_milliseconds
from bot_core.postgresql_root_proof_issuance_authority_v2 import (
    PostgreSQLRootProofIssuanceAuthorityV2,
)
from deployment import (
    windows_production_cha_account_reservation as installed_account,
    windows_production_cha_operation as installed_cha,
    windows_production_lppi_operation as installed_lppi,
)
from tests.security import (
    test_postgresql_atomic_root_proof_commit as atomic_tests,
    test_postgresql_root_proof_preparation_v2 as preparation_tests,
    test_root_proof_issuer_runtime as preflight_tests,
)
from tests.security.test_postgresql_checkpoint_authority import BASE_DSN
from tests.security.test_postgresql_freshness_authority_core import (
    _restart_controlled_postgresql,
)

pytestmark = pytest.mark.external_postgresql

lineage_monkeypatch = atomic_tests.lineage_monkeypatch
lineage_path = atomic_tests.lineage_path
lineage_challenge = atomic_tests.lineage_challenge
lineage_integration = atomic_tests.lineage_integration
lineage_package = atomic_tests.lineage_package
lineage_lppi = atomic_tests.lineage_lppi
lineage_active = atomic_tests.lineage_active
lineage_committed = atomic_tests.lineage_committed
lineage_cha = atomic_tests.lineage_cha
lineage_reserved = atomic_tests.lineage_reserved
module_native_keyring = atomic_tests.module_native_keyring
genuine_attempt = atomic_tests.genuine_attempt


def _runtime_module():
    from bot_core import postgresql_history_successor_reproposal

    return postgresql_history_successor_reproposal


def _read_all(value):
    pairs = atomic_tests._data_tables(value) + [
        (value.setup.schema, "lifecycle_credentials"),
        (value.setup.schema, "lifecycle_operations"),
        (value.setup.schema, "lifecycle_completions"),
        (value.checkpoint._schema, "checkpoints"),
        (value.checkpoint._schema, "entries"),
        (value.setup.schema, "history_proposals"),
        (value.setup.schema, "history_proposal_current"),
    ]
    with psycopg.connect(BASE_DSN) as observer:
        return {
            pair: observer.execute(
                sql.SQL("SELECT * FROM {} ORDER BY 1,2").format(sql.Identifier(*pair))
            ).fetchall()
            for pair in pairs
        }


def _restore(value, baseline=None):
    """Restore only the isolated fixture, after actual outcome assertions."""
    baseline = value.baseline if baseline is None else baseline
    if _read_all(value) == baseline:
        return
    with psycopg.connect(BASE_DSN) as administrator:
        pairs = list(baseline)
        triggers = administrator.execute(
            "SELECT n.nspname,c.relname,t.tgname,t.tgenabled FROM pg_catalog.pg_trigger t "
            "JOIN pg_catalog.pg_class c ON c.oid=t.tgrelid "
            "JOIN pg_catalog.pg_namespace n ON n.oid=c.relnamespace WHERE n.nspname=ANY(%s)",
            ([schema for schema, _ in pairs],),
        ).fetchall()
        for pair in pairs:
            administrator.execute(
                sql.SQL("ALTER TABLE {} DISABLE TRIGGER ALL").format(sql.Identifier(*pair))
            )
        administrator.execute(
            sql.SQL("TRUNCATE {}").format(sql.SQL(",").join(sql.Identifier(*p) for p in pairs))
        )
        for pair, rows in baseline.items():
            for row in rows:
                administrator.execute(
                    sql.SQL("INSERT INTO {} VALUES({})").format(
                        sql.Identifier(*pair), sql.SQL(",").join(sql.Placeholder() for _ in row)
                    ),
                    tuple(Jsonb(part) if isinstance(part, dict) else part for part in row),
                )
        modes = {"O": "ENABLE", "A": "ENABLE ALWAYS", "R": "ENABLE REPLICA", "D": "DISABLE"}
        for schema, table, trigger, mode in triggers:
            if (schema, table) in pairs:
                administrator.execute(
                    sql.SQL("ALTER TABLE {} {} TRIGGER {}").format(
                        sql.Identifier(schema, table), sql.SQL(modes[mode]), sql.Identifier(trigger)
                    )
                )
        administrator.commit()
    assert _read_all(value) == baseline


def _second_genuine(value, active, path, monkeypatch, *, subject=None):
    """Create a second real V2 request; never manufacture attempt evidence."""
    path.mkdir(mode=0o700)
    subject = value.secondary_subject if subject is None else subject
    connections = value.preparation_connections
    authority = PostgreSQLRootProofIssuanceAuthorityV2(
        connections.entitlement_runtime,
        connections.requester_runtime,
        connections.claimant_runtime,
        deployment=value.preparation_deployment,
        entitlement_subject=subject,
    )
    attempt_path = value.genuine.path.with_name(path.name + "-signed-attempt.sqlite3")
    with monkeypatch.context() as selected:
        selected.setattr(installed_lppi, "_state_path", lambda: path / "operation.json")
        selected.setattr(installed_cha, "_state_path", lambda: path / "cha.json")
        selected.setattr(installed_account, "_state_path", lambda: path / "account.json")
        selected.setattr(attempt_boundary, "_attempt_store_path", lambda: attempt_path)
        upstream = installed_lppi.establish_installed_lppi_authenticated_operation(active)
        cha = installed_cha.establish_installed_cha_logical_operation(upstream)
        binding = installed_account.establish_installed_account_initial_binding(cha)
        authorization = attempt_boundary.resolve_root_proof_issuance_authorization_v2(
            binding, authority
        )
        attempt = attempt_boundary.reserve_root_proof_issuance_attempt(binding, authorization)
        signed = signed_boundary.sign_root_proof_issuance_attempt(attempt)
        current, retained = signed_boundary._signed_snapshot(signed)
        intent = preparation_runtime.PostgreSQLRootProofSigningIntentV2(
            connections.intent_writer,
            deployment=value.preparation_deployment,
            schema=value.intent_setup.schema,
            scope=value.scope,
            verification_authority=value.signer,
            issuer_authority=authority,
        )
        retention = preparation_runtime.PostgreSQLRootProofRetentionV2(
            connections.retention_writer,
            deployment=value.preparation_deployment,
            schema=value.retention_setup.schema,
            scope=value.scope,
            verification_authority=value.signer,
            signing_intent=intent,
        )
        reserved = intent.reserve(binding, signed)
        signature = value.signer.sign_root_proof(reserved.bytes_to_sign)
        prepared = retention.retain_prepared(
            binding, signed, canonical_payload=reserved.canonical_payload, signature=signature
        )
        preparation = atomic_runtime.prepare_atomic_root_proof_commit(
            binding,
            signed,
            signing_intent=intent,
            retention=retention,
            issuer_authority=authority,
            history=value.history,
        )
    return SimpleNamespace(
        binding=binding,
        signed=signed,
        authorization=authorization,
        current=current,
        retained=retained,
        authority=authority,
        intent=intent,
        retention=retention,
        reserved=reserved,
        prepared=prepared,
        preparation=preparation,
        path=attempt_path,
        namespace=path,
        subject=subject,
    )


@pytest.fixture(scope="module")
def module_reproposal_boundary(genuine_attempt, lineage_active, lineage_path, lineage_monkeypatch):
    from bot_core.postgresql_history_successor_reproposal_qualification import (
        migrate_history_successor_reproposal_v1,
    )

    # Run the reviewed genuine fresh-V2 fixture, inserting the real new
    # migration before it resolves or signs its first V2 authorization.
    migrate_preparation = preparation_qualification.migrate_root_proof_preparation_v2
    installed = []

    def install_before_first_attempt(*args, **kwargs):
        preparation, lifecycle = migrate_preparation(*args, **kwargs)
        deployment = migrate_history_successor_reproposal_v1(
            PostgreSQLConnectionConfig(BASE_DSN),
            lifecycle,
            preparation_deployment=preparation,
        )
        installed.append(deployment)
        return deployment.preparation_deployment, deployment.lifecycle_deployment

    with lineage_monkeypatch.context() as selected:
        selected.setattr(
            preparation_qualification,
            "migrate_root_proof_preparation_v2",
            install_before_first_attempt,
        )
        source = preparation_tests.module_v2_boundary.__wrapped__(
            genuine_attempt, lineage_path, lineage_monkeypatch
        )
        value = next(source)
        try:
            assert len(installed) == 1
            value.reproposal_deployment = installed[0]
            value.reproposal = _runtime_module().PostgreSQLHistorySuccessorReproposalV1(
                value.reproposal_deployment,
                admin_connection=atomic_tests._connection(value.setup.test_admin_role),
            )
            value.reserved, value.prepared = preparation_tests._stage(value)
            value.preparation = atomic_runtime.prepare_atomic_root_proof_commit(
                value.genuine.binding,
                value.genuine.opaque,
                signing_intent=value.intent,
                retention=value.retention,
                issuer_authority=value.genuine.authority,
                history=value.history,
            )
            value.secondary = _second_genuine(
                value, lineage_active, lineage_path / "second-fresh-v2", lineage_monkeypatch
            )
            assert value.preparation.locator.attempt != value.secondary.preparation.locator.attempt
            assert (
                value.preparation.locator.entitlement
                != value.secondary.preparation.locator.entitlement
            )
            assert value.preparation.history_record.predecessor_authenticated_digest == (
                value.secondary.preparation.history_record.predecessor_authenticated_digest
            )
            state = value.genuine.authority.entitlement_registry.authoritative_state(
                value.genuine.subject
            ).state
            third_subject = RegistrySubject(
                "reproposal-third-" + uuid.uuid4().hex,
                state.subject.environment,
                state.subject.trust_domain,
            )
            third_identity = EntitlementIdentity(
                mint_uuid7("ent_", reservation_epoch_milliseconds(datetime.now(timezone.utc))),
                1,
                state.identity.environment,
                state.identity.trust_domain,
                state.identity.product_scope,
            )
            third_state = initial_state_for(
                ProvisionEntitlementRequest(third_subject, third_identity, state.provenance)
            )
            with psycopg.connect(value.genuine.entitlement_admin._connection.dsn) as admin:
                admin.execute("SET TRANSACTION ISOLATION LEVEL SERIALIZABLE")
                admin.execute(
                    sql.SQL("SELECT {}.provision(%s,%s,%s,%s,%s,%s::json)").format(
                        sql.Identifier(value.genuine.entitlement_admin._schema)
                    ),
                    (
                        third_subject.lookup_handle,
                        third_subject.environment,
                        third_subject.trust_domain,
                        third_identity.product_scope,
                        third_identity.bootstrap_entitlement_id,
                        Jsonb(_state_json(third_state)),
                    ),
                )
                admin.commit()
            value.third = _second_genuine(
                value,
                lineage_active,
                lineage_path / "third-fresh-v2",
                lineage_monkeypatch,
                subject=third_subject,
            )
            value.first_proposal = _propose_first(value)
            value.second_proposal = _repropose(value, predecessor=None)
            value.third_proposal = _propose_candidate(value, value.third, None)
            lineage_monkeypatch.setattr(
                attempt_boundary, "_attempt_store_path", lambda: value.secondary.path
            )
            for module, filename in (
                (installed_lppi, "operation.json"),
                (installed_cha, "cha.json"),
                (installed_account, "account.json"),
            ):
                lineage_monkeypatch.setattr(
                    module,
                    "_state_path",
                    lambda filename=filename: value.secondary.namespace / filename,
                )
            assert value.first_proposal.revision == value.second_proposal.revision == 1
            value.signed_attempt_dumps = (
                preflight_tests._database_dump(value.genuine.path),
                preflight_tests._database_dump(value.secondary.path),
            )
            value.baseline = _read_all(value)
            yield value
        finally:
            source.close()


@pytest.fixture
def reproposal_boundary(module_reproposal_boundary):
    value = module_reproposal_boundary
    assert _read_all(value) == value.baseline
    try:
        yield value
    finally:
        _restore(value)


def _propose_first(value):
    return value.reproposal.repropose(
        value.preparation,
        value.genuine.binding,
        value.genuine.opaque,
        signing_intent=value.intent,
        retention=value.retention,
        issuer_authority=value.genuine.authority,
        history=value.history,
        history_signer=value.history_boundary.signer,
        fences=value.fences,
    )


_DEFAULT = object()


def _repropose(value, predecessor=_DEFAULT, **overrides):
    return _propose_candidate(
        value,
        value.secondary,
        value.second_proposal if predecessor is _DEFAULT else predecessor,
        **overrides,
    )


def _propose_candidate(value, candidate, predecessor, **overrides):
    arguments = dict(
        signing_intent=candidate.intent,
        retention=candidate.retention,
        issuer_authority=candidate.authority,
        history=value.history,
        history_signer=value.history_boundary.signer,
        fences=value.fences,
        predecessor=predecessor,
    )
    arguments.update(overrides)
    with pytest.MonkeyPatch.context() as selected:
        selected.setattr(attempt_boundary, "_attempt_store_path", lambda: candidate.path)
        for module, filename in (
            (installed_lppi, "operation.json"),
            (installed_cha, "cha.json"),
            (installed_account, "account.json"),
        ):
            selected.setattr(
                module, "_state_path", lambda filename=filename: candidate.namespace / filename
            )
        return value.reproposal.repropose(
            candidate.preparation, candidate.binding, candidate.signed, **arguments
        )


def _wait_for_competing_database_lock(value):
    roles = [value.setup.issuer_writer_role, value.setup.test_admin_role]
    roles.extend(role for _, role in value.preparation_deployment.component_roles)
    deadline = time.monotonic() + 120
    with psycopg.connect(BASE_DSN, autocommit=True) as observer:
        while time.monotonic() < deadline:
            waits = observer.execute(
                "SELECT wait_event FROM pg_catalog.pg_stat_activity "
                "WHERE usename=ANY(%s) AND wait_event_type='Lock'",
                (roles,),
            ).fetchall()
            if any(wait[0] in {"transactionid", "tuple", "advisory"} for wait in waits):
                return
            time.sleep(0.02)
    pytest.fail("competing action did not reach the actual authority database lock")


def _winner(value):
    decision = value.reproposal.commit(value.first_proposal)
    assert decision.state == "COMMITTED"
    return decision


def _proof_bytes_unchanged(value):
    observed = _read_all(value)
    for schema, table in (
        (value.intent_setup.schema, "intents"),
        (value.retention_setup.schema, "prepared"),
    ):
        assert observed[(schema, table)] == value.baseline[(schema, table)]
    assert (
        preflight_tests._database_dump(value.genuine.path),
        preflight_tests._database_dump(value.secondary.path),
    ) == value.signed_attempt_dumps


def _assert_complete(value, first, second):
    assert second.state == "COMMITTED"
    assert (second.payload, second.signature, second.envelope) == (
        value.secondary.prepared.canonical_payload,
        value.secondary.prepared.signature,
        value.secondary.prepared.canonical_envelope,
    )
    snapshot = _read_all(value)
    records = snapshot[(value.history._schema, "records")]
    attestations = snapshot[(value.history._schema, "attestations")]
    assert len(records) == len(attestations) == 2
    assert len(snapshot[(value.setup.schema, "decisions")]) == 2
    registry_rows = snapshot[(value.genuine.entitlement_admin._schema, "history")]
    for subject in (value.genuine.subject, value.secondary.subject):
        states = [row[5] for row in registry_rows if row[0] == subject.lookup_handle]
        assert len(states) == 2
        assert states[0]["binding"]["kind"] == "UNBOUND"
        assert states[1]["binding"]["kind"] == "BOUND"
    authenticated = value.history.authenticated_snapshot(value.history_boundary.signer)
    assert len(authenticated.heads) == 2
    history = authenticated.records
    assert tuple(record.sequence for record in history) == (1, 2)
    assert history[1].predecessor_authenticated_digest == history[0].authenticated_digest
    assert history[0].event_identity == value.preparation.history_record.event_identity
    assert history[1].event_identity == value.secondary.preparation.history_record.event_identity
    assert history[1].event_digest == value.secondary.preparation.history_record.event_digest
    assert history[1].canonical_event_payload == (
        value.secondary.preparation.history_record.canonical_event_payload
    )
    _proof_bytes_unchanged(value)
    assert value.reproposal.reconcile(value.first_proposal) == first
    return snapshot


def test_stale_second_original_reproposes_history_and_commits_original_exact_proof(
    reproposal_boundary, monkeypatch
):
    value = reproposal_boundary
    first = _winner(value)
    after_first = _read_all(value)
    with pytest.raises(atomic_runtime.AtomicRootProofCommitConflict):
        value.reproposal.commit(value.second_proposal)
    assert _read_all(value) == after_first

    def no_root_sign(*args, **kwargs):
        pytest.fail("history reproposal attempted to resign the root proof")

    monkeypatch.setattr(custody.LocalRootProofSigningProvider, "sign_root_proof", no_root_sign)
    replacement = _repropose(value)
    assert replacement.revision == 2
    assert replacement.predecessor_candidate_id == value.second_proposal.candidate_id
    assert replacement.candidate_id != value.second_proposal.candidate_id
    admitted = _read_all(value)
    for pair in after_first:
        if pair[1] not in {"history_proposals", "history_proposal_current"}:
            assert admitted[pair] == after_first[pair]
    for row in after_first[(value.setup.schema, "history_proposals")]:
        assert row in admitted[(value.setup.schema, "history_proposals")]
    _proof_bytes_unchanged(value)
    with pytest.raises(atomic_runtime.AtomicRootProofCommitConflict):
        value.reproposal.commit(value.second_proposal)
    second = value.reproposal.commit(replacement)
    complete = _assert_complete(value, first, second)
    assert value.reproposal.commit(replacement) == second
    assert value.reproposal.reconcile(replacement) == second
    assert _read_all(value) == complete


def test_parallel_reproposals_have_one_durable_revision_and_exact_replay(reproposal_boundary):
    value = reproposal_boundary
    _winner(value)
    barrier = threading.Barrier(2)

    def propose():
        barrier.wait(timeout=30)
        try:
            return _repropose(value)
        except atomic_runtime.AtomicRootProofCommitConflict as exc:
            return exc

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = list(executor.map(lambda _: propose(), range(2)))
    refs = [item for item in outcomes if not isinstance(item, Exception)]
    assert refs
    assert len({item.candidate_id for item in refs}) == 1
    current = value.reproposal.read_current_proposal(value.secondary.preparation)
    assert current == refs[0] and current.revision == 2
    rows = _read_all(value)[(value.setup.schema, "history_proposals")]
    assert len(rows) == len(value.baseline[(value.setup.schema, "history_proposals")]) + 1
    assert _repropose(value) == current
    assert len(_read_all(value)[(value.setup.schema, "history_proposals")]) == len(rows)


def test_unknown_decision_never_authorizes_an_absent_replacement(reproposal_boundary):
    value = reproposal_boundary
    unknown = replace(value.secondary.preparation.locator, attempt=b'"unknown-attempt"')
    before = _read_all(value)
    with pytest.raises(atomic_runtime.AtomicRootProofCommitIndeterminate):
        value.reproposal.reconcile(unknown)
    assert _read_all(value) == before


def test_committed_attempt_returns_original_decision_without_new_candidate(reproposal_boundary):
    value = reproposal_boundary
    _winner(value)
    replacement = _repropose(value)
    decision = value.reproposal.commit(replacement)
    before = _read_all(value)
    assert _repropose(value, predecessor=replacement) == decision
    assert _read_all(value) == before


def test_real_database_and_fresh_process_restart_replay_original_decisions(reproposal_boundary):
    value = reproposal_boundary
    first = _winner(value)
    replacement = _repropose(value)
    second = value.reproposal.commit(replacement)
    before = _assert_complete(value, first, second)
    _restart_controlled_postgresql()
    assert _read_all(value) == before
    # A fresh OS process inherits only this isolated test's genuine references;
    # construction opens new physical qualified connections after DB restart.
    context = multiprocessing.get_context("fork")
    reader, writer = context.Pipe(duplex=False)

    def recover():
        try:
            runtime = _runtime_module().PostgreSQLHistorySuccessorReproposalV1(
                value.reproposal_deployment,
                admin_connection=atomic_tests._connection(value.setup.test_admin_role),
            )
            decision = runtime.reconcile(replacement)
            assert runtime.commit(replacement) == decision
            writer.send((decision.payload, decision.signature, decision.envelope))
        finally:
            writer.close()

    process = context.Process(target=recover)
    process.start()
    writer.close()
    try:
        assert reader.poll(60), "fresh recovery process produced no authoritative result"
        assert reader.recv() == (second.payload, second.signature, second.envelope)
    finally:
        process.join(timeout=60)
        if process.is_alive():
            process.kill()
            process.join(timeout=10)
        reader.close()
    assert process.exitcode == 0
    assert _read_all(value) == before


def test_history_advancing_again_requires_another_exact_candidate(reproposal_boundary):
    value = reproposal_boundary
    _winner(value)
    second_revision = _repropose(value)
    third_revision = _propose_candidate(value, value.third, value.third_proposal)
    third_decision = value.reproposal.commit(third_revision)
    assert third_decision.payload == value.third.prepared.canonical_payload
    before = _read_all(value)
    with pytest.raises(atomic_runtime.AtomicRootProofCommitConflict):
        value.reproposal.commit(second_revision)
    assert _read_all(value) == before
    latest = _repropose(value, predecessor=second_revision)
    assert latest.revision == 3
    assert latest.predecessor_candidate_id == second_revision.candidate_id
    decision = value.reproposal.commit(latest)
    assert decision.signature == value.secondary.prepared.signature
    authenticated = value.history.authenticated_snapshot(value.history_boundary.signer)
    assert tuple(record.sequence for record in authenticated.records) == (1, 2, 3)
    for preceding, following in zip(authenticated.records, authenticated.records[1:]):
        assert following.predecessor_authenticated_digest == preceding.authenticated_digest
    _proof_bytes_unchanged(value)


@pytest.mark.parametrize(
    "cut", ["before_proposal_admission", "before_proposal_commit", "after_proposal_commit"]
)
def test_admission_fault_cuts_keep_original_proof_and_resolve_only_durable_candidates(
    reproposal_boundary, monkeypatch, cut
):
    value = reproposal_boundary
    _winner(value)
    before = _read_all(value)
    module = _runtime_module()

    def interrupt(name, connection):
        if name == cut:
            raise ConnectionError("injected proposal response loss")

    monkeypatch.setattr(module, "_persistence_cut", interrupt)
    expected = (
        atomic_runtime.AtomicRootProofCommitUnavailable
        if cut == "before_proposal_admission"
        else atomic_runtime.AtomicRootProofCommitIndeterminate
    )
    with pytest.raises(expected):
        _repropose(value)
    monkeypatch.setattr(module, "_persistence_cut", lambda *_: None)
    _restart_controlled_postgresql()
    current = value.reproposal.read_current_proposal(value.secondary.preparation)
    if cut == "after_proposal_commit":
        assert current.revision == 2
        assert _repropose(value) == current
    else:
        assert current == value.second_proposal
        assert _read_all(value) == before
    with pytest.raises(atomic_runtime.AtomicRootProofCommitIndeterminate):
        value.reproposal.reconcile(value.second_proposal.locator)
    _proof_bytes_unchanged(value)


@pytest.mark.parametrize("phase", ["admission", "decision"])
def test_native_commit_response_loss_recovers_durable_exact_result(
    reproposal_boundary, monkeypatch, phase
):
    value = reproposal_boundary
    _winner(value)
    module = _runtime_module()
    proposal = None if phase == "admission" else _repropose(value)
    target = "before_proposal_commit" if phase == "admission" else "before_commit"
    disconnected = []

    def drop_response(name, connection):
        if name == target:
            with socket.fromfd(
                connection.pgconn.socket, socket.AF_INET, socket.SOCK_STREAM
            ) as wire:
                wire.shutdown(socket.SHUT_RD)
            disconnected.append(connection.info.backend_pid)

    monkeypatch.setattr(module, "_persistence_cut", drop_response)
    with pytest.raises(atomic_runtime.AtomicRootProofCommitIndeterminate):
        if phase == "admission":
            _repropose(value)
        else:
            value.reproposal.commit(proposal)
    assert len(disconnected) == 1
    monkeypatch.setattr(module, "_persistence_cut", lambda *_: None)
    deadline = time.monotonic() + 15
    while True:
        try:
            if phase == "admission":
                result = value.reproposal.read_current_proposal(value.secondary.preparation)
                assert result.revision == 2
            else:
                result = value.reproposal.reconcile(proposal)
            break
        except (atomic_runtime.AtomicRootProofCommitIndeterminate, AssertionError):
            if time.monotonic() >= deadline:
                raise
            time.sleep(0.02)
    if phase == "admission":
        assert _repropose(value) == result
    else:
        assert result.signature == value.secondary.prepared.signature
        assert value.reproposal.commit(proposal) == result
    _proof_bytes_unchanged(value)


@pytest.mark.parametrize("cut", ["before_commit", "after_commit"])
def test_decision_commit_cuts_leave_one_whole_physical_outcome(
    reproposal_boundary, monkeypatch, cut
):
    value = reproposal_boundary
    _winner(value)
    proposal = _repropose(value)
    before = _read_all(value)

    def interrupt(name, connection):
        if name == cut:
            raise ConnectionError("injected decision response loss")

    monkeypatch.setattr(_runtime_module(), "_persistence_cut", interrupt)
    with pytest.raises(atomic_runtime.AtomicRootProofCommitIndeterminate):
        value.reproposal.commit(proposal)
    monkeypatch.setattr(_runtime_module(), "_persistence_cut", lambda *_: None)
    _restart_controlled_postgresql()
    if cut == "before_commit":
        assert _read_all(value) == before
        with pytest.raises(atomic_runtime.AtomicRootProofCommitIndeterminate):
            value.reproposal.reconcile(proposal)
    else:
        decision = value.reproposal.reconcile(proposal)
        assert decision.signature == value.secondary.prepared.signature
        assert len(_read_all(value)[(value.setup.schema, "decisions")]) == 2
        assert value.reproposal.commit(proposal) == decision
    _proof_bytes_unchanged(value)


def test_proposal_racing_old_final_commit_cannot_cross_the_selector(
    reproposal_boundary, monkeypatch
):
    value = reproposal_boundary
    _winner(value)
    entered, release = threading.Event(), threading.Event()

    def hold(name, connection):
        if name == "before_proposal_commit":
            entered.set()
            assert release.wait(180)

    monkeypatch.setattr(_runtime_module(), "_persistence_cut", hold)
    with ThreadPoolExecutor(max_workers=2) as executor:
        admission = executor.submit(_repropose, value)
        assert entered.wait(180)
        stale = executor.submit(value.reproposal.commit, value.second_proposal)
        _wait_for_competing_database_lock(value)
        assert not stale.done(), "old commit did not participate in the proposal row lock"
        release.set()
        replacement = admission.result(timeout=180)
        with pytest.raises(atomic_runtime.AtomicRootProofCommitConflict):
            stale.result(timeout=180)
    assert value.reproposal.read_current_proposal(value.secondary.preparation) == replacement
    assert len(_read_all(value)[(value.setup.schema, "decisions")]) == 1
    _proof_bytes_unchanged(value)


@pytest.mark.parametrize("phase", ["admission", "decision"])
def test_real_lifecycle_close_serializes_against_proposal_and_final_commit(
    reproposal_boundary, monkeypatch, phase
):
    from bot_core import postgresql_custody_lifecycle as lifecycle
    from bot_core.root_proof_issuer_substrate import ProviderRole

    value = reproposal_boundary
    _winner(value)
    proposal = None if phase == "admission" else _repropose(value)
    request = lifecycle.prepare_custody_lifecycle_request(
        value.reproposal_deployment.lifecycle_deployment,
        value.directory,
        security=value.security,
        role=ProviderRole.HISTORY_ATTESTATION_SIGNING,
        administrative_operation_id="reproposal-close-" + uuid.uuid4().hex,
        target=custody.SigningKeyLifecycle.REVOKED,
        fence_key=value.fences.history_key,
    )
    administrator = lifecycle.PostgreSQLCustodyLifecycleAdministrator(
        value.reproposal_deployment.lifecycle_deployment
    )
    entered, release = threading.Event(), threading.Event()
    cut = "before_proposal_commit" if phase == "admission" else "before_commit"

    def hold(name, connection):
        if name == cut:
            entered.set()
            assert release.wait(180)

    monkeypatch.setattr(_runtime_module(), "_persistence_cut", hold)
    with ThreadPoolExecutor(max_workers=2) as executor:
        protected = (
            executor.submit(_repropose, value)
            if phase == "admission"
            else executor.submit(value.reproposal.commit, proposal)
        )
        assert entered.wait(180)
        closing = executor.submit(administrator.begin_close, request)
        _wait_for_competing_database_lock(value)
        assert not closing.done(), "lifecycle close crossed held final eligibility locks"
        release.set()
        result = protected.result(timeout=180)
        assert closing.result(timeout=180) is not None
    with psycopg.connect(BASE_DSN) as observer:
        fence = observer.execute(
            sql.SQL("SELECT state FROM {}.fences WHERE identity=%s").format(
                sql.Identifier(value.setup.schema)
            ),
            (value.fences.history_key,),
        ).fetchone()
    assert fence == ("PENDING_CLOSED",)
    monkeypatch.setattr(_runtime_module(), "_persistence_cut", lambda *_: None)
    if phase == "admission":
        with pytest.raises(atomic_runtime.AtomicRootProofCommitConflict):
            value.reproposal.commit(result)
        assert len(_read_all(value)[(value.setup.schema, "decisions")]) == 1
    else:
        assert value.reproposal.reconcile(proposal) == result
        assert value.reproposal.commit(proposal) == result
        assert len(_read_all(value)[(value.setup.schema, "decisions")]) == 2
    _proof_bytes_unchanged(value)


def test_unresolved_live_prior_commit_blocks_supersession_until_its_positive_decision(
    reproposal_boundary, monkeypatch
):
    value = reproposal_boundary
    _winner(value)
    proposal = _repropose(value)
    before = _read_all(value)
    entered, release = threading.Event(), threading.Event()

    def hold(name, connection):
        if name == "before_commit":
            entered.set()
            assert release.wait(180)

    monkeypatch.setattr(_runtime_module(), "_persistence_cut", hold)
    with ThreadPoolExecutor(max_workers=2) as executor:
        committing = executor.submit(value.reproposal.commit, proposal)
        assert entered.wait(180)
        # A distinct committed snapshot sees no decision while the original
        # backend has already written it and retains all transaction locks.
        with pytest.raises(atomic_runtime.AtomicRootProofCommitIndeterminate):
            value.reproposal.reconcile(proposal)
        superseding = executor.submit(_repropose, value, proposal)
        _wait_for_competing_database_lock(value)
        assert not superseding.done(), "negative read escaped the still-live final commit"
        release.set()
        decision = committing.result(timeout=180)
        try:
            result = superseding.result(timeout=180)
            assert result == decision
        except atomic_runtime.AtomicRootProofCommitConflict:
            pass
    assert value.reproposal.reconcile(proposal) == decision
    after = _read_all(value)
    assert (
        after[(value.setup.schema, "history_proposals")]
        == before[(value.setup.schema, "history_proposals")]
    )
    assert (
        after[(value.setup.schema, "history_proposal_current")]
        == before[(value.setup.schema, "history_proposal_current")]
    )
    _proof_bytes_unchanged(value)


def test_database_restart_after_admission_keeps_exact_pending_candidate(reproposal_boundary):
    value = reproposal_boundary
    _winner(value)
    proposal = _repropose(value)
    before = _read_all(value)
    _restart_controlled_postgresql()
    assert _read_all(value) == before
    restarted = _runtime_module().PostgreSQLHistorySuccessorReproposalV1(
        value.reproposal_deployment,
        admin_connection=atomic_tests._connection(value.setup.test_admin_role),
    )
    assert restarted.read_current_proposal(proposal.locator) == proposal
    with pytest.raises(atomic_runtime.AtomicRootProofCommitIndeterminate):
        restarted.reconcile(proposal)
    decision = restarted.commit(proposal)
    assert decision.signature == value.secondary.prepared.signature
    assert restarted.commit(proposal) == decision
    _proof_bytes_unchanged(value)


@pytest.mark.parametrize("phase", ["admission", "decision"])
def test_backend_termination_before_commit_aborts_entire_proposal_or_decision(
    reproposal_boundary, monkeypatch, phase
):
    value = reproposal_boundary
    _winner(value)
    proposal = None if phase == "admission" else _repropose(value)
    before = _read_all(value)
    target = "before_proposal_commit" if phase == "admission" else "before_commit"

    def terminate(name, connection):
        if name == target:
            with psycopg.connect(BASE_DSN, autocommit=True) as administrator:
                assert administrator.execute(
                    "SELECT pg_catalog.pg_terminate_backend(%s)",
                    (connection.info.backend_pid,),
                ).fetchone() == (True,)

    monkeypatch.setattr(_runtime_module(), "_persistence_cut", terminate)
    with pytest.raises(atomic_runtime.AtomicRootProofCommitIndeterminate):
        if phase == "admission":
            _repropose(value)
        else:
            value.reproposal.commit(proposal)
    monkeypatch.setattr(_runtime_module(), "_persistence_cut", lambda *_: None)
    assert _read_all(value) == before
    with pytest.raises(atomic_runtime.AtomicRootProofCommitIndeterminate):
        value.reproposal.reconcile(value.second_proposal.locator)
    _proof_bytes_unchanged(value)


@pytest.mark.parametrize("corruption", ["signature", "missing_head"])
def test_retained_history_needs_actual_authentic_attestation(reproposal_boundary, corruption):
    value = reproposal_boundary
    _winner(value)
    schema = value.history._schema
    with psycopg.connect(BASE_DSN) as administrator:
        triggers = administrator.execute(
            "SELECT tgname,tgenabled FROM pg_catalog.pg_trigger "
            "WHERE tgrelid=pg_catalog.to_regclass(%s)",
            (schema + ".attestations",),
        ).fetchall()
        administrator.execute(
            sql.SQL("ALTER TABLE {}.attestations DISABLE TRIGGER ALL").format(
                sql.Identifier(schema)
            )
        )
        if corruption == "signature":
            administrator.execute(
                sql.SQL("UPDATE {}.attestations SET signature=%s").format(sql.Identifier(schema)),
                (b"F" * 64,),
            )
        else:
            administrator.execute(
                sql.SQL("DELETE FROM {}.attestations").format(sql.Identifier(schema))
            )
        modes = {"O": "ENABLE", "A": "ENABLE ALWAYS", "R": "ENABLE REPLICA", "D": "DISABLE"}
        for trigger, mode in triggers:
            administrator.execute(
                sql.SQL("ALTER TABLE {}.attestations {} TRIGGER {}").format(
                    sql.Identifier(schema), sql.SQL(modes[mode]), sql.Identifier(trigger)
                )
            )
        administrator.commit()
    before = _read_all(value)
    with pytest.raises(atomic_runtime.AtomicRootProofCommitError):
        _repropose(value)
    assert _read_all(value) == before
    _proof_bytes_unchanged(value)


def test_invalid_signature_from_genuine_history_signer_never_admits_candidate(
    reproposal_boundary, monkeypatch
):
    value = reproposal_boundary
    _winner(value)
    before = _read_all(value)
    calls = []

    def invalid_signature(self, payload):
        calls.append(payload)
        return b"F" * 64

    monkeypatch.setattr(
        custody.LocalHistoryAttestationSigningProvider, "sign_history_head", invalid_signature
    )
    with pytest.raises(atomic_runtime.AtomicRootProofCommitError):
        _repropose(value)
    assert len(calls) == 1
    assert _read_all(value) == before
    _proof_bytes_unchanged(value)


# Collect the complete positive transaction protocol before imported refusals.
from tests.security._history_reproposal_negative_cases import (
    test_direct_role_oid_substitution_refuses_qualified_reproposal,
    test_fresh_process_reconstructs_reviewed_deployment_and_exact_decisions,
    test_fake_history_signer_cannot_supply_reproposal_authority,
    test_caller_head_and_signature_are_not_reproposal_inputs,
    test_changed_original_preparation_cannot_authorize_reproposal,
    test_foreign_genuine_attempt_cannot_replace_original_proposal,
    test_runtime_and_admin_have_no_direct_proposal_dml,
    test_proposal_catalog_acl_and_function_oid_drift_fail_closed,
    test_reproposal_rejects_changed_physical_database_pin,
    test_reproposal_rejects_actual_other_physical_database,
    test_closed_fences_refuse_proposal_and_atomic_commit,
    test_missing_signer_enrollment_refuses_reproposal_and_commit,
)


def test_protected_raw_admission_replay_and_changed_packages_are_fail_closed(reproposal_boundary):
    from copy import deepcopy

    value = reproposal_boundary
    first = _winner(value)
    old = atomic_tests._call_sql(
        value,
        "read_history_proposal",
        (value.second_proposal.locator.attempt, value.second_proposal.candidate_id),
    )
    replacement = _repropose(value)
    before = _read_all(value)
    repeated = atomic_tests._call_sql(
        value, "approve_history_proposal", (Jsonb(old["package"]), 1, None), admin=True
    )
    assert repeated["candidate_id"] == value.second_proposal.candidate_id
    assert value.reproposal.read_current_proposal(value.secondary.preparation) == replacement
    assert _read_all(value) == before
    for field in ("payload_hex", "signature_hex", "reserved_envelope_hex", "account_hex"):
        changed = deepcopy(old["package"])
        prefix = "01" if changed[field].startswith("00") else "00"
        changed[field] = prefix + changed[field][2:]
        with pytest.raises(psycopg.errors.UniqueViolation):
            atomic_tests._call_sql(
                value, "approve_history_proposal", (Jsonb(changed), 1, None), admin=True
            )
        assert _read_all(value) == before

    # An unrelated genuine atomic winner makes the second candidate stale
    # again; forged successor data then reaches the writable admission boundary.
    third = _propose_candidate(value, value.third, value.third_proposal)
    value.reproposal.commit(third)
    retained = atomic_tests._call_sql(
        value, "read_history_proposal", (replacement.locator.attempt, replacement.candidate_id)
    )
    before = _read_all(value)
    for field in (
        "history_head_hex",
        "history_signature_hex",
        "expected_history_digest",
        "history_record_hex",
    ):
        changed = deepcopy(retained["package"])
        changed[field] = "0" * len(changed[field])
        with pytest.raises((psycopg.errors.CheckViolation, psycopg.errors.InvalidParameterValue)):
            atomic_tests._call_sql(
                value,
                "approve_history_proposal",
                (Jsonb(changed), 3, replacement.candidate_id),
                admin=True,
            )
        assert _read_all(value) == before
    with pytest.raises(psycopg.errors.CheckViolation):
        atomic_tests._call_sql(
            value,
            "approve_history_proposal",
            (Jsonb(retained["package"]), 3, value.first_proposal.candidate_id),
            admin=True,
        )
    assert _read_all(value) == before
    assert value.reproposal.reconcile(value.first_proposal) == first
    _proof_bytes_unchanged(value)

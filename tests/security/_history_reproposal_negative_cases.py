"""Imported PostgreSQL reproposal refusals with genuine fixture authority.

This helper is intentionally outside pytest's file discovery pattern. The main
integration module imports its tests so they share the exact durable fixture.
"""

from __future__ import annotations

import importlib
import json
import subprocess
import sys
import uuid
from dataclasses import asdict, replace

import psycopg
import pytest
from psycopg import sql
from psycopg.conninfo import make_conninfo

from bot_core import postgresql_atomic_root_proof_commit as atomic_runtime
from bot_core.postgresql_entitlement_registry import PostgreSQLConnectionConfig
from tests.security.test_postgresql_checkpoint_authority import BASE_DSN


def _main():
    return importlib.import_module("tests.security.test_postgresql_history_successor_reproposal")


def _call(value, *, original=None, binding=None, signed_attempt=None, **overrides):
    candidate = value.secondary
    arguments = dict(
        signing_intent=candidate.intent,
        retention=candidate.retention,
        issuer_authority=candidate.authority,
        history=value.history,
        history_signer=value.history_boundary.signer,
        fences=value.fences,
        predecessor=value.second_proposal,
    )
    arguments.update(overrides)
    return value.reproposal.repropose(
        candidate.preparation if original is None else original,
        candidate.binding if binding is None else binding,
        candidate.signed if signed_attempt is None else signed_attempt,
        **arguments,
    )


def _assert_unmutated(value, before):
    assert _main()._read_all(value) == before
    _main()._proof_bytes_unchanged(value)


def _reconstruct(value, **overrides):
    from bot_core.postgresql_history_successor_reproposal_qualification import (
        requalify_existing_history_successor_reproposal_v1,
    )

    arguments = dict(
        expected_identity=value.reproposal_deployment.atomic_deployment.identity,
        preparation_deployment=value.reproposal_deployment.preparation_deployment,
    )
    arguments.update(overrides)
    return requalify_existing_history_successor_reproposal_v1(
        PostgreSQLConnectionConfig(BASE_DSN),
        value.setup,
        value.writer_connection,
        _main().atomic_tests._connection(value.setup.test_admin_role),
        **arguments,
    )


def _flip(value: bytes) -> bytes:
    return bytes([value[0] ^ 1]) + value[1:]


def _change_fixture_row(value, table, query, parameters):
    """Privileged corruption control restores the original trigger modes."""
    with psycopg.connect(BASE_DSN) as administrator:
        triggers = administrator.execute(
            "SELECT tgname,tgenabled FROM pg_catalog.pg_trigger "
            "WHERE tgrelid=pg_catalog.to_regclass(%s)",
            (value.setup.schema + "." + table,),
        ).fetchall()
        administrator.execute(
            sql.SQL("ALTER TABLE {} DISABLE TRIGGER ALL").format(
                sql.Identifier(value.setup.schema, table)
            )
        )
        administrator.execute(query, parameters)
        modes = {"O": "ENABLE", "A": "ENABLE ALWAYS", "R": "ENABLE REPLICA", "D": "DISABLE"}
        for trigger, mode in triggers:
            administrator.execute(
                sql.SQL("ALTER TABLE {} {} TRIGGER {}").format(
                    sql.Identifier(value.setup.schema, table),
                    sql.SQL(modes[mode]),
                    sql.Identifier(trigger),
                )
            )
        administrator.commit()


def test_fake_history_signer_cannot_supply_reproposal_authority(reproposal_boundary):
    value = reproposal_boundary
    _main()._winner(value)
    before = _main()._read_all(value)

    class ForgedSigner:
        called = False

        def sign_history_attestation(self, payload):
            self.called = True
            return b"F" * 64

    forged = ForgedSigner()
    with pytest.raises(atomic_runtime.AtomicRootProofCommitConflict):
        _call(value, history_signer=forged)
    assert forged.called is False
    _assert_unmutated(value, before)


@pytest.mark.parametrize("keyword", ["head", "history_head", "signature", "history_signature"])
def test_caller_head_and_signature_are_not_reproposal_inputs(reproposal_boundary, keyword):
    value = reproposal_boundary
    before = _main()._read_all(value)
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        _call(value, **{keyword: b"F" * 64})
    _assert_unmutated(value, before)


@pytest.mark.parametrize("changed", ["base_identity", "payload", "signature", "reserved_envelope"])
def test_changed_original_preparation_cannot_authorize_reproposal(reproposal_boundary, changed):
    value = reproposal_boundary
    original = value.secondary.preparation
    if changed == "base_identity":
        altered = replace(
            original,
            locator=replace(original.locator, root_proof=value.preparation.locator.root_proof),
        )
    elif changed == "reserved_envelope":
        altered = replace(
            original,
            reserved=replace(
                original.reserved, canonical_envelope=_flip(original.reserved.canonical_envelope)
            ),
        )
    else:
        field = "canonical_payload" if changed == "payload" else "signature"
        altered = replace(
            original,
            prepared=replace(
                original.prepared, **{field: _flip(getattr(original.prepared, field))}
            ),
        )
    before = _main()._read_all(value)
    with pytest.raises(
        atomic_runtime.AtomicRootProofCommitConflict, match="GENUINE_IMMUTABLE_PREPARATION"
    ):
        _call(value, original=altered)
    _assert_unmutated(value, before)


@pytest.mark.parametrize(
    "changed", ["original", "binding", "signed_attempt", "foreign_predecessor"]
)
def test_foreign_genuine_attempt_cannot_replace_original_proposal(reproposal_boundary, changed):
    value = reproposal_boundary
    _main()._winner(value)
    arguments = {
        "original": {"original": value.preparation},
        "binding": {"binding": value.genuine.binding},
        "signed_attempt": {"signed_attempt": value.genuine.opaque},
        "foreign_predecessor": {"predecessor": value.first_proposal},
    }[changed]
    before = _main()._read_all(value)
    with pytest.raises(atomic_runtime.AtomicRootProofCommitError):
        _call(value, **arguments)
    _assert_unmutated(value, before)


@pytest.mark.parametrize("principal", ["issuer_writer", "test_admin"])
@pytest.mark.parametrize("operation", ["INSERT", "UPDATE", "DELETE", "TRUNCATE"])
def test_runtime_and_admin_have_no_direct_proposal_dml(reproposal_boundary, principal, operation):
    value = reproposal_boundary
    before = _main()._read_all(value)
    connection = _main().atomic_tests._connection(getattr(value.setup, principal + "_role"))
    for table in ("history_proposals", "history_proposal_current", "history_reproposal_contract"):
        relation = sql.Identifier(value.setup.schema, table)
        column = "schema_version" if table == "history_reproposal_contract" else "revision"
        statements = {
            "INSERT": sql.SQL("INSERT INTO {} DEFAULT VALUES").format(relation),
            "UPDATE": sql.SQL("UPDATE {} SET {}=1").format(relation, sql.Identifier(column)),
            "DELETE": sql.SQL("DELETE FROM {}").format(relation),
            "TRUNCATE": sql.SQL("TRUNCATE {}").format(relation),
        }
        with psycopg.connect(connection.dsn) as unprivileged:
            with pytest.raises(psycopg.errors.InsufficientPrivilege):
                unprivileged.execute(statements[operation])
            unprivileged.rollback()
    _assert_unmutated(value, before)


@pytest.mark.parametrize("drift", ["acl", "function_source", "function_oid"])
def test_proposal_catalog_acl_and_function_oid_drift_fail_closed(reproposal_boundary, drift):
    from bot_core.postgresql_history_successor_reproposal_qualification import (
        HistorySuccessorReproposalQualificationError,
        qualify_history_successor_reproposal_v1,
    )

    value = reproposal_boundary
    before = _main()._read_all(value)
    signature = value.setup.schema + ".commit_history_proposal(bytea,text)"
    function = sql.SQL("{}.commit_history_proposal(bytea,text)").format(
        sql.Identifier(value.setup.schema)
    )
    renamed = "saved_reproposal_commit_" + uuid.uuid4().hex[:8]
    with psycopg.connect(BASE_DSN, autocommit=True) as administrator:
        original_oid, definition = administrator.execute(
            "SELECT p.oid,pg_catalog.pg_get_functiondef(p.oid) "
            "FROM pg_catalog.pg_proc p WHERE p.oid=pg_catalog.to_regprocedure(%s)",
            (signature,),
        ).fetchone()
        try:
            if drift == "acl":
                administrator.execute(
                    sql.SQL("GRANT EXECUTE ON FUNCTION {} TO PUBLIC").format(function)
                )
            elif drift == "function_source":
                administrator.execute(
                    definition.replace(
                        "AS $function$", "AS $function$\n-- unreviewed successor source\n", 1
                    )
                )
            else:
                administrator.execute(
                    sql.SQL("ALTER FUNCTION {} RENAME TO {}").format(
                        function, sql.Identifier(renamed)
                    )
                )
                administrator.execute(definition)
                substituted_oid = administrator.execute(
                    "SELECT pg_catalog.to_regprocedure(%s)::oid", (signature,)
                ).fetchone()[0]
                assert substituted_oid != original_oid
            with psycopg.connect(value.writer_connection.dsn) as writer:
                with pytest.raises(HistorySuccessorReproposalQualificationError):
                    qualify_history_successor_reproposal_v1(writer, value.reproposal_deployment)
            with pytest.raises(HistorySuccessorReproposalQualificationError):
                _reconstruct(value)
            with pytest.raises(
                (
                    atomic_runtime.AtomicRootProofCommitError,
                    HistorySuccessorReproposalQualificationError,
                )
            ):
                _call(value)
            with pytest.raises(atomic_runtime.AtomicRootProofCommitError):
                value.reproposal.commit(value.second_proposal)
            _assert_unmutated(value, before)
        finally:
            if drift == "acl":
                administrator.execute(
                    sql.SQL("REVOKE EXECUTE ON FUNCTION {} FROM PUBLIC").format(function)
                )
            elif drift == "function_source":
                administrator.execute(definition)
            else:
                administrator.execute(sql.SQL("DROP FUNCTION {}").format(function))
                administrator.execute(
                    sql.SQL(
                        "ALTER FUNCTION {}.{}(bytea,text) RENAME TO commit_history_proposal"
                    ).format(sql.Identifier(value.setup.schema), sql.Identifier(renamed))
                )
                assert administrator.execute(
                    "SELECT pg_catalog.to_regprocedure(%s)::oid", (signature,)
                ).fetchone() == (original_oid,)
    _assert_unmutated(value, before)


@pytest.mark.parametrize("identity_field", ["system_identifier", "database_oid"])
def test_reproposal_rejects_changed_physical_database_pin(reproposal_boundary, identity_field):
    from bot_core.postgresql_history_successor_reproposal import (
        PostgreSQLHistorySuccessorReproposalV1,
    )
    from bot_core.postgresql_history_successor_reproposal_qualification import (
        HistorySuccessorReproposalQualificationError,
    )

    value = reproposal_boundary
    deployment = value.reproposal_deployment
    atomic = deployment.atomic_deployment
    identity = replace(
        atomic.identity, **{identity_field: getattr(atomic.identity, identity_field) + 1}
    )
    altered = replace(
        deployment,
        lifecycle_deployment=replace(
            deployment.lifecycle_deployment, atomic_deployment=replace(atomic, identity=identity)
        ),
        preparation_deployment=replace(deployment.preparation_deployment, identity=identity),
    )
    before = _main()._read_all(value)
    with pytest.raises(atomic_runtime.AtomicRootProofCommitUnavailable):
        PostgreSQLHistorySuccessorReproposalV1(altered)
    with pytest.raises(HistorySuccessorReproposalQualificationError):
        _reconstruct(value, expected_identity=identity)
    _assert_unmutated(value, before)


def test_reproposal_rejects_actual_other_physical_database(reproposal_boundary):
    from bot_core.postgresql_history_successor_reproposal import (
        PostgreSQLHistorySuccessorReproposalV1,
    )
    from bot_core.postgresql_history_successor_reproposal_qualification import (
        HistorySuccessorReproposalQualificationError,
    )

    value = reproposal_boundary
    name = "reproposal_other_" + uuid.uuid4().hex[:12]
    deployment = value.reproposal_deployment
    atomic = deployment.atomic_deployment
    before = _main()._read_all(value)
    with psycopg.connect(BASE_DSN, autocommit=True) as administrator:
        administrator.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(name)))
        try:
            other = PostgreSQLConnectionConfig(
                make_conninfo(value.writer_connection.dsn, dbname=name)
            )
            with psycopg.connect(other.dsn) as observer:
                observed_oid = observer.execute(
                    "SELECT oid FROM pg_catalog.pg_database WHERE datname=pg_catalog.current_database()"
                ).fetchone()[0]
                assert observed_oid != atomic.identity.database_oid
            altered = replace(
                deployment,
                lifecycle_deployment=replace(
                    deployment.lifecycle_deployment,
                    atomic_deployment=replace(atomic, writer_connection=other),
                ),
            )
            with pytest.raises(atomic_runtime.AtomicRootProofCommitUnavailable):
                PostgreSQLHistorySuccessorReproposalV1(altered)
            from bot_core.postgresql_history_successor_reproposal_qualification import (
                requalify_existing_history_successor_reproposal_v1,
            )

            with pytest.raises(HistorySuccessorReproposalQualificationError):
                requalify_existing_history_successor_reproposal_v1(
                    PostgreSQLConnectionConfig(make_conninfo(BASE_DSN, dbname=name)),
                    value.setup,
                    other,
                    _main().atomic_tests._connection(value.setup.test_admin_role),
                    expected_identity=atomic.identity,
                )
            _assert_unmutated(value, before)
        finally:
            administrator.execute(
                sql.SQL("DROP DATABASE {} WITH (FORCE)").format(sql.Identifier(name))
            )


def test_direct_role_oid_substitution_refuses_qualified_reproposal(reproposal_boundary):
    from bot_core.postgresql_history_successor_reproposal import (
        PostgreSQLHistorySuccessorReproposalV1,
    )
    from bot_core.postgresql_history_successor_reproposal_qualification import (
        HistorySuccessorReproposalQualificationError,
        qualify_history_successor_reproposal_v1,
    )

    value = reproposal_boundary
    before = _main()._read_all(value)
    role = value.setup.issuer_writer_role
    saved = "saved_reproposal_writer_" + uuid.uuid4().hex[:12]
    substitute_created = False
    with psycopg.connect(BASE_DSN, autocommit=True) as administrator:
        original_oid = administrator.execute(
            "SELECT oid FROM pg_catalog.pg_roles WHERE rolname=%s", (role,)
        ).fetchone()[0]
        administrator.execute(
            sql.SQL("ALTER ROLE {} RENAME TO {}").format(
                sql.Identifier(role), sql.Identifier(saved)
            )
        )
        try:
            administrator.execute(
                sql.SQL(
                    "CREATE ROLE {} LOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE NOINHERIT NOREPLICATION NOBYPASSRLS"
                ).format(sql.Identifier(role))
            )
            substitute_created = True
            assert (
                administrator.execute(
                    "SELECT oid FROM pg_catalog.pg_roles WHERE rolname=%s", (role,)
                ).fetchone()[0]
                != original_oid
            )
            with pytest.raises(atomic_runtime.AtomicRootProofCommitUnavailable):
                PostgreSQLHistorySuccessorReproposalV1(value.reproposal_deployment)
            with psycopg.connect(value.writer_connection.dsn) as substituted:
                with pytest.raises(HistorySuccessorReproposalQualificationError):
                    qualify_history_successor_reproposal_v1(
                        substituted, value.reproposal_deployment
                    )
            with pytest.raises(HistorySuccessorReproposalQualificationError):
                _reconstruct(value)
            with pytest.raises(atomic_runtime.AtomicRootProofCommitError):
                value.reproposal.commit(value.second_proposal)
            _assert_unmutated(value, before)
        finally:
            if substitute_created:
                administrator.execute(sql.SQL("DROP ROLE {}").format(sql.Identifier(role)))
            administrator.execute(
                sql.SQL("ALTER ROLE {} RENAME TO {}").format(
                    sql.Identifier(saved), sql.Identifier(role)
                )
            )
            assert administrator.execute(
                "SELECT oid FROM pg_catalog.pg_roles WHERE rolname=%s", (role,)
            ).fetchone() == (original_oid,)
    _assert_unmutated(value, before)


def test_fresh_process_reconstructs_reviewed_deployment_and_exact_decisions(reproposal_boundary):
    value = reproposal_boundary
    first = _main()._winner(value)
    replacement = _main()._repropose(value)
    second = value.reproposal.commit(replacement)
    before = _main()._assert_complete(value, first, second)
    deployment = value.reproposal_deployment
    request = dict(
        bootstrap=BASE_DSN,
        writer=value.writer_connection.dsn,
        admin=_main().atomic_tests._connection(value.setup.test_admin_role).dsn,
        provisioning=asdict(value.setup),
        identity=asdict(deployment.atomic_deployment.identity),
        locators=[
            {
                name: getattr(proposal.locator, name).hex()
                for name in proposal.locator.__dataclass_fields__
            }
            for proposal in (value.first_proposal, replacement)
        ],
    )
    code = """
import json,sys
from bot_core.postgresql_atomic_root_proof_commit import AtomicRootProofCommitLocator
from bot_core.postgresql_atomic_root_proof_qualification import PostgreSQLAtomicRootProofPhysicalIdentity
from bot_core.postgresql_atomic_root_proof_schema import PostgreSQLAtomicRootProofProvisioning
from bot_core.postgresql_entitlement_registry import PostgreSQLConnectionConfig
from bot_core.postgresql_history_successor_reproposal import PostgreSQLHistorySuccessorReproposalV1
from bot_core.postgresql_history_successor_reproposal_qualification import requalify_existing_history_successor_reproposal_v1
request=json.load(sys.stdin)
admin=PostgreSQLConnectionConfig(request['admin'])
deployment=requalify_existing_history_successor_reproposal_v1(
 PostgreSQLConnectionConfig(request['bootstrap']),
 PostgreSQLAtomicRootProofProvisioning(**request['provisioning']),
 PostgreSQLConnectionConfig(request['writer']),admin,
 expected_identity=PostgreSQLAtomicRootProofPhysicalIdentity(**request['identity']))
client=PostgreSQLHistorySuccessorReproposalV1(deployment,admin_connection=admin)
results=[]
for material in request['locators']:
 locator=AtomicRootProofCommitLocator(**{name:bytes.fromhex(value) for name,value in material.items()})
 proposal=client.read_current_proposal(locator)
 decision=client.reconcile(locator)
 assert client.commit(proposal)==decision
 results.append({'candidate_id':proposal.candidate_id,'revision':proposal.revision,
  'payload':decision.payload.hex(),'signature':decision.signature.hex(),'envelope':decision.envelope.hex()})
print(json.dumps(results))
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        input=json.dumps(request),
        text=True,
        capture_output=True,
        check=True,
        timeout=120,
    )
    assert json.loads(completed.stdout) == [
        dict(
            candidate_id=proposal.candidate_id,
            revision=proposal.revision,
            payload=decision.payload.hex(),
            signature=decision.signature.hex(),
            envelope=decision.envelope.hex(),
        )
        for proposal, decision in ((value.first_proposal, first), (replacement, second))
    ]
    _assert_unmutated(value, before)


@pytest.mark.parametrize("fence", ["root_key", "history_key", "trust_key"])
@pytest.mark.parametrize("state", ["PENDING_CLOSED", "VERIFY_ONLY", "REVOKED"])
@pytest.mark.parametrize("operation", ["repropose", "commit"])
def test_closed_fences_refuse_proposal_and_atomic_commit(
    reproposal_boundary, fence, state, operation
):
    value = reproposal_boundary
    if operation == "repropose":
        _main()._winner(value)
    _change_fixture_row(
        value,
        "fences",
        sql.SQL("UPDATE {}.fences SET state=%s WHERE identity=%s").format(
            sql.Identifier(value.setup.schema)
        ),
        (state, getattr(value.fences, fence)),
    )
    before = _main()._read_all(value)
    with pytest.raises(atomic_runtime.AtomicRootProofCommitError):
        if operation == "repropose":
            _call(value)
        else:
            value.reproposal.commit(value.second_proposal)
    _assert_unmutated(value, before)


@pytest.mark.parametrize("fence", ["root_key", "history_key"])
@pytest.mark.parametrize("operation", ["repropose", "commit"])
def test_missing_signer_enrollment_refuses_reproposal_and_commit(
    reproposal_boundary, fence, operation
):
    value = reproposal_boundary
    if operation == "repropose":
        _main()._winner(value)
    _change_fixture_row(
        value,
        "lifecycle_credentials",
        sql.SQL("DELETE FROM {}.lifecycle_credentials WHERE fence_key=%s").format(
            sql.Identifier(value.setup.schema)
        ),
        (getattr(value.fences, fence),),
    )
    before = _main()._read_all(value)
    with pytest.raises(atomic_runtime.AtomicRootProofCommitError):
        if operation == "repropose":
            _call(value)
        else:
            value.reproposal.commit(value.second_proposal)
    _assert_unmutated(value, before)

"""Genuine V1 signed evidence survives explicit successor-proposal migration."""

from __future__ import annotations

import uuid
from types import SimpleNamespace

import psycopg
import pytest
from psycopg import sql

from bot_core import (
    postgresql_atomic_root_proof_commit as atomic_runtime,
    postgresql_custody_lifecycle as lifecycle,
    postgresql_root_proof_preparation_v2 as preparation_runtime,
)
from bot_core.licensing import (
    cha_root_proof_attempt_reservation as attempt_boundary,
    cha_root_proof_signed_attempt as signed_boundary,
)
from bot_core.licensing.canonical import parse_canonical
from bot_core.postgresql_atomic_root_proof_migration import (
    PostgreSQLAtomicRootProofLegacyConnections,
)
from bot_core.postgresql_custody_lifecycle_qualification import migrate_custody_lifecycle_v2
from bot_core.postgresql_entitlement_registry import PostgreSQLConnectionConfig
from bot_core.postgresql_history_successor_reproposal import PostgreSQLHistorySuccessorReproposalV1
from bot_core.postgresql_history_successor_reproposal_qualification import (
    migrate_history_successor_reproposal_v1,
)
from bot_core.postgresql_root_proof_issuance_authority import PostgreSQLRootProofIssuanceAuthority
from bot_core.postgresql_root_proof_issuance_authority_v2 import (
    PostgreSQLRootProofIssuanceAuthorityV2,
)
from bot_core.postgresql_root_proof_preparation_qualification import (
    migrate_root_proof_preparation_v2,
)
from bot_core.root_proof_issuer_substrate import ProviderRole
from deployment import (
    windows_production_cha_account_reservation as installed_account,
    windows_production_cha_operation as installed_cha,
    windows_production_lppi_operation as installed_lppi,
)
from tests.security import (
    test_postgresql_atomic_root_proof_commit as atomic_tests,
    test_root_proof_issuer_runtime as preflight_tests,
)
from tests.security.test_postgresql_checkpoint_authority import BASE_DSN

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
module_atomic_commit_boundary = atomic_tests.module_atomic_commit_boundary


def _original_opaque_sources(signed):
    snapshot = signed_boundary._SIGNED[signed]
    authorization = attempt_boundary._AUTHORIZATIONS[snapshot.authorization]
    assert type(authorization.provider) is PostgreSQLRootProofIssuanceAuthority
    assert (
        parse_canonical(authorization.evidence_raw)["providers"]["entitlement"]["capabilities"][
            "compare_and_swap"
        ]
        is True
    )
    ports = tuple(
        (port, port.identity, port.capabilities, port._connection, port._schema)
        for port in (
            authorization.provider.entitlement_registry,
            authorization.provider.requester_registry,
            authorization.provider.claimant_registry,
        )
    )
    return snapshot, authorization, ports


def _v2_ports(value, original, deployment, connections):
    authority = PostgreSQLRootProofIssuanceAuthorityV2(
        connections.entitlement_runtime,
        connections.requester_runtime,
        connections.claimant_runtime,
        deployment=deployment,
        entitlement_subject=original.bound_state.subject,
    )
    intent = preparation_runtime.PostgreSQLRootProofSigningIntentV2(
        connections.intent_writer,
        deployment=deployment,
        schema=value.intent_setup.schema,
        scope=value.scope,
        verification_authority=value.signer,
        issuer_authority=authority,
    )
    retention = preparation_runtime.PostgreSQLRootProofRetentionV2(
        connections.retention_writer,
        deployment=deployment,
        schema=value.retention_setup.schema,
        scope=value.scope,
        verification_authority=value.signer,
        signing_intent=intent,
    )
    return SimpleNamespace(authority=authority, intent=intent, retention=retention)


@pytest.fixture(scope="module")
def legacy_reproposal_boundary(module_atomic_commit_boundary):
    value = module_atomic_commit_boundary
    pins = atomic_runtime._trusted_preparation(value.preparation).pins
    connections = PostgreSQLAtomicRootProofLegacyConnections(
        pins[0][0],
        pins[1][0],
        pins[2][0],
        pins[3][0],
        requester_runtime=pins[4][0],
        claimant_runtime=pins[5][0],
    )
    preparation_schema = "legacy_reproposal_" + uuid.uuid4().hex[:12]
    bindings = tuple(
        lifecycle.prepare_custody_lifecycle_binding(
            value.atomic_deployment.identity,
            value.directory,
            security=value.security,
            role=role,
            fence_key=key,
        )
        for role, key in (
            (ProviderRole.ROOT_PROOF_SIGNING, value.fences.root_key),
            (ProviderRole.HISTORY_ATTESTATION_SIGNING, value.fences.history_key),
        )
    )
    original_sources = (
        _original_opaque_sources(value.genuine.opaque),
        _original_opaque_sources(value.secondary.signed),
    )
    before = atomic_tests._fixture_data(value)
    signed_dumps = tuple(
        preflight_tests._database_dump(path) for path in (value.genuine.path, value.secondary.path)
    )
    lifecycle_deployment = migrate_custody_lifecycle_v2(
        PostgreSQLConnectionConfig(BASE_DSN),
        value.atomic_deployment,
        atomic_tests._connection(value.setup.test_admin_role),
        bindings,
    )
    preparation, lifecycle_deployment = migrate_root_proof_preparation_v2(
        PostgreSQLConnectionConfig(BASE_DSN),
        lifecycle_deployment,
        preparation_schema=preparation_schema,
        connections=connections,
    )
    try:
        deployment = migrate_history_successor_reproposal_v1(
            PostgreSQLConnectionConfig(BASE_DSN),
            lifecycle_deployment,
            preparation_deployment=preparation,
        )
        assert atomic_tests._fixture_data(value) == before
        assert deployment.preparation_deployment is not None
        primary_ports = _v2_ports(
            value, value.preparation, deployment.preparation_deployment, connections
        )
        secondary_ports = _v2_ports(
            value, value.secondary.preparation, deployment.preparation_deployment, connections
        )
        history = preparation_runtime.PostgreSQLAuthenticatedIssuerHistoryV2(
            connections.history_runtime,
            deployment=deployment.preparation_deployment,
            schema=value.history_boundary.history_setup.schema,
            stream=value.history_boundary.stream,
        )
        client = PostgreSQLHistorySuccessorReproposalV1(
            deployment, admin_connection=atomic_tests._connection(value.setup.test_admin_role)
        )
        yield SimpleNamespace(
            value=value,
            client=client,
            history=history,
            primary_ports=primary_ports,
            secondary_ports=secondary_ports,
            original_sources=original_sources,
            signed_dumps=signed_dumps,
            before=before,
        )
    finally:
        with psycopg.connect(BASE_DSN, autocommit=True) as administrator:
            administrator.execute(
                sql.SQL("DROP SCHEMA IF EXISTS {} CASCADE").format(
                    sql.Identifier(preparation_schema)
                )
            )


def test_real_legacy_original_stale_successor_reproposes_without_resigning(
    legacy_reproposal_boundary,
):
    fixture = legacy_reproposal_boundary
    value, client = fixture.value, fixture.client
    first = client.read_current_proposal(value.preparation)
    stale = client.read_current_proposal(value.secondary.preparation)
    assert first is not None and stale is not None
    assert first.revision == stale.revision == 1
    winner = client.commit(first)
    before_stale = atomic_tests._fixture_data(value)
    with pytest.raises(atomic_runtime.AtomicRootProofCommitConflict):
        client.commit(stale)
    assert atomic_tests._fixture_data(value) == before_stale
    with pytest.raises(atomic_runtime.AtomicRootProofCommitIndeterminate):
        client.reconcile(stale.locator)
    candidate = value.secondary
    ports = fixture.secondary_ports
    with pytest.MonkeyPatch.context() as selected:
        selected.setattr(attempt_boundary, "_attempt_store_path", lambda: candidate.path)
        for module, filename in (
            (installed_lppi, "initial-authenticated-operation.json"),
            (installed_cha, "initial-cha-logical-operation.json"),
            (installed_account, "initial-cha-account-reservation.json"),
        ):
            selected.setattr(
                module, "_state_path", lambda filename=filename: candidate.namespace / filename
            )
        replacement = client.repropose(
            candidate.preparation,
            candidate.binding,
            candidate.signed,
            signing_intent=ports.intent,
            retention=ports.retention,
            issuer_authority=ports.authority,
            history=fixture.history,
            history_signer=value.history_boundary.signer,
            fences=value.fences,
            predecessor=stale,
        )
    assert replacement.revision == 2
    assert replacement.predecessor_candidate_id == stale.candidate_id
    recovered = client.commit(replacement)
    assert (recovered.payload, recovered.signature, recovered.envelope) == (
        candidate.prepared.canonical_payload,
        candidate.prepared.signature,
        candidate.prepared.canonical_envelope,
    )
    assert client.reconcile(candidate.preparation.locator) == recovered
    assert client.reconcile(value.preparation.locator) == winner
    records = fixture.history.authenticated_snapshot(value.history_boundary.signer).records
    assert len(records) == 2
    assert records[0].event_identity == value.preparation.history_record.event_identity
    assert records[1].event_identity == candidate.preparation.history_record.event_identity
    assert records[1].event_digest == candidate.preparation.history_record.event_digest
    assert (
        records[1].canonical_event_payload
        == candidate.preparation.history_record.canonical_event_payload
    )
    assert records[1].predecessor_authenticated_digest == records[0].authenticated_digest
    after = atomic_tests._fixture_data(value)
    for pair in (
        (value.intent_setup.schema, "intents"),
        (value.retention_setup.schema, "prepared"),
        (value.setup.schema, "approved_commits"),
    ):
        assert after[pair] == fixture.before[pair]
    assert (
        tuple(preflight_tests._database_dump(path) for path in (value.genuine.path, candidate.path))
        == fixture.signed_dumps
    )
    for signed, (signed_snapshot, authorization_snapshot, original_ports) in zip(
        (value.genuine.opaque, candidate.signed), fixture.original_sources, strict=True
    ):
        assert signed_boundary._SIGNED[signed] is signed_snapshot
        assert (
            attempt_boundary._AUTHORIZATIONS[signed_snapshot.authorization]
            is authorization_snapshot
        )
        assert type(authorization_snapshot.provider) is PostgreSQLRootProofIssuanceAuthority
        for port, identity, capabilities, connection, schema in original_ports:
            assert (port.identity, port.capabilities, port._connection, port._schema) == (
                identity,
                capabilities,
                connection,
                schema,
            )
        assert (
            parse_canonical(authorization_snapshot.evidence_raw)["providers"]["entitlement"][
                "capabilities"
            ]["compare_and_swap"]
            is True
        )
    assert ports.authority.entitlement_registry.capabilities.compare_and_swap is False

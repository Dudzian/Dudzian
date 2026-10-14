"""Fresh V2 preparation, genuine PostgreSQL bytes and isolated authority effects.

The fixture migrates empty signing-intent/retention stores before its first
RESERVED or STAGED. Real custody, signed CHA evidence and PostgreSQL transactions
remain authoritative. Administrative fixture restoration is test teardown only.
"""

from __future__ import annotations

import multiprocessing
import os
import socket
import time
import uuid
from dataclasses import replace
from datetime import datetime, timezone
from types import SimpleNamespace

import psycopg
import pytest
from psycopg import sql
from psycopg.conninfo import make_conninfo
from psycopg.types.json import Jsonb

from bot_core import (
    local_signing_custody as custody,
    postgresql_atomic_root_proof_commit as atomic_runtime,
    postgresql_atomic_root_proof_migration as atomic_migration,
    postgresql_atomic_root_proof_schema as atomic_schema,
    postgresql_custody_lifecycle as lifecycle,
    postgresql_root_proof_preparation_qualification as preparation_qualification,
    postgresql_root_proof_preparation_v2 as preparation_runtime,
)
from bot_core.authenticated_issuer_history import _credential_material, attest_head
from bot_core.entitlement_registry_contract import (
    AdminOutcome,
    EntitlementIdentity,
    ProvisionEntitlementRequest,
    RegistrySubject,
)
from bot_core.licensing import (
    cha_root_proof_attempt_reservation as attempt_boundary,
    cha_root_proof_signed_attempt as signed_boundary,
)
from bot_core.licensing.canonical import canonical_json_bytes, parse_canonical
from bot_core.postgresql_custody_lifecycle_qualification import migrate_custody_lifecycle_v2
from bot_core.postgresql_entitlement_registry import PostgreSQLConnectionConfig
from bot_core.postgresql_root_proof_issuance_authority_v2 import (
    PostgreSQLRootProofIssuanceAuthorityV2,
)
from bot_core.postgresql_root_proof_retention import (
    ROOT_PROOF_SIGNING_DOMAIN,
    RootProofRetentionConflict,
    RootProofRetentionIndeterminate,
    RootProofRetentionUnavailable,
    _scope_data,
    _snapshot_data,
)
from bot_core.postgresql_root_proof_signing_intent import (
    RootProofSigningIntentConflict,
    RootProofSigningIntentError,
    RootProofSigningIntentIndeterminate,
    RootProofSigningIntentUnavailable,
)
from bot_core.root_proof_issuer_substrate import ProviderRole
from bot_core.uuid7 import mint_uuid7, reservation_epoch_milliseconds
from deployment import (
    windows_production_cha_account_reservation as installed_account,
    windows_production_cha_operation as installed_cha,
    windows_production_lppi_operation as installed_lppi,
)
from tests.security import (
    test_postgresql_atomic_root_proof_commit as atomic_tests,
    test_root_proof_atomic_composition_integration as existing_atomic,
    test_root_proof_issuer_runtime as preflight_tests,
)
from tests.security._local_signing_platform import requires_native_custody_locking
from tests.security.test_postgresql_checkpoint_authority import BASE_DSN

pytestmark = [pytest.mark.external_postgresql, requires_native_custody_locking]
requires_fork = pytest.mark.skipif(
    "fork" not in multiprocessing.get_all_start_methods(),
    reason="independent processes inherit genuine test-only capabilities",
)

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


def _read_all(value):
    """A fresh physical backend observes every preparation/issuance authority."""
    tables = atomic_tests._data_tables(value) + [
        (value.setup.schema, "lifecycle_credentials"),
        (value.setup.schema, "lifecycle_operations"),
        (value.setup.schema, "lifecycle_completions"),
        (value.checkpoint._schema, "checkpoints"),
        (value.checkpoint._schema, "entries"),
    ]
    with psycopg.connect(BASE_DSN) as observer:
        return {
            pair: observer.execute(
                sql.SQL("SELECT * FROM {} ORDER BY 1").format(sql.Identifier(*pair))
            ).fetchall()
            for pair in tables
        }


def _restore_fixture(value, baseline=None):
    """Privileged teardown restores a strictly isolated fixture after assertions."""
    baseline = value.baseline if baseline is None else baseline
    if _read_all(value) == baseline:
        return
    pairs = list(baseline)
    with psycopg.connect(BASE_DSN) as administrator:
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
        for pair in pairs:
            for row in baseline[pair]:
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


@pytest.fixture(scope="module")
def module_v2_boundary(genuine_attempt, lineage_path, lineage_monkeypatch):
    path = lineage_path / "fresh-v2-preparation"
    path.mkdir()
    source = existing_atomic.atomic_boundary.__wrapped__(genuine_attempt, path, lineage_monkeypatch)
    value = next(source)
    token = uuid.uuid4().hex[:12]
    installed = False
    preparation_schema = f"prpv2_{token}"
    try:
        value.legacy_intent, value.legacy_retention = value.intent, value.retention
        value.legacy_authority = value.genuine.authority
        value.legacy_history = value.history
        state = value.genuine.authority.entitlement_registry.authoritative_state(
            value.genuine.subject
        ).state
        value.secondary_subject = RegistrySubject(
            "v2-unbound-close-control-" + token,
            state.subject.environment,
            state.subject.trust_domain,
        )
        secondary_identity = EntitlementIdentity(
            mint_uuid7("ent_", reservation_epoch_milliseconds(datetime.now(timezone.utc))),
            1,
            state.identity.environment,
            state.identity.trust_domain,
            state.identity.product_scope,
        )
        secondary = value.genuine.entitlement_admin.provision_entitlement(
            ProvisionEntitlementRequest(
                value.secondary_subject, secondary_identity, state.provenance
            )
        )
        assert secondary.outcome is AdminOutcome.COMMITTED
        value.setup = atomic_schema.PostgreSQLAtomicRootProofProvisioning(
            f"pv2_{token}",
            f"pv2o_{token}",
            f"pv2w_{token}",
            f"pv2a_{token}",
            value.genuine.entitlement_admin._schema,
            value.intent_setup.schema,
            value.retention_setup.schema,
            value.history_boundary.history_setup.schema,
            value.auth.trust_domain,
            f"isolated-fresh-v2-{token}",
            requester_schema=value.genuine.pair.requester._schema,
            claimant_schema=value.genuine.pair.claimant._schema,
        )
        value.writer_connection = atomic_tests._connection(value.setup.issuer_writer_role)
        connections = atomic_migration.PostgreSQLAtomicRootProofLegacyConnections(
            value.genuine.authority.entitlement_registry._connection,
            existing_atomic._provider_pin(value.intent).connection,
            existing_atomic._provider_pin(value.retention).connection,
            value.history._connection,
            requester_runtime=value.genuine.pair.requester._connection,
            claimant_runtime=value.genuine.pair.claimant._connection,
        )
        value.atomic_deployment = atomic_migration.administer_atomic_root_proof_v1(
            PostgreSQLConnectionConfig(BASE_DSN),
            value.writer_connection,
            value.setup,
            legacy_connections=connections,
        )
        installed = True
        replacement_directory = path / "replacement-root-custody"
        replacement_directory.mkdir(mode=0o700)
        replacement_administrator = custody.NativeKeyringSigningSecretAdministrator(
            index_path=replacement_directory / "keyring-index.json",
            security=value.security,
            role=ProviderRole.ROOT_PROOF_SIGNING,
        )
        custody.provision_local_signing_authority(
            replacement_directory,
            replacement_administrator,
            security=value.security,
            role=ProviderRole.ROOT_PROOF_SIGNING,
        )
        value.replacement_signer = custody.LocalRootProofSigningProvider(
            replacement_directory,
            keyring_index_path=replacement_directory / "keyring-index.json",
            security=value.security,
        )
        replacement_key = atomic_runtime._fence_key(
            "ROOT_PROOF_SIGNING",
            canonical_json_bytes(_snapshot_data(value.replacement_signer.credential_snapshot())),
        )
        atomic_tests._call_sql(
            value, "provision_fence", (replacement_key, "ROOT_PROOF_SIGNING", 1), admin=True
        )
        atomic_tests._call_sql(
            value, "transition_fence", (replacement_key, 1, "ACTIVE", 1), admin=True
        )
        snapshot = value.signer.credential_snapshot()
        history_snapshot = value.history_boundary.signer.credential_snapshot()
        value.fences = atomic_runtime.AtomicRootProofCommitFences(
            atomic_runtime._fence_key(
                "ROOT_PROOF_SIGNING", canonical_json_bytes(_snapshot_data(snapshot))
            ),
            snapshot.lifecycle_generation,
            atomic_runtime._fence_key(
                "HISTORY_ATTESTATION_SIGNING",
                canonical_json_bytes(_credential_material(history_snapshot.credential_identity)),
            ),
            history_snapshot.lifecycle_generation,
            atomic_runtime._fence_key(
                "TRUST_ACTIVATION", canonical_json_bytes(_scope_data(value.scope))
            ),
            1,
        )
        for role, key in (
            ("ROOT_PROOF_SIGNING", value.fences.root_key),
            ("HISTORY_ATTESTATION_SIGNING", value.fences.history_key),
            ("TRUST_ACTIVATION", value.fences.trust_key),
        ):
            atomic_tests._call_sql(value, "provision_fence", (key, role, 1), admin=True)
            atomic_tests._call_sql(value, "transition_fence", (key, 1, "ACTIVE", 1), admin=True)
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
        bindings += (
            lifecycle.prepare_custody_lifecycle_binding(
                value.atomic_deployment.identity,
                replacement_directory,
                security=value.security,
                role=ProviderRole.ROOT_PROOF_SIGNING,
                fence_key=replacement_key,
            ),
        )
        lifecycle_deployment = migrate_custody_lifecycle_v2(
            PostgreSQLConnectionConfig(BASE_DSN),
            value.atomic_deployment,
            atomic_tests._connection(value.setup.test_admin_role),
            bindings,
        )
        value.preparation_deployment, value.lifecycle_deployment = (
            preparation_qualification.migrate_root_proof_preparation_v2(
                PostgreSQLConnectionConfig(BASE_DSN),
                lifecycle_deployment,
                preparation_schema=preparation_schema,
                connections=connections,
            )
        )
        value.atomic_deployment = value.lifecycle_deployment.atomic_deployment
        value.preparation_connections = connections
        legacy_attempt_path = value.genuine.path
        legacy_attempt_dump = preflight_tests._database_dump(legacy_attempt_path)
        value.genuine.authority = PostgreSQLRootProofIssuanceAuthorityV2(
            connections.entitlement_runtime,
            connections.requester_runtime,
            connections.claimant_runtime,
            deployment=value.preparation_deployment,
            entitlement_subject=value.genuine.subject,
        )
        production_selector = attempt_boundary._issuance_authority_provider
        value.genuine.path = legacy_attempt_path.with_name("fresh-v2-signed-attempt.sqlite3")
        lineage_monkeypatch.setattr(
            attempt_boundary, "_attempt_store_path", lambda: value.genuine.path
        )
        value.genuine.authorization = attempt_boundary.resolve_root_proof_issuance_authorization_v2(
            value.genuine.binding, value.genuine.authority
        )
        value.auth = value.genuine.authorization.authorization
        value.genuine.reservation = attempt_boundary.reserve_root_proof_issuance_attempt(
            value.genuine.binding, value.genuine.authorization
        )
        value.genuine.opaque = signed_boundary.sign_root_proof_issuance_attempt(
            value.genuine.reservation
        )
        value.genuine.current, value.genuine.retained = signed_boundary._signed_snapshot(
            value.genuine.opaque
        )
        assert preflight_tests._database_dump(legacy_attempt_path) == legacy_attempt_dump
        value.signed_attempt_dump = preflight_tests._database_dump(value.genuine.path)
        assert attempt_boundary._issuance_authority_provider is production_selector
        value.history = preparation_runtime.PostgreSQLAuthenticatedIssuerHistoryV2(
            connections.history_runtime,
            deployment=value.preparation_deployment,
            schema=value.history_boundary.history_setup.schema,
            stream=value.history_boundary.stream,
        )
        value.intent = preparation_runtime.PostgreSQLRootProofSigningIntentV2(
            connections.intent_writer,
            deployment=value.preparation_deployment,
            schema=value.intent_setup.schema,
            scope=value.scope,
            verification_authority=value.signer,
            issuer_authority=value.genuine.authority,
        )
        value.retention = preparation_runtime.PostgreSQLRootProofRetentionV2(
            connections.retention_writer,
            deployment=value.preparation_deployment,
            schema=value.retention_setup.schema,
            scope=value.scope,
            verification_authority=value.signer,
            signing_intent=value.intent,
        )
        value.commit = atomic_runtime.PostgreSQLAtomicRootProofCommit(
            value.atomic_deployment,
            admin_connection=atomic_tests._connection(value.setup.test_admin_role),
        )
        value.baseline = _read_all(value)
        # This assertion is the evidence distinguishing the fixture from the
        # existing pre-migration atomic candidate harness.
        for pair in (
            (value.intent_setup.schema, "intents"),
            (value.retention_setup.schema, "prepared"),
            (value.setup.schema, "approved_commits"),
            (value.setup.schema, "decisions"),
        ):
            assert value.baseline[pair] == []
        yield value
    finally:
        if installed:
            with psycopg.connect(BASE_DSN, autocommit=True) as administrator:
                for schema in (preparation_schema, value.setup.schema):
                    administrator.execute(
                        sql.SQL("DROP SCHEMA IF EXISTS {} CASCADE").format(sql.Identifier(schema))
                    )
                for role in (
                    value.setup.issuer_writer_role,
                    value.setup.test_admin_role,
                    value.setup.schema_owner_role,
                ):
                    administrator.execute(
                        sql.SQL("DROP OWNED BY {} CASCADE").format(sql.Identifier(role))
                    )
                    administrator.execute(
                        sql.SQL("DROP ROLE IF EXISTS {}").format(sql.Identifier(role))
                    )
        source.close()


@pytest.fixture
def v2_boundary(module_v2_boundary):
    value = module_v2_boundary
    assert _read_all(value) == value.baseline
    try:
        yield value
    finally:
        _restore_fixture(value)


def _genuine_preparation_baselines(value):
    """Generate authentic rows once; negative cases restore those exact bytes."""
    if not hasattr(value, "negative_preparation"):
        assert _read_all(value) == value.baseline
        try:
            reserved, prepared = _stage(value)
            staged_baseline = _read_all(value)
            _assert_only_preparation_changed(value, value.baseline, staged_baseline)
            assert staged_baseline[(value.setup.schema, "approved_commits")] == []
            assert staged_baseline[(value.setup.schema, "decisions")] == []
            reserved_baseline = dict(staged_baseline)
            reserved_baseline[(value.retention_setup.schema, "prepared")] = []
            value.negative_preparation = reserved, prepared, reserved_baseline, staged_baseline
        finally:
            _restore_fixture(value)
    return value.negative_preparation


@pytest.fixture
def prepared_v2_boundary(module_v2_boundary):
    value = module_v2_boundary
    assert _read_all(value) == value.baseline
    reserved, prepared, _, baseline = _genuine_preparation_baselines(value)
    _restore_fixture(value, baseline)
    try:
        yield value, reserved, prepared
    finally:
        _restore_fixture(value)


@pytest.fixture
def reserved_v2_boundary(module_v2_boundary):
    value = module_v2_boundary
    assert _read_all(value) == value.baseline
    reserved, _, baseline, _ = _genuine_preparation_baselines(value)
    _restore_fixture(value, baseline)
    try:
        yield value, reserved
    finally:
        _restore_fixture(value)


def _reserve(value):
    return value.intent.reserve(value.genuine.binding, value.genuine.opaque)


def _stage(value, reserved=None):
    reserved = _reserve(value) if reserved is None else reserved
    signature = value.signer.sign_root_proof(reserved.bytes_to_sign)
    prepared = value.retention.retain_prepared(
        value.genuine.binding,
        value.genuine.opaque,
        canonical_payload=reserved.canonical_payload,
        signature=signature,
    )
    return reserved, prepared


def _assert_only_preparation_changed(value, before, after):
    preparation = {
        (value.intent_setup.schema, "intents"),
        (value.retention_setup.schema, "prepared"),
    }
    for pair in before:
        if pair not in preparation:
            assert after[pair] == before[pair]


def _approval(value):
    preparation = atomic_runtime.prepare_atomic_root_proof_commit(
        value.genuine.binding,
        value.genuine.opaque,
        signing_intent=value.intent,
        retention=value.retention,
        issuer_authority=value.genuine.authority,
        history=value.history,
    )
    head = attest_head(preparation.history_record, value.history_boundary.signer)
    value.commit.approve_prepared(
        preparation, head, value.history_boundary.signer, fences=value.fences
    )
    return preparation, head


def test_fresh_reserved_staged_retries_keep_exact_original_bytes_and_no_decision(
    v2_boundary, monkeypatch
):
    value = v2_boundary
    before = _read_all(value)
    reserved = _reserve(value)
    assert reserved.state == "RESERVED" and reserved.authorizes_operations is False
    assert _read_all(value)[(value.retention_setup.schema, "prepared")] == []
    reserved_again = _reserve(value)
    assert reserved_again == reserved
    assert value.intent.read_intent(value.genuine.binding, value.genuine.opaque) == reserved
    _, prepared = _stage(value, reserved)
    assert prepared.state == "STAGED" and prepared.authorizes_operations is False
    assert prepared.canonical_payload == reserved.canonical_payload
    assert prepared.credential_snapshot == reserved.credential_snapshot
    staged = _read_all(value)
    _assert_only_preparation_changed(value, before, staged)

    def no_sign(*args, **kwargs):
        pytest.fail("exact STAGED retry invoked a signing operation")

    monkeypatch.setattr(custody.LocalRootProofSigningProvider, "sign_root_proof", no_sign)
    replay = value.retention.retain_prepared(
        value.genuine.binding,
        value.genuine.opaque,
        canonical_payload=prepared.canonical_payload,
        signature=prepared.signature,
        canonical_envelope=prepared.canonical_envelope,
    )
    assert replay == prepared
    assert replay.canonical_envelope == prepared.canonical_envelope
    assert replay.signature == prepared.signature
    assert (
        value.retention.read_prepared(
            authorization=value.auth, retained_request=value.genuine.retained
        )
        == prepared
    )
    assert _read_all(value) == staged
    assert preflight_tests._database_dump(value.genuine.path) == value.signed_attempt_dump


def test_the_new_postmigration_staged_is_approved_committed_and_replayed(v2_boundary, monkeypatch):
    value = v2_boundary
    reserved, prepared = _stage(value)
    staged = _read_all(value)
    preparation, head = _approval(value)
    approved = _read_all(value)
    for pair in approved:
        if pair != (value.setup.schema, "approved_commits"):
            assert approved[pair] == staged[pair]
    assert len(approved[(value.setup.schema, "approved_commits")]) == 1
    decision = value.commit.commit(preparation)
    assert (decision.payload, decision.signature, decision.envelope) == (
        prepared.canonical_payload,
        prepared.signature,
        prepared.canonical_envelope,
    )
    value.preparation, value.prepared, value.reserved, value.head = (
        preparation,
        prepared,
        reserved,
        head,
    )
    observed = _read_all(value)
    assert len(observed[(value.setup.schema, "decisions")]) == 1
    assert len(observed[(value.history._schema, "records")]) == 1
    assert len(observed[(value.history._schema, "attestations")]) == 1
    own_history = [
        row
        for row in observed[(value.genuine.entitlement_admin._schema, "history")]
        if row[0] == value.genuine.subject.lookup_handle
    ]
    assert len(own_history) == 2
    marker = observed[(value.setup.schema, "decisions")][0]
    assert reserved.canonical_envelope in tuple(
        bytes(part) for part in marker if isinstance(part, bytes)
    )
    committed = _read_all(value)
    monkeypatch.setattr(
        custody.LocalRootProofSigningProvider,
        "sign_root_proof",
        lambda *_: pytest.fail("committed replay attempted to sign"),
    )
    assert value.commit.commit(preparation) == decision
    assert value.commit.reconcile(preparation.locator) == decision
    assert _read_all(value) == committed


@pytest.mark.parametrize(
    "changed", ["signature", "payload", "envelope", "generation", "request_identity"]
)
def test_changed_staged_input_is_refused_without_authority_mutation(prepared_v2_boundary, changed):
    value, reserved, prepared = prepared_v2_boundary
    payload, signature, envelope = (
        prepared.canonical_payload,
        prepared.signature,
        prepared.canonical_envelope,
    )
    if changed == "signature":
        signature = bytes([signature[0] ^ 1]) + signature[1:]
    elif changed == "envelope":
        document = parse_canonical(envelope)
        document["canonical_payload_sha256"] = "0" * 64
        envelope = canonical_json_bytes(document)
    else:
        document = parse_canonical(payload)
        field = {
            "payload": "root_proof_id",
            "generation": "entitlement_generation",
            "request_identity": "issuance_attempt_id",
        }[changed]
        document[field] = (
            2
            if changed == "generation"
            else (
                "rpf_018f3e70-7b5e-7c21-8b9a-0123456789ab"
                if changed == "payload"
                else "rpa_018f3e70-7b5d-7c21-8b9a-0123456789ac"
            )
        )
        payload = canonical_json_bytes(document)
        signature = value.signer.sign_root_proof(ROOT_PROOF_SIGNING_DOMAIN + payload)
    before = _read_all(value)
    with pytest.raises(RootProofRetentionConflict):
        value.retention.retain_prepared(
            value.genuine.binding,
            value.genuine.opaque,
            canonical_payload=payload,
            signature=signature,
            canonical_envelope=envelope,
        )
    assert _read_all(value) == before


@pytest.mark.parametrize(
    "changed",
    [
        "requester_inactive",
        "claimant_inactive",
        "requester_rotation",
        "claimant_rotation",
        "entitlement_revoked",
        "entitlement_superseded",
    ],
)
def test_live_requester_claimant_and_generation_facts_are_rechecked(v2_boundary, changed):
    value = v2_boundary
    if changed.startswith(("requester", "claimant")):
        requester = changed.startswith("requester")
        administrator = (
            value.genuine.pair.requester_admin if requester else value.genuine.pair.claimant_admin
        )
        principal = "CryptoHunterAccountAuthority" if requester else value.genuine.principal
        if changed.endswith("rotation"):
            payload = administrator._provision_payload(
                "ROTATE", principal, "v2-new-credential", "v2-new-key", 2, b"N" * 32, 1
            )
        else:
            payload = {
                "operation": "TRANSITION",
                "principal_id": principal,
                "credential_id": None,
                "key_id": None,
                "key_version": None,
                "public_key_hex": None,
                "key_material_identity": None,
                "expected_revision": 1,
                "lifecycle": "VERIFY_ONLY",
            }
        # Real administrative SQL retains its original principal and typed
        # history semantics; its legacy Python catalog qualifier is closed.
        with psycopg.connect(administrator._connection.dsn) as connection:
            connection.execute("SET TRANSACTION ISOLATION LEVEL SERIALIZABLE")
            connection.execute(
                sql.SQL("SELECT {}.apply(%s)").format(sql.Identifier(administrator._schema)),
                (Jsonb(payload),),
            ).fetchone()
            connection.commit()
    else:
        preflight_tests._change_authority(value.genuine, changed)
    before = _read_all(value)
    with pytest.raises(RootProofSigningIntentError):
        _reserve(value)
    assert _read_all(value) == before


@pytest.mark.parametrize("component", ["intent", "retention"])
@pytest.mark.parametrize("principal", ["issuer_writer", "test_admin", "bootstrap"])
def test_preparation_rejects_wrong_direct_principal(v2_boundary, component, principal):
    value = v2_boundary
    connection = {
        "issuer_writer": value.writer_connection,
        "test_admin": atomic_tests._connection(value.setup.test_admin_role),
        "bootstrap": PostgreSQLConnectionConfig(BASE_DSN),
    }[principal]
    before = _read_all(value)
    if component == "intent":
        with pytest.raises(RootProofSigningIntentUnavailable):
            preparation_runtime.PostgreSQLRootProofSigningIntentV2(
                connection,
                deployment=value.preparation_deployment,
                schema=value.intent_setup.schema,
                scope=value.scope,
                verification_authority=value.signer,
                issuer_authority=value.genuine.authority,
            )
    else:
        with pytest.raises(RootProofRetentionUnavailable):
            preparation_runtime.PostgreSQLRootProofRetentionV2(
                connection,
                deployment=value.preparation_deployment,
                schema=value.retention_setup.schema,
                scope=value.scope,
                verification_authority=value.signer,
                signing_intent=value.intent,
            )
    assert _read_all(value) == before


def test_other_physical_database_identity_is_refused(v2_boundary):
    value = v2_boundary
    identity = replace(
        value.preparation_deployment.identity,
        system_identifier=value.preparation_deployment.identity.system_identifier + 1,
    )
    changed = replace(value.preparation_deployment, identity=identity)
    before = _read_all(value)
    with pytest.raises(RootProofSigningIntentUnavailable):
        preparation_runtime.PostgreSQLRootProofSigningIntentV2(
            value.preparation_connections.intent_writer,
            deployment=changed,
            schema=value.intent_setup.schema,
            scope=value.scope,
            verification_authority=value.signer,
            issuer_authority=value.genuine.authority,
        )
    assert _read_all(value) == before


def test_actual_other_database_is_refused_before_preparation(v2_boundary):
    value = v2_boundary
    name = "v2_physical_control_" + uuid.uuid4().hex[:12]
    before = _read_all(value)
    with psycopg.connect(BASE_DSN, autocommit=True) as administrator:
        administrator.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(name)))
        try:
            connection = PostgreSQLConnectionConfig(
                make_conninfo(value.preparation_connections.intent_writer.dsn, dbname=name)
            )
            with psycopg.connect(connection.dsn) as other:
                identity = other.execute(
                    "SELECT oid FROM pg_catalog.pg_database WHERE datname=pg_catalog.current_database()"
                ).fetchone()[0]
                assert identity != value.preparation_deployment.identity.database_oid
            with pytest.raises(RootProofSigningIntentUnavailable):
                preparation_runtime.PostgreSQLRootProofSigningIntentV2(
                    connection,
                    deployment=value.preparation_deployment,
                    schema=value.intent_setup.schema,
                    scope=value.scope,
                    verification_authority=value.signer,
                    issuer_authority=value.genuine.authority,
                )
            assert _read_all(value) == before
        finally:
            administrator.execute(
                sql.SQL("DROP DATABASE {} WITH (FORCE)").format(sql.Identifier(name))
            )


def test_preparation_principals_cannot_invoke_final_atomic_bind(v2_boundary):
    value = v2_boundary
    before = _read_all(value)
    for connection in (
        value.preparation_connections.intent_writer,
        value.preparation_connections.retention_writer,
    ):
        with psycopg.connect(connection.dsn) as preparation:
            with pytest.raises(psycopg.errors.InsufficientPrivilege):
                preparation.execute(
                    sql.SQL("SELECT {}.commit_root_proof(%s)").format(
                        sql.Identifier(value.setup.schema)
                    ),
                    (b'"unused"',),
                )
            preparation.rollback()
    assert _read_all(value) == before


def test_reviewed_preparation_pin_can_be_reconstructed_without_mutation(v2_boundary):
    value = v2_boundary
    before = _read_all(value)
    recovered = preparation_qualification.requalify_root_proof_preparation_v2(
        PostgreSQLConnectionConfig(BASE_DSN),
        value.lifecycle_deployment,
        preparation_schema=value.preparation_deployment.preparation_schema,
        connections=value.preparation_connections,
    )
    assert recovered == value.preparation_deployment
    assert _read_all(value) == before


@pytest.mark.parametrize(
    "drift", ["immutable", "strict", "default_false", "trigger_when_false", "trigger_removed"]
)
def test_reconstruction_rejects_unreviewed_preparation_structure(v2_boundary, drift):
    value = v2_boundary
    before = _read_all(value)
    schema = value.preparation_deployment.preparation_schema
    function = sql.SQL("{}.lock_active_signer_v2(bytea)").format(sql.Identifier(schema))
    with psycopg.connect(BASE_DSN, autocommit=True) as administrator:
        trigger = administrator.execute(
            "SELECT t.tgname,pg_catalog.pg_get_triggerdef(t.oid),t.tgenabled FROM pg_catalog.pg_trigger t "
            "WHERE t.tgrelid=pg_catalog.to_regclass(%s) AND NOT t.tgisinternal ORDER BY t.tgname LIMIT 1",
            (schema + ".metadata",),
        ).fetchone()
        try:
            if drift in {"immutable", "strict"}:
                administrator.execute(
                    sql.SQL("ALTER FUNCTION {} {}").format(
                        function, sql.SQL("IMMUTABLE" if drift == "immutable" else "STRICT")
                    )
                )
            elif drift == "default_false":
                administrator.execute(
                    sql.SQL(
                        "ALTER TABLE {}.metadata ALTER COLUMN singleton SET DEFAULT false"
                    ).format(sql.Identifier(schema))
                )
            else:
                administrator.execute(
                    sql.SQL("DROP TRIGGER {} ON {}.metadata").format(
                        sql.Identifier(trigger[0]), sql.Identifier(schema)
                    )
                )
                if drift == "trigger_when_false":
                    administrator.execute(
                        trigger[1].replace(" EXECUTE FUNCTION", " WHEN (false) EXECUTE FUNCTION")
                    )
                    administrator.execute(
                        sql.SQL("ALTER TABLE {}.metadata ENABLE ALWAYS TRIGGER {}").format(
                            sql.Identifier(schema), sql.Identifier(trigger[0])
                        )
                    )
            with pytest.raises(preparation_qualification.RootProofPreparationV2QualificationError):
                preparation_qualification.requalify_root_proof_preparation_v2(
                    PostgreSQLConnectionConfig(BASE_DSN),
                    value.lifecycle_deployment,
                    preparation_schema=schema,
                    connections=value.preparation_connections,
                )
            assert _read_all(value) == before
        finally:
            if drift in {"immutable", "strict"}:
                administrator.execute(
                    sql.SQL("ALTER FUNCTION {} {}").format(
                        function,
                        sql.SQL("VOLATILE" if drift == "immutable" else "CALLED ON NULL INPUT"),
                    )
                )
            elif drift == "default_false":
                administrator.execute(
                    sql.SQL(
                        "ALTER TABLE {}.metadata ALTER COLUMN singleton SET DEFAULT true"
                    ).format(sql.Identifier(schema))
                )
            else:
                administrator.execute(
                    sql.SQL("DROP TRIGGER IF EXISTS {} ON {}.metadata").format(
                        sql.Identifier(trigger[0]), sql.Identifier(schema)
                    )
                )
                administrator.execute(trigger[1])
                mode = {"A": "ENABLE ALWAYS", "O": "ENABLE", "R": "ENABLE REPLICA", "D": "DISABLE"}[
                    trigger[2]
                ]
                administrator.execute(
                    sql.SQL("ALTER TABLE {}.metadata {} TRIGGER {}").format(
                        sql.Identifier(schema), sql.SQL(mode), sql.Identifier(trigger[0])
                    )
                )


@pytest.mark.parametrize("drift", ["acl", "function_source"])
def test_catalog_and_acl_drift_refuse_preparation_without_mutation(v2_boundary, drift):
    value = v2_boundary
    before = _read_all(value)
    function = sql.SQL("{}.reserve_intent(bytea,bytea,bytea,bytea,bytea,bytea,bytea)").format(
        sql.Identifier(value.intent_setup.schema)
    )
    with psycopg.connect(BASE_DSN, autocommit=True) as administrator:
        original = administrator.execute(
            "SELECT pg_catalog.pg_get_functiondef(pg_catalog.to_regprocedure(%s))",
            (
                value.intent_setup.schema
                + ".reserve_intent(bytea,bytea,bytea,bytea,bytea,bytea,bytea)",
            ),
        ).fetchone()[0]
        try:
            if drift == "acl":
                administrator.execute(
                    sql.SQL("GRANT EXECUTE ON FUNCTION {} TO PUBLIC").format(function)
                )
            else:
                administrator.execute(
                    original.replace("AS $function$", "AS $function$\n-- catalog drift\n", 1)
                )
            with pytest.raises(RootProofSigningIntentUnavailable):
                _reserve(value)
            assert _read_all(value) == before
        finally:
            if drift == "acl":
                administrator.execute(
                    sql.SQL("REVOKE EXECUTE ON FUNCTION {} FROM PUBLIC").format(function)
                )
            else:
                administrator.execute(original)


def test_missing_enrollment_cannot_prepare_even_with_active_local_signer(v2_boundary):
    value = v2_boundary
    assert value.signer.credential_snapshot().lifecycle_state is custody.SigningKeyLifecycle.ACTIVE
    with psycopg.connect(BASE_DSN) as administrator:
        administrator.execute(
            sql.SQL(
                "ALTER TABLE {}.lifecycle_credentials DISABLE TRIGGER lifecycle_immutable_row"
            ).format(sql.Identifier(value.setup.schema))
        )
        administrator.execute(
            sql.SQL("DELETE FROM {}.lifecycle_credentials WHERE fence_key=%s").format(
                sql.Identifier(value.setup.schema)
            ),
            (value.fences.root_key,),
        )
        administrator.execute(
            sql.SQL(
                "ALTER TABLE {}.lifecycle_credentials ENABLE ALWAYS TRIGGER lifecycle_immutable_row"
            ).format(sql.Identifier(value.setup.schema))
        )
        administrator.commit()
    before = _read_all(value)
    with pytest.raises(RootProofSigningIntentError):
        _reserve(value)
    assert _read_all(value) == before


def test_other_enrolled_active_signer_cannot_replace_original_reserved(prepared_v2_boundary):
    value, original, prepared = prepared_v2_boundary
    replacement_intent = preparation_runtime.PostgreSQLRootProofSigningIntentV2(
        value.preparation_connections.intent_writer,
        deployment=value.preparation_deployment,
        schema=value.intent_setup.schema,
        scope=value.scope,
        verification_authority=value.replacement_signer,
        issuer_authority=value.genuine.authority,
    )
    replacement_retention = preparation_runtime.PostgreSQLRootProofRetentionV2(
        value.preparation_connections.retention_writer,
        deployment=value.preparation_deployment,
        schema=value.retention_setup.schema,
        scope=value.scope,
        verification_authority=value.replacement_signer,
        signing_intent=replacement_intent,
    )
    assert (
        value.replacement_signer.credential_snapshot().lifecycle_state
        is custody.SigningKeyLifecycle.ACTIVE
    )
    before = _read_all(value)
    with pytest.raises(RootProofSigningIntentConflict):
        replacement_intent.reserve(value.genuine.binding, value.genuine.opaque)
    signature = value.replacement_signer.sign_root_proof(original.bytes_to_sign)
    with pytest.raises(RootProofRetentionConflict):
        replacement_retention.retain_prepared(
            value.genuine.binding,
            value.genuine.opaque,
            canonical_payload=original.canonical_payload,
            signature=signature,
        )
    assert _read_all(value) == before


def test_same_entitlement_distinct_attempt_stays_fail_closed_at_unique_constraint(
    reserved_v2_boundary,
):
    """SQL cardinality negative control, not two lawful candidates or a race."""
    value, reserved = reserved_v2_boundary
    envelope, payload = (
        parse_canonical(reserved.canonical_envelope),
        parse_canonical(reserved.canonical_payload),
    )
    for field, prefix in (
        ("issuance_attempt_id", "rpa_"),
        ("logical_operation_id", "ago_"),
        ("root_proof_id", "rpf_"),
        ("account_id", "acct_"),
    ):
        changed = mint_uuid7(prefix, reservation_epoch_milliseconds(datetime.now(timezone.utc)))
        envelope[field] = payload[field] = changed
    before = _read_all(value)
    with psycopg.connect(value.preparation_connections.intent_writer.dsn) as writer:
        with pytest.raises(psycopg.errors.UniqueViolation):
            writer.execute(
                sql.SQL("SELECT * FROM {}.reserve_intent(%s,%s,%s,%s,%s,%s,%s)").format(
                    sql.Identifier(value.intent_setup.schema)
                ),
                (
                    canonical_json_bytes(_scope_data(value.scope)),
                    canonical_json_bytes(envelope["issuance_attempt_id"]),
                    canonical_json_bytes(envelope["logical_operation_id"]),
                    canonical_json_bytes(
                        {
                            "bootstrap_entitlement_id": value.auth.bootstrap_entitlement_id,
                            "entitlement_generation": value.auth.entitlement_generation,
                        }
                    ),
                    canonical_json_bytes(envelope["root_proof_id"]),
                    canonical_json_bytes(envelope),
                    canonical_json_bytes(payload),
                ),
            )
        writer.rollback()
    assert _read_all(value) == before
    with psycopg.connect(BASE_DSN) as observer:
        names = observer.execute(
            "SELECT indexname FROM pg_catalog.pg_indexes WHERE schemaname=%s AND tablename='intents'",
            (value.intent_setup.schema,),
        ).fetchall()
    assert ("intents_entitlement_key_key",) in names


def test_changed_request_bytes_for_same_reserved_attempt_conflict(reserved_v2_boundary):
    value, reserved = reserved_v2_boundary
    envelope = parse_canonical(reserved.canonical_envelope)
    envelope["retained_request"]["canonical_bytes_hex"] += "00"
    before = _read_all(value)
    with psycopg.connect(value.preparation_connections.intent_writer.dsn) as preparation:
        with pytest.raises(psycopg.errors.UniqueViolation):
            preparation.execute(
                sql.SQL("SELECT * FROM {}.reserve_intent(%s,%s,%s,%s,%s,%s,%s)").format(
                    sql.Identifier(value.intent_setup.schema)
                ),
                (
                    canonical_json_bytes(_scope_data(value.scope)),
                    canonical_json_bytes(value.genuine.retained.attempt_id),
                    canonical_json_bytes(value.auth.logical_operation_id),
                    canonical_json_bytes(
                        {
                            "bootstrap_entitlement_id": value.auth.bootstrap_entitlement_id,
                            "entitlement_generation": value.auth.entitlement_generation,
                        }
                    ),
                    canonical_json_bytes(reserved.root_proof_id),
                    canonical_json_bytes(envelope),
                    reserved.canonical_payload,
                ),
            )
        preparation.rollback()
    assert _read_all(value) == before


@pytest.mark.parametrize("untrusted", [None, b"{}", {}])
def test_fabricated_binding_and_attempt_cannot_create_preparation(v2_boundary, untrusted):
    value = v2_boundary
    before = _read_all(value)
    with pytest.raises(RootProofSigningIntentConflict):
        value.intent.reserve(untrusted, value.genuine.opaque)
    with pytest.raises(RootProofSigningIntentConflict):
        value.intent.reserve(value.genuine.binding, untrusted)
    with pytest.raises(RootProofRetentionConflict):
        value.retention.retain_prepared(
            untrusted, value.genuine.opaque, canonical_payload=b"{}", signature=b"x" * 64
        )
    assert _read_all(value) == before


def test_valid_signature_without_genuine_reserved_cannot_be_staged(v2_boundary):
    value = v2_boundary
    payload = canonical_json_bytes({"caller": "cannot supply an authorization"})
    signature = value.signer.sign_root_proof(ROOT_PROOF_SIGNING_DOMAIN + payload)
    before = _read_all(value)
    with pytest.raises(RootProofRetentionConflict):
        value.retention.retain_prepared(
            value.genuine.binding,
            value.genuine.opaque,
            canonical_payload=payload,
            signature=signature,
        )
    assert _read_all(value) == before


def test_invalid_original_signature_cannot_create_staged(reserved_v2_boundary):
    value, reserved = reserved_v2_boundary
    before = _read_all(value)
    with pytest.raises(RootProofRetentionConflict):
        value.retention.retain_prepared(
            value.genuine.binding,
            value.genuine.opaque,
            canonical_payload=reserved.canonical_payload,
            signature=b"x" * 64,
        )
    assert _read_all(value) == before


@pytest.mark.parametrize("phase", ["RESERVED", "STAGED"])
@pytest.mark.parametrize("cut", ["BEFORE_COMMIT", "AFTER_COMMIT"])
def test_unknown_preparation_result_uses_durable_exact_bytes(v2_boundary, monkeypatch, phase, cut):
    value = v2_boundary
    reserved = _reserve(value) if phase == "STAGED" else None
    signature = (
        value.signer.sign_root_proof(reserved.bytes_to_sign) if reserved is not None else None
    )
    before = _read_all(value)

    def lose(name):
        if name == phase + "_" + cut:
            raise RuntimeError("real preparation response lost")

    monkeypatch.setattr(preparation_runtime, "_persistence_cut", lose)
    expected = (
        RootProofSigningIntentIndeterminate
        if phase == "RESERVED"
        else RootProofRetentionIndeterminate
    )
    with pytest.raises(expected):
        if phase == "RESERVED":
            _reserve(value)
        else:
            value.retention.retain_prepared(
                value.genuine.binding,
                value.genuine.opaque,
                canonical_payload=reserved.canonical_payload,
                signature=signature,
            )
    monkeypatch.setattr(preparation_runtime, "_persistence_cut", lambda *_: None)
    after = _read_all(value)
    _assert_only_preparation_changed(value, before, after)
    pair = (
        (value.intent_setup.schema, "intents")
        if phase == "RESERVED"
        else (value.retention_setup.schema, "prepared")
    )
    assert len(after[pair]) == (1 if cut == "AFTER_COMMIT" else 0)
    if cut == "BEFORE_COMMIT":
        assert after == before
    if phase == "RESERVED":
        recovered = value.intent.read_intent(value.genuine.binding, value.genuine.opaque)
        if recovered is not None:
            assert _reserve(value) == recovered
            assert _read_all(value) == after
    else:
        recovered = value.retention.read_prepared(
            authorization=value.auth, retained_request=value.genuine.retained
        )
        if recovered is not None:
            assert (recovered.canonical_payload, recovered.signature) == (
                reserved.canonical_payload,
                signature,
            )
            assert (
                value.retention.retain_prepared(
                    value.genuine.binding,
                    value.genuine.opaque,
                    canonical_payload=reserved.canonical_payload,
                    signature=signature,
                )
                == recovered
            )
            assert _read_all(value) == after


@pytest.mark.skipif(
    not hasattr(socket, "fromfd"), reason="native TCP receive shutdown requires fromfd"
)
@pytest.mark.parametrize("phase", ["RESERVED", "STAGED"])
def test_native_commit_receive_disconnect_recovers_original_preparation(
    v2_boundary, monkeypatch, phase
):
    value = v2_boundary
    reserved = _reserve(value) if phase == "STAGED" else None
    signature = (
        value.signer.sign_root_proof(reserved.bytes_to_sign) if reserved is not None else None
    )
    selected_dsn = (
        value.preparation_connections.intent_writer.dsn
        if phase == "RESERVED"
        else value.preparation_connections.retention_writer.dsn
    )
    before = _read_all(value)
    native_connect = preparation_runtime._connect
    real_connections = []
    interrupted = []

    def observe_connection(config):
        connection = native_connect(config)
        if config.dsn == selected_dsn:
            real_connections.append(connection)
        return connection

    def interrupt_receive(name):
        if name == phase + "_BEFORE_COMMIT":
            connection = next(conn for conn in reversed(real_connections) if not conn.closed)
            with socket.fromfd(
                connection.pgconn.socket, socket.AF_INET, socket.SOCK_STREAM
            ) as wire:
                wire.shutdown(socket.SHUT_RD)
            interrupted.append(connection.info.backend_pid)

    monkeypatch.setattr(preparation_runtime, "_connect", observe_connection)
    monkeypatch.setattr(preparation_runtime, "_persistence_cut", interrupt_receive)
    error = (
        RootProofSigningIntentIndeterminate
        if phase == "RESERVED"
        else RootProofRetentionIndeterminate
    )
    with pytest.raises(error):
        if phase == "RESERVED":
            _reserve(value)
        else:
            value.retention.retain_prepared(
                value.genuine.binding,
                value.genuine.opaque,
                canonical_payload=reserved.canonical_payload,
                signature=signature,
            )
    assert len(interrupted) == 1
    monkeypatch.setattr(preparation_runtime, "_persistence_cut", lambda *_: None)
    deadline = time.monotonic() + 10
    while True:
        recovered = (
            value.intent.read_intent(value.genuine.binding, value.genuine.opaque)
            if phase == "RESERVED"
            else value.retention.read_prepared(
                authorization=value.auth, retained_request=value.genuine.retained
            )
        )
        if recovered is not None:
            break
        if time.monotonic() >= deadline:
            pytest.fail("native COMMIT did not retain preparation after receive disconnect")
        time.sleep(0.02)
    after = _read_all(value)
    _assert_only_preparation_changed(value, before, after)
    if phase == "RESERVED":
        assert _reserve(value) == recovered
    else:
        assert (recovered.canonical_payload, recovered.signature) == (
            reserved.canonical_payload,
            signature,
        )
        assert (
            value.retention.retain_prepared(
                value.genuine.binding,
                value.genuine.opaque,
                canonical_payload=reserved.canonical_payload,
                signature=signature,
            )
            == recovered
        )
    assert _read_all(value) == after


@requires_fork
@pytest.mark.parametrize("phase", ["RESERVED", "STAGED"])
def test_real_process_crash_preserves_preparation_without_approval(v2_boundary, phase):
    value = v2_boundary
    before = _read_all(value)
    context = multiprocessing.get_context("fork")
    reader, writer = context.Pipe(duplex=False)
    selected_cut = f"{phase}_AFTER_COMMIT"

    def crash():
        reader.close()

        def crash_after_commit(name):
            if name == selected_cut:
                writer.send(name)
                os._exit(73)

        preparation_runtime._persistence_cut = crash_after_commit
        reserved = _reserve(value)
        if phase == "STAGED":
            _stage(value, reserved)

    process = context.Process(target=crash)
    process.start()
    writer.close()
    try:
        # Native authorization runs before the crash cut under the unchanged
        # CI job deadline. The exit bound starts after the real durable commit.
        while not reader.poll(1):
            if not process.is_alive():
                process.join(timeout=20)
                pytest.fail("independent preparation exited before its durable crash cut")
        try:
            assert reader.recv() == selected_cut
        except EOFError:
            process.join(timeout=20)
            pytest.fail("independent preparation closed its pipe before the durable crash cut")
        process.join(timeout=20)
        if process.is_alive():
            pytest.fail(
                "independent preparation process failed to exit after its durable crash cut"
            )
    finally:
        reader.close()
        if process.is_alive():
            process.kill()
            process.join(timeout=5)
    assert process.exitcode == 73
    after = _read_all(value)
    _assert_only_preparation_changed(value, before, after)
    reserved = value.intent.read_intent(value.genuine.binding, value.genuine.opaque)
    assert reserved is not None and _reserve(value) == reserved
    prepared = value.retention.read_prepared(
        authorization=value.auth, retained_request=value.genuine.retained
    )
    assert (prepared is not None) is (phase == "STAGED")
    if prepared is not None:
        assert prepared.canonical_payload == reserved.canonical_payload
    assert after[(value.setup.schema, "approved_commits")] == []
    assert after[(value.setup.schema, "decisions")] == []
    assert _read_all(value) == after


def test_legacy_adapters_still_reject_the_migrated_catalog(v2_boundary):
    value = v2_boundary
    before = _read_all(value)
    with pytest.raises(RootProofSigningIntentUnavailable):
        value.legacy_intent.reserve(value.genuine.binding, value.genuine.opaque)
    with pytest.raises(RootProofRetentionUnavailable):
        value.legacy_retention.read_prepared(
            authorization=value.auth, retained_request=value.genuine.retained
        )
    assert _read_all(value) == before


def _second_unbound_attempt(value, lineage_active, namespace, selected):
    """A distinct unbound entitlement is a lifecycle control, never a race proof."""
    namespace.mkdir(mode=0o700)
    authority = PostgreSQLRootProofIssuanceAuthorityV2(
        value.preparation_connections.entitlement_runtime,
        value.preparation_connections.requester_runtime,
        value.preparation_connections.claimant_runtime,
        deployment=value.preparation_deployment,
        entitlement_subject=value.secondary_subject,
    )
    selected.setattr(
        installed_lppi, "_state_path", lambda: namespace / "initial-authenticated-operation.json"
    )
    selected.setattr(
        installed_cha, "_state_path", lambda: namespace / "initial-cha-logical-operation.json"
    )
    selected.setattr(
        installed_account, "_state_path", lambda: namespace / "initial-cha-account-reservation.json"
    )
    selected.setattr(
        attempt_boundary,
        "_attempt_store_path",
        lambda: value.genuine.path.with_name("fresh-v2-secondary-attempt.sqlite3"),
    )
    upstream = installed_lppi.establish_installed_lppi_authenticated_operation(lineage_active)
    cha = installed_cha.establish_installed_cha_logical_operation(upstream)
    binding = installed_account.establish_installed_account_initial_binding(cha)
    authorization = attempt_boundary.resolve_root_proof_issuance_authorization_v2(
        binding, authority
    )
    attempt = attempt_boundary.reserve_root_proof_issuance_attempt(binding, authorization)
    signed = signed_boundary.sign_root_proof_issuance_attempt(attempt)
    intent = preparation_runtime.PostgreSQLRootProofSigningIntentV2(
        value.preparation_connections.intent_writer,
        deployment=value.preparation_deployment,
        schema=value.intent_setup.schema,
        scope=value.scope,
        verification_authority=value.signer,
        issuer_authority=authority,
    )
    return SimpleNamespace(binding=binding, signed=signed, intent=intent)


def test_durable_pending_verify_only_revoked_close_blocks_new_preparation_and_keeps_history(
    module_v2_boundary,
    lineage_active,
    lineage_path,
    monkeypatch,
):
    """Final irreversible custody case; local metadata is never reset or reopened."""
    value = module_v2_boundary
    reserved, prepared = _stage(value)
    preparation, _ = _approval(value)
    decision = value.commit.commit(preparation)
    with monkeypatch.context() as selected:
        candidate = _second_unbound_attempt(
            value, lineage_active, lineage_path / "close-unbound-candidate", selected
        )
        # A genuine new attempt is admissible while the enrolled signer is
        # ACTIVE, independently of the first candidate's already-BOUND state.
        assert candidate.intent.read_intent(candidate.binding, candidate.signed) is None
        administrator = lifecycle.PostgreSQLCustodyLifecycleAdministrator(
            value.lifecycle_deployment
        )
        request = lifecycle.prepare_custody_lifecycle_request(
            value.lifecycle_deployment,
            value.directory,
            security=value.security,
            role=ProviderRole.ROOT_PROOF_SIGNING,
            administrative_operation_id="v2-close-" + uuid.uuid4().hex,
            target=custody.SigningKeyLifecycle.VERIFY_ONLY,
            fence_key=value.fences.root_key,
        )
        authorization = administrator.begin_close(request)
        assert (
            value.signer.credential_snapshot().lifecycle_state is custody.SigningKeyLifecycle.ACTIVE
        )
        pending = _read_all(value)
        pending_fence = next(
            row
            for row in pending[(value.setup.schema, "fences")]
            if row[0] == value.fences.root_key
        )
        assert "PENDING_CLOSED" in pending_fence
        with pytest.raises(RootProofSigningIntentError):
            candidate.intent.reserve(candidate.binding, candidate.signed)
        assert _read_all(value) == pending
        completion = administrator.mutate_custody(authorization, request)
        administrator.publish_completion(request, completion)
        for state in (custody.SigningKeyLifecycle.VERIFY_ONLY, custody.SigningKeyLifecycle.REVOKED):
            if state is custody.SigningKeyLifecycle.REVOKED:
                revoke = lifecycle.prepare_custody_lifecycle_request(
                    value.lifecycle_deployment,
                    value.directory,
                    security=value.security,
                    role=ProviderRole.ROOT_PROOF_SIGNING,
                    administrative_operation_id="v2-revoke-" + uuid.uuid4().hex,
                    target=state,
                    fence_key=value.fences.root_key,
                )
                administrator.execute(revoke)
            before = _read_all(value)
            with pytest.raises(RootProofSigningIntentConflict):
                candidate.intent.reserve(candidate.binding, candidate.signed)
            recovered = value.commit.reconcile(preparation.locator)
            assert recovered == decision
            assert (recovered.payload, recovered.signature, recovered.envelope) == (
                prepared.canonical_payload,
                prepared.signature,
                prepared.canonical_envelope,
            )
            assert value.commit.commit(preparation) == decision
            assert prepared.credential_snapshot == reserved.credential_snapshot
            assert _read_all(value) == before

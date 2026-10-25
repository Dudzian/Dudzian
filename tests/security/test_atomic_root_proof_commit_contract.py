"""Fast refusal-path tests; actual atomic writes are proved on PostgreSQL 16."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from types import SimpleNamespace

import pytest

from bot_core import (
    postgresql_atomic_root_proof_commit as runtime,
    postgresql_atomic_root_proof_migration as migration,
    postgresql_atomic_root_proof_preaccount_sql as preaccount_fences,
    postgresql_atomic_root_proof_qualification as qualification,
    postgresql_atomic_root_proof_schema as schema,
    root_proof_atomic_composition as production_gate,
)
from bot_core.authenticated_issuer_history import (
    NO_PREDECESSOR,
    HistoryEventIdentity,
    HistoryStreamIdentity,
    build_record,
    canonical_json_bytes as history_json_bytes,
)
from bot_core.entitlement_registry_contract import (
    AuthoritativeEntitlementState,
    BoundBinding,
    EntitlementIdentity,
    EntitlementLifecycle,
    EntitlementProvenance,
    RegistrySubject,
    UnboundBinding,
)
from bot_core.licensing.canonical import canonical_json_bytes as contract_json_bytes
from bot_core.postgresql_entitlement_registry import PostgreSQLConnectionConfig
from bot_core.postgresql_root_proof_retention import RootProofRetentionScope
from bot_core.root_proof_issuer_substrate import CredentialSemanticRole, SecurityProfile
from tests.licensing.test_cha_attempt_store_stage9 import authorization


def test_actual_registry_enum_projection_builds_exact_unchanged_history_event():
    auth = authorization(bootstrap_entitlement_id="ent_018f3e70-7b59-7c21-8b9a-0123456789ab")
    predecessor = AuthoritativeEntitlementState(
        RegistrySubject("entitlement-lookup", auth.environment, auth.trust_domain),
        EntitlementIdentity(
            auth.bootstrap_entitlement_id,
            auth.entitlement_generation,
            auth.environment,
            auth.trust_domain,
            auth.product_scope,
        ),
        EntitlementProvenance(
            auth.provisioning_principal_id,
            auth.claimant_key_id,
            auth.claimant_key_version,
            "provisioning-authority",
            "authenticated-creation",
            "a" * 64,
        ),
        EntitlementLifecycle.ACTIVE,
        UnboundBinding(),
        1,
        None,
    )
    attempt_id = "rpa_018f3e70-7b5d-7c21-8b9a-0123456789ab"
    root_id = "rpf_018f3e70-7b5e-7c21-8b9a-0123456789ab"
    binding = BoundBinding(
        auth.logical_operation_id,
        auth.account_id,
        auth.canonical_genesis_request_fingerprint_sha256,
        auth.entitlement_generation,
        attempt_id,
        auth.requester_principal_id,
        auth.requester_key_id,
        auth.requester_key_version,
        auth.provisioning_principal_id,
        auth.claimant_key_id,
        auth.claimant_key_version,
        "b" * 64,
        "retained-request",
        root_id,
        "issuer-credential",
        1,
    )
    bound = replace(
        predecessor, binding=binding, authoritative_state_revision=2, predecessor_revision=1
    )
    stream = HistoryStreamIdentity(
        "history",
        "issuer",
        "PRODUCTION_LOCAL",
        auth.environment,
        auth.trust_domain,
        auth.product_scope,
        1,
    )
    event = HistoryEventIdentity(auth.logical_operation_id, attempt_id, root_id)
    raw_projection = runtime._expected_issuance_history_payload(auth, bound, predecessor, "c" * 64)
    with pytest.raises(TypeError, match="non-canonical JSON type"):
        build_record(stream, 1, NO_PREDECESSOR, event, raw_projection)
    projection = runtime._issuance_history_payload(auth, bound, predecessor, "c" * 64)
    record = build_record(stream, 1, NO_PREDECESSOR, event, projection)
    assert record.event_identity == event
    assert projection["entitlement"]["lifecycle"] == "ACTIVE"
    assert projection["entitlement"]["binding"]["kind"] == "BOUND"
    assert history_json_bytes(projection) == contract_json_bytes(raw_projection)


def _provisioning():
    return schema.PostgreSQLAtomicRootProofProvisioning(
        schema="atomic",
        schema_owner_role="atomic_owner",
        issuer_writer_role="atomic_writer",
        test_admin_role="atomic_verifier",
        entitlement_schema="entitlement",
        intent_schema="intent",
        retention_schema="retention",
        history_schema="history",
        trust_domain="trust",
        instance_id="isolated-test",
        requester_schema="requester",
        claimant_schema="claimant",
    )


def _identity():
    return qualification.PostgreSQLAtomicRootProofPhysicalIdentity(
        7412345678901234567,
        16001,
        "authority",
        "isolated",
        "/tmp/postgresql",
        5432,
        None,
        None,
        "isolated-test",
    )


def _deployment(*, administrative=False):
    config = _provisioning()
    roles = tuple(
        (name, oid)
        for oid, name in enumerate(
            (config.schema_owner_role, config.issuer_writer_role, config.test_admin_role), 16010
        )
    )
    return qualification.PostgreSQLAtomicRootProofTestDeployment(
        PostgreSQLConnectionConfig("dbname=authority user=atomic_writer password=private-pin"),
        config,
        _identity(),
        "ab" * 32,
        roles,
        (config.schema_owner_role,),
        qualification._ADMINISTRATIVE_ORIGIN if administrative else None,
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema", "public;DROP SCHEMA public"),
        ("schema_owner_role", "Owner"),
        ("issuer_writer_role", "postgres;SET ROLE postgres"),
        ("test_admin_role", ""),
        ("entitlement_schema", "a" * 64),
        ("intent_schema", "schema.with.dot"),
        ("retention_schema", "schema-name"),
        ("history_schema", "history\n"),
        ("requester_schema", "requester;SELECT true"),
        ("claimant_schema", "claimant.role"),
        ("trust_domain", " "),
        ("instance_id", ""),
        ("trust_domain", None),
        ("instance_id", True),
    ],
)
def test_provisioning_refuses_malformed_authority_configuration(field, value):
    with pytest.raises((TypeError, ValueError)):
        replace(_provisioning(), **{field: value})


@pytest.mark.parametrize(
    "changes",
    [
        {"issuer_writer_role": "atomic_owner"},
        {"test_admin_role": "atomic_writer"},
        {"test_admin_role": "atomic_owner"},
        {"entitlement_schema": "atomic"},
        {"intent_schema": "entitlement"},
        {"retention_schema": "history"},
        {"requester_schema": "atomic"},
        {"requester_schema": "entitlement"},
        {"claimant_schema": "requester"},
        {"claimant_schema": "history"},
    ],
)
def test_provisioning_refuses_collapsed_semantic_authorities(changes):
    with pytest.raises(ValueError, match="distinct"):
        replace(_provisioning(), **changes)


def test_test_deployment_is_immutable_and_does_not_disclose_transport_secrets():
    deployment = _deployment()
    assert "private-pin" not in repr(deployment)
    assert "dbname=authority" not in repr(deployment)
    assert "ISOLATED_TEST_ONLY" in repr(deployment)
    with pytest.raises(FrozenInstanceError):
        deployment.catalog_sha256 = "00" * 32


@pytest.mark.parametrize(
    "field,value",
    [
        ("system_identifier", True),
        ("system_identifier", "7412345678901234567"),
        ("system_identifier", 0),
        ("database_oid", False),
        ("database_oid", -1),
        ("database_name", ""),
        ("database_name", object()),
        ("cluster_name", None),
        ("connection_host", ""),
        ("connection_port", True),
        ("connection_port", 0),
        ("connection_port", 65536),
        ("server_address", b"127.0.0.1"),
        ("server_port", True),
        ("server_port", 0),
        ("server_port", 65536),
        ("instance_id", ""),
    ],
)
def test_physical_identity_refuses_type_confusion_and_unknown_service_coordinates(field, value):
    with pytest.raises(ValueError):
        replace(_identity(), **{field: value})


class _NoProbe:
    def execute(self, *args, **kwargs):
        pytest.fail("untrusted configuration must be refused before probing authority")


@pytest.mark.parametrize(
    "candidate",
    [
        object(),
        {"boundary": "ISOLATED_TEST_ONLY", "authorizes_operations": True},
        lambda: _deployment(administrative=True),
        SimpleNamespace(test_mode=True, provisioning=_provisioning()),
        _deployment(),
    ],
)
def test_untrusted_receipts_dtos_and_callbacks_cannot_authorize_live_qualification(candidate):
    with pytest.raises(
        qualification.AtomicRootProofQualificationError, match="administrative test deployment"
    ):
        qualification.qualify_atomic_root_proof_authority(_NoProbe(), candidate)


@pytest.mark.parametrize("expected_role", ["schema_owner", "postgres", "", None])
def test_no_schema_owner_or_caller_selected_principal_can_qualify(expected_role):
    with pytest.raises(
        qualification.AtomicRootProofQualificationError, match="administrative test deployment"
    ):
        qualification.qualify_atomic_root_proof_authority(
            _NoProbe(), _deployment(administrative=True), expected_role=expected_role
        )


class _CatalogRead:
    """Only read-only catalog replies, never a transaction or commit result."""

    def __init__(
        self, *, principal=None, version=160006, durable=None, physical=None, metadata=None
    ):
        self.autocommit = False
        identity = _identity()
        self.info = SimpleNamespace(
            server_version=version, host=identity.connection_host, port=identity.connection_port
        )
        self.principal = ("atomic_writer", "atomic_writer") if principal is None else principal
        self.durable = ("on", "on", "on", False) if durable is None else durable
        self.physical = (
            (
                identity.system_identifier,
                identity.database_oid,
                identity.database_name,
                identity.cluster_name,
                identity.server_address,
                identity.server_port,
            )
            if physical is None
            else physical
        )
        self.metadata = [] if metadata is None else metadata
        self.queries = []

    def execute(self, query, params=None):
        self.queries.append(query)
        if query == "SET LOCAL search_path=pg_catalog":
            row, rows = None, []
        elif query == "SELECT session_user::text,current_user::text":
            row, rows = self.principal, []
        elif isinstance(query, str) and "current_setting('fsync')" in query:
            row, rows = self.durable, []
        elif isinstance(query, str) and "pg_control_system()" in query:
            row, rows = self.physical, []
        elif not isinstance(query, str) and "metadata" in str(query):
            row, rows = None, self.metadata
        else:
            pytest.fail(f"unexpected catalog read after refusal boundary: {query}")
        return SimpleNamespace(fetchone=lambda: row, fetchall=lambda: rows)


@pytest.mark.parametrize(
    "principal",
    [
        ("atomic_verifier", "atomic_verifier"),
        ("atomic_owner", "atomic_owner"),
        ("atomic_verifier", "atomic_writer"),
        ("atomic_writer", "atomic_owner"),
        ("postgres", "atomic_writer"),
        None,
    ],
)
def test_writer_qualification_requires_exact_direct_login_not_set_role(principal):
    conn = _CatalogRead(principal=principal)
    if principal is None:
        conn.principal = None
    with pytest.raises(qualification.AtomicRootProofQualificationError, match="direct-login"):
        qualification.qualify_atomic_root_proof_authority(conn, _deployment(administrative=True))
    assert len(conn.queries) == 2


@pytest.mark.parametrize("version", [150009, 170001, 0])
def test_atomic_writer_refuses_every_unqualified_postgresql_version(version):
    conn = _CatalogRead(version=version)
    with pytest.raises(qualification.AtomicRootProofQualificationError, match="PostgreSQL 16"):
        qualification.qualify_atomic_root_proof_authority(conn, _deployment(administrative=True))
    assert len(conn.queries) == 2


@pytest.mark.parametrize(
    "durable",
    [
        ("off", "on", "on", False),
        ("on", "off", "on", False),
        ("on", "local", "on", False),
        ("on", "on", "off", False),
        ("on", "on", "on", True),
        ("on", "on", "on"),
    ],
)
def test_missing_durability_or_standby_cannot_authorize_new_commit(durable):
    conn = _CatalogRead(durable=durable)
    with pytest.raises(qualification.AtomicRootProofQualificationError, match="durable"):
        qualification.qualify_atomic_root_proof_authority(conn, _deployment(administrative=True))
    assert len(conn.queries) == 3


@pytest.mark.parametrize(
    "field,value",
    [
        ("system_identifier", 7412345678901234568),
        ("database_oid", 16002),
        ("database_name", "other_authority"),
        ("cluster_name", "cloned_cluster"),
        ("connection_host", "/tmp/other-postgresql"),
        ("connection_port", 5433),
        ("server_address", "127.0.0.2"),
        ("server_port", 5433),
        ("instance_id", "other-isolated-test"),
    ],
)
def test_same_dsn_or_database_name_cannot_hide_changed_physical_identity(field, value):
    deployment = replace(
        _deployment(administrative=True), identity=replace(_identity(), **{field: value})
    )
    conn = _CatalogRead()
    with pytest.raises(qualification.AtomicRootProofQualificationError, match="physical database"):
        qualification.qualify_atomic_root_proof_authority(conn, deployment)
    assert len(conn.queries) == 4


@pytest.mark.parametrize("physical", [None, (), (7412345678901234567, 16001, "authority")])
def test_absent_or_partial_control_catalog_row_does_not_establish_physical_authority(physical):
    conn = _CatalogRead()
    conn.physical = physical
    with pytest.raises(qualification.AtomicRootProofQualificationError, match="physical"):
        qualification.read_atomic_root_proof_physical_identity(conn, "isolated-test")


def test_database_catalog_failure_is_a_redacted_qualification_failure():
    class FailingRead:
        def execute(self, *args, **kwargs):
            raise qualification.psycopg.OperationalError("private-dsn-secret")

    with pytest.raises(qualification.AtomicRootProofQualificationError) as error:
        qualification.read_atomic_root_proof_physical_identity(FailingRead(), "isolated-test")
    assert "private-dsn-secret" not in str(error.value)


def test_aliasing_authority_role_oids_are_rejected_before_reading_role_catalog():
    deployment = _deployment(administrative=True)
    deployment = replace(
        deployment,
        role_oids=tuple((name, 16010) for name, _ in deployment.role_oids),
    )
    with pytest.raises(qualification.AtomicRootProofQualificationError, match="identities alias"):
        qualification._qualify_roles(_NoProbe(), deployment)


@pytest.mark.parametrize("changed_index", range(8))
def test_changed_owner_oid_login_or_privileged_role_attributes_are_not_authority(changed_index):
    deployment = _deployment(administrative=True)
    name, oid = deployment.role_oids[0]
    observed = [oid, False, False, False, False, False, False, False]
    observed[changed_index] = oid + 1 if changed_index == 0 else True

    class RoleRead:
        def execute(self, query, params):
            assert params == (name,)
            return SimpleNamespace(fetchone=lambda: tuple(observed))

    with pytest.raises(qualification.AtomicRootProofQualificationError, match="OID or attributes"):
        qualification._qualify_roles(RoleRead(), deployment)


def test_role_membership_remains_forbidden_even_with_unchanged_role_oids():
    deployment = _deployment(administrative=True)
    roles = dict(deployment.role_oids)

    class MembershipRead:
        def execute(self, query, params):
            if "pg_auth_members" in query:
                row = (1,)
            else:
                name = params[0]
                row = (
                    roles[name],
                    name not in deployment.owner_roles,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                )
            return SimpleNamespace(fetchone=lambda: row)

    with pytest.raises(
        qualification.AtomicRootProofQualificationError, match="membership forbidden"
    ):
        qualification._qualify_roles(MembershipRead(), deployment)


def _preaccount_helper_configuration():
    return {
        "owner": "requester_owner",
        "writer": "atomic_writer",
        "schema": "requester",
        "peer_schema": "claimant",
        "semantic_role": CredentialSemanticRole.ROOT_PROOF_REQUESTER.value,
        "trust_domain": "trust",
    }


@pytest.mark.parametrize(
    "field,value",
    [
        ("owner", "requester_owner;SET ROLE postgres"),
        ("writer", "postgres;SELECT true"),
        ("schema", "public.requester"),
        ("peer_schema", "claimant\n"),
        ("owner", None),
        ("writer", lambda: "atomic_writer"),
        ("semantic_role", "ROOT_PROOF_SIGNING"),
        ("semantic_role", CredentialSemanticRole.ROOT_PROOF_REQUESTER),
        ("semantic_role", {"role": "ROOT_PROOF_REQUESTER"}),
        ("trust_domain", " "),
        ("trust_domain", None),
        ("trust_domain", lambda: "trust"),
    ],
)
def test_preaccount_helper_refuses_caller_sql_identifiers_callbacks_and_wrong_semantic_roles(
    field, value
):
    config = {**_preaccount_helper_configuration(), field: value}
    with pytest.raises(ValueError):
        preaccount_fences.reviewed_preaccount_helper(**config)


@pytest.mark.parametrize("changes", [{"peer_schema": "requester"}, {"owner": "atomic_writer"}])
def test_preaccount_helper_refuses_collapsed_authority_ownership(changes):
    with pytest.raises(ValueError, match="independent"):
        preaccount_fences.reviewed_preaccount_helper(
            **{**_preaccount_helper_configuration(), **changes}
        )


@pytest.mark.parametrize(
    "candidate",
    [
        None,
        object(),
        {"lifecycle": "ACTIVE", "registry_revision": 1, "authorizes_operations": True},
        SimpleNamespace(lifecycle="ACTIVE", lifecycle_generation=1, test_mode=True),
        lambda: {"lifecycle": "ACTIVE"},
    ],
)
def test_credential_capture_never_promotes_missing_or_caller_supplied_active_snapshots(candidate):
    with pytest.raises(
        runtime.AtomicRootProofCommitConflict, match="GENUINE_PUBLIC_CREDENTIAL_GENERATION"
    ):
        runtime._credential_generation_data(candidate)


@pytest.mark.parametrize(
    "expected_identity",
    [object(), {"database_name": "authority"}, lambda: _identity(), None],
)
def test_restart_reconstruction_never_accepts_caller_physical_pin_dtos(
    expected_identity, monkeypatch
):
    monkeypatch.setattr(
        runtime.psycopg, "connect", lambda *args, **kwargs: pytest.fail("unexpected probe")
    )
    with pytest.raises(TypeError, match="explicit administrator reconstruction"):
        migration.requalify_existing_atomic_root_proof_v1(
            PostgreSQLConnectionConfig("dbname=authority user=postgres"),
            PostgreSQLConnectionConfig("dbname=authority user=atomic_writer"),
            _provisioning(),
            expected_identity=expected_identity,
        )


@pytest.mark.parametrize("metadata", [[], [("unreviewed-version",)], [(), ()]])
def test_missing_unknown_or_duplicate_migration_metadata_refuses_authority(metadata):
    conn = _CatalogRead(metadata=metadata)
    with pytest.raises(qualification.AtomicRootProofQualificationError, match="migration identity"):
        qualification.qualify_atomic_root_proof_authority(conn, _deployment(administrative=True))
    assert len(conn.queries) == 5


def _locator():
    return runtime.AtomicRootProofCommitLocator(
        b'"scope"', b'"attempt"', b'"operation"', b'"entitlement"', b'"root"', b'"account"'
    )


def _fences():
    return runtime.AtomicRootProofCommitFences("root", 1, "history", 2, "trust", 3)


@pytest.mark.parametrize(
    "field", ["scope", "attempt", "operation", "entitlement", "root_proof", "account"]
)
@pytest.mark.parametrize("value", ["caller-json", bytearray(b"mutable"), memoryview(b"view"), None])
def test_recovery_locators_require_immutable_identity_bytes(field, value):
    with pytest.raises(TypeError, match="IMMUTABLE_COMMIT_LOCATOR"):
        replace(_locator(), **{field: value})


@pytest.mark.parametrize("field", ["root_key", "history_key", "trust_key"])
@pytest.mark.parametrize("value", ["", None, True, b"caller-key", lambda: "caller-key"])
def test_lifecycle_fence_keys_refuse_callbacks_and_type_confusion(field, value):
    with pytest.raises(TypeError, match="LIFECYCLE_FENCE_IDENTITY"):
        replace(_fences(), **{field: value})


@pytest.mark.parametrize("field", ["root_generation", "history_generation", "trust_generation"])
@pytest.mark.parametrize("value", [0, -1, True, 1.0, "1", None, float("nan"), float("inf")])
def test_lifecycle_fence_generations_refuse_nonpositive_and_nonintegral_values(field, value):
    with pytest.raises(TypeError, match="POSITIVE_LIFECYCLE_GENERATION"):
        replace(_fences(), **{field: value})


@pytest.mark.parametrize(
    "candidate",
    [
        object(),
        {"state": "READY", "test_mode": True, "authorizes_operations": True},
        SimpleNamespace(scope="PRODUCTION", production_ready=True),
        lambda: _deployment(administrative=True),
    ],
)
def test_arbitrary_dtos_and_admin_flags_cannot_construct_internal_commit_authority(
    candidate, monkeypatch
):
    def reject_connect(*args, **kwargs):
        pytest.fail("caller flags and DTOs must be refused before connecting")

    monkeypatch.setattr(runtime.psycopg, "connect", reject_connect)
    with pytest.raises(TypeError, match="QUALIFIED_INTERNAL_TEST_AUTHORITY"):
        runtime.PostgreSQLAtomicRootProofCommit(
            candidate, admin_connection=PostgreSQLConnectionConfig("dbname=authority")
        )


@pytest.mark.parametrize("admin_connection", [object(), "dbname=postgres", lambda: None])
def test_arbitrary_admin_connection_does_not_become_a_verifier_authority(
    admin_connection, monkeypatch
):
    monkeypatch.setattr(
        runtime.psycopg, "connect", lambda *args, **kwargs: pytest.fail("unexpected probe")
    )
    with pytest.raises(TypeError, match="QUALIFIED_INTERNAL_TEST_AUTHORITY"):
        runtime.PostgreSQLAtomicRootProofCommit(
            _deployment(administrative=True), admin_connection=admin_connection
        )


def test_internal_commit_authority_cannot_be_subclassed_to_replace_qualification():
    with pytest.raises(TypeError, match="CANNOT_BE_SUBCLASSED"):
        type("CallerAuthority", (runtime.PostgreSQLAtomicRootProofCommit,), {})


def test_isolated_atomic_implementation_never_opens_public_production_composition_gate(monkeypatch):
    connection = PostgreSQLConnectionConfig("dbname=authority user=runtime")
    scope = RootProofRetentionScope(
        SecurityProfile.PRODUCTION_LOCAL, "PRODUCTION", "trust", "product", "issuer"
    )
    pins = {
        name: production_gate.AtomicRootProofAuthorityPin(
            connection, name + "_schema", name + "_runtime"
        )
        for name in production_gate._AUTHORITY_NAMES
    }
    deployment = production_gate.AtomicRootProofDeployment(
        scope, **pins, transaction_owner=connection, transaction_owner_role="atomic_owner"
    )
    authority = object.__new__(runtime.PostgreSQLAtomicRootProofCommit)
    monkeypatch.setattr(
        runtime.psycopg, "connect", lambda *args, **kwargs: pytest.fail("unexpected probe")
    )
    result = production_gate.qualify_atomic_root_proof_composition(
        deployment,
        issuance_authority=authority,
        signing_intent=authority,
        retention=authority,
        history=authority,
        checkpoint=authority,
    )
    assert result.state == "NOT_READY"
    assert result.authorizes_operations is False
    assert "EXACT_INTERNAL_COMPOSITION_REQUIRED" in result.failures
    assert schema.PRODUCTION_LIFECYCLE_FENCE_AVAILABLE is False


@pytest.mark.parametrize(
    "candidate",
    [
        object(),
        {"authorizes_operations": True, "state": "VERIFIED_NOT_AUTHORIZED_TO_ISSUE"},
        SimpleNamespace(test_mode=True, bound_state="BOUND"),
        lambda: None,
    ],
)
def test_preparation_dtos_and_callbacks_are_not_genuine_signing_authorities(candidate):
    with pytest.raises(
        runtime.AtomicRootProofCommitConflict, match="GENUINE_PREPARATION_AUTHORITIES"
    ):
        runtime.prepare_atomic_root_proof_commit(
            object(),
            object(),
            signing_intent=candidate,
            retention=candidate,
            issuer_authority=candidate,
            history=candidate,
        )


def test_caller_constructed_preparation_does_not_authorize_commit_or_admin_approval():
    candidate = runtime.PreparedAtomicRootProofCommit(
        object(), object(), object(), object(), _locator()
    )
    with pytest.raises(
        runtime.AtomicRootProofCommitConflict, match="GENUINE_IMMUTABLE_PREPARATION"
    ):
        runtime._trusted_preparation(candidate)
    authority = object.__new__(runtime.PostgreSQLAtomicRootProofCommit)
    object.__setattr__(
        authority,
        "_PostgreSQLAtomicRootProofCommit__admin_connection",
        PostgreSQLConnectionConfig("dbname=authority user=atomic_verifier"),
    )
    with pytest.raises(
        runtime.AtomicRootProofCommitConflict, match="GENUINE_IMMUTABLE_PREPARATION"
    ):
        authority.commit(candidate)
    with pytest.raises(
        runtime.AtomicRootProofCommitConflict, match="GENUINE_IMMUTABLE_PREPARATION"
    ):
        authority.approve_prepared(candidate, object(), lambda: b"signature", fences=_fences())


def test_unsigned_deployment_is_rejected_before_connection_even_with_test_admin_credentials(
    monkeypatch,
):
    monkeypatch.setattr(
        runtime.psycopg, "connect", lambda *args, **kwargs: pytest.fail("unexpected probe")
    )
    with pytest.raises(qualification.AtomicRootProofQualificationError, match="administrative"):
        runtime.PostgreSQLAtomicRootProofCommit(
            _deployment(),
            admin_connection=PostgreSQLConnectionConfig("dbname=authority user=atomic_verifier"),
        )


@pytest.mark.parametrize(
    "candidate", [object(), {"attempt": b"caller"}, lambda: _locator(), "attempt"]
)
def test_reconciliation_does_not_resolve_arbitrary_callbacks_or_locator_dtos(candidate):
    with pytest.raises(runtime.AtomicRootProofCommitConflict, match="EXACT_RECOVERY_LOCATOR"):
        runtime.PostgreSQLAtomicRootProofCommit._locator(candidate)


def _decision_identity():
    value = _locator()
    return {
        "scope_hex": value.scope.hex(),
        "attempt_hex": value.attempt.hex(),
        "operation_hex": value.operation.hex(),
        "entitlement_hex": value.entitlement.hex(),
        "root_proof_hex": value.root_proof.hex(),
        "account_hex": value.account.hex(),
    }


def test_absent_reconciliation_decision_never_proves_prior_commit_rolled_back_or_unbound():
    with pytest.raises(
        runtime.AtomicRootProofCommitIndeterminate, match="PRIOR_OUTCOME_UNRESOLVED"
    ):
        runtime.PostgreSQLAtomicRootProofCommit._decision(None, _locator())


@pytest.mark.parametrize("candidate", [True, "COMMITTED", [], SimpleNamespace(state="COMMITTED")])
def test_detached_committed_flags_do_not_establish_database_decision(candidate):
    with pytest.raises(runtime.AtomicRootProofCommitCorrupt, match="EXACT_DATABASE_DECISION"):
        runtime.PostgreSQLAtomicRootProofCommit._decision(candidate, _locator())


@pytest.mark.parametrize(
    "field",
    [
        "scope_hex",
        "attempt_hex",
        "operation_hex",
        "entitlement_hex",
        "root_proof_hex",
        "account_hex",
    ],
)
@pytest.mark.parametrize("replacement", ["616e6f74686572", None])
def test_recovery_refuses_changed_or_missing_exact_winner_identity(field, replacement):
    candidate = _decision_identity()
    if replacement is None:
        del candidate[field]
    else:
        candidate[field] = replacement
    with pytest.raises(runtime.AtomicRootProofCommitConflict, match="IMMUTABLE_DECISION_IDENTITY"):
        runtime.PostgreSQLAtomicRootProofCommit._decision(candidate, _locator())


@pytest.mark.parametrize(
    "field,value",
    [
        ("payload_hex", "invalid-hex"),
        ("payload_hex", "AB"),
        ("payload_hex", "aa bb"),
        ("payload_hex", b"aa"),
        ("signature_hex", "aa" * 63),
        ("signature_hex", "aa" * 65),
        ("signature_hex", None),
        ("staged_envelope_hex", "AA"),
        ("staged_envelope_hex", True),
    ],
)
def test_recovery_refuses_corrupt_or_noncanonical_retained_proof_bytes(field, value):
    candidate = {
        **_decision_identity(),
        "payload_hex": "aa",
        "signature_hex": "aa" * 64,
        "staged_envelope_hex": "bb",
        field: value,
    }
    with pytest.raises(runtime.AtomicRootProofCommitCorrupt, match="DECISION_BYTES_INVALID"):
        runtime.PostgreSQLAtomicRootProofCommit._decision(candidate, _locator())


@pytest.mark.parametrize("missing", ["payload_hex", "signature_hex", "staged_envelope_hex"])
def test_recovery_refuses_digest_only_marker_missing_original_bytes(missing):
    candidate = {
        **_decision_identity(),
        "payload_hex": "aa",
        "signature_hex": "aa" * 64,
        "staged_envelope_hex": "bb",
    }
    del candidate[missing]
    with pytest.raises(runtime.AtomicRootProofCommitCorrupt, match="DECISION_BYTES_INVALID"):
        runtime.PostgreSQLAtomicRootProofCommit._decision(candidate, _locator())

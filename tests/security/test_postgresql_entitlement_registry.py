"""Real PostgreSQL integration tests for the PRODUCTION_LOCAL registry."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import replace
import json
import os
import uuid

import psycopg
from psycopg import sql
from psycopg.errors import InsufficientPrivilege
import pytest

import bot_core.postgresql_entitlement_registry as postgres_registry

from bot_core.entitlement_registry_contract import (
    AdminOutcome,
    BindOutcome,
    BindRequest,
    BoundBinding,
    ContractValidationError,
    EntitlementIdentity,
    EntitlementProvenance,
    ProvisionEntitlementRequest,
    RegistryReadOutcome,
    RegistrySubject,
    RevokeEntitlementRequest,
    SupersedeEntitlementRequest,
    admin_predecessor_for,
    initial_state_for,
    predecessor_for,
    supersession_states_for,
)
from bot_core.postgresql_entitlement_registry import (
    PostgreSQLConnectionConfig,
    PostgreSQLEntitlementProvisioningAdminProvider,
    PostgreSQLEntitlementRegistryProvider,
    PostgreSQLRegistryProvisioning,
    RegistryQualificationError,
    SCHEMA_IDENTITY,
    SCHEMA_VERSION,
    provision_postgresql_entitlement_registry,
)
from bot_core.root_proof_issuer_substrate import (
    EntitlementRegistryProvider,
    ProviderQualificationPolicy,
    SecurityProfile,
)


U = "018f3e70-7b5a-7c21-8b9a-0123456789ab"
BASE_DSN = os.environ.get(
    "DUDZIAN_TEST_POSTGRES_DSN",
    "host=127.0.0.1 port=55432 dbname=postgres user=postgres",
)


def _role_dsn(role: str) -> PostgreSQLConnectionConfig:
    return PostgreSQLConnectionConfig(f"host=127.0.0.1 port=55432 dbname=postgres user={role}")


@pytest.fixture(scope="module")
def registry() -> dict[str, object]:
    suffix = uuid.uuid4().hex[:10]
    setup = PostgreSQLRegistryProvisioning(
        f"er_{suffix}", f"ero_{suffix}", f"err_{suffix}", f"era_{suffix}", "td_registry"
    )
    try:
        provision_postgresql_entitlement_registry(PostgreSQLConnectionConfig(BASE_DSN), setup)
    except psycopg.OperationalError as exc:
        pytest.fail(f"real PostgreSQL integration cluster unavailable: {exc}")
    runtime = PostgreSQLEntitlementRegistryProvider(
        _role_dsn(setup.runtime_role),
        schema=setup.schema,
        environment="PRODUCTION_LOCAL",
        trust_domain=setup.trust_domain,
    )
    admin = PostgreSQLEntitlementProvisioningAdminProvider(
        _role_dsn(setup.admin_role),
        schema=setup.schema,
        environment="PRODUCTION_LOCAL",
        trust_domain=setup.trust_domain,
    )
    yield {"setup": setup, "runtime": runtime, "admin": admin}
    with psycopg.connect(BASE_DSN, autocommit=True) as conn:
        conn.execute(
            sql.SQL("DROP SCHEMA IF EXISTS {} CASCADE").format(sql.Identifier(setup.schema))
        )
        for role in (setup.runtime_role, setup.admin_role, setup.schema_owner_role):
            conn.execute(sql.SQL("DROP ROLE IF EXISTS {}").format(sql.Identifier(role)))


_counter = 0


def _subject(prefix: str = "subject") -> RegistrySubject:
    global _counter
    _counter += 1
    return RegistrySubject(f"{prefix}-{_counter}", "PRODUCTION_LOCAL", "td_registry")


def _identity(product: str = "ProductA", generation: int = 1, raw: str = U) -> EntitlementIdentity:
    return EntitlementIdentity(f"ent_{raw}", generation, "PRODUCTION_LOCAL", "td_registry", product)


def _provenance(reference: str = "immutable:provision:1") -> EntitlementProvenance:
    return EntitlementProvenance(
        "provisioner", "claimant-key", 1, "security-authority", reference, "a" * 64
    )


def _binding(tag: int = 0, generation: int = 1) -> BoundBinding:
    tail = f"{tag + 1:012x}"
    raw = f"018f3e70-7b5a-7c21-8b9a-{tail}"
    return BoundBinding(
        f"ago_{raw}",
        f"acct_{raw}",
        "b" * 64,
        generation,
        f"rpa_{raw}",
        "requester",
        "requester-key",
        1,
        "provisioner",
        "claimant-key",
        1,
        "c" * 64,
        f"immutable:request:{tag}",
        f"rpf_{raw}",
        "issuer-key",
        1,
    )


def _providers(registry: dict[str, object]):
    return registry["runtime"], registry["admin"]


@contextmanager
def _isolated_registry():
    suffix = uuid.uuid4().hex[:10]
    setup = PostgreSQLRegistryProvisioning(
        f"erx_{suffix}",
        f"erxo_{suffix}",
        f"erxr_{suffix}",
        f"erxa_{suffix}",
        "td_registry",
    )
    provision_postgresql_entitlement_registry(PostgreSQLConnectionConfig(BASE_DSN), setup)
    try:
        yield setup
    finally:
        with psycopg.connect(BASE_DSN, autocommit=True) as conn:
            conn.execute(
                sql.SQL("DROP SCHEMA IF EXISTS {} CASCADE").format(sql.Identifier(setup.schema))
            )
            for role in (setup.runtime_role, setup.admin_role, setup.schema_owner_role):
                conn.execute(sql.SQL("DROP ROLE IF EXISTS {}").format(sql.Identifier(role)))


def _provision(admin, subject: RegistrySubject, *, product: str = "ProductA", raw: str = U):
    result = admin.provision_entitlement(
        ProvisionEntitlementRequest(subject, _identity(product, raw=raw), _provenance())
    )
    assert result.outcome is AdminOutcome.COMMITTED
    assert result.state is not None
    return result.state


def test_server_durability_serializable_and_metadata(registry) -> None:
    setup = registry["setup"]
    with psycopg.connect(BASE_DSN, autocommit=True) as conn:
        row = conn.execute(
            "SELECT version(), current_setting('fsync'), current_setting('synchronous_commit')"
        ).fetchone()
        assert row is not None and "PostgreSQL 16.15" in row[0]
        assert row[1:] == ("on", "on")
        with conn.transaction():
            conn.execute("SET TRANSACTION ISOLATION LEVEL SERIALIZABLE")
            isolation = conn.execute("SELECT current_setting('transaction_isolation')").fetchone()
            assert isolation == ("serializable",)
        metadata = conn.execute(
            sql.SQL(
                "SELECT schema_identity,schema_version,schema_owner_role,"
                "schema_owner_role_oid,runtime_role,runtime_role_oid,"
                "admin_role,admin_role_oid FROM {}.metadata"
            ).format(sql.Identifier(setup.schema))
        ).fetchone()
        assert metadata is not None
        assert metadata[:2] == (SCHEMA_IDENTITY, SCHEMA_VERSION)
        assert metadata[2::2] == (
            setup.schema_owner_role,
            setup.runtime_role,
            setup.admin_role,
        )
        assert all(type(role_oid) is int and role_oid > 0 for role_oid in metadata[3::2])


def test_runtime_composition_role_only_and_secret_redaction(registry) -> None:
    runtime, admin = _providers(registry)
    assert isinstance(runtime, EntitlementRegistryProvider)
    assert not isinstance(admin, EntitlementRegistryProvider)
    assert "user=" not in repr(runtime)
    assert "user=" not in repr(runtime._connection)  # noqa: SLF001
    failures = ProviderQualificationPolicy().failures_for(
        SecurityProfile.PRODUCTION_LOCAL, runtime.identity, runtime.capabilities
    )
    assert failures == ()
    assert not hasattr(runtime, "provision_entitlement")
    assert not hasattr(runtime, "revoke_entitlement")
    assert not hasattr(runtime, "supersede_entitlement")


def test_provider_role_identity_is_self_bound_to_persisted_metadata(registry) -> None:
    setup = registry["setup"]
    runtime_arguments = {
        "schema": setup.schema,
        "environment": "PRODUCTION_LOCAL",
        "trust_domain": setup.trust_domain,
    }
    for dsn in (
        PostgreSQLConnectionConfig(BASE_DSN),
        _role_dsn(setup.schema_owner_role),
        _role_dsn(setup.admin_role),
    ):
        with pytest.raises(RegistryQualificationError):
            PostgreSQLEntitlementRegistryProvider(dsn, **runtime_arguments)
    for dsn in (
        PostgreSQLConnectionConfig(BASE_DSN),
        _role_dsn(setup.schema_owner_role),
        _role_dsn(setup.runtime_role),
    ):
        with pytest.raises(RegistryQualificationError):
            PostgreSQLEntitlementProvisioningAdminProvider(dsn, **runtime_arguments)


def test_runtime_escalation_membership_is_rejected(registry) -> None:
    setup = registry["setup"]
    with psycopg.connect(BASE_DSN, autocommit=True) as conn:
        conn.execute(
            sql.SQL("GRANT {} TO {}").format(
                sql.Identifier(setup.schema_owner_role),
                sql.Identifier(setup.runtime_role),
            )
        )
    try:
        with pytest.raises(RegistryQualificationError):
            PostgreSQLEntitlementRegistryProvider(
                _role_dsn(setup.runtime_role),
                schema=setup.schema,
                environment="PRODUCTION_LOCAL",
                trust_domain=setup.trust_domain,
            )
    finally:
        with psycopg.connect(BASE_DSN, autocommit=True) as conn:
            conn.execute(
                sql.SQL("REVOKE {} FROM {}").format(
                    sql.Identifier(setup.schema_owner_role),
                    sql.Identifier(setup.runtime_role),
                )
            )


@pytest.mark.parametrize(
    ("role_field", "columns"),
    [
        ("runtime_role", ("state",)),
        ("runtime_role", ("state", "state_integrity")),
        ("admin_role", ("state",)),
    ],
)
def test_column_level_update_acl_is_rejected(role_field: str, columns: tuple[str, ...]) -> None:
    with _isolated_registry() as setup:
        role = getattr(setup, role_field)
        column_list = sql.SQL(",").join(sql.Identifier(column) for column in columns)
        with psycopg.connect(BASE_DSN, autocommit=True) as conn:
            conn.execute(
                sql.SQL("GRANT UPDATE ({}) ON {}.history TO {}").format(
                    column_list,
                    sql.Identifier(setup.schema),
                    sql.Identifier(role),
                )
            )
        provider_class = (
            PostgreSQLEntitlementRegistryProvider
            if role_field == "runtime_role"
            else PostgreSQLEntitlementProvisioningAdminProvider
        )
        with pytest.raises(RegistryQualificationError):
            provider_class(
                _role_dsn(role),
                schema=setup.schema,
                environment="PRODUCTION_LOCAL",
                trust_domain=setup.trust_domain,
            )


def test_column_acl_would_allow_semantically_valid_bound_winner_rewrite() -> None:
    with _isolated_registry() as setup:
        runtime = PostgreSQLEntitlementRegistryProvider(
            _role_dsn(setup.runtime_role),
            schema=setup.schema,
            environment="PRODUCTION_LOCAL",
            trust_domain=setup.trust_domain,
        )
        admin = PostgreSQLEntitlementProvisioningAdminProvider(
            _role_dsn(setup.admin_role),
            schema=setup.schema,
            environment="PRODUCTION_LOCAL",
            trust_domain=setup.trust_domain,
        )
        requested = _subject("column-rewrite")
        initial = _provision(admin, requested, raw="018f3e70-7b5a-7c21-8b9a-000000000611")
        committed = runtime.compare_and_swap_bind(
            BindRequest(requested, predecessor_for(initial), _binding(611))
        )
        assert committed.outcome is BindOutcome.NEW_BIND_COMMITTED
        with psycopg.connect(BASE_DSN, autocommit=True) as conn:
            conn.execute(
                sql.SQL("GRANT UPDATE (state,state_integrity) ON {}.history TO {}").format(
                    sql.Identifier(setup.schema), sql.Identifier(setup.runtime_role)
                )
            )
        with pytest.raises(RegistryQualificationError):
            PostgreSQLEntitlementRegistryProvider(
                _role_dsn(setup.runtime_role),
                schema=setup.schema,
                environment="PRODUCTION_LOCAL",
                trust_domain=setup.trust_domain,
            )
        rewritten_account = "acct_018f3e70-7b5a-7c21-8b9a-000000000999"
        with psycopg.connect(_role_dsn(setup.runtime_role).dsn, autocommit=True) as conn:
            conn.execute(
                sql.SQL(
                    "WITH rewritten AS (SELECT lookup_handle,environment,trust_domain,revision,"
                    "jsonb_set(state,'{{binding,account_id}}',to_jsonb(%s::text)) new_state "
                    "FROM {}.history WHERE lookup_handle=%s AND revision=2) "
                    "UPDATE {}.history h SET state=r.new_state,"
                    "state_integrity=md5(r.new_state::text) FROM rewritten r "
                    "WHERE h.lookup_handle=r.lookup_handle AND h.environment=r.environment "
                    "AND h.trust_domain=r.trust_domain AND h.revision=r.revision"
                ).format(sql.Identifier(setup.schema), sql.Identifier(setup.schema)),
                (rewritten_account, requested.lookup_handle),
            )
        history = runtime.retained_history(requested)
        assert history.outcome is RegistryReadOutcome.FOUND
        assert isinstance(history.states[-1].binding, BoundBinding)
        assert history.states[-1].binding.account_id == rewritten_account


def test_outsider_schema_and_function_acl_is_rejected() -> None:
    with _isolated_registry() as setup:
        outsider = f"out_{uuid.uuid4().hex[:10]}"
        try:
            with psycopg.connect(BASE_DSN, autocommit=True) as conn:
                conn.execute(sql.SQL("CREATE ROLE {} NOLOGIN").format(sql.Identifier(outsider)))
                conn.execute(
                    sql.SQL("GRANT USAGE ON SCHEMA {} TO {}").format(
                        sql.Identifier(setup.schema), sql.Identifier(outsider)
                    )
                )
                conn.execute(
                    sql.SQL(
                        "GRANT EXECUTE ON FUNCTION {}.provision(text,text,text,text,text,json) TO {}"
                    ).format(sql.Identifier(setup.schema), sql.Identifier(outsider))
                )
            for provider_class, role in (
                (PostgreSQLEntitlementRegistryProvider, setup.runtime_role),
                (PostgreSQLEntitlementProvisioningAdminProvider, setup.admin_role),
            ):
                with pytest.raises(RegistryQualificationError):
                    provider_class(
                        _role_dsn(role),
                        schema=setup.schema,
                        environment="PRODUCTION_LOCAL",
                        trust_domain=setup.trust_domain,
                    )
        finally:
            with psycopg.connect(BASE_DSN, autocommit=True) as conn:
                conn.execute(sql.SQL("DROP OWNED BY {}").format(sql.Identifier(outsider)))
                conn.execute(sql.SQL("DROP ROLE IF EXISTS {}").format(sql.Identifier(outsider)))


@pytest.mark.parametrize(
    ("role_field", "signature"),
    [
        ("runtime_role", "append_bind(text,text,text,bigint,json)"),
        ("admin_role", "provision(text,text,text,text,text,json)"),
    ],
)
def test_function_execute_grant_option_is_rejected(role_field: str, signature: str) -> None:
    with _isolated_registry() as setup:
        role = getattr(setup, role_field)
        with psycopg.connect(BASE_DSN, autocommit=True) as conn:
            conn.execute(
                sql.SQL("GRANT EXECUTE ON FUNCTION {}.{} TO {} WITH GRANT OPTION").format(
                    sql.Identifier(setup.schema),
                    sql.SQL(signature),
                    sql.Identifier(role),
                )
            )
        provider_class = (
            PostgreSQLEntitlementRegistryProvider
            if role_field == "runtime_role"
            else PostgreSQLEntitlementProvisioningAdminProvider
        )
        with pytest.raises(RegistryQualificationError):
            provider_class(
                _role_dsn(role),
                schema=setup.schema,
                environment="PRODUCTION_LOCAL",
                trust_domain=setup.trust_domain,
            )


@pytest.mark.parametrize("role_field", ["runtime_role", "admin_role"])
@pytest.mark.parametrize("superuser", [False, True])
def test_arbitrary_membership_and_set_role_escalation_are_rejected(
    role_field: str, superuser: bool
) -> None:
    with _isolated_registry() as setup:
        authority_role = getattr(setup, role_field)
        helper = f"esc_{uuid.uuid4().hex[:10]}"
        provider_class = (
            PostgreSQLEntitlementRegistryProvider
            if role_field == "runtime_role"
            else PostgreSQLEntitlementProvisioningAdminProvider
        )
        try:
            with psycopg.connect(BASE_DSN, autocommit=True) as conn:
                attributes = sql.SQL("SUPERUSER") if superuser else sql.SQL("NOSUPERUSER")
                conn.execute(
                    sql.SQL("CREATE ROLE {} {} NOLOGIN").format(sql.Identifier(helper), attributes)
                )
                conn.execute(
                    sql.SQL("GRANT {} TO {} WITH INHERIT FALSE, SET TRUE").format(
                        sql.Identifier(helper), sql.Identifier(authority_role)
                    )
                )
            with pytest.raises(RegistryQualificationError):
                provider_class(
                    _role_dsn(authority_role),
                    schema=setup.schema,
                    environment="PRODUCTION_LOCAL",
                    trust_domain=setup.trust_domain,
                )
        finally:
            with psycopg.connect(BASE_DSN, autocommit=True) as conn:
                conn.execute(
                    sql.SQL("REVOKE {} FROM {}").format(
                        sql.Identifier(helper), sql.Identifier(authority_role)
                    )
                )
                conn.execute(sql.SQL("DROP ROLE IF EXISTS {}").format(sql.Identifier(helper)))


def test_session_user_current_user_laundering_is_rejected(registry) -> None:
    setup = registry["setup"]
    laundering_dsn = PostgreSQLConnectionConfig(
        f"{BASE_DSN} options='-c role={setup.runtime_role}'"
    )
    with psycopg.connect(laundering_dsn.dsn) as conn:
        identities = conn.execute("SELECT session_user,current_user").fetchone()
        assert identities is not None
        assert identities[0] != identities[1]
        assert identities[1] == setup.runtime_role
    with pytest.raises(RegistryQualificationError):
        PostgreSQLEntitlementRegistryProvider(
            laundering_dsn,
            schema=setup.schema,
            environment="PRODUCTION_LOCAL",
            trust_domain=setup.trust_domain,
        )


def test_rolinherit_true_is_rejected(registry) -> None:
    setup = registry["setup"]
    with psycopg.connect(BASE_DSN, autocommit=True) as conn:
        conn.execute(sql.SQL("ALTER ROLE {} INHERIT").format(sql.Identifier(setup.runtime_role)))
    try:
        with pytest.raises(RegistryQualificationError):
            PostgreSQLEntitlementRegistryProvider(
                _role_dsn(setup.runtime_role),
                schema=setup.schema,
                environment="PRODUCTION_LOCAL",
                trust_domain=setup.trust_domain,
            )
    finally:
        with psycopg.connect(BASE_DSN, autocommit=True) as conn:
            conn.execute(
                sql.SQL("ALTER ROLE {} NOINHERIT").format(sql.Identifier(setup.runtime_role))
            )


def test_unexpected_acl_cleanup_restores_qualification() -> None:
    with _isolated_registry() as setup:
        outsider = f"acl_{uuid.uuid4().hex[:10]}"
        try:
            with psycopg.connect(BASE_DSN, autocommit=True) as conn:
                conn.execute(sql.SQL("CREATE ROLE {} NOLOGIN").format(sql.Identifier(outsider)))
                conn.execute(
                    sql.SQL("GRANT USAGE ON SCHEMA {} TO {}").format(
                        sql.Identifier(setup.schema), sql.Identifier(outsider)
                    )
                )
            with pytest.raises(RegistryQualificationError):
                PostgreSQLEntitlementRegistryProvider(
                    _role_dsn(setup.runtime_role),
                    schema=setup.schema,
                    environment="PRODUCTION_LOCAL",
                    trust_domain=setup.trust_domain,
                )
            with psycopg.connect(BASE_DSN, autocommit=True) as conn:
                conn.execute(
                    sql.SQL("REVOKE USAGE ON SCHEMA {} FROM {}").format(
                        sql.Identifier(setup.schema), sql.Identifier(outsider)
                    )
                )
            PostgreSQLEntitlementRegistryProvider(
                _role_dsn(setup.runtime_role),
                schema=setup.schema,
                environment="PRODUCTION_LOCAL",
                trust_domain=setup.trust_domain,
            )
            PostgreSQLEntitlementProvisioningAdminProvider(
                _role_dsn(setup.admin_role),
                schema=setup.schema,
                environment="PRODUCTION_LOCAL",
                trust_domain=setup.trust_domain,
            )
        finally:
            with psycopg.connect(BASE_DSN, autocommit=True) as conn:
                conn.execute(sql.SQL("DROP OWNED BY {}").format(sql.Identifier(outsider)))
                conn.execute(sql.SQL("DROP ROLE IF EXISTS {}").format(sql.Identifier(outsider)))


@pytest.mark.parametrize("authority_field", ["runtime_role", "admin_role", "schema_owner_role"])
@pytest.mark.parametrize("inherit_option", [False, True])
def test_incoming_authority_membership_and_set_role_are_rejected(
    authority_field: str, inherit_option: bool
) -> None:
    with _isolated_registry() as setup:
        authority = getattr(setup, authority_field)
        outsider = f"mem_{uuid.uuid4().hex[:10]}"
        try:
            with psycopg.connect(BASE_DSN, autocommit=True) as conn:
                conn.execute(
                    sql.SQL("CREATE ROLE {} LOGIN NOINHERIT").format(sql.Identifier(outsider))
                )
                inherit = sql.SQL("TRUE") if inherit_option else sql.SQL("FALSE")
                conn.execute(
                    sql.SQL("GRANT {} TO {} WITH INHERIT {}, SET TRUE").format(
                        sql.Identifier(authority), sql.Identifier(outsider), inherit
                    )
                )
            outsider_dsn = f"host=127.0.0.1 port=55432 dbname=postgres user={outsider}"
            with psycopg.connect(outsider_dsn) as conn:
                assert conn.execute("SELECT current_user").fetchone() == (outsider,)
                conn.execute(sql.SQL("SET ROLE {}").format(sql.Identifier(authority)))
                assert conn.execute("SELECT current_user").fetchone() == (authority,)
            for provider_class, role in (
                (PostgreSQLEntitlementRegistryProvider, setup.runtime_role),
                (PostgreSQLEntitlementProvisioningAdminProvider, setup.admin_role),
            ):
                with pytest.raises(RegistryQualificationError):
                    provider_class(
                        _role_dsn(role),
                        schema=setup.schema,
                        environment="PRODUCTION_LOCAL",
                        trust_domain=setup.trust_domain,
                    )
        finally:
            with psycopg.connect(BASE_DSN, autocommit=True) as conn:
                conn.execute(
                    sql.SQL("REVOKE {} FROM {}").format(
                        sql.Identifier(authority), sql.Identifier(outsider)
                    )
                )
                conn.execute(sql.SQL("DROP ROLE IF EXISTS {}").format(sql.Identifier(outsider)))


def test_direct_runtime_malformed_bind_payloads_are_atomic() -> None:
    from bot_core.postgresql_entitlement_registry import _state_json  # noqa: PLC0415

    with _isolated_registry() as setup:
        runtime = PostgreSQLEntitlementRegistryProvider(
            _role_dsn(setup.runtime_role),
            schema=setup.schema,
            environment="PRODUCTION_LOCAL",
            trust_domain=setup.trust_domain,
        )
        admin = PostgreSQLEntitlementProvisioningAdminProvider(
            _role_dsn(setup.admin_role),
            schema=setup.schema,
            environment="PRODUCTION_LOCAL",
            trust_domain=setup.trust_domain,
        )
        requested = _subject("malformed-bind")
        initial = _provision(admin, requested, raw="018f3e70-7b5a-7c21-8b9a-000000000701")
        valid_successor = replace(
            initial,
            binding=_binding(701),
            authoritative_state_revision=2,
            predecessor_revision=1,
        )
        valid = _state_json(valid_successor)
        extra_top = {**deepcopy(valid), "unexpected": True}
        extra_nested = deepcopy(valid)
        extra_nested["binding"]["unexpected"] = "field"
        wrong_revision = deepcopy(valid)
        wrong_revision["authoritative_state_revision"] = "2"
        wrong_generation = deepcopy(valid)
        wrong_generation["identity"]["entitlement_generation"] = True
        wrong_subject = deepcopy(valid)
        wrong_subject["subject"] = []
        wrong_binding = deepcopy(valid)
        wrong_binding["binding"] = "BOUND"
        wrong_lifecycle = deepcopy(valid)
        wrong_lifecycle["lifecycle"] = None
        payloads = (
            ({}, True),
            (None, True),
            (None, False),
            ({"binding": {"kind": "BOUND"}}, True),
            (extra_top, True),
            (extra_nested, True),
            (wrong_revision, True),
            (wrong_generation, True),
            (wrong_subject, True),
            (wrong_binding, True),
            (wrong_lifecycle, True),
        )
        before = runtime.retained_history(requested)
        for payload, as_json in payloads:
            with psycopg.connect(_role_dsn(setup.runtime_role).dsn, autocommit=True) as conn:
                with pytest.raises(psycopg.Error) as raised:
                    conn.execute(
                        sql.SQL("SELECT {}.append_bind(%s,%s,%s,%s,%s)").format(
                            sql.Identifier(setup.schema)
                        ),
                        (
                            requested.lookup_handle,
                            requested.environment,
                            requested.trust_domain,
                            1,
                            psycopg.types.json.Json(payload) if as_json else None,
                        ),
                    )
                assert raised.value.sqlstate == "22023"
            assert runtime.retained_history(requested) == before


def _replace_json_path(payload: dict[str, object], path: tuple[str, ...], value: object) -> None:
    target: dict[str, object] = payload
    for key in path[:-1]:
        nested = target[key]
        assert isinstance(nested, dict)
        target = nested
    target[path[-1]] = value


def test_python_sql_text_validation_differential_parity() -> None:
    from bot_core.postgresql_entitlement_registry import (  # noqa: PLC0415
        _state_from_json,
        _state_json,
    )

    python_whitespace = tuple(
        chr(codepoint) for codepoint in range(0x110000) if chr(codepoint).isspace()
    )
    assert tuple(ord(value) for value in python_whitespace) == (
        9,
        10,
        11,
        12,
        13,
        28,
        29,
        30,
        31,
        32,
        133,
        160,
        5760,
        8192,
        8193,
        8194,
        8195,
        8196,
        8197,
        8198,
        8199,
        8200,
        8201,
        8202,
        8232,
        8233,
        8239,
        8287,
        12288,
    )
    paths = (
        ("subject", "lookup_handle"),
        ("subject", "environment"),
        ("subject", "trust_domain"),
        ("identity", "environment"),
        ("identity", "trust_domain"),
        ("identity", "product_scope"),
        ("provenance", "provisioning_principal_id"),
        ("provenance", "claimant_key_id"),
        ("provenance", "creation_authority_identity"),
        ("provenance", "authenticated_creation_reference"),
        ("binding", "requester_principal_id"),
        ("binding", "requester_key_id"),
        ("binding", "provisioning_principal_id"),
        ("binding", "claimant_key_id"),
        ("binding", "signed_request_canonical_bytes_reference"),
        ("binding", "issuer_signing_credential_id"),
    )
    with _isolated_registry() as setup:
        runtime = PostgreSQLEntitlementRegistryProvider(
            _role_dsn(setup.runtime_role),
            schema=setup.schema,
            environment="PRODUCTION_LOCAL",
            trust_domain=setup.trust_domain,
        )
        admin = PostgreSQLEntitlementProvisioningAdminProvider(
            _role_dsn(setup.admin_role),
            schema=setup.schema,
            environment="PRODUCTION_LOCAL",
            trust_domain=setup.trust_domain,
        )
        requested = _subject("text-parity")
        initial = _provision(admin, requested, raw="018f3e70-7b5a-7c21-8b9a-000000000711")
        successor = replace(
            initial,
            binding=_binding(711),
            authoritative_state_revision=2,
            predecessor_revision=1,
        )
        canonical = _state_json(successor)
        mutations = [
            (("binding", "requester_principal_id"), whitespace) for whitespace in python_whitespace
        ] + [(path, "".join(python_whitespace)) for path in paths]
        before = runtime.retained_history(requested)
        for path, value in mutations:
            payload = deepcopy(canonical)
            _replace_json_path(payload, path, value)
            with pytest.raises(ContractValidationError):
                _state_from_json(payload)
            with psycopg.connect(_role_dsn(setup.runtime_role).dsn, autocommit=True) as conn:
                with pytest.raises(psycopg.Error) as raised:
                    conn.execute(
                        sql.SQL("SELECT {}.append_bind(%s,%s,%s,%s,%s)").format(
                            sql.Identifier(setup.schema)
                        ),
                        (*_subject_params_for_test(requested), 1, psycopg.types.json.Json(payload)),
                    )
                assert raised.value.sqlstate == "22023"
        assert runtime.retained_history(requested) == before


def _subject_params_for_test(subject: RegistrySubject) -> tuple[str, str, str]:
    return subject.lookup_handle, subject.environment, subject.trust_domain


def test_python_sql_nontext_validation_differential_parity() -> None:
    from bot_core.postgresql_entitlement_registry import (  # noqa: PLC0415
        _state_from_json,
        _state_json,
    )

    with _isolated_registry() as setup:
        runtime = PostgreSQLEntitlementRegistryProvider(
            _role_dsn(setup.runtime_role),
            schema=setup.schema,
            environment="PRODUCTION_LOCAL",
            trust_domain=setup.trust_domain,
        )
        admin = PostgreSQLEntitlementProvisioningAdminProvider(
            _role_dsn(setup.admin_role),
            schema=setup.schema,
            environment="PRODUCTION_LOCAL",
            trust_domain=setup.trust_domain,
        )
        requested = _subject("nontext-parity")
        initial = _provision(admin, requested, raw="018f3e70-7b5a-7c21-8b9a-000000000712")
        canonical = _state_json(
            replace(
                initial,
                binding=_binding(712),
                authoritative_state_revision=2,
                predecessor_revision=1,
            )
        )
        mutations = (
            (("authoritative_state_revision",), 0),
            (("authoritative_state_revision",), True),
            (("identity", "entitlement_generation"), -1),
            (("identity", "entitlement_generation"), True),
            (("identity", "bootstrap_entitlement_id"), "ENT_018f3e70-7b5a-7c21-8b9a-000000000712"),
            (("binding", "account_id"), "ago_018f3e70-7b5a-7c21-8b9a-000000000712"),
            (("provenance", "authenticated_creation_digest_sha256"), "A" * 64),
            (("provenance", "authenticated_creation_digest_sha256"), "a" * 63),
            (("binding", "kind"), "UNKNOWN"),
            (("lifecycle",), "UNKNOWN"),
        )
        before = runtime.retained_history(requested)
        for path, value in mutations:
            payload = deepcopy(canonical)
            _replace_json_path(payload, path, value)
            with pytest.raises((ContractValidationError, ValueError)):
                _state_from_json(payload)
            with psycopg.connect(_role_dsn(setup.runtime_role).dsn, autocommit=True) as conn:
                with pytest.raises(psycopg.Error):
                    conn.execute(
                        sql.SQL("SELECT {}.append_bind(%s,%s,%s,%s,%s)").format(
                            sql.Identifier(setup.schema)
                        ),
                        (*_subject_params_for_test(requested), 1, psycopg.types.json.Json(payload)),
                    )
        assert runtime.retained_history(requested) == before


def test_direct_json_numeric_lexical_forms_are_rejected_for_every_integer_field() -> None:
    lexical_forms = ("1e0", "1E0", "1.0", "1.00", "1e1")
    integer_paths = (
        ("authoritative_state_revision",),
        ("predecessor_revision",),
        ("identity", "entitlement_generation"),
        ("provenance", "claimant_key_version"),
        ("binding", "entitlement_generation"),
        ("binding", "requester_key_version"),
        ("binding", "claimant_key_version"),
        ("binding", "issuer_signing_key_version"),
    )
    with psycopg.connect(BASE_DSN) as conn:
        observed = {
            value: conn.execute(
                "SELECT (%s::json #> '{value}')::text", ('{"value":' + value + "}",)
            ).fetchone()[0]
            for value in lexical_forms
        }
    assert observed == {value: value for value in lexical_forms}
    with _isolated_registry() as setup:
        admin = PostgreSQLEntitlementProvisioningAdminProvider(
            _role_dsn(setup.admin_role),
            schema=setup.schema,
            environment="PRODUCTION_LOCAL",
            trust_domain=setup.trust_domain,
        )
        subject = _subject("numeric-lexical")
        initial = _provision(admin, subject, raw="018f3e70-7b5a-7c21-8b9a-000000000713")
        canonical = postgres_registry._state_json(
            replace(
                initial,
                binding=_binding(713),
                authoritative_state_revision=2,
                predecessor_revision=1,
            )
        )
        for path in integer_paths:
            for lexical in lexical_forms:
                payload = deepcopy(canonical)
                _replace_json_path(payload, path, 777777)
                raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
                assert raw.count("777777") == 1
                raw = raw.replace("777777", lexical)
                with psycopg.connect(_role_dsn(setup.runtime_role).dsn, autocommit=True) as conn:
                    with pytest.raises(psycopg.Error) as raised:
                        conn.execute(
                            sql.SQL("SELECT {}.append_bind(%s,%s,%s,%s,%s::json)").format(
                                sql.Identifier(setup.schema)
                            ),
                            (*_subject_params_for_test(subject), 1, raw),
                        )
                    assert raised.value.sqlstate == "22023"


def test_direct_admin_malformed_provision_payloads_have_zero_side_effects() -> None:
    with _isolated_registry() as setup:
        handle = "malformed-direct-provision"
        payloads = (
            ({}, True),
            (None, True),
            (None, False),
            ({"subject": {"lookup_handle": handle}}, True),
        )
        with psycopg.connect(_role_dsn(setup.admin_role).dsn, autocommit=True) as conn:
            for payload, as_json in payloads:
                with pytest.raises(psycopg.Error) as raised:
                    conn.execute(
                        sql.SQL("SELECT {}.provision(%s,%s,%s,%s,%s,%s)").format(
                            sql.Identifier(setup.schema)
                        ),
                        (
                            handle,
                            "PRODUCTION_LOCAL",
                            setup.trust_domain,
                            "ProductA",
                            "ent_018f3e70-7b5a-7c21-8b9a-000000000702",
                            psycopg.types.json.Json(payload) if as_json else None,
                        ),
                    )
                assert raised.value.sqlstate == "22023"
        with psycopg.connect(BASE_DSN) as conn:
            counts = conn.execute(
                sql.SQL(
                    "SELECT (SELECT count(*) FROM {}.lineages WHERE lookup_handle=%s),"
                    "(SELECT count(*) FROM {}.history WHERE lookup_handle=%s)"
                ).format(sql.Identifier(setup.schema), sql.Identifier(setup.schema)),
                (handle, handle),
            ).fetchone()
            assert counts == (0, 0)


def test_direct_admin_malformed_revoke_and_supersede_are_atomic() -> None:
    with _isolated_registry() as setup:
        runtime = PostgreSQLEntitlementRegistryProvider(
            _role_dsn(setup.runtime_role),
            schema=setup.schema,
            environment="PRODUCTION_LOCAL",
            trust_domain=setup.trust_domain,
        )
        admin = PostgreSQLEntitlementProvisioningAdminProvider(
            _role_dsn(setup.admin_role),
            schema=setup.schema,
            environment="PRODUCTION_LOCAL",
            trust_domain=setup.trust_domain,
        )
        requested = _subject("malformed-admin")
        _provision(admin, requested, raw="018f3e70-7b5a-7c21-8b9a-000000000703")
        before = runtime.retained_history(requested)
        calls = (
            ("revoke", ({},)),
            ("revoke", ({"lifecycle": "REVOKED"},)),
            ("supersede", ({}, {})),
            ("supersede", ({"lifecycle": "SUPERSEDED"}, {"lifecycle": "ACTIVE"})),
        )
        with psycopg.connect(_role_dsn(setup.admin_role).dsn, autocommit=True) as conn:
            for function, payloads in calls:
                placeholders = "%s,%s,%s,%s,%s" if function == "revoke" else "%s,%s,%s,%s,%s,%s"
                parameters = (
                    requested.lookup_handle,
                    requested.environment,
                    requested.trust_domain,
                    1,
                    *(psycopg.types.json.Json(payload) for payload in payloads),
                )
                with pytest.raises(psycopg.Error) as raised:
                    conn.execute(
                        sql.SQL("SELECT {}.{}(" + placeholders + ")").format(
                            sql.Identifier(setup.schema), sql.Identifier(function)
                        ),
                        parameters,
                    )
                assert raised.value.sqlstate == "22023"
                assert runtime.retained_history(requested) == before


@pytest.mark.parametrize(
    "mutation",
    ["drop_not_null", "change_type", "drop_default"],
)
def test_physical_column_shape_drift_is_rejected(mutation: str) -> None:
    with _isolated_registry() as setup:
        with psycopg.connect(BASE_DSN, autocommit=True) as conn:
            if mutation == "drop_not_null":
                statement = sql.SQL(
                    "ALTER TABLE {}.history ALTER COLUMN state_integrity DROP NOT NULL"
                ).format(sql.Identifier(setup.schema))
            elif mutation == "change_type":
                statement = sql.SQL(
                    "ALTER TABLE {}.lineages ALTER COLUMN product_scope TYPE varchar"
                ).format(sql.Identifier(setup.schema))
            else:
                statement = sql.SQL(
                    "ALTER TABLE {}.metadata ALTER COLUMN singleton DROP DEFAULT"
                ).format(sql.Identifier(setup.schema))
            conn.execute(statement)
        for provider_class, role in (
            (PostgreSQLEntitlementRegistryProvider, setup.runtime_role),
            (PostgreSQLEntitlementProvisioningAdminProvider, setup.admin_role),
        ):
            with pytest.raises(RegistryQualificationError):
                provider_class(
                    _role_dsn(role),
                    schema=setup.schema,
                    environment="PRODUCTION_LOCAL",
                    trust_domain=setup.trust_domain,
                )


@pytest.mark.parametrize("relation", ["metadata", "history", "lineages"])
def test_unlogged_registry_relation_is_rejected(relation: str) -> None:
    with _isolated_registry() as setup:
        with psycopg.connect(BASE_DSN, autocommit=True) as conn:
            if relation == "lineages":
                conn.execute(
                    sql.SQL("ALTER TABLE {}.history SET UNLOGGED").format(
                        sql.Identifier(setup.schema)
                    )
                )
            conn.execute(
                sql.SQL("ALTER TABLE {}.{} SET UNLOGGED").format(
                    sql.Identifier(setup.schema), sql.Identifier(relation)
                )
            )
        for provider_class, role in (
            (PostgreSQLEntitlementRegistryProvider, setup.runtime_role),
            (PostgreSQLEntitlementProvisioningAdminProvider, setup.admin_role),
        ):
            with pytest.raises(RegistryQualificationError):
                provider_class(
                    _role_dsn(role),
                    schema=setup.schema,
                    environment="PRODUCTION_LOCAL",
                    trust_domain=setup.trust_domain,
                )


def test_registry_inheritance_edge_is_rejected() -> None:
    with _isolated_registry() as setup:
        parent = f"parent_{uuid.uuid4().hex[:10]}"
        try:
            with psycopg.connect(BASE_DSN, autocommit=True) as conn:
                conn.execute(
                    sql.SQL("CREATE TABLE public.{} (lookup_handle text)").format(
                        sql.Identifier(parent)
                    )
                )
                conn.execute(
                    sql.SQL("ALTER TABLE {}.history INHERIT public.{}").format(
                        sql.Identifier(setup.schema), sql.Identifier(parent)
                    )
                )
            with pytest.raises(RegistryQualificationError):
                PostgreSQLEntitlementRegistryProvider(
                    _role_dsn(setup.runtime_role),
                    schema=setup.schema,
                    environment="PRODUCTION_LOCAL",
                    trust_domain=setup.trust_domain,
                )
        finally:
            with psycopg.connect(BASE_DSN, autocommit=True) as conn:
                conn.execute(
                    sql.SQL("DROP TABLE IF EXISTS public.{} CASCADE").format(sql.Identifier(parent))
                )


def test_rls_false_not_found_attack_is_rejected_at_qualification() -> None:
    with _isolated_registry() as setup:
        runtime = PostgreSQLEntitlementRegistryProvider(
            _role_dsn(setup.runtime_role),
            schema=setup.schema,
            environment="PRODUCTION_LOCAL",
            trust_domain=setup.trust_domain,
        )
        admin = PostgreSQLEntitlementProvisioningAdminProvider(
            _role_dsn(setup.admin_role),
            schema=setup.schema,
            environment="PRODUCTION_LOCAL",
            trust_domain=setup.trust_domain,
        )
        requested = _subject("rls-hidden")
        _provision(admin, requested, raw="018f3e70-7b5a-7c21-8b9a-000000000721")
        assert runtime.retained_history(requested).outcome is RegistryReadOutcome.FOUND
        with psycopg.connect(_role_dsn(setup.schema_owner_role).dsn, autocommit=True) as conn:
            for relation in ("history", "lineages"):
                conn.execute(
                    sql.SQL("ALTER TABLE {}.{} ENABLE ROW LEVEL SECURITY").format(
                        sql.Identifier(setup.schema), sql.Identifier(relation)
                    )
                )
                conn.execute(
                    sql.SQL(
                        "CREATE POLICY hide_all ON {}.{} FOR SELECT TO {} USING (false)"
                    ).format(
                        sql.Identifier(setup.schema),
                        sql.Identifier(relation),
                        sql.Identifier(setup.runtime_role),
                    )
                )
        with psycopg.connect(_role_dsn(setup.runtime_role).dsn) as conn:
            assert conn.execute(
                sql.SQL("SELECT count(*) FROM {}.history").format(sql.Identifier(setup.schema))
            ).fetchone() == (0,)
        assert runtime.retained_history(requested).outcome is RegistryReadOutcome.NOT_FOUND
        with pytest.raises(RegistryQualificationError):
            PostgreSQLEntitlementRegistryProvider(
                _role_dsn(setup.runtime_role),
                schema=setup.schema,
                environment="PRODUCTION_LOCAL",
                trust_domain=setup.trust_domain,
            )


def test_force_rls_without_policy_is_rejected() -> None:
    with _isolated_registry() as setup:
        with psycopg.connect(_role_dsn(setup.schema_owner_role).dsn, autocommit=True) as conn:
            conn.execute(
                sql.SQL("ALTER TABLE {}.history FORCE ROW LEVEL SECURITY").format(
                    sql.Identifier(setup.schema)
                )
            )
        with pytest.raises(RegistryQualificationError):
            PostgreSQLEntitlementRegistryProvider(
                _role_dsn(setup.runtime_role),
                schema=setup.schema,
                environment="PRODUCTION_LOCAL",
                trust_domain=setup.trust_domain,
            )


def test_unexpected_user_trigger_is_rejected_and_cleanup_restores_qualification() -> None:
    with _isolated_registry() as setup:
        function_name = f"trigger_{uuid.uuid4().hex[:10]}"
        with psycopg.connect(BASE_DSN, autocommit=True) as conn:
            conn.execute(
                sql.SQL(
                    "CREATE FUNCTION public.{}() RETURNS trigger LANGUAGE plpgsql "
                    "AS 'BEGIN RETURN NEW; END'"
                ).format(sql.Identifier(function_name))
            )
            conn.execute(
                sql.SQL(
                    "CREATE TRIGGER unexpected BEFORE INSERT ON {}.history "
                    "FOR EACH ROW EXECUTE FUNCTION public.{}()"
                ).format(sql.Identifier(setup.schema), sql.Identifier(function_name))
            )
        with pytest.raises(RegistryQualificationError):
            PostgreSQLEntitlementRegistryProvider(
                _role_dsn(setup.runtime_role),
                schema=setup.schema,
                environment="PRODUCTION_LOCAL",
                trust_domain=setup.trust_domain,
            )
        with psycopg.connect(BASE_DSN, autocommit=True) as conn:
            conn.execute(
                sql.SQL("DROP TRIGGER unexpected ON {}.history").format(
                    sql.Identifier(setup.schema)
                )
            )
            conn.execute(sql.SQL("DROP FUNCTION public.{}()").format(sql.Identifier(function_name)))
        PostgreSQLEntitlementRegistryProvider(
            _role_dsn(setup.runtime_role),
            schema=setup.schema,
            environment="PRODUCTION_LOCAL",
            trust_domain=setup.trust_domain,
        )


def test_unexpected_rewrite_rule_is_rejected() -> None:
    with _isolated_registry() as setup:
        with psycopg.connect(_role_dsn(setup.schema_owner_role).dsn, autocommit=True) as conn:
            conn.execute(
                sql.SQL(
                    "CREATE RULE unexpected AS ON INSERT TO {}.history DO ALSO NOTIFY registry_rule"
                ).format(sql.Identifier(setup.schema))
            )
        with pytest.raises(RegistryQualificationError):
            PostgreSQLEntitlementRegistryProvider(
                _role_dsn(setup.runtime_role),
                schema=setup.schema,
                environment="PRODUCTION_LOCAL",
                trust_domain=setup.trust_domain,
            )


@pytest.mark.parametrize("version", [3, 4])
def test_prior_function_semantics_schema_version_is_rejected(version: int) -> None:
    with _isolated_registry() as setup:
        with psycopg.connect(BASE_DSN, autocommit=True) as conn:
            conn.execute(
                sql.SQL("UPDATE {}.metadata SET schema_version=%s,storage_family=%s").format(
                    sql.Identifier(setup.schema)
                ),
                (version, f"POSTGRESQL_APPEND_ONLY_V{version}"),
            )
        for provider_class, role in (
            (PostgreSQLEntitlementRegistryProvider, setup.runtime_role),
            (PostgreSQLEntitlementProvisioningAdminProvider, setup.admin_role),
        ):
            with pytest.raises(RegistryQualificationError):
                provider_class(
                    _role_dsn(role),
                    schema=setup.schema,
                    environment="PRODUCTION_LOCAL",
                    trust_domain=setup.trust_domain,
                )


def _actual_function_source(setup: PostgreSQLRegistryProvisioning, signature: str) -> str:
    with psycopg.connect(BASE_DSN) as conn:
        row = conn.execute(
            "SELECT prosrc FROM pg_catalog.pg_proc WHERE oid=to_regprocedure(%s)",
            (f"{setup.schema}.{signature}",),
        ).fetchone()
    assert row is not None and type(row[0]) is str
    return row[0]


def _persist_actual_function_fingerprint(
    setup: PostgreSQLRegistryProvisioning, signature: str
) -> str:
    source = _actual_function_source(setup, signature)
    fingerprint = postgres_registry._function_source_fingerprint(signature, source)
    with psycopg.connect(_role_dsn(setup.schema_owner_role).dsn, autocommit=True) as conn:
        manifest = conn.execute(
            sql.SQL("SELECT function_definition_sha256 FROM {}.metadata").format(
                sql.Identifier(setup.schema)
            )
        ).fetchone()[0]
        manifest[signature] = fingerprint
        conn.execute(
            sql.SQL("UPDATE {}.metadata SET function_definition_sha256=%s").format(
                sql.Identifier(setup.schema)
            ),
            (psycopg.types.json.Jsonb(manifest),),
        )
    return fingerprint


def test_clean_function_identity_is_three_way_bound_to_reviewed_code() -> None:
    with _isolated_registry() as setup:
        expected_sources = postgres_registry._reviewed_function_sources(setup)
        expected_manifest = postgres_registry._reviewed_function_manifest(setup)
        with psycopg.connect(_role_dsn(setup.runtime_role).dsn) as conn:
            persisted = conn.execute(
                sql.SQL("SELECT function_definition_sha256 FROM {}.metadata").format(
                    sql.Identifier(setup.schema)
                )
            ).fetchone()[0]
        assert persisted == expected_manifest
        for signature, expected_source in expected_sources.items():
            actual = _actual_function_source(setup, signature)
            assert actual == expected_source
            assert (
                postgres_registry._function_source_fingerprint(signature, actual)
                == expected_manifest[signature]
            )


def test_provisioning_rolls_back_when_actual_function_differs_from_code_anchor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    suffix = uuid.uuid4().hex[:10]
    setup = PostgreSQLRegistryProvisioning(
        f"erf_{suffix}",
        f"erfo_{suffix}",
        f"erfr_{suffix}",
        f"erfa_{suffix}",
        "td_registry",
    )
    original = postgres_registry._reviewed_function_sources

    def mismatching_sources(config: PostgreSQLRegistryProvisioning) -> dict[str, str]:
        sources = original(config)
        sources["append_bind(text,text,text,bigint,json)"] += "\n-- reviewed mismatch"
        return sources

    monkeypatch.setattr(postgres_registry, "_reviewed_function_sources", mismatching_sources)
    with pytest.raises(RegistryQualificationError):
        provision_postgresql_entitlement_registry(PostgreSQLConnectionConfig(BASE_DSN), setup)
    with psycopg.connect(BASE_DSN) as conn:
        assert conn.execute("SELECT to_regnamespace(%s)", (setup.schema,)).fetchone() == (None,)
        assert conn.execute(
            "SELECT count(*) FROM pg_catalog.pg_roles WHERE rolname=ANY(%s)",
            ([setup.schema_owner_role, setup.runtime_role, setup.admin_role],),
        ).fetchone() == (0,)


def test_metadata_only_function_manifest_tamper_is_rejected() -> None:
    with _isolated_registry() as setup:
        with psycopg.connect(_role_dsn(setup.schema_owner_role).dsn, autocommit=True) as conn:
            conn.execute(
                sql.SQL(
                    "UPDATE {}.metadata SET function_definition_sha256="
                    "jsonb_set(function_definition_sha256,ARRAY[%s],to_jsonb(%s::text))"
                ).format(sql.Identifier(setup.schema)),
                ("append_bind(text,text,text,bigint,json)", "0" * 64),
            )
        with pytest.raises(RegistryQualificationError):
            PostgreSQLEntitlementRegistryProvider(
                _role_dsn(setup.runtime_role),
                schema=setup.schema,
                environment="PRODUCTION_LOCAL",
                trust_domain=setup.trust_domain,
            )


def test_body_only_function_tamper_is_rejected() -> None:
    with _isolated_registry() as setup:
        with psycopg.connect(_role_dsn(setup.schema_owner_role).dsn, autocommit=True) as conn:
            conn.execute(
                sql.SQL(
                    "CREATE OR REPLACE FUNCTION {}.validate_state(p_state json) RETURNS void "
                    "LANGUAGE plpgsql SECURITY DEFINER SET search_path=pg_catalog "
                    "AS 'BEGIN RETURN; END'"
                ).format(sql.Identifier(setup.schema))
            )
        with pytest.raises(RegistryQualificationError):
            PostgreSQLEntitlementRegistryProvider(
                _role_dsn(setup.runtime_role),
                schema=setup.schema,
                environment="PRODUCTION_LOCAL",
                trust_domain=setup.trust_domain,
            )


@pytest.mark.parametrize(
    ("signature", "replacement"),
    [
        (
            "provision(text,text,text,text,text,json)",
            "CREATE OR REPLACE FUNCTION {schema}.provision(p_handle text,p_environment text,p_trust text,p_product text,p_entitlement text,p_state json) "
            "RETURNS void LANGUAGE plpgsql SECURITY DEFINER SET search_path=pg_catalog "
            "AS 'BEGIN DELETE FROM {schema}.history; END'",
        ),
        (
            "validate_state(json)",
            "CREATE OR REPLACE FUNCTION {schema}.validate_state(p_state json) RETURNS void "
            "LANGUAGE plpgsql SECURITY DEFINER SET search_path=pg_catalog "
            "AS 'BEGIN RETURN; END'",
        ),
    ],
)
def test_coordinated_admin_or_validator_body_and_manifest_tamper_is_rejected(
    signature: str, replacement: str
) -> None:
    with _isolated_registry() as setup:
        with psycopg.connect(_role_dsn(setup.schema_owner_role).dsn, autocommit=True) as conn:
            conn.execute(sql.SQL(replacement).format(schema=sql.Identifier(setup.schema)))
        fingerprint = _persist_actual_function_fingerprint(setup, signature)
        with psycopg.connect(BASE_DSN) as conn:
            persisted = conn.execute(
                sql.SQL("SELECT function_definition_sha256->>%s FROM {}.metadata").format(
                    sql.Identifier(setup.schema)
                ),
                (signature,),
            ).fetchone()[0]
        assert persisted == fingerprint
        with pytest.raises(RegistryQualificationError):
            PostgreSQLEntitlementProvisioningAdminProvider(
                _role_dsn(setup.admin_role),
                schema=setup.schema,
                environment="PRODUCTION_LOCAL",
                trust_domain=setup.trust_domain,
            )


def test_coordinated_append_bind_exploit_and_manifest_tamper_is_rejected() -> None:
    signature = "append_bind(text,text,text,bigint,json)"
    with _isolated_registry() as setup:
        runtime = PostgreSQLEntitlementRegistryProvider(
            _role_dsn(setup.runtime_role),
            schema=setup.schema,
            environment="PRODUCTION_LOCAL",
            trust_domain=setup.trust_domain,
        )
        admin = PostgreSQLEntitlementProvisioningAdminProvider(
            _role_dsn(setup.admin_role),
            schema=setup.schema,
            environment="PRODUCTION_LOCAL",
            trust_domain=setup.trust_domain,
        )
        subject = _subject("coordinated-function-tamper")
        initial = _provision(admin, subject, raw="018f3e70-7b5a-7c21-8b9a-000000000731")
        first = replace(
            initial,
            binding=_binding(731),
            authoritative_state_revision=2,
            predecessor_revision=1,
        )
        assert (
            runtime.compare_and_swap_bind(
                BindRequest(subject, predecessor_for(initial), first.binding)
            ).outcome
            is BindOutcome.NEW_BIND_COMMITTED
        )
        replacement_body = sql.SQL(
            "CREATE OR REPLACE FUNCTION {}.append_bind(p_handle text,p_environment text,p_trust text,p_expected bigint,p_state json) "
            "RETURNS void LANGUAGE plpgsql SECURITY DEFINER SET search_path=pg_catalog "
            "AS 'BEGIN UPDATE {}.history SET state=$5::jsonb,"
            "state_integrity=md5(($5::jsonb)::text) WHERE lookup_handle=$1 "
            "AND environment=$2 AND trust_domain=$3 AND revision=2; END'"
        ).format(sql.Identifier(setup.schema), sql.Identifier(setup.schema))
        with psycopg.connect(_role_dsn(setup.schema_owner_role).dsn, autocommit=True) as conn:
            conn.execute(replacement_body)
        fingerprint = _persist_actual_function_fingerprint(setup, signature)
        assert fingerprint != postgres_registry._reviewed_function_manifest(setup)[signature]
        with psycopg.connect(BASE_DSN) as conn:
            persisted = conn.execute(
                sql.SQL("SELECT function_definition_sha256->>%s FROM {}.metadata").format(
                    sql.Identifier(setup.schema)
                ),
                (signature,),
            ).fetchone()[0]
        assert persisted == fingerprint
        second = replace(first, binding=_binding(732))
        with psycopg.connect(_role_dsn(setup.runtime_role).dsn, autocommit=True) as conn:
            conn.execute(
                sql.SQL("SELECT {}.append_bind(%s,%s,%s,%s,%s)").format(
                    sql.Identifier(setup.schema)
                ),
                (
                    *_subject_params_for_test(subject),
                    2,
                    psycopg.types.json.Json(postgres_registry._state_json(second)),
                ),
            )
        assert runtime.retained_history(subject).states[-1] == second
        for provider_class, role in (
            (PostgreSQLEntitlementRegistryProvider, setup.runtime_role),
            (PostgreSQLEntitlementProvisioningAdminProvider, setup.admin_role),
        ):
            with pytest.raises(RegistryQualificationError):
                provider_class(
                    _role_dsn(role),
                    schema=setup.schema,
                    environment="PRODUCTION_LOCAL",
                    trust_domain=setup.trust_domain,
                )


@pytest.mark.parametrize(
    "mutation",
    [
        "wrong_function_owner",
        "security_invoker",
        "unsafe_search_path",
        "public_execute",
        "runtime_forbidden_dml",
        "runtime_forbidden_create",
        "runtime_forbidden_admin",
        "runtime_table_grant_option",
        "admin_forbidden_dml",
        "admin_forbidden_bind",
    ],
)
def test_function_and_privilege_shape_mutations_fail_qualification(mutation: str) -> None:
    with _isolated_registry() as setup:
        with psycopg.connect(BASE_DSN, autocommit=True) as conn:
            append_signature = sql.SQL("{}.append_bind(text,text,text,bigint,json)").format(
                sql.Identifier(setup.schema)
            )
            if mutation == "wrong_function_owner":
                conn.execute(
                    sql.SQL("ALTER FUNCTION {} OWNER TO {}").format(
                        append_signature, sql.Identifier(setup.admin_role)
                    )
                )
            elif mutation == "security_invoker":
                conn.execute(sql.SQL("ALTER FUNCTION {} SECURITY INVOKER").format(append_signature))
            elif mutation == "unsafe_search_path":
                conn.execute(
                    sql.SQL("ALTER FUNCTION {} SET search_path = public").format(append_signature)
                )
            elif mutation == "public_execute":
                conn.execute(
                    sql.SQL("GRANT EXECUTE ON FUNCTION {} TO PUBLIC").format(append_signature)
                )
            elif mutation == "runtime_forbidden_dml":
                conn.execute(
                    sql.SQL("GRANT UPDATE ON {}.history TO {}").format(
                        sql.Identifier(setup.schema), sql.Identifier(setup.runtime_role)
                    )
                )
            elif mutation == "runtime_forbidden_create":
                conn.execute(
                    sql.SQL("GRANT CREATE ON SCHEMA {} TO {}").format(
                        sql.Identifier(setup.schema), sql.Identifier(setup.runtime_role)
                    )
                )
            elif mutation == "runtime_forbidden_admin":
                conn.execute(
                    sql.SQL(
                        "GRANT EXECUTE ON FUNCTION {}.provision(text,text,text,text,text,json) TO {}"
                    ).format(sql.Identifier(setup.schema), sql.Identifier(setup.runtime_role))
                )
            elif mutation == "admin_forbidden_dml":
                conn.execute(
                    sql.SQL("GRANT DELETE ON {}.history TO {}").format(
                        sql.Identifier(setup.schema), sql.Identifier(setup.admin_role)
                    )
                )
            elif mutation == "runtime_table_grant_option":
                conn.execute(
                    sql.SQL("GRANT SELECT ON {}.history TO {} WITH GRANT OPTION").format(
                        sql.Identifier(setup.schema), sql.Identifier(setup.runtime_role)
                    )
                )
            else:
                conn.execute(
                    sql.SQL("GRANT EXECUTE ON FUNCTION {} TO {}").format(
                        append_signature, sql.Identifier(setup.admin_role)
                    )
                )
        provider_class = (
            PostgreSQLEntitlementProvisioningAdminProvider
            if mutation in ("admin_forbidden_bind", "admin_forbidden_dml")
            else PostgreSQLEntitlementRegistryProvider
        )
        role = (
            setup.admin_role
            if mutation in ("admin_forbidden_bind", "admin_forbidden_dml")
            else setup.runtime_role
        )
        with pytest.raises(RegistryQualificationError):
            provider_class(
                _role_dsn(role),
                schema=setup.schema,
                environment="PRODUCTION_LOCAL",
                trust_domain=setup.trust_domain,
            )


def test_missing_composite_unique_rejected_before_reverse_cardinality_break() -> None:
    with _isolated_registry() as setup:
        admin = PostgreSQLEntitlementProvisioningAdminProvider(
            _role_dsn(setup.admin_role),
            schema=setup.schema,
            environment="PRODUCTION_LOCAL",
            trust_domain=setup.trust_domain,
        )
        with psycopg.connect(BASE_DSN, autocommit=True) as conn:
            constraint = conn.execute(
                "SELECT conname FROM pg_constraint WHERE conrelid=%s::regclass AND contype='u'",
                (f"{setup.schema}.lineages",),
            ).fetchone()
            assert constraint is not None
            conn.execute(
                sql.SQL("ALTER TABLE {}.lineages DROP CONSTRAINT {}").format(
                    sql.Identifier(setup.schema), sql.Identifier(constraint[0])
                )
            )
        first = _subject("unqualified-reverse-a")
        second = _subject("unqualified-reverse-b")
        assert (
            admin.provision_entitlement(
                ProvisionEntitlementRequest(first, _identity(), _provenance())
            ).outcome
            is AdminOutcome.COMMITTED
        )
        assert (
            admin.provision_entitlement(
                ProvisionEntitlementRequest(second, _identity(), _provenance())
            ).outcome
            is AdminOutcome.COMMITTED
        )
        for provider_class, role in (
            (PostgreSQLEntitlementRegistryProvider, setup.runtime_role),
            (PostgreSQLEntitlementProvisioningAdminProvider, setup.admin_role),
        ):
            with pytest.raises(RegistryQualificationError):
                provider_class(
                    _role_dsn(role),
                    schema=setup.schema,
                    environment="PRODUCTION_LOCAL",
                    trust_domain=setup.trust_domain,
                )


@pytest.mark.parametrize(
    ("relation", "constraint_type"),
    [
        ("lineages", "p"),
        ("lineages", "c"),
        ("history", "p"),
        ("history", "c"),
        ("history", "f"),
    ],
)
def test_missing_reviewed_constraint_is_rejected(relation: str, constraint_type: str) -> None:
    with _isolated_registry() as setup:
        with psycopg.connect(BASE_DSN, autocommit=True) as conn:
            constraint = conn.execute(
                "SELECT conname FROM pg_constraint WHERE conrelid=%s::regclass AND contype=%s",
                (f"{setup.schema}.{relation}", constraint_type),
            ).fetchone()
            assert constraint is not None
            conn.execute(
                sql.SQL("ALTER TABLE {}.{} DROP CONSTRAINT {} CASCADE").format(
                    sql.Identifier(setup.schema),
                    sql.Identifier(relation),
                    sql.Identifier(constraint[0]),
                )
            )
        with pytest.raises(RegistryQualificationError):
            PostgreSQLEntitlementRegistryProvider(
                _role_dsn(setup.runtime_role),
                schema=setup.schema,
                environment="PRODUCTION_LOCAL",
                trust_domain=setup.trust_domain,
            )


def test_identity_uniqueness_and_cross_product_raw_id(registry) -> None:
    _, admin = _providers(registry)
    first = _subject("identity-a")
    second = _subject("identity-b")
    _provision(admin, first, product="ProductA")
    conflict = admin.provision_entitlement(
        ProvisionEntitlementRequest(second, _identity("ProductA"), _provenance())
    )
    assert conflict.outcome is AdminOutcome.CONFLICT
    cross_product = admin.provision_entitlement(
        ProvisionEntitlementRequest(second, _identity("ProductB"), _provenance())
    )
    assert cross_product.outcome is AdminOutcome.COMMITTED


def test_exact_replay_after_revoke_and_restart(registry) -> None:
    runtime, admin = _providers(registry)
    requested = _subject("revoke-replay")
    initial = _provision(admin, requested, raw="018f3e70-7b5a-7c21-8b9a-000000000101")
    request = BindRequest(requested, predecessor_for(initial), _binding(100))
    committed = runtime.compare_and_swap_bind(request)
    assert committed.outcome is BindOutcome.NEW_BIND_COMMITTED
    assert committed.authoritative_state is not None
    revoked = admin.revoke_entitlement(
        RevokeEntitlementRequest(admin_predecessor_for(committed.authoritative_state))
    )
    assert revoked.outcome is AdminOutcome.COMMITTED
    setup = registry["setup"]
    restarted = PostgreSQLEntitlementRegistryProvider(
        _role_dsn(setup.runtime_role),
        schema=setup.schema,
        environment="PRODUCTION_LOCAL",
        trust_domain=setup.trust_domain,
    )
    before = restarted.retained_history(requested)
    replay = restarted.compare_and_swap_bind(request)
    after = restarted.retained_history(requested)
    assert replay.outcome is BindOutcome.EXACT_REPLAY
    assert replay.authoritative_state == committed.authoritative_state
    assert before == after
    assert after.current_authoritative_state_revision == 3


def test_exact_replay_after_supersession_and_global_non_reuse(registry) -> None:
    runtime, admin = _providers(registry)
    requested = _subject("supersede-replay")
    initial = _provision(admin, requested, raw="018f3e70-7b5a-7c21-8b9a-000000000102")
    original = BindRequest(requested, predecessor_for(initial), _binding(101))
    bound = runtime.compare_and_swap_bind(original)
    assert bound.authoritative_state is not None
    superseded = admin.supersede_entitlement(
        SupersedeEntitlementRequest(
            admin_predecessor_for(bound.authoritative_state),
            replace(bound.authoritative_state.identity, entitlement_generation=2),
            _provenance("immutable:provision:2"),
        )
    )
    assert superseded.outcome is AdminOutcome.COMMITTED
    before = runtime.retained_history(requested)
    replay = runtime.compare_and_swap_bind(original)
    assert replay.outcome is BindOutcome.EXACT_REPLAY
    assert replay.authoritative_state == bound.authoritative_state
    assert runtime.retained_history(requested) == before
    assert superseded.state is not None
    second = runtime.compare_and_swap_bind(
        BindRequest(requested, predecessor_for(superseded.state), _binding(102, generation=2))
    )
    assert second.outcome is BindOutcome.CONFLICT_BOUND_TO_DIFFERENT_TUPLE


def test_real_database_permissions_and_security_definer_hardening(registry) -> None:
    setup = registry["setup"]
    runtime_dsn = _role_dsn(setup.runtime_role).dsn
    admin_dsn = _role_dsn(setup.admin_role).dsn
    forbidden = [
        sql.SQL("INSERT INTO {}.history VALUES ('x','x','x',1,NULL,'{{}}')").format(
            sql.Identifier(setup.schema)
        ),
        sql.SQL("UPDATE {}.history SET state='{{}}'").format(sql.Identifier(setup.schema)),
        sql.SQL("DELETE FROM {}.history").format(sql.Identifier(setup.schema)),
        sql.SQL("UPDATE {}.metadata SET schema_version=2").format(sql.Identifier(setup.schema)),
        sql.SQL("ALTER TABLE {}.history ADD COLUMN evil text").format(sql.Identifier(setup.schema)),
        sql.SQL("DROP TABLE {}.history").format(sql.Identifier(setup.schema)),
    ]
    for dsn in (runtime_dsn, admin_dsn):
        for statement in forbidden:
            with psycopg.connect(dsn, autocommit=True) as conn:
                with pytest.raises(InsufficientPrivilege):
                    conn.execute(statement)
    with psycopg.connect(runtime_dsn, autocommit=True) as conn:
        with pytest.raises(InsufficientPrivilege):
            conn.execute(
                sql.SQL("SELECT {}.revoke('x','x','x',1,'{{}}')").format(
                    sql.Identifier(setup.schema)
                )
            )
        conn.execute("SET search_path = pg_catalog")
        assert conn.execute(
            sql.SQL("SELECT count(*) FROM {}.metadata").format(sql.Identifier(setup.schema))
        ).fetchone() == (1,)
    with psycopg.connect(BASE_DSN) as conn:
        public_acl = conn.execute(
            """SELECT count(*) FROM pg_proc p JOIN pg_namespace n ON n.oid=p.pronamespace
               CROSS JOIN LATERAL aclexplode(coalesce(p.proacl, acldefault('f',p.proowner))) a
               WHERE n.nspname=%s AND a.grantee=0 AND a.privilege_type='EXECUTE'""",
            (setup.schema,),
        ).fetchone()
        assert public_acl == (0,)


def test_transaction_abort_leaves_no_partial_or_orphan_rows(registry) -> None:
    setup = registry["setup"]
    admin = registry["admin"]
    requested = _subject("rollback")
    state = _provision(admin, requested, raw="018f3e70-7b5a-7c21-8b9a-000000000103")
    runtime = registry["runtime"]
    before = runtime.retained_history(requested)
    attempted = replace(
        state,
        binding=_binding(103),
        authoritative_state_revision=2,
        predecessor_revision=1,
    )
    from bot_core.postgresql_entitlement_registry import _state_json  # noqa: PLC0415

    with psycopg.connect(_role_dsn(setup.runtime_role).dsn) as conn:
        with pytest.raises(RuntimeError), conn.transaction():
            conn.execute("SET TRANSACTION ISOLATION LEVEL SERIALIZABLE")
            conn.execute(
                sql.SQL("SELECT {}.append_bind(%s,%s,%s,%s,%s)").format(
                    sql.Identifier(setup.schema)
                ),
                (
                    requested.lookup_handle,
                    requested.environment,
                    requested.trust_domain,
                    1,
                    psycopg.types.json.Json(_state_json(attempted)),
                ),
            )
            raise RuntimeError("deterministic pre-commit connection-loss analogue")
    assert runtime.retained_history(requested) == before


@pytest.mark.parametrize(
    "tamper",
    [
        "gap",
        "physical_revision_gap",
        "physical_predecessor",
        "rewind",
        "head_beyond",
        "subject",
        "product",
        "entitlement",
        "generation",
        "provenance",
        "bound_to_unbound",
        "bound_mutation",
        "illegal_lifecycle",
    ],
)
def test_stored_corruption_fails_closed_without_repair(registry, tamper: str) -> None:
    setup = registry["setup"]
    runtime, admin = _providers(registry)
    requested = _subject(f"corrupt-{tamper}")
    raw = f"018f3e70-7b5a-7c21-8b9a-{200 + _counter:012x}"
    initial = _provision(admin, requested, raw=raw)
    bound = runtime.compare_and_swap_bind(
        BindRequest(requested, predecessor_for(initial), _binding(200 + _counter))
    )
    assert bound.outcome is BindOutcome.NEW_BIND_COMMITTED
    table = sql.Identifier(setup.schema, "history")
    lineage = sql.Identifier(setup.schema, "lineages")
    with psycopg.connect(_role_dsn(setup.schema_owner_role).dsn, autocommit=True) as conn:
        if tamper == "gap":
            conn.execute(
                sql.SQL("DELETE FROM {} WHERE lookup_handle=%s AND revision=1").format(table),
                (requested.lookup_handle,),
            )
        elif tamper == "physical_revision_gap":
            conn.execute(
                sql.SQL("UPDATE {} SET revision=3 WHERE lookup_handle=%s AND revision=2").format(
                    table
                ),
                (requested.lookup_handle,),
            )
        elif tamper == "physical_predecessor":
            conn.execute(
                sql.SQL(
                    "UPDATE {} SET predecessor_revision=999 WHERE lookup_handle=%s AND revision=2"
                ).format(table),
                (requested.lookup_handle,),
            )
        elif tamper == "rewind":
            conn.execute(
                sql.SQL("UPDATE {} SET current_revision=1 WHERE lookup_handle=%s").format(lineage),
                (requested.lookup_handle,),
            )
        elif tamper == "head_beyond":
            conn.execute(
                sql.SQL("UPDATE {} SET current_revision=99 WHERE lookup_handle=%s").format(lineage),
                (requested.lookup_handle,),
            )
        elif tamper == "subject":
            conn.execute(
                sql.SQL(
                    "UPDATE {} SET state=jsonb_set(state,'{{subject,lookup_handle}}','\"other\"') WHERE lookup_handle=%s AND revision=2"
                ).format(table),
                (requested.lookup_handle,),
            )
        elif tamper == "product":
            conn.execute(
                sql.SQL(
                    "UPDATE {} SET state=jsonb_set(state,'{{identity,product_scope}}','\"tampered\"') WHERE lookup_handle=%s AND revision=2"
                ).format(table),
                (requested.lookup_handle,),
            )
        elif tamper == "entitlement":
            conn.execute(
                sql.SQL(
                    "UPDATE {} SET state=jsonb_set(state,'{{identity,bootstrap_entitlement_id}}','\"ent_018f3e70-7b5a-7c21-8b9a-999999999999\"') WHERE lookup_handle=%s AND revision=2"
                ).format(table),
                (requested.lookup_handle,),
            )
        elif tamper == "generation":
            conn.execute(
                sql.SQL(
                    "UPDATE {} SET state=jsonb_set(state,'{{identity,entitlement_generation}}','2') WHERE lookup_handle=%s AND revision=2"
                ).format(table),
                (requested.lookup_handle,),
            )
        elif tamper == "provenance":
            conn.execute(
                sql.SQL(
                    "UPDATE {} SET state=jsonb_set(state,'{{provenance,authenticated_creation_reference}}','\"tampered\"') WHERE lookup_handle=%s AND revision=2"
                ).format(table),
                (requested.lookup_handle,),
            )
        elif tamper == "bound_to_unbound":
            conn.execute(
                sql.SQL(
                    "UPDATE {} SET state=jsonb_set(state,'{{binding}}','{{\"kind\":\"UNBOUND\"}}') WHERE lookup_handle=%s AND revision=2"
                ).format(table),
                (requested.lookup_handle,),
            )
        elif tamper == "bound_mutation":
            conn.execute(
                sql.SQL(
                    "UPDATE {} SET state=jsonb_set(state,'{{binding,account_id}}','\"acct_018f3e70-7b5a-7c21-8b9a-999999999999\"') WHERE lookup_handle=%s AND revision=2"
                ).format(table),
                (requested.lookup_handle,),
            )
        else:
            conn.execute(
                sql.SQL(
                    "UPDATE {} SET state=jsonb_set(state,'{{lifecycle}}','\"REVOKED\"') WHERE lookup_handle=%s AND revision=1"
                ).format(table),
                (requested.lookup_handle,),
            )
    first = runtime.retained_history(requested)
    second = runtime.retained_history(requested)
    assert first.outcome is RegistryReadOutcome.CORRUPT
    assert second == first


def test_provision_and_supersession_rollback_are_atomic(registry) -> None:
    setup = registry["setup"]
    runtime, admin = _providers(registry)
    fresh = _subject("rollback-provision")
    request = ProvisionEntitlementRequest(
        fresh,
        _identity(raw="018f3e70-7b5a-7c21-8b9a-000000000250"),
        _provenance(),
    )
    genesis = initial_state_for(request)
    from bot_core.postgresql_entitlement_registry import _state_json  # noqa: PLC0415

    with psycopg.connect(_role_dsn(setup.admin_role).dsn) as conn:
        with pytest.raises(RuntimeError), conn.transaction():
            conn.execute("SET TRANSACTION ISOLATION LEVEL SERIALIZABLE")
            conn.execute(
                sql.SQL("SELECT {}.provision(%s,%s,%s,%s,%s,%s)").format(
                    sql.Identifier(setup.schema)
                ),
                (
                    fresh.lookup_handle,
                    fresh.environment,
                    fresh.trust_domain,
                    request.identity.product_scope,
                    request.identity.bootstrap_entitlement_id,
                    psycopg.types.json.Json(_state_json(genesis)),
                ),
            )
            raise RuntimeError("abort before commit")
    assert runtime.authoritative_state(fresh).outcome is RegistryReadOutcome.NOT_FOUND

    existing = _subject("rollback-supersession")
    current = _provision(admin, existing, raw="018f3e70-7b5a-7c21-8b9a-000000000251")
    supersede = SupersedeEntitlementRequest(
        admin_predecessor_for(current),
        replace(current.identity, entitlement_generation=2),
        _provenance("immutable:rollback-successor"),
    )
    retired, successor = supersession_states_for(current, supersede)
    with psycopg.connect(_role_dsn(setup.admin_role).dsn) as conn:
        with pytest.raises(RuntimeError), conn.transaction():
            conn.execute("SET TRANSACTION ISOLATION LEVEL SERIALIZABLE")
            conn.execute(
                sql.SQL("SELECT {}.supersede(%s,%s,%s,%s,%s,%s)").format(
                    sql.Identifier(setup.schema)
                ),
                (
                    existing.lookup_handle,
                    existing.environment,
                    existing.trust_domain,
                    1,
                    psycopg.types.json.Json(_state_json(retired)),
                    psycopg.types.json.Json(_state_json(successor)),
                ),
            )
            raise RuntimeError("abort both supersession records")
    history = runtime.retained_history(existing)
    assert len(history.states) == 1
    assert history.states[0] == current


@pytest.mark.parametrize("transition", ["revoked_to_active", "superseded_to_active"])
def test_terminal_lifecycle_corruption_fails_closed(registry, transition: str) -> None:
    setup = registry["setup"]
    runtime, admin = _providers(registry)
    requested = _subject(transition)
    raw = f"018f3e70-7b5a-7c21-8b9a-{300 + _counter:012x}"
    initial = _provision(admin, requested, raw=raw)
    bound = runtime.compare_and_swap_bind(
        BindRequest(requested, predecessor_for(initial), _binding(300 + _counter))
    )
    assert bound.authoritative_state is not None
    if transition == "revoked_to_active":
        result = admin.revoke_entitlement(
            RevokeEntitlementRequest(admin_predecessor_for(bound.authoritative_state))
        )
    else:
        result = admin.supersede_entitlement(
            SupersedeEntitlementRequest(
                admin_predecessor_for(bound.authoritative_state),
                replace(bound.authoritative_state.identity, entitlement_generation=2),
                _provenance("immutable:terminal-tamper"),
            )
        )
    assert result.outcome is AdminOutcome.COMMITTED
    with psycopg.connect(_role_dsn(setup.schema_owner_role).dsn, autocommit=True) as conn:
        conn.execute(
            sql.SQL(
                "UPDATE {}.history SET state=jsonb_set(state,'{{lifecycle}}','\"ACTIVE\"') "
                "WHERE lookup_handle=%s AND revision=3"
            ).format(sql.Identifier(setup.schema)),
            (requested.lookup_handle,),
        )
    assert runtime.retained_history(requested).outcome is RegistryReadOutcome.CORRUPT


def _run_pair(first, second):
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = (pool.submit(first), pool.submit(second))
        return tuple(future.result() for future in futures)


def test_concurrent_provision_and_bind_races_use_real_connections(registry) -> None:
    runtime, admin = _providers(registry)
    raw = "018f3e70-7b5a-7c21-8b9a-000000000301"
    a, b = _subject("race-provision-a"), _subject("race-provision-b")
    provision_results = _run_pair(
        lambda: admin.provision_entitlement(
            ProvisionEntitlementRequest(a, _identity(raw=raw), _provenance())
        ),
        lambda: admin.provision_entitlement(
            ProvisionEntitlementRequest(b, _identity(raw=raw), _provenance())
        ),
    )
    assert sum(item.outcome is AdminOutcome.COMMITTED for item in provision_results) == 1
    winner = a if runtime.authoritative_state(a).outcome is RegistryReadOutcome.FOUND else b
    current = runtime.authoritative_state(winner).state
    assert current is not None
    requests = (
        BindRequest(winner, predecessor_for(current), _binding(301)),
        BindRequest(winner, predecessor_for(current), _binding(302)),
    )
    bind_results = _run_pair(
        lambda: runtime.compare_and_swap_bind(requests[0]),
        lambda: runtime.compare_and_swap_bind(requests[1]),
    )
    assert sum(item.outcome is BindOutcome.NEW_BIND_COMMITTED for item in bind_results) == 1
    history = runtime.retained_history(winner)
    assert len(history.states) == 2


@pytest.mark.parametrize(
    ("left", "right"),
    [
        ("revoke", "revoke"),
        ("revoke", "supersede"),
        ("supersede", "supersede"),
        ("bind", "revoke"),
        ("bind", "supersede"),
    ],
)
def test_concurrent_admin_and_runtime_races_have_one_successor(
    registry, left: str, right: str
) -> None:
    runtime, admin = _providers(registry)
    requested = _subject(f"race-{left}-{right}")
    raw = f"018f3e70-7b5a-7c21-8b9a-{400 + _counter:012x}"
    current = _provision(admin, requested, raw=raw)

    def action(kind: str, tag: int):
        if kind == "bind":
            return runtime.compare_and_swap_bind(
                BindRequest(requested, predecessor_for(current), _binding(tag))
            ).outcome
        expected = admin_predecessor_for(current)
        if kind == "revoke":
            return admin.revoke_entitlement(RevokeEntitlementRequest(expected)).outcome
        return admin.supersede_entitlement(
            SupersedeEntitlementRequest(
                expected,
                replace(current.identity, entitlement_generation=2),
                _provenance(f"immutable:race:{tag}"),
            )
        ).outcome

    results = _run_pair(lambda: action(left, 500), lambda: action(right, 501))
    successes = sum(
        result in (AdminOutcome.COMMITTED, BindOutcome.NEW_BIND_COMMITTED) for result in results
    )
    assert successes == 1
    history = runtime.retained_history(requested)
    supersede_committed = (left == "supersede" and results[0] is AdminOutcome.COMMITTED) or (
        right == "supersede" and results[1] is AdminOutcome.COMMITTED
    )
    expected_length = 3 if supersede_committed else 2
    assert len(history.states) == expected_length

"""Fast negative boundary tests; physical topology is tested on PostgreSQL 16."""

from dataclasses import FrozenInstanceError, replace
from types import SimpleNamespace

import pytest

from bot_core import root_proof_atomic_composition as implementation
from bot_core.postgresql_entitlement_registry import PostgreSQLConnectionConfig
from bot_core.postgresql_root_proof_retention import RootProofRetentionScope
from bot_core.root_proof_issuer_substrate import SecurityProfile


def _deployment():
    connection = PostgreSQLConnectionConfig("dbname=authority user=runtime connect_timeout=1")
    scope = RootProofRetentionScope(
        SecurityProfile.PRODUCTION_LOCAL, "PRODUCTION", "trust", "product", "issuer"
    )
    pins = {
        name: implementation.AtomicRootProofAuthorityPin(
            connection, name + "_schema", name + "_runtime"
        )
        for name in implementation._AUTHORITY_NAMES
    }
    return implementation.AtomicRootProofDeployment(
        scope, **pins, transaction_owner=connection, transaction_owner_role="atomic_owner"
    )


def test_configuration_is_frozen_and_redacts_transports():
    deployment = _deployment()
    assert "dbname=authority" not in repr(deployment)
    with pytest.raises(FrozenInstanceError):
        deployment.transaction_owner_role = "another"


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema", "public;DROP SCHEMA public"),
        ("login_role", "owner;SET ROLE postgres"),
        ("connection", lambda: PostgreSQLConnectionConfig("dbname=caller")),
    ],
)
def test_pins_reject_arbitrary_callbacks_and_malformed_identifiers(field, value):
    with pytest.raises(TypeError, match="EXACT_DEPLOYMENT_AUTHORITY_PIN_REQUIRED"):
        replace(_deployment().entitlement, **{field: value})


@pytest.mark.parametrize(
    "field,value",
    [
        ("scope", {"security_profile": "PRODUCTION_LOCAL"}),
        ("entitlement", object()),
        ("transaction_owner", "dbname=caller"),
        ("transaction_owner_role", "postgres;SELECT true"),
    ],
)
def test_deployment_rejects_untrusted_dtos(field, value):
    with pytest.raises(TypeError, match="EXACT_INTERNAL_ATOMIC_DEPLOYMENT_REQUIRED"):
        replace(_deployment(), **{field: value})


def test_arbitrary_ports_fail_closed_before_any_probe(monkeypatch):
    def reject_connect(*args, **kwargs):
        pytest.fail("caller-supplied ports must never trigger an authority probe")

    monkeypatch.setattr(implementation.psycopg, "connect", reject_connect)
    result = implementation.qualify_atomic_root_proof_composition(
        _deployment(),
        issuance_authority=object(),
        signing_intent=object(),
        retention=object(),
        history=object(),
        checkpoint=object(),
    )
    assert result.state == "NOT_READY"
    assert result.authorizes_operations is False
    assert "EXACT_INTERNAL_COMPOSITION_REQUIRED" in result.failures
    assert "ATOMIC_COORDINATOR_NOT_IMPLEMENTED" in result.failures
    assert "SAME_CONNECTION_PARTICIPATION_NOT_IMPLEMENTED" in result.failures
    assert "INDEPENDENT_CONNECTION_SAME_PHYSICAL_TRANSACTION_UNPROVEN" in result.failures
    assert result.authorities == ()


@pytest.mark.parametrize(
    "changes",
    [
        {"state": "READY"},
        {"authorizes_operations": True},
        {"authorizes_operations": 0},
        {"failures": ()},
    ],
)
def test_detached_report_cannot_express_ready_or_authorization(changes):
    with pytest.raises(TypeError):
        implementation.AtomicRootProofCompositionQualification(
            **{"failures": ("ATOMIC_COORDINATOR_NOT_IMPLEMENTED",), **changes}
        )


class _ControlCatalogRead:
    def __init__(self, rows, version=160006):
        self.rows = rows
        self.info = SimpleNamespace(server_version=version)
        self.queries = []

    def execute(self, query):
        self.queries.append(query)
        return SimpleNamespace(fetchall=lambda: self.rows)


def test_database_lineage_comes_from_control_and_catalog_queries():
    conn = _ControlCatalogRead([("7412345678901234567", 16001, "authority", 10)])
    observed = implementation._physical_identity(conn)
    assert observed.system_identifier == "7412345678901234567"
    assert observed.database_oid == 16001
    assert observed.database_name == "authority"
    assert observed.database_owner_oid == 10
    assert len(conn.queries) == 1
    assert "pg_catalog.pg_control_system()" in conn.queries[0]
    assert "pg_catalog.pg_database" in conn.queries[0]
    assert "current_database()" in conn.queries[0]
    assert "host" not in conn.queries[0] and "dsn" not in conn.queries[0]


@pytest.mark.parametrize(
    "rows",
    [
        [],
        [("741", 10, "authority", 10), ("741", 11, "authority", 10)],
        [("caller-cluster", 10, "authority", 10)],
        [("0", 10, "authority", 10)],
        [("٠١", 10, "authority", 10)],
        [("18446744073709551616", 10, "authority", 10)],
        [(741, 10, "authority", 10)],
        [("741", True, "authority", 10)],
        [("741", -1, "authority", 10)],
        [("741", 10, "", 10)],
        [("741", 10, "authority", False)],
        [("741", 10, "authority")],
    ],
)
def test_unknown_or_malformed_physical_identity_is_not_attested(rows):
    with pytest.raises(ValueError, match="PHYSICAL_DATABASE_CATALOG_EVIDENCE_REQUIRED"):
        implementation._physical_identity(_ControlCatalogRead(rows))


@pytest.mark.parametrize("version", [150009, 170001])
def test_database_version_must_be_postgresql_16(version):
    conn = _ControlCatalogRead([("741", 10, "authority", 10)], version)
    with pytest.raises(ValueError, match="POSTGRESQL_16_REQUIRED"):
        implementation._physical_identity(conn)
    assert conn.queries == []


@pytest.mark.parametrize(
    "error,category",
    [
        (RuntimeError("authority role has a membership grant"), "ROLE_MEMBERSHIP_FORBIDDEN"),
        (RuntimeError("requires exact direct-login writer"), "ROLE_OR_PRINCIPAL_REJECTED"),
        (RuntimeError("writer has an extra object grant"), "ACL_OR_OWNERSHIP_REJECTED"),
        (RuntimeError("schema owner mismatch"), "ACL_OR_OWNERSHIP_REJECTED"),
        (RuntimeError("physical definition mismatch"), "SCHEMA_QUALIFICATION_FAILED"),
    ],
)
def test_diagnostics_distinguish_source_qualification_failures(error, category):
    assert implementation._qualification_failure_detail(error) == category

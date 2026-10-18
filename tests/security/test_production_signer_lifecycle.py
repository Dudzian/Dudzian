"""Real PostgreSQL public signer observations, without physical custody claims.

The reused fixture installs closed public rows and injects ACTIVE only as an
isolated test superuser. Observation never activates a credential, signs a
request, qualifies an OS service or authorizes a production BIND.
"""

from __future__ import annotations

import hashlib
import time
from concurrent.futures import ThreadPoolExecutor
from queue import Queue

import psycopg
import pytest
from psycopg import sql
from psycopg.types.json import Jsonb

from bot_core import postgresql_production_root_lifecycle_schema as lifecycle
from bot_core.postgresql_atomic_root_proof_qualification import (
    read_atomic_root_proof_physical_identity,
)
from bot_core.postgresql_production_root_lifecycle_qualification import (
    pin_production_root_lifecycle_catalog,
)
from tests.security import test_production_root_lifecycle_races as races

pytestmark = pytest.mark.external_postgresql
isolated_database = races.isolated_database
authority = races.authority
active_authority = races.active_authority
ROOT = races.ROOT
HISTORY = "HISTORY_ATTESTATION_SIGNING"


def _observe(value, role=ROOT, identity=None):
    with races._transaction(value) as issuer:
        return races._call(
            issuer,
            value,
            "observe_signer_custody",
            (role, identity if identity is not None else value.binding(role)["identity"]),
        )


def _expected(binding, state, *, pending=None, current=True, generation=None):
    return dict(
        binding,
        lifecycle_generation=(
            binding["lifecycle_generation"] if generation is None else generation
        ),
        state=state,
        pending_operation_id=pending,
        is_current=current,
        new_operations_authorized=False,
    )


@pytest.mark.parametrize("role", [ROOT, HISTORY])
def test_closed_public_binding_is_observable_without_new_operation_authority(authority, role):
    binding = authority.binding(role)
    state, generation, pending = races._fence(authority, role)
    assert (state, generation) == ("PENDING_CLOSED", 1)
    assert _observe(authority, role) == _expected(binding, state, pending=pending)
    assert races._fence(authority, role) == (state, generation, pending)
    with pytest.raises(psycopg.errors.CheckViolation, match="closed or mismatched"):
        races._assert_gate(authority)


@pytest.mark.parametrize("role", [ROOT, HISTORY])
def test_active_observation_is_exact_and_remains_diagnostic(active_authority, role):
    value = active_authority
    binding = value.binding(role)
    assert _observe(value, role) == _expected(binding, "ACTIVE")
    assert _observe(value, role, "missing-public-signer") is None
    other = HISTORY if role == ROOT else ROOT
    assert _observe(value, role, value.binding(other)["identity"]) is None
    assert races._fence(value, role) == ("ACTIVE", 1, None)


@pytest.mark.parametrize(
    "role,identity",
    [
        ("TRUST_ACTIVATION", "identity"),
        ("UNKNOWN", "identity"),
        (None, "identity"),
        (ROOT, None),
        (ROOT, ""),
    ],
)
def test_observation_rejects_non_signing_roles_or_missing_identity(authority, role, identity):
    with races._transaction(authority) as issuer:
        with pytest.raises(psycopg.errors.InvalidParameterValue, match="exact signing role"):
            races._call(issuer, authority, "observe_signer_custody", (role, identity))


@pytest.mark.parametrize("mode", ["READ COMMITTED", "SERIALIZABLE READ ONLY"])
def test_observation_requires_locking_serializable_transaction(authority, mode):
    with psycopg.connect(authority.dsn()) as issuer:
        issuer.execute(sql.SQL("SET TRANSACTION ISOLATION LEVEL {} ").format(sql.SQL(mode)))
        with pytest.raises(psycopg.errors.ActiveSqlTransaction, match="writable SERIALIZABLE"):
            races._call(
                issuer, authority, "observe_signer_custody", (ROOT, authority.binding()["identity"])
            )


@pytest.mark.parametrize("login", ["administrator", "bootstrap", "set_role", "set_session"])
def test_observation_requires_exact_direct_installed_issuer_login(authority, login):
    value = authority
    dsn = value.dsn(True) if login == "administrator" else value.base_dsn
    with psycopg.connect(dsn) as connection:
        connection.execute("SET TRANSACTION ISOLATION LEVEL SERIALIZABLE")
        if login == "set_role":
            connection.execute(
                sql.SQL("SET ROLE {}").format(sql.Identifier(value.config.issuer_role))
            )
        elif login == "set_session":
            connection.execute(
                sql.SQL("SET SESSION AUTHORIZATION {}").format(
                    sql.Identifier(value.config.issuer_role)
                )
            )
        with pytest.raises(psycopg.errors.InsufficientPrivilege):
            races._call(
                connection, value, "observe_signer_custody", (ROOT, value.binding()["identity"])
            )


def test_new_observation_port_is_in_reviewed_catalog_and_issuer_acl(authority):
    value = authority
    signature = "observe_signer_custody(text,text)"
    assert signature in lifecycle.PRODUCTION_LIFECYCLE_FUNCTION_DECLARATIONS
    assert signature in lifecycle.ISSUER_FUNCTIONS
    assert signature not in lifecycle.ADMINISTRATOR_FUNCTIONS
    with psycopg.connect(value.base_dsn) as bootstrap:
        bootstrap.execute("SET TRANSACTION ISOLATION LEVEL SERIALIZABLE")
        physical = read_atomic_root_proof_physical_identity(bootstrap, value.config.installation_id)
        pin_production_root_lifecycle_catalog(bootstrap, value.config, expected_identity=physical)
    assert _observe(value)["new_operations_authorized"] is False


def test_observation_rejects_catalog_drift(authority):
    value = authority
    with psycopg.connect(value.base_dsn) as bootstrap:
        bootstrap.execute(
            sql.SQL("GRANT SELECT ON {}.fences TO {}").format(
                sql.Identifier(value.config.schema), sql.Identifier(value.config.issuer_role)
            )
        )
    with pytest.raises(psycopg.errors.InsufficientPrivilege, match="reviewed catalog drift"):
        _observe(value)


@pytest.mark.parametrize("role", [ROOT, HISTORY])
def test_pending_verify_only_and_revoked_observation_tracks_exact_current_generation(
    active_authority, role
):
    value = active_authority
    binding = value.binding(role)
    close = races._request(value, role=role, target="VERIFY_ONLY")
    pending = races._begin(value, close)
    assert _observe(value, role) == _expected(
        binding, "PENDING_CLOSED", pending=close["operation_id"]
    )
    races._complete(value, close, pending)
    assert _observe(value, role) == _expected(binding, "VERIFY_ONLY", generation=2)
    revoke = races._request(value, role=role, generation=2, predecessor="VERIFY_ONLY")
    pending = races._begin(value, revoke)
    assert _observe(value, role) == _expected(
        binding, "PENDING_CLOSED", pending=revoke["operation_id"], generation=2
    )
    races._complete(value, revoke, pending)
    assert _observe(value, role) == _expected(binding, "REVOKED", generation=3)
    with races._transaction(value) as issuer:
        assert (
            races._call(issuer, value, "read_historical_fence", (role, binding["identity"], 1))
            is None
        )
    with pytest.raises(psycopg.errors.CheckViolation):
        races._assert_gate(value)


def _successor_enrollment(value):
    successor = dict(
        value.binding(),
        identity=value.binding()["identity"] + "-successor",
        public_material_identity="sha256:" + hashlib.sha256(b"isolated-successor").hexdigest(),
        key_version=2,
        lifecycle_generation=3,
    )
    return {
        "operation_id": "isolated-test-observation-rotation",
        "predecessor_identity": value.binding()["identity"],
        "expected_generation": 2,
        "successor": successor,
    }


def test_rotation_observes_retired_identity_separately_from_current_closed_successor(
    active_authority,
):
    value = active_authority
    close = races._request(value, target="VERIFY_ONLY")
    races._complete(value, close, races._begin(value, close))
    enrollment = _successor_enrollment(value)
    with races._transaction(value, administrator=True) as administrator:
        races._call(administrator, value, "enroll_closed_successor", (Jsonb(enrollment),))
    assert _observe(value) == _expected(value.binding(), "VERIFY_ONLY", generation=2, current=False)
    successor = enrollment["successor"]
    assert _observe(value, ROOT, successor["identity"]) == _expected(
        successor, "PENDING_CLOSED", pending=enrollment["operation_id"]
    )
    with pytest.raises(psycopg.errors.CheckViolation):
        races._assert_gate(value)


def _mutate(value, port, arguments, backends):
    with races._transaction(value, administrator=True) as administrator:
        administrator.execute("SET LOCAL statement_timeout='10s'")
        backends.put(administrator.info.backend_pid)
        return races._call(administrator, value, port, arguments)


@pytest.mark.parametrize("mutation", ["begin", "complete", "rotation"])
def test_observation_holds_slot_then_fence_until_close_or_rotation_can_commit(
    active_authority, mutation
):
    value = active_authority
    close = races._request(value, target="VERIFY_ONLY")
    port, arguments = "begin_close", (Jsonb(close),)
    before = _expected(value.binding(), "ACTIVE")
    if mutation in {"complete", "rotation"}:
        pending = races._begin(value, close)
        before = _expected(value.binding(), "PENDING_CLOSED", pending=close["operation_id"])
        port, arguments = (
            "complete_close",
            (
                close["operation_id"],
                pending["request_sha256"],
                races._receipt(value, close, pending),
            ),
        )
        if mutation == "rotation":
            races._complete(value, close, pending)
            before = _expected(value.binding(), "VERIFY_ONLY", generation=2)
            port, arguments = "enroll_closed_successor", (Jsonb(_successor_enrollment(value)),)
    backends = Queue()
    with ThreadPoolExecutor(max_workers=1) as executor:
        with races._transaction(value) as issuer:
            observed = races._call(
                issuer, value, "observe_signer_custody", (ROOT, value.binding()["identity"])
            )
            assert observed == before
            future = executor.submit(_mutate, value, port, arguments, backends)
            pid = backends.get(timeout=5)
            deadline = time.monotonic() + 5
            while True:
                with psycopg.connect(value.base_dsn) as observer:
                    blocked = observer.execute(
                        "SELECT cardinality(pg_catalog.pg_blocking_pids(%s))>0", (pid,)
                    ).fetchone()[0]
                if blocked:
                    break
                assert time.monotonic() < deadline, (
                    "administrator did not wait for observation locks"
                )
                time.sleep(0.01)
            assert not future.done()
            assert (
                races._call(
                    issuer, value, "observe_signer_custody", (ROOT, value.binding()["identity"])
                )
                == before
            )
        assert future.result(timeout=10)["phase"] in {"PENDING_CLOSED", "COMPLETED"}
    observed = _observe(value)
    assert observed["new_operations_authorized"] is False
    if mutation == "begin":
        assert observed == _expected(
            value.binding(), "PENDING_CLOSED", pending=close["operation_id"]
        )
    elif mutation == "complete":
        assert observed == _expected(value.binding(), "VERIFY_ONLY", generation=2)
    else:
        assert observed == _expected(value.binding(), "VERIFY_ONLY", generation=2, current=False)

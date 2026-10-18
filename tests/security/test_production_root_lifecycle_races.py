"""Real PostgreSQL lifecycle gate proofs without enrollment or Root Proof writes.

The fixtures inject ACTIVE and historical closed public rows as an isolated test
superuser because production exposes neither activation nor qualified completion.
These tests prove lifecycle slots, fences, denied receipt publication and public
historical reads. Fixture injection supplies no custody completion evidence and
does not authorize a production BIND or qualify an immutable attempt.
"""

from __future__ import annotations

import hashlib
import json
import multiprocessing
import os
import socket
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from types import SimpleNamespace

import psycopg
import pytest
from psycopg import sql
from psycopg.conninfo import make_conninfo
from psycopg.types.json import Jsonb

from bot_core import (
    postgresql_production_root_lifecycle_schema as lifecycle,
    production_root_lifecycle as coordinator,
)
from bot_core.postgresql_atomic_root_proof_qualification import (
    read_atomic_root_proof_physical_identity,
)
from bot_core.postgresql_production_root_lifecycle_qualification import (
    pin_production_root_lifecycle_catalog,
)
from tests.security.test_postgresql_checkpoint_authority import BASE_DSN
from tests.security.test_postgresql_freshness_authority_core import _restart_controlled_postgresql

pytestmark = pytest.mark.external_postgresql
requires_fork = pytest.mark.skipif(os.name != "posix", reason="isolated real process faults")
ROOT = "ROOT_PROOF_SIGNING"


@dataclass(frozen=True)
class _Authority:
    config: lifecycle.PostgreSQLProductionRootLifecycleProvisioning
    bindings: tuple[dict, ...]
    base_dsn: str

    def dsn(self, administrator=False):
        role = self.config.administrator_role if administrator else self.config.issuer_role
        return make_conninfo(self.base_dsn, user=role)

    def binding(self, role=ROOT):
        return next(dict(binding) for binding in self.bindings if binding["semantic_role"] == role)


@contextmanager
def _transaction(value, *, administrator=False):
    with psycopg.connect(value.dsn(administrator)) as connection:
        connection.execute("SET TRANSACTION ISOLATION LEVEL SERIALIZABLE")
        yield connection


def _call(connection, value, name, arguments):
    return connection.execute(
        sql.SQL("SELECT {}.{}({})").format(
            sql.Identifier(value.config.schema),
            sql.Identifier(name),
            sql.SQL(",").join(sql.Placeholder() for _ in arguments),
        ),
        arguments,
    ).fetchone()[0]


@pytest.fixture(scope="module")
def isolated_database():
    name = "isolated_lifecycle_" + uuid.uuid4().hex[:12]
    with psycopg.connect(BASE_DSN, autocommit=True) as bootstrap:
        bootstrap.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(name)))
    dsn = make_conninfo(BASE_DSN, dbname=name)
    try:
        with psycopg.connect(dsn, autocommit=True) as bootstrap:
            bootstrap.execute(
                sql.SQL("REVOKE CREATE,TEMPORARY ON DATABASE {} FROM PUBLIC").format(
                    sql.Identifier(name)
                )
            )
        yield dsn
    finally:
        with psycopg.connect(BASE_DSN, autocommit=True) as bootstrap:
            bootstrap.execute(sql.SQL("DROP DATABASE {} WITH (FORCE)").format(sql.Identifier(name)))


@pytest.fixture
def authority(isolated_database):
    token = uuid.uuid4().hex[:12]
    config = lifecycle.PostgreSQLProductionRootLifecycleProvisioning(
        f"prl_{token}",
        f"prlo_{token}",
        f"prli_{token}",
        f"prla_{token}",
        f"isolated-test-installation-{token}",
        f"isolated-test-trust-{token}",
    )
    bindings = tuple(
        {
            "semantic_role": role,
            "authority_identity": f"isolated-test-authority-{index}-{token}",
            "identity": f"isolated-test-credential-{index}-{token}",
            "public_material_identity": "sha256:"
            + hashlib.sha256(f"{index}:{token}".encode()).hexdigest(),
            "key_version": 1,
            "lifecycle_generation": 1,
        }
        for index, role in enumerate(sorted(lifecycle.SEMANTIC_ROLES))
    )
    value = _Authority(config, bindings, isolated_database)
    installed = False
    try:
        with psycopg.connect(value.base_dsn) as bootstrap:
            bootstrap.execute("SET TRANSACTION ISOLATION LEVEL SERIALIZABLE")
            assert bootstrap.info.server_version // 10000 == 16
            lifecycle.install_production_root_lifecycle_authority(
                bootstrap,
                config,
                bindings,
                bootstrap_operation_id="isolated-test-bootstrap",
            )
        installed = True
        yield value
    finally:
        if installed:
            with psycopg.connect(value.base_dsn, autocommit=True) as bootstrap:
                bootstrap.execute(
                    sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(config.schema))
                )
                for role in (
                    config.issuer_role,
                    config.administrator_role,
                    config.schema_owner_role,
                ):
                    bootstrap.execute(
                        sql.SQL("DROP OWNED BY {} CASCADE").format(sql.Identifier(role))
                    )
                    bootstrap.execute(sql.SQL("DROP ROLE {}").format(sql.Identifier(role)))


@pytest.fixture
def active_authority(authority):
    # Test-only injection. No production path grants this transition or these
    # table privileges. The production catalog is restored before testing it.
    with psycopg.connect(authority.base_dsn) as bootstrap:
        schema = sql.Identifier(authority.config.schema)
        bootstrap.execute(
            sql.SQL("ALTER TABLE {}.fences DISABLE TRIGGER lifecycle_update").format(schema)
        )
        bootstrap.execute(
            sql.SQL("UPDATE {}.fences SET state='ACTIVE',pending_operation_id=NULL").format(schema)
        )
        bootstrap.execute(
            sql.SQL("ALTER TABLE {}.fences ENABLE ALWAYS TRIGGER lifecycle_update").format(schema)
        )
    return authority


def _request(
    value, *, role=ROOT, target="REVOKED", operation_id=None, generation=1, predecessor="ACTIVE"
):
    return {
        "operation_id": operation_id or "isolated-test-close-" + uuid.uuid4().hex,
        "semantic_role": role,
        "identity": value.binding(role)["identity"],
        "generation": generation,
        "predecessor_state": predecessor,
        "target_state": target,
    }


def _begin(value, request):
    with _transaction(value, administrator=True) as administrator:
        return _call(administrator, value, "begin_close", (Jsonb(request),))


def _resolve(value, request, pending):
    with psycopg.connect(value.dsn(True)) as administrator:
        return _call(
            administrator,
            value,
            "read_operation",
            (request["semantic_role"], request["operation_id"], pending["request_sha256"]),
        )


def _receipt(value, request, pending):
    binding = value.binding(request["semantic_role"])
    receipt = json.dumps(
        {
            "operation_id": request["operation_id"],
            "request_sha256": pending["request_sha256"],
            "semantic_role": request["semantic_role"],
            "identity": binding["identity"],
            "public_material_identity": binding["public_material_identity"],
            "key_version": binding["key_version"],
            "generation": request["generation"] + 1,
            "state": request["target_state"],
            "custody_receipt_sha256": hashlib.sha256(
                b"isolated-test-public-custody-receipt"
            ).hexdigest(),
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return receipt


def _assert_completion_blocked(value, request, pending, receipt=None):
    if receipt is None:
        receipt = _receipt(value, request, pending)
    with _transaction(value, administrator=True) as administrator:
        with pytest.raises(psycopg.errors.InsufficientPrivilege):
            _call(
                administrator,
                value,
                "complete_close",
                (request["operation_id"], pending["request_sha256"], receipt),
            )
    assert _fence(value, request["semantic_role"]) == (
        "PENDING_CLOSED",
        request["generation"],
        request["operation_id"],
    )
    assert _resolve(value, request, pending)["phase"] == "PENDING_CLOSED"
    with psycopg.connect(value.base_dsn) as observer:
        assert observer.execute(
            sql.SQL("SELECT count(*) FROM {}.completions WHERE operation_id=%s").format(
                sql.Identifier(value.config.schema)
            ),
            (request["operation_id"],),
        ).fetchone() == (0,)


def _inject_test_only_closed_fixture(value, request):
    """Set historical fixture rows; this is never a completion authority/proof.

    Keep public observation/rotation regressions independent of the disabled
    completion port. Only an isolated database superuser changes these rows, and
    the reviewed trigger catalog is restored in the same transaction. No receipt
    or completion record is fabricated by this setup.
    """
    with psycopg.connect(value.base_dsn) as bootstrap:
        assert bootstrap.info.dbname.startswith("isolated_lifecycle_")
        assert bootstrap.execute(
            "SELECT rolsuper FROM pg_catalog.pg_roles WHERE rolname=session_user"
        ).fetchone() == (True,)
        schema = sql.Identifier(value.config.schema)
        bootstrap.execute(
            sql.SQL("ALTER TABLE {}.fences DISABLE TRIGGER lifecycle_update").format(schema)
        )
        updated = bootstrap.execute(
            sql.SQL(
                "UPDATE {}.fences SET state=%s,generation=generation+1,pending_operation_id=NULL "
                "WHERE semantic_role=%s AND identity=%s AND state='PENDING_CLOSED' "
                "AND generation=%s AND pending_operation_id=%s"
            ).format(schema),
            (
                request["target_state"],
                request["semantic_role"],
                request["identity"],
                request["generation"],
                request["operation_id"],
            ),
        )
        assert updated.rowcount == 1
        bootstrap.execute(
            sql.SQL("ALTER TABLE {}.fences ENABLE ALWAYS TRIGGER lifecycle_update").format(schema)
        )
        bootstrap.execute(
            sql.SQL("INSERT INTO {}.observations VALUES(%s,%s,%s,%s)").format(schema),
            (
                request["semantic_role"],
                request["identity"],
                request["generation"] + 1,
                request["target_state"],
            ),
        )


def _fence(value, role=ROOT):
    with psycopg.connect(value.base_dsn) as observer:
        return observer.execute(
            sql.SQL(
                "SELECT state,generation,pending_operation_id FROM {}.fences WHERE semantic_role=%s"
            ).format(sql.Identifier(value.config.schema)),
            (role,),
        ).fetchone()


def _assert_gate(value, bindings=None):
    with _transaction(value) as issuer:
        return _call(
            issuer, value, "assert_active_fences", (Jsonb(list(bindings or value.bindings)),)
        )


def test_production_bootstrap_is_closed_and_has_no_activation_or_raw_write_port(authority):
    assert _fence(authority)[0:2] == ("PENDING_CLOSED", 1)
    with pytest.raises(psycopg.errors.CheckViolation, match="closed or mismatched"):
        _assert_gate(authority)
    sources = lifecycle.reviewed_production_lifecycle_function_sources(authority.config)
    assert not any("activate" in signature for signature in sources)
    for administrator in (False, True):
        with _transaction(authority, administrator=administrator) as connection:
            with pytest.raises(psycopg.errors.InsufficientPrivilege):
                connection.execute(
                    sql.SQL("UPDATE {}.fences SET state='ACTIVE'").format(
                        sql.Identifier(authority.config.schema)
                    )
                )


@pytest.mark.parametrize("administrator", [False, True])
def test_direct_roles_cannot_call_the_other_administrative_or_issuer_port(
    active_authority, administrator
):
    value = active_authority
    with _transaction(value, administrator=administrator) as connection:
        with pytest.raises(psycopg.errors.InsufficientPrivilege):
            if administrator:
                _call(connection, value, "assert_active_fences", (Jsonb(list(value.bindings)),))
            else:
                _call(connection, value, "begin_close", (Jsonb(_request(value)),))
    with psycopg.connect(value.base_dsn) as bootstrap:
        bootstrap.execute("SET TRANSACTION ISOLATION LEVEL SERIALIZABLE")
        with pytest.raises(psycopg.errors.InsufficientPrivilege, match="direct production issuer"):
            _call(bootstrap, value, "assert_active_fences", (Jsonb(list(value.bindings)),))


@pytest.mark.parametrize(
    "change", ["missing", "duplicate", "generation", "identity", "material", "authority", "extra"]
)
def test_incomplete_or_substituted_fences_fail_closed(active_authority, change):
    value = active_authority
    bindings = [dict(binding) for binding in value.bindings]
    if change == "missing":
        bindings.pop()
    elif change == "duplicate":
        bindings[0] = dict(bindings[1])
    else:
        key, changed = {
            "generation": ("lifecycle_generation", 2),
            "identity": ("identity", "caller-substituted-signer"),
            "material": ("public_material_identity", "sha256:" + "f" * 64),
            "authority": ("authority_identity", "caller-substituted-authority"),
            "extra": ("TEST_ONLY", True),
        }[change]
        bindings[0][key] = changed
    with pytest.raises(psycopg.errors.CheckViolation):
        _assert_gate(value, bindings)
    assert _fence(value) == ("ACTIVE", 1, None)


def test_rotation_after_preparation_rejects_old_identity_and_generation(active_authority):
    value = active_authority
    prepared = tuple(dict(binding) for binding in value.bindings)
    request = _request(value, target="VERIFY_ONLY")
    _begin(value, request)
    _inject_test_only_closed_fixture(value, request)
    successor = value.binding()
    successor.update(
        identity=successor["identity"] + "-successor",
        public_material_identity="sha256:" + "f" * 64,
        key_version=2,
        lifecycle_generation=3,
    )
    enrollment = {
        "operation_id": "isolated-test-rotation",
        "predecessor_identity": request["identity"],
        "expected_generation": 2,
        "successor": successor,
    }
    with _transaction(value, administrator=True) as administrator:
        result = _call(administrator, value, "enroll_closed_successor", (Jsonb(enrollment),))
    assert result["phase"] == "PENDING_CLOSED"
    with pytest.raises(psycopg.errors.CheckViolation):
        _assert_gate(value, prepared)
    with _transaction(value) as issuer:
        historical = _call(issuer, value, "read_historical_fence", (ROOT, request["identity"], 1))
    assert historical["current_state"] == "VERIFY_ONLY"
    assert historical["new_operations_authorized"] is False


def test_verify_only_preserves_historical_read_without_new_operation_authority(active_authority):
    value = active_authority
    request = _request(value, target="VERIFY_ONLY")
    pending = _begin(value, request)
    _assert_completion_blocked(value, request, pending)
    _inject_test_only_closed_fixture(value, request)
    assert _fence(value) == ("VERIFY_ONLY", 2, None)
    with _transaction(value) as issuer:
        historical = _call(issuer, value, "read_historical_fence", (ROOT, request["identity"], 1))
    assert historical["historical_state"] == "ACTIVE"
    assert historical["new_operations_authorized"] is False
    with pytest.raises(psycopg.errors.CheckViolation):
        _assert_gate(value)
    # Even a previously closed public fixture cannot authorize receipt replay.
    with _transaction(value, administrator=True) as administrator:
        with pytest.raises(psycopg.errors.InsufficientPrivilege):
            _call(
                administrator,
                value,
                "complete_close",
                (
                    request["operation_id"],
                    pending["request_sha256"],
                    _receipt(value, request, pending),
                ),
            )
    revoked = _request(value, generation=2, predecessor="VERIFY_ONLY")
    _begin(value, revoked)
    _inject_test_only_closed_fixture(value, revoked)
    with _transaction(value) as issuer:
        assert _call(issuer, value, "read_historical_fence", (ROOT, request["identity"], 1)) is None


def _close_process(value, request, ready, results, *, crash=None):
    connection = psycopg.connect(value.dsn(True))
    connection.execute("SET TRANSACTION ISOLATION LEVEL SERIALIZABLE")
    results.put(("backend", connection.info.backend_pid))
    ready.wait(10)
    try:
        result = _call(connection, value, "begin_close", (Jsonb(request),))
        if crash == "before_commit":
            os._exit(71)
        connection.commit()
        if crash == "after_commit":
            os._exit(72)
        results.put(("result", result))
    except psycopg.Error as exc:
        results.put(("error", exc.sqlstate))
    finally:
        connection.close()


def _join(process, expected=0):
    process.join(10)
    if process.is_alive():
        process.kill()
        process.join(5)
        pytest.fail("isolated lifecycle process deadlocked")
    assert process.exitcode == expected


@requires_fork
def test_revocation_waits_for_issuer_transaction_then_closes_before_next_gate(active_authority):
    value = active_authority
    context = multiprocessing.get_context("fork")
    ready, results = context.Event(), context.Queue()
    with _transaction(value) as issuer:
        _call(issuer, value, "assert_active_fences", (Jsonb(list(value.bindings)),))
        process = context.Process(
            target=_close_process, args=(value, _request(value), ready, results)
        )
        process.start()
        kind, pid = results.get(timeout=5)
        assert kind == "backend"
        ready.set()
        deadline = time.monotonic() + 5
        while True:
            with psycopg.connect(value.base_dsn) as observer:
                blocked = observer.execute(
                    "SELECT cardinality(pg_catalog.pg_blocking_pids(%s))>0", (pid,)
                ).fetchone()[0]
            if blocked:
                break
            assert time.monotonic() < deadline, "administrator did not wait for real issuer locks"
            time.sleep(0.01)
        assert _fence(value) == ("ACTIVE", 1, None)
    _join(process)
    status, response = results.get(timeout=5)
    assert (status, response["phase"]) == ("result", "PENDING_CLOSED")
    with pytest.raises(psycopg.errors.CheckViolation):
        _assert_gate(value)


@requires_fork
def test_two_processes_compete_for_one_lifecycle_slot_without_duplicate_close(active_authority):
    value = active_authority
    context = multiprocessing.get_context("fork")
    ready, results = context.Event(), context.Queue()
    processes = [
        context.Process(target=_close_process, args=(value, _request(value), ready, results))
        for _ in range(2)
    ]
    for process in processes:
        process.start()
    assert all(results.get(timeout=5)[0] == "backend" for _ in processes)
    ready.set()
    for process in processes:
        _join(process)
    outcomes = [results.get(timeout=5) for _ in processes]
    assert sum(kind == "result" for kind, _ in outcomes) == 1
    assert next(response for kind, response in outcomes if kind == "error") in {"23514", "40001"}
    with psycopg.connect(value.base_dsn) as observer:
        count = observer.execute(
            sql.SQL(
                "SELECT count(*) FROM {}.operations WHERE request ? 'predecessor_state'"
            ).format(sql.Identifier(value.config.schema))
        ).fetchone()[0]
    assert count == 1


@requires_fork
@pytest.mark.parametrize(
    "phase,exitcode,expected",
    [("before_commit", 71, "ACTIVE"), ("after_commit", 72, "PENDING_CLOSED")],
)
def test_crash_before_and_after_commit_preserves_exact_durable_close(
    active_authority, phase, exitcode, expected
):
    value = active_authority
    request = _request(value)
    context = multiprocessing.get_context("fork")
    ready, results = context.Event(), context.Queue()
    process = context.Process(
        target=_close_process, args=(value, request, ready, results), kwargs={"crash": phase}
    )
    process.start()
    assert results.get(timeout=5)[0] == "backend"
    ready.set()
    _join(process, exitcode)
    assert _fence(value)[0] == expected
    if phase == "after_commit":
        replay = _begin(value, request)
        assert replay["phase"] == "PENDING_CLOSED"
        _assert_completion_blocked(value, request, replay)


def test_lost_native_commit_response_requires_exact_positive_durable_resolution(active_authority):
    value = active_authority
    request = _request(value)
    connection = psycopg.connect(value.dsn(True))
    try:
        connection.execute("SET TRANSACTION ISOLATION LEVEL SERIALIZABLE")
        pending = _call(connection, value, "begin_close", (Jsonb(request),))
        with socket.fromfd(connection.pgconn.socket, socket.AF_INET, socket.SOCK_STREAM) as wire:
            wire.shutdown(socket.SHUT_RD)
        with pytest.raises(psycopg.Error):
            connection.commit()
    finally:
        connection.close()
    deadline = time.monotonic() + 5
    while _fence(value)[0] != "PENDING_CLOSED":
        assert time.monotonic() < deadline, "successful native COMMIT was not retained"
        time.sleep(0.01)
    resolved = _resolve(value, request, pending)
    assert resolved["phase"] == "PENDING_CLOSED"
    assert resolved["request"] == request
    assert resolved["request_sha256"] == pending["request_sha256"]
    altered = dict(request, target_state="VERIFY_ONLY")
    with pytest.raises(psycopg.errors.UniqueViolation):
        _begin(value, altered)
    _assert_completion_blocked(value, request, pending)


def test_restart_pending_close_never_reopens_or_accepts_exact_receipt_replay(active_authority):
    if not (
        os.environ.get("DUDZIAN_TEST_POSTGRES_RESTART_COMMAND")
        or os.environ.get("DUDZIAN_TEST_POSTGRES_RESTART_CONTAINER_ID")
    ):
        pytest.skip("explicit isolated PostgreSQL restart controller required")
    value = active_authority
    request = _request(value)
    pending = _begin(value, request)
    _restart_controlled_postgresql()
    assert _fence(value) == ("PENDING_CLOSED", 1, request["operation_id"])
    resolved = _resolve(value, request, pending)
    assert resolved["phase"] == "PENDING_CLOSED"
    assert resolved["request"] == request
    with pytest.raises(psycopg.errors.CheckViolation):
        _assert_gate(value)
    receipt = _receipt(value, request, pending)
    _assert_completion_blocked(value, request, pending, receipt)
    _restart_controlled_postgresql()
    _assert_completion_blocked(value, request, pending, receipt)


def test_resolution_is_exact_and_absence_does_not_change_a_closed_fence(active_authority):
    value = active_authority
    request = _request(value)
    pending = _begin(value, request)
    resolved = _resolve(value, request, pending)
    assert resolved["phase"] == "PENDING_CLOSED"
    assert resolved["request"] == request
    with pytest.raises(psycopg.errors.CheckViolation, match="operation differs"):
        _resolve(value, request, dict(pending, request_sha256="0" * 64))
    unknown = dict(request, operation_id="unobserved-operation")
    assert _resolve(value, unknown, pending) is None
    assert _fence(value) == ("PENDING_CLOSED", 1, request["operation_id"])
    with pytest.raises(psycopg.errors.ActiveSqlTransaction, match="fresh lock-synchronized"):
        with _transaction(value, administrator=True) as administrator:
            _call(
                administrator,
                value,
                "read_operation",
                (ROOT, request["operation_id"], pending["request_sha256"]),
            )


def _resolution_process(value, request, pending, results):
    results.put(_resolve(value, request, pending))


@requires_fork
def test_fresh_process_resolves_pending_close_without_reopening_or_repeating_write(
    active_authority,
):
    value = active_authority
    request = _request(value)
    pending = _begin(value, request)
    context = multiprocessing.get_context("fork")
    results = context.Queue()
    process = context.Process(target=_resolution_process, args=(value, request, pending, results))
    process.start()
    resolved = results.get(timeout=5)
    _join(process)
    assert resolved["phase"] == "PENDING_CLOSED"
    assert resolved["request"] == request
    assert _fence(value) == ("PENDING_CLOSED", 1, request["operation_id"])
    _assert_completion_blocked(value, request, pending)


@pytest.mark.parametrize("change", ["malformed", "generation", "forged", "hash_only"])
def test_caller_completion_cannot_replace_exact_retained_public_facts(active_authority, change):
    value = active_authority
    request = _request(value)
    pending = _begin(value, request)
    receipt = _receipt(value, request, pending)
    if change == "malformed":
        receipt = b"{}"
    elif change == "generation":
        changed = json.loads(receipt)
        changed["generation"] = 9
        receipt = json.dumps(changed).encode()
    elif change == "hash_only":
        receipt = hashlib.sha256(receipt).hexdigest().encode()
    _assert_completion_blocked(value, request, pending, receipt)
    _assert_completion_blocked(value, request, pending, receipt)


@pytest.mark.parametrize("port", ["completion", "resolution"])
def test_uncommitted_close_cannot_authorize_completion_or_positive_resolution(
    active_authority, port
):
    value = active_authority
    request = _request(value)
    with psycopg.connect(value.dsn(True)) as administrator:
        pending = _call(administrator, value, "begin_close", (Jsonb(request),))
        assert _fence(value) == ("ACTIVE", 1, None)
        error, message = (
            (psycopg.errors.InsufficientPrivilege, "permission denied")
            if port == "completion"
            else (psycopg.errors.ActiveSqlTransaction, "close.*(commit|observation)")
        )
        with pytest.raises(error, match=message):
            if port == "completion":
                _call(
                    administrator,
                    value,
                    "complete_close",
                    (
                        request["operation_id"],
                        pending["request_sha256"],
                        _receipt(value, request, pending),
                    ),
                )
            else:
                _call(
                    administrator,
                    value,
                    "read_operation",
                    (ROOT, request["operation_id"], pending["request_sha256"]),
                )
    assert _fence(value) == ("ACTIVE", 1, None)
    assert _resolve(value, request, pending) is None


@pytest.fixture
def simulated_installed_administrator(active_authority, monkeypatch):
    value = active_authority
    with psycopg.connect(value.base_dsn) as bootstrap:
        bootstrap.execute("SET TRANSACTION ISOLATION LEVEL SERIALIZABLE")
        identity = read_atomic_root_proof_physical_identity(bootstrap, value.config.installation_id)
        pin = pin_production_root_lifecycle_catalog(
            bootstrap, value.config, expected_identity=identity
        )
    installed = SimpleNamespace(pin=pin)
    boundary = coordinator.ProductionRootLifecycleBoundary(installed, "administrator")
    # Isolated SQL qualification only: this fixture does not qualify an installed
    # OS service, seven separate system accounts, peer auth or signer processes.
    # The actual database pin, physical identity, direct login, ACLs and catalog
    # qualification all execute unchanged in the coordinator's real transaction.
    monkeypatch.setattr(boundary, "_current", lambda: installed)
    monkeypatch.setattr(boundary, "_require_peer_authentication", lambda *_: None)
    monkeypatch.setattr(coordinator, "_qualify_principals", lambda *_: None)
    return value, boundary


def test_coordinator_acknowledges_durable_close_but_blocks_administrator_receipt(
    simulated_installed_administrator,
):
    value, boundary = simulated_installed_administrator
    request = _request(value)
    with psycopg.connect(value.dsn(True), autocommit=True) as administrator:
        pending = boundary.begin_close(administrator, request)
        assert administrator.info.transaction_status is psycopg.pq.TransactionStatus.IDLE
        assert _fence(value) == ("PENDING_CLOSED", 1, request["operation_id"])
        with pytest.raises(
            coordinator.ProductionRootLifecycleError,
            match="INDEPENDENT_CUSTODY_COMPLETION_AUTHORITY_REQUIRED",
        ):
            boundary.publish_administrator_completion(
                administrator,
                request["operation_id"],
                pending["request_sha256"],
                _receipt(value, request, pending),
            )
        assert administrator.info.transaction_status is psycopg.pq.TransactionStatus.IDLE
        resolved = boundary.resolve_close(
            administrator, ROOT, request["operation_id"], pending["request_sha256"]
        )
    assert resolved["phase"] == "PENDING_CLOSED"
    assert resolved["request"] == request
    assert _fence(value) == ("PENDING_CLOSED", 1, request["operation_id"])


def test_coordinator_absent_resolution_stays_indeterminate_without_retry(
    simulated_installed_administrator,
):
    value, boundary = simulated_installed_administrator
    with psycopg.connect(value.dsn(True), autocommit=True) as administrator:
        with pytest.raises(coordinator.ProductionRootLifecycleIndeterminate):
            boundary.resolve_close(administrator, ROOT, "unobserved-operation", "0" * 64)
    assert _fence(value) == ("ACTIVE", 1, None)
    with psycopg.connect(value.base_dsn) as observer:
        assert observer.execute(
            sql.SQL(
                "SELECT count(*) FROM {}.operations WHERE request ? 'predecessor_state'"
            ).format(sql.Identifier(value.config.schema))
        ).fetchone() == (0,)


def test_actual_trust_authenticated_connection_cannot_qualify_installed_peer(authority):
    installed = SimpleNamespace(principal=lambda _role: ("isolated-service", 1234, 1234, "role"))
    boundary = coordinator.ProductionRootLifecycleBoundary(installed, "administrator")
    with psycopg.connect(authority.dsn(True), autocommit=True) as connection:
        # This isolated cluster uses trust solely for storage tests. PostgreSQL
        # reports NULL and the production service boundary must refuse it.
        assert connection.execute("SELECT system_user").fetchone() == (None,)
        with pytest.raises(coordinator.ProductionRootLifecycleError, match="PEER_AUTHENTICATION"):
            boundary._require_peer_authentication(connection, installed)


def test_coordinator_native_commit_response_loss_is_unknown_and_never_retries(
    simulated_installed_administrator, monkeypatch
):
    value, boundary = simulated_installed_administrator
    request = _request(value)
    executed = []
    execute = psycopg.Connection.execute

    def disconnect_after_close_write(connection, query, *args, **kwargs):
        result = execute(connection, query, *args, **kwargs)
        text = query.as_string(connection) if isinstance(query, sql.Composable) else str(query)
        if f'"{value.config.schema}"."begin_close"' in text:
            executed.append(connection.info.backend_pid)
            with socket.fromfd(
                connection.pgconn.socket, socket.AF_INET, socket.SOCK_STREAM
            ) as wire:
                wire.shutdown(socket.SHUT_RD)
        return result

    with monkeypatch.context() as cut:
        cut.setattr(psycopg.Connection, "execute", disconnect_after_close_write)
        with psycopg.connect(value.dsn(True), autocommit=True) as administrator:
            with pytest.raises(
                coordinator.ProductionRootLifecycleIndeterminate, match="COMMIT_OUTCOME_UNKNOWN"
            ):
                boundary.begin_close(administrator, request)
    assert len(executed) == 1
    deadline = time.monotonic() + 5
    while _fence(value)[0] != "PENDING_CLOSED":
        assert time.monotonic() < deadline, "native COMMIT did not retain the closed fence"
        time.sleep(0.01)
    with psycopg.connect(value.base_dsn) as observer:
        digest = observer.execute(
            sql.SQL("SELECT request_sha256 FROM {}.operations WHERE operation_id=%s").format(
                sql.Identifier(value.config.schema)
            ),
            (request["operation_id"],),
        ).fetchone()[0]
    with psycopg.connect(value.dsn(True), autocommit=True) as administrator:
        resolved = boundary.resolve_close(administrator, ROOT, request["operation_id"], digest)
    assert resolved["phase"] == "PENDING_CLOSED"
    assert resolved["request"] == request
    assert len(executed) == 1

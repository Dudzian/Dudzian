"""Odczyt istniejącej władzy CHA nie może zmieniać jej trwałego stanu."""

from __future__ import annotations

import os
import sqlite3
from dataclasses import replace
from pathlib import Path

import pytest

import bot_core.cha_attempt_store as cha
from tests.licensing import test_cha_root_proof_signed_attempt as signed_tests
from tests.security.test_cha_attempt_store_production_local import evidence

store_flow = signed_tests.store_flow


def authority_snapshot(path: Path) -> tuple[object, ...]:
    connection = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)
    try:
        history = tuple(connection.iterdump())
    finally:
        connection.close()
    files = tuple(
        (
            candidate.name,
            candidate.stat().st_mode & 0o777,
            None if candidate.name.endswith("-shm") else candidate.read_bytes(),
        )
        for candidate in sorted(path.parent.iterdir())
        if candidate.is_file()
    )
    return history, files, path.parent.stat().st_mode & 0o777


def complete_attempt(store, current, keys, identities):
    for private, signer in zip(keys, identities, strict=True):
        current = signed_tests.checkpoint(store, current, private, signer)
    identity = signed_tests.final_identity(store, current)
    return store.finalize_attempt(identity, expected_fence=current.fence)


@pytest.mark.parametrize("checkpoints", [0, 1, 2, 3])
def test_readonly_retains_exact_current_and_signed_request_across_reopens(store_flow, checkpoints):
    path, auth, current, keys, identities = store_flow
    retained = None
    with cha.SQLiteCHAAttemptStore(path, auth.trust_domain) as writer:
        for private, signer in list(zip(keys, identities, strict=True))[:checkpoints]:
            current = signed_tests.checkpoint(writer, current, private, signer)
        if checkpoints == 3:
            identity = signed_tests.final_identity(writer, current)
            current = writer.finalize_attempt(identity, expected_fence=current.fence)
        if checkpoints:
            retained = writer.signed_request(auth.logical_operation_id)
    before = authority_snapshot(path)
    for _ in range(2):
        with cha.SQLiteCHAAttemptStore.open_readonly(path, auth.trust_domain) as reader:
            assert reader.attempt(auth.logical_operation_id) == current
            assert reader.identity == writer.identity
            assert reader.capabilities.authoritative_reads
            assert reader.capabilities.durable_state
            assert not reader.capabilities.compare_and_swap
            assert reader._connection.execute("PRAGMA query_only").fetchone() == (1,)
            if retained is None:
                with pytest.raises(cha.AttemptNotFoundError):
                    reader.signed_request(auth.logical_operation_id)
            else:
                assert reader.signed_request(auth.logical_operation_id) == retained
    assert authority_snapshot(path) == before


def test_readonly_sees_committed_wal_without_configuration_or_permission_changes(
    store_flow, monkeypatch
):
    path, auth, current, keys, identities = store_flow
    with cha.SQLiteCHAAttemptStore(path, auth.trust_domain) as writer:
        current = complete_attempt(writer, current, keys, identities)
        retained = writer.signed_request(auth.logical_operation_id)
        if os.name == "posix":
            path.chmod(0o640)
            path.parent.chmod(0o750)
        assert Path(str(path) + "-wal").is_file()
        assert Path(str(path) + "-shm").is_file()
        before = authority_snapshot(path)

        def forbidden(*args, **kwargs):
            raise AssertionError("odczyt nie może tworzyć ani konfigurować władzy CHA")

        for name in ("_configure", "_create_schema", "_migrate_v5"):
            monkeypatch.setattr(cha.SQLiteCHAAttemptStore, name, forbidden)
        monkeypatch.setattr(cha.os, "chmod", forbidden)
        monkeypatch.setattr(Path, "mkdir", forbidden)
        with cha.SQLiteCHAAttemptStore.open_readonly(path, auth.trust_domain) as reader:
            assert reader.attempt(auth.logical_operation_id) == current
            assert reader.signed_request(auth.logical_operation_id) == retained
        assert authority_snapshot(path) == before


@pytest.mark.parametrize("existing_parent", [False, True])
def test_readonly_missing_database_creates_no_files_or_directories(tmp_path, existing_parent):
    directory = tmp_path if existing_parent else tmp_path / "nie-istnieje"
    path = directory / "attempts.db"
    before = tuple(tmp_path.rglob("*"))
    with pytest.raises(cha.AttemptNotFoundError):
        cha.SQLiteCHAAttemptStore.open_readonly(path, "td-production")
    assert tuple(tmp_path.rglob("*")) == before
    assert not path.exists()


def test_readonly_schema_v5_is_unsupported_and_never_migrated(store_flow, monkeypatch):
    path, auth, _, _, _ = store_flow
    signed_tests.downgrade_unsigned_fixture_to_v5(path)
    before = authority_snapshot(path)

    def forbidden(*args, **kwargs):
        raise AssertionError("odczyt nie może migrować schematu v5")

    monkeypatch.setattr(cha.SQLiteCHAAttemptStore, "_migrate_v5", forbidden)
    with pytest.raises(cha.AttemptSchemaUnsupportedError, match="migration is forbidden"):
        cha.SQLiteCHAAttemptStore.open_readonly(path, auth.trust_domain)
    assert authority_snapshot(path) == before


@pytest.mark.parametrize("corruption", ["projection", "history", "schema"])
def test_readonly_corrupt_authority_fails_closed_without_repair(store_flow, corruption):
    path, auth, _, _, _ = store_flow
    with sqlite3.connect(path) as connection:
        if corruption == "projection":
            connection.execute("UPDATE current_attempts SET state='INVALID'")
        elif corruption == "history":
            connection.execute("DROP TRIGGER reservations_immutable_update")
            connection.execute("UPDATE reservations SET authorization_json=x'00'")
            connection.execute(
                "CREATE TRIGGER reservations_immutable_update BEFORE UPDATE ON reservations "
                "BEGIN SELECT RAISE(ABORT,'immutable reservation'); END"
            )
        else:
            connection.execute("DROP TRIGGER reservations_immutable_update")
    connection.close()
    before = authority_snapshot(path)
    with pytest.raises(cha.AttemptCorruptError):
        cha.SQLiteCHAAttemptStore.open_readonly(path, auth.trust_domain)
    assert authority_snapshot(path) == before


def test_readonly_trust_domain_mismatch_is_rejected_without_mutation(store_flow):
    path, _, _, _, _ = store_flow
    before = authority_snapshot(path)
    with pytest.raises(cha.AttemptConflictError):
        cha.SQLiteCHAAttemptStore.open_readonly(path, "inna-domena")
    assert authority_snapshot(path) == before


def test_readonly_non_database_file_is_corrupt_without_repair(tmp_path):
    path = tmp_path / "attempts.db"
    contents = b"uszkodzone dane poza formatem SQLite" * 32
    path.write_bytes(contents)
    before = path.stat()
    with pytest.raises(cha.AttemptCorruptError, match="database is corrupt"):
        cha.SQLiteCHAAttemptStore.open_readonly(path, "td-production")
    assert path.read_bytes() == contents
    assert path.stat().st_mtime_ns == before.st_mtime_ns
    assert tuple(tmp_path.iterdir()) == (path,)


@pytest.mark.parametrize("readonly", [False, True])
@pytest.mark.parametrize(
    "errorcode",
    [
        sqlite3.SQLITE_CORRUPT,
        sqlite3.SQLITE_NOTADB,
        sqlite3.SQLITE_CORRUPT | (3 << 8),
        sqlite3.SQLITE_NOTADB | (2 << 8),
    ],
)
def test_readonly_physical_corruption_during_read_preserves_typed_outcome(
    store_flow, readonly, errorcode
):
    path, auth, current, _, _ = store_flow
    opener = cha.SQLiteCHAAttemptStore.open_readonly if readonly else cha.SQLiteCHAAttemptStore
    with opener(path, auth.trust_domain) as reader:
        connection = reader._connection

        class FailedRead:
            failed = False

            def __getattr__(self, name):
                return getattr(connection, name)

            def execute(self, statement, *args):
                if statement.startswith("SELECT") and not self.failed:
                    self.failed = True
                    error = sqlite3.DatabaseError("zasymulowana fizyczna awaria SQLite")
                    error.sqlite_errorcode = errorcode
                    raise error
                return connection.execute(statement, *args)

        reader._connection = FailedRead()
        expected = cha.AttemptCorruptError if readonly else cha.AttemptStoreUnavailableError
        with pytest.raises(expected):
            reader.attempt(auth.logical_operation_id)
        assert not connection.in_transaction
        assert reader.attempt(auth.logical_operation_id) == current


@pytest.mark.parametrize(
    "method",
    [
        "reserve_or_resolve_attempt_id",
        "finalize_attempt",
        "prepare_signature",
        "persist_signature",
        "supersede_pre_send",
        "replace_after_authoritative_unbound",
        "record_recovery_resolution",
    ],
)
def test_readonly_rejects_every_public_write_method(store_flow, method):
    path, auth, current, keys, identities = store_flow
    with cha.SQLiteCHAAttemptStore(path, auth.trust_domain) as writer:
        current = complete_attempt(writer, current, keys, identities)
        retained = writer.signed_request(auth.logical_operation_id)
    before = authority_snapshot(path)
    attempt_id = current.reservation.issuance_attempt_id
    with cha.SQLiteCHAAttemptStore.open_readonly(path, auth.trust_domain) as reader:
        actions = {
            "reserve_or_resolve_attempt_id": lambda: reader.reserve_or_resolve_attempt_id(auth),
            "finalize_attempt": lambda: reader.finalize_attempt(
                current.identity, expected_fence=current.fence
            ),
            "prepare_signature": lambda: reader.prepare_signature(current, identities[0]),
            "persist_signature": lambda: reader.persist_signature(
                current, identities[0], retained.requester[1]
            ),
            "supersede_pre_send": lambda: reader.supersede_pre_send(
                current, replace(auth, requester_key_version=2)
            ),
            "replace_after_authoritative_unbound": (
                lambda: reader.replace_after_authoritative_unbound(
                    auth, evidence(attempt_id, auth), expected_fence=current.fence
                )
            ),
            "record_recovery_resolution": lambda: reader.record_recovery_resolution(
                auth.logical_operation_id,
                cha.RecoveryResolution(
                    attempt_id,
                    cha.AttemptState.MAY_HAVE_BEEN_SENT_OUTCOME_UNKNOWN,
                    "issuer:authenticated:unknown",
                    "5" * 64,
                ),
                expected_fence=current.fence,
            ),
        }
        with pytest.raises(cha.AttemptStoreUnavailableError, match="read-only"):
            actions[method]()
        assert reader.attempt(auth.logical_operation_id) == current
    assert authority_snapshot(path) == before


@pytest.mark.parametrize("disable_query_only", [False, True])
def test_readonly_sqlite_connection_rejects_direct_sql_writes(store_flow, disable_query_only):
    path, auth, current, _, _ = store_flow
    before = authority_snapshot(path)
    with cha.SQLiteCHAAttemptStore.open_readonly(path, auth.trust_domain) as reader:
        if disable_query_only:
            reader._connection.execute("PRAGMA query_only=OFF")
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            reader._connection.execute("DELETE FROM current_attempts")
        assert reader.attempt(auth.logical_operation_id) == current
    assert authority_snapshot(path) == before

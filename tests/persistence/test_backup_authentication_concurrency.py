from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from threading import Barrier, Event, Thread

import pytest

from bot_core.persistence.backup_authentication import (
    AuthorityKeyState,
    AuthorityScopeCondition,
    BackupArtifactAuthenticationAuthority,
    BackupArtifactVerificationResult,
    SQLiteBackupAuthenticationMetadataStore,
    SecretStorageBackupAuthenticationSecureCustody,
    StaleAuthorityRevisionError,
)
from tests.persistence.test_backup_authentication import MemoryStorage, ident, manifest
from bot_core.persistence.backup_authentication import BackupArtifactAuthenticationScope


def setup(tmp_path):
    scope = BackupArtifactAuthenticationScope(ident("acct"), ident("dev"), "PAPER")
    storage = MemoryStorage()
    metadata = SQLiteBackupAuthenticationMetadataStore(tmp_path / "authority.sqlite")
    authority = BackupArtifactAuthenticationAuthority(
        metadata, SecretStorageBackupAuthenticationSecureCustody(storage)
    )
    return scope, storage, metadata, authority


def race(left, right):
    barrier = Barrier(2)

    def run(call):
        barrier.wait()
        try:
            call()
            return "ok"
        except StaleAuthorityRevisionError:
            return "stale"

    with ThreadPoolExecutor(max_workers=2) as pool:
        return list(pool.map(run, (left, right)))


@pytest.mark.parametrize("iteration", range(5))
def test_rekey_vs_rekey(tmp_path, iteration):
    scope, _, _, authority = setup(tmp_path)
    first = authority.provision(scope)
    authority.revoke(scope, first.keys[0].authority_key_id, 1)
    assert sorted(
        race(lambda: authority.rekey(scope, 2), lambda: authority.rekey(scope, 2))
    ) == ["ok", "stale"]
    snapshot = authority.read_snapshot(scope)
    assert snapshot is not None and snapshot.authority_revision == 3
    assert sum(key.state is AuthorityKeyState.ACTIVE for key in snapshot.keys) == 1


@pytest.mark.parametrize("winner", ["rotate", "revoke"])
def test_rotate_vs_revoke_forces_requested_commit_order(tmp_path, winner):
    scope, _, _, authority = setup(tmp_path)
    first = authority.provision(scope)
    committed = Event()
    results = []
    stale_results = []
    first_call = (
        (lambda: authority.rotate(scope, 1))
        if winner == "rotate"
        else (lambda: authority.revoke(scope, first.keys[0].authority_key_id, 1))
    )
    second_call = (
        (lambda: authority.revoke(scope, first.keys[0].authority_key_id, 1))
        if winner == "rotate"
        else (lambda: authority.rotate(scope, 1))
    )

    def commit_first():
        results.append(first_call())
        committed.set()

    def attempt_stale_second():
        assert committed.wait(5)
        try:
            second_call()
        except StaleAuthorityRevisionError:
            stale_results.append("stale")

    threads = [Thread(target=commit_first), Thread(target=attempt_stale_second)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(5)
        assert not thread.is_alive()
    snapshot = authority.read_snapshot(scope)
    assert snapshot is not None and snapshot.authority_revision == 2
    assert snapshot == results[0]
    assert stale_results == ["stale"]
    if winner == "rotate":
        assert [key.state for key in snapshot.keys].count(AuthorityKeyState.ACTIVE) == 1
        assert [key.state for key in snapshot.keys].count(
            AuthorityKeyState.VERIFY_ONLY
        ) == 1
    else:
        assert snapshot.condition is AuthorityScopeCondition.PROVISIONED_WITHOUT_ACTIVE
        assert snapshot.keys[0].state is AuthorityKeyState.REVOKED


def test_revoke_verify_only_commits_before_stale_rekey(tmp_path):
    scope, _, _, authority = setup(tmp_path)
    first = authority.provision(scope)
    rotated = authority.rotate(scope, 1)
    active = next(key for key in rotated.keys if key.state is AuthorityKeyState.ACTIVE)
    old = next(
        key for key in rotated.keys if key.state is AuthorityKeyState.VERIFY_ONLY
    )
    authority.revoke(scope, active.authority_key_id, 2)
    revoked = authority.revoke(scope, old.authority_key_id, 3)
    with pytest.raises(StaleAuthorityRevisionError):
        authority.rekey(scope, 3)
    assert revoked.authority_revision == 4
    assert all(key.state is AuthorityKeyState.REVOKED for key in revoked.keys)


def test_first_revoke_commits_before_second_stale_revoke(tmp_path):
    scope2, _, _, second = setup(tmp_path)
    key = second.provision(scope2).keys[0]
    first_result = second.revoke(scope2, key.authority_key_id, 1)
    with pytest.raises(StaleAuthorityRevisionError):
        second.revoke(scope2, key.authority_key_id, 1)
    final = second.read_snapshot(scope2)
    assert final == first_result
    assert final.authority_revision == 2
    assert final.keys[0].state is AuthorityKeyState.REVOKED


class BlockingCustody:
    def __init__(self, delegate):
        self.delegate = delegate
        self.entered = Event()
        self.release = Event()
        self.block = False

    def persist(self, handle, material):
        self.delegate.persist(handle, material)

    def digest(self, handle, payload):
        if self.block:
            self.entered.set()
            assert self.release.wait(5)
        return self.delegate.digest(handle, payload)


@pytest.mark.parametrize("operation", ["verify", "rotate", "revoke"])
def test_read_before_mutation_obeys_linearization(tmp_path, operation):
    scope, storage, metadata, base = setup(tmp_path)
    first = base.provision(scope)
    item = manifest(scope)
    proof = base.authenticate(scope, item)
    blocking = BlockingCustody(SecretStorageBackupAuthenticationSecureCustody(storage))
    reader = BackupArtifactAuthenticationAuthority(metadata, blocking)
    blocking.block = True
    result = []
    call = (
        (lambda: reader.verify(scope, item, proof))
        if operation == "verify"
        else (lambda: reader.authenticate(scope, item))
    )
    thread = Thread(target=lambda: result.append(call()))
    thread.start()
    assert blocking.entered.wait(5)
    if operation == "rotate":
        base.rotate(scope, 1)
    else:
        base.revoke(scope, first.keys[0].authority_key_id, 1)
    blocking.release.set()
    thread.join(5)
    assert result
    if operation == "verify":
        assert result[0] is BackupArtifactVerificationResult.VERIFIED
        assert (
            base.verify(scope, item, proof)
            is BackupArtifactVerificationResult.REVOKED_KEY
        )
    elif operation == "rotate":
        assert (
            base.verify(scope, item, result[0])
            is BackupArtifactVerificationResult.VERIFIED
        )
    else:
        assert (
            base.verify(scope, item, result[0])
            is BackupArtifactVerificationResult.REVOKED_KEY
        )


def test_mutation_returns_its_own_snapshot_when_later_commit_wins(tmp_path):
    scope, storage, metadata, base = setup(tmp_path)
    base.provision(scope)
    committed = Event()
    release = Event()

    def hook(point):
        if point == "AFTER_METADATA_COMMIT_BEFORE_RETURN":
            committed.set()
            assert release.wait(5)

    first = BackupArtifactAuthenticationAuthority(
        metadata,
        SecretStorageBackupAuthenticationSecureCustody(storage),
        _fault_hook=hook,
    )
    result = []
    thread = Thread(target=lambda: result.append(first.rotate(scope, 1)))
    thread.start()
    assert committed.wait(5)
    base.rotate(scope, 2)
    release.set()
    thread.join(5)
    assert result[0].authority_revision == 2
    assert base.read_snapshot(scope).authority_revision == 3

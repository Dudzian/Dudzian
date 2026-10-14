"""Fast negative checks of root/history custody mutation admission.

The native keyring fixture replaces only the OS secret-storage transport. No
test creates, registers, or patches a positive lifecycle close authorization.
Genuine successful DB-first transitions belong to PostgreSQL integration tests.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import multiprocessing
import os
from pathlib import Path

import pytest

import bot_core.local_signing_custody as custody
from bot_core.local_signing_custody import (
    LocalSigningCustodyError,
    LocalSigningLifecycleError,
    NativeKeyringSigningSecretAdministrator,
    NativeKeyringSigningSecretReader,
    SigningKeyLifecycle,
)
from bot_core.postgresql_custody_lifecycle import (
    CustodyLifecycleError,
    DurableCloseAuthorization,
)
from bot_core.root_proof_issuer_substrate import ProviderRole
from tests.security._local_signing_platform import requires_native_custody_locking
from tests.security.test_local_signing_custody import (
    _admin,
    _custody_file_bytes,
    native_keyring,
    security,
)


pytestmark = requires_native_custody_locking
ROOT_HISTORY_ROLES = (
    ProviderRole.ROOT_PROOF_SIGNING,
    ProviderRole.HISTORY_ATTESTATION_SIGNING,
)


@pytest.fixture(params=ROOT_HISTORY_ROLES)
def authority(tmp_path, native_keyring, security, request):
    return custody.provision_local_signing_authority(
        tmp_path,
        _admin(tmp_path, security, request.param),
        security=security,
        role=request.param,
    )


def _successor(authority):
    return replace(
        authority,
        lifecycle_generation=authority.lifecycle_generation + 1,
        lifecycle_state=SigningKeyLifecycle.VERIFY_ONLY,
    )


@pytest.mark.parametrize("exact_credential", [False, True])
def test_legacy_public_transition_has_no_unenrolled_v1_exception(
    tmp_path, native_keyring, security, authority, exact_credential
) -> None:
    before_files = _custody_file_bytes(tmp_path)
    before_secrets = dict(native_keyring)
    arguments = dict(
        security=security,
        role=authority.provider_role,
        target=SigningKeyLifecycle.VERIFY_ONLY,
    )
    if exact_credential:
        arguments.update(
            credential_identity=authority.credential_identity(),
            expected_lifecycle_generation=authority.lifecycle_generation,
        )
        transition = custody.transition_local_signing_credential_lifecycle
    else:
        transition = custody.transition_local_signing_lifecycle
    with pytest.raises(LocalSigningLifecycleError):
        transition(tmp_path, **arguments)
    assert _custody_file_bytes(tmp_path) == before_files
    assert native_keyring == before_secrets


@pytest.mark.parametrize("authorization_kind", ["missing", "lookalike", "unregistered"])
def test_missing_or_forged_close_cannot_mutate_through_locked_transition(
    tmp_path, native_keyring, security, authority, authorization_kind
) -> None:
    descriptor = b"{}"
    authorization = {
        "missing": None,
        "lookalike": object(),
        "unregistered": DurableCloseAuthorization(descriptor),
    }[authorization_kind]
    before_files = _custody_file_bytes(tmp_path)
    before_secrets = dict(native_keyring)
    with custody._custody_lock(tmp_path, exclusive=True):
        with pytest.raises((LocalSigningCustodyError, CustodyLifecycleError)):
            custody._transition_exact_locked(
                tmp_path,
                security,
                authority.provider_role,
                authority.credential_identity(),
                authority.lifecycle_generation,
                SigningKeyLifecycle.VERIFY_ONLY,
                close_authorization=authorization,
                operation_descriptor=None if authorization is None else descriptor,
            )
    assert _custody_file_bytes(tmp_path) == before_files
    assert native_keyring == before_secrets


@pytest.mark.parametrize("admission_kind", ["missing", "lookalike", "unregistered"])
def test_low_level_writer_rejects_caller_fabricated_mutation_admission(
    tmp_path, native_keyring, authority, admission_kind
) -> None:
    descriptor = b"{}"
    admission = {
        "missing": None,
        "lookalike": object(),
        "unregistered": object.__new__(custody._LocalSigningMutationAdmission),
    }[admission_kind]
    before_files = _custody_file_bytes(tmp_path)
    before_secrets = dict(native_keyring)
    with custody._custody_lock(tmp_path, exclusive=True):
        with pytest.raises(LocalSigningLifecycleError):
            custody._replace_record(
                custody._record_path(tmp_path, authority.provider_role),
                custody._encode_record(_successor(authority)),
                close_authorization=DurableCloseAuthorization(descriptor),
                operation_descriptor=descriptor,
                mutation_admission=admission,
            )
    assert _custody_file_bytes(tmp_path) == before_files
    assert native_keyring == before_secrets


@pytest.mark.parametrize("destination", ["current", "arbitrary", "retained", "prepared"])
@pytest.mark.parametrize("writer", ["new", "replace"])
def test_low_level_writer_cannot_publish_root_history_successor_without_admission(
    tmp_path, native_keyring, authority, destination, writer
) -> None:
    role = authority.provider_role
    paths = {
        "current": custody._record_path(tmp_path, role),
        "arbitrary": tmp_path / "other-name.json",
        "retained": custody._retained_path(tmp_path, role, authority.credential_identity()),
        "prepared": custody._rotation_paths(tmp_path, role)[2],
    }
    path = paths[destination]
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    before_files = _custody_file_bytes(tmp_path)
    before_secrets = dict(native_keyring)
    write = custody._write_new_record if writer == "new" else custody._replace_record
    with pytest.raises(LocalSigningCustodyError):
        write(path, custody._encode_record(_successor(authority)))
    assert _custody_file_bytes(tmp_path) == before_files
    assert native_keyring == before_secrets


@pytest.mark.parametrize("destination", ["current", "retained"])
@pytest.mark.parametrize("writer", ["new", "replace"])
@pytest.mark.parametrize("payload", [b"{", b"{}"])
def test_malformed_payload_cannot_escape_reserved_root_history_writer_admission(
    tmp_path, native_keyring, authority, destination, writer, payload
) -> None:
    role = authority.provider_role
    path = (
        custody._record_path(tmp_path, role)
        if destination == "current"
        else custody._retained_path(tmp_path, role, authority.credential_identity())
    )
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    before_files = _custody_file_bytes(tmp_path)
    before_secrets = dict(native_keyring)
    write = custody._write_new_record if writer == "new" else custody._replace_record
    with pytest.raises(LocalSigningCustodyError):
        write(path, payload)
    assert _custody_file_bytes(tmp_path) == before_files
    assert native_keyring == before_secrets


@pytest.mark.parametrize("administrator_kind", ["missing", "reader", "subclass", "wrong-role"])
def test_initial_root_history_writer_requires_exact_scoped_native_administrator(
    tmp_path, native_keyring, security, authority, administrator_kind
) -> None:
    class AdministratorSubclass(NativeKeyringSigningSecretAdministrator):
        pass

    role = authority.provider_role
    opposite_role = (
        ROOT_HISTORY_ROLES[1] if role is ROOT_HISTORY_ROLES[0] else ROOT_HISTORY_ROLES[0]
    )
    arguments = dict(index_path=tmp_path / "keyring-index.json", security=security, role=role)
    administrator = {
        "missing": None,
        "reader": NativeKeyringSigningSecretReader(**arguments),
        "subclass": AdministratorSubclass(**arguments),
        "wrong-role": _admin(tmp_path, security, opposite_role),
    }[administrator_kind]
    path = custody._record_path(tmp_path, role)
    # Offline fixture removal models attempting an initial metadata publication.
    path.unlink()
    before_files = _custody_file_bytes(tmp_path)
    before_secrets = dict(native_keyring)
    with custody._custody_lock(tmp_path, exclusive=True) as lease:
        with pytest.raises(LocalSigningCustodyError):
            custody._write_new_record(
                path,
                custody._encode_record(authority),
                provisioning_administrator=administrator,
                custody_lease=lease,
            )
    assert _custody_file_bytes(tmp_path) == before_files
    assert native_keyring == before_secrets


def test_provisioning_capability_cannot_publish_a_lifecycle_successor(
    tmp_path, native_keyring, security, authority
) -> None:
    path = custody._record_path(tmp_path, authority.provider_role)
    administrator = _admin(tmp_path, security, authority.provider_role)
    before_files = _custody_file_bytes(tmp_path)
    before_secrets = dict(native_keyring)
    with custody._custody_lock(tmp_path, exclusive=True) as lease:
        with pytest.raises(LocalSigningCustodyError):
            custody._write_new_record(
                path,
                custody._encode_record(_successor(authority)),
                provisioning_administrator=administrator,
                custody_lease=lease,
            )
    assert _custody_file_bytes(tmp_path) == before_files
    assert native_keyring == before_secrets


@pytest.mark.parametrize("writer", ["new", "replace"])
def test_low_level_writer_rejects_path_subclass_before_caller_semantic_methods(
    tmp_path, native_keyring, authority, writer
) -> None:
    calls = []

    class CallerPath(type(Path())):
        @property
        def name(self):
            calls.append("name")
            return "freshness-authority-finalization-signing.json"

        @property
        def parent(self):
            calls.append("parent")
            return tmp_path

        def exists(self):
            calls.append("exists")
            return False

        def __fspath__(self):
            calls.append("fspath")
            return str(tmp_path / "caller-selected-destination.json")

    path = CallerPath(custody._record_path(tmp_path, authority.provider_role))
    before_files = _custody_file_bytes(tmp_path)
    before_secrets = dict(native_keyring)
    write = custody._write_new_record if writer == "new" else custody._replace_record
    with pytest.raises(LocalSigningCustodyError, match="exact platform Path"):
        write(path, custody._encode_record(_successor(authority)))
    assert calls == []
    assert _custody_file_bytes(tmp_path) == before_files
    assert native_keyring == before_secrets


@pytest.mark.parametrize("writer", ["new", "replace"])
@pytest.mark.parametrize("payload_kind", ["str", "bytes-subclass"])
def test_low_level_writer_requires_exact_bytes_before_payload_interpretation(
    tmp_path, native_keyring, authority, writer, payload_kind
) -> None:
    calls = []

    class CallerBytes(bytes):
        def decode(self, *args, **kwargs):
            calls.append("decode")
            return "{}"

    raw = custody._encode_record(_successor(authority))
    payload = raw.decode("utf-8") if payload_kind == "str" else CallerBytes(raw)
    before_files = _custody_file_bytes(tmp_path)
    before_secrets = dict(native_keyring)
    write = custody._write_new_record if writer == "new" else custody._replace_record
    with pytest.raises(LocalSigningCustodyError, match="exact custody record bytes"):
        write(custody._record_path(tmp_path, authority.provider_role), payload)
    assert calls == []
    assert _custody_file_bytes(tmp_path) == before_files
    assert native_keyring == before_secrets


@pytest.mark.parametrize("exclusive", [False, True])
def test_custody_lock_rejects_caller_path_before_scope_or_os_lock_access(tmp_path, exclusive):
    calls = []
    target = tmp_path / "target-custody"
    other = tmp_path / "other-custody"
    target.mkdir(mode=0o700)
    other.mkdir(mode=0o700)

    class CallerPath(type(Path())):
        def __eq__(self, candidate):
            calls.append("eq")
            return True

        def __truediv__(self, filename):
            calls.append("divide")
            return other / filename

        def __fspath__(self):
            calls.append("fspath")
            return str(other)

    with pytest.raises(LocalSigningCustodyError, match="exact platform Path"):
        with custody._custody_lock(CallerPath(other), exclusive=exclusive):
            pytest.fail("caller-selected path yielded a live custody lease")
    assert calls == []
    assert not (target / custody._LOCK_FILENAME).exists()
    assert not (other / custody._LOCK_FILENAME).exists()


@pytest.mark.parametrize("caller_truth", [False, True])
def test_custody_lock_rejects_spoofed_exclusive_mode_before_caller_methods(tmp_path, caller_truth):
    calls = []

    class CallerMode:
        def __bool__(self):
            calls.append("bool")
            return caller_truth

        def __eq__(self, candidate):
            calls.append("eq")
            return True

    with pytest.raises(LocalSigningCustodyError):
        with custody._custody_lock(tmp_path, exclusive=CallerMode()):
            pytest.fail("caller-selected mode yielded a live custody lease")
    assert calls == []
    assert not (tmp_path / custody._LOCK_FILENAME).exists()


def _initial_writer(tmp_path, security, authority, lease):
    return custody._write_new_record(
        custody._record_path(tmp_path, authority.provider_role),
        custody._encode_record(authority),
        provisioning_administrator=_admin(tmp_path, security, authority.provider_role),
        custody_lease=lease,
    )


def test_initial_writer_lease_expires_when_exclusive_native_lock_is_released(
    tmp_path, native_keyring, security, authority
) -> None:
    before_files = _custody_file_bytes(tmp_path)
    before_secrets = dict(native_keyring)
    with custody._custody_lock(tmp_path, exclusive=True) as lease:
        # Valid initial provisioning remains idempotent with genuine ownership.
        assert _initial_writer(tmp_path, security, authority, lease) is False
    with pytest.raises(LocalSigningLifecycleError, match="exclusive custody provisioning lock"):
        _initial_writer(tmp_path, security, authority, lease)
    assert _custody_file_bytes(tmp_path) == before_files
    assert native_keyring == before_secrets


@pytest.mark.parametrize("lease_kind", ["missing", "lookalike", "unregistered"])
def test_initial_writer_rejects_caller_created_lease_even_while_native_lock_is_held(
    tmp_path, native_keyring, security, authority, lease_kind
) -> None:
    lease = {
        "missing": None,
        "lookalike": object(),
        "unregistered": object.__new__(custody._CustodyLockLease),
    }[lease_kind]
    before_files = _custody_file_bytes(tmp_path)
    before_secrets = dict(native_keyring)
    with custody._custody_lock(tmp_path, exclusive=True):
        with pytest.raises(LocalSigningLifecycleError, match="exclusive custody provisioning lock"):
            _initial_writer(tmp_path, security, authority, lease)
    assert _custody_file_bytes(tmp_path) == before_files
    assert native_keyring == before_secrets


def test_initial_writer_rejects_shared_native_lock_lease(
    tmp_path, native_keyring, security, authority
) -> None:
    before_files = _custody_file_bytes(tmp_path)
    before_secrets = dict(native_keyring)
    with custody._custody_lock(tmp_path, exclusive=False) as lease:
        with pytest.raises(LocalSigningLifecycleError, match="exclusive custody provisioning lock"):
            _initial_writer(tmp_path, security, authority, lease)
    assert _custody_file_bytes(tmp_path) == before_files
    assert native_keyring == before_secrets


def test_initial_writer_rejects_exclusive_native_lease_for_different_custody_directory(
    tmp_path, native_keyring, security, authority
) -> None:
    other = tmp_path / "other-custody"
    other.mkdir(mode=0o700)
    before_secrets = dict(native_keyring)
    before_files = _custody_file_bytes(tmp_path)
    with custody._custody_lock(other, exclusive=True) as lease:
        with pytest.raises(LocalSigningLifecycleError, match="exclusive custody provisioning lock"):
            _initial_writer(tmp_path, security, authority, lease)
    # On Windows the OS correctly forbids reading a byte-range-locked file
    # while the foreign lease is held. Snapshot only after its release, and
    # account explicitly for the lock sentinel that acquiring it creates.
    after_files = _custody_file_bytes(tmp_path)
    foreign_lock = Path("other-custody") / custody._LOCK_FILENAME
    assert foreign_lock not in before_files
    assert foreign_lock in after_files
    assert (
        {path: data for path, data in after_files.items() if path != foreign_lock} == before_files
    )
    assert native_keyring == before_secrets


def test_initial_writer_rejects_live_native_lease_owned_by_another_thread(
    tmp_path, native_keyring, security, authority
) -> None:
    before_files = _custody_file_bytes(tmp_path)
    before_secrets = dict(native_keyring)
    with custody._custody_lock(tmp_path, exclusive=True) as lease:
        assert _initial_writer(tmp_path, security, authority, lease) is False
        with ThreadPoolExecutor(max_workers=1) as executor:
            attempt = executor.submit(_initial_writer, tmp_path, security, authority, lease)
            with pytest.raises(
                LocalSigningLifecycleError, match="exclusive custody provisioning lock"
            ):
                attempt.result(timeout=5)
    assert _custody_file_bytes(tmp_path) == before_files
    assert native_keyring == before_secrets


def _forked_initial_writer(tmp_path, security, authority, lease, sender):
    try:
        _initial_writer(tmp_path, security, authority, lease)
    except LocalSigningLifecycleError:
        sender.send("DENIED")
    else:
        sender.send("ACCEPTED")
    finally:
        sender.close()


@pytest.mark.skipif(not hasattr(os, "fork"), reason="fork ownership check requires POSIX fork")
def test_initial_writer_rejects_live_native_lease_inherited_by_forked_process(
    tmp_path, native_keyring, security, authority
) -> None:
    context = multiprocessing.get_context("fork")
    receiver, sender = context.Pipe(duplex=False)
    before_files = _custody_file_bytes(tmp_path)
    before_secrets = dict(native_keyring)
    with custody._custody_lock(tmp_path, exclusive=True) as lease:
        assert _initial_writer(tmp_path, security, authority, lease) is False
        process = context.Process(
            target=_forked_initial_writer, args=(tmp_path, security, authority, lease, sender)
        )
        try:
            process.start()
            sender.close()
            process.join(5)
            assert not process.is_alive()
            assert process.exitcode == 0
            assert receiver.poll(1)
            assert receiver.recv() == "DENIED"
        finally:
            if process.is_alive():
                process.terminate()
                process.join(5)
            receiver.close()
            sender.close()
    assert _custody_file_bytes(tmp_path) == before_files
    assert native_keyring == before_secrets


@pytest.mark.parametrize(
    "helper_name",
    [
        "_retained_rotation_state",
        "_publish_or_preserve_lifecycle_state",
        "_recover_rotation_locked",
        "_completed_rotation_replay",
        "_cleanup_rotation_artifacts",
        "rotate_local_freshness_signing_authority",
    ],
)
def test_freshness_rotation_helpers_cannot_mutate_root_or_history_custody(
    tmp_path, native_keyring, security, authority, helper_name
) -> None:
    role = authority.provider_role
    # Persisted artifact fixtures must remain byte-identical on a denied call.
    for path in (
        *custody._rotation_paths(tmp_path, role),
        custody._completed_rotation_path(tmp_path, security, role, "op-1"),
    ):
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        path.write_bytes(b"retained legacy artifact fixture")
        path.chmod(0o600)
    before_files = _custody_file_bytes(tmp_path)
    before_tree = {path.relative_to(tmp_path) for path in tmp_path.rglob("*")}
    before_secrets = dict(native_keyring)
    helper = getattr(custody, helper_name)
    with pytest.raises(LocalSigningLifecycleError):
        if helper_name in {"_retained_rotation_state", "_publish_or_preserve_lifecycle_state"}:
            helper(tmp_path, security, role, _successor(authority))
        elif helper_name == "_recover_rotation_locked":
            helper(tmp_path, _admin(tmp_path, security, role), security, role, "op-1", None, None)
        elif helper_name == "_completed_rotation_replay":
            helper(tmp_path, security, role, "op-1", None, None)
        elif helper_name == "_cleanup_rotation_artifacts":
            helper(tmp_path, role, None)
        else:
            helper(
                tmp_path,
                _admin(tmp_path, security, role),
                security=security,
                role=role,
                rotation_operation_id="op-1",
            )
    assert _custody_file_bytes(tmp_path) == before_files
    assert {path.relative_to(tmp_path) for path in tmp_path.rglob("*")} == before_tree
    assert native_keyring == before_secrets

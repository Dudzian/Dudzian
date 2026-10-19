"""S9-RPI-20: cooperative credential-lifecycle lease over real TEST source + CCA.

Use the already existing CCA rotation worker and retained log. This module is
test-only, same-UID and non-authorizing: advisory locks and fixture-selected
credentials do not qualify historical eligibility or production CCA.
"""

from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, field
import errno
import multiprocessing
import os
from pathlib import Path
import stat
import sys
import time

import pytest

from bot_core import independent_cca_decision as cca
from bot_core import test_only_cca_decision_log as log_module
from bot_core.licensing.canonical import canonical_json_bytes
from tests.security._cca_decision_fixtures import sign_cca_payload
from tests.security._cca_log_process import start_rotation
from tests.security.test_retained_source_real_linux_disk_bridge import (
    _LinuxFixtureOriginalReader,
    _inspect,
)
from tests.security.test_retained_source_test_floor_witness import _FloorGuardedReader
from tests.security.test_source_cca_cryptographic_chain_linux import (
    _cca_policy_for_retained_decision,
    _prepare_decision,
    verified_chain as verified_chain,
)
from tests.security.test_source_custody_shared_lease_linux import (
    hold_synthetic_source_read_lease,
)

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(sys.platform != "linux", reason="requires Linux flock + dirfd"),
]

_CREDENTIAL_WAIT = 1.0
_CREDENTIAL_POLL = 0.01
_ACTIVE = b"ACTIVE:1:1"


class TestOnlyCredentialLeaseError(RuntimeError):
    """Credential fixture cannot support a new decision."""


@dataclass(frozen=True, slots=True)
class SyntheticCredentialReadLease:
    status: str = field(default="NOT_QUALIFIED", init=False)
    production_authorization: str = field(default="BLOCKED", init=False)
    historical_eligibility_qualified: bool = field(default=False, init=False)


def _ident(info):
    return (
        info.st_dev,
        info.st_ino,
        info.st_mode,
        info.st_uid,
        info.st_gid,
        info.st_nlink,
        info.st_size,
        info.st_mtime_ns,
        info.st_ctime_ns,
    )


def _state_ident(info):
    # Ignore timestamps/size so a revoked state is diagnosed by its bytes.
    return (
        info.st_dev,
        info.st_ino,
        info.st_mode,
        info.st_uid,
        info.st_gid,
        info.st_nlink,
    )


def _require(ok, message):
    if not ok:
        raise TestOnlyCredentialLeaseError(message)


def _check_entry(dir_fd, name, fd, pinned, mode, *, state=False):
    actual = os.fstat(fd)
    path = os.stat(name, dir_fd=dir_fd, follow_symlinks=False)
    identity = _state_ident if state else _ident
    _require(
        identity(actual) == pinned and identity(path) == pinned,
        "TEST CCA credential lease inode or metadata changed",
    )
    _require(
        stat.S_ISREG(actual.st_mode)
        and stat.S_IMODE(actual.st_mode) == mode
        and actual.st_nlink == 1
        and actual.st_uid == os.geteuid()
        and actual.st_gid == os.getegid(),
        "TEST CCA credential lease owner/type/mode invalid",
    )


def _check_state(dir_fd, lease_fd, state_fd, pin_lease, pin_state):
    _check_entry(dir_fd, "credential-lease", lease_fd, pin_lease, 0o600)
    _check_entry(dir_fd, "credential-state", state_fd, pin_state, 0o600, state=True)
    os.lseek(state_fd, 0, os.SEEK_SET)
    _require(os.read(state_fd, 64) == _ACTIVE, "TEST CCA credential rotated or revoked")
    _check_entry(dir_fd, "credential-state", state_fd, pin_state, 0o600, state=True)


def _check_credential_dir(root_fd, directory, expected):
    observed = os.fstat(root_fd)
    current = Path(directory).lstat()
    _require(
        stat.S_ISDIR(observed.st_mode)
        and stat.S_ISDIR(current.st_mode)
        and stat.S_IMODE(observed.st_mode) == 0o700
        and stat.S_IMODE(current.st_mode) == 0o700
        and observed.st_uid == current.st_uid == os.geteuid()
        and observed.st_gid == current.st_gid == os.getegid()
        and (observed.st_dev, observed.st_ino) == (current.st_dev, current.st_ino) == expected,
        "TEST CCA credential lease directory invalid or replaced",
    )


@contextmanager
def hold_synthetic_cca_credential_read_lease(directory, fixture_pin):
    """A bounded, actual SH flock; exact active generation checked before/after.

    The nested ordering is source LOCK_SH -> credential LOCK_SH -> CCA log EX.
    Rotation cooperating on credential-lease must wait for complete CCA decision.
    """
    if sys.platform != "linux":
        raise TestOnlyCredentialLeaseError("Linux-only credential fixture")
    import fcntl

    root = lease_fd = state_fd = None
    acquired = False
    try:
        root = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
        _check_credential_dir(root, directory, fixture_pin["directory"])
        lease_fd = os.open(
            "credential-lease",
            os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC,
            dir_fd=root,
        )
        state_fd = os.open(
            "credential-state",
            os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC,
            dir_fd=root,
        )
        pin_lease, pin_state = fixture_pin["lock"], fixture_pin["state"]
        _check_entry(root, "credential-lease", lease_fd, pin_lease, 0o600)
        _check_entry(root, "credential-state", state_fd, pin_state, 0o600, state=True)
        deadline = time.monotonic() + _CREDENTIAL_WAIT
        while time.monotonic() < deadline:
            try:
                fcntl.flock(lease_fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
            except BlockingIOError:
                remaining = deadline - time.monotonic()
                if remaining > 0:
                    time.sleep(min(_CREDENTIAL_POLL, remaining))
            else:
                acquired = True
                break
        _require(acquired, "TEST CCA credential shared lease acquisition timed out")
        _check_credential_dir(root, directory, fixture_pin["directory"])
        _check_state(root, lease_fd, state_fd, pin_lease, pin_state)
        try:
            yield SyntheticCredentialReadLease()
        finally:
            _check_state(root, lease_fd, state_fd, pin_lease, pin_state)
            _check_credential_dir(root, directory, fixture_pin["directory"])
    except OSError as exc:
        raise TestOnlyCredentialLeaseError(
            "TEST CCA credential lease OS boundary unavailable"
        ) from exc
    finally:
        if lease_fd is not None:
            try:
                if acquired:
                    fcntl.flock(lease_fd, fcntl.LOCK_UN)
            finally:
                os.close(lease_fd)
        if state_fd is not None:
            os.close(state_fd)
        if root is not None:
            os.close(root)


@pytest.fixture
def credential_case(verified_chain, tmp_path):
    config, original, raw_policy, floor_pin, key = verified_chain
    case, base_policy = _prepare_decision(original, raw_policy, key)
    directory = tmp_path / "isolated-test-credential-lease"
    directory.mkdir(mode=0o700)
    for name, data in (("credential-lease", b""), ("credential-state", _ACTIVE)):
        path = directory / name
        path.write_bytes(data)
        path.chmod(0o600)
    log_dir, log_floor = tmp_path / "isolated-cca-log", tmp_path / "isolated-cca-floor"
    log_dir.mkdir(mode=0o700)
    log_floor.mkdir(mode=0o700)
    pin = log_module.provision_test_only_cca_log(
        log_dir,
        log_floor,
        installation_id=case.payload["installation"]["installation_id"],
        namespace=case.payload["completion_authority"]["provider_namespace"],
    )
    immutable = canonical_json_bytes(
        {
            name: value
            for name, value in case.payload.items()
            if name not in {"decision_id", "decision_log_position"}
        }
    )
    cca.validate_cca_immutable_binding(immutable)
    fixture_pin = {
        "directory": (directory.stat().st_dev, directory.stat().st_ino),
        "lock": _ident((directory / "credential-lease").lstat()),
        "state": _state_ident((directory / "credential-state").lstat()),
    }
    return config, original, floor_pin, case, base_policy, directory, pin, immutable, fixture_pin


def _decide_and_verify(case, base_policy, pin, immutable, *, fault=None):
    def construct(position):
        payload = deepcopy(case.payload)
        payload["decision_id"] = position.entry_id
        payload["decision_log_position"] = position.document
        return sign_cca_payload(case, payload)

    log = log_module.TestOnlyCCADecisionLog(pin)
    raw = log.decide(
        case.payload["close"]["operation_id"],
        immutable,
        create_decision=construct,
        fault_injector=fault,
    )
    decision = cca.parse_independent_cca_decision(raw)
    policy = _cca_policy_for_retained_decision(base_policy, decision)
    verdict = cca.verify_independent_cca_decision(decision, policy_loader=lambda: policy)
    assert verdict.status == "NOT_QUALIFIED"
    assert verdict.production_emission == "BLOCKED"
    assert verdict.signature == "MATCHED_EXPLICIT_POLICY"
    assert verdict.source_signature == "MATCHED_EXPLICIT_POLICY"
    assert verdict.blockers
    return raw


def test_credential_lease_covers_source_verified_cca_retention(credential_case):
    (config, original, floor_pin, case, policy, directory, pin, immutable, fixture_pin) = (
        credential_case
    )
    reader = _LinuxFixtureOriginalReader(config, original)
    guarded = _FloorGuardedReader(reader, floor_pin)
    with hold_synthetic_source_read_lease(config):
        with hold_synthetic_cca_credential_read_lease(directory, fixture_pin) as lease:
            assert lease.status == "NOT_QUALIFIED"
            assert lease.historical_eligibility_qualified is False
            with _inspect(guarded, original) as evidence:
                assert evidence.status == "NOT_QUALIFIED"
                raw = _decide_and_verify(case, policy, pin, immutable)
                assert reader.lookups == 2
            assert reader.lookups == 3
    assert (
        log_module.TestOnlyCCADecisionLog(pin).exact_lookup(
            case.payload["close"]["operation_id"], immutable
        )
        == raw
    )
    assert len(list(Path(pin.log_directory).glob("*.decision"))) == 1


@pytest.mark.parametrize("state", [b"VERIFY_ONLY:2:1", b"REVOKED:2:1"])
def test_credential_rotation_waits_for_durable_cca_ack_and_denies_new_lease(credential_case, state):
    (config, original, floor_pin, case, policy, directory, pin, immutable, fixture_pin) = (
        credential_case
    )
    reader = _LinuxFixtureOriginalReader(config, original)
    guarded = _FloorGuardedReader(reader, floor_pin)
    rotation = connection = None
    try:
        with hold_synthetic_source_read_lease(config):
            with hold_synthetic_cca_credential_read_lease(directory, fixture_pin):
                with _inspect(guarded, original):

                    def fault(phase):
                        nonlocal rotation, connection
                        if phase == "after_floor_reservation":
                            rotation, connection = start_rotation(str(directory), state)
                            assert connection.poll(15)
                            assert connection.recv() == (
                                "contended",
                                "credential_lifecycle",
                                (
                                    (directory / "credential-lease").stat().st_dev,
                                    (directory / "credential-lease").stat().st_ino,
                                ),
                            )
                        if phase == "after_durable_decision":
                            assert rotation is not None
                            assert not connection.poll(0.1)

                    raw = _decide_and_verify(case, policy, pin, immutable, fault=fault)
                    assert not connection.poll(0.1)
                assert reader.lookups == 3
                assert not connection.poll(0.1)
            assert connection.poll(15)
            assert connection.recv() == "rotated"
        rotation.join(10)
        assert rotation.exitcode == 0
        with pytest.raises(TestOnlyCredentialLeaseError, match="rotated or revoked"):
            with hold_synthetic_cca_credential_read_lease(directory, fixture_pin):
                pytest.fail("rotated credential cannot grant a new synthetic decision")
        assert (
            log_module.TestOnlyCCADecisionLog(pin).exact_lookup(
                case.payload["close"]["operation_id"], immutable
            )
            == raw
        )
        assert len(list(Path(pin.log_directory).glob("*.decision"))) == 1
    finally:
        if connection is not None:
            connection.close()
        if rotation is not None:
            if rotation.is_alive():
                rotation.kill()
            rotation.join(5)


def test_previously_revoked_credential_refuses_before_cca_intent(credential_case):
    (config, original, floor_pin, case, policy, directory, pin, immutable, fixture_pin) = (
        credential_case
    )
    (directory / "credential-state").write_bytes(b"REVOKED:2:1")
    reader = _LinuxFixtureOriginalReader(config, original)
    guarded = _FloorGuardedReader(reader, floor_pin)
    with hold_synthetic_source_read_lease(config):
        with pytest.raises(TestOnlyCredentialLeaseError, match="rotated or revoked"):
            with hold_synthetic_cca_credential_read_lease(directory, fixture_pin):
                with _inspect(guarded, original):
                    _decide_and_verify(case, policy, pin, immutable)
    assert not list(Path(pin.log_directory).glob("*.intent"))
    assert not list(Path(pin.log_directory).glob("*.decision"))


def _hold_exclusive_credential_lease(connection, directory):
    """Open the existing inode independently and retain EX until explicit release."""
    import fcntl

    try:
        descriptor = os.open(
            Path(directory) / "credential-lease", os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC
        )
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            info = os.fstat(descriptor)
            connection.send(("acquired-exclusive", info.st_dev, info.st_ino))
            if not connection.poll(15) or connection.recv() != "release-exclusive":
                raise RuntimeError("TEST credential exclusive holder release signal missing")
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)
        connection.send("released-exclusive")
    finally:
        connection.close()


def test_exclusive_credential_holder_bounds_shared_wait_and_allows_retry(
    credential_case, monkeypatch
):
    config, _original, _floor, _case, _policy, directory, pin, _binding, fixture_pin = (
        credential_case
    )
    context = multiprocessing.get_context("spawn")
    parent, child = context.Pipe()
    holder = context.Process(target=_hold_exclusive_credential_lease, args=(child, directory))
    opened = []
    real_open = os.open

    def record_open(path, *args, **kwargs):
        descriptor = real_open(path, *args, **kwargs)
        opened.append((path, descriptor))
        return descriptor

    try:
        holder.start()
        child.close()
        assert parent.poll(15), "exclusive credential holder never acquired lock"
        assert parent.recv() == ("acquired-exclusive", *fixture_pin["lock"][:2])
        with hold_synthetic_source_read_lease(config):
            lease = None
            started = time.monotonic()
            with monkeypatch.context() as patch:
                patch.setattr(os, "open", record_open)
                with pytest.raises(TestOnlyCredentialLeaseError) as refused:
                    with hold_synthetic_cca_credential_read_lease(directory, fixture_pin) as lease:
                        pytest.fail("exclusive credential holder must prevent a shared lease")
            elapsed = time.monotonic() - started
            assert str(refused.value) == "TEST CCA credential shared lease acquisition timed out"
            assert refused.value.__cause__ is None
            assert _CREDENTIAL_WAIT <= elapsed < 5
            assert lease is None
            assert [path for path, _fd in opened] == [
                directory,
                "credential-lease",
                "credential-state",
            ]
            for _path, descriptor in opened:
                with pytest.raises(OSError) as closed:
                    os.fstat(descriptor)
                assert closed.value.errno == errno.EBADF
            assert not list(Path(pin.log_directory).glob("*.intent"))
            assert not list(Path(pin.log_directory).glob("*.decision"))
            assert holder.is_alive(), "exclusive credential holder exited before release"
            parent.send("release-exclusive")
            assert parent.poll(15), "exclusive credential holder did not acknowledge release"
            assert parent.recv() == "released-exclusive"
            holder.join(10)
            assert holder.exitcode == 0
            with hold_synthetic_cca_credential_read_lease(directory, fixture_pin) as lease:
                assert isinstance(lease, SyntheticCredentialReadLease)
                assert lease.status == "NOT_QUALIFIED"
                assert lease.production_authorization == "BLOCKED"
                assert lease.historical_eligibility_qualified is False
        assert not list(Path(pin.log_directory).glob("*.intent"))
        assert not list(Path(pin.log_directory).glob("*.decision"))
    finally:
        parent.close()
        child.close()
        if holder.pid is not None:
            if holder.is_alive():
                holder.kill()
            holder.join(5)
        holder.close()


def test_uncooperative_state_mutation_under_lease_refuses_exit(credential_case):
    (config, original, floor_pin, case, policy, directory, pin, immutable, fixture_pin) = (
        credential_case
    )
    reader = _LinuxFixtureOriginalReader(config, original)
    guarded = _FloorGuardedReader(reader, floor_pin)
    with hold_synthetic_source_read_lease(config):
        with pytest.raises(TestOnlyCredentialLeaseError, match="rotated or revoked"):
            with hold_synthetic_cca_credential_read_lease(directory, fixture_pin):
                with _inspect(guarded, original):
                    _decide_and_verify(case, policy, pin, immutable)
                (directory / "credential-state").write_bytes(b"REVOKED:2:1")
    # A durable synthetic original may exist, but the lease must refuse
    # completion of its post-retention validation. No production ACK exists.
    assert len(list(Path(pin.log_directory).glob("*.decision"))) == 1


@pytest.mark.parametrize(
    ("attack", "message", "cause_type", "cause_errno"),
    [
        (
            "missing",
            "TEST CCA credential lease OS boundary unavailable",
            FileNotFoundError,
            errno.ENOENT,
        ),
        (
            "symlink",
            "TEST CCA credential lease OS boundary unavailable",
            OSError,
            errno.ELOOP,
        ),
        ("permissions", "TEST CCA credential lease inode or metadata changed", None, None),
        ("inode", "TEST CCA credential lease inode or metadata changed", None, None),
    ],
    ids=["missing", "symlink", "permissions", "inode"],
)
def test_unsafe_credential_lease_inode_refuses_before_cca_intent(
    credential_case, attack, message, cause_type, cause_errno
):
    (config, _original, _floor, _case, _policy, directory, pin, _binding, fixture_pin) = (
        credential_case
    )
    path = directory / "credential-lease"
    if attack == "missing":
        path.unlink()
    elif attack == "symlink":
        path.rename(directory / "saved-lease")
        path.symlink_to("saved-lease")
    elif attack == "permissions":
        path.chmod(0o644)
    else:
        path.rename(directory / "saved-lease")
        path.write_bytes(b"")
        path.chmod(0o600)
    with hold_synthetic_source_read_lease(config):
        with pytest.raises(TestOnlyCredentialLeaseError) as refused:
            with hold_synthetic_cca_credential_read_lease(directory, fixture_pin):
                pytest.fail("unsafe fixture lock cannot become a credential lease")
    assert str(refused.value) == message
    if cause_type is None:
        assert refused.value.__cause__ is None
    else:
        assert type(refused.value.__cause__) is cause_type
        assert refused.value.__cause__.errno == cause_errno
    assert not list(Path(pin.log_directory).glob("*.intent"))
    assert not list(Path(pin.log_directory).glob("*.decision"))


def test_credential_directory_replaced_during_held_lease_refuses(credential_case):
    config, _original, _floor, _case, _policy, directory, pin, _binding, fixture_pin = (
        credential_case
    )
    with hold_synthetic_source_read_lease(config):
        with pytest.raises(TestOnlyCredentialLeaseError, match="directory invalid or replaced"):
            with hold_synthetic_cca_credential_read_lease(directory, fixture_pin):
                directory.rename(directory.with_name("retained-credential-directory"))
                directory.mkdir(mode=0o700)
    assert not list(Path(pin.log_directory).glob("*.intent"))
    assert not list(Path(pin.log_directory).glob("*.decision"))

"""Real same-UID Linux processes for the separate TEST-only CCA log fixture.

No production module imports this helper. Its installed synthetic payload and
temporary test key are configuration, never a caller-facing signing endpoint.
The source lease and credential lifecycle locks are actual flock descriptors;
neither the synthetic close nor same-UID DAC is production-qualified.
"""

from __future__ import annotations

from copy import deepcopy
import errno
import multiprocessing
import os
from pathlib import Path

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from bot_core.test_only_cca_decision_log import TestOnlyCCADecisionLog
from tests.security._cca_decision_fixtures import sign_cca_payload


def _acquire_witnessed_contention(flock, descriptor, connection, label):
    """Witness real kernel contention before performing the blocking acquire."""
    import fcntl

    info = os.fstat(descriptor)
    try:
        flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        if exc.errno not in {errno.EAGAIN, errno.EWOULDBLOCK}:
            raise
        connection.send(("contended", label, (info.st_dev, info.st_ino)))
    else:
        flock(descriptor, fcntl.LOCK_UN)
        raise AssertionError("TEST expected a competing process to hold the actual lock")
    flock(descriptor, fcntl.LOCK_EX)


def _worker(pin, fixture, connection, crash_at, pause_at, forbid_signing, observe_contention):
    import fcntl

    leases = []
    try:
        for name in ("source-lease", "credential-lease"):
            descriptor = os.open(
                Path(fixture["lease_directory"]) / name, os.O_RDONLY | os.O_NOFOLLOW
            )
            leases.append(descriptor)
            fcntl.flock(descriptor, fcntl.LOCK_EX)
        if (Path(fixture["lease_directory"]) / "credential-state").read_bytes() != b"ACTIVE:1:1":
            raise ValueError("synthetic credential rotated/revoked")

        def fault(phase):
            if phase == crash_at:
                os._exit(73)
            if phase == pause_at:
                connection.send(("barrier", phase))
                if not connection.poll(15) or connection.recv() != "resume":
                    raise ValueError("TEST barrier timeout")

        def construct(position):
            if forbid_signing:
                raise AssertionError("recovery attempted to sign another decision")
            from types import SimpleNamespace

            payload = deepcopy(fixture["payload"])
            payload["decision_id"] = position.entry_id
            payload["decision_log_position"] = position.document
            case = SimpleNamespace(
                payload=payload,
                private_key=Ed25519PrivateKey.from_private_bytes(fixture["test_key"]),
            )
            return sign_cca_payload(case)

        actual_flock = fcntl.flock
        lock_info = (Path(pin.log_directory) / "lock").stat(follow_symlinks=False)
        expected_lock = lock_info.st_dev, lock_info.st_ino
        witnessed = []

        def witness_log_flock(descriptor, operation):
            info = os.fstat(descriptor)
            if operation == fcntl.LOCK_EX and (info.st_dev, info.st_ino) == expected_lock:
                _acquire_witnessed_contention(actual_flock, descriptor, connection, "decision_log")
                witnessed.append(True)
            else:
                actual_flock(descriptor, operation)

        # Scoped to this disposable TEST child. The application retains its real
        # lock and has no test hook; only the actual log-lock inode is observed.
        if observe_contention:
            fcntl.flock = witness_log_flock
        try:
            original = TestOnlyCCADecisionLog(pin).decide(
                fixture["operation_id"],
                fixture["binding"],
                create_decision=construct,
                fault_injector=fault,
            )
            if observe_contention and not witnessed:
                raise AssertionError("TEST decision did not acquire the actual log-lock inode")
        finally:
            fcntl.flock = actual_flock
        connection.send(("ok", original))
    except Exception as exc:
        connection.send(("refused", type(exc).__name__, str(exc)))
    finally:
        for descriptor in reversed(leases):
            fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)
        connection.close()


def start_log_process(
    pin, fixture, *, crash_at=None, pause_at=None, forbid_signing=False, observe_contention=False
):
    context = multiprocessing.get_context("spawn")
    parent, child = context.Pipe()
    process = context.Process(
        target=_worker,
        args=(pin, fixture, child, crash_at, pause_at, forbid_signing, observe_contention),
    )
    process.start()
    child.close()
    return process, parent


def _rotate(directory, state, connection):
    import fcntl

    descriptor = os.open(Path(directory) / "credential-lease", os.O_RDONLY | os.O_NOFOLLOW)
    try:
        _acquire_witnessed_contention(fcntl.flock, descriptor, connection, "credential_lifecycle")
        with (Path(directory) / "credential-state").open("wb") as stream:
            stream.write(state)
            stream.flush()
            os.fsync(stream.fileno())
        connection.send("rotated")
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)
        connection.close()


def start_rotation(directory, state):
    context = multiprocessing.get_context("spawn")
    parent, child = context.Pipe()
    process = context.Process(target=_rotate, args=(directory, state, child))
    process.start()
    child.close()
    return process, parent

"""S9-RPI-15: Linux TEST-only disk bridge to the existing source worker.

The source worker and the fixture reader run as the same effective UID. Paths,
pins, policy and expected original are fixture-derived; no production trust,
rollback resistance, lifecycle lease or independent source attestation exists.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import shutil
import stat
import sys

import pytest

from bot_core import authenticated_custody_mutation_source as source
from bot_core.root_proof_issuer_substrate import ProviderRole
from bot_core.test_only_retained_source_original import (
    SyntheticRetainedSourcePin,
    TestOnlyRetainedSourceError,
    inspect_synthetic_retained_source_original,
)
from tests.security._custody_source_demo import start_process

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(sys.platform != "linux", reason="requires actual Linux filesystem"),
]
_REQUEST = {"operation_id": "controlled-operation-1", "target_state": "VERIFY_ONLY"}


def _receive(process, connection):
    try:
        assert connection.poll(15), "isolated source TEST worker did not respond"
        response = connection.recv()
        process.join(15)
        assert not process.is_alive() and process.exitcode == 0
        return response
    finally:
        connection.close()
        if process.is_alive():
            process.kill()
            process.join(5)


@pytest.fixture
def source_on_disk(tmp_path):
    root = tmp_path / "linux-retained-source"
    root.mkdir(mode=0o700)
    status, config = _receive(
        *start_process(root=str(root), role=ProviderRole.ROOT_PROOF_SIGNING.value)
    )
    assert status == "ok"
    status, result = _receive(*start_process(config=config, request=_REQUEST))
    assert status == "ok" and result["status"] == "NOT_QUALIFIED"
    assert result["production_emission"] == "BLOCKED"
    assert (root / "events" / "original").read_bytes() == result["event"]
    return config, result


def _stat_identity(info):
    # Pin inode, directory entries' mutation timestamps and exact file mode.
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


def _require(condition, message):
    if not condition:
        raise ValueError(message)


class _LinuxFixtureOriginalReader:
    """NOFOLLOW/dirfd test adapter: not an installed reader or permission boundary."""

    def __init__(self, config, original):
        self.root = Path(config["root"])
        self.expected_payload = source.parse_custody_mutation_execution(original).payload
        self.lookups = 0
        self.identity = {
            name: _stat_identity((self.root / name).lstat())
            for name in ("events", "events/original", "events/public-policy")
        }
        self.identity["."] = _stat_identity(self.root.lstat())

    def _read(self, name):
        _require(name in {"original", "public-policy"}, "closed TEST file selection required")
        root_fd = os.open(
            self.root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
        )
        try:
            self._verify(root_fd, ".", directory=True)
            events_fd = os.open(
                "events", os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=root_fd
            )
            try:
                self._verify(events_fd, "events", directory=True)
                descriptor = os.open(
                    name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=events_fd
                )
                try:
                    before = os.fstat(descriptor)
                    self._verify(descriptor, "events/" + name)
                    limit = source.MAX_ENVELOPE_BYTES if name == "original" else source.MAX_POLICY_BYTES
                    _require(0 < before.st_size <= limit, "bounded retained TEST bytes required")
                    result = bytearray()
                    while len(result) <= limit:
                        part = os.read(descriptor, min(65536, limit + 1 - len(result)))
                        if not part:
                            break
                        result.extend(part)
                    _require(len(result) == before.st_size, "retained source file size changed")
                    self._verify(descriptor, "events/" + name)
                    _require(
                        _stat_identity(os.stat(name, dir_fd=events_fd, follow_symlinks=False))
                        == self.identity["events/" + name],
                        "retained TEST file path substituted",
                    )
                    self._verify(events_fd, "events", directory=True)
                    self._verify(root_fd, ".", directory=True)
                    return bytes(result)
                finally:
                    os.close(descriptor)
            finally:
                os.close(events_fd)
        finally:
            os.close(root_fd)

    def _verify(self, fd, name, *, directory=False):
        current = os.fstat(fd)
        _require(
            _stat_identity(current) == self.identity[name],
            "retained TEST inode or metadata changed",
        )
        _require(
            current.st_uid == os.geteuid() and current.st_gid == os.getegid(),
            "retained TEST ownership changed",
        )
        _require(
            stat.S_ISDIR(current.st_mode) if directory else stat.S_ISREG(current.st_mode),
            "retained TEST file type changed",
        )
        _require(
            stat.S_IMODE(current.st_mode) == (0o700 if directory else 0o600),
            "retained TEST permissions changed",
        )
        _require(
            current.st_nlink >= 2 if directory else current.st_nlink == 1,
            "retained TEST link count unsafe",
        )
        _require(
            _stat_identity((self.root / name).lstat()) == self.identity[name],
            "retained TEST path replaced",
        )

    def lookup_original_event(self, *, installation_id, operation_id, execution_id):
        expected = self.expected_payload
        _require(
            installation_id == expected["installation"]["installation_id"]
            and operation_id == expected["close"]["operation_id"]
            and execution_id == expected["execution_id"],
            "test disk reader queried wrong original identity",
        )
        self.lookups += 1
        return self._read("original")

    def load_policy(self):
        return source.parse_source_verification_policy(self._read("public-policy"))


def _synthetic_pin(original):
    payload = source.parse_custody_mutation_execution(original).payload
    position = payload["source_log_position"]
    return SyntheticRetainedSourcePin(
        installation_id=payload["installation"]["installation_id"],
        operation_id=payload["close"]["operation_id"],
        execution_id=payload["execution_id"],
        namespace=position["namespace"],
        sequence=position["sequence"],
        previous_entry_sha256=position["previous_entry_sha256"],
        retained_envelope_sha256=hashlib.sha256(original).hexdigest(),
    )


def _inspect(reader, original):
    return inspect_synthetic_retained_source_original(
        pin=_synthetic_pin(original),
        claimed_original=original,
        reader=reader,
        policy_loader=reader.load_policy,
    )


def test_real_linux_worker_original_is_retrieved_thrice_but_not_qualified(source_on_disk):
    config, result = source_on_disk
    original = result["event"]
    reader = _LinuxFixtureOriginalReader(config, original)
    with _inspect(reader, original) as evidence:
        assert reader.lookups == 2
        assert evidence.original_envelope == original
        assert evidence.status == "NOT_QUALIFIED"
        assert evidence.provenance_qualified is False
        assert evidence.durability_attested is False
        assert evidence.production_authorization == "BLOCKED"
    assert reader.lookups == 3


def test_real_linux_worker_restart_preserves_retained_original(source_on_disk):
    config, result = source_on_disk
    old = result["event"]
    reader = _LinuxFixtureOriginalReader(config, old)
    status, restarted = _receive(*start_process(config=config, request=_REQUEST))
    assert status == "ok" and restarted["recovered"] is True
    assert restarted["event"] == old
    with _inspect(reader, old) as evidence:
        assert evidence.original_envelope == old
        assert evidence.durability_attested is False


@pytest.mark.parametrize(
    "attack",
    ["chmod", "same-bytes-replace", "symlink", "events-directory", "truncate"],
)
def test_disk_original_replacement_or_permissions_deny_before_yield(source_on_disk, attack):
    config, result = source_on_disk
    original = result["event"]
    reader = _LinuxFixtureOriginalReader(config, original)
    file = Path(config["root"]) / "events" / "original"
    if attack == "chmod":
        file.chmod(0o644)
    elif attack == "same-bytes-replace":
        replacement = file.with_name("replacement")
        replacement.write_bytes(original)
        replacement.chmod(0o600)
        replacement.replace(file)
    elif attack == "symlink":
        file.rename(file.with_name("saved"))
        file.symlink_to("saved")
    elif attack == "events-directory":
        folder = file.parent
        folder.rename(folder.with_name("events-old"))
        folder.mkdir(mode=0o700)
    else:
        file.write_bytes(original[:-1])
    with pytest.raises(TestOnlyRetainedSourceError, match="lookup denied"):
        with _inspect(reader, original):
            pytest.fail("changed source file must not return original evidence")


def test_disk_original_changed_during_consumer_denies_successful_exit(source_on_disk):
    config, result = source_on_disk
    original = result["event"]
    reader = _LinuxFixtureOriginalReader(config, original)
    with pytest.raises(TestOnlyRetainedSourceError, match="changed during consumer"):
        with _inspect(reader, original) as evidence:
            assert evidence.status == "NOT_QUALIFIED"
            (Path(config["root"]) / "events" / "original").write_bytes(original[:-1])
    assert reader.lookups == 3


def test_disk_source_policy_inode_replaced_during_consumer_denies(source_on_disk):
    config, result = source_on_disk
    original = result["event"]
    reader = _LinuxFixtureOriginalReader(config, original)
    policy = Path(config["root"]) / "events" / "public-policy"
    with pytest.raises(TestOnlyRetainedSourceError, match="changed during consumer"):
        with _inspect(reader, original):
            replacement = policy.with_name("other-policy")
            shutil.copyfile(policy, replacement)
            replacement.replace(policy)

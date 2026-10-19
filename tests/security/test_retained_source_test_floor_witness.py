"""S9-RPI-16: one-shot same-host TEST source floor vs real Linux retained originals.

This is a test fixture witness, NOT a trusted antirollback authority.
The floor, worker and reader share a UID and rollback domain. Independent
provisioning, a sealed anchor, durable source custody and power-loss proof
remain completely unqualified.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import stat
import sys

import pytest

from bot_core import authenticated_custody_mutation_source as source
from bot_core.licensing.canonical import canonical_json_bytes
from bot_core.test_only_cca_decision_log import _Directory
from bot_core.test_only_retained_source_original import TestOnlyRetainedSourceError
from tests.security._custody_source_demo import start_process
from tests.security.test_retained_source_real_linux_disk_bridge import (
    _LinuxFixtureOriginalReader,
    _REQUEST,
    _inspect,
    _receive,
    source_on_disk as source_on_disk,
)

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(sys.platform != "linux", reason="requires Linux fsync and dirfds"),
]
_FILE = "source-original-floor"
_SCHEMA = "cryptohunter.TEST-ONLY-source-original-floor-v1"


class TestOnlySourceFloorError(RuntimeError):
    """No durable, exact test witness; no evidence is returned."""


@dataclass(frozen=True, slots=True)
class _SourceFloorPin:
    directory: str
    directory_identity: tuple[int, int]
    file_identity: tuple[int, int]
    document_sha256: str


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise TestOnlySourceFloorError(reason)


def _document(original: bytes, policy_bytes: bytes) -> dict[str, object]:
    event = source.parse_custody_mutation_execution(original)
    payload = event.payload
    return {
        "schema": _SCHEMA,
        "installation_id": payload["installation"]["installation_id"],
        "operation_id": payload["close"]["operation_id"],
        "execution_id": payload["execution_id"],
        "source_log_position": payload["source_log_position"],
        "original_sha256": _sha(original),
        "source_policy_sha256": _sha(policy_bytes),
    }


def _once_provision_fixture_floor(
    directory: Path, original: bytes, policy_bytes: bytes
) -> _SourceFloorPin:
    """TEST setup ONLY: write once and fsync; never used as recovery/re-pin."""
    value = canonical_json_bytes(_document(original, policy_bytes))
    folder = _Directory(directory)
    try:
        _require(not os.listdir(folder.fd), "source floor requires existing empty directory")
        folder.stage(_FILE, value)
        folder.sync(_FILE)
        info = os.stat(_FILE, dir_fd=folder.fd, follow_symlinks=False)
        return _SourceFloorPin(
            str(directory),
            (os.fstat(folder.fd).st_dev, os.fstat(folder.fd).st_ino),
            (info.st_dev, info.st_ino),
            _sha(value),
        )
    finally:
        folder.close()


def _read_fixture_floor(pin: _SourceFloorPin) -> dict[str, object]:
    """Compare a separately retained fixture pin, without trusting its origin."""
    if type(pin) is not _SourceFloorPin:
        raise TestOnlySourceFloorError("exact TEST floor pin required")
    folder = None
    try:
        folder = _Directory(Path(pin.directory), pin.directory_identity)
        _require(set(os.listdir(folder.fd)) == {_FILE}, "source floor missing or interrupted")
        before = os.stat(_FILE, dir_fd=folder.fd, follow_symlinks=False)
        _require(
            (before.st_dev, before.st_ino) == pin.file_identity,
            "source floor file inode replaced",
        )
        raw = folder.read(_FILE, limit=2048)
        after = os.stat(_FILE, dir_fd=folder.fd, follow_symlinks=False)
        _require(
            (after.st_dev, after.st_ino) == pin.file_identity,
            "source floor file inode changed",
        )
        _require(_sha(raw) == pin.document_sha256, "source floor bytes differ from pin")
        record = json.loads(raw)
        _require(type(record) is dict, "source floor record must be exact object")
        _require(
            canonical_json_bytes(record) == raw,
            "source floor requires exact canonical record",
        )
        folder.recheck()
        return record
    except Exception as exc:
        raise TestOnlySourceFloorError("source floor unavailable or changed") from exc
    finally:
        if folder is not None:
            folder.close()


def _check_fixture_floor(pin: _SourceFloorPin, original: bytes, policy_bytes: bytes) -> None:
    record = _read_fixture_floor(pin)
    _require(
        record.get("schema") == _SCHEMA
        and record.get("original_sha256") == _sha(original)
        and record.get("source_policy_sha256") == _sha(policy_bytes),
        "source original/policy differs from retained TEST floor",
    )
    _require(
        record == _document(original, policy_bytes),
        "source event identity/position differs from retained TEST floor",
    )


class _FloorGuardedReader:
    """Test adapter only; floor checks accompany all lookups and policy loads."""

    def __init__(self, reader: _LinuxFixtureOriginalReader, pin: _SourceFloorPin):
        self.reader = reader
        self.pin = pin

    def lookup_original_event(
        self, *, installation_id: str, operation_id: str, execution_id: str
    ) -> bytes:
        original = self.reader.lookup_original_event(
            installation_id=installation_id,
            operation_id=operation_id,
            execution_id=execution_id,
        )
        policy_bytes = self.reader._read("public-policy")
        _check_fixture_floor(self.pin, original, policy_bytes)
        return original

    def load_policy(self) -> source.ExplicitSourceVerificationPolicy:
        original = self.reader._read("original")
        policy_bytes = self.reader._read("public-policy")
        _check_fixture_floor(self.pin, original, policy_bytes)
        return source.parse_source_verification_policy(policy_bytes)


@pytest.fixture
def witnessed_source(source_on_disk, tmp_path):
    config, result = source_on_disk
    original = result["event"]
    disk_reader = _LinuxFixtureOriginalReader(config, original)
    # Require prior success of the existing non-authorizing verifier.
    with _inspect(disk_reader, original) as evidence:
        assert evidence.status == "NOT_QUALIFIED"
        assert evidence.production_authorization == "BLOCKED"
    floor_dir = tmp_path / "external-same-uid-test-source-floor"
    floor_dir.mkdir(mode=0o700)
    policy_bytes = (Path(config["root"]) / "events" / "public-policy").read_bytes()
    pin = _once_provision_fixture_floor(floor_dir, original, policy_bytes)
    return config, original, pin


def test_saved_original_matches_independent_test_floor_over_three_reads(witnessed_source):
    config, original, pin = witnessed_source
    reader = _LinuxFixtureOriginalReader(config, original)
    guarded = _FloorGuardedReader(reader, pin)
    with _inspect(guarded, original) as evidence:
        assert evidence.original_envelope == original
        assert evidence.status == "NOT_QUALIFIED"
        assert evidence.production_authorization == "BLOCKED"
        assert evidence.provenance_qualified is False
        assert evidence.durability_attested is False
        assert reader.lookups == 2
    assert reader.lookups == 3


def test_source_worker_restart_preserves_external_test_floor(witnessed_source):
    config, original, pin = witnessed_source
    status, result = _receive(*start_process(config=config, request=_REQUEST))
    assert status == "ok" and result["recovered"] is True
    assert result["event"] == original
    reader = _LinuxFixtureOriginalReader(config, original)
    with _inspect(_FloorGuardedReader(reader, pin), original) as evidence:
        assert evidence.original_sha256 == _sha(original)
        assert evidence.status == "NOT_QUALIFIED"


@pytest.mark.parametrize("attack", ["missing", "permissions", "inode", "symlink", "interrupted"])
def test_witness_loss_or_substitution_fails_closed(witnessed_source, attack):
    config, original, pin = witnessed_source
    path = Path(pin.directory) / _FILE
    if attack == "missing":
        path.unlink()
    elif attack == "permissions":
        path.chmod(0o600)
    elif attack == "inode":
        alternate = path.with_name("alternate")
        alternate.write_bytes(path.read_bytes())
        alternate.chmod(0o400)
        alternate.replace(path)
    elif attack == "symlink":
        path.rename(path.with_name("saved"))
        path.symlink_to("saved")
    else:
        path.rename(path.with_name("interrupted"))
    reader = _LinuxFixtureOriginalReader(config, original)
    with pytest.raises(TestOnlyRetainedSourceError, match="lookup denied") as denied:
        with _inspect(_FloorGuardedReader(reader, pin), original):
            pytest.fail("missing, replaced or interrupted witness must fail")
    assert isinstance(denied.value.__cause__, TestOnlySourceFloorError)


@pytest.mark.parametrize("repin_source", [False, True], ids=["existing-reader", "repinned-source"])
def test_witness_content_corruption_reaches_sha256_check(witnessed_source, repin_source):
    config, original, pin = witnessed_source
    reader = _LinuxFixtureOriginalReader(config, original)
    path = Path(pin.directory) / _FILE
    before = path.stat()
    raw = path.read_bytes()
    expected_sha256 = pin.document_sha256
    assert (before.st_dev, before.st_ino) == pin.file_identity
    assert before.st_size == len(raw)
    assert stat.S_IMODE(before.st_mode) == 0o400
    assert _sha(raw) == expected_sha256
    record = _read_fixture_floor(pin)
    assert canonical_json_bytes(record) == raw

    digest = record["original_sha256"]
    record["original_sha256"] = ("0" if digest[0] != "0" else "1") + digest[1:]
    corrupted = canonical_json_bytes(record)
    assert len(corrupted) == len(raw)
    offsets = [index for index, (old, new) in enumerate(zip(raw, corrupted)) if old != new]
    assert len(offsets) == 1
    offset = offsets[0]
    # Controlled same-UID TEST tampering: no truncate, rename or new floor pin.
    path.chmod(0o600)
    try:
        with path.open("r+b") as checkpoint:
            checkpoint.seek(offset)
            assert checkpoint.write(corrupted[offset : offset + 1]) == 1
            checkpoint.flush()
            os.fsync(checkpoint.fileno())
    finally:
        path.chmod(0o400)

    after = path.stat()
    assert (after.st_dev, after.st_ino) == pin.file_identity
    assert after.st_size == before.st_size
    assert stat.S_IMODE(after.st_mode) == stat.S_IMODE(before.st_mode) == 0o400
    assert path.read_bytes() == corrupted
    assert canonical_json_bytes(json.loads(corrupted)) == corrupted
    assert _sha(corrupted) != expected_sha256

    if repin_source:
        source_path = Path(config["root"]) / "events" / "original"
        source_before = source_path.stat()
        replacement = source_path.with_name("replacement-original")
        replacement.write_bytes(original)
        replacement.chmod(stat.S_IMODE(source_before.st_mode))
        replacement.replace(source_path)
        source_after = source_path.stat()
        assert (source_after.st_dev, source_after.st_ino) != (
            source_before.st_dev,
            source_before.st_ino,
        )
        # Re-pin the unchanged source bytes, retaining the original floor expectation.
        reader = _LinuxFixtureOriginalReader(config, original)
    assert reader._read("original") == original
    guarded = _FloorGuardedReader(reader, pin)
    assert guarded.pin is pin
    assert pin.document_sha256 == expected_sha256
    with pytest.raises(TestOnlyRetainedSourceError, match="lookup denied") as denied:
        with _inspect(guarded, original):
            pytest.fail("changed checkpoint digest must fail even after source re-pin")
    floor_error = denied.value.__cause__
    assert type(floor_error) is TestOnlySourceFloorError
    assert str(floor_error) == "source floor unavailable or changed"
    digest_error = floor_error.__cause__
    assert type(digest_error) is TestOnlySourceFloorError
    assert str(digest_error) == "source floor bytes differ from pin"


def test_source_bytes_change_with_new_reader_pin_cannot_replace_floor(witnessed_source):
    config, original, pin = witnessed_source
    path = Path(config["root"]) / "events" / "original"
    path.write_bytes(original[:-1])
    # The new reader is deliberately pinned AFTER tampering (test-only TOFU).
    # The separately retained witness must still reject its observed bytes.
    reader = _LinuxFixtureOriginalReader(config, original)
    altered = reader._read("original")
    assert altered != original
    with pytest.raises(TestOnlyRetainedSourceError, match="lookup denied") as denied:
        with _inspect(_FloorGuardedReader(reader, pin), original):
            pytest.fail("new source inode/time pin does not replace witness")
    assert isinstance(denied.value.__cause__, TestOnlySourceFloorError)
    assert "original/policy differs" in str(denied.value.__cause__)


def test_source_policy_bytes_change_with_new_reader_pin_is_denied(witnessed_source):
    config, original, pin = witnessed_source
    path = Path(config["root"]) / "events" / "public-policy"
    path.write_bytes(b"{}")
    reader = _LinuxFixtureOriginalReader(config, original)
    with pytest.raises(TestOnlyRetainedSourceError, match="lookup denied") as denied:
        with _inspect(_FloorGuardedReader(reader, pin), original):
            pytest.fail("new policy inode/time pin does not replace witness")
    assert isinstance(denied.value.__cause__, TestOnlySourceFloorError)
    assert "original/policy differs" in str(denied.value.__cause__)


def test_existing_floor_cannot_be_silently_provisioned_twice(witnessed_source):
    config, original, pin = witnessed_source
    policy_bytes = (Path(config["root"]) / "events" / "public-policy").read_bytes()
    with pytest.raises(TestOnlySourceFloorError, match="existing empty directory"):
        _once_provision_fixture_floor(Path(pin.directory), original, policy_bytes)


def test_orphan_stage_blocks_test_floor_provision(tmp_path, witnessed_source):
    _config, original, _pin = witnessed_source
    folder = tmp_path / "partial-floor"
    folder.mkdir(mode=0o700)
    (folder / "source-floor.stage").write_bytes(b"partial")
    policy_bytes = (Path(_config["root"]) / "events" / "public-policy").read_bytes()
    with pytest.raises(TestOnlySourceFloorError, match="existing empty directory"):
        _once_provision_fixture_floor(folder, original, policy_bytes)

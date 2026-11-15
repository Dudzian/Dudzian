"""SYNTHETIC image-boundary tests; no kernel immutable/host qualification.

Temporary files contain inert raw/QCOW2 data. Replaced host observations below
test fail-closed orchestration; they never certify filesystem enforcement.
"""

from __future__ import annotations

import hashlib
import os
import stat
import struct
import sys
import time
from pathlib import Path
from typing import Any

import pytest

from deployment import stage9_observer_kvm_image_v5d1 as image

pytestmark = [
    pytest.mark.unit,
    pytest.mark.security,
    pytest.mark.skipif(sys.platform != "linux", reason="Linux-only image descriptor mechanics"),
]


def _expected(path: Path, format: str = "raw") -> image.ExpectedImageV5d1:
    assert format in {"raw", "qcow2"}
    return image.ExpectedImageV5d1(str(path), hashlib.sha256(path.read_bytes()).hexdigest(), format)


def _qcow(*, backing: str | None = None, snapshots: int = 0, incompatible: int = 0) -> bytes:
    content = bytearray(4096)
    struct.pack_into(">4sI", content, 0, b"QFI\xfb", 3)
    struct.pack_into(">I", content, 20, 12)
    struct.pack_into(">I", content, 60, snapshots)
    struct.pack_into(">Q", content, 72, incompatible)
    struct.pack_into(">I", content, 100, 104)
    if backing is not None:
        encoded = backing.encode("utf-8")
        struct.pack_into(">QI", content, 8, 256, len(encoded))
        content[256 : 256 + len(encoded)] = encoded
    return bytes(content)


@pytest.fixture
def synthetic_host(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace host enforcement, explicitly producing synthetic observations."""
    monkeypatch.setattr(image, "_protected", lambda info, *, directory: None)
    monkeypatch.setattr(image, "_immutable", lambda fd: True)

    def info(fd: int, format: str, deadline: float) -> dict[str, object]:
        backing, _ = (
            image._qcow_header(fd, os.fstat(fd).st_size) if format == "qcow2" else (None, 0)
        )
        return {"format": format, "backing-filename": backing}

    monkeypatch.setattr(image, "_qemu_info", info)


def test_synthetic_raw_measurement_has_bound_artifact_and_no_qualification(
    tmp_path: Path, synthetic_host: None
) -> None:
    path = tmp_path / "disk.raw"
    path.write_bytes(b"synthetic raw disk bytes")
    expected = _expected(path)
    measured = image.measure_image_v5d1((expected,))
    assert measured.status == "IMAGE_MEASURED_LOCAL"
    assert measured.qualification == "BLOCKED"
    assert measured.artifacts[0].sha256 == expected.sha256
    assert measured.artifacts[0].inode == path.stat().st_ino
    assert measured.artifacts[0].path == str(path)


def test_synthetic_qcow2_chain_is_measured_including_base(
    tmp_path: Path, synthetic_host: None
) -> None:
    base = tmp_path / "base.raw"
    top = tmp_path / "top.qcow2"
    base.write_bytes(b"base bytes")
    top.write_bytes(_qcow(backing=str(base)))
    result = image.measure_image_v5d1((_expected(top, "qcow2"), _expected(base)))
    assert result.reason_codes == ()
    assert len(result.artifacts) == 2
    assert result.artifacts[0].backing_path == str(base)


def test_stable_writable_image_is_insufficient(
    tmp_path: Path, synthetic_host: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "disk.raw"
    path.write_bytes(b"unchanging is not immutable")
    monkeypatch.setattr(image, "_immutable", lambda fd: False)
    result = image.measure_image_v5d1((_expected(path),))
    assert "IMAGE_IMMUTABLE_VIEW_UNPROVEN" in result.reason_codes
    assert result.artifacts == ()


@pytest.mark.parametrize(
    "path", ["relative.raw", "/tmp/../etc/passwd", "/tmp//disk", "/tmp/./disk", "/", "/tmp/x\x00y"]
)
def test_unverified_paths_are_refused(path: str, synthetic_host: None) -> None:
    result = image.measure_image_v5d1((image.ExpectedImageV5d1(path, "a" * 64, "raw"),))
    assert result.status == "IMAGE_MEASUREMENT_BLOCKED"
    assert "IMAGE_PATH_UNSAFE" in result.reason_codes


@pytest.mark.parametrize("kind", ["file", "directory"])
def test_symlink_components_are_refused(kind: str, tmp_path: Path, synthetic_host: None) -> None:
    real = tmp_path / "real"
    real.mkdir()
    path = real / "disk.raw"
    path.write_bytes(b"bytes")
    alias = tmp_path / "alias"
    alias.symlink_to(path if kind == "file" else real)
    selected = alias if kind == "file" else alias / "disk.raw"
    result = image.measure_image_v5d1((_expected(selected),))
    assert result.status == "IMAGE_MEASUREMENT_BLOCKED"


def test_real_unprotected_ancestor_refused(tmp_path: Path) -> None:
    path = tmp_path / "disk.raw"
    path.write_bytes(b"bytes")
    result = image.measure_image_v5d1((_expected(path),))
    assert "IMAGE_PATH_UNPROTECTED" in result.reason_codes


def test_substituted_image_digest_refused(tmp_path: Path, synthetic_host: None) -> None:
    path = tmp_path / "disk.raw"
    path.write_bytes(b"original")
    expected = _expected(path)
    path.write_bytes(b"replacement")
    result = image.measure_image_v5d1((expected,))
    assert "IMAGE_DIGEST_MISMATCH" in result.reason_codes


@pytest.mark.parametrize(
    "backing", ["relative.raw", "../escape", "/etc/passwd", "/tmp/../etc/passwd"]
)
def test_unexpected_or_escaping_backing_never_reaches_qemu(
    backing: str, tmp_path: Path, synthetic_host: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "top.qcow2"
    path.write_bytes(_qcow(backing=backing))
    calls: list[int] = []
    monkeypatch.setattr(image, "_qemu_info", lambda *args: calls.append(1))
    result = image.measure_image_v5d1((_expected(path, "qcow2"),))
    assert result.status == "IMAGE_MEASUREMENT_BLOCKED"
    assert not calls


@pytest.mark.parametrize(
    "kind", ["snapshot", "external-data", "unknown-feature", "encryption", "truncated", "version"]
)
def test_qcow2_unsupported_state_never_reaches_qemu(
    kind: str, tmp_path: Path, synthetic_host: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw = bytearray(
        _qcow(
            snapshots=int(kind == "snapshot"),
            incompatible=4
            if kind == "external-data"
            else (1 << 63 if kind == "unknown-feature" else 0),
        )
    )
    if kind == "encryption":
        struct.pack_into(">I", raw, 32, 1)
    if kind == "version":
        struct.pack_into(">I", raw, 4, 1)
    path = tmp_path / "disk.qcow2"
    path.write_bytes(bytes(raw[:30] if kind == "truncated" else raw))
    calls: list[int] = []
    monkeypatch.setattr(image, "_qemu_info", lambda *args: calls.append(1))
    result = image.measure_image_v5d1((_expected(path, "qcow2"),))
    assert result.status == "IMAGE_MEASUREMENT_BLOCKED"
    assert not calls


def test_changed_path_detected_even_when_held_descriptor_unchanged(
    tmp_path: Path, synthetic_host: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "disk.raw"
    path.write_bytes(b"original")
    expected = _expected(path)

    def replace_path(*args: object) -> dict[str, object]:
        replacement = tmp_path / "replacement"
        replacement.write_bytes(b"original")
        replacement.replace(path)
        return {"format": "raw"}

    monkeypatch.setattr(image, "_qemu_info", replace_path)
    result = image.measure_image_v5d1((expected,))
    assert "IMAGE_CHANGED_DURING_MEASUREMENT" in result.reason_codes


def test_flag_removed_during_measurement_refused(
    tmp_path: Path, synthetic_host: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "disk.raw"
    path.write_bytes(b"bytes")
    flags = iter((True, False))
    monkeypatch.setattr(image, "_immutable", lambda fd: next(flags))
    result = image.measure_image_v5d1((_expected(path),))
    assert "IMAGE_CHANGED_DURING_MEASUREMENT" in result.reason_codes


@pytest.mark.parametrize(
    "observed",
    [
        {},
        {"format": "vmdk"},
        {"format": "raw", "backing-filename": "/etc/passwd"},
        {"format": "raw", "snapshots": [1]},
    ],
)
def test_incomplete_qemu_observation_refused(
    observed: dict[str, object],
    tmp_path: Path,
    synthetic_host: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "disk.raw"
    path.write_bytes(b"bytes")
    monkeypatch.setattr(image, "_qemu_info", lambda *args: observed)
    result = image.measure_image_v5d1((_expected(path),))
    assert result.status == "IMAGE_MEASUREMENT_BLOCKED"


def test_chain_cycles_refused(tmp_path: Path, synthetic_host: None) -> None:
    path = tmp_path / "disk.raw"
    path.write_bytes(b"bytes")
    expected = _expected(path)
    result = image.measure_image_v5d1((expected, expected))
    assert "IMAGE_CHAIN_CYCLE" in result.reason_codes


@pytest.mark.parametrize(
    "chain", [(), [], (None,), (image.ExpectedImageV5d1("/x", "not-sha", "raw"),)]
)
def test_malformed_expectation_refuses_without_io(chain: Any) -> None:
    result = image.measure_image_v5d1(chain)
    assert result.status == "IMAGE_MEASUREMENT_BLOCKED"


def test_hardlinked_image_refused(tmp_path: Path, synthetic_host: None) -> None:
    path = tmp_path / "disk.raw"
    path.write_bytes(b"bytes")
    os.link(path, tmp_path / "another")
    result = image.measure_image_v5d1((_expected(path),))
    assert "IMAGE_FILE_UNSUPPORTED" in result.reason_codes


def test_empty_image_refused(tmp_path: Path, synthetic_host: None) -> None:
    path = tmp_path / "disk.raw"
    path.write_bytes(b"")
    assert "IMAGE_FILE_UNSUPPORTED" in image.measure_image_v5d1((_expected(path),)).reason_codes


def test_read_truncation_and_timeout_refuse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "disk.raw"
    path.write_bytes(b"bytes")
    fd = os.open(path, os.O_RDONLY)
    try:
        with pytest.raises(image._ImageRefusal, match="TIMEOUT"):
            image._digest(fd, 5, -1)
        with pytest.raises(image._ImageRefusal, match="CHANGED"):
            image._digest(fd, 6, float("inf"))
    finally:
        os.close(fd)


def test_real_protection_checks_owner_permissions_and_type() -> None:
    for mode, uid in (
        (stat.S_IFREG | 0o666, 0),
        (stat.S_IFREG | 0o600, 1000),
        (stat.S_IFDIR | 0o700, 0),
    ):
        info = os.stat_result((mode, 1, 1, 1, uid, 0, 1, 0, 0, 0))
        with pytest.raises(image._ImageRefusal, match="UNPROTECTED"):
            image._protected(info, directory=False)


class _SyntheticProcess:
    def __init__(self, output: bytes, *, returncode: int = 0) -> None:
        reader, writer = os.pipe()
        os.write(writer, output)
        os.close(writer)
        self.stdout = os.fdopen(reader, "rb")
        self.returncode = returncode

    def poll(self) -> int:
        return self.returncode

    def wait(self, **kwargs: object) -> int:
        return self.returncode

    def kill(self) -> None:
        raise AssertionError("completed synthetic process must not be killed")


@pytest.mark.parametrize(
    ("output", "returncode", "reason"),
    [
        (b'{"format":"raw"}', 0, None),
        (b"[]", 0, "INCOMPLETE"),
        (b"garbage", 0, "JSON"),
        (b"{}", 1, "FAILED"),
        (b" " * 1024, 0, "LIMIT"),
    ],
)
def test_qemu_fixed_command_descriptor_binding_and_failures(
    output: bytes,
    returncode: int,
    reason: str | None,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "inert.raw"
    path.write_bytes(b"synthetic disk")
    fd = os.open(path, os.O_RDONLY)
    opened: list[str] = []
    commands: list[tuple[list[str], dict[str, Any]]] = []

    def open_binary(name: str) -> int:
        opened.append(name)
        return os.dup(fd)

    def process(arguments: list[str], **kwargs: Any) -> _SyntheticProcess:
        commands.append((arguments, kwargs))
        return _SyntheticProcess(output, returncode=returncode)

    monkeypatch.setattr(image, "_open_protected", open_binary)
    monkeypatch.setattr(image.subprocess, "Popen", process)
    monkeypatch.setattr(image, "_MAX_OUTPUT", 512)
    try:
        if reason == "JSON":
            with pytest.raises(ValueError):
                image._qemu_info(fd, "raw", time.monotonic() + 1)
        elif reason is not None:
            with pytest.raises(image._ImageRefusal, match=reason):
                image._qemu_info(fd, "raw", time.monotonic() + 1)
        else:
            assert image._qemu_info(fd, "raw", time.monotonic() + 1) == {"format": "raw"}
        assert opened == ["/usr/bin/qemu-img"]
        arguments, options = commands[0]
        assert arguments[1:5] == ["info", "--output=json", "-f", "raw"]
        assert arguments[-1] == f"/proc/self/fd/{fd}"
        assert options["pass_fds"][0] == fd
        assert "shell" not in options
        assert "--backing-chain" not in arguments
        assert "-U" not in arguments
    finally:
        os.close(fd)


def test_qemu_timeout_kills_and_reaps_child(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "inert.raw"
    path.write_bytes(b"bytes")
    fd = os.open(path, os.O_RDONLY)
    killed: list[bool] = []
    waited: list[bool] = []

    class PendingProcess(_SyntheticProcess):
        def poll(self) -> int | None:
            return None

        def kill(self) -> None:
            killed.append(True)

        def wait(self, **kwargs: object) -> int:
            waited.append(True)
            return 0

    monkeypatch.setattr(image, "_open_protected", lambda name: os.dup(fd))
    monkeypatch.setattr(image.subprocess, "Popen", lambda *args, **kwargs: PendingProcess(b""))
    try:
        with pytest.raises(image._ImageRefusal, match="TIMEOUT"):
            image._qemu_info(fd, "raw", -1)
        assert killed == [True]
        assert waited == [True]
    finally:
        os.close(fd)


@pytest.mark.parametrize("failure", ["digest", "malformed", "size", "changed"])
def test_approved_tool_mismatch_refuses_before_process_creation(
    failure: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "synthetic-tool"
    path.write_bytes(b"inert tool bytes")
    fd = os.open(path, os.O_RDONLY)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    calls: list[bool] = []
    monkeypatch.setattr(image, "_open_protected", lambda name: os.dup(fd))
    monkeypatch.setattr(image.subprocess, "Popen", lambda *args, **kwargs: calls.append(True))
    if failure == "digest":
        digest = "0" * 64
    if failure == "malformed":
        digest = "not-approved"
    if failure == "size":
        monkeypatch.setattr(image, "_MAX_TOOL_BYTES", 1)
    if failure == "changed":

        def changed_digest(*args: object) -> str:
            path.write_bytes(b"changed tool bytes while hashing")
            return digest

        monkeypatch.setattr(image, "_digest", changed_digest)
    try:
        with pytest.raises(image._ImageRefusal, match="IMAGE_TOOL_IDENTITY_MISMATCH"):
            image._qemu_info(fd, "raw", time.monotonic() + 1, approved_qemu_sha256=digest)
        assert not calls
    finally:
        os.close(fd)


def test_approved_tool_is_hashed_before_process_creation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "synthetic-tool"
    path.write_bytes(b"inert approved tool")
    fd = os.open(path, os.O_RDONLY)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    hashes: list[str] = []
    actual_digest = image._digest

    def hash_tool(descriptor: int, size: int, deadline: float) -> str:
        value = actual_digest(descriptor, size, deadline)
        hashes.append(value)
        return value

    def process(*args: object, **kwargs: object) -> _SyntheticProcess:
        assert hashes == [digest]
        return _SyntheticProcess(b'{"format":"raw"}')

    monkeypatch.setattr(image, "_open_protected", lambda name: os.dup(fd))
    monkeypatch.setattr(image, "_digest", hash_tool)
    monkeypatch.setattr(image.subprocess, "Popen", process)
    try:
        assert image._qemu_info(fd, "raw", time.monotonic() + 1, approved_qemu_sha256=digest) == {
            "format": "raw"
        }
    finally:
        os.close(fd)


def test_aggregate_tool_pin_propagates_to_image_parser(
    tmp_path: Path, synthetic_host: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "disk.raw"
    path.write_bytes(b"synthetic disk bytes")
    pins: list[str] = []

    def info(
        fd: int, format: str, deadline: float, *, approved_qemu_sha256: str
    ) -> dict[str, object]:
        pins.append(approved_qemu_sha256)
        return {"format": "raw"}

    monkeypatch.setattr(image, "_qemu_info", info)
    result = image.measure_image_v5d1((_expected(path),), approved_qemu_sha256="a" * 64)
    assert result.status == "IMAGE_MEASURED_LOCAL"
    assert result.qualification == "BLOCKED"
    assert pins == ["a" * 64]

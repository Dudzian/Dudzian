"""Offline refusal boundaries for the frozen real-DEB measurement orchestration."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import pytest

from scripts import verify_stage9_real_deb_v2 as experiment

pytestmark = [
    pytest.mark.security,
    pytest.mark.unit,
    pytest.mark.skipif(sys.platform != "linux", reason="read-only DEB experiment is Linux-only"),
]


def _forbid_decoder(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*args: Any, **kwargs: Any) -> None:
        pytest.fail("decoder reached before frozen expectations and full archive hash/size passed")

    monkeypatch.setattr(experiment, "_ZstdReader", forbidden)


def test_frozen_inventory_tamper_refuses_before_archive_io(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    evidence = tmp_path / "inventory.json"
    evidence.write_bytes(experiment._INVENTORY.read_bytes() + b" ")
    monkeypatch.setattr(experiment, "_INVENTORY", evidence)
    _forbid_decoder(monkeypatch)
    with pytest.raises(ValueError, match="frozen inventory digest mismatch"):
        experiment.measure(tmp_path / "absent-input-directory")


@pytest.mark.parametrize("same_size", [False, True], ids=["size-mismatch", "hash-mismatch"])
def test_archive_size_or_hash_mismatch_refuses_before_decoder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, same_size: bool
) -> None:
    _forbid_decoder(monkeypatch)
    frozen = json.loads(experiment._INVENTORY.read_bytes())
    package = next(p for p in frozen["packages"] if p["package"] == "python3.12-minimal")
    path = tmp_path / Path(package["filename"]).name
    with path.open("wb") as stream:
        stream.truncate(package["size"] if same_size else 1)
    expected = "archive SHA-256 mismatch" if same_size else "archive type/size mismatch"
    with pytest.raises(ValueError, match=expected):
        experiment.measure(tmp_path, "python3.12-minimal")


def test_failed_measurement_keeps_all_qualification_blocks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _forbid_decoder(monkeypatch)
    monkeypatch.setattr(sys, "argv", ["verify_stage9_real_deb_v2", "--directory", str(tmp_path)])
    assert experiment.main() == 2
    result = json.loads(capsys.readouterr().out)
    assert result["REAL_DEB_ZSTD_PARSED"] == "FAIL"
    assert result["REAL_DEB_MEMBER_INVENTORY"] == "FAIL"
    assert result["REAL_DEB_BYTE_CONSISTENCY"] == "INCOMPLETE"
    assert result["SOURCE_AUTHENTICITY"] == "UNVERIFIED"
    assert result["V2_COMPLETE"] == "NO"
    assert result["ARCHIVE_TO_TARGET_METADATA_MAPPING"] == "NOT_YET_PROVEN"
    assert result["stage_9"] == "IN_PROGRESS"
    assert result["qualification"] == "BLOCKED"
    assert result["CONTRACT_APPROVED"] == "NO"
    assert result["STAGE_3A_COMPLETE"] == "NO"
    assert result["PRODUCTION_QUALIFIED"] == "NO"
    assert result["production_status"] == "PRODUCTION_NO_GO"
    assert result["runtime_attestation"] == "MISSING"
    assert result["host_pid_namespace"] == "UNVERIFIED"


def test_selecting_an_undeclared_package_never_reads_an_archive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _forbid_decoder(monkeypatch)
    with pytest.raises(ValueError, match="no declared package selected"):
        experiment.measure(tmp_path, "unapproved-package")


@pytest.fixture
def synthetic_declared_archive(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A test-only prior declaration; it provides no independent source evidence."""
    import hashlib

    import zstandard

    from tests.deployment.test_stage9_observer_offline import _tar

    control = (
        b"Package: python3.12-minimal\nVersion: 3.12.3-1ubuntu0.17\n"
        b"Architecture: amd64\nDescription: inert orchestration fixture\n"
    )
    payload = b"fixed synthetic comparison bytes with no authorization\n"
    decoded = {
        "control.tar.zst": _tar([("control", control)]),
        "data.tar.zst": _tar([("usr/share/observed.txt", payload)]),
    }
    bodies = [("debian-binary", b"2.0\n")] + [
        (name, zstandard.ZstdCompressor(write_checksum=True).compress(body))
        for name, body in decoded.items()
    ]
    archive = bytearray(b"!<arch>\n")
    records: list[dict[str, Any]] = []
    for name, body in bodies:
        header = (f"{name + '/':<16}{0:<12}{0:<6}{0:<6}{'100644':<8}{len(body):<10}`\n").encode(
            "ascii"
        )
        archive.extend(header)
        record: dict[str, Any] = {
            "name": name,
            "offset": len(archive),
            "size": len(body),
            "header_ascii": header.decode("ascii"),
        }
        if name in decoded:
            record.update(
                zstd={},
                unpacked_bytes=len(decoded[name]),
                unpacked_sha256=hashlib.sha256(decoded[name]).hexdigest(),
                records=[
                    {
                        "name": "./usr/share/observed.txt",
                        "typeflag": "0",
                        "sha256": hashlib.sha256(payload).hexdigest(),
                        "size": len(payload),
                    }
                ]
                if name == "data.tar.zst"
                else [],
            )
        records.append(record)
        archive.extend(body)
        if len(body) % 2:
            archive.extend(b"\n")
    directory = tmp_path / "debs"
    directory.mkdir()
    (directory / "observed.deb").write_bytes(archive)
    inventory = {
        "classification": "SYNTHETIC_TEST_ONLY_NOT_INDEPENDENT_EVIDENCE",
        "packages": [
            {
                "package": "python3.12-minimal",
                "version": "3.12.3-1ubuntu0.17",
                "architecture": "amd64",
                "filename": "observed.deb",
                "size": len(archive),
                "sha256": hashlib.sha256(archive).hexdigest(),
                "ar": records,
            }
        ],
    }
    evidence = tmp_path / "test-only-prior-declaration.json"
    evidence.write_text(json.dumps(inventory), encoding="utf8")
    monkeypatch.setattr(experiment, "_INVENTORY", evidence)
    monkeypatch.setattr(
        experiment, "_INVENTORY_SHA256", hashlib.sha256(evidence.read_bytes()).hexdigest()
    )
    return directory


def test_synthetic_complete_byte_match_keeps_source_and_qualification_unverified(
    synthetic_declared_archive: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["verify_stage9_real_deb_v2", "--directory", str(synthetic_declared_archive)],
    )
    assert experiment.main() == 0
    result = json.loads(capsys.readouterr().out)
    package = result["packages"][0]
    assert package["REAL_DEB_ZSTD_PARSED"] == "PASS"
    assert package["REAL_DEB_MEMBER_INVENTORY"] == "PASS"
    assert package["REAL_DEB_BYTE_CONSISTENCY"] == "MATCH"
    assert package["SOURCE_AUTHENTICITY"] == "UNVERIFIED"
    assert result["expectation_classification"] == "SYNTHETIC_TEST_ONLY_NOT_INDEPENDENT_EVIDENCE"
    assert result["V2_COMPLETE"] == "NO"
    assert result["CONTRACT_APPROVED"] == "NO"
    assert result["qualification"] == "BLOCKED"
    assert result["production_status"] == "PRODUCTION_NO_GO"


@pytest.mark.parametrize("fault", ["ar-header", "decoded-hash"])
def test_inconsistent_frozen_comparison_refuses_a_hash_pinned_archive(
    synthetic_declared_archive: Path, monkeypatch: pytest.MonkeyPatch, fault: str
) -> None:
    import hashlib

    inventory = json.loads(experiment._INVENTORY.read_bytes())
    if fault == "ar-header":
        inventory["packages"][0]["ar"][0]["header_ascii"] = "x" * 60
        message = "frozen ar header mismatch"
        _forbid_decoder(monkeypatch)
    else:
        inventory["packages"][0]["ar"][1]["unpacked_sha256"] = "0" * 64
        message = "frozen decoded-frame hash/size mismatch"
    experiment._INVENTORY.write_text(json.dumps(inventory), encoding="utf8")
    monkeypatch.setattr(
        experiment,
        "_INVENTORY_SHA256",
        hashlib.sha256(experiment._INVENTORY.read_bytes()).hexdigest(),
    )
    with pytest.raises(ValueError, match=message):
        experiment.measure(synthetic_declared_archive)


def test_root_measurement_refuses_before_inventory_or_archive_io(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(os, "geteuid", lambda: 0)
    monkeypatch.setattr(experiment, "_INVENTORY", tmp_path / "missing-inventory")
    _forbid_decoder(monkeypatch)
    with pytest.raises(experiment.ContractRefusal, match="FILE_POLICY_REJECTED"):
        experiment.measure(tmp_path)


def test_symlinked_ancestor_refuses_before_archive_decode(
    synthetic_declared_archive: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    alias = tmp_path / "alias"
    alias.symlink_to(tmp_path, target_is_directory=True)
    _forbid_decoder(monkeypatch)
    with pytest.raises(OSError):
        experiment.measure(alias / synthetic_declared_archive.name)


def test_hardlinked_archive_refuses_before_hash_or_decode(
    synthetic_declared_archive: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    os.link(synthetic_declared_archive / "observed.deb", tmp_path / "second-name.deb")
    _forbid_decoder(monkeypatch)
    with pytest.raises(experiment.ContractRefusal, match="FILE_POLICY_REJECTED"):
        experiment.measure(synthetic_declared_archive)


@pytest.mark.parametrize("mutation", ["mode", "link", "replacement", "directory"])
def test_archive_metadata_or_directory_binding_change_never_matches(
    synthetic_declared_archive: Path, monkeypatch: pytest.MonkeyPatch, mutation: str
) -> None:
    original = experiment._pread
    mutated = False
    descriptors: list[int] = []

    def mutate_on_read(fd: int, size: int, offset: int) -> bytes:
        nonlocal mutated
        data = original(fd, size, offset)
        if not mutated:
            mutated = True
            descriptors.append(fd)
            archive = synthetic_declared_archive / "observed.deb"
            if mutation == "mode":
                archive.chmod(0o600 if archive.stat().st_mode & 0o077 else 0o644)
            elif mutation == "link":
                os.link(archive, synthetic_declared_archive.parent / "second-name.deb")
            elif mutation == "replacement":
                replacement = synthetic_declared_archive.parent / "replacement.deb"
                replacement.write_bytes(archive.read_bytes())
                replacement.replace(archive)
            else:
                synthetic_declared_archive.rename(synthetic_declared_archive.with_name("moved"))
                synthetic_declared_archive.mkdir()
        return data

    monkeypatch.setattr(experiment, "_pread", mutate_on_read)
    with pytest.raises(ValueError, match="archive identity changed|FILE_CHANGED"):
        experiment.measure(synthetic_declared_archive)
    assert mutated
    for fd in descriptors:
        with pytest.raises(OSError):
            os.fstat(fd)


@pytest.mark.parametrize("failure", ["native-read", "native-close", "double-failure"])
@pytest.mark.parametrize("kind", ["zstd", "os"])
def test_native_decoder_failure_is_controlled_and_sanitized(
    synthetic_declared_archive: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    failure: str,
    kind: str,
) -> None:
    import zstandard

    original = zstandard.ZstdDecompressor
    error_type = zstandard.ZstdError if kind == "zstd" else OSError
    closed: list[bool] = []

    class FailingStream:
        def __init__(self, stream: Any) -> None:
            self.stream = stream

        def read(self, size: int) -> bytes:
            if failure in {"native-read", "double-failure"}:
                raise error_type("untrusted archive message")
            return self.stream.read(size)

        def close(self) -> None:
            self.stream.close()
            closed.append(True)
            if failure == "native-close":
                raise error_type("untrusted cleanup message")
            if failure == "double-failure":
                # Deliberately use a different refusal for the second failure.
                raise MemoryError("untrusted cleanup message")

    class FailingDecoder:
        def __init__(self, **kwargs: Any) -> None:
            self.context = original(**kwargs)

        def memory_size(self) -> int:
            return self.context.memory_size()

        def stream_reader(self, *args: Any, **kwargs: Any) -> FailingStream:
            return FailingStream(self.context.stream_reader(*args, **kwargs))

    monkeypatch.setattr(zstandard, "ZstdDecompressor", FailingDecoder)
    monkeypatch.setattr(
        sys,
        "argv",
        ["verify_stage9_real_deb_v2", "--directory", str(synthetic_declared_archive)],
    )
    assert experiment.main() == 2
    output = capsys.readouterr().out
    result = json.loads(output)
    expected = "FORMAT_REJECTED" if kind == "zstd" else "PROVENANCE_INCOMPLETE"
    assert result["error"] == expected
    assert result["REAL_DEB_BYTE_CONSISTENCY"] == "INCOMPLETE"
    assert result["V2_COMPLETE"] == "NO"
    assert closed == [True]
    assert "untrusted" not in output


def test_os_error_diagnostic_never_exposes_untrusted_paths(
    synthetic_declared_archive: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fail_read(*args: Any) -> bytes:
        raise OSError("untrusted input path and native diagnostic")

    monkeypatch.setattr(experiment, "_pread", fail_read)
    monkeypatch.setattr(
        sys,
        "argv",
        ["verify_stage9_real_deb_v2", "--directory", str(synthetic_declared_archive)],
    )
    assert experiment.main() == 2
    output = capsys.readouterr().out
    assert json.loads(output)["error"] == "PROVENANCE_INCOMPLETE"
    assert "untrusted" not in output


def test_unfinished_decoder_cannot_be_reported_as_complete(
    synthetic_declared_archive: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    closed: list[bool] = []

    class UnfinishedReader:
        finished = False

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def read(self, size: int) -> bytes:
            return b""

        def close(self) -> None:
            closed.append(True)

    monkeypatch.setattr(experiment, "_ZstdReader", UnfinishedReader)
    with pytest.raises(experiment.ContractRefusal, match="PROVENANCE_INCOMPLETE"):
        experiment.measure(synthetic_declared_archive)
    assert closed == [True]


def test_directory_substitution_during_final_path_rebinding_refuses(
    synthetic_declared_archive: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = experiment._absolute_directory
    opens = 0
    descriptors: list[int] = []

    def substitute_on_rebind(*args: Any) -> int:
        nonlocal opens
        opens += 1
        if opens == 2:
            synthetic_declared_archive.rename(synthetic_declared_archive.with_name("moved"))
            synthetic_declared_archive.mkdir()
        fd = original(*args)
        descriptors.append(fd)
        return fd

    monkeypatch.setattr(experiment, "_absolute_directory", substitute_on_rebind)
    with pytest.raises(experiment.ContractRefusal, match="FILE_CHANGED"):
        experiment.measure(synthetic_declared_archive)
    assert opens == 2
    for fd in descriptors:
        with pytest.raises(OSError):
            os.fstat(fd)

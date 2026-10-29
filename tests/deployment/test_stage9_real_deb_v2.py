"""Offline refusal boundaries for the frozen real-DEB measurement orchestration."""

from __future__ import annotations

import json
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

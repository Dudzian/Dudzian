"""Frozen declarations and advisory-only reporting for the real-package V3 tool."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from deployment.stage9_observer_contract import ContractRefusal
from scripts import verify_stage9_real_deb_v3 as experiment
from tests.deployment.test_stage9_observer_package_v3 import PackageCase, build_package_case

pytestmark = [
    pytest.mark.security,
    pytest.mark.unit,
    pytest.mark.skipif(
        sys.platform != "linux", reason="frozen descriptor experiment is Linux-only"
    ),
]


@pytest.fixture
def frozen_case(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> PackageCase:
    case = build_package_case(tmp_path / "debs", variant="zst")
    evidence_directory = tmp_path / "evidence"
    evidence_directory.mkdir()
    evidence = evidence_directory / "frozen.json"
    evidence.write_bytes(case.raw_inventory)
    monkeypatch.setattr(experiment, "_INVENTORY", evidence)
    monkeypatch.setattr(
        experiment, "_INVENTORY_SHA256", hashlib.sha256(case.raw_inventory).hexdigest()
    )
    return case


def _forbid_package_io(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*args: Any, **kwargs: Any) -> None:
        pytest.fail("package IO reached before frozen declaration checks passed")

    monkeypatch.setattr(experiment, "verify_ubuntu_package_inventory_v3", forbidden)


def _assert_blocks(report: dict[str, Any]) -> None:
    assert report["scope"] == "INERT_PACKAGE_INVENTORY_ONLY"
    assert report["classification"] == "SAME_ORIGIN_OBSERVED_COMPARISON_ONLY"
    assert report["independent_acceptance"] == "NOT_INDEPENDENT_ACCEPTANCE"
    assert report["SOURCE_AUTHENTICITY"] == "UNVERIFIED"
    assert report["source_authenticity"] == "UNVERIFIED"
    assert report["acceptance"] == "UNVERIFIED"
    assert report["execution"] == "MISSING"
    assert report["ARCHIVE_TO_TARGET_METADATA_MAPPING"] == "NOT_YET_PROVEN"
    assert report["stage_9"] == "IN_PROGRESS"
    assert report["qualification"] == "BLOCKED"
    assert report["CONTRACT_APPROVED"] == "NO"
    assert report["STAGE_3A_COMPLETE"] == "NO"
    assert report["PRODUCTION_QUALIFIED"] == "NO"
    assert report["production_status"] == "PRODUCTION_NO_GO"
    assert report["runtime_attestation"] == "MISSING"
    assert report["host_pid_namespace"] == "UNVERIFIED"
    assert report["V2_COMPLETE"] == "NO"
    assert report["refused"] is True


def test_synthetic_package_match_is_not_source_image_or_production_acceptance(
    frozen_case: PackageCase, capsys: pytest.CaptureFixture[str]
) -> None:
    assert experiment.main(["--directory", str(frozen_case.directory)]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["consistency"] == "MATCH"
    assert report["reason_codes"] == []
    assert len(report["packages"]) == 1
    package = report["packages"][0]
    assert package["REAL_DEB_MEMBER_INVENTORY"] == "PASS"
    assert package["REAL_DEB_BYTE_CONSISTENCY"] == "MATCH"
    assert package["SOURCE_AUTHENTICITY"] == "UNVERIFIED"
    assert package["full_archive_sha256_and_size"] == "PASS"
    assert package["archive_sha256"] == frozen_case.package["sha256"]
    assert package["control"]["regular"] == 2
    assert package["control"]["directory"] == 1
    assert package["data"]["regular"] == 1
    assert package["data"]["directory"] == 2
    assert package["data"]["symlink"] == 2
    assert package["data"]["complete_eof"] == "PASS"
    assert package["control"]["header_profile"] == "gnu-inert-v3"
    assert package["data"]["header_profile"] == "gnu-inert-v3"
    assert "control" not in package["control"]
    _assert_blocks(report)


def test_frozen_digest_mismatch_refuses_before_archive_io(
    frozen_case: PackageCase, monkeypatch: pytest.MonkeyPatch
) -> None:
    experiment._INVENTORY.write_bytes(frozen_case.raw_inventory + b" ")
    _forbid_package_io(monkeypatch)
    with pytest.raises(ContractRefusal, match="ARTIFACT_MISMATCH"):
        experiment.measure(frozen_case.directory)


@pytest.mark.parametrize(
    "mutation", ["symlink", "hardlink", "empty", "oversized", "directory", "ancestor"]
)
def test_unsafe_frozen_declaration_refuses_before_package_io(
    frozen_case: PackageCase, monkeypatch: pytest.MonkeyPatch, mutation: str
) -> None:
    evidence = experiment._INVENTORY
    if mutation == "symlink":
        original = evidence.with_name("original.json")
        evidence.rename(original)
        evidence.symlink_to(original)
    elif mutation == "hardlink":
        os.link(evidence, evidence.with_name("alias.json"))
    elif mutation == "empty":
        evidence.write_bytes(b"")
    elif mutation == "oversized":
        with evidence.open("wb") as stream:
            stream.truncate(2 * 1024 * 1024 + 1)
    elif mutation == "directory":
        evidence.unlink()
        evidence.mkdir()
    else:
        alias = evidence.parent.parent / "evidence-alias"
        alias.symlink_to(evidence.parent, target_is_directory=True)
        monkeypatch.setattr(experiment, "_INVENTORY", alias / evidence.name)
    _forbid_package_io(monkeypatch)
    with pytest.raises((ContractRefusal, OSError)):
        experiment.measure(frozen_case.directory)


def test_frozen_declaration_cross_device_refuses_before_package_io(
    frozen_case: PackageCase, monkeypatch: pytest.MonkeyPatch
) -> None:
    native_open, native_fstat = os.open, os.fstat
    watched: set[int] = set()

    def track_open(path: Any, flags: int, mode: int = 0o600, *, dir_fd: int | None = None) -> int:
        descriptor = native_open(path, flags, mode, dir_fd=dir_fd)
        if path == experiment._INVENTORY.name:
            watched.add(descriptor)
        return descriptor

    def changed_device(descriptor: int) -> Any:
        info = native_fstat(descriptor)
        if descriptor in watched:
            return SimpleNamespace(
                st_mode=info.st_mode, st_nlink=info.st_nlink, st_dev=info.st_dev + 1
            )
        return info

    monkeypatch.setattr(os, "open", track_open)
    monkeypatch.setattr(os, "fstat", changed_device)
    _forbid_package_io(monkeypatch)
    with pytest.raises(ContractRefusal, match="FILE_POLICY_REJECTED"):
        experiment.measure(frozen_case.directory)


@pytest.mark.parametrize(
    "mutation", ["mode", "link", "replacement", "directory", "directory-rebinding"]
)
def test_frozen_file_or_directory_change_during_read_refuses(
    frozen_case: PackageCase, monkeypatch: pytest.MonkeyPatch, mutation: str
) -> None:
    evidence = experiment._INVENTORY
    native_read = os.read
    changed = False

    def read_then_mutate(descriptor: int, size: int) -> bytes:
        nonlocal changed
        raw = native_read(descriptor, size)
        if not changed:
            changed = True
            if mutation == "mode":
                evidence.chmod(0o600)
            elif mutation == "link":
                os.link(evidence, evidence.with_name("concurrent-link.json"))
            elif mutation == "replacement":
                replacement = evidence.with_name("replacement.json")
                replacement.write_bytes(evidence.read_bytes())
                replacement.replace(evidence)
            elif mutation == "directory":
                evidence.parent.chmod(0o700)
            else:
                evidence.parent.rename(evidence.parent.with_name("moved-evidence"))
                evidence.parent.mkdir()
        return raw

    monkeypatch.setattr(os, "read", read_then_mutate)
    _forbid_package_io(monkeypatch)
    with pytest.raises(ContractRefusal, match="FILE_CHANGED"):
        experiment.measure(frozen_case.directory)


def test_incomplete_frozen_file_read_refuses_before_package_io(
    frozen_case: PackageCase, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(os, "read", lambda *args: b"")
    _forbid_package_io(monkeypatch)
    with pytest.raises(ContractRefusal, match="PROVENANCE_INCOMPLETE"):
        experiment.measure(frozen_case.directory)


def _fail_frozen_file_close_after_native_release(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[list[int], list[int], list[int]]:
    """Track only operation-owned handles; release before injecting the error."""
    native_open, native_close = os.open, os.close
    native_directory = experiment._absolute_directory
    directories: list[int] = []
    files: list[int] = []
    closed: list[int] = []

    def track_directory(*args: Any, **kwargs: Any) -> int:
        descriptor = native_directory(*args, **kwargs)
        directories.append(descriptor)
        return descriptor

    def track_open(path: Any, flags: int, mode: int = 0o600, *, dir_fd: int | None = None) -> int:
        descriptor = native_open(path, flags, mode, dir_fd=dir_fd)
        if path == experiment._INVENTORY.name:
            files.append(descriptor)
        return descriptor

    def fail_file_close(descriptor: int) -> None:
        native_close(descriptor)
        if descriptor in directories or descriptor in files:
            closed.append(descriptor)
        if descriptor in files:
            raise OSError("untrusted cleanup diagnostic")

    monkeypatch.setattr(experiment, "_absolute_directory", track_directory)
    monkeypatch.setattr(os, "open", track_open)
    monkeypatch.setattr(os, "close", fail_file_close)
    return directories, files, closed


def test_frozen_cleanup_failure_preserves_first_refusal_and_closes_parent(
    frozen_case: PackageCase, monkeypatch: pytest.MonkeyPatch
) -> None:
    experiment._INVENTORY.write_bytes(frozen_case.raw_inventory + b" ")
    directories, files, closed = _fail_frozen_file_close_after_native_release(monkeypatch)
    _forbid_package_io(monkeypatch)
    with pytest.raises(ContractRefusal, match="ARTIFACT_MISMATCH"):
        experiment.measure(frozen_case.directory)
    assert len(directories) == len(files) == 1
    assert closed == [files[0], directories[0]]


def test_frozen_cleanup_failure_after_valid_read_refuses_and_closes_all_handles(
    frozen_case: PackageCase,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    directories, files, closed = _fail_frozen_file_close_after_native_release(monkeypatch)
    _forbid_package_io(monkeypatch)
    assert experiment.main(["--directory", str(frozen_case.directory)]) == 2
    report = json.loads(capsys.readouterr().out)
    assert report["consistency"] == "INCOMPLETE"
    assert report["reason_codes"] == ["PROVENANCE_INCOMPLETE"]
    assert report["packages"] == []
    _assert_blocks(report)
    assert len(directories) == 2
    assert len(files) == 1
    assert closed == [directories[1], files[0], directories[0]]


@pytest.mark.parametrize("platform", ["root", "win32"])
def test_unsupported_execution_refuses_before_frozen_declaration_read(
    frozen_case: PackageCase, monkeypatch: pytest.MonkeyPatch, platform: str
) -> None:
    def forbidden() -> bytes:
        pytest.fail("frozen declaration opened on unsupported execution context")

    monkeypatch.setattr(experiment, "_frozen_inventory", forbidden)
    if platform == "root":
        monkeypatch.setattr(os, "geteuid", lambda: 0)
        code = "FILE_POLICY_REJECTED"
    else:
        monkeypatch.setattr(sys, "platform", platform)
        code = "UNSUPPORTED_SOURCE"
    with pytest.raises(ContractRefusal, match=code):
        experiment.measure(frozen_case.directory)


@pytest.mark.parametrize("failure", ["whole-size", "whole-hash", "frozen-pin", "missing-archive"])
def test_failed_measurement_never_emits_partial_package_success(
    frozen_case: PackageCase, capsys: pytest.CaptureFixture[str], failure: str
) -> None:
    if failure == "whole-size":
        frozen_case.path.write_bytes(b"bad archive")
    elif failure == "whole-hash":
        raw = bytearray(frozen_case.path.read_bytes())
        raw[-1] ^= 1
        frozen_case.path.write_bytes(raw)
    elif failure == "frozen-pin":
        experiment._INVENTORY.write_bytes(frozen_case.raw_inventory + b" ")
    else:
        frozen_case.path.unlink()
    assert experiment.main(["--directory", str(frozen_case.directory)]) == 2
    report = json.loads(capsys.readouterr().out)
    assert report["consistency"] != "MATCH"
    assert report["reason_codes"]
    assert report["packages"] == []
    _assert_blocks(report)


@pytest.mark.parametrize(
    "failure", [OSError, ValueError, MemoryError, OverflowError, RecursionError]
)
def test_cli_sanitizes_untrusted_failure_text_and_retains_blocks(
    frozen_case: PackageCase,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    failure: type[Exception],
) -> None:
    def fail(*args: Any, **kwargs: Any) -> Any:
        raise failure("/secret/host-path attacker-controlled native message")

    monkeypatch.setattr(experiment, "measure", fail)
    assert experiment.main(["--directory", str(frozen_case.directory)]) == 2
    output = capsys.readouterr().out
    assert "secret" not in output
    assert "attacker" not in output
    report = json.loads(output)
    assert report["packages"] == []
    _assert_blocks(report)


@pytest.mark.parametrize("option", ["--download", "--update-expectations", "--package"])
def test_cli_has_no_acquisition_update_or_partial_inventory_mode(
    frozen_case: PackageCase, option: str
) -> None:
    with pytest.raises(SystemExit) as refused:
        experiment.main(["--directory", str(frozen_case.directory), option])
    assert refused.value.code == 2

"""The static evidence inspector reads bytes and never executes wheel sources."""

from __future__ import annotations

import json
from pathlib import Path
import zipfile

import pytest

from scripts import stage9_v5d6_inspect_greenlet as inspector
from tests.deployment.test_stage9_v5d4_dependencies import wheel
from tests.deployment.test_stage9_v5d6_headers import rewrite


@pytest.fixture
def inert_wheel(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    content = b"/* TEST ONLY inert header */".ljust(4755, b" ")
    path, _ = wheel(
        tmp_path,
        "greenlet",
        "3.5.6",
        tag="cp312-cp312-manylinux_2_24_x86_64.manylinux_2_28_x86_64",
        files={inspector.HEADER: content},
    )
    # Production constants remain pinned; only this synthetic fixture substitutes them.
    monkeypatch.setattr(inspector, "WHEEL_SHA256", inspector.sha256(path.read_bytes()))
    monkeypatch.setattr(inspector, "HEADER_SHA256", inspector.sha256(content))
    return path


def test_inspection_retains_record_and_original_member_evidence(inert_wheel: Path) -> None:
    before = inert_wheel.read_bytes()
    report = inspector.inspect_original(inert_wheel)
    assert report["approval"] == "NOT_GRANTED"
    assert report["execution"] == "NOT_EXECUTED"
    assert report["package"]["record_sha256_size_result"] == "PASS"
    assert report["package"]["unsupported_members"] == [inspector.HEADER]
    assert report["header"]["mode"] == 0o100644
    assert report["header"]["size"] == 4755
    assert report["header"]["record_entry"][0] == inspector.HEADER
    assert report["package"]["record_file_set_exact"] is True
    assert inert_wheel.read_bytes() == before


def test_archive_mutation_refuses_even_if_members_are_identical(inert_wheel: Path) -> None:
    with zipfile.ZipFile(inert_wheel, "a") as archive:
        archive.comment = b"different archive"
    with pytest.raises(ValueError, match="INSPECTION_WHEEL_IDENTITY"):
        inspector.inspect_original(inert_wheel)


@pytest.mark.parametrize("change", ["missing", "bad_hash", "extra"])
def test_full_record_and_exact_unsupported_set_are_required(
    inert_wheel: Path, monkeypatch: pytest.MonkeyPatch, change: str
) -> None:
    if change == "missing":
        rewrite(inert_wheel, omit_record_row=inspector.HEADER)
        reason = "INSPECTION_RECORD_FILE_SET"
    elif change == "bad_hash":
        rewrite(inert_wheel, bad_record_sha=inspector.HEADER)
        reason = "INSPECTION_RECORD_INTEGRITY"
    else:
        rewrite(
            inert_wheel,
            updates={"greenlet-3.5.6.data/headers/extra.h": b"TEST ONLY"},
            reseal_record=True,
        )
        reason = "INSPECTION_UNSUPPORTED_MEMBERS"
    # Permit inspection to reach the independently checked RECORD boundary.
    monkeypatch.setattr(inspector, "WHEEL_SHA256", inspector.sha256(inert_wheel.read_bytes()))
    with pytest.raises(ValueError, match=reason):
        inspector.inspect_original(inert_wheel)


def test_symlink_does_not_substitute_for_original_archive(
    inert_wheel: Path, tmp_path: Path
) -> None:
    alias = tmp_path / "alias"
    alias.mkdir()
    link = alias / inert_wheel.name
    try:
        link.symlink_to(inert_wheel)
    except (OSError, NotImplementedError):
        pytest.skip("Host cannot create an unprivileged symlink")
    with pytest.raises(ValueError, match="INSPECTION_WHEEL_NOT_REGULAR"):
        inspector.inspect_original(link)


def test_cli_reports_refusal_without_writing_evidence(
    inert_wheel: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    output = tmp_path / "evidence.json"
    with zipfile.ZipFile(inert_wheel, "a") as archive:
        archive.comment = b"changed"
    monkeypatch.setattr(
        "sys.argv", ["inspect", "--wheel", str(inert_wheel), "--output", str(output)]
    )
    assert inspector.main() == 2
    assert json.loads(capsys.readouterr().out)["reason"] == "INSPECTION_WHEEL_IDENTITY"
    assert not output.exists()


def test_cli_success_is_observation_without_approval(
    inert_wheel: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "evidence.json"
    monkeypatch.setattr(
        "sys.argv", ["inspect", "--wheel", str(inert_wheel), "--output", str(output)]
    )
    assert inspector.main() == 0
    assert json.loads(output.read_text())["approval"] == "NOT_GRANTED"


@pytest.mark.parametrize("target", ["input", "symlink"])
def test_evidence_output_cannot_overwrite_original_wheel(
    inert_wheel: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys, target: str
) -> None:
    output = inert_wheel
    if target == "symlink":
        output = tmp_path / "report.json"
        try:
            output.symlink_to(inert_wheel)
        except (OSError, NotImplementedError):
            pytest.skip("Host cannot create an unprivileged symlink")
    before = inert_wheel.read_bytes()
    monkeypatch.setattr(
        "sys.argv", ["inspect", "--wheel", str(inert_wheel), "--output", str(output)]
    )
    assert inspector.main() == 2
    assert json.loads(capsys.readouterr().out)["reason"] == "INSPECTION_OUTPUT_EXISTS"
    assert inert_wheel.read_bytes() == before

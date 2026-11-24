"""Diagnostics retain measured refusals and cannot create independent approvals."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import stage9_v5d6_inspect_wheelhouse as inspector
from tests.deployment.test_stage9_v5d4_dependencies import wheel


def test_partial_wheelhouse_never_claims_69_or_grants_approval(tmp_path: Path) -> None:
    directory = tmp_path / "wheels"
    directory.mkdir()
    path, origin = wheel(directory, "ordinary")
    original = path.read_bytes()
    report = inspector.inspect(tmp_path / "unused.toml", directory, {path.name: origin})
    assert report["status"] == "REFUSED"
    assert report["counts"]["WHEELS_CHECKED"] == 1
    assert report["refusals"] == [{"reason": "WHEELHOUSE_EXPECTED_COUNTS_MISMATCH"}]
    assert report["independent_approvals"] == "NOT_SUPPLIED_NOT_GENERATED"
    assert report["authorization"] == "NOT_AUTHORIZED"
    assert path.read_bytes() == original


def test_unsupported_category_preserves_next_refusal_and_archive_identity(tmp_path: Path) -> None:
    path, origin = wheel(tmp_path, "ordinary", files={"ordinary-1.0.data/data/new": b"x"})
    report = inspector.inspect(tmp_path / "unused.toml", tmp_path, {path.name: origin})
    assert report["counts"]["WHEELS_CHECKED"] == 1
    assert report["archives"][0]["sha256"] == origin["sha256"]
    assert report["archives"][0]["reason"] == "UNSUPPORTED_WHEEL_DATA"
    assert report["refusals"][0] == {"filename": path.name, "reason": "UNSUPPORTED_WHEEL_DATA"}


def test_missing_origin_does_not_invent_provenance(tmp_path: Path) -> None:
    path, _ = wheel(tmp_path, "ordinary")
    report = inspector.inspect(tmp_path / "unused.toml", tmp_path, {})
    assert report["refusals"][0] == {"filename": path.name, "reason": "ORIGIN_MISSING"}


def test_symlink_archive_is_refused_before_reading(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    path, origin = wheel(source, "ordinary")
    directory = tmp_path / "wheels"
    directory.mkdir()
    try:
        (directory / path.name).symlink_to(path)
    except (OSError, NotImplementedError):
        pytest.skip("Host cannot create an unprivileged symlink")
    report = inspector.inspect(tmp_path / "unused.toml", directory, {path.name: origin})
    assert report["status"] == "REFUSED"
    assert report["counts"]["WHEELS_CHECKED"] == 0
    assert report["refusals"][0]["filename"] == path.name


def test_complete_synthetic_diagnostic_checks_closure_without_creating_approvals(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    directory = tmp_path / "wheels"
    directory.mkdir()
    path, origin = wheel(directory, "ordinary")
    project = tmp_path / "pyproject.toml"
    project.write_text(
        '[project]\nrequires-python=">=3.11"\ndependencies=["ordinary==1.0"]\n'
        "[project.optional-dependencies]\nobserver-kvm-preflight=[]\n"
    )
    monkeypatch.setattr(
        inspector,
        "EXPECTED",
        {
            **inspector.EXPECTED,
            "WHEELS_CHECKED": 1,
            "EXPECTED_SHARED_PATHS": 0,
            "EXPECTED_FOREIGN_PYC_OMISSIONS": 0,
            "EXPECTED_HEADER_OMISSIONS": 0,
        },
    )
    report = inspector.inspect(project, directory, {path.name: origin})
    assert report["refusals"] == []
    assert report["status"] == "STATIC_CANDIDATES_INSPECTED_REQUIRES_INDEPENDENT_APPROVAL"
    assert len(report["proposal_sha256"]) == 64
    assert report["authorization"] == "NOT_AUTHORIZED"


def test_cli_keeps_invalid_origin_failure_as_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    origins = tmp_path / "origins.json"
    origins.write_text("[]")
    monkeypatch.setattr(
        "sys.argv",
        [
            "inspect",
            "--pyproject",
            str(tmp_path / "project.toml"),
            "--wheelhouse",
            str(tmp_path),
            "--origins",
            str(origins),
        ],
    )
    assert inspector.main() == 2
    assert json.loads(capsys.readouterr().out)["reason"] == "ORIGIN_SCHEMA"

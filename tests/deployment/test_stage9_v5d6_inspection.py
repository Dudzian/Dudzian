"""Diagnostics retain measured refusals and cannot create independent approvals."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from scripts import (
    stage9_v5d4_dependencies as deps,
    stage9_v5d5_shared as shared,
    stage9_v5d6_headers as headers,
    stage9_v5d6_inspect_wheelhouse as inspector,
)
from tests.deployment.test_stage9_v5d4_dependencies import cache_material, pin_test_cache, wheel
from tests.deployment.test_stage9_v5d5_shared import shared_material as shared_material
from tests.deployment.test_stage9_v5d6_headers import (
    HEADER_CONTENT,
    header_wheel,
    pin_test_header,
    rewrite,
)


def test_partial_wheelhouse_never_claims_69_or_grants_approval(tmp_path: Path) -> None:
    directory = tmp_path / "wheels"
    directory.mkdir()
    path, origin = wheel(directory, "ordinary")
    original = path.read_bytes()
    report = inspector.inspect(tmp_path / "unused.toml", directory, {path.name: origin})
    assert report["status"] == "REFUSED"
    assert report["expected_counts"]["WHEELS_CHECKED"] == 69
    assert report["measurements"]["WHEELS_CHECKED"] == {
        "status": "PARTIAL",
        "value": 1,
        "wheels_evaluated": 1,
    }
    assert report["measurements"]["SHARED_PATHS"]["status"] == "PARTIAL"
    assert report["closure_status"] == "NOT_TESTED"
    assert report["refusals"] == [{"reason": "WHEELHOUSE_EXPECTED_COUNTS_MISMATCH"}]
    assert report["independent_approvals"] == "NOT_SUPPLIED_NOT_GENERATED"
    assert report["authorization"] == "NOT_AUTHORIZED"
    assert path.read_bytes() == original


def test_unsupported_category_preserves_next_refusal_and_archive_identity(tmp_path: Path) -> None:
    path, origin = wheel(tmp_path, "ordinary", files={"ordinary-1.0.data/data/new": b"x"})
    report = inspector.inspect(tmp_path / "unused.toml", tmp_path, {path.name: origin})
    assert report["measurements"]["WHEELS_CHECKED"]["value"] == 1
    assert report["measurements"]["SHARED_PATHS"]["status"] == "NOT_TESTED"
    assert report["measurements"]["SHARED_PATHS"]["value"] is None
    assert report["archives"][0]["sha256"] == origin["sha256"]
    assert report["archives"][0]["reason"] == "UNSUPPORTED_WHEEL_DATA"
    assert report["refusals"][0] == {"filename": path.name, "reason": "UNSUPPORTED_WHEEL_DATA"}


def test_missing_origin_does_not_invent_provenance(tmp_path: Path) -> None:
    path, _ = wheel(tmp_path, "ordinary")
    report = inspector.inspect(tmp_path / "unused.toml", tmp_path, {})
    assert report["refusals"][0] == {"filename": path.name, "reason": "ORIGIN_MISSING"}


def test_malformed_csv_retains_previous_archive_observations(tmp_path: Path) -> None:
    accepted, accepted_origin = wheel(tmp_path, "aaa")
    rejected, rejected_origin = wheel(tmp_path, "zzz")
    rewrite(rejected, updates={"zzz-1.0.dist-info/RECORD": b"x" * (csv.field_size_limit() + 1)})
    rejected_origin["sha256"] = deps.digest(rejected.read_bytes())
    report = inspector.inspect(
        tmp_path / "unused.toml",
        tmp_path,
        {accepted.name: accepted_origin, rejected.name: rejected_origin},
    )
    assert report["status"] == "REFUSED"
    assert report["archives"][0]["status"] == "STATIC_CANDIDATE_INSPECTED"
    assert report["archives"][1]["reason"].startswith("field larger than field limit")
    assert report["archive_scan"]["bytes_read"] == accepted.stat().st_size + rejected.stat().st_size
    assert report["archive_scan"]["wheels_fully_inspected"] == 1
    assert report["measurements"]["HEADER_OMISSIONS"] == {
        "status": "PARTIAL",
        "value": 0,
        "wheels_evaluated": 1,
    }


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
    for result in report["measurements"].values():
        assert result == {"status": "NOT_TESTED", "value": None, "wheels_evaluated": 0}
    assert report["archive_scan"]["wheels_read"] == 0
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
    assert report["closure_status"] == "MEASURED"
    assert all(result["status"] == "MEASURED" for result in report["measurements"].values())
    assert len(report["proposal_sha256"]) == 64
    assert report["authorization"] == "NOT_AUTHORIZED"

    project.write_text(project.read_text().replace("ordinary==1.0", "missing==1.0"))
    refused = inspector.inspect(project, directory, {path.name: origin})
    assert refused["status"] == "REFUSED"
    assert refused["closure_status"] == "REFUSED"
    assert refused["measurements"] == report["measurements"]
    assert refused["refusals"] == [{"reason": "MISSING_DEPENDENCY:missing"}]
    assert "proposal_sha256" not in refused


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
    report = json.loads(capsys.readouterr().out)
    assert report["reason"] == "ORIGIN_SCHEMA"
    assert all(item["status"] == "NOT_TESTED" for item in report["measurements"].values())


def test_observation_does_not_change_successful_inspection_value(tmp_path: Path) -> None:
    path, origin = wheel(tmp_path, "ordinary")
    observations = {}
    with_observation = deps.inspect_wheel_bytes(
        path.read_bytes(), path.name, origin, observations=observations
    )
    assert with_observation == deps.inspect_wheel(path, origin)
    assert (
        observations["materialization_input"]["materialized_members"]
        == with_observation["materialized_members"]
    )


def test_cmd_refusal_retains_header_candidate_and_reports_runtime_alias(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path, origin = header_wheel(
        tmp_path,
        "greenlet",
        "3.5.6",
        {
            headers.HEADER["member"]: HEADER_CONTENT,
            "greenlet/greenlet.h": HEADER_CONTENT,
            "greenlet/platform/setup_switch_x64_masm.cmd": b"REM TEST ONLY\r\n",
        },
    )
    pin_test_header(monkeypatch, path, HEADER_CONTENT)
    monkeypatch.setattr(inspector, "dependency_tools", lambda: vars(deps))
    with pytest.raises(deps.Refusal, match="^WHEEL_EXECUTABLE_FORMAT_UNSUPPORTED$"):
        deps.inspect_wheel(path, origin)
    report = inspector.inspect(tmp_path / "unused.toml", tmp_path, {path.name: origin})
    assert report["status"] == "REFUSED"
    assert report["archive_scan"]["wheels_fully_inspected"] == 0
    assert report["refusals"][0] == {
        "filename": path.name,
        "reason": "WHEEL_EXECUTABLE_FORMAT_UNSUPPORTED",
    }
    assert {
        "check": "header_inventory",
        "reason": "HEADER_RUNTIME_INCLUSION_FORBIDDEN",
    } in report["refusals"]
    assert report["archives"][0]["observations"]["header_omissions"] == [headers.pinned_row()]
    assert report["measurements"]["HEADER_OMISSIONS"] == {
        "status": "PARTIAL",
        "value": 1,
        "wheels_evaluated": 1,
    }
    assert report["header_omissions"] is None
    assert "proposal_sha256" not in report


def test_header_observation_requires_complete_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path, origin = header_wheel(
        tmp_path, "greenlet", "3.5.6", {headers.HEADER["member"]: HEADER_CONTENT}
    )
    rewrite(path, bad_record_sha=headers.HEADER["member"])
    origin["sha256"] = deps.digest(path.read_bytes())
    pin_test_header(monkeypatch, path, HEADER_CONTENT)
    monkeypatch.setattr(inspector, "dependency_tools", lambda: vars(deps))
    report = inspector.inspect(tmp_path / "unused.toml", tmp_path, {path.name: origin})
    assert report["refusals"][0]["reason"] == "WHEEL_RECORD_INTEGRITY"
    assert report["archives"][0]["observations"] == {}
    assert report["measurements"]["HEADER_OMISSIONS"]["status"] == "NOT_TESTED"
    assert report["measurements"]["HEADER_OMISSIONS"]["value"] is None


def test_refused_member_does_not_hide_two_validated_pyc_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    files = cache_material()
    files["pyside6_essentials/rejected.sh"] = b"# TEST ONLY no code\n"
    path, origin = wheel(
        tmp_path,
        "pyside6_essentials",
        "6.10.3",
        tag="cp39-abi3-manylinux_2_34_x86_64",
        files=files,
    )
    pin_test_cache(monkeypatch, path, files)
    monkeypatch.setattr(inspector, "dependency_tools", lambda: vars(deps))
    report = inspector.inspect(tmp_path / "unused.toml", tmp_path, {path.name: origin})
    assert report["refusals"][0]["reason"] == "WHEEL_EXECUTABLE_FORMAT_UNSUPPORTED"
    assert report["measurements"]["FOREIGN_PYC_OMISSIONS"] == {
        "status": "PARTIAL",
        "value": 2,
        "wheels_evaluated": 1,
    }


@pytest.mark.parametrize("unknown_owner", [False, True])
def test_refusal_elsewhere_does_not_discard_all_62_sharing_candidates(
    shared_material, monkeypatch: pytest.MonkeyPatch, unknown_owner: bool
) -> None:
    project, directory, origins, proposal, _ = shared_material
    files = (
        {next(iter(shared.MEMBERS)): b"TEST ONLY unapproved owner"}
        if unknown_owner
        else {"third-1.0.data/data/unsupported": b"TEST ONLY"}
    )
    path, origin = wheel(directory, "third", files=files)
    monkeypatch.setattr(inspector, "dependency_tools", lambda: vars(deps))
    report = inspector.inspect(project, directory, {**origins, path.name: origin})
    assert report["status"] == "REFUSED"
    assert report["closure_status"] == "NOT_TESTED"
    assert report["shared_wheel_members"] == proposal["shared_wheel_members"]
    assert report["measurements"]["SHARED_PATHS"] == {
        "status": "PARTIAL",
        "value": 62,
        "wheels_evaluated": 2,
    }
    if unknown_owner:
        assert report["shared_wheel_members_scope"] == "PINNED_WHEEL_PAIR"
        assert report["refusals"][0]["reason"].startswith("SHARED_WHEEL_OWNER_MISMATCH:")
    else:
        assert report["shared_wheel_members_scope"] == "AVAILABLE_MATERIALIZATION_MAPS"
        assert report["refusals"][0]["reason"] == "UNSUPPORTED_WHEEL_DATA"


def test_collision_without_measured_pinned_pair_does_not_publish_zero(tmp_path: Path) -> None:
    first, first_origin = wheel(tmp_path, "first", files={"collision.txt": b"first"})
    second, second_origin = wheel(tmp_path, "second", files={"collision.txt": b"second"})
    report = inspector.inspect(
        tmp_path / "unused.toml", tmp_path, {first.name: first_origin, second.name: second_origin}
    )
    assert report["status"] == "REFUSED"
    assert report["refusals"] == [
        {"check": "shared_inventory", "reason": "SHARED_WHEEL_MEMBER_SET_MISMATCH"}
    ]
    assert report["shared_wheel_members"] is None
    assert report["measurements"]["SHARED_PATHS"] == {
        "status": "NOT_TESTED",
        "value": None,
        "wheels_evaluated": 0,
    }

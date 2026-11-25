"""V5d7 collects independent refusals without using diagnostic data for enforcement."""

from __future__ import annotations

from pathlib import Path
import zipfile

import pytest

from scripts import stage9_v5d4_dependencies as deps
from scripts import stage9_v5d7_inspect_wheelhouse as inspector
from scripts import stage9_v5d7_omissions as omissions
from scripts import stage9_v5d7_shared as shared
from tests.deployment.test_stage9_v5d4_dependencies import wheel
from tests.deployment.test_stage9_v5d6_headers import rewrite
from tests.deployment.test_stage9_v5d7_omissions import scipy_candidate as scipy_candidate
from tests.deployment.test_stage9_v5d7_sharing import TARGET, three_wheels as three_wheels

pytestmark = [pytest.mark.security, pytest.mark.unit]


def test_refusals_in_same_wheel_do_not_stop_inspection_or_closure(tmp_path: Path) -> None:
    directory = tmp_path / "wheels"
    directory.mkdir()
    project = tmp_path / "pyproject.toml"
    project.write_text(
        '[project]\nrequires-python=">=3.11"\ndependencies=["sample==1.0"]\n'
        "[project.optional-dependencies]\nobserver-kvm-preflight=[]\n",
        encoding="utf-8",
    )
    unsafe = {
        "sample/one.cmd": b"REM TEST ONLY\r\n",
        "sample/two.pyx": b"#!synthetic\n",
        "sample/_three.cpython-311-x86_64-linux-gnu.so": b"\x7fELF synthetic",
    }
    path, origin = wheel(directory, "sample", files=unsafe)
    bindings = {path.name: {"sha256": origin["sha256"], "size": path.stat().st_size}}
    report = inspector.inspect(project, directory, {path.name: origin}, expected_archives=bindings)
    assert report["status"] == "REFUSED"
    assert report["closure_status"] == "MEASURED"
    assert report["authorization"] == "NOT_AUTHORIZED"
    assert "proposal_sha256" not in report
    assert {row["member"]: row["reason"] for row in report["refusals"]} == {
        "sample/one.cmd": "WHEEL_EXECUTABLE_FORMAT_UNSUPPORTED",
        "sample/two.pyx": "WHEEL_EXECUTABLE_FORMAT_UNSUPPORTED",
        "sample/_three.cpython-311-x86_64-linux-gnu.so": "NATIVE_MEMBER_SOABI_MISMATCH",
    }
    for entry in report["refusals"]:
        assert entry["sha256"] == deps.digest(unsafe[entry["member"]])
        assert entry["wheel_sha256"] == origin["sha256"]
        assert entry["size"] == len(unsafe[entry["member"]])
        assert entry["mode"] == 0o100644
    assert (
        next(row for row in report["refusals"] if row["member"].endswith(".pyx"))[
            "format_prefix_hex"
        ]
        == "23217379"
    )
    with pytest.raises(deps.Refusal):
        deps.inspect_wheel_bytes(path.read_bytes(), path.name, origin, v5d7=True)


def test_unsupported_metadata_retains_record_and_member_findings(tmp_path: Path) -> None:
    path, origin = wheel(
        tmp_path,
        "sample",
        files={
            "sample-1.0.dist-info/METADATA": b"Metadata-Version: 3.0\nName: sample\nVersion: 1.0\n\n",
            "sample/foreign.cmd": b"REM TEST ONLY",
        },
    )
    observed = []

    def observe(member: str, content: bytes, mode: int, target: str | None) -> None:
        observed.append((member, content, mode, target))

    result = inspector.inspect_archive(path.read_bytes(), path.name, origin, on_member=observe)
    assert result["metadata_status"] == "REFUSED"
    assert result["record_status"] == "MEASURED"
    assert result["materialization_map_status"] == "PARTIAL"
    assert result["metadata"]["headers"][0] == ["Metadata-Version", "3.0"]
    assert result["dependency_candidate"] is None
    assert {entry["reason"] for entry in result["refusals"]} == {
        "WHEEL_METADATA_VERSION_UNSUPPORTED",
        "WHEEL_EXECUTABLE_FORMAT_UNSUPPORTED",
    }
    assert ("sample/foreign.cmd", b"REM TEST ONLY", 0o100644, "sample/foreign.cmd") in observed
    assert result["record_entries_verified"] == len(observed)


def test_every_independent_record_error_refuses_before_callbacks(tmp_path: Path) -> None:
    files = {"sample/first.py": b"# original one", "sample/second.py": b"# original two"}
    path, origin = wheel(tmp_path, "sample", files=files)
    changed = dict.fromkeys(files, b"# changed without updating RECORD")
    rewrite(path, updates=changed)
    data = path.read_bytes()
    origin["sha256"] = deps.digest(data)

    def observe(*_args: object) -> None:
        pytest.fail("received data from a failed RECORD")

    result = inspector.inspect_archive(data, path.name, origin, on_member=observe)
    assert result["record_status"] == "REFUSED"
    assert result["metadata_status"] == "NOT_TESTED"
    assert result["members"] == []
    assert result["record_entries_verified"] == 4
    assert {
        (entry["member"], entry["reason"], entry["sha256"]) for entry in result["refusals"]
    } == {
        (member, "WHEEL_RECORD_INTEGRITY", deps.digest(content))
        for member, content in changed.items()
    }


def test_all_unsafe_paths_are_retained_without_opening_members(tmp_path: Path) -> None:
    path, origin = wheel(tmp_path, "sample", files={"../outside": b"x", "bad//path": b"y"})

    def observe(*_args: object) -> None:
        pytest.fail("received data after an unsafe path")

    result = inspector.inspect_archive(path.read_bytes(), path.name, origin, on_member=observe)
    assert result["record_status"] == "NOT_TESTED"
    assert {entry["member"] for entry in result["refusals"]} == {"../outside", "bad//path"}
    assert {entry["reason"] for entry in result["refusals"]} == {"WHEEL_MEMBER_PATH"}


def test_pin_mismatch_refuses_before_member_callback(tmp_path: Path) -> None:
    path, origin = wheel(tmp_path, "sample")

    def observe(*_args: object) -> None:
        pytest.fail("received an archive that differs from the evidence")

    report = inspector.inspect(
        tmp_path / "unused.toml",
        tmp_path,
        {path.name: origin},
        on_member=observe,
        expected_archives={path.name: {"sha256": "0" * 64, "size": path.stat().st_size}},
    )
    assert report["status"] == "REFUSED"
    assert report["closure_status"] == "NOT_TESTED"
    assert report["archives"][0]["record_status"] == "NOT_TESTED"
    assert report["archives"][0]["refusals"][0]["reason"] == "EVIDENCE_ARCHIVE_IDENTITY"


def test_coff_member_refuses_without_exact_omission(tmp_path: Path) -> None:
    coff = bytes.fromhex("64860100") + bytes(56)
    path, origin = wheel(
        tmp_path, "sample", files={"sample/foreign.dat": coff, "sample/source.obj": b"text"}
    )
    result = inspector.inspect_archive(path.read_bytes(), path.name, origin)
    assert result["record_status"] == "MEASURED"
    assert result["legacy_refusals"] == []
    assert {entry["member"] for entry in result["refusals"]} == {
        "sample/foreign.dat",
        "sample/source.obj",
    }
    assert {entry["check"] for entry in result["refusals"]} == {"coff"}


def test_collects_internal_alias_and_cross_archive_file_directory_conflict(tmp_path: Path) -> None:
    first, first_origin = wheel(
        tmp_path,
        "first",
        files={
            "parent": b"x",
            "same.txt": b"x",
            "first-1.0.data/purelib/same.txt": b"x",
        },
    )
    second, second_origin = wheel(
        tmp_path, "second", files={"parent/child.txt": b"y", "same.txt": b"x"}
    )
    report = inspector.inspect(
        tmp_path / "unused.toml", tmp_path, {first.name: first_origin, second.name: second_origin}
    )
    violations = {row.get("path"): row for row in report["refusals"]}
    assert {(row["path"], row["reason"]) for row in report["refusals"] if "path" in row} == {
        ("same.txt", "WHEEL_MATERIALIZATION_COLLISION"),
        ("same.txt", "SHARED_WHEEL_MEMBER_SET_MISMATCH"),
        ("parent", "WHEEL_MATERIALIZATION_FILE_DIRECTORY_COLLISION"),
    }
    assert violations["parent"]["owners"][0]["sha256"] == deps.digest(b"x")
    assert report["shared_wheel_members"][0]["path"] == "same.txt"
    assert len(report["shared_wheel_members"][0]["owners"]) == 3
    assert report["measurements"]["SHARED_PATHS"]["status"] == "PARTIAL"


def test_cli_refuses_duplicate_origins_keys(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    origins = tmp_path / "origins.json"
    origins.write_text('{"duplicate":0,"duplicate":1}', encoding="utf-8")
    monkeypatch.setattr(
        "sys.argv",
        [
            "inspect",
            "--pyproject",
            str(tmp_path / "pyproject.toml"),
            "--wheelhouse",
            str(tmp_path),
            "--origins",
            str(origins),
        ],
    )
    assert inspector.main() == 2
    assert "DUPLICATE_JSON_KEY" in capsys.readouterr().out


@pytest.mark.parametrize("alias_kind", ["source", "projected", "descendant"])
def test_omission_diagnostics_find_all_cross_wheel_aliases(
    monkeypatch: pytest.MonkeyPatch, scipy_candidate, three_wheels, alias_kind: str
) -> None:
    member, source, origin, _ = scipy_candidate
    project, directory, origins = three_wheels
    (directory / source.name).write_bytes(source.read_bytes())
    origins[source.name] = origin
    project.write_text(
        project.read_text(encoding="utf-8").replace(
            "dependencies=[", 'dependencies=["scipy==1.18.1","other==1.0",'
        ),
        encoding="utf-8",
    )
    with zipfile.ZipFile(source) as archive:
        omitted_content = archive.read(member)
    alias = {
        "source": member,
        "projected": "other-1.0.data/purelib/" + member,
        "descendant": "other-1.0.data/purelib/" + member + "/extra.txt",
    }[alias_kind]
    copied = "other/copy.py"
    path, other_origin = wheel(
        directory, "other", files={alias: b"TEST ONLY no matching bytes", copied: omitted_content}
    )
    origins[path.name] = other_origin
    bindings = {
        filename: {"sha256": row["sha256"], "size": (directory / filename).stat().st_size}
        for filename, row in origins.items()
    }
    monkeypatch.setattr(inspector, "dependency_tools", lambda: vars(deps))
    with pytest.raises(ValueError, match="^ARTIFACT_OMISSION_RUNTIME_INCLUSION_FORBIDDEN$"):
        deps.propose(project, directory, origins, v5d7=True)
    report = inspector.inspect(project, directory, origins, expected_archives=bindings)
    assert report["measurements"]["ARTIFACT_OMISSIONS"]["status"] == "MEASURED"
    assert report["manifest_checks"]["artifact_omissions"] == {
        "coverage": "MEASURED",
        "status": "REFUSED",
        "candidates": 1,
    }
    assert report["manifest_checks"]["sharing"]["status"] == "MEASURED"
    findings = {row["member"]: row for row in report["refusals"]}
    assert set(findings) == {alias, copied}
    assert all(row["filename"] == path.name for row in findings.values())
    assert all(row["wheel_sha256"] == other_origin["sha256"] for row in findings.values())
    assert {row["reason"] for row in findings.values()} == {
        "ARTIFACT_OMISSION_RUNTIME_INCLUSION_FORBIDDEN"
    }
    assert findings[copied]["sha256"] == deps.digest(omitted_content)
    expected_target = member + "/extra.txt" if alias_kind == "descendant" else member
    assert findings[alias]["projected_path"] == expected_target


def test_all_pinned_omission_failures_retain_independent_member_evidence(
    monkeypatch: pytest.MonkeyPatch, scipy_candidate
) -> None:
    member, path, origin, _ = scipy_candidate
    pin = omissions.MEMBERS[origin["sha256"]]
    digest, size, kind = pin[member]
    missing = "scipy/missing.pyx"
    pin[member] = ("0" * 64, size + 1, kind)
    pin[missing] = (digest, size, kind)
    monkeypatch.setattr(inspector, "dependency_tools", lambda: vars(deps))
    result = inspector.inspect_archive(path.read_bytes(), path.name, origin)
    assert result["artifact_omissions"] is None
    assert result["record_status"] == "MEASURED"
    assert {row["member"]: row["status"] for row in result["artifact_candidate_checks"]} == {
        member: "REFUSED",
        missing: "REFUSED",
    }
    assert {
        (row["member"], row["reason"])
        for row in result["refusals"]
        if row["check"] == "artifact_omissions"
    } == {
        (member, "ARTIFACT_OMISSION_MEMBER_METADATA_MISMATCH:" + member),
        (member, "ARTIFACT_OMISSION_MEMBER_HASH_MISMATCH:" + member),
        (missing, "ARTIFACT_OMISSION_SET_MISMATCH"),
    }


def test_same_filename_without_matching_omission_hash_uses_ordinary_policy(
    monkeypatch: pytest.MonkeyPatch, scipy_candidate
) -> None:
    member, path, origin, _ = scipy_candidate
    rewrite(path, removes={member}, reseal_record=True)
    data = path.read_bytes()
    origin["sha256"] = deps.digest(data)
    monkeypatch.setattr(inspector, "dependency_tools", lambda: vars(deps))
    assert deps.inspect_wheel(path, origin, v5d7=True)["artifact_omissions"] == []
    result = inspector.inspect_archive(data, path.name, origin)
    assert result["artifact_omissions"] == []
    assert result["artifact_candidate_checks"] == []
    assert result["refusals"] == []


def test_bound_sharing_reports_all_absent_archives_without_inventing_measurements(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, three_wheels
) -> None:
    project, directory, origins = three_wheels
    bindings = {
        filename: {"sha256": row["sha256"], "size": (directory / filename).stat().st_size}
        for filename, row in origins.items()
    }
    empty = tmp_path / "empty"
    empty.mkdir()
    monkeypatch.setattr(inspector, "dependency_tools", lambda: vars(deps))
    report = inspector.inspect(project, empty, {}, expected_archives=bindings)
    assert report["manifest_checks"]["sharing"]["coverage"] == "NOT_TESTED"
    assert report["manifest_checks"]["sharing"]["status"] == "REFUSED"
    refusal = next(
        row for row in report["refusals"] if row["reason"] == "SHARED_WHEEL_OWNER_SET_MISMATCH"
    )
    assert {row["filename"] for row in refusal["missing_owners"]} == bindings.keys()


@pytest.mark.parametrize(
    "removed,reason",
    [
        (1, "SHARED_WHEEL_OWNER_MISMATCH"),
        (2, "SHARED_WHEEL_MEMBER_SET_MISMATCH"),
        (3, "SHARED_WHEEL_MEMBER_SET_MISMATCH"),
    ],
)
def test_complete_sharing_measurement_exposes_missing_paths_and_owners(
    monkeypatch: pytest.MonkeyPatch, three_wheels, removed: int, reason: str
) -> None:
    project, directory, origins = three_wheels
    for _, identity in list(shared.WHEELS.items())[:removed]:
        path = directory / identity["wheel_filename"]
        rewrite(path, removes={TARGET}, reseal_record=True)
        sha256 = deps.digest(path.read_bytes())
        identity["wheel_sha256"] = sha256
        origins[path.name]["sha256"] = sha256
    bindings = {
        filename: {"sha256": row["sha256"], "size": (directory / filename).stat().st_size}
        for filename, row in origins.items()
    }
    monkeypatch.setattr(inspector, "dependency_tools", lambda: vars(deps))
    with pytest.raises(ValueError, match="^" + reason):
        deps.propose(project, directory, origins, v5d7=True)
    report = inspector.inspect(project, directory, origins, expected_archives=bindings)
    assert report["manifest_checks"]["sharing"] == {
        "coverage": "MEASURED",
        "status": "REFUSED",
        "policy": shared.POLICY_ID,
        "expected_paths": 62,
        "expected_owners_per_path": 3,
    }
    assert report["measurements"]["SHARED_PATHS"]["status"] == "MEASURED"
    findings = [row for row in report["refusals"] if row.get("path") == TARGET]
    assert any(row["reason"] == reason for row in findings)
    mismatch = next(row for row in findings if row["reason"] == "SHARED_WHEEL_OWNER_MISMATCH")
    assert len(mismatch["owners"]) == 3 - removed
    assert len(mismatch["expected_owners"]) == 3
    assert {row["path"] for row in report["refusals"]} == {TARGET}

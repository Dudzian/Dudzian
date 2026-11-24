"""V5d7 collects independent refusals without using diagnostic data for enforcement."""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts import stage9_v5d4_dependencies as deps
from scripts import stage9_v5d7_inspect_wheelhouse as inspector
from tests.deployment.test_stage9_v5d4_dependencies import wheel
from tests.deployment.test_stage9_v5d6_headers import rewrite

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

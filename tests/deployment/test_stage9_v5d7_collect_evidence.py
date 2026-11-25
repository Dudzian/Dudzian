from __future__ import annotations

import hashlib
import io
import json
import sys
import tarfile
from pathlib import Path
from typing import Any

import pytest

from scripts import stage9_v5d7_collect_evidence as collector


def write_reference(official: dict, inventory: dict) -> bytes:
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w:gz") as archive:
        for name, report in (
            ("v5d6-official.json", official),
            ("member-inventory.json", inventory),
        ):
            raw = json.dumps(report).encode()
            entry = tarfile.TarInfo(name)
            entry.size = len(raw)
            archive.addfile(entry, io.BytesIO(raw))
    return stream.getvalue()


def test_reference_binds_both_inventories(monkeypatch: pytest.MonkeyPatch) -> None:
    deps = collector.tools("stage9_v5d4_dependencies.py")
    pin = {"sha256": "a" * 64, "size": 23}
    official = {"archives": [{"filename": "test.whl", **pin}]}
    inventory: dict[str, Any] = {
        "archives": [{"wheel": "test.whl", "origin_sha_matches": True, **pin}],
        "cross_wheel_collisions": [],
    }
    raw = write_reference(official, inventory)
    with pytest.raises(ValueError, match="V5D6_REFERENCE_SHA256"):
        collector.read_reference(raw, deps)
    monkeypatch.setattr(collector, "REFERENCE_SHA256", hashlib.sha256(raw).hexdigest())
    assert collector.read_reference(raw, deps) == ({"test.whl": pin}, [])
    inventory["archives"][0]["size"] = 17
    raw = write_reference(official, inventory)
    monkeypatch.setattr(collector, "REFERENCE_SHA256", hashlib.sha256(raw).hexdigest())
    with pytest.raises(ValueError, match="V5D6_REFERENCE_ARCHIVES_DISAGREE"):
        collector.read_reference(raw, deps)


@pytest.mark.parametrize(
    ("record", "callback_refusal", "native_status", "focus_status", "measured_native"),
    [
        ("MEASURED", False, "MEASURED", "MEASURED", "MEASURED"),
        ("MEASURED", False, "PARTIAL", "MEASURED", "PARTIAL"),
        ("MEASURED", True, "MEASURED", "PARTIAL", "PARTIAL"),
        ("NOT_TESTED", False, "MEASURED", "NOT_TESTED", "NOT_TESTED"),
    ],
)
def test_collect_qualifies_partial_results(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    record: str,
    callback_refusal: bool,
    native_status: str,
    focus_status: str,
    measured_native: str,
) -> None:
    load = collector.tools
    deps = load("stage9_v5d4_dependencies.py")
    wheel = next(iter(load("stage9_v5d7_focused.py")["METADATA_WHEELS"]))
    expected = {wheel: {"sha256": "a" * 64, "size": 23}}
    monkeypatch.setattr(collector, "read_reference", lambda *_: (expected, []))
    calls: list[str] = []

    def inspect(pyproject, wheelhouse, origins, *, on_member, expected_archives):
        calls.append("diagnostic")
        assert pyproject == wheelhouse == tmp_path
        assert origins == {} and expected_archives == expected
        if record == "MEASURED":
            on_member(
                wheel, "example.dist-info/METADATA", b"Metadata-Version: 2.5\n\n", 0o100644, None
            )
            on_member(wheel, "PySide6/__init__.py", b"# source", 0o100644, "PySide6/__init__.py")
            on_member(wheel, "module.so", b"same", 0o100755, "module.so")
            on_member(wheel, "other.so", b"same", 0o100755, "other.so")
        return {
            "status": "MEASURED",
            "archives": [
                {
                    "filename": wheel,
                    "sha256": "a" * 64,
                    "record_status": record,
                    "member_scan_status": record,
                }
            ],
            "refusals": [{"check": "member_reader"}] if callback_refusal else [],
        }

    def propose(pyproject, wheelhouse, origins, *, v5d7):
        calls.append("propose")
        assert v5d7 and pyproject == wheelhouse == tmp_path and origins == {}
        raise deps["Refusal"]("ARTIFACT_OMISSION_RUNTIME_INCLUSION_FORBIDDEN")

    deps["propose"] = propose

    def analyze_native(_data, member, _mode):
        if member.endswith(".so"):
            return {"format": "ELF", "status": native_status, "details": {"elf": "measurement"}}
        return None

    def tools(name):
        if name.endswith("_inspect_wheelhouse.py"):
            return {
                "dependency_tools": lambda: deps,
                "inspect": inspect,
                "STATIC_ERRORS": (ValueError,),
            }
        if name.endswith("_native.py"):
            return {
                "native_tool_versions": dict,
                "analyze_native": analyze_native,
                "summarize_dependencies": lambda *_: {
                    "bundled_candidates_by_needed_name": {},
                    "unmatched_needed_names": ["libexample.so"],
                    "program_interpreters": ["/lib/example-loader"],
                },
                "measure_host_loader_cache": lambda names: {"requested": sorted(names)},
                "HOST_ABI_LIBRARY_NAMES": {"libvirt.so.0"},
                "measure_host_native_files": lambda cache, loaders: {"loaders": loaders, **cache},
            }
        return load(name)

    monkeypatch.setattr(collector, "tools", tools)
    reference = tmp_path / "reference"
    reference.write_bytes(b"test")
    origins = tmp_path / "origins.json"
    origins.write_text("{}", encoding="utf-8")
    report = collector.collect(reference, tmp_path, tmp_path, origins, measure_host_abi=True)
    native = report["native_inventory"]
    assert calls == ["diagnostic", "propose"]
    assert report["focus"]["status"] == focus_status
    assert native["status"] == measured_native
    assert report["measurement_status"] == measured_native
    assert report["status"] == "REFUSED"
    assert report["proposal_attempt"]["reason"] == "ARTIFACT_OMISSION_RUNTIME_INCLUSION_FORBIDDEN"
    assert native["host_loader_cache"] == {"requested": ["libexample.so", "libvirt.so.0"]}
    assert native["host_abi_inputs"] == {
        **native["host_loader_cache"],
        "loaders": ["/lib/example-loader"],
    }
    assert report["authorization"] == "NOT_AUTHORIZED"
    if record == "MEASURED":
        assert report["focus"]["metadata"][wheel][0]["metadata_version"] == "2.5"
        assert native["counts_by_detected_format"] == {"ELF": 2}
        assert len(native["details"]) == 1
    else:
        assert native["members"] == []


@pytest.mark.parametrize("case", ["match", "mismatch", "existing", "no_output"])
def test_proposal_attempt_binds_and_writes_exact_canonical_bytes(tmp_path: Path, case: str) -> None:
    deps = collector.tools("stage9_v5d4_dependencies.py")
    expected = {"wheel.whl": {"sha256": "a" * 64}}
    proposal = {
        "profile": deps["V5D7_PROPOSAL"],
        "status": "PROPOSAL_REQUIRES_INDEPENDENT_APPROVAL",
        "packages": [
            {"filename": "wheel.whl", "sha256": ("b" if case == "mismatch" else "a") * 64}
        ],
    }
    deps["propose"] = lambda *_args, **_kwargs: proposal
    output = tmp_path / "proposal.json"
    if case == "existing":
        output.write_bytes(b"existing bytes")
    result = collector.attempt_proposal(
        deps,
        (ValueError,),
        tmp_path,
        tmp_path,
        {},
        expected,
        None if case == "no_output" else output,
    )
    canonical = deps["canonical"](proposal)
    assert result["status"] == proposal["status"]
    assert result["sha256"] == hashlib.sha256(canonical).hexdigest()
    assert result["size"] == len(canonical)
    if case == "match":
        assert result["reference_binding"] == "MEASURED"
        assert result["canonical_output"] == "WRITTEN"
        assert output.read_bytes() == canonical
    elif case == "mismatch":
        assert result["reference_reason"] == "PROPOSAL_V5D6_ARCHIVE_IDENTITY_MISMATCH"
        assert result["reference_binding"] == "REFUSED"
        assert result["canonical_output"] == "NOT_WRITTEN"
        assert not output.exists()
    elif case == "existing":
        assert result["canonical_output"] == "REFUSED"
        assert output.read_bytes() == b"existing bytes"
    else:
        assert result["canonical_output"] == "NOT_REQUESTED"
        assert not output.exists()


@pytest.mark.parametrize("failed", [True, False])
def test_command_writes_structured_status(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], failed: bool
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["collector", "--reference", "ref", "--wheelhouse", "wheels", "--origins", "origins"],
    )

    def collect(*_, **_kwargs):
        if failed:
            raise ValueError("V5D6_REFERENCE_SHA256")
        return {"status": "MEASURED"}

    monkeypatch.setattr(collector, "collect", collect)
    assert collector.main() == (2 if failed else 0)
    assert json.loads(capsys.readouterr().out)["status"] == ("REFUSED" if failed else "MEASURED")

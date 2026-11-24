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

    def inspect(pyproject, wheelhouse, origins, *, on_member, expected_archives):
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

    def analyze_native(_data, member, _mode):
        if member.endswith(".so"):
            return {"format": "ELF", "status": native_status, "details": {"elf": "measurement"}}
        return None

    def tools(name):
        if name.endswith("_inspect_wheelhouse.py"):
            return {"dependency_tools": lambda: deps, "inspect": inspect}
        if name.endswith("_native.py"):
            return {
                "native_tool_versions": dict,
                "analyze_native": analyze_native,
                "summarize_dependencies": lambda *_: {
                    "bundled_candidates_by_needed_name": {},
                    "unmatched_needed_names": ["libexample.so"],
                },
                "measure_host_loader_cache": lambda names: {"requested": sorted(names)},
            }
        return load(name)

    monkeypatch.setattr(collector, "tools", tools)
    reference = tmp_path / "reference"
    reference.write_bytes(b"test")
    origins = tmp_path / "origins.json"
    origins.write_text("{}", encoding="utf-8")
    report = collector.collect(reference, tmp_path, tmp_path, origins)
    native = report["native_inventory"]
    assert report["focus"]["status"] == focus_status
    assert native["status"] == measured_native
    assert native["host_loader_cache"] == {"requested": ["libexample.so"]}
    assert report["authorization"] == "NOT_AUTHORIZED"
    if record == "MEASURED":
        assert report["focus"]["metadata"][wheel][0]["metadata_version"] == "2.5"
        assert native["counts_by_detected_format"] == {"ELF": 2}
        assert len(native["details"]) == 1
    else:
        assert native["members"] == []


@pytest.mark.parametrize("failed", [True, False])
def test_command_writes_structured_status(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], failed: bool
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["collector", "--reference", "ref", "--wheelhouse", "wheels", "--origins", "origins"],
    )

    def collect(*_):
        if failed:
            raise ValueError("V5D6_REFERENCE_SHA256")
        return {"status": "MEASURED"}

    monkeypatch.setattr(collector, "collect", collect)
    assert collector.main() == (2 if failed else 0)
    assert json.loads(capsys.readouterr().out)["status"] == ("REFUSED" if failed else "MEASURED")

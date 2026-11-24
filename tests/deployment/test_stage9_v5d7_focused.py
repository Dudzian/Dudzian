"""Focused static evidence records exact bytes and reports incomplete observations."""

from __future__ import annotations

import base64
import hashlib
from typing import Any

import pytest

from scripts import stage9_v5d7_focused as focused


def test_focused_records_complete_bytes_and_bounded_prefix() -> None:
    member = "scipy/linalg/tests/_cython_examples/extending.pyx"
    assert focused.focused_kind(focused.SCIPY_WHEEL, member) == "SCIPY_CYTHON_SOURCE_CANDIDATE"
    assert focused.focused_kind(focused.SCIPY_WHEEL, member.upper()) is None
    assert focused.focused_kind("other.whl", member) is None

    data = b"#!cython \xff" + b"a" * focused.PREFIX_SIZE + b"\nend"
    row = focused.focused_evidence(data)
    assert base64.b64decode(row["content_b64"]) == data
    assert bytes.fromhex(row["prefix_hex"]) == data[: focused.PREFIX_SIZE]
    assert row["prefix_printable"].startswith("#!cython .")
    assert row["first_line_preview"].startswith(r"#!cython \xff")
    assert row["first_line_preview_complete"] is False
    assert row["prefix_has_shebang"] is True
    assert row["sha256"] == hashlib.sha256(data).hexdigest()
    assert row["size"] == len(data)
    assert focused.focused_evidence(b"plain\nsecond")["first_line_preview_complete"] is True
    assert focused.focused_evidence(b"plain")["prefix_has_shebang"] is False


def test_reference_measurements_distinguish_scope_and_partial_search() -> None:
    wheel = focused.GREENLET_WHEEL
    assert focused.focus_reference_scope(wheel, "greenlet/platform/setup.cfg.py") is True
    assert focused.focus_reference_scope(wheel, "greenlet/Makefile") is True
    assert focused.focus_references(wheel, "greenlet/runtime.bin", b"greenlet.h") is None
    assert focused.focus_references("other.whl", "module.py", b"greenlet.h") is None
    assert focused.focus_references(wheel, "build.py", b"none here") is None
    assert focused.focus_references(wheel, "build.py", b"\x00") == {
        "status": "PARTIAL",
        "reason": "REFERENCE_MEMBER_CONTAINS_NUL",
    }
    assert focused.focus_references(
        wheel, "build.py", b"a" * (focused.REFERENCE_SIZE_LIMIT + 1)
    ) == {"status": "PARTIAL", "reason": "REFERENCE_MEMBER_TOO_LARGE"}

    evidence = focused.focus_references(
        wheel, "greenlet/platform/setup.py", b"text\ninclude GREENLET.H switch_x64_masm.obj"
    )
    assert evidence == {
        "status": "MEASURED",
        "matched_line_count": 1,
        "hits": [
            {
                "line_number": 2,
                "matched_references": ["greenlet.h", "switch_x64_masm.obj"],
                "line_preview": "include GREENLET.H switch_x64_masm.obj",
                "line_preview_truncated": False,
            }
        ],
    }

    data = (b"from extending import feature # " + b"a" * 512 + b"\n") * (
        focused.REFERENCE_HIT_LIMIT + 1
    )
    truncated = focused.focus_references(focused.SCIPY_WHEEL, "test_example.py", data)
    assert truncated is not None
    assert truncated["status"] == "PARTIAL"
    assert truncated["matched_line_count"] == focused.REFERENCE_HIT_LIMIT + 1
    assert len(truncated["hits"]) == focused.REFERENCE_HIT_LIMIT
    assert all(hit["line_preview_truncated"] for hit in truncated["hits"])


def test_metadata_evidence_preserves_folding_and_repeated_fields() -> None:
    raw = (
        b"Metadata-Version: 2.5\r\nName: example\r\nVersion: 1\r\n"
        b"Import-Name: example;\r\n private\r\nImport-Namespace: example.contrib\r\n"
        b"Requires-Dist: first\r\nRequires-Dist: second\r\nRequires-Python: >=3.12\r\n"
        b"Provides-Extra: test\r\n\r\nExample text.\r\n"
    )
    report = focused.metadata_evidence(raw)
    assert report["status"] == "MEASURED"
    assert base64.b64decode(report["content_b64"]) == raw
    assert report["metadata_version"] == "2.5"
    assert report["requires_dist"] == ["first", "second"]
    assert report["requires_python"] == [">=3.12"]
    assert report["provides_extra"] == ["test"]
    assert report["import_name"] == ["example; private"]
    assert report["import_namespace"] == ["example.contrib"]
    assert {"name": "Import-Name", "value": "example;\r\n private"} in report["fields"]


@pytest.mark.parametrize(
    ("raw", "defect_type"),
    [
        (b"Metadata-Version: 2.4\nMetadata-Version: 2.5\n", "RequiredFieldCardinality"),
        (b"Requires-Python: >=3.12\nrequires-python: >=3.11\n", "RepeatedSingletonField"),
        (b"Requires-Python:\n", "EmptyRequiresPython"),
        (b"Description: invalid \xff\n", "InvalidUtf8"),
        (b"invalid header\n", "MissingHeaderBodySeparatorDefect"),
    ],
)
def test_metadata_evidence_reports_defects(raw: bytes, defect_type: str) -> None:
    report = focused.metadata_evidence(raw)
    assert report["status"] == "PARTIAL"
    assert base64.b64decode(report["content_b64"]) == raw
    assert any(defect["type"] == defect_type for defect in report["parser_defects"])


def shared_owners(path: str) -> list[dict[str, Any]]:
    data = b"# synthetic identical source\n"
    return [
        {
            "wheel": wheel,
            "wheel_sha256": digest,
            "member": path,
            "sha256": hashlib.sha256(data).hexdigest(),
            "size": len(data),
            "mode": "0o100644",
            "type": "file",
            "content_b64": base64.b64encode(data).decode("ascii"),
        }
        for wheel, digest in focused.SHARED_ARCHIVES.items()
    ]


def test_shared_path_count_keeps_comparison_evidence_separate_from_approval() -> None:
    paths = ["PySide6/__init__.py"] + [
        f"PySide6/member{index}.pyi" for index in range(focused.SHARED_DECLARATION_LIMIT - 1)
    ]
    groups = {path: shared_owners(path) for path in paths}
    archive_hashes = dict(focused.SHARED_ARCHIVES)
    partial = focused.shared_evidence(groups, archive_hashes=archive_hashes)
    assert partial["status"] == "PARTIAL"
    expected = [{"target": path, "owners": groups[path]} for path in paths]
    members = {path: groups[path][0] for path in paths}
    measured = focused.shared_evidence(groups, expected, archive_hashes, reference_members=members)
    assert measured["status"] == "MEASURED"
    assert measured["observed_collision_paths"] == 62
    assert measured["observed_three_owner_collision_paths"] == 62
    assert measured["source_review_required"] == ["PySide6/__init__.py"]
    assert all(all(row["matching_fingerprints"].values()) for row in measured["paths"])


@pytest.mark.parametrize(
    ("mutation", "reason"),
    [
        ("missing", "SHARED_WHEEL_OWNER_MISMATCH"),
        ("duplicate", "SHARED_WHEEL_INTERNAL_COLLISION"),
        ("directory", "SHARED_WHEEL_FILE_DIRECTORY_CONFLICT"),
        ("ancestor", "SHARED_WHEEL_FILE_DIRECTORY_CONFLICT"),
        ("archive", "SHARED_WHEEL_ARCHIVE_HASH_MISMATCH"),
        ("member", "SHARED_WHEEL_MEMBER_PATH_MISMATCH"),
        ("hash", "SHARED_WHEEL_MEMBER_HASH_MISMATCH"),
        ("size", "SHARED_WHEEL_CONTENT_OR_MODE_MISMATCH"),
        ("mode", "SHARED_WHEEL_CONTENT_OR_MODE_MISMATCH"),
        ("type", "SHARED_WHEEL_LEGACY_MEMBER_IDENTITY_MISMATCH"),
        ("base64", "SHARED_WHEEL_PYTHON_SOURCE_EVIDENCE_MISSING"),
        ("bytes", "SHARED_WHEEL_PYTHON_SOURCE_BYTES_MISMATCH"),
    ],
)
def test_shared_measurement_reports_independent_refusals(mutation: str, reason: str) -> None:
    path = "PySide6/submodule/__init__.py"
    owners = shared_owners(path)
    original = [dict(owner) for owner in owners]
    groups = {path: owners}
    directories = set()
    if mutation == "missing":
        owners.pop()
    elif mutation == "duplicate":
        owners.append(dict(owners[0]))
    elif mutation == "directory":
        directories.add(path)
    elif mutation == "ancestor":
        groups["PySide6/submodule"] = shared_owners("PySide6/submodule")[:1]
    elif mutation == "archive":
        owners[0]["wheel_sha256"] = "0" * 64
    elif mutation == "member":
        owners[0]["member"] = "changed.py"
    elif mutation == "hash":
        owners[0]["sha256"] = "0" * 64
    elif mutation == "size":
        owners[0]["size"] += 1
    elif mutation == "mode":
        owners[0]["mode"] = "0o100755"
    elif mutation == "type":
        owners[0]["type"] = "symlink"
    elif mutation == "base64":
        owners[0]["content_b64"] = "invalid?"
    elif mutation == "bytes":
        changed = b"# other source\n"
        owners[0].update(
            sha256=hashlib.sha256(changed).hexdigest(),
            size=len(changed),
            content_b64=base64.b64encode(changed).decode("ascii"),
        )

    report = focused.shared_evidence(
        groups,
        [{"target": path, "owners": original}],
        dict(focused.SHARED_ARCHIVES),
        directories,
        {path: original[0]},
    )
    assert report["status"] == "REFUSED"
    assert "SHARED_WHEEL_TARGET_REFUSED" in report["reasons"]
    assert reason in report["paths"][0]["reasons"]


@pytest.mark.parametrize("value", [True, -1, 0x10000, "0o9", None])
def test_unix_mode_refuses_unrepresentable_values(value: object) -> None:
    assert focused.unix_mode(value) is None

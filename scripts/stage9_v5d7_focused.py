"""Targeted measurements for the Stage 9 V5d7 wheelhouse diagnostic."""

from __future__ import annotations

import base64
import binascii
from collections import Counter
from email.parser import BytesHeaderParser
from email.policy import compat32
import hashlib
from pathlib import PurePosixPath
import re
import stat
from typing import Any


GREENLET_WHEEL = "greenlet-3.5.6-cp312-cp312-manylinux_2_24_x86_64.manylinux_2_28_x86_64.whl"
SCIPY_WHEEL = "scipy-1.18.1-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl"
PYSIDE6_ESSENTIALS = "pyside6_essentials-6.10.3-cp39-abi3-manylinux_2_34_x86_64.whl"
SHARED_ARCHIVES = {
    "pyside6-6.10.3-cp39-abi3-manylinux_2_34_x86_64.whl": (
        "36a12f000622e77ef0d98597d2640ae8775954ed900b578eacb6c9fca6cf78fb"
    ),
    "pyside6_addons-6.10.3-cp39-abi3-manylinux_2_34_x86_64.whl": (
        "a9a53a824ba8429953345c89a2dd5dc65d1f1d7e88b6cf1b975a6ef4f4d9760f"
    ),
    PYSIDE6_ESSENTIALS: "5fb32b48010f513335a2166ac3df279c5996fb96402fd91a4875fc717fe55de0",
}
SHARED_PYTHON_PATHS = {
    "PySide6/__init__.py",
    "PySide6/_config.py",
    "PySide6/_git_pyside_version.py",
}
METADATA_WHEELS = {
    "more_itertools-11.2.1-py3-none-any.whl",
    "pydantic-2.14.3-py3-none-any.whl",
    "python_dateutil-4.0.0-py3-none-any.whl",
    "typing_inspection-0.4.5-py3-none-any.whl",
    "urllib3-2.8.0-py3-none-any.whl",
}
FOCUSED_ARTIFACTS = {
    GREENLET_WHEEL: {
        "greenlet-3.5.6.data/headers/greenlet.h": "EXISTING_GREENLET_HEADER_CANDIDATE",
        "greenlet/greenlet.h": "GREENLET_RUNTIME_HEADER_CANDIDATE",
        "greenlet/platform/setup_switch_x64_masm.cmd": "GREENLET_MASM_SCRIPT_CANDIDATE",
        "greenlet/platform/switch_arm64_masm.obj": "GREENLET_COFF_OBJECT_CANDIDATE",
        "greenlet/platform/switch_x64_masm.obj": "GREENLET_COFF_OBJECT_CANDIDATE",
    },
    SCIPY_WHEEL: {
        f"scipy/{package}/tests/_cython_examples/extending.pyx": "SCIPY_CYTHON_SOURCE_CANDIDATE"
        for package in ("linalg", "optimize", "special")
    },
    PYSIDE6_ESSENTIALS: {
        f"PySide6/support/__pycache__/{module}.cpython-311.pyc": "EXISTING_PYC_OMISSION_CANDIDATE"
        for module in ("__init__", "deprecated")
    },
}
REFERENCE_PATTERNS = {
    GREENLET_WHEEL: {
        "greenlet.h": re.compile(rb"\bgreenlet\.h\b", re.IGNORECASE),
        "setup_switch_x64_masm.cmd": re.compile(
            rb"\bsetup_switch_x64_masm(?:\.cmd)?\b", re.IGNORECASE
        ),
        "switch_arm64_masm.obj": re.compile(rb"\bswitch_arm64_masm(?:\.obj)?\b", re.IGNORECASE),
        "switch_x64_masm.obj": re.compile(rb"\bswitch_x64_masm(?:\.obj)?\b", re.IGNORECASE),
    },
    SCIPY_WHEEL: {
        "extending.pyx": re.compile(rb"\bextending\.pyx\b", re.IGNORECASE),
        "_cython_examples": re.compile(rb"\b_cython_examples\b", re.IGNORECASE),
        "_test_cython_extension": re.compile(rb"\b_test_cython_extension\b"),
        "extending_module": re.compile(rb"\b(?:import|from)\s+\.?extending\b"),
    },
}
REFERENCE_SUFFIXES = {
    ".asm",
    ".c",
    ".cc",
    ".cmd",
    ".cpp",
    ".cxx",
    ".h",
    ".hh",
    ".hpp",
    ".hxx",
    ".in",
    ".ini",
    ".json",
    ".md",
    ".pxd",
    ".pxi",
    ".py",
    ".pyi",
    ".pyx",
    ".rst",
    ".s",
    ".sh",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}
REFERENCE_FILENAMES = {"Makefile", "CMakeLists.txt", "meson.build", "meson_options.txt", "METADATA"}
REFERENCE_SIZE_LIMIT = 4 * 1024 * 1024
REFERENCE_HIT_LIMIT = 32
PREFIX_SIZE = 128
SHARED_DECLARATION_LIMIT = 62
HEX_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


def focused_kind(wheel_name: str, member_path: str) -> str | None:
    return FOCUSED_ARTIFACTS.get(wheel_name, {}).get(member_path)


def focused_evidence(raw: bytes) -> dict[str, Any]:
    """Capture exact bytes and the prefix for offline evidence review."""
    prefix = raw[:PREFIX_SIZE]
    first_line = prefix.split(b"\n", 1)[0]
    return {
        "content_b64": base64.b64encode(raw).decode("ascii"),
        "prefix_hex": prefix.hex(),
        "prefix_printable": "".join(chr(byte) if 32 <= byte <= 126 else "." for byte in prefix),
        "first_line_preview": first_line.decode("ascii", errors="backslashreplace"),
        "first_line_preview_complete": b"\n" in prefix or len(raw) <= PREFIX_SIZE,
        "prefix_has_shebang": raw.startswith(b"#!"),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "size": len(raw),
    }


def focus_reference_scope(wheel_name: str, member_path: str) -> bool:
    path = PurePosixPath(member_path)
    return wheel_name in REFERENCE_PATTERNS and (
        path.suffix.lower() in REFERENCE_SUFFIXES or path.name in REFERENCE_FILENAMES
    )


def focus_references(wheel_name: str, member_path: str, raw: bytes) -> dict[str, Any] | None:
    """Search selected source/build text; these are lexical references only."""
    if not focus_reference_scope(wheel_name, member_path):
        return None
    if len(raw) > REFERENCE_SIZE_LIMIT:
        return {"status": "PARTIAL", "reason": "REFERENCE_MEMBER_TOO_LARGE"}
    if b"\x00" in raw:
        return {"status": "PARTIAL", "reason": "REFERENCE_MEMBER_CONTAINS_NUL"}
    patterns = REFERENCE_PATTERNS[wheel_name]
    hits: list[dict[str, Any]] = []
    matched_line_count = 0
    for line_number, line in enumerate(raw.splitlines(), start=1):
        matches = [label for label, pattern in patterns.items() if pattern.search(line)]
        if not matches:
            continue
        matched_line_count += 1
        if len(hits) < REFERENCE_HIT_LIMIT:
            hits.append(
                {
                    "line_number": line_number,
                    "matched_references": matches,
                    "line_preview": line[:512].decode("utf-8", errors="backslashreplace"),
                    "line_preview_truncated": len(line) > 512,
                }
            )
    if not hits:
        return None
    return {
        "status": "MEASURED" if matched_line_count == len(hits) else "PARTIAL",
        "matched_line_count": matched_line_count,
        "hits": hits,
    }


def metadata_evidence(raw: bytes) -> dict[str, Any]:
    """Measure the original METADATA including malformed or duplicated fields."""
    defects: list[dict[str, str]] = []
    try:
        raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        defects.append({"type": "InvalidUtf8", "detail": str(exc)})

    message = BytesHeaderParser(policy=compat32).parsebytes(raw)
    defects.extend(
        {"type": type(defect).__name__, "detail": str(defect)} for defect in message.defects
    )
    fields = []
    normalized: dict[str, list[str]] = {}
    for name, value in message.raw_items():
        original = value.encode("utf-8", errors="surrogateescape")
        text = original.decode("utf-8", errors="replace")
        fields.append({"name": name, "value": text})
        unfolded = re.sub(r"(?:\r\n|\r|\n)(?=[ \t])", "", text).strip(" \t")
        normalized.setdefault(name.lower(), []).append(unfolded)

    versions = normalized.get("metadata-version", [])
    for name in ("metadata-version", "name", "version"):
        values = normalized.get(name, [])
        if len(values) != 1 or not values[0]:
            defects.append({"type": "RequiredFieldCardinality", "detail": name})
    requires_python = normalized.get("requires-python", [])
    if len(requires_python) > 1:
        defects.append({"type": "RepeatedSingletonField", "detail": "requires-python"})
    elif requires_python and not requires_python[0]:
        defects.append({"type": "EmptyRequiresPython", "detail": "requires-python"})

    return {
        "status": "PARTIAL" if defects else "MEASURED",
        "content_b64": base64.b64encode(raw).decode("ascii"),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "size": len(raw),
        "fields": fields,
        "parser_defects": defects,
        "metadata_version": versions[0] if len(versions) == 1 and versions[0] else None,
        "metadata_versions": versions,
        "names": normalized.get("name", []),
        "versions": normalized.get("version", []),
        "requires_dist": normalized.get("requires-dist", []),
        "requires_python": requires_python,
        "provides_extra": normalized.get("provides-extra", []),
        "import_name": normalized.get("import-name", []),
        "import_namespace": normalized.get("import-namespace", []),
    }


def unix_mode(value: Any) -> int | None:
    if type(value) is int and 0 <= value <= 0xFFFF:
        return value
    if isinstance(value, str) and re.fullmatch(r"0o[0-7]{1,6}", value):
        return unix_mode(int(value, 8))
    return None


def _member_identity(owner: dict[str, Any]) -> tuple[str, int, int, str] | None:
    digest = owner.get("sha256")
    size = owner.get("size")
    mode = unix_mode(owner.get("mode"))
    kind = owner.get("type")
    if (
        not isinstance(digest, str)
        or not HEX_SHA256.fullmatch(digest)
        or type(size) is not int
        or size < 0
        or mode is None
        or not stat.S_ISREG(mode)
        or kind not in ("file", "regular")
    ):
        return None
    return digest, size, mode, "regular"


def _retained_content(owner: dict[str, Any]) -> bytes | None:
    content = owner.get("content_b64")
    identity = _member_identity(owner)
    if not isinstance(content, str) or not identity:
        return None
    try:
        raw = base64.b64decode(content, validate=True)
    except (ValueError, binascii.Error):
        return None
    if len(raw) != identity[1] or hashlib.sha256(raw).hexdigest() != identity[0]:
        return None
    return raw


def shared_evidence(
    groups: dict[str, list[dict[str, Any]]],
    reference_shared: list[dict[str, Any]] | None = None,
    archive_hashes: dict[str, str] | None = None,
    directories: set[str] | None = None,
    reference_members: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Compare projected collisions to the measured V5d6 path and archive identities."""
    reference = {row["target"]: row["owners"] for row in reference_shared or []}
    if reference_members:
        for path in reference_members:
            reference.setdefault(path, [])
    expected_paths = set(reference)
    collisions = {
        path
        for path, owners in groups.items()
        if len(owners) > 1 and (path.startswith("PySide6/") or path in expected_paths)
    }
    targets = expected_paths | collisions
    directory_paths = set(directories or ())
    for path in groups:
        directory_paths.update(
            parent.as_posix() for parent in PurePosixPath(path).parents if parent.parts
        )

    rows = []
    for target in sorted(targets):
        owners = sorted(groups.get(target, []), key=lambda owner: (owner["wheel"], owner["member"]))
        owner_counts = Counter(owner["wheel"] for owner in owners)
        defects = []
        if set(owner_counts) != set(SHARED_ARCHIVES):
            defects.append("SHARED_WHEEL_OWNER_MISMATCH")
        if any(count > 1 for count in owner_counts.values()):
            defects.append("SHARED_WHEEL_INTERNAL_COLLISION")
        if target in directory_paths or any(
            parent.as_posix() in groups for parent in PurePosixPath(target).parents if parent.parts
        ):
            defects.append("SHARED_WHEEL_FILE_DIRECTORY_CONFLICT")

        reference_owners = {owner["wheel"]: owner for owner in reference.get(target, [])}
        for owner in owners:
            wheel = owner["wheel"]
            measured_digest = (archive_hashes or {}).get(wheel)
            row_digest = owner.get("wheel_sha256")
            if (measured_digest or row_digest) != SHARED_ARCHIVES.get(wheel) or (
                measured_digest and row_digest and measured_digest != row_digest
            ):
                defects.append("SHARED_WHEEL_ARCHIVE_HASH_MISMATCH")
            expected_owner = reference_owners.get(wheel, {})
            if owner["member"] != expected_owner.get("member", target):
                defects.append("SHARED_WHEEL_MEMBER_PATH_MISMATCH")
            if expected_owner and owner.get("sha256") != expected_owner.get("sha256"):
                defects.append("SHARED_WHEEL_MEMBER_HASH_MISMATCH")

        identities = [_member_identity(owner) for owner in owners]
        dimensions = ("sha256", "size", "mode", "type")
        equality = {
            key: bool(identities)
            and None not in identities
            and len({identity[index] for identity in identities if identity is not None}) == 1
            for index, key in enumerate(dimensions)
        }
        if not all(equality.values()):
            defects.append("SHARED_WHEEL_CONTENT_OR_MODE_MISMATCH")
        if reference_members and target in reference_members:
            expected_identity = _member_identity(reference_members[target])
            if not expected_identity or any(
                identity != expected_identity for identity in identities
            ):
                defects.append("SHARED_WHEEL_LEGACY_MEMBER_IDENTITY_MISMATCH")

        source_required = target.endswith(".py")
        if source_required:
            retained_bytes = [_retained_content(owner) for owner in owners]
            all_source_retained = bool(owners) and None not in retained_bytes
            if all_source_retained and len(set(retained_bytes)) != 1:
                defects.append("SHARED_WHEEL_PYTHON_SOURCE_BYTES_MISMATCH")
            if not all_source_retained:
                defects.append("SHARED_WHEEL_PYTHON_SOURCE_EVIDENCE_MISSING")
        status = "MEASURED" if not defects else "REFUSED"
        rows.append(
            {
                "target": target,
                "owners": owners,
                "owner_count": len(owner_counts),
                "matching_fingerprints": equality,
                "source_review": "REQUIRED" if source_required else "NOT_APPLICABLE",
                "status": status,
                "reasons": sorted(set(defects)),
            }
        )

    reasons = []
    if len(collisions) != SHARED_DECLARATION_LIMIT or (
        expected_paths and collisions != expected_paths
    ):
        reasons.append("SHARED_WHEEL_MEMBER_SET_MISMATCH")
    if any(row["status"] == "REFUSED" for row in rows):
        reasons.append("SHARED_WHEEL_TARGET_REFUSED")
    status = "REFUSED" if reasons else "MEASURED" if expected_paths else "PARTIAL"
    return {
        "status": status,
        "expected_path_count": SHARED_DECLARATION_LIMIT,
        "expected_path_manifest_supplied": bool(expected_paths),
        "observed_collision_paths": len(collisions),
        "observed_three_owner_collision_paths": sum(
            row["target"] in collisions and row["owner_count"] == 3 for row in rows
        ),
        "measured_archive_sha256": {
            wheel: (archive_hashes or {}).get(wheel) for wheel in SHARED_ARCHIVES
        },
        "missing_paths": sorted(expected_paths - collisions),
        "unexpected_paths": sorted(collisions - expected_paths) if expected_paths else [],
        "reasons": reasons,
        "paths": rows,
        "source_review_required": [
            row["target"] for row in rows if row["source_review"] == "REQUIRED"
        ],
    }

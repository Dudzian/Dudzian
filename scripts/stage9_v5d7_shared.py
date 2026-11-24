"""Exact sharing contract for three PySide6 archives measured in V5d7."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import runpy
from typing import Any, cast

POLICY_ID = "stage9-v5d7-pyside6-6.10.3-shared-members-v2"
LEGACY_POLICY_ID = "stage9-v5d5-pyside6-6.10.3-shared-members-v1"
LEGACY_PINS_SHA256 = "71fe307d2f6372ddfc7c59bc76e5e3eee8790e4636a24d773b680dadeb65074b"
APPROVAL_REQUIRED = "SHARED_WHEEL_MEMBERS_REQUIRES_INDEPENDENT_APPROVAL"
SOURCE_REVIEW_DECISION = "APPROVED_FOR_SHARED_MATERIALIZATION"
AUDIT_FILENAME = "shared-audit-v5d7.py"

_path = Path(__file__)
_legacy = runpy.run_path(
    str(
        _path.with_name(
            "shared-audit.py" if _path.name == AUDIT_FILENAME else "stage9_v5d5_shared.py"
        )
    )
)
if _legacy["POLICY_ID"] != LEGACY_POLICY_ID:
    raise ValueError("SHARED_WHEEL_BASELINE_POLICY_MISMATCH")
_pins = json.dumps(
    [_legacy["WHEELS"], _legacy["MEMBERS"]],
    sort_keys=True,
    separators=(",", ":"),
    allow_nan=False,
).encode("utf-8")
if hashlib.sha256(_pins).hexdigest() != LEGACY_PINS_SHA256:
    raise ValueError("SHARED_WHEEL_BASELINE_PINS_MISMATCH")

exact = _legacy["exact"]
safe_path = _legacy["safe_path"]
require = _legacy["require"]

WHEELS: dict[str, dict[str, str]] = {
    **_legacy["WHEELS"],
    "pyside6": {
        "version": "6.10.3",
        "wheel_filename": "pyside6-6.10.3-cp39-abi3-manylinux_2_34_x86_64.whl",
        "wheel_sha256": "36a12f000622e77ef0d98597d2640ae8775954ed900b578eacb6c9fca6cf78fb",
    },
}
# All 186 source rows match these 62 pins, including regular type and Unix mode.
MEMBERS = cast(dict[str, dict[str, Any]], _legacy["MEMBERS"])


def collision_inventory(packages: list[dict[str, Any]]) -> dict[str, Any]:
    """Measure the complete collision set and bind every owner without approving it."""
    require(type(packages) is list, "SHARED_WHEEL_PACKAGE_SCHEMA")
    declarations: dict[str, list[dict[str, Any]]] = {}
    distributions: set[str] = set()
    for package in packages:
        require(
            type(package) is dict
            and all(
                type(package.get(key)) is str and bool(package[key])
                for key in ("name", "version", "filename", "sha256")
            )
            and re.fullmatch(r"[0-9a-f]{64}", package["sha256"]) is not None
            and type(package.get("materialized_members")) is dict
            and type(package.get("member_metadata")) is dict,
            "SHARED_WHEEL_PACKAGE_SCHEMA",
        )
        distribution = package["name"]
        require(distribution not in distributions, "SHARED_WHEEL_DUPLICATE_DISTRIBUTION")
        distributions.add(distribution)
        targets: set[str] = set()
        for member, mapping in package["materialized_members"].items():
            require(
                type(mapping) is dict
                and set(mapping) == {"path", "sha256"}
                and type(mapping["sha256"]) is str
                and re.fullmatch(r"[0-9a-f]{64}", mapping["sha256"]) is not None,
                "SHARED_WHEEL_MEMBER_SCHEMA",
            )
            path = safe_path(mapping["path"])
            safe_path(member)
            metadata = package["member_metadata"].get(member)
            require(
                type(metadata) is dict
                and set(metadata) == {"type", "mode", "size"}
                and all(
                    type(metadata[key]) is int and metadata[key] >= 0 for key in ("mode", "size")
                )
                and metadata["type"] == "regular",
                "SHARED_WHEEL_MEMBER_METADATA_SCHEMA",
            )
            require(path not in targets, "WHEEL_MATERIALIZATION_COLLISION:" + path)
            targets.add(path)
            declarations.setdefault(path, []).append(
                {
                    "owner": {
                        "distribution": distribution,
                        "version": package["version"],
                        "wheel_filename": package["filename"],
                        "wheel_sha256": package["sha256"],
                        "member": member,
                    },
                    "identity": {"sha256": mapping["sha256"], **metadata},
                }
            )
    for path in declarations:
        require(
            all(parent.as_posix() not in declarations for parent in PurePosixPath(path).parents),
            "WHEEL_MATERIALIZATION_COLLISION:" + path,
        )
    require(set(WHEELS) <= distributions, "SHARED_WHEEL_OWNER_SET_MISMATCH")
    shared = {path for path, rows in declarations.items() if len(rows) > 1}
    require(shared == set(MEMBERS), "SHARED_WHEEL_MEMBER_SET_MISMATCH")
    members: list[dict[str, Any]] = []
    for path in sorted(shared):
        expected = [
            {"distribution": name, **identity, "member": path}
            for name, identity in sorted(WHEELS.items())
        ]
        owners = sorted(
            (row["owner"] for row in declarations[path]), key=lambda owner: owner["distribution"]
        )
        require(exact(owners, expected), "SHARED_WHEEL_OWNER_MISMATCH:" + path)
        pin = MEMBERS[path]
        require(
            all(exact(row["identity"], pin) for row in declarations[path]),
            "SHARED_WHEEL_CONTENT_OR_MODE_MISMATCH:" + path,
        )
        members.append({"path": path, **pin, "owners": owners})
    return {
        "policy": POLICY_ID,
        "status": APPROVAL_REQUIRED,
        "members": members,
        "required_source_reviews": [
            {"path": row["path"], "sha256": row["sha256"]}
            for row in members
            if row["path"].endswith(".py")
        ],
    }


def verify_shared(
    packages: list[dict[str, Any]], approved_shared: dict[str, Any] | None
) -> dict[str, Any]:
    """Require the separate V5d7 expectation and current Python source review."""
    inventory = collision_inventory(packages)
    if type(approved_shared) is not dict:
        raise ValueError(APPROVAL_REQUIRED)
    require(
        set(approved_shared) == {"policy", "members", "source_reviews"}
        and approved_shared["policy"] == POLICY_ID
        and exact(approved_shared["members"], inventory["members"]),
        "SHARED_WHEEL_EXPECTATION_MISMATCH",
    )
    reviews = approved_shared["source_reviews"]
    sources = inventory["required_source_reviews"]
    require(
        type(reviews) is list and len(reviews) == len(sources),
        "SHARED_WHEEL_PYTHON_SOURCE_REVIEW_REQUIRED",
    )
    for review, source in zip(reviews, sources, strict=True):
        require(
            type(review) is dict
            and set(review) == {"path", "sha256", "decision", "review_reference"}
            and exact(
                {key: review[key] for key in ("path", "sha256", "decision")},
                {**source, "decision": SOURCE_REVIEW_DECISION},
            )
            and type(review["review_reference"]) is str
            and bool(review["review_reference"].strip()),
            "SHARED_WHEEL_PYTHON_SOURCE_REVIEW_REQUIRED",
        )
    return inventory

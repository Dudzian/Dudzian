"""V5d7 exact omissions for build inputs and development tests.

The caller validates the complete original archive and RECORD first. The pins
describe candidate eligibility for a prebuilt Linux runtime. A separate review
must authorize their omission, including the loss of downstream Greenlet
C-extension builds and the bundled SciPy Cython example tests.
"""

from __future__ import annotations

import hashlib
import json
import re
import stat
from typing import Any
from zipfile import ZipFile

POLICY_ID = "stage9-v5d7-non-runtime-build-and-test-members-v1"
APPROVAL_REQUIRED = "ARTIFACT_OMISSIONS_REQUIRE_INDEPENDENT_APPROVAL"
REVIEW_DECISION = "APPROVED_FOR_ARTIFACT_OMISSION"
INSTALLATION_EXCEPTION = "OMIT_PINNED_NON_RUNTIME_BUILD_AND_TEST_MEMBERS"

GREENLET_SHA256 = "975736b002ed080d124cf81a79cb7e05cb26d6b3f5c7a7b651c0fcce70353aa1"
SCIPY_SHA256 = "f55fa87b6c612ecd6b058f167c53231b1d14e412efe361d3d6e38b3631c73218"
WHEELS = {
    GREENLET_SHA256: {
        "distribution": "greenlet",
        "version": "3.5.6",
        "wheel_filename": "greenlet-3.5.6-cp312-cp312-manylinux_2_24_x86_64.manylinux_2_28_x86_64.whl",
    },
    SCIPY_SHA256: {
        "distribution": "scipy",
        "version": "1.18.1",
        "wheel_filename": "scipy-1.18.1-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl",
    },
}
MEMBERS = {
    GREENLET_SHA256: {
        "greenlet/greenlet.h": (
            "b33e69611490a9e7603add80320c4b65d4e33bea9caff24cbe0884251f48009f",
            4755,
            "C_API_BUILD_HEADER",
        ),
        "greenlet/platform/setup_switch_x64_masm.cmd": (
            "6690a5509794d2e8c43d24d63527a93f45b67fd5e2610280f10292a15a2ef045",
            143,
            "WINDOWS_MASM_BUILD_SCRIPT",
        ),
        "greenlet/platform/switch_arm64_masm.obj": (
            "0e62e7201fe2728107033d676ae7bfa493d366047d12533bf8d9b3b6bfa8f7f5",
            746,
            "WINDOWS_MASM_COFF_BUILD_OBJECT",
        ),
        "greenlet/platform/switch_x64_masm.obj": (
            "18db5337161da3b89d15462c42ffa6ad75a0c93e4427ddfef6af7494deacbed4",
            1078,
            "WINDOWS_MASM_COFF_BUILD_OBJECT",
        ),
    },
    SCIPY_SHA256: {
        "scipy/linalg/tests/_cython_examples/extending.pyx": (
            "ab7f9c28180e8508b1bf91dc0b50f630672f3d62067729267bcc5e754ce56338",
            1054,
            "CYTHON_TEST_BUILD_INPUT",
        ),
        "scipy/optimize/tests/_cython_examples/extending.pyx": (
            "e5309817d86f218bbc4bd63b3c8aa5fb17497dc9ff2c8e74c83adfe2e87b8b63",
            1314,
            "CYTHON_TEST_BUILD_INPUT",
        ),
        "scipy/special/tests/_cython_examples/extending.pyx": (
            "d084858571c59f0b965e0e66f552187561a33ff5bb871504f12c05364bc033fb",
            292,
            "CYTHON_TEST_BUILD_INPUT",
        ),
    },
}
ARCHIVE_MODE = 0o100644
METADATA_KEYS = ("size", "mode", "type")
SHA256_FORMAT = re.compile(r"[0-9a-f]{64}\Z")


class Refusal(ValueError):
    """A pin, independent approval, or runtime exclusion failed validation."""


def require(condition: bool, reason: str) -> None:
    if not condition:
        raise Refusal(reason)


def exact(left: Any, right: Any) -> bool:
    """Keep JSON types significant during approval and receipt comparison."""
    try:
        return json.dumps(left, sort_keys=True, allow_nan=False) == json.dumps(
            right, sort_keys=True, allow_nan=False
        )
    except (TypeError, ValueError):
        return False


def pinned_rows(wheel_sha256: str) -> list[dict[str, Any]]:
    return [
        {
            **WHEELS[wheel_sha256],
            "wheel_sha256": wheel_sha256,
            "member": member,
            "runtime_path": member,
            "sha256": sha256,
            "size": size,
            "mode": ARCHIVE_MODE,
            "type": "regular",
            "kind": kind,
            "runtime_inclusion": "EXCLUDED",
        }
        for member, (sha256, size, kind) in sorted(MEMBERS.get(wheel_sha256, {}).items())
    ]


def inspect_omissions(
    distribution: str,
    version: str,
    filename: str,
    wheel_sha256: str,
    archive: ZipFile,
    files: set[str],
) -> list[dict[str, Any]]:
    """Check exact archive and member pins after full RECORD verification."""
    if wheel_sha256 not in WHEELS:
        return []
    require(
        exact(
            {"distribution": distribution, "version": version, "wheel_filename": filename},
            WHEELS[wheel_sha256],
        ),
        "ARTIFACT_OMISSION_WHEEL_IDENTITY_MISMATCH",
    )
    rows = pinned_rows(wheel_sha256)
    require({row["member"] for row in rows} <= files, "ARTIFACT_OMISSION_SET_MISMATCH")
    for row in rows:
        member = row["member"]
        info = archive.getinfo(member)
        mode = info.external_attr >> 16
        require(
            info.orig_filename == member
            and not info.is_dir()
            and stat.S_ISREG(mode)
            and mode == row["mode"]
            and info.file_size == row["size"],
            "ARTIFACT_OMISSION_MEMBER_METADATA_MISMATCH:" + member,
        )
        require(
            hashlib.sha256(archive.read(info)).hexdigest() == row["sha256"],
            "ARTIFACT_OMISSION_MEMBER_HASH_MISMATCH:" + member,
        )
    return rows


def requires_omissions(package: dict[str, Any]) -> bool:
    digest = package.get("sha256")
    return bool(package.get("artifact_omissions")) or (isinstance(digest, str) and digest in WHEELS)


def omission_inventory(packages: list[dict[str, Any]]) -> dict[str, Any]:
    """Validate all candidate declarations and their exclusion from the manifest."""
    require(type(packages) is list, "ARTIFACT_OMISSION_PACKAGE_SCHEMA")
    members: list[dict[str, Any]] = []
    used: set[str] = set()
    for package in packages:
        require(type(package) is dict, "ARTIFACT_OMISSION_PACKAGE_SCHEMA")
        omissions = package.get("artifact_omissions", [])
        require(type(omissions) is list, "ARTIFACT_OMISSION_PACKAGE_SCHEMA")
        if not requires_omissions(package):
            continue
        digest = package.get("sha256")
        require(
            type(digest) is str and digest in WHEELS and digest not in used,
            "ARTIFACT_OMISSION_WHEEL_IDENTITY_MISMATCH",
        )
        assert type(digest) is str
        used.add(digest)
        require(
            exact(
                {
                    "distribution": package.get("name"),
                    "version": package.get("version"),
                    "wheel_filename": package.get("filename"),
                },
                WHEELS[digest],
            ),
            "ARTIFACT_OMISSION_WHEEL_IDENTITY_MISMATCH",
        )
        expected = pinned_rows(digest)
        require(exact(omissions, expected), "ARTIFACT_OMISSION_SET_MISMATCH")
        metadata = package.get("member_metadata")
        require(type(metadata) is dict, "ARTIFACT_OMISSION_PACKAGE_SCHEMA")
        assert type(metadata) is dict
        for row in expected:
            require(
                exact(metadata.get(row["member"]), {key: row[key] for key in METADATA_KEYS}),
                "ARTIFACT_OMISSION_MEMBER_METADATA_MISMATCH:" + row["member"],
            )
        members.extend(expected)

    members.sort(key=lambda row: (row["distribution"], row["member"]))
    inventory = {
        "policy": POLICY_ID,
        "status": APPROVAL_REQUIRED if members else "NO_ARTIFACT_OMISSIONS",
        "members": members,
        "installation_contract_exception": INSTALLATION_EXCEPTION if members else "NONE",
        "runtime_inclusion": "EXCLUDED",
    }
    if members:
        _verify_manifest_absence(packages, inventory)
    return inventory


def _verify_manifest_absence(packages: list[dict[str, Any]], inventory: dict[str, Any]) -> None:
    sources = {row["member"] for row in inventory["members"]}
    for package in packages:
        materialized = package.get("materialized_members")
        require(type(materialized) is dict, "ARTIFACT_OMISSION_PACKAGE_SCHEMA")
        assert type(materialized) is dict
        files: dict[str, str] = {}
        for source, entry in materialized.items():
            require(
                type(source) is str
                and type(entry) is dict
                and type(entry.get("path")) is str
                and type(entry.get("sha256")) is str,
                "ARTIFACT_OMISSION_PACKAGE_SCHEMA",
            )
            require(source not in sources, "ARTIFACT_OMISSION_RUNTIME_INCLUSION_FORBIDDEN")
            require(entry["path"] not in files, "ARTIFACT_OMISSION_PACKAGE_SCHEMA")
            files[entry["path"]] = entry["sha256"]
        verify_runtime_absence(files, inventory)


def verify_omissions(
    packages: list[dict[str, Any]], approval: dict[str, Any] | None
) -> dict[str, Any]:
    """Compare the independently supplied expectation without authenticating it."""
    inventory = omission_inventory(packages)
    if not inventory["members"] and approval is None:
        return inventory
    require(type(approval) is dict, APPROVAL_REQUIRED)
    assert type(approval) is dict
    require(
        bool(inventory["members"])
        and set(approval)
        == {"policy", "installation_contract_exception", "members", "decision", "review_reference"}
        and approval["policy"] == POLICY_ID
        and approval["installation_contract_exception"] == INSTALLATION_EXCEPTION
        and exact(approval["members"], inventory["members"])
        and approval["decision"] == REVIEW_DECISION
        and type(approval["review_reference"]) is str
        and bool(approval["review_reference"].strip()),
        "ARTIFACT_OMISSION_EXPECTATION_MISMATCH",
    )
    return inventory


def verify_runtime_absence(files: dict[str, str], inventory: dict[str, Any]) -> None:
    """Check package-relative or receipt-root paths and byte aliases."""
    require(
        type(files) is dict
        and type(inventory) is dict
        and inventory.get("policy") == POLICY_ID
        and type(inventory.get("members")) is list,
        "ARTIFACT_OMISSION_RUNTIME_SCHEMA",
    )
    forbidden = {row[field] for row in inventory["members"] for field in ("member", "runtime_path")}
    forbidden_hashes = {row["sha256"] for row in inventory["members"]}
    for name, digest in files.items():
        require(
            type(name) is str
            and type(digest) is str
            and SHA256_FORMAT.fullmatch(digest) is not None,
            "ARTIFACT_OMISSION_RUNTIME_SCHEMA",
        )
        runtime_path = name.removeprefix("packages/")
        require(
            digest not in forbidden_hashes
            and not any(
                runtime_path == path or runtime_path.startswith(path + "/") for path in forbidden
            ),
            "ARTIFACT_OMISSION_RUNTIME_INCLUSION_FORBIDDEN",
        )

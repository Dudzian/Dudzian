"""V5d6: one pinned non-runtime header candidate, with separate approval.

The dependency inspector calls this policy only after verifying the complete
original RECORD. Policy eligibility never supplies independent expectations or
authorizes native loading, execution, installation, or production.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import PurePosixPath
import stat
from typing import Any

POLICY_ID = "stage9-v5d6-greenlet-3.5.6-non-runtime-header-v1"
APPROVAL_REQUIRED = "HEADER_OMISSIONS_REQUIRE_INDEPENDENT_APPROVAL"
REVIEW_DECISION = "APPROVED_FOR_HEADER_OMISSION"
INSTALLATION_EXCEPTION = "OMIT_PINNED_NON_RUNTIME_HEADER"
WHEEL: dict[str, str] = {
    "distribution": "greenlet",
    "version": "3.5.6",
    "wheel_filename": "greenlet-3.5.6-cp312-cp312-manylinux_2_24_x86_64.manylinux_2_28_x86_64.whl",
    "wheel_sha256": "975736b002ed080d124cf81a79cb7e05cb26d6b3f5c7a7b651c0fcce70353aa1",
}
HEADER: dict[str, Any] = {
    "member": "greenlet-3.5.6.data/headers/greenlet.h",
    "sha256": "b33e69611490a9e7603add80320c4b65d4e33bea9caff24cbe0884251f48009f",
    "size": 4755,
    "mode": 0o100644,
    "type": "regular",
}


class Refusal(ValueError):
    """An unknown header or incomplete separate expectation is refused."""


def require(condition: bool, reason: str) -> None:
    if not condition:
        raise Refusal(reason)


def exact(left: Any, right: Any) -> bool:
    """JSON types matter; booleans and floats cannot substitute for integers."""
    try:
        return json.dumps(left, sort_keys=True, allow_nan=False) == json.dumps(
            right, sort_keys=True, allow_nan=False
        )
    except (TypeError, ValueError):
        return False


def pinned_row() -> dict[str, Any]:
    return {**WHEEL, **HEADER, "policy": POLICY_ID, "runtime_inclusion": "EXCLUDED"}


def inspect_header(
    distribution: str,
    version: str,
    wheel_filename: str,
    wheel_sha256: str,
    member: str,
    content: bytes,
    mode: int,
) -> dict[str, Any]:
    """Check all bytes and original type/mode after the caller's full RECORD.

    This is deliberately separate from the purelib/platlib destination mapper.
    There is no destination for this header in the runtime environment.
    """
    identity = {
        "distribution": distribution,
        "version": version,
        "wheel_filename": wheel_filename,
        "wheel_sha256": wheel_sha256,
    }
    require(exact(identity, WHEEL), "UNSUPPORTED_WHEEL_DATA")
    require(member == HEADER["member"], "UNSUPPORTED_WHEEL_DATA")
    require(
        type(mode) is int
        and stat.S_ISREG(mode)
        and not mode & 0o111
        and not content.startswith((b"\x7fELF", b"MZ", b"#!"))
        and content[:4]
        not in {
            b"\xfe\xed\xfa\xce",
            b"\xce\xfa\xed\xfe",
            b"\xfe\xed\xfa\xcf",
            b"\xcf\xfa\xed\xfe",
            b"\xca\xfe\xba\xbe",
            b"\xbe\xba\xfe\xca",
            b"\xca\xfe\xba\xbf",
            b"\xbf\xba\xfe\xca",
        },
        "HEADER_NATIVE_OR_EXECUTABLE_FORBIDDEN",
    )
    require(
        mode == HEADER["mode"]
        and len(content) == HEADER["size"]
        and hashlib.sha256(content).hexdigest() == HEADER["sha256"],
        "HEADER_PIN_MISMATCH",
    )
    return pinned_row()


def header_inventory(packages: list[dict[str, Any]]) -> dict[str, Any]:
    """Validate declarations and runtime exclusion without granting approval."""
    require(type(packages) is list, "HEADER_PACKAGE_SCHEMA")
    members: list[dict[str, Any]] = []
    for package in packages:
        require(type(package) is dict, "HEADER_PACKAGE_SCHEMA")
        omissions = package.get("header_omissions", [])
        materialized = package.get("materialized_members")
        require(
            type(omissions) is list and type(materialized) is dict,
            "HEADER_PACKAGE_SCHEMA",
        )
        assert isinstance(materialized, dict)
        identity = {
            "distribution": package.get("name"),
            "version": package.get("version"),
            "wheel_filename": package.get("filename"),
            "wheel_sha256": package.get("sha256"),
        }
        pinned_distribution = all(
            identity[key] == WHEEL[key] for key in ("distribution", "version", "wheel_filename")
        )
        if omissions or pinned_distribution:
            require(exact(identity, WHEEL), "HEADER_WHEEL_IDENTITY_MISMATCH")
            require(exact(omissions, [pinned_row()]), "HEADER_OMISSION_SET_MISMATCH")
            metadata = package.get("member_metadata")
            require(
                type(metadata) is dict
                and exact(
                    metadata.get(HEADER["member"]),
                    {key: HEADER[key] for key in ("size", "mode", "type")},
                ),
                "HEADER_MEMBER_METADATA_MISMATCH",
            )
            members.extend(omissions)
        for member, entry in materialized.items():
            require(
                type(member) is str and type(entry) is dict and type(entry.get("path")) is str,
                "HEADER_PACKAGE_SCHEMA",
            )
            parts = PurePosixPath(entry["path"]).parts
            require(
                member != HEADER["member"]
                and entry["path"] != HEADER["member"]
                and entry.get("sha256") != HEADER["sha256"]
                and not (len(parts) > 1 and parts[0].endswith(".data") and parts[1] == "headers"),
                "HEADER_RUNTIME_INCLUSION_FORBIDDEN",
            )
    require(len(members) <= 1, "HEADER_OMISSION_SET_MISMATCH")
    return {
        "policy": POLICY_ID,
        "status": APPROVAL_REQUIRED if members else "NO_HEADER_OMISSIONS",
        "members": members,
        "installation_contract_exception": INSTALLATION_EXCEPTION if members else "NONE",
        "runtime_inclusion": "EXCLUDED",
    }


def verify_headers(
    packages: list[dict[str, Any]], approved_headers: dict[str, Any] | None
) -> dict[str, Any]:
    """Require separately supplied exact expectations and a review reference.

    Authentication of the reviewer and expectation origin remains external.
    Matching local declarations cannot create an independent approval.
    """
    inventory = header_inventory(packages)
    if not inventory["members"] and approved_headers is None:
        return inventory
    require(type(approved_headers) is dict, APPROVAL_REQUIRED)
    assert approved_headers is not None
    require(
        set(approved_headers)
        == {"policy", "installation_contract_exception", "members", "review_reference", "decision"}
        and approved_headers["policy"] == POLICY_ID
        and approved_headers["installation_contract_exception"]
        == inventory["installation_contract_exception"]
        and exact(approved_headers["members"], inventory["members"])
        and approved_headers["decision"] == REVIEW_DECISION
        and type(approved_headers["review_reference"]) is str
        and bool(approved_headers["review_reference"].strip()),
        "HEADER_EXPECTATION_MISMATCH",
    )
    return inventory

"""V5d5: exact shared-wheel candidates; independent approval remains mandatory.

Policy pins describe the only admissible candidates, never an approval. No
wheel contents are imported or executed; all declarations come from the full
RECORD verifier and must be checked again against the original wheel bytes.
"""

from __future__ import annotations

import json
from pathlib import PurePosixPath
import re
from typing import Any

POLICY_ID = "stage9-v5d5-pyside6-6.10.3-shared-members-v1"
APPROVAL_REQUIRED = "SHARED_WHEEL_MEMBERS_REQUIRES_INDEPENDENT_APPROVAL"
SOURCE_REVIEW_DECISION = "APPROVED_FOR_SHARED_MATERIALIZATION"
WHEELS: dict[str, dict[str, str]] = {
    "pyside6-addons": {
        "version": "6.10.3",
        "wheel_filename": "pyside6_addons-6.10.3-cp39-abi3-manylinux_2_34_x86_64.whl",
        "wheel_sha256": "a9a53a824ba8429953345c89a2dd5dc65d1f1d7e88b6cf1b975a6ef4f4d9760f",
    },
    "pyside6-essentials": {
        "version": "6.10.3",
        "wheel_filename": "pyside6_essentials-6.10.3-cp39-abi3-manylinux_2_34_x86_64.whl",
        "wheel_sha256": "5fb32b48010f513335a2166ac3df279c5996fb96402fd91a4875fc717fe55de0",
    },
}
# Reviewed candidate identity from V5d4; no approval is inferred from this table.
MEMBERS: dict[str, dict[str, Any]] = {
    "PySide6/Qt3DAnimation.pyi": {
        "sha256": "ee3b1c65b2ff4ccfb69db77a2b3e287cffee9ae4b19a483b31914289ac8a8625",
        "size": 30930,
        "mode": 33188,
        "type": "regular",
    },
    "PySide6/Qt3DCore.pyi": {
        "sha256": "abeefd82a06c54e0ceb811e7a6432868fc6f31a930ae0d101b359aa050b5b87f",
        "size": 35578,
        "mode": 33188,
        "type": "regular",
    },
    "PySide6/Qt3DExtras.pyi": {
        "sha256": "df36057ecb02dc736dc69a949bcbd1bcd40bf2a0ceb46b3b26939cb9421b5706",
        "size": 75668,
        "mode": 33188,
        "type": "regular",
    },
    "PySide6/Qt3DInput.pyi": {
        "sha256": "2efae1da887b5a8414c1365a60a93759cfb5511ee454a6150c790451956779e4",
        "size": 23767,
        "mode": 33188,
        "type": "regular",
    },
    "PySide6/Qt3DLogic.pyi": {
        "sha256": "6e12a6c870387d6ed37785dd36347c3a799040a6b28af37ea644f3fde7d22fbf",
        "size": 994,
        "mode": 33188,
        "type": "regular",
    },
    "PySide6/Qt3DRender.pyi": {
        "sha256": "fcf5352dc5d4d1463a589f61493757852e33981e5ac1bf3142f7fa926e08816a",
        "size": 147702,
        "mode": 33188,
        "type": "regular",
    },
    "PySide6/QtBluetooth.pyi": {
        "sha256": "900ff951a4a9df65c654dd5fabf3b0f488ceeaec0e65855c2d6ef0422036a4e8",
        "size": 72822,
        "mode": 33188,
        "type": "regular",
    },
    "PySide6/QtCharts.pyi": {
        "sha256": "a058788c9d2782b02477411c6c42b6fef501a9137ba4ace87048c22cc8835f41",
        "size": 106955,
        "mode": 33188,
        "type": "regular",
    },
    "PySide6/QtConcurrent.pyi": {
        "sha256": "a3f978e025c2f8f850d8c1b979fd47781d509729e58862eda92038ba5b22a3d5",
        "size": 3817,
        "mode": 33188,
        "type": "regular",
    },
    "PySide6/QtCore.pyi": {
        "sha256": "0a8c9dfda27cede5051cabfd582570bf26e1051d5d30bee6114b359627391803",
        "size": 552930,
        "mode": 33188,
        "type": "regular",
    },
    "PySide6/QtDBus.pyi": {
        "sha256": "3172df3ac2b505ebf040ca6160cf4b0a9b650bb3755d5476f6abca06c00e765a",
        "size": 33933,
        "mode": 33188,
        "type": "regular",
    },
    "PySide6/QtDataVisualization.pyi": {
        "sha256": "f3e4259937840185eeb8a8b53bfe01bec2845f79fc5d8aaa34995d023d20d233",
        "size": 114681,
        "mode": 33188,
        "type": "regular",
    },
    "PySide6/QtDesigner.pyi": {
        "sha256": "5f32fc2fc4d4290144936f60e825be6d8820530f8afab883410d78abeb6c5f4e",
        "size": 29943,
        "mode": 33188,
        "type": "regular",
    },
    "PySide6/QtGraphs.pyi": {
        "sha256": "d820798c931b8783cf8a026e43c1f9c9334fb9cff21d03c771bfd0d1296bf82d",
        "size": 154716,
        "mode": 33188,
        "type": "regular",
    },
    "PySide6/QtGraphsWidgets.pyi": {
        "sha256": "02d9fe330ac5aa3749a1a197cc5969ec4bdcb932aba6154e692c10c65adefe29",
        "size": 20508,
        "mode": 33188,
        "type": "regular",
    },
    "PySide6/QtGui.pyi": {
        "sha256": "2a597735578534cb7275510e072731e7728f13ea7f4935ce824eb4a61750a718",
        "size": 625997,
        "mode": 33188,
        "type": "regular",
    },
    "PySide6/QtHelp.pyi": {
        "sha256": "f0cce35edff64b0373ab510a47f8b7a288274fbc2e4aa061082c3020134fac46",
        "size": 16618,
        "mode": 33188,
        "type": "regular",
    },
    "PySide6/QtHttpServer.pyi": {
        "sha256": "17669881470c1da13a83d896d635ea09377d2799695347e9985338ead707db52",
        "size": 15355,
        "mode": 33188,
        "type": "regular",
    },
    "PySide6/QtLocation.pyi": {
        "sha256": "cb059fe355c91bf697f489eec4bbe310e5d499af13e518f91283c160ef4a0428",
        "size": 57311,
        "mode": 33188,
        "type": "regular",
    },
    "PySide6/QtMultimedia.pyi": {
        "sha256": "c81b630413ca09a7e2c60cb379ae0667bf37d2cf5c64e856757088b0c0094d5c",
        "size": 88555,
        "mode": 33188,
        "type": "regular",
    },
    "PySide6/QtMultimediaWidgets.pyi": {
        "sha256": "1af5ec4a8fbe91fc67c64019818e07fbefed506212da23f1864bd2399c7c3d10",
        "size": 3339,
        "mode": 33188,
        "type": "regular",
    },
    "PySide6/QtNetwork.pyi": {
        "sha256": "ab4cca9fd5239ecb811392d90934908b34ee540c2fc4587f3d7ca1d1f8a43769",
        "size": 147030,
        "mode": 33188,
        "type": "regular",
    },
    "PySide6/QtNetworkAuth.pyi": {
        "sha256": "6f6611cd03b7bf2b2831432a878bd69edd49bcc5596861a95936b48616593b76",
        "size": 27670,
        "mode": 33188,
        "type": "regular",
    },
    "PySide6/QtNfc.pyi": {
        "sha256": "8bf5b10d2cfc6832e1eb2ab84084e474ac6a49a1136cc86d3d233dfad1202736",
        "size": 18368,
        "mode": 33188,
        "type": "regular",
    },
    "PySide6/QtOpenGL.pyi": {
        "sha256": "6720bf58471ac3820fa1b166a7c5368352330b2da3bf9fca918a1d8c8e9a7a5f",
        "size": 1206589,
        "mode": 33188,
        "type": "regular",
    },
    "PySide6/QtOpenGLWidgets.pyi": {
        "sha256": "518400b2516f2bca39a554acae2acdbb17f307e38a5d4b0fef2db69b300271c5",
        "size": 3264,
        "mode": 33188,
        "type": "regular",
    },
    "PySide6/QtPdf.pyi": {
        "sha256": "9574a99062e491da8fb7eb14ec3f7f22776b4a5b2fade39fa4eaf73f757d10fa",
        "size": 16191,
        "mode": 33188,
        "type": "regular",
    },
    "PySide6/QtPdfWidgets.pyi": {
        "sha256": "15ec6774a171f740f00a8c3e9ea5ec6e55dc0a8818e56821444d2a8002b93fee",
        "size": 5281,
        "mode": 33188,
        "type": "regular",
    },
    "PySide6/QtPositioning.pyi": {
        "sha256": "b29a7375e8dd366a8437bae4d97f73d4d4d3b2d06e894b31bd0648a35b76356f",
        "size": 38444,
        "mode": 33188,
        "type": "regular",
    },
    "PySide6/QtPrintSupport.pyi": {
        "sha256": "9d953419e558fdf930139c6715fe70c9f384dab2a9104d0ad58ce51092eec5ce",
        "size": 16889,
        "mode": 33188,
        "type": "regular",
    },
    "PySide6/QtQml.pyi": {
        "sha256": "d0bef60b580a75515edeaa9c4c2cdfeb76340bb9da5b455dad071e79f869e20e",
        "size": 51704,
        "mode": 33188,
        "type": "regular",
    },
    "PySide6/QtQuick.pyi": {
        "sha256": "42c4a63e80323e3a9f761bab3a3b6767193be361a15e19317d9ecd5069ad78f2",
        "size": 88751,
        "mode": 33188,
        "type": "regular",
    },
    "PySide6/QtQuick3D.pyi": {
        "sha256": "08d62baeb5a7c3f9de7f77f05c44eaf00ca0d1aa2caccaad20a4aae5eaf0cdb9",
        "size": 14348,
        "mode": 33188,
        "type": "regular",
    },
    "PySide6/QtQuickControls2.pyi": {
        "sha256": "3e07d01d0144d883f78d867f3b93c3383a4363f2f980b281e1285d87bbead28f",
        "size": 1370,
        "mode": 33188,
        "type": "regular",
    },
    "PySide6/QtQuickTest.pyi": {
        "sha256": "b0977987bbfdea5953e871f7b2ddff971d50dea40ef197de28eac18f39e94a01",
        "size": 694,
        "mode": 33188,
        "type": "regular",
    },
    "PySide6/QtQuickWidgets.pyi": {
        "sha256": "d3caa79cd9efd5a9440b737e358e2f423199025337a52a14e3b9f20f4ae7f930",
        "size": 5471,
        "mode": 33188,
        "type": "regular",
    },
    "PySide6/QtRemoteObjects.pyi": {
        "sha256": "7c31cb6bae4f2d7276a9aac7d6008f2fc4798c2c074b66c7bf73f622aa78819e",
        "size": 19499,
        "mode": 33188,
        "type": "regular",
    },
    "PySide6/QtScxml.pyi": {
        "sha256": "e7a769fa9637759e3dc0b311ed9dedc936320089399686e86ceb5c6cf9da7e5c",
        "size": 17522,
        "mode": 33188,
        "type": "regular",
    },
    "PySide6/QtSensors.pyi": {
        "sha256": "6233707ce0df307ef6facba50b54378552bd69a98b59e866d9e7f27e257e7fbf",
        "size": 27936,
        "mode": 33188,
        "type": "regular",
    },
    "PySide6/QtSerialBus.pyi": {
        "sha256": "27597e0c768cbed3c98cdcef83b06c1197f7b9c514c954d2fd990664a8548e2f",
        "size": 37587,
        "mode": 33188,
        "type": "regular",
    },
    "PySide6/QtSerialPort.pyi": {
        "sha256": "105ae6fe2c88c1f678b7e660b3b1063dd76058897568c81656a159073f6dbbfa",
        "size": 9541,
        "mode": 33188,
        "type": "regular",
    },
    "PySide6/QtSpatialAudio.pyi": {
        "sha256": "8f1bcfdd5513c93520ebc2f455b89d22c5683fa80b64508c5e9c8db416a5dd6b",
        "size": 12209,
        "mode": 33188,
        "type": "regular",
    },
    "PySide6/QtSql.pyi": {
        "sha256": "44ed3a1e86b8e131516261b03f9fadca708de3c7b299188fa8bd400dda81f77b",
        "size": 34228,
        "mode": 33188,
        "type": "regular",
    },
    "PySide6/QtStateMachine.pyi": {
        "sha256": "a136d181e090e6d409c3609e6714671a63cb99ec57a8099a7a65950a1265d4af",
        "size": 17056,
        "mode": 33188,
        "type": "regular",
    },
    "PySide6/QtSvg.pyi": {
        "sha256": "1d2f7b7f0e49286232a2d269638d0834def5131afbf0b0d7189824c5f111489a",
        "size": 6746,
        "mode": 33188,
        "type": "regular",
    },
    "PySide6/QtSvgWidgets.pyi": {
        "sha256": "04d9ae9008f7b3248e5a7b8ececff8573ca2cd9b99049e41bcac91fe937cc45c",
        "size": 2574,
        "mode": 33188,
        "type": "regular",
    },
    "PySide6/QtTest.pyi": {
        "sha256": "b6d253062f958528635e0312f5cf381a18e3cae9f05f1dccd608d64a33c52da2",
        "size": 21983,
        "mode": 33188,
        "type": "regular",
    },
    "PySide6/QtTextToSpeech.pyi": {
        "sha256": "dced47228b32ec1bad29320d4a1f58229ff63852b0cbdda411a05088db546623",
        "size": 9852,
        "mode": 33188,
        "type": "regular",
    },
    "PySide6/QtUiTools.pyi": {
        "sha256": "86f1dbf95fbb6d64f8cbdc319eb83a633bea08e64e2d7a10f99c01f323a41619",
        "size": 2347,
        "mode": 33188,
        "type": "regular",
    },
    "PySide6/QtWebChannel.pyi": {
        "sha256": "ca69e4cf3879fd211e6cb303f648fdba7f610fddad6d399cba3128ce59a57752",
        "size": 1948,
        "mode": 33188,
        "type": "regular",
    },
    "PySide6/QtWebEngineCore.pyi": {
        "sha256": "bcef577be7b4d63fb9a7c1336c54c5ccbe58c501411a6d2cf2d1f9f30d5a1205",
        "size": 72637,
        "mode": 33188,
        "type": "regular",
    },
    "PySide6/QtWebEngineQuick.pyi": {
        "sha256": "147d416bc53c50e252ba2993713579d310317c3b17229fdc41de1218e6ff638a",
        "size": 9035,
        "mode": 33188,
        "type": "regular",
    },
    "PySide6/QtWebEngineWidgets.pyi": {
        "sha256": "f77a57e20d1e40c093cfaa4d6284533b746d62e96424c6a5ad18c58054db2dbf",
        "size": 6359,
        "mode": 33188,
        "type": "regular",
    },
    "PySide6/QtWebSockets.pyi": {
        "sha256": "d5016bf161daef5272b43d767e16ae3146c1d03fe8ffbab460c881dc37fb3bd5",
        "size": 12630,
        "mode": 33188,
        "type": "regular",
    },
    "PySide6/QtWebView.pyi": {
        "sha256": "1da437dc95e285037fe9c02e785f6cf58847f07f3076b08a65f34c7c7b0f3dfd",
        "size": 548,
        "mode": 33188,
        "type": "regular",
    },
    "PySide6/QtWidgets.pyi": {
        "sha256": "88691b6e26b78707e39d2279911da76b9be2aeb76573abb0f85122bba950ca39",
        "size": 603070,
        "mode": 33188,
        "type": "regular",
    },
    "PySide6/QtXml.pyi": {
        "sha256": "9eb600340820b022d2959d03a723d1423655ba12b624f6578339cf79f0061c5c",
        "size": 20166,
        "mode": 33188,
        "type": "regular",
    },
    "PySide6/__feature__.pyi": {
        "sha256": "3f3f79ca62c186e1c5d07b228a6db57809c184804449426bda4cd31f269072e5",
        "size": 246,
        "mode": 33204,
        "type": "regular",
    },
    "PySide6/__init__.py": {
        "sha256": "fe42fff5fcd2e9291495519bd068660a2861173bd106e5f145d8b6f72c1281fe",
        "size": 6220,
        "mode": 33188,
        "type": "regular",
    },
    "PySide6/_config.py": {
        "sha256": "216206d44725c34fbb9add3ad00e6df452bc3f4eec2e923a0f716607a438d071",
        "size": 823,
        "mode": 33188,
        "type": "regular",
    },
    "PySide6/_git_pyside_version.py": {
        "sha256": "bac9879d5cf3f32d412f1e94254d1a1b6b6ed2959841519fda9d5645cc561c7b",
        "size": 693,
        "mode": 33188,
        "type": "regular",
    },
    "PySide6/py.typed": {
        "sha256": "154d2b3a1255946ec4511a7952fdb07cfbe01a6ee9c7c6eae5dfabd460c833c1",
        "size": 33,
        "mode": 33204,
        "type": "regular",
    },
}


class Refusal(ValueError):
    """Any incomplete or mismatched sharing declaration is refused."""


def require(condition: bool, reason: str) -> None:
    if not condition:
        raise Refusal(reason)


def exact(left: Any, right: Any) -> bool:
    """JSON types are significant: bool/int/float aliases never approve bytes."""
    try:
        return json.dumps(left, sort_keys=True, allow_nan=False) == json.dumps(
            right, sort_keys=True, allow_nan=False
        )
    except (TypeError, ValueError):
        return False


def safe_path(value: Any) -> str:
    require(isinstance(value, str), "SHARED_WHEEL_MEMBER_PATH")
    assert isinstance(value, str)
    path = PurePosixPath(value)
    require(
        bool(path.parts)
        and not path.is_absolute()
        and path.as_posix() == value
        and ".." not in path.parts
        and "\\" not in value
        and ":" not in value
        and all(ord(char) >= 32 and ord(char) != 127 for char in value),
        "SHARED_WHEEL_MEMBER_PATH",
    )
    return value


def collision_inventory(packages: list[dict[str, Any]]) -> dict[str, Any]:
    """List exact candidates without granting sharing or source approval.

    Every owner is collected before comparing the *entire* collision set with
    policy. Byte equality alone has no authorization meaning. Within-wheel
    aliases and file/directory conflicts can never become shared candidates.
    """
    require(type(packages) is list, "SHARED_WHEEL_PACKAGE_SCHEMA")
    declarations: dict[str, list[dict[str, Any]]] = {}
    identities: set[str] = set()
    for package in packages:
        require(type(package) is dict, "SHARED_WHEEL_PACKAGE_SCHEMA")
        require(
            all(
                isinstance(package.get(field), str) and bool(package[field])
                for field in ("name", "version", "filename", "sha256")
            )
            and re.fullmatch(r"[0-9a-f]{64}", package["sha256"]) is not None
            and type(package.get("materialized_members")) is dict
            and type(package.get("member_metadata")) is dict,
            "SHARED_WHEEL_PACKAGE_SCHEMA",
        )
        name = package["name"]
        require(name not in identities, "SHARED_WHEEL_DUPLICATE_DISTRIBUTION")
        identities.add(name)
        local: set[str] = set()
        for member, entry in package["materialized_members"].items():
            require(
                type(entry) is dict
                and set(entry) == {"path", "sha256"}
                and isinstance(entry["sha256"], str)
                and re.fullmatch(r"[0-9a-f]{64}", entry["sha256"]) is not None,
                "SHARED_WHEEL_MEMBER_SCHEMA",
            )
            path = safe_path(entry["path"])
            safe_path(member)
            metadata = package["member_metadata"].get(member)
            require(
                type(metadata) is dict
                and set(metadata) == {"mode", "size", "type"}
                and type(metadata["mode"]) is int
                and metadata["mode"] >= 0
                and type(metadata["size"]) is int
                and metadata["size"] >= 0
                and metadata["type"] == "regular",
                "SHARED_WHEEL_MEMBER_METADATA_SCHEMA",
            )
            require(path not in local, "WHEEL_MATERIALIZATION_COLLISION:" + path)
            local.add(path)
            declarations.setdefault(path, []).append(
                {
                    "owner": {
                        "distribution": name,
                        "version": package["version"],
                        "wheel_filename": package["filename"],
                        "wheel_sha256": package["sha256"],
                        "member": member,
                    },
                    "sha256": entry["sha256"],
                    "metadata": metadata,
                }
            )
    for path in declarations:
        require(
            not any(
                parent.as_posix() in declarations
                for parent in PurePosixPath(path).parents
                if parent.parts
            ),
            "WHEEL_MATERIALIZATION_COLLISION:" + path,
        )
    collisions = {path: owners for path, owners in declarations.items() if len(owners) > 1}
    if collisions or set(WHEELS) <= identities:
        require(set(collisions) == set(MEMBERS), "SHARED_WHEEL_MEMBER_SET_MISMATCH")
    members: list[dict[str, Any]] = []
    for path in sorted(collisions):
        declarations_for_path = collisions[path]
        pin = MEMBERS[path]
        owners = sorted(
            (declaration["owner"] for declaration in declarations_for_path),
            key=lambda owner: (owner["distribution"], owner["wheel_filename"], owner["member"]),
        )
        expected_owners = [
            {"distribution": name, **identity, "member": path}
            for name, identity in sorted(WHEELS.items())
        ]
        require(exact(owners, expected_owners), "SHARED_WHEEL_OWNER_MISMATCH:" + path)
        require(
            all(
                declaration["sha256"] == pin["sha256"]
                and exact(
                    declaration["metadata"],
                    {"size": pin["size"], "mode": pin["mode"], "type": pin["type"]},
                )
                for declaration in declarations_for_path
            ),
            "SHARED_WHEEL_CONTENT_OR_MODE_MISMATCH:" + path,
        )
        members.append({"path": path, **pin, "owners": owners})
    return {
        "policy": POLICY_ID,
        "status": APPROVAL_REQUIRED if members else "NO_SHARED_WHEEL_MEMBERS",
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
    """Compare caller's independent expectation, never manufacture approval.

    The caller must authenticate the separate expectation and its origin. A
    successful local comparison does not authenticate a reviewer or authorize
    native loading, launch, Stage 9 qualification, or production.
    """
    inventory = collision_inventory(packages)
    if not inventory["members"] and approved_shared is None:
        return inventory
    require(isinstance(approved_shared, dict), APPROVAL_REQUIRED)
    assert approved_shared is not None
    require(
        set(approved_shared) == {"policy", "members", "source_reviews"}
        and approved_shared["policy"] == POLICY_ID
        and exact(approved_shared["members"], inventory["members"]),
        "SHARED_WHEEL_EXPECTATION_MISMATCH",
    )
    reviews = approved_shared["source_reviews"]
    require(isinstance(reviews, list), "SHARED_WHEEL_PYTHON_SOURCE_REVIEW_REQUIRED")
    require(
        len(reviews) == len(inventory["required_source_reviews"]),
        "SHARED_WHEEL_PYTHON_SOURCE_REVIEW_REQUIRED",
    )
    for review, source in zip(reviews, inventory["required_source_reviews"], strict=True):
        require(
            isinstance(review, dict)
            and set(review) == {"path", "sha256", "decision", "review_reference"}
            and exact(
                {key: review[key] for key in ("path", "sha256", "decision")},
                {**source, "decision": SOURCE_REVIEW_DECISION},
            )
            and isinstance(review["review_reference"], str)
            and bool(review["review_reference"].strip()),
            "SHARED_WHEEL_PYTHON_SOURCE_REVIEW_REQUIRED",
        )
    return inventory

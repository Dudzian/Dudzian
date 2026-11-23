"""Synthetic V5d5 wheels; sharing approval is TEST ONLY and never emitted by tools."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import zipfile

import pytest

from scripts import stage9_v5d4_dependencies as deps
from scripts import stage9_v5d5_shared as shared
from tests.deployment.test_stage9_v5d4_dependencies import wheel

pytestmark = [pytest.mark.security, pytest.mark.unit]


@pytest.fixture
def shared_material(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple:
    project = tmp_path / "pyproject.toml"
    project.write_text(
        '[project]\nrequires-python=">=3.11"\n'
        'dependencies=["PySide6-Essentials==6.10.3","PySide6-Addons==6.10.3"]\n'
        "[project.optional-dependencies]\nobserver-kvm-preflight=[]\n",
        encoding="utf-8",
    )
    wheelhouse = tmp_path / "wheels"
    wheelhouse.mkdir()
    content = {name: ("TEST ONLY inert source: " + name).encode() for name in shared.MEMBERS}
    members = {
        name: {**pin, "sha256": deps.digest(content[name]), "size": len(content[name])}
        for name, pin in shared.MEMBERS.items()
    }
    identities = copy.deepcopy(shared.WHEELS)
    origins = {}
    for distribution in sorted(identities):
        path, origin = wheel(
            wheelhouse,
            distribution.replace("-", "_"),
            "6.10.3",
            tag="cp39-abi3-manylinux_2_34_x86_64",
            files=content,
        )
        with zipfile.ZipFile(path) as archive:
            entries = [(info, archive.read(info)) for info in archive.infolist()]
        with zipfile.ZipFile(path, "w") as archive:
            for info, data in entries:
                if info.filename in members:
                    info.external_attr = members[info.filename]["mode"] << 16
                archive.writestr(info, data)
        origin["sha256"] = deps.digest(path.read_bytes())
        origins[path.name] = origin
        identities[distribution]["wheel_sha256"] = origin["sha256"]
    monkeypatch.setattr(shared, "WHEELS", identities)
    monkeypatch.setattr(shared, "MEMBERS", members)
    monkeypatch.setattr(deps, "shared_tools", lambda: vars(shared))
    proposal = deps.propose(project, wheelhouse, origins, allow_shared_candidates=True)
    inventory = proposal["shared_wheel_members"]
    # Explicit synthetic reviewer input, deliberately confined to this test fixture.
    expectation = {
        "policy": shared.POLICY_ID,
        "members": copy.deepcopy(inventory["members"]),
        "source_reviews": [
            {
                **source,
                "decision": shared.SOURCE_REVIEW_DECISION,
                "review_reference": "TEST ONLY independent reviewer fixture",
            }
            for source in inventory["required_source_reviews"]
        ],
    }
    return project, wheelhouse, origins, proposal, expectation


def test_policy_exactly_matches_all_62_v5d4_evidence_rows() -> None:
    evidence = json.loads(
        Path("docs/architecture/evidence/stage9_v5d4_wheel_bytecode_inspection.json").read_text()
    )
    rows = evidence["shared_member_collisions"]
    assert len(rows) == len(shared.MEMBERS) == 62
    assert shared.MEMBERS == {
        row["member"]: {
            "sha256": row["sha256"],
            "size": row["size"],
            "mode": int(row["mode"], 8),
            "type": "regular",
        }
        for row in rows
    }
    assert sum(path.endswith(".pyi") for path in shared.MEMBERS) == 58
    assert sum(path.endswith(".py") for path in shared.MEMBERS) == 3


def test_exact_62_candidates_need_separate_independent_approval(shared_material) -> None:
    project, directory, origins, proposal, expectation = shared_material
    inventory = proposal["shared_wheel_members"]
    assert len(inventory["members"]) == 62
    assert inventory["status"] == shared.APPROVAL_REQUIRED
    assert len(inventory["required_source_reviews"]) == 3
    assert "source_reviews" not in inventory
    assert all(len(row["owners"]) == 2 for row in inventory["members"])
    with pytest.raises(deps.Refusal, match="WHEEL_MATERIALIZATION_COLLISION"):
        deps.propose(project, directory, origins)
    with pytest.raises(deps.Refusal, match=shared.APPROVAL_REQUIRED):
        deps.verify(
            project, directory, origins, proposal, deps.digest(deps.canonical(proposal)), {}
        )
    verification = deps.verify(
        project,
        directory,
        origins,
        proposal,
        deps.digest(deps.canonical(proposal)),
        {},
        approved_shared=expectation,
    )
    assert verification["authorization"] == "NOT_AUTHORIZED"
    assert verification["shared_wheel_members"] == inventory
    assert verification["shared_wheel_expectations"] == expectation
    assert verification["shared_wheel_policy"] == shared.POLICY_ID


def test_wheel_order_never_changes_inventory_or_canonical_owners(shared_material) -> None:
    _, _, _, proposal, expectation = shared_material
    original = shared.verify_shared(proposal["packages"], expectation)
    assert shared.verify_shared(list(reversed(proposal["packages"])), expectation) == original
    assert original["members"] == sorted(original["members"], key=lambda row: row["path"])
    for row in original["members"]:
        assert row["owners"] == sorted(row["owners"], key=lambda owner: owner["distribution"])
        assert all(owner["member"] == row["path"] for owner in row["owners"])


@pytest.mark.parametrize("change", ["bytes", "mode", "size", "type", "wheel_sha", "version"])
def test_every_original_member_and_wheel_identity_is_bound(shared_material, change: str) -> None:
    _, _, _, proposal, expectation = shared_material
    packages = copy.deepcopy(proposal["packages"])
    first = packages[0]
    member = next(iter(shared.MEMBERS))
    if change == "bytes":
        first["materialized_members"][member]["sha256"] = "0" * 64
    elif change in {"mode", "size"}:
        first["member_metadata"][member][change] += 1
    elif change == "type":
        first["member_metadata"][member]["type"] = "symlink"
    elif change == "wheel_sha":
        first["sha256"] = "0" * 64
    else:
        first["version"] = "6.10.4"
    with pytest.raises(shared.Refusal):
        shared.verify_shared(packages, expectation)


def test_third_unapproved_owner_refuses_even_identical_bytes(shared_material) -> None:
    _, _, _, proposal, expectation = shared_material
    packages = copy.deepcopy(proposal["packages"])
    third = copy.deepcopy(packages[0])
    third["name"] = "third-owner"
    third["materialized_members"] = {
        member: entry
        for member, entry in third["materialized_members"].items()
        if member in shared.MEMBERS
    }
    packages.append(third)
    with pytest.raises(shared.Refusal, match="SHARED_WHEEL_OWNER_MISMATCH"):
        shared.verify_shared(packages, expectation)


@pytest.mark.parametrize("change", ["missing", "extra", "no_shared_files"])
def test_complete_shared_set_is_mandatory(shared_material, change: str) -> None:
    _, _, _, proposal, expectation = shared_material
    packages = copy.deepcopy(proposal["packages"])
    member = next(iter(shared.MEMBERS))
    for package in packages:
        if change == "missing":
            package["materialized_members"].pop(member)
        elif change == "no_shared_files":
            for path in shared.MEMBERS:
                package["materialized_members"].pop(path)
        else:
            package["materialized_members"]["PySide6/additional.pyi"] = {
                "path": "PySide6/additional.pyi",
                "sha256": "0" * 64,
            }
            package["member_metadata"]["PySide6/additional.pyi"] = {
                "mode": 33188,
                "size": 0,
                "type": "regular",
            }
    with pytest.raises(shared.Refusal, match="SHARED_WHEEL_MEMBER_SET_MISMATCH"):
        shared.verify_shared(packages, expectation)


@pytest.mark.parametrize(
    "change",
    ["member", "owner", "policy", "field", "source", "source_reference", "source_decision"],
)
def test_incomplete_or_unreviewed_expectations_refuse(shared_material, change: str) -> None:
    _, _, _, proposal, approved = shared_material
    expectation = copy.deepcopy(approved)
    if change == "member":
        expectation["members"].pop()
    elif change == "owner":
        expectation["members"][0]["owners"].pop()
    elif change == "policy":
        expectation["policy"] += "-other"
    elif change == "field":
        expectation["unexpected"] = True
    elif change == "source":
        expectation["source_reviews"].pop()
    elif change == "source_reference":
        expectation["source_reviews"][0]["review_reference"] = "   "
    else:
        expectation["source_reviews"][0]["decision"] = "NOT_REVIEWED"
    with pytest.raises(shared.Refusal):
        shared.verify_shared(proposal["packages"], expectation)


def test_extra_source_and_startup_content_still_follow_original_record_policy(
    shared_material,
) -> None:
    _, directory, origins, _, _ = shared_material
    path = next(directory.glob("pyside6_addons*.whl"))
    with zipfile.ZipFile(path) as archive:
        original_record = archive.read("pyside6_addons-6.10.3.dist-info/RECORD")
    with zipfile.ZipFile(path, "a") as archive:
        archive.writestr("PySide6/extra.pth", b"import injected\n")
    origins[path.name]["sha256"] = deps.digest(path.read_bytes())
    with pytest.raises(deps.Refusal, match="WHEEL_RECORD_INCOMPLETE"):
        deps.inspect_wheel(path, origins[path.name])
    with zipfile.ZipFile(path) as archive:
        assert archive.read("pyside6_addons-6.10.3.dist-info/RECORD") == original_record

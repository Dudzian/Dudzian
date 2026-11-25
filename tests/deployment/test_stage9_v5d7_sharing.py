"""V5d7 sharing declarations and approval boundaries, without wheel execution."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import runpy
import shutil
from types import SimpleNamespace
from typing import Any
import zipfile

import pytest

from scripts import stage9_v5d5_shared as legacy
from scripts import stage9_v5d7_shared as shared
from scripts import stage9_v5d4_dependencies as deps
from scripts import stage9_v5d4_environment as env
from scripts import stage9_v5d4_launch as launch
from scripts import stage9_v5d7_omissions as omissions
from tests.deployment.test_stage9_v5d4_dependencies import wheel
from tests.deployment.test_stage9_v5d4_environment import staging as staging

pytestmark = [pytest.mark.security, pytest.mark.unit]
TARGET = "PySide6/Qt3DAnimation.pyi"


@pytest.fixture
def declarations() -> list[dict[str, Any]]:
    """In-memory inventory for exercising the contract independently of ZIP reads."""
    return [
        {
            "name": name,
            "version": identity["version"],
            "filename": identity["wheel_filename"],
            "sha256": identity["wheel_sha256"],
            "materialized_members": {
                path: {"path": path, "sha256": member["sha256"]}
                for path, member in shared.MEMBERS.items()
            },
            "member_metadata": {
                path: {field: member[field] for field in ("type", "mode", "size")}
                for path, member in shared.MEMBERS.items()
            },
        }
        for name, identity in sorted(shared.WHEELS.items())
    ]


def test_three_owners_and_complete_set_require_separate_approval(declarations):
    inventory = shared.collision_inventory(declarations)
    assert inventory["policy"] == shared.POLICY_ID
    assert inventory["status"] == shared.APPROVAL_REQUIRED
    assert len(inventory["members"]) == 62
    assert len(inventory["required_source_reviews"]) == 3
    assert "source_reviews" not in inventory
    assert shared.collision_inventory(list(reversed(declarations))) == inventory
    assert all(len(row["owners"]) == 3 for row in inventory["members"])
    assert all(
        owner["member"] == row["path"] for row in inventory["members"] for owner in row["owners"]
    )
    with pytest.raises(ValueError, match=shared.APPROVAL_REQUIRED):
        shared.verify_shared(declarations, None)


@pytest.mark.parametrize(
    ("change", "reason"),
    [
        ("missing_archive", "SHARED_WHEEL_OWNER_SET_MISMATCH"),
        ("missing_owner", "SHARED_WHEEL_OWNER_MISMATCH"),
        ("fourth_owner", "SHARED_WHEEL_OWNER_MISMATCH"),
        ("wheel_hash", "SHARED_WHEEL_OWNER_MISMATCH"),
        ("member_hash", "SHARED_WHEEL_CONTENT_OR_MODE_MISMATCH"),
        ("mode", "SHARED_WHEEL_CONTENT_OR_MODE_MISMATCH"),
        ("size", "SHARED_WHEEL_CONTENT_OR_MODE_MISMATCH"),
        ("type", "SHARED_WHEEL_MEMBER_METADATA_SCHEMA"),
        ("source_path", "SHARED_WHEEL_OWNER_MISMATCH"),
        ("same_archive_alias", "WHEEL_MATERIALIZATION_COLLISION"),
        ("file_directory_collision", "WHEEL_MATERIALIZATION_COLLISION"),
        ("extra_collision", "SHARED_WHEEL_MEMBER_SET_MISMATCH"),
    ],
)
def test_changed_archive_or_member_is_refused(declarations, change, reason):
    primary = declarations[0]
    entries = primary["materialized_members"]
    metadata = primary["member_metadata"]
    alias = "pyside6-6.10.3.data/purelib/" + TARGET
    if change == "missing_archive":
        declarations.pop(0)
    elif change == "missing_owner":
        entries.pop(TARGET)
    elif change == "fourth_owner":
        declarations.append(copy.deepcopy(primary) | {"name": "unrelated"})
    elif change == "wheel_hash":
        primary["sha256"] = "0" * 64
    elif change == "member_hash":
        entries[TARGET]["sha256"] = "0" * 64
    elif change == "mode":
        metadata[TARGET]["mode"] = 0o100755
    elif change == "size":
        metadata[TARGET]["size"] += 1
    elif change == "type":
        metadata[TARGET]["type"] = "symlink"
    elif change == "source_path":
        entries[alias] = entries.pop(TARGET)
        metadata[alias] = metadata.pop(TARGET)
    elif change == "same_archive_alias":
        entries[alias] = dict(entries[TARGET])
        metadata[alias] = dict(metadata[TARGET])
    elif change == "file_directory_collision":
        entries["PySide6"] = {"path": "PySide6", "sha256": "0" * 64}
        metadata["PySide6"] = {"type": "regular", "mode": 0o100644, "size": 1}
    else:
        for package in declarations[:2]:
            package["materialized_members"]["extra.py"] = {"path": "extra.py", "sha256": "0" * 64}
            package["member_metadata"]["extra.py"] = {
                "type": "regular",
                "mode": 0o100644,
                "size": 1,
            }
    with pytest.raises(ValueError, match="^" + reason):
        shared.collision_inventory(declarations)


def approval(tools, inventory):
    return {
        "policy": tools.POLICY_ID,
        "members": copy.deepcopy(inventory["members"]),
        "source_reviews": [
            {
                **row,
                "decision": tools.SOURCE_REVIEW_DECISION,
                "review_reference": "TEST ONLY independent reviewer fixture",
            }
            for row in inventory["required_source_reviews"]
        ],
    }


def test_legacy_pair_approval_cannot_expand_to_three_owners(declarations):
    legacy_declarations = [row for row in declarations if row["name"] in legacy.WHEELS]
    old_inventory = legacy.collision_inventory(legacy_declarations)
    old_approval = approval(legacy, old_inventory)
    assert legacy.verify_shared(legacy_declarations, old_approval) == old_inventory
    with pytest.raises(ValueError, match="^SHARED_WHEEL_EXPECTATION_MISMATCH$"):
        shared.verify_shared(declarations, old_approval)
    new_inventory = shared.collision_inventory(declarations)
    new_approval = approval(shared, new_inventory)
    assert shared.verify_shared(declarations, new_approval) == new_inventory
    with pytest.raises(ValueError, match="^SHARED_WHEEL_OWNER_MISMATCH"):
        legacy.verify_shared(declarations, new_approval)
    with pytest.raises(ValueError, match="^SHARED_WHEEL_EXPECTATION_MISMATCH$"):
        legacy.verify_shared(legacy_declarations, new_approval)
    assert old_inventory["required_source_reviews"] == new_inventory["required_source_reviews"]
    new_approval["source_reviews"].pop()
    with pytest.raises(ValueError, match="^SHARED_WHEEL_PYTHON_SOURCE_REVIEW_REQUIRED$"):
        shared.verify_shared(declarations, new_approval)


def test_receipt_audit_uses_packaged_fixed_baseline(tmp_path: Path, declarations):
    target = tmp_path / shared.AUDIT_FILENAME
    shutil.copyfile(shared.__file__, target)
    baseline = tmp_path / "shared-audit.py"
    shutil.copyfile(legacy.__file__, baseline)
    audit = runpy.run_path(str(target))
    inventory = shared.collision_inventory(declarations)
    assert audit["verify_shared"](declarations, approval(shared, inventory)) == inventory
    baseline.write_text(
        baseline.read_text(encoding="utf-8") + f"\nMEMBERS.pop({TARGET!r})\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="^SHARED_WHEEL_BASELINE_PINS_MISMATCH$"):
        runpy.run_path(str(target))


@pytest.fixture
def three_wheels(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    project = tmp_path / "three-project.toml"
    project.write_text(
        '[project]\nrequires-python=">=3.11"\n'
        'dependencies=["PySide6==6.10.3","PySide6-Essentials==6.10.3","PySide6-Addons==6.10.3"]\n'
        "[project.optional-dependencies]\nobserver-kvm-preflight=[]\n",
        encoding="utf-8",
    )
    directory = tmp_path / "three-wheels"
    directory.mkdir()
    contents = {path: ("TEST ONLY inert bytes for " + path).encode() for path in shared.MEMBERS}
    members = {
        path: {"sha256": deps.digest(raw), "size": len(raw), "mode": 0o100644, "type": "regular"}
        for path, raw in contents.items()
    }
    wheels = copy.deepcopy(shared.WHEELS)
    origins = {}
    for name, identity in wheels.items():
        path, origin = wheel(
            directory,
            name.replace("-", "_"),
            "6.10.3",
            tag="cp39-abi3-manylinux_2_34_x86_64",
            files=contents,
            include_package_init=False,
        )
        assert path.name == identity["wheel_filename"]
        origins[path.name] = origin
        identity["wheel_sha256"] = origin["sha256"]
    monkeypatch.setattr(shared, "MEMBERS", members)
    monkeypatch.setattr(shared, "WHEELS", wheels)
    # The legacy subset has its own test pins, so regressions reach the owner checks.
    monkeypatch.setattr(legacy, "MEMBERS", members)
    monkeypatch.setattr(legacy, "WHEELS", {key: wheels[key] for key in legacy.WHEELS})
    monkeypatch.setattr(
        deps, "shared_tools", lambda *, v5d7=False: vars(shared if v5d7 else legacy)
    )
    return project, directory, origins


def test_proposal_rebuild_requires_v5d7_digest_and_reviews(three_wheels):
    project, directory, origins = three_wheels
    with pytest.raises(deps.Refusal, match="^SHARED_WHEEL_OWNER_MISMATCH"):
        deps.propose(project, directory, origins, allow_shared_candidates=True)
    proposal = deps.propose(project, directory, origins, v5d7=True)
    assert proposal["profile"] == deps.V5D7_PROPOSAL
    assert proposal["authorization"] == "NOT_AUTHORIZED"
    inventory = proposal["shared_wheel_members"]
    expectation = approval(shared, inventory)
    sha256 = deps.digest(deps.canonical(proposal))
    with pytest.raises(deps.Refusal, match="^PROPOSAL_SHA256_MISMATCH$"):
        deps.verify(
            project, directory, origins, proposal, "0" * 64, {}, approved_shared=expectation
        )
    with pytest.raises(deps.Refusal, match="^" + shared.APPROVAL_REQUIRED + "$"):
        deps.verify(project, directory, origins, proposal, sha256, {})
    result = deps.verify(
        project, directory, origins, proposal, sha256, {}, approved_shared=expectation
    )
    assert result["profile"] == deps.V5D7_VERIFICATION
    assert result["proposal_sha256"] == sha256
    assert result["authorization"] == "NOT_AUTHORIZED"
    assert result["shared_wheel_expectations"] == expectation
    assert result["shared_wheel_policy"] == shared.POLICY_ID
    assert result["artifact_omissions"] == proposal["artifact_omissions"]
    assert result["artifact_expectations"] is None
    assert result["artifact_policy"] == proposal["artifact_omissions"]["policy"]


@pytest.fixture
def three_wheel_staging(staging, three_wheels, monkeypatch):
    inputs, approved, output, _ = staging
    project, directory, origins = three_wheels
    proposal = deps.propose(project, directory, origins, v5d7=True)
    (Path(inputs["repository"]) / "pyproject.toml").write_bytes(project.read_bytes())
    inputs["wheelhouse"] = str(directory)
    Path(inputs["origins"]).write_text(json.dumps(origins), encoding="utf-8")
    Path(inputs["proposal"]).write_text(json.dumps(proposal), encoding="utf-8")
    approved["proposal_sha256"] = deps.digest(deps.canonical(proposal))
    approved["wheel_shared_members"] = approval(shared, proposal["shared_wheel_members"])
    original_helper = env.helper
    closure = original_helper("stage9_v5d4_closure.py")["inventory"](Path(inputs["repository"]))
    approved["dynamic_review_sha256"] = deps.digest(deps.canonical(closure))
    modules = {
        "stage9_v5d4_dependencies.py": deps,
        "stage9_v5d7_shared.py": shared,
        "stage9_v5d7_omissions.py": omissions,
    }

    def helper(name):
        return vars(modules[name]) if name in modules else original_helper(name)

    monkeypatch.setattr(env, "helper", helper)
    real_run = runpy.run_path
    synthetic_tools = SimpleNamespace(
        run_path=lambda name: vars(shared)
        if Path(name).name in {"stage9_v5d7_shared.py", shared.AUDIT_FILENAME}
        else real_run(name)
    )
    monkeypatch.setattr(env, "runpy", synthetic_tools)
    monkeypatch.setattr(launch, "runpy", synthetic_tools)
    return inputs, approved, output, proposal


def test_builder_and_launcher_bind_three_owners_without_start(three_wheel_staging, monkeypatch):
    inputs, approved, output, proposal = three_wheel_staging
    writes: list[str] = []
    original_write = env.write_new

    def track(root, name, raw):
        writes.append(name)
        original_write(root, name, raw)

    monkeypatch.setattr(env, "write_new", track)
    receipt = env.build(inputs, approved, output)
    assert receipt["profile"] == "stage9-v5d7-prepared-environment-v1"
    assert receipt["authorization"] == "NOT_AUTHORIZED"
    assert receipt["native_load"] == "NOT_EXECUTED"
    assert {"shared-audit.py", shared.AUDIT_FILENAME, "artifact-audit.py"} <= set(receipt["files"])
    assert all(writes.count(path) == 1 for path in shared.MEMBERS)
    for package in proposal["packages"]:
        path = Path(inputs["wheelhouse"]) / package["filename"]
        assert deps.digest(path.read_bytes()) == package["sha256"]
        with zipfile.ZipFile(path) as archive:
            record = next(name for name in archive.namelist() if name.endswith(".dist-info/RECORD"))
            assert (output / "packages" / record).read_bytes() == archive.read(record)
    launch.verify_tree(output, receipt["files"], receipt["directories"])
    launch.verify_shared_receipt(output, receipt)
    launch.verify_artifact_receipt(output, receipt)
    assert launch.main([]) == 2


def test_materializer_requires_same_independent_inventory(three_wheel_staging, tmp_path):
    inputs, approved, _, proposal = three_wheel_staging
    package_set = proposal["packages"]
    wheelhouse = Path(inputs["wheelhouse"])
    approved_shared = approved["wheel_shared_members"]
    snapshots, _, inventory = env.prepare_wheel_set(
        package_set, wheelhouse, approved_shared, v5d7=True
    )
    destination = tmp_path / "checked-snapshots"
    destination.mkdir()
    with pytest.raises(ValueError, match="V5D7_WHEEL_SET_REQUIRED"):
        env.materialize_wheel_set(snapshots, destination, inventory)
    with pytest.raises(ValueError, match=shared.APPROVAL_REQUIRED):
        env.materialize_wheel_set(snapshots, destination, inventory, v5d7=True)
    forged = copy.deepcopy(inventory)
    forged["members"][0]["owners"].pop()
    with pytest.raises(ValueError, match="SHARED_WHEEL_MATERIALIZATION_PLAN_CHANGED"):
        env.materialize_wheel_set(
            snapshots, destination, forged, v5d7=True, approved_shared=approved_shared
        )
    assert not list(destination.iterdir())


@pytest.mark.parametrize("family", ["shared", "artifact"])
@pytest.mark.parametrize("mutation", ["audit", "verification", "profile"])
def test_receipt_cannot_downgrade_approval_or_audit(three_wheel_staging, family, mutation):
    inputs, approved, output, _ = three_wheel_staging
    receipt = env.build(inputs, approved, output)
    if family == "shared":
        verifier = launch.verify_shared_receipt
        field = "shared_wheel_expectations"
        audit = shared.AUDIT_FILENAME
    else:
        verifier = launch.verify_artifact_receipt
        field = "artifact_expectations"
        audit = "artifact-audit.py"
    if mutation == "audit":
        receipt["files"].pop(audit)
    elif mutation == "verification":
        receipt["dependency_verification"].pop(field)
    else:
        receipt["profile"] = "stage9-v5d5-prepared-environment-v1"
    with pytest.raises(ValueError):
        verifier(output, receipt)

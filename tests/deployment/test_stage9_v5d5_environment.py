"""V5d5 materialization uses inert TEST ONLY wheels, never native imports."""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import runpy
import stat
from types import SimpleNamespace
import zipfile

import pytest

from scripts import stage9_v5d4_dependencies as deps
from scripts import stage9_v5d4_environment as env
from scripts import stage9_v5d4_launch as launch
from scripts import stage9_v5d5_shared as shared
from tests.deployment.test_stage9_v5d4_environment import staging
from tests.deployment.test_stage9_v5d5_shared import shared_material

pytestmark = [pytest.mark.security, pytest.mark.unit]


@pytest.fixture
def shared_staging(staging, shared_material, monkeypatch: pytest.MonkeyPatch):
    """Real verifier/writer; replace only host/native boundary and TEST pins."""
    inputs, approval, output, _ = staging
    project, directory, origins, proposal, expectation = shared_material
    (Path(inputs["repository"]) / "pyproject.toml").write_bytes(project.read_bytes())
    inputs["wheelhouse"] = str(directory)
    Path(inputs["origins"]).write_text(json.dumps(origins), encoding="utf-8")
    Path(inputs["proposal"]).write_text(json.dumps(proposal), encoding="utf-8")
    approval["proposal_sha256"] = deps.digest(deps.canonical(proposal))
    approval["wheel_shared_members"] = copy.deepcopy(expectation)
    original_helper = env.helper
    closure = original_helper("stage9_v5d4_closure.py")["inventory"](Path(inputs["repository"]))
    approval["dynamic_review_sha256"] = deps.digest(deps.canonical(closure))
    monkeypatch.setattr(
        env,
        "helper",
        lambda name: vars(deps) if name.endswith("dependencies.py") else original_helper(name),
    )
    real_run = runpy.run_path
    synthetic_tools = SimpleNamespace(
        run_path=lambda name: vars(shared)
        if Path(name).name in {"stage9_v5d5_shared.py", "shared-audit.py"}
        else real_run(name)
    )
    monkeypatch.setattr(env, "runpy", synthetic_tools)
    monkeypatch.setattr(launch, "runpy", synthetic_tools)
    return inputs, approval, output, proposal


def test_build_materializes_all_62_paths_once_and_keeps_original_records(
    shared_staging, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs, approval, output, proposal = shared_staging
    originals = {
        package["filename"]: (Path(inputs["wheelhouse"]) / package["filename"]).read_bytes()
        for package in proposal["packages"]
    }
    writes: list[str] = []
    original_write = env.write_new

    def tracked_write(root, name, data):
        writes.append(name)
        original_write(root, name, data)

    monkeypatch.setattr(env, "write_new", tracked_write)
    receipt = env.build(inputs, approval, output)
    assert len(receipt["shared_wheel_members"]["members"]) == 62
    assert set(shared.MEMBERS) <= set(receipt["package_files"])
    assert all(writes.count(name) == 1 for name in shared.MEMBERS)
    assert receipt["shared_wheel_expectations"] == approval["wheel_shared_members"]
    assert receipt["shared_wheel_policy"] == shared.POLICY_ID
    assert receipt["authorization"] == "NOT_AUTHORIZED"
    assert receipt["native_load"] == "NOT_EXECUTED"
    for row in receipt["shared_wheel_output"]:
        assert row["sha256"] == env.sha((output / "packages" / row["path"]).read_bytes())
        assert row["mode"] == 0o600 and row["type"] == "regular"
    for package in proposal["packages"]:
        filename = package["filename"]
        assert (Path(inputs["wheelhouse"]) / filename).read_bytes() == originals[filename]
        with zipfile.ZipFile(Path(inputs["wheelhouse"]) / filename) as archive:
            record = next(name for name in archive.namelist() if name.endswith(".dist-info/RECORD"))
            assert (output / "packages" / record).read_bytes() == archive.read(record)
    launch.verify_tree(output, receipt["files"], receipt["directories"])
    launch.verify_shared_receipt(output, receipt)
    assert launch.main([]) == 2


def test_canonical_owner_order_is_independent_of_wheel_order(shared_staging, tmp_path) -> None:
    inputs, approval, _, proposal = shared_staging
    normal = env.prepare_wheel_set(
        proposal["packages"], Path(inputs["wheelhouse"]), approval["wheel_shared_members"]
    )
    reversed_order = env.prepare_wheel_set(
        list(reversed(proposal["packages"])),
        Path(inputs["wheelhouse"]),
        approval["wheel_shared_members"],
    )
    assert normal == reversed_order
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    env.materialize_wheel_set(normal[0], first, normal[2])
    env.materialize_wheel_set(reversed_order[0], second, reversed_order[2])
    env.verify_materialized(first, normal[1])
    env.verify_materialized(second, normal[1])


@pytest.mark.parametrize("change", ["missing", "incomplete", "no_source_review"])
def test_missing_independent_sharing_expectations_refuse_before_output(shared_staging, change):
    inputs, approval, output, _ = shared_staging
    if change == "missing":
        approval.pop("wheel_shared_members")
    elif change == "incomplete":
        approval["wheel_shared_members"]["members"].pop()
    else:
        approval["wheel_shared_members"]["source_reviews"].pop()
    with pytest.raises(ValueError):
        env.build(inputs, approval, output)
    assert not output.exists()


@pytest.mark.parametrize("target", ["controller.py", "libvirt.py"])
def test_full_plan_keeps_controller_and_binding_collisions_forbidden(
    shared_staging, monkeypatch: pytest.MonkeyPatch, target: str
) -> None:
    inputs, approval, output, proposal = shared_staging
    # Only the trusted dependency result is replaced to isolate the final
    # packages/controller/binding reservation boundary. Exact sharing remains real.
    package = proposal["packages"][0]
    package["materialized_members"][target] = {"path": target, "sha256": "f" * 64}
    package["member_metadata"][target] = {"size": 0, "mode": 33188, "type": "regular"}
    Path(inputs["proposal"]).write_text(json.dumps(proposal), encoding="utf-8")
    original_helper = env.helper
    monkeypatch.setattr(
        env,
        "helper",
        lambda name: {**vars(deps), "verify": lambda *args: {"status": "TEST ONLY"}}
        if name.endswith("dependencies.py")
        else original_helper(name),
    )
    monkeypatch.setattr(env, "wheel_plan", lambda data, package: package["materialized_members"])
    with pytest.raises(ValueError, match="MATERIALIZATION_COLLISION"):
        env.build(inputs, approval, output)
    assert not output.exists()


def test_changed_archive_after_inspection_refuses_before_first_write(shared_staging) -> None:
    inputs, approval, output, proposal = shared_staging
    wheel_path = Path(inputs["wheelhouse"]) / proposal["packages"][0]["filename"]
    wheel_path.write_bytes(wheel_path.read_bytes() + b"changed after inspection")
    with pytest.raises(ValueError):
        env.build(inputs, approval, output)
    assert not output.exists()


def test_materializer_uses_immutable_validated_bytes(shared_staging, tmp_path) -> None:
    inputs, approval, _, proposal = shared_staging
    snapshots, planned, inventory = env.prepare_wheel_set(
        proposal["packages"], Path(inputs["wheelhouse"]), approval["wheel_shared_members"]
    )
    for package in proposal["packages"]:
        (Path(inputs["wheelhouse"]) / package["filename"]).write_bytes(b"replacement must not read")
    destination = tmp_path / "retained-snapshot"
    destination.mkdir()
    env.materialize_wheel_set(snapshots, destination, inventory)
    env.verify_materialized(destination, planned)


def test_preexisting_shared_target_refuses_before_any_materialization(shared_staging, tmp_path):
    inputs, approval, _, proposal = shared_staging
    snapshots, _, inventory = env.prepare_wheel_set(
        proposal["packages"], Path(inputs["wheelhouse"]), approval["wheel_shared_members"]
    )
    destination = tmp_path / "preexisting"
    path = destination / next(iter(shared.MEMBERS))
    path.parent.mkdir(parents=True)
    path.write_bytes(b"existing bytes remain unchanged")
    with pytest.raises(ValueError, match="MATERIALIZATION_COLLISION"):
        env.materialize_wheel_set(snapshots, destination, inventory)
    assert path.read_bytes() == b"existing bytes remain unchanged"
    assert list(destination.rglob("*")) == [path.parent, path]


@pytest.mark.parametrize("change", ["bytes", "mode", "directory"])
def test_postwrite_mutation_never_issues_receipt(shared_staging, monkeypatch, change):
    inputs, approval, output, _ = shared_staging
    original_helper = env.helper

    def changing_helper(name):
        tools = original_helper(name)
        if not name.endswith("abi.py"):
            return tools

        def diagnose(*args):
            result = tools["diagnose"](*args)
            path = output / "packages" / next(iter(shared.MEMBERS))
            if change == "bytes":
                path.write_bytes(b"tampered")
            elif change == "mode":
                path.chmod(0o644)
            else:
                (output / "packages/unapproved-empty-directory").mkdir()
            return result

        return {"diagnose": diagnose}

    monkeypatch.setattr(env, "helper", changing_helper)
    with pytest.raises(ValueError):
        env.build(inputs, approval, output)
    assert not (output / "ENVIRONMENT.json").exists()


@pytest.mark.skipif(os.name != "posix", reason="The writer's no-follow descriptors require POSIX.")
def test_secure_writer_refuses_symlink_parent_and_existing_file(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (root / "redirect").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="MATERIALIZATION_SYMLINK"):
        env.write_new(root, "redirect/never.py", b"must not escape")
    assert not list(outside.iterdir())
    env.write_new(root, "safe.py", b"original")
    assert stat.S_IMODE((root / "safe.py").stat().st_mode) == 0o600
    with pytest.raises(ValueError, match="MATERIALIZATION_COLLISION"):
        env.write_new(root, "safe.py", b"replacement")
    assert (root / "safe.py").read_bytes() == b"original"

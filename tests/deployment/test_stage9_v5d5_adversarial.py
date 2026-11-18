"""Independent malformed-contract and filesystem tests; no native loading."""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path

import pytest

from scripts import stage9_v5d4_environment as environment
from scripts import stage9_v5d4_dependencies as dependencies
from scripts import stage9_v5d4_launch as launcher
from scripts import stage9_v5d5_shared as shared
from tests.deployment.test_stage9_v5d5_shared import shared_material as shared_material


@pytest.fixture
def policy_packages():
    """In-memory declarations only; these objects never attest wheel bytes."""
    return [
        {
            "name": distribution,
            "version": identity["version"],
            "filename": identity["wheel_filename"],
            "sha256": identity["wheel_sha256"],
            "materialized_members": {
                path: {"path": path, "sha256": pin["sha256"]}
                for path, pin in shared.MEMBERS.items()
            },
            "member_metadata": {
                path: {key: pin[key] for key in ("size", "mode", "type")}
                for path, pin in shared.MEMBERS.items()
            },
        }
        for distribution, identity in shared.WHEELS.items()
    ]


def test_candidate_policy_preserves_every_v5d4_collision():
    source = Path(__file__).resolve().parents[2]
    report = json.loads(
        (source / "docs/architecture/evidence/stage9_v5d4_wheel_bytecode_inspection.json")
        .read_bytes()
        .decode("utf-8")
    )
    rows = report["shared_member_collisions"]
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
    assert sum(path.endswith(".py") for path in shared.MEMBERS) == 3
    assert sum(path.endswith(".pyi") for path in shared.MEMBERS) == 58


@pytest.mark.parametrize("field", ["size", "mode"])
def test_archive_integer_metadata_cannot_be_replaced_by_equal_float(policy_packages, field):
    path = next(iter(shared.MEMBERS))
    metadata = policy_packages[0]["member_metadata"][path]
    metadata[field] = float(metadata[field])
    with pytest.raises(ValueError, match="SHARED_WHEEL"):
        shared.collision_inventory(policy_packages)


def test_both_named_owners_cannot_hide_all_collisions(policy_packages):
    for package in policy_packages:
        package["materialized_members"] = {}
        package["member_metadata"] = {}
    with pytest.raises(ValueError, match="SHARED_WHEEL"):
        shared.collision_inventory(policy_packages)


def test_duplicate_distribution_cannot_impersonate_another_owner(policy_packages):
    policy_packages.append(copy.deepcopy(policy_packages[0]))
    with pytest.raises(ValueError, match="SHARED_WHEEL_DUPLICATE_DISTRIBUTION"):
        shared.collision_inventory(policy_packages)


def test_approved_target_cannot_be_reached_through_different_archive_member(policy_packages):
    path = next(iter(shared.MEMBERS))
    package = policy_packages[0]
    relocated = "pyside6_addons-6.10.3.data/purelib/" + path
    package["materialized_members"][relocated] = package["materialized_members"].pop(path)
    package["member_metadata"][relocated] = package["member_metadata"].pop(path)
    with pytest.raises(ValueError, match="SHARED_WHEEL_OWNER_MISMATCH"):
        shared.collision_inventory(policy_packages)


def test_one_archive_cannot_declare_two_members_for_a_shared_target(policy_packages):
    path = next(iter(shared.MEMBERS))
    package = policy_packages[0]
    package["materialized_members"]["alias.data/platlib/" + path] = dict(
        package["materialized_members"][path]
    )
    package["member_metadata"]["alias.data/platlib/" + path] = dict(
        package["member_metadata"][path]
    )
    with pytest.raises(ValueError, match="WHEEL_MATERIALIZATION_COLLISION"):
        shared.collision_inventory(policy_packages)


def test_regular_file_cannot_replace_parent_of_shared_targets(policy_packages):
    policy_packages[0]["materialized_members"]["PySide6"] = {
        "path": "PySide6",
        "sha256": "0" * 64,
    }
    policy_packages[0]["member_metadata"]["PySide6"] = {
        "size": 1,
        "mode": 0o100644,
        "type": "regular",
    }
    with pytest.raises(ValueError, match="WHEEL_MATERIALIZATION_COLLISION"):
        shared.collision_inventory(policy_packages)


def test_identical_unrelated_wheel_files_never_inherit_pyside_approval():
    packages = [
        {
            "name": name,
            "version": "1.0",
            "filename": name + "-1.0-py3-none-any.whl",
            "sha256": "1" * 64,
            "materialized_members": {"unrelated.py": {"path": "unrelated.py", "sha256": "2" * 64}},
            "member_metadata": {"unrelated.py": {"size": 1, "mode": 0o100644, "type": "regular"}},
        }
        for name in ("first", "second")
    ]
    with pytest.raises(ValueError, match="SHARED_WHEEL_MEMBER_SET_MISMATCH"):
        shared.collision_inventory(packages)


@pytest.mark.parametrize("value", [None, False, 1, "packages", {}])
def test_malformed_package_collection_is_controlled_refusal(value):
    with pytest.raises(ValueError, match="SHARED_WHEEL"):
        shared.collision_inventory(value)


@pytest.mark.parametrize("field", ["materialized_members", "member_metadata"])
@pytest.mark.parametrize("value", [None, False, 1, "members", []])
def test_malformed_member_collection_is_controlled_refusal(policy_packages, field, value):
    policy_packages[0][field] = value
    with pytest.raises(ValueError, match="SHARED_WHEEL"):
        shared.collision_inventory(policy_packages)


@pytest.mark.parametrize("value", [None, False, 1, "entry", []])
def test_malformed_member_entry_is_controlled_refusal(policy_packages, value):
    path = next(iter(shared.MEMBERS))
    policy_packages[0]["materialized_members"][path] = value
    with pytest.raises(ValueError, match="SHARED_WHEEL"):
        shared.collision_inventory(policy_packages)


@pytest.mark.parametrize("value", [None, False, 1, "metadata", []])
def test_malformed_member_metadata_is_controlled_refusal(policy_packages, value):
    path = next(iter(shared.MEMBERS))
    policy_packages[0]["member_metadata"][path] = value
    with pytest.raises(ValueError, match="SHARED_WHEEL"):
        shared.collision_inventory(policy_packages)


@pytest.mark.skipif(os.name != "posix", reason="Directory descriptors require POSIX.")
def test_writer_refuses_symlink_swap_between_parent_check_and_open(tmp_path, monkeypatch):
    root = tmp_path / "destination"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    (root / "package").mkdir()
    original = os.open

    def swap_before_open(path, flags, mode=0o777, *, dir_fd=None):
        if path == "package" and dir_fd is not None:
            (root / "package").rmdir()
            (root / "package").symlink_to(outside, target_is_directory=True)
        return original(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(os, "open", swap_before_open)
    with pytest.raises((ValueError, OSError)):
        environment.write_new(root, "package/shared.py", b"never written outside")
    assert list(outside.iterdir()) == []


@pytest.mark.skipif(os.name != "posix", reason="Directory descriptors require POSIX.")
def test_writer_never_follows_swapped_ancestor_when_opening_root(tmp_path, monkeypatch):
    ancestor = tmp_path / "ancestor"
    root = ancestor / "destination"
    outside = tmp_path / "outside"
    root.mkdir(parents=True)
    (outside / "destination").mkdir(parents=True)
    original = os.open
    attacked = False

    def swap_before_root_open(path, flags, mode=0o777, *, dir_fd=None):
        nonlocal attacked
        if not attacked and (
            (path == root and dir_fd is None) or (path == "ancestor" and dir_fd is not None)
        ):
            attacked = True
            ancestor.rename(tmp_path / "moved-ancestor")
            ancestor.symlink_to(outside, target_is_directory=True)
        return original(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(os, "open", swap_before_root_open)
    try:
        environment.write_new(root, "shared.py", b"never written outside")
    except (ValueError, OSError):
        pass
    assert attacked
    assert not (outside / "destination/shared.py").exists()


@pytest.mark.skipif(os.name != "posix", reason="Directory descriptors require POSIX.")
def test_writer_rejects_root_replacement_with_another_regular_directory(tmp_path, monkeypatch):
    root = tmp_path / "destination"
    root.mkdir()
    original = os.open
    attacked = False

    def replace_before_open(path, flags, mode=0o777, *, dir_fd=None):
        nonlocal attacked
        if not attacked and (
            (path == root and dir_fd is None) or (path == "destination" and dir_fd is not None)
        ):
            attacked = True
            root.rename(tmp_path / "original-destination")
            root.mkdir()
        return original(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(os, "open", replace_before_open)
    with pytest.raises((ValueError, OSError)):
        environment.write_new(root, "shared.py", b"must not merge with replacement directory")
    assert attacked
    assert list(root.iterdir()) == []


@pytest.mark.skipif(os.name != "posix", reason="Directory descriptors require POSIX.")
def test_writer_rejects_child_directory_replacement_before_descent(tmp_path, monkeypatch):
    root = tmp_path / "destination"
    root.mkdir()
    (root / "package").mkdir()
    original = os.open
    attacked = False

    def replace_before_descent(path, flags, mode=0o777, *, dir_fd=None):
        nonlocal attacked
        if not attacked and path == "package" and dir_fd is not None:
            attacked = True
            (root / "package").rename(root / "original-package")
            (root / "package").mkdir()
            (root / "package/unplanned.txt").write_bytes(b"uncontrolled directory contents")
        return original(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(os, "open", replace_before_descent)
    with pytest.raises(ValueError, match="MATERIALIZATION_DIRECTORY_CHANGED"):
        environment.write_new(root, "package/shared.py", b"must not merge into a replacement")
    assert attacked
    assert not (root / "package/shared.py").exists()
    assert (root / "package/unplanned.txt").read_bytes() == b"uncontrolled directory contents"


def test_writer_refuses_existing_file_without_changing_bytes(tmp_path):
    target = tmp_path / "shared.py"
    target.write_bytes(b"existing bytes")
    with pytest.raises(ValueError, match="MATERIALIZATION_COLLISION"):
        environment.write_new(tmp_path, "shared.py", b"replacement")
    assert target.read_bytes() == b"existing bytes"


@pytest.mark.skipif(os.name != "posix", reason="Test requires unprivileged directory symlinks.")
@pytest.mark.parametrize("link_is_root", [True, False])
def test_launcher_refuses_symlinked_runtime_root_or_ancestor(tmp_path, link_is_root):
    real = tmp_path / "real"
    actual_root = real if link_is_root else real / "runtime"
    actual_root.mkdir(parents=True)
    content = b"inert runtime source"
    (actual_root / "module.py").write_bytes(content)
    linked = tmp_path / "linked"
    linked.symlink_to(real, target_is_directory=True)
    root = linked if link_is_root else linked / "runtime"
    with pytest.raises(ValueError, match="SYMLINK_RUNTIME"):
        launcher.verify_tree(root, {"module.py": environment.sha(content)})


@pytest.fixture
def shared_receipt(tmp_path, monkeypatch, shared_material):
    if os.name != "posix":
        pytest.skip("Receipt mode verification belongs to the Linux-only launcher.")
    _, wheelhouse, _, proposal, expectation = shared_material
    monkeypatch.setattr(environment, "helper", lambda _: vars(dependencies))
    monkeypatch.setattr(environment.runpy, "run_path", lambda _: vars(shared))
    snapshots, planned, inventory = environment.prepare_wheel_set(
        proposal["packages"], wheelhouse, expectation
    )
    root = tmp_path / "receipt-runtime"
    root.mkdir()
    (root / "packages").mkdir()
    environment.materialize_wheel_set(snapshots, root / "packages", inventory)
    receipt = {
        "profile": "stage9-v5d5-prepared-environment-v1",
        "directories": ["packages", "packages/PySide6"],
        "files": {"shared-audit.py": "0" * 64, **{"packages/" + k: v for k, v in planned.items()}},
        "package_files": planned,
        "wheel_materialization": proposal["packages"],
        "wheel_archives": proposal["packages"],
        "shared_wheel_members": inventory,
        "shared_wheel_expectations": expectation,
        "shared_wheel_policy": shared.POLICY_ID,
        "shared_wheel_output": [
            {"path": row["path"], "sha256": row["sha256"], "mode": 0o600, "type": "regular"}
            for row in inventory["members"]
        ],
    }
    launcher.verify_shared_receipt(root, receipt)
    return root, receipt


@pytest.mark.parametrize("field", ["size", "mode"])
def test_receipt_report_integer_cannot_be_replaced_by_equal_float(shared_receipt, field):
    root, receipt = shared_receipt
    receipt["shared_wheel_members"] = copy.deepcopy(receipt["shared_wheel_members"])
    row = receipt["shared_wheel_members"]["members"][0]
    row[field] = float(row[field])
    with pytest.raises(ValueError, match="SHARED_WHEEL_RECEIPT_MISMATCH"):
        launcher.verify_shared_receipt(root, receipt)


def test_receipt_output_mode_cannot_be_replaced_by_equal_float(shared_receipt):
    root, receipt = shared_receipt
    receipt["shared_wheel_output"][0]["mode"] = float(receipt["shared_wheel_output"][0]["mode"])
    with pytest.raises(ValueError, match="SHARED_WHEEL_OUTPUT_MISMATCH"):
        launcher.verify_shared_receipt(root, receipt)


@pytest.mark.parametrize("field", ["package_files", "wheel_archives", "wheel_materialization"])
@pytest.mark.parametrize("value", [None, "wrong", False, 1, []])
def test_malformed_receipt_collections_are_controlled_refusals(shared_receipt, field, value):
    root, receipt = shared_receipt
    receipt[field] = value
    with pytest.raises(ValueError):
        launcher.verify_shared_receipt(root, receipt)


def test_receipt_cannot_omit_separately_approved_expectations(shared_receipt):
    root, receipt = shared_receipt
    receipt.pop("shared_wheel_expectations")
    with pytest.raises(ValueError, match="SHARED_WHEEL_RECEIPT_INCOMPLETE"):
        launcher.verify_shared_receipt(root, receipt)

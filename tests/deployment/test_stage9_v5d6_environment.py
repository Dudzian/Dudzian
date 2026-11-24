"""Synthetic header omissions stay separate from runtime and native authority."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import runpy
from types import SimpleNamespace
import zipfile

import pytest

from scripts import stage9_v5d4_dependencies as deps
from scripts import stage9_v5d4_environment as env
from scripts import stage9_v5d4_launch as launch
from scripts import stage9_v5d6_headers as headers
from tests.deployment.test_stage9_v5d4_environment import staging
from tests.deployment.test_stage9_v5d6_headers import (
    HEADER_CONTENT,
    header_material,
    package_path,
    rewrite,
)

pytestmark = [pytest.mark.security, pytest.mark.unit]


@pytest.fixture
def header_staging(staging, header_material, monkeypatch):
    """Keep real dependency/header verification and writes, model only host/ABI."""
    inputs, approval, output, _ = staging
    project, directory, origins, proposal, expectation = header_material
    (Path(inputs["repository"]) / "pyproject.toml").write_bytes(project.read_bytes())
    inputs["wheelhouse"] = str(directory)
    Path(inputs["origins"]).write_text(json.dumps(origins), encoding="utf-8")
    Path(inputs["proposal"]).write_text(json.dumps(proposal), encoding="utf-8")
    approval["proposal_sha256"] = deps.digest(deps.canonical(proposal))
    approval["wheel_header_omissions"] = copy.deepcopy(expectation)
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
        run_path=lambda name: vars(headers)
        if Path(name).name in {"stage9_v5d6_headers.py", "header-audit.py"}
        else real_run(name)
    )
    monkeypatch.setattr(env, "runpy", synthetic_tools)
    monkeypatch.setattr(launch, "runpy", synthetic_tools)
    return inputs, approval, output, proposal


def test_builder_receipt_binds_header_identity_and_excludes_all_header_bytes(header_staging):
    inputs, approval, output, proposal = header_staging
    package = proposal["packages"][0]
    path = Path(inputs["wheelhouse"]) / package["filename"]
    original = path.read_bytes()
    with zipfile.ZipFile(path) as archive:
        record = "greenlet-3.5.6.dist-info/RECORD"
        original_record = archive.read(record)
    receipt = env.build(inputs, approval, output)
    assert receipt["header_omissions"]["members"] == [headers.pinned_row()]
    assert receipt["header_omission_expectations"] == approval["wheel_header_omissions"]
    assert receipt["header_omission_policy"] == headers.POLICY_ID
    assert receipt["header_runtime_inclusion"] == "EXCLUDED"
    assert receipt["wheel_materialization"][0]["header_omissions"] == [headers.pinned_row()]
    assert receipt["wheel_omissions"] == []
    assert receipt["native_load"] == "NOT_EXECUTED"
    assert receipt["authorization"] == "NOT_AUTHORIZED"
    assert headers.HEADER["member"] not in receipt["package_files"]
    assert headers.HEADER["sha256"] not in receipt["files"].values()
    assert (output / "packages" / record).read_bytes() == original_record
    assert path.read_bytes() == original
    launch.verify_tree(output, receipt["files"], receipt["directories"])
    launch.verify_shared_receipt(output, receipt)
    launch.verify_header_receipt(output, receipt)
    assert launch.main([]) == 2


@pytest.mark.parametrize("change", ["missing", "decision", "member", "policy", "reference"])
def test_builder_refuses_absent_or_inconsistent_separate_approval_before_writes(
    header_staging, change
):
    inputs, approval, output, _ = header_staging
    expectation = approval["wheel_header_omissions"]
    if change == "missing":
        approval.pop("wheel_header_omissions")
    elif change == "member":
        expectation["members"].clear()
    elif change == "reference":
        expectation["review_reference"] = " "
    else:
        expectation[change] = "UNREVIEWED"
    with pytest.raises(ValueError):
        env.build(inputs, approval, output)
    assert not output.exists()


def test_plan_and_single_archive_extraction_cannot_implicitly_approve(header_staging, tmp_path):
    inputs, approval, _, proposal = header_staging
    package = proposal["packages"][0]
    data = (Path(inputs["wheelhouse"]) / package["filename"]).read_bytes()
    plan = env.wheel_plan(data, package)
    assert headers.HEADER["member"] not in plan
    destination = tmp_path / "extract"
    destination.mkdir()
    with pytest.raises(ValueError, match=headers.APPROVAL_REQUIRED):
        env.extract_wheel(data, destination, package)
    assert not list(destination.iterdir())
    env.extract_wheel(data, destination, package, approval["wheel_header_omissions"])
    env.verify_materialized(destination, {row["path"]: row["sha256"] for row in plan.values()})
    with pytest.raises(ValueError, match="UNSUPPORTED_WHEEL_DATA"):
        env.wheel_plan(data)


@pytest.mark.parametrize("change", ["archive", "plan", "declaration", "erased_candidate_identity"])
def test_immutable_snapshot_and_plan_recheck_refuse_mutations_before_first_write(
    header_staging, tmp_path, change
):
    inputs, approval, _, proposal = header_staging
    snapshots, _, shared = env.prepare_wheel_set(
        proposal["packages"],
        Path(inputs["wheelhouse"]),
        None,
        approval["wheel_header_omissions"],
    )
    package, data, plan = snapshots[0]
    if change == "archive":
        snapshots[0] = (package, data + b"changed snapshot", plan)
    elif change == "declaration":
        package["materialized_members"][headers.HEADER["member"]] = {
            "path": "greenlet.h",
            "sha256": headers.HEADER["sha256"],
        }
    else:
        plan[headers.HEADER["member"]] = {
            "path": "injected/greenlet.h",
            "sha256": headers.HEADER["sha256"],
        }
        if change == "erased_candidate_identity":
            package.update(name="unrelated", version="1.0", header_omissions=[])
    destination = tmp_path / "refused-plan"
    destination.mkdir()
    with pytest.raises(ValueError):
        env.materialize_wheel_set(
            snapshots,
            destination,
            shared,
            None if change == "erased_candidate_identity" else approval["wheel_header_omissions"],
        )
    assert not list(destination.iterdir())


def test_snapshot_materializer_requires_its_own_separate_header_expectations(
    header_staging, tmp_path
):
    inputs, approval, _, proposal = header_staging
    snapshots, _, shared = env.prepare_wheel_set(
        proposal["packages"],
        Path(inputs["wheelhouse"]),
        None,
        approval["wheel_header_omissions"],
    )
    with pytest.raises(ValueError, match=headers.APPROVAL_REQUIRED):
        env.materialize_wheel_set(snapshots, tmp_path / "never-created", shared)
    assert not (tmp_path / "never-created").exists()


def test_changed_header_after_inspection_refuses_before_output(header_staging, header_material):
    inputs, approval, output, _ = header_staging
    rewrite(
        package_path(header_material),
        updates={headers.HEADER["member"]: b"changed"},
        reseal_record=True,
    )
    with pytest.raises(ValueError, match="ORIGIN_SHA256"):
        env.build(inputs, approval, output)
    assert not output.exists()


def test_symlink_wheel_refuses_before_output(header_staging, tmp_path):
    inputs, approval, output, proposal = header_staging
    wheel = Path(inputs["wheelhouse"]) / proposal["packages"][0]["filename"]
    retained = tmp_path / "retained.whl"
    wheel.rename(retained)
    try:
        wheel.symlink_to(retained)
    except (OSError, NotImplementedError):
        pytest.skip("Host cannot create unprivileged symlinks")
    with pytest.raises(ValueError):
        env.build(inputs, approval, output)
    assert not output.exists()


def test_symlink_runtime_destination_refuses_without_escaping(header_staging, tmp_path):
    inputs, approval, _, proposal = header_staging
    snapshots, _, shared = env.prepare_wheel_set(
        proposal["packages"],
        Path(inputs["wheelhouse"]),
        None,
        approval["wheel_header_omissions"],
    )
    outside = tmp_path / "outside"
    outside.mkdir()
    destination = tmp_path / "redirected"
    try:
        destination.symlink_to(outside, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("Host cannot create unprivileged symlinks")
    with pytest.raises(ValueError, match="MATERIALIZATION_SYMLINK"):
        env.materialize_wheel_set(
            snapshots, destination, shared, approval["wheel_header_omissions"]
        )
    assert not list(outside.iterdir())


def test_builder_cannot_issue_receipt_if_header_appears_after_materialization(
    header_staging, monkeypatch
):
    inputs, approval, output, _ = header_staging
    original_helper = env.helper

    def mutating_helper(name):
        tools = original_helper(name)
        if not name.endswith("abi.py"):
            return tools

        def diagnose(*args):
            result = tools["diagnose"](*args)
            (output / "unapproved-header-alias.bin").write_bytes(HEADER_CONTENT)
            return result

        return {**tools, "diagnose": diagnose}

    monkeypatch.setattr(env, "helper", mutating_helper)
    with pytest.raises(ValueError, match="HEADER_RUNTIME_INCLUSION_FORBIDDEN"):
        env.build(inputs, approval, output)
    assert not (output / "ENVIRONMENT.json").exists()


@pytest.mark.parametrize(
    "field",
    [
        "header_omissions",
        "header_omission_expectations",
        "header_omission_policy",
        "header_runtime_inclusion",
    ],
)
def test_launcher_requires_complete_header_receipt_even_when_tree_matches(header_staging, field):
    inputs, approval, output, _ = header_staging
    receipt = env.build(inputs, approval, output)
    receipt.pop(field)
    launch.verify_tree(output, receipt["files"], receipt["directories"])
    with pytest.raises(ValueError, match="HEADER_OMISSION_RECEIPT_INCOMPLETE"):
        launch.verify_header_receipt(output, receipt)


@pytest.mark.parametrize(
    "change", ["expectation", "policy", "runtime", "member", "archive", "verification"]
)
def test_launcher_refuses_inconsistent_header_evidence_even_when_tree_matches(
    header_staging, change
):
    inputs, approval, output, _ = header_staging
    receipt = env.build(inputs, approval, output)
    if change == "expectation":
        receipt["header_omission_expectations"]["decision"] = "NOT_APPROVED"
    elif change == "policy":
        receipt["header_omission_policy"] += "other"
    elif change == "runtime":
        receipt["header_runtime_inclusion"] = "INCLUDED"
    elif change == "member":
        receipt["header_omissions"]["members"][0]["size"] += 1
    elif change == "archive":
        receipt["wheel_archives"][0]["sha256"] = "0" * 64
    else:
        receipt["dependency_verification"]["header_expectations"] = None
    launch.verify_tree(output, receipt["files"], receipt["directories"])
    with pytest.raises(ValueError):
        launch.verify_header_receipt(output, receipt)


@pytest.mark.parametrize(
    "target", ["packages/greenlet.h", "packages/alias.bin", "alias-outside-packages.bin"]
)
def test_launcher_refuses_copied_header_despite_matching_updated_file_inventory(
    header_staging, target
):
    inputs, approval, output, _ = header_staging
    receipt = env.build(inputs, approval, output)
    path = output / target
    path.write_bytes(HEADER_CONTENT)
    receipt["files"][target] = deps.digest(HEADER_CONTENT)
    if target.startswith("packages/"):
        receipt["package_files"][target.removeprefix("packages/")] = deps.digest(HEADER_CONTENT)
    launch.verify_tree(output, receipt["files"], receipt["directories"])
    with pytest.raises(ValueError, match="HEADER_OMISSION_MATERIALIZED"):
        launch.verify_header_receipt(output, receipt)


def test_launcher_detects_undeclared_header_alias_and_symlink(header_staging, tmp_path):
    inputs, approval, output, _ = header_staging
    receipt = env.build(inputs, approval, output)
    alias = output / "packages/undeclared.bin"
    alias.write_bytes(HEADER_CONTENT)
    with pytest.raises(ValueError, match="HEADER_OMISSION_MATERIALIZED"):
        launch.verify_header_receipt(output, receipt)
    alias.unlink()
    retained = tmp_path / "external-header.h"
    retained.write_bytes(HEADER_CONTENT)
    try:
        alias.symlink_to(retained)
    except (OSError, NotImplementedError):
        pytest.skip("Host cannot create unprivileged symlinks")
    with pytest.raises(ValueError, match="SYMLINK_RUNTIME"):
        launch.verify_header_receipt(output, receipt)


def test_launcher_old_receipt_needs_no_new_header_contract(tmp_path):
    launch.verify_header_receipt(tmp_path, {"files": {}})


@pytest.mark.parametrize("receipt", [None, [], {}, {"files": []}])
def test_launcher_malformed_receipt_fails_closed(tmp_path, receipt):
    with pytest.raises(ValueError, match="HEADER_OMISSION_RECEIPT_SCHEMA"):
        launch.verify_header_receipt(tmp_path, receipt)

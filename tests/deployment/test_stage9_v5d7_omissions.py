"""Exercise exact build/test artifact rules with inert synthetic archive data."""

from __future__ import annotations

import copy
import hashlib
import io
import shutil
import stat
import zipfile

import pytest

from scripts import stage9_v5d4_dependencies as dependencies
from scripts import stage9_v5d4_environment as environment
from scripts import stage9_v5d7_omissions as omissions
from scripts import stage9_v5d7_shared as sharing
from tests.deployment.test_stage9_v5d4_dependencies import wheel
from tests.deployment.test_stage9_v5d6_headers import rewrite
from tests.deployment.test_stage9_v5d7_sharing import approval as shared_approval
from tests.deployment.test_stage9_v5d7_sharing import three_wheels as three_wheels

pytestmark = [pytest.mark.security, pytest.mark.unit]


def archive_bytes(contents, modes=None):
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        for name, data in contents.items():
            info = zipfile.ZipInfo(name)
            info.create_system = 3
            info.external_attr = (modes or {}).get(name, omissions.ARCHIVE_MODE) << 16
            archive.writestr(info, data)
    return stream.getvalue()


@pytest.fixture
def candidate(monkeypatch):
    identity = omissions.WHEELS[omissions.GREENLET_SHA256]
    artifacts = omissions.MEMBERS[omissions.GREENLET_SHA256]
    contents = {name: f"TEST ONLY inert fixture for {name}".encode() for name in artifacts}
    archive = archive_bytes(contents)
    digest = hashlib.sha256(archive).hexdigest()
    monkeypatch.setattr(omissions, "WHEELS", {digest: identity})
    monkeypatch.setattr(
        omissions,
        "MEMBERS",
        {
            digest: {
                name: (hashlib.sha256(content).hexdigest(), len(content), artifacts[name][2])
                for name, content in contents.items()
            }
        },
    )
    return contents, archive, digest, identity


def inspect(candidate, raw=None, *, files=None, digest=None, **changes):
    contents, archive, original_digest, identity = candidate
    expected = {**identity, **changes}
    with zipfile.ZipFile(io.BytesIO(archive if raw is None else raw)) as data:
        return omissions.inspect_omissions(
            expected["distribution"],
            expected["version"],
            expected["wheel_filename"],
            original_digest if digest is None else digest,
            data,
            set(contents) if files is None else files,
        )


def test_archive_requires_exact_identity_and_complete_artifact_set(candidate):
    contents, _, _, identity = candidate
    assert {row["member"] for row in inspect(candidate)} == set(contents)
    assert inspect(candidate, digest="f" * 64) == []
    for changes in (
        {"distribution": "moved"},
        {"version": "99"},
        {"wheel_filename": "other-" + identity["wheel_filename"]},
    ):
        with pytest.raises(omissions.Refusal, match="ARTIFACT_OMISSION_WHEEL_IDENTITY_MISMATCH"):
            inspect(candidate, **changes)
    changed = set(contents)
    changed.pop()
    changed.add("elsewhere/greenlet.h")
    with pytest.raises(omissions.Refusal, match="ARTIFACT_OMISSION_SET_MISMATCH"):
        inspect(candidate, files=changed)


@pytest.mark.parametrize(
    "change,reason",
    [
        ("mode", "ARTIFACT_OMISSION_MEMBER_METADATA_MISMATCH"),
        ("type", "ARTIFACT_OMISSION_MEMBER_METADATA_MISMATCH"),
        ("size", "ARTIFACT_OMISSION_MEMBER_METADATA_MISMATCH"),
        ("hash", "ARTIFACT_OMISSION_MEMBER_HASH_MISMATCH"),
    ],
)
def test_changed_member_cannot_use_an_existing_archive_pin(candidate, change, reason):
    contents = candidate[0].copy()
    name = next(iter(contents))
    modes = {}
    if change == "mode":
        modes[name] = stat.S_IFREG | 0o755
    elif change == "type":
        modes[name] = stat.S_IFLNK | 0o644
    elif change == "size":
        contents[name] += b"!"
    else:
        contents[name] = b"!" + contents[name][1:]
    with pytest.raises(omissions.Refusal, match=reason):
        inspect(candidate, archive_bytes(contents, modes))


@pytest.fixture
def declarations(candidate):
    _, _, sha, identity = candidate
    rows = inspect(candidate)
    package = {
        "name": identity["distribution"],
        "version": identity["version"],
        "filename": identity["wheel_filename"],
        "sha256": sha,
        "artifact_omissions": rows,
        "materialized_members": {},
        "member_metadata": {
            row["member"]: {key: row[key] for key in omissions.METADATA_KEYS} for row in rows
        },
    }
    expectation = {
        "policy": omissions.POLICY_ID,
        "installation_contract_exception": omissions.INSTALLATION_EXCEPTION,
        "members": copy.deepcopy(rows),
        "decision": omissions.REVIEW_DECISION,
        "review_reference": "TEST ONLY independent reviewer fixture",
    }
    return package, expectation


def test_new_policy_requires_independent_exact_approval(declarations):
    package, approval = declarations
    with pytest.raises(omissions.Refusal, match=omissions.APPROVAL_REQUIRED):
        omissions.verify_omissions([package], None)
    inventory = omissions.verify_omissions([package], approval)
    assert inventory["members"] == approval["members"]
    for change in ({"policy": "V5d6"}, {"review_reference": "  "}, {"decision": ""}):
        with pytest.raises(omissions.Refusal, match="ARTIFACT_OMISSION_EXPECTATION_MISMATCH"):
            omissions.verify_omissions([package], {**approval, **change})
    changed = copy.deepcopy(approval)
    changed["members"][0]["size"] = float(changed["members"][0]["size"])
    with pytest.raises(omissions.Refusal, match="ARTIFACT_OMISSION_EXPECTATION_MISMATCH"):
        omissions.verify_omissions([package], changed)
    package.pop("artifact_omissions")
    with pytest.raises(omissions.Refusal, match="ARTIFACT_OMISSION_SET_MISMATCH"):
        omissions.verify_omissions([package], approval)


def test_runtime_absence_covers_source_target_hash_alias_and_other_wheels(declarations):
    package, approval = declarations
    inventory = omissions.verify_omissions([package], approval)
    omitted = inventory["members"][0]
    different = "f" * 64
    assert different != omitted["sha256"]
    for path, digest in (
        ("packages/" + omitted["runtime_path"], different),
        (omitted["runtime_path"] + "/child", different),
        ("copied/alias", omitted["sha256"]),
    ):
        with pytest.raises(
            omissions.Refusal, match="ARTIFACT_OMISSION_RUNTIME_INCLUSION_FORBIDDEN"
        ):
            omissions.verify_runtime_absence({path: digest}, inventory)
    other = {
        "materialized_members": {omitted["member"]: {"path": "elsewhere", "sha256": different}}
    }
    with pytest.raises(omissions.Refusal, match="ARTIFACT_OMISSION_RUNTIME_INCLUSION_FORBIDDEN"):
        omissions.verify_omissions([package, other], approval)
    omissions.verify_runtime_absence({"unrelated.txt": different}, inventory)


@pytest.fixture
def scipy_candidate(tmp_path, monkeypatch):
    member = "scipy/special/tests/_cython_examples/extending.pyx"
    content = b"#!/usr/bin/env python3\n# TEST ONLY Cython data.\n"
    wheelhouse = tmp_path / "wheels"
    wheelhouse.mkdir()
    path, origin = wheel(wheelhouse, "scipy", "1.18.1", files={member: content})

    def register_fixture_pins():
        sha = dependencies.digest(path.read_bytes())
        identity = {
            "distribution": "scipy",
            "version": "1.18.1",
            "wheel_filename": path.name,
        }
        monkeypatch.setattr(omissions, "WHEELS", {sha: identity})
        monkeypatch.setattr(
            omissions,
            "MEMBERS",
            {
                sha: {
                    member: (dependencies.digest(content), len(content), "CYTHON_TEST_BUILD_INPUT")
                }
            },
        )
        origin["sha256"] = sha

    register_fixture_pins()
    monkeypatch.setattr(dependencies, "artifact_tools", lambda: vars(omissions))
    return member, path, origin, register_fixture_pins


def test_full_record_is_required_before_allowing_exact_omission(scipy_candidate):
    member, path, origin, register_fixture_pins = scipy_candidate
    with pytest.raises(dependencies.Refusal, match="WHEEL_EXECUTABLE_FORMAT_UNSUPPORTED"):
        dependencies.inspect_wheel(path, origin)
    report = dependencies.inspect_wheel(path, origin, v5d7=True)
    assert len(report["artifact_omissions"]) == 1
    assert report["artifact_omissions"][0]["member"] == member
    assert member not in report["materialized_members"]
    assert "scipy/__init__.py" in report["materialized_members"]

    rewrite(path, bad_record_sha=member)
    register_fixture_pins()
    with pytest.raises(dependencies.Refusal, match="WHEEL_RECORD_INTEGRITY"):
        dependencies.inspect_wheel(path, origin, v5d7=True)


def test_artifact_approval_binds_proposal_and_materialization(
    tmp_path, monkeypatch, scipy_candidate, three_wheels
):
    member, path, origin, _ = scipy_candidate
    project, wheelhouse, origins = three_wheels
    project.write_text(
        '[project]\nrequires-python=">=3.11"\n'
        'dependencies=["scipy==1.18.1", "PySide6==6.10.3",'
        '"PySide6-Essentials==6.10.3", "PySide6-Addons==6.10.3"]\n'
        "[project.optional-dependencies]\nobserver-kvm-preflight=[]\n",
        encoding="utf-8",
    )
    path = shutil.copyfile(path, wheelhouse / path.name)
    origins[path.name] = origin
    proposal = dependencies.propose(project, wheelhouse, origins, v5d7=True)
    proposal_hash = dependencies.digest(dependencies.canonical(proposal))
    artifact_inventory = proposal["artifact_omissions"]
    sharing_expectation = shared_approval(sharing, proposal["shared_wheel_members"])
    expectation = {
        "policy": omissions.POLICY_ID,
        "installation_contract_exception": omissions.INSTALLATION_EXCEPTION,
        "members": copy.deepcopy(artifact_inventory["members"]),
        "decision": omissions.REVIEW_DECISION,
        "review_reference": "TEST ONLY independent artifact review",
    }
    original_bytes = path.read_bytes()
    with pytest.raises(ValueError, match=omissions.APPROVAL_REQUIRED):
        dependencies.verify(
            project,
            wheelhouse,
            origins,
            proposal,
            proposal_hash,
            {},
            approved_shared=sharing_expectation,
        )
    verified = dependencies.verify(
        project,
        wheelhouse,
        origins,
        proposal,
        proposal_hash,
        {},
        approved_shared=sharing_expectation,
        approved_artifacts=expectation,
    )
    assert verified["artifact_omissions"] == artifact_inventory
    assert verified["artifact_expectations"] == expectation

    original_helper = environment.helper
    tools = {
        "stage9_v5d4_dependencies.py": dependencies,
        "stage9_v5d7_omissions.py": omissions,
        "stage9_v5d7_shared.py": sharing,
    }
    monkeypatch.setattr(
        environment,
        "helper",
        lambda name: vars(tools[name]) if name in tools else original_helper(name),
    )
    snapshots, files, shared = environment.prepare_wheel_set(
        proposal["packages"],
        wheelhouse,
        sharing_expectation,
        v5d7=True,
        approved_artifacts=expectation,
    )
    assert len(files) == len({target.casefold() for target in files})
    destination = tmp_path / "checked-snapshots"
    destination.mkdir()
    with pytest.raises(ValueError, match=omissions.APPROVAL_REQUIRED):
        environment.materialize_wheel_set(
            snapshots, destination, shared, v5d7=True, approved_shared=sharing_expectation
        )
    assert not list(destination.iterdir())
    environment.materialize_wheel_set(
        snapshots,
        destination,
        shared,
        v5d7=True,
        approved_shared=sharing_expectation,
        approved_artifacts=expectation,
    )
    assert not (destination / member).exists()
    omissions.verify_runtime_absence(files, artifact_inventory)
    assert path.read_bytes() == original_bytes
    record = "scipy-1.18.1.dist-info/RECORD"
    with zipfile.ZipFile(path) as archive:
        assert (destination / record).read_bytes() == archive.read(record)

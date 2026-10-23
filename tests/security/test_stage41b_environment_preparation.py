"""Unprivileged preparation logic tests; never install into /opt or start services."""

from __future__ import annotations

import base64
from copy import deepcopy
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
from types import SimpleNamespace
import zipfile

import pytest

from scripts import prepare_stage41b_environment as preparation

pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="Linux qualification preparation")


def _json(value):
    return json.dumps(value, sort_keys=True).encode()


def _b64(raw):
    return base64.b64encode(raw).decode()


def _wheel(payload=b"answer = 42\n", *, extra=None, alter_record=False, member_type=None):
    name = "synthetic_stage41b-1.0.dist-info/"
    members = {
        "synthetic_stage41b/__init__.py": payload,
        name + "METADATA": b"Metadata-Version: 2.1\nName: synthetic-stage41b\nVersion: 1.0\n",
        name + "WHEEL": b"Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
    }
    members.update(extra or {})
    rows = []
    for path, content in members.items():
        digest = base64.urlsafe_b64encode(hashlib.sha256(content).digest()).rstrip(b"=").decode()
        rows.append([path, "sha256=" + digest, str(len(content))])
    if alter_record:
        rows[0][1] = "sha256=wrong"
    rows.append([name + "RECORD", "", ""])
    record_text = io.StringIO()
    csv.writer(record_text, lineterminator="\n").writerows(rows)
    members[name + "RECORD"] = record_text.getvalue().encode()
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w") as stream:
        for path, content in members.items():
            entry = zipfile.ZipInfo(path)
            if path == member_type:
                entry.external_attr = (stat.S_IFLNK | 0o777) << 16
            stream.writestr(entry, content)
    raw = archive.getvalue()
    record = {
        "name": "synthetic-stage41b",
        "version": "1.0",
        "filename": "synthetic_stage41b-1.0-py3-none-any.whl",
        "sha256": hashlib.sha256(raw).hexdigest(),
        "url": "https://files.pythonhosted.org/packages/aa/bb/synthetic_stage41b-1.0-py3-none-any.whl",
    }
    return raw, record, members


def _proof(files):
    hierarchy = {}
    for path, content in files.items():
        node = hierarchy
        parts = path.split("/")
        for component in parts[:-1]:
            node = node.setdefault(component, {})
        node[parts[-1]] = content
    trees, blobs = {}, {}

    def tree(node, prefix=""):
        entries = []
        for name, content in sorted(node.items()):
            if type(content) is dict:
                child = tree(content, prefix + name + "/")
                mode = b"40000"
            else:
                child = preparation.git_oid("blob", content)
                blobs[prefix + name] = {"oid": child, "content_b64": _b64(content)}
                mode = b"100644"
            entries.append(mode + b" " + name.encode() + b"\0" + bytes.fromhex(child))
        raw = b"".join(entries)
        oid = preparation.git_oid("tree", raw)
        trees[oid] = _b64(raw)
        return oid

    root = tree(hierarchy)
    raw_commit = (
        "tree "
        + root
        + "\nauthor Stage41B <unit@example.invalid> 0 +0000\ncommitter Stage41B <unit@example.invalid> 0 +0000\n\nsynthetic fixture\n"
    ).encode()
    commit = preparation.git_oid("commit", raw_commit)
    return {
        "schema": 1,
        "commit": commit,
        "commit_b64": _b64(raw_commit),
        "trees": trees,
        "files": blobs,
    }


@pytest.fixture
def bundle_fixture(tmp_path):
    wheel, record, _members = _wheel()
    lock = {"schema": 1, "python": "3.12", "platform": "linux-x86_64", "wheels": [record]}
    files = {
        preparation.INSTALLER: Path(preparation.__file__).read_bytes(),
        preparation.LOCK: _json(lock),
        preparation.REQUIREMENTS: (
            record["name"] + "==1.0 --hash=sha256:" + record["sha256"] + "\n"
        ).encode(),
        "deployment/stage41b_role_worker.py": b'"""Synthetic fixture module."""\n',
        "scripts/qualify_stage41b_signer_isolation.py": b'"""Synthetic fixture harness."""\n',
        "bot_core/__init__.py": b'"""Synthetic original source fixture."""\n',
    }
    proof = _proof(files)
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "bundle.json").write_bytes(_json(proof))
    (bundle / "wheels").mkdir()
    (bundle / "wheels" / record["filename"]).write_bytes(wheel)
    return SimpleNamespace(
        bundle=bundle, proof=proof, commit=proof["commit"], files=files, wheel=wheel, record=record
    )


def test_proof_authenticates_exact_commit_tree_and_original_file_bytes(bundle_fixture):
    fixture = bundle_fixture
    assert preparation.verify_proof(fixture.proof, fixture.commit) == fixture.files


@pytest.mark.parametrize("field", ["commit_b64", "trees", "files"])
def test_proof_refuses_tampered_commit_tree_or_blob(bundle_fixture, field):
    fixture = bundle_fixture
    proof = deepcopy(fixture.proof)
    if field == "commit_b64":
        proof[field] = _b64(b"tree " + b"0" * 40 + b"\n")
    elif field == "trees":
        key = next(iter(proof[field]))
        proof[field][key] = _b64(b"invalid modified tree")
    else:
        proof[field][preparation.INSTALLER]["content_b64"] = _b64(b"modified installer")
    with pytest.raises(preparation.PreparationBlocked):
        preparation.verify_proof(proof, fixture.commit)


@pytest.mark.parametrize(
    "change",
    [
        "missing_file",
        "extra_file",
        "missing_tree",
        "extra_tree",
        "wrong_oid",
        "wrong_commit",
        "boolean_schema",
    ],
)
def test_proof_refuses_incomplete_extra_or_malformed_evidence(bundle_fixture, change):
    fixture = bundle_fixture
    proof = deepcopy(fixture.proof)
    if change == "missing_file":
        proof["files"].pop(preparation.INSTALLER)
    elif change == "extra_file":
        proof["files"]["deployment/examples/root-proof-signer-custody.json"] = {
            "oid": "a" * 40,
            "content_b64": _b64(b"forbidden"),
        }
    elif change == "missing_tree":
        proof["trees"].pop(next(iter(proof["trees"])))
    elif change == "extra_tree":
        proof["trees"]["a" * 40] = _b64(b"")
    elif change == "wrong_oid":
        proof["files"][preparation.INSTALLER]["oid"] = "a" * 40
    elif change == "wrong_commit":
        proof["commit"] = "a" * 40
    else:
        proof["schema"] = True
    with pytest.raises(preparation.PreparationBlocked):
        preparation.verify_proof(proof, fixture.commit)


@pytest.mark.parametrize(
    "commit",
    ["main", "a" * 39, "A" * 40, "a" * 41, None, True],
    ids=["mutable-ref", "short-sha", "uppercase-sha", "long-sha", "missing-sha", "boolean-sha"],
)
def test_only_complete_exact_reviewed_sha_is_accepted(commit):
    with pytest.raises(preparation.PreparationBlocked):
        preparation.require_commit(commit)


@pytest.mark.parametrize(
    "name",
    [b"../escape", b"x/y", b".", b"..", b"bad\\name"],
    ids=["parent-traversal", "nested-path", "dot", "dot-dot", "backslash"],
)
def test_unsafe_git_tree_paths_are_rejected(name):
    with pytest.raises(preparation.PreparationBlocked):
        preparation._tree_entries(b"100644 " + name + b"\0" + b"a" * 20)


def test_source_selection_excludes_production_units_configurations_and_caches():
    for path in (
        "deployment/systemd/signer.service",
        "deployment/examples/root-proof-signer-custody.json",
        "config.toml",
        "bot_core/private.key",
        "bot_core/__pycache__/x.pyc",
        ".git/config",
    ):
        assert preparation.selected_source(path) is False
    assert preparation.selected_source("bot_core/security/keyring_storage.py") is True


def test_source_proof_reads_clean_git_objects_and_refuses_dirty_or_wrong_head(
    tmp_path, bundle_fixture
):
    repository = tmp_path / "source"
    repository.mkdir()
    for relative, content in bundle_fixture.files.items():
        path = repository / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)

    def git(*args):
        return subprocess.run(
            ["git", "-C", str(repository), *args], check=True, capture_output=True
        ).stdout

    git("init", "--quiet")
    git("add", ".")
    git(
        "-c",
        "user.name=Stage41B",
        "-c",
        "user.email=unit@example.invalid",
        "commit",
        "-qm",
        "synthetic source",
    )
    commit = git("rev-parse", "HEAD").decode().strip()
    proof = preparation.source_proof(repository, commit)
    assert preparation.verify_proof(proof, commit) == bundle_fixture.files
    with pytest.raises(preparation.PreparationBlocked, match="SOURCE_HEAD_MISMATCH"):
        preparation.source_proof(repository, "a" * 40)
    (repository / "untracked.tmp").write_text("untrusted artifact")
    with pytest.raises(preparation.PreparationBlocked, match="CLEAN_SOURCE_REPOSITORY_REQUIRED"):
        preparation.source_proof(repository, commit)
    (repository / "untracked.tmp").unlink()
    (repository / preparation.INSTALLER).write_text("modified source")
    with pytest.raises(preparation.PreparationBlocked, match="CLEAN_SOURCE_REPOSITORY_REQUIRED"):
        preparation.source_proof(repository, commit)


def test_checked_in_runtime_lock_matches_requirements_and_is_complete():
    root = Path(__file__).resolve().parents[2]
    entries = preparation.wheel_lock((root / preparation.LOCK).read_bytes())
    expected = {
        item["name"] + "==" + item["version"] + " --hash=sha256:" + item["sha256"]
        for item in entries
    }
    requirements = {
        line
        for line in (root / preparation.REQUIREMENTS).read_text().splitlines()
        if line and not line.startswith("#")
    }
    assert requirements == expected
    assert {item["name"].lower() for item in entries} >= {
        "cryptography",
        "keyring",
        "secretstorage",
        "jeepney",
        "cffi",
    }
    assert len(entries) == 10


@pytest.mark.parametrize(
    "field,value",
    [
        ("url", "http://files.pythonhosted.org/packages/test.whl"),
        ("url", "https://evil.invalid/packages/test.whl"),
        ("url", "https://files.pythonhosted.org/packages/test.whl?token=bad"),
        ("sha256", "A" * 64),
        ("filename", "../escape.whl"),
    ],
    ids=["insecure-http", "foreign-host", "query-string", "uppercase-digest", "parent-traversal"],
)
def test_uncontrolled_wheel_sources_and_artifacts_are_rejected(bundle_fixture, field, value):
    document = json.loads(bundle_fixture.files[preparation.LOCK])
    document["wheels"][0][field] = value
    with pytest.raises(preparation.PreparationBlocked):
        preparation.wheel_lock(_json(document))


@pytest.mark.parametrize("change", ["duplicate", "platform", "python", "schema", "empty"])
def test_unsupported_or_duplicate_wheel_lock_fails_closed(bundle_fixture, change):
    document = json.loads(bundle_fixture.files[preparation.LOCK])
    if change == "duplicate":
        document["wheels"].append(deepcopy(document["wheels"][0]))
    elif change in {"platform", "python"}:
        document[change] = "unsupported"
    elif change == "schema":
        document[change] = True
    else:
        document["wheels"] = []
    with pytest.raises(preparation.PreparationBlocked):
        preparation.wheel_lock(_json(document))


def test_wheel_record_authenticates_all_extracted_member_bytes():
    raw, record, members = _wheel()
    assert preparation.wheel_members(raw, record) == members


@pytest.mark.parametrize(
    "path",
    [
        "../escape",
        "/absolute",
        "pkg/back\\slash.py",
        "startup.pth",
        "pkg/cached.pyc",
        "package.data/scripts/execute",
    ],
    ids=[
        "parent-traversal",
        "absolute-path",
        "backslash",
        "startup-hook",
        "bytecode",
        "installed-script",
    ],
)
def test_wheel_rejects_startup_hooks_and_path_traversal(path):
    raw, record, _members = _wheel(extra={path: b"untrusted"})
    with pytest.raises(preparation.PreparationBlocked, match="UNSAFE_WHEEL_MEMBER"):
        preparation.wheel_members(raw, record)


def test_wheel_rejects_symlink_members_and_forged_record():
    raw, record, _members = _wheel(member_type="synthetic_stage41b/__init__.py")
    with pytest.raises(preparation.PreparationBlocked, match="UNSAFE_WHEEL_MEMBER"):
        preparation.wheel_members(raw, record)
    raw, record, _members = _wheel(alter_record=True)
    with pytest.raises(preparation.PreparationBlocked, match="WHEEL_RECORD_INTEGRITY_FAILURE"):
        preparation.wheel_members(raw, record)
    with pytest.raises(preparation.PreparationBlocked, match="WHEEL_INTEGRITY_FAILURE"):
        preparation.wheel_members(raw + b"modified", record)


def test_wheel_distribution_must_match_reviewed_record():
    raw, record, _members = _wheel()
    record["version"] = "9.0"
    with pytest.raises(preparation.PreparationBlocked, match="WHEEL_DISTRIBUTION_MISMATCH"):
        preparation.wheel_members(raw, record)


def test_read_rejects_symlinks_hardlinks_nonregular_and_oversized_artifacts(tmp_path):
    source = tmp_path / "source"
    source.write_bytes(b"public test artifact")
    assert preparation._read(source) == source.read_bytes()
    with pytest.raises(preparation.PreparationBlocked):
        preparation._read(source, limit=3)
    alias = tmp_path / "symlink"
    alias.symlink_to(source)
    with pytest.raises(OSError):
        preparation._read(alias)
    hardlink = tmp_path / "hardlink"
    os.link(source, hardlink)
    with pytest.raises(preparation.PreparationBlocked):
        preparation._read(source)
    with pytest.raises((preparation.PreparationBlocked, OSError)):
        preparation._read(tmp_path)


def test_write_new_never_overwrites_existing_file_or_symlink(tmp_path):
    target = tmp_path / "nested/file"
    preparation._write_new(target, b"first")
    with pytest.raises(FileExistsError):
        preparation._write_new(target, b"second")
    assert target.read_bytes() == b"first"


def test_write_new_defaults_to_owner_only_even_with_permissive_umask(tmp_path):
    previous = os.umask(0)
    try:
        target = tmp_path / "private-manifest.json"
        preparation._write_new(target, b'{"synthetic": true}')
    finally:
        os.umask(previous)
    assert stat.S_IMODE(target.stat().st_mode) == 0o600


def test_public_runtime_is_private_until_complete_bytes_are_synced(tmp_path, monkeypatch):
    target = tmp_path / "verified-public-code.py"
    content = b"# synthetic authenticated public code\n"
    observed_modes = []
    fsync = os.fsync

    def observe(descriptor):
        observed_modes.append(stat.S_IMODE(os.fstat(descriptor).st_mode))
        assert target.read_bytes() == content
        fsync(descriptor)

    monkeypatch.setattr(preparation.os, "fsync", observe)
    preparation._write_new(target, content, 0o644)
    assert observed_modes == [0o600, 0o644]


def test_failed_content_sync_keeps_public_artifact_private(tmp_path, monkeypatch):
    target = tmp_path / "incomplete-public-code.py"

    def fail_content_sync(_descriptor):
        raise OSError("synthetic content sync failure")

    monkeypatch.setattr(preparation.os, "fsync", fail_content_sync)
    with pytest.raises(OSError, match="synthetic content sync failure"):
        preparation._write_new(target, b"# synthetic public code\n", 0o644)
    assert stat.S_IMODE(target.stat().st_mode) == 0o600


def test_writer_refuses_unreviewed_permission_modes_before_creating_files(tmp_path):
    target = tmp_path / "uncreated/artifact"
    with pytest.raises(preparation.PreparationBlocked, match="SUPPORTED_ARTIFACT_MODE_REQUIRED"):
        preparation._write_new(target, b"synthetic", 0o666)
    assert not target.parent.exists()


@pytest.fixture
def unprivileged_install_logic(monkeypatch, tmp_path):
    """Model root metadata only within this private test tree; no real /opt writes."""
    prefix = tmp_path / "protected-opt"
    interpreter = tmp_path / "system-python-fixture"
    interpreter.write_bytes(b"synthetic interpreter fixture; never executed")
    original_read = preparation._read

    def read(path, limit=preparation.MAX_FILE):
        return original_read(interpreter if path == Path("/usr/bin/python3.12") else path, limit)

    monkeypatch.setattr(preparation, "_read", read)
    monkeypatch.setattr(preparation, "PREFIX", prefix)
    monkeypatch.setattr(preparation, "_host_installation_guard", lambda: None)

    def protected(path, *, directory):
        info = (interpreter if path == Path("/usr/bin/python3.12") else path).lstat()
        assert not stat.S_ISLNK(info.st_mode)
        assert (stat.S_ISDIR if directory else stat.S_ISREG)(info.st_mode)
        return info

    monkeypatch.setattr(preparation, "_protected", protected)

    class FakeVenv:
        def __init__(self, **options):
            assert options == {"with_pip": False, "symlinks": False}

        def create(self, root):
            (root / "bin").mkdir(parents=True)
            (root / "bin/python").write_bytes(interpreter.read_bytes())
            (root / "pyvenv.cfg").write_text(
                "include-system-site-packages = false\nversion = 3.12.0\nexecutable = /usr/bin/python3.12\n"
            )

    monkeypatch.setattr(preparation.venv, "EnvBuilder", FakeVenv)
    return prefix


def test_offline_install_verifies_proof_and_wheels_without_starting_services(
    bundle_fixture, unprivileged_install_logic
):
    fixture = bundle_fixture
    result = preparation.install_bundle(fixture.bundle, fixture.commit)
    assert result["status"] == "PREPARED_ENVIRONMENT_VERIFIED"
    assert result["physical_executed"] is False
    root = unprivileged_install_logic / fixture.commit
    assert preparation.verify_installation(root, fixture.commit) == result
    assert preparation.install_bundle(fixture.bundle, fixture.commit) == result
    assert not (root / "source/deployment/systemd").exists()
    assert not (root / "source/deployment/examples").exists()


def test_installation_keeps_metadata_private_and_verified_runtime_readable(
    bundle_fixture, unprivileged_install_logic
):
    fixture = bundle_fixture
    previous = os.umask(0o077)
    try:
        preparation.install_bundle(fixture.bundle, fixture.commit)
    finally:
        os.umask(previous)
    root = unprivileged_install_logic / fixture.commit
    private = [
        root / "preparation-origin.json",
        root / "bundle.json",
        root / "installation.json",
        root / "artifacts/wheels" / fixture.record["filename"],
    ]
    for path in private:
        assert stat.S_IMODE(path.stat().st_mode) == 0o600, path.name
    for relative, expected in fixture.files.items():
        path = root / "source" / relative
        assert path.read_bytes() == expected
        assert stat.S_IMODE(path.stat().st_mode) == 0o644, relative
    members = preparation.wheel_members(fixture.wheel, fixture.record)
    for relative, expected in members.items():
        path = root / "venv/lib/python3.12/site-packages" / relative
        assert path.read_bytes() == expected
        assert stat.S_IMODE(path.stat().st_mode) == 0o644, relative


@pytest.mark.parametrize("artifact", ["origin", "bundle", "receipt", "archive"])
def test_verification_refuses_disclosure_of_private_installation_metadata(
    bundle_fixture, unprivileged_install_logic, artifact
):
    fixture = bundle_fixture
    preparation.install_bundle(fixture.bundle, fixture.commit)
    root = unprivileged_install_logic / fixture.commit
    target = {
        "origin": root / "preparation-origin.json",
        "bundle": root / "bundle.json",
        "receipt": root / "installation.json",
        "archive": root / "artifacts/wheels" / fixture.record["filename"],
    }[artifact]
    target.chmod(0o644)
    with pytest.raises(preparation.PreparationBlocked, match="PRIVATE_ARTIFACT_MODE_REQUIRED"):
        preparation.verify_installation(root, fixture.commit)


@pytest.mark.parametrize("change", ["source", "wheel", "extra", "pyc", "symlink", "receipt"])
def test_verification_refuses_modified_or_unexpected_installed_artifacts(
    bundle_fixture, unprivileged_install_logic, change
):
    fixture = bundle_fixture
    preparation.install_bundle(fixture.bundle, fixture.commit)
    root = unprivileged_install_logic / fixture.commit
    if change == "source":
        (root / "source" / preparation.INSTALLER).write_bytes(b"modified source")
    elif change == "wheel":
        path = root / "venv/lib/python3.12/site-packages/synthetic_stage41b/__init__.py"
        path.write_bytes(b"modified library")
    elif change == "extra":
        (root / "unexpected.file").write_bytes(b"unreviewed")
    elif change == "pyc":
        (root / "unexpected.pyc").write_bytes(b"unreviewed cache")
    elif change == "symlink":
        (root / "untrusted-link").symlink_to(root / "source" / preparation.INSTALLER)
    else:
        receipt = json.loads((root / "installation.json").read_bytes())
        receipt["commit"] = "a" * 40
        (root / "installation.json").write_bytes(_json(receipt))
    with pytest.raises((preparation.PreparationBlocked, AssertionError)):
        preparation.verify_installation(root, fixture.commit)


def test_installation_never_reuses_incomplete_existing_scope(
    bundle_fixture, unprivileged_install_logic
):
    root = unprivileged_install_logic / bundle_fixture.commit
    root.mkdir(parents=True)
    marker = root / "unproven-file"
    marker.write_bytes(b"do not remove")
    with pytest.raises(OSError):
        preparation.install_bundle(bundle_fixture.bundle, bundle_fixture.commit)
    assert marker.read_bytes() == b"do not remove"


def test_installation_host_guard_runs_before_untrusted_artifact_read(monkeypatch, tmp_path):
    calls = []

    def denied():
        calls.append("guard")
        raise preparation.PreparationBlocked("TEST_HOST_GUARD")

    monkeypatch.setattr(preparation, "_host_installation_guard", denied)
    monkeypatch.setattr(
        preparation, "_read", lambda *_args: pytest.fail("artifact read before host guard")
    )
    with pytest.raises(preparation.PreparationBlocked, match="TEST_HOST_GUARD"):
        preparation.install_bundle(tmp_path, "a" * 40)
    assert calls == ["guard"]


def test_unprivileged_build_uses_verified_source_and_pinned_downloads(
    monkeypatch, tmp_path, bundle_fixture
):
    fixture = bundle_fixture
    source = tmp_path / "clean-source"
    source.mkdir()
    output = tmp_path / "new-bundle"
    monkeypatch.setattr(preparation, "source_proof", lambda path, commit: fixture.proof)
    requested = []

    class Opener:
        def open(self, url, *, timeout):
            requested.append((url, timeout))
            return io.BytesIO(fixture.wheel)

    monkeypatch.setattr(preparation, "build_opener", lambda *_args: Opener())
    result = preparation.build_bundle(source, fixture.commit, output)
    assert result["status"] == "BUNDLE_READY" and result["physical_executed"] is False
    assert requested == [(fixture.record["url"], 60)]
    assert (output / "wheels" / fixture.record["filename"]).read_bytes() == fixture.wheel
    with pytest.raises(FileExistsError):
        preparation.build_bundle(source, fixture.commit, output)


def test_build_rejects_root_source_inside_destination_and_changed_wheel(
    monkeypatch, tmp_path, bundle_fixture
):
    source = tmp_path / "source"
    source.mkdir()
    monkeypatch.setattr(preparation.os, "geteuid", lambda: 0)
    with pytest.raises(preparation.PreparationBlocked, match="UNPRIVILEGED_BUILD_REQUIRED"):
        preparation.build_bundle(source, bundle_fixture.commit, tmp_path / "bundle")
    monkeypatch.setattr(preparation.os, "geteuid", lambda: 1000)
    with pytest.raises(preparation.PreparationBlocked, match="BUNDLE_MUST_BE_OUTSIDE_SOURCE"):
        preparation.build_bundle(source, bundle_fixture.commit, source / "bundle")
    monkeypatch.setattr(preparation, "source_proof", lambda *_args: bundle_fixture.proof)

    class Opener:
        def open(self, *_args, **_kwargs):
            return io.BytesIO(b"modified untrusted download")

    monkeypatch.setattr(preparation, "build_opener", lambda *_args: Opener())
    with pytest.raises(preparation.PreparationBlocked, match="DOWNLOADED_WHEEL_INTEGRITY_FAILURE"):
        preparation.build_bundle(source, bundle_fixture.commit, tmp_path / "failed-bundle")
    assert (tmp_path / "failed-bundle/bundle.json").exists()


@pytest.mark.parametrize(
    "raw",
    [b'{"schema":1,"schema":1}', b'{"x":NaN}', b'{"x":Infinity}', b'{"x":1e999}'],
    ids=["duplicate-field", "nan", "infinity", "float-overflow"],
)
def test_json_refuses_duplicate_or_nonfinite_metadata(raw):
    with pytest.raises(preparation.PreparationBlocked):
        preparation._json(raw)


def test_main_modes_are_explicit_sanitized_and_never_execute_harness(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(
        preparation,
        "build_bundle",
        lambda *_args: {"status": "BUNDLE_READY", "physical_executed": False},
    )
    assert (
        preparation.main(
            [
                "--build",
                "--source",
                str(tmp_path),
                "--bundle",
                str(tmp_path / "bundle"),
                "--commit",
                "a" * 40,
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["physical_executed"] is False
    monkeypatch.setattr(
        preparation,
        "install_bundle",
        lambda *_args: {"status": "PREPARED_ENVIRONMENT_VERIFIED", "physical_executed": False},
    )
    assert preparation.main(["--install", "--bundle", str(tmp_path), "--commit", "a" * 40]) == 0
    assert json.loads(capsys.readouterr().out)["physical_executed"] is False
    monkeypatch.setattr(
        preparation,
        "verify_installation",
        lambda *_args: {"status": "PREPARED_ENVIRONMENT_VERIFIED", "physical_executed": False},
    )
    assert preparation.main(["--verify-installed", "--commit", "a" * 40]) == 0
    capsys.readouterr()

    def unavailable(*_args):
        raise RuntimeError("UNTRUSTED PRIVATE BACKEND TEXT")

    monkeypatch.setattr(preparation, "install_bundle", unavailable)
    assert preparation.main(["--install", "--bundle", str(tmp_path), "--commit", "a" * 40]) == 2
    report = capsys.readouterr().out
    assert "UNTRUSTED PRIVATE" not in report
    assert json.loads(report)["status"] == "BLOCKED"
    assert preparation.main(["--build", "--commit", "a" * 40]) == 2
    assert json.loads(capsys.readouterr().out)["reason"] == "SOURCE_AND_NEW_BUNDLE_REQUIRED"


def test_invalid_artifact_read_always_closes_descriptor(tmp_path):
    descriptor = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
    with pytest.raises(preparation.PreparationBlocked):
        preparation._read_descriptor(descriptor, preparation.MAX_FILE)
    with pytest.raises(OSError):
        os.fstat(descriptor)


def test_bundle_directory_and_relative_artifact_reads_reject_symlink_ancestors(tmp_path):
    actual = tmp_path / "actual"
    actual.mkdir()
    (actual / "record").write_bytes(b"pinned public bytes")
    alias = tmp_path / "alias"
    alias.symlink_to(actual, target_is_directory=True)
    with pytest.raises(OSError):
        preparation._open_directory(alias)
    with pytest.raises(preparation.PreparationBlocked):
        preparation._open_directory(Path("relative"))
    descriptor = preparation._open_directory(actual)
    try:
        assert preparation._read_at(descriptor, "record") == b"pinned public bytes"
        with pytest.raises(preparation.PreparationBlocked):
            preparation._read_at(descriptor, "../escape")
    finally:
        os.close(descriptor)


def test_redirected_wheel_download_is_refused():
    with pytest.raises(preparation.PreparationBlocked, match="WHEEL_REDIRECT_FORBIDDEN"):
        preparation._NoRedirect().redirect_request(
            None, None, 302, "redirect", {}, "https://other.invalid/file"
        )


@pytest.mark.parametrize(
    "uid,mode", [(1000, stat.S_IFREG | 0o644), (0, stat.S_IFREG | 0o664), (0, stat.S_IFLNK | 0o777)]
)
def test_protected_file_requires_exact_root_owner_regular_type_and_no_untrusted_writes(
    monkeypatch, tmp_path, uid, mode
):
    monkeypatch.setattr(Path, "lstat", lambda _self: SimpleNamespace(st_uid=uid, st_mode=mode))
    with pytest.raises(preparation.PreparationBlocked, match="ROOT_OWNED_PROTECTED_PATH_REQUIRED"):
        preparation._protected(tmp_path / "fixture", directory=False)


def test_real_host_guard_refuses_unprivileged_process_before_root_paths(monkeypatch):
    monkeypatch.setattr(preparation.os, "geteuid", lambda: 1000)
    monkeypatch.setattr(
        preparation,
        "_protected",
        lambda *_args, **_kwargs: pytest.fail("root path inspected before privilege guard"),
    )
    with pytest.raises(
        preparation.PreparationBlocked, match="ROOT_UBUNTU_PYTHON_3_12_X86_64_REQUIRED"
    ):
        preparation._host_installation_guard()


@pytest.mark.parametrize(
    "field,value,reason",
    [
        ("include-system-site-packages", "true", "ISOLATED_VENV_CONFIGURATION_REQUIRED"),
        ("version", "3.13.0", "ISOLATED_VENV_CONFIGURATION_REQUIRED"),
        ("executable", "/tmp/untrusted-python", "VENV_SYSTEM_INTERPRETER_MISMATCH"),
    ],
    ids=["system-site-packages", "wrong-python-version", "foreign-interpreter"],
)
def test_even_self_consistent_receipt_requires_isolated_pinned_system_python(
    bundle_fixture, unprivileged_install_logic, field, value, reason
):
    fixture = bundle_fixture
    preparation.install_bundle(fixture.bundle, fixture.commit)
    root = unprivileged_install_logic / fixture.commit
    config = root / "venv/pyvenv.cfg"
    settings = dict(line.split(" = ", 1) for line in config.read_text().splitlines())
    settings[field] = value
    config.write_text("\n".join(name + " = " + item for name, item in settings.items()) + "\n")
    receipt = json.loads((root / "installation.json").read_bytes())
    receipt["files"]["venv/pyvenv.cfg"] = hashlib.sha256(config.read_bytes()).hexdigest()
    (root / "installation.json").write_bytes(_json(receipt))
    with pytest.raises(preparation.PreparationBlocked, match=reason):
        preparation.verify_installation(root, fixture.commit)


def test_installation_rejects_changed_wheel_before_creating_protected_scope(
    bundle_fixture, unprivileged_install_logic
):
    fixture = bundle_fixture
    (fixture.bundle / "wheels" / fixture.record["filename"]).write_bytes(b"untrusted replacement")
    with pytest.raises(preparation.PreparationBlocked, match="WHEEL_INTEGRITY_FAILURE"):
        preparation.install_bundle(fixture.bundle, fixture.commit)
    assert not unprivileged_install_logic.exists()


@pytest.mark.parametrize("schema", [True, 1.0])
def test_installation_receipt_schema_requires_exact_integer(
    bundle_fixture, unprivileged_install_logic, schema
):
    fixture = bundle_fixture
    preparation.install_bundle(fixture.bundle, fixture.commit)
    root = unprivileged_install_logic / fixture.commit
    receipt_path = root / "installation.json"
    receipt = json.loads(receipt_path.read_bytes())
    receipt["schema"] = schema
    receipt_path.write_bytes(_json(receipt))
    with pytest.raises(preparation.PreparationBlocked, match="INSTALLATION_RECEIPT_MISMATCH"):
        preparation.verify_installation(root, fixture.commit)


def test_invalid_commit_never_produces_a_manual_preserved_installation_path(capsys):
    assert preparation.main(["--install", "--commit", "../../etc"]) == 2
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "BLOCKED"
    assert report["reason"] == "FULL_REVIEWED_COMMIT_REQUIRED"
    assert report["preserved_installation"] is None
    assert report["physical_executed"] is False

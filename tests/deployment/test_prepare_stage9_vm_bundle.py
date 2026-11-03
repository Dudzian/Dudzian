"""Rootless eksport v2 z rzeczywistych lokalnych Git blobs; bez instalacji."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from scripts import prepare_stage9_vm_bundle as exporter

pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="Rootless eksporter VM Linux")


def git(source: Path, *arguments: str) -> str:
    return subprocess.check_output(
        ["/usr/bin/git", "-C", str(source), *arguments],
        env={"PATH": "/usr/bin:/bin", "GIT_CONFIG_NOSYSTEM": "1"},
        stderr=subprocess.DEVNULL,
        text=True,
    ).strip()


def commit(source: Path) -> str:
    git(source, "add", ".")
    git(
        source,
        "-c",
        "user.name=Fixture",
        "-c",
        "user.email=fixture@example.invalid",
        "commit",
        "-m",
        "fixture",
    )
    return git(source, "rev-parse", "HEAD")


@pytest.fixture
def candidate(tmp_path: Path, monkeypatch):
    # Source and artifacts are disposable data; the installer is never imported.
    monkeypatch.setattr(exporter.os, "geteuid", lambda: 1000)
    original_lstat = Path.lstat

    def ubuntu_system_ownership(path: Path, *args, **kwargs):
        info = original_lstat(path, *args, **kwargs)
        if path in {Path("/"), Path("/tmp")}:
            # The executor remaps base-image ownership to nobody. Model only
            # these immutable system ancestors as on Ubuntu; Git and candidate
            # files, permissions, symlinks and output writes remain real.
            fields = list(info)
            fields[4] = 0
            return os.stat_result(fields)
        return info

    monkeypatch.setattr(Path, "lstat", ubuntu_system_ownership)
    source = tmp_path / "source"
    source.mkdir()
    git(source, "init")
    required = {
        "bot_core/__init__.py",
        "bot_core/test_extra.py",
        "deployment/__init__.py",
        exporter.UNIT,
        "deployment/freshness_runtime.py",
        "deployment/freshness_runtime_bootstrap.py",
    }
    template = (
        b"[Service]\nExecStartPre=/usr/bin/python3 bootstrap.py verify-installed\n"
        b"ExecStart=@FRESHNESS_RUNTIME_PYTHON@ -I -s -B entrypoint.py\n"
    )
    values = {
        relative: b"# reviewed fixture\n" for relative in required | set(exporter.BOOTSTRAP_FILES)
    }
    values["deployment/production_local_installer.py"] = (
        "raise RuntimeError('AST data must never execute')\n"
        "SCOPE='cryptohunter.freshness.substrate.v2'\n"
        f"REQUIRED_FILES=frozenset({required!r})\n"
    ).encode()
    values["deployment/freshness_runtime.py"] = f"UNIT_TEMPLATE={template!r}\n".encode()
    values[exporter.UNIT] = template
    for relative, value in values.items():
        target = source / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(value)
    (source / "deployment/install-production-local.sh").chmod(0o755)
    revision = commit(source)
    binding = {
        "schema": 1,
        "environment_id": "test-only-vm-001",
        "manifest_sha256": "a" * 64,
        "source_revision": revision,
        "executable": (
            "/opt/cryptohunter/freshness-runtime/releases/test-only-vm-001/bin/python3.12"
        ),
        "base_executable": "/usr/bin/python3.12",
        "base_executable_sha256": "b" * 64,
    }
    binding_path = tmp_path / "binding.json"
    binding_path.write_text(json.dumps(binding))
    return source, revision, binding_path, tmp_path / "new-candidate", binding, required, template


def prepare(candidate):
    return exporter.prepare(*candidate[:4])


def test_exports_exact_git_sources_dynamic_inventory_and_v2_binding(candidate):
    source, revision, binding_path, output, binding, required, template = candidate
    result = prepare(candidate)
    manifest = json.loads((output / "release.json").read_bytes())
    assert result["result"] == "UNSIGNED_TEST_ONLY_CANDIDATE"
    assert manifest["schema_version"] == 2
    assert manifest["scope"] == "cryptohunter.freshness.substrate.v2"
    assert manifest["source_revision"] == revision
    assert manifest["runtime"] == {key: value for key, value in binding.items() if key != "schema"}
    assert {item["path"] for item in manifest["files"]} == required
    assert (output / "payload" / exporter.UNIT).read_bytes() == template.replace(
        exporter.TOKEN, binding["executable"].encode()
    )
    for item in manifest["files"]:
        target = output / "payload" / item["path"]
        assert hashlib.sha256(target.read_bytes()).hexdigest() == item["sha256"]
        assert target.stat().st_size == item["size"]
        assert target.stat().st_mode & 0o777 == item["mode"]
        if item["path"] != exporter.UNIT:
            assert target.read_bytes() == (source / item["path"]).read_bytes()
    for relative in exporter.BOOTSTRAP_FILES:
        assert (output / "bootstrap" / relative).read_bytes() == (source / relative).read_bytes()
    assert len((output / "bootstrap.sha256").read_text().splitlines()) == 5
    assert (
        output / "bootstrap/deployment/install-production-local.sh"
    ).stat().st_mode & 0o777 == 0o755
    assert not list(output.rglob("*.sig"))
    assert not list(output.rglob("*.pem"))
    assert binding_path.read_text() == json.dumps(binding)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("schema", True),
        ("schema", 2),
        ("environment_id", "../foreign"),
        ("executable", "/usr/bin/python3.12"),
        ("source_revision", "f" * 40),
        ("base_executable", "/usr/bin/python3"),
        ("manifest_sha256", "wrong"),
        ("base_executable_sha256", None),
        ("unexpected", True),
    ],
)
def test_invalid_binding_refuses_before_writing(candidate, key, value):
    candidate[4][key] = value
    candidate[2].write_text(json.dumps(candidate[4]))
    with pytest.raises(exporter.BundleError):
        prepare(candidate)
    assert not candidate[3].exists()


def test_duplicate_binding_field_refused(candidate):
    candidate[2].write_text('{"schema":1,"schema":1}')
    with pytest.raises(exporter.BundleError, match="DUPLICATE_BINDING_FIELD"):
        prepare(candidate)
    assert not candidate[3].exists()


def test_existing_output_retained(candidate):
    candidate[3].mkdir()
    marker = candidate[3] / "foreign-data"
    marker.write_bytes(b"preserve")
    with pytest.raises(exporter.BundleError, match="OUTPUT_ALREADY_EXISTS"):
        prepare(candidate)
    assert marker.read_bytes() == b"preserve"


@pytest.mark.parametrize("mode", [0o775, 0o777])
def test_writable_output_ancestor_refuses_before_export(candidate, tmp_path, mode):
    parent = tmp_path / "shared-parent"
    parent.mkdir()
    parent.chmod(mode)
    output = parent / "new-candidate"
    try:
        with pytest.raises(exporter.BundleError, match="OUTPUT_PARENT_UNPROTECTED"):
            exporter.prepare(candidate[0], candidate[1], candidate[2], output)
        assert not output.exists()
    finally:
        parent.chmod(0o700)


def test_foreign_owned_output_ancestor_refused(candidate, tmp_path, monkeypatch):
    parent = tmp_path / "foreign-owner"
    parent.mkdir(mode=0o755)
    original_lstat = Path.lstat

    def foreign_owner(path: Path, *args, **kwargs):
        info = original_lstat(path, *args, **kwargs)
        if path == parent:
            fields = list(info)
            fields[4] = 12345
            return os.stat_result(fields)
        return info

    monkeypatch.setattr(Path, "lstat", foreign_owner)
    with pytest.raises(exporter.BundleError, match="OUTPUT_PARENT_UNPROTECTED"):
        exporter.prepare(candidate[0], candidate[1], candidate[2], parent / "new-candidate")
    assert not (parent / "new-candidate").exists()


def test_tmp_without_sticky_bit_refused(candidate, monkeypatch):
    original_lstat = Path.lstat

    def without_sticky(path: Path, *args, **kwargs):
        info = original_lstat(path, *args, **kwargs)
        if path == Path("/tmp"):
            fields = list(info)
            fields[0] &= ~0o1000
            return os.stat_result(fields)
        return info

    monkeypatch.setattr(Path, "lstat", without_sticky)
    with pytest.raises(exporter.BundleError, match="OUTPUT_PARENT_UNPROTECTED"):
        prepare(candidate)
    assert not candidate[3].exists()


@pytest.mark.parametrize("location", ["binding", "output", "source-file", "parent"])
def test_symlinks_refused(candidate, tmp_path, location):
    source, revision, binding_path, output, *_ = candidate
    if location == "binding":
        linked = tmp_path / "linked-binding"
        linked.symlink_to(binding_path)
        binding_path = linked
    elif location == "output":
        output.symlink_to(tmp_path / "missing-foreign-dir")
    elif location == "parent":
        linked = tmp_path / "linked-parent"
        linked.symlink_to(tmp_path, target_is_directory=True)
        output = linked / "new-candidate"
    else:
        path = source / "bot_core/test_extra.py"
        original = tmp_path / "other.py"
        path.rename(original)
        path.symlink_to(original)
    with pytest.raises(exporter.BundleError):
        exporter.prepare(source, revision, binding_path, output)


def test_changed_worktree_refuses(candidate):
    (candidate[0] / "bot_core/test_extra.py").write_bytes(b"# changed\n")
    with pytest.raises(exporter.BundleError, match="WORKTREE_SOURCE_CHANGED"):
        prepare(candidate)
    assert not candidate[3].exists()


def test_git_replace_cannot_substitute_approved_commit(candidate):
    source, revision, binding_path, output, *_ = candidate
    relative = "bot_core/test_extra.py"
    original = (source / relative).read_bytes()
    substituted = b"# foreign replacement must not acquire the approved revision\n"
    (source / relative).write_bytes(substituted)
    replacement_revision = commit(source)
    # Keep the literal approved HEAD, but retain the foreign working tree and
    # a real refs/replace mapping. Ordinary Git reads now report foreign bytes.
    git(source, "update-ref", "HEAD", revision)
    git(source, "replace", revision, replacement_revision)
    assert git(source, "rev-parse", "HEAD") == revision
    assert git(source, "show", revision + ":" + relative) == substituted.decode().strip()
    assert substituted != original
    with pytest.raises(exporter.BundleError, match="WORKTREE_SOURCE_CHANGED"):
        exporter.prepare(source, revision, binding_path, output)
    assert not output.exists()


def test_wrong_head_refuses(candidate):
    with pytest.raises(exporter.BundleError, match="HEAD_CHANGED"):
        exporter.prepare(candidate[0], "f" * 40, candidate[2], candidate[3])
    assert not candidate[3].exists()


@pytest.mark.parametrize("change", ["v1", "inventory-list", "inventory-traversal", "unit-drift"])
def test_unapproved_source_contract_refuses(candidate, change):
    source = candidate[0]
    path = source / "deployment/production_local_installer.py"
    content = path.read_text()
    if change == "v1":
        content = content.replace("substrate.v2", "substrate.v1")
    elif change == "inventory-list":
        content = content.replace("frozenset(", "list(")
    elif change == "inventory-traversal":
        content = content.replace("bot_core/test_extra.py", "bot_core/../foreign.py")
    else:
        (source / exporter.UNIT).write_bytes(b"[Service]\nExecStart=/bin/false\n")
    path.write_text(content)
    revision = commit(source)
    binding = candidate[4]
    binding["source_revision"] = revision
    candidate[2].write_text(json.dumps(binding))
    with pytest.raises(exporter.BundleError):
        exporter.prepare(source, revision, candidate[2], candidate[3])
    assert not candidate[3].exists()


def test_root_cannot_export(candidate, monkeypatch):
    monkeypatch.setattr(exporter.os, "geteuid", lambda: 0)
    with pytest.raises(exporter.BundleError, match="ROOTLESS_BUILDER_REQUIRED"):
        prepare(candidate)
    assert not candidate[3].exists()


def test_relative_output_refused(candidate):
    with pytest.raises(exporter.BundleError, match="ABSOLUTE_PATH_REQUIRED"):
        exporter.prepare(candidate[0], candidate[1], candidate[2], Path("relative"))


def test_cli_sanitizes_bad_json(candidate, capsys):
    candidate[2].write_bytes(b"invalid sensitive private fixture")
    result = exporter.main(
        [
            "--source",
            str(candidate[0]),
            "--revision",
            candidate[1],
            "--binding",
            str(candidate[2]),
            "--output",
            str(candidate[3]),
        ]
    )
    assert result == 1
    report = capsys.readouterr().out
    assert "sensitive" not in report
    assert json.loads(report)["reason"] == "INPUT_OR_EXPORT_REFUSED"
    assert not candidate[3].exists()


def test_restrictive_umask_affects_only_new_output(candidate):
    previous = os.umask(0o077)
    try:
        prepare(candidate)
    finally:
        os.umask(previous)
    output = candidate[3]
    assert output.stat().st_mode & 0o777 == 0o755
    assert all(path.stat().st_mode & 0o777 == 0o755 for path in output.rglob("*") if path.is_dir())
    assert (output / "release.json").stat().st_mode & 0o777 == 0o644

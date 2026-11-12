"""Runtime import records are checked before any source or native execution.

These tests model already authenticated manifest records. The sole executed
fixture is an exact copy of the host's trusted stdlib textwrap source; attack
fixtures remain data, including the synthetic bytecode and native objects.
"""

from __future__ import annotations

import hashlib
import importlib.machinery
import importlib.util
import stat
import sys
import textwrap
from pathlib import Path

import pytest

from deployment import stage9_vm_trial as trial

pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="Linux runtime import boundary")


@pytest.fixture
def runtime_source(tmp_path, monkeypatch):
    original_source = Path(textwrap.__file__).read_bytes()
    source = tmp_path / "trusted_textwrap.py"
    source.write_bytes(original_source)
    source.chmod(0o444)
    loader = trial._HarnessLoader({}, tmp_path)
    record = {
        "kind": "file",
        "mode": 0o444,
        "size": len(original_source),
        "sha256": hashlib.sha256(original_source).hexdigest(),
    }
    loader.runtime = {str(source): record}
    original_protected = trial._protected

    def model_owner_and_ancestors(path, *, directory=False):
        if path.is_relative_to(tmp_path):
            # Root ownership and protected ancestors are modeled; regular-file
            # type, permissions, link count and bytes use the real filesystem.
            info = path.lstat()
            trial._require(
                not stat.S_ISLNK(info.st_mode)
                and not info.st_mode & 0o022
                and (stat.S_ISDIR(info.st_mode) if directory else stat.S_ISREG(info.st_mode))
                and (directory or info.st_nlink == 1),
                "INPUT_NOT_PROTECTED",
            )
            return
        original_protected(path, directory=directory)

    monkeypatch.setattr(trial, "_protected", model_owner_and_ancestors)
    return loader, source, record, original_source


@pytest.mark.parametrize("attack", ["absent", "hash", "size", "mode", "kind", "path"])
def test_external_import_record_refuses_changed_or_unlisted_object(runtime_source, attack):
    loader, source, record, _ = runtime_source
    expected = "RUNTIME_IMPORT_INTEGRITY_CHANGED"
    if attack == "absent":
        loader.runtime.clear()
        expected = "RUNTIME_IMPORT_NOT_APPROVED"
    elif attack == "hash":
        record["sha256"] = "0" * 64
    elif attack == "size":
        record["size"] += 1
    elif attack == "mode":
        record["mode"] = 0o400
    elif attack == "kind":
        record["kind"] = "directory"
        expected = "RUNTIME_IMPORT_NOT_APPROVED"
    elif attack == "path":
        loader.runtime[str(source.parent / "elsewhere.py")] = loader.runtime.pop(str(source))
        expected = "RUNTIME_IMPORT_NOT_APPROVED"
    with pytest.raises(trial.TrialError, match=expected):
        loader.find_spec("trusted_textwrap", [str(source.parent)])
    assert loader.sources == {}


@pytest.mark.parametrize("kind", ["directory", "symlink", "hardlink"])
def test_external_import_record_does_not_authorize_special_path(runtime_source, kind):
    loader, source, record, original_source = runtime_source
    replacement = source.parent / "replacement.py"
    replacement.write_bytes(original_source)
    replacement.chmod(0o444)
    source.unlink()
    if kind == "directory":
        source.mkdir()
    elif kind == "symlink":
        source.symlink_to(replacement)
    else:
        source.hardlink_to(replacement)
    with pytest.raises(trial.TrialError, match="INPUT_NOT_PROTECTED"):
        loader._runtime_bytes(source)
    assert record["kind"] == "file"
    assert loader.sources == {}


def test_even_manifest_listed_sourceless_bytecode_is_not_executed(runtime_source):
    loader, source, _, _ = runtime_source
    source.unlink()
    cached = source.with_suffix(".pyc")
    raw = b"non-executable bytecode sentinel; never deserialize or execute"
    cached.write_bytes(raw)
    cached.chmod(0o444)
    loader.runtime = {
        str(cached): {
            "kind": "file",
            "mode": 0o444,
            "size": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
        }
    }
    with pytest.raises(trial.TrialError, match="RUNTIME_LOADER_NOT_APPROVED"):
        loader.find_spec("trusted_textwrap", [str(cached.parent)])
    assert loader.sources == {}


def test_native_extension_requires_authenticated_record_before_loader(runtime_source):
    loader, source, _, _ = runtime_source
    source.unlink()
    extension = source.with_suffix(importlib.machinery.EXTENSION_SUFFIXES[0])
    extension.write_bytes(b"non-executable native sentinel")
    extension.chmod(0o444)
    # Discovery is read-only; no create_module/exec_module is ever called.
    with pytest.raises(trial.TrialError, match="RUNTIME_IMPORT_NOT_APPROVED"):
        loader.find_spec("trusted_textwrap", [str(extension.parent)])
    assert loader.sources == {}


def test_exact_trusted_stdlib_source_bytes_execute_without_bytecode(runtime_source, monkeypatch):
    loader, source, record, original_source = runtime_source
    spec = loader.find_spec("trusted_textwrap", [str(source.parent)])
    assert spec is not None and spec.loader is loader
    assert loader.sources["trusted_textwrap"] == original_source
    assert hashlib.sha256(original_source).hexdigest() == record["sha256"]
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "trusted_textwrap", module)
    loader.exec_module(module)
    assert module.dedent("    authenticated\n    source\n") == "authenticated\nsource\n"
    assert module.__file__ == str(source)
    assert not list(source.parent.rglob("*.pyc"))


def test_namespace_package_cannot_become_runtime_dependency(runtime_source):
    loader, source, _, _ = runtime_source
    source.unlink()
    package = source.parent / "trusted_textwrap"
    package.mkdir()
    with pytest.raises(trial.TrialError, match="RUNTIME_IMPORT_NOT_APPROVED"):
        loader.find_spec("trusted_textwrap", [str(package.parent)])
    assert loader.sources == {}

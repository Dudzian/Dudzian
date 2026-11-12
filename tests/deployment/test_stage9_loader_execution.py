"""Dynamic loading regressions using reviewed sources and installed dependencies.

No fixture generates keys/signatures or executes substitute attack code. The
full graph test models root ownership and prior external-manifest authentication;
it is a development compatibility check, never a signed runtime or VM proof.
"""

from __future__ import annotations

import email
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import sysconfig

import pytest

from deployment import stage9_vm_trial as trial

pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="Linux source loader")
ROOT = Path(__file__).resolve().parents[2]
MISSING = "stage9_absent_optional_dependency_3265"


@pytest.mark.parametrize("bound", [False, True])
def test_absent_optional_dependency_retains_python_import_error_semantics(tmp_path, bound):
    loader = trial._HarnessLoader({}, ROOT)
    if bound:
        loader.runtime = {}
    with pytest.raises(ModuleNotFoundError) as caught:
        loader.find_spec(MISSING, [str(tmp_path)])
    assert caught.value.name == MISSING


@pytest.mark.parametrize("bound", [False, True])
def test_missing_dependency_does_not_fall_through_to_another_finder(tmp_path, monkeypatch, bound):
    loader = trial._HarnessLoader({}, ROOT)
    if bound:
        loader.runtime = {}
    callbacks = []

    class ForbiddenFinder:
        def find_spec(self, fullname, path=None, target=None):
            callbacks.append(fullname)
            raise AssertionError("Unapproved fallback finder executed")

    monkeypatch.setattr(sys, "path", [str(tmp_path)])
    monkeypatch.setattr(sys, "meta_path", [loader, ForbiddenFinder()])
    with pytest.raises(ModuleNotFoundError) as caught:
        importlib.import_module(MISSING)
    assert caught.value.name == MISSING
    assert callbacks == []


@pytest.mark.parametrize("bound", [False, True])
def test_existing_unapproved_dependency_is_hard_refusal_before_execution(tmp_path, bound):
    # Read-only discovery sees data at a module name, but must never load it.
    source = tmp_path / (MISSING + ".py")
    source.write_bytes(b"non-executable unauthorized source sentinel")
    loader = trial._HarnessLoader({}, ROOT)
    if bound:
        loader.runtime = {}
    with pytest.raises(trial.TrialError):
        loader.find_spec(MISSING, [str(tmp_path)])
    assert loader.sources == {}


def test_stdlib_name_cannot_authorize_source_from_a_foreign_directory(tmp_path, monkeypatch):
    source = tmp_path / "fractions.py"
    source.write_bytes(b"non-executable source with an approved stdlib name")
    loader = trial._HarnessLoader({}, ROOT)

    def forbidden_read(*args, **kwargs):
        raise AssertionError("Foreign stdlib shadow was opened as executable source")

    monkeypatch.setattr(trial, "_read", forbidden_read)
    with pytest.raises(trial.TrialError):
        loader.find_spec("fractions", [str(tmp_path)])
    assert loader.sources == {}


_GRAPH_PROBE = r"""
import hashlib
import importlib
import json
from pathlib import Path
import runpy
import stat
import sys

root = Path(sys.argv[1])
site = Path(sys.argv[2])
stdlib = Path(sys.argv[3])
namespace = runpy.run_path(str(root / "deployment/stage9_vm_trial.py"))
namespace["_import_state"]()
# Add only this executor's established dependency installation, without .pth
# processing. This is an explicit test fixture; production uses its verified path.
sys.path.append(str(site))
sources = {name.replace(".", "/") + ".py":
           (root / (name.replace(".", "/") + ".py")).read_bytes()
           for name in namespace["HARNESS_MODULES"]}
sources.update({name: (root / name).read_bytes() for name in namespace["HARNESS_HELPERS"]})
namespace["_check_graph"](sources)

def model_root_metadata(path, *, directory=False):
    allowed = path.is_relative_to(stdlib) or path.is_relative_to(site)
    info = path.lstat()
    namespace["_require"](
        allowed and not stat.S_ISLNK(info.st_mode)
        and (stat.S_ISDIR(info.st_mode) if directory else stat.S_ISREG(info.st_mode))
        and (directory or info.st_nlink == 1),
        "DEVELOPMENT_FIXTURE_OBJECT_REFUSED",
    )

# Real descriptor reads, hashes, object types and modes remain enabled. Only
# root ownership/ancestor metadata is modeled for unprivileged development data.
namespace["_read"].__globals__["_protected"] = model_root_metadata
loader = namespace["_HarnessLoader"](sources, root)
sys.meta_path.insert(0, loader)
# This precise first import previously failed: dataclasses -> copy tries absent
# org.python.core and expects ImportError, before runtime binding is available.
importlib.import_module("deployment.freshness_runtime")
assert loader.runtime is None

external = {}
def record(path):
    if path.is_symlink() or not path.is_file():
        return
    content = path.read_bytes()
    external[str(path)] = {"kind": "file", "mode": stat.S_IMODE(path.lstat().st_mode),
                           "size": len(content), "sha256": hashlib.sha256(content).hexdigest()}

# Snapshot trusted development objects before executing dependency source.
# Exclude unrelated installed distributions; no native runtime is built here.
for path in stdlib.rglob("*"):
    if not {"site-packages", "dist-packages", "__pycache__"} & set(path.relative_to(stdlib).parts):
        if path.suffix in {".py", ".so"}:
            record(path)
for package in ("cryptography", "psycopg", "psycopg_binary", "cffi", "pycparser"):
    package_root = site / package
    assert package_root.is_dir(), package
    for path in package_root.rglob("*"):
        if path.suffix in {".py", ".so"}:
            record(path)
for path in site.glob("_cffi_backend*.so"):
    record(path)
record(site / "typing_extensions.py")
loader.bind_runtime({"installation_path": "/uninstalled-development-model", "files": {},
                     "external_files": external})
for name in sorted(namespace["HARNESS_MODULES"]):
    module = importlib.import_module(name)
    assert module.__loader__ is loader, name
assert sys.modules["deployment"].__path__ == []
assert sys.modules["bot_core"].__path__ == []
assert sys.modules["psycopg.pq"].__impl__ == "binary"
for name, module in list(sys.modules.items()):
    if name.startswith(("cryptography.", "psycopg.")):
        origin = getattr(getattr(module, "__spec__", None), "origin", None)
        if origin and origin.endswith(".py"):
            assert module.__loader__ is loader, name
print(json.dumps({"executed_project_modules": sorted(namespace["HARNESS_MODULES"]),
                  "external_records": len(external), "psycopg_impl": "binary",
                  "modeled_boundaries": ["root_metadata", "prior_manifest_authentication"],
                  "status": "DEVELOPMENT_GRAPH_EXECUTED", "vm_qualification": "NOT_EXECUTED"}))
"""


def test_exact_bootstrap_order_and_full_dependency_graph_execute_verified_sources(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-S",
            "-B",
            "-c",
            _GRAPH_PROBE,
            str(ROOT),
            sysconfig.get_path("purelib"),
            sysconfig.get_path("stdlib"),
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=90,
        cwd=tmp_path,
        env={"PATH": "/usr/bin:/bin", "LC_ALL": "C"},
    )
    assert result.returncode == 0, result.stderr
    observed = json.loads(result.stdout)
    assert observed["executed_project_modules"] == sorted(trial.HARNESS_MODULES)
    assert observed["psycopg_impl"] == "binary"
    assert observed["status"] == "DEVELOPMENT_GRAPH_EXECUTED"
    assert observed["vm_qualification"] == "NOT_EXECUTED"
    if evidence := os.environ.get("STAGE9_SAFE_OUTPUT"):
        (Path(evidence) / "dynamic-loader-execution.json").write_text(
            json.dumps(
                {"argv": result.args, "exit_code": result.returncode, "observation": observed},
                indent=2,
            )
            + "\n"
        )
    assert not list(tmp_path.rglob("*.pyc"))
    assert not list(tmp_path.iterdir())


def test_authenticated_source_package_keeps_python_location_metadata(tmp_path, monkeypatch):
    source = tmp_path / "trusted_email" / "__init__.py"
    source.parent.mkdir()
    raw = Path(email.__file__).read_bytes()
    source.write_bytes(raw)
    source.chmod(0o444)
    loader = trial._HarnessLoader({}, ROOT)
    loader.runtime = {
        str(source): {
            "kind": "file",
            "mode": 0o444,
            "size": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
        }
    }
    real_protected = trial._protected

    def model_root_metadata(path, *, directory=False):
        if path == source:
            info = path.lstat()
            assert stat.S_ISREG(info.st_mode) and info.st_nlink == 1
            assert not path.is_symlink() and stat.S_IMODE(info.st_mode) == 0o444
        else:
            real_protected(path, directory=directory)

    monkeypatch.setattr(trial, "_protected", model_root_metadata)
    spec = loader.find_spec("trusted_email", [str(tmp_path)])
    assert spec is not None and spec.has_location
    assert spec.origin == str(source)
    assert spec.submodule_search_locations == [str(source.parent)]
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "trusted_email", module)
    loader.exec_module(module)
    assert module.__file__ == str(source)
    assert module.__package__ == "trusted_email"
    assert module.__path__ == [str(source.parent)]
    assert "message_from_bytes" in module.__all__
    assert not list(tmp_path.rglob("*.pyc"))


def test_bootstrap_path_does_not_import_environment_selected_sysconfig_data(tmp_path):
    script = r"""
import runpy, sys
from pathlib import Path
ns = runpy.run_path(sys.argv[1])
loader = ns['_HarnessLoader']({}, Path(sys.argv[1]).parent.parent)
assert loader.bootstrap_stdlib == (Path(sys.base_prefix) / sys.platlibdir /
    f'python{sys.version_info.major}.{sys.version_info.minor}')
assert 'stage9_no_such_sysconfig_data_3265' not in sys.modules
print('NO_CONFIG_SELECTED_IMPORT')
"""
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-S",
            "-B",
            "-c",
            script,
            str(ROOT / "deployment/stage9_vm_trial.py"),
        ],
        capture_output=True,
        text=True,
        check=False,
        cwd=tmp_path,
        env={
            "PATH": "/usr/bin:/bin",
            "_PYTHON_SYSCONFIGDATA_NAME": "stage9_no_such_sysconfig_data_3265",
        },
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "NO_CONFIG_SELECTED_IMPORT"

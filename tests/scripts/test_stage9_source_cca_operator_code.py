"""Local code consistency regressions, never root bootstrap authentication."""

from __future__ import annotations

import argparse
import builtins
import hashlib
import importlib
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest

import deployment
from scripts import qualify_stage9_source_cca_installation as operator
from scripts import qualify_stage41b_signer_isolation as stage41b

_CONTROLS = "deployment.stage9_source_cca_test_controls"
_REFUSAL = "^PROTECTED_REVIEWED_OPERATOR_REQUIRED$"
_RUN_ID = "012345abcdef"
_COMMIT = "a" * 40

pytestmark = pytest.mark.skipif(
    sys.platform == "win32", reason="POSIX filesystem ownership/mode code-protection boundary"
)


@pytest.fixture
def code_tree(monkeypatch, tmp_path):
    """Virtualize scaffolding/ownership; real code modes reach the checker."""
    tmp_path = tmp_path.resolve()
    scaffolding = {tmp_path, *tmp_path.parents}
    root = tmp_path / "checkout"
    package = root / "deployment"
    scripts = root / "scripts"
    package.mkdir(parents=True)
    scripts.mkdir()
    controls = package / "stage9_source_cca_test_controls.py"
    script = scripts / "qualify_stage9_source_cca_installation.py"
    files = {
        controls: b"# synthetic reviewed controls; never executed by inventory\n",
        script: b"# synthetic reviewed operator\n",
        package / "__init__.py": b"# package\n",
        scripts / "__init__.py": b"# package\n",
    }
    for path, raw in files.items():
        path.write_bytes(raw)
        path.chmod(0o644)
    for directory in (root, package, scripts):
        directory.chmod(0o755)
    (root / ".git").mkdir(mode=0o755)
    required = {root, script, controls, package / "__init__.py", scripts / "__init__.py"}
    search = set()
    foreign = set()
    checked = []
    original_locations = operator._operator_code_locations
    original_lstat = Path.lstat
    original_check = stage41b._check_protected

    def lstat(path):
        info = original_lstat(path)
        values = list(info)
        values[4] = 1000 if path in foreign else 0
        # pytest's /tmp or macOS cache ancestry is not the code under test.
        if path in scaffolding:
            values[0] &= ~0o022
        return os.stat_result(values)

    def check(path, *, directory):
        checked.append((path, directory))
        original_check(path, directory=directory)

    monkeypatch.setattr(Path, "lstat", lstat)
    monkeypatch.setattr(stage41b, "_check_protected", check)
    monkeypatch.setattr(operator, "__file__", str(script))
    monkeypatch.setattr(operator, "_operator_code_locations", lambda: (required, search))
    return SimpleNamespace(
        root=root,
        controls=controls,
        script=script,
        package=package,
        required=required,
        search=search,
        foreign=foreign,
        checked=checked,
        tmp=tmp_path,
        locations=original_locations,
    )


def test_declared_inventory_includes_controls_not_present_in_loaded_modules(code_tree, monkeypatch):
    interpreter = code_tree.tmp / "protected-python"
    interpreter.write_bytes(b"synthetic interpreter location")
    missing_search = code_tree.tmp / "missing-search" / "python312.zip"
    monkeypatch.setattr(
        operator,
        "sys",
        SimpleNamespace(executable=str(interpreter), path=[str(missing_search)], modules={}),
    )
    required, search = code_tree.locations()
    assert code_tree.controls in required
    assert code_tree.script in required
    assert code_tree.package / "__init__.py" in required
    assert code_tree.root / "scripts" / "__init__.py" in required
    assert interpreter in required
    assert missing_search in search


def test_loaded_module_outside_search_tree_is_also_required(code_tree, monkeypatch):
    interpreter = code_tree.tmp / "protected-python"
    interpreter.write_bytes(b"synthetic interpreter location")
    external = code_tree.tmp / "external-module.py"
    external.write_bytes(b"# inert loaded module location\n")
    monkeypatch.setattr(
        operator,
        "sys",
        SimpleNamespace(
            executable=str(interpreter),
            path=[],
            modules={"synthetic_external": SimpleNamespace(__file__=str(external))},
        ),
    )
    required, _search = code_tree.locations()
    assert external in required


def test_namespace_package_search_locations_are_inventoried(code_tree, monkeypatch):
    interpreter = code_tree.tmp / "protected-python"
    interpreter.write_bytes(b"synthetic interpreter location")
    external = code_tree.tmp / "late-namespace"
    monkeypatch.setattr(
        operator,
        "sys",
        SimpleNamespace(
            executable=str(interpreter),
            path=[],
            modules={"synthetic_namespace": SimpleNamespace(__path__=[str(external)])},
        ),
    )
    _required, search = code_tree.locations()
    assert external in search


def test_inventory_covers_unloaded_module_without_executing_source(code_tree, monkeypatch):
    marker = code_tree.tmp / "must-not-exist"
    code_tree.controls.write_text(
        f"from pathlib import Path\nPath({str(marker)!r}).write_text('imported')\n",
        encoding="utf-8",
    )
    monkeypatch.delitem(sys.modules, _CONTROLS, raising=False)
    operator._trusted_operator()
    assert (code_tree.controls, False) in code_tree.checked
    assert (code_tree.package / "__init__.py", False) in code_tree.checked
    assert not marker.exists()
    assert _CONTROLS not in sys.modules


@pytest.mark.parametrize("name", ["later_module.py", "cached_module.pyc", "startup.pth"])
def test_unloaded_executable_files_in_import_tree_require_protection(code_tree, name):
    late = code_tree.package / name
    late.write_bytes(b"# inert synthetic code artifact\n")
    late.chmod(0o666)
    with pytest.raises(operator.HarnessRefusal, match=_REFUSAL):
        operator._trusted_operator()
    assert (late, False) in code_tree.checked


def test_foreign_owned_unloaded_controls_is_refused(code_tree):
    code_tree.foreign.add(code_tree.controls)
    with pytest.raises(operator.HarnessRefusal, match=_REFUSAL):
        operator._trusted_operator()


def test_permission_change_without_content_change_is_rechecked(code_tree):
    before = hashlib.sha256(code_tree.controls.read_bytes()).hexdigest()
    operator._trusted_operator()
    code_tree.controls.chmod(0o664)
    assert hashlib.sha256(code_tree.controls.read_bytes()).hexdigest() == before
    with pytest.raises(operator.HarnessRefusal, match=_REFUSAL):
        operator._trusted_operator()


@pytest.mark.parametrize("missing", ["controls", "package-init"])
def test_missing_required_source_is_not_treated_as_safe_missing_search_root(code_tree, missing):
    path = code_tree.controls if missing == "controls" else code_tree.package / "__init__.py"
    path.unlink()
    with pytest.raises(operator.HarnessRefusal, match=_REFUSAL):
        operator._trusted_operator()


def test_missing_import_search_root_requires_protected_existing_ancestry(code_tree):
    parent = code_tree.tmp / "external-search"
    parent.mkdir(mode=0o755)
    missing = parent / "not-created" / "python312.zip"
    code_tree.search.add(missing)
    operator._trusted_operator()
    assert (parent, True) in code_tree.checked
    assert not missing.exists()
    parent.chmod(0o777)
    with pytest.raises(operator.HarnessRefusal, match=_REFUSAL):
        operator._trusted_operator()
    assert not missing.exists()


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX import-tree symlink boundary")
@pytest.mark.parametrize("directory", [False, True])
def test_symlink_in_import_tree_is_refused(code_tree, directory):
    target = code_tree.tmp / "outside"
    if directory:
        target.mkdir()
    else:
        target.write_bytes(b"# inert\n")
    (code_tree.package / "alias").symlink_to(target, target_is_directory=directory)
    with pytest.raises(operator.HarnessRefusal, match=_REFUSAL):
        operator._trusted_operator()


def _deny_controls_import(monkeypatch):
    original_import = builtins.__import__
    original_import_module = importlib.import_module

    def importer(name, globals=None, locals=None, fromlist=(), level=0):
        if name == _CONTROLS or (
            name == "deployment" and "stage9_source_cca_test_controls" in fromlist
        ):
            pytest.fail("unprotected controls reached Python's import boundary")
        return original_import(name, globals, locals, fromlist, level)

    def import_module(name, package=None):
        if name == _CONTROLS:
            pytest.fail("unprotected controls reached Python's import boundary")
        return original_import_module(name, package)

    monkeypatch.setattr(builtins, "__import__", importer)
    monkeypatch.setattr(importlib, "import_module", import_module)


def test_controls_protection_is_checked_before_import(code_tree, monkeypatch):
    code_tree.controls.chmod(0o666)
    _deny_controls_import(monkeypatch)
    with pytest.raises(operator.HarnessRefusal, match=_REFUSAL):
        operator._prepare_privileged_controls()


def test_prepare_checks_before_import_and_rechecks_imported_code(code_tree, monkeypatch):
    events = []
    module = ModuleType(_CONTROLS)
    module.__file__ = str(code_tree.controls)
    expected = lambda _run: []
    module.run_controls = expected
    monkeypatch.setitem(sys.modules, _CONTROLS, module)
    monkeypatch.setattr(deployment, "stage9_source_cca_test_controls", module, raising=False)
    original_import = builtins.__import__
    original_import_module = importlib.import_module

    def importer(name, globals=None, locals=None, fromlist=(), level=0):
        if name == _CONTROLS or (
            name == "deployment" and "stage9_source_cca_test_controls" in fromlist
        ):
            events.append("import")
        return original_import(name, globals, locals, fromlist, level)

    def import_module(name, package=None):
        if name == _CONTROLS:
            events.append("import")
        return original_import_module(name, package)

    monkeypatch.setattr(operator, "_trusted_operator", lambda: events.append("check"))
    monkeypatch.setattr(builtins, "__import__", importer)
    monkeypatch.setattr(importlib, "import_module", import_module)
    assert operator._prepare_privileged_controls() is expected
    assert events[0] == events[-1] == "check"
    assert "import" in events
    assert events.index("check") < events.index("import") < len(events) - 1


def test_controls_from_unexpected_origin_are_refused(code_tree, monkeypatch):
    module = ModuleType(_CONTROLS)
    module.__file__ = str(code_tree.tmp / "outside-controls.py")
    module.run_controls = lambda _run: pytest.fail("foreign-origin controls must not be called")
    monkeypatch.setitem(sys.modules, _CONTROLS, module)
    monkeypatch.setattr(deployment, "stage9_source_cca_test_controls", module, raising=False)
    with pytest.raises(operator.HarnessRefusal, match=_REFUSAL):
        operator._prepare_privileged_controls()


def _args():
    return argparse.Namespace(
        run_id=_RUN_ID,
        authorize_test_only=_RUN_ID,
        physical_host="Dell-Latitude-5490",
        package_sha256="b" * 64,
        package=Path("synthetic-package"),
        uid_base=62000,
    )


def test_unprotected_late_module_refuses_execute_before_host_mutation(code_tree, monkeypatch):
    code_tree.controls.chmod(0o666)
    monkeypatch.setattr(operator, "sys", SimpleNamespace(platform="linux"))
    monkeypatch.setattr(operator, "_authorize", lambda _args: None)
    _deny_controls_import(monkeypatch)

    def forbidden(*_args, **_kwargs):
        pytest.fail("unprotected code reached package or host operations")

    monkeypatch.setattr(operator, "validate_package", forbidden)
    monkeypatch.setattr(operator, "preflight", forbidden)
    monkeypatch.setattr(operator, "PhysicalRun", forbidden)
    report = {"commit": _COMMIT}
    with pytest.raises(operator.HarnessRefusal, match=_REFUSAL):
        operator._execute(_args(), report)
    assert report == {"commit": _COMMIT}


def test_permission_change_during_preflight_refuses_before_resource_creation(
    code_tree, monkeypatch
):
    monkeypatch.setattr(operator, "sys", SimpleNamespace(platform="linux"))
    monkeypatch.setattr(operator, "_authorize", lambda _args: None)
    monkeypatch.setattr(operator, "_prepare_privileged_controls", lambda: lambda _run: [])
    resources = operator.plan(_RUN_ID)
    monkeypatch.setattr(
        operator,
        "validate_package",
        lambda *_args: {"plan": resources, "commit": _COMMIT, "package_sha256": "b" * 64},
    )

    def preflight(_resources, *, administrative):
        assert administrative is True
        code_tree.controls.chmod(0o666)
        return {"ready": True}

    monkeypatch.setattr(operator, "preflight", preflight)
    monkeypatch.setattr(
        operator,
        "signal",
        SimpleNamespace(SIGINT=2, SIGTERM=15, SIGHUP=1, getsignal=lambda _number: None),
    )
    monkeypatch.setattr(
        operator,
        "PhysicalRun",
        lambda *_args: pytest.fail("chmod-only loss of protection reached host mutation"),
    )
    with pytest.raises(operator.HarnessRefusal, match=_REFUSAL):
        operator._execute(_args(), {"commit": _COMMIT})

"""Authenticate data before execution; attack fixtures are never imported."""

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

from deployment import stage9_vm_trial as trial

pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="Linux harness boundary")
REVISION = "a" * 40
REPOSITORY = Path(__file__).resolve().parents[2]


@pytest.fixture
def authenticated_tree(tmp_path, monkeypatch):
    root = tmp_path / "tree"
    required = {name.replace(".", "/") + ".py" for name in trial.HARNESS_MODULES} | set(
        trial.HARNESS_HELPERS
    )
    objects = []
    for name in sorted(required):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        raw = (REPOSITORY / name).read_bytes()
        path.write_bytes(raw)
        path.chmod(0o444)
        objects.append(
            {"path": name, "kind": "file", "mode": 0o444, "sha256": hashlib.sha256(raw).hexdigest()}
        )
    for path in [root, *sorted(p for p in root.rglob("*") if p.is_dir())]:
        path.chmod(0o555)
        objects.append({"path": str(path.relative_to(root)), "kind": "directory", "mode": 0o555})
    inventory = tmp_path / "inventory.json"
    document = {
        "schema": 1,
        "tree_root": str(root),
        "source_revision": REVISION,
        "objects": sorted(objects, key=lambda record: record["path"]),
    }
    inventory.write_text(json.dumps(document))
    monkeypatch.setattr(trial, "__file__", str(root / "deployment/stage9_vm_trial.py"))
    monkeypatch.setattr(trial, "HARNESS_INVENTORY", inventory)
    original = trial._protected

    def modeled_owner(path, *, directory=False):
        # Model only root ownership/ancestor protection in this unprivileged
        # data fixture. Exact modes/types/inventory/hashes remain real checks.
        if path.is_relative_to(tmp_path):
            info = path.lstat()
            assert not path.is_symlink()
            assert path.is_dir() if directory else path.is_file()
            assert directory or info.st_nlink == 1
            return
        original(path, directory=directory)

    monkeypatch.setattr(trial, "_protected", modeled_owner)
    yield root, inventory, document
    for path in root.rglob("*"):
        if path.is_dir():
            path.chmod(0o755)
    root.chmod(0o755)


def add(root, name, raw=None):
    path = root / name
    path.parent.chmod(0o755)
    if raw is None:
        path.mkdir()
    else:
        path.write_bytes(raw)
    path.parent.chmod(0o555)
    return path


def test_exact_graph_and_sources_are_verified_before_compile(authenticated_tree):
    sources = trial._harness_sources(REVISION)
    assert (
        sources["deployment/stage9_vm_native.py"]
        == (REPOSITORY / "deployment/stage9_vm_native.py").read_bytes()
    )
    module = ModuleType("deployment.stage9_vm_native")
    loader = trial._HarnessLoader(sources, authenticated_tree[0])
    # Only authenticated repository source executes; no attack fixture does.
    loader.exec_module(module)
    assert module.UNIT == "cryptohunter-freshness-verifier.service"


@pytest.mark.parametrize(
    "name,raw",
    [
        ("deployment/stage9_vm_native", None),
        ("deployment/stage9_vm_native.so", b"non-executable native sentinel"),
        ("deployment/stage9_vm_native.cpython-312-x86_64-linux-gnu.so", b"ELF sentinel"),
        ("deployment/unapproved.py", b"must never be executed"),
        ("deployment/__pycache__", None),
        ("bot_core/empty-extra-directory", None),
    ],
)
def test_additional_package_extension_dependency_cache_or_directory_refused(
    authenticated_tree, name, raw, monkeypatch
):
    root, _, _ = authenticated_tree
    extra = add(root, name, raw)
    if extra.is_dir() and extra.name == "stage9_vm_native":
        (extra / "__init__.py").write_bytes(b"must never be executed")
    monkeypatch.setattr(trial, "_import_state", lambda: pytest.fail("execution after extra object"))
    with pytest.raises(trial.TrialError, match="HARNESS_EXTRA_OBJECT"):
        trial._native_module()


def test_even_manifest_listed_native_object_is_not_approved(authenticated_tree):
    root, inventory, document = authenticated_tree
    add(root, "deployment/stage9_vm_native.so", b"non-executable sentinel")
    document["objects"].append(
        {
            "path": "deployment/stage9_vm_native.so",
            "kind": "file",
            "mode": 0o444,
            "sha256": "a" * 64,
        }
    )
    inventory.write_text(json.dumps(document))
    with pytest.raises(trial.TrialError, match="HARNESS_OBJECT_NOT_APPROVED"):
        trial._harness_sources()


@pytest.mark.parametrize("attack", ["content", "mode", "missing", "revision", "tree", "duplicate"])
def test_integrity_and_binding_refusals(authenticated_tree, attack):
    root, inventory, document = authenticated_tree
    native = root / "deployment/stage9_vm_native.py"
    expected = "HARNESS_INTEGRITY_CHANGED"
    if attack == "content":
        native.chmod(0o644)
        native.write_bytes(b"unapproved bytes are never compiled")
        native.chmod(0o444)
    elif attack == "mode":
        native.chmod(0o644)
        expected = "HARNESS_MODE_CHANGED"
    elif attack == "missing":
        native.parent.chmod(0o755)
        native.unlink()
        native.parent.chmod(0o555)
        expected = "HARNESS_INVENTORY_INCOMPLETE"
    elif attack == "revision":
        document["source_revision"] = "b" * 40
        expected = "HARNESS_INVENTORY_INVALID"
    elif attack == "tree":
        document["tree_root"] += "/other"
        expected = "HARNESS_INVENTORY_INVALID"
    elif attack == "duplicate":
        document["objects"].append(document["objects"][0])
        expected = "HARNESS_RECORD_DUPLICATE"
    inventory.write_text(json.dumps(document))
    with pytest.raises(trial.TrialError, match=expected):
        trial._harness_sources(REVISION)


@pytest.mark.parametrize(
    "statement",
    [
        b"\nfrom bot_core import unapproved_dependency\n",
        b"\ndef delayed():\n    import deployment.unapproved_dependency\n",
        b"\nimport unapproved_external_dependency\n",
        b"\n__import__('deployment.stage9_vm_native')\n",
    ],
)
def test_complete_graph_rejects_unapproved_imports_as_data(authenticated_tree, statement):
    sources = trial._harness_sources()
    sources["deployment/stage9_vm_native.py"] += statement
    with pytest.raises(trial.TrialError, match="HARNESS_(IMPORT|DEPENDENCY|DYNAMIC_IMPORT)"):
        trial._check_graph(sources)


@pytest.mark.parametrize("name", ["deployment", "deployment.stage9_vm_native", "bot_core.poison"])
def test_preloaded_namespace_or_descendant_is_refused(monkeypatch, name):
    monkeypatch.setitem(sys.modules, name, ModuleType(name))
    # Isolated flag check is independent; here model it to reach cache test.
    monkeypatch.setattr(sys, "flags", type("Flags", (), {"isolated": 1, "no_user_site": 1})())
    monkeypatch.setattr(sys, "dont_write_bytecode", True)
    with pytest.raises(trial.TrialError, match="PACKAGE_PRELOADED"):
        trial._import_state()


def test_unapproved_project_import_never_falls_back_to_resolver():
    loader = trial._HarnessLoader({}, REPOSITORY)
    with pytest.raises(trial.TrialError, match="HARNESS_IMPORT_NOT_APPROVED"):
        loader.find_spec("deployment.unapproved")


def test_authentication_precedes_any_environment_helper(monkeypatch, capsys):
    def refused(revision=None):
        raise trial.TrialError("HARNESS_INTEGRITY_CHANGED")

    monkeypatch.setattr(trial, "_harness_sources", refused)
    monkeypatch.setattr(
        trial, "_environment_helpers", lambda execute: pytest.fail("unverified helper")
    )
    assert trial.main(["--mode", "INSPECT"]) == 2
    assert json.loads(capsys.readouterr().out)["reason"] == "HARNESS_INTEGRITY_CHANGED"


def test_real_isolated_import_uses_authenticated_source_and_no_bytecode(tmp_path):
    script = """
import runpy, sys
from pathlib import Path
namespace = runpy.run_path(sys.argv[1])
namespace['_import_state']()
root = Path(sys.argv[1]).parent.parent
sources = {name.replace('.', '/') + '.py': (root / (name.replace('.', '/') + '.py')).read_bytes()
           for name in namespace['HARNESS_MODULES']}
sources.update({name: (root / name).read_bytes() for name in namespace['HARNESS_HELPERS']})
namespace['_check_graph'](sources)
loader = namespace['_HarnessLoader'](sources, root)
# Model only host stdlib ownership in this development interpreter. Source
# bytes, file types, modes, links and descriptor-stability checks remain real.
g = namespace['_HarnessLoader'].find_spec.__globals__
original_protected = g['_protected']
def protected_development_stdlib(path, *, directory=False):
    if path.is_relative_to(loader.bootstrap_stdlib):
        info = path.lstat()
        assert not path.is_symlink() and not info.st_mode & 0o022
        assert path.is_dir() if directory else path.is_file() and info.st_nlink == 1
        return
    original_protected(path, directory=directory)
g['_protected'] = protected_development_stdlib
sys.meta_path.insert(0, loader)
import deployment.stage9_vm_native as native
assert native.__loader__ is loader
assert native.UNIT == 'cryptohunter-freshness-verifier.service'
assert sys.modules['deployment'].__path__ == []
assert not any(name.endswith('.pyc') for name in sources)
print('AUTHENTICATED_SOURCE_EXECUTED')
"""
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-S",
            "-B",
            "-c",
            script,
            str(REPOSITORY / "deployment/stage9_vm_trial.py"),
        ],
        capture_output=True,
        text=True,
        check=False,
        env={"PATH": "/usr/bin:/bin"},
        cwd=tmp_path,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "AUTHENTICATED_SOURCE_EXECUTED"


@pytest.mark.parametrize(
    "attack",
    [
        "cryptography",
        "cryptography.hazmat.unapproved",
        "psycopg",
        "unknown",
        "meta",
        "hook",
        "cache",
    ],
)
def test_isolated_worker_refuses_external_cache_or_import_hooks_without_callbacks(tmp_path, attack):
    script = """
import runpy, sys
from types import ModuleType
namespace = runpy.run_path(sys.argv[1])
attack = sys.argv[2]
class Inert:
    def __eq__(self, other):
        raise AssertionError('unapproved equality executed')
    def __call__(self, *args):
        raise AssertionError('unapproved hook executed')
    def invalidate_caches(self):
        raise AssertionError('unapproved cache callback executed')
if attack == 'meta':
    sys.meta_path[0] = Inert()
elif attack == 'hook':
    sys.path_hooks.append(Inert())
elif attack == 'cache':
    sys.path_importer_cache['/unapproved'] = Inert()
else:
    sys.modules[attack] = ModuleType(attack)
try:
    namespace['_import_state']()
except namespace['TrialError'] as error:
    print(str(error))
else:
    raise AssertionError('poisoned import state accepted')
"""
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-S",
            "-B",
            "-c",
            script,
            str(REPOSITORY / "deployment/stage9_vm_trial.py"),
            attack,
        ],
        capture_output=True,
        text=True,
        check=False,
        cwd=tmp_path,
        env={"PATH": "/usr/bin:/bin"},
    )
    assert result.returncode == 0, result.stderr
    expected = (
        "IMPORT_HOOK_NOT_APPROVED"
        if attack in {"meta", "hook"}
        else ("IMPORT_CACHE_NOT_APPROVED" if attack == "cache" else "EXTERNAL_MODULE_PRELOADED")
    )
    assert result.stdout.strip() == expected


@pytest.mark.parametrize(
    "attack",
    [
        "owner",
        "writable",
        "symlink",
        "hardlink",
        "directory",
        "ancestor-owner",
        "ancestor-writable",
    ],
)
def test_protection_rejects_bad_metadata_before_read(tmp_path, monkeypatch, attack):
    import stat

    path = tmp_path / "source.py"
    path.write_bytes(b"data only")
    original = Path.lstat

    def metadata(candidate):
        fields = list(original(candidate))
        fields[stat.ST_UID] = 0
        fields[stat.ST_MODE] &= ~0o022
        if candidate == path:
            if attack == "owner":
                fields[stat.ST_UID] = 1001
            elif attack == "writable":
                fields[stat.ST_MODE] |= 0o020
            elif attack == "symlink":
                fields[stat.ST_MODE] = stat.S_IFLNK | 0o444
            elif attack == "hardlink":
                fields[stat.ST_NLINK] = 2
            elif attack == "directory":
                fields[stat.ST_MODE] = stat.S_IFDIR | 0o555
        elif candidate == tmp_path:
            if attack == "ancestor-owner":
                fields[stat.ST_UID] = 1001
            elif attack == "ancestor-writable":
                fields[stat.ST_MODE] |= 0o002
        return os.stat_result(fields)

    monkeypatch.setattr(Path, "lstat", metadata)
    monkeypatch.setattr(os, "open", lambda *args: pytest.fail("read before metadata approval"))
    with pytest.raises(trial.TrialError, match="INPUT_NOT_PROTECTED"):
        trial._read(path, 1024)


@pytest.mark.parametrize("helper", sorted(trial.HARNESS_HELPERS))
def test_helper_import_graph_is_also_checked_before_execution(authenticated_tree, helper):
    sources = trial._harness_sources()
    sources[helper] += b"\nimport unapproved_helper_dependency\n"
    with pytest.raises(trial.TrialError, match="HARNESS_DEPENDENCY_NOT_APPROVED"):
        trial._check_graph(sources)


def test_descriptor_swap_is_refused_before_reading_source(tmp_path, monkeypatch):
    import stat

    source = tmp_path / "source.py"
    source.write_bytes(b"non-executable data")
    monkeypatch.setattr(trial, "_protected", lambda *args, **kwargs: None)
    original_fstat = os.fstat

    def replaced_descriptor(fd):
        values = list(original_fstat(fd))
        values[stat.ST_INO] += 1
        return os.stat_result(values)

    monkeypatch.setattr(os, "fstat", replaced_descriptor)
    monkeypatch.setattr(
        os, "fdopen", lambda *args, **kwargs: pytest.fail("read before identity check")
    )
    with pytest.raises(trial.TrialError, match="INPUT_CHANGED_DURING_READ"):
        trial._read(source, 1024)


@pytest.mark.parametrize("container", ["meta_path", "path_hooks", "path_importer_cache", "path"])
def test_import_state_refuses_foreign_containers_without_callbacks(tmp_path, container):
    script = r"""
import runpy, sys
namespace = runpy.run_path(sys.argv[1])
class ForbiddenList(list):
    def __iter__(self):
        raise AssertionError('foreign container iterated')
    def __len__(self):
        raise AssertionError('foreign container length evaluated')
class ForbiddenDict(dict):
    def values(self):
        raise AssertionError('foreign cache values called')
key = sys.argv[2]
value = ForbiddenDict() if key == 'path_importer_cache' else ForbiddenList()
setattr(sys, key, value)
try:
    namespace['_import_state']()
except namespace['TrialError'] as error:
    print(str(error))
else:
    raise AssertionError('foreign container accepted')
"""
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-S",
            "-B",
            "-c",
            script,
            str(REPOSITORY / "deployment/stage9_vm_trial.py"),
            container,
        ],
        capture_output=True,
        text=True,
        cwd=tmp_path,
        env={"PATH": "/usr/bin:/bin"},
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "IMPORT_CACHE_NOT_APPROVED"

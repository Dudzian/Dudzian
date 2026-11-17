"""Statyczne domknięcie bez importowania badanego kodu ani uruchamiania VM."""

from __future__ import annotations

import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from scripts import stage9_v5d4_closure as closure


def repository(tmp_path: Path, source: str = "") -> Path:
    (tmp_path / "app").mkdir()
    (tmp_path / "app/__init__.py").write_text("")
    (tmp_path / "app/main.py").write_text(source)
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nrequires-python=">=3.11"\ndependencies=["cryptography>=50.0.1,<51"]\n'
        '[project.optional-dependencies]\nobserver-kvm-preflight=["defusedxml==0.7.1"]\n'
    )
    return tmp_path


def analyze(tmp_path: Path, source: str = "") -> dict:
    return closure.inventory(repository(tmp_path, source), ("app/main.py",))


def test_initializers_transitive_and_deferred_imports(tmp_path: Path) -> None:
    root = repository(tmp_path, "from . import child\ndef f():\n import optional_external\n")
    (root / "app/__init__.py").write_text("from . import initialized\n")
    (root / "app/child.py").write_text("import external_dependency\n")
    (root / "app/initialized.py").write_text("raise RuntimeError('NIE IMPORTOWAĆ')\n")
    result = closure.inventory(root, ("app/main.py",))
    assert result["static_graph_closed"]
    assert {item["path"] for item in result["sources"]} == {
        "app/main.py",
        "app/__init__.py",
        "app/child.py",
        "app/initialized.py",
    }
    assert result["preflight_external_static_roots"] == ["external_dependency", "optional_external"]
    assert any(item["phase"] == "deferred" for item in result["static_edges"])
    assert not result["installation_contract"]["reduced_preflight_installation_authorized"]


def test_missing_local_module_refuses(tmp_path: Path) -> None:
    result = analyze(tmp_path, "import app.absent\n")
    assert result["missing_local_modules"] == ["app.absent"]
    assert not result["static_graph_closed"]
    assert not result["installation_approved"]


@pytest.mark.parametrize(
    "source",
    [
        "collector.execute(request)\n",
        "import re\nre.compile('a')\n",
        "object.execute_safely()\n",
    ],
)
def test_false_dynamic_imports(source: str) -> None:
    visitor = closure.SourceVisitor("app/main.py")
    visitor.visit(ast.parse(source))
    assert visitor.dynamic == []


@pytest.mark.parametrize(
    "source",
    [
        "import re as regex\nregex.compile('pattern')\n",
        "from re import compile\ncompile('pattern')\n",
        "from re import compile as c\nc('pattern')\n",
        "import re\nc = re.compile\nother = c\nother('pattern')\n",
        "import re as regex\nc = regex.compile\nc('pattern')\n",
        "import re\ncompile = re.compile\ncompile('pattern')\n",
        "import re\nc: object = re.compile\nc('pattern')\n",
        "import re\ngetattr(re, 'compile')('pattern')\n",
        "import re\nfrom builtins import getattr as get\nget(re, 'compile')('pattern')\n",
        "import re\ndef f():\n import re as regex\n regex.compile('pattern')\nre.compile('pattern')\n",
        "import re\nclass Regex:\n from re import compile as c\n c('pattern')\nre.compile('pattern')\n",
        "import re\n[re.compile('pattern') for item in things]\n",
    ],
)
def test_proven_re_compile_aliases_are_not_dynamic(tmp_path: Path, source: str) -> None:
    result = analyze(tmp_path, source)
    assert result["dynamic_mechanisms"] == []
    assert result["static_graph_closed"]


@pytest.mark.parametrize(
    "source",
    [
        "compile(source, 'module', 'exec')\n",
        "import builtins\nbuiltins.compile(source, 'module', 'exec')\n",
        "import builtins as b\nb.compile(source, 'module', 'exec')\n",
        "from builtins import compile as c\nc(source, 'module', 'exec')\n",
        "c = compile\nother = c\nother(source, 'module', 'exec')\n",
        "import builtins as b\nc = b.compile\nc(source, 'module', 'exec')\n",
        "__builtins__['compile'](source, 'module', 'exec')\n",
        "import builtins\ngetattr(builtins, 'compile')(source, 'module', 'exec')\n",
        "import builtins as b\ngetattr(b, mechanism)(source)\n",
        "__builtins__[mechanism](source)\n",
        "import builtins as b\nb.__dict__[mechanism](source)\n",
        "unknown.compile(source, 'module', 'exec')\n",
        "re.compile(source, 'module', 'exec')\n",
        "import re\nre = unknown\nre.compile(source, 'module', 'exec')\n",
        "import re\nre = factory()\nre.compile(source, 'module', 'exec')\n",
        "import re\nc = re.compile\nc = unknown\nc(source, 'module', 'exec')\n",
        "from re import compile\ncompile = unknown\ncompile(source, 'module', 'exec')\n",
        "import re\nre.compile = exec\nre.compile(source)\n",
        "from re import compile\ndef f(compile):\n compile(source, 'module', 'exec')\n",
        "import re\nf = lambda re: re.compile(source, 'module', 'exec')\n",
        "from re import compile\nfor compile in compilers:\n compile(source, 'module', 'exec')\n",
        "from re import compile\nwith context as compile:\n compile(source, 'module', 'exec')\n",
        "from re import compile\n[compile(source, 'module', 'exec') for compile in compilers]\n",
        "from re import compile as c\ntry:\n pass\nexcept Exception as c:\n c(source)\n",
        "def f():\n from re import compile\ncompile(source, 'module', 'exec')\n",
        "class Regex:\n from re import compile\ncompile(source, 'module', 'exec')\n",
        "class Outer:\n from re import compile\n class Inner:\n  compile(source, 'module', 'exec')\n",
        "class Outer:\n from re import compile\n class Inner:\n  def f():\n   compile(source, 'module', 'exec')\n",
        "from re import compile as c\ndef f():\n c(source, 'module', 'exec')\nfrom builtins import compile as c\nf()\n",
        "def outer():\n from re import compile as c\n def inner():\n  c(source, 'module', 'exec')\n from builtins import compile as c\n inner()\n",
        "if condition:\n from re import compile\ncompile(source, 'module', 'exec')\n",
        "while condition:\n from re import compile\ncompile(source, 'module', 'exec')\n",
        "if condition:\n compile = exec\nelse:\n from re import compile\ncompile(source)\n",
        "try:\n raise Exception()\n from re import compile\nexcept Exception:\n pass\ncompile(source, 'module', 'exec')\n",
        "import re\ncondition and (compile := re.compile)\ncompile(source, 'module', 'exec')\n",
        "import re\ncompile = re.compile if condition else unknown\ncompile(source, 'module', 'exec')\n",
    ],
)
def test_compile_and_unknown_rebindings_require_review(tmp_path: Path, source: str) -> None:
    result = analyze(tmp_path, source)
    assert result["unresolved_dynamic_mechanisms"]
    assert not result["static_graph_closed"]


def test_function_import_aliases_do_not_leak_into_module_scope() -> None:
    visitor = closure.SourceVisitor("app/main.py")
    visitor.visit(
        ast.parse(
            "import re\ndef f():\n import builtins as re\n re.compile(source, 'module', 'exec')\n"
            "re.compile('pattern')\n"
        )
    )
    assert [item["mechanism"] for item in visitor.dynamic] == ["builtins.compile"]
    assert visitor.dynamic[0]["phase"] == "deferred"


@pytest.mark.parametrize(
    "source",
    [
        "def f(*args: compile(source, 'module', 'exec')): pass\n",
        "import builtins\ndef f(**kwargs: builtins.compile(source, 'module', 'exec')): pass\n",
    ],
)
def test_variadic_argument_annotations_require_dynamic_review(tmp_path: Path, source: str) -> None:
    result = analyze(tmp_path, source)
    assert result["unresolved_dynamic_mechanisms"]
    assert not result["static_graph_closed"]


@pytest.mark.skipif(sys.version_info < (3, 12), reason="Wymagana składnia type_params CPython 3.12")
@pytest.mark.parametrize(
    "source",
    [
        "def f[T: compile(source, 'module', 'exec')](): pass\n",
        "import builtins\ndef f[T: builtins.compile(source, 'module', 'exec')](): pass\n",
        "import builtins as b\nclass C[T: b.compile(source, 'module', 'exec')]: pass\n",
        "def f[T: exec(source)](): pass\n",
        "class C[T: eval(source)]: pass\n",
        "import importlib\ndef f[T: importlib.import_module(module)](): pass\n",
        "import builtins\nclass C[T: getattr(builtins, mechanism)(source)]: pass\n",
        "import re\ndef f[re: re.compile(source, 'module', 'exec')](): pass\n",
        "from re import compile\ndef f[compile: compile(source, 'module', 'exec')](): pass\n",
        "import re\nclass C[re]:\n def f():\n  re.compile(source, 'module', 'exec')\n",
        "import re\ndef f[re](value: re.compile(source, 'module', 'exec')): pass\n",
    ],
)
def test_cpython312_type_parameter_bounds_require_dynamic_review(
    tmp_path: Path, source: str
) -> None:
    result = analyze(tmp_path, source)
    assert result["unresolved_dynamic_mechanisms"]
    assert not result["static_graph_closed"]


@pytest.mark.skipif(sys.version_info < (3, 12), reason="Wymagana składnia type_params CPython 3.12")
@pytest.mark.parametrize(
    "source",
    [
        "import re\ndef f[T: re.compile('pattern')](): pass\nre.compile('pattern')\n",
        "import re as regex\nclass C[T: regex.compile('pattern')]: pass\n",
        "from re import compile as c\ndef f[T: c('pattern')](): pass\n",
        "import re\ndef f[re](): pass\nre.compile('pattern')\n",
    ],
)
def test_cpython312_type_parameters_preserve_proven_regex_and_outer_scope(
    tmp_path: Path, source: str
) -> None:
    result = analyze(tmp_path, source)
    assert result["dynamic_mechanisms"] == []
    assert result["static_graph_closed"]


@pytest.mark.parametrize(
    "source",
    [
        "exec(value)\n",
        "eval(value)\n",
        "compile(value, 'x', 'exec')\n",
        "import builtins as b\nb.exec(value)\n",
        "from builtins import eval as run\nrun(value)\n",
        "from importlib import import_module as load\nload(module_name)\n",
        "import importlib as il\nf = il.import_module\nf(module_name)\n",
        "import importlib\ngetattr(importlib, mechanism)(module_name)\n",
        "__builtins__['exec'](value)\n",
        "import runpy\nrunpy.run_path(path)\n",
        "loader.exec_module(module)\n",
        "import ctypes\nctypes.CDLL(path)\n",
    ],
)
def test_unresolved_dynamic_mechanisms_refuse(tmp_path: Path, source: str) -> None:
    result = analyze(tmp_path, source)
    assert result["unresolved_dynamic_mechanisms"]
    assert not result["static_graph_closed"]


def test_literal_dynamic_target_is_recorded_not_approved(tmp_path: Path) -> None:
    result = analyze(tmp_path, "import importlib\nimportlib.import_module('libvirt')\n")
    assert result["preflight_dynamic_targets"] == ["libvirt"]
    assert result["static_graph_closed"]
    assert not result["installation_approved"]


def test_initializer_dynamic_code_not_omitted(tmp_path: Path) -> None:
    root = repository(tmp_path)
    (root / "app/__init__.py").write_text("eval(value)\n")
    result = closure.inventory(root, ("app/main.py",))
    assert result["unresolved_dynamic_mechanisms"][0]["source"] == "app/__init__.py"


def test_symlink_source_refused(tmp_path: Path) -> None:
    root = repository(tmp_path)
    (root / "app/main.py").unlink()
    (root / "app/main.py").symlink_to(root / "app/__init__.py")
    with pytest.raises(closure.Refusal):
        closure.inventory(root, ("app/main.py",))


def test_real_preflight_graph_preserves_contract() -> None:
    result = closure.inventory(Path(__file__).resolve().parents[2])
    assert result["static_graph_closed"]
    assert result["preflight_external_static_roots"] == ["cryptography", "defusedxml"]
    assert result["preflight_dynamic_targets"] == ["libvirt", "zstandard"]
    assert len(result["sources"]) == 27
    assert "PySide6>=6.7,<6.11" in result["installation_contract"]["project_dependencies"]
    assert {item["mechanism"] for item in result["dynamic_mechanisms"]} == {
        "importlib.import_module"
    }
    assert all(
        "re.compile" not in item["mechanism"] and "collector.execute" not in item["mechanism"]
        for item in result["dynamic_mechanisms"]
    )


def test_cli_isolated_no_site_output(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[2]
    output = tmp_path / "inventory.json"
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-S",
            "-B",
            str(root / "scripts/stage9_v5d4_closure.py"),
            "--repository",
            str(root),
            "--output",
            str(output),
        ],
        capture_output=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(output.read_text())["static_graph_closed"]


def test_base_runtime_superset_hashes_cache_and_all_native_bytes(tmp_path: Path) -> None:
    base = tmp_path / "python3.12"
    (base / "lib-dynload").mkdir(parents=True)
    (base / "__pycache__").mkdir()
    values = {
        "site.py": b"raise RuntimeError('do not execute base code')\n",
        "__pycache__/site.cpython-312.pyc": b"opaque cached bytecode",
        "cached.pyo": b"opaque legacy bytecode",
        "lib-dynload/fcntl.cpython-312-x86_64-linux-gnu.so": b"\x7fELF synthetic",
    }
    for relative, data in values.items():
        (base / relative).write_bytes(data)
    for name in ("site-packages", "dist-packages"):
        (base / name).mkdir()
        (base / name / "excluded.pth").write_text("import dangerous\n")
    result = closure.base_runtime_inventory([base])
    assert result["files"] == {
        str(base / name): hashlib.sha256(data).hexdigest() for name, data in values.items()
    }
    assert result["native"] == [str(base / "lib-dynload/fcntl.cpython-312-x86_64-linux-gnu.so")]
    assert result["excluded"] == sorted(
        str(base / name) for name in ("site-packages", "dist-packages")
    )
    assert not result["installation_approved"]


def test_base_runtime_zip_file_root_and_external_symlink_target(tmp_path: Path) -> None:
    archive = tmp_path / "python312.zip"
    archive.write_bytes(b"opaque zip bytes, never opened as code")
    base = tmp_path / "python3.12"
    base.mkdir()
    target = tmp_path / "outside.py"
    target.write_text("raise RuntimeError('never execute symlink target')\n")
    alias = base / "sitecustomize.py"
    alias.symlink_to("../outside.py")
    result = closure.base_runtime_inventory([base, archive])
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    assert result["files"][str(target)] == digest
    assert result["files"][str(archive)] == hashlib.sha256(archive.read_bytes()).hexdigest()
    assert str(alias) not in result["files"]
    assert result["symlinks"] == {
        str(alias): {"target": "../outside.py", "resolved": str(target), "sha256": digest}
    }


@pytest.mark.parametrize("kind", ["directory", "loop", "broken", "ancestor"])
def test_base_runtime_refuses_unsafe_symlinks(tmp_path: Path, kind: str) -> None:
    base = tmp_path / "base"
    base.mkdir()
    (base / "source.py").write_text("pass\n")
    if kind == "directory":
        (base / "alias").symlink_to(base, target_is_directory=True)
    elif kind == "loop":
        (base / "first.py").symlink_to("second.py")
        (base / "second.py").symlink_to("first.py")
    elif kind == "broken":
        (base / "absent.py").symlink_to("does-not-exist.py")
    else:
        (tmp_path / "root-alias").symlink_to(base, target_is_directory=True)
        base = tmp_path / "root-alias/subdir"
        (tmp_path / "base/subdir").mkdir()
    with pytest.raises((closure.Refusal, OSError)):
        closure.base_runtime_inventory([base])


def test_base_runtime_bounded_chain_records_every_external_hop(tmp_path: Path) -> None:
    base = tmp_path / "python3.12"
    base.mkdir()
    final = tmp_path / "libpython3.12.so.1.0"
    final.write_bytes(b"\x7fELF synthetic")
    middle = tmp_path / "libpython3.12.so.1"
    middle.symlink_to(final.name)
    alias = base / "libpython3.12.so"
    alias.symlink_to("../" + middle.name)
    result = closure.base_runtime_inventory([base])
    digest = hashlib.sha256(final.read_bytes()).hexdigest()
    assert result["files"] == {str(final): digest}
    assert result["native"] == [str(final)]
    assert result["symlinks"] == {
        str(alias): {"target": "../" + middle.name, "resolved": str(final), "sha256": digest},
        str(middle): {"target": final.name, "resolved": str(final), "sha256": digest},
    }


def test_base_runtime_chain_limit_refuses(tmp_path: Path) -> None:
    base = tmp_path / "base"
    base.mkdir()
    (tmp_path / "final").write_bytes(b"final")
    previous = tmp_path / "final"
    for number in range(17):
        link = tmp_path / f"alias-{number}"
        link.symlink_to(previous.name)
        previous = link
    (base / "start").symlink_to("../" + previous.name)
    with pytest.raises(closure.Refusal, match="limit 16"):
        closure.base_runtime_inventory([base])


def test_base_runtime_refuses_special_file_without_opening_it(tmp_path: Path) -> None:
    os.mkfifo(tmp_path / "pipe")
    with pytest.raises(closure.Refusal, match="specjalny"):
        closure.base_runtime_inventory([tmp_path])


@pytest.mark.parametrize("kind", ["file", "directory"])
def test_base_runtime_unreadable_evidence_refuses(tmp_path: Path, kind: str) -> None:
    target = tmp_path / "evidence"
    if kind == "file":
        target.write_bytes(b"required evidence")
    else:
        target.mkdir()
    target.chmod(0)
    try:
        with pytest.raises(closure.Refusal, match="Nieczytelny"):
            closure.base_runtime_inventory([target])
    finally:
        target.chmod(0o700)


def test_base_runtime_empty_or_pth_evidence_refuses(tmp_path: Path) -> None:
    with pytest.raises(closure.Refusal):
        closure.base_runtime_inventory([])
    with pytest.raises(closure.Refusal):
        closure.base_runtime_inventory([tmp_path])
    (tmp_path / "unexpected.pth").write_text("import dangerous\n")
    with pytest.raises(closure.Refusal, match=".pth"):
        closure.base_runtime_inventory([tmp_path])


def test_base_runtime_changed_file_changes_inventory(tmp_path: Path) -> None:
    target = tmp_path / "stdlib.py"
    target.write_bytes(b"first revision")
    before = closure.base_runtime_inventory([tmp_path])
    target.write_bytes(b"different revision")
    after = closure.base_runtime_inventory([tmp_path])
    assert before["files"] != after["files"]


def test_base_runtime_cli_isolated_without_repository(tmp_path: Path) -> None:
    base = tmp_path / "base"
    base.mkdir()
    (base / "required.py").write_text("raise RuntimeError('never execute')\n")
    archive = tmp_path / "python312.zip"
    archive.write_bytes(b"zip")
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-S",
            "-B",
            str(root / "scripts/stage9_v5d4_closure.py"),
            "--base-runtime-root",
            str(base),
            "--base-runtime-root",
            str(archive),
        ],
        capture_output=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["roots"] == sorted([str(base), str(archive)])
    assert report["profile"] == "stage9-v5d4-base-runtime-inventory-v1"

"""Syntetyczne dowody ABI; testy nie ładują libvirt ani badanego interpretera."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from scripts import stage9_v5d4_abi as abi

ROOT = Path(__file__).resolve().parents[2]
pytestmark = [pytest.mark.security, pytest.mark.unit]


def _elf(**updates: Any) -> abi.ELF:
    base = abi.ELF(
        machine="Advanced Micro Devices X86-64",
        elf_class="ELF64",
        endian="2's complement, little endian",
        kind="DYN",
        interpreter=None,
        needed=(),
        soname=None,
        search_paths=(),
        version_needs={},
        version_definitions=(),
    )
    return replace(base, **updates)


def _pin(expectations: dict[str, Any], path: Path) -> None:
    expectations["sha256"][str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def environment(tmp_path: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, abi.ELF]]:
    site = tmp_path / "site"
    site.mkdir()
    header = tmp_path / "patchlevel.h"
    header.write_text(
        "#define PY_MAJOR_VERSION 3\n#define PY_MINOR_VERSION 12\n#define PY_MICRO_VERSION 3\n",
        encoding="utf-8",
    )
    config = tmp_path / "_sysconfigdata.py"
    config.write_text(
        "build_time_vars = "
        + repr(
            {
                "VERSION": "3.12",
                "SOABI": abi.SOABI,
                "Py_DEBUG": 0,
                "MULTIARCH": "x86_64-linux-gnu",
            }
        )
        + "\nraise RuntimeError('Nie wolno uruchamiać sysconfigdata')\n",
        encoding="utf-8",
    )
    module = site / "libvirt.py"
    module.write_text(
        "import libvirtmod\nraise RuntimeError('Nie wolno importować')\n", encoding="utf-8"
    )
    extension = site / ("libvirtmod" + abi.EXTENSION_SUFFIX)
    interpreter = tmp_path / "python3.12"
    loader = tmp_path / "ld-linux-x86-64.so.2"
    native = tmp_path / "libvirt.so.0"
    libc = tmp_path / "libc.so.6"
    for path in (extension, interpreter, loader, native, libc):
        path.write_bytes(b"Syntetyczne bajty ELF: " + path.name.encode())
    venv = tmp_path / "pyvenv.cfg"
    venv.write_text("include-system-site-packages = false\n", encoding="utf-8")
    request: dict[str, Any] = {
        "profile": "stage9-v5d4-static-abi-v1",
        "interpreter": str(interpreter),
        "patchlevel_header": str(header),
        "sysconfigdata": str(config),
        "libvirt_module": str(module),
        "libvirt_extension": str(extension),
        "additional_extensions": [],
        "import_roots": [str(site)],
        "required_imports": {"libvirt": str(module), "libvirtmod": str(extension)},
        "library_map": {
            "/lib64/ld-linux-x86-64.so.2": str(loader),
            "libvirt.so.0": str(native),
            "libc.so.6": str(libc),
        },
        "pyvenv_cfg": str(venv),
    }
    expectations: dict[str, Any] = {
        "profile": "stage9-v5d4-static-expectations-v1",
        "review_reference": "TEST-ONLY niezależnie dostarczony zestaw oczekiwań",
        "sha256": {},
    }
    for path in (header, config, module, extension, interpreter, loader, native, libc, venv):
        _pin(expectations, path)
    records = {
        str(interpreter): _elf(
            kind="EXEC", interpreter="/lib64/ld-linux-x86-64.so.2", needed=("libc.so.6",)
        ),
        str(extension): _elf(
            needed=("libvirt.so.0",),
            version_needs={"libvirt.so.0": ("LIBVIRT_10.0.0",)},
            defined_python_symbols=("PyInit_libvirtmod",),
        ),
        str(loader): _elf(),
        str(native): _elf(
            soname="libvirt.so.0", needed=("libc.so.6",), version_definitions=("LIBVIRT_10.0.0",)
        ),
        str(libc): _elf(soname="libc.so.6"),
    }
    return request, expectations, records


def _diagnose(
    environment: tuple[dict[str, Any], dict[str, Any], dict[str, abi.ELF]],
) -> dict[str, Any]:
    request, expectations, records = environment
    return abi.diagnose(request, expectations, reader=lambda path: records[str(path)])


def test_complete_static_closure_preserves_execution_refusals(environment: Any) -> None:
    request, expectations, records = environment
    report = _diagnose(environment)
    assert report["status"] == "STATIC_ABI_MATCH"
    assert set(report["elf"]) == set(records)
    assert report["imports"] == request["required_imports"]
    assert report["measurements"] == expectations["sha256"]
    assert report["native_load"] == "NOT_EXECUTED_REQUIRES_SEPARATE_EXPLICIT_AUTHORIZATION"
    assert report["launch_identity"] == "NOT_VERIFIED"
    assert report["approval_authority"] == "NOT_ESTABLISHED_BY_THIS_TOOL"
    assert report["safety"] == abi.SAFETY
    assert report["safety"]["Production"] == "NO-GO"


@pytest.mark.parametrize(
    ("field", "content", "reason"),
    [
        (
            "patchlevel_header",
            "#define PY_MAJOR_VERSION 3\n#define PY_MINOR_VERSION 11\n#define PY_MICRO_VERSION 3\n",
            "CPYTHON_VERSION_MISMATCH",
        ),
        (
            "sysconfigdata",
            "build_time_vars={'VERSION':'3.11','SOABI':'cpython-311-x86_64-linux-gnu'}\n",
            "CPYTHON_SOABI_MISMATCH",
        ),
        (
            "sysconfigdata",
            "build_time_vars={'VERSION':'3.12','SOABI':'cpython-312-x86_64-linux-gnu','Py_DEBUG':1}\n",
            "CPYTHON_DEBUG_ABI_REJECTED",
        ),
        (
            "sysconfigdata",
            "build_time_vars={'VERSION':'3.12','SOABI':'cpython-312-x86_64-linux-gnu','MULTIARCH':'aarch64-linux-gnu','Py_DEBUG':0}\n",
            "CPYTHON_ARCH_MISMATCH",
        ),
        ("pyvenv_cfg", "include-system-site-packages = true\n", "SYSTEM_SITE_PACKAGES_REJECTED"),
        ("libvirt_module", "pass\n", "LIBVIRT_NATIVE_IMPORT_MISSING"),
    ],
)
def test_wrong_python_or_launch_contract(
    environment: Any, field: str, content: str, reason: str
) -> None:
    request, expectations, _ = environment
    path = Path(request[field])
    path.write_text(content, encoding="utf-8")
    _pin(expectations, path)
    with pytest.raises(abi.Refusal, match=reason):
        _diagnose(environment)


def test_cpython_311_system_binding_is_refused(environment: Any) -> None:
    request, expectations, _ = environment
    current = Path(request["libvirt_extension"])
    wrong = current.with_name("libvirtmod.cpython-311-x86_64-linux-gnu.so")
    current.rename(wrong)
    request["libvirt_extension"] = str(wrong)
    request["required_imports"]["libvirtmod"] = str(wrong)
    _pin(expectations, wrong)
    with pytest.raises(abi.Refusal, match="LIBVIRTMOD_SOABI_MISMATCH"):
        _diagnose(environment)


def test_missing_transitive_library(environment: Any) -> None:
    request, _, _ = environment
    del request["library_map"]["libc.so.6"]
    with pytest.raises(abi.Refusal, match="ELF_DEPENDENCY_MISSING"):
        _diagnose(environment)


def test_unapproved_native_library(environment: Any) -> None:
    request, expectations, _ = environment
    del expectations["sha256"][request["library_map"]["libvirt.so.0"]]
    with pytest.raises(abi.Refusal, match="ARTIFACT_NOT_EXTERNALLY_PINNED"):
        _diagnose(environment)


def test_hash_mismatch_never_loads_target(environment: Any) -> None:
    request, expectations, _ = environment
    expectations["sha256"][request["libvirt_extension"]] = "0" * 64
    with pytest.raises(abi.Refusal, match="ARTIFACT_SHA256_MISMATCH"):
        _diagnose(environment)


@pytest.mark.parametrize(
    ("updates", "reason"),
    [
        ({"machine": "AArch64"}, "ELF_ARCH_MISMATCH"),
        ({"elf_class": "ELF32"}, "ELF_ARCH_MISMATCH"),
        ({"endian": "2's complement, big endian"}, "ELF_ARCH_MISMATCH"),
        ({"search_paths": ("/tmp",)}, "ELF_LOADER_CONFIGURATION_REJECTED"),
        ({"kind": "REL"}, "ELF_LOADER_CONFIGURATION_REJECTED"),
        ({"version_needs": {"libvirt.so.0": ("LIBVIRT_99.0.0",)}}, "ELF_SYMBOL_VERSION_MISMATCH"),
    ],
)
def test_native_abi_and_loader_fail_closed(
    environment: Any, updates: dict[str, Any], reason: str
) -> None:
    request, _, records = environment
    key = request["libvirt_extension"]
    records[key] = replace(records[key], **updates)
    with pytest.raises(abi.Refusal, match=reason):
        _diagnose(environment)


def test_missing_elf_interpreter(environment: Any) -> None:
    request, _, records = environment
    records[request["interpreter"]] = _elf(kind="EXEC")
    with pytest.raises(abi.Refusal, match="ELF_INTERPRETER_MISSING"):
        _diagnose(environment)


def test_wrong_soname(environment: Any) -> None:
    request, _, records = environment
    key = request["library_map"]["libvirt.so.0"]
    records[key] = replace(records[key], soname="libvirt.so.999")
    with pytest.raises(abi.Refusal, match="ELF_SONAME_MISMATCH"):
        _diagnose(environment)


@pytest.mark.parametrize(
    "filename",
    [
        "unsafe.pth",
        "old.pyc",
        "old.pyo",
        "sitecustomize.py",
        "usercustomize.py",
        "sitecustomize/__init__.py",
        "usercustomize/__init__.py",
        "sitecustomize.abi3.so",
        "usercustomize.cpython-312-x86_64-linux-gnu.so",
        "package/sitecustomize/module.py",
        "package/__pycache__/cached.txt",
    ],
)
def test_hidden_startup_and_bytecode_paths_are_refused(environment: Any, filename: str) -> None:
    request, _, _ = environment
    path = Path(request["import_roots"][0]) / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("import malicious\n", encoding="utf-8")
    with pytest.raises(abi.Refusal, match="STARTUP_IMPORT_CONFIGURATION_REJECTED"):
        _diagnose(environment)


def test_empty_unapproved_bytecode_cache_is_refused(environment: Any) -> None:
    request, _, _ = environment
    (Path(request["import_roots"][0]) / "__pycache__").mkdir()
    with pytest.raises(abi.Refusal, match="STARTUP_IMPORT_CONFIGURATION_REJECTED"):
        _diagnose(environment)


def test_shadowed_libvirt_module(environment: Any, tmp_path: Path) -> None:
    request, _, _ = environment
    shadow = tmp_path / "shadow"
    shadow.mkdir()
    (shadow / "libvirt.py").write_text("raise RuntimeError('Nie uruchamiać')\n", encoding="utf-8")
    request["import_roots"].insert(0, str(shadow))
    with pytest.raises(abi.Refusal, match="IMPORT_RESOLUTION_MISMATCH"):
        _diagnose(environment)


def test_wrong_explicit_import_target(environment: Any) -> None:
    request, _, _ = environment
    request["required_imports"]["libvirt"] += ".wrong"
    with pytest.raises(abi.Refusal, match="LIBVIRT_IMPORT_BINDING_REQUIRED"):
        _diagnose(environment)


def test_missing_package_initializer(environment: Any) -> None:
    request, _, _ = environment
    package = Path(request["import_roots"][0]) / "package"
    package.mkdir()
    source = package / "module.py"
    source.write_text("pass\n", encoding="utf-8")
    request["required_imports"]["package.module"] = str(source)
    with pytest.raises(abi.Refusal, match="PACKAGE_INITIALIZER_AMBIGUOUS_OR_MISSING"):
        _diagnose(environment)


def test_missing_expectations_and_bad_schema_are_refused(environment: Any) -> None:
    request, _, records = environment
    with pytest.raises(abi.Refusal, match="EXTERNAL_EXPECTATIONS_REQUIRED"):
        abi.diagnose(request, {}, reader=lambda path: records[str(path)])
    incomplete = copy.deepcopy(request)
    del incomplete["required_imports"]
    with pytest.raises(abi.Refusal, match="REQUEST_SCHEMA_REJECTED"):
        abi.diagnose(incomplete, {}, reader=lambda path: records[str(path)])


def test_symlink_dependency_is_refused(environment: Any, tmp_path: Path) -> None:
    request, expectations, _ = environment
    target = Path(request["library_map"]["libvirt.so.0"])
    linked = tmp_path / "libvirt-link.so.0"
    linked.symlink_to(target)
    request["library_map"]["libvirt.so.0"] = str(linked)
    expectations["sha256"][str(linked)] = hashlib.sha256(target.read_bytes()).hexdigest()
    with pytest.raises(abi.Refusal, match="SYMLINK_REJECTED"):
        _diagnose(environment)


@pytest.mark.parametrize("kind", ["file", "ancestor_directory"])
def test_symlink_refusal_category_precedes_windows_backend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    target = tmp_path / "target"
    target.mkdir()
    (target / "evidence.bin").write_bytes(b"inert data")
    link = tmp_path / "alias"
    if kind == "file":
        link.symlink_to(target / "evidence.bin")
        attempted = link
    else:
        link.symlink_to(target, target_is_directory=True)
        attempted = link / "evidence.bin"

    def forbidden_backend(*args, **kwargs):
        pytest.fail("Publiczna odmowa ABI musi poprzedzać wybór backendu Windows.")

    # Modelujemy tylko kolejność odmowy, nie kwalifikację natywnych uchwytów.
    monkeypatch.setattr(abi, "os", SimpleNamespace(name="nt", path=os.path))
    monkeypatch.setattr(abi, "runpy", SimpleNamespace(run_path=forbidden_backend))
    with pytest.raises(abi.Refusal, match="^SYMLINK_REJECTED$"):
        abi.read_bytes(attempted)


def test_readelf_parser_records_versions_without_execution() -> None:
    record = abi.parse_elf(
        """ELF Header:
 Class: ELF64
 Data: 2's complement, little endian
 Type: DYN (Shared object file)
 Machine: Advanced Micro Devices X86-64
 [Requesting program interpreter: /lib64/ld-linux-x86-64.so.2]
 0x00000001 (NEEDED) Shared library: [libvirt.so.0]
 0x0000000e (SONAME) Library soname: [binding.so]
 0x0000001d (RUNPATH) Library runpath: [$ORIGIN/.libs]
Version needs section '.gnu.version_r' contains 1 entry:
 000000: Version: 1 File: libvirt.so.0 Cnt: 1
 0x0010: Name: LIBVIRT_10.0.0 Flags: none Version: 2
Version definition section '.gnu.version_d' contains 1 entry:
 0x0010: Rev: 1 Flags: none Index: 2 Cnt: 1 Name: BINDING_1
"""
    )
    assert record.interpreter == "/lib64/ld-linux-x86-64.so.2"
    assert record.needed == ("libvirt.so.0",)
    assert record.soname == "binding.so"
    assert record.search_paths == ("$ORIGIN/.libs",)
    assert record.version_needs == {"libvirt.so.0": ("LIBVIRT_10.0.0",)}
    assert record.version_definitions == ("BINDING_1",)
    with pytest.raises(abi.Refusal, match="ELF_HEADER_INCOMPLETE"):
        abi.parse_elf("Niekompletny wynik readelf")


def test_cli_missing_evidence_emits_refusal(tmp_path: Path) -> None:
    request = tmp_path / "request.json"
    expectations = tmp_path / "expectations.json"
    request.write_text("{}", encoding="utf-8")
    expectations.write_text("{}", encoding="utf-8")
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-S",
            "-B",
            str(ROOT / "scripts/stage9_v5d4_abi.py"),
            str(request),
            "--expectations",
            str(expectations),
            "--expectations-sha256",
            hashlib.sha256(expectations.read_bytes()).hexdigest(),
        ],
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    assert result.returncode == 2
    report = json.loads(result.stdout)
    assert report["status"] == "REFUSED"
    assert report["reason"] == "REQUEST_SCHEMA_REJECTED"
    assert report["native_load"] == "NOT_EXECUTED"
    assert report["safety"] == abi.SAFETY


def test_libvirt_extension_missing_init_symbol(environment: Any) -> None:
    request, _, records = environment
    key = request["libvirt_extension"]
    records[key] = replace(records[key], defined_python_symbols=())
    with pytest.raises(abi.Refusal, match="EXTENSION_INIT_SYMBOL_MISSING"):
        _diagnose(environment)


def test_extension_missing_python_symbol(environment: Any) -> None:
    request, _, records = environment
    key = request["libvirt_extension"]
    records[key] = replace(records[key], undefined_python_symbols=("PyMissing312",))
    with pytest.raises(abi.Refusal, match="PYTHON_SYMBOL_MISSING"):
        _diagnose(environment)


def test_readelf_only_executes_fixed_host_tool(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    target = tmp_path / "untrusted-target.so"
    target.write_bytes(b"not executed")
    seen: list[list[str]] = []

    def fake_run(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[bytes]:
        seen.append(argv)
        assert argv[0] == "/usr/bin/readelf"
        assert argv[-1] == str(target)
        assert kwargs["timeout"] == 30
        assert kwargs["env"] == abi.ENV
        return subprocess.CompletedProcess(
            argv,
            0,
            stdout=b"Class: ELF64\nData: 2's complement, little endian\nType: DYN\nMachine: Advanced Micro Devices X86-64\n",
            stderr=b"",
        )

    monkeypatch.setattr(abi.subprocess, "run", fake_run)
    record = abi.read_elf(target)
    assert record.elf_class == "ELF64"
    assert len(seen) == 1
    assert "--dynamic" in seen[0]
    assert "--version-info" in seen[0]


def test_readelf_failure_refuses(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    target = tmp_path / "broken.so"
    target.write_bytes(b"invalid elf")
    monkeypatch.setattr(
        abi.subprocess,
        "run",
        lambda argv, **kwargs: subprocess.CompletedProcess(
            argv, 1, stdout=b"", stderr=b"private data"
        ),
    )
    with pytest.raises(abi.Refusal, match="READELF_FAILED"):
        abi.read_elf(target)


def test_undeclared_native_artifact_in_import_root_is_refused(environment: Any) -> None:
    request, _, _ = environment
    unexpected = Path(request["import_roots"][0]) / "unexpected.so"
    unexpected.write_bytes(b"Nie przejrzana biblioteka natywna")
    with pytest.raises(abi.Refusal, match="NATIVE_ARTIFACT_UNDECLARED"):
        _diagnose(environment)


def _vendored_library(environment: Any) -> Path:
    request, expectations, records = environment
    previous = Path(request["library_map"]["libvirt.so.0"])
    directory = Path(request["import_roots"][0]) / ".libs"
    directory.mkdir()
    target = directory / previous.name
    target.write_bytes(previous.read_bytes())
    request["library_map"]["libvirt.so.0"] = str(target)
    records[str(target)] = records.pop(str(previous))
    _pin(expectations, target)
    extension = request["libvirt_extension"]
    records[extension] = replace(records[extension], search_paths=("$ORIGIN/.libs",))
    return target


def test_origin_runpath_stays_in_explicit_native_closure(environment: Any) -> None:
    target = _vendored_library(environment)
    report = _diagnose(environment)
    assert report["status"] == "STATIC_ABI_MATCH"
    assert str(target) in report["elf"]


def test_origin_runpath_escape_is_refused(environment: Any) -> None:
    request, _, records = environment
    extension = request["libvirt_extension"]
    records[extension] = replace(records[extension], search_paths=("$ORIGIN/../outside",))
    with pytest.raises(abi.Refusal, match="ELF_SEARCH_PATH_ESCAPE"):
        _diagnose(environment)


def test_origin_runpath_shadowing_is_refused(environment: Any) -> None:
    request, expectations, records = environment
    target = _vendored_library(environment)
    shadow_directory = target.parent.parent / ".shadow"
    shadow_directory.mkdir()
    shadow = shadow_directory / target.name
    shadow.write_bytes(b"Odmienne bajty biblioteki")
    request["library_map"]["unused.so.0"] = str(shadow)
    _pin(expectations, shadow)
    records[str(shadow)] = _elf(soname="unused.so.0")
    extension = request["libvirt_extension"]
    records[extension] = replace(
        records[extension], search_paths=("$ORIGIN/.shadow:$ORIGIN/.libs",)
    )
    with pytest.raises(abi.Refusal, match="ELF_SEARCH_PATH_SHADOWING"):
        _diagnose(environment)


def test_dynamic_python_symbols_are_parsed() -> None:
    record = abi.parse_elf(
        """Class: ELF64
Data: 2's complement, little endian
Type: DYN
Machine: Advanced Micro Devices X86-64
Symbol table '.dynsym' contains 4 entries:
 Num: Value Size Type Bind Vis Ndx Name
 1: 0000000000000000 0 NOTYPE GLOBAL DEFAULT UND PyLong_FromLong@PYTHON_3.12
 2: 0000000000000000 0 NOTYPE GLOBAL DEFAULT UND _Py_Dealloc
 3: 0000000000010000 4 FUNC GLOBAL DEFAULT 12 PyInit_libvirtmod
 4: 0000000000010200 4 FUNC GLOBAL DEFAULT 12 unrelated_symbol
"""
    )
    assert record.defined_python_symbols == ("PyInit_libvirtmod",)
    assert record.undefined_python_symbols == ("PyLong_FromLong", "_Py_Dealloc")


def test_missing_debug_abi_evidence_is_refused(environment: Any) -> None:
    request, expectations, _ = environment
    config = Path(request["sysconfigdata"])
    config.write_text(
        "build_time_vars={'VERSION':'3.12','SOABI':'cpython-312-x86_64-linux-gnu','MULTIARCH':'x86_64-linux-gnu'}\n",
        encoding="utf-8",
    )
    _pin(expectations, config)
    with pytest.raises(abi.Refusal, match="CPYTHON_DEBUG_ABI_REJECTED"):
        _diagnose(environment)


def test_ambiguous_patchlevel_evidence_is_refused(environment: Any) -> None:
    request, expectations, _ = environment
    header = Path(request["patchlevel_header"])
    header.write_text(header.read_text() + "#define PY_MICRO_VERSION 99\n", encoding="utf-8")
    _pin(expectations, header)
    with pytest.raises(abi.Refusal, match="CPYTHON_VERSION_MISMATCH"):
        _diagnose(environment)


def test_python_symbols_must_come_from_interpreter_link_closure(environment: Any) -> None:
    request, _, records = environment
    key = request["libvirt_extension"]
    records[key] = replace(records[key], undefined_python_symbols=("PyLong_FromLong",))
    interpreter = request["interpreter"]
    records[interpreter] = replace(
        records[interpreter], defined_python_symbols=("PyLong_FromLong",)
    )
    assert _diagnose(environment)["status"] == "STATIC_ABI_MATCH"
    records[interpreter] = replace(records[interpreter], defined_python_symbols=())
    native = request["library_map"]["libvirt.so.0"]
    records[native] = replace(records[native], defined_python_symbols=("PyLong_FromLong",))
    with pytest.raises(abi.Refusal, match="PYTHON_SYMBOL_MISSING"):
        _diagnose(environment)


@pytest.mark.parametrize("kind", ["DYN", "EXEC"])
def test_generic_native_executable_has_pinned_loader(environment: Any, kind: str) -> None:
    request, expectations, records = environment
    executable = Path(request["import_roots"][0]) / "QtWebEngineProcess"
    executable.write_bytes(b"Syntetyczny ELF bez rozszerzenia nazwy")
    _pin(expectations, executable)
    request["additional_extensions"].append(str(executable))
    loader = next(name for name in request["library_map"] if name.startswith("/"))
    records[str(executable)] = _elf(kind=kind, interpreter=loader, needed=("libc.so.6",))
    report = _diagnose(environment)
    assert str(executable) in report["elf_measurements"]
    assert (str(executable), loader, request["library_map"][loader]) in report["elf_edges"]
    records[str(executable)] = replace(records[str(executable)], kind="REL")
    with pytest.raises(abi.Refusal, match="ELF_LOADER_CONFIGURATION_REJECTED"):
        _diagnose(environment)


def test_generic_native_executable_without_loader_refuses(environment: Any) -> None:
    request, expectations, records = environment
    executable = Path(request["import_roots"][0]) / "QtWebEngineProcess"
    executable.write_bytes(b"Syntetyczny ELF")
    _pin(expectations, executable)
    request["additional_extensions"].append(str(executable))
    records[str(executable)] = _elf(kind="EXEC")
    with pytest.raises(abi.Refusal, match="NATIVE_EXECUTABLE_LOADER_REQUIRED"):
        _diagnose(environment)
    records[str(executable)] = _elf(kind="EXEC", interpreter="/niezatwierdzony-loader")
    with pytest.raises(abi.Refusal, match="ELF_DEPENDENCY_MISSING"):
        _diagnose(environment)


def test_large_native_artifact_read_and_oversized_refusal(tmp_path: Path) -> None:
    path = tmp_path / "libQtWebEngineCore.so.6"
    size = 65 * 1024 * 1024
    with path.open("wb") as stream:
        stream.seek(size - 1)
        stream.write(b"\x00")
    assert len(abi.read_bytes(path)) == size
    with path.open("r+b") as stream:
        stream.truncate(abi.LIMIT + 1)
    with pytest.raises(abi.Refusal, match="FILE_LIMIT_EXCEEDED"):
        abi.read_bytes(path)


def test_cli_parses_exact_pinned_expectation_bytes_once(
    environment: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    request, expectations, records = environment
    request_path = tmp_path / "request.json"
    request_path.write_text(json.dumps(request), encoding="utf-8")
    expectations_path = tmp_path / "expectations.json"
    pinned = json.dumps(expectations).encode()
    expectations_path.write_bytes(pinned)
    original_read = abi.read_bytes
    original_diagnose = abi.diagnose
    reads: list[Path] = []

    def mutate_after_pinned_read(path):
        raw = original_read(path)
        if path == expectations_path:
            reads.append(path)
            expectations_path.write_bytes(b'{"changed_after_pinned_read":"must not be parsed"}')
        return raw

    monkeypatch.setattr(abi, "read_bytes", mutate_after_pinned_read)
    monkeypatch.setattr(
        abi,
        "diagnose",
        lambda current_request, current_expectations: original_diagnose(
            current_request,
            current_expectations,
            reader=lambda path: records[str(path)],
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            abi.__file__,
            str(request_path),
            "--expectations",
            str(expectations_path),
            "--expectations-sha256",
            abi.sha256(pinned),
        ],
    )
    assert abi.main() == 0
    report = json.loads(capsys.readouterr().out)
    assert reads == [expectations_path]
    assert expectations_path.read_bytes() != pinned
    assert report["measurements"] == expectations["sha256"]
    assert report["approval_authority"] == "NOT_ESTABLISHED_BY_THIS_TOOL"

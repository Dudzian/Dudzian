"""Statyczne porównanie ABI i importów V5d4; nie ładuje badanego kodu.

Oczekiwania muszą pochodzić z osobnego przeglądu. Zgodność z podanym plikiem
nie uwierzytelnia jego autora ani nie autoryzuje uruchomienia kontrolera.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
import stat
import subprocess
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

SOABI = "cpython-312-x86_64-linux-gnu"
EXTENSION_SUFFIX = f".{SOABI}.so"
LIMIT = 512 * 1024 * 1024
READELF_OUTPUT_LIMIT = 64 * 1024 * 1024
ENV = {"PATH": "/usr/bin:/bin", "LANG": "C", "LC_ALL": "C"}
SAFETY = {
    "Stage 9": "IN_PROGRESS",
    "Observer 3A": "BLOCKED",
    "Stage 3B": "NOT AUTHORIZED",
    "Production": "NO-GO",
}


class Refusal(ValueError):
    """Stała diagnoza odmowy bez ujawniania treści badanego pliku."""


def require(condition: bool, code: str) -> None:
    if not condition:
        raise Refusal(code)


def file_identity(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_nlink,
        value.st_uid,
        value.st_gid,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def read_bytes(path: Path) -> bytes:
    require(path.is_absolute() and str(path) == os.path.abspath(path), "PATH_REJECTED")
    for component in (*reversed(path.parents), path):
        require(not component.is_symlink(), "SYMLINK_REJECTED")
    metadata = path.lstat()
    require(stat.S_ISREG(metadata.st_mode) and metadata.st_nlink == 1, "FILE_REJECTED")
    require(0 <= metadata.st_size <= LIMIT, "FILE_LIMIT_EXCEEDED")
    descriptor = os.open(
        path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    )
    with os.fdopen(descriptor, "rb") as stream:
        require(
            file_identity(os.fstat(stream.fileno())) == file_identity(metadata), "ARTIFACT_CHANGED"
        )
        data = stream.read(LIMIT + 1)
        require(
            file_identity(os.fstat(stream.fileno())) == file_identity(metadata), "ARTIFACT_CHANGED"
        )
    require(len(data) <= LIMIT, "FILE_LIMIT_EXCEEDED")
    require(
        file_identity(metadata) == file_identity(path.lstat()) and len(data) == metadata.st_size,
        "ARTIFACT_CHANGED",
    )
    return data


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in items:
        require(key not in result, "DUPLICATE_JSON_KEY")
        result[key] = value
    return result


def parse_json(raw: bytes) -> dict[str, Any]:
    value = json.loads(raw, object_pairs_hook=_pairs)
    require(type(value) is dict, "JSON_OBJECT_REQUIRED")
    return cast(dict[str, Any], value)


def read_json(path: Path) -> dict[str, Any]:
    return parse_json(read_bytes(path))


@dataclass(frozen=True)
class ELF:
    machine: str
    elf_class: str
    endian: str
    kind: str
    interpreter: str | None
    needed: tuple[str, ...]
    soname: str | None
    search_paths: tuple[str, ...]
    version_needs: dict[str, tuple[str, ...]]
    version_definitions: tuple[str, ...]
    defined_python_symbols: tuple[str, ...] = ()
    undefined_python_symbols: tuple[str, ...] = ()


def parse_elf(output: str) -> ELF:
    """Parse tylko jawne pola readelf; nie wywołuje loadera ani `ldd`."""

    def field(name: str) -> str:
        match = re.search(rf"^\s*{re.escape(name)}:\s*(.+)$", output, re.MULTILINE)
        require(match is not None, "ELF_HEADER_INCOMPLETE")
        assert match is not None
        return match.group(1).strip()

    loader = re.search(r"\[Requesting program interpreter: ([^\]]+)\]", output)
    soname = re.search(r"\(SONAME\).*\[([^\]]+)\]", output)
    version_needs: dict[str, list[str]] = {}
    definitions: list[str] = []
    section, library = "", ""
    for line in output.splitlines():
        if line.startswith("Version needs section"):
            section = "needs"
        elif line.startswith("Version definition section"):
            section = "definitions"
        elif line.startswith("Version symbols section"):
            section = "symbols"
        if section == "needs":
            match = re.search(r"\bFile: (\S+)", line)
            if match:
                library = match.group(1)
                version_needs.setdefault(library, [])
            match = re.search(r"\bName: (\S+)", line)
            if match:
                require(bool(library), "ELF_VERSION_SECTION_REJECTED")
                version_needs[library].append(match.group(1))
        elif section == "definitions":
            match = re.search(r"\bName: (\S+)", line)
            if match:
                definitions.append(match.group(1))
    defined_python_symbols: set[str] = set()
    undefined_python_symbols: set[str] = set()
    for line in output.splitlines():
        fields = line.split()
        if len(fields) >= 8 and re.fullmatch(r"[0-9]+:", fields[0]):
            name = fields[7].split("@")[0]
            if name.startswith(("Py", "_Py")):
                (undefined_python_symbols if fields[6] == "UND" else defined_python_symbols).add(
                    name
                )
    return ELF(
        field("Machine"),
        field("Class"),
        field("Data"),
        field("Type").split()[0],
        loader.group(1) if loader else None,
        tuple(re.findall(r"\(NEEDED\).*\[([^\]]+)\]", output)),
        soname.group(1) if soname else None,
        tuple(re.findall(r"\((?:RPATH|RUNPATH)\).*\[([^\]]*)\]", output)),
        {name: tuple(versions) for name, versions in version_needs.items()},
        tuple(definitions),
        tuple(sorted(defined_python_symbols)),
        tuple(sorted(undefined_python_symbols)),
    )


def read_elf(path: Path) -> ELF:
    # readelf jest narzędziem zaufanego hosta, plik docelowy jest tylko danymi.
    before = sha256(read_bytes(path))
    result = subprocess.run(
        [
            "/usr/bin/readelf",
            "--file-header",
            "--program-headers",
            "--dynamic",
            "--version-info",
            "--dyn-syms",
            "--wide",
            str(path),
        ],
        env=ENV,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=30,
        check=False,
    )
    require(result.returncode == 0 and len(result.stdout) <= READELF_OUTPUT_LIMIT, "READELF_FAILED")
    require(sha256(read_bytes(path)) == before, "ARTIFACT_CHANGED")
    return parse_elf(result.stdout.decode("utf-8", errors="strict"))


def _path(value: Any) -> Path:
    require(
        type(value) is str
        and bool(value)
        and not any(ord(char) < 32 or ord(char) == 127 for char in value),
        "PATH_REJECTED",
    )
    path = Path(value)
    require(path.is_absolute() and str(path) == os.path.abspath(path), "PATH_REJECTED")
    return path


def _strings(value: Any, code: str) -> list[str]:
    require(type(value) is list and all(type(item) is str for item in value), code)
    require(len(value) == len(set(value)), code)
    return cast(list[str], value)


def _pins(expectations: dict[str, Any]) -> dict[str, str]:
    require(
        set(expectations) == {"profile", "review_reference", "sha256"}
        and expectations["profile"] == "stage9-v5d4-static-expectations-v1"
        and type(expectations["review_reference"]) is str
        and bool(expectations["review_reference"]),
        "EXTERNAL_EXPECTATIONS_REQUIRED",
    )
    pins = expectations["sha256"]
    require(type(pins) is dict and bool(pins), "EXTERNAL_EXPECTATIONS_REQUIRED")
    for path, digest in pins.items():
        _path(path)
        require(
            type(digest) is str and re.fullmatch(r"[a-f0-9]{64}", digest) is not None,
            "SHA256_REJECTED",
        )
    return dict(pins)


def _measure(path: Path, pins: dict[str, str], measurements: dict[str, str]) -> bytes:
    require(str(path) in pins, "ARTIFACT_NOT_EXTERNALLY_PINNED")
    data = read_bytes(path)
    digest = sha256(data)
    require(digest == pins[str(path)], "ARTIFACT_SHA256_MISMATCH")
    if str(path) in measurements:
        require(measurements[str(path)] == digest, "ARTIFACT_CHANGED")
    measurements[str(path)] = digest
    return data


def _python_contract(
    request: dict[str, Any], pins: dict[str, str], measured: dict[str, str]
) -> None:
    header = _measure(_path(request["patchlevel_header"]), pins, measured).decode("utf-8")
    for name, expected in (("MAJOR", "3"), ("MINOR", "12"), ("MICRO", "3")):
        matches = re.findall(rf"^#\s*define\s+PY_{name}_VERSION\s+(\d+)\s*$", header, re.MULTILINE)
        require(matches == [expected], "CPYTHON_VERSION_MISMATCH")
    tree = ast.parse(_measure(_path(request["sysconfigdata"]), pins, measured))
    values = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "build_time_vars"
            for target in node.targets
        ):
            values.append(ast.literal_eval(node.value))
    require(len(values) == 1 and type(values[0]) is dict, "SYSCONFIGDATA_REJECTED")
    variables = values[0]
    require(
        variables.get("VERSION") == "3.12" and variables.get("SOABI") == SOABI,
        "CPYTHON_SOABI_MISMATCH",
    )
    require(variables.get("Py_DEBUG") == 0, "CPYTHON_DEBUG_ABI_REJECTED")
    require(variables.get("MULTIARCH") == "x86_64-linux-gnu", "CPYTHON_ARCH_MISMATCH")


def _import_audit(
    request: dict[str, Any], pins: dict[str, str], measured: dict[str, str]
) -> dict[str, str]:
    roots = [_path(item) for item in _strings(request["import_roots"], "IMPORT_ROOTS_REJECTED")]
    require(bool(roots), "IMPORT_ROOTS_REQUIRED")
    for root in roots:
        require(root.is_dir() and not root.is_symlink(), "IMPORT_ROOT_REJECTED")
        for directory in (*reversed(root.parents), root):
            require(not directory.is_symlink(), "SYMLINK_REJECTED")
        for path in root.rglob("*"):
            require(not path.is_symlink(), "SYMLINK_REJECTED")
            if (
                any(
                    part in {"sitecustomize", "usercustomize"}
                    or part.startswith(("sitecustomize.", "usercustomize."))
                    for part in path.relative_to(root).parts
                )
                or path.suffix in {".pth", ".pyc", ".pyo"}
                or "__pycache__" in path.relative_to(root).parts
            ):
                raise Refusal("STARTUP_IMPORT_CONFIGURATION_REJECTED")
    imports = request["required_imports"]
    require(
        type(imports) is dict
        and imports.get("libvirt") == request["libvirt_module"]
        and imports.get("libvirtmod") == request["libvirt_extension"],
        "LIBVIRT_IMPORT_BINDING_REQUIRED",
    )
    resolved: dict[str, str] = {}
    for module, expected in imports.items():
        require(
            type(module) is str
            and re.fullmatch(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*", module) is not None,
            "MODULE_NAME_REJECTED",
        )
        parts = module.split(".")
        for prefix in range(1, len(parts)):
            initializers = [root.joinpath(*parts[:prefix], "__init__.py") for root in roots]
            initializers = [path for path in initializers if path.is_file()]
            require(len(initializers) == 1, "PACKAGE_INITIALIZER_AMBIGUOUS_OR_MISSING")
            _measure(initializers[0], pins, measured)
        relative = Path(*parts)
        candidates = [
            path
            for root in roots
            for path in (
                root / relative.with_suffix(".py"),
                root / relative / "__init__.py",
                root / (str(relative) + EXTENSION_SUFFIX),
                root / (str(relative) + ".abi3.so"),
            )
            if path.is_file()
        ]
        require(
            len(candidates) == 1 and str(candidates[0]) == expected, "IMPORT_RESOLUTION_MISMATCH"
        )
        _measure(candidates[0], pins, measured)
        resolved[module] = str(candidates[0])
    module_tree = ast.parse(_measure(_path(request["libvirt_module"]), pins, measured))
    require(
        any(
            isinstance(node, ast.Import) and any(alias.name == "libvirtmod" for alias in node.names)
            for node in ast.walk(module_tree)
        ),
        "LIBVIRT_NATIVE_IMPORT_MISSING",
    )
    if request["pyvenv_cfg"] is not None:
        config = _measure(_path(request["pyvenv_cfg"]), pins, measured).decode("utf-8")
        entries = {}
        for line in config.splitlines():
            if "=" in line:
                key, value = line.split("=", 1)
                require(key.strip().lower() not in entries, "VENV_CONFIG_REJECTED")
                entries[key.strip().lower()] = value.strip().lower()
        require(
            entries.get("include-system-site-packages") == "false", "SYSTEM_SITE_PACKAGES_REJECTED"
        )
    return resolved


def diagnose(
    request: dict[str, Any],
    expectations: dict[str, Any],
    *,
    reader: Callable[[Path], ELF] = read_elf,
) -> dict[str, Any]:
    """Porównuje jawny plan; nie dowodzi rzeczywistego launch ani dlopen closure."""
    required = {
        "profile",
        "interpreter",
        "patchlevel_header",
        "sysconfigdata",
        "libvirt_module",
        "libvirt_extension",
        "additional_extensions",
        "import_roots",
        "required_imports",
        "library_map",
        "pyvenv_cfg",
    }
    require(
        set(request) == required and request["profile"] == "stage9-v5d4-static-abi-v1",
        "REQUEST_SCHEMA_REJECTED",
    )
    pins = _pins(expectations)
    measured: dict[str, str] = {}
    _python_contract(request, pins, measured)
    extension = _path(request["libvirt_extension"])
    require(extension.name == "libvirtmod" + EXTENSION_SUFFIX, "LIBVIRTMOD_SOABI_MISMATCH")
    imports = _import_audit(request, pins, measured)
    library_map = request["library_map"]
    require(type(library_map) is dict and bool(library_map), "LIBRARY_MAP_REQUIRED")
    for name, path in library_map.items():
        require(
            type(name) is str and bool(name) and ("/" not in name or name.startswith("/")),
            "LIBRARY_NAME_REJECTED",
        )
        _path(path)
    pending = [_path(request["interpreter"]), extension]
    additional = _strings(request["additional_extensions"], "EXTENSIONS_REJECTED")
    for path in additional:
        require(
            ".cpython-" not in Path(path).name or path.endswith(EXTENSION_SUFFIX),
            "EXTENSION_ABI_MISMATCH",
        )
        pending.append(_path(path))
    # Każde natywne rozszerzenie importów ma być korzeniem grafu ELF.
    for path in imports.values():
        if path.endswith(".so") and _path(path) not in pending:
            pending.append(_path(path))
    declared_native = {str(path) for path in pending} | set(library_map.values())
    for root in request["import_roots"]:
        for path in Path(root).rglob("*"):
            if path.is_file() and (path.name.endswith(".so") or ".so." in path.name):
                require(str(path) in declared_native, "NATIVE_ARTIFACT_UNDECLARED")
                _measure(path, pins, measured)
                if path not in pending:
                    pending.append(path)
    records: dict[str, ELF] = {}
    edges: list[tuple[str, str, str]] = []
    while pending:
        path = pending.pop()
        if str(path) in records:
            continue
        require(len(records) < 1024, "ELF_CLOSURE_LIMIT_EXCEEDED")
        _measure(path, pins, measured)
        record = reader(path)
        require(
            record.machine == "Advanced Micro Devices X86-64"
            and record.elf_class == "ELF64"
            and "little endian" in record.endian,
            "ELF_ARCH_MISMATCH",
        )
        require(
            record.kind in {"DYN", "EXEC"},
            "ELF_LOADER_CONFIGURATION_REJECTED",
        )
        if path == _path(request["interpreter"]):
            require(record.interpreter is not None, "ELF_INTERPRETER_MISSING")
        elif record.kind == "EXEC":
            require(record.interpreter is not None, "NATIVE_EXECUTABLE_LOADER_REQUIRED")
        require(set(record.version_needs) <= set(record.needed), "ELF_VERSION_SECTION_REJECTED")
        records[str(path)] = record
        names = (*record.needed, *((record.interpreter,) if record.interpreter else ()))
        for name in names:
            require(name in library_map, "ELF_DEPENDENCY_MISSING")
            dependency = _path(library_map[name])
            edges.append((str(path), name, str(dependency)))
            pending.append(dependency)
    interpreter_closure = {str(_path(request["interpreter"]))}
    previous_size = -1
    while len(interpreter_closure) != previous_size:
        previous_size = len(interpreter_closure)
        interpreter_closure.update(
            target for source, _, target in edges if source in interpreter_closure
        )
    python_exports = {
        symbol for path in interpreter_closure for symbol in records[path].defined_python_symbols
    }
    extensions = {_path(request["libvirt_extension"]), *(_path(item) for item in additional)}
    for path in extensions:
        python_extension = path.name.endswith((EXTENSION_SUFFIX, ".abi3.so")) or any(
            symbol.startswith("PyInit_") for symbol in records[str(path)].defined_python_symbols
        )
        if python_extension:
            initializer = "PyInit_" + path.name.split(".", 1)[0]
            require(
                initializer in records[str(path)].defined_python_symbols,
                "EXTENSION_INIT_SYMBOL_MISSING",
            )
        require(
            set(records[str(path)].undefined_python_symbols) <= python_exports,
            "PYTHON_SYMBOL_MISSING",
        )
    import_roots = [_path(item) for item in request["import_roots"]]
    search_dirs: dict[str, list[Path]] = {}
    for source, record in records.items():
        require(len(record.search_paths) <= 1, "ELF_SEARCH_PATH_AMBIGUOUS")
        directories = []
        for expression in record.search_paths:
            for item in expression.split(":"):
                require(
                    item.startswith(("$ORIGIN/", "${ORIGIN}/")) or item in {"$ORIGIN", "${ORIGIN}"},
                    "ELF_LOADER_CONFIGURATION_REJECTED",
                )
                expanded = item.replace("${ORIGIN}", str(Path(source).parent)).replace(
                    "$ORIGIN", str(Path(source).parent)
                )
                require("$" not in expanded, "ELF_LOADER_CONFIGURATION_REJECTED")
                directory = Path(os.path.abspath(expanded))
                require(
                    any(directory.is_relative_to(root) for root in import_roots),
                    "ELF_SEARCH_PATH_ESCAPE",
                )
                require(
                    directory.is_dir() and not directory.is_symlink(),
                    "ELF_SEARCH_DIRECTORY_REJECTED",
                )
                directories.append(directory)
        search_dirs[source] = directories
    for source, name, target in edges:
        for directory in search_dirs[source]:
            candidate = directory / name
            if candidate.exists():
                require(str(candidate) == target, "ELF_SEARCH_PATH_SHADOWING")
                break
        if search_dirs[source] and any(Path(target).is_relative_to(root) for root in import_roots):
            require(
                any(str(directory / name) == target for directory in search_dirs[source]),
                "ELF_SEARCH_PATH_TARGET_MISSING",
            )
        record = records[target]
        if not name.startswith("/"):
            require(record.soname == name, "ELF_SONAME_MISMATCH")
        requested = records[source].version_needs.get(name, ())
        require(set(requested) <= set(record.version_definitions), "ELF_SYMBOL_VERSION_MISMATCH")
    for path, digest in measured.items():
        require(sha256(read_bytes(Path(path))) == digest, "ARTIFACT_CHANGED")
    return {
        "profile": "stage9-v5d4-static-abi-report-v1",
        "status": "STATIC_ABI_MATCH",
        "target": {
            "implementation": "CPython",
            "version": "3.12.3",
            "soabi": SOABI,
            "architecture": "x86_64",
        },
        "expectations_review_reference": expectations["review_reference"],
        "approval_authority": "NOT_ESTABLISHED_BY_THIS_TOOL",
        "measurements": measured,
        "imports": imports,
        "import_scope": "EXPLICIT_REQUIRED_MODULES_AND_PACKAGE_INITIALIZERS",
        "elf": {path: asdict(record) for path, record in records.items()},
        "elf_edges": edges,
        "python_symbol_sources": sorted(interpreter_closure),
        "elf_measurements": {path: measured[path] for path in records},
        "dynamic_resolution": "EXPLICIT_MAP_ONLY_NOT_LOADER_ATTESTATION",
        "native_load": "NOT_EXECUTED_REQUIRES_SEPARATE_EXPLICIT_AUTHORIZATION",
        "runtime_dlopen_closure": "NOT_PROVEN_BY_STATIC_ELF_NEEDED",
        "launch_identity": "NOT_VERIFIED",
        "safety": SAFETY,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("request", type=Path)
    parser.add_argument("--expectations", required=True, type=Path)
    parser.add_argument("--expectations-sha256", required=True)
    args = parser.parse_args()
    try:
        require(
            re.fullmatch(r"[a-f0-9]{64}", args.expectations_sha256) is not None, "SHA256_REJECTED"
        )
        expectations_raw = read_bytes(args.expectations)
        require(
            sha256(expectations_raw) == args.expectations_sha256,
            "EXPECTATIONS_SHA256_MISMATCH",
        )
        report = diagnose(read_json(args.request), parse_json(expectations_raw))
        status = 0
    except (
        Refusal,
        OSError,
        ValueError,
        KeyError,
        TypeError,
        SyntaxError,
        RecursionError,
        subprocess.SubprocessError,
    ) as exc:
        report = {
            "status": "REFUSED",
            "reason": str(exc) if isinstance(exc, Refusal) else "STATIC_ANALYSIS_FAILED",
            "native_load": "NOT_EXECUTED",
            "safety": SAFETY,
        }
        status = 2
    print(json.dumps(report, sort_keys=True, ensure_ascii=False))
    return status


if __name__ == "__main__":
    raise SystemExit(main())

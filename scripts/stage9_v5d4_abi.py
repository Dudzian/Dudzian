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
import runpy
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
    if os.name != "nt":
        require(
            os.name == "posix"
            and bool(getattr(os, "O_NOFOLLOW", None))
            and bool(getattr(os, "O_NONBLOCK", None)),
            "SECURE_FILE_READER_UNAVAILABLE",
        )
    # Zachowaj publiczne kategorie odmowy ABI przed wyborem backendu.
    for component in (*reversed(path.parents), path):
        require(not component.is_symlink(), "SYMLINK_REJECTED")
    metadata = path.lstat()
    require(stat.S_ISREG(metadata.st_mode) and metadata.st_nlink == 1, "FILE_REJECTED")
    require(0 <= metadata.st_size <= LIMIT, "FILE_LIMIT_EXCEEDED")
    if os.name == "nt":
        tools = runpy.run_path(str(Path(__file__).with_name("stage9_v5d4_dependencies.py")))
        try:
            return cast(bytes, tools["windows_read_regular"](path, LIMIT))
        except ValueError as exc:
            code = "FILE_LIMIT_EXCEEDED" if str(exc) == "INPUT_TOO_LARGE" else str(exc)
            raise Refusal(code) from exc
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
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


_VERSION_ERROR = "ELF_VERSION_SECTION_REJECTED"
_SYMBOL_ERROR = "ELF_DYNAMIC_SYMBOL_SECTION_REJECTED"
_SECTION_NAMES = {
    "Version needs section": ".gnu.version_r",
    "Version definition section": ".gnu.version_d",
    "Version symbols section": ".gnu.version",
    "Symbol table": ".dynsym",
}
_SECTION_PREFIXES = (
    "Version needs section",
    "Version definition section",
    "Version symbols section",
    "Symbol table '.dynsym'",
)
_SECTION_HEADER = re.compile(
    r"(Version (?:needs|definition|symbols) section|Symbol table)"
    r" '([^']+)' contains ([0-9]+) entr(?:y|ies):"
)
_VERSION_LINK = re.compile(r"Addr: 0x[0-9a-f]+\s+Offset: 0x[0-9a-f]+\s+Link: [0-9]+ \((\S+)\)")
_VERSION_ROW = re.compile(r"(?:0x)?([0-9a-f]+):\s+(.+)")
_NEED_LIBRARY = re.compile(r"Version: 1\s+File: (\S+)\s+Cnt: ([0-9]+)")
_NEED_NAME = re.compile(r"Name: (\S+)\s+Flags: .+?\s+Version: ([0-9]+)")
_DEFINITION = re.compile(r"Rev: 1\s+Flags: .+?\s+Index: ([0-9]+)\s+Cnt: ([0-9]+)\s+Name: (\S+)")
_DEFINITION_PARENT = re.compile(r"Parent ([0-9]+): (\S+)")
_VERSION_SYMBOL = re.compile(r"\s*([0-9a-f]+)(h|\s+)\(([^()]+)\)")
_VERSION_HIDDEN = 0x8000
_VERSION_INDEX_MASK = _VERSION_HIDDEN - 1
_RESERVED_VERSIONS = {0: "*local*", 1: "*global*"}
# GNU readelf can show the numeric OS-specific form for System V OS/ABI.
_GNU_OS_SPECIFIC = "<OS specific>: 10"
_SYMBOL_ROW = re.compile(
    r"([0-9]+):\s+[0-9a-f]+\s+(?:[0-9]+|0x[0-9a-f]+)"
    rf"\s+(NOTYPE|OBJECT|FUNC|SECTION|FILE|COMMON|TLS|IFUNC|RELC|SRELC|{_GNU_OS_SPECIFIC})"
    rf"\s+(LOCAL|GLOBAL|WEAK|UNIQUE|{_GNU_OS_SPECIFIC})\s+(DEFAULT|INTERNAL|HIDDEN|PROTECTED)"
    r"\s+(UND|ABS|COM|[0-9]+)(?:\s+(\S+)(?:\s+\(([0-9]+)\))?)?"
)
_TABLE_TAG = re.compile(
    r"^[ \t]+0x[0-9a-f]+[ \t]+\((SYMTAB|VERSYM|VERNEEDNUM|VERDEFNUM)\)[ \t]+(.+)$",
    re.MULTILINE,
)

_ReadelfSection = tuple[int, list[str]]


@dataclass(frozen=True)
class DynamicSymbol:
    name: str
    version: str | None
    default_version: bool
    kind: str
    binding: str
    visibility: str
    section_index: str
    version_index: int | None = None

    @property
    def is_export(self) -> bool:
        return (
            bool(self.name)
            and self.section_index != "UND"
            and self.binding in {"GLOBAL", "WEAK", "UNIQUE"}
            and self.visibility in {"DEFAULT", "PROTECTED"}
            and self.kind not in {"FILE", "SECTION"}
        )


def _readelf_sections(output: str) -> dict[str, _ReadelfSection]:
    sections: dict[str, _ReadelfSection] = {}
    rows: list[str] | None = None
    for line in output.splitlines():
        if not line.strip():
            continue
        if line[0].isspace():
            if rows is not None:
                rows.append(line.strip())
            continue
        # GNU readelf starts each report section at column zero. This includes
        # relocation and Notes headers, and sections added by newer versions.
        rows = None
        if line.startswith(_SECTION_PREFIXES):
            header = _SECTION_HEADER.fullmatch(line)
            code = _SYMBOL_ERROR if line.startswith("Symbol table") else _VERSION_ERROR
            require(header is not None, code)
            assert header is not None
            name, section_name, count = header.groups()
            require(name not in sections and section_name == _SECTION_NAMES[name], code)
            rows = []
            sections[name] = (int(count), rows)
    return sections


def _version_rows(section: _ReadelfSection, linked_table: str = ".dynstr") -> list[tuple[int, str]]:
    _, rows = section
    link = _VERSION_LINK.fullmatch(rows[0]) if rows else None
    require(link is not None and link[1] == linked_table, _VERSION_ERROR)
    data = []
    offsets: set[int] = set()
    for row in rows[1:]:
        match = _VERSION_ROW.fullmatch(row)
        require(match is not None, _VERSION_ERROR)
        assert match is not None
        offset = int(match[1], 16)
        require(offset not in offsets and (bool(offsets) or offset == 0), _VERSION_ERROR)
        offsets.add(offset)
        data.append((offset, match[2]))
    return data


def _version_requirements(
    section: _ReadelfSection,
) -> tuple[dict[str, list[str]], dict[int, str]]:
    expected, _ = section
    requirements: dict[str, list[str]] = {}
    current: list[str] = []
    remaining = 0
    indices: set[int] = set()
    required_by_index: dict[int, str] = {}
    library_offset = -1
    auxiliary_offset = -1
    for offset, row in _version_rows(section):
        library = _NEED_LIBRARY.fullmatch(row)
        if library is not None:
            name, count = library.groups()
            require(
                remaining == 0
                and name not in requirements
                and int(count) > 0
                and offset > library_offset,
                _VERSION_ERROR,
            )
            remaining = int(count)
            current = requirements[name] = []
            library_offset = auxiliary_offset = offset
        else:
            version = _NEED_NAME.fullmatch(row)
            require(
                version is not None and remaining > 0 and offset > auxiliary_offset, _VERSION_ERROR
            )
            assert version is not None
            index = int(version[2])
            base_index = index & _VERSION_INDEX_MASK
            require(
                index <= 0xFFFF and base_index >= 2 and base_index not in indices, _VERSION_ERROR
            )
            indices.add(base_index)
            required_by_index[index] = version[1]
            current.append(version[1])
            remaining -= 1
            auxiliary_offset = offset
    require(remaining == 0 and len(requirements) == expected, _VERSION_ERROR)
    return requirements, required_by_index


def _version_definitions(section: _ReadelfSection) -> dict[int, str]:
    expected, _ = section
    definitions: dict[int, str] = {}
    parents: set[str] = set()
    pending_parents = 0
    parent_number = 0
    definition_offset = -1
    auxiliary_offset = -1
    for offset, row in _version_rows(section):
        definition = _DEFINITION.fullmatch(row)
        if definition is not None:
            index, count = int(definition[1]), int(definition[2])
            require(
                pending_parents == 0
                and 1 <= index <= _VERSION_INDEX_MASK
                and index not in definitions
                and count > 0
                and offset > definition_offset,
                _VERSION_ERROR,
            )
            definitions[index] = definition[3]
            pending_parents = count - 1
            parent_number = 1
            definition_offset = auxiliary_offset = offset
        else:
            parent = _DEFINITION_PARENT.fullmatch(row)
            require(
                parent is not None
                and pending_parents > 0
                and int(parent[1]) == parent_number
                and offset > auxiliary_offset,
                _VERSION_ERROR,
            )
            assert parent is not None
            parents.add(parent[2])
            parent_number += 1
            pending_parents -= 1
            auxiliary_offset = offset
    require(
        pending_parents == 0
        and len(definitions) == expected
        and parents <= set(definitions.values()),
        _VERSION_ERROR,
    )
    return definitions


def _match_version_symbol(
    symbol: DynamicSymbol,
    index: int,
    hidden: bool,
    name: str,
    required: dict[int, str],
    provided: dict[int, str],
) -> None:
    require(index <= _VERSION_INDEX_MASK, _VERSION_ERROR)
    if index in _RESERVED_VERSIONS:
        # The versym index describes versioning, not the ELF symbol binding.
        require(
            not hidden and name == _RESERVED_VERSIONS[index] and symbol.version is None,
            _VERSION_ERROR,
        )
        return
    raw_index = index | (_VERSION_HIDDEN if hidden else 0)
    if raw_index in required:
        # GNU readelf prints the verneed index for imports and for defined data
        # provided by a copy relocation, including aliases of that storage.
        require(
            name == required[raw_index] == symbol.version
            and not symbol.default_version
            and symbol.version_index == raw_index,
            _VERSION_ERROR,
        )
        return
    require(
        name == provided.get(index)
        and symbol.section_index != "UND"
        and symbol.version_index is None,
        _VERSION_ERROR,
    )
    # readelf omits the suffix when the symbol and definition share their name
    # in the string table, as in the absolute symbols emitted by GNU ld.
    if symbol.version is None:
        require(symbol.name == name, _VERSION_ERROR)
    else:
        require(symbol.version == name and symbol.default_version != hidden, _VERSION_ERROR)


def _validate_version_symbols(
    section: _ReadelfSection,
    symbols: list[DynamicSymbol],
    required: dict[int, str],
    provided: dict[int, str],
) -> set[int]:
    expected, _ = section
    require(len(symbols) == expected, _VERSION_ERROR)
    require(
        all((index & _VERSION_INDEX_MASK) not in provided for index in required), _VERSION_ERROR
    )
    non_exported_symbols = set()
    count = 0
    for offset, row in _version_rows(section, ".dynsym"):
        require(offset == count, _VERSION_ERROR)
        position = 0
        while entry := _VERSION_SYMBOL.match(row, position):
            require(count < expected, _VERSION_ERROR)
            index_text, flag, name = entry.groups()
            index = int(index_text, 16)
            hidden = flag == "h"
            _match_version_symbol(symbols[count], index, hidden, name, required, provided)
            if hidden or index == 0:
                non_exported_symbols.add(count)
            position = entry.end()
            count += 1
        require(position == len(row), _VERSION_ERROR)
    require(count == expected, _VERSION_ERROR)
    return non_exported_symbols


def _dynamic_symbols(section: _ReadelfSection) -> list[DynamicSymbol]:
    expected, rows = section
    require(
        bool(rows)
        and rows[0].split() == ["Num:", "Value", "Size", "Type", "Bind", "Vis", "Ndx", "Name"],
        _SYMBOL_ERROR,
    )
    require(len(rows) - 1 == expected, _SYMBOL_ERROR)
    symbols = []
    for index, row in enumerate(rows[1:]):
        match = _SYMBOL_ROW.fullmatch(row)
        require(match is not None and int(match[1]) == index, _SYMBOL_ERROR)
        assert match is not None
        _, kind, binding, visibility, section_index, label, rendered_index = match.groups()
        name, version_mark, version = (label or "").partition("@")
        default_version = version.startswith("@")
        if default_version:
            version = version[1:]
        require(
            (not version_mark or bool(name) and bool(version) and "@" not in version)
            and (rendered_index is None or bool(version)),
            _SYMBOL_ERROR,
        )
        symbols.append(
            DynamicSymbol(
                name,
                version or None,
                default_version,
                "IFUNC" if kind == _GNU_OS_SPECIFIC else kind,
                "UNIQUE" if binding == _GNU_OS_SPECIFIC else binding,
                visibility,
                section_index,
                int(rendered_index) if rendered_index is not None else None,
            )
        )
    return symbols


def parse_elf(output: str) -> ELF:
    """Parse tylko jawne pola readelf; nie wywołuje loadera ani `ldd`."""

    def field(name: str) -> str:
        match = re.search(rf"^\s*{re.escape(name)}:\s*(.+)$", output, re.MULTILINE)
        require(match is not None, "ELF_HEADER_INCOMPLETE")
        assert match is not None
        return match.group(1).strip()

    machine, elf_class, endian, kind = (field(key) for key in ("Machine", "Class", "Data", "Type"))
    loader = re.search(r"\[Requesting program interpreter: ([^\]]+)\]", output)
    soname = re.search(r"\(SONAME\).*\[([^\]]+)\]", output)
    needed = tuple(re.findall(r"\(NEEDED\).*\[([^\]]+)\]", output))
    sections = _readelf_sections(output)
    needs = sections.get("Version needs section")
    definitions = sections.get("Version definition section")
    versions = sections.get("Version symbols section")
    dynamic = sections.get("Symbol table")
    if versions is not None:
        require(dynamic is not None, _SYMBOL_ERROR)
        assert dynamic is not None
        require(versions[0] == dynamic[0], _VERSION_ERROR)
    table_references = {
        "SYMTAB": dynamic,
        "VERSYM": versions,
        "VERNEEDNUM": needs,
        "VERDEFNUM": definitions,
    }
    seen_tags: set[str] = set()
    for tag, value in _TABLE_TAG.findall(output):
        section = table_references[tag]
        code = _SYMBOL_ERROR if tag == "SYMTAB" else _VERSION_ERROR
        require(tag not in seen_tags and section is not None, code)
        assert section is not None
        seen_tags.add(tag)
        if tag.endswith("NUM"):
            require(re.fullmatch(r"[0-9]+", value) is not None and int(value) == section[0], code)
        else:
            require(re.fullmatch(r"0x[0-9a-f]+", value) is not None, code)
    require(dynamic is not None or not needed, _SYMBOL_ERROR)
    version_needs, required = _version_requirements(needs) if needs is not None else ({}, {})
    provided = _version_definitions(definitions) if definitions is not None else {}
    symbols = _dynamic_symbols(dynamic) if dynamic is not None else []
    non_exported_symbols: set[int] = set()
    if versions is not None:
        non_exported_symbols = _validate_version_symbols(versions, symbols, required, provided)
    else:
        require(
            not required and not provided and all(s.version is None for s in symbols),
            _VERSION_ERROR,
        )
    defined_python_symbols: set[str] = set()
    undefined_python_symbols: set[str] = set()
    for index, symbol in enumerate(symbols):
        if not symbol.name.startswith(("Py", "_Py")):
            continue
        if symbol.section_index == "UND":
            undefined_python_symbols.add(symbol.name)
        elif (
            symbol.is_export
            and index not in non_exported_symbols
            and (symbol.version is None or symbol.default_version)
            and (not symbol.name.startswith("PyInit_") or symbol.kind in {"FUNC", "IFUNC"})
        ):
            defined_python_symbols.add(symbol.name)
    return ELF(
        machine,
        elf_class,
        endian,
        kind.split()[0],
        loader.group(1) if loader else None,
        needed,
        soname.group(1) if soname else None,
        tuple(re.findall(r"\((?:RPATH|RUNPATH)\).*\[([^\]]*)\]", output)),
        {library: tuple(entries) for library, entries in version_needs.items()},
        tuple(provided.values()),
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
    observer_base_libpython: Path | None = None,
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
    if observer_base_libpython is not None:
        require(str(observer_base_libpython) in additional, "OBSERVER_LIBPYTHON_NOT_DECLARED")
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
        record = records[str(path)]
        if path == observer_base_libpython:
            # The distro shared library exports initializers for built-in modules.
            require(
                path.name == record.soname == "libpython3.12.so.1.0"
                and record.kind == "DYN"
                and record.interpreter is None
                and str(path) not in imports.values(),
                "OBSERVER_LIBPYTHON_ROLE_MISMATCH",
            )
        elif path.name.endswith((EXTENSION_SUFFIX, ".abi3.so")) or any(
            symbol.startswith("PyInit_") for symbol in record.defined_python_symbols
        ):
            initializer = "PyInit_" + path.name.split(".", 1)[0]
            require(initializer in record.defined_python_symbols, "EXTENSION_INIT_SYMBOL_MISSING")
        require(
            set(record.undefined_python_symbols) <= python_exports,
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

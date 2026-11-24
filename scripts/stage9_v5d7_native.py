"""Static evidence for native and executable-format wheel members."""

from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
from typing import Any
import os
import re
import selectors
import shutil
import struct
import subprocess
import sys
import tempfile
import time


_TOOL_OUTPUT_LIMIT = 64 * 1024 * 1024
_TOOL_TIMEOUT_SECONDS = 120
_SOABI = re.compile(r"\.(cpython-[\w-]+|pypy[\w-]+|cp\d+[\w-]*|abi3)\.(?:so|pyd)$")
_SHARED_SUFFIX = re.compile(r"\.(?:so(?:\.\d+)*|pyd|dll|dylib)$")
_SCRIPT_NAMES = {
    ".bat": "WINDOWS_BATCH",
    ".cmd": "WINDOWS_BATCH",
    ".ps1": "POWERSHELL_SOURCE",
    ".sh": "SHELL_SOURCE",
    ".zsh": "SHELL_SOURCE",
    ".fish": "SHELL_SOURCE",
    ".py": "PYTHON_SOURCE",
    ".pyx": "CYTHON_SOURCE",
    ".spec": "BUILD_SPEC_SOURCE",
}
_BINARY_SUFFIXES = {".a", ".com", ".efi", ".exe", ".lib", ".o", ".obj", ".wasm"}
_MACH_MAGICS = {
    b"\xfe\xed\xfa\xce",
    b"\xce\xfa\xed\xfe",
    b"\xfe\xed\xfa\xcf",
    b"\xcf\xfa\xed\xfe",
    b"\xbe\xba\xfe\xca",
    b"\xca\xfe\xba\xbf",
    b"\xbf\xba\xfe\xca",
}
_COFF_MACHINES = {
    0x014C: "I386",
    0x0166: "MIPS",
    0x01C0: "ARM",
    0x01C2: "THUMB",
    0x01C4: "ARMNT",
    0x01F0: "POWERPC",
    0x0200: "IA64",
    0x5032: "RISCV32",
    0x5064: "RISCV64",
    0x5128: "RISCV128",
    0x6232: "LOONGARCH32",
    0x6264: "LOONGARCH64",
    0x8664: "AMD64",
    0xAA64: "ARM64",
    0xA641: "ARM64EC",
    0xA64E: "ARM64X",
}


def _run_tool(name: str, args: list[str], data: bytes | None = None) -> tuple[str, str, str]:
    if os.name != "posix":
        return "", "", f"{name} capture requires a POSIX host"
    executable = shutil.which(name, path=os.defpath)
    if executable is None:
        return "", "", f"{name} unavailable"

    with tempfile.TemporaryDirectory(prefix="stage9-v5d7-static-") as tmp:
        command = [str(Path(executable).resolve()), *args]
        if data is not None:
            # A private, non-executable regular file with a fixed name. Only the
            # analysis command is executed. The member is passed as plain data.
            fd = os.open(Path(tmp) / "member.bin", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, "wb") as member:
                member.write(data)
            command.extend(("--", "member.bin"))

        try:
            process = subprocess.Popen(
                command,
                cwd=tmp,
                env={"LC_ALL": "C", "PATH": os.defpath},
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except OSError as exc:
            return "", "", f"{name} could not start: {exc}"

        buffers = {"stdout": bytearray(), "stderr": bytearray()}
        size = 0
        error = ""
        deadline = time.monotonic() + _TOOL_TIMEOUT_SECONDS
        with process, selectors.DefaultSelector() as selector:
            assert process.stdout is not None and process.stderr is not None
            selector.register(process.stdout, selectors.EVENT_READ, "stdout")
            selector.register(process.stderr, selectors.EVENT_READ, "stderr")
            while selector.get_map():
                remaining = deadline - time.monotonic()
                events = selector.select(max(0, remaining)) if remaining > 0 else []
                if not events:
                    error = f"{name} timed out"
                    break
                for key, _ in events:
                    chunk = os.read(key.fd, 65536)
                    if not chunk:
                        selector.unregister(key.fd)
                        continue
                    size += len(chunk)
                    if size > _TOOL_OUTPUT_LIMIT:
                        error = f"{name} output exceeded {_TOOL_OUTPUT_LIMIT} bytes"
                        break
                    buffers[key.data].extend(chunk)
                if error:
                    break

            if error:
                process.kill()
            try:
                returncode = process.wait(timeout=max(0.1, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                error = f"{name} timed out"
                process.kill()
                returncode = process.wait()
            if returncode and not error:
                error = f"{name} exited with status {returncode}"

        return (
            buffers["stdout"].decode("utf-8", errors="replace"),
            buffers["stderr"].decode("utf-8", errors="replace"),
            error,
        )


def native_tool_versions() -> dict:
    tools = {}
    for name in ("readelf", "objdump"):
        stdout, stderr, error = _run_tool(name, ["--version"])
        tools[name] = {
            "version": stdout.splitlines()[0] if stdout else None,
            "stderr": stderr.strip() or None,
            "error": error or None,
        }
    return tools


_ELF_HEADER_KEYS = {
    "Class": "elf_class",
    "Data": "data_encoding",
    "OS/ABI": "os_abi",
    "ABI Version": "abi_version",
    "Type": "elf_type",
    "Machine": "machine",
    "Flags": "machine_flags",
    "Entry point address": "entry_point",
}
_DYNAMIC_STRINGS = {
    "NEEDED",
    "SONAME",
    "RPATH",
    "RUNPATH",
    "AUDIT",
    "DEPAUDIT",
    "FILTER",
    "AUXILIARY",
}
_DYNAMIC_OTHER = {
    "FLAGS",
    "FLAGS_1",
    "FEATURE_1",
    "TEXTREL",
    "SYMBOLIC",
    "VERNEED",
    "VERNEEDNUM",
    "VERDEF",
    "VERDEFNUM",
    "INIT",
    "FINI",
    "INIT_ARRAY",
    "FINI_ARRAY",
    "PREINIT_ARRAY",
}
_DYNAMIC_LINE = re.compile(r"^\s*0x[0-9a-f]+\s+\((\w+)\)\s*(.*)$")
_RELOCATION_SECTION = re.compile(r"^Relocation section '(.+)' .* contains (\d+) entr")
_RELOCATION_ROW = re.compile(r"^\s*[0-9a-f]+\s+[0-9a-f]+\s+(\S+)")
_RELR_ROW = re.compile(r"^\s*[0-9a-f]{8,16}\s*$")
_RELR_LOCATION_ROW = re.compile(r"^\s*(?:[0-9a-f]+:\s+[0-9a-f]+\s+)?[0-9a-f]{8,16}(?:\s+.*)?$")
_SYMBOL_ROW = re.compile(
    r"^\s*\d+:\s+(?:[0-9a-f]+)\s+\S+\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)(?:\s+(\S+))?"
)
_VERSION_FILE = re.compile(r"^\s*\w+: Version: \d+\s+File: (.+?)\s+Cnt: (\d+)")
_VERSION_NEED_NAME = re.compile(r"^\s*\w+:\s+Name: (.+?)\s+Flags: ")
_VERSION_DEF_NAME = re.compile(r"^\s*\w+: Rev: .*\s+Name: (.+)$")
_NUMERIC_VERSION = re.compile(r"\d+(?:\.\d+)*")
_ABI_FAMILIES = ("GLIBC", "GLIBCXX", "CXXABI")


def _version_requirements(needs: dict, imported: Counter) -> dict:
    declared = set().union(*needs.values())
    result = {}
    for family in _ABI_FAMILIES:
        prefix = f"{family}_"
        # Version-need records, rather than arbitrary exported symbols, define
        # what the loader requires. Import names are also reported independently.
        matching = sorted(version for version in declared if version.startswith(prefix))
        numeric = [
            version[len(prefix) :]
            for version in matching
            if _NUMERIC_VERSION.fullmatch(version[len(prefix) :])
        ]
        result[family] = {
            "required": matching,
            "max_numeric_required": max(numeric, key=_version_sort_key) if numeric else None,
            "import_symbol_versions": sorted(name for name in imported if name.startswith(prefix)),
        }
    return result


def _version_sort_key(value: str) -> tuple[tuple[int, str], ...]:
    parts = [part.lstrip("0") for part in value.split(".")]
    return tuple((len(part), part) for part in parts)


def _measure_elf(data: bytes) -> tuple[dict, list[str]]:
    stdout, stderr, error = _run_tool(
        "readelf",
        ["-W", "-h", "-l", "-d", "-V", "--dyn-syms", "-r", "-n"],
        data,
    )
    issues = [error] if error else []
    notes_warnings = []
    for message in stderr.splitlines():
        if message.startswith("readelf: Warning: Gap in build notes detected from "):
            # GNU build-note coverage warnings are preserved separately. They
            # do not indicate missing dynamic tags, symbols or relocations.
            notes_warnings.append(message)
        elif message.strip():
            issues.append(message[:1024])
    header = {}
    fields = defaultdict(list)
    version_needs = defaultdict(set)
    needs_counts: defaultdict[str, list[int]] = defaultdict(lambda: [0, 0])
    version_definitions = set()
    imported_versions: Counter[str] = Counter()
    exported_versions: Counter[str] = Counter()
    pyinit = set()
    qt_plugin_exports = set()
    relocations: list[dict[str, Any]] = []
    interpreter = None
    relocation: dict[str, Any] | None = None
    version_state = ""
    library = None
    in_symbols = False
    symbol_rows = 0
    symbol_expected = None
    abi_notes = []

    for line in stdout.splitlines():
        if line.startswith("  ") and ":" in line:
            key, value = line.split(":", 1)
            if key.strip() in _ELF_HEADER_KEYS:
                header[_ELF_HEADER_KEYS[key.strip()]] = value.strip()
        if "[Requesting program interpreter: " in line:
            interpreter = line.split("[Requesting program interpreter: ", 1)[1].rstrip("]")
        if "NT_GNU_ABI_TAG" in line or "NT_GNU_PROPERTY_TYPE_0" in line:
            abi_notes.append(line.strip())
        if line.lstrip().startswith(("Properties:", "OS: Linux,", "x86 ISA needed:")):
            abi_notes.append(line.strip())

        tag = _DYNAMIC_LINE.match(line)
        if tag:
            name, value = tag.groups()
            if name in _DYNAMIC_STRINGS:
                start, end = value.find("["), value.rfind("]")
                if 0 <= start < end:
                    fields[name].append(value[start + 1 : end])
                else:
                    issues.append(f"readelf output missing bracketed data for {name}")
            if name in _DYNAMIC_OTHER:
                fields[name].append(value)

        section = _RELOCATION_SECTION.match(line)
        if section:
            locations = re.search(r"which relocate (\d+) locations", line)
            relocation = {
                "section": section[1],
                "encoded_entries": int(section[2]),
                "types": Counter(),
                "decoded_locations_expected": int(locations[1]) if locations else None,
            }
            relocations.append(relocation)
        elif relocation is not None:
            record = _RELOCATION_ROW.match(line)
            if not line.strip():
                relocation = None
            elif line.startswith("Index:") or line.strip().rstrip(":").endswith("offsets"):
                continue
            elif relocation["decoded_locations_expected"] is not None and _RELR_LOCATION_ROW.match(
                line
            ):
                relocation["types"]["RELR_RELATIVE"] += 1
            elif record:
                relocation["types"][record[1]] += 1
            elif _RELR_ROW.match(line):
                relocation["types"]["RELR_RELATIVE"] += 1
            elif not line.startswith(" "):
                relocation = None

        if line.startswith("Symbol table "):
            in_symbols = True
            total = re.search(r"contains (\d+) entr", line)
            symbol_expected = int(total[1]) if total else None
        if not line.strip():
            in_symbols = False
        if in_symbols and (symbol := _SYMBOL_ROW.match(line)):
            symbol_rows += 1
            _, binding, visibility, index, name = symbol.groups()
            if name:
                base, separator, version = name.partition("@")
                if index == "UND" and separator:
                    imported_versions[version.lstrip("@")] += 1
                if (
                    index != "UND"
                    and binding in {"GLOBAL", "WEAK", "UNIQUE"}
                    and visibility in {"DEFAULT", "PROTECTED"}
                ):
                    if separator:
                        exported_versions[version.lstrip("@")] += 1
                    if base.startswith(("PyInit_", "PyInitU_")):
                        pyinit.add(base)
                    if base.startswith(("qt_plugin_instance", "qt_plugin_query_metadata")):
                        qt_plugin_exports.add(base)

        if line.startswith("Version "):
            version_state = line.split()[1]
            library = None
        if version_state == "needs":
            if file_match := _VERSION_FILE.search(line):
                library = file_match[1]
                needs_counts[library][0] += int(file_match[2])
            if library is not None and (name_match := _VERSION_NEED_NAME.search(line)):
                version_needs[library].add(name_match[1])
                needs_counts[library][1] += 1
        if version_state == "definition" and (name_match := _VERSION_DEF_NAME.search(line)):
            version_definitions.add(name_match[1])

    if not set(_ELF_HEADER_KEYS.values()).issubset(header):
        issues.append("ELF header fields missing from readelf output")
    if fields["VERNEED"] and not version_needs:
        issues.append("ELF version-need records were not decoded")
    for needed_lib, (expected, decoded) in needs_counts.items():
        if expected != decoded:
            issues.append(
                f"readelf version-need counts differ for {needed_lib} ({decoded}/{expected})"
            )
    if fields["VERDEF"] and not version_definitions:
        issues.append("ELF version-definition records were not decoded")
    if symbol_expected is not None and symbol_rows != symbol_expected:
        issues.append(f"readelf dynamic symbol counts differ ({symbol_rows}/{symbol_expected})")
    if fields["NEEDED"] and symbol_expected is None:
        issues.append("ELF declares dependencies but readelf provided no dynamic symbol table")
    for relocation_section in relocations:
        entry_count = sum(relocation_section["types"].values())
        expected = relocation_section["decoded_locations_expected"]
        if expected is None and "RELR_RELATIVE" in relocation_section["types"]:
            # Older readelf reports RELR offsets separately from encoded entries.
            continue
        if entry_count != (relocation_section["encoded_entries"] if expected is None else expected):
            issues.append(
                f"readelf did not decode every relocation in {relocation_section['section']}"
            )

    return {
        "header": header,
        "interpreter": interpreter,
        "dt_needed": fields["NEEDED"],
        "soname": fields["SONAME"],
        "rpath": fields["RPATH"],
        "runpath": fields["RUNPATH"],
        "other_dynamic_tags": {
            tag: values
            for tag, values in sorted(fields.items())
            if values and tag not in {"NEEDED", "SONAME", "RPATH", "RUNPATH"}
        },
        "version_needs_by_library": {
            key: sorted(values) for key, values in sorted(version_needs.items())
        },
        "version_definitions": sorted(version_definitions),
        "dynamic_symbol_rows": symbol_rows,
        "imported_symbol_versions": dict(sorted(imported_versions.items())),
        "exported_symbol_versions": dict(sorted(exported_versions.items())),
        "abi_version_requirements": _version_requirements(version_needs, imported_versions),
        "pyinit_exports": sorted(pyinit),
        "qt_plugin_exports": sorted(qt_plugin_exports),
        "relocations": relocations,
        "abi_notes": abi_notes,
        "build_note_warnings": notes_warnings,
    }, issues


def _coff_magic(data: bytes) -> bool:
    if len(data) < 20:
        return False
    machine, sections, _, _, _, opt_size, _ = struct.unpack_from("<HHIIIHH", data)
    return (
        machine in _COFF_MACHINES
        and 0 < sections <= 96
        and 20 + opt_size + sections * 40 <= len(data)
    )


def _coff_string(data: bytes, strings_offset: int, strings_size: int, offset: int) -> str:
    if not 4 <= offset < strings_size:
        raise ValueError("invalid COFF string table offset")
    begin = strings_offset + offset
    end = data.find(b"\0", begin, strings_offset + strings_size)
    if end < 0:
        raise ValueError("unterminated COFF long name")
    return data[begin:end].decode("ascii", errors="backslashreplace")


def _measure_coff(data: bytes) -> tuple[dict, list[str]]:
    machine, count, timestamp, symbol_offset, symbol_count, opt_size, flags = struct.unpack_from(
        "<HHIIIHH",
        data,
    )
    sections: list[dict] = []
    undefined_external = set()
    public_symbols = set()
    directives = []
    issues = []
    details = {
        "machine": _COFF_MACHINES[machine],
        "machine_code": f"0x{machine:04x}",
        "timestamp_raw": timestamp,
        "section_count": count,
        "optional_header_size": opt_size,
        "characteristics": f"0x{flags:04x}",
        "symbol_entries_including_aux": symbol_count,
        "sections": sections,
    }
    try:
        strings_offset = symbol_offset + 18 * symbol_count
        strings_size = 0
        if symbol_offset:
            if symbol_offset < 20 + opt_size + 40 * count or strings_offset + 4 > len(data):
                raise ValueError("invalid COFF symbol table extent")
            strings_size = struct.unpack_from("<I", data, strings_offset)[0]
            if strings_size < 4 or strings_offset + strings_size > len(data):
                raise ValueError("invalid COFF string table extent")
        elif symbol_count:
            raise ValueError("COFF symbol entries lack a table pointer")

        index = 0
        while index < symbol_count:
            offset = symbol_offset + 18 * index
            name_bytes, _, section, _, storage_class, auxiliary_count = struct.unpack_from(
                "<8sIhHBB",
                data,
                offset,
            )
            index += 1 + auxiliary_count
            if index > symbol_count:
                raise ValueError("invalid COFF auxiliary symbol count")
            if storage_class not in {2, 105}:
                continue
            if name_bytes[:4] == b"\0\0\0\0":
                relative_name = struct.unpack_from("<I", name_bytes, 4)[0]
                symbol = _coff_string(data, strings_offset, strings_size, relative_name)
            else:
                symbol = name_bytes.split(b"\0", 1)[0].decode("ascii", errors="backslashreplace")
            if section == 0:
                undefined_external.add(symbol)
            elif section > 0:
                public_symbols.add(symbol)

        for index in range(count):
            offset = 20 + opt_size + index * 40
            name_bytes, _, _, raw_size, raw_offset, reloc_offset, _, relocs, _, attributes = (
                struct.unpack_from("<8sIIIIIIHHI", data, offset)
            )
            inline_name = name_bytes.split(b"\0", 1)[0]
            if inline_name.startswith(b"/") and inline_name[1:].isdigit():
                name = _coff_string(data, strings_offset, strings_size, int(inline_name[1:]))
            else:
                name = inline_name.decode("ascii", errors="backslashreplace")
            if raw_offset and raw_offset + raw_size > len(data):
                raise ValueError(f"invalid COFF data extent in {name}")
            if name == ".drectve" and raw_offset:
                directives.append(
                    data[raw_offset : raw_offset + min(raw_size, 8192)].decode(
                        "ascii", errors="backslashreplace"
                    )
                )
                if raw_size > 8192:
                    issues.append("COFF linker directive output truncated")
            if relocs == 0xFFFF and attributes & 0x01000000:
                if reloc_offset + 10 > len(data):
                    raise ValueError(f"invalid COFF extended relocation count in {name}")
                relocs = struct.unpack_from("<I", data, reloc_offset)[0]
                if not relocs:
                    raise ValueError(f"zero COFF extended relocation count in {name}")
                relocs -= 1
                reloc_offset += 10
            if relocs and (not reloc_offset or reloc_offset + relocs * 10 > len(data)):
                raise ValueError(f"invalid COFF relocation extent in {name}")
            types: Counter[str] = Counter()
            for entry in range(relocs):
                _, _, reloc_type = struct.unpack_from("<IIH", data, reloc_offset + entry * 10)
                types[f"0x{reloc_type:04x}"] += 1
            sections.append(
                {
                    "name": name,
                    "data_size": raw_size,
                    "characteristics": f"0x{attributes:08x}",
                    "relocations": relocs,
                    "relocation_type_codes": dict(sorted(types.items())),
                }
            )
    except ValueError as exc:
        issues.append(str(exc))

    details["undefined_external_symbols"] = sorted(undefined_external)
    details["public_symbols"] = sorted(public_symbols)
    details["linker_directives"] = directives
    return details, issues


def _measure_objdump(data: bytes) -> tuple[dict, list[str]]:
    stdout, stderr, error = _run_tool("objdump", ["-f", "-p", "-h", "-r"], data)
    issues = [message for message in (error, stderr.strip()[:4096]) if message]
    if len(stdout) > 65536:
        stdout = stdout[:65536]
        issues.append("objdump evidence truncated")
    return {"objdump_headers_and_relocations": stdout}, issues


def _measure_archive(data: bytes) -> tuple[dict, list[str]]:
    if data.startswith(b"!<thin>\n"):
        return {}, ["thin archive external member paths were not followed"]
    members = []
    offset = 8
    long_names = b""
    issues: list[str] = []
    try:
        while offset < len(data):
            header = data[offset : offset + 60]
            if len(header) != 60 or header[58:] != b"`\n":
                raise ValueError("invalid static archive member header")
            name = header[:16].rstrip()
            size = int(header[48:58].strip())
            start = offset + 60
            end = start + size
            offset = end + (size % 2)
            if size < 0 or offset > len(data):
                raise ValueError("invalid static archive member extent")
            if name == b"//":
                long_names = data[start:end]
                continue
            if name in {b"/", b"/SYM64/", b"__.SYMDEF", b"__.SYMDEF SORTED"}:
                continue
            if name.startswith(b"#1/"):
                name_end = start + int(name[3:])
                if not start < name_end <= end:
                    raise ValueError("invalid BSD archive member name extent")
                name, start = data[start:name_end].rstrip(b"\0"), name_end
            elif name.startswith(b"/"):
                name_start = int(name[1:])
                name_end = long_names.find(b"/\n", name_start)
                if not 0 <= name_start < name_end:
                    raise ValueError("invalid GNU archive member name offset")
                name = long_names[name_start:name_end]
            else:
                name = name.removesuffix(b"/")
            content = data[start:end]
            if content.startswith(b"\x7fELF"):
                member_format, measure = "ELF", _measure_elf
            elif _coff_magic(content):
                member_format, measure = "COFF", _measure_coff
            else:
                member_format, measure = "UNRECOGNIZED", None
            member_name = name.decode("utf-8", errors="backslashreplace")
            details, member_issues = (
                measure(content) if measure else ({}, ["unknown object format"])
            )
            members.append(
                {
                    "name": member_name,
                    "size": len(content),
                    "format": member_format,
                    "details": details,
                }
            )
            issues.extend(f"archive member {member_name}: {issue}" for issue in member_issues)
    except ValueError as exc:
        issues.append(str(exc))
    return {"archive_members": members}, issues


def _elf_roles(path: str, details: dict) -> list[str]:
    elf_type = details["header"].get("elf_type", "").split(" ", 1)[0]
    roles = []
    interpreter = details["interpreter"]
    pie = any("PIE" in flags.split() for flags in details["other_dynamic_tags"].get("FLAGS_1", []))
    # Some shared libraries have a standalone entry point and PT_INTERP.
    # Preserve their shared-library role when SONAME is present without PIE.
    if elf_type == "DYN" and not pie and (not interpreter or details["soname"]):
        if details["pyinit_exports"]:
            roles.append("PYTHON_EXTENSION")
        else:
            roles.append("SHARED_LIBRARY")
        if interpreter:
            roles.append("INTERPRETER_BEARING_SHARED_LIBRARY")
        if details["qt_plugin_exports"]:
            roles.append("QT_PLUGIN")
        elif path.startswith("PySide6/Qt/") and "/plugins/" in path:
            roles.append("QT_PLUGIN_PATH_CANDIDATE")
    elif elf_type == "EXEC" or (elf_type == "DYN" and (interpreter or pie)):
        roles.append("NATIVE_EXECUTABLE")
        if path.startswith("PySide6/") or "/libexec/" in path or "/bin/" in f"/{path}":
            roles.append("PACKAGED_TOOL_CANDIDATE")
    elif elf_type == "REL":
        roles.append("BUILD_OBJECT")
    else:
        roles.append("UNCLASSIFIED_ELF_ROLE")
    return roles


def _mach_magic(data: bytes, suffix: str) -> bool:
    if data[:4] in _MACH_MAGICS:
        return True
    # Java class files share the FAT_MAGIC value. Choose the Mach-O
    # interpretation only when the architecture table could fit.
    if data[:4] != b"\xca\xfe\xba\xbe" or len(data) < 8 or suffix == ".class":
        return False
    architectures = int.from_bytes(data[4:8], "big")
    return 0 < architectures <= 64 and 8 + architectures * 20 <= len(data)


def analyze_native(data: bytes, member_path: str, mode: int) -> dict | None:
    """Classify a member without loading it. Roles may be path-derived."""
    path = PurePosixPath(member_path)
    suffix = path.suffix.lower()
    executable = bool(mode & 0o111)
    soabi = _SOABI.search(path.name)
    native_suffix = bool(_SHARED_SUFFIX.search(path.name.lower()))
    details: dict[str, Any] = {}
    issues: list[str] = []

    if data.startswith(b"\x7fELF"):
        binary_format = "ELF"
        details, issues = _measure_elf(data)
        roles = _elf_roles(member_path, details)
    elif _coff_magic(data):
        binary_format = "COFF"
        details, issues = _measure_coff(data)
        roles = ["BUILD_OBJECT"]
    elif data.startswith(b"MZ"):
        header_at = int.from_bytes(data[60:64], "little") if len(data) >= 64 else 0
        binary_format = (
            "PE"
            if 64 <= header_at <= len(data) - 24 and data[header_at : header_at + 4] == b"PE\0\0"
            else "DOS_MZ_CANDIDATE"
        )
        details, issues = _measure_objdump(data)
        roles = ["FOREIGN_SHARED_LIBRARY" if suffix in {".dll", ".pyd"} else "FOREIGN_EXECUTABLE"]
    elif _mach_magic(data, suffix):
        binary_format = "MACH_O"
        details, issues = _measure_objdump(data)
        roles = ["FOREIGN_NATIVE_ARTIFACT"]
    elif data.startswith((b"!<arch>\n", b"!<thin>\n")):
        binary_format = "STATIC_ARCHIVE"
        details, issues = _measure_archive(data)
        roles = ["BUILD_ARCHIVE"]
    elif data.startswith(b"\0asm"):
        binary_format = "WEBASSEMBLY_CANDIDATE"
        details = {"version_bytes": data[4:8].hex()}
        roles = ["FOREIGN_EXECUTABLE"]
        issues = ["WebAssembly imports and relocations were not parsed"]
    elif data.startswith(b"#!"):
        binary_format = "SHEBANG_SCRIPT"
        first_line = data.partition(b"\n")[0]
        details = {
            "first_line_bytes_hex": first_line[:512].hex(),
            "first_line_size": len(first_line),
        }
        roles = ["SCRIPT"]
        if suffix == ".pyx":
            roles.append("CYTHON_SOURCE_CANDIDATE_BY_PATH")
    elif (
        executable
        or suffix in _BINARY_SUFFIXES
        or native_suffix
        or suffix in {".cmd", ".bat", ".ps1", ".sh", ".zsh", ".fish"}
    ):
        binary_format = "UNRECOGNIZED"
        roles = ["UNCLASSIFIED_EXECUTABLE_FORMAT"]
        initial = data[:16]
        details = {"prefix_hex": initial.hex()}
        source = _SCRIPT_NAMES.get(suffix)
        if source and b"\0" not in data:
            binary_format = "SOURCE_CANDIDATE_BY_FILENAME"
            roles = [source]
        elif suffix in {".icns", ".ico", ".jpg", ".png", ".txt"} or (
            path.parent.name.endswith(".dist-info")
            and path.name in {"WHEEL", "METADATA", "LICENSE", "RECORD"}
            and b"\0" not in data
        ):
            binary_format = "DATA_CANDIDATE_BY_FILENAME"
            roles = ["EXECUTABLE_MODE_ON_DATA_CANDIDATE"]
        else:
            issues = ["format detected only by executable mode or filename hint"]
    else:
        return None

    if issues:
        details["limitations"] = issues
    if binary_format == "ELF" and soabi and "PYTHON_EXTENSION" not in roles:
        roles.append("PYTHON_EXTENSION_FILENAME_CANDIDATE")
    return {
        "format": binary_format,
        "roles": roles,
        "execute_bits": executable,
        "native_suffix_candidate": suffix in {".so", ".pyd", ".dll", ".dylib"},
        "soabi_from_filename": soabi[1] if soabi else None,
        "status": "PARTIAL" if issues else "MEASURED",
        "details": details,
    }


def summarize_dependencies(
    native_members: list[dict[str, Any]],
    native_details: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Compare ELF library names with measured wheel members and SONAME values."""
    elf = [member for member in native_members if member["format"] == "ELF"]
    providers: dict[str, list[dict[str, Any]]] = defaultdict(list)
    consumers = []
    names: set[str] = set()
    interpreters = set()
    for member in elf:
        details = native_details[member["native_details"]]
        needed = details["dt_needed"]
        interpreter = details["interpreter"]
        if interpreter:
            interpreters.add(interpreter)
        identity = {
            key: member[key] for key in ("wheel", "member", "sha256", "projected", "native_details")
        }
        if needed:
            names.update(needed)
            consumers.append({"identity": identity, "needed": needed})
        elf_type = details["header"].get("elf_type", "").split(" ", 1)[0]
        if elf_type != "DYN" or "NATIVE_EXECUTABLE" in member["roles"]:
            continue
        basename = PurePosixPath(member["projected"] or member["member"]).name
        sonames = set(details["soname"])
        for name in {basename, *sonames}:
            providers[name].append(
                {
                    **identity,
                    "name_match": [
                        match
                        for match, matches in (
                            ("BASENAME", name == basename),
                            ("SONAME", name in sonames),
                        )
                        if matches
                    ],
                }
            )
    return {
        "scope": "ELF_DT_NEEDED_NAME_COMPARISON",
        "loader_resolution": "NOT_TESTED",
        "host_library_compatibility": "NOT_TESTED",
        "runtime_dynamic_loads": "NOT_TESTED",
        "program_interpreters": sorted(interpreters),
        "consumers": consumers,
        "bundled_candidates_by_needed_name": {
            name: providers[name] for name in sorted(names & providers.keys())
        },
        "unmatched_needed_names": sorted(names - providers.keys()),
    }


_CACHE_HEADER = re.compile(r"^(\d+) libs? found in cache ")
_CACHE_ENTRY = re.compile(r"^\s*(\S+)\s+\(([^()]*)\)\s+=>\s+(/.+)$")


def measure_host_loader_cache(needed_names: set[str]) -> dict[str, Any]:
    """List the Linux host's cached candidates without loading any libraries."""
    result: dict[str, Any] = {
        "scope": "HOST_LDCONFIG_CACHE_LISTING_ONLY",
        "loader_resolution": "NOT_TESTED",
        "abi_compatibility": "NOT_TESTED",
        "status": "NOT_TESTED",
        "entries_by_needed_name": {},
        "unlisted_needed_names": None,
    }
    if not sys.platform.startswith("linux"):
        return {**result, "reason": "Linux loader cache unavailable on this host"}
    if not needed_names:
        return {**result, "reason": "No DT_NEEDED names were supplied"}
    executable = next(
        (name for name in ("/usr/sbin/ldconfig", "/sbin/ldconfig") if shutil.which(name)),
        None,
    )
    if executable is None:
        return {**result, "reason": "ldconfig unavailable at the standard absolute paths"}
    stdout, stderr, error = _run_tool(executable, ["-p"])
    entries: dict[str, list[dict[str, str]]] = defaultdict(list)
    expected, decoded = None, 0
    for line in stdout.splitlines():
        if header := _CACHE_HEADER.match(line):
            expected = int(header[1])
        if entry := _CACHE_ENTRY.match(line):
            decoded += 1
            name, abi, path = entry.groups()
            if name in needed_names:
                entries[name].append({"path": path, "abi_as_listed": abi})
    issues = [message for message in (error, stderr.strip()[:4096]) if message]
    if expected != decoded:
        issues.append(f"loader cache entry counts differ ({decoded}/{expected})")
    return {
        **result,
        "tool": executable,
        "status": "PARTIAL" if issues else "MEASURED",
        "limitations": issues,
        "entries_by_needed_name": dict(sorted(entries.items())),
        "unlisted_needed_names": sorted(needed_names - entries.keys()) if not issues else None,
    }

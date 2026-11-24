import os
import struct
import sys

import pytest

from scripts import stage9_v5d7_native as native


_ELF_OUTPUT = """ELF Header:
  Class:                             ELF64
  Data:                              2's complement, little endian
  OS/ABI:                            UNIX - System V
  ABI Version:                       0
  Type:                              DYN (Shared object file)
  Machine:                           Advanced Micro Devices X86-64
  Entry point address:               0x0
  Flags:                             0x0

Dynamic section at offset 0x1230 contains 10 entries:
 0x0000000000000001 (NEEDED)        Shared library: [libc.so.6]
 0x0000000000000001 (NEEDED)        Shared library: [libstdc++.so.6]
 0x000000000000000e (SONAME)        Library soname: [module.so]
 0x000000000000000f (RPATH)         Library rpath: [$ORIGIN]
 0x000000000000001d (RUNPATH)       Library runpath: [$ORIGIN/../lib]
 0x000000000000000c (INIT)          0x2050
 0x000000006ffffffe (VERNEED)       0x628
 0x000000006ffffffc (VERDEF)        0x608

Relocation section '.rela.dyn' at offset 0x6a8 contains 2 entries:
    Offset             Info             Type               Symbol's Value  Symbol's Name + Addend
0000000000006dc8  0000000000000008 R_X86_64_RELATIVE                         4fa0
0000000000006fd8  0000001800000006 R_X86_64_GLOB_DAT      0000000000000000 _ITM_registerTMCloneTable + 0

Relocation section '.relr.dyn' at offset 0x723 contains 1 entry which relocate 2 locations:
Index: Entry            Address           Symbolic Address
0000:  00000000001e1ba0 00000000001e1ba0  .tdata
                        00000000001e1bb0  .tdata + 0x10

Symbol table '.dynsym' contains 6 entries:
   Num:    Value          Size Type    Bind   Vis      Ndx Name
     0: 0000000000000000     0 NOTYPE  LOCAL  DEFAULT  UND
     1: 0000000000000000     0 FUNC    GLOBAL DEFAULT  UND memcpy@GLIBC_2.14 (3)
     2: 0000000000001230    42 FUNC    GLOBAL DEFAULT    8 PyInit_mod@@CUSTOM_1.0
     3: 0000000000001270    37 FUNC    GLOBAL DEFAULT    8 qt_plugin_instance
     4: 0000000000001270    37 FUNC    GLOBAL HIDDEN     8 PyInit_hidden
     5: 0000000000001270    37 FUNC    WEAK   DEFAULT  UND PyInit_missing

Version definition section '.gnu.version_d' contains 1 entry:
  000000: Rev: 1  Flags: none  Index: 1  Cnt: 1  Name: CUSTOM_1.0

Version needs section '.gnu.version_r' contains 2 entries:
  000000: Version: 1  File: libc.so.6  Cnt: 3
  0x0010:   Name: GLIBC_2.14  Flags: none  Version: 3
  0x0020:   Name: GLIBC_2.7  Flags: none  Version: 4
  0x0030:   Name: GLIBC_PRIVATE  Flags: none  Version: 5
  0x0040: Version: 1  File: libstdc++.so.6  Cnt: 2
  0x0050:   Name: GLIBCXX_3.4.9  Flags: none  Version: 6
  0x0060:   Name: CXXABI_1.3.2  Flags: none  Version: 7

Displaying notes found in: .note.stapsdt
    Name: not_a_version_requirement
  GNU  0x00000010  NT_GNU_ABI_TAG (ABI version tag)
    OS: Linux, ABI: 3.2.0
"""


def test_elf_static_evidence(monkeypatch):
    warning = "readelf: Warning: Gap in build notes detected from 0x1010 to 0x2010"
    monkeypatch.setattr(native, "_run_tool", lambda *_: (_ELF_OUTPUT, warning, ""))
    result = native.analyze_native(b"\x7fELFfake", "mod.cpython-312-x86_64-linux-gnu.so", 0o100755)
    assert result["format"] == "ELF"
    assert result["status"] == "MEASURED"
    assert result["roles"] == ["PYTHON_EXTENSION", "QT_PLUGIN"]
    assert result["execute_bits"] is True
    assert result["native_suffix_candidate"] is True
    assert result["soabi_from_filename"] == "cpython-312-x86_64-linux-gnu"
    details = result["details"]
    assert details["header"]["elf_class"] == "ELF64"
    assert details["dt_needed"] == ["libc.so.6", "libstdc++.so.6"]
    assert details["soname"] == ["module.so"]
    assert details["rpath"] == ["$ORIGIN"]
    assert details["runpath"] == ["$ORIGIN/../lib"]
    assert details["other_dynamic_tags"]["INIT"] == ["0x2050"]
    assert details["version_definitions"] == ["CUSTOM_1.0"]
    assert details["imported_symbol_versions"] == {"GLIBC_2.14": 1}
    assert details["exported_symbol_versions"] == {"CUSTOM_1.0": 1}
    assert details["abi_version_requirements"]["GLIBC"] == {
        "max_numeric_required": "2.14",
        "required": ["GLIBC_2.14", "GLIBC_2.7", "GLIBC_PRIVATE"],
        "import_symbol_versions": ["GLIBC_2.14"],
    }
    assert details["abi_version_requirements"]["GLIBCXX"]["max_numeric_required"] == "3.4.9"
    assert details["abi_version_requirements"]["CXXABI"]["max_numeric_required"] == "1.3.2"
    assert details["relocations"][1]["types"] == {"RELR_RELATIVE": 2}
    assert details["pyinit_exports"] == ["PyInit_mod"]
    assert details["qt_plugin_exports"] == ["qt_plugin_instance"]
    assert details["build_note_warnings"] == [warning]


def test_elf_tool_failure_and_truncated_measurement(monkeypatch):
    output = _ELF_OUTPUT.replace("contains 6 entries:", "contains 7 entries:")
    output = output.replace("Cnt: 3", "Cnt: 4")
    output = output.replace("which relocate 2 locations:", "which relocate 3 locations:")
    monkeypatch.setattr(native, "_run_tool", lambda *_: (output, "readelf: Error: invalid", ""))
    result = native.analyze_native(b"\x7fELFfake", "mod.so", 0o100644)
    assert result["status"] == "PARTIAL"
    assert len(result["details"]["limitations"]) == 4

    monkeypatch.setattr(native, "_run_tool", lambda *_: ("", "", "readelf unavailable"))
    result = native.analyze_native(b"\x7fELFfake", "mod.abi3.so", 0o100644)
    assert result["status"] == "PARTIAL"
    assert result["roles"] == ["UNCLASSIFIED_ELF_ROLE", "PYTHON_EXTENSION_FILENAME_CANDIDATE"]
    assert "readelf unavailable" in result["details"]["limitations"]


@pytest.mark.parametrize(
    ("elf_type", "path", "extra", "roles"),
    [
        ("EXEC", "PySide6/assistant", "", ["NATIVE_EXECUTABLE", "PACKAGED_TOOL_CANDIDATE"]),
        ("DYN", "opaque", "[Requesting program interpreter: /loader]", ["NATIVE_EXECUTABLE"]),
        ("DYN", "PySide6/Qt/plugins/p.so", "", ["SHARED_LIBRARY", "QT_PLUGIN_PATH_CANDIDATE"]),
        ("DYN", "lib.so.2", "", ["SHARED_LIBRARY"]),
        ("REL", "input.o", "", ["BUILD_OBJECT"]),
    ],
)
def test_elf_roles(monkeypatch, elf_type, path, extra, roles):
    output = _ELF_OUTPUT.split("Dynamic section")[0].replace("DYN", elf_type) + extra
    monkeypatch.setattr(native, "_run_tool", lambda *_: (output, "", ""))
    assert native.analyze_native(b"\x7fELFfake", path, 0o644)["roles"] == roles


def test_shared_library_with_interpreter_keeps_shared_library_role(monkeypatch):
    output = (
        _ELF_OUTPUT.split("Relocation section")[0] + "[Requesting program interpreter: /loader]\n"
    )
    monkeypatch.setattr(native, "_run_tool", lambda *_: (output, "", ""))
    result = native.analyze_native(b"\x7fELFfake", "PySide6/Qt/lib/libQt6Core.so.6", 0o755)
    assert result["roles"] == ["SHARED_LIBRARY", "INTERPRETER_BEARING_SHARED_LIBRARY"]


def _coff():
    directives = b"/DEFAULTLIB:demo"
    raw_offset = 20 + 40
    relocation_offset = raw_offset + len(directives)
    symbol_offset = relocation_offset + 10
    header = struct.pack("<HHIIIHH", 0xAA64, 1, 0, symbol_offset, 2, 0, 0)
    section = struct.pack(
        "<8sIIIIIIHHI",
        b".drectve",
        0,
        0,
        len(directives),
        raw_offset,
        relocation_offset,
        0,
        1,
        0,
        0,
    )
    reloc = struct.pack("<IIH", 0, 0, 3)
    external = struct.pack("<8sIhHBB", b"external", 0, 0, 0, 2, 0)
    public = struct.pack("<8sIhHBB", struct.pack("<II", 0, 4), 0, 1, 0, 2, 0)
    names = struct.pack("<I", 17) + b"long_public\0\0"
    return header + section + directives + reloc + external + public + names


def test_coff_architecture_relocations_and_linker_symbols():
    data = _coff()
    result = native.analyze_native(data, "build.obj", 0o100644)
    assert result["format"] == "COFF"
    assert result["status"] == "MEASURED"
    assert result["roles"] == ["BUILD_OBJECT"]
    details = result["details"]
    assert details["machine"] == "ARM64"
    assert details["sections"][0]["relocation_type_codes"] == {"0x0003": 1}
    assert details["undefined_external_symbols"] == ["external"]
    assert details["public_symbols"] == ["long_public"]
    assert details["linker_directives"] == ["/DEFAULTLIB:demo"]

    truncated = native.analyze_native(data[:-1], "build.obj", 0o100644)
    assert truncated["status"] == "PARTIAL"
    assert truncated["details"]["limitations"] == ["invalid COFF string table extent"]


def _archive_member(name, data):
    header = f"{name:<16}{0:<12}{0:<6}{0:<6}{644:<8}{len(data):<10}`\n".encode()
    return header + data + (b"\n" if len(data) % 2 else b"")


def test_static_archives_and_external_members():
    ar = b"!<arch>\n" + _archive_member("//", b"quite_long_object_name.obj/\n")
    ar += _archive_member("/0", _coff())
    result = native.analyze_native(ar, "build.a", 0o644)
    assert result["format"] == "STATIC_ARCHIVE"
    assert result["status"] == "MEASURED"
    assert result["details"]["archive_members"][0]["name"] == "quite_long_object_name.obj"
    assert result["details"]["archive_members"][0]["format"] == "COFF"
    assert native.analyze_native(ar[:-1], "build.a", 0o644)["status"] == "PARTIAL"
    assert native.analyze_native(b"!<thin>\n", "build.a", 0o644)["status"] == "PARTIAL"


@pytest.mark.parametrize(
    ("data", "path", "mode", "binary_format", "status"),
    [
        (b"#!/usr/bin/env python3\n", "tests/extending.pyx", 0o644, "SHEBANG_SCRIPT", "MEASURED"),
        (b"@echo off\n", "build.cmd", 0o644, "SOURCE_CANDIDATE_BY_FILENAME", "MEASURED"),
        (b"code", "script.py", 0o755, "SOURCE_CANDIDATE_BY_FILENAME", "MEASURED"),
        (
            b"Metadata-Version: 2.5\n",
            "p.dist-info/METADATA",
            0o755,
            "DATA_CANDIDATE_BY_FILENAME",
            "MEASURED",
        ),
        (b"\0asm\x01\0\0\0", "module.wasm", 0o644, "WEBASSEMBLY_CANDIDATE", "PARTIAL"),
        (b"\0\1", "unknown", 0o700, "UNRECOGNIZED", "PARTIAL"),
        (b"MZ", "program.exe", 0o644, "DOS_MZ_CANDIDATE", "MEASURED"),
        (b"\xfe\xed\xfa\xcf", "library.dylib", 0o644, "MACH_O", "MEASURED"),
    ],
)
def test_static_executable_formats(monkeypatch, data, path, mode, binary_format, status):
    monkeypatch.setattr(native, "_run_tool", lambda *_: ("format evidence", "", ""))
    result = native.analyze_native(data, path, mode)
    assert result["format"] == binary_format
    assert result["status"] == status
    assert native.analyze_native(b"data", "plain.txt", 0o644) is None


def test_needed_name_candidates_do_not_imply_loader_resolution():
    def member(path, projected, key, roles):
        return {
            "format": "ELF",
            "member": path,
            "projected": projected,
            "wheel": "demo.whl",
            "sha256": key,
            "native_details": key,
            "roles": roles,
        }

    members = [
        member("uses.so", "runtime/uses.so", "A", ["PYTHON_EXTENSION"]),
        member("other.so", "runtime/libdemo.so", "B", ["SHARED_LIBRARY"]),
        member("another.so", None, "C", ["SHARED_LIBRARY"]),
        member("libmissing.so", None, "D", ["NATIVE_EXECUTABLE"]),
    ]
    details = {
        "A": {
            "header": {"elf_type": "DYN"},
            "dt_needed": ["libdemo.so", "libmissing.so"],
            "soname": [],
            "interpreter": None,
        },
        "B": {
            "header": {"elf_type": "DYN"},
            "dt_needed": [],
            "soname": ["libdemo.so"],
            "interpreter": None,
        },
        "C": {
            "header": {"elf_type": "DYN"},
            "dt_needed": [],
            "soname": ["libdemo.so"],
            "interpreter": None,
        },
        "D": {
            "header": {"elf_type": "DYN"},
            "dt_needed": [],
            "soname": [],
            "interpreter": "/loader",
        },
    }
    summary = native.summarize_dependencies(members, details)
    assert summary["program_interpreters"] == ["/loader"]
    assert summary["loader_resolution"] == "NOT_TESTED"
    assert summary["host_library_compatibility"] == "NOT_TESTED"
    assert summary["unmatched_needed_names"] == ["libmissing.so"]
    assert [
        candidate["name_match"]
        for candidate in summary["bundled_candidates_by_needed_name"]["libdemo.so"]
    ] == [
        ["BASENAME", "SONAME"],
        ["SONAME"],
    ]


def test_unavailable_static_tools(monkeypatch):
    monkeypatch.setattr(native.shutil, "which", lambda *args, **kwargs: None)
    result = native.native_tool_versions()
    assert set(result) == {"readelf", "objdump"}
    assert all(tool["version"] is None and tool["error"] for tool in result.values())


def test_host_loader_cache_lists_only_requested_names(monkeypatch):
    monkeypatch.setattr(native, "sys", type("Host", (), {"platform": "linux"}))
    monkeypatch.setattr(native.shutil, "which", lambda name: name)
    calls = []

    def run(tool, args):
        calls.append((tool, args))
        return (
            "2 libs found in cache /etc/ld.so.cache\n"
            "\tliba.so (libc6,x86-64, OS ABI: Linux 3.2.0) => /lib/liba.so\n"
            "\tlibb.so (libc6,x86-64) => /lib/libb.so\n",
            "",
            "",
        )

    monkeypatch.setattr(native, "_run_tool", run)
    result = native.measure_host_loader_cache({"liba.so", "libmissing.so"})
    assert calls == [("/usr/sbin/ldconfig", ["-p"])]
    assert result["status"] == "MEASURED"
    assert result["entries_by_needed_name"] == {
        "liba.so": [{"path": "/lib/liba.so", "abi_as_listed": "libc6,x86-64, OS ABI: Linux 3.2.0"}]
    }
    assert result["unlisted_needed_names"] == ["libmissing.so"]
    assert result["abi_compatibility"] == "NOT_TESTED"


def test_host_loader_cache_reports_gaps(monkeypatch):
    monkeypatch.setattr(native, "sys", type("Host", (), {"platform": "darwin"}))
    assert native.measure_host_loader_cache({"liba.so"})["status"] == "NOT_TESTED"
    monkeypatch.setattr(native, "sys", type("Host", (), {"platform": "linux"}))
    monkeypatch.setattr(native.shutil, "which", lambda _: None)
    assert native.measure_host_loader_cache({"liba.so"})["status"] == "NOT_TESTED"
    monkeypatch.setattr(native.shutil, "which", lambda name: name)
    monkeypatch.setattr(native, "_run_tool", lambda *_: ("", "cache unavailable", "failure"))
    result = native.measure_host_loader_cache({"liba.so"})
    assert result["status"] == "PARTIAL"
    assert result["unlisted_needed_names"] is None


@pytest.mark.skipif(os.name != "posix", reason="pipe capture uses POSIX selectors")
def test_static_tool_gets_private_non_executable_data(monkeypatch):
    monkeypatch.setattr(native.shutil, "which", lambda *args, **kwargs: sys.executable)
    source = (
        "from pathlib import Path; p=Path('member.bin');"
        "print(p.read_bytes().hex()); print(oct(p.stat().st_mode & 0o777))"
    )
    stdout, stderr, error = native._run_tool("trusted-tool", ["-c", source], b"\x7fELF")
    assert stdout.splitlines() == ["7f454c46", "0o600"]
    assert not stderr and not error
    monkeypatch.setattr(native, "_TOOL_OUTPUT_LIMIT", 1)
    assert "exceeded" in native._run_tool("trusted-tool", ["-c", "print('large')"])[2]

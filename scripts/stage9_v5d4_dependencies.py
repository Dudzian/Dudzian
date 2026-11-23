"""Propozycja offline wheel closure; odczyt danych, bez importów pakietów.

Obsługiwany profil: CPython 3.12, Linux x86_64, glibc 2.39, finalne
wersje numeryczne PEP 440 (także post). Inne wersje/markery odmawiają,
zamiast zgadywać. Narzędzie nie jest resolverem pobierającym pakiety:
wheelhouse musi zawierać jeden wybrany wheel na każdą potrzebną nazwę.
"""

from __future__ import annotations

import argparse
import base64
import csv
import email.parser
import hashlib
import io
import itertools
import json
import os
import re
import runpy
import stat
import sys
import tomllib
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any, cast

TARGET = {
    "implementation_name": "cpython",
    "implementation_version": "3.12.3",
    "os_name": "posix",
    "platform_machine": "x86_64",
    "platform_python_implementation": "CPython",
    "platform_system": "Linux",
    "python_full_version": "3.12.3",
    "python_version": "3.12",
    "sys_platform": "linux",
    "glibc": "2.39",
    "soabi": "cpython-312-x86_64-linux-gnu",
}
MAX_WHEEL = 512 * 1024 * 1024
MAX_EXPANDED = 2 * 1024 * 1024 * 1024
SHA256 = re.compile(r"[0-9a-f]{64}")
BYTECODE_POLICY_ID = "pyside6-essentials-6.10.3-cp311-cache-omission-v1"
# Wyłącznie zidentyfikowane bajty upstream. To kandydat wyjątku kontraktu,
# nie aprobata instalacji ani zaufanie do dowolnego cache CPython 3.11.
BYTECODE_POLICY: dict[str, Any] = {
    "filename": "pyside6_essentials-6.10.3-cp39-abi3-manylinux_2_34_x86_64.whl",
    "sha256": "5fb32b48010f513335a2166ac3df279c5996fb96402fd91a4875fc717fe55de0",
    "members": {
        "PySide6/support/__pycache__/__init__.cpython-311.pyc": {
            "sha256": "5e8a29507803d23847f5444596f792cbf95aa49ece2217cac2cd62489459a883",
            "source": "PySide6/support/__init__.py",
            "source_sha256": "544cd819bb55076c22ad952087d0dcef414dc64f6f09a4ac65ad1ee0216b5695",
            "header": "a70d0d0a00000000dd6ccb69df000000",
        },
        "PySide6/support/__pycache__/deprecated.cpython-311.pyc": {
            "sha256": "a5f978d1d2e82a5d08e4a1bbe3a098e486ce601ed18ac1da4ba359aac02d5a02",
            "source": "PySide6/support/deprecated.py",
            "source_sha256": "95431e9fc9d51387f56f2b1a75f38be22794a4addcd7c416f09b515ac0c35506",
            "header": "a70d0d0a00000000dd6ccb695d020000",
        },
    },
}


class Refusal(ValueError):
    """Niepełne, nieobsługiwane lub niespójne dane są odmową."""


def require(condition: bool, code: str) -> None:
    if not condition:
        raise Refusal(code)


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


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


def windows_read_regular(path: Path, limit: int) -> bytes:
    """Odczyt Win32 bez reparse i bez współdzielenia zapisu/usuwania.

    Wzorzec przypięcia rodziców pochodzi z windows_custody_filesystem,
    bez importu custody/pywin32 i bez ustanawiania zaufania do ACL.
    Używamy natywnej tożsamości; CRT fstat starszego Pythona jej nie gwarantuje.
    """
    import ctypes
    from ctypes import wintypes
    from contextlib import ExitStack

    loader = getattr(ctypes, "WinDLL", None)
    if os.name != "nt" or loader is None:
        raise Refusal("SECURE_FILE_READER_UNAVAILABLE")
    path = path.absolute()
    require(
        re.fullmatch(r"[A-Za-z]:", path.drive) is not None
        and "\x00" not in str(path)
        and all(":" not in part and not part.endswith((" ", ".")) for part in path.parts[1:]),
        "WINDOWS_INPUT_PATH",
    )
    before = path.lstat()
    require(
        stat.S_ISREG(before.st_mode) and before.st_nlink == 1,
        "NONREGULAR_INPUT",
    )
    require(0 <= before.st_size <= limit, "INPUT_TOO_LARGE")

    class Information(ctypes.Structure):
        _fields_ = [
            ("attributes", wintypes.DWORD),
            ("created", wintypes.FILETIME),
            ("accessed", wintypes.FILETIME),
            ("written", wintypes.FILETIME),
            ("volume", wintypes.DWORD),
            ("size_high", wintypes.DWORD),
            ("size_low", wintypes.DWORD),
            ("links", wintypes.DWORD),
            ("index_high", wintypes.DWORD),
            ("index_low", wintypes.DWORD),
        ]

    class BasicInformation(ctypes.Structure):
        _fields_ = [
            ("created", ctypes.c_longlong),
            ("accessed", ctypes.c_longlong),
            ("written", ctypes.c_longlong),
            ("changed", ctypes.c_longlong),
            ("attributes", wintypes.DWORD),
        ]

    kernel = loader("kernel32", use_last_error=True)
    kernel.CreateFileW.argtypes = (
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.LPVOID,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.HANDLE,
    )
    kernel.CreateFileW.restype = wintypes.HANDLE
    kernel.GetFileInformationByHandle.argtypes = (wintypes.HANDLE, ctypes.POINTER(Information))
    kernel.GetFileInformationByHandle.restype = wintypes.BOOL
    kernel.GetFileInformationByHandleEx.argtypes = (
        wintypes.HANDLE,
        ctypes.c_int,
        wintypes.LPVOID,
        wintypes.DWORD,
    )
    kernel.GetFileInformationByHandleEx.restype = wintypes.BOOL
    kernel.GetFileType.argtypes = (wintypes.HANDLE,)
    kernel.GetFileType.restype = wintypes.DWORD
    kernel.GetFinalPathNameByHandleW.argtypes = (
        wintypes.HANDLE,
        wintypes.LPWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
    )
    kernel.GetFinalPathNameByHandleW.restype = wintypes.DWORD
    kernel.ReadFile.argtypes = (
        wintypes.HANDLE,
        wintypes.LPVOID,
        wintypes.DWORD,
        ctypes.POINTER(wintypes.DWORD),
        wintypes.LPVOID,
    )
    kernel.ReadFile.restype = wintypes.BOOL
    kernel.CloseHandle.argtypes = (wintypes.HANDLE,)
    kernel.CloseHandle.restype = wintypes.BOOL

    def qualify(handle: Any, expected: Path, directory: bool) -> tuple[int, ...]:
        info = Information()
        basic = BasicInformation()
        require(
            bool(kernel.GetFileInformationByHandle(handle, ctypes.byref(info)))
            and bool(
                kernel.GetFileInformationByHandleEx(
                    handle, 0, ctypes.byref(basic), ctypes.sizeof(basic)
                )
            ),
            "WINDOWS_FILE_INFORMATION_UNAVAILABLE",
        )
        require(
            kernel.GetFileType(handle) == 1
            and not info.attributes & 0x400
            and bool(info.attributes & 0x10) == directory
            and (directory or info.links == 1),
            "WINDOWS_REPARSE_OR_SPECIAL_INPUT",
        )
        final = ctypes.create_unicode_buffer(32768)
        length = kernel.GetFinalPathNameByHandleW(handle, final, len(final), 0)
        require(0 < length < len(final), "WINDOWS_FINAL_PATH_UNAVAILABLE")
        actual = final.value
        require(
            actual.startswith("\\\\?\\") and not actual.startswith("\\\\?\\UNC\\"),
            "WINDOWS_INPUT_PATH",
        )
        require(
            os.path.normcase(actual[4:]) == os.path.normcase(str(expected)),
            "WINDOWS_INPUT_REDIRECTED",
        )
        # AccessTime może zmienić się podczas odczytu; pozostałe pola są przypięte.
        return (
            info.attributes,
            info.volume,
            info.index_high,
            info.index_low,
            info.size_high,
            info.size_low,
            info.links,
            basic.created,
            basic.written,
            basic.changed,
            basic.attributes,
        )

    with ExitStack() as stack:

        def open_handle(selected: Path, directory: bool) -> Any:
            handle = kernel.CreateFileW(
                str(selected),
                0x81 if directory else 0x80000000,
                3 if directory else 1,
                None,
                3,
                0x02200000 if directory else 0x00200000,
                None,
            )
            require(
                handle not in {None, ctypes.c_void_p(-1).value}, "WINDOWS_PROTECTED_OPEN_FAILED"
            )
            stack.callback(kernel.CloseHandle, handle)
            qualify(handle, selected, directory)
            return handle

        # LIST_DIRECTORY aktywuje zakaz rename/delete dla całego łańcucha.
        parents = [(parent, open_handle(parent, True)) for parent in reversed(path.parents)]
        handle = open_handle(path, False)
        pinned = qualify(handle, path, False)
        size = (pinned[4] << 32) | pinned[5]
        require(
            ((pinned[2] << 32) | pinned[3]) == before.st_ino and size == before.st_size,
            "INPUT_CHANGED",
        )
        chunks: list[bytes] = []
        remaining = size
        while remaining:
            requested = min(1024 * 1024, remaining)
            buffer = ctypes.create_string_buffer(requested)
            received = wintypes.DWORD()
            require(
                bool(kernel.ReadFile(handle, buffer, requested, ctypes.byref(received), None))
                and 0 < received.value <= requested,
                "WINDOWS_INPUT_READ_FAILED",
            )
            chunks.append(buffer.raw[: received.value])
            remaining -= received.value
        require(qualify(handle, path, False) == pinned, "INPUT_CHANGED")
        require(file_identity(path.lstat()) == file_identity(before), "INPUT_CHANGED")
        for parent, parent_handle in parents:
            qualify(parent_handle, parent, True)
        return b"".join(chunks)
    raise Refusal("WINDOWS_PROTECTED_READ_FAILED")


def read_regular(path: Path, limit: int = MAX_WHEEL) -> bytes:
    if os.name == "nt":
        return windows_read_regular(path, limit)
    require(
        os.name == "posix"
        and bool(getattr(os, "O_NOFOLLOW", None))
        and bool(getattr(os, "O_NONBLOCK", None)),
        "SECURE_FILE_READER_UNAVAILABLE",
    )
    require(not any(part.is_symlink() for part in (path, *path.parents)), "SYMLINK_INPUT")
    metadata = path.lstat()
    require(stat.S_ISREG(metadata.st_mode) and metadata.st_nlink == 1, "NONREGULAR_INPUT")
    require(metadata.st_size <= limit, "INPUT_TOO_LARGE")
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(descriptor, "rb") as stream:
        require(
            file_identity(os.fstat(stream.fileno())) == file_identity(metadata), "INPUT_CHANGED"
        )
        data = stream.read(limit + 1)
        require(
            file_identity(os.fstat(stream.fileno())) == file_identity(metadata), "INPUT_CHANGED"
        )
    require(
        file_identity(path.lstat()) == file_identity(metadata) and len(data) == metadata.st_size,
        "INPUT_CHANGED",
    )
    return data


def json_data(path: Path) -> Any:
    def pairs(values: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in values:
            require(key not in result, "DUPLICATE_JSON_KEY")
            result[key] = value
        return result

    return json.loads(read_regular(path, 16 * 1024 * 1024), object_pairs_hook=pairs)


def normalized(name: str) -> str:
    require(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", name) is not None, "PACKAGE_NAME")
    return re.sub(r"[-_.]+", "-", name).lower()


def version(value: str) -> tuple[int, ...]:
    match = re.fullmatch(r"(\d+(?:\.\d+)*)(?:\.post(\d+))?", value)
    require(match is not None, "UNSUPPORTED_VERSION")
    assert match is not None
    release = tuple(map(int, match[1].split(".")))
    require(len(release) <= 8, "UNSUPPORTED_VERSION")
    return (*release, *((0,) * (8 - len(release))), int(match[2]) if match[2] else -1)


def satisfies(actual: str, specification: str) -> bool:
    current = version(actual)
    for part in specification.strip().strip("()").split(","):
        if not part.strip():
            continue
        match = re.fullmatch(
            r"\s*(===|==|!=|>=|<=|~=|>|<)\s*(\d+(?:\.\d+)*(?:\.post\d+)?)(\.\*)?\s*", part
        )
        require(match is not None, "UNSUPPORTED_SPECIFIER")
        assert match is not None
        operation, expected, wildcard = match.groups()
        target = version(expected)
        if wildcard:
            require(operation in {"==", "!="}, "UNSUPPORTED_SPECIFIER")
            count = len(expected.split("."))
            equal = current[:count] == target[:count]
            result = equal if operation == "==" else not equal
        elif operation == "~=":
            pieces = expected.split(".")
            require(len(pieces) >= 2 and "post" not in expected, "UNSUPPORTED_SPECIFIER")
            upper = list(map(int, pieces[:-1]))
            upper[-1] += 1
            result = current >= target and current < version(".".join(map(str, upper)))
        else:
            result = {
                "==": current == target,
                "===": actual == expected,
                "!=": current != target,
                ">=": current >= target,
                "<=": current <= target,
                ">": current > target,
                "<": current < target,
            }[operation]
            # PEP 440: >V nie wybiera post-release tej samej release,
            # chyba że samo V jest post-release.
            if operation == ">" and current[:8] == target[:8] and ".post" not in expected:
                result = False
        if not result:
            return False
    return True


def marker(expression: str, environment: dict[str, str]) -> bool:
    """Parser małej gramatyki PEP 508; nigdy eval/exec/uruchamianie resolvera."""
    if not expression.strip():
        return True
    tokens: list[str] = []
    rest = expression
    pattern = re.compile(
        r"\s*(\(|\)|===|==|!=|<=|>=|~=|<|>|\bnot\b|\bin\b|\band\b|\bor\b|[A-Za-z_][A-Za-z_0-9]*|'[^'\\]*'|\"[^\"\\]*\")"
    )
    while rest.strip():
        match = pattern.match(rest)
        require(match is not None, "UNSUPPORTED_MARKER")
        assert match is not None
        tokens.append(match[1])
        rest = rest[match.end() :]
    position = 0

    def operand() -> tuple[str, str]:
        nonlocal position
        require(position < len(tokens), "MARKER_SYNTAX")
        token = tokens[position]
        position += 1
        if token.startswith(("'", '"')):
            return token[1:-1], "literal"
        require(token in environment, "UNKNOWN_MARKER_ENVIRONMENT")
        return environment[token], token

    def atom() -> bool:
        nonlocal position
        if position < len(tokens) and tokens[position] == "(":
            position += 1
            result = disjunction()
            require(position < len(tokens) and tokens[position] == ")", "MARKER_SYNTAX")
            position += 1
            return result
        left, left_name = operand()
        require(position < len(tokens), "MARKER_SYNTAX")
        operation = tokens[position]
        position += 1
        if operation == "not":
            require(position < len(tokens) and tokens[position] == "in", "MARKER_SYNTAX")
            position += 1
            operation = "not in"
        right, right_name = operand()
        if operation in {"in", "not in"}:
            return (left in right) if operation == "in" else (left not in right)
        require(operation in {"==", "!=", "===", ">=", "<=", ">", "<", "~="}, "MARKER_SYNTAX")
        if left_name in {
            "python_version",
            "python_full_version",
            "implementation_version",
        } or right_name in {"python_version", "python_full_version", "implementation_version"}:
            return satisfies(left, operation + right)
        require(operation in {"==", "!=", "==="}, "UNSUPPORTED_MARKER_COMPARISON")
        equal = (
            (normalized(left) if left else "") == (normalized(right) if right else "")
            if "extra" in {left_name, right_name}
            else left == right
        )
        return not equal if operation == "!=" else equal

    def conjunction() -> bool:
        nonlocal position
        result = atom()
        while position < len(tokens) and tokens[position] == "and":
            position += 1
            following = atom()
            result = result and following
        return result

    def disjunction() -> bool:
        nonlocal position
        result = conjunction()
        while position < len(tokens) and tokens[position] == "or":
            position += 1
            following = conjunction()
            result = result or following
        return result

    result = disjunction()
    require(position == len(tokens), "MARKER_SYNTAX")
    return result


def requirement(raw: str) -> tuple[str, set[str], str, str]:
    parts = raw.split(";", 1)
    match = re.fullmatch(
        r"\s*([A-Za-z0-9][A-Za-z0-9._-]*)(?:\[([A-Za-z0-9_,. -]+)\])?\s*(.*?)\s*", parts[0]
    )
    require(match is not None and "@" not in parts[0], "UNSUPPORTED_REQUIREMENT")
    assert match is not None
    extras = {normalized(item.strip()) for item in (match[2] or "").split(",") if item.strip()}
    return normalized(match[1]), extras, match[3], parts[1].strip() if len(parts) == 2 else ""


def compatible_tag(tag: str) -> bool:
    pieces = tag.split("-")
    require(len(pieces) == 3, "WHEEL_TAG")
    interpreter, abi, platform = pieces
    if platform == "any":
        return interpreter in {"py3", "py312", "cp312"} and abi == "none"
    if not (
        (interpreter == "cp312" and abi in {"cp312", "none"})
        or (interpreter in {"py3", "py312"} and abi == "none")
    ):
        old = re.fullmatch(r"cp3(\d+)", interpreter)
        if not (old and 2 <= int(old[1]) <= 12 and abi == "abi3"):
            return False
    if platform in {"manylinux1_x86_64", "manylinux2010_x86_64", "manylinux2014_x86_64"}:
        return True
    match = re.fullmatch(r"manylinux_(\d+)_(\d+)_x86_64", platform)
    return match is not None and int(match[1]) == 2 and 5 <= int(match[2]) <= 39


def member_path(value: str) -> str:
    relative = PurePosixPath(value)
    require(
        bool(relative.parts)
        and not relative.is_absolute()
        and relative.as_posix() == value
        and ".." not in relative.parts
        and "\\" not in value
        and ":" not in value
        and all(ord(char) >= 32 and ord(char) != 127 for char in value),
        "WHEEL_MEMBER_PATH",
    )
    return value


def materialized_path(member: str, dist_info: str) -> str:
    parts = PurePosixPath(member_path(member)).parts
    if parts[0].endswith(".data"):
        require(
            parts[0] == dist_info.removesuffix(".dist-info") + ".data"
            and len(parts) > 2
            and parts[1] in {"purelib", "platlib"},
            "UNSUPPORTED_WHEEL_DATA",
        )
        return member_path("/".join(parts[2:]))
    require(
        not any(part.endswith(".dist-info") and part != dist_info for part in parts),
        "WHEEL_DIST_INFO_MISMATCH",
    )
    return member


def check_startup(member: str) -> None:
    path = PurePosixPath(member)
    require(
        path.suffix not in {".pth", ".pyc", ".pyo"}
        and not any(
            part in {"sitecustomize", "usercustomize"}
            or part.startswith(("sitecustomize.", "usercustomize."))
            for part in path.parts
        )
        and "__pycache__" not in path.parts,
        "WHEEL_STARTUP_CODE",
    )


def validate_materialization(packages: list[dict[str, Any]]) -> None:
    """Nawet identyczne współdzielone bajty wymagają odrębnej analizy kolizji."""
    seen: set[str] = set()
    directories: set[str] = set()
    for package in packages:
        for entry in package["materialized_members"].values():
            path = PurePosixPath(member_path(entry["path"]))
            parents = {parent.as_posix() for parent in path.parents if parent.parts}
            require(
                path.as_posix() not in seen | directories and not parents & seen,
                "WHEEL_MATERIALIZATION_COLLISION:" + path.as_posix(),
            )
            seen.add(path.as_posix())
            directories.update(parents)


def shared_tools() -> dict[str, Any]:
    """Trusted adjacent code, never wheel or candidate-provided Python."""
    return runpy.run_path(str(Path(__file__).with_name("stage9_v5d5_shared.py")))


def shared_inventory(packages: list[dict[str, Any]]) -> dict[str, Any]:
    try:
        return cast(dict[str, Any], shared_tools()["collision_inventory"](packages))
    except ValueError as exc:
        raise Refusal(str(exc)) from exc


def inspect_wheel(path: Path, origin: Any) -> dict[str, Any]:
    return inspect_wheel_bytes(read_regular(path), path.name, origin)


def inspect_wheel_bytes(data: bytes, filename: str, origin: Any) -> dict[str, Any]:
    """Wspólny kontrakt inspektora i buildera; nigdy import/marshal badanego kodu."""
    require(len(data) <= MAX_WHEEL, "INPUT_TOO_LARGE")
    path = Path(filename)
    require(path.name == filename, "WHEEL_FILENAME")
    require(
        isinstance(origin, dict) and set(origin) == {"url", "source", "sha256"}, "ORIGIN_MISSING"
    )
    require(all(isinstance(value, str) and value for value in origin.values()), "ORIGIN_SCHEMA")
    require(
        origin["url"].startswith("https://") and digest(data) == origin["sha256"], "ORIGIN_SHA256"
    )
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        members = archive.infolist()
        names = [item.filename for item in members]
        require(len(names) <= 100000 and len(set(names)) == len(names), "WHEEL_MEMBER_SET")
        require(sum(item.file_size for item in members) <= MAX_EXPANDED, "WHEEL_EXPANSION_LIMIT")
        for item in members:
            require(
                item.orig_filename == item.filename,
                "WHEEL_MEMBER_PATH",
            )
            member_path(item.filename[:-1] if item.is_dir() else item.filename)
            mode = item.external_attr >> 16
            require(stat.S_IFMT(mode) in {0, stat.S_IFREG, stat.S_IFDIR}, "WHEEL_MEMBER_SPECIAL")
            require(
                stat.S_IFMT(mode) != (stat.S_IFREG if item.is_dir() else stat.S_IFDIR),
                "WHEEL_MEMBER_SPECIAL",
            )
            require(
                not item.flag_bits & 1
                and item.compress_type in {zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED},
                "WHEEL_ZIP_PROFILE",
            )
            if item.is_dir():
                require(archive.read(item) == b"", "WHEEL_DIRECTORY_CONTENT")
        metadata_files = [name for name in names if name.endswith(".dist-info/METADATA")]
        wheel_files = [name for name in names if name.endswith(".dist-info/WHEEL")]
        require(len(metadata_files) == len(wheel_files) == 1, "WHEEL_METADATA")
        require(
            metadata_files[0].rsplit("/", 1)[0] == wheel_files[0].rsplit("/", 1)[0],
            "WHEEL_METADATA",
        )
        parser = email.parser.BytesParser()
        metadata = parser.parsebytes(archive.read(metadata_files[0]))
        wheel = parser.parsebytes(archive.read(wheel_files[0]))
        require(not metadata.defects and not wheel.defects, "WHEEL_METADATA")
        for header in ("Metadata-Version", "Name", "Version"):
            require(len(metadata.get_all(header, [])) == 1, "WHEEL_METADATA_DUPLICATE_OR_MISSING")
        require(
            metadata["Metadata-Version"] in {"2.1", "2.2", "2.3", "2.4"},
            "WHEEL_METADATA_VERSION_UNSUPPORTED",
        )
        require(
            len(metadata.get_all("Requires-Python", [])) <= 1, "WHEEL_METADATA_DUPLICATE_OR_MISSING"
        )
        for header in ("Wheel-Version", "Root-Is-Purelib"):
            require(len(wheel.get_all(header, [])) == 1, "WHEEL_HEADER_DUPLICATE_OR_MISSING")
        require(
            wheel["Wheel-Version"] == "1.0" and wheel["Root-Is-Purelib"] in {"true", "false"},
            "WHEEL_FORMAT_UNSUPPORTED",
        )
        name, selected = metadata.get("Name", ""), metadata.get("Version", "")
        version(selected)
        tags = sorted(set(wheel.get_all("Tag", [])))
        require(bool(tags) and any(compatible_tag(tag) for tag in tags), "WHEEL_ABI_MISMATCH")
        filename_parts = path.name.removesuffix(".whl").split("-")
        require(path.name.endswith(".whl") and len(filename_parts) in {5, 6}, "WHEEL_FILENAME")
        require(
            normalized(filename_parts[0]) == normalized(name)
            and version(filename_parts[1]) == version(selected),
            "WHEEL_FILENAME_METADATA_MISMATCH",
        )
        if len(filename_parts) == 6:
            require(
                re.fullmatch(r"[0-9][A-Za-z0-9_]*", filename_parts[2]) is not None,
                "WHEEL_BUILD_TAG",
            )
        filename_tags = {
            "-".join(parts)
            for parts in itertools.product(*(part.split(".") for part in filename_parts[-3:]))
        }
        require(set(tags) == filename_tags, "WHEEL_FILENAME_TAG_MISMATCH")
        dist_info = metadata_files[0].rsplit("/", 1)[0]
        dist_identity = dist_info.removesuffix(".dist-info").rsplit("-", 1)
        require(
            "/" not in dist_info
            and len(dist_identity) == 2
            and normalized(dist_identity[0]) == normalized(filename_parts[0])
            and version(dist_identity[1]) == version(filename_parts[1]),
            "WHEEL_DIST_INFO_MISMATCH",
        )
        record_name = dist_info + "/RECORD"
        require(record_name in names, "WHEEL_RECORD_MISSING")
        records: dict[str, tuple[str, str]] = {}
        for row in csv.reader(io.StringIO(archive.read(record_name).decode("utf-8"))):
            require(len(row) == 3 and row[0] not in records, "WHEEL_RECORD_SCHEMA")
            records[row[0]] = row[1], row[2]
        files = {item.filename for item in members if not item.is_dir()}
        require(set(records) == files, "WHEEL_RECORD_INCOMPLETE")
        for member, (member_hash, size) in records.items():
            if member == record_name:
                require(member_hash == size == "", "WHEEL_RECORD_SELF_HASH")
                continue
            content = archive.read(member)
            expected = (
                "sha256="
                + base64.urlsafe_b64encode(hashlib.sha256(content).digest()).rstrip(b"=").decode()
            )
            require(member_hash == expected and size == str(len(content)), "WHEEL_RECORD_INTEGRITY")
        directories = {
            parent.as_posix()
            for item in members
            for parent in PurePosixPath(item.filename).parents
            if parent.parts
        } | {item.filename[:-1] for item in members if item.is_dir()}
        require(not directories & files, "WHEEL_MEMBER_FILE_DIRECTORY_COLLISION")
        for item in members:
            if item.is_dir():
                check_startup(item.filename[:-1])
        requires_python = metadata.get("Requires-Python", "")
        require(
            satisfies(TARGET["python_full_version"], requires_python), "PYTHON_VERSION_MISMATCH"
        )
        # Politykę startup stosujemy dopiero po pełnym RECORD, łącznie z plikami,
        # które zostaną odrzucone lub pominięte. ZIP sprawdza CRC przy odczycie.
        bytecode = {member for member in files if member.endswith(".pyc")}
        omitted: list[dict[str, str]] = []
        if bytecode:
            require(
                normalized(name) == "pyside6-essentials"
                and selected == "6.10.3"
                and filename == BYTECODE_POLICY["filename"]
                and digest(data) == BYTECODE_POLICY["sha256"]
                and bytecode == set(BYTECODE_POLICY["members"]),
                "WHEEL_STARTUP_CODE",
            )
            for member in sorted(bytecode):
                pin = BYTECODE_POLICY["members"][member]
                content = archive.read(member)
                require(pin["source"] in files, "WHEEL_BYTECODE_SOURCE_MISSING")
                source = archive.read(pin["source"])
                require(
                    digest(content) == pin["sha256"]
                    and digest(source) == pin["source_sha256"]
                    and len(content) > 16
                    and content[:16].hex() == pin["header"]
                    and content[:8] == bytes.fromhex("a70d0d0a00000000")
                    and int.from_bytes(content[12:16], "little") == len(source)
                    and TARGET["python_version"] == "3.12",
                    "WHEEL_BYTECODE_PIN_MISMATCH",
                )
                omitted.append(
                    {
                        "distribution": normalized(name),
                        "version": selected,
                        "wheel_filename": filename,
                        "wheel_sha256": digest(data),
                        "member": member,
                        "sha256": digest(content),
                        "source": pin["source"],
                        "source_sha256": digest(source),
                        "cache_tag": "cpython-311",
                        "magic": "a70d0d0a",
                        "policy": BYTECODE_POLICY_ID,
                    }
                )
        materialized: dict[str, dict[str, str]] = {}
        for member in sorted(files - bytecode):
            destination = materialized_path(member, dist_info)
            check_startup(member)
            check_startup(destination)
            materialized[member] = {"path": destination, "sha256": digest(archive.read(member))}
        validate_materialization([{"materialized_members": materialized}])
        native: dict[str, str] = {}
        for item in members:
            if item.is_dir():
                continue
            with archive.open(item) as stream:
                prefix = stream.read(4)
            # Profil Linux nie uznaje obcych formatów ani skryptów instalacji
            # za zwykłe dane. Źródła .py są jawnie związane pełnym manifestem.
            suffix = PurePosixPath(item.filename).suffix.lower()
            require(
                not prefix.startswith(b"MZ")
                and prefix
                not in {
                    b"\xfe\xed\xfa\xce",
                    b"\xce\xfa\xed\xfe",
                    b"\xfe\xed\xfa\xcf",
                    b"\xcf\xfa\xed\xfe",
                    b"\xca\xfe\xba\xbe",
                    b"\xbe\xba\xfe\xca",
                    b"\xca\xfe\xba\xbf",
                    b"\xbf\xba\xfe\xca",
                }
                and suffix not in {".exe", ".com", ".bat", ".cmd", ".ps1", ".sh"}
                and (not prefix.startswith(b"#!") or suffix == ".py"),
                "WHEEL_EXECUTABLE_FORMAT_UNSUPPORTED",
            )
            if (
                prefix == b"\x7fELF"
                or item.filename.endswith((".so", ".pyd", ".dll", ".dylib"))
                or ".so." in item.filename
            ):
                tagged = re.search(r"\.cpython-([A-Za-z0-9_-]+)\.so$", item.filename)
                require(
                    tagged is None or tagged[1] == "312-x86_64-linux-gnu",
                    "NATIVE_MEMBER_SOABI_MISMATCH",
                )
                native[item.filename] = digest(archive.read(item))
        require(
            not any(".data/" in member for member in native),
            "NATIVE_DATA_RELOCATION_REQUIRES_REVIEW",
        )
        return {
            "name": normalized(name),
            "version": selected,
            "filename": path.name,
            "sha256": digest(data),
            "origin": origin,
            "tags": tags,
            "requires_python": requires_python,
            "requires_dist": sorted(metadata.get_all("Requires-Dist", [])),
            "provides_extra": sorted(
                normalized(value) for value in metadata.get_all("Provides-Extra", [])
            ),
            "native_members": native,
            "omitted_members": omitted,
            "materialized_members": materialized,
            "member_metadata": {
                item.filename: {
                    "mode": item.external_attr >> 16,
                    "type": "regular",
                    "size": item.file_size,
                }
                for item in members
                if not item.is_dir()
            },
        }


def propose(
    pyproject: Path,
    wheelhouse: Path,
    origins: dict[str, Any],
    allow_shared_candidates: bool = False,
) -> dict[str, Any]:
    raw = read_regular(pyproject, 8 * 1024 * 1024)
    project = tomllib.loads(raw.decode())["project"]
    require(
        satisfies(TARGET["python_full_version"], project["requires-python"]),
        "PROJECT_PYTHON_VERSION",
    )
    roots = list(project["dependencies"]) + list(
        project["optional-dependencies"]["observer-kvm-preflight"]
    )
    require(all(isinstance(value, str) for value in roots), "PROJECT_REQUIREMENTS")
    require(not wheelhouse.is_symlink() and wheelhouse.is_dir(), "WHEELHOUSE_DIRECTORY")
    candidates: dict[str, dict[str, Any]] = {}
    for path in sorted(wheelhouse.iterdir()):
        require(path.suffix == ".whl", "NON_WHEEL_DISTRIBUTION")
        package = inspect_wheel(path, origins.get(path.name))
        require(package["name"] not in candidates, "AMBIGUOUS_PACKAGE_VERSION")
        candidates[package["name"]] = package
    require(
        set(origins) == {package["filename"] for package in candidates.values()}, "ORIGIN_INVENTORY"
    )
    selected: dict[str, dict[str, Any]] = {}
    enabled: dict[str, set[str]] = {}
    pending = [(raw_requirement, {""}, "project") for raw_requirement in roots]
    edges: set[tuple[str, str, str]] = set()
    while pending:
        raw_requirement, parent_extras, parent = pending.pop()
        name, extras, specifier, expression = requirement(raw_requirement)
        if not any(marker(expression, {**TARGET, "extra": extra}) for extra in parent_extras):
            continue
        require(name in candidates, "MISSING_DEPENDENCY:" + name)
        package = candidates[name]
        require(satisfies(package["version"], specifier), "DEPENDENCY_VERSION:" + name)
        require(extras <= set(package["provides_extra"]), "MISSING_PACKAGE_EXTRA:" + name)
        edges.add((parent, name, raw_requirement))
        next_extras = enabled.get(name, set()) | extras | {""}
        if name in selected and next_extras == enabled[name]:
            continue
        enabled[name] = next_extras
        selected[name] = package
        pending.extend((item, next_extras, name) for item in package["requires_dist"])
    require(set(selected) == set(candidates), "UNREACHABLE_WHEEL")
    packages = [selected[name] for name in sorted(selected)]
    if allow_shared_candidates:
        sharing = shared_inventory(packages)
    else:
        validate_materialization(packages)
    result = {
        "profile": (
            "stage9-v5d5-dependency-proposal-v1"
            if allow_shared_candidates
            else "stage9-v5d4-dependency-proposal-v1"
        ),
        "status": "PROPOSAL_REQUIRES_INDEPENDENT_APPROVAL",
        "target": TARGET,
        "pyproject_sha256": digest(raw),
        "root_requirements": roots,
        "packages": packages,
        "edges": [list(item) for item in sorted(edges)],
        "system_binding": "SEPARATE_REVIEW_REQUIRED:python3-libvirt/libvirtmod/ELF",
        "bytecode_policy": BYTECODE_POLICY_ID,
        "installation_contract_exception": (
            "OMIT_PINNED_FOREIGN_BYTECODE_REQUIRES_INDEPENDENT_APPROVAL"
            if any(package["omitted_members"] for package in selected.values())
            else "NONE"
        ),
        "authorization": "NOT_AUTHORIZED",
    }
    if allow_shared_candidates:
        result["shared_wheel_members"] = sharing
    return result


def omission_expectation(proposal: dict[str, Any]) -> dict[str, Any]:
    return {
        "policy": BYTECODE_POLICY_ID,
        "installation_contract_exception": "OMIT_PINNED_FOREIGN_BYTECODE",
        "members": sorted(
            (member for package in proposal["packages"] for member in package["omitted_members"]),
            key=lambda item: (item["distribution"], item["wheel_filename"], item["member"]),
        ),
    }


def verify(
    pyproject: Path,
    wheelhouse: Path,
    origins: dict[str, Any],
    proposal: dict[str, Any],
    expected_sha256: str,
    approved_native: dict[str, str],
    approved_omissions: dict[str, Any] | None = None,
    approved_shared: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Oczekiwany digest i native inventory muszą przyjść od odrębnego recenzenta.

    Nie ustanawia tożsamości recenzenta ani podpisu operatora. Hash odczytany
    z tej samej propozycji nie jest niezależną akceptacją.
    """
    require(
        isinstance(expected_sha256, str) and SHA256.fullmatch(expected_sha256) is not None,
        "INDEPENDENT_EXPECTATION_MISSING",
    )
    require(digest(canonical(proposal)) == expected_sha256, "PROPOSAL_SHA256_MISMATCH")
    uses_shared_contract = proposal.get("profile") == "stage9-v5d5-dependency-proposal-v1"
    rebuilt = propose(pyproject, wheelhouse, origins, allow_shared_candidates=uses_shared_contract)
    require(canonical(proposal) == canonical(rebuilt), "PROPOSAL_CONTENT_MISMATCH")
    native = {
        package["filename"] + ":" + member: sha
        for package in rebuilt["packages"]
        for member, sha in package["native_members"].items()
    }
    require(native == approved_native, "UNAPPROVED_NATIVE_LIBRARY")
    omissions = omission_expectation(rebuilt)
    if omissions["members"] or approved_omissions is not None:
        require(approved_omissions == omissions, "UNAPPROVED_WHEEL_OMISSIONS")
    sharing: dict[str, Any] | None = None
    if uses_shared_contract or approved_shared is not None:
        try:
            sharing = shared_tools()["verify_shared"](rebuilt["packages"], approved_shared)
        except ValueError as exc:
            raise Refusal(str(exc)) from exc
    result = {
        "profile": (
            "stage9-v5d5-dependency-verification-v1"
            if uses_shared_contract
            else "stage9-v5d4-dependency-verification-v1"
        ),
        "status": "LOCAL_CONSISTENCY_VERIFIED",
        "proposal_sha256": expected_sha256,
        "packages": len(rebuilt["packages"]),
        "native_members": len(native),
        "omitted_members": omissions["members"],
        "installation_contract_exception": (
            omissions["installation_contract_exception"] if omissions["members"] else "NONE"
        ),
        "authority": "CALLER_SUPPLIED_EXPECTATIONS_NOT_AUTHENTICATED",
        "authorization": "NOT_AUTHORIZED",
    }
    if sharing is not None:
        result.update(
            {
                "shared_wheel_members": sharing,
                "shared_wheel_expectations": approved_shared,
                "shared_wheel_policy": sharing["policy"],
            }
        )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("propose", "verify"))
    parser.add_argument("--pyproject", type=Path, required=True)
    parser.add_argument("--wheelhouse", type=Path, required=True)
    parser.add_argument("--origins", type=Path, required=True)
    parser.add_argument("--proposal", type=Path)
    parser.add_argument("--expected-sha256")
    parser.add_argument("--approved-native", type=Path)
    parser.add_argument("--approved-omissions", type=Path)
    parser.add_argument("--approved-shared", type=Path)
    parser.add_argument("--allow-shared-candidates", action="store_true")
    args = parser.parse_args()
    try:
        origins = json_data(args.origins)
        require(type(origins) is dict, "ORIGIN_SCHEMA")
        if args.mode == "propose":
            result = propose(
                args.pyproject,
                args.wheelhouse,
                origins,
                allow_shared_candidates=args.allow_shared_candidates,
            )
        else:
            require(
                args.proposal is not None and args.approved_native is not None,
                "INDEPENDENT_EXPECTATION_MISSING",
            )
            result = verify(
                args.pyproject,
                args.wheelhouse,
                origins,
                json_data(args.proposal),
                args.expected_sha256,
                json_data(args.approved_native),
                json_data(args.approved_omissions) if args.approved_omissions else None,
                json_data(args.approved_shared) if args.approved_shared else None,
            )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except (Refusal, OSError, ValueError, KeyError, TypeError, zipfile.BadZipFile) as exc:
        code = str(exc) if isinstance(exc, Refusal) else "DEPENDENCY_INPUT_REJECTED"
        print(json.dumps({"status": "REFUSED", "reason": code, "authorization": "NOT_AUTHORIZED"}))
        return 2


if __name__ == "__main__":
    sys.exit(main())

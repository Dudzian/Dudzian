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
import stat
import sys
import tomllib
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

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


def read_regular(path: Path, limit: int = MAX_WHEEL) -> bytes:
    require(not any(part.is_symlink() for part in (path, *path.parents)), "SYMLINK_INPUT")
    metadata = path.lstat()
    require(stat.S_ISREG(metadata.st_mode) and metadata.st_nlink == 1, "NONREGULAR_INPUT")
    require(metadata.st_size <= limit, "INPUT_TOO_LARGE")
    descriptor = os.open(
        path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    )
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


def inspect_wheel(path: Path, origin: Any) -> dict[str, Any]:
    data = read_regular(path)
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
            relative = PurePosixPath(item.filename)
            require(
                not relative.is_absolute()
                and ".." not in relative.parts
                and "\\" not in item.filename
                and all(
                    part not in {"", ".", ".."} for part in item.filename.rstrip("/").split("/")
                ),
                "WHEEL_MEMBER_PATH",
            )
            mode = item.external_attr >> 16
            require(stat.S_IFMT(mode) in {0, stat.S_IFREG, stat.S_IFDIR}, "WHEEL_MEMBER_SPECIAL")
            require(not item.filename.endswith((".pth", ".pyc")), "WHEEL_STARTUP_CODE")
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
        for header in ("Name", "Version"):
            require(len(metadata.get_all(header, [])) == 1, "WHEEL_METADATA_DUPLICATE_OR_MISSING")
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
        requires_python = metadata.get("Requires-Python", "")
        require(
            satisfies(TARGET["python_full_version"], requires_python), "PYTHON_VERSION_MISMATCH"
        )
        native: dict[str, str] = {}
        for item in members:
            if item.is_dir():
                continue
            with archive.open(item) as stream:
                prefix = stream.read(4)
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
        }


def propose(pyproject: Path, wheelhouse: Path, origins: dict[str, Any]) -> dict[str, Any]:
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
    return {
        "profile": "stage9-v5d4-dependency-proposal-v1",
        "status": "PROPOSAL_REQUIRES_INDEPENDENT_APPROVAL",
        "target": TARGET,
        "pyproject_sha256": digest(raw),
        "root_requirements": roots,
        "packages": [selected[name] for name in sorted(selected)],
        "edges": [list(item) for item in sorted(edges)],
        "system_binding": "SEPARATE_REVIEW_REQUIRED:python3-libvirt/libvirtmod/ELF",
        "authorization": "NOT_AUTHORIZED",
    }


def verify(
    pyproject: Path,
    wheelhouse: Path,
    origins: dict[str, Any],
    proposal: dict[str, Any],
    expected_sha256: str,
    approved_native: dict[str, str],
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
    rebuilt = propose(pyproject, wheelhouse, origins)
    require(canonical(proposal) == canonical(rebuilt), "PROPOSAL_CONTENT_MISMATCH")
    native = {
        package["filename"] + ":" + member: sha
        for package in rebuilt["packages"]
        for member, sha in package["native_members"].items()
    }
    require(native == approved_native, "UNAPPROVED_NATIVE_LIBRARY")
    return {
        "profile": "stage9-v5d4-dependency-verification-v1",
        "status": "LOCAL_CONSISTENCY_VERIFIED",
        "proposal_sha256": expected_sha256,
        "packages": len(rebuilt["packages"]),
        "native_members": len(native),
        "authority": "CALLER_SUPPLIED_EXPECTATIONS_NOT_AUTHENTICATED",
        "authorization": "NOT_AUTHORIZED",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("propose", "verify"))
    parser.add_argument("--pyproject", type=Path, required=True)
    parser.add_argument("--wheelhouse", type=Path, required=True)
    parser.add_argument("--origins", type=Path, required=True)
    parser.add_argument("--proposal", type=Path)
    parser.add_argument("--expected-sha256")
    parser.add_argument("--approved-native", type=Path)
    args = parser.parse_args()
    try:
        origins = json_data(args.origins)
        require(type(origins) is dict, "ORIGIN_SCHEMA")
        if args.mode == "propose":
            result = propose(args.pyproject, args.wheelhouse, origins)
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
            )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except (Refusal, OSError, ValueError, KeyError, TypeError, zipfile.BadZipFile) as exc:
        code = str(exc) if isinstance(exc, Refusal) else "DEPENDENCY_INPUT_REJECTED"
        print(json.dumps({"status": "REFUSED", "reason": code, "authorization": "NOT_AUTHORIZED"}))
        return 2


if __name__ == "__main__":
    sys.exit(main())

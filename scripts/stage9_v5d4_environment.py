"""Budowa rootless offline z przejrzanych danych; bez uruchamiania kontrolera.

Narzędzie i sąsiednie helpery muszą pochodzić z niezależnie przejrzanej kopii.
Oczekiwany SHA aprobaty pochodzi z kanału recenzenta, nigdy z samej propozycji.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import platform
import runpy
import stat
import sys
import sysconfig
from typing import Any
import venv
import zipfile


class Refusal(ValueError):
    """Brak dowodu lub nieobsługiwany profil zatrzymuje budowę."""


def require(value: bool, code: str) -> None:
    if not value:
        raise Refusal(code)


def sha(data: bytes) -> str:
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


def read(path: Path) -> bytes:
    require(not any(p.is_symlink() for p in (path, *path.parents)), "SYMLINK_INPUT")
    before = path.lstat()
    require(stat.S_ISREG(before.st_mode) and before.st_nlink == 1, "NONREGULAR_INPUT")
    require(before.st_size <= 512 * 1024 * 1024, "INPUT_LIMIT")
    descriptor = os.open(
        path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    )
    with os.fdopen(descriptor, "rb") as stream:
        require(file_identity(os.fstat(stream.fileno())) == file_identity(before), "INPUT_CHANGED")
        data = stream.read(512 * 1024 * 1024 + 1)
        require(file_identity(os.fstat(stream.fileno())) == file_identity(before), "INPUT_CHANGED")
    require(
        file_identity(path.lstat()) == file_identity(before) and len(data) == before.st_size,
        "INPUT_CHANGED",
    )
    return data


def document(path: Path) -> Any:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            require(key not in result, "DUPLICATE_JSON_KEY")
            result[key] = value
        return result

    return json.loads(read(path), object_pairs_hook=pairs)


def relative(value: str) -> Path:
    p = PurePosixPath(value)
    require(
        bool(p.parts)
        and not p.is_absolute()
        and ".." not in p.parts
        and "\\" not in value
        and p.as_posix() == value,
        "UNSAFE_MEMBER",
    )
    return Path(*p.parts)


def check_startup(path: Path) -> None:
    require(
        path.suffix not in {".pth", ".pyc", ".pyo"}
        and path.name not in {"sitecustomize.py", "usercustomize.py"}
        and "__pycache__" not in path.parts,
        "STARTUP_OR_BYTECODE_REJECTED",
    )


def write_new(root: Path, name: str, data: bytes) -> None:
    path = root / relative(name)
    check_startup(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(data)
    path.chmod(0o600)


def extract_wheel(data: bytes, destination: Path) -> None:
    """Tylko purelib/platlib/data, bez skryptów instalacyjnych i bez hooks.

    Dane .data/data są zachowane w jawnej lokalnej lokalizacji; koła potrzebujące
    systemowego prefixu, nagłówków lub skryptów wymagają osobnego projektu.
    """
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        require(len(archive.infolist()) < 100_000, "WHEEL_MEMBER_LIMIT")
        require(sum(i.file_size for i in archive.infolist()) <= 2 * 1024**3, "WHEEL_SIZE_LIMIT")
        seen: set[str] = set()
        for info in archive.infolist():
            require(info.filename not in seen, "DUPLICATE_WHEEL_MEMBER")
            seen.add(info.filename)
            p = relative(info.filename.rstrip("/"))
            require(not stat.S_ISLNK(info.external_attr >> 16), "WHEEL_SYMLINK")
            if info.is_dir():
                continue
            parts = p.parts
            if parts[0].endswith(".data"):
                require(
                    len(parts) > 2 and parts[1] in {"purelib", "platlib"}, "UNSUPPORTED_WHEEL_DATA"
                )
                p = Path(*parts[2:])
            write_new(destination, p.as_posix(), archive.read(info))


def helper(name: str) -> dict[str, Any]:
    # Kod zaufanego zestawu narzędzi, nigdy plik z wheelhouse ani obrazu gościa.
    return runpy.run_path(str(Path(__file__).with_name(name)))


def host_contract() -> dict[str, str]:
    release = platform.freedesktop_os_release()
    return {
        "python_full_version": platform.python_version(),
        "implementation": platform.python_implementation(),
        "machine": platform.machine(),
        "system": platform.system(),
        "soabi": str(sysconfig.get_config_var("SOABI")),
        "interpreter_sha256": sha(Path(sys.executable).resolve().read_bytes()),
        "distribution": release.get("ID", ""),
        "distribution_version": release.get("VERSION_ID", ""),
        "glibc": str(getattr(os, "confstr")("CS_GNU_LIBC_VERSION")),
    }


def build(inputs: dict[str, Any], approval: dict[str, Any], output: Path) -> dict[str, Any]:
    require(sys.flags.isolated == 1 and sys.flags.no_site == 1, "REQUIRE_ISOLATED_NO_SITE")
    require(
        sys.platform == "linux" and getattr(os, "geteuid")() != 0, "UNPRIVILEGED_ACCOUNT_REQUIRED"
    )
    require(
        set(inputs)
        == {"repository", "wheelhouse", "origins", "proposal", "abi_request", "abi_expectations"},
        "INPUT_SCHEMA_REJECTED",
    )
    require(
        set(approval)
        == {
            "host",
            "proposal_sha256",
            "wheel_native",
            "code",
            "dynamic_review_sha256",
            "abi_request_sha256",
            "abi_expectations_sha256",
            "bindings",
            "base_runtime",
            "host_tools",
        },
        "APPROVAL_SCHEMA_REJECTED",
    )
    expected_host = approval["host"]
    require(host_contract() == expected_host, "BUILD_HOST_MISMATCH")
    require(
        expected_host["python_full_version"] == "3.12.3"
        and expected_host["implementation"] == "CPython"
        and expected_host["machine"] == "x86_64"
        and expected_host["system"] == "Linux"
        and expected_host["soabi"] == "cpython-312-x86_64-linux-gnu",
        "CP312_UBUNTU_PROFILE_REQUIRED",
    )
    require(
        expected_host["distribution"] == "ubuntu"
        and expected_host["distribution_version"] == "24.04"
        and expected_host["glibc"] == "glibc 2.39",
        "UBUNTU_GLIBC_PROFILE_REQUIRED",
    )
    require(not output.exists() and not output.is_symlink(), "OUTPUT_ALREADY_EXISTS")
    require(
        output.is_absolute() and str(output) == os.path.abspath(output), "ABSOLUTE_OUTPUT_REQUIRED"
    )
    require(not any(p.is_symlink() for p in output.parents), "OUTPUT_SYMLINK")
    require(
        not any(
            output.absolute().is_relative_to(p)
            for p in (
                Path("/opt"),
                Path("/etc"),
                Path("/usr"),
                Path("/boot"),
                Path("/dev"),
                Path("/proc"),
                Path("/sys"),
            )
        ),
        "SYSTEM_OUTPUT_REJECTED",
    )
    repository = Path(inputs["repository"])
    deps = helper("stage9_v5d4_dependencies.py")
    proposal = document(Path(inputs["proposal"]))
    verification = deps["verify"](
        repository / "pyproject.toml",
        Path(inputs["wheelhouse"]),
        document(Path(inputs["origins"])),
        proposal,
        approval["proposal_sha256"],
        approval["wheel_native"],
    )
    closure_tools = helper("stage9_v5d4_closure.py")
    closure = closure_tools["inventory"](repository)
    require(closure["static_graph_closed"], "SOURCE_CLOSURE_INCOMPLETE")
    # Dynamiczne mechanizmy nie stanowią dowodu pełności; podlegają osobnemu review.
    require(
        approval["dynamic_review_sha256"] == sha(deps["canonical"](closure)),
        "DYNAMIC_REVIEW_MISSING",
    )
    code = {item["path"]: item["sha256"] for item in closure["sources"]}
    require(code == approval["code"], "CODE_APPROVAL_INCOMPLETE")
    base_runtime = closure_tools["base_runtime_inventory"](
        [Path(p) for p in approval["base_runtime"]["roots"]]
    )
    require(base_runtime == approval["base_runtime"], "BASE_RUNTIME_APPROVAL_MISMATCH")
    host_tools = approval["host_tools"]
    require(set(host_tools) == {"/usr/bin/qemu-img"}, "HOST_TOOL_INVENTORY_REQUIRED")
    for path, digest in host_tools.items():
        require(sha(read(Path(path))) == digest, "HOST_TOOL_SHA256_MISMATCH")
    for entry in sys.path:
        path = Path(entry)
        if path.exists():
            require(
                path.is_absolute()
                and any(
                    path == Path(root) or path.is_relative_to(Path(root))
                    for root in base_runtime["roots"]
                ),
                "BASE_IMPORT_PATH_UNCOVERED",
            )
    abi_request_raw = read(Path(inputs["abi_request"]))
    require(sha(abi_request_raw) == approval["abi_request_sha256"], "ABI_REQUEST_MISMATCH")
    abi_request = document(Path(inputs["abi_request"]))
    abi_expectations_raw = read(Path(inputs["abi_expectations"]))
    require(
        sha(abi_expectations_raw) == approval["abi_expectations_sha256"],
        "ABI_EXPECTATIONS_MISMATCH",
    )
    abi_expectations = document(Path(inputs["abi_expectations"]))
    # Wszystkie kontrole poprzedzają tworzenie nowego katalogu. Częściowy wynik
    # po I/O failure nie ma ENVIRONMENT.json i nie nadaje się do uruchomienia.
    output.mkdir(mode=0o700)
    runtime = output / "venv"
    venv.EnvBuilder(system_site_packages=False, with_pip=False, symlinks=False).create(runtime)
    # venv tworzy ten alias nawet przy symlinks=False. Nie jest używany przy -S.
    lib64 = runtime / "lib64"
    if lib64.is_symlink() and lib64.readlink() == Path("lib"):
        lib64.unlink()
    packages = output / "packages"
    packages.mkdir(mode=0o700)
    require(abi_request["import_roots"] == [str(packages)], "IMPORT_ROOT_PLAN_MISMATCH")
    require(
        {"libvirt", "libvirtmod", "cryptography", "defusedxml"}
        <= set(abi_request["required_imports"]),
        "REQUIRED_IMPORTS_MISSING",
    )
    for package in proposal["packages"]:
        data = read(Path(inputs["wheelhouse"]) / package["filename"])
        require(sha(data) == package["sha256"], "WHEEL_CHANGED_AFTER_VERIFICATION")
        extract_wheel(data, packages)
    for name, digest in code.items():
        data = read(repository / relative(name))
        require(sha(data) == digest, "SOURCE_CHANGED_AFTER_REVIEW")
        write_new(packages, name, data)
    bindings = approval["bindings"]
    require(
        set(bindings) == {"libvirt.py", "libvirtmod.cpython-312-x86_64-linux-gnu.so"},
        "BINDING_INVENTORY",
    )
    binding_paths = {
        "libvirt.py": abi_request["libvirt_module"],
        "libvirtmod.cpython-312-x86_64-linux-gnu.so": abi_request["libvirt_extension"],
    }
    for name, expected in bindings.items():
        data = read(Path(expected["path"]))
        require(sha(data) == expected["sha256"], "SYSTEM_BINDING_MISMATCH")
        # Weryfikator analizuje KOPIĘ w nowym środowisku, nie inny plik hosta.
        require(binding_paths[name] == str(packages / name), "BINDING_ABI_PATH_MISMATCH")
        write_new(packages, name, data)
    expected_interpreter = runtime / "bin" / "python3.12"
    require(
        abi_request["interpreter"] == str(expected_interpreter), "INTERPRETER_ABI_PATH_MISMATCH"
    )
    native_paths = {
        str(packages / relative(member))
        for package in proposal["packages"]
        for member in package["native_members"]
    } | {
        str(packages / "libvirtmod.cpython-312-x86_64-linux-gnu.so"),
        *base_runtime["native"],
        *host_tools,
    }
    # .data relocation musi mieć to samo mapowanie także w zatwierdzonym ABI.
    require(
        not any(".data/" in name for name in native_paths), "NATIVE_DATA_RELOCATION_REQUIRES_REVIEW"
    )
    require(
        native_paths
        <= set(abi_request["additional_extensions"]) | {abi_request["libvirt_extension"]},
        "NATIVE_ABI_INVENTORY_INCOMPLETE",
    )
    abi_result = helper("stage9_v5d4_abi.py")["diagnose"](abi_request, abi_expectations)
    require(abi_result["status"] == "STATIC_ABI_MATCH", "ABI_REFUSED")
    require(
        native_paths <= set(abi_result["elf_measurements"]), "NATIVE_ABI_MEASUREMENT_INCOMPLETE"
    )
    launcher = read(Path(__file__).with_name("stage9_v5d4_launch.py"))
    write_new(output, "launch.py", launcher)
    write_new(output, "runtime-audit.py", read(Path(__file__).with_name("stage9_v5d4_closure.py")))
    files = {
        p.relative_to(output).as_posix(): sha(p.read_bytes())
        for p in sorted(output.rglob("*"))
        if p.is_file() and not p.is_symlink()
    }
    receipt = {
        "profile": "stage9-v5d4-prepared-environment-v1",
        "files": files,
        "host": expected_host,
        "root": str(output),
        "dependency_verification": verification,
        "base_runtime": base_runtime,
        "external_files": {
            **base_runtime["files"],
            **host_tools,
            **{
                p: digest
                for p, digest in abi_expectations["sha256"].items()
                if not Path(p).is_relative_to(output)
            },
        },
        "abi": abi_result,
        "authorization": "NOT_AUTHORIZED",
        "site": "DISABLED",
        "native_load": "NOT_EXECUTED",
        "safety": {
            "Stage 9": "IN_PROGRESS",
            "Observer 3A": "BLOCKED",
            "Stage 3B": "NOT AUTHORIZED",
            "Production": "NO-GO",
        },
    }
    write_new(output, "ENVIRONMENT.json", deps["canonical"](receipt) + b"\n")
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", required=True, type=Path)
    parser.add_argument("--approval", required=True, type=Path)
    parser.add_argument("--approval-sha256", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        raw = read(args.approval)
        require(sha(raw) == args.approval_sha256, "INDEPENDENT_APPROVAL_DIGEST_MISMATCH")
        result = build(document(args.inputs), document(args.approval), args.output)
        print(
            json.dumps(
                {
                    "status": "PREPARED_FOR_REVIEW",
                    "files": len(result["files"]),
                    "authorization": "NOT_AUTHORIZED",
                }
            )
        )
        return 0
    except (ValueError, OSError, KeyError, TypeError, zipfile.BadZipFile):
        print(json.dumps({"status": "REFUSED", "authorization": "NOT_AUTHORIZED"}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

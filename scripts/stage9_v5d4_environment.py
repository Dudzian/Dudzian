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
from typing import Any, cast
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
    if os.name == "nt":
        try:
            return cast(
                bytes,
                helper("stage9_v5d4_dependencies.py")["windows_read_regular"](
                    path, 512 * 1024 * 1024
                ),
            )
        except ValueError as exc:
            code = "INPUT_LIMIT" if str(exc) == "INPUT_TOO_LARGE" else str(exc)
            raise Refusal(code) from exc
    require(
        os.name == "posix"
        and bool(getattr(os, "O_NOFOLLOW", None))
        and bool(getattr(os, "O_NONBLOCK", None)),
        "SECURE_FILE_READER_UNAVAILABLE",
    )
    require(not any(p.is_symlink() for p in (path, *path.parents)), "SYMLINK_INPUT")
    before = path.lstat()
    require(stat.S_ISREG(before.st_mode) and before.st_nlink == 1, "NONREGULAR_INPUT")
    require(before.st_size <= 512 * 1024 * 1024, "INPUT_LIMIT")
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(descriptor, "rb") as stream:
        require(file_identity(os.fstat(stream.fileno())) == file_identity(before), "INPUT_CHANGED")
        data = stream.read(512 * 1024 * 1024 + 1)
        require(file_identity(os.fstat(stream.fileno())) == file_identity(before), "INPUT_CHANGED")
    require(
        file_identity(path.lstat()) == file_identity(before) and len(data) == before.st_size,
        "INPUT_CHANGED",
    )
    return data


def parse_document(raw: bytes) -> Any:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            require(key not in result, "DUPLICATE_JSON_KEY")
            result[key] = value
        return result

    return json.loads(raw, object_pairs_hook=pairs)


def document(path: Path) -> Any:
    return parse_document(read(path))


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
        and not any(
            part in {"sitecustomize", "usercustomize"}
            or part.startswith(("sitecustomize.", "usercustomize."))
            for part in path.parts
        )
        and "__pycache__" not in path.parts,
        "STARTUP_OR_BYTECODE_REJECTED",
    )


def write_new(root: Path, name: str, data: bytes) -> None:
    member = relative(name)
    path = root / member
    check_startup(path)
    require(not any(p.is_symlink() for p in (root, *root.parents)), "MATERIALIZATION_SYMLINK")
    if os.name == "posix":
        fchmod = getattr(os, "fchmod", None)
        if not callable(fchmod):
            raise Refusal("SECURE_FILE_WRITER_UNAVAILABLE")
        require(
            bool(getattr(os, "O_NOFOLLOW", None)) and bool(getattr(os, "O_DIRECTORY", None)),
            "SECURE_FILE_WRITER_UNAVAILABLE",
        )
        absolute_root = root.absolute()
        before = absolute_root.lstat()
        require(stat.S_ISDIR(before.st_mode), "MATERIALIZATION_SYMLINK")
        descriptor = os.open(absolute_root.anchor, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            # Opening an absolute final component with O_NOFOLLOW still follows
            # ancestor links. Traverse every component relative to held handles.
            for component in absolute_root.parts[1:]:
                child = os.open(
                    component, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=descriptor
                )
                os.close(descriptor)
                descriptor = child
            require(
                file_identity(os.fstat(descriptor)) == file_identity(before),
                "MATERIALIZATION_ROOT_CHANGED",
            )
        except OSError as exc:
            os.close(descriptor)
            raise Refusal("MATERIALIZATION_SYMLINK") from exc
        except ValueError:
            os.close(descriptor)
            raise
        try:
            for component in member.parts[:-1]:
                try:
                    os.mkdir(component, mode=0o700, dir_fd=descriptor)
                except FileExistsError:
                    pass
                metadata = os.stat(component, dir_fd=descriptor, follow_symlinks=False)
                require(stat.S_ISDIR(metadata.st_mode), "MATERIALIZATION_SYMLINK")
                child = os.open(
                    component, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=descriptor
                )
                if file_identity(os.fstat(child)) != file_identity(metadata):
                    os.close(child)
                    raise Refusal("MATERIALIZATION_DIRECTORY_CHANGED")
                os.close(descriptor)
                descriptor = child
            try:
                target = os.open(
                    member.name,
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                    0o600,
                    dir_fd=descriptor,
                )
            except FileExistsError as exc:
                raise Refusal("MATERIALIZATION_COLLISION") from exc
            with os.fdopen(target, "wb") as stream:
                stream.write(data)
                fchmod(stream.fileno(), 0o600)
        finally:
            os.close(descriptor)
    else:
        # Budowa Linux odmawia na Windows; przenośny ekstraktor nadal nie
        # może nadpisać celu ani świadomie przejść przez dowiązany katalog.
        path.parent.mkdir(parents=True, exist_ok=True)
        require(not any(p.is_symlink() for p in path.parents), "MATERIALIZATION_SYMLINK")
        try:
            with path.open("xb") as stream:
                stream.write(data)
        except FileExistsError as exc:
            raise Refusal("MATERIALIZATION_COLLISION") from exc
        path.chmod(0o600)


def reserve_path(files: dict[str, str], directories: set[str], name: str, digest: str) -> None:
    """Nawet identyczne bajty nie ustanawiają zgody na współdzielenie pliku."""
    path = relative(name)
    check_startup(path)
    require(name not in files and name not in directories, "MATERIALIZATION_COLLISION")
    parents = [parent.as_posix() for parent in path.parents if parent != Path(".")]
    require(not any(parent in files for parent in parents), "MATERIALIZATION_COLLISION")
    files[name] = digest
    directories.update(parents)


def wheel_plan(data: bytes, package: dict[str, Any] | None = None) -> dict[str, dict[str, str]]:
    """Pełna walidacja poprzedza pierwszy zapis; nic z wheel nie jest wykonywane.

    Budowa zawsze przekazuje kompletny pakiet z propozycji i ponownie używa
    inspektora. Wariant bez pakietu służy wyłącznie surowemu kopiowaniu ZIP:
    nie może pomijać bytecode ani przyznawać wyjątku od kontroli startup.
    """
    if package is not None:
        deps = helper("stage9_v5d4_dependencies.py")
        rebuilt = deps["inspect_wheel_bytes"](data, package["filename"], package["origin"])
        require(
            deps["canonical"](rebuilt) == deps["canonical"](package),
            "WHEEL_PLAN_CHANGED_AFTER_VERIFICATION",
        )
        return dict(rebuilt["materialized_members"])
    plan: dict[str, dict[str, str]] = {}
    files: dict[str, str] = {}
    directories: set[str] = set()
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        require(len(archive.infolist()) < 100_000, "WHEEL_MEMBER_LIMIT")
        require(sum(i.file_size for i in archive.infolist()) <= 2 * 1024**3, "WHEEL_SIZE_LIMIT")
        seen: set[str] = set()
        for info in archive.infolist():
            require(info.filename not in seen, "DUPLICATE_WHEEL_MEMBER")
            seen.add(info.filename)
            require(info.orig_filename == info.filename, "UNSAFE_MEMBER")
            p = relative(info.filename.rstrip("/"))
            mode = info.external_attr >> 16
            require(stat.S_IFMT(mode) in {0, stat.S_IFREG, stat.S_IFDIR}, "WHEEL_MEMBER_SPECIAL")
            check_startup(p)
            if info.is_dir():
                continue
            parts = p.parts
            if parts[0].endswith(".data"):
                require(
                    len(parts) > 2 and parts[1] in {"purelib", "platlib"}, "UNSUPPORTED_WHEEL_DATA"
                )
                p = Path(*parts[2:])
            content_sha = sha(archive.read(info))
            reserve_path(files, directories, p.as_posix(), content_sha)
            plan[info.filename] = {"path": p.as_posix(), "sha256": content_sha}
    return plan


def extract_wheel(data: bytes, destination: Path, package: dict[str, Any] | None = None) -> None:
    """Materializuj dokładny plan; oryginalny wheel i RECORD pozostają niezmienione."""
    plan = wheel_plan(data, package)
    # Sprawdź wszystkie istniejące cele przed jakimkolwiek zapisem tego wheel.
    for member in plan.values():
        target = destination / relative(member["path"])
        check_startup(target)
        require(not target.exists() and not target.is_symlink(), "MATERIALIZATION_COLLISION")
        for parent in target.parents:
            require(not parent.is_symlink(), "MATERIALIZATION_SYMLINK")
            require(not parent.exists() or parent.is_dir(), "MATERIALIZATION_COLLISION")
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        for name, member in plan.items():
            content = archive.read(name)
            require(sha(content) == member["sha256"], "WHEEL_MEMBER_PLAN_MISMATCH")
            write_new(destination, member["path"], content)


def prepare_wheel_set(
    packages: list[dict[str, Any]], wheelhouse: Path, approved_shared: dict[str, Any] | None
) -> tuple[
    list[tuple[dict[str, Any], bytes, dict[str, dict[str, str]]]], dict[str, str], dict[str, Any]
]:
    """Capture immutable archives and validate every declaration before writing.

    Only the separate exact contract can identify a shared target. The ordinary
    reservation guard is retained for every other target and its ancestors.
    """
    shared_tools = runpy.run_path(str(Path(__file__).with_name("stage9_v5d5_shared.py")))
    shared = shared_tools["verify_shared"](packages, approved_shared)
    shared_paths = {row["path"] for row in shared["members"]}
    snapshots: list[tuple[dict[str, Any], bytes, dict[str, dict[str, str]]]] = []
    planned: dict[str, str] = {}
    directories: set[str] = set()
    for package in sorted(packages, key=lambda item: (item["name"], item["filename"])):
        data = read(wheelhouse / relative(package["filename"]))
        require(sha(data) == package["sha256"], "WHEEL_CHANGED_AFTER_VERIFICATION")
        plan = wheel_plan(data, package)
        snapshots.append((package, data, plan))
        for member in plan.values():
            target = member["path"]
            if target in planned:
                require(
                    target in shared_paths and planned[target] == member["sha256"],
                    "MATERIALIZATION_COLLISION",
                )
            else:
                reserve_path(planned, directories, target, member["sha256"])
    return snapshots, planned, shared


def materialize_wheel_set(
    snapshots: list[tuple[dict[str, Any], bytes, dict[str, dict[str, str]]]],
    destination: Path,
    shared: dict[str, Any],
) -> None:
    """Use retained immutable bytes; write each approved shared target once."""
    for _, _, plan in snapshots:
        for member in plan.values():
            target_path = destination / relative(member["path"])
            require(
                not target_path.exists() and not target_path.is_symlink(),
                "MATERIALIZATION_COLLISION",
            )
            for parent in target_path.parents:
                require(not parent.is_symlink(), "MATERIALIZATION_SYMLINK")
                require(not parent.exists() or parent.is_dir(), "MATERIALIZATION_COLLISION")
    shared_paths = {row["path"] for row in shared["members"]}
    written: set[str] = set()
    for _, data, plan in snapshots:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            for name, member in sorted(plan.items()):
                target = member["path"]
                if target in written:
                    require(target in shared_paths, "MATERIALIZATION_COLLISION")
                    continue
                content = archive.read(name)
                require(sha(content) == member["sha256"], "WHEEL_MEMBER_PLAN_MISMATCH")
                write_new(destination, target, content)
                written.add(target)


def verify_materialized(root: Path, expected: dict[str, str]) -> None:
    actual: dict[str, str] = {}
    directories: set[str] = set()
    for path in root.rglob("*"):
        require(not path.is_symlink(), "MATERIALIZATION_SYMLINK")
        check_startup(path)
        if path.is_dir():
            directories.add(path.relative_to(root).as_posix())
            continue
        require(stat.S_ISREG(path.lstat().st_mode), "MATERIALIZATION_SPECIAL")
        actual[path.relative_to(root).as_posix()] = sha(read(path))
    require(actual == expected, "MATERIALIZED_INVENTORY_MISMATCH")
    require(
        directories
        == {
            parent.as_posix()
            for name in expected
            for parent in relative(name).parents
            if parent != Path(".")
        },
        "MATERIALIZED_DIRECTORY_MISMATCH",
    )


def helper(name: str) -> dict[str, Any]:
    # Kod zaufanego zestawu narzędzi, nigdy plik z wheelhouse ani obrazu gościa.
    return runpy.run_path(str(Path(__file__).with_name(name)))


def host_contract() -> dict[str, str]:
    require(sys.platform == "linux", "UBUNTU_LINUX_HOST_REQUIRED")
    require(
        callable(getattr(platform, "freedesktop_os_release", None))
        and callable(getattr(os, "confstr", None)),
        "LINUX_HOST_CAPABILITY_MISSING",
    )
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
    require(sys.platform == "linux", "UBUNTU_LINUX_HOST_REQUIRED")
    geteuid = getattr(os, "geteuid", None)
    if not callable(geteuid):
        raise Refusal("LINUX_HOST_CAPABILITY_MISSING")
    require(geteuid() != 0, "UNPRIVILEGED_ACCOUNT_REQUIRED")
    require(
        set(inputs)
        == {"repository", "wheelhouse", "origins", "proposal", "abi_request", "abi_expectations"},
        "INPUT_SCHEMA_REJECTED",
    )
    require(
        set(approval) - {"wheel_omissions", "wheel_shared_members"}
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
        approval.get("wheel_omissions"),
        approval.get("wheel_shared_members"),
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
    abi_request = parse_document(abi_request_raw)
    abi_expectations_raw = read(Path(inputs["abi_expectations"]))
    require(
        sha(abi_expectations_raw) == approval["abi_expectations_sha256"],
        "ABI_EXPECTATIONS_MISMATCH",
    )
    abi_expectations = parse_document(abi_expectations_raw)
    packages = output / "packages"
    runtime = output / "venv"
    require(abi_request["import_roots"] == [str(packages)], "IMPORT_ROOT_PLAN_MISMATCH")
    require(
        {"libvirt", "libvirtmod", "cryptography", "defusedxml"}
        <= set(abi_request["required_imports"]),
        "REQUIRED_IMPORTS_MISSING",
    )
    bindings = approval["bindings"]
    require(
        set(bindings) == {"libvirt.py", "libvirtmod.cpython-312-x86_64-linux-gnu.so"},
        "BINDING_INVENTORY",
    )
    binding_paths = {
        "libvirt.py": abi_request["libvirt_module"],
        "libvirtmod.cpython-312-x86_64-linux-gnu.so": abi_request["libvirt_extension"],
    }
    snapshots, planned, shared = prepare_wheel_set(
        proposal["packages"], Path(inputs["wheelhouse"]), approval.get("wheel_shared_members")
    )
    planned_directories = {
        parent.as_posix()
        for name in planned
        for parent in relative(name).parents
        if parent != Path(".")
    }
    for name, digest in code.items():
        require(sha(read(repository / relative(name))) == digest, "SOURCE_CHANGED_AFTER_REVIEW")
        reserve_path(planned, planned_directories, name, digest)
    for name, expected in bindings.items():
        require(sha(read(Path(expected["path"]))) == expected["sha256"], "SYSTEM_BINDING_MISMATCH")
        require(binding_paths[name] == str(packages / name), "BINDING_ABI_PATH_MISMATCH")
        reserve_path(planned, planned_directories, name, expected["sha256"])
    expected_interpreter = runtime / "bin" / "python3.12"
    require(
        abi_request["interpreter"] == str(expected_interpreter), "INTERPRETER_ABI_PATH_MISMATCH"
    )
    # Dotychczasowy profil nie przyznaje zgody na relokację native z .data.
    require(
        not any(
            ".data/" in member
            for package in proposal["packages"]
            for member in package["native_members"]
        ),
        "NATIVE_DATA_RELOCATION_REQUIRES_REVIEW",
    )
    native_paths = {
        str(packages / relative(package["materialized_members"][member]["path"]))
        for package in proposal["packages"]
        for member in package["native_members"]
    } | {
        str(packages / "libvirtmod.cpython-312-x86_64-linux-gnu.so"),
        *base_runtime["native"],
        *host_tools,
    }
    require(
        native_paths
        <= set(abi_request["additional_extensions"]) | {abi_request["libvirt_extension"]},
        "NATIVE_ABI_INVENTORY_INCOMPLETE",
    )
    # Pełne plany, aprobaty i kolizje sprawdzono przed pierwszym zapisem.
    # ABI bada dopiero kopie. Częściowy wynik nie ma ENVIRONMENT.json.
    output.mkdir(mode=0o700)
    venv.EnvBuilder(system_site_packages=False, with_pip=False, symlinks=False).create(runtime)
    # venv tworzy ten alias nawet przy symlinks=False. Nie jest używany przy -S.
    lib64 = runtime / "lib64"
    if lib64.is_symlink() and lib64.readlink() == Path("lib"):
        lib64.unlink()
    packages.mkdir(mode=0o700)
    materialize_wheel_set(snapshots, packages, shared)
    for name, digest in code.items():
        data = read(repository / relative(name))
        require(sha(data) == digest, "SOURCE_CHANGED_AFTER_REVIEW")
        write_new(packages, name, data)
    for name, expected in bindings.items():
        data = read(Path(expected["path"]))
        require(sha(data) == expected["sha256"], "SYSTEM_BINDING_MISMATCH")
        # Weryfikator analizuje KOPIĘ w nowym środowisku, nie inny plik hosta.
        require(binding_paths[name] == str(packages / name), "BINDING_ABI_PATH_MISMATCH")
        write_new(packages, name, data)
    verify_materialized(packages, planned)
    abi_result = helper("stage9_v5d4_abi.py")["diagnose"](abi_request, abi_expectations)
    require(abi_result["status"] == "STATIC_ABI_MATCH", "ABI_REFUSED")
    require(
        native_paths <= set(abi_result["elf_measurements"]), "NATIVE_ABI_MEASUREMENT_INCOMPLETE"
    )
    launcher = read(Path(__file__).with_name("stage9_v5d4_launch.py"))
    write_new(output, "launch.py", launcher)
    write_new(output, "runtime-audit.py", read(Path(__file__).with_name("stage9_v5d4_closure.py")))
    write_new(output, "shared-audit.py", read(Path(__file__).with_name("stage9_v5d5_shared.py")))
    files = {
        p.relative_to(output).as_posix(): sha(read(p))
        for p in sorted(output.rglob("*"))
        if p.is_file() and not p.is_symlink()
    }
    require(
        {
            name.removeprefix("packages/"): digest
            for name, digest in files.items()
            if name.startswith("packages/")
        }
        == planned,
        "MATERIALIZED_INVENTORY_CHANGED_BEFORE_RECEIPT",
    )
    verify_materialized(packages, planned)
    receipt = {
        "profile": "stage9-v5d5-prepared-environment-v1",
        "files": files,
        "directories": sorted(
            path.relative_to(output).as_posix() for path in output.rglob("*") if path.is_dir()
        ),
        "host": expected_host,
        "root": str(output),
        "dependency_verification": verification,
        "wheel_archives": [
            {key: package[key] for key in ("name", "version", "filename", "sha256", "origin")}
            for package in proposal["packages"]
        ],
        "wheel_materialization": [
            {
                key: package[key]
                for key in (
                    "name",
                    "version",
                    "filename",
                    "sha256",
                    "materialized_members",
                    "member_metadata",
                )
            }
            for package, _, _ in snapshots
        ],
        "shared_wheel_members": shared,
        "shared_wheel_expectations": approval.get("wheel_shared_members"),
        "shared_wheel_policy": shared["policy"],
        "shared_wheel_output": [
            {"path": row["path"], "sha256": planned[row["path"]], "mode": 0o600, "type": "regular"}
            for row in shared["members"]
        ],
        "wheel_omissions": [
            member for package in proposal["packages"] for member in package["omitted_members"]
        ],
        "installation_contract_exception": approval.get("wheel_omissions"),
        "wheel_record_semantics": "UNCHANGED_UPSTREAM_ARCHIVE_RECORD_NOT_INSTALLED_INVENTORY",
        "package_files": planned,
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
    for row in receipt["shared_wheel_output"]:
        metadata = (packages / row["path"]).lstat()
        require(
            stat.S_ISREG(metadata.st_mode)
            and metadata.st_nlink == 1
            and stat.S_IMODE(metadata.st_mode) == row["mode"],
            "SHARED_WHEEL_OUTPUT_MODE_CHANGED_BEFORE_RECEIPT",
        )
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
        result = build(document(args.inputs), parse_document(raw), args.output)
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

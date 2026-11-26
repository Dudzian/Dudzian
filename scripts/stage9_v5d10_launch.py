"""Verify a sealed A receipt and dispatch only the authorized read-only preflight.

The operator must start a trusted copy with a clean environment in an externally
confined Linux filesystem/IPC view. The Python finder enforces the import
boundary inside that view. It does not confine native code on its own.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable
import errno
from importlib import abc, machinery, util
import json
import os
from pathlib import Path
import re
import stat
import sys
from types import ModuleType
from typing import Any, cast

LAUNCH_AUTHORIZATION = "stage9-v5d10-observer-preflight-launch-authorization-v1"
COMMAND_ENV = {"LANG": "C", "LC_ALL": "C", "PATH": "/usr/bin:/bin"}
DEPENDENCIES = {
    "deployment",
    "scripts",
    "cryptography",
    "defusedxml",
    "cffi",
    "_cffi_backend",
    "pycparser",
    "libvirt",
    "libvirtmod",
}
FORBIDDEN = {
    "site",
    "sitecustomize",
    "usercustomize",
    "zstandard",
    "ctypes",
    "_ctypes",
    "cygvirtmod",
}
HELPERS = {"contract.py": "stage9_v5d10_contract.py", "runtime-audit.py": "stage9_v5d4_closure.py"}
LOCAL_SOCKETS = {"/run/libvirt/libvirt-sock-ro", "/run/libvirt/virtqemud-sock-ro"}
RECEIPT = "ENVIRONMENT.json"


class Refusal(ValueError):
    """A launch input, state or boundary did not match independent evidence."""


def _check(condition: bool, code: str) -> None:
    if not condition:
        raise Refusal(code)


def _regular(path: Path, limit: int = 16 * 1024 * 1024) -> bytes:
    # This small bootstrap runs before the reviewed contract helper is loaded.
    _check(not any(item.is_symlink() for item in (path, *path.parents)), "BOOTSTRAP_SYMLINK")
    flags = getattr(os, "O_NOFOLLOW", None), getattr(os, "O_NONBLOCK", None)
    _check(all(type(value) is int and value > 0 for value in flags), "BOOTSTRAP_READER_UNAVAILABLE")
    no_follow, nonblock = flags
    assert no_follow is not None and nonblock is not None
    fd = os.open(path, os.O_RDONLY | no_follow | nonblock)
    with os.fdopen(fd, "rb") as stream:
        before = os.fstat(stream.fileno())
        _check(
            stat.S_ISREG(before.st_mode) and before.st_nlink == 1 and before.st_size <= limit,
            "BOOTSTRAP_FILE",
        )
        data = stream.read(limit + 1)
        after = os.fstat(stream.fileno())
    _check(
        len(data) == before.st_size
        and (before.st_dev, before.st_ino, before.st_mtime_ns, before.st_ctime_ns)
        == (after.st_dev, after.st_ino, after.st_mtime_ns, after.st_ctime_ns),
        "BOOTSTRAP_CHANGED",
    )
    _check(
        (path.lstat().st_dev, path.lstat().st_ino) == (before.st_dev, before.st_ino),
        "BOOTSTRAP_CHANGED",
    )
    return data


def _sha(data: bytes) -> str:
    # Hash the launcher helpers before executing them. This is part of the
    # externally trusted launcher's bootstrap, before the profile finder.
    import hashlib

    return hashlib.sha256(data).hexdigest()


def _helper(root: Path, name: str, receipt: dict[str, Any]) -> dict[str, Any]:
    path = root / name
    raw = _regular(path)
    _check(_sha(raw) == receipt["files"][name]["sha256"], "HELPER_PIN_MISMATCH")
    # Use the measured in-memory source. Never reopen it or load a cache.
    module_name = f"_observer_{HELPERS[name].removesuffix('.py')}"
    loader = PinnedSourceLoader(module_name, str(path), lambda _: raw)
    spec = util.spec_from_loader(module_name, loader)
    if spec is None:
        raise Refusal("HELPER_SPEC_REJECTED")
    module = util.module_from_spec(spec)
    loader.exec_module(module)
    return vars(module)


def verify(
    root: Path, expected_receipt: str, expected_approval: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Check bytes, modes, base runtime and external objects without A imports."""
    raw = _regular(root / RECEIPT)
    _check(stat.S_IMODE((root / RECEIPT).lstat().st_mode) == 0o600, "RECEIPT_MODE_REJECTED")
    _check(
        re.fullmatch(r"[a-f0-9]{64}", expected_receipt) is not None
        and _sha(raw) == expected_receipt,
        "INDEPENDENT_RECEIPT_PIN_MISMATCH",
    )
    receipt = json.loads(raw)
    _check(
        type(receipt) is dict
        and receipt.get("profile") == "stage9-v5d10-observer-preflight-receipt-v1",
        "RECEIPT_PROFILE_REJECTED",
    )
    _check(
        receipt.get("approval_sha256") == expected_approval
        and re.fullmatch(r"[a-f0-9]{64}", expected_approval) is not None,
        "INDEPENDENT_APPROVAL_PIN_MISMATCH",
    )
    contract = _helper(root, "contract.py", receipt)
    receipt = contract["loads"](raw)
    contract["validate_receipt"](receipt)
    _check(receipt["root"] == str(root), "RECEIPT_ROOT_MISMATCH")
    plan = contract["inventory"](root, exclude_receipt=True)
    _check(
        plan == {"files": receipt["files"], "directories": receipt["directories"]},
        "RECEIPT_PLAN_MISMATCH",
    )
    auditor = _helper(root, "runtime-audit.py", receipt)
    base = receipt["base_runtime"]
    _check(
        auditor["base_runtime_inventory"]([Path(item) for item in base["roots"]]) == base,
        "BASE_RUNTIME_MISMATCH",
    )
    for path, expected in receipt["external_files"].items():
        _check(
            contract["digest"](contract["secure_read"](Path(path))) == expected,
            "EXTERNAL_PIN_MISMATCH",
        )
    return receipt, contract


class PinnedSourceLoader(machinery.SourceFileLoader):
    def __init__(self, fullname: str, filename: str, read: Any) -> None:
        super().__init__(fullname, filename)
        self.read = read

    def get_code(self, fullname: str):
        filename = self.get_filename(fullname)
        return self.source_to_code(self.read(filename), filename)


class ApprovedImports(abc.MetaPathFinder):
    """Resolves builtin/frozen or exact pinned files, including child packages."""

    def __init__(self, root: Path, receipt: dict[str, Any], contract: dict[str, Any]) -> None:
        self.package_root = root / "packages"
        self.base_roots = tuple(Path(value) for value in receipt["base_runtime"]["roots"])
        self.files = {
            **receipt["base_runtime"]["files"],
            **{
                str(root / name): row["sha256"]
                for name, row in receipt["files"].items()
                if name.startswith("packages/")
            },
        }
        self.links = receipt["base_runtime"]["symlinks"]
        self.contract = contract

    def _pin(self, filename: str) -> tuple[Path, str]:
        path = Path(filename)
        if filename in self.links:
            row = self.links[filename]
            _check(
                path.is_symlink()
                and os.readlink(path) == row["target"]
                and str(path.resolve(strict=True)) == row["resolved"],
                "IMPORT_LINK_CHANGED",
            )
            path = Path(row["resolved"])
        expected = self.files.get(str(path))
        if expected is None:
            raise Refusal("IMPORT_FILE_NOT_PINNED")
        return path, expected

    def read(self, filename: str) -> bytes:
        path, expected = self._pin(filename)
        result = cast(bytes, self.contract["secure_read"](path))
        _check(self.contract["digest"](result) == expected, "IMPORT_PIN_MISMATCH")
        return result

    def _search(self, fullname: str, path: Iterable[str] | None) -> tuple[str, ...]:
        first = fullname.partition(".")[0]
        if first in FORBIDDEN or first not in DEPENDENCIES | sys.stdlib_module_names:
            raise ModuleNotFoundError(f"Observer A import refused: {fullname}", name=fullname)
        roots = (self.package_root,) if first in DEPENDENCIES else self.base_roots
        if path is None:
            _check("." not in fullname, "IMPORT_PARENT_MISSING")
            candidates = (
                roots
                if first in DEPENDENCIES
                else tuple(
                    candidate for root in roots for candidate in (root, root / "lib-dynload")
                )
            )
        else:
            parent = Path(*fullname.split(".")[:-1])
            candidates = tuple(root / parent for root in roots)
            _check(all(Path(item) in candidates for item in path), "IMPORT_PARENT_PATH_REJECTED")
        return tuple(str(dirpath) for dirpath in candidates if dirpath.is_dir())

    def find_spec(
        self, fullname: str, path: Iterable[str] | None = None, target: ModuleType | None = None
    ):
        search = self._search(fullname, path)
        if (spec := machinery.BuiltinImporter.find_spec(fullname, None, target)) is not None:
            return spec
        if (spec := machinery.FrozenImporter.find_spec(fullname, None, target)) is not None:
            return spec
        specs, namespaces = [], []
        for directory in search:
            finder = machinery.FileFinder(
                directory,
                (machinery.ExtensionFileLoader, machinery.EXTENSION_SUFFIXES),
                (machinery.SourceFileLoader, machinery.SOURCE_SUFFIXES),
            )
            found = finder.find_spec(fullname, target)
            if found is not None:
                if found.origin is not None:
                    specs.append(found)
                elif found.submodule_search_locations is not None:
                    namespaces.extend(found.submodule_search_locations)
        _check(len(specs) <= 1, "IMPORT_ORIGIN_AMBIGUOUS")
        if specs:
            spec = specs[0]
            filename = spec.origin
            assert filename is not None
            self.read(filename)
            if isinstance(spec.loader, machinery.SourceFileLoader):
                spec.loader = PinnedSourceLoader(fullname, filename, self.read)
                spec.cached = None
            return spec
        if len(namespaces) == 1 and fullname in {"scripts", "deployment"}:
            prefix = str(Path(namespaces[0])) + "/"
            _check(
                any(name.startswith(prefix) for name in self.files), "IMPORT_NAMESPACE_NOT_PINNED"
            )
            namespace = machinery.ModuleSpec(fullname, None, is_package=True)
            namespace.submodule_search_locations = namespaces
            return namespace
        raise ModuleNotFoundError(f"Observer A import refused: {fullname}", name=fullname)

    def validate_startup(self) -> None:
        _check(
            tuple(sys.meta_path)
            == (machinery.BuiltinImporter, machinery.FrozenImporter, machinery.PathFinder),
            "STARTUP_FINDERS_REJECTED",
        )
        allowed_paths = {str(root) for root in self.base_roots} | {
            str(root / "lib-dynload") for root in self.base_roots
        }
        for entry in sys.path:
            _check(
                entry in allowed_paths
                or (entry == "/usr/lib/python312.zip" and not os.path.lexists(entry)),
                "STARTUP_IMPORT_ROOT_REJECTED",
            )
        for name, module in tuple(sys.modules.items()):
            if name == "__main__":
                continue
            if name in {"typing.io", "typing.re"}:
                parent = sys.modules.get("typing")
                _check(
                    parent is not None and module is vars(parent).get(name.split(".")[1]),
                    "PRELOADED_MODULE_REJECTED",
                )
                continue
            first = name.partition(".")[0]
            _check(
                first not in DEPENDENCIES | FORBIDDEN and first in sys.stdlib_module_names,
                "PRELOADED_MODULE_REJECTED",
            )
            spec = getattr(module, "__spec__", None)
            origin = getattr(spec, "origin", None)
            if origin not in {"built-in", "frozen"}:
                if type(origin) is not str:
                    raise Refusal("PRELOADED_ORIGIN_REJECTED")
                self.read(origin)


def _audit(finder: ApprovedImports, qemu_pin: str, event: str, args: tuple[Any, ...]) -> None:
    if event == "import":
        first = args[0].partition(".")[0]
        _check(
            first not in FORBIDDEN and first in DEPENDENCIES | sys.stdlib_module_names,
            "AUDIT_IMPORT_DENIED",
        )
        if len(args) > 1 and type(args[1]) is str:
            finder.read(args[1])
    if event.startswith("ctypes.") or event in {
        "os.exec",
        "os.posix_spawn",
        "os.system",
        "os.fork",
        "os.forkpty",
    }:
        raise Refusal("DYNAMIC_EXECUTION_REJECTED")
    if event == "subprocess.Popen":
        executable, argv, cwd, env = args
        _check(
            type(executable) is str
            and re.fullmatch(r"/proc/self/fd/[0-9]+", executable) is not None
            and type(argv) in {list, tuple}
            and len(argv) == 6
            and list(argv[:4]) == [executable, "info", "--output=json", "-f"]
            and argv[4] in {"raw", "qcow2"}
            and type(argv[5]) is str
            and re.fullmatch(r"/proc/self/fd/[0-9]+", argv[5]) is not None
            and cwd is None
            and env == COMMAND_ENV,
            "QEMU_INVOCATION_REJECTED",
        )
        fd = int(executable.rsplit("/", 1)[1])
        pin_path = Path("/usr/bin/qemu-img")
        metadata = os.fstat(fd)
        trusted = pin_path.stat()
        _check(
            (metadata.st_dev, metadata.st_ino) == (trusted.st_dev, trusted.st_ino),
            "QEMU_INODE_REJECTED",
        )
        _check(
            finder.contract["digest"](finder.contract["secure_read"](pin_path)) == qemu_pin,
            "QEMU_PIN_REJECTED",
        )


def _proc(path: str, limit: int = 4 * 1024 * 1024) -> bytes:
    with open(path, "rb") as stream:
        data = stream.read(limit + 1)
    _check(len(data) <= limit, "KERNEL_BOUNDARY_READ_LIMIT")
    return data


def _mount_view(raw: bytes, socket: str) -> None:
    mounts = {}
    for row in raw.splitlines():
        before, separator, after = row.partition(b" - ")
        details = before.split()
        _check(bool(separator) and len(details) >= 6 and bool(after.split()), "MOUNT_INFO_REJECTED")
        location = details[4].decode("ascii")
        flags = set(details[5].split(b","))
        _check(
            location not in mounts and location in {"/", "/proc", "/dev/null", socket},
            "UNAPPROVED_SUBMOUNT",
        )
        _check(
            not any(item.startswith((b"shared:", b"master:")) for item in details[6:]),
            "MOUNT_PROPAGATION_REJECTED",
        )
        _check(location == "/dev/null" or b"ro" in flags, "WRITABLE_SUBMOUNT")
        mounts[location] = (after.split()[0], flags)
    _check(
        "/" in mounts
        and "/proc" in mounts
        and {b"nosuid", b"nodev"} <= mounts["/"][1]
        and mounts["/proc"][0] == b"proc"
        and {b"nosuid", b"nodev", b"noexec"} <= mounts["/proc"][1],
        "PROC_VIEW_REQUIRED",
    )


def _file_descriptors() -> None:
    for entry in os.listdir("/proc/self/fd"):
        _check(entry.isascii() and entry.isdecimal(), "FD_VIEW_REJECTED")
        fd = int(entry)
        if fd > 2:
            try:
                os.fstat(fd)
            except OSError as exc:
                if exc.errno == errno.EBADF:  # transient directory fd from listdir
                    continue
                raise
            raise Refusal("UNEXPECTED_INHERITED_FD")
    null_device = getattr(Path("/dev/null").stat(), "st_rdev", None)
    _check(type(null_device) is int, "VIEW_DEVICE_REJECTED")
    for fd in (0, 1, 2):
        info = os.fstat(fd)
        _check(
            (stat.S_ISCHR(info.st_mode) and getattr(info, "st_rdev", None) == null_device)
            or (fd != 0 and stat.S_ISFIFO(info.st_mode)),
            "INHERITED_STDIO_REJECTED",
        )


def _abi_link_targets(abi_request: dict[str, Any], base_links: dict[str, Any]) -> dict[str, str]:
    # A uses Ubuntu's merged-/usr layout. The physical ABI request pins each
    # DT_NEEDED provider and the PT_INTERP loader rather than permitting an
    # arbitrary alias to any file in the approved view.
    targets = {name: row["resolved"] for name, row in base_links.items()}
    targets.update({"/lib": "/usr/lib", "/lib64": "/usr/lib64"})
    for name, provider in abi_request["library_map"].items():
        lookup = (
            "/usr" + name if name.startswith("/lib64/") else f"/usr/lib/x86_64-linux-gnu/{name}"
        )
        if lookup != provider:
            _check(targets.get(lookup, provider) == provider, "ABI_ALIAS_CONFLICT")
            targets[lookup] = provider
    return targets


def _view(
    authorization: dict[str, Any],
    root: Path,
    receipt: dict[str, Any],
    contract: dict[str, Any],
    authority: Path,
    abi_request: Path,
    targets: dict[str, str],
) -> None:
    """Require the separately measured namespace, confinement and exact visible set."""
    pin_paths = (
        {str(root / name) for name in receipt["files"]}
        | {str(root / RECEIPT), str(authority), str(abi_request)}
        | set(receipt["external_files"])
        | set(receipt["base_runtime"]["files"])
    )
    links = authorization["aliases"]
    extra = authorization["data_paths"]
    _check(
        type(links) is dict and type(extra) is list and all(type(name) is str for name in extra),
        "VIEW_SCHEMA_REJECTED",
    )
    _check(
        all(
            contract["path_value"](item, absolute=True)
            and item.startswith(("/etc/cryptohunter/observer-v5d1/", "/var/lib/libvirt/images/"))
            for item in extra
        ),
        "VIEW_DATA_PATH_REJECTED",
    )
    _check(
        all(
            contract["path_value"](alias, absolute=True) and type(target) is str
            for alias, target in links.items()
        ),
        "VIEW_ALIAS_INVALID",
    )
    _check(set(links) == set(targets), "VIEW_ABI_ALIAS_SET_REJECTED")
    _check(not (set(links) & pin_paths) and not (set(extra) & pin_paths), "VIEW_ENTRY_COLLISION")
    _check(
        not any(
            os.path.lexists(path)
            for path in (
                "/etc/ld.so.preload",
                "/usr/lib/python312.zip",
                "/etc/ld.so.cache",
                "/usr/lib/x86_64-linux-gnu/qemu",
                "/usr/lib/x86_64-linux-gnu/libvirt/connection-driver",
            )
        ),
        "UNQUALIFIED_LOADER_PATH",
    )
    config = Path("/etc/libvirt/libvirt.conf")
    _check(
        contract["sha256_value"](authorization["libvirt_config_sha256"])
        and _sha(contract["secure_read"](config, limit=1024 * 1024))
        == authorization["libvirt_config_sha256"],
        "LIBVIRT_CONFIG_PIN_MISMATCH",
    )
    singleton = {authorization["libvirt_socket"], "/dev/null", str(config)}
    expected_paths = pin_paths | set(links) | set(extra) | singleton
    expected_dirs = (
        {str(parent) for name in expected_paths for parent in Path(name).parents}
        | {str(root / name) for name in receipt["directories"]}
        | {"/proc"}
    )
    seen: set[str] = set()
    pending = [Path("/")]
    while pending:
        item = pending.pop()
        name = str(item)
        info = item.lstat()
        if stat.S_ISDIR(info.st_mode):
            _check(name in expected_dirs, "UNEXPECTED_VIEW_DIRECTORY")
            if name != "/proc":
                pending.extend(item.iterdir())
        elif name in links:
            _check(
                stat.S_ISLNK(info.st_mode) and os.readlink(item) == links[name],
                "VIEW_ALIAS_CHANGED",
            )
            resolved = str(item.resolve(strict=True))
            _check(resolved == targets[name], "VIEW_ALIAS_TARGET_REJECTED")
            seen.add(name)
        else:
            _check(name in expected_paths, "UNEXPECTED_VIEW_OBJECT")
            if name == authorization["libvirt_socket"]:
                _check(
                    stat.S_ISSOCK(info.st_mode)
                    and info.st_uid == 0
                    and authorization["libvirt_socket_identity"]
                    == [
                        info.st_dev,
                        info.st_ino,
                        info.st_uid,
                        info.st_gid,
                        stat.S_IMODE(info.st_mode),
                    ],
                    "LIBVIRT_SOCKET_REJECTED",
                )
            elif name == "/dev/null":
                makedev = getattr(os, "makedev", None)
                _check(
                    stat.S_ISCHR(info.st_mode)
                    and callable(makedev)
                    and getattr(info, "st_rdev", None) == makedev(1, 3),
                    "VIEW_DEVICE_REJECTED",
                )
            else:
                _check(stat.S_ISREG(info.st_mode), "VIEW_FILE_REJECTED")
                if name in extra:
                    _check(not info.st_mode & 0o111, "VIEW_DATA_MODE_REJECTED")
                    with item.open("rb") as stream:
                        prefix = stream.read(4)
                    _check(
                        not prefix.startswith((b"\x7fELF", b"#!", b"PK\x03\x04")),
                        "VIEW_EXECUTABLE_DATA_REJECTED",
                    )
            seen.add(name)
    _check(seen == expected_paths, "VIEW_PATHS_MISSING")


def _kernel_boundary(authorization: dict[str, Any]) -> None:
    status = _proc("/proc/self/status").splitlines()
    for required in (
        b"NoNewPrivs:\t1",
        b"Seccomp:\t2",
        b"CapEff:\t0000000000000000",
        b"CapPrm:\t0000000000000000",
    ):
        _check(required in status, "KERNEL_CONFINEMENT_REQUIRED")
    mountinfo = _proc("/proc/self/mountinfo")
    _check(_sha(mountinfo) == authorization["mountinfo_sha256"], "MOUNT_VIEW_MISMATCH")
    _mount_view(mountinfo, authorization["libvirt_socket"])
    nss = authorization["namespaces"]
    _check(
        type(nss) is dict
        and set(nss) == {"mnt", "pid", "net"}
        and all(type(number) is int and number > 0 for number in nss.values()),
        "NAMESPACE_PINS_REQUIRED",
    )
    _check(
        all(os.stat(f"/proc/self/ns/{name}").st_ino == inode for name, inode in nss.items()),
        "NAMESPACE_ID_MISMATCH",
    )
    statvfs = getattr(os, "statvfs", None)
    readonly = getattr(os, "ST_RDONLY", 0)
    _check(callable(statvfs) and bool(statvfs("/").f_flag & readonly), "READ_ONLY_VIEW_REQUIRED")
    label = authorization["apparmor_label"]
    _check(
        type(label) is str
        and label.startswith("cryptohunter-observer-preflight-")
        and label.endswith(" (enforce)")
        and _proc("/proc/self/attr/apparmor/current", 4096).strip().decode("ascii") == label,
        "APPARMOR_CONFINEMENT_REQUIRED",
    )


def start(
    root: Path,
    receipt: dict[str, Any],
    contract: dict[str, Any],
    authority: Path,
    authority_sha256: str,
    receipt_sha256: str,
) -> int:
    """Enter the preflight only after separate authorization for this OS view."""
    _check(
        authority.is_absolute() and not authority.is_relative_to(root),
        "LAUNCH_AUTHORITY_PATH_REJECTED",
    )
    interpreter = root / "venv/bin/python3.12"
    _check(
        Path(sys.executable) == interpreter and os.path.samefile("/proc/self/exe", interpreter),
        "LAUNCH_INTERPRETER_MISMATCH",
    )
    raw = contract["secure_read"](authority, limit=1024 * 1024)
    _check(
        contract["sha256_value"](authority_sha256) and _sha(raw) == authority_sha256,
        "LAUNCH_AUTHORIZATION_PIN_MISMATCH",
    )
    authorization = contract["loads"](raw)
    _check(
        set(authorization)
        == {
            "profile",
            "receipt_sha256",
            "approval_sha256",
            "mountinfo_sha256",
            "namespaces",
            "apparmor_label",
            "libvirt_socket",
            "libvirt_socket_identity",
            "libvirt_config_sha256",
            "abi_request_path",
            "aliases",
            "data_paths",
        }
        and authorization["profile"] == LAUNCH_AUTHORIZATION
        and authorization["receipt_sha256"] == receipt_sha256
        and authorization["approval_sha256"] == receipt["approval_sha256"]
        and authorization["libvirt_socket"] in LOCAL_SOCKETS,
        "LAUNCH_AUTHORIZATION_REJECTED",
    )
    _check(dict(os.environ) == COMMAND_ENV and os.getcwd() == "/", "LAUNCH_ENVIRONMENT_REJECTED")
    _kernel_boundary(authorization)
    request_path = authorization["abi_request_path"]
    _check(contract["path_value"](request_path, absolute=True), "ABI_REQUEST_PATH_REJECTED")
    abi_request = Path(request_path)
    _check(
        not abi_request.is_relative_to(root) and abi_request != authority,
        "ABI_REQUEST_PATH_REJECTED",
    )
    request_bytes = contract["secure_read"](abi_request, limit=1024 * 1024)
    _check(_sha(request_bytes) == receipt["abi_request_sha256"], "LAUNCH_ABI_REQUEST_PIN_MISMATCH")
    targets = _abi_link_targets(
        contract["loads"](request_bytes), receipt["base_runtime"]["symlinks"]
    )
    _view(authorization, root, receipt, contract, authority, abi_request, targets)
    _file_descriptors()
    finder = ApprovedImports(root, receipt, contract)
    finder.validate_startup()
    sys.path_importer_cache.clear()
    sys.meta_path[:] = [finder]
    tools = receipt["external_files"]
    sys.addaudithook(lambda event, args: _audit(finder, tools["/usr/bin/qemu-img"], event, args))
    profile = __import__(receipt["entrypoint"], fromlist=["main"])
    return int(profile.main([]))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("verify", "preflight"))
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--receipt-sha256", required=True)
    parser.add_argument("--approval-sha256", required=True)
    parser.add_argument("--launch-authorization", type=Path)
    parser.add_argument("--launch-authorization-sha256")
    args = parser.parse_args(argv)
    try:
        _check(
            sys.platform == "linux"
            and sys.flags.isolated == 1
            and sys.flags.no_site == 1
            and sys.dont_write_bytecode,
            "ISOLATED_LINUX_REQUIRED",
        )
        uid = getattr(os, "geteuid", None)
        _check(callable(uid) and uid() != 0, "UNPRIVILEGED_LAUNCH_REQUIRED")
        _check(args.root.is_absolute(), "ABSOLUTE_ROOT_REQUIRED")
        explicit = (
            args.launch_authorization is not None and args.launch_authorization_sha256 is not None
        )
        _check(
            (args.mode == "preflight" and explicit)
            or (
                args.mode == "verify"
                and args.launch_authorization is None
                and args.launch_authorization_sha256 is None
            ),
            "SEPARATE_LAUNCH_AUTHORIZATION_REQUIRED",
        )
        receipt, contract = verify(args.root, args.receipt_sha256, args.approval_sha256)
        if args.mode == "preflight":
            return start(
                args.root,
                receipt,
                contract,
                args.launch_authorization,
                args.launch_authorization_sha256,
                args.receipt_sha256,
            )
        print(
            '{"status":"STATIC_RECEIPT_MATCH","authorization":"NOT_AUTHORIZED","native_load":"NOT_EXECUTED"}'
        )
        return 0
    except (OSError, ValueError, TypeError, KeyError, ImportError, RuntimeError) as exc:
        # Avoid leaking physical path details to logs.
        print(
            json.dumps(
                {
                    "status": "REFUSED",
                    "reason": str(exc)
                    if isinstance(exc, Refusal)
                    else "OBSERVER_A_EVIDENCE_REJECTED",
                }
            )
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

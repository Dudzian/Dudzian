"""Odczytowa bramka jednorazowej VM; nigdy nie przygotowuje ani nie naprawia hosta.

Zewnętrzne zatwierdzenie operatora wiąże konkretny boot, obraz i snapshot.
Deklaracje operatora i lokalne sondy nie stanowią atestacji hypervisora.
"""

from __future__ import annotations

import json
import os
import platform
import re
import stat
import subprocess
import sys
from pathlib import Path
from typing import Any

DEFAULT_APPROVAL = Path("/etc/cryptohunter/stage9-vm-trial-approval.json")
REQUIRED_PACKAGES = frozenset(
    {
        "systemd",
        "python3.12",
        "python3.12-venv",
        "postgresql-16",
        "postgresql-client-16",
        "postgresql-common",
        "openssl",
    }
)
_VM_TYPES = frozenset(
    {
        "kvm",
        "qemu",
        "vmware",
        "microsoft",
        "oracle",
        "xen",
        "bochs",
        "parallels",
        "bhyve",
        "amazon",
        "zvm",
        "powervm",
        "apple",
        "google",
    }
)
_APPROVAL_FLAGS = frozenset(
    {
        "approved",
        "disposable_vm",
        "isolation_no_host_shares",
        "network_disconnected",
        "no_production_material",
    }
)
_APPROVAL_FIELDS = _APPROVAL_FLAGS | {
    "schema",
    "purpose",
    "machine_id",
    "boot_id",
    "revision",
    "environment_id",
    "vm_image_sha256",
    "vm_snapshot",
    "packages",
}
_UUID = re.compile(r"[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}")
_MAX_APPROVAL = 65536
_CONTAINERS = frozenset(
    {
        "docker",
        "podman",
        "lxc",
        "lxc-libvirt",
        "systemd-nspawn",
        "wsl",
        "proot",
        "pouch",
        "rkt",
        "openvz",
    }
)


class VMGuardError(RuntimeError):
    """Stały kod i bezpieczna przyczyna, bez zawartości wejść ani stderr."""

    code = "VM_ENVIRONMENT_REFUSED"

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


def _query(arguments: list[str]) -> tuple[int, str] | None:
    try:
        result = subprocess.run(
            arguments,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=15,
            check=False,
            text=True,
            encoding="utf-8",
            errors="strict",
            cwd="/",
            env={"PATH": "/usr/sbin:/usr/bin:/sbin:/bin", "LANG": "C", "LC_ALL": "C"},
        )
        if len(result.stdout) > 1024 * 1024:
            return None
        return result.returncode, result.stdout.strip()
    except (OSError, subprocess.TimeoutExpired, UnicodeError):
        return None


def _public_text(path: str, limit: int = 65536) -> str | None:
    """Publiczne metadane; /etc/os-release może być symlinkiem dystrybucji."""
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
        try:
            if not stat.S_ISREG(os.fstat(fd).st_mode):
                return None
            chunks: list[bytes] = []
            length = 0
            while length <= limit:
                block = os.read(fd, min(65536, limit + 1 - length))
                if not block:
                    return b"".join(chunks).decode("utf-8", errors="strict").strip()
                chunks.append(block)
                length += len(block)
            return None
        finally:
            os.close(fd)
    except (OSError, UnicodeError, ValueError):
        return None


def _os_identity(raw: str | None) -> tuple[str | None, str | None]:
    if raw is None:
        return None, None
    fields: dict[str, str] = {}
    for line in raw.splitlines():
        if "=" not in line or line.startswith("#"):
            continue
        key, value = line.split("=", 1)
        if key not in {"ID", "VERSION_ID"}:
            continue
        if key in fields:
            return None, None
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        if re.fullmatch(r"[a-zA-Z0-9._-]+", value) is None:
            return None, None
        fields[key] = value
    return fields.get("ID"), fields.get("VERSION_ID")


def _packages(result: tuple[int, str] | None) -> dict[str, str] | None:
    if result is None or result[0] != 0:
        return None
    observed: dict[str, str] = {}
    for line in result[1].splitlines():
        parts = line.split("\t")
        if len(parts) != 3:
            return None
        name, version, status = parts
        if (
            name not in REQUIRED_PACKAGES
            or name in observed
            or status.strip() != "ii"
            or not version
            or len(version) > 256
            or re.fullmatch(r"[0-9A-Za-z.+:~_-]+", version) is None
        ):
            return None
        observed[name] = version
    return observed if set(observed) == REQUIRED_PACKAGES else None


def _shared_mounts(raw: str | None) -> list[str] | None:
    if not raw:
        return None
    shared: set[str] = set()
    for line in raw.splitlines():
        parts = line.split(" - ")
        if len(parts) != 2 or len(parts[0].split()) < 6 or len(parts[1].split()) < 3:
            return None
        filesystem = parts[1].split()[0]
        if (
            filesystem in {"virtiofs", "9p", "vboxsf", "cifs"}
            or filesystem.startswith("fuse.vmhgfs")
            or filesystem.startswith("nfs")
        ):
            shared.add(filesystem)
    return sorted(shared)


def _json_rows(result: tuple[int, str] | None) -> list[dict[str, Any]] | None:
    if result is None or result[0] != 0:
        return None
    try:
        value = json.loads(result[1])
    except (ValueError, TypeError):
        return None
    if type(value) is not list or any(type(item) is not dict for item in value):
        return None
    return value


def _network_disconnected(
    links: tuple[int, str] | None,
    addresses: tuple[int, str] | None,
    routes: tuple[int, str] | None,
) -> bool | None:
    link_rows, address_rows, route_rows = map(_json_rows, (links, addresses, routes))
    if (
        link_rows is None
        or address_rows is None
        or route_rows is None
        or not link_rows
        or not address_rows
        or not any(row.get("ifname") == "lo" for row in link_rows)
        or not any(row.get("ifname") == "lo" for row in address_rows)
    ):
        return None
    for row in link_rows:
        if (
            type(row.get("ifname")) is not str
            or type(row.get("flags")) is not list
            or any(type(flag) is not str for flag in row["flags"])
        ):
            return None
        if row["ifname"] != "lo" and ("UP" in row["flags"] or row.get("operstate") == "UP"):
            return False
    for row in address_rows:
        if type(row.get("ifname")) is not str or type(row.get("addr_info")) is not list:
            return None
        if row["ifname"] != "lo" and row["addr_info"]:
            return False
    return all(row.get("dev") == "lo" for row in route_rows)


def inspect_environment() -> dict[str, Any]:
    """Zwróć wyłącznie publiczne fakty; brak zapisów i operacji systemowych."""
    facts: dict[str, Any] = {
        "os_id": None,
        "os_version": None,
        "architecture": (
            architecture
            if (architecture := platform.machine())
            in {"x86_64", "aarch64", "i386", "i686", "armv7l"}
            else None
        ),
        "python_executable": "/usr/bin/python3.12",
        "python_version": None,
        "pid1": None,
        "virtualization": None,
        "container": None,
        "container_markers": None,
        "shared_mounts": None,
        "network_disconnected": None,
        "packages": None,
        "machine_id": None,
        "boot_id": None,
        "observation_complete": False,
    }
    if not sys.platform.startswith("linux"):
        return facts
    facts["os_id"], facts["os_version"] = _os_identity(_public_text("/etc/os-release"))
    facts["pid1"] = "systemd" if _public_text("/proc/1/comm") == "systemd" else None
    machine_id = _public_text("/etc/machine-id")
    boot_id = _public_text("/proc/sys/kernel/random/boot_id")
    if machine_id is not None and re.fullmatch(r"[0-9a-f]{32}", machine_id):
        facts["machine_id"] = machine_id
    if boot_id is not None and _UUID.fullmatch(boot_id):
        facts["boot_id"] = boot_id
    facts["shared_mounts"] = _shared_mounts(_public_text("/proc/self/mountinfo", 1024 * 1024))
    markers: list[str] = []
    try:
        for marker in ("/.dockerenv", "/run/.containerenv"):
            try:
                os.lstat(marker)
            except FileNotFoundError:
                continue
            markers.append(marker)
        facts["container_markers"] = markers
    except OSError:
        pass
    vm = _query(["/usr/bin/systemd-detect-virt", "--vm"])
    container = _query(["/usr/bin/systemd-detect-virt", "--container"])
    if vm is not None and ((vm[0] == 0 and vm[1] in _VM_TYPES) or vm == (1, "none")):
        facts["virtualization"] = vm[1]
    if container is not None and (
        container == (1, "none") or (container[0] == 0 and container[1] in _CONTAINERS)
    ):
        facts["container"] = container[1]
    python = _query(["/usr/bin/python3.12", "-I", "-s", "-B", "--version"])
    if python is not None and python[0] == 0 and re.fullmatch(r"Python 3\.12\.[0-9]+", python[1]):
        facts["python_version"] = python[1]
    facts["packages"] = _packages(
        _query(
            [
                "/usr/bin/dpkg-query",
                "-W",
                "-f=${binary:Package}\t${Version}\t${db:Status-Abbrev}\n",
                "--",
                *sorted(REQUIRED_PACKAGES),
            ]
        )
    )
    facts["network_disconnected"] = _network_disconnected(
        _query(["/usr/sbin/ip", "-j", "link", "show"]),
        _query(["/usr/sbin/ip", "-j", "address", "show"]),
        _query(["/usr/sbin/ip", "-j", "route", "show", "table", "all"]),
    )
    facts["observation_complete"] = all(value is not None for value in facts.values())
    return facts


def _protected(value: os.stat_result, *, directory: bool) -> bool:
    return (
        (stat.S_ISDIR(value.st_mode) if directory else stat.S_ISREG(value.st_mode))
        and value.st_uid == 0
        and not value.st_mode & 0o022
        and (directory or value.st_nlink == 1)
    )


def _fingerprint(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
        value.st_mode,
        value.st_uid,
        value.st_gid,
        value.st_nlink,
    )


def _read_approval(path: Path) -> bytes:
    if not path.is_absolute() or path == Path("/") or ".." in path.parts:
        raise VMGuardError("APPROVAL_INVALID")
    fd = -1
    try:
        fd = os.open("/", os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        if not _protected(os.fstat(fd), directory=True):
            raise VMGuardError("APPROVAL_INVALID")
        for name in path.parts[1:-1]:
            child = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            os.close(fd)
            fd = child
            if not _protected(os.fstat(fd), directory=True):
                raise VMGuardError("APPROVAL_INVALID")
        child = os.open(path.name, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW, dir_fd=fd)
        os.close(fd)
        fd = child
        before = os.fstat(fd)
        if not _protected(before, directory=False):
            raise VMGuardError("APPROVAL_INVALID")
        data = bytearray()
        while len(data) <= _MAX_APPROVAL:
            block = os.read(fd, min(65536, _MAX_APPROVAL + 1 - len(data)))
            if not block:
                if (
                    _fingerprint(before) != _fingerprint(os.fstat(fd))
                    or len(data) != before.st_size
                ):
                    raise VMGuardError("APPROVAL_CHANGED_DURING_READ")
                return bytes(data)
            data.extend(block)
        raise VMGuardError("APPROVAL_INVALID")
    except (OSError, ValueError) as exc:
        raise VMGuardError("APPROVAL_INVALID") from exc
    finally:
        if fd >= 0:
            os.close(fd)


def _unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise VMGuardError("APPROVAL_INVALID")
        result[key] = value
    return result


def _approval(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(_read_approval(path), object_pairs_hook=_unique)
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise VMGuardError("APPROVAL_INVALID") from exc
    if (
        type(value) is not dict
        or set(value) != _APPROVAL_FIELDS
        or type(value["schema"]) is not int
        or value["schema"] != 1
        or value["purpose"] != "TEST_ONLY"
        or any(value[key] is not True for key in _APPROVAL_FLAGS)
    ):
        raise VMGuardError("APPROVAL_INVALID")
    patterns = {
        "machine_id": r"[0-9a-f]{32}",
        "boot_id": _UUID.pattern,
        "revision": r"[0-9a-f]{40}",
        "environment_id": r"[a-z0-9][a-z0-9._-]{0,95}",
        "vm_image_sha256": r"[0-9a-f]{64}",
        "vm_snapshot": r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}",
    }
    if any(
        type(value[key]) is not str or re.fullmatch(pattern, value[key]) is None
        for key, pattern in patterns.items()
    ):
        raise VMGuardError("APPROVAL_INVALID")
    packages = value["packages"]
    if (
        type(packages) is not dict
        or set(packages) != REQUIRED_PACKAGES
        or any(
            type(version) is not str
            or not version
            or len(version) > 256
            or re.fullmatch(r"[0-9A-Za-z.+:~_-]+", version) is None
            for version in packages.values()
        )
    ):
        raise VMGuardError("APPROVAL_INVALID")
    return value


def require_approved_vm(
    approval_path: Path = DEFAULT_APPROVAL, *, revision: str, environment_id: str
) -> dict[str, Any]:
    """Odmów przed mutacjami, jeśli dowolna obserwacja/bramka jest niedostępna."""
    if not sys.platform.startswith("linux") or os.geteuid() != 0:
        raise VMGuardError("EXECUTION_REQUIRES_VM_ADMINISTRATOR")
    approval = _approval(approval_path)
    facts = inspect_environment()
    if (
        facts["observation_complete"] is not True
        or facts["os_id"] != "ubuntu"
        or facts["os_version"] != "24.04"
        or facts["architecture"] != "x86_64"
        or facts["python_executable"] != "/usr/bin/python3.12"
        or type(facts["python_version"]) is not str
        or re.fullmatch(r"Python 3\.12\.[0-9]+", facts["python_version"]) is None
        or facts["pid1"] != "systemd"
        or facts["virtualization"] not in _VM_TYPES
        or facts["container"] != "none"
        or facts["container_markers"] != []
        or facts["shared_mounts"] != []
        or facts["network_disconnected"] is not True
    ):
        raise VMGuardError("ENVIRONMENT_UNSAFE_OR_UNAVAILABLE")
    if (
        approval["machine_id"] != facts["machine_id"]
        or approval["boot_id"] != facts["boot_id"]
        or approval["revision"] != revision
        or approval["environment_id"] != environment_id
    ):
        raise VMGuardError("APPROVAL_BINDING_MISMATCH")
    if approval["packages"] != facts["packages"]:
        raise VMGuardError("PACKAGE_BASELINE_MISMATCH")
    return {"approval": approval, "environment": facts}

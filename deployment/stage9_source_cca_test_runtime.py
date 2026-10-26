"""Owned TEST ONLY Linux resources, never a production installer or attestation.

An interrupted journal is an inspection aid, not authority to adopt resources.
Only receipts and pidfds captured by this instance authorize cleanup.
"""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
from itertools import islice
import json
import os
from pathlib import Path
import platform
import re
import signal
import stat
import subprocess
import sys
import time
from typing import Any, Iterator, NoReturn, cast

from deployment.stage9_source_cca_test_package import HarnessRefusal
from deployment.stage9_trusted_installation_contract import parse_manifest
from deployment.stage9_native_linux_cgroup_membership_observation import _parse_proc_cgroup_v2
from deployment.stage9_native_linux_process_identity_observation import _read_proc_file
from deployment.stage9_native_linux_source_cca_preflight import (
    _PROPOSED,
    _KNOWN_REFERENCE_AUTHORITIES,
)
from deployment.stage9_trusted_installation_runtime import _unit
from scripts.qualify_stage41b_signer_isolation import (
    QualificationBlocked,
    _defer_interrupts,
    _process_finished,
    _process_identity,
    _process_lifetime,
    inspect_unit,
    unit_is_unused,
)

_ENV = {"PATH": "/usr/sbin:/usr/bin:/sbin:/bin", "LANG": "C", "LC_ALL": "C"}
_PROGRAMS = {
    name: Path(
        "/usr/sbin" if name in {"useradd", "groupadd", "userdel", "groupdel"} else "/usr/bin"
    )
    / name
    for name in (
        "systemctl",
        "getent",
        "systemd-detect-virt",
        "useradd",
        "groupadd",
        "userdel",
        "groupdel",
    )
}
_MAX_OUTPUT = 1024 * 1024


def _refuse(code: str) -> NoReturn:
    raise HarnessRefusal(code)


def _protected(path: Path, *, directory: bool) -> os.stat_result:
    info = path.lstat()
    kind = stat.S_ISDIR if directory else stat.S_ISREG
    if not kind(info.st_mode) or info.st_uid != 0 or info.st_gid != 0 or info.st_mode & 0o7022:
        _refuse("UNPROTECTED_HOST_PATH")
    if not directory and info.st_nlink != 1:
        _refuse("UNSAFE_HARDLINK")
    return info


def _program(name: str) -> Path:
    program = _PROGRAMS.get(name)
    if program is None:
        _refuse("HOST_PROGRAM_NOT_ALLOWLISTED")
    assert program is not None
    target = program.resolve(strict=True)
    _protected(target, directory=False)
    for parent in target.parents:
        _protected(parent, directory=True)
    # Ubuntu's /sbin and nologin aliases may be root-owned symlinks. No caller
    # controlled search path or executable pathname is accepted.
    for parent in program.parents:
        metadata = parent.lstat()
        if stat.S_ISLNK(metadata.st_mode):
            if metadata.st_uid != 0:
                _refuse("UNPROTECTED_HOST_PATH")
        else:
            _protected(parent, directory=True)
    return program


def _command(name: str, *arguments: str, missing: bool = False) -> tuple[int, str]:
    completed = subprocess.run(
        [str(_program(name)), *arguments],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=15,
        env=_ENV,
    )
    if len(completed.stdout) > _MAX_OUTPUT or len(completed.stderr) > _MAX_OUTPUT:
        _refuse("HOST_OUTPUT_LIMIT")
    if completed.returncode and not (missing and name == "getent" and completed.returncode == 2):
        _refuse("HOST_COMMAND_FAILED")
    try:
        # NSS descriptions are UTF-8 even under LC_ALL=C on Ubuntu. Keep every
        # other host command ASCII-only; field policy belongs to _nss below.
        encoding = "utf-8" if name == "getent" else "ascii"
        return completed.returncode, completed.stdout.decode(encoding, errors="strict")
    except UnicodeError:
        _refuse("HOST_OBSERVATION_INVALID")
    raise AssertionError("unreachable")


def _nss(database: str, key: str | None = None) -> list[list[str]]:
    if database not in {"passwd", "group"}:
        _refuse("NSS_OBSERVATION_INVALID")
    args = (database,) if key is None else (database, key)
    code, raw = _command("getent", *args, missing=key is not None)
    if code == 2:
        if raw:
            _refuse("NSS_OBSERVATION_INVALID")
        return []
    # Only LF delimits records. splitlines() also accepts Unicode separators,
    # which must never manufacture accounts from a descriptive field.
    lines = raw.removesuffix("\n").split("\n") if raw else []
    fields = [line.split(":") for line in lines]
    width = 7 if database == "passwd" else 4
    names: set[str] = set()

    def account(value: str) -> bool:
        return re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.-]*\$?", value) is not None

    def identifier(value: str) -> bool:
        return re.fullmatch(r"(?:0|[1-9][0-9]{0,9})", value) is not None and int(value) < 2**32 - 1

    for row in fields:
        if len(row) != width or any(not field.isprintable() for field in row if field):
            _refuse("NSS_OBSERVATION_INVALID")
        if not account(row[0]) or row[0] in names or not identifier(row[2]):
            _refuse("NSS_OBSERVATION_INVALID")
        names.add(row[0])
        # Only passwd's GECOS field may contain international text. Identity,
        # password placeholders, paths and membership syntax remain ASCII.
        if any(
            not field.isascii()
            for index, field in enumerate(row)
            if not (width == 7 and index == 4)
        ):
            _refuse("NSS_OBSERVATION_INVALID")
        if width == 7:
            if not identifier(row[3]) or any(not row[index].startswith("/") for index in (5, 6)):
                _refuse("NSS_OBSERVATION_INVALID")
        elif row[3] and any(not account(member) for member in row[3].split(",")):
            _refuse("NSS_OBSERVATION_INVALID")
    if key is not None and (len(fields) != 1 or (fields[0][0] != key and fields[0][2] != key)):
        _refuse("NSS_OBSERVATION_INVALID")
    return fields


def _absent(path: Path) -> bool:
    try:
        path.lstat()
    except FileNotFoundError:
        return True
    except OSError:
        _refuse("PATH_OBSERVATION_UNAVAILABLE")
    return False


def _role_paths(plan: dict[str, Any]) -> list[Path]:
    result = [Path(plan["root"])]
    for role in plan["roles"]:
        name = role["unit"]
        result.extend(
            (
                Path(role["unit_path"]),
                Path(role["unit_path"] + ".d"),
                Path("/etc/systemd/system") / name,
                Path("/etc/systemd/system") / (name + ".d"),
                Path("/usr/lib/systemd/system") / name,
                Path("/run/systemd/transient") / name,
                Path("/run/credentials") / name,
                Path("/var/mail") / role["principal"],
                Path("/var/spool/mail") / role["principal"],
                Path("/sys/fs/cgroup") / role["control_group"].lstrip("/"),
            )
        )
    return result


def _processes(plan: dict[str, Any]) -> list[int]:
    """Strict executable sweep, used only by the administrative observer.

    Keep this independent sweep in addition to scoped metadata classification:
    an executable-only alias outside reserved credentials/cgroups must be seen.
    """
    uids = {role["uid"] for role in plan["roles"]}
    root = plan["root"]
    found = []
    self_seen = False
    for item in Path("/proc").iterdir():
        if not item.name.isdigit():
            continue
        if item.name == str(os.getpid()):
            self_seen = True
        try:
            raw = (item / "status").read_text(encoding="ascii")
            fields = dict(line.split(":", 1) for line in raw.splitlines() if ":" in line)
            values = fields.get("Uid", "").split()
            if len(values) != 4 or any(not value.isdigit() for value in values):
                _refuse("PROCESS_OBSERVATION_UNAVAILABLE")
            if any(int(value) in uids for value in values):
                found.append(int(item.name))
                continue
            try:
                executable = os.readlink(item / "exe")
            except FileNotFoundError:
                # Kernel threads have no executable; a UID observation above
                # still covers test principals. A vanished process is harmless.
                continue
            if executable == root or executable.startswith(root + "/"):
                found.append(int(item.name))
        except FileNotFoundError:
            continue
        except (OSError, UnicodeError, ValueError):
            _refuse("PROCESS_OBSERVATION_UNAVAILABLE")
    if not self_seen:
        _refuse("PROCESS_OBSERVATION_UNAVAILABLE")
    return found


def _procfs_visibility() -> bool:
    """A hidden PID set cannot support even a scoped absence observation."""
    with Path("/proc/self/mountinfo").open("rb") as stream:
        raw = stream.read(1024 * 1024 + 1)
    if not raw or len(raw) > 1024 * 1024:
        return False
    mounts = []
    for line in raw.splitlines():
        before, separator, after = line.partition(b" - ")
        fields, filesystem = before.split(), after.split()
        if len(fields) >= 6 and fields[4] == b"/proc":
            if (
                not separator
                or fields[3] != b"/"
                or len(filesystem) != 3
                or filesystem[0] != b"proc"
            ):
                return False
            mounts.append(fields[5].split(b",") + filesystem[2].split(b","))
    return len(mounts) == 1 and all(
        option == b"hidepid=0" or not option.startswith(b"hidepid") for option in mounts[0]
    )


def _scan_stat(raw: bytes, pid: int) -> int:
    """Classify host processes, including PID 1 and unrelated zombies.

    Service qualification's parser rejects zombies. This narrower scanner must
    retain test zombies as residue while allowing stable unrelated ones.
    """
    header, separator, suffix = raw.rpartition(b") ")
    fields = suffix.split()
    if (
        not separator
        or not header.startswith(str(pid).encode("ascii") + b" (")
        or len(fields) < 20
        or fields[0] not in (b"R", b"S", b"D", b"Z", b"T", b"t", b"W", b"X", b"x", b"K", b"P", b"I")
        or not fields[19].isdigit()
        or not 0 <= int(fields[19]) < 2**64
    ):
        _refuse("PROCESS_PID_OR_IDENTITY_CHANGED")
    return int(fields[19])


def _scan_identity(raw: bytes, pid: int) -> tuple[tuple[int, ...], ...]:
    required = {b"Pid", b"Tgid", b"Uid", b"Gid", b"Groups"}
    fields = {}
    for line in raw.splitlines():
        key, separator, value = line.partition(b":")
        if key in required:
            if not separator or key in fields:
                _refuse("PROCESS_IDENTITY_METADATA_INVALID")
            fields[key] = value.split()
    if set(fields) != required or any(
        not value.isdigit() or len(value) > 10 for values in fields.values() for value in values
    ):
        _refuse("PROCESS_IDENTITY_METADATA_INVALID")
    numbers = {key: tuple(int(value) for value in values) for key, values in fields.items()}
    if (
        numbers[b"Pid"] != (pid,)
        or numbers[b"Tgid"] != (pid,)
        or len(numbers[b"Uid"]) != 4
        or len(numbers[b"Gid"]) != 4
        or any(value >= 2**32 - 1 for values in numbers.values() for value in values)
    ):
        _refuse("PROCESS_IDENTITY_METADATA_INVALID")
    return numbers[b"Uid"], numbers[b"Gid"], numbers[b"Groups"]


def _scan_cgroup(raw: bytes) -> str:
    # The service parser intentionally excludes the host's root cgroup. A host
    # scan accepts its canonical exact membership, without claiming exe absence.
    if raw == b"0::/\n":
        return "/"
    try:
        return cast(str, _parse_proc_cgroup_v2(raw))
    except ValueError:
        _refuse("PROCESS_CGROUP_AMBIGUOUS")


def rootless_process_observation(plan: dict[str, Any]) -> dict[str, Any]:
    """Observe visible UID/GID/cgroup scope; never claim a full executable scan.

    No executable read is needed to find a reserved identity or test cgroup.
    Exclusion requires stable, complete metadata. An unreadable or changed
    process is uncertainty, not evidence that a test process is absent.
    """
    found: set[int] = set()
    uncertain: set[int] = set()
    reasons: set[str] = set()
    excluded = 0
    self_seen = False
    uids = {role["uid"] for role in plan["roles"]}
    gids = {role["gid"] for role in plan["roles"]}
    cgroups = tuple(role["control_group"] for role in plan["roles"])
    try:
        if not _procfs_visibility():
            reasons.add("PROCFS_VISIBILITY_UNCERTAIN")
        for item in Path("/proc").iterdir():
            if not item.name.isascii() or not item.name.isdigit():
                continue
            pid = int(item.name)
            if not 0 < pid < 2**31 or str(pid) != item.name:
                reasons.add("PROCESS_PID_OR_IDENTITY_CHANGED")
                continue
            self_seen |= pid == os.getpid()
            descriptor = -1
            try:
                descriptor = os.open(
                    item, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
                )
                retained = os.fstat(descriptor)
                before = _scan_stat(_read_proc_file(descriptor, "stat"), pid)
                identity = _scan_identity(_read_proc_file(descriptor, "status"), pid)
                relevant = bool(
                    uids.intersection(identity[0])
                    or gids.intersection((*identity[1], *identity[2]))
                )
                if relevant:
                    found.add(pid)
                cgroup = _scan_cgroup(_read_proc_file(descriptor, "cgroup"))
                relevant |= any(
                    cgroup == group or cgroup.startswith(group + "/") for group in cgroups
                )
                if relevant:
                    found.add(pid)
                after_identity = _scan_identity(_read_proc_file(descriptor, "status"), pid)
                after_cgroup = _scan_cgroup(_read_proc_file(descriptor, "cgroup"))
                after = _scan_stat(_read_proc_file(descriptor, "stat"), pid)
                current = item.stat(follow_symlinks=False)
                if (
                    before != after
                    or identity != after_identity
                    or cgroup != after_cgroup
                    or (retained.st_dev, retained.st_ino) != (current.st_dev, current.st_ino)
                ):
                    _refuse("PROCESS_PID_OR_IDENTITY_CHANGED")
                if not relevant:
                    excluded += 1
            except FileNotFoundError:
                # A missing status/exe is not by itself a vanished process.
                # Only a vanished directory can be ignored; known test residue
                # remains recorded even if it exited during this observation.
                if not _absent(item):
                    uncertain.add(pid)
                    reasons.add("PROCESS_METADATA_UNAVAILABLE")
            except HarnessRefusal as exc:
                uncertain.add(pid)
                reasons.add(exc.code)
            except (OSError, ValueError, UnicodeError):
                uncertain.add(pid)
                reasons.add("PROCESS_METADATA_UNAVAILABLE")
            finally:
                if descriptor >= 0:
                    os.close(descriptor)
    except (OSError, ValueError, UnicodeError):
        reasons.add("PROCESS_ENUMERATION_UNAVAILABLE")
    if not self_seen:
        reasons.add("PROCFS_VISIBILITY_UNCERTAIN")
    return {
        "scope": "VISIBLE_UID_GID_CGROUP",
        "state": "UNCERTAIN"
        if reasons
        else "RESIDUAL_PROCESS_FOUND"
        if found
        else "SCOPED_ABSENCE_OBSERVED",
        "matched_pids": sorted(found),
        "uncertain_pids": sorted(uncertain),
        "uncertainty": sorted(reasons),
        "complete": not reasons,
        "full_confirmation": False,
        "executable_scan_complete": False,
        "excluded_processes": excluded,
    }


def process_observation(plan: dict[str, Any], *, administrative: bool = False) -> dict[str, Any]:
    """Select an explicit observation scope; privilege never upgrades it."""
    if not administrative:
        return rootless_process_observation(plan)
    result: dict[str, Any] = {
        "scope": "ADMINISTRATIVE_FULL_PROCFS",
        "state": "UNCERTAIN",
        "matched_pids": [],
        "uncertain_pids": [],
        "uncertainty": ["ADMINISTRATIVE_PROCESS_OBSERVATION_REQUIRED"],
        "complete": False,
        "full_confirmation": False,
        "executable_scan_complete": False,
        "excluded_processes": 0,
    }
    if sys.platform != "linux" or os.geteuid() != 0:
        return result
    scoped = rootless_process_observation(plan)
    result.update(scoped)
    result["scope"] = "ADMINISTRATIVE_FULL_PROCFS"
    try:
        result["matched_pids"] = sorted(set(scoped["matched_pids"]) | set(_processes(plan)))
        result["executable_scan_complete"] = True
    except (OSError, ValueError, UnicodeError):
        result["uncertainty"] = sorted(
            set(scoped["uncertainty"]) | {"PROCESS_EXECUTABLE_SCAN_UNCERTAIN"}
        )
        result["complete"] = False
    result["full_confirmation"] = result["complete"] and not result["matched_pids"]
    result["state"] = (
        "UNCERTAIN"
        if not result["complete"]
        else "RESIDUAL_PROCESS_FOUND"
        if result["matched_pids"]
        else "FULL_ABSENCE_CONFIRMED"
    )
    return result


def _subordinate_absent(plan: dict[str, Any]) -> bool:
    keys = {str(role[key]) for role in plan["roles"] for key in ("principal", "uid", "gid")}
    for filename in ("/etc/subuid", "/etc/subgid"):
        try:
            raw = Path(filename).read_text(encoding="ascii")
        except FileNotFoundError:
            continue
        for line in raw.splitlines():
            if not line or line.startswith("#"):
                continue
            fields = line.split(":")
            if (
                len(fields) != 3
                or not fields[0]
                or not fields[1].isdigit()
                or not fields[2].isdigit()
            ):
                _refuse("SUBORDINATE_IDS_OBSERVATION_INVALID")
            if fields[0] in keys:
                return False
    return True


def _collisions(plan: dict[str, Any]) -> dict[str, bool]:
    checks: dict[str, bool] = {}
    _protected(Path("/sys/fs/cgroup"), directory=True)
    _protected(Path("/sys/fs/cgroup/cgroup.controllers"), directory=False)
    checks["subordinate_ids_absent"] = _subordinate_absent(plan)
    for role in plan["roles"]:
        name = role["principal"]
        for database, key in (
            ("passwd", name),
            ("group", name),
            ("passwd", str(role["uid"])),
            ("group", str(role["gid"])),
        ):
            checks[f"nss:{database}:{key}"] = not _nss(database, key)
        checks["unit:" + role["unit"]] = unit_is_unused(inspect_unit(role["unit"]))
    for path in _role_paths(plan):
        checks["path:" + str(path)] = _absent(path)
    return checks


def _stage41b_diagnostics() -> dict[str, Any]:
    """Bounded metadata only. Never interprets remnants as coexistence authority.

    No file contents, symlink targets, installed code or process command lines
    are read. Unavailable observations stay explicit and cannot clear a gate.
    """
    result: dict[str, Any] = {
        "read_only": True,
        "coexistence": "BLOCKED",
        "reason": "STAGE41B_ISOLATION_NOT_ESTABLISHED",
        "process_coverage": "NOT_ASSESSED_FOR_STAGE41B",
        "paths": {},
        "accounts": {},
        "units": {},
    }
    for name in ("/opt", "/opt/cryptohunter-stage41b"):
        try:
            info = Path(name).lstat()
            result["paths"][name] = {
                "state": "PRESENT",
                "uid": info.st_uid,
                "gid": info.st_gid,
                "mode": oct(stat.S_IMODE(info.st_mode)),
                "directory": stat.S_ISDIR(info.st_mode),
                "symlink": stat.S_ISLNK(info.st_mode),
            }
        except FileNotFoundError:
            result["paths"][name] = {"state": "ABSENT"}
        except OSError:
            result["paths"][name] = {"state": "UNAVAILABLE"}
    for database in ("passwd", "group"):
        try:
            rows = _nss(database)
            matches = [row[0] for row in rows if row[0].startswith("chs41b")]
            result["accounts"][database] = {"names": matches[:128], "truncated": len(matches) > 128}
        except Exception:
            result["accounts"][database] = {"state": "UNAVAILABLE"}
    for listing in ("list-unit-files", "list-units"):
        try:
            _, raw = _command("systemctl", listing, "--all", "--plain", "--no-pager", "--no-legend")
            matches = []
            for line in raw.splitlines():
                columns = line.split()
                if columns and (
                    "cryptohunter" in columns[0].lower() or "stage41b" in columns[0].lower()
                ):
                    if re.fullmatch(r"[A-Za-z0-9_.@:\\-]+", columns[0]) is None:
                        _refuse("UNIT_OBSERVATION_INVALID")
                    matches.append(columns[0])
            result["units"][listing] = {"names": matches[:128], "truncated": len(matches) > 128}
        except Exception:
            result["units"][listing] = {"state": "UNAVAILABLE"}
    result["runtime_paths"] = {}
    for pattern in ("ch-stage41b-*", "cryptohunter-stage41b-*"):
        try:
            paths = list(islice(Path("/run").glob(pattern), 129))
            result["runtime_paths"][pattern] = {
                "count_up_to_limit": len(paths),
                "truncated": len(paths) > 128,
            }
        except OSError:
            result["runtime_paths"][pattern] = {"state": "UNAVAILABLE"}
    return result


def preflight(plan: dict[str, Any], *, administrative: bool = False) -> dict[str, Any]:
    """Read-only host observations; an unavailable fact is a blocker."""
    checks: dict[str, Any] = {}
    blockers: list[str] = []

    def check(name: str, probe: Any) -> None:
        try:
            value = probe()
            checks[name] = value
            if value is not True:
                blockers.append(name)
        except (Exception,):
            checks[name] = False
            blockers.append(name)

    check("linux", lambda: sys.platform == "linux")
    check(
        "ubuntu_24_04",
        lambda: platform.freedesktop_os_release().get("ID") == "ubuntu"
        and platform.freedesktop_os_release().get("VERSION_ID") == "24.04",
    )
    check(
        "systemd_pid1",
        lambda: Path("/proc/1/comm").read_text(encoding="ascii").strip() == "systemd",
    )
    check(
        "physical_host",
        lambda: subprocess.run(
            [str(_program("systemd-detect-virt")), "--quiet"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=5,
            check=False,
            env=_ENV,
        ).returncode
        == 1,
    )
    check(
        "dell_latitude_5490",
        lambda: "Dell" in Path("/sys/class/dmi/id/sys_vendor").read_text(encoding="ascii")
        and Path("/sys/class/dmi/id/product_name").read_text(encoding="ascii").strip()
        == "Latitude 5490",
    )
    check("cgroup_v2", lambda: Path("/sys/fs/cgroup/cgroup.controllers").is_file())
    check("pidfd", lambda: hasattr(os, "pidfd_open") and hasattr(signal, "pidfd_send_signal"))
    for name in ("net", "mnt", "ipc"):
        check("namespace:" + name, lambda name=name: os.stat("/proc/self/ns/" + name).st_ino > 0)
    for name in _PROGRAMS:
        check("program:" + name, lambda name=name: bool(_program(name)))
    for path in ("/run", "/run/systemd/system", "/sys/fs/cgroup"):
        check("protected:" + path, lambda path=path: bool(_protected(Path(path), directory=True)))
    try:
        for name, value in _collisions(plan).items():
            checks[name] = value
            if not value:
                blockers.append(name)
    except Exception:
        checks["resource_observation"] = False
        blockers.append("resource_observation")
    check(
        "no_prior_test_paths",
        lambda: not list(Path("/run").glob("ch-s9-test-*"))
        and not list(Path("/run/systemd/system").glob("ch-s9t-*")),
    )
    check(
        "no_prior_test_accounts",
        lambda: not any(
            row[0].startswith("chs9t_")
            for database in ("passwd", "group")
            for row in _nss(database)
        ),
    )
    check(
        "no_other_installations",
        lambda: all(
            _absent(Path(path))
            for path in (
                "/etc/cryptohunter",
                "/var/lib/cryptohunter",
                "/opt/cryptohunter",
                "/opt/cryptohunter-stage41b",
                "/run/cryptohunter",
            )
        ),
    )

    for principal in [row[1] for row in _PROPOSED] + [
        row[1] for row in _KNOWN_REFERENCE_AUTHORITIES
    ]:
        check(
            "known_account_absent:" + principal,
            lambda principal=principal: not _nss("passwd", principal)
            and not _nss("group", principal),
        )
    check(
        "no_cryptohunter_units",
        lambda: not any(
            "cryptohunter" in line.lower()
            for line in _command("systemctl", "list-unit-files", "--no-pager", "--no-legend")[
                1
            ].splitlines()
        ),
    )
    processes = process_observation(plan, administrative=administrative)
    check(
        "no_test_processes", lambda: processes["complete"] is True and not processes["matched_pids"]
    )
    if administrative:
        check("full_process_confirmation", lambda: processes["full_confirmation"] is True)
    return {
        "ready": not blockers,
        "checks": checks,
        "blockers": blockers,
        "kernel": platform.release(),
        "read_only": True,
        "process_observation": processes,
        "stage41b_diagnostics": _stage41b_diagnostics(),
        "administrative_preflight_required": not administrative,
        "physical_test_status": "PHYSICAL_TEST_PENDING",
        "installation_discovery_scope": "CONVENTIONAL_PATHS_AND_KNOWN_ACCOUNTS_UNITS",
    }


def inspect_cleanup(plan: dict[str, Any], *, administrative: bool = False) -> dict[str, Any]:
    """Independent absence checks. Never consumes cleanup's success claim."""
    checks: dict[str, Any] = {}
    blockers: list[str] = []
    resource_uncertain = False
    try:
        checks.update(_collisions(plan))
    except Exception:
        checks["resource_observation"] = False
        resource_uncertain = True
    processes = process_observation(plan, administrative=administrative)
    checks["processes_absent"] = processes["complete"] is True and not processes["matched_pids"]
    blockers.extend(name for name, value in checks.items() if value is not True)
    scope_clear = not blockers
    full = administrative and processes["full_confirmation"] is True
    if not administrative:
        blockers.append("ADMINISTRATIVE_INSPECTION_REQUIRED")
    return {
        "clean": scope_clear and full,
        "scope_clear": scope_clear,
        "full_confirmation": scope_clear and full,
        "state": "FULL_ABSENCE_CONFIRMED"
        if scope_clear and full
        else "SCOPED_ABSENCE_OBSERVED"
        if scope_clear
        else "UNCERTAIN"
        if not processes["complete"] or resource_uncertain
        else "RESIDUALS_FOUND",
        "checks": checks,
        "blockers": blockers,
        "process_observation": processes,
        "read_only": True,
        "physical_test_status": "PHYSICAL_TEST_PENDING",
    }


class PhysicalRun:
    """Receipts from this live instance are the sole cleanup authority."""

    def __init__(self, package: dict[str, Any]):
        self.package = package
        self.plan = package["plan"]
        self.resources = self.plan["roles"]
        self.root = Path(self.plan["root"])
        self.manifest_path = self.root / "manifest.json"
        self.pin_path = self.root / "TEST_ONLY-pin.json"
        self.receipts: dict[Path, tuple[int, int, bool]] = {}
        self.file_integrity: dict[Path, tuple[str, int]] = {}
        self.directory_modes: dict[Path, int] = {}
        self.accounts: dict[str, list[str]] = {}
        self.groups: dict[str, list[str]] = {}
        self.pins: dict[str, dict[str, str]] = {}
        self.handles: dict[str, int] = {}
        self.lifetimes: dict[str, dict[str, Any]] = {}
        self.pending: list[str] = []
        self.started: set[str] = set()
        self.cleanup_result: dict[str, Any] = {"clean": False, "blockers": ["NOT_EXECUTED"]}
        self._journal_ready = False
        self._cleaned = False
        self._entered = False

    def _role(self, role: str | dict[str, Any]) -> dict[str, Any]:
        name = role["role"] if isinstance(role, dict) else role
        for resource in self.resources:
            if resource["role"] == name:
                return cast(dict[str, Any], resource)
        _refuse("UNKNOWN_TEST_ROLE")
        raise AssertionError("unreachable")

    def _record(self, path: Path, directory: bool) -> None:
        info = _protected(path, directory=directory)
        self.receipts[path] = (info.st_dev, info.st_ino, directory)
        if directory:
            self.directory_modes[path] = stat.S_IMODE(info.st_mode)

    def _journal(self) -> None:
        if not self._journal_ready:
            return
        document = {
            "schema": "CryptoHunter.SourceCCATestResourceJournalV1",
            "version": 1,
            "TEST_ONLY": True,
            "run_id": self.plan["run_id"],
            "commit": self.package["commit"],
            "package_sha256": self.package["package_sha256"],
            "plan": self.plan,
            "files": [
                {"path": str(path), "device": value[0], "inode": value[1], "directory": value[2]}
                for path, value in self.receipts.items()
            ],
            "accounts": self.accounts,
            "groups": self.groups,
            "units": self.pins,
            "processes": self.lifetimes,
            "pending": self.pending,
        }
        self.write_owned(
            self.root / "registry.json",
            json.dumps(document, sort_keys=True).encode("ascii"),
            mode=0o600,
        )

    def _intent(self, name: str) -> None:
        self.pending.append(name)
        self._journal()

    def _complete(self, name: str) -> None:
        self.pending.remove(name)
        self._journal()

    def assert_owned_file(self, path: str | Path) -> os.stat_result:
        path = Path(path)
        receipt = self.receipts.get(path)
        if receipt is None or receipt[2]:
            _refuse("UNOWNED_RESOURCE")
        for parent in path.parents:
            metadata = _protected(parent, directory=True)
            if parent in self.receipts:
                expected = self.receipts[parent]
                if (metadata.st_dev, metadata.st_ino, True) != expected or stat.S_IMODE(
                    metadata.st_mode
                ) != self.directory_modes[parent]:
                    _refuse("RESOURCE_IDENTITY_CHANGED")
        info = _protected(path, directory=False)
        if (info.st_dev, info.st_ino, False) != receipt:
            _refuse("RESOURCE_IDENTITY_CHANGED")
        self._verify_integrity(path, info)
        return info

    def _verify_integrity(self, path: Path, info: os.stat_result) -> None:
        expected = self.file_integrity.get(path)
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK)
        try:
            before = os.fstat(descriptor)
            if (
                (before.st_dev, before.st_ino, before.st_size, before.st_mode)
                != (info.st_dev, info.st_ino, info.st_size, info.st_mode)
                or not stat.S_ISREG(before.st_mode)
                or before.st_nlink != 1
                or before.st_size > 16 * 1024 * 1024
            ):
                _refuse("RESOURCE_IDENTITY_CHANGED")
            digest = hashlib.sha256()
            total = 0
            while True:
                chunk = os.read(descriptor, 65536)
                if not chunk:
                    break
                total += len(chunk)
                if total > 16 * 1024 * 1024:
                    _refuse("RESOURCE_INTEGRITY_CHANGED")
                digest.update(chunk)
            after = os.fstat(descriptor)
            if (before.st_size, before.st_mtime_ns, before.st_ctime_ns, before.st_mode) != (
                after.st_size,
                after.st_mtime_ns,
                after.st_ctime_ns,
                after.st_mode,
            ) or expected != (digest.hexdigest(), stat.S_IMODE(after.st_mode)):
                _refuse("RESOURCE_INTEGRITY_CHANGED")
        finally:
            os.close(descriptor)

    def write_owned(self, path: str | Path, data: bytes, mode: int | None = None) -> None:
        if sys.platform != "linux":
            _refuse("NATIVE_LINUX_REQUIRED")
        path = Path(path)
        info = self.assert_owned_file(path)
        descriptor = os.open(path, os.O_WRONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
        try:
            current = os.fstat(descriptor)
            if (current.st_dev, current.st_ino) != (info.st_dev, info.st_ino):
                _refuse("RESOURCE_IDENTITY_CHANGED")
            os.ftruncate(descriptor, 0)
            view = memoryview(data)
            while view:
                written = os.write(descriptor, view)
                if written <= 0:
                    _refuse("RESOURCE_WRITE_FAILED")
                view = view[written:]
            if mode is not None:
                os.fchmod(descriptor, mode)
            os.fsync(descriptor)
            self.file_integrity[path] = (
                hashlib.sha256(data).hexdigest(),
                stat.S_IMODE(os.fstat(descriptor).st_mode),
            )
        finally:
            os.close(descriptor)

    def chmod_owned(self, path: str | Path, mode: int) -> None:
        """A scoped permission probe; restore accepts its own unsafe mode."""
        if sys.platform != "linux":
            _refuse("NATIVE_LINUX_REQUIRED")
        path = Path(path)
        if path not in {
            Path(resource["executable"]) for resource in self.resources
        } or mode not in {0o755, 0o777}:
            _refuse("MUTATION_SCOPE_INVALID")
        receipt = self.receipts.get(path)
        for parent in path.parents:
            _protected(parent, directory=True)
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
        try:
            info = os.fstat(descriptor)
            if (
                receipt != (info.st_dev, info.st_ino, False)
                or not stat.S_ISREG(info.st_mode)
                or info.st_uid != 0
                or info.st_gid != 0
                or info.st_nlink != 1
            ):
                _refuse("RESOURCE_IDENTITY_CHANGED")
            self._verify_integrity(path, info)
            os.fchmod(descriptor, mode)
            os.fsync(descriptor)
            self.file_integrity[path] = (self.file_integrity[path][0], mode)
        finally:
            os.close(descriptor)

    def _new(self, path: Path, raw: bytes, mode: int) -> None:
        if sys.platform != "linux":
            _refuse("NATIVE_LINUX_REQUIRED")
        with _defer_interrupts():
            descriptor = os.open(
                path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600
            )
            try:
                os.fchmod(descriptor, mode)
                self._record(path, False)
                view = memoryview(raw)
                while view:
                    written = os.write(descriptor, view)
                    if written <= 0:
                        _refuse("RESOURCE_WRITE_FAILED")
                    view = view[written:]
                os.fsync(descriptor)
                self.file_integrity[path] = (hashlib.sha256(raw).hexdigest(), mode)
            finally:
                os.close(descriptor)
        self._journal()

    def __enter__(self) -> PhysicalRun:
        if self._cleaned or self._entered:
            _refuse("RUN_INSTANCE_REUSE_FORBIDDEN")
        if sys.platform != "linux":
            _refuse("NATIVE_LINUX_REQUIRED")
        if os.geteuid() != 0:
            _refuse("ROOT_EXECUTION_REQUIRED")
        observed = preflight(self.plan, administrative=True)
        if observed["ready"] is not True:
            _refuse("PREFLIGHT_BLOCKED")
        self._entered = True
        try:
            with _defer_interrupts():
                self.root.mkdir(mode=0o755)
                self._record(self.root, True)
            self._new(self.root / "registry.json", b"{}", 0o600)
            self._journal_ready = True
            self._new(
                self.root / "TEST_ONLY-owner.json",
                json.dumps(
                    {
                        "run_id": self.plan["run_id"],
                        "TEST_ONLY": True,
                        "package_sha256": self.package["package_sha256"],
                    },
                    sort_keys=True,
                ).encode("ascii"),
                0o600,
            )
            for resource in self.resources:
                name = resource["principal"]
                self._intent("group:" + name)
                with _defer_interrupts():
                    if _nss("group", name) or _nss("group", str(resource["gid"])):
                        _refuse("ACCOUNT_COLLISION")
                    _command("groupadd", "--system", "--gid", str(resource["gid"]), name)
                    rows = _nss("group", name)
                    if len(rows) != 1 or rows[0][2:] != [str(resource["gid"]), ""]:
                        _refuse("ACCOUNT_CREATION_UNPROVEN")
                    self.groups[name] = rows[0]
                self._complete("group:" + name)
                self._intent("user:" + name)
                with _defer_interrupts():
                    if _nss("passwd", name) or _nss("passwd", str(resource["uid"])):
                        _refuse("ACCOUNT_COLLISION")
                    _command(
                        "useradd",
                        "--system",
                        "--key",
                        "CREATE_MAIL_SPOOL=no",
                        "--key",
                        "SUB_UID_COUNT=0",
                        "--key",
                        "SUB_GID_COUNT=0",
                        "--uid",
                        str(resource["uid"]),
                        "--gid",
                        str(resource["gid"]),
                        "--no-user-group",
                        "--no-create-home",
                        "--no-log-init",
                        "--home-dir",
                        "/nonexistent",
                        "--shell",
                        "/usr/sbin/nologin",
                        "--comment",
                        "CH S9 TEST ONLY " + self.plan["run_id"],
                        name,
                    )
                    rows = _nss("passwd", name)
                    if len(rows) != 1 or rows[0][2:] != [
                        str(resource["uid"]),
                        str(resource["gid"]),
                        "CH S9 TEST ONLY " + self.plan["run_id"],
                        "/nonexistent",
                        "/usr/sbin/nologin",
                    ]:
                        _refuse("ACCOUNT_CREATION_UNPROVEN")
                    self.accounts[name] = rows[0]
                self._complete("user:" + name)
            for resource in self.resources:
                role = resource["role"]
                self._intent("file:" + resource["executable"])
                self._new(Path(resource["executable"]), self.package["files"][role], 0o755)
                self._complete("file:" + resource["executable"])
                self._intent("file:" + resource["unit_path"])
                self._new(
                    Path(resource["unit_path"]), self.package["files"][role + ".service"], 0o644
                )
                self._complete("file:" + resource["unit_path"])
            for name in ("manifest.json", "TEST_ONLY-pin.json", "resources.json"):
                self._intent("file:" + str(self.root / name))
                self._new(self.root / name, self.package["files"][name], 0o600)
                self._complete("file:" + str(self.root / name))
            _command("systemctl", "daemon-reload")
            return self
        except BaseException:
            self.cleanup()
            raise

    def _owns_unit(
        self, resource: dict[str, Any], observed: dict[str, str], *, ended: bool = False
    ) -> bool:
        try:
            self.assert_owned_file(Path(resource["unit_path"]))
        except Exception:
            return False
        if not (
            observed.get("Id") == resource["unit"]
            and observed.get("FragmentPath") == resource["unit_path"]
            and observed.get("DropInPaths") == ""
            and observed.get("Transient") == "no"
            and observed.get("Description")
            == "Stage9 SourceCCA TEST ONLY " + self.plan["run_id"] + " " + resource["role"]
        ):
            return False
        pin = self.pins.get(resource["role"])
        if pin is None:
            return (
                resource["role"] in self.started
                and re.fullmatch(r"[0-9a-f]{32}", observed.get("InvocationID", "")) is not None
            )
        return observed.get("InvocationID") == pin["InvocationID"] and observed.get("MainPID") in (
            {pin["MainPID"], "0"} if ended else {pin["MainPID"]}
        )

    def _capture(self, resource: dict[str, Any]) -> bool:
        if sys.platform != "linux":
            _refuse("NATIVE_LINUX_REQUIRED")
        role = resource["role"]
        observed = inspect_unit(resource["unit"])
        if not self._owns_unit(resource, observed):
            _refuse("UNIT_IDENTITY_CHANGED")
        if not observed.get("MainPID", "").isdigit() or int(observed["MainPID"]) <= 1:
            return False
        pid = int(observed["MainPID"])
        old = self.pins.get(role)
        if old is not None and observed != old:
            _refuse("UNIT_IDENTITY_CHANGED")
        if role not in self.handles:
            with _defer_interrupts():
                descriptor = os.pidfd_open(pid, 0)
            retained = False
            try:
                if _process_finished(descriptor):
                    _refuse("PROCESS_UNAVAILABLE")
                before = _process_lifetime(pid)
                after_unit = inspect_unit(resource["unit"])
                after = _process_lifetime(pid)
                if (
                    _process_finished(descriptor)
                    or before != after
                    or observed != after_unit
                    or before["control_group"] != resource["control_group"]
                ):
                    _refuse("PROCESS_IDENTITY_CHANGED")
                with _defer_interrupts():
                    self.handles[role] = descriptor
                    self.lifetimes[role] = before
                    self.pins[role] = observed
                    retained = True
                self._journal()
            finally:
                if not retained:
                    os.close(descriptor)
        if _process_finished(self.handles[role]) or _process_lifetime(pid) != self.lifetimes[role]:
            _refuse("PROCESS_IDENTITY_CHANGED")
        return True

    def start_role(self, role: str | dict[str, Any]) -> None:
        resource = self._role(role)
        name = resource["role"]
        if name in self.handles:
            if not _process_finished(self.handles[name]) or not self._owns_unit(
                resource, inspect_unit(resource["unit"]), ended=True
            ):
                _refuse("PROCESS_IDENTITY_CHANGED")
            os.close(self.handles.pop(name))
            self.pins.pop(name)
            self.lifetimes.pop(name)
        self.assert_owned_file(Path(resource["unit_path"]))
        self.assert_owned_file(Path(resource["executable"]))
        self._intent("start:" + name)
        self.started.add(name)
        try:
            _command("systemctl", "start", resource["unit"])
        finally:
            try:
                self._capture(resource)
            except Exception:
                pass
        deadline = time.monotonic() + 10
        expectations = {
            value.role: value
            for value in parse_manifest(self.package["files"]["manifest.json"]).roles
        }
        while time.monotonic() < deadline:
            try:
                if self._capture(resource):
                    pid, _ = _unit(expectations[name])
                    identity = _process_identity(pid)
                    if identity["uid"] != resource["uid"] or identity["gid"] != resource["gid"]:
                        _refuse("PROCESS_CREDENTIAL_MISMATCH")
                    self._complete("start:" + name)
                    return
            except (QualificationBlocked, ValueError, OSError):
                pass
            time.sleep(0.1)
        _refuse("SERVICE_START_UNPROVEN")

    def start(self) -> None:
        for resource in self.resources:
            self.start_role(resource)

    def stop(self, role: str | dict[str, Any]) -> None:
        if sys.platform != "linux":
            _refuse("NATIVE_LINUX_REQUIRED")
        resource = self._role(role)
        name = resource["role"]
        descriptor = self.handles.get(name)
        lifetime = self.lifetimes.get(name)
        if descriptor is None or lifetime is None:
            _refuse("PROCESS_OWNERSHIP_UNPROVEN")
        if _process_finished(descriptor):
            self._wait_stopped(resource)
            return
        observed = inspect_unit(resource["unit"])
        if (
            not self._owns_unit(resource, observed)
            or _process_lifetime(lifetime["pid"]) != lifetime
        ):
            _refuse("PROCESS_IDENTITY_CHANGED")
        signal.pidfd_send_signal(descriptor, signal.SIGTERM, None, 0)
        if not _process_finished(descriptor, 3):
            signal.pidfd_send_signal(descriptor, signal.SIGKILL, None, 0)
            if not _process_finished(descriptor, 2):
                _refuse("PROCESS_TERMINATION_UNPROVEN")
        self._wait_stopped(resource)

    def _wait_stopped(self, resource: dict[str, Any]) -> None:
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            observed = inspect_unit(resource["unit"])
            if not self._owns_unit(resource, observed, ended=True):
                _refuse("UNIT_IDENTITY_CHANGED")
            if observed.get("MainPID") == "0" and observed.get("ActiveState") in {
                "inactive",
                "failed",
            }:
                return
            time.sleep(0.05)
        _refuse("UNIT_STOP_UNPROVEN")

    def restart(self, role: str | dict[str, Any]) -> None:
        resource = self._role(role)
        name = resource["role"]
        self.stop(resource)
        observed = inspect_unit(resource["unit"])
        if not self._owns_unit(resource, observed, ended=True):
            _refuse("UNIT_IDENTITY_CHANGED")
        os.close(self.handles.pop(name))
        self.pins.pop(name)
        self.lifetimes.pop(name)
        self.start_role(resource)

    def _remove(self, path: Path) -> None:
        receipt = self.receipts[path]
        if receipt[2]:
            metadata = _protected(path, directory=True)
            if (metadata.st_dev, metadata.st_ino, True) != receipt or stat.S_IMODE(
                metadata.st_mode
            ) != self.directory_modes[path]:
                _refuse("RESOURCE_IDENTITY_CHANGED")
            path.rmdir()
        else:
            self.assert_owned_file(path)
            path.unlink()
        del self.receipts[path]
        self.file_integrity.pop(path, None)
        self.directory_modes.pop(path, None)

    @contextmanager
    def replace_owned(
        self, path: str | Path, replacement: bytes | None = None, symlink_target: str | None = None
    ) -> Iterator[None]:
        """Keep the original inode; replacements are never executed."""
        path = Path(path)
        original = self.assert_owned_file(path)
        held = path.with_name(path.name + ".held")
        if not _absent(held) or (replacement is None) == (symlink_target is None):
            _refuse("MUTATION_SCOPE_INVALID")
        self._intent("replace:" + str(path))
        with _defer_interrupts():
            path.rename(held)
            self.receipts[held] = self.receipts.pop(path)
            self.file_integrity[held] = self.file_integrity.pop(path)
        replacement_identity: tuple[int, int] | None = None
        try:
            if replacement is not None:
                self._new(path, replacement, stat.S_IMODE(original.st_mode))
            else:
                with _defer_interrupts():
                    assert symlink_target is not None
                    os.symlink(symlink_target, path)
                    info = path.lstat()
                    replacement_identity = (info.st_dev, info.st_ino)
            yield
        finally:
            with _defer_interrupts():
                if replacement is not None:
                    self._remove(path)
                else:
                    info = path.lstat()
                    if (
                        not stat.S_ISLNK(info.st_mode)
                        or (info.st_dev, info.st_ino) != replacement_identity
                        or info.st_uid != 0
                    ):
                        _refuse("RESOURCE_IDENTITY_CHANGED")
                    path.unlink()
                self.assert_owned_file(held)
                if not _absent(path):
                    _refuse("RESOURCE_IDENTITY_CHANGED")
                held.rename(path)
                self.receipts[path] = self.receipts.pop(held)
                self.file_integrity[path] = self.file_integrity.pop(held)
            self._complete("replace:" + str(path))

    @contextmanager
    def test_dropin(self, role: str | dict[str, Any], raw: bytes) -> Iterator[None]:
        resource = self._role(role)
        self.assert_owned_file(Path(resource["unit_path"]))
        directory = Path(resource["unit_path"] + ".d")
        path = directory / "test-only.conf"
        self._intent("dropin:" + str(path))
        with _defer_interrupts():
            directory.mkdir(mode=0o755)
            self._record(directory, True)
        try:
            self._new(path, raw, 0o644)
            _command("systemctl", "daemon-reload")
            yield
        finally:
            self._remove(path)
            self._remove(directory)
            _command("systemctl", "daemon-reload")
            self._complete("dropin:" + str(path))

    def cleanup(self) -> dict[str, Any]:
        if self._cleaned:
            return self.cleanup_result
        blockers: list[str] = []
        if not (self.receipts or self.accounts or self.groups or self.handles):
            inspection = inspect_cleanup(self.plan, administrative=True)
            if inspection["clean"] is not True or inspection.get("full_confirmation") is not True:
                blockers.append("INDEPENDENT_INSPECTION_BLOCKED")
            if self.pending:
                blockers.append("UNRESOLVED_CREATION_OR_MUTATION")
            self.cleanup_result = {
                "clean": not blockers,
                "blockers": blockers,
                "inspection": inspection,
            }
            self._cleaned = True
            return self.cleanup_result
        with _defer_interrupts():
            for resource in reversed(self.resources):
                name = resource["role"]
                if name not in self.started:
                    continue
                try:
                    if name not in self.handles:
                        self._capture(resource)
                    self.stop(resource)
                    observed = inspect_unit(resource["unit"])
                    if not self._owns_unit(resource, observed, ended=True):
                        _refuse("UNIT_IDENTITY_CHANGED")
                    # Reset only the exact installed unit after the pinned
                    # lifetime has exited; this does not signal a process.
                    _command("systemctl", "reset-failed", resource["unit"])
                    intent = "start:" + name
                    if intent in self.pending:
                        self._complete(intent)
                except Exception:
                    blockers.append("UNIT_CLEANUP_UNPROVEN:" + name)
            for resource in reversed(self.resources):
                name = resource["principal"]
                if name in self.accounts:
                    try:
                        if (
                            _nss("passwd", name) != [self.accounts[name]]
                            or _nss("passwd", str(resource["uid"])) != [self.accounts[name]]
                            or _processes({**self.plan, "roles": [resource]})
                            or any(
                                name in row[3].split(",") for row in _nss("group") if row[0] != name
                            )
                        ):
                            _refuse("ACCOUNT_IDENTITY_CHANGED")
                        _command("userdel", name)
                        if _nss("passwd", name) or _nss("passwd", str(resource["uid"])):
                            _refuse("ACCOUNT_REMOVAL_UNPROVEN")
                        del self.accounts[name]
                    except Exception:
                        blockers.append("USER_CLEANUP_UNPROVEN:" + name)
                if name in self.groups:
                    try:
                        if (
                            name in self.accounts
                            or _nss("group", name) != [self.groups[name]]
                            or _nss("group", str(resource["gid"])) != [self.groups[name]]
                            or any(row[3] == str(resource["gid"]) for row in _nss("passwd"))
                        ):
                            _refuse("ACCOUNT_IDENTITY_CHANGED")
                        _command("groupdel", name)
                        if _nss("group", name) or _nss("group", str(resource["gid"])):
                            _refuse("ACCOUNT_REMOVAL_UNPROVEN")
                        del self.groups[name]
                    except Exception:
                        blockers.append("GROUP_CLEANUP_UNPROVEN:" + name)
            if self.pending:
                blockers.append("UNRESOLVED_CREATION_OR_MUTATION")
            # Retain evidence and unit files when a process/unit origin is
            # ambiguous. Known independent files may otherwise be removed.
            blocked_units = {
                resource["unit_path"]
                for resource in self.resources
                if "UNIT_CLEANUP_UNPROVEN:" + resource["role"] in blockers
            }
            for path in reversed(list(self.receipts)):
                if str(path) in blocked_units or path in {
                    self.root,
                    self.root / "registry.json",
                    self.root / "TEST_ONLY-owner.json",
                }:
                    continue
                try:
                    self._remove(path)
                except Exception:
                    blockers.append("FILE_CLEANUP_UNPROVEN")
            try:
                _command("systemctl", "daemon-reload")
            except Exception:
                blockers.append("DAEMON_RELOAD_UNPROVEN")
            try:
                if self.root in self.receipts and any(
                    path not in self.receipts for path in self.root.iterdir()
                ):
                    blockers.append("UNKNOWN_RESOURCES_RETAINED")
            except Exception:
                blockers.append("RESOURCE_INVENTORY_UNPROVEN")
            try:
                self._journal()
            except Exception:
                blockers.append("JOURNAL_UPDATE_UNPROVEN")
            if not blockers:
                for path in (
                    self.root / "TEST_ONLY-owner.json",
                    self.root / "registry.json",
                    self.root,
                ):
                    try:
                        self._remove(path)
                    except Exception:
                        blockers.append("JOURNAL_CLEANUP_UNPROVEN")
                        break
            for descriptor in self.handles.values():
                try:
                    os.close(descriptor)
                except OSError:
                    blockers.append("PIDFD_CLOSE_UNPROVEN")
            self.handles.clear()
        inspection = inspect_cleanup(self.plan, administrative=True)
        if inspection["clean"] is not True or inspection.get("full_confirmation") is not True:
            blockers.append("INDEPENDENT_INSPECTION_BLOCKED")
        self.cleanup_result = {
            "clean": not blockers,
            "blockers": sorted(set(blockers)),
            "inspection": inspection,
        }
        self._cleaned = True
        return self.cleanup_result

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.cleanup()

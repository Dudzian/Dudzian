"""Review V5d11 libvirt and glibc evidence offline, without loading profile A."""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Callable
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
from typing import Any, cast

from scripts import stage9_v5d10_abi as abi
from scripts import stage9_v5d10_contract as contract
from scripts import stage9_v5d10_launch as launch
from scripts import stage9_v5d11_libvirt as virt

PROFILE = "stage9-v5d11-profile-a-native-evidence-v2"
STATIC_REPORT = "stage9-v5d4-static-abi-report-v1"
READS = {
    "getURI",
    "getType",
    "getLibVersion",
    "lookupByUUIDString",
    "UUIDString",
    "XMLDesc(0)",
    "XMLDesc(2)",
    "state(0)",
    "info",
    "schedulerParameters",
    "ID",
    "isActive",
    "isPersistent",
    "snapshotListNames(0)",
    "hasManagedSaveImage(0)",
    "networkLookupByName",
    "network.UUIDString",
    "network.isActive",
    "network.XMLDesc(0)",
    "close",
}
PROBES = {
    "python-startup",
    "python-imports",
    "python-libvirt",
    "qemu-raw",
    "qemu-qcow2",
}
VDSO = "linux-vdso.so.1"
CLIENT_BYTES = b'uri_default = "qemu:///system"\n'
MAX_ARTIFACT = 32 * 1024 * 1024
MAX_MAP_SAMPLES = 16
PARENT_LABEL = b"cryptohunter-observer-preflight-v5d11-a (enforce)"
QEMU_LABEL = b"cryptohunter-observer-preflight-v5d11-qemu (enforce)"
DIAGNOSTIC_LABEL = b"cryptohunter-observer-preflight-v5d11-a-diagnostic"
DIAGNOSTIC_QEMU_LABEL = (
    DIAGNOSTIC_LABEL + b"//&cryptohunter-observer-preflight-v5d11-qemu-diagnostic (enforce)"
)
TRACE_LINE = re.compile(r"\s*(\d+):\s*(.*?)\s*")
GENERATOR = re.compile(r"file=(\S+) \[(\d+)\];\s+generating link map")
DLOPEN = re.compile(r"file=(\S+) \[(\d+)\];\s+dynamically loaded by (\S+) \[(\d+)\]")
NEEDED = re.compile(r"file=(\S+) \[(\d+)\];\s+needed by (\S+) \[(\d+)\]")
INIT = re.compile(r"calling (?:init|fini):\s*(\S+)")
VERSION = re.compile(
    r"checking for version [`'](\S+)' in file (\S+) \[(\d+)\] required by file (\S+) \[(\d+)\]"
)
BINDING = re.compile(
    r"binding file (\S+) \[(\d+)\] to (\S+) \[(\d+)\]: "
    r"\S+ symbol [`']([^']+)'(?: \[([^\]]+)\])?"
)
DYNSYM = re.compile(
    r"^[ \t]*\d+:[ \t]+[0-9a-fA-F]+[ \t]+\d+[ \t]+\S+[ \t]+(GLOBAL|WEAK|UNIQUE)[ \t]+"
    r"(DEFAULT|PROTECTED)[ \t]+(\S+)[ \t]+(\S+)(?:[ \t]+\(\d+\))?[ \t]*$",
    re.MULTILINE,
)
MAP_HEADER = re.compile(
    r"[0-9a-f]+-[0-9a-f]+ [r-][w-][x-][ps] [0-9a-f]+ [0-9a-f]+:[0-9a-f]+ +[0-9]+"
)

check = contract.require


def _fields(value: Any, names: set[str], error: str) -> None:
    check(type(value) is dict and set(value) == names, error)


def _ints(value: Any, count: int, error: str) -> None:
    check(
        type(value) is list
        and len(value) == count
        and all(type(item) is int and item >= 0 for item in value),
        error,
    )


def _positive(value: Any) -> bool:
    return type(value) is int and value > 0


def _pins(value: Any) -> bool:
    return (
        type(value) is dict
        and bool(value)
        and all(
            contract.path_value(path, absolute=True) and contract.sha256_value(digest)
            for path, digest in value.items()
        )
    )


def _read(path: Path, expected: str, *, limit: int = MAX_ARTIFACT) -> bytes:
    check(contract.sha256_value(expected), "QUALIFICATION_PIN_REQUIRED")
    raw = cast(bytes, contract.secure_read(path.absolute(), limit=limit))
    check(contract.digest(raw) == expected, "QUALIFICATION_ARTIFACT_PIN_MISMATCH")
    return raw


def _artifact(artifacts: Path, row: dict[str, Any], *, limit: int = MAX_ARTIFACT) -> bytes:
    _fields(row, {"path", "sha256"}, "QUALIFICATION_ARTIFACT_SCHEMA")
    check(contract.path_value(row["path"], absolute=False), "QUALIFICATION_ARTIFACT_PATH")
    return _read(artifacts / row["path"], row["sha256"], limit=limit)


def _libvirt(
    record: dict[str, Any],
    artifacts: Path,
    session: dict[str, str],
    namespaces: dict[str, int],
) -> None:
    _fields(
        record,
        {
            "socket",
            "socket_identity",
            "peer",
            "host_peer",
            "host_socket_identity",
            "daemon",
            "client_config_sha256",
            "uri",
            "connect_paths",
            "read_results",
            "fixture_denial",
        },
        "LIBVIRT_EVIDENCE_SCHEMA",
    )
    check(record["socket"] in launch.LOCAL_SOCKETS, "LIBVIRT_SOCKET_SCOPE")
    _ints(record["socket_identity"], 5, "LIBVIRT_SOCKET_IDENTITY")
    device, inode, uid, _, mode = record["socket_identity"]
    check(device > 0 and inode > 0 and uid == 0 and 0 < mode < 0o1000, "LIBVIRT_SOCKET_IDENTITY")
    _ints(record["peer"], 3, "LIBVIRT_PEER_IDENTITY")
    _ints(record["host_peer"], 3, "LIBVIRT_PEER_IDENTITY")
    peer_pid, peer_uid, peer_gid = record["peer"]
    host_pid, host_uid, host_gid = record["host_peer"]
    # A peer outside the view's PID namespace is reported as zero there.
    check(
        peer_pid == 0
        and peer_uid == host_uid == 0
        and peer_gid == host_gid
        and host_pid > 0
        and record["host_socket_identity"] == record["socket_identity"],
        "LIBVIRT_PEER_IDENTITY",
    )
    daemon = record["daemon"]
    _fields(
        daemon,
        {
            "pid",
            "start_ticks",
            "executable",
            "executable_sha256",
            "cmdline_sha256",
            "config",
            "unit_files",
            "socket_unit",
            "service_unit",
            "listener_inode",
            "reviewed_readonly_implementation_sha256",
        },
        "LIBVIRT_DAEMON_SCHEMA",
    )
    modular = record["socket"] == "/run/libvirt/virtqemud-sock-ro"
    name = "virtqemud" if modular else "libvirtd"
    config = f"/etc/libvirt/{name}.conf"
    check(
        _positive(daemon["pid"])
        and _positive(daemon["start_ticks"])
        and _positive(daemon["listener_inode"])
        and daemon["executable"] == f"/usr/sbin/{name}"
        and host_pid in {daemon["pid"], 1}
        and daemon["socket_unit"] == f"{name}-ro.socket"
        and daemon["service_unit"] == f"{name}.service"
        and _pins(daemon["config"])
        and config in daemon["config"]
        and _pins(daemon["unit_files"])
        and {Path(path).name for path in daemon["unit_files"]}
        >= {daemon["socket_unit"], daemon["service_unit"]}
        and all(
            contract.sha256_value(daemon[field])
            for field in (
                "executable_sha256",
                "cmdline_sha256",
                "reviewed_readonly_implementation_sha256",
            )
        ),
        "LIBVIRT_DAEMON_IDENTITY",
    )
    check(
        record["uri"] == "qemu:///system"
        and record["connect_paths"] == [record["socket"]]
        and record["client_config_sha256"] == contract.digest(CLIENT_BYTES),
        "LIBVIRT_CLIENT_BOUNDARY",
    )
    reads = record["read_results"]
    _fields(reads, READS, "LIBVIRT_READ_SCOPE")
    check(all(value == "PASS" for value in reads.values()), "LIBVIRT_READ_FAILED")
    denial = record["fixture_denial"]
    _fields(
        denial,
        {
            "scope",
            "daemon_sha256",
            "reviewed_readonly_implementation_sha256",
            "config",
            "unit_files",
            "client_connect_flags",
            "fixture",
            "trace",
        },
        "LIBVIRT_DENIAL_SCHEMA",
    )
    # A nonmutating request for secure XML distinguishes the server RO
    # boundary from a client that only checks VIR_CONNECT_RO locally.
    check(
        denial["scope"] == "DISPOSABLE_FIXTURE"
        and denial["daemon_sha256"] == daemon["executable_sha256"]
        and denial["reviewed_readonly_implementation_sha256"]
        == daemon["reviewed_readonly_implementation_sha256"]
        and denial["config"] == daemon["config"]
        and denial["unit_files"] == daemon["unit_files"]
        and type(denial["client_connect_flags"]) is int
        and denial["client_connect_flags"] == 0,
        "LIBVIRT_SERVER_READONLY_UNPROVEN",
    )
    fixture = denial["fixture"]
    _fields(
        fixture,
        {
            "socket",
            "socket_identity",
            "connection_identity",
            "peer",
            "start_ticks",
            "namespaces",
            "host_namespaces",
            "executable",
            "cmdline_sha256",
            "domain_uuid",
        },
        "LIBVIRT_FIXTURE_IDENTITY",
    )
    _ints(fixture["socket_identity"], 5, "LIBVIRT_FIXTURE_IDENTITY")
    _ints(fixture["connection_identity"], 2, "LIBVIRT_FIXTURE_IDENTITY")
    _ints(fixture["peer"], 3, "LIBVIRT_FIXTURE_IDENTITY")
    socket_dev, socket_inode, socket_uid, _, socket_mode = fixture["socket_identity"]
    fixture_pid, fixture_uid, _ = fixture["peer"]
    check(
        contract.path_value(fixture["socket"], absolute=True)
        and fixture["socket"] not in launch.LOCAL_SOCKETS
        and socket_dev > 0
        and socket_inode > 0
        and socket_uid == 0
        and 0 < socket_mode < 0o1000
        and fixture["socket_identity"][:2] != record["socket_identity"][:2]
        and all(_positive(item) for item in fixture["connection_identity"])
        and fixture_pid > 1
        and fixture_pid != daemon["pid"]
        and fixture_uid == 0
        and _positive(fixture["start_ticks"])
        and fixture["executable"] == daemon["executable"]
        and contract.sha256_value(fixture["cmdline_sha256"])
        and type(fixture["domain_uuid"]) is str,
        "LIBVIRT_FIXTURE_IDENTITY",
    )
    namespace_names = {"mnt", "net", "pid"}
    _fields(fixture["namespaces"], namespace_names, "LIBVIRT_FIXTURE_NAMESPACE_REQUIRED")
    _fields(fixture["host_namespaces"], namespace_names, "LIBVIRT_FIXTURE_NAMESPACE_REQUIRED")
    check(
        all(
            _positive(observed)
            and _positive(fixture["host_namespaces"][name])
            and observed not in {namespaces[name], fixture["host_namespaces"][name]}
            for name, observed in fixture["namespaces"].items()
        ),
        "LIBVIRT_FIXTURE_NAMESPACE_REQUIRED",
    )
    raw = _artifact(artifacts, denial["trace"], limit=virt.TRACE_LIMIT)
    try:
        virt.verify_denial_trace(
            raw, session, {key: value for key, value in denial.items() if key != "trace"}
        )
    except virt.Refusal as exc:
        raise contract.Refusal(str(exc)) from exc


def _objects(
    report: dict[str, Any], receipt: dict[str, Any], request: dict[str, Any]
) -> tuple[dict[str, str], dict[str, str], dict[tuple[str, str], str]]:
    measured = report.get("elf_measurements")
    check(
        report.get("profile") == STATIC_REPORT
        and report.get("status") == "STATIC_ABI_MATCH"
        and _pins(measured)
        and type(report.get("elf")) is dict
        and set(report["elf"]) == set(report["elf_measurements"]),
        "STATIC_ABI_REPORT_REQUIRED",
    )
    elf: dict[str, str] = report["elf_measurements"]
    root = PurePosixPath(receipt["root"])
    pinned = {
        **receipt["external_files"],
        **receipt["base_runtime"]["files"],
        **{(root / key).as_posix(): row["sha256"] for key, row in receipt["files"].items()},
    }
    check(
        all(pinned.get(path) == digest for path, digest in elf.items()), "STATIC_ABI_PIN_MISMATCH"
    )
    aliases = launch._abi_link_targets(request, receipt["base_runtime"]["symlinks"])
    edges: dict[tuple[str, str], str] = {}
    check(type(report["elf_edges"]) is list and bool(report["elf_edges"]), "STATIC_ABI_EDGES")
    for row in report["elf_edges"]:
        check(
            type(row) is list and len(row) == 3 and all(type(item) is str for item in row),
            "STATIC_ABI_EDGE_SCHEMA",
        )
        source, needed, provider = row
        check(
            source in elf
            and provider in elf
            and request["library_map"].get(needed) == provider
            and (source, needed) not in edges,
            "STATIC_ABI_EDGE_MISMATCH",
        )
        edges[source, needed] = provider
    for path, meta in report["elf"].items():
        names = {*meta["needed"]}
        if meta["interpreter"] is not None:
            names.add(meta["interpreter"])
        check(
            names == {needed for source, needed in edges if source == path},
            "STATIC_ABI_EDGES_INCOMPLETE",
        )
    return elf, aliases, edges


def _canonical(
    path: str, objects: set[str], aliases: dict[str, str], *, argv0: str, executable: str
) -> str:
    if path == argv0:
        return executable
    if path == VDSO:
        return path
    check(contract.path_value(path, absolute=True), "RUNTIME_OBJECT_PATH")
    parent, _, tail = path[1:].partition("/")
    if parent in {"lib", "lib64"}:
        path = f"{aliases['/' + parent]}/{tail}"
    path = aliases.get(path, path)
    check(path in objects, "UNAPPROVED_RUNTIME_OBJECT")
    return path


def _maps(raw: bytes, canonicalize: Callable[[str], str]) -> dict[str, tuple[int, ...]]:
    objects: dict[str, tuple[int, ...]] = {}
    for line in raw.decode("utf-8").splitlines():
        fields = line.split(maxsplit=5)
        check(
            len(fields) in {5, 6} and MAP_HEADER.fullmatch(" ".join(fields[:5])) is not None,
            "RUNTIME_MAPS_FORMAT",
        )
        if len(fields) != 6:
            check("x" not in fields[1], "RUNTIME_ANONYMOUS_EXECUTABLE")
            continue
        path = fields[5]
        if "x" in fields[1] and not path.startswith("/"):
            check(path in {"[vdso]", "[vsyscall]"}, "RUNTIME_ANONYMOUS_EXECUTABLE")
        if path == "[vdso]":
            objects[VDSO] = ()
        elif path.startswith("/") and ("x" in fields[1] or ".so" in PurePosixPath(path).name):
            check(not path.endswith(" (deleted)"), "RUNTIME_DELETED_MAPPING")
            check("x" not in fields[1] or "w" not in fields[1], "RUNTIME_WRITABLE_EXECUTABLE")
            name = canonicalize(path)
            major, minor = fields[3].split(":")
            identity = int(major, 16), int(minor, 16), int(fields[4])
            check(
                identity[2] > 0 and objects.get(name, identity) == identity, "RUNTIME_MAP_IDENTITY"
            )
            objects[name] = identity
    check(VDSO in objects, "RUNTIME_MAPS_INCOMPLETE")
    return objects


def normalize_maps(raw: bytes, staging_root: Path) -> bytes:
    prefix = os.fsencode(staging_root) + b"/"
    records = []
    for line in raw.splitlines():
        parts = line.split(maxsplit=5)
        if len(parts) == 6 and parts[5].startswith(prefix):
            parts[5] = b"/" + parts[5][len(prefix) :]
        records.append(b" ".join(parts))
    return b"\n".join(records) + b"\n"


def merge_mappings(
    observed: dict[str, tuple[int, ...]], current: dict[str, tuple[int, ...]]
) -> bool:
    check(
        all(observed.get(name, identity) == identity for name, identity in current.items()),
        "RUNTIME_MAP_IDENTITY_CHANGED",
    )
    extended = not current.keys() <= observed.keys()
    observed.update(current)
    return extended


def _trace(
    raw: bytes, pid: int, canonicalize: Callable[[str], str], library_map: dict[str, str]
) -> tuple[
    set[str],
    set[str],
    set[tuple[str, str]],
    set[tuple[str, str, str]],
    set[tuple[str, str, str, str | None]],
]:
    objects: set[str] = set()
    dlopen: set[str] = set()
    needed: set[tuple[str, str]] = set()
    versions: set[tuple[str, str, str]] = set()
    bindings: set[tuple[str, str, str, str | None]] = set()
    for line in raw.decode("utf-8").splitlines():
        parsed = TRACE_LINE.fullmatch(line)
        if parsed is None:
            continue  # program stderr and glibc link-map continuation lines
        process, event = parsed.groups()
        check(int(process) == pid, "RUNTIME_TRACE_PROCESS_MISMATCH")
        if match := GENERATOR.fullmatch(event):
            path, scope = match.groups()
            check(scope == "0", "RUNTIME_LINK_NAMESPACE")
            target = library_map.get(path, path)
            objects.add(canonicalize(path if path.startswith("/") else target))
        elif match := DLOPEN.fullmatch(event):
            target, scope, source, src_scope = match.groups()
            check(scope == src_scope == "0", "RUNTIME_LINK_NAMESPACE")
            canonicalize(source)
            dlopen.add(
                canonicalize(target if target.startswith("/") else library_map.get(target, target))
            )
        elif match := NEEDED.fullmatch(event):
            name, scope, source, src_scope = match.groups()
            check(scope == src_scope == "0", "RUNTIME_LINK_NAMESPACE")
            needed.add((canonicalize(source), name))
        elif match := INIT.fullmatch(event):
            objects.add(canonicalize(match[1]))
        elif match := VERSION.fullmatch(event):
            version, target, scope, source, src_scope = match.groups()
            check(scope == src_scope == "0", "RUNTIME_LINK_NAMESPACE")
            versions.add((canonicalize(source), canonicalize(target), version))
        elif match := BINDING.fullmatch(event):
            source, src_scope, target, scope, symbol, version = match.groups()
            check(scope == src_scope == "0", "RUNTIME_LINK_NAMESPACE")
            bindings.add((canonicalize(source), canonicalize(target), symbol, version))
        else:
            check(
                "error: symbol lookup error" not in event
                and not event.startswith(("binding file ", "checking for version "))
                and not any(
                    token in event
                    for token in ("generating link map", "dynamically loaded by", "needed by")
                ),
                "RUNTIME_TRACE_EVENT_REJECTED",
            )
    return objects, dlopen, needed, versions, bindings


def _symbols(
    raw: bytes,
) -> tuple[set[tuple[str, str | None]], set[tuple[str, str | None]]]:
    text = raw.decode("utf-8")
    check("Symbol table '.dynsym' contains " in text, "DYNAMIC_SYMBOL_TABLE_REQUIRED")
    exported: set[tuple[str, str | None]] = set()
    weak: set[tuple[str, str | None]] = set()
    for bind, _, section, raw_name in DYNSYM.findall(text):
        name, mark, version = raw_name.partition("@")
        symbol = name, version.lstrip("@") if mark else None
        if section != "UND":
            exported.add(symbol)
            if version.startswith("@"):
                exported.add((name, None))
        elif bind == "WEAK":
            weak.add(symbol)
    return exported, weak


def _kernel_status(raw: bytes, pid: int) -> None:
    """Check process status collected externally, including qemu-img descendants."""
    required = {b"Uid", b"Gid", b"Groups", b"NSpid", b"NoNewPrivs", b"Seccomp"}
    capabilities = {b"CapInh", b"CapPrm", b"CapEff", b"CapBnd", b"CapAmb"}
    values: dict[bytes, list[bytes]] = {}
    for line in raw.splitlines():
        name, _, data = line.partition(b":")
        if name in required | capabilities:
            check(name not in values, "RUNTIME_KERNEL_STATUS_REJECTED")
            values[name] = data.split()
    check(set(values) == required | capabilities, "RUNTIME_KERNEL_STATUS_REJECTED")
    check(
        values[b"NoNewPrivs"] == [b"1"]
        and values[b"Seccomp"] == [b"2"]
        and values[b"Groups"] == []
        and values[b"NSpid"][-1:] == [str(pid).encode("ascii")]
        and all(values[name] == [b"0000000000000000"] for name in capabilities)
        and all(
            len(values[name]) == 4
            and len(set(values[name])) == 1
            and values[name][0].isdigit()
            and int(values[name][0]) > 0
            for name in (b"Uid", b"Gid")
        ),
        "RUNTIME_KERNEL_STATUS_REJECTED",
    )


def _case(
    name: str,
    row: dict[str, Any],
    artifacts: Path,
    objects: dict[str, str],
    aliases: dict[str, str],
    edges: dict[tuple[str, str], str],
    report: dict[str, Any],
    request: dict[str, Any],
    *,
    diagnostic: bool = False,
) -> dict[str, Any]:
    _fields(
        row,
        {
            "pid",
            "argv0",
            "trace",
            "maps",
            "maps_root",
            "objects",
            "symbols",
            "status",
            "label",
            "outcome",
        },
        "RUNTIME_CASE_SCHEMA",
    )
    qemu = name.startswith("qemu-")
    executable = contract.QEMU_IMG if qemu else request["interpreter"]
    check(
        _positive(row["pid"])
        and (
            re.fullmatch(r"/proc/self/fd/[0-9]+", row["argv0"]) is not None
            if qemu
            else row["argv0"] == executable
        )
        and executable in objects,
        "RUNTIME_PROCESS_IDENTITY",
    )
    _kernel_status(_artifact(artifacts, row["status"]), row["pid"])
    label = QEMU_LABEL if qemu else PARENT_LABEL
    if diagnostic:
        label = DIAGNOSTIC_QEMU_LABEL if qemu else DIAGNOSTIC_LABEL + b" (enforce)"
    check(
        _artifact(artifacts, row["label"]).strip() == label,
        "RUNTIME_APPARMOR_LABEL_REJECTED",
    )

    outcome = row["outcome"]
    if qemu:
        _fields(outcome, {"argv", "fixture_sha256", "exit", "json_format"}, "QEMU_PROBE_SCHEMA")
        fmt = name.removeprefix("qemu-")
        argv = outcome["argv"]
        check(
            type(argv) is list
            and len(argv) == 6
            and argv[:5] == [row["argv0"], "info", "--output=json", "-f", fmt]
            and re.fullmatch(r"/proc/self/fd/[0-9]+", argv[5]) is not None
            and argv[5] != row["argv0"]
            and contract.sha256_value(outcome["fixture_sha256"])
            and type(outcome["exit"]) is int
            and outcome["exit"] == 0
            and outcome["json_format"] == fmt,
            "QEMU_PROBE_REJECTED",
        )
    else:
        check(outcome == "PASS", "PYTHON_PROBE_REJECTED")

    approved = set(objects)

    def canonicalize(path: str) -> str:
        return _canonical(path, approved, aliases, argv0=row["argv0"], executable=executable)

    traced, dlopen, needed, versions, bindings = _trace(
        _artifact(artifacts, row["trace"]), row["pid"], canonicalize, request["library_map"]
    )
    check(
        type(row["maps"]) is list
        and 1 <= len(row["maps"]) <= MAX_MAP_SAMPLES
        and contract.path_value(row["maps_root"], absolute=True),
        "RUNTIME_MAP_SAMPLES_REJECTED",
    )
    map_identities: dict[str, tuple[int, ...]] = {}
    map_bytes = 0
    for sample in row["maps"]:
        _fields(sample, {"raw", "normalized"}, "RUNTIME_MAP_SAMPLE_SCHEMA")
        raw = _artifact(artifacts, sample["raw"])
        normalized = _artifact(artifacts, sample["normalized"])
        map_bytes += len(raw)
        check(map_bytes <= MAX_ARTIFACT, "RUNTIME_MAP_SAMPLES_REJECTED")
        check(
            normalize_maps(raw, Path(row["maps_root"])) == normalized,
            "RUNTIME_MAP_NORMALIZATION_MISMATCH",
        )
        current = _maps(normalized, canonicalize)
        check(executable in current, "RUNTIME_PROBE_EXECUTABLE_UNMAPPED")
        merge_mappings(map_identities, current)
    mapped = set(map_identities)
    check(executable in mapped and bool(bindings) and bool(versions), "RUNTIME_TRACE_INCOMPLETE")
    check(traced <= mapped, "RUNTIME_TRACE_MAPPING_UNOBSERVED")
    check(
        mapped <= traced | {executable, VDSO}
        and dlopen <= traced
        and all((source, target) in edges for source, target in needed)
        and all(edges[source, target] in mapped for source, target in needed),
        "RUNTIME_NEEDED_OR_DLOPEN_MISMATCH",
    )
    loaded = mapped
    required = {executable}
    if name == "python-libvirt":
        required.add(request["libvirt_extension"])
    if name == "python-imports":
        required.add(request["required_imports"]["_cffi_backend"])
        rust = {
            path for path in objects if path.endswith("/cryptography/hazmat/bindings/_rust.abi3.so")
        }
        check(len(rust) == 1, "RUNTIME_CRYPTOGRAPHY_OBJECT_MISSING")
        required.update(rust)
    check(required <= loaded, "RUNTIME_PROBE_OBJECT_MISSING")
    check(
        all(target in loaded for (source, _), target in edges.items() if source in loaded),
        "RUNTIME_ABI_EDGE_TARGET_MISSING",
    )
    for source, target, version in versions:
        version_needs = report["elf"][source]["version_needs"]
        providers = {
            edges.get((source, name))
            for name, requested in version_needs.items()
            if version in requested
        }
        check(
            source in loaded and target in loaded and target in providers,
            "RUNTIME_VERSION_PROVIDER",
        )
    measurements = row["objects"]
    _fields(measurements, loaded - {VDSO}, "RUNTIME_OBJECT_MEASUREMENTS_INCOMPLETE")
    for source, entry in measurements.items():
        _fields(entry, {"sha256", "identity"}, "RUNTIME_OBJECT_MEASUREMENT_SCHEMA")
        identity = entry["identity"]
        _ints(identity, 3, "RUNTIME_OBJECT_MEASUREMENT_IDENTITY")
        check(
            identity[2] > 0
            and tuple(identity) == map_identities[source]
            and entry["sha256"] == objects[source],
            "RUNTIME_OBJECT_MEASUREMENT_MISMATCH",
        )
    symbols = row["symbols"]
    _fields(symbols, loaded - {VDSO}, "RUNTIME_SYMBOL_ARTIFACTS_INCOMPLETE")
    tables = {
        source: _symbols(_artifact(artifacts, artifact)) for source, artifact in symbols.items()
    }
    seen: dict[str, set[tuple[str, str | None]]] = defaultdict(set)
    for source, target, symbol_name, symbol_version in bindings:
        check(source in loaded and target in loaded, "RUNTIME_BINDING_OBJECT_UNMEASURED")
        symbol = symbol_name, symbol_version
        if target != VDSO:
            check(symbol in tables[target][0], "RUNTIME_SYMBOL_PROVIDER_MISMATCH")
        seen[source].add(symbol)
    return {
        "observed_elf_objects": len(loaded - {VDSO}),
        "observed_dlopen_objects": len(dlopen),
        "observed_symbol_bindings": len(bindings),
        "observed_version_checks": len(versions),
        "weak_references_observed_bound": sum(
            len(weak & seen[path]) for path, (_, weak) in tables.items()
        ),
        "weak_references_without_observed_binding": sum(
            len(weak - seen[path]) for path, (_, weak) in tables.items()
        ),
    }


def assess(
    receipt: dict[str, Any],
    request: dict[str, Any],
    static_report: dict[str, Any],
    evidence: dict[str, Any],
    *,
    receipt_sha256: str,
    static_report_sha256: str,
    artifacts: Path,
) -> dict[str, Any]:
    """Cross-check native evidence and pinned ABI; never grant authorization."""
    check(
        evidence.get("profile") != "stage9-v5d11-profile-a-native-evidence-v1",
        "NATIVE_EVIDENCE_SOURCE_ARTIFACTS_REQUIRED",
    )
    _fields(
        evidence,
        {
            "profile",
            "receipt_sha256",
            "abi_request_sha256",
            "static_report_sha256",
            "session",
            "view",
            "libvirt",
            "loader",
        },
        "NATIVE_EVIDENCE_SCHEMA",
    )
    check(
        evidence["profile"] == PROFILE
        and evidence["receipt_sha256"] == receipt_sha256
        and evidence["abi_request_sha256"] == receipt["abi_request_sha256"]
        and evidence["static_report_sha256"] == static_report_sha256,
        "NATIVE_EVIDENCE_PIN_MISMATCH",
    )
    view = evidence["view"]
    view_fields = {"mountinfo", "namespaces", "apparmor_sha256", "seccomp_sha256"}
    diagnostic = type(view) is dict and view.get("purpose") == "diagnostic"
    _fields(view, view_fields | ({"purpose"} if diagnostic else set()), "NATIVE_VIEW_SCHEMA")
    _fields(view["namespaces"], {"mnt", "net", "pid"}, "NATIVE_NAMESPACE_SCHEMA")
    check(
        all(_positive(inode) for inode in view["namespaces"].values())
        and all(contract.sha256_value(view[key]) for key in {"apparmor_sha256", "seccomp_sha256"}),
        "NATIVE_VIEW_PIN_REQUIRED",
    )
    session = evidence["session"]
    _fields(session, virt.SESSION_FIELDS, "LIBVIRT_FIXTURE_SESSION")
    check(
        all(contract.sha256_value(value) for value in session.values()),
        "LIBVIRT_FIXTURE_SESSION",
    )
    _libvirt(evidence["libvirt"], artifacts, session, view["namespaces"])
    socket = evidence["libvirt"]["socket"]
    mountinfo = _artifact(artifacts, view["mountinfo"])
    launch._mount_view(mountinfo, socket)
    check(
        {row.split()[4].decode("ascii") for row in mountinfo.splitlines()}
        == {"/", "/proc", "/dev/null", socket},
        "NATIVE_MOUNT_SET_REJECTED",
    )
    objects, aliases, edges = _objects(static_report, receipt, request)
    _fields(evidence["loader"], PROBES, "NATIVE_PROBE_CASES_REQUIRED")
    observations = {
        name: _case(
            name,
            row,
            artifacts,
            objects,
            aliases,
            edges,
            static_report,
            request,
            diagnostic=diagnostic,
        )
        for name, row in evidence["loader"].items()
    }
    return {
        "profile": PROFILE,
        "status": "EVIDENCE_CONSISTENT_REVIEW_REQUIRED",
        "authorization": "NOT_GRANTED",
        "dynamic_probe_coverage": "OBSERVED_CASES_ONLY",
        "weak_symbol_absence": "NOT_INFERRED_FROM_TRACE",
        "cases": observations,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", required=True, type=Path)
    parser.add_argument("--receipt-sha256", required=True)
    parser.add_argument("--abi-request", required=True, type=Path)
    report_mode = parser.add_mutually_exclusive_group(required=True)
    report_mode.add_argument("--static-report", type=Path)
    report_mode.add_argument("--emit-static-report", action="store_true")
    parser.add_argument("--static-report-sha256")
    parser.add_argument("--abi-expectations", type=Path)
    parser.add_argument("--evidence", type=Path)
    parser.add_argument("--evidence-sha256")
    parser.add_argument("--artifacts", type=Path)
    args = parser.parse_args(argv)
    try:
        receipt = contract.loads(_read(args.receipt, args.receipt_sha256))
        contract.validate_receipt(receipt)
        request = contract.loads(_read(args.abi_request, receipt["abi_request_sha256"]))
        native = (args.evidence, args.evidence_sha256, args.artifacts)
        if args.emit_static_report:
            check(
                args.abi_expectations is not None
                and args.static_report_sha256 is None
                and not any(native),
                "STATIC_REPORT_GENERATION_INPUTS_REQUIRED",
            )
            expected = contract.loads(
                _read(args.abi_expectations, receipt["abi_expectations_sha256"])
            )
            raw = contract.canonical(abi.diagnose(request, expected))
            _objects(contract.loads(raw), receipt, request)
            print(raw.decode("ascii"))
            return 0
        check(args.abi_expectations is None, "STATIC_REPORT_GENERATION_INPUTS_REQUIRED")
        report = contract.loads(_read(args.static_report, args.static_report_sha256))
        check(all(native) or not any(native), "NATIVE_EVIDENCE_INPUTS_REQUIRED")
        if all(native):
            result = assess(
                receipt,
                request,
                report,
                contract.loads(_read(args.evidence, args.evidence_sha256)),
                receipt_sha256=args.receipt_sha256,
                static_report_sha256=args.static_report_sha256,
                artifacts=args.artifacts,
            )
        else:
            _objects(report, receipt, request)
            result = {"status": "NOT_TESTED", "authorization": "NOT_GRANTED"}
        status = 0
    except (
        OSError,
        ValueError,
        TypeError,
        KeyError,
        AttributeError,
        RecursionError,
        subprocess.SubprocessError,
    ) as exc:
        result = {
            "status": "REFUSED",
            "authorization": "NOT_GRANTED",
            "reason": str(exc)
            if isinstance(exc, contract.Refusal)
            else "QUALIFICATION_INPUT_REJECTED",
        }
        status = 2
    print(contract.canonical(result).decode("ascii"))
    return status


if __name__ == "__main__":
    raise SystemExit(main())

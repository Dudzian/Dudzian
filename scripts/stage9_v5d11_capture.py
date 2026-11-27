"""Bounded host-side collection of the five authorized V5d11 cases."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import os
from pathlib import Path
import select
import stat
import subprocess
import time
from typing import Any, BinaryIO, cast

from scripts import stage9_v5d10_contract as contract
from scripts import stage9_v5d11_diagnostic as diagnostic
from scripts import stage9_v5d11_qualify as qualify

MAX_ARTIFACT = qualify.MAX_ARTIFACT
CASES = (*diagnostic.PYTHON_CASES, *(f"qemu-{fmt}" for fmt in diagnostic.IMAGE_FORMATS))
PARENT_LABEL = b"cryptohunter-observer-preflight-v5d11-a-diagnostic (enforce)"
QEMU_LABEL = (
    b"cryptohunter-observer-preflight-v5d11-a-diagnostic//&"
    b"cryptohunter-observer-preflight-v5d11-qemu-diagnostic (enforce)"
)
ENV = diagnostic.COMMAND_ENV
check = contract.require


@dataclass(frozen=True)
class Target:
    host_pid: int
    namespace_pid: int
    start: int

    @property
    def proc(self) -> Path:
        return Path("/proc") / str(self.host_pid)


@dataclass(frozen=True)
class Snapshot:
    maps: bytes
    raw_maps: bytes
    status: bytes
    label: bytes
    mapped: dict[str, tuple[int, ...]]


def bounded(path: Path, limit: int = MAX_ARTIFACT) -> bytes:
    with path.open("rb") as stream:
        raw = stream.read(limit + 1)
    check(0 < len(raw) <= limit, "CAPTURE_ARTIFACT_SIZE")
    return raw


def artifact(output: Path, name: str, raw: bytes) -> dict[str, str]:
    check(0 < len(raw) <= MAX_ARTIFACT, "CAPTURE_ARTIFACT_SIZE")
    check(name == Path(name).name and name not in {".", ".."}, "CAPTURE_ARTIFACT_PATH")
    fd = os.open(output / name, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as file:
        file.write(raw)
    return {"path": name, "sha256": contract.digest(raw)}


def field(status: bytes, key: bytes) -> list[bytes]:
    found = [
        row.partition(b":")[2].split() for row in status.splitlines() if row.startswith(key + b":")
    ]
    check(len(found) == 1, "CAPTURE_PROC_STATUS")
    return found[0]


def ticks(proc: Path) -> int:
    # comm may contain ')' and spaces; split at its final delimiter.
    raw = bounded(proc / "stat", 4096).rpartition(b") ")[2].split()
    check(len(raw) > 19 and raw[19].isdigit() and int(raw[19]) > 0, "CAPTURE_PROC_LIFETIME")
    return int(raw[19])


def namespaces(proc: Path) -> dict[str, int]:
    return {name: (proc / "ns" / name).stat().st_ino for name in ("mnt", "pid", "net")}


def target(carrier: Target, pid: Any, expected: dict[str, int]) -> Target:
    check(type(pid) is int and pid > 1 and ticks(carrier.proc) == carrier.start, "CAPTURE_TARGET")
    raw = bounded(carrier.proc / "task" / str(carrier.host_pid) / "children", 4096)
    candidates = raw.split()
    check(len(candidates) == 1 and candidates[0].isdigit(), "CAPTURE_CHILD_AMBIGUOUS")
    host_pid = int(candidates[0])
    proc = Path("/proc") / str(host_pid)
    stamp = ticks(proc)
    status = bounded(proc / "status", 65536)
    check(
        field(status, b"PPid") == [str(carrier.host_pid).encode()]
        and field(status, b"NSpid") == [str(host_pid).encode(), str(pid).encode()]
        and namespaces(proc) == expected
        and ticks(proc) == stamp,
        "CAPTURE_CHILD_IDENTITY",
    )
    return Target(host_pid, pid, stamp)


def file_identity(info: os.stat_result) -> tuple[int, int, int]:
    return (
        int(getattr(os, "major")(info.st_dev)),
        int(getattr(os, "minor")(info.st_dev)),
        info.st_ino,
    )


def normalize_maps(raw: bytes, staging_root: Path) -> bytes:
    prefix = os.fsencode(staging_root) + b"/"
    records = []
    for line in raw.splitlines():
        parts = line.split(maxsplit=5)
        if len(parts) == 6 and parts[5].startswith(prefix):
            parts[5] = b"/" + parts[5][len(prefix) :]
        records.append(b" ".join(parts))
    return b"\n".join(records) + b"\n"


def snapshot(
    subject: Target,
    expected: dict[str, int],
    staging_root: Path,
    executable: str,
    canonical: Callable[[str], str],
    *,
    qemu: bool,
) -> Snapshot | None:
    proc = subject.proc
    try:
        before = ticks(proc)
        check(before == subject.start, "CAPTURE_CHILD_REUSED")
        actual = (proc / "exe").stat()
        label = bounded(proc / "attr/apparmor/current", 4096).strip()
        object_info = (staging_root / executable[1:]).stat()
        if qemu and label == PARENT_LABEL:
            return None  # forked worker is waiting to cross execve
        check(
            file_identity(actual) == file_identity(object_info)
            and label == (QEMU_LABEL if qemu else PARENT_LABEL),
            "CAPTURE_EXECUTABLE_IDENTITY",
        )
        status = bounded(proc / "status", 65536)
        qualify._kernel_status(status, subject.namespace_pid)
        environment = bounded(proc / "environ", 4096).split(b"\0")
        check(
            len(environment) == len(diagnostic.TRACE_ENV) + 1
            and set(environment)
            == {b"", *(f"{k}={v}".encode() for k, v in diagnostic.TRACE_ENV.items())},
            "CAPTURE_ENVIRONMENT_REJECTED",
        )
        raw_maps = bounded(proc / "maps")
        raw = normalize_maps(raw_maps, staging_root)
        mapped = qualify._maps(raw, canonical)
        check(
            ticks(proc) == before and namespaces(proc) == expected and executable in mapped,
            "CAPTURE_CHILD_IDENTITY",
        )
        return Snapshot(raw, raw_maps, status, label + b"\n", mapped)
    except (FileNotFoundError, ProcessLookupError):
        if qemu:
            return None
        raise


class Channel:
    """Demultiplex stdout events from glibc stderr without losing case barriers."""

    def __init__(self, process: subprocess.Popen[bytes], timeout: float = 90) -> None:
        assert (
            process.stdin is not None and process.stdout is not None and process.stderr is not None
        )
        self.input: BinaryIO = cast(BinaryIO, process.stdin)
        self.events = process.stdout.fileno()
        self.trace_fd = process.stderr.fileno()
        self.deadline = time.monotonic() + timeout
        self.pending = bytearray()
        self.trace = bytearray()

    def command(self, raw: bytes) -> None:
        self.input.write(raw)
        self.input.flush()

    def _trace_chunk(self, sample: Callable[[], None] | None) -> None:
        if sample is not None:
            sample()
        raw = os.read(self.trace_fd, 512 if sample is not None else 65536)
        check(bool(raw), "CAPTURE_TRACE_CLOSED")
        self.trace.extend(raw)
        check(len(self.trace) <= MAX_ARTIFACT, "CAPTURE_TRACE_TOO_LARGE")

    def event(self, sample: Callable[[], None] | None = None) -> dict[str, Any]:
        while b"\n" not in self.pending:
            available = self.deadline - time.monotonic()
            check(available > 0, "CAPTURE_TIMEOUT")
            ready, _, _ = select.select([self.events, self.trace_fd], [], [], available)
            check(bool(ready), "CAPTURE_TIMEOUT")
            if self.trace_fd in ready:
                self._trace_chunk(sample)
            if self.events in ready:
                raw = os.read(self.events, 4096)
                check(bool(raw), "CAPTURE_CONTROL_CLOSED")
                self.pending.extend(raw)
                check(len(self.pending) <= 8192, "CAPTURE_EVENT_TOO_LARGE")
        line, _, remaining = self.pending.partition(b"\n")
        self.pending = remaining
        check(0 < len(line) <= 4096, "CAPTURE_EVENT_TOO_LARGE")
        return cast(dict[str, Any], contract.loads(bytes(line)))

    def finish_case(self) -> bytes:
        # Caller has received a waitpid barrier and the carrier is parked.
        while select.select([self.trace_fd], [], [], 0)[0]:
            self._trace_chunk(None)
        raw = bytes(self.trace)
        self.trace.clear()
        check(bool(raw), "CAPTURE_LOADER_TRACE_MISSING")
        return raw


def expect(
    event: dict[str, Any], case: str, kind: str, extra: set[str], pid: int | None = None
) -> None:
    check(
        set(event) == {"event", "case", "pid", *extra}
        and event["event"] == kind
        and event["case"] == case
        and type(event["pid"]) is int
        and event["pid"] > 1
        and (pid is None or event["pid"] == pid),
        "CAPTURE_CASE_PROTOCOL",
    )


def object_evidence(
    output: Path,
    case: str,
    staging_root: Path,
    trace: bytes,
    sampled: Snapshot,
    subject: Target,
    canonical: Callable[[str], str],
    pins: dict[str, str],
    request: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    traced, _, _, _, _, _ = qualify._trace(
        trace, subject.namespace_pid, canonical, request["library_map"]
    )
    paths = sorted((traced | sampled.mapped.keys()) - {qualify.VDSO})
    objects: dict[str, Any] = {}
    symbols: dict[str, Any] = {}
    for i, name in enumerate(paths):
        original = staging_root / name[1:]
        info = original.lstat()
        identity = file_identity(info)
        raw = contract.secure_read(original, limit=MAX_ARTIFACT)
        digest = contract.digest(raw)
        check(
            stat.S_ISREG(info.st_mode)
            and info.st_uid == 0
            and info.st_nlink == 1
            and digest == pins[name]
            and identity == sampled.mapped.get(name, identity)
            and file_identity(original.lstat()) == identity,
            "CAPTURE_OBJECT_PIN_MISMATCH",
        )
        result = subprocess.run(
            ["/usr/bin/readelf", "--dyn-syms", "--wide", str(original)],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            env=ENV,
            cwd="/",
            timeout=10,
            check=False,
        )
        check(result.returncode == 0, "CAPTURE_SYMBOLS_FAILED")
        qualify._symbols(result.stdout)
        symbols[name] = artifact(output, f"{case}.{i}.symbols", result.stdout)
        objects[name] = {"sha256": digest, "identity": list(identity)}
    return objects, symbols


def collect(
    process: subprocess.Popen[bytes],
    host_pid: int,
    view: dict[str, Any],
    receipt: dict[str, Any],
    request: dict[str, Any],
    report: dict[str, Any],
    output: Path,
    *,
    staging_root: Path,
    on_libvirt: Callable[[int], list[str]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Drive only the parked fixed carrier, then emit exact qualifier case rows."""
    carrier_proc = Path("/proc") / str(host_pid)
    expected = view["namespaces"]
    initial = bounded(carrier_proc / "status", 65536)
    check(
        type(host_pid) is int
        and host_pid > 1
        and field(initial, b"PPid") == [str(process.pid).encode()]
        and field(initial, b"NSpid") == [str(host_pid).encode(), b"1"]
        and namespaces(carrier_proc) == expected,
        "CAPTURE_CARRIER_IDENTITY",
    )
    carrier = Target(host_pid, 1, ticks(carrier_proc))
    pins, aliases, _ = qualify._objects(report, receipt, request)
    channel = Channel(process)
    channel.command(b"DIAGNOSTIC\n")
    rows: dict[str, Any] = {}
    libvirt: dict[str, Any] = {}
    for case in CASES:
        first = channel.event()
        qemu = case.startswith("qemu-")
        libvirt_case = case == "python-libvirt"
        initial_extra = {"argv"} if qemu else ({"view"} if libvirt_case else set())
        expect(first, case, "begin" if qemu else "ready", initial_extra)
        subject = target(carrier, first["pid"], expected)
        executable = contract.QEMU_IMG if qemu else request["interpreter"]
        argv0 = first["argv"][0] if qemu else executable
        approved = set(pins)

        def canonical(path: str) -> str:
            return cast(
                str, qualify._canonical(path, approved, aliases, argv0=argv0, executable=executable)
            )

        if qemu:
            snapshots: dict[str, Snapshot] = {}

            def observe() -> None:
                sample = snapshot(subject, expected, staging_root, executable, canonical, qemu=True)
                if sample is not None and (
                    "best" not in snapshots or len(sample.mapped) >= len(snapshots["best"].mapped)
                ):
                    snapshots["best"] = sample

            channel.command(b"RUN\n")
            result = channel.event(observe)
            sample = snapshots.get("best")
            expect(result, case, "result", {"outcome"}, subject.namespace_pid)
            check(result["outcome"]["argv"] == first["argv"], "CAPTURE_QEMU_ARGV_DRIFT")
            outcome: Any = result["outcome"]
        else:
            sample = snapshot(subject, expected, staging_root, executable, canonical, qemu=False)
            if libvirt_case:
                libvirt["view"] = first["view"]
                libvirt["connect_paths"] = on_libvirt(subject.host_pid)
            channel.command(b"ACK\n")
            result = channel.event()
            expect(
                result,
                case,
                "result",
                {"read_results"} if libvirt_case else set(),
                subject.namespace_pid,
            )
            if libvirt_case:
                libvirt["read_results"] = result["read_results"]
            expect(channel.event(), case, "end", set(), subject.namespace_pid)
            outcome = "PASS"
        check(sample is not None, "CAPTURE_PROCESS_MAPS_UNOBSERVED")
        assert sample is not None
        trace = channel.finish_case()
        objects, symbols = object_evidence(
            output,
            case,
            staging_root,
            trace,
            sample,
            subject,
            canonical,
            pins,
            request,
        )
        artifact(output, f"{case}.maps.raw", sample.raw_maps)
        row = {
            "pid": subject.namespace_pid,
            "argv0": argv0,
            "trace": artifact(output, f"{case}.trace", trace),
            "maps": artifact(output, f"{case}.maps", sample.maps),
            "status": artifact(output, f"{case}.status", sample.status),
            "label": artifact(output, f"{case}.label", sample.label),
            "objects": objects,
            "symbols": symbols,
            "outcome": outcome,
        }
        rows[case] = row
        channel.command(b"ACK\n")
    check(process.wait(timeout=10) == 0, "CAPTURE_CARRIER_FAILED")
    return rows, libvirt

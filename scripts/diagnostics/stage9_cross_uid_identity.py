"""Wąska próba odmowy klienta; uruchamiać wyłącznie w zatwierdzonej VM.

Listener liczy bajty i nie implementuje protokołu ani operacji przygotowania.
Żaden tryb nie instaluje plików, kont, jednostek, bibliotek ani uprawnień.
"""

from __future__ import annotations

import argparse
import errno
import grp
import json
import os
from pathlib import Path
import pwd
import signal
import socket
import stat
import struct
import subprocess
import sys
import time
from collections.abc import Callable
from types import FrameType
from typing import Any

UNIT = "cryptohunter-freshness-verifier.service"
ADDRESS = "/run/cryptohunter/freshness/verifier.sock"
REPORT = Path("/run/cryptohunter/freshness/native-listener.json")
RUNTIME = "os_freshness_runtime"
VERIFIER = "os_freshness_crypto_verifier"
GROUP = "freshness_verifier_ipc"
PROPERTIES = (
    "MainPID",
    "ExecMainPID",
    "ExecMainStartTimestampMonotonic",
    "ControlGroup",
    "ExecStart",
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def emit(value: object) -> None:
    print(json.dumps(value, sort_keys=True, ensure_ascii=False), flush=True)


def systemd() -> dict[str, str]:
    result = subprocess.run(
        [
            "/usr/bin/systemctl",
            "show",
            UNIT,
            "--no-pager",
            *[f"--property={p}" for p in PROPERTIES],
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=5,
        stdin=subprocess.DEVNULL,
        cwd="/",
        env={"PATH": "/usr/sbin:/usr/bin:/sbin:/bin", "LANG": "C", "LC_ALL": "C"},
    )
    fields = dict(line.split("=", 1) for line in result.stdout.splitlines())
    require(set(fields) == set(PROPERTIES), "Niepełna obserwacja systemd")
    require(int(fields["MainPID"]) == int(fields["ExecMainPID"]) > 0, "Niezgodny MainPID")
    require(int(fields["ExecMainStartTimestampMonotonic"]) > 0, "Brak okresu usługi")
    require(fields["ControlGroup"] == f"/system.slice/{UNIT}", "Inna cgroup")
    return fields


def boot_id() -> str:
    return Path("/proc/sys/kernel/random/boot_id").read_text().strip()


def require_linux() -> None:
    require(sys.platform == "linux", "Wymagany natywny Linux")
    require(Path("/proc/1/comm").read_text().strip() == "systemd", "Wymagany systemd jako PID 1")
    require(
        Path("/etc/os-release").read_text().find('VERSION_ID="24.04"') >= 0, "Wymagana Ubuntu 24.04"
    )
    require(Path("/etc/os-release").read_text().find("ID=ubuntu\n") >= 0, "Wymagana Ubuntu")


def listener() -> None:
    require_linux()
    account = pwd.getpwnam(VERIFIER)
    gid = grp.getgrnam(GROUP).gr_gid
    require(os.getresuid() == (account.pw_uid,) * 3 and account.pw_uid > 0, "Inny UID listenera")
    require(os.getresgid() == (gid,) * 3, "Inny GID listenera")
    require(set(os.getgroups()) <= {gid}, "Nadmiarowe grupy listenera")
    require(not Path(ADDRESS).exists() and not REPORT.exists(), "Zastany socket lub raport")
    observations = []
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as server:
        # Jednowątkowy fixture publikuje socket od razu z dokładnym trybem 0660.
        previous_umask = os.umask(0o117)
        try:
            server.bind(ADDRESS)
        finally:
            os.umask(previous_umask)
        server.listen(2)
        server.settimeout(45)
        for _ in range(2):
            peer, _ = server.accept()
            with peer:
                peer.settimeout(5)
                credentials = struct.unpack(
                    "3i", peer.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, 12)
                )
                count = 0
                while chunk := peer.recv(4096):
                    count += len(chunk)
                    require(count <= 8 * 1024 * 1024, "Nadmierna liczba bajtów")
                observations.append(
                    {
                        "pid": credentials[0],
                        "uid": credentials[1],
                        "gid": credentials[2],
                        "bytes": count,
                        "eof": True,
                    }
                )
    report = {
        "scope": "TEST_ONLY_LISTENER",
        "boot_id": boot_id(),
        "pid": os.getpid(),
        "connections": observations,
        "preparation_operation": "NIE_ISTNIEJE_W_LISTENERZE",
        "product_qualification": "BLOCKED",
    }
    with REPORT.open("x", encoding="utf-8") as output:
        json.dump(report, output, sort_keys=True, ensure_ascii=False)
    emit(report)
    # Proces pozostaje żywy do zebrania raportów i jawnego zatrzymania przez operatora.
    signal.pause()


def approved(args: argparse.Namespace) -> dict[str, Any]:
    from deployment.stage9_vm_environment import require_approved_vm

    result: dict[str, Any] = require_approved_vm(
        revision=args.revision, environment_id=args.environment_id
    )
    return result


def snapshot(args: argparse.Namespace) -> None:
    require_linux()
    approval = approved(args)
    from bot_core.linux_service_ipc import capture_process_identity

    before = systemd()
    process = capture_process_identity(int(before["MainPID"]))
    require(systemd() == before, "Zmiana okresu usługi podczas snapshotu")
    require(process["uid"] == pwd.getpwnam(VERIFIER).pw_uid, "Inny UID procesu")
    require(process["gid"] == grp.getgrnam(GROUP).gr_gid, "Inny GID procesu")
    require(process["cgroup"] == before["ControlGroup"], "Inna cgroup procesu")
    emit(
        {
            "scope": "TEST_ONLY_ROOT_SNAPSHOT",
            "boot_id": boot_id(),
            "revision": args.revision,
            "environment_id": args.environment_id,
            "vm_image_sha256": approval["approval"]["vm_image_sha256"],
            "systemd": before,
            "process": process,
        }
    )


def protected_snapshot(path: Path) -> dict[str, Any]:
    require(path.is_absolute(), "Wymagana bezwzględna ścieżka snapshotu")
    for item in (path, *path.parents):
        info = item.lstat()
        require(
            info.st_uid == 0 and not info.st_mode & 0o022 and not stat.S_ISLNK(info.st_mode),
            "Niezabezpieczony snapshot lub przodek",
        )
    info = path.lstat()
    require(
        stat.S_ISREG(info.st_mode) and info.st_nlink == 1 and info.st_size < 65536,
        "Nieprawidłowy snapshot",
    )
    value: dict[str, Any] = json.loads(path.read_text())
    require(
        type(value) is dict and value["scope"] == "TEST_ONLY_ROOT_SNAPSHOT", "Inny zakres snapshotu"
    )
    require(value["boot_id"] == boot_id(), "Snapshot z innego bootu")
    return value


def executable_denial(pid: int) -> int:
    try:
        descriptor = os.open(f"/proc/{pid}/exe", os.O_RDONLY)
    except PermissionError as exc:
        if exc.errno is None or exc.errno not in {errno.EACCES, errno.EPERM}:
            raise ValueError("Inny błąd executable") from exc
        return exc.errno
    else:
        os.close(descriptor)
        raise ValueError("Executable jest dostępne; ten przypadek negatywny nie występuje")


def require_guard_trace(trace: dict[str, Any], observed: dict[str, Any]) -> None:
    require(
        type(trace) is dict and set(trace) == {"calls", "exceptions"},
        "Brak pełnego rejestru kontroli serwera",
    )
    require(
        trace["calls"] == [observed], "Brak dokładnego wywołania kontroli serwera przez klienta"
    )
    require(
        trace["exceptions"]
        == [
            {
                "type": "PermissionError",
                "reason": "APPROVED_VERIFIER_PROCESS_REQUIRED",
                "errno": None,
            }
        ],
        "Brak konkretnej odmowy kontroli serwera przez klienta",
    )


def probe(path: Path) -> None:
    require_linux()
    from bot_core.freshness_runtime_ipc import (
        FreshnessIPCOutcome,
        RuntimeFreshnessVerifierClient,
        _check_server,
        _require_runtime_identity,
    )
    from bot_core.linux_service_ipc import PinnedReadOnlyPeer, capture_process_identity
    from deployment.linux_physical_ipc import ExpectedPeerProcess, _pinned_verifier_socket

    snapshot_value = protected_snapshot(path)
    process = snapshot_value["process"]
    expected = ExpectedPeerProcess(
        process["pid"],
        process["uid"],
        process["gid"],
        process["start_time_ticks"],
        process["exe_sha256"],
    )
    _require_runtime_identity(expected)
    require(
        expected.uid != os.getuid() and expected.gid != os.getgid(), "UID/GID muszą być odrębne"
    )
    require(systemd() == snapshot_value["systemd"], "Zmiana usługi od snapshotu")
    executable_errno = executable_denial(expected.pid)
    with _pinned_verifier_socket(ADDRESS, expected) as (address, recheck):
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as peer:
            peer.settimeout(5)
            peer.connect(address)
            with PinnedReadOnlyPeer(peer, uid=expected.uid, gid=expected.gid) as pinned:
                require(
                    pinned.process["exe_sha256"] is None,
                    "Brak rzeczywistej niedostępności executable",
                )
                require(
                    all(
                        value == pinned.process[key]
                        for key, value in process.items()
                        if key != "exe_sha256"
                    ),
                    "Inne przesłanki procesu nie są zgodne",
                )
                try:
                    _check_server(pinned, expected)
                except PermissionError as exc:
                    require(
                        str(exc) == "APPROVED_VERIFIER_PROCESS_REQUIRED",
                        "Inna odmowa kontroli serwera",
                    )
                    server_denial = str(exc)
                    server_denial_errno = exc.errno
                else:
                    raise ValueError("Kontrola przyjęła brak executable")
                recheck()
                observed = pinned.process
                credentials = pinned.credentials
    # Syntetyczne bajty są celowo niesekretne; żadna część ramki nie może trafić do listenera.
    trace: dict[str, Any] = {"calls": [], "exceptions": []}

    def observe_guard(frame: FrameType, event: str, argument: Any) -> Callable[..., Any] | None:
        if frame.f_code is not _check_server.__code__:
            return None
        if event == "call":
            current = dict(frame.f_locals["server"].process)
            current["groups"] = list(current["groups"])
            trace["calls"].append(current)
        elif event == "exception":
            exception = argument[1]
            trace["exceptions"].append(
                {
                    "type": type(exception).__name__,
                    "reason": str(exception),
                    "errno": getattr(exception, "errno", None),
                }
            )
        return observe_guard

    # Obserwator nie modyfikuje locals/globals, argumentów, wyniku ani wyjątku klienta.
    require(sys.gettrace() is None, "Zastany rejestrator wykonania Pythona")
    sys.settrace(observe_guard)
    try:
        result = RuntimeFreshnessVerifierClient(expected).verify_exact_candidate(
            b"{}", b"{}", b"{}"
        )
    finally:
        sys.settrace(None)
    require_guard_trace(trace, observed)
    require(result.outcome is FreshnessIPCOutcome.UNAVAILABLE, "Klient nie odmówił przed wysłaniem")
    require(result.reason == "TRANSPORT_OR_IDENTITY_UNAVAILABLE", "Inny powód wyniku klienta")
    require(systemd() == snapshot_value["systemd"], "Usługa zmieniła się podczas próby")
    _require_runtime_identity(expected)
    require(
        capture_process_identity(expected.pid, allow_missing_executable=True) == observed,
        "Proces zmienił się podczas rzeczywistego wywołania klienta",
    )
    require(executable_denial(expected.pid) == executable_errno, "Zmieniła się odmowa executable")
    emit(
        {
            "scope": "TEST_ONLY_CLIENT_DENIAL",
            "result": "ODMOWA_POTWIERDZONA_OCZEKIWANIE_NA_LICZNIK",
            "boot_id": boot_id(),
            "pid": os.getpid(),
            "uid": os.getuid(),
            "gid": os.getgid(),
            "exe_errno": executable_errno,
            "server_denial": server_denial,
            "server_denial_errno": server_denial_errno,
            "peer_credentials": credentials,
            "peer_process": observed,
            "socket_bound_pidfd": True,
            "actual_server_guard": trace,
            "outcome": result.outcome.value,
            "reason": result.reason,
            "product_qualification": "BLOCKED",
        }
    )


def compare(client_path: Path, server_path: Path) -> None:
    client = json.loads(client_path.read_text())
    # Oczekiwanie dotyczy tylko publikacji lokalnego licznika, nigdy ponowienia wymiany.
    deadline = time.monotonic() + 5
    while True:
        try:
            server = json.loads(server_path.read_text())
            break
        except (FileNotFoundError, json.JSONDecodeError):
            require(time.monotonic() < deadline, "Brak ukończonego raportu listenera")
            time.sleep(0.05)
    require(
        client["scope"] == "TEST_ONLY_CLIENT_DENIAL" and server["scope"] == "TEST_ONLY_LISTENER",
        "Inny zakres raportów",
    )
    require(client["boot_id"] == server["boot_id"] == boot_id(), "Inny boot raportów")
    require(client["socket_bound_pidfd"] is True, "Brak dowodu socket-bound pidfd")
    require(type(client.get("actual_server_guard")) is dict, "Brak rejestru wykonania klienta")
    require_guard_trace(client["actual_server_guard"], client["peer_process"])
    require(
        client["server_denial"] == "APPROVED_VERIFIER_PROCESS_REQUIRED"
        and client["server_denial_errno"] is None,
        "Inna odmowa kontroli serwera",
    )
    require(
        client["uid"] != client["peer_credentials"]["uid"]
        and client["gid"] != client["peer_credentials"]["gid"],
        "Raporty nie dotyczą odrębnych UID/GID",
    )
    require(
        all(
            client["peer_process"][key] == value
            for key, value in client["peer_credentials"].items()
        ),
        "Rozbieżne credentials i proces partnera",
    )
    require(
        client["product_qualification"] == server["product_qualification"] == "BLOCKED"
        and server["preparation_operation"] == "NIE_ISTNIEJE_W_LISTENERZE",
        "Raport rozszerza zakres kwalifikacji",
    )
    require(client["peer_credentials"]["pid"] == server["pid"], "Inny listener")
    require(len(server["connections"]) == 2, "Brak obu rzeczywistych połączeń")
    require(
        all(
            item
            == {
                "pid": client["pid"],
                "uid": client["uid"],
                "gid": client["gid"],
                "bytes": 0,
                "eof": True,
            }
            for item in server["connections"]
        ),
        "Połączenia przesłały dane lub nie odpowiadają klientowi",
    )
    require(
        client["exe_errno"] in {errno.EACCES, errno.EPERM}
        and client["peer_process"]["exe_sha256"] is None,
        "Brak dowodu odmowy procfs",
    )
    require(
        client["outcome"] == "UNAVAILABLE"
        and client["reason"] == "TRANSPORT_OR_IDENTITY_UNAVAILABLE",
        "Inny wynik klienta",
    )
    emit(
        {
            "scope": "TEST_ONLY_NATIVE_DENIAL",
            "result": "PASS",
            "bytes_received": 0,
            "connections": 2,
            "preparation_operation": "NIE_ISTNIEJE_W_LISTENERZE",
            "postgresql_no_preparation_proof": "NOT_RUN",
            "product_qualification": "BLOCKED",
        }
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    sub.add_parser("listener")
    for mode in ("preflight", "snapshot"):
        item = sub.add_parser(mode)
        item.add_argument("--revision", required=True)
        item.add_argument("--environment-id", required=True)
    sub.add_parser("probe").add_argument("snapshot", type=Path)
    item = sub.add_parser("compare")
    item.add_argument("client", type=Path)
    item.add_argument("server", type=Path)
    args = parser.parse_args()
    try:
        if args.mode == "listener":
            listener()
        elif args.mode == "preflight":
            emit(approved(args))
        elif args.mode == "snapshot":
            snapshot(args)
        elif args.mode == "probe":
            probe(args.snapshot)
        else:
            compare(args.client, args.server)
    except (
        OSError,
        ValueError,
        KeyError,
        TypeError,
        RuntimeError,
        ImportError,
        subprocess.SubprocessError,
    ) as exc:
        emit({"result": "BLOCKED", "reason": str(exc), "product_qualification": "BLOCKED"})
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

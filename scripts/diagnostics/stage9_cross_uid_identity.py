"""Wąska próba odmowy klienta; uruchamiać wyłącznie w zatwierdzonej VM.

Listener liczy bajty i nie implementuje protokołu ani operacji przygotowania.
Narzędzie nie instaluje produktu, kont, usług, bibliotek ani uprawnień.
Administrator może opublikować manifest próby i przejąć jej dowody.
"""

from __future__ import annotations

import argparse
import errno
import grp
import hashlib
import json
import os
from pathlib import Path
import pwd
import select
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

from scripts.diagnostics.stage9_cross_uid_trial import (
    MANIFEST,
    ROOT,
    binding,
    create_trial,
    read_protected,
    require_trial,
    strict_json,
    write_new,
)

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


def capture_snapshot(trial: dict[str, Any]) -> dict[str, Any]:
    """Niezależny od raportów endpointów odczyt administratora lub własnego procesu."""
    if sys.platform != "linux":
        raise ValueError("Wymagany natywny Linux")
    from bot_core.linux_service_ipc import capture_process_identity

    before = systemd()
    process = capture_process_identity(int(before["MainPID"]))
    require(systemd() == before, "Zmiana okresu usługi podczas snapshotu")
    require(process["uid"] == pwd.getpwnam(VERIFIER).pw_uid > 0, "Inny UID procesu")
    require(process["gid"] == grp.getgrnam(GROUP).gr_gid > 0, "Inny GID procesu")
    require(set(process["groups"]) <= {process["gid"]}, "Nadmiarowe grupy procesu")
    require(process["cgroup"] == before["ControlGroup"], "Inna cgroup procesu")
    return {
        "scope": "TEST_ONLY_ROOT_SNAPSHOT",
        "trial": binding(trial),
        "systemd": before,
        "process": process,
    }


def listener(args: argparse.Namespace) -> None:
    # Zgoda jest sprawdzana przed utworzeniem socketu i zajęciem nazwy produkcyjnej.
    trial = require_trial(args)
    if sys.platform != "linux":
        raise ValueError("Wymagany natywny Linux")
    account = pwd.getpwnam(VERIFIER)
    gid = grp.getgrnam(GROUP).gr_gid
    require(os.getresuid() == (account.pw_uid,) * 3 and account.pw_uid > 0, "Inny UID listenera")
    require(os.getresgid() == (gid,) * 3, "Inny GID listenera")
    require(set(os.getgroups()) <= {gid}, "Nadmiarowe grupy listenera")
    require(not Path(ADDRESS).exists() and not REPORT.exists(), "Zastany socket lub raport")
    own = capture_snapshot(trial)
    require(own["process"]["pid"] == os.getpid(), "Listener nie jest MainPID")
    observations = []
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as server:
        # Jednowątkowy fixture publikuje socket od razu z dokładnym trybem 0660.
        previous_umask = os.umask(0o117)
        try:
            server.bind(ADDRESS)
        finally:
            os.umask(previous_umask)
        server.listen(2)
        server.settimeout(120)
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
    require(require_trial(args) == trial, "Zatwierdzenie zmieniło się podczas próby")
    require(capture_snapshot(trial) == own, "Zmiana procesu listenera")
    report = {
        "scope": "TEST_ONLY_LISTENER",
        "trial": binding(trial),
        "systemd": own["systemd"],
        "process": own["process"],
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
    trial = require_trial(args, administrator=True)
    emit(capture_snapshot(trial))


def protected_snapshot(path: Path, trial: dict[str, Any]) -> tuple[dict[str, Any], bytes]:
    raw = read_protected(path)
    value = strict_json(raw)
    require(
        type(value) is dict and value["scope"] == "TEST_ONLY_ROOT_SNAPSHOT", "Inny zakres snapshotu"
    )
    require(value["trial"] == binding(trial), "Snapshot z innej zatwierdzonej próby")
    return value, raw


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


def probe(args: argparse.Namespace) -> None:
    trial = require_trial(args)
    if sys.platform != "linux":
        raise ValueError("Wymagany natywny Linux")
    from bot_core.freshness_runtime_ipc import (
        FreshnessIPCOutcome,
        RuntimeFreshnessVerifierClient,
        _check_server,
        _require_runtime_identity,
    )
    from bot_core.linux_service_ipc import PinnedReadOnlyPeer, capture_process_identity
    from deployment.linux_physical_ipc import ExpectedPeerProcess, _pinned_verifier_socket

    snapshot_value, snapshot_raw = protected_snapshot(args.snapshot, trial)
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
    require(require_trial(args) == trial, "Zatwierdzenie zmieniło się podczas próby")
    emit(
        {
            "scope": "TEST_ONLY_CLIENT_DENIAL",
            "result": "ODMOWA_POTWIERDZONA_OCZEKIWANIE_NA_LICZNIK",
            "trial": binding(trial),
            "systemd": snapshot_value["systemd"],
            "root_snapshot_sha256": hashlib.sha256(snapshot_raw).hexdigest(),
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


def validate_reports(
    client: dict[str, Any],
    server: dict[str, Any],
    snapshot_value: dict[str, Any],
    *,
    snapshot_sha256: str,
) -> None:
    """Sprawdź spójność treści; sam parser nigdy nie stwierdza wykonania próby."""
    try:
        _validate_reports(client, server, snapshot_value, snapshot_sha256=snapshot_sha256)
    except (KeyError, TypeError, AttributeError, IndexError) as exc:
        raise ValueError("Brak wymaganej obserwacji lub nieprawidłowy typ raportu") from exc


def _validate_reports(
    client: dict[str, Any],
    server: dict[str, Any],
    snapshot_value: dict[str, Any],
    *,
    snapshot_sha256: str,
) -> None:
    from deployment.linux_physical_ipc import ExpectedPeerProcess

    require(snapshot_value["scope"] == "TEST_ONLY_ROOT_SNAPSHOT", "Inny zakres snapshotu")
    trial_binding = snapshot_value["trial"]
    require(
        type(trial_binding) is dict
        and set(trial_binding)
        == {
            "scope",
            "trial_id",
            "revision",
            "environment_id",
            "machine_id",
            "boot_id",
            "vm_image_sha256",
            "manifest_sha256",
        }
        and trial_binding["scope"] == "TEST_ONLY_STAGE9_CROSS_UID_DENIAL"
        and all(type(value) is str and bool(value) for value in trial_binding.values()),
        "Niepełne powiązanie zatwierdzonej próby",
    )
    process = snapshot_value["process"]
    require(
        set(process) == {"pid", "uid", "gid", "groups", "start_time_ticks", "exe_sha256", "cgroup"},
        "Brak niezależnej obserwacji procesu",
    )
    ExpectedPeerProcess(
        process["pid"],
        process["uid"],
        process["gid"],
        process["start_time_ticks"],
        process["exe_sha256"],
    )
    require(
        min(process["uid"], process["gid"]) > 0
        and type(process["groups"]) is list
        and all(type(group) is int for group in process["groups"])
        and process["groups"] == sorted(set(process["groups"]))
        and set(process["groups"]) <= {process["gid"]}
        and process["cgroup"] == f"/system.slice/{UNIT}",
        "Niezgodna izolacja procesu snapshotu",
    )
    properties = snapshot_value["systemd"]
    require(
        set(properties) == set(PROPERTIES)
        and all(type(value) is str for value in properties.values())
        and properties["MainPID"] == properties["ExecMainPID"] == str(process["pid"])
        and properties["ExecMainStartTimestampMonotonic"].isdecimal()
        and int(properties["ExecMainStartTimestampMonotonic"]) > 0
        and properties["ControlGroup"] == process["cgroup"]
        and bool(properties["ExecStart"]),
        "Niezgodny MainPID lub okres jednostki",
    )
    require(
        client["scope"] == "TEST_ONLY_CLIENT_DENIAL" and server["scope"] == "TEST_ONLY_LISTENER",
        "Inny zakres raportów",
    )
    require(
        client["trial"] == server["trial"] == snapshot_value["trial"],
        "Raporty lub snapshot dotyczą innej zatwierdzonej próby",
    )
    require(
        client["systemd"] == server["systemd"] == properties,
        "Niezgodny MainPID lub okres jednostki",
    )
    require(
        client["root_snapshot_sha256"] == snapshot_sha256,
        "Raport klienta nie jest związany z dokładnym snapshotem",
    )
    observed = {**process, "exe_sha256": None}
    require(client["peer_process"] == observed, "Niezgodna niezależna obserwacja procesu")
    require(server["process"] == process, "Raport listenera nie odpowiada snapshotowi")
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
        and client["gid"] != client["peer_credentials"]["gid"]
        and all(type(client[key]) is int and client[key] > 0 for key in ("pid", "uid", "gid")),
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
    require(client["peer_credentials"]["pid"] == server["pid"] == process["pid"], "Inny listener")
    require(
        type(server["connections"]) is list and len(server["connections"]) == 2,
        "Brak obu rzeczywistych połączeń",
    )
    require(
        all(
            type(item["bytes"]) is int
            and item["eof"] is True
            and item
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
        type(client["exe_errno"]) is int
        and client["exe_errno"] in {errno.EACCES, errno.EPERM}
        and client["peer_process"]["exe_sha256"] is None,
        "Brak dowodu odmowy procfs",
    )
    require(
        client["outcome"] == "UNAVAILABLE"
        and client["reason"] == "TRANSPORT_OR_IDENTITY_UNAVAILABLE",
        "Inny wynik klienta",
    )


def format_reports(client_path: Path, server_path: Path, snapshot_path: Path) -> None:
    """Tryb parsera jest dostępny poza VM i nie ma ścieżki do natywnego PASS."""
    snapshot_raw = snapshot_path.read_bytes()
    validate_reports(
        strict_json(client_path.read_bytes()),
        strict_json(server_path.read_bytes()),
        strict_json(snapshot_raw),
        snapshot_sha256=hashlib.sha256(snapshot_raw).hexdigest(),
    )
    emit(
        {
            "scope": "TEST_ONLY_REPORT_FORMAT",
            "result": "FORMAT_CONSISTENT",
            "native_status": "NOT_RUN",
            "product_qualification": "BLOCKED",
        }
    )


def encoded(value: dict[str, Any]) -> bytes:
    return (json.dumps(value, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")


def probe_command(trial: dict[str, Any], manifest: Path, snapshot_path: Path) -> list[str]:
    return [
        "/usr/sbin/runuser",
        "--user",
        RUNTIME,
        "--",
        trial["python_executable"],
        "-s",
        "-B",
        "-m",
        "scripts.diagnostics.stage9_cross_uid_identity",
        "probe",
        "--manifest",
        str(manifest),
        "--revision",
        trial["revision"],
        "--environment-id",
        trial["environment_id"],
        "--snapshot",
        str(snapshot_path),
    ]


def collect_probe(command: list[str], trial: dict[str, Any]) -> bytes:
    """Jednorazowe wykonanie zatwierdzonego klienta; bez przyjmowania cudzych JSON."""
    if sys.platform != "linux":
        raise ValueError("Wymagany natywny Linux")
    process = subprocess.Popen(
        command,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        cwd=trial["source_root"],
        env={"PATH": "/usr/sbin:/usr/bin:/sbin:/bin", "LANG": "C", "LC_ALL": "C"},
        start_new_session=True,
    )
    deadline = time.monotonic() + 90
    result = bytearray()
    try:
        pipe = process.stdout
        if pipe is None:
            raise ValueError("Brak kanału raportu klienta")
        poller = select.poll()
        poller.register(pipe.fileno(), select.POLLIN | select.POLLHUP | select.POLLERR)
        while True:
            remaining = deadline - time.monotonic()
            require(
                remaining > 0 and bool(poller.poll(max(1, int(remaining * 1000)))),
                "Timeout klienta",
            )
            chunk = os.read(pipe.fileno(), 65537 - len(result))
            if not chunk:
                break
            result.extend(chunk)
            require(len(result) <= 65536, "Nadmierny raport klienta")
        remaining = deadline - time.monotonic()
        require(remaining > 0, "Timeout zakończenia klienta")
        require(
            process.wait(timeout=remaining) == 0,
            "Rzeczywisty klient zakończył próbę odmową wykonania",
        )
        return bytes(result)
    finally:
        if process.poll() is None:
            # Grupa należy wyłącznie do procesu runuser uruchomionego przez tę próbę.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=5)
        if process.stdout is not None:
            process.stdout.close()


def collect_listener(uid: int, gid: int) -> bytes:
    """Przejmij dokładne bajty stałego licznika, bez podążania za symlinkiem."""
    deadline = time.monotonic() + 5
    while True:
        try:
            descriptor = os.open(REPORT, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
            try:
                before = os.fstat(descriptor)
                require(
                    stat.S_ISREG(before.st_mode)
                    and before.st_nlink == 1
                    and (before.st_uid, before.st_gid) == (uid, gid)
                    and stat.S_IMODE(before.st_mode) == 0o640,
                    "Niezabezpieczony raport listenera",
                )
                raw = os.read(descriptor, 65537)
                after = os.fstat(descriptor)
                metadata = lambda info: (
                    info.st_dev,
                    info.st_ino,
                    info.st_size,
                    info.st_mtime_ns,
                    info.st_ctime_ns,
                    info.st_mode,
                    info.st_uid,
                    info.st_gid,
                    info.st_nlink,
                )
                require(
                    metadata(before) == metadata(after) and len(raw) == before.st_size,
                    "Raport listenera zmienił się podczas odczytu",
                )
                strict_json(raw)
                return raw
            finally:
                os.close(descriptor)
        except (FileNotFoundError, ValueError):
            require(time.monotonic() < deadline, "Brak ukończonego raportu listenera")
            time.sleep(0.05)


def run_probe(args: argparse.Namespace) -> None:
    trial = require_trial(args, administrator=True)
    # Nowy katalog nigdy nie adoptuje poprzednich dowodów ani nie nadpisuje receipt.
    for parent in (ROOT, *ROOT.parents):
        info = parent.lstat()
        require(
            stat.S_ISDIR(info.st_mode) and info.st_uid == 0 and not info.st_mode & 0o022,
            "Niezabezpieczony katalog prób",
        )
    directory = ROOT / trial["trial_id"]
    directory.mkdir(mode=0o755)
    os.chmod(directory, 0o755)
    before = capture_snapshot(trial)
    before_raw = encoded(before)
    write_new(directory / "before.json", before_raw, 0o644)
    command = probe_command(trial, args.manifest, directory / "before.json")
    client_raw = collect_probe(command, trial)
    server_raw = collect_listener(before["process"]["uid"], before["process"]["gid"])
    require(
        require_trial(args, administrator=True) == trial, "Zmiana zatwierdzenia podczas wykonania"
    )
    after = capture_snapshot(trial)
    require(after == before, "Zmiana MainPID, procesu lub okresu podczas wykonania")
    validate_reports(
        strict_json(client_raw),
        strict_json(server_raw),
        before,
        snapshot_sha256=hashlib.sha256(before_raw).hexdigest(),
    )
    # Po zakończeniu klienta wszystkie przejęte dowody są dostępne tylko root.
    os.chmod(directory, 0o700)
    files = {
        "before.json": before_raw,
        "after.json": encoded(after),
        "client.json": client_raw,
        "listener.json": server_raw,
    }
    for name in ("after.json", "client.json", "listener.json"):
        write_new(directory / name, files[name], 0o600)
    receipt = {
        "scope": "TEST_ONLY_ROOT_EXECUTION",
        "trial": binding(trial),
        "collector": "ROOT_RUNUSER_PIPE_V1",
        "argv": command,
        "returncode": 0,
        "hashes": {name: hashlib.sha256(raw).hexdigest() for name, raw in files.items()},
    }
    write_new(directory / "execution.json", encoded(receipt), 0o600)
    compare(args)


def compare(args: argparse.Namespace) -> None:
    """Natywny wynik wymaga chronionego zapisu rzeczywistego wykonania przez root."""
    trial = require_trial(args, administrator=True)
    directory = ROOT / trial["trial_id"]
    receipt = strict_json(read_protected(directory / "execution.json"))
    require(
        receipt["scope"] == "TEST_ONLY_ROOT_EXECUTION"
        and receipt["trial"] == binding(trial)
        and receipt["collector"] == "ROOT_RUNUSER_PIPE_V1"
        and type(receipt["returncode"]) is int
        and receipt["returncode"] == 0
        and receipt["argv"] == probe_command(trial, args.manifest, directory / "before.json"),
        "Brak zapisu kontrolowanego wykonania klienta",
    )
    names = {"before.json", "after.json", "client.json", "listener.json"}
    require(set(receipt["hashes"]) == names, "Niepełny zapis integralności wykonania")
    files = {name: read_protected(directory / name) for name in names}
    require(
        all(
            hashlib.sha256(raw).hexdigest() == receipt["hashes"][name]
            for name, raw in files.items()
        ),
        "Podmieniony raport lub snapshot po przejęciu dowodów",
    )
    before = strict_json(files["before.json"])
    after = strict_json(files["after.json"])
    require(before["trial"] == binding(trial), "Snapshot z innej zatwierdzonej próby")
    require(
        before == after == capture_snapshot(trial), "Zmiana MainPID, procesu lub okresu jednostki"
    )
    validate_reports(
        strict_json(files["client.json"]),
        strict_json(files["listener.json"]),
        before,
        snapshot_sha256=hashlib.sha256(files["before.json"]).hexdigest(),
    )
    emit(
        {
            "scope": "TEST_ONLY_NATIVE_DENIAL",
            "TEST_ONLY_NATIVE_DENIAL": "PASS",
            "result": "PASS",
            "trial": binding(trial),
            "connections": 2,
            "bytes_received": 0,
            "preparation_operation": "NIE_ISTNIEJE_W_LISTENERZE",
            "postgresql_no_preparation_proof": "NOT_RUN",
            "product_qualification": "BLOCKED",
            "evidence": str(directory),
        }
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    for mode in (
        "preflight",
        "prepare-trial",
        "listener",
        "snapshot",
        "probe",
        "run-probe",
        "compare",
    ):
        item = sub.add_parser(mode)
        item.add_argument("--revision", required=True)
        item.add_argument("--environment-id", required=True)
        if mode != "preflight":
            item.add_argument("--manifest", type=Path, default=MANIFEST)
        if mode == "prepare-trial":
            item.add_argument("--trial-id", required=True)
            item.add_argument("--source-root", type=Path, required=True)
            item.add_argument("--python-executable", type=Path, required=True)
        if mode == "probe":
            item.add_argument("--snapshot", type=Path, required=True)
    item = sub.add_parser("format-reports")
    item.add_argument("client", type=Path)
    item.add_argument("server", type=Path)
    item.add_argument("snapshot", type=Path)
    args = parser.parse_args()
    try:
        if args.mode == "listener":
            listener(args)
        elif args.mode == "preflight":
            emit(approved(args))
        elif args.mode == "prepare-trial":
            emit({"scope": "TEST_ONLY_TRIAL_MANIFEST", "trial": binding(create_trial(args))})
        elif args.mode == "snapshot":
            snapshot(args)
        elif args.mode == "probe":
            probe(args)
        elif args.mode == "run-probe":
            run_probe(args)
        elif args.mode == "compare":
            compare(args)
        else:
            format_reports(args.client, args.server, args.snapshot)
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

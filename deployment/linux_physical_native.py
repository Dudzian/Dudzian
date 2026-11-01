"""Natywne odczyty i jawnie zatwierdzane działania na wydzielonym hoście Linux."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import platform
import re
import secrets
import stat
import subprocess
import sys
import time
from typing import Any

from deployment.linux_physical_observations import (
    ObservationError,
    UNIT_PROPERTIES,
    canonical,
    require,
    sha256,
    validate_installation,
    validate_observation_prefix,
    _snapshot,
)

MAX_FILE = 256 * 1024 * 1024
SAFE_ENV = {"PATH": "/usr/sbin:/usr/bin:/sbin:/bin", "LANG": "C", "LC_ALL": "C"}


def _safe_parents(path: Path, *, owners: set[int] | None = None) -> None:
    require(path.is_absolute(), "Wymagana ścieżka bezwzględna")
    for parent in reversed(path.parents):
        info = parent.lstat()
        require(
            stat.S_ISDIR(info.st_mode)
            and info.st_uid in (owners or {0})
            and not info.st_mode & 0o022,
            "Niechroniony katalog nadrzędny",
        )


def read_regular(path: Path, *, protected: bool = False, maximum: int = MAX_FILE) -> bytes:
    if protected:
        _safe_parents(path)
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        before = os.fstat(descriptor)
        require(
            stat.S_ISREG(before.st_mode) and before.st_size <= maximum,
            "Nieprawidłowy plik wejściowy",
        )
        if protected:
            require(
                before.st_uid == 0 and not before.st_mode & 0o022, "Niechroniony plik wejściowy"
            )
        result = bytearray()
        while True:
            chunk = os.read(descriptor, min(1024 * 1024, maximum + 1 - len(result)))
            if not chunk:
                break
            result.extend(chunk)
            require(len(result) <= maximum, "Przekroczony limit pliku")
        after = os.fstat(descriptor)
        require(
            (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
            == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns),
            "Plik zmienił się podczas odczytu",
        )
        return bytes(result)
    finally:
        os.close(descriptor)


def command(arguments: list[str], *, timeout: int = 30) -> subprocess.CompletedProcess[str]:
    require(
        bool(arguments)
        and arguments[0]
        in {
            "/usr/bin/systemctl",
            "/usr/bin/dpkg",
            "/usr/bin/dpkg-query",
            "/usr/bin/git",
            "/usr/bin/journalctl",
            "/usr/bin/systemd-detect-virt",
        },
        "Niedozwolone narzędzie natywne",
    )
    try:
        return subprocess.run(
            arguments,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout,
            env=SAFE_ENV,
            stdin=subprocess.DEVNULL,
        )
    except (OSError, subprocess.SubprocessError, UnicodeError) as exc:
        raise ObservationError("Nieudane lub zbyt długie wykonanie narzędzia") from exc


def host_identity() -> dict[str, Any]:
    require(
        sys.platform == "linux" and os.geteuid() == 0,
        "Wymagany natywny Linux i osobna zgoda administratora",
    )
    require(
        Path("/run/systemd/system").is_dir()
        and read_regular(Path("/proc/1/comm"), maximum=1024).strip() == b"systemd",
        "Wymagany rzeczywisty systemd jako PID 1",
    )
    machine = read_regular(Path("/etc/machine-id"), protected=True, maximum=128).strip()
    dmi = read_regular(Path("/sys/class/dmi/id/product_uuid"), maximum=4096).strip().lower()
    boot = read_regular(Path("/proc/sys/kernel/random/boot_id"), maximum=128).decode().strip()
    require(
        re.fullmatch(rb"[0-9a-f]{32}", machine) is not None
        and re.fullmatch(rb"[0-9a-f-]{36}", dmi) is not None
        and re.fullmatch(r"[0-9a-f-]{36}", boot) is not None,
        "Brak tożsamości fizycznego hosta",
    )
    os_release = read_regular(Path("/usr/lib/os-release"), protected=True, maximum=8192).decode()
    values = {}
    for line in os_release.splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            require(key not in values, "Powtórzone pole systemu operacyjnego")
            values[key] = value.strip('"')
    detected = command(["/usr/bin/systemd-detect-virt", "--quiet"])
    require(
        detected.returncode == 1, "Kontener lub maszyna wirtualna nie kwalifikuje hosta fizycznego"
    )
    require(
        values.get("ID") == "ubuntu"
        and values.get("VERSION_ID") == "24.04"
        and platform.machine() == "x86_64",
        "Wymagany Ubuntu 24.04 x86_64",
    )
    executable = Path(sys.executable).resolve(strict=True)
    return {
        "host_id": sha256(machine + b"\n" + dmi),
        "machine_id_sha256": sha256(machine),
        "dmi_product_uuid_sha256": sha256(dmi),
        "boot_id": boot,
        "os_id": values["ID"],
        "os_version": values["VERSION_ID"],
        "kernel": platform.release(),
        "architecture": platform.machine(),
        "virtualization": "none",
        "python_version": platform.python_version(),
        "python_executable_sha256": sha256(read_regular(executable, protected=True)),
    }


def file_observation(path: Path) -> dict[str, Any]:
    _safe_parents(path)
    raw = read_regular(path)
    info = path.lstat()
    require(stat.S_ISREG(info.st_mode), "Podmieniony plik")
    return {
        "path": str(path),
        "kind": "regular",
        "sha256": sha256(raw),
        "uid": info.st_uid,
        "gid": info.st_gid,
        "mode": stat.S_IMODE(info.st_mode),
    }


def _start_ticks(raw: bytes, pid: int) -> int:
    require(raw.startswith(str(pid).encode() + b" ("), "Podmieniona tożsamość PID")
    fields = raw[raw.rfind(b")") + 2 :].split()
    require(
        len(fields) >= 20 and fields[0] not in {b"Z", b"X"} and fields[19].isdigit(),
        "Nieprawidłowy proces",
    )
    result = int(fields[19])
    require(result > 0, "Nieprawidłowa tożsamość procesu")
    return result


def process_observation(pid: int) -> dict[str, Any]:
    root = Path("/proc") / str(pid)
    before = _start_ticks(read_regular(root / "stat", maximum=8192), pid)
    rows: dict[str, str] = {}
    for line in read_regular(root / "status", maximum=65536).decode().splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            require(key not in rows, "Powtórzone pole procesu")
            rows[key] = value.strip()
    uids = [int(value) for value in rows["Uid"].split()]
    gids = [int(value) for value in rows["Gid"].split()]
    require(
        len(uids) == len(gids) == 4 and len(set(uids)) == len(set(gids)) == 1,
        "Niejednoznaczne UID/GID procesu",
    )
    # /proc/<pid>/exe jest świadomie otwieranym uchwytem jądra, nie ścieżką od operatora.
    descriptor = os.open(root / "exe", os.O_RDONLY)
    try:
        hasher = hashlib.sha256()
        while chunk := os.read(descriptor, 1024 * 1024):
            hasher.update(chunk)
        executable_digest = hasher.hexdigest()
    finally:
        os.close(descriptor)
    require(
        before == _start_ticks(read_regular(root / "stat", maximum=8192), pid),
        "Proces zmienił się podczas obserwacji",
    )
    return {
        "pid": pid,
        "start_time_ticks": before,
        "uid": uids[0],
        "gid": gids[0],
        "groups": [int(value) for value in rows["Groups"].split()],
        "no_new_privs": int(rows["NoNewPrivs"]),
        "exe_sha256": executable_digest,
        "capabilities": {
            key: rows[key] for key in ("CapInh", "CapPrm", "CapEff", "CapBnd", "CapAmb")
        },
    }


class NativeDriver:
    """Stały zestaw narzędzi; kontrakt nie zawiera żadnych poleceń powłoki."""

    def __init__(self, installation: dict[str, Any], challenge_id: str):
        self.installation = validate_installation(installation)
        require(re.fullmatch(r"[0-9a-f]{64}", challenge_id) is not None, "Nieprawidłowy challenge")
        self.challenge_id = challenge_id
        self.state_path = Path(installation["state_directory"]) / (
            ".stage9-evidence-" + challenge_id
        )

    def package(self) -> dict[str, Any] | None:
        result = command(
            [
                "/usr/bin/dpkg-query",
                "--show",
                "--showformat=${Package}\t${Version}\t${Status}",
                self.installation["package_name"],
            ]
        )
        if result.returncode == 1 and not result.stdout:
            return None
        require(result.returncode == 0, "Nie można sprawdzić pakietu")
        fields = result.stdout.split("\t")
        require(len(fields) == 3, "Nieprawidłowa odpowiedź dpkg")
        return dict(zip(("name", "version", "status"), fields, strict=True))

    def properties(self, unit: str) -> dict[str, str]:
        result = command(
            [
                "/usr/bin/systemctl",
                "--system",
                "show",
                unit,
                "--property=" + ",".join(UNIT_PROPERTIES),
            ]
        )
        require(result.returncode == 0, "Brak obserwacji systemd")
        values = {}
        for line in result.stdout.splitlines():
            key, separator, value = line.partition("=")
            require(
                separator == "=" and key in UNIT_PROPERTIES and key not in values,
                "Nieprawidłowy wynik systemd",
            )
            values[key] = value
        require(set(values) == set(UNIT_PROPERTIES), "Niepełny wynik systemd")
        return values

    def journal(self, unit: str, boot: str | None = None) -> dict[str, Any]:
        args = [
            "/usr/bin/journalctl",
            "--no-pager",
            "--unit=" + unit,
            "--lines=1",
            "--output=json",
            "--output-fields=_BOOT_ID,_SYSTEMD_UNIT,UNIT,__REALTIME_TIMESTAMP",
        ]
        if boot is not None:
            args.append("--boot=" + boot.replace("-", ""))
        result = command(args)
        require(result.returncode == 0, "Nie można odczytać dziennika")
        lines = result.stdout.splitlines()
        require(len(lines) == 1, "Brak wpisu dziennika usługi")
        row = json.loads(lines[0])
        require(type(row) is dict, "Nieprawidłowy dziennik")
        return {
            "boot_id": row.get("_BOOT_ID"),
            "unit": row.get("_SYSTEMD_UNIT", row.get("UNIT")),
            "realtime_usec": int(row["__REALTIME_TIMESTAMP"]),
        }

    def postgres(self) -> dict[str, Any]:
        """Obserwacja administratora; nie dowodzi uprawnień runtime/verifier."""
        import psycopg
        from deployment.linux_physical_postgresql import blocked_service_qualification

        pg = self.installation["postgres"]
        with psycopg.connect(
            host=pg["socket"],
            port=pg["port"],
            dbname=pg["database"],
            user=pg["user"],
            password=None,
            connect_timeout=5,
            options="-c default_transaction_read_only=on -c statement_timeout=5000",
            passfile="/dev/null",
            sslmode="disable",
        ) as connection:
            settings = dict(
                connection.execute(
                    "SELECT name,setting FROM pg_settings WHERE name=ANY(%s)",
                    (["listen_addresses", "fsync", "synchronous_commit"],),
                ).fetchall()
            )
            identity = connection.execute(
                "SELECT system_user,current_user,session_user,inet_client_addr(),inet_server_addr(),"
                "coalesce((SELECT ssl FROM pg_stat_ssl WHERE pid=pg_backend_pid()),false)"
            ).fetchone()
            require(identity is not None, "Brak tożsamości PostgreSQL")
            hba = connection.execute(
                "SELECT type,database,user_name,address,netmask,auth_method,options,error "
                "FROM pg_hba_file_rules ORDER BY rule_number"
            ).fetchall()
            ident = connection.execute(
                "SELECT map_name,sys_name,pg_username,error FROM pg_ident_file_mappings ORDER BY map_number"
            ).fetchall()
            cluster = connection.execute(
                "SELECT system_identifier::text FROM pg_control_system()"
            ).fetchone()
            require(cluster is not None, "Brak tożsamości klastra")
        administrative_observation = {
            **settings,
            **dict(
                zip(
                    (
                        "system_user",
                        "current_user",
                        "session_user",
                        "client_addr",
                        "server_addr",
                        "ssl",
                    ),
                    identity,
                    strict=True,
                )
            ),
            "hba_sha256": sha256(canonical(hba)),
            "ident_sha256": sha256(canonical(ident)),
            "system_identifier": cluster[0],
        }
        return {
            "administrative_observation": administrative_observation,
            "service_qualification": blocked_service_qualification(),
        }

    def snapshot(self) -> dict[str, Any]:
        from deployment.linux_physical_ipc import blocked_ipc_qualification

        units = []
        for spec in self.installation["units"]:
            props = self.properties(spec["name"])
            cgroup = Path("/sys/fs/cgroup") / props["ControlGroup"].lstrip("/") / "cgroup.procs"
            require(props["ControlGroup"] == "/system.slice/" + spec["name"], "Podmieniona cgroup")
            pids = sorted(set(int(value) for value in read_regular(cgroup, maximum=65536).split()))
            processes = [process_observation(pid) for pid in pids]
            require(
                props == self.properties(spec["name"]), "Jednostka zmieniła się podczas obserwacji"
            )
            units.append(
                {
                    "name": spec["name"],
                    "properties": props,
                    "processes": processes,
                    "unit_file": file_observation(Path(spec["unit_file"]["path"])),
                    "executable": file_observation(Path(spec["executable"])),
                    "journal": self.journal(spec["name"]),
                }
            )
        sockets = []
        for spec in self.installation["ipc"]:
            path = Path(spec["path"])
            _safe_parents(path, owners={0, self.installation["identities"]["verifier"]["uid"]})
            info = path.lstat()
            sockets.append(
                {
                    "path": str(path),
                    "kind": "socket" if stat.S_ISSOCK(info.st_mode) else "other",
                    "uid": info.st_uid,
                    "gid": info.st_gid,
                    "mode": stat.S_IMODE(info.st_mode),
                }
            )
        journal = Path("/var/log/journal")
        _safe_parents(journal)
        info = journal.lstat()
        return {
            "package": self.package(),
            "units": units,
            "configuration": [
                file_observation(Path(item["path"])) for item in self.installation["configuration"]
            ],
            "ipc": sockets,
            "ipc_qualification": blocked_ipc_qualification(),
            "postgres": self.postgres(),
            "state": file_observation(self.state_path),
            "persistent_journal": {
                "kind": "directory" if stat.S_ISDIR(info.st_mode) else "other",
                "uid": info.st_uid,
            },
        }

    def baseline(self) -> dict[str, Any]:
        paths = []
        for raw in self.installation["clean_paths"]:
            path = Path(raw)
            try:
                path.lstat()
                kind = "present"
            except FileNotFoundError:
                kind = "absent"
            paths.append({"path": raw, "kind": kind})
        units = []
        for unit in self.installation["units"]:
            try:
                load_state = self.properties(unit["name"])["LoadState"]
            except ObservationError:
                # Nieistniejąca jednostka nie musi mieć pełnego zestawu pól show.
                # Akceptujemy wyłącznie jedno dokładne pole not-found, nigdy
                # sam niezerowy kod polecenia ani dowolny komunikat stderr.
                result = command(
                    ["/usr/bin/systemctl", "--system", "show", unit["name"], "--property=LoadState"]
                )
                require(
                    result.returncode in {0, 1, 4}
                    and result.stdout.strip() == "LoadState=not-found",
                    "Nie potwierdzono braku jednostki",
                )
                load_state = "not-found"
            units.append({"name": unit["name"], "load_state": load_state})
        return {"package": self.package(), "paths": paths, "units": units}

    def install(self, kind: str) -> int:
        item = self.installation["packages"][kind]
        raw = read_regular(Path(item["path"]), protected=True)
        require(sha256(raw) == item["sha256"], "Podmieniony instalator")
        # Zweryfikowane bajty przekazujemy przez chroniony, wyłączny plik; dpkg nie
        # otwiera ponownie oryginalnej ścieżki dostarczonej przez operatora.
        directory = Path("/var/lib/cryptohunter-stage9-evidence")
        directory.mkdir(mode=0o700, exist_ok=True)
        _safe_parents(directory / "placeholder")
        require(stat.S_IMODE(directory.stat().st_mode) == 0o700, "Niechroniony katalog instalatora")
        target = directory / (self.challenge_id + "-" + kind + ".deb")
        descriptor = os.open(target, os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW, 0o600)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(raw)
                stream.flush()
                os.fsync(stream.fileno())
            result = command(["/usr/bin/dpkg", "--install", str(target)], timeout=300)
        finally:
            # Usuwany jest wyłącznie własny plik O_EXCL. Nie usuwamy zasobów hosta.
            target.unlink(missing_ok=True)
        return result.returncode

    def create_state(self) -> None:
        if sys.platform != "linux":
            raise ObservationError("Tworzenie stanu dowodowego wymaga natywnego Linux")
        _safe_parents(self.state_path)
        runtime = self.installation["identities"]["runtime"]
        descriptor = os.open(
            self.state_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW, 0o600
        )
        try:
            os.fchown(descriptor, runtime["uid"], runtime["gid"])
            os.write(descriptor, secrets.token_bytes(32))
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        parent_fd = os.open(self.state_path.parent, os.O_DIRECTORY | os.O_RDONLY)
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)

    def _wait_snapshot(self) -> dict[str, Any]:
        last_error: Exception | None = None
        for _ in range(30):
            try:
                snapshot = self.snapshot()
                if all(
                    u["properties"]["ActiveState"] == "active" and u["processes"]
                    for u in snapshot["units"]
                ):
                    return snapshot
            except (OSError, ValueError, KeyError) as exc:
                last_error = exc
            time.sleep(1)
        raise ObservationError("Usługi nie odzyskały gotowości w limicie") from last_error

    def lifecycle(self) -> dict[str, Any]:
        before = self.snapshot()
        _snapshot(before, self.installation, updated=False)
        commands = []
        snapshots: dict[str, Any] = {}
        for action in ("stop", "start", "restart", "crash"):
            for spec in self.installation["units"]:
                args = ["/usr/bin/systemctl", "--system"]
                args += (
                    ["kill", "--kill-whom=all", "--signal=SIGKILL", spec["name"]]
                    if action == "crash"
                    else [action, spec["name"]]
                )
                result = command(args)
                commands.append(
                    {"action": action, "unit": spec["name"], "exit_code": result.returncode}
                )
                require(result.returncode == 0, "Nieudane działanie lifecycle")
            if action == "stop":
                stopped = []
                for spec, previous in zip(self.installation["units"], before["units"], strict=True):
                    props = self.properties(spec["name"])
                    cgroup = Path("/sys/fs/cgroup/system.slice") / spec["name"] / "cgroup.procs"
                    try:
                        cgroup_pids = [int(v) for v in read_regular(cgroup, maximum=65536).split()]
                    except FileNotFoundError:
                        cgroup_pids = []
                    alive = []
                    for proc in previous["processes"]:
                        try:
                            now = process_observation(proc["pid"])
                        except (OSError, ValueError, KeyError):
                            continue
                        if now["start_time_ticks"] == proc["start_time_ticks"]:
                            alive.append(proc["pid"])
                    stopped.append(
                        {
                            "name": spec["name"],
                            "active_state": props["ActiveState"],
                            "main_pid": int(props["MainPID"]),
                            "cgroup_pids": cgroup_pids,
                            "old_processes_alive": alive,
                        }
                    )
                snapshots["stopped"] = stopped
            else:
                snapshots[
                    {"start": "started", "restart": "restarted", "crash": "crashed"}[action]
                ] = self._wait_snapshot()
        return {"before": before, **snapshots, "commands": commands}

    def collect(self, phase: str, previous_records: list[dict[str, Any]]) -> dict[str, Any]:
        if phase == "baseline":
            return self.baseline()
        if phase == "clean_install":
            current_baseline = self.baseline()
            require(
                bool(previous_records)
                and current_baseline == previous_records[0]["payload"]["observations"],
                "Warunki czystej instalacji zmieniły się przed mutacją",
            )
            validate_observation_prefix(
                [{"phase": "baseline", "observations": current_baseline}], self.installation
            )
            code = self.install("baseline")
            require(code == 0, "Nieudana instalacja; zachowaj diagnostykę poza raportem")
            self.create_state()
            return {"install_exit_code": code, "snapshot": self._wait_snapshot()}
        if phase == "systemd_lifecycle":
            return self.lifecycle()
        if phase == "pre_reboot":
            # Sam restart wymaga osobnej zgody operatora i osobnego polecenia.
            return {"snapshot": self.snapshot()}
        if phase == "post_reboot":
            previous_boot = previous_records[3]["payload"]["host"]["boot_id"]
            require(
                host_identity()["boot_id"] != previous_boot, "Nie potwierdzono restartu systemu"
            )
            return {
                "snapshot": self.snapshot(),
                "previous_boot_journal": [
                    self.journal(u["name"], previous_boot) for u in self.installation["units"]
                ],
            }
        require(phase in {"update", "failed_update"}, "Nieznana faza")
        before = self.snapshot()
        _snapshot(before, self.installation, updated=phase == "failed_update")
        prior_observations = previous_records[-1]["payload"]["observations"]
        prior_snapshot = prior_observations.get("snapshot", prior_observations.get("after"))
        require(
            prior_snapshot is not None
            and before["state"] == prior_snapshot["state"]
            and before["postgres"]["administrative_observation"]["system_identifier"]
            == prior_snapshot["postgres"]["administrative_observation"]["system_identifier"],
            "Stan lub klaster zmienił się przed aktualizacją",
        )
        code = self.install(phase)
        return {"before": before, "install_exit_code": code, "after": self._wait_snapshot()}

"""Zamknięty kontrakt surowych obserwacji fizycznej kwalifikacji Linux.

Moduł nie nadaje zaufania podpisom ani hostom. Wywołujący musi wcześniej
zweryfikować pochodzenie, challenge, kolejność i wszystkie materiały.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import PurePosixPath
import re
from typing import Any, cast


class ObservationError(ValueError):
    """Bezpieczna odmowa; komunikaty nie zawierają danych hosta ani sekretów."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ObservationError(message)


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _fields(value: Any, names: set[str]) -> dict[str, Any]:
    require(type(value) is dict and set(value) == names, "Nieprawidłowe pola kontraktu")
    return cast(dict[str, Any], value)


def _hash(value: Any) -> bool:
    return type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _path(value: Any) -> bool:
    return (
        type(value) is str
        and re.fullmatch(r"/[A-Za-z0-9_./-]+", value) is not None
        and str(PurePosixPath(value)) == value
        and all(part not in {".", ".."} for part in value.split("/")[1:])
        and value != "/"
    )


def _integer(value: Any, minimum: int = 0, maximum: int = 2**31 - 1) -> bool:
    return type(value) is int and minimum <= value <= maximum


def validate_installation(value: Any) -> dict[str, Any]:
    """Sprawdź strukturę instalacji; sama integralność nie identyfikuje produktu.

    Brak release_identity dopuszcza tylko diagnostykę. Pełny weryfikator
    osobno wymaga niezależnego podpisu wydania i rzeczywistych testów usług.
    """
    data = _fields(
        value,
        {
            "schema_version",
            "purpose",
            "source_root",
            "package_name",
            "packages",
            "identities",
            "units",
            "configuration",
            "ipc",
            "state_directory",
            "clean_paths",
            "postgres",
        }
        | (
            {"release_identity"}
            if isinstance(value, dict) and "release_identity" in value
            else set()
        )
        | (
            {"functional_qualification"}
            if isinstance(value, dict) and "functional_qualification" in value
            else set()
        ),
    )
    require(
        type(data["schema_version"]) is int and data["schema_version"] == 1,
        "Nieobsługiwana wersja instalacji",
    )
    require(
        data["purpose"] == "STAGE9_DISPOSABLE_PHYSICAL_HOST",
        "Wymagany wydzielony host kwalifikacyjny",
    )
    require(_path(data["source_root"]), "Nieprawidłowa ścieżka źródeł")
    require(
        type(data["package_name"]) is str
        and re.fullmatch(r"[a-z0-9][a-z0-9+.-]{1,100}", data["package_name"]) is not None,
        "Nieprawidłowa nazwa pakietu",
    )
    packages = _fields(data["packages"], {"baseline", "update", "failed_update"})
    for item in packages.values():
        _fields(item, {"path", "sha256", "version"})
        require(
            _path(item["path"])
            and item["path"].endswith(".deb")
            and _hash(item["sha256"])
            and type(item["version"]) is str
            and re.fullmatch(r"[0-9][A-Za-z0-9.+:~-]{0,100}", item["version"]) is not None,
            "Nieprawidłowy materiał pakietu",
        )
    require(len({p["sha256"] for p in packages.values()}) == 3, "Wymagane trzy różne artefakty")
    require(
        len({p["version"] for p in packages.values()}) == 3, "Wymagane trzy różne wersje artefaktów"
    )
    identities = _fields(data["identities"], {"runtime", "verifier", "administrator"})
    for role, identity in identities.items():
        _fields(identity, {"principal", "uid", "gid"})
        require(
            type(identity["principal"]) is str
            and re.fullmatch(r"[a-z_][a-z0-9_-]{0,63}", identity["principal"]) is not None
            and _integer(identity["uid"])
            and _integer(identity["gid"]),
            "Nieprawidłowa tożsamość usługi",
        )
        if role != "administrator":
            require(
                identity["uid"] > 0 and identity["gid"] > 0,
                "Usługa nie może działać jako administrator",
            )
    for key in ("principal", "uid", "gid"):
        require(
            len({item[key] for item in identities.values()}) == 3, "Brak rozdzielenia tożsamości"
        )
    units = data["units"]
    require(type(units) is list and 2 <= len(units) <= 16, "Brak kompletu jednostek")
    require(
        {unit.get("role") for unit in units if type(unit) is dict} == {"runtime", "verifier"},
        "Brak usług runtime i verifier",
    )
    names: list[str] = []
    for unit in units:
        _fields(
            unit,
            {
                "name",
                "role",
                "executable",
                "executable_sha256",
                "update_executable_sha256",
                "unit_file",
            },
        )
        require(
            type(unit["name"]) is str
            and re.fullmatch(r"[a-z0-9][a-z0-9_.-]{0,100}\.service", unit["name"]) is not None,
            "Nieprawidłowa nazwa jednostki",
        )
        require(
            _path(unit["executable"])
            and _hash(unit["executable_sha256"])
            and _hash(unit["update_executable_sha256"]),
            "Nieprawidłowy program usługi",
        )
        _fields(unit["unit_file"], {"path", "sha256"})
        require(
            unit["unit_file"]["path"] == "/etc/systemd/system/" + unit["name"]
            and _hash(unit["unit_file"]["sha256"]),
            "Nieprawidłowy fragment systemd",
        )
        names.append(unit["name"])
    require(len(names) == len(set(names)), "Powtórzona jednostka")
    require(
        any(u["executable_sha256"] != u["update_executable_sha256"] for u in units),
        "Aktualizacja musi zmieniać program instalacji",
    )
    require(
        type(data["configuration"]) is list and 1 <= len(data["configuration"]) <= 64,
        "Brak przypiętej konfiguracji",
    )
    for item in data["configuration"]:
        _fields(item, {"path", "sha256", "uid", "gid", "mode"})
        require(
            _path(item["path"])
            and _hash(item["sha256"])
            and _integer(item["uid"], 0, 0)
            and _integer(item["gid"], 0, 0)
            and _integer(item["mode"], 0, 0o777)
            and not item["mode"] & 0o022,
            "Niechroniona konfiguracja",
        )
    require(type(data["ipc"]) is list and 1 <= len(data["ipc"]) <= 32, "Brak IPC")
    for item in data["ipc"]:
        _fields(item, {"path", "uid", "gid", "mode"})
        require(
            _path(item["path"])
            and item["uid"] == identities["verifier"]["uid"]
            and item["gid"] == identities["verifier"]["gid"]
            and _integer(item["mode"], 0, 0o777)
            and not item["mode"] & 0o007,
            "Niechronione IPC",
        )
    require(
        _path(data["state_directory"])
        and data["state_directory"].startswith("/var/lib/cryptohunter-stage9-qualification/"),
        "Stan testowy musi znajdować się w wydzielonej przestrzeni",
    )
    paths = data["clean_paths"]
    require(
        type(paths) is list
        and 1 <= len(paths) <= 64
        and all(_path(p) for p in paths)
        and len(paths) == len(set(paths))
        and data["state_directory"] in paths,
        "Nieprawidłowa lista ścieżek czystej instalacji",
    )
    require(
        all(u["unit_file"]["path"] in paths for u in units),
        "Brak jednostek w dowodzie czystej instalacji",
    )
    pg = _fields(
        data["postgres"],
        {
            "socket",
            "port",
            "database",
            "user",
            "expected_system_user",
            "expected_hba_sha256",
            "expected_ident_sha256",
        },
    )
    require(
        _path(pg["socket"])
        and _integer(pg["port"], 1, 65535)
        and all(
            type(pg[k]) is str and re.fullmatch(r"[a-z_][a-z0-9_]{0,62}", pg[k])
            for k in ("database", "user")
        )
        and pg["expected_system_user"] == "peer:" + identities["administrator"]["principal"]
        and _hash(pg["expected_hba_sha256"])
        and _hash(pg["expected_ident_sha256"]),
        "Nieprawidłowa izolacja PostgreSQL",
    )
    if "functional_qualification" in data:
        qualification = _fields(data["functional_qualification"], {"installation_id"})
        require(_hash(qualification["installation_id"]), "Nieprawidłowa tożsamość instalacji")
        policies = [
            item
            for item in data["configuration"]
            if item["path"] == "/etc/cryptohunter/linux-qualification.json"
        ]
        require(
            len(policies) == 1
            and policies[0]["mode"] == 0o644
            and len([unit for unit in units if unit["role"] == "verifier"]) == 1
            and any(
                item["path"] == "/run/cryptohunter/freshness/verifier.sock" for item in data["ipc"]
            ),
            "Brak przypiętej polityki odczytowej rzeczywistego verifiera",
        )
    if "release_identity" in data:
        from deployment.linux_release_identity import validate_release_statement

        validate_release_statement(data["release_identity"], data)
    return data


UNIT_PROPERTIES = (
    "Id",
    "LoadState",
    "ActiveState",
    "SubState",
    "UnitFileState",
    "MainPID",
    "User",
    "Group",
    "FragmentPath",
    "ControlGroup",
    "Restart",
    "KillMode",
    "NoNewPrivileges",
    "ProtectSystem",
    "ProtectHome",
    "PrivateTmp",
    "CapabilityBoundingSet",
    "AmbientCapabilities",
    "RestrictAddressFamilies",
    "NRestarts",
)


def _file(observed: Any, path: str, digest: str, *, executable: bool = False) -> None:
    _fields(observed, {"path", "kind", "sha256", "uid", "gid", "mode"})
    require(
        observed["path"] == path
        and observed["kind"] == "regular"
        and observed["sha256"] == digest
        and _integer(observed["uid"], 0, 0)
        and _integer(observed["gid"], 0, 0)
        and _integer(observed["mode"], 0, 0o777)
        and not observed["mode"] & 0o022
        and (not executable or bool(observed["mode"] & 0o100)),
        "Podmiana pliku instalacji",
    )


def _snapshot(
    snapshot: Any,
    installation: dict[str, Any],
    *,
    updated: bool,
    challenge_id: str | None = None,
) -> None:
    _fields(
        snapshot,
        {
            "package",
            "units",
            "configuration",
            "ipc",
            "ipc_qualification",
            "postgres",
            "state",
            "persistent_journal",
        }
        | ({"functional_observations"} if "functional_qualification" in installation else set()),
    )
    from deployment.linux_physical_ipc import validate_blocked_ipc_qualification
    from deployment.linux_physical_postgresql import validate_blocked_service_qualification

    validate_blocked_ipc_qualification(snapshot["ipc_qualification"])
    package = _fields(snapshot["package"], {"name", "version", "status"})
    expected = installation["packages"]["update" if updated else "baseline"]
    require(
        package
        == {
            "name": installation["package_name"],
            "version": expected["version"],
            "status": "install ok installed",
        },
        "Nieprawidłowy stan pakietu",
    )
    require(
        type(snapshot["units"]) is list and len(snapshot["units"]) == len(installation["units"]),
        "Brak obserwacji jednostki",
    )
    pids: set[int] = set()
    for actual, spec in zip(snapshot["units"], installation["units"], strict=True):
        _fields(actual, {"name", "properties", "processes", "unit_file", "executable", "journal"})
        require(actual["name"] == spec["name"], "Podmiana jednostki")
        props = _fields(actual["properties"], set(UNIT_PROPERTIES))
        require(all(type(v) is str for v in props.values()), "Nieprawidłowy wynik systemd")
        identity = installation["identities"][spec["role"]]
        exact = {
            "Id": spec["name"],
            "LoadState": "loaded",
            "ActiveState": "active",
            "SubState": "running",
            "UnitFileState": "enabled",
            "User": identity["principal"],
            "Group": identity["principal"],
            "FragmentPath": spec["unit_file"]["path"],
            "KillMode": "control-group",
            "NoNewPrivileges": "yes",
            "ProtectSystem": "strict",
            "CapabilityBoundingSet": "",
            "AmbientCapabilities": "",
        }
        require(
            all(props[k] == v for k, v in exact.items())
            and props["Restart"] in {"on-failure", "always"}
            and props["ProtectHome"] in {"yes", "read-only", "tmpfs"}
            and props["PrivateTmp"] in {"yes", "disconnected"}
            and props["RestrictAddressFamilies"] == "AF_UNIX"
            and props["ControlGroup"] == "/system.slice/" + spec["name"],
            "Niezgodne zabezpieczenia jednostki",
        )
        require(
            re.fullmatch(r"[1-9][0-9]*", props["MainPID"]) is not None
            and re.fullmatch(r"[0-9]+", props["NRestarts"]) is not None,
            "Nieprawidłowa tożsamość procesu",
        )
        digest = spec["update_executable_sha256" if updated else "executable_sha256"]
        _file(actual["unit_file"], spec["unit_file"]["path"], spec["unit_file"]["sha256"])
        _file(actual["executable"], spec["executable"], digest, executable=True)
        require(
            type(actual["processes"]) is list and 1 <= len(actual["processes"]) <= 128,
            "Brak obserwacji procesów cgroup",
        )
        unit_pids = []
        for proc in actual["processes"]:
            _fields(
                proc,
                {
                    "pid",
                    "start_time_ticks",
                    "uid",
                    "gid",
                    "groups",
                    "no_new_privs",
                    "capabilities",
                    "exe_sha256",
                },
            )
            require(
                _integer(proc["pid"], 1)
                and _integer(proc["start_time_ticks"], 1, 2**64 - 1)
                and proc["uid"] == identity["uid"]
                and proc["gid"] == identity["gid"]
                and type(proc["groups"]) is list
                and all(type(gid) is int and gid == identity["gid"] for gid in proc["groups"])
                and type(proc["no_new_privs"]) is int
                and proc["no_new_privs"] == 1
                and proc["exe_sha256"] == digest
                and proc["capabilities"]
                == {
                    key: "0000000000000000"
                    for key in ("CapInh", "CapPrm", "CapEff", "CapBnd", "CapAmb")
                },
                "Brak izolacji procesu",
            )
            require(proc["pid"] not in pids, "Proces należy do dwóch jednostek")
            pids.add(proc["pid"])
            unit_pids.append(proc["pid"])
        require(int(props["MainPID"]) in unit_pids, "MainPID poza cgroup")
        journal = _fields(actual["journal"], {"boot_id", "unit", "realtime_usec"})
        require(
            journal["unit"] == spec["name"]
            and type(journal["boot_id"]) is str
            and re.fullmatch(r"[0-9a-f]{32}", journal["boot_id"]) is not None
            and _integer(journal["realtime_usec"], 1, 2**64 - 1),
            "Brak dziennika usługi",
        )
    require(
        type(snapshot["configuration"]) is list
        and len(snapshot["configuration"]) == len(installation["configuration"]),
        "Brak obserwacji konfiguracji",
    )
    for actual, expected_file in zip(
        snapshot["configuration"], installation["configuration"], strict=True
    ):
        _file(actual, expected_file["path"], expected_file["sha256"])
        require(actual["mode"] == expected_file["mode"], "Niezgodne prawa konfiguracji")
    require(
        type(snapshot["ipc"]) is list and len(snapshot["ipc"]) == len(installation["ipc"]),
        "Brak obserwacji IPC",
    )
    for actual, expected_socket in zip(snapshot["ipc"], installation["ipc"], strict=True):
        require(actual == {**expected_socket, "kind": "socket"}, "Niezgodne IPC")
    pg_evidence = _fields(
        snapshot["postgres"], {"administrative_observation", "service_qualification"}
    )
    validate_blocked_service_qualification(pg_evidence["service_qualification"])
    pg = _fields(
        pg_evidence["administrative_observation"],
        {
            "listen_addresses",
            "fsync",
            "synchronous_commit",
            "system_user",
            "current_user",
            "session_user",
            "client_addr",
            "server_addr",
            "ssl",
            "hba_sha256",
            "ident_sha256",
            "system_identifier",
        }
        | (
            {"hba_rules", "ident_mappings"} if "functional_qualification" in installation else set()
        ),
    )
    expected_pg = installation["postgres"]
    require(
        pg["listen_addresses"] == ""
        and pg["fsync"] == "on"
        and pg["synchronous_commit"] == "on"
        and pg["system_user"] == expected_pg["expected_system_user"]
        and pg["current_user"] == expected_pg["user"]
        and pg["session_user"] == expected_pg["user"]
        and pg["client_addr"] is None
        and pg["server_addr"] is None
        and pg["ssl"] is False
        and pg["hba_sha256"] == expected_pg["expected_hba_sha256"]
        and pg["ident_sha256"] == expected_pg["expected_ident_sha256"]
        and type(pg["system_identifier"]) is str
        and re.fullmatch(r"[1-9][0-9]{0,24}", pg["system_identifier"]) is not None,
        "Niepotwierdzona izolacja PostgreSQL",
    )
    if "functional_qualification" in installation:
        for name, width, digest_name in (
            ("hba_rules", 8, "hba_sha256"),
            ("ident_mappings", 4, "ident_sha256"),
        ):
            rows = pg[name]
            require(
                type(rows) is list
                and len(rows) <= 256
                and all(type(row) is list and len(row) == width and row[-1] is None for row in rows)
                and len(canonical(rows)) <= 65536
                and sha256(canonical(rows)) == pg[digest_name],
                "Surowe reguły PostgreSQL nie odpowiadają przypiętej konfiguracji",
            )
    state = _fields(snapshot["state"], {"path", "kind", "sha256", "uid", "gid", "mode"})
    runtime = installation["identities"]["runtime"]
    require(
        state["path"].startswith(installation["state_directory"] + "/.stage9-evidence-")
        and state["kind"] == "regular"
        and _hash(state["sha256"])
        and state["uid"] == runtime["uid"]
        and state["gid"] == runtime["gid"]
        and state["mode"] == 0o600,
        "Nieprawidłowy stan trwałości",
    )
    require(
        snapshot["persistent_journal"] == {"kind": "directory", "uid": 0}
        and type(snapshot["persistent_journal"]["uid"]) is int,
        "Brak trwałego dziennika systemowego",
    )
    if "functional_qualification" in installation:
        validate_functional_observations(
            snapshot["functional_observations"], installation, challenge_id=challenge_id
        )
        selected = next(unit for unit in installation["units"] if unit["role"] == "verifier")
        observed_unit = next(unit for unit in snapshot["units"] if unit["name"] == selected["name"])
        peer = snapshot["functional_observations"]["verifier"]["server_process"]
        require(
            str(peer["pid"]) == observed_unit["properties"]["MainPID"]
            and any(
                process["pid"] == peer["pid"]
                and process["start_time_ticks"] == peer["start_time_ticks"]
                and process["exe_sha256"] == peer["exe_sha256"]
                for process in observed_unit["processes"]
            ),
            "Obserwacja funkcjonalna dotyczy innego procesu usługi",
        )


def validate_functional_observations(
    value: Any, installation: dict[str, Any], *, challenge_id: str | None
) -> None:
    """Odtwórz częściową obserwację; nie przyznawaj uprawnień produkcyjnych."""
    from deployment.linux_physical_ipc import ExpectedPeerProcess, validate_verifier_observation

    _fields(value, {"schema_version", "status", "verifier"})
    require(
        type(value["schema_version"]) is int
        and value["schema_version"] == 1
        and value["status"] == "PARTIAL_BLOCKED"
        and "functional_qualification" in installation
        and _hash(challenge_id),
        "Brak niezależnego powiązania częściowych obserwacji z wyzwaniem",
    )
    require(type(value["verifier"]) is dict, "Brak surowej obserwacji verifiera")
    verifier = value["verifier"]
    process = verifier.get("server_process")
    require(type(process) is dict, "Brak surowej tożsamości serwera")
    specs = [unit for unit in installation["units"] if unit["role"] == "verifier"]
    require(len(specs) == 1, "Niejednoznaczna rzeczywista usługa verifiera")
    identity = installation["identities"]["verifier"]
    spec = specs[0]
    require(
        process.get("exe_sha256") in {spec["executable_sha256"], spec["update_executable_sha256"]},
        "Program serwera nie pochodzi z przypiętego wydania",
    )
    expected = ExpectedPeerProcess(
        pid=process["pid"],
        uid=identity["uid"],
        gid=identity["gid"],
        start_time_ticks=process["start_time_ticks"],
        executable_sha256=process["exe_sha256"],
    )
    validate_verifier_observation(
        verifier,
        expected=expected,
        expected_cgroup="/system.slice/" + spec["name"],
        challenge_id=cast(str, challenge_id),
        installation_id=installation["functional_qualification"]["installation_id"],
    )


def _tokens(snapshot: dict[str, Any]) -> set[tuple[int, int]]:
    return {(p["pid"], p["start_time_ticks"]) for u in snapshot["units"] for p in u["processes"]}


def validate_observations(records: list[dict[str, Any]], installation: dict[str, Any]) -> None:
    """Niezależnie sprawdź fakty wszystkich faz; żadna deklaracja PASS nie jest wejściem."""
    validate_installation(installation)
    require(type(records) is list and len(records) == 7, "Brak wymaganych faz")
    validate_observation_prefix(records, installation)
    phases = (
        "baseline",
        "clean_install",
        "systemd_lifecycle",
        "pre_reboot",
        "post_reboot",
        "update",
        "failed_update",
    )
    require(tuple(record.get("phase") for record in records) == phases, "Niezgodna kolejność faz")
    baseline = _fields(records[0]["observations"], {"package", "paths", "units"})
    require(
        baseline["package"] is None
        and baseline["paths"]
        == [{"path": path, "kind": "absent"} for path in installation["clean_paths"]]
        and baseline["units"]
        == [{"name": unit["name"], "load_state": "not-found"} for unit in installation["units"]],
        "Instalacja nie była czysta",
    )
    clean = _fields(records[1]["observations"], {"install_exit_code", "snapshot"})
    require(
        type(clean["install_exit_code"]) is int and clean["install_exit_code"] == 0,
        "Instalacja zakończyła się błędem",
    )
    lifecycle = _fields(
        records[2]["observations"],
        {"before", "stopped", "started", "restarted", "crashed", "commands"},
    )
    expected_commands = [
        {"action": action, "unit": u["name"], "exit_code": 0}
        for action in ("stop", "start", "restart", "crash")
        for u in installation["units"]
    ]
    require(lifecycle["commands"] == expected_commands, "Brak wykonania lifecycle")
    require(
        lifecycle["stopped"]
        == [
            {
                "name": u["name"],
                "active_state": "inactive",
                "main_pid": 0,
                "cgroup_pids": [],
                "old_processes_alive": [],
            }
            for u in installation["units"]
        ],
        "Procesy osierocone po zatrzymaniu",
    )
    pre = _fields(records[3]["observations"], {"snapshot"})
    post = _fields(records[4]["observations"], {"snapshot", "previous_boot_journal"})
    update = _fields(records[5]["observations"], {"before", "install_exit_code", "after"})
    failure = _fields(records[6]["observations"], {"before", "install_exit_code", "after"})
    require(
        type(update["install_exit_code"]) is int and update["install_exit_code"] == 0,
        "Aktualizacja zakończyła się błędem",
    )
    require(
        type(failure["install_exit_code"]) is int and 1 <= failure["install_exit_code"] <= 255,
        "Brak rzeczywistej odmowy nieudanej aktualizacji",
    )
    snapshots = [
        (clean["snapshot"], False),
        *[(lifecycle[k], False) for k in ("before", "started", "restarted", "crashed")],
        (pre["snapshot"], False),
        (post["snapshot"], False),
        (update["before"], False),
        (update["after"], True),
        (failure["before"], True),
        (failure["after"], True),
    ]
    for snapshot, updated in snapshots:
        _snapshot(
            snapshot, installation, updated=updated, challenge_id=records[0].get("challenge_id")
        )
    state = clean["snapshot"]["state"]
    require(
        state["path"]
        == installation["state_directory"] + "/.stage9-evidence-" + records[0]["challenge_id"],
        "Stan trwałości pochodzi z innego wyzwania",
    )
    cluster = clean["snapshot"]["postgres"]["administrative_observation"]["system_identifier"]
    for snapshot, _ in snapshots:
        require(
            snapshot["state"] == state
            and snapshot["postgres"]["administrative_observation"]["system_identifier"] == cluster,
            "Utracona ciągłość stanu lub klastra",
        )
    for record in records[1:]:
        observations = record["observations"]
        current_snapshots = [
            observations[k]
            for k in ("snapshot", "before", "after", "started", "restarted", "crashed")
            if k in observations
        ]
        for snapshot in current_snapshots:
            require(
                all(
                    u["journal"]["boot_id"] == record["host"]["boot_id"].replace("-", "")
                    for u in snapshot["units"]
                ),
                "Dziennik pochodzi z innego rozruchu",
            )
    require(
        not (_tokens(update["before"]) & _tokens(update["after"])),
        "Aktualizacja nie uruchomiła nowego programu",
    )
    require(
        not (_tokens(lifecycle["before"]) & _tokens(lifecycle["started"]))
        and not (_tokens(lifecycle["started"]) & _tokens(lifecycle["restarted"]))
        and not (_tokens(lifecycle["restarted"]) & _tokens(lifecycle["crashed"])),
        "Brak wymiany procesów w lifecycle",
    )
    for old, new in zip(
        lifecycle["restarted"]["units"], lifecycle["crashed"]["units"], strict=True
    ):
        require(
            int(new["properties"]["NRestarts"]) > int(old["properties"]["NRestarts"]),
            "Brak automatycznego restartu po awarii",
        )
    boot_before = records[3]["host"]["boot_id"]
    boot_after = records[4]["host"]["boot_id"]
    require(
        all(r["host"]["boot_id"] == boot_before for r in records[:4])
        and boot_before != boot_after
        and all(r["host"]["boot_id"] == boot_after for r in records[4:]),
        "Brak potwierdzonego restartu systemu",
    )
    require(
        type(post["previous_boot_journal"]) is list
        and len(post["previous_boot_journal"]) == len(installation["units"]),
        "Brak trwałych logów poprzedniego rozruchu",
    )
    for row, spec in zip(post["previous_boot_journal"], installation["units"], strict=True):
        require(
            type(row) is dict
            and row.get("unit") == spec["name"]
            and row.get("boot_id") == boot_before.replace("-", "")
            and set(row) == {"unit", "boot_id", "realtime_usec"}
            and _integer(row["realtime_usec"], 1, 2**64 - 1),
            "Podmiana logów rozruchu",
        )


def validate_observation_prefix(
    records: list[dict[str, Any]], installation: dict[str, Any]
) -> None:
    """Odmów kontynuacji po błędzie dowolnej wcześniejszej fazy."""
    validate_installation(installation)
    require(type(records) is list and len(records) <= 7, "Nieprawidłowy prefiks obserwacji")
    phases = (
        "baseline",
        "clean_install",
        "systemd_lifecycle",
        "pre_reboot",
        "post_reboot",
        "update",
        "failed_update",
    )
    for index, record in enumerate(records):
        require(record.get("phase") == phases[index], "Niezgodny prefiks faz")
        observations = record["observations"]
        if index == 0:
            _fields(observations, {"package", "paths", "units"})
            require(
                observations
                == {
                    "package": None,
                    "paths": [{"path": p, "kind": "absent"} for p in installation["clean_paths"]],
                    "units": [
                        {"name": u["name"], "load_state": "not-found"}
                        for u in installation["units"]
                    ],
                },
                "Host nie spełnia warunków czystej instalacji",
            )
            continue
        if index in {1, 3, 4}:
            expected = {"snapshot"} | (
                {"install_exit_code"}
                if index == 1
                else {"previous_boot_journal"}
                if index == 4
                else set()
            )
            _fields(observations, expected)
            if index == 1:
                require(
                    type(observations["install_exit_code"]) is int
                    and observations["install_exit_code"] == 0,
                    "Nieudana instalacja",
                )
            _snapshot(
                observations["snapshot"],
                installation,
                updated=False,
                challenge_id=record.get("challenge_id"),
            )
            if index == 4:
                previous_boot = records[3]["host"]["boot_id"].replace("-", "")
                rows = observations["previous_boot_journal"]
                require(
                    type(rows) is list and len(rows) == len(installation["units"]),
                    "Brak trwałych logów poprzedniego rozruchu",
                )
                for row, unit in zip(rows, installation["units"], strict=True):
                    _fields(row, {"unit", "boot_id", "realtime_usec"})
                    require(
                        row["unit"] == unit["name"]
                        and row["boot_id"] == previous_boot
                        and _integer(row["realtime_usec"], 1, 2**64 - 1),
                        "Niezgodne logi poprzedniego rozruchu",
                    )
        elif index == 2:
            _fields(
                observations, {"before", "stopped", "started", "restarted", "crashed", "commands"}
            )
            for key in ("before", "started", "restarted", "crashed"):
                _snapshot(
                    observations[key],
                    installation,
                    updated=False,
                    challenge_id=record.get("challenge_id"),
                )
            require(
                observations["commands"]
                == [
                    {"action": action, "unit": u["name"], "exit_code": 0}
                    for action in ("stop", "start", "restart", "crash")
                    for u in installation["units"]
                ],
                "Niepełne wykonanie lifecycle",
            )
            require(
                all(type(command["exit_code"]) is int for command in observations["commands"]),
                "Nieprawidłowy typ kodu zakończenia",
            )
            require(
                observations["stopped"]
                == [
                    {
                        "name": u["name"],
                        "active_state": "inactive",
                        "main_pid": 0,
                        "cgroup_pids": [],
                        "old_processes_alive": [],
                    }
                    for u in installation["units"]
                ],
                "Nie usunięto procesów jednostki",
            )
            require(
                all(type(unit["main_pid"]) is int for unit in observations["stopped"]),
                "Nieprawidłowy typ PID zatrzymanej jednostki",
            )
            for old_key, new_key in (
                ("before", "started"),
                ("started", "restarted"),
                ("restarted", "crashed"),
            ):
                require(
                    not (_tokens(observations[old_key]) & _tokens(observations[new_key])),
                    "Brak rzeczywistej zmiany procesów",
                )
            require(
                all(
                    int(new["properties"]["NRestarts"]) > int(old["properties"]["NRestarts"])
                    for old, new in zip(
                        observations["restarted"]["units"],
                        observations["crashed"]["units"],
                        strict=True,
                    )
                ),
                "Brak restartu po awarii",
            )
        else:
            _fields(observations, {"before", "install_exit_code", "after"})
            code = observations["install_exit_code"]
            require(
                type(code) is int and (code == 0 if index == 5 else 1 <= code <= 255),
                "Nieprawidłowy wynik aktualizacji",
            )
            _snapshot(
                observations["before"],
                installation,
                updated=index == 6,
                challenge_id=record.get("challenge_id"),
            )
            _snapshot(
                observations["after"],
                installation,
                updated=True,
                challenge_id=record.get("challenge_id"),
            )
            if index == 5:
                require(
                    not (_tokens(observations["before"]) & _tokens(observations["after"])),
                    "Aktualizacja nie uruchomiła nowego programu",
                )
        for key in ("snapshot", "before", "after", "started", "restarted", "crashed"):
            if key not in observations:
                continue
            snapshot = observations[key]
            require(
                snapshot["state"] == records[1]["observations"]["snapshot"]["state"]
                and snapshot["state"]["path"]
                == installation["state_directory"]
                + "/.stage9-evidence-"
                + records[0]["challenge_id"],
                "Utracona ciągłość stanu",
            )
            require(
                snapshot["postgres"]["administrative_observation"]["system_identifier"]
                == records[1]["observations"]["snapshot"]["postgres"]["administrative_observation"][
                    "system_identifier"
                ],
                "Utracona ciągłość klastra PostgreSQL",
            )
            require(
                all(
                    u["journal"]["boot_id"] == record["host"]["boot_id"].replace("-", "")
                    for u in snapshot["units"]
                ),
                "Dziennik pochodzi z innego rozruchu",
            )

"""Wyłącznie odczytowe obserwacje hosta dla instalatora Freshness.

Import modułu i testy kontraktu nie wymagają kont POSIX ani systemd.
"""

from __future__ import annotations

import json
import os
import re
import stat
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REQUIRED_PACKAGES = frozenset(
    {
        "python3",
        "python3-venv",
        "postgresql-16",
        "postgresql-client-16",
        "postgresql-common",
        "systemd",
    }
)
USERS = ("os_freshness_crypto_verifier", "os_freshness_runtime")
IPC_GROUP = "freshness_verifier_ipc"
UNIT_QUERY = [
    "/usr/bin/systemctl",
    "show",
    "cryptohunter-freshness-verifier.service",
    "--property=LoadState,FragmentPath,DropInPaths",
]
POSTGRESQL_UNIT_QUERY = [
    "/usr/bin/systemctl",
    "show",
    "postgresql@16-main.service",
    "--property=LoadState,ActiveState",
]
# Dwa zamknięte profile: przygotowany Ubuntu i dostarczony profil aplikacji.
# Kolejność reguł jest częścią polityki; żaden dodatkowy wiersz nie jest pomijany.
_PG_UBUNTU_HBA = [
    ["local", ["all"], ["postgres"], None, None, "peer", None],
    ["local", ["all"], ["all"], None, None, "peer", None],
    ["host", ["all"], ["all"], "127.0.0.1", "255.255.255.255", "scram-sha-256", None],
    [
        "host",
        ["all"],
        ["all"],
        "::1",
        "ffff:ffff:ffff:ffff:ffff:ffff:ffff:ffff",
        "scram-sha-256",
        None,
    ],
    ["local", ["replication"], ["all"], None, None, "peer", None],
    ["host", ["replication"], ["all"], "127.0.0.1", "255.255.255.255", "scram-sha-256", None],
    [
        "host",
        ["replication"],
        ["all"],
        "::1",
        "ffff:ffff:ffff:ffff:ffff:ffff:ffff:ffff",
        "scram-sha-256",
        None,
    ],
]
_PG_DEPLOYED_HBA = [
    ["local", ["all"], ["postgres"], None, None, "peer", ["map=freshness_admin_map"]],
    [
        "local",
        ["freshness_gate"],
        ["freshness_crypto_verifier"],
        None,
        None,
        "peer",
        ["map=freshness_authority_map"],
    ],
    [
        "local",
        ["freshness_gate"],
        ["freshness_runtime"],
        None,
        None,
        "peer",
        ["map=freshness_authority_map"],
    ],
    ["local", ["freshness_gate"], ["all"], None, None, "reject", None],
    ["local", ["all"], ["all"], None, None, "reject", None],
]
_PG_DEPLOYED_IDENT = [
    ["freshness_authority_map", "os_freshness_crypto_verifier", "freshness_crypto_verifier"],
    ["freshness_authority_map", "os_freshness_runtime", "freshness_runtime"],
    ["freshness_admin_map", "root", "postgres"],
]
_PG_HBA_ROWS = (
    "COALESCE((SELECT pg_catalog.jsonb_agg(pg_catalog.jsonb_build_array(type,database,user_name,address,netmask,"
    "auth_method,options) ORDER BY rule_number) FROM hba), '[]'::jsonb)"
)
_PG_IDENT_ROWS = (
    "COALESCE((SELECT pg_catalog.jsonb_agg(pg_catalog.jsonb_build_array(map_name,sys_name,pg_username) "
    "ORDER BY map_number) FROM ident), '[]'::jsonb)"
)
_POSTGRESQL_SQL = (
    "WITH hba AS MATERIALIZED (SELECT * FROM pg_catalog.pg_hba_file_rules), "
    "ident AS MATERIALIZED (SELECT * FROM pg_catalog.pg_ident_file_mappings), "
    "files AS MATERIALIZED (SELECT * FROM pg_catalog.pg_file_settings) "
    "SELECT pg_catalog.json_build_object('ready', ("
    "pg_catalog.current_setting('config_file') = '/etc/postgresql/16/main/postgresql.conf' "
    "AND pg_catalog.current_setting('data_directory') = '/var/lib/postgresql/16/main' "
    "AND pg_catalog.current_setting('port') = '5432' "
    "AND pg_catalog.current_setting('server_version_num')::int BETWEEN 160000 AND 169999 "
    "AND system_user::text IN ('peer:root', 'peer:postgres') "
    "AND session_user = 'postgres' AND current_user = 'postgres' "
    "AND pg_catalog.current_setting('transaction_read_only') = 'on' "
    "AND pg_catalog.current_setting('fsync') = 'on' "
    "AND pg_catalog.current_setting('synchronous_commit') = 'on' "
    "AND NOT EXISTS (SELECT FROM files WHERE error IS NOT NULL) "
    "AND NOT EXISTS (SELECT FROM files "
    "WHERE sourcefile IS NULL OR (sourcefile <> '/etc/postgresql/16/main/postgresql.conf' "
    "AND sourcefile !~ '^/etc/postgresql/16/main/conf[.]d/[^/.][^/]*[.]conf$')) "
    "AND NOT EXISTS (SELECT FROM hba WHERE error IS NOT NULL "
    "OR file_name IS DISTINCT FROM pg_catalog.current_setting('hba_file')) "
    "AND NOT EXISTS (SELECT FROM ident WHERE error IS NOT NULL "
    "OR file_name IS DISTINCT FROM pg_catalog.current_setting('ident_file')) "
    "AND NOT EXISTS (SELECT FROM pg_catalog.pg_settings WHERE pending_restart) "
    "AND NOT EXISTS (SELECT FROM files AS f JOIN pg_catalog.pg_settings AS s ON s.name=f.name "
    "WHERE f.applied AND f.name IN ('listen_addresses','unix_socket_directories','port',"
    "'data_directory','hba_file','ident_file','fsync','synchronous_commit') "
    "AND f.setting IS DISTINCT FROM s.setting) "
    "AND ((pg_catalog.current_setting('hba_file') = '/etc/postgresql/16/main/pg_hba.conf' "
    "AND pg_catalog.current_setting('ident_file') = '/etc/postgresql/16/main/pg_ident.conf' "
    "AND pg_catalog.current_setting('listen_addresses') = 'localhost' "
    "AND pg_catalog.current_setting('unix_socket_directories') = '/var/run/postgresql' "
    f"AND {_PG_HBA_ROWS} = '{json.dumps(_PG_UBUNTU_HBA)}'::jsonb "
    f"AND {_PG_IDENT_ROWS} = '[]'::jsonb) "
    "OR (pg_catalog.current_setting('hba_file') = '/etc/cryptohunter/postgresql/pg_hba.conf' "
    "AND pg_catalog.current_setting('ident_file') = '/etc/cryptohunter/postgresql/pg_ident.conf' "
    "AND pg_catalog.current_setting('listen_addresses') = '' "
    "AND pg_catalog.current_setting('unix_socket_directories') = '/run/postgresql' "
    f"AND {_PG_HBA_ROWS} = '{json.dumps(_PG_DEPLOYED_HBA)}'::jsonb "
    f"AND {_PG_IDENT_ROWS} = '{json.dumps(_PG_DEPLOYED_IDENT)}'::jsonb))), "
    "'postmaster_start_ns', "
    "(EXTRACT(EPOCH FROM pg_catalog.pg_postmaster_start_time()) * 1000000000)::bigint)"
)
POSTGRESQL_CONFIG_QUERY = [
    "/usr/sbin/runuser",
    "-u",
    "postgres",
    "--",
    "/usr/lib/postgresql/16/bin/psql",
    "-X",
    "-w",
    "-qAt",
    "-h",
    "/var/run/postgresql",
    "-p",
    "5432",
    "-U",
    "postgres",
    "-d",
    "dbname=postgres connect_timeout=5 passfile=/dev/null "
    "options='-c statement_timeout=5000 -c default_transaction_read_only=on -c search_path=pg_catalog'",
    "-v",
    "ON_ERROR_STOP=1",
    "-c",
    _POSTGRESQL_SQL,
]
# Zastany Ubuntu używa peer postgres -> postgres; wdrożony HBA mapuje
# root -> postgres. Obie stałe sondy pozostają lokalne i wyłącznie odczytowe.
POSTGRESQL_ROOT_CONFIG_QUERY = POSTGRESQL_CONFIG_QUERY[4:]
_PG_DIRECTORY = "etc/postgresql/16/main"
_PG_MAX_FILE = 1024 * 1024
_PG_MAX_TOTAL = 8 * _PG_MAX_FILE
_PG_LINE = re.compile(
    r"[ \t]*([A-Za-z_][A-Za-z_0-9.]*)"
    r"(?:[ \t]*=[ \t]*|[ \t]+)"
    r"('(?:[^'\\\r\n]|\\.|'')*'|[^ \t\r\n#'=]+)[ \t]*(?:#.*)?"
)


def _pg_directives(content: bytes) -> list[tuple[str, str]]:
    """Ograniczony profil jednowierszowy; pełną semantykę sprawdza serwer.

    Nie interpretujemy wielowierszowych wartości ani zagnieżdżonych include.
    Nieznana składnia oznacza odmowę, również gdy serwer nadal działa ze starą
    konfiguracją. include_dir nie jest GUC zwracanym przez pg_conftool.
    """
    text = content.decode("utf-8", errors="strict")
    if "\x00" in text:
        raise ValueError("NUL w konfiguracji PostgreSQL")
    result = []
    for line in text.split("\n"):
        line = line.removesuffix("\r")
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        match = _PG_LINE.fullmatch(line)
        if match is None:
            raise ValueError("Niejednoznaczna konfiguracja PostgreSQL")
        result.append((match[1].lower(), match[2]))
    return result


def _pg_auth_lines(content: bytes, *, ident: bool) -> None:
    """Zamknięcie zależności HBA/ident obejmuje również puste include i @file.

    Widoki PostgreSQL nie zwracają wierszy dla pustych plików dołączonych.
    Profil nie obsługuje include, list z plików ani kontynuacji wierszy.
    Pełną semantykę dopuszczonych wierszy sprawdza niezależnie serwer.
    """
    text = content.decode("utf-8", errors="strict")
    if "\x00" in text:
        raise ValueError("NUL w konfiguracji uwierzytelniania PostgreSQL")
    heads = {"freshness_admin_map", "freshness_authority_map"} if ident else {"local", "host"}
    for line in text.split("\n"):
        active = line.partition("#")[0].strip()
        if active and (active.split()[0] not in heads or "@" in active or "\\" in active):
            raise ValueError("Nieobsługiwane zależności konfiguracji uwierzytelniania PostgreSQL")


def _pg_snapshot(
    postgres_uid: int,
    *,
    root: Path = Path("/"),
    owner_uid: int = 0,
    postgres_gid: int | None = None,
) -> dict[str, tuple[tuple[int, ...], bytes]]:
    """Odczyt main/conf.d/HBA/ident/auto.conf bez zapisów.

    Katalogi pozostają root:0755 zgodnie z wcześniejszym kontraktem instalatora.
    postgres:0700 nie chroni przestrzeni nazw podpisanego drop-inu przed serwerem.
    Odczyt auto.conf ma odrębną politykę trzech stałych przodków danych;
    nie zmienia ona wymagań własności konfiguracji w /etc.
    root/owner_uid służą wyłącznie hermetycznemu modelowi filesystemu w testach.
    """
    result = {}

    def identity(metadata: os.stat_result) -> tuple[int, ...]:
        return tuple(
            getattr(metadata, name)
            for name in (
                "st_dev",
                "st_ino",
                "st_mode",
                "st_uid",
                "st_gid",
                "st_nlink",
                "st_size",
                "st_mtime_ns",
                "st_ctime_ns",
            )
        )

    def check(
        fd: int, *, directory: bool, installed_auth: bool = False, data_ancestor: bool = False
    ) -> os.stat_result:
        metadata = os.fstat(fd)
        if directory:
            valid = (
                stat.S_ISDIR(metadata.st_mode)
                and metadata.st_uid == owner_uid
                and stat.S_IMODE(metadata.st_mode) == (0o750 if installed_auth else 0o755)
            )
            if installed_auth:
                if sys.platform == "win32":
                    raise OSError("Inspekcja PostgreSQL wymaga POSIX")
                import grp

                expected_gid = (
                    grp.getgrnam("postgres").gr_gid if postgres_gid is None else postgres_gid
                )
                valid = valid and metadata.st_gid == expected_gid
            if data_ancestor:
                # Wyłącznie trzy stałe katalogi PGDATA podczas odczytu auto.conf.
                # Nie jest to wyjątek dla katalogów /etc ani miejsca publikacji.
                valid = (
                    stat.S_ISDIR(metadata.st_mode)
                    and metadata.st_uid in {owner_uid, postgres_uid}
                    and stat.S_IMODE(metadata.st_mode) in {0o700, 0o750, 0o755}
                )
        else:
            valid = (
                stat.S_ISREG(metadata.st_mode)
                and metadata.st_nlink == 1
                and metadata.st_uid in {owner_uid, postgres_uid}
                and metadata.st_size <= _PG_MAX_FILE
                and not metadata.st_mode & 0o022
            )
        if not valid:
            raise PermissionError("Niechroniona konfiguracja PostgreSQL")
        return metadata

    def read(parent: int, name: str, key: str) -> bytes:
        fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent)
        try:
            before = check(fd, directory=False)
            with os.fdopen(os.dup(fd), "rb") as stream:
                content = stream.read(_PG_MAX_FILE + 1)
            if len(content) > _PG_MAX_FILE or identity(before) != identity(os.fstat(fd)):
                raise ValueError("Zmiana konfiguracji PostgreSQL podczas odczytu")
            result[key] = (identity(before), content)
            if sum(len(data) for _metadata, data in result.values()) > _PG_MAX_TOTAL:
                raise ValueError("Zbyt duża konfiguracja PostgreSQL")
            return content
        finally:
            os.close(fd)

    root_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)

    def directory(relative: str) -> int:
        fd = os.dup(root_fd)
        try:
            check(fd, directory=True)
            parts = relative.split("/")
            for index, part in enumerate(parts):
                next_fd = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
                os.close(fd)
                fd = next_fd
                check(
                    fd,
                    directory=True,
                    installed_auth=relative == "etc/cryptohunter/postgresql"
                    and index == len(parts) - 1,
                    data_ancestor=relative == "var/lib/postgresql/16/main" and index >= 2,
                )
            return fd
        except BaseException:
            os.close(fd)
            raise

    try:
        fd = directory(_PG_DIRECTORY)
        try:
            main = _pg_directives(read(fd, "postgresql.conf", "postgresql.conf"))
            includes = [(key, value) for key, value in main if key.startswith("include")]
            if includes not in [[("include_dir", "'conf.d'")], [("include_dir", "conf.d")]]:
                raise ValueError("Wymagana pojedyncza dyrektywa include_dir conf.d")
            _pg_auth_lines(read(fd, "pg_hba.conf", "pg_hba.conf"), ident=False)
            _pg_auth_lines(read(fd, "pg_ident.conf", "pg_ident.conf"), ident=True)
        finally:
            os.close(fd)
        fd = directory(_PG_DIRECTORY + "/conf.d")
        try:
            result["conf.d"] = (identity(check(fd, directory=True)), b"")
            names = sorted(
                name
                for name in os.listdir(fd)
                if not name.startswith(".") and name.endswith(".conf")
            )
            if len(names) > 1024:
                raise ValueError("Zbyt wiele plików konfiguracji PostgreSQL")
            for name in names:
                directives = _pg_directives(read(fd, name, "conf.d/" + name))
                if any(key.startswith("include") for key, _value in directives):
                    raise ValueError("Nieobsługiwane zagnieżdżone include PostgreSQL")
        finally:
            os.close(fd)
        try:
            fd = directory("etc/cryptohunter/postgresql")
        except FileNotFoundError:
            pass
        else:
            try:
                _pg_auth_lines(read(fd, "pg_hba.conf", "installed/pg_hba.conf"), ident=False)
                _pg_auth_lines(read(fd, "pg_ident.conf", "installed/pg_ident.conf"), ident=True)
            finally:
                os.close(fd)
        fd = directory("var/lib/postgresql/16/main")
        try:
            auto = read(fd, "postgresql.auto.conf", "postgresql.auto.conf").decode(
                "utf-8", errors="strict"
            )
            if "\x00" in auto or any(
                line.strip() and not line.lstrip().startswith("#") for line in auto.split("\n")
            ):
                raise ValueError("Wymagany pusty postgresql.auto.conf bez dyrektyw")
        finally:
            os.close(fd)
        return result
    finally:
        os.close(root_fd)


def _pg_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Niejednoznaczna odpowiedź PostgreSQL")
        result[key] = value
    return result


def _postgresql_config_ready(postgres_uid: int) -> bool:
    try:
        before = _pg_snapshot(postgres_uid)
        for command in (POSTGRESQL_ROOT_CONFIG_QUERY, POSTGRESQL_CONFIG_QUERY):
            output = _query(command)
            if output is None:
                continue  # Niedostępny peer root może być zastąpiony peer postgres.
            observation = json.loads(output, object_pairs_hook=_pg_json_object)
            if (
                type(observation) is not dict
                or set(observation) != {"ready", "postmaster_start_ns"}
                or observation["ready"] is not True
                or type(observation["postmaster_start_ns"]) is not int
                or not 0 < observation["postmaster_start_ns"] <= time.time_ns()
            ):
                return False
            # Widoki HBA/ident opisują dysk, nie wszystkie załadowane reguły.
            # Wymagamy plików i namespace'u sprzed pełnego startu postmastera:
            # startup z błędnym HBA jest fatalny. Sam reload_time nie dowodzi
            # udanego reload HBA. Zmiany po starcie oznaczają odmowę, bez reload.
            if any(
                max(metadata[-2:]) > observation["postmaster_start_ns"]
                for metadata, _content in before.values()
            ):
                return False
            return before == _pg_snapshot(postgres_uid)
        return False
    except (KeyError, OSError, UnicodeError, ValueError):
        return False


@dataclass(frozen=True)
class User:
    name: str
    uid: int
    gid: int
    home: str
    shell: str


@dataclass(frozen=True)
class Group:
    name: str
    gid: int
    members: tuple[str, ...]


@dataclass(frozen=True)
class HostState:
    systemd: bool
    packages: frozenset[str]
    cluster_online: bool
    python_ready: bool
    users: tuple[User, ...]
    groups: tuple[Group, ...]
    verifier_active: bool = False
    nss_ready: bool = True
    unit_state: str = "not-found"
    unit_fragment: str = ""
    unit_dropins: tuple[str, ...] = ()


def _query(arguments: list[str]) -> str | None:
    """Stałe polecenia, bez shella, sekretów i odziedziczonych importów."""
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
    except (OSError, subprocess.TimeoutExpired, UnicodeError):
        return None
    output = result.stdout.strip()
    # Ubuntu/systemd może zwrócić 1 lub 4 dla brakującej jednostki. To jest
    # dowód braku wyłącznie przy pełnej, dokładnej odpowiedzi trzech pól;
    # inne błędy i częściowe odpowiedzi pozostają niedostępne.
    absent_unit = (
        arguments == UNIT_QUERY
        and result.returncode in {1, 4}
        and len(output.splitlines()) == 3
        and set(output.splitlines()) == {"LoadState=not-found", "FragmentPath=", "DropInPaths="}
    )
    return output if result.returncode == 0 or absent_unit else None


def inspect_runtime(runtime: Any) -> bool:
    """Sonda importów dopiero po niezależnej weryfikacji podpisu i plików.

    Wywołujący dostarcza VerifiedRuntime z zaufanego bootstrapu. Sonda nie
    ustanawia zaufania do interpretera i nigdy nie zastępuje jego weryfikacji.
    """
    executable = str(runtime.executable)
    packages = runtime.packages
    expected = json.dumps(packages, sort_keys=True)
    script = (
        "import sys,site,json,pathlib,importlib.metadata as m; "
        "import cryptography,psycopg,psycopg.pq,cffi,pycparser,typing_extensions,psycopg_binary; "
        f"expected=json.loads({expected!r}); "
        "actual={d.metadata['Name'].lower().replace('_','-'):d.version "
        "for d in m.distributions()}; "
        "root=pathlib.Path(sys.prefix).resolve(); "
        "origins=[pathlib.Path(x.__file__).resolve() for x in "
        "(cryptography,psycopg,cffi,pycparser,typing_extensions,psycopg_binary)]; "
        f"ok=(sys.executable=={executable!r} and sys.version_info[:2]==(3,12) "
        "and sys.flags.isolated==1 and sys.flags.no_user_site==1 and not site.ENABLE_USER_SITE "
        "and actual==expected and all(p.is_relative_to(root) for p in origins) "
        "and psycopg.pq.__impl__=='binary' and psycopg.pq.version()>=160000); "
        "sys.exit(0 if ok else 1)"
    )
    return _query([executable, "-I", "-s", "-B", "-c", script]) is not None


def inspect_host(runtime: Any | None = None) -> HostState:
    """Odmowa przy niedostępnej obserwacji; nigdy nie tworzy klastra ani kont."""
    empty = HostState(False, frozenset(), False, False, (), ())
    if sys.platform == "win32":
        return empty
    if os.name != "posix" or not sys.platform.startswith("linux"):
        return empty
    # Lazy imports zachowują kolekcję testów i inspekcję kontraktu na Windows.
    import grp
    import pwd

    try:
        users = tuple(
            User(p.pw_name, p.pw_uid, p.pw_gid, p.pw_dir, p.pw_shell) for p in pwd.getpwall()
        )
        groups = tuple(Group(g.gr_name, g.gr_gid, tuple(g.gr_mem)) for g in grp.getgrall())
        pid_one = Path("/proc/1/comm").read_text(encoding="ascii").strip() == "systemd"
    except (OSError, UnicodeError):
        return empty
    # Enumeracja NSS nie dowodzi kompletności (np. LDAP bez enumeracji).
    # Wymagamy zgodności także lookup name -> id -> name dla lokalnych kont.
    nss_ready = True
    try:
        for name in USERS:
            account = pwd.getpwnam(name)
            nss_ready = nss_ready and pwd.getpwuid(account.pw_uid).pw_name == name
            nss_ready = nss_ready and set(os.getgrouplist(name, account.pw_gid)) == {
                account.pw_gid,
                grp.getgrnam(IPC_GROUP).gr_gid,
            }
            nss_ready = nss_ready and any(
                item.name == name
                and item.uid == account.pw_uid
                and item.gid == account.pw_gid
                and item.home == account.pw_dir
                and item.shell == account.pw_shell
                for item in users
            )
        for name in (*USERS, IPC_GROUP, "postgres"):
            group = grp.getgrnam(name)
            nss_ready = nss_ready and grp.getgrgid(group.gr_gid).gr_name == name
            nss_ready = nss_ready and any(
                item.name == name
                and item.gid == group.gr_gid
                and set(item.members) == set(group.gr_mem)
                for item in groups
            )
    except (KeyError, OSError):
        nss_ready = False
    installed = frozenset(
        package
        for package in REQUIRED_PACKAGES
        if _query(["/usr/bin/dpkg-query", "-W", "-f=${db:Status-Abbrev}", package]) == "ii"
    )
    cluster = _query(["/usr/bin/pg_lsclusters", "--no-header", "16", "main"])
    columns = cluster.split() if cluster is not None else []
    unit_lines = (_query(POSTGRESQL_UNIT_QUERY) or "").splitlines()
    try:
        postgres_uid = pwd.getpwnam("postgres").pw_uid
    except (KeyError, OSError):
        postgres_uid = -1
    cluster_online = (
        len(columns) >= 7
        and columns[:5] == ["16", "main", "5432", "online", "postgres"]
        and len(unit_lines) == 2
        and set(unit_lines) == {"LoadState=loaded", "ActiveState=active"}
        and postgres_uid > 0
        and _postgresql_config_ready(postgres_uid)
    )
    # Brak argumentu oznacza wyłącznie obserwację systemu. Nie wykonujemy
    # systemowego interpretera jako zastępczego dowodu dla verifiera.
    python_ready = runtime is not None and inspect_runtime(runtime)
    manager = _query(["/usr/bin/systemctl", "show", "--property=Version", "--value"])
    active = _query(["/usr/bin/systemctl", "is-active", "cryptohunter-freshness-verifier.service"])
    unit = _query(UNIT_QUERY)
    observed = dict(line.split("=", 1) for line in (unit or "").splitlines() if "=" in line)
    unit_valid = set(observed) == {"LoadState", "FragmentPath", "DropInPaths"}
    return HostState(
        pid_one and bool(manager),
        installed,
        cluster_online,
        python_ready,
        users,
        groups,
        active == "active",
        nss_ready,
        observed.get("LoadState", "unknown") if unit_valid else "unknown",
        observed.get("FragmentPath", "") if unit_valid else "",
        tuple(observed.get("DropInPaths", "").split()) if unit_valid else (),
    )

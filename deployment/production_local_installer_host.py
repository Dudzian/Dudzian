"""Wyłącznie odczytowe obserwacje hosta dla instalatora Freshness.

Import modułu i testy kontraktu nie wymagają kont POSIX ani systemd.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
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
    cluster_online = (
        len(columns) >= 7
        and columns[:5] == ["16", "main", "5432", "online", "postgres"]
        and _query(
            [
                "/usr/bin/systemctl",
                "show",
                "postgresql@16-main.service",
                "--property=LoadState",
                "--value",
            ]
        )
        == "loaded"
        and _query(["/usr/bin/pg_conftool", "16", "main", "show", "include_dir"])
        in {"include_dir = 'conf.d'", 'include_dir = "conf.d"', "include_dir = conf.d"}
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

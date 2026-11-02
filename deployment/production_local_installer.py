"""Instalacja wyłącznie zatwierdzonego podłoża Freshness, bez kwalifikacji produktu.

Preflight nie mutuje hosta. Instalacja obsługuje nowy, przygotowany host lub
identyczne zakończone wdrożenie. Konta, klaster, klucze i schematy wymagają
osobnych ceremonii. Awaria pozostawia journal i artefakty do ręcznej inspekcji;
nie udajemy atomowego rollbacku wielu filesystemów i systemd.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import re
import runpy
import signal
import stat
import sys
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any, Callable

# Uruchomienie z -I nie dodaje katalogu skryptu do sys.path. Ładujemy tylko
# konkretny sibling, nigdy katalog bundle ani importy z cwd/PYTHONPATH.
if TYPE_CHECKING:
    from deployment.production_local_installer_host import (
        IPC_GROUP,
        REQUIRED_PACKAGES,
        USERS,
        Group,
        HostState,
        User,
        inspect_host,
    )
elif __package__:
    from deployment.production_local_installer_host import (
        IPC_GROUP,
        REQUIRED_PACKAGES,
        USERS,
        Group,
        HostState,
        User,
        inspect_host,
    )
else:
    sys.pycache_prefix = "/dev/null"
    _helper_path = Path(__file__).absolute().with_name("production_local_installer_host.py")
    if os.geteuid() == 0:
        for _path in (*reversed(_helper_path.parents), _helper_path):
            _status = _path.lstat()
            if (
                stat.S_ISLNK(_status.st_mode)
                or _status.st_uid != 0
                or _status.st_mode & 0o022
                or (
                    _path == _helper_path
                    and (not stat.S_ISREG(_status.st_mode) or _status.st_nlink != 1)
                )
            ):
                raise SystemExit("Odmowa: bootstrap instalatora nie jest chroniony")
    # run_path dla chronionego pliku .py czyta źródło bez sąsiedniego pyc;
    # SourceFileLoader i samo -B nie wykluczałyby odczytu obcego cache.
    _helper = runpy.run_path(str(_helper_path), run_name="_cryptohunter_installer_host")
    Group, HostState, User = _helper["Group"], _helper["HostState"], _helper["User"]
    IPC_GROUP, REQUIRED_PACKAGES, USERS = (
        _helper["IPC_GROUP"],
        _helper["REQUIRED_PACKAGES"],
        _helper["USERS"],
    )
    inspect_host = _helper["inspect_host"]

SCOPE = "cryptohunter.freshness.substrate.v1"
REQUIRED_FILES = frozenset(
    {
        "bot_core/__init__.py",
        "deployment/__init__.py",
        "bot_core/freshness_verifier_service.py",
        "bot_core/freshness_semantic_verifier.py",
        "bot_core/postgresql_freshness_authority.py",
        "bot_core/linux_service_ipc.py",
        "bot_core/linux_service_postgresql.py",
        "bot_core/production_signer_ipc.py",
        "deployment/freshness_verifier_entrypoint.py",
        "deployment/systemd/cryptohunter-freshness-verifier.service",
        "deployment/postgresql/pg_hba.conf",
        "deployment/postgresql/pg_ident.conf",
        "deployment/postgresql/production-local.conf",
    }
)
CODE = "usr/lib/cryptohunter"
STATE = "var/lib/cryptohunter-installer"
CONFIG_FILES = (
    (
        "deployment/systemd/cryptohunter-freshness-verifier.service",
        "etc/systemd/system/cryptohunter-freshness-verifier.service",
        0o644,
        False,
    ),
    ("deployment/postgresql/pg_hba.conf", "etc/cryptohunter/postgresql/pg_hba.conf", 0o640, True),
    (
        "deployment/postgresql/pg_ident.conf",
        "etc/cryptohunter/postgresql/pg_ident.conf",
        0o640,
        True,
    ),
    (
        "deployment/postgresql/production-local.conf",
        "etc/postgresql/16/main/conf.d/cryptohunter-production-local.conf",
        0o640,
        True,
    ),
)
MAX_FILE = 8 * 1024 * 1024
MAX_TOTAL = 128 * 1024 * 1024
MAX_MANIFEST = 4 * 1024 * 1024


class InstallerError(RuntimeError):
    """Stały kod i bezpieczna przyczyna, bez zawartości plików i stderr hosta."""

    def __init__(self, code: int, reason: str):
        super().__init__(reason)
        self.code = code
        self.reason = reason


@dataclass(frozen=True)
class Config:
    bundle: Path
    manifest: Path
    signature: Path
    trusted_key: Path
    expected_revision: str
    root: Path = Path("/")
    owner_uid: int = 0


@dataclass(frozen=True)
class Artifact:
    path: str
    sha256: str
    size: int
    mode: int
    content: bytes


@dataclass(frozen=True)
class Plan:
    state: str
    manifest_sha256: str
    source_revision: str
    artifacts: tuple[Artifact, ...]
    postgres_gid: int


def _fail(condition: bool, code: int, reason: str) -> None:
    if not condition:
        raise InstallerError(code, reason)


def _parts(path: str) -> tuple[str, ...]:
    candidate = PurePosixPath(path)
    _fail(
        type(path) is str
        and bool(path)
        and not candidate.is_absolute()
        and str(candidate) == path
        and all(part not in {".", ".."} for part in candidate.parts),
        23,
        "Nieprawidłowa ścieżka względna",
    )
    return candidate.parts


def _read_fd(fd: int, limit: int) -> bytes:
    chunks: list[bytes] = []
    length = 0
    while True:
        chunk = os.read(fd, min(65536, limit + 1 - length))
        if not chunk:
            return b"".join(chunks)
        length += len(chunk)
        _fail(length <= limit, 12, "Artefakt przekracza dozwolony rozmiar")
        chunks.append(chunk)


def _open_absolute(path: Path) -> int:
    """Otwórz dane bez podążania za symlinkiem w dowolnym komponencie."""
    _fail(path.is_absolute() and ".." not in path.parts, 23, "Wymagana ścieżka absolutna")
    fd = os.open("/", os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        for index, part in enumerate(path.parts[1:]):
            flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK
            if index != len(path.parts) - 2:
                flags |= os.O_DIRECTORY
            next_fd = os.open(part, flags, dir_fd=fd)
            os.close(fd)
            fd = next_fd
        return fd
    except BaseException:
        os.close(fd)
        raise


def _read_input(path: Path, limit: int) -> bytes:
    fd = _open_absolute(path)
    try:
        before = os.fstat(fd)
        _fail(
            stat.S_ISREG(before.st_mode) and before.st_nlink == 1,
            12,
            "Artefakt nie jest pojedynczym zwykłym plikiem",
        )
        data = _read_fd(fd, limit)
        after = os.fstat(fd)
        _fail(
            (before.st_size, before.st_mtime_ns, before.st_ctime_ns)
            == (after.st_size, after.st_mtime_ns, after.st_ctime_ns),
            12,
            "Artefakt zmienił się podczas odczytu",
        )
        return data
    finally:
        os.close(fd)


def _chmod_fd(fd: int, mode: int) -> None:
    if sys.platform == "win32":
        raise InstallerError(20, "Zmiana metadanych wymaga natywnego Linux")
    os.fchmod(fd, mode)


def _chown_fd(fd: int, gid: int) -> None:
    if sys.platform == "win32":
        raise InstallerError(20, "Zmiana metadanych wymaga natywnego Linux")
    os.fchown(fd, -1, gid)


class _Tree:
    """Wszystkie mutacje są względne do chronionych, przypiętych dirfd."""

    def __init__(self, root: Path, owner_uid: int):
        self.owner_uid = owner_uid
        self.fd = _open_absolute(root)
        try:
            self._directory(self.fd)
        except BaseException:
            os.close(self.fd)
            raise

    def close(self) -> None:
        os.close(self.fd)

    def _directory(self, fd: int) -> None:
        value = os.fstat(fd)
        _fail(
            stat.S_ISDIR(value.st_mode)
            and value.st_uid == self.owner_uid
            and not value.st_mode & 0o022,
            23,
            "Niezgodna własność lub uprawnienia chronionego katalogu",
        )

    def directory(
        self, path: str, *, create: bool = False, mode: int = 0o755, gid: int | None = None
    ) -> int:
        fd = os.dup(self.fd)
        try:
            parts = _parts(path)
            for index, part in enumerate(parts):
                try:
                    next_fd = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
                except FileNotFoundError:
                    if not create:
                        raise
                    os.mkdir(part, 0o700, dir_fd=fd)
                    next_fd = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
                    _chmod_fd(next_fd, mode if index == len(parts) - 1 else 0o755)
                    if gid is not None and index == len(parts) - 1:
                        _chown_fd(next_fd, gid)
                    os.fsync(fd)
                self._directory(next_fd)
                os.close(fd)
                fd = next_fd
            return fd
        except BaseException:
            os.close(fd)
            raise

    def parent(self, path: str) -> tuple[int, str]:
        parts = _parts(path)
        fd = self.directory("/".join(parts[:-1])) if len(parts) > 1 else os.dup(self.fd)
        return fd, parts[-1]

    def status(self, path: str) -> os.stat_result | None:
        try:
            fd, name = self.parent(path)
        except FileNotFoundError:
            return None
        try:
            try:
                return os.stat(name, dir_fd=fd, follow_symlinks=False)
            except FileNotFoundError:
                return None
        finally:
            os.close(fd)

    def read(
        self, path: str, *, mode: int | None = None, gid: int | None = None, limit: int = MAX_FILE
    ) -> bytes:
        fd, name = self.parent(path)
        try:
            file_fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=fd)
            try:
                value = os.fstat(file_fd)
                _fail(
                    stat.S_ISREG(value.st_mode)
                    and value.st_nlink == 1
                    and value.st_uid == self.owner_uid
                    and not value.st_mode & 0o022,
                    23,
                    "Niezgodny chroniony plik",
                )
                _fail(
                    mode is None or stat.S_IMODE(value.st_mode) == mode,
                    23,
                    "Niezgodne uprawnienia pliku docelowego",
                )
                _fail(gid is None or value.st_gid == gid, 23, "Niezgodna grupa pliku docelowego")
                return _read_fd(file_fd, limit)
            finally:
                os.close(file_fd)
        finally:
            os.close(fd)

    def write_new(self, path: str, data: bytes, mode: int, gid: int | None = None) -> None:
        fd, name = self.parent(path)
        try:
            target = os.open(
                name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=fd
            )
            try:
                view = memoryview(data)
                while view:
                    written = os.write(target, view)
                    _fail(written > 0, 30, "Przerwano zapis pliku")
                    view = view[written:]
                if gid is not None:
                    _chown_fd(target, gid)
                _chmod_fd(target, mode)
                os.fsync(target)
            finally:
                os.close(target)
            os.fsync(fd)
        finally:
            os.close(fd)


def _inventory(path: Path, *, owner_uid: int | None = None) -> dict[str, tuple[bytes, int]]:
    root_fd = _open_absolute(path)
    result: dict[str, tuple[bytes, int]] = {}
    total = 0
    nodes = 0
    directories: set[str] = set()

    def walk(fd: int, prefix: str) -> None:
        nonlocal total, nodes
        _fail(len(PurePosixPath(prefix).parts) <= 32, 12, "Zbyt głęboki inwentarz")
        if owner_uid is not None:
            directory_stat = os.fstat(fd)
            _fail(
                directory_stat.st_uid == owner_uid
                and stat.S_IMODE(directory_stat.st_mode) == 0o755,
                23,
                "Niezgodny katalog zainstalowanego kodu",
            )
        for name in sorted(os.listdir(fd)):
            nodes += 1
            _fail(nodes <= 32768, 12, "Zbyt duży inwentarz ścieżek")
            relative = f"{prefix}/{name}" if prefix else name
            _parts(relative)
            _fail(
                not any(part in {"__pycache__", ".git"} for part in PurePosixPath(relative).parts)
                and not relative.endswith((".pyc", ".pyo")),
                12,
                "Niedozwolony plik importu lub cache w artefakcie",
            )
            child = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=fd)
            try:
                value = os.fstat(child)
                if stat.S_ISDIR(value.st_mode):
                    directories.add(relative)
                    walk(child, relative)
                else:
                    _fail(
                        stat.S_ISREG(value.st_mode) and value.st_nlink == 1,
                        12,
                        "Symlink, hardlink lub specjalny plik w artefakcie",
                    )
                    content = _read_fd(child, MAX_FILE)
                    total += len(content)
                    _fail(
                        total <= MAX_TOTAL and len(result) < 16384,
                        12,
                        "Przekroczono limit inwentarza",
                    )
                    result[relative] = (content, stat.S_IMODE(value.st_mode))
            finally:
                os.close(child)

    try:
        _fail(stat.S_ISDIR(os.fstat(root_fd).st_mode), 12, "Bundle nie jest katalogiem")
        walk(root_fd, "")
        expected_directories = {
            str(parent)
            for item in result
            for parent in PurePosixPath(item).parents
            if str(parent) != "."
        }
        _fail(
            directories == expected_directories,
            12,
            "Dodatkowy niezatwierdzony katalog w artefakcie",
        )
        return result
    finally:
        os.close(root_fd)


def _manifest(config: Config, tree: _Tree) -> tuple[str, tuple[Artifact, ...]]:
    _fail(
        re.fullmatch(r"[0-9a-f]{40}", config.expected_revision) is not None,
        10,
        "Nieprawidłowa oczekiwana rewizja",
    )
    try:
        key_relative = config.trusted_key.relative_to(config.root).as_posix()
        _fail(
            not config.trusted_key.is_relative_to(config.bundle),
            11,
            "Autorytet musi pochodzić spoza bundle",
        )
        key_bytes = tree.read(key_relative, limit=32)
        signature = _read_input(config.signature, 64)
        raw = _read_input(config.manifest, MAX_MANIFEST)
        _fail(
            len(key_bytes) == 32 and len(signature) == 64,
            11,
            "Niepoprawny format klucza lub podpisu",
        )
        from cryptography.exceptions import InvalidSignature
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

        try:
            Ed25519PublicKey.from_public_bytes(key_bytes).verify(signature, raw)
        except (InvalidSignature, ValueError) as exc:
            raise InstallerError(11, "Nieprawidłowy podpis zatwierdzenia instalacji") from exc

        def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
            value: dict[str, Any] = {}
            for key, item in pairs:
                _fail(key not in value, 10, "Powtórzone pole manifestu")
                value[key] = item
            return value

        data = json.loads(raw, object_pairs_hook=unique)
    except InstallerError:
        raise
    except (ValueError, OSError, ImportError) as exc:
        raise InstallerError(
            10, "Nie można zweryfikować manifestu i niezależnego autorytetu"
        ) from exc
    _fail(
        type(data) is dict and set(data) == {"schema_version", "scope", "source_revision", "files"},
        10,
        "Nieprawidłowy kontrakt manifestu",
    )
    _fail(
        type(data["schema_version"]) is int
        and data["schema_version"] == 1
        and data["scope"] == SCOPE
        and data["source_revision"] == config.expected_revision,
        10,
        "Manifest nie zatwierdza tego zakresu i rewizji",
    )
    files = data["files"]
    _fail(type(files) is list and 0 < len(files) <= 16384, 10, "Nieprawidłowy inwentarz manifestu")
    inventory = _inventory(config.bundle)
    artifacts: list[Artifact] = []
    seen: set[str] = set()
    for entry in files:
        _fail(
            type(entry) is dict and set(entry) == {"path", "sha256", "size", "mode"},
            10,
            "Nieprawidłowy rekord artefaktu",
        )
        path = entry["path"]
        _fail(type(path) is str, 10, "Nieprawidłowa nazwa artefaktu")
        parts = _parts(path)
        _fail(
            len(parts) >= 2 and parts[0] in {"bot_core", "deployment"} and path not in seen,
            10,
            "Ścieżka artefaktu poza zakresem lub powtórzona",
        )
        _fail(
            type(entry["sha256"]) is str
            and re.fullmatch(r"[0-9a-f]{64}", entry["sha256"]) is not None
            and type(entry["size"]) is int
            and 0 <= entry["size"] <= MAX_FILE
            and type(entry["mode"]) is int
            and entry["mode"] in {0o644, 0o755},
            10,
            "Nieprawidłowe metadane artefaktu",
        )
        _fail(path in inventory, 12, "Brak podpisanego artefaktu")
        content, mode = inventory[path]
        _fail(
            len(content) == entry["size"]
            and hashlib.sha256(content).hexdigest() == entry["sha256"]
            and mode == entry["mode"],
            12,
            "Zmieniona treść lub uprawnienia artefaktu",
        )
        seen.add(path)
        artifacts.append(Artifact(path, entry["sha256"], entry["size"], mode, content))
    _fail(
        seen == set(inventory) and REQUIRED_FILES <= seen,
        12,
        "Niepełny lub dodatkowy niezatwierdzony kod w bundle",
    )
    return hashlib.sha256(raw).hexdigest(), tuple(artifacts)


def _accounts(host: HostState) -> int:
    _fail(host.nss_ready, 22, "Niespójna lub niekompletna obserwacja NSS")
    chosen_users = []
    chosen_groups = []
    for name in USERS:
        candidates = [item for item in host.users if item.name == name]
        primary = [item for item in host.groups if item.name == name]
        _fail(
            len(candidates) == len(primary) == 1, 22, "Brak lub kolizja przygotowanych kont i grup"
        )
        user, group = candidates[0], primary[0]
        _fail(
            0 < user.uid < 1000
            and 0 < group.gid < 1000
            and user.gid == group.gid
            and user.home == "/nonexistent"
            and user.shell == "/usr/sbin/nologin"
            and not group.members,
            22,
            "Niezgodny profil konta systemowego",
        )
        _fail(
            sum(item.uid == user.uid for item in host.users) == 1
            and sum(item.gid == group.gid for item in host.groups) == 1,
            22,
            "Kolizja UID lub GID",
        )
        chosen_users.append(user)
        chosen_groups.append(group)
    ipc = [item for item in host.groups if item.name == IPC_GROUP]
    postgres = [item for item in host.groups if item.name == "postgres"]
    _fail(len(ipc) == len(postgres) == 1, 22, "Brak lub kolizja grup IPC/PostgreSQL")
    _fail(
        postgres[0].gid > 0 and sum(item.gid == postgres[0].gid for item in host.groups) == 1,
        22,
        "Kolizja grupy PostgreSQL",
    )
    group = ipc[0]
    _fail(
        0 < group.gid < 1000
        and len(group.members) == len(USERS)
        and set(group.members) == set(USERS)
        and sum(item.gid == group.gid for item in host.groups) == 1,
        22,
        "Niezgodna grupa IPC",
    )
    _fail(
        chosen_users[0].uid != chosen_users[1].uid
        and len({item.gid for item in chosen_groups} | {group.gid, postgres[0].gid}) == 4,
        22,
        "Brak separacji tożsamości",
    )
    restricted_gids = {item.gid for item in chosen_groups} | {group.gid}
    _fail(
        all(item.name in USERS or item.gid not in restricted_gids for item in host.users),
        22,
        "Obce konto z chronioną grupą podstawową",
    )
    for other in host.groups:
        _fail(
            other.name == IPC_GROUP or not set(other.members).intersection(USERS),
            22,
            "Nadmiarowe członkostwo konta usługowego",
        )
    return int(postgres[0].gid)


def _receipt(plan: Plan, status: str) -> bytes:
    return json.dumps(
        {
            "scope": SCOPE,
            "source_revision": plan.source_revision,
            "manifest_sha256": plan.manifest_sha256,
            "status": status,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")


def _installed(tree: _Tree, config: Config, plan: Plan) -> None:
    _fail(
        tree.status(f"{STATE}/receipt.json") is not None
        and tree.status(f"{STATE}/journal.json") is not None,
        24,
        "Przerwana instalacja wymaga ręcznego odzyskiwania",
    )
    _fail(
        tree.read(f"{STATE}/receipt.json", mode=0o600) == _receipt(plan, "complete")
        and tree.read(f"{STATE}/journal.json", mode=0o600) == _receipt(plan, "complete"),
        24,
        "Niekompletne lub inne wdrożenie wymaga odzyskiwania",
    )
    state_fd = tree.directory(STATE)
    try:
        _fail(
            stat.S_IMODE(os.fstat(state_fd).st_mode) == 0o700
            and set(os.listdir(state_fd)) == {"receipt.json", "journal.json"},
            24,
            "Niezgodny katalog transakcji instalatora",
        )
    finally:
        os.close(state_fd)
    _verify_files(tree, config, plan)


def _verify_files(tree: _Tree, config: Config, plan: Plan) -> None:
    actual = _inventory(config.root / CODE, owner_uid=config.owner_uid)
    expected = {item.path: (item.content, item.mode) for item in plan.artifacts}
    _fail(actual == expected, 24, "Drift inwentarza zainstalowanego kodu")
    for item in plan.artifacts:
        _fail(
            tree.read(f"{CODE}/{item.path}", mode=item.mode) == item.content,
            24,
            "Drift integralności zainstalowanego kodu",
        )
    for source, target, mode, database in CONFIG_FILES:
        artifact = next(item for item in plan.artifacts if item.path == source)
        _fail(
            tree.read(target, mode=mode, gid=plan.postgres_gid if database else None)
            == artifact.content,
            24,
            "Drift plików konfiguracji wdrożenia",
        )


def _preflight(config: Config, host: HostState) -> Plan:
    tree = _Tree(config.root, config.owner_uid)
    try:
        digest, artifacts = _manifest(config, tree)
        _fail(
            host.systemd and REQUIRED_PACKAGES <= host.packages and host.python_ready,
            20,
            "Brak systemd, wymaganych pakietów lub zależności izolowanego Pythona",
        )
        _fail(
            host.cluster_online,
            21,
            "Brak gotowego klastra PostgreSQL 16/main na porcie 5432; "
            "instalacja odmówiona bez zmian",
        )
        postgres_gid = _accounts(host)
        for path in ("usr/lib", "etc/systemd/system", "var/lib", "etc/postgresql/16/main/conf.d"):
            for prefix in (
                str(item) for item in reversed(PurePosixPath(path).parents) if str(item) != "."
            ):
                fd = tree.directory(prefix)
                try:
                    _fail(
                        stat.S_IMODE(os.fstat(fd).st_mode) == 0o755,
                        23,
                        "Niezgodne uprawnienia przejścia do katalogu usług",
                    )
                finally:
                    os.close(fd)
            fd = tree.directory(path)
            try:
                _fail(
                    stat.S_IMODE(os.fstat(fd).st_mode) == 0o755,
                    23,
                    "Niezgodne uprawnienia katalogu usług",
                )
            finally:
                os.close(fd)
        # Wszystkie istniejące komponenty chronione; właściciel postgres w
        # conf.d lub jego parentach wymaga odrębnego planu administratora.
        for optional in ("etc/cryptohunter", "etc/cryptohunter/postgresql"):
            if tree.status(optional) is not None:
                fd = tree.directory(optional)
                try:
                    if optional == "etc/cryptohunter":
                        _fail(
                            stat.S_IMODE(os.fstat(fd).st_mode) == 0o755,
                            23,
                            "Niezgodny katalog konfiguracji usług",
                        )
                finally:
                    os.close(fd)
        _fail(
            not host.unit_dropins and host.unit_state in {"not-found", "loaded"},
            24,
            "Obca, zmieniona lub niedostępna konfiguracja jednostki systemd",
        )
        for directory in ("etc/systemd/system", "run/systemd/system", "usr/lib/systemd/system"):
            for dropin in (
                "cryptohunter-freshness-verifier.service.d",
                "cryptohunter-freshness-.service.d",
                "cryptohunter-.service.d",
                "service.d",
            ):
                _fail(
                    tree.status(f"{directory}/{dropin}") is None,
                    24,
                    "Niezatwierdzony drop-in jednostki systemd",
                )
        if tree.status("etc/cryptohunter/postgresql") is not None:
            directory_stat = tree.status("etc/cryptohunter/postgresql")
            _fail(
                directory_stat is not None
                and stat.S_IMODE(directory_stat.st_mode) == 0o750
                and directory_stat.st_gid == postgres_gid,
                23,
                "Niezgodny katalog konfiguracji PostgreSQL",
            )
        plan = Plan("fresh", digest, config.expected_revision, artifacts, postgres_gid)
        status = tree.status(STATE)
        if status is not None:
            _fail(
                host.unit_state == "not-found"
                or host.unit_fragment
                == "/etc/systemd/system/cryptohunter-freshness-verifier.service",
                24,
                "Obcy fragment jednostki systemd",
            )
            _installed(tree, config, plan)
            return Plan("installed", digest, config.expected_revision, artifacts, postgres_gid)
        _fail(
            not host.verifier_active and host.unit_state == "not-found" and not host.unit_fragment,
            24,
            "Wcześniejsza lub nieznana jednostka systemd wymaga osobnej migracji",
        )
        for directory in ("run/systemd/system", "usr/lib/systemd/system"):
            _fail(
                tree.status(f"{directory}/cryptohunter-freshness-verifier.service") is None,
                24,
                "Wcześniejsza jednostka dostawcy wymaga osobnej migracji",
            )
        _fail(
            tree.status(CODE) is None
            and tree.status(f"usr/lib/.cryptohunter-stage-{digest}") is None
            and all(tree.status(target) is None for _, target, _, _ in CONFIG_FILES),
            24,
            "Wcześniejsze, częściowe lub obce wdrożenie wymaga ochrony danych i odzyskiwania",
        )
        _fail(
            sys.platform.startswith("linux") and hasattr(ctypes.CDLL(None), "renameat2"),
            20,
            "Brak bezpiecznej publikacji Linux renameat2",
        )
        return plan
    finally:
        tree.close()


def preflight(config: Config, host: HostState | None = None) -> Plan:
    """Tylko odczyt; jawne kody odmowy, bez importowania kodu bundle."""
    try:
        return _preflight(config, inspect_host() if host is None else host)
    except InstallerError:
        raise
    except (OSError, ValueError) as exc:
        raise InstallerError(23, "Brak, podmiana lub niedostępność chronionej ścieżki") from exc


def _publish(tree: _Tree, source: str, destination: str) -> None:
    source_fd, source_name = tree.parent(source)
    target_fd, target_name = tree.parent(destination)
    try:
        libc = ctypes.CDLL(None, use_errno=True)
        function = libc.renameat2
        function.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        function.restype = ctypes.c_int
        result = function(
            source_fd, os.fsencode(source_name), target_fd, os.fsencode(target_name), 1
        )
        _fail(result == 0, 30, "Odmowa publikacji bez nadpisywania; wymagane odzyskiwanie")
        os.fsync(target_fd)
    finally:
        os.close(source_fd)
        os.close(target_fd)


def _journal(tree: _Tree, plan: Plan, phase: str) -> None:
    tree.write_new(f"{STATE}/journal.next", _receipt(plan, phase), 0o600)
    fd = tree.directory(STATE)
    try:
        os.replace("journal.next", "journal.json", src_dir_fd=fd, dst_dir_fd=fd)
        os.fsync(fd)
    finally:
        os.close(fd)


def install(
    config: Config, host: HostState | None = None, checkpoint: Callable[[str], None] | None = None
) -> Plan:
    """Po pełnym preflight, bez overwrite/cleanup i bez mutacji usług/NSS/DB."""
    if sys.platform == "win32":
        raise InstallerError(20, "Instalacja wymaga natywnego Linux")
    _fail(
        os.geteuid() == config.owner_uid, 31, "Instalacja wymaga administratora chronionego drzewa"
    )
    plan = preflight(config, host)
    if plan.state == "installed":
        return plan
    tree = _Tree(config.root, config.owner_uid)
    stage = f"usr/lib/.cryptohunter-stage-{plan.manifest_sha256}"
    try:
        # mkdir jest bramką wyłączności. Nie adoptujemy katalogu utworzonego
        # pomiędzy preflight a instalacją, również przez inny instalator.
        parent, name = tree.parent(STATE)
        try:
            os.mkdir(name, 0o700, dir_fd=parent)
            os.fsync(parent)
        finally:
            os.close(parent)
        tree.write_new(f"{STATE}/journal.json", _receipt(plan, "preparing"), 0o600)
        parent, name = tree.parent(stage)
        try:
            os.mkdir(name, 0o700, dir_fd=parent)
            os.fsync(parent)
        finally:
            os.close(parent)
        for item in plan.artifacts:
            fd = tree.directory(f"{stage}/{PurePosixPath(item.path).parent}", create=True)
            os.close(fd)
            tree.write_new(f"{stage}/{item.path}", item.content, item.mode)
            _fail(
                tree.read(f"{stage}/{item.path}", mode=item.mode) == item.content,
                30,
                "Błąd integralności kopii staging",
            )
        fd = tree.directory(stage)
        _chmod_fd(fd, 0o755)
        os.fsync(fd)
        os.close(fd)
        _journal(tree, plan, "staged")
        if checkpoint is not None:
            checkpoint("staged")
        _publish(tree, stage, CODE)
        _journal(tree, plan, "code-published")
        if checkpoint is not None:
            checkpoint("code-published")
        for index, (source, target, mode, database) in enumerate(CONFIG_FILES):
            fd = tree.directory(
                str(PurePosixPath(target).parent),
                create=True,
                mode=0o750 if database and target.startswith("etc/cryptohunter/") else 0o755,
                gid=plan.postgres_gid
                if database and target.startswith("etc/cryptohunter/")
                else None,
            )
            os.close(fd)
            artifact = next(item for item in plan.artifacts if item.path == source)
            tree.write_new(target, artifact.content, mode, plan.postgres_gid if database else None)
            _fail(
                tree.read(target, mode=mode, gid=plan.postgres_gid if database else None)
                == artifact.content,
                30,
                "Błąd integralności konfiguracji",
            )
            _journal(tree, plan, f"configuration-{index + 1}")
            if checkpoint is not None:
                checkpoint(f"configuration-{index + 1}")
        _verify_files(tree, config, plan)
        tree.write_new(f"{STATE}/receipt.json", _receipt(plan, "complete"), 0o600)
        # Dziennik kończymy przez rename własnego nowego pliku. Nigdy nie
        # nadpisujemy kodu, kluczy ani konfiguracji poprzedniego wdrożenia.
        _journal(tree, plan, "complete")
        _installed(tree, config, plan)
        return Plan(
            "installed",
            plan.manifest_sha256,
            plan.source_revision,
            plan.artifacts,
            plan.postgres_gid,
        )
    except (Exception, KeyboardInterrupt) as exc:
        if isinstance(exc, InstallerError):
            raise
        raise InstallerError(
            30, "Instalacja przerwana; zachowano artefakty i journal do ręcznego odzyskiwania"
        ) from exc
    finally:
        tree.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Chroniony instalator podłoża Freshness; bez kwalifikacji całego CryptoHunter"
    )
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--preflight", action="store_true")
    action.add_argument("--install", action="store_true")
    for name in ("bundle", "manifest", "signature", "trusted-key", "expected-revision"):
        parser.add_argument(f"--{name}", required=True)
    args = parser.parse_args(argv)
    config = Config(
        Path(args.bundle),
        Path(args.manifest),
        Path(args.signature),
        Path(args.trusted_key),
        args.expected_revision,
    )

    def interrupted(_signum: int, _frame: Any) -> None:
        raise KeyboardInterrupt

    previous = signal.signal(signal.SIGTERM, interrupted)
    try:
        plan = install(config) if args.install else preflight(config)
        print(
            json.dumps(
                {
                    "result": "READY" if args.preflight else "INSTALLED",
                    "scope": SCOPE,
                    "state": plan.state,
                    "manifest_sha256": plan.manifest_sha256,
                    "production_qualification": "WITHHELD",
                },
                sort_keys=True,
            )
        )
        return 0
    except InstallerError as exc:
        print(
            json.dumps(
                {
                    "result": "REFUSED",
                    "code": exc.code,
                    "reason": exc.reason,
                    "production_qualification": "WITHHELD",
                },
                ensure_ascii=True,
                sort_keys=True,
            )
        )
        return exc.code
    except KeyboardInterrupt:
        print(
            '{"result":"REFUSED","code":30,"reason":"Przerwano diagnostyke",'
            '"production_qualification":"WITHHELD"}'
        )
        return 30
    finally:
        signal.signal(signal.SIGTERM, previous)


if __name__ == "__main__":
    raise SystemExit(main())

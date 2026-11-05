"""Chroniony, jednorazowy kontrakt próby odmowy między UID.

To narzędzie operatora i odczytowa bramka; nie jest brokerem ani demonem.
Nie tworzy zatwierdzenia VM i nie przyznaje kwalifikacji produktu.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import sys
from typing import Any

from deployment.stage9_vm_environment import (
    DEFAULT_APPROVAL,
    REQUIRED_PACKAGES,
    _directory_without_acl,
    _protected,
    _read_approval,
    _VM_TYPES,
    inspect_environment,
    require_approved_vm,
)

MANIFEST = Path("/etc/cryptohunter/stage9-cross-uid-trial.json")
ROOT = Path("/var/lib/stage9-cross-uid")
SCOPE = "TEST_ONLY_STAGE9_CROSS_UID_DENIAL"
IPC_GROUP = "freshness_verifier_ipc"
RUNTIME_GROUP = "os_freshness_runtime"
_ACL_NAMES = {"system.posix_acl_access", "system.posix_acl_default"}
_MAX_BYTES = 65536
_HARNESS = Path("scripts/diagnostics/stage9_cross_uid_identity.py")
_CONTRACT = Path("scripts/diagnostics/stage9_cross_uid_trial.py")
_FIELDS = {
    "schema",
    "scope",
    "trial_id",
    "revision",
    "environment_id",
    "machine_id",
    "boot_id",
    "vm_image_sha256",
    "vm_snapshot",
    "packages",
    "approval_sha256",
    "source_root",
    "python_executable",
    "harness_sha256",
    "contract_sha256",
}
_PATTERNS = {
    "trial_id": r"[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}",
    "revision": r"[0-9a-f]{40}",
    "environment_id": r"[a-z0-9][a-z0-9._-]{0,95}",
    "machine_id": r"[0-9a-f]{32}",
    "boot_id": r"[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}",
    "vm_image_sha256": r"[0-9a-f]{64}",
    "vm_snapshot": r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}",
    "approval_sha256": r"[0-9a-f]{64}",
    "harness_sha256": r"[0-9a-f]{64}",
    "contract_sha256": r"[0-9a-f]{64}",
}
_BINDING_FIELDS = (
    "scope",
    "trial_id",
    "revision",
    "environment_id",
    "machine_id",
    "boot_id",
    "vm_image_sha256",
)


class TrialError(ValueError):
    """Niekompletna lub zmieniona granica próby daje odmowę."""


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise TrialError(reason)


def read_protected(path: Path) -> bytes:
    """Odczyt fd-relative, bez symlinków, hardlinków i zapisywalnych przodków."""
    result: bytes = _read_approval(Path(path))
    return result


def _unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        _require(key not in result, "Powtórzone pole JSON")
        result[key] = value
    return result


def _constant(value: str) -> None:
    raise TrialError("Niedozwolona stała JSON")


def strict_json(raw: bytes) -> dict[str, Any]:
    _require(type(raw) is bytes and 0 < len(raw) <= _MAX_BYTES, "Nieprawidłowy rozmiar JSON")
    try:
        value = json.loads(raw, object_pairs_hook=_unique, parse_constant=_constant)
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise TrialError("Nieprawidłowy JSON kontraktu") from exc
    _require(type(value) is dict, "Wymagany obiekt JSON")
    result: dict[str, Any] = value
    return result


def _absolute(value: object) -> Path:
    _require(type(value) is str, "Wymagana tekstowa ścieżka bezwzględna")
    path = Path(str(value))
    _require(
        path.is_absolute() and path != Path("/") and ".." not in path.parts and str(path) == value,
        "Wymagana dokładna ścieżka bezwzględna",
    )
    return path


def _open_directory(path: Path) -> int:
    """Walker O_RDONLY wymaga r katalogów i x do lookup dzieci; nie dopuszcza ACL."""
    _require(path.is_absolute() and ".." not in path.parts, "Nieprawidłowy katalog chroniony")
    descriptor = os.open("/", os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        _require(_directory_without_acl(descriptor), "Niezabezpieczony katalog root lub ACL")
        for component in path.parts[1:]:
            child = os.open(
                component, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=descriptor
            )
            os.close(descriptor)
            descriptor = child
            _require(_directory_without_acl(descriptor), "Niezabezpieczony przodek lub ACL")
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _protected_directory(path: Path) -> None:
    descriptor = _open_directory(path)
    os.close(descriptor)


def _protected_interpreter(path: Path) -> None:
    """Dopuszczaj tylko końcowe symlinki root, np. bin/python w venv."""
    visited: set[Path] = set()
    for _ in range(16):
        _require(path not in visited, "Pętla symlinków interpretera")
        visited.add(path)
        parent = _open_directory(path.parent)
        try:
            info = os.stat(path.name, dir_fd=parent, follow_symlinks=False)
            if stat.S_ISLNK(info.st_mode):
                _require(
                    info.st_uid == 0 and info.st_nlink == 1, "Niezabezpieczony symlink interpretera"
                )
                target = Path(os.readlink(path.name, dir_fd=parent))
                path = Path(
                    os.path.normpath(target if target.is_absolute() else path.parent / target)
                )
            else:
                _require(
                    _protected(info, directory=False) and bool(info.st_mode & 0o111),
                    "Niezabezpieczony plik interpretera",
                )
                return
        finally:
            os.close(parent)
    raise TrialError("Nadmierny łańcuch symlinków interpretera")


def require_administrator() -> None:
    if sys.platform != "linux":
        raise TrialError("Wymagany natywny Linux")
    _require(os.geteuid() == 0, "Zapis kontraktu wymaga administratora VM")


def group_gid(group: str) -> int:
    if sys.platform != "linux":
        raise TrialError("Wymagany natywny Linux")
    import grp

    _require(group in {IPC_GROUP, RUNTIME_GROUP}, "Nieautoryzowana grupa odczytu próby")
    gid = grp.getgrnam(group).gr_gid
    _require(type(gid) is int and gid > 0, "Brak odrębnej grupy odczytu próby")
    return gid


def _no_acl(descriptor: int) -> None:
    if sys.platform != "linux":
        raise TrialError("Wymagany natywny Linux")
    _require(
        not _ACL_NAMES.intersection(os.listxattr(descriptor)), "Niedozwolone ACL nowego dowodu"
    )


def _file_state(
    descriptor: int, parent: int, name: str, size: int, mode: int, gid: int | None
) -> None:
    after = os.fstat(descriptor)
    linked = os.stat(name, dir_fd=parent, follow_symlinks=False)
    _require(
        (after.st_dev, after.st_ino) == (linked.st_dev, linked.st_ino)
        and after.st_size == linked.st_size == size
        and stat.S_IMODE(after.st_mode) == stat.S_IMODE(linked.st_mode) == mode
        and (gid is None or after.st_gid == linked.st_gid == gid)
        and _protected(after, directory=False)
        and _protected(linked, directory=False),
        "Nowy dowód zmienił się podczas zapisu",
    )
    _no_acl(descriptor)


def write_new(path: Path, raw: bytes, mode: int, *, group: str | None = None) -> None:
    """Zapisuj prywatnie; pełny, sprawdzony plik udostępniaj tylko wskazanej grupie."""
    if sys.platform != "linux":
        raise TrialError("Wymagany natywny Linux")
    require_administrator()
    _require(
        type(mode) is int
        and (
            (mode == 0o600 and group is None)
            or (mode == 0o640 and group in {IPC_GROUP, RUNTIME_GROUP})
        ),
        "Niedozwolony tryb pliku dowodu",
    )
    gid = group_gid(group) if group is not None else 0
    _require(type(raw) is bytes and 0 < len(raw) <= _MAX_BYTES, "Nieprawidłowy rozmiar dowodu")
    path = _absolute(str(path))
    parent = _open_directory(path.parent)
    descriptor = -1
    try:
        descriptor = os.open(
            path.name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o600,
            dir_fd=parent,
        )
        _no_acl(descriptor)
        os.fchmod(descriptor, 0o600)
        _require(_protected(os.fstat(descriptor), directory=False), "Niezabezpieczony nowy dowód")
        remaining = memoryview(raw)
        while remaining:
            written = os.write(descriptor, remaining)
            _require(written > 0, "Niepełny zapis dowodu")
            remaining = remaining[written:]
        os.fsync(descriptor)
        _file_state(descriptor, parent, path.name, len(raw), 0o600, None)
        # fchmod nie może uaktywnić odziedziczonego ACL ani zastanej grupy SGID.
        os.fchown(descriptor, 0, gid)
        _file_state(descriptor, parent, path.name, len(raw), 0o600, gid)
        if group is None:
            os.fchmod(descriptor, 0o600)
        else:
            os.fchmod(descriptor, 0o640)
        _file_state(descriptor, parent, path.name, len(raw), mode, gid)
        os.fsync(descriptor)
        os.fsync(parent)
    except BaseException:
        if descriptor >= 0:
            # Nie zostawiaj grupie niepotwierdzonej publikacji po błędzie końcowych kontroli.
            os.fchmod(descriptor, 0o600)
        raise
    finally:
        try:
            if descriptor >= 0:
                os.close(descriptor)
        finally:
            os.close(parent)


def _directory_state(descriptor: int, parent: int, name: str, mode: int, gid: int) -> None:
    info = os.fstat(descriptor)
    linked = os.stat(name, dir_fd=parent, follow_symlinks=False)
    _require(
        _protected(info, directory=True)
        and _protected(linked, directory=True)
        and (info.st_dev, info.st_ino) == (linked.st_dev, linked.st_ino)
        and stat.S_IMODE(info.st_mode) == stat.S_IMODE(linked.st_mode) == mode
        and info.st_gid == linked.st_gid == gid,
        "Podmieniony katalog udostępnienia snapshotu",
    )
    _no_acl(descriptor)


def new_snapshot_directory(path: Path) -> int:
    """Utwórz prywatny katalog root:runtime; wywołujący przejmuje jego otwarty FD."""
    if sys.platform != "linux":
        raise TrialError("Wymagany natywny Linux")
    require_administrator()
    path = _absolute(str(path))
    gid = group_gid(RUNTIME_GROUP)
    _require(gid != group_gid(IPC_GROUP), "Grupy runtime i verifiera muszą być odrębne")
    parent = _open_directory(path.parent)
    descriptor = -1
    try:
        os.mkdir(path.name, 0o700, dir_fd=parent)
        descriptor = os.open(path.name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent)
        _no_acl(descriptor)
        os.fchown(descriptor, 0, gid)
        os.fchmod(descriptor, 0o700)
        _directory_state(descriptor, parent, path.name, 0o700, gid)
        os.fsync(descriptor)
        os.fsync(parent)
        return descriptor
    except BaseException:
        if descriptor >= 0:
            os.close(descriptor)
        raise
    finally:
        os.close(parent)


def publish_snapshot_directory(descriptor: int, path: Path) -> None:
    """Udostępniaj katalog dopiero po zapisaniu pełnego before.json."""
    if sys.platform != "linux":
        raise TrialError("Wymagany natywny Linux")
    require_administrator()
    gid = group_gid(RUNTIME_GROUP)
    _require(gid != group_gid(IPC_GROUP), "Grupy runtime i verifiera muszą być odrębne")
    parent = _open_directory(path.parent)
    try:
        _directory_state(descriptor, parent, path.name, 0o700, gid)
        os.fchmod(descriptor, 0o750)
        _directory_state(descriptor, parent, path.name, 0o750, gid)
        os.fsync(descriptor)
        os.fsync(parent)
    finally:
        os.close(parent)


def validate_manifest(value: dict[str, Any]) -> None:
    _require(type(value) is dict and set(value) == _FIELDS, "Niepełny schemat manifestu próby")
    _require(type(value["schema"]) is int and value["schema"] == 1, "Inna wersja manifestu próby")
    _require(value["scope"] == SCOPE, "Inny zakres manifestu próby")
    for key, pattern in _PATTERNS.items():
        _require(
            type(value[key]) is str and re.fullmatch(pattern, value[key]) is not None,
            "Nieprawidłowe pole manifestu: " + key,
        )
    packages = value["packages"]
    _require(
        type(packages) is dict
        and set(packages) == REQUIRED_PACKAGES
        and all(
            type(version) is str
            and 0 < len(version) <= 256
            and re.fullmatch(r"[0-9A-Za-z.+:~_-]+", version) is not None
            for version in packages.values()
        ),
        "Nieprawidłowa baza pakietów manifestu",
    )
    _absolute(value["source_root"])
    _absolute(value["python_executable"])


def _environment(facts: dict[str, Any], manifest: dict[str, Any]) -> None:
    _require(
        facts.get("observation_complete") is True
        and facts.get("os_id") == "ubuntu"
        and facts.get("os_version") == "24.04"
        and facts.get("architecture") == "x86_64"
        and facts.get("pid1") == "systemd"
        and facts.get("python_executable") == "/usr/bin/python3.12"
        and type(facts.get("python_version")) is str
        and re.fullmatch(r"Python 3\.12\.[0-9]+", facts["python_version"]) is not None
        and facts.get("virtualization") in _VM_TYPES
        and facts.get("container") == "none"
        and facts.get("container_markers") == []
        and facts.get("shared_mounts") == []
        and facts.get("network_disconnected") is True,
        "Środowisko próby jest niebezpieczne lub niekompletne",
    )
    _require(
        all(facts.get(key) == manifest[key] for key in ("machine_id", "boot_id", "packages")),
        "Niezgodna maszyna, boot lub baza pakietów próby",
    )


def _sources(manifest: dict[str, Any]) -> None:
    source = _absolute(manifest["source_root"])
    _require(source == Path(__file__).absolute().parents[2], "Inny korzeń uruchomionych źródeł")
    _protected_directory(source)
    for relative, field in ((_HARNESS, "harness_sha256"), (_CONTRACT, "contract_sha256")):
        _require(
            hashlib.sha256(read_protected(source / relative)).hexdigest() == manifest[field],
            "Zmienione chronione źródła próby",
        )
    _require(sys.executable == manifest["python_executable"], "Inny uruchomiony interpreter próby")
    _protected_interpreter(_absolute(manifest["python_executable"]))


def _approved(args: argparse.Namespace) -> tuple[dict[str, Any], str]:
    raw = read_protected(DEFAULT_APPROVAL)
    result: dict[str, Any] = require_approved_vm(
        revision=args.revision, environment_id=args.environment_id
    )
    _require(
        strict_json(raw) == result["approval"] and read_protected(DEFAULT_APPROVAL) == raw,
        "Zmienione zatwierdzenie VM",
    )
    return result, hashlib.sha256(raw).hexdigest()


def _manifest_path(args: argparse.Namespace) -> Path:
    return Path(getattr(args, "manifest", None) or MANIFEST)


def create_trial(args: argparse.Namespace) -> dict[str, Any]:
    """Po oryginalnej bramce VM opublikuj wyłącznie nowy kontrakt próby."""
    if sys.platform != "linux":
        raise TrialError("Wymagany natywny Linux")
    _require(os.geteuid() == 0, "Utworzenie próby wymaga administratora VM")
    result, approval_sha256 = _approved(args)
    approval = result["approval"]
    source = _absolute(str(args.source_root))
    value = {
        "schema": 1,
        "scope": SCOPE,
        "trial_id": args.trial_id,
        "revision": args.revision,
        "environment_id": args.environment_id,
        **{
            key: approval[key]
            for key in ("machine_id", "boot_id", "vm_image_sha256", "vm_snapshot", "packages")
        },
        "approval_sha256": approval_sha256,
        "source_root": str(source),
        "python_executable": str(args.python_executable),
        "harness_sha256": hashlib.sha256(read_protected(source / _HARNESS)).hexdigest(),
        "contract_sha256": hashlib.sha256(read_protected(source / _CONTRACT)).hexdigest(),
    }
    validate_manifest(value)
    _environment(result["environment"], value)
    _sources(value)
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )
    write_new(_manifest_path(args), raw, 0o640, group=IPC_GROUP)
    return {**value, "_manifest_sha256": hashlib.sha256(raw).hexdigest()}


def require_trial(args: argparse.Namespace, administrator: bool = False) -> dict[str, Any]:
    """Nieuprawniony proces sprawdza fakty i kontrakt bez wywoływania rootguard."""
    if sys.platform != "linux":
        raise TrialError("Wymagany natywny Linux")
    raw = read_protected(_manifest_path(args))
    value = strict_json(raw)
    validate_manifest(value)
    _require(
        value["revision"] == args.revision and value["environment_id"] == args.environment_id,
        "Inna rewizja lub środowisko wybrane w CLI",
    )
    facts: dict[str, Any] = inspect_environment()
    _environment(facts, value)
    _sources(value)
    if administrator:
        result, digest = _approved(args)
        _require(digest == value["approval_sha256"], "Inny hash zatwierdzenia VM")
        _require(
            all(
                result["approval"][key] == value[key]
                for key in (
                    "revision",
                    "environment_id",
                    "machine_id",
                    "boot_id",
                    "vm_image_sha256",
                    "vm_snapshot",
                    "packages",
                )
            ),
            "Zatwierdzenie VM nie odpowiada manifestowi próby",
        )
    return {**value, "_manifest_sha256": hashlib.sha256(raw).hexdigest()}


def binding(trial: dict[str, Any]) -> dict[str, Any]:
    """Dokładne powiązanie raportu z bajtami chronionego manifestu."""
    return {
        **{key: trial[key] for key in _BINDING_FIELDS},
        "manifest_sha256": trial["_manifest_sha256"],
    }

"""Odczytowa granica podpisanego payloadu Freshness, bez kwalifikacji produktu.

Bootstrap i zewnętrzny klucz są uprzednio zaufane. Podpis nie zatwierdza
własnego weryfikatora ani AccountGenesis. Sprawdzamy ponownie dokładne bajty
zainstalowanej aplikacji przed importem jej modułów.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import runpy
import stat
import sys
from pathlib import Path, PurePosixPath
from typing import Any, cast

SCOPE = "cryptohunter.freshness.substrate.v2"
PROFILE = "SIGNED_FRESHNESS_SUBSTRATE"
INSTALLATION = Path("/usr/lib/cryptohunter")
APPROVAL = Path("/etc/cryptohunter/freshness-application")
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
        "deployment/freshness_application.py",
        "deployment/freshness_runtime.py",
        "deployment/freshness_runtime_bootstrap.py",
        "deployment/systemd/cryptohunter-freshness-verifier.service",
        "deployment/postgresql/pg_hba.conf",
        "deployment/postgresql/pg_ident.conf",
        "deployment/postgresql/production-local.conf",
    }
)
# Odrębny profil obejmuje tylko istniejące, zatwierdzone piny usługi.
# AccountGenesis pozostaje obowiązkowy w pełnej kwalifikacji produktu.
REVIEWED_SUBSTRATE_DIGESTS = {
    "bot_core/linux_service_ipc.py": "099e6a3a08717b48e7a01606b9908f56f0eec40aced260d554d7dfc4562b451e",
    "bot_core/linux_service_postgresql.py": "d09d359535e18408d8cee282cb757c4eca8a4e8c422bfce8fa804860ad59f0d1",
    "bot_core/production_signer_ipc.py": "be755e67cb1ea4466c074e4c2c0dbe8e6528552f4a70fd7769a5e876c878f552",
    "bot_core/freshness_verifier_service.py": "df23a3c2fcd1e04877192fe2fe376909a29aa5183a516485fbfdad5a72f9f057",
    "bot_core/freshness_semantic_verifier.py": "64dd105d54a2eabd9be9164846fe642ca0a274b30ca5d0a83735a4436d9f39e1",
}
_SHA = re.compile(r"[0-9a-f]{64}")
_REVISION = re.compile(r"[0-9a-f]{40}")
MAX_FILE = 8 * 1024 * 1024
MAX_TOTAL = 128 * 1024 * 1024
MAX_NODES = 32768


class ApplicationTrustError(RuntimeError):
    """Podpisany payload aplikacji nie został odtworzony."""


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise ApplicationTrustError(reason)


def _path(root: Path, absolute: Path) -> Path:
    return root / absolute.relative_to("/")


def _protected(path: Path, owner: int, boundary: Path) -> os.stat_result:
    _require(path.is_relative_to(boundary), "Ścieżka poza granicą aplikacji")
    current = boundary
    for part in path.relative_to(boundary).parts[:-1]:
        info = current.lstat()
        _require(
            stat.S_ISDIR(info.st_mode) and info.st_uid == owner and not info.st_mode & 0o022,
            "Niechroniony przodek aplikacji",
        )
        current /= part
    info = current.lstat()
    _require(
        stat.S_ISDIR(info.st_mode) and info.st_uid == owner and not info.st_mode & 0o022,
        "Niechroniony przodek aplikacji",
    )
    info = path.lstat()
    _require(
        info.st_uid == owner and not stat.S_ISLNK(info.st_mode) and not info.st_mode & 0o022,
        "Niechroniony obiekt aplikacji",
    )
    return info


def _read(path: Path, owner: int, boundary: Path, limit: int, mode: int = 0o644) -> bytes:
    info = _protected(path, owner, boundary)
    _require(stat.S_IMODE(info.st_mode) == mode, "Niezgodny tryb pliku aplikacji")
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        before = os.fstat(fd)
        _require(
            stat.S_ISREG(before.st_mode)
            and before.st_uid == owner
            and before.st_nlink == 1
            and stat.S_IMODE(before.st_mode) == mode
            and 0 <= before.st_size <= limit
            and (info.st_dev, info.st_ino) == (before.st_dev, before.st_ino),
            "Niechroniony plik aplikacji lub przekroczony rozmiar",
        )
        chunks: list[bytes] = []
        size = 0
        while block := os.read(fd, min(65536, limit + 1 - size)):
            chunks.append(block)
            size += len(block)
            _require(size <= limit, "Przekroczony rozmiar aplikacji")
        after = os.fstat(fd)
        fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
        _require(
            all(getattr(before, field) == getattr(after, field) for field in fields)
            and size == before.st_size,
            "Plik aplikacji zmieniono podczas odczytu",
        )
        return b"".join(chunks)
    finally:
        os.close(fd)


def _json(raw: bytes) -> dict[str, Any]:
    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            _require(key not in result, "Powtórzone pole zatwierdzenia aplikacji")
            result[key] = value
        return result

    result = json.loads(raw, object_pairs_hook=unique)
    _require(type(result) is dict, "Nieprawidłowy dokument zatwierdzenia aplikacji")
    return cast(dict[str, Any], result)


def _helpers() -> dict[str, Any]:
    if __package__:
        from deployment.freshness_runtime import verify_installed_runtime
        from deployment.freshness_runtime_bootstrap import verify_ed25519

        return {
            "verify_ed25519": verify_ed25519,
            "verify_installed_runtime": verify_installed_runtime,
        }
    result: dict[str, Any] = {}
    for filename, symbol in (
        ("freshness_runtime_bootstrap.py", "verify_ed25519"),
        ("freshness_runtime.py", "verify_installed_runtime"),
    ):
        helper = Path(__file__).absolute().with_name(filename)
        info = _protected(helper, 0, Path("/"))
        _require(stat.S_ISREG(info.st_mode) and info.st_nlink == 1, "Niechroniony bootstrap")
        result[symbol] = runpy.run_path(str(helper))[symbol]
    return result


def _runtime_binding(runtime: Any, manifest: dict[str, Any]) -> dict[str, str]:
    """Bind an already verified runtime to the already authenticated application.

    This is only the equality gate. It neither verifies signatures nor turns
    arbitrary manifest data into an authenticated application.
    """
    actual = {
        "environment_id": runtime.environment_id,
        "manifest_sha256": runtime.manifest_sha256,
        "source_revision": runtime.revision,
        "executable": str(runtime.executable),
        "base_executable": str(runtime.base_executable),
        "base_executable_sha256": runtime.base_executable_sha256,
    }
    _require(
        actual == manifest["runtime"] and actual["source_revision"] == manifest["source_revision"],
        "Runtime nie odpowiada podpisanej aplikacji",
    )
    return actual


def verify_installed_application(*, root: Path = Path("/"), owner_uid: int = 0) -> dict[str, Any]:
    """Sprawdź tylko minimalny podpisany payload; nigdy pełny PRODUCTION_LOCAL."""
    try:
        _require(sys.platform == "linux", "Weryfikacja aplikacji wymaga natywnego Linux")
        _require(root.is_absolute() and ".." not in root.parts, "Niekanoniczny root aplikacji")
        directory = _path(root, APPROVAL)
        directory_status = _protected(directory, owner_uid, root)
        _require(
            stat.S_ISDIR(directory_status.st_mode)
            and stat.S_IMODE(directory_status.st_mode) == 0o755
            and {item.name for item in directory.iterdir()}
            == {"manifest.json", "manifest.sig", "authority.pub", "binding.json"},
            "Niepełny lub dodatkowy materiał zatwierdzenia aplikacji",
        )
        helpers = _helpers()
        raw = _read(directory / "manifest.json", owner_uid, root, 4 * 1024 * 1024)
        signature = _read(directory / "manifest.sig", owner_uid, root, 64)
        key = _read(directory / "authority.pub", owner_uid, root, 32)
        _require(
            len(signature) == 64 and len(key) == 32, "Nieprawidłowy materiał podpisu aplikacji"
        )
        helpers["verify_ed25519"](raw, signature, key)
        manifest = _json(raw)
        _require(
            set(manifest) == {"schema_version", "scope", "source_revision", "files", "runtime"}
            and type(manifest["schema_version"]) is int
            and manifest["schema_version"] == 2
            and manifest["scope"] == SCOPE
            and type(manifest["source_revision"]) is str
            and _REVISION.fullmatch(manifest["source_revision"]) is not None,
            "Niezgodny zakres lub rewizja podpisanej aplikacji",
        )
        binding = _json(_read(directory / "binding.json", owner_uid, root, 65536))
        expected_binding = {
            "schema_version": 1,
            "scope": SCOPE,
            "manifest_sha256": hashlib.sha256(raw).hexdigest(),
            "source_revision": manifest["source_revision"],
            "runtime": manifest["runtime"],
        }
        _require(
            type(binding.get("schema_version")) is int and binding == expected_binding,
            "Powiązanie aplikacji nie odpowiada podpisanemu wydaniu",
        )
        files = manifest["files"]
        _require(
            type(files) is list and 0 < len(files) <= 16384, "Nieprawidłowy inwentarz aplikacji"
        )
        inventory: dict[str, dict[str, Any]] = {}
        for entry in files:
            _require(
                type(entry) is dict and set(entry) == {"path", "sha256", "size", "mode"},
                "Nieprawidłowy rekord aplikacji",
            )
            name = entry["path"]
            _require(
                type(name) is str and bool(name) and "\\" not in name and "\x00" not in name,
                "Nieprawidłowa nazwa aplikacji",
            )
            relative = PurePosixPath(name)
            _require(
                not relative.is_absolute()
                and str(relative) == name
                and ".." not in relative.parts
                and len(relative.parts) >= 2
                and relative.parts[0] in {"deployment", "bot_core"}
                and not {"__pycache__", ".git"} & set(relative.parts)
                and not name.endswith((".pyc", ".pyo"))
                and len(relative.parts) <= 32
                and name not in inventory,
                "Nieprawidłowa ścieżka inwentarza aplikacji",
            )
            _require(
                type(entry["sha256"]) is str
                and _SHA.fullmatch(entry["sha256"]) is not None
                and type(entry["size"]) is int
                and 0 <= entry["size"] <= MAX_FILE
                and type(entry["mode"]) is int
                and entry["mode"] in {0o644, 0o755},
                "Nieprawidłowe metadane aplikacji",
            )
            inventory[name] = entry
        _require(REQUIRED_FILES <= set(inventory), "Brak wymaganego pliku aplikacji")
        for name, digest in REVIEWED_SUBSTRATE_DIGESTS.items():
            _require(
                inventory[name]["sha256"] == digest, "Źródło nie odpowiada zatwierdzonym bajtom"
            )
        installation = _path(root, INSTALLATION)
        expected_directories = {"."}
        for name in inventory:
            expected_directories.update(str(parent) for parent in PurePosixPath(name).parents)
        observed: set[str] = set()
        total = 0
        nodes = 0

        def visit(path: Path) -> None:
            nonlocal total, nodes
            nodes += 1
            _require(nodes <= MAX_NODES, "Zbyt duży inwentarz ścieżek aplikacji")
            relative = path.relative_to(installation).as_posix()
            info = _protected(path, owner_uid, root)
            if stat.S_ISDIR(info.st_mode):
                _require(
                    relative in expected_directories and stat.S_IMODE(info.st_mode) == 0o755,
                    "Dodatkowy lub niechroniony katalog aplikacji",
                )
                for child in sorted(path.iterdir()):
                    visit(child)
                return
            _require(
                relative in inventory and relative not in observed,
                "Dodatkowy niepodpisany plik aplikacji",
            )
            entry = inventory[relative]
            content = _read(path, owner_uid, root, MAX_FILE, entry["mode"])
            total += len(content)
            _require(total <= MAX_TOTAL, "Przekroczony rozmiar payloadu aplikacji")
            _require(
                len(content) == entry["size"]
                and hashlib.sha256(content).hexdigest() == entry["sha256"],
                "Niezgodne bajty podpisanej aplikacji",
            )
            observed.add(relative)

        visit(installation)
        _require(observed == set(inventory), "Brak podpisanego pliku aplikacji")
        runtime = helpers["verify_installed_runtime"](root=root, owner_uid=owner_uid)
        actual_runtime = _runtime_binding(runtime, manifest)
        return {
            "scope": SCOPE,
            "qualification_profile": PROFILE,
            "production_qualification": "WITHHELD",
            "source_revision": manifest["source_revision"],
            "manifest_sha256": expected_binding["manifest_sha256"],
            "runtime": actual_runtime,
        }
    except ApplicationTrustError:
        raise
    except (OSError, RuntimeError, ValueError, TypeError, KeyError, RecursionError) as exc:
        raise ApplicationTrustError("Nie można potwierdzić podpisanej aplikacji") from exc

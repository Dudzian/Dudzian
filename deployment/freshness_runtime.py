"""Podpisany, izolowany runtime Freshness oraz publikacja bez nadpisywania.

Odczyt i weryfikacja nie uruchamiają żadnego pliku z runtime. Manifest jest
weryfikowany przez zewnętrzny autorytet przed interpretacją jego kontraktu.
Runtime nie stanowi zatwierdzenia aplikacji ani fizycznego dowodu Stage 9.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import platform
import re
import runpy
import stat
import sys
from dataclasses import dataclass, replace
from email.parser import BytesParser
from pathlib import Path, PurePosixPath
from typing import Any, cast

if __package__:
    from deployment.freshness_runtime_bootstrap import RuntimeTrustError, verify_ed25519
else:
    # Skrypt -I ładuje wyłącznie chroniony sibling, bez cwd i PYTHONPATH.
    helper = Path(__file__).absolute().with_name("freshness_runtime_bootstrap.py")
    for component in (*reversed(helper.parents), helper):
        info = component.lstat()
        if (
            sys.platform != "win32"
            and os.geteuid() == 0
            and (info.st_uid != 0 or info.st_mode & 0o022 or stat.S_ISLNK(info.st_mode))
        ):
            raise SystemExit("Odmowa: niechroniony bootstrap runtime")
    namespace = runpy.run_path(str(helper))
    globals()["RuntimeTrustError"] = namespace["RuntimeTrustError"]
    verify_ed25519 = namespace["verify_ed25519"]

RUNTIME_BASE = Path("/opt/cryptohunter/freshness-runtime")
AUTHORITY = Path("/etc/cryptohunter/freshness-runtime-authority.pub")
BINDING = Path("/etc/cryptohunter/freshness-runtime.json")
MANIFEST = "runtime-manifest.json"
SIGNATURE = "runtime-manifest.sig"
SCOPE = "cryptohunter.freshness.python-runtime.v1"
MAX_MANIFEST = 32 * 1024 * 1024
# Niezależnie przejrzany kontrakt: nigdy nie pochodzi z podpisywanego bundle.
UNIT_TEMPLATE = (
    b"[Unit]\n"
    b"Description=CryptoHunter isolated AccountGenesis freshness semantic "
    b"verifier\n"
    b"Requires=postgresql.service\n"
    b"After=postgresql.service\n"
    b"StartLimitIntervalSec=60s\n"
    b"StartLimitBurst=5\n"
    b"\n"
    b"[Service]\n"
    b"Type=simple\n"
    b"User=os_freshness_crypto_verifier\n"
    b"Group=freshness_verifier_ipc\n"
    b"ExecStartPre=/usr/bin/python3 -I -s -B "
    b"/usr/lib/cryptohunter/deployment/freshness_runtime_bootstrap.py verify-installed\n"
    b"ExecStart=@FRESHNESS_RUNTIME_PYTHON@ -I -s -B "
    b"/usr/lib/cryptohunter/deployment/freshness_verifier_entrypoint.py\n"
    b"Environment=PYTHONNOUSERSITE=1\n"
    b"WorkingDirectory=/usr/lib/cryptohunter\n"
    b"RuntimeDirectory=cryptohunter/freshness\n"
    b"RuntimeDirectoryMode=0750\n"
    b"UMask=0027\n"
    b"Restart=on-failure\n"
    b"RestartSec=5s\n"
    b"NoNewPrivileges=yes\n"
    b"PrivateTmp=yes\n"
    b"ProtectSystem=strict\n"
    b"ProtectHome=yes\n"
    b"ProtectKernelTunables=yes\n"
    b"ProtectKernelModules=yes\n"
    b"ProtectKernelLogs=yes\n"
    b"ProtectControlGroups=yes\n"
    b"RestrictSUIDSGID=yes\n"
    b"LockPersonality=yes\n"
    b"RestrictNamespaces=yes\n"
    b"RestrictRealtime=yes\n"
    b"MemoryDenyWriteExecute=yes\n"
    b"CapabilityBoundingSet=\n"
    b"AmbientCapabilities=\n"
    b"RestrictAddressFamilies=AF_UNIX\n"
    b"SystemCallArchitectures=native\n"
    b"\n"
    b"[Install]\n"
    b"WantedBy=multi-user.target\n"
)
_SHA = re.compile(r"[0-9a-f]{64}")
_REV = re.compile(r"[0-9a-f]{40}")
_ENV = re.compile(r"[a-z0-9][a-z0-9._-]{0,95}")
_METADATA = frozenset({MANIFEST, SIGNATURE})


class RuntimeError(RuntimeTrustError):
    """Odmowa runtime; przyczyna nie ujawnia danych bibliotek ani sekretów."""


@dataclass(frozen=True)
class RuntimeConfig:
    runtime: Path
    manifest: Path
    signature: Path
    trusted_key: Path
    expected_environment_id: str
    expected_revision: str
    owner_uid: int = 0
    root: Path = Path("/")
    check_platform: bool = True


@dataclass(frozen=True)
class VerifiedRuntime:
    environment_id: str
    revision: str
    manifest_sha256: str
    executable: Path
    base_executable: Path
    executable_sha256: str
    base_executable_sha256: str
    packages: dict[str, str]
    manifest: dict[str, Any]


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise RuntimeError(reason)


def render_runtime_unit(executable: str) -> bytes:
    """Materializuj wyłącznie kanoniczną ścieżkę zatwierdzonego wydania."""
    _require(
        type(executable) is str
        and re.fullmatch(
            r"/opt/cryptohunter/freshness-runtime/releases/"
            r"[a-z0-9][a-z0-9._-]{0,95}/bin/python3\.12",
            executable,
        )
        is not None,
        "Niekanoniczna ścieżka interpretera jednostki runtime",
    )
    return UNIT_TEMPLATE.replace(b"@FRESHNESS_RUNTIME_PYTHON@", executable.encode("ascii"))


def validate_runtime_unit(content: bytes, executable: str) -> bytes:
    """Sprawdź wszystkie bajty kontraktu, bez uproszczonego parsera systemd."""
    expected = render_runtime_unit(executable)
    _require(
        type(content) is bytes and content == expected,
        "Jednostka nie odpowiada pełnemu zatwierdzonemu kontraktowi runtime",
    )
    return expected


def _json(raw: bytes) -> dict[str, Any]:
    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            _require(key not in result, "Powtórzone pole manifestu runtime")
            result[key] = value
        return result

    value = json.loads(raw, object_pairs_hook=unique)
    _require(type(value) is dict, "Nieprawidłowy dokument runtime")
    return cast(dict[str, Any], value)


def _relative(value: str) -> bool:
    if type(value) is not str or not value or "\\" in value or "\x00" in value:
        return False
    path = PurePosixPath(value)
    return not path.is_absolute() and str(path) == value and ".." not in path.parts


def _absolute(value: str) -> bool:
    if type(value) is not str or not value.startswith("/"):
        return False
    return _relative(value[1:])


def _mapped(root: Path, path: Path) -> Path:
    _require(path.is_absolute() and ".." not in path.parts, "Nieprawidłowa ścieżka runtime")
    return root / path.relative_to("/")


def _status(path: Path, owner: int, *, symlink: bool = False) -> os.stat_result:
    info = path.lstat()
    _require(
        info.st_uid == owner and (stat.S_ISLNK(info.st_mode) or not info.st_mode & 0o022),
        "Niechroniony obiekt runtime",
    )
    _require(symlink or not stat.S_ISLNK(info.st_mode), "Niezatwierdzony symlink runtime")
    return info


def _parents(path: Path, owner: int, boundary: Path) -> None:
    _require(
        path.is_absolute()
        and boundary.is_absolute()
        and ".." not in path.parts
        and ".." not in boundary.parts,
        "Wymagana ścieżka absolutna bez przejść rodzica",
    )
    _require(path.is_relative_to(boundary), "Ścieżka poza granicą runtime")
    relative = path.relative_to(boundary)
    current = boundary
    for part in relative.parts[:-1]:
        info = _status(current, owner)
        _require(stat.S_ISDIR(info.st_mode), "Przodek runtime nie jest katalogiem")
        current /= part
    info = _status(current, owner)
    _require(stat.S_ISDIR(info.st_mode), "Przodek runtime nie jest katalogiem")


def _read(path: Path, owner: int, boundary: Path, limit: int = MAX_MANIFEST) -> bytes:
    _parents(path, owner, boundary)
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        before = os.fstat(fd)
        _require(
            stat.S_ISREG(before.st_mode)
            and before.st_nlink == 1
            and before.st_uid == owner
            and not before.st_mode & 0o022
            and 0 <= before.st_size <= limit,
            "Niechroniony plik runtime lub przekroczony rozmiar",
        )
        chunks: list[bytes] = []
        total = 0
        while block := os.read(fd, min(65536, limit + 1 - total)):
            chunks.append(block)
            total += len(block)
            _require(total <= limit, "Przekroczony rozmiar runtime")
        after = os.fstat(fd)
        _require(
            (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
            == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns),
            "Plik runtime zmieniono podczas obserwacji",
        )
        return b"".join(chunks)
    finally:
        os.close(fd)


def _record(path: Path) -> dict[str, Any]:
    info = path.lstat()
    mode = stat.S_IMODE(info.st_mode)
    if stat.S_ISLNK(info.st_mode):
        return {"kind": "symlink", "mode": mode, "target": os.readlink(path)}
    if stat.S_ISDIR(info.st_mode):
        return {"kind": "directory", "mode": mode}
    _require(stat.S_ISREG(info.st_mode) and info.st_nlink == 1, "Niedozwolony typ pliku runtime")
    _require(info.st_size <= 512 * 1024 * 1024, "Przekroczony rozmiar pliku runtime")
    return {
        "kind": "file",
        "mode": mode,
        "size": info.st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _tree(root: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}

    def visit(directory: Path) -> None:
        for item in sorted(directory.iterdir()):
            relative = item.relative_to(root).as_posix()
            if relative in _METADATA:
                continue
            _require(_relative(relative), "Nieprawidłowa nazwa pliku runtime")
            record = _record(item)
            result[relative] = record
            if record["kind"] == "directory":
                visit(item)

    visit(root)
    return result


def collect_runtime_manifest(
    runtime: Path,
    *,
    environment_id: str,
    release_revision: str,
    interpreter: dict[str, str],
    packages: dict[str, str],
    external_paths: tuple[Path, ...],
    lock_sha256: str,
) -> dict[str, Any]:
    """Zbierz kandydat do niezależnego przeglądu; ta funkcja niczego nie zatwierdza."""
    return {
        "schema": 1,
        "scope": SCOPE,
        "environment_id": environment_id,
        "release_revision": release_revision,
        "target": {"os": "ubuntu", "version": "24.04", "architecture": "amd64"},
        "installation_path": str(RUNTIME_BASE / "releases" / environment_id),
        "interpreter": interpreter,
        "packages": packages,
        "files": _tree(runtime),
        "external_files": {str(path): _record(path) for path in sorted(set(external_paths))},
        "lock_sha256": lock_sha256,
    }


def _schema(data: dict[str, Any], config: RuntimeConfig) -> None:
    _require(
        set(data)
        == {
            "schema",
            "scope",
            "environment_id",
            "release_revision",
            "target",
            "installation_path",
            "interpreter",
            "packages",
            "files",
            "external_files",
            "lock_sha256",
        },
        "Nieprawidłowy schemat manifestu runtime",
    )
    _require(
        type(data["schema"]) is int and data["schema"] == 1 and data["scope"] == SCOPE,
        "Nieobsługiwany kontrakt runtime",
    )
    _require(
        _ENV.fullmatch(config.expected_environment_id) is not None
        and _REV.fullmatch(config.expected_revision) is not None,
        "Nieprawidłowa oczekiwana tożsamość runtime",
    )
    _require(
        data["environment_id"] == config.expected_environment_id
        and data["release_revision"] == config.expected_revision,
        "Niezgodna tożsamość lub rewizja runtime",
    )
    _require(
        data["target"] == {"os": "ubuntu", "version": "24.04", "architecture": "amd64"},
        "Niezgodna platforma manifestu runtime",
    )
    expected_path = RUNTIME_BASE / "releases" / config.expected_environment_id
    _require(data["installation_path"] == str(expected_path), "Niezgodna ścieżka wydania runtime")
    interpreter = data["interpreter"]
    _require(
        type(interpreter) is dict
        and set(interpreter)
        == {
            "implementation",
            "version",
            "abi",
            "executable",
            "base_executable",
            "provenance",
        },
        "Nieprawidłowy kontrakt interpretera",
    )
    _require(
        interpreter["implementation"] == "CPython"
        and type(interpreter["version"]) is str
        and re.fullmatch(r"3\.12\.[0-9]+", interpreter["version"]) is not None
        and interpreter["abi"] == "cpython-312-x86_64-linux-gnu"
        and interpreter["executable"] == "bin/python3.12"
        and interpreter["base_executable"] == "/usr/bin/python3.12"
        and type(interpreter["provenance"]) is str
        and interpreter["provenance"].startswith("Ubuntu dpkg python3.12 "),
        "Niezgodny interpreter, ABI lub pochodzenie runtime",
    )
    packages = data["packages"]
    _require(
        type(packages) is dict
        and {"cryptography", "psycopg"} <= set(packages)
        and all(type(k) is str and type(v) is str for k, v in packages.items()),
        "Brak wymaganych bibliotek runtime",
    )
    _require(
        re.fullmatch(r"50\.0\.[1-9][0-9]*", packages["cryptography"]) is not None
        and re.fullmatch(r"3\.(?:[2-9]|[1-9][0-9]+)\.[0-9]+", packages["psycopg"]) is not None,
        "Niezatwierdzony zakres bibliotek runtime",
    )
    _require(
        type(data["files"]) is dict
        and type(data["external_files"]) is dict
        and _SHA.fullmatch(data["lock_sha256"]) is not None,
        "Nieprawidłowa lista integralności runtime",
    )
    for entries, absolute in ((data["files"], False), (data["external_files"], True)):
        for name, record in entries.items():
            _require(
                (_absolute(name) if absolute else _relative(name))
                and type(record) is dict
                and type(record.get("mode")) is int,
                "Nieprawidłowy wpis integralności runtime",
            )
            kind = record.get("kind")
            fields = {"kind", "mode"}
            if kind == "file":
                fields |= {"sha256", "size"}
                _require(
                    type(record.get("sha256")) is str
                    and _SHA.fullmatch(record["sha256"]) is not None
                    and type(record.get("size")) is int
                    and 0 <= record["size"] <= 512 * 1024 * 1024,
                    "Nieprawidłowy hash pliku runtime",
                )
            elif kind == "symlink":
                fields.add("target")
                _require(
                    type(record.get("target")) is str
                    and bool(record["target"])
                    and "\x00" not in record["target"],
                    "Nieprawidłowy symlink runtime",
                )
            else:
                _require(kind == "directory", "Niedozwolony obiekt runtime")
            _require(
                set(record) == fields
                and 0 <= record["mode"] <= 0o777
                and (kind == "symlink" or not record["mode"] & 0o022),
                "Niezgodne uprawnienia runtime",
            )
    files = data["files"]
    _require(not _METADATA & set(files), "Materiały podpisu w inwentarzu runtime")
    _require(
        {"bin/python3.12", "pyvenv.cfg", "requirements.lock", "wheelhouse"} <= set(files),
        "Niekompletny runtime lub wheelhouse",
    )
    _require(
        files["wheelhouse"]["kind"] == "directory"
        and any(name.startswith("wheelhouse/") and name.endswith(".whl") for name in files),
        "Niekompletny wheelhouse runtime",
    )
    _require(
        not any(
            name.endswith((".pth", ".pyc"))
            or PurePosixPath(name).name in {"sitecustomize.py", "usercustomize.py"}
            or "__pycache__" in PurePosixPath(name).parts
            for name in files
        ),
        "Niedozwolony import hook lub cache runtime",
    )
    _require(
        files["requirements.lock"].get("sha256") == data["lock_sha256"], "Niezgodny lock runtime"
    )
    _require(
        interpreter["base_executable"] in data["external_files"],
        "Brak niezależnej tożsamości bazowego interpretera",
    )


def _verify_dependency_closure(data: dict[str, Any], config: RuntimeConfig) -> None:
    """Zwiąż lock, wszystkie wheels, installed metadata oraz bazowe stdlib."""
    lock = _read(config.runtime / "requirements.lock", config.owner_uid, config.runtime).decode(
        "ascii"
    )
    requirements: dict[str, tuple[str, str]] = {}
    for line in lock.replace("\\\n", " ").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        match = re.fullmatch(
            r"([A-Za-z0-9_.-]+)==([0-9.]+) --hash=sha256:([0-9a-f]{64})", line.strip()
        )
        _require(match is not None, "Lock zawiera niezatwierdzoną dyrektywę")
        assert match is not None
        name = match[1].lower().replace("_", "-")
        _require(name not in requirements, "Powtórzona zależność lock")
        requirements[name] = (match[2], match[3])
    _require(
        {name: value[0] for name, value in requirements.items()} == data["packages"],
        "Lock i zatwierdzony zestaw bibliotek są różne",
    )
    wheels: dict[str, tuple[str, str]] = {}
    installed: dict[str, str] = {}
    for name, record in data["files"].items():
        if name.startswith("wheelhouse/"):
            _require(
                record["kind"] == "file"
                and name.endswith(".whl")
                and len(PurePosixPath(name).parts) == 2,
                "Dodatkowy artefakt wheelhouse",
            )
            wheel_parts = PurePosixPath(name).name.split("-")
            _require(len(wheel_parts) >= 5, "Nieprawidłowa nazwa wheel")
            package = wheel_parts[0].lower().replace("_", "-")
            _require(package not in wheels, "Powtórzony wheel runtime")
            wheels[package] = (wheel_parts[1], record["sha256"])
        if name.endswith(".dist-info/METADATA"):
            _require(
                name.startswith("lib/python3.12/site-packages/") and record["kind"] == "file",
                "Obce metadata pakietu runtime",
            )
            metadata = BytesParser().parsebytes(
                _read(config.runtime / name, config.owner_uid, config.runtime), headersonly=True
            )
            package = str(metadata.get("Name", "")).lower().replace("_", "-")
            version = str(metadata.get("Version", ""))
            _require(bool(package) and package not in installed, "Powtórzony pakiet runtime")
            installed[package] = version
    _require(wheels == requirements, "Niekompletny lub dodatkowy wheelhouse")
    _require(installed == data["packages"], "Niekompletny lub dodatkowy zainstalowany pakiet")
    stdlib_name = "/usr/lib/python3.12"
    stdlib = _mapped(config.root, Path(stdlib_name))
    _require(
        data["external_files"].get(stdlib_name, {}).get("kind") == "directory",
        "Brak zatwierdzonego bazowego stdlib",
    )
    actual = {
        str(Path(stdlib_name) / name): value
        for name, value in _tree(stdlib).items()
        if not {"site-packages", "dist-packages"} & set(PurePosixPath(name).parts)
    }
    expected = {
        name: value
        for name, value in data["external_files"].items()
        if name.startswith(stdlib_name + "/")
    }
    _require(actual == expected, "Niezgodny kompletny inwentarz bazowego stdlib")
    # Nieobecny zip nadal figuruje w sys.path CPython przed katalogiem stdlib.
    # Jego późniejsze dodanie nie może wprowadzić obcego kodu przed entrypointem.
    for archive in ("/usr/lib/python312.zip", "/usr/lib64/python312.zip"):
        path = _mapped(config.root, Path(archive))
        if path.exists() or path.is_symlink():
            _require(archive in data["external_files"], "Dodatkowy bazowy stdlib zip")
    _require(
        data["external_files"].get("/etc/ld.so.cache", {}).get("kind") == "file",
        "Brak zatwierdzonego cache natywnego loadera",
    )
    preload = _mapped(config.root, Path("/etc/ld.so.preload"))
    _require(
        not preload.exists() and not preload.is_symlink(), "Niedozwolony preload natywnego loadera"
    )
    native_names = {PurePosixPath(name).name for name in (*data["files"], *data["external_files"])}
    for library in ("libpq", "libssl", "libcrypto"):
        _require(
            any(name.startswith(library) and ".so" in name for name in native_names),
            "Brak zatwierdzonej zależności natywnej",
        )


def _boundaries(config: RuntimeConfig, path: Path) -> Path:
    # Testy używają jawnego root i UID; produkcyjne CLI nie udostępnia osłabień.
    return config.root if path.is_relative_to(config.root) else Path("/")


def _verify_object(
    path: Path, expected: dict[str, Any], config: RuntimeConfig, boundary: Path
) -> None:
    _parents(path, config.owner_uid, boundary)
    info = _status(path, config.owner_uid, symlink=expected["kind"] == "symlink")
    _require(stat.S_IMODE(info.st_mode) == expected["mode"], "Niezgodne uprawnienia runtime")
    kind = expected["kind"]
    if kind == "file":
        raw = _read(path, config.owner_uid, boundary, 512 * 1024 * 1024)
        _require(
            len(raw) == expected["size"] and hashlib.sha256(raw).hexdigest() == expected["sha256"],
            "Niezgodna integralność pliku runtime",
        )
    elif kind == "directory":
        _require(stat.S_ISDIR(info.st_mode), "Niezgodny typ katalogu runtime")
    else:
        _require(
            stat.S_ISLNK(info.st_mode) and os.readlink(path) == expected["target"],
            "Niezgodny symlink runtime",
        )


def _resolved(name: str, data: dict[str, Any], config: RuntimeConfig) -> Path:
    current = PurePosixPath(data["installation_path"]) / name
    for _ in range(32):
        text = str(current)
        installation = data["installation_path"] + "/"
        internal = text.startswith(installation)
        key = text[len(installation) :] if internal else text
        record = (data["files"] if internal else data["external_files"]).get(key)
        _require(type(record) is dict, "Symlink runtime wskazuje poza zatwierdzone materiały")
        if record["kind"] != "symlink":
            _require(record["kind"] in {"file", "directory"}, "Niedozwolony cel symlinka runtime")
            return Path(text)
        target = PurePosixPath(record["target"])
        if not target.is_absolute():
            target = current.parent / target
        normalized = os.path.normpath(str(target))
        _require(normalized.startswith("/"), "Nieprawidłowy cel symlinka runtime")
        current = PurePosixPath(normalized)
    raise RuntimeError("Cykl symlinków runtime")


def verify_runtime(config: RuntimeConfig) -> VerifiedRuntime:
    """Pełna odczytowa weryfikacja przed jakimkolwiek wykonaniem runtime."""
    try:
        _require(
            all(
                path.is_absolute() and ".." not in path.parts
                for path in (
                    config.runtime,
                    config.manifest,
                    config.signature,
                    config.trusted_key,
                    config.root,
                )
            ),
            "Nieprawidłowa ścieżka wejściowa runtime",
        )
        _require(
            not config.trusted_key.is_relative_to(config.runtime),
            "Klucz autorytetu pochodzi z runtime",
        )
        raw = _read(config.manifest, config.owner_uid, _boundaries(config, config.manifest))
        signature = _read(
            config.signature, config.owner_uid, _boundaries(config, config.signature), 64
        )
        key = _read(
            config.trusted_key, config.owner_uid, _boundaries(config, config.trusted_key), 32
        )
        verify_ed25519(raw, signature, key)
        data = _json(raw)
        _schema(data, config)
        if config.check_platform:
            release = platform.freedesktop_os_release()
            _require(
                release.get("ID") == "ubuntu"
                and release.get("VERSION_ID") == "24.04"
                and platform.machine() == "x86_64"
                and sys.platform == "linux",
                "Runtime wymaga Ubuntu 24.04 amd64",
            )
        _parents(config.runtime, config.owner_uid, _boundaries(config, config.runtime))
        root_info = _status(config.runtime, config.owner_uid)
        _require(
            stat.S_ISDIR(root_info.st_mode) and stat.S_IMODE(root_info.st_mode) == 0o755,
            "Runtime nie jest katalogiem o zatwierdzonych uprawnieniach",
        )
        # Lista nazw nie podąża za symlinkami; każdy obiekt ma osobną kontrolę.
        observed = _tree(config.runtime)
        _require(observed == data["files"], "Niepełny lub dodatkowy plik runtime")
        for name, record in data["files"].items():
            _verify_object(config.runtime / name, record, config, config.runtime)
        for name, record in data["external_files"].items():
            actual = _mapped(config.root, Path(name))
            _verify_object(actual, record, config, config.root)
        for name, record in data["files"].items():
            if record["kind"] == "symlink":
                _resolved(name, data, config)
        for name, record in data["external_files"].items():
            if record["kind"] == "symlink":
                _resolved(name, data, config)
        _verify_dependency_closure(data, config)
        cfg = _read(config.runtime / "pyvenv.cfg", config.owner_uid, config.runtime, 65536).decode(
            "utf-8"
        )
        settings: dict[str, str] = {}
        for line in cfg.splitlines():
            key_name, separator, value = line.partition("=")
            _require(
                bool(separator) and key_name.strip() not in settings, "Nieprawidłowy pyvenv.cfg"
            )
            settings[key_name.strip()] = value.strip()
        _require(
            settings.get("include-system-site-packages") == "false"
            and settings.get("version") == data["interpreter"]["version"]
            and settings.get("home") == "/usr/bin",
            "Niezgodna izolacja pyvenv.cfg",
        )
        base = _resolved(data["interpreter"]["executable"], data, config)
        _require(
            str(base) == data["interpreter"]["base_executable"], "Niezgodny bazowy interpreter"
        )
        base_record = data["external_files"][str(base)]
        _require(base_record["kind"] == "file", "Bazowy interpreter nie jest zwykłym plikiem")
        for metadata_name in _METADATA:
            path = config.runtime / metadata_name
            if path.exists() or path.is_symlink():
                expected_bytes = raw if metadata_name == MANIFEST else signature
                _require(
                    _read(path, config.owner_uid, config.runtime) == expected_bytes,
                    "Niezgodne materiały podpisu w wydaniu",
                )
        return VerifiedRuntime(
            data["environment_id"],
            data["release_revision"],
            hashlib.sha256(raw).hexdigest(),
            Path(data["installation_path"]) / data["interpreter"]["executable"],
            base,
            base_record["sha256"],
            base_record["sha256"],
            data["packages"],
            data,
        )
    except RuntimeTrustError:
        raise
    except (OSError, ValueError, TypeError, KeyError, UnicodeError, RecursionError) as exc:
        raise RuntimeError("Nie można ustanowić integralności runtime") from exc


def verify_installed_runtime(
    *,
    binding: Path = BINDING,
    root: Path = Path("/"),
    owner_uid: int = 0,
    check_platform: bool = True,
) -> VerifiedRuntime:
    """Powiąż chronioną konfigurację instalacji z niezależnie podpisanym wydaniem."""
    try:
        data = _json(_read(_mapped(root, binding), owner_uid, root, 65536))
        _require(
            set(data)
            == {
                "schema",
                "environment_id",
                "manifest_sha256",
                "source_revision",
                "executable",
                "base_executable",
                "base_executable_sha256",
            }
            and type(data["schema"]) is int
            and data["schema"] == 1,
            "Nieprawidłowe powiązanie runtime",
        )
        _require(
            type(data["environment_id"]) is str
            and _ENV.fullmatch(data["environment_id"]) is not None,
            "Nieprawidłowa tożsamość powiązania runtime",
        )
        base = _mapped(root, RUNTIME_BASE)
        _require(
            not (base / (data["environment_id"] + ".journal")).exists()
            and not (base / "staging" / data["environment_id"]).exists(),
            "Niedokończona publikacja zainstalowanego runtime",
        )
        runtime = _mapped(root, RUNTIME_BASE / "releases" / data["environment_id"])
        verified = verify_runtime(
            RuntimeConfig(
                runtime,
                runtime / MANIFEST,
                runtime / SIGNATURE,
                _mapped(root, AUTHORITY),
                data["environment_id"],
                data["source_revision"],
                owner_uid,
                root,
                check_platform,
            )
        )
        _require(
            data["manifest_sha256"] == verified.manifest_sha256
            and data["executable"] == str(verified.executable)
            and data["base_executable"] == str(verified.base_executable)
            and data["base_executable_sha256"] == verified.base_executable_sha256,
            "Powiązanie runtime nie odpowiada podpisanemu wydaniu",
        )
        return verified
    except RuntimeTrustError:
        raise
    except (OSError, ValueError, TypeError, KeyError) as exc:
        raise RuntimeError("Nie można zweryfikować zainstalowanego runtime") from exc


def _fsync_directory(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _write_new(path: Path, raw: bytes, mode: int) -> None:
    if sys.platform == "win32":
        raise RuntimeError("Publikacja runtime wymaga natywnego Linux")
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, mode)
    try:
        cursor = 0
        while cursor < len(raw):
            cursor += os.write(fd, raw[cursor:])
        os.fchmod(fd, mode)
        os.fsync(fd)
    finally:
        os.close(fd)


def publish_runtime(config: RuntimeConfig) -> VerifiedRuntime:
    """Osobna ceremonia publikacji; nigdy nie zastępuje istniejącego wydania.

    Journal i staging pozostają po awarii. Recovery wymaga ręcznego przeglądu;
    nie usuwamy starych wydań ani nie kontynuujemy częściowej publikacji.
    """
    verified = verify_runtime(config)
    base = _mapped(config.root, RUNTIME_BASE)
    _parents(base, config.owner_uid, config.root)
    _status(base.parent, config.owner_uid)
    # Host docelowy ma wcześniej przygotowany root katalogu runtime.
    _require(base.exists() and not base.is_symlink(), "Brak przygotowanego katalogu runtime")
    _status(base, config.owner_uid)
    destination = _mapped(config.root, Path(verified.manifest["installation_path"]))
    releases = base / "releases"
    staging_root = base / "staging"
    for directory in (releases, staging_root):
        if not directory.exists():
            directory.mkdir(mode=0o755)
            directory.chmod(0o755)
            _fsync_directory(base)
        _status(directory, config.owner_uid)
    journal = base / f"{verified.environment_id}.journal"
    staging = staging_root / verified.environment_id
    if destination.exists():
        _require(not journal.exists() and not staging.exists(), "Niedokończona publikacja runtime")
        installed = verify_runtime(
            replace(
                config,
                runtime=destination,
                manifest=destination / MANIFEST,
                signature=destination / SIGNATURE,
            )
        )
        _require(
            installed.manifest_sha256 == verified.manifest_sha256,
            "Istniejące wydanie runtime ma inną tożsamość",
        )
        return installed
    _require(not journal.exists() and not staging.exists(), "Niedokończona publikacja runtime")
    _write_new(
        journal,
        json.dumps(
            {
                "schema": 1,
                "state": "publishing",
                "environment_id": verified.environment_id,
                "manifest_sha256": verified.manifest_sha256,
            }
        ).encode(),
        0o600,
    )
    _fsync_directory(base)
    staging.mkdir(mode=0o755)
    staging.chmod(0o755)
    _fsync_directory(staging_root)
    for name, record in sorted(
        verified.manifest["files"].items(),
        key=lambda item: (len(PurePosixPath(item[0]).parts), item[0]),
    ):
        destination_file = staging / name
        if record["kind"] == "directory":
            destination_file.mkdir(mode=record["mode"])
            destination_file.chmod(record["mode"])
        elif record["kind"] == "symlink":
            destination_file.symlink_to(record["target"])
        else:
            source = config.runtime / name
            _write_new(
                destination_file,
                _read(source, config.owner_uid, config.runtime, 512 * 1024 * 1024),
                record["mode"],
            )
    _write_new(
        staging / MANIFEST,
        _read(config.manifest, config.owner_uid, _boundaries(config, config.manifest)),
        0o644,
    )
    _write_new(
        staging / SIGNATURE,
        _read(config.signature, config.owner_uid, _boundaries(config, config.signature), 64),
        0o644,
    )
    verify_runtime(
        replace(config, runtime=staging, manifest=staging / MANIFEST, signature=staging / SIGNATURE)
    )
    for directory in [
        staging,
        *(
            staging / name
            for name, record in verified.manifest["files"].items()
            if record["kind"] == "directory"
        ),
    ]:
        _fsync_directory(directory)
    # renameat2(RENAME_NOREPLACE) zabrania zastąpienia także w wyścigu z root.
    libc = ctypes.CDLL(None, use_errno=True)
    rename = libc.renameat2
    rename.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    rename.restype = ctypes.c_int
    _require(
        rename(-100, os.fsencode(staging), -100, os.fsencode(destination), 1) == 0,
        "Publikacja runtime nie powiodła się; wymagany ręczny recovery",
    )
    _fsync_directory(releases)
    _fsync_directory(staging_root)
    installed = verify_runtime(
        replace(
            config,
            runtime=destination,
            manifest=destination / MANIFEST,
            signature=destination / SIGNATURE,
        )
    )
    journal.unlink()
    _fsync_directory(base)
    return installed


def main() -> int:
    parser = argparse.ArgumentParser(description="Weryfikacja lub publikacja podpisanego runtime")
    parser.add_argument("operation", choices=("verify", "publish"))
    for argument in ("runtime", "manifest", "signature", "trusted-key"):
        parser.add_argument(f"--{argument}", required=True, type=Path)
    parser.add_argument("--environment-id", required=True)
    parser.add_argument("--revision", required=True)
    args = parser.parse_args()
    try:
        config = RuntimeConfig(
            args.runtime,
            args.manifest,
            args.signature,
            args.trusted_key,
            args.environment_id,
            args.revision,
        )
        result = publish_runtime(config) if args.operation == "publish" else verify_runtime(config)
    except (RuntimeTrustError, OSError, ValueError):
        print("Odmowa: runtime nie spełnia granicy zaufania")
        return 1
    print(
        json.dumps(
            {"environment_id": result.environment_id, "manifest_sha256": result.manifest_sha256},
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

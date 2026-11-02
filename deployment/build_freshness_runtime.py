"""Odtwarzalny builder Freshness na zaufanym Ubuntu 24.04, bez instalacji globalnej.

Wynik jest kandydatem do niezależnego podpisania, nigdy własnym zatwierdzeniem.
Pobieranie wheelhouse to oddzielny krok; właściwy build działa bez sieci.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import runpy
import shutil
import stat
import subprocess
import sys
import urllib.request
import venv
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any, cast

HERE = Path(__file__).resolve().parent
CATALOG = HERE / "freshness-runtime-wheels.json"
LOCK = HERE / "requirements-freshness-runtime.lock"
BASE_PYTHON = Path("/usr/bin/python3.12")
VERSIONS = {
    "cffi": "2.1.1",
    "cryptography": "50.0.2",
    "psycopg": "3.3.6",
    "psycopg-binary": "3.3.6",
    "pycparser": "3.12",
    "typing-extensions": "4.16.0",
}
_ENV = {
    "PATH": "/usr/bin:/bin",
    "HOME": "/nonexistent",
    "LC_ALL": "C",
    "PYTHONNOUSERSITE": "1",
    "PYTHONDONTWRITEBYTECODE": "1",
    "PIP_CONFIG_FILE": "/dev/null",
    "PIP_DISABLE_PIP_VERSION_CHECK": "1",
    "PSYCOPG_IMPL": "binary",
}
_PROBE = r"""
import importlib.metadata as metadata, json, pathlib, site, ssl, sys, sysconfig
import cryptography, psycopg, psycopg.pq
from cryptography.hazmat.backends.openssl.backend import backend
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
key = Ed25519PrivateKey.generate()
message = b"cryptohunter-runtime-build-probe"
key.public_key().verify(key.sign(message), message)
root = pathlib.Path(sys.prefix).resolve()
assert sys.flags.isolated and sys.flags.no_user_site and sys.flags.dont_write_bytecode
assert site.ENABLE_USER_SITE is False and sys.prefix != sys.base_prefix
assert psycopg.pq.__impl__ == "binary" and psycopg.pq.version() >= 160000
origins = {}
for module in (cryptography, psycopg, psycopg.pq):
    origin = pathlib.Path(module.__file__).resolve()
    assert origin.is_relative_to(root)
    origins[module.__name__] = str(origin.relative_to(root))
packages = {d.metadata["Name"].lower().replace("_", "-"): d.version
            for d in metadata.distributions()}
print(json.dumps({"packages": packages, "origins": origins,
 "abi": sysconfig.get_config_var("SOABI"), "python": sys.version.split()[0],
 "openssl": ssl.OPENSSL_VERSION, "cryptography_openssl": backend.openssl_version_text(),
 "libpq": psycopg.pq.version(), "libpq_build": psycopg.pq.__build_version__,
 "base_prefix": sys.base_prefix}, sort_keys=True))
"""


class BuildError(ValueError):
    """Kandydat nie spełnia zamrożonego kontraktu budowania."""


def _digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _catalog(path: Path = CATALOG) -> list[dict[str, Any]]:
    data = json.loads(path.read_text())
    if (
        type(data) is not dict
        or set(data) != {"schema", "target", "wheels"}
        or data["schema"] != "cryptohunter.freshness.wheelhouse.v1"
        or data["target"]
        != {"os": "ubuntu", "version": "24.04", "architecture": "amd64", "python": "3.12"}
        or type(data["wheels"]) is not list
    ):
        raise BuildError("Nieprawidłowy katalog wheelhouse")
    expected = {**VERSIONS, "pip": "26.2.1"}
    found = {}
    for wheel in data["wheels"]:
        if (
            type(wheel) is not dict
            or set(wheel) != {"name", "version", "filename", "sha256", "url", "size", "role"}
            or wheel["name"] not in expected
            or wheel["name"] in found
            or wheel["version"] != expected[wheel["name"]]
            or wheel["role"] != ("build-tool" if wheel["name"] == "pip" else "runtime")
            or type(wheel["filename"]) is not str
            or PurePosixPath(wheel["filename"]).name != wheel["filename"]
            or not wheel["filename"].endswith(".whl")
            or type(wheel["sha256"]) is not str
            or not re.fullmatch("[0-9a-f]{64}", wheel["sha256"])
            or type(wheel["url"]) is not str
            or not wheel["url"].startswith("https://files.pythonhosted.org/packages/")
            or not wheel["url"].endswith("/" + wheel["filename"])
            or type(wheel["size"]) is not int
            or not 0 < wheel["size"] < 32 * 1024 * 1024
        ):
            raise BuildError("Nieprawidłowy wpis wheelhouse")
        found[wheel["name"]] = wheel["version"]
    if found != expected:
        raise BuildError("Niekompletny katalog wheelhouse")
    return data["wheels"]


def _wheel(path: Path, entry: dict[str, Any]) -> None:
    info = path.lstat()
    if (
        not stat.S_ISREG(info.st_mode)
        or info.st_nlink != 1
        or info.st_size != entry["size"]
        or _digest(path) != entry["sha256"]
    ):
        raise BuildError("Niezgodny hash lub metadane wheel")
    with zipfile.ZipFile(path) as archive:
        seen = set()
        for member in archive.infolist():
            name = PurePosixPath(member.filename)
            if (
                member.filename in seen
                or name.is_absolute()
                or ".." in name.parts
                or "\\" in member.filename
                or stat.S_ISLNK(member.external_attr >> 16)
                or name.suffix == ".pth"
                or name.name in {"sitecustomize.py", "usercustomize.py"}
            ):
                raise BuildError("Niedozwolony plik wheel")
            seen.add(member.filename)


def verify_wheelhouse(wheelhouse: Path) -> list[dict[str, Any]]:
    """Pełny zbiór wejść musi odpowiadać recenzowanemu katalogowi bajt po bajcie."""
    entries = _catalog()
    if wheelhouse.is_symlink() or not wheelhouse.is_dir():
        raise BuildError("Wheelhouse nie jest rzeczywistym katalogiem")
    if {path.name for path in wheelhouse.iterdir()} != {e["filename"] for e in entries}:
        raise BuildError("Niekompletny wheelhouse lub dodatkowy pakiet")
    for entry in entries:
        _wheel(wheelhouse / entry["filename"], entry)
    expected_lock = (
        "# Ubuntu 24.04 amd64 / CPython 3.12. Exact runtime closure; no sources.\n"
        + "\n".join(
            f"{w['name']}=={w['version']} --hash=sha256:{w['sha256']}"
            for w in entries
            if w["role"] == "runtime"
        )
        + "\n"
    )
    if LOCK.read_text() != expected_lock:
        raise BuildError("Lock nie odpowiada katalogowi wheelhouse")
    return entries


def download_wheelhouse(destination: Path) -> None:
    """Oddzielny krok online: wyłącznie dokładne źródła i zatwierdzone hashe."""
    if destination.exists():
        verify_wheelhouse(destination)
        return
    destination.mkdir(mode=0o755)
    for entry in _catalog():
        path = destination / entry["filename"]
        request = urllib.request.Request(entry["url"], headers={"User-Agent": "CryptoHunter/1"})
        with urllib.request.urlopen(request, timeout=60) as response, path.open("xb") as output:
            remaining = entry["size"] + 1
            while remaining:
                block = response.read(min(65536, remaining))
                if not block:
                    break
                output.write(block)
                remaining -= len(block)
        _wheel(path, entry)
    verify_wheelhouse(destination)


def _run(arguments: list[str]) -> str:
    result = subprocess.run(
        arguments, env=_ENV, capture_output=True, text=True, timeout=180, check=False
    )
    if result.returncode:
        raise BuildError(f"Nieudany krok buildera: {Path(arguments[0]).name}")
    return result.stdout.strip()


def _check_platform() -> str:
    release = platform.freedesktop_os_release()
    if (
        release.get("ID") != "ubuntu"
        or release.get("VERSION_ID") != "24.04"
        or platform.machine() != "x86_64"
        or platform.python_implementation() != "CPython"
        or sys.version_info[:2] != (3, 12)
        or Path(sys.executable).resolve() != BASE_PYTHON
        or sys.prefix != sys.base_prefix
    ):
        raise BuildError("Builder wymaga systemowego CPython 3.12 Ubuntu 24.04 amd64")
    return _run(["/usr/bin/dpkg-query", "-W", "-f=${Version}", "python3.12"])


def _native_dependencies(roots: list[Path]) -> set[Path]:
    """ldd tylko na zaufanych wejściach buildera; wynik staje się podpisaną listą."""
    dependencies = set()
    pending = list(roots)
    visited = set()
    while pending:
        path = pending.pop()
        if path in visited:
            continue
        visited.add(path)
        # Static Rust OpenSSL wheel nie ma DT_NEEDED libssl. Manifest obejmuje
        # bytes .so oraz backend.openssl_version_text() w retained build-probe.
        result = subprocess.run(
            ["/usr/bin/ldd", str(path)], env=_ENV, capture_output=True, text=True, timeout=30
        )
        if result.returncode or "not found" in result.stdout:
            raise BuildError("Nierozwiązana zależność natywna")
        for line in result.stdout.splitlines():
            matches = re.findall(r"(?:=>\s+|^\s*)(/[^\s()]+)", line)
            for match in matches:
                dependency = Path(match)
                if dependency not in dependencies:
                    dependencies.add(dependency)
                    pending.append(dependency)
    return dependencies


def _external_inventory(destination: Path, probe: dict[str, Any]) -> tuple[Path, ...]:
    stdlib = Path(probe["base_prefix"]) / "lib/python3.12"
    paths = {BASE_PYTHON, stdlib, Path("/etc/ld.so.cache")}
    if Path("/etc/ld.so.preload").exists():
        raise BuildError("Niedozwolony preload loadera")
    for config in (Path("/etc/ld.so.conf"), Path("/etc/ld.so.conf.d")):
        if config.exists():
            paths.add(config)
            if config.is_dir():
                paths.update(config.rglob("*"))
    archive = stdlib.parent / "python312.zip"
    if archive.exists():
        paths.add(archive)
    native_roots = [BASE_PYTHON]
    for path in sorted(stdlib.rglob("*")):
        if "site-packages" in path.parts or "dist-packages" in path.parts:
            continue
        paths.add(path)
        if path.suffix == ".so" and not path.is_symlink():
            native_roots.append(path)
    for path in destination.rglob("*"):
        if path.is_file() and (path.suffix == ".so" or ".so." in path.name):
            native_roots.append(path)
    for path in _native_dependencies(native_roots):
        if not path.resolve().is_relative_to(destination):
            paths.add(path.parent.resolve() / path.name)
            paths.add(path.resolve())
    paths.update(
        Path(name) for name in ("/lib", "/lib64", "/bin", "/sbin") if Path(name).is_symlink()
    )
    # Bind both literal symlink object and resolved target, with every directory
    # crossed by their resolution. Ubuntu usrmerge is an explicitly signed link.
    for path in tuple(paths):
        paths.add(path.resolve())
        paths.update(path.parents)
        paths.update(path.resolve().parents)
    paths.discard(Path("/"))
    return tuple(sorted(paths))


def build_runtime(
    wheelhouse: Path,
    destination: Path,
    *,
    environment_id: str,
    release_revision: str,
) -> dict[str, Any]:
    """Nowy kandydat, pełna instalacja offline i rzeczywista próba kryptografii/PQ."""
    provenance = _check_platform()
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,63}", environment_id):
        raise BuildError("Nieprawidłowy environment-id")
    if not re.fullmatch(r"[0-9a-f]{40}", release_revision):
        raise BuildError("Nieprawidłowa rewizja wydania")
    entries = verify_wheelhouse(wheelhouse)
    destination = destination.absolute()
    if destination.exists() or destination.is_symlink():
        raise BuildError("Nie wolno nadpisywać kandydata runtime")
    if any(parent.is_symlink() for parent in destination.parents):
        raise BuildError("Symlink w ścieżce staging")
    destination.parent.mkdir(parents=True, exist_ok=True)
    venv.EnvBuilder(with_pip=False, symlinks=True).create(destination)
    # Cfg nie zawiera losowej ścieżki staging. Venv ma wyłącznie symlinki do
    # dokładnie zatwierdzonej binarki Ubuntu; ich integralność kontroluje core.
    (destination / "pyvenv.cfg").write_text(
        "home = /usr/bin\ninclude-system-site-packages = false\n"
        f"version = {platform.python_version()}\n"
        "executable = /usr/bin/python3.12\n"
    )
    target_python = destination / "bin/python3.12"
    if not target_python.is_symlink() or target_python.resolve() != BASE_PYTHON:
        raise BuildError("Nieprawidłowe powiązanie interpretera venv")
    # Pip jest osobnym, zahashowanym build-tool wheel. Nie trafia do site-packages.
    pip = next(entry for entry in entries if entry["name"] == "pip")
    boot = (
        "import runpy,sys;sys.path.insert(0,sys.argv.pop(1));"
        "runpy.run_module('pip',run_name='__main__')"
    )
    _run(
        [
            str(BASE_PYTHON),
            "-I",
            "-s",
            "-B",
            "-c",
            boot,
            str(wheelhouse.absolute() / pip["filename"]),
            "--python",
            str(target_python),
            "install",
            "--no-index",
            "--require-hashes",
            "--only-binary=:all:",
            "--no-compile",
            "--no-cache-dir",
            "--disable-pip-version-check",
            "--find-links",
            str(wheelhouse.absolute()),
            "-r",
            str(LOCK),
        ]
    )
    # Aktywacja i skrypty konsolowe nie są częścią ścieżki usługi. Nie zostawiaj
    # shebang wskazujących na staging ani narzędzi instalacyjnych runtime.
    for script in (destination / "bin").iterdir():
        if not script.is_symlink():
            script.unlink()
    for path in destination.rglob("*"):
        if path.suffix == ".pth" or path.name in {"sitecustomize.py", "usercustomize.py"}:
            raise BuildError("Niedozwolony hook importów runtime")
    probe = json.loads(_run([str(target_python), "-I", "-s", "-B", "-c", _PROBE]))
    if probe["packages"] != VERSIONS or probe["abi"] != "cpython-312-x86_64-linux-gnu":
        raise BuildError("Niezgodny zbiór bibliotek lub ABI")
    shutil.copytree(wheelhouse, destination / "wheelhouse")
    (destination / "build-tools").mkdir()
    (destination / "wheelhouse" / pip["filename"]).rename(
        destination / "build-tools" / pip["filename"]
    )
    shutil.copyfile(LOCK, destination / "requirements.lock")
    shutil.copyfile(CATALOG, destination / "wheelhouse.json")
    (destination / "build-probe.json").write_text(json.dumps(probe, sort_keys=True) + "\n")
    for path in (destination, *destination.rglob("*")):
        if not path.is_symlink():
            path.chmod(0o755 if path.is_dir() else 0o644)
    collector = runpy.run_path(str(HERE / "freshness_runtime.py"))["collect_runtime_manifest"]
    collected = collector(
        destination,
        environment_id=environment_id,
        release_revision=release_revision,
        interpreter={
            "implementation": "CPython",
            "version": probe["python"],
            "abi": probe["abi"],
            "executable": "bin/python3.12",
            "base_executable": str(BASE_PYTHON),
            "provenance": "Ubuntu dpkg python3.12 " + provenance,
        },
        packages=VERSIONS,
        external_paths=_external_inventory(destination, probe),
        lock_sha256=_digest(LOCK),
    )
    if type(collected) is not dict:
        raise BuildError("Nieprawidłowy manifest kandydata")
    return cast(dict[str, Any], collected)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download-wheelhouse", type=Path)
    parser.add_argument("--wheelhouse", type=Path)
    parser.add_argument("--destination", type=Path)
    parser.add_argument("--environment-id")
    parser.add_argument("--release-revision")
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()
    try:
        if args.download_wheelhouse is not None:
            if any(
                (
                    args.wheelhouse,
                    args.destination,
                    args.environment_id,
                    args.release_revision,
                    args.manifest,
                )
            ):
                raise BuildError("Pobieranie i budowanie są odrębnymi operacjami")
            download_wheelhouse(args.download_wheelhouse)
        else:
            if not all(
                (
                    args.wheelhouse,
                    args.destination,
                    args.environment_id,
                    args.release_revision,
                    args.manifest,
                )
            ):
                raise BuildError("Brak wymaganych parametrów build/manifest")
            if args.manifest.exists() or args.manifest.is_relative_to(args.destination):
                raise BuildError("Manifest musi być nowym plikiem poza kandydatem")
            manifest = build_runtime(
                args.wheelhouse,
                args.destination,
                environment_id=args.environment_id,
                release_revision=args.release_revision,
            )
            with args.manifest.open("x") as output:
                output.write(json.dumps(manifest, sort_keys=True, separators=(",", ":")))
    except (BuildError, OSError, ValueError, subprocess.SubprocessError) as exc:
        print(f"Odmowa budowania runtime: {type(exc).__name__}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

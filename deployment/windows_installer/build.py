"""Deterministic end-to-end builder for the canonical Windows x64 MSI."""

from __future__ import annotations
import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import struct
import subprocess
import sys
import tempfile
from urllib.request import urlopen
import zipfile

from .contract import CONTRACT, is_reviewed_wix_version
from .dependency_contract import REQUIRED_WIN32_MODULES

WIX_EULA_ACCEPTANCE = "PER_INVOCATION"
WIX_EULA_ACCEPTANCE_FLAG = "-acceptEula wix7"


class InstallerBuildError(RuntimeError):
    """A build input or output violates the reviewed installer contract."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def msi_version(version: str) -> str:
    match = re.fullmatch(r"(\d+)\.(\d+)\.(\d+)(?:[-+].*)?", version)
    if not match:
        raise InstallerBuildError("product version must be semantic numeric version")
    fields = tuple(map(int, match.groups()))
    if fields[0] > 255 or fields[1] > 255 or fields[2] > 65535:
        raise InstallerBuildError("product version exceeds MSI field limits")
    return ".".join(map(str, fields))


def verify_postgresql(archive: Path, expected_sha256: str) -> None:
    if not archive.is_file():
        raise InstallerBuildError("pinned PostgreSQL archive is required")
    actual = sha256(archive)
    if actual != expected_sha256:
        raise InstallerBuildError(
            f"PostgreSQL SHA-256 mismatch: expected {expected_sha256}, got {actual}"
        )


def qualify_windows_x64() -> None:
    if sys.platform != "win32" or platform.machine().upper() not in {"AMD64", "X86_64"}:
        raise InstallerBuildError("canonical MSI requires a native Windows x64 runner")
    if (
        struct.calcsize("P") != 8
        or os.environ.get("PROCESSOR_ARCHITEW6432", "").upper() == "AMD64"
    ):
        raise InstallerBuildError("build tool must be native x64, not WOW64")


def canonical_wix_version(output: str) -> str:
    if not is_reviewed_wix_version(output):
        raise InstallerBuildError(
            f"expected stable WiX {CONTRACT.wix_version}, observed {output.strip()!r}"
        )
    return output.strip()


def qualify_pe_x64(path: Path) -> None:
    data = path.read_bytes()
    if data[:2] != b"MZ" or len(data) < 0x40:
        raise InstallerBuildError(f"not a PE executable: {path.name}")
    offset = struct.unpack_from("<I", data, 0x3C)[0]
    if (
        data[offset : offset + 4] != b"PE\0\0"
        or struct.unpack_from("<H", data, offset + 4)[0] != 0x8664
    ):
        raise InstallerBuildError(f"production executable is not PE x64: {path.name}")


def normalize_postgresql_archive(archive: Path, destination: Path) -> None:
    required_files = {
        "pgsql/bin/postgres.exe",
        "pgsql/bin/pg_ctl.exe",
        "pgsql/bin/initdb.exe",
        "pgsql/bin/psql.exe",
        "pgsql/bin/pg_isready.exe",
    }
    with zipfile.ZipFile(archive) as bundle:
        names = {name.replace("\\", "/").rstrip("/") for name in bundle.namelist()}
        roots = {name.split("/", 1)[0] for name in names if name}
        missing = required_files - names
        if (
            roots != {"pgsql"}
            or missing
            or not any(n.startswith("pgsql/lib/") for n in names)
            or not any(n.startswith("pgsql/share/") for n in names)
        ):
            raise InstallerBuildError(
                f"unexpected EDB PostgreSQL archive layout; missing={sorted(missing)} roots={sorted(roots)}"
            )
        with tempfile.TemporaryDirectory(prefix="cryptohunter-pg-") as temporary:
            bundle.extractall(temporary)
            shutil.copytree(Path(temporary) / "pgsql", destination)


def acquire_postgresql(
    pins: dict[str, object], supplied: Path | None, cache: Path
) -> Path:
    pg = pins["postgresql"]
    assert isinstance(pg, dict)
    archive = supplied or cache / str(pg["archive"])
    if supplied is None and not archive.exists():
        cache.mkdir(parents=True, exist_ok=True)
        temporary = archive.with_suffix(".download")
        with (
            urlopen(str(pg["source_url"]), timeout=120) as response,
            temporary.open("wb") as output,
        ):  # noqa: S310
            shutil.copyfileobj(response, output)
        temporary.replace(archive)
    verify_postgresql(archive, str(pg["sha256"]))
    return archive


ENTRYPOINTS = {
    "CryptoHunterBackend": "backend_service.py",
    "CryptoHunterFreshnessVerifier": "verifier_service.py",
    "CryptoHunterPostgreSQL": "postgresql_service.py",
    "CryptoHunterProvision": "provision.py",
}


def qualify_win32_imports() -> None:
    """Fail before packaging unless the reviewed pywin32 surface is importable."""
    missing: list[str] = []
    for module in REQUIRED_WIN32_MODULES:
        try:
            importlib.import_module(module)
        except ImportError:
            missing.append(module)
    if missing:
        raise InstallerBuildError(
            "required pywin32 imports unavailable: " + ", ".join(missing)
        )


def numpy_openblas_dll() -> Path:
    """Locate NumPy's wheel-owned BLAS runtime required by production imports."""
    numpy = importlib.import_module("numpy")
    package = Path(numpy.__file__).resolve().parent
    candidates = sorted(package.parent.glob("numpy.libs/libopenblas*.dll"))
    if len(candidates) != 1:
        raise InstallerBuildError(
            f"expected one NumPy OpenBLAS runtime DLL, found {len(candidates)}"
        )
    return candidates[0]


def qualify_pyinstaller_warnings(warning_file: Path) -> None:
    """Reject missing required Win32 modules without banning optional backends."""
    if not warning_file.is_file():
        raise InstallerBuildError(f"PyInstaller warning file absent: {warning_file}")
    warnings = warning_file.read_text(encoding="utf-8", errors="replace")
    missing = [
        module
        for module in REQUIRED_WIN32_MODULES
        if re.search(rf"missing module named ['\"]{re.escape(module)}['\"]", warnings)
    ]
    if missing:
        raise InstallerBuildError(
            "PyInstaller reports missing required Win32 imports: " + ", ".join(missing)
        )


def smoke_executable(executable: Path) -> None:
    """Exercise the packaged import graph through a read-only CLI path."""
    result = subprocess.run(
        [str(executable), "--build-smoke"], capture_output=True, text=True, timeout=60
    )
    if result.returncode != 0 or "BUILD_SMOKE = PASS" not in result.stdout:
        raise InstallerBuildError(
            f"packaged executable smoke failed: {executable.name} "
            f"(exit={result.returncode})"
        )


def build_executables(payload: Path, work: Path) -> list[Path]:
    outputs = []
    source_root = Path(__file__).parent
    qualify_win32_imports()
    openblas = numpy_openblas_dll()
    print("WIN32_IMPORT_PREFLIGHT = PASS", flush=True)
    print(f"NUMPY_OPENBLAS = {openblas.name}", flush=True)
    for name, script in ENTRYPOINTS.items():
        subprocess.run(
            [
                sys.executable,
                "-m",
                "PyInstaller",
                "--noconfirm",
                "--clean",
                "--onefile",
                "--name",
                name,
                "--distpath",
                str(payload),
                "--workpath",
                str(work / name),
                "--specpath",
                str(work),
                "--hidden-import",
                "win32timezone",
                "--add-binary",
                f"{openblas}{os.pathsep}numpy.libs",
                str(source_root / script),
            ],
            check=True,
        )
        output = payload / f"{name}.exe"
        qualify_pyinstaller_warnings(work / name / name / f"warn-{name}.txt")
        qualify_pe_x64(output)
        smoke_executable(output)
        outputs.append(output)
    print("PYINSTALLER_WIN32_WARNINGS = PASS", flush=True)
    print("PACKAGED_EXE_SMOKE = PASS", flush=True)
    return outputs


def build(args: argparse.Namespace) -> Path:
    qualify_windows_x64()
    pins = json.loads(args.pins.read_text(encoding="utf-8"))
    if pins["wix"]["version"] != CONTRACT.wix_version:
        raise InstallerBuildError("WiX pin differs from reviewed contract")
    wix_version = canonical_wix_version(
        subprocess.run(
            ["wix", "--version"], check=True, capture_output=True, text=True
        ).stdout
    )
    args.output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="cryptohunter-build-") as temporary:
        root = Path(temporary)
        payload = root / "Payload"
        payload.mkdir()
        executables = build_executables(payload, root / "pyinstaller")
        print("EXE_BUILD = PASS", flush=True)
        archive = acquire_postgresql(pins, args.postgresql_archive, args.cache)
        normalize_postgresql_archive(archive, payload / "PostgreSQL")
        artifact = args.output / f"CryptoHunter-{args.version}-windows-x64.msi"
        source = Path(__file__).with_name("Product.wxs")
        print(f"WIX_VERSION = {CONTRACT.wix_version}", flush=True)
        print(f"WIX_EULA_ACCEPTANCE = {WIX_EULA_ACCEPTANCE}", flush=True)
        print(f"WIX_EULA_ACCEPTANCE_FLAG = {WIX_EULA_ACCEPTANCE_FLAG}", flush=True)
        print("WIX_COMPILE = NOT_RUN", flush=True)
        subprocess.run(
            [
                "wix",
                "build",
                "-acceptEula",
                "wix7",
                str(source),
                "-arch",
                "x64",
                "-d",
                f"ProductVersion={msi_version(args.version)}",
                "-d",
                f"PayloadDir={payload}",
                "-o",
                str(artifact),
            ],
            check=True,
        )
        print("WIX_COMPILE = PASS", flush=True)
        if not artifact.is_file():
            raise InstallerBuildError(
                "WiX exited successfully without creating the canonical MSI"
            )
        print("MSI_CREATED = YES", flush=True)
        manifest = {
            "schema_version": 1,
            "product_version": args.version,
            "architecture": "x64",
            "msi": {"file": artifact.name, "sha256": sha256(artifact)},
            "wix_version": wix_version,
            "wix_eula_acceptance": WIX_EULA_ACCEPTANCE,
            "wix_eula_acceptance_flag": WIX_EULA_ACCEPTANCE_FLAG,
            "postgresql_version": pins["postgresql"]["version"],
            "postgresql_packaging_revision": pins["postgresql"]["packaging_revision"],
            "postgresql_source_url": pins["postgresql"]["source_url"],
            "postgresql_payload_sha256": pins["postgresql"]["sha256"],
            "production_executables": {p.name: sha256(p) for p in executables},
            "installed_files": {
                str(
                    (Path("PostgreSQL") / path.relative_to(payload / "PostgreSQL"))
                    if path.is_relative_to(payload / "PostgreSQL")
                    else (
                        Path("PostgreSQL") / path.name
                        if path.name == "CryptoHunterPostgreSQL.exe"
                        else Path(path.name)
                    )
                ).replace("\\", "/"): sha256(path)
                for path in sorted(payload.rglob("*"))
                if path.is_file()
            },
        }
        (args.output / "installer-manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    return artifact


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", required=True)
    parser.add_argument("--postgresql-archive", type=Path)
    parser.add_argument("--cache", type=Path, default=Path(".cache/windows-installer"))
    parser.add_argument("--output", type=Path, default=Path("dist/windows"))
    parser.add_argument(
        "--pins", type=Path, default=Path(__file__).with_name("pins.json")
    )
    build(parser.parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

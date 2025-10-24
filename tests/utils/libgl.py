"""Helpers zapewniające dostępność libGL dla testów Qt."""
from __future__ import annotations

import ctypes
import ctypes.util
import os
import shutil
import subprocess
from pathlib import Path
from typing import Final
from urllib.request import urlopen

import hashlib

_CACHE_DIRNAME: Final[str] = ".libgl-cache"
_PACKAGES: Final[tuple[tuple[str, str, str, tuple[str, ...]], ...]] = (
    (
        "https://deb.debian.org/debian/pool/main/libg/libglvnd/libgl1_1.6.0-1_amd64.deb",
        "libgl1_1.6.0-1_amd64.deb",
        "6f89b1702c48e9a2437bb3c1ffac8e1ab2d828fc28b3d14b2eecd4cc19b2c790",
        ("libGL.so.1",),
    ),
    (
        "https://deb.debian.org/debian/pool/main/libg/libglvnd/libglvnd0_1.6.0-1_amd64.deb",
        "libglvnd0_1.6.0-1_amd64.deb",
        "b6da5b153dd62d8b5e5fbe25242db1fc05c068707c365db49abda8c2427c75f8",
        ("libGLdispatch.so.0",),
    ),
    (
        "https://deb.debian.org/debian/pool/main/libg/libglvnd/libglx0_1.6.0-1_amd64.deb",
        "libglx0_1.6.0-1_amd64.deb",
        "95f568df73dedf43ae66834a75502112e0d4f3ad7124f3dbfa790b739383b896",
        ("libGLX.so.0",),
    ),
    (
        "https://deb.debian.org/debian/pool/main/libg/libglvnd/libegl1_1.6.0-1_amd64.deb",
        "libegl1_1.6.0-1_amd64.deb",
        "fe4d8b39f6e6fe1a32ab1efd85893553eaa9cf3866aa668ccf355f585b37d523",
        ("libEGL.so.1",),
    ),
    (
        "https://deb.debian.org/debian/pool/main/libx/libxkbcommon/libxkbcommon0_1.5.0-1_amd64.deb",
        "libxkbcommon0_1.5.0-1_amd64.deb",
        "e3fe045b9a33a101de1c5a912a4a10928db055c3f68930f47eccbb44d7c7d54e",
        ("libxkbcommon.so.0",),
    ),
)


def _has_libgl() -> bool:
    """Sprawdza, czy aktualny proces ma dostęp do libGL."""
    candidate = ctypes.util.find_library("GL")
    try:
        if candidate:
            ctypes.CDLL(candidate)
            return True
        ctypes.CDLL("libGL.so.1")
        return True
    except OSError:
        return False


def _download_deb(url: str, target: Path, expected_sha256: str) -> None:
    with urlopen(url) as response, target.open("wb") as deb_file:
        shutil.copyfileobj(response, deb_file)

    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    if digest != expected_sha256:
        raise RuntimeError(
            "Niepoprawna suma SHA256 dla pobranego pakietu libGL: "
            f"{digest} != {expected_sha256}"
        )


def _extract_deb(deb_path: Path, destination: Path) -> Path:
    subprocess.run(["dpkg-deb", "-x", str(deb_path), str(destination)], check=True)
    lib_dir = destination / "usr" / "lib" / "x86_64-linux-gnu"
    return lib_dir


def ensure_libgl_available(cache_root: Path | None = None) -> Path | None:
    """Zapewnia obecność libGL w środowisku testów.

    Zwraca ścieżkę, która została dopisana do zmiennej ``LD_LIBRARY_PATH`` lub ``None``
    jeśli biblioteka była już dostępna.
    """

    if _has_libgl():
        return None

    cache_root = cache_root or Path(__file__).resolve().parent / _CACHE_DIRNAME
    cache_root.mkdir(parents=True, exist_ok=True)

    lib_dir = cache_root / "usr" / "lib" / "x86_64-linux-gnu"

    for url, filename, checksum, expected_files in _PACKAGES:
        deb_path = cache_root / filename
        marker = cache_root / f"{filename}.ok"

        should_extract = not marker.exists()
        if not should_extract:
            if not lib_dir.exists() or any(not (lib_dir / name).exists() for name in expected_files):
                should_extract = True

        if should_extract:
            _download_deb(url, deb_path, checksum)
            lib_dir = _extract_deb(deb_path, cache_root)
            marker.write_text(str(lib_dir))

    missing = [
        name
        for _, _, _, expected_files in _PACKAGES
        for name in expected_files
        if not (lib_dir / name).exists()
    ]
    if missing:
        raise RuntimeError(
            "Nie udało się przygotować wszystkich bibliotek GL: brakujące pliki "
            + ", ".join(sorted(set(missing)))
        )

    current = os.environ.get("LD_LIBRARY_PATH", "")
    lib_dir_str = str(lib_dir)
    paths = [p for p in current.split(":") if p]
    if lib_dir_str not in paths:
        os.environ["LD_LIBRARY_PATH"] = ":".join([lib_dir_str, *paths]) if paths else lib_dir_str

    try:
        for dependency in (
            "libGLdispatch.so.0",
            "libGLX.so.0",
            "libEGL.so.1",
            "libxkbcommon.so.0",
            "libGL.so.1",
        ):
            ctypes.CDLL(str(lib_dir / dependency), mode=ctypes.RTLD_GLOBAL)
    except OSError as exc:  # pragma: no cover - środowisko bez wsparcia ELF
        raise RuntimeError(
            f"Nie udało się załadować bibliotek GL z {lib_dir}: {exc}"
        ) from exc

    return lib_dir

"""Automates cross-platform packaging via PyInstaller and Briefcase."""
from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
from pathlib import Path
from typing import Sequence


LOGGER = logging.getLogger(__name__)
SUPPORTED_PLATFORMS = {"linux", "windows", "macos"}


def _normalize_platform(value: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise argparse.ArgumentTypeError("Należy podać docelową platformę")
    lowered = normalized.lower()
    if lowered == "osx":
        lowered = "macos"
    if lowered not in SUPPORTED_PLATFORMS:
        choices = ", ".join(sorted(SUPPORTED_PLATFORMS))
        raise argparse.ArgumentTypeError(f"Nieobsługiwana platforma '{value}'. Dostępne: {choices}.")
    return lowered


def _run(command: Sequence[str], *, cwd: Path | None = None) -> None:
    display = " ".join(command)
    LOGGER.info("Uruchamiam: %s", display)
    try:
        subprocess.run(command, check=True, cwd=cwd)
    except subprocess.CalledProcessError as exc:  # pragma: no cover - CLI diagnostics
        raise SystemExit(f"Polecenie zakończyło się niepowodzeniem ({exc.returncode}): {display}") from exc


def _build_pyinstaller(entrypoint: Path, dist_dir: Path, work_dir: Path, platform_id: str) -> Path:
    dist_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)
    args = [
        "pyinstaller",
        "--clean",
        "--noconfirm",
        "--distpath",
        str(dist_dir),
        "--workpath",
        str(work_dir),
        str(entrypoint),
    ]
    _run(args)
    extension = ".exe" if platform_id == "windows" else ""
    return dist_dir / entrypoint.stem / f"{entrypoint.stem}{extension}"


def _package_briefcase(app: str, platform_id: str, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    briefcase_platform = "macOS" if platform_id == "macos" else platform_id
    for command in (("create",), ("build",), ("package",)):
        _run(["briefcase", *command, briefcase_platform, app])
    artifact_dir = Path("dist")
    packages = sorted(artifact_dir.glob(f"{app}*"))
    if not packages:
        raise SystemExit("Briefcase nie wygenerował pakietów w katalogu dist/")
    bundled: list[Path] = []
    for candidate in packages:
        destination = output_dir / candidate.name
        if candidate.is_dir():
            if destination.exists():
                shutil.rmtree(destination)
            shutil.copytree(candidate, destination)
        else:
            shutil.copy2(candidate, destination)
        bundled.append(destination)
    LOGGER.info("Zapisano pakiety Briefcase: %s", ", ".join(str(path) for path in bundled))
    return output_dir


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pyinstaller-entry", help="Plik wejściowy PyInstaller (opcjonalnie)")
    parser.add_argument(
        "--pyinstaller-dist",
        type=Path,
        default=Path("var/dist/pyinstaller"),
        help="Katalog wyjściowy PyInstaller",
    )
    parser.add_argument(
        "--pyinstaller-work",
        type=Path,
        default=Path("var/build/pyinstaller"),
        help="Katalog roboczy PyInstaller",
    )
    parser.add_argument("--briefcase-app", help="Nazwa aplikacji Briefcase (opcjonalnie)")
    parser.add_argument(
        "--briefcase-output",
        type=Path,
        default=Path("var/dist/briefcase"),
        help="Katalog wyjściowy pakietów Briefcase",
    )
    parser.add_argument(
        "--platform",
        type=_normalize_platform,
        choices=sorted(SUPPORTED_PLATFORMS),
        default="linux",
        help="Docelowa platforma pakietów",
    )
    parser.add_argument("--verbose", action="store_true", help="Włącz logowanie DEBUG")
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="[%(levelname)s] %(message)s")

    platform_id = args.platform

    if args.pyinstaller_entry:
        entry_path = Path(args.pyinstaller_entry).expanduser().resolve()
        if not entry_path.exists():
            raise SystemExit(f"Entrypoint PyInstaller nie istnieje: {entry_path}")
        executable = _build_pyinstaller(
            entry_path,
            Path(args.pyinstaller_dist).expanduser().resolve(),
            Path(args.pyinstaller_work).expanduser().resolve(),
            platform_id,
        )
        if not executable.exists():
            raise SystemExit(f"PyInstaller nie wygenerował binarki w {executable}")
        LOGGER.info("PyInstaller artefakt: %s", executable)

    if args.briefcase_app:
        _package_briefcase(args.briefcase_app, platform_id, Path(args.briefcase_output).expanduser().resolve())

    if not args.pyinstaller_entry and not args.briefcase_app:
        LOGGER.warning("Nie wskazano żadnych zadań build – pomijam.")

    return 0


if __name__ == "__main__":  # pragma: no cover - wejście CLI
    raise SystemExit(main())

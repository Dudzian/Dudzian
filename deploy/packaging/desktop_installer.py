"""Budowanie wieloplatformowych instalatorów desktopowych dla KryptoŁowcy."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import tomllib
import zipfile


def _normalize_path(raw: str, base_dir: Path) -> Path:
    """Zamienia ścieżki z profilu (również Windowsowe) na absolutne ``Path``."""

    normalized = raw.replace("\\", os.sep)
    path = Path(normalized)
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return path


def _coerce_path(value: object, base_dir: Path, *, default: str | None = None) -> Path:
    if value is None:
        if default is None:
            raise SystemExit("Wymagana ścieżka w profilu instalatora jest pusta")
        value = default
    if not isinstance(value, str):
        raise SystemExit(f"Oczekiwano ścieżki jako string, otrzymano {type(value).__name__}")
    return _normalize_path(value, base_dir)


@dataclass(slots=True, frozen=True)
class IncludeSpec:
    name: str
    source: Path


@dataclass(slots=True, frozen=True)
class BundleConfig:
    platform: str
    output_dir: Path
    work_dir: Path
    qt_dist: Path | None
    includes: tuple[IncludeSpec, ...]
    wheels: tuple[Path, ...]
    metadata_path: Path
    raw_profile: dict[str, object]


class DesktopInstallerBuilder:
    """Generator paczek instalacyjnych z wbudowaną walidacją HWID."""

    def __init__(
        self,
        *,
        version: str,
        profiles_dir: Path,
        hwid_hook_source: Path | None = None,
        clean: bool = True,
        verify_hook: bool = False,
    ) -> None:
        self._version = version
        self._profiles_dir = profiles_dir
        self._hwid_hook_source = (hwid_hook_source or Path("probe_keyring.py")).resolve()
        self._clean = clean
        self._verify_hook = verify_hook

    # ------------------------------------------------------------------
    def build(self, platform: str) -> Path:
        config = self._load_profile(platform)
        stage_dir = config.work_dir / f"desktop-installer-{self._version}-{platform}"
        if self._clean and stage_dir.exists():
            shutil.rmtree(stage_dir)
        stage_dir.mkdir(parents=True, exist_ok=True)

        # Kopiowanie zasobów bundla.
        for include in config.includes:
            destination = stage_dir / include.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            if include.source.is_dir():
                shutil.copytree(include.source, destination, dirs_exist_ok=True)
            else:
                shutil.copy2(include.source, destination)

        if config.wheels:
            wheels_dir = stage_dir / "wheels"
            wheels_dir.mkdir(parents=True, exist_ok=True)
            for wheel in config.wheels:
                if not wheel.exists():
                    raise FileNotFoundError(
                        f"Zadeklarowany pakiet wheel nie istnieje: {wheel}"
                    )
                if wheel.is_dir():
                    destination = wheels_dir / wheel.name
                    if destination.exists():
                        shutil.rmtree(destination)
                    shutil.copytree(wheel, destination, dirs_exist_ok=True)
                else:
                    shutil.copy2(wheel, wheels_dir / wheel.name)

        if config.qt_dist and config.qt_dist.exists():
            qt_dest = stage_dir / "qt"
            shutil.copytree(config.qt_dist, qt_dest, dirs_exist_ok=True)

        hooks_dir = stage_dir / "hooks"
        hooks_dir.mkdir(parents=True, exist_ok=True)
        hook_source = self._hwid_hook_source
        if not hook_source.exists():
            raise FileNotFoundError(f"Brak pliku hooka HWID: {hook_source}")
        shutil.copy2(hook_source, hooks_dir / "probe_keyring.py")
        self._write_validate_hook(hooks_dir)

        manifest = self._write_manifest(stage_dir, config)
        config.metadata_path.parent.mkdir(parents=True, exist_ok=True)
        config.metadata_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

        config.output_dir.mkdir(parents=True, exist_ok=True)
        archive_prefix = config.output_dir / f"desktop-installer-{self._version}-{platform}"
        archive_path = Path(shutil.make_archive(str(archive_prefix), "zip", root_dir=stage_dir))
        manifest["archive"] = archive_path.name
        config.metadata_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

        if self._verify_hook:
            self._verify_archive(archive_path)

        return archive_path

    # ------------------------------------------------------------------
    def _verify_archive(self, archive_path: Path) -> None:
        with tempfile.TemporaryDirectory(prefix="installer-verify-") as temp_dir:
            temp_path = Path(temp_dir)
            with zipfile.ZipFile(archive_path) as zf:
                zf.extractall(temp_path)

            hook_path = temp_path / "hooks" / "validate_hwid.py"
            if not hook_path.exists():
                raise RuntimeError("Pakiet nie zawiera hooka validate_hwid.py")

            fingerprint = f"TEST-FINGERPRINT-{self._version}"
            expected_path = temp_path / "fingerprint.expected.json"
            expected_path.write_text(
                json.dumps({"fingerprint": fingerprint}, ensure_ascii=False),
                encoding="utf-8",
            )

            log_path = temp_path / "hwid_verify.log"

            env = os.environ.copy()
            env.update(
                {
                    "KBOT_EXPECTED_HWID_FILE": str(expected_path),
                    "KBOT_INSTALL_LOG": str(log_path),
                    "KBOT_FAKE_FINGERPRINT": fingerprint,
                }
            )

            repo_root = Path.cwd()
            python_path = env.get("PYTHONPATH")
            env["PYTHONPATH"] = (
                os.pathsep.join([str(repo_root), python_path]) if python_path else str(repo_root)
            )

            result = subprocess.run(
                [sys.executable, str(hook_path)],
                check=False,
                env=env,
                capture_output=True,
                text=True,
            )

            if result.returncode != 0:
                raise RuntimeError(
                    "Walidacja hooka HWID nie powiodła się: "
                    + (result.stderr or result.stdout or str(result.returncode))
                )

            if not log_path.exists():
                raise RuntimeError("Hook HWID nie zapisał logu walidacji")

            log_text = log_path.read_text(encoding="utf-8").strip()
            if fingerprint not in log_text:
                raise RuntimeError("Log walidacji HWID nie zawiera odczytanego fingerprintu")

    # ------------------------------------------------------------------
    def _load_profile(self, platform: str) -> BundleConfig:
        profile_path = (self._profiles_dir / f"{platform}.toml").resolve()
        if not profile_path.exists():
            raise FileNotFoundError(f"Profil instalatora {platform} nie istnieje: {profile_path}")
        raw = tomllib.loads(profile_path.read_text(encoding="utf-8"))
        bundle_section = raw.get("bundle")
        if not isinstance(bundle_section, dict):
            raise SystemExit(f"Profil {platform} nie zawiera sekcji [bundle]")

        base_dir = profile_path.parent

        include_specs: list[IncludeSpec] = []
        for entry in bundle_section.get("include", []):
            if not isinstance(entry, str) or "=" not in entry:
                raise SystemExit(f"Pozycja include musi mieć postać nazwa=ścieżka, otrzymano: {entry!r}")
            name, raw_path = entry.split("=", 1)
            name = name.strip()
            if not name:
                raise SystemExit("Alias include nie może być pusty")
            source = _normalize_path(raw_path.strip(), base_dir)
            include_specs.append(IncludeSpec(name=name, source=source))

        wheels: list[Path] = []
        for entry in bundle_section.get("wheels_extra", []) or []:
            if not isinstance(entry, str) or not entry.strip():
                raise SystemExit(
                    "Element listy wheels_extra musi być niepustym stringiem wskazującym plik lub katalog"
                )
            wheels.append(_normalize_path(entry.strip(), base_dir))

        qt_dist = bundle_section.get("qt_dist")
        qt_path = _normalize_path(qt_dist, base_dir) if isinstance(qt_dist, str) else None

        output_dir = _coerce_path(bundle_section.get("output_dir"), base_dir, default="var/dist/installers")
        work_dir = _coerce_path(bundle_section.get("work_dir"), base_dir, default="var/build/installers")
        metadata_path = _coerce_path(
            bundle_section.get("metadata_path"), base_dir, default="installer_metadata.json"
        )

        return BundleConfig(
            platform=raw.get("platform", platform),
            output_dir=output_dir,
            work_dir=work_dir,
            qt_dist=qt_path,
            includes=tuple(include_specs),
            wheels=tuple(wheels),
            metadata_path=metadata_path,
            raw_profile=raw,
        )

    # ------------------------------------------------------------------
    def _write_validate_hook(self, hooks_dir: Path) -> None:
        hook_path = hooks_dir / "validate_hwid.py"
        hook_path.write_text(
            """#!/usr/bin/env python3
from __future__ import annotations

import os
from pathlib import Path

from probe_keyring import HwidValidationError, install_hook_main


def _default_expected_path() -> Path | None:
    fallback = Path(__file__).resolve().parents[1] / "config" / "fingerprint.expected.json"
    return fallback if fallback.exists() else None


def main() -> None:
    expected_path = os.environ.get("KBOT_EXPECTED_HWID_FILE")
    if expected_path:
        path = Path(expected_path).expanduser()
    else:
        fallback = _default_expected_path()
        path = fallback if fallback is not None else None

    try:
        fingerprint = install_hook_main(str(path) if path is not None else None)
    except HwidValidationError as exc:
        raise SystemExit(f"HWID validation failed: {exc}") from exc

    log_target = os.environ.get("KBOT_INSTALL_LOG")
    if log_target:
        target = Path(log_target).expanduser()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(f"HWID validation successful for fingerprint: {fingerprint}\\n", encoding="utf-8")


if __name__ == "__main__":  # pragma: no cover
    main()
""",
            encoding="utf-8",
        )

    # ------------------------------------------------------------------
    def _write_manifest(self, stage_dir: Path, config: BundleConfig) -> dict[str, object]:
        files = []
        for path in sorted(stage_dir.rglob("*")):
            if path.is_file():
                relative = path.relative_to(stage_dir).as_posix()
                files.append(
                    {
                        "path": relative,
                        "sha256": _hash_file(path),
                        "size": path.stat().st_size,
                    }
                )

        manifest = {
            "version": self._version,
            "platform": config.platform,
            "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "files": files,
            "hwid_validation": {
                "hook": "hooks/validate_hwid.py",
                "probe": "hooks/probe_keyring.py",
            },
            "profile": {k: v for k, v in config.raw_profile.items() if k != "bundle"},
        }

        (stage_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return manifest


def _hash_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_from_cli(args: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Buduje paczki instalatora desktopowego.")
    parser.add_argument("--version", required=True, help="Wersja instalatora umieszczona w nazwach plików")
    parser.add_argument(
        "--platform",
        choices=("linux", "windows", "macos", "all"),
        default="all",
        help="Docelowa platforma lub 'all' dla wszystkich profili",
    )
    parser.add_argument(
        "--profiles-dir",
        default=Path(__file__).resolve().parent / "profiles",
        type=Path,
        help="Katalog z profilami TOML",
    )
    parser.add_argument(
        "--hook-source",
        default=Path("probe_keyring.py"),
        type=Path,
        help="Źródłowy skrypt walidacji HWID kopiowany do instalatora",
    )
    parser.add_argument("--no-clean", action="store_true", help="Nie usuwaj wcześniejszych buildów w katalogu roboczym")
    parser.add_argument("--verify-hook", action="store_true", help="Uruchom walidację hooka HWID na zbudowanym archiwum")

    parsed = parser.parse_args(args=args)
    builder = DesktopInstallerBuilder(
        version=parsed.version,
        profiles_dir=parsed.profiles_dir.resolve(),
        hwid_hook_source=parsed.hook_source.resolve(),
        clean=not parsed.no_clean,
        verify_hook=parsed.verify_hook,
    )

    platforms = ("linux", "windows", "macos") if parsed.platform == "all" else (parsed.platform,)
    built: list[Path] = []
    for platform in platforms:
        archive = builder.build(platform)
        built.append(archive)
        print(f"Zbudowano instalator {archive}")

    print("Gotowe:")
    for archive in built:
        print(f" - {archive}")


if __name__ == "__main__":  # pragma: no cover
    build_from_cli()


__all__ = ["DesktopInstallerBuilder", "build_from_cli"]

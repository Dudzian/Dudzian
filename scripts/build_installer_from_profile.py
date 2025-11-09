"""Buduje instalator w oparciu o profil PyInstaller/Briefcase."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

try:  # Python 3.11+
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - środowiska <3.11
    import tomli as tomllib  # type: ignore[no-redef]

from deploy.packaging import build_pyinstaller_bundle
from scripts import oem_provision_license


@dataclass(slots=True)
class PyInstallerProfile:
    entrypoint: Path
    runtime_name: str | None
    hidden_imports: tuple[str, ...]
    dist_dir: Path | None
    work_dir: Path | None


@dataclass(slots=True)
class BriefcaseProfile:
    project_path: Path
    app_name: str
    output_dir: Path | None


@dataclass(slots=True)
class BundleProfile:
    output_dir: Path
    work_dir: Path
    qt_dist: Path | None
    include: tuple[str, ...]
    metadata_path: Path
    signing_key: str | None
    signing_key_id: str | None


@dataclass(slots=True)
class Profile:
    platform: str
    pyinstaller: PyInstallerProfile | None
    briefcase: BriefcaseProfile | None
    bundle: BundleProfile


def _read_profile(path: Path) -> Profile:
    document = tomllib.loads(path.read_text(encoding="utf-8"))

    platform = document.get("platform")
    if not isinstance(platform, str) or not platform.strip():
        raise SystemExit("Profil musi zawierać pole 'platform'.")

    pyinstaller_section = document.get("pyinstaller") or {}
    pyinstaller: PyInstallerProfile | None = None
    if pyinstaller_section:
        entry_raw = pyinstaller_section.get("entrypoint")
        if not isinstance(entry_raw, str):
            raise SystemExit("Sekcja 'pyinstaller' wymaga pola 'entrypoint'.")
        runtime_name = pyinstaller_section.get("runtime_name")
        if runtime_name is not None and not isinstance(runtime_name, str):
            raise SystemExit("Pole 'runtime_name' musi być tekstem.")
        hidden_imports = tuple(
            entry
            for entry in (pyinstaller_section.get("hidden_imports") or [])
            if isinstance(entry, str) and entry.strip()
        )
        dist_dir = pyinstaller_section.get("dist_dir")
        work_dir = pyinstaller_section.get("work_dir")
        pyinstaller = PyInstallerProfile(
            entrypoint=(path.parent / entry_raw).resolve(),
            runtime_name=runtime_name,
            hidden_imports=hidden_imports,
            dist_dir=(path.parent / dist_dir).resolve() if isinstance(dist_dir, str) else None,
            work_dir=(path.parent / work_dir).resolve() if isinstance(work_dir, str) else None,
        )

    briefcase_section = document.get("briefcase") or {}
    briefcase: BriefcaseProfile | None = None
    if briefcase_section:
        project_raw = briefcase_section.get("project")
        app_name = briefcase_section.get("app")
        if not isinstance(project_raw, str) or not isinstance(app_name, str):
            raise SystemExit("Sekcja 'briefcase' wymaga pól 'project' i 'app'.")
        output_dir = briefcase_section.get("output_dir")
        briefcase = BriefcaseProfile(
            project_path=(path.parent / project_raw).resolve(),
            app_name=app_name,
            output_dir=(path.parent / output_dir).resolve() if isinstance(output_dir, str) else None,
        )

    bundle_section = document.get("bundle") or {}
    output_dir_raw = bundle_section.get("output_dir")
    work_dir_raw = bundle_section.get("work_dir")
    if not isinstance(output_dir_raw, str) or not isinstance(work_dir_raw, str):
        raise SystemExit("Sekcja 'bundle' wymaga pól 'output_dir' i 'work_dir'.")
    qt_dist_raw = bundle_section.get("qt_dist")
    include_list = bundle_section.get("include") or []
    include = tuple(entry for entry in include_list if isinstance(entry, str) and entry.strip())
    metadata_raw = bundle_section.get("metadata_path")
    if not isinstance(metadata_raw, str) or not metadata_raw.strip():
        raise SystemExit("Sekcja 'bundle' wymaga pola 'metadata_path'.")
    signing_key = bundle_section.get("signing_key")
    if signing_key is not None and not isinstance(signing_key, str):
        raise SystemExit("Pole 'signing_key' musi być tekstem.")
    signing_key_id = bundle_section.get("signing_key_id")
    if signing_key_id is not None and not isinstance(signing_key_id, str):
        raise SystemExit("Pole 'signing_key_id' musi być tekstem.")

    bundle = BundleProfile(
        output_dir=(path.parent / output_dir_raw).resolve(),
        work_dir=(path.parent / work_dir_raw).resolve(),
        qt_dist=(path.parent / qt_dist_raw).resolve() if isinstance(qt_dist_raw, str) else None,
        include=include,
        metadata_path=(path.parent / metadata_raw).resolve(),
        signing_key=signing_key,
        signing_key_id=signing_key_id,
    )

    return Profile(platform=platform.strip(), pyinstaller=pyinstaller, briefcase=briefcase, bundle=bundle)


def _ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _build_pyinstaller(profile: PyInstallerProfile, platform_id: str) -> Path:
    args = [
        "pyinstaller",
        "--clean",
        "--noconfirm",
    ]
    if profile.runtime_name:
        args.extend(["--name", profile.runtime_name])
    if profile.dist_dir is not None:
        _ensure_directory(profile.dist_dir)
        args.extend(["--distpath", str(profile.dist_dir)])
    if profile.work_dir is not None:
        _ensure_directory(profile.work_dir)
        args.extend(["--workpath", str(profile.work_dir)])
    for hidden in profile.hidden_imports:
        args.extend(["--hidden-import", hidden])
    args.append(str(profile.entrypoint))

    subprocess.run(args, check=True)

    executable_dir = (profile.dist_dir or profile.entrypoint.parent / "dist") / profile.entrypoint.stem
    extension = ".exe" if platform_id == "windows" else ""
    candidate = executable_dir / f"{profile.entrypoint.stem}{extension}"
    if not candidate.exists():
        raise SystemExit(f"PyInstaller nie wygenerował binarki runtime pod {candidate}")
    return candidate.resolve()


def _run_briefcase(profile: BriefcaseProfile, platform_id: str) -> Path:
    output = profile.output_dir or profile.project_path / "dist"
    _ensure_directory(output)
    briefcase_platform = "macOS" if platform_id == "macos" else platform_id
    env = os.environ.copy()
    env.setdefault("BRIEFCASE_ROOT", str((profile.project_path / "build").resolve()))
    for command in (("create",), ("build",), ("package",)):
        subprocess.run(["briefcase", *command, briefcase_platform, profile.app_name], cwd=profile.project_path, env=env, check=True)
    return output.resolve()


def _compose_namespace(
    *,
    entrypoint: Path,
    platform_id: str,
    version: str,
    output_dir: Path,
    work_dir: Path,
    hidden_imports: Iterable[str],
    runtime_name: str | None,
    qt_dist: Path | None,
    briefcase_project: Path | None,
    include: Iterable[str],
    signing_key: str | None,
    signing_key_id: str | None,
) -> argparse.Namespace:
    return argparse.Namespace(
        entrypoint=str(entrypoint),
        platform=platform_id,
        version=version,
        output_dir=str(output_dir),
        workdir=str(work_dir),
        hidden_import=list(hidden_imports),
        runtime_name=runtime_name,
        include=list(include),
        qt_dist=str(qt_dist) if qt_dist else None,
        briefcase_project=str(briefcase_project) if briefcase_project else None,
        signing_key=signing_key,
        signing_key_id=signing_key_id,
        license_json=None,
        license_fingerprint=None,
        license_output_name="license_store.json",
        license_hmac_key=None,
        metadata=None,
        metadata_file=None,
        metadata_url=None,
        metadata_url_header=None,
        metadata_url_timeout=None,
        metadata_url_max_size=None,
        metadata_url_allow_http=False,
        metadata_url_allowed_host=None,
        metadata_url_cert_fingerprint=None,
        metadata_url_cert_subject=None,
        metadata_url_cert_issuer=None,
        metadata_url_cert_san=None,
        metadata_url_cert_eku=None,
        metadata_url_cert_policy=None,
        metadata_url_cert_serial=None,
        metadata_url_ca=None,
        metadata_url_capath=None,
        metadata_url_client_cert=None,
        metadata_url_client_key=None,
        metadata_ini=None,
        metadata_toml=None,
        metadata_yaml=None,
        metadata_dotenv=None,
        metadata_env_prefix=None,
        allowed_profile=None,
        metadata_url_header_basic=None,
    )


def _write_metadata(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True, help="Ścieżka do pliku profilu TOML")
    parser.add_argument("--version", required=True, help="Wersja pakietu zapisywana w bundlu")
    parser.add_argument("--platform", choices=["linux", "macos", "windows"], help="Nadpisz platformę z profilu")
    parser.add_argument("--skip-pyinstaller", action="store_true", help="Pomiń etap budowania PyInstaller")
    parser.add_argument("--skip-briefcase", action="store_true", help="Pomiń etap Briefcase")
    parser.add_argument("--metadata-out", help="Ścieżka alternatywnego pliku metadanych")
    parser.add_argument("--verify-fingerprint", help="Plik fingerprint.expected.json do weryfikacji po buildzie")
    parser.add_argument(
        "--verify-fingerprint-key",
        action="append",
        dest="verify_fingerprint_keys",
        help="Klucz HMAC fingerprintu (key_id=sekret) przekazywany do --verify",
    )
    parser.add_argument(
        "--verify-license-key",
        action="append",
        dest="verify_license_keys",
        help="Klucz HMAC licencji (key_id=sekret) przekazywany do --verify",
    )
    parser.add_argument(
        "--verify-registry",
        help="Ścieżka rejestru licencji JSONL do weryfikacji (domyślnie var/licenses/registry.jsonl)",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    profile_path = Path(args.profile).expanduser().resolve()
    if not profile_path.exists():
        raise SystemExit(f"Nie znaleziono profilu: {profile_path}")

    profile = _read_profile(profile_path)
    platform_id = args.platform or profile.platform
    if platform_id not in {"linux", "macos", "windows"}:
        raise SystemExit(f"Nieobsługiwana platforma: {platform_id}")

    runtime_executable: Path | None = None
    if profile.pyinstaller and not args.skip_pyinstaller:
        runtime_executable = _build_pyinstaller(profile.pyinstaller, platform_id)
    elif profile.pyinstaller:
        runtime_executable = (profile.pyinstaller.dist_dir or Path("dist")) / profile.pyinstaller.entrypoint.stem

    briefcase_output: Path | None = None
    if profile.briefcase and not args.skip_briefcase:
        briefcase_output = _run_briefcase(profile.briefcase, platform_id)

    bundle_args = _compose_namespace(
        entrypoint=profile.pyinstaller.entrypoint if profile.pyinstaller else profile_path.parent / "scripts" / "run_local_bot.py",
        platform_id=platform_id,
        version=args.version,
        output_dir=profile.bundle.output_dir,
        work_dir=profile.bundle.work_dir,
        hidden_imports=profile.pyinstaller.hidden_imports if profile.pyinstaller else (),
        runtime_name=profile.pyinstaller.runtime_name if profile.pyinstaller else None,
        qt_dist=profile.bundle.qt_dist,
        briefcase_project=profile.briefcase.project_path if profile.briefcase else None,
        include=profile.bundle.include,
        signing_key=profile.bundle.signing_key,
        signing_key_id=profile.bundle.signing_key_id,
    )

    archive_path = build_pyinstaller_bundle.build_bundle(bundle_args)

    metadata_path = Path(args.metadata_out).expanduser().resolve() if args.metadata_out else profile.bundle.metadata_path
    payload: dict[str, object] = {
        "profile": str(profile_path),
        "platform": platform_id,
        "version": args.version,
        "bundle_archive": str(archive_path),
        "output_dir": str(profile.bundle.output_dir),
        "work_dir": str(profile.bundle.work_dir),
    }
    if runtime_executable is not None:
        payload["runtime_executable"] = str(runtime_executable)
    if profile.pyinstaller and profile.pyinstaller.dist_dir is not None:
        payload["pyinstaller_dist"] = str(profile.pyinstaller.dist_dir)
    if profile.briefcase:
        payload["briefcase_project"] = str(profile.briefcase.project_path)
        if profile.briefcase.output_dir:
            payload["briefcase_output"] = str(profile.briefcase.output_dir)
    if briefcase_output is not None:
        payload["briefcase_build_artifacts"] = str(briefcase_output)

    _write_metadata(metadata_path, payload)
    print(f"Zapisano metadane bundla w {metadata_path}")

    verify_args: list[str] = []
    if args.verify_fingerprint or args.verify_fingerprint_keys or args.verify_license_keys:
        if args.verify_fingerprint and not args.verify_fingerprint_keys:
            raise SystemExit(
                "Weryfikacja fingerprintu wymaga co najmniej jednego klucza (--verify-fingerprint-key).",
            )
        if args.verify_license_keys and not (args.verify_registry or Path("var/licenses/registry.jsonl").exists()):
            raise SystemExit(
                "Podaj --verify-registry lub utwórz var/licenses/registry.jsonl przed weryfikacją podpisów licencji.",
            )
        verify_args.append("--verify")
        if args.verify_fingerprint:
            verify_args.extend(["--fingerprint", str(Path(args.verify_fingerprint).expanduser().resolve())])
        registry_path = args.verify_registry or str(Path("var/licenses/registry.jsonl").resolve())
        if args.verify_license_keys:
            verify_args.extend(["--output", registry_path])
            for entry in args.verify_license_keys:
                verify_args.extend(["--license-key", entry])
        if args.verify_fingerprint_keys:
            for entry in args.verify_fingerprint_keys:
                verify_args.extend(["--fingerprint-key", entry])
        exit_code = oem_provision_license.main(verify_args)
        if exit_code != 0:
            raise SystemExit(f"Weryfikacja fingerprintu/licencji nie powiodła się (kod {exit_code}).")
        print("Automatyczna weryfikacja fingerprintu/licencji zakończona powodzeniem.")
    return 0


if __name__ == "__main__":  # pragma: no cover - wejście CLI
    raise SystemExit(main())

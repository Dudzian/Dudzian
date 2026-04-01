"""Obsługa profilu instalatora dla CLI build_installer_from_profile."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from deploy.packaging._toml_compat import tomllib


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


def normalize_profile_path(path: str, *, base_dir: Path) -> Path:
    """Resolve profile paths while tolerating Windows separators on POSIX hosts."""

    normalized = path.replace("\\", os.sep)
    candidate = Path(normalized)
    if not candidate.is_absolute():
        candidate = (base_dir / candidate).resolve()
    return candidate


def read_profile(path: Path) -> Profile:
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
            entrypoint=normalize_profile_path(entry_raw, base_dir=path.parent),
            runtime_name=runtime_name,
            hidden_imports=hidden_imports,
            dist_dir=(
                normalize_profile_path(dist_dir, base_dir=path.parent)
                if isinstance(dist_dir, str)
                else None
            ),
            work_dir=(
                normalize_profile_path(work_dir, base_dir=path.parent)
                if isinstance(work_dir, str)
                else None
            ),
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
            project_path=normalize_profile_path(project_raw, base_dir=path.parent),
            app_name=app_name,
            output_dir=normalize_profile_path(output_dir, base_dir=path.parent)
            if isinstance(output_dir, str)
            else None,
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
        output_dir=normalize_profile_path(output_dir_raw, base_dir=path.parent),
        work_dir=normalize_profile_path(work_dir_raw, base_dir=path.parent),
        qt_dist=(
            normalize_profile_path(qt_dist_raw, base_dir=path.parent)
            if isinstance(qt_dist_raw, str)
            else None
        ),
        include=include,
        metadata_path=normalize_profile_path(metadata_raw, base_dir=path.parent),
        signing_key=signing_key,
        signing_key_id=signing_key_id,
    )

    return Profile(
        platform=platform.strip(), pyinstaller=pyinstaller, briefcase=briefcase, bundle=bundle
    )

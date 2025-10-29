"""Obsługa paczek offline z modelami i strategiami."""
from __future__ import annotations

import hashlib
import json
import shutil
import tarfile
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from bot_core.security.signing import build_hmac_signature, verify_hmac_signature


@dataclass(slots=True)
class ReleaseArtifact:
    """Opis pojedynczego artefaktu aktualizacji."""

    relative_path: Path
    absolute_path: Path
    size: int
    sha384: str


def _hash_file(path: Path) -> str:
    digest = hashlib.sha384()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _gather_artifacts(sources: Mapping[str, Path]) -> list[ReleaseArtifact]:
    artifacts: list[ReleaseArtifact] = []
    for prefix, directory in sources.items():
        directory = directory.expanduser()
        if not directory.exists():
            continue
        for file_path in sorted(directory.rglob("*")):
            if not file_path.is_file():
                continue
            relative = Path(prefix) / file_path.relative_to(directory)
            artifacts.append(
                ReleaseArtifact(
                    relative_path=relative,
                    absolute_path=file_path,
                    size=file_path.stat().st_size,
                    sha384=_hash_file(file_path),
                )
            )
    return artifacts


def _build_manifest(version: str, artifacts: Iterable[ReleaseArtifact]) -> dict[str, Any]:
    artifact_entries = [
        {
            "path": artifact.relative_path.as_posix(),
            "sha384": artifact.sha384,
            "size": artifact.size,
        }
        for artifact in artifacts
    ]
    generated = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    return {"version": version, "generated_at": generated, "artifacts": artifact_entries}


def _write_manifest(manifest_path: Path, manifest: Mapping[str, Any]) -> None:
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_signature(manifest: Mapping[str, Any], signature_path: Path, *, key: bytes, key_id: str | None) -> None:
    signature = build_hmac_signature(manifest, key=key, key_id=key_id)
    payload = json.dumps({"signature": signature}, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    signature_path.write_text(payload, encoding="utf-8")


def create_release_archive(
    *,
    version: str,
    output_path: Path,
    models_dir: Path | None,
    strategies_dir: Path | None,
    signing_key: bytes | None = None,
    signing_key_id: str | None = None,
) -> Path:
    """Tworzy archiwum offline z modelami i strategiami."""

    sources: dict[str, Path] = {}
    if models_dir is not None:
        sources["models"] = Path(models_dir)
    if strategies_dir is not None:
        sources["strategies"] = Path(strategies_dir)
    artifacts = _gather_artifacts(sources)
    manifest = _build_manifest(version, artifacts)

    staging_dir = Path(tempfile.mkdtemp(prefix="offline_release_"))
    try:
        for artifact in artifacts:
            destination = staging_dir / artifact.relative_path
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(artifact.absolute_path, destination)
        _write_manifest(staging_dir / "manifest.json", manifest)
        if signing_key is not None:
            _write_signature(manifest, staging_dir / "manifest.sig", key=signing_key, key_id=signing_key_id)

        output_path = output_path.expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(output_path, "w:gz") as archive:
            for item in sorted(staging_dir.iterdir()):
                archive.add(item, arcname=item.name)
    finally:
        shutil.rmtree(staging_dir, ignore_errors=True)

    return output_path


def _extract_archive(archive_path: Path) -> Path:
    temp_dir = Path(tempfile.mkdtemp(prefix="offline_release_extract_"))
    try:
        with tarfile.open(archive_path, "r:gz") as archive:
            archive.extractall(temp_dir)
    except tarfile.TarError as exc:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise RuntimeError(f"Nie można rozpakować archiwum {archive_path}: {exc}") from exc
    return temp_dir


def _load_manifest(extracted: Path) -> Mapping[str, Any]:
    manifest_path = extracted / "manifest.json"
    if not manifest_path.exists():
        raise RuntimeError("Archiwum nie zawiera manifest.json")
    try:
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError("Manifest zawiera niepoprawny JSON") from exc


def _load_signature(extracted: Path) -> Mapping[str, Any] | None:
    signature_path = extracted / "manifest.sig"
    if not signature_path.exists():
        return None
    try:
        return json.loads(signature_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError("Podpis manifestu zawiera niepoprawny JSON") from exc


def verify_release_archive(
    archive_path: Path,
    *,
    signing_key: bytes | None = None,
) -> Mapping[str, Any]:
    """Weryfikuje integralność paczki offline i zwraca manifest."""

    archive_path = archive_path.expanduser()
    extracted = _extract_archive(archive_path)
    try:
        manifest = _load_manifest(extracted)
        version_value = manifest.get("version")
        if not isinstance(version_value, str) or not version_value.strip():
            raise RuntimeError("Manifest zawiera niepoprawną wersję release'u")
        signature_doc = _load_signature(extracted)
        if signing_key is not None:
            if not signature_doc:
                raise RuntimeError("Brakuje pliku manifest.sig wymaganego przy weryfikacji podpisu")
            if not verify_hmac_signature(manifest, signature_doc.get("signature"), key=signing_key):
                raise RuntimeError("Podpis HMAC manifestu jest niepoprawny")
        for entry in manifest.get("artifacts", []):
            if not isinstance(entry, Mapping):
                raise RuntimeError("Manifest zawiera uszkodzone wpisy artefaktów")
            path_value = entry.get("path")
            sha384_value = entry.get("sha384")
            if not isinstance(path_value, str) or not isinstance(sha384_value, str):
                raise RuntimeError("Manifest zawiera niepoprawne wpisy artefaktów")
            artifact_path = extracted / path_value
            if not artifact_path.exists():
                raise RuntimeError(f"Archiwum nie zawiera wymaganego pliku: {path_value}")
            if _hash_file(artifact_path) != sha384_value:
                raise RuntimeError(f"Suma kontrolna nie zgadza się dla pliku: {path_value}")
        return manifest
    finally:
        shutil.rmtree(extracted, ignore_errors=True)


def install_release_archive(
    archive_path: Path,
    *,
    signing_key: bytes | None = None,
    models_target: Path | None = None,
    strategies_target: Path | None = None,
    backup_dir: Path | None = None,
) -> Mapping[str, Any]:
    """Instaluje paczkę offline, tworząc kopie zapasowe istniejących zasobów."""

    archive_path = archive_path.expanduser()
    extracted = _extract_archive(archive_path)
    try:
        manifest = _load_manifest(extracted)
        version_value = manifest.get("version")
        if not isinstance(version_value, str) or not version_value.strip():
            raise RuntimeError("Manifest zawiera niepoprawną wersję release'u")
        signature_doc = _load_signature(extracted)
        if signing_key is not None:
            if not signature_doc:
                raise RuntimeError("Brakuje pliku manifest.sig wymaganego przy instalacji")
            if not verify_hmac_signature(manifest, signature_doc.get("signature"), key=signing_key):
                raise RuntimeError("Podpis HMAC manifestu jest niepoprawny")
        for entry in manifest.get("artifacts", []):
            if not isinstance(entry, Mapping):
                raise RuntimeError("Manifest zawiera uszkodzone wpisy artefaktów")
            path_value = entry.get("path")
            sha384_value = entry.get("sha384")
            if not isinstance(path_value, str) or not isinstance(sha384_value, str):
                raise RuntimeError("Manifest zawiera niepoprawne wpisy artefaktów")
            artifact_path = extracted / path_value
            if not artifact_path.exists():
                raise RuntimeError(f"Brakuje artefaktu: {path_value}")
            if _hash_file(artifact_path) != sha384_value:
                raise RuntimeError(f"Suma kontrolna nie zgadza się dla pliku: {path_value}")

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        backups: list[tuple[str, Path]] = []
        if backup_dir is not None:
            backup_dir = backup_dir.expanduser()
            backup_dir.mkdir(parents=True, exist_ok=True)
        staging_models = extracted / "models"
        staging_strategies = extracted / "strategies"
        if models_target and staging_models.exists():
            target_models = models_target.expanduser()
            if target_models.exists() and backup_dir is not None:
                backup_path = backup_dir / f"models-{timestamp}"
                shutil.copytree(target_models, backup_path)
                backups.append(("models", backup_path))
            target_models.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(staging_models, target_models, dirs_exist_ok=True)
        if strategies_target and staging_strategies.exists():
            target_strategies = strategies_target.expanduser()
            if target_strategies.exists() and backup_dir is not None:
                backup_path = backup_dir / f"strategies-{timestamp}"
                shutil.copytree(target_strategies, backup_path)
                backups.append(("strategies", backup_path))
            target_strategies.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(staging_strategies, target_strategies, dirs_exist_ok=True)
        return {
            "version": manifest.get("version"),
            "generated_at": manifest.get("generated_at"),
            "installed_models": staging_models.exists(),
            "installed_strategies": staging_strategies.exists(),
            "backups": [(name, str(path)) for name, path in backups],
        }
    finally:
        shutil.rmtree(extracted, ignore_errors=True)


__all__ = [
    "create_release_archive",
    "verify_release_archive",
    "install_release_archive",
]

"""Weryfikacja i import podpisanych pakietów aktualizacji `.kbot`."""
from __future__ import annotations

import json
import shutil
import tarfile
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, MutableMapping, Sequence

from bot_core.security.signing import verify_hmac_signature


class OfflinePackageError(RuntimeError):
    """Wyjątek zgłaszany w przypadku problemów z pakietem aktualizacji."""


@dataclass(slots=True)
class OfflinePackageArtifact:
    """Pojedynczy artefakt znajdujący się w paczce aktualizacji."""

    path: str
    size: int
    sha256: str


@dataclass(slots=True)
class OfflinePackageManifest:
    """Znormalizowany manifest pakietu `.kbot`."""

    package_id: str
    version: str
    fingerprint: str | None
    created_at: str | None
    artifacts: list[OfflinePackageArtifact]
    metadata: Mapping[str, object]
    raw: Mapping[str, object]


@dataclass(slots=True)
class ImportedOfflinePackage:
    """Opis pakietu po prawidłowej weryfikacji i imporcie."""

    manifest: OfflinePackageManifest
    target_directory: Path
    artifacts: Mapping[str, Path]
    signature: Mapping[str, object] | None


def _ensure_exists(path: Path) -> Path:
    if not path.exists():
        raise OfflinePackageError(f"Brak wymaganego pliku: {path}")
    return path


def _extract_package(package_path: Path) -> Path:
    package_path = package_path.expanduser()
    if not package_path.exists():
        raise OfflinePackageError(f"Pakiet {package_path} nie istnieje")

    staging_dir = Path(tempfile.mkdtemp(prefix="kbot_package_"))
    try:
        with tarfile.open(package_path, mode="r:gz") as archive:
            archive.extractall(staging_dir)
    except (FileNotFoundError, tarfile.TarError) as exc:  # pragma: no cover - defensywne
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise OfflinePackageError(f"Nie można rozpakować pakietu {package_path}: {exc}") from exc
    return staging_dir


def _load_manifest(directory: Path) -> Mapping[str, object]:
    manifest_path = _ensure_exists(directory / "manifest.json")
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:  # pragma: no cover - defensywne
        raise OfflinePackageError(f"Manifest {manifest_path} zawiera niepoprawny JSON: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise OfflinePackageError("Manifest pakietu powinien być obiektem JSON")
    return payload


def _load_signature(directory: Path) -> Mapping[str, object] | None:
    signature_path = directory / "manifest.sig"
    if not signature_path.exists():
        return None
    try:
        payload = json.loads(signature_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:  # pragma: no cover - defensywne
        raise OfflinePackageError(f"Podpis manifestu jest uszkodzony: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise OfflinePackageError("Plik manifest.sig powinien zawierać obiekt JSON")
    return payload


def _normalise_artifacts(entries: Sequence[Mapping[str, object]]) -> list[OfflinePackageArtifact]:
    artifacts: list[OfflinePackageArtifact] = []
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise OfflinePackageError("Lista artefaktów zawiera niepoprawny wpis")
        path_value = entry.get("path")
        size_value = entry.get("size")
        hash_value = entry.get("sha256")
        if not isinstance(path_value, str) or path_value.strip() == "":
            raise OfflinePackageError("Artefakt musi zawierać relatywną ścieżkę 'path'")
        candidate = Path(path_value)
        if candidate.is_absolute() or ".." in candidate.parts:
            raise OfflinePackageError(f"Artefakt {path_value} posiada niedozwoloną ścieżkę")
        if not isinstance(size_value, int) or size_value < 0:
            raise OfflinePackageError(f"Artefakt {path_value} posiada niepoprawny rozmiar")
        if not isinstance(hash_value, str) or len(hash_value) < 32:
            raise OfflinePackageError(f"Artefakt {path_value} posiada niepoprawny hash SHA-256")
        artifacts.append(OfflinePackageArtifact(path=path_value, size=size_value, sha256=hash_value.lower()))
    return artifacts


def _normalise_manifest(payload: Mapping[str, object]) -> OfflinePackageManifest:
    package_id = payload.get("id")
    version = payload.get("version")
    if not isinstance(package_id, str) or package_id.strip() == "":
        raise OfflinePackageError("Manifest pakietu musi zawierać pole 'id'")
    if not isinstance(version, str) or version.strip() == "":
        raise OfflinePackageError("Manifest pakietu musi zawierać pole 'version'")

    fingerprint = payload.get("fingerprint")
    if fingerprint is not None and not isinstance(fingerprint, str):
        raise OfflinePackageError("Pole 'fingerprint' powinno być napisem lub null")

    created_at = payload.get("created_at")
    if created_at is not None and not isinstance(created_at, str):
        raise OfflinePackageError("Pole 'created_at' musi być napisem ISO-8601")

    metadata_raw = payload.get("metadata")
    metadata: Mapping[str, object]
    if metadata_raw is None:
        metadata = {}
    elif isinstance(metadata_raw, Mapping):
        metadata = metadata_raw
    else:
        raise OfflinePackageError("Pole 'metadata' musi być mapą")

    artifacts_raw = payload.get("artifacts")
    if not isinstance(artifacts_raw, Sequence):
        raise OfflinePackageError("Manifest musi zawierać listę 'artifacts'")

    artifacts = _normalise_artifacts(artifacts_raw)
    return OfflinePackageManifest(
        package_id=package_id.strip(),
        version=version.strip(),
        fingerprint=fingerprint.strip() if isinstance(fingerprint, str) and fingerprint.strip() else None,
        created_at=created_at,
        artifacts=artifacts,
        metadata=metadata,
        raw=dict(payload),
    )


def _hash_sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_artifacts(manifest: OfflinePackageManifest, staging_dir: Path) -> dict[str, Path]:
    resolved: dict[str, Path] = {}
    for artifact in manifest.artifacts:
        artifact_path = staging_dir / artifact.path
        if not artifact_path.exists():
            raise OfflinePackageError(f"Pakiet nie zawiera wymaganego pliku: {artifact.path}")
        actual_size = artifact_path.stat().st_size
        if actual_size != artifact.size:
            raise OfflinePackageError(
                f"Artefakt {artifact.path} posiada niepoprawny rozmiar ({actual_size} != {artifact.size})"
            )
        actual_hash = _hash_sha256(artifact_path)
        if actual_hash != artifact.sha256:
            raise OfflinePackageError(
                f"Artefakt {artifact.path} posiada niepoprawny hash (oczekiwano {artifact.sha256}, otrzymano {actual_hash})"
            )
        resolved[artifact.path] = artifact_path
    return resolved


def verify_kbot_package(
    package_path: Path,
    *,
    expected_fingerprint: str | None = None,
    hmac_key: bytes | None = None,
) -> tuple[OfflinePackageManifest, Mapping[str, object] | None, Path, dict[str, Path]]:
    """Weryfikuje podpis i integralność pakietu `.kbot`.

    Funkcja zwraca manifest, podpis, katalog roboczy oraz mapę zweryfikowanych
    artefaktów. Katalog roboczy powinien zostać usunięty przez wywołującego po
    zakończeniu pracy.
    """

    staging_dir = _extract_package(package_path)
    try:
        manifest_payload = _load_manifest(staging_dir)
        signature_payload = _load_signature(staging_dir)
        manifest = _normalise_manifest(manifest_payload)

        if expected_fingerprint and manifest.fingerprint and manifest.fingerprint != expected_fingerprint:
            raise OfflinePackageError(
                "Pakiet został przygotowany dla innego fingerprintu urządzenia"
            )

        if hmac_key is not None:
            if signature_payload is None:
                raise OfflinePackageError("Pakiet nie zawiera pliku manifest.sig wymaganego do weryfikacji")
            if not verify_hmac_signature(manifest.raw, signature_payload, key=hmac_key):
                raise OfflinePackageError("Podpis kryptograficzny pakietu jest niepoprawny")

        artifacts = _verify_artifacts(manifest, staging_dir)
        return manifest, signature_payload, staging_dir, artifacts
    except Exception:
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise


def _signature_string(
    manifest: OfflinePackageManifest,
    signature: Mapping[str, object] | None,
    artifacts: Mapping[str, Path],
) -> str:
    digest_source: str | None = None
    if manifest.artifacts:
        digest_source = manifest.artifacts[0].sha256
    if signature is None or not isinstance(signature, Mapping):
        return digest_source or ""
    value = signature.get("value")
    key_id = signature.get("key_id") or signature.get("keyId")
    suffix_parts = [str(value)] if value else []
    if key_id:
        suffix_parts.append(str(key_id))
    suffix = ":".join(filter(None, suffix_parts))
    if digest_source and suffix:
        return f"{digest_source}:{suffix}"
    if digest_source:
        return digest_source
    return suffix


def import_kbot_package(
    package_path: Path,
    destination_dir: Path,
    *,
    expected_fingerprint: str | None = None,
    hmac_key: bytes | None = None,
) -> ImportedOfflinePackage:
    """Weryfikuje pakiet `.kbot` i kopiuje jego zawartość do katalogu aktualizacji."""

    manifest, signature_payload, staging_dir, artifacts = verify_kbot_package(
        package_path,
        expected_fingerprint=expected_fingerprint,
        hmac_key=hmac_key,
    )

    destination_dir = Path(destination_dir).expanduser()
    if not destination_dir.exists():
        destination_dir.mkdir(parents=True, exist_ok=True)

    target = destination_dir / f"{manifest.package_id}-{manifest.version}"
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True, exist_ok=True)

    try:
        manifest_to_write: MutableMapping[str, object] = dict(manifest.raw)
        manifest_to_write["signature"] = _signature_string(manifest, signature_payload, artifacts)
        if manifest_to_write.get("created_at") is None:
            manifest_to_write["created_at"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

        manifest_path = target / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest_to_write, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        if signature_payload is not None:
            (target / "manifest.sig").write_text(
                json.dumps(signature_payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

        copied: dict[str, Path] = {}
        for artifact, source_path in artifacts.items():
            destination_path = target / artifact
            destination_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, destination_path)
            copied[artifact] = destination_path

        return ImportedOfflinePackage(
            manifest=manifest,
            target_directory=target,
            artifacts=copied,
            signature=signature_payload,
        )
    finally:
        shutil.rmtree(staging_dir, ignore_errors=True)


__all__ = [
    "OfflinePackageArtifact",
    "OfflinePackageError",
    "OfflinePackageManifest",
    "ImportedOfflinePackage",
    "import_kbot_package",
    "verify_kbot_package",
]

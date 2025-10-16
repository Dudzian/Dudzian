"""Budowanie i weryfikacja pakietów odpornościowych Stage6."""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
from dataclasses import dataclass
from fnmatch import fnmatch
from pathlib import Path
from typing import Mapping, MutableMapping, Sequence

from bot_core.security.signing import build_hmac_signature


_SCHEMA_MANIFEST = "stage6.resilience.bundle.manifest"
_SCHEMA_SIGNATURE = "stage6.resilience.bundle.signature"
_SCHEMA_VERSION = "1.0"
_DIGEST_BLOCK_SIZE = 1024 * 1024


def _now_timestamp() -> str:
    return _dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _filename_timestamp() -> str:
    return _dt.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")


def _sha256(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_DIGEST_BLOCK_SIZE), b""):
            size += len(chunk)
            digest.update(chunk)
    return digest.hexdigest(), size


def _matches_any(relative: str, patterns: Sequence[str]) -> bool:
    return any(fnmatch(relative, pattern) for pattern in patterns)


@dataclass(slots=True)
class BundleEntry:
    """Pojedynczy plik w pakiecie odpornościowym."""

    path: Path
    relative_path: str
    sha256: str
    size_bytes: int


@dataclass(slots=True)
class ResilienceBundleArtifacts:
    """Opis wygenerowanych artefaktów bundla odpornościowego."""

    bundle_path: Path
    manifest_path: Path
    manifest: Mapping[str, object]
    signature_path: Path | None
    signature: Mapping[str, object] | None


class ResilienceBundleBuilder:
    """Buduje archiwum ZIP oraz manifest odporności Stage6."""

    def __init__(
        self,
        source_root: Path,
        *,
        include: Sequence[str] | None = None,
        exclude: Sequence[str] | None = None,
    ) -> None:
        self._source_root = source_root.expanduser().resolve()
        if not self._source_root.is_dir():
            raise ValueError(f"Katalog źródłowy nie istnieje: {self._source_root}")
        self._include = tuple(include or ("**",))
        self._exclude = tuple(exclude or ())

    def _iter_entries(self) -> list[BundleEntry]:
        entries: list[BundleEntry] = []
        for candidate in sorted(self._source_root.rglob("*")):
            if candidate.is_dir():
                continue
            if candidate.is_symlink():
                raise ValueError(f"Zabroniony symlink w źródłach: {candidate}")
            if not candidate.is_file():
                continue
            relative = candidate.relative_to(self._source_root).as_posix()
            if self._exclude and _matches_any(relative, self._exclude):
                continue
            if self._include and not _matches_any(relative, self._include):
                continue
            digest, size = _sha256(candidate)
            entries.append(
                BundleEntry(
                    path=candidate,
                    relative_path=relative,
                    sha256=digest,
                    size_bytes=size,
                )
            )
        if not entries:
            raise ValueError("Brak plików do spakowania – sprawdź wzorce include/exclude")
        return entries

    def _build_manifest(
        self,
        *,
        bundle_name: str,
        entries: Sequence[BundleEntry],
        metadata: Mapping[str, object] | None,
    ) -> MutableMapping[str, object]:
        total_size = sum(entry.size_bytes for entry in entries)
        manifest: MutableMapping[str, object] = {
            "schema": _SCHEMA_MANIFEST,
            "schema_version": _SCHEMA_VERSION,
            "generated_at": _now_timestamp(),
            "bundle_name": bundle_name,
            "source_root": self._source_root.as_posix(),
            "file_count": len(entries),
            "total_size_bytes": total_size,
            "files": [
                {
                    "path": entry.relative_path,
                    "size_bytes": entry.size_bytes,
                    "sha256": entry.sha256,
                }
                for entry in entries
            ],
            "metadata": dict(metadata or {}),
        }
        return manifest

    def build(
        self,
        *,
        bundle_name: str,
        output_dir: Path,
        metadata: Mapping[str, object] | None = None,
        signing_key: bytes | None = None,
        signing_key_id: str | None = None,
    ) -> ResilienceBundleArtifacts:
        entries = self._iter_entries()
        output_dir = output_dir.expanduser()
        output_dir.mkdir(parents=True, exist_ok=True)
        suffix = _filename_timestamp()
        archive_name = f"{bundle_name}-{suffix}.zip"
        bundle_path = (output_dir / archive_name).resolve()

        import zipfile

        with zipfile.ZipFile(bundle_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for entry in entries:
                archive.write(entry.path, arcname=entry.relative_path)

        manifest = self._build_manifest(
            bundle_name=bundle_name,
            entries=entries,
            metadata=metadata,
        )
        manifest_path = bundle_path.with_suffix(".manifest.json")
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        signature_path: Path | None = None
        signature_doc: Mapping[str, object] | None = None
        if signing_key is not None:
            signature_doc = {
                "schema": _SCHEMA_SIGNATURE,
                "schema_version": _SCHEMA_VERSION,
                "signed_at": _now_timestamp(),
                "target": manifest_path.name,
                "signature": build_hmac_signature(
                    manifest,
                    key=signing_key,
                    algorithm="HMAC-SHA256",
                    key_id=signing_key_id,
                ),
            }
            signature_path = manifest_path.with_suffix(".sig")
            signature_path.write_text(
                json.dumps(signature_doc, ensure_ascii=False, indent=2, sort_keys=True)
                + "\n",
                encoding="utf-8",
            )

        return ResilienceBundleArtifacts(
            bundle_path=bundle_path,
            manifest_path=manifest_path,
            manifest=manifest,
            signature_path=signature_path,
            signature=signature_doc,
        )


class ResilienceBundleVerifier:
    """Waliduje zawartość paczki odpornościowej."""

    def __init__(self, bundle_path: Path, manifest: Mapping[str, object]) -> None:
        self._bundle_path = bundle_path.expanduser().resolve()
        self._manifest = manifest
        if not self._bundle_path.is_file():
            raise ValueError(f"Archiwum nie istnieje: {self._bundle_path}")

    def verify_files(self) -> list[str]:
        import zipfile

        errors: list[str] = []
        manifest_files = {
            str(entry["path"]): (int(entry["size_bytes"]), str(entry["sha256"]))
            for entry in self._manifest.get("files", [])
        }

        if not manifest_files:
            return ["Manifest nie zawiera listy plików"]

        with zipfile.ZipFile(self._bundle_path, "r") as archive:
            archive_files = {
                name
                for name in archive.namelist()
                if name and not name.endswith("/")
            }
            for name, (expected_size, expected_digest) in manifest_files.items():
                try:
                    info = archive.getinfo(name)
                except KeyError:
                    errors.append(f"Brak pliku w archiwum: {name}")
                    continue
                if info.is_dir():
                    errors.append(f"Pozycja manifestu wskazuje na katalog: {name}")
                    continue
                size = 0
                digest = hashlib.sha256()
                with archive.open(info, "r") as handle:
                    for chunk in iter(lambda: handle.read(_DIGEST_BLOCK_SIZE), b""):
                        size += len(chunk)
                        digest.update(chunk)
                if size != expected_size:
                    errors.append(
                        f"Rozmiar pliku {name} niezgodny (manifest {expected_size}B, archiwum {size}B)"
                    )
                actual_digest = digest.hexdigest()
                if actual_digest != expected_digest:
                    errors.append(
                        f"Digest SHA-256 pliku {name} niezgodny (manifest {expected_digest}, archiwum {actual_digest})"
                    )
            extra_files = archive_files - set(manifest_files)
            if extra_files:
                extras = ", ".join(sorted(extra_files))
                errors.append(f"Archiwum zawiera nieoczekiwane pliki: {extras}")
        return errors


def load_manifest(path: Path) -> Mapping[str, object]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:  # noqa: BLE001 - komunikat wywołującego
        raise ValueError(f"Nie znaleziono manifestu: {path}") from exc
    except json.JSONDecodeError as exc:  # noqa: BLE001 - komunikat wywołującego
        raise ValueError(f"Manifest nie jest poprawnym JSON: {exc}") from exc
    if not isinstance(data, Mapping):
        raise ValueError("Manifest powinien być obiektem JSON")
    return data


def load_signature(path: Path | None) -> Mapping[str, object] | None:
    if path is None:
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:  # noqa: BLE001 - komunikat wywołującego
        raise ValueError(f"Nie znaleziono pliku podpisu: {path}") from exc
    except json.JSONDecodeError as exc:  # noqa: BLE001 - komunikat wywołującego
        raise ValueError(f"Plik podpisu nie jest poprawnym JSON: {exc}") from exc
    if not isinstance(data, Mapping):
        raise ValueError("Dokument podpisu powinien być obiektem JSON")
    return data


def verify_signature(
    manifest: Mapping[str, object],
    signature_doc: Mapping[str, object],
    *,
    key: bytes,
) -> list[str]:
    signature = signature_doc.get("signature")
    if not isinstance(signature, Mapping):
        return ["Dokument podpisu nie zawiera sekcji 'signature'"]
    algorithm = signature.get("algorithm")
    if algorithm != "HMAC-SHA256":
        return [f"Nieobsługiwany algorytm podpisu: {algorithm!r}"]
    expected = build_hmac_signature(
        manifest,
        key=key,
        algorithm="HMAC-SHA256",
        key_id=signature.get("key_id"),
    )
    if dict(expected) != dict(signature):
        return ["Podpis HMAC nie zgadza się z manifestem"]
    return []


__all__ = [
    "BundleEntry",
    "ResilienceBundleArtifacts",
    "ResilienceBundleBuilder",
    "ResilienceBundleVerifier",
    "load_manifest",
    "load_signature",
    "verify_signature",
]


"""Kontrakt oraz implementacje repozytorium modeli AI."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, MutableMapping, Protocol, Sequence, runtime_checkable

from .models import ModelArtifact


@runtime_checkable
class ModelRepository(Protocol):
    """Interfejs repozytorium modeli wykorzystywany przez inference i trening."""

    def load(self, artifact: str | Path | Mapping[str, object]) -> ModelArtifact: ...

    def load_model(self, reference: str | Path | Mapping[str, object] | None = None) -> ModelArtifact: ...

    def save(
        self,
        artifact: ModelArtifact,
        name: str,
        *,
        version: str | None = None,
        aliases: Sequence[str] | None = None,
        activate: bool | None = None,
    ) -> Path: ...

    def save_model(
        self,
        artifact: ModelArtifact,
        *,
        version: str | None = None,
        filename: str | None = None,
        aliases: Sequence[str] | None = None,
        activate: bool | None = None,
    ) -> Path: ...

    def publish(
        self,
        artifact: ModelArtifact,
        *,
        version: str,
        filename: str | None = None,
        aliases: Sequence[str] | None = None,
        activate: bool = True,
    ) -> Path: ...

    def resolve(self, reference: str | None = None) -> Path: ...

    def get_manifest(self) -> Mapping[str, object]: ...

    def list_versions(self) -> Sequence[str]: ...

    def get_active_version(self) -> str | None: ...

    def set_active_version(self, version: str) -> None: ...

    def promote_version(self, version: str, *, aliases: Sequence[str] | None = None) -> None: ...

    def attach_quality_report(self, version: str, report: Mapping[str, object]) -> Mapping[str, object]: ...

    def get_quality_report(self, version: str) -> Mapping[str, object] | None: ...

    def remove_version(
        self,
        version: str,
        *,
        delete_file: bool = False,
        missing_ok: bool = False,
    ) -> None: ...


@dataclass(slots=True)
class FilesystemModelRepository:
    """Repozytorium modeli oparte o lokalny filesystem z manifestem JSON."""

    base_path: Path
    manifest_name: str = "manifest.json"
    _manifest_cache: dict[str, object] | None = field(init=False, repr=False, default=None)

    def __post_init__(self) -> None:
        self.base_path = Path(self.base_path).expanduser().resolve()
        self.base_path.mkdir(parents=True, exist_ok=True)
        self._manifest_cache = None

    # ------------------------------------------------------------------ manifest --
    @property
    def _manifest_path(self) -> Path:
        return self.base_path / self.manifest_name

    def _default_manifest(self) -> dict[str, object]:
        return {"versions": {}, "aliases": {}, "active": None}

    def _ensure_manifest_maps(
        self, manifest: MutableMapping[str, object]
    ) -> tuple[MutableMapping[str, object], MutableMapping[str, str]]:
        versions_obj = manifest.get("versions")
        if isinstance(versions_obj, MutableMapping):
            versions: MutableMapping[str, object] = versions_obj
        elif isinstance(versions_obj, Mapping):
            versions = {
                str(key): dict(value)
                for key, value in versions_obj.items()
                if isinstance(key, str) and isinstance(value, Mapping)
            }
            manifest["versions"] = versions
        else:
            versions = {}
            manifest["versions"] = versions

        aliases_obj = manifest.get("aliases")
        if isinstance(aliases_obj, MutableMapping):
            aliases: MutableMapping[str, str] = aliases_obj  # type: ignore[assignment]
        elif isinstance(aliases_obj, Mapping):
            aliases = {
                str(key): str(value)
                for key, value in aliases_obj.items()
                if isinstance(key, str) and isinstance(value, str)
            }
            manifest["aliases"] = aliases
        else:
            aliases = {}
            manifest["aliases"] = aliases

        return versions, aliases

    def _load_manifest(self) -> dict[str, object]:
        if self._manifest_cache is not None:
            return self._manifest_cache
        manifest = self._default_manifest()
        try:
            if self._manifest_path.exists():
                with self._manifest_path.open("r", encoding="utf-8") as handle:
                    loaded = json.load(handle)
                if isinstance(loaded, Mapping):
                    versions = loaded.get("versions", {})
                    aliases = loaded.get("aliases", {})
                    active = loaded.get("active")
                    manifest["versions"] = {
                        str(version): dict(entry)
                        for version, entry in dict(versions).items()
                        if isinstance(version, str) and isinstance(entry, Mapping)
                    }
                    manifest["aliases"] = {
                        str(alias): str(version)
                        for alias, version in dict(aliases).items()
                        if isinstance(alias, str) and isinstance(version, str)
                    }
                    manifest["active"] = str(active) if isinstance(active, str) else None
        except (OSError, json.JSONDecodeError):
            manifest = self._default_manifest()
        self._ensure_manifest_maps(manifest)
        self._synchronise_aliases(manifest)
        self._manifest_cache = manifest
        return manifest

    def _write_manifest(self, manifest: Mapping[str, object]) -> None:
        payload = {
            "versions": dict(manifest.get("versions", {})),
            "aliases": dict(manifest.get("aliases", {})),
            "active": manifest.get("active"),
        }
        tmp_path = self._manifest_path.with_suffix(self._manifest_path.suffix + ".tmp")
        with tmp_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        tmp_path.replace(self._manifest_path)
        self._manifest_cache = dict(payload)

    def get_manifest(self) -> dict[str, object]:
        manifest = self._load_manifest()
        return json.loads(json.dumps(manifest))  # deep copy defensywny

    def list_versions(self) -> tuple[str, ...]:
        manifest = self._load_manifest()
        versions = list(manifest.get("versions", {}).keys())
        versions.sort(key=self._version_sort_key)
        return tuple(versions)

    def get_active_version(self) -> str | None:
        manifest = self._load_manifest()
        active = manifest.get("active")
        return str(active) if isinstance(active, str) else None

    def set_active_version(self, version: str) -> None:
        version = str(version).strip()
        if not version:
            raise ValueError("version must be a non-empty string")
        manifest = self._load_manifest()
        versions = manifest.get("versions", {})
        if version not in versions:
            raise KeyError(f"Brak wersji '{version}' w manifestie modeli")
        manifest["active"] = version
        self._write_manifest(manifest)

    # ------------------------------------------------------------------ helpers --
    def _version_sort_key(self, version: str) -> tuple[object, ...]:
        parts: list[object] = []
        for chunk in str(version).replace("-", ".").split("."):
            chunk = chunk.strip()
            if not chunk:
                continue
            if chunk.isdigit():
                parts.append(int(chunk))
            else:
                parts.append(chunk)
        return tuple(parts) or (str(version),)

    def _json_safe(self, value: object) -> object:
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, datetime):
            return value.astimezone(timezone.utc).isoformat()
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, Mapping):
            return {str(key): self._json_safe(val) for key, val in value.items()}
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            return [self._json_safe(item) for item in value]
        return str(value)

    def _extract_version(self, artifact: ModelArtifact, provided: str | None) -> str | None:
        if provided:
            return provided.strip()
        meta_version = artifact.metadata.get("model_version")
        if isinstance(meta_version, str) and meta_version.strip():
            return meta_version.strip()
        return None

    def _update_manifest_entry(
        self,
        *,
        version: str,
        destination: Path,
        artifact: ModelArtifact,
        aliases: Sequence[str] | None,
        activate: bool | None,
    ) -> None:
        manifest = self._load_manifest()
        versions, aliases_map = self._ensure_manifest_maps(manifest)

        relative_path = destination.relative_to(self.base_path)
        now = datetime.now(timezone.utc).isoformat()
        entry_aliases = tuple({str(alias).strip() for alias in (aliases or ()) if str(alias).strip()})
        existing_entry = versions.get(version)
        quality_report = None
        if isinstance(existing_entry, Mapping):
            quality_report = existing_entry.get("quality_report")
        versions[version] = {
            "file": str(relative_path),
            "saved_at": now,
            "metrics": self._json_safe(artifact.metrics),
            "metadata": self._json_safe(artifact.metadata),
            "aliases": list(entry_aliases),
            "quality_report": self._json_safe(quality_report) if quality_report is not None else None,
        }
        for alias in entry_aliases:
            aliases_map[alias] = version

        if activate is True or (
            activate is None and (manifest.get("active") in (None, version))
        ):
            manifest["active"] = version

        self._synchronise_aliases(manifest)

        self._write_manifest(manifest)

    def _resolve_reference(self, reference: str) -> Path:
        reference = reference.strip()
        manifest = self._load_manifest()
        if reference in {"active", "@active"}:
            active = manifest.get("active")
            if not isinstance(active, str) or not active:
                raise KeyError("Brak aktywnej wersji modelu w repozytorium")
            reference = active

        versions = manifest.get("versions", {})
        aliases = manifest.get("aliases", {})
        mapped_version = aliases.get(reference) if isinstance(aliases, Mapping) else None
        version = reference if reference in versions else mapped_version
        if isinstance(version, str) and version in versions:
            entry = versions[version]
            if isinstance(entry, Mapping):
                file_path = entry.get("file")
                if isinstance(file_path, str):
                    path = Path(file_path)
                    return path if path.is_absolute() else (self.base_path / path)
            raise KeyError(f"Manifest nie zawiera ścieżki dla wersji '{version}'")

        path = Path(reference)
        return path if path.is_absolute() else (self.base_path / path)

    # ------------------------------------------------------------------ API --
    def load(self, artifact: str | Path | Mapping[str, object]) -> ModelArtifact:
        if isinstance(artifact, Mapping):
            return ModelArtifact.from_dict(artifact)
        if isinstance(artifact, Path):
            path = artifact if artifact.is_absolute() else self.base_path / artifact
        else:
            path = self._resolve_reference(str(artifact))
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return ModelArtifact.from_dict(payload)

    def load_model(self, reference: str | Path | Mapping[str, object] | None = None) -> ModelArtifact:
        target = reference if reference is not None else self.resolve()
        return self.load(target)

    def save(
        self,
        artifact: ModelArtifact,
        name: str,
        *,
        version: str | None = None,
        aliases: Sequence[str] | None = None,
        activate: bool | None = None,
    ) -> Path:
        destination = self.base_path / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = destination.with_suffix(destination.suffix + ".tmp")
        with tmp_path.open("w", encoding="utf-8") as handle:
            json.dump(artifact.to_dict(), handle, ensure_ascii=False, indent=2)
        tmp_path.replace(destination)

        resolved_version = self._extract_version(artifact, version)
        if resolved_version:
            self._update_manifest_entry(
                version=resolved_version,
                destination=destination,
                artifact=artifact,
                aliases=aliases,
                activate=activate,
            )
        return destination

    def save_model(
        self,
        artifact: ModelArtifact,
        *,
        version: str | None = None,
        filename: str | None = None,
        aliases: Sequence[str] | None = None,
        activate: bool | None = None,
    ) -> Path:
        resolved_version = self._extract_version(artifact, version)
        name = filename or (
            f"model-{resolved_version}.json" if resolved_version else "model.json"
        )
        if resolved_version:
            return self.publish(
                artifact,
                version=resolved_version,
                filename=name,
                aliases=aliases,
                activate=activate if activate is not None else False,
            )
        return self.save(artifact, name, version=version, aliases=aliases, activate=activate)

    def publish(
        self,
        artifact: ModelArtifact,
        *,
        version: str,
        filename: str | None = None,
        aliases: Sequence[str] | None = None,
        activate: bool = True,
    ) -> Path:
        if not version or not version.strip():
            raise ValueError("version must be a non-empty string")
        filename = filename or f"model-{version}.json"
        base_metadata = dict(artifact.metadata)
        base_metadata.setdefault("model_version", version)
        published = ModelArtifact(
            feature_names=artifact.feature_names,
            model_state=artifact.model_state,
            trained_at=artifact.trained_at,
            metrics=artifact.metrics,
            metadata=base_metadata,
            target_scale=artifact.target_scale,
            training_rows=artifact.training_rows,
            validation_rows=artifact.validation_rows,
            test_rows=artifact.test_rows,
            feature_scalers=artifact.feature_scalers,
            decision_journal_entry_id=artifact.decision_journal_entry_id,
            backend=artifact.backend,
        )
        publish_aliases = tuple(aliases) if aliases is not None else ("latest",)
        return self.save(
            published,
            filename,
            version=version,
            aliases=publish_aliases,
            activate=activate,
        )

    def resolve(self, reference: str | None = None) -> Path:
        if reference is None:
            active = self.get_active_version()
            if active is None:
                raise KeyError("Brak aktywnej wersji modelu")
            reference = active
        else:
            reference = str(reference)
        path = self._resolve_reference(reference)
        return path

    def get_version_entry(self, version: str) -> Mapping[str, object] | None:
        version = str(version).strip()
        if not version:
            raise ValueError("version must be a non-empty string")
        manifest = self._load_manifest()
        versions, _ = self._ensure_manifest_maps(manifest)
        payload = versions.get(version)
        if isinstance(payload, Mapping):
            return json.loads(json.dumps(payload))
        return None

    def get_alias_target(self, alias: str) -> str | None:
        alias = str(alias).strip()
        if not alias:
            raise ValueError("alias must be a non-empty string")
        manifest = self._load_manifest()
        _, aliases_map = self._ensure_manifest_maps(manifest)
        target = aliases_map.get(alias)
        return str(target) if isinstance(target, str) and target else None

    def assign_alias(self, alias: str, version: str) -> None:
        alias = str(alias).strip()
        version = str(version).strip()
        if not alias:
            raise ValueError("alias must be a non-empty string")
        if not version:
            raise ValueError("version must be a non-empty string")
        manifest = self._load_manifest()
        versions, aliases_map = self._ensure_manifest_maps(manifest)
        if version not in versions:
            raise KeyError(f"Brak wersji '{version}' w manifestie modeli")
        current = aliases_map.get(alias)
        if current == version:
            return
        aliases_map[alias] = version
        self._synchronise_aliases(manifest)
        self._write_manifest(manifest)

    def attach_quality_report(self, version: str, report: Mapping[str, object]) -> Mapping[str, object]:
        version = str(version).strip()
        if not version:
            raise ValueError("version must be a non-empty string")
        manifest = self._load_manifest()
        versions, _ = self._ensure_manifest_maps(manifest)
        entry = versions.get(version)
        if entry is None:
            raise KeyError(f"Brak wersji '{version}' w manifestie modeli")
        if isinstance(entry, MutableMapping):
            updated_entry: MutableMapping[str, object] = entry
        elif isinstance(entry, Mapping):
            updated_entry = dict(entry)
            versions[version] = updated_entry
        else:
            updated_entry = {}
            versions[version] = updated_entry

        updated_entry["quality_report"] = self._json_safe(report)
        self._write_manifest(manifest)
        stored = updated_entry.get("quality_report")
        return json.loads(json.dumps(stored)) if stored is not None else {}

    def get_quality_report(self, version: str) -> Mapping[str, object] | None:
        version = str(version).strip()
        if not version:
            raise ValueError("version must be a non-empty string")
        manifest = self._load_manifest()
        versions, _ = self._ensure_manifest_maps(manifest)
        entry = versions.get(version)
        if not isinstance(entry, Mapping):
            return None
        report = entry.get("quality_report")
        if isinstance(report, Mapping):
            return json.loads(json.dumps(report))
        return None

    def promote_version(
        self,
        version: str,
        *,
        aliases: Sequence[str] | None = None,
    ) -> None:
        """Promuje wybraną wersję do aktywnej oraz aktualizuje aliasy."""

        version = str(version).strip()
        if not version:
            raise ValueError("version must be a non-empty string")

        manifest = self._load_manifest()
        versions, aliases_map = self._ensure_manifest_maps(manifest)

        if version not in versions:
            raise KeyError(f"Brak wersji '{version}' w manifestie modeli")

        manifest["active"] = version

        if aliases is not None:
            normalized_aliases = {
                str(alias).strip()
                for alias in aliases
                if str(alias).strip()
            }
            if normalized_aliases:
                for alias in normalized_aliases:
                    aliases_map[alias] = version
            else:
                for alias, mapped in list(aliases_map.items()):
                    if mapped == version:
                        aliases_map.pop(alias, None)

        self._synchronise_aliases(manifest)
        self._write_manifest(manifest)

    def remove_alias(self, alias: str, *, missing_ok: bool = False) -> None:
        alias = str(alias).strip()
        if not alias:
            raise ValueError("alias must be a non-empty string")
        manifest = self._load_manifest()
        _, aliases_map = self._ensure_manifest_maps(manifest)
        if alias not in aliases_map:
            if missing_ok:
                return
            raise KeyError(f"Alias '{alias}' nie istnieje w repozytorium modeli")
        aliases_map.pop(alias, None)
        self._synchronise_aliases(manifest)
        self._write_manifest(manifest)

    def remove_version(
        self,
        version: str,
        *,
        delete_file: bool = False,
        missing_ok: bool = False,
    ) -> None:
        version = str(version).strip()
        if not version:
            raise ValueError("version must be a non-empty string")
        manifest = self._load_manifest()
        versions, aliases_map = self._ensure_manifest_maps(manifest)
        payload = versions.get(version)
        if payload is None:
            if missing_ok:
                return
            raise KeyError(f"Brak wersji '{version}' w manifestie modeli")

        versions.pop(version, None)
        for alias, mapped in list(aliases_map.items()):
            if mapped == version:
                aliases_map.pop(alias, None)
        if manifest.get("active") == version:
            manifest["active"] = None

        self._synchronise_aliases(manifest)
        self._write_manifest(manifest)

        if delete_file and isinstance(payload, Mapping):
            file_path = payload.get("file")
            if isinstance(file_path, str) and file_path:
                path = Path(file_path)
                resolved = path if path.is_absolute() else (self.base_path / path)
                try:
                    resolved.unlink()
                except FileNotFoundError:
                    pass

    def _synchronise_aliases(self, manifest: MutableMapping[str, object]) -> None:
        versions, aliases_map = self._ensure_manifest_maps(manifest)
        for ver, payload in list(versions.items()):
            if not isinstance(payload, Mapping):
                continue
            current_aliases = [
                alias
                for alias, mapped in aliases_map.items()
                if mapped == ver
            ]
            payload = dict(payload)
            payload["aliases"] = current_aliases
            versions[ver] = payload


@dataclass(slots=True)
class CloudModelRepository:
    """Szkic repozytorium opartego o zewnętrzny storage (np. S3, GCS)."""

    bucket: str
    prefix: str

    def load(self, artifact: str | Path | Mapping[str, object]) -> ModelArtifact:  # pragma: no cover - szkic
        raise NotImplementedError("CloudModelRepository.load nie jest jeszcze zaimplementowane")

    def load_model(self, reference: str | Path | Mapping[str, object] | None = None) -> ModelArtifact:  # pragma: no cover - szkic
        raise NotImplementedError("CloudModelRepository.load_model nie jest jeszcze zaimplementowane")

    def save(
        self,
        artifact: ModelArtifact,
        name: str,
        *,
        version: str | None = None,
        aliases: Sequence[str] | None = None,
        activate: bool | None = None,
    ) -> Path:  # pragma: no cover - szkic
        raise NotImplementedError("CloudModelRepository.save nie jest jeszcze zaimplementowane")

    def save_model(
        self,
        artifact: ModelArtifact,
        *,
        version: str | None = None,
        filename: str | None = None,
        aliases: Sequence[str] | None = None,
        activate: bool | None = None,
    ) -> Path:  # pragma: no cover - szkic
        raise NotImplementedError("CloudModelRepository.save_model nie jest jeszcze zaimplementowane")

    def publish(
        self,
        artifact: ModelArtifact,
        *,
        version: str,
        filename: str | None = None,
        aliases: Sequence[str] | None = None,
        activate: bool = True,
    ) -> Path:  # pragma: no cover - szkic
        raise NotImplementedError("CloudModelRepository.publish nie jest jeszcze zaimplementowane")

    def resolve(self, reference: str | None = None) -> Path:  # pragma: no cover - szkic
        raise NotImplementedError("CloudModelRepository.resolve nie jest jeszcze zaimplementowane")

    def get_manifest(self) -> Mapping[str, object]:  # pragma: no cover - szkic
        raise NotImplementedError("CloudModelRepository.get_manifest nie jest jeszcze zaimplementowane")

    def list_versions(self) -> Sequence[str]:  # pragma: no cover - szkic
        raise NotImplementedError("CloudModelRepository.list_versions nie jest jeszcze zaimplementowane")

    def get_active_version(self) -> str | None:  # pragma: no cover - szkic
        raise NotImplementedError("CloudModelRepository.get_active_version nie jest jeszcze zaimplementowane")

    def set_active_version(self, version: str) -> None:  # pragma: no cover - szkic
        raise NotImplementedError("CloudModelRepository.set_active_version nie jest jeszcze zaimplementowane")

    def promote_version(self, version: str, *, aliases: Sequence[str] | None = None) -> None:  # pragma: no cover - szkic
        raise NotImplementedError("CloudModelRepository.promote_version nie jest jeszcze zaimplementowane")

    def attach_quality_report(self, version: str, report: Mapping[str, object]) -> Mapping[str, object]:  # pragma: no cover - szkic
        raise NotImplementedError("CloudModelRepository.attach_quality_report nie jest jeszcze zaimplementowane")

    def get_quality_report(self, version: str) -> Mapping[str, object] | None:  # pragma: no cover - szkic
        raise NotImplementedError("CloudModelRepository.get_quality_report nie jest jeszcze zaimplementowane")

    def remove_version(
        self,
        version: str,
        *,
        delete_file: bool = False,
        missing_ok: bool = False,
    ) -> None:  # pragma: no cover - szkic
        raise NotImplementedError("CloudModelRepository.remove_version nie jest jeszcze zaimplementowane")


__all__ = [
    "ModelRepository",
    "FilesystemModelRepository",
    "CloudModelRepository",
]
